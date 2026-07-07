#include "DeepLearning/Api/Training/DeviceDatasetStorageSelection.h"

#include "DeepLearning/Api/Loaders/DeviceResidentNamedBatchLoader.h"
#include "DeepLearning/Api/Loaders/DeviceResidentWindowedNamedBatchLoader.h"
#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Loaders/DeviceResidentNamedDataset.h"
#include "Utilities/Loaders/MaterializedNamedDatasetSnapshot.h"
#include "Utilities/Loaders/NamedDatasetMaterializer.h"

#include <cuda_runtime_api.h>

#include <chrono>
#include <exception>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace Thor {
namespace {

constexpr uint64_t DEVICE_DATASET_BEST_EFFORT_SLACK_BYTES = 2ull * 1024ull * 1024ull * 1024ull;

uint64_t checkedAdd(uint64_t left, uint64_t right, const char *context) {
    if (left > std::numeric_limits<uint64_t>::max() - right) {
        throw std::runtime_error(std::string(context) + " overflow while adding.");
    }
    return left + right;
}

uint64_t checkedMul(uint64_t left, uint64_t right, const char *context) {
    if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left) {
        throw std::runtime_error(std::string(context) + " overflow while multiplying.");
    }
    return left * right;
}

uint64_t directBytesPerExample(const LocalNamedExampleLayout &layout) {
    uint64_t bytes = 0;
    for (const LocalNamedExampleLayout::TensorSpec &spec : layout.tensors()) {
        bytes = checkedAdd(bytes, spec.numBytes, "Device dataset direct bytes per example");
    }
    return bytes;
}

uint64_t windowedBytesPerExample(const LocalNamedExampleLayout &layout) {
    uint64_t bytes = 0;
    for (const LocalNamedExampleLayout::WindowedTensorSpec &spec : layout.windowedTensors()) {
        bytes = checkedAdd(bytes, spec.outputNumBytes(), "Device dataset windowed bytes per example");
        if (spec.maskName.has_value()) {
            bytes = checkedAdd(bytes, spec.windowLength(), "Device dataset windowed mask bytes per example");
        }
    }
    return bytes;
}

uint64_t allBytesPerExample(const LocalNamedExampleLayout &layout) {
    return checkedAdd(directBytesPerExample(layout), windowedBytesPerExample(layout), "Device dataset bytes per example");
}

uint64_t totalSplitExamples(const DeviceDatasetMaterializationView &view) {
    uint64_t examples = 0;
    for (const DeviceDatasetMaterializationSplitView &split : view.splits) {
        examples = checkedAdd(examples, split.numExamples(), "Device dataset split example count");
    }
    return examples;
}

uint64_t nonEmptySplitCount(const DeviceDatasetMaterializationView &view) {
    uint64_t count = 0;
    for (const DeviceDatasetMaterializationSplitView &split : view.splits) {
        if (split.numExamples() != 0) {
            count += 1;
        }
    }
    return count;
}

uint64_t estimateDeviceResidentBytesForPerExampleBytes(const DeviceDatasetMaterializationView &view,
                                                       uint64_t batchQueueDepth,
                                                       uint64_t bytesPerExample) {
    if (batchQueueDepth == 0) {
        throw std::runtime_error("Device dataset batch_queue_depth must be >= 1.");
    }

    const uint64_t residentExamples = totalSplitExamples(view);
    const uint64_t residentTensorBytes = checkedMul(residentExamples, bytesPerExample, "Device dataset resident tensor bytes");

    const uint64_t populatedSplits = nonEmptySplitCount(view);
    const uint64_t batchTensorBytesPerSplit = checkedMul(view.batchSize, bytesPerExample, "Device dataset batch tensor bytes");
    const uint64_t queuedBatchTensorBytes = checkedMul(checkedMul(populatedSplits,
                                                                  batchQueueDepth,
                                                                  "Device dataset queued batch count"),
                                                       batchTensorBytesPerSplit,
                                                       "Device dataset queued batch tensor bytes");
    const uint64_t rowIndexBytesPerSplit = checkedMul(view.batchSize,
                                                      static_cast<uint64_t>(sizeof(uint64_t)),
                                                      "Device dataset row-index tensor bytes");
    const uint64_t rowIndexBytes = checkedMul(populatedSplits, rowIndexBytesPerSplit, "Device dataset row-index bytes");

    return checkedAdd(checkedAdd(residentTensorBytes,
                                queuedBatchTensorBytes,
                                "Device dataset required bytes"),
                      rowIndexBytes,
                      "Device dataset required bytes");
}

uint64_t estimateDeviceResidentWindowedDatasetRequiredBytes(const DeviceDatasetMaterializationView &view,
                                                           uint64_t batchQueueDepth) {
    return estimateDeviceResidentBytesForPerExampleBytes(view, batchQueueDepth, windowedBytesPerExample(view.layout));
}

bool fitsWithSlack(uint64_t requiredBytes, uint64_t availableBytes, DeviceDatasetStorage requested) {
    if (requested == DeviceDatasetStorage::STRICT) {
        return requiredBytes <= availableBytes;
    }
    if (availableBytes <= DEVICE_DATASET_BEST_EFFORT_SLACK_BYTES) {
        return false;
    }
    return requiredBytes <= availableBytes - DEVICE_DATASET_BEST_EFFORT_SLACK_BYTES;
}

std::set<std::string> windowedTensorNames(const LocalNamedExampleLayout &layout) {
    std::set<std::string> names;
    for (const LocalNamedExampleLayout::WindowedTensorSpec &spec : layout.windowedTensors()) {
        names.insert(spec.name);
        if (spec.maskName.has_value()) {
            names.insert(spec.maskName.value());
        }
    }
    return names;
}

std::runtime_error strictFailure(const DeviceDatasetStorageReport &report) {
    std::ostringstream out;
    out << "device_dataset_storage=strict could not materialize device-resident dataset";
    if (!report.reason.empty()) {
        out << ": " << report.reason;
    }
    if (report.requiredBytes != 0) {
        out << " required_bytes=" << report.requiredBytes;
    }
    if (report.availableBytesAfterPlacement != 0) {
        out << " available_bytes_after_model_placement=" << report.availableBytesAfterPlacement;
    }
    return std::runtime_error(out.str());
}

DeviceDatasetStorageSelection fallbackSelection(const std::shared_ptr<Loader> &sourceLoader,
                                                DeviceDatasetStorageReport report,
                                                DeviceDatasetStorage requested) {
    report.requested = requested;
    if (requested == DeviceDatasetStorage::STRICT) {
        throw strictFailure(report);
    }
    return DeviceDatasetStorageSelection{sourceLoader, std::move(report)};
}

std::optional<uint64_t> queryAvailableDeviceBytes(ThorImplementation::TensorPlacement devicePlacement,
                                                  std::string &failureReason) {
    if (devicePlacement.getMemDevice() != ThorImplementation::TensorPlacement::MemDevices::GPU) {
        failureReason = "invalid_device_placement";
        return std::nullopt;
    }

    ScopedGpu scopedGpu(devicePlacement.getDeviceNum());
    size_t freeBytes = 0;
    size_t totalBytes = 0;
    const cudaError_t status = cudaMemGetInfo(&freeBytes, &totalBytes);
    (void)totalBytes;
    if (status != cudaSuccess) {
        failureReason = std::string("device_memory_query_failed:") + cudaGetErrorString(status);
        return std::nullopt;
    }
    return static_cast<uint64_t>(freeBytes);
}

}  // namespace

uint64_t estimateDeviceResidentNamedDatasetRequiredBytes(const DeviceDatasetMaterializationView &view,
                                                         uint64_t batchQueueDepth) {
    return estimateDeviceResidentBytesForPerExampleBytes(view, batchQueueDepth, allBytesPerExample(view.layout));
}

DeviceDatasetStorageSelection selectDeviceDatasetStorageLoader(const std::shared_ptr<Loader> &sourceLoader,
                                                              DeviceDatasetStorage requested,
                                                              ThorImplementation::TensorPlacement devicePlacement,
                                                              uint64_t batchQueueDepth,
                                                              std::optional<uint64_t> availableBytesOverride) {
    THOR_THROW_IF_FALSE(sourceLoader != nullptr);
    DeviceDatasetStorageReport report;
    report.requested = requested;

    if (requested == DeviceDatasetStorage::OFF) {
        return DeviceDatasetStorageSelection{sourceLoader, report};
    }

    report.attempted = true;

    if (!sourceLoader->supportsDeviceDatasetMaterialization()) {
        report.reason = sourceLoader->getDeviceDatasetMaterializationUnsupportedReason();
        if (report.reason.empty()) {
            report.reason = "loader_not_materializable";
        }
        return fallbackSelection(sourceLoader, std::move(report), requested);
    }

    DeviceDatasetMaterializationView view;
    try {
        view = sourceLoader->describeDeviceDatasetMaterialization();
    } catch (const std::exception &e) {
        report.reason = std::string("loader_materialization_description_failed:") + e.what();
        return fallbackSelection(sourceLoader, std::move(report), requested);
    }

    report.examples = totalSplitExamples(view);

    const NamedDatasetMaterializationSupport support = checkNamedDatasetSnapshotMaterializationSupport(view);
    if (!support.supported) {
        report.reason = support.reason.empty() ? "loader_not_materializable" : support.reason;
        return fallbackSelection(sourceLoader, std::move(report), requested);
    }

    uint64_t fullRequiredBytes = 0;
    uint64_t windowedRequiredBytes = 0;
    try {
        fullRequiredBytes = estimateDeviceResidentNamedDatasetRequiredBytes(view, batchQueueDepth);
        windowedRequiredBytes = estimateDeviceResidentWindowedDatasetRequiredBytes(view, batchQueueDepth);
        report.requiredBytes = fullRequiredBytes;
    } catch (const std::exception &e) {
        report.reason = std::string("device_dataset_size_estimate_failed:") + e.what();
        return fallbackSelection(sourceLoader, std::move(report), requested);
    }

    if (availableBytesOverride.has_value()) {
        report.availableBytesAfterPlacement = availableBytesOverride.value();
    } else {
        std::string memoryQueryFailure;
        std::optional<uint64_t> availableBytes = queryAvailableDeviceBytes(devicePlacement, memoryQueryFailure);
        if (!availableBytes.has_value()) {
            report.reason = memoryQueryFailure;
            return fallbackSelection(sourceLoader, std::move(report), requested);
        }
        report.availableBytesAfterPlacement = availableBytes.value();
    }

    const bool fullFits = fitsWithSlack(fullRequiredBytes, report.availableBytesAfterPlacement, requested);
    if (!fullFits && requested == DeviceDatasetStorage::STRICT) {
        report.reason = "insufficient_device_memory";
        return fallbackSelection(sourceLoader, std::move(report), requested);
    }

    const bool hasWindowedTensors = view.layout.hasWindowedTensors() && windowedBytesPerExample(view.layout) != 0;
    const bool windowedOnlyFits = hasWindowedTensors && fitsWithSlack(windowedRequiredBytes, report.availableBytesAfterPlacement, requested);

    if (!fullFits && !windowedOnlyFits) {
        report.reason = hasWindowedTensors ? "insufficient_device_memory_for_full_or_windowed_dataset" : "insufficient_device_memory";
        return fallbackSelection(sourceLoader, std::move(report), requested);
    }

    const auto started = std::chrono::steady_clock::now();
    MaterializedNamedDatasetSnapshot snapshot;
    try {
        snapshot = materializeNamedDatasetSnapshot(view);
    } catch (const std::exception &e) {
        report.reason = std::string("device_dataset_materialization_failed:") + e.what();
        report.materializationSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        return fallbackSelection(sourceLoader, std::move(report), requested);
    }

    std::exception_ptr fullDeviceFailure;
    if (fullFits) {
        try {
            std::shared_ptr<DeviceResidentNamedDataset> resident = DeviceResidentNamedDataset::fromSnapshot(snapshot, devicePlacement);
            auto effectiveLoader = std::make_shared<DeviceResidentNamedBatchLoader>(resident, batchQueueDepth);
            effectiveLoader->setDatasetName(sourceLoader->getDatasetName());
            report.used = true;
            report.reason.clear();
            report.examples = resident->totalExamples();
            report.requiredBytes = fullRequiredBytes;
            report.materializationSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            return DeviceDatasetStorageSelection{effectiveLoader, std::move(report)};
        } catch (...) {
            fullDeviceFailure = std::current_exception();
            if (requested == DeviceDatasetStorage::STRICT || !windowedOnlyFits) {
                try {
                    std::rethrow_exception(fullDeviceFailure);
                } catch (const std::exception &e) {
                    report.reason = std::string("device_dataset_materialization_failed:") + e.what();
                }
                report.materializationSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
                return fallbackSelection(sourceLoader, std::move(report), requested);
            }
        }
    }

    try {
        std::shared_ptr<DeviceResidentNamedDataset> residentWindows =
            DeviceResidentNamedDataset::fromSnapshot(snapshot, devicePlacement, windowedTensorNames(view.layout));
        auto effectiveLoader = std::make_shared<DeviceResidentWindowedNamedBatchLoader>(view, residentWindows, batchQueueDepth);
        effectiveLoader->setDatasetName(sourceLoader->getDatasetName());
        report.used = true;
        report.reason = fullDeviceFailure == nullptr ? "windowed_features_only" : "full_dataset_failed_windowed_features_only";
        report.examples = residentWindows->totalExamples();
        report.requiredBytes = windowedRequiredBytes;
        report.materializationSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        return DeviceDatasetStorageSelection{effectiveLoader, std::move(report)};
    } catch (const std::exception &e) {
        report.reason = std::string("device_dataset_materialization_failed:") + e.what();
        report.materializationSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        return fallbackSelection(sourceLoader, std::move(report), requested);
    }
}

}  // namespace Thor
