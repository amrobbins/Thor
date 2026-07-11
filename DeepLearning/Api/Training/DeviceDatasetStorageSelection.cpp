#include "DeepLearning/Api/Training/DeviceDatasetStorageSelection.h"

#include "DeepLearning/Api/Data/NamedDataset.h"
#include "DeepLearning/Api/Data/LocalNamedDataset.h"
#include "DeepLearning/Api/Data/TrainingData.h"
#include "DeepLearning/Api/Loaders/DeviceResidentNamedBatchSession.h"
#include "DeepLearning/Api/Loaders/DeviceResidentWindowedNamedBatchSession.h"
#include "DeepLearning/Api/Training/DeviceDatasetResidency.h"
#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Loaders/DeviceResidentNamedDataset.h"
#include "Utilities/Loaders/MaterializedNamedDatasetSnapshot.h"
#include "Utilities/Loaders/NamedDatasetMaterializer.h"

#include <chrono>
#include <exception>
#include <limits>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace Thor {
namespace {

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
        bytes = checkedAdd(
            bytes,
            spec.numBytes,
            "Device dataset direct bytes per example");
    }
    return bytes;
}

uint64_t windowedBytesPerExample(const LocalNamedExampleLayout &layout) {
    uint64_t bytes = 0;
    for (const LocalNamedExampleLayout::WindowedTensorSpec &spec :
         layout.windowedTensors()) {
        bytes = checkedAdd(
            bytes,
            spec.outputNumBytes(),
            "Device dataset windowed bytes per example");
        if (spec.maskName.has_value()) {
            bytes = checkedAdd(
                bytes,
                spec.windowLength(),
                "Device dataset windowed mask bytes per example");
        }
    }
    return bytes;
}

uint64_t allBytesPerExample(const LocalNamedExampleLayout &layout) {
    return checkedAdd(
        directBytesPerExample(layout),
        windowedBytesPerExample(layout),
        "Device dataset bytes per example");
}

uint64_t nonEmptySplitCount(const DatasetSplitManifest &splits) {
    uint64_t count = 0;
    if (!splits.getTrain().empty()) {
        count += 1;
    }
    if (!splits.getValidate().empty()) {
        count += 1;
    }
    if (!splits.getTest().empty()) {
        count += 1;
    }
    return count;
}

uint64_t estimateRequiredBytesForPerExampleBytes(
    const DatasetMaterializationDescription &dataset,
    const DeviceDatasetSessionDescription &session,
    uint64_t batchQueueDepth,
    uint64_t bytesPerExample) {
    if (batchQueueDepth == 0) {
        throw std::runtime_error(
            "Device dataset batch_queue_depth must be >= 1.");
    }

    const uint64_t residentTensorBytes = checkedMul(
        dataset.numExamples,
        bytesPerExample,
        "Device dataset resident tensor bytes");

    const uint64_t populatedSplits =
        nonEmptySplitCount(session.getSplits());
    const uint64_t batchSize = session.getBatching().getBatchSize();
    const uint64_t batchTensorBytesPerSplit = checkedMul(
        batchSize,
        bytesPerExample,
        "Device dataset batch tensor bytes");
    const uint64_t queuedBatchTensorBytes = checkedMul(
        checkedMul(
            populatedSplits,
            batchQueueDepth,
            "Device dataset queued batch count"),
        batchTensorBytesPerSplit,
        "Device dataset queued batch tensor bytes");
    const uint64_t rowIndexBytesPerSplit = checkedMul(
        batchSize,
        static_cast<uint64_t>(sizeof(uint64_t)),
        "Device dataset row-index tensor bytes");
    const uint64_t rowIndexBytes = checkedMul(
        populatedSplits,
        rowIndexBytesPerSplit,
        "Device dataset row-index bytes");

    return checkedAdd(
        checkedAdd(
            residentTensorBytes,
            queuedBatchTensorBytes,
            "Device dataset required bytes"),
        rowIndexBytes,
        "Device dataset required bytes");
}

uint64_t estimateDeviceResidentWindowedDatasetStorageBytes(
    const DatasetMaterializationDescription &dataset) {
    return checkedMul(
        dataset.numExamples,
        windowedBytesPerExample(dataset.layout),
        "Device dataset canonical windowed storage bytes");
}

uint64_t estimateDeviceResidentWindowedDatasetRequiredBytes(
    const DatasetMaterializationDescription &dataset,
    const DeviceDatasetSessionDescription &session,
    uint64_t batchQueueDepth) {
    return estimateRequiredBytesForPerExampleBytes(
        dataset,
        session,
        batchQueueDepth,
        windowedBytesPerExample(dataset.layout));
}

std::set<std::string> windowedTensorNames(
    const LocalNamedExampleLayout &layout) {
    std::set<std::string> names;
    for (const LocalNamedExampleLayout::WindowedTensorSpec &spec :
         layout.windowedTensors()) {
        names.insert(spec.name);
        if (spec.maskName.has_value()) {
            names.insert(spec.maskName.value());
        }
    }
    return names;
}

std::set<DatasetFieldId> allFieldIds(const DatasetSchema &schema) {
    std::set<DatasetFieldId> ids;
    for (const DatasetField &field : schema.getFields()) {
        ids.insert(field.id);
    }
    return ids;
}

std::set<DatasetFieldId> fieldIdsForNames(
    const DatasetSchema &schema,
    const std::set<std::string> &names) {
    std::set<DatasetFieldId> ids;
    for (const std::string &name : names) {
        ids.insert(schema.getField(name).id);
    }
    return ids;
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
        out << " available_bytes_after_model_placement="
            << report.availableBytesAfterPlacement;
    }
    return std::runtime_error(out.str());
}

DeviceDatasetStorageSelection fallbackSelection(
    const std::shared_ptr<BatchSession> &sourceSession,
    DeviceDatasetStorageReport report,
    DeviceDatasetStorage requested) {
    report.requested = requested;
    if (requested == DeviceDatasetStorage::STRICT) {
        throw strictFailure(report);
    }
    return DeviceDatasetStorageSelection{sourceSession, std::move(report)};
}

void applyAcquisitionTelemetry(
    DeviceDatasetStorageReport &report,
    const DeviceDatasetResidencyAcquisition &acquisition,
    uint64_t residentBytes,
    std::optional<uint64_t> availableBytesOverride) {
    report.residentBytes = residentBytes;
    report.residentCacheHit = acquisition.cacheHit;
    report.residentConstructionJoined = acquisition.joinedConstruction;
    report.residentConstructionStarted = acquisition.startedConstruction;
    if (availableBytesOverride.has_value()) {
        report.availableBytesAfterPlacement = availableBytesOverride.value();
    } else if (acquisition.availableBytesAtAdmission != 0) {
        report.availableBytesAfterPlacement = acquisition.availableBytesAtAdmission;
    }
}

DeviceDatasetStorageSelection selectSharedResidencySession(
    const std::shared_ptr<BatchSession> &sourceSession,
    const std::shared_ptr<const NamedDataset> &namedDataset,
    const DatasetMaterializationDescription &dataset,
    const DeviceDatasetSessionDescription &session,
    DeviceDatasetStorage requested,
    ThorImplementation::TensorPlacement devicePlacement,
    uint64_t batchQueueDepth,
    uint64_t fullRequiredBytes,
    uint64_t windowedRequiredBytes,
    std::optional<uint64_t> availableBytesOverride,
    DeviceDatasetStorageReport report) {
    DeviceDatasetResidencyCache &cache =
        namedDataset->getDeviceDatasetResidencyCache();
    const auto started = std::chrono::steady_clock::now();
    const uint64_t fullResidentBytes =
        estimateDeviceResidentNamedDatasetStorageBytes(dataset);
    const std::set<DatasetFieldId> fullFields = allFieldIds(dataset.schema);
    const std::set<std::string> windowNames = windowedTensorNames(dataset.layout);
    const bool hasWindowedTensors = !windowNames.empty();
    const uint64_t windowedResidentBytes = hasWindowedTensors
                                               ? estimateDeviceResidentWindowedDatasetStorageBytes(dataset)
                                               : 0;
    const std::set<DatasetFieldId> windowFields = hasWindowedTensors
                                                      ? fieldIdsForNames(dataset.schema, windowNames)
                                                      : std::set<DatasetFieldId>{};

    std::exception_ptr fullConstructionFailure;
    bool fullAdmissionFailed = false;
    try {
        DeviceDatasetResidencyRequest request(
            dataset.datasetId,
            dataset.numExamples,
            devicePlacement,
            fullFields,
            fullResidentBytes,
            fullRequiredBytes,
            requested,
            availableBytesOverride,
            [&dataset, devicePlacement]() {
                MaterializedNamedDatasetSnapshot snapshot =
                    materializeNamedDatasetSnapshot(dataset);
                return std::shared_ptr<const DeviceResidentNamedDataset>(
                    DeviceResidentNamedDataset::fromSnapshot(
                        snapshot,
                        devicePlacement));
            });
        DeviceDatasetResidencyAcquisition acquisition = cache.acquire(request);
        auto effectiveSession = std::make_shared<DeviceResidentNamedBatchSession>(
            acquisition.lease,
            session,
            batchQueueDepth);
        effectiveSession->setDatasetName(sourceSession->getDatasetName());
        report.used = true;
        report.reason.clear();
        report.examples = acquisition.lease->getNumExamples();
        report.requiredBytes = fullRequiredBytes;
        applyAcquisitionTelemetry(
            report,
            acquisition,
            fullResidentBytes,
            availableBytesOverride);
        report.materializationSeconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        return DeviceDatasetStorageSelection{effectiveSession, std::move(report)};
    } catch (const DeviceDatasetResidencyAdmissionError &e) {
        fullAdmissionFailed = true;
        report.reason = "insufficient_device_memory";
        report.requiredBytes = fullRequiredBytes;
        report.availableBytesAfterPlacement = availableBytesOverride.has_value()
                                                        ? availableBytesOverride.value()
                                                        : e.getAvailableBytes();
        if (requested == DeviceDatasetStorage::STRICT) {
            report.materializationSeconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started).count();
            return fallbackSelection(sourceSession, std::move(report), requested);
        }
    } catch (...) {
        fullConstructionFailure = std::current_exception();
        try {
            std::rethrow_exception(fullConstructionFailure);
        } catch (const std::exception &e) {
            report.reason =
                std::string("device_dataset_materialization_failed:") + e.what();
        }
        if (requested == DeviceDatasetStorage::STRICT || !hasWindowedTensors) {
            report.materializationSeconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started).count();
            return fallbackSelection(sourceSession, std::move(report), requested);
        }
    }

    if (!hasWindowedTensors) {
        report.reason = "insufficient_device_memory";
        report.materializationSeconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        return fallbackSelection(sourceSession, std::move(report), requested);
    }

    try {
        DeviceDatasetResidencyRequest request(
            dataset.datasetId,
            dataset.numExamples,
            devicePlacement,
            windowFields,
            windowedResidentBytes,
            windowedRequiredBytes,
            requested,
            availableBytesOverride,
            [&dataset, devicePlacement, &windowNames]() {
                MaterializedNamedDatasetSnapshot snapshot =
                    materializeNamedDatasetSnapshot(dataset);
                return std::shared_ptr<const DeviceResidentNamedDataset>(
                    DeviceResidentNamedDataset::fromSnapshot(
                        snapshot,
                        devicePlacement,
                        windowNames));
            });
        DeviceDatasetResidencyAcquisition acquisition = cache.acquire(request);
        auto effectiveSession =
            std::make_shared<DeviceResidentWindowedNamedBatchSession>(
                dataset,
                session,
                acquisition.lease,
                batchQueueDepth);
        effectiveSession->setDatasetName(sourceSession->getDatasetName());
        report.used = true;
        report.reason = fullConstructionFailure == nullptr
                            ? "windowed_features_only"
                            : "full_dataset_failed_windowed_features_only";
        report.examples = acquisition.lease->getNumExamples();
        report.requiredBytes = windowedRequiredBytes;
        applyAcquisitionTelemetry(
            report,
            acquisition,
            windowedResidentBytes,
            availableBytesOverride);
        report.materializationSeconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        return DeviceDatasetStorageSelection{effectiveSession, std::move(report)};
    } catch (const DeviceDatasetResidencyAdmissionError &e) {
        report.reason = fullAdmissionFailed
                            ? "insufficient_device_memory_for_full_or_windowed_dataset"
                            : "insufficient_device_memory_for_windowed_dataset";
        report.requiredBytes = windowedRequiredBytes;
        report.availableBytesAfterPlacement = availableBytesOverride.has_value()
                                                        ? availableBytesOverride.value()
                                                        : e.getAvailableBytes();
    } catch (const std::exception &e) {
        report.reason =
            std::string("device_dataset_materialization_failed:") + e.what();
    }
    report.materializationSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    return fallbackSelection(sourceSession, std::move(report), requested);
}


}  // namespace

uint64_t estimateDeviceResidentNamedDatasetStorageBytes(
    const DatasetMaterializationDescription &dataset) {
    return checkedMul(
        dataset.numExamples,
        allBytesPerExample(dataset.layout),
        "Device dataset canonical storage bytes");
}

uint64_t estimateDeviceResidentNamedDatasetRequiredBytes(
    const DatasetMaterializationDescription &dataset,
    const DeviceDatasetSessionDescription &session,
    uint64_t batchQueueDepth) {
    return estimateRequiredBytesForPerExampleBytes(
        dataset,
        session,
        batchQueueDepth,
        allBytesPerExample(dataset.layout));
}

DatasetMaterializationDescription describeDatasetMaterialization(
    const LocalNamedDataset& dataset) {
    return DatasetMaterializationDescription(
        dataset.getPath(),
        dataset.getId(),
        dataset.getSchema(),
        dataset.getLayout(),
        dataset.getNumExamples());
}

DatasetMaterializationDescription describeDatasetMaterialization(
    const TrainingData& trainingData) {
    std::shared_ptr<const LocalNamedDataset> localDataset =
        std::dynamic_pointer_cast<const LocalNamedDataset>(trainingData.getDataset());
    if (localDataset == nullptr) {
        throw std::runtime_error(
            "TrainingData dataset backend does not support device materialization.");
    }
    return describeDatasetMaterialization(*localDataset);
}

DeviceDatasetSessionDescription describeDeviceDatasetSession(
    const DatasetSplitManifest& splits,
    const BatchPolicy& batching,
    const std::set<DatasetFieldId>& requiredFieldIds) {
    return DeviceDatasetSessionDescription(splits, batching, requiredFieldIds);
}

DeviceDatasetSessionDescription describeDeviceDatasetSession(
    const TrainingData& trainingData,
    const std::set<DatasetFieldId>& requiredFieldIds) {
    return describeDeviceDatasetSession(
        trainingData.getSplits(),
        trainingData.getBatching(),
        requiredFieldIds);
}

DeviceDatasetStorageSelection selectDeviceDatasetStorageSession(
    const std::shared_ptr<BatchSession>& sourceSession,
    const TrainingData& trainingData,
    ThorImplementation::TensorPlacement devicePlacement,
    uint64_t batchQueueDepth,
    std::optional<uint64_t> availableBytesOverride) {
    THOR_THROW_IF_FALSE(sourceSession != nullptr);
    const DeviceDatasetStorage requested =
        trainingData.getAccessPolicy().deviceStorage;
    DeviceDatasetStorageReport report;
    report.requested = requested;

    if (requested == DeviceDatasetStorage::OFF) {
        return DeviceDatasetStorageSelection{sourceSession, report};
    }

    report.attempted = true;

    std::optional<DatasetMaterializationDescription> datasetDescription;
    try {
        datasetDescription.emplace(describeDatasetMaterialization(trainingData));
    } catch (const std::exception& e) {
        report.reason = std::string("dataset_not_materializable:") + e.what();
        return fallbackSelection(sourceSession, std::move(report), requested);
    }

    DeviceDatasetSessionDescription sessionDescription =
        describeDeviceDatasetSession(
            trainingData,
            sourceSession->getRequiredDatasetFieldIds());

    report.examples = datasetDescription->numExamples;
    if (sessionDescription.getSplits().getDatasetId() !=
            datasetDescription->datasetId ||
        sessionDescription.getSplits().getNumExamples() !=
            datasetDescription->numExamples) {
        report.reason = "session_dataset_identity_mismatch";
        return fallbackSelection(sourceSession, std::move(report), requested);
    }

    const NamedDatasetMaterializationSupport support =
        checkNamedDatasetSnapshotMaterializationSupport(*datasetDescription);
    if (!support.supported) {
        report.reason = support.reason.empty()
                            ? "dataset_not_materializable"
                            : support.reason;
        return fallbackSelection(sourceSession, std::move(report), requested);
    }

    uint64_t fullRequiredBytes = 0;
    uint64_t windowedRequiredBytes = 0;
    try {
        fullRequiredBytes = estimateDeviceResidentNamedDatasetRequiredBytes(
            *datasetDescription,
            sessionDescription,
            batchQueueDepth);
        windowedRequiredBytes =
            estimateDeviceResidentWindowedDatasetRequiredBytes(
                *datasetDescription,
                sessionDescription,
                batchQueueDepth);
        report.requiredBytes = fullRequiredBytes;
    } catch (const std::exception& e) {
        report.reason =
            std::string("device_dataset_size_estimate_failed:") + e.what();
        return fallbackSelection(sourceSession, std::move(report), requested);
    }

    const std::shared_ptr<const NamedDataset>& namedDataset =
        trainingData.getDataset();
    if (namedDataset->getId() != datasetDescription->datasetId ||
        namedDataset->getNumExamples() != datasetDescription->numExamples) {
        report.reason = "training_data_dataset_identity_mismatch";
        return fallbackSelection(sourceSession, std::move(report), requested);
    }

    return selectSharedResidencySession(
        sourceSession,
        namedDataset,
        *datasetDescription,
        sessionDescription,
        requested,
        devicePlacement,
        batchQueueDepth,
        fullRequiredBytes,
        windowedRequiredBytes,
        availableBytesOverride,
        std::move(report));
}

}  // namespace Thor
