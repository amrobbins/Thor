#include "DeepLearning/Implementation/Data/Residency/DeviceDatasetStorageSelection.h"

#include "DeepLearning/Implementation/Training/DeviceStartupCoordinator.h"
#include "DeepLearning/Implementation/Data/FileDatasetRuntimeAccess.h"
#include "DeepLearning/Implementation/Data/Residency/NamedDatasetRuntimeAccess.h"

#include "DeepLearning/Api/Data/NamedDataset.h"
#include "DeepLearning/Api/Data/FileDataset.h"
#include "DeepLearning/Api/Data/TrainingData.h"
#include "DeepLearning/Implementation/Data/Sessions/DeviceResidentNamedBatchSession.h"
#include "DeepLearning/Implementation/Data/Sessions/DeviceResidentWindowedNamedBatchSession.h"
#include "DeepLearning/Implementation/Data/Residency/DeviceDatasetResidency.h"
#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Data/Residency/DeviceResidentNamedDataset.h"
#include "DeepLearning/Implementation/Data/Materialization/MaterializedNamedDatasetSnapshot.h"
#include "DeepLearning/Implementation/Data/Materialization/NamedDatasetMaterializer.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"

#include <chrono>
#include <exception>
#include <limits>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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

uint64_t directBytesPerExample(const DatasetLayout &layout) {
    uint64_t bytes = 0;
    for (const DatasetLayout::TensorSpec &spec : layout.tensors()) {
        bytes = checkedAdd(
            bytes,
            spec.numBytes,
            "Device dataset direct bytes per example");
    }
    return bytes;
}

uint64_t windowedBytesPerExample(const DatasetLayout &layout) {
    uint64_t bytes = 0;
    for (const DatasetLayout::WindowedTensorSpec &spec :
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

uint64_t allBytesPerExample(const DatasetLayout &layout) {
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

uint64_t compactRequiredBytes(
    uint64_t residentBytes,
    const DeviceDatasetSessionDescription &session,
    uint64_t batchQueueDepth);

uint64_t estimateDeviceResidentWindowedDatasetStorageBytes(
    const DatasetMaterializationDescription &dataset) {
    std::set<std::string> names;
    for (const DatasetLayout::WindowedTensorSpec &spec : dataset.layout.windowedTensors()) {
        names.insert(spec.name);
        if (spec.maskName.has_value()) {
            names.insert(spec.maskName.value());
        }
    }
    return DeviceResidentNamedDataset::estimateCompactFileDatasetBytes(
        dataset,
        names);
}

uint64_t estimateDeviceResidentWindowedDatasetRequiredBytes(
    const DatasetMaterializationDescription &dataset,
    const DeviceDatasetSessionDescription &session,
    uint64_t batchQueueDepth) {
    return compactRequiredBytes(
        estimateDeviceResidentWindowedDatasetStorageBytes(dataset),
        session,
        batchQueueDepth);
}

bool usesCompactWindowedResidency(
    const DatasetMaterializationDescription &dataset) {
    return dataset.source == DatasetMaterializationSource::FILE_DATASET &&
           dataset.layout.hasWindowedTensors();
}

std::set<std::string> windowedTensorNames(
    const DatasetLayout &layout) {
    std::set<std::string> names;
    for (const DatasetLayout::WindowedTensorSpec &spec :
         layout.windowedTensors()) {
        names.insert(spec.name);
        if (spec.maskName.has_value()) {
            names.insert(spec.maskName.value());
        }
    }
    return names;
}

std::set<std::string> directTensorNames(const DatasetLayout &layout) {
    std::set<std::string> names;
    for (const DatasetLayout::TensorSpec &spec : layout.tensors()) {
        names.insert(spec.name);
    }
    return names;
}

std::set<std::string> allCompactFieldNames(const DatasetLayout &layout) {
    std::set<std::string> names = windowedTensorNames(layout);
    const std::set<std::string> directNames = directTensorNames(layout);
    names.insert(directNames.begin(), directNames.end());
    return names;
}

uint64_t compactSelectionRingBytes(
    const DeviceDatasetSessionDescription &session,
    uint64_t batchQueueDepth) {
    if (batchQueueDepth == 0) {
        throw std::runtime_error(
            "Device dataset batch_queue_depth must be >= 1.");
    }
    const uint64_t selectionBytesPerSlot = checkedMul(
        session.getBatching().getBatchSize(),
        static_cast<uint64_t>(sizeof(uint64_t)),
        "Device dataset selection row-index bytes");
    const uint64_t selectionSlotCount = checkedMul(
        nonEmptySplitCount(session.getSplits()),
        batchQueueDepth,
        "Device dataset selection slot count");
    return checkedMul(
        selectionSlotCount,
        selectionBytesPerSlot,
        "Device dataset selection-ring bytes");
}

uint64_t compactRequiredBytes(
    uint64_t residentBytes,
    const DeviceDatasetSessionDescription &session,
    uint64_t batchQueueDepth) {
    return checkedAdd(
        residentBytes,
        compactSelectionRingBytes(session, batchQueueDepth),
        "Device dataset compact reference required bytes");
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

MaterializedNamedDatasetSnapshot materializeCanonicalSnapshot(
    const std::shared_ptr<const NamedDataset> &dataset,
    const DatasetMaterializationDescription &description) {
    MaterializedNamedDatasetSnapshot snapshot =
        ThorImplementation::NamedDatasetRuntimeAccess::materializeSnapshot(*dataset);
    if (snapshot.datasetId != description.datasetId) {
        throw std::runtime_error(
            "NamedDataset materialization returned the wrong dataset identity.");
    }
    if (snapshot.numExamples != description.numExamples) {
        throw std::runtime_error(
            "NamedDataset materialization returned the wrong example count.");
    }
    if (snapshot.schema != description.schema) {
        throw std::runtime_error(
            "NamedDataset materialization returned the wrong schema.");
    }
    snapshot.layout.validateRequestedLayoutExact(description.layout);
    if (snapshot.fields.size() != description.schema.size()) {
        throw std::runtime_error(
            "NamedDataset materialization returned an unexpected field count.");
    }
    for (const DatasetField &field : description.schema.getFields()) {
        if (!snapshot.hasField(field.id)) {
            throw std::runtime_error(
                "NamedDataset materialization omitted field '" + field.name + "'.");
        }
        std::vector<uint64_t> expectedDimensions;
        expectedDimensions.reserve(field.dimensions.size() + 1);
        expectedDimensions.push_back(description.numExamples);
        expectedDimensions.insert(
            expectedDimensions.end(),
            field.dimensions.begin(),
            field.dimensions.end());
        const ThorImplementation::TensorDescriptor expectedDescriptor(
            field.dataType,
            expectedDimensions);
        if (snapshot.field(field.id).getDescriptor() != expectedDescriptor) {
            throw std::runtime_error(
                "NamedDataset materialization returned the wrong tensor descriptor for field '" +
                field.name + "'.");
        }
    }
    return snapshot;
}

bool isStrictDeviceDatasetStorage(DeviceDatasetStorage storage) {
    return storage == DeviceDatasetStorage::STRICT ||
           storage == DeviceDatasetStorage::STRICT_WINDOWED_ONLY;
}

std::runtime_error strictFailure(const DeviceDatasetStorageReport &report) {
    std::ostringstream out;
    out << "device_dataset_storage="
        << deviceDatasetStorageName(report.requested)
        << " could not materialize device-resident dataset";
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
    out << " required_unused_bytes="
        << ThorImplementation::DEVICE_STARTUP_SAFETY_RESERVE_BYTES;
    return std::runtime_error(out.str());
}

DeviceDatasetStorageSelection fallbackSelection(
    const std::shared_ptr<BatchSession> &sourceSession,
    DeviceDatasetStorageReport report,
    DeviceDatasetStorage requested) {
    report.requested = requested;
    if (isStrictDeviceDatasetStorage(requested)) {
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
    const std::string &datasetName,
    const std::shared_ptr<const NamedDataset> &namedDataset,
    const DatasetMaterializationDescription &dataset,
    const DeviceDatasetSessionDescription &session,
    DeviceDatasetStorage requested,
    ThorImplementation::TensorPlacement devicePlacement,
    uint64_t batchQueueDepth,
    uint64_t requiredBytes,
    std::optional<uint64_t> availableBytesOverride,
    DeviceDatasetStorageReport report) {
    DeviceDatasetResidencyCache &cache =
        ThorImplementation::NamedDatasetRuntimeAccess::residencyCache(*namedDataset);
    const auto started = std::chrono::steady_clock::now();
    const bool compactWindowedResidency = usesCompactWindowedResidency(dataset);
    const bool strictWindowedOnly =
        requested == DeviceDatasetStorage::STRICT_WINDOWED_ONLY;
    const std::set<std::string> windowNames =
        compactWindowedResidency ? windowedTensorNames(dataset.layout)
                                 : std::set<std::string>{};

    // File-backed windowed datasets must never enter the expanded canonical
    // full-dataset path. Other materializable datasets, including memory-backed
    // datasets and file datasets without windows, retain canonical shared
    // residency.
    if (compactWindowedResidency) {
        const std::set<std::string> allNames = allCompactFieldNames(dataset.layout);
        const uint64_t windowResidentBytes =
            DeviceResidentNamedDataset::estimateCompactFileDatasetBytes(
                dataset,
                windowNames);
        const uint64_t windowRequiredBytes = compactRequiredBytes(
            windowResidentBytes,
            session,
            batchQueueDepth);
        const uint64_t fullResidentBytes =
            DeviceResidentNamedDataset::estimateCompactFileDatasetBytes(
                dataset,
                allNames);
        const uint64_t fullRequiredBytes = compactRequiredBytes(
            fullResidentBytes,
            session,
            batchQueueDepth);

        struct CompactAttemptFailure {
            std::string reason;
            uint64_t availableBytes = 0;
        };

        auto attemptCompactResidency =
            [&](const std::set<std::string> &fieldNames,
                uint64_t residentBytes,
                uint64_t attemptRequiredBytes,
                const char *successReason,
                CompactAttemptFailure &failure)
                -> std::optional<DeviceDatasetStorageSelection> {
            const std::set<DatasetFieldId> fields =
                fieldIdsForNames(dataset.schema, fieldNames);
            try {
                DeviceDatasetResidencyRequest request(
                    dataset.datasetId,
                    dataset.numExamples,
                    devicePlacement,
                    fields,
                    residentBytes,
                    attemptRequiredBytes,
                    requested,
                    availableBytesOverride,
                    [dataset, devicePlacement, fieldNames]() {
                        return std::shared_ptr<const DeviceResidentNamedDataset>(
                            DeviceResidentNamedDataset::fromCompactFileDataset(
                                dataset,
                                devicePlacement,
                                fieldNames));
                    });
                DeviceDatasetResidencyAcquisition acquisition = cache.acquire(request);
                auto effectiveSession =
                    std::make_shared<DeviceResidentWindowedNamedBatchSession>(
                        dataset,
                        session,
                        acquisition.lease,
                        batchQueueDepth,
                        32,
                        datasetName);
                report.used = true;
                report.reason = successReason;
                report.examples = acquisition.lease->getNumExamples();
                report.requiredBytes = attemptRequiredBytes;
                applyAcquisitionTelemetry(
                    report,
                    acquisition,
                    residentBytes,
                    availableBytesOverride);
                report.materializationSeconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - started).count();
                return DeviceDatasetStorageSelection{
                    effectiveSession,
                    std::move(report)};
            } catch (const DeviceDatasetResidencyAdmissionError &e) {
                failure.reason = "insufficient_device_memory";
                failure.availableBytes = availableBytesOverride.has_value()
                                             ? availableBytesOverride.value()
                                             : e.getAvailableBytes();
            } catch (const std::exception &e) {
                failure.reason =
                    std::string("device_dataset_materialization_failed:") + e.what();
            }
            return std::nullopt;
        };

        // Prefer full compact residency when it fits. If the direct record
        // ranges add too much memory (notably for affine-only windows), retry
        // with all windowed fields resident and direct fields CPU-backed.
        CompactAttemptFailure fullFailure;
        if (!strictWindowedOnly && allNames != windowNames) {
            if (std::optional<DeviceDatasetStorageSelection> full =
                    attemptCompactResidency(
                        allNames,
                        fullResidentBytes,
                        fullRequiredBytes,
                        "compact_file_residency",
                        fullFailure);
                full.has_value()) {
                return std::move(full.value());
            }
        }

        CompactAttemptFailure windowFailure;
        const char *windowSuccessReason =
            strictWindowedOnly || allNames != windowNames
                ? "compact_windowed_residency"
                : "compact_file_residency";
        if (std::optional<DeviceDatasetStorageSelection> windowed =
                attemptCompactResidency(
                    windowNames,
                    windowResidentBytes,
                    windowRequiredBytes,
                    windowSuccessReason,
                    windowFailure);
            windowed.has_value()) {
            return std::move(windowed.value());
        }

        report.reason = windowFailure.reason.empty()
                            ? "device_dataset_materialization_failed"
                            : windowFailure.reason;
        if (report.reason == "insufficient_device_memory") {
            report.reason = "insufficient_device_memory_for_windowed_dataset";
        }
        report.requiredBytes = windowRequiredBytes;
        report.availableBytesAfterPlacement = windowFailure.availableBytes;
        report.materializationSeconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        return fallbackSelection(sourceSession, std::move(report), requested);
    }

    const uint64_t residentBytes =
        estimateDeviceResidentNamedDatasetStorageBytes(dataset);
    const std::set<DatasetFieldId> fullFields = allFieldIds(dataset.schema);
    try {
        DeviceDatasetResidencyRequest request(
            dataset.datasetId,
            dataset.numExamples,
            devicePlacement,
            fullFields,
            residentBytes,
            requiredBytes,
            requested,
            availableBytesOverride,
            [namedDataset, dataset, devicePlacement]() {
                MaterializedNamedDatasetSnapshot snapshot =
                    materializeCanonicalSnapshot(namedDataset, dataset);
                return std::shared_ptr<const DeviceResidentNamedDataset>(
                    DeviceResidentNamedDataset::fromSnapshot(
                        snapshot,
                        devicePlacement));
            });
        DeviceDatasetResidencyAcquisition acquisition = cache.acquire(request);
        auto effectiveSession = std::make_shared<DeviceResidentNamedBatchSession>(
            acquisition.lease,
            session,
            batchQueueDepth,
            datasetName);
        report.used = true;
        report.reason.clear();
        report.examples = acquisition.lease->getNumExamples();
        report.requiredBytes = requiredBytes;
        applyAcquisitionTelemetry(
            report,
            acquisition,
            residentBytes,
            availableBytesOverride);
        report.materializationSeconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        return DeviceDatasetStorageSelection{effectiveSession, std::move(report)};
    } catch (const DeviceDatasetResidencyAdmissionError &e) {
        report.reason = "insufficient_device_memory";
        report.requiredBytes = requiredBytes;
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
    if (usesCompactWindowedResidency(dataset)) {
        return estimateDeviceResidentWindowedDatasetStorageBytes(dataset);
    }
    return checkedMul(
        dataset.numExamples,
        allBytesPerExample(dataset.layout),
        "Device dataset canonical storage bytes");
}

uint64_t estimateDeviceResidentNamedDatasetRequiredBytes(
    const DatasetMaterializationDescription &dataset,
    const DeviceDatasetSessionDescription &session,
    uint64_t batchQueueDepth) {
    if (usesCompactWindowedResidency(dataset)) {
        return estimateDeviceResidentWindowedDatasetRequiredBytes(
            dataset,
            session,
            batchQueueDepth);
    }
    return estimateRequiredBytesForPerExampleBytes(
        dataset,
        session,
        batchQueueDepth,
        allBytesPerExample(dataset.layout));
}

DatasetMaterializationDescription describeDatasetMaterialization(
    const FileDataset& dataset) {
    return DatasetMaterializationDescription(
        dataset.getPath(),
        dataset.getId(),
        dataset.getSchema(),
        ThorImplementation::FileDatasetRuntimeAccess::layout(dataset),
        dataset.getNumExamples());
}

DatasetMaterializationDescription describeDatasetMaterialization(
    const TrainingData& trainingData) {
    std::unique_ptr<DatasetMaterializationDescription> description =
        ThorImplementation::NamedDatasetRuntimeAccess::describeMaterialization(
            *trainingData.getDataset());
    if (description == nullptr) {
        throw std::runtime_error(
            "TrainingData dataset backend does not support device materialization.");
    }
    return std::move(*description);
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
    if (requested == DeviceDatasetStorage::STRICT_WINDOWED_ONLY &&
        !usesCompactWindowedResidency(*datasetDescription)) {
        report.reason =
            "strict_windowed_only_requires_file_backed_windowed_dataset";
        return fallbackSelection(sourceSession, std::move(report), requested);
    }

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

    uint64_t requiredBytes = 0;
    try {
        requiredBytes = estimateDeviceResidentNamedDatasetRequiredBytes(
            *datasetDescription,
            sessionDescription,
            batchQueueDepth);
        report.requiredBytes = requiredBytes;
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
        trainingData.getDatasetName(),
        namedDataset,
        *datasetDescription,
        sessionDescription,
        requested,
        devicePlacement,
        batchQueueDepth,
        requiredBytes,
        availableBytesOverride,
        std::move(report));
}

}  // namespace Thor
