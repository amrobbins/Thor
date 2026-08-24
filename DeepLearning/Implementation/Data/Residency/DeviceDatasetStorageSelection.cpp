#include "DeepLearning/Implementation/Data/Residency/DeviceDatasetStorageSelection.h"
#include "DeepLearning/Implementation/Data/Sessions/BatchSessionRuntimeAccess.h"

#include "DeepLearning/Implementation/Training/DeviceStartupCoordinator.h"
#include "DeepLearning/Implementation/Data/FileDatasetRuntimeAccess.h"
#include "DeepLearning/Implementation/Data/Residency/NamedDatasetRuntimeAccess.h"

#include "DeepLearning/Api/Data/NamedDataset.h"
#include "DeepLearning/Api/Data/FileDataset.h"
#include "DeepLearning/Api/Data/TrainingData.h"
#include "DeepLearning/Implementation/Data/Sessions/DeviceResidentNamedBatchSession.h"
#include "DeepLearning/Implementation/Data/Sessions/DeviceResidentFileNamedBatchSession.h"
#include "DeepLearning/Implementation/Data/Residency/DeviceDatasetResidency.h"
#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Data/Residency/DeviceResidentNamedDataset.h"
#include "DeepLearning/Implementation/Data/Materialization/MaterializedNamedDatasetSnapshot.h"
#include "DeepLearning/Implementation/Data/Materialization/NamedDatasetMaterializer.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Implementation/Tensor/RaggedTensorDescriptor.h"

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

uint64_t canonicalSnapshotResidentBytes(
    const MaterializedNamedDatasetSnapshot &snapshot) {
    uint64_t bytes = 0;
    for (const auto &[fieldId, tensor] : snapshot.fields) {
        (void)fieldId;
        bytes = checkedAdd(
            bytes,
            tensor.getArraySizeInBytes(),
            "Device dataset canonical snapshot tensor bytes");
    }
    for (const auto &[fieldId, ragged] : snapshot.raggedFields) {
        (void)fieldId;
        const uint64_t valuesBytes = ragged.values.isInitialized()
                                         ? ragged.values.getArraySizeInBytes()
                                         : 0;
        bytes = checkedAdd(
            bytes,
            valuesBytes,
            "Device dataset canonical ragged values bytes");
        bytes = checkedAdd(
            bytes,
            checkedMul(
                snapshot.numExamples,
                2 * static_cast<uint64_t>(sizeof(uint64_t)),
                "Device dataset canonical ragged references bytes"),
            "Device dataset canonical ragged resident bytes");
    }
    return bytes;
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

std::set<std::string> windowedTensorNames(const DatasetLayout &layout) {
    std::set<std::string> names;
    for (const DatasetLayout::WindowedTensorSpec &spec : layout.windowedTensors()) {
        names.insert(spec.name);
        if (spec.maskName.has_value()) names.insert(spec.maskName.value());
    }
    return names;
}

std::set<std::string> raggedTensorNames(const DatasetLayout &layout) {
    std::set<std::string> names;
    for (const DatasetLayout::RaggedTensorSpec &spec : layout.raggedTensors()) {
        names.insert(spec.name);
    }
    return names;
}

std::set<std::string> directTensorNames(const DatasetLayout &layout) {
    std::set<std::string> names;
    for (const DatasetLayout::TensorSpec &spec : layout.tensors()) names.insert(spec.name);
    return names;
}

std::set<std::string> mandatoryCompactFieldNames(const DatasetLayout &layout) {
    std::set<std::string> names = windowedTensorNames(layout);
    const std::set<std::string> raggedNames = raggedTensorNames(layout);
    names.insert(raggedNames.begin(), raggedNames.end());
    return names;
}

std::set<std::string> allCompactFieldNames(const DatasetLayout &layout) {
    std::set<std::string> names = mandatoryCompactFieldNames(layout);
    const std::set<std::string> directNames = directTensorNames(layout);
    names.insert(directNames.begin(), directNames.end());
    return names;
}

bool usesCompactFileResidency(const DatasetMaterializationDescription &dataset) {
    return dataset.source == DatasetMaterializationSource::FILE_DATASET &&
           (dataset.layout.hasWindowedTensors() || dataset.layout.hasRaggedTensors());
}

uint64_t estimateDeviceResidentCompactDatasetStorageBytes(
    const DatasetMaterializationDescription &dataset) {
    return DeviceResidentNamedDataset::estimateCompactFileDatasetBytes(
        dataset,
        mandatoryCompactFieldNames(dataset.layout));
}

uint64_t estimateDeviceResidentCompactDatasetRequiredBytes(
    const DatasetMaterializationDescription &dataset,
    const DeviceDatasetSessionDescription &session,
    uint64_t batchQueueDepth) {
    return compactRequiredBytes(
        estimateDeviceResidentCompactDatasetStorageBytes(dataset),
        session,
        batchQueueDepth);
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

uint64_t compactRaggedBatchRingBytes(
    const DeviceDatasetSessionDescription &session,
    uint64_t batchQueueDepth) {
    const uint64_t slotCount = checkedMul(
        nonEmptySplitCount(session.getSplits()),
        batchQueueDepth,
        "Device dataset ragged slot count");
    uint64_t bytesPerSlot = 0;
    for (const auto &[fieldId, requirement] : session.getFieldRequirements()) {
        (void)fieldId;
        if (!requirement.raggedTensorDescriptor.has_value()) continue;
        const ThorImplementation::RaggedTensorDescriptor &descriptor =
            requirement.raggedTensorDescriptor.value();
        bytesPerSlot = checkedAdd(
            bytesPerSlot,
            descriptor.getValuesDescriptor().getArraySizeInBytes(),
            "Device dataset ragged batch values bytes");
        bytesPerSlot = checkedAdd(
            bytesPerSlot,
            descriptor.getOffsetsDescriptor().getArraySizeInBytes(),
            "Device dataset ragged batch offsets bytes");
    }
    return checkedMul(slotCount, bytesPerSlot, "Device dataset ragged batch ring bytes");
}

uint64_t canonicalSnapshotRequiredBytes(
    uint64_t residentBytes,
    const DatasetMaterializationDescription &dataset,
    const DeviceDatasetSessionDescription &session,
    uint64_t batchQueueDepth) {
    if (batchQueueDepth == 0) {
        throw std::runtime_error(
            "Device dataset batch_queue_depth must be >= 1.");
    }
    const uint64_t populatedSplits = nonEmptySplitCount(session.getSplits());
    const uint64_t batchSize = session.getBatching().getBatchSize();
    const uint64_t denseBytesPerBatch = checkedMul(
        batchSize,
        directBytesPerExample(dataset.layout),
        "Device dataset canonical batch tensor bytes");
    const uint64_t denseBatchRingBytes = checkedMul(
        checkedMul(
            populatedSplits,
            batchQueueDepth,
            "Device dataset canonical batch slot count"),
        denseBytesPerBatch,
        "Device dataset canonical batch-ring bytes");
    const uint64_t rowIndexBytes = checkedMul(
        populatedSplits,
        checkedMul(
            batchSize,
            static_cast<uint64_t>(sizeof(uint64_t)),
            "Device dataset canonical row-index bytes per split"),
        "Device dataset canonical row-index bytes");
    return checkedAdd(
        checkedAdd(
            checkedAdd(
                residentBytes,
                denseBatchRingBytes,
                "Device dataset canonical required bytes"),
            compactRaggedBatchRingBytes(session, batchQueueDepth),
            "Device dataset canonical ragged required bytes"),
        rowIndexBytes,
        "Device dataset canonical required bytes");
}

uint64_t compactRequiredBytes(
    uint64_t residentBytes,
    const DeviceDatasetSessionDescription &session,
    uint64_t batchQueueDepth) {
    return checkedAdd(
        checkedAdd(
            residentBytes,
            compactSelectionRingBytes(session, batchQueueDepth),
            "Device dataset compact reference required bytes"),
        compactRaggedBatchRingBytes(session, batchQueueDepth),
        "Device dataset compact ragged required bytes");
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
    if (snapshot.fields.size() + snapshot.raggedFields.size() != description.schema.size()) {
        throw std::runtime_error(
            "NamedDataset materialization returned an unexpected field count.");
    }
    for (const DatasetField &field : description.schema.getFields()) {
        if (!snapshot.hasField(field.id)) {
            throw std::runtime_error(
                "NamedDataset materialization omitted field '" + field.name + "'.");
        }
        if (field.kind == DatasetFieldKind::RAGGED) {
            if (!snapshot.hasRaggedField(field.id)) {
                throw std::runtime_error(
                    "NamedDataset materialization returned dense storage for ragged field '" +
                    field.name + "'.");
            }
            const MaterializedRaggedFieldSnapshot &ragged = snapshot.raggedField(field.id);
            if (ragged.valuesDataType != field.dataType ||
                ragged.trailingDimensions != field.dimensions ||
                ragged.valueBytes != description.layout.raggedTensor(field.name).valueNumBytes() ||
                !ragged.offsets.isInitialized() ||
                ragged.offsets.getDimensions() != std::vector<uint64_t>{description.numExamples + 1} ||
                !ThorImplementation::RowPartitionDescriptor::isValidOffsetsDataType(
                    ragged.offsets.getDataType())) {
                throw std::runtime_error(
                    "NamedDataset materialization returned the wrong ragged descriptor for field '" +
                    field.name + "'.");
            }
            if (ragged.storedValueCount == 0) {
                if (ragged.values.isInitialized()) {
                    throw std::runtime_error(
                        "NamedDataset materialization allocated values for all-empty ragged field '" +
                        field.name + "'.");
                }
            } else {
                std::vector<uint64_t> expectedValueDimensions;
                expectedValueDimensions.reserve(field.dimensions.size() + 1);
                expectedValueDimensions.push_back(ragged.storedValueCount);
                expectedValueDimensions.insert(
                    expectedValueDimensions.end(), field.dimensions.begin(), field.dimensions.end());
                const ThorImplementation::TensorDescriptor expectedValuesDescriptor(
                    field.dataType, expectedValueDimensions);
                if (!ragged.values.isInitialized() ||
                    ragged.values.getDescriptor() != expectedValuesDescriptor) {
                    throw std::runtime_error(
                        "NamedDataset materialization returned the wrong packed values descriptor for field '" +
                        field.name + "'.");
                }
            }
            continue;
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
    std::shared_ptr<const MaterializedNamedDatasetSnapshot> canonicalSnapshot,
    std::optional<uint64_t> availableBytesOverride,
    DeviceDatasetStorageReport report) {
    DeviceDatasetResidencyCache &cache =
        ThorImplementation::NamedDatasetRuntimeAccess::residencyCache(*namedDataset);
    const auto started = std::chrono::steady_clock::now();
    const bool compactFileResidency = usesCompactFileResidency(dataset);
    const bool strictWindowedOnly =
        requested == DeviceDatasetStorage::STRICT_WINDOWED_ONLY;
    const std::set<std::string> mandatoryNames =
        compactFileResidency ? mandatoryCompactFieldNames(dataset.layout)
                             : std::set<std::string>{};

    // File-backed windowed/ragged datasets stay in compact storage. Other
    // materializable datasets retain canonical shared residency.
    if (compactFileResidency) {
        const std::set<std::string> allNames = allCompactFieldNames(dataset.layout);
        const uint64_t mandatoryResidentBytes =
            DeviceResidentNamedDataset::estimateCompactFileDatasetBytes(
                dataset,
                mandatoryNames);
        const uint64_t mandatoryRequiredBytes = compactRequiredBytes(
            mandatoryResidentBytes,
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
                    std::make_shared<DeviceResidentFileNamedBatchSession>(
                        dataset,
                        session,
                        acquisition.lease,
                        batchQueueDepth,
                        32,
                        datasetName);
                ThorImplementation::BatchSessionRuntimeAccess::setTailMode(
                    *effectiveSession,
                    ThorImplementation::BatchSessionRuntimeAccess::getTailMode(*sourceSession));
                report.used = true;
                report.reason = successReason;
                report.windowedDeviceCache =
                    effectiveSession->getWindowedDeviceCacheReport();
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
            } catch (const WindowedDeviceCacheRequiredError &) {
                throw;
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
        // with all mandatory windowed/ragged fields resident and direct fields CPU-backed.
        CompactAttemptFailure fullFailure;
        if (!strictWindowedOnly && allNames != mandatoryNames) {
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

        CompactAttemptFailure mandatoryFailure;
        const char *mandatorySuccessReason =
            allNames == mandatoryNames
                ? "compact_file_residency"
                : (dataset.layout.hasWindowedTensors()
                       ? "compact_windowed_residency"
                       : "compact_ragged_residency");
        if (std::optional<DeviceDatasetStorageSelection> compact =
                attemptCompactResidency(
                    mandatoryNames,
                    mandatoryResidentBytes,
                    mandatoryRequiredBytes,
                    mandatorySuccessReason,
                    mandatoryFailure);
            compact.has_value()) {
            return std::move(compact.value());
        }

        report.reason = mandatoryFailure.reason.empty()
                            ? "device_dataset_materialization_failed"
                            : mandatoryFailure.reason;
        if (report.reason == "insufficient_device_memory") {
            report.reason = dataset.layout.hasWindowedTensors()
                                ? "insufficient_device_memory_for_windowed_dataset"
                                : "insufficient_device_memory_for_ragged_dataset";
        }
        report.requiredBytes = mandatoryRequiredBytes;
        report.availableBytesAfterPlacement = mandatoryFailure.availableBytes;
        report.materializationSeconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        return fallbackSelection(sourceSession, std::move(report), requested);
    }

    const uint64_t residentBytes = canonicalSnapshot != nullptr
                                       ? canonicalSnapshotResidentBytes(*canonicalSnapshot)
                                       : estimateDeviceResidentNamedDatasetStorageBytes(dataset);
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
            [namedDataset, dataset, devicePlacement, canonicalSnapshot]() {
                if (canonicalSnapshot != nullptr) {
                    return std::shared_ptr<const DeviceResidentNamedDataset>(
                        DeviceResidentNamedDataset::fromSnapshot(
                            *canonicalSnapshot,
                            devicePlacement));
                }
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
        ThorImplementation::BatchSessionRuntimeAccess::setTailMode(
            *effectiveSession,
            ThorImplementation::BatchSessionRuntimeAccess::getTailMode(*sourceSession));
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
    if (usesCompactFileResidency(dataset)) {
        return estimateDeviceResidentCompactDatasetStorageBytes(dataset);
    }
    if (dataset.layout.hasRaggedTensors()) {
        throw std::runtime_error(
            "Canonical ragged device residency size requires a materialized dataset snapshot.");
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
    if (usesCompactFileResidency(dataset)) {
        return estimateDeviceResidentCompactDatasetRequiredBytes(
            dataset,
            session,
            batchQueueDepth);
    }
    if (dataset.layout.hasRaggedTensors()) {
        throw std::runtime_error(
            "Canonical ragged device residency size requires a materialized dataset snapshot.");
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
    const DatasetFieldMaterializationRequirements& fieldRequirements,
    WindowedDeviceCache windowedDeviceCache) {
    return DeviceDatasetSessionDescription(
        splits, batching, fieldRequirements, windowedDeviceCache);
}

DeviceDatasetSessionDescription describeDeviceDatasetSession(
    const TrainingData& trainingData,
    const DatasetFieldMaterializationRequirements& fieldRequirements) {
    return describeDeviceDatasetSession(
        trainingData.getSplits(),
        trainingData.getBatching(),
        fieldRequirements,
        trainingData.getAccessPolicy().windowedDeviceCache);
}

DeviceDatasetStorageSelection selectDeviceDatasetStorageSession(
    const std::shared_ptr<BatchSession>& sourceSession,
    const TrainingData& trainingData,
    ThorImplementation::TensorPlacement devicePlacement,
    uint64_t batchQueueDepth,
    std::optional<uint64_t> availableBytesOverride) {
    return selectDeviceDatasetStorageSession(
        sourceSession,
        trainingData,
        trainingData.getSplits(),
        devicePlacement,
        batchQueueDepth,
        availableBytesOverride);
}

DeviceDatasetStorageSelection selectDeviceDatasetStorageSession(
    const std::shared_ptr<BatchSession>& sourceSession,
    const TrainingData& trainingData,
    const DatasetSplitManifest& sessionSplits,
    ThorImplementation::TensorPlacement devicePlacement,
    uint64_t batchQueueDepth,
    std::optional<uint64_t> availableBytesOverride) {
    THOR_THROW_IF_FALSE(sourceSession != nullptr);
    sessionSplits.validateAgainst(*trainingData.getDataset());
    const DeviceDatasetStorage requested =
        trainingData.getAccessPolicy().deviceStorage;
    DeviceDatasetStorageReport report;
    report.requested = requested;
    report.windowedDeviceCache.requested =
        trainingData.getAccessPolicy().windowedDeviceCache;

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
            sessionSplits,
            trainingData.getBatching(),
            sourceSession->getDatasetFieldMaterializationRequirements(),
            trainingData.getAccessPolicy().windowedDeviceCache);

    report.examples = datasetDescription->numExamples;
    if (requested == DeviceDatasetStorage::STRICT_WINDOWED_ONLY &&
        !(datasetDescription->source == DatasetMaterializationSource::FILE_DATASET &&
          datasetDescription->layout.hasWindowedTensors())) {
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

    if (!usesCompactFileResidency(*datasetDescription)) {
        const NamedDatasetMaterializationSupport support =
            checkNamedDatasetSnapshotMaterializationSupport(*datasetDescription);
        if (!support.supported) {
            report.reason = support.reason.empty()
                                ? "dataset_not_materializable"
                                : support.reason;
            return fallbackSelection(sourceSession, std::move(report), requested);
        }
    }

    const std::shared_ptr<const NamedDataset>& namedDataset =
        trainingData.getDataset();
    if (namedDataset->getId() != datasetDescription->datasetId ||
        namedDataset->getNumExamples() != datasetDescription->numExamples) {
        report.reason = "training_data_dataset_identity_mismatch";
        return fallbackSelection(sourceSession, std::move(report), requested);
    }

    uint64_t requiredBytes = 0;
    std::shared_ptr<const MaterializedNamedDatasetSnapshot> canonicalSnapshot;
    try {
        if (!usesCompactFileResidency(*datasetDescription) &&
            datasetDescription->layout.hasRaggedTensors()) {
            canonicalSnapshot = std::make_shared<MaterializedNamedDatasetSnapshot>(
                materializeCanonicalSnapshot(namedDataset, *datasetDescription));
            const uint64_t residentBytes =
                canonicalSnapshotResidentBytes(*canonicalSnapshot);
            requiredBytes = canonicalSnapshotRequiredBytes(
                residentBytes,
                *datasetDescription,
                sessionDescription,
                batchQueueDepth);
        } else {
            requiredBytes = estimateDeviceResidentNamedDatasetRequiredBytes(
                *datasetDescription,
                sessionDescription,
                batchQueueDepth);
        }
        report.requiredBytes = requiredBytes;
    } catch (const std::exception& e) {
        report.reason =
            std::string("device_dataset_size_estimate_failed:") + e.what();
        return fallbackSelection(sourceSession, std::move(report), requested);
    }

    std::string datasetName = sourceSession->getDatasetName();
    if (datasetName.empty()) {
        datasetName = trainingData.getDatasetName();
    }
    return selectSharedResidencySession(
        sourceSession,
        datasetName,
        namedDataset,
        *datasetDescription,
        sessionDescription,
        requested,
        devicePlacement,
        batchQueueDepth,
        requiredBytes,
        std::move(canonicalSnapshot),
        availableBytesOverride,
        std::move(report));
}

}  // namespace Thor
