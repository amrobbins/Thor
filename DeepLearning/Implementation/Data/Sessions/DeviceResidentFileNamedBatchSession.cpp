#include "DeepLearning/Implementation/Data/Sessions/DeviceResidentFileNamedBatchSession.h"
#include "DeepLearning/Implementation/Data/BatchCardinality.h"

#include "DeepLearning/Implementation/ThorError.h"

#include <stdexcept>
#include <utility>

using ThorImplementation::DataType;
using ThorImplementation::RaggedTensor;
using ThorImplementation::RaggedTensorDescriptor;
using ThorImplementation::Tensor;
using ThorImplementation::TensorDescriptor;
using ThorImplementation::TensorPlacement;

struct DeviceResidentFileSelectionState {
    uint64_t slotIndex = 0;
    Tensor rowIndicesHost;
    Tensor rowIndicesDevice;
    Event rowsReadyEvent;
};

struct DeviceResidentFileSelectionSlot {
    std::shared_ptr<DeviceResidentFileSelectionState> state;
};

struct DeviceResidentFileDirectSlot {
    uint64_t slotIndex = 0;
    std::map<std::string, Tensor> directTensors;
};

struct DeviceResidentFileRaggedSlot {
    uint64_t slotIndex = 0;
    std::map<std::string, RaggedTensor> raggedTensors;
    Event valuesReadyEvent;
};

struct DeviceResidentFilePendingSelection {
    std::shared_ptr<DeviceResidentFileSelectionSlot> slot;
    std::vector<Event> consumedEvents;
};

struct DeviceResidentFilePendingDirect {
    std::shared_ptr<DeviceResidentFileDirectSlot> slot;
    std::vector<Event> consumedEvents;
};

struct DeviceResidentFilePendingRagged {
    std::shared_ptr<DeviceResidentFileRaggedSlot> slot;
    std::vector<Event> consumedEvents;
};

namespace {

struct BatchFieldSpec {
    std::string name;
    DataType dataType = DataType::FP32;
    std::vector<uint64_t> exampleDimensions;
    bool windowed = false;
};

class CompactResidentFieldMaterializer final : public Thor::DeviceBatchMaterializer {
   public:
    CompactResidentFieldMaterializer(
        std::shared_ptr<const DeviceResidentNamedDataset> dataset,
        std::string fieldName,
        std::shared_ptr<const DeviceResidentFileSelectionState> selection,
        TensorDescriptor outputDescriptor)
        : dataset(std::move(dataset)),
          fieldName(std::move(fieldName)),
          selection(std::move(selection)),
          outputDescriptor(std::move(outputDescriptor)) {
        THOR_THROW_IF_FALSE(this->dataset != nullptr);
        THOR_THROW_IF_FALSE(this->dataset->hasCompactField(this->fieldName));
        THOR_THROW_IF_FALSE(!this->dataset->hasCompactRaggedField(this->fieldName));
        THOR_THROW_IF_FALSE(this->selection != nullptr);
        THOR_THROW_IF_FALSE(this->selection->rowIndicesDevice.isInitialized());
    }

    TensorDescriptor getOutputDescriptor() const override { return outputDescriptor; }
    TensorPlacement getOutputPlacement() const override { return dataset->getPlacement(); }

    void enqueueMaterialization(
        Tensor &destination,
        Stream &destinationStream) const override {
        THOR_THROW_IF_FALSE(destination.isInitialized());
        THOR_THROW_IF_FALSE(destination.getDescriptor() == outputDescriptor);
        THOR_THROW_IF_FALSE(destination.getPlacement() == dataset->getPlacement());
        THOR_THROW_IF_FALSE(selection->rowsReadyEvent.isInitialized());
        destinationStream.waitEvent(selection->rowsReadyEvent);
        dataset->enqueueCompactFieldMaterialization(
            fieldName,
            selection->rowIndicesDevice,
            destination,
            destinationStream);
    }

   private:
    std::shared_ptr<const DeviceResidentNamedDataset> dataset;
    std::string fieldName;
    std::shared_ptr<const DeviceResidentFileSelectionState> selection;
    TensorDescriptor outputDescriptor;
};

uint64_t batchesFor(uint64_t numExamples, uint64_t batchSize) {
    THOR_THROW_IF_FALSE(batchSize > 0);
    return (numExamples / batchSize) + ((numExamples % batchSize) == 0 ? 0 : 1);
}

const char *splitNameFor(ExampleType exampleType) {
    if (exampleType == ExampleType::TRAIN) return "train";
    if (exampleType == ExampleType::VALIDATE) return "validate";
    if (exampleType == ExampleType::TEST) return "test";
    return "unknown";
}

std::vector<uint64_t> batchDimensionsFor(
    const std::vector<uint64_t> &exampleDimensions,
    uint64_t batchSize) {
    std::vector<uint64_t> dimensions;
    dimensions.reserve(exampleDimensions.size() + 1);
    dimensions.push_back(batchSize);
    dimensions.insert(dimensions.end(), exampleDimensions.begin(), exampleDimensions.end());
    return dimensions;
}

std::vector<BatchFieldSpec> batchFieldSpecsFor(const DatasetLayout &layout) {
    std::vector<BatchFieldSpec> specs;
    specs.reserve(layout.tensors().size() + layout.windowedTensors().size() * 2);
    for (const DatasetLayout::TensorSpec &spec : layout.tensors()) {
        specs.push_back(BatchFieldSpec{spec.name, spec.dataType, spec.dimensions, false});
    }
    for (const DatasetLayout::WindowedTensorSpec &spec : layout.windowedTensors()) {
        specs.push_back(BatchFieldSpec{spec.name, spec.dataType, spec.dimensions, true});
        if (spec.maskName.has_value()) {
            specs.push_back(BatchFieldSpec{
                spec.maskName.value(), DataType::UINT8,
                std::vector<uint64_t>{spec.windowLength()}, true});
        }
    }
    return specs;
}

std::optional<BatchFieldSpec> findBatchFieldSpec(
    const DatasetLayout &layout,
    const std::string &name) {
    for (const BatchFieldSpec &spec : batchFieldSpecsFor(layout)) {
        if (spec.name == name) return spec;
    }
    return std::nullopt;
}


}  // namespace

DeviceResidentFileNamedBatchSession::DeviceResidentFileNamedBatchSession(
    Thor::DatasetMaterializationDescription datasetDescription,
    Thor::DeviceDatasetSessionDescription sessionDescription,
    Thor::DeviceDatasetLease residentDataset,
    uint64_t batchQueueDepth,
    uint64_t readerQueueDepth,
    std::string datasetName)
    : Thor::BatchSession(std::move(datasetName)),
      datasetDescription(std::move(datasetDescription)),
      sessionDescription(std::move(sessionDescription)),
      residentDataset(std::move(residentDataset)),
      batchQueueDepth(batchQueueDepth),
      readerQueueDepth(readerQueueDepth) {
    if (!this->residentDataset) {
        throw std::runtime_error(
            "DeviceResidentFileNamedBatchSession requires a device dataset.");
    }
    if (!this->datasetDescription.layout.hasWindowedTensors() &&
        !this->datasetDescription.layout.hasRaggedTensors()) {
        throw std::runtime_error(
            "DeviceResidentFileNamedBatchSession requires at least one windowed or ragged tensor.");
    }
    if (!this->residentDataset->usesCompactFileStorage()) {
        throw std::runtime_error(
            "DeviceResidentFileNamedBatchSession requires compact file device storage.");
    }
    if (batchQueueDepth == 0) {
        throw std::runtime_error(
            "DeviceResidentFileNamedBatchSession batch_queue_depth must be >= 1.");
    }
    if (this->residentDataset->getDatasetId() != this->datasetDescription.datasetId ||
        this->residentDataset->getNumExamples() != this->datasetDescription.numExamples) {
        throw std::runtime_error(
            "DeviceResidentFileNamedBatchSession resident dataset does not match source dataset description.");
    }
    const Thor::DatasetSplitManifest &splits = this->sessionDescription.getSplits();
    if (splits.getDatasetId() != this->datasetDescription.datasetId ||
        splits.getNumExamples() != this->datasetDescription.numExamples) {
        throw std::runtime_error(
            "DeviceResidentFileNamedBatchSession split manifest does not match source dataset.");
    }

    this->batchSize = this->sessionDescription.getBatching().getBatchSize();
    fieldRequirements = this->sessionDescription.getFieldRequirements();
    if (fieldRequirements.empty()) {
        for (const Thor::DatasetField &field : this->datasetDescription.schema.getFields()) {
            if (field.kind == Thor::DatasetFieldKind::RAGGED) {
                throw std::runtime_error(
                    "DeviceResidentFileNamedBatchSession requires an explicit materialization descriptor for ragged field '" +
                    field.name + "'.");
            }
            fieldRequirements.emplace(
                field.id, Thor::DatasetFieldMaterializationRequirement::dense(field.id));
        }
    }
    for (const auto &[fieldId, requirement] : fieldRequirements) {
        if (fieldId != requirement.fieldId) {
            throw std::runtime_error(
                "DeviceResidentFileNamedBatchSession field requirement key/id mismatch.");
        }
        const Thor::DatasetField &field = this->datasetDescription.schema.getField(fieldId);
        if (field.kind == Thor::DatasetFieldKind::RAGGED) {
            if (!requirement.raggedTensorDescriptor.has_value()) {
                throw std::runtime_error(
                    "DeviceResidentFileNamedBatchSession ragged field '" + field.name +
                    "' requires a materialization descriptor.");
            }
            const RaggedTensorDescriptor &descriptor =
                requirement.raggedTensorDescriptor.value();
            if (descriptor.getValuesDataType() != field.dataType ||
                descriptor.getTrailingDimensions() != field.dimensions ||
                descriptor.getBatchSize() != this->batchSize) {
                throw std::runtime_error(
                    "DeviceResidentFileNamedBatchSession ragged materialization contract does not match field '" +
                    field.name + "'.");
            }
            if (!this->residentDataset->hasCompactRaggedField(field.name)) {
                throw std::runtime_error(
                    "DeviceResidentFileNamedBatchSession compact resident dataset is missing ragged field '" +
                    field.name + "'.");
            }
            raggedFieldNames.insert(field.name);
        } else if (requirement.raggedTensorDescriptor.has_value()) {
            throw std::runtime_error(
                "DeviceResidentFileNamedBatchSession non-ragged field '" + field.name +
                "' cannot carry a RaggedTensor materialization descriptor.");
        }
    }

    uint64_t residentDirectFieldCount = 0;
    for (const BatchFieldSpec &spec : batchFieldSpecsFor(this->datasetDescription.layout)) {
        if (this->residentDataset->hasCompactField(spec.name)) {
            residentReferenceFieldNames.insert(spec.name);
            if (!spec.windowed) residentDirectFieldCount += 1;
        } else if (spec.windowed) {
            throw std::runtime_error(
                "DeviceResidentFileNamedBatchSession compact resident dataset is missing window field '" +
                spec.name + "'.");
        } else {
            directFieldNames.insert(spec.name);
        }
    }
    if (residentDirectFieldCount != 0 && !directFieldNames.empty()) {
        throw std::runtime_error(
            "DeviceResidentFileNamedBatchSession requires direct fields to be either all compact-resident or all CPU-backed.");
    }
    if (!directFieldNames.empty()) {
        if (readerQueueDepth == 0) {
            throw std::runtime_error(
                "DeviceResidentFileNamedBatchSession reader_queue_depth must be >= 1 when CPU direct fields are used.");
        }
        this->reader = IndexedDatasetReader::openDataset(
            this->datasetDescription.datasetPath,
            this->datasetDescription.layout);
        if (this->reader->getNumExamples() != this->datasetDescription.numExamples) {
            throw std::runtime_error(
                "DeviceResidentFileNamedBatchSession source dataset row count changed.");
        }
    }

    initializeSplit(ExampleType::TRAIN, splits.getSharedTrain(),
                    this->sessionDescription.getBatching().getRandomizeTrain(),
                    this->sessionDescription.getBatching().getRandomSeed());
    initializeSplit(ExampleType::VALIDATE, splits.getSharedValidate(), false, std::nullopt);
    initializeSplit(ExampleType::TEST, splits.getSharedTest(), false, std::nullopt);
}

DeviceResidentFileNamedBatchSession::~DeviceResidentFileNamedBatchSession() {
    cancel();
    for (auto &entry : splitRuntimes) {
        if (entry.second == nullptr) continue;
        try {
            std::lock_guard<std::mutex> guard(entry.second->mutex);
            for (DeviceResidentFilePendingSelection &pending : entry.second->pendingSelections) {
                for (Event &event : pending.consumedEvents) event.synchronize();
            }
            for (DeviceResidentFilePendingDirect &pending : entry.second->pendingDirectSlots) {
                for (Event &event : pending.consumedEvents) event.synchronize();
            }
            for (DeviceResidentFilePendingRagged &pending : entry.second->pendingRaggedSlots) {
                for (Event &event : pending.consumedEvents) event.synchronize();
            }
        } catch (...) {
        }
    }
}

void DeviceResidentFileNamedBatchSession::cancel() {
    if (cancelled.exchange(true, std::memory_order_acq_rel)) return;
    for (auto &entry : splitRuntimes) {
        if (entry.second != nullptr) entry.second->notEmpty.notify_all();
    }
}

void DeviceResidentFileNamedBatchSession::initializeSplit(
    ExampleType exampleType,
    std::shared_ptr<const Thor::ExampleIndexSet> sourceIndices,
    bool randomized,
    std::optional<uint64_t> seed) {
    auto runtime = std::make_unique<SplitRuntime>();
    runtime->exampleType = exampleType;
    runtime->splitName = splitNameFor(exampleType);
    runtime->sourceIndices = std::move(sourceIndices);
    runtime->randomized = randomized;
    runtime->seed = seed;
    runtime->batchesPerEpoch = batchesFor(runtime->numExamples(), batchSize);
    runtime->selectionUploadStream = Stream(residentDataset->getPlacement());
    runtime->raggedGatherStream = Stream(residentDataset->getPlacement());

    if (runtime->numExamples() != 0) {
        if (reader != nullptr) runtime->readerSession = reader->createSession(readerQueueDepth);
        if (runtime->randomized) {
            runtime->randomizer = std::make_unique<FullPeriodRandom>(runtime->numExamples(), false);
            if (runtime->seed.has_value()) runtime->randomizer->reseed(runtime->seed.value());
        }
        for (uint64_t i = 0; i < batchQueueDepth; ++i) {
            runtime->availableSelections.push_back(allocateSelectionSlot(i));
            if (!directFieldNames.empty()) runtime->availableDirectSlots.push_back(allocateDirectSlot(i));
            if (!raggedFieldNames.empty()) runtime->availableRaggedSlots.push_back(allocateRaggedSlot(i));
        }
    }

    auto [it, inserted] = splitRuntimes.emplace(exampleType, std::move(runtime));
    THOR_THROW_IF_FALSE(inserted);
    (void)it;
}

std::map<std::string, Tensor>
DeviceResidentFileNamedBatchSession::allocateDirectTensorSet() const {
    std::map<std::string, Tensor> tensors;
    const TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    for (const BatchFieldSpec &spec : batchFieldSpecsFor(datasetDescription.layout)) {
        if (directFieldNames.find(spec.name) == directFieldNames.end()) continue;
        tensors.emplace(
            spec.name,
            Tensor(cpuPlacement,
                   TensorDescriptor(spec.dataType,
                                    batchDimensionsFor(spec.exampleDimensions, batchSize))));
    }
    return tensors;
}

std::shared_ptr<DeviceResidentFileSelectionSlot>
DeviceResidentFileNamedBatchSession::allocateSelectionSlot(uint64_t slotIndex) const {
    auto slot = std::make_shared<DeviceResidentFileSelectionSlot>();
    slot->state = std::make_shared<DeviceResidentFileSelectionState>();
    slot->state->slotIndex = slotIndex;
    slot->state->rowIndicesHost = Tensor(
        TensorPlacement(TensorPlacement::MemDevices::CPU),
        TensorDescriptor(DataType::UINT64, {batchSize}));
    slot->state->rowIndicesDevice = Tensor(
        residentDataset->getPlacement(),
        TensorDescriptor(DataType::UINT64, {batchSize}));
    return slot;
}

std::shared_ptr<DeviceResidentFileDirectSlot>
DeviceResidentFileNamedBatchSession::allocateDirectSlot(uint64_t slotIndex) const {
    auto slot = std::make_shared<DeviceResidentFileDirectSlot>();
    slot->slotIndex = slotIndex;
    slot->directTensors = allocateDirectTensorSet();
    return slot;
}

std::shared_ptr<DeviceResidentFileRaggedSlot>
DeviceResidentFileNamedBatchSession::allocateRaggedSlot(uint64_t slotIndex) const {
    auto slot = std::make_shared<DeviceResidentFileRaggedSlot>();
    slot->slotIndex = slotIndex;
    for (const auto &[fieldId, requirement] : fieldRequirements) {
        if (!requirement.raggedTensorDescriptor.has_value()) continue;
        const Thor::DatasetField &field = datasetDescription.schema.getField(fieldId);
        const RaggedTensorDescriptor &descriptor = requirement.raggedTensorDescriptor.value();
        Tensor values(residentDataset->getPlacement(), descriptor.getValuesDescriptor());
        Tensor offsets(residentDataset->getPlacement(), descriptor.getOffsetsDescriptor());
        slot->raggedTensors.emplace(field.name, RaggedTensor(values, offsets));
    }
    return slot;
}

DeviceResidentFileNamedBatchSession::SplitRuntime &
DeviceResidentFileNamedBatchSession::runtimeFor(ExampleType exampleType) {
    const auto found = splitRuntimes.find(exampleType);
    if (found == splitRuntimes.end() || found->second == nullptr) {
        throw std::runtime_error(
            "DeviceResidentFileNamedBatchSession does not contain requested split.");
    }
    return *found->second;
}

const DeviceResidentFileNamedBatchSession::SplitRuntime &
DeviceResidentFileNamedBatchSession::runtimeFor(ExampleType exampleType) const {
    const auto found = splitRuntimes.find(exampleType);
    if (found == splitRuntimes.end() || found->second == nullptr) {
        throw std::runtime_error(
            "DeviceResidentFileNamedBatchSession does not contain requested split.");
    }
    return *found->second;
}

void DeviceResidentFileNamedBatchSession::fillRowIndexTensor(
    SplitRuntime &runtime,
    DeviceResidentFileSelectionSlot &selectionSlot,
    uint32_t validExampleCount) {
    THOR_THROW_IF_FALSE(runtime.numExamples() > 0);
    THOR_THROW_IF_FALSE(selectionSlot.state != nullptr);
    THOR_THROW_IF_FALSE(validExampleCount >= 1);
    THOR_THROW_IF_FALSE(validExampleCount <= batchSize);
    uint64_t *rowIndices = selectionSlot.state->rowIndicesHost.getMemPtr<uint64_t>();
    for (uint64_t slot = 0; slot < validExampleCount; ++slot) {
        uint64_t logicalPosition = 0;
        if (runtime.randomized) {
            THOR_THROW_IF_FALSE(runtime.randomizer != nullptr);
            logicalPosition = runtime.randomizer->getRandomNumber();
        } else {
            logicalPosition = runtime.nextLogicalPosition;
            runtime.nextLogicalPosition += 1;
            if (runtime.nextLogicalPosition == runtime.numExamples()) runtime.nextLogicalPosition = 0;
        }
        THOR_THROW_IF_FALSE(logicalPosition < runtime.numExamples());
        const uint64_t sourceRow = runtime.sourceIndices->at(logicalPosition);
        THOR_THROW_IF_FALSE(sourceRow < datasetDescription.numExamples);
        rowIndices[slot] = sourceRow;
    }
    const uint64_t paddingSourceRow = rowIndices[validExampleCount - 1];
    for (uint64_t slot = validExampleCount; slot < batchSize; ++slot) rowIndices[slot] = paddingSourceRow;
}

Batch DeviceResidentFileNamedBatchSession::acquireBatch(
    ExampleType exampleType,
    uint64_t &batchNum) {
    if (cancelled.load(std::memory_order_acquire)) {
        throw std::runtime_error("DeviceResidentFileNamedBatchSession has been cancelled.");
    }
    SplitRuntime &runtime = runtimeFor(exampleType);
    std::unique_lock<std::mutex> lock(runtime.mutex);
    if (runtime.numExamples() == 0) {
        throw std::runtime_error(
            "DeviceResidentFileNamedBatchSession cannot get a batch from an empty split.");
    }
    runtime.notEmpty.wait(lock, [&] {
        const bool selectionReady =
            !runtime.availableSelections.empty() || !runtime.pendingSelections.empty();
        const bool directReady = directFieldNames.empty() ||
                                 !runtime.availableDirectSlots.empty() ||
                                 !runtime.pendingDirectSlots.empty();
        const bool raggedReady = raggedFieldNames.empty() ||
                                 !runtime.availableRaggedSlots.empty() ||
                                 !runtime.pendingRaggedSlots.empty();
        return cancelled.load(std::memory_order_acquire) ||
               (selectionReady && directReady && raggedReady);
    });
    if (cancelled.load(std::memory_order_acquire)) {
        throw std::runtime_error("DeviceResidentFileNamedBatchSession has been cancelled.");
    }

    std::shared_ptr<DeviceResidentFileSelectionSlot> selectionSlot;
    if (!runtime.availableSelections.empty()) {
        selectionSlot = std::move(runtime.availableSelections.front());
        runtime.availableSelections.pop_front();
    } else {
        DeviceResidentFilePendingSelection pending = std::move(runtime.pendingSelections.front());
        runtime.pendingSelections.pop_front();
        lock.unlock();
        for (Event &event : pending.consumedEvents) event.synchronize();
        lock.lock();
        selectionSlot = std::move(pending.slot);
    }

    std::shared_ptr<DeviceResidentFileDirectSlot> directSlot;
    if (!directFieldNames.empty()) {
        if (!runtime.availableDirectSlots.empty()) {
            directSlot = std::move(runtime.availableDirectSlots.front());
            runtime.availableDirectSlots.pop_front();
        } else {
            DeviceResidentFilePendingDirect pending = std::move(runtime.pendingDirectSlots.front());
            runtime.pendingDirectSlots.pop_front();
            lock.unlock();
            for (Event &event : pending.consumedEvents) event.synchronize();
            lock.lock();
            directSlot = std::move(pending.slot);
        }
    }

    std::shared_ptr<DeviceResidentFileRaggedSlot> raggedSlot;
    if (!raggedFieldNames.empty()) {
        if (!runtime.availableRaggedSlots.empty()) {
            raggedSlot = std::move(runtime.availableRaggedSlots.front());
            runtime.availableRaggedSlots.pop_front();
        } else {
            DeviceResidentFilePendingRagged pending = std::move(runtime.pendingRaggedSlots.front());
            runtime.pendingRaggedSlots.pop_front();
            lock.unlock();
            for (Event &event : pending.consumedEvents) event.synchronize();
            lock.lock();
            raggedSlot = std::move(pending.slot);
        }
    }

    if (cancelled.load(std::memory_order_acquire)) {
        runtime.availableSelections.push_front(std::move(selectionSlot));
        if (directSlot != nullptr) runtime.availableDirectSlots.push_front(std::move(directSlot));
        if (raggedSlot != nullptr) runtime.availableRaggedSlots.push_front(std::move(raggedSlot));
        throw std::runtime_error("DeviceResidentFileNamedBatchSession has been cancelled.");
    }

    batchNum = runtime.nextBatchNum;
    const uint32_t validExampleCount =
        usesWrappedBatchTailForRuntime()
            ? ThorImplementation::fullBatchValidExampleCount(batchSize)
            : ThorImplementation::validExamplesForBatch(
                  batchNum, runtime.numExamples(), batchSize);
    runtime.nextBatchNum = (runtime.nextBatchNum + 1) % runtime.batchesPerEpoch;
    fillRowIndexTensor(runtime, *selectionSlot, validExampleCount);
    lock.unlock();

    bool selectionManagedByOwner = false;
    bool directManagedByOwner = false;
    bool raggedManagedByOwner = false;
    Event raggedReadyEvent;
    try {
        if (directSlot != nullptr) {
            std::lock_guard<std::mutex> readerGuard(runtime.readerMutex);
            std::vector<uint8_t *> directPointers(reader->getTensorCount(), nullptr);
            for (const DatasetLayout::TensorSpec &spec : datasetDescription.layout.tensors()) {
                if (directFieldNames.find(spec.name) == directFieldNames.end()) continue;
                const uint64_t ordinal = reader->getLayoutTensorOrdinal(spec.name);
                directPointers.at(static_cast<size_t>(ordinal)) =
                    static_cast<uint8_t *>(directSlot->directTensors.at(spec.name).getMemPtr());
            }
            const uint64_t *sourceRows = selectionSlot->state->rowIndicesHost.getMemPtr<uint64_t>();
            THOR_THROW_IF_FALSE(runtime.readerSession != nullptr);
            for (uint64_t slot = 0; slot < batchSize; ++slot) {
                runtime.readerSession->loadDirectExampleInto(
                    sourceRows[slot], slot, directPointers);
            }
            runtime.readerSession->drain();
        }

        if (raggedSlot != nullptr) {
            for (const std::string &fieldName : raggedFieldNames) {
                const Thor::DatasetField &field = datasetDescription.schema.getField(fieldName);
                const RaggedTensorDescriptor &descriptor =
                    fieldRequirements.at(field.id).raggedTensorDescriptor.value();
                const uint64_t activeRows = residentDataset->validateCompactRaggedBatchCapacity(
                    fieldName,
                    selectionSlot->state->rowIndicesHost,
                    validExampleCount,
                    descriptor.getMaxTotalValues());
                raggedSlot->raggedTensors.at(fieldName)
                    .getRowPartitionRuntime()
                    .setHostActiveValueCount(activeRows);
            }
        }

        selectionSlot->state->rowIndicesDevice.copyFromAsync(
            selectionSlot->state->rowIndicesHost,
            runtime.selectionUploadStream);
        runtime.selectionUploadStream.putEvent(selectionSlot->state->rowsReadyEvent);

        if (raggedSlot != nullptr) {
            runtime.raggedGatherStream.waitEvent(selectionSlot->state->rowsReadyEvent);
            for (const std::string &fieldName : raggedFieldNames) {
                RaggedTensor &ragged = raggedSlot->raggedTensors.at(fieldName);
                residentDataset->enqueueCompactRaggedFieldMaterialization(
                    fieldName,
                    selectionSlot->state->rowIndicesDevice,
                    validExampleCount,
                    ragged,
                    runtime.raggedGatherStream);
            }
            runtime.raggedGatherStream.putEvent(
                raggedSlot->valuesReadyEvent, false, true);
            raggedReadyEvent = raggedSlot->valuesReadyEvent;
        }

        Batch batch;
        for (const BatchFieldSpec &spec : batchFieldSpecsFor(datasetDescription.layout)) {
            const TensorDescriptor outputDescriptor(
                spec.dataType,
                batchDimensionsFor(spec.exampleDimensions, batchSize));
            if (residentReferenceFieldNames.find(spec.name) == residentReferenceFieldNames.end()) {
                THOR_THROW_IF_FALSE(directSlot != nullptr);
                batch.insert(spec.name, directSlot->directTensors.at(spec.name));
                continue;
            }
            std::shared_ptr<const Thor::DeviceBatchMaterializer> materializer =
                std::make_shared<CompactResidentFieldMaterializer>(
                    residentDataset.getShared(), spec.name, selectionSlot->state, outputDescriptor);
            batch.insert(
                spec.name,
                Thor::DeviceBatchReference(
                    std::move(materializer),
                    ThorImplementation::fullBatchValidExampleCount(batchSize)));
        }
        if (raggedSlot != nullptr) {
            for (const std::string &fieldName : raggedFieldNames) {
                batch.insert(fieldName, raggedSlot->raggedTensors.at(fieldName));
            }
        }
        if (validExampleCount < batchSize) batch.setValidExampleCount(validExampleCount);

        std::shared_ptr<DeviceResidentFileNamedBatchSession> sharedSelf =
            std::dynamic_pointer_cast<DeviceResidentFileNamedBatchSession>(shared_from_this());
        THOR_THROW_IF_FALSE(sharedSelf != nullptr);
        std::weak_ptr<DeviceResidentFileNamedBatchSession> weakSelf = sharedSelf;

        if (directSlot != nullptr) {
            Thor::BatchSourceOwner directOwner(
                [weakSelf, exampleType, directSlot](std::vector<Event> consumedEvents) mutable {
                    if (auto session = weakSelf.lock()) {
                        session->releaseDirectSlot(
                            exampleType, std::move(directSlot), std::move(consumedEvents));
                    }
                });
            directManagedByOwner = true;
            addBatchSourceResource(batch, directFieldNames, std::move(directOwner));
        }

        if (raggedSlot != nullptr) {
            THOR_THROW_IF_FALSE(raggedReadyEvent.isInitialized());
            Thor::BatchSourceOwner raggedOwner(
                [weakSelf, exampleType, raggedSlot](std::vector<Event> consumedEvents) mutable {
                    if (auto session = weakSelf.lock()) {
                        session->releaseRaggedSlot(
                            exampleType, std::move(raggedSlot), std::move(consumedEvents));
                    }
                },
                raggedReadyEvent);
            raggedManagedByOwner = true;
            addBatchSourceResource(batch, raggedFieldNames, std::move(raggedOwner));
        }

        if (!residentReferenceFieldNames.empty()) {
            Thor::BatchSourceOwner selectionOwner(
                [weakSelf, exampleType, selectionSlot, raggedReadyEvent](
                    std::vector<Event> consumedEvents) mutable {
                    if (raggedReadyEvent.isInitialized()) {
                        consumedEvents.push_back(raggedReadyEvent);
                    } else if (consumedEvents.empty()) {
                        consumedEvents.push_back(selectionSlot->state->rowsReadyEvent);
                    }
                    if (auto session = weakSelf.lock()) {
                        session->releaseSelectionSlot(
                            exampleType, std::move(selectionSlot), std::move(consumedEvents));
                    }
                });
            selectionManagedByOwner = true;
            addBatchSourceResource(batch, residentReferenceFieldNames, std::move(selectionOwner));
        } else {
            std::vector<Event> selectionConsumedEvents;
            if (raggedReadyEvent.isInitialized()) {
                selectionConsumedEvents.push_back(raggedReadyEvent);
            } else {
                selectionConsumedEvents.push_back(selectionSlot->state->rowsReadyEvent);
            }
            releaseSelectionSlot(
                exampleType, std::move(selectionSlot), std::move(selectionConsumedEvents));
            selectionManagedByOwner = true;
        }

        validateReturnedBatch(batch);
        return batch;
    } catch (...) {
        std::vector<Event> producerEvents;
        if (raggedReadyEvent.isInitialized()) producerEvents.push_back(raggedReadyEvent);
        if (!selectionManagedByOwner) {
            releaseSelectionSlot(exampleType, std::move(selectionSlot), producerEvents);
        }
        if (directSlot != nullptr && !directManagedByOwner) {
            releaseDirectSlot(exampleType, std::move(directSlot), {});
        }
        if (raggedSlot != nullptr && !raggedManagedByOwner) {
            releaseRaggedSlot(exampleType, std::move(raggedSlot), std::move(producerEvents));
        }
        throw;
    }
}

void DeviceResidentFileNamedBatchSession::validateReturnedBatch(const Batch &batch) const {
    const std::vector<BatchFieldSpec> specs = batchFieldSpecsFor(datasetDescription.layout);
    if (batch.size() != specs.size() + raggedFieldNames.size()) {
        throw std::runtime_error(
            "DeviceResidentFileNamedBatchSession returned batch has wrong field count.");
    }
    for (const BatchFieldSpec &spec : specs) {
        if (!batch.contains(spec.name)) {
            throw std::runtime_error(
                "DeviceResidentFileNamedBatchSession returned batch is missing field '" +
                spec.name + "'.");
        }
        if (!batch.getSourceReference(spec.name).has_value()) {
            throw std::runtime_error(
                "DeviceResidentFileNamedBatchSession returned field '" + spec.name +
                "' is missing its source-resource reference.");
        }
        const TensorDescriptor expected(
            spec.dataType,
            batchDimensionsFor(spec.exampleDimensions, batchSize));
        if (residentReferenceFieldNames.find(spec.name) == residentReferenceFieldNames.end()) {
            if (!batch.isTensor(spec.name)) {
                throw std::runtime_error(
                    "DeviceResidentFileNamedBatchSession returned direct field '" +
                    spec.name + "' is not a tensor.");
            }
            const Tensor &tensor = batch.getTensor(spec.name);
            if (!tensor.isInitialized() ||
                tensor.getPlacement().getMemDevice() != TensorPlacement::MemDevices::CPU ||
                tensor.getDescriptor() != expected) {
                throw std::runtime_error(
                    "DeviceResidentFileNamedBatchSession returned direct field '" +
                    spec.name + "' has the wrong placement or descriptor.");
            }
            continue;
        }
        if (!batch.isDeviceBatchReference(spec.name)) {
            throw std::runtime_error(
                "DeviceResidentFileNamedBatchSession returned resident field '" +
                spec.name + "' is not a device reference.");
        }
        const Thor::DeviceBatchReference &reference = batch.getDeviceBatchReference(spec.name);
        if (!reference.isInitialized() || reference.getBatchCapacity() != batchSize ||
            reference.getOutputPlacement() != residentDataset->getPlacement() ||
            reference.getOutputDescriptor() != expected) {
            throw std::runtime_error(
                "DeviceResidentFileNamedBatchSession returned resident field '" +
                spec.name + "' has the wrong reference metadata.");
        }
    }

    for (const std::string &fieldName : raggedFieldNames) {
        if (!batch.contains(fieldName) || !batch.isRaggedTensor(fieldName) ||
            !batch.getSourceReference(fieldName).has_value()) {
            throw std::runtime_error(
                "DeviceResidentFileNamedBatchSession returned ragged field '" +
                fieldName + "' with the wrong value/source kind.");
        }
        const Thor::DatasetField &field = datasetDescription.schema.getField(fieldName);
        const RaggedTensorDescriptor &expected =
            fieldRequirements.at(field.id).raggedTensorDescriptor.value();
        const RaggedTensor &ragged = batch.getRaggedTensor(fieldName);
        if (!ragged.isInitialized() || ragged.getPlacement() != residentDataset->getPlacement() ||
            ragged.getDescriptor() != expected) {
            throw std::runtime_error(
                "DeviceResidentFileNamedBatchSession returned ragged field '" +
                fieldName + "' with the wrong placement or descriptor.");
        }
    }
}

void DeviceResidentFileNamedBatchSession::releaseSelectionSlot(
    ExampleType exampleType,
    std::shared_ptr<DeviceResidentFileSelectionSlot> selectionSlot,
    std::vector<Event> consumedEvents) noexcept {
    if (selectionSlot == nullptr) return;
    auto found = splitRuntimes.find(exampleType);
    if (found == splitRuntimes.end() || found->second == nullptr) return;
    SplitRuntime &runtime = *found->second;
    try {
        std::lock_guard<std::mutex> guard(runtime.mutex);
        if (consumedEvents.empty()) runtime.availableSelections.push_back(std::move(selectionSlot));
        else runtime.pendingSelections.push_back(
            DeviceResidentFilePendingSelection{std::move(selectionSlot), std::move(consumedEvents)});
    } catch (...) { return; }
    runtime.notEmpty.notify_one();
}

void DeviceResidentFileNamedBatchSession::releaseDirectSlot(
    ExampleType exampleType,
    std::shared_ptr<DeviceResidentFileDirectSlot> directSlot,
    std::vector<Event> consumedEvents) noexcept {
    if (directSlot == nullptr) return;
    auto found = splitRuntimes.find(exampleType);
    if (found == splitRuntimes.end() || found->second == nullptr) return;
    SplitRuntime &runtime = *found->second;
    try {
        std::lock_guard<std::mutex> guard(runtime.mutex);
        if (consumedEvents.empty()) runtime.availableDirectSlots.push_back(std::move(directSlot));
        else runtime.pendingDirectSlots.push_back(
            DeviceResidentFilePendingDirect{std::move(directSlot), std::move(consumedEvents)});
    } catch (...) { return; }
    runtime.notEmpty.notify_one();
}

void DeviceResidentFileNamedBatchSession::releaseRaggedSlot(
    ExampleType exampleType,
    std::shared_ptr<DeviceResidentFileRaggedSlot> raggedSlot,
    std::vector<Event> consumedEvents) noexcept {
    if (raggedSlot == nullptr) return;
    auto found = splitRuntimes.find(exampleType);
    if (found == splitRuntimes.end() || found->second == nullptr) return;
    SplitRuntime &runtime = *found->second;
    try {
        std::lock_guard<std::mutex> guard(runtime.mutex);
        if (consumedEvents.empty()) runtime.availableRaggedSlots.push_back(std::move(raggedSlot));
        else runtime.pendingRaggedSlots.push_back(
            DeviceResidentFilePendingRagged{std::move(raggedSlot), std::move(consumedEvents)});
    } catch (...) { return; }
    runtime.notEmpty.notify_one();
}

void DeviceResidentFileNamedBatchSession::setBatchTailModeForRuntimeImpl(
    ThorImplementation::BatchTailMode mode) {
    (void)mode;
    for (const auto &[exampleType, runtime] : splitRuntimes) {
        (void)exampleType;
        THOR_THROW_IF_FALSE(runtime != nullptr);
        std::lock_guard<std::mutex> lock(runtime->mutex);
        THOR_THROW_IF_FALSE(runtime->nextBatchNum == 0);
        THOR_THROW_IF_FALSE(runtime->pendingSelections.empty());
        THOR_THROW_IF_FALSE(runtime->pendingDirectSlots.empty());
        THOR_THROW_IF_FALSE(runtime->pendingRaggedSlots.empty());
    }
}

void DeviceResidentFileNamedBatchSession::recycleBatch(
    ExampleType exampleType,
    Batch &&batch) {
    (void)exampleType;
    if (!cancelled.load(std::memory_order_acquire)) validateReturnedBatch(batch);
    batch.clear();
}

uint64_t DeviceResidentFileNamedBatchSession::getNumBatchesPerEpoch(ExampleType exampleType) {
    return runtimeFor(exampleType).batchesPerEpoch;
}

uint64_t DeviceResidentFileNamedBatchSession::getNumExamples(ExampleType exampleType) {
    return runtimeFor(exampleType).numExamples();
}

uint64_t DeviceResidentFileNamedBatchSession::getNextBatchNum(ExampleType exampleType) {
    SplitRuntime &runtime = runtimeFor(exampleType);
    std::lock_guard<std::mutex> guard(runtime.mutex);
    return runtime.numExamples() == 0 ? 0 : runtime.nextBatchNum;
}

std::optional<TensorPlacement>
DeviceResidentFileNamedBatchSession::getBatchTensorPlacement(const std::string &tensorName) const {
    if (raggedFieldNames.find(tensorName) != raggedFieldNames.end()) {
        return residentDataset->getPlacement();
    }
    const std::optional<BatchFieldSpec> spec = findBatchFieldSpec(datasetDescription.layout, tensorName);
    if (!spec.has_value() ||
        residentReferenceFieldNames.find(tensorName) != residentReferenceFieldNames.end()) {
        return std::nullopt;
    }
    return TensorPlacement(TensorPlacement::MemDevices::CPU);
}

Thor::BatchFieldSourceDescription
DeviceResidentFileNamedBatchSession::getBatchFieldSourceDescription(
    const std::string &fieldName) const {
    if (raggedFieldNames.find(fieldName) != raggedFieldNames.end()) {
        return Thor::BatchFieldSourceDescription::materialized(residentDataset->getPlacement());
    }
    const std::optional<BatchFieldSpec> spec = findBatchFieldSpec(datasetDescription.layout, fieldName);
    if (!spec.has_value()) return Thor::BatchFieldSourceDescription::materialized();
    if (residentReferenceFieldNames.find(fieldName) != residentReferenceFieldNames.end()) {
        return Thor::BatchFieldSourceDescription::deviceReference(residentDataset->getPlacement());
    }
    return Thor::BatchFieldSourceDescription::materialized(
        TensorPlacement(TensorPlacement::MemDevices::CPU));
}

std::vector<Event> DeviceResidentFileNamedBatchSession::getSynchronizeEvents() const {
    std::vector<Event> events;
    for (const auto &entry : splitRuntimes) {
        if (entry.second == nullptr || entry.second->numExamples() == 0) continue;
        std::lock_guard<std::mutex> guard(entry.second->mutex);
        events.push_back(entry.second->selectionUploadStream.putEvent(false, true));
        events.push_back(entry.second->raggedGatherStream.putEvent(false, true));
        for (const DeviceResidentFilePendingSelection &pending : entry.second->pendingSelections) {
            events.insert(events.end(), pending.consumedEvents.begin(), pending.consumedEvents.end());
        }
        for (const DeviceResidentFilePendingDirect &pending : entry.second->pendingDirectSlots) {
            events.insert(events.end(), pending.consumedEvents.begin(), pending.consumedEvents.end());
        }
        for (const DeviceResidentFilePendingRagged &pending : entry.second->pendingRaggedSlots) {
            events.insert(events.end(), pending.consumedEvents.begin(), pending.consumedEvents.end());
        }
    }
    return events;
}
