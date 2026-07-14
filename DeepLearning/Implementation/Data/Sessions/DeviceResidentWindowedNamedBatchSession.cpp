#include "DeepLearning/Implementation/Data/Sessions/DeviceResidentWindowedNamedBatchSession.h"

#include "DeepLearning/Implementation/ThorError.h"

#include <stdexcept>
#include <utility>

using ThorImplementation::DataType;
using ThorImplementation::Tensor;
using ThorImplementation::TensorDescriptor;
using ThorImplementation::TensorPlacement;

struct DeviceResidentWindowedSelectionState {
    uint64_t slotIndex = 0;
    Tensor rowIndicesHost;
    Tensor rowIndicesDevice;
    Event rowsReadyEvent;
};

struct DeviceResidentWindowedSelectionSlot {
    std::shared_ptr<DeviceResidentWindowedSelectionState> state;
};

struct DeviceResidentWindowedDirectSlot {
    uint64_t slotIndex = 0;
    std::map<std::string, Tensor> directTensors;
};

struct DeviceResidentWindowedPendingSelection {
    std::shared_ptr<DeviceResidentWindowedSelectionSlot> slot;
    std::vector<Event> consumedEvents;
};

struct DeviceResidentWindowedPendingDirect {
    std::shared_ptr<DeviceResidentWindowedDirectSlot> slot;
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
        std::shared_ptr<const DeviceResidentWindowedSelectionState> selection,
        TensorDescriptor outputDescriptor)
        : dataset(std::move(dataset)),
          fieldName(std::move(fieldName)),
          selection(std::move(selection)),
          outputDescriptor(std::move(outputDescriptor)) {
        THOR_THROW_IF_FALSE(this->dataset != nullptr);
        THOR_THROW_IF_FALSE(this->dataset->hasCompactField(this->fieldName));
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
    std::shared_ptr<const DeviceResidentWindowedSelectionState> selection;
    TensorDescriptor outputDescriptor;
};

uint64_t batchesFor(uint64_t numExamples, uint64_t batchSize) {
    THOR_THROW_IF_FALSE(batchSize > 0);
    return (numExamples / batchSize) + ((numExamples % batchSize) == 0 ? 0 : 1);
}

const char *splitNameFor(ExampleType exampleType) {
    if (exampleType == ExampleType::TRAIN) {
        return "train";
    }
    if (exampleType == ExampleType::VALIDATE) {
        return "validate";
    }
    if (exampleType == ExampleType::TEST) {
        return "test";
    }
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
        specs.push_back(BatchFieldSpec{
            spec.name,
            spec.dataType,
            spec.dimensions,
            false});
    }
    for (const DatasetLayout::WindowedTensorSpec &spec : layout.windowedTensors()) {
        specs.push_back(BatchFieldSpec{
            spec.name,
            spec.dataType,
            spec.dimensions,
            true});
        if (spec.maskName.has_value()) {
            specs.push_back(BatchFieldSpec{
                spec.maskName.value(),
                DataType::UINT8,
                std::vector<uint64_t>{spec.windowLength()},
                true});
        }
    }
    return specs;
}

std::optional<BatchFieldSpec> findBatchFieldSpec(
    const DatasetLayout &layout,
    const std::string &name) {
    for (const DatasetLayout::TensorSpec &spec : layout.tensors()) {
        if (spec.name == name) {
            return BatchFieldSpec{spec.name, spec.dataType, spec.dimensions, false};
        }
    }
    for (const DatasetLayout::WindowedTensorSpec &spec : layout.windowedTensors()) {
        if (spec.name == name) {
            return BatchFieldSpec{spec.name, spec.dataType, spec.dimensions, true};
        }
        if (spec.maskName.has_value() && spec.maskName.value() == name) {
            return BatchFieldSpec{
                spec.maskName.value(),
                DataType::UINT8,
                std::vector<uint64_t>{spec.windowLength()},
                true};
        }
    }
    return std::nullopt;
}

}  // namespace

DeviceResidentWindowedNamedBatchSession::DeviceResidentWindowedNamedBatchSession(
    Thor::DatasetMaterializationDescription datasetDescription,
    Thor::DeviceDatasetSessionDescription sessionDescription,
    Thor::DeviceDatasetLease windowedDataset,
    uint64_t batchQueueDepth,
    uint64_t readerQueueDepth,
    std::string datasetName)
    : Thor::BatchSession(std::move(datasetName)),
      datasetDescription(std::move(datasetDescription)),
      sessionDescription(std::move(sessionDescription)),
      windowedDataset(std::move(windowedDataset)),
      batchQueueDepth(batchQueueDepth),
      readerQueueDepth(readerQueueDepth) {
    if (!this->windowedDataset) {
        throw std::runtime_error(
            "DeviceResidentWindowedNamedBatchSession requires a windowed device dataset.");
    }
    if (!this->datasetDescription.layout.hasWindowedTensors()) {
        throw std::runtime_error(
            "DeviceResidentWindowedNamedBatchSession requires at least one windowed tensor.");
    }
    if (!this->windowedDataset->usesCompactFileStorage()) {
        throw std::runtime_error(
            "DeviceResidentWindowedNamedBatchSession requires compact file device storage.");
    }
    if (batchQueueDepth == 0) {
        throw std::runtime_error(
            "DeviceResidentWindowedNamedBatchSession batch_queue_depth must be >= 1.");
    }
    if (this->windowedDataset->getDatasetId() != this->datasetDescription.datasetId ||
        this->windowedDataset->getNumExamples() != this->datasetDescription.numExamples) {
        throw std::runtime_error(
            "DeviceResidentWindowedNamedBatchSession resident dataset does not match source dataset description.");
    }
    const Thor::DatasetSplitManifest &splits = this->sessionDescription.getSplits();
    if (splits.getDatasetId() != this->datasetDescription.datasetId ||
        splits.getNumExamples() != this->datasetDescription.numExamples) {
        throw std::runtime_error(
            "DeviceResidentWindowedNamedBatchSession split manifest does not match source dataset.");
    }

    requiredFieldIds = this->sessionDescription.getRequiredFieldIds();
    if (requiredFieldIds.empty()) {
        for (const Thor::DatasetField &field : this->datasetDescription.schema.getFields()) {
            requiredFieldIds.insert(field.id);
        }
    }
    for (Thor::DatasetFieldId fieldId : requiredFieldIds) {
        (void)this->datasetDescription.schema.getField(fieldId);
    }

    this->batchSize = this->sessionDescription.getBatching().getBatchSize();
    uint64_t residentDirectFieldCount = 0;
    for (const BatchFieldSpec &spec : batchFieldSpecsFor(this->datasetDescription.layout)) {
        if (this->windowedDataset->hasCompactField(spec.name)) {
            residentFieldNames.insert(spec.name);
            if (!spec.windowed) {
                residentDirectFieldCount += 1;
            }
        } else if (spec.windowed) {
            throw std::runtime_error(
                "DeviceResidentWindowedNamedBatchSession compact resident dataset is missing window field '" +
                spec.name + "'.");
        } else {
            directFieldNames.insert(spec.name);
        }
    }
    if (residentDirectFieldCount != 0 && !directFieldNames.empty()) {
        throw std::runtime_error(
            "DeviceResidentWindowedNamedBatchSession requires direct fields to be either all compact-resident or all CPU-backed.");
    }
    if (!directFieldNames.empty()) {
        if (readerQueueDepth == 0) {
            throw std::runtime_error(
                "DeviceResidentWindowedNamedBatchSession reader_queue_depth must be >= 1 when CPU direct fields are used.");
        }
        this->reader = IndexedDatasetReader::openDataset(
            this->datasetDescription.datasetPath,
            this->datasetDescription.layout);
        if (this->reader->getNumExamples() != this->datasetDescription.numExamples) {
            throw std::runtime_error(
                "DeviceResidentWindowedNamedBatchSession source dataset row count changed.");
        }
    }

    initializeSplit(
        ExampleType::TRAIN,
        splits.getSharedTrain(),
        this->sessionDescription.getBatching().getRandomizeTrain(),
        this->sessionDescription.getBatching().getRandomSeed());
    initializeSplit(
        ExampleType::VALIDATE,
        splits.getSharedValidate(),
        false,
        std::nullopt);
    initializeSplit(
        ExampleType::TEST,
        splits.getSharedTest(),
        false,
        std::nullopt);
}

DeviceResidentWindowedNamedBatchSession::~DeviceResidentWindowedNamedBatchSession() {
    cancel();
    // Source owners transfer slots to pending-reuse queues before releasing
    // the session's final shared ownership. Keep their tensors alive until all
    // source-consumed events have completed, even during manual-session teardown.
    for (auto &entry : splitRuntimes) {
        if (entry.second == nullptr) {
            continue;
        }
        try {
            std::lock_guard<std::mutex> guard(entry.second->mutex);
            for (DeviceResidentWindowedPendingSelection &pending :
                 entry.second->pendingSelections) {
                for (Event &event : pending.consumedEvents) {
                    event.synchronize();
                }
            }
            for (DeviceResidentWindowedPendingDirect &pending :
                 entry.second->pendingDirectSlots) {
                for (Event &event : pending.consumedEvents) {
                    event.synchronize();
                }
            }
        } catch (...) {
            // Destructors must not throw while a CUDA context is already failing.
        }
    }
}

void DeviceResidentWindowedNamedBatchSession::cancel() {
    if (cancelled.exchange(true, std::memory_order_acq_rel)) {
        return;
    }
    for (auto &entry : splitRuntimes) {
        if (entry.second != nullptr) {
            entry.second->notEmpty.notify_all();
        }
    }
}

void DeviceResidentWindowedNamedBatchSession::initializeSplit(
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
    runtime->selectionUploadStream = Stream(windowedDataset->getPlacement());

    if (runtime->numExamples() != 0) {
        if (reader != nullptr) {
            runtime->readerSession = reader->createSession(readerQueueDepth);
        }
        if (runtime->randomized) {
            runtime->randomizer =
                std::make_unique<FullPeriodRandom>(runtime->numExamples(), false);
            if (runtime->seed.has_value()) {
                runtime->randomizer->reseed(runtime->seed.value());
            }
        }
        for (uint64_t i = 0; i < batchQueueDepth; ++i) {
            runtime->availableSelections.push_back(allocateSelectionSlot(i));
            if (!directFieldNames.empty()) {
                runtime->availableDirectSlots.push_back(allocateDirectSlot(i));
            }
        }
    }

    auto [it, inserted] = splitRuntimes.emplace(exampleType, std::move(runtime));
    THOR_THROW_IF_FALSE(inserted);
    (void)it;
}

std::map<std::string, Tensor>
DeviceResidentWindowedNamedBatchSession::allocateDirectTensorSet() const {
    std::map<std::string, Tensor> tensors;
    const TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    for (const BatchFieldSpec &spec : batchFieldSpecsFor(datasetDescription.layout)) {
        if (directFieldNames.find(spec.name) == directFieldNames.end()) {
            continue;
        }
        tensors.emplace(
            spec.name,
            Tensor(
                cpuPlacement,
                TensorDescriptor(
                    spec.dataType,
                    batchDimensionsFor(spec.exampleDimensions, batchSize))));
    }
    return tensors;
}

std::shared_ptr<DeviceResidentWindowedSelectionSlot>
DeviceResidentWindowedNamedBatchSession::allocateSelectionSlot(uint64_t slotIndex) const {
    auto slot = std::make_shared<DeviceResidentWindowedSelectionSlot>();
    slot->state = std::make_shared<DeviceResidentWindowedSelectionState>();
    slot->state->slotIndex = slotIndex;
    slot->state->rowIndicesHost = Tensor(
        TensorPlacement(TensorPlacement::MemDevices::CPU),
        TensorDescriptor(DataType::UINT64, {batchSize}));
    slot->state->rowIndicesDevice = Tensor(
        windowedDataset->getPlacement(),
        TensorDescriptor(DataType::UINT64, {batchSize}));
    return slot;
}

std::shared_ptr<DeviceResidentWindowedDirectSlot>
DeviceResidentWindowedNamedBatchSession::allocateDirectSlot(uint64_t slotIndex) const {
    auto slot = std::make_shared<DeviceResidentWindowedDirectSlot>();
    slot->slotIndex = slotIndex;
    slot->directTensors = allocateDirectTensorSet();
    return slot;
}

DeviceResidentWindowedNamedBatchSession::SplitRuntime &
DeviceResidentWindowedNamedBatchSession::runtimeFor(ExampleType exampleType) {
    const auto found = splitRuntimes.find(exampleType);
    if (found == splitRuntimes.end() || found->second == nullptr) {
        throw std::runtime_error(
            "DeviceResidentWindowedNamedBatchSession does not contain requested split.");
    }
    return *found->second;
}

const DeviceResidentWindowedNamedBatchSession::SplitRuntime &
DeviceResidentWindowedNamedBatchSession::runtimeFor(ExampleType exampleType) const {
    const auto found = splitRuntimes.find(exampleType);
    if (found == splitRuntimes.end() || found->second == nullptr) {
        throw std::runtime_error(
            "DeviceResidentWindowedNamedBatchSession does not contain requested split.");
    }
    return *found->second;
}

void DeviceResidentWindowedNamedBatchSession::fillRowIndexTensor(
    SplitRuntime &runtime,
    DeviceResidentWindowedSelectionSlot &selectionSlot) {
    THOR_THROW_IF_FALSE(runtime.numExamples() > 0);
    THOR_THROW_IF_FALSE(selectionSlot.state != nullptr);
    uint64_t *rowIndices = selectionSlot.state->rowIndicesHost.getMemPtr<uint64_t>();
    for (uint64_t slot = 0; slot < batchSize; ++slot) {
        uint64_t logicalPosition = 0;
        if (runtime.randomized) {
            THOR_THROW_IF_FALSE(runtime.randomizer != nullptr);
            logicalPosition = runtime.randomizer->getRandomNumber();
        } else {
            logicalPosition = runtime.nextLogicalPosition;
            runtime.nextLogicalPosition =
                (runtime.nextLogicalPosition + 1) % runtime.numExamples();
        }
        THOR_THROW_IF_FALSE(logicalPosition < runtime.numExamples());
        const uint64_t sourceRow = runtime.sourceIndices->at(logicalPosition);
        THOR_THROW_IF_FALSE(sourceRow < datasetDescription.numExamples);
        rowIndices[slot] = sourceRow;
    }
}

Batch DeviceResidentWindowedNamedBatchSession::acquireBatch(
    ExampleType exampleType,
    uint64_t &batchNum) {
    if (cancelled.load(std::memory_order_acquire)) {
        throw std::runtime_error("DeviceResidentWindowedNamedBatchSession has been cancelled.");
    }
    SplitRuntime &runtime = runtimeFor(exampleType);
    std::unique_lock<std::mutex> lock(runtime.mutex);
    if (runtime.numExamples() == 0) {
        throw std::runtime_error(
            "DeviceResidentWindowedNamedBatchSession cannot get a batch from an empty split.");
    }
    runtime.notEmpty.wait(lock, [&] {
        const bool selectionReady =
            !runtime.availableSelections.empty() || !runtime.pendingSelections.empty();
        const bool directReady = directFieldNames.empty() ||
                                 !runtime.availableDirectSlots.empty() ||
                                 !runtime.pendingDirectSlots.empty();
        return cancelled.load(std::memory_order_acquire) ||
               (selectionReady && directReady);
    });
    if (cancelled.load(std::memory_order_acquire)) {
        throw std::runtime_error("DeviceResidentWindowedNamedBatchSession has been cancelled.");
    }

    std::shared_ptr<DeviceResidentWindowedSelectionSlot> selectionSlot;
    if (!runtime.availableSelections.empty()) {
        selectionSlot = std::move(runtime.availableSelections.front());
        runtime.availableSelections.pop_front();
    } else {
        THOR_THROW_IF_FALSE(!runtime.pendingSelections.empty());
        DeviceResidentWindowedPendingSelection pending =
            std::move(runtime.pendingSelections.front());
        runtime.pendingSelections.pop_front();
        lock.unlock();
        for (Event &event : pending.consumedEvents) {
            event.synchronize();
        }
        lock.lock();
        selectionSlot = std::move(pending.slot);
    }
    THOR_THROW_IF_FALSE(selectionSlot != nullptr);
    THOR_THROW_IF_FALSE(selectionSlot->state != nullptr);

    std::shared_ptr<DeviceResidentWindowedDirectSlot> directSlot;
    if (!directFieldNames.empty()) {
        if (!runtime.availableDirectSlots.empty()) {
            directSlot = std::move(runtime.availableDirectSlots.front());
            runtime.availableDirectSlots.pop_front();
        } else {
            THOR_THROW_IF_FALSE(!runtime.pendingDirectSlots.empty());
            DeviceResidentWindowedPendingDirect pending =
                std::move(runtime.pendingDirectSlots.front());
            runtime.pendingDirectSlots.pop_front();
            lock.unlock();
            for (Event &event : pending.consumedEvents) {
                event.synchronize();
            }
            lock.lock();
            directSlot = std::move(pending.slot);
        }
        THOR_THROW_IF_FALSE(directSlot != nullptr);
    }

    if (cancelled.load(std::memory_order_acquire)) {
        runtime.availableSelections.push_front(std::move(selectionSlot));
        if (directSlot != nullptr) {
            runtime.availableDirectSlots.push_front(std::move(directSlot));
        }
        throw std::runtime_error("DeviceResidentWindowedNamedBatchSession has been cancelled.");
    }

    batchNum = runtime.nextBatchNum;
    runtime.nextBatchNum =
        (runtime.nextBatchNum + 1) % runtime.batchesPerEpoch;
    fillRowIndexTensor(runtime, *selectionSlot);
    lock.unlock();

    bool selectionManagedByOwner = false;
    bool directManagedByOwner = false;
    try {
        if (directSlot != nullptr) {
            std::lock_guard<std::mutex> readerGuard(runtime.readerMutex);
            std::vector<uint8_t *> directPointers(reader->getTensorCount(), nullptr);
            for (const DatasetLayout::TensorSpec &spec : datasetDescription.layout.tensors()) {
                if (directFieldNames.find(spec.name) == directFieldNames.end()) {
                    continue;
                }
                const uint64_t ordinal = reader->getLayoutTensorOrdinal(spec.name);
                directPointers.at(static_cast<size_t>(ordinal)) =
                    static_cast<uint8_t *>(directSlot->directTensors.at(spec.name).getMemPtr());
            }
            const uint64_t *sourceRows =
                selectionSlot->state->rowIndicesHost.getMemPtr<uint64_t>();
            THOR_THROW_IF_FALSE(runtime.readerSession != nullptr);
            for (uint64_t slot = 0; slot < batchSize; ++slot) {
                runtime.readerSession->loadDirectExampleInto(
                    sourceRows[slot],
                    slot,
                    directPointers);
            }
            runtime.readerSession->drain();
        }

        selectionSlot->state->rowIndicesDevice.copyFromAsync(
            selectionSlot->state->rowIndicesHost,
            runtime.selectionUploadStream);
        runtime.selectionUploadStream.putEvent(selectionSlot->state->rowsReadyEvent);

        Batch batch;
        for (const BatchFieldSpec &spec : batchFieldSpecsFor(datasetDescription.layout)) {
            const TensorDescriptor outputDescriptor(
                spec.dataType,
                batchDimensionsFor(spec.exampleDimensions, batchSize));
            if (residentFieldNames.find(spec.name) == residentFieldNames.end()) {
                THOR_THROW_IF_FALSE(directSlot != nullptr);
                batch.insert(spec.name, directSlot->directTensors.at(spec.name));
                continue;
            }

            std::shared_ptr<const Thor::DeviceBatchMaterializer> materializer =
                std::make_shared<CompactResidentFieldMaterializer>(
                    windowedDataset.getShared(),
                    spec.name,
                    selectionSlot->state,
                    outputDescriptor);
            batch.insert(
                spec.name,
                Thor::DeviceBatchReference(
                    std::move(materializer),
                    static_cast<uint32_t>(batchSize)));
        }

        std::shared_ptr<DeviceResidentWindowedNamedBatchSession> sharedSelf =
            std::dynamic_pointer_cast<DeviceResidentWindowedNamedBatchSession>(shared_from_this());
        THOR_THROW_IF_FALSE(sharedSelf != nullptr);
        std::weak_ptr<DeviceResidentWindowedNamedBatchSession> weakSelf = sharedSelf;

        if (directSlot != nullptr) {
            Thor::BatchSourceOwner directOwner(
                [weakSelf, exampleType, directSlot](std::vector<Event> consumedEvents) mutable {
                    if (std::shared_ptr<DeviceResidentWindowedNamedBatchSession> session = weakSelf.lock()) {
                        session->releaseDirectSlot(
                            exampleType,
                            std::move(directSlot),
                            std::move(consumedEvents));
                    }
                });
            directManagedByOwner = true;
            addBatchSourceResource(
                batch,
                directFieldNames,
                std::move(directOwner));
        }

        Thor::BatchSourceOwner selectionOwner(
            [weakSelf, exampleType, selectionSlot](std::vector<Event> consumedEvents) mutable {
                // A materialization-consumed event is transitively ordered after
                // rowsReadyEvent because every resident materializer waits on it.
                // When no resident input was submitted, retain the upload event
                // itself so the host/device row buffers still cannot be overwritten.
                if (consumedEvents.empty()) {
                    consumedEvents.push_back(selectionSlot->state->rowsReadyEvent);
                }
                if (std::shared_ptr<DeviceResidentWindowedNamedBatchSession> session = weakSelf.lock()) {
                    session->releaseSelectionSlot(
                        exampleType,
                        std::move(selectionSlot),
                        std::move(consumedEvents));
                }
            });
        selectionManagedByOwner = true;
        addBatchSourceResource(
            batch,
            residentFieldNames,
            std::move(selectionOwner));

        validateReturnedBatch(batch);
        return batch;
    } catch (...) {
        if (!selectionManagedByOwner) {
            releaseSelectionSlot(exampleType, std::move(selectionSlot), {});
        }
        if (directSlot != nullptr && !directManagedByOwner) {
            releaseDirectSlot(exampleType, std::move(directSlot), {});
        }
        throw;
    }
}

void DeviceResidentWindowedNamedBatchSession::validateReturnedBatch(
    const Batch &batch) const {
    const std::vector<BatchFieldSpec> specs = batchFieldSpecsFor(datasetDescription.layout);
    if (batch.size() != specs.size()) {
        throw std::runtime_error(
            "DeviceResidentWindowedNamedBatchSession returned batch has wrong field count.");
    }
    for (const BatchFieldSpec &spec : specs) {
        if (!batch.contains(spec.name)) {
            throw std::runtime_error(
                "DeviceResidentWindowedNamedBatchSession returned batch is missing field '" +
                spec.name + "'.");
        }
        if (!batch.getSourceReference(spec.name).has_value()) {
            throw std::runtime_error(
                "DeviceResidentWindowedNamedBatchSession returned field '" +
                spec.name + "' is missing its source-resource reference.");
        }
        const TensorDescriptor expected(
            spec.dataType,
            batchDimensionsFor(spec.exampleDimensions, batchSize));
        if (residentFieldNames.find(spec.name) == residentFieldNames.end()) {
            if (!batch.isTensor(spec.name)) {
                throw std::runtime_error(
                    "DeviceResidentWindowedNamedBatchSession returned direct field '" +
                    spec.name + "' is not a tensor.");
            }
            const Tensor &tensor = batch.getTensor(spec.name);
            if (!tensor.isInitialized() ||
                tensor.getPlacement().getMemDevice() != TensorPlacement::MemDevices::CPU ||
                tensor.getDescriptor() != expected) {
                throw std::runtime_error(
                    "DeviceResidentWindowedNamedBatchSession returned direct field '" +
                    spec.name + "' has the wrong placement or descriptor.");
            }
            continue;
        }

        if (!batch.isDeviceBatchReference(spec.name)) {
            throw std::runtime_error(
                "DeviceResidentWindowedNamedBatchSession returned resident field '" +
                spec.name + "' is not a device reference.");
        }
        const Thor::DeviceBatchReference &reference =
            batch.getDeviceBatchReference(spec.name);
        if (!reference.isInitialized() ||
            reference.getBatchSize() != batchSize ||
            reference.getOutputPlacement() != windowedDataset->getPlacement() ||
            reference.getOutputDescriptor() != expected) {
            throw std::runtime_error(
                "DeviceResidentWindowedNamedBatchSession returned resident field '" +
                spec.name + "' has the wrong reference metadata.");
        }
    }
}

void DeviceResidentWindowedNamedBatchSession::releaseSelectionSlot(
    ExampleType exampleType,
    std::shared_ptr<DeviceResidentWindowedSelectionSlot> selectionSlot,
    std::vector<Event> consumedEvents) noexcept {
    if (selectionSlot == nullptr) {
        return;
    }
    auto found = splitRuntimes.find(exampleType);
    if (found == splitRuntimes.end() || found->second == nullptr) {
        return;
    }
    SplitRuntime &runtime = *found->second;
    try {
        std::lock_guard<std::mutex> guard(runtime.mutex);
        if (consumedEvents.empty()) {
            runtime.availableSelections.push_back(std::move(selectionSlot));
        } else {
            runtime.pendingSelections.push_back(
                DeviceResidentWindowedPendingSelection{
                    std::move(selectionSlot),
                    std::move(consumedEvents)});
        }
    } catch (...) {
        return;
    }
    runtime.notEmpty.notify_one();
}

void DeviceResidentWindowedNamedBatchSession::releaseDirectSlot(
    ExampleType exampleType,
    std::shared_ptr<DeviceResidentWindowedDirectSlot> directSlot,
    std::vector<Event> consumedEvents) noexcept {
    if (directSlot == nullptr) {
        return;
    }
    auto found = splitRuntimes.find(exampleType);
    if (found == splitRuntimes.end() || found->second == nullptr) {
        return;
    }
    SplitRuntime &runtime = *found->second;
    try {
        std::lock_guard<std::mutex> guard(runtime.mutex);
        if (consumedEvents.empty()) {
            runtime.availableDirectSlots.push_back(std::move(directSlot));
        } else {
            runtime.pendingDirectSlots.push_back(
                DeviceResidentWindowedPendingDirect{
                    std::move(directSlot),
                    std::move(consumedEvents)});
        }
    } catch (...) {
        return;
    }
    runtime.notEmpty.notify_one();
}

void DeviceResidentWindowedNamedBatchSession::recycleBatch(
    ExampleType exampleType,
    Batch &&batch) {
    (void)exampleType;
    if (!cancelled.load(std::memory_order_acquire)) {
        validateReturnedBatch(batch);
    }
    // Any resource retained for a late host-side observer (such as an input
    // scalar training statistic), or any resource from a caller that did not
    // explicitly release after submission, is safely sealed here.
    batch.clear();
}

uint64_t DeviceResidentWindowedNamedBatchSession::getNumBatchesPerEpoch(
    ExampleType exampleType) {
    return runtimeFor(exampleType).batchesPerEpoch;
}

uint64_t DeviceResidentWindowedNamedBatchSession::getNumExamples(
    ExampleType exampleType) {
    return runtimeFor(exampleType).numExamples();
}

uint64_t DeviceResidentWindowedNamedBatchSession::getNextBatchNum(
    ExampleType exampleType) {
    SplitRuntime &runtime = runtimeFor(exampleType);
    std::lock_guard<std::mutex> guard(runtime.mutex);
    return runtime.numExamples() == 0 ? 0 : runtime.nextBatchNum;
}

std::optional<TensorPlacement>
DeviceResidentWindowedNamedBatchSession::getBatchTensorPlacement(
    const std::string &tensorName) const {
    const std::optional<BatchFieldSpec> spec =
        findBatchFieldSpec(datasetDescription.layout, tensorName);
    if (!spec.has_value() || residentFieldNames.find(tensorName) != residentFieldNames.end()) {
        return std::nullopt;
    }
    return TensorPlacement(TensorPlacement::MemDevices::CPU);
}

Thor::BatchFieldSourceDescription
DeviceResidentWindowedNamedBatchSession::getBatchFieldSourceDescription(
    const std::string &fieldName) const {
    const std::optional<BatchFieldSpec> spec =
        findBatchFieldSpec(datasetDescription.layout, fieldName);
    if (!spec.has_value()) {
        return Thor::BatchFieldSourceDescription::materialized();
    }
    if (residentFieldNames.find(fieldName) != residentFieldNames.end()) {
        return Thor::BatchFieldSourceDescription::deviceReference(
            windowedDataset->getPlacement());
    }
    return Thor::BatchFieldSourceDescription::materialized(
        TensorPlacement(TensorPlacement::MemDevices::CPU));
}

std::vector<Event> DeviceResidentWindowedNamedBatchSession::getSynchronizeEvents() const {
    std::vector<Event> events;
    for (const auto &entry : splitRuntimes) {
        if (entry.second == nullptr || entry.second->numExamples() == 0) {
            continue;
        }
        std::lock_guard<std::mutex> guard(entry.second->mutex);
        events.push_back(entry.second->selectionUploadStream.putEvent(false, true));
        for (const DeviceResidentWindowedPendingSelection &pending :
             entry.second->pendingSelections) {
            events.insert(
                events.end(),
                pending.consumedEvents.begin(),
                pending.consumedEvents.end());
        }
        for (const DeviceResidentWindowedPendingDirect &pending :
             entry.second->pendingDirectSlots) {
            events.insert(
                events.end(),
                pending.consumedEvents.begin(),
                pending.consumedEvents.end());
        }
    }
    return events;
}
