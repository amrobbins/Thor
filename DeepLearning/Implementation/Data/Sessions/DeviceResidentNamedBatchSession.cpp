#include "DeepLearning/Implementation/Data/Sessions/DeviceResidentNamedBatchSession.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Data/Residency/DeviceResidentNamedGatherKernel.h"

#include <stdexcept>
#include <utility>

using ThorImplementation::DataType;
using ThorImplementation::Tensor;
using ThorImplementation::TensorDescriptor;
using ThorImplementation::TensorPlacement;

namespace {

struct BatchTensorSpec {
    std::string name;
    DataType dataType = DataType::FP32;
    std::vector<uint64_t> exampleDimensions;
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

std::vector<BatchTensorSpec> batchTensorSpecsFor(const DatasetLayout &layout) {
    std::vector<BatchTensorSpec> specs;
    specs.reserve(layout.tensors().size() + layout.windowedTensors().size() * 2);
    for (const DatasetLayout::TensorSpec &spec : layout.tensors()) {
        specs.push_back(BatchTensorSpec{spec.name, spec.dataType, spec.dimensions});
    }
    for (const DatasetLayout::WindowedTensorSpec &spec : layout.windowedTensors()) {
        specs.push_back(BatchTensorSpec{spec.name, spec.dataType, spec.dimensions});
        if (spec.maskName.has_value()) {
            specs.push_back(BatchTensorSpec{
                spec.maskName.value(),
                DataType::UINT8,
                std::vector<uint64_t>{spec.windowLength()}});
        }
    }
    return specs;
}

}  // namespace

DeviceResidentNamedBatchSession::DeviceResidentNamedBatchSession(
    Thor::DeviceDatasetLease dataset,
    Thor::DatasetSplitManifest splits,
    Thor::BatchPolicy batching,
    uint64_t batchQueueDepth,
    std::string datasetName)
    : Thor::BatchSession(std::move(datasetName)),
      dataset(std::move(dataset)),
      splits(std::move(splits)),
      batching(std::move(batching)),
      batchQueueDepth(batchQueueDepth) {
    if (!this->dataset) {
        throw std::runtime_error("DeviceResidentNamedBatchSession requires a dataset.");
    }
    if (batchQueueDepth == 0) {
        throw std::runtime_error(
            "DeviceResidentNamedBatchSession batch_queue_depth must be >= 1.");
    }
    if (this->splits.getDatasetId() != this->dataset->getDatasetId()) {
        throw std::runtime_error(
            "DeviceResidentNamedBatchSession split manifest belongs to a different dataset.");
    }
    if (this->splits.getNumExamples() != this->dataset->getNumExamples()) {
        throw std::runtime_error(
            "DeviceResidentNamedBatchSession split manifest row count does not match resident dataset.");
    }

    this->batchSize = this->batching.getBatchSize();
    for (const Thor::DatasetField &field : this->dataset->getSchema().getFields()) {
        requiredFieldIds.insert(field.id);
    }
    for (const BatchTensorSpec &spec : batchTensorSpecsFor(this->dataset->getLayout())) {
        if (!this->dataset->hasTensor(spec.name)) {
            throw std::runtime_error(
                "DeviceResidentNamedBatchSession resident dataset is missing tensor '" +
                spec.name + "'.");
        }
    }

    initializeSplit(
        ExampleType::TRAIN,
        this->splits.getSharedTrain(),
        this->batching.getRandomizeTrain(),
        this->batching.getRandomSeed());
    initializeSplit(
        ExampleType::VALIDATE,
        this->splits.getSharedValidate(),
        false,
        std::nullopt);
    initializeSplit(
        ExampleType::TEST,
        this->splits.getSharedTest(),
        false,
        std::nullopt);
}

DeviceResidentNamedBatchSession::DeviceResidentNamedBatchSession(
    Thor::DeviceDatasetLease dataset,
    Thor::DeviceDatasetSessionDescription session,
    uint64_t batchQueueDepth,
    std::string datasetName)
    : DeviceResidentNamedBatchSession(
          std::move(dataset),
          session.getSplits(),
          session.getBatching(),
          batchQueueDepth,
          std::move(datasetName)) {
    requiredFieldIds = session.getRequiredFieldIds();
    if (requiredFieldIds.empty()) {
        for (const Thor::DatasetField &field : this->dataset->getSchema().getFields()) {
            requiredFieldIds.insert(field.id);
        }
    }
    for (Thor::DatasetFieldId fieldId : requiredFieldIds) {
        (void)this->dataset->getSchema().getField(fieldId);
    }
}

DeviceResidentNamedBatchSession::~DeviceResidentNamedBatchSession() {
    cancel();
    // Pending batch tensors may still be read by NetworkInput D2D copies. Keep
    // those allocations alive until every source-consumed event has completed.
    for (auto &entry : splitRuntimes) {
        if (entry.second == nullptr) {
            continue;
        }
        try {
            std::lock_guard<std::mutex> guard(entry.second->mutex);
            for (SplitRuntime::PendingBatch &pending : entry.second->pendingBatches) {
                for (Event &event : pending.consumedEvents) {
                    event.synchronize();
                }
            }
        } catch (...) {
            // Destructors must not throw while a CUDA context is already failing.
        }
    }
}

void DeviceResidentNamedBatchSession::cancel() {
    if (cancelled.exchange(true, std::memory_order_acq_rel)) {
        return;
    }
    for (auto &entry : splitRuntimes) {
        if (entry.second != nullptr) {
            entry.second->notEmpty.notify_all();
        }
    }
}

void DeviceResidentNamedBatchSession::initializeSplit(
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
    runtime->gatherStream = Stream(dataset->getPlacement());

    if (runtime->numExamples() != 0) {
        runtime->rowIndicesHost = Tensor(
            TensorPlacement(TensorPlacement::MemDevices::CPU),
            TensorDescriptor(DataType::UINT64, {batchSize}));
        runtime->rowIndicesDevice = Tensor(
            dataset->getPlacement(),
            TensorDescriptor(DataType::UINT64, {batchSize}));
        if (runtime->randomized) {
            runtime->randomizer =
                std::make_unique<FullPeriodRandom>(runtime->numExamples(), false);
            if (runtime->seed.has_value()) {
                runtime->randomizer->reseed(runtime->seed.value());
            }
        }
        for (uint64_t i = 0; i < batchQueueDepth; ++i) {
            runtime->availableBatches.push_back(allocateBatchTensorSet());
        }
    }

    auto [it, inserted] = splitRuntimes.emplace(exampleType, std::move(runtime));
    THOR_THROW_IF_FALSE(inserted);
    (void)it;
}

std::map<std::string, Tensor>
DeviceResidentNamedBatchSession::allocateBatchTensorSet() const {
    std::map<std::string, Tensor> tensors;
    for (const BatchTensorSpec &spec : batchTensorSpecsFor(dataset->getLayout())) {
        tensors.emplace(
            spec.name,
            Tensor(
                dataset->getPlacement(),
                TensorDescriptor(
                    spec.dataType,
                    batchDimensionsFor(spec.exampleDimensions, batchSize))));
    }
    return tensors;
}

DeviceResidentNamedBatchSession::SplitRuntime &
DeviceResidentNamedBatchSession::runtimeFor(ExampleType exampleType) {
    const auto found = splitRuntimes.find(exampleType);
    if (found == splitRuntimes.end() || found->second == nullptr) {
        throw std::runtime_error(
            "DeviceResidentNamedBatchSession does not contain requested split.");
    }
    return *found->second;
}

const DeviceResidentNamedBatchSession::SplitRuntime &
DeviceResidentNamedBatchSession::runtimeFor(ExampleType exampleType) const {
    const auto found = splitRuntimes.find(exampleType);
    if (found == splitRuntimes.end() || found->second == nullptr) {
        throw std::runtime_error(
            "DeviceResidentNamedBatchSession does not contain requested split.");
    }
    return *found->second;
}

void DeviceResidentNamedBatchSession::fillRowIndexTensor(SplitRuntime &runtime) {
    THOR_THROW_IF_FALSE(runtime.numExamples() > 0);
    uint64_t *rowIndices = runtime.rowIndicesHost.getMemPtr<uint64_t>();
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
        const uint64_t sourceRow =
            runtime.sourceIndices->at(logicalPosition);
        THOR_THROW_IF_FALSE(sourceRow < dataset->getNumExamples());
        rowIndices[slot] = sourceRow;
    }
}

Batch DeviceResidentNamedBatchSession::acquireBatch(
    ExampleType exampleType,
    uint64_t &batchNum) {
    if (cancelled.load(std::memory_order_acquire)) {
        throw std::runtime_error("DeviceResidentNamedBatchSession has been cancelled.");
    }
    SplitRuntime &runtime = runtimeFor(exampleType);
    std::unique_lock<std::mutex> lock(runtime.mutex);
    if (runtime.numExamples() == 0) {
        throw std::runtime_error(
            "DeviceResidentNamedBatchSession cannot get a batch from an empty split.");
    }
    runtime.notEmpty.wait(lock, [&] {
        return cancelled.load(std::memory_order_acquire) ||
               !runtime.availableBatches.empty() ||
               !runtime.pendingBatches.empty();
    });
    if (cancelled.load(std::memory_order_acquire)) {
        throw std::runtime_error("DeviceResidentNamedBatchSession has been cancelled.");
    }

    std::map<std::string, Tensor> tensors;
    if (!runtime.availableBatches.empty()) {
        tensors = std::move(runtime.availableBatches.front());
        runtime.availableBatches.pop_front();
    } else {
        THOR_THROW_IF_FALSE(!runtime.pendingBatches.empty());
        SplitRuntime::PendingBatch pending = std::move(runtime.pendingBatches.front());
        runtime.pendingBatches.pop_front();
        lock.unlock();
        for (Event &event : pending.consumedEvents) {
            event.synchronize();
        }
        lock.lock();
        tensors = std::move(pending.tensors);
    }

    if (cancelled.load(std::memory_order_acquire)) {
        runtime.availableBatches.push_front(std::move(tensors));
        throw std::runtime_error("DeviceResidentNamedBatchSession has been cancelled.");
    }

    batchNum = runtime.nextBatchNum;
    runtime.nextBatchNum =
        (runtime.nextBatchNum + 1) % runtime.batchesPerEpoch;
    fillRowIndexTensor(runtime);
    runtime.rowIndicesDevice.copyFromAsync(
        runtime.rowIndicesHost,
        runtime.gatherStream);

    for (const BatchTensorSpec &spec : batchTensorSpecsFor(dataset->getLayout())) {
        Tensor &destination = tensors.at(spec.name);
        const Tensor &source = dataset->tensor(spec.name);
        launchDeviceResidentNamedGatherKernel(
            source,
            destination,
            runtime.rowIndicesDevice,
            runtime.gatherStream);
    }
    runtime.gatherStream.synchronize();
    runtime.batchesGathered += 1;

    auto sourceTensors =
        std::make_shared<std::map<std::string, Tensor>>(std::move(tensors));
    Batch batch = batchFromTensorMap(*sourceTensors);
    std::set<std::string> fieldNames;
    for (const auto &[name, tensor] : *sourceTensors) {
        (void)tensor;
        fieldNames.insert(name);
    }

    std::shared_ptr<DeviceResidentNamedBatchSession> sharedSelf =
        std::dynamic_pointer_cast<DeviceResidentNamedBatchSession>(shared_from_this());
    THOR_THROW_IF_FALSE(sharedSelf != nullptr);
    std::weak_ptr<DeviceResidentNamedBatchSession> weakSelf = sharedSelf;
    Thor::BatchSourceOwner sourceOwner(
        [weakSelf, exampleType, sourceTensors](std::vector<Event> consumedEvents) mutable {
            if (std::shared_ptr<DeviceResidentNamedBatchSession> session = weakSelf.lock()) {
                session->releaseBatchTensorSet(
                    exampleType,
                    std::move(sourceTensors),
                    std::move(consumedEvents));
                return;
            }
            try {
                for (Event &event : consumedEvents) {
                    event.synchronize();
                }
            } catch (...) {
            }
        });
    addBatchSourceResource(batch, std::move(fieldNames), std::move(sourceOwner));
    return batch;
}

void DeviceResidentNamedBatchSession::validateReturnedBatch(
    const std::map<std::string, Tensor> &tensors) const {
    const std::vector<BatchTensorSpec> specs =
        batchTensorSpecsFor(dataset->getLayout());
    if (tensors.size() != specs.size()) {
        throw std::runtime_error(
            "DeviceResidentNamedBatchSession returned batch has wrong tensor count.");
    }
    for (const BatchTensorSpec &spec : specs) {
        const auto found = tensors.find(spec.name);
        if (found == tensors.end()) {
            throw std::runtime_error(
                "DeviceResidentNamedBatchSession returned batch is missing tensor '" +
                spec.name + "'.");
        }
        const Tensor &tensor = found->second;
        if (!tensor.isInitialized()) {
            throw std::runtime_error(
                "DeviceResidentNamedBatchSession returned batch contains uninitialized tensor '" +
                spec.name + "'.");
        }
        if (tensor.getPlacement() != dataset->getPlacement()) {
            throw std::runtime_error(
                "DeviceResidentNamedBatchSession returned batch tensor '" +
                spec.name + "' is not on the resident dataset placement.");
        }
        const TensorDescriptor expected(
            spec.dataType,
            batchDimensionsFor(spec.exampleDimensions, batchSize));
        if (tensor.getDescriptor() != expected) {
            throw std::runtime_error(
                "DeviceResidentNamedBatchSession returned batch tensor '" +
                spec.name + "' has the wrong descriptor.");
        }
    }
}

void DeviceResidentNamedBatchSession::releaseBatchTensorSet(
    ExampleType exampleType,
    std::shared_ptr<std::map<std::string, Tensor>> tensors,
    std::vector<Event> consumedEvents) noexcept {
    if (tensors == nullptr) {
        return;
    }
    auto found = splitRuntimes.find(exampleType);
    if (found == splitRuntimes.end() || found->second == nullptr) {
        try {
            for (Event &event : consumedEvents) {
                event.synchronize();
            }
        } catch (...) {
        }
        return;
    }

    SplitRuntime &runtime = *found->second;
    if (cancelled.load(std::memory_order_acquire)) {
        try {
            for (Event &event : consumedEvents) {
                event.synchronize();
            }
        } catch (...) {
        }
        return;
    }

    try {
        std::lock_guard<std::mutex> guard(runtime.mutex);
        if (consumedEvents.empty()) {
            runtime.availableBatches.push_back(*tensors);
        } else {
            runtime.pendingBatches.push_back(
                SplitRuntime::PendingBatch{
                    *tensors,
                    consumedEvents});
        }
        runtime.batchesReturned += 1;
        runtime.notEmpty.notify_one();
        return;
    } catch (...) {
    }

    // Queue allocation failed. Keep the source tensor handles alive until all
    // asynchronous D2D reads complete before allowing this callback to return.
    try {
        for (Event &event : consumedEvents) {
            event.synchronize();
        }
    } catch (...) {
    }
}

void DeviceResidentNamedBatchSession::recycleBatch(
    ExampleType exampleType,
    Batch &&batch) {
    if (batch.ownsSourceResourceLifecycle()) {
        THOR_THROW_IF_FALSE(batch.allFieldsHaveSourceReferences());
        std::map<std::string, Tensor> tensors = denseTensorMapFromBatchOrThrow(
            batch,
            "DeviceResidentNamedBatchSession returned source-tracked batch");
        validateReturnedBatch(tensors);
        // Clearing seals an owner that a manual caller did not release after
        // submission. Batch copies do not own this lifecycle and therefore
        // continue through the ordinary exact validation/recycle path below.
        batch.clear();
        return;
    }

    if (cancelled.load(std::memory_order_acquire)) {
        return;
    }
    SplitRuntime &runtime = runtimeFor(exampleType);
    std::map<std::string, Tensor> tensors = denseTensorMapFromBatchOrThrow(
        batch,
        "DeviceResidentNamedBatchSession returned batch");
    validateReturnedBatch(tensors);
    {
        std::lock_guard<std::mutex> guard(runtime.mutex);
        runtime.availableBatches.push_back(std::move(tensors));
        runtime.batchesReturned += 1;
    }
    runtime.notEmpty.notify_one();
}

uint64_t DeviceResidentNamedBatchSession::getNumBatchesPerEpoch(
    ExampleType exampleType) {
    return runtimeFor(exampleType).batchesPerEpoch;
}

uint64_t DeviceResidentNamedBatchSession::getNumExamples(
    ExampleType exampleType) {
    return runtimeFor(exampleType).numExamples();
}

uint64_t DeviceResidentNamedBatchSession::getNextBatchNum(
    ExampleType exampleType) {
    SplitRuntime &runtime = runtimeFor(exampleType);
    std::lock_guard<std::mutex> guard(runtime.mutex);
    return runtime.numExamples() == 0 ? 0 : runtime.nextBatchNum;
}

std::optional<TensorPlacement>
DeviceResidentNamedBatchSession::getBatchTensorPlacement(
    const std::string &tensorName) const {
    if (dataset->hasTensor(tensorName)) {
        return dataset->getPlacement();
    }
    return std::nullopt;
}

std::vector<Event> DeviceResidentNamedBatchSession::getSynchronizeEvents() const {
    std::vector<Event> events;
    events.reserve(splitRuntimes.size());
    for (const auto &entry : splitRuntimes) {
        if (entry.second == nullptr || entry.second->numExamples() == 0) {
            continue;
        }
        std::lock_guard<std::mutex> guard(entry.second->mutex);
        events.push_back(entry.second->gatherStream.putEvent(false, true));
        for (const SplitRuntime::PendingBatch &pending :
             entry.second->pendingBatches) {
            events.insert(
                events.end(),
                pending.consumedEvents.begin(),
                pending.consumedEvents.end());
        }
    }
    return events;
}

DeviceResidentNamedBatchSessionStats
DeviceResidentNamedBatchSession::getStatsSnapshot(ExampleType exampleType) const {
    const SplitRuntime &runtime = runtimeFor(exampleType);
    std::lock_guard<std::mutex> guard(runtime.mutex);
    DeviceResidentNamedBatchSessionStats stats;
    stats.splitName = runtime.splitName;
    stats.residentExamples = dataset->getNumExamples();
    stats.residentBytes = dataset->totalBytes();
    stats.batchesGathered = runtime.batchesGathered;
    stats.batchesReturned = runtime.batchesReturned;
    stats.currentAvailableBatches = runtime.availableBatches.size();
    stats.batchQueueDepth = batchQueueDepth;
    return stats;
}
