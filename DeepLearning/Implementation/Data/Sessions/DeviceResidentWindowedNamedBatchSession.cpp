#include "DeepLearning/Implementation/Data/Sessions/DeviceResidentWindowedNamedBatchSession.h"

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
    TensorPlacement placement;
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

std::vector<BatchTensorSpec> batchTensorSpecsFor(
    const DatasetLayout &layout,
    TensorPlacement windowPlacement) {
    std::vector<BatchTensorSpec> specs;
    specs.reserve(layout.tensors().size() + layout.windowedTensors().size() * 2);
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    for (const DatasetLayout::TensorSpec &spec : layout.tensors()) {
        specs.push_back(BatchTensorSpec{
            spec.name,
            spec.dataType,
            spec.dimensions,
            cpuPlacement});
    }
    for (const DatasetLayout::WindowedTensorSpec &spec : layout.windowedTensors()) {
        specs.push_back(BatchTensorSpec{
            spec.name,
            spec.dataType,
            spec.dimensions,
            windowPlacement});
        if (spec.maskName.has_value()) {
            specs.push_back(BatchTensorSpec{
                spec.maskName.value(),
                DataType::UINT8,
                std::vector<uint64_t>{spec.windowLength()},
                windowPlacement});
        }
    }
    return specs;
}

std::vector<std::string> windowedTensorNames(
    const DatasetLayout &layout) {
    std::vector<std::string> names;
    for (const DatasetLayout::WindowedTensorSpec &spec : layout.windowedTensors()) {
        names.push_back(spec.name);
        if (spec.maskName.has_value()) {
            names.push_back(spec.maskName.value());
        }
    }
    return names;
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
    if (batchQueueDepth == 0) {
        throw std::runtime_error(
            "DeviceResidentWindowedNamedBatchSession batch_queue_depth must be >= 1.");
    }
    if (readerQueueDepth == 0) {
        throw std::runtime_error(
            "DeviceResidentWindowedNamedBatchSession reader_queue_depth must be >= 1.");
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
    this->reader = IndexedLocalNamedExampleReader::openDataset(
        this->datasetDescription.datasetPath,
        this->datasetDescription.layout);
    if (this->reader->getNumExamples() != this->datasetDescription.numExamples) {
        throw std::runtime_error(
            "DeviceResidentWindowedNamedBatchSession source dataset row count changed.");
    }
    for (const std::string &name : windowedTensorNames(this->datasetDescription.layout)) {
        if (!this->windowedDataset->hasTensor(name)) {
            throw std::runtime_error(
                "DeviceResidentWindowedNamedBatchSession resident dataset is missing tensor '" +
                name + "'.");
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
    runtime->gatherStream = Stream(windowedDataset->getPlacement());

    if (runtime->numExamples() != 0) {
        runtime->rowIndicesHost = Tensor(
            TensorPlacement(TensorPlacement::MemDevices::CPU),
            TensorDescriptor(DataType::UINT64, {batchSize}));
        runtime->rowIndicesDevice = Tensor(
            windowedDataset->getPlacement(),
            TensorDescriptor(DataType::UINT64, {batchSize}));
        runtime->readerSession = reader->createSession(readerQueueDepth);
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
DeviceResidentWindowedNamedBatchSession::allocateBatchTensorSet() const {
    std::map<std::string, Tensor> tensors;
    for (const BatchTensorSpec &spec : batchTensorSpecsFor(
             datasetDescription.layout,
             windowedDataset->getPlacement())) {
        tensors.emplace(
            spec.name,
            Tensor(
                spec.placement,
                TensorDescriptor(
                    spec.dataType,
                    batchDimensionsFor(spec.exampleDimensions, batchSize))));
    }
    return tensors;
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
    SplitRuntime &runtime) {
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
        return cancelled.load(std::memory_order_acquire) || !runtime.availableBatches.empty();
    });
    if (cancelled.load(std::memory_order_acquire)) {
        throw std::runtime_error("DeviceResidentWindowedNamedBatchSession has been cancelled.");
    }

    std::map<std::string, Tensor> tensors =
        std::move(runtime.availableBatches.front());
    runtime.availableBatches.pop_front();

    batchNum = runtime.nextBatchNum;
    runtime.nextBatchNum =
        (runtime.nextBatchNum + 1) % runtime.batchesPerEpoch;
    fillRowIndexTensor(runtime);

    std::vector<uint8_t *> directPointers(reader->getTensorCount(), nullptr);
    for (const DatasetLayout::TensorSpec &spec : datasetDescription.layout.tensors()) {
        const uint64_t ordinal = reader->getLayoutTensorOrdinal(spec.name);
        directPointers.at(static_cast<size_t>(ordinal)) =
            static_cast<uint8_t *>(tensors.at(spec.name).getMemPtr());
    }
    const uint64_t *sourceRows = runtime.rowIndicesHost.getMemPtr<uint64_t>();
    THOR_THROW_IF_FALSE(runtime.readerSession != nullptr);
    for (uint64_t slot = 0; slot < batchSize; ++slot) {
        runtime.readerSession->loadDirectExampleInto(
            sourceRows[slot],
            slot,
            directPointers);
    }
    runtime.readerSession->drain();

    runtime.rowIndicesDevice.copyFromAsync(
        runtime.rowIndicesHost,
        runtime.gatherStream);
    for (const std::string &name : windowedTensorNames(datasetDescription.layout)) {
        Tensor &destination = tensors.at(name);
        const Tensor &source = windowedDataset->tensor(name);
        launchDeviceResidentNamedGatherKernel(
            source,
            destination,
            runtime.rowIndicesDevice,
            runtime.gatherStream);
    }
    runtime.gatherStream.synchronize();

    return batchFromTensorMap(std::move(tensors));
}

void DeviceResidentWindowedNamedBatchSession::validateReturnedBatch(
    const std::map<std::string, Tensor> &tensors) const {
    const std::vector<BatchTensorSpec> specs = batchTensorSpecsFor(
        datasetDescription.layout,
        windowedDataset->getPlacement());
    if (tensors.size() != specs.size()) {
        throw std::runtime_error(
            "DeviceResidentWindowedNamedBatchSession returned batch has wrong tensor count.");
    }
    for (const BatchTensorSpec &spec : specs) {
        const auto found = tensors.find(spec.name);
        if (found == tensors.end()) {
            throw std::runtime_error(
                "DeviceResidentWindowedNamedBatchSession returned batch is missing tensor '" +
                spec.name + "'.");
        }
        const Tensor &tensor = found->second;
        if (!tensor.isInitialized()) {
            throw std::runtime_error(
                "DeviceResidentWindowedNamedBatchSession returned batch contains uninitialized tensor '" +
                spec.name + "'.");
        }
        if (tensor.getPlacement() != spec.placement) {
            throw std::runtime_error(
                "DeviceResidentWindowedNamedBatchSession returned batch tensor '" +
                spec.name + "' is on the wrong placement.");
        }
        const TensorDescriptor expected(
            spec.dataType,
            batchDimensionsFor(spec.exampleDimensions, batchSize));
        if (tensor.getDescriptor() != expected) {
            throw std::runtime_error(
                "DeviceResidentWindowedNamedBatchSession returned batch tensor '" +
                spec.name + "' has the wrong descriptor.");
        }
    }
}

void DeviceResidentWindowedNamedBatchSession::recycleBatch(
    ExampleType exampleType,
    Batch &&batch) {
    if (cancelled.load(std::memory_order_acquire)) {
        return;
    }
    SplitRuntime &runtime = runtimeFor(exampleType);
    std::map<std::string, Tensor> tensors = denseTensorMapFromBatchOrThrow(
        batch,
        "DeviceResidentWindowedNamedBatchSession returned batch");
    validateReturnedBatch(tensors);
    {
        std::lock_guard<std::mutex> guard(runtime.mutex);
        runtime.availableBatches.push_back(std::move(tensors));
    }
    runtime.notEmpty.notify_one();
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
    for (const BatchTensorSpec &spec : batchTensorSpecsFor(
             datasetDescription.layout,
             windowedDataset->getPlacement())) {
        if (spec.name == tensorName) {
            return spec.placement;
        }
    }
    return std::nullopt;
}

std::vector<Event> DeviceResidentWindowedNamedBatchSession::getSynchronizeEvents() const {
    std::vector<Event> events;
    events.reserve(splitRuntimes.size());
    for (const auto &entry : splitRuntimes) {
        if (entry.second == nullptr || entry.second->numExamples() == 0) {
            continue;
        }
        std::lock_guard<std::mutex> guard(entry.second->mutex);
        events.push_back(entry.second->gatherStream.putEvent(false, true));
    }
    return events;
}
