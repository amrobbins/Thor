#include "DeepLearning/Api/Loaders/DeviceResidentWindowedNamedBatchLoader.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Loaders/DeviceResidentNamedGatherKernel.h"

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

std::vector<uint64_t> batchDimensionsFor(const std::vector<uint64_t> &exampleDimensions, uint64_t batchSize) {
    std::vector<uint64_t> dimensions;
    dimensions.reserve(exampleDimensions.size() + 1);
    dimensions.push_back(batchSize);
    dimensions.insert(dimensions.end(), exampleDimensions.begin(), exampleDimensions.end());
    return dimensions;
}

std::vector<BatchTensorSpec> batchTensorSpecsFor(const LocalNamedExampleLayout &layout,
                                                 uint64_t batchSize,
                                                 TensorPlacement windowPlacement) {
    (void)batchSize;
    std::vector<BatchTensorSpec> specs;
    specs.reserve(layout.tensors().size() + layout.windowedTensors().size() * 2);
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    for (const LocalNamedExampleLayout::TensorSpec &spec : layout.tensors()) {
        specs.push_back(BatchTensorSpec{spec.name, spec.dataType, spec.dimensions, cpuPlacement});
    }
    for (const LocalNamedExampleLayout::WindowedTensorSpec &spec : layout.windowedTensors()) {
        specs.push_back(BatchTensorSpec{spec.name, spec.dataType, spec.dimensions, windowPlacement});
        if (spec.maskName.has_value()) {
            specs.push_back(BatchTensorSpec{spec.maskName.value(), DataType::UINT8, std::vector<uint64_t>{spec.windowLength()}, windowPlacement});
        }
    }
    return specs;
}

std::vector<std::string> windowedTensorNames(const LocalNamedExampleLayout &layout) {
    std::vector<std::string> names;
    for (const LocalNamedExampleLayout::WindowedTensorSpec &spec : layout.windowedTensors()) {
        names.push_back(spec.name);
        if (spec.maskName.has_value()) {
            names.push_back(spec.maskName.value());
        }
    }
    return names;
}

}  // namespace

DeviceResidentWindowedNamedBatchLoader::DeviceResidentWindowedNamedBatchLoader(
    DeviceDatasetMaterializationView view,
    std::shared_ptr<DeviceResidentNamedDataset> windowedDataset,
    uint64_t batchQueueDepth,
    uint64_t readerQueueDepth)
    : view(std::move(view)),
      windowedDataset(std::move(windowedDataset)),
      batchQueueDepth(batchQueueDepth),
      readerQueueDepth(readerQueueDepth) {
    if (this->windowedDataset == nullptr) {
        throw std::runtime_error("DeviceResidentWindowedNamedBatchLoader requires a windowed device dataset.");
    }
    if (!this->view.layout.hasWindowedTensors()) {
        throw std::runtime_error("DeviceResidentWindowedNamedBatchLoader requires at least one windowed tensor.");
    }
    if (batchQueueDepth == 0) {
        throw std::runtime_error("DeviceResidentWindowedNamedBatchLoader batch_queue_depth must be >= 1.");
    }
    if (readerQueueDepth == 0) {
        throw std::runtime_error("DeviceResidentWindowedNamedBatchLoader reader_queue_depth must be >= 1.");
    }
    if (this->view.batchSize == 0) {
        throw std::runtime_error("DeviceResidentWindowedNamedBatchLoader requires batch_size >= 1.");
    }
    this->batchSize = this->view.batchSize;
    this->reader = IndexedLocalNamedExampleReader::openDataset(this->view.datasetPath, this->view.layout);

    for (ExampleType exampleType : {ExampleType::TRAIN, ExampleType::VALIDATE, ExampleType::TEST}) {
        const DeviceResidentNamedSplit *split = this->windowedDataset->findSplit(exampleType);
        if (split != nullptr) {
            initializeSplit(*split);
        }
    }
}

DeviceResidentWindowedNamedBatchLoader::~DeviceResidentWindowedNamedBatchLoader() {
    for (auto &entry : splitRuntimes) {
        if (entry.second != nullptr) {
            std::lock_guard<std::mutex> guard(entry.second->mutex);
            entry.second->notEmpty.notify_all();
        }
    }
}

void DeviceResidentWindowedNamedBatchLoader::initializeSplit(const DeviceResidentNamedSplit &split) {
    auto runtime = std::make_unique<SplitRuntime>();
    runtime->split = &split;
    runtime->gatherStream = Stream(windowedDataset->getPlacement());

    if (split.numExamples() != 0) {
        runtime->rowIndicesHost = Tensor(TensorPlacement(TensorPlacement::MemDevices::CPU), TensorDescriptor(DataType::UINT64, {batchSize}));
        runtime->rowIndicesDevice = Tensor(windowedDataset->getPlacement(), TensorDescriptor(DataType::UINT64, {batchSize}));
        runtime->readerSession = reader->createSession(readerQueueDepth);
        if (split.randomized) {
            runtime->randomizer = std::make_unique<FullPeriodRandom>(split.numExamples(), false);
            if (split.seed.has_value()) {
                runtime->randomizer->reseed(split.seed.value());
            }
        }
        for (uint64_t i = 0; i < batchQueueDepth; ++i) {
            runtime->availableBatches.push_back(allocateBatchTensorSet());
        }
    }

    auto [it, inserted] = splitRuntimes.emplace(split.exampleType, std::move(runtime));
    THOR_THROW_IF_FALSE(inserted);
    (void)it;
}

std::map<std::string, Tensor> DeviceResidentWindowedNamedBatchLoader::allocateBatchTensorSet() const {
    std::map<std::string, Tensor> tensors;
    for (const BatchTensorSpec &spec : batchTensorSpecsFor(view.layout, batchSize, windowedDataset->getPlacement())) {
        tensors.emplace(spec.name,
                        Tensor(spec.placement, TensorDescriptor(spec.dataType, batchDimensionsFor(spec.exampleDimensions, batchSize))));
    }
    return tensors;
}

DeviceResidentWindowedNamedBatchLoader::SplitRuntime &DeviceResidentWindowedNamedBatchLoader::runtimeFor(ExampleType exampleType) {
    const auto found = splitRuntimes.find(exampleType);
    if (found == splitRuntimes.end() || found->second == nullptr) {
        throw std::runtime_error("DeviceResidentWindowedNamedBatchLoader does not contain requested split.");
    }
    return *found->second;
}

const DeviceResidentWindowedNamedBatchLoader::SplitRuntime &DeviceResidentWindowedNamedBatchLoader::runtimeFor(ExampleType exampleType) const {
    const auto found = splitRuntimes.find(exampleType);
    if (found == splitRuntimes.end() || found->second == nullptr) {
        throw std::runtime_error("DeviceResidentWindowedNamedBatchLoader does not contain requested split.");
    }
    return *found->second;
}

void DeviceResidentWindowedNamedBatchLoader::fillRowIndexTensor(SplitRuntime &runtime) {
    THOR_THROW_IF_FALSE(runtime.split != nullptr);
    THOR_THROW_IF_FALSE(runtime.split->numExamples() > 0);
    uint64_t *rowIndices = runtime.rowIndicesHost.getMemPtr<uint64_t>();
    for (uint64_t slot = 0; slot < batchSize; ++slot) {
        uint64_t logicalPosition = 0;
        if (runtime.split->randomized) {
            THOR_THROW_IF_FALSE(runtime.randomizer != nullptr);
            logicalPosition = runtime.randomizer->getRandomNumber();
        } else {
            logicalPosition = runtime.nextLogicalPosition;
            runtime.nextLogicalPosition = (runtime.nextLogicalPosition + 1) % runtime.split->numExamples();
        }
        THOR_THROW_IF_FALSE(logicalPosition < runtime.split->numExamples());
        rowIndices[slot] = logicalPosition;
    }
}

Batch DeviceResidentWindowedNamedBatchLoader::getBatch(ExampleType exampleType, uint64_t &batchNum) {
    SplitRuntime &runtime = runtimeFor(exampleType);
    std::unique_lock<std::mutex> lock(runtime.mutex);
    if (runtime.split == nullptr || runtime.split->numExamples() == 0) {
        throw std::runtime_error("DeviceResidentWindowedNamedBatchLoader cannot get a batch from an empty split.");
    }
    while (runtime.availableBatches.empty()) {
        runtime.notEmpty.wait(lock);
    }

    std::map<std::string, Tensor> tensors = std::move(runtime.availableBatches.front());
    runtime.availableBatches.pop_front();

    batchNum = runtime.nextBatchNum;
    runtime.nextBatchNum = (runtime.nextBatchNum + 1) % runtime.split->batchesPerEpoch;
    fillRowIndexTensor(runtime);

    std::vector<uint8_t *> directPointers(reader->getTensorCount(), nullptr);
    for (const LocalNamedExampleLayout::TensorSpec &spec : view.layout.tensors()) {
        const uint64_t ordinal = reader->getLayoutTensorOrdinal(spec.name);
        directPointers.at(static_cast<size_t>(ordinal)) = static_cast<uint8_t *>(tensors.at(spec.name).getMemPtr());
    }
    const uint64_t *logicalRows = runtime.rowIndicesHost.getMemPtr<uint64_t>();
    THOR_THROW_IF_FALSE(runtime.readerSession != nullptr);
    for (uint64_t slot = 0; slot < batchSize; ++slot) {
        const uint64_t logicalRow = logicalRows[slot];
        THOR_THROW_IF_FALSE(logicalRow < runtime.split->sourceIndices.size());
        runtime.readerSession->loadDirectExampleInto(runtime.split->sourceIndices.at(static_cast<size_t>(logicalRow)), slot, directPointers);
    }
    runtime.readerSession->drain();

    runtime.rowIndicesDevice.copyFromAsync(runtime.rowIndicesHost, runtime.gatherStream);
    for (const std::string &name : windowedTensorNames(view.layout)) {
        Tensor &destination = tensors.at(name);
        const Tensor &source = runtime.split->tensor(name);
        launchDeviceResidentNamedGatherKernel(source, destination, runtime.rowIndicesDevice, runtime.gatherStream);
    }
    runtime.gatherStream.synchronize();

    return batchFromTensorMap(std::move(tensors));
}

void DeviceResidentWindowedNamedBatchLoader::validateReturnedBatch(const std::map<std::string, Tensor> &tensors) const {
    const std::vector<BatchTensorSpec> specs = batchTensorSpecsFor(view.layout, batchSize, windowedDataset->getPlacement());
    if (tensors.size() != specs.size()) {
        throw std::runtime_error("DeviceResidentWindowedNamedBatchLoader returned batch has wrong tensor count.");
    }
    for (const BatchTensorSpec &spec : specs) {
        const auto found = tensors.find(spec.name);
        if (found == tensors.end()) {
            throw std::runtime_error("DeviceResidentWindowedNamedBatchLoader returned batch is missing tensor '" + spec.name + "'.");
        }
        const Tensor &tensor = found->second;
        if (!tensor.isInitialized()) {
            throw std::runtime_error("DeviceResidentWindowedNamedBatchLoader returned batch contains uninitialized tensor '" + spec.name + "'.");
        }
        if (tensor.getPlacement() != spec.placement) {
            throw std::runtime_error("DeviceResidentWindowedNamedBatchLoader returned batch tensor '" + spec.name +
                                     "' is on the wrong placement.");
        }
        const TensorDescriptor expected(spec.dataType, batchDimensionsFor(spec.exampleDimensions, batchSize));
        if (tensor.getDescriptor() != expected) {
            throw std::runtime_error("DeviceResidentWindowedNamedBatchLoader returned batch tensor '" + spec.name +
                                     "' has the wrong descriptor.");
        }
    }
}

void DeviceResidentWindowedNamedBatchLoader::returnBatchBuffers(ExampleType exampleType, Batch &&batch) {
    SplitRuntime &runtime = runtimeFor(exampleType);
    std::map<std::string, Tensor> tensors = denseTensorMapFromBatchOrThrow(batch, "DeviceResidentWindowedNamedBatchLoader returned batch");
    validateReturnedBatch(tensors);
    {
        std::lock_guard<std::mutex> guard(runtime.mutex);
        runtime.availableBatches.push_back(std::move(tensors));
    }
    runtime.notEmpty.notify_one();
}

uint64_t DeviceResidentWindowedNamedBatchLoader::getNumBatchesPerEpoch(ExampleType exampleType) {
    const SplitRuntime &runtime = runtimeFor(exampleType);
    return runtime.split == nullptr ? 0 : runtime.split->batchesPerEpoch;
}

uint64_t DeviceResidentWindowedNamedBatchLoader::getNumExamples(ExampleType exampleType) {
    const SplitRuntime &runtime = runtimeFor(exampleType);
    return runtime.split == nullptr ? 0 : runtime.split->numExamples();
}

uint64_t DeviceResidentWindowedNamedBatchLoader::getNextBatchNum(ExampleType exampleType) {
    SplitRuntime &runtime = runtimeFor(exampleType);
    std::lock_guard<std::mutex> guard(runtime.mutex);
    return runtime.split == nullptr || runtime.split->numExamples() == 0 ? 0 : runtime.nextBatchNum;
}
