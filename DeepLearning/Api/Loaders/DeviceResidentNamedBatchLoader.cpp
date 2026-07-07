#include "DeepLearning/Api/Loaders/DeviceResidentNamedBatchLoader.h"

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
};

std::vector<uint64_t> batchDimensionsFor(const std::vector<uint64_t> &exampleDimensions, uint64_t batchSize) {
    std::vector<uint64_t> dimensions;
    dimensions.reserve(exampleDimensions.size() + 1);
    dimensions.push_back(batchSize);
    dimensions.insert(dimensions.end(), exampleDimensions.begin(), exampleDimensions.end());
    return dimensions;
}

std::vector<BatchTensorSpec> batchTensorSpecsFor(const LocalNamedExampleLayout &layout) {
    std::vector<BatchTensorSpec> specs;
    specs.reserve(layout.tensors().size() + layout.windowedTensors().size() * 2);
    for (const LocalNamedExampleLayout::TensorSpec &spec : layout.tensors()) {
        specs.push_back(BatchTensorSpec{spec.name, spec.dataType, spec.dimensions});
    }
    for (const LocalNamedExampleLayout::WindowedTensorSpec &spec : layout.windowedTensors()) {
        specs.push_back(BatchTensorSpec{spec.name, spec.dataType, spec.dimensions});
        if (spec.maskName.has_value()) {
            specs.push_back(BatchTensorSpec{spec.maskName.value(), DataType::UINT8, std::vector<uint64_t>{spec.windowLength()}});
        }
    }
    return specs;
}

}  // namespace

DeviceResidentNamedBatchLoader::DeviceResidentNamedBatchLoader(std::shared_ptr<DeviceResidentNamedDataset> dataset,
                                                               uint64_t batchQueueDepth)
    : dataset(std::move(dataset)), batchQueueDepth(batchQueueDepth) {
    if (this->dataset == nullptr) {
        throw std::runtime_error("DeviceResidentNamedBatchLoader requires a dataset.");
    }
    if (batchQueueDepth == 0) {
        throw std::runtime_error("DeviceResidentNamedBatchLoader batch_queue_depth must be >= 1.");
    }
    if (this->dataset->getBatchSize() == 0) {
        throw std::runtime_error("DeviceResidentNamedBatchLoader requires dataset batch_size >= 1.");
    }

    this->batchSize = this->dataset->getBatchSize();
    for (ExampleType exampleType : {ExampleType::TRAIN, ExampleType::VALIDATE, ExampleType::TEST}) {
        const DeviceResidentNamedSplit *split = this->dataset->findSplit(exampleType);
        if (split != nullptr) {
            initializeSplit(*split);
        }
    }
}

DeviceResidentNamedBatchLoader::~DeviceResidentNamedBatchLoader() {
    for (auto &entry : splitRuntimes) {
        if (entry.second != nullptr) {
            std::lock_guard<std::mutex> guard(entry.second->mutex);
            entry.second->notEmpty.notify_all();
        }
    }
}

void DeviceResidentNamedBatchLoader::initializeSplit(const DeviceResidentNamedSplit &split) {
    auto runtime = std::make_unique<SplitRuntime>();
    runtime->split = &split;
    runtime->gatherStream = Stream(dataset->getPlacement());

    if (split.numExamples() != 0) {
        runtime->rowIndicesHost = Tensor(TensorPlacement(TensorPlacement::MemDevices::CPU), TensorDescriptor(DataType::UINT64, {batchSize}));
        runtime->rowIndicesDevice = Tensor(dataset->getPlacement(), TensorDescriptor(DataType::UINT64, {batchSize}));
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

std::map<std::string, Tensor> DeviceResidentNamedBatchLoader::allocateBatchTensorSet() const {
    std::map<std::string, Tensor> tensors;
    for (const BatchTensorSpec &spec : batchTensorSpecsFor(dataset->getLayout())) {
        tensors.emplace(spec.name,
                        Tensor(dataset->getPlacement(),
                               TensorDescriptor(spec.dataType, batchDimensionsFor(spec.exampleDimensions, batchSize))));
    }
    return tensors;
}

DeviceResidentNamedBatchLoader::SplitRuntime &DeviceResidentNamedBatchLoader::runtimeFor(ExampleType exampleType) {
    const auto found = splitRuntimes.find(exampleType);
    if (found == splitRuntimes.end() || found->second == nullptr) {
        throw std::runtime_error("DeviceResidentNamedBatchLoader does not contain requested split.");
    }
    return *found->second;
}

const DeviceResidentNamedBatchLoader::SplitRuntime &DeviceResidentNamedBatchLoader::runtimeFor(ExampleType exampleType) const {
    const auto found = splitRuntimes.find(exampleType);
    if (found == splitRuntimes.end() || found->second == nullptr) {
        throw std::runtime_error("DeviceResidentNamedBatchLoader does not contain requested split.");
    }
    return *found->second;
}

void DeviceResidentNamedBatchLoader::fillRowIndexTensor(SplitRuntime &runtime) {
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

Batch DeviceResidentNamedBatchLoader::getBatch(ExampleType exampleType, uint64_t &batchNum) {
    SplitRuntime &runtime = runtimeFor(exampleType);
    std::unique_lock<std::mutex> lock(runtime.mutex);
    if (runtime.split == nullptr || runtime.split->numExamples() == 0) {
        throw std::runtime_error("DeviceResidentNamedBatchLoader cannot get a batch from an empty split.");
    }
    while (runtime.availableBatches.empty()) {
        runtime.notEmpty.wait(lock);
    }

    std::map<std::string, Tensor> tensors = std::move(runtime.availableBatches.front());
    runtime.availableBatches.pop_front();

    batchNum = runtime.nextBatchNum;
    runtime.nextBatchNum = (runtime.nextBatchNum + 1) % runtime.split->batchesPerEpoch;
    fillRowIndexTensor(runtime);
    runtime.rowIndicesDevice.copyFromAsync(runtime.rowIndicesHost, runtime.gatherStream);

    for (const BatchTensorSpec &spec : batchTensorSpecsFor(dataset->getLayout())) {
        Tensor &destination = tensors.at(spec.name);
        const Tensor &source = runtime.split->tensor(spec.name);
        launchDeviceResidentNamedGatherKernel(source, destination, runtime.rowIndicesDevice, runtime.gatherStream);
    }
    runtime.gatherStream.synchronize();
    runtime.batchesGathered += 1;

    return batchFromTensorMap(std::move(tensors));
}

void DeviceResidentNamedBatchLoader::validateReturnedBatch(const std::map<std::string, Tensor> &tensors) const {
    const std::vector<BatchTensorSpec> specs = batchTensorSpecsFor(dataset->getLayout());
    if (tensors.size() != specs.size()) {
        throw std::runtime_error("DeviceResidentNamedBatchLoader returned batch has wrong tensor count.");
    }
    for (const BatchTensorSpec &spec : specs) {
        const auto found = tensors.find(spec.name);
        if (found == tensors.end()) {
            throw std::runtime_error("DeviceResidentNamedBatchLoader returned batch is missing tensor '" + spec.name + "'.");
        }
        const Tensor &tensor = found->second;
        if (!tensor.isInitialized()) {
            throw std::runtime_error("DeviceResidentNamedBatchLoader returned batch contains uninitialized tensor '" + spec.name + "'.");
        }
        if (tensor.getPlacement() != dataset->getPlacement()) {
            throw std::runtime_error("DeviceResidentNamedBatchLoader returned batch tensor '" + spec.name +
                                     "' is not on the resident dataset placement.");
        }
        const TensorDescriptor expected(spec.dataType, batchDimensionsFor(spec.exampleDimensions, batchSize));
        if (tensor.getDescriptor() != expected) {
            throw std::runtime_error("DeviceResidentNamedBatchLoader returned batch tensor '" + spec.name +
                                     "' has the wrong descriptor.");
        }
    }
}

void DeviceResidentNamedBatchLoader::returnBatchBuffers(ExampleType exampleType, Batch &&batch) {
    SplitRuntime &runtime = runtimeFor(exampleType);
    std::map<std::string, Tensor> tensors = denseTensorMapFromBatchOrThrow(batch, "DeviceResidentNamedBatchLoader returned batch");
    validateReturnedBatch(tensors);
    {
        std::lock_guard<std::mutex> guard(runtime.mutex);
        runtime.availableBatches.push_back(std::move(tensors));
        runtime.batchesReturned += 1;
    }
    runtime.notEmpty.notify_one();
}

uint64_t DeviceResidentNamedBatchLoader::getNumBatchesPerEpoch(ExampleType exampleType) {
    const SplitRuntime &runtime = runtimeFor(exampleType);
    return runtime.split == nullptr ? 0 : runtime.split->batchesPerEpoch;
}

uint64_t DeviceResidentNamedBatchLoader::getNumExamples(ExampleType exampleType) {
    const SplitRuntime &runtime = runtimeFor(exampleType);
    return runtime.split == nullptr ? 0 : runtime.split->numExamples();
}

uint64_t DeviceResidentNamedBatchLoader::getNextBatchNum(ExampleType exampleType) {
    SplitRuntime &runtime = runtimeFor(exampleType);
    std::lock_guard<std::mutex> guard(runtime.mutex);
    return runtime.split == nullptr || runtime.split->numExamples() == 0 ? 0 : runtime.nextBatchNum;
}

DeviceResidentNamedBatchLoaderStats DeviceResidentNamedBatchLoader::getStatsSnapshot(ExampleType exampleType) const {
    const SplitRuntime &runtime = runtimeFor(exampleType);
    std::lock_guard<std::mutex> guard(runtime.mutex);
    DeviceResidentNamedBatchLoaderStats stats;
    if (runtime.split != nullptr) {
        stats.splitName = runtime.split->splitName;
        stats.residentExamples = runtime.split->numExamples();
        stats.residentBytes = runtime.split->totalBytes();
    }
    stats.batchesGathered = runtime.batchesGathered;
    stats.batchesReturned = runtime.batchesReturned;
    stats.currentAvailableBatches = runtime.availableBatches.size();
    stats.batchQueueDepth = batchQueueDepth;
    return stats;
}
