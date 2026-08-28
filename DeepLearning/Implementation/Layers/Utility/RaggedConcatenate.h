#pragma once

#include "DeepLearning/Implementation/Layers/MultiConnectionLayer.h"
#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Expression/CudaHelpers.h"
#include "Utilities/TensorOperations/Ragged/RaggedConcatenate.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ThorImplementation {

// Concatenate canonical rank-1 RaggedTensor packed values along a trailing
// value axis. The final input port is the shared structural offsets tensor.
// Forward and backward kernels use offsets[batchSize] on-device and therefore
// never read inactive packed capacity.
class RaggedConcatenate : public MultiConnectionLayer {
   public:
    RaggedConcatenate(unsigned int valuesAxis, uint32_t expectedValueInputs, uint64_t batchSize)
        : axis(valuesAxis), valueInputCount(expectedValueInputs), batchSize(batchSize) {
        if (valueInputCount < 2) throw std::invalid_argument("RaggedConcatenate requires at least two values inputs.");
        if (batchSize == 0) throw std::invalid_argument("RaggedConcatenate batch size must be positive.");
        offsetsInputIndex = valueInputCount;
        const uint32_t totalInputCount = valueInputCount + 1;
        previousLayers.resize(totalInputCount);
        featureInputs.resize(totalInputCount);
        streams.resize(totalInputCount);
        errorOutputs.resize(totalInputCount);
    }

    ~RaggedConcatenate() override = default;

    std::string getType() override { return "RaggedConcatenate"; }

    std::optional<Tensor> createFeatureOutputTensor() override {
        if (!featureInputs[0].has_value()) throw std::logic_error("RaggedConcatenate input[0] is missing.");
        const TensorDescriptor& reference = featureInputs[0]->getDescriptor();
        const auto& referenceDimensions = reference.getDimensions();
        if (referenceDimensions.size() < 2 || axis == 0 || axis >= referenceDimensions.size()) {
            throw std::logic_error("RaggedConcatenate axis must name a trailing packed-value dimension.");
        }
        const uint64_t capacityRows = referenceDimensions[0];
        uint64_t newAxisSize = 0;
        for (uint32_t i = 0; i < valueInputCount; ++i) {
            if (!featureInputs[i].has_value()) throw std::logic_error("RaggedConcatenate values input is missing.");
            const TensorDescriptor& descriptor = featureInputs[i]->getDescriptor();
            const auto& dimensions = descriptor.getDimensions();
            if (descriptor.getDataType() != reference.getDataType() || dimensions.size() != referenceDimensions.size() ||
                dimensions[0] != capacityRows) {
                throw std::invalid_argument("RaggedConcatenate values inputs must share dtype, rank, and packed capacity.");
            }
            for (uint32_t d = 1; d < dimensions.size(); ++d) {
                if (d != axis && dimensions[d] != referenceDimensions[d]) {
                    throw std::invalid_argument("RaggedConcatenate non-concatenated trailing dimensions must match.");
                }
            }
            newAxisSize += dimensions[axis];
        }
        if (!featureInputs[offsetsInputIndex].has_value()) throw std::logic_error("RaggedConcatenate offsets input is missing.");
        const TensorDescriptor& offsetsDescriptor = featureInputs[offsetsInputIndex]->getDescriptor();
        const auto& offsetsDimensions = offsetsDescriptor.getDimensions();
        if (offsetsDimensions != std::vector<uint64_t>{batchSize + 1}) {
            throw std::invalid_argument("RaggedConcatenate offsets shape must be [batch_size + 1].");
        }
        if (offsetsDescriptor.getDataType() != DataType::UINT32 && offsetsDescriptor.getDataType() != DataType::UINT64) {
            throw std::invalid_argument("RaggedConcatenate offsets must use UINT32 or UINT64 storage.");
        }

        std::vector<uint64_t> outputDimensions = referenceDimensions;
        outputDimensions[axis] = newAxisSize;
        return Tensor(featureInputs[0]->getPlacement(), TensorDescriptor(reference.getDataType(), outputDimensions));
    }

    void compileImpl() override {
        MultiConnectionLayer::compileImpl();
        THOR_THROW_IF_FALSE(featureOutputs.size() == 1 && featureOutputs[0].has_value());
        THOR_THROW_IF_FALSE(nextLayers.size() == 1);
        THOR_THROW_IF_FALSE(featureInputs[offsetsInputIndex].has_value());
        THOR_THROW_IF_FALSE(featureInputs[0].has_value());
        THOR_THROW_IF_FALSE(featureInputs[0]->getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);

        ScopedGpu scopedGpu(featureInputs[0]->getPlacement().getDeviceNum());
        splitTensorFeatureInputMemoriesArray_d = nullptr;
        splitTensorErrorOutputMemoriesArray_d = nullptr;
        axisElementsPerSplitTensor_d = nullptr;
        stridePerPackedTensorDimension_d = nullptr;
        stridePerSplitTensorDimension_d = nullptr;

        std::vector<void*> valuePointers(valueInputCount);
        for (uint32_t i = 0; i < valueInputCount; ++i) valuePointers[i] = featureInputs[i]->getMemPtr();
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&splitTensorFeatureInputMemoriesArray_d), valueInputCount * sizeof(void*)));
        CUDA_CHECK(cudaMemcpyAsync(splitTensorFeatureInputMemoriesArray_d, valuePointers.data(), valueInputCount * sizeof(void*),
                                   cudaMemcpyHostToDevice, streams[0].getStream()));

        if (errorInputs[0].has_value()) {
            discardedErrorOutputs.resize(valueInputCount);
            std::vector<void*> errorPointers(valueInputCount);
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&splitTensorErrorOutputMemoriesArray_d), valueInputCount * sizeof(void*)));
            for (uint32_t i = 0; i < valueInputCount; ++i) {
                if (errorOutputs[i].has_value()) {
                    errorPointers[i] = errorOutputs[i]->getMemPtr();
                } else {
                    discardedErrorOutputs[i] = featureInputs[i]->clone();
                    errorPointers[i] = discardedErrorOutputs[i]->getMemPtr();
                }
            }
            CUDA_CHECK(cudaMemcpyAsync(splitTensorErrorOutputMemoriesArray_d, errorPointers.data(), valueInputCount * sizeof(void*),
                                       cudaMemcpyHostToDevice, streams[0].getStream()));
        }

        std::vector<long> axisElements(valueInputCount);
        for (uint32_t i = 0; i < valueInputCount; ++i) {
            axisElements[i] = static_cast<long>(featureInputs[i]->getDescriptor().getDimensions()[axis]);
        }
        CUDA_CHECK(cudaMalloc(&axisElementsPerSplitTensor_d, valueInputCount * sizeof(long)));
        CUDA_CHECK(cudaMemcpyAsync(axisElementsPerSplitTensor_d, axisElements.data(), valueInputCount * sizeof(long),
                                   cudaMemcpyHostToDevice, streams[0].getStream()));

        const unsigned int numDimensions = featureInputs[0]->getDescriptor().getDimensions().size();
        std::vector<long> splitStrides(numDimensions * valueInputCount);
        for (uint32_t t = 0; t < valueInputCount; ++t) {
            splitStrides[t * numDimensions + (numDimensions - 1)] = 1;
            for (int d = static_cast<int>(numDimensions) - 2; d >= 0; --d) {
                splitStrides[t * numDimensions + d] = splitStrides[t * numDimensions + d + 1] *
                    static_cast<long>(featureInputs[t]->getDescriptor().getDimensions()[d + 1]);
            }
        }
        CUDA_CHECK(cudaMalloc(&stridePerSplitTensorDimension_d, splitStrides.size() * sizeof(long)));
        CUDA_CHECK(cudaMemcpyAsync(stridePerSplitTensorDimension_d, splitStrides.data(), splitStrides.size() * sizeof(long),
                                   cudaMemcpyHostToDevice, streams[0].getStream()));

        const auto outputDimensions = featureOutputs[0]->getDescriptor().getDimensions();
        std::vector<long> packedStrides(outputDimensions.size());
        packedStrides.back() = 1;
        for (int d = static_cast<int>(outputDimensions.size()) - 2; d >= 0; --d) {
            packedStrides[d] = static_cast<long>(outputDimensions[d + 1]) * packedStrides[d + 1];
        }
        CUDA_CHECK(cudaMalloc(&stridePerPackedTensorDimension_d, packedStrides.size() * sizeof(long)));
        CUDA_CHECK(cudaMemcpyAsync(stridePerPackedTensorDimension_d, packedStrides.data(), packedStrides.size() * sizeof(long),
                                   cudaMemcpyHostToDevice, streams[0].getStream()));
        streams[0].synchronize();

        for (uint32_t i = 0; i < featureInputs.size(); ++i) allFeatureInputTensorIds.insert(featureInputs[i]->getTensorId());
    }

    void initialize() override {
        MultiConnectionLayer::initialize();
        stillWaitingForFeatureInputTensors = allFeatureInputTensorIds;
        currentValidExampleCount = 0;
        batchCardinalitySet = false;
    }

    void infer(std::optional<Tensor>, std::optional<Tensor>, Stream, unsigned int) override {}
    void backProp(std::optional<Tensor>, std::optional<Tensor>, std::optional<Tensor>, Stream, unsigned int) override {}

    void forward(std::optional<Tensor> featureInput, bool validationPass, uint32_t runtimeBatchSize = 0) override {
        THOR_THROW_IF_FALSE(featureInput.has_value());
        // The fixed descriptor batchSize still bounds offsets[batchSize] for the
        // ragged kernels. runtimeBatchSize is valid-example metadata only; keep it
        // intact for downstream losses/optimizers and require every input port to agree.
        THOR_THROW_IF_FALSE(batchSize <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()));
        const uint32_t physicalBatchCapacity = static_cast<uint32_t>(batchSize);
        const uint32_t resolvedValidExampleCount =
            runtimeBatchSize == 0 ? physicalBatchCapacity : runtimeBatchSize;
        THOR_THROW_IF_FALSE(resolvedValidExampleCount >= 1);
        THOR_THROW_IF_FALSE(resolvedValidExampleCount <= physicalBatchCapacity);
        if (batchCardinalitySet) {
            THOR_THROW_IF_FALSE(currentValidExampleCount == resolvedValidExampleCount);
        } else {
            currentValidExampleCount = resolvedValidExampleCount;
            batchCardinalitySet = true;
        }

        auto it = stillWaitingForFeatureInputTensors.find(featureInput->getTensorId());
        THOR_THROW_IF_FALSE(it != stillWaitingForFeatureInputTensors.end());
        stillWaitingForFeatureInputTensors.erase(it);
        if (!stillWaitingForFeatureInputTensors.empty()) return;
        stillWaitingForFeatureInputTensors = allFeatureInputTensorIds;

        for (uint32_t i = 1; i < featureInputs.size(); ++i) streams[0].waitEvent(streams[i].putEvent());
        refreshValueInputMemoryArray(streams[0]);

        const TensorDescriptor& outputDescriptor = featureOutputs[0]->getDescriptor();
        const auto& outputDimensions = outputDescriptor.getDimensions();
        THOR_THROW_IF_FALSE(!outputDimensions.empty() && outputDimensions[0] > 0);
        const uint64_t elementsPerOutputValue = outputDescriptor.getTotalNumElements() / outputDimensions[0];
        THOR_THROW_IF_FALSE(outputDescriptor.getTotalNumElements() <= static_cast<uint64_t>(std::numeric_limits<long>::max()));
        const TensorDescriptor& offsetsDescriptor = featureInputs[offsetsInputIndex]->getDescriptor();
        launchRaggedConcatenate(
            featureOutputs[0]->getMemPtr(),
            splitTensorFeatureInputMemoriesArray_d,
            TensorDescriptor::getElementSizeInBytes(outputDescriptor.getDataType()),
            static_cast<long>(outputDescriptor.getTotalNumElements()),
            elementsPerOutputValue,
            outputDimensions.size(),
            valueInputCount,
            axis,
            axisElementsPerSplitTensor_d,
            stridePerPackedTensorDimension_d,
            stridePerSplitTensorDimension_d,
            featureInputs[offsetsInputIndex]->getMemPtr(),
            TensorDescriptor::getElementSizeInBytes(offsetsDescriptor.getDataType()),
            batchSize,
            streams[0]);

        nextLayers[0].value()->forward(featureOutputs[0], validationPass, currentValidExampleCount);
        currentValidExampleCount = 0;
        batchCardinalitySet = false;
    }

    void backward(std::optional<Tensor> errorInput, uint32_t runtimeBatchSize = 0) override {
        if (!errorInput.has_value()) return;
        THOR_THROW_IF_FALSE(batchSize <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()));
        const uint32_t physicalBatchCapacity = static_cast<uint32_t>(batchSize);
        const uint32_t resolvedValidExampleCount =
            runtimeBatchSize == 0 ? physicalBatchCapacity : runtimeBatchSize;
        THOR_THROW_IF_FALSE(resolvedValidExampleCount >= 1);
        THOR_THROW_IF_FALSE(resolvedValidExampleCount <= physicalBatchCapacity);
        if (splitTensorErrorOutputMemoriesArray_d != nullptr) {
            const TensorDescriptor& errorDescriptor = errorInput->getDescriptor();
            const auto& dimensions = errorDescriptor.getDimensions();
            const uint64_t elementsPerSourceValue = errorDescriptor.getTotalNumElements() / dimensions[0];
            const TensorDescriptor& offsetsDescriptor = featureInputs[offsetsInputIndex]->getDescriptor();
            launchRaggedSplit(
                splitTensorErrorOutputMemoriesArray_d,
                errorInput->getMemPtr(),
                TensorDescriptor::getElementSizeInBytes(errorDescriptor.getDataType()),
                static_cast<long>(errorDescriptor.getTotalNumElements()),
                elementsPerSourceValue,
                dimensions.size(),
                valueInputCount,
                axis,
                axisElementsPerSplitTensor_d,
                stridePerPackedTensorDimension_d,
                stridePerSplitTensorDimension_d,
                featureInputs[offsetsInputIndex]->getMemPtr(),
                TensorDescriptor::getElementSizeInBytes(offsetsDescriptor.getDataType()),
                batchSize,
                streams[0]);
        }

        Event readyEvent = streams[0].putEvent();
        for (uint32_t i = 0; i < valueInputCount; ++i) {
            if (i != 0) streams[i].waitEvent(readyEvent);
            if (previousLayers[i].has_value())
                previousLayers[i].value()->backward(errorOutputs[i], resolvedValidExampleCount);
        }
    }

    void cleanup() override {
        THOR_THROW_IF_FALSE(featureInputs[0].has_value());
        ScopedGpu scopedGpu(featureInputs[0]->getPlacement().getDeviceNum());
        if (splitTensorFeatureInputMemoriesArray_d != nullptr) CUDA_CHECK(cudaFree(splitTensorFeatureInputMemoriesArray_d));
        if (splitTensorErrorOutputMemoriesArray_d != nullptr) CUDA_CHECK(cudaFree(splitTensorErrorOutputMemoriesArray_d));
        if (axisElementsPerSplitTensor_d != nullptr) CUDA_CHECK(cudaFree(axisElementsPerSplitTensor_d));
        if (stridePerPackedTensorDimension_d != nullptr) CUDA_CHECK(cudaFree(stridePerPackedTensorDimension_d));
        if (stridePerSplitTensorDimension_d != nullptr) CUDA_CHECK(cudaFree(stridePerSplitTensorDimension_d));
        splitTensorFeatureInputMemoriesArray_d = nullptr;
        splitTensorErrorOutputMemoriesArray_d = nullptr;
        axisElementsPerSplitTensor_d = nullptr;
        stridePerPackedTensorDimension_d = nullptr;
        stridePerSplitTensorDimension_d = nullptr;
        discardedErrorOutputs.clear();
        MultiConnectionLayer::cleanup();
    }

    void connectToNextLayer(Layer *nextLayer, int driverConnectionType = 0, int loaderConnectionType = 0) override {
        (void)driverConnectionType;
        THOR_THROW_IF_FALSE(!running);
        nextLayers.push_back(nextLayer);
        featureOutputs.emplace_back(createFeatureOutputTensor());
        errorInputs.emplace_back(nextLayer->connectToPreviousLayer(
            this, featureOutputs.back(), streams[0], shouldConnectToBackPropErrorIn() && !isBackPropStub(), loaderConnectionType));
        if (!errorInputs.back().has_value()) {
            for (uint32_t i = 0; i < valueInputCount; ++i) {
                if (errorOutputs[i].has_value() && previousLayers[i].has_value())
                    previousLayers[i].value()->replaceErrorInput(errorOutputs[i], std::nullopt);
                errorOutputs[i] = std::nullopt;
            }
        }
        ensureNoDeviceCrossing();
    }

    std::optional<Tensor> connectToPreviousLayer(
        Layer *previousLayer, std::optional<Tensor> featureInput, Stream stream,
        bool backPropagateError, int connectionType) override {
        THOR_THROW_IF_FALSE(!running && featureInput.has_value() && previousLayer != nullptr);
        if (connectionType < 0 || static_cast<uint32_t>(connectionType) > offsetsInputIndex)
            throw std::logic_error("RaggedConcatenate connection type is outside its declared input range.");
        const uint32_t inputIndex = static_cast<uint32_t>(connectionType);
        if (featureInputs[inputIndex].has_value() || previousLayers[inputIndex].has_value())
            throw std::logic_error("RaggedConcatenate input port was connected more than once.");

        streams[inputIndex] = stream;
        previousLayers[inputIndex] = previousLayer;
        featureInputs[inputIndex] = featureInput;
        if (inputIndex < valueInputCount && backPropagateError && !isInferenceOnly())
            errorOutputs[inputIndex] = featureInput->clone();
        else
            errorOutputs[inputIndex] = std::nullopt;
        ensureNoDeviceCrossing();
        return errorOutputs[inputIndex];
    }

   private:
    struct ValueInputMemoryArrayRefreshArgs : public HostFunctionArgsBase { std::vector<void*> pointers; };
    static void releaseValueInputMemoryArrayRefresh(void*) {}

    void refreshValueInputMemoryArray(Stream stream) {
        auto args = std::make_unique<ValueInputMemoryArrayRefreshArgs>();
        args->pointers.resize(valueInputCount);
        for (uint32_t i = 0; i < valueInputCount; ++i) args->pointers[i] = featureInputs[i]->getMemPtr();
        CUDA_CHECK(cudaMemcpyAsync(splitTensorFeatureInputMemoriesArray_d, args->pointers.data(), valueInputCount * sizeof(void*),
                                   cudaMemcpyHostToDevice, stream));
        stream.enqueueHostFunction(&releaseValueInputMemoryArrayRefresh, std::move(args));
    }

    unsigned int axis;
    uint32_t valueInputCount;
    uint32_t offsetsInputIndex;
    uint64_t batchSize;
    void **splitTensorFeatureInputMemoriesArray_d = nullptr;
    void **splitTensorErrorOutputMemoriesArray_d = nullptr;
    long *axisElementsPerSplitTensor_d = nullptr;
    long *stridePerPackedTensorDimension_d = nullptr;
    long *stridePerSplitTensorDimension_d = nullptr;
    std::vector<std::optional<Tensor>> discardedErrorOutputs;
    std::set<uint64_t> allFeatureInputTensorIds;
    std::set<uint64_t> stillWaitingForFeatureInputTensors;
    uint32_t currentValidExampleCount = 0;
    bool batchCardinalitySet = false;
};

}  // namespace ThorImplementation
