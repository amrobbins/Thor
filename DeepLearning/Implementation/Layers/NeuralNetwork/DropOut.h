#pragma once

#include <optional>
#include <stdexcept>
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Implementation/Layers/Layer.h"
#include "DeepLearning/Implementation/Layers/TrainingDropoutControllable.h"
#include "DeepLearning/Implementation/Layers/NeuralNetwork/DropOutKernel.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"

namespace ThorImplementation {

/**
 * Performs DropOut, and corresponding scaling, during training.
 * Inference aliases the input tensor as the output tensor for both dense and ragged
 * execution. A zero-rate training layer is likewise a true identity. Row-partition
 * state lives on the offsets-owned RowPartitionRuntime, so aliasing packed values does
 * not require DropOut to copy or mutate ragged metadata.
 *
 * When instantiating a trained network for inference only, this layer should be skipped
 * (not instantiated as part of the network), to save memory and memory bandwidth.
 *
 * However, when it will be used in a network that is being trained, it will need to
 * support both training and inference modes
 */

class DropOut : public Layer, public TrainingDropoutControllable {
   public:
    struct RaggedConfiguration {
        uint64_t fullCapacityRows;
        uint64_t elementsPerValue;
    };

    ~DropOut() override {}

    void setTrainingMode(bool training) { this->training = training; }

    void setTrainingDropoutEnabled(bool enabled) override { trainingDropoutEnabled = enabled; }
    [[nodiscard]] bool isTrainingDropoutEnabled() const override { return trainingDropoutEnabled; }

    void forward(std::optional<Tensor> arrivingInput, bool validationPass, uint32_t batchSize = 0) override {
        if (!raggedConfiguration.has_value()) {
            forwardValues(arrivingInput, validationPass, batchSize);
            return;
        }

        THOR_THROW_IF_FALSE(arrivingInput.has_value());
        THOR_THROW_IF_FALSE(featureInput.has_value());
        THOR_THROW_IF_FALSE(rowPartitionInput.has_value());
        const bool valuesArrival = arrivingInput.value() == featureInput.value();
        const bool partitionArrival = arrivingInput.value() == rowPartitionInput.value();
        THOR_THROW_IF_FALSE(valuesArrival || partitionArrival);

        // Inference-only and zero-rate DropOut are physical identities. Preserve the
        // zero-copy path for ragged values too: the structural offsets edge remains
        // connected in the graph, but identity execution neither waits for it nor
        // queries RowPartitionRuntime because no packed rows are read or written.
        if (!training || probabilityOfDroppingOut == 0.0f) {
            if (partitionArrival) return;
            forwardValues(featureInput, validationPass, batchSize);
            return;
        }

        if (!pendingRaggedValidationPass.has_value()) {
            pendingRaggedValidationPass = validationPass;
            pendingRaggedBatchSize = batchSize;
        } else {
            THOR_THROW_IF_FALSE(pendingRaggedValidationPass.value() == validationPass);
            THOR_THROW_IF_FALSE(pendingRaggedBatchSize.value() == batchSize);
        }

        if (valuesArrival) raggedValuesArrived = true;
        if (partitionArrival) raggedPartitionArrived = true;
        if (!raggedValuesArrived || !raggedPartitionArrived) return;

        const bool resolvedValidationPass = pendingRaggedValidationPass.value();
        const uint32_t resolvedBatchSize = pendingRaggedBatchSize.value();
        raggedValuesArrived = false;
        raggedPartitionArrived = false;
        pendingRaggedValidationPass.reset();
        pendingRaggedBatchSize.reset();
        forwardValues(featureInput, resolvedValidationPass, resolvedBatchSize);
    }

    DropOut(float probabilityOfDroppingOut,
            bool training,
            bool trainingDropoutEnabled = true,
            std::optional<RaggedConfiguration> raggedConfiguration = std::nullopt)
        : raggedConfiguration(raggedConfiguration) {
        THOR_THROW_IF_FALSE(probabilityOfDroppingOut >= 0.0f);
        THOR_THROW_IF_FALSE(probabilityOfDroppingOut <= 1.0f);
        if (this->raggedConfiguration.has_value()) {
            THOR_THROW_IF_FALSE(this->raggedConfiguration->fullCapacityRows > 0);
            THOR_THROW_IF_FALSE(this->raggedConfiguration->elementsPerValue > 0);
        }
        this->probabilityOfDroppingOut = probabilityOfDroppingOut;
        this->training = training;
        this->trainingDropoutEnabled = trainingDropoutEnabled;
        std::random_device rd;
        randomSeed = Tensor::Tensor::getThreadIdHash64(rd());
    }

    std::optional<Tensor> connectToPreviousLayer(
        Layer *previousLayer, std::optional<Tensor> connectedInput, Stream connectedStream, bool backPropagateError, int connectionType = 0) override {
        if (!raggedConfiguration.has_value()) {
            THOR_THROW_IF_FALSE(connectionType == 0);
            return Layer::connectToPreviousLayer(previousLayer, connectedInput, connectedStream, backPropagateError, connectionType);
        }

        if (connectionType == 0) {
            std::optional<Tensor> result =
                Layer::connectToPreviousLayer(previousLayer, connectedInput, connectedStream, backPropagateError, connectionType);
            if (rowPartitionInput.has_value()) {
                THOR_THROW_IF_FALSE(featureInput.has_value());
                THOR_THROW_IF_FALSE(featureInput->getPlacement() == rowPartitionInput->getPlacement());
            }
            return result;
        }
        if (connectionType != 1) {
            throw std::runtime_error("Ragged DropOut received an unknown physical input port.");
        }
        THOR_THROW_IF_FALSE(!compiled);
        THOR_THROW_IF_FALSE(connectedInput.has_value());
        THOR_THROW_IF_FALSE(!rowPartitionInput.has_value());
        rowPartitionInput = connectedInput.value();
        if (featureInput.has_value()) {
            THOR_THROW_IF_FALSE(featureInput->getPlacement() == rowPartitionInput->getPlacement());
        }
        // The partition is a forward-only structural dependency. Gradients flow
        // exclusively through packed values, never through offsets.
        return std::nullopt;
    }

    std::optional<Tensor> createFeatureOutputTensor() override {
        THOR_THROW_IF_FALSE(featureInput.has_value());
        if (!training || probabilityOfDroppingOut == 0.0f) {
            // Preserve a distinct logical API tensor while aliasing the physical
            // feature storage.  Downstream layers therefore receive the input
            // directly and no dense output allocation or copy is required.
            return featureInput.value();
        }
        return Layer::createFeatureOutputTensor();
    }

    void connectToNextLayer(Layer *nextLayer, int driverConnectionType = 0, int loaderConnectionType = 0) override {
        Layer::connectToNextLayer(nextLayer, driverConnectionType, loaderConnectionType);
        if (!training || probabilityOfDroppingOut == 0.0f)
            fuseBackwardIdentityAlias();
    }

    void seed(uint64_t seed) {
        THOR_THROW_IF_FALSE(!compiled);
        this->randomSeed = seed;
    }

    static bool usesNativeKernel(DataType dataType) { return dataType == DataType::BF16; }
    static bool nativeKernelSupportsDataType(DataType dataType) {
        return dataType == DataType::FP16 || dataType == DataType::FP32 || dataType == DataType::BF16;
    }
    [[nodiscard]] bool isRagged() const { return raggedConfiguration.has_value(); }

    static size_t getNativeReserveSpaceSizeInBytes(const std::vector<unsigned long> &featureInputDimensions) {
        size_t numElements = 1;
        for (unsigned long dimension : featureInputDimensions)
            numElements *= dimension;
        return numElements * sizeof(uint8_t);
    }

    static size_t getReservedSpaceSizeInBytes(std::vector<unsigned long> featureInputDimensions, DataType dataType) {
        size_t numBytes;

        cudnnTensorDescriptor_t descriptor = createCudnnTensorDescriptor(featureInputDimensions, dataType);
        cudnnStatus_t cudnnStatus = cudnnDropoutGetReserveSpaceSize(descriptor, &numBytes);
        THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);
        cudnnStatus = cudnnDestroyTensorDescriptor(descriptor);
        THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);
        return numBytes;
    }

    static size_t getRandomStateSizeInBytes(cudnnHandle_t cudnnHandle) {
        size_t numBytes;
        cudnnStatus_t cudnnStatus = cudnnDropoutGetStatesSize(cudnnHandle, &numBytes);
        THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);
        return numBytes;
    }

    void compileImpl() override {
        Layer::compileImpl();

        // The random state or keep mask may not change between a training
        // forward and its matching backward, so a DropOut instance supports one
        // active input/output pair at a time.
        THOR_THROW_IF_FALSE(featureInput.has_value());

        const DataType dataType = featureInput.value().getDescriptor().getDataType();
        if (raggedConfiguration.has_value()) {
            THOR_THROW_IF_FALSE(nativeKernelSupportsDataType(dataType));
            THOR_THROW_IF_FALSE(rowPartitionInput.has_value());
            validateRaggedTensorShape(featureInput.value());
            validateRowPartitionTensor(rowPartitionInput.value());
            if (!training || probabilityOfDroppingOut == 0.0f)
                return;
            ScopedGpu scopedGpu(featureInput.value().getPlacement().getDeviceNum());
            reserveSpaceBytes = featureInput.value().getTotalNumElements();
            reserveSpace = Tensor(featureInput.value().getPlacement(), TensorDescriptor(DataType::UINT8, {reserveSpaceBytes}));
            return;
        }

        if (!training || probabilityOfDroppingOut == 0.0f)
            return;

        ScopedGpu scopedGpu(featureInput.value().getPlacement().getDeviceNum());
        if (usesNativeKernel(dataType)) {
            reserveSpaceBytes =
                getNativeReserveSpaceSizeInBytes(featureInput.value().getDescriptor().getDimensions());
            reserveSpace =
                Tensor(featureInput.value().getPlacement(), TensorDescriptor(DataType::UINT8, {reserveSpaceBytes}));
            return;
        }

        cudnnStatus_t cudnnStatus;
        randomStateBytes = getRandomStateSizeInBytes(stream.getCudnnHandle());
        randomState = Tensor(featureInput.value().getPlacement(), TensorDescriptor(DataType::UINT8, {randomStateBytes}));

        cudnnStatus = cudnnCreateDropoutDescriptor(&dropoutDescriptor);
        THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);

        cudnnTensorDescriptor = createCudnnTensorDescriptor(featureInput.value().getDescriptor().getDimensions(), dataType);
        reserveSpaceBytes =
            getReservedSpaceSizeInBytes(featureInput.value().getDescriptor().getDimensions(), dataType);
        reserveSpace = Tensor(featureInput.value().getPlacement(), TensorDescriptor(DataType::UINT8, {reserveSpaceBytes}));

        std::lock_guard<std::mutex> lock(mtx);
        cudnnStatus = cudnnSetDropoutDescriptor(
            dropoutDescriptor, stream.getCudnnHandle(), probabilityOfDroppingOut, randomState.getMemPtr(), randomStateBytes, randomSeed);
        THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);
    }

    void cleanup() override {
        if (dropoutDescriptor != nullptr) {
            cudnnStatus_t cudnnStatus = cudnnDestroyDropoutDescriptor(dropoutDescriptor);
            THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);
            dropoutDescriptor = nullptr;
        }
        if (cudnnTensorDescriptor != nullptr) {
            cudnnStatus_t cudnnStatus = cudnnDestroyTensorDescriptor(cudnnTensorDescriptor);
            THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);
            cudnnTensorDescriptor = nullptr;
        }
        Layer::cleanup();
    }

    void infer(std::optional<Tensor> inputTensor, std::optional<Tensor> outputTensor, Stream stream) override {
        THOR_THROW_IF_FALSE(inputTensor.has_value());
        THOR_THROW_IF_FALSE(outputTensor.has_value());

        if (raggedConfiguration.has_value()) {
            validateRaggedTensorShape(inputTensor.value());
            if (outputTensor.value() == inputTensor.value()) {
                THOR_THROW_IF_FALSE(!applyDropoutThisForward);
                return;
            }
            const uint64_t activeRows = requireRaggedActiveValueCount();
            const uint64_t activeElements = activeRows * raggedConfiguration->elementsPerValue;
            raggedActiveValueCountForBackward = activeRows;

            if (outputTensor.value() != inputTensor.value()) {
                if (applyDropoutThisForward) {
                    const uint64_t forwardSequence = nativeForwardSequence++;
                    if (activeElements > 0) {
                        THOR_THROW_IF_FALSE(inputTensor.value().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
                        ScopedGpu scopedGpu(inputTensor.value().getPlacement().getDeviceNum());
                        launchDropOutForward(inputTensor.value().getMemPtr(),
                                             outputTensor.value().getMemPtr(),
                                             static_cast<uint8_t *>(reserveSpace.getMemPtr()),
                                             inputTensor.value().getDescriptor().getDataType(),
                                             activeElements,
                                             probabilityOfDroppingOut,
                                             randomSeed,
                                             forwardSequence,
                                             stream);
                    }
                } else {
                    copyActivePrefix(inputTensor.value(), outputTensor.value(), activeElements, stream);
                }
                zeroRaggedInactiveTail(outputTensor.value(), activeElements, stream);
            }
            return;
        }

        if (applyDropoutThisForward) {
            THOR_THROW_IF_FALSE(inputTensor.value().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
            ScopedGpu scopedGpu(inputTensor.value().getPlacement().getDeviceNum());

            const DataType dataType = inputTensor.value().getDescriptor().getDataType();
            if (usesNativeKernel(dataType)) {
                launchBfloat16DropOutForward(inputTensor.value().getMemPtr(),
                                             outputTensor.value().getMemPtr(),
                                             static_cast<uint8_t *>(reserveSpace.getMemPtr()),
                                             inputTensor.value().getTotalNumElements(),
                                             probabilityOfDroppingOut,
                                             randomSeed,
                                             nativeForwardSequence++,
                                             stream);
            } else {
                const cudnnStatus_t cudnnStatus = cudnnDropoutForward(stream.getCudnnHandle(),
                                                                     dropoutDescriptor,
                                                                     cudnnTensorDescriptor,
                                                                     inputTensor.value().getMemPtr(),
                                                                     cudnnTensorDescriptor,
                                                                     outputTensor.value().getMemPtr(),
                                                                     reserveSpace.getMemPtr(),
                                                                     reserveSpaceBytes);
                THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);
            }
        } else if (outputTensor.value() != inputTensor.value()) {
            outputTensor.value().copyFromAsync(inputTensor.value(), stream);
        }
    }

    void backProp(std::optional<Tensor> dataIn, std::optional<Tensor> errorIn, std::optional<Tensor> errorOut, Stream stream) override {
        if (!errorOut.has_value()) {
            applyDropoutForBackward = false;
            return;
        }
        THOR_THROW_IF_FALSE(errorIn.has_value());
        THOR_THROW_IF_FALSE(training);

        const bool applyDropout = applyDropoutForBackward;
        applyDropoutForBackward = false;

        if (raggedConfiguration.has_value()) {
            if (probabilityOfDroppingOut == 0.0f) {
                THOR_THROW_IF_FALSE(errorOut.value() == errorIn.value());
                raggedActiveValueCountForBackward.reset();
                return;
            }
            if (!raggedActiveValueCountForBackward.has_value())
                throw std::runtime_error("Ragged DropOut backward requires a preceding training forward pass.");
            const uint64_t activeRows = raggedActiveValueCountForBackward.value();
            raggedActiveValueCountForBackward.reset();
            THOR_THROW_IF_FALSE(activeRows <= raggedConfiguration->fullCapacityRows);
            const uint64_t activeElements = activeRows * raggedConfiguration->elementsPerValue;

            if (applyDropout) {
                if (activeElements > 0) {
                    THOR_THROW_IF_FALSE(errorIn.value().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
                    ScopedGpu scopedGpu(errorIn.value().getPlacement().getDeviceNum());
                    launchDropOutBackward(errorIn.value().getMemPtr(),
                                          errorOut.value().getMemPtr(),
                                          static_cast<const uint8_t *>(reserveSpace.getMemPtr()),
                                          errorIn.value().getDescriptor().getDataType(),
                                          activeElements,
                                          probabilityOfDroppingOut,
                                          stream);
                }
            } else if (errorOut.value() != errorIn.value()) {
                copyActivePrefix(errorIn.value(), errorOut.value(), activeElements, stream);
            }
            if (errorOut.value() != errorIn.value()) {
                zeroRaggedInactiveTail(errorOut.value(), activeElements, stream);
            }
            return;
        }

        if (probabilityOfDroppingOut == 0.0f) {
            THOR_THROW_IF_FALSE(errorOut.value() == errorIn.value());
            return;
        }

        if (!applyDropout) {
            if (errorOut.value() != errorIn.value()) {
                errorOut.value().copyFromAsync(errorIn.value(), stream);
            }
            return;
        }

        THOR_THROW_IF_FALSE(errorIn.value().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        ScopedGpu scopedGpu(errorIn.value().getPlacement().getDeviceNum());

        const DataType dataType = errorIn.value().getDescriptor().getDataType();
        if (usesNativeKernel(dataType)) {
            launchBfloat16DropOutBackward(errorIn.value().getMemPtr(),
                                          errorOut.value().getMemPtr(),
                                          static_cast<const uint8_t *>(reserveSpace.getMemPtr()),
                                          errorIn.value().getTotalNumElements(),
                                          probabilityOfDroppingOut,
                                          stream);
        } else {
            const cudnnStatus_t cudnnStatus = cudnnDropoutBackward(stream.getCudnnHandle(),
                                                                  dropoutDescriptor,
                                                                  cudnnTensorDescriptor,
                                                                  errorIn.value().getMemPtr(),
                                                                  cudnnTensorDescriptor,
                                                                  errorOut.value().getMemPtr(),
                                                                  reserveSpace.getMemPtr(),
                                                                  reserveSpaceBytes);
            THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);
        }
    }

    bool isTrainingMode() { return training; }

    float getDropOutRate() const { return probabilityOfDroppingOut; }

   private:
    void validateRaggedTensorShape(const Tensor& tensor) const {
        THOR_THROW_IF_FALSE(raggedConfiguration.has_value());
        const uint64_t totalElements = tensor.getTotalNumElements();
        THOR_THROW_IF_FALSE(totalElements % raggedConfiguration->fullCapacityRows == 0);
        THOR_THROW_IF_FALSE(totalElements / raggedConfiguration->fullCapacityRows == raggedConfiguration->elementsPerValue);
    }

    void validateRowPartitionTensor(const Tensor& offsets) const {
        THOR_THROW_IF_FALSE(raggedConfiguration.has_value());
        const TensorDescriptor descriptor = offsets.getDescriptor();
        if (descriptor.getNumDimensions() != 1 || descriptor.getDimensions()[0] == 0 ||
            !RowPartitionDescriptor::isValidOffsetsDataType(descriptor.getDataType())) {
            throw std::runtime_error("Ragged DropOut row-partition offsets input is not canonical.");
        }
    }

    [[nodiscard]] uint64_t requireRaggedActiveValueCount() const {
        THOR_THROW_IF_FALSE(raggedConfiguration.has_value());
        if (!rowPartitionInput.has_value())
            throw std::runtime_error("Ragged DropOut row-partition offsets input is not connected.");
        validateRowPartitionTensor(rowPartitionInput.value());
        const TensorDescriptor descriptor = rowPartitionInput->getDescriptor();
        const uint64_t batchSize = descriptor.getDimensions()[0] - 1;
        RowPartitionRuntime rowPartition(
            rowPartitionInput.value(),
            RowPartitionDescriptor(batchSize, raggedConfiguration->fullCapacityRows, descriptor.getDataType()));
        const uint64_t activeRows = rowPartition.requireHostActiveValueCount();
        if (activeRows > raggedConfiguration->fullCapacityRows)
            throw std::runtime_error("Ragged DropOut active row count exceeds packed capacity.");
        return activeRows;
    }

    void forwardValues(std::optional<Tensor> valuesInput, bool validationPass, uint32_t batchSize) {
        // Dropout is active only for gradient-training passes. Validation and
        // inference must observe the deterministic, unmasked network. Capture
        // the exact decision for the matching backward pass so a later control
        // change cannot pair an identity forward with a masked backward.
        const bool previousApplyDropout = applyDropoutThisForward;
        const bool applyDropout =
            training && trainingDropoutEnabled && !validationPass && probabilityOfDroppingOut > 0.0f;
        applyDropoutThisForward = applyDropout;
        try {
            Layer::forward(valuesInput, validationPass, batchSize);
        } catch (...) {
            applyDropoutThisForward = previousApplyDropout;
            throw;
        }
        if (training && !validationPass) {
            applyDropoutForBackward = applyDropout;
        }
        applyDropoutThisForward = previousApplyDropout;
    }

    static void copyActivePrefix(const Tensor& source, Tensor destination, uint64_t activeElements, Stream stream) {
        if (activeElements == 0 || source == destination) return;
        Tensor sourcePrefix = source.aliasView({activeElements}, {1}, 0);
        Tensor destinationPrefix = destination.aliasView({activeElements}, {1}, 0);
        destinationPrefix.copyFromAsync(sourcePrefix, stream);
    }

    static void zeroRaggedInactiveTail(Tensor tensor, uint64_t activeElements, Stream stream) {
        THOR_THROW_IF_FALSE(activeElements <= tensor.getTotalNumElements());
        const uint64_t tailElements = tensor.getTotalNumElements() - activeElements;
        if (tailElements == 0) return;
        Tensor tail = tensor.aliasView({tailElements}, {1}, activeElements);
        tail.memsetAsync(stream, 0);
    }

    void fuseBackwardIdentityAlias() {
        if (!errorInput.has_value() || !errorOutput.has_value())
            return;
        if (previousLayer.has_value())
            previousLayer.value()->replaceErrorInput(errorOutput, errorInput);
        errorOutput = errorInput;
    }

    float probabilityOfDroppingOut;
    bool training;
    bool trainingDropoutEnabled = true;
    bool applyDropoutThisForward = false;
    bool applyDropoutForBackward = false;

    static std::mutex mtx;
    uint64_t randomSeed = 0;
    uint64_t nativeForwardSequence = 0;
    std::optional<RaggedConfiguration> raggedConfiguration;
    std::optional<uint64_t> raggedActiveValueCountForBackward;
    std::optional<Tensor> rowPartitionInput;
    bool raggedValuesArrived = false;
    bool raggedPartitionArrived = false;
    std::optional<bool> pendingRaggedValidationPass;
    std::optional<uint32_t> pendingRaggedBatchSize;

    Tensor randomState;
    size_t randomStateBytes;
    Tensor reserveSpace;
    size_t reserveSpaceBytes;

    cudnnDropoutDescriptor_t dropoutDescriptor = nullptr;
    cudnnTensorDescriptor_t cudnnTensorDescriptor = nullptr;
};

}  // namespace ThorImplementation
