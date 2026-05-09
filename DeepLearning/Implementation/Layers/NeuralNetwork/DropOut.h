#pragma once

#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Implementation/Layers/Layer.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"

namespace ThorImplementation {

/**
 * Performs DropOut, and corresponding scaling, during training.
 * Returns the input tensor as the output tensor during inference.
 *
 * When instantiating a trained network for inference only, this layer should be skipped
 * (not instantiated as part of the network), to save memory and memory bandwidth.
 *
 * However, when it will be used in a network that is being trained, it will need to
 * support both training and inference modes
 */

class DropOut : public Layer {
   public:
    ~DropOut() override {}

    void setTrainingMode(bool training) { this->training = training; }

    DropOut(float probabilityOfDroppingOut, bool training) {
        THOR_THROW_IF_FALSE(probabilityOfDroppingOut >= 0.0f);
        THOR_THROW_IF_FALSE(probabilityOfDroppingOut <= 1.0f);
        this->probabilityOfDroppingOut = probabilityOfDroppingOut;
        this->training = training;
        std::random_device rd;
        randomSeed = Tensor::Tensor::getThreadIdHash64(rd());
    }

    void seed(uint64_t seed) {
        THOR_THROW_IF_FALSE(!compiled);
        this->randomSeed = seed;
    }

    static size_t getReservedSpaceSizeInBytes(std::vector<unsigned long> featureInputDimensions, TensorDescriptor::DataType dataType) {
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

        cudnnStatus_t cudnnStatus;

        // The random state may not change between calls of cudnnDropoutForward(...) and cudnnDropoutBackward(...),
        // so this dropout layer can only be used for 1 input/output pair.
        THOR_THROW_IF_FALSE(featureInput.isPresent());

        ScopedGpu scopedGpu(featureInput.get().getPlacement().getDeviceNum());

        randomStateBytes = getRandomStateSizeInBytes(stream.getCudnnHandle());
        randomState = Tensor(featureInput.get().getPlacement(), TensorDescriptor(TensorDescriptor::DataType::UINT8, {randomStateBytes}));

        cudnnStatus = cudnnCreateDropoutDescriptor(&dropoutDescriptor);
        THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);

        cudnnTensorDescriptor = createCudnnTensorDescriptor(featureInput.get().getDescriptor().getDimensions(),
                                                            featureInput.get().getDescriptor().getDataType());
        reserveSpaceBytes = getReservedSpaceSizeInBytes(featureInput.get().getDescriptor().getDimensions(),
                                                        featureInput.get().getDescriptor().getDataType());
        reserveSpace = Tensor(featureInput.get().getPlacement(), TensorDescriptor(TensorDescriptor::DataType::UINT8, {reserveSpaceBytes}));

        mtx.lock();
        cudnnStatus = cudnnSetDropoutDescriptor(
            dropoutDescriptor, stream.getCudnnHandle(), probabilityOfDroppingOut, randomState.getMemPtr(), randomStateBytes, randomSeed);
        THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);

        mtx.unlock();
    }

    void cleanup() {
        if (compiled) {
            cudnnStatus_t cudnnStatus;

            cudnnStatus = cudnnDestroyDropoutDescriptor(dropoutDescriptor);
            THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);
            cudnnStatus = cudnnDestroyTensorDescriptor(cudnnTensorDescriptor);
            THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);
        }
        compiled = false;
    }

    void infer(Optional<Tensor> inputTensor, Optional<Tensor> outputTensor, Stream stream) override {
        THOR_THROW_IF_FALSE(inputTensor.isPresent());
        THOR_THROW_IF_FALSE(outputTensor.isPresent());

        if (training) {
            THOR_THROW_IF_FALSE(inputTensor.get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
            ScopedGpu scopedGpu(inputTensor.get().getPlacement().getDeviceNum());

            cudnnStatus_t cudnnStatus;
            cudnnStatus = cudnnDropoutForward(stream.getCudnnHandle(),
                                              dropoutDescriptor,
                                              cudnnTensorDescriptor,
                                              inputTensor.get().getMemPtr(),
                                              cudnnTensorDescriptor,
                                              outputTensor.get().getMemPtr(),
                                              reserveSpace.getMemPtr(),
                                              reserveSpaceBytes);
            THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);
        } else {
            outputTensor.get().copyFromAsync(inputTensor, stream);
        }
    }

    void backProp(Optional<Tensor> dataIn, Optional<Tensor> errorIn, Optional<Tensor> errorOut, Stream stream) override {
        if (errorOut.isEmpty())
            return;
        THOR_THROW_IF_FALSE(errorIn.isPresent());
        THOR_THROW_IF_FALSE(training);

        THOR_THROW_IF_FALSE(errorIn.get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        ScopedGpu scopedGpu(errorIn.get().getPlacement().getDeviceNum());

        cudnnStatus_t cudnnStatus;
        cudnnStatus = cudnnDropoutBackward(stream.getCudnnHandle(),
                                           dropoutDescriptor,
                                           cudnnTensorDescriptor,
                                           errorIn.get().getMemPtr(),
                                           cudnnTensorDescriptor,
                                           errorOut.get().getMemPtr(),
                                           reserveSpace.getMemPtr(),
                                           reserveSpaceBytes);
        THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);
    }

    bool isTrainingMode() { return training; }

    float getDropOutRate() const { return probabilityOfDroppingOut; }

   private:
    float probabilityOfDroppingOut;
    bool training;

    static std::mutex mtx;
    static uint64_t randomSeed;

    Tensor randomState;
    size_t randomStateBytes;
    Tensor reserveSpace;
    size_t reserveSpaceBytes;

    cudnnDropoutDescriptor_t dropoutDescriptor = nullptr;
    cudnnTensorDescriptor_t cudnnTensorDescriptor = nullptr;
};

}  // namespace ThorImplementation
