#pragma once

#include "DeepLearning/Implementation/Layers/Layer.h"

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
    virtual ~DropOut() {}

    void setTrainingMode(bool training) {
        assert(running == false);
        this->training = training;
    }

    DropOut(float probabilityOfDroppingOut, bool training) {
        assert(probabilityOfDroppingOut >= 0.0f);
        assert(probabilityOfDroppingOut <= 1.0f);
        this->probabilityOfDroppingOut = probabilityOfDroppingOut;
        this->training = training;
    }

    static size_t getReservedSpaceSizeInBytes(vector<unsigned long> featureInputDimensions, TensorDescriptor::DataType dataType) {
        size_t numBytes;

        cudnnTensorDescriptor_t descriptor = createCudnnTensorDescriptor(featureInputDimensions, dataType);
        cudnnStatus_t cudnnStatus = cudnnDropoutGetReserveSpaceSize(descriptor, &numBytes);
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
        cudnnStatus = cudnnDestroyTensorDescriptor(descriptor);
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
        return numBytes;
    }

    static size_t getRandomStateSizeInBytes(cudnnHandle_t cudnnHandle) {
        size_t numBytes;
        cudnnStatus_t cudnnStatus = cudnnDropoutGetStatesSize(cudnnHandle, &numBytes);
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
        return numBytes;
    }

    virtual void compile() {
        cudnnStatus_t cudnnStatus;

        // The random state may not change between calls of cudnnDropoutForward(...) and cudnnDropoutBackward(...),
        // so this dropout layer can only be used for 1 input/output pair.
        assert(featureInput.isPresent());

        ScopedGpu scopedGpu(featureInput.get().getPlacement().getDeviceNum());

        randomStateBytes = getRandomStateSizeInBytes(stream.getCudnnHandle());
        randomState = Tensor(featureInput.get().getPlacement(), TensorDescriptor(TensorDescriptor::DataType::UINT8, {randomStateBytes}));

        cudnnStatus = cudnnCreateDropoutDescriptor(&dropoutDescriptor);
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);

        cudnnTensorDescriptor = createCudnnTensorDescriptor(featureInput.get().getDescriptor().getDimensions(),
                                                            featureInput.get().getDescriptor().getDataType());
        reserveSpaceBytes = getReservedSpaceSizeInBytes(featureInput.get().getDescriptor().getDimensions(),
                                                        featureInput.get().getDescriptor().getDataType());
        reserveSpace = Tensor(featureInput.get().getPlacement(), TensorDescriptor(TensorDescriptor::DataType::UINT8, {reserveSpaceBytes}));

        mtx.lock();

        uint64_t longRand = 0;
        for (int i = 0; i < 4; ++i) {
            longRand |= (rand() & 0xFFFF);
            longRand <<= 16;
        }
        seed += longRand;

        cudnnStatus = cudnnSetDropoutDescriptor(
            dropoutDescriptor, stream.getCudnnHandle(), probabilityOfDroppingOut, randomState.getMemPtr(), randomStateBytes, seed);
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);

        mtx.unlock();
    }

    void cleanup() {
        cudnnStatus_t cudnnStatus;

        cudnnStatus = cudnnDestroyDropoutDescriptor(dropoutDescriptor);
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
        cudnnStatus = cudnnDestroyTensorDescriptor(cudnnTensorDescriptor);
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
    }

    virtual void infer(Optional<Tensor> inputTensor, Optional<Tensor> outputTensor, Stream stream) {
        assert(inputTensor.isPresent());
        assert(outputTensor.isPresent());

        if (training) {
            assert(inputTensor.get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
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
            assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
        } else {
            outputTensor.get().copyFromAsync(inputTensor, stream);
        }
    }

    virtual void backProp(Optional<Tensor> dataIn, Optional<Tensor> errorIn, Optional<Tensor> errorOut, Stream stream) {
        if (errorOut.isEmpty())
            return;
        assert(errorIn.isPresent());

        if (training) {
            assert(errorIn.get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
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
            assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
        } else {
            errorOut.get().copyFromAsync(errorIn, stream);
        }
    }

    bool getTrainingMode(bool training) { return training; }

   private:
    float probabilityOfDroppingOut;
    bool training;

    static mutex mtx;
    static uint64_t seed;

    Tensor randomState;
    size_t randomStateBytes;
    Tensor reserveSpace;
    size_t reserveSpaceBytes;

    cudnnDropoutDescriptor_t dropoutDescriptor;
    cudnnTensorDescriptor_t cudnnTensorDescriptor;
};

}  // namespace ThorImplementation
