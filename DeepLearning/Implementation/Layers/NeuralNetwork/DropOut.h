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

    virtual void compile() {
        cudnnStatus_t cudnnStatus;

        // The random state may not change between calls of cudnnDropoutForward(...) and cudnnDropoutBackward(...),
        // so this dropout layer can only be used for 1 input/output pair.
        assert(featureInput.isPresent());

        ScopedGpu scopedGpu(featureInput.get().getPlacement().getDeviceNum());

        vector<unsigned long> randomStateDimensions;
        cudnnStatus = cudnnDropoutGetStatesSize(stream.getCudnnHandle(), &randomStateBytes);
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
        randomStateDimensions.push_back(randomStateBytes);
        randomState = Tensor(featureInput.get().getPlacement(), TensorDescriptor(TensorDescriptor::DataType::UINT8, randomStateDimensions));

        cudnnStatus = cudnnCreateDropoutDescriptor(&dropoutDescriptor);
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);

        cudnnStatus = cudnnCreateTensorDescriptor(&cudnnTensorDescriptor);
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
        vector<unsigned long> tensorDimensions = featureInput.get().getDescriptor().getDimensions();
        // Tensors must have at least 4 dimensions and not more than CUDNN_DIM_MAX, per cudnn.
        // Unused dimensions will be set to size 1.
        // https://docs.nvidia.com/deeplearning/sdk/cudnn-api/index.html#cudnnSetTensorNdDescriptor
        assert(tensorDimensions.size() <= CUDNN_DIM_MAX);
        vector<int> dimensionsMin4;
        vector<int> noGapsStride;
        for (unsigned int i = 0; i < tensorDimensions.size(); ++i) {
            dimensionsMin4.push_back(tensorDimensions[i]);
            // no overflow:
            assert(dimensionsMin4.back() == (long)tensorDimensions[i]);
            noGapsStride.push_back(1);
        }

        while (dimensionsMin4.size() < 4) {
            dimensionsMin4.push_back(1);
            noGapsStride.push_back(1);
        }

        for (int i = (int)dimensionsMin4.size() - 2; i >= 0; --i) {
            noGapsStride[i] = noGapsStride[i + 1] * dimensionsMin4[i + 1];
        }

        cudnnStatus = cudnnSetTensorNdDescriptor(cudnnTensorDescriptor,
                                                 CudnnHelper::getCudnnDataType(featureInput.get().getDescriptor().getDataType()),
                                                 dimensionsMin4.size(),
                                                 dimensionsMin4.data(),
                                                 noGapsStride.data());
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);

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

        vector<unsigned long> reserveSpaceDimensions;
        cudnnStatus = cudnnDropoutGetReserveSpaceSize(cudnnTensorDescriptor, &reserveSpaceBytes);
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
        reserveSpaceDimensions.push_back(reserveSpaceBytes);
        reserveSpace =
            Tensor(featureInput.get().getPlacement(), TensorDescriptor(TensorDescriptor::DataType::UINT8, reserveSpaceDimensions));
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
