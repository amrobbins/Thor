#pragma once

#include "DeepLearning/Implementation/Layers/Layer.h"

class Pooling : public Layer {
   public:
    enum class Type { MAX = 11, AVERAGE = 12 };

    virtual ~Pooling() {}

    Pooling(Type poolingType,
            int windowHeight,
            int windowWidth,
            int verticalStride,
            int horizontalStride,
            int batchSize,
            int numFeatures,
            int inputHeight,
            int inputWidth)
        : poolingType(poolingType),
          windowHeight(windowHeight),
          windowWidth(windowWidth),
          verticalStride(verticalStride),
          horizontalStride(horizontalStride),
          batchSize(batchSize),
          numFeatures(numFeatures),
          inputHeight(inputHeight),
          inputWidth(inputWidth) {
        assert(poolingType == Type::MAX || poolingType == Type::AVERAGE);
        assert(windowHeight > 0);
        assert(windowWidth > 0);
        assert(verticalStride > 0);
        assert(horizontalStride > 0);
        assert(batchSize > 0);
        assert(numFeatures > 0);
        assert(inputHeight > 0);
        assert(inputWidth > 0);
        assert(inputHeight >= windowHeight);
        assert(inputWidth >= windowWidth);
    }

    virtual void compile() {
        cudnnStatus_t cudnnStatus;

        assert(featureInput.isPresent());
        assert(featureOutput.isPresent());

        // Dimensions are NCHW
        vector<unsigned long> inputDimensions = featureInput.get().getDescriptor().getDimensions();
        assert(inputDimensions.size() == 4);
        assert(inputDimensions[0] == (uint32_t)batchSize);
        assert(inputDimensions[1] == (uint32_t)numFeatures);
        assert(inputDimensions[2] == (uint32_t)inputHeight);
        assert(inputDimensions[3] == (uint32_t)inputWidth);
        vector<unsigned long> outputDimensions = featureOutput.get().getDescriptor().getDimensions();
        int outputWidth = 1 + (inputWidth - windowWidth) / horizontalStride;
        int outputHeight = 1 + (inputHeight - windowHeight) / verticalStride;
        assert(outputDimensions.size() == 4);
        assert(outputDimensions[0] == (uint32_t)batchSize);
        assert(outputDimensions[1] == (uint32_t)numFeatures);
        assert(outputDimensions[2] == (uint32_t)outputHeight);
        assert(outputDimensions[3] == (uint32_t)outputWidth);

        ScopedGpu scopedGpu(featureInput.get().getPlacement().getDeviceNum());

        cudnnStatus = cudnnCreatePoolingDescriptor(&poolingDescriptor.get());
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
        cudnnStatus =
            cudnnSetPooling2dDescriptor(poolingDescriptor,
                                        poolingType == Type::MAX ? CUDNN_POOLING_MAX : CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
                                        CUDNN_NOT_PROPAGATE_NAN,
                                        windowHeight,
                                        windowWidth,
                                        0,
                                        0,
                                        verticalStride,
                                        horizontalStride);
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);

        featureInputDescriptor = cudnnTensorDescriptor_t();
        cudnnStatus = cudnnCreateTensorDescriptor(&featureInputDescriptor.get());
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
        cudnnStatus = cudnnSetTensor4dDescriptor(
            featureInputDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize, numFeatures, inputHeight, inputWidth);
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);

        featureOutputDescriptor = cudnnTensorDescriptor_t();
        cudnnStatus = cudnnCreateTensorDescriptor(&featureOutputDescriptor.get());
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
        cudnnStatus = cudnnSetTensor4dDescriptor(
            featureOutputDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize, numFeatures, outputHeight, outputWidth);
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
    }

    void cleanup() {
        cudnnStatus_t cudnnStatus;

        if (poolingDescriptor.isPresent()) {
            cudnnStatus = cudnnDestroyPoolingDescriptor(poolingDescriptor.get());
            assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
            poolingDescriptor.clear();
        }

        if (featureInputDescriptor.isPresent()) {
            cudnnStatus = cudnnDestroyTensorDescriptor(featureInputDescriptor.get());
            assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
            featureInputDescriptor.clear();
        }

        if (featureOutputDescriptor.isPresent()) {
            cudnnStatus = cudnnDestroyTensorDescriptor(featureOutputDescriptor.get());
            assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
            featureOutputDescriptor.clear();
        }
    }

    virtual void infer(Optional<Tensor> inputTensor, Optional<Tensor> outputTensor, Stream stream) {
        assert(inputTensor.isPresent());
        assert(outputTensor.isPresent());

        cudnnStatus_t cudnnStatus;
        cudnnStatus = cudnnPoolingForward(stream.getCudnnHandle(),
                                          poolingDescriptor,
                                          &ALPHA_NO_SCALE,
                                          featureInputDescriptor,
                                          inputTensor.get().getMemPtr(),
                                          &BETA_CLEAR,
                                          featureOutputDescriptor,
                                          outputTensor.get().getMemPtr());
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
    }

    virtual void backProp(Optional<Tensor> dataIn, Optional<Tensor> errorIn, Optional<Tensor> errorOut, Stream stream) {
        if (errorOut.isEmpty())
            return;
        assert(errorIn.isPresent());

        cudnnStatus_t cudnnStatus;
        cudnnStatus = cudnnPoolingBackward(stream.getCudnnHandle(),
                                           poolingDescriptor,
                                           &ALPHA_NO_SCALE,
                                           featureOutputDescriptor,
                                           featureOutput.get().getMemPtr(),
                                           featureOutputDescriptor,
                                           errorIn.get().getMemPtr(),
                                           featureInputDescriptor,
                                           dataIn.get().getMemPtr(),
                                           &BETA_CLEAR,
                                           featureInputDescriptor,
                                           errorOut.get().getMemPtr());
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
    }

   private:
    static const float ALPHA_NO_SCALE;
    static const float BETA_CLEAR;

    Type poolingType;
    int windowHeight;
    int windowWidth;
    int verticalStride;
    int horizontalStride;
    int batchSize;
    int numFeatures;
    int inputHeight;
    int inputWidth;

    Optional<cudnnPoolingDescriptor_t> poolingDescriptor;
    Optional<cudnnTensorDescriptor_t> featureOutputDescriptor;
    Optional<cudnnTensorDescriptor_t> featureInputDescriptor;
};
