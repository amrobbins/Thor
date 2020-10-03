#pragma once

#include "DeepLearning/Implementation/Layers/Layer.h"

namespace ThorImplementation {

class Pooling : public Layer {
   public:
    enum class Type { MAX = 11, AVERAGE = 12 };

    virtual ~Pooling() {}

    Pooling(Type poolingType,
            int windowHeight,
            int windowWidth,
            int verticalStride,
            int horizontalStride,
            int verticalPadding,
            int horizontalPadding)
        : poolingType(poolingType),
          windowHeight(windowHeight),
          windowWidth(windowWidth),
          verticalStride(verticalStride),
          horizontalStride(horizontalStride),
          verticalPadding(verticalPadding),
          horizontalPadding(horizontalPadding) {
        assert(poolingType == Type::MAX || poolingType == Type::AVERAGE);
        assert(windowHeight > 0);
        assert(windowWidth > 0);
        assert(verticalStride > 0);
        assert(horizontalStride > 0);
    }

    virtual void compile() {
        cudnnStatus_t cudnnStatus;

        assert(featureInput.isPresent());
        assert(featureOutput.isPresent());

        // Dimensions are NCHW
        vector<unsigned long> inputDimensions = featureInput.get().getDescriptor().getDimensions();
        assert(inputDimensions.size() == 4);
        batchSize = inputDimensions[0];
        numFeatures = inputDimensions[1];
        inputHeight = inputDimensions[2];
        inputWidth = inputDimensions[3];
        assert(batchSize > 0);
        assert(numFeatures > 0);
        assert(inputHeight > 0);
        assert(inputWidth > 0);
        assert(inputHeight >= windowHeight);
        assert(inputWidth >= windowWidth);
        assert(verticalPadding < windowHeight);
        assert(horizontalPadding < windowWidth);
        outputHeight = computeOutputDimensionSize(inputHeight, verticalPadding, windowHeight, verticalStride);
        outputWidth = computeOutputDimensionSize(inputWidth, horizontalPadding, windowWidth, horizontalStride);

        vector<unsigned long> outputDimensions = featureOutput.get().getDescriptor().getDimensions();
        assert(outputDimensions.size() == 4);
        assert(outputDimensions[0] == (uint32_t)batchSize);
        assert(outputDimensions[1] == (uint32_t)numFeatures);
        assert(outputDimensions[2] == (uint32_t)outputHeight);
        assert(outputDimensions[3] == (uint32_t)outputWidth);

        ScopedGpu scopedGpu(featureInput.get().getPlacement().getDeviceNum());

        poolingDescriptor = cudnnPoolingDescriptor_t();
        cudnnStatus = cudnnCreatePoolingDescriptor(&poolingDescriptor.get());
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
        cudnnStatus =
            cudnnSetPooling2dDescriptor(poolingDescriptor,
                                        poolingType == Type::MAX ? CUDNN_POOLING_MAX : CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
                                        CUDNN_NOT_PROPAGATE_NAN,
                                        windowHeight,
                                        windowWidth,
                                        verticalPadding,
                                        horizontalPadding,
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

    virtual Optional<Tensor> createFeatureOutputTensor() {
        assert(featureInput.isPresent());

        vector<unsigned long> inputDimensions = featureInput.get().getDescriptor().getDimensions();
        assert(inputDimensions.size() == 4);
        batchSize = inputDimensions[0];
        numFeatures = inputDimensions[1];
        inputHeight = inputDimensions[2];
        inputWidth = inputDimensions[3];
        assert(batchSize > 0);
        assert(numFeatures > 0);
        assert(inputHeight > 0);
        assert(inputWidth > 0);
        assert(inputHeight >= windowHeight);
        assert(inputWidth >= windowWidth);
        outputHeight = computeOutputDimensionSize(inputHeight, verticalPadding, windowHeight, verticalStride);
        outputWidth = computeOutputDimensionSize(inputWidth, horizontalPadding, windowWidth, horizontalStride);

        vector<unsigned long> featureOutputDimensions;
        featureOutputDimensions.push_back(batchSize);
        featureOutputDimensions.push_back(numFeatures);
        featureOutputDimensions.push_back(outputHeight);
        featureOutputDimensions.push_back(outputWidth);
        TensorDescriptor featureOutputDescriptor(featureInput.get().getDescriptor().getDataType(), featureOutputDimensions);
        return Tensor(featureInput.get().getPlacement(), featureOutputDescriptor);
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

    static uint32_t computeOutputDimensionSize(uint32_t inputDimensionSize,
                                               uint32_t perSidePadding,
                                               uint32_t windowSize,
                                               uint32_t windowStride) {
        uint32_t paddedSize = inputDimensionSize + 2 * perSidePadding;
        assert(windowSize <= paddedSize);
        uint32_t outputSize = 1 + ((paddedSize - windowSize) / windowStride);
        assert(outputSize > 0);
        return outputSize;
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
    int verticalPadding;
    int horizontalPadding;
    int outputHeight;
    int outputWidth;

    Optional<cudnnPoolingDescriptor_t> poolingDescriptor;
    Optional<cudnnTensorDescriptor_t> featureOutputDescriptor;
    Optional<cudnnTensorDescriptor_t> featureInputDescriptor;
};

}  // namespace ThorImplementation
