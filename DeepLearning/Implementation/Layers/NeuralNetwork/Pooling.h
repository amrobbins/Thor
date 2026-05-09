#pragma once

#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Implementation/Layers/Layer.h"

namespace ThorImplementation {

class Pooling : public Layer {
   public:
    enum class Type { MAX = 11, AVERAGE = 12 };

    ~Pooling() override {}

    Pooling(Type poolingType,
            uint32_t windowHeight,
            uint32_t windowWidth,
            uint32_t verticalStride,
            uint32_t horizontalStride,
            uint32_t verticalPadding,
            uint32_t horizontalPadding)
        : poolingType(poolingType),
          windowHeight(windowHeight),
          windowWidth(windowWidth),
          verticalStride(verticalStride),
          horizontalStride(horizontalStride),
          verticalPadding(verticalPadding),
          horizontalPadding(horizontalPadding) {
        THOR_THROW_IF_FALSE(poolingType == Type::MAX || poolingType == Type::AVERAGE);
        THOR_THROW_IF_FALSE(windowHeight > 0);
        THOR_THROW_IF_FALSE(windowWidth > 0);
        THOR_THROW_IF_FALSE(verticalStride > 0);
        THOR_THROW_IF_FALSE(horizontalStride > 0);
    }

    void compileImpl() override {
        Layer::compileImpl();
        cudnnStatus_t cudnnStatus;

        THOR_THROW_IF_FALSE(featureInput.isPresent());
        THOR_THROW_IF_FALSE(featureOutput.isPresent());

        // Dimensions are NCHW
        std::vector<unsigned long> inputDimensions = featureInput.get().getDescriptor().getDimensions();
        THOR_THROW_IF_FALSE(inputDimensions.size() == 4);
        batchSize = inputDimensions[0];
        numFeatures = inputDimensions[1];
        inputHeight = inputDimensions[2];
        inputWidth = inputDimensions[3];
        THOR_THROW_IF_FALSE(batchSize > 0);
        THOR_THROW_IF_FALSE(numFeatures > 0);
        THOR_THROW_IF_FALSE(inputHeight > 0);
        THOR_THROW_IF_FALSE(inputWidth > 0);
        THOR_THROW_IF_FALSE(inputHeight >= windowHeight);
        THOR_THROW_IF_FALSE(inputWidth >= windowWidth);
        THOR_THROW_IF_FALSE(verticalPadding < windowHeight);
        THOR_THROW_IF_FALSE(horizontalPadding < windowWidth);
        outputHeight = computeOutputDimensionSize(inputHeight, verticalPadding, windowHeight, verticalStride);
        outputWidth = computeOutputDimensionSize(inputWidth, horizontalPadding, windowWidth, horizontalStride);

        // Ensure that cudnnHandle is preallocated
        stream.getCudnnHandle();

        std::vector<unsigned long> outputDimensions = featureOutput.get().getDescriptor().getDimensions();
        THOR_THROW_IF_FALSE(outputDimensions.size() == 4);
        THOR_THROW_IF_FALSE(outputDimensions[0] == (uint32_t)batchSize);
        THOR_THROW_IF_FALSE(outputDimensions[1] == (uint32_t)numFeatures);
        THOR_THROW_IF_FALSE(outputDimensions[2] == (uint32_t)outputHeight);
        THOR_THROW_IF_FALSE(outputDimensions[3] == (uint32_t)outputWidth);

        ScopedGpu scopedGpu(featureInput.get().getPlacement().getDeviceNum());

        poolingDescriptor = cudnnPoolingDescriptor_t();
        cudnnStatus = cudnnCreatePoolingDescriptor(&poolingDescriptor.get());
        THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);
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
        THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);

        featureInputDescriptor = cudnnTensorDescriptor_t();
        cudnnStatus = cudnnCreateTensorDescriptor(&featureInputDescriptor.get());
        THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);
        cudnnStatus = cudnnSetTensor4dDescriptor(
            featureInputDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize, numFeatures, inputHeight, inputWidth);
        THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);

        featureOutputDescriptor = cudnnTensorDescriptor_t();
        cudnnStatus = cudnnCreateTensorDescriptor(&featureOutputDescriptor.get());
        THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);
        cudnnStatus = cudnnSetTensor4dDescriptor(
            featureOutputDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize, numFeatures, outputHeight, outputWidth);
        THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);
    }

    Optional<Tensor> createFeatureOutputTensor() override {
        THOR_THROW_IF_FALSE(featureInput.isPresent());

        std::vector<unsigned long> inputDimensions = featureInput.get().getDescriptor().getDimensions();
        THOR_THROW_IF_FALSE(inputDimensions.size() == 4);
        batchSize = inputDimensions[0];
        numFeatures = inputDimensions[1];
        inputHeight = inputDimensions[2];
        inputWidth = inputDimensions[3];
        THOR_THROW_IF_FALSE(batchSize > 0);
        THOR_THROW_IF_FALSE(numFeatures > 0);
        THOR_THROW_IF_FALSE(inputHeight > 0);
        THOR_THROW_IF_FALSE(inputWidth > 0);
        THOR_THROW_IF_FALSE(inputHeight >= windowHeight);
        THOR_THROW_IF_FALSE(inputWidth >= windowWidth);
        outputHeight = computeOutputDimensionSize(inputHeight, verticalPadding, windowHeight, verticalStride);
        outputWidth = computeOutputDimensionSize(inputWidth, horizontalPadding, windowWidth, horizontalStride);

        std::vector<unsigned long> featureOutputDimensions;
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
            THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);
            poolingDescriptor.clear();
        }

        if (featureInputDescriptor.isPresent()) {
            cudnnStatus = cudnnDestroyTensorDescriptor(featureInputDescriptor.get());
            THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);
            featureInputDescriptor.clear();
        }

        if (featureOutputDescriptor.isPresent()) {
            cudnnStatus = cudnnDestroyTensorDescriptor(featureOutputDescriptor.get());
            THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);
            featureOutputDescriptor.clear();
        }
    }

    void infer(Optional<Tensor> inputTensor, Optional<Tensor> outputTensor, Stream stream) override {
        THOR_THROW_IF_FALSE(inputTensor.isPresent());
        THOR_THROW_IF_FALSE(outputTensor.isPresent());

        cudnnStatus_t cudnnStatus;
        cudnnStatus = cudnnPoolingForward(stream.getCudnnHandle(),
                                          poolingDescriptor,
                                          &ALPHA_NO_SCALE,
                                          featureInputDescriptor,
                                          inputTensor.get().getMemPtr(),
                                          &BETA_CLEAR,
                                          featureOutputDescriptor,
                                          outputTensor.get().getMemPtr());
        THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);
    }

    void backProp(Optional<Tensor> dataIn, Optional<Tensor> errorIn, Optional<Tensor> errorOut, Stream stream) override {
        if (errorOut.isEmpty())
            return;
        THOR_THROW_IF_FALSE(errorIn.isPresent());

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
        THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);
    }

    static uint32_t computeOutputDimensionSize(uint32_t inputDimensionSize,
                                               uint32_t perSidePadding,
                                               uint32_t windowSize,
                                               uint32_t windowStride) {
        uint32_t paddedSize = inputDimensionSize + 2 * perSidePadding;
        THOR_THROW_IF_FALSE(windowSize <= paddedSize);
        uint32_t outputSize = 1 + ((paddedSize - windowSize) / windowStride);
        THOR_THROW_IF_FALSE(outputSize > 0);
        return outputSize;
    }

    Type getPoolingType() { return poolingType; }

    uint32_t getWindowHeight() { return windowHeight; }
    uint32_t getWindowWidth() { return windowWidth; }
    uint32_t getVerticalStride() { return verticalStride; }
    uint32_t getHorizontalStride() { return horizontalStride; }
    uint32_t getVerticalPadding() { return verticalPadding; }
    uint32_t getHorizontalPadding() { return horizontalPadding; }
    uint32_t getOutputHeight() { return outputHeight; }
    uint32_t getOutputWidth() { return outputWidth; }

   private:
    static const float ALPHA_NO_SCALE;
    static const float BETA_CLEAR;

    Type poolingType;
    uint32_t windowHeight;
    uint32_t windowWidth;
    uint32_t verticalStride;
    uint32_t horizontalStride;
    uint32_t batchSize;
    uint32_t numFeatures;
    uint32_t inputHeight;
    uint32_t inputWidth;
    uint32_t verticalPadding;
    uint32_t horizontalPadding;
    uint32_t outputHeight;
    uint32_t outputWidth;

    Optional<cudnnPoolingDescriptor_t> poolingDescriptor;
    Optional<cudnnTensorDescriptor_t> featureOutputDescriptor;
    Optional<cudnnTensorDescriptor_t> featureInputDescriptor;
};

}  // namespace ThorImplementation
