#pragma once

#include <optional>
#include <stdexcept>
#include <string>
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

        THOR_THROW_IF_FALSE(featureInput.has_value());
        THOR_THROW_IF_FALSE(featureOutput.has_value());

        // Dimensions are NCHW
        std::vector<unsigned long> inputDimensions = featureInput.value().getDescriptor().getDimensions();
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

        std::vector<unsigned long> outputDimensions = featureOutput.value().getDescriptor().getDimensions();
        DataType inputDataType = featureInput.value().getDescriptor().getDataType();
        THOR_THROW_IF_FALSE(featureOutput.value().getDescriptor().getDataType() == inputDataType);
        cudnnDataType_t cudnnDataType = toCudnnPoolingDataType(inputDataType);
        THOR_THROW_IF_FALSE(outputDimensions.size() == 4);
        THOR_THROW_IF_FALSE(outputDimensions[0] == (uint32_t)batchSize);
        THOR_THROW_IF_FALSE(outputDimensions[1] == (uint32_t)numFeatures);
        THOR_THROW_IF_FALSE(outputDimensions[2] == (uint32_t)outputHeight);
        THOR_THROW_IF_FALSE(outputDimensions[3] == (uint32_t)outputWidth);

        ScopedGpu scopedGpu(featureInput.value().getPlacement().getDeviceNum());

        poolingDescriptor = cudnnPoolingDescriptor_t();
        cudnnStatus = cudnnCreatePoolingDescriptor(&poolingDescriptor.value());
        THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);
        cudnnStatus =
            cudnnSetPooling2dDescriptor(poolingDescriptor.value(),
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
        cudnnStatus = cudnnCreateTensorDescriptor(&featureInputDescriptor.value());
        THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);
        cudnnStatus = cudnnSetTensor4dDescriptor(
            featureInputDescriptor.value(), CUDNN_TENSOR_NCHW, cudnnDataType, batchSize, numFeatures, inputHeight, inputWidth);
        THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);

        featureOutputDescriptor = cudnnTensorDescriptor_t();
        cudnnStatus = cudnnCreateTensorDescriptor(&featureOutputDescriptor.value());
        THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);
        cudnnStatus = cudnnSetTensor4dDescriptor(
            featureOutputDescriptor.value(), CUDNN_TENSOR_NCHW, cudnnDataType, batchSize, numFeatures, outputHeight, outputWidth);
        THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);
    }

    std::optional<Tensor> createFeatureOutputTensor() override {
        THOR_THROW_IF_FALSE(featureInput.has_value());

        std::vector<unsigned long> inputDimensions = featureInput.value().getDescriptor().getDimensions();
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
        TensorDescriptor featureOutputDescriptor(featureInput.value().getDescriptor().getDataType(), featureOutputDimensions);
        return Tensor(featureInput.value().getPlacement(), featureOutputDescriptor);
    }

    void cleanup() {
        cudnnStatus_t cudnnStatus;

        if (poolingDescriptor.has_value()) {
            cudnnStatus = cudnnDestroyPoolingDescriptor(poolingDescriptor.value());
            THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);
            poolingDescriptor.reset();
        }

        if (featureInputDescriptor.has_value()) {
            cudnnStatus = cudnnDestroyTensorDescriptor(featureInputDescriptor.value());
            THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);
            featureInputDescriptor.reset();
        }

        if (featureOutputDescriptor.has_value()) {
            cudnnStatus = cudnnDestroyTensorDescriptor(featureOutputDescriptor.value());
            THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);
            featureOutputDescriptor.reset();
        }
    }

    void infer(std::optional<Tensor> inputTensor, std::optional<Tensor> outputTensor, Stream stream) override {
        THOR_THROW_IF_FALSE(inputTensor.has_value());
        THOR_THROW_IF_FALSE(outputTensor.has_value());

        cudnnStatus_t cudnnStatus;
        cudnnStatus = cudnnPoolingForward(stream.getCudnnHandle(),
                                          poolingDescriptor.value(),
                                          &ALPHA_NO_SCALE,
                                          featureInputDescriptor.value(),
                                          inputTensor.value().getMemPtr(),
                                          &BETA_CLEAR,
                                          featureOutputDescriptor.value(),
                                          outputTensor.value().getMemPtr());
        THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);
    }

    void backProp(std::optional<Tensor> dataIn, std::optional<Tensor> errorIn, std::optional<Tensor> errorOut, Stream stream) override {
        if (!errorOut.has_value())
            return;
        THOR_THROW_IF_FALSE(errorIn.has_value());

        cudnnStatus_t cudnnStatus;
        cudnnStatus = cudnnPoolingBackward(stream.getCudnnHandle(),
                                           poolingDescriptor.value(),
                                           &ALPHA_NO_SCALE,
                                           featureOutputDescriptor.value(),
                                           featureOutput.value().getMemPtr(),
                                           featureOutputDescriptor.value(),
                                           errorIn.value().getMemPtr(),
                                           featureInputDescriptor.value(),
                                           dataIn.value().getMemPtr(),
                                           &BETA_CLEAR,
                                           featureInputDescriptor.value(),
                                           errorOut.value().getMemPtr());
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

    std::optional<cudnnPoolingDescriptor_t> poolingDescriptor;
    std::optional<cudnnTensorDescriptor_t> featureOutputDescriptor;
    std::optional<cudnnTensorDescriptor_t> featureInputDescriptor;

    static cudnnDataType_t toCudnnPoolingDataType(DataType dtype) {
        switch (dtype) {
            case DataType::FP16:
                return CUDNN_DATA_HALF;
            case DataType::FP32:
                return CUDNN_DATA_FLOAT;
            case DataType::BF16:
                return CUDNN_DATA_BFLOAT16;
            default:
                throw std::runtime_error("Pooling supports FP16, FP32, and BF16 tensors with the cuDNN pooling API; got " +
                                         TensorDescriptor::getElementTypeName(dtype));
        }
    }
};

}  // namespace ThorImplementation
