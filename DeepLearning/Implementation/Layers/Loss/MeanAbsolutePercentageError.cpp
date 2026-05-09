#include <optional>
#include "MeanAbsolutePercentageError.h"

#include "DeepLearning/Implementation/Layers/Loss.h"
#include "Utilities/TensorOperations/Loss/MeanAbsolutePercentageError.h"

#include <chrono>
#include <thread>

#include "DeepLearning/Implementation/ThorError.h"
using namespace ThorImplementation;
using namespace std;

MeanAbsolutePercentageError::~MeanAbsolutePercentageError() {}

MeanAbsolutePercentageError::MeanAbsolutePercentageError(TensorDescriptor::DataType lossDataType, float epsilon, float maxMagnitude)
    : Loss(lossDataType) {
    this->epsilon = epsilon;
    this->maxMagnitude = maxMagnitude;
}

void MeanAbsolutePercentageError::compileImpl() {
    Layer::compileImpl();
    THOR_THROW_IF_FALSE(featureInput.has_value());
    THOR_THROW_IF_FALSE(featureOutput.has_value());
    THOR_THROW_IF_FALSE(featureInput.value().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    THOR_THROW_IF_FALSE(featureInput.value().getDescriptor().getDimensions().size() == 2);

    if (!isInferenceOnly()) {
        THOR_THROW_IF_FALSE(errorOutput.has_value());
        THOR_THROW_IF_FALSE(errorOutput.value().isInitialized());
        THOR_THROW_IF_FALSE(errorOutput.value().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        THOR_THROW_IF_FALSE(errorOutput.value().getPlacement().getDeviceNum() == featureInput.value().getPlacement().getDeviceNum());
        THOR_THROW_IF_FALSE(errorOutput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16 ||
               errorOutput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32);

        THOR_THROW_IF_FALSE(labelsInput.has_value());
        THOR_THROW_IF_FALSE(labelsInput.value().isInitialized());
        THOR_THROW_IF_FALSE(labelsInput.value().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        THOR_THROW_IF_FALSE(labelsInput.value().getPlacement().getDeviceNum() == featureInput.value().getPlacement().getDeviceNum());

        vector<uint64_t> labelDimensions = labelsInput.value().getDescriptor().getDimensions();
        vector<uint64_t> featureInputDimensions = featureInput.value().getDescriptor().getDimensions();
        THOR_THROW_IF_FALSE(featureInputDimensions == labelDimensions);

        errorOutputCudnnTensorDescriptor =
            createCudnnTensorDescriptor(errorOutput.value().getDescriptor().getDimensions(), errorOutput.value().getDescriptor().getDataType());
    }

    batchSize = featureInput.value().getDescriptor().getDimensions()[0];
}

void MeanAbsolutePercentageError::infer(std::optional<Tensor> predictions, std::optional<Tensor> elementLoss, Stream stream) {
    THOR_THROW_IF_FALSE(predictions.has_value());
    THOR_THROW_IF_FALSE(elementLoss.has_value());
    THOR_THROW_IF_FALSE(labelsInput.has_value());
    THOR_THROW_IF_FALSE(elementLoss.value().getDescriptor().getDimensions() == predictions.value().getDescriptor().getDimensions());
    if (!isInferenceOnly())
        THOR_THROW_IF_FALSE(errorOutput.has_value());

    ScopedGpu scopedGpu(predictions.value().getPlacement().getDeviceNum());

    THOR_THROW_IF_FALSE(compiled);
    THOR_THROW_IF_FALSE(predictions.value().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16 ||
           predictions.value().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32);

    stream.waitEvent(labelsStream.putEvent());

    if (predictions.value().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
        launchMeanAbsolutePercentageErrorWithFP16Predictions();
    } else if (predictions.value().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32) {
        launchMeanAbsolutePercentageErrorWithFP32Predictions();
    } else {
        THOR_UNREACHABLE();
    }
}

void MeanAbsolutePercentageError::backProp(std::optional<Tensor> labels,
                                           std::optional<Tensor> normalizedPredictions,
                                           std::optional<Tensor> lossGradient,
                                           Stream stream) {
    // Mean absolute loss gradient is pre-computed during infer() for efficiency
    THOR_THROW_IF_FALSE(lossGradient.has_value());
    THOR_THROW_IF_FALSE(lossGradient.value().getDataType() == TensorDescriptor::DataType::FP32 ||
           lossGradient.value().getDataType() == TensorDescriptor::DataType::FP16);

    if (lossScalingFactor != 1) {
        cudnnStatus_t cudnnStatus;
        float lsffloat = (float)lossScalingFactor;

        cudnnStatus = cudnnScaleTensor(stream.getCudnnHandle(), errorOutputCudnnTensorDescriptor, errorOutput.value().getMemPtr(), &lsffloat);
        THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);
    }
}

void MeanAbsolutePercentageError::launchMeanAbsolutePercentageErrorWithFP16Predictions() {
    THOR_THROW_IF_FALSE(featureInput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16);

    if (featureOutput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16)
        launchMeanAbsolutePercentageErrorWithFP16PredictionsAndFP16Loss();
    else if (featureOutput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32)
        launchMeanAbsolutePercentageErrorWithFP16PredictionsAndFP32Loss();
    else
        THOR_UNREACHABLE();
}

void MeanAbsolutePercentageError::launchMeanAbsolutePercentageErrorWithFP32Predictions() {
    THOR_THROW_IF_FALSE(featureInput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32);

    if (featureOutput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16)
        launchMeanAbsolutePercentageErrorWithFP32PredictionsAndFP16Loss();
    else if (featureOutput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32)
        launchMeanAbsolutePercentageErrorWithFP32PredictionsAndFP32Loss();
    else
        THOR_UNREACHABLE();
}

void MeanAbsolutePercentageError::launchMeanAbsolutePercentageErrorWithFP16PredictionsAndFP16Loss() {
    THOR_THROW_IF_FALSE(featureInput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16);
    THOR_THROW_IF_FALSE(featureOutput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16);

    if (labelsInput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
        launchMeanAbsolutePercentageError<half, half, half>(labelsInput.value().getMemPtr(),
                                                            featureInput.value().getMemPtr(),
                                                            featureOutput.value().getMemPtr(),
                                                            isInferenceOnly() ? nullptr : (half *)errorOutput.value().getMemPtr(),
                                                            featureOutput.value().getDescriptor().getDimensions()[1],
                                                            batchSize,
                                                            !isInferenceOnly(),
                                                            stream,
                                                            epsilon,
                                                            maxMagnitude);
    } else if (labelsInput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32) {
        launchMeanAbsolutePercentageError<float, half, half>(labelsInput.value().getMemPtr(),
                                                             featureInput.value().getMemPtr(),
                                                             featureOutput.value().getMemPtr(),
                                                             isInferenceOnly() ? nullptr : (half *)errorOutput.value().getMemPtr(),
                                                             featureOutput.value().getDescriptor().getDimensions()[1],
                                                             batchSize,
                                                             !isInferenceOnly(),
                                                             stream,
                                                             epsilon,
                                                             maxMagnitude);
    } else if (labelsInput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT8) {
        launchMeanAbsolutePercentageError<uint8_t, half, half>(labelsInput.value().getMemPtr(),
                                                               featureInput.value().getMemPtr(),
                                                               featureOutput.value().getMemPtr(),
                                                               isInferenceOnly() ? nullptr : (half *)errorOutput.value().getMemPtr(),
                                                               featureOutput.value().getDescriptor().getDimensions()[1],
                                                               batchSize,
                                                               !isInferenceOnly(),
                                                               stream,
                                                               epsilon,
                                                               maxMagnitude);
    } else if (labelsInput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT16) {
        launchMeanAbsolutePercentageError<uint16_t, half, half>(labelsInput.value().getMemPtr(),
                                                                featureInput.value().getMemPtr(),
                                                                featureOutput.value().getMemPtr(),
                                                                isInferenceOnly() ? nullptr : (half *)errorOutput.value().getMemPtr(),
                                                                featureOutput.value().getDescriptor().getDimensions()[1],
                                                                batchSize,
                                                                !isInferenceOnly(),
                                                                stream,
                                                                epsilon,
                                                                maxMagnitude);
    } else if (labelsInput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT32) {
        launchMeanAbsolutePercentageError<uint32_t, half, half>(labelsInput.value().getMemPtr(),
                                                                featureInput.value().getMemPtr(),
                                                                featureOutput.value().getMemPtr(),
                                                                isInferenceOnly() ? nullptr : (half *)errorOutput.value().getMemPtr(),
                                                                featureOutput.value().getDescriptor().getDimensions()[1],
                                                                batchSize,
                                                                !isInferenceOnly(),
                                                                stream,
                                                                epsilon,
                                                                maxMagnitude);
    } else if (labelsInput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::BOOLEAN) {
        launchMeanAbsolutePercentageError<bool, half, half>(labelsInput.value().getMemPtr(),
                                                            featureInput.value().getMemPtr(),
                                                            featureOutput.value().getMemPtr(),
                                                            isInferenceOnly() ? nullptr : (half *)errorOutput.value().getMemPtr(),
                                                            featureOutput.value().getDescriptor().getDimensions()[1],
                                                            batchSize,
                                                            !isInferenceOnly(),
                                                            stream,
                                                            epsilon,
                                                            maxMagnitude);
    } else {
        THOR_UNREACHABLE();
    }
}

void MeanAbsolutePercentageError::launchMeanAbsolutePercentageErrorWithFP16PredictionsAndFP32Loss() {
    THOR_THROW_IF_FALSE(featureInput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16);
    THOR_THROW_IF_FALSE(featureOutput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32);

    if (labelsInput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
        launchMeanAbsolutePercentageError<half, half, float>(labelsInput.value().getMemPtr(),
                                                             featureInput.value().getMemPtr(),
                                                             featureOutput.value().getMemPtr(),
                                                             isInferenceOnly() ? nullptr : (half *)errorOutput.value().getMemPtr(),
                                                             featureOutput.value().getDescriptor().getDimensions()[1],
                                                             batchSize,
                                                             !isInferenceOnly(),
                                                             stream,
                                                             epsilon,
                                                             maxMagnitude);
    } else if (labelsInput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32) {
        launchMeanAbsolutePercentageError<float, half, float>(labelsInput.value().getMemPtr(),
                                                              featureInput.value().getMemPtr(),
                                                              featureOutput.value().getMemPtr(),
                                                              isInferenceOnly() ? nullptr : (half *)errorOutput.value().getMemPtr(),
                                                              featureOutput.value().getDescriptor().getDimensions()[1],
                                                              batchSize,
                                                              !isInferenceOnly(),
                                                              stream,
                                                              epsilon,
                                                              maxMagnitude);
    } else if (labelsInput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT8) {
        launchMeanAbsolutePercentageError<uint8_t, half, float>(labelsInput.value().getMemPtr(),
                                                                featureInput.value().getMemPtr(),
                                                                featureOutput.value().getMemPtr(),
                                                                isInferenceOnly() ? nullptr : (half *)errorOutput.value().getMemPtr(),
                                                                featureOutput.value().getDescriptor().getDimensions()[1],
                                                                batchSize,
                                                                !isInferenceOnly(),
                                                                stream,
                                                                epsilon,
                                                                maxMagnitude);
    } else if (labelsInput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT16) {
        launchMeanAbsolutePercentageError<uint16_t, half, float>(labelsInput.value().getMemPtr(),
                                                                 featureInput.value().getMemPtr(),
                                                                 featureOutput.value().getMemPtr(),
                                                                 isInferenceOnly() ? nullptr : (half *)errorOutput.value().getMemPtr(),
                                                                 featureOutput.value().getDescriptor().getDimensions()[1],
                                                                 batchSize,
                                                                 !isInferenceOnly(),
                                                                 stream,
                                                                 epsilon,
                                                                 maxMagnitude);
    } else if (labelsInput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT32) {
        launchMeanAbsolutePercentageError<uint32_t, half, float>(labelsInput.value().getMemPtr(),
                                                                 featureInput.value().getMemPtr(),
                                                                 featureOutput.value().getMemPtr(),
                                                                 isInferenceOnly() ? nullptr : (half *)errorOutput.value().getMemPtr(),
                                                                 featureOutput.value().getDescriptor().getDimensions()[1],
                                                                 batchSize,
                                                                 !isInferenceOnly(),
                                                                 stream,
                                                                 epsilon,
                                                                 maxMagnitude);
    } else if (labelsInput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::BOOLEAN) {
        launchMeanAbsolutePercentageError<bool, half, float>(labelsInput.value().getMemPtr(),
                                                             featureInput.value().getMemPtr(),
                                                             featureOutput.value().getMemPtr(),
                                                             isInferenceOnly() ? nullptr : (half *)errorOutput.value().getMemPtr(),
                                                             featureOutput.value().getDescriptor().getDimensions()[1],
                                                             batchSize,
                                                             !isInferenceOnly(),
                                                             stream,
                                                             epsilon,
                                                             maxMagnitude);
    } else {
        THOR_UNREACHABLE();
    }
}

void MeanAbsolutePercentageError::launchMeanAbsolutePercentageErrorWithFP32PredictionsAndFP16Loss() {
    THOR_THROW_IF_FALSE(featureInput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32);
    THOR_THROW_IF_FALSE(featureOutput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16);

    if (labelsInput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
        launchMeanAbsolutePercentageError<half, float, half>(labelsInput.value().getMemPtr(),
                                                             featureInput.value().getMemPtr(),
                                                             featureOutput.value().getMemPtr(),
                                                             isInferenceOnly() ? nullptr : (half *)errorOutput.value().getMemPtr(),
                                                             featureOutput.value().getDescriptor().getDimensions()[1],
                                                             batchSize,
                                                             !isInferenceOnly(),
                                                             stream,
                                                             epsilon,
                                                             maxMagnitude);
    } else if (labelsInput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32) {
        launchMeanAbsolutePercentageError<float, float, half>(labelsInput.value().getMemPtr(),
                                                              featureInput.value().getMemPtr(),
                                                              featureOutput.value().getMemPtr(),
                                                              isInferenceOnly() ? nullptr : (half *)errorOutput.value().getMemPtr(),
                                                              featureOutput.value().getDescriptor().getDimensions()[1],
                                                              batchSize,
                                                              !isInferenceOnly(),
                                                              stream,
                                                              epsilon,
                                                              maxMagnitude);
    } else if (labelsInput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT8) {
        launchMeanAbsolutePercentageError<uint8_t, float, half>(labelsInput.value().getMemPtr(),
                                                                featureInput.value().getMemPtr(),
                                                                featureOutput.value().getMemPtr(),
                                                                isInferenceOnly() ? nullptr : (half *)errorOutput.value().getMemPtr(),
                                                                featureOutput.value().getDescriptor().getDimensions()[1],
                                                                batchSize,
                                                                !isInferenceOnly(),
                                                                stream,
                                                                epsilon,
                                                                maxMagnitude);
    } else if (labelsInput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT16) {
        launchMeanAbsolutePercentageError<uint16_t, float, half>(labelsInput.value().getMemPtr(),
                                                                 featureInput.value().getMemPtr(),
                                                                 featureOutput.value().getMemPtr(),
                                                                 isInferenceOnly() ? nullptr : (half *)errorOutput.value().getMemPtr(),
                                                                 featureOutput.value().getDescriptor().getDimensions()[1],
                                                                 batchSize,
                                                                 !isInferenceOnly(),
                                                                 stream,
                                                                 epsilon,
                                                                 maxMagnitude);
    } else if (labelsInput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT32) {
        launchMeanAbsolutePercentageError<uint32_t, float, half>(labelsInput.value().getMemPtr(),
                                                                 featureInput.value().getMemPtr(),
                                                                 featureOutput.value().getMemPtr(),
                                                                 isInferenceOnly() ? nullptr : (half *)errorOutput.value().getMemPtr(),
                                                                 featureOutput.value().getDescriptor().getDimensions()[1],
                                                                 batchSize,
                                                                 !isInferenceOnly(),
                                                                 stream,
                                                                 epsilon,
                                                                 maxMagnitude);
    } else if (labelsInput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::BOOLEAN) {
        launchMeanAbsolutePercentageError<bool, float, half>(labelsInput.value().getMemPtr(),
                                                             featureInput.value().getMemPtr(),
                                                             featureOutput.value().getMemPtr(),
                                                             isInferenceOnly() ? nullptr : (half *)errorOutput.value().getMemPtr(),
                                                             featureOutput.value().getDescriptor().getDimensions()[1],
                                                             batchSize,
                                                             !isInferenceOnly(),
                                                             stream,
                                                             epsilon,
                                                             maxMagnitude);
    } else {
        THOR_UNREACHABLE();
    }
}

void MeanAbsolutePercentageError::launchMeanAbsolutePercentageErrorWithFP32PredictionsAndFP32Loss() {
    THOR_THROW_IF_FALSE(featureInput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32);
    THOR_THROW_IF_FALSE(featureOutput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32);

    if (labelsInput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
        launchMeanAbsolutePercentageError<half, float, float>(labelsInput.value().getMemPtr(),
                                                              featureInput.value().getMemPtr(),
                                                              featureOutput.value().getMemPtr(),
                                                              isInferenceOnly() ? nullptr : (half *)errorOutput.value().getMemPtr(),
                                                              featureOutput.value().getDescriptor().getDimensions()[1],
                                                              batchSize,
                                                              !isInferenceOnly(),
                                                              stream,
                                                              epsilon,
                                                              maxMagnitude);
    } else if (labelsInput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32) {
        launchMeanAbsolutePercentageError<float, float, float>(labelsInput.value().getMemPtr(),
                                                               featureInput.value().getMemPtr(),
                                                               featureOutput.value().getMemPtr(),
                                                               isInferenceOnly() ? nullptr : (half *)errorOutput.value().getMemPtr(),
                                                               featureOutput.value().getDescriptor().getDimensions()[1],
                                                               batchSize,
                                                               !isInferenceOnly(),
                                                               stream,
                                                               epsilon,
                                                               maxMagnitude);
    } else if (labelsInput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT8) {
        launchMeanAbsolutePercentageError<uint8_t, float, float>(labelsInput.value().getMemPtr(),
                                                                 featureInput.value().getMemPtr(),
                                                                 featureOutput.value().getMemPtr(),
                                                                 isInferenceOnly() ? nullptr : (half *)errorOutput.value().getMemPtr(),
                                                                 featureOutput.value().getDescriptor().getDimensions()[1],
                                                                 batchSize,
                                                                 !isInferenceOnly(),
                                                                 stream,
                                                                 epsilon,
                                                                 maxMagnitude);
    } else if (labelsInput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT16) {
        launchMeanAbsolutePercentageError<uint16_t, float, float>(labelsInput.value().getMemPtr(),
                                                                  featureInput.value().getMemPtr(),
                                                                  featureOutput.value().getMemPtr(),
                                                                  isInferenceOnly() ? nullptr : (half *)errorOutput.value().getMemPtr(),
                                                                  featureOutput.value().getDescriptor().getDimensions()[1],
                                                                  batchSize,
                                                                  !isInferenceOnly(),
                                                                  stream,
                                                                  epsilon,
                                                                  maxMagnitude);
    } else if (labelsInput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT32) {
        launchMeanAbsolutePercentageError<uint32_t, float, float>(labelsInput.value().getMemPtr(),
                                                                  featureInput.value().getMemPtr(),
                                                                  featureOutput.value().getMemPtr(),
                                                                  isInferenceOnly() ? nullptr : (half *)errorOutput.value().getMemPtr(),
                                                                  featureOutput.value().getDescriptor().getDimensions()[1],
                                                                  batchSize,
                                                                  !isInferenceOnly(),
                                                                  stream,
                                                                  epsilon,
                                                                  maxMagnitude);
    } else if (labelsInput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::BOOLEAN) {
        launchMeanAbsolutePercentageError<bool, float, float>(labelsInput.value().getMemPtr(),
                                                              featureInput.value().getMemPtr(),
                                                              featureOutput.value().getMemPtr(),
                                                              isInferenceOnly() ? nullptr : (half *)errorOutput.value().getMemPtr(),
                                                              featureOutput.value().getDescriptor().getDimensions()[1],
                                                              batchSize,
                                                              !isInferenceOnly(),
                                                              stream,
                                                              epsilon,
                                                              maxMagnitude);
    } else {
        THOR_UNREACHABLE();
    }
}
