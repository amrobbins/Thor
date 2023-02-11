#include "MeanAbsolutePercentageError.h"

#include "DeepLearning/Implementation/Layers/Loss.h"
#include "Utilities/TensorOperations/Loss/MeanAbsolutePercentageError.h"

#include <chrono>
#include <thread>

using namespace ThorImplementation;
using namespace std;

MeanAbsolutePercentageError::~MeanAbsolutePercentageError() {}

MeanAbsolutePercentageError::MeanAbsolutePercentageError(float epsilon, float maxMagnitude) : Loss() {
    this->epsilon = epsilon;
    this->maxMagnitude = maxMagnitude;
}

void MeanAbsolutePercentageError::compile() {
    assert(featureInput.isPresent());
    assert(featureOutput.isPresent());
    assert(featureInput.get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert(featureInput.get().getDescriptor().getDimensions().size() == 2);

    if (!isInferenceOnly()) {
        assert(errorOutput.isPresent());
        assert(errorOutput.get().isInitialized());
        assert(errorOutput.get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        assert(errorOutput.get().getPlacement().getDeviceNum() == featureInput.get().getPlacement().getDeviceNum());
        assert(errorOutput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16 ||
               errorOutput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32);

        assert(labelsInput.isPresent());
        assert(labelsInput.get().isInitialized());
        assert(labelsInput.get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        assert(labelsInput.get().getPlacement().getDeviceNum() == featureInput.get().getPlacement().getDeviceNum());

        vector<uint64_t> labelDimensions = labelsInput.get().getDescriptor().getDimensions();
        vector<uint64_t> featureInputDimensions = featureInput.get().getDescriptor().getDimensions();
        assert(featureInputDimensions == labelDimensions);

        errorOutputCudnnTensorDescriptor =
            createCudnnTensorDescriptor(errorOutput.get().getDescriptor().getDimensions(), errorOutput.get().getDescriptor().getDataType());
    }

    batchSize = featureInput.get().getDescriptor().getDimensions()[0];
}

void MeanAbsolutePercentageError::infer(Optional<Tensor> predictions, Optional<Tensor> elementLoss, Stream stream) {
    assert(predictions.isPresent());
    assert(elementLoss.isPresent());
    assert(labelsInput.isPresent());
    assert(elementLoss.get().getDescriptor().getDimensions() == predictions.get().getDescriptor().getDimensions());
    if (!isInferenceOnly())
        assert(errorOutput.isPresent());

    ScopedGpu scopedGpu(predictions.get().getPlacement().getDeviceNum());

    assert(compiled);
    assert(predictions.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16 ||
           predictions.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32);

    stream.waitEvent(labelsStream.putEvent());

    if (predictions.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
        launchMeanAbsolutePercentageErrorWithFP16Predictions();
    } else if (predictions.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32) {
        launchMeanAbsolutePercentageErrorWithFP32Predictions();
    } else {
        assert(false);
    }
}

void MeanAbsolutePercentageError::backProp(Optional<Tensor> labels,
                                           Optional<Tensor> normalizedPredictions,
                                           Optional<Tensor> lossGradient,
                                           Stream stream) {
    // Mean absolute loss gradient is pre-computed during infer() for efficiency
    assert(lossGradient.isPresent());
    assert(lossGradient.get().getDataType() == TensorDescriptor::DataType::FP32 ||
           lossGradient.get().getDataType() == TensorDescriptor::DataType::FP16);

    if (lossScalingFactor != 1) {
        cudnnStatus_t cudnnStatus;
        float lsffloat = (float)lossScalingFactor;

        cudnnStatus = cudnnScaleTensor(stream.getCudnnHandle(), errorOutputCudnnTensorDescriptor, errorOutput.get().getMemPtr(), &lsffloat);
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
    }
}

void MeanAbsolutePercentageError::launchMeanAbsolutePercentageErrorWithFP16Predictions() {
    assert(featureInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16);

    if (featureOutput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16)
        launchMeanAbsolutePercentageErrorWithFP16PredictionsAndFP16Loss();
    else if (featureOutput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32)
        launchMeanAbsolutePercentageErrorWithFP16PredictionsAndFP32Loss();
    else
        assert(false);
}

void MeanAbsolutePercentageError::launchMeanAbsolutePercentageErrorWithFP32Predictions() {
    assert(featureInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32);

    if (featureOutput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16)
        launchMeanAbsolutePercentageErrorWithFP32PredictionsAndFP16Loss();
    else if (featureOutput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32)
        launchMeanAbsolutePercentageErrorWithFP32PredictionsAndFP32Loss();
    else
        assert(false);
}

void MeanAbsolutePercentageError::launchMeanAbsolutePercentageErrorWithFP16PredictionsAndFP16Loss() {
    assert(featureInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16);
    assert(featureOutput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16);

    if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
        launchMeanAbsolutePercentageError<half, half, half>(labelsInput.get().getMemPtr(),
                                                            featureInput.get().getMemPtr(),
                                                            featureOutput.get().getMemPtr(),
                                                            isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                            featureOutput.get().getDescriptor().getDimensions()[1],
                                                            batchSize,
                                                            !isInferenceOnly(),
                                                            stream,
                                                            epsilon,
                                                            maxMagnitude);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32) {
        launchMeanAbsolutePercentageError<float, half, half>(labelsInput.get().getMemPtr(),
                                                             featureInput.get().getMemPtr(),
                                                             featureOutput.get().getMemPtr(),
                                                             isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                             featureOutput.get().getDescriptor().getDimensions()[1],
                                                             batchSize,
                                                             !isInferenceOnly(),
                                                             stream,
                                                             epsilon,
                                                             maxMagnitude);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT8) {
        launchMeanAbsolutePercentageError<uint8_t, half, half>(labelsInput.get().getMemPtr(),
                                                               featureInput.get().getMemPtr(),
                                                               featureOutput.get().getMemPtr(),
                                                               isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                               featureOutput.get().getDescriptor().getDimensions()[1],
                                                               batchSize,
                                                               !isInferenceOnly(),
                                                               stream,
                                                               epsilon,
                                                               maxMagnitude);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT16) {
        launchMeanAbsolutePercentageError<uint16_t, half, half>(labelsInput.get().getMemPtr(),
                                                                featureInput.get().getMemPtr(),
                                                                featureOutput.get().getMemPtr(),
                                                                isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                                featureOutput.get().getDescriptor().getDimensions()[1],
                                                                batchSize,
                                                                !isInferenceOnly(),
                                                                stream,
                                                                epsilon,
                                                                maxMagnitude);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT32) {
        launchMeanAbsolutePercentageError<uint32_t, half, half>(labelsInput.get().getMemPtr(),
                                                                featureInput.get().getMemPtr(),
                                                                featureOutput.get().getMemPtr(),
                                                                isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                                featureOutput.get().getDescriptor().getDimensions()[1],
                                                                batchSize,
                                                                !isInferenceOnly(),
                                                                stream,
                                                                epsilon,
                                                                maxMagnitude);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::BOOLEAN) {
        launchMeanAbsolutePercentageError<bool, half, half>(labelsInput.get().getMemPtr(),
                                                            featureInput.get().getMemPtr(),
                                                            featureOutput.get().getMemPtr(),
                                                            isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                            featureOutput.get().getDescriptor().getDimensions()[1],
                                                            batchSize,
                                                            !isInferenceOnly(),
                                                            stream,
                                                            epsilon,
                                                            maxMagnitude);
    } else {
        assert(false);
    }
}

void MeanAbsolutePercentageError::launchMeanAbsolutePercentageErrorWithFP16PredictionsAndFP32Loss() {
    assert(featureInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16);
    assert(featureOutput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32);

    if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
        launchMeanAbsolutePercentageError<half, half, float>(labelsInput.get().getMemPtr(),
                                                             featureInput.get().getMemPtr(),
                                                             featureOutput.get().getMemPtr(),
                                                             isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                             featureOutput.get().getDescriptor().getDimensions()[1],
                                                             batchSize,
                                                             !isInferenceOnly(),
                                                             stream,
                                                             epsilon,
                                                             maxMagnitude);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32) {
        launchMeanAbsolutePercentageError<float, half, float>(labelsInput.get().getMemPtr(),
                                                              featureInput.get().getMemPtr(),
                                                              featureOutput.get().getMemPtr(),
                                                              isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                              featureOutput.get().getDescriptor().getDimensions()[1],
                                                              batchSize,
                                                              !isInferenceOnly(),
                                                              stream,
                                                              epsilon,
                                                              maxMagnitude);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT8) {
        launchMeanAbsolutePercentageError<uint8_t, half, float>(labelsInput.get().getMemPtr(),
                                                                featureInput.get().getMemPtr(),
                                                                featureOutput.get().getMemPtr(),
                                                                isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                                featureOutput.get().getDescriptor().getDimensions()[1],
                                                                batchSize,
                                                                !isInferenceOnly(),
                                                                stream,
                                                                epsilon,
                                                                maxMagnitude);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT16) {
        launchMeanAbsolutePercentageError<uint16_t, half, float>(labelsInput.get().getMemPtr(),
                                                                 featureInput.get().getMemPtr(),
                                                                 featureOutput.get().getMemPtr(),
                                                                 isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                                 featureOutput.get().getDescriptor().getDimensions()[1],
                                                                 batchSize,
                                                                 !isInferenceOnly(),
                                                                 stream,
                                                                 epsilon,
                                                                 maxMagnitude);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT32) {
        launchMeanAbsolutePercentageError<uint32_t, half, float>(labelsInput.get().getMemPtr(),
                                                                 featureInput.get().getMemPtr(),
                                                                 featureOutput.get().getMemPtr(),
                                                                 isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                                 featureOutput.get().getDescriptor().getDimensions()[1],
                                                                 batchSize,
                                                                 !isInferenceOnly(),
                                                                 stream,
                                                                 epsilon,
                                                                 maxMagnitude);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::BOOLEAN) {
        launchMeanAbsolutePercentageError<bool, half, float>(labelsInput.get().getMemPtr(),
                                                             featureInput.get().getMemPtr(),
                                                             featureOutput.get().getMemPtr(),
                                                             isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                             featureOutput.get().getDescriptor().getDimensions()[1],
                                                             batchSize,
                                                             !isInferenceOnly(),
                                                             stream,
                                                             epsilon,
                                                             maxMagnitude);
    } else {
        assert(false);
    }
}

void MeanAbsolutePercentageError::launchMeanAbsolutePercentageErrorWithFP32PredictionsAndFP16Loss() {
    assert(featureInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32);
    assert(featureOutput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16);

    if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
        launchMeanAbsolutePercentageError<half, float, half>(labelsInput.get().getMemPtr(),
                                                             featureInput.get().getMemPtr(),
                                                             featureOutput.get().getMemPtr(),
                                                             isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                             featureOutput.get().getDescriptor().getDimensions()[1],
                                                             batchSize,
                                                             !isInferenceOnly(),
                                                             stream,
                                                             epsilon,
                                                             maxMagnitude);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32) {
        launchMeanAbsolutePercentageError<float, float, half>(labelsInput.get().getMemPtr(),
                                                              featureInput.get().getMemPtr(),
                                                              featureOutput.get().getMemPtr(),
                                                              isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                              featureOutput.get().getDescriptor().getDimensions()[1],
                                                              batchSize,
                                                              !isInferenceOnly(),
                                                              stream,
                                                              epsilon,
                                                              maxMagnitude);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT8) {
        launchMeanAbsolutePercentageError<uint8_t, float, half>(labelsInput.get().getMemPtr(),
                                                                featureInput.get().getMemPtr(),
                                                                featureOutput.get().getMemPtr(),
                                                                isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                                featureOutput.get().getDescriptor().getDimensions()[1],
                                                                batchSize,
                                                                !isInferenceOnly(),
                                                                stream,
                                                                epsilon,
                                                                maxMagnitude);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT16) {
        launchMeanAbsolutePercentageError<uint16_t, float, half>(labelsInput.get().getMemPtr(),
                                                                 featureInput.get().getMemPtr(),
                                                                 featureOutput.get().getMemPtr(),
                                                                 isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                                 featureOutput.get().getDescriptor().getDimensions()[1],
                                                                 batchSize,
                                                                 !isInferenceOnly(),
                                                                 stream,
                                                                 epsilon,
                                                                 maxMagnitude);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT32) {
        launchMeanAbsolutePercentageError<uint32_t, float, half>(labelsInput.get().getMemPtr(),
                                                                 featureInput.get().getMemPtr(),
                                                                 featureOutput.get().getMemPtr(),
                                                                 isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                                 featureOutput.get().getDescriptor().getDimensions()[1],
                                                                 batchSize,
                                                                 !isInferenceOnly(),
                                                                 stream,
                                                                 epsilon,
                                                                 maxMagnitude);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::BOOLEAN) {
        launchMeanAbsolutePercentageError<bool, float, half>(labelsInput.get().getMemPtr(),
                                                             featureInput.get().getMemPtr(),
                                                             featureOutput.get().getMemPtr(),
                                                             isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                             featureOutput.get().getDescriptor().getDimensions()[1],
                                                             batchSize,
                                                             !isInferenceOnly(),
                                                             stream,
                                                             epsilon,
                                                             maxMagnitude);
    } else {
        assert(false);
    }
}

void MeanAbsolutePercentageError::launchMeanAbsolutePercentageErrorWithFP32PredictionsAndFP32Loss() {
    assert(featureInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32);
    assert(featureOutput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32);

    if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
        launchMeanAbsolutePercentageError<half, float, float>(labelsInput.get().getMemPtr(),
                                                              featureInput.get().getMemPtr(),
                                                              featureOutput.get().getMemPtr(),
                                                              isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                              featureOutput.get().getDescriptor().getDimensions()[1],
                                                              batchSize,
                                                              !isInferenceOnly(),
                                                              stream,
                                                              epsilon,
                                                              maxMagnitude);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32) {
        launchMeanAbsolutePercentageError<float, float, float>(labelsInput.get().getMemPtr(),
                                                               featureInput.get().getMemPtr(),
                                                               featureOutput.get().getMemPtr(),
                                                               isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                               featureOutput.get().getDescriptor().getDimensions()[1],
                                                               batchSize,
                                                               !isInferenceOnly(),
                                                               stream,
                                                               epsilon,
                                                               maxMagnitude);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT8) {
        launchMeanAbsolutePercentageError<uint8_t, float, float>(labelsInput.get().getMemPtr(),
                                                                 featureInput.get().getMemPtr(),
                                                                 featureOutput.get().getMemPtr(),
                                                                 isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                                 featureOutput.get().getDescriptor().getDimensions()[1],
                                                                 batchSize,
                                                                 !isInferenceOnly(),
                                                                 stream,
                                                                 epsilon,
                                                                 maxMagnitude);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT16) {
        launchMeanAbsolutePercentageError<uint16_t, float, float>(labelsInput.get().getMemPtr(),
                                                                  featureInput.get().getMemPtr(),
                                                                  featureOutput.get().getMemPtr(),
                                                                  isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                                  featureOutput.get().getDescriptor().getDimensions()[1],
                                                                  batchSize,
                                                                  !isInferenceOnly(),
                                                                  stream,
                                                                  epsilon,
                                                                  maxMagnitude);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT32) {
        launchMeanAbsolutePercentageError<uint32_t, float, float>(labelsInput.get().getMemPtr(),
                                                                  featureInput.get().getMemPtr(),
                                                                  featureOutput.get().getMemPtr(),
                                                                  isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                                  featureOutput.get().getDescriptor().getDimensions()[1],
                                                                  batchSize,
                                                                  !isInferenceOnly(),
                                                                  stream,
                                                                  epsilon,
                                                                  maxMagnitude);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::BOOLEAN) {
        launchMeanAbsolutePercentageError<bool, float, float>(labelsInput.get().getMemPtr(),
                                                              featureInput.get().getMemPtr(),
                                                              featureOutput.get().getMemPtr(),
                                                              isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                              featureOutput.get().getDescriptor().getDimensions()[1],
                                                              batchSize,
                                                              !isInferenceOnly(),
                                                              stream,
                                                              epsilon,
                                                              maxMagnitude);
    } else {
        assert(false);
    }
}
