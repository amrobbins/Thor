#include "MeanSquaredError.h"

#include "DeepLearning/Implementation/Layers/Loss.h"
#include "Utilities/TensorOperations/Loss/MeanSquaredError.h"

#include <chrono>
#include <thread>

using namespace ThorImplementation;
using namespace std;

MeanSquaredError::~MeanSquaredError() {}

MeanSquaredError::MeanSquaredError(TensorDescriptor::DataType lossDataType) : Loss(lossDataType) {}

void MeanSquaredError::compileImpl() {
    Layer::compileImpl();
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

void MeanSquaredError::infer(Optional<Tensor> predictions, Optional<Tensor> elementLoss, Stream stream) {
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
        launchMeanSquaredErrorWithFP16Predictions();
    } else if (predictions.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32) {
        launchMeanSquaredErrorWithFP32Predictions();
    } else {
        assert(false);
    }
}

void MeanSquaredError::backProp(Optional<Tensor> labels,
                                Optional<Tensor> normalizedPredictions,
                                Optional<Tensor> lossGradient,
                                Stream stream) {
    // Mean squared loss gradient is pre-computed during infer() for efficiency
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

void MeanSquaredError::launchMeanSquaredErrorWithFP16Predictions() {
    assert(featureInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16);

    if (featureOutput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16)
        launchMeanSquaredErrorWithFP16PredictionsAndFP16Loss();
    else if (featureOutput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32)
        launchMeanSquaredErrorWithFP16PredictionsAndFP32Loss();
    else
        assert(false);
}

void MeanSquaredError::launchMeanSquaredErrorWithFP32Predictions() {
    assert(featureInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32);

    if (featureOutput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16)
        launchMeanSquaredErrorWithFP32PredictionsAndFP16Loss();
    else if (featureOutput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32)
        launchMeanSquaredErrorWithFP32PredictionsAndFP32Loss();
    else
        assert(false);
}

void MeanSquaredError::launchMeanSquaredErrorWithFP16PredictionsAndFP16Loss() {
    assert(featureInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16);
    assert(featureOutput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16);

    if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
        launchMeanSquaredError<half, half, half>(labelsInput.get().getMemPtr(),
                                                 featureInput.get().getMemPtr(),
                                                 featureOutput.get().getMemPtr(),
                                                 isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                 featureOutput.get().getDescriptor().getDimensions()[1],
                                                 batchSize,
                                                 !isInferenceOnly(),
                                                 stream);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32) {
        launchMeanSquaredError<float, half, half>(labelsInput.get().getMemPtr(),
                                                  featureInput.get().getMemPtr(),
                                                  featureOutput.get().getMemPtr(),
                                                  isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                  featureOutput.get().getDescriptor().getDimensions()[1],
                                                  batchSize,
                                                  !isInferenceOnly(),
                                                  stream);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT8) {
        launchMeanSquaredError<uint8_t, half, half>(labelsInput.get().getMemPtr(),
                                                    featureInput.get().getMemPtr(),
                                                    featureOutput.get().getMemPtr(),
                                                    isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                    featureOutput.get().getDescriptor().getDimensions()[1],
                                                    batchSize,
                                                    !isInferenceOnly(),
                                                    stream);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT16) {
        launchMeanSquaredError<uint16_t, half, half>(labelsInput.get().getMemPtr(),
                                                     featureInput.get().getMemPtr(),
                                                     featureOutput.get().getMemPtr(),
                                                     isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                     featureOutput.get().getDescriptor().getDimensions()[1],
                                                     batchSize,
                                                     !isInferenceOnly(),
                                                     stream);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT32) {
        launchMeanSquaredError<uint32_t, half, half>(labelsInput.get().getMemPtr(),
                                                     featureInput.get().getMemPtr(),
                                                     featureOutput.get().getMemPtr(),
                                                     isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                     featureOutput.get().getDescriptor().getDimensions()[1],
                                                     batchSize,
                                                     !isInferenceOnly(),
                                                     stream);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::BOOLEAN) {
        launchMeanSquaredError<bool, half, half>(labelsInput.get().getMemPtr(),
                                                 featureInput.get().getMemPtr(),
                                                 featureOutput.get().getMemPtr(),
                                                 isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                 featureOutput.get().getDescriptor().getDimensions()[1],
                                                 batchSize,
                                                 !isInferenceOnly(),
                                                 stream);
    } else {
        assert(false);
    }
}

void MeanSquaredError::launchMeanSquaredErrorWithFP16PredictionsAndFP32Loss() {
    assert(featureInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16);
    assert(featureOutput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32);

    if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
        launchMeanSquaredError<half, half, float>(labelsInput.get().getMemPtr(),
                                                  featureInput.get().getMemPtr(),
                                                  featureOutput.get().getMemPtr(),
                                                  isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                  featureOutput.get().getDescriptor().getDimensions()[1],
                                                  batchSize,
                                                  !isInferenceOnly(),
                                                  stream);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32) {
        launchMeanSquaredError<float, half, float>(labelsInput.get().getMemPtr(),
                                                   featureInput.get().getMemPtr(),
                                                   featureOutput.get().getMemPtr(),
                                                   isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                   featureOutput.get().getDescriptor().getDimensions()[1],
                                                   batchSize,
                                                   !isInferenceOnly(),
                                                   stream);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT8) {
        launchMeanSquaredError<uint8_t, half, float>(labelsInput.get().getMemPtr(),
                                                     featureInput.get().getMemPtr(),
                                                     featureOutput.get().getMemPtr(),
                                                     isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                     featureOutput.get().getDescriptor().getDimensions()[1],
                                                     batchSize,
                                                     !isInferenceOnly(),
                                                     stream);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT16) {
        launchMeanSquaredError<uint16_t, half, float>(labelsInput.get().getMemPtr(),
                                                      featureInput.get().getMemPtr(),
                                                      featureOutput.get().getMemPtr(),
                                                      isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                      featureOutput.get().getDescriptor().getDimensions()[1],
                                                      batchSize,
                                                      !isInferenceOnly(),
                                                      stream);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT32) {
        launchMeanSquaredError<uint32_t, half, float>(labelsInput.get().getMemPtr(),
                                                      featureInput.get().getMemPtr(),
                                                      featureOutput.get().getMemPtr(),
                                                      isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                      featureOutput.get().getDescriptor().getDimensions()[1],
                                                      batchSize,
                                                      !isInferenceOnly(),
                                                      stream);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::BOOLEAN) {
        launchMeanSquaredError<bool, half, float>(labelsInput.get().getMemPtr(),
                                                  featureInput.get().getMemPtr(),
                                                  featureOutput.get().getMemPtr(),
                                                  isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                  featureOutput.get().getDescriptor().getDimensions()[1],
                                                  batchSize,
                                                  !isInferenceOnly(),
                                                  stream);
    } else {
        assert(false);
    }
}

void MeanSquaredError::launchMeanSquaredErrorWithFP32PredictionsAndFP16Loss() {
    assert(featureInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32);
    assert(featureOutput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16);

    if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
        launchMeanSquaredError<half, float, half>(labelsInput.get().getMemPtr(),
                                                  featureInput.get().getMemPtr(),
                                                  featureOutput.get().getMemPtr(),
                                                  isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                  featureOutput.get().getDescriptor().getDimensions()[1],
                                                  batchSize,
                                                  !isInferenceOnly(),
                                                  stream);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32) {
        launchMeanSquaredError<float, float, half>(labelsInput.get().getMemPtr(),
                                                   featureInput.get().getMemPtr(),
                                                   featureOutput.get().getMemPtr(),
                                                   isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                   featureOutput.get().getDescriptor().getDimensions()[1],
                                                   batchSize,
                                                   !isInferenceOnly(),
                                                   stream);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT8) {
        launchMeanSquaredError<uint8_t, float, half>(labelsInput.get().getMemPtr(),
                                                     featureInput.get().getMemPtr(),
                                                     featureOutput.get().getMemPtr(),
                                                     isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                     featureOutput.get().getDescriptor().getDimensions()[1],
                                                     batchSize,
                                                     !isInferenceOnly(),
                                                     stream);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT16) {
        launchMeanSquaredError<uint16_t, float, half>(labelsInput.get().getMemPtr(),
                                                      featureInput.get().getMemPtr(),
                                                      featureOutput.get().getMemPtr(),
                                                      isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                      featureOutput.get().getDescriptor().getDimensions()[1],
                                                      batchSize,
                                                      !isInferenceOnly(),
                                                      stream);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT32) {
        launchMeanSquaredError<uint32_t, float, half>(labelsInput.get().getMemPtr(),
                                                      featureInput.get().getMemPtr(),
                                                      featureOutput.get().getMemPtr(),
                                                      isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                      featureOutput.get().getDescriptor().getDimensions()[1],
                                                      batchSize,
                                                      !isInferenceOnly(),
                                                      stream);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::BOOLEAN) {
        launchMeanSquaredError<bool, float, half>(labelsInput.get().getMemPtr(),
                                                  featureInput.get().getMemPtr(),
                                                  featureOutput.get().getMemPtr(),
                                                  isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                  featureOutput.get().getDescriptor().getDimensions()[1],
                                                  batchSize,
                                                  !isInferenceOnly(),
                                                  stream);
    } else {
        assert(false);
    }
}

void MeanSquaredError::launchMeanSquaredErrorWithFP32PredictionsAndFP32Loss() {
    assert(featureInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32);
    assert(featureOutput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32);

    if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
        launchMeanSquaredError<half, float, float>(labelsInput.get().getMemPtr(),
                                                   featureInput.get().getMemPtr(),
                                                   featureOutput.get().getMemPtr(),
                                                   isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                   featureOutput.get().getDescriptor().getDimensions()[1],
                                                   batchSize,
                                                   !isInferenceOnly(),
                                                   stream);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32) {
        launchMeanSquaredError<float, float, float>(labelsInput.get().getMemPtr(),
                                                    featureInput.get().getMemPtr(),
                                                    featureOutput.get().getMemPtr(),
                                                    isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                    featureOutput.get().getDescriptor().getDimensions()[1],
                                                    batchSize,
                                                    !isInferenceOnly(),
                                                    stream);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT8) {
        launchMeanSquaredError<uint8_t, float, float>(labelsInput.get().getMemPtr(),
                                                      featureInput.get().getMemPtr(),
                                                      featureOutput.get().getMemPtr(),
                                                      isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                      featureOutput.get().getDescriptor().getDimensions()[1],
                                                      batchSize,
                                                      !isInferenceOnly(),
                                                      stream);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT16) {
        launchMeanSquaredError<uint16_t, float, float>(labelsInput.get().getMemPtr(),
                                                       featureInput.get().getMemPtr(),
                                                       featureOutput.get().getMemPtr(),
                                                       isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                       featureOutput.get().getDescriptor().getDimensions()[1],
                                                       batchSize,
                                                       !isInferenceOnly(),
                                                       stream);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT32) {
        launchMeanSquaredError<uint32_t, float, float>(labelsInput.get().getMemPtr(),
                                                       featureInput.get().getMemPtr(),
                                                       featureOutput.get().getMemPtr(),
                                                       isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                       featureOutput.get().getDescriptor().getDimensions()[1],
                                                       batchSize,
                                                       !isInferenceOnly(),
                                                       stream);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::BOOLEAN) {
        launchMeanSquaredError<bool, float, float>(labelsInput.get().getMemPtr(),
                                                   featureInput.get().getMemPtr(),
                                                   featureOutput.get().getMemPtr(),
                                                   isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                   featureOutput.get().getDescriptor().getDimensions()[1],
                                                   batchSize,
                                                   !isInferenceOnly(),
                                                   stream);
    } else {
        assert(false);
    }
}
