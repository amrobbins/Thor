#include "MeanSquaredError.h"

#include "DeepLearning/Implementation/Layers/Loss.h"
#include "Utilities/TensorOperations/Loss/CrossEntropyLoss.h"
#include "Utilities/TensorOperations/Loss/MeanSquaredError.h"
#include "Utilities/TensorOperations/Misc/BatchReduce.h"

#include <chrono>
#include <thread>

using namespace ThorImplementation;

MeanSquaredError::~MeanSquaredError() {
    if (batchReduce)
        delete batchReduce;
    batchReduce = nullptr;
}

MeanSquaredError::MeanSquaredError() : Loss() { batchReduce = nullptr; }

void MeanSquaredError::compile() {
    assert(featureInput.isPresent());
    assert(featureOutput.isPresent());
    assert(featureInput.get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert(featureInput.get().getDescriptor().getDimensions().size() == 2);

    if (!isInferenceOnly()) {
        assert(errorOutput.isPresent());
        assert(errorOutput.get().isInitialized());
        assert(errorOutput.get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        assert(errorOutput.get().getPlacement().getDeviceNum() == featureInput.get().getPlacement().getDeviceNum());
        assert(errorOutput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16);

        assert(labelsInput.isPresent());
        assert(labelsInput.get().isInitialized());
        assert(labelsInput.get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        assert(labelsInput.get().getPlacement().getDeviceNum() == featureInput.get().getPlacement().getDeviceNum());

        labelsDataType = labelsInput.get().getDescriptor().getDataType();
        assert(labelsInput.isPresent());
        vector<uint64_t> labelDimensions = labelsInput.get().getDescriptor().getDimensions();
        assert(featureInput.isPresent());
        vector<uint64_t> featureInputDimensions = featureInput.get().getDescriptor().getDimensions();
        assert(featureInputDimensions == labelDimensions);
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
    assert(predictions.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16);

    stream.waitEvent(labelsStream.putEvent());

    if (predictions.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
        if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
            if (elementLoss.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
                launchMeanSquaredError<half, half, half>(labelsInput.get().getMemPtr(),
                                                         predictions.get().getMemPtr(),
                                                         elementLoss.get().getMemPtr(),
                                                         isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                         elementLoss.get().getDescriptor().getDimensions()[1],
                                                         batchSize,
                                                         stream,
                                                         true,  // FIXME
                                                         !isInferenceOnly());
            } else {
                assert(elementLoss.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32);
                launchMeanSquaredError<half, half, float>(labelsInput.get().getMemPtr(),
                                                          predictions.get().getMemPtr(),
                                                          elementLoss.get().getMemPtr(),
                                                          isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                          elementLoss.get().getDescriptor().getDimensions()[1],
                                                          batchSize,
                                                          stream,
                                                          true,  // FIXME
                                                          !isInferenceOnly());
            }
        } else {
            assert(labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32);
            if (elementLoss.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
                launchMeanSquaredError<float, half, half>(labelsInput.get().getMemPtr(),
                                                          predictions.get().getMemPtr(),
                                                          elementLoss.get().getMemPtr(),
                                                          isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                          elementLoss.get().getDescriptor().getDimensions()[1],
                                                          batchSize,
                                                          stream,
                                                          true,  // FIXME
                                                          !isInferenceOnly());
            } else {
                assert(elementLoss.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32);
                launchMeanSquaredError<float, half, float>(labelsInput.get().getMemPtr(),
                                                           predictions.get().getMemPtr(),
                                                           elementLoss.get().getMemPtr(),
                                                           isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                           elementLoss.get().getDescriptor().getDimensions()[1],
                                                           batchSize,
                                                           stream,
                                                           true,  // FIXME
                                                           !isInferenceOnly());
            }
        }
    } else {
        assert(predictions.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32);
        if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
            if (elementLoss.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
                launchMeanSquaredError<half, half, half>(labelsInput.get().getMemPtr(),
                                                         predictions.get().getMemPtr(),
                                                         elementLoss.get().getMemPtr(),
                                                         isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                         elementLoss.get().getDescriptor().getDimensions()[1],
                                                         batchSize,
                                                         stream,
                                                         true,  // FIXME
                                                         !isInferenceOnly());
            } else {
                assert(elementLoss.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32);
                launchMeanSquaredError<half, half, float>(labelsInput.get().getMemPtr(),
                                                          predictions.get().getMemPtr(),
                                                          elementLoss.get().getMemPtr(),
                                                          isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                          elementLoss.get().getDescriptor().getDimensions()[1],
                                                          batchSize,
                                                          stream,
                                                          true,  // FIXME
                                                          !isInferenceOnly());
            }
        } else {
            assert(labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32);
            if (elementLoss.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
                launchMeanSquaredError<float, half, half>(labelsInput.get().getMemPtr(),
                                                          predictions.get().getMemPtr(),
                                                          elementLoss.get().getMemPtr(),
                                                          isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                          elementLoss.get().getDescriptor().getDimensions()[1],
                                                          batchSize,
                                                          stream,
                                                          true,  // FIXME
                                                          !isInferenceOnly());
            } else {
                assert(elementLoss.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32);
                launchMeanSquaredError<float, half, float>(labelsInput.get().getMemPtr(),
                                                           predictions.get().getMemPtr(),
                                                           elementLoss.get().getMemPtr(),
                                                           isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                           elementLoss.get().getDescriptor().getDimensions()[1],
                                                           batchSize,
                                                           stream,
                                                           true,  // FIXME
                                                           !isInferenceOnly());
            }
        }
    }
}

void MeanSquaredError::backProp(Optional<Tensor> labels,
                                Optional<Tensor> normalizedPredictions,
                                Optional<Tensor> lossGradient,
                                Stream stream) {
    assert(lossGradient.isPresent());
}