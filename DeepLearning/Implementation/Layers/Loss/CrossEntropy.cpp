#include "DeepLearning/Implementation/Layers/Loss/CrossEntropy.h"

using namespace ThorImplementation;
using namespace std;

CrossEntropy::CrossEntropy() : Loss(TensorDescriptor::DataType::FP32) { crossEntropyLossType = CrossEntropyLossType::UNINITIALIZED; }

CrossEntropy::CrossEntropy(CrossEntropyLossType crossEntropyLossType, TensorDescriptor::DataType lossDataType, bool indexLabels)
    : Loss(lossDataType) {
    // Just to be clear, index labels is a feature for categorical only:
    assert(!(crossEntropyLossType == CrossEntropyLossType::BINARY && indexLabels == true));

    this->indexLabels = indexLabels;

    assert(crossEntropyLossType == CrossEntropyLossType::BINARY || crossEntropyLossType == CrossEntropyLossType::CATEGORICAL);
    this->crossEntropyLossType = crossEntropyLossType;
}

CrossEntropy::~CrossEntropy() {}

void CrossEntropy::compile() {
    assert(featureInput.isPresent());
    assert(featureOutput.isPresent());
    assert(featureInput.get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    if (crossEntropyLossType == CrossEntropyLossType::BINARY)
        assert(featureInput.get().getDescriptor().getDimensions().size() == 1);
    else  // CrossEntropyLossType::CATEGORICAL
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

        assert(labelsInput.isPresent());
        vector<uint64_t> labelDimensions = labelsInput.get().getDescriptor().getDimensions();
        assert(featureInput.isPresent());
        vector<uint64_t> featureInputDimensions = featureInput.get().getDescriptor().getDimensions();

        if (indexLabels) {
            assert(labelDimensions[0] == featureInputDimensions[0]);
            assert(labelDimensions.size() == 1 || (labelDimensions.size() == 2 && labelDimensions[1] == 1));
        } else {
            // label per class
            assert(featureInputDimensions == labelDimensions);
        }
    }

    batchSize = featureInput.get().getDescriptor().getDimensions()[0];

    if (crossEntropyLossType == CrossEntropyLossType::BINARY)
        numClasses = 2;
    else
        numClasses = featureOutput.get().getDimensions()[1];
}

void CrossEntropy::infer(Optional<Tensor> predictions, Optional<Tensor> elementLoss, Stream stream) {
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

    if (predictions.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16)
        launchCrossEntropyWithFP16Predictions();
    else if (predictions.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32)
        launchCrossEntropyWithFP32Predictions();
    else
        assert(false);
}

void CrossEntropy::backProp(Optional<Tensor> labels, Optional<Tensor> predictions, Optional<Tensor> lossGradient, Stream stream) {
    // Cross entropy loss gradient is pre-computed during infer() for efficiency
    assert(lossGradient.isPresent());
    assert(lossGradient.get().getDataType() == TensorDescriptor::DataType::FP32 ||
           lossGradient.get().getDataType() == TensorDescriptor::DataType::FP16);
}

// Yuck, but at least its flattened and contained.
void CrossEntropy::launchCrossEntropyWithFP16Predictions() {
    assert(featureInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16);

    if (featureOutput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16)
        launchCrossEntropyWithFP16PredictionsAndFP16Loss();
    else if (featureOutput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32)
        launchCrossEntropyWithFP16PredictionsAndFP32Loss();
    else
        assert(false);
}

void CrossEntropy::launchCrossEntropyWithFP32Predictions() {
    assert(featureInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32);

    if (featureOutput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16)
        launchCrossEntropyWithFP32PredictionsAndFP16Loss();
    else if (featureOutput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32)
        launchCrossEntropyWithFP32PredictionsAndFP32Loss();
    else
        assert(false);
}

void CrossEntropy::launchCrossEntropyWithFP16PredictionsAndFP16Loss() {
    assert(featureInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16);
    assert(featureOutput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16);

    if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT8) {
        launchElementWiseCrossEntropyLoss<uint8_t, half, half>(labelsInput.get().getMemPtr(),
                                                               featureInput.get().getMemPtr(),
                                                               featureOutput.get().getMemPtr(),
                                                               isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                               numClasses,
                                                               batchSize,
                                                               !isInferenceOnly(),
                                                               lossScalingFactor,
                                                               crossEntropyLossType,
                                                               indexLabels,
                                                               stream);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT16) {
        launchElementWiseCrossEntropyLoss<uint16_t, half, half>(labelsInput.get().getMemPtr(),
                                                                featureInput.get().getMemPtr(),
                                                                featureOutput.get().getMemPtr(),
                                                                isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                                numClasses,
                                                                batchSize,
                                                                !isInferenceOnly(),
                                                                lossScalingFactor,
                                                                crossEntropyLossType,
                                                                indexLabels,
                                                                stream);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT32) {
        launchElementWiseCrossEntropyLoss<uint32_t, half, half>(labelsInput.get().getMemPtr(),
                                                                featureInput.get().getMemPtr(),
                                                                featureOutput.get().getMemPtr(),
                                                                isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                                numClasses,
                                                                batchSize,
                                                                !isInferenceOnly(),
                                                                lossScalingFactor,
                                                                crossEntropyLossType,
                                                                indexLabels,
                                                                stream);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
        launchElementWiseCrossEntropyLoss<half, half, half>(labelsInput.get().getMemPtr(),
                                                            featureInput.get().getMemPtr(),
                                                            featureOutput.get().getMemPtr(),
                                                            isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                            numClasses,
                                                            batchSize,
                                                            !isInferenceOnly(),
                                                            lossScalingFactor,
                                                            crossEntropyLossType,
                                                            indexLabels,
                                                            stream);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32) {
        launchElementWiseCrossEntropyLoss<float, half, half>(labelsInput.get().getMemPtr(),
                                                             featureInput.get().getMemPtr(),
                                                             featureOutput.get().getMemPtr(),
                                                             isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                             numClasses,
                                                             batchSize,
                                                             !isInferenceOnly(),
                                                             lossScalingFactor,
                                                             crossEntropyLossType,
                                                             indexLabels,
                                                             stream);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::BOOLEAN) {
        launchElementWiseCrossEntropyLoss<bool, half, half>(labelsInput.get().getMemPtr(),
                                                            featureInput.get().getMemPtr(),
                                                            featureOutput.get().getMemPtr(),
                                                            isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                            numClasses,
                                                            batchSize,
                                                            !isInferenceOnly(),
                                                            lossScalingFactor,
                                                            crossEntropyLossType,
                                                            indexLabels,
                                                            stream);
    } else {
        assert(false);
    }
}

void CrossEntropy::launchCrossEntropyWithFP16PredictionsAndFP32Loss() {
    assert(featureInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16);
    assert(featureOutput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32);

    if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT8) {
        launchElementWiseCrossEntropyLoss<uint8_t, half, float>(labelsInput.get().getMemPtr(),
                                                                featureInput.get().getMemPtr(),
                                                                featureOutput.get().getMemPtr(),
                                                                isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                                numClasses,
                                                                batchSize,
                                                                !isInferenceOnly(),
                                                                lossScalingFactor,
                                                                crossEntropyLossType,
                                                                indexLabels,
                                                                stream);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT16) {
        launchElementWiseCrossEntropyLoss<uint16_t, half, float>(labelsInput.get().getMemPtr(),
                                                                 featureInput.get().getMemPtr(),
                                                                 featureOutput.get().getMemPtr(),
                                                                 isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                                 numClasses,
                                                                 batchSize,
                                                                 !isInferenceOnly(),
                                                                 lossScalingFactor,
                                                                 crossEntropyLossType,
                                                                 indexLabels,
                                                                 stream);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT32) {
        launchElementWiseCrossEntropyLoss<uint32_t, half, float>(labelsInput.get().getMemPtr(),
                                                                 featureInput.get().getMemPtr(),
                                                                 featureOutput.get().getMemPtr(),
                                                                 isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                                 numClasses,
                                                                 batchSize,
                                                                 !isInferenceOnly(),
                                                                 lossScalingFactor,
                                                                 crossEntropyLossType,
                                                                 indexLabels,
                                                                 stream);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
        launchElementWiseCrossEntropyLoss<half, half, float>(labelsInput.get().getMemPtr(),
                                                             featureInput.get().getMemPtr(),
                                                             featureOutput.get().getMemPtr(),
                                                             isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                             numClasses,
                                                             batchSize,
                                                             !isInferenceOnly(),
                                                             lossScalingFactor,
                                                             crossEntropyLossType,
                                                             indexLabels,
                                                             stream);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32) {
        launchElementWiseCrossEntropyLoss<float, half, float>(labelsInput.get().getMemPtr(),
                                                              featureInput.get().getMemPtr(),
                                                              featureOutput.get().getMemPtr(),
                                                              isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                              numClasses,
                                                              batchSize,
                                                              !isInferenceOnly(),
                                                              lossScalingFactor,
                                                              crossEntropyLossType,
                                                              indexLabels,
                                                              stream);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::BOOLEAN) {
        launchElementWiseCrossEntropyLoss<bool, half, float>(labelsInput.get().getMemPtr(),
                                                             featureInput.get().getMemPtr(),
                                                             featureOutput.get().getMemPtr(),
                                                             isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                             numClasses,
                                                             batchSize,
                                                             !isInferenceOnly(),
                                                             lossScalingFactor,
                                                             crossEntropyLossType,
                                                             indexLabels,
                                                             stream);
    } else {
        assert(false);
    }
}

void CrossEntropy::launchCrossEntropyWithFP32PredictionsAndFP16Loss() {
    assert(featureInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32);
    assert(featureOutput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16);

    if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT8) {
        launchElementWiseCrossEntropyLoss<uint8_t, float, half>(labelsInput.get().getMemPtr(),
                                                                featureInput.get().getMemPtr(),
                                                                featureOutput.get().getMemPtr(),
                                                                isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                                numClasses,
                                                                batchSize,
                                                                !isInferenceOnly(),
                                                                lossScalingFactor,
                                                                crossEntropyLossType,
                                                                indexLabels,
                                                                stream);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT16) {
        launchElementWiseCrossEntropyLoss<uint16_t, float, half>(labelsInput.get().getMemPtr(),
                                                                 featureInput.get().getMemPtr(),
                                                                 featureOutput.get().getMemPtr(),
                                                                 isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                                 numClasses,
                                                                 batchSize,
                                                                 !isInferenceOnly(),
                                                                 lossScalingFactor,
                                                                 crossEntropyLossType,
                                                                 indexLabels,
                                                                 stream);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT32) {
        launchElementWiseCrossEntropyLoss<uint32_t, float, half>(labelsInput.get().getMemPtr(),
                                                                 featureInput.get().getMemPtr(),
                                                                 featureOutput.get().getMemPtr(),
                                                                 isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                                 numClasses,
                                                                 batchSize,
                                                                 !isInferenceOnly(),
                                                                 lossScalingFactor,
                                                                 crossEntropyLossType,
                                                                 indexLabels,
                                                                 stream);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
        launchElementWiseCrossEntropyLoss<half, float, half>(labelsInput.get().getMemPtr(),
                                                             featureInput.get().getMemPtr(),
                                                             featureOutput.get().getMemPtr(),
                                                             isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                             numClasses,
                                                             batchSize,
                                                             !isInferenceOnly(),
                                                             lossScalingFactor,
                                                             crossEntropyLossType,
                                                             indexLabels,
                                                             stream);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32) {
        launchElementWiseCrossEntropyLoss<float, float, half>(labelsInput.get().getMemPtr(),
                                                              featureInput.get().getMemPtr(),
                                                              featureOutput.get().getMemPtr(),
                                                              isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                              numClasses,
                                                              batchSize,
                                                              !isInferenceOnly(),
                                                              lossScalingFactor,
                                                              crossEntropyLossType,
                                                              indexLabels,
                                                              stream);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::BOOLEAN) {
        launchElementWiseCrossEntropyLoss<bool, float, half>(labelsInput.get().getMemPtr(),
                                                             featureInput.get().getMemPtr(),
                                                             featureOutput.get().getMemPtr(),
                                                             isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                             numClasses,
                                                             batchSize,
                                                             !isInferenceOnly(),
                                                             lossScalingFactor,
                                                             crossEntropyLossType,
                                                             indexLabels,
                                                             stream);
    } else {
        assert(false);
    }
}

void CrossEntropy::launchCrossEntropyWithFP32PredictionsAndFP32Loss() {
    assert(featureInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32);
    assert(featureOutput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32);

    if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT8) {
        launchElementWiseCrossEntropyLoss<uint8_t, float, float>(labelsInput.get().getMemPtr(),
                                                                 featureInput.get().getMemPtr(),
                                                                 featureOutput.get().getMemPtr(),
                                                                 isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                                 numClasses,
                                                                 batchSize,
                                                                 !isInferenceOnly(),
                                                                 lossScalingFactor,
                                                                 crossEntropyLossType,
                                                                 indexLabels,
                                                                 stream);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT16) {
        launchElementWiseCrossEntropyLoss<uint16_t, float, float>(labelsInput.get().getMemPtr(),
                                                                  featureInput.get().getMemPtr(),
                                                                  featureOutput.get().getMemPtr(),
                                                                  isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                                  numClasses,
                                                                  batchSize,
                                                                  !isInferenceOnly(),
                                                                  lossScalingFactor,
                                                                  crossEntropyLossType,
                                                                  indexLabels,
                                                                  stream);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT32) {
        launchElementWiseCrossEntropyLoss<uint32_t, float, float>(labelsInput.get().getMemPtr(),
                                                                  featureInput.get().getMemPtr(),
                                                                  featureOutput.get().getMemPtr(),
                                                                  isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                                  numClasses,
                                                                  batchSize,
                                                                  !isInferenceOnly(),
                                                                  lossScalingFactor,
                                                                  crossEntropyLossType,
                                                                  indexLabels,
                                                                  stream);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
        launchElementWiseCrossEntropyLoss<half, float, float>(labelsInput.get().getMemPtr(),
                                                              featureInput.get().getMemPtr(),
                                                              featureOutput.get().getMemPtr(),
                                                              isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                              numClasses,
                                                              batchSize,
                                                              !isInferenceOnly(),
                                                              lossScalingFactor,
                                                              crossEntropyLossType,
                                                              indexLabels,
                                                              stream);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32) {
        launchElementWiseCrossEntropyLoss<float, float, float>(labelsInput.get().getMemPtr(),
                                                               featureInput.get().getMemPtr(),
                                                               featureOutput.get().getMemPtr(),
                                                               isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                               numClasses,
                                                               batchSize,
                                                               !isInferenceOnly(),
                                                               lossScalingFactor,
                                                               crossEntropyLossType,
                                                               indexLabels,
                                                               stream);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::BOOLEAN) {
        launchElementWiseCrossEntropyLoss<bool, float, float>(labelsInput.get().getMemPtr(),
                                                              featureInput.get().getMemPtr(),
                                                              featureOutput.get().getMemPtr(),
                                                              isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                              numClasses,
                                                              batchSize,
                                                              !isInferenceOnly(),
                                                              lossScalingFactor,
                                                              crossEntropyLossType,
                                                              indexLabels,
                                                              stream);
    } else {
        assert(false);
    }
}

string CrossEntropy::getType() {
    return string("CrossEntropy ") + (crossEntropyLossType == CrossEntropyLossType::BINARY ? string("(Binary)") : string("(Categorical)"));
}