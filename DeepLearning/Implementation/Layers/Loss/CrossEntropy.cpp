#include "DeepLearning/Implementation/Layers/Loss/CrossEntropy.h"

#include "DeepLearning/Implementation/ThorError.h"
using namespace ThorImplementation;
using namespace std;

CrossEntropy::CrossEntropy() : Loss(TensorDescriptor::DataType::FP32) { crossEntropyLossType = CrossEntropyLossType::UNINITIALIZED; }

CrossEntropy::CrossEntropy(CrossEntropyLossType crossEntropyLossType, TensorDescriptor::DataType lossDataType, bool indexLabels)
    : Loss(lossDataType) {
    // Just to be clear, index labels is a feature for categorical only:
    THOR_THROW_IF_FALSE(!(crossEntropyLossType == CrossEntropyLossType::BINARY && indexLabels == true));

    this->indexLabels = indexLabels;

    THOR_THROW_IF_FALSE(crossEntropyLossType == CrossEntropyLossType::BINARY || crossEntropyLossType == CrossEntropyLossType::CATEGORICAL);
    this->crossEntropyLossType = crossEntropyLossType;
}

CrossEntropy::~CrossEntropy() {}

void CrossEntropy::compileImpl() {
    Layer::compileImpl();
    THOR_THROW_IF_FALSE(featureInput.isPresent());
    THOR_THROW_IF_FALSE(featureOutput.isPresent());
    THOR_THROW_IF_FALSE(featureInput.get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    if (crossEntropyLossType == CrossEntropyLossType::BINARY) {
        bool oneDimension = (featureInput.get().getDimensions().size() == 1);
        bool twoDimensionsSecondIsSingleton =
            (featureInput.get().getDimensions().size() == 2 && featureInput.get().getDimensions()[1] == 1);
        THOR_THROW_IF_FALSE(oneDimension || twoDimensionsSecondIsSingleton);
    } else {
        // CrossEntropyLossType::CATEGORICAL
        THOR_THROW_IF_FALSE(featureInput.get().getDescriptor().getDimensions().size() == 2);
    }

    if (!isInferenceOnly()) {
        THOR_THROW_IF_FALSE(errorOutput.isPresent());
        THOR_THROW_IF_FALSE(errorOutput.get().isInitialized());
        THOR_THROW_IF_FALSE(errorOutput.get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        THOR_THROW_IF_FALSE(errorOutput.get().getPlacement().getDeviceNum() == featureInput.get().getPlacement().getDeviceNum());
        THOR_THROW_IF_FALSE(errorOutput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16);

        THOR_THROW_IF_FALSE(labelsInput.isPresent());
        THOR_THROW_IF_FALSE(labelsInput.get().isInitialized());
        THOR_THROW_IF_FALSE(labelsInput.get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        THOR_THROW_IF_FALSE(labelsInput.get().getPlacement().getDeviceNum() == featureInput.get().getPlacement().getDeviceNum());

        THOR_THROW_IF_FALSE(labelsInput.isPresent());
        vector<uint64_t> labelDimensions = labelsInput.get().getDescriptor().getDimensions();
        THOR_THROW_IF_FALSE(featureInput.isPresent());
        vector<uint64_t> featureInputDimensions = featureInput.get().getDescriptor().getDimensions();

        if (indexLabels) {
            THOR_THROW_IF_FALSE(labelDimensions[0] == featureInputDimensions[0]);
            THOR_THROW_IF_FALSE(labelDimensions.size() == 1 || (labelDimensions.size() == 2 && labelDimensions[1] == 1));
        } else {
            // label per class
            THOR_THROW_IF_FALSE(featureInputDimensions == labelDimensions);
        }
    }

    batchSize = featureInput.get().getDescriptor().getDimensions()[0];

    if (crossEntropyLossType == CrossEntropyLossType::BINARY)
        numClasses = 2;
    else
        numClasses = featureOutput.get().getDimensions()[1];
}

void CrossEntropy::infer(Optional<Tensor> predictions, Optional<Tensor> elementLoss, Stream stream) {
    THOR_THROW_IF_FALSE(predictions.isPresent());
    THOR_THROW_IF_FALSE(elementLoss.isPresent());
    THOR_THROW_IF_FALSE(labelsInput.isPresent());
    THOR_THROW_IF_FALSE(elementLoss.get().getDescriptor().getDimensions() == predictions.get().getDescriptor().getDimensions());
    if (!isInferenceOnly())
        THOR_THROW_IF_FALSE(errorOutput.isPresent());

    ScopedGpu scopedGpu(predictions.get().getPlacement().getDeviceNum());

    THOR_THROW_IF_FALSE(compiled);
    THOR_THROW_IF_FALSE(predictions.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16);

    stream.waitEvent(labelsStream.putEvent());

    if (predictions.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16)
        launchCrossEntropyWithFP16Predictions();
    else if (predictions.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32)
        launchCrossEntropyWithFP32Predictions();
    else
        THOR_UNREACHABLE();
}

void CrossEntropy::backProp(Optional<Tensor> labels, Optional<Tensor> predictions, Optional<Tensor> lossGradient, Stream stream) {
    // Cross entropy loss gradient is pre-computed during infer() for efficiency
    THOR_THROW_IF_FALSE(lossGradient.isPresent());
    THOR_THROW_IF_FALSE(lossGradient.get().getDataType() == TensorDescriptor::DataType::FP32 ||
           lossGradient.get().getDataType() == TensorDescriptor::DataType::FP16);
}

// Yuck, but at least its flattened and contained.
void CrossEntropy::launchCrossEntropyWithFP16Predictions() {
    THOR_THROW_IF_FALSE(featureInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16);

    if (featureOutput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16)
        launchCrossEntropyWithFP16PredictionsAndFP16Loss();
    else if (featureOutput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32)
        launchCrossEntropyWithFP16PredictionsAndFP32Loss();
    else
        THOR_UNREACHABLE();
}

void CrossEntropy::launchCrossEntropyWithFP32Predictions() {
    THOR_THROW_IF_FALSE(featureInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32);

    if (featureOutput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16)
        launchCrossEntropyWithFP32PredictionsAndFP16Loss();
    else if (featureOutput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32)
        launchCrossEntropyWithFP32PredictionsAndFP32Loss();
    else
        THOR_UNREACHABLE();
}

void CrossEntropy::launchCrossEntropyWithFP16PredictionsAndFP16Loss() {
    THOR_THROW_IF_FALSE(featureInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16);
    THOR_THROW_IF_FALSE(featureOutput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16);

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
        THOR_UNREACHABLE();
    }
}

void CrossEntropy::launchCrossEntropyWithFP16PredictionsAndFP32Loss() {
    THOR_THROW_IF_FALSE(featureInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16);
    THOR_THROW_IF_FALSE(featureOutput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32);

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
        THOR_UNREACHABLE();
    }
}

void CrossEntropy::launchCrossEntropyWithFP32PredictionsAndFP16Loss() {
    THOR_THROW_IF_FALSE(featureInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32);
    THOR_THROW_IF_FALSE(featureOutput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16);

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
        THOR_UNREACHABLE();
    }
}

void CrossEntropy::launchCrossEntropyWithFP32PredictionsAndFP32Loss() {
    THOR_THROW_IF_FALSE(featureInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32);
    THOR_THROW_IF_FALSE(featureOutput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32);

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
        THOR_UNREACHABLE();
    }
}

string CrossEntropy::getType() {
    return string("CrossEntropy ") + (crossEntropyLossType == CrossEntropyLossType::BINARY ? string("(Binary)") : string("(Categorical)"));
}
