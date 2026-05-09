#include <optional>
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
    THOR_THROW_IF_FALSE(featureInput.has_value());
    THOR_THROW_IF_FALSE(featureOutput.has_value());
    THOR_THROW_IF_FALSE(featureInput.value().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    if (crossEntropyLossType == CrossEntropyLossType::BINARY) {
        bool oneDimension = (featureInput.value().getDimensions().size() == 1);
        bool twoDimensionsSecondIsSingleton =
            (featureInput.value().getDimensions().size() == 2 && featureInput.value().getDimensions()[1] == 1);
        THOR_THROW_IF_FALSE(oneDimension || twoDimensionsSecondIsSingleton);
    } else {
        // CrossEntropyLossType::CATEGORICAL
        THOR_THROW_IF_FALSE(featureInput.value().getDescriptor().getDimensions().size() == 2);
    }

    if (!isInferenceOnly()) {
        THOR_THROW_IF_FALSE(errorOutput.has_value());
        THOR_THROW_IF_FALSE(errorOutput.value().isInitialized());
        THOR_THROW_IF_FALSE(errorOutput.value().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        THOR_THROW_IF_FALSE(errorOutput.value().getPlacement().getDeviceNum() == featureInput.value().getPlacement().getDeviceNum());
        THOR_THROW_IF_FALSE(errorOutput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16);

        THOR_THROW_IF_FALSE(labelsInput.has_value());
        THOR_THROW_IF_FALSE(labelsInput.value().isInitialized());
        THOR_THROW_IF_FALSE(labelsInput.value().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        THOR_THROW_IF_FALSE(labelsInput.value().getPlacement().getDeviceNum() == featureInput.value().getPlacement().getDeviceNum());

        THOR_THROW_IF_FALSE(labelsInput.has_value());
        vector<uint64_t> labelDimensions = labelsInput.value().getDescriptor().getDimensions();
        THOR_THROW_IF_FALSE(featureInput.has_value());
        vector<uint64_t> featureInputDimensions = featureInput.value().getDescriptor().getDimensions();

        if (indexLabels) {
            THOR_THROW_IF_FALSE(labelDimensions[0] == featureInputDimensions[0]);
            THOR_THROW_IF_FALSE(labelDimensions.size() == 1 || (labelDimensions.size() == 2 && labelDimensions[1] == 1));
        } else {
            // label per class
            THOR_THROW_IF_FALSE(featureInputDimensions == labelDimensions);
        }
    }

    batchSize = featureInput.value().getDescriptor().getDimensions()[0];

    if (crossEntropyLossType == CrossEntropyLossType::BINARY)
        numClasses = 2;
    else
        numClasses = featureOutput.value().getDimensions()[1];
}

void CrossEntropy::infer(std::optional<Tensor> predictions, std::optional<Tensor> elementLoss, Stream stream) {
    THOR_THROW_IF_FALSE(predictions.has_value());
    THOR_THROW_IF_FALSE(elementLoss.has_value());
    THOR_THROW_IF_FALSE(labelsInput.has_value());
    THOR_THROW_IF_FALSE(elementLoss.value().getDescriptor().getDimensions() == predictions.value().getDescriptor().getDimensions());
    if (!isInferenceOnly())
        THOR_THROW_IF_FALSE(errorOutput.has_value());

    ScopedGpu scopedGpu(predictions.value().getPlacement().getDeviceNum());

    THOR_THROW_IF_FALSE(compiled);
    THOR_THROW_IF_FALSE(predictions.value().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16);

    stream.waitEvent(labelsStream.putEvent());

    if (predictions.value().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16)
        launchCrossEntropyWithFP16Predictions();
    else if (predictions.value().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32)
        launchCrossEntropyWithFP32Predictions();
    else
        THOR_UNREACHABLE();
}

void CrossEntropy::backProp(std::optional<Tensor> labels, std::optional<Tensor> predictions, std::optional<Tensor> lossGradient, Stream stream) {
    // Cross entropy loss gradient is pre-computed during infer() for efficiency
    THOR_THROW_IF_FALSE(lossGradient.has_value());
    THOR_THROW_IF_FALSE(lossGradient.value().getDataType() == TensorDescriptor::DataType::FP32 ||
           lossGradient.value().getDataType() == TensorDescriptor::DataType::FP16);
}

// Yuck, but at least its flattened and contained.
void CrossEntropy::launchCrossEntropyWithFP16Predictions() {
    THOR_THROW_IF_FALSE(featureInput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16);

    if (featureOutput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16)
        launchCrossEntropyWithFP16PredictionsAndFP16Loss();
    else if (featureOutput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32)
        launchCrossEntropyWithFP16PredictionsAndFP32Loss();
    else
        THOR_UNREACHABLE();
}

void CrossEntropy::launchCrossEntropyWithFP32Predictions() {
    THOR_THROW_IF_FALSE(featureInput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32);

    if (featureOutput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16)
        launchCrossEntropyWithFP32PredictionsAndFP16Loss();
    else if (featureOutput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32)
        launchCrossEntropyWithFP32PredictionsAndFP32Loss();
    else
        THOR_UNREACHABLE();
}

void CrossEntropy::launchCrossEntropyWithFP16PredictionsAndFP16Loss() {
    THOR_THROW_IF_FALSE(featureInput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16);
    THOR_THROW_IF_FALSE(featureOutput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16);

    if (labelsInput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT8) {
        launchElementWiseCrossEntropyLoss<uint8_t, half, half>(labelsInput.value().getMemPtr(),
                                                               featureInput.value().getMemPtr(),
                                                               featureOutput.value().getMemPtr(),
                                                               isInferenceOnly() ? nullptr : (half *)errorOutput.value().getMemPtr(),
                                                               numClasses,
                                                               batchSize,
                                                               !isInferenceOnly(),
                                                               lossScalingFactor,
                                                               crossEntropyLossType,
                                                               indexLabels,
                                                               stream);
    } else if (labelsInput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT16) {
        launchElementWiseCrossEntropyLoss<uint16_t, half, half>(labelsInput.value().getMemPtr(),
                                                                featureInput.value().getMemPtr(),
                                                                featureOutput.value().getMemPtr(),
                                                                isInferenceOnly() ? nullptr : (half *)errorOutput.value().getMemPtr(),
                                                                numClasses,
                                                                batchSize,
                                                                !isInferenceOnly(),
                                                                lossScalingFactor,
                                                                crossEntropyLossType,
                                                                indexLabels,
                                                                stream);
    } else if (labelsInput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT32) {
        launchElementWiseCrossEntropyLoss<uint32_t, half, half>(labelsInput.value().getMemPtr(),
                                                                featureInput.value().getMemPtr(),
                                                                featureOutput.value().getMemPtr(),
                                                                isInferenceOnly() ? nullptr : (half *)errorOutput.value().getMemPtr(),
                                                                numClasses,
                                                                batchSize,
                                                                !isInferenceOnly(),
                                                                lossScalingFactor,
                                                                crossEntropyLossType,
                                                                indexLabels,
                                                                stream);
    } else if (labelsInput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
        launchElementWiseCrossEntropyLoss<half, half, half>(labelsInput.value().getMemPtr(),
                                                            featureInput.value().getMemPtr(),
                                                            featureOutput.value().getMemPtr(),
                                                            isInferenceOnly() ? nullptr : (half *)errorOutput.value().getMemPtr(),
                                                            numClasses,
                                                            batchSize,
                                                            !isInferenceOnly(),
                                                            lossScalingFactor,
                                                            crossEntropyLossType,
                                                            indexLabels,
                                                            stream);
    } else if (labelsInput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32) {
        launchElementWiseCrossEntropyLoss<float, half, half>(labelsInput.value().getMemPtr(),
                                                             featureInput.value().getMemPtr(),
                                                             featureOutput.value().getMemPtr(),
                                                             isInferenceOnly() ? nullptr : (half *)errorOutput.value().getMemPtr(),
                                                             numClasses,
                                                             batchSize,
                                                             !isInferenceOnly(),
                                                             lossScalingFactor,
                                                             crossEntropyLossType,
                                                             indexLabels,
                                                             stream);
    } else if (labelsInput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::BOOLEAN) {
        launchElementWiseCrossEntropyLoss<bool, half, half>(labelsInput.value().getMemPtr(),
                                                            featureInput.value().getMemPtr(),
                                                            featureOutput.value().getMemPtr(),
                                                            isInferenceOnly() ? nullptr : (half *)errorOutput.value().getMemPtr(),
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
    THOR_THROW_IF_FALSE(featureInput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16);
    THOR_THROW_IF_FALSE(featureOutput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32);

    if (labelsInput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT8) {
        launchElementWiseCrossEntropyLoss<uint8_t, half, float>(labelsInput.value().getMemPtr(),
                                                                featureInput.value().getMemPtr(),
                                                                featureOutput.value().getMemPtr(),
                                                                isInferenceOnly() ? nullptr : (half *)errorOutput.value().getMemPtr(),
                                                                numClasses,
                                                                batchSize,
                                                                !isInferenceOnly(),
                                                                lossScalingFactor,
                                                                crossEntropyLossType,
                                                                indexLabels,
                                                                stream);
    } else if (labelsInput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT16) {
        launchElementWiseCrossEntropyLoss<uint16_t, half, float>(labelsInput.value().getMemPtr(),
                                                                 featureInput.value().getMemPtr(),
                                                                 featureOutput.value().getMemPtr(),
                                                                 isInferenceOnly() ? nullptr : (half *)errorOutput.value().getMemPtr(),
                                                                 numClasses,
                                                                 batchSize,
                                                                 !isInferenceOnly(),
                                                                 lossScalingFactor,
                                                                 crossEntropyLossType,
                                                                 indexLabels,
                                                                 stream);
    } else if (labelsInput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT32) {
        launchElementWiseCrossEntropyLoss<uint32_t, half, float>(labelsInput.value().getMemPtr(),
                                                                 featureInput.value().getMemPtr(),
                                                                 featureOutput.value().getMemPtr(),
                                                                 isInferenceOnly() ? nullptr : (half *)errorOutput.value().getMemPtr(),
                                                                 numClasses,
                                                                 batchSize,
                                                                 !isInferenceOnly(),
                                                                 lossScalingFactor,
                                                                 crossEntropyLossType,
                                                                 indexLabels,
                                                                 stream);
    } else if (labelsInput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
        launchElementWiseCrossEntropyLoss<half, half, float>(labelsInput.value().getMemPtr(),
                                                             featureInput.value().getMemPtr(),
                                                             featureOutput.value().getMemPtr(),
                                                             isInferenceOnly() ? nullptr : (half *)errorOutput.value().getMemPtr(),
                                                             numClasses,
                                                             batchSize,
                                                             !isInferenceOnly(),
                                                             lossScalingFactor,
                                                             crossEntropyLossType,
                                                             indexLabels,
                                                             stream);
    } else if (labelsInput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32) {
        launchElementWiseCrossEntropyLoss<float, half, float>(labelsInput.value().getMemPtr(),
                                                              featureInput.value().getMemPtr(),
                                                              featureOutput.value().getMemPtr(),
                                                              isInferenceOnly() ? nullptr : (half *)errorOutput.value().getMemPtr(),
                                                              numClasses,
                                                              batchSize,
                                                              !isInferenceOnly(),
                                                              lossScalingFactor,
                                                              crossEntropyLossType,
                                                              indexLabels,
                                                              stream);
    } else if (labelsInput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::BOOLEAN) {
        launchElementWiseCrossEntropyLoss<bool, half, float>(labelsInput.value().getMemPtr(),
                                                             featureInput.value().getMemPtr(),
                                                             featureOutput.value().getMemPtr(),
                                                             isInferenceOnly() ? nullptr : (half *)errorOutput.value().getMemPtr(),
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
    THOR_THROW_IF_FALSE(featureInput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32);
    THOR_THROW_IF_FALSE(featureOutput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16);

    if (labelsInput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT8) {
        launchElementWiseCrossEntropyLoss<uint8_t, float, half>(labelsInput.value().getMemPtr(),
                                                                featureInput.value().getMemPtr(),
                                                                featureOutput.value().getMemPtr(),
                                                                isInferenceOnly() ? nullptr : (half *)errorOutput.value().getMemPtr(),
                                                                numClasses,
                                                                batchSize,
                                                                !isInferenceOnly(),
                                                                lossScalingFactor,
                                                                crossEntropyLossType,
                                                                indexLabels,
                                                                stream);
    } else if (labelsInput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT16) {
        launchElementWiseCrossEntropyLoss<uint16_t, float, half>(labelsInput.value().getMemPtr(),
                                                                 featureInput.value().getMemPtr(),
                                                                 featureOutput.value().getMemPtr(),
                                                                 isInferenceOnly() ? nullptr : (half *)errorOutput.value().getMemPtr(),
                                                                 numClasses,
                                                                 batchSize,
                                                                 !isInferenceOnly(),
                                                                 lossScalingFactor,
                                                                 crossEntropyLossType,
                                                                 indexLabels,
                                                                 stream);
    } else if (labelsInput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT32) {
        launchElementWiseCrossEntropyLoss<uint32_t, float, half>(labelsInput.value().getMemPtr(),
                                                                 featureInput.value().getMemPtr(),
                                                                 featureOutput.value().getMemPtr(),
                                                                 isInferenceOnly() ? nullptr : (half *)errorOutput.value().getMemPtr(),
                                                                 numClasses,
                                                                 batchSize,
                                                                 !isInferenceOnly(),
                                                                 lossScalingFactor,
                                                                 crossEntropyLossType,
                                                                 indexLabels,
                                                                 stream);
    } else if (labelsInput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
        launchElementWiseCrossEntropyLoss<half, float, half>(labelsInput.value().getMemPtr(),
                                                             featureInput.value().getMemPtr(),
                                                             featureOutput.value().getMemPtr(),
                                                             isInferenceOnly() ? nullptr : (half *)errorOutput.value().getMemPtr(),
                                                             numClasses,
                                                             batchSize,
                                                             !isInferenceOnly(),
                                                             lossScalingFactor,
                                                             crossEntropyLossType,
                                                             indexLabels,
                                                             stream);
    } else if (labelsInput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32) {
        launchElementWiseCrossEntropyLoss<float, float, half>(labelsInput.value().getMemPtr(),
                                                              featureInput.value().getMemPtr(),
                                                              featureOutput.value().getMemPtr(),
                                                              isInferenceOnly() ? nullptr : (half *)errorOutput.value().getMemPtr(),
                                                              numClasses,
                                                              batchSize,
                                                              !isInferenceOnly(),
                                                              lossScalingFactor,
                                                              crossEntropyLossType,
                                                              indexLabels,
                                                              stream);
    } else if (labelsInput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::BOOLEAN) {
        launchElementWiseCrossEntropyLoss<bool, float, half>(labelsInput.value().getMemPtr(),
                                                             featureInput.value().getMemPtr(),
                                                             featureOutput.value().getMemPtr(),
                                                             isInferenceOnly() ? nullptr : (half *)errorOutput.value().getMemPtr(),
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
    THOR_THROW_IF_FALSE(featureInput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32);
    THOR_THROW_IF_FALSE(featureOutput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32);

    if (labelsInput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT8) {
        launchElementWiseCrossEntropyLoss<uint8_t, float, float>(labelsInput.value().getMemPtr(),
                                                                 featureInput.value().getMemPtr(),
                                                                 featureOutput.value().getMemPtr(),
                                                                 isInferenceOnly() ? nullptr : (half *)errorOutput.value().getMemPtr(),
                                                                 numClasses,
                                                                 batchSize,
                                                                 !isInferenceOnly(),
                                                                 lossScalingFactor,
                                                                 crossEntropyLossType,
                                                                 indexLabels,
                                                                 stream);
    } else if (labelsInput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT16) {
        launchElementWiseCrossEntropyLoss<uint16_t, float, float>(labelsInput.value().getMemPtr(),
                                                                  featureInput.value().getMemPtr(),
                                                                  featureOutput.value().getMemPtr(),
                                                                  isInferenceOnly() ? nullptr : (half *)errorOutput.value().getMemPtr(),
                                                                  numClasses,
                                                                  batchSize,
                                                                  !isInferenceOnly(),
                                                                  lossScalingFactor,
                                                                  crossEntropyLossType,
                                                                  indexLabels,
                                                                  stream);
    } else if (labelsInput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT32) {
        launchElementWiseCrossEntropyLoss<uint32_t, float, float>(labelsInput.value().getMemPtr(),
                                                                  featureInput.value().getMemPtr(),
                                                                  featureOutput.value().getMemPtr(),
                                                                  isInferenceOnly() ? nullptr : (half *)errorOutput.value().getMemPtr(),
                                                                  numClasses,
                                                                  batchSize,
                                                                  !isInferenceOnly(),
                                                                  lossScalingFactor,
                                                                  crossEntropyLossType,
                                                                  indexLabels,
                                                                  stream);
    } else if (labelsInput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
        launchElementWiseCrossEntropyLoss<half, float, float>(labelsInput.value().getMemPtr(),
                                                              featureInput.value().getMemPtr(),
                                                              featureOutput.value().getMemPtr(),
                                                              isInferenceOnly() ? nullptr : (half *)errorOutput.value().getMemPtr(),
                                                              numClasses,
                                                              batchSize,
                                                              !isInferenceOnly(),
                                                              lossScalingFactor,
                                                              crossEntropyLossType,
                                                              indexLabels,
                                                              stream);
    } else if (labelsInput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32) {
        launchElementWiseCrossEntropyLoss<float, float, float>(labelsInput.value().getMemPtr(),
                                                               featureInput.value().getMemPtr(),
                                                               featureOutput.value().getMemPtr(),
                                                               isInferenceOnly() ? nullptr : (half *)errorOutput.value().getMemPtr(),
                                                               numClasses,
                                                               batchSize,
                                                               !isInferenceOnly(),
                                                               lossScalingFactor,
                                                               crossEntropyLossType,
                                                               indexLabels,
                                                               stream);
    } else if (labelsInput.value().getDescriptor().getDataType() == TensorDescriptor::DataType::BOOLEAN) {
        launchElementWiseCrossEntropyLoss<bool, float, float>(labelsInput.value().getMemPtr(),
                                                              featureInput.value().getMemPtr(),
                                                              featureOutput.value().getMemPtr(),
                                                              isInferenceOnly() ? nullptr : (half *)errorOutput.value().getMemPtr(),
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
