#include <optional>
#include <limits>
#include "DeepLearning/Implementation/Layers/Loss/CrossEntropy.h"

#include "DeepLearning/Implementation/ThorError.h"
using namespace ThorImplementation;
using namespace std;

namespace {

uint64_t productDimensions(const vector<uint64_t>& dims, size_t begin, size_t end) {
    uint64_t result = 1;
    THOR_THROW_IF_FALSE(begin <= end && end <= dims.size());
    for (size_t i = begin; i < end; ++i) {
        THOR_THROW_IF_FALSE(dims[i] > 0);
        result *= dims[i];
    }
    return result;
}

bool sparseLabelDimensionsMatchFeaturePrefix(const vector<uint64_t>& labelDimensions, const vector<uint64_t>& featureInputDimensions) {
    THOR_THROW_IF_FALSE(featureInputDimensions.size() >= 2);
    const size_t prefixRank = featureInputDimensions.size() - 1;
    if (labelDimensions.size() == prefixRank) {
        for (size_t i = 0; i < prefixRank; ++i) {
            if (labelDimensions[i] != featureInputDimensions[i])
                return false;
        }
        return true;
    }
    if (labelDimensions.size() == prefixRank + 1 && labelDimensions.back() == 1) {
        for (size_t i = 0; i < prefixRank; ++i) {
            if (labelDimensions[i] != featureInputDimensions[i])
                return false;
        }
        return true;
    }
    return false;
}

}  // namespace

CrossEntropy::CrossEntropy() : Loss(DataType::FP32) { crossEntropyLossType = CrossEntropyLossType::UNINITIALIZED; }

CrossEntropy::CrossEntropy(CrossEntropyLossType crossEntropyLossType, DataType lossDataType, bool indexLabels)
    : Loss(lossDataType) {
    // Just to be clear, index labels is a feature for categorical only:
    THOR_THROW_IF_FALSE(!(crossEntropyLossType == CrossEntropyLossType::BINARY && indexLabels == true));

    this->indexLabels = indexLabels;

    THOR_THROW_IF_FALSE(crossEntropyLossType == CrossEntropyLossType::BINARY || crossEntropyLossType == CrossEntropyLossType::CATEGORICAL);
    this->crossEntropyLossType = crossEntropyLossType;
}

CrossEntropy::~CrossEntropy() {}

std::optional<Tensor> CrossEntropy::createErrorOutputTensor(bool backPropagateError) {
    // CrossEntropy computes dLoss/dPredictions, so the error tensor must match the prediction
    // tensor descriptor.  This keeps the normal Layer connection invariant intact and lets the
    // same loss implementation train FP16 and FP32 prediction graphs.
    if (backPropagateError && !isInferenceOnly()) {
        THOR_THROW_IF_FALSE(featureInput.has_value());
        return featureInput.value().clone();
    }
    return std::nullopt;
}

void CrossEntropy::compileImpl() {
    Layer::compileImpl();
    THOR_THROW_IF_FALSE(featureInput.has_value());
    THOR_THROW_IF_FALSE(featureOutput.has_value());
    THOR_THROW_IF_FALSE(featureInput.value().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    THOR_THROW_IF_FALSE(featureInput.value().getDescriptor().getDataType() == DataType::FP16 ||
                        featureInput.value().getDescriptor().getDataType() == DataType::FP32);
    THOR_THROW_IF_FALSE(featureOutput.value().getDescriptor().getDataType() == DataType::FP16 ||
                        featureOutput.value().getDescriptor().getDataType() == DataType::FP32);
    if (crossEntropyLossType == CrossEntropyLossType::BINARY) {
        bool oneDimension = (featureInput.value().getDimensions().size() == 1);
        bool twoDimensionsSecondIsSingleton =
            (featureInput.value().getDimensions().size() == 2 && featureInput.value().getDimensions()[1] == 1);
        THOR_THROW_IF_FALSE(oneDimension || twoDimensionsSecondIsSingleton);
    } else {
        // CrossEntropyLossType::CATEGORICAL.  The final tensor dimension is the
        // class dimension; all preceding dimensions are flattened into the
        // effective item/batch dimension for the CUDA kernel.  This covers both
        // ordinary [batch, classes] classification and sequence losses such as
        // [batch, tokens, vocab].
        const vector<uint64_t> featureInputDimensions = featureInput.value().getDescriptor().getDimensions();
        THOR_THROW_IF_FALSE(featureInputDimensions.size() >= 2);
        THOR_THROW_IF_FALSE(featureInputDimensions.back() > 1);
    }

    if (!isInferenceOnly()) {
        THOR_THROW_IF_FALSE(errorOutput.has_value());
        THOR_THROW_IF_FALSE(errorOutput.value().isInitialized());
        THOR_THROW_IF_FALSE(errorOutput.value().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        THOR_THROW_IF_FALSE(errorOutput.value().getPlacement().getDeviceNum() == featureInput.value().getPlacement().getDeviceNum());
        THOR_THROW_IF_FALSE(errorOutput.value().getDescriptor() == featureInput.value().getDescriptor());

        THOR_THROW_IF_FALSE(labelsInput.has_value());
        THOR_THROW_IF_FALSE(labelsInput.value().isInitialized());
        THOR_THROW_IF_FALSE(labelsInput.value().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        THOR_THROW_IF_FALSE(labelsInput.value().getPlacement().getDeviceNum() == featureInput.value().getPlacement().getDeviceNum());

        THOR_THROW_IF_FALSE(labelsInput.has_value());
        vector<uint64_t> labelDimensions = labelsInput.value().getDescriptor().getDimensions();
        THOR_THROW_IF_FALSE(featureInput.has_value());
        vector<uint64_t> featureInputDimensions = featureInput.value().getDescriptor().getDimensions();

        if (indexLabels) {
            THOR_THROW_IF_FALSE(sparseLabelDimensionsMatchFeaturePrefix(labelDimensions, featureInputDimensions));
        } else {
            // label per class
            THOR_THROW_IF_FALSE(featureInputDimensions == labelDimensions);
        }
    }

    const vector<uint64_t> featureInputDimensions = featureInput.value().getDescriptor().getDimensions();
    if (crossEntropyLossType == CrossEntropyLossType::BINARY) {
        THOR_THROW_IF_FALSE(featureInputDimensions[0] <= std::numeric_limits<unsigned int>::max());
        batchSize = static_cast<unsigned int>(featureInputDimensions[0]);
        numClasses = 2;
    } else {
        const uint64_t effectiveBatchSize = productDimensions(featureInputDimensions, 0, featureInputDimensions.size() - 1);
        THOR_THROW_IF_FALSE(effectiveBatchSize <= std::numeric_limits<unsigned int>::max());
        batchSize = static_cast<unsigned int>(effectiveBatchSize);
        THOR_THROW_IF_FALSE(featureInputDimensions.back() <= std::numeric_limits<uint32_t>::max());
        numClasses = static_cast<uint32_t>(featureInputDimensions.back());
    }
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
    THOR_THROW_IF_FALSE(predictions.value().getDescriptor().getDataType() == DataType::FP16 ||
                        predictions.value().getDescriptor().getDataType() == DataType::FP32);

    stream.waitEvent(labelsStream.putEvent());

    if (predictions.value().getDescriptor().getDataType() == DataType::FP16)
        launchCrossEntropyWithFP16Predictions();
    else if (predictions.value().getDescriptor().getDataType() == DataType::FP32)
        launchCrossEntropyWithFP32Predictions();
    else
        THOR_UNREACHABLE();
}

void CrossEntropy::backProp(std::optional<Tensor> labels, std::optional<Tensor> predictions, std::optional<Tensor> lossGradient, Stream stream) {
    // Cross entropy loss gradient is pre-computed during infer() for efficiency
    THOR_THROW_IF_FALSE(lossGradient.has_value());
    THOR_THROW_IF_FALSE(lossGradient.value().getDataType() == DataType::FP32 ||
           lossGradient.value().getDataType() == DataType::FP16);
}

// Yuck, but at least its flattened and contained.
void CrossEntropy::launchCrossEntropyWithFP16Predictions() {
    THOR_THROW_IF_FALSE(featureInput.value().getDescriptor().getDataType() == DataType::FP16);

    if (featureOutput.value().getDescriptor().getDataType() == DataType::FP16)
        launchCrossEntropyWithFP16PredictionsAndFP16Loss();
    else if (featureOutput.value().getDescriptor().getDataType() == DataType::FP32)
        launchCrossEntropyWithFP16PredictionsAndFP32Loss();
    else
        THOR_UNREACHABLE();
}

void CrossEntropy::launchCrossEntropyWithFP32Predictions() {
    THOR_THROW_IF_FALSE(featureInput.value().getDescriptor().getDataType() == DataType::FP32);

    if (featureOutput.value().getDescriptor().getDataType() == DataType::FP16)
        launchCrossEntropyWithFP32PredictionsAndFP16Loss();
    else if (featureOutput.value().getDescriptor().getDataType() == DataType::FP32)
        launchCrossEntropyWithFP32PredictionsAndFP32Loss();
    else
        THOR_UNREACHABLE();
}

void CrossEntropy::launchCrossEntropyWithFP16PredictionsAndFP16Loss() {
    THOR_THROW_IF_FALSE(featureInput.value().getDescriptor().getDataType() == DataType::FP16);
    THOR_THROW_IF_FALSE(featureOutput.value().getDescriptor().getDataType() == DataType::FP16);

    if (labelsInput.value().getDescriptor().getDataType() == DataType::UINT8) {
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
    } else if (labelsInput.value().getDescriptor().getDataType() == DataType::UINT16) {
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
    } else if (labelsInput.value().getDescriptor().getDataType() == DataType::UINT32) {
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
    } else if (labelsInput.value().getDescriptor().getDataType() == DataType::FP16) {
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
    } else if (labelsInput.value().getDescriptor().getDataType() == DataType::FP32) {
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
    } else if (labelsInput.value().getDescriptor().getDataType() == DataType::BOOLEAN) {
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
    THOR_THROW_IF_FALSE(featureInput.value().getDescriptor().getDataType() == DataType::FP16);
    THOR_THROW_IF_FALSE(featureOutput.value().getDescriptor().getDataType() == DataType::FP32);

    if (labelsInput.value().getDescriptor().getDataType() == DataType::UINT8) {
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
    } else if (labelsInput.value().getDescriptor().getDataType() == DataType::UINT16) {
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
    } else if (labelsInput.value().getDescriptor().getDataType() == DataType::UINT32) {
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
    } else if (labelsInput.value().getDescriptor().getDataType() == DataType::FP16) {
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
    } else if (labelsInput.value().getDescriptor().getDataType() == DataType::FP32) {
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
    } else if (labelsInput.value().getDescriptor().getDataType() == DataType::BOOLEAN) {
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
    THOR_THROW_IF_FALSE(featureInput.value().getDescriptor().getDataType() == DataType::FP32);
    THOR_THROW_IF_FALSE(featureOutput.value().getDescriptor().getDataType() == DataType::FP16);

    if (labelsInput.value().getDescriptor().getDataType() == DataType::UINT8) {
        launchElementWiseCrossEntropyLoss<uint8_t, float, half>(labelsInput.value().getMemPtr(),
                                                                featureInput.value().getMemPtr(),
                                                                featureOutput.value().getMemPtr(),
                                                                isInferenceOnly() ? nullptr : (float *)errorOutput.value().getMemPtr(),
                                                                numClasses,
                                                                batchSize,
                                                                !isInferenceOnly(),
                                                                lossScalingFactor,
                                                                crossEntropyLossType,
                                                                indexLabels,
                                                                stream);
    } else if (labelsInput.value().getDescriptor().getDataType() == DataType::UINT16) {
        launchElementWiseCrossEntropyLoss<uint16_t, float, half>(labelsInput.value().getMemPtr(),
                                                                 featureInput.value().getMemPtr(),
                                                                 featureOutput.value().getMemPtr(),
                                                                 isInferenceOnly() ? nullptr : (float *)errorOutput.value().getMemPtr(),
                                                                 numClasses,
                                                                 batchSize,
                                                                 !isInferenceOnly(),
                                                                 lossScalingFactor,
                                                                 crossEntropyLossType,
                                                                 indexLabels,
                                                                 stream);
    } else if (labelsInput.value().getDescriptor().getDataType() == DataType::UINT32) {
        launchElementWiseCrossEntropyLoss<uint32_t, float, half>(labelsInput.value().getMemPtr(),
                                                                 featureInput.value().getMemPtr(),
                                                                 featureOutput.value().getMemPtr(),
                                                                 isInferenceOnly() ? nullptr : (float *)errorOutput.value().getMemPtr(),
                                                                 numClasses,
                                                                 batchSize,
                                                                 !isInferenceOnly(),
                                                                 lossScalingFactor,
                                                                 crossEntropyLossType,
                                                                 indexLabels,
                                                                 stream);
    } else if (labelsInput.value().getDescriptor().getDataType() == DataType::FP16) {
        launchElementWiseCrossEntropyLoss<half, float, half>(labelsInput.value().getMemPtr(),
                                                             featureInput.value().getMemPtr(),
                                                             featureOutput.value().getMemPtr(),
                                                             isInferenceOnly() ? nullptr : (float *)errorOutput.value().getMemPtr(),
                                                             numClasses,
                                                             batchSize,
                                                             !isInferenceOnly(),
                                                             lossScalingFactor,
                                                             crossEntropyLossType,
                                                             indexLabels,
                                                             stream);
    } else if (labelsInput.value().getDescriptor().getDataType() == DataType::FP32) {
        launchElementWiseCrossEntropyLoss<float, float, half>(labelsInput.value().getMemPtr(),
                                                              featureInput.value().getMemPtr(),
                                                              featureOutput.value().getMemPtr(),
                                                              isInferenceOnly() ? nullptr : (float *)errorOutput.value().getMemPtr(),
                                                              numClasses,
                                                              batchSize,
                                                              !isInferenceOnly(),
                                                              lossScalingFactor,
                                                              crossEntropyLossType,
                                                              indexLabels,
                                                              stream);
    } else if (labelsInput.value().getDescriptor().getDataType() == DataType::BOOLEAN) {
        launchElementWiseCrossEntropyLoss<bool, float, half>(labelsInput.value().getMemPtr(),
                                                             featureInput.value().getMemPtr(),
                                                             featureOutput.value().getMemPtr(),
                                                             isInferenceOnly() ? nullptr : (float *)errorOutput.value().getMemPtr(),
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
    THOR_THROW_IF_FALSE(featureInput.value().getDescriptor().getDataType() == DataType::FP32);
    THOR_THROW_IF_FALSE(featureOutput.value().getDescriptor().getDataType() == DataType::FP32);

    if (labelsInput.value().getDescriptor().getDataType() == DataType::UINT8) {
        launchElementWiseCrossEntropyLoss<uint8_t, float, float>(labelsInput.value().getMemPtr(),
                                                                 featureInput.value().getMemPtr(),
                                                                 featureOutput.value().getMemPtr(),
                                                                 isInferenceOnly() ? nullptr : (float *)errorOutput.value().getMemPtr(),
                                                                 numClasses,
                                                                 batchSize,
                                                                 !isInferenceOnly(),
                                                                 lossScalingFactor,
                                                                 crossEntropyLossType,
                                                                 indexLabels,
                                                                 stream);
    } else if (labelsInput.value().getDescriptor().getDataType() == DataType::UINT16) {
        launchElementWiseCrossEntropyLoss<uint16_t, float, float>(labelsInput.value().getMemPtr(),
                                                                  featureInput.value().getMemPtr(),
                                                                  featureOutput.value().getMemPtr(),
                                                                  isInferenceOnly() ? nullptr : (float *)errorOutput.value().getMemPtr(),
                                                                  numClasses,
                                                                  batchSize,
                                                                  !isInferenceOnly(),
                                                                  lossScalingFactor,
                                                                  crossEntropyLossType,
                                                                  indexLabels,
                                                                  stream);
    } else if (labelsInput.value().getDescriptor().getDataType() == DataType::UINT32) {
        launchElementWiseCrossEntropyLoss<uint32_t, float, float>(labelsInput.value().getMemPtr(),
                                                                  featureInput.value().getMemPtr(),
                                                                  featureOutput.value().getMemPtr(),
                                                                  isInferenceOnly() ? nullptr : (float *)errorOutput.value().getMemPtr(),
                                                                  numClasses,
                                                                  batchSize,
                                                                  !isInferenceOnly(),
                                                                  lossScalingFactor,
                                                                  crossEntropyLossType,
                                                                  indexLabels,
                                                                  stream);
    } else if (labelsInput.value().getDescriptor().getDataType() == DataType::FP16) {
        launchElementWiseCrossEntropyLoss<half, float, float>(labelsInput.value().getMemPtr(),
                                                              featureInput.value().getMemPtr(),
                                                              featureOutput.value().getMemPtr(),
                                                              isInferenceOnly() ? nullptr : (float *)errorOutput.value().getMemPtr(),
                                                              numClasses,
                                                              batchSize,
                                                              !isInferenceOnly(),
                                                              lossScalingFactor,
                                                              crossEntropyLossType,
                                                              indexLabels,
                                                              stream);
    } else if (labelsInput.value().getDescriptor().getDataType() == DataType::FP32) {
        launchElementWiseCrossEntropyLoss<float, float, float>(labelsInput.value().getMemPtr(),
                                                               featureInput.value().getMemPtr(),
                                                               featureOutput.value().getMemPtr(),
                                                               isInferenceOnly() ? nullptr : (float *)errorOutput.value().getMemPtr(),
                                                               numClasses,
                                                               batchSize,
                                                               !isInferenceOnly(),
                                                               lossScalingFactor,
                                                               crossEntropyLossType,
                                                               indexLabels,
                                                               stream);
    } else if (labelsInput.value().getDescriptor().getDataType() == DataType::BOOLEAN) {
        launchElementWiseCrossEntropyLoss<bool, float, float>(labelsInput.value().getMemPtr(),
                                                              featureInput.value().getMemPtr(),
                                                              featureOutput.value().getMemPtr(),
                                                              isInferenceOnly() ? nullptr : (float *)errorOutput.value().getMemPtr(),
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
