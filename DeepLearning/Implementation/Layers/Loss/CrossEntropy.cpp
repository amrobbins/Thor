#include "DeepLearning/Implementation/Layers/Loss/CrossEntropy.h"

using namespace ThorImplementation;
using namespace std;

CrossEntropy::CrossEntropy() {
    computeCategoricalCrossEntropyGradient = false;
    computeBinaryCrossEntropyGradient = false;
}

CrossEntropy::CrossEntropy(bool computeCategoricalCrossEntropyGradient, bool computeBinaryCrossEntropyGradient) {
    assert(!(computeCategoricalCrossEntropyGradient && computeBinaryCrossEntropyGradient));

    this->computeCategoricalCrossEntropyGradient = computeCategoricalCrossEntropyGradient;
    this->computeBinaryCrossEntropyGradient = computeBinaryCrossEntropyGradient;
}

CrossEntropy::~CrossEntropy() {
    if (batchReduce)
        delete batchReduce;
    batchReduce = nullptr;
}

CrossEntropy::CrossEntropy(bool indexLabels) : Loss() {
    batchReduce = nullptr;
    this->indexLabels = indexLabels;
}

void CrossEntropy::compile() {
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

        assert(labelsInput.isPresent());
        vector<uint64_t> labelDimensions = labelsInput.get().getDescriptor().getDimensions();
        assert(featureInput.isPresent());
        vector<uint64_t> featureInputDimensions = featureInput.get().getDescriptor().getDimensions();

        if (indexLabels) {
            assert(featureInputDimensions.size() == 1 || (featureInputDimensions.size() == 2 && featureInputDimensions[1] == 1));
            assert(featureInputDimensions[0] == labelDimensions[0]);
        } else {
            // label per class
            assert(featureInputDimensions == labelDimensions);
        }
    }

    batchSize = featureInput.get().getDescriptor().getDimensions()[0];
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

    if (indexLabels) {
        if (predictions.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16)
            launchCrossEntropyForIndexLabelsWithFP16Predictions();
        else if (predictions.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32)
            launchCrossEntropyForIndexLabelsWithFP32Predictions();
        else
            assert(false);
    } else {
        if (predictions.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16)
            launchCrossEntropyForPerClassLabelsWithFP16Predictions();
        else if (predictions.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32)
            launchCrossEntropyForPerClassLabelsWithFP32Predictions();
        else
            assert(false);
    }
}

void CrossEntropy::backProp(Optional<Tensor> labels, Optional<Tensor> predictions, Optional<Tensor> lossGradient, Stream stream) {
    // Cross entropy loss gradient is pre-computed during infer() for efficiency
    assert(lossGradient.isPresent());
    assert(lossGradient.get().getDataType() == TensorDescriptor::DataType::FP32 ||
           lossGradient.get().getDataType() == TensorDescriptor::DataType::FP16);
}

// Yuck, but at least its flattened and contained.
void CrossEntropy::launchCrossEntropyForIndexLabelsWithFP16Predictions() {
    assert(featureInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16);

    if (featureOutput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16)
        launchCrossEntropyForIndexLabelsWithFP16PredictionsAndFP16Loss();
    else if (featureOutput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32)
        launchCrossEntropyForIndexLabelsWithFP16PredictionsAndFP32Loss();
    else
        assert(false);
}

void CrossEntropy::launchCrossEntropyForIndexLabelsWithFP32Predictions() {
    assert(featureInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32);

    if (featureOutput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16)
        launchCrossEntropyForIndexLabelsWithFP32PredictionsAndFP16Loss();
    else if (featureOutput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32)
        launchCrossEntropyForIndexLabelsWithFP32PredictionsAndFP32Loss();
    else
        assert(false);
}

void CrossEntropy::launchCrossEntropyForPerClassLabelsWithFP16Predictions() {
    assert(featureInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16);

    if (featureOutput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16)
        launchCrossEntropyForPerClassLabelsWithFP16PredictionsAndFP16Loss();
    else if (featureOutput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32)
        launchCrossEntropyForPerClassLabelsWithFP16PredictionsAndFP32Loss();
    else
        assert(false);
}

void CrossEntropy::launchCrossEntropyForPerClassLabelsWithFP32Predictions() {
    assert(featureInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32);

    if (featureOutput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16)
        launchCrossEntropyForPerClassLabelsWithFP32PredictionsAndFP16Loss();
    else if (featureOutput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32)
        launchCrossEntropyForPerClassLabelsWithFP32PredictionsAndFP32Loss();
    else
        assert(false);
}

void CrossEntropy::launchCrossEntropyForIndexLabelsWithFP16PredictionsAndFP16Loss() {
    assert(featureInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16);
    assert(featureOutput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16);

    if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT8) {
        launchElementWiseCrossEntropyLoss_oneHotSpecialCase<uint8_t, half, half>(
            labelsInput.get().getMemPtr(),
            featureInput.get().getMemPtr(),
            featureOutput.get().getMemPtr(),
            isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
            featureOutput.get().getDescriptor().getDimensions()[1],
            batchSize,
            !isInferenceOnly(),
            lossScalingFactor,
            computeCategoricalCrossEntropyGradient,
            computeBinaryCrossEntropyGradient,
            stream);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT16) {
        launchElementWiseCrossEntropyLoss_oneHotSpecialCase<uint16_t, half, half>(
            labelsInput.get().getMemPtr(),
            featureInput.get().getMemPtr(),
            featureOutput.get().getMemPtr(),
            isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
            featureOutput.get().getDescriptor().getDimensions()[1],
            batchSize,
            !isInferenceOnly(),
            lossScalingFactor,
            computeCategoricalCrossEntropyGradient,
            computeBinaryCrossEntropyGradient,
            stream);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT32) {
        launchElementWiseCrossEntropyLoss_oneHotSpecialCase<uint32_t, half, half>(
            labelsInput.get().getMemPtr(),
            featureInput.get().getMemPtr(),
            featureOutput.get().getMemPtr(),
            isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
            featureOutput.get().getDescriptor().getDimensions()[1],
            batchSize,
            !isInferenceOnly(),
            lossScalingFactor,
            computeCategoricalCrossEntropyGradient,
            computeBinaryCrossEntropyGradient,
            stream);
    } else {
        assert(false);
    }
}

void CrossEntropy::launchCrossEntropyForIndexLabelsWithFP16PredictionsAndFP32Loss() {
    assert(featureInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16);
    assert(featureOutput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32);

    if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT8) {
        launchElementWiseCrossEntropyLoss_oneHotSpecialCase<uint8_t, half, float>(
            labelsInput.get().getMemPtr(),
            featureInput.get().getMemPtr(),
            featureOutput.get().getMemPtr(),
            isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
            featureOutput.get().getDescriptor().getDimensions()[1],
            batchSize,
            !isInferenceOnly(),
            lossScalingFactor,
            computeCategoricalCrossEntropyGradient,
            computeBinaryCrossEntropyGradient,
            stream);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT16) {
        launchElementWiseCrossEntropyLoss_oneHotSpecialCase<uint16_t, half, float>(
            labelsInput.get().getMemPtr(),
            featureInput.get().getMemPtr(),
            featureOutput.get().getMemPtr(),
            isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
            featureOutput.get().getDescriptor().getDimensions()[1],
            batchSize,
            !isInferenceOnly(),
            lossScalingFactor,
            computeCategoricalCrossEntropyGradient,
            computeBinaryCrossEntropyGradient,
            stream);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT32) {
        launchElementWiseCrossEntropyLoss_oneHotSpecialCase<uint32_t, half, float>(
            labelsInput.get().getMemPtr(),
            featureInput.get().getMemPtr(),
            featureOutput.get().getMemPtr(),
            isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
            featureOutput.get().getDescriptor().getDimensions()[1],
            batchSize,
            !isInferenceOnly(),
            lossScalingFactor,
            computeCategoricalCrossEntropyGradient,
            computeBinaryCrossEntropyGradient,
            stream);
    } else {
        assert(false);
    }
}

void CrossEntropy::launchCrossEntropyForIndexLabelsWithFP32PredictionsAndFP16Loss() {
    assert(featureInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32);
    assert(featureOutput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16);

    if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT8) {
        launchElementWiseCrossEntropyLoss_oneHotSpecialCase<uint8_t, float, half>(
            labelsInput.get().getMemPtr(),
            featureInput.get().getMemPtr(),
            featureOutput.get().getMemPtr(),
            isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
            featureOutput.get().getDescriptor().getDimensions()[1],
            batchSize,
            !isInferenceOnly(),
            lossScalingFactor,
            computeCategoricalCrossEntropyGradient,
            computeBinaryCrossEntropyGradient,
            stream);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT16) {
        launchElementWiseCrossEntropyLoss_oneHotSpecialCase<uint16_t, float, half>(
            labelsInput.get().getMemPtr(),
            featureInput.get().getMemPtr(),
            featureOutput.get().getMemPtr(),
            isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
            featureOutput.get().getDescriptor().getDimensions()[1],
            batchSize,
            !isInferenceOnly(),
            lossScalingFactor,
            computeCategoricalCrossEntropyGradient,
            computeBinaryCrossEntropyGradient,
            stream);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT32) {
        launchElementWiseCrossEntropyLoss_oneHotSpecialCase<uint32_t, float, half>(
            labelsInput.get().getMemPtr(),
            featureInput.get().getMemPtr(),
            featureOutput.get().getMemPtr(),
            isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
            featureOutput.get().getDescriptor().getDimensions()[1],
            batchSize,
            !isInferenceOnly(),
            lossScalingFactor,
            computeCategoricalCrossEntropyGradient,
            computeBinaryCrossEntropyGradient,
            stream);
    } else {
        assert(false);
    }
}

void CrossEntropy::launchCrossEntropyForIndexLabelsWithFP32PredictionsAndFP32Loss() {
    assert(featureInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32);
    assert(featureOutput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32);

    if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT8) {
        launchElementWiseCrossEntropyLoss_oneHotSpecialCase<uint8_t, float, float>(
            labelsInput.get().getMemPtr(),
            featureInput.get().getMemPtr(),
            featureOutput.get().getMemPtr(),
            isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
            featureOutput.get().getDescriptor().getDimensions()[1],
            batchSize,
            !isInferenceOnly(),
            lossScalingFactor,
            computeCategoricalCrossEntropyGradient,
            computeBinaryCrossEntropyGradient,
            stream);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT16) {
        launchElementWiseCrossEntropyLoss_oneHotSpecialCase<uint16_t, float, float>(
            labelsInput.get().getMemPtr(),
            featureInput.get().getMemPtr(),
            featureOutput.get().getMemPtr(),
            isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
            featureOutput.get().getDescriptor().getDimensions()[1],
            batchSize,
            !isInferenceOnly(),
            lossScalingFactor,
            computeCategoricalCrossEntropyGradient,
            computeBinaryCrossEntropyGradient,
            stream);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::UINT32) {
        launchElementWiseCrossEntropyLoss_oneHotSpecialCase<uint32_t, float, float>(
            labelsInput.get().getMemPtr(),
            featureInput.get().getMemPtr(),
            featureOutput.get().getMemPtr(),
            isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
            featureOutput.get().getDescriptor().getDimensions()[1],
            batchSize,
            !isInferenceOnly(),
            lossScalingFactor,
            computeCategoricalCrossEntropyGradient,
            computeBinaryCrossEntropyGradient,
            stream);
    } else {
        assert(false);
    }
}

void CrossEntropy::launchCrossEntropyForPerClassLabelsWithFP16PredictionsAndFP16Loss() {
    assert(featureInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16);
    assert(featureOutput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16);

    if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
        launchElementWiseCrossEntropyLoss<half, half, half>(labelsInput.get().getMemPtr(),
                                                            featureInput.get().getMemPtr(),
                                                            featureOutput.get().getMemPtr(),
                                                            isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                            featureOutput.get().getDescriptor().getDimensions()[1],
                                                            batchSize,
                                                            !isInferenceOnly(),
                                                            lossScalingFactor,
                                                            computeCategoricalCrossEntropyGradient,
                                                            computeBinaryCrossEntropyGradient,
                                                            stream);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32) {
        launchElementWiseCrossEntropyLoss<float, half, half>(labelsInput.get().getMemPtr(),
                                                             featureInput.get().getMemPtr(),
                                                             featureOutput.get().getMemPtr(),
                                                             isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                             featureOutput.get().getDescriptor().getDimensions()[1],
                                                             batchSize,
                                                             !isInferenceOnly(),
                                                             lossScalingFactor,
                                                             computeCategoricalCrossEntropyGradient,
                                                             computeBinaryCrossEntropyGradient,
                                                             stream);
    } else {
        assert(false);
    }
}

void CrossEntropy::launchCrossEntropyForPerClassLabelsWithFP16PredictionsAndFP32Loss() {
    assert(featureInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16);
    assert(featureOutput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32);

    if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
        launchElementWiseCrossEntropyLoss<half, half, float>(labelsInput.get().getMemPtr(),
                                                             featureInput.get().getMemPtr(),
                                                             featureOutput.get().getMemPtr(),
                                                             isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                             featureOutput.get().getDescriptor().getDimensions()[1],
                                                             batchSize,
                                                             !isInferenceOnly(),
                                                             lossScalingFactor,
                                                             computeCategoricalCrossEntropyGradient,
                                                             computeBinaryCrossEntropyGradient,
                                                             stream);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32) {
        launchElementWiseCrossEntropyLoss<float, half, float>(labelsInput.get().getMemPtr(),
                                                              featureInput.get().getMemPtr(),
                                                              featureOutput.get().getMemPtr(),
                                                              isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                              featureOutput.get().getDescriptor().getDimensions()[1],
                                                              batchSize,
                                                              !isInferenceOnly(),
                                                              lossScalingFactor,
                                                              computeCategoricalCrossEntropyGradient,
                                                              computeBinaryCrossEntropyGradient,
                                                              stream);
    } else {
        assert(false);
    }
}

void CrossEntropy::launchCrossEntropyForPerClassLabelsWithFP32PredictionsAndFP16Loss() {
    assert(featureInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32);
    assert(featureOutput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16);

    if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
        launchElementWiseCrossEntropyLoss<half, float, half>(labelsInput.get().getMemPtr(),
                                                             featureInput.get().getMemPtr(),
                                                             featureOutput.get().getMemPtr(),
                                                             isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                             featureOutput.get().getDescriptor().getDimensions()[1],
                                                             batchSize,
                                                             !isInferenceOnly(),
                                                             lossScalingFactor,
                                                             computeCategoricalCrossEntropyGradient,
                                                             computeBinaryCrossEntropyGradient,
                                                             stream);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32) {
        launchElementWiseCrossEntropyLoss<float, float, half>(labelsInput.get().getMemPtr(),
                                                              featureInput.get().getMemPtr(),
                                                              featureOutput.get().getMemPtr(),
                                                              isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                              featureOutput.get().getDescriptor().getDimensions()[1],
                                                              batchSize,
                                                              !isInferenceOnly(),
                                                              lossScalingFactor,
                                                              computeCategoricalCrossEntropyGradient,
                                                              computeBinaryCrossEntropyGradient,
                                                              stream);
    } else {
        assert(false);
    }
}

void CrossEntropy::launchCrossEntropyForPerClassLabelsWithFP32PredictionsAndFP32Loss() {
    assert(featureInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32);
    assert(featureOutput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32);

    if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
        launchElementWiseCrossEntropyLoss<half, float, float>(labelsInput.get().getMemPtr(),
                                                              featureInput.get().getMemPtr(),
                                                              featureOutput.get().getMemPtr(),
                                                              isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                              featureOutput.get().getDescriptor().getDimensions()[1],
                                                              batchSize,
                                                              !isInferenceOnly(),
                                                              lossScalingFactor,
                                                              computeCategoricalCrossEntropyGradient,
                                                              computeBinaryCrossEntropyGradient,
                                                              stream);
    } else if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32) {
        launchElementWiseCrossEntropyLoss<float, float, float>(labelsInput.get().getMemPtr(),
                                                               featureInput.get().getMemPtr(),
                                                               featureOutput.get().getMemPtr(),
                                                               isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                               featureOutput.get().getDescriptor().getDimensions()[1],
                                                               batchSize,
                                                               !isInferenceOnly(),
                                                               lossScalingFactor,
                                                               computeCategoricalCrossEntropyGradient,
                                                               computeBinaryCrossEntropyGradient,
                                                               stream);
    } else {
        assert(false);
    }
}