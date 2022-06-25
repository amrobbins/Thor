#pragma once

#include "DeepLearning/Implementation/Layers/Loss.h"
#include "Utilities/TensorOperations/Loss/CrossEntropyLoss.h"

#include <chrono>
#include <thread>

namespace ThorImplementation {

/**
 * This is equivalent to a softmax activation layer followed by a cross entropy loss.
 *
 * The input predicted values to the loss layer will sum to 1.0 since they are put through a softmax activation first.
 * Those values are clamped to a minimum value of 10e-15, to avoid log(0.0f).
 *
 * https://gombru.github.io/2018/05/23/cross_entropy_loss/
 */

class CategoricalCrossEntropyLoss : public Loss {
   public:
    virtual ~CategoricalCrossEntropyLoss(){};

    CategoricalCrossEntropyLoss() : Loss() {}

    virtual void compile() {
        if (!isInferenceOnly()) {
            assert(labelsInput.isPresent());
            assert(errorOutput.isPresent());
            assert(errorOutput.get().isInitialized());
            assert(errorOutput.get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
            assert(errorOutput.get().getPlacement().getDeviceNum() == featureInput.get().getPlacement().getDeviceNum());
            assert(errorOutput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16);
        }

        assert(labelsInput.isPresent());
        assert(featureInput.isPresent());

        assert(labelsInput.get().isInitialized());
        assert(labelsInput.get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        assert(labelsInput.get().getPlacement().getDeviceNum() == featureInput.get().getPlacement().getDeviceNum());

        vector<uint64_t> featureInputDimensions = featureInput.get().getDescriptor().getDimensions();
        vector<uint64_t> labelDimensions = labelsInput.get().getDescriptor().getDimensions();
        TensorDescriptor::DataType labelsDataType = labelsInput.get().getDescriptor().getDataType();
        perClassLabels = featureInputDimensions == labelDimensions &&
                         (labelsDataType == TensorDescriptor::DataType::UINT8 || labelsDataType == TensorDescriptor::DataType::FP16 ||
                          labelsDataType == TensorDescriptor::DataType::FP32);
        classIndexLabels = labelDimensions.size() == 2 && featureInputDimensions[0] == labelDimensions[0] && labelDimensions[1] == 1 &&
                           (labelsDataType == TensorDescriptor::DataType::UINT8 || labelsDataType == TensorDescriptor::DataType::UINT16 ||
                            labelsDataType == TensorDescriptor::DataType::UINT32);
        assert(perClassLabels ^ classIndexLabels);

        assert(featureInput.get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        assert(featureInput.get().getDescriptor().getDimensions().size() == 2);

        lossWorkspace = Tensor(featureInput.get().getPlacement(),
                               TensorDescriptor(TensorDescriptor::DataType::FP32, featureInput.get().getDescriptor().getDimensions()));

        batchSize = featureInput.get().getDescriptor().getDimensions()[0];
        numClasses = featureInput.get().getDescriptor().getDimensions()[1];

        // When there are two classes and the label is a single 1 or 0, binary cross entropy can be used, instead of categorical cross
        // entropy.
        assert(numClasses >= 2);

        inverseSumOfInputExponentials =
            Tensor(featureInput.get().getPlacement(), TensorDescriptor(TensorDescriptor::DataType::FP32, {batchSize}));

        if (lossScalingFactor == 1) {
        } else if (!isInferenceOnly()) {
            errorOutputWorkspace = errorOutput.get().clone(TensorDescriptor::DataType::FP32);
        }
    }

    virtual void cleanup() {}

    virtual void infer(Optional<Tensor> rawPredictionsIn, Optional<Tensor> normalizedPredictionsOut, Stream stream) {
        assert(rawPredictionsIn.isPresent());
        assert(normalizedPredictionsOut.isPresent());
        assert(featureInput.isPresent());
        assert(rawPredictionsIn.get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        assert(rawPredictionsIn.get() == featureInput.get());
        ScopedGpu scopedGpu(rawPredictionsIn.get().getPlacement().getDeviceNum());

        // Softmax

        // Take the e^rawPrediction
        if (rawPredictionsIn.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
            launchExponentiation((half*)rawPredictionsIn.get().getMemPtr(),
                                 (float*)normalizedPredictionsOut.get().getMemPtr(),
                                 rawPredictionsIn.get().getDescriptor().getTotalNumElements(),
                                 stream);
        } else {
            launchExponentiation((float*)rawPredictionsIn.get().getMemPtr(),
                                 (float*)normalizedPredictionsOut.get().getMemPtr(),
                                 rawPredictionsIn.get().getDescriptor().getTotalNumElements(),
                                 stream);
        }
        // sum the exponentials per batch item
        launchSumManyToOne((float*)normalizedPredictionsOut.get().getMemPtr(),
                           (float*)inverseSumOfInputExponentials.getMemPtr(),
                           numClasses,
                           batchSize,
                           true,
                           false,
                           stream);

        // Normalize predictions to sum to 1 per batch item
        launchMultiplyByScalar((float*)normalizedPredictionsOut.get().getMemPtr(),
                               (float*)inverseSumOfInputExponentials.getMemPtr(),
                               (float*)normalizedPredictionsOut.get().getMemPtr(),
                               numClasses,
                               batchSize,
                               stream);
    }

    // normalizedPredictions is featureOutput and loss is errorOutput
    virtual void computeElementwiseLoss(Tensor labels, Tensor normalizedPredictions, Tensor loss, Stream stream) {
        // Cross Entropy Loss
        if (perClassLabels) {
            if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::UINT8) {
                launchCrossEntropyLoss_perClassLabels((uint8_t*)labels.getMemPtr(),
                                                      (float*)normalizedPredictions.getMemPtr(),
                                                      (float*)lossWorkspace.getMemPtr(),
                                                      (float*)loss.getMemPtr(),
                                                      numClasses,
                                                      batchSize,
                                                      stream);
            } else if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
                launchCrossEntropyLoss_perClassLabels((half*)labels.getMemPtr(),
                                                      (float*)normalizedPredictions.getMemPtr(),
                                                      (float*)lossWorkspace.getMemPtr(),
                                                      (float*)loss.getMemPtr(),
                                                      numClasses,
                                                      batchSize,
                                                      stream);
            } else if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::FP32) {
                launchCrossEntropyLoss_perClassLabels((float*)labels.getMemPtr(),
                                                      (float*)normalizedPredictions.getMemPtr(),
                                                      (float*)lossWorkspace.getMemPtr(),
                                                      (float*)loss.getMemPtr(),
                                                      numClasses,
                                                      batchSize,
                                                      stream);
            } else {
                assert(false);
            }
        } else if (classIndexLabels) {
            if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::UINT8) {
                launchCrossEntropyLoss_classIndexLabels((uint8_t*)labels.getMemPtr(),
                                                        (float*)normalizedPredictions.getMemPtr(),
                                                        (float*)lossWorkspace.getMemPtr(),
                                                        (float*)loss.getMemPtr(),
                                                        numClasses,
                                                        batchSize,
                                                        stream);
            } else if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::UINT16) {
                launchCrossEntropyLoss_classIndexLabels((uint16_t*)labels.getMemPtr(),
                                                        (float*)normalizedPredictions.getMemPtr(),
                                                        (float*)lossWorkspace.getMemPtr(),
                                                        (float*)loss.getMemPtr(),
                                                        numClasses,
                                                        batchSize,
                                                        stream);
            } else if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::UINT32) {
                launchCrossEntropyLoss_classIndexLabels((uint32_t*)labels.getMemPtr(),
                                                        (float*)normalizedPredictions.getMemPtr(),
                                                        (float*)lossWorkspace.getMemPtr(),
                                                        (float*)loss.getMemPtr(),
                                                        numClasses,
                                                        batchSize,
                                                        stream);
            } else {
                assert(false);
            }
        } else {
            assert(false);
        }
    }

    virtual void computeLossGradient(Tensor labels, Tensor normalizedPredictions, Tensor lossGradient, Stream stream) {
        if (perClassLabels) {
            if (lossScalingFactor == 1) {
                if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::UINT8) {
                    launchElementwiseSubtract((float*)normalizedPredictions.getMemPtr(),
                                              (uint8_t*)labels.getMemPtr(),
                                              (half*)lossGradient.getMemPtr(),
                                              lossGradient.getDescriptor().getTotalNumElements(),
                                              stream);
                } else if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
                    launchElementwiseSubtract((float*)normalizedPredictions.getMemPtr(),
                                              (half*)labels.getMemPtr(),
                                              (half*)lossGradient.getMemPtr(),
                                              lossGradient.getDescriptor().getTotalNumElements(),
                                              stream);
                } else {
                    launchElementwiseSubtract((float*)normalizedPredictions.getMemPtr(),
                                              (float*)labels.getMemPtr(),
                                              (half*)lossGradient.getMemPtr(),
                                              lossGradient.getDescriptor().getTotalNumElements(),
                                              stream);
                }
            } else {
                if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::UINT8) {
                    launchElementwiseSubtract((float*)normalizedPredictions.getMemPtr(),
                                              (uint8_t*)labels.getMemPtr(),
                                              (float*)errorOutputWorkspace.get().getMemPtr(),
                                              lossGradient.getDescriptor().getTotalNumElements(),
                                              stream);
                } else if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
                    launchElementwiseSubtract((float*)normalizedPredictions.getMemPtr(),
                                              (half*)labels.getMemPtr(),
                                              (float*)errorOutputWorkspace.get().getMemPtr(),
                                              lossGradient.getDescriptor().getTotalNumElements(),
                                              stream);
                } else if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::FP32) {
                    launchElementwiseSubtract((float*)normalizedPredictions.getMemPtr(),
                                              (float*)labels.getMemPtr(),
                                              (float*)errorOutputWorkspace.get().getMemPtr(),
                                              lossGradient.getDescriptor().getTotalNumElements(),
                                              stream);
                } else {
                    assert(false);
                }
                launchScale((float*)errorOutputWorkspace.get().getMemPtr(),
                            (float)lossScalingFactor,
                            (half*)lossGradient.getMemPtr(),
                            lossGradient.getDescriptor().getTotalNumElements(),
                            stream);
            }
        } else if (classIndexLabels) {
            uint64_t batchSize = featureInput.get().getDescriptor().getDimensions()[0];
            uint64_t numClasses = featureInput.get().getDescriptor().getTotalNumElements() / batchSize;

            if (lossScalingFactor == 1) {
                if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::UINT8) {
                    launchLossGradient_classIndexLabels((uint8_t*)labels.getMemPtr(),
                                                        (float*)normalizedPredictions.getMemPtr(),
                                                        (half*)lossGradient.getMemPtr(),
                                                        numClasses,
                                                        batchSize,
                                                        stream);
                } else if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::UINT16) {
                    launchLossGradient_classIndexLabels((uint16_t*)labels.getMemPtr(),
                                                        (float*)normalizedPredictions.getMemPtr(),
                                                        (half*)lossGradient.getMemPtr(),
                                                        numClasses,
                                                        batchSize,
                                                        stream);
                } else if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::UINT32) {
                    launchLossGradient_classIndexLabels((uint32_t*)labels.getMemPtr(),
                                                        (float*)normalizedPredictions.getMemPtr(),
                                                        (half*)lossGradient.getMemPtr(),
                                                        numClasses,
                                                        batchSize,
                                                        stream);
                } else {
                    assert(false);
                }
            } else {
                if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::UINT8) {
                    launchLossGradient_classIndexLabels((uint8_t*)labels.getMemPtr(),
                                                        (float*)normalizedPredictions.getMemPtr(),
                                                        (float*)errorOutputWorkspace.get().getMemPtr(),
                                                        numClasses,
                                                        batchSize,
                                                        stream);
                } else if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::UINT16) {
                    launchLossGradient_classIndexLabels((uint16_t*)labels.getMemPtr(),
                                                        (float*)normalizedPredictions.getMemPtr(),
                                                        (float*)errorOutputWorkspace.get().getMemPtr(),
                                                        numClasses,
                                                        batchSize,
                                                        stream);
                } else if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::UINT32) {
                    launchLossGradient_classIndexLabels((uint32_t*)labels.getMemPtr(),
                                                        (float*)normalizedPredictions.getMemPtr(),
                                                        (float*)errorOutputWorkspace.get().getMemPtr(),
                                                        numClasses,
                                                        batchSize,
                                                        stream);
                } else {
                    assert(false);
                }
                launchScale((float*)errorOutputWorkspace.get().getMemPtr(),
                            (float)lossScalingFactor,
                            (half*)lossGradient.getMemPtr(),
                            lossGradient.getDescriptor().getTotalNumElements(),
                            stream);
            }
        } else {
            assert(false);
        }

        // FIXME: At the API layer I should offer 3 options: loss per batch, loss per batch item, loss per batch item per class.
        //        Currently this layer outputs loss per batch item, it should output loss per batch item per class and a subsequent layer
        //        should reduce as desired.
    }

    virtual void backProp(Optional<Tensor> labels, Optional<Tensor> normalizedPredictions, Optional<Tensor> lossGradient, Stream stream) {
        assert(lossGradient.isPresent());
    }

   protected:
    bool perClassLabels;
    bool classIndexLabels;

   private:
    unsigned int batchSize;
    unsigned int numClasses;

    Tensor inverseSumOfInputExponentials;
    Tensor lossWorkspace;

    Optional<Tensor> errorOutputWorkspace;
};

}  // namespace ThorImplementation
