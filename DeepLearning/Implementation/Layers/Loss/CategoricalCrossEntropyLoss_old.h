/*
#pragma once

#include <optional>
#include "DeepLearning/Implementation/Layers/Loss.h"
#include "Utilities/TensorOperations/Loss/CrossEntropyLoss.h"

#include <chrono>
#include <thread>

namespace ThorImplementation {
*/
/*
FIXME Reconsider the design of this considering Mean Squared Error. Also now that I have soft max with backwards,
      and I have batchReduce, maybe I should redo this altogether and use those pieces. Then in the API layer
      there will be an option to include a softmax layer in front of the categorical cross entropy.
      I do still want to support BATCH_LOSS, CLASSWISE_PER_ELEMENT_LOSS, CLASSWISE_LOSS, ELEMENTWISE_LOSS
      for analysis purposes. When computing batch loss, element wise loss is computed and then batch reduced.
      But first test mean squared error and make sure it works.
*/

/**
 * This is equivalent to a softmax activation layer followed by a cross entropy loss.
 *
 * The input predicted values to the loss layer will sum to 1.0 since they are put through a softmax activation first.
 * Those values are clamped to a minimum value of 10e-15, to avoid log(0.0f).
 *
 * https://gombru.github.io/2018/05/23/cross_entropy_loss/
 */
/*
class CategoricalCrossEntropyLoss : public Loss {
   public:
    virtual ~CategoricalCrossEntropyLoss(){};

    CategoricalCrossEntropyLoss() : Loss() {}

    virtual void compile() {
        if (!isInferenceOnly()) {
            THOR_THROW_IF_FALSE(labelsInput.has_value());
            THOR_THROW_IF_FALSE(errorOutput.has_value());
            THOR_THROW_IF_FALSE(errorOutput.value().isInitialized());
            THOR_THROW_IF_FALSE(errorOutput.value().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
            THOR_THROW_IF_FALSE(errorOutput.value().getPlacement().getDeviceNum() == featureInput.value().getPlacement().getDeviceNum());
            THOR_THROW_IF_FALSE(errorOutput.value().getDescriptor().getDataType() == DataType::FP16);
        }

        THOR_THROW_IF_FALSE(labelsInput.has_value());
        THOR_THROW_IF_FALSE(featureInput.has_value());

        THOR_THROW_IF_FALSE(labelsInput.value().isInitialized());
        THOR_THROW_IF_FALSE(labelsInput.value().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        THOR_THROW_IF_FALSE(labelsInput.value().getPlacement().getDeviceNum() == featureInput.value().getPlacement().getDeviceNum());

        vector<uint64_t> featureInputDimensions = featureInput.value().getDescriptor().getDimensions();
        vector<uint64_t> labelDimensions = labelsInput.value().getDescriptor().getDimensions();
        DataType labelsDataType = labelsInput.value().getDescriptor().getDataType();
        perClassLabels = featureInputDimensions == labelDimensions &&
                         (labelsDataType == DataType::UINT8 || labelsDataType == DataType::FP16 ||
                          labelsDataType == DataType::FP32);
        classIndexLabels = labelDimensions.size() == 2 && featureInputDimensions[0] == labelDimensions[0] && labelDimensions[1] == 1 &&
                           (labelsDataType == DataType::UINT8 || labelsDataType == DataType::UINT16 ||
                            labelsDataType == DataType::UINT32);
        THOR_THROW_IF_FALSE(perClassLabels ^ classIndexLabels);

        THOR_THROW_IF_FALSE(featureInput.value().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        THOR_THROW_IF_FALSE(featureInput.value().getDescriptor().getDimensions().size() == 2);

        lossWorkspace = Tensor(featureInput.value().getPlacement(),
                               TensorDescriptor(DataType::FP32, featureInput.value().getDescriptor().getDimensions()));

        batchSize = featureInput.value().getDescriptor().getDimensions()[0];
        numClasses = featureInput.value().getDescriptor().getDimensions()[1];

        // When there are two classes and the label is a single 1 or 0, binary cross entropy can be used, instead of categorical cross
        // entropy.
        THOR_THROW_IF_FALSE(numClasses >= 2);

        inverseSumOfInputExponentials =
            Tensor(featureInput.value().getPlacement(), TensorDescriptor(DataType::FP32, {batchSize}));

        if (lossScalingFactor == 1) {
        } else if (!isInferenceOnly()) {
            errorOutputWorkspace = errorOutput.value().clone(DataType::FP32);
        }
    }

    virtual void cleanup() {}

    virtual void infer(std::optional<Tensor> rawPredictionsIn, std::optional<Tensor> normalizedPredictionsOut, Stream stream) {
        THOR_THROW_IF_FALSE(rawPredictionsIn.has_value());
        THOR_THROW_IF_FALSE(normalizedPredictionsOut.has_value());
        THOR_THROW_IF_FALSE(featureInput.has_value());
        THOR_THROW_IF_FALSE(rawPredictionsIn.value().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        THOR_THROW_IF_FALSE(rawPredictionsIn.value() == featureInput.value());
        ScopedGpu scopedGpu(rawPredictionsIn.value().getPlacement().getDeviceNum());

        // Softmax

        // Take the e^rawPrediction
        if (rawPredictionsIn.value().getDescriptor().getDataType() == DataType::FP16) {
            launchExponentiation((half*)rawPredictionsIn.value().getMemPtr(),
                                 (float*)normalizedPredictionsOut.value().getMemPtr(),
                                 rawPredictionsIn.value().getDescriptor().getTotalNumElements(),
                                 stream);
        } else {
            launchExponentiation((float*)rawPredictionsIn.value().getMemPtr(),
                                 (float*)normalizedPredictionsOut.value().getMemPtr(),
                                 rawPredictionsIn.value().getDescriptor().getTotalNumElements(),
                                 stream);
        }
        // sum the exponentials per batch item
        launchSumManyToOne((float*)normalizedPredictionsOut.value().getMemPtr(),
                           (float*)inverseSumOfInputExponentials.getMemPtr(),
                           numClasses,
                           batchSize,
                           true,
                           false,
                           stream);

        // Normalize predictions to sum to 1 per batch item
        launchMultiplyByScalar((float*)normalizedPredictionsOut.value().getMemPtr(),
                               (float*)inverseSumOfInputExponentials.getMemPtr(),
                               (float*)normalizedPredictionsOut.value().getMemPtr(),
                               numClasses,
                               batchSize,
                               stream);
    }

    // normalizedPredictions is featureOutput and loss is errorOutput
    virtual void computeElementwiseLoss(Tensor labels, Tensor normalizedPredictions, Tensor loss, Stream stream) {
        // Cross Entropy Loss
        if (perClassLabels) {
            if (labels.getDescriptor().getDataType() == DataType::UINT8) {
                launchCrossEntropyLoss_perClassLabels((uint8_t*)labels.getMemPtr(),
                                                      (float*)normalizedPredictions.getMemPtr(),
                                                      (float*)lossWorkspace.getMemPtr(),
                                                      (float*)loss.getMemPtr(),
                                                      numClasses,
                                                      batchSize,
                                                      stream);
            } else if (labels.getDescriptor().getDataType() == DataType::FP16) {
                launchCrossEntropyLoss_perClassLabels((half*)labels.getMemPtr(),
                                                      (float*)normalizedPredictions.getMemPtr(),
                                                      (float*)lossWorkspace.getMemPtr(),
                                                      (float*)loss.getMemPtr(),
                                                      numClasses,
                                                      batchSize,
                                                      stream);
            } else if (labels.getDescriptor().getDataType() == DataType::FP32) {
                launchCrossEntropyLoss_perClassLabels((float*)labels.getMemPtr(),
                                                      (float*)normalizedPredictions.getMemPtr(),
                                                      (float*)lossWorkspace.getMemPtr(),
                                                      (float*)loss.getMemPtr(),
                                                      numClasses,
                                                      batchSize,
                                                      stream);
            } else {
                THOR_UNREACHABLE();
            }
        } else if (classIndexLabels) {
            if (labels.getDescriptor().getDataType() == DataType::UINT8) {
                launchCrossEntropyLoss_classIndexLabels((uint8_t*)labels.getMemPtr(),
                                                        (float*)normalizedPredictions.getMemPtr(),
                                                        (float*)lossWorkspace.getMemPtr(),
                                                        (float*)loss.getMemPtr(),
                                                        numClasses,
                                                        batchSize,
                                                        stream);
            } else if (labels.getDescriptor().getDataType() == DataType::UINT16) {
                launchCrossEntropyLoss_classIndexLabels((uint16_t*)labels.getMemPtr(),
                                                        (float*)normalizedPredictions.getMemPtr(),
                                                        (float*)lossWorkspace.getMemPtr(),
                                                        (float*)loss.getMemPtr(),
                                                        numClasses,
                                                        batchSize,
                                                        stream);
            } else if (labels.getDescriptor().getDataType() == DataType::UINT32) {
                launchCrossEntropyLoss_classIndexLabels((uint32_t*)labels.getMemPtr(),
                                                        (float*)normalizedPredictions.getMemPtr(),
                                                        (float*)lossWorkspace.getMemPtr(),
                                                        (float*)loss.getMemPtr(),
                                                        numClasses,
                                                        batchSize,
                                                        stream);
            } else {
                THOR_UNREACHABLE();
            }
        } else {
            THOR_UNREACHABLE();
        }
    }

    virtual void computeLossGradient(Tensor labels, Tensor normalizedPredictions, Tensor lossGradient, Stream stream) {
        if (perClassLabels) {
            if (lossScalingFactor == 1) {
                if (labels.getDescriptor().getDataType() == DataType::UINT8) {
                    launchElementwiseSubtract((float*)normalizedPredictions.getMemPtr(),
                                              (uint8_t*)labels.getMemPtr(),
                                              (half*)lossGradient.getMemPtr(),
                                              lossGradient.getDescriptor().getTotalNumElements(),
                                              stream);
                } else if (labels.getDescriptor().getDataType() == DataType::FP16) {
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
                if (labels.getDescriptor().getDataType() == DataType::UINT8) {
                    launchElementwiseSubtract((float*)normalizedPredictions.getMemPtr(),
                                              (uint8_t*)labels.getMemPtr(),
                                              (float*)errorOutputWorkspace.value().getMemPtr(),
                                              lossGradient.getDescriptor().getTotalNumElements(),
                                              stream);
                } else if (labels.getDescriptor().getDataType() == DataType::FP16) {
                    launchElementwiseSubtract((float*)normalizedPredictions.getMemPtr(),
                                              (half*)labels.getMemPtr(),
                                              (float*)errorOutputWorkspace.value().getMemPtr(),
                                              lossGradient.getDescriptor().getTotalNumElements(),
                                              stream);
                } else if (labels.getDescriptor().getDataType() == DataType::FP32) {
                    launchElementwiseSubtract((float*)normalizedPredictions.getMemPtr(),
                                              (float*)labels.getMemPtr(),
                                              (float*)errorOutputWorkspace.value().getMemPtr(),
                                              lossGradient.getDescriptor().getTotalNumElements(),
                                              stream);
                } else {
                    THOR_UNREACHABLE();
                }
                launchScale((float*)errorOutputWorkspace.value().getMemPtr(),
                            (float)lossScalingFactor,
                            (half*)lossGradient.getMemPtr(),
                            lossGradient.getDescriptor().getTotalNumElements(),
                            stream);
            }
        } else if (classIndexLabels) {
            uint64_t batchSize = featureInput.value().getDescriptor().getDimensions()[0];
            uint64_t numClasses = featureInput.value().getDescriptor().getTotalNumElements() / batchSize;

            if (lossScalingFactor == 1) {
                if (labels.getDescriptor().getDataType() == DataType::UINT8) {
                    launchLossGradient_classIndexLabels((uint8_t*)labels.getMemPtr(),
                                                        (float*)normalizedPredictions.getMemPtr(),
                                                        (half*)lossGradient.getMemPtr(),
                                                        numClasses,
                                                        batchSize,
                                                        stream);
                } else if (labels.getDescriptor().getDataType() == DataType::UINT16) {
                    launchLossGradient_classIndexLabels((uint16_t*)labels.getMemPtr(),
                                                        (float*)normalizedPredictions.getMemPtr(),
                                                        (half*)lossGradient.getMemPtr(),
                                                        numClasses,
                                                        batchSize,
                                                        stream);
                } else if (labels.getDescriptor().getDataType() == DataType::UINT32) {
                    launchLossGradient_classIndexLabels((uint32_t*)labels.getMemPtr(),
                                                        (float*)normalizedPredictions.getMemPtr(),
                                                        (half*)lossGradient.getMemPtr(),
                                                        numClasses,
                                                        batchSize,
                                                        stream);
                } else {
                    THOR_UNREACHABLE();
                }
            } else {
                if (labels.getDescriptor().getDataType() == DataType::UINT8) {
                    launchLossGradient_classIndexLabels((uint8_t*)labels.getMemPtr(),
                                                        (float*)normalizedPredictions.getMemPtr(),
                                                        (float*)errorOutputWorkspace.value().getMemPtr(),
                                                        numClasses,
                                                        batchSize,
                                                        stream);
                } else if (labels.getDescriptor().getDataType() == DataType::UINT16) {
                    launchLossGradient_classIndexLabels((uint16_t*)labels.getMemPtr(),
                                                        (float*)normalizedPredictions.getMemPtr(),
                                                        (float*)errorOutputWorkspace.value().getMemPtr(),
                                                        numClasses,
                                                        batchSize,
                                                        stream);
                } else if (labels.getDescriptor().getDataType() == DataType::UINT32) {
                    launchLossGradient_classIndexLabels((uint32_t*)labels.getMemPtr(),
                                                        (float*)normalizedPredictions.getMemPtr(),
                                                        (float*)errorOutputWorkspace.value().getMemPtr(),
                                                        numClasses,
                                                        batchSize,
                                                        stream);
                } else {
                    THOR_UNREACHABLE();
                }
                launchScale((float*)errorOutputWorkspace.value().getMemPtr(),
                            (float)lossScalingFactor,
                            (half*)lossGradient.getMemPtr(),
                            lossGradient.getDescriptor().getTotalNumElements(),
                            stream);
            }
        } else {
            THOR_UNREACHABLE();
        }

        // FIXME: At the API layer I should offer 3 options: loss per batch, loss per batch item, loss per batch item per class.
        //        Currently this layer outputs loss per batch item, it should output loss per batch item per class and a subsequent layer
        //        should reduce as desired.
    }

    virtual void backProp(std::optional<Tensor> labels, std::optional<Tensor> normalizedPredictions, std::optional<Tensor> lossGradient, Stream stream) {
        THOR_THROW_IF_FALSE(lossGradient.has_value());
    }

   protected:
    bool perClassLabels;
    bool classIndexLabels;

   private:
    unsigned int batchSize;
    unsigned int numClasses;

    Tensor inverseSumOfInputExponentials;
    Tensor lossWorkspace;

    std::optional<Tensor> errorOutputWorkspace;
};

}  // namespace ThorImplementation
*/