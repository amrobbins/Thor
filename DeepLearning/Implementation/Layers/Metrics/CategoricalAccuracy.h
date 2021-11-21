#include "DeepLearning/Implementation/Layers/Layer.h"

namespace ThorImplementation {

/**
 * Returns the proportion of the predictions where the class with the highest prediction probability is the true class.
 */

class CategoricalAccuracy : public Metric {
   public:
    virtual ~CategoricalAccuracy() {}
    CategoricalAccuracy() {}

    virtual void computeMetric(Tensor labels, Tensor predictions, Tensor metric, Stream stream) {}
};

/*
class CategoricalAccuracy : public Metric {
   public:
    virtual ~CategoricalAccuracy() {}

    CategoricalAccuracy() {}

    virtual Optional<Tensor> createFeatureOutputTensor() {
        assert(featureInput.isPresent());
        return Tensor(featureInput.get().getPlacement(), TensorDescriptor(TensorDescriptor::DataType::FP32, {1}));
    }

    virtual void compile() {
        assert(errorOutput.isEmpty());
        assert(labelsInput.isPresent());
        assert(labelsInput.get().isInitialized());
        assert(labelsInput.get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        assert(labelsInput.get().getPlacement().getDeviceNum() == featureInput.get().getPlacement().getDeviceNum());

        vector<uint64_t> featureInputDimensions = featureInput.get().getDescriptor().getDimensions();
        vector<uint64_t> labelDimensions = labelsInput.get().getDescriptor().getDimensions();
        TensorDescriptor::DataType labelsDataType = labelsInput.get().getDescriptor().getDataType();
        perClassLabels = featureInputDimensions == labelDimensions &&
                         (labelsDataType == TensorDescriptor::DataType::UINT8 || labelsDataType == TensorDescriptor::DataType::FP16 ||
                          labelsDataType == TensorDescriptor::DataType::FP32);
        classIndexLabels =
            labelDimensions.size() == 2 && featureInputDimensions[0] == labelDimensions[0] && labelDimensions[1] == 1 &&
            (labelsDataType == TensorDescriptor::DataType::UINT8 || labelsDataType == TensorDescriptor::DataType::UINT16 ||
             labelsDataType == TensorDescriptor::DataType::UINT32);
        assert(perClassLabels ^ classIndexLabels);

        assert(featureInput.isPresent());
        assert(featureInput.get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        assert(featureInput.get().getDescriptor().getDimensions().size() == 2);

        batchSize = featureInputDimensions[0];
        numClasses = featureInputDimensions[1];

        workspace = Tensor(featureInput.get().getPlacement(), TensorDescriptor(TensorDescriptor::DataType::FP32, {batchSize}));

        // When there are two classes and the label is a single 1 or 0, binary accuracy can be used, instead of categorical accuracy
        assert(numClasses >= 2);
    }

    virtual void cleanup() {}

    virtual void infer(Optional<Tensor> rawPredictionsIn, Optional<Tensor> normalizedPredictionsOut, Stream stream) {
        assert(rawPredictionsIn.isPresent());
        assert(normalizedPredictionsOut.isPresent());
        assert(rawPredictionsIn.get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        if (rawPredictionsIn.get() != featureInput.get())
            return;
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

        // FIXME: At the API layer I should offer 3 options: loss per batch, loss per batch item, loss per batch item per class.
        //        Currently this layer outputs loss per batch item, it should output loss per batch item per class and a subsequent layer
        //        should reduce as desired.
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

    Tensor workspace;
};
*/
}  // namespace ThorImplementation