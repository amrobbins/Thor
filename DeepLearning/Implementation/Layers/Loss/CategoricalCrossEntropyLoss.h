#pragma once

#include "DeepLearning/Implementation/Layers/Loss.h"

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

    CategoricalCrossEntropyLoss(Optional<float> lossScalingFactor) : Loss(lossScalingFactor.isPresent() ? lossScalingFactor.get() : 1.0f) {}

    virtual void compile() {
        if (!isInferenceOnly()) {
            assert(labelsInput.isPresent());
            assert(errorOutput.isPresent());
            assert(errorOutput.get().isInitialized());
            assert(errorOutput.get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
            assert(errorOutput.get().getPlacement().getDeviceNum() == featureInput.get().getPlacement().getDeviceNum());
            assert(errorOutput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16);
        }
        if (labelsInput.isPresent()) {
            assert(labelsInput.get().isInitialized());
            assert(labelsInput.get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
            assert(labelsInput.get().getPlacement().getDeviceNum() == featureInput.get().getPlacement().getDeviceNum());
            assert(labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32 ||
                   labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16);
        }
        assert(featureInput.isPresent());
        assert(featureInput.get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        assert(featureInput.get().getDescriptor().getDimensions() == labelsInput.get().getDescriptor().getDimensions());

        ScopedGpu scopedGpu(featureInput.get().getPlacement().getDeviceNum());

        lossWorkspace = Tensor(featureInput.get().getPlacement(),
                               TensorDescriptor(TensorDescriptor::DataType::FP32, featureInput.get().getDescriptor().getDimensions()));

        assert(featureInput.get().getDescriptor().getDimensions().size() >= 2);
        batchSize = featureInput.get().getDescriptor().getDimensions()[0];
        elementsPerBatch = featureInput.get().getDescriptor().getTotalNumElements() / batchSize;

        inverseSumOfInputExponentials =
            Tensor(featureInput.get().getPlacement(), TensorDescriptor(TensorDescriptor::DataType::FP32, {batchSize}));

        if (lossScalingFactor == 1.0) {
        } else if (!isInferenceOnly()) {
            errorOutputWorkspace = errorOutput.get().clone(TensorDescriptor::DataType::FP32);
        }
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
                           elementsPerBatch,
                           batchSize,
                           true,
                           false,
                           stream);

        // Normalize predictions to sum to 1 per batch item
        launchMultiplyByScalar((float*)normalizedPredictionsOut.get().getMemPtr(),
                               (float*)inverseSumOfInputExponentials.getMemPtr(),
                               (float*)normalizedPredictionsOut.get().getMemPtr(),
                               elementsPerBatch,
                               batchSize,
                               stream);
    }

    // normalizedPredictions is featureOutput and loss is errorOutput
    virtual void computeLoss(Tensor labels, Tensor normalizedPredictions, Tensor loss, Stream stream) {
        // Cross Entropy Loss
        if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
            launchCrossEntropyLoss((half*)labels.getMemPtr(),
                                   (float*)normalizedPredictions.getMemPtr(),
                                   (float*)lossWorkspace.getMemPtr(),
                                   (float*)loss.getMemPtr(),
                                   elementsPerBatch,
                                   batchSize,
                                   stream);
        } else if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::FP32) {
            launchCrossEntropyLoss((float*)labels.getMemPtr(),
                                   (float*)normalizedPredictions.getMemPtr(),
                                   (float*)lossWorkspace.getMemPtr(),
                                   (float*)loss.getMemPtr(),
                                   elementsPerBatch,
                                   batchSize,
                                   stream);
        } else {
            assert(false);
        }

        // FIXME: At the API layer I should offer 3 options: loss per batch, loss per batch item, loss per batch item per class.
        //        Currently this layer outputs loss per batch item, it should output loss per batch item per class and a subsequent layer
        //        should reduce as desired.
    }

    virtual void computeLossGradient(Tensor labels, Tensor normalizedPredictions, Tensor lossGradient, Stream stream) {
        if (lossScalingFactor == 1.0f) {
            if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
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
            if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
                launchElementwiseSubtract((float*)normalizedPredictions.getMemPtr(),
                                          (half*)labels.getMemPtr(),
                                          (float*)errorOutputWorkspace.get().getMemPtr(),
                                          lossGradient.getDescriptor().getTotalNumElements(),
                                          stream);
            } else {
                launchElementwiseSubtract((float*)normalizedPredictions.getMemPtr(),
                                          (float*)labels.getMemPtr(),
                                          (float*)errorOutputWorkspace.get().getMemPtr(),
                                          lossGradient.getDescriptor().getTotalNumElements(),
                                          stream);
            }
            launchMultiplyByScalar((float*)errorOutputWorkspace.get().getMemPtr(),
                                   (float*)lossScalingFactorTensor.getMemPtr(),
                                   (half*)lossGradient.getMemPtr(),
                                   lossGradient.getDescriptor().getTotalNumElements(),
                                   1,
                                   stream);
        }
    }

    virtual void backProp(Optional<Tensor> labels, Optional<Tensor> normalizedPredictions, Optional<Tensor> lossGradient, Stream stream) {
        assert(lossGradient.isPresent());
    }

   private:
    unsigned int batchSize;
    unsigned int elementsPerBatch;

    Tensor inverseSumOfInputExponentials;
    Tensor lossWorkspace;

    Optional<Tensor> errorOutputWorkspace;
};
