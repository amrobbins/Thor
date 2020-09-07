#pragma once

#include "DeepLearning/Implementation/Layers/Loss.h"

/**
 * This is equivalent to a softmax activation layer followed by a categorical cross entropy loss.
 *
 * The input predicted values to the loss layer will sum to 1.0 since they are put through a softmax activation first.
 * Those values are clamped to a minimum value of 10e-15, to avoid log(0.0f).
 *
 * https://gombru.github.io/2018/05/23/cross_entropy_loss/
 */

class CategoricalCrossEntropyLoss : public Loss {
   public:
    virtual ~CategoricalCrossEntropyLoss(){};

    CategoricalCrossEntropyLoss(float lossScalingFactor = 1.0f) : Loss(lossScalingFactor) {}

    virtual void compile() {
        assert(featureInput.isPresent());
        assert(labelsInput.isPresent());
        assert(errorOutput.isPresent());
        assert(labelsInput.get().isInitialized());
        assert(errorOutput.get().isInitialized());
        assert(featureInput.get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        assert(labelsInput.get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        assert(labelsInput.get().getPlacement().getDeviceNum() == featureInput.get().getPlacement().getDeviceNum());
        assert(labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32 ||
               labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16);
        assert(featureInput.get().getDescriptor().getDimensions() == labelsInput.get().getDescriptor().getDimensions());
        assert(errorOutput.get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        assert(errorOutput.get().getPlacement().getDeviceNum() == featureInput.get().getPlacement().getDeviceNum());
        assert(errorOutput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16);

        ScopedGpu scopedGpu(featureInput.get().getPlacement().getDeviceNum());

        lossWorkspace = Tensor(featureInput.get().getPlacement(),
                               TensorDescriptor(TensorDescriptor::DataType::FP32, featureInput.get().getDescriptor().getDimensions()));

        assert(featureInput.get().getDescriptor().getDimensions().size() >= 2);
        batchSize = featureInput.get().getDescriptor().getDimensions()[0];
        elementsPerBatch = featureInput.get().getDescriptor().getTotalNumElements() / batchSize;

        inverseSumOfInputExponentials =
            Tensor(featureInput.get().getPlacement(), TensorDescriptor(TensorDescriptor::DataType::FP32, {batchSize}));
    }

    virtual void cleanup() {}

    virtual void infer(Optional<Tensor> rawPredictions, Optional<Tensor> predictions, Stream stream) {
        assert(rawPredictions.isPresent());
        assert(predictions.isPresent());
        assert(rawPredictions.get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        if (rawPredictions.get() != featureInput.get())
            return;
        ScopedGpu scopedGpu(rawPredictions.get().getPlacement().getDeviceNum());

        // Softmax
        if (rawPredictions.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
            launchExponentiation((half*)rawPredictions.get().getMemPtr(),
                                 (float*)predictions.get().getMemPtr(),
                                 rawPredictions.get().getDescriptor().getTotalNumElements(),
                                 stream);
        } else {
            launchExponentiation((float*)rawPredictions.get().getMemPtr(),
                                 (float*)predictions.get().getMemPtr(),
                                 rawPredictions.get().getDescriptor().getTotalNumElements(),
                                 stream);
        }
        // Compute the norm
        launchSumManyToOne((float*)predictions.get().getMemPtr(),
                           (float*)inverseSumOfInputExponentials.getMemPtr(),
                           elementsPerBatch,
                           batchSize,
                           true,
                           false,
                           stream);
        // Normalize predictions
        launchMultiplyByScalar((float*)predictions.get().getMemPtr(),
                               (float*)inverseSumOfInputExponentials.getMemPtr(),
                               (float*)predictions.get().getMemPtr(),
                               elementsPerBatch,
                               batchSize,
                               stream);

        // FIXME: inverse sum of exponentials is the loss per batch item, I want to output that from this layer.
        // And when a scalar loss is desired then connect a SumAll layer between this and the network output.
        // At the API layer I should offer 3 options: loss per batch, loss per batch item, loss per batch item per class.
    }

    // predictions is featureOutput and loss is errorOutput
    virtual void computeLoss(Tensor labels, Tensor predictions, Tensor loss, Stream dataStream, Stream stream) {
        // Cross Entropy Loss
        if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
            launchCrossEntropyLoss((half*)labels.getMemPtr(),
                                   (float*)predictions.getMemPtr(),
                                   (float*)lossWorkspace.getMemPtr(),
                                   (half*)loss.getMemPtr(),
                                   elementsPerBatch,
                                   batchSize,
                                   stream);
        } else if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::FP32) {
            launchCrossEntropyLoss((float*)labels.getMemPtr(),
                                   (float*)predictions.getMemPtr(),
                                   (float*)lossWorkspace.getMemPtr(),
                                   (half*)loss.getMemPtr(),
                                   elementsPerBatch,
                                   batchSize,
                                   stream);
        } else {
            assert(false);
        }

        if (lossScalingFactor == 1.0f) {
            if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
                launchElementwiseSubtract((float*)predictions.getMemPtr(),
                                          (half*)labels.getMemPtr(),
                                          (half*)loss.getMemPtr(),
                                          loss.getDescriptor().getTotalNumElements(),
                                          stream);
            } else {
                launchElementwiseSubtract((float*)predictions.getMemPtr(),
                                          (float*)labels.getMemPtr(),
                                          (half*)loss.getMemPtr(),
                                          loss.getDescriptor().getTotalNumElements(),
                                          stream);
            }
        } else {
            if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
                launchElementwiseSubtract((float*)predictions.getMemPtr(),
                                          (half*)labels.getMemPtr(),
                                          (float*)predictions.getMemPtr(),
                                          loss.getDescriptor().getTotalNumElements(),
                                          stream);
            } else {
                launchElementwiseSubtract((float*)predictions.getMemPtr(),
                                          (float*)labels.getMemPtr(),
                                          (float*)predictions.getMemPtr(),
                                          loss.getDescriptor().getTotalNumElements(),
                                          stream);
            }
            launchMultiplyByScalar((float*)predictions.getMemPtr(),
                                   (float*)lossScalingFactorTensor.getMemPtr(),
                                   (half*)loss.getMemPtr(),
                                   loss.getDescriptor().getTotalNumElements(),
                                   1,
                                   stream);
        }
    }

    virtual void backProp(Optional<Tensor> labels, Optional<Tensor> predictions, Optional<Tensor> loss, Stream stream) {
        assert(loss.isPresent());
    }

   private:
    unsigned int batchSize;
    unsigned int elementsPerBatch;

    Tensor inverseSumOfInputExponentials;
    Tensor lossWorkspace;
};
