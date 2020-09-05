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
        assert(labelsTensor.isPresent());
        assert(lossTensor.isPresent());
        assert(labelsTensor.get().isInitialized());
        assert(lossTensor.get().isInitialized());
        assert(featureInput.get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        assert(labelsTensor.get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        assert(labelsTensor.get().getPlacement().getDeviceNum() == featureInput.get().getPlacement().getDeviceNum());
        assert(labelsTensor.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32 ||
               labelsTensor.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16);
        assert(featureInput.get().getDescriptor().getDimensions() == labelsTensor.get().getDescriptor().getDimensions());
        assert(lossTensor.get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        assert(lossTensor.get().getPlacement().getDeviceNum() == featureInput.get().getPlacement().getDeviceNum());
        assert(lossTensor.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32);

        ScopedGpu scopedGpu(featureInput.get().getPlacement().getDeviceNum());

        softmaxWorkspace = Tensor(featureInput.get().getPlacement(),
                                  TensorDescriptor(TensorDescriptor::DataType::FP32, featureInput.get().getDescriptor().getDimensions()));
        lossWorkspace = Tensor(featureInput.get().getPlacement(),
                               TensorDescriptor(TensorDescriptor::DataType::FP32, featureInput.get().getDescriptor().getDimensions()));

        assert(featureInput.get().getDescriptor().getDimensions().size() >= 2);
        batchSize = featureInput.get().getDescriptor().getDimensions()[0];
        elementsPerBatch = featureInput.get().getDescriptor().getTotalNumElements() / batchSize;

        vector<unsigned long> onePerBatchDimensions;
        onePerBatchDimensions.push_back(batchSize);
        inverseSumOfInputExponentials =
            Tensor(featureInput.get().getPlacement(), TensorDescriptor(TensorDescriptor::DataType::FP32, onePerBatchDimensions));
    }

    virtual void cleanup() {}

    virtual void infer(Optional<Tensor> inputTensor, Optional<Tensor> outputTensor, Stream stream) {
        assert(inputTensor.isPresent());
        assert(inputTensor.get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        if (inputTensor.get() != featureInput.get())
            return;
        ScopedGpu scopedGpu(inputTensor.get().getPlacement().getDeviceNum());

        // FIXME: what cudnn functions can I use here? cudnnReduceTensor(), cudnnScaleTensor(), cudnnOpTensor() ... ?

        // Softmax
        if (inputTensor.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
            launchExponentiation((half*)inputTensor.get().getMemPtr(),
                                 (float*)softmaxWorkspace.getMemPtr(),
                                 inputTensor.get().getDescriptor().getTotalNumElements(),
                                 stream);
        } else {
            launchExponentiation((float*)inputTensor.get().getMemPtr(),
                                 (float*)softmaxWorkspace.getMemPtr(),
                                 inputTensor.get().getDescriptor().getTotalNumElements(),
                                 stream);
        }
        launchSumManyToOne((float*)softmaxWorkspace.getMemPtr(),
                           (float*)inverseSumOfInputExponentials.getMemPtr(),
                           elementsPerBatch,
                           batchSize,
                           true,
                           false,
                           stream);
        launchMultiplyByScalar((float*)softmaxWorkspace.getMemPtr(),
                               (float*)inverseSumOfInputExponentials.getMemPtr(),
                               (float*)softmaxWorkspace.getMemPtr(),
                               elementsPerBatch,
                               batchSize,
                               stream);

        // Cross Entropy Loss
        if (labelsTensor.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
            launchCrossEntropyLoss((half*)labelsTensor.get().getMemPtr(),
                                   (float*)softmaxWorkspace.getMemPtr(),
                                   (float*)lossWorkspace.getMemPtr(),
                                   (float*)lossTensor.get().getMemPtr(),
                                   elementsPerBatch,
                                   batchSize,
                                   stream);
        } else {
            launchCrossEntropyLoss((float*)labelsTensor.get().getMemPtr(),
                                   (float*)softmaxWorkspace.getMemPtr(),
                                   (float*)lossWorkspace.getMemPtr(),
                                   (float*)lossTensor.get().getMemPtr(),
                                   elementsPerBatch,
                                   batchSize,
                                   stream);
        }
    }

    virtual void backProp(Optional<Tensor> dataIn, Optional<Tensor> errorIn, Optional<Tensor> errorOut, Stream stream) {
        assert(errorOut.isPresent());

        if (lossScalingFactor == 1.0f) {
            if (labelsTensor.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
                launchElementwiseSubtract((float*)softmaxWorkspace.getMemPtr(),
                                          (half*)labelsTensor.get().getMemPtr(),
                                          (half*)errorOut.get().getMemPtr(),
                                          errorOut.get().getDescriptor().getTotalNumElements(),
                                          stream);
            } else {
                launchElementwiseSubtract((float*)softmaxWorkspace.getMemPtr(),
                                          (float*)labelsTensor.get().getMemPtr(),
                                          (half*)errorOut.get().getMemPtr(),
                                          errorOut.get().getDescriptor().getTotalNumElements(),
                                          stream);
            }
        } else {
            if (labelsTensor.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
                launchElementwiseSubtract((float*)softmaxWorkspace.getMemPtr(),
                                          (half*)labelsTensor.get().getMemPtr(),
                                          (float*)softmaxWorkspace.getMemPtr(),
                                          errorOut.get().getDescriptor().getTotalNumElements(),
                                          stream);
            } else {
                launchElementwiseSubtract((float*)softmaxWorkspace.getMemPtr(),
                                          (float*)labelsTensor.get().getMemPtr(),
                                          (float*)softmaxWorkspace.getMemPtr(),
                                          errorOut.get().getDescriptor().getTotalNumElements(),
                                          stream);
            }
            launchMultiplyByScalar((float*)softmaxWorkspace.getMemPtr(),
                                   (float*)lossScalingFactorTensor.getMemPtr(),
                                   (half*)errorOut.get().getMemPtr(),
                                   errorOut.get().getDescriptor().getTotalNumElements(),
                                   1,
                                   stream);
        }
    }

   private:
    unsigned int batchSize;
    unsigned int elementsPerBatch;

    Tensor inverseSumOfInputExponentials;
    Tensor softmaxWorkspace;
    Tensor lossWorkspace;
};
