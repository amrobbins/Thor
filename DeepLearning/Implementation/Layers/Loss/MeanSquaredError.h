#pragma once

#include "DeepLearning/Implementation/Layers/Loss.h"
#include "Utilities/TensorOperations/Loss/CrossEntropyLoss.h"
#include "Utilities/TensorOperations/Loss/MeanSquaredError.h"
#include "Utilities/TensorOperations/Misc/BatchReduce.h"

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

class MeanSquaredError : public Loss {
   public:
    virtual ~MeanSquaredError() {
        if (batchReduce)
            delete batchReduce;
        batchReduce = nullptr;
    };

    MeanSquaredError() : Loss() { batchReduce = nullptr; }

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
        labelsDataType = labelsInput.get().getDescriptor().getDataType();
        assert(featureInputDimensions == labelDimensions);

        assert(featureInput.get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        assert(featureInput.get().getDescriptor().getDimensions().size() == 2);

        batchSize = featureInput.get().getDescriptor().getDimensions()[0];

        batchReduce = new BatchReduce(batchSize,
                                      batchSize,
                                      featureInput.get().getDescriptor().getDimensions()[1],
                                      featureInput.get().getDescriptor().getDataType(),
                                      featureOutput.get().getDescriptor().getDataType(),
                                      featureOutput.get().getDescriptor().getDataType(),
                                      stream);

        workspace = ThorImplementation::Tensor(ThorImplementation::TensorPlacement(ThorImplementation::TensorPlacement::MemDevices::GPU, 0),
                                               ThorImplementation::TensorDescriptor(ThorImplementation::TensorDescriptor::DataType::FP16,
                                                                                    featureInput.get().getDescriptor().getDimensions()));
    }

    virtual void cleanup() {}

    // Loss and gradient computation are all handled in infer. Gradient is not computed when in inference only mode.
    virtual void computeElementwiseLoss(Tensor labels, Tensor predictions, Tensor loss, Stream stream) {}
    virtual void computeLossGradient(Tensor labels, Tensor normalizedPredictions, Tensor lossGradient, Stream stream) {}

    virtual void infer(Optional<Tensor> predictions, Optional<Tensor> loss, Stream stream) {
        assert(predictions.isPresent());
        assert(loss.isPresent());
        assert(labelsInput.isPresent());
        assert(workspace.getDescriptor().getDimensions() == loss.get().getDescriptor().getDimensions());
        if (!isInferenceOnly())
            assert(errorOutput.isPresent());

        ScopedGpu scopedGpu(predictions.get().getPlacement().getDeviceNum());

        assert(compiled);
        assert(predictions.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16);

        if (predictions.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
            if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
                if (loss.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
                    launchMeanSquaredError<half, half, half>(labelsInput.get().getMemPtr(),
                                                             predictions.get().getMemPtr(),
                                                             loss.get().getMemPtr(),
                                                             workspace.getMemPtr(),
                                                             isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                             loss.get().getDescriptor().getDimensions()[1],
                                                             batchSize,
                                                             stream,
                                                             batchReduce,
                                                             !isInferenceOnly());
                } else {
                    assert(loss.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32);
                    launchMeanSquaredError<half, half, float>(labelsInput.get().getMemPtr(),
                                                              predictions.get().getMemPtr(),
                                                              loss.get().getMemPtr(),
                                                              workspace.getMemPtr(),
                                                              isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                              loss.get().getDescriptor().getDimensions()[1],
                                                              batchSize,
                                                              stream,
                                                              batchReduce,
                                                              !isInferenceOnly());
                }
            } else {
                assert(labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32);
                if (loss.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
                    launchMeanSquaredError<float, half, half>(labelsInput.get().getMemPtr(),
                                                              predictions.get().getMemPtr(),
                                                              loss.get().getMemPtr(),
                                                              workspace.getMemPtr(),
                                                              isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                              loss.get().getDescriptor().getDimensions()[1],
                                                              batchSize,
                                                              stream,
                                                              batchReduce,
                                                              !isInferenceOnly());
                } else {
                    assert(loss.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32);
                    launchMeanSquaredError<float, half, float>(labelsInput.get().getMemPtr(),
                                                               predictions.get().getMemPtr(),
                                                               loss.get().getMemPtr(),
                                                               workspace.getMemPtr(),
                                                               isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                               loss.get().getDescriptor().getDimensions()[1],
                                                               batchSize,
                                                               stream,
                                                               batchReduce,
                                                               !isInferenceOnly());
                }
            }
        } else {
            assert(predictions.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32);
            if (labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
                if (loss.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
                    launchMeanSquaredError<half, half, half>(labelsInput.get().getMemPtr(),
                                                             predictions.get().getMemPtr(),
                                                             loss.get().getMemPtr(),
                                                             workspace.getMemPtr(),
                                                             isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                             loss.get().getDescriptor().getDimensions()[1],
                                                             batchSize,
                                                             stream,
                                                             batchReduce,
                                                             !isInferenceOnly());
                } else {
                    assert(loss.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32);
                    launchMeanSquaredError<half, half, float>(labelsInput.get().getMemPtr(),
                                                              predictions.get().getMemPtr(),
                                                              loss.get().getMemPtr(),
                                                              workspace.getMemPtr(),
                                                              isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                              loss.get().getDescriptor().getDimensions()[1],
                                                              batchSize,
                                                              stream,
                                                              batchReduce,
                                                              !isInferenceOnly());
                }
            } else {
                assert(labelsInput.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32);
                if (loss.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
                    launchMeanSquaredError<float, half, half>(labelsInput.get().getMemPtr(),
                                                              predictions.get().getMemPtr(),
                                                              loss.get().getMemPtr(),
                                                              workspace.getMemPtr(),
                                                              isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                              loss.get().getDescriptor().getDimensions()[1],
                                                              batchSize,
                                                              stream,
                                                              batchReduce,
                                                              !isInferenceOnly());
                } else {
                    assert(loss.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32);
                    launchMeanSquaredError<float, half, float>(labelsInput.get().getMemPtr(),
                                                               predictions.get().getMemPtr(),
                                                               loss.get().getMemPtr(),
                                                               workspace.getMemPtr(),
                                                               isInferenceOnly() ? nullptr : (half *)errorOutput.get().getMemPtr(),
                                                               loss.get().getDescriptor().getDimensions()[1],
                                                               batchSize,
                                                               stream,
                                                               batchReduce,
                                                               !isInferenceOnly());
                }
            }
        }
    }

    /*    virtual void backProp(Optional<Tensor> labels, Optional<Tensor> normalizedPredictions, Optional<Tensor> lossGradient, Stream
       stream) { assert(lossGradient.isPresent());
        }
        */

   private:
    TensorDescriptor::DataType labelsDataType;
    unsigned int batchSize;
    BatchReduce *batchReduce;
    Tensor workspace;
};

}  // namespace ThorImplementation
