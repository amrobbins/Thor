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

class MeanSquaredError : public Loss {
   public:
    virtual ~MeanSquaredError(){};

    MeanSquaredError() : Loss() {}

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

        // FIXME: create BatchReduce
    }

    virtual void cleanup() {}

    virtual void infer(Optional<Tensor> rawPredictionsIn, Optional<Tensor> normalizedPredictionsOut, Stream stream) {
        assert(rawPredictionsIn.isPresent());
        assert(normalizedPredictionsOut.isPresent());
        assert(featureInput.isPresent());
        assert(rawPredictionsIn.get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        assert(rawPredictionsIn.get() == featureInput.get());
        ScopedGpu scopedGpu(rawPredictionsIn.get().getPlacement().getDeviceNum());
    }

    // normalizedPredictions is featureOutput and loss is errorOutput
    virtual void computeElementwiseLoss(Tensor labels, Tensor normalizedPredictions, Tensor loss, Stream stream) {}

    virtual void computeLossGradient(Tensor labels, Tensor normalizedPredictions, Tensor lossGradient, Stream stream) {}

    virtual void backProp(Optional<Tensor> labels, Optional<Tensor> normalizedPredictions, Optional<Tensor> lossGradient, Stream stream) {
        assert(lossGradient.isPresent());
    }

   private:
    TensorDescriptor::DataType labelsDataType;
    unsigned int batchSize;
};

}  // namespace ThorImplementation
