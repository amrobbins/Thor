#pragma once

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Implementation/Layers/Loss.h"

#include <assert.h>
#include <atomic>
#include <utility>

using std::atomic;

namespace Thor {

class Loss : public Layer {
   public:
    enum class InputLossType { NUMERICAL_LOSS = 5, CATEGORICAL_LOSS };
    enum class OutputLossType { BATCH_LOSS = 11, CLASSWISE_LOSS };

    Loss() { numInputConnectionsMade = 0; }
    virtual ~Loss() {}

    virtual bool mustConnectAllInputsToDriveOutput() { return true; }
    virtual void informThatInputConnectionMade(Tensor inputTensor) {
        numInputConnectionsMade += 1;
        // Only one type of loss is supported at a time.
        assert(numInputConnectionsMade < 3);
    }

    virtual Tensor getPredictions() const { return predictionsTensor; }
    virtual Tensor getLabels() const { return labelsTensor; }
    virtual Tensor getLoss() const { return lossTensor; }

    // getPredictions() and getLoss() are synonyms for getFeatureInput() and getFeatureOutput() in losses:
    virtual Optional<Tensor> getFeatureInput() { return getPredictions(); }
    virtual Optional<Tensor> getFeatureOutput() { return getLoss(); }

    virtual int getConnectionType(Tensor connectingTensor) const {
        if (connectingTensor == getLabels())
            return (int)ThorImplementation::Loss::ConnectionType::LABELS;
        else if (connectingTensor == getPredictions())
            return (int)ThorImplementation::Loss::ConnectionType::PREDICTIONS;
        else if (connectingTensor == getLoss())
            return (int)lossType;
        assert(false);
    }

    virtual vector<Tensor> getOutputsFromInput(Tensor inputTensor) {
        if (numInputConnectionsMade == 2)
            return {lossTensor, predictionsTensor};
        else
            return vector<Tensor>();
    }

    virtual vector<Tensor> getAllOutputTensors() const { return {getPredictions(), getLoss()}; }

   protected:
    Tensor labelsTensor;
    Tensor predictionsTensor;
    Tensor lossTensor;

    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, TensorPlacement tensorPlacement) const {
        uint32_t fixedMem = 4;  // loss scaling factor, FP32

        // Labels
        uint64_t labelsBytes = labelsTensor.getTotalSizeInBytes();

        // Error Output
        uint64_t errorOutputBytes = featureInput.get().getTotalSizeInBytes();  // FIXME this is not present for inference only

        // Predictions
        uint64_t predictionsOutputBytes = predictionsTensor.getTotalSizeInBytes();

        // Loss
        uint64_t lossBytes = lossTensor.getTotalSizeInBytes();

        return fixedMem + batchSize * (predictionsOutputBytes + labelsBytes + errorOutputBytes + lossBytes);
    }

    // Loss type must be set by deriving class
    ThorImplementation::Loss::LossType lossType;

   private:
    uint32_t numInputConnectionsMade = 0;
};

}  // namespace Thor
