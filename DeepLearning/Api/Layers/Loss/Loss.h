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
    Loss() {}
    virtual ~Loss() {}

    // virtual Optional<Tensor> getFeatureInput() const { return Layer::getFeatureInput(); }
    virtual Tensor getPredictions() const { return predictionsTensor; }
    virtual Tensor getLabels() const { return labelsTensor; }

    virtual Tensor getLoss() const { return lossTensor; }

    virtual int getConnectionType(Tensor connectingTensor) const {
        printf("called loss getConnectionType\n");
        if (connectingTensor == getFeatureInput())
            return (int)ThorImplementation::Loss::ConnectionType::FORWARD_BACKWARD;
        else if (connectingTensor == getLabels())
            return (int)ThorImplementation::Loss::ConnectionType::LABELS;
        else if (connectingTensor == getPredictions())
            return (int)ThorImplementation::Loss::ConnectionType::PREDICTIONS;
        else if (connectingTensor == getLoss())
            return (int)ThorImplementation::Loss::ConnectionType::LOSS;
        assert(false);
    }

    virtual vector<Tensor> getAllOutputTensors() const { return {getPredictions(), getLoss()}; }

   protected:
    Tensor labelsTensor;
    Tensor predictionsTensor;
    Tensor lossTensor;

    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize) const {
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

   private:
    Tensor getFeatureOutput() { assert(false); }
};

}  // namespace Thor
