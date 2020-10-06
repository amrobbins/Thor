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

    virtual int getConnectionType(Tensor connectingTensor) {
        assert(connectingTensor == getFeatureInput() || connectingTensor == getLabels() || connectingTensor == getPredictions() ||
               connectingTensor == getLoss());
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

   protected:
    Tensor labelsTensor;
    Tensor predictionsTensor;
    Tensor lossTensor;

    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize) const {
        uint32_t fixedMem = 4;  // loss scaling factor, FP32

        // Predictions
        uint64_t predictionsOutputBytes = featureOutput.get().getTotalSizeInBytes();

        // Labels
        uint64_t labelsBytes = featureInput.get().getTotalNumElements() * 4;  // FP32, 1 per class (soft labels supported)

        // Error Output
        uint64_t errorOutputBytes = featureInput.get().getTotalSizeInBytes();  // FIXME this is not present for inference only

        // Loss
        uint64_t lossBytes = 4;  // FP32 per batch item

        return fixedMem + batchSize * (predictionsOutputBytes + labelsBytes + errorOutputBytes + lossBytes);
    }

   private:
    Tensor getFeatureOutput() { assert(false); }
};

}  // namespace Thor
