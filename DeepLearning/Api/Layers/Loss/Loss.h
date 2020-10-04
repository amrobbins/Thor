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

   private:
    Tensor getFeatureOutput() { assert(false); }
};

}  // namespace Thor
