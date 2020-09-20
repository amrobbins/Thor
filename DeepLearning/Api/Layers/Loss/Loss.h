#pragma once

#include "DeepLearning/Api/Layers/Loss/LossBase.h"

#include <assert.h>
#include <memory>

namespace Thor {

class Network;

using std::shared_ptr;

class Loss {
   public:
    Loss();
    Loss(LossBase *lossBase) { loss = shared_ptr<LossBase>(lossBase); }

    virtual ~Loss() {}

    uint32_t getId() const { return loss->getId(); }

    Tensor getFeatureInput() const { return loss->getFeatureInput(); }
    Tensor getPredictions() const { return loss->getFeatureOutput(); }
    Tensor getLossTensor() const { return loss->getLossTensor(); }

   protected:
    shared_ptr<LossBase> loss;

    LossBase *getRawLoss();

    friend class Network;
};

}  // namespace Thor
