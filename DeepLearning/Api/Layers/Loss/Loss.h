#pragma once

#include "DeepLearning/Api/Layers/Loss/LossBase.h"
#include "DeepLearning/Api/Tensor.h"

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

    Tensor getLossTensor() { return loss->getLossTensor(); }

   private:
    shared_ptr<LossBase> loss;

    LossBase *getRawLoss();

    friend class Network;
};

}  // namespace Thor
