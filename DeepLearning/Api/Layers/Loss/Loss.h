#pragma once

#include "DeepLearning/Api/Layers/Loss/LossBase.h"

#include <assert.h>
#include <memory>

namespace Thor {

using std::shared_ptr;

class Loss {
   public:
    Loss();
    Loss(LossBase *lossBase) { loss = shared_ptr<LossBase>(lossBase); }

    virtual ~Loss() {}

   private:
    shared_ptr<LossBase> loss;
};

}  // namespace Thor
