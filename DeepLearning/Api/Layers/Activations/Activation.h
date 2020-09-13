#pragma once

#include "DeepLearning/Api/Layers/Activations/ActivationBase.h"

#include <assert.h>
#include <memory>

namespace Thor {

using std::shared_ptr;

class Activation {
   public:
    Activation();
    Activation(ActivationBase *activationBase) { activation = shared_ptr<ActivationBase>(activationBase); }

    virtual ~Activation() {}

   private:
    shared_ptr<ActivationBase> activation;
};

}  // namespace Thor
