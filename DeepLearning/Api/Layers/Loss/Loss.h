#pragma once

#include "DeepLearning/Api/Layers/Layer.h"

#include <assert.h>
#include <atomic>
#include <utility>

using std::atomic;

namespace Thor {

class Loss : public Layer {
   public:
    Loss() {}
    virtual ~Loss() {}

    Tensor getPredictions() const { return featureOutput.get(); }
    Tensor getLoss() const { return lossTensor; }

   protected:
    Tensor lossTensor;

   private:
    Tensor getFeatureOutput() { assert(false); }
};

}  // namespace Thor
