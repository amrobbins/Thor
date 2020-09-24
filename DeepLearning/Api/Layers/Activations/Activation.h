#pragma once

#include "DeepLearning/Api/Layers/Layer.h"

#include <assert.h>
#include <atomic>
#include <utility>

using std::atomic;

namespace Thor {

class Activation : public Layer {
   public:
    Activation() {}
    virtual ~Activation() {}

    virtual shared_ptr<Layer> clone() const { assert(false); }

    virtual shared_ptr<Activation> cloneWithReconnect(Tensor newFeatureInput) { assert(false); }

   protected:
    virtual ThorImplementation::Layer *stamp(ThorImplementation::TensorPlacement, uint32_t batchSize) const { assert(false); }
};

}  // namespace Thor
