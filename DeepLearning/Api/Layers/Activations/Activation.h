#pragma once

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Network/Network.h"

#include <assert.h>
#include <atomic>
#include <utility>

using std::atomic;

namespace Thor {

class Activation : public Layer {
   public:
    class Builder;

    Activation() {}
    virtual ~Activation() {}

    virtual shared_ptr<Layer> clone() const { assert(false); }

    virtual shared_ptr<Activation> cloneReconnectAndAddToNetwork(Tensor newFeatureInput, Network &network) { assert(false); }

   protected:
    virtual ThorImplementation::Layer *stamp(ThorImplementation::TensorPlacement, uint32_t batchSize) const { assert(false); }
};

class Activation::Builder {
   public:
    virtual ~Builder() {}
    virtual void network(Network &_network) { assert(false); }
    virtual void featureInput(Tensor featureInput) { assert(false); }
    virtual shared_ptr<Layer> build() { assert(false); }
    virtual shared_ptr<Builder> clone() { assert(false); }
};

}  // namespace Thor
