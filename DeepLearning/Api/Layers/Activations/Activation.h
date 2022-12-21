#pragma once

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Network/Network.h"

#include <assert.h>
#include <atomic>
#include <utility>

namespace Thor {

class Activation : public Layer {
   public:
    class Builder;

    Activation() {}
    virtual ~Activation() {}
};

class Activation::Builder {
   public:
    virtual ~Builder() {}
    // Note: the builder functions return void because if they can't return an Activation::Builder because it is abstract.
    virtual void network(Network &_network) = 0;
    virtual void featureInput(Tensor featureInput) = 0;
    virtual std::shared_ptr<Layer> build() = 0;
    // You can clone a builder to instantiate multiple distinct instances because the id is only generated when build() is called.
    virtual std::shared_ptr<Builder> clone() = 0;
};

}  // namespace Thor
