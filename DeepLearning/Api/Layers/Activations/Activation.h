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

    virtual shared_ptr<Layer> clone() const = 0;

   protected:
    virtual ThorImplementation::Layer *stamp(ThorImplementation::TensorPlacement placement,
                                             ThorImplementation::Layer *drivingLayer,
                                             Thor::Layer *drivingApiLayer = nullptr,
                                             Thor::Tensor connectingApiTensor = Thor::Tensor()) const = 0;
};

class Activation::Builder {
   public:
    virtual ~Builder() {}
    virtual void network(Network &_network) = 0;
    virtual void featureInput(Tensor featureInput) = 0;
    virtual shared_ptr<Layer> build() = 0;
    virtual shared_ptr<Builder> clone() = 0;
};

}  // namespace Thor
