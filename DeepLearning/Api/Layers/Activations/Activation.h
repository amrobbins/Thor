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
    virtual Activation::Builder& network(Network& _network) {
        assert(!this->_network.isPresent());
        this->_network = &_network;
        return *this;
    }
    virtual Activation::Builder& featureInput(Tensor _featureInput) {
        assert(!this->_featureInput.isPresent());
        assert(!_featureInput.getDimensions().empty());
        this->_featureInput = _featureInput;
        return *this;
    }

    virtual std::shared_ptr<Layer> build() = 0;
    // You can clone a builder to instantiate multiple distinct instances because the id is only generated when build() is called.
    // So each builder that is built into an activation will have its own unique id.
    virtual std::shared_ptr<Builder> clone() = 0;

   protected:
    Optional<Network*> _network;
    Optional<Tensor> _featureInput;
};

}  // namespace Thor
