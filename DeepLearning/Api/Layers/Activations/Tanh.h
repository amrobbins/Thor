#pragma once

#include "DeepLearning/Api/Layers/Activations/Activation.h"

namespace Thor {

class Tanh : public Activation {
   public:
    class Builder;
    Tanh() {}

    virtual ~Tanh() {}

    virtual shared_ptr<Layer> clone() const { return make_shared<Tanh>(*this); }

   protected:
    virtual ThorImplementation::Layer *stamp(ThorImplementation::TensorPlacement, uint32_t batchSize) const {
        // FIXME
        return nullptr;
    }

    virtual shared_ptr<Activation> cloneWithReconnect(Tensor newFeatureInput) {
        shared_ptr<Tanh> tanh = make_shared<Tanh>();
        *tanh = *this;
        tanh->featureInput = newFeatureInput;
        tanh->featureOutput = newFeatureInput.clone();
        return tanh;
    }
};

class Tanh::Builder {
   public:
    virtual Tanh build() {
        assert(_network.isPresent());
        assert(_featureInput.isPresent());

        Tanh tanh;
        tanh.featureInput = _featureInput;
        tanh.featureOutput = _featureInput.get().clone();
        tanh.initialized = true;
        tanh.addToNetwork(_network.get());
        return tanh;
    }

    virtual Tanh::Builder &network(Network &_network) {
        assert(!this->_network.isPresent());
        this->_network = &_network;
        return *this;
    }

    virtual Tanh::Builder &featureInput(Tensor _featureInput) {
        assert(!this->_featureInput.isPresent());
        this->_featureInput = _featureInput;
        return *this;
    }

   private:
    Optional<Network *> _network;
    Optional<Tensor> _featureInput;
};

}  // namespace Thor
