#pragma once

#include "DeepLearning/Api/Layers/Activations/Activation.h"

namespace Thor {

class Relu : public Activation {
   public:
    class Builder;
    Relu() {}

    virtual ~Relu() {}

    virtual shared_ptr<Layer> clone() const { return make_shared<Relu>(*this); }

   protected:
    virtual ThorImplementation::Layer *stamp(ThorImplementation::TensorPlacement, uint32_t batchSize) const {
        // FIXME
        return nullptr;
    }

    virtual shared_ptr<Activation> cloneWithReconnect(Tensor newFeatureInput) {
        shared_ptr<Relu> relu = make_shared<Relu>();
        *relu = *this;
        relu->featureInput = newFeatureInput;
        relu->featureOutput = newFeatureInput.clone();
        return relu;
    }
};

class Relu::Builder {
   public:
    virtual Relu build() {
        assert(_network.isPresent());
        assert(_featureInput.isPresent());

        Relu relu;
        relu.featureInput = _featureInput;
        relu.featureOutput = _featureInput.get().clone();
        relu.initialized = true;
        relu.addToNetwork(_network.get());
        return relu;
    }

    virtual Relu::Builder &network(Network &_network) {
        assert(!this->_network.isPresent());
        this->_network = &_network;
        return *this;
    }

    virtual Relu::Builder &featureInput(Tensor _featureInput) {
        assert(!this->_featureInput.isPresent());
        assert(!_featureInput.getDimensions().empty());
        this->_featureInput = _featureInput;
        return *this;
    }

   private:
    Optional<Network *> _network;
    Optional<Tensor> _featureInput;
};

}  // namespace Thor
