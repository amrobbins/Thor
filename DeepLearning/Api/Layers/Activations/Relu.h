#pragma once

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Implementation/Layers/Activation/Relu.h"

namespace Thor {

class Relu : public Activation {
   public:
    class Builder;
    Relu() {}

    virtual ~Relu() {}

    virtual shared_ptr<Layer> clone() const { return make_shared<Relu>(*this); }

   protected:
    virtual ThorImplementation::Layer *stamp(ThorImplementation::TensorPlacement placement, uint32_t batchSize) const {
        assert(initialized);
        return new ThorImplementation::Relu();
    }
};

class Relu::Builder : public Activation::Builder {
   public:
    virtual shared_ptr<Layer> build() {
        assert(_network.isPresent());
        assert(_featureInput.isPresent());

        Relu relu;
        relu.featureInput = _featureInput;
        relu.featureOutput = _featureInput.get().clone();
        relu.initialized = true;
        relu.addToNetwork(_network.get());
        return relu.clone();
    }

    virtual void network(Network &_network) { this->_network = &_network; }

    virtual void featureInput(Tensor _featureInput) {
        assert(!_featureInput.getDimensions().empty());
        this->_featureInput = _featureInput;
    }

   private:
    Optional<Network *> _network;
    Optional<Tensor> _featureInput;
};

}  // namespace Thor
