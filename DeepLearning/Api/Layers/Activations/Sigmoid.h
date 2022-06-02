#pragma once

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Implementation/Layers/Activation/Sigmoid.h"

namespace Thor {

class Sigmoid : public Activation {
   public:
    class Builder;
    Sigmoid() {}

    virtual ~Sigmoid() {}

    virtual shared_ptr<Layer> clone() const { return make_shared<Sigmoid>(*this); }

    virtual string getLayerType() const { return "Sigmoid"; }

   protected:
    virtual ThorImplementation::Layer *stamp(ThorImplementation::TensorPlacement placement,
                                             ThorImplementation::Layer *drivingLayer,
                                             Thor::Layer *drivingApiLayer,
                                             Thor::Tensor connectingApiTensor,
                                             vector<shared_ptr<Initializer>> &initializers) const {
        assert(initialized);
        assert(connectingApiTensor == featureInput.get());

        ThorImplementation::Sigmoid *sigmoid = new ThorImplementation::Sigmoid();
        Thor::Layer::connectTwoLayers(drivingLayer, sigmoid, drivingApiLayer, this, connectingApiTensor);
        return sigmoid;
    }

    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, TensorPlacement tensorPlacement) const {
        // feature out and error out
        return batchSize * (featureOutput.get().getTotalSizeInBytes() + featureInput.get().getTotalSizeInBytes());
    }
};

class Sigmoid::Builder : public Activation::Builder {
   public:
    virtual shared_ptr<Layer> build() {
        assert(_network.isPresent());
        assert(_featureInput.isPresent());

        Sigmoid sigmoid;
        sigmoid.featureInput = _featureInput;
        sigmoid.featureOutput = _featureInput.get().clone();
        sigmoid.initialized = true;
        sigmoid.addToNetwork(_network.get());
        return sigmoid.clone();
    }

    virtual void network(Network &_network) { this->_network = &_network; }

    virtual void featureInput(Tensor _featureInput) {
        assert(!_featureInput.getDimensions().empty());
        this->_featureInput = _featureInput;
    }

    virtual shared_ptr<Activation::Builder> clone() { return make_shared<Sigmoid::Builder>(*this); }

   private:
    Optional<Network *> _network;
    Optional<Tensor> _featureInput;
};

}  // namespace Thor
