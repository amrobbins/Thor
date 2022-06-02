#pragma once

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Implementation/Layers/Activation/Swish.h"

namespace Thor {

class Swish : public Activation {
   public:
    class Builder;
    Swish() {}

    virtual ~Swish() {}

    virtual shared_ptr<Layer> clone() const { return make_shared<Swish>(*this); }

    virtual string getLayerType() const { return "Swish"; }

   protected:
    virtual ThorImplementation::Layer *stamp(ThorImplementation::TensorPlacement placement,
                                             ThorImplementation::Layer *drivingLayer,
                                             Thor::Layer *drivingApiLayer,
                                             Thor::Tensor connectingApiTensor,
                                             vector<shared_ptr<Initializer>> &initializers) const {
        assert(initialized);
        assert(connectingApiTensor == featureInput.get());

        ThorImplementation::Swish *swish = new ThorImplementation::Swish();
        Thor::Layer::connectTwoLayers(drivingLayer, swish, drivingApiLayer, this, connectingApiTensor);
        return swish;
    }

    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, TensorPlacement tensorPlacement) const {
        // feature out and error out
        return batchSize * (featureOutput.get().getTotalSizeInBytes() + featureInput.get().getTotalSizeInBytes());
    }
};

class Swish::Builder : public Activation::Builder {
   public:
    virtual shared_ptr<Layer> build() {
        assert(_network.isPresent());
        assert(_featureInput.isPresent());

        Swish swish;
        swish.featureInput = _featureInput;
        swish.featureOutput = _featureInput.get().clone();
        swish.initialized = true;
        swish.addToNetwork(_network.get());
        return swish.clone();
    }

    virtual void network(Network &_network) { this->_network = &_network; }

    virtual void featureInput(Tensor _featureInput) {
        assert(!_featureInput.getDimensions().empty());
        this->_featureInput = _featureInput;
    }

    virtual shared_ptr<Activation::Builder> clone() { return make_shared<Swish::Builder>(*this); }

   private:
    Optional<Network *> _network;
    Optional<Tensor> _featureInput;
};

}  // namespace Thor
