#pragma once

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Implementation/Layers/Activation/Exponential.h"

namespace Thor {

class Exponential : public Activation {
   public:
    class Builder;
    Exponential() {}

    virtual ~Exponential() {}

    virtual shared_ptr<Layer> clone() const { return make_shared<Exponential>(*this); }

    virtual string getLayerType() const { return "Exponential"; }

   protected:
    virtual ThorImplementation::Layer *stamp(ThorImplementation::TensorPlacement placement,
                                             ThorImplementation::Layer *drivingLayer,
                                             Thor::Layer *drivingApiLayer,
                                             Thor::Tensor connectingApiTensor,
                                             vector<shared_ptr<Initializer>> &initializers) const {
        assert(initialized);
        assert(connectingApiTensor == featureInput.get());

        ThorImplementation::Exponential *exponential = new ThorImplementation::Exponential();
        Thor::Layer::connectTwoLayers(drivingLayer, exponential, drivingApiLayer, this, connectingApiTensor);
        return exponential;
    }

    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, TensorPlacement tensorPlacement) const {
        // feature out and error out
        return batchSize * (featureOutput.get().getTotalSizeInBytes() + featureInput.get().getTotalSizeInBytes());
    }
};

class Exponential::Builder : public Activation::Builder {
   public:
    virtual shared_ptr<Layer> build() {
        assert(_network.isPresent());
        assert(_featureInput.isPresent());

        Exponential exponential;
        exponential.featureInput = _featureInput;
        exponential.featureOutput = _featureInput.get().clone();
        exponential.initialized = true;
        exponential.addToNetwork(_network.get());
        return exponential.clone();
    }

    virtual void network(Network &_network) {
        assert(!this->_network.isPresent());
        this->_network = &_network;
    }

    virtual void featureInput(Tensor _featureInput) {
        assert(!this->_featureInput.isPresent());
        assert(!_featureInput.getDimensions().empty());
        this->_featureInput = _featureInput;
    }

    virtual shared_ptr<Activation::Builder> clone() { return make_shared<Exponential::Builder>(*this); }

   private:
    Optional<Network *> _network;
    Optional<Tensor> _featureInput;
    Optional<float> _alpha;
};

}  // namespace Thor