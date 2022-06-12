#pragma once

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Implementation/Layers/Activation/SoftPlus.h"

namespace Thor {

class SoftPlus : public Activation {
   public:
    class Builder;
    SoftPlus() {}

    virtual ~SoftPlus() {}

    virtual shared_ptr<Layer> clone() const { return make_shared<SoftPlus>(*this); }

    virtual string getLayerType() const { return "SoftPlus"; }

   protected:
    virtual ThorImplementation::Layer *stamp(ThorImplementation::TensorPlacement placement,
                                             ThorImplementation::Layer *drivingLayer,
                                             Thor::Layer *drivingApiLayer,
                                             Thor::Tensor connectingApiTensor,
                                             vector<shared_ptr<Initializer>> &initializers) const {
        assert(initialized);
        assert(connectingApiTensor == featureInput.get());

        ThorImplementation::SoftPlus *softPlus = new ThorImplementation::SoftPlus();
        Thor::Layer::connectTwoLayers(drivingLayer, softPlus, drivingApiLayer, this, connectingApiTensor);
        return softPlus;
    }

    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, TensorPlacement tensorPlacement) const {
        // feature out and error out
        return batchSize * (featureOutput.get().getTotalSizeInBytes() + featureInput.get().getTotalSizeInBytes());
    }
};

class SoftPlus::Builder : public Activation::Builder {
   public:
    virtual shared_ptr<Layer> build() {
        assert(_network.isPresent());
        assert(_featureInput.isPresent());

        SoftPlus softPlus;
        softPlus.featureInput = _featureInput;
        softPlus.featureOutput = _featureInput.get().clone();
        softPlus.initialized = true;
        softPlus.addToNetwork(_network.get());
        return softPlus.clone();
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

    virtual shared_ptr<Activation::Builder> clone() { return make_shared<SoftPlus::Builder>(*this); }

   private:
    Optional<Network *> _network;
    Optional<Tensor> _featureInput;
};

}  // namespace Thor
