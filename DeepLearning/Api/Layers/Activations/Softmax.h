#pragma once

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Implementation/Layers/Activation/Softmax.h"

namespace Thor {

class Softmax : public Activation {
   public:
    class Builder;
    Softmax() {}

    virtual ~Softmax() {}

    virtual shared_ptr<Layer> clone() const { return make_shared<Softmax>(*this); }

    virtual string getLayerType() const { return "Softmax"; }

   protected:
    virtual ThorImplementation::Layer *stamp(ThorImplementation::TensorPlacement placement,
                                             ThorImplementation::Layer *drivingLayer,
                                             Thor::Layer *drivingApiLayer,
                                             Thor::Tensor connectingApiTensor,
                                             vector<shared_ptr<Initializer>> &initializers) const {
        assert(initialized);
        assert(connectingApiTensor == featureInput.get());

        ThorImplementation::Softmax *softmax = new ThorImplementation::Softmax();
        Thor::Layer::connectTwoLayers(drivingLayer, softmax, drivingApiLayer, this, connectingApiTensor);
        return softmax;
    }

    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, TensorPlacement tensorPlacement) const {
        // feature out and error out
        return batchSize * (featureOutput.get().getTotalSizeInBytes() + featureInput.get().getTotalSizeInBytes());
    }
};

class Softmax::Builder : public Activation::Builder {
   public:
    virtual shared_ptr<Layer> build() {
        assert(_network.isPresent());
        assert(_featureInput.isPresent());

        Softmax softmax;
        softmax.featureInput = _featureInput;
        softmax.featureOutput = _featureInput.get().clone();
        softmax.initialized = true;
        softmax.addToNetwork(_network.get());
        return softmax.clone();
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

    virtual shared_ptr<Activation::Builder> clone() { return make_shared<Softmax::Builder>(*this); }

   private:
    Optional<Network *> _network;
    Optional<Tensor> _featureInput;
};

}  // namespace Thor
