#pragma once

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Implementation/Layers/Activation/SoftSign.h"

namespace Thor {

class SoftSign : public Activation {
   public:
    class Builder;
    SoftSign() {}

    virtual ~SoftSign() {}

    virtual shared_ptr<Layer> clone() const { return make_shared<SoftSign>(*this); }

    virtual string getLayerType() const { return "SoftSign"; }

   protected:
    virtual ThorImplementation::Layer *stamp(ThorImplementation::TensorPlacement placement,
                                             ThorImplementation::Layer *drivingLayer,
                                             Thor::Layer *drivingApiLayer,
                                             Thor::Tensor connectingApiTensor,
                                             vector<shared_ptr<Initializer>> &initializers) const {
        assert(initialized);
        assert(connectingApiTensor == featureInput.get());

        ThorImplementation::SoftSign *softSign = new ThorImplementation::SoftSign();
        Thor::Layer::connectTwoLayers(drivingLayer, softSign, drivingApiLayer, this, connectingApiTensor);
        return softSign;
    }

    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, TensorPlacement tensorPlacement) const {
        // feature out and error out
        return batchSize * (featureOutput.get().getTotalSizeInBytes() + featureInput.get().getTotalSizeInBytes());
    }
};

class SoftSign::Builder : public Activation::Builder {
   public:
    virtual shared_ptr<Layer> build() {
        assert(_network.isPresent());
        assert(_featureInput.isPresent());

        SoftSign softSign;
        softSign.featureInput = _featureInput;
        softSign.featureOutput = _featureInput.get().clone();
        softSign.initialized = true;
        softSign.addToNetwork(_network.get());
        return softSign.clone();
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

    virtual shared_ptr<Activation::Builder> clone() { return make_shared<SoftSign::Builder>(*this); }

   private:
    Optional<Network *> _network;
    Optional<Tensor> _featureInput;
};

}  // namespace Thor
