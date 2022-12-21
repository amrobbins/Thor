#pragma once

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Implementation/Layers/Activation/Gelu.h"

namespace Thor {

class Gelu : public Activation {
   public:
    class Builder;
    Gelu() {}

    virtual ~Gelu() {}

    virtual std::shared_ptr<Layer> clone() const { return std::make_shared<Gelu>(*this); }

    virtual std::string getLayerType() const { return "Gelu"; }

   protected:
    virtual ThorImplementation::Layer *stamp(ThorImplementation::TensorPlacement placement,
                                             ThorImplementation::Layer *drivingLayer,
                                             Thor::Layer *drivingApiLayer,
                                             Thor::Tensor connectingApiTensor,
                                             std::vector<std::shared_ptr<Initializer>> &initializers) const {
        assert(initialized);
        assert(connectingApiTensor == featureInput.get());

        ThorImplementation::Gelu *gelu = new ThorImplementation::Gelu();
        Thor::Layer::connectTwoLayers(drivingLayer, gelu, drivingApiLayer, this, connectingApiTensor);
        return gelu;
    }

    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const {
        // feature out and error out
        return batchSize * (featureOutput.get().getTotalSizeInBytes() + featureInput.get().getTotalSizeInBytes());
    }
};

class Gelu::Builder : public Activation::Builder {
   public:
    virtual std::shared_ptr<Layer> build() {
        assert(_network.isPresent());
        assert(_featureInput.isPresent());

        Gelu gelu;
        gelu.featureInput = _featureInput;
        gelu.featureOutput = _featureInput.get().clone();
        gelu.initialized = true;
        gelu.addToNetwork(_network.get());
        return gelu.clone();
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

    virtual std::shared_ptr<Activation::Builder> clone() { return std::make_shared<Gelu::Builder>(*this); }

   private:
    Optional<Network *> _network;
    Optional<Tensor> _featureInput;
};

}  // namespace Thor
