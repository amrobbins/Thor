#pragma once

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Implementation/Layers/Activation/Selu.h"

namespace Thor {

class Selu : public Activation {
   public:
    class Builder;
    Selu() {}

    virtual ~Selu() {}

    virtual std::shared_ptr<Layer> clone() const { return std::make_shared<Selu>(*this); }

    virtual std::string getLayerType() const { return "Selu"; }

   protected:
    virtual std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                             std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                             std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                             Thor::Tensor connectingApiTensor) const {
        assert(initialized);
        assert(connectingApiTensor == featureInput.get());

        std::shared_ptr<ThorImplementation::Selu> selu = std::make_shared<ThorImplementation::Selu>();
        return selu;
    }

    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const {
        // feature out and error out
        return batchSize * (featureOutput.get().getTotalSizeInBytes() + featureInput.get().getTotalSizeInBytes());
    }
};

class Selu::Builder : public Activation::Builder {
   public:
    virtual std::shared_ptr<Layer> build() {
        assert(_network.isPresent());
        assert(_featureInput.isPresent());

        Selu selu;
        selu.featureInput = _featureInput;
        selu.featureOutput = _featureInput.get().clone();
        selu.initialized = true;
        selu.addToNetwork(_network.get());
        return selu.clone();
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

    virtual std::shared_ptr<Activation::Builder> clone() { return std::make_shared<Selu::Builder>(*this); }

   private:
    Optional<Network *> _network;
    Optional<Tensor> _featureInput;
};

}  // namespace Thor
