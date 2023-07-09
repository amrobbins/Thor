#pragma once

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Implementation/Layers/Activation/Relu.h"

namespace Thor {

class Relu : public Activation {
   public:
    class Builder;
    Relu() {}

    virtual ~Relu() {}

    virtual std::shared_ptr<Layer> clone() const { return std::make_shared<Relu>(*this); }

    virtual std::string getLayerType() const { return "Relu"; }

   protected:
    virtual std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                             std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                             std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                             Thor::Tensor connectingApiTensor) const {
        assert(initialized);
        assert(connectingApiTensor == featureInput.get());

        std::shared_ptr<ThorImplementation::Relu> relu = std::make_shared<ThorImplementation::Relu>();
        return relu;
    }

    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const {
        // feature out and error out
        return batchSize * (featureOutput.get().getTotalSizeInBytes() + featureInput.get().getTotalSizeInBytes());
    }
};

class Relu::Builder : public Activation::Builder {
   public:
    virtual std::shared_ptr<Layer> build() {
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

    virtual std::shared_ptr<Activation::Builder> clone() { return std::make_shared<Relu::Builder>(*this); }

   private:
    Optional<Network *> _network;
    Optional<Tensor> _featureInput;
};

}  // namespace Thor
