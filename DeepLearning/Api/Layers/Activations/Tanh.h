#pragma once

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Implementation/Layers/Activation/Tanh.h"

namespace Thor {

class Tanh : public Activation {
   public:
    class Builder;
    Tanh() {}

    virtual ~Tanh() {}

    virtual std::shared_ptr<Layer> clone() const { return std::make_shared<Tanh>(*this); }

    virtual std::string getLayerType() const { return "Tanh"; }

   protected:
    virtual std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                             std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                             std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                             Thor::Tensor connectingApiTensor) const {
        assert(initialized);
        assert(connectingApiTensor == featureInput.get());

        std::shared_ptr<ThorImplementation::Tanh> tanh = std::make_shared<ThorImplementation::Tanh>();
        return tanh;
    }

    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const {
        // feature out and error out
        return batchSize * (featureOutput.get().getTotalSizeInBytes() + featureInput.get().getTotalSizeInBytes());
    }

    virtual nlohmann::json serialize() { return nlohmann::json{{"type", "tanh"}}; }
};

class Tanh::Builder : public Activation::Builder {
   public:
    virtual std::shared_ptr<Layer> build() {
        assert(_network.isPresent());
        assert(_featureInput.isPresent());

        Tanh tanh;
        tanh.featureInput = _featureInput;
        tanh.featureOutput = _featureInput.get().clone();
        tanh.initialized = true;
        tanh.addToNetwork(_network.get());
        return tanh.clone();
    }

    virtual void network(Network &_network) { this->_network = &_network; }

    virtual void featureInput(Tensor _featureInput) {
        assert(!_featureInput.getDimensions().empty());
        this->_featureInput = _featureInput;
    }

    virtual std::shared_ptr<Activation::Builder> clone() { return std::make_shared<Tanh::Builder>(*this); }

   private:
    Optional<Network *> _network;
    Optional<Tensor> _featureInput;
};

}  // namespace Thor
