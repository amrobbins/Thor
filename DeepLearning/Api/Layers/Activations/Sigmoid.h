#pragma once

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Implementation/Layers/Activation/Sigmoid.h"

namespace Thor {

class BinaryCrossEntropy;

class Sigmoid : public Activation {
   public:
    class Builder;
    Sigmoid() {}

    virtual ~Sigmoid() {}

    virtual std::shared_ptr<Layer> clone() const { return std::make_shared<Sigmoid>(*this); }

    virtual std::string getLayerType() const { return "Sigmoid"; }

   protected:
    virtual std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                             std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                             std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                             Thor::Tensor connectingApiTensor) const {
        assert(initialized);
        assert(connectingApiTensor == featureInput.get());

        std::shared_ptr<ThorImplementation::Sigmoid> sigmoid = std::make_shared<ThorImplementation::Sigmoid>(backwardComputedExternally);
        return sigmoid;
    }

    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const {
        // feature out and error out
        return batchSize * (featureOutput.get().getTotalSizeInBytes() + featureInput.get().getTotalSizeInBytes());
    }

    virtual nlohmann::json serialize() {
        return nlohmann::json{{"type", "sigmoid"}};
    }

    bool backwardComputedExternally;
};

class Sigmoid::Builder : public Activation::Builder {
   public:
    virtual std::shared_ptr<Layer> build() {
        assert(_network.isPresent());
        assert(_featureInput.isPresent());

        Sigmoid sigmoid;
        sigmoid.featureInput = _featureInput;
        sigmoid.featureOutput = _featureInput.get().clone();
        if (_backwardComputedExternally.isPresent() && _backwardComputedExternally.get() == true)
            sigmoid.backwardComputedExternally = true;
        else
            sigmoid.backwardComputedExternally = false;
        sigmoid.initialized = true;
        sigmoid.addToNetwork(_network.get());
        return sigmoid.clone();
    }

    virtual void network(Network &_network) { this->_network = &_network; }

    virtual void featureInput(Tensor _featureInput) {
        assert(!_featureInput.getDimensions().empty());
        this->_featureInput = _featureInput;
    }

    virtual std::shared_ptr<Activation::Builder> clone() { return std::make_shared<Sigmoid::Builder>(*this); }

   protected:
    void backwardComputedExternally() {
        assert(!_backwardComputedExternally.isPresent());
        _backwardComputedExternally = true;
    }

   private:
    Optional<Network *> _network;
    Optional<Tensor> _featureInput;
    Optional<bool> _backwardComputedExternally;

    friend class BinaryCrossEntropy;
};

}  // namespace Thor
