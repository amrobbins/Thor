#pragma once

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Implementation/Layers/Activation/Softmax.h"

namespace Thor {

class CategoricalCrossEntropy;

class Softmax : public Activation {
   public:
    class Builder;
    Softmax() {}

    virtual ~Softmax() {}

    virtual std::shared_ptr<Layer> clone() const { return std::make_shared<Softmax>(*this); }

    virtual std::string getLayerType() const { return "Softmax"; }

   protected:
    virtual std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                             std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                             std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                             Thor::Tensor connectingApiTensor) const {
        assert(initialized);
        assert(connectingApiTensor == featureInput.get());

        std::shared_ptr<ThorImplementation::Softmax> softmax = std::make_shared<ThorImplementation::Softmax>(backwardComputedExternally);
        return softmax;
    }

    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const {
        // feature out and error out
        return batchSize * (featureOutput.get().getTotalSizeInBytes() + featureInput.get().getTotalSizeInBytes());
    }

    bool backwardComputedExternally;
};

class Softmax::Builder : public Activation::Builder {
   public:
    virtual std::shared_ptr<Layer> build() {
        assert(_network.isPresent());
        assert(_featureInput.isPresent());

        Softmax softmax;
        softmax.featureInput = _featureInput;
        softmax.featureOutput = _featureInput.get().clone();
        if (_backwardComputedExternally.isPresent() && _backwardComputedExternally.get() == true)
            softmax.backwardComputedExternally = true;
        else
            softmax.backwardComputedExternally = false;
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

    virtual std::shared_ptr<Activation::Builder> clone() { return std::make_shared<Softmax::Builder>(*this); }

   protected:
    void backwardComputedExternally() {
        assert(!_backwardComputedExternally.isPresent());
        _backwardComputedExternally = true;
    }

   private:
    Optional<Network *> _network;
    Optional<Tensor> _featureInput;
    Optional<bool> _backwardComputedExternally;

    friend class Thor::CategoricalCrossEntropy;
};

}  // namespace Thor
