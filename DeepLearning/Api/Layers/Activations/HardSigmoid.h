#pragma once

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Implementation/Layers/Activation/HardSigmoid.h"

namespace Thor {

class HardSigmoid : public Activation {
   public:
    class Builder;
    HardSigmoid() {}

    virtual ~HardSigmoid() {}

    virtual std::shared_ptr<Layer> clone() const { return std::make_shared<HardSigmoid>(*this); }

    virtual std::string getLayerType() const { return "HardSigmoid"; }

   protected:
    virtual ThorImplementation::Layer *stamp(ThorImplementation::TensorPlacement placement,
                                             ThorImplementation::Layer *drivingLayer,
                                             Thor::Layer *drivingApiLayer,
                                             Thor::Tensor connectingApiTensor,
                                             std::vector<std::shared_ptr<Initializer>> &initializers) const {
        assert(initialized);
        assert(connectingApiTensor == featureInput.get());

        ThorImplementation::HardSigmoid *hardSigmoid = new ThorImplementation::HardSigmoid();
        Thor::Layer::connectTwoLayers(drivingLayer, hardSigmoid, drivingApiLayer, this, connectingApiTensor);
        return hardSigmoid;
    }

    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const {
        // feature out and error out
        return batchSize * (featureOutput.get().getTotalSizeInBytes() + featureInput.get().getTotalSizeInBytes());
    }
};

class HardSigmoid::Builder : public Activation::Builder {
   public:
    virtual std::shared_ptr<Layer> build() {
        assert(_network.isPresent());
        assert(_featureInput.isPresent());

        HardSigmoid hardSigmoid;
        hardSigmoid.featureInput = _featureInput;
        hardSigmoid.featureOutput = _featureInput.get().clone();
        hardSigmoid.initialized = true;
        hardSigmoid.addToNetwork(_network.get());
        return hardSigmoid.clone();
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

    virtual std::shared_ptr<Activation::Builder> clone() { return std::make_shared<HardSigmoid::Builder>(*this); }

   private:
    Optional<Network *> _network;
    Optional<Tensor> _featureInput;
};

}  // namespace Thor
