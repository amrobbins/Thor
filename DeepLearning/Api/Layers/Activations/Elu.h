#pragma once

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Implementation/Layers/Activation/Elu.h"

namespace Thor {

class Elu : public Activation {
   public:
    class Builder;
    Elu() {}

    virtual ~Elu() {}

    virtual shared_ptr<Layer> clone() const { return make_shared<Elu>(*this); }

    virtual string getLayerType() const { return "Elu"; }

   protected:
    virtual ThorImplementation::Layer *stamp(ThorImplementation::TensorPlacement placement,
                                             ThorImplementation::Layer *drivingLayer,
                                             Thor::Layer *drivingApiLayer,
                                             Thor::Tensor connectingApiTensor,
                                             vector<shared_ptr<Initializer>> &initializers) const {
        assert(initialized);
        assert(connectingApiTensor == featureInput.get());

        ThorImplementation::Elu *elu = new ThorImplementation::Elu(alpha);
        Thor::Layer::connectTwoLayers(drivingLayer, elu, drivingApiLayer, this, connectingApiTensor);
        return elu;
    }

    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, TensorPlacement tensorPlacement) const {
        // feature out and error out
        return batchSize * (featureOutput.get().getTotalSizeInBytes() + featureInput.get().getTotalSizeInBytes());
    }

    float alpha;
};

class Elu::Builder : public Activation::Builder {
   public:
    virtual shared_ptr<Layer> build() {
        assert(_network.isPresent());
        assert(_featureInput.isPresent());

        Elu elu;
        elu.featureInput = _featureInput;
        elu.featureOutput = _featureInput.get().clone();
        if (_alpha.isPresent())
            elu.alpha = _alpha;
        else
            elu.alpha = 1.0f;
        elu.initialized = true;
        elu.addToNetwork(_network.get());
        return elu.clone();
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

    virtual void alpha(float _alpha) {
        assert(!this->_alpha.isPresent());
        assert(_alpha >= 0);
        this->_alpha = _alpha;
    }

    virtual shared_ptr<Activation::Builder> clone() { return make_shared<Elu::Builder>(*this); }

   private:
    Optional<Network *> _network;
    Optional<Tensor> _featureInput;
    Optional<float> _alpha;
};

}  // namespace Thor
