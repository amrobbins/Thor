#pragma once

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Implementation/Layers/NeuralNetwork/DropOut.h"

namespace Thor {

class DropOut : public Layer {
   public:
    class Builder;
    DropOut() {}

    virtual ~DropOut() {}

    virtual shared_ptr<Layer> clone() const { return make_shared<DropOut>(*this); }

    virtual float getDropProportion() { return dropProportion; }

   protected:
    virtual ThorImplementation::Layer *stamp(ThorImplementation::TensorPlacement placement,
                                             ThorImplementation::Layer *drivingLayer,
                                             Thor::Layer *drivingApiLayer = nullptr,
                                             Thor::Tensor connectingApiTensor = Thor::Tensor()) const {
        assert(initialized);
        assert(connectingApiTensor == getFeatureInput());

        ThorImplementation::DropOut *dropOut = new ThorImplementation::DropOut(dropProportion, true);
        Thor::Layer::connectTwoLayers(drivingLayer, dropOut, drivingApiLayer, this, connectingApiTensor);
        return dropOut;
    }

   private:
    float dropProportion;

    // friend class Network;
};

class DropOut::Builder {
   public:
    virtual DropOut build() {
        assert(_network.isPresent());
        assert(_featureInput.isPresent());
        assert(_dropProportion.isPresent());

        DropOut dropOut;
        dropOut.featureInput = _featureInput;
        dropOut.featureOutput = _featureInput.get().clone();
        dropOut.dropProportion = _dropProportion;
        dropOut.initialized = true;
        dropOut.addToNetwork(_network.get());
        return dropOut;
    }

    virtual DropOut::Builder &network(Network &_network) {
        assert(!this->_network.isPresent());
        this->_network = &_network;
        return *this;
    }

    virtual DropOut::Builder &featureInput(Tensor _featureInput) {
        assert(!this->_featureInput.isPresent());
        this->_featureInput = _featureInput;
        return *this;
    }

    virtual DropOut::Builder &dropProportion(float dropProportion) {
        assert(!_dropProportion.isPresent());
        assert(dropProportion > 0.0);
        assert(dropProportion <= 1.0);
        this->_dropProportion = dropProportion;
        return *this;
    }

   private:
    Optional<Network *> _network;
    Optional<Tensor> _featureInput;
    Optional<float> _dropProportion;
};

}  // namespace Thor
