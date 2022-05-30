#pragma once

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Implementation/Layers/Utility/Reshape.h"

namespace Thor {

class Reshape : public Layer {
   public:
    class Builder;
    Reshape() {}

    virtual ~Reshape() {}

    virtual shared_ptr<Layer> clone() const { return make_shared<Reshape>(*this); }

    virtual string getLayerType() const { return "Reshape"; }

   protected:
    virtual ThorImplementation::Layer *stamp(ThorImplementation::TensorPlacement placement,
                                             ThorImplementation::Layer *drivingLayer,
                                             Thor::Layer *drivingApiLayer,
                                             Thor::Tensor connectingApiTensor,
                                             vector<shared_ptr<Initializer>> &initializers) const {
        assert(initialized);
        assert(connectingApiTensor == getFeatureInput());

        // Implementation has 1 extra dimension due to having the batchSize dimension
        ThorImplementation::Reshape *Reshape = new ThorImplementation::Reshape(newDimensions);
        Thor::Layer::connectTwoLayers(drivingLayer, Reshape, drivingApiLayer, this, connectingApiTensor);
        return Reshape;
    }

    // Reshape only changes the descriptor, no tensor is allocated
    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, TensorPlacement tensorPlacement) const { return 0; }

    vector<uint64_t> newDimensions;
};

class Reshape::Builder {
   public:
    virtual Reshape build() {
        assert(_network.isPresent());
        assert(_featureInput.isPresent());
        assert(_newDimensions.isPresent());

        Reshape Reshape;
        Reshape.featureInput = _featureInput;
        Reshape.featureOutput = Tensor(_featureInput.get().getDataType(), _newDimensions.get());
        assert(Reshape.featureInput.get().getTotalNumElements() == Reshape.featureOutput.get().getTotalNumElements());
        Reshape.initialized = true;
        Reshape.addToNetwork(_network.get());
        return Reshape;
    }

    virtual Reshape::Builder &network(Network &_network) {
        assert(!this->_network.isPresent());
        this->_network = &_network;
        return *this;
    }

    virtual Reshape::Builder &featureInput(Tensor _featureInput) {
        assert(!this->_featureInput.isPresent());
        this->_featureInput = _featureInput;
        return *this;
    }

    virtual Reshape::Builder &newDimensions(vector<uint64_t> _newDimensions) {
        assert(!this->_newDimensions.isPresent());
        assert(_newDimensions.size() > 0);
        this->_newDimensions = _newDimensions;
        return *this;
    }

   private:
    Optional<Network *> _network;
    Optional<Tensor> _featureInput;
    Optional<vector<uint64_t>> _newDimensions;
};

}  // namespace Thor
