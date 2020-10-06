#pragma once

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Implementation/Layers/Utility/Flatten.h"

namespace Thor {

class Flatten : public Layer {
   public:
    class Builder;
    Flatten() {}

    virtual ~Flatten() {}

    virtual shared_ptr<Layer> clone() const { return make_shared<Flatten>(*this); }

   protected:
    virtual ThorImplementation::Layer *stamp(ThorImplementation::TensorPlacement placement,
                                             ThorImplementation::Layer *drivingLayer,
                                             Thor::Layer *drivingApiLayer,
                                             Thor::Tensor connectingApiTensor,
                                             vector<shared_ptr<Initializer>> &initializers) const {
        assert(initialized);
        assert(connectingApiTensor == getFeatureInput());

        // Implemenattion has 1 extra dimension due to having the batchSize dimension
        ThorImplementation::Flatten *flatten = new ThorImplementation::Flatten(getFeatureOutput().get().getDimensions().size() + 1);
        Thor::Layer::connectTwoLayers(drivingLayer, flatten, drivingApiLayer, this, connectingApiTensor);
        return flatten;
    }

    // Flatten only changes the descriptor, no tensor is allocated
    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize) const { return 0; }
};

class Flatten::Builder {
   public:
    virtual Flatten build() {
        assert(_network.isPresent());
        assert(_featureInput.isPresent());
        assert(_numOutputDimensions.isPresent());
        assert(_numOutputDimensions.get() < _featureInput.get().getDimensions().size());

        vector<uint64_t> inputDimensions = _featureInput.get().getDimensions();
        assert(inputDimensions.size() > 0);
        vector<uint64_t> outputDimensions;
        for (uint32_t i = 0; i < inputDimensions.size(); ++i) {
            if (i < _numOutputDimensions)
                outputDimensions.push_back(inputDimensions[i]);
            else
                outputDimensions.back() *= inputDimensions[i];
        }

        Flatten flatten;
        flatten.featureInput = _featureInput;
        flatten.featureOutput = Tensor(_featureInput.get().getDataType(), outputDimensions);
        flatten.initialized = true;
        flatten.addToNetwork(_network.get());
        return flatten;
    }

    virtual Flatten::Builder &network(Network &_network) {
        assert(!this->_network.isPresent());
        this->_network = &_network;
        return *this;
    }

    virtual Flatten::Builder &featureInput(Tensor _featureInput) {
        assert(!this->_featureInput.isPresent());
        this->_featureInput = _featureInput;
        return *this;
    }

    virtual Flatten::Builder &numOutputDimensions(float _numOutputDimensions) {
        assert(!this->_numOutputDimensions.isPresent());
        assert(_numOutputDimensions > 0);
        this->_numOutputDimensions = _numOutputDimensions;
        return *this;
    }

   private:
    Optional<Network *> _network;
    Optional<Tensor> _featureInput;
    Optional<uint32_t> _numOutputDimensions;
};

}  // namespace Thor
