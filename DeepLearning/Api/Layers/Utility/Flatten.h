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

    virtual std::shared_ptr<Layer> clone() const { return std::make_shared<Flatten>(*this); }

    virtual std::string getLayerType() const { return "Flatten"; }

   protected:
    virtual std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                             std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                             std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                             Thor::Tensor connectingApiTensor) const {
        assert(initialized);
        assert(connectingApiTensor == getFeatureInput());

        // Implemenattion has 1 extra dimension due to having the batchSize dimension
        std::shared_ptr<ThorImplementation::Flatten> flatten =
            std::make_shared<ThorImplementation::Flatten>(getFeatureOutput().get().getDimensions().size() + 1);
        return flatten;
    }

    // Flatten only changes the descriptor, no tensor is allocated
    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const {
        return 0;
    }
};

class Flatten::Builder {
   public:
    virtual Flatten build() {
        assert(_network.isPresent());
        assert(_featureInput.isPresent());
        assert(_numOutputDimensions.isPresent());
        assert(_numOutputDimensions.get() < _featureInput.get().getDimensions().size());

        std::vector<uint64_t> inputDimensions = _featureInput.get().getDimensions();
        assert(inputDimensions.size() > 0);
        std::vector<uint64_t> outputDimensions;
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
