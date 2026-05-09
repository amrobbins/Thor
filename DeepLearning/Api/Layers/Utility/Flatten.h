#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Implementation/Layers/Utility/Flatten.h"
#include <optional>

namespace Thor {

class Flatten : public Layer {
   public:
    class Builder;
    Flatten() {}

    ~Flatten() override {}

    std::shared_ptr<Layer> clone() const override { return std::make_shared<Flatten>(*this); }

    std::string getLayerType() const override { return "Flatten"; }

    nlohmann::json architectureJson() const override;
    static void deserialize(const nlohmann::json &j, Network *network);

   protected:
    std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                     std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                     std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                     Thor::Tensor connectingApiTensor,
                                                     const bool inferenceOnly) const override {
        (void)inferenceOnly;
        THOR_THROW_IF_FALSE(initialized);
        THOR_THROW_IF_FALSE(connectingApiTensor == getFeatureInput().value());

        // Implemenattion has 1 extra dimension due to having the batchSize dimension
        std::shared_ptr<ThorImplementation::Flatten> flatten =
            std::make_shared<ThorImplementation::Flatten>(getFeatureOutput().value().getDimensions().size() + 1);
        return flatten;
    }

    // Flatten only changes the descriptor, no tensor is allocated
    uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const override {
        return 0;
    }
};

class Flatten::Builder {
   public:
    virtual Flatten build() {
        THOR_THROW_IF_FALSE(_network.has_value());
        THOR_THROW_IF_FALSE(_featureInput.has_value());
        THOR_THROW_IF_FALSE(_numOutputDimensions.has_value());
        THOR_THROW_IF_FALSE(_numOutputDimensions.value() < _featureInput.value().getDimensions().size());
        THOR_THROW_IF_FALSE(_numOutputDimensions.value() > 0);

        std::vector<uint64_t> inputDimensions = _featureInput.value().getDimensions();
        THOR_THROW_IF_FALSE(inputDimensions.size() > 0);
        std::vector<uint64_t> outputDimensions;
        for (uint32_t i = 0; i < inputDimensions.size(); ++i) {
            if (i < _numOutputDimensions.value())
                outputDimensions.push_back(inputDimensions[i]);
            else
                outputDimensions.back() *= inputDimensions[i];
        }

        Flatten flatten;
        flatten.featureInput = _featureInput;
        flatten.featureOutput = Tensor(_featureInput.value().getDataType(), outputDimensions);
        flatten.initialized = true;
        flatten.addToNetwork(_network.value());
        return flatten;
    }

    virtual Flatten::Builder &network(Network &_network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &_network;
        return *this;
    }

    virtual Flatten::Builder &featureInput(Tensor _featureInput) {
        THOR_THROW_IF_FALSE(!this->_featureInput.has_value());
        this->_featureInput = _featureInput;
        return *this;
    }

    virtual Flatten::Builder &numOutputDimensions(uint32_t _numOutputDimensions) {
        THOR_THROW_IF_FALSE(!this->_numOutputDimensions.has_value());
        THOR_THROW_IF_FALSE(_numOutputDimensions > 0);
        this->_numOutputDimensions = _numOutputDimensions;
        return *this;
    }

   private:
    std::optional<Network *> _network;
    std::optional<Tensor> _featureInput;
    std::optional<uint32_t> _numOutputDimensions;
};

}  // namespace Thor
