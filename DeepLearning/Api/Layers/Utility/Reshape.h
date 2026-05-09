#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Implementation/Layers/Utility/Reshape.h"
#include <optional>

namespace Thor {

class Reshape : public Layer {
   public:
    class Builder;
    Reshape();
    ~Reshape() override;

    std::shared_ptr<Layer> clone() const override { return std::make_shared<Reshape>(*this); }

    std::string getLayerType() const override { return "Reshape"; }

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

        // Implementation has 1 extra dimension due to having the batchSize dimension, this is handled by the builder
        std::shared_ptr<ThorImplementation::Reshape> Reshape = std::make_shared<ThorImplementation::Reshape>(newDimensions);
        return Reshape;
    }

    // Reshape only changes the descriptor, no tensor is allocated
    uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const override {
        return 0;
    }

    std::vector<uint64_t> newDimensions;
};

class Reshape::Builder {
   public:
    virtual Reshape build() {
        THOR_THROW_IF_FALSE(_network.has_value());
        THOR_THROW_IF_FALSE(_featureInput.has_value());
        THOR_THROW_IF_FALSE(_newDimensions.has_value());

        Reshape reshape;
        reshape.featureInput = _featureInput;
        reshape.featureOutput = Tensor(_featureInput.value().getDataType(), _newDimensions.value());
        THOR_THROW_IF_FALSE(reshape.featureInput.value().getTotalNumElements() == reshape.featureOutput.value().getTotalNumElements());

        // Implementation layer has one extra (batch) dimension, set to 0 to tell implementation layer to get it from featureIn
        reshape.newDimensions.push_back(0);
        for (uint32_t i = 0; i < _newDimensions.value().size(); ++i)
            reshape.newDimensions.push_back(_newDimensions.value()[i]);

        reshape.initialized = true;
        reshape.addToNetwork(_network.value());
        return reshape;
    }

    virtual Reshape::Builder &network(Network &_network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &_network;
        return *this;
    }

    virtual Reshape::Builder &featureInput(Tensor _featureInput) {
        THOR_THROW_IF_FALSE(!this->_featureInput.has_value());
        this->_featureInput = _featureInput;
        return *this;
    }

    virtual Reshape::Builder &newDimensions(std::vector<uint64_t> _newDimensions) {
        THOR_THROW_IF_FALSE(!this->_newDimensions.has_value());
        THOR_THROW_IF_FALSE(_newDimensions.size() > 0);
        this->_newDimensions = _newDimensions;
        return *this;
    }

   private:
    std::optional<Network *> _network;
    std::optional<Tensor> _featureInput;
    std::optional<std::vector<uint64_t>> _newDimensions;
};

}  // namespace Thor
