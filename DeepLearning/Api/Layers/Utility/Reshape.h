#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"
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

    [[nodiscard]] bool getUseRagged() const { return raggedFeatureInput.has_value(); }
    [[nodiscard]] std::optional<RaggedTensor> getRaggedFeatureInput() const { return raggedFeatureInput; }
    [[nodiscard]] std::optional<RaggedTensor> getRaggedFeatureOutput() const { return raggedFeatureOutput; }

    [[nodiscard]] bool outputTensorDimensionsIncludeBatch(const Tensor& outputTensor) const override {
        THOR_THROW_IF_FALSE(featureOutput.has_value());
        THOR_THROW_IF_FALSE(outputTensor == featureOutput.value());
        return raggedFeatureInput.has_value();
    }

    nlohmann::json architectureJson() const override;
    static void deserialize(const nlohmann::json &j, Network *network);

   protected:
    std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                     std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                     std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                     Thor::Tensor connectingApiTensor,
                                                     const bool inferenceOnly) const override {
        (void)placement;
        (void)drivingLayer;
        (void)drivingApiLayer;
        (void)inferenceOnly;
        THOR_THROW_IF_FALSE(initialized);
        THOR_THROW_IF_FALSE(connectingApiTensor == getFeatureInput().value());

        return std::make_shared<ThorImplementation::Reshape>(newDimensions);
    }

    // Reshape only changes the descriptor, no tensor is allocated.
    uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const override {
        (void)batchSize;
        (void)tensorPlacement;
        return 0;
    }

    std::vector<uint64_t> newDimensions;
    std::optional<RaggedTensor> raggedFeatureInput;
    std::optional<RaggedTensor> raggedFeatureOutput;

    friend class Builder;
};

class Reshape::Builder {
   public:
    virtual Reshape build();

    virtual Reshape::Builder &network(Network &_network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &_network;
        return *this;
    }

    virtual Reshape::Builder &featureInput(Tensor _featureInput) {
        THOR_THROW_IF_FALSE(!this->_featureInput.has_value());
        THOR_THROW_IF_FALSE(!this->_raggedFeatureInput.has_value());
        THOR_THROW_IF_FALSE(_featureInput.isInitialized());
        this->_featureInput = _featureInput;
        return *this;
    }

    virtual Reshape::Builder &featureInput(RaggedTensor _featureInput) {
        THOR_THROW_IF_FALSE(!this->_featureInput.has_value());
        THOR_THROW_IF_FALSE(!this->_raggedFeatureInput.has_value());
        THOR_THROW_IF_FALSE(_featureInput.isInitialized());
        this->_raggedFeatureInput = _featureInput;
        this->_featureInput = _featureInput.getValues();
        return *this;
    }

    virtual Reshape::Builder &newDimensions(std::vector<uint64_t> _newDimensions) {
        THOR_THROW_IF_FALSE(!this->_newDimensions.has_value());
        THOR_THROW_IF_FALSE(_newDimensions.size() > 0);
        for (uint64_t dim : _newDimensions) THOR_THROW_IF_FALSE(dim > 0);
        this->_newDimensions = std::move(_newDimensions);
        return *this;
    }

   private:
    std::optional<Network *> _network;
    std::optional<Tensor> _featureInput;
    std::optional<RaggedTensor> _raggedFeatureInput;
    std::optional<std::vector<uint64_t>> _newDimensions;
};

}  // namespace Thor
