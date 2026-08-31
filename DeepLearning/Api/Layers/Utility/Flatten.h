#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"
#include "DeepLearning/Implementation/Layers/Utility/Flatten.h"
#include <optional>

namespace Thor {

class Flatten : public Layer {
   public:
    class Builder;
    Flatten() = default;
    ~Flatten() override = default;

    std::shared_ptr<Layer> clone() const override { return std::make_shared<Flatten>(*this); }

    std::string getLayerType() const override { return "Flatten"; }

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

        const uint32_t physicalOutputRank = static_cast<uint32_t>(getFeatureOutput().value().getDimensions().size()) +
                                            (raggedFeatureInput.has_value() ? 0U : 1U);
        return std::make_shared<ThorImplementation::Flatten>(physicalOutputRank);
    }

    // Flatten only changes the descriptor, no tensor is allocated.
    uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const override {
        (void)batchSize;
        (void)tensorPlacement;
        return 0;
    }

   private:
    std::optional<RaggedTensor> raggedFeatureInput;
    std::optional<RaggedTensor> raggedFeatureOutput;

    friend class Builder;
};

class Flatten::Builder {
   public:
    virtual Flatten build();

    virtual Flatten::Builder &network(Network &_network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &_network;
        return *this;
    }

    virtual Flatten::Builder &featureInput(Tensor _featureInput) {
        THOR_THROW_IF_FALSE(!this->_featureInput.has_value());
        THOR_THROW_IF_FALSE(!this->_raggedFeatureInput.has_value());
        THOR_THROW_IF_FALSE(_featureInput.isInitialized());
        this->_featureInput = _featureInput;
        return *this;
    }

    virtual Flatten::Builder &featureInput(RaggedTensor _featureInput) {
        THOR_THROW_IF_FALSE(!this->_featureInput.has_value());
        THOR_THROW_IF_FALSE(!this->_raggedFeatureInput.has_value());
        THOR_THROW_IF_FALSE(_featureInput.isInitialized());
        this->_raggedFeatureInput = _featureInput;
        this->_featureInput = _featureInput.getValues();
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
    std::optional<RaggedTensor> _raggedFeatureInput;
    std::optional<uint32_t> _numOutputDimensions;
};

}  // namespace Thor
