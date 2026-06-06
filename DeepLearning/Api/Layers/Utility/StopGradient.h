#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Implementation/Layers/Utility/StopGradient.h"

#include <optional>

namespace Thor {

class StopGradient : public Layer {
   public:
    class Builder;
    StopGradient();
    ~StopGradient() override;

    std::shared_ptr<Layer> clone() const override { return std::make_shared<StopGradient>(*this); }

    std::string getLayerType() const override { return "StopGradient"; }

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

        return std::make_shared<ThorImplementation::StopGradient>();
    }

    uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const override {
        (void)batchSize;
        (void)tensorPlacement;
        return 0;
    }
};

class StopGradient::Builder {
   public:
    virtual StopGradient build() {
        THOR_THROW_IF_FALSE(_network.has_value());
        THOR_THROW_IF_FALSE(_featureInput.has_value());

        StopGradient stopGradient;
        stopGradient.featureInput = _featureInput;
        stopGradient.featureOutput = _featureInput.value().clone();
        stopGradient.initialized = true;
        stopGradient.addToNetwork(_network.value());
        return stopGradient;
    }

    virtual StopGradient::Builder &network(Network &_network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &_network;
        return *this;
    }

    virtual StopGradient::Builder &featureInput(Tensor _featureInput) {
        THOR_THROW_IF_FALSE(!this->_featureInput.has_value());
        THOR_THROW_IF_FALSE(_featureInput.isInitialized());
        this->_featureInput = _featureInput;
        return *this;
    }

   private:
    std::optional<Network *> _network;
    std::optional<Tensor> _featureInput;
};

}  // namespace Thor
