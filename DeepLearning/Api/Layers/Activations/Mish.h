#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "Utilities/Expression/DynamicExpression.h"

namespace Thor {

class Mish : public Activation {
   public:
    class Builder;
    Mish() {}

    ~Mish() override {}

    std::shared_ptr<Layer> clone() const override {
        std::shared_ptr<Mish> myClone = std::make_shared<Mish>(*this);
        myClone->id = getUnusedId();
        return myClone;
    }

    ThorImplementation::Expression toExpression(const ThorImplementation::Expression& input) const override {
        return input.mish();

    }

    std::string getLayerType() const override { return "Mish"; }

    static void deserialize(const nlohmann::json &j, Network *network) {
        if (j.at("version").get<std::string>() != "1.0.0")
            throw std::runtime_error("Unsupported version in Mish::deserialize: " + j["version"].get<std::string>());
        if (j.at("layer_type").get<std::string>() != "mish")
            throw std::runtime_error("Layer type mismatch in Mish::deserialize: " + j.at("layer_type").get<std::string>());

        Mish activation;
        activation.deserializeStandaloneFields(j, network);
        activation.addToNetwork(network);
    }

   protected:
    std::optional<std::string> placementNeutralStandaloneVersion() const override { return "1.1.0"; }

    std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                     std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                     std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                     Thor::Tensor connectingApiTensor,
                                                     const bool inferenceOnly) const override {
        (void)drivingApiLayer;
        return stampExpressionBackedActivation(placement, connectingApiTensor, inferenceOnly, drivingLayer);
    }

    uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const override {
        (void)tensorPlacement;
        return getExpressionBackedActivationMemRequirementInBytes(batchSize);
    }

};

class Mish::Builder : public Activation::Builder {
   public:
    std::shared_ptr<Activation> build() override {
        std::shared_ptr<Mish> activation = std::make_shared<Mish>();
        if (_featureInput.has_value()) {
            THOR_THROW_IF_FALSE(_network.has_value());
            applyStandaloneConfiguration(*activation);
            activation->initialized = true;
            activation->addToNetwork(_network.value());
        } else {
            activation->initialized = true;
        }

        return activation;
    }

    Mish::Builder &network(Network &_network) override {
        Activation::Builder::network(_network);
        return *this;
    }

    Mish::Builder &featureInput(Tensor _featureInput) override {
        Activation::Builder::featureInput(_featureInput);
        return *this;
    }

    Mish::Builder &featureInput(RaggedTensor _featureInput) override {
        Activation::Builder::featureInput(_featureInput);
        return *this;
    }

};

}  // namespace Thor
