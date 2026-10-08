#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Activations/Activation.h"

namespace Thor {

class Tanh : public Activation {
   public:
    class Builder;
    Tanh() {}

    ~Tanh() override {}

    std::shared_ptr<Layer> clone() const override {
        std::shared_ptr<Tanh> myClone = std::make_shared<Tanh>(*this);
        myClone->id = getUnusedId();
        return myClone;
    }

    ThorImplementation::Expression toExpression(const ThorImplementation::Expression& input) const override {
        return input.tanh();
    }

    std::string getLayerType() const override { return "Tanh"; }

    static void deserialize(const nlohmann::json &j, Network *network) {
        if (j.at("version").get<std::string>() != "1.0.0")
            throw std::runtime_error("Unsupported version in Tanh::deserialize: " + j["version"].get<std::string>());
        if (j.at("layer_type").get<std::string>() != "tanh")
            throw std::runtime_error("Layer type mismatch in Tanh::deserialize: " + j.at("layer_type").get<std::string>());

        Tanh activation;
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

class Tanh::Builder : public Activation::Builder {
   public:
    std::shared_ptr<Activation> build() override {
        std::shared_ptr<Tanh> tanh = std::make_shared<Tanh>();
        if (_featureInput.has_value()) {
            THOR_THROW_IF_FALSE(_network.has_value());
            applyStandaloneConfiguration(*tanh);
            tanh->initialized = true;
            tanh->addToNetwork(_network.value());
        } else {
            tanh->initialized = true;
        }

        return tanh;
    }

    Tanh::Builder &network(Network &_network) override {
        Activation::Builder::network(_network);
        return *this;
    }

    Tanh::Builder &featureInput(Tensor _featureInput) override {
        Activation::Builder::featureInput(_featureInput);
        return *this;
    }

    Tanh::Builder &featureInput(RaggedTensor _featureInput) override {
        Activation::Builder::featureInput(_featureInput);
        return *this;
    }
};

}  // namespace Thor
