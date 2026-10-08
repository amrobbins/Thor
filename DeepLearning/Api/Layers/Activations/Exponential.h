#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Activations/Activation.h"

namespace Thor {

class Exponential : public Activation {
   public:
    class Builder;
    Exponential() {}

    ~Exponential() override {}

    std::shared_ptr<Layer> clone() const override {
        std::shared_ptr<Exponential> myClone = std::make_shared<Exponential>(*this);
        myClone->id = getUnusedId();
        return myClone;
    }

    ThorImplementation::Expression toExpression(const ThorImplementation::Expression& input) const override {
        return input.exp();
    }

    std::string getLayerType() const override { return "Exponential"; }

    static void deserialize(const nlohmann::json &j, Network *network) {
        if (j.at("version").get<std::string>() != "1.0.0")
            throw std::runtime_error("Unsupported version in Exponential::deserialize: " + j["version"].get<std::string>());
        if (j.at("layer_type").get<std::string>() != "exponential")
            throw std::runtime_error("Layer type mismatch in Exponential::deserialize: " + j.at("layer_type").get<std::string>());

        Exponential exponential;
        exponential.deserializeStandaloneFields(j, network);
        exponential.addToNetwork(network);
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

class Exponential::Builder : public Activation::Builder {
   public:
    std::shared_ptr<Activation> build() override {
        std::shared_ptr<Exponential> exponential = std::make_shared<Exponential>();
        if (_featureInput.has_value()) {
            THOR_THROW_IF_FALSE(_network.has_value());
            applyStandaloneConfiguration(*exponential);
            exponential->initialized = true;
            exponential->addToNetwork(_network.value());
        } else {
            exponential->initialized = true;
        }

        return exponential;
    }

    Exponential::Builder &network(Network &_network) override {
        Activation::Builder::network(_network);
        return *this;
    }

    Exponential::Builder &featureInput(Tensor _featureInput) override {
        Activation::Builder::featureInput(_featureInput);
        return *this;
    }

    Exponential::Builder &featureInput(RaggedTensor _featureInput) override {
        Activation::Builder::featureInput(_featureInput);
        return *this;
    }
};

}  // namespace Thor
