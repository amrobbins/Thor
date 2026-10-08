#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Activations/Activation.h"

namespace Thor {

class Swish : public Activation {
   public:
    class Builder;
    Swish() {}

    ~Swish() override {}

    std::shared_ptr<Layer> clone() const override {
        std::shared_ptr<Swish> myClone = std::make_shared<Swish>(*this);
        myClone->id = getUnusedId();
        return myClone;
    }

    ThorImplementation::Expression toExpression(const ThorImplementation::Expression& input) const override {
        return input.swish();
    }

    std::string getLayerType() const override { return "Swish"; }

    static void deserialize(const nlohmann::json &j, Network *network) {
        if (j.at("version").get<std::string>() != "1.0.0")
            throw std::runtime_error("Unsupported version in Swish::deserialize: " + j["version"].get<std::string>());
        if (j.at("layer_type").get<std::string>() != "swish")
            throw std::runtime_error("Layer type mismatch in Swish::deserialize: " + j.at("layer_type").get<std::string>());

        Swish activation;
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

class Swish::Builder : public Activation::Builder {
   public:
    std::shared_ptr<Activation> build() override {
        std::shared_ptr<Swish> swish = std::make_shared<Swish>();
        if (_featureInput.has_value()) {
            THOR_THROW_IF_FALSE(_network.has_value());
            applyStandaloneConfiguration(*swish);
            swish->initialized = true;
            swish->addToNetwork(_network.value());
        } else {
            swish->initialized = true;
        }

        return swish;
    }

    Swish::Builder &network(Network &_network) override {
        Activation::Builder::network(_network);
        return *this;
    }

    Swish::Builder &featureInput(Tensor _featureInput) override {
        Activation::Builder::featureInput(_featureInput);
        return *this;
    }

    Swish::Builder &featureInput(RaggedTensor _featureInput) override {
        Activation::Builder::featureInput(_featureInput);
        return *this;
    }
};

}  // namespace Thor
