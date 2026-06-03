#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Activations/Activation.h"

namespace Thor {

class Sigmoid : public Activation {
   public:
    class Builder;
    Sigmoid() = default;

    ~Sigmoid() override = default;

    std::shared_ptr<Layer> clone() const override {
        std::shared_ptr<Sigmoid> myClone = std::make_shared<Sigmoid>(*this);
        myClone->id = getUnusedId();
        return myClone;
    }

    ThorImplementation::Expression toExpression(const ThorImplementation::Expression& input) const override {
        return input.sigmoid();
    }

    std::string getLayerType() const override { return "Sigmoid"; }

    static void deserialize(const nlohmann::json &j, Network *network) {
        if (j.at("version").get<std::string>() != "1.0.0")
            throw std::runtime_error("Unsupported version in Sigmoid::deserialize: " + j["version"].get<std::string>());
        if (j.at("layer_type").get<std::string>() != "sigmoid")
            throw std::runtime_error("Layer type mismatch in Sigmoid::deserialize: " + j.at("layer_type").get<std::string>());

        nlohmann::json input = j["feature_input"].get<nlohmann::json>();
        uint64_t originalTensorId = input.at("id").get<uint64_t>();
        Tensor featureInput = network->getApiTensorByOriginalId(originalTensorId);

        Tensor featureOutput = Tensor::deserialize(j.at("feature_output").get<nlohmann::json>());

        Sigmoid sigmoid;
        sigmoid.featureInput = featureInput;
        sigmoid.featureOutput = featureOutput;
        sigmoid.initialized = true;
        sigmoid.addToNetwork(network);
    }

   protected:
    std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                     std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                     std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                     Thor::Tensor connectingApiTensor,
                                                     const bool inferenceOnly) const override {
        (void)drivingLayer;
        (void)drivingApiLayer;
        THOR_THROW_IF_FALSE(initialized);
        THOR_THROW_IF_FALSE(connectingApiTensor == featureInput.value());
        return stampExpressionBackedActivation(placement, connectingApiTensor, inferenceOnly);
    }

    uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const override {
        (void)tensorPlacement;
        return batchSize * (featureOutput.value().getTotalSizeInBytes() + featureInput.value().getTotalSizeInBytes());
    }
};

class Sigmoid::Builder : public Activation::Builder {
   public:
    std::shared_ptr<Activation> build() override {
        std::shared_ptr<Sigmoid> sigmoid = std::make_shared<Sigmoid>();
        if (_featureInput.has_value()) {
            // Standalone layer support.
            THOR_THROW_IF_FALSE(_network.has_value());
            sigmoid->featureInput = _featureInput;
            sigmoid->featureOutput = _featureInput.value().clone();
            sigmoid->initialized = true;
            sigmoid->addToNetwork(_network.value());
        } else {
            // Template activation support.
            sigmoid->initialized = true;
        }
        return sigmoid;
    }

    Sigmoid::Builder &network(Network &_network) override {
        Activation::Builder::network(_network);
        return *this;
    }

    Sigmoid::Builder &featureInput(Tensor _featureInput) override {
        Activation::Builder::featureInput(_featureInput);
        return *this;
    }
};

}  // namespace Thor
