#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Implementation/Layers/Activation/SoftPlus.h"

namespace Thor {

class SoftPlus : public Activation {
   public:
    class Builder;
    SoftPlus() {}

    ~SoftPlus() override {}

    std::shared_ptr<Layer> clone() const override {
        std::shared_ptr<SoftPlus> myClone = std::make_shared<SoftPlus>(*this);
        myClone->id = getUnusedId();
        return myClone;
    }

    ThorImplementation::Expression toExpression(const ThorImplementation::Expression& input) const override {
        return input.softplus();
    }

    std::string getLayerType() const override { return "SoftPlus"; }

    static void deserialize(const nlohmann::json &j, Network *network) {
        if (j.at("version").get<std::string>() != "1.0.0")
            throw std::runtime_error("Unsupported version in SoftPlus::deserialize: " + j["version"].get<std::string>());
        if (j.at("layer_type").get<std::string>() != "soft_plus")
            throw std::runtime_error("Layer type mismatch in SoftPlus::deserialize: " + j.at("layer_type").get<std::string>());

        nlohmann::json input = j["feature_input"].get<nlohmann::json>();
        uint64_t originalTensorId = input.at("id").get<uint64_t>();
        Tensor featureInput = network->getApiTensorByOriginalId(originalTensorId);

        Tensor featureOutput = Tensor::deserialize(j.at("feature_output").get<nlohmann::json>());

        SoftPlus softPlus;
        softPlus.featureInput = featureInput;
        softPlus.featureOutput = featureOutput;
        softPlus.initialized = true;
        softPlus.addToNetwork(network);
    }

   protected:
    std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                     std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                     std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                     Thor::Tensor connectingApiTensor,
                                                     const bool inferenceOnly) const override {
        (void)inferenceOnly;
        THOR_THROW_IF_FALSE(initialized);
        THOR_THROW_IF_FALSE(connectingApiTensor == featureInput.value());

        std::shared_ptr<ThorImplementation::SoftPlus> softPlus = std::make_shared<ThorImplementation::SoftPlus>();
        return softPlus;
    }

    uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const override {
        // feature out and error out
        return batchSize * (featureOutput.value().getTotalSizeInBytes() + featureInput.value().getTotalSizeInBytes());
    }
};

class SoftPlus::Builder : public Activation::Builder {
   public:
    std::shared_ptr<Activation> build() override {
        std::shared_ptr<SoftPlus> softPlus = std::make_shared<SoftPlus>();
        if (_featureInput.has_value()) {
            // Standalone layer support.
            THOR_THROW_IF_FALSE(_network.has_value());
            softPlus->featureInput = _featureInput;
            softPlus->featureOutput = _featureInput.value().clone();
            softPlus->initialized = true;
            softPlus->addToNetwork(_network.value());
        } else {
            // Template activation support
            softPlus->initialized = true;
        }

        return softPlus;
    }

    SoftPlus::Builder &network(Network &_network) override {
        Activation::Builder::network(_network);
        return *this;
    }

    SoftPlus::Builder &featureInput(Tensor _featureInput) override {
        Activation::Builder::featureInput(_featureInput);
        return *this;
    }
};

}  // namespace Thor
