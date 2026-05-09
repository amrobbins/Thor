#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Implementation/Layers/Activation/Selu.h"

namespace Thor {

class Selu : public Activation {
   public:
    class Builder;
    Selu() {}

    ~Selu() override {}

    std::shared_ptr<Layer> clone() const override {
        std::shared_ptr<Selu> myClone = std::make_shared<Selu>(*this);
        myClone->id = getUnusedId();
        return myClone;
    }

    ThorImplementation::Expression toExpression(const ThorImplementation::Expression& input) const override {
        return input.selu();
    }

    std::string getLayerType() const override { return "Selu"; }

    static void deserialize(const nlohmann::json &j, Network *network) {
        if (j.at("version").get<std::string>() != "1.0.0")
            throw std::runtime_error("Unsupported version in Selu::deserialize: " + j["version"].get<std::string>());
        if (j.at("layer_type").get<std::string>() != "selu")
            throw std::runtime_error("Layer type mismatch in Selu::deserialize: " + j.at("layer_type").get<std::string>());

        nlohmann::json input = j["feature_input"].get<nlohmann::json>();
        uint64_t originalTensorId = input.at("id").get<uint64_t>();
        Tensor featureInput = network->getApiTensorByOriginalId(originalTensorId);

        Tensor featureOutput = Tensor::deserialize(j.at("feature_output").get<nlohmann::json>());

        Selu selu;
        selu.featureInput = featureInput;
        selu.featureOutput = featureOutput;
        selu.initialized = true;
        selu.addToNetwork(network);
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

        std::shared_ptr<ThorImplementation::Selu> selu = std::make_shared<ThorImplementation::Selu>();
        return selu;
    }

    uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const override {
        // feature out and error out
        return batchSize * (featureOutput.value().getTotalSizeInBytes() + featureInput.value().getTotalSizeInBytes());
    }
};

class Selu::Builder : public Activation::Builder {
   public:
    std::shared_ptr<Activation> build() override {
        std::shared_ptr<Selu> selu = std::make_shared<Selu>();
        if (_featureInput.has_value()) {
            // Standalone layer support.
            THOR_THROW_IF_FALSE(_network.has_value());
            selu->featureInput = _featureInput;
            selu->featureOutput = _featureInput.value().clone();
            selu->initialized = true;
            selu->addToNetwork(_network.value());
        } else {
            // Template activation support
            selu->initialized = true;
        }

        return selu;
    }

    Selu::Builder &network(Network &_network) override {
        Activation::Builder::network(_network);
        return *this;
    }

    Selu::Builder &featureInput(Tensor _featureInput) override {
        Activation::Builder::featureInput(_featureInput);
        return *this;
    }
};

}  // namespace Thor
