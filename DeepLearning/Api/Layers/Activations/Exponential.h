#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Implementation/Layers/Activation/Exponential.h"

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

        nlohmann::json input = j["feature_input"].get<nlohmann::json>();
        uint64_t originalTensorId = input.at("id").get<uint64_t>();
        Tensor featureInput = network->getApiTensorByOriginalId(originalTensorId);

        Tensor featureOutput = Tensor::deserialize(j.at("feature_output").get<nlohmann::json>());

        Exponential exponential;
        exponential.featureInput = featureInput;
        exponential.featureOutput = featureOutput;
        exponential.initialized = true;
        exponential.addToNetwork(network);
    }

   protected:
    std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                     std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                     std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                     Thor::Tensor connectingApiTensor,
                                                     const bool inferenceOnly) const override {
        (void)inferenceOnly;
        THOR_THROW_IF_FALSE(initialized);
        THOR_THROW_IF_FALSE(connectingApiTensor == featureInput.get());

        std::shared_ptr<ThorImplementation::Exponential> exponential = std::make_shared<ThorImplementation::Exponential>();
        return exponential;
    }

    uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const override {
        // feature out and error out
        return batchSize * (featureOutput.get().getTotalSizeInBytes() + featureInput.get().getTotalSizeInBytes());
    }
};

class Exponential::Builder : public Activation::Builder {
   public:
    std::shared_ptr<Activation> build() override {
        std::shared_ptr<Exponential> exponential = std::make_shared<Exponential>();
        if (_featureInput.isPresent()) {
            // Standalone layer support.
            THOR_THROW_IF_FALSE(_network.isPresent());
            exponential->featureInput = _featureInput;
            exponential->featureOutput = _featureInput.get().clone();
            exponential->initialized = true;
            exponential->addToNetwork(_network.get());
        } else {
            // Template activation support
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
};

}  // namespace Thor
