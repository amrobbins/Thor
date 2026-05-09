#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Implementation/Layers/Activation/Relu.h"

namespace Thor {

class Relu : public Activation {
   public:
    class Builder;
    Relu() {}

    ~Relu() override {}

    std::shared_ptr<Layer> clone() const override {
        std::shared_ptr<Relu> myClone = std::make_shared<Relu>(*this);
        myClone->id = getUnusedId();
        return myClone;
    }

    ThorImplementation::Expression toExpression(const ThorImplementation::Expression& input) const override {
        return input.max(ThorImplementation::Expression(0.0));
    }

    std::string getLayerType() const override { return "Relu"; }

    static void deserialize(const nlohmann::json &j, Network *network) {
        if (j.at("version").get<std::string>() != "1.0.0")
            throw std::runtime_error("Unsupported version in Relu::deserialize: " + j["version"].get<std::string>());
        if (j.at("layer_type").get<std::string>() != "relu")
            throw std::runtime_error("Layer type mismatch in Relu::deserialize: " + j.at("layer_type").get<std::string>());

        nlohmann::json input = j["feature_input"].get<nlohmann::json>();
        uint64_t originalTensorId = input.at("id").get<uint64_t>();
        Tensor featureInput = network->getApiTensorByOriginalId(originalTensorId);

        Tensor featureOutput = Tensor::deserialize(j.at("feature_output").get<nlohmann::json>());

        Relu relu;
        relu.featureInput = featureInput;
        relu.featureOutput = featureOutput;
        relu.initialized = true;
        relu.addToNetwork(network);
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

        std::shared_ptr<ThorImplementation::Relu> relu = std::make_shared<ThorImplementation::Relu>();
        return relu;
    }

    uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const override {
        // feature out and error out
        return batchSize * (featureOutput.value().getTotalSizeInBytes() + featureInput.value().getTotalSizeInBytes());
    }
};

class Relu::Builder : public Activation::Builder {
   public:
    std::shared_ptr<Activation> build() override {
        std::shared_ptr<Relu> relu = std::make_shared<Relu>();
        if (_featureInput.has_value()) {
            // Standalone layer support.
            THOR_THROW_IF_FALSE(_network.has_value());
            relu->featureInput = _featureInput;
            relu->featureOutput = _featureInput.value().clone();
            relu->initialized = true;
            relu->addToNetwork(_network.value());
        } else {
            // Template activation support
            relu->initialized = true;
        }

        return relu;
    }

    Relu::Builder &network(Network &_network) override {
        Activation::Builder::network(_network);
        return *this;
    }

    Relu::Builder &featureInput(Tensor _featureInput) override {
        Activation::Builder::featureInput(_featureInput);
        return *this;
    }
};

}  // namespace Thor
