#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Implementation/Layers/Activation/HardSigmoid.h"

namespace Thor {

class HardSigmoid : public Activation {
   public:
    class Builder;
    HardSigmoid() {}

    ~HardSigmoid() override {}

    std::shared_ptr<Layer> clone() const override {
        std::shared_ptr<HardSigmoid> myClone = std::make_shared<HardSigmoid>(*this);
        myClone->id = getUnusedId();
        return myClone;
    }

    ThorImplementation::Expression toExpression(const ThorImplementation::Expression& input) const override {
        const ThorImplementation::Expression zero(0.0);
        const ThorImplementation::Expression one(1.0);
        return ((input * ThorImplementation::Expression(0.2)) + ThorImplementation::Expression(0.5)).min(one).max(zero);
    }

    std::string getLayerType() const override { return "HardSigmoid"; }

    static void deserialize(const nlohmann::json &j, Network *network) {
        if (j.at("version").get<std::string>() != "1.0.0")
            throw std::runtime_error("Unsupported version in HardSigmoid::deserialize: " + j["version"].get<std::string>());
        if (j.at("layer_type").get<std::string>() != "hard_sigmoid")
            throw std::runtime_error("Layer type mismatch in HardSigmoid::deserialize: " + j.at("layer_type").get<std::string>());

        nlohmann::json input = j["feature_input"].get<nlohmann::json>();
        uint64_t originalTensorId = input.at("id").get<uint64_t>();
        Tensor featureInput = network->getApiTensorByOriginalId(originalTensorId);

        Tensor featureOutput = Tensor::deserialize(j.at("feature_output").get<nlohmann::json>());

        HardSigmoid hardSigmoid;
        hardSigmoid.featureInput = featureInput;
        hardSigmoid.featureOutput = featureOutput;
        hardSigmoid.initialized = true;
        hardSigmoid.addToNetwork(network);
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

        std::shared_ptr<ThorImplementation::HardSigmoid> hardSigmoid = std::make_shared<ThorImplementation::HardSigmoid>();
        return hardSigmoid;
    }

    uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const override {
        // feature out and error out
        return batchSize * (featureOutput.value().getTotalSizeInBytes() + featureInput.value().getTotalSizeInBytes());
    }
};

class HardSigmoid::Builder : public Activation::Builder {
   public:
    std::shared_ptr<Activation> build() override {
        std::shared_ptr<HardSigmoid> hardSigmoid = std::make_shared<HardSigmoid>();
        if (_featureInput.has_value()) {
            // Standalone layer support.
            THOR_THROW_IF_FALSE(_network.has_value());
            hardSigmoid->featureInput = _featureInput;
            hardSigmoid->featureOutput = _featureInput.value().clone();
            hardSigmoid->initialized = true;
            hardSigmoid->addToNetwork(_network.value());
        } else {
            // Template activation support
            hardSigmoid->initialized = true;
        }

        return hardSigmoid;
    }

    HardSigmoid::Builder &network(Network &_network) override {
        Activation::Builder::network(_network);
        return *this;
    }

    HardSigmoid::Builder &featureInput(Tensor _featureInput) override {
        Activation::Builder::featureInput(_featureInput);
        return *this;
    }
};

}  // namespace Thor
