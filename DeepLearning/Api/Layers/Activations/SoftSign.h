#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Implementation/Layers/Activation/SoftSign.h"

namespace Thor {

class SoftSign : public Activation {
   public:
    class Builder;
    SoftSign() {}

    ~SoftSign() override {}

    std::shared_ptr<Layer> clone() const override {
        std::shared_ptr<SoftSign> myClone = std::make_shared<SoftSign>(*this);
        myClone->id = getUnusedId();
        return myClone;
    }

    ThorImplementation::Expression toExpression(const ThorImplementation::Expression& input) const override {
        return input / (input.abs() + ThorImplementation::Expression(1.0));
    }

    std::string getLayerType() const override { return "SoftSign"; }

    static void deserialize(const nlohmann::json &j, Network *network) {
        if (j.at("version").get<std::string>() != "1.0.0")
            throw std::runtime_error("Unsupported version in SoftSign::deserialize: " + j["version"].get<std::string>());
        if (j.at("layer_type").get<std::string>() != "soft_sign")
            throw std::runtime_error("Layer type mismatch in SoftSign::deserialize: " + j.at("layer_type").get<std::string>());

        nlohmann::json input = j["feature_input"].get<nlohmann::json>();
        uint64_t originalTensorId = input.at("id").get<uint64_t>();
        Tensor featureInput = network->getApiTensorByOriginalId(originalTensorId);

        Tensor featureOutput = Tensor::deserialize(j.at("feature_output").get<nlohmann::json>());

        SoftSign softSign;
        softSign.featureInput = featureInput;
        softSign.featureOutput = featureOutput;
        softSign.initialized = true;
        softSign.addToNetwork(network);
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

        std::shared_ptr<ThorImplementation::SoftSign> softSign = std::make_shared<ThorImplementation::SoftSign>();
        return softSign;
    }

    uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const override {
        // feature out and error out
        return batchSize * (featureOutput.value().getTotalSizeInBytes() + featureInput.value().getTotalSizeInBytes());
    }
};

class SoftSign::Builder : public Activation::Builder {
   public:
    std::shared_ptr<Activation> build() override {
        std::shared_ptr<SoftSign> softSign = std::make_shared<SoftSign>();
        if (_featureInput.has_value()) {
            // Standalone layer support.
            THOR_THROW_IF_FALSE(_network.has_value());
            softSign->featureInput = _featureInput;
            softSign->featureOutput = _featureInput.value().clone();
            softSign->initialized = true;
            softSign->addToNetwork(_network.value());
        } else {
            // Template activation support
            softSign->initialized = true;
        }

        return softSign;
    }

    SoftSign::Builder &network(Network &_network) override {
        Activation::Builder::network(_network);
        return *this;
    }

    SoftSign::Builder &featureInput(Tensor _featureInput) override {
        Activation::Builder::featureInput(_featureInput);
        return *this;
    }
};

}  // namespace Thor
