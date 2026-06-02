#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Activations/GatedLinearUnitActivation.h"

namespace Thor {

class Swiglu : public GatedLinearUnitActivation {
   public:
    class Builder;
    Swiglu() : GatedLinearUnitActivation(GateKind::Swish) {}

    ~Swiglu() override = default;

    std::shared_ptr<Layer> clone() const override {
        std::shared_ptr<Swiglu> myClone = std::make_shared<Swiglu>(*this);
        myClone->id = getUnusedId();
        return myClone;
    }

    std::string getLayerType() const override { return "Swiglu"; }

    static void deserialize(const nlohmann::json& j, Network* network) {
        if (j.at("version").get<std::string>() != "1.0.0")
            throw std::runtime_error("Unsupported version in Swiglu::deserialize: " + j["version"].get<std::string>());
        if (j.at("layer_type").get<std::string>() != "swiglu")
            throw std::runtime_error("Layer type mismatch in Swiglu::deserialize: " + j.at("layer_type").get<std::string>());

        nlohmann::json input = j["feature_input"].get<nlohmann::json>();
        uint64_t originalTensorId = input.at("id").get<uint64_t>();
        Tensor featureInput = network->getApiTensorByOriginalId(originalTensorId);
        Tensor featureOutput = Tensor::deserialize(j.at("feature_output").get<nlohmann::json>());

        Swiglu activation;
        activation.featureInput = featureInput;
        activation.featureOutput = featureOutput;
        activation.initialized = true;
        activation.addToNetwork(network);
    }
};

class Swiglu::Builder : public Activation::Builder {
   public:
    std::shared_ptr<Activation> build() override {
        std::shared_ptr<Swiglu> activation = std::make_shared<Swiglu>();
        if (_featureInput.has_value()) {
            THOR_THROW_IF_FALSE(_network.has_value());
            activation->featureInput = _featureInput;
            activation->featureOutput = GatedLinearUnitActivation::outputTensorForInput(_featureInput.value());
            activation->initialized = true;
            activation->addToNetwork(_network.value());
        } else {
            activation->initialized = true;
        }
        return activation;
    }

    Swiglu::Builder& network(Network& _network) override {
        Activation::Builder::network(_network);
        return *this;
    }

    Swiglu::Builder& featureInput(Tensor _featureInput) override {
        Activation::Builder::featureInput(_featureInput);
        return *this;
    }
};

}  // namespace Thor
