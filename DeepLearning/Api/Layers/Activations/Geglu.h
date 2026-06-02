#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Activations/GatedLinearUnitActivation.h"

namespace Thor {

class Geglu : public GatedLinearUnitActivation {
   public:
    class Builder;
    Geglu() : GatedLinearUnitActivation(GateKind::Gelu) {}

    ~Geglu() override = default;

    std::shared_ptr<Layer> clone() const override {
        std::shared_ptr<Geglu> myClone = std::make_shared<Geglu>(*this);
        myClone->id = getUnusedId();
        return myClone;
    }

    std::string getLayerType() const override { return "Geglu"; }

    static void deserialize(const nlohmann::json& j, Network* network) {
        if (j.at("version").get<std::string>() != "1.0.0")
            throw std::runtime_error("Unsupported version in Geglu::deserialize: " + j["version"].get<std::string>());
        if (j.at("layer_type").get<std::string>() != "geglu")
            throw std::runtime_error("Layer type mismatch in Geglu::deserialize: " + j.at("layer_type").get<std::string>());

        nlohmann::json input = j["feature_input"].get<nlohmann::json>();
        uint64_t originalTensorId = input.at("id").get<uint64_t>();
        Tensor featureInput = network->getApiTensorByOriginalId(originalTensorId);
        Tensor featureOutput = Tensor::deserialize(j.at("feature_output").get<nlohmann::json>());

        Geglu activation;
        activation.featureInput = featureInput;
        activation.featureOutput = featureOutput;
        activation.initialized = true;
        activation.addToNetwork(network);
    }
};

class Geglu::Builder : public Activation::Builder {
   public:
    std::shared_ptr<Activation> build() override {
        std::shared_ptr<Geglu> activation = std::make_shared<Geglu>();
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

    Geglu::Builder& network(Network& _network) override {
        Activation::Builder::network(_network);
        return *this;
    }

    Geglu::Builder& featureInput(Tensor _featureInput) override {
        Activation::Builder::featureInput(_featureInput);
        return *this;
    }
};

}  // namespace Thor
