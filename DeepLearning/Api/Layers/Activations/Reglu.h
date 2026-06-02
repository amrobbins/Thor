#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Activations/GatedLinearUnitActivation.h"

namespace Thor {

class Reglu : public GatedLinearUnitActivation {
   public:
    class Builder;
    Reglu() : GatedLinearUnitActivation(GateKind::Relu) {}

    ~Reglu() override = default;

    std::shared_ptr<Layer> clone() const override {
        std::shared_ptr<Reglu> myClone = std::make_shared<Reglu>(*this);
        myClone->id = getUnusedId();
        return myClone;
    }

    std::string getLayerType() const override { return "Reglu"; }

    static void deserialize(const nlohmann::json& j, Network* network) {
        if (j.at("version").get<std::string>() != "1.0.0")
            throw std::runtime_error("Unsupported version in Reglu::deserialize: " + j["version"].get<std::string>());
        if (j.at("layer_type").get<std::string>() != "reglu")
            throw std::runtime_error("Layer type mismatch in Reglu::deserialize: " + j.at("layer_type").get<std::string>());

        nlohmann::json input = j["feature_input"].get<nlohmann::json>();
        uint64_t originalTensorId = input.at("id").get<uint64_t>();
        Tensor featureInput = network->getApiTensorByOriginalId(originalTensorId);
        Tensor featureOutput = Tensor::deserialize(j.at("feature_output").get<nlohmann::json>());

        Reglu activation;
        activation.featureInput = featureInput;
        activation.featureOutput = featureOutput;
        activation.initialized = true;
        activation.addToNetwork(network);
    }
};

class Reglu::Builder : public Activation::Builder {
   public:
    std::shared_ptr<Activation> build() override {
        std::shared_ptr<Reglu> activation = std::make_shared<Reglu>();
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

    Reglu::Builder& network(Network& _network) override {
        Activation::Builder::network(_network);
        return *this;
    }

    Reglu::Builder& featureInput(Tensor _featureInput) override {
        Activation::Builder::featureInput(_featureInput);
        return *this;
    }
};

}  // namespace Thor
