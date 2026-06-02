#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Activations/GatedLinearUnitActivation.h"

namespace Thor {

class Glu : public GatedLinearUnitActivation {
   public:
    class Builder;
    Glu() : GatedLinearUnitActivation(GateKind::Sigmoid) {}

    ~Glu() override = default;

    std::shared_ptr<Layer> clone() const override {
        std::shared_ptr<Glu> myClone = std::make_shared<Glu>(*this);
        myClone->id = getUnusedId();
        return myClone;
    }

    std::string getLayerType() const override { return "Glu"; }

    static void deserialize(const nlohmann::json& j, Network* network) {
        if (j.at("version").get<std::string>() != "1.0.0")
            throw std::runtime_error("Unsupported version in Glu::deserialize: " + j["version"].get<std::string>());
        if (j.at("layer_type").get<std::string>() != "glu")
            throw std::runtime_error("Layer type mismatch in Glu::deserialize: " + j.at("layer_type").get<std::string>());

        nlohmann::json input = j["feature_input"].get<nlohmann::json>();
        uint64_t originalTensorId = input.at("id").get<uint64_t>();
        Tensor featureInput = network->getApiTensorByOriginalId(originalTensorId);
        Tensor featureOutput = Tensor::deserialize(j.at("feature_output").get<nlohmann::json>());

        Glu activation;
        activation.featureInput = featureInput;
        activation.featureOutput = featureOutput;
        activation.initialized = true;
        activation.addToNetwork(network);
    }
};

class Glu::Builder : public Activation::Builder {
   public:
    std::shared_ptr<Activation> build() override {
        std::shared_ptr<Glu> activation = std::make_shared<Glu>();
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

    Glu::Builder& network(Network& _network) override {
        Activation::Builder::network(_network);
        return *this;
    }

    Glu::Builder& featureInput(Tensor _featureInput) override {
        Activation::Builder::featureInput(_featureInput);
        return *this;
    }
};

}  // namespace Thor
