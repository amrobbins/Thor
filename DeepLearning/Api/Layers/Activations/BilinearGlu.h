#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Activations/GatedLinearUnitActivation.h"

namespace Thor {

class BilinearGlu : public GatedLinearUnitActivation {
   public:
    class Builder;
    BilinearGlu() : GatedLinearUnitActivation(GateKind::Bilinear) {}

    ~BilinearGlu() override = default;

    std::shared_ptr<Layer> clone() const override {
        std::shared_ptr<BilinearGlu> myClone = std::make_shared<BilinearGlu>(*this);
        myClone->id = getUnusedId();
        return myClone;
    }

    std::string getLayerType() const override { return "BilinearGlu"; }

    static void deserialize(const nlohmann::json& j, Network* network) {
        if (j.at("version").get<std::string>() != "1.0.0")
            throw std::runtime_error("Unsupported version in BilinearGlu::deserialize: " + j["version"].get<std::string>());
        if (j.at("layer_type").get<std::string>() != "bilinear_glu")
            throw std::runtime_error("Layer type mismatch in BilinearGlu::deserialize: " + j.at("layer_type").get<std::string>());

        nlohmann::json input = j["feature_input"].get<nlohmann::json>();
        uint64_t originalTensorId = input.at("id").get<uint64_t>();
        Tensor featureInput = network->getApiTensorByOriginalId(originalTensorId);
        Tensor featureOutput = Tensor::deserialize(j.at("feature_output").get<nlohmann::json>());

        BilinearGlu activation;
        activation.featureInput = featureInput;
        activation.featureOutput = featureOutput;
        activation.initialized = true;
        activation.addToNetwork(network);
    }
};

class BilinearGlu::Builder : public Activation::Builder {
   public:
    std::shared_ptr<Activation> build() override {
        if (_epilogue.has_value() || !_epilogueInputBindings.empty()) {
            throw std::invalid_argument(
                "Standalone gated linear unit activations do not currently support activation epilogues.");
        }
        std::shared_ptr<BilinearGlu> activation = std::make_shared<BilinearGlu>();
        if (_featureInput.has_value()) {
            THOR_THROW_IF_FALSE(_network.has_value());
            applyStandaloneConfiguration(*activation);
            activation->initialized = true;
            activation->addToNetwork(_network.value());
        } else {
            activation->initialized = true;
        }
        return activation;
    }

    BilinearGlu::Builder& network(Network& _network) override {
        Activation::Builder::network(_network);
        return *this;
    }

    BilinearGlu::Builder& featureInput(Tensor _featureInput) override {
        Activation::Builder::featureInput(_featureInput);
        return *this;
    }

    BilinearGlu::Builder& featureInput(RaggedTensor _featureInput) override {
        Activation::Builder::featureInput(_featureInput);
        return *this;
    }
};

}  // namespace Thor
