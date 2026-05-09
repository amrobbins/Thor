#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Implementation/Layers/Activation/Elu.h"
#include <optional>

namespace Thor {

class Elu : public Activation {
   public:
    class Builder;
    Elu(float alpha) : alpha(alpha) {}

    ~Elu() override {}

    std::shared_ptr<Layer> clone() const override {
        std::shared_ptr<Elu> myClone = std::make_shared<Elu>(*this);
        myClone->id = getUnusedId();
        return myClone;
    }

    ThorImplementation::Expression toExpression(const ThorImplementation::Expression& input) const override {
        return input.elu(alpha);
    }

    std::string getLayerType() const override { return "Elu"; }

    nlohmann::json architectureJson() const override {
        nlohmann::json j = Activation::architectureJson();
        j["alpha"] = alpha;
        return j;
    }

    static void deserialize(const nlohmann::json &j, Network *network) {
        if (j.at("version").get<std::string>() != "1.0.0")
            throw std::runtime_error("Unsupported version in Elu::deserialize: " + j["version"].get<std::string>());
        if (j.at("layer_type").get<std::string>() != "elu")
            throw std::runtime_error("Layer type mismatch in Elu::deserialize: " + j.at("layer_type").get<std::string>());
        float alpha = j.at("alpha").get<float>();

        nlohmann::json input = j["feature_input"].get<nlohmann::json>();
        uint64_t originalTensorId = input.at("id").get<uint64_t>();
        Tensor featureInput = network->getApiTensorByOriginalId(originalTensorId);

        Tensor featureOutput = Tensor::deserialize(j.at("feature_output").get<nlohmann::json>());

        Elu elu(alpha);
        elu.featureInput = featureInput;
        elu.featureOutput = featureOutput;
        elu.initialized = true;
        elu.addToNetwork(network);
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

        std::shared_ptr<ThorImplementation::Elu> elu = std::make_shared<ThorImplementation::Elu>(alpha);
        return elu;
    }

    uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const override {
        // feature out and error out
        return batchSize * (featureOutput.value().getTotalSizeInBytes() + featureInput.value().getTotalSizeInBytes());
    }

    const float alpha;
};

class Elu::Builder : public Activation::Builder {
   public:
    std::shared_ptr<Activation> build() override {
        float alpha = 1.0f;
        if (_alpha.has_value())
            alpha = _alpha.value();

        std::shared_ptr<Elu> elu = std::make_shared<Elu>(alpha);
        if (_featureInput.has_value()) {
            // Standalone layer support.
            THOR_THROW_IF_FALSE(_network.has_value());
            elu->featureInput = _featureInput;
            elu->featureOutput = _featureInput.value().clone();
            elu->initialized = true;
            elu->addToNetwork(_network.value());
        } else {
            // Template activation support
            elu->initialized = true;
        }

        return elu;
    }

    virtual Elu::Builder &alpha(float _alpha) {
        THOR_THROW_IF_FALSE(!this->_alpha.has_value());
        THOR_THROW_IF_FALSE(_alpha >= 0);
        this->_alpha = _alpha;
        return *this;
    }

    Elu::Builder &network(Network &_network) override {
        Activation::Builder::network(_network);
        return *this;
    }

    Elu::Builder &featureInput(Tensor _featureInput) override {
        Activation::Builder::featureInput(_featureInput);
        return *this;
    }

   private:
    std::optional<float> _alpha;
};

}  // namespace Thor
