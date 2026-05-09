#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Implementation/Layers/Activation/Swish.h"

namespace Thor {

class Swish : public Activation {
   public:
    class Builder;
    Swish() {}

    ~Swish() override {}

    std::shared_ptr<Layer> clone() const override {
        std::shared_ptr<Swish> myClone = std::make_shared<Swish>(*this);
        myClone->id = getUnusedId();
        return myClone;
    }

    ThorImplementation::Expression toExpression(const ThorImplementation::Expression& input) const override {
        return input.swish();
    }

    std::string getLayerType() const override { return "Swish"; }

    static void deserialize(const nlohmann::json &j, Network *network) {
        if (j.at("version").get<std::string>() != "1.0.0")
            throw std::runtime_error("Unsupported version in Swish::deserialize: " + j["version"].get<std::string>());
        if (j.at("layer_type").get<std::string>() != "swish")
            throw std::runtime_error("Layer type mismatch in Swish::deserialize: " + j.at("layer_type").get<std::string>());

        nlohmann::json input = j["feature_input"].get<nlohmann::json>();
        uint64_t originalTensorId = input.at("id").get<uint64_t>();
        Tensor featureInput = network->getApiTensorByOriginalId(originalTensorId);

        Tensor featureOutput = Tensor::deserialize(j.at("feature_output").get<nlohmann::json>());

        Swish swish;
        swish.featureInput = featureInput;
        swish.featureOutput = featureOutput;
        swish.initialized = true;
        swish.addToNetwork(network);
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

        std::shared_ptr<ThorImplementation::Swish> swish = std::make_shared<ThorImplementation::Swish>();
        return swish;
    }

    uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const override {
        // feature out and error out
        return batchSize * (featureOutput.get().getTotalSizeInBytes() + featureInput.get().getTotalSizeInBytes());
    }
};

class Swish::Builder : public Activation::Builder {
   public:
    std::shared_ptr<Activation> build() override {
        std::shared_ptr<Swish> swish = std::make_shared<Swish>();
        if (_featureInput.isPresent()) {
            // Standalone layer support.
            THOR_THROW_IF_FALSE(_network.isPresent());
            swish->featureInput = _featureInput;
            swish->featureOutput = _featureInput.get().clone();
            swish->initialized = true;
            swish->addToNetwork(_network.get());
        } else {
            // Template activation support
            swish->initialized = true;
        }

        return swish;
    }

    Swish::Builder &network(Network &_network) override {
        Activation::Builder::network(_network);
        return *this;
    }

    Swish::Builder &featureInput(Tensor _featureInput) override {
        Activation::Builder::featureInput(_featureInput);
        return *this;
    }
};

}  // namespace Thor
