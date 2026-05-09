#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Implementation/Layers/Activation/Softmax.h"

namespace Thor {

class CategoricalCrossEntropy;

class Softmax : public Activation {
   public:
    class Builder;
    Softmax() {}

    ~Softmax() override {}

    std::shared_ptr<Layer> clone() const override {
        std::shared_ptr<Softmax> myClone = std::make_shared<Softmax>(*this);
        myClone->id = getUnusedId();
        return myClone;
    }

    ThorImplementation::Expression toExpression(const ThorImplementation::Expression& input) const override {
        return input.softmax();
    }

    std::string getLayerType() const override { return "Softmax"; }

    static void deserialize(const nlohmann::json &j, Network *network) {
        if (j.at("version").get<std::string>() != "1.0.0")
            throw std::runtime_error("Unsupported version in Softmax::deserialize: " + j["version"].get<std::string>());
        if (j.at("layer_type").get<std::string>() != "softmax")
            throw std::runtime_error("Layer type mismatch in Softmax::deserialize: " + j.at("layer_type").get<std::string>());

        nlohmann::json input = j["feature_input"].get<nlohmann::json>();
        uint64_t originalTensorId = input.at("id").get<uint64_t>();
        Tensor featureInput = network->getApiTensorByOriginalId(originalTensorId);

        Tensor featureOutput = Tensor::deserialize(j.at("feature_output").get<nlohmann::json>());

        Softmax softmax;
        softmax.featureInput = featureInput;
        softmax.featureOutput = featureOutput;
        softmax.initialized = true;
        softmax.addToNetwork(network);
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

        std::shared_ptr<ThorImplementation::Softmax> softmax = std::make_shared<ThorImplementation::Softmax>(backwardComputedExternally);
        return softmax;
    }

    uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const override {
        // feature out and error out
        return batchSize * (featureOutput.get().getTotalSizeInBytes() + featureInput.get().getTotalSizeInBytes());
    }

    bool backwardComputedExternally;
};

class Softmax::Builder : public Activation::Builder {
   public:
    std::shared_ptr<Activation> build() override {
        std::shared_ptr<Softmax> softmax = std::make_shared<Softmax>();
        if (_featureInput.isPresent()) {
            // Standalone layer support.
            THOR_THROW_IF_FALSE(_network.isPresent());
            softmax->featureInput = _featureInput;
            softmax->featureOutput = _featureInput.get().clone();
            softmax->initialized = true;
            softmax->addToNetwork(_network.get());
        } else {
            // Template activation support
            softmax->initialized = true;
        }

        return softmax;
    }

    Softmax::Builder &network(Network &_network) override {
        Activation::Builder::network(_network);
        return *this;
    }

    Softmax::Builder &featureInput(Tensor _featureInput) override {
        Activation::Builder::featureInput(_featureInput);
        return *this;
    }

   protected:
    Softmax::Builder &backwardComputedExternally() {
        THOR_THROW_IF_FALSE(!_backwardComputedExternally.isPresent());
        _backwardComputedExternally = true;
        return *this;
    }

   private:
    Optional<bool> _backwardComputedExternally;

    friend class Thor::CategoricalCrossEntropy;
};

}  // namespace Thor
