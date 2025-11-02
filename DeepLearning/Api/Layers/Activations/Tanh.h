#pragma once

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Implementation/Layers/Activation/Tanh.h"

namespace Thor {

class Tanh : public Activation {
   public:
    class Builder;
    Tanh() {}

    virtual ~Tanh() {}

    virtual std::shared_ptr<Layer> clone() const { return std::make_shared<Tanh>(*this); }

    virtual std::string getLayerType() const { return "Tanh"; }

    static void deserialize(const nlohmann::json &j, Network *network) {
        if (j.at("version").get<std::string>() != "1.0.0")
            throw std::runtime_error("Unsupported version in Tanh::deserialize: " + j["version"].get<std::string>());
        if (j.at("layer_type").get<std::string>() != "tanh")
            throw std::runtime_error("Layer type mismatch in Tanh::deserialize: " + j.at("layer_type").get<std::string>());

        nlohmann::json input = j["feature_input"].get<nlohmann::json>();
        uint64_t originalTensorId = input.at("id").get<uint64_t>();
        Tensor featureInput = network->getApiTensorByOriginalId(originalTensorId);

        Tensor featureOutput = Tensor::deserialize(j.at("feature_output").get<nlohmann::json>());

        Tanh tanh;
        tanh.featureInput = featureInput;
        tanh.featureOutput = featureOutput;
        tanh.initialized = true;
        tanh.addToNetwork(network);
    }

   protected:
    virtual std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                             std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                             std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                             Thor::Tensor connectingApiTensor) const {
        assert(initialized);
        assert(connectingApiTensor == featureInput.get());

        std::shared_ptr<ThorImplementation::Tanh> tanh = std::make_shared<ThorImplementation::Tanh>();
        return tanh;
    }

    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const {
        // feature out and error out
        return batchSize * (featureOutput.get().getTotalSizeInBytes() + featureInput.get().getTotalSizeInBytes());
    }
};

class Tanh::Builder : public Activation::Builder {
   public:
    virtual std::shared_ptr<Layer> build() {
        assert(_network.isPresent());
        assert(_featureInput.isPresent());

        Tanh tanh;
        tanh.featureInput = _featureInput;
        tanh.featureOutput = _featureInput.get().clone();
        tanh.initialized = true;
        tanh.addToNetwork(_network.get());
        return tanh.clone();
    }

    virtual Tanh::Builder &network(Network &_network) {
        Activation::Builder::network(_network);
        return *this;
    }

    virtual Tanh::Builder &featureInput(Tensor _featureInput) {
        Activation::Builder::featureInput(_featureInput);
        return *this;
    }

    virtual std::shared_ptr<Activation::Builder> clone() { return std::make_shared<Tanh::Builder>(*this); }
};

}  // namespace Thor
