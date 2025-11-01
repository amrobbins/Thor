#pragma once

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Implementation/Layers/Activation/Gelu.h"

namespace Thor {

class Gelu : public Activation {
   public:
    class Builder;
    Gelu() {}

    virtual ~Gelu() {}

    virtual std::shared_ptr<Layer> clone() const { return std::make_shared<Gelu>(*this); }

    virtual std::string getLayerType() const { return "Gelu"; }

    static void deserialize(const nlohmann::json &j, Network *network) {
        if (j.at("version").get<std::string>() != "1.0.0")
            throw std::runtime_error("Unsupported version in FullyConnected::deserialize: " + j["version"].get<std::string>());
        if (j.at("layer_type").get<std::string>() != "gelu")
            throw std::runtime_error("Layer type mismatch in FullyConnected::deserialize: " + j.at("layer_type").get<std::string>());

        nlohmann::json input = j["feature_input"].get<nlohmann::json>();
        uint64_t originalTensorId = input.at("id").get<uint64_t>();
        Tensor featureInput = network->getApiTensorByOriginalId(originalTensorId);

        Tensor featureOutput = Tensor::deserialize(j.at("feature_output").get<nlohmann::json>());

        Gelu gelu;
        gelu.featureInput = featureInput;
        gelu.featureOutput = featureOutput;
        gelu.initialized = true;
        gelu.addToNetwork(network);
    }

   protected:
    virtual std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                             std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                             std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                             Thor::Tensor connectingApiTensor) const {
        assert(initialized);
        assert(connectingApiTensor == featureInput.get());

        std::shared_ptr<ThorImplementation::Gelu> gelu = std::make_shared<ThorImplementation::Gelu>();
        return gelu;
    }

    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const {
        // feature out and error out
        return batchSize * (featureOutput.get().getTotalSizeInBytes() + featureInput.get().getTotalSizeInBytes());
    }
};

class Gelu::Builder : public Activation::Builder {
   public:
    virtual std::shared_ptr<Layer> build() {
        assert(_network.isPresent());
        assert(_featureInput.isPresent());

        Gelu gelu;
        gelu.featureInput = _featureInput;
        gelu.featureOutput = _featureInput.get().clone();
        gelu.initialized = true;
        gelu.addToNetwork(_network.get());
        return gelu.clone();
    }

    virtual Gelu::Builder &network(Network &_network) {
        Activation::Builder::network(_network);
        return *this;
    }

    virtual Gelu::Builder &featureInput(Tensor _featureInput) {
        Activation::Builder::featureInput(_featureInput);
        return *this;
    }

    virtual std::shared_ptr<Activation::Builder> clone() { return std::make_shared<Gelu::Builder>(*this); }
};

}  // namespace Thor
