#pragma once

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Implementation/Layers/Activation/Swish.h"

namespace Thor {

class Swish : public Activation {
   public:
    class Builder;
    Swish() {}

    virtual ~Swish() {}

    virtual std::shared_ptr<Layer> clone() const { return std::make_shared<Swish>(*this); }

    virtual std::string getLayerType() const { return "Swish"; }

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
    virtual std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                             std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                             std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                             Thor::Tensor connectingApiTensor) const {
        assert(initialized);
        assert(connectingApiTensor == featureInput.get());

        std::shared_ptr<ThorImplementation::Swish> swish = std::make_shared<ThorImplementation::Swish>();
        return swish;
    }

    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const {
        // feature out and error out
        return batchSize * (featureOutput.get().getTotalSizeInBytes() + featureInput.get().getTotalSizeInBytes());
    }
};

class Swish::Builder : public Activation::Builder {
   public:
    virtual std::shared_ptr<Layer> build() {
        assert(_network.isPresent());
        assert(_featureInput.isPresent());

        Swish swish;
        swish.featureInput = _featureInput;
        swish.featureOutput = _featureInput.get().clone();
        swish.initialized = true;
        swish.addToNetwork(_network.get());
        return swish.clone();
    }

    virtual Swish::Builder &network(Network &_network) {
        Activation::Builder::network(_network);
        return *this;
    }

    virtual Swish::Builder &featureInput(Tensor _featureInput) {
        Activation::Builder::featureInput(_featureInput);
        return *this;
    }

    virtual std::shared_ptr<Activation::Builder> clone() { return std::make_shared<Swish::Builder>(*this); }
};

}  // namespace Thor
