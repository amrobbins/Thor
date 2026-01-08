#pragma once

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Implementation/Layers/Activation/Exponential.h"

namespace Thor {

class Exponential : public Activation {
   public:
    class Builder;
    Exponential() {}

    virtual ~Exponential() {}

    virtual std::shared_ptr<Layer> clone() const {
        std::shared_ptr<Exponential> myClone = std::make_shared<Exponential>(*this);
        myClone->id = getUnusedId();
        return myClone;
    }

    virtual std::string getLayerType() const { return "Exponential"; }

    static void deserialize(const nlohmann::json &j, Network *network) {
        if (j.at("version").get<std::string>() != "1.0.0")
            throw std::runtime_error("Unsupported version in Exponential::deserialize: " + j["version"].get<std::string>());
        if (j.at("layer_type").get<std::string>() != "exponential")
            throw std::runtime_error("Layer type mismatch in Exponential::deserialize: " + j.at("layer_type").get<std::string>());

        nlohmann::json input = j["feature_input"].get<nlohmann::json>();
        uint64_t originalTensorId = input.at("id").get<uint64_t>();
        Tensor featureInput = network->getApiTensorByOriginalId(originalTensorId);

        Tensor featureOutput = Tensor::deserialize(j.at("feature_output").get<nlohmann::json>());

        Exponential exponential;
        exponential.featureInput = featureInput;
        exponential.featureOutput = featureOutput;
        exponential.initialized = true;
        exponential.addToNetwork(network);
    }

   protected:
    virtual std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                             std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                             std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                             Thor::Tensor connectingApiTensor) const {
        assert(initialized);
        assert(connectingApiTensor == featureInput.get());

        std::shared_ptr<ThorImplementation::Exponential> exponential = std::make_shared<ThorImplementation::Exponential>();
        return exponential;
    }

    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const {
        // feature out and error out
        return batchSize * (featureOutput.get().getTotalSizeInBytes() + featureInput.get().getTotalSizeInBytes());
    }
};

class Exponential::Builder : public Activation::Builder {
   public:
    virtual std::shared_ptr<Activation> build() {
        std::shared_ptr<Exponential> exponential = std::make_shared<Exponential>();
        if (_featureInput.isPresent()) {
            // Standalone layer support.
            assert(_network.isPresent());
            exponential->featureInput = _featureInput;
            exponential->featureOutput = _featureInput.get().clone();
            exponential->initialized = true;
            exponential->addToNetwork(_network.get());
        } else {
            // Template activation support
            exponential->initialized = true;
        }

        return exponential;
    }

    virtual Exponential::Builder &network(Network &_network) {
        Activation::Builder::network(_network);
        return *this;
    }

    virtual Exponential::Builder &featureInput(Tensor _featureInput) {
        Activation::Builder::featureInput(_featureInput);
        return *this;
    }
};

}  // namespace Thor
