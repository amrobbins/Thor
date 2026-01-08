#pragma once

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Implementation/Layers/Activation/SoftPlus.h"

namespace Thor {

class SoftPlus : public Activation {
   public:
    class Builder;
    SoftPlus() {}

    virtual ~SoftPlus() {}

    virtual std::shared_ptr<Layer> clone() const {
        std::shared_ptr<SoftPlus> myClone = std::make_shared<SoftPlus>(*this);
        myClone->id = getUnusedId();
        return myClone;
    }

    virtual std::string getLayerType() const { return "SoftPlus"; }

    static void deserialize(const nlohmann::json &j, Network *network) {
        if (j.at("version").get<std::string>() != "1.0.0")
            throw std::runtime_error("Unsupported version in SoftPlus::deserialize: " + j["version"].get<std::string>());
        if (j.at("layer_type").get<std::string>() != "soft_plus")
            throw std::runtime_error("Layer type mismatch in SoftPlus::deserialize: " + j.at("layer_type").get<std::string>());

        nlohmann::json input = j["feature_input"].get<nlohmann::json>();
        uint64_t originalTensorId = input.at("id").get<uint64_t>();
        Tensor featureInput = network->getApiTensorByOriginalId(originalTensorId);

        Tensor featureOutput = Tensor::deserialize(j.at("feature_output").get<nlohmann::json>());

        SoftPlus softPlus;
        softPlus.featureInput = featureInput;
        softPlus.featureOutput = featureOutput;
        softPlus.initialized = true;
        softPlus.addToNetwork(network);
    }

   protected:
    virtual std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                             std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                             std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                             Thor::Tensor connectingApiTensor) const {
        assert(initialized);
        assert(connectingApiTensor == featureInput.get());

        std::shared_ptr<ThorImplementation::SoftPlus> softPlus = std::make_shared<ThorImplementation::SoftPlus>();
        return softPlus;
    }

    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const {
        // feature out and error out
        return batchSize * (featureOutput.get().getTotalSizeInBytes() + featureInput.get().getTotalSizeInBytes());
    }
};

class SoftPlus::Builder : public Activation::Builder {
   public:
    virtual std::shared_ptr<Activation> build() {
        std::shared_ptr<SoftPlus> softPlus = std::make_shared<SoftPlus>();
        if (_featureInput.isPresent()) {
            // Standalone layer support.
            assert(_network.isPresent());
            softPlus->featureInput = _featureInput;
            softPlus->featureOutput = _featureInput.get().clone();
            softPlus->initialized = true;
            softPlus->addToNetwork(_network.get());
        } else {
            // Template activation support
            softPlus->initialized = true;
        }

        return softPlus;
    }

    virtual SoftPlus::Builder &network(Network &_network) {
        Activation::Builder::network(_network);
        return *this;
    }

    virtual SoftPlus::Builder &featureInput(Tensor _featureInput) {
        Activation::Builder::featureInput(_featureInput);
        return *this;
    }
};

}  // namespace Thor
