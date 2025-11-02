#pragma once

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Implementation/Layers/Activation/HardSigmoid.h"

namespace Thor {

class HardSigmoid : public Activation {
   public:
    class Builder;
    HardSigmoid() {}

    virtual ~HardSigmoid() {}

    virtual std::shared_ptr<Layer> clone() const { return std::make_shared<HardSigmoid>(*this); }

    virtual std::string getLayerType() const { return "HardSigmoid"; }

    static void deserialize(const nlohmann::json &j, Network *network) {
        if (j.at("version").get<std::string>() != "1.0.0")
            throw std::runtime_error("Unsupported version in HardSigmoid::deserialize: " + j["version"].get<std::string>());
        if (j.at("layer_type").get<std::string>() != "hard_sigmoid")
            throw std::runtime_error("Layer type mismatch in HardSigmoid::deserialize: " + j.at("layer_type").get<std::string>());

        nlohmann::json input = j["feature_input"].get<nlohmann::json>();
        uint64_t originalTensorId = input.at("id").get<uint64_t>();
        Tensor featureInput = network->getApiTensorByOriginalId(originalTensorId);

        Tensor featureOutput = Tensor::deserialize(j.at("feature_output").get<nlohmann::json>());

        HardSigmoid hardSigmoid;
        hardSigmoid.featureInput = featureInput;
        hardSigmoid.featureOutput = featureOutput;
        hardSigmoid.initialized = true;
        hardSigmoid.addToNetwork(network);
    }

   protected:
    virtual std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                             std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                             std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                             Thor::Tensor connectingApiTensor) const {
        assert(initialized);
        assert(connectingApiTensor == featureInput.get());

        std::shared_ptr<ThorImplementation::HardSigmoid> hardSigmoid = std::make_shared<ThorImplementation::HardSigmoid>();
        return hardSigmoid;
    }

    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const {
        // feature out and error out
        return batchSize * (featureOutput.get().getTotalSizeInBytes() + featureInput.get().getTotalSizeInBytes());
    }
};

class HardSigmoid::Builder : public Activation::Builder {
   public:
    virtual std::shared_ptr<Layer> build() {
        assert(_network.isPresent());
        assert(_featureInput.isPresent());

        HardSigmoid hardSigmoid;
        hardSigmoid.featureInput = _featureInput;
        hardSigmoid.featureOutput = _featureInput.get().clone();
        hardSigmoid.initialized = true;
        hardSigmoid.addToNetwork(_network.get());
        return hardSigmoid.clone();
    }

    virtual HardSigmoid::Builder &network(Network &_network) {
        Activation::Builder::network(_network);
        return *this;
    }

    virtual HardSigmoid::Builder &featureInput(Tensor _featureInput) {
        Activation::Builder::featureInput(_featureInput);
        return *this;
    }

    virtual std::shared_ptr<Activation::Builder> clone() { return std::make_shared<HardSigmoid::Builder>(*this); }
};

}  // namespace Thor
