#pragma once

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Implementation/Layers/Activation/Sigmoid.h"

namespace Thor {

class BinaryCrossEntropy;

class Sigmoid : public Activation {
   public:
    class Builder;
    Sigmoid() {}

    virtual ~Sigmoid() {}

    virtual std::shared_ptr<Layer> clone() const { return std::make_shared<Sigmoid>(*this); }

    virtual std::string getLayerType() const { return "Sigmoid"; }

    static void deserialize(const nlohmann::json &j, Network *network) {
        if (j.at("version").get<std::string>() != "1.0.0")
            throw std::runtime_error("Unsupported version in FullyConnected::deserialize: " + j["version"].get<std::string>());
        if (j.at("layer_type").get<std::string>() != "sigmoid")
            throw std::runtime_error("Layer type mismatch in FullyConnected::deserialize: " + j.at("layer_type").get<std::string>());

        nlohmann::json input = j["feature_input"].get<nlohmann::json>();
        uint64_t originalTensorId = input.at("id").get<uint64_t>();
        Tensor featureInput = network->getApiTensorByOriginalId(originalTensorId);

        Tensor featureOutput = Tensor::deserialize(j.at("feature_output").get<nlohmann::json>());

        Sigmoid sigmoid;
        sigmoid.featureInput = featureInput;
        sigmoid.featureOutput = featureOutput;
        sigmoid.initialized = true;
        sigmoid.addToNetwork(network);
    }

   protected:
    virtual std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                             std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                             std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                             Thor::Tensor connectingApiTensor) const {
        assert(initialized);
        assert(connectingApiTensor == featureInput.get());

        std::shared_ptr<ThorImplementation::Sigmoid> sigmoid = std::make_shared<ThorImplementation::Sigmoid>(backwardComputedExternally);
        return sigmoid;
    }

    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const {
        // feature out and error out
        return batchSize * (featureOutput.get().getTotalSizeInBytes() + featureInput.get().getTotalSizeInBytes());
    }

    bool backwardComputedExternally;
};

class Sigmoid::Builder : public Activation::Builder {
   public:
    virtual std::shared_ptr<Layer> build() {
        assert(_network.isPresent());
        assert(_featureInput.isPresent());

        Sigmoid sigmoid;
        sigmoid.featureInput = _featureInput;
        sigmoid.featureOutput = _featureInput.get().clone();
        if (_backwardComputedExternally.isPresent() && _backwardComputedExternally.get() == true)
            sigmoid.backwardComputedExternally = true;
        else
            sigmoid.backwardComputedExternally = false;
        sigmoid.initialized = true;
        sigmoid.addToNetwork(_network.get());
        return sigmoid.clone();
    }

    virtual Sigmoid::Builder &network(Network &_network) {
        Activation::Builder::network(_network);
        return *this;
    }

    virtual Sigmoid::Builder &featureInput(Tensor _featureInput) {
        Activation::Builder::featureInput(_featureInput);
        return *this;
    }

    virtual std::shared_ptr<Activation::Builder> clone() { return std::make_shared<Sigmoid::Builder>(*this); }

   protected:
    void backwardComputedExternally() {
        assert(!_backwardComputedExternally.isPresent());
        _backwardComputedExternally = true;
    }

   private:
    Optional<bool> _backwardComputedExternally;

    friend class BinaryCrossEntropy;
};

}  // namespace Thor
