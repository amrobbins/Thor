#pragma once

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Implementation/Layers/Activation/Softmax.h"

namespace Thor {

class CategoricalCrossEntropy;

class Softmax : public Activation {
   public:
    class Builder;
    Softmax() {}

    virtual ~Softmax() {}

    virtual std::shared_ptr<Layer> clone() const { return std::make_shared<Softmax>(*this); }

    virtual std::string getLayerType() const { return "Softmax"; }

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
    virtual std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                             std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                             std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                             Thor::Tensor connectingApiTensor) const {
        assert(initialized);
        assert(connectingApiTensor == featureInput.get());

        std::shared_ptr<ThorImplementation::Softmax> softmax = std::make_shared<ThorImplementation::Softmax>(backwardComputedExternally);
        return softmax;
    }

    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const {
        // feature out and error out
        return batchSize * (featureOutput.get().getTotalSizeInBytes() + featureInput.get().getTotalSizeInBytes());
    }

    bool backwardComputedExternally;
};

class Softmax::Builder : public Activation::Builder {
   public:
    virtual std::shared_ptr<Layer> build() {
        assert(_network.isPresent());
        assert(_featureInput.isPresent());

        Softmax softmax;
        softmax.featureInput = _featureInput;
        softmax.featureOutput = _featureInput.get().clone();
        if (_backwardComputedExternally.isPresent() && _backwardComputedExternally.get() == true)
            softmax.backwardComputedExternally = true;
        else
            softmax.backwardComputedExternally = false;
        softmax.initialized = true;
        softmax.addToNetwork(_network.get());
        return softmax.clone();
    }

    virtual Softmax::Builder &network(Network &_network) {
        Activation::Builder::network(_network);
        return *this;
    }

    virtual Softmax::Builder &featureInput(Tensor _featureInput) {
        Activation::Builder::featureInput(_featureInput);
        return *this;
    }

    virtual std::shared_ptr<Activation::Builder> clone() { return std::make_shared<Softmax::Builder>(*this); }

   protected:
    void backwardComputedExternally() {
        assert(!_backwardComputedExternally.isPresent());
        _backwardComputedExternally = true;
    }

   private:
    Optional<bool> _backwardComputedExternally;

    friend class Thor::CategoricalCrossEntropy;
};

}  // namespace Thor
