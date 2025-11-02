#pragma once

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Implementation/Layers/Activation/Elu.h"

namespace Thor {

class Elu : public Activation {
   public:
    class Builder;
    Elu() {}

    virtual ~Elu() {}

    virtual std::shared_ptr<Layer> clone() const { return std::make_shared<Elu>(*this); }

    virtual std::string getLayerType() const { return "Elu"; }

    virtual nlohmann::json serialize(const std::string &storageDir, Stream stream) {
        assert(initialized);
        assert(featureInput.isPresent());
        assert(featureOutput.isPresent());

        nlohmann::json j;
        j["factory"] = Layer::Factory::Activation.value();
        j["version"] = getLayerVersion();
        j["layer_type"] = to_snake_case(getLayerType());
        j["alpha"] = alpha;
        j["feature_input"] = featureInput.get().serialize();
        j["feature_output"] = featureOutput.get().serialize();
        return j;
    }

    static void deserialize(const nlohmann::json &j, Network *network) {
        if (j.at("version").get<std::string>() != "1.0.0")
            throw std::runtime_error("Unsupported version in Elu::deserialize: " + j["version"].get<std::string>());
        if (j.at("layer_type").get<std::string>() != "elu")
            throw std::runtime_error("Layer type mismatch in Elu::deserialize: " + j.at("layer_type").get<std::string>());
        float alpha = j.at("alpha").get<float>();

        nlohmann::json input = j["feature_input"].get<nlohmann::json>();
        uint64_t originalTensorId = input.at("id").get<uint64_t>();
        Tensor featureInput = network->getApiTensorByOriginalId(originalTensorId);

        Tensor featureOutput = Tensor::deserialize(j.at("feature_output").get<nlohmann::json>());

        Elu elu;
        elu.alpha = alpha;
        elu.featureInput = featureInput;
        elu.featureOutput = featureOutput;
        elu.initialized = true;
        elu.addToNetwork(network);
    }

   protected:
    virtual std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                             std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                             std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                             Thor::Tensor connectingApiTensor) const {
        assert(initialized);
        assert(connectingApiTensor == featureInput.get());

        std::shared_ptr<ThorImplementation::Elu> elu = std::make_shared<ThorImplementation::Elu>(alpha);
        return elu;
    }

    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const {
        // feature out and error out
        return batchSize * (featureOutput.get().getTotalSizeInBytes() + featureInput.get().getTotalSizeInBytes());
    }

    float alpha;
};

class Elu::Builder : public Activation::Builder {
   public:
    virtual std::shared_ptr<Layer> build() {
        assert(_network.isPresent());
        assert(_featureInput.isPresent());

        Elu elu;
        elu.featureInput = _featureInput;
        elu.featureOutput = _featureInput.get().clone();
        if (_alpha.isPresent())
            elu.alpha = _alpha;
        else
            elu.alpha = 1.0f;
        elu.initialized = true;
        elu.addToNetwork(_network.get());
        return elu.clone();
    }

    virtual Elu::Builder &network(Network &_network) {
        Activation::Builder::network(_network);
        return *this;
    }

    virtual Elu::Builder &featureInput(Tensor _featureInput) {
        Activation::Builder::featureInput(_featureInput);
        return *this;
    }

    virtual Elu::Builder &alpha(float _alpha) {
        assert(!this->_alpha.isPresent());
        assert(_alpha >= 0);
        this->_alpha = _alpha;
        return *this;
    }

    virtual std::shared_ptr<Activation::Builder> clone() { return std::make_shared<Elu::Builder>(*this); }

   private:
    Optional<float> _alpha;
};

}  // namespace Thor
