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

    virtual nlohmann::json serialize(const std::string &storageDir, Stream stream) {
        return nlohmann::json{{"version", "1.0.0"}, {"type", "swish"}};
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

   private:
    Optional<Network *> _network;
    Optional<Tensor> _featureInput;
};

}  // namespace Thor
