#pragma once

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Implementation/Layers/Activation/SoftSign.h"

namespace Thor {

class SoftSign : public Activation {
   public:
    class Builder;
    SoftSign() {}

    virtual ~SoftSign() {}

    virtual std::shared_ptr<Layer> clone() const { return std::make_shared<SoftSign>(*this); }

    virtual std::string getLayerType() const { return "SoftSign"; }

    virtual nlohmann::json serialize(const std::string &storageDir, Stream stream) {
        return nlohmann::json{{"version", "1.0.0"}, {"type", "soft_sign"}};
    }

   protected:
    virtual std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                             std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                             std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                             Thor::Tensor connectingApiTensor) const {
        assert(initialized);
        assert(connectingApiTensor == featureInput.get());

        std::shared_ptr<ThorImplementation::SoftSign> softSign = std::make_shared<ThorImplementation::SoftSign>();
        return softSign;
    }

    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const {
        // feature out and error out
        return batchSize * (featureOutput.get().getTotalSizeInBytes() + featureInput.get().getTotalSizeInBytes());
    }
};

class SoftSign::Builder : public Activation::Builder {
   public:
    virtual std::shared_ptr<Layer> build() {
        assert(_network.isPresent());
        assert(_featureInput.isPresent());

        SoftSign softSign;
        softSign.featureInput = _featureInput;
        softSign.featureOutput = _featureInput.get().clone();
        softSign.initialized = true;
        softSign.addToNetwork(_network.get());
        return softSign.clone();
    }

    virtual SoftSign::Builder &network(Network &_network) {
        Activation::Builder::network(_network);
        return *this;
    }

    virtual SoftSign::Builder &featureInput(Tensor _featureInput) {
        Activation::Builder::featureInput(_featureInput);
        return *this;
    }

    virtual std::shared_ptr<Activation::Builder> clone() { return std::make_shared<SoftSign::Builder>(*this); }

   private:
    Optional<Network *> _network;
    Optional<Tensor> _featureInput;
};

}  // namespace Thor
