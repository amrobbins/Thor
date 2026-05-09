#pragma once
#include <optional>

#include "DeepLearning/Api/Layers/Layer.h"
#include "test/DeepLearning/Implementation/Layers/Helpers/GradientRivet.h"

// GradientRivet prevents pruning of the backward path gradient tensors that are undriven and would be pruned in a non-test network.
namespace Thor {

class GradientRivet : public Layer {
   public:
    class Builder;

    GradientRivet() = default;
    virtual ~GradientRivet() = default;

    virtual std::shared_ptr<Layer> clone() const { return std::make_shared<GradientRivet>(*this); }

    virtual std::string getLayerType() const { return "GradientRivet"; }

    virtual nlohmann::json serialize(const std::string &storageDir, Stream stream) const;
    static void deserialize(const nlohmann::json &j, Network *network);

   protected:
    std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                     std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                     std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                     Thor::Tensor connectingApiTensor,
                                                     const bool inferenceOnly) const override {
        (void)inferenceOnly;
        assert(initialized);
        assert(connectingApiTensor == getFeatureInput());

        std::shared_ptr<ThorImplementation::GradientRivet> gradientRivet = std::make_shared<ThorImplementation::GradientRivet>();
        return gradientRivet;
    }

    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const {
        return 0;
    }
};

class GradientRivet::Builder {
   public:
    virtual GradientRivet build() {
        assert(_network.has_value());
        assert(_tensor.has_value());

        GradientRivet gradientRivet;
        gradientRivet.featureInput = _tensor;
        gradientRivet.featureOutput = _tensor.value().clone();
        gradientRivet.initialized = true;
        gradientRivet.addToNetwork(_network.value());
        return gradientRivet;
    }

    virtual GradientRivet::Builder &network(Network &_network) {
        assert(!this->_network.has_value());
        this->_network = &_network;
        return *this;
    }

    virtual GradientRivet::Builder &tensor(Tensor _tensor) {
        assert(_tensor.isInitialized());
        this->_tensor = _tensor;
        return *this;
    }

   private:
    std::optional<Network *> _network;
    std::optional<Tensor> _tensor;
};

}  // namespace Thor
