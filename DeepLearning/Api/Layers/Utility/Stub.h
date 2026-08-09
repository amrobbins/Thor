#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Implementation/Layers/Utility/Stub.h"
#include <optional>

// Attach Stub to output tensors that would be dangling and are not wanted as NetworkOutputs.
namespace Thor {

class Stub : public Layer {
   public:
    class Builder;

    Stub();
    ~Stub() override;

    std::vector<Tensor> getOutputsFromInput(Tensor inputTensor) override {
        THOR_THROW_IF_FALSE(featureInput.has_value());
        THOR_THROW_IF_FALSE(inputTensor == featureInput.value());
        return {};
    }

    std::vector<Tensor> getAllOutputTensors() const override { return {}; }

    std::shared_ptr<Layer> clone() const override { return std::make_shared<Stub>(*this); }

    std::string getLayerType() const override { return "Stub"; }

    nlohmann::json architectureJson() const override;
    static void deserialize(const nlohmann::json &j, Network *network);

   protected:
    virtual std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement, uint32_t batchSize) const {
        THOR_UNREACHABLE();
    }

    std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                     std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                     std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                     Thor::Tensor connectingApiTensor,
                                                     const bool inferenceOnly) const override {
        (void)placement;
        (void)drivingLayer;
        (void)drivingApiLayer;
        (void)inferenceOnly;
        THOR_THROW_IF_FALSE(initialized);
        THOR_THROW_IF_FALSE(featureInput.has_value());
        THOR_THROW_IF_FALSE(connectingApiTensor == featureInput.value());
        return std::make_shared<ThorImplementation::Stub>();
    }

    uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const override {
        return 0;
    }
};

class Stub::Builder {
   public:
    virtual Stub build() {
        THOR_THROW_IF_FALSE(_network.has_value());
        THOR_THROW_IF_FALSE(_inputTensor.has_value());

        Stub stub;
        stub.featureInput = _inputTensor;
        stub.initialized = true;
        stub.addToNetwork(_network.value());
        return stub;
    }

    virtual Stub::Builder &network(Network &_network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &_network;
        return *this;
    }

    virtual Stub::Builder &inputTensor(Tensor _inputTensor) {
        THOR_THROW_IF_FALSE(_inputTensor.isInitialized());
        this->_inputTensor = _inputTensor;
        return *this;
    }

   private:
    std::optional<Network *> _network;
    std::optional<Tensor> _inputTensor;
};

}  // namespace Thor
