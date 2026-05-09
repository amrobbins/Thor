#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Layer.h"
#include <optional>

// Attach Stub to output tensors that would be dangling and are not wanted as NetworkOutputs.
namespace Thor {

class Stub : public Layer {
   public:
    class Builder;

    Stub();
    ~Stub() override;

    std::vector<Tensor> getOutputsFromInput(Tensor inputTensor) override { return std::vector<Tensor>(); }

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
        (void)inferenceOnly;
        THOR_UNREACHABLE();
    }

    uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const override {
        return 0;
    }

   private:
    Tensor getFeatureOutput();
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
