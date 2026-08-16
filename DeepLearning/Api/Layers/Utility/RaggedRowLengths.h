#pragma once

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"

#include <cstdint>
#include <memory>
#include <optional>

namespace Thor {

// Materialize the authoritative row lengths of a canonical rank-1 RaggedTensor
// as a dense INT32 tensor with logical shape [1] and physical shape [B, 1].
// This is structural metadata: the layer depends only on the offsets tensor and
// has no differentiable values input.
class RaggedRowLengths : public Layer {
   public:
    class Builder;

    RaggedRowLengths() = default;
    ~RaggedRowLengths() override = default;

    std::shared_ptr<Layer> clone() const override { return std::make_shared<RaggedRowLengths>(*this); }
    std::string getLayerType() const override { return "RaggedRowLengths"; }

    [[nodiscard]] RaggedTensor getRaggedFeatureInput() const { return raggedFeatureInput; }

    [[nodiscard]] uint64_t getOutputTensorBytes(uint32_t batchSize) const override;
    [[nodiscard]] uint64_t getFirstInstanceMemRequirementInBytes(
        uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const override;

    nlohmann::json architectureJson() const override;
    static void deserialize(const nlohmann::json& j, Network* network);

   protected:
    std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                     std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                     std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                     Thor::Tensor connectingApiTensor,
                                                     bool inferenceOnly) const override;

   private:
    RaggedTensor raggedFeatureInput;

    friend class Builder;
};

class RaggedRowLengths::Builder {
   public:
    virtual ~Builder() = default;
    virtual RaggedRowLengths build();

    virtual Builder& network(Network& network) {
        if (_network.has_value()) throw std::runtime_error("RaggedRowLengths network may only be set once.");
        _network = &network;
        return *this;
    }

    virtual Builder& featureInput(RaggedTensor featureInput) {
        if (_featureInput.has_value()) throw std::runtime_error("RaggedRowLengths feature input may only be set once.");
        if (!featureInput.isInitialized()) throw std::invalid_argument("RaggedRowLengths input must be initialized.");
        _featureInput = featureInput;
        return *this;
    }

   private:
    std::optional<Network*> _network;
    std::optional<RaggedTensor> _featureInput;
};

}  // namespace Thor
