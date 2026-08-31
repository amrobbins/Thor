#pragma once

#include "DeepLearning/Api/Layers/MultiConnectionLayer.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"
#include "DeepLearning/Implementation/Layers/Utility/RaggedToPaddedDense.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace Thor {

// Convert a canonical rank-1 RaggedTensor to a normal padded dense tensor.
// The ragged input must carry a finite max_values_per_row bound W. The logical
// API output shape is [W, ...trailing], while the stamped physical shape is
// [B, W, ...trailing]. Inactive/padded positions are filled with paddingValue.
class RaggedToPaddedDense : public MultiConnectionLayer {
   public:
    class Builder;

    RaggedToPaddedDense() = default;
    ~RaggedToPaddedDense() override = default;

    std::shared_ptr<Layer> clone() const override { return std::make_shared<RaggedToPaddedDense>(*this); }
    std::string getLayerType() const override { return "RaggedToPaddedDense"; }

    [[nodiscard]] RaggedTensor getRaggedFeatureInput() const { return raggedFeatureInput; }
    [[nodiscard]] Tensor getPaddedFeatureOutput() const { return featureOutputs.front(); }
    [[nodiscard]] double getPaddingValue() const { return paddingValue; }

    std::optional<Tensor> getFeatureInput() const override { return raggedFeatureInput.getValues(); }
    std::optional<Tensor> getFeatureOutput() const override {
        return featureOutputs.empty() ? std::nullopt : std::optional<Tensor>(featureOutputs.front());
    }

    std::vector<Tensor> getOutputsFromInput(Tensor inputTensor) override;
    bool mustConnectAllInputsToDriveOutput() const override { return true; }
    void informThatInputConnectionMade(Tensor inputTensor) override;
    void resetGraphTraversalState() override;
    int getConnectionType(Tensor connectingTensor) const override;
    [[nodiscard]] std::optional<std::string> getInputPortName(const Tensor& inputTensor) const override;
    [[nodiscard]] std::optional<std::string> getOutputPortName(const Tensor& outputTensor) const override;

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
    double paddingValue = 0.0;
    std::set<uint32_t> connectedInputPortIndices;
    bool emittedFeatureOutputAfterAllInputsConnected = false;

    friend class Builder;
};

class RaggedToPaddedDense::Builder {
   public:
    virtual ~Builder() = default;
    virtual RaggedToPaddedDense build();

    virtual Builder& network(Network& network) {
        if (_network.has_value()) throw std::runtime_error("RaggedToPaddedDense network may only be set once.");
        _network = &network;
        return *this;
    }

    virtual Builder& featureInput(RaggedTensor featureInput) {
        if (_featureInput.has_value()) throw std::runtime_error("RaggedToPaddedDense feature input may only be set once.");
        if (!featureInput.isInitialized()) throw std::invalid_argument("RaggedToPaddedDense feature input must be initialized.");
        _featureInput = featureInput;
        return *this;
    }

    virtual Builder& paddingValue(double paddingValue) {
        _paddingValue = paddingValue;
        return *this;
    }

   private:
    std::optional<Network*> _network;
    std::optional<RaggedTensor> _featureInput;
    double _paddingValue = 0.0;
};

}  // namespace Thor
