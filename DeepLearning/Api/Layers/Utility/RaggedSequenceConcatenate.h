#pragma once

#include "DeepLearning/Api/Layers/MultiConnectionLayer.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"
#include "DeepLearning/Implementation/Layers/Utility/RaggedSequenceConcatenate.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace Thor {

// Concatenate canonical rank-1 RaggedTensor inputs along the variable-length
// sequence axis. Unlike Concatenate(RaggedTensor), which joins trailing feature
// dimensions and preserves one exact partition, this layer explicitly produces
// a new canonical row partition Q from independently partitioned inputs.
class RaggedSequenceConcatenate : public MultiConnectionLayer {
   public:
    class Builder;

    RaggedSequenceConcatenate() = default;
    ~RaggedSequenceConcatenate() override = default;

    std::shared_ptr<Layer> clone() const override { return std::make_shared<RaggedSequenceConcatenate>(*this); }
    std::string getLayerType() const override { return "RaggedSequenceConcatenate"; }

    [[nodiscard]] const std::vector<RaggedTensor>& getRaggedFeatureInputs() const { return raggedFeatureInputs; }
    [[nodiscard]] RaggedTensor getRaggedFeatureOutput() const { return raggedFeatureOutput; }

    std::optional<Tensor> getFeatureInput() const override {
        if (featureInputs.empty()) return std::nullopt;
        return featureInputs.front();
    }
    std::optional<Tensor> getFeatureOutput() const override {
        if (featureOutputs.empty()) return std::nullopt;
        return featureOutputs.front();
    }
    Tensor getFeatureOutput(Tensor inputTensor) const override {
        for (const Tensor& input : featureInputs) {
            if (inputTensor == input) return raggedFeatureOutput.getValues();
        }
        throw std::logic_error("Tensor is not an input to this RaggedSequenceConcatenate layer.");
    }

    Tensor getFeatureInput(Tensor outputTensor) const override {
        (void)outputTensor;
        throw std::logic_error("RaggedSequenceConcatenate output cannot be mapped to one unique input tensor.");
    }

    std::vector<Tensor> getOutputsFromInput(Tensor inputTensor) override;
    bool mustConnectAllInputsToDriveOutput() const override { return true; }
    void informThatInputConnectionMade(Tensor inputTensor) override;
    void resetGraphTraversalState() override;
    int getConnectionType(Tensor connectingTensor) const override;

    [[nodiscard]] std::optional<std::string> getInputPortName(const Tensor& inputTensor) const override;
    [[nodiscard]] std::optional<std::string> getOutputPortName(const Tensor& outputTensor) const override;
    [[nodiscard]] bool outputTensorDimensionsIncludeBatch(const Tensor& outputTensor) const override;
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
    std::vector<RaggedTensor> raggedFeatureInputs;
    RaggedTensor raggedFeatureOutput;
    std::vector<Tensor> uniqueOffsetsInputs;
    std::vector<uint32_t> offsetPortForInput;
    std::set<uint32_t> connectedInputPortIndices;
    bool emittedOutputsAfterAllInputsConnected = false;

    static RaggedSequenceConcatenate makeLayer(const std::vector<RaggedTensor>& inputs,
                                               const std::optional<RaggedTensor>& serializedOutput = std::nullopt);

    friend class Builder;
};

class RaggedSequenceConcatenate::Builder {
   public:
    virtual ~Builder() = default;
    virtual RaggedSequenceConcatenate build();

    virtual Builder& network(Network& network) {
        if (_network.has_value()) throw std::runtime_error("RaggedSequenceConcatenate network may only be set once.");
        _network = &network;
        return *this;
    }

    virtual Builder& featureInput(RaggedTensor featureInput) {
        if (!featureInput.isInitialized()) {
            throw std::invalid_argument("RaggedSequenceConcatenate feature inputs must be initialized RaggedTensor objects.");
        }
        _featureInputs.push_back(featureInput);
        return *this;
    }

   private:
    std::optional<Network*> _network;
    std::vector<RaggedTensor> _featureInputs;
};

}  // namespace Thor
