#pragma once

#include "DeepLearning/Api/Layers/MultiConnectionLayer.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"
#include "DeepLearning/Implementation/Layers/Utility/RaggedSequenceSlice.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace Thor {

// Slice every logical row of a canonical rank-1 RaggedTensor along its
// variable-length sequence axis. The fixed [start, start + length) window is
// clipped independently to each row, compacted into new packed values, and a
// new canonical offsets tensor Q is explicitly produced.
class RaggedSequenceSlice : public MultiConnectionLayer {
   public:
    class Builder;

    RaggedSequenceSlice() = default;
    ~RaggedSequenceSlice() override = default;

    std::shared_ptr<Layer> clone() const override { return std::make_shared<RaggedSequenceSlice>(*this); }
    std::string getLayerType() const override { return "RaggedSequenceSlice"; }

    [[nodiscard]] RaggedTensor getRaggedFeatureInput() const { return raggedFeatureInput; }
    [[nodiscard]] RaggedTensor getRaggedFeatureOutput() const { return raggedFeatureOutput; }
    [[nodiscard]] uint64_t getStart() const { return start; }
    [[nodiscard]] uint64_t getLength() const { return length; }

    std::optional<Tensor> getFeatureInput() const override { return raggedFeatureInput.getValues(); }
    std::optional<Tensor> getFeatureOutput() const override { return raggedFeatureOutput.getValues(); }

    Tensor getFeatureOutput(Tensor inputTensor) const override {
        if (inputTensor == raggedFeatureInput.getValues()) return raggedFeatureOutput.getValues();
        if (inputTensor == raggedFeatureInput.getOffsets()) return raggedFeatureOutput.getOffsets();
        throw std::logic_error("Tensor is not an input to this RaggedSequenceSlice layer.");
    }

    Tensor getFeatureInput(Tensor outputTensor) const override {
        if (outputTensor == raggedFeatureOutput.getValues()) return raggedFeatureInput.getValues();
        if (outputTensor == raggedFeatureOutput.getOffsets()) return raggedFeatureInput.getOffsets();
        throw std::logic_error("Tensor is not an output of this RaggedSequenceSlice layer.");
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
    uint64_t start = 0;
    uint64_t length = 0;
    RaggedTensor raggedFeatureInput;
    RaggedTensor raggedFeatureOutput;
    std::set<uint32_t> connectedInputPortIndices;
    bool emittedOutputsAfterAllInputsConnected = false;

    static RaggedSequenceSlice makeLayer(const RaggedTensor& input,
                                         uint64_t start,
                                         uint64_t length,
                                         const std::optional<RaggedTensor>& serializedOutput = std::nullopt);

    friend class Builder;
};

class RaggedSequenceSlice::Builder {
   public:
    virtual ~Builder() = default;
    virtual RaggedSequenceSlice build();

    virtual Builder& network(Network& network) {
        if (_network.has_value()) throw std::runtime_error("RaggedSequenceSlice network may only be set once.");
        _network = &network;
        return *this;
    }

    virtual Builder& featureInput(RaggedTensor featureInput) {
        if (_featureInput.has_value()) throw std::runtime_error("RaggedSequenceSlice feature input may only be set once.");
        if (!featureInput.isInitialized()) {
            throw std::invalid_argument("RaggedSequenceSlice feature input must be an initialized RaggedTensor.");
        }
        _featureInput = featureInput;
        return *this;
    }

    virtual Builder& start(uint64_t start) {
        if (_start.has_value()) throw std::runtime_error("RaggedSequenceSlice start may only be set once.");
        _start = start;
        return *this;
    }

    virtual Builder& length(uint64_t length) {
        if (_length.has_value()) throw std::runtime_error("RaggedSequenceSlice length may only be set once.");
        _length = length;
        return *this;
    }

   private:
    std::optional<Network*> _network;
    std::optional<RaggedTensor> _featureInput;
    std::optional<uint64_t> _start;
    std::optional<uint64_t> _length;
};

}  // namespace Thor
