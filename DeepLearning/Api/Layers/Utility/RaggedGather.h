#pragma once

#include "DeepLearning/Api/Layers/MultiConnectionLayer.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"
#include "DeepLearning/Implementation/Layers/Utility/RaggedGather.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace Thor {

// Row-local gather for canonical rank-1 ragged tensors. sourceInput owns source
// partition P. indicesInput is a scalar UINT32/UINT64 ragged tensor whose
// partition Q defines the output row lengths. Every active index is interpreted
// relative to its corresponding source row. The output reuses Q exactly; only
// its packed values are newly produced. Duplicate indices are valid and their
// backward contributions accumulate into the same source token.
class RaggedGather : public MultiConnectionLayer {
   public:
    class Builder;

    RaggedGather() = default;
    ~RaggedGather() override = default;

    std::shared_ptr<Layer> clone() const override { return std::make_shared<RaggedGather>(*this); }
    std::string getLayerType() const override { return "RaggedGather"; }

    [[nodiscard]] RaggedTensor getRaggedSourceInput() const { return raggedSourceInput; }
    [[nodiscard]] RaggedTensor getRaggedIndicesInput() const { return raggedIndicesInput; }
    [[nodiscard]] RaggedTensor getRaggedFeatureOutput() const { return raggedFeatureOutput; }

    std::optional<Tensor> getFeatureInput() const override { return raggedSourceInput.getValues(); }
    std::optional<Tensor> getFeatureOutput() const override { return raggedFeatureOutput.getValues(); }

    Tensor getFeatureOutput(Tensor inputTensor) const override {
        (void)getConnectionType(inputTensor);
        return raggedFeatureOutput.getValues();
    }

    Tensor getFeatureInput(Tensor outputTensor) const override {
        if (outputTensor == raggedFeatureOutput.getValues()) return raggedSourceInput.getValues();
        throw std::logic_error("Tensor is not an output of this RaggedGather layer.");
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
    RaggedTensor raggedSourceInput;
    RaggedTensor raggedIndicesInput;
    RaggedTensor raggedFeatureOutput;
    bool sharedOffsets = false;
    uint32_t indicesOffsetsInputPort = 3;
    std::set<uint32_t> connectedInputPortIndices;
    bool emittedOutputsAfterAllInputsConnected = false;

    static RaggedGather makeLayer(const RaggedTensor& sourceInput,
                                  const RaggedTensor& indicesInput,
                                  const std::optional<RaggedTensor>& serializedOutput = std::nullopt);

    friend class Builder;
};

class RaggedGather::Builder {
   public:
    virtual ~Builder() = default;
    virtual RaggedGather build();

    virtual Builder& network(Network& network) {
        if (_network.has_value()) throw std::runtime_error("RaggedGather network may only be set once.");
        _network = &network;
        return *this;
    }

    virtual Builder& sourceInput(RaggedTensor sourceInput) {
        if (_sourceInput.has_value()) throw std::runtime_error("RaggedGather source input may only be set once.");
        if (!sourceInput.isInitialized()) {
            throw std::invalid_argument("RaggedGather source input must be an initialized RaggedTensor.");
        }
        _sourceInput = sourceInput;
        return *this;
    }

    virtual Builder& indicesInput(RaggedTensor indicesInput) {
        if (_indicesInput.has_value()) throw std::runtime_error("RaggedGather indices input may only be set once.");
        if (!indicesInput.isInitialized()) {
            throw std::invalid_argument("RaggedGather indices input must be an initialized RaggedTensor.");
        }
        _indicesInput = indicesInput;
        return *this;
    }

   private:
    std::optional<Network*> _network;
    std::optional<RaggedTensor> _sourceInput;
    std::optional<RaggedTensor> _indicesInput;
};

}  // namespace Thor
