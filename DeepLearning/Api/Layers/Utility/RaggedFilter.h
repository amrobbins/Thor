#pragma once

#include "DeepLearning/Api/Layers/MultiConnectionLayer.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"
#include "DeepLearning/Implementation/Layers/Utility/RaggedFilter.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace Thor {

// Stable row-local filtering for canonical rank-1 ragged tensors. The BOOLEAN
// scalar mask must share the exact input row partition. Retained tokens preserve
// their original order within each row and are compacted into packed output
// values with a newly produced canonical offsets tensor Q.
class RaggedFilter : public MultiConnectionLayer {
   public:
    class Builder;

    RaggedFilter() = default;
    ~RaggedFilter() override = default;

    std::shared_ptr<Layer> clone() const override { return std::make_shared<RaggedFilter>(*this); }
    std::string getLayerType() const override { return "RaggedFilter"; }

    [[nodiscard]] RaggedTensor getRaggedFeatureInput() const { return raggedFeatureInput; }
    [[nodiscard]] RaggedTensor getRaggedMaskInput() const { return raggedMaskInput; }
    [[nodiscard]] RaggedTensor getRaggedFeatureOutput() const { return raggedFeatureOutput; }

    std::optional<Tensor> getFeatureInput() const override { return raggedFeatureInput.getValues(); }
    std::optional<Tensor> getFeatureOutput() const override { return raggedFeatureOutput.getValues(); }

    Tensor getFeatureOutput(Tensor inputTensor) const override {
        if (inputTensor == raggedFeatureInput.getValues() || inputTensor == raggedMaskInput.getValues()) {
            return raggedFeatureOutput.getValues();
        }
        if (inputTensor == raggedFeatureInput.getOffsets()) return raggedFeatureOutput.getOffsets();
        throw std::logic_error("Tensor is not an input to this RaggedFilter layer.");
    }

    Tensor getFeatureInput(Tensor outputTensor) const override {
        if (outputTensor == raggedFeatureOutput.getValues()) return raggedFeatureInput.getValues();
        if (outputTensor == raggedFeatureOutput.getOffsets()) return raggedFeatureInput.getOffsets();
        throw std::logic_error("Tensor is not an output of this RaggedFilter layer.");
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
    RaggedTensor raggedFeatureInput;
    RaggedTensor raggedMaskInput;
    RaggedTensor raggedFeatureOutput;
    std::set<uint32_t> connectedInputPortIndices;
    bool emittedOutputsAfterAllInputsConnected = false;

    static RaggedFilter makeLayer(const RaggedTensor& featureInput,
                                  const RaggedTensor& maskInput,
                                  const std::optional<RaggedTensor>& serializedOutput = std::nullopt);

    friend class Builder;
};

class RaggedFilter::Builder {
   public:
    virtual ~Builder() = default;
    virtual RaggedFilter build();

    virtual Builder& network(Network& network) {
        if (_network.has_value()) throw std::runtime_error("RaggedFilter network may only be set once.");
        _network = &network;
        return *this;
    }

    virtual Builder& featureInput(RaggedTensor featureInput) {
        if (_featureInput.has_value()) throw std::runtime_error("RaggedFilter feature input may only be set once.");
        if (!featureInput.isInitialized()) {
            throw std::invalid_argument("RaggedFilter feature input must be an initialized RaggedTensor.");
        }
        _featureInput = featureInput;
        return *this;
    }

    virtual Builder& maskInput(RaggedTensor maskInput) {
        if (_maskInput.has_value()) throw std::runtime_error("RaggedFilter mask input may only be set once.");
        if (!maskInput.isInitialized()) {
            throw std::invalid_argument("RaggedFilter mask input must be an initialized RaggedTensor.");
        }
        _maskInput = maskInput;
        return *this;
    }

   private:
    std::optional<Network*> _network;
    std::optional<RaggedTensor> _featureInput;
    std::optional<RaggedTensor> _maskInput;
};

}  // namespace Thor
