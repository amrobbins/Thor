#pragma once

#include "DeepLearning/Api/Layers/MultiConnectionLayer.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"

#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>

namespace Thor {

// Log-softmax across the active tokens of each ragged row, independently for
// every trailing component. The canonical row partition is preserved exactly.
class SegmentedLogSoftmax : public MultiConnectionLayer {
   public:
    class Builder;

    SegmentedLogSoftmax() = default;
    ~SegmentedLogSoftmax() override = default;

    std::shared_ptr<Layer> clone() const override { return std::make_shared<SegmentedLogSoftmax>(*this); }
    std::string getLayerType() const override { return "SegmentedLogSoftmax"; }

    [[nodiscard]] RaggedTensor getRaggedFeatureInput() const { return raggedFeatureInput; }
    [[nodiscard]] RaggedTensor getRaggedFeatureOutput() const { return raggedFeatureOutput; }

    std::optional<Tensor> getFeatureInput() const override {
        if (featureInputs.empty()) return std::nullopt;
        return featureInputs.front();
    }
    std::optional<Tensor> getFeatureOutput() const override {
        if (featureOutputs.empty()) return std::nullopt;
        return featureOutputs.front();
    }

    std::vector<Tensor> getOutputsFromInput(Tensor inputTensor) override;
    bool mustConnectAllInputsToDriveOutput() const override { return true; }
    void informThatInputConnectionMade(Tensor inputTensor) override;
    void resetGraphTraversalState() override;
    int getConnectionType(Tensor connectingTensor) const override;

    uint64_t getOutputTensorBytes(uint32_t batchSize) const override;
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
    RaggedTensor raggedFeatureOutput;
    std::set<uint32_t> connectedInputPortIndices;
    bool emittedFeatureOutputAfterAllInputsConnected = false;

    friend class Builder;
};

class SegmentedLogSoftmax::Builder {
   public:
    virtual ~Builder() = default;
    virtual SegmentedLogSoftmax build();

    virtual Builder& network(Network& network) {
        if (_network.has_value()) throw std::runtime_error("SegmentedLogSoftmax network may only be set once.");
        _network = &network;
        return *this;
    }

    virtual Builder& featureInput(RaggedTensor featureInput) {
        if (_featureInput.has_value()) throw std::runtime_error("SegmentedLogSoftmax feature input may only be set once.");
        if (!featureInput.isInitialized()) throw std::invalid_argument("SegmentedLogSoftmax feature input must be initialized.");
        _featureInput = featureInput;
        return *this;
    }

   private:
    std::optional<Network*> _network;
    std::optional<RaggedTensor> _featureInput;
};

}  // namespace Thor
