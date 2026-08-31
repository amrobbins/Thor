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

// Broadcast one dense value per logical batch row to every active token in the
// corresponding ragged row. Only the partition input's offsets are consumed;
// its packed values are not a data dependency. The output reuses the exact same
// canonical offsets tensor.
class SegmentedBroadcast : public MultiConnectionLayer {
   public:
    class Builder;

    SegmentedBroadcast() = default;
    ~SegmentedBroadcast() override = default;

    std::shared_ptr<Layer> clone() const override { return std::make_shared<SegmentedBroadcast>(*this); }
    std::string getLayerType() const override { return "SegmentedBroadcast"; }

    [[nodiscard]] Tensor getDenseFeatureInput() const { return denseFeatureInput; }
    [[nodiscard]] RaggedTensor getPartitionInput() const { return partitionInput; }
    [[nodiscard]] RaggedTensor getRaggedFeatureOutput() const { return raggedFeatureOutput; }

    std::optional<Tensor> getFeatureInput() const override { return denseFeatureInput; }
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
    Tensor denseFeatureInput;
    RaggedTensor partitionInput;
    RaggedTensor raggedFeatureOutput;
    std::set<uint32_t> connectedInputPortIndices;
    bool emittedFeatureOutputAfterAllInputsConnected = false;

    friend class Builder;
};

class SegmentedBroadcast::Builder {
   public:
    virtual ~Builder() = default;
    virtual SegmentedBroadcast build();

    virtual Builder& network(Network& network) {
        if (_network.has_value()) throw std::runtime_error("SegmentedBroadcast network may only be set once.");
        _network = &network;
        return *this;
    }

    virtual Builder& featureInput(Tensor featureInput) {
        if (_featureInput.has_value()) throw std::runtime_error("SegmentedBroadcast feature input may only be set once.");
        if (!featureInput.isInitialized()) throw std::invalid_argument("SegmentedBroadcast feature input must be initialized.");
        if (featureInput.getDimensions().empty()) {
            throw std::invalid_argument("SegmentedBroadcast dense feature input must have at least one feature dimension.");
        }
        _featureInput = featureInput;
        return *this;
    }

    virtual Builder& partitionInput(RaggedTensor partitionInput) {
        if (_partitionInput.has_value()) throw std::runtime_error("SegmentedBroadcast partition input may only be set once.");
        if (!partitionInput.isInitialized()) throw std::invalid_argument("SegmentedBroadcast partition input must be initialized.");
        _partitionInput = partitionInput;
        return *this;
    }

   private:
    std::optional<Network*> _network;
    std::optional<Tensor> _featureInput;
    std::optional<RaggedTensor> _partitionInput;
};

}  // namespace Thor
