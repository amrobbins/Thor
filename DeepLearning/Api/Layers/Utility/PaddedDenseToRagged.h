#pragma once

#include "DeepLearning/Api/Layers/MultiConnectionLayer.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"
#include "DeepLearning/Implementation/Layers/Utility/PaddedDenseToRagged.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace Thor {

// Pack a normal padded dense tensor back into canonical ragged storage using
// an existing partition as the sole source of row membership. Only the
// partition input's offsets are consumed; its packed values are not a data
// dependency. Output offsets alias the partition input exactly.
class PaddedDenseToRagged : public MultiConnectionLayer {
   public:
    class Builder;

    PaddedDenseToRagged() = default;
    ~PaddedDenseToRagged() override = default;

    std::shared_ptr<Layer> clone() const override { return std::make_shared<PaddedDenseToRagged>(*this); }
    std::string getLayerType() const override { return "PaddedDenseToRagged"; }

    [[nodiscard]] Tensor getDenseFeatureInput() const { return denseFeatureInput; }
    [[nodiscard]] RaggedTensor getPartitionInput() const { return partitionInput; }
    [[nodiscard]] RaggedTensor getRaggedFeatureOutput() const { return raggedFeatureOutput; }

    std::optional<Tensor> getFeatureInput() const override { return denseFeatureInput; }
    std::optional<Tensor> getFeatureOutput() const override { return raggedFeatureOutput.getValues(); }

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
    Tensor denseFeatureInput;
    RaggedTensor partitionInput;
    RaggedTensor raggedFeatureOutput;
    std::set<uint32_t> connectedInputPortIndices;
    bool emittedFeatureOutputAfterAllInputsConnected = false;

    friend class Builder;
};

class PaddedDenseToRagged::Builder {
   public:
    virtual ~Builder() = default;
    virtual PaddedDenseToRagged build();

    virtual Builder& network(Network& network) {
        if (_network.has_value()) throw std::runtime_error("PaddedDenseToRagged network may only be set once.");
        _network = &network;
        return *this;
    }

    virtual Builder& featureInput(Tensor featureInput) {
        if (_featureInput.has_value()) throw std::runtime_error("PaddedDenseToRagged feature input may only be set once.");
        if (!featureInput.isInitialized() || featureInput.getDimensions().empty()) {
            throw std::invalid_argument("PaddedDenseToRagged feature input must be an initialized padded tensor with shape [width, ...].");
        }
        _featureInput = featureInput;
        return *this;
    }

    virtual Builder& partitionInput(RaggedTensor partitionInput) {
        if (_partitionInput.has_value()) throw std::runtime_error("PaddedDenseToRagged partition input may only be set once.");
        if (!partitionInput.isInitialized()) throw std::invalid_argument("PaddedDenseToRagged partition input must be initialized.");
        _partitionInput = partitionInput;
        return *this;
    }

   private:
    std::optional<Network*> _network;
    std::optional<Tensor> _featureInput;
    std::optional<RaggedTensor> _partitionInput;
};

}  // namespace Thor
