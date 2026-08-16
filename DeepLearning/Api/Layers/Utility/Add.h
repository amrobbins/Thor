#pragma once

#include "DeepLearning/Api/Layers/MultiConnectionLayer.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <vector>

namespace Thor {

// Elementwise addition for dense or canonical rank-1 ragged tensors. The
// ragged form requires both operands to share the exact same row-partition
// offsets tensor; the output preserves that partition.
class Add : public MultiConnectionLayer {
   public:
    class Builder;

    Add() = default;
    ~Add() override = default;

    std::shared_ptr<Layer> clone() const override { return std::make_shared<Add>(*this); }
    std::string getLayerType() const override { return "Add"; }

    [[nodiscard]] bool getUseRagged() const { return raggedLeft.has_value(); }
    [[nodiscard]] std::optional<RaggedTensor> getRaggedFeatureOutput() const { return raggedOutput; }

    std::vector<Tensor> getOutputsFromInput(Tensor inputTensor) override;
    bool mustConnectAllInputsToDriveOutput() const override { return true; }
    void informThatInputConnectionMade(Tensor inputTensor) override;
    void resetGraphTraversalState() override;
    int getConnectionType(Tensor connectingTensor) const override;

    [[nodiscard]] uint64_t getOutputTensorBytes(uint32_t batchSize) const override;

    nlohmann::json architectureJson() const override;
    static void deserialize(const nlohmann::json& j, Network* network);

   protected:
    std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                     std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                     std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                     Thor::Tensor connectingApiTensor,
                                                     bool inferenceOnly) const override;

    uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize,
                                                   ThorImplementation::TensorPlacement tensorPlacement) const override;

   private:
    std::optional<RaggedTensor> raggedLeft;
    std::optional<RaggedTensor> raggedRight;
    std::optional<RaggedTensor> raggedOutput;
    std::set<uint32_t> connectedInputPortIndices;
    bool emittedFeatureOutputAfterAllInputsConnected = false;

    friend class Builder;
};

class Add::Builder {
   public:
    virtual ~Builder() = default;
    virtual Add build();

    virtual Builder& network(Network& network) {
        if (_network.has_value()) throw std::runtime_error("Add network may only be set once.");
        _network = &network;
        return *this;
    }

    virtual Builder& left(Tensor tensor) {
        if (_left.has_value() || _raggedLeft.has_value()) throw std::runtime_error("Add left input may only be set once.");
        if (!tensor.isInitialized()) throw std::invalid_argument("Add left input must be initialized.");
        _left = tensor;
        return *this;
    }

    virtual Builder& right(Tensor tensor) {
        if (_right.has_value() || _raggedRight.has_value()) throw std::runtime_error("Add right input may only be set once.");
        if (!tensor.isInitialized()) throw std::invalid_argument("Add right input must be initialized.");
        _right = tensor;
        return *this;
    }

    virtual Builder& left(RaggedTensor tensor) {
        if (_left.has_value() || _raggedLeft.has_value()) throw std::runtime_error("Add left input may only be set once.");
        if (!tensor.isInitialized()) throw std::invalid_argument("Add ragged left input must be initialized.");
        _raggedLeft = tensor;
        _left = tensor.getValues();
        return *this;
    }

    virtual Builder& right(RaggedTensor tensor) {
        if (_right.has_value() || _raggedRight.has_value()) throw std::runtime_error("Add right input may only be set once.");
        if (!tensor.isInitialized()) throw std::invalid_argument("Add ragged right input must be initialized.");
        _raggedRight = tensor;
        _right = tensor.getValues();
        return *this;
    }

   private:
    std::optional<Network*> _network;
    std::optional<Tensor> _left;
    std::optional<Tensor> _right;
    std::optional<RaggedTensor> _raggedLeft;
    std::optional<RaggedTensor> _raggedRight;
};

}  // namespace Thor
