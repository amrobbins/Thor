#pragma once

#include "DeepLearning/Api/Layers/MultiConnectionLayer.h"
#include "DeepLearning/Implementation/Layers/Utility/EinsumLayer.h"
#include "Utilities/TensorOperations/Einsum/EinsumParser.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Thor {

/**
 * User-visible trainable einsum layer.
 *
 * The textual equation describes per-example feature dimensions only. Thor's
 * runtime batch dimension is implicit and is preserved by the physical
 * EinsumLayer implementation.
 */
class Einsum : public MultiConnectionLayer {
   public:
    class Builder;

    Einsum() = default;
    ~Einsum() override = default;

    std::shared_ptr<Layer> clone() const override { return std::make_shared<Einsum>(*this); }

    [[nodiscard]] const std::string& getEquation() const { return equation; }
    std::string getLayerType() const override { return "Einsum"; }

    std::optional<Tensor> getFeatureOutput() const override {
        THOR_THROW_IF_FALSE(featureOutputs.size() == 1);
        return featureOutputs[0];
    }

    Tensor getFeatureOutput(Tensor inputTensor) const override;
    Tensor getFeatureInput(Tensor outputTensor) const override;

    int getConnectionType(Tensor connectingTensor) const override;
    [[nodiscard]] std::optional<std::string> getInputPortName(const Tensor& inputTensor) const override;
    [[nodiscard]] std::optional<std::string> getOutputPortName(const Tensor& outputTensor) const override;

    bool mustConnectAllInputsToDriveOutput() const override { return true; }
    void informThatInputConnectionMade(Tensor inputTensor) override;
    std::vector<Tensor> getOutputsFromInput(Tensor inputTensor) override;
    void resetGraphTraversalState() override;

    nlohmann::json architectureJson() const override;
    static void deserialize(const nlohmann::json& j, Network* network);

   protected:
    std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                     std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                     std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                     Thor::Tensor connectingApiTensor,
                                                     const bool inferenceOnly) const override;

    uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize,
                                                   ThorImplementation::TensorPlacement tensorPlacement) const override;

   private:
    static bool isSupportedStorageDType(DataType dataType);
    static void validateAndResolve(const std::string& equation,
                                   const std::vector<Tensor>& featureInputs,
                                   ThorImplementation::ResolvedEinsumEquation* resolved);
    void rebuildInputBindings();

    std::string equation;

    // A single API tensor may intentionally occupy more than one einsum operand
    // position. Physical stamping asks getConnectionType() once per graph
    // connection, so repeated calls rotate deterministically through those
    // logical operand bindings (the same contract used by CustomLayer).
    std::unordered_map<uint64_t, std::vector<uint32_t>> inputOperandBindingsByTensorOriginalId;
    mutable std::unordered_map<uint64_t, uint32_t> nextInputBindingConnectionCursorByTensorOriginalId;

    std::set<uint32_t> connectedInputOperandIndices;
    bool emittedFeatureOutputAfterAllInputsConnected = false;

    friend class Builder;
};

class Einsum::Builder {
   public:
    virtual ~Builder() = default;

    virtual Einsum build();

    virtual Builder& network(Network& network) {
        if (_network.has_value()) {
            throw std::runtime_error("Einsum network may only be set once.");
        }
        _network = &network;
        return *this;
    }

    virtual Builder& equation(std::string equation) {
        if (_equation.has_value()) {
            throw std::runtime_error("Einsum equation may only be set once.");
        }
        _equation = std::move(equation);
        return *this;
    }

    virtual Builder& featureInput(Tensor featureInput) {
        if (!featureInput.isInitialized()) {
            throw std::invalid_argument("Einsum feature input must be initialized.");
        }
        _featureInputs.push_back(std::move(featureInput));
        return *this;
    }

    virtual Builder& featureInputs(std::vector<Tensor> featureInputs) {
        if (!_featureInputs.empty()) {
            throw std::runtime_error("Einsum feature inputs may only be set once.");
        }
        for (const Tensor& featureInput : featureInputs) {
            if (!featureInput.isInitialized()) {
                throw std::invalid_argument("Einsum feature inputs must be initialized.");
            }
        }
        _featureInputs = std::move(featureInputs);
        return *this;
    }

   private:
    std::optional<Network*> _network;
    std::optional<std::string> _equation;
    std::vector<Tensor> _featureInputs;
};

}  // namespace Thor
