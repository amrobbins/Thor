#pragma once

#include "DeepLearning/Api/Layers/MultiConnectionLayer.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace Thor {

class SegmentedReduction : public MultiConnectionLayer {
   public:
    enum class Type { SUM, MEAN, MIN, MAX };
    class Builder;

    SegmentedReduction() = default;
    ~SegmentedReduction() override = default;

    std::shared_ptr<Layer> clone() const override { return std::make_shared<SegmentedReduction>(*this); }
    std::string getLayerType() const override { return "SegmentedReduction"; }

    [[nodiscard]] Type getReductionType() const { return reductionType; }
    [[nodiscard]] RaggedTensor getRaggedFeatureInput() const { return raggedFeatureInput; }

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
    static const char* typeName(Type type);
    static Type typeFromName(const std::string& name);

    RaggedTensor raggedFeatureInput;
    Type reductionType = Type::SUM;
    std::set<uint32_t> connectedInputPortIndices;
    bool emittedFeatureOutputAfterAllInputsConnected = false;

    friend class Builder;
};

class SegmentedReduction::Builder {
   public:
    virtual ~Builder() = default;
    virtual SegmentedReduction build();

    virtual Builder& network(Network& network) {
        if (_network.has_value()) throw std::runtime_error("SegmentedReduction network may only be set once.");
        _network = &network;
        return *this;
    }

    virtual Builder& featureInput(RaggedTensor featureInput) {
        if (_featureInput.has_value()) throw std::runtime_error("SegmentedReduction feature input may only be set once.");
        if (!featureInput.isInitialized()) throw std::invalid_argument("SegmentedReduction feature input must be initialized.");
        _featureInput = featureInput;
        return *this;
    }

    virtual Builder& reductionType(Type type) {
        if (_reductionType.has_value()) throw std::runtime_error("SegmentedReduction reduction type may only be set once.");
        _reductionType = type;
        return *this;
    }

   private:
    std::optional<Network*> _network;
    std::optional<RaggedTensor> _featureInput;
    std::optional<Type> _reductionType;
};

}  // namespace Thor
