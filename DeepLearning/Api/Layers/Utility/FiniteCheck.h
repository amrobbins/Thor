#pragma once

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"
#include "DeepLearning/Implementation/Layers/Utility/FiniteCheck.h"

#include <cstdint>
#include <optional>
#include <set>
#include <string>

namespace Thor {

class FiniteCheck : public Layer {
   public:
    class Builder;

    FiniteCheck();
    ~FiniteCheck() override;

    std::shared_ptr<Layer> clone() const override { return std::make_shared<FiniteCheck>(*this); }
    std::string getLayerType() const override { return "FiniteCheck"; }
    std::string getLayerVersion() const override { return "1.1.0"; }

    nlohmann::json architectureJson() const override;
    static void deserialize(const nlohmann::json &j, Network *network);

    const std::string &getTensorLabel() const { return tensorLabel; }
    bool getEnabled() const { return enabled; }
    bool getCheckForward() const { return checkForward; }
    bool getCheckBackward() const { return checkBackward; }
    bool getFailOnNonFinite() const { return failOnNonFinite; }
    uint32_t getMaxReportedIndices() const { return maxReportedIndices; }
    [[nodiscard]] bool getUseRagged() const { return raggedFeatureInput.has_value(); }
    [[nodiscard]] std::optional<RaggedTensor> getRaggedFeatureInput() const { return raggedFeatureInput; }
    [[nodiscard]] std::optional<RaggedTensor> getRaggedFeatureOutput() const { return raggedFeatureOutput; }

    std::vector<Tensor> getAllInputTensors() const override;
    std::vector<Tensor> getOutputsFromInput(Tensor inputTensor) override;
    bool mustConnectAllInputsToDriveOutput() const override { return raggedFeatureInput.has_value(); }
    void informThatInputConnectionMade(Tensor inputTensor) override;
    void resetGraphTraversalState() override;
    int getConnectionType(Tensor connectingTensor) const override;

    [[nodiscard]] bool outputTensorDimensionsIncludeBatch(const Tensor& outputTensor) const override {
        THOR_THROW_IF_FALSE(featureOutput.has_value());
        THOR_THROW_IF_FALSE(outputTensor == featureOutput.value());
        return raggedFeatureInput.has_value();
    }

   protected:
    std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                     std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                     std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                     Thor::Tensor connectingApiTensor,
                                                     const bool inferenceOnly) const override;

    uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize,
                                                   ThorImplementation::TensorPlacement tensorPlacement) const override;

   private:
    std::string tensorLabel;
    bool enabled = true;
    bool checkForward = true;
    bool checkBackward = true;
    bool failOnNonFinite = true;
    uint32_t maxReportedIndices = 8;
    std::optional<RaggedTensor> raggedFeatureInput;
    std::optional<RaggedTensor> raggedFeatureOutput;
    std::set<uint32_t> connectedInputPortIndices;
    bool emittedFeatureOutputAfterAllInputsConnected = false;
};

class FiniteCheck::Builder {
   public:
    FiniteCheck build();

    Builder &network(Network &network);
    Builder &featureInput(Tensor featureInput);
    Builder &featureInput(RaggedTensor featureInput);
    Builder &tensorLabel(std::string tensorLabel);
    Builder &enabled(bool enabled);
    Builder &checkForward(bool checkForward);
    Builder &checkBackward(bool checkBackward);
    Builder &failOnNonFinite(bool failOnNonFinite);
    Builder &maxReportedIndices(uint32_t maxReportedIndices);

   private:
    std::optional<Network *> _network;
    std::optional<Tensor> _featureInput;
    std::optional<RaggedTensor> _raggedFeatureInput;
    std::string _tensorLabel;
    bool _enabled = true;
    bool _checkForward = true;
    bool _checkBackward = true;
    bool _failOnNonFinite = true;
    uint32_t _maxReportedIndices = 8;
};

}  // namespace Thor
