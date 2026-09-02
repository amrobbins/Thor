#pragma once

#include "DeepLearning/Api/Layers/MultiConnectionLayer.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"
#include "DeepLearning/Implementation/Layers/Loss/RaggedLossShaper.h"

#include <memory>
#include <optional>
#include <set>
#include <string>

namespace Thor {

class RaggedLossShaper : public MultiConnectionLayer {
   public:
    class Builder;

    RaggedLossShaper() = default;
    ~RaggedLossShaper() override = default;

    std::shared_ptr<Layer> clone() const override { return std::make_shared<RaggedLossShaper>(*this); }
    std::string getLayerType() const override { return "RaggedLossShaper"; }

    [[nodiscard]] RaggedTensor getRaggedLossInput() const { return raggedLossInput; }
    [[nodiscard]] Tensor getLossOutput() const { return lossOutput; }

    int getConnectionType(Tensor connectingTensor) const override;
    std::vector<Tensor> getOutputsFromInput(Tensor inputTensor) override;
    void informThatInputConnectionMade(Tensor inputTensor) override;
    void resetGraphTraversalState() override;
    [[nodiscard]] std::optional<std::string> getInputPortName(const Tensor& inputTensor) const override;
    [[nodiscard]] std::optional<std::string> getOutputPortName(const Tensor& outputTensor) const override;
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
    RaggedTensor raggedLossInput;
    Tensor lossOutput;
    ThorImplementation::RaggedLossShaper::OutputLossType outputLossType;
    std::set<uint32_t> connectedInputPortIndices;
    bool emittedOutput = false;

    friend class Builder;
};

class RaggedLossShaper::Builder {
   public:
    Builder& network(Network& network) {
        THOR_THROW_IF_FALSE(!_network.has_value());
        _network = &network;
        return *this;
    }
    Builder& lossInput(RaggedTensor lossInput) {
        THOR_THROW_IF_FALSE(!_lossInput.has_value());
        _lossInput = std::move(lossInput);
        return *this;
    }
    Builder& reportsBatchLoss() {
        THOR_THROW_IF_FALSE(!_outputLossType.has_value());
        _outputLossType = ThorImplementation::RaggedLossShaper::OutputLossType::BATCH;
        return *this;
    }
    Builder& reportsPerExampleLoss() {
        THOR_THROW_IF_FALSE(!_outputLossType.has_value());
        _outputLossType = ThorImplementation::RaggedLossShaper::OutputLossType::PER_EXAMPLE;
        return *this;
    }
    RaggedLossShaper build();

   private:
    std::optional<Network*> _network;
    std::optional<RaggedTensor> _lossInput;
    std::optional<ThorImplementation::RaggedLossShaper::OutputLossType> _outputLossType;
};

}  // namespace Thor
