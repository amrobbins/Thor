#pragma once

#include "DeepLearning/Api/Layers/Loss/Loss.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"
#include "DeepLearning/Implementation/Layers/Loss/RaggedCustomLoss.h"
#include "Utilities/Expression/DynamicExpression.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Thor {

// Graph/API adapter for the internal rank-1 ragged valuewise loss primitive.
// The API graph still connects the physical values and offsets tensors
// explicitly, while callers retain the logical RaggedTensor views.
class RaggedCustomLoss : public Loss {
   public:
    class Builder;

    ~RaggedCustomLoss() override = default;

    std::shared_ptr<Layer> clone() const override { return std::make_shared<RaggedCustomLoss>(*this); }
    std::string getLayerType() const override { return "RaggedCustomLoss"; }

    [[nodiscard]] RaggedTensor getRaggedPredictions() const { return raggedPredictions; }
    [[nodiscard]] RaggedTensor getRaggedLabels() const { return raggedLabels; }
    [[nodiscard]] RaggedTensor getRaggedRawLoss() const { return raggedRawLoss; }
    [[nodiscard]] std::optional<RaggedTensor> getRaggedExampleWeights() const { return raggedExampleWeights; }
    [[nodiscard]] std::optional<RaggedTensor> getRaggedSecondaryInput() const {
        return raggedSecondaryInputs.empty() ? std::nullopt : std::optional<RaggedTensor>(raggedSecondaryInputs.front());
    }
    [[nodiscard]] const std::vector<RaggedTensor>& getRaggedSecondaryInputs() const { return raggedSecondaryInputs; }

    std::vector<Tensor> getLossInputTensors() const override {
        std::vector<Tensor> inputs{raggedPredictions.getValues(), raggedLabels.getValues(), raggedPredictions.getOffsets()};
        for (const RaggedTensor& secondary : raggedSecondaryInputs)
            inputs.push_back(secondary.getValues());
        if (raggedExampleWeights.has_value())
            inputs.push_back(raggedExampleWeights->getValues());
        return inputs;
    }

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
    RaggedCustomLoss(ThorImplementation::DynamicExpression lossExpression,
                     ThorImplementation::DynamicExpression gradientExpression);

    ThorImplementation::DynamicExpression lossExpression;
    ThorImplementation::DynamicExpression gradientExpression;
    RaggedTensor raggedPredictions;
    RaggedTensor raggedLabels;
    RaggedTensor raggedRawLoss;
    std::optional<RaggedTensor> raggedExampleWeights;
    std::vector<RaggedTensor> raggedSecondaryInputs;
    std::string predictionsName = "predictions";
    std::string labelsName = "labels";
    std::string lossName = "loss";
    std::string gradientName = "predictions_grad";
    std::string exampleWeightsName = "example_weights";
    std::vector<std::string> secondaryInputNames;
    std::vector<std::string> secondaryGradientNames;

    friend class Builder;
};

class RaggedCustomLoss::Builder {
   public:
    Builder& network(Network& network) {
        THOR_THROW_IF_FALSE(!_network.has_value());
        _network = &network;
        return *this;
    }
    Builder& lossExpression(ThorImplementation::DynamicExpression expression) {
        THOR_THROW_IF_FALSE(!_lossExpression.has_value());
        _lossExpression = std::move(expression);
        return *this;
    }
    Builder& gradientExpression(ThorImplementation::DynamicExpression expression) {
        THOR_THROW_IF_FALSE(!_gradientExpression.has_value());
        _gradientExpression = std::move(expression);
        return *this;
    }
    Builder& predictions(RaggedTensor predictions) {
        THOR_THROW_IF_FALSE(!_predictions.has_value());
        _predictions = std::move(predictions);
        return *this;
    }
    Builder& labels(RaggedTensor labels) {
        THOR_THROW_IF_FALSE(!_labels.has_value());
        _labels = std::move(labels);
        return *this;
    }
    Builder& secondaryInput(RaggedTensor input, std::string name, std::string gradientName) {
        _secondaryInputs.push_back(std::move(input));
        _secondaryInputNames.push_back(std::move(name));
        _secondaryGradientNames.push_back(std::move(gradientName));
        return *this;
    }
    Builder& exampleWeights(RaggedTensor exampleWeights) {
        THOR_THROW_IF_FALSE(!_exampleWeights.has_value());
        _exampleWeights = std::move(exampleWeights);
        return *this;
    }
    Builder& predictionsName(std::string name) {
        _predictionsName = std::move(name);
        return *this;
    }
    Builder& labelsName(std::string name) {
        _labelsName = std::move(name);
        return *this;
    }
    Builder& lossName(std::string name) {
        _lossName = std::move(name);
        return *this;
    }
    Builder& gradientName(std::string name) {
        _gradientName = std::move(name);
        return *this;
    }
    Builder& exampleWeightsName(std::string name) {
        _exampleWeightsName = std::move(name);
        return *this;
    }
    Builder& lossDataType(DataType dataType) {
        THOR_THROW_IF_FALSE(!_lossDataType.has_value());
        _lossDataType = dataType;
        return *this;
    }
    Builder& lossWeight(float lossWeight) {
        THOR_THROW_IF_FALSE(!_lossWeight.has_value());
        ThorImplementation::validateLossWeight(lossWeight);
        _lossWeight = lossWeight;
        return *this;
    }

    RaggedCustomLoss build();

   private:
    std::optional<Network*> _network;
    std::optional<ThorImplementation::DynamicExpression> _lossExpression;
    std::optional<ThorImplementation::DynamicExpression> _gradientExpression;
    std::optional<RaggedTensor> _predictions;
    std::optional<RaggedTensor> _labels;
    std::optional<RaggedTensor> _exampleWeights;
    std::vector<RaggedTensor> _secondaryInputs;
    std::string _predictionsName = "predictions";
    std::string _labelsName = "labels";
    std::string _lossName = "loss";
    std::string _gradientName = "predictions_grad";
    std::string _exampleWeightsName = "example_weights";
    std::vector<std::string> _secondaryInputNames;
    std::vector<std::string> _secondaryGradientNames;
    std::optional<DataType> _lossDataType;
    std::optional<float> _lossWeight;
};

}  // namespace Thor
