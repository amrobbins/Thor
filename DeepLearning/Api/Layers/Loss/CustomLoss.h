#pragma once

#include "DeepLearning/Api/BatchValidity.h"
#include "DeepLearning/Api/DataType.h"
#include "DeepLearning/Api/Layers/Loss/Loss.h"
#include "DeepLearning/Api/Layers/Loss/LossShaper.h"
#include "DeepLearning/Implementation/Layers/Loss/CustomLoss.h"
#include "Utilities/Expression/DynamicExpression.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Thor {

class CustomLoss : public Loss {
   public:
    class Builder;

    CustomLoss(ThorImplementation::DynamicExpression lossExpression,
               ThorImplementation::DynamicExpression gradientExpression,
               Tensor predictions,
               Tensor labels,
               std::string predictionsName = "predictions",
               std::string labelsName = "labels",
               std::string lossName = "loss",
               std::string gradientName = "predictions_grad",
               std::optional<Tensor> lossTensor = std::nullopt,
               std::optional<DataType> lossDataType = std::nullopt,
               std::optional<float> lossWeight = std::nullopt,
               bool usesBatchValidity = false,
               bool requiresFullBatch = false);

    ~CustomLoss() override = default;

    std::shared_ptr<Layer> clone() const override { return std::make_shared<CustomLoss>(*this); }
    std::string getLayerType() const override { return "CustomLoss"; }

    const std::string& getPredictionsName() const { return predictionsName; }
    const std::string& getLabelsName() const { return labelsName; }
    const std::string& getLossName() const { return lossName; }
    const std::string& getGradientName() const { return gradientName; }
    const ThorImplementation::DynamicExpression& getLossExpression() const { return lossExpression; }
    const ThorImplementation::DynamicExpression& getGradientExpression() const { return gradientExpression; }
    bool usesBatchValidity() const { return batchValidityMaskEnabled; }
    bool requiresFullBatch() const { return fullBatchRequired; }
    [[nodiscard]] std::optional<std::string> getInputPortName(const Tensor& inputTensor) const override {
        if (predictionsTensor.isInitialized() && inputTensor == predictionsTensor) {
            return predictionsName;
        }
        if (labelsTensor.isInitialized() && inputTensor == labelsTensor) {
            return labelsName;
        }
        return Loss::getInputPortName(inputTensor);
    }
    [[nodiscard]] std::optional<std::string> getOutputPortName(const Tensor& outputTensor) const override {
        if (lossShaperInput.isInitialized() && outputTensor == lossShaperInput) {
            return lossName;
        }
        return Loss::getOutputPortName(outputTensor);
    }

    nlohmann::json architectureJson() const override;
    static void deserialize(const nlohmann::json& j, Network* network);

   protected:
    std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                     std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                     std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                     Thor::Tensor connectingApiTensor,
                                                     const bool inferenceOnly) const override;

    virtual bool isMultiLayer() const { return lossShape != LossShape::RAW && lossShape != LossShape::NONE; }
    virtual void buildSupportLayersAndAddToNetwork();

    uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const override;

   private:
    using PhysicalTensor = ThorImplementation::Tensor;
    using PhysicalTensorMap = std::unordered_map<std::string, PhysicalTensor>;

    static void validateName(const std::string& name, const std::string& what);
    static std::set<std::string> toNameSet(const std::vector<std::string>& names);
    static std::string joinNames(const std::set<std::string>& names);
    static PhysicalTensor makeFakePlacedTensor(const Tensor& apiTensor);
    static Tensor logicalLossTensorFromFakeOutput(const std::vector<uint64_t>& fakeOutputDims, DataType dtype);

    void validateExpressionNames(const ThorImplementation::DynamicExpression& expression,
                                 const std::string& outputName,
                                 const std::string& what) const;
    Tensor inferExpressionTensor(const ThorImplementation::DynamicExpression& expression,
                                 const std::string& outputName,
                                 const std::string& what) const;
    Tensor inferLossTensor() const;
    void validateGradientTensor() const;

    ThorImplementation::DynamicExpression lossExpression;
    ThorImplementation::DynamicExpression gradientExpression;
    std::string predictionsName = "predictions";
    std::string labelsName = "labels";
    std::string lossName = "loss";
    std::string gradientName = "predictions_grad";
    bool batchValidityMaskEnabled = false;
    bool fullBatchRequired = false;
};

class CustomLoss::Builder {
   public:
    virtual ~Builder() = default;

    virtual CustomLoss build() {
        THOR_THROW_IF_FALSE(_network.has_value());
        THOR_THROW_IF_FALSE(_lossExpression != nullptr);
        THOR_THROW_IF_FALSE(_gradientExpression != nullptr);
        THOR_THROW_IF_FALSE(_predictions.has_value());
        THOR_THROW_IF_FALSE(_labels.has_value());
        THOR_THROW_IF_FALSE(_predictions.value() != _labels.value());

        std::string predictionsName = _predictionsName.value_or("predictions");
        std::string labelsName = _labelsName.value_or("labels");
        std::string lossName = _lossName.value_or("loss");

        std::vector<std::string> publicExpectedInputs = _lossExpression->getExpectedInputNames();
        if (_usesBatchValidity) {
            publicExpectedInputs.erase(
                std::remove(publicExpectedInputs.begin(),
                            publicExpectedInputs.end(),
                            std::string(Thor::BATCH_VALIDITY_MASK_NAME)),
                publicExpectedInputs.end());
        }
        if (!_predictionsName.has_value() && !_labelsName.has_value() && publicExpectedInputs.size() == 2) {
            const bool hasPredictionsDefault =
                std::find(publicExpectedInputs.begin(), publicExpectedInputs.end(), std::string("predictions")) !=
                publicExpectedInputs.end();
            const bool hasLabelsDefault =
                std::find(publicExpectedInputs.begin(), publicExpectedInputs.end(), std::string("labels")) !=
                publicExpectedInputs.end();
            if (!(hasPredictionsDefault && hasLabelsDefault)) {
                predictionsName = publicExpectedInputs[0];
                labelsName = publicExpectedInputs[1];
            }
        }

        const std::vector<std::string>& expectedOutputs = _lossExpression->getExpectedOutputNames();
        if (!_lossName.has_value() && expectedOutputs.size() == 1)
            lossName = expectedOutputs[0];

        std::string gradientName = _gradientName.value_or(predictionsName + "_grad");
        const std::vector<std::string>& expectedGradientOutputs = _gradientExpression->getExpectedOutputNames();
        if (!_gradientName.has_value() && expectedGradientOutputs.size() == 1)
            gradientName = expectedGradientOutputs[0];

        LossShape lossShape = _lossShape.value_or(LossShape::BATCH);

        CustomLoss customLoss(*_lossExpression,
                              *_gradientExpression,
                              _predictions.value(),
                              _labels.value(),
                              std::move(predictionsName),
                              std::move(labelsName),
                              std::move(lossName),
                              std::move(gradientName),
                              _lossTensor,
                              _lossDataType,
                              ThorImplementation::normalizeLossWeight(_lossWeight),
                              _usesBatchValidity,
                              _requiresFullBatch);
        customLoss.lossShape = lossShape;
        customLoss.lossWeight = ThorImplementation::normalizeLossWeight(_lossWeight);
        customLoss.network = _network.value();

        if (customLoss.isMultiLayer()) {
            customLoss.buildSupportLayersAndAddToNetwork();
        } else {
            customLoss.lossShaperInput = customLoss.lossTensor;
            customLoss.addToNetwork(_network.value());
            customLoss.finalizeLossReporting();
        }

        return customLoss;
    }

    virtual CustomLoss::Builder& network(Network& network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &network;
        return *this;
    }

    virtual CustomLoss::Builder& lossExpression(ThorImplementation::DynamicExpression expression) {
        THOR_THROW_IF_FALSE(this->_lossExpression == nullptr);
        this->_lossExpression = std::make_shared<ThorImplementation::DynamicExpression>(std::move(expression));
        return *this;
    }

    virtual CustomLoss::Builder& gradientExpression(ThorImplementation::DynamicExpression expression) {
        THOR_THROW_IF_FALSE(this->_gradientExpression == nullptr);
        this->_gradientExpression = std::make_shared<ThorImplementation::DynamicExpression>(std::move(expression));
        return *this;
    }

    virtual CustomLoss::Builder& predictions(Tensor predictions) {
        THOR_THROW_IF_FALSE(!this->_predictions.has_value());
        THOR_THROW_IF_FALSE(predictions.isInitialized());
        this->_predictions = std::move(predictions);
        return *this;
    }

    virtual CustomLoss::Builder& labels(Tensor labels) {
        THOR_THROW_IF_FALSE(!this->_labels.has_value());
        THOR_THROW_IF_FALSE(labels.isInitialized());
        this->_labels = std::move(labels);
        return *this;
    }

    virtual CustomLoss::Builder& predictionsName(std::string name) {
        THOR_THROW_IF_FALSE(!this->_predictionsName.has_value());
        this->_predictionsName = std::move(name);
        return *this;
    }

    virtual CustomLoss::Builder& labelsName(std::string name) {
        THOR_THROW_IF_FALSE(!this->_labelsName.has_value());
        this->_labelsName = std::move(name);
        return *this;
    }

    virtual CustomLoss::Builder& lossName(std::string name) {
        THOR_THROW_IF_FALSE(!this->_lossName.has_value());
        this->_lossName = std::move(name);
        return *this;
    }

    virtual CustomLoss::Builder& gradientName(std::string name) {
        THOR_THROW_IF_FALSE(!this->_gradientName.has_value());
        this->_gradientName = std::move(name);
        return *this;
    }

    virtual CustomLoss::Builder& lossTensor(Tensor tensor) {
        THOR_THROW_IF_FALSE(!this->_lossTensor.has_value());
        THOR_THROW_IF_FALSE(tensor.isInitialized());
        this->_lossTensor = std::move(tensor);
        return *this;
    }

    virtual CustomLoss::Builder& lossWeight(float lossWeight) {
        THOR_THROW_IF_FALSE(!this->_lossWeight.has_value());
        ThorImplementation::validateLossWeight(lossWeight);
        this->_lossWeight = ThorImplementation::normalizeLossWeight(lossWeight);
        return *this;
    }

    virtual CustomLoss::Builder& lossDataType(DataType lossDataType) {
        THOR_THROW_IF_FALSE(!this->_lossDataType.has_value());
        THOR_THROW_IF_FALSE(lossDataType == DataType::FP32 || lossDataType == DataType::FP16);
        this->_lossDataType = lossDataType;
        return *this;
    }

    virtual CustomLoss::Builder& reportsBatchLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::BATCH;
        return *this;
    }

    virtual CustomLoss::Builder& reportsPerExampleLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::PER_EXAMPLE;
        return *this;
    }

    virtual CustomLoss::Builder& reportsPerOutputLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::PER_OUTPUT;
        return *this;
    }

    virtual CustomLoss::Builder& reportsNoLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::NONE;
        return *this;
    }

    virtual CustomLoss::Builder& reportsRawLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::RAW;
        return *this;
    }

    /**
     * Declares that both expressions consume runtime batch-validity information through the current reserved FP32
     * Thor::BATCH_VALIDITY_MASK_NAME expression input.
     */
    virtual CustomLoss::Builder& usesBatchValidity() {
        THOR_THROW_IF_FALSE(!_usesBatchValidity);
        THOR_THROW_IF_FALSE(!_requiresFullBatch);
        _usesBatchValidity = true;
        return *this;
    }

    virtual CustomLoss::Builder& requiresFullBatch() {
        THOR_THROW_IF_FALSE(!_requiresFullBatch);
        THOR_THROW_IF_FALSE(!_usesBatchValidity);
        _requiresFullBatch = true;
        return *this;
    }

   private:
    std::optional<Network*> _network;
    std::shared_ptr<ThorImplementation::DynamicExpression> _lossExpression;
    std::shared_ptr<ThorImplementation::DynamicExpression> _gradientExpression;
    std::optional<Tensor> _predictions;
    std::optional<Tensor> _labels;
    std::optional<Tensor> _lossTensor;
    std::optional<std::string> _predictionsName;
    std::optional<std::string> _labelsName;
    std::optional<std::string> _lossName;
    std::optional<std::string> _gradientName;
    std::optional<LossShape> _lossShape;
    std::optional<DataType> _lossDataType;
    std::optional<float> _lossWeight;
    bool _usesBatchValidity = false;
    bool _requiresFullBatch = false;
};

}  // namespace Thor
