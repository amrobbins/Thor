#pragma once

#include "DeepLearning/Api/DataType.h"
#include "DeepLearning/Api/Layers/Metrics/Metric.h"
#include "DeepLearning/Implementation/Layers/Metrics/CustomMetric.h"
#include "Utilities/Expression/DynamicExpression.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Thor {

class CustomMetric : public Metric {
   public:
    using TensorMap = std::unordered_map<std::string, Tensor>;

    class Builder;

    CustomMetric(ThorImplementation::DynamicExpression expr,
                 Tensor predictions,
                 Tensor labels,
                 std::string predictionsName = "predictions",
                 std::string labelsName = "labels",
                 std::string metricName = "metric",
                 std::optional<Tensor> metricTensor = std::nullopt,
                 std::string displayName = "Metric");

    ~CustomMetric() override = default;

    std::shared_ptr<Layer> clone() const override { return std::make_shared<CustomMetric>(*this); }
    std::string getLayerType() const override { return "CustomMetric"; }

    const std::string& getPredictionsName() const { return predictionsName; }
    const std::string& getLabelsName() const { return labelsName; }
    const std::string& getMetricName() const { return metricName; }
    const std::string& getDisplayName() const { return displayName; }
    const ThorImplementation::DynamicExpression& getExpression() const { return expr; }

    nlohmann::json architectureJson() const override;
    static void deserialize(const nlohmann::json& j, Network* network);

   protected:
    std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                     std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                     std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                     Thor::Tensor connectingApiTensor,
                                                     const bool inferenceOnly) const override;

    uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const override;

   private:
    static void validateName(const std::string& name, const std::string& what);
    static ThorImplementation::Tensor makeFakePlacedTensor(const Tensor& apiTensor);
    static Tensor logicalMetricTensorFromFakeOutput(const std::vector<uint64_t>& fakeOutputDims, DataType dtype);
    Tensor inferMetricTensor() const;

    ThorImplementation::DynamicExpression expr;
    std::string predictionsName = "predictions";
    std::string labelsName = "labels";
    std::string metricName = "metric";
    std::string displayName = "Metric";
};

class CustomMetric::Builder {
   public:
    virtual ~Builder() = default;

    virtual CustomMetric build() {
        THOR_THROW_IF_FALSE(_network.has_value());
        THOR_THROW_IF_FALSE(_expr != nullptr);
        THOR_THROW_IF_FALSE(_predictions.has_value());
        THOR_THROW_IF_FALSE(_labels.has_value());
        THOR_THROW_IF_FALSE(_predictions.value() != _labels.value());

        std::string predictionsName = _predictionsName.value_or("predictions");
        std::string labelsName = _labelsName.value_or("labels");
        std::string metricName = _metricName.value_or("metric");
        std::string displayName = _displayName.value_or("Metric");

        const std::vector<std::string>& expectedInputs = _expr->getExpectedInputNames();
        if (!_predictionsName.has_value() && !_labelsName.has_value() && expectedInputs.size() == 2) {
            const bool hasPredictionsDefault =
                std::find(expectedInputs.begin(), expectedInputs.end(), std::string("predictions")) != expectedInputs.end();
            const bool hasLabelsDefault =
                std::find(expectedInputs.begin(), expectedInputs.end(), std::string("labels")) != expectedInputs.end();
            if (!(hasPredictionsDefault && hasLabelsDefault)) {
                predictionsName = expectedInputs[0];
                labelsName = expectedInputs[1];
            }
        }

        const std::vector<std::string>& expectedOutputs = _expr->getExpectedOutputNames();
        if (!_metricName.has_value() && expectedOutputs.size() == 1)
            metricName = expectedOutputs[0];

        CustomMetric customMetric(*_expr,
                                  _predictions.value(),
                                  _labels.value(),
                                  std::move(predictionsName),
                                  std::move(labelsName),
                                  std::move(metricName),
                                  _metricTensor,
                                  std::move(displayName));
        customMetric.addToNetwork(_network.value());
        return customMetric;
    }

    virtual CustomMetric::Builder& network(Network& network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &network;
        return *this;
    }

    virtual CustomMetric::Builder& expression(ThorImplementation::DynamicExpression expr) {
        THOR_THROW_IF_FALSE(this->_expr == nullptr);
        this->_expr = std::make_shared<ThorImplementation::DynamicExpression>(std::move(expr));
        return *this;
    }

    virtual CustomMetric::Builder& predictions(Tensor predictions) {
        THOR_THROW_IF_FALSE(!this->_predictions.has_value());
        THOR_THROW_IF_FALSE(predictions.isInitialized());
        this->_predictions = std::move(predictions);
        return *this;
    }

    virtual CustomMetric::Builder& labels(Tensor labels) {
        THOR_THROW_IF_FALSE(!this->_labels.has_value());
        THOR_THROW_IF_FALSE(labels.isInitialized());
        this->_labels = std::move(labels);
        return *this;
    }

    virtual CustomMetric::Builder& predictionsName(std::string name) {
        THOR_THROW_IF_FALSE(!this->_predictionsName.has_value());
        this->_predictionsName = std::move(name);
        return *this;
    }

    virtual CustomMetric::Builder& labelsName(std::string name) {
        THOR_THROW_IF_FALSE(!this->_labelsName.has_value());
        this->_labelsName = std::move(name);
        return *this;
    }

    virtual CustomMetric::Builder& metricName(std::string name) {
        THOR_THROW_IF_FALSE(!this->_metricName.has_value());
        this->_metricName = std::move(name);
        return *this;
    }

    virtual CustomMetric::Builder& metricTensor(Tensor tensor) {
        THOR_THROW_IF_FALSE(!this->_metricTensor.has_value());
        THOR_THROW_IF_FALSE(tensor.isInitialized());
        this->_metricTensor = std::move(tensor);
        return *this;
    }

    virtual CustomMetric::Builder& displayName(std::string name) {
        THOR_THROW_IF_FALSE(!this->_displayName.has_value());
        this->_displayName = std::move(name);
        return *this;
    }

   private:
    std::optional<Network*> _network;
    std::shared_ptr<ThorImplementation::DynamicExpression> _expr;
    std::optional<Tensor> _predictions;
    std::optional<Tensor> _labels;
    std::optional<Tensor> _metricTensor;
    std::optional<std::string> _predictionsName;
    std::optional<std::string> _labelsName;
    std::optional<std::string> _metricName;
    std::optional<std::string> _displayName;
};

}  // namespace Thor
