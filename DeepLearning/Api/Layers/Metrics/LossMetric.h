#pragma once

#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Metrics/Metric.h"
#include "DeepLearning/Implementation/Layers/Loss/LossExpression.h"
#include "DeepLearning/Implementation/Layers/Metrics/CustomMetric.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace Thor {

class LossMetric : public Metric {
   public:
    using Formula = ThorImplementation::LossExpression::Formula;

    class Builder;

    LossMetric() = default;
    ~LossMetric() override = default;

    std::shared_ptr<Layer> clone() const override { return std::make_shared<LossMetric>(*this); }
    std::string getLayerType() const override { return "LossMetric"; }

    Formula getFormula() const { return formula; }
    float getEpsilon() const { return epsilon; }
    float getMaxMagnitude() const { return maxMagnitude; }
    const std::string& getDisplayName() const { return displayName; }

    nlohmann::json architectureJson() const override;
    static void deserialize(const nlohmann::json& j, Network* network);

   protected:
    std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                     std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                     std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                     Thor::Tensor connectingApiTensor,
                                                     const bool inferenceOnly) const override {
        (void)placement;
        (void)drivingLayer;
        (void)drivingApiLayer;
        (void)inferenceOnly;
        THOR_THROW_IF_FALSE(initialized);
        THOR_THROW_IF_FALSE(connectingApiTensor == getPredictions() || connectingApiTensor == getLabels());

        ThorImplementation::LossExpression::Options options;
        options.formula = formula;
        options.computeDataType = DataType::FP32;
        options.epsilon = epsilon;
        options.maxMagnitude = maxMagnitude;
        options.predictionsName = predictionsName;
        options.labelsName = labelsName;
        options.lossName = metricName;

        return std::make_shared<ThorImplementation::CustomMetric>(
            ThorImplementation::LossExpression::makeBatchLossMetricExpression(std::move(options)),
            predictionsName,
            labelsName,
            metricName,
            displayName);
    }

   private:
    void initialize(Network* network,
                    Tensor predictions,
                    Tensor labels,
                    Formula formula,
                    float epsilon,
                    float maxMagnitude,
                    std::string displayName,
                    std::optional<Tensor> metricTensor = std::nullopt);

    static void validateInputs(const Tensor& predictions, const Tensor& labels);

    Formula formula = Formula::MEAN_SQUARED_ERROR;
    float epsilon = 0.0001f;
    float maxMagnitude = 1000.0f;
    std::string predictionsName = "predictions";
    std::string labelsName = "labels";
    std::string metricName = "metric";
    std::string displayName = "Loss";
};

class LossMetric::Builder {
   public:
    virtual LossMetric build() {
        THOR_THROW_IF_FALSE(_network.has_value());
        THOR_THROW_IF_FALSE(_predictions.has_value());
        THOR_THROW_IF_FALSE(_labels.has_value());
        THOR_THROW_IF_FALSE(_predictions.value() != _labels.value());

        Formula formula = _formula.value_or(Formula::MEAN_SQUARED_ERROR);
        float epsilon = _epsilon.value_or(0.0001f);
        float maxMagnitude = _maxMagnitude.value_or(1000.0f);
        std::string displayName = _displayName.value_or(ThorImplementation::LossExpression::displayName(formula));

        LossMetric metric;
        metric.initialize(_network.value(),
                          _predictions.value(),
                          _labels.value(),
                          formula,
                          epsilon,
                          maxMagnitude,
                          std::move(displayName));
        return metric;
    }

    virtual LossMetric::Builder& network(Network& network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &network;
        return *this;
    }

    virtual LossMetric::Builder& predictions(Tensor predictions) {
        THOR_THROW_IF_FALSE(!this->_predictions.has_value());
        THOR_THROW_IF_FALSE(predictions.isInitialized());
        this->_predictions = std::move(predictions);
        return *this;
    }

    virtual LossMetric::Builder& labels(Tensor labels) {
        THOR_THROW_IF_FALSE(!this->_labels.has_value());
        THOR_THROW_IF_FALSE(labels.isInitialized());
        this->_labels = std::move(labels);
        return *this;
    }

    virtual LossMetric::Builder& formula(Formula formula) {
        THOR_THROW_IF_FALSE(!this->_formula.has_value());
        this->_formula = formula;
        return *this;
    }

    virtual LossMetric::Builder& meanSquaredError() { return formula(Formula::MEAN_SQUARED_ERROR); }
    virtual LossMetric::Builder& meanAbsoluteError() { return formula(Formula::MEAN_ABSOLUTE_ERROR); }

    virtual LossMetric::Builder& meanAbsolutePercentageError(float epsilon = 0.0001f, float maxMagnitude = 1000.0f) {
        THOR_THROW_IF_FALSE(epsilon >= 0.0f);
        THOR_THROW_IF_FALSE(maxMagnitude >= 0.0f);
        formula(Formula::MEAN_ABSOLUTE_PERCENTAGE_ERROR);
        this->_epsilon = epsilon;
        this->_maxMagnitude = maxMagnitude;
        return *this;
    }

    virtual LossMetric::Builder& displayName(std::string displayName) {
        THOR_THROW_IF_FALSE(!this->_displayName.has_value());
        this->_displayName = std::move(displayName);
        return *this;
    }

   private:
    std::optional<Network*> _network;
    std::optional<Tensor> _predictions;
    std::optional<Tensor> _labels;
    std::optional<Formula> _formula;
    std::optional<float> _epsilon;
    std::optional<float> _maxMagnitude;
    std::optional<std::string> _displayName;
};

}  // namespace Thor
