#pragma once

#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Metrics/Metric.h"
#include "DeepLearning/Implementation/Layers/Metrics/ReductionMetricDType.h"
#include "DeepLearning/Implementation/Layers/Metrics/ReductionMetrics.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace Thor {

class UnaryReductionMetric : public Metric {
   public:
    bool requiresLabels() const override { return false; }
    Tensor getValues() const { return getFeatureInput().value(); }

    nlohmann::json architectureJson() const override {
        nlohmann::json j;
        j["factory"] = Layer::Factory::Metric.value();
        j["version"] = getLayerVersion();
        j["layer_type"] = to_snake_case(getLayerType());
        j["values"] = getValues().architectureJson();
        j["metric"] = metricTensor.architectureJson();
        return j;
    }

   protected:
    void initializeUnaryReductionMetric(Network* network, Tensor values) {
        THOR_THROW_IF_FALSE(network != nullptr);
        THOR_THROW_IF_FALSE(values.isInitialized());
        THOR_THROW_IF_FALSE(!values.getDimensions().empty());
        ThorImplementation::ReductionMetricDType::validateValueDType(getLayerType(), "values", values.getDataType());

        featureInput = std::move(values);
        metricTensor = Tensor(DataType::FP32, {1});
        initialized = true;
        addToNetwork(network);
    }

    template <typename ImplementationMetric>
    std::shared_ptr<ThorImplementation::Layer> stampUnaryReductionMetric(Thor::Tensor connectingApiTensor) const {
        THOR_THROW_IF_FALSE(initialized);
        THOR_THROW_IF_FALSE(connectingApiTensor == getFeatureInput().value());
        return std::make_shared<ImplementationMetric>();
    }
};

#define THOR_DECLARE_UNARY_REDUCTION_METRIC(ApiName, ImplName)                                                        \
class ApiName : public UnaryReductionMetric {                                                                         \
   public:                                                                                                            \
    class Builder;                                                                                                    \
    ApiName() = default;                                                                                              \
    ~ApiName() override = default;                                                                                    \
    std::shared_ptr<Layer> clone() const override { return std::make_shared<ApiName>(*this); }                        \
    std::string getLayerType() const override { return #ApiName; }                                                    \
    static void deserialize(const nlohmann::json& j, Network* network);                                               \
                                                                                                                      \
   protected:                                                                                                         \
    std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,                   \
                                                     std::shared_ptr<ThorImplementation::Layer> drivingLayer,         \
                                                     std::shared_ptr<Thor::Layer> drivingApiLayer,                    \
                                                     Thor::Tensor connectingApiTensor,                                \
                                                     const bool inferenceOnly) const override {                       \
        (void)placement;                                                                                              \
        (void)drivingLayer;                                                                                           \
        (void)drivingApiLayer;                                                                                        \
        (void)inferenceOnly;                                                                                          \
        return stampUnaryReductionMetric<ThorImplementation::ImplName>(connectingApiTensor);                          \
    }                                                                                                                 \
};                                                                                                                    \
                                                                                                                      \
class ApiName::Builder {                                                                                              \
   public:                                                                                                            \
    virtual ApiName build() {                                                                                         \
        THOR_THROW_IF_FALSE(_network.has_value());                                                                    \
        THOR_THROW_IF_FALSE(_values.has_value());                                                                     \
        ApiName metric;                                                                                               \
        metric.initializeUnaryReductionMetric(_network.value(), _values.value());                                      \
        return metric;                                                                                                \
    }                                                                                                                 \
                                                                                                                      \
    virtual ApiName::Builder& network(Network& network) {                                                             \
        THOR_THROW_IF_FALSE(!this->_network.has_value());                                                             \
        this->_network = &network;                                                                                    \
        return *this;                                                                                                 \
    }                                                                                                                 \
                                                                                                                      \
    virtual ApiName::Builder& values(Tensor values) {                                                                 \
        THOR_THROW_IF_FALSE(!this->_values.has_value());                                                              \
        THOR_THROW_IF_FALSE(values.isInitialized());                                                                  \
        THOR_THROW_IF_FALSE(!values.getDimensions().empty());                                                         \
        ThorImplementation::ReductionMetricDType::validateValueDType(                                     \
            #ApiName, "values", values.getDataType());                                                               \
        this->_values = std::move(values);                                                                            \
        return *this;                                                                                                 \
    }                                                                                                                 \
                                                                                                                      \
   private:                                                                                                           \
    std::optional<Network*> _network;                                                                                 \
    std::optional<Tensor> _values;                                                                                    \
};

THOR_DECLARE_UNARY_REDUCTION_METRIC(Mean, Mean)
THOR_DECLARE_UNARY_REDUCTION_METRIC(Sum, Sum)
THOR_DECLARE_UNARY_REDUCTION_METRIC(Min, Min)
THOR_DECLARE_UNARY_REDUCTION_METRIC(Max, Max)

#undef THOR_DECLARE_UNARY_REDUCTION_METRIC

class WeightedMean : public Metric {
   public:
    class Builder;
    WeightedMean() = default;
    ~WeightedMean() override = default;

    std::shared_ptr<Layer> clone() const override { return std::make_shared<WeightedMean>(*this); }
    std::string getLayerType() const override { return "WeightedMean"; }

    Tensor getValues() const { return getFeatureInput().value(); }
    Tensor getWeights() const { return labelsTensor; }

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
        THOR_THROW_IF_FALSE(connectingApiTensor == getFeatureInput().value() || connectingApiTensor == labelsTensor);
        return std::make_shared<ThorImplementation::WeightedMean>();
    }
};

class WeightedMean::Builder {
   public:
    virtual WeightedMean build() {
        THOR_THROW_IF_FALSE(_network.has_value());
        THOR_THROW_IF_FALSE(_values.has_value());
        THOR_THROW_IF_FALSE(_weights.has_value());
        THOR_THROW_IF_FALSE(_values.value() != _weights.value());
        THOR_THROW_IF_FALSE(!_values.value().getDimensions().empty());
        THOR_THROW_IF_FALSE(_values.value().getDimensions() == _weights.value().getDimensions());
        ThorImplementation::ReductionMetricDType::validateValueDType(
            "WeightedMean", "values", _values.value().getDataType());
        ThorImplementation::ReductionMetricDType::validateValueDType(
            "WeightedMean", "weights", _weights.value().getDataType());

        WeightedMean metric;
        metric.featureInput = _values.value();
        metric.labelsTensor = _weights.value();
        metric.metricTensor = Tensor(DataType::FP32, {1});
        metric.initialized = true;
        metric.addToNetwork(_network.value());
        return metric;
    }

    virtual WeightedMean::Builder& network(Network& network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &network;
        return *this;
    }

    virtual WeightedMean::Builder& values(Tensor values) {
        THOR_THROW_IF_FALSE(!this->_values.has_value());
        THOR_THROW_IF_FALSE(values.isInitialized());
        THOR_THROW_IF_FALSE(!values.getDimensions().empty());
        ThorImplementation::ReductionMetricDType::validateValueDType(
            "WeightedMean", "values", values.getDataType());
        this->_values = std::move(values);
        return *this;
    }

    virtual WeightedMean::Builder& weights(Tensor weights) {
        THOR_THROW_IF_FALSE(!this->_weights.has_value());
        THOR_THROW_IF_FALSE(weights.isInitialized());
        THOR_THROW_IF_FALSE(!weights.getDimensions().empty());
        ThorImplementation::ReductionMetricDType::validateValueDType(
            "WeightedMean", "weights", weights.getDataType());
        this->_weights = std::move(weights);
        return *this;
    }

   private:
    std::optional<Network*> _network;
    std::optional<Tensor> _values;
    std::optional<Tensor> _weights;
};

}  // namespace Thor
