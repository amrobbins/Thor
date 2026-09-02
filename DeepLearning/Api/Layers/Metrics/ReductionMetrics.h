#pragma once

#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Metrics/Metric.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"
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
    std::vector<Tensor> getAllInputTensors() const override {
        if (raggedValues.has_value())
            return {raggedValues->getValues(), raggedValues->getOffsets()};
        return {getValues()};
    }
    int getConnectionType(Tensor connectingTensor) const override {
        if (connectingTensor == getValues())
            return static_cast<int>(ThorImplementation::Metric::ConnectionType::FORWARD);
        if (raggedValues.has_value() && connectingTensor == raggedValues->getOffsets())
            return static_cast<int>(ThorImplementation::Metric::ConnectionType::STRUCTURAL);
        if (connectingTensor == getMetric())
            return static_cast<int>(ThorImplementation::Metric::ConnectionType::METRIC);
        THOR_UNREACHABLE();
    }
    [[nodiscard]] std::optional<std::string> getInputPortName(const Tensor& inputTensor) const override {
        if (inputTensor == getValues())
            return "values";
        if (raggedValues.has_value() && inputTensor == raggedValues->getOffsets())
            return "offsets";
        return std::nullopt;
    }
    std::optional<RaggedTensor> getRaggedValues() const { return raggedValues; }
    bool getUseRagged() const { return raggedValues.has_value(); }
    MetricAggregation getAggregation() const override { return aggregation; }

    nlohmann::json architectureJson() const override {
        nlohmann::json j;
        j["factory"] = Layer::Factory::Metric.value();
        j["version"] = getLayerVersion();
        j["layer_type"] = to_snake_case(getLayerType());
        j["aggregation"] = getAggregation();
        if (raggedValues.has_value())
            j["ragged_values"] = raggedValues->architectureJson();
        else
            j["values"] = getValues().architectureJson();
        j["metric"] = metricTensor.architectureJson();
        return j;
    }

   protected:
    uint64_t getFirstInstanceMemRequirementInBytes(
        uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const override {
        if (!raggedValues.has_value())
            return Metric::getFirstInstanceMemRequirementInBytes(batchSize, tensorPlacement);

        (void)batchSize;
        (void)tensorPlacement;
        // Ragged reductions do not allocate the dense per-row validity mask.
        // RATIO metrics do use CustomMetric's two device statistics plus the
        // slot-local device buffers used to publish exact batch statistics.
        uint64_t bytes = metricTensor.getTotalSizeInBytes();
        if (aggregation == MetricAggregation::RATIO)
            bytes += 4 * sizeof(float);
        return bytes;
    }

    void initializeUnaryReductionMetric(Network* network,
                                        Tensor values,
                                        MetricAggregation denseAggregation) {
        THOR_THROW_IF_FALSE(network != nullptr);
        THOR_THROW_IF_FALSE(values.isInitialized());
        THOR_THROW_IF_FALSE(!values.getDimensions().empty());
        ThorImplementation::ReductionMetricDType::validateValueDType(getLayerType(), "values", values.getDataType());

        featureInput = std::move(values);
        raggedValues.reset();
        aggregation = denseAggregation;
        metricTensor = Tensor(DataType::FP32, {1});
        initialized = true;
        addToNetwork(network);
    }

    void initializeUnaryReductionMetric(Network* network,
                                        RaggedTensor values,
                                        MetricAggregation raggedAggregation) {
        THOR_THROW_IF_FALSE(network != nullptr);
        if (!values.isInitialized())
            throw std::invalid_argument(getLayerType() + " ragged values must be initialized.");
        ThorImplementation::ReductionMetricDType::validateValueDType(
            getLayerType(), "values", values.getValuesDataType());

        raggedValues = values;
        featureInput = values.getValues();
        // Keep the structural offsets tensor available for stamping, but do not
        // expose it as a semantic labels input. getAllInputTensors() and
        // getConnectionType() publish it as a dedicated structural port.
        labelsTensor = values.getOffsets();
        aggregation = raggedAggregation;
        metricTensor = Tensor(DataType::FP32, {1});
        initialized = true;
        addToNetwork(network);
    }

    template <typename ImplementationMetric>
    std::shared_ptr<ThorImplementation::Layer> stampDenseUnaryReductionMetric(Thor::Tensor connectingApiTensor) const {
        THOR_THROW_IF_FALSE(initialized);
        THOR_THROW_IF_FALSE(!raggedValues.has_value());
        THOR_THROW_IF_FALSE(connectingApiTensor == getFeatureInput().value());
        return std::make_shared<ImplementationMetric>();
    }

    std::shared_ptr<ThorImplementation::Layer> stampRaggedUnaryReductionMetric(
        ThorImplementation::RaggedReductionMetric::Kind kind,
        Thor::Tensor connectingApiTensor) const {
        THOR_THROW_IF_FALSE(initialized);
        THOR_THROW_IF_FALSE(raggedValues.has_value());
        if (connectingApiTensor != raggedValues->getValues() && connectingApiTensor != raggedValues->getOffsets())
            throw std::invalid_argument(getLayerType() + " ragged stamp received an unrelated tensor.");
        return std::make_shared<ThorImplementation::RaggedReductionMetric>(
            kind, raggedValues->getBatchSize(), raggedValues->getMaxTotalValues());
    }

    std::optional<RaggedTensor> raggedValues;
    MetricAggregation aggregation = MetricAggregation::MEAN_BY_EXAMPLE;
};

#define THOR_DECLARE_UNARY_REDUCTION_METRIC(ApiName, ImplName, DenseAggregationValue, RaggedSupported, RaggedAggregationValue, RaggedKind) \
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
        if (getUseRagged()) {                                                                                         \
            if (!(RaggedSupported))                                                                                   \
                throw std::invalid_argument(#ApiName " does not support RaggedTensor values.");                     \
            return stampRaggedUnaryReductionMetric(RaggedKind, connectingApiTensor);                                 \
        }                                                                                                             \
        return stampDenseUnaryReductionMetric<ThorImplementation::ImplName>(connectingApiTensor);                    \
    }                                                                                                                 \
};                                                                                                                    \
                                                                                                                      \
class ApiName::Builder {                                                                                              \
   public:                                                                                                            \
    virtual ApiName build() {                                                                                         \
        THOR_THROW_IF_FALSE(_network.has_value());                                                                    \
        if (_values.has_value() == _raggedValues.has_value())                                                         \
            throw std::invalid_argument(#ApiName " requires exactly one dense Tensor or RaggedTensor values input."); \
        ApiName metric;                                                                                               \
        if (_raggedValues.has_value()) {                                                                              \
            if (!(RaggedSupported))                                                                                   \
                throw std::invalid_argument(#ApiName " does not support RaggedTensor values.");                     \
            metric.initializeUnaryReductionMetric(_network.value(), _raggedValues.value(), RaggedAggregationValue);  \
        } else {                                                                                                      \
            metric.initializeUnaryReductionMetric(_network.value(), _values.value(), DenseAggregationValue);         \
        }                                                                                                             \
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
        THOR_THROW_IF_FALSE(!this->_values.has_value() && !this->_raggedValues.has_value());                          \
        THOR_THROW_IF_FALSE(values.isInitialized());                                                                  \
        THOR_THROW_IF_FALSE(!values.getDimensions().empty());                                                         \
        ThorImplementation::ReductionMetricDType::validateValueDType(#ApiName, "values", values.getDataType());      \
        this->_values = std::move(values);                                                                            \
        return *this;                                                                                                 \
    }                                                                                                                 \
                                                                                                                      \
    virtual ApiName::Builder& values(RaggedTensor values) {                                                           \
        THOR_THROW_IF_FALSE(!this->_values.has_value() && !this->_raggedValues.has_value());                          \
        if (!(RaggedSupported))                                                                                       \
            throw std::invalid_argument(#ApiName " does not support RaggedTensor values.");                         \
        if (!values.isInitialized())                                                                                  \
            throw std::invalid_argument(#ApiName " ragged values must be initialized.");                            \
        ThorImplementation::ReductionMetricDType::validateValueDType(#ApiName, "values", values.getValuesDataType()); \
        this->_raggedValues = std::move(values);                                                                      \
        return *this;                                                                                                 \
    }                                                                                                                 \
                                                                                                                      \
   private:                                                                                                           \
    std::optional<Network*> _network;                                                                                 \
    std::optional<Tensor> _values;                                                                                    \
    std::optional<RaggedTensor> _raggedValues;                                                                        \
};

THOR_DECLARE_UNARY_REDUCTION_METRIC(Mean,
                                    Mean,
                                    MetricAggregation::MEAN_BY_EXAMPLE,
                                    true,
                                    MetricAggregation::RATIO,
                                    ThorImplementation::RaggedReductionMetric::Kind::MEAN)
THOR_DECLARE_UNARY_REDUCTION_METRIC(Sum,
                                    Sum,
                                    MetricAggregation::SUM,
                                    true,
                                    MetricAggregation::SUM,
                                    ThorImplementation::RaggedReductionMetric::Kind::SUM)
THOR_DECLARE_UNARY_REDUCTION_METRIC(Min,
                                    Min,
                                    MetricAggregation::MIN,
                                    false,
                                    MetricAggregation::MIN,
                                    ThorImplementation::RaggedReductionMetric::Kind::SUM)
THOR_DECLARE_UNARY_REDUCTION_METRIC(Max,
                                    Max,
                                    MetricAggregation::MAX,
                                    false,
                                    MetricAggregation::MAX,
                                    ThorImplementation::RaggedReductionMetric::Kind::SUM)

#undef THOR_DECLARE_UNARY_REDUCTION_METRIC

class WeightedMean : public Metric {
   public:
    class Builder;
    WeightedMean() = default;
    ~WeightedMean() override = default;

    std::shared_ptr<Layer> clone() const override { return std::make_shared<WeightedMean>(*this); }
    std::string getLayerType() const override { return "WeightedMean"; }
    MetricAggregation getAggregation() const override { return MetricAggregation::RATIO; }

    Tensor getValues() const { return getFeatureInput().value(); }
    Tensor getWeights() const { return labelsTensor; }

    nlohmann::json architectureJson() const override;
    static void deserialize(const nlohmann::json& j, Network* network);

   protected:
    uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize,
                                                   ThorImplementation::TensorPlacement tensorPlacement) const override {
        return Metric::getFirstInstanceMemRequirementInBytes(batchSize, tensorPlacement) + 4 * sizeof(float);
    }

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
