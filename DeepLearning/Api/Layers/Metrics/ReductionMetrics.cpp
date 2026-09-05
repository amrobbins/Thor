#include "DeepLearning/Api/Layers/Metrics/ReductionMetrics.h"

#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Layers/Utility/SegmentedPrimitiveCommon.h"

#include <stdexcept>
#include <string>

using namespace std;
using json = nlohmann::json;

namespace Thor {

#define THOR_DEFINE_UNARY_REDUCTION_DESERIALIZE(ApiName, snake_name, DenseAggregation, RaggedSupported, RaggedAggregation) \
void ApiName::deserialize(const json& j, Network* network) {                                                          \
    if (j.at("version").get<std::string>() != "1.0.0")                                                             \
        throw runtime_error("Unsupported version in " #ApiName "::deserialize: " + j["version"].get<std::string>()); \
    if (j.at("layer_type").get<std::string>() != snake_name)                                                        \
        throw runtime_error("Layer type mismatch in " #ApiName "::deserialize: " + j.at("layer_type").get<std::string>()); \
                                                                                                                      \
    ApiName metric;                                                                                                   \
    if (j.contains("ragged_values")) {                                                                               \
        if (!(RaggedSupported))                                                                                       \
            throw runtime_error("Serialized " #ApiName " does not support ragged values.");                       \
        RaggedTensor values = SegmentedPrimitiveDetail::reconstructInput(j.at("ragged_values"), network, #ApiName); \
        ThorImplementation::ReductionMetricDType::validateValueDType(                                                 \
            #ApiName, "values", values.getValuesDataType());                                                        \
        metric.raggedValues = values;                                                                                 \
        metric.featureInput = values.getValues();                                                                     \
        metric.labelsTensor = values.getOffsets();                                                                    \
        metric.aggregation = RaggedAggregation;                                                                       \
    } else {                                                                                                          \
        nlohmann::json valuesJson = j.at("values").get<nlohmann::json>();                                           \
        uint64_t originalTensorId = valuesJson.at("id").get<uint64_t>();                                            \
        Tensor values = network->getApiTensorByOriginalId(originalTensorId);                                          \
        ThorImplementation::ReductionMetricDType::validateValueDType(                                                 \
            #ApiName, "values", values.getDataType());                                                              \
        metric.featureInput = values;                                                                                 \
        metric.raggedValues.reset();                                                                                  \
        metric.aggregation = DenseAggregation;                                                                        \
    }                                                                                                                 \
    if (j.at("aggregation").get<MetricAggregation>() != metric.aggregation)                                         \
        throw runtime_error("Serialized " #ApiName " aggregation does not match its dense/ragged contract.");     \
    metric.metricTensor = Tensor::deserialize(j.at("metric").get<nlohmann::json>());                                \
    if (metric.metricTensor.getDataType() != DataType::FP32 ||                                                        \
        metric.metricTensor.getDimensions() != vector<uint64_t>{1})                                                   \
        throw runtime_error("Serialized " #ApiName " metric output must be FP32 [1].");                           \
    metric.initialized = true;                                                                                        \
    metric.addToNetwork(network);                                                                                     \
}

THOR_DEFINE_UNARY_REDUCTION_DESERIALIZE(Mean,
                                        "mean",
                                        MetricAggregation::MEAN_BY_EXAMPLE,
                                        true,
                                        MetricAggregation::RATIO)
THOR_DEFINE_UNARY_REDUCTION_DESERIALIZE(Sum,
                                        "sum",
                                        MetricAggregation::SUM,
                                        true,
                                        MetricAggregation::SUM)
THOR_DEFINE_UNARY_REDUCTION_DESERIALIZE(Min,
                                        "min",
                                        MetricAggregation::MIN,
                                        true,
                                        MetricAggregation::MIN)
THOR_DEFINE_UNARY_REDUCTION_DESERIALIZE(Max,
                                        "max",
                                        MetricAggregation::MAX,
                                        true,
                                        MetricAggregation::MAX)

#undef THOR_DEFINE_UNARY_REDUCTION_DESERIALIZE

json WeightedMean::architectureJson() const {
    json j;
    j["factory"] = Layer::Factory::Metric.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = "weighted_mean";
    j["aggregation"] = getAggregation();
    if (getUseRagged()) {
        THOR_THROW_IF_FALSE(raggedValues.has_value() && raggedWeights.has_value());
        j["ragged_values"] = raggedValues->architectureJson();
        j["ragged_weights"] = raggedWeights->architectureJson();
    } else {
        j["values"] = getValues().architectureJson();
        j["weights"] = getWeights().architectureJson();
    }
    j["metric"] = metricTensor.architectureJson();
    return j;
}

void WeightedMean::deserialize(const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in WeightedMean::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "weighted_mean")
        throw runtime_error("Layer type mismatch in WeightedMean::deserialize: " + j.at("layer_type").get<std::string>());
    if (j.at("aggregation").get<MetricAggregation>() != MetricAggregation::RATIO)
        throw runtime_error("Serialized WeightedMean must use RATIO aggregation.");

    const bool hasRaggedValues = j.contains("ragged_values");
    const bool hasRaggedWeights = j.contains("ragged_weights");
    if (hasRaggedValues != hasRaggedWeights)
        throw runtime_error("Serialized WeightedMean ragged values and weights must both be present.");
    if (hasRaggedValues && (j.contains("values") || j.contains("weights")))
        throw runtime_error("Serialized WeightedMean cannot mix dense and ragged inputs.");

    WeightedMean metric;
    if (hasRaggedValues) {
        RaggedTensor values = SegmentedPrimitiveDetail::reconstructInput(j.at("ragged_values"), network, "WeightedMean");
        RaggedTensor weights = SegmentedPrimitiveDetail::reconstructInput(j.at("ragged_weights"), network, "WeightedMean");
        ThorImplementation::ReductionMetricDType::validateValueDType(
            "WeightedMean", "values", values.getValuesDataType());
        ThorImplementation::ReductionMetricDType::validateValueDType(
            "WeightedMean", "weights", weights.getValuesDataType());
        if (!values.sharesPartitionWith(weights))
            throw runtime_error("WeightedMean ragged values and weights must use the exact same row partition during deserialization.");
        if (values.getBatchSize() != weights.getBatchSize() ||
            values.getMaxTotalValues() != weights.getMaxTotalValues() ||
            values.getOffsetsDataType() != weights.getOffsetsDataType() ||
            values.getTrailingDimensions() != weights.getTrailingDimensions() ||
            values.hasMaxValuesPerRow() != weights.hasMaxValuesPerRow() ||
            (values.hasMaxValuesPerRow() && values.getMaxValuesPerRow() != weights.getMaxValuesPerRow())) {
            throw runtime_error("WeightedMean ragged values and weights metadata must match during deserialization.");
        }
        metric.raggedValues = values;
        metric.raggedWeights = weights;
        metric.featureInput = values.getValues();
        metric.labelsTensor = weights.getValues();
    } else {
        nlohmann::json valuesJson = j.at("values").get<nlohmann::json>();
        uint64_t originalTensorId = valuesJson.at("id").get<uint64_t>();
        Tensor values = network->getApiTensorByOriginalId(originalTensorId);

        nlohmann::json weightsJson = j.at("weights").get<nlohmann::json>();
        originalTensorId = weightsJson.at("id").get<uint64_t>();
        Tensor weights = network->getApiTensorByOriginalId(originalTensorId);

        ThorImplementation::ReductionMetricDType::validateValueDType(
            "WeightedMean", "values", values.getDataType());
        ThorImplementation::ReductionMetricDType::validateValueDType(
            "WeightedMean", "weights", weights.getDataType());
        if (values.getDimensions() != weights.getDimensions())
            throw runtime_error("WeightedMean values and weights dimensions must match during deserialization.");
        metric.featureInput = values;
        metric.labelsTensor = weights;
    }

    metric.metricTensor = Tensor::deserialize(j.at("metric").get<nlohmann::json>());
    if (metric.metricTensor.getDataType() != DataType::FP32 ||
        metric.metricTensor.getDimensions() != vector<uint64_t>{1}) {
        throw runtime_error("Serialized WeightedMean metric output must be FP32 [1].");
    }
    metric.initialized = true;
    metric.addToNetwork(network);
}

}  // namespace Thor

namespace {
static const bool registered_reduction_metrics = [] {
    Thor::Metric::register_layer("mean", &Thor::Mean::deserialize);
    Thor::Metric::register_layer("sum", &Thor::Sum::deserialize);
    Thor::Metric::register_layer("min", &Thor::Min::deserialize);
    Thor::Metric::register_layer("max", &Thor::Max::deserialize);
    Thor::Metric::register_layer("weighted_mean", &Thor::WeightedMean::deserialize);
    return true;
}();
}  // namespace
