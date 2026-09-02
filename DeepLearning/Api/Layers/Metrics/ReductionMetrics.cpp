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
                                        false,
                                        MetricAggregation::MIN)
THOR_DEFINE_UNARY_REDUCTION_DESERIALIZE(Max,
                                        "max",
                                        MetricAggregation::MAX,
                                        false,
                                        MetricAggregation::MAX)

#undef THOR_DEFINE_UNARY_REDUCTION_DESERIALIZE

json WeightedMean::architectureJson() const {
    json j;
    j["factory"] = Layer::Factory::Metric.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = "weighted_mean";
    j["aggregation"] = getAggregation();
    j["values"] = getValues().architectureJson();
    j["weights"] = getWeights().architectureJson();
    j["metric"] = metricTensor.architectureJson();
    return j;
}

void WeightedMean::deserialize(const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in WeightedMean::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "weighted_mean")
        throw runtime_error("Layer type mismatch in WeightedMean::deserialize: " + j.at("layer_type").get<std::string>());

    nlohmann::json valuesJson = j["values"].get<nlohmann::json>();
    uint64_t originalTensorId = valuesJson.at("id").get<uint64_t>();
    Tensor values = network->getApiTensorByOriginalId(originalTensorId);

    nlohmann::json weightsJson = j["weights"].get<nlohmann::json>();
    originalTensorId = weightsJson.at("id").get<uint64_t>();
    Tensor weights = network->getApiTensorByOriginalId(originalTensorId);

    ThorImplementation::ReductionMetricDType::validateValueDType(
        "WeightedMean", "values", values.getDataType());
    ThorImplementation::ReductionMetricDType::validateValueDType(
        "WeightedMean", "weights", weights.getDataType());
    if (values.getDimensions() != weights.getDimensions())
        throw runtime_error("WeightedMean values and weights dimensions must match during deserialization.");

    Tensor metricTensor = Tensor::deserialize(j.at("metric").get<nlohmann::json>());

    WeightedMean metric;
    metric.featureInput = values;
    metric.labelsTensor = weights;
    metric.metricTensor = metricTensor;
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
