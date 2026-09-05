#include "DeepLearning/Api/Layers/Metrics/BinaryAccuracy.h"
#include "DeepLearning/Api/Layers/Utility/SegmentedPrimitiveCommon.h"
#include "DeepLearning/Api/Network/Network.h"

#include <stdexcept>
#include <vector>

using namespace std;
using json = nlohmann::json;

namespace Thor {
namespace {

bool binaryLabelDTypeSupported(DataType dtype) {
    return dtype == DataType::UINT8 || dtype == DataType::UINT16 || dtype == DataType::UINT32 ||
           dtype == DataType::INT8 || dtype == DataType::INT16 || dtype == DataType::INT32 ||
           dtype == DataType::FP16 || dtype == DataType::FP32;
}

void validateRaggedBinaryPair(const RaggedTensor& predictions, const RaggedTensor& labels) {
    if (!predictions.sharesPartitionWith(labels))
        throw runtime_error("BinaryAccuracy ragged predictions and labels must use the exact same row partition.");
    if (predictions.getBatchSize() != labels.getBatchSize() ||
        predictions.getMaxTotalValues() != labels.getMaxTotalValues() ||
        predictions.getOffsetsDataType() != labels.getOffsetsDataType() ||
        predictions.hasMaxValuesPerRow() != labels.hasMaxValuesPerRow() ||
        (predictions.hasMaxValuesPerRow() && predictions.getMaxValuesPerRow() != labels.getMaxValuesPerRow())) {
        throw runtime_error("BinaryAccuracy ragged predictions and labels partition metadata must match.");
    }
    if (predictions.getTrailingDimensions() != vector<uint64_t>{1} || labels.getTrailingDimensions() != vector<uint64_t>{1})
        throw runtime_error("BinaryAccuracy ragged predictions and labels must contain one scalar per token.");
    if (predictions.getValuesDataType() != DataType::FP16 && predictions.getValuesDataType() != DataType::FP32)
        throw runtime_error("BinaryAccuracy ragged predictions must be FP16 or FP32.");
    if (!binaryLabelDTypeSupported(labels.getValuesDataType()))
        throw runtime_error("BinaryAccuracy ragged labels use an unsupported dtype.");
}

}  // namespace

json BinaryAccuracy::architectureJson() const {
    if (!getUseRagged())
        return Metric::architectureJson();
    THOR_THROW_IF_FALSE(raggedLabels.has_value());
    return json{{"factory", Layer::Factory::Metric.value()},
                {"version", getLayerVersion()},
                {"layer_type", "binary_accuracy"},
                {"aggregation", getAggregation()},
                {"ragged_predictions", raggedPredictions->architectureJson()},
                {"ragged_labels", raggedLabels->architectureJson()},
                {"metric", metricTensor.architectureJson()}};
}

void BinaryAccuracy::deserialize(const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in BinaryAccuracy::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "binary_accuracy")
        throw runtime_error("Layer type mismatch in BinaryAccuracy::deserialize: " + j.at("layer_type").get<std::string>());

    BinaryAccuracy metric;
    const bool hasRaggedPredictions = j.contains("ragged_predictions");
    const bool hasRaggedLabels = j.contains("ragged_labels");
    if (hasRaggedPredictions != hasRaggedLabels)
        throw runtime_error("Serialized BinaryAccuracy ragged predictions and labels must both be present.");
    if (hasRaggedPredictions) {
        if (j.contains("predictions") || j.contains("labels"))
            throw runtime_error("Serialized BinaryAccuracy cannot mix dense and ragged inputs.");
        RaggedTensor predictions =
            SegmentedPrimitiveDetail::reconstructInput(j.at("ragged_predictions"), network, "BinaryAccuracy");
        RaggedTensor labels =
            SegmentedPrimitiveDetail::reconstructInput(j.at("ragged_labels"), network, "BinaryAccuracy");
        validateRaggedBinaryPair(predictions, labels);
        metric.raggedPredictions = predictions;
        metric.raggedLabels = labels;
        metric.featureInput = predictions.getValues();
        metric.labelsTensor = labels.getValues();
    } else {
        const nlohmann::json input = j.at("predictions").get<nlohmann::json>();
        const uint64_t predictionId = input.at("id").get<uint64_t>();
        metric.featureInput = network->getApiTensorByOriginalId(predictionId);
        const nlohmann::json labels = j.at("labels").get<nlohmann::json>();
        const uint64_t labelId = labels.at("id").get<uint64_t>();
        metric.labelsTensor = network->getApiTensorByOriginalId(labelId);
    }

    const MetricAggregation expectedAggregation =
        hasRaggedPredictions ? MetricAggregation::RATIO : MetricAggregation::MEAN_BY_EXAMPLE;
    if (j.at("aggregation").get<MetricAggregation>() != expectedAggregation)
        throw runtime_error("Serialized BinaryAccuracy aggregation does not match its dense/ragged contract.");

    metric.metricTensor = Tensor::deserialize(j.at("metric").get<nlohmann::json>());
    if (metric.metricTensor.getDataType() != DataType::FP32 || metric.metricTensor.getDimensions() != vector<uint64_t>{1})
        throw runtime_error("Serialized BinaryAccuracy metric output must be FP32 [1].");
    metric.initialized = true;
    metric.addToNetwork(network);
}

}  // namespace Thor

namespace {
static const bool registered = [] {
    Thor::Metric::register_layer("binary_accuracy", &Thor::BinaryAccuracy::deserialize);
    return true;
}();
}  // namespace
