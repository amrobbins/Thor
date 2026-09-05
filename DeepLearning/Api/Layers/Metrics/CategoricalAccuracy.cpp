#include "DeepLearning/Api/Layers/Metrics/CategoricalAccuracy.h"
#include "DeepLearning/Api/Layers/Utility/SegmentedPrimitiveCommon.h"
#include "DeepLearning/Api/Network/Network.h"

#include <limits>
#include <stdexcept>
#include <vector>

using namespace std;
using json = nlohmann::json;

namespace Thor {
namespace {

bool perClassLabelDTypeSupported(DataType dtype) {
    return dtype == DataType::UINT8 || dtype == DataType::UINT16 || dtype == DataType::UINT32 ||
           dtype == DataType::INT8 || dtype == DataType::INT16 || dtype == DataType::INT32 ||
           dtype == DataType::FP16 || dtype == DataType::FP32;
}

bool publicClassIndexLabelDTypeSupported(DataType dtype) {
    return dtype == DataType::UINT8 || dtype == DataType::UINT16 || dtype == DataType::UINT32;
}

void validateRaggedCategoricalPair(const RaggedTensor& predictions,
                                   const RaggedTensor& labels,
                                   CategoricalAccuracy::LabelType labelType) {
    if (!predictions.sharesPartitionWith(labels))
        throw runtime_error("CategoricalAccuracy ragged predictions and labels must use the exact same row partition.");
    if (predictions.getBatchSize() != labels.getBatchSize() ||
        predictions.getMaxTotalValues() != labels.getMaxTotalValues() ||
        predictions.getOffsetsDataType() != labels.getOffsetsDataType() ||
        predictions.hasMaxValuesPerRow() != labels.hasMaxValuesPerRow() ||
        (predictions.hasMaxValuesPerRow() && predictions.getMaxValuesPerRow() != labels.getMaxValuesPerRow())) {
        throw runtime_error("CategoricalAccuracy ragged predictions and labels partition metadata must match.");
    }
    if (predictions.getValuesDataType() != DataType::FP16 && predictions.getValuesDataType() != DataType::FP32)
        throw runtime_error("CategoricalAccuracy ragged predictions must be FP16 or FP32.");
    const vector<uint64_t> predictionTrailing = predictions.getTrailingDimensions();
    if (predictionTrailing.size() != 1 || predictionTrailing[0] < 2 ||
        predictionTrailing[0] > numeric_limits<uint32_t>::max())
        throw runtime_error("CategoricalAccuracy ragged predictions must have one valid trailing class dimension.");
    if (labelType == CategoricalAccuracy::LabelType::INDEX) {
        if (labels.getTrailingDimensions() != vector<uint64_t>{1})
            throw runtime_error("CategoricalAccuracy ragged class-index labels must contain one scalar per token.");
        if (!publicClassIndexLabelDTypeSupported(labels.getValuesDataType()))
            throw runtime_error("CategoricalAccuracy ragged class-index labels must use UINT8, UINT16, or UINT32.");
    } else {
        if (labels.getTrailingDimensions() != predictionTrailing)
            throw runtime_error("CategoricalAccuracy ragged one-hot/per-class labels must match prediction class width.");
        if (!perClassLabelDTypeSupported(labels.getValuesDataType()))
            throw runtime_error("CategoricalAccuracy ragged per-class labels use an unsupported dtype.");
    }
}

}  // namespace

json CategoricalAccuracy::architectureJson() const {
    if (!getUseRagged()) {
        json j = Metric::architectureJson();
        j["label_type"] = labelType;
        return j;
    }
    THOR_THROW_IF_FALSE(raggedLabels.has_value());
    return json{{"factory", Layer::Factory::Metric.value()},
                {"version", getLayerVersion()},
                {"layer_type", "categorical_accuracy"},
                {"aggregation", getAggregation()},
                {"label_type", labelType},
                {"ragged_predictions", raggedPredictions->architectureJson()},
                {"ragged_labels", raggedLabels->architectureJson()},
                {"metric", metricTensor.architectureJson()}};
}

void CategoricalAccuracy::deserialize(const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in CategoricalAccuracy::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "categorical_accuracy")
        throw runtime_error("Layer type mismatch in CategoricalAccuracy::deserialize: " + j.at("layer_type").get<std::string>());

    const LabelType restoredLabelType = j.at("label_type").get<LabelType>();
    CategoricalAccuracy metric;
    metric.labelType = restoredLabelType;
    const bool hasRaggedPredictions = j.contains("ragged_predictions");
    const bool hasRaggedLabels = j.contains("ragged_labels");
    if (hasRaggedPredictions != hasRaggedLabels)
        throw runtime_error("Serialized CategoricalAccuracy ragged predictions and labels must both be present.");
    if (hasRaggedPredictions) {
        if (j.contains("predictions") || j.contains("labels"))
            throw runtime_error("Serialized CategoricalAccuracy cannot mix dense and ragged inputs.");
        RaggedTensor predictions =
            SegmentedPrimitiveDetail::reconstructInput(j.at("ragged_predictions"), network, "CategoricalAccuracy");
        RaggedTensor labels =
            SegmentedPrimitiveDetail::reconstructInput(j.at("ragged_labels"), network, "CategoricalAccuracy");
        validateRaggedCategoricalPair(predictions, labels, restoredLabelType);
        metric.raggedPredictions = predictions;
        metric.raggedLabels = labels;
        metric.featureInput = predictions.getValues();
        metric.labelsTensor = labels.getValues();
        metric.numClasses = static_cast<uint32_t>(predictions.getTrailingDimensions().at(0));
    } else {
        const nlohmann::json input = j.at("predictions").get<nlohmann::json>();
        const uint64_t predictionId = input.at("id").get<uint64_t>();
        metric.featureInput = network->getApiTensorByOriginalId(predictionId);
        const nlohmann::json labels = j.at("labels").get<nlohmann::json>();
        const uint64_t labelId = labels.at("id").get<uint64_t>();
        metric.labelsTensor = network->getApiTensorByOriginalId(labelId);
        const vector<uint64_t> predictionDims = metric.featureInput->getDimensions();
        if (predictionDims.size() == 1 && predictionDims[0] <= numeric_limits<uint32_t>::max())
            metric.numClasses = static_cast<uint32_t>(predictionDims[0]);
    }

    const MetricAggregation expectedAggregation =
        hasRaggedPredictions ? MetricAggregation::RATIO : MetricAggregation::MEAN_BY_EXAMPLE;
    if (j.at("aggregation").get<MetricAggregation>() != expectedAggregation)
        throw runtime_error("Serialized CategoricalAccuracy aggregation does not match its dense/ragged contract.");

    metric.metricTensor = Tensor::deserialize(j.at("metric").get<nlohmann::json>());
    if (metric.metricTensor.getDataType() != DataType::FP32 || metric.metricTensor.getDimensions() != vector<uint64_t>{1})
        throw runtime_error("Serialized CategoricalAccuracy metric output must be FP32 [1].");
    metric.initialized = true;
    metric.addToNetwork(network);
}

}  // namespace Thor

namespace {
static const bool registered = [] {
    Thor::Metric::register_layer("categorical_accuracy", &Thor::CategoricalAccuracy::deserialize);
    return true;
}();
}  // namespace
