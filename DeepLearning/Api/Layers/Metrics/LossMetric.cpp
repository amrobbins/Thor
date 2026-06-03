#include "DeepLearning/Api/Layers/Metrics/LossMetric.h"

#include "DeepLearning/Api/Network/Network.h"

#include <stdexcept>
#include <string>
#include <utility>

using namespace std;
using json = nlohmann::json;

namespace Thor {

void LossMetric::validateInputs(const Tensor& predictions, const Tensor& labels) {
    THOR_THROW_IF_FALSE(predictions.isInitialized());
    THOR_THROW_IF_FALSE(labels.isInitialized());
    THOR_THROW_IF_FALSE(predictions.getDimensions() == labels.getDimensions());
    THOR_THROW_IF_FALSE(!predictions.getDimensions().empty());
    THOR_THROW_IF_FALSE(ThorImplementation::LossExpression::isPredictionDTypeSupported(predictions.getDataType()));
    THOR_THROW_IF_FALSE(ThorImplementation::LossExpression::isLabelDTypeSupported(labels.getDataType()));
}

void LossMetric::initialize(Network* network,
                            Tensor predictions,
                            Tensor labels,
                            Formula formula,
                            float epsilon,
                            float maxMagnitude,
                            std::string displayName,
                            std::optional<Tensor> metricTensorOverride) {
    THOR_THROW_IF_FALSE(network != nullptr);
    validateInputs(predictions, labels);
    THOR_THROW_IF_FALSE(epsilon >= 0.0f);
    THOR_THROW_IF_FALSE(maxMagnitude >= 0.0f);

    featureInput = std::move(predictions);
    labelsTensor = std::move(labels);
    this->formula = formula;
    this->epsilon = epsilon;
    this->maxMagnitude = maxMagnitude;
    this->displayName = displayName.empty() ? ThorImplementation::LossExpression::displayName(formula) : std::move(displayName);

    Tensor inferredMetric(DataType::FP32, {1});
    if (metricTensorOverride.has_value()) {
        THOR_THROW_IF_FALSE(metricTensorOverride.value().getDataType() == inferredMetric.getDataType());
        THOR_THROW_IF_FALSE(metricTensorOverride.value().getDimensions() == inferredMetric.getDimensions());
        metricTensor = metricTensorOverride.value();
    } else {
        metricTensor = inferredMetric;
    }

    initialized = true;
    addToNetwork(network);
}

json LossMetric::architectureJson() const {
    json j = Metric::architectureJson();
    j["layer_type"] = "loss_metric";
    j["formula"] = ThorImplementation::LossExpression::toString(formula);
    j["epsilon"] = epsilon;
    j["max_magnitude"] = maxMagnitude;
    j["predictions_name"] = predictionsName;
    j["labels_name"] = labelsName;
    j["metric_name"] = metricName;
    j["display_name"] = displayName;
    j["reduction"] = "batch";
    return j;
}

void LossMetric::deserialize(const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in LossMetric::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "loss_metric")
        throw runtime_error("Layer type mismatch in LossMetric::deserialize: " + j.at("layer_type").get<std::string>());

    const uint64_t predictionsOriginalId = j.at("predictions").at("id").get<uint64_t>();
    Tensor predictions = network->getApiTensorByOriginalId(predictionsOriginalId);

    const uint64_t labelsOriginalId = j.at("labels").at("id").get<uint64_t>();
    Tensor labels = network->getApiTensorByOriginalId(labelsOriginalId);

    Tensor metricTensor = Tensor::deserialize(j.at("metric").get<json>());
    Formula formula = ThorImplementation::LossExpression::formulaFromString(j.at("formula").get<std::string>());
    float epsilon = j.value("epsilon", 0.0001f);
    float maxMagnitude = j.value("max_magnitude", 1000.0f);
    std::string displayName = j.value("display_name", ThorImplementation::LossExpression::displayName(formula));

    LossMetric metric;
    metric.initialize(network, predictions, labels, formula, epsilon, maxMagnitude, std::move(displayName), metricTensor);
}

}  // namespace Thor

namespace {
static const bool registered_loss_metric = [] {
    Thor::Metric::register_layer("loss_metric", &Thor::LossMetric::deserialize);
    return true;
}();
}  // namespace
