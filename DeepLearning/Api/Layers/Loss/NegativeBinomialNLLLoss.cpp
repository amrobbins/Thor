#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Api/Layers/Loss/NegativeBinomialNLLLoss.h"

#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/Expression.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {
namespace {

constexpr const char* kMeanName = "mean";
constexpr const char* kDispersionName = "dispersion";
constexpr const char* kLabelsName = "labels";
constexpr const char* kExampleWeightsName = "example_weights";
constexpr const char* kLossName = "loss";
constexpr const char* kMeanGradientName = "mean_grad";
constexpr const char* kDispersionGradientName = "dispersion_grad";

void validateFloatingDType(const char* tensorName, DataType dtype) {
    if (dtype != DataType::FP16 && dtype != DataType::FP32) {
        throw runtime_error(string("Unsupported NegativeBinomialNLLLoss ") + tensorName + " dtype: " +
                            ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
    }
}

void validateLabelsDType(DataType dtype) {
    switch (dtype) {
        case DataType::BOOLEAN:
        case DataType::UINT8:
        case DataType::UINT16:
        case DataType::UINT32:
        case DataType::FP16:
        case DataType::FP32:
            return;
        default:
            throw runtime_error("Unsupported NegativeBinomialNLLLoss labels dtype: " +
                                ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
    }
}

void validateExampleWeights(Tensor mean, Tensor dispersion, Tensor labels, std::optional<Tensor> exampleWeights) {
    if (!exampleWeights.has_value())
        return;
    if (exampleWeights.value() == mean || exampleWeights.value() == dispersion || exampleWeights.value() == labels)
        throw runtime_error("NegativeBinomialNLLLoss example_weights tensor must be distinct from mean, dispersion, and labels.");
    validateFloatingDType("example_weights", exampleWeights.value().getDataType());
    const vector<uint64_t>& dims = exampleWeights.value().getDimensions();
    if (dims != vector<uint64_t>{1} && dims != mean.getDimensions()) {
        throw runtime_error(
            "NegativeBinomialNLLLoss example_weights dimensions must be [1] for per-example weights or match mean dimensions.");
    }
}

ThorImplementation::Expression safePositive(const ThorImplementation::Expression& value, float eps) {
    return value.max(ThorImplementation::Expression(eps));
}

ThorImplementation::Expression positiveValue(const ThorImplementation::Expression& value, bool logValue, float eps) {
    return logValue ? value.exp() : safePositive(value, eps);
}

ThorImplementation::Expression logPositiveValue(const ThorImplementation::Expression& value, bool logValue, float eps) {
    return logValue ? value : safePositive(value, eps).ln();
}

ThorImplementation::Expression negativeBinomialNLL(const ThorImplementation::Expression& meanInput,
                                                   const ThorImplementation::Expression& dispersionInput,
                                                   const ThorImplementation::Expression& labels,
                                                   bool logMean,
                                                   bool logDispersion,
                                                   float eps) {
    ThorImplementation::Expression mean = positiveValue(meanInput, logMean, eps);
    ThorImplementation::Expression dispersion = positiveValue(dispersionInput, logDispersion, eps);
    ThorImplementation::Expression logMeanValue = logPositiveValue(meanInput, logMean, eps);
    ThorImplementation::Expression logDispersionValue = logPositiveValue(dispersionInput, logDispersion, eps);
    ThorImplementation::Expression concentration = ThorImplementation::Expression(1.0) / dispersion;
    ThorImplementation::Expression logOnePlusDispersionMean = (dispersion * mean).log1p();

    return concentration.lgamma() + (labels + ThorImplementation::Expression(1.0)).lgamma() - (labels + concentration).lgamma() +
           (concentration + labels) * logOnePlusDispersionMean - labels * logDispersionValue - labels * logMeanValue;
}

ThorImplementation::DynamicExpression makeNegativeBinomialNLLLossExpression(DataType lossDataType,
                                                                            bool logMean,
                                                                            bool logDispersion,
                                                                            float eps,
                                                                            bool weighted) {
    validateFloatingDType("loss", lossDataType);
    ThorImplementation::Expression mean = ThorImplementation::Expression::input(kMeanName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression dispersion =
        ThorImplementation::Expression::input(kDispersionName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels = ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression loss = negativeBinomialNLL(mean, dispersion, labels, logMean, logDispersion, eps);
    if (weighted) {
        ThorImplementation::Expression exampleWeights =
            ThorImplementation::Expression::input(kExampleWeightsName, DataType::FP32, DataType::FP32);
        loss = loss * exampleWeights;
    }
    loss = loss.withOutputDType(lossDataType);

    ThorImplementation::ExpressionDefinition definition =
        ThorImplementation::ExpressionDefinition::fromOutputs(ThorImplementation::Expression::outputs({{kLossName, loss}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

ThorImplementation::DynamicExpression makeNegativeBinomialNLLGradientExpression(DataType meanDType,
                                                                                DataType dispersionDType,
                                                                                bool logMean,
                                                                                bool logDispersion,
                                                                                float eps,
                                                                                bool weighted) {
    validateFloatingDType("mean", meanDType);
    validateFloatingDType("dispersion", dispersionDType);

    ThorImplementation::Expression meanInput = ThorImplementation::Expression::input(kMeanName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression dispersionInput =
        ThorImplementation::Expression::input(kDispersionName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels = ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression mean = positiveValue(meanInput, logMean, eps);
    ThorImplementation::Expression dispersion = positiveValue(dispersionInput, logDispersion, eps);
    ThorImplementation::Expression concentration = ThorImplementation::Expression(1.0) / dispersion;
    ThorImplementation::Expression onePlusDispersionMean = ThorImplementation::Expression(1.0) + dispersion * mean;
    ThorImplementation::Expression logOnePlusDispersionMean = (dispersion * mean).log1p();

    ThorImplementation::Expression logMeanGradient = (mean - labels) / onePlusDispersionMean;
    ThorImplementation::Expression meanGradient = logMean ? logMeanGradient : logMeanGradient / mean;

    ThorImplementation::Expression concentrationTerm = concentration.digamma() - (labels + concentration).digamma() +
                                                        logOnePlusDispersionMean;
    ThorImplementation::Expression logDispersionGradient =
        (ThorImplementation::Expression(0.0) - concentration) * concentrationTerm +
        (mean - labels) / onePlusDispersionMean;
    ThorImplementation::Expression dispersionGradient =
        logDispersion ? logDispersionGradient : logDispersionGradient / dispersion;

    if (weighted) {
        ThorImplementation::Expression exampleWeights =
            ThorImplementation::Expression::input(kExampleWeightsName, DataType::FP32, DataType::FP32);
        meanGradient = meanGradient * exampleWeights;
        dispersionGradient = dispersionGradient * exampleWeights;
    }

    ThorImplementation::Expression scale(ThorImplementation::Loss::getLossScalingFactor());
    meanGradient = (meanGradient * scale).withOutputDType(meanDType);
    dispersionGradient = (dispersionGradient * scale).withOutputDType(dispersionDType);

    ThorImplementation::ExpressionDefinition definition = ThorImplementation::ExpressionDefinition::fromOutputs(
        ThorImplementation::Expression::outputs({{kMeanGradientName, meanGradient}, {kDispersionGradientName, dispersionGradient}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

}  // namespace

void NegativeBinomialNLLLoss::buildSupportLayersAndAddToNetwork() {
    validateFloatingDType("mean", predictionsTensor.getDataType());
    validateFloatingDType("dispersion", dispersionTensor.getDataType());
    validateLabelsDType(labelsTensor.getDataType());
    validateExampleWeights(predictionsTensor, dispersionTensor, labelsTensor, exampleWeightsTensor);
    THOR_THROW_IF_FALSE(predictionsTensor.getDimensions() == dispersionTensor.getDimensions());
    THOR_THROW_IF_FALSE(predictionsTensor.getDimensions() == labelsTensor.getDimensions());
    THOR_THROW_IF_FALSE(eps > 0.0f);

    MultiInputCustomLoss::Builder builder;
    builder.network(*network)
        .lossExpression(makeNegativeBinomialNLLLossExpression(lossDataType, logMean, logDispersion, eps, exampleWeightsTensor.has_value()))
        .gradientExpression(makeNegativeBinomialNLLGradientExpression(predictionsTensor.getDataType(),
                                                                      dispersionTensor.getDataType(),
                                                                      logMean,
                                                                      logDispersion,
                                                                      eps,
                                                                      exampleWeightsTensor.has_value()))
        .input(kMeanName, predictionsTensor, std::string(kMeanGradientName))
        .input(kDispersionName, dispersionTensor, std::string(kDispersionGradientName))
        .auxiliaryInput(kLabelsName, labelsTensor)
        .lossName(kLossName)
        .lossDataType(lossDataType)
        .lossWeight(lossWeight.value_or(1.0f))
        .reportsRawLoss();
    if (exampleWeightsTensor.has_value())
        builder.auxiliaryInput(kExampleWeightsName, exampleWeightsTensor.value());

    MultiInputCustomLoss rawNegativeBinomialNLLLoss = builder.build();
    lossShaperInput = rawNegativeBinomialNLLLoss.getLoss();
    finalizeLossReporting();
}

json NegativeBinomialNLLLoss::architectureJson() const {
    json j = Loss::architectureJson();
    j["layer_type"] = "negative_binomial_nll_loss";
    j["loss_shape"] = lossShape;
    j["dispersion_tensor"] = dispersionTensor.architectureJson();
    j["log_mean"] = logMean;
    j["log_dispersion"] = logDispersion;
    j["eps"] = eps;
    return j;
}

void NegativeBinomialNLLLoss::deserialize(const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in NegativeBinomialNLLLoss::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "negative_binomial_nll_loss")
        throw runtime_error("Layer type mismatch in NegativeBinomialNLLLoss::deserialize: " + j.at("layer_type").get<std::string>());

    uint64_t originalTensorId = j["predictions_tensor"].at("id").get<uint64_t>();
    Tensor mean = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j["dispersion_tensor"].at("id").get<uint64_t>();
    Tensor dispersion = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j["labels_tensor"].at("id").get<uint64_t>();
    Tensor labels = network->getApiTensorByOriginalId(originalTensorId);

    NegativeBinomialNLLLoss loss;
    loss.lossShape = j.at("loss_shape").get<LossShape>();
    loss.lossDataType = j.at("loss_data_type").get<DataType>();
    loss.lossWeight = ThorImplementation::lossWeightFromJson(j);
    loss.logMean = j.value("log_mean", true);
    loss.logDispersion = j.value("log_dispersion", true);
    loss.eps = j.value("eps", 1.0e-8f);
    loss.predictionsTensor = mean;
    loss.dispersionTensor = dispersion;
    loss.labelsTensor = labels;
    if (j.contains("example_weights_tensor")) {
        originalTensorId = j["example_weights_tensor"].at("id").get<uint64_t>();
        loss.exampleWeightsTensor = network->getApiTensorByOriginalId(originalTensorId);
    }
    loss.network = network;
    loss.initialized = true;
    loss.buildSupportLayersAndAddToNetwork();
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Loss::register_layer("negative_binomial_nll_loss", &Thor::NegativeBinomialNLLLoss::deserialize);
    return true;
}();
}  // namespace
