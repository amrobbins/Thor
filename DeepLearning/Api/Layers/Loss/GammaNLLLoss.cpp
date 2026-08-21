#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Api/Layers/Loss/GammaNLLLoss.h"
#include "DeepLearning/Api/Layers/Loss/CustomLoss.h"

#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/Expression.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {
namespace {

// Preserve the legacy CustomLoss support-graph port names for the original
// two-input GammaNLLLoss path. Publicly this tensor is still the distribution mean.
constexpr const char* kMeanName = "predictions";
constexpr const char* kLabelsName = "labels";
constexpr const char* kDispersionName = "dispersion";
constexpr const char* kExampleWeightsName = "example_weights";
constexpr const char* kLossName = "loss";
constexpr const char* kMeanGradientName = "gradient";
constexpr const char* kDispersionGradientName = "dispersion_grad";

void validateFloatingDType(const char* tensorName, DataType dtype) {
    if (dtype != DataType::FP16 && dtype != DataType::FP32) {
        throw runtime_error(string("Unsupported GammaNLLLoss ") + tensorName + " dtype: " +
                            ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
    }
}

void validateExampleWeights(Tensor mean,
                            Tensor labels,
                            std::optional<Tensor> dispersion,
                            std::optional<Tensor> exampleWeights) {
    if (!exampleWeights.has_value())
        return;
    if (exampleWeights.value() == mean || exampleWeights.value() == labels ||
        (dispersion.has_value() && exampleWeights.value() == dispersion.value()))
        throw runtime_error("GammaNLLLoss example_weights tensor must be distinct from mean, labels, and dispersion.");
    validateFloatingDType("example_weights", exampleWeights.value().getDataType());
    const vector<uint64_t>& dims = exampleWeights.value().getDimensions();
    if (dims != vector<uint64_t>{1} && dims != mean.getDimensions()) {
        throw runtime_error("GammaNLLLoss example_weights dimensions must be [1] for per-example weights or match mean dimensions.");
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

ThorImplementation::Expression gammaLoss(const ThorImplementation::Expression& meanInput,
                                         const ThorImplementation::Expression& labels,
                                         const std::optional<ThorImplementation::Expression>& dispersionInput,
                                         bool logMean,
                                         bool logDispersion,
                                         float eps) {
    ThorImplementation::Expression mean = positiveValue(meanInput, logMean, eps);
    ThorImplementation::Expression logMeanValue = logPositiveValue(meanInput, logMean, eps);

    if (!dispersionInput.has_value())
        return logMeanValue + labels / mean;

    ThorImplementation::Expression dispersion = positiveValue(dispersionInput.value(), logDispersion, eps);
    ThorImplementation::Expression logDispersionValue = logPositiveValue(dispersionInput.value(), logDispersion, eps);
    ThorImplementation::Expression concentration = ThorImplementation::Expression(1.0) / dispersion;
    ThorImplementation::Expression safeTarget = safePositive(labels, eps);
    ThorImplementation::Expression logTarget = safeTarget.ln();

    return concentration.lgamma() + concentration * (logMeanValue + logDispersionValue) -
           (concentration - ThorImplementation::Expression(1.0)) * logTarget +
           labels / (mean * dispersion);
}

ThorImplementation::DynamicExpression makeGammaNLLLossExpression(DataType lossDataType,
                                                                 bool hasDispersion,
                                                                 bool logMean,
                                                                 bool logDispersion,
                                                                 float eps,
                                                                 bool weighted) {
    validateFloatingDType("loss", lossDataType);

    ThorImplementation::Expression mean = ThorImplementation::Expression::input(kMeanName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels = ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    std::optional<ThorImplementation::Expression> dispersion;
    if (hasDispersion)
        dispersion = ThorImplementation::Expression::input(kDispersionName, DataType::FP32, DataType::FP32);

    ThorImplementation::Expression loss = gammaLoss(mean, labels, dispersion, logMean, logDispersion, eps);
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

ThorImplementation::DynamicExpression makeGammaNLLGradientExpression(DataType meanDType,
                                                                     std::optional<DataType> dispersionDType,
                                                                     bool logMean,
                                                                     bool logDispersion,
                                                                     float eps,
                                                                     bool weighted) {
    validateFloatingDType("mean", meanDType);
    if (dispersionDType.has_value())
        validateFloatingDType("dispersion", dispersionDType.value());

    ThorImplementation::Expression meanInput = ThorImplementation::Expression::input(kMeanName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels = ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression mean = positiveValue(meanInput, logMean, eps);
    ThorImplementation::Expression scale(ThorImplementation::Loss::getLossScalingFactor());

    ThorImplementation::Expression meanGradient(0.0);
    std::optional<ThorImplementation::Expression> dispersionGradient;

    if (!dispersionDType.has_value()) {
        meanGradient = logMean ? (ThorImplementation::Expression(1.0) - labels / mean)
                               : (ThorImplementation::Expression(1.0) / mean - labels / (mean * mean));
    } else {
        ThorImplementation::Expression dispersionInput =
            ThorImplementation::Expression::input(kDispersionName, DataType::FP32, DataType::FP32);
        ThorImplementation::Expression dispersion = positiveValue(dispersionInput, logDispersion, eps);
        ThorImplementation::Expression logMeanValue = logPositiveValue(meanInput, logMean, eps);
        ThorImplementation::Expression logDispersionValue = logPositiveValue(dispersionInput, logDispersion, eps);
        ThorImplementation::Expression concentration = ThorImplementation::Expression(1.0) / dispersion;
        ThorImplementation::Expression safeTarget = safePositive(labels, eps);
        ThorImplementation::Expression logTarget = safeTarget.ln();

        ThorImplementation::Expression directMeanGradient =
            concentration * (ThorImplementation::Expression(1.0) / mean - labels / (mean * mean));
        meanGradient = logMean ? directMeanGradient * mean : directMeanGradient;

        ThorImplementation::Expression derivativeWrtConcentration = concentration.digamma() + logMeanValue + logDispersionValue -
                                                                     ThorImplementation::Expression(1.0) - logTarget + labels / mean;
        ThorImplementation::Expression logDispersionGradient =
            (ThorImplementation::Expression(0.0) - concentration) * derivativeWrtConcentration;
        dispersionGradient = logDispersion ? logDispersionGradient : logDispersionGradient / dispersion;
    }

    if (weighted) {
        ThorImplementation::Expression exampleWeights =
            ThorImplementation::Expression::input(kExampleWeightsName, DataType::FP32, DataType::FP32);
        meanGradient = meanGradient * exampleWeights;
        if (dispersionGradient.has_value())
            dispersionGradient = dispersionGradient.value() * exampleWeights;
    }

    meanGradient = (meanGradient * scale).withOutputDType(meanDType);
    std::vector<std::pair<std::string, ThorImplementation::Expression>> outputs{{kMeanGradientName, meanGradient}};
    if (dispersionGradient.has_value()) {
        outputs.emplace_back(kDispersionGradientName,
                             (dispersionGradient.value() * scale).withOutputDType(dispersionDType.value()));
    }

    ThorImplementation::ExpressionDefinition definition =
        ThorImplementation::ExpressionDefinition::fromOutputs(ThorImplementation::Expression::outputs(outputs));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

}  // namespace

void GammaNLLLoss::buildSupportLayersAndAddToNetwork() {
    validateFloatingDType("mean", predictionsTensor.getDataType());
    validateFloatingDType("labels", labelsTensor.getDataType());
    if (dispersionTensor.has_value()) {
        validateFloatingDType("dispersion", dispersionTensor.value().getDataType());
        THOR_THROW_IF_FALSE(dispersionTensor.value().getDimensions() == predictionsTensor.getDimensions());
    }
    validateExampleWeights(predictionsTensor, labelsTensor, dispersionTensor, exampleWeightsTensor);
    THOR_THROW_IF_FALSE(eps > 0.0f);

    if (dispersionTensor.has_value() || exampleWeightsTensor.has_value()) {
        MultiInputCustomLoss::Builder builder;
        builder.network(*network)
            .lossExpression(makeGammaNLLLossExpression(lossDataType,
                                                       dispersionTensor.has_value(),
                                                       logMean,
                                                       logDispersion,
                                                       eps,
                                                       exampleWeightsTensor.has_value()))
            .gradientExpression(makeGammaNLLGradientExpression(predictionsTensor.getDataType(),
                                                               dispersionTensor.has_value()
                                                                   ? std::optional<DataType>(dispersionTensor.value().getDataType())
                                                                   : std::nullopt,
                                                               logMean,
                                                               logDispersion,
                                                               eps,
                                                               exampleWeightsTensor.has_value()))
            .input(kMeanName, predictionsTensor, std::string(kMeanGradientName))
            .auxiliaryInput(kLabelsName, labelsTensor)
            .lossName(kLossName)
            .lossDataType(lossDataType)
            .lossWeight(lossWeight.value_or(1.0f))
            .reportsRawLoss();
        if (dispersionTensor.has_value())
            builder.input(kDispersionName, dispersionTensor.value(), std::string(kDispersionGradientName));
        if (exampleWeightsTensor.has_value())
            builder.auxiliaryInput(kExampleWeightsName, exampleWeightsTensor.value());
        MultiInputCustomLoss rawGammaNLLLoss = builder.build();
        lossShaperInput = rawGammaNLLLoss.getLoss();
    } else {
        CustomLoss rawGammaNLLLoss = CustomLoss::Builder()
                                         .network(*network)
                                         .lossExpression(makeGammaNLLLossExpression(lossDataType, false, logMean, false, eps, false))
                                         .gradientExpression(
                                             makeGammaNLLGradientExpression(predictionsTensor.getDataType(), std::nullopt, logMean, false, eps, false))
                                         .predictions(predictionsTensor)
                                         .labels(labelsTensor)
                                         .predictionsName(kMeanName)
                                         .labelsName(kLabelsName)
                                         .lossName(kLossName)
                                         .gradientName(kMeanGradientName)
                                         .lossDataType(lossDataType)
                                         .lossWeight(lossWeight.value_or(1.0f))
                                         .reportsRawLoss()
                                         .build();
        lossShaperInput = rawGammaNLLLoss.getLoss();
    }

    finalizeLossReporting();
}

json GammaNLLLoss::architectureJson() const {
    json j = Loss::architectureJson();
    j["layer_type"] = "gamma_nll_loss";
    j["loss_shape"] = lossShape;
    j["log_mean"] = logMean;
    j["log_dispersion"] = logDispersion;
    if (dispersionTensor.has_value())
        j["dispersion_tensor"] = dispersionTensor.value().architectureJson();
    j["eps"] = eps;
    return j;
}

void GammaNLLLoss::deserialize(const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in GammaNLLLoss::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "gamma_nll_loss")
        throw runtime_error("Layer type mismatch in GammaNLLLoss::deserialize: " + j.at("layer_type").get<std::string>());

    uint64_t originalTensorId = j["predictions_tensor"].at("id").get<uint64_t>();
    Tensor predictions = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j["labels_tensor"].at("id").get<uint64_t>();
    Tensor labels = network->getApiTensorByOriginalId(originalTensorId);

    GammaNLLLoss loss;
    loss.lossShape = j.at("loss_shape").get<LossShape>();
    loss.lossDataType = j.at("loss_data_type").get<DataType>();

    loss.lossWeight = ThorImplementation::lossWeightFromJson(j);
    loss.logMean = j.value("log_mean", false);
    loss.logDispersion = j.value("log_dispersion", false);
    loss.eps = j.value("eps", 1.0e-6f);
    loss.predictionsTensor = predictions;
    loss.labelsTensor = labels;
    if (j.contains("dispersion_tensor")) {
        originalTensorId = j["dispersion_tensor"].at("id").get<uint64_t>();
        loss.dispersionTensor = network->getApiTensorByOriginalId(originalTensorId);
    }
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
    Thor::Loss::register_layer("gamma_nll_loss", &Thor::GammaNLLLoss::deserialize);
    return true;
}();
}  // namespace
