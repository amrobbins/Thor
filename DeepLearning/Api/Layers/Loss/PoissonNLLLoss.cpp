#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Api/Layers/Loss/PoissonNLLLoss.h"

#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/Expression.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {
namespace {

constexpr const char* kPredictionsName = "predictions";
constexpr const char* kLabelsName = "labels";
constexpr const char* kLossName = "loss";
constexpr const char* kGradientName = "predictions_grad";
constexpr double kTwoPi = 6.283185307179586476925286766559;

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
            throw runtime_error("Unsupported PoissonNLLLoss label dtype: " + ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
    }
}

void validatePredictionsDType(DataType dtype) {
    if (dtype != DataType::FP16 && dtype != DataType::FP32) {
        throw runtime_error("Unsupported PoissonNLLLoss predictions dtype: " +
                            ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
    }
}

ThorImplementation::Expression poissonStirlingTerm(const ThorImplementation::Expression& labels) {
    ThorImplementation::Expression zero(0.0);
    ThorImplementation::Expression one(1.0);
    ThorImplementation::Expression half(0.5);
    ThorImplementation::Expression safeLabels = labels.max(one);
    ThorImplementation::Expression stirling = labels * safeLabels.ln() - labels + half * (safeLabels * ThorImplementation::Expression(kTwoPi)).ln();
    return ThorImplementation::Expression::where(labels > one, stirling, zero);
}

ThorImplementation::DynamicExpression makePoissonNLLLossExpression(DataType lossDataType, bool logInput, bool full, float eps) {
    ThorImplementation::Expression predictions = ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels = ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);

    ThorImplementation::Expression loss = logInput ? (predictions.exp() - labels * predictions)
                                                   : (predictions - labels * (predictions + ThorImplementation::Expression(eps)).ln());
    if (full)
        loss = loss + poissonStirlingTerm(labels);
    loss = loss.withOutputDType(lossDataType);

    ThorImplementation::ExpressionDefinition definition =
        ThorImplementation::ExpressionDefinition::fromOutputs(ThorImplementation::Expression::outputs({{kLossName, loss}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

ThorImplementation::DynamicExpression makePoissonNLLGradientExpression(DataType predictionsDataType, bool logInput, float eps) {
    validatePredictionsDType(predictionsDataType);

    ThorImplementation::Expression predictions = ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels = ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression gradient = logInput ? (predictions.exp() - labels)
                                                       : (ThorImplementation::Expression(1.0) -
                                                          labels / (predictions + ThorImplementation::Expression(eps)));
    gradient = (gradient * ThorImplementation::Expression(ThorImplementation::Loss::getLossScalingFactor()))
                   .withOutputDType(predictionsDataType);

    ThorImplementation::ExpressionDefinition definition = ThorImplementation::ExpressionDefinition::fromOutputs(
        ThorImplementation::Expression::outputs({{kGradientName, gradient}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

}  // namespace

void PoissonNLLLoss::buildSupportLayersAndAddToNetwork() {
    validatePredictionsDType(predictionsTensor.getDataType());
    validateLabelsDType(labelsTensor.getDataType());
    THOR_THROW_IF_FALSE(eps > 0.0f);

    CustomLoss rawPoissonNLLLoss = CustomLoss::Builder()
                                       .network(*network)
                                       .lossExpression(makePoissonNLLLossExpression(lossDataType, logInput, full, eps))
                                       .gradientExpression(makePoissonNLLGradientExpression(predictionsTensor.getDataType(), logInput, eps))
                                       .predictions(predictionsTensor)
                                       .labels(labelsTensor)
                                       .predictionsName(kPredictionsName)
                                       .labelsName(kLabelsName)
                                       .lossName(kLossName)
                                       .gradientName(kGradientName)
                                       .lossDataType(lossDataType)
                                       .reportsRawLoss()
                                       .build();

    lossShaperInput = rawPoissonNLLLoss.getLoss();

    if (lossShape == LossShape::BATCH) {
        LossShaper lossShaper = LossShaper::Builder().network(*network).lossInput(lossShaperInput).reportsBatchLoss().build();
        lossTensor = lossShaper.getLossOutput();
    } else if (lossShape == LossShape::ELEMENTWISE) {
        LossShaper lossShaper = LossShaper::Builder().network(*network).lossInput(lossShaperInput).reportsElementwiseLoss().build();
        lossTensor = lossShaper.getLossOutput();
    } else if (lossShape == LossShape::CLASSWISE) {
        LossShaper lossShaper = LossShaper::Builder().network(*network).lossInput(lossShaperInput).reportsClasswiseLoss().build();
        lossTensor = lossShaper.getLossOutput();
    } else {
        THOR_THROW_IF_FALSE(lossShape == LossShape::RAW);
        lossTensor = lossShaperInput;
    }
}

json PoissonNLLLoss::architectureJson() const {
    json j = Loss::architectureJson();
    j["layer_type"] = "poisson_nll_loss";
    j["loss_shape"] = lossShape;
    j["log_input"] = logInput;
    j["full"] = full;
    j["eps"] = eps;
    return j;
}

void PoissonNLLLoss::deserialize(const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in PoissonNLLLoss::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "poisson_nll_loss")
        throw runtime_error("Layer type mismatch in PoissonNLLLoss::deserialize: " + j.at("layer_type").get<std::string>());

    uint64_t originalTensorId = j["predictions_tensor"].at("id").get<uint64_t>();
    Tensor predictions = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j["labels_tensor"].at("id").get<uint64_t>();
    Tensor labels = network->getApiTensorByOriginalId(originalTensorId);

    PoissonNLLLoss loss;
    loss.lossShape = j.at("loss_shape").get<LossShape>();
    loss.lossDataType = j.at("loss_data_type").get<DataType>();
    loss.logInput = j.value("log_input", true);
    loss.full = j.value("full", false);
    loss.eps = j.value("eps", 1.0e-8f);
    loss.predictionsTensor = predictions;
    loss.labelsTensor = labels;
    loss.network = network;
    loss.initialized = true;
    loss.buildSupportLayersAndAddToNetwork();
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Loss::register_layer("poisson_nll_loss", &Thor::PoissonNLLLoss::deserialize);
    return true;
}();
}  // namespace
