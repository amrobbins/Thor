#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Api/Layers/Loss/GammaNLLLoss.h"

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

void validateFloatingDType(const char* tensorName, DataType dtype) {
    if (dtype != DataType::FP16 && dtype != DataType::FP32) {
        throw runtime_error(string("Unsupported GammaNLLLoss ") + tensorName + " dtype: " +
                            ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
    }
}

ThorImplementation::Expression safePositive(const ThorImplementation::Expression& value, float eps) {
    return value.max(ThorImplementation::Expression(eps));
}

ThorImplementation::DynamicExpression makeGammaNLLLossExpression(DataType lossDataType, float eps) {
    validateFloatingDType("loss", lossDataType);

    ThorImplementation::Expression predictions = ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels = ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression mean = safePositive(predictions, eps);

    ThorImplementation::Expression loss = (mean.ln() + labels / mean).withOutputDType(lossDataType);

    ThorImplementation::ExpressionDefinition definition =
        ThorImplementation::ExpressionDefinition::fromOutputs(ThorImplementation::Expression::outputs({{kLossName, loss}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

ThorImplementation::DynamicExpression makeGammaNLLGradientExpression(DataType predictionsDataType, float eps) {
    validateFloatingDType("predictions", predictionsDataType);

    ThorImplementation::Expression predictions = ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels = ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression mean = safePositive(predictions, eps);
    ThorImplementation::Expression scale(ThorImplementation::Loss::getLossScalingFactor());

    ThorImplementation::Expression gradient =
        ((ThorImplementation::Expression(1.0) / mean - labels / (mean * mean)) * scale).withOutputDType(predictionsDataType);

    ThorImplementation::ExpressionDefinition definition = ThorImplementation::ExpressionDefinition::fromOutputs(
        ThorImplementation::Expression::outputs({{kGradientName, gradient}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

}  // namespace

void GammaNLLLoss::buildSupportLayersAndAddToNetwork() {
    validateFloatingDType("predictions", predictionsTensor.getDataType());
    validateFloatingDType("labels", labelsTensor.getDataType());
    THOR_THROW_IF_FALSE(eps > 0.0f);

    CustomLoss rawGammaNLLLoss = CustomLoss::Builder()
                                      .network(*network)
                                      .lossExpression(makeGammaNLLLossExpression(lossDataType, eps))
                                      .gradientExpression(makeGammaNLLGradientExpression(predictionsTensor.getDataType(), eps))
                                      .predictions(predictionsTensor)
                                      .labels(labelsTensor)
                                      .predictionsName(kPredictionsName)
                                      .labelsName(kLabelsName)
                                      .lossName(kLossName)
                                      .gradientName(kGradientName)
                                      .lossDataType(lossDataType)
                                       .lossWeight(lossWeight.value_or(1.0f))
                                      .reportsRawLoss()
                                      .build();

    lossShaperInput = rawGammaNLLLoss.getLoss();

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

json GammaNLLLoss::architectureJson() const {
    json j = Loss::architectureJson();
    j["layer_type"] = "gamma_nll_loss";
    j["loss_shape"] = lossShape;
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
    loss.eps = j.value("eps", 1.0e-6f);
    loss.predictionsTensor = predictions;
    loss.labelsTensor = labels;
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
