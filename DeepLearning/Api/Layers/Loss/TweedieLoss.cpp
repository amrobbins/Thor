#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Api/Layers/Loss/TweedieLoss.h"

#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/Expression.h"

#include <cmath>

using namespace std;
using json = nlohmann::json;

namespace Thor {
namespace {

constexpr const char* kPredictionsName = "predictions";
constexpr const char* kLabelsName = "labels";
constexpr const char* kLossName = "loss";
constexpr const char* kGradientName = "predictions_grad";
constexpr float kSpecialPowerTolerance = 1.0e-6f;

void validateFloatingDType(const char* tensorName, DataType dtype) {
    if (dtype != DataType::FP16 && dtype != DataType::FP32) {
        throw runtime_error(string("Unsupported TweedieLoss ") + tensorName + " dtype: " +
                            ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
    }
}

bool isSpecialPower(float power, float special) { return std::fabs(power - special) <= kSpecialPowerTolerance; }

ThorImplementation::Expression safePositive(const ThorImplementation::Expression& value, float eps) {
    return value.max(ThorImplementation::Expression(eps));
}

ThorImplementation::DynamicExpression makeTweedieLossExpression(DataType lossDataType, float power, float eps) {
    validateFloatingDType("loss", lossDataType);
    THOR_THROW_IF_FALSE(std::isfinite(power));

    ThorImplementation::Expression predictions = ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels = ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression mean = safePositive(predictions, eps);
    ThorImplementation::Expression target = labels.max(ThorImplementation::Expression(0.0));
    ThorImplementation::Expression safeTarget = safePositive(target, eps);

    ThorImplementation::Expression two(2.0);
    ThorImplementation::Expression loss = [&]() -> ThorImplementation::Expression {
        if (isSpecialPower(power, 0.0f)) {
            ThorImplementation::Expression diff = target - mean;
            return diff * diff;
        }
        if (isSpecialPower(power, 1.0f)) {
            return two * (target * (safeTarget / mean).ln() - target + mean);
        }
        if (isSpecialPower(power, 2.0f)) {
            return two * ((mean / safeTarget).ln() + target / mean - ThorImplementation::Expression(1.0));
        }

        ThorImplementation::Expression p(power);
        ThorImplementation::Expression one(1.0);
        ThorImplementation::Expression twoMinusP = two - p;
        ThorImplementation::Expression oneMinusP = one - p;
        return two * (safeTarget.pow(twoMinusP) / (oneMinusP * twoMinusP) -
                      target * mean.pow(oneMinusP) / oneMinusP + mean.pow(twoMinusP) / twoMinusP);
    }();
    loss = loss.withOutputDType(lossDataType);

    ThorImplementation::ExpressionDefinition definition =
        ThorImplementation::ExpressionDefinition::fromOutputs(ThorImplementation::Expression::outputs({{kLossName, loss}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

ThorImplementation::DynamicExpression makeTweedieGradientExpression(DataType predictionsDataType, float power, float eps) {
    validateFloatingDType("predictions", predictionsDataType);
    THOR_THROW_IF_FALSE(std::isfinite(power));

    ThorImplementation::Expression predictions = ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels = ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression mean = safePositive(predictions, eps);
    ThorImplementation::Expression target = labels.max(ThorImplementation::Expression(0.0));
    ThorImplementation::Expression p(power);
    ThorImplementation::Expression two(2.0);
    ThorImplementation::Expression scale(ThorImplementation::Loss::getLossScalingFactor());

    ThorImplementation::Expression gradient =
        two * (mean.pow(ThorImplementation::Expression(1.0) - p) - target * mean.pow(ThorImplementation::Expression(0.0) - p));
    gradient = (gradient * scale).withOutputDType(predictionsDataType);

    ThorImplementation::ExpressionDefinition definition = ThorImplementation::ExpressionDefinition::fromOutputs(
        ThorImplementation::Expression::outputs({{kGradientName, gradient}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

}  // namespace

void TweedieLoss::buildSupportLayersAndAddToNetwork() {
    validateFloatingDType("predictions", predictionsTensor.getDataType());
    validateFloatingDType("labels", labelsTensor.getDataType());
    THOR_THROW_IF_FALSE(std::isfinite(power));
    THOR_THROW_IF_FALSE(eps > 0.0f);

    CustomLoss rawTweedieLoss = CustomLoss::Builder()
                                    .network(*network)
                                    .lossExpression(makeTweedieLossExpression(lossDataType, power, eps))
                                    .gradientExpression(makeTweedieGradientExpression(predictionsTensor.getDataType(), power, eps))
                                    .predictions(predictionsTensor)
                                    .labels(labelsTensor)
                                    .predictionsName(kPredictionsName)
                                    .labelsName(kLabelsName)
                                    .lossName(kLossName)
                                    .gradientName(kGradientName)
                                    .lossDataType(lossDataType)
                                    .reportsRawLoss()
                                    .build();

    lossShaperInput = rawTweedieLoss.getLoss();

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

json TweedieLoss::architectureJson() const {
    json j = Loss::architectureJson();
    j["layer_type"] = "tweedie_loss";
    j["loss_shape"] = lossShape;
    j["power"] = power;
    j["eps"] = eps;
    return j;
}

void TweedieLoss::deserialize(const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in TweedieLoss::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "tweedie_loss")
        throw runtime_error("Layer type mismatch in TweedieLoss::deserialize: " + j.at("layer_type").get<std::string>());

    uint64_t originalTensorId = j["predictions_tensor"].at("id").get<uint64_t>();
    Tensor predictions = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j["labels_tensor"].at("id").get<uint64_t>();
    Tensor labels = network->getApiTensorByOriginalId(originalTensorId);

    TweedieLoss loss;
    loss.lossShape = j.at("loss_shape").get<LossShape>();
    loss.lossDataType = j.at("loss_data_type").get<DataType>();
    loss.power = j.value("power", 1.5f);
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
    Thor::Loss::register_layer("tweedie_loss", &Thor::TweedieLoss::deserialize);
    return true;
}();
}  // namespace
