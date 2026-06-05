#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Api/Layers/Loss/SmoothL1Loss.h"

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
            throw runtime_error("Unsupported SmoothL1Loss label dtype: " + ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
    }
}

void validatePredictionsDType(DataType dtype) {
    if (dtype != DataType::FP16 && dtype != DataType::FP32) {
        throw runtime_error("Unsupported SmoothL1Loss predictions dtype: " +
                            ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
    }
}

ThorImplementation::Expression signOf(const ThorImplementation::Expression& diff) {
    ThorImplementation::Expression zero(0.0);
    ThorImplementation::Expression positive(1.0);
    ThorImplementation::Expression negative(-1.0);
    return ThorImplementation::Expression::where(diff > zero,
                                                 positive,
                                                 ThorImplementation::Expression::where(diff < zero, negative, zero));
}

ThorImplementation::DynamicExpression makeSmoothL1LossExpression(DataType lossDataType, float beta) {
    ThorImplementation::Expression predictions = ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels = ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression betaExpr(beta);
    ThorImplementation::Expression half(0.5);
    ThorImplementation::Expression diff = predictions - labels;
    ThorImplementation::Expression absDiff = diff.abs();
    ThorImplementation::Expression quadratic = half * diff * diff / betaExpr;
    ThorImplementation::Expression linear = absDiff - (half * betaExpr);
    ThorImplementation::Expression loss = ThorImplementation::Expression::where(absDiff < betaExpr, quadratic, linear).withOutputDType(lossDataType);
    ThorImplementation::ExpressionDefinition definition =
        ThorImplementation::ExpressionDefinition::fromOutputs(ThorImplementation::Expression::outputs({{kLossName, loss}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

ThorImplementation::DynamicExpression makeSmoothL1GradientExpression(DataType predictionsDataType, float beta) {
    validatePredictionsDType(predictionsDataType);

    ThorImplementation::Expression predictions = ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels = ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression betaExpr(beta);
    ThorImplementation::Expression diff = predictions - labels;
    ThorImplementation::Expression absDiff = diff.abs();
    ThorImplementation::Expression quadraticGrad = diff / betaExpr;
    ThorImplementation::Expression linearGrad = signOf(diff);
    ThorImplementation::Expression gradient =
        (ThorImplementation::Expression::where(absDiff < betaExpr, quadraticGrad, linearGrad) *
         ThorImplementation::Expression(ThorImplementation::Loss::getLossScalingFactor()))
            .withOutputDType(predictionsDataType);
    ThorImplementation::ExpressionDefinition definition = ThorImplementation::ExpressionDefinition::fromOutputs(
        ThorImplementation::Expression::outputs({{kGradientName, gradient}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

}  // namespace

void SmoothL1Loss::buildSupportLayersAndAddToNetwork() {
    validatePredictionsDType(predictionsTensor.getDataType());
    validateLabelsDType(labelsTensor.getDataType());
    THOR_THROW_IF_FALSE(beta > 0.0f);

    CustomLoss rawSmoothL1Loss = CustomLoss::Builder()
                                      .network(*network)
                                      .lossExpression(makeSmoothL1LossExpression(lossDataType, beta))
                                      .gradientExpression(makeSmoothL1GradientExpression(predictionsTensor.getDataType(), beta))
                                      .predictions(predictionsTensor)
                                      .labels(labelsTensor)
                                      .predictionsName(kPredictionsName)
                                      .labelsName(kLabelsName)
                                      .lossName(kLossName)
                                      .gradientName(kGradientName)
                                      .lossDataType(lossDataType)
                                      .reportsRawLoss()
                                      .build();

    lossShaperInput = rawSmoothL1Loss.getLoss();

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

json SmoothL1Loss::architectureJson() const {
    json j = Loss::architectureJson();
    j["loss_shape"] = lossShape;
    j["beta"] = beta;
    return j;
}

void SmoothL1Loss::deserialize(const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in SmoothL1Loss::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "smooth_l1_loss")
        throw runtime_error("Layer type mismatch in SmoothL1Loss::deserialize: " + j.at("layer_type").get<std::string>());

    uint64_t originalTensorId = j["predictions_tensor"].at("id").get<uint64_t>();
    Tensor predictions = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j["labels_tensor"].at("id").get<uint64_t>();
    Tensor labels = network->getApiTensorByOriginalId(originalTensorId);

    SmoothL1Loss smoothL1Loss;
    smoothL1Loss.lossShape = j.at("loss_shape").get<LossShape>();
    smoothL1Loss.lossDataType = j.at("loss_data_type").get<DataType>();
    smoothL1Loss.beta = j.value("beta", 1.0f);
    smoothL1Loss.predictionsTensor = predictions;
    smoothL1Loss.labelsTensor = labels;
    smoothL1Loss.network = network;
    smoothL1Loss.initialized = true;
    smoothL1Loss.buildSupportLayersAndAddToNetwork();
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Loss::register_layer("smooth_l1_loss", &Thor::SmoothL1Loss::deserialize);
    return true;
}();
}  // namespace
