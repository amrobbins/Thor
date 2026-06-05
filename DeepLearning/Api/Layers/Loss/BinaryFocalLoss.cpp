#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Api/Layers/Loss/BinaryFocalLoss.h"

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
constexpr double kFocalEpsilon = 1.0e-7;

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
            throw runtime_error("Unsupported BinaryFocalLoss label dtype: " + ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
    }
}

void validatePredictionsDType(DataType dtype) {
    if (dtype != DataType::FP16 && dtype != DataType::FP32) {
        throw runtime_error("Unsupported BinaryFocalLoss predictions dtype: " +
                            ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
    }
}

ThorImplementation::Expression binaryCrossEntropyWithLogits(const ThorImplementation::Expression& logits,
                                                            const ThorImplementation::Expression& labels) {
    ThorImplementation::Expression zero(0.0);
    // Numerically stable BCE-with-logits:
    //   max(x, 0) - x * y + log1p(exp(-abs(x)))
    return logits.max(zero) - (logits * labels) + (-logits.abs()).exp().log1p();
}

ThorImplementation::Expression binaryAlphaFactor(const ThorImplementation::Expression& labels, float alpha) {
    ThorImplementation::Expression one(1.0);
    ThorImplementation::Expression alphaExpr(alpha);
    return labels * alphaExpr + (one - labels) * (one - alphaExpr);
}

ThorImplementation::DynamicExpression makeBinaryFocalLossExpression(DataType lossDataType, float gamma, float alpha) {
    validatePredictionsDType(lossDataType);

    ThorImplementation::Expression logits = ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels = ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression one(1.0);
    ThorImplementation::Expression bce = binaryCrossEntropyWithLogits(logits, labels);
    ThorImplementation::Expression alphaFactor = binaryAlphaFactor(labels, alpha);

    ThorImplementation::Expression loss = alphaFactor * bce;
    if (gamma != 0.0f) {
        ThorImplementation::Expression pt = (-bce).exp();
        ThorImplementation::Expression focalWeight = (one - pt).pow(ThorImplementation::Expression(gamma));
        loss = loss * focalWeight;
    }

    loss = loss.withOutputDType(lossDataType);
    ThorImplementation::ExpressionDefinition definition =
        ThorImplementation::ExpressionDefinition::fromOutputs(ThorImplementation::Expression::outputs({{kLossName, loss}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

ThorImplementation::DynamicExpression makeBinaryFocalGradientExpression(DataType predictionsDataType, float gamma, float alpha) {
    validatePredictionsDType(predictionsDataType);

    ThorImplementation::Expression logits = ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels = ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression one(1.0);
    ThorImplementation::Expression bce = binaryCrossEntropyWithLogits(logits, labels);
    ThorImplementation::Expression pt = (-bce).exp();
    ThorImplementation::Expression oneMinusPt = (one - pt).max(ThorImplementation::Expression(kFocalEpsilon));
    ThorImplementation::Expression alphaFactor = binaryAlphaFactor(labels, alpha);
    ThorImplementation::Expression bceGradient = logits.sigmoid() - labels;

    ThorImplementation::Expression gradient = alphaFactor * bceGradient;
    if (gamma != 0.0f) {
        ThorImplementation::Expression focalWeight = oneMinusPt.pow(ThorImplementation::Expression(gamma));
        ThorImplementation::Expression focalDerivativeTerm =
            ThorImplementation::Expression(gamma) * bce * pt * oneMinusPt.pow(ThorImplementation::Expression(gamma - 1.0f));
        gradient = gradient * (focalWeight + focalDerivativeTerm);
    }

    gradient = (gradient * ThorImplementation::Expression(ThorImplementation::Loss::getLossScalingFactor()))
                   .withOutputDType(predictionsDataType);
    ThorImplementation::ExpressionDefinition definition = ThorImplementation::ExpressionDefinition::fromOutputs(
        ThorImplementation::Expression::outputs({{kGradientName, gradient}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

}  // namespace

void BinaryFocalLoss::buildSupportLayersAndAddToNetwork() {
    validatePredictionsDType(predictionsTensor.getDataType());
    validateLabelsDType(labelsTensor.getDataType());
    THOR_THROW_IF_FALSE(gamma >= 0.0f);
    THOR_THROW_IF_FALSE(alpha >= 0.0f && alpha <= 1.0f);

    CustomLoss rawBinaryFocalLoss = CustomLoss::Builder()
                                        .network(*network)
                                        .lossExpression(makeBinaryFocalLossExpression(lossDataType, gamma, alpha))
                                        .gradientExpression(makeBinaryFocalGradientExpression(predictionsTensor.getDataType(), gamma, alpha))
                                        .predictions(predictionsTensor)
                                        .labels(labelsTensor)
                                        .predictionsName(kPredictionsName)
                                        .labelsName(kLabelsName)
                                        .lossName(kLossName)
                                        .gradientName(kGradientName)
                                        .lossDataType(lossDataType)
                                        .reportsRawLoss()
                                        .build();

    lossShaperInput = rawBinaryFocalLoss.getLoss();

    if (lossShape == LossShape::BATCH) {
        LossShaper lossShaper = LossShaper::Builder().network(*network).lossInput(lossShaperInput).reportsBatchLoss().build();
        lossTensor = lossShaper.getLossOutput();
    } else if (lossShape == LossShape::ELEMENTWISE) {
        LossShaper lossShaper = LossShaper::Builder().network(*network).lossInput(lossShaperInput).reportsElementwiseLoss().build();
        lossTensor = lossShaper.getLossOutput();
    } else {
        THOR_THROW_IF_FALSE(lossShape == LossShape::RAW);
        lossTensor = lossShaperInput;
    }
}

json BinaryFocalLoss::architectureJson() const {
    json j = Loss::architectureJson();
    j["loss_shape"] = lossShape;
    j["gamma"] = gamma;
    j["alpha"] = alpha;
    return j;
}

void BinaryFocalLoss::deserialize(const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in BinaryFocalLoss::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "binary_focal_loss")
        throw runtime_error("Layer type mismatch in BinaryFocalLoss::deserialize: " + j.at("layer_type").get<std::string>());

    uint64_t originalTensorId = j["predictions_tensor"].at("id").get<uint64_t>();
    Tensor predictions = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j["labels_tensor"].at("id").get<uint64_t>();
    Tensor labels = network->getApiTensorByOriginalId(originalTensorId);

    BinaryFocalLoss binaryFocalLoss;
    binaryFocalLoss.lossShape = j.at("loss_shape").get<LossShape>();
    binaryFocalLoss.lossDataType = j.at("loss_data_type").get<DataType>();
    binaryFocalLoss.gamma = j.value("gamma", 2.0f);
    binaryFocalLoss.alpha = j.value("alpha", 0.25f);
    binaryFocalLoss.predictionsTensor = predictions;
    binaryFocalLoss.labelsTensor = labels;
    binaryFocalLoss.network = network;
    binaryFocalLoss.initialized = true;
    binaryFocalLoss.buildSupportLayersAndAddToNetwork();
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Loss::register_layer("binary_focal_loss", &Thor::BinaryFocalLoss::deserialize);
    return true;
}();
}  // namespace
