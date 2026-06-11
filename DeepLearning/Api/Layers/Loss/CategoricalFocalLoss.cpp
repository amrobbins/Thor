#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Api/Layers/Loss/CategoricalFocalLoss.h"

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
constexpr double kFocalEpsilon = 1.0e-7;

void validateLabelsDType(DataType dtype) {
    if (dtype != DataType::FP16 && dtype != DataType::FP32) {
        throw runtime_error("Unsupported CategoricalFocalLoss label dtype: " +
                            ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
    }
}

void validatePredictionsDType(DataType dtype) {
    if (dtype != DataType::FP16 && dtype != DataType::FP32) {
        throw runtime_error("Unsupported CategoricalFocalLoss predictions dtype: " +
                            ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
    }
}

ThorImplementation::DynamicExpression makeCategoricalFocalLossExpression(DataType lossDataType, float gamma, float alpha) {
    ThorImplementation::Expression logits = ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels = ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression one(1.0);
    ThorImplementation::Expression probabilities = logits.softmax();
    ThorImplementation::Expression logProbabilities = logits.logSoftmax();
    ThorImplementation::Expression focalWeight = one;
    if (gamma != 0.0f)
        focalWeight = (one - probabilities).pow(ThorImplementation::Expression(gamma));
    ThorImplementation::Expression loss = (-(ThorImplementation::Expression(alpha) * labels * focalWeight * logProbabilities)).withOutputDType(lossDataType);
    ThorImplementation::ExpressionDefinition definition =
        ThorImplementation::ExpressionDefinition::fromOutputs(ThorImplementation::Expression::outputs({{kLossName, loss}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

ThorImplementation::DynamicExpression makeCategoricalFocalGradientExpression(DataType predictionsDataType, float gamma, float alpha) {
    validatePredictionsDType(predictionsDataType);

    ThorImplementation::Expression logits = ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels = ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression one(1.0);
    ThorImplementation::Expression probabilities = logits.softmax();
    ThorImplementation::Expression logProbabilities = logits.logSoftmax();
    ThorImplementation::Expression probabilitySafe = probabilities.max(ThorImplementation::Expression(kFocalEpsilon));
    ThorImplementation::Expression oneMinusProbability = (one - probabilities).max(ThorImplementation::Expression(kFocalEpsilon));

    ThorImplementation::Expression dLossDProbability = -(ThorImplementation::Expression(alpha) * labels / probabilitySafe);
    if (gamma != 0.0f) {
        ThorImplementation::Expression focalWeight = oneMinusProbability.pow(ThorImplementation::Expression(gamma));
        ThorImplementation::Expression focalDerivative =
            ThorImplementation::Expression(gamma) * oneMinusProbability.pow(ThorImplementation::Expression(gamma - 1.0f)) * logProbabilities;
        dLossDProbability = ThorImplementation::Expression(alpha) * labels * (focalDerivative - (focalWeight / probabilitySafe));
    }

    ThorImplementation::Expression weighted = probabilities * dLossDProbability;
    ThorImplementation::Expression weightedSum = weighted.reduce_sum({1}, {});
    ThorImplementation::Expression gradient =
        ((weighted - (probabilities * weightedSum)) * ThorImplementation::Expression(ThorImplementation::Loss::getLossScalingFactor()))
            .withOutputDType(predictionsDataType);
    ThorImplementation::ExpressionDefinition definition = ThorImplementation::ExpressionDefinition::fromOutputs(
        ThorImplementation::Expression::outputs({{kGradientName, gradient}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

}  // namespace

void CategoricalFocalLoss::buildSupportLayersAndAddToNetwork() {
    validatePredictionsDType(predictionsTensor.getDataType());
    validateLabelsDType(labelsTensor.getDataType());
    THOR_THROW_IF_FALSE(gamma >= 0.0f);
    THOR_THROW_IF_FALSE(alpha >= 0.0f);

    CustomLoss rawCategoricalFocalLoss = CustomLoss::Builder()
                                             .network(*network)
                                             .lossExpression(makeCategoricalFocalLossExpression(lossDataType, gamma, alpha))
                                             .gradientExpression(makeCategoricalFocalGradientExpression(predictionsTensor.getDataType(), gamma, alpha))
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

    lossShaperInput = rawCategoricalFocalLoss.getLoss();

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

json CategoricalFocalLoss::architectureJson() const {
    json j = Loss::architectureJson();
    j["loss_shape"] = lossShape;
    j["gamma"] = gamma;
    j["alpha"] = alpha;
    return j;
}

void CategoricalFocalLoss::deserialize(const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in CategoricalFocalLoss::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "categorical_focal_loss")
        throw runtime_error("Layer type mismatch in CategoricalFocalLoss::deserialize: " + j.at("layer_type").get<std::string>());

    uint64_t originalTensorId = j["predictions_tensor"].at("id").get<uint64_t>();
    Tensor predictions = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j["labels_tensor"].at("id").get<uint64_t>();
    Tensor labels = network->getApiTensorByOriginalId(originalTensorId);

    CategoricalFocalLoss categoricalFocalLoss;
    categoricalFocalLoss.lossShape = j.at("loss_shape").get<LossShape>();
    categoricalFocalLoss.lossDataType = j.at("loss_data_type").get<DataType>();

    categoricalFocalLoss.lossWeight = ThorImplementation::lossWeightFromJson(j);
    categoricalFocalLoss.gamma = j.value("gamma", 2.0f);
    categoricalFocalLoss.alpha = j.value("alpha", 1.0f);
    categoricalFocalLoss.predictionsTensor = predictions;
    categoricalFocalLoss.labelsTensor = labels;
    categoricalFocalLoss.network = network;
    categoricalFocalLoss.initialized = true;
    categoricalFocalLoss.buildSupportLayersAndAddToNetwork();
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Loss::register_layer("categorical_focal_loss", &Thor::CategoricalFocalLoss::deserialize);
    return true;
}();
}  // namespace
