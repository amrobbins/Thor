#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Api/Layers/Loss/TweedieLoss.h"
#include "DeepLearning/Api/Layers/Loss/MultiInputCustomLoss.h"
#include "DeepLearning/Api/Layers/Loss/RaggedCustomLoss.h"
#include "DeepLearning/Api/Layers/Loss/RaggedLossShaper.h"
#include "DeepLearning/Api/Layers/Utility/SegmentedBroadcast.h"
#include "DeepLearning/Api/Layers/Utility/SegmentedPrimitiveCommon.h"
#include "DeepLearning/Api/Layers/Utility/Stub.h"
#include "DeepLearning/Api/Layers/Utility/TypeConverter.h"

#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/Expression.h"

#include <cmath>

using namespace std;
using json = nlohmann::json;

namespace Thor {
namespace {

constexpr const char* kPredictionsName = "predictions";
constexpr const char* kLabelsName = "labels";
constexpr const char* kExampleWeightsName = "example_weights";
constexpr const char* kLossName = "loss";
constexpr const char* kGradientName = "predictions_grad";
constexpr float kSpecialPowerTolerance = 1.0e-6f;

void validateFloatingDType(const char* tensorName, DataType dtype) {
    if (dtype != DataType::FP16 && dtype != DataType::FP32) {
        throw runtime_error(string("Unsupported TweedieLoss ") + tensorName + " dtype: " +
                            ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
    }
}

void validateExampleWeights(Tensor predictions, Tensor labels, std::optional<Tensor> exampleWeights) {
    if (!exampleWeights.has_value())
        return;
    if (exampleWeights.value() == predictions || exampleWeights.value() == labels)
        throw runtime_error("TweedieLoss example_weights tensor must be distinct from predictions and labels.");
    validateFloatingDType("example_weights", exampleWeights.value().getDataType());
    const vector<uint64_t>& dims = exampleWeights.value().getDimensions();
    if (dims != vector<uint64_t>{1} && dims != predictions.getDimensions()) {
        throw runtime_error(
            "TweedieLoss example_weights dimensions must be [1] for per-example weights or match predictions dimensions.");
    }
}

bool isSpecialPower(float power, float special) { return std::fabs(power - special) <= kSpecialPowerTolerance; }

ThorImplementation::Expression safePositive(const ThorImplementation::Expression& value, float eps) {
    return value.max(ThorImplementation::Expression(eps));
}

vector<uint64_t> raggedPackedScalarWeightBroadcastDimensions(const vector<uint64_t>& predictionValueDimensions) {
    if (predictionValueDimensions.empty()) throw invalid_argument("TweedieLoss ragged prediction values must have a packed-capacity dimension.");
    vector<uint64_t> dimensions(predictionValueDimensions.size(), 1);
    dimensions.front() = predictionValueDimensions.front();
    return dimensions;
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

ThorImplementation::DynamicExpression makeWeightedTweedieLossExpression(DataType lossDataType, float power, float eps, optional<vector<uint64_t>> raggedPredictionValueDimensions = nullopt) {
    validateFloatingDType("loss", lossDataType);
    THOR_THROW_IF_FALSE(std::isfinite(power));

    ThorImplementation::Expression predictions = ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels = ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression exampleWeights =
        ThorImplementation::Expression::input(kExampleWeightsName, DataType::FP32, DataType::FP32);
    if (raggedPredictionValueDimensions.has_value())
        exampleWeights = exampleWeights.reshape(raggedPackedScalarWeightBroadcastDimensions(raggedPredictionValueDimensions.value()));
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
    loss = (loss * exampleWeights).withOutputDType(lossDataType);

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

ThorImplementation::DynamicExpression makeWeightedTweedieGradientExpression(DataType predictionsDataType, float power, float eps, optional<vector<uint64_t>> raggedPredictionValueDimensions = nullopt) {
    validateFloatingDType("predictions", predictionsDataType);
    THOR_THROW_IF_FALSE(std::isfinite(power));

    ThorImplementation::Expression predictions = ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels = ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression exampleWeights =
        ThorImplementation::Expression::input(kExampleWeightsName, DataType::FP32, DataType::FP32);
    if (raggedPredictionValueDimensions.has_value())
        exampleWeights = exampleWeights.reshape(raggedPackedScalarWeightBroadcastDimensions(raggedPredictionValueDimensions.value()));
    ThorImplementation::Expression mean = safePositive(predictions, eps);
    ThorImplementation::Expression target = labels.max(ThorImplementation::Expression(0.0));
    ThorImplementation::Expression p(power);
    ThorImplementation::Expression two(2.0);
    ThorImplementation::Expression scale(ThorImplementation::Loss::getLossScalingFactor());

    ThorImplementation::Expression gradient =
        two * (mean.pow(ThorImplementation::Expression(1.0) - p) -
               target * mean.pow(ThorImplementation::Expression(0.0) - p));
    gradient = (gradient * exampleWeights * scale).withOutputDType(predictionsDataType);

    ThorImplementation::ExpressionDefinition definition = ThorImplementation::ExpressionDefinition::fromOutputs(
        ThorImplementation::Expression::outputs({{kGradientName, gradient}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

}  // namespace

void TweedieLoss::buildSupportLayersAndAddToNetwork() {
    validateFloatingDType("predictions", predictionsTensor.getDataType());
    validateFloatingDType("labels", labelsTensor.getDataType());
    validateExampleWeights(predictionsTensor, labelsTensor, exampleWeightsTensor);
    THOR_THROW_IF_FALSE(std::isfinite(power));
    THOR_THROW_IF_FALSE(eps > 0.0f);

    if (isRagged()) {
        if (lossShape == LossShape::PER_OUTPUT) throw invalid_argument("TweedieLoss LossShape::PER_OUTPUT is undefined for ragged sequences.");
        RaggedCustomLoss::Builder rawBuilder;
        rawBuilder.network(*network).predictions(raggedPredictionsTensor.value()).labels(raggedLabelsTensor.value())
            .predictionsName(kPredictionsName).labelsName(kLabelsName).lossName(kLossName).gradientName(kGradientName)
            .lossDataType(lossDataType).lossWeight(lossWeight.value_or(1.0f));
        if (exampleWeightsTensor.has_value()) {
            if (exampleWeightsTensor->getDimensions() != vector<uint64_t>{1}) throw invalid_argument("TweedieLoss ragged example_weights must have dimensions [1] for one scalar weight per logical row.");
            TypeConverter weightConverter = TypeConverter::Builder().network(*network).featureInput(exampleWeightsTensor.value()).newDataType(DataType::FP32).build();
            SegmentedBroadcast weightBroadcast = SegmentedBroadcast::Builder().network(*network).featureInput(weightConverter.getFeatureOutput().value()).partitionInput(raggedPredictionsTensor.value()).build();
            const vector<uint64_t> dims = raggedPredictionsTensor->getValuesDimensions();
            rawBuilder.lossExpression(makeWeightedTweedieLossExpression(lossDataType, power, eps, dims))
                .gradientExpression(makeWeightedTweedieGradientExpression(predictionsTensor.getDataType(), power, eps, dims))
                .exampleWeights(weightBroadcast.getRaggedFeatureOutput()).exampleWeightsName(kExampleWeightsName);
        } else {
            rawBuilder.lossExpression(makeTweedieLossExpression(lossDataType, power, eps))
                .gradientExpression(makeTweedieGradientExpression(predictionsTensor.getDataType(), power, eps));
        }
        RaggedCustomLoss rawLoss = rawBuilder.build();
        raggedRawLossTensor = rawLoss.getRaggedRawLoss();
        lossShaperInput = raggedRawLossTensor->getValues();
        if (lossShape == LossShape::NONE) { lossTensor = lossShaperInput; Stub::Builder().network(*network).inputTensor(lossShaperInput).build(); }
        else if (lossShape == LossShape::RAW) lossTensor = lossShaperInput;
        else if (lossShape == LossShape::PER_EXAMPLE) lossTensor = RaggedLossShaper::Builder().network(*network).lossInput(raggedRawLossTensor.value()).reportsPerExampleLoss().build().getLossOutput();
        else if (lossShape == LossShape::BATCH) lossTensor = RaggedLossShaper::Builder().network(*network).lossInput(raggedRawLossTensor.value()).reportsBatchLoss().build().getLossOutput();
        else THOR_UNREACHABLE();
        return;
    }

    if (exampleWeightsTensor.has_value()) {
        MultiInputCustomLoss rawTweedieLoss = MultiInputCustomLoss::Builder()
                                                   .network(*network)
                                                   .lossExpression(makeWeightedTweedieLossExpression(lossDataType, power, eps))
                                                   .gradientExpression(
                                                       makeWeightedTweedieGradientExpression(predictionsTensor.getDataType(), power, eps))
                                                   .input(kPredictionsName, predictionsTensor, std::string(kGradientName))
                                                   .auxiliaryInput(kLabelsName, labelsTensor)
                                                   .auxiliaryInput(kExampleWeightsName, exampleWeightsTensor.value())
                                                   .lossName(kLossName)
                                                   .lossDataType(lossDataType)
                                                   .lossWeight(lossWeight.value_or(1.0f))
                                                   .reportsRawLoss()
                                                   .build();
        lossShaperInput = rawTweedieLoss.getLoss();
    } else {
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
                                        .lossWeight(lossWeight.value_or(1.0f))
                                        .reportsRawLoss()
                                        .build();
        lossShaperInput = rawTweedieLoss.getLoss();
    }

    finalizeLossReporting();
}

json TweedieLoss::architectureJson() const {
    json j = Loss::architectureJson();
    j["layer_type"] = "tweedie_loss";
    j["loss_shape"] = lossShape;
    j["power"] = power;
    j["eps"] = eps;
    if (isRagged()) { j["ragged_predictions"] = raggedPredictionsTensor->architectureJson(); j["ragged_labels"] = raggedLabelsTensor->architectureJson(); if (raggedRawLossTensor.has_value()) j["ragged_raw_loss"] = raggedRawLossTensor->architectureJson(); }
    return j;
}

void TweedieLoss::deserialize(const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in TweedieLoss::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "tweedie_loss")
        throw runtime_error("Layer type mismatch in TweedieLoss::deserialize: " + j.at("layer_type").get<std::string>());

    if (j.contains("ragged_predictions")) {
        RaggedTensor predictions = SegmentedPrimitiveDetail::reconstructInput(j.at("ragged_predictions"), network, "TweedieLoss");
        RaggedTensor labels = SegmentedPrimitiveDetail::reconstructInput(j.at("ragged_labels"), network, "TweedieLoss");
        Builder builder; builder.network(*network).predictions(predictions).labels(labels).power(j.value("power", 1.5f)).eps(j.value("eps", 1.0e-6f))
            .lossDataType(j.at("loss_data_type").get<DataType>()).lossWeight(ThorImplementation::lossWeightFromJson(j).value_or(1.0f));
        if (j.contains("example_weights_tensor")) builder.exampleWeights(network->getApiTensorByOriginalId(j["example_weights_tensor"].at("id").get<uint64_t>()));
        switch (j.at("loss_shape").get<LossShape>()) { case LossShape::NONE: builder.reportsNoLoss(); break; case LossShape::RAW: builder.reportsRawLoss(); break; case LossShape::PER_EXAMPLE: builder.reportsPerExampleLoss(); break; case LossShape::BATCH: builder.reportsBatchLoss(); break; case LossShape::PER_OUTPUT: throw runtime_error("TweedieLoss serialized ragged PER_OUTPUT is unsupported."); }
        (void)builder.build(); return;
    }

    uint64_t originalTensorId = j["predictions_tensor"].at("id").get<uint64_t>();
    Tensor predictions = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j["labels_tensor"].at("id").get<uint64_t>();
    Tensor labels = network->getApiTensorByOriginalId(originalTensorId);

    TweedieLoss loss;
    loss.lossShape = j.at("loss_shape").get<LossShape>();
    loss.lossDataType = j.at("loss_data_type").get<DataType>();

    loss.lossWeight = ThorImplementation::lossWeightFromJson(j);
    loss.power = j.value("power", 1.5f);
    loss.eps = j.value("eps", 1.0e-6f);
    loss.predictionsTensor = predictions;
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
    Thor::Loss::register_layer("tweedie_loss", &Thor::TweedieLoss::deserialize);
    return true;
}();
}  // namespace
