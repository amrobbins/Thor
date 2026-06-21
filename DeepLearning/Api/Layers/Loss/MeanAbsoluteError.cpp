#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Api/Layers/Loss/MeanAbsoluteError.h"

#include "DeepLearning/Api/Layers/Loss/MultiInputCustomLoss.h"

#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/Expression.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {
namespace {

constexpr const char* kPredictionsName = "predictions";
constexpr const char* kLabelsName = "labels";
constexpr const char* kExampleWeightsName = "example_weights";
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
            throw runtime_error("Unsupported MAE label dtype: " + ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
    }
}

void validatePredictionsDType(DataType dtype) {
    if (dtype != DataType::FP16 && dtype != DataType::FP32) {
        throw runtime_error("Unsupported MAE predictions dtype: " +
                            ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
    }
}

void validateExampleWeights(Tensor predictions, Tensor labels, std::optional<Tensor> exampleWeights) {
    if (!exampleWeights.has_value())
        return;
    if (exampleWeights.value() == predictions || exampleWeights.value() == labels)
        throw runtime_error("MAE example_weights tensor must be distinct from predictions and labels.");
    const DataType dtype = exampleWeights.value().getDataType();
    if (dtype != DataType::FP16 && dtype != DataType::FP32)
        throw runtime_error("Unsupported MAE example_weights dtype: " + ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
    const vector<uint64_t>& dims = exampleWeights.value().getDimensions();
    if (dims != vector<uint64_t>{1} && dims != predictions.getDimensions()) {
        throw runtime_error("MAE example_weights dimensions must be [1] for per-example weights or match predictions dimensions.");
    }
}

ThorImplementation::DynamicExpression makeMAELossExpression(DataType lossDataType) {
    ThorImplementation::Expression predictions = ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels = ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression loss = (predictions - labels).abs().withOutputDType(lossDataType);
    ThorImplementation::ExpressionDefinition definition =
        ThorImplementation::ExpressionDefinition::fromOutputs(ThorImplementation::Expression::outputs({{kLossName, loss}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

ThorImplementation::DynamicExpression makeWeightedMAELossExpression(DataType lossDataType) {
    ThorImplementation::Expression predictions = ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels = ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression exampleWeights =
        ThorImplementation::Expression::input(kExampleWeightsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression loss = ((predictions - labels).abs() * exampleWeights).withOutputDType(lossDataType);
    ThorImplementation::ExpressionDefinition definition =
        ThorImplementation::ExpressionDefinition::fromOutputs(ThorImplementation::Expression::outputs({{kLossName, loss}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

ThorImplementation::DynamicExpression makeMAEGradientExpression(DataType predictionsDataType) {
    validatePredictionsDType(predictionsDataType);

    ThorImplementation::Expression predictions = ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels = ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression zero(0.0);
    ThorImplementation::Expression positive(1.0);
    ThorImplementation::Expression negative(-1.0);
    ThorImplementation::Expression diff = predictions - labels;
    ThorImplementation::Expression sign =
        ThorImplementation::Expression::where(diff > zero, positive, ThorImplementation::Expression::where(diff < zero, negative, zero));
    ThorImplementation::Expression gradient =
        (sign * ThorImplementation::Expression(ThorImplementation::Loss::getLossScalingFactor())).withOutputDType(predictionsDataType);
    ThorImplementation::ExpressionDefinition definition = ThorImplementation::ExpressionDefinition::fromOutputs(
        ThorImplementation::Expression::outputs({{kGradientName, gradient}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

ThorImplementation::DynamicExpression makeWeightedMAEGradientExpression(DataType predictionsDataType) {
    validatePredictionsDType(predictionsDataType);

    ThorImplementation::Expression predictions = ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels = ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression exampleWeights =
        ThorImplementation::Expression::input(kExampleWeightsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression zero(0.0);
    ThorImplementation::Expression positive(1.0);
    ThorImplementation::Expression negative(-1.0);
    ThorImplementation::Expression diff = predictions - labels;
    ThorImplementation::Expression sign =
        ThorImplementation::Expression::where(diff > zero, positive, ThorImplementation::Expression::where(diff < zero, negative, zero));
    ThorImplementation::Expression gradient =
        (sign * exampleWeights * ThorImplementation::Expression(ThorImplementation::Loss::getLossScalingFactor()))
            .withOutputDType(predictionsDataType);
    ThorImplementation::ExpressionDefinition definition = ThorImplementation::ExpressionDefinition::fromOutputs(
        ThorImplementation::Expression::outputs({{kGradientName, gradient}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

}  // namespace

void MAE::buildSupportLayersAndAddToNetwork() {
    validatePredictionsDType(predictionsTensor.getDataType());
    validateLabelsDType(labelsTensor.getDataType());
    validateExampleWeights(predictionsTensor, labelsTensor, exampleWeightsTensor);

    if (exampleWeightsTensor.has_value()) {
        MultiInputCustomLoss rawMAE = MultiInputCustomLoss::Builder()
                                          .network(*network)
                                          .lossExpression(makeWeightedMAELossExpression(lossDataType))
                                          .gradientExpression(makeWeightedMAEGradientExpression(predictionsTensor.getDataType()))
                                          .input(kPredictionsName, predictionsTensor, std::string(kGradientName))
                                          .auxiliaryInput(kLabelsName, labelsTensor)
                                          .auxiliaryInput(kExampleWeightsName, exampleWeightsTensor.value())
                                          .lossName(kLossName)
                                          .lossDataType(lossDataType)
                                          .lossWeight(lossWeight.value_or(1.0f))
                                          .reportsRawLoss()
                                          .build();
        lossShaperInput = rawMAE.getLoss();
    } else {
        CustomLoss rawMAE = CustomLoss::Builder()
                                              .network(*network)
                                              .lossExpression(makeMAELossExpression(lossDataType))
                                              .gradientExpression(makeMAEGradientExpression(predictionsTensor.getDataType()))
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

        lossShaperInput = rawMAE.getLoss();
    }

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

json MAE::architectureJson() const {
    json j = Loss::architectureJson();
    j["layer_type"] = "mae";
    return j;
}

void MAE::deserialize(const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in MAE::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "mae")
        throw runtime_error("Layer type mismatch in MAE::deserialize: " + j.at("layer_type").get<std::string>());

    uint64_t originalTensorId = j["predictions_tensor"].at("id").get<uint64_t>();
    Tensor predictions = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j["labels_tensor"].at("id").get<uint64_t>();
    Tensor labels = network->getApiTensorByOriginalId(originalTensorId);

    MAE meanAbsoluteError;
    meanAbsoluteError.lossShape = j.at("loss_shape").get<LossShape>();
    meanAbsoluteError.lossDataType = j.at("loss_data_type").get<DataType>();

    meanAbsoluteError.lossWeight = ThorImplementation::lossWeightFromJson(j);
    meanAbsoluteError.predictionsTensor = predictions;
    meanAbsoluteError.labelsTensor = labels;
    if (j.contains("example_weights_tensor")) {
        originalTensorId = j["example_weights_tensor"].at("id").get<uint64_t>();
        meanAbsoluteError.exampleWeightsTensor = network->getApiTensorByOriginalId(originalTensorId);
    }
    meanAbsoluteError.network = network;
    meanAbsoluteError.initialized = true;
    meanAbsoluteError.buildSupportLayersAndAddToNetwork();
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Loss::register_layer("mae", &Thor::MAE::deserialize);
    return true;
}();
}  // namespace
