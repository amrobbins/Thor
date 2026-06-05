#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Api/Layers/Loss/MeanAbsoluteError.h"

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
            throw runtime_error("Unsupported MAE label dtype: " + ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
    }
}

void validatePredictionsDType(DataType dtype) {
    if (dtype != DataType::FP16 && dtype != DataType::FP32) {
        throw runtime_error("Unsupported MAE predictions dtype: " +
                            ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
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

}  // namespace

void MAE::buildSupportLayersAndAddToNetwork() {
    validatePredictionsDType(predictionsTensor.getDataType());
    validateLabelsDType(labelsTensor.getDataType());

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
                                          .reportsRawLoss()
                                          .build();

    lossShaperInput = rawMAE.getLoss();

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
    meanAbsoluteError.predictionsTensor = predictions;
    meanAbsoluteError.labelsTensor = labels;
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
