#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Api/Layers/Loss/SoftTargetCrossEntropy.h"

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
    if (dtype != DataType::FP16 && dtype != DataType::FP32) {
        throw runtime_error("Unsupported SoftTargetCrossEntropy label dtype: " +
                            ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
    }
}

void validatePredictionsDType(DataType dtype) {
    if (dtype != DataType::FP16 && dtype != DataType::FP32) {
        throw runtime_error("Unsupported SoftTargetCrossEntropy predictions dtype: " +
                            ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
    }
}

ThorImplementation::DynamicExpression makeSoftTargetCrossEntropyLossExpression(DataType lossDataType) {
    ThorImplementation::Expression logits = ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels = ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression logProbabilities = logits.logSoftmax();
    ThorImplementation::Expression loss = (-(labels * logProbabilities)).withOutputDType(lossDataType);
    ThorImplementation::ExpressionDefinition definition =
        ThorImplementation::ExpressionDefinition::fromOutputs(ThorImplementation::Expression::outputs({{kLossName, loss}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

ThorImplementation::DynamicExpression makeSoftTargetCrossEntropyGradientExpression(DataType predictionsDataType) {
    validatePredictionsDType(predictionsDataType);

    ThorImplementation::Expression logits = ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels = ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression gradient =
        ((logits.softmax() - labels) * ThorImplementation::Expression(ThorImplementation::Loss::getLossScalingFactor()))
            .withOutputDType(predictionsDataType);
    ThorImplementation::ExpressionDefinition definition = ThorImplementation::ExpressionDefinition::fromOutputs(
        ThorImplementation::Expression::outputs({{kGradientName, gradient}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

}  // namespace

void SoftTargetCrossEntropy::buildSupportLayersAndAddToNetwork() {
    validatePredictionsDType(predictionsTensor.getDataType());
    validateLabelsDType(labelsTensor.getDataType());

    CustomLoss rawSoftTargetCrossEntropy = CustomLoss::Builder()
                                               .network(*network)
                                               .lossExpression(makeSoftTargetCrossEntropyLossExpression(lossDataType))
                                               .gradientExpression(makeSoftTargetCrossEntropyGradientExpression(predictionsTensor.getDataType()))
                                               .predictions(predictionsTensor)
                                               .labels(labelsTensor)
                                               .predictionsName(kPredictionsName)
                                               .labelsName(kLabelsName)
                                               .lossName(kLossName)
                                               .gradientName(kGradientName)
                                               .lossDataType(lossDataType)
                                               .reportsRawLoss()
                                               .build();

    lossShaperInput = rawSoftTargetCrossEntropy.getLoss();

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

json SoftTargetCrossEntropy::architectureJson() const {
    json j = Loss::architectureJson();
    j["loss_shape"] = lossShape;
    return j;
}

void SoftTargetCrossEntropy::deserialize(const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in SoftTargetCrossEntropy::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "soft_target_cross_entropy")
        throw runtime_error("Layer type mismatch in SoftTargetCrossEntropy::deserialize: " + j.at("layer_type").get<std::string>());

    uint64_t originalTensorId = j["predictions_tensor"].at("id").get<uint64_t>();
    Tensor predictions = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j["labels_tensor"].at("id").get<uint64_t>();
    Tensor labels = network->getApiTensorByOriginalId(originalTensorId);

    SoftTargetCrossEntropy softTargetCrossEntropy;
    softTargetCrossEntropy.lossShape = j.at("loss_shape").get<LossShape>();
    softTargetCrossEntropy.lossDataType = j.at("loss_data_type").get<DataType>();
    softTargetCrossEntropy.predictionsTensor = predictions;
    softTargetCrossEntropy.labelsTensor = labels;
    softTargetCrossEntropy.network = network;
    softTargetCrossEntropy.initialized = true;
    softTargetCrossEntropy.buildSupportLayersAndAddToNetwork();
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Loss::register_layer("soft_target_cross_entropy", &Thor::SoftTargetCrossEntropy::deserialize);
    return true;
}();
}  // namespace
