#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Api/Layers/Loss/MarginRankingLoss.h"

#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/Expression.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {
namespace {

constexpr const char* kInput1Name = "input1";
constexpr const char* kInput2Name = "input2";
constexpr const char* kTargetName = "target";
constexpr const char* kLossName = "loss";
constexpr const char* kInput1GradientName = "input1_grad";
constexpr const char* kInput2GradientName = "input2_grad";

void validateScoreDType(const char* tensorName, DataType dtype) {
    if (dtype != DataType::FP16 && dtype != DataType::FP32) {
        throw runtime_error(string("Unsupported MarginRankingLoss ") + tensorName + " dtype: " +
                            ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
    }
}

void validateTargetDType(DataType dtype) {
    switch (dtype) {
        case DataType::INT8:
        case DataType::INT16:
        case DataType::INT32:
        case DataType::INT64:
        case DataType::FP16:
        case DataType::FP32:
            return;
        default:
            throw runtime_error("Unsupported MarginRankingLoss target dtype: " +
                                ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
    }
}

void validateMarginRankingDTypes(DataType input1DType, DataType input2DType, DataType targetDType) {
    validateScoreDType("input1", input1DType);
    validateScoreDType("input2", input2DType);
    validateTargetDType(targetDType);
    if (input1DType != input2DType)
        throw runtime_error("MarginRankingLoss input1 and input2 tensors must have the same dtype.");
}

ThorImplementation::DynamicExpression makeMarginRankingLossExpression(DataType lossDataType, float margin) {
    validateScoreDType("loss", lossDataType);

    ThorImplementation::Expression input1 = ThorImplementation::Expression::input(kInput1Name, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression input2 = ThorImplementation::Expression::input(kInput2Name, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression target = ThorImplementation::Expression::input(kTargetName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression zero(0.0);

    ThorImplementation::Expression marginExpression(margin);
    ThorImplementation::Expression loss = (marginExpression - target * (input1 - input2)).max(zero).withOutputDType(lossDataType);

    ThorImplementation::ExpressionDefinition definition =
        ThorImplementation::ExpressionDefinition::fromOutputs(ThorImplementation::Expression::outputs({{kLossName, loss}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

ThorImplementation::DynamicExpression makeMarginRankingGradientExpression(DataType inputDataType, float margin) {
    validateScoreDType("input", inputDataType);

    ThorImplementation::Expression input1 = ThorImplementation::Expression::input(kInput1Name, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression input2 = ThorImplementation::Expression::input(kInput2Name, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression target = ThorImplementation::Expression::input(kTargetName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression zero(0.0);
    ThorImplementation::Expression one(1.0);
    ThorImplementation::Expression scale(ThorImplementation::Loss::getLossScalingFactor());

    ThorImplementation::Expression hinge = ThorImplementation::Expression(margin) - target * (input1 - input2);
    ThorImplementation::Expression active = ThorImplementation::Expression::where(hinge > zero, one, zero);
    ThorImplementation::Expression input1Gradient = (-target * active * scale).withOutputDType(inputDataType);
    ThorImplementation::Expression input2Gradient = (target * active * scale).withOutputDType(inputDataType);

    ThorImplementation::ExpressionDefinition definition = ThorImplementation::ExpressionDefinition::fromOutputs(
        ThorImplementation::Expression::outputs({{kInput1GradientName, input1Gradient}, {kInput2GradientName, input2Gradient}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

}  // namespace

void MarginRankingLoss::buildSupportLayersAndAddToNetwork() {
    validateMarginRankingDTypes(input1Tensor.getDataType(), input2Tensor.getDataType(), targetTensor.getDataType());
    THOR_THROW_IF_FALSE(!input1Tensor.getDimensions().empty());
    THOR_THROW_IF_FALSE(input1Tensor.getDimensions() == input2Tensor.getDimensions());
    THOR_THROW_IF_FALSE(input1Tensor.getDimensions() == targetTensor.getDimensions());
    THOR_THROW_IF_FALSE(margin >= 0.0f);

    MultiInputCustomLoss rawMarginRankingLoss = MultiInputCustomLoss::Builder()
                                                    .network(*network)
                                                    .lossExpression(makeMarginRankingLossExpression(lossDataType, margin))
                                                    .gradientExpression(makeMarginRankingGradientExpression(input1Tensor.getDataType(), margin))
                                                    .input(kInput1Name, input1Tensor, kInput1GradientName)
                                                    .input(kInput2Name, input2Tensor, kInput2GradientName)
                                                    .auxiliaryInput(kTargetName, targetTensor)
                                                    .lossName(kLossName)
                                                    .lossDataType(lossDataType)
                                                    .reportsRawLoss()
                                                    .build();

    lossShaperInput = rawMarginRankingLoss.getLoss();

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

json MarginRankingLoss::architectureJson() const {
    json j;
    j["factory"] = Factory::Loss.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = "margin_ranking_loss";
    j["layer_name"] = string("layer") + to_string(getId());
    j["loss_shape"] = lossShape;
    j["loss_data_type"] = lossDataType;
    j["input1_tensor"] = input1Tensor.architectureJson();
    j["input2_tensor"] = input2Tensor.architectureJson();
    j["target_tensor"] = targetTensor.architectureJson();
    j["loss_shaper_input_tensor"] = lossShaperInput.architectureJson();
    j["loss_tensor"] = lossTensor.architectureJson();
    j["margin"] = margin;
    return j;
}

void MarginRankingLoss::deserialize(const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in MarginRankingLoss::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "margin_ranking_loss")
        throw runtime_error("Layer type mismatch in MarginRankingLoss::deserialize: " + j.at("layer_type").get<std::string>());

    uint64_t originalTensorId = j["input1_tensor"].at("id").get<uint64_t>();
    Tensor input1 = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j["input2_tensor"].at("id").get<uint64_t>();
    Tensor input2 = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j["target_tensor"].at("id").get<uint64_t>();
    Tensor target = network->getApiTensorByOriginalId(originalTensorId);

    MarginRankingLoss marginRankingLoss;
    marginRankingLoss.lossShape = j.at("loss_shape").get<LossShape>();
    marginRankingLoss.lossDataType = j.at("loss_data_type").get<DataType>();
    marginRankingLoss.margin = j.value("margin", 0.0f);
    marginRankingLoss.input1Tensor = input1;
    marginRankingLoss.input2Tensor = input2;
    marginRankingLoss.targetTensor = target;
    marginRankingLoss.network = network;
    marginRankingLoss.initialized = true;
    marginRankingLoss.buildSupportLayersAndAddToNetwork();
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Loss::register_layer("margin_ranking_loss", &Thor::MarginRankingLoss::deserialize);
    return true;
}();
}  // namespace
