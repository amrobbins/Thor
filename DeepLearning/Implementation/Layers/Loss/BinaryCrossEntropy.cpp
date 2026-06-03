#include "DeepLearning/Implementation/Layers/Loss/BinaryCrossEntropy.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/FusedEquation.h"

#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>

using namespace ThorImplementation;
using namespace std;

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
        case DataType::INT8:
        case DataType::INT16:
        case DataType::INT32:
        case DataType::FP16:
        case DataType::FP32:
            return;
        default:
            throw runtime_error("Unsupported BinaryCrossEntropy label dtype: " + TensorDescriptor::getElementTypeName(dtype));
    }
}

}  // namespace

BinaryCrossEntropy::BinaryCrossEntropy(DataType lossDataType)
    : CustomLoss(makeForwardExpression(lossDataType),
                 makeGradientExpression(),
                 kPredictionsName,
                 kLabelsName,
                 kLossName,
                 kGradientName,
                 lossDataType) {}

void BinaryCrossEntropy::compileImpl() {
    THOR_THROW_IF_FALSE(featureInput.has_value());
    THOR_THROW_IF_FALSE(labelsInput.has_value());

    const DataType predictionDType = featureInput.value().getDescriptor().getDataType();
    THOR_THROW_IF_FALSE(predictionDType == DataType::FP16 || predictionDType == DataType::FP32);
    validateLabelsDType(labelsInput.value().getDescriptor().getDataType());

    CustomLoss::compileImpl();
}

DynamicExpression BinaryCrossEntropy::makeForwardExpression(DataType lossDataType) {
    Expression logits = Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    Expression labels = Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    Expression zero(0.0);

    // Numerically stable BCE-with-logits:
    //   max(x, 0) - x * y + log1p(exp(-abs(x)))
    Expression loss = (logits.max(zero) - (logits * labels) + (-logits.abs()).exp().log1p()).withOutputDType(lossDataType);
    ExpressionDefinition definition = ExpressionDefinition::fromOutputs(Expression::outputs({{kLossName, loss}}));
    return DynamicExpression::fromExpressionDefinition(definition);
}

DynamicExpression BinaryCrossEntropy::makeGradientExpression() {
    return DynamicExpression({kPredictionsName, kLabelsName},
                             {kGradientName},
                             [](const DynamicExpression::TensorMap& inputs,
                                const DynamicExpression::TensorMap& outputs,
                                Stream& stream) -> DynamicExpressionBuild {
                                 const auto predictionsIt = inputs.find(kPredictionsName);
                                 if (predictionsIt == inputs.end())
                                     throw std::runtime_error("BinaryCrossEntropy gradient expression missing predictions input.");
                                 const auto labelsIt = inputs.find(kLabelsName);
                                 if (labelsIt == inputs.end())
                                     throw std::runtime_error("BinaryCrossEntropy gradient expression missing labels input.");

                                 const DataType predictionDType = predictionsIt->second.getDescriptor().getDataType();
                                 THOR_THROW_IF_FALSE(predictionDType == DataType::FP16 || predictionDType == DataType::FP32);
                                 validateLabelsDType(labelsIt->second.getDescriptor().getDataType());

                                 Expression logits = Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
                                 Expression labels = Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
                                 Expression grad =
                                     ((logits.sigmoid() - labels) * Expression(Loss::getLossScalingFactor())).withOutputDType(predictionDType);
                                 auto gradientOutputs = Expression::outputs({{kGradientName, grad}});
                                 return DynamicExpressionBuild{
                                     .equation = std::make_shared<FusedEquation>(
                                         FusedEquation::compile(gradientOutputs.physicalOutputs(), stream.getGpuNum())),
                                     .stamp_inputs = inputs,
                                     .tensor_scalar_inputs = {},
                                     .preallocated_outputs = outputs,
                                     .requested_output_shapes = {},
                                 };
                             });
}
