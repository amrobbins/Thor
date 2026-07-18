#include "DeepLearning/Implementation/Layers/Loss/MeanPowerError.h"

#include "DeepLearning/Implementation/Layers/Loss/RegressionLossDType.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/FusedEquation.h"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>

using namespace ThorImplementation;
using namespace std;

namespace {

constexpr const char* kPredictionsName = "predictions";
constexpr const char* kLabelsName = "labels";
constexpr const char* kLossName = "loss";
constexpr const char* kGradientName = "predictions_grad";

void validateLabelsDType(DataType dtype) {
    RegressionLossDType::validateLabelsDType("MeanPowerError", dtype);
}

void validatePredictionsDType(DataType dtype) {
    RegressionLossDType::validatePredictionsDType("MeanPowerError", dtype);
}

void validateDynamicInputs(const DynamicExpression::TensorMap& inputs) {
    const auto predictionsIt = inputs.find(kPredictionsName);
    if (predictionsIt == inputs.end())
        throw runtime_error("MeanPowerError expression missing predictions input.");
    const auto labelsIt = inputs.find(kLabelsName);
    if (labelsIt == inputs.end())
        throw runtime_error("MeanPowerError expression missing labels input.");

    validatePredictionsDType(predictionsIt->second.getDescriptor().getDataType());
    validateLabelsDType(labelsIt->second.getDescriptor().getDataType());
    THOR_THROW_IF_FALSE(predictionsIt->second.getDescriptor().getDimensions() == labelsIt->second.getDescriptor().getDimensions());
}

DynamicExpressionBuild compileOutputs(const Outputs& outputs,
                                      const DynamicExpression::TensorMap& stampInputs,
                                      const DynamicExpression::TensorMap& preallocatedOutputs,
                                      Stream& stream) {
    return DynamicExpressionBuild{
        .equation = std::make_shared<FusedEquation>(FusedEquation::compile(outputs.physicalOutputs(), stream.getGpuNum())),
        .stamp_inputs = stampInputs,
        .tensor_scalar_inputs = {},
        .preallocated_outputs = preallocatedOutputs,
        .requested_output_shapes = {},
    };
}

Expression signOf(const Expression& diff) {
    Expression zero(0.0);
    Expression positive(1.0);
    Expression negative(-1.0);
    return Expression::where(diff > zero, positive, Expression::where(diff < zero, negative, zero));
}

}  // namespace

MeanPowerError::MeanPowerError(DataType lossDataType, float exponent)
    : CustomLoss(makeForwardExpression(exponent, lossDataType),
                 makeGradientExpression(exponent),
                 kPredictionsName,
                 kLabelsName,
                 kLossName,
                 kGradientName,
                 lossDataType),
      exponent(exponent) {
    RegressionLossDType::validateLossDType("MeanPowerError", lossDataType);
    validateExponent(exponent);
}

void MeanPowerError::validateExponent(float exponent) {
    if (!std::isfinite(exponent) || exponent < 1.0f) {
        throw runtime_error("MeanPowerError exponent must be finite and greater than or equal to 1.0.");
    }
}

void MeanPowerError::compileImpl() {
    THOR_THROW_IF_FALSE(featureInput.has_value());
    THOR_THROW_IF_FALSE(labelsInput.has_value());
    THOR_THROW_IF_FALSE(featureInput.value().getDescriptor().getDimensions() == labelsInput.value().getDescriptor().getDimensions());

    validateExponent(exponent);
    validatePredictionsDType(featureInput.value().getDescriptor().getDataType());
    validateLabelsDType(labelsInput.value().getDescriptor().getDataType());

    CustomLoss::compileImpl();
}

DynamicExpression MeanPowerError::makeForwardExpression(float exponent, DataType lossDataType) {
    validateExponent(exponent);
    return DynamicExpression({kPredictionsName, kLabelsName},
                             {kLossName},
                             [exponent, lossDataType](const DynamicExpression::TensorMap& inputs,
                                                      const DynamicExpression::TensorMap& outputs,
                                                      Stream& stream) -> DynamicExpressionBuild {
                                 validateDynamicInputs(inputs);

                                 Expression predictions = Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
                                 Expression labels = Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
                                 Expression diff = predictions - labels;
                                 Expression loss = [&]() -> Expression {
                                     if (exponent == 1.0f) {
                                         return diff.abs().withOutputDType(lossDataType);
                                     }
                                     if (exponent == 2.0f) {
                                         return (diff * diff).withOutputDType(lossDataType);
                                     }

                                     Expression absDiff = diff.abs();
                                     Expression exponentExpr(exponent);
                                     return absDiff.pow(exponentExpr).withOutputDType(lossDataType);
                                 }();
                                 return compileOutputs(Expression::outputs({{kLossName, loss}}), inputs, outputs, stream);
                             });
}

DynamicExpression MeanPowerError::makeGradientExpression(float exponent) {
    validateExponent(exponent);
    return DynamicExpression({kPredictionsName, kLabelsName},
                             {kGradientName},
                             [exponent](const DynamicExpression::TensorMap& inputs,
                                        const DynamicExpression::TensorMap& outputs,
                                        Stream& stream) -> DynamicExpressionBuild {
                                 validateDynamicInputs(inputs);

                                 const DataType predictionDType = inputs.at(kPredictionsName).getDescriptor().getDataType();
                                 Expression predictions = Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
                                 Expression labels = Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
                                 Expression diff = predictions - labels;
                                 Expression absDiff = diff.abs();
                                 Expression sign = signOf(diff);
                                 Expression scale(exponent * Loss::getLossScalingFactor());
                                 Expression grad = [&]() -> Expression {
                                     if (exponent == 1.0f) {
                                         return (sign * scale).withOutputDType(predictionDType);
                                     }

                                     Expression power(exponent - 1.0f);
                                     return (sign * absDiff.pow(power) * scale).withOutputDType(predictionDType);
                                 }();
                                 return compileOutputs(Expression::outputs({{kGradientName, grad}}), inputs, outputs, stream);
                             });
}
