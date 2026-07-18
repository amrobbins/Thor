#include "DeepLearning/Implementation/Layers/Loss/MeanAbsoluteError.h"

#include "DeepLearning/Implementation/Layers/Loss/RegressionLossDType.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/FusedEquation.h"

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
    RegressionLossDType::validateLabelsDType("MeanAbsoluteError", dtype);
}

void validatePredictionsDType(DataType dtype) {
    RegressionLossDType::validatePredictionsDType("MeanAbsoluteError", dtype);
}

void validateDynamicInputs(const DynamicExpression::TensorMap& inputs) {
    const auto predictionsIt = inputs.find(kPredictionsName);
    if (predictionsIt == inputs.end())
        throw runtime_error("MeanAbsoluteError expression missing predictions input.");
    const auto labelsIt = inputs.find(kLabelsName);
    if (labelsIt == inputs.end())
        throw runtime_error("MeanAbsoluteError expression missing labels input.");

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

}  // namespace

MeanAbsoluteError::MeanAbsoluteError(DataType lossDataType)
    : CustomLoss(makeForwardExpression(lossDataType),
                 makeGradientExpression(),
                 kPredictionsName,
                 kLabelsName,
                 kLossName,
                 kGradientName,
                 lossDataType) {
    RegressionLossDType::validateLossDType("MeanAbsoluteError", lossDataType);
}

void MeanAbsoluteError::compileImpl() {
    THOR_THROW_IF_FALSE(featureInput.has_value());
    THOR_THROW_IF_FALSE(labelsInput.has_value());
    THOR_THROW_IF_FALSE(featureInput.value().getDescriptor().getDimensions() == labelsInput.value().getDescriptor().getDimensions());

    validatePredictionsDType(featureInput.value().getDescriptor().getDataType());
    validateLabelsDType(labelsInput.value().getDescriptor().getDataType());

    CustomLoss::compileImpl();
}

DynamicExpression MeanAbsoluteError::makeForwardExpression(DataType lossDataType) {
    return DynamicExpression({kPredictionsName, kLabelsName},
                             {kLossName},
                             [lossDataType](const DynamicExpression::TensorMap& inputs,
                                            const DynamicExpression::TensorMap& outputs,
                                            Stream& stream) -> DynamicExpressionBuild {
                                 validateDynamicInputs(inputs);

                                 Expression predictions = Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
                                 Expression labels = Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
                                 Expression loss = (predictions - labels).abs().withOutputDType(lossDataType);
                                 return compileOutputs(Expression::outputs({{kLossName, loss}}), inputs, outputs, stream);
                             });
}

DynamicExpression MeanAbsoluteError::makeGradientExpression() {
    return DynamicExpression({kPredictionsName, kLabelsName},
                             {kGradientName},
                             [](const DynamicExpression::TensorMap& inputs,
                                const DynamicExpression::TensorMap& outputs,
                                Stream& stream) -> DynamicExpressionBuild {
                                 validateDynamicInputs(inputs);

                                 const DataType predictionDType = inputs.at(kPredictionsName).getDescriptor().getDataType();
                                 Expression predictions = Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
                                 Expression labels = Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
                                 Expression zero(0.0);
                                 Expression positive(1.0);
                                 Expression negative(-1.0);
                                 Expression diff = predictions - labels;
                                 Expression sign =
                                     Expression::where(diff > zero, positive, Expression::where(diff < zero, negative, zero));
                                 Expression scale(Loss::getLossScalingFactor());
                                 Expression grad = (sign * scale).withOutputDType(predictionDType);
                                 return compileOutputs(Expression::outputs({{kGradientName, grad}}), inputs, outputs, stream);
                             });
}
