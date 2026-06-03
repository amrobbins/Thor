#include "DeepLearning/Implementation/Layers/Loss/MeanAbsoluteError.h"

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
    switch (dtype) {
        case DataType::BOOLEAN:
        case DataType::UINT8:
        case DataType::UINT16:
        case DataType::UINT32:
        case DataType::FP16:
        case DataType::FP32:
            return;
        default:
            throw runtime_error("Unsupported MeanAbsoluteError label dtype: " + TensorDescriptor::getElementTypeName(dtype));
    }
}

void validatePredictionsDType(DataType dtype) {
    if (dtype != DataType::FP16 && dtype != DataType::FP32) {
        throw runtime_error("Unsupported MeanAbsoluteError predictions dtype: " + TensorDescriptor::getElementTypeName(dtype));
    }
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
                 lossDataType) {}

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

                                 const DataType predictionDType = inputs.at(kPredictionsName).getDescriptor().getDataType();
                                 Expression predictions = Expression::input(kPredictionsName, predictionDType, predictionDType);
                                 Expression labels = Expression::input(kLabelsName, predictionDType, predictionDType);
                                 Expression diff = (predictions - labels).withDTypes(predictionDType, predictionDType);
                                 Expression loss = diff.abs().withDTypes(predictionDType, lossDataType);
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
                                 Expression predictions = Expression::input(kPredictionsName, predictionDType, predictionDType);
                                 Expression labels = Expression::input(kLabelsName, predictionDType, predictionDType);
                                 Expression zero = Expression(0.0).withDTypes(predictionDType, predictionDType);
                                 Expression positive = Expression(1.0).withDTypes(predictionDType, predictionDType);
                                 Expression negative = Expression(-1.0).withDTypes(predictionDType, predictionDType);
                                 Expression diff = (predictions - labels).withDTypes(predictionDType, predictionDType);
                                 Expression sign = Expression::where(diff > zero, positive, Expression::where(diff < zero, negative, zero))
                                                       .withDTypes(predictionDType, predictionDType);
                                 Expression scale = Expression(Loss::getLossScalingFactor()).withDTypes(predictionDType, predictionDType);
                                 Expression grad = (sign * scale).withDTypes(predictionDType, predictionDType);
                                 return compileOutputs(Expression::outputs({{kGradientName, grad}}), inputs, outputs, stream);
                             });
}
