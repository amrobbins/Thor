#pragma once

#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/DataType.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/FusedEquation.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ThorImplementation::LossExpression {

enum class Formula { MEAN_SQUARED_ERROR = 1, MEAN_ABSOLUTE_ERROR, MEAN_ABSOLUTE_PERCENTAGE_ERROR };

struct Options {
    Formula formula = Formula::MEAN_SQUARED_ERROR;
    DataType computeDataType = DataType::FP32;
    float epsilon = 0.0001f;
    float maxMagnitude = 1000.0f;
    std::string predictionsName = "predictions";
    std::string labelsName = "labels";
    std::string lossName = "metric";
};

inline std::string toString(Formula formula) {
    switch (formula) {
        case Formula::MEAN_SQUARED_ERROR:
            return "mean_squared_error";
        case Formula::MEAN_ABSOLUTE_ERROR:
            return "mean_absolute_error";
        case Formula::MEAN_ABSOLUTE_PERCENTAGE_ERROR:
            return "mean_absolute_percentage_error";
        default:
            throw std::invalid_argument("Unsupported loss expression formula.");
    }
}

inline Formula formulaFromString(const std::string& formula) {
    if (formula == "mean_squared_error")
        return Formula::MEAN_SQUARED_ERROR;
    if (formula == "mean_absolute_error")
        return Formula::MEAN_ABSOLUTE_ERROR;
    if (formula == "mean_absolute_percentage_error")
        return Formula::MEAN_ABSOLUTE_PERCENTAGE_ERROR;
    throw std::invalid_argument("Unsupported loss expression formula: " + formula);
}

inline std::string displayName(Formula formula) {
    switch (formula) {
        case Formula::MEAN_SQUARED_ERROR:
            return "Mean Squared Error";
        case Formula::MEAN_ABSOLUTE_ERROR:
            return "Mean Absolute Error";
        case Formula::MEAN_ABSOLUTE_PERCENTAGE_ERROR:
            return "Mean Absolute Percentage Error";
        default:
            throw std::invalid_argument("Unsupported loss expression formula.");
    }
}

inline bool isPredictionDTypeSupported(DataType dtype) { return dtype == DataType::FP16 || dtype == DataType::FP32; }

inline bool isLabelDTypeSupported(DataType dtype) {
    return dtype == DataType::UINT8 || dtype == DataType::UINT16 || dtype == DataType::UINT32 || dtype == DataType::INT8 ||
           dtype == DataType::INT16 || dtype == DataType::INT32 || dtype == DataType::FP16 || dtype == DataType::FP32;
}

inline std::vector<uint64_t> allAxes(uint64_t rank) {
    std::vector<uint64_t> axes;
    axes.reserve(rank);
    for (uint64_t axis = 0; axis < rank; ++axis)
        axes.push_back(axis);
    return axes;
}

inline std::vector<uint64_t> squeezeAllButOneAxis(uint64_t rank) {
    std::vector<uint64_t> axes;
    if (rank <= 1)
        return axes;
    axes.reserve(rank - 1);
    for (uint64_t axis = 0; axis + 1 < rank; ++axis)
        axes.push_back(axis);
    return axes;
}

inline void validateElementwiseRegressionLossInputs(const Tensor& predictions, const Tensor& labels) {
    const std::vector<uint64_t> predictionDims = predictions.getDescriptor().getDimensions();
    const std::vector<uint64_t> labelDims = labels.getDescriptor().getDimensions();
    const DataType predictionDType = predictions.getDescriptor().getDataType();
    const DataType labelDType = labels.getDescriptor().getDataType();

    THOR_THROW_IF_FALSE(!predictionDims.empty());
    THOR_THROW_IF_FALSE(predictionDims == labelDims);
    THOR_THROW_IF_FALSE(isPredictionDTypeSupported(predictionDType));
    THOR_THROW_IF_FALSE(isLabelDTypeSupported(labelDType));
}

inline Expression rawLossExpression(Formula formula, const Expression& predictions, const Expression& labels, const Options& options) {
    Expression diff = predictions - labels;
    switch (formula) {
        case Formula::MEAN_SQUARED_ERROR:
            return diff * diff;
        case Formula::MEAN_ABSOLUTE_ERROR:
            return diff.abs();
        case Formula::MEAN_ABSOLUTE_PERCENTAGE_ERROR: {
            Expression absLabels = labels.abs();
            Expression safeDenominator = Expression::where(absLabels < Expression(options.epsilon), Expression(options.epsilon), absLabels);
            Expression loss = ((diff / safeDenominator).abs()) * Expression(100.0);
            if (options.maxMagnitude > 0.0f)
                loss = loss.min(Expression(options.maxMagnitude));
            return loss;
        }
        default:
            throw std::invalid_argument("Unsupported loss expression formula.");
    }
}

inline Expression batchLossFromRawLoss(const Expression& rawLoss, const std::vector<uint64_t>& rawLossDims, DataType computeDataType) {
    THOR_THROW_IF_FALSE(!rawLossDims.empty());
    const double batchSize = static_cast<double>(rawLossDims[0]);
    THOR_THROW_IF_FALSE(batchSize > 0.0);

    // Match LossShaper's batch behavior: sum each example's loss vector, sum across the batch,
    // then divide by batch size only. This is intentionally not reduce_mean over every element.
    Expression summed = rawLoss.reduce_sum(allAxes(rawLossDims.size()), squeezeAllButOneAxis(rawLossDims.size()), computeDataType);
    return summed / Expression::constantScalar(batchSize);
}

inline DynamicExpression makeBatchLossMetricExpression(Options options) {
    if (options.predictionsName.empty() || options.labelsName.empty() || options.lossName.empty())
        throw std::invalid_argument("Loss expression input and output names cannot be empty.");
    if (options.predictionsName == options.labelsName)
        throw std::invalid_argument("Loss expression predictions and labels names must be distinct.");

    const std::vector<std::string> expectedInputNames{options.predictionsName, options.labelsName};
    const std::vector<std::string> expectedOutputNames{options.lossName};

    return DynamicExpression(expectedInputNames,
                             expectedOutputNames,
                             [options = std::move(options)](const DynamicExpression::TensorMap& inputs,
                                                            const DynamicExpression::TensorMap& outputs,
                                                            Stream& stream) -> DynamicExpressionBuild {
                                 auto predictionsIt = inputs.find(options.predictionsName);
                                 auto labelsIt = inputs.find(options.labelsName);
                                 if (predictionsIt == inputs.end() || labelsIt == inputs.end())
                                     throw std::invalid_argument("Loss metric expression requires predictions and labels inputs.");

                                 const Tensor& predictionsTensor = predictionsIt->second;
                                 const Tensor& labelsTensor = labelsIt->second;
                                 validateElementwiseRegressionLossInputs(predictionsTensor, labelsTensor);

                                 const std::vector<uint64_t> predictionDims = predictionsTensor.getDescriptor().getDimensions();
                                 Expression predictions = Expression::input(options.predictionsName, options.computeDataType, options.computeDataType);
                                 Expression labels = Expression::input(options.labelsName, options.computeDataType, options.computeDataType);
                                 Expression raw = rawLossExpression(options.formula, predictions, labels, options);
                                 Expression loss = batchLossFromRawLoss(raw, predictionDims, options.computeDataType);

                                 ExpressionDefinition definition =
                                     ExpressionDefinition::fromOutputs(Expression::outputs({{options.lossName, loss}}));
                                 return DynamicExpressionBuild{
                                     std::make_shared<FusedEquation>(FusedEquation::compile(definition.outputs, stream.getGpuNum())),
                                     inputs,
                                     {},
                                     outputs,
                                     {},
                                 };
                             });
}

}  // namespace ThorImplementation::LossExpression
