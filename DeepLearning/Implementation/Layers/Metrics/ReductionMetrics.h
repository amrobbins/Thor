#pragma once

#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Layers/Metrics/CustomMetric.h"
#include "DeepLearning/Implementation/Layers/Metrics/ReductionMetricDType.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/FusedEquation.h"

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ThorImplementation {
namespace ReductionMetricDetail {


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

inline Expression reduce(Expression values, ExprOp op, const std::vector<uint64_t>& reductionAxes, const std::vector<uint64_t>& squeezeAxes) {
    switch (op) {
        case ExprOp::REDUCE_AVG:
            return values.reduce_mean(reductionAxes, squeezeAxes, DataType::FP32);
        case ExprOp::REDUCE_SUM:
            return values.reduce_sum(reductionAxes, squeezeAxes, DataType::FP32);
        case ExprOp::REDUCE_MIN:
            return values.reduce_min(reductionAxes, squeezeAxes, DataType::FP32);
        case ExprOp::REDUCE_MAX:
            return values.reduce_max(reductionAxes, squeezeAxes, DataType::FP32);
        default:
            throw std::invalid_argument("Unsupported reduction metric op.");
    }
}

inline DynamicExpression makeUnaryReductionExpression(ExprOp op) {
    return DynamicExpression({"values"},
                             {"metric"},
                             [op](const DynamicExpression::TensorMap& inputs,
                                  const DynamicExpression::TensorMap& outputs,
                                  Stream& stream) -> DynamicExpressionBuild {
                                 auto valuesIt = inputs.find("values");
                                 if (valuesIt == inputs.end())
                                     throw std::invalid_argument("Reduction metric expression requires a values input.");

                                 const Tensor& valuesTensor = valuesIt->second;
                                 const std::vector<uint64_t> valueDims = valuesTensor.getDescriptor().getDimensions();
                                 const DataType valueDType = valuesTensor.getDescriptor().getDataType();
                                 THOR_THROW_IF_FALSE(!valueDims.empty());
                                 ReductionMetricDType::validateValueDType("reduction metric", "values", valueDType);

                                 const std::vector<uint64_t> reductionAxes = allAxes(valueDims.size());
                                 const std::vector<uint64_t> squeezeAxes = squeezeAllButOneAxis(valueDims.size());
                                 Expression values = Expression::input("values", DataType::FP32, DataType::FP32);
                                 Expression metric = reduce(values, op, reductionAxes, squeezeAxes);

                                 ExpressionDefinition definition =
                                     ExpressionDefinition::fromOutputs(Expression::outputs({{"metric", metric}}));
                                 return DynamicExpressionBuild{
                                     std::make_shared<FusedEquation>(FusedEquation::compile(definition.outputs, stream.getGpuNum())),
                                     inputs,
                                     {},
                                     outputs,
                                     {},
                                 };
                             });
}

inline DynamicExpression makeWeightedMeanExpression() {
    return DynamicExpression({"values", "weights"},
                             {"metric"},
                             [](const DynamicExpression::TensorMap& inputs,
                                const DynamicExpression::TensorMap& outputs,
                                Stream& stream) -> DynamicExpressionBuild {
                                 auto valuesIt = inputs.find("values");
                                 auto weightsIt = inputs.find("weights");
                                 if (valuesIt == inputs.end() || weightsIt == inputs.end())
                                     throw std::invalid_argument("WeightedMean metric expression requires values and weights inputs.");

                                 const Tensor& valuesTensor = valuesIt->second;
                                 const Tensor& weightsTensor = weightsIt->second;
                                 const std::vector<uint64_t> valueDims = valuesTensor.getDescriptor().getDimensions();
                                 const std::vector<uint64_t> weightDims = weightsTensor.getDescriptor().getDimensions();
                                 const DataType valueDType = valuesTensor.getDescriptor().getDataType();
                                 const DataType weightDType = weightsTensor.getDescriptor().getDataType();
                                 THOR_THROW_IF_FALSE(!valueDims.empty());
                                 THOR_THROW_IF_FALSE(valueDims == weightDims);
                                 ReductionMetricDType::validateValueDType("WeightedMean", "values", valueDType);
                                 ReductionMetricDType::validateValueDType("WeightedMean", "weights", weightDType);

                                 const std::vector<uint64_t> reductionAxes = allAxes(valueDims.size());
                                 const std::vector<uint64_t> squeezeAxes = squeezeAllButOneAxis(valueDims.size());
                                 Expression values = Expression::input("values", DataType::FP32, DataType::FP32);
                                 Expression weights = Expression::input("weights", DataType::FP32, DataType::FP32);
                                 Expression numerator = (values * weights).reduce_sum(reductionAxes, squeezeAxes, DataType::FP32);
                                 Expression denominator = weights.reduce_sum(reductionAxes, squeezeAxes, DataType::FP32);
                                 Expression weightedMean = numerator / denominator;
                                 Expression metric = Expression::where(denominator == Expression(0.0), Expression(0.0), weightedMean);

                                 ExpressionDefinition definition =
                                     ExpressionDefinition::fromOutputs(Expression::outputs({{"metric", metric}}));
                                 return DynamicExpressionBuild{
                                     std::make_shared<FusedEquation>(FusedEquation::compile(definition.outputs, stream.getGpuNum())),
                                     inputs,
                                     {},
                                     outputs,
                                     {},
                                 };
                             });
}

class UnaryReductionMetric : public CustomMetric {
   public:
    UnaryReductionMetric(DynamicExpression expr, std::string displayName)
        : CustomMetric(std::move(expr), "values", "", "metric", std::move(displayName)) {}

    ~UnaryReductionMetric() override = default;

    std::optional<Tensor> createFeatureOutputTensor() override {
        if (isInferenceOnly())
            return std::nullopt;
        validateValuesInput();
        return CustomMetric::createFeatureOutputTensor();
    }

    void compileImpl() override {
        if (!isInferenceOnly())
            validateValuesInput();
        CustomMetric::compileImpl();
    }

   protected:
    void validateValuesInput() const {
        THOR_THROW_IF_FALSE(featureInput.has_value());
        THOR_THROW_IF_FALSE(featureInput.value().isInitialized());
        THOR_THROW_IF_FALSE(featureInput.value().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        THOR_THROW_IF_FALSE(!featureInput.value().getDescriptor().getDimensions().empty());
        ReductionMetricDType::validateValueDType(
            "reduction metric", "values", featureInput.value().getDescriptor().getDataType());
    }
};

}  // namespace ReductionMetricDetail

class Mean : public ReductionMetricDetail::UnaryReductionMetric {
   public:
    Mean() : UnaryReductionMetric(ReductionMetricDetail::makeUnaryReductionExpression(ExprOp::REDUCE_AVG), "Mean") {}
    std::string getType() override { return "Mean"; }
};

class Sum : public ReductionMetricDetail::UnaryReductionMetric {
   public:
    Sum() : UnaryReductionMetric(ReductionMetricDetail::makeUnaryReductionExpression(ExprOp::REDUCE_SUM), "Sum") {}
    std::string getType() override { return "Sum"; }
};

class Min : public ReductionMetricDetail::UnaryReductionMetric {
   public:
    Min() : UnaryReductionMetric(ReductionMetricDetail::makeUnaryReductionExpression(ExprOp::REDUCE_MIN), "Min") {}
    std::string getType() override { return "Min"; }
};

class Max : public ReductionMetricDetail::UnaryReductionMetric {
   public:
    Max() : UnaryReductionMetric(ReductionMetricDetail::makeUnaryReductionExpression(ExprOp::REDUCE_MAX), "Max") {}
    std::string getType() override { return "Max"; }
};

class WeightedMean : public CustomMetric {
   public:
    WeightedMean() : CustomMetric(ReductionMetricDetail::makeWeightedMeanExpression(), "values", "weights", "metric", "Weighted Mean") {}

    ~WeightedMean() override = default;

    std::optional<Tensor> createFeatureOutputTensor() override {
        if (isInferenceOnly())
            return std::nullopt;
        validateWeightedMeanInputs();
        return CustomMetric::createFeatureOutputTensor();
    }

    void compileImpl() override {
        if (!isInferenceOnly())
            validateWeightedMeanInputs();
        CustomMetric::compileImpl();
    }

    std::string getType() override { return "WeightedMean"; }

   private:
    void validateWeightedMeanInputs() const {
        THOR_THROW_IF_FALSE(featureInput.has_value());
        THOR_THROW_IF_FALSE(labelsInput.has_value());
        THOR_THROW_IF_FALSE(featureInput.value().isInitialized());
        THOR_THROW_IF_FALSE(labelsInput.value().isInitialized());
        THOR_THROW_IF_FALSE(featureInput.value().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        THOR_THROW_IF_FALSE(labelsInput.value().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        THOR_THROW_IF_FALSE(featureInput.value().getPlacement() == labelsInput.value().getPlacement());
        THOR_THROW_IF_FALSE(!featureInput.value().getDescriptor().getDimensions().empty());
        THOR_THROW_IF_FALSE(featureInput.value().getDescriptor().getDimensions() == labelsInput.value().getDescriptor().getDimensions());
        ReductionMetricDType::validateValueDType(
            "WeightedMean", "values", featureInput.value().getDescriptor().getDataType());
        ReductionMetricDType::validateValueDType(
            "WeightedMean", "weights", labelsInput.value().getDescriptor().getDataType());
    }
};

}  // namespace ThorImplementation
