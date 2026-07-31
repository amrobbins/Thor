#pragma once

#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Layers/Metrics/CustomMetric.h"
#include "DeepLearning/Implementation/Layers/Metrics/ReductionMetricDType.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/FusedEquation.h"

#include <limits>
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

inline Expression reduceValidValues(Expression values,
                                    Expression validity,
                                    ExprOp op,
                                    const std::vector<uint64_t>& valueDimensions,
                                    const std::vector<uint64_t>& reductionAxes,
                                    const std::vector<uint64_t>& squeezeAxes) {
    switch (op) {
        case ExprOp::REDUCE_AVG: {
            uint64_t elementsPerExample = 1;
            for (size_t axis = 1; axis < valueDimensions.size(); ++axis)
                elementsPerExample *= valueDimensions[axis];
            Expression numerator = (values * validity).reduce_sum(reductionAxes, squeezeAxes, DataType::FP32);
            Expression validExamples = validity.reduce_sum(reductionAxes, squeezeAxes, DataType::FP32);
            return numerator / (validExamples * Expression::constantScalar(static_cast<double>(elementsPerExample)));
        }
        case ExprOp::REDUCE_SUM:
            return (values * validity).reduce_sum(reductionAxes, squeezeAxes, DataType::FP32);
        case ExprOp::REDUCE_MIN: {
            Expression selected = Expression::where(validity > Expression(0.0),
                                                    values,
                                                    Expression(std::numeric_limits<float>::max()));
            return selected.reduce_min(reductionAxes, squeezeAxes, DataType::FP32);
        }
        case ExprOp::REDUCE_MAX: {
            Expression selected = Expression::where(validity > Expression(0.0),
                                                    values,
                                                    Expression(std::numeric_limits<float>::lowest()));
            return selected.reduce_max(reductionAxes, squeezeAxes, DataType::FP32);
        }
        default:
            throw std::invalid_argument("Unsupported reduction metric op.");
    }
}

inline DynamicExpression makeUnaryReductionExpression(ExprOp op) {
    return DynamicExpression({"values", Thor::BATCH_VALIDITY_MASK_NAME},
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
                                 Expression validity =
                                     Expression::input(Thor::BATCH_VALIDITY_MASK_NAME, DataType::FP32, DataType::FP32);
                                 Expression metric =
                                     reduceValidValues(values, validity, op, valueDims, reductionAxes, squeezeAxes);

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
    return DynamicExpression({"values", "weights", Thor::BATCH_VALIDITY_MASK_NAME},
                             {"metric",
                              Thor::METRIC_AGGREGATION_NUMERATOR_NAME,
                              Thor::METRIC_AGGREGATION_DENOMINATOR_NAME},
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
                                 Expression validity =
                                     Expression::input(Thor::BATCH_VALIDITY_MASK_NAME, DataType::FP32, DataType::FP32);
                                 Expression effectiveWeights = weights * validity;
                                 Expression numerator =
                                     (values * effectiveWeights).reduce_sum(reductionAxes, squeezeAxes, DataType::FP32);
                                 Expression denominator = effectiveWeights.reduce_sum(reductionAxes, squeezeAxes, DataType::FP32);
                                 Expression weightedMean = numerator / denominator;
                                 Expression metric = Expression::where(denominator == Expression(0.0), Expression(0.0), weightedMean);

                                 ExpressionDefinition definition = ExpressionDefinition::fromOutputs(Expression::outputs({
                                     {"metric", metric},
                                     {Thor::METRIC_AGGREGATION_NUMERATOR_NAME, numerator},
                                     {Thor::METRIC_AGGREGATION_DENOMINATOR_NAME, denominator},
                                 }));
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
    UnaryReductionMetric(DynamicExpression expr,
                         std::string displayName,
                         Thor::MetricAggregation aggregation)
        : CustomMetric(std::move(expr),
                       "values",
                       "",
                       "metric",
                       std::move(displayName),
                       aggregation,
                       Thor::BATCH_VALIDITY_MASK_NAME) {}

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
    Mean()
        : UnaryReductionMetric(ReductionMetricDetail::makeUnaryReductionExpression(ExprOp::REDUCE_AVG),
                               "Mean",
                               Thor::MetricAggregation::MEAN_BY_EXAMPLE) {}
    std::string getType() override { return "Mean"; }
};

class Sum : public ReductionMetricDetail::UnaryReductionMetric {
   public:
    Sum()
        : UnaryReductionMetric(ReductionMetricDetail::makeUnaryReductionExpression(ExprOp::REDUCE_SUM),
                               "Sum",
                               Thor::MetricAggregation::SUM) {}
    std::string getType() override { return "Sum"; }
};

class Min : public ReductionMetricDetail::UnaryReductionMetric {
   public:
    Min()
        : UnaryReductionMetric(ReductionMetricDetail::makeUnaryReductionExpression(ExprOp::REDUCE_MIN),
                               "Min",
                               Thor::MetricAggregation::MIN) {}
    std::string getType() override { return "Min"; }
};

class Max : public ReductionMetricDetail::UnaryReductionMetric {
   public:
    Max()
        : UnaryReductionMetric(ReductionMetricDetail::makeUnaryReductionExpression(ExprOp::REDUCE_MAX),
                               "Max",
                               Thor::MetricAggregation::MAX) {}
    std::string getType() override { return "Max"; }
};

class WeightedMean : public CustomMetric {
   public:
    WeightedMean()
        : CustomMetric(ReductionMetricDetail::makeWeightedMeanExpression(),
                       "values",
                       "weights",
                       "metric",
                       "Weighted Mean",
                       Thor::MetricAggregation::RATIO,
                       Thor::BATCH_VALIDITY_MASK_NAME) {}

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
