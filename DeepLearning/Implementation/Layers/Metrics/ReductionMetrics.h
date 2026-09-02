#pragma once

#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Layers/Metrics/CustomMetric.h"
#include "DeepLearning/Implementation/Layers/Metrics/ReductionMetricDType.h"
#include "DeepLearning/Implementation/Tensor/RaggedTensorDescriptor.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionDescriptor.h"
#include "Utilities/TensorOperations/Ragged/RowPartition.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/FusedEquation.h"
#include "Utilities/Expression/RaggedExpression.h"

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
                                    DataType valueDType,
                                    const std::vector<uint64_t>& valueDimensions,
                                    const std::vector<uint64_t>& reductionAxes,
                                    const std::vector<uint64_t>& squeezeAxes) {
    switch (op) {
        case ExprOp::REDUCE_AVG: {
            uint64_t elementsPerExample = 1;
            for (size_t axis = 1; axis < valueDimensions.size(); ++axis)
                elementsPerExample *= valueDimensions[axis];
            // Multiplying by the FP32 validity mask is only a 0/1 selection. Publish that
            // materialized parent in the original value storage dtype so CUB can widen lazily
            // while accumulating, rather than creating a full FP32 compatibility tensor.
            Expression numerator =
                (values * validity).withOutputDType(valueDType).reduce_sum(reductionAxes, squeezeAxes, DataType::FP32);
            Expression validExamples = validity.reduce_sum(reductionAxes, squeezeAxes, DataType::FP32);
            return numerator / (validExamples * Expression::constantScalar(static_cast<double>(elementsPerExample)));
        }
        case ExprOp::REDUCE_SUM:
            return (values * validity).withOutputDType(valueDType).reduce_sum(reductionAxes, squeezeAxes, DataType::FP32);
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
                                 const DataType valueExpressionDType =
                                     (op == ExprOp::REDUCE_SUM || op == ExprOp::REDUCE_AVG) ? valueDType : DataType::FP32;
                                 Expression values = Expression::input("values", DataType::FP32, valueExpressionDType);
                                 Expression validity =
                                     Expression::input(Thor::BATCH_VALIDITY_MASK_NAME, DataType::FP32, DataType::FP32);
                                 Expression metric =
                                     reduceValidValues(values, validity, op, valueDType, valueDims, reductionAxes, squeezeAxes);

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
                                 Expression values = Expression::input("values", DataType::FP32, valueDType);
                                 Expression weights = Expression::input("weights", DataType::FP32, weightDType);
                                 Expression validity =
                                     Expression::input(Thor::BATCH_VALIDITY_MASK_NAME, DataType::FP32, DataType::FP32);
                                 // The validity multiply is an exact 0/1 mask of an already-quantized weight. Keep
                                 // that intermediate in weight storage and let CUB widen only while reducing it.
                                 Expression effectiveWeights = (weights * validity).withOutputDType(weightDType);
                                 // values * weights is real arithmetic, not merely a storage mask. Preserve the
                                 // existing FP32 product before reducing the weighted numerator.
                                 Expression numerator = (values * effectiveWeights)
                                                            .withOutputDType(DataType::FP32)
                                                            .reduce_sum(reductionAxes, squeezeAxes, DataType::FP32);
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

struct RaggedReductionRuntimeState {
    uint64_t validRowCount = 0;
};

inline uint64_t checkedRaggedElementsPerValue(const std::vector<uint64_t>& dimensions) {
    if (dimensions.empty())
        throw std::invalid_argument("Ragged reduction metric values must have rank >= 1.");
    uint64_t elements = 1;
    for (size_t axis = 1; axis < dimensions.size(); ++axis) {
        if (dimensions[axis] == 0 || elements > std::numeric_limits<uint64_t>::max() / dimensions[axis])
            throw std::invalid_argument("Ragged reduction metric trailing element count overflows uint64_t.");
        elements *= dimensions[axis];
    }
    return elements;
}

inline DynamicExpression makeRaggedReductionExpression(ExprOp op,
                                                       uint64_t batchSize,
                                                       uint64_t maxTotalValues,
                                                       std::shared_ptr<RaggedReductionRuntimeState> runtimeState) {
    const bool ratio = op == ExprOp::REDUCE_AVG;
    if (!ratio && op != ExprOp::REDUCE_SUM)
        throw std::invalid_argument("Ragged reduction metric supports only sum and mean.");

    std::vector<std::string> outputNames{"metric"};
    if (ratio) {
        outputNames.push_back(Thor::METRIC_AGGREGATION_NUMERATOR_NAME);
        outputNames.push_back(Thor::METRIC_AGGREGATION_DENOMINATOR_NAME);
    }

    return DynamicExpression(
        {"values", "offsets"},
        std::move(outputNames),
        [op, batchSize, maxTotalValues, runtimeState](const DynamicExpression::TensorMap& inputs,
                                                       const DynamicExpression::TensorMap& outputs,
                                                       Stream& stream) -> DynamicExpressionBuild {
            const Tensor& valuesTensor = inputs.at("values");
            const Tensor& offsetsTensor = inputs.at("offsets");
            const std::vector<uint64_t> valueDims = valuesTensor.getDimensions();
            const DataType valuesDType = valuesTensor.getDataType();
            const DataType offsetsDType = offsetsTensor.getDataType();
            if (valueDims.empty() || valueDims.front() != maxTotalValues)
                throw std::invalid_argument("Ragged reduction metric values must use max_total_values as the packed leading dimension.");
            ReductionMetricDType::validateValueDType("ragged reduction metric", "values", valuesDType);
            const RowPartitionDescriptor partition(batchSize, maxTotalValues, offsetsDType);
            if (offsetsTensor.getDescriptor() != partition.getOffsetsDescriptor())
                throw std::invalid_argument("Ragged reduction metric offsets must have canonical shape [batch_size + 1].");

            const uint64_t elementsPerValue = checkedRaggedElementsPerValue(valueDims);
            const std::vector<uint64_t> trailingDims(valueDims.begin() + 1, valueDims.end());
            const RaggedTensorDescriptor descriptor(valuesDType, trailingDims, batchSize, maxTotalValues, offsetsDType);

            Tensor effectiveOffsets = offsetsTensor.clone();
            Tensor activeScalarCount;
            if (op == ExprOp::REDUCE_AVG)
                activeScalarCount = Tensor(valuesTensor.getPlacement(), TensorDescriptor(DataType::FP32, {1}));

            const Expression values = Expression::input("values", valuesDType, valuesDType);
            const Expression effectiveOffsetsExpr =
                Expression::input("__thor_ragged_metric_effective_offsets", std::nullopt, offsetsDType);
            const RaggedExpression ragged(values, effectiveOffsetsExpr, descriptor);
            Expression perRow = ragged.segment_sum().withOutputDType(DataType::FP32);
            const std::vector<uint64_t> perRowDims = [&] {
                std::vector<uint64_t> dims{batchSize};
                dims.insert(dims.end(), trailingDims.begin(), trailingDims.end());
                return dims;
            }();
            const std::vector<uint64_t> reductionAxes = allAxes(perRowDims.size());
            const std::vector<uint64_t> squeezeAxes = squeezeAllButOneAxis(perRowDims.size());
            Expression total = perRow.reduce_sum(reductionAxes, squeezeAxes, DataType::FP32);

            PhysicalOutputs expressionOutputs;
            if (op == ExprOp::REDUCE_AVG) {
                const Expression denominator =
                    Expression::input("__thor_ragged_metric_active_scalar_count", DataType::FP32, DataType::FP32);
                const Expression metric = Expression::where(
                    denominator == Expression(0.0), Expression(0.0), total / denominator);
                expressionOutputs = Expression::outputs({
                    {"metric", metric},
                    {Thor::METRIC_AGGREGATION_NUMERATOR_NAME, total},
                    {Thor::METRIC_AGGREGATION_DENOMINATOR_NAME, denominator},
                }).physicalOutputs();
            } else {
                expressionOutputs = Expression::outputs({{"metric", total}}).physicalOutputs();
            }

            DynamicExpression::TensorMap stampInputs{{"values", valuesTensor},
                                                      {"__thor_ragged_metric_effective_offsets", effectiveOffsets}};
            if (op == ExprOp::REDUCE_AVG)
                stampInputs.emplace("__thor_ragged_metric_active_scalar_count", activeScalarCount);

            DynamicExpressionBuild build{
                .equation = std::make_shared<FusedEquation>(FusedEquation::compile(expressionOutputs, stream.getGpuNum())),
                .stamp_inputs = std::move(stampInputs),
                .tensor_scalar_inputs = {},
                .preallocated_outputs = outputs,
                .requested_output_shapes = {},
                .pre_forward_hook = [offsetsTensor,
                                     effectiveOffsets,
                                     activeScalarCount,
                                     batchSize,
                                     elementsPerValue,
                                     ratio = op == ExprOp::REDUCE_AVG,
                                     runtimeState](Stream& runStream) mutable {
                    if (!runtimeState || runtimeState->validRowCount == 0 || runtimeState->validRowCount > batchSize)
                        throw std::logic_error("Ragged reduction metric has invalid runtime valid-row count.");
                    rowPartitionClampOffsetsToValidRows(
                        offsetsTensor, effectiveOffsets, batchSize, runtimeState->validRowCount, runStream);
                    if (ratio) {
                        rowPartitionActiveScalarCount(
                            offsetsTensor, activeScalarCount, runtimeState->validRowCount, elementsPerValue, runStream);
                    }
                },
            };
            build.pre_forward_only_inputs.emplace("offsets", offsetsTensor);
            return build;
        });
}

}  // namespace ReductionMetricDetail

class RaggedReductionMetric : public CustomMetric {
   public:
    enum class Kind { SUM, MEAN };

    RaggedReductionMetric(Kind kind, uint64_t batchSize, uint64_t maxTotalValues)
        : RaggedReductionMetric(
              kind, batchSize, maxTotalValues, std::make_shared<ReductionMetricDetail::RaggedReductionRuntimeState>()) {}

    bool supportsPartialBatches() const override { return true; }

    std::optional<Tensor> connectToPreviousLayer(Layer* previousLayer,
                                                 std::optional<Tensor> input,
                                                 Stream inputStream,
                                                 bool backPropagateError,
                                                 int connectionType = 0) override {
        if (connectionType == static_cast<int>(ConnectionType::FORWARD))
            return connectToFeatureInputLayer(previousLayer, input, inputStream, backPropagateError);
        if (connectionType == static_cast<int>(ConnectionType::STRUCTURAL))
            return connectToLabelsInputLayer(previousLayer, input, inputStream);
        throw std::invalid_argument("Ragged reduction metric received an unsupported connection type.");
    }

    std::optional<Tensor> connectToFeatureInputLayer(Layer* featureInputLayer,
                                                     std::optional<Tensor> input,
                                                     Stream inputStream,
                                                     bool backPropagateError) override {
        (void)backPropagateError;
        if (!input.has_value())
            throw std::invalid_argument("Ragged reduction metric requires packed values.");
        if (featureInput.has_value())
            throw std::logic_error("Ragged reduction metric values are already connected.");
        const std::vector<uint64_t> dims = input->getDimensions();
        if (dims.empty() || dims.front() != maxTotalValues)
            throw std::invalid_argument("Ragged reduction metric values must use max_total_values as the packed leading dimension.");
        ReductionMetricDType::validateValueDType("ragged reduction metric", "values", input->getDataType());
        if (labelsInput.has_value() && labelsInput->getPlacement() != input->getPlacement())
            throw std::invalid_argument("Ragged reduction metric values and offsets must share placement.");
        Layer::connectToPreviousLayer(featureInputLayer, input, inputStream, false);
        return std::nullopt;
    }

    std::optional<Tensor> connectToLabelsInputLayer(Layer* labelsLayer,
                                                    std::optional<Tensor> offsets,
                                                    Stream offsetsStream) override {
        (void)labelsLayer;
        if (!offsets.has_value())
            throw std::invalid_argument("Ragged reduction metric requires structural offsets.");
        if (labelsInput.has_value())
            throw std::logic_error("Ragged reduction metric offsets are already connected.");
        const RowPartitionDescriptor partition(batchSize, maxTotalValues, offsets->getDataType());
        if (offsets->getDescriptor() != partition.getOffsetsDescriptor())
            throw std::invalid_argument("Ragged reduction metric offsets must have canonical shape [batch_size + 1].");
        if (featureInput.has_value() && featureInput->getPlacement() != offsets->getPlacement())
            throw std::invalid_argument("Ragged reduction metric values and offsets must share placement.");
        labelsInput = offsets;
        labelsStream = offsetsStream;
        return std::nullopt;
    }

    void forward(std::optional<Tensor> inputTensor, bool validationPass, uint32_t validExampleCount = 0) override {
        THOR_THROW_IF_FALSE(running);
        if (!inputTensor.has_value())
            throw std::invalid_argument("Ragged reduction metric forward requires a connected input tensor.");
        const uint32_t logicalBatch = static_cast<uint32_t>(batchSize);
        const uint32_t resolved = validExampleCount == 0 ? logicalBatch : validExampleCount;
        if (resolved == 0 || resolved > logicalBatch)
            throw std::invalid_argument("Ragged reduction metric valid example count exceeds logical batch size.");
        if (batchCardinalitySet && currentValidExampleCount != resolved)
            throw std::invalid_argument("Ragged reduction metric inputs disagreed on valid logical example count.");
        currentValidExampleCount = resolved;
        batchCardinalitySet = true;
        runtimeState->validRowCount = resolved;

        if (featureInput.has_value() && inputTensor.value() == featureInput.value()) {
            forwardFeatures(inputTensor.value(), validationPass);
        } else if (labelsInput.has_value() && inputTensor.value() == labelsInput.value()) {
            forwardLabels(inputTensor.value(), validationPass);
        } else {
            throw std::invalid_argument("Ragged reduction metric received an unconnected input tensor.");
        }
    }

    void computeMetric(Tensor labels,
                       Tensor predictions,
                       Tensor metric,
                       Stream runStream,
                       uint32_t validExampleCount) override {
        THOR_THROW_IF_FALSE(labelsInput.has_value() && labels == labelsInput.value());
        THOR_THROW_IF_FALSE(featureInput.has_value() && predictions == featureInput.value());
        THOR_THROW_IF_FALSE(featureOutput.has_value() && metric == featureOutput.value());
        THOR_THROW_IF_FALSE(validExampleCount >= 1 && validExampleCount <= batchSize);
        runtimeState->validRowCount = validExampleCount;
        runPreparedMetricExpression(runStream);
    }

    std::string getType() override { return kind == Kind::MEAN ? "RaggedMean" : "RaggedSum"; }

   private:
    RaggedReductionMetric(Kind kind,
                          uint64_t batchSize,
                          uint64_t maxTotalValues,
                          std::shared_ptr<ReductionMetricDetail::RaggedReductionRuntimeState> runtimeState)
        : CustomMetric(ReductionMetricDetail::makeRaggedReductionExpression(
                           kind == Kind::MEAN ? ExprOp::REDUCE_AVG : ExprOp::REDUCE_SUM,
                           batchSize,
                           maxTotalValues,
                           runtimeState),
                       "values",
                       "offsets",
                       "metric",
                       kind == Kind::MEAN ? "Mean" : "Sum",
                       kind == Kind::MEAN ? Thor::MetricAggregation::RATIO : Thor::MetricAggregation::SUM),
          kind(kind),
          batchSize(batchSize),
          maxTotalValues(maxTotalValues),
          runtimeState(std::move(runtimeState)) {
        if (batchSize == 0 || batchSize > std::numeric_limits<uint32_t>::max())
            throw std::invalid_argument("Ragged reduction metric logical batch size must fit uint32 and be non-zero.");
        if (maxTotalValues == 0)
            throw std::invalid_argument("Ragged reduction metric max_total_values must be non-zero.");
        this->runtimeState->validRowCount = batchSize;
    }

    Kind kind;
    uint64_t batchSize;
    uint64_t maxTotalValues;
    std::shared_ptr<ReductionMetricDetail::RaggedReductionRuntimeState> runtimeState;
};

namespace ReductionMetricDetail {

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
