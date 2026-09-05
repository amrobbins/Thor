#pragma once

#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Layers/Metrics/CustomMetric.h"
#include "DeepLearning/Implementation/Layers/Metrics/ReductionMetricDType.h"
#include "DeepLearning/Implementation/Tensor/RaggedTensorDescriptor.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionDescriptor.h"
#include "Utilities/TensorOperations/Ragged/RowPartition.h"
#include "Utilities/TensorOperations/Ragged/RaggedWeightedReduction.h"
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
    // R10M extrema publish this structural count to the reporting path so an
    // all-empty batch can be represented as no contribution rather than by an
    // identity sentinel or fake zero.
    Tensor activeScalarCount;
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
    const bool extrema = op == ExprOp::REDUCE_MIN || op == ExprOp::REDUCE_MAX;
    if (!ratio && !extrema && op != ExprOp::REDUCE_SUM)
        throw std::invalid_argument("Ragged reduction metric supports only sum, mean, min, and max.");

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

            const bool ratio = op == ExprOp::REDUCE_AVG;
            const bool extrema = op == ExprOp::REDUCE_MIN || op == ExprOp::REDUCE_MAX;
            const uint64_t elementsPerValue = checkedRaggedElementsPerValue(valueDims);
            const std::vector<uint64_t> trailingDims(valueDims.begin() + 1, valueDims.end());
            const RaggedTensorDescriptor descriptor(valuesDType, trailingDims, batchSize, maxTotalValues, offsetsDType);

            Tensor effectiveOffsets = offsetsTensor.clone();
            Tensor activeScalarCount;
            if (ratio || extrema) {
                activeScalarCount = Tensor(valuesTensor.getPlacement(), TensorDescriptor(DataType::FP32, {1}));
                if (extrema)
                    runtimeState->activeScalarCount = activeScalarCount;
            }

            const Expression values = Expression::input("values", valuesDType, valuesDType);
            const Expression effectiveOffsetsExpr =
                Expression::input("__thor_ragged_metric_effective_offsets", std::nullopt, offsetsDType);
            const RaggedExpression ragged(values, effectiveOffsetsExpr, descriptor);
            const Expression perRow = [&]() -> Expression {
                switch (op) {
                    case ExprOp::REDUCE_SUM:
                    case ExprOp::REDUCE_AVG:
                        return ragged.segment_sum().withOutputDType(DataType::FP32);
                    case ExprOp::REDUCE_MIN:
                        return ragged.segment_min().withOutputDType(DataType::FP32);
                    case ExprOp::REDUCE_MAX:
                        return ragged.segment_max().withOutputDType(DataType::FP32);
                    default:
                        THOR_UNREACHABLE();
                }
            }();
            const std::vector<uint64_t> perRowDims = [&] {
                std::vector<uint64_t> dims{batchSize};
                dims.insert(dims.end(), trailingDims.begin(), trailingDims.end());
                return dims;
            }();
            const std::vector<uint64_t> reductionAxes = allAxes(perRowDims.size());
            const std::vector<uint64_t> squeezeAxes = squeezeAllButOneAxis(perRowDims.size());
            const Expression total =
                op == ExprOp::REDUCE_MIN ? perRow.reduce_min(reductionAxes, squeezeAxes, DataType::FP32) :
                op == ExprOp::REDUCE_MAX ? perRow.reduce_max(reductionAxes, squeezeAxes, DataType::FP32) :
                                           perRow.reduce_sum(reductionAxes, squeezeAxes, DataType::FP32);

            PhysicalOutputs expressionOutputs;
            if (ratio) {
                const Expression denominator =
                    Expression::input("__thor_ragged_metric_active_scalar_count", DataType::FP32, DataType::FP32);
                const Expression metric = Expression::where(
                    denominator == Expression(0.0), Expression(0.0), total / denominator);
                expressionOutputs = Expression::outputs({
                    {"metric", metric},
                    {Thor::METRIC_AGGREGATION_NUMERATOR_NAME, total},
                    {Thor::METRIC_AGGREGATION_DENOMINATOR_NAME, denominator},
                }).physicalOutputs();
            } else if (extrema) {
                const Expression contributionCount =
                    Expression::input("__thor_ragged_metric_active_scalar_count", DataType::FP32, DataType::FP32);
                // The public scalar must remain a concrete tensor for graph execution,
                // but the separate contribution count is the aggregation contract.
                // Zero here is never interpreted as an extrema contribution.
                const Expression metric = Expression::where(
                    contributionCount == Expression(0.0), Expression(0.0), total);
                expressionOutputs = Expression::outputs({{"metric", metric}}).physicalOutputs();
            } else {
                expressionOutputs = Expression::outputs({{"metric", total}}).physicalOutputs();
            }

            DynamicExpression::TensorMap stampInputs{{"values", valuesTensor},
                                                      {"__thor_ragged_metric_effective_offsets", effectiveOffsets}};
            if (ratio || extrema)
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
                                     needsCount = ratio || extrema,
                                     runtimeState](Stream& runStream) mutable {
                    if (!runtimeState || runtimeState->validRowCount == 0 || runtimeState->validRowCount > batchSize)
                        throw std::logic_error("Ragged reduction metric has invalid runtime valid-row count.");
                    rowPartitionClampOffsetsToValidRows(
                        offsetsTensor, effectiveOffsets, batchSize, runtimeState->validRowCount, runStream);
                    if (needsCount) {
                        rowPartitionActiveScalarCount(
                            offsetsTensor, activeScalarCount, runtimeState->validRowCount, elementsPerValue, runStream);
                    }
                },
            };
            build.pre_forward_only_inputs.emplace("offsets", offsetsTensor);
            return build;
        });
 }

struct RaggedWeightedMeanRuntimeState {
    uint64_t validRowCount = 0;
};

inline DynamicExpression makeRaggedWeightedMeanExpression(
    uint64_t batchSize,
    uint64_t maxTotalValues,
    std::shared_ptr<RaggedWeightedMeanRuntimeState> runtimeState) {
    return DynamicExpression(
        {"values", "weights", "offsets"},
        {"metric", Thor::METRIC_AGGREGATION_NUMERATOR_NAME, Thor::METRIC_AGGREGATION_DENOMINATOR_NAME},
        [batchSize, maxTotalValues, runtimeState](const DynamicExpression::TensorMap& inputs,
                                                  const DynamicExpression::TensorMap& outputs,
                                                  Stream& stream) -> DynamicExpressionBuild {
            const Tensor& valuesTensor = inputs.at("values");
            const Tensor& weightsTensor = inputs.at("weights");
            const Tensor& offsetsTensor = inputs.at("offsets");
            const std::vector<uint64_t> valueDims = valuesTensor.getDimensions();
            const std::vector<uint64_t> weightDims = weightsTensor.getDimensions();
            const DataType valueDType = valuesTensor.getDataType();
            const DataType weightDType = weightsTensor.getDataType();
            const DataType offsetsDType = offsetsTensor.getDataType();
            if (valueDims.empty() || valueDims.front() != maxTotalValues)
                throw std::invalid_argument("Ragged WeightedMean values must use max_total_values as the packed leading dimension.");
            if (weightDims != valueDims)
                throw std::invalid_argument("Ragged WeightedMean values and weights packed dimensions must match.");
            ReductionMetricDType::validateValueDType("WeightedMean", "values", valueDType);
            ReductionMetricDType::validateValueDType("WeightedMean", "weights", weightDType);
            const RowPartitionDescriptor partition(batchSize, maxTotalValues, offsetsDType);
            if (offsetsTensor.getDescriptor() != partition.getOffsetsDescriptor())
                throw std::invalid_argument("Ragged WeightedMean offsets must have canonical shape [batch_size + 1].");
            const uint64_t elementsPerValue = checkedRaggedElementsPerValue(valueDims);

            // Compute sufficient statistics directly from the active prefix. A
            // pointwise values*weights expression would materialize over the
            // full packed capacity before segmented reduction and therefore
            // read undefined inactive storage. These scalars are produced by an
            // active-prefix kernel instead.
            Tensor numeratorStatistic(valuesTensor.getPlacement(), TensorDescriptor(DataType::FP32, {1}));
            Tensor denominatorStatistic(valuesTensor.getPlacement(), TensorDescriptor(DataType::FP32, {1}));
            const Expression numerator = Expression::input(
                "__thor_ragged_weighted_mean_numerator", DataType::FP32, DataType::FP32);
            const Expression denominator = Expression::input(
                "__thor_ragged_weighted_mean_denominator", DataType::FP32, DataType::FP32);
            const Expression metric = Expression::where(
                denominator == Expression(0.0), Expression(0.0), numerator / denominator);
            const PhysicalOutputs expressionOutputs = Expression::outputs({
                {"metric", metric},
                {Thor::METRIC_AGGREGATION_NUMERATOR_NAME, numerator},
                {Thor::METRIC_AGGREGATION_DENOMINATOR_NAME, denominator},
            }).physicalOutputs();

            DynamicExpressionBuild build{
                .equation = std::make_shared<FusedEquation>(FusedEquation::compile(expressionOutputs, stream.getGpuNum())),
                .stamp_inputs = {{"__thor_ragged_weighted_mean_numerator", numeratorStatistic},
                                 {"__thor_ragged_weighted_mean_denominator", denominatorStatistic}},
                .tensor_scalar_inputs = {},
                .preallocated_outputs = outputs,
                .requested_output_shapes = {},
                .pre_forward_hook = [valuesTensor,
                                     weightsTensor,
                                     offsetsTensor,
                                     numeratorStatistic,
                                     denominatorStatistic,
                                     batchSize,
                                     maxTotalValues,
                                     elementsPerValue,
                                     runtimeState](Stream& runStream) mutable {
                    if (!runtimeState || runtimeState->validRowCount == 0 || runtimeState->validRowCount > batchSize)
                        throw std::logic_error("Ragged WeightedMean has invalid runtime valid-row count.");
                    raggedWeightedMeanStatistics(valuesTensor,
                                                 weightsTensor,
                                                 offsetsTensor,
                                                 numeratorStatistic,
                                                 denominatorStatistic,
                                                 runtimeState->validRowCount,
                                                 maxTotalValues,
                                                 elementsPerValue,
                                                 runStream);
                },
            };
            build.pre_forward_only_inputs.emplace("values", valuesTensor);
            build.pre_forward_only_inputs.emplace("weights", weightsTensor);
            build.pre_forward_only_inputs.emplace("offsets", offsetsTensor);
            return build;
        });
}

}  // namespace ReductionMetricDetail

class RaggedReductionMetric : public CustomMetric {
   public:
    enum class Kind { SUM, MEAN, MIN, MAX };

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
        if (isExtrema())
            captureContributionStatistic();
    }

    std::string getType() override {
        switch (kind) {
            case Kind::SUM: return "RaggedSum";
            case Kind::MEAN: return "RaggedMean";
            case Kind::MIN: return "RaggedMin";
            case Kind::MAX: return "RaggedMax";
        }
        THOR_UNREACHABLE();
    }

    void preallocateMetricStatisticSlots(uint32_t numSlots) override {
        CustomMetric::preallocateMetricStatisticSlots(numSlots);
        if (!isExtrema())
            return;
        THOR_THROW_IF_FALSE(numSlots >= 1);
        THOR_THROW_IF_FALSE(runtimeState->activeScalarCount.isInitialized());
        if (!contributionStatisticDownloadStream.has_value()) {
            contributionStatisticDownloadStream =
                Stream::getNextDownloadStream(runtimeState->activeScalarCount.getPlacement().getDeviceNum());
        }
        const TensorPlacement hostPlacement(TensorPlacement::MemDevices::CPU);
        while (contributionStatisticSlots.size() < numSlots) {
            ContributionStatisticSlot slot;
            slot.host = runtimeState->activeScalarCount.clone(hostPlacement);
            slot.buffer = runtimeState->activeScalarCount.clone();
            contributionStatisticSlots.push_back(std::move(slot));
        }
    }

    void setActiveMetricStatisticSlot(uint32_t slotIndex) override {
        CustomMetric::setActiveMetricStatisticSlot(slotIndex);
        if (!isExtrema())
            return;
        // Direct PlacedNetwork::infer() does not preallocate reporting slots.
        // Selecting slot zero must therefore remain a no-op for the hidden
        // contribution statistic unless reporting explicitly requested slots.
        // Queued/reporting paths preallocate first and retain strict bounds.
        if (contributionStatisticSlots.empty()) {
            activeContributionStatisticSlot = slotIndex;
            return;
        }
        requireContributionStatisticSlot(slotIndex);
        activeContributionStatisticSlot = slotIndex;
    }

    std::optional<MetricBatchStatisticTensors> getMetricBatchStatisticTensorsForSlot(uint32_t slotIndex) const override {
        if (!isExtrema())
            return CustomMetric::getMetricBatchStatisticTensorsForSlot(slotIndex);
        requireContributionStatisticSlot(slotIndex);
        const ContributionStatisticSlot& slot = contributionStatisticSlots[slotIndex];
        const Thor::MetricAggregation aggregation =
            kind == Kind::MIN ? Thor::MetricAggregation::MIN : Thor::MetricAggregation::MAX;
        return MetricBatchStatisticTensors{
            aggregation, std::nullopt, std::nullopt, slot.readyEvent, slot.host};
    }

    void extendMetricStatisticWritableEventForSlot(uint32_t slotIndex, Event event) override {
        CustomMetric::extendMetricStatisticWritableEventForSlot(slotIndex, event);
        if (!isExtrema() || contributionStatisticSlots.empty())
            return;
        requireContributionStatisticSlot(slotIndex);
        contributionStatisticSlots[slotIndex].writableEvent = event;
    }

    std::vector<Event> getSynchronizeEvents() override {
        std::vector<Event> events = CustomMetric::getSynchronizeEvents();
        if (contributionStatisticDownloadStream.has_value())
            events.emplace_back(contributionStatisticDownloadStream.value().putEvent(false, true));
        return events;
    }

    void cleanup() override {
        for (ContributionStatisticSlot& slot : contributionStatisticSlots) {
            slot.host.dropReference();
            slot.buffer.dropReference();
        }
        contributionStatisticSlots.clear();
        contributionStatisticDownloadStream.reset();
        if (runtimeState->activeScalarCount.isInitialized())
            runtimeState->activeScalarCount.dropReference();
        CustomMetric::cleanup();
    }

   private:
    struct ContributionStatisticSlot {
        Tensor host;
        Tensor buffer;
        Event bufferReadyEvent;
        Event readyEvent;
        Event writableEvent;
    };

    bool isExtrema() const { return kind == Kind::MIN || kind == Kind::MAX; }

    void requireContributionStatisticSlot(uint32_t slotIndex) const {
        THOR_THROW_IF_FALSE(isExtrema());
        THOR_THROW_IF_FALSE(slotIndex < contributionStatisticSlots.size());
    }

    void captureContributionStatistic() {
        // Metric statistics are an optional reporting side channel. Ordinary
        // direct inference still computes the concrete Min/Max graph output
        // without allocating or downloading that side channel.
        if (contributionStatisticSlots.empty())
            return;
        requireContributionStatisticSlot(activeContributionStatisticSlot);
        THOR_THROW_IF_FALSE(contributionStatisticDownloadStream.has_value());
        THOR_THROW_IF_FALSE(runtimeState->activeScalarCount.isInitialized());
        ContributionStatisticSlot& slot = contributionStatisticSlots[activeContributionStatisticSlot];

        if (slot.readyEvent.isInitialized())
            stream.waitEvent(slot.readyEvent);
        slot.buffer.copyFromAsync(runtimeState->activeScalarCount, stream);
        stream.putEvent(slot.bufferReadyEvent);

        Stream& downloadStream = contributionStatisticDownloadStream.value();
        downloadStream.waitEvent(slot.bufferReadyEvent);
        if (slot.writableEvent.isInitialized())
            downloadStream.waitEvent(slot.writableEvent);
        slot.host.copyFromAsync(slot.buffer, downloadStream);
        downloadStream.putEvent(slot.readyEvent, false, true);
        slot.writableEvent = slot.readyEvent;
    }

    RaggedReductionMetric(Kind kind,
                          uint64_t batchSize,
                          uint64_t maxTotalValues,
                          std::shared_ptr<ReductionMetricDetail::RaggedReductionRuntimeState> runtimeState)
        : CustomMetric(ReductionMetricDetail::makeRaggedReductionExpression(
                           kind == Kind::MEAN ? ExprOp::REDUCE_AVG :
                           kind == Kind::MIN ? ExprOp::REDUCE_MIN :
                           kind == Kind::MAX ? ExprOp::REDUCE_MAX : ExprOp::REDUCE_SUM,
                           batchSize,
                           maxTotalValues,
                           runtimeState),
                       "values",
                       "offsets",
                       "metric",
                       kind == Kind::MEAN ? "Mean" :
                       kind == Kind::MIN ? "Min" :
                       kind == Kind::MAX ? "Max" : "Sum",
                       kind == Kind::MEAN ? Thor::MetricAggregation::RATIO :
                       kind == Kind::MIN ? Thor::MetricAggregation::MIN :
                       kind == Kind::MAX ? Thor::MetricAggregation::MAX : Thor::MetricAggregation::SUM),
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
    uint32_t activeContributionStatisticSlot = 0;
    std::vector<ContributionStatisticSlot> contributionStatisticSlots;
    std::optional<Stream> contributionStatisticDownloadStream;
};

class RaggedWeightedMean : public CustomMetric {
   public:
    RaggedWeightedMean(uint64_t batchSize, uint64_t maxTotalValues)
        : RaggedWeightedMean(batchSize,
                             maxTotalValues,
                             std::make_shared<ReductionMetricDetail::RaggedWeightedMeanRuntimeState>()) {}

    bool supportsPartialBatches() const override { return true; }

    std::optional<Tensor> connectToPreviousLayer(Layer* previousLayer,
                                                 std::optional<Tensor> input,
                                                 Stream inputStream,
                                                 bool backPropagateError,
                                                 int connectionType = 0) override {
        if (connectionType == static_cast<int>(ConnectionType::FORWARD))
            return connectValues(previousLayer, input, inputStream, backPropagateError);
        if (connectionType == static_cast<int>(ConnectionType::LABELS))
            return connectWeights(previousLayer, input, inputStream);
        if (connectionType == static_cast<int>(ConnectionType::STRUCTURAL))
            return connectOffsets(previousLayer, input, inputStream);
        throw std::invalid_argument("Ragged WeightedMean received an unsupported connection type.");
    }

    std::vector<Stream> getProcessingStreams() override {
        std::vector<Stream> streams = CustomMetric::getProcessingStreams();
        if (offsetsStream.isInitialized())
            streams.push_back(offsetsStream);
        return streams;
    }

    std::vector<Event> getSynchronizeEvents() override {
        std::vector<Event> events = CustomMetric::getSynchronizeEvents();
        if (offsetsStream.isInitialized())
            events.emplace_back(offsetsStream.putEvent(false, true));
        return events;
    }

    void initialize() override {
        CustomMetric::initialize();
        offsetsReceived = false;
    }

    void cleanup() override {
        offsetsReadyEvent = Event();
        offsetsReusableEvent = Event();
        CustomMetric::cleanup();
    }

    void forward(std::optional<Tensor> inputTensor,
                 bool validationPass,
                 uint32_t validExampleCount = 0) override {
        THOR_THROW_IF_FALSE(running);
        if (!inputTensor.has_value())
            throw std::invalid_argument("Ragged WeightedMean forward requires a connected input tensor.");
        const uint32_t logicalBatch = static_cast<uint32_t>(batchSize);
        const uint32_t resolved = validExampleCount == 0 ? logicalBatch : validExampleCount;
        if (resolved == 0 || resolved > logicalBatch)
            throw std::invalid_argument("Ragged WeightedMean valid example count exceeds logical batch size.");
        if (batchCardinalitySet && currentValidExampleCount != resolved)
            throw std::invalid_argument("Ragged WeightedMean inputs disagreed on valid logical example count.");
        currentValidExampleCount = resolved;
        batchCardinalitySet = true;
        runtimeState->validRowCount = resolved;

        if (featureInput.has_value() && inputTensor.value() == featureInput.value()) {
            forwardFeatures(inputTensor.value(), validationPass);
        } else if (labelsInput.has_value() && inputTensor.value() == labelsInput.value()) {
            forwardLabels(inputTensor.value(), validationPass);
        } else if (offsetsInput.has_value() && inputTensor.value() == offsetsInput.value()) {
            if (offsetsReceived)
                throw std::logic_error("Ragged WeightedMean structural offsets were delivered twice for one batch.");
            offsetsReceived = true;
            advanceDataIfReady(validationPass);
        } else {
            throw std::invalid_argument("Ragged WeightedMean received an unconnected input tensor.");
        }
    }

    void computeMetric(Tensor weights,
                       Tensor values,
                       Tensor metric,
                       Stream runStream,
                       uint32_t validExampleCount) override {
        THOR_THROW_IF_FALSE(labelsInput.has_value() && weights == labelsInput.value());
        THOR_THROW_IF_FALSE(featureInput.has_value() && values == featureInput.value());
        THOR_THROW_IF_FALSE(featureOutput.has_value() && metric == featureOutput.value());
        THOR_THROW_IF_FALSE(validExampleCount >= 1 && validExampleCount <= batchSize);
        runtimeState->validRowCount = validExampleCount;
        runPreparedMetricExpression(runStream);
    }

    std::string getType() override { return "RaggedWeightedMean"; }

    std::optional<MetricBatchStatisticTensors> getMetricBatchStatisticTensorsForSlot(uint32_t slotIndex) const override {
        std::optional<MetricBatchStatisticTensors> statistics =
            CustomMetric::getMetricBatchStatisticTensorsForSlot(slotIndex);
        THOR_THROW_IF_FALSE(statistics.has_value());
        THOR_THROW_IF_FALSE(statistics->aggregation == Thor::MetricAggregation::RATIO);
        statistics->zeroDenominatorMeansNoContribution = true;
        return statistics;
    }

   protected:
    TensorMap buildMetricInputs() const override {
        TensorMap inputs = CustomMetric::buildMetricInputs();
        if (!offsetsInput.has_value())
            throw std::logic_error("Ragged WeightedMean structural offsets are not connected.");
        inputs.emplace("offsets", offsetsInput.value());
        return inputs;
    }

    void compileImpl() override {
        validateInputs();
        CustomMetric::compileImpl();
    }

    void advanceDataIfReady(bool validationPass) override {
        if (!(featureInputReceived && labelsReceived && offsetsReceived))
            return;
        THOR_THROW_IF_FALSE(batchCardinalitySet);
        THOR_THROW_IF_FALSE(labelsInput.has_value());
        THOR_THROW_IF_FALSE(offsetsInput.has_value());

        waitForLabelsReady();
        stream.waitFor(offsetsStream, offsetsReadyEvent);
        computeMetric(labelsInput.value(), featureInput.value(), featureOutput.value(), stream, currentValidExampleCount);
        markLabelsReusableAfterCompute();
        offsetsStream.waitFor(stream, offsetsReusableEvent);

        featureInputReceived = false;
        labelsReceived = false;
        offsetsReceived = false;
        batchCardinalitySet = false;

        if (nextLayer.has_value())
            nextLayer.value()->forward(featureOutput, validationPass, currentValidExampleCount);
    }

   private:
    RaggedWeightedMean(uint64_t batchSize,
                       uint64_t maxTotalValues,
                       std::shared_ptr<ReductionMetricDetail::RaggedWeightedMeanRuntimeState> runtimeState)
        : CustomMetric(ReductionMetricDetail::makeRaggedWeightedMeanExpression(
                           batchSize, maxTotalValues, runtimeState),
                       "values",
                       "weights",
                       "metric",
                       "WeightedMean",
                       Thor::MetricAggregation::RATIO),
          batchSize(batchSize),
          maxTotalValues(maxTotalValues),
          runtimeState(std::move(runtimeState)) {
        if (batchSize == 0 || batchSize > std::numeric_limits<uint32_t>::max())
            throw std::invalid_argument("Ragged WeightedMean logical batch size must fit uint32 and be non-zero.");
        if (maxTotalValues == 0)
            throw std::invalid_argument("Ragged WeightedMean max_total_values must be non-zero.");
        this->runtimeState->validRowCount = batchSize;
    }

    std::optional<Tensor> connectValues(Layer* previousLayer,
                                        std::optional<Tensor> input,
                                        Stream inputStream,
                                        bool backPropagateError) {
        (void)backPropagateError;
        if (!input.has_value())
            throw std::invalid_argument("Ragged WeightedMean requires packed values.");
        if (featureInput.has_value())
            throw std::logic_error("Ragged WeightedMean values are already connected.");
        validatePackedTensor("values", input.value());
        if (labelsInput.has_value() && labelsInput->getDimensions() != input->getDimensions())
            throw std::invalid_argument("Ragged WeightedMean packed values and weights dimensions must match.");
        if (labelsInput.has_value() && labelsInput->getPlacement() != input->getPlacement())
            throw std::invalid_argument("Ragged WeightedMean values and weights must share placement.");
        if (offsetsInput.has_value() && offsetsInput->getPlacement() != input->getPlacement())
            throw std::invalid_argument("Ragged WeightedMean values and offsets must share placement.");
        Layer::connectToPreviousLayer(previousLayer, input, inputStream, false);
        return std::nullopt;
    }

    std::optional<Tensor> connectWeights(Layer* previousLayer,
                                         std::optional<Tensor> input,
                                         Stream inputStream) {
        (void)previousLayer;
        if (!input.has_value())
            throw std::invalid_argument("Ragged WeightedMean requires packed weights.");
        if (labelsInput.has_value())
            throw std::logic_error("Ragged WeightedMean weights are already connected.");
        validatePackedTensor("weights", input.value());
        if (featureInput.has_value() && featureInput->getDimensions() != input->getDimensions())
            throw std::invalid_argument("Ragged WeightedMean packed values and weights dimensions must match.");
        if (featureInput.has_value() && featureInput->getPlacement() != input->getPlacement())
            throw std::invalid_argument("Ragged WeightedMean values and weights must share placement.");
        if (offsetsInput.has_value() && offsetsInput->getPlacement() != input->getPlacement())
            throw std::invalid_argument("Ragged WeightedMean weights and offsets must share placement.");
        labelsInput = input;
        labelsStream = inputStream;
        return std::nullopt;
    }

    std::optional<Tensor> connectOffsets(Layer* previousLayer,
                                         std::optional<Tensor> input,
                                         Stream inputStream) {
        (void)previousLayer;
        if (!input.has_value())
            throw std::invalid_argument("Ragged WeightedMean requires structural offsets.");
        if (offsetsInput.has_value())
            throw std::logic_error("Ragged WeightedMean offsets are already connected.");
        const RowPartitionDescriptor partition(batchSize, maxTotalValues, input->getDataType());
        if (input->getDescriptor() != partition.getOffsetsDescriptor())
            throw std::invalid_argument("Ragged WeightedMean offsets must have canonical shape [batch_size + 1].");
        if (featureInput.has_value() && featureInput->getPlacement() != input->getPlacement())
            throw std::invalid_argument("Ragged WeightedMean values and offsets must share placement.");
        if (labelsInput.has_value() && labelsInput->getPlacement() != input->getPlacement())
            throw std::invalid_argument("Ragged WeightedMean weights and offsets must share placement.");
        offsetsInput = input;
        offsetsStream = inputStream;
        return std::nullopt;
    }

    void validatePackedTensor(const char* name, const Tensor& tensor) const {
        const std::vector<uint64_t> dims = tensor.getDimensions();
        if (dims.empty() || dims.front() != maxTotalValues)
            throw std::invalid_argument(std::string("Ragged WeightedMean ") + name +
                                        " must use max_total_values as the packed leading dimension.");
        ReductionMetricDType::validateValueDType("WeightedMean", name, tensor.getDataType());
    }

    void validateInputs() const {
        if (!featureInput.has_value() || !labelsInput.has_value() || !offsetsInput.has_value())
            throw std::logic_error("Ragged WeightedMean requires values, weights, and structural offsets.");
        validatePackedTensor("values", featureInput.value());
        validatePackedTensor("weights", labelsInput.value());
        if (featureInput->getDimensions() != labelsInput->getDimensions())
            throw std::invalid_argument("Ragged WeightedMean packed values and weights dimensions must match.");
        const RowPartitionDescriptor partition(batchSize, maxTotalValues, offsetsInput->getDataType());
        if (offsetsInput->getDescriptor() != partition.getOffsetsDescriptor())
            throw std::invalid_argument("Ragged WeightedMean offsets must have canonical shape [batch_size + 1].");
        if (featureInput->getPlacement() != labelsInput->getPlacement() ||
            featureInput->getPlacement() != offsetsInput->getPlacement())
            throw std::invalid_argument("Ragged WeightedMean inputs must share placement.");
    }

    uint64_t batchSize;
    uint64_t maxTotalValues;
    std::shared_ptr<ReductionMetricDetail::RaggedWeightedMeanRuntimeState> runtimeState;
    std::optional<Tensor> offsetsInput;
    Stream offsetsStream;
    Event offsetsReadyEvent;
    Event offsetsReusableEvent;
    bool offsetsReceived = false;
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
