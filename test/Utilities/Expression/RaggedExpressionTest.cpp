#include "Utilities/Expression/RaggedExpression.h"

#include "Utilities/Expression/AutoDiff.h"

#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/FusedEquation.h"

#include "cuda_runtime.h"

#include "gtest/gtest.h"

#include <cmath>
#include <cstdint>
#include <optional>
#include <set>
#include <unordered_map>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ThorImplementation;

namespace {

RaggedTensorDescriptor makeDescriptor(DataType values_dtype = DataType::FP32,
                                      std::vector<uint64_t> trailing_dimensions = {4},
                                      uint64_t batch_size = 3,
                                      uint64_t max_total_values = 9,
                                      DataType offsets_dtype = DataType::UINT32) {
    return RaggedTensorDescriptor(values_dtype, trailing_dimensions, batch_size, max_total_values, offsets_dtype);
}

ExprNode outputNode(const Expression& expression) {
    const PhysicalExpression physical = expression.expression();
    return physical.nodes.at(physical.output_node);
}

struct MarkedValueNodes {
    ExprNode marker;
    ExprNode values;
};

MarkedValueNodes markedValueNodes(const Expression& expression) {
    const PhysicalExpression physical = expression.expression();
    const ExprNode marker = physical.nodes.at(physical.output_node);
    if (marker.op != ExprOp::RAGGED_VALUEWISE_EXTENT) {
        throw std::runtime_error("test expected a ragged runtime extent marker.");
    }
    return MarkedValueNodes{marker, physical.nodes.at(marker.lhs)};
}

#define REQUIRE_CUDA_DEVICE()                                                                                          \
    do {                                                                                                                \
        int cuda_device_count_for_test = 0;                                                                             \
        const cudaError_t cuda_status_for_test = cudaGetDeviceCount(&cuda_device_count_for_test);                       \
        if (cuda_status_for_test != cudaSuccess || cuda_device_count_for_test <= 0) {                                    \
            GTEST_SKIP() << "CUDA device is required for ragged expression execution tests.";                          \
        }                                                                                                               \
    } while (false)

TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

uint64_t tensorNumel(const Tensor& tensor) {
    uint64_t numel = 1;
    for (uint64_t dim : tensor.getDimensions()) {
        numel *= dim;
    }
    return numel;
}

template <typename T>
DataType dtypeFor();

template <>
DataType dtypeFor<float>() {
    return DataType::FP32;
}

template <>
DataType dtypeFor<uint32_t>() {
    return DataType::UINT32;
}

template <typename T>
Tensor makeGpuTensor(const std::vector<uint64_t>& dims, const std::vector<T>& values, Stream& stream) {
    Tensor cpu(cpuPlacement, TensorDescriptor(dtypeFor<T>(), dims));
    if (tensorNumel(cpu) != values.size()) {
        throw std::runtime_error("makeGpuTensor value count mismatch.");
    }
    T* ptr = cpu.getMemPtr<T>();
    for (size_t i = 0; i < values.size(); ++i) {
        ptr[i] = values[i];
    }

    Tensor gpu(gpuPlacement, TensorDescriptor(dtypeFor<T>(), dims));
    gpu.copyFromAsync(cpu, stream);
    stream.synchronize();
    return gpu;
}

template <typename T>
void overwriteGpuTensor(Tensor& gpu, const std::vector<T>& values, Stream& stream) {
    Tensor cpu(cpuPlacement, TensorDescriptor(dtypeFor<T>(), gpu.getDimensions()));
    if (tensorNumel(cpu) != values.size()) {
        throw std::runtime_error("overwriteGpuTensor value count mismatch.");
    }
    T* ptr = cpu.getMemPtr<T>();
    for (size_t i = 0; i < values.size(); ++i) {
        ptr[i] = values[i];
    }
    gpu.copyFromAsync(cpu, stream);
    stream.synchronize();
}

std::vector<float> copyToCpuValues(const Tensor& gpu, Stream& stream) {
    Tensor cpu = gpu.clone(cpuPlacement);
    cpu.copyFromAsync(gpu, stream);
    stream.synchronize();
    const float* ptr = cpu.getMemPtr<float>();
    return std::vector<float>(ptr, ptr + cpu.getTotalNumElements());
}

void expectNear(const std::vector<float>& actual, const std::vector<float>& expected, float atol = 1.0e-5F) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], atol) << "index " << i;
    }
}

Tensor runExpressionOutput(const Expression& expression,
                           const std::unordered_map<std::string, Tensor>& inputs,
                           const std::string& output_name,
                           Stream& stream,
                           const std::optional<Tensor>& preallocated_output = std::nullopt) {
    FusedEquation equation = FusedEquation::compile(Expression::outputs({{output_name, expression}}).physicalOutputs(), 0);
    std::unordered_map<std::string, Tensor> preallocated;
    if (preallocated_output.has_value()) {
        preallocated.emplace(output_name, preallocated_output.value());
    }
    StampedExecutionPlan plan = equation.stamp(inputs, stream, {}, preallocated);
    plan.run();

    // Keep the stamped plan and its operation-specific temporary storage alive
    // until the enqueued work has completed. In particular, CUB segmented
    // reductions retain their temporary storage through execution. Destroying
    // the plan before synchronization can surface as a cudaFree failure during
    // stack unwinding instead of a useful test failure at the operation site.
    stream.synchronize();
    return plan.output(output_name);
}

}  // namespace

TEST(RaggedExpression, WrapsValuesOffsetsAndBuildsRuntimeExtentAlias) {
    const RaggedTensorDescriptor descriptor = makeDescriptor(DataType::FP32, {4, 2}, 5, 17, DataType::UINT64);
    const Expression values = Expression::input("labels.values");
    const Expression offsets = Expression::input("labels.offsets");

    const RaggedExpression ragged(values, offsets, descriptor);

    EXPECT_TRUE(ragged.isInitialized());
    const MarkedValueNodes marked = markedValueNodes(ragged.getValues());
    EXPECT_EQ(marked.values.op, ExprOp::INPUT);
    EXPECT_EQ(marked.marker.ragged_runtime_batch_size, descriptor.getBatchSize());
    EXPECT_EQ(marked.marker.ragged_runtime_max_active_values, descriptor.getMaxTotalValues());
    EXPECT_EQ(marked.marker.ragged_runtime_elements_per_value, 8ULL);
    EXPECT_TRUE(ragged.getOffsets().isSameLogicalNode(offsets));
    EXPECT_EQ(ragged.getDescriptor(), descriptor);

    const RaggedExpressionRuntimeExtent& extent = ragged.getRuntimeExtent();
    EXPECT_TRUE(extent.isInitialized());
    EXPECT_EQ(extent.maxActiveValues, descriptor.getMaxTotalValues());
    EXPECT_EQ(extent.elementsPerValue, 8);
    EXPECT_EQ(extent.maxLaunchElements(), 17 * 8);

    const ExprNode activeCountNode = outputNode(extent.activeValueCount);
    EXPECT_EQ(activeCountNode.op, ExprOp::STRIDED_VIEW);
    EXPECT_EQ(activeCountNode.view_dims, std::vector<uint64_t>{1});
    EXPECT_EQ(activeCountNode.view_strides, std::vector<uint64_t>{1});
    EXPECT_EQ(activeCountNode.view_element_offset, descriptor.getBatchSize());
}

TEST(RaggedExpression, LogicalInputCreatesValuesAndOffsetsInputs) {
    const RaggedExpression ragged = RaggedExpression::input("labels", makeDescriptor(DataType::FP16, {7}, 2, 11));

    EXPECT_EQ(ragged.getValuesDataType(), DataType::FP16);
    EXPECT_EQ(ragged.getOffsetsDataType(), DataType::UINT32);

    const std::set<std::string> allInputs = ragged.getInputNames();
    EXPECT_TRUE(allInputs.contains("labels.values"));
    EXPECT_TRUE(allInputs.contains("labels.offsets"));

    const std::set<std::string> differentiableInputs = ragged.getDifferentiableInputNames();
    EXPECT_TRUE(differentiableInputs.contains("labels.values"));
    EXPECT_FALSE(differentiableInputs.contains("labels.offsets"));

    const std::set<std::string> metadataInputs = ragged.getMetadataInputNames();
    EXPECT_FALSE(metadataInputs.contains("labels.values"));
    EXPECT_TRUE(metadataInputs.contains("labels.offsets"));
}

TEST(RaggedExpression, UnaryValuewiseOpPreservesOffsetsAndRuntimeExtent) {
    const RaggedExpression ragged = RaggedExpression::input("x", makeDescriptor());

    const RaggedExpression result = ragged.abs();

    EXPECT_TRUE(result.getOffsets().isSameLogicalNode(ragged.getOffsets()));
    EXPECT_TRUE(result.getRuntimeExtent().activeValueCount.isSameLogicalNode(ragged.getRuntimeExtent().activeValueCount));
    EXPECT_EQ(result.getRuntimeExtent().maxActiveValues, ragged.getRuntimeExtent().maxActiveValues);
    EXPECT_EQ(result.getRuntimeExtent().elementsPerValue, ragged.getRuntimeExtent().elementsPerValue);
    EXPECT_EQ(result.getDescriptor(), ragged.getDescriptor());

    EXPECT_EQ(markedValueNodes(result.getValues()).values.op, ExprOp::ABS);
}

TEST(RaggedExpression, CastChangesValuesDTypeButPreservesOffsetsMetadataAndExtent) {
    const RaggedExpression ragged = RaggedExpression::input("x", makeDescriptor(DataType::FP32, {3}, 4, 12, DataType::UINT64));

    const RaggedExpression result = ragged.cast(DataType::FP16);

    EXPECT_EQ(result.getValuesDataType(), DataType::FP16);
    EXPECT_EQ(result.getOffsetsDataType(), DataType::UINT64);
    EXPECT_EQ(result.getValuesDimensions(), ragged.getValuesDimensions());
    EXPECT_TRUE(result.getOffsets().isSameLogicalNode(ragged.getOffsets()));
    EXPECT_TRUE(result.getRuntimeExtent().activeValueCount.isSameLogicalNode(ragged.getRuntimeExtent().activeValueCount));
    EXPECT_EQ(markedValueNodes(result.getValues()).values.op, ExprOp::CAST);
}

TEST(RaggedExpression, BinaryOpWithSameOffsetsSucceeds) {
    const RaggedTensorDescriptor descriptor = makeDescriptor();
    const Expression offsets = Expression::input("shared.offsets");
    const RaggedExpression lhs(Expression::input("lhs.values"), offsets, descriptor);
    const RaggedExpression rhs(Expression::input("rhs.values"), offsets, descriptor);

    const RaggedExpression result = lhs + rhs;

    EXPECT_TRUE(result.getOffsets().isSameLogicalNode(offsets));
    EXPECT_TRUE(result.getRuntimeExtent().activeValueCount.isSameLogicalNode(lhs.getRuntimeExtent().activeValueCount));
    EXPECT_EQ(result.getDescriptor(), lhs.getDescriptor());
    EXPECT_EQ(markedValueNodes(result.getValues()).values.op, ExprOp::ADD);
}

TEST(RaggedExpression, BinaryOpWithDifferentOffsetsRejects) {
    const RaggedTensorDescriptor descriptor = makeDescriptor();
    const RaggedExpression lhs(Expression::input("lhs.values"), Expression::input("lhs.offsets"), descriptor);
    const RaggedExpression rhs(Expression::input("rhs.values"), Expression::input("rhs.offsets"), descriptor);

    EXPECT_THROW((void)(lhs + rhs), std::invalid_argument);
}

TEST(RaggedExpression, BinaryOpWithDifferentValuesDescriptorRejects) {
    const Expression offsets = Expression::input("shared.offsets");
    const RaggedExpression lhs(Expression::input("lhs.values"), offsets, makeDescriptor(DataType::FP32, {4}, 3, 9));
    const RaggedExpression rhs(Expression::input("rhs.values"), offsets, makeDescriptor(DataType::FP32, {5}, 3, 9));

    EXPECT_THROW((void)(lhs + rhs), std::invalid_argument);
}

TEST(RaggedExpression, ComparisonOpProducesBooleanValuesAndPreservesOffsets) {
    const RaggedTensorDescriptor descriptor = makeDescriptor();
    const Expression offsets = Expression::input("shared.offsets");
    const RaggedExpression lhs(Expression::input("lhs.values"), offsets, descriptor);
    const RaggedExpression rhs(Expression::input("rhs.values"), offsets, descriptor);

    const RaggedExpression result = lhs.lessThan(rhs);

    EXPECT_EQ(result.getValuesDataType(), DataType::BOOLEAN);
    EXPECT_TRUE(result.getOffsets().isSameLogicalNode(offsets));
    EXPECT_EQ(markedValueNodes(result.getValues()).values.op, ExprOp::LESS);
}

TEST(RaggedExpression, NonScalarConvenienceSegmentOpsRejectCleanly) {
    const RaggedExpression ragged = RaggedExpression::input("x", makeDescriptor());

    EXPECT_THROW((void)ragged.softmax(), std::invalid_argument);
    EXPECT_THROW((void)ragged.reduce_sum(), std::invalid_argument);
}


TEST(RaggedExpression, SegmentReductionsBuildDensePerRowOutputsForScalarValues) {
    const RaggedExpression ragged = RaggedExpression::input("x", makeDescriptor(DataType::FP32, {}, 4, 12));

    const Expression sum = ragged.segment_sum();
    const Expression min = ragged.segment_min();
    const Expression max = ragged.segment_max();

    EXPECT_EQ(outputNode(sum).op, ExprOp::SEGMENTED_REDUCE_SUM);
    EXPECT_EQ(outputNode(min).op, ExprOp::SEGMENTED_REDUCE_MIN);
    EXPECT_EQ(outputNode(max).op, ExprOp::SEGMENTED_REDUCE_MAX);
}

TEST(RaggedExpression, SegmentReductionsRejectNonScalarRaggedValuesCleanly) {
    const RaggedExpression ragged = RaggedExpression::input("x", makeDescriptor(DataType::FP32, {4}, 3, 9));

    EXPECT_THROW((void)ragged.segment_sum(), std::invalid_argument);
    EXPECT_THROW((void)ragged.segment_min(), std::invalid_argument);
    EXPECT_THROW((void)ragged.segment_max(), std::invalid_argument);
    EXPECT_THROW((void)ragged.segment_softmax(), std::invalid_argument);
}

TEST(RaggedExpression, SegmentMeanBuildsMaskedDensePerRowAverageForScalarValues) {
    const RaggedExpression ragged = RaggedExpression::input("x", makeDescriptor(DataType::FP32, {}, 3, 9));

    const Expression mean = ragged.segment_mean();

    EXPECT_EQ(outputNode(mean).op, ExprOp::WHERE);
}

TEST(RaggedExpression, SegmentSoftmaxPreservesOffsetsAndRuntimeExtentForScalarValues) {
    const RaggedExpression ragged = RaggedExpression::input("x", makeDescriptor(DataType::FP32, {}, 3, 9));

    const RaggedExpression result = ragged.segment_softmax();

    EXPECT_TRUE(result.getOffsets().isSameLogicalNode(ragged.getOffsets()));
    EXPECT_TRUE(result.getRuntimeExtent().activeValueCount.isSameLogicalNode(ragged.getRuntimeExtent().activeValueCount));
    EXPECT_EQ(result.getDescriptor(), ragged.getDescriptor());
    EXPECT_EQ(markedValueNodes(result.getValues()).values.op, ExprOp::DIV);
}

TEST(RaggedExpression, SegmentLogSoftmaxPreservesOffsetsAndRuntimeExtentForScalarValues) {
    const RaggedExpression ragged = RaggedExpression::input("x", makeDescriptor(DataType::FP32, {}, 3, 9));

    const RaggedExpression result = ragged.segment_log_softmax();

    EXPECT_TRUE(result.getOffsets().isSameLogicalNode(ragged.getOffsets()));
    EXPECT_TRUE(result.getRuntimeExtent().activeValueCount.isSameLogicalNode(ragged.getRuntimeExtent().activeValueCount));
    EXPECT_EQ(result.getDescriptor(), ragged.getDescriptor());
    EXPECT_EQ(markedValueNodes(result.getValues()).values.op, ExprOp::SUB);
}

TEST(RaggedExpression, SegmentedScanCanCarryReverseFlagForRowLocalBroadcasts) {
    const RaggedExpression ragged = RaggedExpression::input("x", makeDescriptor(DataType::FP32, {}, 3, 9));

    const Expression reverseScan =
        Expression::input("x.values", DataType::FP32).segmentedScan(ragged.getOffsets(), ScanOp::Sum, true, true);

    const ExprNode scanNode = outputNode(reverseScan);
    EXPECT_EQ(scanNode.op, ExprOp::SEGMENTED_SCAN);
    EXPECT_TRUE(scanNode.scan_reverse);
}


TEST(RaggedExpression, ValuewiseAutodiffRejectsUntilFirstClassRaggedGradientsAreImplemented) {
    const RaggedExpression ragged = RaggedExpression::input("x", makeDescriptor(DataType::FP32, {}, 3, 9));
    const PhysicalOutputs outputs = Expression::outputs({{"y", ragged.relu().getValues()}}).physicalOutputs();

    EXPECT_THROW((void)buildBackwardOutputs(outputs, {"x.values"}), std::runtime_error);
    EXPECT_THROW((void)buildBackwardOutputs(outputs, {"x.offsets"}), std::runtime_error);
}

TEST(RaggedExpression, ValuewiseExecutionReadsActiveExtentOnDeviceAndReusesOneStampedPlan) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    const RaggedTensorDescriptor descriptor = makeDescriptor(DataType::FP32, {2}, 3, 6);
    const RaggedExpression ragged = RaggedExpression::input("x", descriptor);
    const Expression output_expression = ragged.relu().getValues();

    Tensor values = makeGpuTensor<float>({6, 2},
                                         {-1.0F, 2.0F,
                                          -3.0F, 4.0F,
                                          5.0F, -6.0F,
                                          7.0F, -8.0F,
                                          9.0F, -10.0F,
                                          11.0F, -12.0F},
                                         stream);
    Tensor offsets = makeGpuTensor<uint32_t>({4}, {0U, 2U, 2U, 3U}, stream);
    Tensor output(gpuPlacement, TensorDescriptor(DataType::FP32, {6, 2}));
    output.fill(777.0, stream);

    FusedEquation equation = FusedEquation::compile(Expression::outputs({{"y", output_expression}}).physicalOutputs(), 0);
    StampedExecutionPlan plan = equation.stamp({{"x.values", values}, {"x.offsets", offsets}}, stream, {}, {{"y", output}});
    plan.run();
    expectNear(copyToCpuValues(output, stream),
               {0.0F, 2.0F, 0.0F, 4.0F, 5.0F, 0.0F,
                777.0F, 777.0F, 777.0F, 777.0F, 777.0F, 777.0F});

    // Change only offsets[B] and replay the already-stamped plan. This exercises
    // device-side logical extent without shape recompilation or host readback.
    overwriteGpuTensor<uint32_t>(offsets, {0U, 1U, 3U, 5U}, stream);
    output.fill(888.0, stream);
    plan.run();
    expectNear(copyToCpuValues(output, stream),
               {0.0F, 2.0F, 0.0F, 4.0F, 5.0F, 0.0F, 7.0F, 0.0F, 9.0F, 0.0F,
                888.0F, 888.0F});
}

TEST(RaggedExpression, SegmentSumMinMaxAndMeanExecuteForEmptyAndSkewedRows) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    const RaggedExpression ragged = RaggedExpression::input("x", makeDescriptor(DataType::FP32, {}, 4, 9));
    Tensor values = makeGpuTensor<float>({9}, {1.0F, -2.0F, 4.0F, 5.0F, 7.0F, -1.0F, 8.0F, 99.0F, 100.0F}, stream);
    Tensor offsets = makeGpuTensor<uint32_t>({5}, {0U, 3U, 3U, 5U, 7U}, stream);
    const std::unordered_map<std::string, Tensor> inputs{{"x.values", values}, {"x.offsets", offsets}};

    const Tensor sum = runExpressionOutput(ragged.segment_sum(), inputs, "sum", stream);
    const Tensor min = runExpressionOutput(ragged.segment_min(), inputs, "min", stream);
    const Tensor max = runExpressionOutput(ragged.segment_max(), inputs, "max", stream);
    const Tensor mean = runExpressionOutput(ragged.segment_mean(), inputs, "mean", stream);

    expectNear(copyToCpuValues(sum, stream), {3.0F, 0.0F, 12.0F, 7.0F});

    const std::vector<float> min_values = copyToCpuValues(min, stream);
    ASSERT_EQ(min_values.size(), 4U);
    EXPECT_FLOAT_EQ(min_values[0], -2.0F);
    EXPECT_FLOAT_EQ(min_values[2], 5.0F);
    EXPECT_FLOAT_EQ(min_values[3], -1.0F);

    const std::vector<float> max_values = copyToCpuValues(max, stream);
    ASSERT_EQ(max_values.size(), 4U);
    EXPECT_FLOAT_EQ(max_values[0], 4.0F);
    EXPECT_FLOAT_EQ(max_values[2], 7.0F);
    EXPECT_FLOAT_EQ(max_values[3], 8.0F);

    expectNear(copyToCpuValues(mean, stream), {1.0F, 0.0F, 6.0F, 3.5F});
}

TEST(RaggedExpression, SegmentSoftmaxExecutesPerRowAndLeavesUnusedCapacityUntouched) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    const RaggedExpression ragged = RaggedExpression::input("x", makeDescriptor(DataType::FP32, {}, 3, 8));
    Tensor values = makeGpuTensor<float>({8}, {1.0F, 2.0F, 0.0F, -1.0F, 1.0F, 50.0F, 60.0F, 70.0F}, stream);
    Tensor offsets = makeGpuTensor<uint32_t>({4}, {0U, 2U, 2U, 5U}, stream);
    Tensor output(gpuPlacement, TensorDescriptor(DataType::FP32, {8}));
    output.fill(321.0, stream);

    const Tensor actual = runExpressionOutput(ragged.segment_softmax().getValues(),
                                              {{"x.values", values}, {"x.offsets", offsets}},
                                              "softmax",
                                              stream,
                                              output);

    const float row0_denom = std::exp(1.0F) + std::exp(2.0F);
    const float row2_denom = std::exp(0.0F) + std::exp(-1.0F) + std::exp(1.0F);
    expectNear(copyToCpuValues(actual, stream),
               {std::exp(1.0F) / row0_denom,
                std::exp(2.0F) / row0_denom,
                std::exp(0.0F) / row2_denom,
                std::exp(-1.0F) / row2_denom,
                std::exp(1.0F) / row2_denom,
                321.0F,
                321.0F,
                321.0F},
               2.0e-5F);
}

TEST(RaggedExpression, SegmentLogSoftmaxExecutesPerRowAndLeavesUnusedCapacityUntouched) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    const RaggedExpression ragged = RaggedExpression::input("x", makeDescriptor(DataType::FP32, {}, 3, 8));
    Tensor values = makeGpuTensor<float>({8}, {1.0F, 2.0F, 0.0F, -1.0F, 1.0F, 50.0F, 60.0F, 70.0F}, stream);
    Tensor offsets = makeGpuTensor<uint32_t>({4}, {0U, 2U, 2U, 5U}, stream);
    Tensor output(gpuPlacement, TensorDescriptor(DataType::FP32, {8}));
    output.fill(-456.0, stream);

    const Tensor actual = runExpressionOutput(ragged.segment_log_softmax().getValues(),
                                              {{"x.values", values}, {"x.offsets", offsets}},
                                              "log_softmax",
                                              stream,
                                              output);

    const float row0_log_denom = std::log(std::exp(1.0F) + std::exp(2.0F));
    const float row2_log_denom = std::log(std::exp(0.0F) + std::exp(-1.0F) + std::exp(1.0F));
    expectNear(copyToCpuValues(actual, stream),
               {1.0F - row0_log_denom,
                2.0F - row0_log_denom,
                0.0F - row2_log_denom,
                -1.0F - row2_log_denom,
                1.0F - row2_log_denom,
                -456.0F,
                -456.0F,
                -456.0F},
               2.0e-5F);
}
