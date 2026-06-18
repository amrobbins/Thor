#include "Utilities/Expression/RaggedExpression.h"

#include "Utilities/Expression/Expression.h"

#include "gtest/gtest.h"

#include <set>
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

}  // namespace

TEST(RaggedExpression, WrapsValuesOffsetsAndBuildsRuntimeExtentAlias) {
    const RaggedTensorDescriptor descriptor = makeDescriptor(DataType::FP32, {4, 2}, 5, 17, DataType::UINT64);
    const Expression values = Expression::input("labels.values");
    const Expression offsets = Expression::input("labels.offsets");

    const RaggedExpression ragged(values, offsets, descriptor);

    EXPECT_TRUE(ragged.isInitialized());
    EXPECT_TRUE(ragged.getValues().isSameLogicalNode(values));
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

    EXPECT_EQ(outputNode(result.getValues()).op, ExprOp::ABS);
}

TEST(RaggedExpression, CastChangesValuesDTypeButPreservesOffsetsMetadataAndExtent) {
    const RaggedExpression ragged = RaggedExpression::input("x", makeDescriptor(DataType::FP32, {3}, 4, 12, DataType::UINT64));

    const RaggedExpression result = ragged.cast(DataType::FP16);

    EXPECT_EQ(result.getValuesDataType(), DataType::FP16);
    EXPECT_EQ(result.getOffsetsDataType(), DataType::UINT64);
    EXPECT_EQ(result.getValuesDimensions(), ragged.getValuesDimensions());
    EXPECT_TRUE(result.getOffsets().isSameLogicalNode(ragged.getOffsets()));
    EXPECT_TRUE(result.getRuntimeExtent().activeValueCount.isSameLogicalNode(ragged.getRuntimeExtent().activeValueCount));
    EXPECT_EQ(outputNode(result.getValues()).op, ExprOp::CAST);
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
    EXPECT_EQ(outputNode(result.getValues()).op, ExprOp::ADD);
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
    EXPECT_EQ(outputNode(result.getValues()).op, ExprOp::LESS);
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
    EXPECT_EQ(outputNode(result.getValues()).op, ExprOp::DIV);
}

TEST(RaggedExpression, SegmentLogSoftmaxPreservesOffsetsAndRuntimeExtentForScalarValues) {
    const RaggedExpression ragged = RaggedExpression::input("x", makeDescriptor(DataType::FP32, {}, 3, 9));

    const RaggedExpression result = ragged.segment_log_softmax();

    EXPECT_TRUE(result.getOffsets().isSameLogicalNode(ragged.getOffsets()));
    EXPECT_TRUE(result.getRuntimeExtent().activeValueCount.isSameLogicalNode(ragged.getRuntimeExtent().activeValueCount));
    EXPECT_EQ(result.getDescriptor(), ragged.getDescriptor());
    EXPECT_EQ(outputNode(result.getValues()).op, ExprOp::SUB);
}

TEST(RaggedExpression, SegmentedScanCanCarryReverseFlagForRowLocalBroadcasts) {
    const RaggedExpression ragged = RaggedExpression::input("x", makeDescriptor(DataType::FP32, {}, 3, 9));

    const Expression reverseScan = ragged.getValues().segmentedScan(ragged.getOffsets(), ScanOp::Sum, true, true);

    const ExprNode scanNode = outputNode(reverseScan);
    EXPECT_EQ(scanNode.op, ExprOp::SEGMENTED_SCAN);
    EXPECT_TRUE(scanNode.scan_reverse);
}
