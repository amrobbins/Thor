#include "Utilities/Expression/CudaSourceEmitter.h"
#include "Utilities/Expression/EquationCompiler.h"
#include "Utilities/Expression/ExpressionDTypeResolution.h"
#include "Utilities/Expression/RaggedExpression.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace ThorImplementation;

namespace {

const PhysicalExecutionStage& onlyFusedStage(const std::vector<PhysicalExecutionStage>& stages) {
    if (stages.size() != 1u || stages.front().kind != PhysicalExecutionStage::Kind::FusedKernel) {
        throw std::runtime_error("FusedFastPathGateTest expected exactly one fused stage.");
    }
    return stages.front();
}

CompiledExecutionStage makeCompiledStage(const PhysicalOutputs& outputs) {
    if (outputs.outputs.size() != 1u) {
        throw std::runtime_error("FusedFastPathGateTest expected exactly one output.");
    }
    std::vector<uint32_t> input_value_ids;
    input_value_ids.reserve(outputs.expr->inputs.size());
    for (uint32_t i = 0; i < outputs.expr->inputs.size(); ++i) {
        input_value_ids.push_back(i);
    }
    const NamedOutput& output = outputs.outputs.front();
    return CompiledExecutionStage(
        *outputs.expr,
        std::shared_ptr<CompiledEquation>{},
        std::move(input_value_ids),
        {CompiledStageOutput{
            .name = output.name,
            .local_node_idx = output.node_idx,
            .value_id = static_cast<uint32_t>(outputs.expr->inputs.size()),
        }});
}

SpecializedBroadcastGroup makeTwoInputBroadcastGroup(uint64_t rows,
                                                     uint64_t width,
                                                     uint64_t rhs_width,
                                                     SpecializedInputLoadKind rhs_load_kind) {
    SpecializedBroadcastGroup group;
    group.numel = rows * width;
    group.output_dims = {rows, width};
    group.output_indices = {0u};
    group.used_input_slots = {0u, 1u};
    group.used_input_broadcast_offset_required = {true, true};
    group.used_input_visible_dims = {{rows, width}, {1u, rhs_width}};
    group.used_input_visible_strides = {{width, 1u}, {rhs_width, rhs_width == 1u ? 0u : 1u}};
    group.used_input_load_kinds = {SpecializedInputLoadKind::NativeVector, rhs_load_kind};
    group.active_axes = {
        SpecializedBroadcastAxis{.dim = rows, .output_stride = width, .input_strides = {width, 0u}},
        SpecializedBroadcastAxis{
            .dim = width,
            .output_stride = 1u,
            .input_strides = {1u, rhs_width == 1u ? 0u : 1u},
        },
    };
    return group;
}

}  // namespace

TEST(FusedFastPathGate, DenseFlatOrdainedWidthsAreIndependentOfDispatchSelection) {
    {
        const Expression x = Expression::input("x", DataType::BF16, DataType::BF16);
        const Expression y = Expression::input("y", DataType::BF16, DataType::BF16);
        auto physical = Expression::outputs({{"out", x + y}}).physicalOutputs();
        resolveOutputsDTypesInPlace(physical, {DataType::BF16, DataType::BF16});
        auto stages = EquationCompiler::splitAtReductionBoundaries(physical);
        const PhysicalExecutionStage& stage = onlyFusedStage(stages);
        ASSERT_EQ(CudaSourceEmitter::fusedGateRequiredFlatElementsPerThread(stage), std::optional<uint32_t>(8u));
        EXPECT_EQ(CudaSourceEmitter::flatElementsPerThread(stage), 8u);
    }
    {
        const Expression x = Expression::input("x", DataType::FP32, DataType::FP32);
        const Expression y = Expression::input("y", DataType::FP32, DataType::FP32);
        auto physical = Expression::outputs({{"out", x + y}}).physicalOutputs();
        resolveOutputsDTypesInPlace(physical, {DataType::FP32, DataType::FP32});
        auto stages = EquationCompiler::splitAtReductionBoundaries(physical);
        const PhysicalExecutionStage& stage = onlyFusedStage(stages);
        ASSERT_EQ(CudaSourceEmitter::fusedGateRequiredFlatElementsPerThread(stage), std::optional<uint32_t>(4u));
        EXPECT_EQ(CudaSourceEmitter::flatElementsPerThread(stage), 4u);
    }
    {
        const Expression x = Expression::input("x", DataType::BF16, DataType::BF16);
        const Expression y = Expression::input("y", DataType::FP32, DataType::FP32);
        const Expression out = (x + y).withOutputDType(DataType::FP32);
        auto physical = Expression::outputs({{"out", out}}).physicalOutputs();
        resolveOutputsDTypesInPlace(physical, {DataType::BF16, DataType::FP32});
        auto stages = EquationCompiler::splitAtReductionBoundaries(physical);
        const PhysicalExecutionStage& stage = onlyFusedStage(stages);
        ASSERT_EQ(CudaSourceEmitter::fusedGateRequiredFlatElementsPerThread(stage), std::optional<uint32_t>(4u));
        EXPECT_EQ(CudaSourceEmitter::flatElementsPerThread(stage), 4u);
    }
}

TEST(FusedFastPathGate, RaggedFlatOrdainedWidthsCoverLowPrecisionFp32AndMixed) {
    {
        const RaggedTensorDescriptor descriptor(DataType::BF16, {}, 4, 12, DataType::UINT32);
        const RaggedExpression ragged = RaggedExpression::input("x", descriptor);
        auto physical = Expression::outputs({{"out", ragged.relu().getValues()}}).physicalOutputs();
        resolveOutputsDTypesInPlace(physical, {DataType::BF16, DataType::UINT32});
        auto stages = EquationCompiler::splitAtReductionBoundaries(physical);
        const PhysicalExecutionStage& stage = onlyFusedStage(stages);
        ASSERT_EQ(CudaSourceEmitter::fusedGateRequiredFlatElementsPerThread(stage), std::optional<uint32_t>(8u));
        EXPECT_EQ(CudaSourceEmitter::flatElementsPerThread(stage), 8u);
    }
    {
        const RaggedTensorDescriptor descriptor(DataType::FP32, {}, 4, 12, DataType::UINT32);
        const RaggedExpression ragged = RaggedExpression::input("x", descriptor);
        auto physical = Expression::outputs({{"out", ragged.relu().getValues()}}).physicalOutputs();
        resolveOutputsDTypesInPlace(physical, {DataType::FP32, DataType::UINT32});
        auto stages = EquationCompiler::splitAtReductionBoundaries(physical);
        const PhysicalExecutionStage& stage = onlyFusedStage(stages);
        ASSERT_EQ(CudaSourceEmitter::fusedGateRequiredFlatElementsPerThread(stage), std::optional<uint32_t>(4u));
        EXPECT_EQ(CudaSourceEmitter::flatElementsPerThread(stage), 4u);
    }
    {
        const Expression x = Expression::input("x", DataType::BF16, DataType::BF16);
        const Expression y = Expression::input("y", DataType::FP32, DataType::FP32);
        const Expression active_count = Expression::input("active_count", DataType::UINT32, DataType::UINT32);
        const Expression out = (x + y).withOutputDType(DataType::FP32).withRaggedRuntimeExtent(
            active_count, 4, 12, 8, RaggedRuntimeExtentSource::DEVICE_ACTIVE_COUNT);
        auto physical = Expression::outputs({{"out", out}}).physicalOutputs();
        resolveOutputsDTypesInPlace(physical, {DataType::BF16, DataType::FP32, DataType::UINT32});
        auto stages = EquationCompiler::splitAtReductionBoundaries(physical);
        const PhysicalExecutionStage& stage = onlyFusedStage(stages);
        ASSERT_EQ(CudaSourceEmitter::fusedGateRequiredFlatElementsPerThread(stage), std::optional<uint32_t>(4u));
        EXPECT_EQ(CudaSourceEmitter::flatElementsPerThread(stage), 4u);
    }
}

TEST(FusedFastPathGate, RaggedFlatLowPrecisionStorageKeepsWidePacketsWithFp32Compute) {
    for (DataType storage_dtype : {DataType::BF16, DataType::FP16}) {
        const Expression x = Expression::input("x", DataType::FP32, storage_dtype);
        const Expression active_count = Expression::input("active_count", DataType::UINT32, DataType::UINT32);
        const Expression out = (x * x)
                                   .withComputeDType(DataType::FP32)
                                   .withOutputDType(storage_dtype)
                                   .withRaggedRuntimeExtent(
                                       active_count, 4, 12, 5, RaggedRuntimeExtentSource::DEVICE_ACTIVE_COUNT);
        auto physical = Expression::outputs({{"out", out}}).physicalOutputs();
        resolveOutputsDTypesInPlace(physical, {storage_dtype, DataType::UINT32});
        auto stages = EquationCompiler::splitAtReductionBoundaries(physical);
        const PhysicalExecutionStage& stage = onlyFusedStage(stages);
        ASSERT_EQ(CudaSourceEmitter::fusedGateRequiredFlatElementsPerThread(stage), std::optional<uint32_t>(8u));
        EXPECT_EQ(CudaSourceEmitter::flatElementsPerThread(stage), 8u);
    }
}

TEST(FusedFastPathGate, RaggedPhysicalLowPrecisionInputPromotedToFp32UsesFourScalarOwnership) {
    // CustomLayer-style promotion: the expression sees FP32 values, while the
    // connected producer still stores BF16.  Physical storage must continue to
    // drive packet ownership.
    const Expression x = Expression::input("x", DataType::FP32, DataType::FP32);
    const Expression active_count = Expression::input("active_count", DataType::UINT32, DataType::UINT32);
    const Expression out = (x * x)
                               .withComputeDType(DataType::FP32)
                               .withOutputDType(DataType::FP32)
                               .withRaggedRuntimeExtent(
                                   active_count, 4, 12, 5, RaggedRuntimeExtentSource::DEVICE_ACTIVE_COUNT);
    auto physical = Expression::outputs({{"out", out}}).physicalOutputs();
    resolveOutputsDTypesInPlace(physical, {DataType::BF16, DataType::UINT32});
    auto stages = EquationCompiler::splitAtReductionBoundaries(physical);
    const PhysicalExecutionStage& stage = onlyFusedStage(stages);
    ASSERT_EQ(CudaSourceEmitter::fusedGateRequiredFlatElementsPerThread(stage), std::optional<uint32_t>(4u));
    EXPECT_EQ(CudaSourceEmitter::flatElementsPerThread(stage), 4u);
}

TEST(FusedFastPathGate, RaggedFlatFp32StorageBoundaryUsesFourScalarOwnershipWithFp32Compute) {
    {
        const Expression x = Expression::input("x", DataType::FP32, DataType::BF16);
        const Expression active_count = Expression::input("active_count", DataType::UINT32, DataType::UINT32);
        const Expression out = (x * x)
                                   .withComputeDType(DataType::FP32)
                                   .withOutputDType(DataType::FP32)
                                   .withRaggedRuntimeExtent(
                                       active_count, 4, 12, 5, RaggedRuntimeExtentSource::DEVICE_ACTIVE_COUNT);
        auto physical = Expression::outputs({{"out", out}}).physicalOutputs();
        resolveOutputsDTypesInPlace(physical, {DataType::BF16, DataType::UINT32});
        auto stages = EquationCompiler::splitAtReductionBoundaries(physical);
        const PhysicalExecutionStage& stage = onlyFusedStage(stages);
        ASSERT_EQ(CudaSourceEmitter::fusedGateRequiredFlatElementsPerThread(stage), std::optional<uint32_t>(4u));
        EXPECT_EQ(CudaSourceEmitter::flatElementsPerThread(stage), 4u);
    }
    {
        const Expression x = Expression::input("x", DataType::FP32, DataType::FP32);
        const Expression active_count = Expression::input("active_count", DataType::UINT32, DataType::UINT32);
        const Expression out = (x * x)
                                   .withComputeDType(DataType::FP32)
                                   .withOutputDType(DataType::BF16)
                                   .withRaggedRuntimeExtent(
                                       active_count, 4, 12, 5, RaggedRuntimeExtentSource::DEVICE_ACTIVE_COUNT);
        auto physical = Expression::outputs({{"out", out}}).physicalOutputs();
        resolveOutputsDTypesInPlace(physical, {DataType::FP32, DataType::UINT32});
        auto stages = EquationCompiler::splitAtReductionBoundaries(physical);
        const PhysicalExecutionStage& stage = onlyFusedStage(stages);
        ASSERT_EQ(CudaSourceEmitter::fusedGateRequiredFlatElementsPerThread(stage), std::optional<uint32_t>(4u));
        EXPECT_EQ(CudaSourceEmitter::flatElementsPerThread(stage), 4u);
    }
}

TEST(FusedFastPathGate, RaggedBroadcastOrdainedWidthsMatchProvenPacketFamilies) {
    {
        const Expression x = Expression::input("x", DataType::BF16, DataType::BF16);
        const Expression bias = Expression::input("bias", DataType::BF16, DataType::BF16);
        const Expression active_count = Expression::input("active_count", DataType::UINT32, DataType::UINT32);
        const Expression out = (x + bias).withRaggedRuntimeExtent(
            active_count, 2, 6, 8, RaggedRuntimeExtentSource::DEVICE_ACTIVE_COUNT);
        auto physical = Expression::outputs({{"out", out}}).physicalOutputs();
        resolveOutputsDTypesInPlace(physical, {DataType::BF16, DataType::BF16, DataType::UINT32});
        const CompiledExecutionStage stage = makeCompiledStage(physical);
        const SpecializedBroadcastGroup group =
            makeTwoInputBroadcastGroup(6u, 8u, 8u, SpecializedInputLoadKind::NativeVector);
        ASSERT_EQ(CudaSourceEmitter::fusedGateRequiredSpecializedBroadcastElementsPerThread(stage, {group}),
                  std::optional<uint32_t>(8u));
        EXPECT_EQ(CudaSourceEmitter::specializedBroadcastElementsPerThread(stage, {group}), 8u);
    }
    {
        const Expression x = Expression::input("x", DataType::FP32, DataType::FP32);
        const Expression bias = Expression::input("bias", DataType::FP32, DataType::FP32);
        const Expression active_count = Expression::input("active_count", DataType::UINT32, DataType::UINT32);
        const Expression out = (x + bias).withRaggedRuntimeExtent(
            active_count, 2, 6, 8, RaggedRuntimeExtentSource::DEVICE_ACTIVE_COUNT);
        auto physical = Expression::outputs({{"out", out}}).physicalOutputs();
        resolveOutputsDTypesInPlace(physical, {DataType::FP32, DataType::FP32, DataType::UINT32});
        const CompiledExecutionStage stage = makeCompiledStage(physical);
        const SpecializedBroadcastGroup group =
            makeTwoInputBroadcastGroup(6u, 8u, 8u, SpecializedInputLoadKind::NativeVector);
        ASSERT_EQ(CudaSourceEmitter::fusedGateRequiredSpecializedBroadcastElementsPerThread(stage, {group}),
                  std::optional<uint32_t>(4u));
        EXPECT_EQ(CudaSourceEmitter::specializedBroadcastElementsPerThread(stage, {group}), 4u);
    }
    {
        const Expression x = Expression::input("x", DataType::BF16, DataType::BF16);
        const Expression bias = Expression::input("bias", DataType::FP32, DataType::FP32);
        const Expression active_count = Expression::input("active_count", DataType::UINT32, DataType::UINT32);
        const Expression out = (x + bias).withOutputDType(DataType::FP32).withRaggedRuntimeExtent(
            active_count, 2, 6, 8, RaggedRuntimeExtentSource::DEVICE_ACTIVE_COUNT);
        auto physical = Expression::outputs({{"out", out}}).physicalOutputs();
        resolveOutputsDTypesInPlace(physical, {DataType::BF16, DataType::FP32, DataType::UINT32});
        const CompiledExecutionStage stage = makeCompiledStage(physical);
        const SpecializedBroadcastGroup group =
            makeTwoInputBroadcastGroup(6u, 8u, 8u, SpecializedInputLoadKind::NativeVector);
        ASSERT_EQ(CudaSourceEmitter::fusedGateRequiredSpecializedBroadcastElementsPerThread(stage, {group}),
                  std::optional<uint32_t>(4u));
        EXPECT_EQ(CudaSourceEmitter::specializedBroadcastElementsPerThread(stage, {group}), 4u);
    }
}

TEST(FusedFastPathGate, RaggedBroadcastLowPrecisionStorageKeepsVector8OwnershipWithFp32Compute) {
    for (DataType storage_dtype : {DataType::BF16, DataType::FP16}) {
        const Expression x = Expression::input("x", DataType::FP32, storage_dtype);
        const Expression bias = Expression::input("bias", DataType::FP32, storage_dtype);
        const Expression active_count = Expression::input("active_count", DataType::UINT32, DataType::UINT32);
        const Expression out = (x + bias)
                                   .withComputeDType(DataType::FP32)
                                   .withOutputDType(storage_dtype)
                                   .withRaggedRuntimeExtent(
                                       active_count, 2, 6, 8, RaggedRuntimeExtentSource::DEVICE_ACTIVE_COUNT);
        auto physical = Expression::outputs({{"out", out}}).physicalOutputs();
        resolveOutputsDTypesInPlace(physical, {storage_dtype, storage_dtype, DataType::UINT32});
        const CompiledExecutionStage stage = makeCompiledStage(physical);
        const SpecializedBroadcastGroup group =
            makeTwoInputBroadcastGroup(6u, 8u, 8u, SpecializedInputLoadKind::NativeVector);
        ASSERT_EQ(CudaSourceEmitter::fusedGateRequiredSpecializedBroadcastElementsPerThread(stage, {group}),
                  std::optional<uint32_t>(8u));
        EXPECT_EQ(CudaSourceEmitter::specializedBroadcastElementsPerThread(stage, {group}), 8u);
        const std::string source = CudaSourceEmitter::emitSpecializedBroadcast(stage, {group}, "ragged_lowp_fp32_broadcast");
        EXPECT_NE(source.find("const float4 in0_packet = *reinterpret_cast<const float4*>(in0 + in0_offset_base)"),
                  std::string::npos);
        EXPECT_NE(source.find("float4 out0_packet"), std::string::npos);
        EXPECT_NE(source.find("const float t"), std::string::npos);
    }
}

TEST(FusedFastPathGate, AwkwardBroadcastAndTrueGatherRemainIntentionalExceptions) {
    {
        const Expression x = Expression::input("x", DataType::BF16, DataType::BF16);
        const Expression bias = Expression::input("bias", DataType::BF16, DataType::BF16);
        const Expression active_count = Expression::input("active_count", DataType::UINT32, DataType::UINT32);
        const Expression out = (x + bias).withRaggedRuntimeExtent(
            active_count, 2, 6, 7, RaggedRuntimeExtentSource::DEVICE_ACTIVE_COUNT);
        auto physical = Expression::outputs({{"out", out}}).physicalOutputs();
        resolveOutputsDTypesInPlace(physical, {DataType::BF16, DataType::BF16, DataType::UINT32});
        const CompiledExecutionStage stage = makeCompiledStage(physical);
        const SpecializedBroadcastGroup group =
            makeTwoInputBroadcastGroup(6u, 7u, 7u, SpecializedInputLoadKind::NativeVector);
        EXPECT_FALSE(CudaSourceEmitter::fusedGateRequiredSpecializedBroadcastElementsPerThread(stage, {group}).has_value());
        EXPECT_EQ(CudaSourceEmitter::specializedBroadcastElementsPerThread(stage, {group}), 1u);
    }
    {
        const Expression values = Expression::input("values", DataType::BF16, DataType::BF16);
        const Expression indices = Expression::input("indices", DataType::UINT32, DataType::UINT32);
        auto physical = Expression::outputs({{"out", values.takeAlongAxis(indices, 1)}}).physicalOutputs();
        resolveOutputsDTypesInPlace(physical, {DataType::BF16, DataType::UINT32});
        auto stages = EquationCompiler::splitAtReductionBoundaries(physical);
        const PhysicalExecutionStage& stage = onlyFusedStage(stages);
        EXPECT_FALSE(CudaSourceEmitter::fusedGateRequiredFlatElementsPerThread(stage).has_value());
    }
}

TEST(FusedFastPathGate, DenseLowPrecisionBroadcastPreservesCurrentVector2Ordination) {
    const Expression x = Expression::input("x", DataType::BF16, DataType::BF16);
    const Expression bias = Expression::input("bias", DataType::BF16, DataType::BF16);
    auto physical = Expression::outputs({{"out", x + bias}}).physicalOutputs();
    resolveOutputsDTypesInPlace(physical, {DataType::BF16, DataType::BF16});
    const CompiledExecutionStage stage = makeCompiledStage(physical);
    const SpecializedBroadcastGroup group =
        makeTwoInputBroadcastGroup(6u, 8u, 8u, SpecializedInputLoadKind::NativeVector);

    ASSERT_EQ(CudaSourceEmitter::fusedGateRequiredSpecializedBroadcastElementsPerThread(stage, {group}),
              std::optional<uint32_t>(2u));
    EXPECT_EQ(CudaSourceEmitter::specializedBroadcastElementsPerThread(stage, {group}), 2u);
}

TEST(FusedFastPathGate, RaggedTerminalSliceAliasCannotRegressBackIntoScalarIndexedExecution) {
    const RaggedTensorDescriptor descriptor(DataType::BF16, {256}, 128, 104832, DataType::UINT32);
    const RaggedExpression ragged = RaggedExpression::input("x", descriptor);
    const RaggedExpression half = ragged.sliceLastDimension(128, 128);
    auto physical = Expression::outputs({{"half", half.getValues()}}).physicalOutputs();
    resolveOutputsDTypesInPlace(physical, {DataType::BF16, DataType::UINT32});

    auto stages = EquationCompiler::splitAtReductionBoundaries(physical);
    const PhysicalExecutionStage& stage = onlyFusedStage(stages);
    for (const ExprNode& node : stage.expr.nodes) {
        EXPECT_NE(node.op, ExprOp::STRIDED_VIEW);
    }
    ASSERT_EQ(CudaSourceEmitter::fusedGateRequiredFlatElementsPerThread(stage), std::optional<uint32_t>(8u));
    EXPECT_EQ(CudaSourceEmitter::flatElementsPerThread(stage), 8u);
}
