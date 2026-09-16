#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/FusedEquation.h"
#include "Utilities/Expression/RaggedExpression.h"
#include "Utilities/TensorOperations/Ragged/PaddedRaggedSequence.h"

#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ThorImplementation;

namespace {

#define REQUIRE_CUDA_DEVICE()                                                                                          \
    do {                                                                                                                \
        int cuda_device_count_for_test = 0;                                                                             \
        const cudaError_t cuda_status_for_test = cudaGetDeviceCount(&cuda_device_count_for_test);                       \
        if (cuda_status_for_test != cudaSuccess || cuda_device_count_for_test <= 0) {                                    \
            GTEST_SKIP() << "CUDA device is required for ragged-capacity performance verification tests.";             \
        }                                                                                                               \
    } while (false)

TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

uint32_t countStages(const std::shared_ptr<CompiledOutputs>& compiled, CompiledExecutionStage::Kind kind) {
    uint32_t count = 0;
    for (const CompiledExecutionStage& stage : compiled->stages) {
        if (stage.kind == kind) {
            ++count;
        }
    }
    return count;
}

void expectOnlyStages(const std::shared_ptr<CompiledOutputs>& compiled,
                      std::initializer_list<CompiledExecutionStage::Kind> allowed_kinds) {
    for (const CompiledExecutionStage& stage : compiled->stages) {
        EXPECT_TRUE(std::find(allowed_kinds.begin(), allowed_kinds.end(), stage.kind) != allowed_kinds.end())
            << "representative ragged chain acquired an unexpected physical helper stage";
    }
}

Expression packedExtent(const Expression& values,
                        const Expression& offsets,
                        uint64_t capacity,
                        uint64_t elements_per_value) {
    return values.withRaggedRuntimeExtent(offsets, 2, capacity, elements_per_value);
}

Expression packedHostExtent(const Expression& values,
                            const Expression& host_carrier,
                            uint64_t capacity,
                            uint64_t elements_per_value) {
    return values.withRaggedRuntimeExtent(
        host_carrier, 2, capacity, elements_per_value, RaggedRuntimeExtentSource::HOST_EXTENT);
}

}  // namespace

TEST(RaggedCapacityPerformance, HostExtentTerminalMarkerIsMetadataAliasNotFullCapacityKernel) {
    REQUIRE_CUDA_DEVICE();
    constexpr uint64_t capacity = 66;

    Tensor values(gpuPlacement, TensorDescriptor(DataType::FP32, {capacity, 4}));
    Tensor host_carrier(gpuPlacement, TensorDescriptor(DataType::FP32, {capacity, 4}));

    const Expression values_expr = Expression::input("values", DataType::FP32, DataType::FP32);
    const Expression carrier_expr = Expression::input("host_carrier", DataType::FP32, DataType::FP32);
    const Expression output = packedHostExtent(values_expr, carrier_expr, capacity, 4);

    FusedEquation equation = FusedEquation::compile(Expression::outputs({{"y", output}}).physicalOutputs(), 0);
    const auto compiled = equation.compileForInputs({{"values", values}, {"host_carrier", host_carrier}});
    ASSERT_NE(compiled, nullptr);
    EXPECT_TRUE(compiled->stages.empty())
        << "HOST_EXTENT is metadata-only and must not create a full-capacity CUDA identity stage";
    ASSERT_EQ(compiled->final_outputs.size(), 1u);
    EXPECT_EQ(compiled->final_outputs[0].value_id, 0u)
        << "HOST_EXTENT output should reuse the values input storage directly";
}

TEST(RaggedCapacityPerformance, RepresentativeExpressionChainsContainOnlyLogicalComputeAndPhysicalConsumers) {
    REQUIRE_CUDA_DEVICE();
    constexpr uint64_t capacity = 100;

    // FC -> activation -> FC. RAGGED_VALUEWISE_EXTENT is metadata only, so the
    // physical plan must retain the two bucketed MATMUL consumers and may have
    // only active-aware fused valuewise work between them. A future epilogue
    // fusion is allowed to eliminate that valuewise stage.
    {
        Tensor x(gpuPlacement, TensorDescriptor(DataType::FP32, {capacity, 4}));
        Tensor w1(gpuPlacement, TensorDescriptor(DataType::FP32, {4, 8}));
        Tensor w2(gpuPlacement, TensorDescriptor(DataType::FP32, {8, 3}));
        Tensor offsets(gpuPlacement, TensorDescriptor(DataType::UINT32, {3}));

        const Expression x_expr = Expression::input("x", DataType::FP32, DataType::FP32);
        const Expression w1_expr = Expression::input("w1", DataType::FP32, DataType::FP32);
        const Expression w2_expr = Expression::input("w2", DataType::FP32, DataType::FP32);
        const Expression offsets_expr = Expression::input("offsets", DataType::UINT32, DataType::UINT32);

        const Expression fc1 = Expression::matmul(
            packedExtent(x_expr, offsets_expr, capacity, 4), w1_expr, false, false, DataType::FP32, DataType::FP32, capacity);
        const Expression activated = packedExtent(fc1, offsets_expr, capacity, 8).swish();
        const Expression fc2 = Expression::matmul(packedExtent(activated, offsets_expr, capacity, 8),
                                                  w2_expr,
                                                  false,
                                                  false,
                                                  DataType::FP32,
                                                  DataType::FP32,
                                                  capacity);

        FusedEquation equation = FusedEquation::compile(Expression::outputs({{"y", fc2}}).physicalOutputs(), 0);
        const auto compiled = equation.compileForInputs({{"x", x}, {"w1", w1}, {"w2", w2}, {"offsets", offsets}});
        ASSERT_NE(compiled, nullptr);
        EXPECT_EQ(countStages(compiled, CompiledExecutionStage::Kind::Matmul), 2u);
        EXPECT_LE(countStages(compiled, CompiledExecutionStage::Kind::FusedKernel), 1u);
        expectOnlyStages(compiled, {CompiledExecutionStage::Kind::Matmul, CompiledExecutionStage::Kind::FusedKernel});
    }

    // FC -> SwiGLU -> RMSNorm. The GLU work stays active-aware (or may be fused).
    // Tail sanitation is a stamped physical preparation stage, so it intentionally
    // does not appear in this compiler-stage inventory.
    {
        Tensor x(gpuPlacement, TensorDescriptor(DataType::FP32, {capacity, 4}));
        Tensor w(gpuPlacement, TensorDescriptor(DataType::FP32, {4, 8}));
        Tensor scale(gpuPlacement, TensorDescriptor(DataType::FP32, {4}));
        Tensor offsets(gpuPlacement, TensorDescriptor(DataType::UINT32, {3}));

        const Expression x_expr = Expression::input("x", DataType::FP32, DataType::FP32);
        const Expression w_expr = Expression::input("w", DataType::FP32, DataType::FP32);
        const Expression scale_expr = Expression::input("scale", DataType::FP32, DataType::FP32);
        const Expression offsets_expr = Expression::input("offsets", DataType::UINT32, DataType::UINT32);

        const Expression fc = Expression::matmul(
            packedExtent(x_expr, offsets_expr, capacity, 4), w_expr, false, false, DataType::FP32, DataType::FP32, capacity);
        const std::vector<uint64_t> half_dims{capacity, 4};
        const std::vector<uint64_t> full_strides{8, 1};

        // Match the production RaggedExpression/SwiGLU construction: storage aliases
        // are formed from the raw packed values, then the completed logical valuewise
        // result is given its ragged runtime extent.  Attaching the extent before a
        // STRIDED_VIEW would incorrectly ask the view itself to consume row-partition
        // metadata even though it is only a storage alias.
        const Expression value = fc.stridedView(half_dims, full_strides, 0);
        const Expression gate = fc.stridedView(half_dims, full_strides, 4);
        const Expression swiglu = value * gate.swish();
        const Expression normalized = Expression::rmsNorm(packedExtent(swiglu, offsets_expr, capacity, 4),
                                                          scale_expr,
                                                          4,
                                                          1.0e-5,
                                                          DataType::FP32,
                                                          DataType::FP32,
                                                          capacity);

        FusedEquation equation = FusedEquation::compile(Expression::outputs({{"y", normalized}}).physicalOutputs(), 0);
        const auto compiled = equation.compileForInputs({{"x", x}, {"w", w}, {"scale", scale}, {"offsets", offsets}});
        ASSERT_NE(compiled, nullptr);
        EXPECT_EQ(countStages(compiled, CompiledExecutionStage::Kind::Matmul), 1u);
        EXPECT_EQ(countStages(compiled, CompiledExecutionStage::Kind::RmsNorm), 1u);
        EXPECT_LE(countStages(compiled, CompiledExecutionStage::Kind::FusedKernel), 1u);
        expectOnlyStages(compiled,
                         {CompiledExecutionStage::Kind::Matmul,
                          CompiledExecutionStage::Kind::FusedKernel,
                          CompiledExecutionStage::Kind::RmsNorm});
    }

    // Ragged Attention -> activation. cuDNN Attention consumes the explicit row
    // partition; any following valuewise stage is active-aware and requires no
    // producer canonicalization stage.
    {
        Tensor q(gpuPlacement, TensorDescriptor(DataType::FP16, {capacity, 1, 8}));
        Tensor k(gpuPlacement, TensorDescriptor(DataType::FP16, {capacity, 1, 8}));
        Tensor v(gpuPlacement, TensorDescriptor(DataType::FP16, {capacity, 1, 8}));
        Tensor offsets(gpuPlacement, TensorDescriptor(DataType::UINT32, {3}));

        const Expression q_expr = Expression::input("q", DataType::FP32, DataType::FP16);
        const Expression k_expr = Expression::input("k", DataType::FP32, DataType::FP16);
        const Expression v_expr = Expression::input("v", DataType::FP32, DataType::FP16);
        const Expression offsets_expr = Expression::input("offsets", DataType::UINT32, DataType::UINT32);
        AttentionOptions options;
        options.q_layout = AttentionTensorLayout::BSHD;
        options.k_layout = AttentionTensorLayout::BSHD;
        options.v_layout = AttentionTensorLayout::BSHD;
        options.o_layout = AttentionTensorLayout::BSHD;
        options.compute_dtype = DataType::FP32;
        options.output_dtype = DataType::FP16;

        const Expression attention =
            Expression::scaledDotProductAttentionRagged(q_expr, k_expr, v_expr, offsets_expr, offsets_expr, options);
        const Expression activated = packedExtent(attention, offsets_expr, capacity, 8).swish();
        FusedEquation equation = FusedEquation::compile(Expression::outputs({{"y", activated}}).physicalOutputs(), 0);
        const auto compiled = equation.compileForInputs({{"q", q}, {"k", k}, {"v", v}, {"offsets", offsets}});
        ASSERT_NE(compiled, nullptr);
        EXPECT_EQ(countStages(compiled, CompiledExecutionStage::Kind::Attention), 1u);
        EXPECT_LE(countStages(compiled, CompiledExecutionStage::Kind::FusedKernel), 1u);
        expectOnlyStages(compiled,
                         {CompiledExecutionStage::Kind::Attention, CompiledExecutionStage::Kind::FusedKernel});
    }
}

TEST(RaggedCapacityPerformance, SegmentedMeanFlopsUseCachedExactNonEmptyRowCount) {
    REQUIRE_CUDA_DEVICE();
    constexpr uint64_t batch_size = 4;
    constexpr uint64_t capacity = 12;

    Tensor values(gpuPlacement, TensorDescriptor(DataType::FP32, {capacity}));
    Tensor offsets(gpuPlacement, TensorDescriptor(DataType::UINT32, {batch_size + 1}));
    RowPartitionRuntime partition(offsets, RowPartitionDescriptor(batch_size, capacity, DataType::UINT32));
    partition.setHostOffsets({0, 2, 2, 5, 6});  // 6 values, 3 non-empty rows.

    const RaggedTensorDescriptor descriptor(
        DataType::FP32, {}, batch_size, capacity, DataType::UINT32);
    const RaggedExpression input = RaggedExpression::input("tokens", descriptor);
    FusedEquation equation =
        FusedEquation::compile(Expression::outputs({{"y", input.segment_mean()}}).physicalOutputs(), 0);
    Stream stream(0);
    StampedExecutionPlan plan =
        equation.stamp({{"tokens.values", values}, {"tokens.offsets", offsets}}, stream);

    // One reduction FLOP per active scalar plus one division for each non-empty
    // row: 6 + 3 = 9. The non-empty-row term is cached at publication time.
    EXPECT_EQ(plan.flopCount(), 9U);
    // First two valid rows have lengths [2, 0]: 2 active values + 1 division.
    EXPECT_EQ(plan.logicalFlopCount(2), 3U);

    partition.setHostOffsets({0, 0, 1, 1, 1});  // 1 value, 1 non-empty row.
    EXPECT_EQ(plan.flopCount(), 2U);
    // First two valid rows have lengths [0, 1]: 1 active value + 1 division.
    EXPECT_EQ(plan.logicalFlopCount(2), 2U);
}

TEST(RaggedCapacityPerformance, RaggedAttentionFlopsUseRuntimeLogicalRowPairs) {
    REQUIRE_CUDA_DEVICE();
    constexpr uint64_t batch_size = 2;
    constexpr uint64_t capacity = 8;
    constexpr uint64_t heads = 1;
    constexpr uint64_t head_dim = 8;

    Tensor q(gpuPlacement, TensorDescriptor(DataType::FP16, {capacity, heads, head_dim}));
    Tensor k(gpuPlacement, TensorDescriptor(DataType::FP16, {capacity, heads, head_dim}));
    Tensor v(gpuPlacement, TensorDescriptor(DataType::FP16, {capacity, heads, head_dim}));
    Tensor q_offsets(gpuPlacement, TensorDescriptor(DataType::UINT32, {batch_size + 1}));
    Tensor kv_offsets(gpuPlacement, TensorDescriptor(DataType::UINT32, {batch_size + 1}));

    RowPartitionRuntime q_partition(
        q_offsets, RowPartitionDescriptor(batch_size, capacity, DataType::UINT32));
    RowPartitionRuntime kv_partition(
        kv_offsets, RowPartitionDescriptor(batch_size, capacity, DataType::UINT32));
    q_partition.setHostOffsets({0, 2, 5});       // lengths [2, 3]
    kv_partition.setHostOffsets({0, 4, 5});      // lengths [4, 1]

    const Expression q_expr = Expression::input("q", DataType::FP32, DataType::FP16);
    const Expression k_expr = Expression::input("k", DataType::FP32, DataType::FP16);
    const Expression v_expr = Expression::input("v", DataType::FP32, DataType::FP16);
    const Expression q_offsets_expr = Expression::input("q_offsets", DataType::UINT32, DataType::UINT32);
    const Expression kv_offsets_expr = Expression::input("kv_offsets", DataType::UINT32, DataType::UINT32);

    AttentionOptions options;
    options.q_layout = AttentionTensorLayout::BSHD;
    options.k_layout = AttentionTensorLayout::BSHD;
    options.v_layout = AttentionTensorLayout::BSHD;
    options.o_layout = AttentionTensorLayout::BSHD;
    options.compute_dtype = DataType::FP32;
    options.output_dtype = DataType::FP16;

    const Expression attention = Expression::scaledDotProductAttentionRagged(
        q_expr, k_expr, v_expr, q_offsets_expr, kv_offsets_expr, options);
    FusedEquation equation =
        FusedEquation::compile(Expression::outputs({{"y", attention}}).physicalOutputs(), 0);
    Stream stream(0);
    StampedExecutionPlan plan = equation.stamp({{"q", q},
                                                {"k", k},
                                                {"v", v},
                                                {"q_offsets", q_offsets},
                                                {"kv_offsets", kv_offsets}},
                                               stream);

    // Useful score pairs are per-row, never the Cartesian product of packed
    // capacities: 2*4 + 3*1 = 11. The existing Attention convention charges
    // 2*D for QK, 2*D for PV, and 5 softmax/mask/scale FLOPs per score.
    constexpr uint64_t flops_per_score = (2 * head_dim) + (2 * head_dim) + 5;
    constexpr uint64_t bytes_per_token = heads * head_dim * sizeof(uint16_t);
    EXPECT_EQ(plan.flopCount(), 11u * heads * flops_per_score);
    EXPECT_EQ(plan.logicalFlopCount(), 11u * heads * flops_per_score);
    EXPECT_EQ(plan.logicalByteCount(), (5u + 5u + 5u + 5u) * bytes_per_token);
    EXPECT_EQ(plan.logicalFlopCount(1), 8u * heads * flops_per_score);
    EXPECT_EQ(plan.logicalByteCount(1), (2u + 4u + 4u + 2u) * bytes_per_token);

    // FLOP accounting is runtime metadata, not a stamp-time constant. Reuse the
    // same plan with a different logical partition and require the count to move
    // with the useful work while physical capacity remains unchanged.
    q_partition.setHostOffsets({0, 1, 2});       // lengths [1, 1]
    kv_partition.setHostOffsets({0, 1, 3});      // lengths [1, 2]
    EXPECT_EQ(plan.flopCount(), 3u * heads * flops_per_score);
    EXPECT_EQ(plan.logicalFlopCount(), 3u * heads * flops_per_score);
    EXPECT_EQ(plan.logicalByteCount(), (2u + 3u + 3u + 2u) * bytes_per_token);
    EXPECT_EQ(plan.logicalFlopCount(1), 1u * heads * flops_per_score);
    EXPECT_EQ(plan.logicalByteCount(1), (1u + 1u + 1u + 1u) * bytes_per_token);
}

TEST(RaggedCapacityPerformance, MixedDenseQueryRaggedKvAttentionFlopsUseRuntimeLogicalRowPairs) {
    REQUIRE_CUDA_DEVICE();
    constexpr uint64_t batch_size = 2;
    constexpr uint64_t query_length = 2;
    constexpr uint64_t kv_capacity = 8;
    constexpr uint64_t heads = 1;
    constexpr uint64_t head_dim = 8;

    Tensor q(gpuPlacement, TensorDescriptor(DataType::FP16, {batch_size, query_length, heads, head_dim}));
    Tensor k(gpuPlacement, TensorDescriptor(DataType::FP16, {kv_capacity, heads, head_dim}));
    Tensor v(gpuPlacement, TensorDescriptor(DataType::FP16, {kv_capacity, heads, head_dim}));
    Tensor q_offsets(gpuPlacement, TensorDescriptor(DataType::UINT32, {batch_size + 1}));
    Tensor kv_offsets(gpuPlacement, TensorDescriptor(DataType::UINT32, {batch_size + 1}));

    RowPartitionRuntime q_partition(
        q_offsets, RowPartitionDescriptor(batch_size, batch_size * query_length, DataType::UINT32));
    RowPartitionRuntime kv_partition(
        kv_offsets, RowPartitionDescriptor(batch_size, kv_capacity, DataType::UINT32));
    q_partition.setHostOffsets({0, 2, 4});       // fixed dense query lengths [2, 2]
    kv_partition.setHostOffsets({0, 4, 5});      // ragged K/V lengths [4, 1]

    const Expression q_expr = Expression::input("q", DataType::FP32, DataType::FP16);
    const Expression k_expr = Expression::input("k", DataType::FP32, DataType::FP16);
    const Expression v_expr = Expression::input("v", DataType::FP32, DataType::FP16);
    const Expression q_offsets_expr = Expression::input("q_offsets", DataType::UINT32, DataType::UINT32);
    const Expression kv_offsets_expr = Expression::input("kv_offsets", DataType::UINT32, DataType::UINT32);

    AttentionOptions options;
    options.q_layout = AttentionTensorLayout::BSHD;
    options.k_layout = AttentionTensorLayout::BSHD;
    options.v_layout = AttentionTensorLayout::BSHD;
    options.o_layout = AttentionTensorLayout::BSHD;
    options.compute_dtype = DataType::FP32;
    options.output_dtype = DataType::FP16;

    const Expression attention = Expression::scaledDotProductAttentionRagged(
        q_expr, k_expr, v_expr, q_offsets_expr, kv_offsets_expr, options);
    FusedEquation equation =
        FusedEquation::compile(Expression::outputs({{"y", attention}}).physicalOutputs(), 0);
    Stream stream(0);
    StampedExecutionPlan plan = equation.stamp({{"q", q},
                                                {"k", k},
                                                {"v", v},
                                                {"q_offsets", q_offsets},
                                                {"kv_offsets", kv_offsets}},
                                               stream);

    // 2*4 + 2*1 = 10 useful Q/K score pairs. This is the ProductTransformer-style
    // dense-Q / ragged-KV case; K/V packed capacity must not enter the logical
    // FLOP count.
    constexpr uint64_t flops_per_score = (2 * head_dim) + (2 * head_dim) + 5;
    constexpr uint64_t bytes_per_token = heads * head_dim * sizeof(uint16_t);
    EXPECT_EQ(plan.flopCount(), 10u * heads * flops_per_score);
    EXPECT_EQ(plan.logicalFlopCount(), 10u * heads * flops_per_score);
    EXPECT_EQ(plan.logicalByteCount(), (4u + 5u + 5u + 4u) * bytes_per_token);
    EXPECT_EQ(plan.logicalFlopCount(1), 8u * heads * flops_per_score);
    EXPECT_EQ(plan.logicalByteCount(1), (2u + 4u + 4u + 2u) * bytes_per_token);
}

TEST(RaggedCapacityPerformance, Lwa4c1RaggedAttentionForwardLogicalWorkIsPrefixExactAndCapacityInvariant) {
    REQUIRE_CUDA_DEVICE();
    constexpr uint64_t batch_size = 2;
    constexpr uint64_t heads = 1;
    constexpr uint64_t head_dim = 8;
    constexpr uint64_t valid_rows = 1;
    constexpr uint64_t flops_per_score = (2 * head_dim) + (2 * head_dim) + 5;
    constexpr uint64_t bytes_per_token = heads * head_dim * sizeof(uint16_t);

    auto logical_work_for_capacity = [&](uint64_t capacity) {
        Tensor q(gpuPlacement, TensorDescriptor(DataType::FP16, {capacity, heads, head_dim}));
        Tensor k(gpuPlacement, TensorDescriptor(DataType::FP16, {capacity, heads, head_dim}));
        Tensor v(gpuPlacement, TensorDescriptor(DataType::FP16, {capacity, heads, head_dim}));
        Tensor q_offsets(gpuPlacement, TensorDescriptor(DataType::UINT32, {batch_size + 1}));
        Tensor kv_offsets(gpuPlacement, TensorDescriptor(DataType::UINT32, {batch_size + 1}));

        // Device payloads remain deliberately untouched. Logical-work telemetry
        // must use only the authoritative host publications below.
        RowPartitionRuntime q_partition(
            q_offsets, RowPartitionDescriptor(batch_size, capacity, DataType::UINT32));
        RowPartitionRuntime kv_partition(
            kv_offsets, RowPartitionDescriptor(batch_size, capacity, DataType::UINT32));
        q_partition.setHostOffsets({0, 2, 5});       // lengths [2, 3]
        kv_partition.setHostOffsets({0, 4, 5});      // lengths [4, 1]

        const Expression q_expr = Expression::input("q", DataType::FP32, DataType::FP16);
        const Expression k_expr = Expression::input("k", DataType::FP32, DataType::FP16);
        const Expression v_expr = Expression::input("v", DataType::FP32, DataType::FP16);
        const Expression q_offsets_expr = Expression::input("q_offsets", DataType::UINT32, DataType::UINT32);
        const Expression kv_offsets_expr = Expression::input("kv_offsets", DataType::UINT32, DataType::UINT32);

        AttentionOptions options;
        options.q_layout = AttentionTensorLayout::BSHD;
        options.k_layout = AttentionTensorLayout::BSHD;
        options.v_layout = AttentionTensorLayout::BSHD;
        options.o_layout = AttentionTensorLayout::BSHD;
        options.compute_dtype = DataType::FP32;
        options.output_dtype = DataType::FP16;

        const Expression attention = Expression::scaledDotProductAttentionRagged(
            q_expr, k_expr, v_expr, q_offsets_expr, kv_offsets_expr, options);
        FusedEquation equation =
            FusedEquation::compile(Expression::outputs({{"y", attention}}).physicalOutputs(), 0);
        Stream stream(0);
        StampedExecutionPlan plan = equation.stamp({{"q", q},
                                                    {"k", k},
                                                    {"v", v},
                                                    {"q_offsets", q_offsets},
                                                    {"kv_offsets", kv_offsets}},
                                                   stream);

        const std::array<uint64_t, 4> initial{
            plan.logicalFlopCount(),
            plan.logicalByteCount(),
            plan.logicalFlopCount(valid_rows),
            plan.logicalByteCount(valid_rows)};
        EXPECT_EQ(initial[0], 11u * heads * flops_per_score);
        EXPECT_EQ(initial[1], (5u + 5u + 5u + 5u) * bytes_per_token);
        EXPECT_EQ(initial[2], 8u * heads * flops_per_score);
        EXPECT_EQ(initial[3], (2u + 4u + 4u + 2u) * bytes_per_token);

        // Preserve the same total active token counts but change row geometry.
        // Logical bytes stay fixed while exact score-pair FLOPs move from
        // 2*4 + 3*1 = 11 to 1*1 + 4*4 = 17.
        q_partition.setHostOffsets({0, 1, 5});
        kv_partition.setHostOffsets({0, 1, 5});
        EXPECT_EQ(plan.logicalFlopCount(), 17u * heads * flops_per_score);
        EXPECT_EQ(plan.logicalByteCount(), initial[1]);
        EXPECT_EQ(plan.logicalFlopCount(valid_rows), 1u * heads * flops_per_score);
        EXPECT_EQ(plan.logicalByteCount(valid_rows), (1u + 1u + 1u + 1u) * bytes_per_token);

        return initial;
    };

    EXPECT_EQ(logical_work_for_capacity(8), logical_work_for_capacity(16));
}

TEST(RaggedCapacityPerformance, Lwa4c1SelfAttentionUsesExactPrefixSquaredRowGeometry) {
    REQUIRE_CUDA_DEVICE();
    constexpr uint64_t batch_size = 3;
    constexpr uint64_t capacity = 10;
    constexpr uint64_t heads = 1;
    constexpr uint64_t head_dim = 8;
    constexpr uint64_t flops_per_score = (2 * head_dim) + (2 * head_dim) + 5;
    constexpr uint64_t bytes_per_token = heads * head_dim * sizeof(uint16_t);

    Tensor q(gpuPlacement, TensorDescriptor(DataType::FP16, {capacity, heads, head_dim}));
    Tensor k(gpuPlacement, TensorDescriptor(DataType::FP16, {capacity, heads, head_dim}));
    Tensor v(gpuPlacement, TensorDescriptor(DataType::FP16, {capacity, heads, head_dim}));
    Tensor offsets(gpuPlacement, TensorDescriptor(DataType::UINT32, {batch_size + 1}));
    RowPartitionRuntime partition(offsets, RowPartitionDescriptor(batch_size, capacity, DataType::UINT32));
    partition.setHostOffsets({0, 2, 2, 5});  // lengths [2, 0, 3], squares sum to 13.

    const Expression q_expr = Expression::input("q", DataType::FP32, DataType::FP16);
    const Expression k_expr = Expression::input("k", DataType::FP32, DataType::FP16);
    const Expression v_expr = Expression::input("v", DataType::FP32, DataType::FP16);
    const Expression offsets_expr = Expression::input("offsets", DataType::UINT32, DataType::UINT32);
    AttentionOptions options;
    options.q_layout = AttentionTensorLayout::BSHD;
    options.k_layout = AttentionTensorLayout::BSHD;
    options.v_layout = AttentionTensorLayout::BSHD;
    options.o_layout = AttentionTensorLayout::BSHD;
    options.compute_dtype = DataType::FP32;
    options.output_dtype = DataType::FP16;

    const Expression attention = Expression::scaledDotProductAttentionRagged(
        q_expr, k_expr, v_expr, offsets_expr, offsets_expr, options);
    FusedEquation equation =
        FusedEquation::compile(Expression::outputs({{"y", attention}}).physicalOutputs(), 0);
    Stream stream(0);
    StampedExecutionPlan plan =
        equation.stamp({{"q", q}, {"k", k}, {"v", v}, {"offsets", offsets}}, stream);

    EXPECT_EQ(plan.logicalFlopCount(), 13u * heads * flops_per_score);
    EXPECT_EQ(plan.logicalByteCount(), (5u + 5u + 5u + 5u) * bytes_per_token);
    // First two rows have lengths [2,0], so score pairs are exactly 4.
    EXPECT_EQ(plan.logicalFlopCount(2), 4u * heads * flops_per_score);
    EXPECT_EQ(plan.logicalByteCount(2), (2u + 2u + 2u + 2u) * bytes_per_token);
}

TEST(RaggedCapacityPerformance, PackedMatmulFlopsUseRuntimeActiveRowsForForwardDgradAndWgrad) {
    REQUIRE_CUDA_DEVICE();
    constexpr uint64_t batch_size = 2;
    constexpr uint64_t capacity = 16;
    constexpr uint64_t input_width = 4;
    constexpr uint64_t output_width = 6;

    Tensor x(gpuPlacement, TensorDescriptor(DataType::FP32, {capacity, input_width}));
    Tensor w(gpuPlacement, TensorDescriptor(DataType::FP32, {input_width, output_width}));
    Tensor dy(gpuPlacement, TensorDescriptor(DataType::FP32, {capacity, output_width}));
    Tensor offsets(gpuPlacement, TensorDescriptor(DataType::UINT32, {batch_size + 1}));

    RowPartitionRuntime partition(offsets, RowPartitionDescriptor(batch_size, capacity, DataType::UINT32));
    partition.setHostOffsets({0, 2, 5});  // 5 useful packed token rows.

    const Expression x_expr = Expression::input("x", DataType::FP32, DataType::FP32);
    const Expression w_expr = Expression::input("w", DataType::FP32, DataType::FP32);
    const Expression offsets_expr = Expression::input("offsets", DataType::UINT32, DataType::UINT32);
    const Expression y = Expression::matmul(packedExtent(x_expr, offsets_expr, capacity, input_width),
                                            w_expr,
                                            false,
                                            false,
                                            DataType::FP32,
                                            DataType::FP32,
                                            capacity);

    FusedEquation forward = FusedEquation::compile(Expression::outputs({{"y", y}}).physicalOutputs(), 0);
    FusedEquation dgrad = forward.compileBackward({"x"}, "dy");
    FusedEquation wgrad = forward.compileBackward({"w"}, "dy");

    Stream stream(0);
    StampedExecutionPlan forward_plan =
        forward.stamp({{"x", x}, {"w", w}, {"offsets", offsets}}, stream);
    StampedExecutionPlan dgrad_plan =
        dgrad.stamp({{"x", x}, {"w", w}, {"offsets", offsets}, {"dy", dy}}, stream);
    StampedExecutionPlan wgrad_plan =
        wgrad.stamp({{"x", x}, {"w", w}, {"offsets", offsets}, {"dy", dy}}, stream);

    constexpr uint64_t flops_per_active_row = input_width * output_width * 2;
    EXPECT_EQ(forward_plan.flopCount(), 5u * flops_per_active_row);
    EXPECT_EQ(dgrad_plan.flopCount(), 5u * flops_per_active_row);
    EXPECT_EQ(wgrad_plan.flopCount(), 5u * flops_per_active_row);

    // The same stamped kernels retain full capacity, but useful-work reporting
    // follows the runtime row partition rather than the selected cuBLASLt bucket.
    partition.setHostOffsets({0, 1, 2});
    EXPECT_EQ(forward_plan.flopCount(), 2u * flops_per_active_row);
    EXPECT_EQ(dgrad_plan.flopCount(), 2u * flops_per_active_row);
    EXPECT_EQ(wgrad_plan.flopCount(), 2u * flops_per_active_row);

    partition.setHostOffsets({0, 0, 0});
    EXPECT_EQ(forward_plan.flopCount(), 0u);
    EXPECT_EQ(dgrad_plan.flopCount(), 0u);
    EXPECT_EQ(wgrad_plan.flopCount(), 0u);
}

TEST(RaggedCapacityPerformance, Lwa4b3PackedMatmulLogicalWorkUsesOneRuntimeMknGeometry) {
    REQUIRE_CUDA_DEVICE();
    constexpr uint64_t batch_size = 2;
    constexpr uint64_t input_width = 4;
    constexpr uint64_t output_width = 6;
    constexpr uint64_t valid_rows = 1;
    const std::vector<uint64_t> host_offsets{0, 2, 5};

    auto logical_work_for_capacity = [&](uint64_t capacity) {
        Tensor x(gpuPlacement, TensorDescriptor(DataType::FP32, {capacity, input_width}));
        Tensor w(gpuPlacement, TensorDescriptor(DataType::FP32, {input_width, output_width}));
        Tensor dy(gpuPlacement, TensorDescriptor(DataType::FP32, {capacity, output_width}));
        // Keep the device carrier stale. LWA telemetry must use only the
        // authoritative host partition published below.
        Tensor offsets(gpuPlacement, TensorDescriptor(DataType::UINT32, {batch_size + 1}));
        RowPartitionRuntime partition(offsets, RowPartitionDescriptor(batch_size, capacity, DataType::UINT32));
        partition.setHostOffsets(host_offsets);

        const Expression x_expr = Expression::input("x", DataType::FP32, DataType::FP32);
        const Expression w_expr = Expression::input("w", DataType::FP32, DataType::FP32);
        const Expression offsets_expr = Expression::input("offsets", DataType::UINT32, DataType::UINT32);
        const Expression y = Expression::matmul(packedExtent(x_expr, offsets_expr, capacity, input_width),
                                                w_expr,
                                                false,
                                                false,
                                                DataType::FP32,
                                                DataType::FP32,
                                                capacity);

        FusedEquation forward = FusedEquation::compile(Expression::outputs({{"y", y}}).physicalOutputs(), 0);
        FusedEquation dgrad = forward.compileBackward({"x"}, "dy");
        FusedEquation wgrad = forward.compileBackward({"w"}, "dy");

        Stream stream(0);
        StampedExecutionPlan forward_plan =
            forward.stamp({{"x", x}, {"w", w}, {"offsets", offsets}}, stream);
        StampedExecutionPlan dgrad_plan =
            dgrad.stamp({{"x", x}, {"w", w}, {"offsets", offsets}, {"dy", dy}}, stream);
        StampedExecutionPlan wgrad_plan =
            wgrad.stamp({{"x", x}, {"w", w}, {"offsets", offsets}, {"dy", dy}}, stream);

        // These three plans exercise the production packed-row geometries:
        //   forward: RowsA,           M=active, K=input,  N=output
        //   dgrad:   RowsA + rhs^T,   M=active, K=output, N=input
        //   wgrad:   RowsAAndRowsB + lhs^T,
        //                              M=input,  K=active, N=output
        constexpr uint64_t full_active_rows = 5;
        constexpr uint64_t prefix_active_rows = 2;
        constexpr uint64_t full_flops = full_active_rows * input_width * output_width * 2;
        constexpr uint64_t prefix_flops = prefix_active_rows * input_width * output_width * 2;
        constexpr uint64_t full_bytes =
            (full_active_rows * input_width + input_width * output_width +
             full_active_rows * output_width) * sizeof(float);
        constexpr uint64_t prefix_bytes =
            (prefix_active_rows * input_width + input_width * output_width +
             prefix_active_rows * output_width) * sizeof(float);

        for (const StampedExecutionPlan* plan : {&forward_plan, &dgrad_plan, &wgrad_plan}) {
            EXPECT_EQ(plan->logicalFlopCount(), full_flops);
            EXPECT_EQ(plan->logicalByteCount(), full_bytes);
            EXPECT_EQ(plan->logicalFlopCount(valid_rows), prefix_flops);
            EXPECT_EQ(plan->logicalByteCount(valid_rows), prefix_bytes);
        }

        // Tail rows beyond validExampleCount must not perturb prefix work.
        partition.setHostOffsets({0, 2, 7});
        for (const StampedExecutionPlan* plan : {&forward_plan, &dgrad_plan, &wgrad_plan}) {
            EXPECT_EQ(plan->logicalFlopCount(valid_rows), prefix_flops);
            EXPECT_EQ(plan->logicalByteCount(valid_rows), prefix_bytes);
        }

        partition.setHostOffsets(host_offsets);
        std::vector<uint64_t> work{
            forward_plan.logicalFlopCount(), forward_plan.logicalByteCount(),
            dgrad_plan.logicalFlopCount(), dgrad_plan.logicalByteCount(),
            wgrad_plan.logicalFlopCount(), wgrad_plan.logicalByteCount(),
            forward_plan.logicalFlopCount(valid_rows), forward_plan.logicalByteCount(valid_rows),
            dgrad_plan.logicalFlopCount(valid_rows), dgrad_plan.logicalByteCount(valid_rows),
            wgrad_plan.logicalFlopCount(valid_rows), wgrad_plan.logicalByteCount(valid_rows)};

        // With no active rows, forward/dgrad have no logical output. Wgrad still
        // produces the fixed-size zero parameter gradient, so only that output
        // write remains in logical bytes.
        partition.setHostOffsets({0, 0, 0});
        EXPECT_EQ(forward_plan.logicalFlopCount(), 0U);
        EXPECT_EQ(forward_plan.logicalByteCount(), 0U);
        EXPECT_EQ(dgrad_plan.logicalFlopCount(), 0U);
        EXPECT_EQ(dgrad_plan.logicalByteCount(), 0U);
        EXPECT_EQ(wgrad_plan.logicalFlopCount(), 0U);
        EXPECT_EQ(wgrad_plan.logicalByteCount(), input_width * output_width * sizeof(float));
        return work;
    };

    EXPECT_EQ(logical_work_for_capacity(8), logical_work_for_capacity(16));
}

TEST(RaggedCapacityPerformance, RaggedCausalConv1dFlopsUseRuntimeActiveValuesForForwardDgradAndWgrad) {
    REQUIRE_CUDA_DEVICE();
    constexpr uint64_t batch_size = 2;
    constexpr uint64_t max_total_values = 16;
    constexpr uint64_t max_values_per_row = 8;
    constexpr uint64_t input_channels = 4;
    constexpr uint64_t output_channels = 6;
    constexpr uint64_t kernel_width = 3;
    constexpr uint64_t groups = 2;

    Tensor x(gpuPlacement, TensorDescriptor(DataType::FP32, {max_total_values, input_channels}));
    Tensor filter(gpuPlacement,
                  TensorDescriptor(DataType::FP32, {output_channels, input_channels / groups, kernel_width}));
    Tensor dy(gpuPlacement, TensorDescriptor(DataType::FP32, {max_total_values, output_channels}));
    Tensor offsets(gpuPlacement, TensorDescriptor(DataType::UINT32, {batch_size + 1}));

    RowPartitionRuntime partition(
        offsets,
        RowPartitionDescriptor(batch_size, max_total_values, DataType::UINT32, max_values_per_row));
    partition.setHostOffsets({0, 2, 5});  // 5 useful logical convolution positions.

    const RaggedTensorDescriptor descriptor(DataType::FP32,
                                            {input_channels},
                                            batch_size,
                                            max_total_values,
                                            max_values_per_row,
                                            DataType::UINT32);
    const RaggedExpression input = RaggedExpression::input("tokens", descriptor);
    const Expression filter_expr = Expression::input("filter", std::nullopt, DataType::FP32);
    const RaggedExpression output = input.causalConv1d(filter_expr,
                                                       output_channels,
                                                       kernel_width,
                                                       1,
                                                       DataType::FP32,
                                                       DataType::FP32,
                                                       groups);
    FusedEquation forward =
        FusedEquation::compile(Expression::outputs({{"y", output.getValues()}}).physicalOutputs(), 0);
    FusedEquation dgrad = forward.compileBackward({"tokens.values"}, "dy");
    FusedEquation wgrad = forward.compileBackward({"filter"}, "dy");

    Stream stream(0);
    StampedExecutionPlan forward_plan =
        forward.stamp({{"tokens.values", x}, {"tokens.offsets", offsets}, {"filter", filter}}, stream);
    StampedExecutionPlan dgrad_plan = dgrad.stamp(
        {{"tokens.values", x}, {"tokens.offsets", offsets}, {"filter", filter}, {"dy", dy}}, stream);
    StampedExecutionPlan wgrad_plan = wgrad.stamp(
        {{"tokens.values", x}, {"tokens.offsets", offsets}, {"filter", filter}, {"dy", dy}}, stream);

    constexpr uint64_t flops_per_active_value =
        output_channels * (input_channels / groups) * kernel_width * 2;
    EXPECT_EQ(forward_plan.flopCount(), 5u * flops_per_active_value);
    EXPECT_EQ(dgrad_plan.flopCount(), 5u * flops_per_active_value);
    EXPECT_EQ(wgrad_plan.flopCount(), 5u * flops_per_active_value);

    // Reusing the exact same stamped plans with a smaller logical batch must
    // reduce reported useful FLOPs even though packed/padded capacity is fixed.
    partition.setHostOffsets({0, 1, 2});
    EXPECT_EQ(forward_plan.flopCount(), 2u * flops_per_active_value);
    EXPECT_EQ(dgrad_plan.flopCount(), 2u * flops_per_active_value);
    EXPECT_EQ(wgrad_plan.flopCount(), 2u * flops_per_active_value);

    partition.setHostOffsets({0, 0, 0});
    EXPECT_EQ(forward_plan.flopCount(), 0u);
    EXPECT_EQ(dgrad_plan.flopCount(), 0u);
    EXPECT_EQ(wgrad_plan.flopCount(), 0u);
}

TEST(RaggedCapacityPerformance, Lwa4dRaggedCausalConv1dLogicalWorkUsesSemanticPackedExtentNotPaddedWidth) {
    REQUIRE_CUDA_DEVICE();
    constexpr uint64_t batch_size = 3;
    constexpr uint64_t input_channels = 4;
    constexpr uint64_t output_channels = 6;
    constexpr uint64_t kernel_width = 3;
    constexpr uint64_t groups = 2;
    constexpr uint64_t valid_rows = 2;
    const std::vector<uint64_t> host_offsets{0, 2, 5, 7};

    auto logical_work_for_capacity = [&](uint64_t capacity,
                                         uint64_t max_values_per_row,
                                         uint64_t selected_width) {
        Stream stream(0);
        // Deliberately leave the device payload stale. Logical-work telemetry
        // must use only the authoritative host publication below.
        Tensor offsets(gpuPlacement, TensorDescriptor(DataType::UINT32, {batch_size + 1}));
        RowPartitionRuntime partition(
            offsets,
            RowPartitionDescriptor(batch_size, capacity, DataType::UINT32, max_values_per_row));
        partition.setHostOffsets(host_offsets);

        Tensor filter(gpuPlacement,
                      TensorDescriptor(DataType::FP32,
                                       {output_channels, input_channels / groups, kernel_width}));
        Tensor dw(gpuPlacement,
                  TensorDescriptor(DataType::FP32,
                                   {output_channels, input_channels / groups, kernel_width}));

        const PaddedRaggedSequencePlan x_plan =
            preparePaddedRaggedSequencePlan(partition, input_channels, DataType::FP32, selected_width);
        const PaddedRaggedSequencePlan y_plan =
            preparePaddedRaggedSequencePlan(partition, output_channels, DataType::FP32, selected_width);
        auto padded_x =
            std::make_shared<PaddedRaggedSequence>(x_plan, offsets, gpuPlacement, max_values_per_row);
        auto padded_y =
            std::make_shared<PaddedRaggedSequence>(y_plan, offsets, gpuPlacement, max_values_per_row);
        auto padded_dy =
            std::make_shared<PaddedRaggedSequence>(y_plan, offsets, gpuPlacement, max_values_per_row);
        auto padded_dx =
            std::make_shared<PaddedRaggedSequence>(x_plan, offsets, gpuPlacement, max_values_per_row);

        auto compiled_forward = std::make_shared<CompiledRaggedConv1dCausal>(DataType::FP32,
                                                                             DataType::FP32,
                                                                             DataType::FP32,
                                                                             DataType::FP32,
                                                                             DataType::UINT32,
                                                                             batch_size,
                                                                             capacity,
                                                                             max_values_per_row,
                                                                             input_channels,
                                                                             output_channels,
                                                                             kernel_width,
                                                                             groups,
                                                                             /*dilation=*/2);
        auto compiled_dgrad = std::make_shared<CompiledRaggedConv1dCausalBackwardData>(DataType::FP32,
                                                                                       DataType::FP32,
                                                                                       DataType::FP32,
                                                                                       DataType::FP32,
                                                                                       DataType::UINT32,
                                                                                       batch_size,
                                                                                       capacity,
                                                                                       max_values_per_row,
                                                                                       input_channels,
                                                                                       output_channels,
                                                                                       kernel_width,
                                                                                       groups,
                                                                                       /*dilation=*/2);
        auto compiled_wgrad = std::make_shared<CompiledRaggedConv1dCausalBackwardFilter>(DataType::FP32,
                                                                                         DataType::FP32,
                                                                                         DataType::FP32,
                                                                                         DataType::FP32,
                                                                                         DataType::UINT32,
                                                                                         batch_size,
                                                                                         capacity,
                                                                                         max_values_per_row,
                                                                                         input_channels,
                                                                                         output_channels,
                                                                                         kernel_width,
                                                                                         groups,
                                                                                         /*dilation=*/2);

        auto forward = std::make_shared<StampedRaggedConv1dCausal>(
            compiled_forward, padded_x, filter, offsets, padded_y, stream);
        auto dgrad = std::make_shared<StampedRaggedConv1dCausalBackwardData>(
            compiled_dgrad, filter, padded_dy, offsets, padded_dx, stream);
        auto wgrad = std::make_shared<StampedRaggedConv1dCausalBackwardFilter>(
            compiled_wgrad, padded_x, padded_dy, offsets, dw, stream);
        StampedExecutionStage forward_stage(forward);
        StampedExecutionStage dgrad_stage(dgrad);
        StampedExecutionStage wgrad_stage(wgrad);

        // These two configurations have identical semantic rows but explicitly
        // different physical padded widths. The constructors build the whole
        // cuDNN plan family but must preserve the selected prepared width.
        EXPECT_EQ(forward->diagnostic().active_values, 7u);
        EXPECT_EQ(forward->diagnostic().selected_width_capacity, selected_width);
        EXPECT_EQ(dgrad->diagnostic().selected_width_capacity, selected_width);
        EXPECT_EQ(wgrad->diagnostic().selected_width_capacity, selected_width);

        constexpr uint64_t filter_elements =
            output_channels * (input_channels / groups) * kernel_width;
        constexpr uint64_t flops_per_active_value = filter_elements * 2;
        constexpr uint64_t bytes_per_active_value =
            (input_channels + output_channels) * sizeof(float);
        constexpr uint64_t filter_bytes = filter_elements * sizeof(float);

        auto expect_work = [&](uint64_t valid_example_count, uint64_t active_values) {
            const uint64_t expected_flops = active_values * flops_per_active_value;
            const uint64_t expected_forward_or_dgrad_bytes =
                active_values == 0 ? 0 : active_values * bytes_per_active_value + filter_bytes;
            const uint64_t expected_wgrad_bytes = active_values * bytes_per_active_value + filter_bytes;

            EXPECT_EQ(forward_stage.logicalFlopCount(valid_example_count), expected_flops);
            EXPECT_EQ(dgrad_stage.logicalFlopCount(valid_example_count), expected_flops);
            EXPECT_EQ(wgrad_stage.logicalFlopCount(valid_example_count), expected_flops);
            EXPECT_EQ(forward_stage.logicalByteCount(valid_example_count), expected_forward_or_dgrad_bytes);
            EXPECT_EQ(dgrad_stage.logicalByteCount(valid_example_count), expected_forward_or_dgrad_bytes);
            EXPECT_EQ(wgrad_stage.logicalByteCount(valid_example_count), expected_wgrad_bytes);
        };

        expect_work(/*valid_example_count=*/0, /*active_values=*/7);
        expect_work(valid_rows, /*active_values=*/5);

        // Changing only rows beyond validExampleCount must not alter prefix
        // accounting, while the full-batch logical extent follows the new
        // authoritative host publication. No padded representation is touched.
        partition.setHostOffsets({0, 2, 5, 8});
        expect_work(valid_rows, /*active_values=*/5);
        expect_work(/*valid_example_count=*/0, /*active_values=*/8);

        partition.setHostOffsets(host_offsets);
        const std::array<uint64_t, 12> work{
            forward_stage.logicalFlopCount(), forward_stage.logicalByteCount(),
            dgrad_stage.logicalFlopCount(), dgrad_stage.logicalByteCount(),
            wgrad_stage.logicalFlopCount(), wgrad_stage.logicalByteCount(),
            forward_stage.logicalFlopCount(valid_rows), forward_stage.logicalByteCount(valid_rows),
            dgrad_stage.logicalFlopCount(valid_rows), dgrad_stage.logicalByteCount(valid_rows),
            wgrad_stage.logicalFlopCount(valid_rows), wgrad_stage.logicalByteCount(valid_rows)};

        // No semantic positions means no forward/dgrad tensor traffic. Wgrad
        // still produces the fixed-size zero parameter-gradient tensor.
        partition.setHostOffsets({0, 0, 0, 0});
        expect_work(/*valid_example_count=*/0, /*active_values=*/0);
        return work;
    };

    // max_values_per_row=6 allows an explicitly selected padded width of 6;
    // max_values_per_row=16 allows width 8. Both carry the exact same semantic
    // rows [2,3,2], proving logical work is invariant to unused packed capacity
    // and to the selected physical padded implementation width.
    EXPECT_EQ(logical_work_for_capacity(/*capacity=*/8, /*max_values_per_row=*/6, /*selected_width=*/6),
              logical_work_for_capacity(/*capacity=*/16, /*max_values_per_row=*/16, /*selected_width=*/8));
}

TEST(RaggedCapacityPerformance, PackedConsumerSanitationAccountingTracksSelectedBucketNotFullCapacity) {
    REQUIRE_CUDA_DEVICE();
    constexpr uint64_t capacity = 100;
    constexpr uint64_t width = 4;

    Tensor x(gpuPlacement, TensorDescriptor(DataType::FP32, {capacity, width}));
    Tensor w(gpuPlacement, TensorDescriptor(DataType::FP32, {width, width}));
    Tensor scale(gpuPlacement, TensorDescriptor(DataType::FP32, {width}));
    Tensor offsets(gpuPlacement, TensorDescriptor(DataType::UINT32, {3}));

    RowPartitionRuntime row_partition(
        offsets, RowPartitionDescriptor(/*batchSize=*/2, capacity, DataType::UINT32));
    row_partition.setHostOffsets({0, 9, 9});

    const Expression x_expr = Expression::input("x", DataType::FP32, DataType::FP32);
    const Expression w_expr = Expression::input("w", DataType::FP32, DataType::FP32);
    const Expression scale_expr = Expression::input("scale", DataType::FP32, DataType::FP32);
    const Expression offsets_expr = Expression::input("offsets", DataType::UINT32, DataType::UINT32);
    const Expression projected = Expression::matmul(packedExtent(x_expr, offsets_expr, capacity, width),
                                                    w_expr,
                                                    false,
                                                    false,
                                                    DataType::FP32,
                                                    DataType::FP32,
                                                    capacity);
    const Expression normalized = Expression::rmsNorm(packedExtent(projected, offsets_expr, capacity, width),
                                                      scale_expr,
                                                      width,
                                                      1.0e-5,
                                                      DataType::FP32,
                                                      DataType::FP32,
                                                      capacity);

    FusedEquation equation = FusedEquation::compile(Expression::outputs({{"y", normalized}}).physicalOutputs(), 0);
    Stream stream(0);
    StampedExecutionPlan plan = equation.stamp({{"x", x}, {"w", w}, {"scale", scale}, {"offsets", offsets}}, stream);

    auto expect_extent = [&](uint64_t active_rows,
                             uint64_t selected_rows,
                             uint64_t expected_sanitized_bytes_per_consumer,
                             uint64_t expected_full_tail_bytes_per_consumer) {
        row_partition.setHostOffsets({0, active_rows, active_rows});
        const std::vector<PackedRowConsumerDiagnostic> diagnostics = plan.packedRowConsumerDiagnostics();
        ASSERT_EQ(diagnostics.size(), 2u);
        ASSERT_EQ(diagnostics[0].kind, PackedRowConsumerKind::Matmul);
        ASSERT_EQ(diagnostics[1].kind, PackedRowConsumerKind::RmsNorm);
        for (const PackedRowConsumerDiagnostic& diagnostic : diagnostics) {
            EXPECT_EQ(diagnostic.active_rows, active_rows);
            EXPECT_EQ(diagnostic.selected_rows, selected_rows);
            EXPECT_EQ(diagnostic.full_capacity_rows, capacity);
            EXPECT_EQ(diagnostic.sanitized_rows, selected_rows - active_rows);
            EXPECT_EQ(diagnostic.sanitized_operand_count, selected_rows == active_rows ? 0u : 1u);
            EXPECT_EQ(diagnostic.sanitized_bytes, expected_sanitized_bytes_per_consumer);
            EXPECT_EQ(diagnostic.full_tail_bytes, expected_full_tail_bytes_per_consumer);
            EXPECT_LE(diagnostic.sanitized_bytes, diagnostic.full_tail_bytes);
        }
    };

    constexpr uint64_t bytes_per_row = width * sizeof(float);
    expect_extent(/*active_rows=*/0,
                  /*selected_rows=*/8,
                  /*expected_sanitized_bytes_per_consumer=*/8 * bytes_per_row,
                  /*expected_full_tail_bytes_per_consumer=*/100 * bytes_per_row);
    expect_extent(/*active_rows=*/9,
                  /*selected_rows=*/16,
                  /*expected_sanitized_bytes_per_consumer=*/7 * bytes_per_row,
                  /*expected_full_tail_bytes_per_consumer=*/91 * bytes_per_row);
    expect_extent(/*active_rows=*/33,
                  /*selected_rows=*/64,
                  /*expected_sanitized_bytes_per_consumer=*/31 * bytes_per_row,
                  /*expected_full_tail_bytes_per_consumer=*/67 * bytes_per_row);
    expect_extent(/*active_rows=*/64,
                  /*selected_rows=*/64,
                  /*expected_sanitized_bytes_per_consumer=*/0,
                  /*expected_full_tail_bytes_per_consumer=*/36 * bytes_per_row);
}

TEST(RaggedCapacityPerformance, PackedTailSanitizationIsExplicitAndSharedAcrossIndependentConsumers) {
    REQUIRE_CUDA_DEVICE();
    constexpr uint64_t capacity = 100;
    constexpr uint64_t width = 4;

    Tensor x(gpuPlacement, TensorDescriptor(DataType::FP32, {capacity, width}));
    Tensor w(gpuPlacement, TensorDescriptor(DataType::FP32, {width, width}));
    Tensor scale(gpuPlacement, TensorDescriptor(DataType::FP32, {width}));
    Tensor offsets(gpuPlacement, TensorDescriptor(DataType::UINT32, {3}));

    RowPartitionRuntime row_partition(
        offsets, RowPartitionDescriptor(/*batchSize=*/2, capacity, DataType::UINT32));
    row_partition.setHostOffsets({0, 9, 9});

    const Expression x_expr = Expression::input("x", DataType::FP32, DataType::FP32);
    const Expression w_expr = Expression::input("w", DataType::FP32, DataType::FP32);
    const Expression scale_expr = Expression::input("scale", DataType::FP32, DataType::FP32);
    const Expression offsets_expr = Expression::input("offsets", DataType::UINT32, DataType::UINT32);
    const Expression packed_x = packedExtent(x_expr, offsets_expr, capacity, width);
    const Expression projected = Expression::matmul(packed_x,
                                                     w_expr,
                                                     false,
                                                     false,
                                                     DataType::FP32,
                                                     DataType::FP32,
                                                     capacity);
    const Expression normalized = Expression::rmsNorm(packed_x,
                                                       scale_expr,
                                                       width,
                                                       1.0e-5,
                                                       DataType::FP32,
                                                       DataType::FP32,
                                                       capacity);

    FusedEquation equation =
        FusedEquation::compile(Expression::outputs({{"projected", projected}, {"normalized", normalized}}).physicalOutputs(), 0);
    Stream stream(0);
    StampedExecutionPlan plan = equation.stamp({{"x", x}, {"w", w}, {"scale", scale}, {"offsets", offsets}}, stream);

    const std::vector<std::string> kinds = plan.stageKindNames();
    const std::vector<std::vector<uint32_t>> dependencies = plan.stageDependencyIndices();
    ASSERT_EQ(kinds.size(), dependencies.size());
    ASSERT_EQ(std::count(kinds.begin(), kinds.end(), "SanitizePackedTail"), 1);

    const auto sanitize_it = std::find(kinds.begin(), kinds.end(), "SanitizePackedTail");
    const auto matmul_it = std::find(kinds.begin(), kinds.end(), "Matmul");
    const auto rms_norm_it = std::find(kinds.begin(), kinds.end(), "RmsNorm");
    ASSERT_NE(sanitize_it, kinds.end());
    ASSERT_NE(matmul_it, kinds.end());
    ASSERT_NE(rms_norm_it, kinds.end());

    const uint32_t sanitize_idx = static_cast<uint32_t>(std::distance(kinds.begin(), sanitize_it));
    const uint32_t matmul_idx = static_cast<uint32_t>(std::distance(kinds.begin(), matmul_it));
    const uint32_t rms_norm_idx = static_cast<uint32_t>(std::distance(kinds.begin(), rms_norm_it));
    EXPECT_NE(std::find(dependencies[matmul_idx].begin(), dependencies[matmul_idx].end(), sanitize_idx),
              dependencies[matmul_idx].end());
    EXPECT_NE(std::find(dependencies[rms_norm_idx].begin(), dependencies[rms_norm_idx].end(), sanitize_idx),
              dependencies[rms_norm_idx].end());
}

TEST(RaggedCapacityPerformance, CausalConv1dT10RetainedTrainingDoesNotMateriallyRegressAgainstPackedBoundary) {
    if (std::getenv("THOR_T10_RETAINED_RAGGED_TRAINING_GATE") == nullptr) {
        GTEST_SKIP() << "T10 timing qualification runs only through check-retained-ragged-training-production-gate";
    }
    REQUIRE_CUDA_DEVICE();

    constexpr uint64_t batch_size = 4;
    constexpr uint64_t max_total_values = 64;
    constexpr uint64_t max_values_per_row = 16;
    constexpr uint64_t channels = 8;
    constexpr uint64_t kernel_width = 3;
    constexpr size_t warmup_iterations = 12;
    constexpr size_t timing_iterations = 50;
    constexpr size_t timing_rounds = 7;
    constexpr float maximum_relative_regression = 1.15F;
    constexpr float absolute_timing_slack_ms = 0.02F;
    const std::vector<uint32_t> offsets_host{0, 16, 32, 48, 64};
    const std::vector<uint64_t> offsets64(offsets_host.begin(), offsets_host.end());

    Stream stream(0);

    auto make_gpu_float = [&](const std::vector<uint64_t>& dims, const std::vector<float>& values) {
        Tensor cpu(TensorPlacement(TensorPlacement::MemDevices::CPU), TensorDescriptor(DataType::FP32, dims));
        if (cpu.getTotalNumElements() != values.size()) {
            throw std::runtime_error("T10 benchmark float tensor value-count mismatch.");
        }
        float* ptr = cpu.getMemPtr<float>();
        std::copy(values.begin(), values.end(), ptr);
        Tensor gpu(gpuPlacement, TensorDescriptor(DataType::FP32, dims));
        gpu.copyFromAsync(cpu, stream);
        stream.synchronize();
        return gpu;
    };
    auto make_gpu_offsets = [&]() {
        Tensor cpu(TensorPlacement(TensorPlacement::MemDevices::CPU),
                   TensorDescriptor(DataType::UINT32, {batch_size + 1}));
        uint32_t* ptr = cpu.getMemPtr<uint32_t>();
        std::copy(offsets_host.begin(), offsets_host.end(), ptr);
        Tensor gpu(gpuPlacement, TensorDescriptor(DataType::UINT32, {batch_size + 1}));
        gpu.copyFromAsync(cpu, stream);
        stream.synchronize();
        return gpu;
    };

    std::vector<float> x(max_total_values * channels);
    std::vector<float> dy(max_total_values * channels);
    std::vector<float> filter1(channels * channels * kernel_width);
    std::vector<float> filter2(channels * channels * kernel_width);
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = 0.01F * static_cast<float>(static_cast<int>(i % 23) - 7);
        dy[i] = 0.0075F * static_cast<float>(static_cast<int>(i % 19) - 9);
    }
    for (size_t i = 0; i < filter1.size(); ++i) {
        filter1[i] = 0.015F * static_cast<float>(static_cast<int>(i % 17) - 8);
        filter2[i] = 0.0125F * static_cast<float>(static_cast<int>((i * 3) % 19) - 9);
    }

    Tensor gpu_x = make_gpu_float({max_total_values, channels}, x);
    Tensor gpu_dy = make_gpu_float({max_total_values, channels}, dy);
    Tensor gpu_filter1 = make_gpu_float({channels, channels, kernel_width}, filter1);
    Tensor gpu_filter2 = make_gpu_float({channels, channels, kernel_width}, filter2);
    Tensor gpu_offsets = make_gpu_offsets();
    RowPartitionRuntime partition(
        gpu_offsets,
        RowPartitionDescriptor(batch_size, max_total_values, DataType::UINT32, max_values_per_row));
    partition.setHostOffsets(offsets64);

    const RaggedTensorDescriptor descriptor(
        DataType::FP32, {channels}, batch_size, max_total_values, max_values_per_row, DataType::UINT32);

    // Production retained graph: the intermediate activation remains padded in
    // both forward and backward execution.
    const RaggedExpression retained_input = RaggedExpression::input("tokens", descriptor);
    const Expression retained_filter1 = Expression::input("filter1", std::nullopt, DataType::FP32);
    const Expression retained_filter2 = Expression::input("filter2", std::nullopt, DataType::FP32);
    const RaggedExpression retained_hidden =
        retained_input.causalConv1d(retained_filter1, channels, kernel_width, 1, DataType::FP32, DataType::FP32).relu();
    const RaggedExpression retained_output =
        retained_hidden.causalConv1d(retained_filter2, channels, kernel_width, 1, DataType::FP32, DataType::FP32);
    FusedEquation retained_forward =
        FusedEquation::compile(Expression::outputs({{"y", retained_output.getValues()}}).physicalOutputs(), 0);
    StampedExecutionPlan retained_forward_plan = retained_forward.stamp({{"tokens.values", gpu_x},
                                                                         {"tokens.offsets", gpu_offsets},
                                                                         {"filter1", gpu_filter1},
                                                                         {"filter2", gpu_filter2}},
                                                                        stream);
    FusedEquation retained_backward =
        retained_forward.compileBackward({"tokens.values", "filter1", "filter2"}, "dy");
    StampedExecutionPlan retained_backward_plan = retained_backward.stamp({{"tokens.values", gpu_x},
                                                                           {"tokens.offsets", gpu_offsets},
                                                                           {"filter1", gpu_filter1},
                                                                           {"filter2", gpu_filter2},
                                                                           {"dy", gpu_dy}},
                                                                          stream);

    // Explicit packed-boundary baseline: split the same mathematics into two
    // independently stamped equations. The first equation must unpack h and the
    // second must repack it; backward similarly materializes dh between the two
    // equations. This models the pre-retention execution strategy without adding
    // extra mathematical work.
    const RaggedExpression boundary_input = RaggedExpression::input("tokens", descriptor);
    const Expression boundary_filter1 = Expression::input("filter1", std::nullopt, DataType::FP32);
    const RaggedExpression boundary_hidden =
        boundary_input.causalConv1d(boundary_filter1, channels, kernel_width, 1, DataType::FP32, DataType::FP32).relu();
    FusedEquation boundary_forward1 =
        FusedEquation::compile(Expression::outputs({{"h", boundary_hidden.getValues()}}).physicalOutputs(), 0);
    StampedExecutionPlan boundary_forward1_plan = boundary_forward1.stamp({{"tokens.values", gpu_x},
                                                                           {"tokens.offsets", gpu_offsets},
                                                                           {"filter1", gpu_filter1}},
                                                                          stream);

    const RaggedExpression boundary_hidden_input = RaggedExpression::input("hidden", descriptor);
    const Expression boundary_filter2 = Expression::input("filter2", std::nullopt, DataType::FP32);
    const RaggedExpression boundary_output = boundary_hidden_input.causalConv1d(
        boundary_filter2, channels, kernel_width, 1, DataType::FP32, DataType::FP32);
    FusedEquation boundary_forward2 =
        FusedEquation::compile(Expression::outputs({{"y", boundary_output.getValues()}}).physicalOutputs(), 0);
    Tensor boundary_h = boundary_forward1_plan.output("h");
    StampedExecutionPlan boundary_forward2_plan = boundary_forward2.stamp({{"hidden.values", boundary_h},
                                                                           {"hidden.offsets", gpu_offsets},
                                                                           {"filter2", gpu_filter2}},
                                                                          stream);

    FusedEquation boundary_backward2 = boundary_forward2.compileBackward({"hidden.values", "filter2"}, "dy");
    StampedExecutionPlan boundary_backward2_plan = boundary_backward2.stamp({{"hidden.values", boundary_h},
                                                                             {"hidden.offsets", gpu_offsets},
                                                                             {"filter2", gpu_filter2},
                                                                             {"dy", gpu_dy}},
                                                                            stream);
    Tensor boundary_dh = boundary_backward2_plan.output("hidden.values_grad");
    FusedEquation boundary_backward1 = boundary_forward1.compileBackward({"tokens.values", "filter1"}, "dh");
    StampedExecutionPlan boundary_backward1_plan = boundary_backward1.stamp({{"tokens.values", gpu_x},
                                                                             {"tokens.offsets", gpu_offsets},
                                                                             {"filter1", gpu_filter1},
                                                                             {"dh", boundary_dh}},
                                                                            stream);

    // Establish that the explicit-boundary comparison really is the same
    // training computation before treating it as a performance baseline.
    retained_forward_plan.run();
    retained_backward_plan.run();
    boundary_forward1_plan.run();
    boundary_forward2_plan.run();
    boundary_backward2_plan.run();
    boundary_backward1_plan.run();
    stream.synchronize();

    auto copy_gpu_float = [&](const Tensor& gpu) {
        Tensor cpu = gpu.clone(TensorPlacement(TensorPlacement::MemDevices::CPU));
        cpu.copyFromAsync(gpu, stream);
        stream.synchronize();
        const float* ptr = cpu.getMemPtr<float>();
        return std::vector<float>(ptr, ptr + cpu.getTotalNumElements());
    };
    auto expect_equivalent = [&](const Tensor& retained, const Tensor& boundary, const char* label) {
        const std::vector<float> retained_values = copy_gpu_float(retained);
        const std::vector<float> boundary_values = copy_gpu_float(boundary);
        ASSERT_EQ(retained_values.size(), boundary_values.size()) << label;
        for (size_t i = 0; i < retained_values.size(); ++i) {
            EXPECT_NEAR(retained_values[i], boundary_values[i], 2.0e-4F) << label << " index " << i;
        }
    };
    expect_equivalent(retained_forward_plan.output("y"), boundary_forward2_plan.output("y"), "forward y");
    expect_equivalent(retained_backward_plan.output("tokens.values_grad"),
                      boundary_backward1_plan.output("tokens.values_grad"),
                      "dX");
    expect_equivalent(retained_backward_plan.output("filter1_grad"),
                      boundary_backward1_plan.output("filter1_grad"),
                      "dW1");
    expect_equivalent(retained_backward_plan.output("filter2_grad"),
                      boundary_backward2_plan.output("filter2_grad"),
                      "dW2");

    EXPECT_EQ(retained_forward_plan.stageKindNames(),
              (std::vector<std::string>{"PaddedRaggedPack",
                                        "SanitizePaddedRaggedTail",
                                        "RaggedConv1dCausal",
                                        "PaddedRaggedPointwise",
                                        "SanitizePaddedRaggedTail",
                                        "RaggedConv1dCausal",
                                        "PaddedRaggedUnpack"}));
    const std::vector<std::string> retained_backward_stage_names = retained_backward_plan.stageKindNames();
    const std::vector<std::string> boundary_forward1_stage_names = boundary_forward1_plan.stageKindNames();
    const std::vector<std::string> boundary_forward2_stage_names = boundary_forward2_plan.stageKindNames();
    EXPECT_EQ(std::count(retained_backward_stage_names.begin(),
                         retained_backward_stage_names.end(),
                         "PaddedRaggedUnpack"),
              1);
    EXPECT_EQ(std::count(boundary_forward1_stage_names.begin(),
                         boundary_forward1_stage_names.end(),
                         "PaddedRaggedUnpack"),
              1);
    EXPECT_EQ(std::count(boundary_forward2_stage_names.begin(),
                         boundary_forward2_stage_names.end(),
                         "PaddedRaggedPack"),
              1);

    auto run_retained = [&]() {
        retained_forward_plan.run();
        retained_backward_plan.run();
    };
    auto run_boundary = [&]() {
        boundary_forward1_plan.run();
        boundary_forward2_plan.run();
        boundary_backward2_plan.run();
        boundary_backward1_plan.run();
    };

    for (size_t i = 0; i < warmup_iterations; ++i) {
        run_retained();
        run_boundary();
    }
    stream.synchronize();

    auto time_sequence = [&](const std::function<void()>& sequence) {
        Event start = stream.putEvent(true);
        for (size_t i = 0; i < timing_iterations; ++i) {
            sequence();
        }
        Event stop = stream.putEvent(true);
        return stop.synchronizeAndReportElapsedTimeInMilliseconds(start) / static_cast<float>(timing_iterations);
    };
    std::vector<float> retained_samples;
    std::vector<float> boundary_samples;
    retained_samples.reserve(timing_rounds);
    boundary_samples.reserve(timing_rounds);
    for (size_t round = 0; round < timing_rounds; ++round) {
        if ((round & 1U) == 0U) {
            retained_samples.push_back(time_sequence(run_retained));
            boundary_samples.push_back(time_sequence(run_boundary));
        } else {
            boundary_samples.push_back(time_sequence(run_boundary));
            retained_samples.push_back(time_sequence(run_retained));
        }
    }
    auto median = [](std::vector<float> samples) {
        std::sort(samples.begin(), samples.end());
        return samples[samples.size() / 2];
    };
    const float retained_ms = median(retained_samples);
    const float boundary_ms = median(boundary_samples);

    EXPECT_LE(retained_ms, boundary_ms * maximum_relative_regression + absolute_timing_slack_ms)
        << "retained ragged training materially regressed against an explicit packed-boundary baseline: retained="
        << retained_ms << " ms, packed-boundary=" << boundary_ms << " ms";

    for (const RaggedConv1dStageDiagnostic& diagnostic : retained_forward_plan.raggedConv1dStageDiagnostics()) {
        EXPECT_EQ(diagnostic.explicit_unfold_workspace_bytes, 0u);
    }
    for (const RaggedConv1dStageDiagnostic& diagnostic : retained_backward_plan.raggedConv1dStageDiagnostics()) {
        EXPECT_EQ(diagnostic.explicit_unfold_workspace_bytes, 0u);
    }
}

TEST(RaggedCapacityPerformance, RaggedFusedValuewiseFlopsUseRuntimeActiveValues) {
    REQUIRE_CUDA_DEVICE();
    constexpr uint64_t batch_size = 2;
    constexpr uint64_t capacity = 16;
    constexpr uint64_t width = 4;

    Tensor x(gpuPlacement, TensorDescriptor(DataType::FP32, {capacity, width}));
    Tensor offsets(gpuPlacement, TensorDescriptor(DataType::UINT32, {batch_size + 1}));
    RowPartitionRuntime partition(offsets, RowPartitionDescriptor(batch_size, capacity, DataType::UINT32));
    partition.setHostOffsets({0, 2, 5});

    const Expression x_expr = Expression::input("x", DataType::FP32, DataType::FP32);
    const Expression offsets_expr = Expression::input("offsets", DataType::UINT32, DataType::UINT32);
    const Expression active_x = packedExtent(x_expr, offsets_expr, capacity, width);
    const Expression y = active_x.abs();

    FusedEquation equation = FusedEquation::compile(Expression::outputs({{"y", y}}).physicalOutputs(), 0);
    Stream stream(0);
    StampedExecutionPlan plan = equation.stamp({{"x", x}, {"offsets", offsets}}, stream);

    EXPECT_EQ(plan.flopCount(), 5u * width);

    partition.setHostOffsets({0, 0, 2});
    EXPECT_EQ(plan.flopCount(), 2u * width);

    partition.setHostOffsets({0, 0, 0});
    EXPECT_EQ(plan.flopCount(), 0u);
}

TEST(RaggedCapacityPerformance, RaggedFusedValuewiseFlopsKeepDenseSideComputationFixed) {
    REQUIRE_CUDA_DEVICE();
    constexpr uint64_t batch_size = 2;
    constexpr uint64_t capacity = 16;
    constexpr uint64_t width = 4;

    Tensor x(gpuPlacement, TensorDescriptor(DataType::FP32, {capacity, width}));
    Tensor scale(gpuPlacement, TensorDescriptor(DataType::FP32, {width}));
    Tensor offsets(gpuPlacement, TensorDescriptor(DataType::UINT32, {batch_size + 1}));
    RowPartitionRuntime partition(offsets, RowPartitionDescriptor(batch_size, capacity, DataType::UINT32));
    partition.setHostOffsets({0, 2, 5});

    const Expression x_expr = Expression::input("x", DataType::FP32, DataType::FP32);
    const Expression scale_expr = Expression::input("scale", DataType::FP32, DataType::FP32);
    const Expression offsets_expr = Expression::input("offsets", DataType::UINT32, DataType::UINT32);
    const Expression active_x = packedExtent(x_expr, offsets_expr, capacity, width);
    const Expression y = active_x * scale_expr.exp();

    FusedEquation equation = FusedEquation::compile(Expression::outputs({{"y", y}}).physicalOutputs(), 0);
    Stream stream(0);
    StampedExecutionPlan plan = equation.stamp({{"x", x}, {"scale", scale}, {"offsets", offsets}}, stream);

    // EXP(scale) is one dense side computation (10 FLOPs per channel) while
    // the multiply scales with the logical packed population.
    EXPECT_EQ(plan.flopCount(), 5u * width + 10u * width);

    partition.setHostOffsets({0, 0, 2});
    EXPECT_EQ(plan.flopCount(), 2u * width + 10u * width);

    // With no logical ragged values, the fused operation has no useful output
    // work, so even fixed side expressions contribute zero useful FLOPs.
    partition.setHostOffsets({0, 0, 0});
    EXPECT_EQ(plan.flopCount(), 0u);
}

TEST(RaggedCapacityPerformance, RaggedSegmentedMeanFlopsUseRuntimeActiveValuesAndNonemptyRows) {
    REQUIRE_CUDA_DEVICE();
    constexpr uint64_t batch_size = 2;
    constexpr uint64_t capacity = 16;
    constexpr uint64_t width = 4;

    Tensor values(gpuPlacement, TensorDescriptor(DataType::FP32, {capacity, width}));
    Tensor offsets(gpuPlacement, TensorDescriptor(DataType::UINT32, {batch_size + 1}));
    RowPartitionRuntime partition(offsets, RowPartitionDescriptor(batch_size, capacity, DataType::UINT32));
    partition.setHostOffsets({0, 2, 5});

    const RaggedTensorDescriptor descriptor(
        DataType::FP32, {width}, batch_size, capacity, capacity, DataType::UINT32);
    const RaggedExpression input = RaggedExpression::input("tokens", descriptor);
    const Expression mean = input.segment_mean();

    FusedEquation equation = FusedEquation::compile(Expression::outputs({{"mean", mean}}).physicalOutputs(), 0);
    Stream stream(0);
    StampedExecutionPlan plan = equation.stamp(
        {{"tokens.values", values}, {"tokens.offsets", offsets}}, stream);

    // Five useful packed values contribute four channel reductions each, plus
    // one mean division per channel for each of the two non-empty rows.
    EXPECT_EQ(plan.flopCount(), 5u * width + 2u * width);

    partition.setHostOffsets({0, 0, 2});
    EXPECT_EQ(plan.flopCount(), 2u * width + 1u * width);

    partition.setHostOffsets({0, 0, 0});
    EXPECT_EQ(plan.flopCount(), 0u);
}

TEST(RaggedCapacityPerformance, SegmentedScanFlopsUseRuntimeActiveValues) {
    REQUIRE_CUDA_DEVICE();
    constexpr uint64_t batch_size = 2;
    constexpr uint64_t capacity = 16;

    Tensor values(gpuPlacement, TensorDescriptor(DataType::FP32, {capacity}));
    Tensor offsets(gpuPlacement, TensorDescriptor(DataType::UINT32, {batch_size + 1}));
    RowPartitionRuntime partition(offsets, RowPartitionDescriptor(batch_size, capacity, DataType::UINT32));
    partition.setHostOffsets({0, 2, 5});

    const Expression values_expr = Expression::input("values", DataType::FP32, DataType::FP32);
    const Expression offsets_expr = Expression::input("offsets", DataType::UINT32, DataType::UINT32);
    const Expression scan = Expression::segmentedScan(values_expr, offsets_expr, ScanOp::Sum, true, false);

    FusedEquation equation = FusedEquation::compile(Expression::outputs({{"scan", scan}}).physicalOutputs(), 0);
    Stream stream(0);
    StampedExecutionPlan plan = equation.stamp({{"values", values}, {"offsets", offsets}}, stream);

    EXPECT_EQ(plan.flopCount(), 5u);

    partition.setHostOffsets({0, 0, 2});
    EXPECT_EQ(plan.flopCount(), 2u);

    partition.setHostOffsets({0, 0, 0});
    EXPECT_EQ(plan.flopCount(), 0u);
}

TEST(RaggedCapacityPerformance, RetainedPaddedRaggedPointwiseFlopsUseRuntimeActiveValues) {
    REQUIRE_CUDA_DEVICE();
    constexpr uint64_t batch_size = 2;
    constexpr uint64_t capacity = 16;
    constexpr uint64_t max_values_per_row = 8;
    constexpr uint64_t channels = 4;
    constexpr uint64_t kernel_width = 1;

    Tensor values(gpuPlacement, TensorDescriptor(DataType::FP32, {capacity, channels}));
    Tensor offsets(gpuPlacement, TensorDescriptor(DataType::UINT32, {batch_size + 1}));
    Tensor filter1(gpuPlacement, TensorDescriptor(DataType::FP32, {channels, channels, kernel_width}));
    Tensor filter2(gpuPlacement, TensorDescriptor(DataType::FP32, {channels, channels, kernel_width}));
    RowPartitionRuntime partition(
        offsets,
        RowPartitionDescriptor(batch_size, capacity, DataType::UINT32, max_values_per_row));
    partition.setHostOffsets({0, 2, 5});

    const RaggedTensorDescriptor descriptor(
        DataType::FP32, {channels}, batch_size, capacity, max_values_per_row, DataType::UINT32);
    const RaggedExpression input = RaggedExpression::input("tokens", descriptor);
    const Expression filter1_expr = Expression::input("filter1", std::nullopt, DataType::FP32);
    const Expression filter2_expr = Expression::input("filter2", std::nullopt, DataType::FP32);
    const RaggedExpression hidden =
        input.causalConv1d(filter1_expr, channels, kernel_width, 1, DataType::FP32, DataType::FP32).relu();
    const RaggedExpression output =
        hidden.causalConv1d(filter2_expr, channels, kernel_width, 1, DataType::FP32, DataType::FP32);

    FusedEquation equation = FusedEquation::compile(Expression::outputs({{"y", output.getValues()}}).physicalOutputs(), 0);
    Stream stream(0);
    StampedExecutionPlan plan = equation.stamp({{"tokens.values", values},
                                                {"tokens.offsets", offsets},
                                                {"filter1", filter1},
                                                {"filter2", filter2}},
                                               stream);

    ASSERT_EQ(plan.stageKindNames(),
              (std::vector<std::string>{"PaddedRaggedPack",
                                        "SanitizePaddedRaggedTail",
                                        "RaggedConv1dCausal",
                                        "PaddedRaggedPointwise",
                                        "SanitizePaddedRaggedTail",
                                        "RaggedConv1dCausal",
                                        "PaddedRaggedUnpack"}));

    // Each 1x1 convolution uses 2*C*C FLOPs per logical row. ReLU is one
    // useful comparison per channel; pack/unpack/sanitation are data movement, not FLOPs.
    EXPECT_EQ(plan.stageFlopCounts(), (std::vector<uint64_t>{0, 0, 5u * 32u, 5u * channels, 0, 5u * 32u, 0}));

    partition.setHostOffsets({0, 0, 2});
    EXPECT_EQ(plan.stageFlopCounts(), (std::vector<uint64_t>{0, 0, 2u * 32u, 2u * channels, 0, 2u * 32u, 0}));

    partition.setHostOffsets({0, 0, 0});
    EXPECT_EQ(plan.stageFlopCounts(), (std::vector<uint64_t>{0, 0, 0, 0, 0, 0, 0}));
}

TEST(RaggedCapacityPerformance, RaggedNormalizationFlopsUseRuntimeActiveValues) {
    REQUIRE_CUDA_DEVICE();
    constexpr uint64_t batch_size = 2;
    constexpr uint64_t capacity = 16;
    constexpr uint64_t width = 4;

    Tensor values(gpuPlacement, TensorDescriptor(DataType::FP32, {capacity, width}));
    Tensor offsets(gpuPlacement, TensorDescriptor(DataType::UINT32, {batch_size + 1}));
    Tensor scale(gpuPlacement, TensorDescriptor(DataType::FP32, {width}));
    Tensor bias(gpuPlacement, TensorDescriptor(DataType::FP32, {width}));
    Tensor dy(gpuPlacement, TensorDescriptor(DataType::FP32, {capacity, width}));
    RowPartitionRuntime partition(offsets, RowPartitionDescriptor(batch_size, capacity, DataType::UINT32));
    partition.setHostOffsets({0, 2, 5});

    const RaggedTensorDescriptor descriptor(
        DataType::FP32, {width}, batch_size, capacity, capacity, DataType::UINT32);
    const RaggedExpression input = RaggedExpression::input("tokens", descriptor);
    const Expression scale_expr = Expression::input("scale", std::nullopt, DataType::FP32);
    const Expression bias_expr = Expression::input("bias", std::nullopt, DataType::FP32);

    FusedEquation rms_equation = FusedEquation::compile(
        Expression::outputs({{"y", input.rmsNorm(scale_expr, 1.0e-5, DataType::FP32, DataType::FP32).getValues()}})
            .physicalOutputs(),
        0);
    FusedEquation layer_equation = FusedEquation::compile(
        Expression::outputs(
            {{"y", input.layerNorm(scale_expr, bias_expr, 1.0e-5, DataType::FP32, DataType::FP32).getValues()}})
            .physicalOutputs(),
        0);

    Stream stream(0);
    StampedExecutionPlan layer_plan = layer_equation.stamp(
        {{"tokens.values", values}, {"tokens.offsets", offsets}, {"scale", scale}, {"bias", bias}}, stream);
    FusedEquation rms_backward_equation = rms_equation.compileBackward({"tokens.values", "scale"}, "dy");
    ASSERT_TRUE(rms_backward_equation.requiresForwardExecutionForBackward());
    // RMSNorm backward consumes state retained by its matching forward plan.
    // Pair the plans even though this test only queries FLOPs without running them.
    auto [rms_plan, rms_backward_plan] = rms_backward_equation.stampForwardBackwardPair(
        {{"tokens.values", values}, {"tokens.offsets", offsets}, {"scale", scale}, {"dy", dy}}, stream);

    EXPECT_EQ(rms_plan.flopCount(), 5u * width * 6u);
    EXPECT_EQ(layer_plan.flopCount(), 5u * width * 8u);
    EXPECT_EQ(rms_backward_plan.flopCount(), 5u * width * 12u);

    partition.setHostOffsets({0, 0, 2});
    EXPECT_EQ(rms_plan.flopCount(), 2u * width * 6u);
    EXPECT_EQ(layer_plan.flopCount(), 2u * width * 8u);
    EXPECT_EQ(rms_backward_plan.flopCount(), 2u * width * 12u);

    partition.setHostOffsets({0, 0, 0});
    EXPECT_EQ(rms_plan.flopCount(), 0u);
    EXPECT_EQ(layer_plan.flopCount(), 0u);
    EXPECT_EQ(rms_backward_plan.flopCount(), 0u);
}

TEST(RaggedCapacityPerformance, Lwa4c2RaggedAttentionBackwardLogicalWorkUsesRequestedRoutesAndActivePrefix) {
    REQUIRE_CUDA_DEVICE();
    constexpr uint64_t batch_size = 2;
    constexpr uint64_t heads = 1;
    constexpr uint64_t head_dim = 8;
    constexpr uint64_t valid_rows = 1;
    constexpr uint64_t bytes_per_token = heads * head_dim * sizeof(uint16_t);

    auto logical_work_for_capacity = [&](uint64_t capacity) {
        Tensor q(gpuPlacement, TensorDescriptor(DataType::FP16, {capacity, heads, head_dim}));
        Tensor k(gpuPlacement, TensorDescriptor(DataType::FP16, {capacity, heads, head_dim}));
        Tensor v(gpuPlacement, TensorDescriptor(DataType::FP16, {capacity, heads, head_dim}));
        Tensor dy(gpuPlacement, TensorDescriptor(DataType::FP16, {capacity, heads, head_dim}));
        Tensor q_offsets(gpuPlacement, TensorDescriptor(DataType::UINT32, {batch_size + 1}));
        Tensor kv_offsets(gpuPlacement, TensorDescriptor(DataType::UINT32, {batch_size + 1}));

        // Keep device offset payloads stale. Logical-work queries must consume
        // only authoritative host publications and must never synchronize to
        // inspect these device tensors.
        RowPartitionRuntime q_partition(
            q_offsets, RowPartitionDescriptor(batch_size, capacity, DataType::UINT32));
        RowPartitionRuntime kv_partition(
            kv_offsets, RowPartitionDescriptor(batch_size, capacity, DataType::UINT32));
        q_partition.setHostOffsets({0, 2, 5});   // lengths [2, 3]
        kv_partition.setHostOffsets({0, 4, 5});  // lengths [4, 1]

        const Expression q_expr = Expression::input("q", DataType::FP32, DataType::FP16);
        const Expression k_expr = Expression::input("k", DataType::FP32, DataType::FP16);
        const Expression v_expr = Expression::input("v", DataType::FP32, DataType::FP16);
        const Expression q_offsets_expr = Expression::input("q_offsets", DataType::UINT32, DataType::UINT32);
        const Expression kv_offsets_expr = Expression::input("kv_offsets", DataType::UINT32, DataType::UINT32);

        AttentionOptions options;
        options.q_layout = AttentionTensorLayout::BSHD;
        options.k_layout = AttentionTensorLayout::BSHD;
        options.v_layout = AttentionTensorLayout::BSHD;
        options.o_layout = AttentionTensorLayout::BSHD;
        options.compute_dtype = DataType::FP32;
        options.output_dtype = DataType::FP16;

        const Expression attention = Expression::scaledDotProductAttentionRagged(
            q_expr, k_expr, v_expr, q_offsets_expr, kv_offsets_expr, options);
        FusedEquation forward =
            FusedEquation::compile(Expression::outputs({{"y", attention}}).physicalOutputs(), 0);

        auto backward_work = [&](const std::vector<std::string>& wrt) {
            FusedEquation backward = forward.compileBackward(wrt, "dy");
            Stream stream(0);
            auto [forward_plan, backward_plan] = backward.stampForwardBackwardPair(
                {{"q", q},
                 {"k", k},
                 {"v", v},
                 {"q_offsets", q_offsets},
                 {"kv_offsets", kv_offsets},
                 {"dy", dy}},
                stream);
            return std::array<uint64_t, 4>{backward_plan.logicalFlopCount(),
                                           backward_plan.logicalByteCount(),
                                           backward_plan.logicalFlopCount(valid_rows),
                                           backward_plan.logicalByteCount(valid_rows)};
        };

        // Per score pair: dP=dO*V^T is 2*Dv and softmax backward is 5.
        // dQ and dK each add 2*Dqk; dV adds 2*Dv. Shared score-gradient
        // work must count once when both dQ and dK are authored.
        constexpr uint64_t score_gradient_flops = (2 * head_dim) + 5;
        constexpr uint64_t qk_gradient_flops = 2 * head_dim;
        constexpr uint64_t v_gradient_flops = 2 * head_dim;
        constexpr uint64_t full_score_pairs = 11;    // 2*4 + 3*1
        constexpr uint64_t prefix_score_pairs = 8;   // first row: 2*4

        const auto dq = backward_work({"q"});
        EXPECT_EQ(dq[0], full_score_pairs * heads * (score_gradient_flops + qk_gradient_flops));
        EXPECT_EQ(dq[1], (5u + 5u + 5u + 5u + 5u) * bytes_per_token);  // q/k/v/dO reads + dQ write
        EXPECT_EQ(dq[2], prefix_score_pairs * heads * (score_gradient_flops + qk_gradient_flops));
        EXPECT_EQ(dq[3], (2u + 4u + 4u + 2u + 2u) * bytes_per_token);

        const auto dv = backward_work({"v"});
        EXPECT_EQ(dv[0], full_score_pairs * heads * v_gradient_flops);
        EXPECT_EQ(dv[1], (5u + 5u + 5u + 5u + 5u) * bytes_per_token);  // q/k/v/dO reads + dV write
        EXPECT_EQ(dv[2], prefix_score_pairs * heads * v_gradient_flops);
        EXPECT_EQ(dv[3], (2u + 4u + 4u + 2u + 4u) * bytes_per_token);

        const auto all = backward_work({"q", "k", "v"});
        constexpr uint64_t all_gradient_flops =
            score_gradient_flops + qk_gradient_flops + qk_gradient_flops + v_gradient_flops;
        EXPECT_EQ(all[0], full_score_pairs * heads * all_gradient_flops);
        EXPECT_EQ(all[1], (5u + 5u + 5u + 5u + 5u + 5u + 5u) * bytes_per_token);
        EXPECT_EQ(all[2], prefix_score_pairs * heads * all_gradient_flops);
        EXPECT_EQ(all[3], (2u + 4u + 4u + 2u + 2u + 4u + 4u) * bytes_per_token);

        // Same token totals, different row geometry: bytes stay identical while
        // exact score-pair work changes from 11 to 17. This also proves the
        // backward path reuses the cached Q/K prefix geometry rather than a
        // capacity-based estimate.
        q_partition.setHostOffsets({0, 1, 5});
        kv_partition.setHostOffsets({0, 1, 5});
        const auto all_redistributed = backward_work({"q", "k", "v"});
        EXPECT_EQ(all_redistributed[0], 17u * heads * all_gradient_flops);
        EXPECT_EQ(all_redistributed[1], all[1]);
        EXPECT_EQ(all_redistributed[2], 1u * heads * all_gradient_flops);
        EXPECT_EQ(all_redistributed[3], (1u + 1u + 1u + 1u + 1u + 1u + 1u) * bytes_per_token);

        q_partition.setHostOffsets({0, 0, 0});
        kv_partition.setHostOffsets({0, 0, 0});
        const auto all_empty = backward_work({"q", "k", "v"});
        EXPECT_EQ(all_empty, (std::array<uint64_t, 4>{0u, 0u, 0u, 0u}));

        return std::array<uint64_t, 12>{dq[0], dq[1], dq[2], dq[3],
                                        dv[0], dv[1], dv[2], dv[3],
                                        all[0], all[1], all[2], all[3]};
    };

    EXPECT_EQ(logical_work_for_capacity(8), logical_work_for_capacity(16));
}
