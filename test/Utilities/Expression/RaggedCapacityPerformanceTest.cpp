#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/FusedEquation.h"

#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <memory>
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

}  // namespace

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

    // FC -> SwiGLU -> RMSNorm. The GLU work stays active-aware (or may be fused);
    // there is no physical helper/tail-clear stage between the bucketed MATMUL
    // and RMSNorm consumers.
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
    row_partition.setHostActiveValueCount(9);

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
        row_partition.setHostActiveValueCount(active_rows);
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
