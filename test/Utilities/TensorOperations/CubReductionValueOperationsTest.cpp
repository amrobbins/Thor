#include "test/Utilities/TensorOperations/CubReductionTestSupport.h"

#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace ThorImplementation;
using namespace ThorImplementation::CubReductionTestSupport;

namespace {

void expectStageSemantics(CubReductionOp op,
                          CubReductionInternal::CubReductionStageRole role,
                          uint64_t total_reduction_size,
                          CubReductionInternal::CubReductionStageInputTransform input_transform,
                          CubReductionInternal::CubReductionStageCombine combine,
                          CubReductionInternal::CubReductionStageFinalize finalize,
                          uint64_t finalize_divisor = 1) {
    const CubReductionInternal::CubReductionStageSemantics semantics =
        CubReductionInternal::makeValueReductionStageSemantics(op, role, total_reduction_size);
    EXPECT_EQ(semantics.input_transform, input_transform);
    EXPECT_EQ(semantics.combine, combine);
    EXPECT_EQ(semantics.finalize, finalize);
    EXPECT_EQ(semantics.finalize_divisor, finalize_divisor);
}

}  // namespace


#if THOR_CUB_ENABLE_FP8_TYPES
// The retained FP8 RK selector must use the same public VALUE operation and
// floating-output surface as ordinary production reducers. These inputs are
// exactly representable in both FP8 formats: every reduction domain has one
// -1 and all remaining elements are +1. This distinguishes the four combines,
// both input transforms, division and square-root finalization, while keeping
// PRODUCT finite and avoiding FP8 input-rounding differences in the oracle.
TEST(CubReduction, RetainedFp8RKProductionOperationAndOutputDTypeMatrix) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const std::array<CubReductionOp, 8> operations = {
        CubReductionOp::Sum, CubReductionOp::Mean, CubReductionOp::Product,
        CubReductionOp::Min, CubReductionOp::Max, CubReductionOp::L1Norm,
        CubReductionOp::L2Norm, CubReductionOp::SumSquares};
    std::vector<DataType> output_dtypes = {
        DataType::FP8_E4M3, DataType::FP8_E5M2, DataType::FP16,
        DataType::BF16, DataType::FP32};
#if THOR_CUB_ENABLE_64BIT_TYPES
    output_dtypes.push_back(DataType::FP64);
#endif
    // The full operation x output-format matrix runs for a single RK site.
    // Multi-axis cases additionally prove that finalization and conversion
    // occur only after the independent downstream mathematical reduction.
    for (bool multi_axis : {false, true}) {
        const uint64_t outer = 2, r = 16, k = 32, tail = multi_axis ? 2 : 1;
        const uint64_t reduced_count = r * tail;
        const std::vector<uint64_t> input_dims = multi_axis
            ? std::vector<uint64_t>{outer, r, k, tail}
            : std::vector<uint64_t>{outer, r, k};
        const std::vector<uint64_t> output_dims = multi_axis
            ? std::vector<uint64_t>{outer, 1, k, 1}
            : std::vector<uint64_t>{outer, 1, k};
        const std::vector<uint32_t> axes = multi_axis
            ? std::vector<uint32_t>{1, 3} : std::vector<uint32_t>{1};
        std::vector<float> inputs(outer * r * k * tail, 1.0f);
        for (uint64_t o = 0; o < outer; ++o)
            for (uint64_t component = 0; component < k; ++component)
                inputs[(o * r * k + component) * tail] = -1.0f;

        const auto unscaled_reference = [&](CubReductionOp op) {
            switch (op) {
                case CubReductionOp::Sum: return static_cast<float>(reduced_count - 2);
                case CubReductionOp::Mean:
                    return static_cast<float>(reduced_count - 2) / static_cast<float>(reduced_count);
                case CubReductionOp::Product: return -1.0f;
                case CubReductionOp::Min: return -1.0f;
                case CubReductionOp::Max: return 1.0f;
                case CubReductionOp::L1Norm: return static_cast<float>(reduced_count);
                case CubReductionOp::L2Norm: return std::sqrt(static_cast<float>(reduced_count));
                case CubReductionOp::SumSquares: return static_cast<float>(reduced_count);
            }
            throw std::logic_error("Unknown production VALUE operation");
        };

        for (DataType input_dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
            SCOPED_TRACE(static_cast<int>(input_dtype));
            Tensor input = makeGpuTensor(inputs, input_dims, stream, input_dtype);
            for (CubReductionOp op : operations) {
                SCOPED_TRACE(static_cast<int>(op));
                for (DataType output_dtype : output_dtypes) {
                    // Multi-axis FP32 is enough to validate finalizer placement;
                    // the single-axis matrix covers all output conversions.
                    if (multi_axis && output_dtype != DataType::FP32) continue;
                    SCOPED_TRACE(static_cast<int>(output_dtype));
                    SCOPED_TRACE(multi_axis);
                    Tensor output(gpuPlacement, TensorDescriptor(output_dtype, output_dims));
                    const CubReduction reduction(op, axes, output_dtype, 0.5f);
                    auto stamp = reduction.stamp(input, output, stream);
                    ASSERT_TRUE(stamp->usesDenseExecutablePlan());
                    const auto passes = stamp->getDensePhysicalPassDiagnostics();
                    ASSERT_FALSE(passes.empty());
                    // For a single RK site, all operations must take the retained
                    // selector instead of silently reverting to legacy FP8 RK.
                    if (!multi_axis)
                        EXPECT_EQ(passes.front().family.find("retained_fp8_rk:"), 0U);
                    size_t workspace = 0;
                    for (size_t i = 0; i < passes.size(); ++i) {
                        workspace += passes[i].intermediate_bytes + passes[i].scratch_bytes;
                        if (passes[i].intermediate_bytes)
                            EXPECT_EQ(passes[i].output_dtype, DataType::FP32);
                    }
                    EXPECT_EQ(workspace, stamp->getWorkspaceSizeInBytes());
                    EXPECT_EQ(reduction.queryWorkspaceSizeInBytes(input.getDescriptor(), stream),
                              stamp->getWorkspaceSizeInBytes());
                    for (float scale : {0.5f, -1.0f}) {
                        if (scale == 0.5f) stamp->runOn(stream);
                        else stamp->runOn(stream, scale);
                        const float expected_value = unscaled_reference(op) * scale;
                        // Quantize the reference to precisely the requested
                        // destination dtype (including FP8 satfinite behavior).
                        const std::vector<float> expected_values(outer * k, expected_value);
                        Tensor reference = makeGpuTensor(expected_values, output_dims, stream, output_dtype);
                        expectFloatVectorNear(copyGpuTensorAsFloat(output, stream),
                                              copyGpuTensorAsFloat(reference, stream), 0.0625f);
                    }
                }
            }
        }
    }
}

// With no explicit dtype, ordinary CubReduction must store in the input
// dtype. This was rejected by the temporary FP32-only FP8 RK gate.
TEST(CubReduction, RetainedFp8RKDefaultOutputDTypeMatchesInput) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    for (DataType dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
        SCOPED_TRACE(static_cast<int>(dtype));
        const std::vector<uint64_t> input_dims{2, 16, 32};
        const std::vector<uint64_t> output_dims{2, 1, 32};
        Tensor input = makeGpuTensor(std::vector<float>(2 * 16 * 32, 0.5f),
                                     input_dims, stream, dtype);
        Tensor output(gpuPlacement, TensorDescriptor(dtype, output_dims));
        const CubReduction reduction(CubReductionOp::Sum, std::vector<uint32_t>{1});
        auto stamp = reduction.stamp(input, output, stream);
        ASSERT_TRUE(stamp->usesDenseExecutablePlan());
        ASSERT_FALSE(stamp->getDensePhysicalPassDiagnostics().empty());
        EXPECT_EQ(stamp->getDensePhysicalPassDiagnostics().front().family.find("retained_fp8_rk:"), 0U);
        stamp->runOn(stream);
        expectFloatVectorNear(copyGpuTensorAsFloat(output, stream),
                              std::vector<float>(2 * 32, 8.0f), 0.0f);
    }
}
#endif // THOR_CUB_ENABLE_FP8_TYPES

TEST(CubReduction, ValueStageSemanticsSeparateFirstTransformCombineAndFinalization) {
    using CubReductionInternal::CubReductionStageCombine;
    using CubReductionInternal::CubReductionStageFinalize;
    using CubReductionInternal::CubReductionStageInputTransform;
    using CubReductionInternal::CubReductionStageRole;

    constexpr uint64_t total_reduction_size = 105;

    for (CubReductionStageRole role : {CubReductionStageRole::Complete,
                                       CubReductionStageRole::First,
                                       CubReductionStageRole::Intermediate,
                                       CubReductionStageRole::Final}) {
        expectStageSemantics(CubReductionOp::Sum,
                             role,
                             total_reduction_size,
                             CubReductionStageInputTransform::Identity,
                             CubReductionStageCombine::Sum,
                             CubReductionStageFinalize::Identity);
        expectStageSemantics(CubReductionOp::Product,
                             role,
                             total_reduction_size,
                             CubReductionStageInputTransform::Identity,
                             CubReductionStageCombine::Product,
                             CubReductionStageFinalize::Identity);
        expectStageSemantics(CubReductionOp::Min,
                             role,
                             total_reduction_size,
                             CubReductionStageInputTransform::Identity,
                             CubReductionStageCombine::Minimum,
                             CubReductionStageFinalize::Identity);
        expectStageSemantics(CubReductionOp::Max,
                             role,
                             total_reduction_size,
                             CubReductionStageInputTransform::Identity,
                             CubReductionStageCombine::Maximum,
                             CubReductionStageFinalize::Identity);
    }

    expectStageSemantics(CubReductionOp::Mean,
                         CubReductionStageRole::Complete,
                         total_reduction_size,
                         CubReductionStageInputTransform::Identity,
                         CubReductionStageCombine::Sum,
                         CubReductionStageFinalize::Divide,
                         total_reduction_size);
    expectStageSemantics(CubReductionOp::Mean,
                         CubReductionStageRole::First,
                         total_reduction_size,
                         CubReductionStageInputTransform::Identity,
                         CubReductionStageCombine::Sum,
                         CubReductionStageFinalize::Identity);
    expectStageSemantics(CubReductionOp::Mean,
                         CubReductionStageRole::Intermediate,
                         total_reduction_size,
                         CubReductionStageInputTransform::Identity,
                         CubReductionStageCombine::Sum,
                         CubReductionStageFinalize::Identity);
    expectStageSemantics(CubReductionOp::Mean,
                         CubReductionStageRole::Final,
                         total_reduction_size,
                         CubReductionStageInputTransform::Identity,
                         CubReductionStageCombine::Sum,
                         CubReductionStageFinalize::Divide,
                         total_reduction_size);

    expectStageSemantics(CubReductionOp::L1Norm,
                         CubReductionStageRole::Complete,
                         total_reduction_size,
                         CubReductionStageInputTransform::AbsoluteValue,
                         CubReductionStageCombine::Sum,
                         CubReductionStageFinalize::Identity);
    expectStageSemantics(CubReductionOp::L1Norm,
                         CubReductionStageRole::First,
                         total_reduction_size,
                         CubReductionStageInputTransform::AbsoluteValue,
                         CubReductionStageCombine::Sum,
                         CubReductionStageFinalize::Identity);
    expectStageSemantics(CubReductionOp::L1Norm,
                         CubReductionStageRole::Intermediate,
                         total_reduction_size,
                         CubReductionStageInputTransform::Identity,
                         CubReductionStageCombine::Sum,
                         CubReductionStageFinalize::Identity);
    expectStageSemantics(CubReductionOp::L1Norm,
                         CubReductionStageRole::Final,
                         total_reduction_size,
                         CubReductionStageInputTransform::Identity,
                         CubReductionStageCombine::Sum,
                         CubReductionStageFinalize::Identity);

    expectStageSemantics(CubReductionOp::L2Norm,
                         CubReductionStageRole::Complete,
                         total_reduction_size,
                         CubReductionStageInputTransform::Square,
                         CubReductionStageCombine::Sum,
                         CubReductionStageFinalize::SquareRoot);
    expectStageSemantics(CubReductionOp::L2Norm,
                         CubReductionStageRole::First,
                         total_reduction_size,
                         CubReductionStageInputTransform::Square,
                         CubReductionStageCombine::Sum,
                         CubReductionStageFinalize::Identity);
    expectStageSemantics(CubReductionOp::L2Norm,
                         CubReductionStageRole::Intermediate,
                         total_reduction_size,
                         CubReductionStageInputTransform::Identity,
                         CubReductionStageCombine::Sum,
                         CubReductionStageFinalize::Identity);
    expectStageSemantics(CubReductionOp::L2Norm,
                         CubReductionStageRole::Final,
                         total_reduction_size,
                         CubReductionStageInputTransform::Identity,
                         CubReductionStageCombine::Sum,
                         CubReductionStageFinalize::SquareRoot);

    expectStageSemantics(CubReductionOp::SumSquares,
                         CubReductionStageRole::Complete,
                         total_reduction_size,
                         CubReductionStageInputTransform::Square,
                         CubReductionStageCombine::Sum,
                         CubReductionStageFinalize::Identity);
    expectStageSemantics(CubReductionOp::SumSquares,
                         CubReductionStageRole::First,
                         total_reduction_size,
                         CubReductionStageInputTransform::Square,
                         CubReductionStageCombine::Sum,
                         CubReductionStageFinalize::Identity);
    expectStageSemantics(CubReductionOp::SumSquares,
                         CubReductionStageRole::Intermediate,
                         total_reduction_size,
                         CubReductionStageInputTransform::Identity,
                         CubReductionStageCombine::Sum,
                         CubReductionStageFinalize::Identity);
    expectStageSemantics(CubReductionOp::SumSquares,
                         CubReductionStageRole::Final,
                         total_reduction_size,
                         CubReductionStageInputTransform::Identity,
                         CubReductionStageCombine::Sum,
                         CubReductionStageFinalize::Identity);
}

TEST(CubReduction, KParallelTiledCompleteTopologyAppliesStageSemanticsScaleAndOutputConversion) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t outer_size = 1;
    constexpr uint64_t reduction_size = 4;
    constexpr uint64_t inner_size = 8;
    std::vector<float> values;
    values.reserve(outer_size * reduction_size * inner_size);
    for (uint64_t row = 0; row < reduction_size; ++row) {
        for (uint64_t component = 0; component < inner_size; ++component) {
            values.push_back(static_cast<float>(row + component + 1));
        }
    }

    Tensor input = makeGpuTensor(values, {outer_size, reduction_size, inner_size}, stream, DataType::FP16);
    const CubReductionGeometry geometry = CubReduction::analyzeGeometry(input.getDimensions(), 1);
    ASSERT_EQ(geometry.path, CubReductionPath::TiledFixedSegment);

    const CubReductionInternal::KParallelTiledStagePlan plan{
        CubReductionInternal::TiledRKStageTopology::Complete, 16, 32, 1};
    const CubReductionInternal::CubReductionStageSemantics semantics =
        CubReductionInternal::makeValueReductionStageSemantics(
            CubReductionOp::Mean, CubReductionInternal::CubReductionStageRole::Complete, reduction_size);

    Tensor output(gpuPlacement, TensorDescriptor(DataType::BF16, {outer_size, inner_size}));
    constexpr float output_scale = 2.0f;
    CubReductionInternal::launchKParallelTiledStage(
        semantics, input, output, geometry, plan, output_scale, stream);

    std::vector<float> expected(inner_size);
    for (uint64_t component = 0; component < inner_size; ++component) {
        expected[component] = 2.0f * (static_cast<float>(component) + 2.5f);
    }
    expectFloatVectorNear(copyGpuTensorAsFloat(output, stream), expected, 0.03125f);

    // Physical COMPLETE topology means only that this R reduction needs no partial tensor. It must not manufacture
    // mathematical finalization: a composed First pass applies the input transform but not L2's square-root finalizer.
    const CubReductionInternal::CubReductionStageSemantics first_semantics =
        CubReductionInternal::makeValueReductionStageSemantics(
            CubReductionOp::L2Norm, CubReductionInternal::CubReductionStageRole::First, reduction_size);
    Tensor first_output(gpuPlacement, TensorDescriptor(DataType::FP32, {outer_size, inner_size}));
    CubReductionInternal::launchKParallelTiledStage(
        first_semantics, input, first_output, geometry, plan, 1.0f, stream);

    std::vector<float> expected_sum_squares(inner_size, 0.0f);
    for (uint64_t row = 0; row < reduction_size; ++row) {
        for (uint64_t component = 0; component < inner_size; ++component) {
            const float value = static_cast<float>(row + component + 1);
            expected_sum_squares[component] += value * value;
        }
    }
    expectFloatVectorNear(copyGpuTensorAsFloat(first_output, stream), expected_sum_squares, 1.0e-5f);
}

// Regression: Complete KParallel previously capped its launch at 65535 CTAs
// without a grid-stride loop. FP32 continuations with >65535 narrow-K outputs
// silently left the tail unwritten (observed after the FP8 FlatRows first stage).
TEST(CubReduction, KParallelCompleteGridStridesBeyond65535OuterRows) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t outer_size = 65537;
    constexpr uint64_t reduction_size = 32;
    constexpr uint64_t inner_size = 3;
    Tensor input(gpuPlacement, TensorDescriptor(DataType::FP32, {outer_size, reduction_size, inner_size}));
    Tensor output(gpuPlacement, TensorDescriptor(DataType::FP32, {outer_size, inner_size}));
    input.fill(1.0f, stream);
    output.fill(-1.0f, stream);

    const CubReductionGeometry geometry = CubReduction::analyzeGeometry(input.getDimensions(), 1);
    ASSERT_EQ(geometry.path, CubReductionPath::TiledFixedSegment);
    const CubReductionInternal::KParallelTiledStagePlan plan{
        CubReductionInternal::TiledRKStageTopology::Complete, 4, 32, 1};
    const auto semantics = CubReductionInternal::makeValueReductionStageSemantics(
        CubReductionOp::Sum, CubReductionInternal::CubReductionStageRole::Complete, reduction_size);
    CubReductionInternal::launchKParallelTiledStage(semantics, input, output, geometry, plan, 1.0f, stream);
    stream.synchronize();
    expectFloatVectorNear(copyGpuTensorAsFloat(output, stream),
                          std::vector<float>(outer_size * inner_size, 32.0f), 1.0e-5f);
}

// 1136: K-parallel CTAs own consecutive packets of the flat [outer, K / items] packet space, so a CTA spans several
// outer rows at narrow K. Every packet, outer row and shard must land in its own output, including packets that cross
// CTA boundaries and outer counts past the 65535-CTA grid cap.
TEST(CubReduction, KParallelFlatPacketOwnershipCoversNarrowAndWideK) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    struct Case {
        uint64_t outer, r, k;
        size_t packet_bytes;
        uint32_t block_threads;
        uint64_t shards;  // 1: Complete
    };
    const std::vector<Case> cases{
        {70001, 3, 2, 8, 32, 1},    // K = 2: 16 outer rows per warp; grid-strides past 65535 CTAs
        {9001, 5, 6, 8, 32, 1},     // 3 packets per row: rows straddle warps
        {4099, 2, 40, 16, 32, 1},   // 10 packets per row
        {517, 7, 3, 4, 64, 1},
        {33, 4, 1000, 16, 256, 1},  // wide K: 250 packets per row, 6 CTAs straddle rows
        {1000, 50, 3, 4, 32, 5},    // Staged
        {257, 33, 10, 8, 32, 4},
        {3, 64, 1024, 16, 32, 2},
    };
    for (const Case& c : cases) {
        SCOPED_TRACE(::testing::Message() << "outer=" << c.outer << " R=" << c.r << " K=" << c.k << " packet="
                                          << c.packet_bytes << " threads=" << c.block_threads
                                          << " shards=" << c.shards);
        std::vector<float> values(c.outer * c.r * c.k);
        for (uint64_t o = 0; o < c.outer; ++o)
            for (uint64_t r = 0; r < c.r; ++r)
                for (uint64_t k = 0; k < c.k; ++k)
                    values[(o * c.r + r) * c.k + k] = static_cast<float>((o * 7 + r * 3 + k * 5) % 13);
        Tensor input = makeGpuTensor(values, {c.outer, c.r, c.k}, stream, DataType::FP32);
        const CubReductionGeometry geometry = CubReduction::analyzeGeometry(input.getDimensions(), 1);
        ASSERT_EQ(geometry.path, CubReductionPath::TiledFixedSegment);
        const bool complete = c.shards == 1;
        const CubReductionInternal::KParallelTiledStagePlan plan{
            complete ? CubReductionInternal::TiledRKStageTopology::Complete
                     : CubReductionInternal::TiledRKStageTopology::Staged,
            c.packet_bytes, c.block_threads, c.shards};
        const auto semantics = CubReductionInternal::makeValueReductionStageSemantics(
            CubReductionOp::Sum, CubReductionInternal::CubReductionStageRole::Complete, c.r);
        Tensor output(gpuPlacement, TensorDescriptor(DataType::FP32, complete ? std::vector<uint64_t>{c.outer, c.k}
                                                                              : std::vector<uint64_t>{c.outer, c.shards, c.k}));
        output.fill(-1.0f, stream);
        CubReductionInternal::launchKParallelTiledStage(semantics, input, output, geometry, plan, 1.0f, stream);
        stream.synchronize();
        std::vector<float> expected(c.outer * c.shards * c.k, 0.0f);
        const uint64_t base = c.r / c.shards, extra = c.r % c.shards;
        for (uint64_t o = 0; o < c.outer; ++o) {
            for (uint64_t s = 0; s < c.shards; ++s) {
                const uint64_t begin = s * base + std::min(s, extra);
                const uint64_t end = begin + base + (s < extra ? 1 : 0);
                for (uint64_t r = begin; r < end; ++r)
                    for (uint64_t k = 0; k < c.k; ++k)
                        expected[(o * c.shards + s) * c.k + k] += values[(o * c.r + r) * c.k + k];
            }
        }
        expectFloatVectorNear(copyGpuTensorAsFloat(output, stream), expected, 0.0f);
    }
}

TEST(CubReduction, KParallelTiledStagedTopologyDefersFinalizerAndScale) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t outer_size = 1;
    constexpr uint64_t reduction_size = 4;
    constexpr uint64_t inner_size = 8;
    constexpr uint64_t shards_per_output = 2;
    std::vector<float> values;
    values.reserve(outer_size * reduction_size * inner_size);
    for (uint64_t row = 0; row < reduction_size; ++row) {
        for (uint64_t component = 0; component < inner_size; ++component) {
            values.push_back(static_cast<float>(row + 1));
        }
    }

    Tensor input = makeGpuTensor(values, {outer_size, reduction_size, inner_size}, stream, DataType::FP16);
    const CubReductionGeometry geometry = CubReduction::analyzeGeometry(input.getDimensions(), 1);
    ASSERT_EQ(geometry.path, CubReductionPath::TiledFixedSegment);

    // Deliberately pass Complete L2 semantics and a non-unit scale. STAGED topology must use L2's Square input
    // transform but emit raw FP32 shard sums; sqrt and scale belong to the independently planned continuation.
    const CubReductionInternal::CubReductionStageSemantics semantics =
        CubReductionInternal::makeValueReductionStageSemantics(
            CubReductionOp::L2Norm, CubReductionInternal::CubReductionStageRole::Complete, reduction_size);
    const CubReductionInternal::KParallelTiledStagePlan plan{
        CubReductionInternal::TiledRKStageTopology::Staged, 16, 32, shards_per_output};
    Tensor partials(gpuPlacement,
                    TensorDescriptor(DataType::FP32, {outer_size, shards_per_output, inner_size}));
    CubReductionInternal::launchKParallelTiledStage(
        semantics, input, partials, geometry, plan, 7.0f, stream);

    std::vector<float> expected;
    expected.reserve(shards_per_output * inner_size);
    for (uint64_t component = 0; component < inner_size; ++component) {
        expected.push_back(5.0f);  // rows 1 and 2: 1^2 + 2^2
    }
    for (uint64_t component = 0; component < inner_size; ++component) {
        expected.push_back(25.0f);  // rows 3 and 4: 3^2 + 4^2
    }
    expectFloatVectorNear(copyGpuTensorAsFloat(partials, stream), expected, 0.0f);
}

#if THOR_CUB_ENABLE_FP8_TYPES
// The retained DeepRSmallK P16 kernels now select the reduction operation at
// compile time. Exercise both tiny-K ownership geometries, all five depths,
// regular-K component ownership and all input transforms, including a partial
// last shard. The partials must remain unfinalized FP32 values.
TEST(CubReduction, Fp8DeepRSmallKRetainedTypedOperationsAndTransforms) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    using CubReductionInternal::CubReductionStageCombine;
    using CubReductionInternal::CubReductionStageFinalize;
    using CubReductionInternal::CubReductionStageInputTransform;

    constexpr uint64_t outer_size = 2;
    const std::array<uint32_t, 5> depths = {64, 128, 256, 512, 1024};
    const std::array<uint64_t, 7> ownership_k = {3, 7, 17, 47, 95, 151, 239};
    const std::array<CubReductionStageCombine, 4> combines = {
        CubReductionStageCombine::Sum, CubReductionStageCombine::Product,
        CubReductionStageCombine::Minimum, CubReductionStageCombine::Maximum};
    const std::array<CubReductionStageInputTransform, 3> transforms = {
        CubReductionStageInputTransform::Identity,
        CubReductionStageInputTransform::AbsoluteValue,
        CubReductionStageInputTransform::Square};
    // Keep PRODUCT well conditioned under the kernel's parallel reduction tree.
    // At K=239 the P16 kernel assigns two rows per warp group. With the old
    // {0.5, -1, 2, -1} pattern, Squared PRODUCT repeatedly multiplied 0.25
    // in some warps and 4 in others: 4^64 overflows FP32 before their partials
    // can be combined, even though the mathematical product is 1. Adjacent
    // pairs below multiply to 1 under Identity, ABS and Square, while still
    // exercising distinct magnitudes and both signs.
    const std::array<float, 4> pattern = {0.5f, 2.0f, -0.5f, -2.0f};

    for (const DataType dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
        SCOPED_TRACE(static_cast<int>(dtype));
        for (const uint32_t rows : depths) {
            SCOPED_TRACE(rows);
            // Every ownership specialization at depth 64, and regular/tiny
            // controls at all remaining depths.
            for (const uint64_t k : ownership_k) {
                if (rows != 64 && k != 3 && k != 95 && k != 239) continue;
                SCOPED_TRACE(k);
                const uint64_t r = rows + 3;
                constexpr uint64_t shards = 2;
                std::vector<float> values(outer_size * r * k);
                for (uint64_t outer = 0; outer < outer_size; ++outer) {
                    for (uint64_t row = 0; row < r; ++row) {
                        for (uint64_t component = 0; component < k; ++component) {
                            values[(outer * r + row) * k + component] =
                                pattern[(outer + row + component) % pattern.size()];
                        }
                    }
                }
                Tensor input = makeGpuTensor(values, {outer_size, r, k}, stream, dtype);
                const CubReductionGeometry geometry = CubReduction::analyzeGeometry(input.getDimensions(), 1);
                ASSERT_EQ(geometry.path, CubReductionPath::TiledFixedSegment);
                Tensor partials(gpuPlacement, TensorDescriptor(DataType::FP32, {outer_size, shards, k}));
                for (const auto combine : combines) {
                    SCOPED_TRACE(static_cast<int>(combine));
                    for (const auto transform : transforms) {
                        SCOPED_TRACE(static_cast<int>(transform));
                        CubReductionInternal::CubReductionStageSemantics semantics{};
                        semantics.combine = combine;
                        semantics.input_transform = transform;
                        // A first-stage reducer must not apply finalization,
                        // even if given a Complete mathematical descriptor.
                        semantics.finalize = CubReductionStageFinalize::SquareRoot;
                        const auto occ = CubReductionInternal::queryFp8DeepRSmallKRetainedOccupancy(
                            dtype, rows, k, semantics);
                        EXPECT_GT(occ.max_active_blocks_per_sm, 0);
                        CubReductionInternal::launchFp8DeepRSmallKRetainedFirstStage(
                            semantics, input, partials, geometry, rows, shards, stream);
                        std::vector<float> expected;
                        expected.reserve(outer_size * shards * k);
                        for (uint64_t outer = 0; outer < outer_size; ++outer) {
                            for (uint64_t shard = 0; shard < shards; ++shard) {
                                const uint64_t begin = shard * rows;
                                const uint64_t end = std::min(begin + rows, r);
                                for (uint64_t component = 0; component < k; ++component) {
                                    float acc = combine == CubReductionStageCombine::Product ? 1.0f
                                              : combine == CubReductionStageCombine::Minimum ? std::numeric_limits<float>::infinity()
                                              : combine == CubReductionStageCombine::Maximum ? -std::numeric_limits<float>::infinity() : 0.0f;
                                    for (uint64_t row = begin; row < end; ++row) {
                                        float v = values[(outer * r + row) * k + component];
                                        if (transform == CubReductionStageInputTransform::AbsoluteValue) v = std::fabs(v);
                                        else if (transform == CubReductionStageInputTransform::Square) v *= v;
                                        if (combine == CubReductionStageCombine::Sum) acc += v;
                                        else if (combine == CubReductionStageCombine::Product) acc *= v;
                                        else if (combine == CubReductionStageCombine::Minimum) acc = std::min(acc, v);
                                        else acc = std::max(acc, v);
                                    }
                                    expected.push_back(acc);
                                }
                            }
                        }
                        expectFloatVectorNear(copyGpuTensorAsFloat(partials, stream), expected, 0.0f);
                        // Verify that the independently planned FP32
                        // continuation combines the unfinalized shard output
                        // correctly for each non-additive operation as well.
                        if (rows == 128 && k == 95 && transform == CubReductionStageInputTransform::Identity) {
                            const auto continuation_geometry = CubReduction::analyzeGeometry(partials.getDimensions(), 1);
                            Tensor final_output(gpuPlacement, TensorDescriptor(
                                DataType::FP32, continuation_geometry.output_dimensions));
                            const CubReductionInternal::KParallelTiledStagePlan plan{
                                CubReductionInternal::TiledRKStageTopology::Complete, 4, 32, 1};
                            CubReductionInternal::CubReductionStageSemantics final_semantics{};
                            final_semantics.combine = combine;
                            CubReductionInternal::launchKParallelTiledStage(
                                final_semantics, partials, final_output, continuation_geometry, plan, 1.0f, stream);
                            std::vector<float> expected_final;
                            expected_final.reserve(outer_size * k);
                            for (uint64_t outer = 0; outer < outer_size; ++outer) {
                                for (uint64_t component = 0; component < k; ++component) {
                                    const float a = expected[(outer * shards) * k + component];
                                    const float b = expected[(outer * shards + 1) * k + component];
                                    expected_final.push_back(combine == CubReductionStageCombine::Sum ? a + b
                                        : combine == CubReductionStageCombine::Product ? a * b
                                        : combine == CubReductionStageCombine::Minimum ? std::min(a, b) : std::max(a, b));
                                }
                            }
                            expectFloatVectorNear(copyGpuTensorAsFloat(final_output, stream), expected_final, 0.0f);
                        }
                    }
                }
            }
        }
    }
}

// P5-C: every retained DeepRSmallK depth now launches the same kernel with a runtime
// rows-per-shard operand. Run all five depths back to back on one input shape per
// ownership kind, with five shards per output (a short last shard), so a wrong shard
// origin or clamp at any depth shows up, and reject depths outside the retained set.
TEST(CubReduction, Fp8DeepRSmallKRuntimeRowDepthPlacesEveryShard) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    using CubReductionInternal::CubReductionStageCombine;
    constexpr uint64_t outer_size = 3;
    for (const DataType dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
        SCOPED_TRACE(static_cast<int>(dtype));
        for (const uint64_t k : {uint64_t{3}, uint64_t{7}, uint64_t{17}, uint64_t{95}}) {
            SCOPED_TRACE(k);
            for (const uint32_t rows : {64u, 128u, 256u, 512u, 1024u}) {
                SCOPED_TRACE(rows);
                const uint64_t r = 4 * static_cast<uint64_t>(rows) + 5;
                const uint64_t shards = 5;
                // Each value encodes its row so a misplaced shard changes the sum.
                std::vector<float> values(outer_size * r * k);
                for (uint64_t outer = 0; outer < outer_size; ++outer)
                    for (uint64_t row = 0; row < r; ++row)
                        for (uint64_t component = 0; component < k; ++component)
                            values[(outer * r + row) * k + component] =
                                static_cast<float>((row + outer + component) % 8) * 0.125f;
                Tensor input = makeGpuTensor(values, {outer_size, r, k}, stream, dtype);
                const CubReductionGeometry geometry = CubReduction::analyzeGeometry(input.getDimensions(), 1);
                Tensor partials(gpuPlacement, TensorDescriptor(DataType::FP32, {outer_size, shards, k}));
                CubReductionInternal::CubReductionStageSemantics semantics{};
                semantics.combine = CubReductionStageCombine::Sum;
                CubReductionInternal::launchFp8DeepRSmallKRetainedFirstStage(
                    semantics, input, partials, geometry, rows, shards, stream);
                std::vector<float> expected;
                expected.reserve(outer_size * shards * k);
                for (uint64_t outer = 0; outer < outer_size; ++outer)
                    for (uint64_t shard = 0; shard < shards; ++shard)
                        for (uint64_t component = 0; component < k; ++component) {
                            float acc = 0.0f;  // Multiples of 1/8 below 2^21: exact in FP32 in any order.
                            for (uint64_t row = shard * rows; row < std::min<uint64_t>((shard + 1) * rows, r); ++row)
                                acc += values[(outer * r + row) * k + component];
                            expected.push_back(acc);
                        }
                expectFloatVectorNear(copyGpuTensorAsFloat(partials, stream), expected, 0.0f);
            }
        }
        // Depths outside the retained set are rejected before any launch.
        Tensor input = makeGpuTensor(std::vector<float>(2 * 200 * 3, 1.0f), {2, 200, 3}, stream, dtype);
        const CubReductionGeometry geometry = CubReduction::analyzeGeometry(input.getDimensions(), 1);
        Tensor partials(gpuPlacement, TensorDescriptor(DataType::FP32, {2, 3, 3}));
        CubReductionInternal::CubReductionStageSemantics semantics{};
        for (const uint64_t bad_rows : {uint64_t{96}, uint64_t{2048}}) {
            EXPECT_THROW(CubReductionInternal::launchFp8DeepRSmallKRetainedFirstStage(
                             semantics, input, partials, geometry, bad_rows, 3, stream),
                         std::logic_error);
            EXPECT_THROW((void)CubReductionInternal::queryFp8DeepRSmallKRetainedOccupancy(
                             dtype, bad_rows, 3, semantics),
                         std::logic_error);
        }
    }
}

// The retained AlignedK Staged P2/P8 x CTA32/64/128 physical inventory shares one
// launch-uniform SUM/PRODUCT/MIN/MAX kernel. Test the production dispatch rather
// than comparing two entry points that now resolve to that same kernel.
TEST(CubReduction, Fp8AlignedKStagedUnifiedOperationsPreserveShardPartials) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    constexpr uint64_t outer_size = 2;
    constexpr uint64_t reduction_size = 11;  // shard counts are 4,4,3
    constexpr uint64_t inner_size = 520;      // P8-aligned, partial CTA tile
    constexpr uint64_t shards = 3;
    const std::array<float, 7> pattern = {1.0f, -1.0f, 2.0f, -2.0f, 0.0f, 1.0f, -1.0f};
    std::vector<float> values(outer_size * reduction_size * inner_size);
    for (uint64_t outer = 0; outer < outer_size; ++outer) {
        for (uint64_t r = 0; r < reduction_size; ++r) {
            for (uint64_t k = 0; k < inner_size; ++k) {
                values[(outer * reduction_size + r) * inner_size + k] =
                    pattern[(outer + r + k) % pattern.size()];
            }
        }
    }
    const std::array<CubReductionOp, 4> operations = {
        CubReductionOp::Sum, CubReductionOp::Product, CubReductionOp::Min, CubReductionOp::Max};
    const std::array<std::pair<size_t, uint32_t>, 6> physical = {{
        {2, 32}, {2, 64}, {2, 128}, {8, 32}, {8, 64}, {8, 128}
    }};
    for (DataType dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
        SCOPED_TRACE(static_cast<int>(dtype));
        Tensor input = makeGpuTensor(values, {outer_size, reduction_size, inner_size}, stream, dtype);
        const CubReductionGeometry geometry = CubReduction::analyzeGeometry(input.getDimensions(), 1);
        ASSERT_EQ(geometry.path, CubReductionPath::TiledFixedSegment);
        for (const auto& [packet, threads] : physical) {
            SCOPED_TRACE(packet);
            SCOPED_TRACE(threads);
            const CubReductionInternal::KParallelTiledStagePlan plan{
                CubReductionInternal::TiledRKStageTopology::Staged, packet, threads, shards};
            const auto occupancy = CubReductionInternal::queryKParallelTiledStageOccupancy(dtype, plan);
            EXPECT_GT(occupancy.max_active_blocks_per_sm, 0U);
            for (CubReductionOp op : operations) {
                SCOPED_TRACE(static_cast<int>(op));
                Tensor partials(gpuPlacement, TensorDescriptor(
                    DataType::FP32, {outer_size, shards, inner_size}));
                const auto semantics = CubReductionInternal::makeValueReductionStageSemantics(
                    op, CubReductionInternal::CubReductionStageRole::First, reduction_size);
                CubReductionInternal::launchKParallelTiledStage(
                    semantics, input, partials, geometry, plan, 7.0f, stream);
                std::vector<float> expected;
                expected.reserve(outer_size * shards * inner_size);
                for (uint64_t outer = 0; outer < outer_size; ++outer) {
                    for (uint64_t shard = 0; shard < shards; ++shard) {
                        const uint64_t base = reduction_size / shards;
                        const uint64_t extra = reduction_size % shards;
                        const uint64_t first = shard * base + std::min(shard, extra);
                        const uint64_t last = first + base + static_cast<uint64_t>(shard < extra);
                        for (uint64_t k = 0; k < inner_size; ++k) {
                            float acc = op == CubReductionOp::Product ? 1.0f
                                      : op == CubReductionOp::Min ? std::numeric_limits<float>::infinity()
                                      : op == CubReductionOp::Max ? -std::numeric_limits<float>::infinity() : 0.0f;
                            for (uint64_t r = first; r < last; ++r) {
                                const float v = values[(outer * reduction_size + r) * inner_size + k];
                                if (op == CubReductionOp::Sum) acc += v;
                                else if (op == CubReductionOp::Product) acc *= v;
                                else if (op == CubReductionOp::Min) acc = std::min(acc, v);
                                else acc = std::max(acc, v);
                            }
                            expected.push_back(acc);
                        }
                    }
                }
                expectFloatVectorNear(copyGpuTensorAsFloat(partials, stream), expected, 0.0f);
            }
        }
    }
}

// Exercise the real production Complete dispatch, not retired A/B wrappers.
// P2/CTA32 must join before additive finalization, P8/CTA32 must finalize
// inside the additive branch, and P8/CTA64 must mix typed SUM/MEAN with
// the unchanged full-body in-branch PRODUCT/MIN/MAX kernel.
TEST(CubReduction, Fp8AlignedKCompleteRetainedOperationsAndOutputFormats) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    constexpr uint64_t outer_size = 2;
    constexpr uint64_t reduction_size = 4;
    constexpr uint64_t inner_size = 520;  // Multiple warp tiles; P8 FP32 transpose.
    constexpr float output_scale = 0.5f;
    std::vector<float> values(outer_size * reduction_size * inner_size);
    for (uint64_t outer = 0; outer < outer_size; ++outer)
        for (uint64_t r = 0; r < reduction_size; ++r)
            for (uint64_t k = 0; k < inner_size; ++k)
                values[(outer * reduction_size + r) * inner_size + k] =
                    static_cast<float>(static_cast<int>((outer + r + k) % 5) - 2);

    const std::array<CubReductionOp, 5> operations = {
        CubReductionOp::Sum, CubReductionOp::Mean, CubReductionOp::Product,
        CubReductionOp::Min, CubReductionOp::Max};
    const std::array<std::pair<size_t, uint32_t>, 3> physical = {{{2, 32}, {8, 32}, {8, 64}}};
    for (DataType input_dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
        SCOPED_TRACE(static_cast<int>(input_dtype));
        Tensor input = makeGpuTensor(values, {outer_size, reduction_size, inner_size}, stream, input_dtype);
        const auto geometry = CubReduction::analyzeGeometry(input.getDimensions(), 1);
        ASSERT_EQ(geometry.path, CubReductionPath::TiledFixedSegment);
        for (const auto& [packet, cta] : physical) {
            SCOPED_TRACE(packet);
            SCOPED_TRACE(cta);
            const CubReductionInternal::KParallelTiledStagePlan plan{
                CubReductionInternal::TiledRKStageTopology::Complete, packet, cta, 1};
            for (CubReductionOp op : operations) {
                SCOPED_TRACE(static_cast<int>(op));
                const auto semantics = CubReductionInternal::makeValueReductionStageSemantics(
                    op, CubReductionInternal::CubReductionStageRole::Complete, reduction_size);
                for (DataType output_dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2,
                                              DataType::FP16, DataType::BF16, DataType::FP32}) {
                    SCOPED_TRACE(static_cast<int>(output_dtype));
                    const auto occupancy = CubReductionInternal::queryFp8AlignedKCompleteOccupancy(
                        input_dtype, output_dtype, semantics.combine, plan);
                    EXPECT_GT(occupancy.max_active_blocks_per_sm, 0);
                    Tensor output(gpuPlacement, TensorDescriptor(output_dtype, {outer_size, inner_size}));
                    CubReductionInternal::launchKParallelTiledStage(
                        semantics, input, output, geometry, plan, output_scale, stream);
                    std::vector<float> expected;
                    expected.reserve(outer_size * inner_size);
                    for (uint64_t outer = 0; outer < outer_size; ++outer) {
                        for (uint64_t k = 0; k < inner_size; ++k) {
                            float v = op == CubReductionOp::Product ? 1.0f
                                    : op == CubReductionOp::Min ? std::numeric_limits<float>::infinity()
                                    : op == CubReductionOp::Max ? -std::numeric_limits<float>::infinity()
                                    : 0.0f;
                            for (uint64_t r = 0; r < reduction_size; ++r) {
                                const float x = values[(outer * reduction_size + r) * inner_size + k];
                                if (op == CubReductionOp::Sum || op == CubReductionOp::Mean) v += x;
                                else if (op == CubReductionOp::Product) v *= x;
                                else if (op == CubReductionOp::Min) v = std::min(v, x);
                                else v = std::max(v, x);
                            }
                            if (op == CubReductionOp::Mean) v /= static_cast<float>(reduction_size);
                            expected.push_back(v * output_scale);
                        }
                    }
                    // Integer FP8 inputs, power-of-two scaling and four-row means
                    // keep these reference values exactly representable even in E5M2.
                    expectFloatVectorNear(copyGpuTensorAsFloat(output, stream), expected, 0.0f);
                }
            }
        }
    }
}

TEST(CubReduction, Fp8AlignedKStagedWritesFp32PartialsWithoutFinalization) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t outer_size = 2;
    constexpr uint64_t reduction_size = 256;
    constexpr uint64_t inner_size = 528;
    constexpr uint64_t shards_per_output = 2;
    std::vector<float> values(outer_size * reduction_size * inner_size);
    for (uint64_t outer = 0; outer < outer_size; ++outer) {
        for (uint64_t row = 0; row < reduction_size; ++row) {
            const float value = row < 128 ? 1.0f : 2.0f;
            for (uint64_t component = 0; component < inner_size; ++component) {
                values[(outer * reduction_size + row) * inner_size + component] = value;
            }
        }
    }

    for (DataType dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
        SCOPED_TRACE(static_cast<int>(dtype));
        Tensor input = makeGpuTensor(values, {outer_size, reduction_size, inner_size}, stream, dtype);
        const CubReductionGeometry geometry = CubReduction::analyzeGeometry(input.getDimensions(), 1);
        ASSERT_EQ(geometry.path, CubReductionPath::TiledFixedSegment);
        // Square-transform regression across both retained AlignedK packet widths.
        for (size_t packet_bytes : {size_t{8}, size_t{2}}) {
            SCOPED_TRACE(packet_bytes);
            Tensor partials(gpuPlacement,
                            TensorDescriptor(DataType::FP32, {outer_size, shards_per_output, inner_size}));
            const CubReductionInternal::KParallelTiledStagePlan plan{
                CubReductionInternal::TiledRKStageTopology::Staged, packet_bytes, 32, shards_per_output};
            const auto semantics = CubReductionInternal::makeValueReductionStageSemantics(
                CubReductionOp::L2Norm, CubReductionInternal::CubReductionStageRole::Complete, reduction_size);
            // Complete L2 semantics arrive at a staged pass: Square on input; defer sqrt and scale.
            CubReductionInternal::launchKParallelTiledStage(
                semantics, input, partials, geometry, plan, 7.0f, stream);
            std::vector<float> l2_expected;
            l2_expected.reserve(outer_size * shards_per_output * inner_size);
            for (uint64_t outer = 0; outer < outer_size; ++outer) {
                l2_expected.insert(l2_expected.end(), inner_size, 128.0f);
                l2_expected.insert(l2_expected.end(), inner_size, 512.0f);
            }
            expectFloatVectorNear(copyGpuTensorAsFloat(partials, stream), l2_expected, 0.0f);
        }
    }
}

TEST(CubReduction, Fp8AlignedKStagedOccupancyCoversOnlyRetainedPackets) {
    REQUIRE_CUDA_DEVICE();

    for (DataType dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
        SCOPED_TRACE(static_cast<int>(dtype));
        for (size_t packet_bytes : {size_t{8}, size_t{2}}) {
            SCOPED_TRACE(packet_bytes);
            const CubReductionInternal::KParallelTiledStagePlan plan{
                CubReductionInternal::TiledRKStageTopology::Staged, packet_bytes, 32, 2};
            const CubReductionInternal::CubKernelOccupancyInfo occupancy =
                CubReductionInternal::queryKParallelTiledStageOccupancy(dtype, plan);
            EXPECT_GT(occupancy.max_active_blocks_per_sm, 0);
            if (packet_bytes == 8) {
                EXPECT_GT(occupancy.dynamic_shared_bytes, 0U);
            } else {
                EXPECT_EQ(occupancy.dynamic_shared_bytes, 0U);
            }
        }
        // P4/P16 and CTA256 are not compiled for FP8 AlignedK; they must report absent.
        for (const auto& [packet_bytes, threads] :
             std::array<std::pair<size_t, uint32_t>, 3>{{{16, 32}, {4, 32}, {8, 256}}}) {
            SCOPED_TRACE(packet_bytes);
            SCOPED_TRACE(threads);
            const CubReductionInternal::KParallelTiledStagePlan plan{
                CubReductionInternal::TiledRKStageTopology::Staged, packet_bytes, threads, 2};
            EXPECT_THROW((void)CubReductionInternal::queryKParallelTiledStageOccupancy(dtype, plan), std::logic_error);
        }
    }
}

TEST(CubReduction, Fp8AlignedKRetainedPacketsPreserveFp32Writeback) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t reduction_size = 256;
    constexpr uint64_t shards_per_output = 2;
    for (size_t packet_bytes : {size_t{8}, size_t{2}}) {
        const uint64_t inner_size = packet_bytes * 33U;  // one full warp tile plus one producer packet
        SCOPED_TRACE(packet_bytes);
        std::vector<float> values(reduction_size * inner_size);
        for (uint64_t row = 0; row < reduction_size; ++row) {
            const float value = row < 128 ? 1.0f : 2.0f;
            for (uint64_t component = 0; component < inner_size; ++component) {
                values[row * inner_size + component] = value;
            }
        }

        for (DataType dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
            SCOPED_TRACE(static_cast<int>(dtype));
            Tensor input = makeGpuTensor(values, {1, reduction_size, inner_size}, stream, dtype);
            const CubReductionGeometry geometry = CubReduction::analyzeGeometry(input.getDimensions(), 1);
            ASSERT_EQ(geometry.path, CubReductionPath::TiledFixedSegment);
            const CubReductionInternal::CubReductionStageSemantics semantics =
                CubReductionInternal::makeValueReductionStageSemantics(
                    CubReductionOp::Sum, CubReductionInternal::CubReductionStageRole::Complete, reduction_size);

            Tensor partials(gpuPlacement, TensorDescriptor(DataType::FP32, {1, shards_per_output, inner_size}));
            const CubReductionInternal::KParallelTiledStagePlan staged_plan{
                CubReductionInternal::TiledRKStageTopology::Staged, packet_bytes, 32, shards_per_output};
            CubReductionInternal::launchKParallelTiledStage(
                semantics, input, partials, geometry, staged_plan, 9.0f, stream);

            std::vector<float> staged_expected;
            staged_expected.reserve(shards_per_output * inner_size);
            staged_expected.insert(staged_expected.end(), inner_size, 128.0f);
            staged_expected.insert(staged_expected.end(), inner_size, 256.0f);
            expectFloatVectorNear(copyGpuTensorAsFloat(partials, stream), staged_expected, 0.0f);

            Tensor output(gpuPlacement, TensorDescriptor(DataType::FP32, {1, inner_size}));
            const CubReductionInternal::KParallelTiledStagePlan complete_plan{
                CubReductionInternal::TiledRKStageTopology::Complete, packet_bytes, 32, 1};
            CubReductionInternal::launchKParallelTiledStage(
                semantics, input, output, geometry, complete_plan, 1.0f, stream);
            expectFloatVectorNear(copyGpuTensorAsFloat(output, stream),
                                  std::vector<float>(inner_size, 384.0f),
                                  0.0f);
        }
    }
}

TEST(CubReduction, Fp8AlignedKCompleteKeepsRetainedPacketWidthsPacketized) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t reduction_size = 4;
    for (size_t packet_bytes : {size_t{8}, size_t{2}}) {
        const uint64_t inner_size = packet_bytes * 33U;
        SCOPED_TRACE(packet_bytes);
        std::vector<float> values(reduction_size * inner_size);
        std::vector<float> expected(inner_size);
        for (uint64_t component = 0; component < inner_size; ++component) {
            const float value = static_cast<float>(1U << (component % 4U));  // exact in both FP8 formats
            expected[component] = value;
            for (uint64_t row = 0; row < reduction_size; ++row) {
                values[row * inner_size + component] = value;
            }
        }

        for (DataType dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
            SCOPED_TRACE(static_cast<int>(dtype));
            Tensor input = makeGpuTensor(values, {1, reduction_size, inner_size}, stream, dtype);
            const CubReductionGeometry geometry = CubReduction::analyzeGeometry(input.getDimensions(), 1);
            Tensor output(gpuPlacement, TensorDescriptor(dtype, {1, inner_size}));
            const CubReductionInternal::KParallelTiledStagePlan plan{
                CubReductionInternal::TiledRKStageTopology::Complete, packet_bytes, 32, 1};
            const CubReductionInternal::CubReductionStageSemantics semantics =
                CubReductionInternal::makeValueReductionStageSemantics(
                    CubReductionOp::Mean, CubReductionInternal::CubReductionStageRole::Complete, reduction_size);

            CubReductionInternal::launchKParallelTiledStage(
                semantics, input, output, geometry, plan, 1.0f, stream);
            expectFloatVectorNear(copyGpuTensorAsFloat(output, stream), expected, 0.0f);
        }
    }
}

namespace {

// Powers of two only: exact in E4M3/E5M2, and every partial SUM/MAX below is exact in FP32.
float fp8GridMappingValue(uint64_t outer, uint64_t row, uint64_t component) {
    constexpr std::array<float, 5> pattern = {0.5f, -1.0f, 2.0f, -0.25f, 1.0f};
    return pattern[(outer * 7 + row * 3 + component) % pattern.size()];
}

std::vector<float> fp8GridMappingValues(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size) {
    std::vector<float> values(outer_size * reduction_size * inner_size);
    for (uint64_t outer = 0; outer < outer_size; ++outer) {
        for (uint64_t row = 0; row < reduction_size; ++row) {
            for (uint64_t component = 0; component < inner_size; ++component) {
                values[(outer * reduction_size + row) * inner_size + component] =
                    fp8GridMappingValue(outer, row, component);
            }
        }
    }
    return values;
}

enum class Fp8GridMappingOp { Sum, AbsSum, Max };

// Reference [outer, shards, K] partials for explicit per-shard row ranges.
std::vector<float> fp8GridMappingExpected(uint64_t outer_size,
                                          uint64_t inner_size,
                                          const std::vector<std::pair<uint64_t, uint64_t>>& shard_rows,
                                          Fp8GridMappingOp op) {
    const bool maximum = op == Fp8GridMappingOp::Max;
    std::vector<float> expected;
    expected.reserve(outer_size * shard_rows.size() * inner_size);
    for (uint64_t outer = 0; outer < outer_size; ++outer) {
        for (const auto& [begin, end] : shard_rows) {
            for (uint64_t component = 0; component < inner_size; ++component) {
                float acc = maximum ? -std::numeric_limits<float>::infinity() : 0.0f;
                for (uint64_t row = begin; row < end; ++row) {
                    float value = fp8GridMappingValue(outer, row, component);
                    if (op == Fp8GridMappingOp::AbsSum) value = std::fabs(value);
                    acc = maximum ? std::max(acc, value) : acc + value;
                }
                expected.push_back(acc);
            }
        }
    }
    return expected;
}

}  // namespace

// The retained FP8 families map (shard, K tile, outer) onto grid (x, y, z) without device-side division. Outer sizes
// above 65535 exceed the z limit, so these shapes run the outer stride loop (z blocks 0 and 1 own two outer rows) as
// well as uneven shard tails: DeepRSmallK and rotated end with a short shard, AlignedK uses an uneven balanced split.
TEST(CubReduction, Fp8RetainedFamiliesMapWorkPast65535OuterRows) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    using CubReductionInternal::CubReductionStageCombine;
    using CubReductionInternal::CubReductionStageInputTransform;
    constexpr uint64_t outer_size = 65537;

    for (DataType dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
        SCOPED_TRACE(static_cast<int>(dtype));

        {  // DeepRSmallK: rows 64, R = 65 -> shards of 64 and 1 rows.
            SCOPED_TRACE("DeepRSmallK");
            constexpr uint64_t reduction_size = 65;
            constexpr uint64_t rows = 64;
            constexpr uint64_t shards = 2;
            for (uint64_t inner_size : {uint64_t{3}, uint64_t{9}}) {  // tiny-K lanes and component-owned kernels
                SCOPED_TRACE(inner_size);
                Tensor input = makeGpuTensor(fp8GridMappingValues(outer_size, reduction_size, inner_size),
                                             {outer_size, reduction_size, inner_size}, stream, dtype);
                const CubReductionGeometry geometry = CubReduction::analyzeGeometry(input.getDimensions(), 1);
                Tensor partials(gpuPlacement, TensorDescriptor(DataType::FP32, {outer_size, shards, inner_size}));
                for (const bool maximum : {false, true}) {
                    CubReductionInternal::CubReductionStageSemantics semantics{};
                    semantics.combine = maximum ? CubReductionStageCombine::Maximum : CubReductionStageCombine::Sum;
                    CubReductionInternal::launchFp8DeepRSmallKRetainedFirstStage(
                        semantics, input, partials, geometry, rows, shards, stream);
                    expectFloatVectorNear(
                        copyGpuTensorAsFloat(partials, stream),
                        fp8GridMappingExpected(outer_size, inner_size, {{0, 64}, {64, 65}},
                                               maximum ? Fp8GridMappingOp::Max : Fp8GridMappingOp::Sum),
                        0.0f);
                }
            }
        }

        {  // AlignedK: hoisted staged (Identity), typed staged (L1 = ABS + SUM), unified complete (MAX) and typed
           // complete (SUM) kernels.
            SCOPED_TRACE("AlignedK");
            constexpr uint64_t reduction_size = 7;
            constexpr uint64_t inner_size = 8;
            Tensor input = makeGpuTensor(fp8GridMappingValues(outer_size, reduction_size, inner_size),
                                         {outer_size, reduction_size, inner_size}, stream, dtype);
            const CubReductionGeometry geometry = CubReduction::analyzeGeometry(input.getDimensions(), 1);
            constexpr uint64_t shards = 3;  // balanced split 3, 2, 2
            Tensor partials(gpuPlacement, TensorDescriptor(DataType::FP32, {outer_size, shards, inner_size}));
            Tensor output(gpuPlacement, TensorDescriptor(DataType::FP32, geometry.output_dimensions));
            struct Case {
                CubReductionInternal::TiledRKStageTopology topology;
                size_t packet_bytes;
                uint32_t block_threads;
                CubReductionOp op;
            };
            const std::array<Case, 4> cases = {{
                {CubReductionInternal::TiledRKStageTopology::Staged, 8, 32, CubReductionOp::Sum},
                {CubReductionInternal::TiledRKStageTopology::Staged, 2, 64, CubReductionOp::L1Norm},
                {CubReductionInternal::TiledRKStageTopology::Complete, 8, 32, CubReductionOp::Max},
                {CubReductionInternal::TiledRKStageTopology::Complete, 8, 64, CubReductionOp::Sum},
            }};
            for (const Case& c : cases) {
                SCOPED_TRACE(c.packet_bytes);
                SCOPED_TRACE(c.block_threads);
                const bool staged = c.topology == CubReductionInternal::TiledRKStageTopology::Staged;
                const CubReductionInternal::KParallelTiledStagePlan plan{
                    c.topology, c.packet_bytes, c.block_threads, staged ? shards : 1};
                const auto semantics = CubReductionInternal::makeValueReductionStageSemantics(
                    c.op,
                    staged ? CubReductionInternal::CubReductionStageRole::First
                           : CubReductionInternal::CubReductionStageRole::Complete,
                    reduction_size);
                Tensor& destination = staged ? partials : output;
                CubReductionInternal::launchKParallelTiledStage(
                    semantics, input, destination, geometry, plan, 1.0f, stream);
                const std::vector<std::pair<uint64_t, uint64_t>> rows =
                    staged ? std::vector<std::pair<uint64_t, uint64_t>>{{0, 3}, {3, 5}, {5, 7}}
                           : std::vector<std::pair<uint64_t, uint64_t>>{{0, 7}};
                const Fp8GridMappingOp expected_op = c.op == CubReductionOp::Max      ? Fp8GridMappingOp::Max
                                                     : c.op == CubReductionOp::L1Norm ? Fp8GridMappingOp::AbsSum
                                                                                      : Fp8GridMappingOp::Sum;
                expectFloatVectorNear(copyGpuTensorAsFloat(destination, stream),
                                      fp8GridMappingExpected(outer_size, inner_size, rows, expected_op), 0.0f);
            }
        }

        {  // RuntimeK rotated: rows 32, R = 40 -> shards of 32 and 8 rows; awkward K, exact and prefix tiles.
            SCOPED_TRACE("RuntimeK rotated");
            constexpr uint64_t reduction_size = 40;
            constexpr uint64_t inner_size = 5;
            constexpr uint64_t rows = 32;
            constexpr uint64_t shards = 2;
            Tensor input = makeGpuTensor(fp8GridMappingValues(outer_size, reduction_size, inner_size),
                                         {outer_size, reduction_size, inner_size}, stream, dtype);
            const CubReductionGeometry geometry = CubReduction::analyzeGeometry(input.getDimensions(), 1);
            Tensor partials(gpuPlacement, TensorDescriptor(DataType::FP32, {outer_size, shards, inner_size}));
            for (const bool exact_tile : {false, true}) {
                SCOPED_TRACE(exact_tile);
                CubReductionInternal::launchFp8GeneralRParallelKTileRotatedCalibrationFirstStage(
                    input, partials, geometry, exact_tile, rows, shards, stream,
                    CubReductionStageInputTransform::Identity, CubReductionStageCombine::Sum);
                expectFloatVectorNear(
                    copyGpuTensorAsFloat(partials, stream),
                    fp8GridMappingExpected(outer_size, inner_size, {{0, 32}, {32, 40}}, Fp8GridMappingOp::Sum), 0.0f);
            }
        }
    }
}

// RuntimeK packet-complete packs W-warp outputs into persistent blocks and splits the flat [outer, K tile] index.
// W8 with 4 K tiles and 16500 outer rows is 66000 work items on a 65535-block grid, so the persistent loop wraps;
// W1 packs eight work items per block across outer boundaries.
TEST(CubReduction, Fp8RuntimeKPacketCompleteCoversOuterAndKTilesAcrossPersistentGrid) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    constexpr uint64_t reduction_size = 3;
    constexpr uint64_t inner_size = 200;      // tiles of 64, 64, 64 and 8 components
    constexpr uint32_t tile_components = 64;
    for (const auto& [outer_size, warps_per_output] :
         std::vector<std::pair<uint64_t, uint32_t>>{{16500, 8}, {1001, 1}, {37, 2}}) {
        SCOPED_TRACE(outer_size);
        SCOPED_TRACE(warps_per_output);
        for (DataType dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
            SCOPED_TRACE(static_cast<int>(dtype));
            Tensor input = makeGpuTensor(fp8GridMappingValues(outer_size, reduction_size, inner_size),
                                         {outer_size, reduction_size, inner_size}, stream, dtype);
            const CubReductionGeometry geometry = CubReduction::analyzeGeometry(input.getDimensions(), 1);
            Tensor output(gpuPlacement, TensorDescriptor(DataType::FP32, geometry.output_dimensions));
            CubReductionInternal::launchFp8GeneralRParallelKTileRCooperativeCalibrationComplete(
                input, output, geometry, warps_per_output, tile_components, stream);
            expectFloatVectorNear(
                copyGpuTensorAsFloat(output, stream),
                fp8GridMappingExpected(outer_size, inner_size, {{0, reduction_size}}, Fp8GridMappingOp::Sum), 0.0f);
        }
    }
}
#endif

TEST(CubReduction, AwkwardRotatedRCooperativeFirstStageIsAnIndependentPhysicalPass) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t outer_size = 1;
    constexpr uint64_t reduction_size = 256;
    constexpr uint64_t inner_size = 37;
    constexpr uint64_t rows_per_shard = 128;
    constexpr uint64_t shards_per_output = 2;

    std::vector<float> values;
    values.reserve(outer_size * reduction_size * inner_size);
    for (uint64_t row = 0; row < reduction_size; ++row) {
        const float value = row < rows_per_shard ? -2.0f : 3.0f;
        for (uint64_t component = 0; component < inner_size; ++component) {
            values.push_back(value);
        }
    }

    Tensor input = makeGpuTensor(values, {outer_size, reduction_size, inner_size}, stream, DataType::FP32);
    const CubReductionGeometry geometry = CubReduction::analyzeGeometry(input.getDimensions(), 1);
    ASSERT_EQ(geometry.path, CubReductionPath::TiledFixedSegment);
    Tensor partials(gpuPlacement,
                    TensorDescriptor(DataType::FP32, {outer_size, shards_per_output, inner_size}));

    // Deliberately use Complete L2 semantics. This physical stage must consume only the input-side Square transform
    // and associative Sum, leaving sqrt/final output handling to whatever reducer the overall planner chooses next.
    const CubReductionInternal::CubReductionStageSemantics semantics =
        CubReductionInternal::makeValueReductionStageSemantics(
            CubReductionOp::L2Norm, CubReductionInternal::CubReductionStageRole::Complete, reduction_size);
    ASSERT_EQ(semantics.finalize, CubReductionInternal::CubReductionStageFinalize::SquareRoot);

    CubReductionInternal::launchAwkwardAlignmentRotatedShardedFirstStage(
        semantics, input, partials, geometry, rows_per_shard, shards_per_output, stream);

    std::vector<float> expected;
    expected.reserve(shards_per_output * inner_size);
    expected.insert(expected.end(), inner_size, 128.0f * 4.0f);
    expected.insert(expected.end(), inner_size, 128.0f * 9.0f);
    expectFloatVectorNear(copyGpuTensorAsFloat(partials, stream), expected, 0.0f);
}

TEST(CubReduction, KParallelTiledCompleteTopologySupportsProductMinAndMax) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t outer_size = 1;
    constexpr uint64_t reduction_size = 3;
    constexpr uint64_t inner_size = 8;
    std::vector<float> values;
    values.reserve(outer_size * reduction_size * inner_size);
    for (uint64_t row = 0; row < reduction_size; ++row) {
        for (uint64_t component = 0; component < inner_size; ++component) {
            values.push_back(static_cast<float>(row + 1));
        }
    }

    Tensor input = makeGpuTensor(values, {outer_size, reduction_size, inner_size}, stream, DataType::BF16);
    const CubReductionGeometry geometry = CubReduction::analyzeGeometry(input.getDimensions(), 1);
    ASSERT_EQ(geometry.path, CubReductionPath::TiledFixedSegment);
    const CubReductionInternal::KParallelTiledStagePlan plan{
        CubReductionInternal::TiledRKStageTopology::Complete, 16, 32, 1};

    struct Case {
        CubReductionOp op;
        float expected;
    };
    for (const Case c : std::vector<Case>{{CubReductionOp::Product, 6.0f},
                                          {CubReductionOp::Min, 1.0f},
                                          {CubReductionOp::Max, 3.0f}}) {
        SCOPED_TRACE(static_cast<int>(c.op));
        Tensor output(gpuPlacement, TensorDescriptor(DataType::FP32, {outer_size, inner_size}));
        const CubReductionInternal::CubReductionStageSemantics semantics =
            CubReductionInternal::makeValueReductionStageSemantics(
                c.op, CubReductionInternal::CubReductionStageRole::Complete, reduction_size);
        CubReductionInternal::launchKParallelTiledStage(
            semantics, input, output, geometry, plan, 1.0f, stream);
        expectFloatVectorNear(copyGpuTensorAsFloat(output, stream), std::vector<float>(inner_size, c.expected), 0.0f);
    }
}

TEST(CubReduction, DefinesExplicitFp32EmptyReductionValues) {
    EXPECT_FLOAT_EQ(CubReduction::getFp32EmptyReductionValue(CubReductionOp::Sum), 0.0f);
    EXPECT_FLOAT_EQ(CubReduction::getFp32EmptyReductionValue(CubReductionOp::Product), 1.0f);
    EXPECT_FLOAT_EQ(CubReduction::getFp32EmptyReductionValue(CubReductionOp::Mean), 0.0f);
    EXPECT_FLOAT_EQ(CubReduction::getFp32EmptyReductionValue(CubReductionOp::L1Norm), 0.0f);
    EXPECT_FLOAT_EQ(CubReduction::getFp32EmptyReductionValue(CubReductionOp::L2Norm), 0.0f);
    EXPECT_FLOAT_EQ(CubReduction::getFp32EmptyReductionValue(CubReductionOp::SumSquares), 0.0f);
    EXPECT_EQ(CubReduction::getFp32EmptyReductionValue(CubReductionOp::Min),
              std::numeric_limits<float>::infinity());
    EXPECT_EQ(CubReduction::getFp32EmptyReductionValue(CubReductionOp::Max),
              -std::numeric_limits<float>::infinity());
}

TEST(CubReduction, WholeTensorPathSupportsEveryValueOperation) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor input = makeGpuTensor({-2.0f, 3.0f, -4.0f}, {3}, stream);

    expectOperations(input,
                     0,
                     {{CubReductionOp::Sum, {-3.0f}, 0.0f},
                      {CubReductionOp::Product, {24.0f}, 0.0f},
                      {CubReductionOp::Mean, {-1.0f}, 0.0f},
                      {CubReductionOp::Min, {-4.0f}, 0.0f},
                      {CubReductionOp::Max, {3.0f}, 0.0f},
                      {CubReductionOp::L1Norm, {9.0f}, 0.0f},
                      {CubReductionOp::L2Norm, {std::sqrt(29.0f)}, 1.0e-5f},
                      {CubReductionOp::SumSquares, {29.0f}, 0.0f}},
                     stream);
}

TEST(CubReduction, CompactPhysicalPermutationFullReductionSupportsEveryValueOperation) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    // Logical [2,3,4] uses physical order [1,2,0]. Every storage value participates exactly once, so a full value
    // reduction is invariant to that permutation and can consume the compact physical span directly.
    std::vector<float> values;
    values.reserve(24);
    for (int i = 0; i < 12; ++i) {
        values.push_back(-1.0f);
        values.push_back(1.0f);
    }
    Tensor storage = makeGpuTensor(values, {3, 4, 2}, stream);
    Tensor permuted = storage.aliasView({2, 3, 4}, {1, 8, 2}, 0);
    const std::vector<uint32_t> axes = {0, 1, 2};

    const std::vector<OperationExpectation> expectations = {
        {CubReductionOp::Sum, {0.0f}, 0.0f},
        {CubReductionOp::Product, {1.0f}, 0.0f},
        {CubReductionOp::Mean, {0.0f}, 0.0f},
        {CubReductionOp::Min, {-1.0f}, 0.0f},
        {CubReductionOp::Max, {1.0f}, 0.0f},
        {CubReductionOp::L1Norm, {24.0f}, 0.0f},
        {CubReductionOp::L2Norm, {std::sqrt(24.0f)}, 1.0e-5f},
        {CubReductionOp::SumSquares, {24.0f}, 0.0f},
    };

    for (const OperationExpectation& expectation : expectations) {
        SCOPED_TRACE(static_cast<int>(expectation.op));
        std::shared_ptr<StampedCubReduction> stamped = CubReduction(expectation.op, axes).stamp(permuted, stream);
        ASSERT_EQ(stamped->getPath(), CubReductionPath::DeviceTransformReduce);
        stamped->runOn(stream);
        stream.synchronize();
        expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream),
                              expectation.expected,
                              expectation.tolerance);
    }
}

TEST(CubReduction, CompactPhysicalPermutationTrailingReductionUsesContiguousSegmentsForEveryValueOperation) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    // Physical storage is [axis1=2, axis2=3, axis0=2]. The logical [axis0,axis1,axis2] view therefore has physical
    // order [1,2,0]. Reducing axis0 consumes six ordinary contiguous 2-value segments, and physical retained order
    // [1,2] is already the logical dense output order.
    Tensor storage = makeGpuTensor(
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f},
        {2, 3, 2},
        stream);
    Tensor permuted = storage.aliasView({2, 2, 3}, {1, 6, 2}, 0);

    const std::vector<OperationExpectation> expectations = {
        {CubReductionOp::Sum, {3.0f, 7.0f, 11.0f, 15.0f, 19.0f, 23.0f}, 0.0f},
        {CubReductionOp::Product, {2.0f, 12.0f, 30.0f, 56.0f, 90.0f, 132.0f}, 0.0f},
        {CubReductionOp::Mean, {1.5f, 3.5f, 5.5f, 7.5f, 9.5f, 11.5f}, 0.0f},
        {CubReductionOp::Min, {1.0f, 3.0f, 5.0f, 7.0f, 9.0f, 11.0f}, 0.0f},
        {CubReductionOp::Max, {2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f}, 0.0f},
        {CubReductionOp::L1Norm, {3.0f, 7.0f, 11.0f, 15.0f, 19.0f, 23.0f}, 0.0f},
        {CubReductionOp::L2Norm,
         {std::sqrt(5.0f), std::sqrt(25.0f), std::sqrt(61.0f), std::sqrt(113.0f), std::sqrt(181.0f),
          std::sqrt(265.0f)},
         1.0e-5f},
        {CubReductionOp::SumSquares, {5.0f, 25.0f, 61.0f, 113.0f, 181.0f, 265.0f}, 0.0f},
    };

    for (const OperationExpectation& expectation : expectations) {
        SCOPED_TRACE(static_cast<int>(expectation.op));
        std::shared_ptr<StampedCubReduction> stamped = CubReduction(expectation.op, 0).stamp(permuted, stream);
        ASSERT_EQ(stamped->getPath(), CubReductionPath::ContiguousFixedSegment);
        EXPECT_TRUE(stamped->getGeometry().permutation_aware_contiguous_segments);
        stamped->runOn(stream);
        stream.synchronize();
        expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream),
                              expectation.expected,
                              expectation.tolerance);
    }
}

TEST(CubReduction, ContiguousFixedSegmentPathSupportsEveryValueOperation) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor input = makeGpuTensor({-1.0f, 2.0f, -3.0f, 4.0f, -5.0f, 6.0f}, {2, 3}, stream);

    expectOperations(input,
                     1,
                     {{CubReductionOp::Sum, {-2.0f, 5.0f}, 0.0f},
                      {CubReductionOp::Product, {6.0f, -120.0f}, 0.0f},
                      {CubReductionOp::Mean, {-2.0f / 3.0f, 5.0f / 3.0f}, 1.0e-6f},
                      {CubReductionOp::Min, {-3.0f, -5.0f}, 0.0f},
                      {CubReductionOp::Max, {2.0f, 6.0f}, 0.0f},
                      {CubReductionOp::L1Norm, {6.0f, 15.0f}, 0.0f},
                      {CubReductionOp::L2Norm, {std::sqrt(14.0f), std::sqrt(77.0f)}, 1.0e-5f},
                      {CubReductionOp::SumSquares, {14.0f, 77.0f}, 0.0f}},
                     stream);
}

TEST(CubReduction, TiledFixedSegmentPathSupportsEveryValueOperation) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor input = makeGpuTensor(
        {-1.0f, 2.0f, -3.0f, 4.0f, -5.0f, 6.0f, -7.0f, 8.0f, -9.0f, 10.0f, -11.0f, 12.0f},
        {2, 3, 2},
        stream);

    expectOperations(input,
                     1,
                     {{CubReductionOp::Sum, {-9.0f, 12.0f, -27.0f, 30.0f}, 0.0f},
                      {CubReductionOp::Product, {-15.0f, 48.0f, -693.0f, 960.0f}, 0.0f},
                      {CubReductionOp::Mean, {-3.0f, 4.0f, -9.0f, 10.0f}, 0.0f},
                      {CubReductionOp::Min, {-5.0f, 2.0f, -11.0f, 8.0f}, 0.0f},
                      {CubReductionOp::Max, {-1.0f, 6.0f, -7.0f, 12.0f}, 0.0f},
                      {CubReductionOp::L1Norm, {9.0f, 12.0f, 27.0f, 30.0f}, 0.0f},
                      {CubReductionOp::L2Norm,
                       {std::sqrt(35.0f), std::sqrt(56.0f), std::sqrt(251.0f), std::sqrt(308.0f)},
                       1.0e-5f},
                      {CubReductionOp::SumSquares, {35.0f, 56.0f, 251.0f, 308.0f}, 0.0f}},
                     stream);
}

TEST(CubReduction, PermutationAwareTiledPathSupportsEveryValueOperation) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor storage = makeGpuTensor(
        {-1.0f, 2.0f, -3.0f, 4.0f, -5.0f, 6.0f, -7.0f, 8.0f, -9.0f, 10.0f, -11.0f, 12.0f},
        {2, 3, 2},
        stream);
    Tensor permuted = storage.aliasView({2, 2, 3}, {1, 6, 2}, 0);

    expectOperations(permuted,
                     2,
                     {{CubReductionOp::Sum, {-9.0f, -27.0f, 12.0f, 30.0f}, 0.0f},
                      {CubReductionOp::Product, {-15.0f, -693.0f, 48.0f, 960.0f}, 0.0f},
                      {CubReductionOp::Mean, {-3.0f, -9.0f, 4.0f, 10.0f}, 0.0f},
                      {CubReductionOp::Min, {-5.0f, -11.0f, 2.0f, 8.0f}, 0.0f},
                      {CubReductionOp::Max, {-1.0f, -7.0f, 6.0f, 12.0f}, 0.0f},
                      {CubReductionOp::L1Norm, {9.0f, 27.0f, 12.0f, 30.0f}, 0.0f},
                      {CubReductionOp::L2Norm,
                       {std::sqrt(35.0f), std::sqrt(251.0f), std::sqrt(56.0f), std::sqrt(308.0f)},
                       1.0e-5f},
                      {CubReductionOp::SumSquares, {35.0f, 251.0f, 56.0f, 308.0f}, 0.0f}},
                     stream);
}

TEST(CubReduction, TiledFixedSegmentCanCoalescePermutedDenseOutputWithoutIntermediate) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t outer = 35;
    constexpr uint64_t reduction = 3;
    constexpr uint64_t inner = 37;
    std::vector<float> values(outer * reduction * inner);
    std::vector<float> expected(inner * outer, 0.0f);
    for (uint64_t i = 0; i < outer; ++i) {
        for (uint64_t j = 0; j < reduction; ++j) {
            for (uint64_t k = 0; k < inner; ++k) {
                const float value = static_cast<float>((i % 7) * 0.5 + j * 2.0 + (k % 11) * 0.25);
                values[(i * reduction + j) * inner + k] = value;
                expected[k * outer + i] += value;
            }
        }
    }

    Tensor storage = makeGpuTensor(values, {outer, reduction, inner}, stream);
    Tensor permuted = storage.aliasView(
        {inner, outer, reduction}, {1, reduction * inner, inner}, 0);
    Tensor squeezed_output(gpuPlacement, TensorDescriptor(DataType::FP32, {inner, outer}));
    std::shared_ptr<StampedCubReduction> stamped =
        CubReduction(CubReductionOp::Sum, 2, DataType::FP32).stamp(permuted, squeezed_output, stream);

    ASSERT_EQ(stamped->getPath(), CubReductionPath::TiledFixedSegment);
    EXPECT_TRUE(stamped->getGeometry().tiled_output_permuted);
    EXPECT_TRUE(stamped->getGeometry().tiled_output_shared_transpose);
    EXPECT_EQ(stamped->getOutputTensor().getDimensions(), (std::vector<uint64_t>{inner, outer}));
    EXPECT_EQ(stamped->getOutputTensor().getMemPtr<void>(), squeezed_output.getMemPtr<void>());

    stamped->run();
    stream.synchronize();
    expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), expected);
}

TEST(CubReduction, PermutationAwareExactWarpWidthHandlesShortReduction) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t outer = 11;
    constexpr uint64_t reduction = 2;
    constexpr uint64_t inner = 32;
    std::vector<float> values(outer * reduction * inner);
    std::vector<float> expected(inner * outer, 0.0f);
    for (uint64_t i = 0; i < outer; ++i) {
        for (uint64_t j = 0; j < reduction; ++j) {
            for (uint64_t k = 0; k < inner; ++k) {
                const float value = static_cast<float>(static_cast<int>((i * 3 + j * 5 + k * 7) % 13) - 6);
                values[(i * reduction + j) * inner + k] = value;
                expected[k * outer + i] += value;
            }
        }
    }

    Tensor storage = makeGpuTensor(values, {outer, reduction, inner}, stream);
    Tensor permuted = storage.aliasView(
        {inner, outer, reduction}, {1, reduction * inner, inner}, 0);
    std::shared_ptr<StampedCubReduction> stamped =
        CubReduction(CubReductionOp::Sum, 2, DataType::FP32).stamp(permuted, stream);

    ASSERT_EQ(stamped->getPath(), CubReductionPath::TiledFixedSegment);
    ASSERT_TRUE(stamped->getGeometry().tiled_output_shared_transpose);
    stamped->run();
    stream.synchronize();
    expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), expected);
}

TEST(CubReduction, SharedTransposePermutedTiledWriterSupportsStorageDTypesAndTailTiles) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t outer = 513;  // enough retained tiles to exercise one-warp/component-tile production geometry
    constexpr uint64_t reduction = 2;
    constexpr uint64_t inner = 65;   // both component and outer dimensions have partial final tiles
    std::vector<float> values(outer * reduction * inner);
    std::vector<float> expected(inner * outer, 0.0f);
    for (uint64_t i = 0; i < outer; ++i) {
        for (uint64_t j = 0; j < reduction; ++j) {
            for (uint64_t k = 0; k < inner; ++k) {
                const float value = static_cast<float>(static_cast<int>((i + j * 3 + k * 5) % 9) - 4) * 0.5f;
                values[(i * reduction + j) * inner + k] = value;
                expected[k * outer + i] += value;
            }
        }
    }

    std::vector<DataType> input_dtypes = {DataType::FP16, DataType::BF16, DataType::FP32};
#if THOR_CUB_ENABLE_FP8_TYPES
    input_dtypes.push_back(DataType::FP8_E4M3);
    input_dtypes.push_back(DataType::FP8_E5M2);
#endif
#if THOR_CUB_ENABLE_64BIT_TYPES
    input_dtypes.push_back(DataType::FP64);
#endif
    for (DataType dtype : input_dtypes) {
        SCOPED_TRACE(static_cast<int>(dtype));
        Tensor storage = makeGpuTensor(values, {outer, reduction, inner}, stream, dtype);
        Tensor input = storage.aliasView(
            {inner, outer, reduction}, {1, reduction * inner, inner}, 0);
        std::shared_ptr<StampedCubReduction> stamped =
            CubReduction(CubReductionOp::Sum, 2, DataType::FP32).stamp(input, stream);
        ASSERT_TRUE(stamped->getGeometry().tiled_output_shared_transpose);
        stamped->run();
        stream.synchronize();
        expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), expected, 1.0e-4f);
    }
}

TEST(CubReduction, SharedTransposePermutedTiledWriterCoversRetainedWidthRegimes) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t outer = 9;
    constexpr uint64_t reduction = 3;
    for (uint64_t inner : {2ULL,
                           31ULL,
                           32ULL,
                           33ULL,
                           64ULL,
                           65ULL,
                           128ULL,
                           129ULL,
                           256ULL,
                           257ULL,
                           511ULL,
                           512ULL,
                           513ULL,
                           1024ULL,
                           2048ULL,
                           4096ULL,
                           4097ULL,
                           8193ULL}) {
        SCOPED_TRACE(inner);
        std::vector<float> values(outer * reduction * inner);
        std::vector<float> expected(inner * outer, 0.0f);
        for (uint64_t i = 0; i < outer; ++i) {
            for (uint64_t j = 0; j < reduction; ++j) {
                for (uint64_t k = 0; k < inner; ++k) {
                    const float value = static_cast<float>(static_cast<int>((i * 7 + j * 5 + k * 3) % 17) - 8)
                                        * 0.125f;
                    values[(i * reduction + j) * inner + k] = value;
                    expected[k * outer + i] += value;
                }
            }
        }

        Tensor storage = makeGpuTensor(values, {outer, reduction, inner}, stream);
        Tensor permuted = storage.aliasView(
            {inner, outer, reduction}, {1, reduction * inner, inner}, 0);
        std::shared_ptr<StampedCubReduction> stamped =
            CubReduction(CubReductionOp::Sum, 2, DataType::FP32).stamp(permuted, stream);
        ASSERT_EQ(stamped->getPath(), CubReductionPath::TiledFixedSegment);
        ASSERT_TRUE(stamped->getGeometry().tiled_output_shared_transpose);

        stamped->run();
        stream.synchronize();
        expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), expected, 1.0e-5f);
    }
}

TEST(CubReduction, SharedTransposePermutedTiledWriterHonorsDenseViewStorageOffset) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t outer = 9;
    constexpr uint64_t reduction = 5;
    constexpr uint64_t inner = 35;
    constexpr uint64_t offset = 17;
    const uint64_t visible_elements = outer * reduction * inner;
    std::vector<float> storage_values(offset + visible_elements + 11, -999.0f);
    std::vector<float> expected(inner * outer, 0.0f);
    for (uint64_t i = 0; i < outer; ++i) {
        for (uint64_t j = 0; j < reduction; ++j) {
            for (uint64_t k = 0; k < inner; ++k) {
                const float value = static_cast<float>(static_cast<int>((i * 3 + j * 7 + k) % 11) - 5);
                storage_values[offset + (i * reduction + j) * inner + k] = value;
                expected[k * outer + i] += value;
            }
        }
    }

    Tensor storage = makeGpuTensor(storage_values, {storage_values.size()}, stream);
    Tensor input = storage.aliasView(
        {inner, outer, reduction}, {1, reduction * inner, inner}, offset);
    std::shared_ptr<StampedCubReduction> stamped =
        CubReduction(CubReductionOp::Sum, 2, DataType::FP32).stamp(input, stream);
    ASSERT_TRUE(stamped->getGeometry().tiled_output_shared_transpose);

    stamped->run();
    stream.synchronize();
    expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), expected);
}

TEST(CubReduction, SharedTransposePermutedTiledWriterGridStridesPast65535OuterRows) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t outer = 65537;
    constexpr uint64_t reduction = 2;
    constexpr uint64_t inner = 33;
    std::vector<float> values(outer * reduction * inner, 1.0f);
    Tensor storage = makeGpuTensor(values, {outer, reduction, inner}, stream);
    Tensor input = storage.aliasView(
        {inner, outer, reduction}, {1, reduction * inner, inner}, 0);
    std::shared_ptr<StampedCubReduction> stamped =
        CubReduction(CubReductionOp::Sum, 2, DataType::FP32).stamp(input, stream);
    ASSERT_TRUE(stamped->getGeometry().tiled_output_shared_transpose);

    stamped->run();
    stream.synchronize();
    expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream),
                          std::vector<float>(inner * outer, 2.0f));
}

TEST(CubReduction, TiledFixedSegmentCoversWidthPoliciesAndAsyncStaging) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t outer_size = 2;
    constexpr uint64_t reduction_size = 65;
    for (uint64_t inner_size :
         {2ULL, 3ULL, 4ULL, 5ULL, 8ULL, 9ULL, 16ULL, 17ULL, 31ULL, 32ULL, 33ULL, 63ULL, 64ULL, 65ULL, 127ULL, 128ULL, 129ULL, 255ULL, 256ULL, 257ULL, 511ULL, 512ULL, 513ULL, 1023ULL, 1024ULL, 1025ULL, 2047ULL, 2048ULL, 2049ULL, 4095ULL, 4096ULL, 4097ULL}) {
        SCOPED_TRACE(inner_size);
        std::vector<float> values(outer_size * reduction_size * inner_size);
        std::vector<float> expected(outer_size * inner_size, 0.0f);
        for (uint64_t outer = 0; outer < outer_size; ++outer) {
            for (uint64_t row = 0; row < reduction_size; ++row) {
                for (uint64_t component = 0; component < inner_size; ++component) {
                    const int value = static_cast<int>((outer * 3 + row * 5 + component * 7) % 11) - 5;
                    const uint64_t input_index = (outer * reduction_size + row) * inner_size + component;
                    values[input_index] = static_cast<float>(value);
                    expected[outer * inner_size + component] += static_cast<float>(value);
                }
            }
        }

        Tensor input = makeGpuTensor(values, {outer_size, reduction_size, inner_size}, stream);
        std::shared_ptr<StampedCubReduction> stamped =
            CubReduction(CubReductionOp::Sum, 1, DataType::FP32).stamp(input, stream);
        EXPECT_EQ(stamped->getPath(), CubReductionPath::TiledFixedSegment);
        // Workspace is an implementation detail of the selected physical plan. Complete modern RK passes require no
        // persistent partial buffer, while staged plans do; correctness below is the invariant this width sweep owns.
        stamped->run();
        stream.synchronize();
        expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), expected);
    }
}

TEST(CubReduction, ShallowOddLowPrecisionNonAdditiveRKUsesGenericTiledFallback) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t outer_size = 2;
    constexpr uint64_t reduction_size = 2;
    constexpr uint64_t inner_size = 3;
    const std::vector<float> values = {
        1.0f, 5.0f, -2.0f,
        4.0f, 3.0f,  7.0f,
        9.0f, 0.0f,  2.0f,
        6.0f, 8.0f,  1.0f,
    };
    const std::vector<float> expected = {4.0f, 5.0f, 7.0f, 9.0f, 8.0f, 2.0f};

    for (DataType dtype : {DataType::FP16, DataType::BF16}) {
        SCOPED_TRACE(static_cast<int>(dtype));
        Tensor input = makeGpuTensor(values, {outer_size, reduction_size, inner_size}, stream, dtype);
        CubReduction reduction(CubReductionOp::Max, 1, DataType::FP32);
        EXPECT_EQ(reduction.queryWorkspaceSizeInBytes(input.getDescriptor(), stream), 0U);
        std::shared_ptr<StampedCubReduction> stamped = reduction.stamp(input, stream);
        EXPECT_EQ(stamped->getPath(), CubReductionPath::TiledFixedSegment);
        EXPECT_FALSE(stamped->usesModernRKFamilyPlan());
        EXPECT_EQ(stamped->getWorkspaceSizeInBytes(), 0U);

        stamped->run();
        stream.synchronize();
        expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), expected);
    }
}

TEST(CubReduction, PacketAdaptiveCooperativeTiledPathPreservesAdditiveTransformsAndFinalizers) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t outer_size = 1;
    constexpr uint64_t reduction_size = 205;
    constexpr uint64_t inner_size = 128;
    Tensor input = makeGpuTensor(std::vector<float>(outer_size * reduction_size * inner_size, 1.0f),
                                 {outer_size, reduction_size, inner_size},
                                 stream,
                                 DataType::BF16);

    // This is the ordinary second-stage geometry produced by a 512-row shard of the deep transformer reduction.
    // It is inside the release-calibrated packet-adaptive cooperative gate (8-byte BF16 packets, many cooperating
    // warps) but still exercises the generic additive transforms/finalizers rather than a Sum-only special case.
    const std::vector<OperationExpectation> expectations = {
        {CubReductionOp::Sum, std::vector<float>(inner_size, 205.0f), 0.0f},
        {CubReductionOp::Mean, std::vector<float>(inner_size, 1.0f), 0.0f},
        {CubReductionOp::L1Norm, std::vector<float>(inner_size, 205.0f), 0.0f},
        {CubReductionOp::L2Norm, std::vector<float>(inner_size, std::sqrt(205.0f)), 1.0e-4f},
        {CubReductionOp::SumSquares, std::vector<float>(inner_size, 205.0f), 0.0f},
    };
    expectOperations(input, 1, expectations, stream);
}

// 1227: was ModernDenseRKProductionExposesBothRCooperativeAccessModes; its aligned and rotated cases were FP32, which
// the retained FP32 RK passes now run (CubReductionFp32Routing covers them).
TEST(CubReduction, ModernDenseRKProductionRunsLowPrecisionFlatRows) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    struct Case {
        DataType dtype;
        std::vector<uint64_t> dimensions;
        std::string expected_fragment;
    };
    const std::vector<Case> cases = {
        {DataType::BF16, {64, 32768, 17}, "r_cooperative:flat_rows"},
    };

    for (const Case& c : cases) {
        SCOPED_TRACE(c.expected_fragment);
        Tensor input(TensorPlacement(TensorPlacement::MemDevices::GPU, stream.getGpuNum()),
                     TensorDescriptor(c.dtype, c.dimensions));
        std::shared_ptr<StampedCubReduction> stamped =
            CubReduction(CubReductionOp::Sum, 1, DataType::FP32).stamp(input, stream);
        ASSERT_TRUE(stamped->usesDenseExecutablePlan());
        EXPECT_GT(stamped->getDenseExecutablePassCount(), 0U);
        ASSERT_TRUE(stamped->usesModernRKFamilyPlan());
        EXPECT_NE(stamped->getModernRKStrategyChain().find(c.expected_fragment), std::string::npos)
            << stamped->getModernRKStrategyChain();
    }
}

TEST(CubReduction, ModernDenseRKProductionOwnsFormerFullRowTransformerRegimes) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t outer_size = 1;
    constexpr uint64_t reduction_size = 4096;
    for (uint64_t inner_size : {512ULL, 768ULL}) {
        SCOPED_TRACE(inner_size);
        std::vector<float> values(outer_size * reduction_size * inner_size, 1.0f);
        Tensor input = makeGpuTensor(values, {outer_size, reduction_size, inner_size}, stream, DataType::BF16);
        CubReduction reduction(CubReductionOp::Sum, 1, DataType::FP32);
        std::shared_ptr<StampedCubReduction> stamped = reduction.stamp(input, stream);
        EXPECT_EQ(stamped->getPath(), CubReductionPath::TiledFixedSegment);
        EXPECT_TRUE(stamped->usesDenseExecutablePlan());
        EXPECT_GT(stamped->getDenseExecutablePassCount(), 0U);
        EXPECT_TRUE(stamped->usesModernRKFamilyPlan());
        EXPECT_FALSE(stamped->getModernRKStrategyChain().empty());
        EXPECT_EQ(stamped->getWorkspaceSizeInBytes(), reduction.queryWorkspaceSizeInBytes(input.getDescriptor(), stream));
        stamped->run();
        stream.synchronize();
        expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream),
                              std::vector<float>(inner_size, static_cast<float>(reduction_size)));
    }
}

TEST(CubReduction, DenseRKPreservesGenericReductionSemanticsAndFinalizers) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t reduction_size = 1024;
    constexpr uint64_t inner_size = 17;
    std::vector<float> values(reduction_size * inner_size);
    for (uint64_t row = 0; row < reduction_size; ++row) {
        const float value = (row & 1U) == 0 ? -1.0f : 1.0f;
        for (uint64_t component = 0; component < inner_size; ++component) {
            values[row * inner_size + component] = value;
        }
    }
    Tensor input = makeGpuTensor(values, {1, reduction_size, inner_size}, stream, DataType::BF16);

    const std::vector<OperationExpectation> expectations = {
        {CubReductionOp::Sum, std::vector<float>(inner_size, 0.0f), 0.0f},
        {CubReductionOp::Mean, std::vector<float>(inner_size, 0.0f), 0.0f},
        {CubReductionOp::L1Norm, std::vector<float>(inner_size, 1024.0f), 0.0f},
        {CubReductionOp::L2Norm, std::vector<float>(inner_size, 32.0f), 0.0f},
        {CubReductionOp::SumSquares, std::vector<float>(inner_size, 1024.0f), 0.0f},
        {CubReductionOp::Product, std::vector<float>(inner_size, 1.0f), 0.0f},
        {CubReductionOp::Min, std::vector<float>(inner_size, -1.0f), 0.0f},
        {CubReductionOp::Max, std::vector<float>(inner_size, 1.0f), 0.0f},
    };

    for (const OperationExpectation& expectation : expectations) {
        SCOPED_TRACE(static_cast<int>(expectation.op));
        CubReduction reduction(expectation.op, 1, DataType::FP32);
        std::shared_ptr<StampedCubReduction> stamped = reduction.stamp(input, stream);
        // K=17 is packet-misaligned. Some operation families may legitimately
        // use a generic physical pass rather than a modern tiled RK kernel.
        // The invariant here is identical mathematical semantics/finalization,
        // not a particular per-operation physical strategy.
        ASSERT_TRUE(stamped->usesDenseExecutablePlan());
        stamped->run();
        stream.synchronize();
        expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream),
                              expectation.expected,
                              expectation.tolerance);
    }
}

TEST(CubReduction, ModernDenseRKOwnsFormerRowSplitBoundaryShapes) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    for (const TensorDescriptor& descriptor : {
             TensorDescriptor(DataType::BF16, {129, 2048, 512}),
             TensorDescriptor(DataType::BF16, {1, 65, 512}),
             TensorDescriptor(DataType::BF16, {1, 32768, 1536}),
         }) {
        Tensor input(TensorPlacement(TensorPlacement::MemDevices::GPU, stream.getGpuNum()), descriptor);
        std::shared_ptr<StampedCubReduction> stamped =
            CubReduction(CubReductionOp::Sum, 1, DataType::FP32).stamp(input, stream);
        EXPECT_TRUE(stamped->usesModernRKFamilyPlan());
        EXPECT_FALSE(stamped->getModernRKStrategyChain().empty());
    }
}

TEST(CubReduction, PermutationAwareTiledPathCoversTunedWidthPolicies) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t outer_size = 3;
    constexpr uint64_t reduction_size = 65;
    for (uint64_t inner_size :
         {2ULL, 3ULL, 5ULL, 17ULL, 32ULL, 33ULL, 63ULL, 64ULL, 65ULL, 127ULL, 128ULL, 129ULL, 255ULL, 256ULL,
          257ULL, 511ULL, 512ULL, 513ULL, 1024ULL, 2048ULL, 4096ULL, 4097ULL}) {
        SCOPED_TRACE(inner_size);
        std::vector<float> values(outer_size * reduction_size * inner_size);
        std::vector<float> expected(inner_size * outer_size, 0.0f);
        for (uint64_t outer = 0; outer < outer_size; ++outer) {
            for (uint64_t row = 0; row < reduction_size; ++row) {
                for (uint64_t component = 0; component < inner_size; ++component) {
                    const int value = static_cast<int>((outer * 3 + row * 5 + component * 7) % 11) - 5;
                    values[(outer * reduction_size + row) * inner_size + component] = static_cast<float>(value);
                    expected[component * outer_size + outer] += static_cast<float>(value);
                }
            }
        }

        Tensor storage = makeGpuTensor(values, {outer_size, reduction_size, inner_size}, stream);
        Tensor permuted = storage.aliasView(
            {inner_size, outer_size, reduction_size},
            {1, reduction_size * inner_size, inner_size},
            0);
        std::shared_ptr<StampedCubReduction> stamped =
            CubReduction(CubReductionOp::Sum, 2, DataType::FP32).stamp(permuted, stream);
        ASSERT_EQ(stamped->getPath(), CubReductionPath::TiledFixedSegment);
        ASSERT_TRUE(stamped->getGeometry().permutation_aware_tiled_geometry.has_value());
        EXPECT_TRUE(stamped->getGeometry().tiled_output_permuted);
        EXPECT_EQ(stamped->getWorkspaceSizeInBytes(), 0U);
        stamped->run();
        stream.synchronize();
        expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), expected);
    }
}

TEST(CubReduction, TiledFixedSegmentBlockShardingScalesBeyondOneBlockWidth) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t outer_size = 1;
    constexpr uint64_t reduction_size = 7;
    for (uint64_t inner_size : {4097ULL, 6144ULL, 8191ULL, 8192ULL, 8193ULL, 16384ULL, 16385ULL, 65536ULL, 65537ULL}) {
        SCOPED_TRACE(inner_size);
        std::vector<float> values(outer_size * reduction_size * inner_size);
        std::vector<float> expected(outer_size * inner_size, 0.0f);
        for (uint64_t row = 0; row < reduction_size; ++row) {
            for (uint64_t component = 0; component < inner_size; ++component) {
                const int value = static_cast<int>((row * 5 + component * 7) % 11) - 5;
                values[row * inner_size + component] = static_cast<float>(value);
                expected[component] += static_cast<float>(value);
            }
        }

        Tensor input = makeGpuTensor(values, {outer_size, reduction_size, inner_size}, stream);
        std::shared_ptr<StampedCubReduction> stamped =
            CubReduction(CubReductionOp::Sum, 1, DataType::FP32).stamp(input, stream);
        EXPECT_EQ(stamped->getPath(), CubReductionPath::TiledFixedSegment);
        stamped->run();
        stream.synchronize();
        expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), expected);
    }
}

TEST(CubReduction, NewOperationsDefaultOutputStorageToInputDtype) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor input = makeGpuTensor({1.0f, 2.0f, 3.0f, -1.0f, -2.0f, -3.0f}, {2, 3}, stream, DataType::BF16);

    const std::vector<OperationExpectation> expectations = {
        {CubReductionOp::Product, {6.0f, -6.0f}, 0.0f},
        {CubReductionOp::Mean, {2.0f, -2.0f}, 0.0f},
        {CubReductionOp::L1Norm, {6.0f, 6.0f}, 0.0f},
        {CubReductionOp::L2Norm, {std::sqrt(14.0f), std::sqrt(14.0f)}, 2.0e-2f},
        {CubReductionOp::SumSquares, {14.0f, 14.0f}, 2.0e-2f},
    };

    for (const OperationExpectation& expectation : expectations) {
        SCOPED_TRACE(static_cast<int>(expectation.op));
        std::shared_ptr<StampedCubReduction> stamped = CubReduction(expectation.op, 1).stamp(input, stream);
        EXPECT_EQ(stamped->getOutputDataType(), DataType::BF16);
        stamped->run();
        stream.synchronize();
        expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream),
                              expectation.expected,
                              expectation.tolerance);
    }
}

TEST(CubReduction, MeanAndL2FinalizeInFp32BeforeFp16StorageConversion) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor mean_input = makeGpuTensor({40000.0f, 40000.0f, -40000.0f, -40000.0f}, {2, 2}, stream);
    std::shared_ptr<StampedCubReduction> mean =
        CubReduction(CubReductionOp::Mean, 1, DataType::FP16).stamp(mean_input, stream);
    mean->run();

    Tensor l2_input = makeGpuTensor({300.0f, 400.0f, 500.0f, 1200.0f}, {2, 2}, stream);
    std::shared_ptr<StampedCubReduction> l2 =
        CubReduction(CubReductionOp::L2Norm, 1, DataType::FP16).stamp(l2_input, stream);
    l2->run();
    stream.synchronize();

    expectFloatVectorNear(copyGpuTensorAsFloat(mean->getOutputTensor(), stream), {40000.0f, -40000.0f});
    expectFloatVectorNear(copyGpuTensorAsFloat(l2->getOutputTensor(), stream), {500.0f, 1300.0f});
}

TEST(CubReduction, MinimumAndMaximumPropagateNanAcrossAllPaths) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const float nan = std::numeric_limits<float>::quiet_NaN();

    Tensor whole = makeGpuTensor({1.0f, nan, -2.0f}, {3}, stream);
    Tensor contiguous = makeGpuTensor({nan, 1.0f, 2.0f, 3.0f, nan, 4.0f}, {2, 3}, stream);
    Tensor tiled = makeGpuTensor(
        {1.0f, 2.0f, nan, nan, 5.0f, 6.0f, 7.0f, 8.0f, nan, nan, 11.0f, 12.0f},
        {2, 3, 2},
        stream);

    for (CubReductionOp op : {CubReductionOp::Min, CubReductionOp::Max}) {
        SCOPED_TRACE(static_cast<int>(op));
        expectFloatVectorNear(executeFp32Output(whole, op, 0, stream), {nan});
        expectFloatVectorNear(executeFp32Output(contiguous, op, 1, stream), {nan, nan});
        expectFloatVectorNear(executeFp32Output(tiled, op, 1, stream), {nan, nan, nan, nan});
    }
}

TEST(CubReduction, ArithmeticAndNormOperationsNaturallyPropagateNan) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    Tensor input = makeGpuTensor({1.0f, nan, 3.0f}, {3}, stream);

    for (CubReductionOp op : {CubReductionOp::Sum,
                              CubReductionOp::Product,
                              CubReductionOp::Mean,
                              CubReductionOp::L1Norm,
                              CubReductionOp::L2Norm,
                              CubReductionOp::SumSquares}) {
        SCOPED_TRACE(static_cast<int>(op));
        expectFloatVectorNear(executeFp32Output(input, op, 0, stream), {nan});
    }
}

TEST(CubReduction, HandlesPositiveAndNegativeInfinity) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const float infinity = std::numeric_limits<float>::infinity();
    Tensor input = makeGpuTensor({-infinity, 1.0f, infinity}, {3}, stream);

    expectFloatVectorNear(executeFp32Output(input, CubReductionOp::Min, 0, stream), {-infinity});
    expectFloatVectorNear(executeFp32Output(input, CubReductionOp::Max, 0, stream), {infinity});
    expectFloatVectorNear(executeFp32Output(input, CubReductionOp::L1Norm, 0, stream), {infinity});
    expectFloatVectorNear(executeFp32Output(input, CubReductionOp::L2Norm, 0, stream), {infinity});
}

TEST(CubReduction, StampedReductionAcceptsRuntimeOutputScale) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor input = makeGpuTensor({1.0f, 2.0f, 3.0f}, {3}, stream);
    std::shared_ptr<StampedCubReduction> reduction =
        CubReduction(CubReductionOp::Sum, 0, DataType::FP32, 1.0f).stamp(input, stream);

    reduction->run(0.5f);
    stream.synchronize();
    expectFloatVectorNear(copyGpuTensorAsFloat(reduction->getOutputTensor(), stream), {3.0f});

    reduction->runOn(stream, 2.0f);
    stream.synchronize();
    expectFloatVectorNear(copyGpuTensorAsFloat(reduction->getOutputTensor(), stream), {12.0f});
}

TEST(CubReduction, NativePropagatingMinMaxPreservesValueNaNSemantics) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::vector<float> values = {
        3.0f,  3.0f, -4.0f,  8.0f,
        nan,   2.0f, -2.0f, -1.0f,
        2.0f,  5.0f, -8.0f,  4.0f,
        1.0f,  4.0f, -1.0f,  2.0f,
    };
    Tensor input = makeGpuTensor(values, {1, 4, 4}, stream, DataType::FP32);
    const CubReductionGeometry geometry = CubReduction::analyzeGeometry(input.getDimensions(), 1);
    ASSERT_EQ(geometry.path, CubReductionPath::TiledFixedSegment);
    const CubReductionInternal::KParallelTiledStagePlan plan{
        CubReductionInternal::TiledRKStageTopology::Complete, 16, 32, 1};

    for (const auto& [op, expected] : std::vector<std::pair<CubReductionOp, std::vector<float>>>{
             {CubReductionOp::Min, {nan, 2.0f, -8.0f, -1.0f}},
             {CubReductionOp::Max, {nan, 5.0f, -1.0f, 8.0f}},
         }) {
        Tensor output(gpuPlacement, TensorDescriptor(DataType::FP32, {1, 4}));
        const auto semantics = CubReductionInternal::makeValueReductionStageSemantics(
            op, CubReductionInternal::CubReductionStageRole::Complete, geometry.reduction_size);
        CubReductionInternal::launchKParallelTiledStage(semantics, input, output, geometry, plan, 1.0f, stream);
        const std::vector<float> actual = copyGpuTensorAsFloat(output, stream);
        ASSERT_EQ(actual.size(), expected.size());
        EXPECT_TRUE(std::isnan(actual[0]));
        for (size_t i = 1; i < expected.size(); ++i) {
            EXPECT_FLOAT_EQ(actual[i], expected[i]);
        }
    }
}

TEST(CubReduction, NarrowLowPrecisionFlatRCooperativeFirstStageReducesK17WithoutFinalization) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t outer_size = 1;
    constexpr uint64_t reduction_size = 256;
    constexpr uint64_t inner_size = 17;
    constexpr uint64_t rows_per_shard = 128;
    constexpr uint64_t shards_per_output = 2;
    const std::vector<float> values(outer_size * reduction_size * inner_size, 0.125f);

    for (DataType dtype : {DataType::FP16, DataType::BF16}) {
        for (bool explicit_outer_dimension : {true, false}) {
            SCOPED_TRACE(static_cast<int>(dtype));
            SCOPED_TRACE(explicit_outer_dimension);
            // The ordinary planner may collapse a leading singleton O axis.
            // The same contiguous physical stage must accept either rank.
            const std::vector<uint64_t> input_shape = explicit_outer_dimension
                ? std::vector<uint64_t>{outer_size, reduction_size, inner_size}
                : std::vector<uint64_t>{reduction_size, inner_size};
            const std::vector<uint64_t> partial_shape = explicit_outer_dimension
                ? std::vector<uint64_t>{outer_size, shards_per_output, inner_size}
                : std::vector<uint64_t>{shards_per_output, inner_size};
            Tensor input = makeGpuTensor(values, input_shape, stream, dtype);
            const CubReductionGeometry geometry = CubReduction::analyzeGeometry(
                input.getDimensions(), explicit_outer_dimension ? 1U : 0U);
            ASSERT_EQ(geometry.path, CubReductionPath::TiledFixedSegment);
            Tensor partials(gpuPlacement, TensorDescriptor(DataType::FP32, partial_shape));
            const auto semantics = CubReductionInternal::makeValueReductionStageSemantics(
                CubReductionOp::Sum, CubReductionInternal::CubReductionStageRole::First, reduction_size);

            CubReductionInternal::launchNarrowLowPrecisionFlatRCooperativeFirstStage(
                semantics, input, partials, geometry, rows_per_shard, shards_per_output, stream);
            const std::vector<float> actual = copyGpuTensorAsFloat(partials, stream);
            ASSERT_EQ(actual.size(), outer_size * shards_per_output * inner_size);
            for (float value : actual) {
                EXPECT_NEAR(value, 16.0f, 1.0e-5f);
            }
            // Relaxing logical-rank validation must not allow an undersized
            // physical FP32 partial buffer into a CUDA launch.
            Tensor undersized(gpuPlacement, TensorDescriptor(
                DataType::FP32, {shards_per_output * inner_size - 1}));
            EXPECT_THROW(
                CubReductionInternal::launchNarrowLowPrecisionFlatRCooperativeFirstStage(
                    semantics, input, undersized, geometry, rows_per_shard, shards_per_output, stream),
                std::logic_error);
        }
    }
}

TEST(CubReduction, OrdinaryDenseValueComposedGeometryExecutesThroughFlatDensePlan) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    const std::vector<float> values = {
        1, 2, 3, 4,   5, 6, 7, 8,   9, 10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
    };
    Tensor input = makeGpuTensor(values, {2, 3, 4}, stream, DataType::FP32);
    CubReduction reduction(CubReductionOp::Sum, std::vector<uint32_t>{0, 2}, DataType::FP32);
    std::shared_ptr<StampedCubReduction> stamped = reduction.stamp(input, stream);

    ASSERT_EQ(stamped->getPath(), CubReductionPath::ComposedDense);
    ASSERT_TRUE(stamped->usesDenseExecutablePlan());
    EXPECT_EQ(stamped->getDenseExecutablePassCount(), 2U);
    EXPECT_EQ(stamped->getComposedStageAxes().size(), 2U);
    EXPECT_EQ(stamped->getWorkspaceSizeInBytes(), reduction.queryWorkspaceSizeInBytes(input.getDescriptor(), stream));

    stamped->run();
    stream.synchronize();
    // Reduce dimensions 0 and 2, retaining the middle dimension.
    expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream),
                          {68.0f, 100.0f, 132.0f});
}

TEST(CubReduction, OrdinaryDenseValueDirectRAndKRExecuteThroughFlatDensePlan) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    {
        Tensor input = makeGpuTensor({1, 2, 3, 4}, {4}, stream, DataType::FP32);
        std::shared_ptr<StampedCubReduction> stamped =
            CubReduction(CubReductionOp::Sum, 0, DataType::FP32).stamp(input, stream);
        ASSERT_EQ(stamped->getPath(), CubReductionPath::DeviceTransformReduce);
        EXPECT_TRUE(stamped->usesDenseExecutablePlan());
        EXPECT_EQ(stamped->getDenseExecutablePassCount(), 1U);
        stamped->run();
        stream.synchronize();
        expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), {10.0f});
    }

    {
        Tensor input = makeGpuTensor({1, 2, 3, 4, 5, 6}, {2, 3}, stream, DataType::FP32);
        std::shared_ptr<StampedCubReduction> stamped =
            CubReduction(CubReductionOp::Sum, 1, DataType::FP32).stamp(input, stream);
        ASSERT_EQ(stamped->getPath(), CubReductionPath::ContiguousFixedSegment);
        EXPECT_TRUE(stamped->usesDenseExecutablePlan());
        EXPECT_EQ(stamped->getDenseExecutablePassCount(), 1U);
        stamped->run();
        stream.synchronize();
        expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), {6.0f, 15.0f});
    }
}
