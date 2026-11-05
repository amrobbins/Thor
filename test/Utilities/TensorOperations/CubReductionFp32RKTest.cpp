// 1185: retained RK VALUE kernels on FP32 input (plans/Thor_FP32_Dense_Reduction_Plan, phase 5).
// OneWindow: the FP32 unit inventory (FP8's units of buckets <= 4) and its K coverage, and every unit at K values
// across its bucket against a CPU reference, for every combine and transform, W1/W2/W4/W8, a negative output scale and
// a narrow output dtype. Inputs are quarters with small sums and products of +-1/2/0.5, so every result is exact in
// FP32 regardless of order. 1186: DeepRSmallK's FP32 partials, the same way. 1188: AlignedK, Complete and Staged.
// 1190: RuntimeK PacketComplete (Complete) and the rotated PrefixReserved first stage (Staged).
// 1198: FP32 RK ARG: the AlignedK twin, Complete and Staged with the packed continuation. 1200: the rotated twin.
// 1201: the PacketComplete twin.

#include "test/Utilities/TensorOperations/CubReductionTestSupport.h"
#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"
#include "Utilities/TensorOperations/Cub/CubReductionStageSemantics.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8DeepRSmallKSelector.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace ThorImplementation;
using namespace ThorImplementation::CubReductionTestSupport;

namespace {

// FP32 OneWindow covers K <= 125 (P16 O1 holds 128 elements, less 3 for a misaligned row), and K = 128.
bool fp32OneWindowShouldCover(uint64_t k) { return k >= 1 && (k <= 125 || k == 128); }

float fp32RkReference(CubReductionInternal::CubReductionStageCombine combine,
                      CubReductionInternal::CubReductionStageInputTransform transform, const std::vector<float>& x,
                      uint64_t outer, uint64_t rows, uint64_t k, uint64_t component) {
    using namespace CubReductionInternal;
    const bool product = combine == CubReductionStageCombine::Product;
    float acc = combine == CubReductionStageCombine::Sum ? 0.0f
              : product                                 ? 1.0f
              : combine == CubReductionStageCombine::Minimum ? std::numeric_limits<float>::infinity()
                                                             : -std::numeric_limits<float>::infinity();
    for (uint64_t r = 0; r < rows; ++r) {
        float v = x[(outer * rows + r) * k + component];
        if (transform == CubReductionStageInputTransform::AbsoluteValue) {
            v = std::fabs(v);
        } else if (transform == CubReductionStageInputTransform::Square) {
            v = v * v;
        }
        acc = combine == CubReductionStageCombine::Sum ? acc + v
            : product                                 ? acc * v
            : combine == CubReductionStageCombine::Minimum ? std::min(acc, v)
                                                            : std::max(acc, v);
    }
    return acc;
}

}  // namespace

TEST(CubReductionFp32RK, OneWindowInventoryAndCoverage) {
    using namespace CubReductionInternal;
    for (uint64_t k = 1; k <= 300; ++k) {
        const auto units = fp32OneWindowRuntimeKUnits(k);
        EXPECT_EQ(fp32OneWindowShouldCover(k), !units.empty()) << k;
        for (const auto& unit : units) {
            EXPECT_LE(unit.max_components_per_lane, 4u) << k;
            EXPECT_EQ(16u, unit.packet_bytes);
        }
    }
    // Per ownership: O2 holds 61 (64 aligned), O4 29 (32), O8 13 (16).
    const auto owns = [](uint64_t k, uint32_t outputs_per_warp) {
        const auto units = fp32OneWindowRuntimeKUnits(k);
        return std::any_of(units.begin(), units.end(),
                           [&](const auto& unit) { return unit.outputs_per_warp == outputs_per_warp; });
    };
    EXPECT_TRUE(owns(61, 2));
    EXPECT_FALSE(owns(62, 2));
    EXPECT_TRUE(owns(64, 2));
    EXPECT_TRUE(owns(29, 4));
    EXPECT_FALSE(owns(30, 4));
    EXPECT_TRUE(owns(13, 8));
    EXPECT_FALSE(owns(14, 8));
    EXPECT_TRUE(owns(16, 8));
}

TEST(CubReductionFp32RK, OneWindowUnitsMatchReference) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    const std::array<float, 8> pattern = {-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f, 2.0f};
    // (outer, R): a few outputs over a moderate R, one output over a long R (multi-warp O1 rounds), many outputs.
    const std::vector<std::pair<uint64_t, uint64_t>> shapes{{5, 37}, {1, 301}, {700, 3}};
    // K >= 2: [outer, R, 1] is a KR site (not the tiled RK path), as in the FP8 OneWindow test.
    const std::vector<uint64_t> ks{2, 3, 4, 5, 7, 8, 9, 13, 16, 17, 29, 31, 32, 33, 61, 64, 65, 100, 124, 125, 128};
    for (const auto& [outer_size, rows] : shapes) {
        for (uint64_t k : ks) {
            SCOPED_TRACE(::testing::Message() << "outer=" << outer_size << " R=" << rows << " K=" << k);
            std::vector<float> additive(outer_size * rows * k);
            std::vector<float> multiplicative(additive.size(), 1.0f);
            for (uint64_t i = 0; i < additive.size(); ++i) {
                additive[i] = pattern[(i * 5 + i / k) % pattern.size()];
                if (i % 7 == 3) multiplicative[i] = (i % 3 == 0) ? -1.0f : ((i % 3 == 1) ? 2.0f : 0.5f);
            }
            const auto geometry = CubReduction::analyzeValueGeometry(CubReductionOp::Sum, {outer_size, rows, k}, {1});
            ASSERT_EQ(geometry.path, CubReductionPath::TiledFixedSegment);
            const Tensor additive_input = makeGpuTensor(additive, {outer_size, rows, k}, stream, DataType::FP32);
            const Tensor product_input = makeGpuTensor(multiplicative, {outer_size, rows, k}, stream, DataType::FP32);
            const auto units = fp32OneWindowRuntimeKUnits(k);
            ASSERT_FALSE(units.empty());
            for (const Fp8OneWindowRuntimeKUnit& unit : units) {
                const uint32_t o = unit.outputs_per_warp;
                for (uint32_t w : {1u, 2u, 4u, 8u}) {
                    if (o > 1 && w != 1) continue;
                    for (CubReductionStageCombine combine :
                         {CubReductionStageCombine::Sum, CubReductionStageCombine::Product,
                          CubReductionStageCombine::Minimum, CubReductionStageCombine::Maximum}) {
                        for (CubReductionStageInputTransform transform :
                             {CubReductionStageInputTransform::Identity,
                              CubReductionStageInputTransform::AbsoluteValue,
                              CubReductionStageInputTransform::Square}) {
                            SCOPED_TRACE(::testing::Message()
                                         << "O" << o << " W" << w << " bucket " << unit.max_components_per_lane
                                         << " load_ahead " << unit.load_ahead << " combine "
                                         << static_cast<int>(combine) << " transform "
                                         << static_cast<int>(transform));
                            const bool product = combine == CubReductionStageCombine::Product;
                            const Tensor& input = product ? product_input : additive_input;
                            const std::vector<float>& values = product ? multiplicative : additive;
                            CubReductionStageSemantics semantics;
                            semantics.combine = combine;
                            semantics.input_transform = transform;
                            constexpr float output_scale = -0.5f;
                            const auto occupancy = queryFp32OneWindowRuntimeKOccupancy(k, unit, w, combine);
                            ASSERT_GT(occupancy.max_active_blocks_per_sm, 0);
                            Tensor output(gpuPlacement, TensorDescriptor(DataType::FP32, geometry.output_dimensions));
                            launchFp32OneWindowRuntimeKComplete(input, output, geometry, unit, w, stream, semantics,
                                                                output_scale);
                            stream.synchronize();
                            const auto actual = copyGpuTensorAsFloat(output, stream);
                            ASSERT_EQ(actual.size(), outer_size * k);
                            for (uint64_t outer = 0; outer < outer_size; ++outer) {
                                for (uint64_t c = 0; c < k; ++c) {
                                    const float expected =
                                        fp32RkReference(combine, transform, values, outer, rows, k, c) * output_scale;
                                    ASSERT_EQ(expected, actual[outer * k + c]) << "outer=" << outer << " k=" << c;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// MEAN finalizes once (divide by R), and a BF16 output converts the finalized, scaled value.
TEST(CubReductionFp32RK, OneWindowFinalizeAndNarrowOutput) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    const uint64_t outer_size = 3, rows = 50, k = 20;
    std::vector<float> values(outer_size * rows * k);
    for (uint64_t i = 0; i < values.size(); ++i) values[i] = static_cast<float>(static_cast<int>(i % 13) - 6) * 0.25f;
    const auto geometry = CubReduction::analyzeValueGeometry(CubReductionOp::Mean, {outer_size, rows, k}, {1});
    const Tensor input = makeGpuTensor(values, {outer_size, rows, k}, stream, DataType::FP32);
    const auto semantics = makeValueReductionStageSemantics(CubReductionOp::Mean, DensePassRole::Complete, rows);
    for (const auto& unit : fp32OneWindowRuntimeKUnits(k)) {
        SCOPED_TRACE(::testing::Message() << "O" << unit.outputs_per_warp << " bucket "
                                          << unit.max_components_per_lane);
        Tensor output(gpuPlacement, TensorDescriptor(DataType::BF16, geometry.output_dimensions));
        launchFp32OneWindowRuntimeKComplete(input, output, geometry, unit, 1, stream, semantics, 2.0f);
        stream.synchronize();
        const auto actual = copyGpuTensorAsFloat(output, stream);
        for (uint64_t outer = 0; outer < outer_size; ++outer) {
            for (uint64_t c = 0; c < k; ++c) {
                double sum = 0;
                for (uint64_t r = 0; r < rows; ++r) sum += values[(outer * rows + r) * k + c];
                const double expected = 2.0 * sum / static_cast<double>(rows);
                ASSERT_NEAR(expected, actual[outer * k + c], 1e-2 * std::fabs(expected) + 1e-3);
            }
        }
    }
}

// The launcher rejects FP8 input, K outside the unit's bucket, a misaligned base, and the wrong output size.
TEST(CubReductionFp32RK, OneWindowRejectsUnsupportedRequests) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    const uint64_t outer_size = 4, rows = 8, k = 16;
    const auto geometry = CubReduction::analyzeValueGeometry(CubReductionOp::Sum, {outer_size, rows, k}, {1});
    const Tensor input = makeGpuTensor(std::vector<float>(outer_size * rows * k, 1.0f), {outer_size, rows, k}, stream,
                                       DataType::FP32);
    // The same FP32 elements 4 bytes past a 16-byte boundary.
    const Tensor bytes = makeGpuUnsignedTensor(std::vector<uint64_t>(4 * outer_size * rows * k + 16, 0),
                                               {4 * outer_size * rows * k + 16}, stream, DataType::UINT8);
    const Tensor misaligned = bytes.aliasByteStorageAs(DataType::FP32, {outer_size, rows, k}, 4);
    const auto units = fp32OneWindowRuntimeKUnits(k);
    ASSERT_FALSE(units.empty());
    const auto unit = units.front();
    Tensor output(gpuPlacement, TensorDescriptor(DataType::FP32, geometry.output_dimensions));
    const CubReductionStageSemantics semantics;
    // The wrong input dtype (FP8 view of the same bytes).
    const Tensor fp8 = makeGpuTensor(std::vector<float>(outer_size * rows * k, 1.0f), {outer_size, rows, k}, stream,
                                     DataType::FP8_E4M3);
    EXPECT_THROW(launchFp32OneWindowRuntimeKComplete(fp8, output, geometry, unit, 1, stream, semantics),
                 std::logic_error);
    // A bucket-8 unit (FP8 only) and an FP32 unit at a K outside its bucket.
    EXPECT_THROW(launchFp32OneWindowRuntimeKComplete(input, output, geometry, {1, 16, 8, true}, 1, stream, semantics),
                 std::logic_error);
    EXPECT_THROW(static_cast<void>(queryFp32OneWindowRuntimeKOccupancy(126, {1, 16, 4, false}, 1)), std::logic_error);
    // A base off 16-byte alignment.
    EXPECT_THROW(launchFp32OneWindowRuntimeKComplete(misaligned, output, geometry, unit, 1, stream, semantics),
                 std::logic_error);
    // O2 with W2 (O2/O4/O8 are warp-local).
    const Fp8OneWindowRuntimeKUnit o2{2, 16, 1, false};
    if (std::find(units.begin(), units.end(), o2) != units.end()) {
        EXPECT_THROW(launchFp32OneWindowRuntimeKComplete(input, output, geometry, o2, 2, stream, semantics),
                     std::logic_error);
    }
    // The wrong output size.
    Tensor short_output(gpuPlacement, TensorDescriptor(DataType::FP32, {outer_size * k - 1}));
    EXPECT_THROW(launchFp32OneWindowRuntimeKComplete(input, short_output, geometry, unit, 1, stream, semantics),
                 std::logic_error);
    EXPECT_NO_THROW(launchFp32OneWindowRuntimeKComplete(input, output, geometry, unit, 1, stream, semantics));
    stream.synchronize();
}

// 1186: DeepRSmallK on FP32 input. Every ownership (TinyK at K <= 4 and <= 8, one / two / four components per lane)
// at K values across it, the five depths for the one-component kernels (compiled depths) and two depths elsewhere,
// R past several shards with a partial last shard, every combine and transform: each FP32 partial [outer, shard, K]
// against a CPU reduction of its rows, exactly (quarters with small sums, products of +-1/2/0.5).
TEST(CubReductionFp32RK, DeepRSmallKPartialsMatchReference) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    const std::array<float, 8> pattern = {-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f, 2.0f};
    const std::vector<uint64_t> ks{1, 2, 3, 4, 5, 7, 8, 9, 16, 31, 32, 33, 63, 64, 65, 100, 124, 125};
    const auto& depths = Fp8RK::Fp8DeepRSmallKPolicy::compiled_rows;
    EXPECT_FALSE(fp32DeepRSmallKSupports(3, 1000, 126, 64));
    EXPECT_FALSE(fp32DeepRSmallKSupports(3, 64, 16, 64));   // R must exceed the depth
    EXPECT_FALSE(fp32DeepRSmallKSupports(3, 1000, 16, 100));  // not a retained depth
    EXPECT_TRUE(fp32DeepRSmallKSupports(3, 1000, 125, 64));
    const uint64_t outer_size = 3;
    for (uint64_t k : ks) {
        std::vector<uint64_t> k_depths{depths.front(), 256};
        if (k >= 9 && k <= 32) k_depths.assign(depths.begin(), depths.end());
        for (uint64_t depth : k_depths) {
            const uint64_t rows = 2 * depth + 37;
            const uint64_t shards = (rows + depth - 1) / depth;
            SCOPED_TRACE(::testing::Message() << "K=" << k << " depth=" << depth << " R=" << rows);
            ASSERT_TRUE(fp32DeepRSmallKSupports(outer_size, rows, k, depth));
            std::vector<float> additive(outer_size * rows * k);
            std::vector<float> multiplicative(additive.size(), 1.0f);
            for (uint64_t i = 0; i < additive.size(); ++i) {
                additive[i] = pattern[(i * 5 + i / k) % pattern.size()];
                if (i % 97 == 3) multiplicative[i] = (i % 3 == 0) ? -1.0f : ((i % 3 == 1) ? 2.0f : 0.5f);
            }
            const auto geometry = CubReduction::analyzeValueGeometry(CubReductionOp::Sum, {outer_size, rows, k}, {1});
            const Tensor additive_input = makeGpuTensor(additive, {outer_size, rows, k}, stream, DataType::FP32);
            const Tensor product_input = makeGpuTensor(multiplicative, {outer_size, rows, k}, stream, DataType::FP32);
            for (CubReductionStageCombine combine :
                 {CubReductionStageCombine::Sum, CubReductionStageCombine::Product, CubReductionStageCombine::Minimum,
                  CubReductionStageCombine::Maximum}) {
                for (CubReductionStageInputTransform transform :
                     {CubReductionStageInputTransform::Identity, CubReductionStageInputTransform::AbsoluteValue,
                      CubReductionStageInputTransform::Square}) {
                    SCOPED_TRACE(::testing::Message() << "combine " << static_cast<int>(combine) << " transform "
                                                      << static_cast<int>(transform));
                    const bool product = combine == CubReductionStageCombine::Product;
                    const Tensor& input = product ? product_input : additive_input;
                    const std::vector<float>& values = product ? multiplicative : additive;
                    CubReductionStageSemantics semantics;
                    semantics.combine = combine;
                    semantics.input_transform = transform;
                    semantics.finalize = CubReductionStageFinalize::Identity;
                    ASSERT_GT(queryFp32DeepRSmallKRetainedOccupancy(depth, k, semantics).max_active_blocks_per_sm, 0);
                    Tensor partials(gpuPlacement, TensorDescriptor(DataType::FP32, {outer_size, shards, k}));
                    launchFp32DeepRSmallKRetainedFirstStage(semantics, input, partials, geometry, depth, shards,
                                                            stream);
                    stream.synchronize();
                    const auto actual = copyGpuTensorAsFloat(partials, stream);
                    for (uint64_t outer = 0; outer < outer_size; ++outer) {
                        for (uint64_t shard = 0; shard < shards; ++shard) {
                            const uint64_t first = shard * depth;
                            const uint64_t count = std::min(depth, rows - first);
                            // The shard's rows as their own [1, count, K] block.
                            std::vector<float> block(count * k);
                            for (uint64_t r = 0; r < count; ++r)
                                for (uint64_t c = 0; c < k; ++c)
                                    block[r * k + c] = values[(outer * rows + first + r) * k + c];
                            for (uint64_t c = 0; c < k; ++c) {
                                const float expected = fp32RkReference(combine, transform, block, 0, count, k, c);
                                ASSERT_EQ(expected, actual[(outer * shards + shard) * k + c])
                                    << "outer=" << outer << " shard=" << shard << " k=" << c;
                            }
                        }
                    }
                }
            }
        }
    }
    // Rejections: FP8 input, K = 126, a wrong partials shape.
    const auto geometry = CubReduction::analyzeValueGeometry(CubReductionOp::Sum, {2, 200, 16}, {1});
    const Tensor input = makeGpuTensor(std::vector<float>(2 * 200 * 16, 1.0f), {2, 200, 16}, stream, DataType::FP32);
    const Tensor fp8 = makeGpuTensor(std::vector<float>(2 * 200 * 16, 1.0f), {2, 200, 16}, stream, DataType::FP8_E4M3);
    Tensor partials(gpuPlacement, TensorDescriptor(DataType::FP32, {2, 4, 16}));
    Tensor wrong(gpuPlacement, TensorDescriptor(DataType::FP32, {2, 3, 16}));
    const CubReductionStageSemantics semantics;
    EXPECT_THROW(launchFp32DeepRSmallKRetainedFirstStage(semantics, fp8, partials, geometry, 64, 4, stream),
                 std::logic_error);
    EXPECT_THROW(launchFp32DeepRSmallKRetainedFirstStage(semantics, input, wrong, geometry, 64, 4, stream),
                 std::logic_error);
    EXPECT_THROW(static_cast<void>(queryFp32DeepRSmallKRetainedOccupancy(64, 126, semantics)), std::logic_error);
    EXPECT_NO_THROW(launchFp32DeepRSmallKRetainedFirstStage(semantics, input, partials, geometry, 64, 4, stream));
    stream.synchronize();
}

// 1188: AlignedK on FP32 input. Both packets (2 and 4 components per lane), CTA 32 / 64 / 128, Complete and Staged
// (1228: and the 4-byte packet, one component per lane, at every K including odd ones)
// (FP8's shard targets of 32 and 128 rows), K values across one and several component tiles, every combine and
// transform: Complete outputs against a CPU reduction (with a negative scale), Staged partials against a CPU reduction
// of each shard's rows at the balanced split (the first R % shards shards own one extra row), exactly.
TEST(CubReductionFp32RK, AlignedKMatchesReference) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    const std::array<float, 8> pattern = {-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f, 2.0f};
    const std::vector<uint64_t> ks{2, 3, 4, 6, 8, 62, 64, 115, 130, 256, 1000, 1001, 1028};
    const uint64_t outer_size = 3, rows = 300;
    EXPECT_FALSE(fp32AlignedKSupports(3, 300, 6, 16, 32, 1));   // K % 4
    EXPECT_FALSE(fp32AlignedKSupports(3, 300, 7, 8, 32, 1));    // K % 2
    EXPECT_TRUE(fp32AlignedKSupports(3, 300, 7, 4, 32, 1));     // 1228: the 4-byte packet takes any K
    EXPECT_FALSE(fp32AlignedKSupports(3, 300, 8, 12, 32, 1));   // packet
    EXPECT_FALSE(fp32AlignedKSupports(3, 300, 8, 16, 256, 1));  // CTA
    EXPECT_FALSE(fp32AlignedKSupports(3, 300, 8, 16, 32, 301)); // more shards than rows
    EXPECT_TRUE(fp32AlignedKSupports(3, 300, 8, 16, 32, 10));
    for (uint64_t k : ks) {
        std::vector<float> additive(outer_size * rows * k);
        std::vector<float> multiplicative(additive.size(), 1.0f);
        for (uint64_t i = 0; i < additive.size(); ++i) {
            additive[i] = pattern[(i * 5 + i / k) % pattern.size()];
            if (i % 97 == 3) multiplicative[i] = (i % 3 == 0) ? -1.0f : ((i % 3 == 1) ? 2.0f : 0.5f);
        }
        const auto geometry = CubReduction::analyzeValueGeometry(CubReductionOp::Sum, {outer_size, rows, k}, {1});
        ASSERT_EQ(geometry.path, CubReductionPath::TiledFixedSegment);
        const Tensor additive_input = makeGpuTensor(additive, {outer_size, rows, k}, stream, DataType::FP32);
        const Tensor product_input = makeGpuTensor(multiplicative, {outer_size, rows, k}, stream, DataType::FP32);
        for (uint64_t shards : {uint64_t{1}, (rows + 31) / 32, (rows + 127) / 128}) {
            const bool staged = shards > 1;
            const uint64_t base = rows / shards, extra = rows % shards;
            for (CubReductionStageCombine combine :
                 {CubReductionStageCombine::Sum, CubReductionStageCombine::Product, CubReductionStageCombine::Minimum,
                  CubReductionStageCombine::Maximum}) {
                for (CubReductionStageInputTransform transform :
                     {CubReductionStageInputTransform::Identity, CubReductionStageInputTransform::AbsoluteValue,
                      CubReductionStageInputTransform::Square}) {
                    const bool product = combine == CubReductionStageCombine::Product;
                    if (product && transform != CubReductionStageInputTransform::Identity) continue;
                    const Tensor& input = product ? product_input : additive_input;
                    const std::vector<float>& values = product ? multiplicative : additive;
                    CubReductionStageSemantics semantics;
                    semantics.combine = combine;
                    semantics.input_transform = transform;
                    constexpr float output_scale = -0.5f;
                    // The expected [outer, shards, K] (Complete: shards = 1, scaled), once for every configuration.
                    std::vector<float> expected(outer_size * shards * k);
                    for (uint64_t outer = 0; outer < outer_size; ++outer) {
                        for (uint64_t shard = 0; shard < shards; ++shard) {
                            const uint64_t first = shard * base + std::min(shard, extra);
                            const uint64_t count = base + (shard < extra ? 1 : 0);
                            const auto row = [&](uint64_t r) {
                                return values.begin() + static_cast<std::ptrdiff_t>((outer * rows + r) * k);
                            };
                            const std::vector<float> block(row(first), row(first + count));
                            for (uint64_t c = 0; c < k; ++c) {
                                const float v = fp32RkReference(combine, transform, block, 0, count, k, c);
                                expected[(outer * shards + shard) * k + c] = staged ? v : v * output_scale;
                            }
                        }
                    }
                    for (uint32_t packet_bytes : {4u, 8u, 16u}) {
                        if (k % (packet_bytes / 4) != 0) continue;
                        for (uint32_t threads : {32u, 64u, 128u}) {
                            SCOPED_TRACE(::testing::Message()
                                         << "K=" << k << " P" << packet_bytes << " CTA" << threads << " shards "
                                         << shards << " combine " << static_cast<int>(combine) << " transform "
                                         << static_cast<int>(transform));
                            ASSERT_GT(
                                queryFp32AlignedKOccupancy(packet_bytes, threads, staged).max_active_blocks_per_sm,
                                0);
                            Tensor output(gpuPlacement,
                                          staged ? TensorDescriptor(DataType::FP32, {outer_size, shards, k})
                                                 : TensorDescriptor(DataType::FP32, geometry.output_dimensions));
                            launchFp32AlignedKStage(semantics, input, output, geometry, packet_bytes, threads, shards,
                                                    output_scale, stream);
                            stream.synchronize();
                            const auto actual = copyGpuTensorAsFloat(output, stream);
                            ASSERT_EQ(expected.size(), actual.size());
                            for (size_t i = 0; i < expected.size(); ++i)
                                ASSERT_EQ(expected[i], actual[i]) << "index " << i;
                        }
                    }
                }
            }
        }
    }
    // MEAN Complete finalizes (divides by R) once, before the scale; a BF16 output converts the result.
    {
        const uint64_t k = 64;
        std::vector<float> values(outer_size * rows * k);
        for (uint64_t i = 0; i < values.size(); ++i) values[i] = pattern[(i * 3 + i / k) % pattern.size()];
        const auto geometry = CubReduction::analyzeValueGeometry(CubReductionOp::Mean, {outer_size, rows, k}, {1});
        const Tensor input = makeGpuTensor(values, {outer_size, rows, k}, stream, DataType::FP32);
        const auto semantics = makeValueReductionStageSemantics(CubReductionOp::Mean, DensePassRole::Complete, rows);
        Tensor output(gpuPlacement, TensorDescriptor(DataType::BF16, geometry.output_dimensions));
        launchFp32AlignedKStage(semantics, input, output, geometry, 16, 64, 1, 2.0f, stream);
        stream.synchronize();
        const auto actual = copyGpuTensorAsFloat(output, stream);
        for (uint64_t outer = 0; outer < outer_size; ++outer) {
            for (uint64_t c = 0; c < k; ++c) {
                double sum = 0;
                for (uint64_t r = 0; r < rows; ++r) sum += values[(outer * rows + r) * k + c];
                const double expected = 2.0 * sum / static_cast<double>(rows);
                ASSERT_NEAR(expected, actual[outer * k + c], 1e-2 * std::fabs(expected) + 1e-3);
            }
        }
    }
    // Rejections: FP8 input, PRODUCT with a transform, a misaligned base, a wrong partials shape.
    {
        const uint64_t k = 8;
        const auto geometry = CubReduction::analyzeValueGeometry(CubReductionOp::Sum, {outer_size, rows, k}, {1});
        const Tensor input = makeGpuTensor(std::vector<float>(outer_size * rows * k, 1.0f), {outer_size, rows, k},
                                           stream, DataType::FP32);
        const Tensor fp8 = makeGpuTensor(std::vector<float>(outer_size * rows * k, 1.0f), {outer_size, rows, k},
                                         stream, DataType::FP8_E4M3);
        const Tensor bytes = makeGpuUnsignedTensor(std::vector<uint64_t>(4 * outer_size * rows * k + 16, 0),
                                                   {4 * outer_size * rows * k + 16}, stream, DataType::UINT8);
        const Tensor misaligned = bytes.aliasByteStorageAs(DataType::FP32, {outer_size, rows, k}, 8);
        const Tensor misaligned4 = bytes.aliasByteStorageAs(DataType::FP32, {outer_size, rows, k}, 4);
        Tensor complete(gpuPlacement, TensorDescriptor(DataType::FP32, geometry.output_dimensions));
        Tensor wrong(gpuPlacement, TensorDescriptor(DataType::FP32, {outer_size, 9, k}));
        CubReductionStageSemantics semantics;
        EXPECT_THROW(launchFp32AlignedKStage(semantics, fp8, complete, geometry, 16, 32, 1, 1.0f, stream),
                     std::logic_error);
        EXPECT_THROW(launchFp32AlignedKStage(semantics, misaligned, complete, geometry, 16, 32, 1, 1.0f, stream),
                     std::logic_error);
        EXPECT_THROW(launchFp32AlignedKStage(semantics, input, wrong, geometry, 16, 32, 10, 1.0f, stream),
                     std::logic_error);
        CubReductionStageSemantics product;
        product.combine = CubReductionStageCombine::Product;
        product.input_transform = CubReductionStageInputTransform::Square;
        EXPECT_THROW(launchFp32AlignedKStage(product, input, complete, geometry, 16, 32, 1, 1.0f, stream),
                     std::logic_error);
        // An 8-byte packet runs on an 8-byte aligned base, the 4-byte packet (1228) on a 4-byte aligned one.
        EXPECT_NO_THROW(launchFp32AlignedKStage(semantics, misaligned, complete, geometry, 8, 32, 1, 1.0f, stream));
        EXPECT_THROW(launchFp32AlignedKStage(semantics, misaligned4, complete, geometry, 8, 32, 1, 1.0f, stream),
                     std::logic_error);
        EXPECT_NO_THROW(launchFp32AlignedKStage(semantics, misaligned4, complete, geometry, 4, 32, 1, 1.0f, stream));
        stream.synchronize();
    }
}

// 1228: the 4-byte AlignedK packet at the small R it was added for: every R below its 16-row load-ahead batch (the
// all-rows-then-reduce tail), one batch, a batch plus a tail; odd K over one and several component tiles; Complete
// and two shards; SUM / PRODUCT / MIN / MAX with their transforms, exactly against a CPU reduction.
TEST(CubReductionFp32RK, AlignedKScalarPacketShortRowsMatchReference) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    const std::array<float, 8> pattern = {-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f, 2.0f};
    const uint64_t outer_size = 3;
    for (uint64_t rows : {uint64_t{2}, uint64_t{3}, uint64_t{5}, uint64_t{8}, uint64_t{15}, uint64_t{16},
                          uint64_t{17}, uint64_t{31}, uint64_t{33}}) {
        for (uint64_t k : {uint64_t{3}, uint64_t{115}, uint64_t{257}}) {
            std::vector<float> additive(outer_size * rows * k);
            std::vector<float> multiplicative(additive.size(), 1.0f);
            for (uint64_t i = 0; i < additive.size(); ++i) {
                additive[i] = pattern[(i * 5 + i / k) % pattern.size()];
                if (i % 7 == 3) multiplicative[i] = (i % 3 == 0) ? -1.0f : ((i % 3 == 1) ? 2.0f : 0.5f);
            }
            const auto geometry = CubReduction::analyzeValueGeometry(CubReductionOp::Sum, {outer_size, rows, k}, {1});
            ASSERT_EQ(geometry.path, CubReductionPath::TiledFixedSegment);
            const Tensor additive_input = makeGpuTensor(additive, {outer_size, rows, k}, stream, DataType::FP32);
            const Tensor product_input = makeGpuTensor(multiplicative, {outer_size, rows, k}, stream, DataType::FP32);
            for (uint64_t shards : {uint64_t{1}, uint64_t{2}}) {
                const bool staged = shards > 1;
                const uint64_t base = rows / shards, extra = rows % shards;
                for (const auto& [combine, transform] :
                     {std::pair{CubReductionStageCombine::Sum, CubReductionStageInputTransform::Identity},
                      std::pair{CubReductionStageCombine::Sum, CubReductionStageInputTransform::Square},
                      std::pair{CubReductionStageCombine::Product, CubReductionStageInputTransform::Identity},
                      std::pair{CubReductionStageCombine::Minimum, CubReductionStageInputTransform::AbsoluteValue},
                      std::pair{CubReductionStageCombine::Maximum, CubReductionStageInputTransform::Identity}}) {
                    const bool product = combine == CubReductionStageCombine::Product;
                    const Tensor& input = product ? product_input : additive_input;
                    const std::vector<float>& values = product ? multiplicative : additive;
                    CubReductionStageSemantics semantics;
                    semantics.combine = combine;
                    semantics.input_transform = transform;
                    constexpr float output_scale = -0.5f;
                    std::vector<float> expected(outer_size * shards * k);
                    for (uint64_t outer = 0; outer < outer_size; ++outer) {
                        for (uint64_t shard = 0; shard < shards; ++shard) {
                            const uint64_t first = shard * base + std::min(shard, extra);
                            const uint64_t count = base + (shard < extra ? 1 : 0);
                            const auto row = [&](uint64_t r) {
                                return values.begin() + static_cast<std::ptrdiff_t>((outer * rows + r) * k);
                            };
                            const std::vector<float> block(row(first), row(first + count));
                            for (uint64_t c = 0; c < k; ++c) {
                                const float v = fp32RkReference(combine, transform, block, 0, count, k, c);
                                expected[(outer * shards + shard) * k + c] = staged ? v : v * output_scale;
                            }
                        }
                    }
                    for (uint32_t threads : {32u, 64u, 128u}) {
                        SCOPED_TRACE(::testing::Message() << "R=" << rows << " K=" << k << " CTA" << threads
                                                          << " shards " << shards << " combine "
                                                          << static_cast<int>(combine) << " transform "
                                                          << static_cast<int>(transform));
                        ASSERT_TRUE(fp32AlignedKSupports(outer_size, rows, k, 4, threads, shards));
                        Tensor output(gpuPlacement, staged ? TensorDescriptor(DataType::FP32, {outer_size, shards, k})
                                                           : TensorDescriptor(DataType::FP32,
                                                                              geometry.output_dimensions));
                        launchFp32AlignedKStage(semantics, input, output, geometry, 4, threads, shards, output_scale,
                                                stream);
                        stream.synchronize();
                        const auto actual = copyGpuTensorAsFloat(output, stream);
                        ASSERT_EQ(expected.size(), actual.size());
                        for (size_t i = 0; i < expected.size(); ++i) ASSERT_EQ(expected[i], actual[i]) << "index " << i;
                    }
                }
            }
        }
    }
}

// 1190: RuntimeK PacketComplete on FP32 input. W1/W2/W4/W8, a full tile (124) and a narrow one (37) at K values across
// one and several tiles (odd K gives every row prefix 0..3), every combine and transform, against a CPU reduction with
// a negative scale, exactly. Then MEAN with a BF16 output, and the rejections.
TEST(CubReductionFp32RK, PacketCompleteMatchesReference) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    const std::array<float, 8> pattern = {-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f, 2.0f};
    const std::vector<std::pair<uint64_t, uint64_t>> shapes{{5, 37}, {1, 301}};
    const std::vector<uint64_t> ks{2, 3, 7, 31, 33, 100, 124, 125, 130, 333, 1000};
    EXPECT_FALSE(fp32PacketCompleteSupports(3, 10, 64, 1, 125));  // tile cap
    EXPECT_FALSE(fp32PacketCompleteSupports(3, 10, 64, 3, 64));   // W
    EXPECT_FALSE(fp32PacketCompleteSupports(3, 10, 64, 1, 0));
    EXPECT_TRUE(fp32PacketCompleteSupports(3, 10, 64, 8, 124));
    for (const auto& [outer_size, rows] : shapes) {
        for (uint64_t k : ks) {
            std::vector<float> additive(outer_size * rows * k);
            std::vector<float> multiplicative(additive.size(), 1.0f);
            for (uint64_t i = 0; i < additive.size(); ++i) {
                additive[i] = pattern[(i * 5 + i / k) % pattern.size()];
                if (i % 7 == 3) multiplicative[i] = (i % 3 == 0) ? -1.0f : ((i % 3 == 1) ? 2.0f : 0.5f);
            }
            const auto geometry = CubReduction::analyzeValueGeometry(CubReductionOp::Sum, {outer_size, rows, k}, {1});
            const Tensor additive_input = makeGpuTensor(additive, {outer_size, rows, k}, stream, DataType::FP32);
            const Tensor product_input = makeGpuTensor(multiplicative, {outer_size, rows, k}, stream, DataType::FP32);
            for (CubReductionStageCombine combine :
                 {CubReductionStageCombine::Sum, CubReductionStageCombine::Product, CubReductionStageCombine::Minimum,
                  CubReductionStageCombine::Maximum}) {
                for (CubReductionStageInputTransform transform :
                     {CubReductionStageInputTransform::Identity, CubReductionStageInputTransform::AbsoluteValue,
                      CubReductionStageInputTransform::Square}) {
                    const bool product = combine == CubReductionStageCombine::Product;
                    const Tensor& input = product ? product_input : additive_input;
                    const std::vector<float>& values = product ? multiplicative : additive;
                    CubReductionStageSemantics semantics;
                    semantics.combine = combine;
                    semantics.input_transform = transform;
                    constexpr float output_scale = -0.5f;
                    std::vector<float> expected(outer_size * k);
                    for (uint64_t outer = 0; outer < outer_size; ++outer)
                        for (uint64_t c = 0; c < k; ++c)
                            expected[outer * k + c] =
                                fp32RkReference(combine, transform, values, outer, rows, k, c) * output_scale;
                    ASSERT_GT(queryFp32PacketCompleteOccupancy(transform, combine).max_active_blocks_per_sm, 0);
                    for (uint32_t tile : {124u, 37u}) {
                        for (uint32_t w : {1u, 2u, 4u, 8u}) {
                            SCOPED_TRACE(::testing::Message()
                                         << "outer=" << outer_size << " R=" << rows << " K=" << k << " tile " << tile
                                         << " W" << w << " combine " << static_cast<int>(combine) << " transform "
                                         << static_cast<int>(transform));
                            ASSERT_TRUE(fp32PacketCompleteSupports(outer_size, rows, k, w, tile));
                            Tensor output(gpuPlacement, TensorDescriptor(DataType::FP32, geometry.output_dimensions));
                            launchFp32PacketComplete(input, output, geometry, w, tile, stream, semantics, output_scale);
                            stream.synchronize();
                            const auto actual = copyGpuTensorAsFloat(output, stream);
                            ASSERT_EQ(expected.size(), actual.size());
                            for (size_t i = 0; i < expected.size(); ++i)
                                ASSERT_EQ(expected[i], actual[i]) << "index " << i;
                        }
                    }
                }
            }
        }
    }
    // MEAN finalizes (divides by R) once, before the scale; a BF16 output converts the result.
    {
        const uint64_t outer_size = 3, rows = 50, k = 77;
        std::vector<float> values(outer_size * rows * k);
        for (uint64_t i = 0; i < values.size(); ++i) values[i] = static_cast<float>(static_cast<int>(i % 13) - 6) * 0.25f;
        const auto geometry = CubReduction::analyzeValueGeometry(CubReductionOp::Mean, {outer_size, rows, k}, {1});
        const Tensor input = makeGpuTensor(values, {outer_size, rows, k}, stream, DataType::FP32);
        const auto semantics = makeValueReductionStageSemantics(CubReductionOp::Mean, DensePassRole::Complete, rows);
        Tensor output(gpuPlacement, TensorDescriptor(DataType::BF16, geometry.output_dimensions));
        launchFp32PacketComplete(input, output, geometry, 2, 64, stream, semantics, 2.0f);
        stream.synchronize();
        const auto actual = copyGpuTensorAsFloat(output, stream);
        for (uint64_t outer = 0; outer < outer_size; ++outer) {
            for (uint64_t c = 0; c < k; ++c) {
                double sum = 0;
                for (uint64_t r = 0; r < rows; ++r) sum += values[(outer * rows + r) * k + c];
                const double expected = 2.0 * sum / static_cast<double>(rows);
                ASSERT_NEAR(expected, actual[outer * k + c], 1e-2 * std::fabs(expected) + 1e-3);
            }
        }
    }
    // Rejections: FP8 input, a tile past 124, W3, a misaligned base, the wrong output size.
    {
        const uint64_t outer_size = 4, rows = 8, k = 16;
        const auto geometry = CubReduction::analyzeValueGeometry(CubReductionOp::Sum, {outer_size, rows, k}, {1});
        const Tensor input = makeGpuTensor(std::vector<float>(outer_size * rows * k, 1.0f), {outer_size, rows, k},
                                           stream, DataType::FP32);
        const Tensor fp8 = makeGpuTensor(std::vector<float>(outer_size * rows * k, 1.0f), {outer_size, rows, k},
                                         stream, DataType::FP8_E4M3);
        const Tensor bytes = makeGpuUnsignedTensor(std::vector<uint64_t>(4 * outer_size * rows * k + 16, 0),
                                                   {4 * outer_size * rows * k + 16}, stream, DataType::UINT8);
        const Tensor misaligned = bytes.aliasByteStorageAs(DataType::FP32, {outer_size, rows, k}, 4);
        Tensor output(gpuPlacement, TensorDescriptor(DataType::FP32, geometry.output_dimensions));
        Tensor short_output(gpuPlacement, TensorDescriptor(DataType::FP32, {outer_size * k - 1}));
        const CubReductionStageSemantics semantics;
        EXPECT_THROW(launchFp32PacketComplete(fp8, output, geometry, 1, 16, stream, semantics), std::logic_error);
        EXPECT_THROW(launchFp32PacketComplete(input, output, geometry, 1, 125, stream, semantics), std::logic_error);
        EXPECT_THROW(launchFp32PacketComplete(input, output, geometry, 3, 16, stream, semantics), std::logic_error);
        EXPECT_THROW(launchFp32PacketComplete(misaligned, output, geometry, 1, 16, stream, semantics),
                     std::logic_error);
        EXPECT_THROW(launchFp32PacketComplete(input, short_output, geometry, 1, 16, stream, semantics),
                     std::logic_error);
        EXPECT_NO_THROW(launchFp32PacketComplete(input, output, geometry, 1, 16, stream, semantics));
        stream.synchronize();
    }
}

// 1190: the RuntimeK rotated first stage (PrefixReserved) on FP32 input. Every retained depth at K values across one
// and several 125-component tiles (two depths at the larger K), R past two shards with a partial last shard, every
// combine and transform: each FP32 partial [outer, shard, K] against a CPU reduction of its rows, exactly. A base 4 and
// 8 bytes past a 16-byte boundary gives the same partials. Then the rejections.
TEST(CubReductionFp32RK, RotatedPartialsMatchReference) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    const std::array<float, 8> pattern = {-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f, 2.0f};
    const std::vector<uint64_t> ks{2, 3, 5, 64, 125, 126, 250, 251, 1001};
    const std::vector<uint64_t> depths{32, 64, 128, 256, 512, 1024};
    const uint64_t outer_size = 3;
    EXPECT_FALSE(fp32RuntimeKRotatedSupports(3, 1000, 64, 100));   // not a retained depth
    EXPECT_FALSE(fp32RuntimeKRotatedSupports(3, 64, 64, 64));      // R must exceed the depth
    EXPECT_TRUE(fp32RuntimeKRotatedSupports(3, 65, 64, 64));
    const auto expected_partials = [&](const std::vector<float>& values, uint64_t rows, uint64_t k, uint64_t depth,
                                       CubReductionStageCombine combine, CubReductionStageInputTransform transform) {
        const uint64_t shards = (rows + depth - 1) / depth;
        std::vector<float> expected(outer_size * shards * k);
        for (uint64_t outer = 0; outer < outer_size; ++outer) {
            for (uint64_t shard = 0; shard < shards; ++shard) {
                const uint64_t first = shard * depth;
                const uint64_t count = std::min(depth, rows - first);
                const auto row = [&](uint64_t r) {
                    return values.begin() + static_cast<std::ptrdiff_t>((outer * rows + r) * k);
                };
                const std::vector<float> block(row(first), row(first + count));
                for (uint64_t c = 0; c < k; ++c)
                    expected[(outer * shards + shard) * k + c] = fp32RkReference(combine, transform, block, 0, count, k, c);
            }
        }
        return expected;
    };
    for (uint64_t k : ks) {
        const std::vector<uint64_t> k_depths = k <= 126 ? depths : std::vector<uint64_t>{32, 256};
        for (uint64_t depth : k_depths) {
            const uint64_t rows = 2 * depth + 37;
            const uint64_t shards = (rows + depth - 1) / depth;
            ASSERT_TRUE(fp32RuntimeKRotatedSupports(outer_size, rows, k, depth));
            std::vector<float> additive(outer_size * rows * k);
            std::vector<float> multiplicative(additive.size(), 1.0f);
            for (uint64_t i = 0; i < additive.size(); ++i) {
                additive[i] = pattern[(i * 5 + i / k) % pattern.size()];
                if (i % 97 == 3) multiplicative[i] = (i % 3 == 0) ? -1.0f : ((i % 3 == 1) ? 2.0f : 0.5f);
            }
            const auto geometry = CubReduction::analyzeValueGeometry(CubReductionOp::Sum, {outer_size, rows, k}, {1});
            const Tensor additive_input = makeGpuTensor(additive, {outer_size, rows, k}, stream, DataType::FP32);
            const Tensor product_input = makeGpuTensor(multiplicative, {outer_size, rows, k}, stream, DataType::FP32);
            for (CubReductionStageCombine combine :
                 {CubReductionStageCombine::Sum, CubReductionStageCombine::Product, CubReductionStageCombine::Minimum,
                  CubReductionStageCombine::Maximum}) {
                for (CubReductionStageInputTransform transform :
                     {CubReductionStageInputTransform::Identity, CubReductionStageInputTransform::AbsoluteValue,
                      CubReductionStageInputTransform::Square}) {
                    SCOPED_TRACE(::testing::Message() << "K=" << k << " depth=" << depth << " R=" << rows
                                                      << " combine " << static_cast<int>(combine) << " transform "
                                                      << static_cast<int>(transform));
                    const bool product = combine == CubReductionStageCombine::Product;
                    const Tensor& input = product ? product_input : additive_input;
                    const std::vector<float>& values = product ? multiplicative : additive;
                    ASSERT_GT(queryFp32RuntimeKRotatedOccupancy(depth, transform, combine).max_active_blocks_per_sm, 0);
                    Tensor partials(gpuPlacement, TensorDescriptor(DataType::FP32, {outer_size, shards, k}));
                    launchFp32RuntimeKRotatedFirstStage(input, partials, geometry, depth, shards, stream, transform,
                                                        combine);
                    stream.synchronize();
                    const auto actual = copyGpuTensorAsFloat(partials, stream);
                    const auto expected = expected_partials(values, rows, k, depth, combine, transform);
                    ASSERT_EQ(expected.size(), actual.size());
                    for (size_t i = 0; i < expected.size(); ++i) ASSERT_EQ(expected[i], actual[i]) << "index " << i;
                }
            }
        }
    }
    // A 4-byte aligned base: the same elements 4 and 8 bytes past a 16-byte boundary.
    {
        const uint64_t k = 251, depth = 64, rows = 2 * depth + 37, shards = 3;
        std::vector<float> values(outer_size * rows * k);
        for (uint64_t i = 0; i < values.size(); ++i) values[i] = pattern[(i * 5 + i / k) % pattern.size()];
        const auto geometry = CubReduction::analyzeValueGeometry(CubReductionOp::Sum, {outer_size, rows, k}, {1});
        const auto expected = expected_partials(values, rows, k, depth, CubReductionStageCombine::Sum,
                                                CubReductionStageInputTransform::Identity);
        for (uint64_t offset : {uint64_t{4}, uint64_t{8}}) {
            SCOPED_TRACE(::testing::Message() << "offset " << offset);
            std::vector<uint64_t> bytes(4 * values.size() + 16, 0);
            for (size_t i = 0; i < values.size(); ++i) {
                uint32_t word = 0;
                std::memcpy(&word, &values[i], sizeof(word));
                for (uint64_t b = 0; b < 4; ++b) bytes[offset + 4 * i + b] = (word >> (8 * b)) & 0xffU;
            }
            const Tensor storage = makeGpuUnsignedTensor(bytes, {bytes.size()}, stream, DataType::UINT8);
            const Tensor input = storage.aliasByteStorageAs(DataType::FP32, {outer_size, rows, k}, offset);
            Tensor partials(gpuPlacement, TensorDescriptor(DataType::FP32, {outer_size, shards, k}));
            launchFp32RuntimeKRotatedFirstStage(input, partials, geometry, depth, shards, stream);
            stream.synchronize();
            const auto actual = copyGpuTensorAsFloat(partials, stream);
            ASSERT_EQ(expected.size(), actual.size());
            for (size_t i = 0; i < expected.size(); ++i) ASSERT_EQ(expected[i], actual[i]) << "index " << i;
        }
    }
    // Rejections: FP8 input, a depth that is not retained, the wrong shard count, a wrong partials shape.
    {
        const uint64_t k = 16, rows = 200;
        const auto geometry = CubReduction::analyzeValueGeometry(CubReductionOp::Sum, {2, rows, k}, {1});
        const Tensor input = makeGpuTensor(std::vector<float>(2 * rows * k, 1.0f), {2, rows, k}, stream, DataType::FP32);
        const Tensor fp8 =
            makeGpuTensor(std::vector<float>(2 * rows * k, 1.0f), {2, rows, k}, stream, DataType::FP8_E4M3);
        Tensor partials(gpuPlacement, TensorDescriptor(DataType::FP32, {2, 4, k}));
        Tensor wrong(gpuPlacement, TensorDescriptor(DataType::FP32, {2, 3, k}));
        EXPECT_THROW(launchFp32RuntimeKRotatedFirstStage(fp8, partials, geometry, 64, 4, stream), std::logic_error);
        EXPECT_THROW(launchFp32RuntimeKRotatedFirstStage(input, partials, geometry, 50, 4, stream), std::logic_error);
        EXPECT_THROW(launchFp32RuntimeKRotatedFirstStage(input, wrong, geometry, 64, 3, stream), std::logic_error);
        EXPECT_THROW(launchFp32RuntimeKRotatedFirstStage(input, wrong, geometry, 64, 4, stream), std::logic_error);
        EXPECT_THROW(static_cast<void>(queryFp32RuntimeKRotatedOccupancy(100)), std::logic_error);
        EXPECT_NO_THROW(launchFp32RuntimeKRotatedFirstStage(input, partials, geometry, 64, 4, stream));
        stream.synchronize();
    }
}

namespace {

// Thor's ARG contract on FP32 rows: a NaN wins (the first NaN), else the minimum (maximum) value with the lowest row;
// -0 and +0 tie. Returns the winning row of [outer, R, K] at (outer, component).
uint64_t fp32ArgReferenceRow(const std::vector<float>& x, uint64_t outer, uint64_t rows, uint64_t k,
                             uint64_t component, bool maximum) {
    uint64_t best = 0;
    for (uint64_t r = 0; r < rows; ++r) {
        const float v = x[(outer * rows + r) * k + component];
        const float b = x[(outer * rows + best) * k + component];
        if (std::isnan(b)) break;
        if (std::isnan(v) || (maximum ? v > b : v < b)) best = r;
    }
    return best;
}

// Values with ties, both zeros, infinities and (optionally) NaNs, so every rule of the contract is exercised.
std::vector<float> fp32ArgValues(uint64_t count, uint64_t k, bool nans) {
    const std::array<float, 12> pattern = {-2.0f, 1.0f, -0.0f, 0.0f, 2.0f, -2.0f, 3.5f, -1.0f, 2.0f,
                                           std::numeric_limits<float>::infinity(),
                                           -std::numeric_limits<float>::infinity(), 0.5f};
    std::vector<float> values(count);
    for (uint64_t i = 0; i < count; ++i) {
        values[i] = pattern[(i * 7 + i / k * 3) % pattern.size()];
        if (nans && i % 997 == 501) values[i] = std::numeric_limits<float>::quiet_NaN();
    }
    return values;
}

}  // namespace

// 1198: the FP32 AlignedK ARG twin. Both packets, CTA 32 / 64 / 128, one shard (Complete) and FP8's shard targets of
// 32 and 128 rows (Staged plus the packed continuation, including its shard-split kernel at one output row with many
// shards), argmin and argmax, with and without NaNs: the value (FP32, sign of zero included) and the index (UINT32 at
// a domain stride, or UINT64) against the contract, exactly. Then the rejections.
TEST(CubReductionFp32RK, AlignedKArgMatchesReference) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    EXPECT_FALSE(fp32AlignedKArgSupports(3, 300, 6, 16, 32, 1));  // K % 4
    EXPECT_FALSE(fp32AlignedKArgSupports(3, 300, 7, 4, 32, 1));   // 1228: the 4-byte packet is VALUE only
    EXPECT_TRUE(fp32AlignedKArgSupports(3, 300, 6, 8, 32, 10));
    EXPECT_EQ(0u, fp32ArgPackedScratchBytes(3, 1, 8));
    EXPECT_EQ(3u * 10u * 8u * 8u, fp32ArgPackedScratchBytes(3, 10, 8));
    struct Site {
        uint64_t outer, rows, k;
    };
    const std::vector<Site> sites{{3, 300, 2}, {3, 300, 6}, {2, 300, 64}, {2, 150, 130}, {1, 77, 1028},
                                  {1, 5000, 4}, {4, 33, 8}};
    for (const Site& site : sites) {
        for (bool nans : {false, true}) {
            const std::vector<float> values = fp32ArgValues(site.outer * site.rows * site.k, site.k, nans);
            const Tensor input = makeGpuTensor(values, {site.outer, site.rows, site.k}, stream, DataType::FP32);
            const auto geometry =
                CubReduction::analyzeGeometry({site.outer, site.rows, site.k}, std::vector<uint32_t>{1});
            std::vector<uint64_t> shard_counts{1};
            for (uint64_t target : {32u, 128u})
                if (site.rows > target) shard_counts.push_back((site.rows + target - 1) / target);
            for (bool maximum : {false, true}) {
                std::vector<uint64_t> expected_rows(site.outer * site.k);
                for (uint64_t o = 0; o < site.outer; ++o)
                    for (uint64_t c = 0; c < site.k; ++c)
                        expected_rows[o * site.k + c] = fp32ArgReferenceRow(values, o, site.rows, site.k, c, maximum);
                for (uint64_t shards : shard_counts) {
                    for (uint32_t packet : {8u, 16u}) {
                        if (site.k % (packet / 4) != 0) continue;
                        for (uint32_t threads : {32u, 64u, 128u}) {
                            SCOPED_TRACE(::testing::Message()
                                         << "outer=" << site.outer << " R=" << site.rows << " K=" << site.k << " P"
                                         << packet << " CTA" << threads << " shards " << shards << " max "
                                         << maximum << " nans " << nans);
                            ASSERT_TRUE(fp32AlignedKArgSupports(site.outer, site.rows, site.k, packet, threads,
                                                                shards));
                            ASSERT_GT(queryFp32AlignedKArgOccupancy(packet, threads, shards > 1)
                                          .max_active_blocks_per_sm,
                                      0);
                            const uint64_t scratch_bytes = fp32ArgPackedScratchBytes(site.outer, shards, site.k);
                            std::optional<Tensor> scratch;
                            if (scratch_bytes != 0)
                                scratch = Tensor(gpuPlacement, TensorDescriptor(DataType::UINT8, {scratch_bytes}));
                            // UINT8 scratch, viewed as the 64-bit partials.
                            uint64_t* packed =
                                scratch ? static_cast<uint64_t*>(scratch->getMemPtr<void>()) : nullptr;
                            // Value (FP32) and UINT32 index at stride 3; then the UINT64 index alone.
                            Tensor value(gpuPlacement, TensorDescriptor(DataType::FP32, {site.outer, site.k}));
                            Tensor index32(gpuPlacement, TensorDescriptor(DataType::UINT32, {site.outer, site.k}));
                            launchFp32AlignedKArg(input, &value, &index32, geometry, packet, threads, shards, packed,
                                                  maximum, 3, stream);
                            Tensor index64(gpuPlacement, TensorDescriptor(DataType::UINT64, {site.outer, site.k}));
                            launchFp32AlignedKArg(input, nullptr, &index64, geometry, packet, threads, shards, packed,
                                                  maximum, 1, stream);
                            stream.synchronize();
                            const auto got_value = copyGpuTensorAsFloat(value, stream);
                            const auto got_index32 = copyGpuTensorAsUnsigned(index32, stream);
                            const auto got_index64 = copyGpuTensorAsUnsigned(index64, stream);
                            for (uint64_t o = 0; o < site.outer; ++o) {
                                for (uint64_t c = 0; c < site.k; ++c) {
                                    const uint64_t i = o * site.k + c;
                                    const uint64_t row = expected_rows[i];
                                    const float want = values[(o * site.rows + row) * site.k + c];
                                    ASSERT_EQ(3 * row, got_index32[i]) << "outer " << o << " k " << c;
                                    ASSERT_EQ(row, got_index64[i]) << "outer " << o << " k " << c;
                                    if (std::isnan(want)) {
                                        ASSERT_TRUE(std::isnan(got_value[i])) << "outer " << o << " k " << c;
                                    } else {
                                        ASSERT_EQ(std::bit_cast<uint32_t>(want), std::bit_cast<uint32_t>(got_value[i]))
                                            << "outer " << o << " k " << c;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // Rejections: FP8 input, no outputs, a staged launch without scratch, a UINT32 index that cannot hold the index.
    {
        const uint64_t outer = 2, rows = 300, k = 8;
        const auto geometry = CubReduction::analyzeGeometry({outer, rows, k}, std::vector<uint32_t>{1});
        const Tensor input = makeGpuTensor(fp32ArgValues(outer * rows * k, k, false), {outer, rows, k}, stream,
                                           DataType::FP32);
        const Tensor fp8 = makeGpuTensor(std::vector<float>(outer * rows * k, 1.0f), {outer, rows, k}, stream,
                                         DataType::FP8_E4M3);
        Tensor index(gpuPlacement, TensorDescriptor(DataType::UINT32, {outer, k}));
        EXPECT_THROW(launchFp32AlignedKArg(fp8, nullptr, &index, geometry, 16, 32, 1, nullptr, false, 1, stream),
                     std::logic_error);
        EXPECT_THROW(launchFp32AlignedKArg(input, nullptr, nullptr, geometry, 16, 32, 1, nullptr, false, 1, stream),
                     std::logic_error);
        EXPECT_THROW(launchFp32AlignedKArg(input, nullptr, &index, geometry, 16, 32, 10, nullptr, false, 1, stream),
                     std::logic_error);
        EXPECT_THROW(launchFp32AlignedKArg(input, nullptr, &index, geometry, 16, 32, 1, nullptr, false,
                                           uint64_t{1} << 32, stream),
                     std::logic_error);
        EXPECT_NO_THROW(launchFp32AlignedKArg(input, nullptr, &index, geometry, 16, 32, 1, nullptr, false, 1, stream));
        stream.synchronize();
    }
}

// 1200: the FP32 rotated ARG twin. Odd and even K across one and several 125-component tiles, R from 1 to 5000, FP8's
// rotated depths and one Complete shard (depth >= R), argmin and argmax, with and without NaNs, on bases 0, 4 and 12
// bytes past a 16-byte boundary: value bits and indices against the contract, exactly. Then the rejections.
TEST(CubReductionFp32RK, RotatedArgMatchesReference) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    EXPECT_FALSE(fp32RuntimeKArgSupports(3, 100, 5, 0));
    EXPECT_TRUE(fp32RuntimeKArgSupports(3, 100, 5, 1000));
    struct Site {
        uint64_t outer, rows, k;
    };
    const std::vector<Site> sites{{3, 1, 5}, {2, 33, 125}, {2, 300, 126}, {1, 300, 251}, {1, 5000, 3}, {2, 70, 1001}};
    for (const Site& site : sites) {
        for (uint64_t offset : {uint64_t{0}, uint64_t{4}, uint64_t{12}}) {
            for (bool nans : {false, true}) {
                const uint64_t count = site.outer * site.rows * site.k;
                const std::vector<float> values = fp32ArgValues(count, site.k, nans);
                std::vector<uint64_t> bytes(4 * count + 16, 0);
                for (uint64_t i = 0; i < count; ++i) {
                    uint32_t word = 0;
                    std::memcpy(&word, &values[i], sizeof(word));
                    for (uint64_t b = 0; b < 4; ++b) bytes[offset + 4 * i + b] = (word >> (8 * b)) & 0xffU;
                }
                const Tensor storage = makeGpuUnsignedTensor(bytes, {bytes.size()}, stream, DataType::UINT8);
                const Tensor input = storage.aliasByteStorageAs(DataType::FP32, {site.outer, site.rows, site.k}, offset);
                const auto geometry =
                    CubReduction::analyzeGeometry({site.outer, site.rows, site.k}, std::vector<uint32_t>{1});
                for (bool maximum : {false, true}) {
                    for (uint64_t depth : {uint64_t{32}, uint64_t{256}, uint64_t{1024}, site.rows}) {
                        SCOPED_TRACE(::testing::Message() << "outer=" << site.outer << " R=" << site.rows << " K="
                                                          << site.k << " offset " << offset << " depth " << depth
                                                          << " max " << maximum << " nans " << nans);
                        ASSERT_TRUE(fp32RuntimeKArgSupports(site.outer, site.rows, site.k, depth));
                        const uint64_t shards = (site.rows + depth - 1) / depth;
                        ASSERT_GT(queryFp32RuntimeKArgOccupancy(shards > 1).max_active_blocks_per_sm, 0);
                        const uint64_t scratch_bytes = fp32ArgPackedScratchBytes(site.outer, shards, site.k);
                        std::optional<Tensor> scratch;
                        if (scratch_bytes != 0)
                            scratch = Tensor(gpuPlacement, TensorDescriptor(DataType::UINT8, {scratch_bytes}));
                        uint64_t* packed = scratch ? static_cast<uint64_t*>(scratch->getMemPtr<void>()) : nullptr;
                        Tensor value(gpuPlacement, TensorDescriptor(DataType::FP32, {site.outer, site.k}));
                        Tensor index(gpuPlacement, TensorDescriptor(DataType::UINT64, {site.outer, site.k}));
                        launchFp32RuntimeKArg(input, &value, &index, geometry, depth, packed, maximum, 2, stream);
                        stream.synchronize();
                        const auto got_value = copyGpuTensorAsFloat(value, stream);
                        const auto got_index = copyGpuTensorAsUnsigned(index, stream);
                        for (uint64_t o = 0; o < site.outer; ++o) {
                            for (uint64_t c = 0; c < site.k; ++c) {
                                const uint64_t i = o * site.k + c;
                                const uint64_t row = fp32ArgReferenceRow(values, o, site.rows, site.k, c, maximum);
                                const float want = values[(o * site.rows + row) * site.k + c];
                                ASSERT_EQ(2 * row, got_index[i]) << "outer " << o << " k " << c;
                                if (std::isnan(want)) {
                                    ASSERT_TRUE(std::isnan(got_value[i])) << "outer " << o << " k " << c;
                                } else {
                                    ASSERT_EQ(std::bit_cast<uint32_t>(want), std::bit_cast<uint32_t>(got_value[i]))
                                        << "outer " << o << " k " << c;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // Rejections: FP8 input, no outputs, a staged depth without scratch, depth 0.
    {
        const uint64_t outer = 2, rows = 300, k = 7;
        const auto geometry = CubReduction::analyzeGeometry({outer, rows, k}, std::vector<uint32_t>{1});
        const Tensor input = makeGpuTensor(fp32ArgValues(outer * rows * k, k, false), {outer, rows, k}, stream,
                                           DataType::FP32);
        const Tensor fp8 = makeGpuTensor(std::vector<float>(outer * rows * k, 1.0f), {outer, rows, k}, stream,
                                         DataType::FP8_E4M3);
        Tensor index(gpuPlacement, TensorDescriptor(DataType::UINT32, {outer, k}));
        EXPECT_THROW(launchFp32RuntimeKArg(fp8, nullptr, &index, geometry, 1024, nullptr, false, 1, stream),
                     std::logic_error);
        EXPECT_THROW(launchFp32RuntimeKArg(input, nullptr, nullptr, geometry, 1024, nullptr, false, 1, stream),
                     std::logic_error);
        EXPECT_THROW(launchFp32RuntimeKArg(input, nullptr, &index, geometry, 64, nullptr, false, 1, stream),
                     std::logic_error);
        EXPECT_THROW(launchFp32RuntimeKArg(input, nullptr, &index, geometry, 0, nullptr, false, 1, stream),
                     std::logic_error);
        EXPECT_NO_THROW(launchFp32RuntimeKArg(input, nullptr, &index, geometry, 1024, nullptr, false, 1, stream));
        stream.synchronize();
    }
}

// 1201: the FP32 PacketComplete ARG twin. W1/W2/W4/W8, tiles of 124, 37 and 4 components, K from 1 to 300 (every row
// prefix), R from 1 to 300, argmin and argmax, with and without NaNs: value bits and indices (UINT32 at stride 5, or
// UINT64) against the contract, exactly. Then the rejections.
TEST(CubReductionFp32RK, PacketCompleteArgMatchesReference) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    EXPECT_FALSE(fp32PacketCompleteArgSupports(3, 10, 64, 1, 125));
    EXPECT_TRUE(fp32PacketCompleteArgSupports(3, 10, 64, 8, 124));
    ASSERT_GT(queryFp32PacketCompleteArgOccupancy().max_active_blocks_per_sm, 0);
    struct Site {
        uint64_t outer, rows, k;
    };
    const std::vector<Site> sites{{5, 1, 3}, {5, 9, 1}, {4, 40, 37}, {3, 2, 124}, {2, 300, 125}, {700, 3, 5},
                                  {1, 64, 300}};
    for (const Site& site : sites) {
        for (bool nans : {false, true}) {
            const std::vector<float> values = fp32ArgValues(site.outer * site.rows * site.k, site.k, nans);
            const Tensor input = makeGpuTensor(values, {site.outer, site.rows, site.k}, stream, DataType::FP32);
            const auto geometry =
                CubReduction::analyzeGeometry({site.outer, site.rows, site.k}, std::vector<uint32_t>{1});
            for (bool maximum : {false, true}) {
                for (uint32_t tile : {124u, 37u, 4u}) {
                    for (uint32_t w : {1u, 2u, 4u, 8u}) {
                        SCOPED_TRACE(::testing::Message() << "outer=" << site.outer << " R=" << site.rows << " K="
                                                          << site.k << " tile " << tile << " W" << w << " max "
                                                          << maximum << " nans " << nans);
                        ASSERT_TRUE(fp32PacketCompleteArgSupports(site.outer, site.rows, site.k, w, tile));
                        Tensor value(gpuPlacement, TensorDescriptor(DataType::FP32, {site.outer, site.k}));
                        Tensor index32(gpuPlacement, TensorDescriptor(DataType::UINT32, {site.outer, site.k}));
                        launchFp32PacketCompleteArg(input, &value, &index32, geometry, w, tile, maximum, 5, stream);
                        Tensor index64(gpuPlacement, TensorDescriptor(DataType::UINT64, {site.outer, site.k}));
                        launchFp32PacketCompleteArg(input, nullptr, &index64, geometry, w, tile, maximum, 1, stream);
                        stream.synchronize();
                        const auto got_value = copyGpuTensorAsFloat(value, stream);
                        const auto got_index32 = copyGpuTensorAsUnsigned(index32, stream);
                        const auto got_index64 = copyGpuTensorAsUnsigned(index64, stream);
                        for (uint64_t o = 0; o < site.outer; ++o) {
                            for (uint64_t c = 0; c < site.k; ++c) {
                                const uint64_t i = o * site.k + c;
                                const uint64_t row = fp32ArgReferenceRow(values, o, site.rows, site.k, c, maximum);
                                const float want = values[(o * site.rows + row) * site.k + c];
                                ASSERT_EQ(5 * row, got_index32[i]) << "outer " << o << " k " << c;
                                ASSERT_EQ(row, got_index64[i]) << "outer " << o << " k " << c;
                                if (std::isnan(want)) {
                                    ASSERT_TRUE(std::isnan(got_value[i])) << "outer " << o << " k " << c;
                                } else {
                                    ASSERT_EQ(std::bit_cast<uint32_t>(want), std::bit_cast<uint32_t>(got_value[i]))
                                        << "outer " << o << " k " << c;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // Rejections: FP8 input, no outputs, a tile past 124, W3, a misaligned base.
    {
        const uint64_t outer = 2, rows = 30, k = 16;
        const auto geometry = CubReduction::analyzeGeometry({outer, rows, k}, std::vector<uint32_t>{1});
        const Tensor input = makeGpuTensor(fp32ArgValues(outer * rows * k, k, false), {outer, rows, k}, stream,
                                           DataType::FP32);
        const Tensor fp8 = makeGpuTensor(std::vector<float>(outer * rows * k, 1.0f), {outer, rows, k}, stream,
                                         DataType::FP8_E4M3);
        const Tensor bytes = makeGpuUnsignedTensor(std::vector<uint64_t>(4 * outer * rows * k + 16, 0),
                                                   {4 * outer * rows * k + 16}, stream, DataType::UINT8);
        const Tensor misaligned = bytes.aliasByteStorageAs(DataType::FP32, {outer, rows, k}, 4);
        Tensor index(gpuPlacement, TensorDescriptor(DataType::UINT32, {outer, k}));
        EXPECT_THROW(launchFp32PacketCompleteArg(fp8, nullptr, &index, geometry, 1, 16, false, 1, stream),
                     std::logic_error);
        EXPECT_THROW(launchFp32PacketCompleteArg(input, nullptr, nullptr, geometry, 1, 16, false, 1, stream),
                     std::logic_error);
        EXPECT_THROW(launchFp32PacketCompleteArg(input, nullptr, &index, geometry, 1, 125, false, 1, stream),
                     std::logic_error);
        EXPECT_THROW(launchFp32PacketCompleteArg(input, nullptr, &index, geometry, 3, 16, false, 1, stream),
                     std::logic_error);
        EXPECT_THROW(launchFp32PacketCompleteArg(misaligned, nullptr, &index, geometry, 1, 16, false, 1, stream),
                     std::logic_error);
        EXPECT_NO_THROW(launchFp32PacketCompleteArg(input, nullptr, &index, geometry, 1, 16, false, 1, stream));
        stream.synchronize();
    }
}

// 1202: the FP32 OneWindow ARG twin. Every FP32 unit whose bucket holds K (O1 W1/W2/W4/W8, O2/O4/O8 W1; one-component
// buckets with power-of-two and other R-worker counts, the load-ahead unit), argmin and argmax, with and without NaNs:
// the value (FP32, sign of zero included) and the index (UINT32 at a domain stride, or UINT64) against the contract,
// exactly. Then the rejections.
TEST(CubReductionFp32RK, OneWindowArgMatchesReference) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    struct Site {
        uint64_t outer, rows;
    };
    const std::vector<Site> sites{{5, 37}, {1, 301}, {700, 3}, {9, 1}};
    const std::vector<uint64_t> ks{2, 3, 4, 5, 8, 13, 16, 17, 29, 32, 33, 61, 64, 65, 100, 125};
    for (const Site& site : sites) {
        for (uint64_t k : ks) {
            for (bool nans : {false, true}) {
                const std::vector<float> values = fp32ArgValues(site.outer * site.rows * k, k, nans);
                const Tensor input = makeGpuTensor(values, {site.outer, site.rows, k}, stream, DataType::FP32);
                const auto geometry = CubReduction::analyzeGeometry({site.outer, site.rows, k}, std::vector<uint32_t>{1});
                const auto units = fp32OneWindowRuntimeKUnits(k);
                ASSERT_FALSE(units.empty());
                for (const Fp8OneWindowRuntimeKUnit& unit : units) {
                    for (uint32_t w : {1u, 2u, 4u, 8u}) {
                        if (unit.outputs_per_warp > 1 && w != 1) continue;
                        ASSERT_TRUE(fp32OneWindowArgSupports(site.outer, site.rows, k, unit, w));
                        ASSERT_GT(queryFp32OneWindowArgOccupancy(k, unit, w).max_active_blocks_per_sm, 0);
                        for (bool maximum : {false, true}) {
                            SCOPED_TRACE(::testing::Message()
                                         << "outer=" << site.outer << " R=" << site.rows << " K=" << k << " O"
                                         << unit.outputs_per_warp << " W" << w << " bucket "
                                         << unit.max_components_per_lane << " load_ahead " << unit.load_ahead
                                         << " max " << maximum << " nans " << nans);
                            Tensor value(gpuPlacement, TensorDescriptor(DataType::FP32, {site.outer, k}));
                            Tensor index32(gpuPlacement, TensorDescriptor(DataType::UINT32, {site.outer, k}));
                            launchFp32OneWindowArg(input, &value, &index32, geometry, unit, w, maximum, 5, stream);
                            Tensor index64(gpuPlacement, TensorDescriptor(DataType::UINT64, {site.outer, k}));
                            launchFp32OneWindowArg(input, nullptr, &index64, geometry, unit, w, maximum, 1, stream);
                            stream.synchronize();
                            const auto got_value = copyGpuTensorAsFloat(value, stream);
                            const auto got_index32 = copyGpuTensorAsUnsigned(index32, stream);
                            const auto got_index64 = copyGpuTensorAsUnsigned(index64, stream);
                            for (uint64_t o = 0; o < site.outer; ++o) {
                                for (uint64_t c = 0; c < k; ++c) {
                                    const uint64_t i = o * k + c;
                                    const uint64_t row = fp32ArgReferenceRow(values, o, site.rows, k, c, maximum);
                                    const float want = values[(o * site.rows + row) * k + c];
                                    ASSERT_EQ(5 * row, got_index32[i]) << "outer " << o << " k " << c;
                                    ASSERT_EQ(row, got_index64[i]) << "outer " << o << " k " << c;
                                    if (std::isnan(want)) {
                                        ASSERT_TRUE(std::isnan(got_value[i])) << "outer " << o << " k " << c;
                                    } else {
                                        ASSERT_EQ(std::bit_cast<uint32_t>(want), std::bit_cast<uint32_t>(got_value[i]))
                                            << "outer " << o << " k " << c;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // Rejections: FP8 input, no outputs, K past the unit's bucket, O2 W2, a misaligned base.
    {
        const uint64_t outer = 2, rows = 30, k = 16;
        const auto geometry = CubReduction::analyzeGeometry({outer, rows, k}, std::vector<uint32_t>{1});
        const Tensor input = makeGpuTensor(fp32ArgValues(outer * rows * k, k, false), {outer, rows, k}, stream,
                                           DataType::FP32);
        const Tensor fp8 = makeGpuTensor(std::vector<float>(outer * rows * k, 1.0f), {outer, rows, k}, stream,
                                         DataType::FP8_E4M3);
        const Tensor bytes = makeGpuUnsignedTensor(std::vector<uint64_t>(4 * outer * rows * k + 16, 0),
                                                   {4 * outer * rows * k + 16}, stream, DataType::UINT8);
        const Tensor misaligned = bytes.aliasByteStorageAs(DataType::FP32, {outer, rows, k}, 4);
        Tensor index(gpuPlacement, TensorDescriptor(DataType::UINT32, {outer, k}));
        const Fp8OneWindowRuntimeKUnit o1{1, 16, 1, false};
        const Fp8OneWindowRuntimeKUnit o2{2, 16, 1, false};
        const Fp8OneWindowRuntimeKUnit o1_bucket4{1, 16, 4, false};
        EXPECT_FALSE(fp32OneWindowArgSupports(outer, rows, k, o1_bucket4, 1));
        EXPECT_FALSE(fp32OneWindowArgSupports(outer, rows, k, o2, 2));
        EXPECT_THROW(launchFp32OneWindowArg(fp8, nullptr, &index, geometry, o1, 1, false, 1, stream), std::logic_error);
        EXPECT_THROW(launchFp32OneWindowArg(input, nullptr, nullptr, geometry, o1, 1, false, 1, stream),
                     std::logic_error);
        EXPECT_THROW(launchFp32OneWindowArg(input, nullptr, &index, geometry, o1_bucket4, 1, false, 1, stream),
                     std::logic_error);
        EXPECT_THROW(launchFp32OneWindowArg(input, nullptr, &index, geometry, o2, 2, false, 1, stream),
                     std::logic_error);
        EXPECT_THROW(launchFp32OneWindowArg(misaligned, nullptr, &index, geometry, o1, 1, false, 1, stream),
                     std::logic_error);
        EXPECT_NO_THROW(launchFp32OneWindowArg(input, nullptr, &index, geometry, o1, 1, false, 1, stream));
        stream.synchronize();
    }
}

// 1203: the FP32 DeepRSmallK ARG twins. Every ownership (TinyK at K <= 4 / 8, one component per lane with and without
// lane sharing, two and four components per lane), every retained depth below R (the packed continuation, its
// shard-split kernel at one output row over many shards), argmin and argmax, with and without NaNs: the value (FP32,
// sign of zero included) and the index (UINT32 at a domain stride, or UINT64) against the contract, exactly. Then the
// rejections.
TEST(CubReductionFp32RK, DeepRSmallKArgMatchesReference) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    EXPECT_FALSE(fp32DeepRSmallKArgSupports(3, 64, 8, 64));    // R <= depth
    EXPECT_FALSE(fp32DeepRSmallKArgSupports(3, 300, 126, 64));  // K > 125
    EXPECT_FALSE(fp32DeepRSmallKArgSupports(3, 300, 8, 32));    // not a retained depth
    EXPECT_TRUE(fp32DeepRSmallKArgSupports(3, 300, 125, 64));
    for (uint64_t k : {3u, 8u, 13u, 32u, 64u, 125u}) {
        ASSERT_GT(queryFp32DeepRSmallKArgOccupancy(k).max_active_blocks_per_sm, 0);
    }
    struct Site {
        uint64_t outer, rows, k;
    };
    const std::vector<Site> sites{{3, 300, 1},  {3, 300, 3},  {2, 700, 8},   {3, 200, 13}, {2, 300, 17},
                                  {2, 129, 32}, {2, 300, 47}, {2, 1500, 64}, {2, 300, 65}, {2, 257, 125},
                                  {1, 5000, 4}, {1, 3000, 37}};
    for (const Site& site : sites) {
        for (bool nans : {false, true}) {
            const std::vector<float> values = fp32ArgValues(site.outer * site.rows * site.k, site.k, nans);
            const Tensor input = makeGpuTensor(values, {site.outer, site.rows, site.k}, stream, DataType::FP32);
            const auto geometry =
                CubReduction::analyzeGeometry({site.outer, site.rows, site.k}, std::vector<uint32_t>{1});
            for (uint64_t depth : {64u, 128u, 256u, 512u, 1024u}) {
                if (site.rows <= depth) continue;
                ASSERT_TRUE(fp32DeepRSmallKArgSupports(site.outer, site.rows, site.k, depth));
                const uint64_t shards = (site.rows + depth - 1) / depth;
                // UINT8 scratch, viewed as the 64-bit partials.
                const uint64_t scratch_bytes = fp32ArgPackedScratchBytes(site.outer, shards, site.k);
                Tensor scratch(gpuPlacement, TensorDescriptor(DataType::UINT8, {scratch_bytes}));
                uint64_t* packed = static_cast<uint64_t*>(scratch.getMemPtr<void>());
                for (bool maximum : {false, true}) {
                    SCOPED_TRACE(::testing::Message() << "outer=" << site.outer << " R=" << site.rows << " K="
                                                      << site.k << " depth " << depth << " max " << maximum
                                                      << " nans " << nans);
                    Tensor value(gpuPlacement, TensorDescriptor(DataType::FP32, {site.outer, site.k}));
                    Tensor index32(gpuPlacement, TensorDescriptor(DataType::UINT32, {site.outer, site.k}));
                    launchFp32DeepRSmallKArg(input, &value, &index32, geometry, depth, packed, maximum, 3, stream);
                    Tensor index64(gpuPlacement, TensorDescriptor(DataType::UINT64, {site.outer, site.k}));
                    launchFp32DeepRSmallKArg(input, nullptr, &index64, geometry, depth, packed, maximum, 1, stream);
                    stream.synchronize();
                    const auto got_value = copyGpuTensorAsFloat(value, stream);
                    const auto got_index32 = copyGpuTensorAsUnsigned(index32, stream);
                    const auto got_index64 = copyGpuTensorAsUnsigned(index64, stream);
                    for (uint64_t o = 0; o < site.outer; ++o) {
                        for (uint64_t c = 0; c < site.k; ++c) {
                            const uint64_t i = o * site.k + c;
                            const uint64_t row = fp32ArgReferenceRow(values, o, site.rows, site.k, c, maximum);
                            const float want = values[(o * site.rows + row) * site.k + c];
                            ASSERT_EQ(3 * row, got_index32[i]) << "outer " << o << " k " << c;
                            ASSERT_EQ(row, got_index64[i]) << "outer " << o << " k " << c;
                            if (std::isnan(want)) {
                                ASSERT_TRUE(std::isnan(got_value[i])) << "outer " << o << " k " << c;
                            } else {
                                ASSERT_EQ(std::bit_cast<uint32_t>(want), std::bit_cast<uint32_t>(got_value[i]))
                                    << "outer " << o << " k " << c;
                            }
                        }
                    }
                }
            }
        }
    }
    // Rejections: FP8 input, no outputs, no scratch, a depth that is not retained, R <= depth, a misaligned base.
    {
        const uint64_t outer = 2, rows = 300, k = 16;
        const auto geometry = CubReduction::analyzeGeometry({outer, rows, k}, std::vector<uint32_t>{1});
        const Tensor input = makeGpuTensor(fp32ArgValues(outer * rows * k, k, false), {outer, rows, k}, stream,
                                           DataType::FP32);
        const Tensor fp8 = makeGpuTensor(std::vector<float>(outer * rows * k, 1.0f), {outer, rows, k}, stream,
                                         DataType::FP8_E4M3);
        const Tensor bytes = makeGpuUnsignedTensor(std::vector<uint64_t>(4 * outer * rows * k + 16, 0),
                                                   {4 * outer * rows * k + 16}, stream, DataType::UINT8);
        const Tensor misaligned = bytes.aliasByteStorageAs(DataType::FP32, {outer, rows, k}, 4);
        Tensor scratch(gpuPlacement, TensorDescriptor(DataType::UINT8, {fp32ArgPackedScratchBytes(outer, 5, k)}));
        uint64_t* packed = static_cast<uint64_t*>(scratch.getMemPtr<void>());
        Tensor index(gpuPlacement, TensorDescriptor(DataType::UINT32, {outer, k}));
        const auto short_geometry = CubReduction::analyzeGeometry({outer, 64, k}, std::vector<uint32_t>{1});
        const Tensor short_input = makeGpuTensor(fp32ArgValues(outer * 64 * k, k, false), {outer, 64, k}, stream,
                                                 DataType::FP32);
        EXPECT_THROW(launchFp32DeepRSmallKArg(fp8, nullptr, &index, geometry, 64, packed, false, 1, stream),
                     std::logic_error);
        EXPECT_THROW(launchFp32DeepRSmallKArg(input, nullptr, nullptr, geometry, 64, packed, false, 1, stream),
                     std::logic_error);
        EXPECT_THROW(launchFp32DeepRSmallKArg(input, nullptr, &index, geometry, 64, nullptr, false, 1, stream),
                     std::logic_error);
        EXPECT_THROW(launchFp32DeepRSmallKArg(input, nullptr, &index, geometry, 96, packed, false, 1, stream),
                     std::logic_error);
        EXPECT_THROW(launchFp32DeepRSmallKArg(short_input, nullptr, &index, short_geometry, 64, packed, false, 1,
                                              stream),
                     std::logic_error);
        EXPECT_THROW(launchFp32DeepRSmallKArg(misaligned, nullptr, &index, geometry, 64, packed, false, 1, stream),
                     std::logic_error);
        EXPECT_NO_THROW(launchFp32DeepRSmallKArg(input, nullptr, &index, geometry, 64, packed, false, 1, stream));
        stream.synchronize();
    }
}

namespace {

// 1208: Thor's ARG contract on carried-index (intermediate) FP32 input: an element's composed index is carried + row *
// stride; the first NaN by composed index wins, else the minimum (maximum) value with the lowest composed index (-0 and
// +0 tie). Returns {winning row, composed index} of [outer, R, K] at (outer, component).
std::pair<uint64_t, uint64_t> fp32CarriedArgReference(const std::vector<float>& x, const std::vector<uint64_t>& carried,
                                                      uint64_t outer, uint64_t rows, uint64_t k, uint64_t component,
                                                      bool maximum, uint64_t stride) {
    const auto beats = [&](float a, uint64_t ia, float b, uint64_t ib) {
        if (std::isnan(a) || std::isnan(b)) return std::isnan(a) && (!std::isnan(b) || ia < ib);
        if (a == b) return ia < ib;
        return maximum ? a > b : a < b;
    };
    uint64_t best = 0;
    uint64_t best_index = carried[outer * rows * k + component];
    for (uint64_t r = 1; r < rows; ++r) {
        const uint64_t e = (outer * rows + r) * k + component;
        const uint64_t index = carried[e] + r * stride;
        if (beats(x[e], index, x[(outer * rows + best) * k + component], best_index)) {
            best = r;
            best_index = index;
        }
    }
    return {best, best_index};
}

// Carried indices that are not monotone in the row (as a later pass over an earlier pass's partials sees them), below
// `limit`.
std::vector<uint64_t> fp32CarriedIndices(uint64_t count, uint64_t limit) {
    std::vector<uint64_t> carried(count);
    uint64_t h = 0x9E3779B97F4A7C15ULL;
    for (uint64_t i = 0; i < count; ++i) {
        h ^= h >> 31;
        h *= 0xBF58476D1CE4E5B9ULL;
        h ^= h >> 29;
        carried[i] = (h + i) % limit;
    }
    return carried;
}

// Checks a carried pass's value bits and composed index (UINT64 output) against the contract.
void expectFp32CarriedArg(const std::vector<float>& values, const std::vector<uint64_t>& carried, uint64_t outer,
                          uint64_t rows, uint64_t k, bool maximum, uint64_t stride, const Tensor& value,
                          const Tensor& index, Stream& stream) {
    const auto got_value = copyGpuTensorAsFloat(value, stream);
    const auto got_index = copyGpuTensorAsUnsigned(index, stream);
    for (uint64_t o = 0; o < outer; ++o) {
        for (uint64_t c = 0; c < k; ++c) {
            const uint64_t i = o * k + c;
            const auto [row, composed] = fp32CarriedArgReference(values, carried, o, rows, k, c, maximum, stride);
            const float want = values[(o * rows + row) * k + c];
            ASSERT_EQ(composed, got_index[i]) << "outer " << o << " k " << c;
            if (std::isnan(want)) {
                ASSERT_TRUE(std::isnan(got_value[i])) << "outer " << o << " k " << c;
            } else {
                ASSERT_EQ(std::bit_cast<uint32_t>(want), std::bit_cast<uint32_t>(got_value[i]))
                    << "outer " << o << " k " << c;
            }
        }
    }
}

}  // namespace

// 1208: the carried-index AlignedK ARG twin (phase 8) and the wide continuation. Both packets, CTA 32 / 64 / 128, one
// shard and FP8's 32- and 128-row targets (the shard-split continuation at one output row over many shards), UINT32 and
// UINT64 carried indices that are not monotone in the row, row strides 0 (an intermediate run) and 7 (an original run),
// argmin and argmax, with and without NaNs: value bits and the composed index exactly. Then the rejections.
TEST(CubReductionFp32RK, AlignedKCarriedArgMatchesReference) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    EXPECT_EQ(0u, fp32ArgWideScratchBytes(3, 1, 8));
    EXPECT_EQ(1024u + 3u * 10u * 8u * 8u, fp32ArgWideScratchBytes(3, 10, 8));  // 960 key bytes round to 1024
    struct Site {
        uint64_t outer, rows, k;
    };
    const std::vector<Site> sites{{3, 300, 2}, {3, 300, 8}, {2, 300, 64}, {1, 77, 1028}, {1, 5000, 4}, {4, 33, 8}};
    for (const Site& site : sites) {
        const uint64_t count = site.outer * site.rows * site.k;
        for (bool nans : {false, true}) {
            const std::vector<float> values = fp32ArgValues(count, site.k, nans);
            const Tensor input = makeGpuTensor(values, {site.outer, site.rows, site.k}, stream, DataType::FP32);
            const auto geometry =
                CubReduction::analyzeGeometry({site.outer, site.rows, site.k}, std::vector<uint32_t>{1});
            std::vector<uint64_t> shard_counts{1};
            for (uint64_t target : {32u, 128u})
                if (site.rows > target) shard_counts.push_back((site.rows + target - 1) / target);
            for (DataType index_dtype : {DataType::UINT32, DataType::UINT64}) {
                const uint64_t limit = index_dtype == DataType::UINT32 ? 1000000 : (uint64_t{1} << 40);
                const std::vector<uint64_t> carried = fp32CarriedIndices(count, limit);
                const Tensor carried_index =
                    makeGpuUnsignedTensor(carried, {site.outer, site.rows, site.k}, stream, index_dtype);
                for (uint64_t stride : {0u, 7u}) {
                    for (bool maximum : {false, true}) {
                        for (uint64_t shards : shard_counts) {
                            for (uint32_t packet : {8u, 16u}) {
                                if (site.k % (packet / 4) != 0) continue;
                                for (uint32_t threads : {32u, 64u, 128u}) {
                                    SCOPED_TRACE(::testing::Message()
                                                 << "outer=" << site.outer << " R=" << site.rows << " K=" << site.k
                                                 << " P" << packet << " CTA" << threads << " shards " << shards
                                                 << " index " << static_cast<int>(index_dtype) << " stride "
                                                 << stride << " max " << maximum << " nans " << nans);
                                    ASSERT_GT(queryFp32AlignedKCarriedArgOccupancy(packet, threads, shards > 1,
                                                                                   index_dtype)
                                                  .max_active_blocks_per_sm,
                                              0);
                                    const uint64_t scratch_bytes =
                                        fp32ArgWideScratchBytes(site.outer, shards, site.k);
                                    std::optional<Tensor> scratch;
                                    if (scratch_bytes != 0)
                                        scratch =
                                            Tensor(gpuPlacement, TensorDescriptor(DataType::UINT8, {scratch_bytes}));
                                    Tensor value(gpuPlacement, TensorDescriptor(DataType::FP32, {site.outer, site.k}));
                                    Tensor index(gpuPlacement,
                                                 TensorDescriptor(DataType::UINT64, {site.outer, site.k}));
                                    launchFp32AlignedKCarriedArg(input, carried_index, &value, &index, geometry, packet,
                                                                 threads, shards,
                                                                 scratch ? scratch->getMemPtr<void>() : nullptr,
                                                                 maximum, stride, stream);
                                    stream.synchronize();
                                    expectFp32CarriedArg(values, carried, site.outer, site.rows, site.k, maximum,
                                                         stride, value, index, stream);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // Rejections: no scratch for a staged pass, a carried index of another size or dtype, no outputs.
    {
        const uint64_t outer = 2, rows = 300, k = 16;
        const auto geometry = CubReduction::analyzeGeometry({outer, rows, k}, std::vector<uint32_t>{1});
        const Tensor input = makeGpuTensor(fp32ArgValues(outer * rows * k, k, false), {outer, rows, k}, stream,
                                           DataType::FP32);
        const Tensor carried = makeGpuUnsignedTensor(fp32CarriedIndices(outer * rows * k, 1000), {outer, rows, k},
                                                     stream, DataType::UINT32);
        const Tensor short_carried =
            makeGpuUnsignedTensor(fp32CarriedIndices(outer * k, 1000), {outer, k}, stream, DataType::UINT32);
        const Tensor byte_carried = makeGpuUnsignedTensor(fp32CarriedIndices(outer * rows * k, 200), {outer, rows, k},
                                                          stream, DataType::UINT8);
        Tensor scratch(gpuPlacement,
                       TensorDescriptor(DataType::UINT8, {fp32ArgWideScratchBytes(outer, 10, k)}));
        Tensor index(gpuPlacement, TensorDescriptor(DataType::UINT32, {outer, k}));
        EXPECT_THROW(launchFp32AlignedKCarriedArg(input, carried, nullptr, &index, geometry, 16, 64, 10, nullptr, false,
                                                  1, stream),
                     std::logic_error);
        EXPECT_THROW(launchFp32AlignedKCarriedArg(input, short_carried, nullptr, &index, geometry, 16, 64, 10,
                                                  scratch.getMemPtr<void>(), false, 1, stream),
                     std::logic_error);
        EXPECT_THROW(launchFp32AlignedKCarriedArg(input, byte_carried, nullptr, &index, geometry, 16, 64, 10,
                                                  scratch.getMemPtr<void>(), false, 1, stream),
                     std::logic_error);
        EXPECT_THROW(launchFp32AlignedKCarriedArg(input, carried, nullptr, nullptr, geometry, 16, 64, 10,
                                                  scratch.getMemPtr<void>(), false, 1, stream),
                     std::logic_error);
        EXPECT_NO_THROW(launchFp32AlignedKCarriedArg(input, carried, nullptr, &index, geometry, 16, 64, 10,
                                                     scratch.getMemPtr<void>(), false, 1, stream));
        stream.synchronize();
    }
}

// 1210: the carried-index rotated ARG twin (phase 8). K over one or several 125-component tiles (odd and even), R 1 to
// 5000, depths 32 / 256 / 1024 and R (one Complete pass), UINT32 and UINT64 carried indices that are not monotone in
// the row, row strides 0 and 7, argmin and argmax, with and without NaNs: value bits and the composed index exactly.
// Then the rejections (unaligned bases, a short carried index, no scratch).
TEST(CubReductionFp32RK, RotatedCarriedArgMatchesReference) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    EXPECT_FALSE(fp32RuntimeKCarriedArgSupports(3, 10, 64, 0));
    EXPECT_TRUE(fp32RuntimeKCarriedArgSupports(3, 10, 64, 1000));
    struct Site {
        uint64_t outer, rows, k;
    };
    const std::vector<Site> sites{{5, 1, 3}, {4, 40, 125}, {3, 300, 126}, {2, 1500, 251}, {1, 5000, 64}, {6, 9, 1}};
    for (const Site& site : sites) {
        const uint64_t count = site.outer * site.rows * site.k;
        for (bool nans : {false, true}) {
            const std::vector<float> values = fp32ArgValues(count, site.k, nans);
            const Tensor input = makeGpuTensor(values, {site.outer, site.rows, site.k}, stream, DataType::FP32);
            const auto geometry =
                CubReduction::analyzeGeometry({site.outer, site.rows, site.k}, std::vector<uint32_t>{1});
            for (DataType index_dtype : {DataType::UINT32, DataType::UINT64}) {
                const uint64_t limit = index_dtype == DataType::UINT32 ? 1000000 : (uint64_t{1} << 40);
                const std::vector<uint64_t> carried = fp32CarriedIndices(count, limit);
                const Tensor carried_index =
                    makeGpuUnsignedTensor(carried, {site.outer, site.rows, site.k}, stream, index_dtype);
                for (uint64_t depth : {uint64_t{32}, uint64_t{256}, uint64_t{1024}, site.rows}) {
                    const uint64_t shards = (site.rows + depth - 1) / depth;
                    ASSERT_GT(queryFp32RuntimeKCarriedArgOccupancy(shards > 1, index_dtype).max_active_blocks_per_sm,
                              0);
                    for (uint64_t stride : {0u, 7u}) {
                        for (bool maximum : {false, true}) {
                            SCOPED_TRACE(::testing::Message()
                                         << "outer=" << site.outer << " R=" << site.rows << " K=" << site.k
                                         << " depth " << depth << " index " << static_cast<int>(index_dtype)
                                         << " stride " << stride << " max " << maximum << " nans " << nans);
                            ASSERT_TRUE(fp32RuntimeKCarriedArgSupports(site.outer, site.rows, site.k, depth));
                            const uint64_t scratch_bytes = fp32ArgWideScratchBytes(site.outer, shards, site.k);
                            std::optional<Tensor> scratch;
                            if (scratch_bytes != 0)
                                scratch = Tensor(gpuPlacement, TensorDescriptor(DataType::UINT8, {scratch_bytes}));
                            Tensor value(gpuPlacement, TensorDescriptor(DataType::FP32, {site.outer, site.k}));
                            Tensor index(gpuPlacement, TensorDescriptor(DataType::UINT64, {site.outer, site.k}));
                            launchFp32RuntimeKCarriedArg(input, carried_index, &value, &index, geometry, depth,
                                                         scratch ? scratch->getMemPtr<void>() : nullptr, maximum,
                                                         stride, stream);
                            stream.synchronize();
                            expectFp32CarriedArg(values, carried, site.outer, site.rows, site.k, maximum, stride,
                                                 value, index, stream);
                        }
                    }
                }
            }
        }
    }
    {
        const uint64_t outer = 2, rows = 300, k = 16;
        const auto geometry = CubReduction::analyzeGeometry({outer, rows, k}, std::vector<uint32_t>{1});
        const Tensor input = makeGpuTensor(fp32ArgValues(outer * rows * k, k, false), {outer, rows, k}, stream,
                                           DataType::FP32);
        const Tensor carried = makeGpuUnsignedTensor(fp32CarriedIndices(outer * rows * k, 1000), {outer, rows, k},
                                                     stream, DataType::UINT32);
        const Tensor short_carried =
            makeGpuUnsignedTensor(fp32CarriedIndices(outer * k, 1000), {outer, k}, stream, DataType::UINT32);
        const Tensor bytes = makeGpuUnsignedTensor(std::vector<uint64_t>(4 * outer * rows * k + 16, 0),
                                                   {4 * outer * rows * k + 16}, stream, DataType::UINT8);
        const Tensor misaligned = bytes.aliasByteStorageAs(DataType::FP32, {outer, rows, k}, 4);
        const Tensor misaligned_carried = bytes.aliasByteStorageAs(DataType::UINT32, {outer, rows, k}, 4);
        Tensor scratch(gpuPlacement, TensorDescriptor(DataType::UINT8, {fp32ArgWideScratchBytes(outer, 10, k)}));
        Tensor index(gpuPlacement, TensorDescriptor(DataType::UINT32, {outer, k}));
        void* wide = scratch.getMemPtr<void>();
        EXPECT_THROW(launchFp32RuntimeKCarriedArg(misaligned, carried, nullptr, &index, geometry, 32, wide, false, 1,
                                                  stream),
                     std::logic_error);
        EXPECT_THROW(launchFp32RuntimeKCarriedArg(input, misaligned_carried, nullptr, &index, geometry, 32, wide, false,
                                                  1, stream),
                     std::logic_error);
        EXPECT_THROW(launchFp32RuntimeKCarriedArg(input, short_carried, nullptr, &index, geometry, 32, wide, false, 1,
                                                  stream),
                     std::logic_error);
        EXPECT_THROW(launchFp32RuntimeKCarriedArg(input, carried, nullptr, &index, geometry, 32, nullptr, false, 1,
                                                  stream),
                     std::logic_error);
        EXPECT_NO_THROW(launchFp32RuntimeKCarriedArg(input, carried, nullptr, &index, geometry, 32, wide, false, 1,
                                                     stream));
        EXPECT_NO_THROW(launchFp32RuntimeKCarriedArg(input, carried, nullptr, &index, geometry, rows, nullptr, false,
                                                     1, stream));
        stream.synchronize();
    }
}

// 1212: the carried-index OneWindow ARG twin (phase 8). Every FP32 unit whose bucket holds K (O1 W1/W2/W4/W8, O2/O4/O8
// W1; power-of-two and other R-worker counts; the load-ahead unit), UINT32 and UINT64 carried indices that are not
// monotone in the row, row strides 0 and 7, argmin and argmax, with and without NaNs: value bits and the composed index
// exactly. Then the rejections (unaligned bases, a short carried index).
TEST(CubReductionFp32RK, OneWindowCarriedArgMatchesReference) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    struct Site {
        uint64_t outer, rows;
    };
    const std::vector<Site> sites{{5, 37}, {1, 301}, {700, 3}, {9, 1}};
    const std::vector<uint64_t> ks{2, 3, 4, 5, 8, 13, 17, 32, 33, 61, 64, 100, 125};
    for (const Site& site : sites) {
        for (uint64_t k : ks) {
            const uint64_t count = site.outer * site.rows * k;
            const std::vector<float> values = fp32ArgValues(count, k, (k % 2) == 1);
            const Tensor input = makeGpuTensor(values, {site.outer, site.rows, k}, stream, DataType::FP32);
            const auto geometry = CubReduction::analyzeGeometry({site.outer, site.rows, k}, std::vector<uint32_t>{1});
            for (DataType index_dtype : {DataType::UINT32, DataType::UINT64}) {
                const uint64_t limit = index_dtype == DataType::UINT32 ? 1000000 : (uint64_t{1} << 40);
                const std::vector<uint64_t> carried = fp32CarriedIndices(count, limit);
                const Tensor carried_index =
                    makeGpuUnsignedTensor(carried, {site.outer, site.rows, k}, stream, index_dtype);
                for (const Fp8OneWindowRuntimeKUnit& unit : fp32OneWindowRuntimeKUnits(k)) {
                    for (uint32_t w : {1u, 2u, 4u, 8u}) {
                        if (unit.outputs_per_warp > 1 && w != 1) continue;
                        ASSERT_TRUE(fp32OneWindowCarriedArgSupports(site.outer, site.rows, k, unit, w));
                        ASSERT_GT(queryFp32OneWindowCarriedArgOccupancy(k, unit, w, index_dtype)
                                      .max_active_blocks_per_sm,
                                  0);
                        for (uint64_t stride : {0u, 7u}) {
                            for (bool maximum : {false, true}) {
                                SCOPED_TRACE(::testing::Message()
                                             << "outer=" << site.outer << " R=" << site.rows << " K=" << k << " O"
                                             << unit.outputs_per_warp << " W" << w << " bucket "
                                             << unit.max_components_per_lane << " index "
                                             << static_cast<int>(index_dtype) << " stride " << stride << " max "
                                             << maximum);
                                Tensor value(gpuPlacement, TensorDescriptor(DataType::FP32, {site.outer, k}));
                                Tensor index(gpuPlacement, TensorDescriptor(DataType::UINT64, {site.outer, k}));
                                launchFp32OneWindowCarriedArg(input, carried_index, &value, &index, geometry, unit, w,
                                                              maximum, stride, stream);
                                stream.synchronize();
                                expectFp32CarriedArg(values, carried, site.outer, site.rows, k, maximum, stride,
                                                     value, index, stream);
                            }
                        }
                    }
                }
            }
        }
    }
    {
        const uint64_t outer = 2, rows = 30, k = 16;
        const auto geometry = CubReduction::analyzeGeometry({outer, rows, k}, std::vector<uint32_t>{1});
        const Tensor input = makeGpuTensor(fp32ArgValues(outer * rows * k, k, false), {outer, rows, k}, stream,
                                           DataType::FP32);
        const Tensor carried = makeGpuUnsignedTensor(fp32CarriedIndices(outer * rows * k, 1000), {outer, rows, k},
                                                     stream, DataType::UINT32);
        const Tensor short_carried =
            makeGpuUnsignedTensor(fp32CarriedIndices(outer * k, 1000), {outer, k}, stream, DataType::UINT32);
        const Tensor bytes = makeGpuUnsignedTensor(std::vector<uint64_t>(4 * outer * rows * k + 16, 0),
                                                   {4 * outer * rows * k + 16}, stream, DataType::UINT8);
        const Tensor misaligned = bytes.aliasByteStorageAs(DataType::FP32, {outer, rows, k}, 4);
        const Tensor misaligned_carried = bytes.aliasByteStorageAs(DataType::UINT32, {outer, rows, k}, 4);
        Tensor index(gpuPlacement, TensorDescriptor(DataType::UINT32, {outer, k}));
        const Fp8OneWindowRuntimeKUnit o1{1, 16, 1, false};
        EXPECT_THROW(launchFp32OneWindowCarriedArg(misaligned, carried, nullptr, &index, geometry, o1, 1, false, 1,
                                                   stream),
                     std::logic_error);
        EXPECT_THROW(launchFp32OneWindowCarriedArg(input, misaligned_carried, nullptr, &index, geometry, o1, 1, false,
                                                   1, stream),
                     std::logic_error);
        EXPECT_THROW(launchFp32OneWindowCarriedArg(input, short_carried, nullptr, &index, geometry, o1, 1, false, 1,
                                                   stream),
                     std::logic_error);
        EXPECT_NO_THROW(launchFp32OneWindowCarriedArg(input, carried, nullptr, &index, geometry, o1, 1, false, 1,
                                                      stream));
        stream.synchronize();
    }
}

// 1213: the carried-index DeepRSmallK ARG twins (phase 8). Every ownership (TinyK at K <= 4 / 8, one component per lane
// with and without lane sharing, two and four components per lane), every retained depth below R (the wide
// continuation, its shard-split form at one output row over many shards), UINT32 and UINT64 carried indices that are
// not monotone in the row, row strides 0 and 7, argmin and argmax, with and without NaNs: value bits and the composed
// index exactly. Then the rejections.
TEST(CubReductionFp32RK, DeepRSmallKCarriedArgMatchesReference) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    struct Site {
        uint64_t outer, rows, k;
    };
    const std::vector<Site> sites{{3, 300, 1},  {3, 300, 3}, {2, 700, 8},   {3, 200, 13}, {2, 300, 17},
                                  {2, 129, 32}, {2, 300, 47}, {2, 1500, 64}, {2, 257, 125}, {1, 5000, 4}};
    for (const Site& site : sites) {
        const uint64_t count = site.outer * site.rows * site.k;
        const std::vector<float> values = fp32ArgValues(count, site.k, (site.k % 2) == 1);
        const Tensor input = makeGpuTensor(values, {site.outer, site.rows, site.k}, stream, DataType::FP32);
        const auto geometry = CubReduction::analyzeGeometry({site.outer, site.rows, site.k}, std::vector<uint32_t>{1});
        for (DataType index_dtype : {DataType::UINT32, DataType::UINT64}) {
            ASSERT_GT(queryFp32DeepRSmallKCarriedArgOccupancy(site.k, index_dtype).max_active_blocks_per_sm, 0);
            const uint64_t limit = index_dtype == DataType::UINT32 ? 1000000 : (uint64_t{1} << 40);
            const std::vector<uint64_t> carried = fp32CarriedIndices(count, limit);
            const Tensor carried_index =
                makeGpuUnsignedTensor(carried, {site.outer, site.rows, site.k}, stream, index_dtype);
            for (uint64_t depth : {64u, 128u, 256u, 512u, 1024u}) {
                if (site.rows <= depth) continue;
                const uint64_t shards = (site.rows + depth - 1) / depth;
                Tensor scratch(gpuPlacement, TensorDescriptor(DataType::UINT8,
                                                              {fp32ArgWideScratchBytes(site.outer, shards, site.k)}));
                for (uint64_t stride : {0u, 7u}) {
                    for (bool maximum : {false, true}) {
                        SCOPED_TRACE(::testing::Message()
                                     << "outer=" << site.outer << " R=" << site.rows << " K=" << site.k << " depth "
                                     << depth << " index " << static_cast<int>(index_dtype) << " stride " << stride
                                     << " max " << maximum);
                        Tensor value(gpuPlacement, TensorDescriptor(DataType::FP32, {site.outer, site.k}));
                        Tensor index(gpuPlacement, TensorDescriptor(DataType::UINT64, {site.outer, site.k}));
                        launchFp32DeepRSmallKCarriedArg(input, carried_index, &value, &index, geometry, depth,
                                                        scratch.getMemPtr<void>(), maximum, stride, stream);
                        stream.synchronize();
                        expectFp32CarriedArg(values, carried, site.outer, site.rows, site.k, maximum, stride, value,
                                             index, stream);
                    }
                }
            }
        }
    }
    {
        const uint64_t outer = 2, rows = 300, k = 16;
        const auto geometry = CubReduction::analyzeGeometry({outer, rows, k}, std::vector<uint32_t>{1});
        const Tensor input = makeGpuTensor(fp32ArgValues(outer * rows * k, k, false), {outer, rows, k}, stream,
                                           DataType::FP32);
        const Tensor carried = makeGpuUnsignedTensor(fp32CarriedIndices(outer * rows * k, 1000), {outer, rows, k},
                                                     stream, DataType::UINT32);
        const Tensor short_carried =
            makeGpuUnsignedTensor(fp32CarriedIndices(outer * k, 1000), {outer, k}, stream, DataType::UINT32);
        Tensor scratch(gpuPlacement, TensorDescriptor(DataType::UINT8, {fp32ArgWideScratchBytes(outer, 5, k)}));
        Tensor index(gpuPlacement, TensorDescriptor(DataType::UINT32, {outer, k}));
        void* wide = scratch.getMemPtr<void>();
        EXPECT_THROW(launchFp32DeepRSmallKCarriedArg(input, carried, nullptr, &index, geometry, 64, nullptr, false, 1,
                                                     stream),
                     std::logic_error);
        EXPECT_THROW(launchFp32DeepRSmallKCarriedArg(input, short_carried, nullptr, &index, geometry, 64, wide, false,
                                                     1, stream),
                     std::logic_error);
        EXPECT_THROW(launchFp32DeepRSmallKCarriedArg(input, carried, nullptr, &index, geometry, 96, wide, false, 1,
                                                     stream),
                     std::logic_error);
        EXPECT_NO_THROW(launchFp32DeepRSmallKCarriedArg(input, carried, nullptr, &index, geometry, 64, wide, false, 1,
                                                        stream));
        stream.synchronize();
    }
}
