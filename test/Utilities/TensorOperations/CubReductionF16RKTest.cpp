// 1240: retained RK VALUE kernels on FP16 / BF16 input (plans/Thor_FP16_BF16_Dense_Reduction_Plan, phase 5; not
// routed yet). OneWindow: the 2-byte unit inventory (FP8's units of buckets <= 8) and its K coverage, and every unit
// at K values across its bucket against a CPU reference, for every combine and transform, W1/W2/W4/W8, both formats, a
// negative output scale and a narrow output dtype. Inputs are quarters with small sums and products of +-1/2/0.5
// (exact in both formats), so every result is exact in FP32 regardless of order. 1242: DeepRSmallK's FP32 partials, the
// same way, at every 2-byte ownership. 1243: AlignedK's Complete outputs and Staged partials at every packet and CTA.
// 1244: RuntimeK PacketComplete's outputs at every W and a full and a narrow tile. 1245: RuntimeK rotated's FP32
// partials at every retained depth and any 2-byte aligned base. 1246: every candidate pool entry (production
// residency) through the shared physical launch adapters. 1248: the AlignedK ARG twin (widened to the FP32 key).
// 1249: the rotated and PacketComplete ARG twins. 1250: the OneWindow and DeepRSmallK ARG twins. 1251: every ARG
// candidate (production residency) through the pass launcher.

#include "test/Utilities/TensorOperations/CubReductionTestSupport.h"
#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"
#include "Utilities/TensorOperations/Cub/CubReductionStageSemantics.h"
#include "Utilities/TensorOperations/Cub/F16RK/F16RKArgCandidates.h"
#include "Utilities/TensorOperations/Cub/F16RK/F16RKCandidates.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKPhysicalKernels.h"
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

// 2-byte OneWindow covers K <= 249 (P16 O1 holds 256 elements, less 7 for a misaligned row), and K = 256.
bool f16OneWindowShouldCover(uint64_t k) { return k >= 1 && (k <= 249 || k == 256); }

float f16RkReference(CubReductionInternal::CubReductionStageCombine combine,
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

const char* formatName(DataType dtype) { return dtype == DataType::FP16 ? "fp16" : "bf16"; }

}  // namespace

TEST(CubReductionF16RK, OneWindowInventoryAndCoverage) {
    using namespace CubReductionInternal;
    for (uint64_t k = 1; k <= 520; ++k) {
        const auto units = f16OneWindowRuntimeKUnits(k);
        EXPECT_EQ(f16OneWindowShouldCover(k), !units.empty()) << k;
        for (const auto& unit : units) {
            EXPECT_LE(unit.max_components_per_lane, 8u) << k;
            EXPECT_EQ(16u, unit.packet_bytes);
        }
    }
    // 14 units: FP8's 15 without the bucket-16 one.
    std::vector<Fp8OneWindowRuntimeKUnit> all;
    for (uint64_t k = 1; k <= 256; ++k)
        for (const auto& unit : f16OneWindowRuntimeKUnits(k))
            if (std::find(all.begin(), all.end(), unit) == all.end()) all.push_back(unit);
    EXPECT_EQ(14u, all.size());
    // Per ownership: O2 holds 121 (128 aligned), O4 57 (64) from K 9 (no bucket-1 O4 unit), O8 up to 16.
    const auto owns = [](uint64_t k, uint32_t outputs_per_warp) {
        const auto units = f16OneWindowRuntimeKUnits(k);
        return std::any_of(units.begin(), units.end(),
                           [&](const auto& unit) { return unit.outputs_per_warp == outputs_per_warp; });
    };
    EXPECT_TRUE(owns(121, 2));
    EXPECT_FALSE(owns(122, 2));
    EXPECT_TRUE(owns(128, 2));
    EXPECT_TRUE(owns(57, 4));
    EXPECT_FALSE(owns(58, 4));
    EXPECT_TRUE(owns(64, 4));
    EXPECT_FALSE(owns(8, 4));
    EXPECT_TRUE(owns(9, 4));
    EXPECT_TRUE(owns(16, 8));
    EXPECT_FALSE(owns(17, 8));
}

TEST(CubReductionF16RK, OneWindowUnitsMatchReference) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    const std::array<float, 8> pattern = {-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f, 2.0f};
    // (outer, R): a few outputs over a moderate R, one output over a long R (multi-warp O1 rounds), many outputs.
    const std::vector<std::pair<uint64_t, uint64_t>> shapes{{5, 37}, {1, 301}, {700, 3}};
    // K >= 2: [outer, R, 1] is a KR site (not the tiled RK path), as in the FP8 and FP32 OneWindow tests.
    const std::vector<uint64_t> ks{2,  3,  4,  5,  7,   8,   9,   13,  16,  17,  25,  31,  32,  33, 57,
                                   61, 64, 65, 100, 121, 124, 128, 129, 160, 200, 248, 249, 256};
    for (const DataType dtype : {DataType::FP16, DataType::BF16}) {
        for (const auto& [outer_size, rows] : shapes) {
            for (uint64_t k : ks) {
                SCOPED_TRACE(::testing::Message() << formatName(dtype) << " outer=" << outer_size << " R=" << rows
                                                  << " K=" << k);
                std::vector<float> additive(outer_size * rows * k);
                std::vector<float> multiplicative(additive.size(), 1.0f);
                for (uint64_t i = 0; i < additive.size(); ++i) {
                    additive[i] = pattern[(i * 5 + i / k) % pattern.size()];
                    if (i % 7 == 3) multiplicative[i] = (i % 3 == 0) ? -1.0f : ((i % 3 == 1) ? 2.0f : 0.5f);
                }
                const auto geometry =
                    CubReduction::analyzeValueGeometry(CubReductionOp::Sum, {outer_size, rows, k}, {1});
                ASSERT_EQ(geometry.path, CubReductionPath::TiledFixedSegment);
                const Tensor additive_input = makeGpuTensor(additive, {outer_size, rows, k}, stream, dtype);
                const Tensor product_input = makeGpuTensor(multiplicative, {outer_size, rows, k}, stream, dtype);
                const auto units = f16OneWindowRuntimeKUnits(k);
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
                                const auto occupancy = queryF16OneWindowRuntimeKOccupancy(dtype, k, unit, w, combine);
                                ASSERT_GT(occupancy.max_active_blocks_per_sm, 0);
                                Tensor output(gpuPlacement,
                                              TensorDescriptor(DataType::FP32, geometry.output_dimensions));
                                launchF16OneWindowRuntimeKComplete(input, output, geometry, unit, w, stream,
                                                                   semantics, output_scale);
                                stream.synchronize();
                                const auto actual = copyGpuTensorAsFloat(output, stream);
                                ASSERT_EQ(actual.size(), outer_size * k);
                                for (uint64_t outer = 0; outer < outer_size; ++outer) {
                                    for (uint64_t c = 0; c < k; ++c) {
                                        const float expected =
                                            f16RkReference(combine, transform, values, outer, rows, k, c) *
                                            output_scale;
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
}

// MEAN finalizes once (divide by R), and a BF16 output converts the finalized, scaled value.
TEST(CubReductionF16RK, OneWindowFinalizeAndNarrowOutput) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    const uint64_t outer_size = 3, rows = 50, k = 20;
    std::vector<float> values(outer_size * rows * k);
    for (uint64_t i = 0; i < values.size(); ++i) values[i] = static_cast<float>(static_cast<int>(i % 13) - 6) * 0.25f;
    const auto geometry = CubReduction::analyzeValueGeometry(CubReductionOp::Mean, {outer_size, rows, k}, {1});
    const auto semantics = makeValueReductionStageSemantics(CubReductionOp::Mean, DensePassRole::Complete, rows);
    for (const DataType dtype : {DataType::FP16, DataType::BF16}) {
        const Tensor input = makeGpuTensor(values, {outer_size, rows, k}, stream, dtype);
        for (const auto& unit : f16OneWindowRuntimeKUnits(k)) {
            SCOPED_TRACE(::testing::Message() << formatName(dtype) << " O" << unit.outputs_per_warp << " bucket "
                                              << unit.max_components_per_lane);
            Tensor output(gpuPlacement, TensorDescriptor(DataType::BF16, geometry.output_dimensions));
            launchF16OneWindowRuntimeKComplete(input, output, geometry, unit, 1, stream, semantics, 2.0f);
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
}

// The launcher rejects other input dtypes, K outside the unit's bucket, a misaligned base, and the wrong output size.
TEST(CubReductionF16RK, OneWindowRejectsUnsupportedRequests) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    const uint64_t outer_size = 4, rows = 8, k = 16;
    const auto geometry = CubReduction::analyzeValueGeometry(CubReductionOp::Sum, {outer_size, rows, k}, {1});
    const Tensor input = makeGpuTensor(std::vector<float>(outer_size * rows * k, 1.0f), {outer_size, rows, k}, stream,
                                       DataType::FP16);
    // The same elements 2 bytes past a 16-byte boundary.
    const Tensor bytes = makeGpuUnsignedTensor(std::vector<uint64_t>(2 * outer_size * rows * k + 16, 0),
                                               {2 * outer_size * rows * k + 16}, stream, DataType::UINT8);
    const Tensor misaligned = bytes.aliasByteStorageAs(DataType::BF16, {outer_size, rows, k}, 2);
    const auto units = f16OneWindowRuntimeKUnits(k);
    ASSERT_FALSE(units.empty());
    const auto unit = units.front();
    Tensor output(gpuPlacement, TensorDescriptor(DataType::FP32, geometry.output_dimensions));
    const CubReductionStageSemantics semantics;
    for (const DataType other : {DataType::FP8_E4M3, DataType::FP32}) {
        const Tensor wrong = makeGpuTensor(std::vector<float>(outer_size * rows * k, 1.0f), {outer_size, rows, k},
                                           stream, other);
        EXPECT_THROW(launchF16OneWindowRuntimeKComplete(wrong, output, geometry, unit, 1, stream, semantics),
                     std::logic_error);
    }
    // The bucket-16 unit (FP8 only) and a 2-byte unit at a K outside its bucket.
    EXPECT_THROW(launchF16OneWindowRuntimeKComplete(input, output, geometry, {1, 16, 16, true}, 1, stream, semantics),
                 std::logic_error);
    EXPECT_THROW(static_cast<void>(queryF16OneWindowRuntimeKOccupancy(DataType::FP16, 250, {1, 16, 8, true}, 1)),
                 std::logic_error);
    EXPECT_THROW(launchF16OneWindowRuntimeKComplete(misaligned, output, geometry, unit, 1, stream, semantics),
                 std::logic_error);
    // O2 with W2 (O2/O4/O8 are warp-local).
    const Fp8OneWindowRuntimeKUnit o2{2, 16, 1, false};
    if (std::find(units.begin(), units.end(), o2) != units.end()) {
        EXPECT_THROW(launchF16OneWindowRuntimeKComplete(input, output, geometry, o2, 2, stream, semantics),
                     std::logic_error);
    }
    Tensor short_output(gpuPlacement, TensorDescriptor(DataType::FP32, {outer_size * k - 1}));
    EXPECT_THROW(launchF16OneWindowRuntimeKComplete(input, short_output, geometry, unit, 1, stream, semantics),
                 std::logic_error);
    EXPECT_NO_THROW(launchF16OneWindowRuntimeKComplete(input, output, geometry, unit, 1, stream, semantics));
    stream.synchronize();
}

// 1242: DeepRSmallK on FP16 / BF16 input. Every ownership (TinyK at K <= 4 and <= 8, one / two / four / five / eight
// components per lane) at K values across it, the five depths for the one-component kernels (compiled depths) and two
// depths elsewhere, R past several shards with a partial last shard, every combine and transform, both formats: each
// FP32 partial [outer, shard, K] against a CPU reduction of its rows, exactly.
TEST(CubReductionF16RK, DeepRSmallKPartialsMatchReference) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    const std::array<float, 8> pattern = {-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f, 2.0f};
    const std::vector<uint64_t> ks{1, 2, 3, 4, 5, 7, 8, 9, 16, 31, 32, 33, 63, 64, 65, 100, 128, 129, 159, 160, 161,
                                   200, 248, 249};
    const auto& depths = Fp8RK::Fp8DeepRSmallKPolicy::compiled_rows;
    EXPECT_FALSE(f16DeepRSmallKSupports(3, 1000, 250, 64));
    EXPECT_FALSE(f16DeepRSmallKSupports(3, 64, 16, 64));    // R must exceed the depth
    EXPECT_FALSE(f16DeepRSmallKSupports(3, 1000, 16, 100));  // not a retained depth
    EXPECT_TRUE(f16DeepRSmallKSupports(3, 1000, 249, 64));
    const uint64_t outer_size = 3;
    for (const DataType dtype : {DataType::FP16, DataType::BF16}) {
        for (uint64_t k : ks) {
            std::vector<uint64_t> k_depths{depths.front(), 256};
            if (k >= 9 && k <= 32) k_depths.assign(depths.begin(), depths.end());
            for (uint64_t depth : k_depths) {
                const uint64_t rows = 2 * depth + 37;
                const uint64_t shards = (rows + depth - 1) / depth;
                SCOPED_TRACE(::testing::Message() << formatName(dtype) << " K=" << k << " depth=" << depth
                                                  << " R=" << rows);
                ASSERT_TRUE(f16DeepRSmallKSupports(outer_size, rows, k, depth));
                std::vector<float> additive(outer_size * rows * k);
                std::vector<float> multiplicative(additive.size(), 1.0f);
                for (uint64_t i = 0; i < additive.size(); ++i) {
                    additive[i] = pattern[(i * 5 + i / k) % pattern.size()];
                    if (i % 97 == 3) multiplicative[i] = (i % 3 == 0) ? -1.0f : ((i % 3 == 1) ? 2.0f : 0.5f);
                }
                const auto geometry =
                    CubReduction::analyzeValueGeometry(CubReductionOp::Sum, {outer_size, rows, k}, {1});
                const Tensor additive_input = makeGpuTensor(additive, {outer_size, rows, k}, stream, dtype);
                const Tensor product_input = makeGpuTensor(multiplicative, {outer_size, rows, k}, stream, dtype);
                for (CubReductionStageCombine combine :
                     {CubReductionStageCombine::Sum, CubReductionStageCombine::Product,
                      CubReductionStageCombine::Minimum, CubReductionStageCombine::Maximum}) {
                    for (CubReductionStageInputTransform transform :
                         {CubReductionStageInputTransform::Identity, CubReductionStageInputTransform::AbsoluteValue,
                          CubReductionStageInputTransform::Square}) {
                        SCOPED_TRACE(::testing::Message() << "combine " << static_cast<int>(combine)
                                                          << " transform " << static_cast<int>(transform));
                        const bool product = combine == CubReductionStageCombine::Product;
                        const Tensor& input = product ? product_input : additive_input;
                        const std::vector<float>& values = product ? multiplicative : additive;
                        CubReductionStageSemantics semantics;
                        semantics.combine = combine;
                        semantics.input_transform = transform;
                        semantics.finalize = CubReductionStageFinalize::Identity;
                        ASSERT_GT(queryF16DeepRSmallKRetainedOccupancy(dtype, depth, k, semantics)
                                      .max_active_blocks_per_sm,
                                  0);
                        Tensor partials(gpuPlacement, TensorDescriptor(DataType::FP32, {outer_size, shards, k}));
                        launchF16DeepRSmallKRetainedFirstStage(semantics, input, partials, geometry, depth, shards,
                                                               stream);
                        stream.synchronize();
                        const auto actual = copyGpuTensorAsFloat(partials, stream);
                        for (uint64_t outer = 0; outer < outer_size; ++outer) {
                            for (uint64_t shard = 0; shard < shards; ++shard) {
                                const uint64_t first = shard * depth;
                                const uint64_t count = std::min(depth, rows - first);
                                std::vector<float> block(count * k);
                                for (uint64_t r = 0; r < count; ++r)
                                    for (uint64_t c = 0; c < k; ++c)
                                        block[r * k + c] = values[(outer * rows + first + r) * k + c];
                                for (uint64_t c = 0; c < k; ++c) {
                                    const float expected = f16RkReference(combine, transform, block, 0, count, k, c);
                                    ASSERT_EQ(expected, actual[(outer * shards + shard) * k + c])
                                        << "outer=" << outer << " shard=" << shard << " k=" << c;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // Rejections: FP32 and FP8 input, K = 250, a wrong partials shape, a misaligned base.
    const auto geometry = CubReduction::analyzeValueGeometry(CubReductionOp::Sum, {2, 200, 16}, {1});
    const Tensor input = makeGpuTensor(std::vector<float>(2 * 200 * 16, 1.0f), {2, 200, 16}, stream, DataType::BF16);
    Tensor partials(gpuPlacement, TensorDescriptor(DataType::FP32, {2, 4, 16}));
    Tensor wrong(gpuPlacement, TensorDescriptor(DataType::FP32, {2, 3, 16}));
    const CubReductionStageSemantics semantics;
    for (const DataType other : {DataType::FP32, DataType::FP8_E4M3}) {
        const Tensor bad = makeGpuTensor(std::vector<float>(2 * 200 * 16, 1.0f), {2, 200, 16}, stream, other);
        EXPECT_THROW(launchF16DeepRSmallKRetainedFirstStage(semantics, bad, partials, geometry, 64, 4, stream),
                     std::logic_error);
    }
    EXPECT_THROW(launchF16DeepRSmallKRetainedFirstStage(semantics, input, wrong, geometry, 64, 4, stream),
                 std::logic_error);
    const Tensor bytes = makeGpuUnsignedTensor(std::vector<uint64_t>(2 * 2 * 200 * 16 + 16, 0), {2 * 2 * 200 * 16 + 16},
                                               stream, DataType::UINT8);
    const Tensor misaligned = bytes.aliasByteStorageAs(DataType::FP16, {2, 200, 16}, 2);
    EXPECT_THROW(launchF16DeepRSmallKRetainedFirstStage(semantics, misaligned, partials, geometry, 64, 4, stream),
                 std::logic_error);
    EXPECT_THROW(static_cast<void>(queryF16DeepRSmallKRetainedOccupancy(DataType::FP16, 64, 250, semantics)),
                 std::logic_error);
    EXPECT_THROW(static_cast<void>(queryF16DeepRSmallKRetainedOccupancy(DataType::FP32, 64, 16, semantics)),
                 std::logic_error);
    EXPECT_NO_THROW(launchF16DeepRSmallKRetainedFirstStage(semantics, input, partials, geometry, 64, 4, stream));
    stream.synchronize();
}

namespace {

// The expected AlignedK [outer, shards, K] at the balanced row split (the first R % shards shards own one extra row);
// Complete (shards = 1) is scaled.
std::vector<float> f16AlignedKExpected(CubReductionInternal::CubReductionStageCombine combine,
                                       CubReductionInternal::CubReductionStageInputTransform transform,
                                       const std::vector<float>& values, uint64_t outer_size, uint64_t rows,
                                       uint64_t k, uint64_t shards, float output_scale) {
    std::vector<float> expected(outer_size * shards * k);
    const uint64_t base = rows / shards, extra = rows % shards;
    for (uint64_t outer = 0; outer < outer_size; ++outer) {
        for (uint64_t shard = 0; shard < shards; ++shard) {
            const uint64_t first = shard * base + std::min(shard, extra);
            const uint64_t count = base + (shard < extra ? 1 : 0);
            const auto row = [&](uint64_t r) {
                return values.begin() + static_cast<std::ptrdiff_t>((outer * rows + r) * k);
            };
            const std::vector<float> block(row(first), row(first + count));
            for (uint64_t c = 0; c < k; ++c) {
                const float v = f16RkReference(combine, transform, block, 0, count, k, c);
                expected[(outer * shards + shard) * k + c] = shards > 1 ? v : v * output_scale;
            }
        }
    }
    return expected;
}

// One AlignedK launch against `expected`, exactly: FP32 partials (Staged) or an FP32 output (Complete).
void expectF16AlignedK(const CubReductionInternal::CubReductionStageSemantics& semantics, const Tensor& input,
                       const CubReductionGeometry& geometry, uint32_t packet_bytes, uint32_t threads, uint64_t shards,
                       float output_scale, const std::vector<float>& expected, Stream& stream) {
    using namespace CubReductionInternal;
    const bool staged = shards > 1;
    const uint64_t outer_size = geometry.outer_size, k = geometry.inner_size;
    ASSERT_TRUE(f16AlignedKSupports(outer_size, geometry.reduction_size, k, packet_bytes, threads, shards));
    ASSERT_GT(queryF16AlignedKOccupancy(input.getDataType(), packet_bytes, threads, staged).max_active_blocks_per_sm,
              0);
    Tensor output(gpuPlacement, staged ? TensorDescriptor(DataType::FP32, {outer_size, shards, k})
                                       : TensorDescriptor(DataType::FP32, geometry.output_dimensions));
    launchF16AlignedKStage(semantics, input, output, geometry, packet_bytes, threads, shards, output_scale, stream);
    stream.synchronize();
    const auto actual = copyGpuTensorAsFloat(output, stream);
    ASSERT_EQ(expected.size(), actual.size());
    for (size_t i = 0; i < expected.size(); ++i) ASSERT_EQ(expected[i], actual[i]) << "index " << i;
}

}  // namespace

// 1243: AlignedK on FP16 / BF16 input. Every packet (2 / 4 / 8 / 16 bytes: 1 / 2 / 4 / 8 components per lane, K a
// multiple of that), CTA 32 / 64 / 128, Complete and Staged (FP8's shard targets of 32 and 128 rows), K values across
// one and several component tiles, every combine and transform, both formats: Complete outputs (negative scale) and
// Staged partials against a CPU reduction, exactly.
TEST(CubReductionF16RK, AlignedKMatchesReference) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    const std::array<float, 8> pattern = {-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f, 2.0f};
    const std::vector<uint64_t> ks{2, 3, 4, 6, 8, 62, 64, 115, 130, 256, 1000, 1001, 1032};
    const uint64_t outer_size = 3, rows = 300;
    EXPECT_FALSE(f16AlignedKSupports(3, 300, 12, 16, 32, 1));   // K % 8
    EXPECT_FALSE(f16AlignedKSupports(3, 300, 6, 8, 32, 1));     // K % 4
    EXPECT_FALSE(f16AlignedKSupports(3, 300, 7, 4, 32, 1));     // K % 2
    EXPECT_TRUE(f16AlignedKSupports(3, 300, 7, 2, 32, 1));      // the 2-byte packet takes any K
    EXPECT_FALSE(f16AlignedKSupports(3, 300, 8, 1, 32, 1));     // packet
    EXPECT_FALSE(f16AlignedKSupports(3, 300, 8, 12, 32, 1));    // packet
    EXPECT_FALSE(f16AlignedKSupports(3, 300, 32, 32, 32, 1));   // packet
    EXPECT_FALSE(f16AlignedKSupports(3, 300, 8, 16, 256, 1));   // CTA
    EXPECT_FALSE(f16AlignedKSupports(3, 300, 8, 16, 32, 301));  // more shards than rows
    EXPECT_TRUE(f16AlignedKSupports(3, 300, 8, 16, 32, 10));
    for (const DataType dtype : {DataType::FP16, DataType::BF16}) {
        for (uint64_t k : ks) {
            std::vector<float> additive(outer_size * rows * k);
            std::vector<float> multiplicative(additive.size(), 1.0f);
            for (uint64_t i = 0; i < additive.size(); ++i) {
                additive[i] = pattern[(i * 5 + i / k) % pattern.size()];
                if (i % 97 == 3) multiplicative[i] = (i % 3 == 0) ? -1.0f : ((i % 3 == 1) ? 2.0f : 0.5f);
            }
            const auto geometry = CubReduction::analyzeValueGeometry(CubReductionOp::Sum, {outer_size, rows, k}, {1});
            ASSERT_EQ(geometry.path, CubReductionPath::TiledFixedSegment);
            const Tensor additive_input = makeGpuTensor(additive, {outer_size, rows, k}, stream, dtype);
            const Tensor product_input = makeGpuTensor(multiplicative, {outer_size, rows, k}, stream, dtype);
            for (uint64_t shards : {uint64_t{1}, (rows + 31) / 32, (rows + 127) / 128}) {
                for (CubReductionStageCombine combine :
                     {CubReductionStageCombine::Sum, CubReductionStageCombine::Product,
                      CubReductionStageCombine::Minimum, CubReductionStageCombine::Maximum}) {
                    for (CubReductionStageInputTransform transform :
                         {CubReductionStageInputTransform::Identity, CubReductionStageInputTransform::AbsoluteValue,
                          CubReductionStageInputTransform::Square}) {
                        const bool product = combine == CubReductionStageCombine::Product;
                        if (product && transform != CubReductionStageInputTransform::Identity) continue;
                        CubReductionStageSemantics semantics;
                        semantics.combine = combine;
                        semantics.input_transform = transform;
                        constexpr float output_scale = -0.5f;
                        const auto expected =
                            f16AlignedKExpected(combine, transform, product ? multiplicative : additive, outer_size,
                                                rows, k, shards, output_scale);
                        for (uint32_t packet_bytes : {2u, 4u, 8u, 16u}) {
                            if (k % (packet_bytes / 2) != 0) continue;
                            for (uint32_t threads : {32u, 64u, 128u}) {
                                SCOPED_TRACE(::testing::Message()
                                             << formatName(dtype) << " K=" << k << " P" << packet_bytes << " CTA"
                                             << threads << " shards " << shards << " combine "
                                             << static_cast<int>(combine) << " transform "
                                             << static_cast<int>(transform));
                                expectF16AlignedK(semantics, product ? product_input : additive_input, geometry,
                                                  packet_bytes, threads, shards, output_scale, expected, stream);
                            }
                        }
                    }
                }
            }
        }
    }
}

// 1243: the short-R side of every packet: R below the 16-row batch of the 2- and 4-byte packets (their load-all tail),
// one batch, a batch plus a tail, and the 8- (8-row batch) and 16-byte (4-row) packets' tails; Complete and two shards;
// SUM / PRODUCT / MIN / MAX with their transforms; both formats; exactly.
TEST(CubReductionF16RK, AlignedKShortRowsMatchReference) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    const std::array<float, 8> pattern = {-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f, 2.0f};
    const uint64_t outer_size = 3;
    for (const DataType dtype : {DataType::FP16, DataType::BF16}) {
        for (uint64_t rows : {uint64_t{2}, uint64_t{3}, uint64_t{5}, uint64_t{8}, uint64_t{15}, uint64_t{16},
                              uint64_t{17}, uint64_t{31}, uint64_t{33}}) {
            for (uint64_t k : {uint64_t{3}, uint64_t{8}, uint64_t{114}, uint64_t{257}, uint64_t{1032}}) {
                std::vector<float> additive(outer_size * rows * k);
                std::vector<float> multiplicative(additive.size(), 1.0f);
                for (uint64_t i = 0; i < additive.size(); ++i) {
                    additive[i] = pattern[(i * 5 + i / k) % pattern.size()];
                    if (i % 7 == 3) multiplicative[i] = (i % 3 == 0) ? -1.0f : ((i % 3 == 1) ? 2.0f : 0.5f);
                }
                const auto geometry =
                    CubReduction::analyzeValueGeometry(CubReductionOp::Sum, {outer_size, rows, k}, {1});
                ASSERT_EQ(geometry.path, CubReductionPath::TiledFixedSegment);
                const Tensor additive_input = makeGpuTensor(additive, {outer_size, rows, k}, stream, dtype);
                const Tensor product_input = makeGpuTensor(multiplicative, {outer_size, rows, k}, stream, dtype);
                for (uint64_t shards : {uint64_t{1}, uint64_t{2}}) {
                    for (const auto& [combine, transform] :
                         {std::pair{CubReductionStageCombine::Sum, CubReductionStageInputTransform::Identity},
                          std::pair{CubReductionStageCombine::Sum, CubReductionStageInputTransform::Square},
                          std::pair{CubReductionStageCombine::Product, CubReductionStageInputTransform::Identity},
                          std::pair{CubReductionStageCombine::Minimum, CubReductionStageInputTransform::AbsoluteValue},
                          std::pair{CubReductionStageCombine::Maximum, CubReductionStageInputTransform::Identity}}) {
                        const bool product = combine == CubReductionStageCombine::Product;
                        CubReductionStageSemantics semantics;
                        semantics.combine = combine;
                        semantics.input_transform = transform;
                        constexpr float output_scale = -0.5f;
                        const auto expected =
                            f16AlignedKExpected(combine, transform, product ? multiplicative : additive, outer_size,
                                                rows, k, shards, output_scale);
                        for (uint32_t packet_bytes : {2u, 4u, 8u, 16u}) {
                            if (k % (packet_bytes / 2) != 0) continue;
                            for (uint32_t threads : {32u, 128u}) {
                                SCOPED_TRACE(::testing::Message()
                                             << formatName(dtype) << " R=" << rows << " K=" << k << " P"
                                             << packet_bytes << " CTA" << threads << " shards " << shards
                                             << " combine " << static_cast<int>(combine) << " transform "
                                             << static_cast<int>(transform));
                                expectF16AlignedK(semantics, product ? product_input : additive_input, geometry,
                                                  packet_bytes, threads, shards, output_scale, expected, stream);
                            }
                        }
                    }
                }
            }
        }
    }
}

// 1243: MEAN Complete finalizes (divides by R) once, before the scale, into an FP16 / BF16 output; the launcher rejects
// FP32 and FP8 input, PRODUCT with a transform, a base misaligned for the packet and a wrong partials shape.
TEST(CubReductionF16RK, AlignedKFinalizeAndRejections) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    const std::array<float, 8> pattern = {-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f, 2.0f};
    const uint64_t outer_size = 3, rows = 300;
    for (const DataType dtype : {DataType::FP16, DataType::BF16}) {
        const uint64_t k = 64;
        std::vector<float> values(outer_size * rows * k);
        for (uint64_t i = 0; i < values.size(); ++i) values[i] = pattern[(i * 3 + i / k) % pattern.size()];
        const auto geometry = CubReduction::analyzeValueGeometry(CubReductionOp::Mean, {outer_size, rows, k}, {1});
        const Tensor input = makeGpuTensor(values, {outer_size, rows, k}, stream, dtype);
        const auto semantics = makeValueReductionStageSemantics(CubReductionOp::Mean, DensePassRole::Complete, rows);
        for (uint32_t packet_bytes : {2u, 16u}) {
            SCOPED_TRACE(::testing::Message() << formatName(dtype) << " P" << packet_bytes);
            Tensor output(gpuPlacement, TensorDescriptor(dtype, geometry.output_dimensions));
            launchF16AlignedKStage(semantics, input, output, geometry, packet_bytes, 64, 1, 2.0f, stream);
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
    const uint64_t k = 8;
    const auto geometry = CubReduction::analyzeValueGeometry(CubReductionOp::Sum, {outer_size, rows, k}, {1});
    const Tensor input = makeGpuTensor(std::vector<float>(outer_size * rows * k, 1.0f), {outer_size, rows, k}, stream,
                                       DataType::FP16);
    const Tensor bytes = makeGpuUnsignedTensor(std::vector<uint64_t>(2 * outer_size * rows * k + 16, 0),
                                               {2 * outer_size * rows * k + 16}, stream, DataType::UINT8);
    const Tensor misaligned8 = bytes.aliasByteStorageAs(DataType::BF16, {outer_size, rows, k}, 8);
    const Tensor misaligned2 = bytes.aliasByteStorageAs(DataType::BF16, {outer_size, rows, k}, 2);
    Tensor complete(gpuPlacement, TensorDescriptor(DataType::FP32, geometry.output_dimensions));
    Tensor wrong(gpuPlacement, TensorDescriptor(DataType::FP32, {outer_size, 9, k}));
    CubReductionStageSemantics semantics;
    for (const DataType other : {DataType::FP32, DataType::FP8_E4M3}) {
        const Tensor bad = makeGpuTensor(std::vector<float>(outer_size * rows * k, 1.0f), {outer_size, rows, k},
                                         stream, other);
        EXPECT_THROW(launchF16AlignedKStage(semantics, bad, complete, geometry, 16, 32, 1, 1.0f, stream),
                     std::logic_error);
    }
    EXPECT_THROW(launchF16AlignedKStage(semantics, input, wrong, geometry, 16, 32, 10, 1.0f, stream),
                 std::logic_error);
    CubReductionStageSemantics product;
    product.combine = CubReductionStageCombine::Product;
    product.input_transform = CubReductionStageInputTransform::Square;
    EXPECT_THROW(launchF16AlignedKStage(product, input, complete, geometry, 16, 32, 1, 1.0f, stream),
                 std::logic_error);
    // Each packet runs on a base aligned to it: 8 bytes past a 16-byte boundary takes the 8-byte packet, 2 bytes
    // past it only the 2-byte packet.
    EXPECT_THROW(launchF16AlignedKStage(semantics, misaligned8, complete, geometry, 16, 32, 1, 1.0f, stream),
                 std::logic_error);
    EXPECT_NO_THROW(launchF16AlignedKStage(semantics, misaligned8, complete, geometry, 8, 32, 1, 1.0f, stream));
    EXPECT_THROW(launchF16AlignedKStage(semantics, misaligned2, complete, geometry, 4, 32, 1, 1.0f, stream),
                 std::logic_error);
    EXPECT_NO_THROW(launchF16AlignedKStage(semantics, misaligned2, complete, geometry, 2, 32, 1, 1.0f, stream));
    EXPECT_THROW(static_cast<void>(queryF16AlignedKOccupancy(DataType::FP32, 16, 32, false)), std::logic_error);
    EXPECT_THROW(static_cast<void>(queryF16AlignedKOccupancy(DataType::FP16, 32, 32, false)), std::logic_error);
    EXPECT_THROW(static_cast<void>(queryF16AlignedKOccupancy(DataType::BF16, 16, 256, true)), std::logic_error);
    stream.synchronize();
}

// 1244: RuntimeK PacketComplete on FP16 / BF16 input. W1/W2/W4/W8, a full tile (248), a narrow one (37) and a one-word
// tile (2) at K values across one and several tiles (K not a multiple of 8 gives every row prefix 0..7), every combine
// and transform, both formats, against a CPU reduction with a negative scale, exactly. Then MEAN into a narrow output,
// and the rejections.
TEST(CubReductionF16RK, PacketCompleteMatchesReference) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    const std::array<float, 8> pattern = {-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f, 2.0f};
    const std::vector<std::pair<uint64_t, uint64_t>> shapes{{5, 37}, {1, 301}};
    const std::vector<uint64_t> ks{2, 3, 7, 9, 33, 100, 248, 249, 250, 257, 333, 1001};
    EXPECT_FALSE(f16PacketCompleteSupports(3, 10, 64, 1, 249));  // tile cap
    EXPECT_FALSE(f16PacketCompleteSupports(3, 10, 64, 3, 64));   // W
    EXPECT_FALSE(f16PacketCompleteSupports(3, 10, 64, 1, 0));
    EXPECT_TRUE(f16PacketCompleteSupports(3, 10, 64, 8, 248));
    for (const DataType dtype : {DataType::FP16, DataType::BF16}) {
        for (const auto& [outer_size, rows] : shapes) {
            for (uint64_t k : ks) {
                std::vector<float> additive(outer_size * rows * k);
                std::vector<float> multiplicative(additive.size(), 1.0f);
                for (uint64_t i = 0; i < additive.size(); ++i) {
                    additive[i] = pattern[(i * 5 + i / k) % pattern.size()];
                    if (i % 7 == 3) multiplicative[i] = (i % 3 == 0) ? -1.0f : ((i % 3 == 1) ? 2.0f : 0.5f);
                }
                const auto geometry =
                    CubReduction::analyzeValueGeometry(CubReductionOp::Sum, {outer_size, rows, k}, {1});
                const Tensor additive_input = makeGpuTensor(additive, {outer_size, rows, k}, stream, dtype);
                const Tensor product_input = makeGpuTensor(multiplicative, {outer_size, rows, k}, stream, dtype);
                for (CubReductionStageCombine combine :
                     {CubReductionStageCombine::Sum, CubReductionStageCombine::Product,
                      CubReductionStageCombine::Minimum, CubReductionStageCombine::Maximum}) {
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
                                    f16RkReference(combine, transform, values, outer, rows, k, c) * output_scale;
                        ASSERT_GT(queryF16PacketCompleteOccupancy(dtype, transform, combine).max_active_blocks_per_sm,
                                  0);
                        for (uint32_t tile : {248u, 37u, 2u}) {
                            for (uint32_t w : {1u, 2u, 4u, 8u}) {
                                SCOPED_TRACE(::testing::Message()
                                             << formatName(dtype) << " outer=" << outer_size << " R=" << rows
                                             << " K=" << k << " tile " << tile << " W" << w << " combine "
                                             << static_cast<int>(combine) << " transform "
                                             << static_cast<int>(transform));
                                ASSERT_TRUE(f16PacketCompleteSupports(outer_size, rows, k, w, tile));
                                Tensor output(gpuPlacement,
                                              TensorDescriptor(DataType::FP32, geometry.output_dimensions));
                                launchF16PacketComplete(input, output, geometry, w, tile, stream, semantics,
                                                        output_scale);
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
    }
    // MEAN finalizes (divides by R) once, before the scale, into the input's format.
    for (const DataType dtype : {DataType::FP16, DataType::BF16}) {
        const uint64_t outer_size = 3, rows = 50, k = 77;
        std::vector<float> values(outer_size * rows * k);
        for (uint64_t i = 0; i < values.size(); ++i)
            values[i] = static_cast<float>(static_cast<int>(i % 13) - 6) * 0.25f;
        const auto geometry = CubReduction::analyzeValueGeometry(CubReductionOp::Mean, {outer_size, rows, k}, {1});
        const Tensor input = makeGpuTensor(values, {outer_size, rows, k}, stream, dtype);
        const auto semantics = makeValueReductionStageSemantics(CubReductionOp::Mean, DensePassRole::Complete, rows);
        Tensor output(gpuPlacement, TensorDescriptor(dtype, geometry.output_dimensions));
        launchF16PacketComplete(input, output, geometry, 2, 64, stream, semantics, 2.0f);
        stream.synchronize();
        const auto actual = copyGpuTensorAsFloat(output, stream);
        for (uint64_t outer = 0; outer < outer_size; ++outer) {
            for (uint64_t c = 0; c < k; ++c) {
                double sum = 0;
                for (uint64_t r = 0; r < rows; ++r) sum += values[(outer * rows + r) * k + c];
                const double expected = 2.0 * sum / static_cast<double>(rows);
                ASSERT_NEAR(expected, actual[outer * k + c], 1e-2 * std::fabs(expected) + 1e-3) << formatName(dtype);
            }
        }
    }
    // Rejections: FP32 and FP8 input, a tile past 248, W3, a misaligned base, the wrong output size.
    const uint64_t outer_size = 4, rows = 8, k = 16;
    const auto geometry = CubReduction::analyzeValueGeometry(CubReductionOp::Sum, {outer_size, rows, k}, {1});
    const Tensor input = makeGpuTensor(std::vector<float>(outer_size * rows * k, 1.0f), {outer_size, rows, k}, stream,
                                       DataType::BF16);
    const Tensor bytes = makeGpuUnsignedTensor(std::vector<uint64_t>(2 * outer_size * rows * k + 16, 0),
                                               {2 * outer_size * rows * k + 16}, stream, DataType::UINT8);
    const Tensor misaligned = bytes.aliasByteStorageAs(DataType::FP16, {outer_size, rows, k}, 2);
    Tensor output(gpuPlacement, TensorDescriptor(DataType::FP32, geometry.output_dimensions));
    Tensor short_output(gpuPlacement, TensorDescriptor(DataType::FP32, {outer_size * k - 1}));
    const CubReductionStageSemantics semantics;
    for (const DataType other : {DataType::FP32, DataType::FP8_E4M3}) {
        const Tensor bad = makeGpuTensor(std::vector<float>(outer_size * rows * k, 1.0f), {outer_size, rows, k},
                                         stream, other);
        EXPECT_THROW(launchF16PacketComplete(bad, output, geometry, 1, 16, stream, semantics), std::logic_error);
    }
    EXPECT_THROW(launchF16PacketComplete(input, output, geometry, 1, 249, stream, semantics), std::logic_error);
    EXPECT_THROW(launchF16PacketComplete(input, output, geometry, 3, 16, stream, semantics), std::logic_error);
    EXPECT_THROW(launchF16PacketComplete(misaligned, output, geometry, 1, 16, stream, semantics), std::logic_error);
    EXPECT_THROW(launchF16PacketComplete(input, short_output, geometry, 1, 16, stream, semantics), std::logic_error);
    EXPECT_THROW(static_cast<void>(queryF16PacketCompleteOccupancy(DataType::FP32)), std::logic_error);
    EXPECT_NO_THROW(launchF16PacketComplete(input, output, geometry, 1, 16, stream, semantics));
    stream.synchronize();
}

// 1245: the RuntimeK rotated first stage (PrefixReserved) on FP16 / BF16 input. Every retained depth at K values across
// one and several 249-component tiles (two depths at the larger K), R past two shards with a partial last shard, every
// combine and transform, both formats: each FP32 partial [outer, shard, K] against a CPU reduction of its rows,
// exactly. A base 2, 6 and 14 bytes past a 16-byte boundary gives the same partials. Then the rejections.
TEST(CubReductionF16RK, RotatedPartialsMatchReference) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    const std::array<float, 8> pattern = {-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f, 2.0f};
    const std::vector<uint64_t> ks{2, 3, 7, 64, 249, 250, 498, 499, 1001};
    const std::vector<uint64_t> depths{32, 64, 128, 256, 512, 1024};
    const uint64_t outer_size = 3;
    EXPECT_FALSE(f16RuntimeKRotatedSupports(3, 1000, 64, 100));  // not a retained depth
    EXPECT_FALSE(f16RuntimeKRotatedSupports(3, 64, 64, 64));     // R must exceed the depth
    EXPECT_TRUE(f16RuntimeKRotatedSupports(3, 65, 64, 64));
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
                    expected[(outer * shards + shard) * k + c] =
                        f16RkReference(combine, transform, block, 0, count, k, c);
            }
        }
        return expected;
    };
    for (const DataType dtype : {DataType::FP16, DataType::BF16}) {
        for (uint64_t k : ks) {
            const std::vector<uint64_t> k_depths = k <= 250 ? depths : std::vector<uint64_t>{32, 256};
            for (uint64_t depth : k_depths) {
                const uint64_t rows = 2 * depth + 37;
                const uint64_t shards = (rows + depth - 1) / depth;
                ASSERT_TRUE(f16RuntimeKRotatedSupports(outer_size, rows, k, depth));
                std::vector<float> additive(outer_size * rows * k);
                std::vector<float> multiplicative(additive.size(), 1.0f);
                for (uint64_t i = 0; i < additive.size(); ++i) {
                    additive[i] = pattern[(i * 5 + i / k) % pattern.size()];
                    if (i % 97 == 3) multiplicative[i] = (i % 3 == 0) ? -1.0f : ((i % 3 == 1) ? 2.0f : 0.5f);
                }
                const auto geometry =
                    CubReduction::analyzeValueGeometry(CubReductionOp::Sum, {outer_size, rows, k}, {1});
                const Tensor additive_input = makeGpuTensor(additive, {outer_size, rows, k}, stream, dtype);
                const Tensor product_input = makeGpuTensor(multiplicative, {outer_size, rows, k}, stream, dtype);
                for (CubReductionStageCombine combine :
                     {CubReductionStageCombine::Sum, CubReductionStageCombine::Product,
                      CubReductionStageCombine::Minimum, CubReductionStageCombine::Maximum}) {
                    for (CubReductionStageInputTransform transform :
                         {CubReductionStageInputTransform::Identity, CubReductionStageInputTransform::AbsoluteValue,
                          CubReductionStageInputTransform::Square}) {
                        SCOPED_TRACE(::testing::Message() << formatName(dtype) << " K=" << k << " depth=" << depth
                                                          << " R=" << rows << " combine " << static_cast<int>(combine)
                                                          << " transform " << static_cast<int>(transform));
                        const bool product = combine == CubReductionStageCombine::Product;
                        const Tensor& input = product ? product_input : additive_input;
                        const std::vector<float>& values = product ? multiplicative : additive;
                        ASSERT_GT(queryF16RuntimeKRotatedOccupancy(dtype, depth, transform, combine)
                                      .max_active_blocks_per_sm,
                                  0);
                        Tensor partials(gpuPlacement, TensorDescriptor(DataType::FP32, {outer_size, shards, k}));
                        launchF16RuntimeKRotatedFirstStage(input, partials, geometry, depth, shards, stream, transform,
                                                           combine);
                        stream.synchronize();
                        const auto actual = copyGpuTensorAsFloat(partials, stream);
                        const auto expected = expected_partials(values, rows, k, depth, combine, transform);
                        ASSERT_EQ(expected.size(), actual.size());
                        for (size_t i = 0; i < expected.size(); ++i)
                            ASSERT_EQ(expected[i], actual[i]) << "index " << i;
                    }
                }
            }
        }
    }
    // A 2-byte aligned base: the same elements 2, 6 and 14 bytes past a 16-byte boundary (copied in as raw bits).
    for (const DataType dtype : {DataType::FP16, DataType::BF16}) {
        const uint64_t k = 499, depth = 64, rows = 2 * depth + 37, shards = 3;
        std::vector<float> values(outer_size * rows * k);
        for (uint64_t i = 0; i < values.size(); ++i) values[i] = pattern[(i * 5 + i / k) % pattern.size()];
        const auto geometry = CubReduction::analyzeValueGeometry(CubReductionOp::Sum, {outer_size, rows, k}, {1});
        const auto expected = expected_partials(values, rows, k, depth, CubReductionStageCombine::Sum,
                                                CubReductionStageInputTransform::Identity);
        const Tensor aligned = makeGpuTensor(values, {outer_size, rows, k}, stream, dtype);
        const std::vector<float> round_trip = copyGpuTensorAsFloat(aligned, stream);
        ASSERT_EQ(values, round_trip);  // the pattern is exact in both formats
        for (uint64_t offset : {uint64_t{2}, uint64_t{6}, uint64_t{14}}) {
            SCOPED_TRACE(::testing::Message() << formatName(dtype) << " offset " << offset);
            std::vector<uint64_t> bytes(2 * values.size() + 16, 0);
            for (size_t i = 0; i < values.size(); ++i) {
                uint32_t word = 0;
                std::memcpy(&word, &values[i], sizeof(word));
                uint16_t bits = static_cast<uint16_t>(word >> 16);  // BF16: the top half (exact for the pattern)
                if (dtype == DataType::FP16) {
                    // FP16 of zero or a normal float with <= 10 mantissa bits: rebias the exponent (15 - 127).
                    const uint32_t exponent = (word >> 23) & 0xffU;
                    const uint32_t magnitude =
                        exponent == 0 ? 0U : (((exponent - 112U) << 10) | ((word >> 13) & 0x3ffU));
                    bits = static_cast<uint16_t>(((word >> 16) & 0x8000U) | magnitude);
                }
                bytes[offset + 2 * i] = bits & 0xffU;
                bytes[offset + 2 * i + 1] = bits >> 8;
            }
            const Tensor storage = makeGpuUnsignedTensor(bytes, {bytes.size()}, stream, DataType::UINT8);
            const Tensor input = storage.aliasByteStorageAs(dtype, {outer_size, rows, k}, offset);
            Tensor partials(gpuPlacement, TensorDescriptor(DataType::FP32, {outer_size, shards, k}));
            launchF16RuntimeKRotatedFirstStage(input, partials, geometry, depth, shards, stream);
            stream.synchronize();
            const auto actual = copyGpuTensorAsFloat(partials, stream);
            ASSERT_EQ(expected.size(), actual.size());
            for (size_t i = 0; i < expected.size(); ++i) ASSERT_EQ(expected[i], actual[i]) << "index " << i;
        }
    }
    // Rejections: FP32 and FP8 input, a depth that is not retained, the wrong shard count, a wrong partials shape.
    const uint64_t k = 16, rows = 200;
    const auto geometry = CubReduction::analyzeValueGeometry(CubReductionOp::Sum, {2, rows, k}, {1});
    const Tensor input = makeGpuTensor(std::vector<float>(2 * rows * k, 1.0f), {2, rows, k}, stream, DataType::FP16);
    Tensor partials(gpuPlacement, TensorDescriptor(DataType::FP32, {2, 4, k}));
    Tensor wrong(gpuPlacement, TensorDescriptor(DataType::FP32, {2, 3, k}));
    for (const DataType other : {DataType::FP32, DataType::FP8_E4M3}) {
        const Tensor bad = makeGpuTensor(std::vector<float>(2 * rows * k, 1.0f), {2, rows, k}, stream, other);
        EXPECT_THROW(launchF16RuntimeKRotatedFirstStage(bad, partials, geometry, 64, 4, stream), std::logic_error);
    }
    EXPECT_THROW(launchF16RuntimeKRotatedFirstStage(input, partials, geometry, 50, 4, stream), std::logic_error);
    EXPECT_THROW(launchF16RuntimeKRotatedFirstStage(input, wrong, geometry, 64, 3, stream), std::logic_error);
    EXPECT_THROW(launchF16RuntimeKRotatedFirstStage(input, wrong, geometry, 64, 4, stream), std::logic_error);
    EXPECT_THROW(static_cast<void>(queryF16RuntimeKRotatedOccupancy(DataType::FP16, 100)), std::logic_error);
    EXPECT_THROW(static_cast<void>(queryF16RuntimeKRotatedOccupancy(DataType::FP32, 64)), std::logic_error);
    EXPECT_NO_THROW(launchF16RuntimeKRotatedFirstStage(input, partials, geometry, 64, 4, stream));
    stream.synchronize();
}

// 1246: the FP16 / BF16 RK candidate pool at sites across every family, with the production residency (the compiled
// kernels' occupancy queries): each entry runs through Fp8RK::launchPhysicalFirstStage, the adapters the selector's
// contest uses, and matches a CPU reduction exactly: a Complete entry's scaled output, a staged entry's FP32
// partials at its shard split (DeepRSmallK and rotated RuntimeK in depth-row blocks, AlignedK at the balanced split).
TEST(CubReductionF16RK, PoolEntriesRunThroughThePhysicalAdapters) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    using Fp8RK::PhysicalFamily;
    using Fp8RK::PhysicalKernel;
    using Fp8RK::RuntimeLayout;
    Stream stream(0);
    const std::array<float, 8> pattern = {-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f, 2.0f};
    const std::vector<std::array<uint64_t, 3>> sites{{3, 300, 7}, {2, 700, 64}, {5, 129, 250}, {1, 2000, 1001},
                                                     {4, 40, 16}, {2, 90, 256}};
    std::array<size_t, 4> family_runs{};
    for (const DataType dtype : {DataType::FP16, DataType::BF16}) {
        for (const auto& [outer_size, rows, k] : sites) {
            std::vector<float> values(outer_size * rows * k);
            for (uint64_t i = 0; i < values.size(); ++i) values[i] = pattern[(i * 5 + i / k) % pattern.size()];
            const auto geometry = CubReduction::analyzeValueGeometry(CubReductionOp::Sum, {outer_size, rows, k}, {1});
            const Tensor input = makeGpuTensor(values, {outer_size, rows, k}, stream, dtype);
            for (const auto& [combine, transform] :
                 {std::pair{CubReductionStageCombine::Sum, CubReductionStageInputTransform::Identity},
                  std::pair{CubReductionStageCombine::Maximum, CubReductionStageInputTransform::Square}}) {
                CubReductionStageSemantics semantics;
                semantics.combine = combine;
                semantics.input_transform = transform;
                F16RK::F16RKSite site;
                site.outer = outer_size;
                site.r = rows;
                site.k = k;
                site.sm_count = 170;
                site.input_dtype = dtype;
                site.semantics = semantics;
                const auto pool = F16RK::f16RKCandidatePool(site, [&](const PhysicalKernel& kernel) {
                    return F16RK::f16RKCompiledResidency(kernel, dtype, k, semantics);
                });
                ASSERT_FALSE(pool.empty());
                for (const auto& entry : pool) {
                    const PhysicalKernel& p = entry.kernel;
                    SCOPED_TRACE(::testing::Message()
                                 << formatName(dtype) << " outer=" << outer_size << " R=" << rows << " K=" << k
                                 << " combine " << static_cast<int>(combine) << " family "
                                 << static_cast<int>(p.family) << " a=" << p.a << " b=" << p.b << " c=" << p.c
                                 << " d=" << p.d << " e=" << p.e << " f=" << p.f);
                    ++family_runs[static_cast<size_t>(p.family)];
                    // The shard split: [first row, row count] of each shard.
                    std::vector<std::pair<uint64_t, uint64_t>> shards;
                    if (!p.staged()) {
                        shards.emplace_back(0, rows);
                    } else if (p.family == PhysicalFamily::AlignedK) {
                        const uint64_t base = rows / p.c, extra = rows % p.c;
                        for (uint64_t s = 0; s < p.c; ++s)
                            shards.emplace_back(s * base + std::min(s, extra), base + (s < extra ? 1 : 0));
                    } else {
                        const uint64_t depth = p.family == PhysicalFamily::DeepRSmallK ? p.a : p.e;
                        for (uint64_t first = 0; first < rows; first += depth)
                            shards.emplace_back(first, std::min(depth, rows - first));
                    }
                    constexpr float output_scale = -0.5f;
                    std::vector<float> expected(outer_size * shards.size() * k);
                    for (uint64_t outer = 0; outer < outer_size; ++outer) {
                        for (size_t s = 0; s < shards.size(); ++s) {
                            const auto row = [&](uint64_t r) {
                                return values.begin() + static_cast<std::ptrdiff_t>((outer * rows + r) * k);
                            };
                            const std::vector<float> block(row(shards[s].first),
                                                           row(shards[s].first + shards[s].second));
                            for (uint64_t c = 0; c < k; ++c) {
                                const float v =
                                    f16RkReference(combine, transform, block, 0, shards[s].second, k, c);
                                expected[(outer * shards.size() + s) * k + c] = p.staged() ? v : v * output_scale;
                            }
                        }
                    }
                    Tensor output(gpuPlacement,
                                  p.staged() ? TensorDescriptor(DataType::FP32, {outer_size, shards.size(), k})
                                             : TensorDescriptor(DataType::FP32, geometry.output_dimensions));
                    Fp8RK::launchPhysicalFirstStage(p, semantics, input, output, geometry, output_scale, stream);
                    stream.synchronize();
                    const auto actual = copyGpuTensorAsFloat(output, stream);
                    ASSERT_EQ(expected.size(), actual.size());
                    for (size_t i = 0; i < expected.size(); ++i) ASSERT_EQ(expected[i], actual[i]) << "index " << i;
                }
            }
        }
    }
    // Every family ran somewhere.
    for (size_t family = 0; family < family_runs.size(); ++family) EXPECT_GT(family_runs[family], 0u) << family;
}

namespace {

// Thor's ARG contract on [outer, R, K]: the first NaN wins, else the minimum (maximum) with the lowest row; -0 and +0
// tie. `x` holds the values as stored (the input's round trip).
uint64_t f16ArgReferenceRow(const std::vector<float>& x, uint64_t outer, uint64_t rows, uint64_t k,
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

// Ties, both zeros, infinities, values that only FP16 or only BF16 tells apart (1 + 2^-10, 1 + 2^-7), FP16's smallest
// subnormal and largest finite value, and sparse NaNs.
std::vector<float> f16ArgValues(uint64_t count, uint64_t k, bool nans) {
    const std::array<float, 16> pattern = {-2.0f, 1.0f, -0.0f, 0.0f, 2.0f, -2.0f, 3.5f, -1.0f, 2.0f,
                                           std::numeric_limits<float>::infinity(),
                                           -std::numeric_limits<float>::infinity(), 0.5f, 1.0009765625f,
                                           1.0078125f, 5.9604645e-08f, 65504.0f};
    std::vector<float> values(count);
    for (uint64_t i = 0; i < count; ++i) {
        values[i] = pattern[(i * 7 + i / k * 3) % pattern.size()];
        if (nans && i % 997 == 501) values[i] = std::numeric_limits<float>::quiet_NaN();
    }
    return values;
}

// Checks an ARG pass's value (bits; any NaN for a NaN winner) and index against the reference rows.
void expectF16ArgOutputs(const std::vector<float>& stored, uint64_t outer_size, uint64_t rows, uint64_t k,
                         bool maximum, const std::vector<float>& got_value, const std::vector<uint64_t>& got_index,
                         uint64_t stride) {
    for (uint64_t o = 0; o < outer_size; ++o) {
        for (uint64_t c = 0; c < k; ++c) {
            const uint64_t i = o * k + c;
            const uint64_t row = f16ArgReferenceRow(stored, o, rows, k, c, maximum);
            ASSERT_EQ(stride * row, got_index[i]) << "outer " << o << " k " << c;
            if (got_value.empty()) continue;
            const float want = stored[(o * rows + row) * k + c];
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

// 1248: the FP16 / BF16 AlignedK ARG twin. Every packet (4 / 8 / 16 bytes), CTA 32 / 64 / 128, one shard (Complete) and
// FP8's shard targets of 32 and 128 rows (Staged plus FP32's packed continuation, its shard-split kernel included at
// one output row with many shards), argmin and argmax, with and without NaNs, both formats: the value (FP32 and the
// input's format; the sign of zero included) and the index (UINT32 at a domain stride, or UINT64) against the
// contract, exactly. Then the rejections.
TEST(CubReductionF16RK, AlignedKArgMatchesReference) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    EXPECT_FALSE(f16AlignedKArgSupports(3, 300, 12, 16, 32, 1));  // K % 8
    EXPECT_FALSE(f16AlignedKArgSupports(3, 300, 7, 2, 32, 1));    // the 2-byte packet is VALUE only
    EXPECT_FALSE(f16AlignedKArgSupports(3, 300, 7, 4, 32, 1));    // K % 2
    EXPECT_TRUE(f16AlignedKArgSupports(3, 300, 6, 4, 32, 10));
    struct Site {
        uint64_t outer, rows, k;
    };
    const std::vector<Site> sites{{3, 300, 2}, {3, 300, 6}, {2, 300, 64}, {2, 150, 130}, {1, 77, 1032},
                                  {1, 5000, 8}, {4, 33, 16}};
    for (const DataType dtype : {DataType::FP16, DataType::BF16}) {
        for (const Site& site : sites) {
            for (bool nans : {false, true}) {
                const Tensor input = makeGpuTensor(f16ArgValues(site.outer * site.rows * site.k, site.k, nans),
                                                   {site.outer, site.rows, site.k}, stream, dtype);
                const std::vector<float> stored = copyGpuTensorAsFloat(input, stream);
                const auto geometry =
                    CubReduction::analyzeGeometry({site.outer, site.rows, site.k}, std::vector<uint32_t>{1});
                std::vector<uint64_t> shard_counts{1};
                for (uint64_t target : {32u, 128u})
                    if (site.rows > target) shard_counts.push_back((site.rows + target - 1) / target);
                for (bool maximum : {false, true}) {
                    for (uint64_t shards : shard_counts) {
                        for (uint32_t packet : {4u, 8u, 16u}) {
                            if (site.k % (packet / 2) != 0) continue;
                            for (uint32_t threads : {32u, 64u, 128u}) {
                                SCOPED_TRACE(::testing::Message()
                                             << formatName(dtype) << " outer=" << site.outer << " R=" << site.rows
                                             << " K=" << site.k << " P" << packet << " CTA" << threads << " shards "
                                             << shards << " max " << maximum << " nans " << nans);
                                ASSERT_TRUE(f16AlignedKArgSupports(site.outer, site.rows, site.k, packet, threads,
                                                                   shards));
                                ASSERT_GT(queryF16AlignedKArgOccupancy(dtype, packet, threads, shards > 1)
                                              .max_active_blocks_per_sm,
                                          0);
                                const uint64_t scratch_bytes = fp32ArgPackedScratchBytes(site.outer, shards, site.k);
                                std::optional<Tensor> scratch;
                                if (scratch_bytes != 0)
                                    scratch = Tensor(gpuPlacement, TensorDescriptor(DataType::UINT8, {scratch_bytes}));
                                uint64_t* packed =
                                    scratch ? static_cast<uint64_t*>(scratch->getMemPtr<void>()) : nullptr;
                                // An FP32 value and a UINT32 index at stride 3; then a value in the input's format
                                // and a UINT64 index.
                                Tensor value32(gpuPlacement, TensorDescriptor(DataType::FP32, {site.outer, site.k}));
                                Tensor index32(gpuPlacement, TensorDescriptor(DataType::UINT32, {site.outer, site.k}));
                                launchF16AlignedKArg(input, &value32, &index32, geometry, packet, threads, shards,
                                                     packed, maximum, 3, stream);
                                Tensor value16(gpuPlacement, TensorDescriptor(dtype, {site.outer, site.k}));
                                Tensor index64(gpuPlacement, TensorDescriptor(DataType::UINT64, {site.outer, site.k}));
                                launchF16AlignedKArg(input, &value16, &index64, geometry, packet, threads, shards,
                                                     packed, maximum, 1, stream);
                                stream.synchronize();
                                expectF16ArgOutputs(stored, site.outer, site.rows, site.k, maximum,
                                                    copyGpuTensorAsFloat(value32, stream),
                                                    copyGpuTensorAsUnsigned(index32, stream), 3);
                                expectF16ArgOutputs(stored, site.outer, site.rows, site.k, maximum,
                                                    copyGpuTensorAsFloat(value16, stream),
                                                    copyGpuTensorAsUnsigned(index64, stream), 1);
                            }
                        }
                    }
                }
            }
        }
    }
    // Rejections: FP32 and FP8 input, no outputs, a staged launch without scratch, a base misaligned for the packet.
    const uint64_t outer = 2, rows = 300, k = 8;
    const auto geometry = CubReduction::analyzeGeometry({outer, rows, k}, std::vector<uint32_t>{1});
    const Tensor input = makeGpuTensor(f16ArgValues(outer * rows * k, k, false), {outer, rows, k}, stream,
                                       DataType::BF16);
    Tensor index(gpuPlacement, TensorDescriptor(DataType::UINT32, {outer, k}));
    for (const DataType other : {DataType::FP32, DataType::FP8_E4M3}) {
        const Tensor bad = makeGpuTensor(std::vector<float>(outer * rows * k, 1.0f), {outer, rows, k}, stream, other);
        EXPECT_THROW(launchF16AlignedKArg(bad, nullptr, &index, geometry, 16, 32, 1, nullptr, false, 1, stream),
                     std::logic_error);
    }
    EXPECT_THROW(launchF16AlignedKArg(input, nullptr, nullptr, geometry, 16, 32, 1, nullptr, false, 1, stream),
                 std::logic_error);
    EXPECT_THROW(launchF16AlignedKArg(input, nullptr, &index, geometry, 16, 32, 10, nullptr, false, 1, stream),
                 std::logic_error);
    const Tensor bytes = makeGpuUnsignedTensor(std::vector<uint64_t>(2 * outer * rows * k + 16, 0),
                                               {2 * outer * rows * k + 16}, stream, DataType::UINT8);
    const Tensor misaligned = bytes.aliasByteStorageAs(DataType::FP16, {outer, rows, k}, 4);
    EXPECT_THROW(launchF16AlignedKArg(misaligned, nullptr, &index, geometry, 8, 32, 1, nullptr, false, 1, stream),
                 std::logic_error);
    EXPECT_NO_THROW(launchF16AlignedKArg(misaligned, nullptr, &index, geometry, 4, 32, 1, nullptr, false, 1, stream));
    EXPECT_THROW(static_cast<void>(queryF16AlignedKArgOccupancy(DataType::FP16, 2, 32, false)), std::logic_error);
    EXPECT_THROW(static_cast<void>(queryF16AlignedKArgOccupancy(DataType::FP32, 16, 32, false)), std::logic_error);
    stream.synchronize();
}

namespace {

// The 16-bit pattern of a value exactly representable in `dtype` (BF16: the top half of the FP32 word; FP16: rebuilt,
// subnormals included).
uint16_t f16Bits(float v, DataType dtype) {
    uint32_t word = 0;
    std::memcpy(&word, &v, sizeof(word));
    if (dtype == DataType::BF16) return static_cast<uint16_t>(word >> 16);
    const uint16_t sign = static_cast<uint16_t>((word >> 16) & 0x8000U);
    if (std::isnan(v)) return 0x7E00;
    if (std::isinf(v)) return static_cast<uint16_t>(sign | 0x7C00U);
    if (v == 0.0f) return sign;
    int exponent = 0;
    const double fraction = std::frexp(std::fabs(static_cast<double>(v)), &exponent);  // |v| = fraction * 2^exponent
    if (exponent - 1 < -14)  // subnormal: |v| = m * 2^-24
        return static_cast<uint16_t>(sign | static_cast<uint16_t>(std::ldexp(std::fabs(static_cast<double>(v)), 24)));
    const uint32_t mantissa = static_cast<uint32_t>(std::ldexp(fraction * 2.0 - 1.0, 10));
    return static_cast<uint16_t>(sign | ((exponent - 1 + 15) << 10) | mantissa);
}

}  // namespace

// 1249: the FP16 / BF16 rotated ARG twin. Odd and even K across one and several 249-component tiles, R from 1 to 5000,
// FP8's rotated depths and one Complete shard (depth >= R), argmin and argmax, with and without NaNs, both formats, on
// bases 0, 2, 6 and 14 bytes past a 16-byte boundary: value bits (FP32 and the input's format) and indices against the
// contract, exactly. Then the rejections.
TEST(CubReductionF16RK, RotatedArgMatchesReference) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    EXPECT_FALSE(f16RuntimeKArgSupports(3, 100, 5, 0));
    EXPECT_TRUE(f16RuntimeKArgSupports(3, 100, 5, 1000));
    struct Site {
        uint64_t outer, rows, k;
    };
    const std::vector<Site> sites{{3, 1, 5},     {2, 33, 249},  {2, 300, 250}, {1, 300, 499},
                                  {1, 5000, 3},  {2, 70, 1001}, {4, 40, 16}};
    for (const DataType dtype : {DataType::FP16, DataType::BF16}) {
        for (const Site& site : sites) {
            for (bool nans : {false, true}) {
                const uint64_t count = site.outer * site.rows * site.k;
                const Tensor aligned = makeGpuTensor(f16ArgValues(count, site.k, nans),
                                                     {site.outer, site.rows, site.k}, stream, dtype);
                const std::vector<float> stored = copyGpuTensorAsFloat(aligned, stream);
                const auto geometry =
                    CubReduction::analyzeGeometry({site.outer, site.rows, site.k}, std::vector<uint32_t>{1});
                for (uint64_t offset : {uint64_t{0}, uint64_t{2}, uint64_t{6}, uint64_t{14}}) {
                    std::vector<uint64_t> bytes(2 * count + 16, 0);
                    for (uint64_t i = 0; i < count; ++i) {
                        const uint16_t bits = f16Bits(stored[i], dtype);
                        bytes[offset + 2 * i] = bits & 0xffU;
                        bytes[offset + 2 * i + 1] = bits >> 8;
                    }
                    const Tensor storage = makeGpuUnsignedTensor(bytes, {bytes.size()}, stream, DataType::UINT8);
                    const Tensor input = storage.aliasByteStorageAs(dtype, {site.outer, site.rows, site.k}, offset);
                    for (bool maximum : {false, true}) {
                        for (uint64_t depth : {uint64_t{32}, uint64_t{256}, uint64_t{1024}, site.rows}) {
                            SCOPED_TRACE(::testing::Message()
                                         << formatName(dtype) << " outer=" << site.outer << " R=" << site.rows
                                         << " K=" << site.k << " offset " << offset << " depth " << depth << " max "
                                         << maximum << " nans " << nans);
                            ASSERT_TRUE(f16RuntimeKArgSupports(site.outer, site.rows, site.k, depth));
                            const uint64_t shards = (site.rows + depth - 1) / depth;
                            ASSERT_GT(queryF16RuntimeKArgOccupancy(dtype, shards > 1).max_active_blocks_per_sm, 0);
                            const uint64_t scratch_bytes = fp32ArgPackedScratchBytes(site.outer, shards, site.k);
                            std::optional<Tensor> scratch;
                            if (scratch_bytes != 0)
                                scratch = Tensor(gpuPlacement, TensorDescriptor(DataType::UINT8, {scratch_bytes}));
                            uint64_t* packed = scratch ? static_cast<uint64_t*>(scratch->getMemPtr<void>()) : nullptr;
                            Tensor value32(gpuPlacement, TensorDescriptor(DataType::FP32, {site.outer, site.k}));
                            Tensor index64(gpuPlacement, TensorDescriptor(DataType::UINT64, {site.outer, site.k}));
                            launchF16RuntimeKArg(input, &value32, &index64, geometry, depth, packed, maximum, 2,
                                                 stream);
                            Tensor value16(gpuPlacement, TensorDescriptor(dtype, {site.outer, site.k}));
                            Tensor index32(gpuPlacement, TensorDescriptor(DataType::UINT32, {site.outer, site.k}));
                            launchF16RuntimeKArg(input, &value16, &index32, geometry, depth, packed, maximum, 1,
                                                 stream);
                            stream.synchronize();
                            expectF16ArgOutputs(stored, site.outer, site.rows, site.k, maximum,
                                                copyGpuTensorAsFloat(value32, stream),
                                                copyGpuTensorAsUnsigned(index64, stream), 2);
                            expectF16ArgOutputs(stored, site.outer, site.rows, site.k, maximum,
                                                copyGpuTensorAsFloat(value16, stream),
                                                copyGpuTensorAsUnsigned(index32, stream), 1);
                        }
                    }
                }
            }
        }
    }
    // Rejections: FP32 and FP8 input, no outputs, a staged depth without scratch, depth 0.
    const uint64_t outer = 2, rows = 300, k = 7;
    const auto geometry = CubReduction::analyzeGeometry({outer, rows, k}, std::vector<uint32_t>{1});
    const Tensor input = makeGpuTensor(f16ArgValues(outer * rows * k, k, false), {outer, rows, k}, stream,
                                       DataType::FP16);
    Tensor index(gpuPlacement, TensorDescriptor(DataType::UINT32, {outer, k}));
    for (const DataType other : {DataType::FP32, DataType::FP8_E4M3}) {
        const Tensor bad = makeGpuTensor(std::vector<float>(outer * rows * k, 1.0f), {outer, rows, k}, stream, other);
        EXPECT_THROW(launchF16RuntimeKArg(bad, nullptr, &index, geometry, 1024, nullptr, false, 1, stream),
                     std::logic_error);
    }
    EXPECT_THROW(launchF16RuntimeKArg(input, nullptr, nullptr, geometry, 1024, nullptr, false, 1, stream),
                 std::logic_error);
    EXPECT_THROW(launchF16RuntimeKArg(input, nullptr, &index, geometry, 64, nullptr, false, 1, stream),
                 std::logic_error);
    EXPECT_THROW(launchF16RuntimeKArg(input, nullptr, &index, geometry, 0, nullptr, false, 1, stream),
                 std::logic_error);
    EXPECT_THROW(static_cast<void>(queryF16RuntimeKArgOccupancy(DataType::FP32, false)), std::logic_error);
    EXPECT_NO_THROW(launchF16RuntimeKArg(input, nullptr, &index, geometry, 1024, nullptr, false, 1, stream));
    stream.synchronize();
}

// 1249: the FP16 / BF16 PacketComplete ARG twin. W1/W2/W4/W8, tiles of 248, 37 and 8 components, K from 1 to 600 (every
// row prefix 0..7), R from 1 to 300, argmin and argmax, with and without NaNs, both formats: value bits (FP32 and the
// input's format) and indices (UINT32 at stride 5, or UINT64) against the contract, exactly. Then the rejections.
TEST(CubReductionF16RK, PacketCompleteArgMatchesReference) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    EXPECT_FALSE(f16PacketCompleteArgSupports(3, 10, 64, 1, 249));
    EXPECT_TRUE(f16PacketCompleteArgSupports(3, 10, 64, 8, 248));
    struct Site {
        uint64_t outer, rows, k;
    };
    const std::vector<Site> sites{{5, 1, 3}, {5, 9, 1},     {4, 40, 37}, {3, 2, 248},
                                  {2, 300, 249}, {700, 3, 5}, {1, 64, 600}};
    for (const DataType dtype : {DataType::FP16, DataType::BF16}) {
        ASSERT_GT(queryF16PacketCompleteArgOccupancy(dtype).max_active_blocks_per_sm, 0);
        for (const Site& site : sites) {
            for (bool nans : {false, true}) {
                const Tensor input = makeGpuTensor(f16ArgValues(site.outer * site.rows * site.k, site.k, nans),
                                                   {site.outer, site.rows, site.k}, stream, dtype);
                const std::vector<float> stored = copyGpuTensorAsFloat(input, stream);
                const auto geometry =
                    CubReduction::analyzeGeometry({site.outer, site.rows, site.k}, std::vector<uint32_t>{1});
                for (bool maximum : {false, true}) {
                    for (uint32_t tile : {248u, 37u, 8u}) {
                        for (uint32_t w : {1u, 2u, 4u, 8u}) {
                            SCOPED_TRACE(::testing::Message()
                                         << formatName(dtype) << " outer=" << site.outer << " R=" << site.rows
                                         << " K=" << site.k << " tile " << tile << " W" << w << " max " << maximum
                                         << " nans " << nans);
                            ASSERT_TRUE(f16PacketCompleteArgSupports(site.outer, site.rows, site.k, w, tile));
                            Tensor value32(gpuPlacement, TensorDescriptor(DataType::FP32, {site.outer, site.k}));
                            Tensor index32(gpuPlacement, TensorDescriptor(DataType::UINT32, {site.outer, site.k}));
                            launchF16PacketCompleteArg(input, &value32, &index32, geometry, w, tile, maximum, 5,
                                                       stream);
                            Tensor value16(gpuPlacement, TensorDescriptor(dtype, {site.outer, site.k}));
                            Tensor index64(gpuPlacement, TensorDescriptor(DataType::UINT64, {site.outer, site.k}));
                            launchF16PacketCompleteArg(input, &value16, &index64, geometry, w, tile, maximum, 1,
                                                       stream);
                            stream.synchronize();
                            expectF16ArgOutputs(stored, site.outer, site.rows, site.k, maximum,
                                                copyGpuTensorAsFloat(value32, stream),
                                                copyGpuTensorAsUnsigned(index32, stream), 5);
                            expectF16ArgOutputs(stored, site.outer, site.rows, site.k, maximum,
                                                copyGpuTensorAsFloat(value16, stream),
                                                copyGpuTensorAsUnsigned(index64, stream), 1);
                        }
                    }
                }
            }
        }
    }
    // Rejections: FP32 and FP8 input, no outputs, a tile past 248, W3, a misaligned base.
    const uint64_t outer = 2, rows = 30, k = 16;
    const auto geometry = CubReduction::analyzeGeometry({outer, rows, k}, std::vector<uint32_t>{1});
    const Tensor input = makeGpuTensor(f16ArgValues(outer * rows * k, k, false), {outer, rows, k}, stream,
                                       DataType::BF16);
    const Tensor bytes = makeGpuUnsignedTensor(std::vector<uint64_t>(2 * outer * rows * k + 16, 0),
                                               {2 * outer * rows * k + 16}, stream, DataType::UINT8);
    const Tensor misaligned = bytes.aliasByteStorageAs(DataType::BF16, {outer, rows, k}, 2);
    Tensor index(gpuPlacement, TensorDescriptor(DataType::UINT32, {outer, k}));
    for (const DataType other : {DataType::FP32, DataType::FP8_E4M3}) {
        const Tensor bad = makeGpuTensor(std::vector<float>(outer * rows * k, 1.0f), {outer, rows, k}, stream, other);
        EXPECT_THROW(launchF16PacketCompleteArg(bad, nullptr, &index, geometry, 1, 16, false, 1, stream),
                     std::logic_error);
    }
    EXPECT_THROW(launchF16PacketCompleteArg(input, nullptr, nullptr, geometry, 1, 16, false, 1, stream),
                 std::logic_error);
    EXPECT_THROW(launchF16PacketCompleteArg(input, nullptr, &index, geometry, 1, 249, false, 1, stream),
                 std::logic_error);
    EXPECT_THROW(launchF16PacketCompleteArg(input, nullptr, &index, geometry, 3, 16, false, 1, stream),
                 std::logic_error);
    EXPECT_THROW(launchF16PacketCompleteArg(misaligned, nullptr, &index, geometry, 1, 16, false, 1, stream),
                 std::logic_error);
    EXPECT_THROW(static_cast<void>(queryF16PacketCompleteArgOccupancy(DataType::FP32)), std::logic_error);
    EXPECT_NO_THROW(launchF16PacketCompleteArg(input, nullptr, &index, geometry, 1, 16, false, 1, stream));
    stream.synchronize();
}

// 1250: the FP16 / BF16 OneWindow ARG twin. Every 2-byte unit whose bucket holds K (O1 W1/W2/W4/W8, O2/O4/O8 W1;
// one-component buckets with power-of-two and other R-worker counts, the load-ahead units), argmin and argmax, with and
// without NaNs, both formats: value bits (FP32 and the input's format) and indices (UINT32 at stride 5, or UINT64)
// against the contract, exactly. Then the rejections.
TEST(CubReductionF16RK, OneWindowArgMatchesReference) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    struct Site {
        uint64_t outer, rows;
    };
    const std::vector<Site> sites{{5, 37}, {1, 301}, {700, 3}, {9, 1}};
    const std::vector<uint64_t> ks{2, 3, 5, 8, 9, 13, 16, 17, 25, 32, 33, 57, 64, 65, 121, 128, 129, 200, 249, 256};
    for (const DataType dtype : {DataType::FP16, DataType::BF16}) {
        for (const Site& site : sites) {
            for (uint64_t k : ks) {
                for (bool nans : {false, true}) {
                    const Tensor input = makeGpuTensor(f16ArgValues(site.outer * site.rows * k, k, nans),
                                                       {site.outer, site.rows, k}, stream, dtype);
                    const std::vector<float> stored = copyGpuTensorAsFloat(input, stream);
                    const auto geometry =
                        CubReduction::analyzeGeometry({site.outer, site.rows, k}, std::vector<uint32_t>{1});
                    const auto units = f16OneWindowRuntimeKUnits(k);
                    ASSERT_FALSE(units.empty()) << k;
                    for (const Fp8OneWindowRuntimeKUnit& unit : units) {
                        for (uint32_t w : {1u, 2u, 4u, 8u}) {
                            if (unit.outputs_per_warp > 1 && w != 1) continue;
                            ASSERT_TRUE(f16OneWindowArgSupports(site.outer, site.rows, k, unit, w));
                            ASSERT_GT(queryF16OneWindowArgOccupancy(dtype, k, unit, w).max_active_blocks_per_sm, 0);
                            for (bool maximum : {false, true}) {
                                SCOPED_TRACE(::testing::Message()
                                             << formatName(dtype) << " outer=" << site.outer << " R=" << site.rows
                                             << " K=" << k << " O" << unit.outputs_per_warp << " W" << w
                                             << " bucket " << unit.max_components_per_lane << " load_ahead "
                                             << unit.load_ahead << " max " << maximum << " nans " << nans);
                                Tensor value32(gpuPlacement, TensorDescriptor(DataType::FP32, {site.outer, k}));
                                Tensor index32(gpuPlacement, TensorDescriptor(DataType::UINT32, {site.outer, k}));
                                launchF16OneWindowArg(input, &value32, &index32, geometry, unit, w, maximum, 5,
                                                      stream);
                                Tensor value16(gpuPlacement, TensorDescriptor(dtype, {site.outer, k}));
                                Tensor index64(gpuPlacement, TensorDescriptor(DataType::UINT64, {site.outer, k}));
                                launchF16OneWindowArg(input, &value16, &index64, geometry, unit, w, maximum, 1,
                                                      stream);
                                stream.synchronize();
                                expectF16ArgOutputs(stored, site.outer, site.rows, k, maximum,
                                                    copyGpuTensorAsFloat(value32, stream),
                                                    copyGpuTensorAsUnsigned(index32, stream), 5);
                                expectF16ArgOutputs(stored, site.outer, site.rows, k, maximum,
                                                    copyGpuTensorAsFloat(value16, stream),
                                                    copyGpuTensorAsUnsigned(index64, stream), 1);
                            }
                        }
                    }
                }
            }
        }
    }
    // Rejections: FP32 and FP8 input, no outputs, K past the unit's bucket, a misaligned base.
    const uint64_t outer = 2, rows = 30, k = 16;
    const auto geometry = CubReduction::analyzeGeometry({outer, rows, k}, std::vector<uint32_t>{1});
    const Tensor input = makeGpuTensor(f16ArgValues(outer * rows * k, k, false), {outer, rows, k}, stream,
                                       DataType::FP16);
    const Tensor bytes = makeGpuUnsignedTensor(std::vector<uint64_t>(2 * outer * rows * k + 16, 0),
                                               {2 * outer * rows * k + 16}, stream, DataType::UINT8);
    const Tensor misaligned = bytes.aliasByteStorageAs(DataType::FP16, {outer, rows, k}, 2);
    Tensor index(gpuPlacement, TensorDescriptor(DataType::UINT32, {outer, k}));
    const auto units = f16OneWindowRuntimeKUnits(k);
    ASSERT_FALSE(units.empty());
    const auto unit = units.front();
    for (const DataType other : {DataType::FP32, DataType::FP8_E4M3}) {
        const Tensor bad = makeGpuTensor(std::vector<float>(outer * rows * k, 1.0f), {outer, rows, k}, stream, other);
        EXPECT_THROW(launchF16OneWindowArg(bad, nullptr, &index, geometry, unit, 1, false, 1, stream),
                     std::logic_error);
    }
    EXPECT_THROW(launchF16OneWindowArg(input, nullptr, nullptr, geometry, unit, 1, false, 1, stream),
                 std::logic_error);
    EXPECT_FALSE(f16OneWindowArgSupports(outer, rows, 300, unit, 1));
    EXPECT_THROW(launchF16OneWindowArg(misaligned, nullptr, &index, geometry, unit, 1, false, 1, stream),
                 std::logic_error);
    EXPECT_THROW(static_cast<void>(queryF16OneWindowArgOccupancy(DataType::FP32, k, unit, 1)), std::logic_error);
    EXPECT_NO_THROW(launchF16OneWindowArg(input, nullptr, &index, geometry, unit, 1, false, 1, stream));
    stream.synchronize();
}

// 1250: the FP16 / BF16 DeepRSmallK ARG twins. Every ownership (TinyK at K <= 4 and <= 8, one / two / four / eight
// components per lane, the shared lanes at K 9..16), FP8's five depths below R, argmin and argmax, with and without
// NaNs, both formats: value bits (FP32 and the input's format) and indices (UINT32 at stride 3, or UINT64) through the
// packed continuation, exactly. Then the rejections.
TEST(CubReductionF16RK, DeepRSmallKArgMatchesReference) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    EXPECT_FALSE(f16DeepRSmallKArgSupports(3, 64, 8, 64));    // R <= depth
    EXPECT_FALSE(f16DeepRSmallKArgSupports(3, 300, 250, 64));  // K > 249
    EXPECT_FALSE(f16DeepRSmallKArgSupports(3, 300, 8, 32));    // not a retained depth
    EXPECT_TRUE(f16DeepRSmallKArgSupports(3, 300, 249, 64));
    struct Site {
        uint64_t outer, rows, k;
    };
    const std::vector<Site> sites{{3, 300, 1},  {3, 300, 3},  {2, 700, 8},    {3, 200, 13}, {2, 300, 17},
                                  {2, 129, 32}, {2, 300, 47}, {2, 1500, 64},  {2, 300, 65}, {2, 257, 128},
                                  {2, 300, 129}, {2, 257, 249}, {1, 5000, 4}, {1, 3000, 37}};
    for (const DataType dtype : {DataType::FP16, DataType::BF16}) {
        for (uint64_t k : {3u, 8u, 13u, 32u, 64u, 128u, 129u, 249u}) {
            ASSERT_GT(queryF16DeepRSmallKArgOccupancy(dtype, k).max_active_blocks_per_sm, 0);
        }
        for (const Site& site : sites) {
            for (bool nans : {false, true}) {
                const Tensor input = makeGpuTensor(f16ArgValues(site.outer * site.rows * site.k, site.k, nans),
                                                   {site.outer, site.rows, site.k}, stream, dtype);
                const std::vector<float> stored = copyGpuTensorAsFloat(input, stream);
                const auto geometry =
                    CubReduction::analyzeGeometry({site.outer, site.rows, site.k}, std::vector<uint32_t>{1});
                for (uint64_t depth : {64u, 128u, 256u, 512u, 1024u}) {
                    if (site.rows <= depth) continue;
                    ASSERT_TRUE(f16DeepRSmallKArgSupports(site.outer, site.rows, site.k, depth));
                    const uint64_t shards = (site.rows + depth - 1) / depth;
                    Tensor scratch(gpuPlacement,
                                   TensorDescriptor(DataType::UINT8,
                                                    {fp32ArgPackedScratchBytes(site.outer, shards, site.k)}));
                    uint64_t* packed = static_cast<uint64_t*>(scratch.getMemPtr<void>());
                    for (bool maximum : {false, true}) {
                        SCOPED_TRACE(::testing::Message() << formatName(dtype) << " outer=" << site.outer << " R="
                                                          << site.rows << " K=" << site.k << " depth " << depth
                                                          << " max " << maximum << " nans " << nans);
                        Tensor value32(gpuPlacement, TensorDescriptor(DataType::FP32, {site.outer, site.k}));
                        Tensor index32(gpuPlacement, TensorDescriptor(DataType::UINT32, {site.outer, site.k}));
                        launchF16DeepRSmallKArg(input, &value32, &index32, geometry, depth, packed, maximum, 3,
                                                stream);
                        Tensor value16(gpuPlacement, TensorDescriptor(dtype, {site.outer, site.k}));
                        Tensor index64(gpuPlacement, TensorDescriptor(DataType::UINT64, {site.outer, site.k}));
                        launchF16DeepRSmallKArg(input, &value16, &index64, geometry, depth, packed, maximum, 1,
                                                stream);
                        stream.synchronize();
                        expectF16ArgOutputs(stored, site.outer, site.rows, site.k, maximum,
                                            copyGpuTensorAsFloat(value32, stream),
                                            copyGpuTensorAsUnsigned(index32, stream), 3);
                        expectF16ArgOutputs(stored, site.outer, site.rows, site.k, maximum,
                                            copyGpuTensorAsFloat(value16, stream),
                                            copyGpuTensorAsUnsigned(index64, stream), 1);
                    }
                }
            }
        }
    }
    // Rejections: FP32 and FP8 input, no outputs, no scratch, a depth that is not retained, a misaligned base.
    const uint64_t outer = 2, rows = 300, k = 16;
    const auto geometry = CubReduction::analyzeGeometry({outer, rows, k}, std::vector<uint32_t>{1});
    const Tensor input = makeGpuTensor(f16ArgValues(outer * rows * k, k, false), {outer, rows, k}, stream,
                                       DataType::BF16);
    const Tensor bytes = makeGpuUnsignedTensor(std::vector<uint64_t>(2 * outer * rows * k + 16, 0),
                                               {2 * outer * rows * k + 16}, stream, DataType::UINT8);
    const Tensor misaligned = bytes.aliasByteStorageAs(DataType::BF16, {outer, rows, k}, 2);
    Tensor scratch(gpuPlacement, TensorDescriptor(DataType::UINT8, {fp32ArgPackedScratchBytes(outer, 5, k)}));
    uint64_t* packed = static_cast<uint64_t*>(scratch.getMemPtr<void>());
    Tensor index(gpuPlacement, TensorDescriptor(DataType::UINT32, {outer, k}));
    for (const DataType other : {DataType::FP32, DataType::FP8_E4M3}) {
        const Tensor bad = makeGpuTensor(std::vector<float>(outer * rows * k, 1.0f), {outer, rows, k}, stream, other);
        EXPECT_THROW(launchF16DeepRSmallKArg(bad, nullptr, &index, geometry, 64, packed, false, 1, stream),
                     std::logic_error);
    }
    EXPECT_THROW(launchF16DeepRSmallKArg(input, nullptr, nullptr, geometry, 64, packed, false, 1, stream),
                 std::logic_error);
    EXPECT_THROW(launchF16DeepRSmallKArg(input, nullptr, &index, geometry, 64, nullptr, false, 1, stream),
                 std::logic_error);
    EXPECT_THROW(launchF16DeepRSmallKArg(input, nullptr, &index, geometry, 32, packed, false, 1, stream),
                 std::logic_error);
    EXPECT_THROW(launchF16DeepRSmallKArg(misaligned, nullptr, &index, geometry, 64, packed, false, 1, stream),
                 std::logic_error);
    EXPECT_THROW(static_cast<void>(queryF16DeepRSmallKArgOccupancy(DataType::FP16, 250)), std::logic_error);
    EXPECT_THROW(static_cast<void>(queryF16DeepRSmallKArgOccupancy(DataType::FP32, 16)), std::logic_error);
    EXPECT_NO_THROW(launchF16DeepRSmallKArg(input, nullptr, &index, geometry, 64, packed, false, 1, stream));
    stream.synchronize();
}

// 1251: every FP16 / BF16 RK ARG candidate (F16RK::f16RKArgCandidates over the production MIN / MAX pool: the twins
// and the ARG-only rotated passes) through launchF16RKArgPass, argmin and argmax, with NaNs, both formats: an FP32
// value with a UINT32 index at stride 3, and the index alone (UINT64), exactly against the contract. Every family ran.
TEST(CubReductionF16RK, ArgCandidatesRunThroughThePassLauncher) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    using Fp8RK::PhysicalKernel;
    Stream stream(0);
    const std::vector<std::array<uint64_t, 3>> sites{{3, 300, 7},  {2, 700, 64},  {5, 129, 250}, {1, 2000, 1001},
                                                     {4, 40, 16},  {2, 90, 256},  {1, 3000, 8},  {2, 1500, 1024}};
    std::array<size_t, 4> family_runs{};
    for (const DataType dtype : {DataType::FP16, DataType::BF16}) {
        for (const auto& [outer_size, rows, k] : sites) {
            const Tensor input =
                makeGpuTensor(f16ArgValues(outer_size * rows * k, k, true), {outer_size, rows, k}, stream, dtype);
            const std::vector<float> stored = copyGpuTensorAsFloat(input, stream);
            const auto geometry = CubReduction::analyzeGeometry({outer_size, rows, k}, std::vector<uint32_t>{1});
            for (bool maximum : {false, true}) {
                CubReductionStageSemantics semantics;
                semantics.combine = maximum ? CubReductionStageCombine::Maximum : CubReductionStageCombine::Minimum;
                F16RK::F16RKSite site;
                site.outer = outer_size;
                site.r = rows;
                site.k = k;
                site.sm_count = 170;
                site.input_dtype = dtype;
                site.semantics = semantics;
                const auto pool = F16RK::f16RKCandidatePool(site, [&](const PhysicalKernel& kernel) {
                    return F16RK::f16RKCompiledResidency(kernel, dtype, k, semantics);
                });
                const auto candidates =
                    F16RK::f16RKArgCandidates(pool, outer_size, rows, k, std::numeric_limits<uint64_t>::max());
                ASSERT_FALSE(candidates.empty());
                for (const auto& candidate : candidates) {
                    const PhysicalKernel& p = candidate.identity;
                    SCOPED_TRACE(::testing::Message()
                                 << formatName(dtype) << " outer=" << outer_size << " R=" << rows << " K=" << k
                                 << " max " << maximum << " family " << static_cast<int>(p.family) << " a=" << p.a
                                 << " b=" << p.b << " c=" << p.c << " d=" << p.d << " e=" << p.e << " f=" << p.f);
                    ++family_runs[static_cast<size_t>(p.family)];
                    std::optional<Tensor> scratch;
                    if (candidate.scratch_bytes != 0)
                        scratch = Tensor(gpuPlacement, TensorDescriptor(DataType::UINT8, {candidate.scratch_bytes}));
                    uint64_t* packed = scratch ? static_cast<uint64_t*>(scratch->getMemPtr<void>()) : nullptr;
                    Tensor value32(gpuPlacement, TensorDescriptor(DataType::FP32, {outer_size, k}));
                    Tensor index32(gpuPlacement, TensorDescriptor(DataType::UINT32, {outer_size, k}));
                    F16RK::launchF16RKArgPass(p, input, &value32, &index32, geometry, packed, maximum, 3, stream);
                    Tensor index64(gpuPlacement, TensorDescriptor(DataType::UINT64, {outer_size, k}));
                    F16RK::launchF16RKArgPass(p, input, nullptr, &index64, geometry, packed, maximum, 1, stream);
                    stream.synchronize();
                    expectF16ArgOutputs(stored, outer_size, rows, k, maximum, copyGpuTensorAsFloat(value32, stream),
                                        copyGpuTensorAsUnsigned(index32, stream), 3);
                    expectF16ArgOutputs(stored, outer_size, rows, k, maximum, {},
                                        copyGpuTensorAsUnsigned(index64, stream), 1);
                }
            }
        }
    }
    for (size_t family = 0; family < family_runs.size(); ++family) EXPECT_GT(family_runs[family], 0u) << family;
    // A staged pass without its scratch, and an identity of another width, are rejected.
    const uint64_t outer = 2, rows = 300, k = 16;
    const auto geometry = CubReduction::analyzeGeometry({outer, rows, k}, std::vector<uint32_t>{1});
    const Tensor input = makeGpuTensor(f16ArgValues(outer * rows * k, k, false), {outer, rows, k}, stream,
                                       DataType::FP16);
    Tensor index(gpuPlacement, TensorDescriptor(DataType::UINT32, {outer, k}));
    PhysicalKernel deep{};
    deep.family = Fp8RK::PhysicalFamily::DeepRSmallK;
    deep.element_bytes = 2;
    deep.a = 64;
    EXPECT_THROW(F16RK::launchF16RKArgPass(deep, input, nullptr, &index, geometry, nullptr, false, 1, stream),
                 std::logic_error);
    deep.element_bytes = 4;
    EXPECT_THROW(F16RK::launchF16RKArgPass(deep, input, nullptr, &index, geometry, nullptr, false, 1, stream),
                 std::logic_error);
}
