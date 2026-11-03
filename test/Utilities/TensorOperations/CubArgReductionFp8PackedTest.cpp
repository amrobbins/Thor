// ARG-A3g: packed staged partials. Each staged ARG twin (DeepRSmallK, rotated RuntimeK, AlignedK) writes one 4-byte
// shard-relative lane candidate per partial; launchFp8ArgPackedContinuation reduces them over shards. Checked against
// the FP32 ARG reference for FP32 value + UINT32 index, FP8 value only and UINT64 index only (stride 3).

#include "test/Utilities/TensorOperations/CubReductionTestSupport.h"
#include "test/Utilities/TensorOperations/Fp8ArgReference.h"
#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"
#include "Utilities/TensorOperations/Cub/DenseRetainedFp8ArgPhysicalPlan.h"
#include "Utilities/TensorOperations/Cub/Fp8ArgKey.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <random>
#include <string>
#include <vector>

using namespace ThorImplementation;
using namespace ThorImplementation::CubReductionTestSupport;
namespace Reference = ThorImplementation::Fp8ArgReference;

#if THOR_CUB_ENABLE_FP8_TYPES

namespace {

std::vector<float> packedInput(uint64_t count, bool e5m2, uint32_t seed) {
    static constexpr std::array<uint8_t, 10> SPECIALS{{0x00, 0x80, 0x7F, 0xFF, 0x7E, 0xFE, 0x7B, 0xFB, 0x01, 0x81}};
    static constexpr std::array<uint8_t, 4> TIES{{0x38, 0xB8, 0x00, 0x80}};
    std::mt19937 rng(seed);
    std::vector<float> values(count);
    const bool tie_heavy = (seed % 3u) == 0;
    for (float& value : values) {
        const uint8_t byte = tie_heavy           ? TIES[rng() % TIES.size()]
                             : (rng() % 64u) == 0 ? SPECIALS[rng() % SPECIALS.size()]
                                                  : static_cast<uint8_t>(rng());
        value = Reference::decode(byte, e5m2);
    }
    return values;
}

// first_stage(input, packed, geometry, maximum, stream) writes packed partials for `shards` shards; the continuation
// uses `uniform_rows` (0: AlignedK's balanced split).
void checkPacked(uint64_t outer,
                 uint64_t r,
                 uint64_t k,
                 uint64_t shards,
                 uint64_t uniform_rows,
                 const std::string& label,
                 const std::function<void(const Tensor&, uint32_t*, const CubReductionGeometry&, bool, Stream&)>&
                     first_stage,
                 uint32_t& seed) {
    using namespace CubReductionInternal;
    Stream stream(0);
    for (const DataType dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
        const bool e5m2 = dtype == DataType::FP8_E5M2;
        Tensor input = makeGpuTensor(packedInput(outer * r * k, e5m2, ++seed), {outer, r, k}, stream, dtype);
        const std::vector<float> stored = copyGpuTensorAsFloat(input, stream);
        const CubReductionGeometry geometry = CubArgReduction::analyzeDenseGeometry({outer, r, k}, {1});
        Tensor partials(gpuPlacement, TensorDescriptor(DataType::UINT32, {outer, shards, k}));
        auto* packed = static_cast<uint32_t*>(partials.getMemPtr<void>());
        for (const bool maximum : {false, true}) {
            first_stage(input, packed, geometry, maximum, stream);
            for (const int mode : {0, 1, 2}) {  // FP32 + UINT32, FP8 value only, UINT64 index only (stride 3)
                SCOPED_TRACE(::testing::Message() << label << " dtype=" << static_cast<int>(dtype) << " O=" << outer
                                                  << " R=" << r << " K=" << k << " shards=" << shards
                                                  << " max=" << maximum << " mode=" << mode);
                const uint64_t stride = mode == 2 ? 3 : 1;
                std::optional<Tensor> value;
                std::optional<Tensor> index;
                if (mode != 2)
                    value = Tensor(gpuPlacement, TensorDescriptor(mode == 1 ? dtype : DataType::FP32, {outer, k}));
                if (mode != 1)
                    index = Tensor(gpuPlacement,
                                   TensorDescriptor(mode == 2 ? DataType::UINT64 : DataType::UINT32, {outer, k}));
                launchFp8ArgPackedContinuation(packed, value ? &*value : nullptr, index ? &*index : nullptr, dtype,
                                               outer, r, k, shards, uniform_rows, maximum, stride, stream);
                stream.synchronize();
                const std::vector<float> got_values =
                    value ? copyGpuTensorAsFloat(*value, stream) : std::vector<float>{};
                const std::vector<uint64_t> got_indices =
                    index ? copyGpuTensorAsUnsigned(*index, stream) : std::vector<uint64_t>{};
                for (uint64_t o = 0; o < outer; ++o) {
                    for (uint64_t c = 0; c < k; ++c) {
                        Reference::Candidate best{0, stored[(o * r) * k + c]};
                        for (uint64_t row = 1; row < r; ++row)
                            best = Reference::combine(best, {row, stored[(o * r + row) * k + c]}, maximum);
                        if (index) {
                            ASSERT_EQ(best.index * stride, got_indices[o * k + c]) << "o=" << o << " c=" << c;
                        }
                        if (value) {
                            ASSERT_TRUE(Reference::sameValue(best.value, got_values[o * k + c]))
                                << "o=" << o << " c=" << c;
                        }
                    }
                }
            }
        }
    }
}

}  // namespace

TEST(CubArgReductionFp8Packed, DeepRSmallKPackedMatchesReference) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    uint32_t seed = 1000;
    for (const uint64_t k : {uint64_t{3}, uint64_t{8}, uint64_t{12}, uint64_t{17}, uint64_t{65}, uint64_t{200}}) {
        for (const uint64_t rows : {uint64_t{64}, uint64_t{256}}) {
            for (const uint64_t r : {rows + 1, 3 * rows + 5}) {
                const uint64_t shards = (r + rows - 1) / rows;
                checkPacked(3, r, k, shards, rows, "deep_r",
                            [&](const Tensor& in, uint32_t* packed, const CubReductionGeometry& g, bool maximum,
                                Stream& s) {
                                launchFp8DeepRSmallKArgFirstStagePacked(in, packed, g, rows, shards, maximum, s);
                            },
                            seed);
            }
        }
    }
}

TEST(CubArgReductionFp8Packed, RuntimeKPackedMatchesReference) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    uint32_t seed = 2000;
    for (const uint64_t k : {uint64_t{7}, uint64_t{250}, uint64_t{600}}) {
        for (const uint64_t rows : {uint64_t{32}, uint64_t{256}}) {
            for (const uint64_t r : {uint64_t{300}, uint64_t{1100}}) {
                const uint64_t shards = (r + rows - 1) / rows;
                checkPacked(2, r, k, shards, rows, "runtime_k",
                            [&](const Tensor& in, uint32_t* packed, const CubReductionGeometry& g, bool maximum,
                                Stream& s) {
                                launchFp8RuntimeKArgPass(in, nullptr, nullptr, g, rows, 8, maximum, 1, s, packed);
                            },
                            seed);
            }
        }
    }
}

TEST(CubArgReductionFp8Packed, AlignedKPackedMatchesReference) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    uint32_t seed = 3000;
    for (const uint64_t k : {uint64_t{16}, uint64_t{264}}) {
        for (const uint64_t r : {uint64_t{9}, uint64_t{300}}) {
            for (const uint32_t packet : {2u, 8u}) {
                for (const uint32_t threads : {32u, 128u}) {
                    for (const uint64_t shards : {uint64_t{2}, uint64_t{7}}) {
                        checkPacked(2, r, k, shards, 0,
                                    "aligned_k p" + std::to_string(packet) + " cta" + std::to_string(threads),
                                    [&](const Tensor& in, uint32_t* packed, const CubReductionGeometry& g,
                                        bool maximum, Stream& s) {
                                        launchFp8AlignedKArgPass(in, nullptr, nullptr, g, packet, threads, shards,
                                                                 maximum, 1, s, packed);
                                    },
                                    seed);
                    }
                }
            }
        }
    }
}

// 1132: the packed continuation widens shard-relative candidates to 55-bit indices, so R may exceed 2^32 - 1. The
// partials are synthetic (three shards of 2^31 rows: R = 3 * 2^31 - 5), so no multi-GiB input is needed.
TEST(CubArgReductionFp8Packed, ContinuationIndexesPastFourBillionRows) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    constexpr uint64_t Outer = 2, Shards = 3, K = 5;
    constexpr uint64_t Depth = uint64_t{1} << 31;
    constexpr uint64_t R = Shards * Depth - 5;
    const Fp8ArgFold fold = fp8ArgFold(false, false);
    std::mt19937 rng(1132);
    std::vector<uint64_t> partials(Outer * Shards * K);
    std::vector<uint64_t> expected(Outer * K, 0);
    for (uint64_t o = 0; o < Outer; ++o) {
        for (uint64_t c = 0; c < K; ++c) {
            uint64_t best = FP8_ARG_WIDE_IDENTITY;
            for (uint64_t s = 0; s < Shards; ++s) {
                // Ties on purpose in some columns: the lowest shard (row) must win.
                // Column 4: shard 2 holds the minimum (-448), so its winner indexes past 2^32.
                const uint32_t byte = c < 2                 ? 0x38u
                                      : (c == 4 && s == 2) ? 0xFEu
                                                           : static_cast<uint32_t>(rng() % 0x7Fu);
                const uint32_t row = rng() % 1000u;
                const uint32_t candidate = fp8ArgLaneCandidate(fp8ArgKeys4(byte, fold), 0, row);
                partials[(o * Shards + s) * K + c] = candidate;
                best = fp8ArgMin(best, fp8ArgWiden(candidate, s * Depth));
            }
            expected[o * K + c] = fp8ArgWideIndex(best);
        }
    }
    Tensor packed = makeGpuUnsignedTensor(partials, {Outer, Shards, K}, stream, DataType::UINT32);
    Tensor index(gpuPlacement, TensorDescriptor(DataType::UINT64, {Outer, K}));
    launchFp8ArgPackedContinuation(static_cast<uint32_t*>(packed.getMemPtr<void>()), nullptr, &index,
                                   DataType::FP8_E4M3, Outer, R, K, Shards, Depth, false, 1, stream);
    stream.synchronize();
    const std::vector<uint64_t> got = copyGpuTensorAsUnsigned(index, stream);
    bool past32 = false;
    for (uint64_t i = 0; i < Outer * K; ++i) {
        EXPECT_EQ(expected[i], got[i]) << "output " << i;
        past32 = past32 || got[i] > 0xFFFFFFFFull;
    }
    EXPECT_TRUE(past32) << "no winner landed past 2^32; the test would not exercise the wide index";
}

// 1135: the shard-split packed continuation (many shards, few outputs) and the per-output kernel agree with a host
// reduction of synthetic packed partials: uniform and balanced shard rows, K below and above the 32-column tile, ties,
// both formats and both folds, FP32 value + UINT32 index and UINT64 index only (stride 3).
TEST(CubArgReductionFp8Packed, ShardSplitContinuationMatchesHost) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    struct Shape {
        uint64_t outer, k, shards, uniform_rows;  // uniform_rows 0: balanced split of r
        uint64_t r;
        bool split;
    };
    const std::vector<Shape> shapes{
        {3, 1024, 1024, 32, 1024 * 32 - 7, true},  // 1134 audit worst case: 3 x 1024 outputs, 1024 shards
        {1, 2, 2048, 0, 2048 * 5 + 3, true},       // K = 2, balanced
        {5, 3, 512, 64, 512 * 64, true},
        {2, 33, 300, 0, 300 * 7 + 299, true},      // partial 32-column tile
        {7, 64, 16, 128, 16 * 128 - 1, true},      // the minimum split shard count
        {1000, 7, 40, 0, 40 * 3 + 11, true},
        {4, 31, 1537, 16, 1537 * 16 - 15, true},   // odd group count
        {9, 5, 15, 64, 15 * 64, false},            // below the split shard count: per-output kernel
        {512, 300, 64, 32, 64 * 32, false},        // >= 2^17 outputs: per-output kernel
    };
    std::mt19937 rng(1135);
    for (const Shape& shape : shapes) {
        const uint64_t outputs = shape.outer * shape.k;
        ASSERT_EQ(shape.split, fp8ArgPackedContinuationSplitsShards(shape.outer, shape.k, shape.shards));
        const uint64_t base = shape.uniform_rows != 0 ? shape.uniform_rows : shape.r / shape.shards;
        const uint64_t extra = shape.uniform_rows != 0 ? 0 : shape.r % shape.shards;
        const auto begin = [&](uint64_t s) { return s * base + std::min(s, extra); };
        const auto rows_in = [&](uint64_t s) {
            return s + 1 == shape.shards ? shape.r - begin(s) : begin(s + 1) - begin(s);
        };
        for (const DataType dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
            const bool e5m2 = dtype == DataType::FP8_E5M2;
            for (const bool maximum : {false, true}) {
                const Fp8ArgFold fold = fp8ArgFold(maximum, e5m2);
                std::vector<uint64_t> partials(outputs * shape.shards);
                std::vector<uint64_t> expected(outputs, FP8_ARG_WIDE_IDENTITY);
                const bool tie_heavy = (rng() % 2u) == 0;
                for (uint64_t o = 0; o < shape.outer; ++o) {
                    for (uint64_t s = 0; s < shape.shards; ++s) {
                        for (uint64_t c = 0; c < shape.k; ++c) {
                            const uint32_t byte = tie_heavy ? (rng() % 2u ? 0x38u : 0x00u) : rng() % 0x100u;
                            const uint32_t row = static_cast<uint32_t>(rng() % rows_in(s));
                            const uint32_t candidate = fp8ArgLaneCandidate(fp8ArgKeys4(byte, fold), 0, row);
                            partials[(o * shape.shards + s) * shape.k + c] = candidate;
                            expected[o * shape.k + c] =
                                fp8ArgMin(expected[o * shape.k + c], fp8ArgWiden(candidate, begin(s)));
                        }
                    }
                }
                Tensor packed = makeGpuUnsignedTensor(partials, {shape.outer, shape.shards, shape.k}, stream,
                                                      DataType::UINT32);
                for (const int mode : {0, 1}) {  // FP32 value + UINT32 index; UINT64 index only (stride 3)
                    SCOPED_TRACE(::testing::Message() << "O=" << shape.outer << " K=" << shape.k << " shards="
                                                      << shape.shards << " R=" << shape.r << " dtype="
                                                      << static_cast<int>(dtype) << " max=" << maximum
                                                      << " mode=" << mode);
                    const uint64_t stride = mode == 1 ? 3 : 1;
                    std::optional<Tensor> value;
                    if (mode == 0) value = Tensor(gpuPlacement, TensorDescriptor(DataType::FP32, {shape.outer, shape.k}));
                    Tensor index(gpuPlacement, TensorDescriptor(mode == 0 ? DataType::UINT32 : DataType::UINT64,
                                                                {shape.outer, shape.k}));
                    launchFp8ArgPackedContinuation(static_cast<uint32_t*>(packed.getMemPtr<void>()),
                                                   value ? &*value : nullptr, &index, dtype, shape.outer, shape.r,
                                                   shape.k, shape.shards, shape.uniform_rows, maximum, stride, stream);
                    stream.synchronize();
                    const std::vector<uint64_t> got_indices = copyGpuTensorAsUnsigned(index, stream);
                    const std::vector<float> got_values =
                        value ? copyGpuTensorAsFloat(*value, stream) : std::vector<float>{};
                    for (uint64_t i = 0; i < outputs; ++i) {
                        ASSERT_EQ(fp8ArgWideIndex(expected[i]) * stride, got_indices[i]) << "output " << i;
                        if (value) {
                            const uint32_t byte = fp8ArgValueByte(fp8ArgWideKey(expected[i]),
                                                                  fp8ArgWideNegativeZero(expected[i]), fold);
                            ASSERT_TRUE(Reference::sameValue(Reference::decode(byte, e5m2), got_values[i]))
                                << "output " << i;
                        }
                    }
                }
            }
        }
    }
}

TEST(CubArgReductionFp8Packed, ShardSplitLayout) {
    using namespace CubReductionInternal;
    const auto wide = fp8ArgPackedSplitLayout(1024, 1024);
    EXPECT_EQ(32u, wide.columns);
    EXPECT_EQ(32u, wide.groups);
    EXPECT_EQ(1024u, wide.threads);
    const auto narrow = fp8ArgPackedSplitLayout(3, 512);
    EXPECT_EQ(3u, narrow.columns);
    EXPECT_EQ(128u, narrow.groups);
    EXPECT_EQ(384u, narrow.threads);
    const auto few = fp8ArgPackedSplitLayout(64, 16);
    EXPECT_EQ(32u, few.columns);
    EXPECT_EQ(4u, few.groups);
    EXPECT_EQ(128u, few.threads);
    const auto odd = fp8ArgPackedSplitLayout(7, 1537);
    EXPECT_EQ(7u, odd.columns);
    EXPECT_EQ(146u, odd.groups);  // 1024 / 7
    EXPECT_EQ(1024u, odd.threads);
    // 1137 routing: < 16 Ki outputs from 16 shards, < 64 Ki outputs from 32 shards, and the layout fills a warp.
    EXPECT_TRUE(fp8ArgPackedContinuationSplitsShards(256, 32, 16));     // 8 Ki outputs
    EXPECT_FALSE(fp8ArgPackedContinuationSplitsShards(625, 32, 16));    // 20 Ki outputs, 16 shards
    EXPECT_TRUE(fp8ArgPackedContinuationSplitsShards(625, 32, 32));     // 20 Ki outputs, 32 shards
    EXPECT_FALSE(fp8ArgPackedContinuationSplitsShards(2048, 32, 1024)); // 64 Ki outputs
    EXPECT_FALSE(fp8ArgPackedContinuationSplitsShards(4096, 2, 16));    // K = 2 x 4 groups: 8 of 32 lanes
    EXPECT_TRUE(fp8ArgPackedContinuationSplitsShards(4096, 2, 64));     // K = 2 x 16 groups
    EXPECT_FALSE(fp8ArgPackedContinuationSplitsShards(3, 1024, 15));    // too few shards
}

// A5 / 1142: a VALUE identity without a legal ARG twin contributes no candidate; the rotated RuntimeK ARG pass at
// depth min(R, 1024) (part of every candidate set) still covers the site.
TEST(CubArgReductionFp8Packed, RotatedDeepPassCoversIllegalTwin) {
    using namespace CubReductionInternal;
    Fp8RK::PhysicalKernel aligned{};
    aligned.family = Fp8RK::PhysicalFamily::AlignedK;
    aligned.a = 8;   // packet bytes
    aligned.b = 64;  // CTA threads
    aligned.c = 4;   // shards
    // K = 1001 is not a multiple of the 8-byte packet: no AlignedK twin.
    EXPECT_FALSE(retainedFp8ArgRawTwin(aligned, DataType::FP8_E4M3, 16, 5000, 1001).has_value());
    const auto deep = retainedFp8ArgRotatedDeepPass(DataType::FP8_E4M3, 16, 5000, 1001);
    ASSERT_TRUE(deep.has_value());
    EXPECT_EQ(Fp8RK::PhysicalFamily::RuntimeK, deep->family);
    EXPECT_EQ(1024u, deep->rows_per_shard);
    EXPECT_EQ(5u, deep->shards);
    const auto shallow = retainedFp8ArgRotatedDeepPass(DataType::FP8_E5M2, 16, 300, 7);
    ASSERT_TRUE(shallow.has_value());
    EXPECT_EQ(300u, shallow->rows_per_shard);
    EXPECT_EQ(1u, shallow->shards);
    EXPECT_FALSE(retainedFp8ArgRotatedDeepPass(DataType::FP16, 16, 300, 7).has_value());
    const std::vector<RetainedFp8ValueSource> sources{{aligned, 1}};
    const auto candidates = retainedFp8ArgCandidates(sources, DataType::FP8_E4M3, 16, 5000, 1001);
    ASSERT_FALSE(candidates.empty());
    EXPECT_TRUE(std::any_of(candidates.begin(), candidates.end(), [](const RetainedFp8ArgCandidate& c) {
        return c.plan.family == Fp8RK::PhysicalFamily::RuntimeK && c.plan.rows_per_shard == 1024;
    }));
}

TEST(CubArgReductionFp8Packed, StagedTwinsArePacked) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    // Deep-R shapes whose VALUE choice is staged: production runs first stage + packed continuation, no FP32 pass.
    const std::vector<std::vector<uint64_t>> shapes = {{4, 8192, 21}, {2, 20000, 3}, {12, 4096, 257}, {64, 512, 1024}};
    uint32_t seed = 4000;
    for (const auto& dims : shapes) {
        Tensor input = makeGpuTensor(packedInput(dims[0] * dims[1] * dims[2], false, ++seed), dims, stream,
                                     DataType::FP8_E4M3);
        CubArgReductionOutputOptions outputs;
        outputs.value_output_dtype = DataType::FP32;
        auto stamped = CubArgReduction(CubArgReductionOp::ArgMin, 1, outputs).stamp(input, stream);
        std::string passes;
        for (const std::string& pass : stamped->getPlannedPassNames()) passes += pass + " ";
        SCOPED_TRACE(passes);
        EXPECT_EQ(std::string::npos, passes.find("direct:")) << "a staged ARG twin left an FP32 intermediate pass";
    }
}

#endif  // THOR_CUB_ENABLE_FP8_TYPES
