// ARG-A3: the AlignedK ARG twin (one thread per aligned 2- or 8-byte packet, CTA 32/64/128, balanced shard rows)
// against the FP32 ARG reference: staged partials per shard and single-shard Complete outputs, across K tile
// boundaries, odd row counts (2-byte packets pair rows), shard counts and R, and AlignedK-shaped production stamps end
// to end.

#include "test/Utilities/TensorOperations/CubReductionTestSupport.h"
#include "test/Utilities/TensorOperations/Fp8ArgReference.h"
#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"
#include "Utilities/TensorOperations/Cub/Fp8ArgKey.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <random>
#include <string>
#include <vector>

using namespace ThorImplementation;
using namespace ThorImplementation::CubReductionTestSupport;
namespace Reference = ThorImplementation::Fp8ArgReference;

#if THOR_CUB_ENABLE_FP8_TYPES

namespace {

std::vector<float> alignedInput(uint64_t count, bool e5m2, uint32_t seed) {
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

// The balanced split the kernel uses: shards [0, R % S) own R / S + 1 rows.
uint64_t shardBegin(uint64_t r, uint64_t shards, uint64_t shard) {
    const uint64_t base = r / shards, extra = r % shards;
    return shard * base + std::min(shard, extra);
}

}  // namespace

TEST(CubArgReductionFp8AlignedK, PassesMatchReference) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    constexpr uint64_t Outer = 2;
    // K multiples of 2 and of 8 around the 32/64/128-thread tiles of both packets (64..1024 components).
    const std::array<uint64_t, 12> ks = {2, 6, 8, 16, 62, 64, 66, 128, 256, 264, 1024, 1040};
    uint32_t seed = 41;
    for (const DataType dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
        const bool e5m2 = dtype == DataType::FP8_E5M2;
        for (const uint64_t k : ks) {
            for (const uint64_t r : {uint64_t{1}, uint64_t{2}, uint64_t{7}, uint64_t{33}, uint64_t{300}}) {
                Tensor input = makeGpuTensor(alignedInput(Outer * r * k, e5m2, ++seed), {Outer, r, k}, stream, dtype);
                const std::vector<float> stored = copyGpuTensorAsFloat(input, stream);
                const CubReductionGeometry geometry = CubArgReduction::analyzeDenseGeometry({Outer, r, k}, {1});
                for (const uint32_t packet : {2u, 8u}) {
                    for (const uint32_t threads : {32u, 64u, 128u}) {
                        for (const uint64_t shards : {uint64_t{1}, uint64_t{2}, uint64_t{3}, uint64_t{10}, r}) {
                            if (!fp8AlignedKArgSupports(Outer, r, k, packet, threads, shards)) continue;
                            for (const bool maximum : {false, true}) {
                                // Staged passes write FP32 + UINT32 partials; Complete passes also run FP8 value-only
                                // and UINT64 index-only (stride 3).
                                for (const int mode : shards > 1 ? std::vector<int>{0} : std::vector<int>{0, 1, 2}) {
                                    SCOPED_TRACE(::testing::Message()
                                                 << "dtype=" << static_cast<int>(dtype) << " K=" << k << " R=" << r
                                                 << " p" << packet << " cta" << threads << " shards=" << shards
                                                 << " max=" << maximum << " mode=" << mode);
                                    const uint64_t stride = mode == 2 ? 3 : 1;
                                    std::optional<Tensor> value;
                                    std::optional<Tensor> index;
                                    if (mode != 2)
                                        value = Tensor(gpuPlacement,
                                                       TensorDescriptor(mode == 1 ? dtype : DataType::FP32,
                                                                        {Outer, shards, k}));
                                    if (mode != 1)
                                        index = Tensor(gpuPlacement,
                                                       TensorDescriptor(mode == 2 ? DataType::UINT64 : DataType::UINT32,
                                                                        {Outer, shards, k}));
                                    launchFp8AlignedKArgPass(input, value ? &*value : nullptr,
                                                             index ? &*index : nullptr, geometry, packet, threads,
                                                             shards, maximum, stride, stream);
                                    stream.synchronize();
                                    const std::vector<float> got_values =
                                        value ? copyGpuTensorAsFloat(*value, stream) : std::vector<float>{};
                                    const std::vector<uint64_t> got_indices =
                                        index ? copyGpuTensorAsUnsigned(*index, stream) : std::vector<uint64_t>{};
                                    for (uint64_t o = 0; o < Outer; ++o) {
                                        for (uint64_t s = 0; s < shards; ++s) {
                                            const uint64_t begin = shardBegin(r, shards, s);
                                            const uint64_t end = shardBegin(r, shards, s + 1);
                                            for (uint64_t c = 0; c < k; ++c) {
                                                Reference::Candidate best{begin, stored[(o * r + begin) * k + c]};
                                                for (uint64_t row = begin + 1; row < end; ++row)
                                                    best = Reference::combine(
                                                        best, {row, stored[(o * r + row) * k + c]}, maximum);
                                                const uint64_t at = (o * shards + s) * k + c;
                                                if (index) {
                                                    ASSERT_EQ(best.index * stride, got_indices[at])
                                                        << "o=" << o << " shard=" << s << " c=" << c;
                                                }
                                                if (value) {
                                                    ASSERT_TRUE(Reference::sameValue(best.value, got_values[at]))
                                                        << "o=" << o << " shard=" << s << " c=" << c;
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
        }
    }
}

TEST(CubArgReductionFp8AlignedK, LegalityRule) {
    using namespace CubReductionInternal;
    EXPECT_TRUE(fp8AlignedKArgSupports(4, 1000, 1024, 8, 64, 8));
    EXPECT_TRUE(fp8AlignedKArgSupports(4, 1000, 1026, 2, 128, 1));
    EXPECT_FALSE(fp8AlignedKArgSupports(4, 1000, 1026, 8, 64, 8));   // K not a multiple of the packet
    EXPECT_FALSE(fp8AlignedKArgSupports(4, 1000, 1024, 4, 64, 8));   // packet
    EXPECT_FALSE(fp8AlignedKArgSupports(4, 1000, 1024, 8, 256, 8));  // CTA size
    EXPECT_FALSE(fp8AlignedKArgSupports(4, 8, 1024, 8, 64, 9));      // more shards than rows
    EXPECT_FALSE(fp8AlignedKArgSupports(1, FP8_ARG_MAX_LANE_ROWS, 8, 8, 32, 1));  // row field
}

// AlignedK-shaped production stamps (K a multiple of 8, moderate R): whatever the selector picks, results are exact.
TEST(CubArgReductionFp8AlignedK, ProductionStampsAreExact) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const std::vector<std::vector<uint64_t>> shapes = {
        {64, 512, 1024}, {8, 2048, 2048}, {256, 32, 512}, {32, 300, 4096}, {16, 96, 1536}, {2, 2048, 8192}};
    uint32_t seed = 700;
    for (const DataType dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
        for (const auto& dims : shapes) {
            const uint64_t elements = dims[0] * dims[1] * dims[2];
            Tensor input =
                makeGpuTensor(alignedInput(elements, dtype == DataType::FP8_E5M2, ++seed), dims, stream, dtype);
            const std::vector<float> stored = copyGpuTensorAsFloat(input, stream);
            for (const CubArgReductionOp op : {CubArgReductionOp::ArgMin, CubArgReductionOp::ArgMax}) {
                CubArgReductionOutputOptions outputs;
                outputs.value_output_dtype = DataType::FP32;
                auto stamped = CubArgReduction(op, 1, outputs).stamp(input, stream);
                std::string passes;
                for (const std::string& pass : stamped->getPlannedPassNames()) passes += pass + " ";
                SCOPED_TRACE(::testing::Message() << "dims=" << dims[0] << "x" << dims[1] << "x" << dims[2]
                                                  << " op=" << static_cast<int>(op) << " passes=" << passes);
                stamped->run();
                stream.synchronize();
                const std::vector<float> values = copyGpuTensorAsFloat(stamped->getValueOutputTensor().value(), stream);
                const std::vector<uint64_t> indices =
                    copyGpuTensorAsUnsigned(stamped->getIndexOutputTensor().value(), stream);
                const uint64_t r = dims[1], k = dims[2];
                for (uint64_t o = 0; o < dims[0]; ++o) {
                    for (uint64_t c = 0; c < k; ++c) {
                        Reference::Candidate best{0, stored[(o * r) * k + c]};
                        for (uint64_t row = 1; row < r; ++row)
                            best = Reference::combine(best, {row, stored[(o * r + row) * k + c]},
                                                      op == CubArgReductionOp::ArgMax);
                        ASSERT_EQ(best.index, indices[o * k + c]) << "o=" << o << " c=" << c;
                        ASSERT_TRUE(Reference::sameValue(best.value, values[o * k + c])) << "o=" << o << " c=" << c;
                    }
                }
            }
        }
    }
}

#endif  // THOR_CUB_ENABLE_FP8_TYPES
