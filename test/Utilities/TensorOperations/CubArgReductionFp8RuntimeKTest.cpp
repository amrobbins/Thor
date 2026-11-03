// ARG-A3: the RuntimeK ARG twins against the FP32 ARG reference: the rotated kernel (8-byte packets, runtime depth;
// staged partials per shard and single-shard Complete outputs) across K tile boundaries, depths and R; the PacketComplete
// twin across W, tiles and R; and wide-K production stamps end to end.

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

std::vector<float> runtimeInput(uint64_t count, bool e5m2, uint32_t seed) {
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

}  // namespace

TEST(CubArgReductionFp8RuntimeK, RotatedPassesMatchReference) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    constexpr uint64_t Outer = 2;
    // K around the 249-component tile and several tiles.
    const std::array<uint64_t, 15> ks = {2, 7, 17, 63, 248, 249, 250, 300, 496, 497, 498, 512, 600, 1024, 2049};
    uint32_t seed = 21;
    for (const DataType dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
        const bool e5m2 = dtype == DataType::FP8_E5M2;
        for (const uint64_t k : ks) {
            const std::vector<uint64_t> rs = k >= 1024 ? std::vector<uint64_t>{1, 33, 700}
                                                       : std::vector<uint64_t>{1, 33, 200, 1500};
            for (const uint64_t r : rs) {
                Tensor input = makeGpuTensor(runtimeInput(Outer * r * k, e5m2, ++seed), {Outer, r, k}, stream, dtype);
                const std::vector<float> stored = copyGpuTensorAsFloat(input, stream);
                const CubReductionGeometry geometry = CubArgReduction::analyzeDenseGeometry({Outer, r, k}, {1});
                for (const uint32_t packet : {8u}) {
                    for (const uint64_t depth : {uint64_t{32}, uint64_t{256}, uint64_t{1024}, r}) {
                        if (!fp8RuntimeKArgSupports(Outer, r, k, depth, packet)) continue;
                        const uint64_t shards = (r + depth - 1) / depth;
                        for (const bool maximum : {false, true}) {
                            // Staged passes write FP32 + UINT32 partials; Complete passes alternate FP8 value-only
                            // and UINT64 index-only (stride 3).
                            for (const int mode : shards > 1 ? std::vector<int>{0} : std::vector<int>{0, 1, 2}) {
                                SCOPED_TRACE(::testing::Message()
                                             << "dtype=" << static_cast<int>(dtype) << " K=" << k << " R=" << r
                                             << " p" << packet << " rows=" << depth << " max=" << maximum
                                             << " mode=" << mode);
                                const uint64_t stride = mode == 2 ? 3 : 1;
                                std::optional<Tensor> value;
                                std::optional<Tensor> index;
                                if (mode != 2)
                                    value = Tensor(gpuPlacement, TensorDescriptor(mode == 1 ? dtype : DataType::FP32,
                                                                                  {Outer, shards, k}));
                                if (mode != 1)
                                    index = Tensor(gpuPlacement,
                                                   TensorDescriptor(mode == 2 ? DataType::UINT64 : DataType::UINT32,
                                                                    {Outer, shards, k}));
                                launchFp8RuntimeKArgPass(input, value ? &*value : nullptr, index ? &*index : nullptr,
                                                         geometry, depth, packet, maximum, stride, stream);
                                stream.synchronize();
                                const std::vector<float> got_values =
                                    value ? copyGpuTensorAsFloat(*value, stream) : std::vector<float>{};
                                const std::vector<uint64_t> got_indices =
                                    index ? copyGpuTensorAsUnsigned(*index, stream) : std::vector<uint64_t>{};
                                for (uint64_t o = 0; o < Outer; ++o) {
                                    for (uint64_t s = 0; s < shards; ++s) {
                                        const uint64_t begin = s * depth;
                                        const uint64_t end = std::min(r, begin + depth);
                                        for (uint64_t c = 0; c < k; ++c) {
                                            Reference::Candidate best{begin, stored[(o * r + begin) * k + c]};
                                            for (uint64_t row = begin + 1; row < end; ++row)
                                                best = Reference::combine(best, {row, stored[(o * r + row) * k + c]},
                                                                          maximum);
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

TEST(CubArgReductionFp8RuntimeK, PacketCompleteMatchesReference) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    constexpr uint64_t Outer = 3;
    uint32_t seed = 77;
    for (const DataType dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
        const bool e5m2 = dtype == DataType::FP8_E5M2;
        for (const uint64_t k : {uint64_t{3}, uint64_t{16}, uint64_t{129}, uint64_t{257}, uint64_t{510},
                                 uint64_t{513}, uint64_t{1000}}) {
            for (const uint64_t r : {uint64_t{1}, uint64_t{4}, uint64_t{9}, uint64_t{33}}) {
                Tensor input = makeGpuTensor(runtimeInput(Outer * r * k, e5m2, ++seed), {Outer, r, k}, stream, dtype);
                const std::vector<float> stored = copyGpuTensorAsFloat(input, stream);
                const CubReductionGeometry geometry = CubArgReduction::analyzeDenseGeometry({Outer, r, k}, {1});
                for (const uint32_t value_tile : {16u, 100u, 131u, 248u, 255u, 257u, 496u}) {
                    const uint32_t tile = fp8PacketCompleteArgTileComponents(k, value_tile);
                    for (const uint32_t w : {1u, 2u, 4u, 8u}) {
                        if (!fp8PacketCompleteArgSupports(Outer, r, k, w, tile)) continue;
                        for (const bool maximum : {false, true}) {
                            for (const int mode : {0, 1, 2}) {  // FP32 + UINT32, FP8 value only, UINT64 index only
                                SCOPED_TRACE(::testing::Message()
                                             << "dtype=" << static_cast<int>(dtype) << " K=" << k << " R=" << r
                                             << " w" << w << " tile=" << tile << " max=" << maximum
                                             << " mode=" << mode);
                                const uint64_t stride = mode == 2 ? 3 : 1;
                                std::optional<Tensor> value;
                                std::optional<Tensor> index;
                                if (mode != 2)
                                    value = Tensor(gpuPlacement,
                                                   TensorDescriptor(mode == 1 ? dtype : DataType::FP32, {Outer, k}));
                                if (mode != 1)
                                    index = Tensor(gpuPlacement, TensorDescriptor(mode == 2 ? DataType::UINT64
                                                                                            : DataType::UINT32,
                                                                                  {Outer, k}));
                                launchFp8PacketCompleteArg(input, value ? &*value : nullptr, index ? &*index : nullptr,
                                                           geometry, w, tile, maximum, stride, stream);
                                stream.synchronize();
                                const std::vector<float> got_values =
                                    value ? copyGpuTensorAsFloat(*value, stream) : std::vector<float>{};
                                const std::vector<uint64_t> got_indices =
                                    index ? copyGpuTensorAsUnsigned(*index, stream) : std::vector<uint64_t>{};
                                for (uint64_t o = 0; o < Outer; ++o) {
                                    for (uint64_t c = 0; c < k; ++c) {
                                        Reference::Candidate best{0, stored[(o * r) * k + c]};
                                        for (uint64_t row = 1; row < r; ++row)
                                            best = Reference::combine(best, {row, stored[(o * r + row) * k + c]},
                                                                      maximum);
                                        if (index) {
                                            ASSERT_EQ(best.index * stride, got_indices[o * k + c])
                                                << "o=" << o << " c=" << c;
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
            }
        }
    }
}

TEST(CubArgReductionFp8RuntimeK, LegalityAndTiles) {
    using namespace CubReductionInternal;
    EXPECT_EQ(249u, fp8RuntimeKArgTileComponents(8));
    EXPECT_TRUE(fp8RuntimeKArgSupports(3, 5000, 4095, 1024, 8));
    EXPECT_TRUE(fp8RuntimeKArgSupports(3, 5000, 4095, 5000, 8));  // single-shard Complete
    EXPECT_FALSE(fp8RuntimeKArgSupports(3, 5000, 4095, 1024, 16));  // 1112: p16 is not compiled
    EXPECT_FALSE(fp8RuntimeKArgSupports(3, 5000, 4095, 1024, 4));
    // PacketComplete (1127): VALUE's tile (capped at 256 and K) unless 256 needs fewer 128-component key words per row.
    EXPECT_EQ(256u, fp8PacketCompleteArgTileComponents(513, 257));   // capped: 5 words per row (1126 halved to 129: 7)
    EXPECT_EQ(255u, fp8PacketCompleteArgTileComponents(510, 255));   // 4 words either way: VALUE's tile
    EXPECT_EQ(256u, fp8PacketCompleteArgTileComponents(2049, 410));  // capped: 17 words (1126 halved to 205: 20)
    EXPECT_EQ(256u, fp8PacketCompleteArgTileComponents(384, 192));   // 3 words, not 4
    EXPECT_EQ(256u, fp8PacketCompleteArgTileComponents(4095, 248));  // 32 words, not 33
    EXPECT_EQ(249u, fp8PacketCompleteArgTileComponents(497, 249));   // 4 words either way
    EXPECT_EQ(100u, fp8PacketCompleteArgTileComponents(200, 100));   // 2 words either way
    EXPECT_EQ(20u, fp8PacketCompleteArgTileComponents(20, 100));
    EXPECT_TRUE(fp8PacketCompleteArgSupports(4096, 4, 513, 1, 256));
    EXPECT_FALSE(fp8PacketCompleteArgSupports(4096, 4, 513, 3, 256));   // W
    EXPECT_FALSE(fp8PacketCompleteArgSupports(4096, 4, 513, 1, 257));   // > 8 components per lane
    EXPECT_FALSE(fp8PacketCompleteArgSupports(1, FP8_ARG_MAX_LANE_ROWS, 8, 1, 8));  // row field
    EXPECT_FALSE(fp8RuntimeKArgSupports(3, FP8_ARG_MAX_LANE_ROWS, 8, FP8_ARG_MAX_LANE_ROWS, 8));  // row field
}

// Wide-K, few-output shapes (the 1108 worst cells, scaled down): whatever the selector picks, results are exact.
TEST(CubArgReductionFp8RuntimeK, WideKProductionStampsAreExact) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const std::vector<std::vector<uint64_t>> shapes = {
        {12, 4096, 257}, {3, 2048, 1024}, {2, 1024, 2049}, {16, 512, 255}, {8, 512, 895}, {1, 4096, 4095},
        // small-R, many-output shapes the VALUE selector sends to PacketComplete (1110 worst Complete cells, scaled)
        {4096, 4, 513}, {2048, 8, 257}, {1024, 16, 510},
        // 1128: few-item PacketComplete choices routed to the rotated pass (if the selector picks PacketComplete)
        {256, 32, 257}, {64, 64, 2049}};
    uint32_t seed = 600;
    for (const DataType dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
        for (const auto& dims : shapes) {
            const uint64_t elements = dims[0] * dims[1] * dims[2];
            Tensor input =
                makeGpuTensor(runtimeInput(elements, dtype == DataType::FP8_E5M2, ++seed), dims, stream, dtype);
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
