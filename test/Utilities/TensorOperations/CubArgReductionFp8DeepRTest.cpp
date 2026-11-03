// ARG-A3: the DeepRSmallK ARG twin (staged FP32 value + carried index partials) against the FP32 ARG reference per
// shard, at every retained depth and every K ownership, and deep-R production stamps end to end.

#include "test/Utilities/TensorOperations/CubReductionTestSupport.h"
#include "test/Utilities/TensorOperations/Fp8ArgReference.h"
#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

using namespace ThorImplementation;
using namespace ThorImplementation::CubReductionTestSupport;
namespace Reference = ThorImplementation::Fp8ArgReference;

#if THOR_CUB_ENABLE_FP8_TYPES

namespace {

std::vector<float> deepInput(uint64_t count, bool e5m2, uint32_t seed) {
    static constexpr std::array<uint8_t, 10> SPECIALS{{0x00, 0x80, 0x7F, 0xFF, 0x7E, 0xFE, 0x7B, 0xFB, 0x01, 0x81}};
    static constexpr std::array<uint8_t, 4> TIES{{0x38, 0xB8, 0x00, 0x80}};
    std::mt19937 rng(seed);
    std::vector<float> values(count);
    const bool tie_heavy = (seed % 3u) == 0;
    for (float& value : values) {
        const uint8_t byte = tie_heavy               ? TIES[rng() % TIES.size()]
                             : (rng() % 64u) == 0     ? SPECIALS[rng() % SPECIALS.size()]
                                                      : static_cast<uint8_t>(rng());
        value = Reference::decode(byte, e5m2);
    }
    return values;
}

}  // namespace

TEST(CubArgReductionFp8DeepR, FirstStagePartialsMatchReferenceAtEveryDepthAndOwnership) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    constexpr uint64_t Outer = 3;
    // K covers every ownership: tiny 8 lanes (<= 4), tiny 4 lanes (<= 8), 1 (with 3 and 2 workers per component at
    // K 9..16, 1129), 2, 4, 5 and 8 components per lane.
    const std::array<uint64_t, 22> ks = {2,  3,  4,  5,  7,   8,   9,   10,  11,  12,  16,
                                         17, 32, 33, 64, 65, 128, 129, 160, 161, 200, 256};
    const std::array<uint64_t, 5> depths = {64, 128, 256, 512, 1024};
    uint32_t seed = 11;
    for (const DataType dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
        const bool e5m2 = dtype == DataType::FP8_E5M2;
        for (const uint64_t k : ks) {
            for (const uint64_t depth : depths) {
                for (const uint64_t r : {depth + 1, 3 * depth + 5}) {
                    ASSERT_TRUE(fp8DeepRSmallKArgSupports(Outer, r, k, depth));
                    Tensor input = makeGpuTensor(deepInput(Outer * r * k, e5m2, ++seed), {Outer, r, k}, stream, dtype);
                    const std::vector<float> stored = copyGpuTensorAsFloat(input, stream);
                    const CubReductionGeometry geometry = CubArgReduction::analyzeDenseGeometry({Outer, r, k}, {1});
                    const uint64_t shards = (r + depth - 1) / depth;
                    for (const bool maximum : {false, true}) {
                        for (const bool wide : {false, true}) {  // UINT32 stride 1, or UINT64 stride 3
                            SCOPED_TRACE(::testing::Message() << "dtype=" << static_cast<int>(dtype) << " K=" << k
                                                              << " rows=" << depth << " R=" << r << " max=" << maximum
                                                              << " wide=" << wide);
                            const uint64_t stride = wide ? 3 : 1;
                            Tensor values(gpuPlacement, TensorDescriptor(DataType::FP32, {Outer, shards, k}));
                            Tensor indices(gpuPlacement, TensorDescriptor(wide ? DataType::UINT64 : DataType::UINT32,
                                                                          {Outer, shards, k}));
                            launchFp8DeepRSmallKArgFirstStage(input, values, indices, geometry, depth, shards, maximum,
                                                              stride, stream);
                            stream.synchronize();
                            const std::vector<float> got_values = copyGpuTensorAsFloat(values, stream);
                            const std::vector<uint64_t> got_indices = copyGpuTensorAsUnsigned(indices, stream);
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
                                        ASSERT_EQ(best.index * stride, got_indices[at])
                                            << "o=" << o << " shard=" << s << " c=" << c;
                                        ASSERT_TRUE(Reference::sameValue(best.value, got_values[at]))
                                            << "o=" << o << " shard=" << s << " c=" << c << " " << best.value
                                            << " vs " << got_values[at];
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

TEST(CubArgReductionFp8DeepR, LegalityRule) {
    using namespace CubReductionInternal;
    EXPECT_TRUE(fp8DeepRSmallKArgSupports(4, 65, 8, 64));
    EXPECT_FALSE(fp8DeepRSmallKArgSupports(4, 64, 8, 64));    // needs two or more shards
    EXPECT_FALSE(fp8DeepRSmallKArgSupports(4, 1000, 257, 64)); // K <= 256
    EXPECT_FALSE(fp8DeepRSmallKArgSupports(4, 1000, 8, 96));   // not a retained depth
}

// Deep-R, few-output shapes (the 1102 worst cells, scaled down): whatever the selector picks, results are exact.
TEST(CubArgReductionFp8DeepR, DeepRProductionStampsAreExact) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const std::vector<std::vector<uint64_t>> shapes = {{4, 8192, 21}, {2, 20000, 3}, {3, 4096, 65}, {2, 3000, 200},
                                                       {6, 2048, 8},  {1, 65536, 2}};
    uint32_t seed = 300;
    for (const DataType dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
        for (const auto& dims : shapes) {
            const uint64_t elements = dims[0] * dims[1] * dims[2];
            Tensor input = makeGpuTensor(deepInput(elements, dtype == DataType::FP8_E5M2, ++seed), dims, stream, dtype);
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
