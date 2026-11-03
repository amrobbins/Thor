// ARG-A2: every OneWindow ARG twin (fp8OneWindowArgCompleteKernel) against the FP32 ARG reference, at every legal
// warps-per-output, across K buckets, R (including R smaller than the R workers), both formats and operations, and
// value/index output storage. Index = row * domain_stride.

#include "test/Utilities/TensorOperations/CubReductionTestSupport.h"
#include "test/Utilities/TensorOperations/Fp8ArgReference.h"
#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"
#include "Utilities/TensorOperations/Cub/Fp8ArgKey.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <random>
#include <vector>

using namespace ThorImplementation;
using namespace ThorImplementation::CubReductionTestSupport;
namespace Reference = ThorImplementation::Fp8ArgReference;

#if THOR_CUB_ENABLE_FP8_TYPES

namespace {

std::vector<float> argInput(uint64_t count, bool e5m2, uint32_t seed) {
    static constexpr std::array<uint8_t, 10> SPECIALS{{0x00, 0x80, 0x7F, 0xFF, 0x7E, 0xFE, 0x7B, 0xFB, 0x01, 0x81}};
    static constexpr std::array<uint8_t, 4> TIES{{0x38, 0xB8, 0x00, 0x80}};
    std::mt19937 rng(seed);
    std::vector<float> values(count);
    const bool tie_heavy = (seed & 1u) != 0;
    for (float& value : values) {
        const uint32_t pick = rng() % 8u;
        const uint8_t byte = tie_heavy ? TIES[rng() % TIES.size()]
                             : pick == 0 ? SPECIALS[rng() % SPECIALS.size()]
                                         : static_cast<uint8_t>(rng());
        value = Reference::decode(byte, e5m2);
    }
    return values;
}

}  // namespace

TEST(CubArgReductionFp8OneWindow, EveryUnitMatchesReference) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    constexpr uint64_t Outer = 5;
    const std::array<uint64_t, 18> ks = {2, 3, 7, 17, 63, 127, 129, 497, 6, 10, 20, 24, 33, 65, 111, 200, 300, 512};
    uint32_t seed = 2;
    for (const DataType dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
        const bool e5m2 = dtype == DataType::FP8_E5M2;
        for (const uint64_t k : ks) {
            const std::vector<uint64_t> rs = k >= 200 ? std::vector<uint64_t>{1, 7, 37} : std::vector<uint64_t>{1, 3, 37, 300};
            for (const uint64_t r : rs) {
                Tensor input = makeGpuTensor(argInput(Outer * r * k, e5m2, ++seed), {Outer, r, k}, stream, dtype);
                const std::vector<float> stored = copyGpuTensorAsFloat(input, stream);
                const CubReductionGeometry geometry = CubArgReduction::analyzeDenseGeometry({Outer, r, k}, {1});
                const auto units = fp8OneWindowRuntimeKUnits(k);
                ASSERT_FALSE(units.empty()) << k;
                for (const Fp8OneWindowRuntimeKUnit& unit : units) {
                    for (const uint32_t w : {1u, 2u, 4u, 8u}) {
                        if (!fp8OneWindowArgSupports(Outer, r, k, unit, w)) continue;
                        for (const bool maximum : {false, true}) {
                            for (const int mode : {0, 1, 2}) {  // 0: FP32 value + UINT32 index, 1: FP8 value only,
                                                                // 2: UINT64 index only (stride 3)
                                SCOPED_TRACE(::testing::Message()
                                             << "dtype=" << static_cast<int>(dtype) << " K=" << k << " R=" << r
                                             << " o" << unit.outputs_per_warp << "_w" << w << "_c"
                                             << unit.max_components_per_lane << (unit.load_ahead ? "_la" : "")
                                             << " max=" << maximum << " mode=" << mode);
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
                                launchFp8OneWindowArgComplete(input, value ? &*value : nullptr,
                                                              index ? &*index : nullptr, geometry, unit, w, maximum,
                                                              stride, stream);
                                stream.synchronize();
                                const std::vector<float> values =
                                    value ? copyGpuTensorAsFloat(*value, stream) : std::vector<float>{};
                                const std::vector<uint64_t> indices =
                                    index ? copyGpuTensorAsUnsigned(*index, stream) : std::vector<uint64_t>{};
                                for (uint64_t o = 0; o < Outer; ++o) {
                                    for (uint64_t c = 0; c < k; ++c) {
                                        Reference::Candidate best{0, stored[(o * r) * k + c]};
                                        for (uint64_t row = 1; row < r; ++row)
                                            best = Reference::combine(best, {row, stored[(o * r + row) * k + c]},
                                                                      maximum);
                                        if (index) {
                                            ASSERT_EQ(best.index * stride, indices[o * k + c]) << "o=" << o << " c=" << c;
                                        }
                                        if (value) {
                                            ASSERT_TRUE(Reference::sameValue(best.value, values[o * k + c]))
                                                << "o=" << o << " c=" << c << " " << best.value << " vs "
                                                << values[o * k + c];
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

TEST(CubArgReductionFp8OneWindow, LegalityRule) {
    using namespace CubReductionInternal;
    const auto units = fp8OneWindowRuntimeKUnits(8);
    ASSERT_FALSE(units.empty());
    EXPECT_TRUE(fp8OneWindowArgSupports(10, 100, 8, units.front(), 1));
    EXPECT_FALSE(fp8OneWindowArgSupports(10, FP8_ARG_MAX_LANE_ROWS, 8, units.front(), 1));  // row field limit
    EXPECT_FALSE(fp8OneWindowArgSupports(0, 100, 8, units.front(), 1));
    EXPECT_FALSE(fp8OneWindowArgSupports(10, 100, 600, units.front(), 1));  // no unit holds K = 600
    EXPECT_FALSE(fp8OneWindowArgSupports(uint64_t{1} << 30, 64, 8, units.front(), 1));  // UINT32 element indexing
}

// Production stamps route OneWindow-favoured shapes through the selector; whatever it picks, results are exact.
TEST(CubArgReductionFp8OneWindow, ProductionStampsAreExact) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const std::vector<std::vector<uint64_t>> shapes = {{65536, 4, 2}, {16384, 4, 8}, {4096, 16, 16}, {2048, 8, 64},
                                                       {512, 24, 200}, {256, 12, 512}};
    uint32_t seed = 900;
    for (const DataType dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
        for (const auto& dims : shapes) {
            const uint64_t elements = dims[0] * dims[1] * dims[2];
            Tensor input = makeGpuTensor(argInput(elements, dtype == DataType::FP8_E5M2, ++seed), dims, stream, dtype);
            const std::vector<float> stored = copyGpuTensorAsFloat(input, stream);
            for (const CubArgReductionOp op : {CubArgReductionOp::ArgMin, CubArgReductionOp::ArgMax}) {
                CubArgReductionOutputOptions outputs;
                outputs.value_output_dtype = DataType::FP32;
                auto stamped = CubArgReduction(op, 1, outputs).stamp(input, stream);
                std::string passes;
                for (const std::string& pass : stamped->getPlannedPassNames()) passes += pass + " ";
                SCOPED_TRACE(::testing::Message() << "dims=" << dims[0] << "x" << dims[1] << "x" << dims[2]
                                                  << " op=" << static_cast<int>(op) << " passes=" << passes);
                EXPECT_FALSE(stamped->getPlannedPassNames().empty());
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
