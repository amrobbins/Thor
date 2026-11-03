// 1133: the ranked-ARG candidate generator (retainedFp8ArgCandidates) and its launcher (launchRetainedFp8ArgPlan), which
// the ranker training sweep times and 1135 will select from. Every candidate the generator yields from every VALUE
// identity family is checked against the FP32 ARG reference (value + index and index only), and rankerPlanForKernel is
// checked to invert the shortlist binding.

#include "test/Utilities/TensorOperations/CubReductionTestSupport.h"
#include "test/Utilities/TensorOperations/Fp8ArgReference.h"
#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"
#include "Utilities/TensorOperations/Cub/DenseRetainedFp8ArgPhysicalPlan.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8AlignedKSelector.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8DeepRSmallKSelector.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKBoundSelection.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RuntimeKSelector.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <variant>
#include <vector>

using namespace ThorImplementation;
using namespace ThorImplementation::CubReductionTestSupport;
namespace Reference = ThorImplementation::Fp8ArgReference;
namespace F = ThorImplementation::CubReductionInternal::Fp8RK;
namespace G = ThorImplementation::CubReductionInternal::Fp8RK::Fp8GeneralRParallelPolicy;

#if THOR_CUB_ENABLE_FP8_TYPES

namespace {

// Every VALUE identity of every family that could be legal at [outer, r, k] (legality is left to the twin), each
// with a distinct local rank per family.
std::vector<CubReductionInternal::RetainedFp8ValueSource> allValueIdentities(uint64_t outer, uint64_t r, uint64_t k) {
    using namespace CubReductionInternal;
    std::vector<RetainedFp8ValueSource> sources;
    std::array<uint32_t, 4> rank{};
    const auto add = [&](F::PhysicalKernel kernel) {
        sources.push_back({kernel, ++rank[static_cast<size_t>(kernel.family)]});
    };
    for (const Fp8OneWindowRuntimeKUnit& unit : fp8OneWindowRuntimeKUnits(k)) {
        for (const uint64_t w : {1u, 2u, 4u, 8u}) {
            F::PhysicalKernel kernel{};
            kernel.family = F::PhysicalFamily::OneWindow;
            kernel.a = unit.outputs_per_warp;
            kernel.b = w;
            kernel.c = unit.packet_bytes;
            add(kernel);
        }
    }
    for (const uint32_t rows : F::Fp8DeepRSmallKPolicy::compiled_rows) {
        F::PhysicalKernel kernel{};
        kernel.family = F::PhysicalFamily::DeepRSmallK;
        kernel.a = rows;
        add(kernel);
    }
    for (const auto& v : F::Fp8AlignedKPolicy::candidates(outer, r, k)) {
        F::PhysicalKernel kernel{};
        kernel.family = F::PhysicalFamily::AlignedK;
        kernel.a = v.packet;
        kernel.b = v.threads;
        kernel.c = v.shards_per_output;
        add(kernel);
    }
    const uint64_t tiles = (k + 495) / 496;
    const uint64_t value_tile = (k + tiles - 1) / tiles;
    for (const uint64_t w : {1u, 2u, 4u, 8u}) {
        F::PhysicalKernel kernel{};
        kernel.family = F::PhysicalFamily::RuntimeK;
        kernel.a = static_cast<uint64_t>(F::RuntimeLayout::PacketComplete);
        kernel.b = w;
        kernel.c = value_tile;
        kernel.d = tiles;
        kernel.e = r;
        kernel.f = 1;
        add(kernel);
    }
    for (const uint64_t rows : F::Fp8RuntimeKPolicy::rotated_row_depths) {
        if (rows >= r) continue;
        F::PhysicalKernel kernel{};
        kernel.family = F::PhysicalFamily::RuntimeK;
        kernel.a = static_cast<uint64_t>(F::RuntimeLayout::RotatedPrefix);
        kernel.c = 16;
        kernel.d = (k + 15) / 16;
        kernel.e = rows;
        kernel.f = (r + rows - 1) / rows;
        add(kernel);
    }
    return sources;
}

std::vector<float> argInput(uint64_t count, bool e5m2, uint32_t seed) {
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

TEST(CubArgReductionFp8Candidates, RankerPlanInvertsBinding) {
    F::PhysicalKernel one_window{};
    one_window.family = F::PhysicalFamily::OneWindow;
    one_window.a = 2;
    one_window.b = 1;
    one_window.c = 16;
    one_window.resident_ctas_per_sm = 7;
    auto plan = F::rankerPlanForKernel(one_window, 3);
    EXPECT_EQ(G::Mode::OneWindow, plan.selection.mode);
    EXPECT_EQ(3u, plan.local_rank);
    EXPECT_EQ(7u, plan.resident_ctas_per_sm);
    const auto ow = std::get<G::OneWindowConfiguration>(plan.selection.configuration).variant;
    EXPECT_EQ(2u, ow.outputs_per_warp);
    EXPECT_EQ(1u, ow.warps_per_output);
    EXPECT_EQ(16u, ow.packet_bytes);

    F::PhysicalKernel aligned{};
    aligned.family = F::PhysicalFamily::AlignedK;
    aligned.a = 8;
    aligned.b = 64;
    aligned.c = 5;
    plan = F::rankerPlanForKernel(aligned, 1);
    const auto ak = std::get<G::AlignedKConfiguration>(plan.selection.configuration).variant;
    EXPECT_EQ(8u, ak.packet);
    EXPECT_EQ(64u, ak.threads);
    EXPECT_EQ(5u, ak.shards_per_output);

    F::PhysicalKernel rotated{};
    rotated.family = F::PhysicalFamily::RuntimeK;
    rotated.a = static_cast<uint64_t>(F::RuntimeLayout::RotatedPrefix);
    rotated.c = 249;
    rotated.d = 3;
    rotated.e = 256;
    rotated.f = 4;
    plan = F::rankerPlanForKernel(rotated, 2);
    const auto rk = std::get<G::RuntimeKConfiguration>(plan.selection.configuration).variant;
    EXPECT_EQ(F::Fp8RuntimeKPolicy::Layout::RotatedPrefix, rk.layout);
    EXPECT_EQ(249u, rk.tile_components);
    EXPECT_EQ(3u, rk.tile_count);
    EXPECT_EQ(256u, rk.rows_per_shard);
    EXPECT_EQ(4u, rk.shards_per_output);
}

TEST(CubArgReductionFp8Candidates, GeneratorIsUniqueAndRanked) {
    using namespace CubReductionInternal;
    for (const auto& [r, k] : std::vector<std::pair<uint64_t, uint64_t>>{{3, 257}, {40, 17}, {300, 8}, {2100, 64}}) {
        const auto sources = allValueIdentities(4, r, k);
        const auto candidates = retainedFp8ArgCandidates(sources, DataType::FP8_E4M3, 4, r, k);
        ASSERT_FALSE(candidates.empty()) << "R=" << r << " K=" << k;
        std::set<std::string> seen;
        for (const auto& c : candidates) {
            EXPECT_GE(c.local_rank, 1u);
            EXPECT_EQ(c.plan.shards > 1, c.plan.packed_partials);
            EXPECT_EQ(c.plan.family, c.identity.family);
            const std::string key = std::to_string(static_cast<int>(c.plan.family)) + "/" +
                                    std::to_string(c.plan.unit.outputs_per_warp) + "/" +
                                    std::to_string(c.plan.unit.max_components_per_lane) + "/" +
                                    std::to_string(c.plan.unit.load_ahead) + "/" +
                                    std::to_string(c.plan.warps_per_output) + "/" +
                                    std::to_string(c.plan.rows_per_shard) + "/" + std::to_string(c.plan.shards) + "/" +
                                    std::to_string(c.plan.packet_bytes) + "/" + std::to_string(c.plan.cta_threads) +
                                    "/" + std::to_string(c.plan.tile_components);
            EXPECT_TRUE(seen.insert(key).second) << "duplicate ARG candidate " << key;
        }
    }
}

// 1142: selections cache only the ARG identity; retainedFp8ArgPlanForIdentity must rebuild exactly the generated pass.
TEST(CubArgReductionFp8Candidates, IdentityRebuildsThePass) {
    using namespace CubReductionInternal;
    for (const auto& [r, k] : std::vector<std::pair<uint64_t, uint64_t>>{{3, 257}, {40, 17}, {300, 8}, {2100, 64},
                                                                         {12, 510}, {70, 1032}}) {
        for (const auto& c : retainedFp8ArgCandidates(allValueIdentities(4, r, k), DataType::FP8_E4M3, 4, r, k)) {
            SCOPED_TRACE(::testing::Message() << "R=" << r << " K=" << k << " family=" << static_cast<int>(c.plan.family)
                                              << " a=" << c.identity.a << " b=" << c.identity.b << " c=" << c.identity.c
                                              << " e=" << c.identity.e);
            const auto rebuilt = retainedFp8ArgPlanForIdentity(c.identity, DataType::FP8_E4M3, 4, r, k);
            ASSERT_TRUE(rebuilt.has_value());
            EXPECT_EQ(c.plan.family, rebuilt->family);
            EXPECT_EQ(c.plan.unit.outputs_per_warp, rebuilt->unit.outputs_per_warp);
            EXPECT_EQ(c.plan.unit.max_components_per_lane, rebuilt->unit.max_components_per_lane);
            EXPECT_EQ(c.plan.unit.load_ahead, rebuilt->unit.load_ahead);
            EXPECT_EQ(c.plan.warps_per_output, rebuilt->warps_per_output);
            EXPECT_EQ(c.plan.rows_per_shard, rebuilt->rows_per_shard);
            EXPECT_EQ(c.plan.shards, rebuilt->shards);
            EXPECT_EQ(c.plan.packet_bytes, rebuilt->packet_bytes);
            EXPECT_EQ(c.plan.cta_threads, rebuilt->cta_threads);
            EXPECT_EQ(c.plan.tile_components, rebuilt->tile_components);
            EXPECT_EQ(c.plan.packed_partials, rebuilt->packed_partials);
            EXPECT_EQ(retainedFp8ArgScratchBytes(c.plan), retainedFp8ArgScratchBytes(*rebuilt));
        }
    }
}

// 1142: duplicated twins keep their first source's local rank in a canonical (family, local rank) order, whatever
// order the caller lists the VALUE candidates in.
TEST(CubArgReductionFp8Candidates, GeneratorIsOrderIndependent) {
    using namespace CubReductionInternal;
    auto sources = allValueIdentities(4, 2100, 64);
    const auto forward = retainedFp8ArgCandidates(sources, DataType::FP8_E4M3, 4, 2100, 64);
    std::reverse(sources.begin(), sources.end());
    const auto backward = retainedFp8ArgCandidates(sources, DataType::FP8_E4M3, 4, 2100, 64);
    ASSERT_EQ(forward.size(), backward.size());
    for (size_t i = 0; i < forward.size(); ++i) {
        EXPECT_EQ(forward[i].plan.family, backward[i].plan.family);
        EXPECT_EQ(forward[i].plan.rows_per_shard, backward[i].plan.rows_per_shard);
        EXPECT_EQ(forward[i].local_rank, backward[i].local_rank);
    }
}

TEST(CubArgReductionFp8Candidates, EveryCandidateMatchesReference) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    constexpr uint64_t Outer = 3;
    const std::vector<std::pair<uint64_t, uint64_t>> shapes{
        {2, 2}, {3, 257}, {8, 510}, {12, 64}, {40, 17}, {300, 8}, {1500, 33}, {2100, 4}, {70, 1032}};
    uint32_t seed = 1133;
    for (const DataType dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
        const bool e5m2 = dtype == DataType::FP8_E5M2;
        for (const auto& [r, k] : shapes) {
            Tensor input = makeGpuTensor(argInput(Outer * r * k, e5m2, ++seed), {Outer, r, k}, stream, dtype);
            const std::vector<float> stored = copyGpuTensorAsFloat(input, stream);
            const CubReductionGeometry geometry = CubArgReduction::analyzeDenseGeometry({Outer, r, k}, {1});
            const auto candidates = retainedFp8ArgCandidates(allValueIdentities(Outer, r, k), dtype, Outer, r, k);
            ASSERT_FALSE(candidates.empty()) << "R=" << r << " K=" << k;
            for (const bool maximum : {false, true}) {
                std::vector<Reference::Candidate> expected(Outer * k);
                for (uint64_t o = 0; o < Outer; ++o) {
                    for (uint64_t c = 0; c < k; ++c) {
                        Reference::Candidate best{0, stored[(o * r) * k + c]};
                        for (uint64_t row = 1; row < r; ++row)
                            best = Reference::combine(best, {row, stored[(o * r + row) * k + c]}, maximum);
                        expected[o * k + c] = best;
                    }
                }
                for (const auto& candidate : candidates) {
                    const uint64_t scratch_bytes = retainedFp8ArgScratchBytes(candidate.plan);
                    std::optional<Tensor> scratch;
                    if (scratch_bytes)
                        scratch = Tensor(gpuPlacement, TensorDescriptor(DataType::UINT8, {scratch_bytes}));
                    uint32_t* packed = scratch ? static_cast<uint32_t*>(scratch->getMemPtr<void>()) : nullptr;
                    // DeepRSmallK is always staged (packed); every other single-pass kernel also runs index only.
                    for (const bool with_value : {true, false}) {
                        SCOPED_TRACE(::testing::Message()
                                     << "dtype=" << static_cast<int>(dtype) << " R=" << r << " K=" << k
                                     << " family=" << static_cast<int>(candidate.plan.family) << " a="
                                     << candidate.identity.a << " b=" << candidate.identity.b << " c="
                                     << candidate.identity.c << " e=" << candidate.identity.e << " shards="
                                     << candidate.plan.shards << " max=" << maximum << " value=" << with_value);
                        std::optional<Tensor> value;
                        if (with_value) value = Tensor(gpuPlacement, TensorDescriptor(DataType::FP32, {Outer, k}));
                        Tensor index(gpuPlacement, TensorDescriptor(DataType::UINT32, {Outer, k}));
                        launchRetainedFp8ArgPlan(candidate.plan, input, value ? &*value : nullptr, &index, geometry,
                                                 packed, maximum, 1, stream);
                        stream.synchronize();
                        const std::vector<uint64_t> got_indices = copyGpuTensorAsUnsigned(index, stream);
                        const std::vector<float> got_values =
                            value ? copyGpuTensorAsFloat(*value, stream) : std::vector<float>{};
                        for (uint64_t at = 0; at < Outer * k; ++at) {
                            ASSERT_EQ(expected[at].index, got_indices[at]) << "output " << at;
                            if (value) ASSERT_TRUE(Reference::sameValue(expected[at].value, got_values[at])) << "output " << at;
                        }
                    }
                }
            }
        }
    }
}

#endif  // THOR_CUB_ENABLE_FP8_TYPES
