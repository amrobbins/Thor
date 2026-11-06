// 1192: the FP32 RK VALUE candidate generator (Fp32RK/Fp32RKCandidates.h), host-only with a stand-in residency: every
// pool entry is a valid FP32 identity within the FP32 launchers' limits, families appear exactly where the FP32 kernels
// cover the site, residency and the scratch budget exclude entries, the families interleave by local rank, the ranker
// features describe the entry, and the untrained shortlist is the pool's prefix. The RuntimeK proposal generator keeps
// FP8's geometry at element size 1.

#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"
#include "Utilities/TensorOperations/Cub/Fp32RK/Fp32RKArgCandidates.h"
#include "Utilities/TensorOperations/Cub/Fp32RK/Fp32RKCandidates.h"
#include "Utilities/TensorOperations/Cub/Fp32RK/Fp32RKCarriedArgCandidates.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKRuntimeProposals.h"
#include "test/Utilities/TensorOperations/Fp32RKRankerParityFixtures.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <set>
#include <tuple>
#include <vector>

using namespace ThorImplementation;
using namespace ThorImplementation::CubReductionInternal;
using Fp8RK::PhysicalFamily;
using Fp8RK::PhysicalKernel;
using Fp8RK::RuntimeLayout;

namespace {

// Every kernel resident (4 CTAs per SM) except OneWindow O4 at K <= 8, which has no FP32 unit (bucket 1).
uint32_t standInResidency(const PhysicalKernel& kernel, uint64_t k) {
    if (kernel.family == PhysicalFamily::OneWindow && kernel.a == 4 && k <= 8) return 0;
    return 4;
}

Fp32RK::Fp32RKSite site(uint64_t outer, uint64_t r, uint64_t k) {
    Fp32RK::Fp32RKSite s;
    s.outer = outer;
    s.r = r;
    s.k = k;
    s.sm_count = 170;
    return s;
}

std::vector<Fp32RK::Fp32RKPoolEntry> pool(const Fp32RK::Fp32RKSite& s) {
    return Fp32RK::fp32RKCandidatePool(s, [&](const PhysicalKernel& kernel) { return standInResidency(kernel, s.k); });
}

bool has(const std::vector<Fp32RK::Fp32RKPoolEntry>& entries, PhysicalFamily family) {
    return std::any_of(entries.begin(), entries.end(), [&](const auto& e) { return e.kernel.family == family; });
}

const std::vector<std::tuple<uint64_t, uint64_t, uint64_t>>& sites() {
    static const std::vector<std::tuple<uint64_t, uint64_t, uint64_t>> s{
        {1, 2, 3},      {1, 4096, 1},    {7, 33, 5},      {64, 1000, 16},  {3, 100000, 125}, {2, 300, 126},
        {16, 2048, 128}, {1, 65, 257},   {100, 31, 1000}, {5, 129, 1001},  {1, 5000, 4096},  {300000, 8, 4},
        {2, 1 << 20, 64}, {1, 64, 62}};
    return s;
}

}  // namespace

TEST(Fp32RKCandidates, PoolEntriesAreValidFp32IdentitiesInterleavedByLocalRank) {
    for (const auto& [outer, r, k] : sites()) {
        SCOPED_TRACE(::testing::Message() << "outer=" << outer << " R=" << r << " K=" << k);
        const auto s = site(outer, r, k);
        const auto entries = pool(s);
        ASSERT_FALSE(entries.empty());
        std::array<uint32_t, 4> next_rank{1, 1, 1, 1};
        uint32_t last_rank = 0;
        for (const auto& entry : entries) {
            const PhysicalKernel& p = entry.kernel;
            EXPECT_EQ(4u, p.element_bytes);
            EXPECT_TRUE(Fp8RK::structurallyValid(p, outer, r, k));
            EXPECT_EQ(4u, p.resident_ctas_per_sm);
            // Local ranks count up within each family, and the pool walks rank by rank across families.
            const size_t family = static_cast<size_t>(p.family);
            EXPECT_EQ(next_rank[family]++, entry.local_rank);
            EXPECT_GE(entry.local_rank, last_rank);
            last_rank = entry.local_rank;
            EXPECT_EQ(1u, std::count_if(entries.begin(), entries.end(), [&](const auto& e) { return e.kernel == p; }));
            // Within the FP32 launchers' limits.
            switch (p.family) {
                case PhysicalFamily::OneWindow: EXPECT_FALSE(p.staged()); break;
                case PhysicalFamily::DeepRSmallK: EXPECT_TRUE(fp32DeepRSmallKSupports(outer, r, k, p.a)); break;
                case PhysicalFamily::AlignedK:
                    EXPECT_TRUE(fp32AlignedKSupports(outer, r, k, static_cast<uint32_t>(p.a),
                                                     static_cast<uint32_t>(p.b), p.c));
                    break;
                case PhysicalFamily::RuntimeK:
                    if (static_cast<RuntimeLayout>(p.a) == RuntimeLayout::PacketComplete) {
                        EXPECT_TRUE(fp32PacketCompleteSupports(outer, r, k, static_cast<uint32_t>(p.b),
                                                               static_cast<uint32_t>(p.c)));
                    } else {
                        EXPECT_EQ(RuntimeLayout::RotatedPrefix, static_cast<RuntimeLayout>(p.a));
                        EXPECT_TRUE(fp32RuntimeKRotatedSupports(outer, r, k, p.e));
                    }
                    break;
            }
        }
    }
}

TEST(Fp32RKCandidates, FamiliesAppearWhereTheFp32KernelsCoverTheSite) {
    for (const auto& [outer, r, k] : sites()) {
        SCOPED_TRACE(::testing::Message() << "outer=" << outer << " R=" << r << " K=" << k);
        const auto entries = pool(site(outer, r, k));
        // OneWindow: K <= 125, or K = 128 (aligned O1).
        EXPECT_EQ(k <= 125 || k == 128, has(entries, PhysicalFamily::OneWindow));
        // DeepRSmallK: K <= 125 and R past the shallowest depth (64).
        EXPECT_EQ(k <= 125 && r > 64, has(entries, PhysicalFamily::DeepRSmallK));
        // AlignedK at every K: the 8-byte packet at even K, the 4-byte one (1228) at odd K.
        EXPECT_TRUE(has(entries, PhysicalFamily::AlignedK));
        // RuntimeK: rotated Prefix once R passes the shallowest depth (32). PacketComplete's tiles hold at least one
        // packet (4 components), as FP8's do (16), so it needs K >= 4; at K >= 124 it always has a one-tile-per-124 plan.
        if (k >= 124) EXPECT_TRUE(has(entries, PhysicalFamily::RuntimeK));
        const bool rotated = std::any_of(entries.begin(), entries.end(), [](const auto& e) {
            return e.kernel.family == PhysicalFamily::RuntimeK &&
                   static_cast<RuntimeLayout>(e.kernel.a) == RuntimeLayout::RotatedPrefix;
        });
        EXPECT_EQ(r > 32, rotated);
        // AlignedK's 16-byte packet only at K % 4 == 0, its 4-byte packet only at odd K.
        for (const auto& e : entries) {
            if (e.kernel.family == PhysicalFamily::AlignedK && e.kernel.a == 16) EXPECT_EQ(0u, k % 4);
            if (e.kernel.family == PhysicalFamily::AlignedK) EXPECT_EQ(k % 2 == 1, e.kernel.a == 4);
        }
    }
    // OneWindow O4 at K <= 8 has no FP32 unit (the stand-in reports it absent): never in the pool.
    for (const auto& entry : pool(site(64, 1000, 8))) {
        EXPECT_FALSE(entry.kernel.family == PhysicalFamily::OneWindow && entry.kernel.a == 4);
    }
}

TEST(Fp32RKCandidates, ResidencyAndScratchBudgetExcludeEntries) {
    auto s = site(16, 2048, 128);
    const auto all = pool(s);
    ASSERT_TRUE(std::any_of(all.begin(), all.end(), [](const auto& e) { return e.kernel.staged(); }));
    // A zero budget keeps only Complete entries.
    s.scratch_budget_bytes = 0;
    const auto complete = pool(s);
    ASSERT_FALSE(complete.empty());
    for (const auto& e : complete) EXPECT_FALSE(e.kernel.staged());
    // A residency of zero everywhere empties the pool.
    s.scratch_budget_bytes = std::numeric_limits<uint64_t>::max();
    EXPECT_TRUE(Fp32RK::fp32RKCandidatePool(s, [](const PhysicalKernel&) { return 0u; }).empty());
    // No residency callback is an error.
    EXPECT_THROW(static_cast<void>(Fp32RK::fp32RKCandidatePool(s, {})), std::invalid_argument);
}

TEST(Fp32RKCandidates, FeaturesDescribeTheEntry) {
    for (const auto& [outer, r, k] : sites()) {
        auto s = site(outer, r, k);
        s.semantics.combine = CubReductionStageCombine::Maximum;
        s.semantics.input_transform = CubReductionStageInputTransform::Square;
        for (const auto& entry : pool(s)) {
            const auto f = Fp32RK::fp32RKRankerFeatures(entry, s, 0);
            for (float x : f) EXPECT_TRUE(std::isfinite(x));
            EXPECT_FLOAT_EQ(std::log2(static_cast<double>(outer)), f[0]);
            EXPECT_FLOAT_EQ(std::log2(static_cast<double>(r)), f[1]);
            EXPECT_FLOAT_EQ(std::log2(static_cast<double>(k)), f[2]);
            EXPECT_EQ(0.0f, f[6]);  // FP8's E4M3 slot
            EXPECT_EQ(3.0f, f[7]);  // Maximum
            EXPECT_EQ(2.0f, f[8]);  // Square
            EXPECT_EQ(static_cast<float>(entry.kernel.family), f[9]);
            EXPECT_EQ(static_cast<float>(entry.local_rank), f[10]);
            EXPECT_EQ(0.0f, f[25]);  // VALUE
        }
    }
}

TEST(Fp32RKCandidates, ShortlistIsThePoolPrefixUntrained) {
    EXPECT_EQ(Fp32RK::fp32RKRankerTrained() ? Fp32RK::FP32_RK_SHORTLIST_DEPTHS[0] : 6u,
              Fp32RK::fp32RKShortlistLength(CubReductionTuningDepth::Quick));
    EXPECT_EQ(Fp32RK::fp32RKRankerTrained() ? Fp32RK::FP32_RK_SHORTLIST_DEPTHS[1] : 12u,
              Fp32RK::fp32RKShortlistLength(CubReductionTuningDepth::Regular));
    EXPECT_EQ(Fp32RK::fp32RKRankerTrained() ? Fp32RK::FP32_RK_SHORTLIST_DEPTHS[2] : 24u,
              Fp32RK::fp32RKShortlistLength(CubReductionTuningDepth::Ultra));
    for (const auto& [outer, r, k] : sites()) {
        const auto s = site(outer, r, k);
        const auto entries = pool(s);
        for (size_t count : {size_t{1}, size_t{6}, size_t{12}, size_t{1000}}) {
            const auto ranked = Fp32RK::fp32RKRankedKernels(entries, s, count);
            ASSERT_EQ(std::min(count, entries.size()), ranked.size());
            for (size_t i = 0; i < ranked.size(); ++i) {
                if (!Fp32RK::fp32RKRankerTrained()) EXPECT_EQ(entries[i].kernel, ranked[i]);
                EXPECT_TRUE(std::any_of(entries.begin(), entries.end(),
                                        [&](const auto& e) { return e.kernel == ranked[i]; }));
            }
        }
    }
}

TEST(Fp32RKCandidates, RuntimeKProposalsAtEachElementWidth) {
    const auto occupancy = [](const PhysicalKernel&) { return 4u; };
    for (const auto& [outer, r, k] : sites()) {
        SCOPED_TRACE(::testing::Message() << "outer=" << outer << " R=" << r << " K=" << k);
        bool fp8_exact = false;
        for (const PhysicalKernel& p : Fp8RK::enumerateRetainedRuntimeK({outer, r, k, 170}, occupancy)) {
            EXPECT_EQ(1u, p.element_bytes);
            EXPECT_TRUE(Fp8RK::structurallyValid(p, outer, r, k));
            const auto layout = static_cast<RuntimeLayout>(p.a);
            if (layout == RuntimeLayout::PacketComplete) {
                EXPECT_GE(p.c, 16u);
                EXPECT_LE(p.c, 496u);
            } else {
                EXPECT_EQ(layout == RuntimeLayout::RotatedExact ? 512u : 497u, p.c);
                fp8_exact |= layout == RuntimeLayout::RotatedExact;
            }
        }
        EXPECT_EQ(r > 32, fp8_exact);
        for (const PhysicalKernel& p : Fp8RK::enumerateRetainedRuntimeK({outer, r, k, 170, 4}, occupancy)) {
            EXPECT_EQ(4u, p.element_bytes);
            EXPECT_TRUE(Fp8RK::structurallyValid(p, outer, r, k));
            const auto layout = static_cast<RuntimeLayout>(p.a);
            EXPECT_NE(RuntimeLayout::RotatedExact, layout);
            if (layout == RuntimeLayout::PacketComplete) {
                EXPECT_GE(p.c, 4u);
                EXPECT_LE(p.c, 124u);
            } else {
                EXPECT_EQ(125u, p.c);
            }
        }
    }
    // 1246: element width 2 (FP16 / BF16) has proposals too (F16RKCandidatesTest); 3 is no element.
    EXPECT_THROW(static_cast<void>(Fp8RK::enumerateRetainedRuntimeK({4, 64, 64, 170, 3}, occupancy)),
                 std::invalid_argument);
}

// 1196: once a trained Fp32RKRankerModel.h is installed, its margins match the training pipeline's fixtures bit for
// bit (the fixtures' features are the sweep's rows plus the aggregate kind, as the model sees them).
TEST(Fp32RKCandidates, ModelMatchesTrainingFixturesExactly) {
    namespace Parity = ThorImplementation::CubReductionInternal::Fp32RK::Fp32RKRankerParity;
    static_assert(Fp32RK::Fp32RKRankerModel::FEATURE_COUNT == 26);
    if (!Fp32RK::Fp32RKRankerModel::TRAINED) {
        GTEST_SKIP() << "no trained FP32 RK ranker installed yet";
    }
    ASSERT_FALSE(Parity::fixtures.empty());
    for (size_t i = 0; i < Parity::fixtures.size(); ++i) {
        const auto& fixture = Parity::fixtures[i];
        const float margin = Fp32RK::Fp32RKRankerModel::margin(fixture.features);
        ASSERT_EQ(std::bit_cast<uint32_t>(fixture.margin), std::bit_cast<uint32_t>(margin)) << "fixture " << i;
    }
}

// 1204: the FP32 RK ARG candidates. Every candidate is a legal, distinct FP32 ARG pass with its packed scratch; the
// twins keep the VALUE pool's order and local ranks; the ARG-only rotated passes (single shard at R <= 1024, depth
// min(R, 1024)) follow with their family's next local rank; the scratch budget excludes staged passes only; and the
// ranker pool mirrors the candidates.
TEST(Fp32RKCandidates, ArgCandidatesAreLegalTwinsThenTheRotatedPasses) {
    for (const auto& [outer, r, k] : sites()) {
        SCOPED_TRACE(::testing::Message() << "outer=" << outer << " R=" << r << " K=" << k);
        Fp32RK::Fp32RKSite s = site(outer, r, k);
        s.semantics.combine = CubReductionStageCombine::Minimum;
        const auto value_pool = pool(s);
        const auto arg = Fp32RK::fp32RKArgCandidates(value_pool, outer, r, k, std::numeric_limits<uint64_t>::max());
        size_t twins = 0;
        for (size_t i = 0; i < arg.size(); ++i) {
            const PhysicalKernel& p = arg[i].identity;
            EXPECT_EQ(4u, p.element_bytes);
            EXPECT_TRUE(Fp32RK::fp32RKArgPassLegal(p, outer, r, k)) << i;
            EXPECT_EQ(Fp32RK::fp32RKArgScratchBytes(p, outer, r, k), arg[i].scratch_bytes);
            for (size_t j = 0; j < i; ++j) {
                const PhysicalKernel& q = arg[j].identity;
                EXPECT_FALSE(p.family == q.family && p.a == q.a && p.b == q.b && p.c == q.c && p.d == q.d &&
                             p.e == q.e && p.f == q.f)
                    << i << " duplicates " << j;
            }
            const auto source = std::find_if(value_pool.begin(), value_pool.end(), [&](const auto& e) {
                return e.kernel.family == p.family && e.kernel.a == p.a && e.kernel.b == p.b && e.kernel.c == p.c &&
                       e.kernel.d == p.d && e.kernel.e == p.e && e.kernel.f == p.f;
            });
            if (source != value_pool.end()) {
                EXPECT_EQ(i, twins) << "twins precede the ARG-only passes";
                EXPECT_EQ(source->local_rank, arg[i].local_rank);
                ++twins;
            } else {
                EXPECT_EQ(PhysicalFamily::RuntimeK, p.family);
                EXPECT_EQ(static_cast<uint64_t>(RuntimeLayout::RotatedPrefix), p.a);
                EXPECT_TRUE(p.e == r || p.e == std::min<uint64_t>(r, 1024)) << p.e;
                uint32_t family_max = 0;
                for (size_t j = 0; j < i; ++j)
                    if (arg[j].identity.family == PhysicalFamily::RuntimeK)
                        family_max = std::max(family_max, arg[j].local_rank);
                EXPECT_EQ(family_max + 1, arg[i].local_rank);
            }
        }
        // The single-shard rotated pass is present at R <= 1024, the 1024-row depth beyond (both as twins or ARG-only).
        const uint64_t depth = std::min<uint64_t>(r, 1024);
        const bool rotated_fits = Fp32RK::fp32RKArgPassLegal(
            [&] {
                PhysicalKernel p{};
                p.family = PhysicalFamily::RuntimeK;
                p.element_bytes = 4;
                p.a = static_cast<uint64_t>(RuntimeLayout::RotatedPrefix);
                p.c = FP32_RUNTIME_K_ROTATED_TILE_COMPONENTS;
                p.d = (k + p.c - 1) / p.c;
                p.e = depth;
                p.f = (r + depth - 1) / depth;
                return p;
            }(),
            outer, r, k);
        if (rotated_fits) {
            EXPECT_TRUE(std::any_of(arg.begin(), arg.end(), [&](const auto& c) {
                return c.identity.family == PhysicalFamily::RuntimeK &&
                       c.identity.a == static_cast<uint64_t>(RuntimeLayout::RotatedPrefix) && c.identity.e == depth;
            }));
        }
        // A zero budget keeps exactly the Complete passes.
        const auto complete = Fp32RK::fp32RKArgCandidates(value_pool, outer, r, k, 0);
        EXPECT_EQ(static_cast<size_t>(std::count_if(arg.begin(), arg.end(),
                                                    [](const auto& c) { return c.scratch_bytes == 0; })),
                  complete.size());
        for (const auto& c : complete) EXPECT_EQ(0u, c.scratch_bytes);
        const auto ranker_pool = Fp32RK::fp32RKArgRankerPool(arg);
        ASSERT_EQ(arg.size(), ranker_pool.size());
        for (size_t i = 0; i < arg.size(); ++i) {
            EXPECT_EQ(arg[i].identity, ranker_pool[i].kernel);
            EXPECT_EQ(arg[i].local_rank, ranker_pool[i].local_rank);
        }
    }
    // An identity of another width, or a rotated layout the FP32 ARG kernel does not have, is not a pass.
    PhysicalKernel fp8{};
    fp8.family = PhysicalFamily::DeepRSmallK;
    fp8.a = 64;
    fp8.element_bytes = 1;
    EXPECT_FALSE(Fp32RK::fp32RKArgPassLegal(fp8, 4, 1000, 16));
    fp8.element_bytes = 4;
    EXPECT_TRUE(Fp32RK::fp32RKArgPassLegal(fp8, 4, 1000, 16));
    PhysicalKernel exact{};
    exact.family = PhysicalFamily::RuntimeK;
    exact.element_bytes = 4;
    exact.a = static_cast<uint64_t>(RuntimeLayout::RotatedExact);
    exact.e = 64;
    EXPECT_FALSE(Fp32RK::fp32RKArgPassLegal(exact, 4, 1000, 16));
}

// 1211-1213: the carried-index ARG candidates, in family order: OneWindow (O1 W1/W2/W4/W8, then O2/O4/O8 W1, at the
// FP32 unit of K's bucket), DeepRSmallK (retained depths below R, K <= 125), AlignedK (16 then 8 bytes, CTA 32 / 64 /
// 128, one shard then the 32- and 128-row targets) for K a multiple of the packet, then rotated (depth R, then the
// retained depths below R); every candidate legal and distinct, local ranks in enumeration order, wide scratch for
// staged passes only, and a zero budget keeps the Complete passes. The rotated Complete pass is always present;
// OneWindow wherever an FP32 unit holds K.
TEST(Fp32RKCandidates, CarriedArgCandidatesInFamilyOrder) {
    for (const auto& [outer, r, k] : sites()) {
        SCOPED_TRACE(::testing::Message() << "outer=" << outer << " R=" << r << " K=" << k);
        const auto arg = Fp32RK::fp32RKCarriedArgCandidates(outer, r, k, std::numeric_limits<uint64_t>::max());
        ASSERT_FALSE(arg.empty());
        std::array<uint32_t, 4> family_rank{};
        int last_family = -1;
        bool complete_rotated = false, one_window = false;
        for (size_t i = 0; i < arg.size(); ++i) {
            const PhysicalKernel& p = arg[i].identity;
            EXPECT_EQ(4u, p.element_bytes);
            EXPECT_TRUE(Fp32RK::fp32RKCarriedArgPassLegal(p, outer, r, k)) << i;
            EXPECT_EQ(Fp32RK::fp32RKCarriedArgScratchBytes(p, outer, r, k), arg[i].scratch_bytes);
            EXPECT_EQ(++family_rank[static_cast<size_t>(p.family)], arg[i].local_rank);
            for (size_t j = 0; j < i; ++j) EXPECT_FALSE(arg[j].identity == p) << i << " duplicates " << j;
            const int family = static_cast<int>(p.family);
            EXPECT_LE(last_family, family) << "families in order";
            last_family = family;
            switch (p.family) {
                case PhysicalFamily::OneWindow:
                    one_window = true;
                    EXPECT_EQ(16u, p.c);
                    EXPECT_TRUE(p.a == 1 || p.b == 1);
                    EXPECT_EQ(0u, arg[i].scratch_bytes);
                    break;
                case PhysicalFamily::DeepRSmallK:
                    EXPECT_LT(p.a, r);
                    EXPECT_LE(k, 125u);
                    EXPECT_NE(0u, arg[i].scratch_bytes);
                    break;
                case PhysicalFamily::AlignedK:
                    EXPECT_EQ(0u, k % (p.a / 4));
                    EXPECT_EQ(p.c > 1, arg[i].scratch_bytes != 0);
                    break;
                case PhysicalFamily::RuntimeK:
                    EXPECT_EQ(static_cast<uint64_t>(RuntimeLayout::RotatedPrefix), p.a);
                    EXPECT_EQ(FP32_RUNTIME_K_ROTATED_TILE_COMPONENTS, p.c);
                    if (p.e == r) complete_rotated = true;
                    EXPECT_EQ(p.f > 1, arg[i].scratch_bytes != 0);
                    break;
                default:
                    ADD_FAILURE() << "unexpected carried family " << family;
            }
        }
        EXPECT_TRUE(complete_rotated);
        EXPECT_EQ(!fp32OneWindowRuntimeKUnits(k).empty() && outer * r * k <= (uint64_t{1} << 30) - 1, one_window);
        const auto complete = Fp32RK::fp32RKCarriedArgCandidates(outer, r, k, 0);
        EXPECT_EQ(static_cast<size_t>(std::count_if(arg.begin(), arg.end(),
                                                    [](const auto& c) { return c.scratch_bytes == 0; })),
                  complete.size());
        for (const auto& c : complete) EXPECT_EQ(0u, c.scratch_bytes);
    }
    // PacketComplete is not carried (user decision 2026-11-04: only if the carried sweep shows a need).
    PhysicalKernel packet{};
    packet.family = PhysicalFamily::RuntimeK;
    packet.element_bytes = 4;
    packet.a = static_cast<uint64_t>(RuntimeLayout::PacketComplete);
    packet.b = 1;
    packet.c = 16;
    EXPECT_FALSE(Fp32RK::fp32RKCarriedArgPassLegal(packet, 4, 1000, 16));
}
