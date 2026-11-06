// 1246: the FP16 / BF16 RK VALUE candidate generator (F16RK/F16RKCandidates.h), host-only with a stand-in residency:
// every pool entry is a valid 2-byte identity within the 2-byte launchers' limits, families appear exactly where the
// 2-byte kernels cover the site, residency and the scratch budget exclude entries, the families interleave by local
// rank, the ranker features describe the entry (the E4M3 slot holding the format), and the untrained shortlist is the
// pool's prefix. The RuntimeK proposal generator at element size 2. 1251: the ARG candidates (F16RKArgCandidates.h).
// 1252: the trained model's parity with the training pipeline's fixtures. 1253: the trained shortlist.

#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"
#include "Utilities/TensorOperations/Cub/F16RK/F16RKArgCandidates.h"
#include "Utilities/TensorOperations/Cub/F16RK/F16RKCandidates.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKRuntimeProposals.h"
#include "test/Utilities/TensorOperations/F16RKRankerParityFixtures.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>
#include <vector>

using namespace ThorImplementation;
using namespace ThorImplementation::CubReductionInternal;
using Fp8RK::PhysicalFamily;
using Fp8RK::PhysicalKernel;
using Fp8RK::RuntimeLayout;

namespace {

// Every kernel resident (4 CTAs per SM) except the OneWindow ownerships with no 2-byte unit at K: O4 at K <= 8, O8 at
// K > 16 (f16OneWindowRuntimeKUnits).
uint32_t standInResidency(const PhysicalKernel& kernel, uint64_t k) {
    if (kernel.family == PhysicalFamily::OneWindow && kernel.a == 4 && k <= 8) return 0;
    if (kernel.family == PhysicalFamily::OneWindow && kernel.a == 8 && k > 16) return 0;
    return 4;
}

F16RK::F16RKSite site(uint64_t outer, uint64_t r, uint64_t k, DataType dtype = DataType::FP16) {
    F16RK::F16RKSite s;
    s.outer = outer;
    s.r = r;
    s.k = k;
    s.sm_count = 170;
    s.input_dtype = dtype;
    return s;
}

std::vector<F16RK::F16RKPoolEntry> pool(const F16RK::F16RKSite& s) {
    return F16RK::f16RKCandidatePool(s, [&](const PhysicalKernel& kernel) { return standInResidency(kernel, s.k); });
}

bool has(const std::vector<F16RK::F16RKPoolEntry>& entries, PhysicalFamily family) {
    return std::any_of(entries.begin(), entries.end(), [&](const auto& e) { return e.kernel.family == family; });
}

const std::vector<std::tuple<uint64_t, uint64_t, uint64_t>>& sites() {
    static const std::vector<std::tuple<uint64_t, uint64_t, uint64_t>> s{
        {1, 2, 3},       {1, 4096, 1},   {7, 33, 5},      {64, 1000, 16},  {3, 100000, 249}, {2, 300, 250},
        {16, 2048, 256}, {1, 65, 257},   {100, 31, 1000}, {5, 129, 1001},  {1, 5000, 4096},  {300000, 8, 8},
        {2, 1 << 20, 64}, {1, 64, 62},   {4, 200, 7},     {9, 77, 120},    {3, 500, 128}};
    return s;
}

}  // namespace

TEST(F16RKCandidates, PoolEntriesAreValidF16IdentitiesInterleavedByLocalRank) {
    for (const auto& [outer, r, k] : sites()) {
        SCOPED_TRACE(::testing::Message() << "outer=" << outer << " R=" << r << " K=" << k);
        const auto s = site(outer, r, k);
        const auto entries = pool(s);
        ASSERT_FALSE(entries.empty());
        std::array<uint32_t, 4> next_rank{1, 1, 1, 1};
        uint32_t last_rank = 0;
        for (const auto& entry : entries) {
            const PhysicalKernel& p = entry.kernel;
            EXPECT_EQ(2u, p.element_bytes);
            EXPECT_TRUE(Fp8RK::structurallyValid(p, outer, r, k));
            EXPECT_EQ(4u, p.resident_ctas_per_sm);
            const size_t family = static_cast<size_t>(p.family);
            EXPECT_EQ(next_rank[family]++, entry.local_rank);
            EXPECT_GE(entry.local_rank, last_rank);
            last_rank = entry.local_rank;
            EXPECT_EQ(1u, std::count_if(entries.begin(), entries.end(), [&](const auto& e) { return e.kernel == p; }));
            // Within the 2-byte launchers' limits.
            switch (p.family) {
                case PhysicalFamily::OneWindow: EXPECT_FALSE(p.staged()); break;
                case PhysicalFamily::DeepRSmallK: EXPECT_TRUE(f16DeepRSmallKSupports(outer, r, k, p.a)); break;
                case PhysicalFamily::AlignedK:
                    EXPECT_TRUE(f16AlignedKSupports(outer, r, k, static_cast<uint32_t>(p.a),
                                                    static_cast<uint32_t>(p.b), p.c));
                    break;
                case PhysicalFamily::RuntimeK:
                    if (static_cast<RuntimeLayout>(p.a) == RuntimeLayout::PacketComplete) {
                        EXPECT_TRUE(f16PacketCompleteSupports(outer, r, k, static_cast<uint32_t>(p.b),
                                                              static_cast<uint32_t>(p.c)));
                    } else {
                        EXPECT_EQ(RuntimeLayout::RotatedPrefix, static_cast<RuntimeLayout>(p.a));
                        EXPECT_TRUE(f16RuntimeKRotatedSupports(outer, r, k, p.e));
                    }
                    break;
            }
        }
    }
}

TEST(F16RKCandidates, FamiliesAppearWhereThe2ByteKernelsCoverTheSite) {
    for (const auto& [outer, r, k] : sites()) {
        SCOPED_TRACE(::testing::Message() << "outer=" << outer << " R=" << r << " K=" << k);
        const auto entries = pool(site(outer, r, k));
        // OneWindow: K <= 249, or K = 256 (aligned O1), within its 32-bit offsets.
        EXPECT_EQ((k <= 249 || k == 256) && outer * r * k <= std::numeric_limits<uint32_t>::max() / 2,
                  has(entries, PhysicalFamily::OneWindow));
        // DeepRSmallK: K <= 249 and R past the shallowest depth (64).
        EXPECT_EQ(k <= 249 && r > 64, has(entries, PhysicalFamily::DeepRSmallK));
        // AlignedK at every K: the 4-byte packet at even K, the 2-byte one at odd K.
        EXPECT_TRUE(has(entries, PhysicalFamily::AlignedK));
        // RuntimeK: rotated Prefix once R passes the shallowest depth (32); PacketComplete's tiles hold at least one
        // packet (8 components), and at K >= 248 there is always a plan.
        if (k >= 248) {
            EXPECT_TRUE(has(entries, PhysicalFamily::RuntimeK));
        }
        const bool rotated = std::any_of(entries.begin(), entries.end(), [](const auto& e) {
            return e.kernel.family == PhysicalFamily::RuntimeK &&
                   static_cast<RuntimeLayout>(e.kernel.a) == RuntimeLayout::RotatedPrefix;
        });
        EXPECT_EQ(r > 32, rotated);
        for (const auto& e : entries) {
            if (e.kernel.family == PhysicalFamily::AlignedK) {
                EXPECT_EQ(0u, k % (e.kernel.a / 2));
                EXPECT_EQ(k % 2 == 1, e.kernel.a == 2);
            }
            if (e.kernel.family == PhysicalFamily::RuntimeK &&
                static_cast<RuntimeLayout>(e.kernel.a) == RuntimeLayout::PacketComplete) {
                EXPECT_GE(e.kernel.c, 8u);
                EXPECT_LE(e.kernel.c, F16_PACKET_COMPLETE_MAX_TILE_COMPONENTS);
            }
        }
    }
    // OneWindow O4 at K <= 8 and O8 past K 16 have no 2-byte unit (the stand-in reports them absent).
    for (uint64_t k : {uint64_t{8}, uint64_t{17}}) {
        for (const auto& entry : pool(site(64, 1000, k))) {
            if (entry.kernel.family != PhysicalFamily::OneWindow) continue;
            EXPECT_FALSE(k <= 8 && entry.kernel.a == 4);
            EXPECT_FALSE(k > 16 && entry.kernel.a == 8);
        }
    }
}

TEST(F16RKCandidates, ResidencyScratchBudgetAndFormatExcludeEntries) {
    auto s = site(16, 2048, 128);
    const auto all = pool(s);
    ASSERT_TRUE(std::any_of(all.begin(), all.end(), [](const auto& e) { return e.kernel.staged(); }));
    s.scratch_budget_bytes = 0;
    const auto complete = pool(s);
    ASSERT_FALSE(complete.empty());
    for (const auto& e : complete) EXPECT_FALSE(e.kernel.staged());
    s.scratch_budget_bytes = std::numeric_limits<uint64_t>::max();
    EXPECT_TRUE(F16RK::f16RKCandidatePool(s, [](const PhysicalKernel&) { return 0u; }).empty());
    EXPECT_THROW(static_cast<void>(F16RK::f16RKCandidatePool(s, {})), std::invalid_argument);
    // The pool is the same for both formats (their kernels share one inventory); other input dtypes are errors.
    const auto bf16 = pool(site(16, 2048, 128, DataType::BF16));
    ASSERT_EQ(all.size(), bf16.size());
    for (size_t i = 0; i < all.size(); ++i) EXPECT_EQ(all[i].kernel, bf16[i].kernel);
    EXPECT_THROW(static_cast<void>(pool(site(16, 2048, 128, DataType::FP32))), std::invalid_argument);
}

TEST(F16RKCandidates, FeaturesDescribeTheEntry) {
    for (const DataType dtype : {DataType::FP16, DataType::BF16}) {
        for (const auto& [outer, r, k] : sites()) {
            auto s = site(outer, r, k, dtype);
            s.semantics.combine = CubReductionStageCombine::Maximum;
            s.semantics.input_transform = CubReductionStageInputTransform::Square;
            for (const auto& entry : pool(s)) {
                const auto f = F16RK::f16RKRankerFeatures(entry, s, 0);
                for (float x : f) EXPECT_TRUE(std::isfinite(x));
                EXPECT_FLOAT_EQ(std::log2(static_cast<double>(outer)), f[0]);
                EXPECT_FLOAT_EQ(std::log2(static_cast<double>(r)), f[1]);
                EXPECT_FLOAT_EQ(std::log2(static_cast<double>(k)), f[2]);
                EXPECT_EQ(dtype == DataType::BF16 ? 1.0f : 0.0f, f[6]);  // the format in FP8's E4M3 slot
                EXPECT_EQ(3.0f, f[7]);                                   // Maximum
                EXPECT_EQ(2.0f, f[8]);                                   // Square
                EXPECT_EQ(static_cast<float>(entry.kernel.family), f[9]);
                EXPECT_EQ(static_cast<float>(entry.local_rank), f[10]);
                // The input-size feature counts 2 bytes per element.
                EXPECT_FLOAT_EQ(static_cast<float>(std::log2(static_cast<double>(outer) * r * k * 2 / 1048576.0 + 1)),
                                f[22]);
                EXPECT_EQ(0.0f, f[25]);  // VALUE
            }
        }
    }
}

TEST(F16RKCandidates, ShortlistIsThePoolPrefixUntrained) {
    const bool trained = F16RK::f16RKRankerTrained();
    EXPECT_EQ(trained ? F16RK::F16_RK_SHORTLIST_DEPTHS[0] : 6u,
              F16RK::f16RKShortlistLength(CubReductionTuningDepth::Quick));
    EXPECT_EQ(trained ? F16RK::F16_RK_SHORTLIST_DEPTHS[1] : 12u,
              F16RK::f16RKShortlistLength(CubReductionTuningDepth::Regular));
    EXPECT_EQ(trained ? F16RK::F16_RK_SHORTLIST_DEPTHS[2] : 24u,
              F16RK::f16RKShortlistLength(CubReductionTuningDepth::Ultra));
    for (const auto& [outer, r, k] : sites()) {
        const auto s = site(outer, r, k);
        const auto entries = pool(s);
        for (size_t count : {size_t{1}, size_t{6}, size_t{12}, size_t{1000}}) {
            const auto ranked = F16RK::f16RKRankedKernels(entries, s, count);
            ASSERT_EQ(std::min(count, entries.size()), ranked.size());
            for (size_t i = 0; i < ranked.size(); ++i) {
                if (!trained) {
                    EXPECT_EQ(entries[i].kernel, ranked[i]);
                }
                EXPECT_TRUE(std::any_of(entries.begin(), entries.end(),
                                        [&](const auto& e) { return e.kernel == ranked[i]; }));
            }
        }
    }
}

TEST(F16RKCandidates, RuntimeKProposalsAtElementWidth2) {
    const auto occupancy = [](const PhysicalKernel&) { return 4u; };
    for (const auto& [outer, r, k] : sites()) {
        SCOPED_TRACE(::testing::Message() << "outer=" << outer << " R=" << r << " K=" << k);
        bool rotated = false;
        for (const PhysicalKernel& p : Fp8RK::enumerateRetainedRuntimeK({outer, r, k, 170, 2}, occupancy)) {
            EXPECT_EQ(2u, p.element_bytes);
            EXPECT_TRUE(Fp8RK::structurallyValid(p, outer, r, k));
            const auto layout = static_cast<RuntimeLayout>(p.a);
            EXPECT_NE(RuntimeLayout::RotatedExact, layout);
            if (layout == RuntimeLayout::PacketComplete) {
                EXPECT_GE(p.c, 8u);
                EXPECT_LE(p.c, 248u);
            } else {
                EXPECT_EQ(F16_RUNTIME_K_ROTATED_TILE_COMPONENTS, p.c);
                rotated = true;
            }
        }
        EXPECT_EQ(r > 32, rotated);
    }
    EXPECT_THROW(static_cast<void>(Fp8RK::enumerateRetainedRuntimeK({4, 64, 64, 170, 3}, occupancy)),
                 std::invalid_argument);
}

// The 2-byte identity rules: a 2-byte AlignedK packet at even K is still a legal identity (the generator only prefers
// wider packets), FP8-only and FP32-only fields are not.
TEST(F16RKCandidates, StructuralValidityAtElementWidth2) {
    PhysicalKernel p{};
    p.element_bytes = 2;
    p.resident_ctas_per_sm = 1;
    p.family = PhysicalFamily::AlignedK;
    p.a = 16;
    p.b = 32;
    p.c = 1;
    p.first_stage_ctas = 3 * 1;  // outer 3, one tile of 256 components
    EXPECT_TRUE(Fp8RK::structurallyValid(p, 3, 100, 256));
    EXPECT_FALSE(Fp8RK::structurallyValid(p, 3, 100, 252));  // K % 8
    p.a = 2;
    p.first_stage_ctas = 3 * 8;  // 256 / 32
    EXPECT_TRUE(Fp8RK::structurallyValid(p, 3, 100, 256));
    p.a = 32;
    EXPECT_FALSE(Fp8RK::structurallyValid(p, 3, 100, 256));
    PhysicalKernel exact{};
    exact.element_bytes = 2;
    exact.resident_ctas_per_sm = 1;
    exact.family = PhysicalFamily::RuntimeK;
    exact.a = static_cast<uint64_t>(RuntimeLayout::RotatedExact);
    exact.c = 512;
    exact.d = 1;
    exact.e = 64;
    exact.f = 2;
    exact.first_stage_ctas = 2;
    EXPECT_FALSE(Fp8RK::structurallyValid(exact, 1, 100, 200));
    PhysicalKernel deep{};
    deep.element_bytes = 2;
    deep.resident_ctas_per_sm = 1;
    deep.family = PhysicalFamily::DeepRSmallK;
    deep.a = 64;
    deep.first_stage_ctas = 2;
    deep.first_stage_scratch_bytes = 2 * 249 * 4;
    EXPECT_TRUE(Fp8RK::structurallyValid(deep, 1, 100, 249));
    deep.first_stage_scratch_bytes = 2 * 250 * 4;
    EXPECT_FALSE(Fp8RK::structurallyValid(deep, 1, 100, 250));
}

// 1251: the FP16 / BF16 RK ARG candidates. Every candidate is a legal, distinct 2-byte ARG pass with its packed
// scratch; the twins keep the VALUE pool's order and local ranks; the ARG-only rotated passes (single shard at
// R <= 1024, depth min(R, 1024)) follow with their family's next local rank; the VALUE pool's 2-byte AlignedK packet
// has no twin; the scratch budget excludes staged passes only; and the ranker pool mirrors the candidates.
TEST(F16RKCandidates, ArgCandidatesAreLegalTwinsThenTheRotatedPasses) {
    for (const auto& [outer, r, k] : sites()) {
        SCOPED_TRACE(::testing::Message() << "outer=" << outer << " R=" << r << " K=" << k);
        F16RK::F16RKSite s = site(outer, r, k);
        s.semantics.combine = CubReductionStageCombine::Minimum;
        const auto value_pool = pool(s);
        const auto arg = F16RK::f16RKArgCandidates(value_pool, outer, r, k, std::numeric_limits<uint64_t>::max());
        size_t twins = 0;
        for (size_t i = 0; i < arg.size(); ++i) {
            const PhysicalKernel& p = arg[i].identity;
            EXPECT_EQ(2u, p.element_bytes);
            EXPECT_TRUE(F16RK::f16RKArgPassLegal(p, outer, r, k)) << i;
            EXPECT_EQ(F16RK::f16RKArgScratchBytes(p, outer, r, k), arg[i].scratch_bytes);
            EXPECT_EQ(arg[i].scratch_bytes, p.first_stage_scratch_bytes);
            EXPECT_FALSE(p.family == PhysicalFamily::AlignedK && p.a == 2) << "the 2-byte packet is VALUE only";
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
                EXPECT_EQ(F16_RUNTIME_K_ROTATED_TILE_COMPONENTS, p.c);
                EXPECT_TRUE(p.e == r || p.e == std::min<uint64_t>(r, 1024)) << p.e;
                uint32_t family_max = 0;
                for (size_t j = 0; j < i; ++j)
                    if (arg[j].identity.family == PhysicalFamily::RuntimeK)
                        family_max = std::max(family_max, arg[j].local_rank);
                EXPECT_EQ(family_max + 1, arg[i].local_rank);
            }
        }
        // Every VALUE entry with a legal twin has it.
        for (const auto& entry : value_pool) {
            PhysicalKernel twin = entry.kernel;
            if (!F16RK::f16RKArgPassLegal(twin, outer, r, k)) continue;
            EXPECT_TRUE(std::any_of(arg.begin(), arg.end(), [&](const auto& c) {
                return c.identity.family == twin.family && c.identity.a == twin.a && c.identity.b == twin.b &&
                       c.identity.c == twin.c && c.identity.d == twin.d && c.identity.e == twin.e &&
                       c.identity.f == twin.f;
            }));
        }
        // The rotated pass at depth min(R, 1024) is present wherever the 2-byte rotated ARG kernel runs.
        const uint64_t depth = std::min<uint64_t>(r, 1024);
        PhysicalKernel rotated{};
        rotated.family = PhysicalFamily::RuntimeK;
        rotated.element_bytes = 2;
        rotated.a = static_cast<uint64_t>(RuntimeLayout::RotatedPrefix);
        rotated.c = F16_RUNTIME_K_ROTATED_TILE_COMPONENTS;
        rotated.d = (k + rotated.c - 1) / rotated.c;
        rotated.e = depth;
        rotated.f = (r + depth - 1) / depth;
        if (F16RK::f16RKArgPassLegal(rotated, outer, r, k)) {
            EXPECT_TRUE(std::any_of(arg.begin(), arg.end(), [&](const auto& c) {
                return c.identity.family == PhysicalFamily::RuntimeK &&
                       c.identity.a == static_cast<uint64_t>(RuntimeLayout::RotatedPrefix) && c.identity.e == depth;
            }));
        }
        // A zero budget keeps exactly the Complete passes.
        const auto complete = F16RK::f16RKArgCandidates(value_pool, outer, r, k, 0);
        EXPECT_EQ(static_cast<size_t>(std::count_if(arg.begin(), arg.end(),
                                                    [](const auto& c) { return c.scratch_bytes == 0; })),
                  complete.size());
        for (const auto& c : complete) EXPECT_EQ(0u, c.scratch_bytes);
        const auto ranker_pool = F16RK::f16RKArgRankerPool(arg);
        ASSERT_EQ(arg.size(), ranker_pool.size());
        for (size_t i = 0; i < arg.size(); ++i) {
            EXPECT_EQ(arg[i].identity, ranker_pool[i].kernel);
            EXPECT_EQ(arg[i].local_rank, ranker_pool[i].local_rank);
        }
    }
    // An identity of another width, or a rotated layout the 2-byte ARG kernel does not have, is not a pass.
    PhysicalKernel deep{};
    deep.family = PhysicalFamily::DeepRSmallK;
    deep.a = 64;
    deep.element_bytes = 4;
    EXPECT_FALSE(F16RK::f16RKArgPassLegal(deep, 4, 1000, 16));
    deep.element_bytes = 2;
    EXPECT_TRUE(F16RK::f16RKArgPassLegal(deep, 4, 1000, 16));
    PhysicalKernel exact{};
    exact.family = PhysicalFamily::RuntimeK;
    exact.element_bytes = 2;
    exact.a = static_cast<uint64_t>(RuntimeLayout::RotatedExact);
    exact.e = 64;
    EXPECT_FALSE(F16RK::f16RKArgPassLegal(exact, 4, 1000, 16));
    PhysicalKernel packet2{};
    packet2.family = PhysicalFamily::AlignedK;
    packet2.element_bytes = 2;
    packet2.a = 2;
    packet2.b = 32;
    packet2.c = 1;
    EXPECT_FALSE(F16RK::f16RKArgPassLegal(packet2, 4, 1000, 7));
    packet2.a = 4;
    EXPECT_FALSE(F16RK::f16RKArgPassLegal(packet2, 4, 1000, 7));  // K % 2
    EXPECT_TRUE(F16RK::f16RKArgPassLegal(packet2, 4, 1000, 8));
}

// 1252: once a trained F16RKRankerModel.h is installed, its margins match the training pipeline's fixtures bit for bit
// (the fixtures' features are the sweep's rows plus the aggregate kind, as the model sees them).
TEST(F16RKCandidates, ModelMatchesTrainingFixturesExactly) {
    namespace Parity = ThorImplementation::CubReductionInternal::F16RK::F16RKRankerParity;
    static_assert(F16RK::F16RKRankerModel::FEATURE_COUNT == 26);
    if (!F16RK::F16RKRankerModel::TRAINED) {
        GTEST_SKIP() << "no trained FP16 / BF16 RK ranker installed yet";
    }
    ASSERT_FALSE(Parity::fixtures.empty());
    for (size_t i = 0; i < Parity::fixtures.size(); ++i) {
        const auto& fixture = Parity::fixtures[i];
        const float margin = F16RK::F16RKRankerModel::margin(fixture.features);
        ASSERT_EQ(std::bit_cast<uint32_t>(fixture.margin), std::bit_cast<uint32_t>(margin)) << "fixture " << i;
    }
}

// 1253: with the trained model, the shortlist is the pool ordered by the model's margin (VALUE aggregate 0, ARG 1 / 2;
// ties in pool order), a prefix of the full ranking, and its depths are F16_RK_SHORTLIST_DEPTHS.
TEST(F16RKCandidates, ShortlistIsARankedPoolPrefixTrained) {
    if (!F16RK::f16RKRankerTrained()) {
        GTEST_SKIP() << "no trained FP16 / BF16 RK ranker installed yet";
    }
    EXPECT_EQ((std::array<size_t, 3>{4, 8, 12}), F16RK::F16_RK_SHORTLIST_DEPTHS);
    for (const auto& [outer, r, k] : sites()) {
        SCOPED_TRACE(::testing::Message() << "outer=" << outer << " R=" << r << " K=" << k);
        for (const DataType dtype : {DataType::FP16, DataType::BF16}) {
            const auto s = site(outer, r, k, dtype);
            const auto entries = pool(s);
            for (uint32_t aggregate : {0u, 1u, 2u}) {
                std::vector<std::pair<float, size_t>> scored;
                for (size_t i = 0; i < entries.size(); ++i)
                    scored.emplace_back(
                        F16RK::F16RKRankerModel::margin(F16RK::f16RKRankerFeatures(entries[i], s, aggregate)), i);
                std::stable_sort(scored.begin(), scored.end(),
                                 [](const auto& a, const auto& b) { return a.first > b.first; });
                const auto full = F16RK::f16RKRankedKernels(entries, s, entries.size(), aggregate);
                ASSERT_EQ(entries.size(), full.size());
                for (size_t i = 0; i < full.size(); ++i) EXPECT_EQ(entries[scored[i].second].kernel, full[i]) << i;
                const auto shortlist = F16RK::f16RKRankedKernels(
                    entries, s, F16RK::f16RKShortlistLength(CubReductionTuningDepth::Regular), aggregate);
                ASSERT_EQ(std::min<size_t>(8, entries.size()), shortlist.size());
                EXPECT_TRUE(std::equal(shortlist.begin(), shortlist.end(), full.begin()));
            }
        }
    }
}
