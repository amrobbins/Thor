// The installed FP8 RK geometry ranker (benchmarks/fp8_ranker; model SHA256 b1a7bc53...) reproduces the training
// pipeline's margins exactly, and the typed shortlist is ordered by that margin alone (no family-leader prefix),
// clamped to the requested depth. Host-only.

#include "Utilities/TensorOperations/Reduction/Contest/ReductionTuningPlan.h"
#include "Utilities/TensorOperations/Reduction/RK/Fp8/Fp8RKProductionCandidates.h"
#include "Utilities/TensorOperations/Reduction/RK/Fp8/Fp8RKSelector.h"
#include "Utilities/TensorOperations/Reduction/Rankers/Fp8GeometryRankerModel.h"
#include "test/Utilities/TensorOperations/Reduction/Fp8GeometryRankerParityFixtures.h"

#include "gtest/gtest.h"

#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <span>

namespace RK = ThorImplementation::ReductionInternal::RK;
namespace G = ThorImplementation::ReductionInternal::RK::Fp8::Fp8RKPolicy;
using ThorImplementation::ReductionInternal::reductionTuningPlan;

TEST(Fp8GeometryRanker, ModelMatchesTrainingFixturesExactly) {
    static_assert(RK::Fp8::Fp8GeometryRankerModel::FEATURE_COUNT == 26);
    for (size_t i = 0; i < RK::Fp8::Fp8GeometryRankerParity::fixtures.size(); ++i) {
        const auto& fixture = RK::Fp8::Fp8GeometryRankerParity::fixtures[i];
        const float margin = RK::Fp8::Fp8GeometryRankerModel::margin(fixture.features);
        ASSERT_EQ(std::bit_cast<uint32_t>(fixture.margin), std::bit_cast<uint32_t>(margin)) << "fixture " << i;
    }
}

TEST(Fp8GeometryRanker, AggregateIsTheLastFeature) {
    const G::CandidatePlan plan{{G::Mode::DeepRSmallK, G::DeepRSmallKConfiguration{{256}}}, 4, 1 << 20, 64};
    const G::RankerSemantics semantics{G::Fp8Format::E4M3, G::Combine::Max, G::InputTransform::Identity};
    const auto value = G::Detail::rankerFeatures(plan, 64, 4096, 32, semantics, 2, G::RankerAggregate::Value);
    const auto index = G::Detail::rankerFeatures(plan, 64, 4096, 32, semantics, 2, G::RankerAggregate::ArgIndex);
    const auto both = G::Detail::rankerFeatures(plan, 64, 4096, 32, semantics, 2, G::RankerAggregate::ArgValueIndex);
    EXPECT_EQ(0.0f, value[25]);
    EXPECT_EQ(1.0f, index[25]);
    EXPECT_EQ(2.0f, both[25]);
    for (size_t f = 0; f < 25; ++f) {
        EXPECT_EQ(value[f], index[f]);
        EXPECT_EQ(value[f], both[f]);
    }
}

TEST(Fp8GeometryRanker, TypedShortlistIsScoreOrderedAndClampedToDepth) {
    const auto ow = [](RK::Fp8::Fp8OneWindowPolicy::Variant) { return 6u; };
    const auto dr = [](RK::Fp8::Fp8DeepRSmallKPolicy::Variant) { return 4u; };
    const auto ak = [](RK::Fp8::Fp8AlignedKPolicy::Variant) { return 6u; };
    const std::array<RK::Fp8::Fp8RuntimeKPolicy::Candidate, 0> runtime{};
    const G::RankerSemantics semantics{G::Fp8Format::E4M3, G::Combine::Sum, G::InputTransform::Identity};
    for (const auto depth : {ThorImplementation::ReductionTuningDepth::Regular,
                             ThorImplementation::ReductionTuningDepth::Quick,
                             ThorImplementation::ReductionTuningDepth::Ultra}) {
        G::GeneralShortlistOptions options;
        options.max_candidates = reductionTuningPlan(depth).candidates;
        const auto plans = G::generalShortlist(16, 512, 16, 170, ow, dr, ak,
                                               std::span<const RK::Fp8::Fp8RuntimeKPolicy::Candidate>(runtime),
                                               semantics, options);
        ASSERT_LE(plans.size(), options.max_candidates);
        if (depth != ThorImplementation::ReductionTuningDepth::Ultra) ASSERT_EQ(options.max_candidates, plans.size());
        float previous = 1e30f;
        for (const auto& plan : plans) {
            const float margin = RK::Fp8::Fp8GeometryRankerModel::margin(G::Detail::rankerFeatures(
                plan, 16, 512, 16, semantics, plan.local_rank, G::RankerAggregate::Value));
            EXPECT_LE(margin, previous);
            previous = margin;
        }
    }
    EXPECT_EQ(12u, reductionTuningPlan(ThorImplementation::ReductionTuningDepth::Regular).candidates);
    EXPECT_EQ(7u, reductionTuningPlan(ThorImplementation::ReductionTuningDepth::Regular).rounds);
    EXPECT_EQ(6u, reductionTuningPlan(ThorImplementation::ReductionTuningDepth::Quick).candidates);
    EXPECT_EQ(5u, reductionTuningPlan(ThorImplementation::ReductionTuningDepth::Quick).rounds);
    EXPECT_EQ(24u, reductionTuningPlan(ThorImplementation::ReductionTuningDepth::Ultra).candidates);
    EXPECT_EQ(11u, reductionTuningPlan(ThorImplementation::ReductionTuningDepth::Ultra).rounds);
    EXPECT_EQ(2u, reductionTuningPlan(ThorImplementation::ReductionTuningDepth::Ultra).trimmed);
    EXPECT_EQ(1u, reductionTuningPlan(ThorImplementation::ReductionTuningDepth::Regular).trimmed);
    EXPECT_EQ(1u, reductionTuningPlan(ThorImplementation::ReductionTuningDepth::Quick).trimmed);
    EXPECT_LT(ThorImplementation::reductionTuningDepthRank(ThorImplementation::ReductionTuningDepth::Quick),
              ThorImplementation::reductionTuningDepthRank(ThorImplementation::ReductionTuningDepth::Regular));
    EXPECT_LT(ThorImplementation::reductionTuningDepthRank(ThorImplementation::ReductionTuningDepth::Regular),
              ThorImplementation::reductionTuningDepthRank(ThorImplementation::ReductionTuningDepth::Ultra));
}

namespace {
const auto productionOw = [](RK::Fp8::Fp8OneWindowPolicy::Variant v) -> uint32_t {
    return v.packet_bytes == 16 ? 6 : 0;
};
const auto productionDr = [](RK::Fp8::Fp8DeepRSmallKPolicy::Variant) -> uint32_t { return 4; };
const auto productionAk = [](RK::Fp8::Fp8AlignedKPolicy::Variant v) -> uint32_t { return v.packet == 8 ? 3 : 0; };
const RK::Fp8::Fp8RuntimeKPolicy::Candidate productionRuntime{
    {RK::Fp8::Fp8RuntimeKPolicy::Layout::RotatedPrefix, 0, 497, 1, 256, 8}, 1024, 128ULL * 8 * 256 * 4, 4};
}  // namespace

// Production candidates are the Regular-depth typed shortlist (per-family limit 32, the scratch budget applied),
// deduplicated, each resident, within budget and with work; a zero budget allows only complete kernels.
TEST(Fp8GeometryRanker, ProductionCandidatesAreTheRegularShortlistWithinBudget) {
    const uint32_t regular = reductionTuningPlan(ThorImplementation::ReductionTuningDepth::Regular).candidates;
    ASSERT_TRUE(RK::Fp8::Fp8RuntimeKPolicy::legal(productionRuntime, 128, 2048, 256, 170));
    const std::array runtime{productionRuntime, productionRuntime};  // an identical physical plan, deduplicated
    for (const auto semantics : {G::RankerSemantics{G::Fp8Format::E4M3, G::Combine::Sum, G::InputTransform::Identity},
                                 G::RankerSemantics{G::Fp8Format::E5M2, G::Combine::Max, G::InputTransform::Square},
                                 G::RankerSemantics{G::Fp8Format::E4M3, G::Combine::Product, G::InputTransform::Abs}}) {
        for (const uint64_t budget : {uint64_t{0}, uint64_t{1 << 16}, std::numeric_limits<uint64_t>::max()}) {
            SCOPED_TRACE(budget);
            const RK::Fp8::ProductionSelectionContext context{128, 2048, 256, 170, semantics, budget};
            const auto production =
                RK::Fp8::selectProductionCandidates(context, productionOw, productionDr, productionAk, runtime);
            G::GeneralShortlistOptions options;
            options.max_candidates = regular;
            options.per_family_limit = 32;
            options.max_scratch_bytes = budget;
            const auto shortlist = G::generalShortlist(context.outer, context.r, context.k, context.sm_count,
                                                       productionOw, productionDr, productionAk,
                                                       std::span<const RK::Fp8::Fp8RuntimeKPolicy::Candidate>(runtime),
                                                       semantics, options);
            ASSERT_EQ(production.size(), shortlist.size());
            for (size_t i = 0; i < production.size(); ++i) {
                EXPECT_TRUE(production[i].selection == shortlist[i].selection);
                EXPECT_EQ(production[i].resident_ctas_per_sm, shortlist[i].resident_ctas_per_sm);
                EXPECT_EQ(production[i].scratch_bytes, shortlist[i].scratch_bytes);
                EXPECT_EQ(production[i].first_stage_work_groups, shortlist[i].first_stage_work_groups);
            }
            EXPECT_LE(production.size(), regular);
            for (const auto& plan : production) {
                EXPECT_GT(plan.resident_ctas_per_sm, 0u);
                EXPECT_LE(plan.scratch_bytes, budget);
                EXPECT_GT(plan.first_stage_work_groups, 0u);
                for (const auto& other : production) {
                    if (&other != &plan) EXPECT_FALSE(plan.selection == other.selection);
                }
                if (budget == 0) EXPECT_FALSE(plan.staged());
            }
            // Score order alone (no family-leader prefix), clamped to the Regular depth.
            if (budget == std::numeric_limits<uint64_t>::max()) EXPECT_EQ(production.size(), 12u);
        }
    }
    // A dense legal inventory clamps to the Regular depth, not a smaller per-family shortlist.
    const auto all_ow = [](RK::Fp8::Fp8OneWindowPolicy::Variant) { return 6u; };
    const auto all_ak = [](RK::Fp8::Fp8AlignedKPolicy::Variant) { return 6u; };
    const std::array<RK::Fp8::Fp8RuntimeKPolicy::Candidate, 0> no_runtime{};
    EXPECT_EQ(RK::Fp8::selectProductionCandidates(
                  {16, 512, 16, 170, {G::Fp8Format::E4M3, G::Combine::Sum, G::InputTransform::Identity},
                   std::numeric_limits<uint64_t>::max()},
                  all_ow, productionDr, all_ak, no_runtime)
                  .size(),
              regular);
}

// Legality comes from the physical availability callbacks and the RuntimeK registry, not the ranker.
TEST(Fp8GeometryRanker, ProductionCandidatesFailClosed) {
    const G::RankerSemantics semantics{G::Fp8Format::E4M3, G::Combine::Sum, G::InputTransform::Identity};
    const uint64_t unlimited = std::numeric_limits<uint64_t>::max();
    const std::array runtime{productionRuntime};
    const std::array<RK::Fp8::Fp8RuntimeKPolicy::Candidate, 0> no_runtime{};
    EXPECT_TRUE(RK::Fp8::selectProductionCandidates({0, 2048, 256, 170, semantics, unlimited}, productionOw,
                                                    productionDr, productionAk, runtime)
                    .empty());
    const auto none_ow = [](RK::Fp8::Fp8OneWindowPolicy::Variant) { return 0u; };
    const auto none_dr = [](RK::Fp8::Fp8DeepRSmallKPolicy::Variant) { return 0u; };
    const auto none_ak = [](RK::Fp8::Fp8AlignedKPolicy::Variant) { return 0u; };
    EXPECT_TRUE(RK::Fp8::selectProductionCandidates({128, 2048, 256, 170, semantics, unlimited}, none_ow, none_dr,
                                                    none_ak, no_runtime)
                    .empty());
    auto malformed = productionRuntime;
    malformed.first_stage_ctas = 1;  // fails RuntimeK validation
    const std::array malformed_registry{malformed};
    for (const auto& plan : RK::Fp8::selectProductionCandidates({128, 2048, 256, 170, semantics, unlimited},
                                                                productionOw, productionDr, productionAk,
                                                                malformed_registry)) {
        EXPECT_NE(plan.selection.mode, G::Mode::RuntimeK);
    }
}
