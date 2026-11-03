// 1141: the installed FP8 RK geometry ranker (benchmarks/fp8_ranker; 1165: model SHA256 b1a7bc53...) reproduces the
// training pipeline's margins exactly, and the typed shortlist is ordered by that margin alone (no family-leader
// prefix), clamped to the requested depth. Host-only.

#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8GeneralRParallelSelector.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8GeometryRankerModel.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKProductionCandidates.h"
#include "test/Utilities/TensorOperations/Fp8GeometryRankerParityFixtures.h"

#include "gtest/gtest.h"

#include <array>
#include <bit>
#include <cstdint>
#include <span>

namespace RK = ThorImplementation::CubReductionInternal::Fp8RK;
namespace G = ThorImplementation::CubReductionInternal::Fp8RK::Fp8GeneralRParallelPolicy;

TEST(Fp8GeometryRanker, ModelMatchesTrainingFixturesExactly) {
    static_assert(RK::Fp8GeometryRankerModel::FEATURE_COUNT == 26);
    for (size_t i = 0; i < RK::Fp8GeometryRankerParity::fixtures.size(); ++i) {
        const auto& fixture = RK::Fp8GeometryRankerParity::fixtures[i];
        const float margin = RK::Fp8GeometryRankerModel::margin(fixture.features);
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
    const auto ow = [](RK::Fp8OneWindowPolicy::Variant) { return 6u; };
    const auto dr = [](RK::Fp8DeepRSmallKPolicy::Variant) { return 4u; };
    const auto ak = [](RK::Fp8AlignedKPolicy::Variant) { return 6u; };
    const std::array<RK::Fp8RuntimeKPolicy::Candidate, 0> runtime{};
    const G::RankerSemantics semantics{G::Fp8Format::E4M3, G::Combine::Sum, G::InputTransform::Identity};
    for (const auto depth : {ThorImplementation::CubReductionTuningDepth::Regular,
                             ThorImplementation::CubReductionTuningDepth::Quick,
                             ThorImplementation::CubReductionTuningDepth::Ultra}) {
        G::GeneralShortlistOptions options;
        options.max_candidates = RK::fp8RKTuningPlan(depth).candidates;
        const auto plans = G::generalShortlist(16, 512, 16, 170, ow, dr, ak,
                                               std::span<const RK::Fp8RuntimeKPolicy::Candidate>(runtime),
                                               semantics, options);
        ASSERT_LE(plans.size(), options.max_candidates);
        if (depth != ThorImplementation::CubReductionTuningDepth::Ultra) ASSERT_EQ(options.max_candidates, plans.size());
        float previous = 1e30f;
        for (const auto& plan : plans) {
            const float margin = RK::Fp8GeometryRankerModel::margin(G::Detail::rankerFeatures(
                plan, 16, 512, 16, semantics, plan.local_rank, G::RankerAggregate::Value));
            EXPECT_LE(margin, previous);
            previous = margin;
        }
    }
    EXPECT_EQ(12u, RK::fp8RKTuningPlan(ThorImplementation::CubReductionTuningDepth::Regular).candidates);
    EXPECT_EQ(7u, RK::fp8RKTuningPlan(ThorImplementation::CubReductionTuningDepth::Regular).rounds);
    EXPECT_EQ(6u, RK::fp8RKTuningPlan(ThorImplementation::CubReductionTuningDepth::Quick).candidates);
    EXPECT_EQ(5u, RK::fp8RKTuningPlan(ThorImplementation::CubReductionTuningDepth::Quick).rounds);
    EXPECT_EQ(24u, RK::fp8RKTuningPlan(ThorImplementation::CubReductionTuningDepth::Ultra).candidates);
    EXPECT_EQ(11u, RK::fp8RKTuningPlan(ThorImplementation::CubReductionTuningDepth::Ultra).rounds);
    EXPECT_EQ(2u, RK::fp8RKTuningPlan(ThorImplementation::CubReductionTuningDepth::Ultra).trimmed);
    EXPECT_EQ(1u, RK::fp8RKTuningPlan(ThorImplementation::CubReductionTuningDepth::Regular).trimmed);
    EXPECT_EQ(1u, RK::fp8RKTuningPlan(ThorImplementation::CubReductionTuningDepth::Quick).trimmed);
    EXPECT_LT(ThorImplementation::cubReductionTuningDepthRank(ThorImplementation::CubReductionTuningDepth::Quick),
              ThorImplementation::cubReductionTuningDepthRank(ThorImplementation::CubReductionTuningDepth::Regular));
    EXPECT_LT(ThorImplementation::cubReductionTuningDepthRank(ThorImplementation::CubReductionTuningDepth::Regular),
              ThorImplementation::cubReductionTuningDepthRank(ThorImplementation::CubReductionTuningDepth::Ultra));
}
