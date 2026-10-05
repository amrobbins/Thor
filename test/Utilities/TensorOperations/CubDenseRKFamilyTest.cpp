#include "Utilities/TensorOperations/Cub/DenseReductionFamilies.h"
#include "Utilities/TensorOperations/Cub/CubDataTypePolicy.h"

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace ThorImplementation;
using namespace ThorImplementation::CubReductionInternal;

namespace {

DenseReductionOutputSpec valueOutput(DataType dtype = DataType::FP32) {
    return DenseReductionOutputSpec{
        .produce_value = true,
        .value_dtype = dtype,
        .produce_index = false,
        .index_dtype = DataType::UINT32,
    };
}

DenseReductionProblem valueProblem(const std::vector<uint64_t>& dimensions,
                                   const std::vector<uint32_t>& axes,
                                   DataType output_dtype = DataType::FP32) {
    return makeInitialDenseReductionProblem(
        dimensions, axes, DenseReductionAggregateKind::Value, valueOutput(output_dtype));
}

DenseRKPlanningContext context(DataType input_dtype,
                               DenseRKValueOperation op = DenseRKValueOperation::Sum,
                               uint32_t multiprocessors = 128,
                               bool allow_generic_fallback = false) {
    return DenseRKPlanningContext{
        .original_input_dtype = input_dtype,
        .value_operation = op,
        .multiprocessors = multiprocessors,
        .max_threads_per_sm = 1536,
        .max_blocks_per_sm = 24,
        .allow_generic_rk_fallback = allow_generic_fallback,
    };
}

std::shared_ptr<const DenseRKFamilyPhysicalPlan> rkPlan(const DenseReductionCandidate& candidate) {
    return std::dynamic_pointer_cast<const DenseRKFamilyPhysicalPlan>(candidate.physical_plan);
}

std::shared_ptr<const DenseDirectReductionPhysicalPlan> directPlan(const DenseReductionCandidate& candidate) {
    return std::dynamic_pointer_cast<const DenseDirectReductionPhysicalPlan>(candidate.physical_plan);
}

void expectModernOnePass(const DenseRKFamilyPhysicalPlan& plan) {
    ASSERT_TRUE(denseRKIsFamilyImplementation(plan.implementation));
    const DenseRKProductionInventoryEntry& inventory = denseRKProductionInventoryEntry(plan.implementation);
    ASSERT_TRUE(inventory.strategy.has_value());
    ASSERT_TRUE(inventory.access.has_value());
    EXPECT_EQ(*inventory.strategy, plan.strategy);
    EXPECT_EQ(*inventory.access, plan.access);
    EXPECT_FALSE(inventory.owns_successor);
}

DenseRKOccupancyInfo fixedTwoCtaOccupancy(const DenseRKFamilyPhysicalPlan& plan, void* user_context) {
    auto* calls = static_cast<uint32_t*>(user_context);
    ++*calls;
    return DenseRKOccupancyInfo{2U, 2U * (plan.block_threads / 32U)};
}

}  // namespace

TEST(CubDenseRKFamily, KParallelOccupancyCreditsUsefulWarpsButChargesIdleWarpsForResidency) {
    // K=130 FP16 is legal only for a 4-byte packet. It needs 65 packet owners: three useful warps packed into a
    // four-warp CTA. The fourth warp performs no reduction work, but it still consumes physical CTA residency.
    const DenseReductionProblem problem = valueProblem({1, 64, 130}, {1});
    const DenseReductionCandidate candidate = ReducersDenseRK::propose(
        problem, {0}, context(DataType::FP16, DenseRKValueOperation::Product));
    const auto plan = rkPlan(candidate);
    ASSERT_NE(plan, nullptr);

    EXPECT_EQ(plan->strategy, DenseRKStrategy::KParallel);
    EXPECT_EQ(plan->implementation, DenseRKProductionImplementation::KParallelPass);
    EXPECT_EQ(plan->progress, DenseRKProgress::Complete);
    EXPECT_EQ(plan->packet_bytes, 4U);
    EXPECT_EQ(plan->block_threads, 128U);
    EXPECT_EQ(plan->first_stage_blocks, 1U);
    EXPECT_EQ(plan->physical_stage_warps, 4U);
    EXPECT_EQ(plan->useful_stage_warps, 3U);
    EXPECT_EQ(plan->launch_warp_supply, 3U);
    EXPECT_LE(plan->useful_resident_warp_capacity, plan->resident_device_warps);
    expectModernOnePass(*plan);
}

TEST(CubDenseRKFamily, KParallelUsefulResidentCapacityRemainsBoundedByPhysicalCtaOccupancy) {
    // The same three-useful-warps/four-physical-warps CTA repeated over many outputs makes the residency distinction
    // observable. With one SM in the deterministic fallback, 128-thread CTAs admit 12 physical resident blocks:
    // 48 physical warps, but only 12 * 3 = 36 useful warps. Idle warps consume residency without adding supply.
    const DenseReductionProblem problem = valueProblem({1000, 64, 130}, {1});
    // O=1000 remains as retained run 0, so the reduced R=64 run is nominated at run 1. (The O=1 test above
    // normalizes its leading singleton away, which is why its reduced run is at site 0.)
    const DenseReductionCandidate candidate = ReducersDenseRK::propose(
        problem, {1}, context(DataType::FP16, DenseRKValueOperation::Product, 1));
    const auto plan = rkPlan(candidate);
    ASSERT_NE(plan, nullptr);

    EXPECT_EQ(plan->strategy, DenseRKStrategy::KParallel);
    EXPECT_EQ(plan->packet_bytes, 4U);
    EXPECT_EQ(plan->block_threads, 128U);
    EXPECT_EQ(plan->physical_stage_warps, 4000U);
    EXPECT_EQ(plan->useful_stage_warps, 3000U);
    EXPECT_EQ(plan->max_active_blocks_per_sm, 12U);
    EXPECT_EQ(plan->max_active_warps_per_sm, 48U);
    EXPECT_EQ(plan->resident_device_warps, 48U);
    EXPECT_EQ(plan->useful_resident_warp_capacity, 36U);
    EXPECT_EQ(plan->launch_warp_supply, 36U);
    expectModernOnePass(*plan);
}

TEST(CubDenseRKFamily, DeepUnderfilledKParallelUsesCalibratedSixteenRowStage) {
    // End-to-end calibration places the Complete->Staged crossover below R=103 for an underfilled low-output launch.
    // Product keeps this case in KParallel so the test isolates that family policy.
    const DenseReductionProblem problem = valueProblem({1, 103, 1536}, {1});
    const DenseReductionCandidate candidate = ReducersDenseRK::propose(
        problem, {0}, context(DataType::FP16, DenseRKValueOperation::Product));
    const auto plan = rkPlan(candidate);
    ASSERT_NE(plan, nullptr);

    EXPECT_EQ(plan->strategy, DenseRKStrategy::KParallel);
    EXPECT_EQ(plan->implementation, DenseRKProductionImplementation::KParallelPass);
    EXPECT_EQ(plan->progress, DenseRKProgress::Staged);
    EXPECT_EQ(plan->shards_per_output, 7U);
    EXPECT_EQ(plan->rows_per_shard, 15U);
    // Staged K-parallel has no CTA-wide cooperative work. Keep each independent packet-owner tile to one warp and
    // preserve the widest legal transaction; R sharding, rather than packet narrowing, supplies the reduction-side
    // parallelism.
    EXPECT_EQ(plan->block_threads, 32U);
    EXPECT_EQ(plan->packet_bytes, 16U);
    ASSERT_EQ(candidate.next_problem.runs.size(), 2U);
    EXPECT_EQ(candidate.next_problem.runs[0].kind, DenseReductionRunKind::IntermediateReduction);
    EXPECT_EQ(candidate.next_problem.runs[0].extent, 7U);
    expectModernOnePass(*plan);
}

TEST(CubDenseRKFamily, RealisticLargeReductionUsesDeepestWellSuppliedKParallelStage) {
    // Realistic end-to-end calibration: O=16,R=32768,K=512 FP16 is materially faster when KParallel takes a
    // ~1024-row first-pass bite instead of manufacturing thousands of tiny R shards. With the widest legal packet,
    // 32 shards x 16 outputs x 2 K tiles supplies 1024 one-warp CTAs, comfortably above the calibrated work floor.
    const DenseReductionProblem problem = valueProblem({16, 32768, 512}, {1});
    const DenseReductionCandidate candidate =
        ReducersDenseRK::propose(problem, {1}, context(DataType::FP16));
    const auto plan = rkPlan(candidate);
    ASSERT_NE(plan, nullptr);

    EXPECT_EQ(plan->strategy, DenseRKStrategy::KParallel);
    EXPECT_EQ(plan->implementation, DenseRKProductionImplementation::KParallelPass);
    EXPECT_EQ(plan->progress, DenseRKProgress::Staged);
    EXPECT_EQ(plan->packet_bytes, 16U);
    EXPECT_EQ(plan->block_threads, 32U);
    EXPECT_EQ(plan->rows_per_shard, 1024U);
    EXPECT_EQ(plan->shards_per_output, 32U);
    EXPECT_EQ(plan->first_stage_blocks, 1024U);
    ASSERT_EQ(candidate.next_problem.runs.size(), 3U);
    EXPECT_EQ(candidate.next_problem.runs[0].kind, DenseReductionRunKind::Retained);
    EXPECT_EQ(candidate.next_problem.runs[0].extent, 16U);
    EXPECT_EQ(candidate.next_problem.runs[1].kind, DenseReductionRunKind::IntermediateReduction);
    EXPECT_EQ(candidate.next_problem.runs[1].extent, 32U);
    EXPECT_EQ(candidate.next_problem.runs[2].kind, DenseReductionRunKind::Retained);
    EXPECT_EQ(candidate.next_problem.runs[2].extent, 512U);
    expectModernOnePass(*plan);
}

#if THOR_CUB_ENABLE_FP8_TYPES
TEST(CubDenseRKFamily, Fp8KParallelUsesWidestSaturatedLegalPacket) {
    struct Case {
        uint64_t inner_size;
        size_t expected_packet_bytes;
    };
    const std::vector<Case> cases = {
        {512, 16},
        {24, 8},
        {12, 4},
    };

    // O=2,R=32768 gives these packet populations enough useful resident work to reach their measured saturation
    // targets. Once saturated, transaction width is the stable tie-break. Packet2's separate physical CTA saturation
    // qualification is covered below.
    for (DataType dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
        for (const Case& test_case : cases) {
            SCOPED_TRACE(static_cast<int>(dtype));
            SCOPED_TRACE(test_case.inner_size);
            const DenseReductionProblem problem = valueProblem({2, 32768, test_case.inner_size}, {1});
            const DenseReductionCandidate candidate =
                ReducersDenseRK::propose(problem, {1}, context(dtype));
            const auto plan = rkPlan(candidate);
            ASSERT_NE(plan, nullptr);

            EXPECT_EQ(plan->strategy, DenseRKStrategy::KParallel);
            EXPECT_EQ(plan->implementation, DenseRKProductionImplementation::KParallelPass);
            EXPECT_EQ(plan->packet_bytes, test_case.expected_packet_bytes);
            EXPECT_EQ(plan->progress, DenseRKProgress::Staged);
            EXPECT_GT(plan->shards_per_output, 1U);
            expectModernOnePass(*plan);
        }
    }
}

TEST(CubDenseRKFamily, Fp8SaturatedWidePacketUsesPacketOwnerCtaGeometry) {
    // Packet16 has 128 independent packet owners at K=2048. Once the launch has already crossed the calibrated
    // 12-useful-warps/SM frontier, the measured staged sweep favors a four-warp CTA that matches that K-side owner
    // geometry rather than CTA64 merely because the latter can report equal/greater resident useful capacity.
    const DenseReductionProblem problem = valueProblem({4, 65536, 2048}, {1});
    for (DataType dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
        SCOPED_TRACE(static_cast<int>(dtype));
        const DenseReductionCandidate candidate =
            ReducersDenseRK::propose(problem, {1}, context(dtype, DenseRKValueOperation::Sum, 170));
        const auto plan = rkPlan(candidate);
        ASSERT_NE(plan, nullptr);
        EXPECT_EQ(plan->strategy, DenseRKStrategy::KParallel);
        EXPECT_EQ(plan->progress, DenseRKProgress::Staged);
        EXPECT_EQ(plan->packet_bytes, 16U);
        EXPECT_EQ(plan->block_threads, 128U);
        EXPECT_EQ(plan->shards_per_output, 128U);
        EXPECT_EQ(plan->rows_per_shard, 512U);
        EXPECT_GE(plan->launch_warp_supply, 170U * 12U);
        expectModernOnePass(*plan);
    }
}

TEST(CubDenseRKFamily, Fp8SaturatedWidePacketDoesNotOvervalueCta32Residency) {
    // K=512 packet16 has exactly one warp of packet owners per output/shard. At 64 shards both CTA32 and CTA64
    // expose the same 2048 useful launch warps on the 170-SM calibration geometry, but the sweep measures CTA64 at
    // line rate while CTA32 is materially slower. Saturated wide-packet ranking therefore uses a two-warp minimum CTA.
    const DenseReductionProblem problem = valueProblem({32, 32768, 512}, {1});
    for (DataType dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
        SCOPED_TRACE(static_cast<int>(dtype));
        const DenseReductionCandidate candidate =
            ReducersDenseRK::propose(problem, {1}, context(dtype, DenseRKValueOperation::Sum, 170));
        const auto plan = rkPlan(candidate);
        ASSERT_NE(plan, nullptr);
        EXPECT_EQ(plan->strategy, DenseRKStrategy::KParallel);
        EXPECT_EQ(plan->progress, DenseRKProgress::Staged);
        EXPECT_EQ(plan->packet_bytes, 16U);
        EXPECT_EQ(plan->block_threads, 64U);
        EXPECT_EQ(plan->shards_per_output, 64U);
        EXPECT_EQ(plan->rows_per_shard, 512U);
        EXPECT_GE(plan->launch_warp_supply, 170U * 12U);
        expectModernOnePass(*plan);
    }
}

TEST(CubDenseRKFamily, Fp8SaturatedLayoutKeepsImmediateCleanerShardNeighbor) {
    // 255 shards are the first packet16 layout to reach 12 useful warps/SM for this shape, but 256 shards divide R
    // exactly into 512-row chunks. The production selector deliberately leaves one shard of quantization freedom at
    // the saturation boundary so the clean partition can win before continuation bytes become the tie-break.
    const DenseReductionProblem problem = valueProblem({2, 131072, 2048}, {1});
    for (DataType dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
        SCOPED_TRACE(static_cast<int>(dtype));
        const DenseReductionCandidate candidate =
            ReducersDenseRK::propose(problem, {1}, context(dtype, DenseRKValueOperation::Sum, 170));
        const auto plan = rkPlan(candidate);
        ASSERT_NE(plan, nullptr);
        EXPECT_EQ(plan->strategy, DenseRKStrategy::KParallel);
        EXPECT_EQ(plan->progress, DenseRKProgress::Staged);
        EXPECT_EQ(plan->packet_bytes, 16U);
        EXPECT_EQ(plan->block_threads, 128U);
        EXPECT_EQ(plan->shards_per_output, 256U);
        EXPECT_EQ(plan->rows_per_shard, 512U);
        EXPECT_GE(plan->launch_warp_supply, 170U * 12U);
        expectModernOnePass(*plan);
    }
}

TEST(CubDenseRKFamily, Fp8KParallelUsesNarrowPacketForUsefulParallelismWhenUnderSupplied) {
    // Same reduction, four packet widths: on this small Complete problem none can reach the saturation envelope.
    // Packet2 exposes eight useful packet-owner warps versus one for packet16, so useful work supply wins.
    const DenseReductionProblem problem = valueProblem({1, 64, 512}, {1});
    for (DataType dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
        SCOPED_TRACE(static_cast<int>(dtype));
        const DenseReductionCandidate candidate = ReducersDenseRK::propose(problem, {0}, context(dtype));
        const auto plan = rkPlan(candidate);
        ASSERT_NE(plan, nullptr);
        EXPECT_EQ(plan->strategy, DenseRKStrategy::KParallel);
        EXPECT_EQ(plan->progress, DenseRKProgress::Complete);
        EXPECT_EQ(plan->packet_bytes, 2U);
        EXPECT_EQ(plan->block_threads, 32U);
        EXPECT_EQ(plan->useful_stage_warps, 8U);
        expectModernOnePass(*plan);
    }
}

TEST(CubDenseRKFamily, Fp8PacketTwoOnlyClaimsSaturationAtFourWarpOrWiderCta) {
    // K=1022 makes packet2 the only legal modern KParallel width while providing enough packet owners for a
    // four-warp CTA to expose the calibrated 48-useful-warps/SM frontier. CTA32/64 can match the useful count on
    // paper but calibration showed that those physical specializations cannot sustain line rate, so only CTA>=128
    // may claim saturation.
    const DenseReductionProblem problem = valueProblem({1, 32768, 1022}, {1});
    for (DataType dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
        SCOPED_TRACE(static_cast<int>(dtype));
        const DenseReductionCandidate candidate = ReducersDenseRK::propose(problem, {0}, context(dtype));
        const auto plan = rkPlan(candidate);
        ASSERT_NE(plan, nullptr);
        EXPECT_EQ(plan->strategy, DenseRKStrategy::KParallel);
        EXPECT_EQ(plan->progress, DenseRKProgress::Staged);
        EXPECT_EQ(plan->packet_bytes, 2U);
        EXPECT_GE(plan->block_threads, 128U);
        EXPECT_GE(plan->launch_warp_supply, 128U * 48U);
        expectModernOnePass(*plan);
    }
}

TEST(CubDenseRKFamily, Fp8KParallelLeavesOddKOnGenericFallback) {
    // Two bytes/lane is the narrowest modern FP8 KParallel specialization. Odd K cannot form a whole-row packet and
    // therefore remains visible as a strict modern-family coverage gap until a later FP8 RCooperative/gap milestone.
    const DenseReductionProblem problem = valueProblem({1, 4096, 5}, {1});

    for (DataType dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
        SCOPED_TRACE(static_cast<int>(dtype));
        EXPECT_THROW(
            static_cast<void>(ReducersDenseRK::propose(problem, {0}, context(dtype))), std::logic_error);

        const DenseReductionCandidate fallback =
            ReducersDenseRK::propose(problem, {0}, context(dtype, DenseRKValueOperation::Sum, 128, true));
        ASSERT_EQ(rkPlan(fallback), nullptr);
        ASSERT_NE(directPlan(fallback), nullptr);
    }
}
#endif

TEST(CubDenseRKFamily, DeepLowOutputUsesMinimumComfortableRCooperativeSharding) {
    const DenseReductionProblem problem = valueProblem({1, 8192, 512}, {1});
    const DenseReductionCandidate candidate =
        ReducersDenseRK::propose(problem, {0}, context(DataType::FP16));
    const auto plan = rkPlan(candidate);
    ASSERT_NE(plan, nullptr);

    EXPECT_EQ(plan->strategy, DenseRKStrategy::RCooperative);
    EXPECT_EQ(plan->access, DenseRKAccess::Rotated);
    EXPECT_EQ(plan->implementation,
              DenseRKProductionImplementation::RCooperativeRotatedStaged);
    EXPECT_EQ(plan->progress, DenseRKProgress::Staged);
    EXPECT_EQ(plan->packet_bytes, 16U);
    EXPECT_EQ(plan->block_threads, 256U);
    EXPECT_EQ(plan->rows_per_shard, 128U);
    EXPECT_EQ(plan->shards_per_output, 64U);
    EXPECT_GE(plan->launch_warp_supply, 12U * 128U);
    ASSERT_EQ(candidate.next_problem.runs.size(), 2U);
    EXPECT_EQ(candidate.next_problem.runs[0].kind, DenseReductionRunKind::IntermediateReduction);
    EXPECT_EQ(candidate.next_problem.runs[0].extent, 64U);
    EXPECT_GT(candidate.scratch_bytes, 0U);
    expectModernOnePass(*plan);
}

TEST(CubDenseRKFamily, NarrowPacketLegalKStaysKParallelRatherThanPayingCooperativeOverhead) {
    const DenseReductionProblem problem = valueProblem({1, 8192, 96}, {1});
    const DenseReductionCandidate candidate =
        ReducersDenseRK::propose(problem, {0}, context(DataType::FP16));
    const auto plan = rkPlan(candidate);
    ASSERT_NE(plan, nullptr);

    EXPECT_EQ(plan->strategy, DenseRKStrategy::KParallel);
    EXPECT_EQ(plan->implementation, DenseRKProductionImplementation::KParallelPass);
    EXPECT_EQ(plan->progress, DenseRKProgress::Staged);
    EXPECT_GT(plan->shards_per_output, 1U);
    expectModernOnePass(*plan);
}

TEST(CubDenseRKFamily, AwkwardLowPrecisionAlignmentUsesRotatedRCooperativeStage) {
    const DenseReductionProblem problem = valueProblem({1, 32768, 33}, {1});
    const DenseReductionCandidate candidate =
        ReducersDenseRK::propose(problem, {0}, context(DataType::BF16));
    const auto plan = rkPlan(candidate);
    ASSERT_NE(plan, nullptr);

    EXPECT_EQ(plan->strategy, DenseRKStrategy::RCooperative);
    EXPECT_EQ(plan->access, DenseRKAccess::Rotated);
    EXPECT_EQ(plan->implementation,
              DenseRKProductionImplementation::RCooperativeRotatedStaged);
    EXPECT_EQ(plan->progress, DenseRKProgress::Staged);
    EXPECT_EQ(plan->packet_bytes, 16U);
    EXPECT_EQ(plan->rows_per_shard, 128U);
    EXPECT_EQ(plan->shards_per_output, 256U);
    EXPECT_EQ(candidate.next_problem.runs[0].extent, 256U);
    expectModernOnePass(*plan);
}

TEST(CubDenseRKFamily, DeepOddLowPrecisionKUsesNarrowFlatRCooperativeStage) {
    // The K=1..32 calibration established that flattening multiple narrow rows into contiguous 16-byte packets owns
    // every odd low-precision K in the narrow band.
    const DenseReductionProblem problem = valueProblem({1, 32768, 17}, {1});
    const DenseReductionCandidate candidate =
        ReducersDenseRK::propose(problem, {0}, context(DataType::BF16));
    const auto plan = rkPlan(candidate);
    ASSERT_NE(plan, nullptr);

    EXPECT_EQ(plan->strategy, DenseRKStrategy::RCooperative);
    EXPECT_EQ(plan->access, DenseRKAccess::FlatRows);
    EXPECT_EQ(plan->implementation,
              DenseRKProductionImplementation::RCooperativeFlatRowsStaged);
    EXPECT_EQ(plan->progress, DenseRKProgress::Staged);
    EXPECT_EQ(plan->rows_per_shard, 512U);
    EXPECT_GT(plan->shards_per_output, 1U);
    EXPECT_EQ(candidate.next_problem.runs[0].kind, DenseReductionRunKind::IntermediateReduction);
    expectModernOnePass(*plan);
}

TEST(CubDenseRKFamily, EvenLowPrecisionK14RemainsKParallelAtCalibratedBoundary) {
    const DenseReductionProblem problem = valueProblem({64, 32768, 14}, {1});
    const DenseReductionCandidate candidate =
        ReducersDenseRK::propose(problem, {1}, context(DataType::FP16));
    const auto plan = rkPlan(candidate);
    ASSERT_NE(plan, nullptr);

    EXPECT_EQ(plan->strategy, DenseRKStrategy::KParallel);
    EXPECT_EQ(plan->implementation, DenseRKProductionImplementation::KParallelPass);
    expectModernOnePass(*plan);
}

TEST(CubDenseRKFamily, EvenLowPrecisionK16UsesNarrowFlatWithDeepShardGeometry) {
    const DenseReductionProblem problem = valueProblem({64, 32768, 16}, {1});
    const DenseReductionCandidate candidate =
        ReducersDenseRK::propose(problem, {1}, context(DataType::FP16));
    const auto plan = rkPlan(candidate);
    ASSERT_NE(plan, nullptr);

    EXPECT_EQ(plan->strategy, DenseRKStrategy::RCooperative);
    EXPECT_EQ(plan->access, DenseRKAccess::FlatRows);
    EXPECT_EQ(plan->implementation,
              DenseRKProductionImplementation::RCooperativeFlatRowsStaged);
    EXPECT_EQ(plan->rows_per_shard, 512U);
    expectModernOnePass(*plan);
}

TEST(CubDenseRKFamily, LowPrecisionK20UsesNarrowFlatWithShallowShardGeometry) {
    const DenseReductionProblem problem = valueProblem({64, 32768, 20}, {1});
    const DenseReductionCandidate candidate =
        ReducersDenseRK::propose(problem, {1}, context(DataType::BF16));
    const auto plan = rkPlan(candidate);
    ASSERT_NE(plan, nullptr);

    EXPECT_EQ(plan->strategy, DenseRKStrategy::RCooperative);
    EXPECT_EQ(plan->access, DenseRKAccess::FlatRows);
    EXPECT_EQ(plan->implementation,
              DenseRKProductionImplementation::RCooperativeFlatRowsStaged);
    EXPECT_EQ(plan->rows_per_shard, 128U);
    expectModernOnePass(*plan);
}

TEST(CubDenseRKFamily, ExactOccupancyHookControlsResidentCapacityUsedByPolicy) {
    uint32_t occupancy_calls = 0;
    DenseRKPlanningContext planning = context(DataType::BF16);
    planning.occupancy_query = fixedTwoCtaOccupancy;
    planning.occupancy_query_context = &occupancy_calls;

    const DenseReductionProblem problem = valueProblem({1, 32768, 33}, {1});
    const DenseReductionCandidate candidate = ReducersDenseRK::propose(problem, {0}, planning);
    const auto plan = rkPlan(candidate);
    ASSERT_NE(plan, nullptr);
    EXPECT_GT(occupancy_calls, 0U);
    EXPECT_EQ(plan->max_active_blocks_per_sm, 2U);
    EXPECT_EQ(plan->max_active_warps_per_sm, 2U * (plan->block_threads / 32U));
    EXPECT_EQ(plan->wave_capacity_blocks, 2U * planning.multiprocessors);
}

TEST(CubDenseRKFamily, IntermediateInputIsReplannedAsFp32RatherThanOriginalLowPrecisionStorage) {
    DenseReductionProblem problem = valueProblem({1, 8192, 1536}, {1});
    problem = advanceDenseReductionProblem(problem, {0}, 64);
    ASSERT_EQ(problem.input_state, DenseReductionInputState::IntermediateAggregate);

    const DenseReductionCandidate candidate =
        ReducersDenseRK::propose(problem, {0}, context(DataType::BF16));
    const auto plan = rkPlan(candidate);
    ASSERT_NE(plan, nullptr);
    EXPECT_EQ(plan->input_dtype, DataType::FP32);
}

TEST(CubDenseRKFamily, ModernPassNeverOwnsItsSuccessor) {
    for (const auto& [dims, dtype] : std::vector<std::pair<std::vector<uint64_t>, DataType>>{
             {{1, 8192, 1536}, DataType::FP16},
             {{1, 8192, 512}, DataType::FP16},
             {{1, 32768, 33}, DataType::BF16},
             {{1024, 256, 32}, DataType::FP32},
         }) {
        const DenseReductionProblem problem = valueProblem(dims, {1});
        const std::vector<DenseReductionSite> sites = enumerateDenseReductionSites(problem);
        ASSERT_EQ(sites.size(), 1U);
        const DenseReductionCandidate candidate = ReducersDenseRK::propose(problem, sites.front(), context(dtype));
        const auto plan = rkPlan(candidate);
        ASSERT_NE(plan, nullptr);
        expectModernOnePass(*plan);
    }
}

TEST(CubDenseRKFamily, UncoveredLegacyOverlapIsExplicitAndNeverPretendsToBeAThirdStrategy) {
    // Product + shallow odd BF16 K cannot use the current naturally aligned K-parallel launcher, packet-adaptive
    // R-cooperative Complete is additive-only, staged paths cannot make progress when R=1, and low-precision odd K is not packet-legal for K-parallel.
    const DenseReductionProblem problem = valueProblem({2, 1, 35}, {1}, DataType::BF16);

    EXPECT_THROW(static_cast<void>(ReducersDenseRK::propose(
                     problem, {1}, context(DataType::BF16, DenseRKValueOperation::Product))),
                 std::logic_error);

    const DenseReductionCandidate fallback = ReducersDenseRK::propose(
        problem, {1}, context(DataType::BF16, DenseRKValueOperation::Product, 128, true));
    const auto direct = directPlan(fallback);
    ASSERT_NE(direct, nullptr);
    EXPECT_EQ(direct->implementation, DenseDirectReductionImplementation::TiledFixedSegment);
    EXPECT_EQ(rkPlan(fallback), nullptr);
}
