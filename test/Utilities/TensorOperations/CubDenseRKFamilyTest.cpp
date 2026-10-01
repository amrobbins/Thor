#include "Utilities/TensorOperations/Cub/DenseReductionFamilies.h"

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

TEST(CubDenseRKFamily, KParallelOccupancyUsesPhysicalWarpsRatherThanCtaCount) {
    // Product has no packet-adaptive Complete path, and R=64 is too shallow for staged R-cooperative. K-parallel is
    // therefore the only modern strategy and exposes the physical-warp accounting used by the calibrated policy.
    const DenseReductionProblem problem = valueProblem({1, 64, 1536}, {1});
    const DenseReductionCandidate candidate = ReducersDenseRK::propose(
        problem, {0}, context(DataType::FP16, DenseRKValueOperation::Product));
    const auto plan = rkPlan(candidate);
    ASSERT_NE(plan, nullptr);

    EXPECT_EQ(plan->strategy, DenseRKStrategy::KParallel);
    EXPECT_EQ(plan->implementation, DenseRKProductionImplementation::KParallelPass);
    EXPECT_EQ(plan->progress, DenseRKProgress::Complete);
    EXPECT_EQ(plan->physical_stage_warps,
              plan->first_stage_blocks * static_cast<uint64_t>(plan->block_threads / 32U));
    EXPECT_GT(plan->max_active_blocks_per_sm, 0U);
    EXPECT_GT(plan->max_active_warps_per_sm, 0U);
    EXPECT_LE(plan->launch_warp_supply,
              128U * static_cast<uint64_t>(plan->max_active_warps_per_sm));
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
