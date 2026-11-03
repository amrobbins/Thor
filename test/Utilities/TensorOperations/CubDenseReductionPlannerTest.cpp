#include "Utilities/TensorOperations/Cub/DenseReductionPlanner.h"
#include "Utilities/TensorOperations/Cub/DenseProductionPhysicalSelectionContext.h"
#include "Utilities/Common/Stream.h"

#include <cuda_runtime_api.h>

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <string>
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

DenseReductionOutputSpec argOutput(DataType value_dtype = DataType::FP32,
                                   DataType index_dtype = DataType::UINT32) {
    return DenseReductionOutputSpec{
        .produce_value = true,
        .value_dtype = value_dtype,
        .produce_index = true,
        .index_dtype = index_dtype,
    };
}

DenseReductionProblem valueProblem(const std::vector<uint64_t>& dimensions,
                                   const std::vector<uint32_t>& axes,
                                   DataType output_dtype = DataType::FP32) {
    return makeInitialDenseReductionProblem(
        dimensions, axes, DenseReductionAggregateKind::Value, valueOutput(output_dtype));
}

DenseReductionProblem argProblem(const std::vector<uint64_t>& dimensions,
                                 const std::vector<uint32_t>& axes,
                                 DataType carried_index_dtype = DataType::UINT32) {
    return makeInitialDenseReductionProblem(
        dimensions, axes, DenseReductionAggregateKind::Arg, argOutput(), carried_index_dtype);
}

DenseReductionPlannerContext plannerContext(DataType input_dtype = DataType::FP32,
                                            DenseRKValueOperation op = DenseRKValueOperation::Sum,
                                            uint32_t multiprocessors = 128) {
    return DenseReductionPlannerContext{
        .rk = DenseRKPlanningContext{
            .original_input_dtype = input_dtype,
            .value_operation = op,
            .multiprocessors = multiprocessors,
        .max_threads_per_sm = 1536,
        .max_blocks_per_sm = 24,
            .allow_generic_rk_fallback = false,
        },
    };
}

}  // namespace

TEST(CubDenseReductionPlanner, PureRValueFastPathsDenseRAndFinishes) {
    const DenseReductionPlan plan = planDenseReduction(valueProblem({257}, {0}), plannerContext());

    ASSERT_EQ(plan.steps.size(), 1U);
    ASSERT_EQ(plan.steps[0].considered_candidates.size(), 1U);
    EXPECT_EQ(plan.steps[0].selected_candidate.family, DenseReducerFamily::R);
    EXPECT_EQ(deriveDensePassRole(plan.steps[0].input_problem, plan.steps[0].selected_candidate.next_problem),
              DensePassRole::Complete);
    EXPECT_FALSE(hasDenseReductionSites(plan.final_problem));
}

TEST(CubDenseReductionPlanner, PureRArgUsesTheSameDenseRFastPath) {
    const DenseReductionPlan plan = planDenseReduction(argProblem({257}, {0}, DataType::UINT64), plannerContext());

    ASSERT_EQ(plan.steps.size(), 1U);
    EXPECT_EQ(plan.steps[0].selected_candidate.family, DenseReducerFamily::R);
    EXPECT_EQ(plan.steps[0].selected_candidate.next_bytes, 4U + 4U);  // requested FP32 value + UINT32 index
    EXPECT_FALSE(hasDenseReductionSites(plan.final_problem));
}

TEST(CubDenseReductionPlanner, SmallestBoundFirstEvenWhenThatIsRKAndACompletePassEndsTheStep) {
    // Current R=8,K=3,R=5. Eliminating left RK leaves 3*5 values; eliminating trailing KR leaves 8*3 values.
    // 1167: RK has the smaller bound, so it is selected first; its pass completes the site, so KR is never selected.
    const DenseReductionPlan plan = planDenseReduction(valueProblem({8, 3, 5}, {0, 2}), plannerContext());

    ASSERT_EQ(plan.steps.size(), 2U);
    ASSERT_EQ(plan.steps[0].site_bounds.size(), 2U);
    EXPECT_EQ(plan.steps[0].site_bounds[0].family, DenseReducerFamily::RK);
    EXPECT_EQ(plan.steps[0].site_bounds[0].complete_next_bytes, 3U * 5U * 4U);
    EXPECT_TRUE(plan.steps[0].site_bounds[0].selected);
    EXPECT_EQ(plan.steps[0].site_bounds[1].family, DenseReducerFamily::KR);
    EXPECT_EQ(plan.steps[0].site_bounds[1].complete_next_bytes, 8U * 3U * 4U);
    EXPECT_FALSE(plan.steps[0].site_bounds[1].selected);
    ASSERT_EQ(plan.steps[0].considered_candidates.size(), 1U);
    EXPECT_EQ(plan.steps[0].considered_candidates[0].next_bytes, 3U * 5U * 4U);
    EXPECT_EQ(plan.steps[0].selected_candidate.family, DenseReducerFamily::RK);
    EXPECT_EQ(plan.steps[0].selected_candidate.site.run_index, 0U);
}

TEST(CubDenseReductionPlanner, EqualNextBytesPreferKROverRK) {
    const DenseReductionPlan plan = planDenseReduction(valueProblem({5, 3, 5}, {0, 2}), plannerContext());

    ASSERT_FALSE(plan.steps.empty());
    ASSERT_EQ(plan.steps[0].site_bounds.size(), 2U);
    EXPECT_EQ(plan.steps[0].site_bounds[0].complete_next_bytes, plan.steps[0].site_bounds[1].complete_next_bytes);
    EXPECT_EQ(plan.steps[0].site_bounds[0].family, DenseReducerFamily::KR);  // 1167: visited first on a tie
    ASSERT_EQ(plan.steps[0].considered_candidates.size(), 1U);  // it completes; the tied RK cannot win
    EXPECT_EQ(plan.steps[0].selected_candidate.family, DenseReducerFamily::KR);
    EXPECT_EQ(plan.steps[0].selected_candidate.site.run_index, 2U);
}

TEST(CubDenseReductionPlanner, EqualSameFamilyCandidatesUseStableSiteOrder) {
    // R=4,K=2,R=4,K=2: both current sites are RK and produce the same next byte count.
    const DenseReductionPlan plan = planDenseReduction(valueProblem({4, 2, 4, 2}, {0, 2}), plannerContext());

    ASSERT_FALSE(plan.steps.empty());
    ASSERT_EQ(plan.steps[0].site_bounds.size(), 2U);
    EXPECT_EQ(plan.steps[0].site_bounds[0].family, DenseReducerFamily::RK);
    EXPECT_EQ(plan.steps[0].site_bounds[1].family, DenseReducerFamily::RK);
    EXPECT_EQ(plan.steps[0].site_bounds[0].complete_next_bytes, plan.steps[0].site_bounds[1].complete_next_bytes);
    EXPECT_EQ(plan.steps[0].site_bounds[0].site.run_index, 0U);  // 1167: the outer site is visited first on a tie
    ASSERT_EQ(plan.steps[0].considered_candidates.size(), 1U);
    EXPECT_EQ(plan.steps[0].selected_candidate.site.run_index, 0U);
}

TEST(CubDenseReductionPlanner, AdjacentOriginalReductionAxesNormalizeBeforeFamilySelection) {
    // K R R K canonicalizes to K R K, therefore there is one RK site rather than competing KR/RK sites.
    const DenseReductionPlan plan = planDenseReduction(valueProblem({2, 3, 5, 7}, {1, 2}), plannerContext());

    ASSERT_EQ(plan.steps.size(), 1U);
    ASSERT_EQ(plan.steps[0].input_problem.runs.size(), 3U);
    EXPECT_EQ(plan.steps[0].input_problem.runs[0].extent, 2U);
    EXPECT_EQ(plan.steps[0].input_problem.runs[1].extent, 15U);
    EXPECT_EQ(plan.steps[0].input_problem.runs[2].extent, 7U);
    EXPECT_EQ(plan.steps[0].selected_candidate.family, DenseReducerFamily::RK);
}

TEST(CubDenseReductionPlanner, SelectedNextProblemIsReplannedFresh) {
    const DenseReductionPlan plan = planDenseReduction(valueProblem({8, 3, 5}, {0, 2}), plannerContext());

    ASSERT_EQ(plan.steps.size(), 2U);
    EXPECT_EQ(plan.steps[0].selected_candidate.family, DenseReducerFamily::RK);
    EXPECT_EQ(plan.steps[1].input_problem, plan.steps[0].selected_candidate.next_problem);
    EXPECT_EQ(plan.steps[1].selected_candidate.family, DenseReducerFamily::KR);
    EXPECT_EQ(deriveDensePassRole(plan.steps[0].input_problem, plan.steps[0].selected_candidate.next_problem),
              DensePassRole::First);
    EXPECT_EQ(deriveDensePassRole(plan.steps[1].input_problem, plan.steps[1].selected_candidate.next_problem),
              DensePassRole::Final);
}

TEST(CubDenseReductionPlanner, ArgPayloadWidthParticipatesInNextBytesWithoutChangingTopology) {
    const DenseReductionPlan value = planDenseReduction(valueProblem({8, 3, 5}, {0, 2}), plannerContext());
    const DenseReductionPlan arg32 = planDenseReduction(argProblem({8, 3, 5}, {0, 2}, DataType::UINT32), plannerContext());
    const DenseReductionPlan arg64 = planDenseReduction(argProblem({8, 3, 5}, {0, 2}, DataType::UINT64), plannerContext());

    ASSERT_FALSE(value.steps.empty());
    ASSERT_FALSE(arg32.steps.empty());
    ASSERT_FALSE(arg64.steps.empty());
    EXPECT_EQ(value.steps[0].selected_candidate.family, arg32.steps[0].selected_candidate.family);
    EXPECT_EQ(value.steps[0].selected_candidate.family, arg64.steps[0].selected_candidate.family);
    EXPECT_EQ(value.steps[0].selected_candidate.site, arg32.steps[0].selected_candidate.site);
    EXPECT_EQ(value.steps[0].selected_candidate.site, arg64.steps[0].selected_candidate.site);

    EXPECT_EQ(value.steps[0].selected_candidate.next_bytes, 3U * 5U * 4U);
    EXPECT_EQ(arg32.steps[0].selected_candidate.next_bytes, 3U * 5U * (4U + 4U));
    EXPECT_EQ(arg64.steps[0].selected_candidate.next_bytes, 3U * 5U * (4U + 8U));
}

TEST(CubDenseReductionPlanner, SingletonSemanticReductionStillPlansExactlyOnePass) {
    const DenseReductionPlan plan = planDenseReduction(valueProblem({3, 1, 5, 1, 7}, {1, 3}, DataType::BF16), plannerContext(DataType::BF16));

    ASSERT_EQ(plan.steps.size(), 1U);
    EXPECT_EQ(plan.steps[0].selected_candidate.family, DenseReducerFamily::RK);
    EXPECT_EQ(deriveDensePassRole(plan.steps[0].input_problem, plan.steps[0].selected_candidate.next_problem),
              DensePassRole::Complete);
    EXPECT_FALSE(hasDenseReductionSites(plan.final_problem));
}

TEST(CubDenseReductionPlanner, DiagnosticsReportEverySiteBoundAndEverySelectedCandidate) {
    const DenseReductionPlan plan = planDenseReduction(valueProblem({8, 3, 5}, {0, 2}), plannerContext());
    const std::string diagnostics = formatDenseReductionPlanDiagnostics(plan);

    EXPECT_NE(diagnostics.find("family=RK"), std::string::npos);
    EXPECT_NE(diagnostics.find("family=KR"), std::string::npos);
    EXPECT_NE(diagnostics.find("next_bytes="), std::string::npos);
    EXPECT_NE(diagnostics.find("physical=KParallel/Aligned/Complete"), std::string::npos);
    EXPECT_NE(diagnostics.find("physical=ContiguousFixedSegment"), std::string::npos);
    EXPECT_NE(diagnostics.find("selected=1"), std::string::npos);
    EXPECT_NE(diagnostics.find("bound site=2 family=KR complete_next_bytes=96 selected=0"), std::string::npos);
    EXPECT_NE(diagnostics.find("dense_plan final="), std::string::npos);
}

namespace {

void expectSamePhysicalSelection(const DenseReductionCandidate& actual, const DenseReductionCandidate& expected) {
    EXPECT_EQ(actual.family, expected.family);
    EXPECT_EQ(actual.site, expected.site);
    EXPECT_EQ(actual.next_problem, expected.next_problem);
    EXPECT_EQ(actual.next_bytes, expected.next_bytes);
    EXPECT_EQ(actual.scratch_bytes, expected.scratch_bytes);
    ASSERT_NE(actual.physical_plan, nullptr);
    ASSERT_NE(expected.physical_plan, nullptr);

    const auto actual_direct = std::dynamic_pointer_cast<const DenseDirectReductionPhysicalPlan>(actual.physical_plan);
    const auto expected_direct = std::dynamic_pointer_cast<const DenseDirectReductionPhysicalPlan>(expected.physical_plan);
    EXPECT_EQ(static_cast<bool>(actual_direct), static_cast<bool>(expected_direct));
    if (actual_direct && expected_direct) {
        EXPECT_EQ(actual_direct->implementation, expected_direct->implementation);
        EXPECT_EQ(actual_direct->role, expected_direct->role);
        EXPECT_EQ(actual_direct->input_dimensions, expected_direct->input_dimensions);
        EXPECT_EQ(actual_direct->reduction_axis, expected_direct->reduction_axis);
        EXPECT_EQ(actual_direct->input_elements, expected_direct->input_elements);
        EXPECT_EQ(actual_direct->reduction_size, expected_direct->reduction_size);
        EXPECT_EQ(actual_direct->output_elements, expected_direct->output_elements);
        EXPECT_EQ(actual_direct->outer_size, expected_direct->outer_size);
        EXPECT_EQ(actual_direct->inner_size, expected_direct->inner_size);
        return;
    }
    const auto actual_rk = std::dynamic_pointer_cast<const DenseRKFamilyPhysicalPlan>(actual.physical_plan);
    const auto expected_rk = std::dynamic_pointer_cast<const DenseRKFamilyPhysicalPlan>(expected.physical_plan);
    ASSERT_TRUE(actual_rk);
    ASSERT_TRUE(expected_rk);
    EXPECT_EQ(actual_rk->strategy, expected_rk->strategy);
    EXPECT_EQ(actual_rk->access, expected_rk->access);
    EXPECT_EQ(actual_rk->implementation, expected_rk->implementation);
    EXPECT_EQ(actual_rk->progress, expected_rk->progress);
    EXPECT_EQ(actual_rk->role, expected_rk->role);
    EXPECT_EQ(actual_rk->input_dimensions, expected_rk->input_dimensions);
    EXPECT_EQ(actual_rk->reduction_axis, expected_rk->reduction_axis);
    EXPECT_EQ(actual_rk->input_dtype, expected_rk->input_dtype);
    EXPECT_EQ(actual_rk->value_operation, expected_rk->value_operation);
    EXPECT_EQ(actual_rk->outer_size, expected_rk->outer_size);
    EXPECT_EQ(actual_rk->reduction_size, expected_rk->reduction_size);
    EXPECT_EQ(actual_rk->inner_size, expected_rk->inner_size);
    EXPECT_EQ(actual_rk->packet_bytes, expected_rk->packet_bytes);
    EXPECT_EQ(actual_rk->block_threads, expected_rk->block_threads);
    EXPECT_EQ(actual_rk->cooperative_warps_per_tile, expected_rk->cooperative_warps_per_tile);
    EXPECT_EQ(actual_rk->rows_per_shard, expected_rk->rows_per_shard);
    EXPECT_EQ(actual_rk->shards_per_output, expected_rk->shards_per_output);
    EXPECT_EQ(actual_rk->component_tiles, expected_rk->component_tiles);
    EXPECT_EQ(actual_rk->first_stage_blocks, expected_rk->first_stage_blocks);
    EXPECT_EQ(actual_rk->useful_stage_warps, expected_rk->useful_stage_warps);
    EXPECT_EQ(actual_rk->max_active_blocks_per_sm, expected_rk->max_active_blocks_per_sm);
    EXPECT_EQ(actual_rk->max_active_warps_per_sm, expected_rk->max_active_warps_per_sm);
    EXPECT_EQ(actual_rk->physical_stage_warps, expected_rk->physical_stage_warps);
    EXPECT_EQ(actual_rk->resident_device_warps, expected_rk->resident_device_warps);
    EXPECT_EQ(actual_rk->useful_resident_warp_capacity, expected_rk->useful_resident_warp_capacity);
    EXPECT_EQ(actual_rk->launch_warp_supply, expected_rk->launch_warp_supply);
    EXPECT_EQ(actual_rk->wave_capacity_blocks, expected_rk->wave_capacity_blocks);
}

DenseReductionCandidate directlyPropose(const DenseReductionProblem& problem,
                                         DenseReductionSite site,
                                         const DensePhysicalSelectionContext& context) {
    switch (classifyDenseReducerFamily(problem, site)) {
        case DenseReducerFamily::R: return ReducersDenseR::propose(problem, site);
        case DenseReducerFamily::KR: return ReducersDenseKR::propose(problem, site);
        case DenseReducerFamily::RK: return ReducersDenseRK::propose(problem, site, context.rk);
    }
    throw std::logic_error("Invalid family in parity fixture.");
}

struct CountingOccupancy {
    uint32_t calls = 0;
};

DenseRKOccupancyInfo countingOccupancy(const DenseRKFamilyPhysicalPlan& plan, void* data) {
    auto& state = *static_cast<CountingOccupancy*>(data);
    ++state.calls;
    return {static_cast<uint32_t>(plan.block_threads <= 128 ? 8 : 4), 32};
}

struct OverrideCalls {
    std::vector<DenseReducerFamily> families;
};

DenseReductionCandidate recordedProposal(const DenseReductionProblem& problem,
                                         DenseReductionSite site,
                                         DenseReducerFamily family,
                                         const DenseRKPlanningContext& rk,
                                         void* data) {
    auto& calls = *static_cast<OverrideCalls*>(data);
    calls.families.push_back(family);
    switch (family) {
        case DenseReducerFamily::R: return ReducersDenseR::propose(problem, site);
        case DenseReducerFamily::KR: return ReducersDenseKR::propose(problem, site);
        case DenseReducerFamily::RK: return ReducersDenseRK::propose(problem, site, rk);
    }
    throw std::logic_error("Invalid injected family.");
}

}  // namespace

TEST(CubDensePhysicalSelector, DescriptorOnlyFP32MatchesEachOriginalProductionFamily) {
    const DensePhysicalSelectionContext context = plannerContext();
    for (const auto& problem : {
             valueProblem({257}, {0}),
             valueProblem({9, 257}, {1}),
             valueProblem({8, 257, 64}, {1}),
             valueProblem({2, 65536, 4}, {1}),
             valueProblem({8, 3, 5}, {0, 2}),
             valueProblem({8, 257, 64}, {1}, DataType::BF16),
             argProblem({4, 257, 8}, {1}, DataType::UINT64),
         }) {
        for (DenseReductionSite site : enumerateDenseReductionSites(problem)) {
            const auto selected = selectDensePhysicalPass(problem, site, context);
            const auto direct = directlyPropose(problem, site, context);
            expectSamePhysicalSelection(selected, direct);
        }
    }
}

TEST(CubDensePhysicalSelector, StagedFP32UsesActualPhysicalOutputNotCompletedDimension) {
    DensePhysicalSelectionContext context = plannerContext();
    // Exercise difficult deep-R/small-K geometries with the unchanged calibrated family policy.
    bool found_staged = false;
    for (uint64_t r : {4096ULL, 65536ULL, 262144ULL}) {
        for (uint64_t k : {2ULL, 4ULL, 8ULL}) {
            const auto problem = valueProblem({2, r, k}, {1});
            const auto selected = selectDensePhysicalPass(problem, DenseReductionSite{1}, context);
            expectSamePhysicalSelection(selected, directlyPropose(problem, DenseReductionSite{1}, context));
            const auto physical = std::dynamic_pointer_cast<const DenseRKFamilyPhysicalPlan>(selected.physical_plan);
            if (physical && physical->progress == DenseRKProgress::Staged) {
                found_staged = true;
                EXPECT_TRUE(hasDenseReductionSites(selected.next_problem));
                EXPECT_EQ(selected.next_problem.input_state, DenseReductionInputState::IntermediateAggregate);
                EXPECT_EQ(selected.next_bytes, denseReductionProblemBytes(selected.next_problem));
                // P1-C will follow this exact descriptor-only FP32 state. The shared selector works without
                // allocating an intermediate Tensor or constructing a second production plan.
                for (DenseReductionSite next_site : enumerateDenseReductionSites(selected.next_problem)) {
                    const auto fp32 = selectDensePhysicalPass(selected.next_problem, next_site, context);
                    expectSamePhysicalSelection(fp32, directlyPropose(selected.next_problem, next_site, context));
                    if (const auto fp32_rk =
                            std::dynamic_pointer_cast<const DenseRKFamilyPhysicalPlan>(fp32.physical_plan)) {
                        EXPECT_EQ(fp32_rk->input_dtype, DataType::FP32);
                    }
                }
            }
        }
    }
    EXPECT_TRUE(found_staged) << "Fixture should cover at least one genuine staged FP32 RK selection.";
}

TEST(CubDensePhysicalSelector, ProductionOccupancyCallbackRetainsPerRequestUserState) {
    CountingOccupancy state;
    DensePhysicalSelectionContext context = plannerContext();
    context.rk.occupancy_query = countingOccupancy;
    context.rk.occupancy_query_context = &state;
    const auto problem = valueProblem({8, 4096, 32}, {1});
    const auto selected = selectDensePhysicalPass(problem, DenseReductionSite{1}, context);
    ASSERT_GT(state.calls, 0U);
    expectSamePhysicalSelection(selected, directlyPropose(problem, DenseReductionSite{1}, context));
    EXPECT_GT(state.calls, 1U);
}

TEST(CubDensePhysicalSelector, ProductionPlannerCallsSharedSelectorIncludingPureR) {
    OverrideCalls calls;
    DensePhysicalSelectionContext context = plannerContext();
    context.proposal_override = recordedProposal;
    context.proposal_override_context = &calls;
    const auto one = planDenseReduction(valueProblem({257}, {0}), context);
    ASSERT_EQ(one.steps.size(), 1U);
    ASSERT_EQ(calls.families.size(), 1U);
    EXPECT_EQ(calls.families[0], DenseReducerFamily::R);

    calls.families.clear();
    const auto two = planDenseReduction(valueProblem({8, 3, 5}, {0, 2}), context);
    ASSERT_EQ(two.steps.size(), 2U);
    // 1167: one selection per step: step 0 selects only the RK site (its complete pass beats KR's bound), step 1 the
    // remaining KR site.
    ASSERT_EQ(calls.families.size(), 2U);
    EXPECT_EQ(calls.families[0], DenseReducerFamily::RK);
    EXPECT_EQ(calls.families[1], DenseReducerFamily::KR);
    EXPECT_EQ(two.steps[0].selected_candidate.family, DenseReducerFamily::RK);
    EXPECT_EQ(two.steps[1].selected_candidate.family, DenseReducerFamily::KR);
    for (const auto& step : two.steps) {
        for (const auto& candidate : step.considered_candidates) {
            expectSamePhysicalSelection(candidate, directlyPropose(step.input_problem, candidate.site, context));
        }
    }
}

TEST(CubDensePhysicalSelector, ProductionDeviceContextPreservesExactOccupancyAndPolicy) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0) {
        GTEST_SKIP() << "Exact production occupancy requires a CUDA GPU.";
    }
    Stream stream(0);
    const DenseProductionPhysicalSelectionContext production_context(
        DataType::FP32, DenseRKValueOperation::Sum, stream);
    const auto& context = production_context.context();
    EXPECT_EQ(context.tuning_stream, &stream);
    EXPECT_EQ(context.gpu_ordinal, static_cast<uint32_t>(stream.getGpuNum()));
    EXPECT_TRUE(context.rk.allow_generic_rk_fallback);
    ASSERT_NE(context.rk.occupancy_query, nullptr);
    ASSERT_NE(context.rk.occupancy_query_context, nullptr);
    const auto problem = valueProblem({8, 4096, 32}, {1});
    const DenseReductionSite site{1};
    const auto actual = selectDensePhysicalPass(problem, site, context);
    const auto expected = directlyPropose(problem, site, context);
    expectSamePhysicalSelection(actual, expected);
    const auto plan = planDenseReduction(problem, context);
    ASSERT_FALSE(plan.steps.empty());
    expectSamePhysicalSelection(plan.steps[0].selected_candidate, expected);
}
