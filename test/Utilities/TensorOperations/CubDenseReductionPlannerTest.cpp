#include "Utilities/TensorOperations/Cub/DenseReductionPhysicalSelector.h"
#include "Utilities/TensorOperations/Cub/DenseReductionPlanner.h"
#include "Utilities/TensorOperations/Cub/CubReduction.h"
#include "Utilities/TensorOperations/Cub/DenseProductionPhysicalSelectionContext.h"
#include "Utilities/TensorOperations/Cub/DenseRetainedFp8RKPhysicalPlan.h"
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
        },
    };
}

// 1258: the injected fixture's physical pass (the planner only carries it).
struct CompletePhysicalPlan final : DenseReductionPhysicalPlan {};

// 1258: every proposal is one Complete pass over the nominated site (it was the legacy ordinary families, gone in
// phase 10; every test here planned Complete passes with them).
DenseReductionCandidate completeProposal(const DenseReductionProblem& problem,
                                         DenseReductionSite site,
                                         DenseReducerFamily family,
                                         const DenseRKPlanningContext&,
                                         void*) {
    DenseReductionCandidate candidate;
    candidate.family = family;
    candidate.site = site;
    candidate.next_problem = advanceDenseReductionProblem(problem, site, 1);
    candidate.next_bytes = denseReductionProblemBytes(candidate.next_problem);
    candidate.physical_plan = std::make_shared<CompletePhysicalPlan>();
    return candidate;
}

// 1227 / 1258: the planner's own logic (bounds, ordering, replanning, diagnostics) on a deterministic host-only
// fixture: Complete proposals injected through proposal_override. Production sends every site to a retained selector,
// which times candidates on a GPU; the CubDensePhysicalSelector tests below cover that routing.
DenseReductionPlannerContext ordinaryPlannerContext() {
    DenseReductionPlannerContext context = plannerContext();
    context.proposal_override = completeProposal;
    return context;
}

}  // namespace

TEST(CubDenseReductionPlanner, PureRValueFastPathsDenseRAndFinishes) {
    const DenseReductionPlan plan = planDenseReduction(valueProblem({257}, {0}), ordinaryPlannerContext());

    ASSERT_EQ(plan.steps.size(), 1U);
    ASSERT_EQ(plan.steps[0].considered_candidates.size(), 1U);
    EXPECT_EQ(plan.steps[0].selected_candidate.family, DenseReducerFamily::R);
    EXPECT_EQ(deriveDensePassRole(plan.steps[0].input_problem, plan.steps[0].selected_candidate.next_problem),
              DensePassRole::Complete);
    EXPECT_FALSE(hasDenseReductionSites(plan.final_problem));
}

TEST(CubDenseReductionPlanner, PureRArgUsesTheSameDenseRFastPath) {
    const DenseReductionPlan plan =
        planDenseReduction(argProblem({257}, {0}, DataType::UINT64), ordinaryPlannerContext());

    ASSERT_EQ(plan.steps.size(), 1U);
    EXPECT_EQ(plan.steps[0].selected_candidate.family, DenseReducerFamily::R);
    EXPECT_EQ(plan.steps[0].selected_candidate.next_bytes, 4U + 4U);  // requested FP32 value + UINT32 index
    EXPECT_FALSE(hasDenseReductionSites(plan.final_problem));
}

TEST(CubDenseReductionPlanner, SmallestBoundFirstEvenWhenThatIsRKAndACompletePassEndsTheStep) {
    // Current R=8,K=3,R=5. Eliminating left RK leaves 3*5 values; eliminating trailing KR leaves 8*3 values.
    // 1167: RK has the smaller bound, so it is selected first; its pass completes the site, so KR is never selected.
    const DenseReductionPlan plan = planDenseReduction(valueProblem({8, 3, 5}, {0, 2}), ordinaryPlannerContext());

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
    const DenseReductionPlan plan = planDenseReduction(valueProblem({5, 3, 5}, {0, 2}), ordinaryPlannerContext());

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
    const DenseReductionPlan plan = planDenseReduction(valueProblem({4, 2, 4, 2}, {0, 2}), ordinaryPlannerContext());

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
    const DenseReductionPlan plan = planDenseReduction(valueProblem({2, 3, 5, 7}, {1, 2}), ordinaryPlannerContext());

    ASSERT_EQ(plan.steps.size(), 1U);
    ASSERT_EQ(plan.steps[0].input_problem.runs.size(), 3U);
    EXPECT_EQ(plan.steps[0].input_problem.runs[0].extent, 2U);
    EXPECT_EQ(plan.steps[0].input_problem.runs[1].extent, 15U);
    EXPECT_EQ(plan.steps[0].input_problem.runs[2].extent, 7U);
    EXPECT_EQ(plan.steps[0].selected_candidate.family, DenseReducerFamily::RK);
}

TEST(CubDenseReductionPlanner, SelectedNextProblemIsReplannedFresh) {
    const DenseReductionPlan plan = planDenseReduction(valueProblem({8, 3, 5}, {0, 2}), ordinaryPlannerContext());

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
    const DenseReductionPlan value = planDenseReduction(valueProblem({8, 3, 5}, {0, 2}), ordinaryPlannerContext());
    const DenseReductionPlan arg32 =
        planDenseReduction(argProblem({8, 3, 5}, {0, 2}, DataType::UINT32), ordinaryPlannerContext());
    const DenseReductionPlan arg64 =
        planDenseReduction(argProblem({8, 3, 5}, {0, 2}, DataType::UINT64), ordinaryPlannerContext());

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
    // 1257: on the ordinary fixture (every real dtype's original sites go to retained selectors that time candidates).
    const DenseReductionPlan plan =
        planDenseReduction(valueProblem({3, 1, 5, 1, 7}, {1, 3}, DataType::BF16), ordinaryPlannerContext());

    ASSERT_EQ(plan.steps.size(), 1U);
    EXPECT_EQ(plan.steps[0].selected_candidate.family, DenseReducerFamily::RK);
    EXPECT_EQ(deriveDensePassRole(plan.steps[0].input_problem, plan.steps[0].selected_candidate.next_problem),
              DensePassRole::Complete);
    EXPECT_FALSE(hasDenseReductionSites(plan.final_problem));
}

TEST(CubDenseReductionPlanner, DiagnosticsReportEverySiteBoundAndEverySelectedCandidate) {
    const DenseReductionPlan plan = planDenseReduction(valueProblem({8, 3, 5}, {0, 2}), ordinaryPlannerContext());
    const std::string diagnostics = formatDenseReductionPlanDiagnostics(plan);

    EXPECT_NE(diagnostics.find("family=RK"), std::string::npos);
    EXPECT_NE(diagnostics.find("family=KR"), std::string::npos);
    EXPECT_NE(diagnostics.find("next_bytes="), std::string::npos);
    EXPECT_NE(diagnostics.find("physical=family-local"), std::string::npos);  // 1258: the injected plan
    EXPECT_NE(diagnostics.find("selected=1"), std::string::npos);
    EXPECT_NE(diagnostics.find("bound site=2 family=KR complete_next_bytes=96 selected=0"), std::string::npos);
    EXPECT_NE(diagnostics.find("dense_plan final="), std::string::npos);
}

namespace {

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
    return completeProposal(problem, site, family, rk, nullptr);
}

}  // namespace

namespace {

std::vector<DenseReductionProblem> originalSiteFixtures() {
    return {
        valueProblem({257}, {0}),
        valueProblem({9, 257}, {1}),
        valueProblem({8, 257, 64}, {1}),
        valueProblem({2, 65536, 4}, {1}),
        valueProblem({8, 3, 5}, {0, 2}),
        valueProblem({8, 257, 64}, {1}, DataType::BF16),
        argProblem({4, 257, 8}, {1}, DataType::UINT64),
    };
}

}  // namespace

// 1257: was DescriptorOnlyLowPrecisionMatchesEachOriginalProductionFamily; original FP16 / BF16 sites go to the
// retained FP16 / BF16 selectors, which time candidates on a GPU, so a host-only context is rejected, not planned.
TEST(CubDensePhysicalSelector, DescriptorOnlyF16SitesGoToTheRetainedF16Selectors) {
    for (const DataType dtype : {DataType::FP16, DataType::BF16}) {
        const DensePhysicalSelectionContext context = plannerContext(dtype);
        for (const auto& problem : originalSiteFixtures()) {
            for (DenseReductionSite site : enumerateDenseReductionSites(problem)) {
                EXPECT_THROW(static_cast<void>(selectDensePhysicalPass(problem, site, context)), std::logic_error);
            }
        }
    }
}

// 1227: every FP32 site goes to the retained FP32 selectors, never the ordinary families: they time candidates, so a
// host-only context (no tuning stream, operation or original size) is rejected rather than planned.
TEST(CubDensePhysicalSelector, DescriptorOnlyFP32SitesGoToTheRetainedFp32Selectors) {
    const DensePhysicalSelectionContext context = plannerContext();
    for (const auto& problem : originalSiteFixtures()) {
        for (DenseReductionSite site : enumerateDenseReductionSites(problem)) {
            EXPECT_THROW(static_cast<void>(selectDensePhysicalPass(problem, site, context)), std::logic_error);
        }
    }
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
            EXPECT_NE(std::dynamic_pointer_cast<const CompletePhysicalPlan>(candidate.physical_plan), nullptr);
        }
    }
}

TEST(CubDensePhysicalSelector, ProductionDeviceContextSelectsTheRetainedF16Pass) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0) {
        GTEST_SKIP() << "Retained selection requires a CUDA GPU.";
    }
    Stream stream(0);
    // 1257: BF16, with the original semantics the retained FP16 / BF16 selector and the FP32 continuation of a staged
    // pass need. Its RK site runs a retained 2-byte first stage.
    const DenseProductionPhysicalSelectionContext production_context(
        DataType::BF16, DenseRKValueOperation::Sum, stream, 4096, 1.0f, CubReductionOp::Sum);
    const auto& context = production_context.context();
    EXPECT_EQ(context.tuning_stream, &stream);
    EXPECT_EQ(context.gpu_ordinal, static_cast<uint32_t>(stream.getGpuNum()));
    EXPECT_GT(context.rk.multiprocessors, 0U);
    const auto problem = valueProblem({8, 4096, 32}, {1});
    const DenseReductionSite site{1};
    const auto actual = selectDensePhysicalPass(problem, site, context);
    const auto retained = std::dynamic_pointer_cast<const DenseRetainedFp8RKPhysicalPlan>(actual.physical_plan);
    ASSERT_NE(retained, nullptr);
    EXPECT_EQ(retained->input_dtype, DataType::BF16);
    EXPECT_EQ(retained->kernel.element_bytes, 2U);
    const auto plan = planDenseReduction(problem, context);
    ASSERT_FALSE(plan.steps.empty());
    const auto first =
        std::dynamic_pointer_cast<const DenseRetainedFp8RKPhysicalPlan>(plan.steps[0].selected_candidate.physical_plan);
    ASSERT_NE(first, nullptr);
    EXPECT_EQ(first->kernel, retained->kernel);  // the same cached selection
}
