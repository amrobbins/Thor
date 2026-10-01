#include "Utilities/TensorOperations/Cub/DenseReductionPlanner.h"

#include <gtest/gtest.h>

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

TEST(CubDenseReductionPlanner, AskAllChoosesSmallestNextAggregateEvenWhenThatIsRK) {
    // Current R=8,K=3,R=5. Eliminating left RK leaves 3*5 values; eliminating trailing KR leaves 8*3 values.
    const DenseReductionPlan plan = planDenseReduction(valueProblem({8, 3, 5}, {0, 2}), plannerContext());

    ASSERT_EQ(plan.steps.size(), 2U);
    ASSERT_EQ(plan.steps[0].considered_candidates.size(), 2U);
    EXPECT_EQ(plan.steps[0].considered_candidates[0].family, DenseReducerFamily::RK);
    EXPECT_EQ(plan.steps[0].considered_candidates[0].next_bytes, 3U * 5U * 4U);
    EXPECT_EQ(plan.steps[0].considered_candidates[1].family, DenseReducerFamily::KR);
    EXPECT_EQ(plan.steps[0].considered_candidates[1].next_bytes, 8U * 3U * 4U);
    EXPECT_EQ(plan.steps[0].selected_candidate.family, DenseReducerFamily::RK);
    EXPECT_EQ(plan.steps[0].selected_candidate.site.run_index, 0U);
}

TEST(CubDenseReductionPlanner, EqualNextBytesPreferKROverRK) {
    const DenseReductionPlan plan = planDenseReduction(valueProblem({5, 3, 5}, {0, 2}), plannerContext());

    ASSERT_FALSE(plan.steps.empty());
    ASSERT_EQ(plan.steps[0].considered_candidates.size(), 2U);
    EXPECT_EQ(plan.steps[0].considered_candidates[0].next_bytes,
              plan.steps[0].considered_candidates[1].next_bytes);
    EXPECT_EQ(plan.steps[0].selected_candidate.family, DenseReducerFamily::KR);
    EXPECT_EQ(plan.steps[0].selected_candidate.site.run_index, 2U);
}

TEST(CubDenseReductionPlanner, EqualSameFamilyCandidatesUseStableSiteOrder) {
    // R=4,K=2,R=4,K=2: both current sites are RK and produce the same next byte count.
    const DenseReductionPlan plan = planDenseReduction(valueProblem({4, 2, 4, 2}, {0, 2}), plannerContext());

    ASSERT_FALSE(plan.steps.empty());
    ASSERT_EQ(plan.steps[0].considered_candidates.size(), 2U);
    EXPECT_EQ(plan.steps[0].considered_candidates[0].family, DenseReducerFamily::RK);
    EXPECT_EQ(plan.steps[0].considered_candidates[1].family, DenseReducerFamily::RK);
    EXPECT_EQ(plan.steps[0].considered_candidates[0].next_bytes,
              plan.steps[0].considered_candidates[1].next_bytes);
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

TEST(CubDenseReductionPlanner, DiagnosticsReportEveryCandidateAndTheSelectedPass) {
    const DenseReductionPlan plan = planDenseReduction(valueProblem({8, 3, 5}, {0, 2}), plannerContext());
    const std::string diagnostics = formatDenseReductionPlanDiagnostics(plan);

    EXPECT_NE(diagnostics.find("family=RK"), std::string::npos);
    EXPECT_NE(diagnostics.find("family=KR"), std::string::npos);
    EXPECT_NE(diagnostics.find("next_bytes="), std::string::npos);
    EXPECT_NE(diagnostics.find("physical=KParallel/Aligned/Complete"), std::string::npos);
    EXPECT_NE(diagnostics.find("physical=ContiguousFixedSegment"), std::string::npos);
    EXPECT_NE(diagnostics.find("selected=1"), std::string::npos);
    EXPECT_NE(diagnostics.find("dense_plan final="), std::string::npos);
}
