#include "Utilities/TensorOperations/Cub/CubReduction.h"
#include "Utilities/TensorOperations/Cub/DenseReductionFamilies.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
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

DenseRKPlanningContext rkContext(DataType input_dtype = DataType::FP32,
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

void expectSameDirectGeometry(const DenseDirectReductionPhysicalPlan& plan, const CubReductionGeometry& geometry) {
    ASSERT_EQ(geometry.axes.size(), 1U);
    EXPECT_EQ(plan.reduction_axis, geometry.axes.front());
    EXPECT_EQ(plan.input_elements, geometry.input_elements);
    EXPECT_EQ(plan.reduction_size, geometry.reduction_size);
    EXPECT_EQ(plan.output_elements, geometry.output_elements);
    EXPECT_EQ(plan.outer_size, geometry.outer_size);
    EXPECT_EQ(plan.inner_size, geometry.inner_size);
}

}  // namespace

TEST(CubDenseReductionFamilyBase, DenseRWrapsExistingDeviceTransformReduceCapability) {
    const DenseReductionProblem problem = valueProblem({257}, {0});
    const DenseReductionCandidate candidate = ReducersDenseR::propose(problem, {0});
    const auto plan = directPlan(candidate);

    ASSERT_NE(plan, nullptr);
    EXPECT_EQ(candidate.family, DenseReducerFamily::R);
    EXPECT_EQ(candidate.site, (DenseReductionSite{0}));
    EXPECT_EQ(plan->implementation, DenseDirectReductionImplementation::DeviceTransformReduce);
    EXPECT_EQ(plan->role, DensePassRole::Complete);

    const CubReductionGeometry existing = CubReduction::analyzeGeometry(plan->input_dimensions, plan->reduction_axis);
    EXPECT_EQ(existing.path, CubReductionPath::DeviceTransformReduce);
    expectSameDirectGeometry(*plan, existing);

    EXPECT_FALSE(hasDenseReductionSites(candidate.next_problem));
    EXPECT_EQ(candidate.next_bytes, denseReductionProblemBytes(candidate.next_problem));
}

TEST(CubDenseReductionFamilyBase, DenseKRWrapsExistingContiguousFixedSegmentCapability) {
    const DenseReductionProblem problem = valueProblem({2, 3, 5}, {2});
    ASSERT_EQ(problem.runs.size(), 2U);  // K=6, R=5 after canonical run collapse.

    const DenseReductionCandidate candidate = ReducersDenseKR::propose(problem, {1});
    const auto plan = directPlan(candidate);
    ASSERT_NE(plan, nullptr);

    EXPECT_EQ(candidate.family, DenseReducerFamily::KR);
    EXPECT_EQ(plan->implementation, DenseDirectReductionImplementation::ContiguousFixedSegment);
    EXPECT_EQ(plan->role, DensePassRole::Complete);
    EXPECT_EQ(plan->input_dimensions, (std::vector<uint64_t>{6, 5}));

    const CubReductionGeometry existing = CubReduction::analyzeGeometry(plan->input_dimensions, plan->reduction_axis);
    EXPECT_EQ(existing.path, CubReductionPath::ContiguousFixedSegment);
    expectSameDirectGeometry(*plan, existing);
}

TEST(CubDenseReductionFamilyBase, DenseRKUsesOneModernFamilyImplementationForTheNominatedSite) {
    const DenseReductionProblem problem = valueProblem({2, 3, 2048}, {1});
    const DenseReductionCandidate candidate = ReducersDenseRK::propose(problem, {1}, rkContext());
    const auto plan = rkPlan(candidate);
    ASSERT_NE(plan, nullptr);

    EXPECT_EQ(candidate.family, DenseReducerFamily::RK);
    EXPECT_EQ(plan->input_dimensions, (std::vector<uint64_t>{2, 3, 2048}));
    EXPECT_EQ(plan->reduction_axis, 1U);
    EXPECT_TRUE(denseRKIsFamilyImplementation(plan->implementation));
    EXPECT_FALSE(denseRKProductionInventoryEntry(plan->implementation).owns_successor);
}

TEST(CubDenseReductionFamilyBase, DisjointProblemProposesOnlyTheNominatedSite) {
    const DenseReductionProblem problem = valueProblem({2, 3, 5}, {0, 2});  // R K R
    const auto sites = enumerateDenseReductionSites(problem);
    ASSERT_EQ(sites.size(), 2U);

    const DenseReductionCandidate left = ReducersDenseRK::propose(problem, sites[0], rkContext());
    const auto left_plan = rkPlan(left);
    ASSERT_NE(left_plan, nullptr);
    EXPECT_EQ(left_plan->reduction_axis, 0U);
    EXPECT_EQ(left_plan->role, DensePassRole::First);
    EXPECT_TRUE(denseRKIsFamilyImplementation(left_plan->implementation));
    ASSERT_EQ(left.next_problem.runs.size(), 2U);
    EXPECT_EQ(left.next_problem.runs[0], (DenseReductionRun{DenseReductionRunKind::Retained, 3, 0}));
    EXPECT_EQ(left.next_problem.runs[1].kind, DenseReductionRunKind::OriginalReduction);
    EXPECT_EQ(left.next_problem.runs[1].extent, 5U);

    const DenseReductionCandidate right = ReducersDenseKR::propose(problem, sites[1]);
    const auto right_plan = directPlan(right);
    ASSERT_NE(right_plan, nullptr);
    EXPECT_EQ(right_plan->implementation, DenseDirectReductionImplementation::ContiguousFixedSegment);
    EXPECT_EQ(right_plan->reduction_axis, 2U);
    EXPECT_EQ(right_plan->role, DensePassRole::First);
    ASSERT_EQ(right.next_problem.runs.size(), 2U);
    EXPECT_EQ(right.next_problem.runs[0].kind, DenseReductionRunKind::OriginalReduction);
    EXPECT_EQ(right.next_problem.runs[0].extent, 2U);
    EXPECT_EQ(right.next_problem.runs[1], (DenseReductionRun{DenseReductionRunKind::Retained, 3, 0}));

    // The old whole-problem analyzer quite correctly sees composition. Family proposals deliberately analyze only
    // their nominated current R and therefore expose one existing direct capability each.
    EXPECT_EQ(CubReduction::analyzeGeometry({2, 3, 5}, {0, 2}).path, CubReductionPath::ComposedDense);
}

TEST(CubDenseReductionFamilyBase, ProposalReusesCanonicalCurrentRunTensorAfterPriorPass) {
    const DenseReductionProblem original = valueProblem({2, 3, 5}, {0, 2});
    const DenseReductionProblem after_left = advanceDenseReductionProblem(original, {0}, 1);
    ASSERT_EQ(after_left.input_state, DenseReductionInputState::IntermediateAggregate);

    const DenseReductionCandidate final = ReducersDenseKR::propose(after_left, {1});
    const auto plan = directPlan(final);
    ASSERT_NE(plan, nullptr);

    EXPECT_EQ(plan->input_dimensions, (std::vector<uint64_t>{3, 5}));
    EXPECT_EQ(plan->implementation, DenseDirectReductionImplementation::ContiguousFixedSegment);
    EXPECT_EQ(plan->role, DensePassRole::Final);
    EXPECT_FALSE(hasDenseReductionSites(final.next_problem));
}

TEST(CubDenseReductionFamilyBase, ValueAndArgShareRKTopologyWhileArgStillUsesTransitionalPhysicalAdapter) {
    const DenseReductionProblem value = valueProblem({8, 3, 5}, {0, 2});
    const DenseReductionProblem arg32 = argProblem({8, 3, 5}, {0, 2}, DataType::UINT32);
    const DenseReductionProblem arg64 = argProblem({8, 3, 5}, {0, 2}, DataType::UINT64);

    const DenseReductionCandidate value_candidate = ReducersDenseRK::propose(value, {0}, rkContext());
    const DenseReductionCandidate arg32_candidate = ReducersDenseRK::propose(arg32, {0}, rkContext());
    const DenseReductionCandidate arg64_candidate = ReducersDenseRK::propose(arg64, {0}, rkContext());

    ASSERT_NE(rkPlan(value_candidate), nullptr);
    ASSERT_NE(directPlan(arg32_candidate), nullptr);
    ASSERT_NE(directPlan(arg64_candidate), nullptr);
    EXPECT_EQ(value_candidate.family, DenseReducerFamily::RK);
    EXPECT_EQ(arg32_candidate.family, DenseReducerFamily::RK);
    EXPECT_EQ(arg64_candidate.family, DenseReducerFamily::RK);

    // Eliminating the left R leaves current [K=3,R=5]. VALUE carries FP32; ARG carries FP32 plus its original index.
    EXPECT_EQ(value_candidate.next_bytes, 3U * 5U * 4U);
    EXPECT_EQ(arg32_candidate.next_bytes, 3U * 5U * (4U + 4U));
    EXPECT_EQ(arg64_candidate.next_bytes, 3U * 5U * (4U + 8U));
}

TEST(CubDenseReductionFamilyBase, FamilyAdaptersRejectSitesOwnedByAnotherTopology) {
    const DenseReductionProblem middle = valueProblem({2, 3, 5}, {1});
    EXPECT_THROW(static_cast<void>(ReducersDenseR::propose(middle, {1})), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(ReducersDenseKR::propose(middle, {1})), std::invalid_argument);
    EXPECT_NO_THROW(static_cast<void>(ReducersDenseRK::propose(middle, {1}, rkContext())));

    const DenseReductionProblem trailing = valueProblem({2, 3, 5}, {2});
    EXPECT_THROW(static_cast<void>(ReducersDenseRK::propose(trailing, {1}, rkContext())), std::invalid_argument);
    EXPECT_NO_THROW(static_cast<void>(ReducersDenseKR::propose(trailing, {1})));
}

TEST(CubDenseReductionFamilyBase, SingletonOnlySemanticPassUsesModernAdditiveRKCoverage) {
    const DenseReductionProblem problem = valueProblem({3, 1, 5, 1, 7}, {1, 3}, DataType::BF16);
    ASSERT_EQ(problem.runs.size(), 3U);

    const DenseReductionCandidate candidate =
        ReducersDenseRK::propose(problem, {1}, rkContext(DataType::BF16));
    const auto plan = rkPlan(candidate);
    ASSERT_NE(plan, nullptr);

    EXPECT_EQ(plan->input_dimensions, (std::vector<uint64_t>{3, 1, 35}));
    EXPECT_EQ(plan->role, DensePassRole::Complete);
    EXPECT_EQ(plan->strategy, DenseRKStrategy::RCooperative);
    EXPECT_EQ(plan->progress, DenseRKProgress::Complete);
    EXPECT_FALSE(hasDenseReductionSites(candidate.next_problem));
    EXPECT_EQ(candidate.next_bytes, 3U * 35U * 2U);
}
