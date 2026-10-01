#include "Utilities/TensorOperations/Cub/DenseReductionPlanning.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <optional>
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

DenseReductionOutputSpec argOutput(bool produce_value = true,
                                   bool produce_index = true,
                                   DataType value_dtype = DataType::FP32,
                                   DataType index_dtype = DataType::UINT32) {
    return DenseReductionOutputSpec{
        .produce_value = produce_value,
        .value_dtype = value_dtype,
        .produce_index = produce_index,
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
                                 DataType carried_index_dtype = DataType::UINT32,
                                 const DenseReductionOutputSpec& output = argOutput()) {
    return makeInitialDenseReductionProblem(
        dimensions, axes, DenseReductionAggregateKind::Arg, output, carried_index_dtype);
}

}  // namespace

TEST(CubDenseReductionPlanning, CanonicalInitialRunsCollapseSingletonsAndPreserveOriginalArgDomainStrides) {
    // R K R K R with singleton dimensions interspersed. The flattened original reduction domain is 2*5*11.
    const DenseReductionProblem problem = argProblem({2, 1, 3, 5, 1, 7, 11}, {0, 1, 3, 4, 6});

    ASSERT_EQ(problem.runs.size(), 5U);
    EXPECT_EQ(problem.runs[0], (DenseReductionRun{DenseReductionRunKind::OriginalReduction, 2, 55}));
    EXPECT_EQ(problem.runs[1], (DenseReductionRun{DenseReductionRunKind::Retained, 3, 0}));
    EXPECT_EQ(problem.runs[2], (DenseReductionRun{DenseReductionRunKind::OriginalReduction, 5, 11}));
    EXPECT_EQ(problem.runs[3], (DenseReductionRun{DenseReductionRunKind::Retained, 7, 0}));
    EXPECT_EQ(problem.runs[4], (DenseReductionRun{DenseReductionRunKind::OriginalReduction, 11, 1}));
    EXPECT_EQ(problem.input_state, DenseReductionInputState::OriginalInput);
    EXPECT_TRUE(problem.finalizer_pending);
}

TEST(CubDenseReductionPlanning, FamilyClassificationIsForTheNominatedCurrentSiteOnly) {
    const DenseReductionProblem scalar = valueProblem({257}, {0});
    ASSERT_EQ(enumerateDenseReductionSites(scalar).size(), 1U);
    EXPECT_EQ(classifyDenseReducerFamily(scalar, {0}), DenseReducerFamily::R);

    const DenseReductionProblem trailing = valueProblem({2, 3, 5}, {2});
    EXPECT_EQ(classifyDenseReducerFamily(trailing, {1}), DenseReducerFamily::KR);

    const DenseReductionProblem middle = valueProblem({2, 3, 5}, {1});
    EXPECT_EQ(classifyDenseReducerFamily(middle, {1}), DenseReducerFamily::RK);

    const DenseReductionProblem disjoint = valueProblem({2, 3, 5}, {0, 2});  // R K R
    const auto sites = enumerateDenseReductionSites(disjoint);
    ASSERT_EQ(sites.size(), 2U);
    EXPECT_EQ(classifyDenseReducerFamily(disjoint, sites[0]), DenseReducerFamily::RK);
    EXPECT_EQ(classifyDenseReducerFamily(disjoint, sites[1]), DenseReducerFamily::KR);
}

TEST(CubDenseReductionPlanning, PartialRRemainsInFreshProblemAndExtentOneEliminatesIt) {
    const DenseReductionProblem original = valueProblem({2, 104832, 512}, {1});
    ASSERT_EQ(original.runs.size(), 3U);

    const DenseReductionProblem partial = advanceDenseReductionProblem(original, {1}, 32);
    ASSERT_EQ(partial.runs.size(), 3U);
    EXPECT_EQ(partial.runs[0], (DenseReductionRun{DenseReductionRunKind::Retained, 2, 0}));
    EXPECT_EQ(partial.runs[1], (DenseReductionRun{DenseReductionRunKind::IntermediateReduction, 32, 0}));
    EXPECT_EQ(partial.runs[2], (DenseReductionRun{DenseReductionRunKind::Retained, 512, 0}));
    EXPECT_EQ(partial.input_state, DenseReductionInputState::IntermediateAggregate);
    EXPECT_TRUE(partial.finalizer_pending);
    EXPECT_EQ(deriveDensePassRole(original, partial), DensePassRole::First);

    const DenseReductionProblem finished = advanceDenseReductionProblem(partial, {1}, 1);
    ASSERT_EQ(finished.runs.size(), 1U);
    EXPECT_EQ(finished.runs[0], (DenseReductionRun{DenseReductionRunKind::Retained, 1024, 0}));
    EXPECT_FALSE(hasDenseReductionSites(finished));
    EXPECT_FALSE(finished.finalizer_pending);
    EXPECT_EQ(deriveDensePassRole(partial, finished), DensePassRole::Final);
}

TEST(CubDenseReductionPlanning, EliminatingOneSiteReclassifiesTheFreshRemainingProblem) {
    const DenseReductionProblem original = valueProblem({2, 3, 5}, {0, 2});  // R K R
    const DenseReductionProblem after_left = advanceDenseReductionProblem(original, {0}, 1);

    ASSERT_EQ(after_left.runs.size(), 2U);
    EXPECT_EQ(after_left.runs[0].kind, DenseReductionRunKind::Retained);
    EXPECT_EQ(after_left.runs[0].extent, 3U);
    EXPECT_EQ(after_left.runs[1].kind, DenseReductionRunKind::OriginalReduction);
    EXPECT_EQ(after_left.runs[1].extent, 5U);
    EXPECT_EQ(after_left.input_state, DenseReductionInputState::IntermediateAggregate);
    EXPECT_TRUE(after_left.finalizer_pending);

    const auto remaining_sites = enumerateDenseReductionSites(after_left);
    ASSERT_EQ(remaining_sites.size(), 1U);
    EXPECT_EQ(classifyDenseReducerFamily(after_left, remaining_sites.front()), DenseReducerFamily::KR);
}

TEST(CubDenseReductionPlanning, PassRoleIsDerivedOnlyFromInputStateAndWhetherReductionRemains) {
    const DenseReductionProblem one_pass = valueProblem({17}, {0});
    const DenseReductionProblem complete = advanceDenseReductionProblem(one_pass, {0}, 1);
    EXPECT_EQ(deriveDensePassRole(one_pass, complete), DensePassRole::Complete);

    const DenseReductionProblem multi = valueProblem({2, 64, 3}, {1});
    const DenseReductionProblem first = advanceDenseReductionProblem(multi, {1}, 8);
    EXPECT_EQ(deriveDensePassRole(multi, first), DensePassRole::First);

    const DenseReductionProblem intermediate = advanceDenseReductionProblem(first, {1}, 2);
    EXPECT_EQ(deriveDensePassRole(first, intermediate), DensePassRole::Intermediate);

    const DenseReductionProblem final = advanceDenseReductionProblem(intermediate, {1}, 1);
    EXPECT_EQ(deriveDensePassRole(intermediate, final), DensePassRole::Final);
}

TEST(CubDenseReductionPlanning, AllSingletonReductionKeepsOneSemanticSiteUntilTransformAndFinalizerRun) {
    const DenseReductionProblem original = valueProblem({3, 1, 5, 1, 7}, {1, 3}, DataType::BF16);
    ASSERT_EQ(original.runs.size(), 3U);
    EXPECT_EQ(original.runs[0], (DenseReductionRun{DenseReductionRunKind::Retained, 3, 0}));
    EXPECT_EQ(original.runs[1], (DenseReductionRun{DenseReductionRunKind::OriginalReduction, 1, 1}));
    EXPECT_EQ(original.runs[2], (DenseReductionRun{DenseReductionRunKind::Retained, 35, 0}));
    EXPECT_EQ(classifyDenseReducerFamily(original, {1}), DenseReducerFamily::RK);

    const DenseReductionProblem complete = advanceDenseReductionProblem(original, {1}, 1);
    ASSERT_EQ(complete.runs.size(), 1U);
    EXPECT_EQ(complete.runs[0], (DenseReductionRun{DenseReductionRunKind::Retained, 105, 0}));
    EXPECT_EQ(deriveDensePassRole(original, complete), DensePassRole::Complete);
    EXPECT_EQ(denseReductionProblemBytes(complete), 105U * 2U);
}

TEST(CubDenseReductionPlanning, NextBytesCountsValueAndArgAggregatePayloads) {
    const DenseReductionProblem value_original = valueProblem({2, 64}, {1});
    const DenseReductionProblem value_partial = advanceDenseReductionProblem(value_original, {1}, 32);
    EXPECT_EQ(denseReductionProblemBytes(value_partial), 2U * 32U * 4U);

    const DenseReductionProblem arg32_original = argProblem({2, 64}, {1}, DataType::UINT32);
    const DenseReductionProblem arg32_partial = advanceDenseReductionProblem(arg32_original, {1}, 32);
    EXPECT_EQ(denseReductionProblemBytes(arg32_partial), 2U * 32U * (4U + 4U));

    const DenseReductionProblem arg64_original = argProblem({2, 64}, {1}, DataType::UINT64);
    const DenseReductionProblem arg64_partial = advanceDenseReductionProblem(arg64_original, {1}, 32);
    EXPECT_EQ(denseReductionProblemBytes(arg64_partial), 2U * 32U * (4U + 8U));
}

TEST(CubDenseReductionPlanning, FinalBytesUseRequestedValueAndIndexOutputs) {
    const DenseReductionProblem value_original = valueProblem({6, 9}, {1}, DataType::BF16);
    const DenseReductionProblem value_final = advanceDenseReductionProblem(value_original, {1}, 1);
    EXPECT_EQ(denseReductionProblemBytes(value_final), 6U * 2U);

    const DenseReductionProblem both_original = argProblem(
        {6, 9}, {1}, DataType::UINT64, argOutput(true, true, DataType::FP16, DataType::UINT64));
    const DenseReductionProblem both_final = advanceDenseReductionProblem(both_original, {1}, 1);
    EXPECT_EQ(denseReductionProblemBytes(both_final), 6U * (2U + 8U));

    const DenseReductionProblem index_only_original = argProblem(
        {6, 9}, {1}, DataType::UINT32, argOutput(false, true, DataType::FP32, DataType::UINT32));
    const DenseReductionProblem index_only_final = advanceDenseReductionProblem(index_only_original, {1}, 1);
    EXPECT_EQ(denseReductionProblemBytes(index_only_final), 6U * 4U);
}

TEST(CubDenseReductionPlanning, ArgOriginalCoordinateContributionIsConsumedWhenRunBecomesPartial) {
    const DenseReductionProblem original = argProblem({8, 3, 5}, {0, 2});  // R K R
    ASSERT_EQ(original.runs.size(), 3U);
    ASSERT_EQ(denseArgOriginalIndexStride(original.runs[0]), std::optional<uint64_t>(5));
    ASSERT_EQ(denseArgOriginalIndexStride(original.runs[2]), std::optional<uint64_t>(1));

    const DenseReductionProblem partial = advanceDenseReductionProblem(original, {0}, 2);
    ASSERT_EQ(partial.runs.size(), 3U);
    EXPECT_EQ(partial.runs[0].kind, DenseReductionRunKind::IntermediateReduction);
    EXPECT_EQ(denseArgOriginalIndexStride(partial.runs[0]), std::nullopt);
    EXPECT_EQ(denseArgOriginalIndexStride(partial.runs[2]), std::optional<uint64_t>(1));
    EXPECT_EQ(partial.input_state, DenseReductionInputState::IntermediateAggregate);

    const DenseReductionProblem left_eliminated = advanceDenseReductionProblem(original, {0}, 1);
    ASSERT_EQ(left_eliminated.runs.size(), 2U);
    EXPECT_EQ(left_eliminated.input_state, DenseReductionInputState::IntermediateAggregate);
    EXPECT_EQ(denseArgOriginalIndexStride(left_eliminated.runs[1]), std::optional<uint64_t>(1));
}

TEST(CubDenseReductionPlanning, RejectsNonProgressingAndInvalidSiteTransitions) {
    const DenseReductionProblem original = valueProblem({2, 16, 3}, {1});
    EXPECT_THROW(static_cast<void>(advanceDenseReductionProblem(original, {1}, 16)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(advanceDenseReductionProblem(original, {1}, 17)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(advanceDenseReductionProblem(original, {1}, 0)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(advanceDenseReductionProblem(original, {0}, 1)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(classifyDenseReducerFamily(original, {3})), std::invalid_argument);
}
