// One RK site planned in isolation: a staged first stage finishes its own R with a local FP32 continuation before the
// recursive planner takes the remaining axes; the input transform runs once, in the first stage; the terminal
// finalizer moves to the continuation; illegal requests are rejected. Host-only.

#include "Utilities/TensorOperations/Reduction/RK/RKIsolatedSite.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace C = ThorImplementation::ReductionInternal;
namespace F = ThorImplementation::ReductionInternal::RK;
using ThorImplementation::DataType;

namespace {
C::DenseReductionProblem problem(std::vector<uint64_t> dimensions, std::vector<uint32_t> axes) {
    C::DenseReductionOutputSpec spec;
    spec.value_dtype = DataType::FP8_E4M3;
    return C::makeInitialDenseReductionProblem(dimensions, axes, C::DenseReductionAggregateKind::Value, spec);
}

C::ReductionStageSemantics squared() {
    C::ReductionStageSemantics semantics;
    semantics.input_transform = C::ReductionStageInputTransform::Square;
    return semantics;
}

C::ReductionStageSemantics mean(uint64_t divisor) {
    C::ReductionStageSemantics semantics;
    semantics.finalize = C::ReductionStageFinalize::Divide;
    semantics.finalize_divisor = divisor;
    return semantics;
}

const F::PhysicalKernel deep{F::PhysicalFamily::DeepRSmallK, 128, 0, 0, 0, 0, 0, 6, 16, 6144};
const F::PhysicalKernel complete{F::PhysicalFamily::OneWindow, 1, 1, 16, 0, 0, 0, 6, 1, 0};
}  // namespace

TEST(RKIsolatedSite, AStagedSiteFinishesItsROnFp32BeforeTheRemainingAxes) {
    // R = 2048, K = 96 is not the whole reduction: the outer planner still has R = 3 after this site.
    const auto multi = problem({2048, 32, 3}, {0, 2});
    const auto site = F::planIsolatedRKSite(multi, C::DenseReductionSite{0}, deep, squared());
    EXPECT_EQ(site.outer, 1u);
    EXPECT_EQ(site.r, 2048u);
    EXPECT_EQ(site.k, 96u);
    EXPECT_EQ(site.logical_role, C::DensePassRole::First);
    EXPECT_EQ(site.site_output_dtype, DataType::FP32);
    EXPECT_TRUE(site.layout.has_fp32_continuation);
    EXPECT_EQ(site.layout.partial_bytes, 6144u);
    EXPECT_EQ(site.layout.retained_result_bytes, 384u);
    EXPECT_EQ(site.layout.minimum_workspace_bytes, 6144u);
    EXPECT_EQ(site.first_stage_semantics.input_transform, C::ReductionStageInputTransform::Square);
    EXPECT_EQ(site.continuation_semantics.input_transform, C::ReductionStageInputTransform::Identity);
    EXPECT_EQ(site.nextBytes(), 384u);
    ASSERT_TRUE(C::hasDenseReductionSites(site.remaining_problem));
    EXPECT_EQ(C::classifyDenseReducerFamily(site.remaining_problem, C::DenseReductionSite{1}),
              C::DenseReducerFamily::KR);
    // The next site reads the FP32 aggregate, with no input transform.
    EXPECT_EQ(site.remaining_problem.input_state, C::DenseReductionInputState::IntermediateAggregate);
}

TEST(RKIsolatedSite, ACompleteKernelOwnsNoContinuation) {
    const auto one = problem({16, 3}, {0});
    const auto site = F::planIsolatedRKSite(one, C::DenseReductionSite{0}, complete, mean(16));
    EXPECT_EQ(site.logical_role, C::DensePassRole::Complete);
    EXPECT_FALSE(site.layout.has_fp32_continuation);
    EXPECT_EQ(site.site_output_dtype, DataType::FP8_E4M3);
    EXPECT_EQ(site.first_stage_semantics.finalize, C::ReductionStageFinalize::Divide);
    EXPECT_FALSE(C::hasDenseReductionSites(site.remaining_problem));
    EXPECT_FALSE(C::hasDenseReductionSites(C::advanceDenseReductionProblem(one, C::DenseReductionSite{0}, 1)));
}

TEST(RKIsolatedSite, StagingMovesTheFinalizerToTheContinuation) {
    const auto last = problem({2048, 32}, {0});
    const F::PhysicalKernel deep32{F::PhysicalFamily::DeepRSmallK, 128, 0, 0, 0, 0, 0, 6, 16, 2048};
    auto semantics = mean(2048);
    semantics.input_transform = C::ReductionStageInputTransform::Square;
    const auto site = F::planIsolatedRKSite(last, C::DenseReductionSite{0}, deep32, semantics);
    EXPECT_TRUE(site.layout.has_fp32_continuation);
    EXPECT_EQ(site.layout.partial_bytes, 2048u);
    EXPECT_EQ(site.first_stage_semantics.finalize, C::ReductionStageFinalize::Identity);
    EXPECT_EQ(site.continuation_semantics.finalize, C::ReductionStageFinalize::Divide);
    EXPECT_EQ(site.continuation_semantics.finalize_divisor, 2048u);
    EXPECT_EQ(site.continuation_semantics.input_transform, C::ReductionStageInputTransform::Identity);
}

TEST(RKIsolatedSite, IllegalRequestsAreRejected) {
    const auto one = problem({16, 3}, {0});
    const auto multi = problem({2048, 32, 3}, {0, 2});
    EXPECT_THROW((void)F::planIsolatedRKSite(one, C::DenseReductionSite{0}, complete, mean(0)), std::invalid_argument);
    // No finalizer while other R sites remain.
    auto finalized = mean(2048);
    finalized.input_transform = C::ReductionStageInputTransform::Square;
    EXPECT_THROW((void)F::planIsolatedRKSite(multi, C::DenseReductionSite{0}, deep, finalized), std::invalid_argument);
    // Site 1 is a KR site, not RK.
    EXPECT_THROW((void)F::planIsolatedRKSite(multi, C::DenseReductionSite{1}, deep, squared()), std::invalid_argument);
}
