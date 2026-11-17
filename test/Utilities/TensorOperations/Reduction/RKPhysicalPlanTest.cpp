// An original FP8 RK site is one ordinary physical candidate to the dense planner: a complete pass finishes its site,
// a staged pass hands real FP32 partials [O, shards, K] to a fresh, independently planned FP32 pass, and the planner
// picks the next site by bytes alone. Invalid requests are rejected. Host-only: the FP8 choice and every other
// proposal are injected, so nothing tunes or launches.

#include "Utilities/TensorOperations/Reduction/Planner/DenseReductionPlanner.h"
#include "Utilities/TensorOperations/Reduction/RK/Fp8/Fp8RKPhysicalSelection.h"
#include "Utilities/TensorOperations/Reduction/RK/RKPhysicalPlan.h"
#include "Utilities/TensorOperations/Reduction/ReductionTypes.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace C = ThorImplementation::ReductionInternal;
namespace F = ThorImplementation::ReductionInternal::RK;
using ThorImplementation::DataType;

namespace {

struct InjectedPhysicalPlan final : C::DenseReductionPhysicalPlan {};

C::DenseReductionProblem problem(std::vector<uint64_t> dimensions, std::vector<uint32_t> axes,
                                 DataType final_dtype = DataType::FP16) {
    C::DenseReductionOutputSpec output;
    output.value_dtype = final_dtype;
    return C::makeInitialDenseReductionProblem(dimensions, axes, C::DenseReductionAggregateKind::Value, output);
}

// O = 1, any R, K <= 512; one CTA.
F::PhysicalKernel oneWindow() { return {F::PhysicalFamily::OneWindow, 1, 1, 16, 0, 0, 0, 6, 1, 0}; }

F::PhysicalKernel deepR(uint64_t r, uint64_t k) {
    const uint64_t shards = r / 128 + (r % 128 != 0);
    return {F::PhysicalFamily::DeepRSmallK, 128, 0, 0, 0, 0, 0, 6, shards, shards * k * 4};
}

C::DensePhysicalSelectionContext context(DataType original = DataType::FP8_E4M3) {
    C::DensePhysicalSelectionContext ctx;
    ctx.rk.original_input_dtype = original;
    ctx.original_value_operation = ThorImplementation::ReductionOp::Sum;
    ctx.original_reduction_size = 131072;
    return ctx;
}

const C::RKPhysicalPlan* retained(const C::DenseReductionCandidate& candidate) {
    return dynamic_cast<const C::RKPhysicalPlan*>(candidate.physical_plan.get());
}

// The original RK site gets the injected FP8 kernel; every other site an injected placeholder that finishes it.
struct Injection {
    C::DensePhysicalSelectionContext context = ::context();
    F::PhysicalKernel physical = deepR(2048, 32);
    unsigned fp8_calls = 0;
    unsigned fp32_calls = 0;

    C::DensePhysicalSelectionContext planningContext() {
        C::DensePhysicalSelectionContext planning = context;
        planning.proposal_override = propose;
        planning.proposal_override_context = this;
        return planning;
    }

    static C::DenseReductionCandidate propose(const C::DenseReductionProblem& current, C::DenseReductionSite site,
                                              C::DenseReducerFamily family, const C::DenseRKPlanningContext&,
                                              void* opaque) {
        auto& self = *static_cast<Injection*>(opaque);
        if (current.input_state == C::DenseReductionInputState::OriginalInput && family == C::DenseReducerFamily::RK) {
            ++self.fp8_calls;
            return C::makeRetainedFp8RKPhysicalCandidate(current, site, self.physical, self.context);
        }
        if (current.input_state == C::DenseReductionInputState::IntermediateAggregate) ++self.fp32_calls;
        C::DenseReductionCandidate result;
        result.family = family;
        result.site = site;
        result.physical_plan = std::make_shared<InjectedPhysicalPlan>();
        result.next_problem = C::advanceDenseReductionProblem(current, site, 1);
        result.next_bytes = C::denseReductionProblemBytes(result.next_problem);
        return result;
    }
};

}  // namespace

TEST(RKPhysicalPlan, ACompleteLastPassFinishesTheReduction) {
    const auto candidate = C::makeRetainedFp8RKPhysicalCandidate(problem({16, 8}, {0}), {0}, oneWindow(), context());
    const auto* pass = retained(candidate);
    ASSERT_NE(pass, nullptr);
    EXPECT_EQ(candidate.family, C::DenseReducerFamily::RK);
    EXPECT_EQ(candidate.scratch_bytes, 0u);
    EXPECT_TRUE(pass->kernel == oneWindow());
    EXPECT_EQ(pass->role, C::DensePassRole::Complete);
    EXPECT_EQ(pass->input_dimensions, (std::vector<uint64_t>{1, 16, 8}));
    EXPECT_EQ(pass->output_dimensions, (std::vector<uint64_t>{1, 1, 8}));
    EXPECT_EQ(pass->output_dtype, DataType::FP16);
    EXPECT_EQ(candidate.next_bytes, 8u * 2u);
    EXPECT_FALSE(C::hasDenseReductionSites(candidate.next_problem));
    EXPECT_FALSE(candidate.next_problem.finalizer_pending);
}

TEST(RKPhysicalPlan, ACompletePassLeavesTheOtherAxesToAnFp32Pass) {
    Injection injection;
    injection.physical = oneWindow();
    injection.context.original_reduction_size = 16 * 3;
    const auto plan = C::planDenseReduction(problem({16, 8, 3}, {0, 2}), injection.planningContext());
    ASSERT_EQ(plan.steps.size(), 2u);
    EXPECT_EQ(injection.fp8_calls, 1u);
    EXPECT_EQ(injection.fp32_calls, 1u);
    const auto& first = plan.steps.front().selected_candidate;
    const auto* pass = retained(first);
    ASSERT_NE(pass, nullptr);
    EXPECT_EQ(pass->role, C::DensePassRole::First);
    EXPECT_EQ(pass->output_dtype, DataType::FP32);
    EXPECT_EQ(pass->output_dimensions, (std::vector<uint64_t>{1, 1, 24}));
    EXPECT_EQ(first.next_bytes, 24u * 4u);
    EXPECT_TRUE(plan.steps[1].input_problem == first.next_problem);
    EXPECT_EQ(plan.steps[1].selected_candidate.family, C::DenseReducerFamily::KR);
    EXPECT_FALSE(C::hasDenseReductionSites(plan.final_problem));
}

TEST(RKPhysicalPlan, AStagedPassHandsRealFp32PartialsToTheNextPass) {
    Injection injection;
    injection.context.original_reduction_size = 2048;
    const auto plan = C::planDenseReduction(problem({2048, 32}, {0}), injection.planningContext());
    ASSERT_EQ(plan.steps.size(), 2u);
    EXPECT_EQ(injection.fp8_calls, 1u);
    EXPECT_EQ(injection.fp32_calls, 1u);
    const auto& first = plan.steps.front().selected_candidate;
    const auto* pass = retained(first);
    ASSERT_NE(pass, nullptr);
    EXPECT_TRUE(pass->kernel.staged());
    EXPECT_EQ(pass->output_dtype, DataType::FP32);
    EXPECT_EQ(pass->role, C::DensePassRole::First);
    EXPECT_EQ(pass->output_dimensions, (std::vector<uint64_t>{1, 16, 32}));
    EXPECT_EQ(first.next_problem.runs[0].kind, C::DenseReductionRunKind::IntermediateReduction);
    EXPECT_EQ(first.next_problem.runs[0].extent, 16u);
    EXPECT_EQ(first.next_bytes, 16u * 32u * 4u);
    // The partials are the pass's output, not scratch.
    EXPECT_EQ(first.scratch_bytes, 0u);
    EXPECT_EQ(pass->kernel.first_stage_scratch_bytes, first.next_bytes);
    EXPECT_TRUE(plan.steps[1].input_problem == first.next_problem);
    EXPECT_EQ(plan.steps[1].selected_candidate.family, C::DenseReducerFamily::RK);
    EXPECT_EQ(C::deriveDensePassRole(plan.steps[1].input_problem, plan.steps[1].selected_candidate.next_problem),
              C::DensePassRole::Final);
}

TEST(RKPhysicalPlan, AfterAStagedPassThePlannerPicksTheNextSiteByBytes) {
    Injection injection;
    injection.physical = deepR(2048, 128);
    injection.context.original_reduction_size = 2048 * 64;
    const auto plan = C::planDenseReduction(problem({2048, 2, 64}, {0, 2}), injection.planningContext());
    ASSERT_EQ(plan.steps.size(), 3u);
    const auto& first = plan.steps[0].selected_candidate;
    EXPECT_EQ(first.family, C::DenseReducerFamily::RK);
    ASSERT_NE(retained(first), nullptr);
    EXPECT_EQ(retained(first)->output_dimensions.at(retained(first)->reduction_axis), 16u);
    EXPECT_EQ(first.next_bytes, 16u * 2u * 64u * 4u);
    EXPECT_TRUE(plan.steps[1].input_problem == first.next_problem);
    // The KR site wins next: 16 x 2 x 4 bytes is smaller than 2 x 64 x 4. The staged pass does not force its R
    // continuation first.
    EXPECT_EQ(plan.steps[1].selected_candidate.site.run_index, 2u);
    EXPECT_EQ(plan.steps[1].selected_candidate.family, C::DenseReducerFamily::KR);
    EXPECT_EQ(plan.steps[1].selected_candidate.next_bytes, 16u * 2u * 4u);
    EXPECT_TRUE(plan.steps[2].input_problem == plan.steps[1].selected_candidate.next_problem);
    EXPECT_EQ(plan.steps[2].selected_candidate.family, C::DenseReducerFamily::RK);
    EXPECT_EQ(plan.steps[2].input_problem.runs[0].kind, C::DenseReductionRunKind::IntermediateReduction);
    EXPECT_FALSE(C::hasDenseReductionSites(plan.final_problem));
    EXPECT_EQ(injection.fp8_calls, 1u);
    EXPECT_GE(injection.fp32_calls, 2u);
    EXPECT_NE(C::formatDenseReductionPlanDiagnostics(plan).find("retained_fp8_rk"), std::string::npos);
}

TEST(RKPhysicalPlan, BothFp8FormatsPlanAlike) {
    for (const DataType dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
        SCOPED_TRACE(static_cast<int>(dtype));
        const auto ctx = context(dtype);
        const auto complete =
            C::makeRetainedFp8RKPhysicalCandidate(problem({16, 8}, {0}, DataType::FP8_E4M3), {0}, oneWindow(), ctx);
        ASSERT_NE(retained(complete), nullptr);
        EXPECT_EQ(retained(complete)->input_dtype, dtype);
        EXPECT_EQ(retained(complete)->output_dtype, DataType::FP8_E4M3);
        EXPECT_EQ(retained(complete)->role, C::DensePassRole::Complete);
        const auto staged = C::makeRetainedFp8RKPhysicalCandidate(problem({2048, 32}, {0}, DataType::FP8_E4M3), {0},
                                                                  deepR(2048, 32), ctx);
        ASSERT_NE(retained(staged), nullptr);
        EXPECT_EQ(retained(staged)->input_dtype, dtype);
        EXPECT_EQ(retained(staged)->output_dtype, DataType::FP32);
        EXPECT_EQ(retained(staged)->output_dimensions.at(retained(staged)->reduction_axis), 16u);
        EXPECT_EQ(staged.next_bytes, 2048u);
    }
}

TEST(RKPhysicalPlan, InvalidRequestsAreRejected) {
    const auto initial = problem({2048, 32}, {0});
    auto inconsistent = deepR(2048, 32);
    inconsistent.first_stage_scratch_bytes--;
    EXPECT_THROW((void)C::makeRetainedFp8RKPhysicalCandidate(initial, {0}, inconsistent, context()),
                 std::invalid_argument);
    auto ctx = context();
    ctx.rk.original_input_dtype = DataType::FP32;
    EXPECT_THROW((void)C::makeRetainedFp8RKPhysicalCandidate(initial, {0}, deepR(2048, 32), ctx),
                 std::invalid_argument);
    ctx = context();
    ctx.output_scale = std::numeric_limits<float>::infinity();
    EXPECT_THROW((void)C::makeRetainedFp8RKPhysicalCandidate(initial, {0}, deepR(2048, 32), ctx),
                 std::invalid_argument);
    ctx = context();
    ctx.original_value_operation.reset();
    EXPECT_THROW((void)C::makeRetainedFp8RKPhysicalCandidate(initial, {0}, deepR(2048, 32), ctx),
                 std::invalid_argument);
    // An intermediate aggregate is not an original FP8 input.
    EXPECT_THROW((void)C::makeRetainedFp8RKPhysicalCandidate(C::advanceDenseReductionProblem(initial, {0}, 16), {0},
                                                             deepR(2048, 32), context()),
                 std::invalid_argument);
}
