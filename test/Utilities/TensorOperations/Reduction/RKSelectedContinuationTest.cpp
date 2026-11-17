// A staged RK first stage's FP32 continuation: the official one-pass selector reduces the actual partials (not a
// hypothetical finished R) until the site is done, in as many passes as it takes, and never touches the other axes. A
// complete first stage asks for nothing; a proposal that breaks the site or makes no progress is an error. Host-only:
// the FP32 proposals are injected.

#include "Utilities/TensorOperations/Reduction/RK/Fp8/Fp8RKPhysicalSelection.h"
#include "Utilities/TensorOperations/Reduction/RK/RKPhysicalPlan.h"
#include "Utilities/TensorOperations/Reduction/RK/RKSelectedContinuation.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace C = ThorImplementation::ReductionInternal;
namespace F = ThorImplementation::ReductionInternal::RK;
using ThorImplementation::DataType;

namespace {

struct InjectedPhysicalPlan final : C::DenseReductionPhysicalPlan {};

struct Injection {
    unsigned calls = 0;
    bool multi = false;       // reduce the partials in two passes (to 4, then 1)
    bool invalidate = false;  // change an unrelated axis
    bool no_progress = false;

    C::DensePhysicalSelectionContext context() {
        C::DensePhysicalSelectionContext ctx;
        ctx.rk.original_input_dtype = DataType::FP8_E4M3;
        ctx.proposal_override = propose;
        ctx.proposal_override_context = this;
        return ctx;
    }

    static C::DenseReductionCandidate propose(const C::DenseReductionProblem& problem, C::DenseReductionSite site,
                                              C::DenseReducerFamily family, const C::DenseRKPlanningContext& context,
                                              void* opaque) {
        auto& self = *static_cast<Injection*>(opaque);
        EXPECT_EQ(context.original_input_dtype, DataType::FP8_E4M3);
        EXPECT_EQ(problem.input_state, C::DenseReductionInputState::IntermediateAggregate);
        EXPECT_EQ(family, C::DenseReducerFamily::RK);
        ++self.calls;
        const uint64_t previous = problem.runs[site.run_index].extent;
        C::DenseReductionCandidate candidate;
        candidate.family = family;
        candidate.site = site;
        candidate.physical_plan = std::make_shared<InjectedPhysicalPlan>();
        candidate.next_problem = self.no_progress
                                     ? problem
                                     : C::advanceDenseReductionProblem(problem, site,
                                                                       self.multi && previous > 4 ? 4 : 1);
        if (self.invalidate) candidate.next_problem.runs.back().extent = 17;
        candidate.next_bytes = C::denseReductionProblemBytes(candidate.next_problem);
        return candidate;
    }
};

C::DenseReductionProblem problem(std::vector<uint64_t> dimensions, std::vector<uint32_t> axes) {
    C::DenseReductionOutputSpec spec;
    spec.value_dtype = DataType::FP16;
    return C::makeInitialDenseReductionProblem(dimensions, axes, C::DenseReductionAggregateKind::Value, spec);
}

C::ReductionStageSemantics squared() {
    C::ReductionStageSemantics semantics;
    semantics.input_transform = C::ReductionStageInputTransform::Square;
    return semantics;
}

const F::PhysicalKernel deep{F::PhysicalFamily::DeepRSmallK, 128, 0, 0, 0, 0, 0, 6, 16, 6144};

}  // namespace

TEST(RKSelectedContinuation, OnePassReducesTheActualPartials) {
    const auto current = problem({2048, 32, 3}, {0, 2});
    const auto site = F::planIsolatedRKSite(current, C::DenseReductionSite{0}, deep, squared());
    // The transform runs once, in the FP8 first stage; nothing is finalized while R sites remain.
    EXPECT_EQ(site.first_stage_semantics.input_transform, C::ReductionStageInputTransform::Square);
    EXPECT_EQ(site.first_stage_semantics.finalize, C::ReductionStageFinalize::Identity);
    EXPECT_EQ(site.continuation_semantics.input_transform, C::ReductionStageInputTransform::Identity);
    EXPECT_EQ(site.continuation_semantics.finalize, C::ReductionStageFinalize::Identity);

    Injection injection;
    const auto chain = F::selectIsolatedFp32Continuation(current, {0}, site, injection.context());
    EXPECT_EQ(chain.first_stage_output_problem.runs[0].kind, C::DenseReductionRunKind::IntermediateReduction);
    EXPECT_EQ(chain.first_stage_output_problem.runs[0].extent, 16u);
    ASSERT_EQ(chain.fp32_passes.size(), 1u);
    EXPECT_EQ(injection.calls, 1u);
    EXPECT_TRUE(chain.completed_site_problem == site.remaining_problem);
    EXPECT_TRUE(C::hasDenseReductionSites(chain.completed_site_problem));
    EXPECT_EQ(chain.completed_site_problem.runs.back().kind, C::DenseReductionRunKind::OriginalReduction);
    EXPECT_EQ(chain.completed_site_problem.input_state, C::DenseReductionInputState::IntermediateAggregate);
    EXPECT_EQ(chain.fp32_passes[0].selected.next_bytes,
              C::denseReductionProblemBytes(chain.fp32_passes[0].selected.next_problem));
}

TEST(RKSelectedContinuation, TheContinuationTakesAsManyPassesAsTheSelectorNeeds) {
    const auto current = problem({2048, 32, 3}, {0, 2});
    const auto site = F::planIsolatedRKSite(current, C::DenseReductionSite{0}, deep, squared());
    Injection injection;
    injection.multi = true;
    const auto chain = F::selectIsolatedFp32Continuation(current, {0}, site, injection.context());
    ASSERT_EQ(chain.fp32_passes.size(), 2u);
    EXPECT_EQ(injection.calls, 2u);
    EXPECT_EQ(chain.fp32_passes[0].selected.next_problem.runs[0].extent, 4u);
    EXPECT_TRUE(chain.fp32_passes[1].input_problem == chain.fp32_passes[0].selected.next_problem);
    EXPECT_TRUE(chain.completed_site_problem == site.remaining_problem);
}

TEST(RKSelectedContinuation, ACompleteFirstStageAsksForNothing) {
    const auto complete_problem = problem({16, 3}, {0});
    const F::PhysicalKernel complete{F::PhysicalFamily::OneWindow, 1, 1, 16, 0, 0, 0, 6, 1, 0};
    const auto site = F::planIsolatedRKSite(complete_problem, {0}, complete, squared());
    Injection injection;
    const auto chain = F::selectIsolatedFp32Continuation(complete_problem, {0}, site, injection.context());
    EXPECT_TRUE(chain.fp32_passes.empty());
    EXPECT_EQ(injection.calls, 0u);
    EXPECT_TRUE(chain.completed_site_problem == site.remaining_problem);
}

TEST(RKSelectedContinuation, AProposalThatBreaksTheSiteOrStallsIsAnError) {
    const auto current = problem({2048, 32, 3}, {0, 2});
    const auto site = F::planIsolatedRKSite(current, C::DenseReductionSite{0}, deep, squared());
    Injection invalid;
    invalid.invalidate = true;
    EXPECT_THROW((void)F::selectIsolatedFp32Continuation(current, {0}, site, invalid.context()), std::logic_error);
    EXPECT_EQ(invalid.calls, 1u);
    Injection stalled;
    stalled.no_progress = true;
    EXPECT_THROW((void)F::selectIsolatedFp32Continuation(current, {0}, site, stalled.context()), std::logic_error);
    EXPECT_EQ(stalled.calls, 1u);
}
