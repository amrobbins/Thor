// Host-only mathematical regression for per-site FP8 RK autotuning.
// The chosen kernel must finish this one site (including a local FP32
// continuation), then the existing recursive planner handles later axes.
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKIsolatedSite.h"

#include <cassert>
#include <stdexcept>
#include <vector>

namespace C = ThorImplementation::CubReductionInternal;
namespace F = ThorImplementation::CubReductionInternal::Fp8RK;
using ThorImplementation::DataType;

static C::DenseReductionProblem problem(std::vector<uint64_t> dimensions,
                                       std::vector<uint32_t> axes,
                                       DataType output = DataType::FP8_E4M3) {
    C::DenseReductionOutputSpec spec;
    spec.value_dtype = output;
    return C::makeInitialDenseReductionProblem(
        dimensions, axes, C::DenseReductionAggregateKind::Value, spec);
}

int main() {
    // Isolated R=2048,K=96 is not the entire logical reduction. The staged
    // variant must finish R locally before the outer planner tackles R=3.
    const auto multi = problem({2048, 32, 3}, {0, 2});
    const F::PhysicalKernel deep{F::PhysicalFamily::DeepRSmallK,
                                  128, 0, 0, 0, 0, 0, 6, 16, 6144};
    C::CubReductionStageSemantics square;
    square.input_transform = C::CubReductionStageInputTransform::Square;
    auto site = F::planIsolatedRKSite(multi, C::DenseReductionSite{0}, deep, square);
    assert(site.outer == 1 && site.r == 2048 && site.k == 96);
    assert(site.logical_role == C::DensePassRole::First);
    assert(site.site_output_dtype == DataType::FP32);
    assert(site.layout.has_fp32_continuation);
    assert(site.layout.partial_bytes == 6144);
    assert(site.layout.retained_result_bytes == 384);
    assert(site.layout.minimum_workspace_bytes == 6144);
    assert(site.first_stage_semantics.input_transform == C::CubReductionStageInputTransform::Square);
    assert(site.continuation_semantics.input_transform == C::CubReductionStageInputTransform::Identity);
    assert(site.nextBytes() == 384);
    assert(C::hasDenseReductionSites(site.remaining_problem));
    assert(C::classifyDenseReducerFamily(site.remaining_problem, C::DenseReductionSite{1})
           == C::DenseReducerFamily::KR);
    // The next site must consume the FP32 aggregate, with no input transform.
    assert(site.remaining_problem.input_state == C::DenseReductionInputState::IntermediateAggregate);

    // A complete first stage does not own a fictitious FP32 continuation.
    const auto one = problem({16, 3}, {0});
    const F::PhysicalKernel complete{F::PhysicalFamily::OneWindow,
                                     1, 1, 16, 0, 0, 0, 6, 1, 0};
    C::CubReductionStageSemantics mean;
    mean.finalize = C::CubReductionStageFinalize::Divide;
    mean.finalize_divisor = 16;
    site = F::planIsolatedRKSite(one, C::DenseReductionSite{0}, complete, mean);
    assert(site.logical_role == C::DensePassRole::Complete);
    assert(!site.layout.has_fp32_continuation);
    assert(site.site_output_dtype == DataType::FP8_E4M3);
    assert(site.first_stage_semantics.finalize == C::CubReductionStageFinalize::Divide);
    assert(!C::hasDenseReductionSites(site.remaining_problem));

    // Staging moves the terminal finalizer to FP32 and applies the original
    // transform only once, in the FP8 first stage.
    const auto deep_last = problem({2048, 32}, {0});
    const F::PhysicalKernel deep32{F::PhysicalFamily::DeepRSmallK,
                                   128, 0, 0, 0, 0, 0, 6, 16, 2048};
    mean.input_transform = C::CubReductionStageInputTransform::Square;
    mean.finalize_divisor = 2048;
    site = F::planIsolatedRKSite(deep_last, C::DenseReductionSite{0}, deep32, mean);
    assert(site.layout.has_fp32_continuation && site.layout.partial_bytes == 2048);
    assert(site.first_stage_semantics.finalize == C::CubReductionStageFinalize::Identity);
    assert(site.continuation_semantics.finalize == C::CubReductionStageFinalize::Divide);
    assert(site.continuation_semantics.finalize_divisor == 2048);
    assert(site.continuation_semantics.input_transform == C::CubReductionStageInputTransform::Identity);

    bool rejected = false;
    try { (void)F::planIsolatedRKSite(one, C::DenseReductionSite{0}, complete,
                                    C::CubReductionStageSemantics{.finalize = C::CubReductionStageFinalize::Divide,
                                                                  .finalize_divisor = 0}); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);

    rejected = false;
    try { (void)F::planIsolatedRKSite(multi, C::DenseReductionSite{0}, deep, mean); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected); // Do not finalize when other R sites remain.

    rejected = false;
    try { (void)F::planIsolatedRKSite(multi, C::DenseReductionSite{1}, deep, square); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected); // This local stage is KR, not RK.

    // Already-aggregated input cannot be transformed twice.
    const auto next = C::advanceDenseReductionProblem(one, C::DenseReductionSite{0}, 1);
    assert(!C::hasDenseReductionSites(next));
}
