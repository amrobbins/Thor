#pragma once

// Benchmark-only layout for timing one nominated original-input FP8 RK
// physical kernel with its hypothetical FP32 completion. This temporary
// continuation belongs to the candidate's timing interval, not to the
// ordinary multi-axis planner or to any production stamp. Production instead
// plans each real first-stage FP32 output as the next reduction problem.

#include "Utilities/TensorOperations/Cub/DenseReductionPlanning.h"
#include "Utilities/TensorOperations/Cub/CubReductionStageSemantics.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKContinuationLayout.h"

namespace ThorImplementation::CubReductionInternal::Fp8RK {

struct IsolatedRKSite {
    DenseReductionSite site;
    DenseReductionProblem remaining_problem;
    DensePassRole logical_role = DensePassRole::Complete;
    DataType site_output_dtype = DataType::FP32;
    uint64_t outer = 0, r = 0, k = 0;
    ContinuationLayout layout;

    // For a staged kernel, first_stage_semantics suppresses output finalizing
    // (including MEAN / L2); continuation_semantics suppresses the already
    // applied input transform and alone receives the finalizer. For complete
    // kernels, the finalizer stays on the first stage.
    CubReductionStageSemantics first_stage_semantics;
    CubReductionStageSemantics continuation_semantics;

    [[nodiscard]] uint64_t nextBytes() const;
};

/**
 * `site_semantics` applies to the complete nominated mathematical site.
 * It must apply the input transform iff current input is original, and
 * finalization iff no further mathematical reduction sites remain.
 * The caller computes these using its existing original-domain reduction
 * semantics (in particular the original MEAN divisor).
 */
[[nodiscard]] IsolatedRKSite planIsolatedRKSite(
    const DenseReductionProblem& current,
    DenseReductionSite site,
    const PhysicalKernel& first_stage,
    const CubReductionStageSemantics& site_semantics);

} // namespace ThorImplementation::CubReductionInternal::Fp8RK
