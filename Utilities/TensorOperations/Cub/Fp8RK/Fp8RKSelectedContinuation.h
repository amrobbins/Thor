#pragma once

// Benchmark-only FP32 lookahead for one nominated retained FP8 RK first stage.
// This is NOT a production reduction plan and never accompanies a cached FP8
// selection. Every continuation pass comes from the ordinary physical selector.
#include "Utilities/TensorOperations/Cub/DenseReductionPhysicalSelector.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKIsolatedSite.h"

#include <vector>

namespace ThorImplementation::CubReductionInternal::Fp8RK {

struct SelectedContinuationPass {
    DenseReductionProblem input_problem;
    DenseReductionCandidate selected;
};

struct SelectedIsolatedContinuation {
    // The actual output after ONE FP8 physical launch. For staged kernels
    // the nominated run is still an IntermediateReduction of extent N>1.
    DenseReductionProblem first_stage_output_problem;
    // Only the nominated dimension is completed by the hypothetical chain.
    DenseReductionProblem completed_site_problem;
    std::vector<SelectedContinuationPass> fp32_passes;
};

/**
 * Selects FP32 physical passes sequentially, through the same selector used
 * by the ordinary production planner. Other reduction sites are never selected.
 * Requires no Tensor, CUDA stream, GPU timing or workspace allocation.
 *
 * The returned descriptors are ephemeral benchmark dependencies: neither
 * IsolatedRKSite nor Fp8RKSelection owns or requires their execution later.
 */
[[nodiscard]] SelectedIsolatedContinuation selectIsolatedFp32Continuation(
    const DenseReductionProblem& original_problem,
    DenseReductionSite nominated_site,
    const IsolatedRKSite& site,
    const DensePhysicalSelectionContext& physical_selector);

}  // namespace ThorImplementation::CubReductionInternal::Fp8RK
