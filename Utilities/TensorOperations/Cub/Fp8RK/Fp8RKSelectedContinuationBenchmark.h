#pragma once

#include "Utilities/TensorOperations/Cub/DensePhysicalPassExecution.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKSelectedContinuation.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKPhysicalKernels.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ThorImplementation::ReductionAutotuning { struct Candidate; }
namespace ThorImplementation::CubReductionInternal::Fp8RK {

// Only legal candidate-local limitations (workspace budget, unavailable
// first-stage epilogue) may exclude a ranked candidate. Selector failures,
// invalid progress, CUDA errors, and unexpected resource-query failures must
// propagate instead of being silently treated as fallback opportunities.
struct UnavailableIsolatedCandidate : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};

struct BenchmarkContinuationPass {
    DenseReductionCandidate selected;
    DensePhysicalPassLayout layout;
    CubReductionStageSemantics semantics;
    DataType output_dtype = DataType::FP32;
    size_t output_offset = 0; // valid only for nonterminal FP32 partials
    size_t physical_scratch_offset = 0;
    size_t physical_scratch_bytes = 0;
    bool last_pass = false;
};

// Not cached: each cache-miss benchmark candidate owns its own temporary chain.
struct SelectedIsolatedBenchmark {
    IsolatedRKSite site;
    PhysicalKernel first_physical;
    CubReductionGeometry first_geometry;
    std::vector<BenchmarkContinuationPass> continuation;
    size_t first_partial_offset = 0;
    size_t scratch_bytes = 0; // FP32 intermediates + physical scratch, not caller output
    float output_scale = 1.0f;

    // Scratch is provided by the autotuner rotating buffers or a GPU smoke.
    // No selecting, allocations, or synchronizations occur in launch().
    void launch(const Tensor& input, Tensor& output, const Tensor& scratch, Stream& stream) const;
    [[nodiscard]] ReductionAutotuning::Candidate asAutotuneCandidate(std::string name) const;
};

/** Selects each hypothetical FP32 dependency before constructing launch
 * resources and before the FP8 GPU measurement mutex is acquired. This is
 * a cache-miss candidate timing program, not a production stamped executor.
 * The FP8 cache stores only the selected first-stage physical kernel.
 */
[[nodiscard]] SelectedIsolatedBenchmark makeSelectedIsolatedBenchmark(
    const DenseReductionProblem& original_problem,
    DenseReductionSite nominated_site,
    IsolatedRKSite site,
    PhysicalKernel first_physical,
    const DensePhysicalSelectionContext& fp32_selector,
    float output_scale,
    uint64_t workspace_budget_bytes,
    const Stream& stream);

}  // namespace ThorImplementation::CubReductionInternal::Fp8RK
