#pragma once

// Host-only storage contract for the *complete nominated RK site*, not just
// the first kernel. Kept separate from CUDA entry points and the XGBoost model
// so it can be compiled/tested independently.
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKPhysicalIdentity.h"

#include <array>
#include <cstdint>

namespace ThorImplementation::CubReductionInternal::Fp8RK {

struct ContinuationLayout {
    // Staged FP8 kernels write FP32 [outer,shards,k], then an FP32 RK
    // continuation reduces axis 1 to [outer,k]. Complete FP8 kernels write
    // directly to [outer,k] and need no local continuation.
    bool has_fp32_continuation = false;
    bool last_reduction_site = true;
    uint64_t shards = 1;
    std::array<uint64_t, 3> continuation_input_dimensions{};
    std::array<uint64_t, 2> site_output_dimensions{};
    uint32_t continuation_reduction_axis = 1;

    // Nonzero only for staged first stages; equals the FP32 partial tensor
    // size, and can be bound as the autotuner's scratch prefix.
    uint64_t partial_bytes = 0;
    // When other reduction runs remain, their FP32 aggregate output must be
    // retained, even if this particular RK pass is physically complete.
    uint64_t retained_result_bytes = 0;
    // Scratch lower bound for the isolated site. Result bytes are stored in
    // the caller-owned output tensor, and do NOT count toward scratch. The
    // benchmark-only selected FP32 continuation scratch is added separately.
    uint64_t minimum_workspace_bytes = 0;
};

// Scratch passed to one ReductionAutotuning::Candidate callback. The tuner
// owns its output ring separately; any retained result is written there, not
// into scratch. All offsets are 32-byte aligned for Tensor subviews.
struct AutotuneScratchLayout {
    uint64_t partial_offset_bytes = 0;
    uint64_t partial_bytes = 0;
    uint64_t continuation_offset_bytes = 0;
    uint64_t continuation_bytes = 0;
    uint64_t total_bytes = 0;
};

/** Compose the layout's physical FP32 partial buffer with the actual queried
 * FP32 continuation temporary storage. No scratch is needed for complete
 * single-stage kernels. Passing continuation scratch to a non-staged kernel
 * is a programming error. Throws on integer overflow.
 */
[[nodiscard]] AutotuneScratchLayout makeAutotuneScratchLayout(
    const ContinuationLayout& layout, uint64_t continuation_scratch_bytes);

/** Fail closed on inconsistent/overflowed physical metadata. A caller must
 * separately validate output dtype, operation/finalization capability, and
 * every continuation's queried scratch before admitting an autotune plan.
 */
[[nodiscard]] ContinuationLayout describeContinuationLayout(
    const PhysicalKernel& first_stage, uint64_t outer, uint64_t r, uint64_t k,
    bool last_reduction_site);

}  // namespace ThorImplementation::CubReductionInternal::Fp8RK
