#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKContinuationLayout.h"

#include <limits>
#include <stdexcept>

namespace ThorImplementation::CubReductionInternal::Fp8RK {
namespace {
[[nodiscard]] uint64_t checkedMul(uint64_t a, uint64_t b) {
    if (b && a > std::numeric_limits<uint64_t>::max() / b)
        throw std::overflow_error("FP8 RK continuation tensor size overflow");
    return a * b;
}
[[nodiscard]] uint64_t checkedAdd(uint64_t a, uint64_t b) {
    if (a > std::numeric_limits<uint64_t>::max() - b)
        throw std::overflow_error("FP8 RK continuation workspace size overflow");
    return a + b;
}
[[nodiscard]] uint64_t ceilDiv(uint64_t n, uint64_t d) { return n / d + (n % d != 0); }
}

ContinuationLayout describeContinuationLayout(
    const PhysicalKernel& first_stage, uint64_t outer, uint64_t r, uint64_t k,
    bool last_reduction_site) {
    if (!structurallyValid(first_stage, outer, r, k))
        throw std::invalid_argument("FP8 RK continuation requires a legal physical first stage");

    ContinuationLayout result;
    result.last_reduction_site = last_reduction_site;
    result.has_fp32_continuation = first_stage.staged();
    result.site_output_dimensions = {outer, k};
    switch (first_stage.family) {
        case PhysicalFamily::OneWindow: result.shards = 1; break;
        case PhysicalFamily::DeepRSmallK: result.shards = ceilDiv(r, first_stage.a); break;
        case PhysicalFamily::AlignedK: result.shards = first_stage.c; break;
        case PhysicalFamily::RuntimeK: result.shards = first_stage.f; break;
    }
    // A staged physical candidate must have multiple R shards. No extra
    // continuation is permitted for a complete first stage.
    if (result.has_fp32_continuation != (result.shards > 1))
        throw std::invalid_argument("FP8 RK staging/shard count mismatch");

    const uint64_t site_values = checkedMul(outer, k);
    result.continuation_input_dimensions = {outer, result.shards, k};
    result.retained_result_bytes = last_reduction_site ? 0 : checkedMul(site_values, 4);
    if (result.has_fp32_continuation) {
        result.partial_bytes = checkedMul(checkedMul(site_values, result.shards), 4);
        if (result.partial_bytes != first_stage.first_stage_scratch_bytes)
            throw std::invalid_argument("FP8 RK scratch does not match FP32 partial topology");
    }
    // The site result belongs to the caller/output ring, not the scratch ring.
    // Do not double-count that result as candidate scratch.
    result.minimum_workspace_bytes = result.partial_bytes;
    return result;
}
AutotuneScratchLayout makeAutotuneScratchLayout(
    const ContinuationLayout& layout, uint64_t continuation_scratch_bytes) {
    if (!layout.has_fp32_continuation && continuation_scratch_bytes)
        throw std::invalid_argument("Complete FP8 RK candidate cannot own FP32 continuation scratch");
    AutotuneScratchLayout result;
    result.partial_bytes = layout.partial_bytes;
    result.continuation_bytes = continuation_scratch_bytes;
    if (!layout.has_fp32_continuation) return result;
    // Avoid needless storage padding when the continuation requires no
    // scratch. The partial buffer itself is a separate FP32 tensor view.
    if (continuation_scratch_bytes) {
        result.continuation_offset_bytes = checkedAdd(layout.partial_bytes, 31) & ~uint64_t{31};
        result.total_bytes = checkedAdd(result.continuation_offset_bytes, continuation_scratch_bytes);
    } else {
        result.continuation_offset_bytes = layout.partial_bytes;
        result.total_bytes = layout.partial_bytes;
    }
    return result;
}

}  // namespace ThorImplementation::CubReductionInternal::Fp8RK
