// Host-only regression: FP8 RK continuation shape/workspace lower bounds.
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKContinuationLayout.h"
#include <cassert>
#include <cstdint>
#include <stdexcept>
namespace P = ThorImplementation::CubReductionInternal::Fp8RK;

int main() {
    const P::PhysicalKernel complete{P::PhysicalFamily::OneWindow, 1, 1, 16, 0, 0, 0, 6, 16, 0};
    auto p = P::describeContinuationLayout(complete, 128, 16, 3, true);
    assert(!p.has_fp32_continuation && p.minimum_workspace_bytes == 0);
    auto none = P::makeAutotuneScratchLayout(p, 0);
    assert(none.total_bytes == 0);
    p = P::describeContinuationLayout(complete, 128, 16, 3, false);
    assert(!p.has_fp32_continuation && p.retained_result_bytes == 1536);

    const P::PhysicalKernel deep{P::PhysicalFamily::DeepRSmallK, 128, 0, 0, 0, 0, 0, 6, 32, 4096};
    p = P::describeContinuationLayout(deep, 2, 2048, 32, true);
    assert(p.has_fp32_continuation && p.shards == 16 && p.partial_bytes == 4096);
    auto workspace = P::makeAutotuneScratchLayout(p, 17);
    assert(workspace.partial_offset_bytes == 0 && workspace.partial_bytes == 4096);
    assert(workspace.continuation_offset_bytes == 4096 && workspace.total_bytes == 4113);
    assert((p.continuation_input_dimensions == std::array<uint64_t, 3>{2, 16, 32}));
    assert(p.continuation_reduction_axis == 1 && p.minimum_workspace_bytes == 4096);
    p = P::describeContinuationLayout(deep, 2, 2048, 32, false);
    assert(p.retained_result_bytes == 256 && p.minimum_workspace_bytes == 4096);

    const P::PhysicalKernel aligned{P::PhysicalFamily::AlignedK, 8, 32, 4, 0, 0, 0, 6, 8, 8192};
    p = P::describeContinuationLayout(aligned, 2, 256, 256, false);
    assert(p.shards == 4 && p.partial_bytes == 8192 && p.retained_result_bytes == 2048);
    workspace = P::makeAutotuneScratchLayout(p, 3);
    assert(workspace.total_bytes == 8195);
    bool rejected_complete_scratch = false;
    try { (void)P::makeAutotuneScratchLayout(P::describeContinuationLayout(complete, 128, 16, 3, true), 1); }
    catch (const std::invalid_argument&) { rejected_complete_scratch = true; }
    assert(rejected_complete_scratch);
    P::ContinuationLayout odd;
    odd.has_fp32_continuation = true;
    odd.partial_bytes = 12;
    workspace = P::makeAutotuneScratchLayout(odd, 7);
    assert(workspace.continuation_offset_bytes == 32 && workspace.total_bytes == 39);
    odd.partial_bytes = UINT64_MAX - 2;
    bool overflow_rejected = false;
    try { (void)P::makeAutotuneScratchLayout(odd, 8); }
    catch (const std::overflow_error&) { overflow_rejected = true; }
    assert(overflow_rejected);

    const P::PhysicalKernel rotated{P::PhysicalFamily::RuntimeK,
        static_cast<uint64_t>(P::RuntimeLayout::RotatedExact), 0, 512, 1, 256, 8,
        6, 16, 32768};
    p = P::describeContinuationLayout(rotated, 2, 2048, 512, true);
    assert(p.shards == 8 && p.partial_bytes == 32768 && p.retained_result_bytes == 0);

    auto bad = rotated;
    bad.first_stage_scratch_bytes = UINT64_MAX;
    bool rejected = false;
    try { (void)P::describeContinuationLayout(bad, 2, 2048, 512, true); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
}
