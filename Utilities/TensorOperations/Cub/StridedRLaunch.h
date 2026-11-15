#pragma once

// 1366: the launch a StridedR stamp holds (StridedReduction.h has the kernels' API). CUDA-free and dependency-free so
// CubReduction.h can hold one. 1371: two families, the stride-0 closed form and Split (Block is gone).

#include <cstdint>

namespace ThorImplementation::CubReductionInternal {

enum class StridedRFamily : uint8_t { ClosedForm = 0, Split = 2 };

/** One StridedR launch: its family, Split's CTA count G, and its workspace (Split: G FP32 partials and the ticket). */
struct StridedRLaunch {
    StridedRFamily family = StridedRFamily::ClosedForm;
    uint32_t grid = 1;
    uint64_t workspace_bytes = 0;
    bool operator==(const StridedRLaunch&) const = default;
};

}  // namespace ThorImplementation::CubReductionInternal
