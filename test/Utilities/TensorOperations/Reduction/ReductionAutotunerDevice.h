#pragma once

// Contest candidates with known relative speeds for ReductionAutotunerContestTest (kernels in
// ReductionAutotunerDevice.cu). Each reduces 4 MiB of bytes to one FP32 partial per 4096-byte block.

#include "Utilities/TensorOperations/Reduction/Contest/ReductionAutotuner.h"

#include <cstdint>
#include <vector>

namespace ThorImplementation::ReductionAutotunerDevice {

inline constexpr uint64_t InputBytes = 4 * 1024 * 1024;
inline constexpr uint64_t BytesPerPartial = 4096;
inline constexpr uint64_t Partials = InputBytes / BytesPerPartial;

// In order: "complete_fast" (one pass), "staged_fp32_continuation" (partials to scratch, then copied to the output)
// and "complete_slow" (the fast kernel repeating its loads 16 times; the same result).
std::vector<ReductionAutotuning::Candidate> candidates();

}  // namespace ThorImplementation::ReductionAutotunerDevice
