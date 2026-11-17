#pragma once

// Host-only construction of executable RuntimeK proposal geometries.  The
// occupancy callback must query the compiled CUDA specialization;
// no synthetic occupancy values may enter the selector.  This is intentionally
// separate from the CUDA object containing the kernels.

#include "Utilities/TensorOperations/Reduction/RK/RKPhysicalIdentity.h"

#include <cstdint>
#include <functional>
#include <vector>

namespace ThorImplementation::ReductionInternal::RK {

struct RuntimeProposalGeometry {
    uint64_t outer = 0, r = 0, k = 0;
    uint32_t sm_count = 0;
    // Input element width. 1 is FP8 (PacketComplete tiles of 16..496, rotated Prefix 497 and Exact 512); 4 is FP32
    // (PacketComplete tiles of 4..124, rotated Prefix 125, no Exact); 2 is FP16 / BF16 (tiles of 8..248, rotated Prefix
    // 249, no Exact). Proposals carry it as element_bytes.
    uint8_t element_bytes = 1;
};

// Returns zero for an unavailable compiled specialization.  Propagate CUDA
// errors rather than treating them as an absent kernel.
using RuntimeProposalOccupancy = std::function<uint32_t(const PhysicalKernel&)>;

[[nodiscard]] std::vector<PhysicalKernel> enumerateRetainedRuntimeK(
    const RuntimeProposalGeometry& geometry,
    const RuntimeProposalOccupancy& occupancy);

} // namespace ThorImplementation::ReductionInternal::RK
