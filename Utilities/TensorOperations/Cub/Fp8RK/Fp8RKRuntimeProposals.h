#pragma once

// Host-only construction of executable RuntimeK proposal geometries.  The
// occupancy callback must query the retained compiled CUDA specialization;
// no synthetic occupancy values may enter the selector.  This is intentionally
// separate from the CUDA object containing the kernels.

#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKPhysicalIdentity.h"

#include <cstdint>
#include <functional>
#include <vector>

namespace ThorImplementation::CubReductionInternal::Fp8RK {

struct RuntimeProposalGeometry {
    uint64_t outer = 0, r = 0, k = 0;
    uint32_t sm_count = 0;
};

// Returns zero for an unavailable compiled specialization.  Propagate CUDA
// errors rather than treating them as an absent kernel.
using RuntimeProposalOccupancy = std::function<uint32_t(const PhysicalKernel&)>;

[[nodiscard]] std::vector<PhysicalKernel> enumerateRetainedRuntimeK(
    const RuntimeProposalGeometry& geometry,
    const RuntimeProposalOccupancy& occupancy);

} // namespace ThorImplementation::CubReductionInternal::Fp8RK
