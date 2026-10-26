#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKPhysicalKernels.h"

#include <stdexcept>

namespace ThorImplementation::CubReductionInternal::Fp8RK {

uint32_t oneWindowResidency(DataType input_dtype, uint64_t k,
                            CubReductionStageCombine combine,
                            uint32_t outputs_per_warp, uint32_t warps_per_output,
                            uint32_t packet_bytes) {
    // The query references the exact retained compiled operation, dtype, K,
    // packet-width and outputs-per-warp specialization; a pair outside
    // Fp8OneWindowPolicy::candidates is not compiled and reports absent.
    // Transform is a runtime operand and does not multiply the inventory.
    const auto occupancy = queryFp8GeneralRParallelRCooperativeCalibrationOccupancy(
        input_dtype, k, outputs_per_warp, warps_per_output, packet_bytes, combine);
    return occupancy.max_active_blocks_per_sm > 0
        ? static_cast<uint32_t>(occupancy.max_active_blocks_per_sm) : 0u;
}

void launchOneWindow(const PhysicalKernel& kernel, const CubReductionStageSemantics& semantics,
                     const Tensor& input, Tensor& output,
                     const CubReductionGeometry& geometry, float output_scale, Stream& stream) {
    if (kernel.family != PhysicalFamily::OneWindow || kernel.staged() ||
        !structurallyValid(kernel, geometry.outer_size, geometry.reduction_size, geometry.inner_size))
        throw std::logic_error("Invalid retained OneWindow physical stage");
    launchFp8GeneralRParallelRCooperativeCalibrationComplete(
        input, output, geometry, static_cast<uint32_t>(kernel.a),
        static_cast<uint32_t>(kernel.b), static_cast<uint32_t>(kernel.c),
        stream, semantics, output_scale);
}

}  // namespace ThorImplementation::CubReductionInternal::Fp8RK
