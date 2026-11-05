#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKPhysicalKernels.h"

#include <stdexcept>

namespace ThorImplementation::CubReductionInternal::Fp8RK {
namespace {
// 1192: the FP32 unit an (O, P16) variant runs at K: the one FP32 unit with that ownership whose bucket holds K.
Fp8OneWindowRuntimeKUnit fp32OneWindowUnit(uint64_t k, uint32_t outputs_per_warp, uint32_t packet_bytes) {
    if (packet_bytes == 16) {
        for (const Fp8OneWindowRuntimeKUnit& unit : fp32OneWindowRuntimeKUnits(k)) {
            if (unit.outputs_per_warp == outputs_per_warp) return unit;
        }
    }
    throw std::logic_error("FP32 OneWindow has no unit for this ownership at K");
}
}  // namespace

uint32_t oneWindowResidency(DataType input_dtype, uint64_t k,
                            CubReductionStageCombine combine,
                            uint32_t outputs_per_warp, uint32_t warps_per_output,
                            uint32_t packet_bytes) {
    // The query references the exact bucketed OneWindow unit this variant runs
    // at K (outputs per warp, packet bytes, components-per-lane bucket; P5-J2).
    // A variant without a unit throws std::logic_error, which the selector
    // treats as absent. Transform is a runtime operand and does not multiply
    // the inventory.
    if (input_dtype == DataType::FP32) {
        const auto occupancy = queryFp32OneWindowRuntimeKOccupancy(
            k, fp32OneWindowUnit(k, outputs_per_warp, packet_bytes), warps_per_output, combine);
        return occupancy.max_active_blocks_per_sm > 0 ? static_cast<uint32_t>(occupancy.max_active_blocks_per_sm) : 0u;
    }
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
    if (kernel.element_bytes == 4) {
        launchFp32OneWindowRuntimeKComplete(
            input, output, geometry,
            fp32OneWindowUnit(geometry.inner_size, static_cast<uint32_t>(kernel.a), static_cast<uint32_t>(kernel.c)),
            static_cast<uint32_t>(kernel.b), stream, semantics, output_scale);
        return;
    }
    launchFp8GeneralRParallelRCooperativeCalibrationComplete(
        input, output, geometry, static_cast<uint32_t>(kernel.a),
        static_cast<uint32_t>(kernel.b), static_cast<uint32_t>(kernel.c),
        stream, semantics, output_scale);
}

}  // namespace ThorImplementation::CubReductionInternal::Fp8RK
