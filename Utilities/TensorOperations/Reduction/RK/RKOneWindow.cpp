#include "Utilities/TensorOperations/Reduction/RK/RKPhysicalKernels.h"

#include <stdexcept>

namespace ThorImplementation::ReductionInternal::RK {
namespace {
// The FP32 unit an (O, P16) variant runs at K: the one FP32 unit with that ownership whose bucket holds K.
OneWindowUnit fp32OneWindowUnit(uint64_t k, uint32_t outputs_per_warp, uint32_t packet_bytes) {
    if (packet_bytes == 16) {
        for (const OneWindowUnit& unit : fp32OneWindowUnits(k)) {
            if (unit.outputs_per_warp == outputs_per_warp) return unit;
        }
    }
    throw std::logic_error("FP32 OneWindow has no unit for this ownership at K");
}
// The same for FP16 / BF16 (f16OneWindowUnits).
OneWindowUnit f16OneWindowUnit(uint64_t k, uint32_t outputs_per_warp, uint32_t packet_bytes) {
    if (packet_bytes == 16) {
        for (const OneWindowUnit& unit : f16OneWindowUnits(k)) {
            if (unit.outputs_per_warp == outputs_per_warp) return unit;
        }
    }
    throw std::logic_error("FP16 / BF16 OneWindow has no unit for this ownership at K");
}
}  // namespace

uint32_t oneWindowResidency(DataType input_dtype, uint64_t k,
                            ReductionStageCombine combine,
                            uint32_t outputs_per_warp, uint32_t warps_per_output,
                            uint32_t packet_bytes) {
    // The query references the exact bucketed OneWindow unit this variant runs at K (outputs per warp, packet bytes,
    // components-per-lane bucket).
    // A variant without a unit throws std::logic_error, which the selector
    // treats as absent. Transform is a runtime operand and does not multiply
    // the inventory.
    if (input_dtype == DataType::FP32) {
        const auto occupancy = queryFp32OneWindowOccupancy(
            k, fp32OneWindowUnit(k, outputs_per_warp, packet_bytes), warps_per_output, combine);
        return occupancy.max_active_blocks_per_sm > 0 ? static_cast<uint32_t>(occupancy.max_active_blocks_per_sm) : 0u;
    }
    if (input_dtype == DataType::FP16 || input_dtype == DataType::BF16) {
        const auto occupancy = queryF16OneWindowOccupancy(
            input_dtype, k, f16OneWindowUnit(k, outputs_per_warp, packet_bytes), warps_per_output, combine);
        return occupancy.max_active_blocks_per_sm > 0 ? static_cast<uint32_t>(occupancy.max_active_blocks_per_sm) : 0u;
    }
    const auto occupancy = queryFp8OneWindowOccupancyForVariant(
        input_dtype, k, outputs_per_warp, warps_per_output, packet_bytes, combine);
    return occupancy.max_active_blocks_per_sm > 0
        ? static_cast<uint32_t>(occupancy.max_active_blocks_per_sm) : 0u;
}

void launchOneWindow(const PhysicalKernel& kernel, const ReductionStageSemantics& semantics,
                     const StridedSpan& input, const StridedSpan& output,
                     const ReductionGeometry& geometry, float output_scale, Stream& stream) {
    if (kernel.family != PhysicalFamily::OneWindow || kernel.staged() ||
        !structurallyValid(kernel, geometry.outer_size, geometry.reduction_size, geometry.inner_size))
        throw std::logic_error("Invalid retained OneWindow physical stage");
    if (kernel.element_bytes == 4) {
        launchFp32OneWindow(
            input, output, geometry,
            fp32OneWindowUnit(geometry.inner_size, static_cast<uint32_t>(kernel.a), static_cast<uint32_t>(kernel.c)),
            static_cast<uint32_t>(kernel.b), stream, semantics, output_scale);
        return;
    }
    if (kernel.element_bytes == 2) {  // FP16 or BF16 (the launcher checks the input dtype)
        launchF16OneWindow(
            input, output, geometry,
            f16OneWindowUnit(geometry.inner_size, static_cast<uint32_t>(kernel.a), static_cast<uint32_t>(kernel.c)),
            static_cast<uint32_t>(kernel.b), stream, semantics, output_scale);
        return;
    }
    launchFp8OneWindowForVariant(
        input, output, geometry, static_cast<uint32_t>(kernel.a),
        static_cast<uint32_t>(kernel.b), static_cast<uint32_t>(kernel.c),
        stream, semantics, output_scale);
}

}  // namespace ThorImplementation::ReductionInternal::RK
