#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKPhysicalKernels.h"

#include <stdexcept>

namespace ThorImplementation::CubReductionInternal::Fp8RK {

uint32_t runtimeKResidency(DataType input_dtype,
                            const CubReductionStageSemantics& semantics,
                            const PhysicalKernel& kernel) {
    if (kernel.family != PhysicalFamily::RuntimeK) return 0;
    CubKernelOccupancyInfo occupancy{};
    if (input_dtype == DataType::FP32) {  // 1192: PacketComplete and RotatedPrefix (no FP32 Exact)
        switch (static_cast<RuntimeLayout>(kernel.a)) {
            case RuntimeLayout::PacketComplete:
                occupancy = queryFp32PacketCompleteOccupancy(semantics.input_transform, semantics.combine);
                break;
            case RuntimeLayout::RotatedPrefix:
                occupancy = queryFp32RuntimeKRotatedOccupancy(kernel.e, semantics.input_transform, semantics.combine);
                break;
            default: return 0;
        }
        return occupancy.max_active_blocks_per_sm > 0 ? static_cast<uint32_t>(occupancy.max_active_blocks_per_sm) : 0u;
    }
    if (input_dtype == DataType::FP16 || input_dtype == DataType::BF16) {  // 1246: as FP32
        switch (static_cast<RuntimeLayout>(kernel.a)) {
            case RuntimeLayout::PacketComplete:
                occupancy = queryF16PacketCompleteOccupancy(input_dtype, semantics.input_transform, semantics.combine);
                break;
            case RuntimeLayout::RotatedPrefix:
                occupancy = queryF16RuntimeKRotatedOccupancy(input_dtype, kernel.e, semantics.input_transform,
                                                             semantics.combine);
                break;
            default: return 0;
        }
        return occupancy.max_active_blocks_per_sm > 0 ? static_cast<uint32_t>(occupancy.max_active_blocks_per_sm) : 0u;
    }
    switch (static_cast<RuntimeLayout>(kernel.a)) {
        case RuntimeLayout::PacketComplete:
            occupancy = queryFp8GeneralRParallelKTileRCooperativeCalibrationOccupancy(
                input_dtype, semantics.input_transform, semantics.combine);
            break;
        case RuntimeLayout::RotatedPrefix:
        case RuntimeLayout::RotatedExact:
            occupancy = queryFp8GeneralRParallelKTileRotatedCalibrationOccupancy(
                input_dtype, static_cast<RuntimeLayout>(kernel.a) == RuntimeLayout::RotatedExact,
                kernel.e, semantics.input_transform, semantics.combine);
            break;
        default: return 0;
    }
    return occupancy.max_active_blocks_per_sm > 0
        ? static_cast<uint32_t>(occupancy.max_active_blocks_per_sm) : 0u;
}

void launchRuntimeK(const PhysicalKernel& kernel, const CubReductionStageSemantics& semantics,
                    const Tensor& input, Tensor& stage_output,
                    const CubReductionGeometry& geometry, float output_scale, Stream& stream) {
    if (kernel.family != PhysicalFamily::RuntimeK ||
        !structurallyValid(kernel, geometry.outer_size, geometry.reduction_size, geometry.inner_size))
        throw std::logic_error("Invalid retained RuntimeK physical stage");
    if (kernel.element_bytes == 4) {
        if (static_cast<RuntimeLayout>(kernel.a) == RuntimeLayout::PacketComplete) {
            launchFp32PacketComplete(input, stage_output, geometry, static_cast<uint32_t>(kernel.b),
                                     static_cast<uint32_t>(kernel.c), stream, semantics, output_scale);
        } else {
            launchFp32RuntimeKRotatedFirstStage(input, stage_output, geometry, kernel.e, kernel.f, stream,
                                                semantics.input_transform, semantics.combine);
        }
        return;
    }
    if (kernel.element_bytes == 2) {  // 1246
        if (static_cast<RuntimeLayout>(kernel.a) == RuntimeLayout::PacketComplete) {
            launchF16PacketComplete(input, stage_output, geometry, static_cast<uint32_t>(kernel.b),
                                    static_cast<uint32_t>(kernel.c), stream, semantics, output_scale);
        } else if (static_cast<RuntimeLayout>(kernel.a) == RuntimeLayout::RotatedPrefix) {
            launchF16RuntimeKRotatedFirstStage(input, stage_output, geometry, kernel.e, kernel.f, stream,
                                               semantics.input_transform, semantics.combine);
        } else {
            throw std::logic_error("FP16 / BF16 RuntimeK has no rotated Exact layout");
        }
        return;
    }
    switch (static_cast<RuntimeLayout>(kernel.a)) {
        case RuntimeLayout::PacketComplete:
            launchFp8GeneralRParallelKTileRCooperativeCalibrationComplete(
                input, stage_output, geometry, static_cast<uint32_t>(kernel.b),
                static_cast<uint32_t>(kernel.c), stream, semantics, output_scale);
            return;
        case RuntimeLayout::RotatedPrefix:
        case RuntimeLayout::RotatedExact:
            launchFp8GeneralRParallelKTileRotatedCalibrationFirstStage(
                input, stage_output, geometry,
                static_cast<RuntimeLayout>(kernel.a) == RuntimeLayout::RotatedExact,
                kernel.e, kernel.f, stream, semantics.input_transform, semantics.combine);
            return;
    }
    throw std::logic_error("No retained RuntimeK physical launcher");
}

}  // namespace ThorImplementation::CubReductionInternal::Fp8RK
