#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKPhysicalKernels.h"

#include <stdexcept>

namespace ThorImplementation::CubReductionInternal::Fp8RK {

uint32_t deepRSmallKResidency(DataType input_dtype, uint64_t k,
                             const CubReductionStageSemantics& semantics,
                             uint64_t rows_per_shard) {
    if (input_dtype == DataType::FP32) {  // 1192
        const auto occupancy = queryFp32DeepRSmallKRetainedOccupancy(rows_per_shard, k, semantics);
        return occupancy.max_active_blocks_per_sm > 0 ? static_cast<uint32_t>(occupancy.max_active_blocks_per_sm) : 0u;
    }
    if (input_dtype == DataType::FP16 || input_dtype == DataType::BF16) {  // 1246
        const auto occupancy = queryF16DeepRSmallKRetainedOccupancy(input_dtype, rows_per_shard, k, semantics);
        return occupancy.max_active_blocks_per_sm > 0 ? static_cast<uint32_t>(occupancy.max_active_blocks_per_sm) : 0u;
    }
    const auto occupancy = queryFp8DeepRSmallKRetainedOccupancy(
        input_dtype, rows_per_shard, k, semantics);
    return occupancy.max_active_blocks_per_sm > 0
        ? static_cast<uint32_t>(occupancy.max_active_blocks_per_sm) : 0u;
}

void launchDeepRSmallK(const PhysicalKernel& kernel, const CubReductionStageSemantics& semantics,
                       const Tensor& input, Tensor& partials,
                       const CubReductionGeometry& geometry, Stream& stream) {
    if (kernel.family != PhysicalFamily::DeepRSmallK || !kernel.staged() ||
        !structurallyValid(kernel, geometry.outer_size, geometry.reduction_size, geometry.inner_size))
        throw std::logic_error("Invalid retained DeepRSmallK physical stage");
    if (kernel.element_bytes == 4) {
        launchFp32DeepRSmallKRetainedFirstStage(semantics, input, partials, geometry, kernel.a,
                                                (geometry.reduction_size + kernel.a - 1) / kernel.a, stream);
        return;
    }
    if (kernel.element_bytes == 2) {  // 1246
        launchF16DeepRSmallKRetainedFirstStage(semantics, input, partials, geometry, kernel.a,
                                               (geometry.reduction_size + kernel.a - 1) / kernel.a, stream);
        return;
    }
    launchFp8DeepRSmallKRetainedFirstStage(
        semantics, input, partials, geometry, kernel.a,
        (geometry.reduction_size + kernel.a - 1) / kernel.a, stream);
}

}  // namespace ThorImplementation::CubReductionInternal::Fp8RK
