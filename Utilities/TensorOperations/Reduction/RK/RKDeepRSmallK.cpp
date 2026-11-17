#include "Utilities/TensorOperations/Reduction/RK/RKPhysicalKernels.h"

#include <stdexcept>

namespace ThorImplementation::ReductionInternal::RK {

uint32_t deepRSmallKResidency(DataType input_dtype, uint64_t k,
                             const ReductionStageSemantics& semantics,
                             uint64_t rows_per_shard) {
    if (input_dtype == DataType::FP32) {
        const auto occupancy = queryFp32DeepRSmallKOccupancy(rows_per_shard, k, semantics);
        return occupancy.max_active_blocks_per_sm > 0 ? static_cast<uint32_t>(occupancy.max_active_blocks_per_sm) : 0u;
    }
    if (input_dtype == DataType::FP16 || input_dtype == DataType::BF16) {
        const auto occupancy = queryF16DeepRSmallKOccupancy(input_dtype, rows_per_shard, k, semantics);
        return occupancy.max_active_blocks_per_sm > 0 ? static_cast<uint32_t>(occupancy.max_active_blocks_per_sm) : 0u;
    }
    const auto occupancy = queryFp8DeepRSmallKOccupancy(
        input_dtype, rows_per_shard, k, semantics);
    return occupancy.max_active_blocks_per_sm > 0
        ? static_cast<uint32_t>(occupancy.max_active_blocks_per_sm) : 0u;
}

void launchDeepRSmallK(const PhysicalKernel& kernel, const ReductionStageSemantics& semantics,
                       const StridedSpan& input, const StridedSpan& partials,
                       const ReductionGeometry& geometry, Stream& stream) {
    if (kernel.family != PhysicalFamily::DeepRSmallK || !kernel.staged() ||
        !structurallyValid(kernel, geometry.outer_size, geometry.reduction_size, geometry.inner_size))
        throw std::logic_error("Invalid retained DeepRSmallK physical stage");
    if (kernel.element_bytes == 4) {
        launchFp32DeepRSmallKFirstStage(semantics, input, partials, geometry, kernel.a,
                                        (geometry.reduction_size + kernel.a - 1) / kernel.a, stream);
        return;
    }
    if (kernel.element_bytes == 2) {
        launchF16DeepRSmallKFirstStage(semantics, input, partials, geometry, kernel.a,
                                       (geometry.reduction_size + kernel.a - 1) / kernel.a, stream);
        return;
    }
    launchFp8DeepRSmallKFirstStage(
        semantics, input, partials, geometry, kernel.a,
        (geometry.reduction_size + kernel.a - 1) / kernel.a, stream);
}

}  // namespace ThorImplementation::ReductionInternal::RK
