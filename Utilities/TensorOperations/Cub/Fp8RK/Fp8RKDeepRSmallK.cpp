#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKPhysicalKernels.h"

#include <stdexcept>

namespace ThorImplementation::CubReductionInternal::Fp8RK {

uint32_t deepRSmallKResidency(DataType input_dtype, uint64_t k,
                             const CubReductionStageSemantics& semantics,
                             uint64_t rows_per_shard) {
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
    launchFp8DeepRSmallKRetainedFirstStage(
        semantics, input, partials, geometry, kernel.a,
        (geometry.reduction_size + kernel.a - 1) / kernel.a, stream);
}

}  // namespace ThorImplementation::CubReductionInternal::Fp8RK
