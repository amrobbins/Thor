#include "Utilities/TensorOperations/Cub/CubArgReductionOperation.cuh"
#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"

#include <limits>

namespace ThorImplementation::CubReductionInternal {

size_t queryArgMaxReductionBytes(const Tensor& input,
                                 Tensor* value_output,
                                 Tensor* index_output,
                                 const CubReductionGeometry& geometry,
                                 const Stream& stream) {
    return queryOperationArgReductionBytes(input,
                                           value_output,
                                           index_output,
                                           geometry,
                                           ArgMaximumCandidateFp32{},
                                           ArgReductionCandidateFp32{std::numeric_limits<uint64_t>::max(),
                                                                     -std::numeric_limits<float>::infinity()},
                                           stream);
}

void launchArgMaxReduction(const Tensor& temp_storage,
                           size_t temp_storage_bytes,
                           const Tensor& input,
                           Tensor* value_output,
                           Tensor* index_output,
                           const CubReductionGeometry& geometry,
                           Stream& stream) {
    launchOperationArgReduction(temp_storage,
                                temp_storage_bytes,
                                input,
                                value_output,
                                index_output,
                                geometry,
                                ArgMaximumCandidateFp32{},
                                ArgReductionCandidateFp32{std::numeric_limits<uint64_t>::max(),
                                                          -std::numeric_limits<float>::infinity()},
                                stream);
}

}  // namespace ThorImplementation::CubReductionInternal
