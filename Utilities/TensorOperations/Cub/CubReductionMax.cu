#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"
#include "Utilities/TensorOperations/Cub/CubReductionOperation.cuh"

#include <limits>

namespace ThorImplementation::CubReductionInternal {

size_t queryMaxReductionBytes(const Tensor& input,
                                  Tensor& output,
                                  const CubReductionGeometry& geometry,
                                  const Stream& stream) {
    return queryOperationReductionBytes(input,
                                        output,
                                        geometry,
                                        PropagatingMaximumFp32{},
                                        -std::numeric_limits<float>::infinity(),
                                        IdentityFp32{},
                                        IdentityFp32{},
                                        stream);
}

void launchMaxReduction(const Tensor& temp_storage,
                            size_t temp_storage_bytes,
                            const Tensor& input,
                            Tensor& output,
                            const CubReductionGeometry& geometry,
                            Stream& stream) {
    launchOperationReduction(temp_storage,
                             temp_storage_bytes,
                             input,
                             output,
                             geometry,
                             PropagatingMaximumFp32{},
                             -std::numeric_limits<float>::infinity(),
                             IdentityFp32{},
                             IdentityFp32{},
                             stream);
}

}  // namespace ThorImplementation::CubReductionInternal
