#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"
#include "Utilities/TensorOperations/Cub/CubReductionOperation.cuh"

#include <cuda/std/functional>

namespace ThorImplementation::CubReductionInternal {

size_t queryProductReductionBytes(const Tensor& input,
                                  Tensor& output,
                                  const CubReductionGeometry& geometry,
                                  const Stream& stream) {
    return queryOperationReductionBytes(input,
                                        output,
                                        geometry,
                                        cuda::std::multiplies<float>{},
                                        1.0f,
                                        IdentityFp32{},
                                        IdentityFp32{},
                                        stream);
}

void launchProductReduction(const Tensor& temp_storage,
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
                             cuda::std::multiplies<float>{},
                             1.0f,
                             IdentityFp32{},
                             IdentityFp32{},
                             stream);
}

}  // namespace ThorImplementation::CubReductionInternal
