#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"
#include "Utilities/TensorOperations/Cub/CubReductionOperation.cuh"

#include <cuda/std/functional>

namespace ThorImplementation::CubReductionInternal {

size_t queryMeanReductionBytes(const Tensor& input,
                                  Tensor& output,
                                  const CubReductionGeometry& geometry,
                                  const Stream& stream) {
    return queryOperationReductionBytes(input,
                                        output,
                                        geometry,
                                        cuda::std::plus<float>{},
                                        0.0f,
                                        IdentityFp32{},
                                        MeanFinalizeFp32{static_cast<float>(geometry.reduction_size)},
                                        stream);
}

void launchMeanReduction(const Tensor& temp_storage,
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
                             cuda::std::plus<float>{},
                             0.0f,
                             IdentityFp32{},
                             MeanFinalizeFp32{static_cast<float>(geometry.reduction_size)},
                             stream);
}

}  // namespace ThorImplementation::CubReductionInternal
