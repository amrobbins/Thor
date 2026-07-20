#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"
#include "Utilities/TensorOperations/Cub/CubReductionOperation.cuh"

#include <cuda/std/functional>

namespace ThorImplementation::CubReductionInternal {

size_t queryMeanReductionBytes(DataType input_dtype,
                                  const void* input,
                                  uint64_t input_elements,
                                  DataType output_dtype,
                                  void* output,
                                  const CubReductionGeometry& geometry,
                                  float output_scale,
                                  const Stream& stream) {
    return queryOperationReductionBytes(input_dtype,
                                        input,
                                        input_elements,
                                        output_dtype,
                                        output,
                                        geometry,
                                        cuda::std::plus<float>{},
                                        0.0f,
                                        IdentityFp32{},
                                        MeanFinalizeFp32{static_cast<float>(geometry.reduction_size)},
                                        output_scale,
                                        stream);
}

void launchMeanReduction(const Tensor& temp_storage,
                            size_t temp_storage_bytes,
                            const Tensor& input,
                            Tensor& output,
                            const CubReductionGeometry& geometry,
                            float output_scale,
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
                             output_scale,
                             stream);
}

}  // namespace ThorImplementation::CubReductionInternal
