#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"
#include "Utilities/TensorOperations/Cub/CubReductionOperation.cuh"

#include <cuda/std/functional>

namespace ThorImplementation::CubReductionInternal {
namespace {

size_t querySquaredAdditiveReductionBytes(DataType input_dtype,
                                          const void* input,
                                          uint64_t input_elements,
                                          DataType output_dtype,
                                          void* output,
                                          const CubReductionGeometry& geometry,
                                          bool square_root,
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
                                        SquareFp32{},
                                        AdditiveFinalizeFp32{1.0f, square_root},
                                        output_scale,
                                        stream);
}

void launchSquaredAdditiveReduction(const Tensor& temp_storage,
                                    size_t temp_storage_bytes,
                                    const Tensor& input,
                                    Tensor& output,
                                    const CubReductionGeometry& geometry,
                                    bool square_root,
                                    float output_scale,
                                    Stream& stream) {
    launchOperationReduction(temp_storage,
                             temp_storage_bytes,
                             input,
                             output,
                             geometry,
                             cuda::std::plus<float>{},
                             0.0f,
                             SquareFp32{},
                             AdditiveFinalizeFp32{1.0f, square_root},
                             output_scale,
                             stream);
}

}  // namespace

size_t querySumSquaresReductionBytes(DataType input_dtype,
                                     const void* input,
                                     uint64_t input_elements,
                                     DataType output_dtype,
                                     void* output,
                                     const CubReductionGeometry& geometry,
                                     float output_scale,
                                     const Stream& stream) {
    return querySquaredAdditiveReductionBytes(input_dtype,
                                              input,
                                              input_elements,
                                              output_dtype,
                                              output,
                                              geometry,
                                              false,
                                              output_scale,
                                              stream);
}

void launchSumSquaresReduction(const Tensor& temp_storage,
                               size_t temp_storage_bytes,
                               const Tensor& input,
                               Tensor& output,
                               const CubReductionGeometry& geometry,
                               float output_scale,
                               Stream& stream) {
    launchSquaredAdditiveReduction(
        temp_storage, temp_storage_bytes, input, output, geometry, false, output_scale, stream);
}

size_t queryL2NormReductionBytes(DataType input_dtype,
                                 const void* input,
                                 uint64_t input_elements,
                                 DataType output_dtype,
                                 void* output,
                                 const CubReductionGeometry& geometry,
                                 float output_scale,
                                 const Stream& stream) {
    return querySquaredAdditiveReductionBytes(input_dtype,
                                              input,
                                              input_elements,
                                              output_dtype,
                                              output,
                                              geometry,
                                              true,
                                              output_scale,
                                              stream);
}

void launchL2NormReduction(const Tensor& temp_storage,
                           size_t temp_storage_bytes,
                           const Tensor& input,
                           Tensor& output,
                           const CubReductionGeometry& geometry,
                           float output_scale,
                           Stream& stream) {
    launchSquaredAdditiveReduction(
        temp_storage, temp_storage_bytes, input, output, geometry, true, output_scale, stream);
}

}  // namespace ThorImplementation::CubReductionInternal
