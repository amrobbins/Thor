#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"
#include "Utilities/TensorOperations/Cub/CubReductionOperation.cuh"

#include <cuda/std/functional>

namespace ThorImplementation::CubReductionInternal {
namespace {

size_t queryAdditiveReductionBytes(DataType input_dtype,
                                   const void* input,
                                   uint64_t input_elements,
                                   DataType output_dtype,
                                   void* output,
                                   const CubReductionGeometry& geometry,
                                   float divisor,
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
                                        IdentityFp32{},
                                        AdditiveFinalizeFp32{divisor, square_root},
                                        output_scale,
                                        stream);
}

void launchAdditiveReduction(const Tensor& temp_storage,
                             size_t temp_storage_bytes,
                             const Tensor& input,
                             Tensor& output,
                             const CubReductionGeometry& geometry,
                             float divisor,
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
                             IdentityFp32{},
                             AdditiveFinalizeFp32{divisor, square_root},
                             output_scale,
                             stream);
}

}  // namespace

size_t querySumReductionBytes(DataType input_dtype,
                              const void* input,
                              uint64_t input_elements,
                              DataType output_dtype,
                              void* output,
                              const CubReductionGeometry& geometry,
                              float output_scale,
                              const Stream& stream) {
    return queryAdditiveReductionBytes(input_dtype,
                                       input,
                                       input_elements,
                                       output_dtype,
                                       output,
                                       geometry,
                                       1.0f,
                                       false,
                                       output_scale,
                                       stream);
}

void launchSumReduction(const Tensor& temp_storage,
                        size_t temp_storage_bytes,
                        const Tensor& input,
                        Tensor& output,
                        const CubReductionGeometry& geometry,
                        float output_scale,
                        Stream& stream) {
    launchAdditiveReduction(
        temp_storage, temp_storage_bytes, input, output, geometry, 1.0f, false, output_scale, stream);
}

size_t querySumDivideReductionBytes(DataType input_dtype,
                                    const void* input,
                                    uint64_t input_elements,
                                    DataType output_dtype,
                                    void* output,
                                    const CubReductionGeometry& geometry,
                                    uint64_t divisor,
                                    float output_scale,
                                    const Stream& stream) {
    return queryAdditiveReductionBytes(input_dtype,
                                       input,
                                       input_elements,
                                       output_dtype,
                                       output,
                                       geometry,
                                       static_cast<float>(divisor),
                                       false,
                                       output_scale,
                                       stream);
}

void launchSumDivideReduction(const Tensor& temp_storage,
                              size_t temp_storage_bytes,
                              const Tensor& input,
                              Tensor& output,
                              const CubReductionGeometry& geometry,
                              uint64_t divisor,
                              float output_scale,
                              Stream& stream) {
    launchAdditiveReduction(temp_storage,
                            temp_storage_bytes,
                            input,
                            output,
                            geometry,
                            static_cast<float>(divisor),
                            false,
                            output_scale,
                            stream);
}

size_t querySumSqrtReductionBytes(DataType input_dtype,
                                  const void* input,
                                  uint64_t input_elements,
                                  DataType output_dtype,
                                  void* output,
                                  const CubReductionGeometry& geometry,
                                  float output_scale,
                                  const Stream& stream) {
    return queryAdditiveReductionBytes(input_dtype,
                                       input,
                                       input_elements,
                                       output_dtype,
                                       output,
                                       geometry,
                                       1.0f,
                                       true,
                                       output_scale,
                                       stream);
}

void launchSumSqrtReduction(const Tensor& temp_storage,
                            size_t temp_storage_bytes,
                            const Tensor& input,
                            Tensor& output,
                            const CubReductionGeometry& geometry,
                            float output_scale,
                            Stream& stream) {
    launchAdditiveReduction(
        temp_storage, temp_storage_bytes, input, output, geometry, 1.0f, true, output_scale, stream);
}

size_t queryMeanReductionBytes(DataType input_dtype,
                               const void* input,
                               uint64_t input_elements,
                               DataType output_dtype,
                               void* output,
                               const CubReductionGeometry& geometry,
                               float output_scale,
                               const Stream& stream) {
    return querySumDivideReductionBytes(input_dtype,
                                        input,
                                        input_elements,
                                        output_dtype,
                                        output,
                                        geometry,
                                        geometry.reduction_size,
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
    launchSumDivideReduction(temp_storage,
                             temp_storage_bytes,
                             input,
                             output,
                             geometry,
                             geometry.reduction_size,
                             output_scale,
                             stream);
}

}  // namespace ThorImplementation::CubReductionInternal
