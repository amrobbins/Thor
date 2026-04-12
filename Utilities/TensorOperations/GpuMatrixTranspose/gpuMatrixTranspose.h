#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "DeepLearning/Implementation/Tensor/Tensor.h"

namespace ThorImplementation {

void launchMatrixTransposeByType(void* output,
                                 const void* input,
                                 uint32_t numRows,
                                 uint32_t numCols,
                                 TensorDescriptor::DataType input_dtype,
                                 TensorDescriptor::DataType output_dtype,
                                 cudaStream_t stream);

void matrixTranspose(float* transposedMatrix_d, const float* matrix_d, int numRows, int numCols, cudaStream_t stream);
void matrixTranspose(__half* transposedMatrix_d, const __half* matrix_d, int numRows, int numCols, cudaStream_t stream);
void matrixTranspose(__nv_bfloat16* transposedMatrix_d, const __nv_bfloat16* matrix_d, int numRows, int numCols, cudaStream_t stream);
void matrixTranspose(__nv_fp8_e4m3* transposedMatrix_d, const __nv_fp8_e4m3* matrix_d, int numRows, int numCols, cudaStream_t stream);
void matrixTranspose(__nv_fp8_e5m2* transposedMatrix_d, const __nv_fp8_e5m2* matrix_d, int numRows, int numCols, cudaStream_t stream);

}  // namespace ThorImplementation
