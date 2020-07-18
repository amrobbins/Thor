#pragma once

#include <assert.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>

void matrixTranspose(half *transposedMatrix_d, const half *matrix_d, int numRows, int numCols, cudaStream_t stream);
void matrixTranspose(float *transposedMatrix_d, const float *matrix_d, int numRows, int numCols, cudaStream_t stream);
