#pragma once

#include "Utilities/TensorOperations/GpuMatrixTranspose/gpuMatrixTranspose.h"

#include <stdio.h>
#include <algorithm>
#include <vector>

#include <assert.h>
#include <cublasLt.h>
#include <cuda.h>
#include <cuda_runtime.h>

using std::vector;

// void matrixTranspose(half *transposedMatrix_d, const half *matrix_d, int numRows, int numCols, cudaStream_t stream);
// void matrixTranspose(float *transposedMatrix_d, const float *matrix_d, int numRows, int numCols, cudaStream_t
// stream);

// C_nxp = A_nxm * B_mxp
// C_mxn = A_mxk * B_kxn

enum class ElementOrder { CPP_ROW_MAJOR = 1, FORTRAN_COLUMN_MAJOR = 2 };

enum class DataType { FP32 = 1, FP16 = 2 };

/* Structure to store information about different run trials */
typedef struct {
    cublasLtMatmulAlgo_t algo;
    cublasStatus_t status;
    float time;
    size_t workspaceSize;  // actual memory workspace needed
    cublasMath_t mathMode;
    cublasLtReductionScheme_t reductionScheme;
    int customOption;
    float wavesCount;
} customMatmulPerf_t;

struct MatrixMultiplyKernelInfo {
    cublasLtMatmulDesc_t operationDescriptor;
    cublasLtMatmulAlgo_t algorithm;
    cublasLtMatrixLayout_t ADescriptor;
    cublasLtMatrixLayout_t BDescriptor;
    cublasLtMatrixLayout_t CDescriptor;
    size_t workspaceSizeInBytes;
    float time;
    cublasStatus_t status;
    int CCols;
    int CRows;
    DataType dataType;
    ElementOrder AElementOrder;
    ElementOrder BElementOrder;
    ElementOrder CElementOrder;
};

/* CAUTION : must match cublasLtMatmulTile_t */
const char *const matmulTileName[] = {
    "UNDEF", "8x8",    "8x16",  "16x8",   "8x32",   "16x16",  "32x8",   "8x64",    "16x32",  "32x16",  "64x8",    "32x32",   "32x64",
    "64x32", "32x128", "64x64", "128x32", "64x128", "128x64", "64x256", "128x128", "256x64", "64x512", "128x256", "256x128", "512x64",
};

cublasStatus_t matrixMultiply(cublasLtHandle_t cublasLtHandle,
                              cudaStream_t stream,
                              const void *A_d,
                              const void *B_d,
                              void *C_d,
                              void *workspace_d,
                              const MatrixMultiplyKernelInfo &kernelInfo,
                              void *transposeBuffer_d = NULL);

// Utility function to print customMatmulPerf_t structure
static void printPerfStructure(const customMatmulPerf_t &perf);

static inline bool time_compare(const customMatmulPerf_t &perf_a, const customMatmulPerf_t &perf_b);

static inline bool info_time_compare(const MatrixMultiplyKernelInfo &a, const MatrixMultiplyKernelInfo &b);

static cublasStatus_t customMatmulRun(cublasLtHandle_t ltHandle,  // to get the capabilities (required a GPU)
                                      cublasLtMatmulDesc_t operationDesc,
                                      const void *alpha, /* host or device pointer */
                                      const void *A,
                                      cublasLtMatrixLayout_t Adesc,
                                      const void *B,
                                      cublasLtMatrixLayout_t Bdesc,
                                      const void *beta, /* host or device pointer */
                                      const void *C,
                                      cublasLtMatrixLayout_t Cdesc,
                                      void *D,
                                      cublasLtMatrixLayout_t Ddesc,
                                      const cublasLtMatmulAlgo_t &algo,
                                      int kernelRepeats,
                                      void *workSpace,
                                      size_t workSpaceSizeInBytes,
                                      customMatmulPerf_t &perfResults,
                                      cudaStream_t stream,
                                      cudaEvent_t &startEvent,
                                      cudaEvent_t &stopEvent);

MatrixMultiplyKernelInfo LtSgemmCustomFind(cublasLtHandle_t ltHandle,
                                           cublasOperation_t transa,
                                           cublasOperation_t transb,
                                           int m,
                                           int n,
                                           int k,
                                           const float *alpha, /* host pointer */
                                           const float *A,
                                           int lda,
                                           const float *B,
                                           int ldb,
                                           const float *beta, /* host pointer */
                                           float *C,
                                           int ldc,
                                           void *workSpace,
                                           size_t workSpaceSize,
                                           DataType dataType,
                                           ElementOrder AElementOrder,
                                           ElementOrder BElementOrder,
                                           ElementOrder CElementOrder,
                                           bool printResults = false);

float randRange(float min, float max);

// Find the fastest matrix multiple kernel for the following equation:
// C_mxn = A_mxk * B_kxn
MatrixMultiplyKernelInfo getBestGemmKernel(unsigned int m,
                                           unsigned int n,
                                           unsigned int k,
                                           int deviceNum,
                                           DataType dataType,
                                           ElementOrder AElementOrder = ElementOrder::CPP_ROW_MAJOR,
                                           ElementOrder BElementOrder = ElementOrder::CPP_ROW_MAJOR,
                                           ElementOrder CElementOrder = ElementOrder::CPP_ROW_MAJOR,
                                           bool printResults = false);
