#include "TensorCoreMatrixMultiply.h"

using namespace nvcuda;

// TODO: create splitK versions of these

#define ACC_WIDTH 7
#define ACC_HEIGHT "hard coded to 1"
#define ldb_shared ((ACC_WIDTH < 8 ? 8 : ACC_WIDTH) * 16 + 8)

__global__ void tensorCoreMatrixMultiplyKernel_Arow128_Bcol112_restrictionAMult16x16(const half *A,
                                                                                     const half *B,
                                                                                     half *C,
                                                                                     const int32_t A_rows,
                                                                                     const int32_t A_cols,
                                                                                     const int32_t B_cols,
                                                                                     const int32_t ld_A,
                                                                                     const int32_t ld_B,
                                                                                     const int32_t ld_C) {
    __shared__ half buffer_shared[16 * ldb_shared];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag[ACC_WIDTH];

    const int loadARow = blockIdx.y * 128 + threadIdx.y * 16;
    int loadACol = 0;                                                       // counts along k
    const int loadBCol = blockIdx.x * ACC_WIDTH * 16 + (threadIdx.x % 16);  // iterates to fill ACC_WIDTH*16
    const int loadBRowShared = threadIdx.y * 2 + threadIdx.x / 16;
    int loadBRow = loadBRowShared;  // counts along k

#pragma unroll 8
    for (int j = 0; j < ACC_WIDTH; ++j) {
        wmma::fill_fragment(acc_frag[j], 0.0f);
    }

#pragma unroll 2
    for (; loadACol < A_cols; loadACol += 16, loadBRow += 16) {
        if (loadARow < A_rows)
            wmma::load_matrix_sync(a_frag, A + loadARow * ld_A + loadACol, ld_A);

        __syncthreads();
#pragma unroll 7
        for (int bColOffset = 0; bColOffset < ACC_WIDTH * 16; bColOffset += 16) {
            // Breaking out early kills performance:
            // if((loadBRow&0xFFFFFFF0) >= A_cols || ((loadBCol + bColOffset)&0xFFFFFFF0) >= B_cols)
            //    break;

            half buff;
            if (loadBRow >= B_rows || loadBCol + bColOffset >= B_cols) {
                buff = half(0.0f);
            } else {
                buff = B[loadBRow * ld_B + (loadBCol + bColOffset)];
            }
            buffer_shared[loadBRowShared * ldb_shared + ((threadIdx.x % 16) + bColOffset)] = buff;
        }
        __syncthreads();

        if (loadARow < A_rows) {
#pragma unroll 7
            for (int bColTile = 0; bColTile < ACC_WIDTH; ++bColTile) {
                // Try breaking out early here.
                wmma::load_matrix_sync(b_frag, buffer_shared + bColTile * 16, ldb_shared);
                wmma::mma_sync(acc_frag[bColTile], a_frag, b_frag, acc_frag[bColTile]);
            }
        }
    }

    // Convert to half and write the result to memory
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> acc_frag_half;
    const int CSharedRow = threadIdx.y * 2 + threadIdx.x / 16;
    const int CSharedCol = threadIdx.x % 16;
    const int CColStart = (blockIdx.x * ACC_WIDTH) * 16 + CSharedCol;
    int CCol = CColStart;
    const int CRowStart = blockIdx.y * 128 + CSharedRow;
#pragma unroll 7
    for (int acc = 0; acc < ACC_WIDTH; ++acc) {
#pragma unroll 8
        for (int8_t i = 0; i < acc_frag_half.num_elements; ++i)
            acc_frag_half.x[i] = (half)acc_frag[acc].x[i];

        __syncthreads();
        wmma::store_matrix_sync(buffer_shared + threadIdx.y * 16, acc_frag_half, ldb_shared, wmma::mem_row_major);
        __syncthreads();

        if (CCol >= C_cols)
            break;

        // keeps ARow and BCol
        int CRow = CRowStart;
#pragma unroll 8
        for (int cRowTile = 0; cRowTile < 8 * 16; cRowTile += 16) {
            if (CRow >= C_rows)
                break;

            C[CRow * ld_C + CCol] = buffer_shared[CSharedRow * ldb_shared + CSharedCol + cRowTile];
            CRow += 16;
        }
        CCol += 16;
    }
}

#undef ACC_HEIGHT
#undef ACC_WIDTH
#undef ldb_shared

#define ACC_WIDTH 7
#define ACC_HEIGHT "hard coded to 1"
#define ldb_shared ((ACC_WIDTH < 8 ? 8 : ACC_WIDTH) * 16 + 8)
#define lda_shared (8 * 16 + 8)

__global__ void tensorCoreMatrixMultiplyKernel_Arow128_Bcol112_noRestriction(const half *A,
                                                                             const half *B,
                                                                             half *C,
                                                                             const int32_t A_rows,
                                                                             const int32_t A_cols,
                                                                             const int32_t B_cols,
                                                                             const int32_t ld_A,
                                                                             const int32_t ld_B,
                                                                             const int32_t ld_C) {
    __shared__ half buffer_shared[16 * ldb_shared];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag[ACC_WIDTH];

    // const int loadARow = blockIdx.y * 128 + threadIdx.y * 16;
    // int loadACol = 0;
    // counts along k
    const int loadBCol = blockIdx.x * ACC_WIDTH * 16 + (threadIdx.x % 16);  // iterates to fill ACC_WIDTH*16
    const int loadBRowShared = threadIdx.y * 2 + threadIdx.x / 16;
    int loadBRow = loadBRowShared;  // counts along k

    const int loadARowShared = threadIdx.y * 2 + threadIdx.x / 16;
    const int loadARow = blockIdx.y * 128 + loadARowShared;
    const int loadAColShared = threadIdx.x % 16;
    int loadACol = loadAColShared;

#pragma unroll 8
    for (int j = 0; j < ACC_WIDTH; ++j) {
        wmma::fill_fragment(acc_frag[j], 0.0f);
    }

#pragma unroll 2
    for (; (loadACol & 0xFFFFFFF0) < A_cols; loadACol += 16, loadBRow += 16) {
#pragma unroll 8
        for (int aRowOffset = 0; aRowOffset < 8 * 16; aRowOffset += 16) {
            // Breaking out early kills performance:
            // if((loadACol&0xFFFFFFF0) >= A_cols || ((loadARow + aRowOffset)&0xFFFFFFF0) >= B_cols)
            //    break;

            half buff;
            if (loadACol >= A_cols || loadARow + aRowOffset >= A_rows) {
                buff = half(0.0f);
            } else {
                buff = A[(loadARow + aRowOffset) * ld_A + loadACol];
            }
            buffer_shared[loadARowShared * lda_shared + aRowOffset + loadAColShared] = buff;
        }
        __syncthreads();

        wmma::load_matrix_sync(a_frag, buffer_shared + threadIdx.y * 16, ldb_shared);
        __syncthreads();

#pragma unroll 7
        for (int bColOffset = 0; bColOffset < ACC_WIDTH * 16; bColOffset += 16) {
            // Breaking out early kills performance:
            // if((loadBRow&0xFFFFFFF0) >= A_cols || ((loadBCol + bColOffset)&0xFFFFFFF0) >= B_cols)
            //    break;

            half buff;
            if (loadBRow >= B_rows || loadBCol + bColOffset >= B_cols) {
                buff = half(0.0f);
            } else {
                buff = B[loadBRow * ld_B + (loadBCol + bColOffset)];
            }
            buffer_shared[loadBRowShared * ldb_shared + ((threadIdx.x % 16) + bColOffset)] = buff;
        }
        __syncthreads();

        if ((loadARow & 0xFFFFFFF0) < A_rows) {
#pragma unroll 7
            for (int bColTile = 0; bColTile < ACC_WIDTH; ++bColTile) {
                // Try breaking out early here.
                wmma::load_matrix_sync(b_frag, buffer_shared + bColTile * 16, ldb_shared);
                wmma::mma_sync(acc_frag[bColTile], a_frag, b_frag, acc_frag[bColTile]);
            }
        }
        __syncthreads();
    }

    // Convert to half and write the result to memory
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> acc_frag_half;
    const int CSharedRow = threadIdx.y * 2 + threadIdx.x / 16;
    const int CSharedCol = threadIdx.x % 16;
    const int CColStart = (blockIdx.x * ACC_WIDTH) * 16 + CSharedCol;
    int CCol = CColStart;
    const int CRowStart = blockIdx.y * 128 + CSharedRow;
#pragma unroll 7
    for (int acc = 0; acc < ACC_WIDTH; ++acc) {
#pragma unroll 8
        for (int8_t i = 0; i < acc_frag_half.num_elements; ++i)
            acc_frag_half.x[i] = (half)acc_frag[acc].x[i];

        __syncthreads();
        wmma::store_matrix_sync(buffer_shared + threadIdx.y * 16, acc_frag_half, ldb_shared, wmma::mem_row_major);
        __syncthreads();

        if (CCol >= C_cols)
            break;

        // keeps ARow and BCol
        int CRow = CRowStart;
#pragma unroll 8
        for (int cRowTile = 0; cRowTile < 8 * 16; cRowTile += 16) {
            if (CRow >= C_rows)
                break;

            C[CRow * ld_C + CCol] = buffer_shared[CSharedRow * ldb_shared + CSharedCol + cRowTile];
            CRow += 16;
        }
        CCol += 16;
    }
}

#undef ACC_HEIGHT
#undef ACC_WIDTH
#undef ldb_shared
#undef lda_shared

void launchTensorCoreMatrixMultiplyKernel_Arow128_Bcol112_restrictionAMult16x16(const half *A,
                                                                                const half *B,
                                                                                half *C,
                                                                                half *workspace,
                                                                                const int32_t A_rows,
                                                                                const int32_t A_cols,
                                                                                const int32_t B_cols,
                                                                                const int32_t ld_A,
                                                                                const int32_t ld_B,
                                                                                const int32_t ld_C,
                                                                                Stream stream) {
    dim3 blockSize(32, 8);
    dim3 gridSize((B_cols + 111) / 112, (A_rows + 127) / 128);
    tensorCoreMatrixMultiplyKernel_Arow128_Bcol112_restrictionAMult16x16<<<gridSize, blockSize, 0, stream.getStream()>>>(
        A, B, C, A_rows, A_cols, B_cols, ld_A, ld_B, ld_C);
}

void launchTensorCoreMatrixMultiplyKernel_Arow128_Bcol112_noRestriction(const half *A,
                                                                        const half *B,
                                                                        half *C,
                                                                        half *workspace,
                                                                        const int32_t A_rows,
                                                                        const int32_t A_cols,
                                                                        const int32_t B_cols,
                                                                        const int32_t ld_A,
                                                                        const int32_t ld_B,
                                                                        const int32_t ld_C,
                                                                        Stream stream) {
    dim3 blockSize(32, 8);
    dim3 gridSize((B_cols + 111) / 112, (A_rows + 127) / 128);
    tensorCoreMatrixMultiplyKernel_Arow128_Bcol112_noRestriction<<<gridSize, blockSize, 0, stream.getStream()>>>(
        A, B, C, A_rows, A_cols, B_cols, ld_A, ld_B, ld_C);
}

vector<KernelWithSpec> TensorCoreMatrixMultiply::getBCol112Kernels() {
    vector<KernelWithSpec> kernels;
    KernelWithSpec kernel;

    kernel.aRowsPerBlock = 128;
    kernel.bColsPerBlock = 112;
    kernel.aRowSizeModulusRequirement = 16;
    kernel.aColSizeModulusRequirement = 16;
    kernel.bRowSizeModulusRequirement = 0;
    kernel.bColSizeModulusRequirement = 0;
    kernel.id = KernelWithSpec::KernelIndex::_128_112_slimSharedBlockA16Restrict;
    kernel.executeKernel = launchTensorCoreMatrixMultiplyKernel_Arow128_Bcol112_restrictionAMult16x16;
    kernels.push_back(kernel);

    kernel.aRowsPerBlock = 128;
    kernel.bColsPerBlock = 112;
    kernel.aRowSizeModulusRequirement = 0;
    kernel.aColSizeModulusRequirement = 0;
    kernel.bRowSizeModulusRequirement = 0;
    kernel.bColSizeModulusRequirement = 0;
    kernel.id = KernelWithSpec::KernelIndex::_128_112_slimSharedBlock;
    kernel.executeKernel = launchTensorCoreMatrixMultiplyKernel_Arow128_Bcol112_noRestriction;
    kernels.push_back(kernel);

    return kernels;
}
