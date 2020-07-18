#include "TensorCoreMatrixMultiply.h"

using namespace nvcuda;

#define A_FRAG_HEIGHT 16
#define B_FRAG_WIDTH 16
#define A_FRAG_WIDTH 16
#define B_FRAG_HEIGHT A_FRAG_WIDTH
#define A_PAD 8
#define LDA_SHARED (A_FRAG_WIDTH + A_PAD)
#define B_PAD 8
#define LDB_SHARED (B_FRAG_WIDTH + B_PAD)
#define THREADS_PER_WARP 32
#define WARPS_PER_BLOCK 8
#define HALVES_PER_FLOAT 2

bool TensorCoreMatrixMultiply::checkPreRequisitesBatch16() {
    if (THREADS_PER_WARP % B_FRAG_WIDTH != 0)
        return false;
    // The following values for SPLIT_K must be divisible by the number of warps per block:
    if (WARPS_PER_BLOCK % 2 != 0)
        return false;
    if (WARPS_PER_BLOCK % 4 != 0)
        return false;
    if (WARPS_PER_BLOCK % 8 != 0)
        return false;
    return true;
}

#define SPLIT_K 8
__global__ void tensorCoreMatrixMultiplyKernel_Arow16_Bcol16_restrictionABMult16(const half *A,
                                                                                 const half *B,
                                                                                 half *C,
                                                                                 const int32_t A_rows,
                                                                                 const int32_t A_cols,
                                                                                 const int32_t B_cols,
                                                                                 const int32_t ld_A,
                                                                                 const int32_t ld_B,
                                                                                 const int32_t ld_C) {
    __shared__ float buffer_shared[SPLIT_K * A_FRAG_HEIGHT * B_FRAG_WIDTH];

    wmma::fragment<wmma::matrix_a, A_FRAG_HEIGHT, B_FRAG_WIDTH, A_FRAG_WIDTH, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, A_FRAG_HEIGHT, B_FRAG_WIDTH, A_FRAG_WIDTH, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, A_FRAG_HEIGHT, B_FRAG_WIDTH, A_FRAG_WIDTH, float> acc_frag;

    int loadARow = blockIdx.y * A_FRAG_HEIGHT;
    int loadBCol = blockIdx.x * B_FRAG_WIDTH;
    int loadACol = threadIdx.y * A_FRAG_WIDTH;  // counts along k
    int loadBRow = threadIdx.y * A_FRAG_WIDTH;  // counts along k

    wmma::fill_fragment(acc_frag, 0.0f);

    constexpr uint32_t BLOCK_MASK = ~(SPLIT_K * A_FRAG_WIDTH - 1);
    constexpr uint32_t WARP_MASK = ~15;

#pragma unroll
    for (; (loadACol & BLOCK_MASK) < A_cols; loadACol += SPLIT_K * A_FRAG_WIDTH, loadBRow += SPLIT_K * B_FRAG_HEIGHT) {
        if ((loadACol & WARP_MASK) < A_cols) {
            wmma::load_matrix_sync(a_frag, A + loadARow * ld_A + loadACol, ld_A);
            wmma::load_matrix_sync(b_frag, B + loadBRow * ld_B + loadBCol, ld_B);
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
        __syncthreads();  // This sync improves speed by 20%, which is the only reason it is here.
    }

    // I have SPLIT_K accumulators, the sum of which is the final sum for this tile of the result matrix
    // Store them to shared in preparation for reduction
    __syncthreads();
    // Shape is A_FRAG_HEIGHTxB_FRAG_WIDTH per tile, and there are SPLIT_K tiles horizontally in shared.
    wmma::store_matrix_sync(buffer_shared + threadIdx.y * B_FRAG_WIDTH, acc_frag, SPLIT_K * B_FRAG_WIDTH, wmma::mem_row_major);
    __syncthreads();

    int reduceRow = threadIdx.y * (THREADS_PER_WARP / B_FRAG_WIDTH) + threadIdx.x / B_FRAG_WIDTH;
    int reduceCol = threadIdx.x % B_FRAG_WIDTH;
    float f = buffer_shared[reduceRow * (SPLIT_K * B_FRAG_WIDTH) + reduceCol];
#pragma unroll 7
    for (int i = 1; i < SPLIT_K; ++i)
        f += buffer_shared[reduceRow * (SPLIT_K * B_FRAG_WIDTH) + reduceCol + B_FRAG_WIDTH * i];
    half h = (half)f;

    int cRow = (loadARow + reduceRow);
    int cCol = (loadBCol + reduceCol);
    if (cRow < C_rows && cCol < C_cols) {
        C[cRow * ld_C + cCol] = h;
    }
}
#undef SPLIT_K

#define SPLIT_K 8
__global__ void tensorCoreMatrixMultiplyKernel_Arow16_Bcol16_restrictionAMult16(const half *A,
                                                                                const half *B,
                                                                                half *C,
                                                                                const int32_t A_rows,
                                                                                const int32_t A_cols,
                                                                                const int32_t B_cols,
                                                                                const int32_t ld_A,
                                                                                const int32_t ld_B,
                                                                                const int32_t ld_C) {
    __shared__ half buffer_shared[SPLIT_K * B_FRAG_WIDTH * A_FRAG_HEIGHT * HALVES_PER_FLOAT];
    float *buffer_shared_float = (float *)buffer_shared;

    wmma::fragment<wmma::matrix_a, A_FRAG_HEIGHT, B_FRAG_WIDTH, A_FRAG_WIDTH, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, A_FRAG_HEIGHT, B_FRAG_WIDTH, A_FRAG_WIDTH, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, A_FRAG_HEIGHT, B_FRAG_WIDTH, A_FRAG_WIDTH, float> acc_frag;

    int loadARow = blockIdx.y * A_FRAG_HEIGHT;
    int loadBCol = blockIdx.x * B_FRAG_WIDTH;
    int loadACol = threadIdx.y * A_FRAG_WIDTH;  // counts along k
    int loadBRow = 0;                           // counts along k

    wmma::fill_fragment(acc_frag, 0.0f);

    constexpr uint32_t MASK = ~(SPLIT_K * A_FRAG_WIDTH - 1);

#pragma unroll
    for (; (loadACol & MASK) < A_cols; loadACol += SPLIT_K * A_FRAG_WIDTH, loadBRow += SPLIT_K * B_FRAG_HEIGHT) {
        wmma::load_matrix_sync(a_frag, A + loadARow * ld_A + loadACol, ld_A);

        int bufferCol = threadIdx.x % B_FRAG_WIDTH;
#pragma unroll 4
        for (int bufferRow = threadIdx.x / B_FRAG_WIDTH + threadIdx.y * B_FRAG_HEIGHT; bufferRow < (threadIdx.y + 1) * B_FRAG_HEIGHT;
             bufferRow += (THREADS_PER_WARP / B_FRAG_WIDTH)) {
            half buff;
            if ((loadBRow + bufferRow) >= B_rows || (loadBCol + bufferCol) >= B_cols) {
                buff = half(0.0f);
            } else {
                buff = B[(loadBRow + bufferRow) * ld_B + loadBCol + bufferCol];
            }
            buffer_shared[(threadIdx.y * LDB_SHARED * B_FRAG_HEIGHT) + (bufferRow % B_FRAG_HEIGHT) * LDB_SHARED + bufferCol] = buff;
        }
        wmma::load_matrix_sync(b_frag, buffer_shared + (threadIdx.y * LDB_SHARED * B_FRAG_HEIGHT), LDB_SHARED);

        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

        __syncthreads();  // This sync improves speed by 20%, which is the only reason it is here.
    }

    // I have SPLIT_K accumulators, the sum of which is the final sum for this tile of the result matrix
    // Store them to shared in preparation for reduction
    __syncthreads();
    // Shape is A_FRAG_HEIGHTxB_FRAG_WIDTH per tile, and there are SPLIT_K tiles horizontally in shared.
    wmma::store_matrix_sync(buffer_shared_float + threadIdx.y * B_FRAG_WIDTH, acc_frag, SPLIT_K * B_FRAG_WIDTH, wmma::mem_row_major);
    __syncthreads();

    int reduceRow = threadIdx.y * (THREADS_PER_WARP / B_FRAG_WIDTH) + threadIdx.x / B_FRAG_WIDTH;
    int reduceCol = threadIdx.x % B_FRAG_WIDTH;
    float f = buffer_shared_float[reduceRow * (SPLIT_K * B_FRAG_WIDTH) + reduceCol];
#pragma unroll 7
    for (int i = 1; i < SPLIT_K; ++i)
        f += buffer_shared_float[reduceRow * (SPLIT_K * B_FRAG_WIDTH) + reduceCol + B_FRAG_WIDTH * i];
    half h = (half)f;

    int cRow = (loadARow + reduceRow);
    int cCol = (loadBCol + reduceCol);
    if (cRow < C_rows && cCol < C_cols) {
        C[cRow * ld_C + cCol] = h;
    }
}
#undef SPLIT_K

#define SPLIT_K 4
__global__ void tensorCoreMatrixMultiplyKernel_Arow32_Bcol16_restrictionAMult16(const half *A,
                                                                                const half *B,
                                                                                half *C,
                                                                                const int32_t A_rows,
                                                                                const int32_t A_cols,
                                                                                const int32_t B_cols,
                                                                                const int32_t ld_A,
                                                                                const int32_t ld_B,
                                                                                const int32_t ld_C) {
    constexpr int SHARED_HALVES = SPLIT_K * B_FRAG_WIDTH * A_FRAG_HEIGHT * HALVES_PER_FLOAT > SPLIT_K * LDB_SHARED * B_FRAG_HEIGHT
                                      ? SPLIT_K * B_FRAG_WIDTH * A_FRAG_HEIGHT * HALVES_PER_FLOAT
                                      : SPLIT_K * LDB_SHARED * B_FRAG_HEIGHT;
    __shared__ half buffer_shared_half[SHARED_HALVES];
    float *buffer_shared = (float *)buffer_shared_half;

    wmma::fragment<wmma::matrix_a, A_FRAG_HEIGHT, B_FRAG_WIDTH, A_FRAG_WIDTH, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, A_FRAG_HEIGHT, B_FRAG_WIDTH, A_FRAG_WIDTH, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, A_FRAG_HEIGHT, B_FRAG_WIDTH, A_FRAG_WIDTH, float> acc_frag;

    int threadTileCol = threadIdx.y % SPLIT_K;
    int threadTileRow = threadIdx.y / SPLIT_K;

    int loadARow = blockIdx.y * ((WARPS_PER_BLOCK / SPLIT_K) * A_FRAG_HEIGHT) + threadTileRow * A_FRAG_HEIGHT;
    int loadBCol = blockIdx.x * B_FRAG_WIDTH;
    int loadACol = threadTileCol * A_FRAG_WIDTH;   // counts along k
    int loadBRow = threadTileCol * B_FRAG_HEIGHT;  // counts along k

    wmma::fill_fragment(acc_frag, 0.0f);

    constexpr uint32_t MASK = ~(SPLIT_K * A_FRAG_WIDTH - 1);

#pragma unroll
    for (; (loadACol & MASK) < A_cols; loadACol += SPLIT_K * A_FRAG_WIDTH, loadBRow += SPLIT_K * B_FRAG_HEIGHT) {
        int bufferCol = threadIdx.x % B_FRAG_WIDTH;
#pragma unroll 2
        for (int bufferRow = threadIdx.x / B_FRAG_WIDTH + threadTileRow * (THREADS_PER_WARP / B_FRAG_WIDTH); bufferRow < B_FRAG_HEIGHT;
             bufferRow += (THREADS_PER_WARP / B_FRAG_WIDTH) * (WARPS_PER_BLOCK / SPLIT_K)) {
            half buff;
            if ((loadBRow + bufferRow) >= B_rows || (loadBCol + bufferCol) >= B_cols) {
                buff = half(0.0f);
            } else {
                buff = B[(loadBRow + bufferRow) * ld_B + loadBCol + bufferCol];
            }
            buffer_shared_half[(threadTileCol * LDB_SHARED * B_FRAG_HEIGHT) + bufferRow * LDB_SHARED + bufferCol] = buff;
        }
        __syncthreads();
        wmma::load_matrix_sync(b_frag, buffer_shared_half + (threadTileCol * LDB_SHARED * B_FRAG_HEIGHT), LDB_SHARED);
        __syncthreads();

        wmma::load_matrix_sync(a_frag, A + loadARow * ld_A + loadACol, ld_A);

        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    int reduceRow = threadIdx.y * (THREADS_PER_WARP / B_FRAG_WIDTH) + threadIdx.x / B_FRAG_WIDTH;
    int reduceCol = threadIdx.x % B_FRAG_WIDTH;
    int cRow = (blockIdx.y * A_FRAG_HEIGHT * WARPS_PER_BLOCK / SPLIT_K + reduceRow);
    int cCol = (loadBCol + reduceCol);
#pragma unroll 2
    for (int tileRow = 0; tileRow < WARPS_PER_BLOCK / SPLIT_K; ++tileRow, cRow += A_FRAG_HEIGHT) {
        // I have SPLIT_K accumulators, the sum of which is the final sum for this tile of the result matrix
        // Store them to shared in preparation for reduction
        __syncthreads();
        // Shape is A_FRAG_HEIGHTxB_FRAG_WIDTH per tile, and there are SPLIT_K tiles horizontally in shared.
        if (threadTileRow == tileRow)
            wmma::store_matrix_sync(buffer_shared + threadTileCol * B_FRAG_WIDTH, acc_frag, SPLIT_K * B_FRAG_WIDTH, wmma::mem_row_major);
        __syncthreads();

        if (cRow < C_rows && cCol < C_cols) {
            float f = buffer_shared[reduceRow * (SPLIT_K * B_FRAG_WIDTH) + reduceCol];
#pragma unroll 3
            for (int i = 1; i < SPLIT_K; ++i)
                f += buffer_shared[reduceRow * (SPLIT_K * B_FRAG_WIDTH) + reduceCol + B_FRAG_WIDTH * i];
            half h = (half)f;

            C[cRow * ld_C + cCol] = h;
        }
    }
}
#undef SPLIT_K

#define SPLIT_K 2
__global__ void tensorCoreMatrixMultiplyKernel_Arow64_Bcol16_restrictionAMult16(const half *A,
                                                                                const half *B,
                                                                                half *C,
                                                                                const int32_t A_rows,
                                                                                const int32_t A_cols,
                                                                                const int32_t B_cols,
                                                                                const int32_t ld_A,
                                                                                const int32_t ld_B,
                                                                                const int32_t ld_C) {
    constexpr int SHARED_HALVES = SPLIT_K * B_FRAG_WIDTH * A_FRAG_HEIGHT * HALVES_PER_FLOAT > SPLIT_K * LDB_SHARED * B_FRAG_HEIGHT
                                      ? SPLIT_K * B_FRAG_WIDTH * A_FRAG_HEIGHT * HALVES_PER_FLOAT
                                      : SPLIT_K * LDB_SHARED * B_FRAG_HEIGHT;
    __shared__ half buffer_shared_half[SHARED_HALVES];
    float *buffer_shared = (float *)buffer_shared_half;

    wmma::fragment<wmma::matrix_a, A_FRAG_HEIGHT, B_FRAG_WIDTH, A_FRAG_WIDTH, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, A_FRAG_HEIGHT, B_FRAG_WIDTH, A_FRAG_WIDTH, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, A_FRAG_HEIGHT, B_FRAG_WIDTH, A_FRAG_WIDTH, float> acc_frag;

    int threadTileCol = threadIdx.y % SPLIT_K;
    int threadTileRow = threadIdx.y / SPLIT_K;

    int loadARow = blockIdx.y * ((WARPS_PER_BLOCK / SPLIT_K) * A_FRAG_HEIGHT) + threadTileRow * A_FRAG_HEIGHT;
    int loadBCol = blockIdx.x * B_FRAG_WIDTH;
    int loadACol = threadTileCol * A_FRAG_WIDTH;   // counts along k
    int loadBRow = threadTileCol * B_FRAG_HEIGHT;  // counts along k

    wmma::fill_fragment(acc_frag, 0.0f);

    constexpr uint32_t MASK = ~(SPLIT_K * A_FRAG_WIDTH - 1);

#pragma unroll
    for (; (loadACol & MASK) < A_cols; loadACol += SPLIT_K * A_FRAG_WIDTH, loadBRow += SPLIT_K * B_FRAG_HEIGHT) {
        int bufferCol = threadIdx.x % B_FRAG_WIDTH;
#pragma unroll 2
        for (int bufferRow = threadIdx.x / B_FRAG_WIDTH + threadTileRow * (THREADS_PER_WARP / B_FRAG_WIDTH); bufferRow < B_FRAG_HEIGHT;
             bufferRow += (THREADS_PER_WARP / B_FRAG_WIDTH) * (WARPS_PER_BLOCK / SPLIT_K)) {
            half buff;
            if ((loadBRow + bufferRow) >= B_rows || (loadBCol + bufferCol) >= B_cols) {
                buff = half(0.0f);
            } else {
                buff = B[(loadBRow + bufferRow) * ld_B + loadBCol + bufferCol];
            }
            buffer_shared_half[(threadTileCol * LDB_SHARED * B_FRAG_HEIGHT) + bufferRow * LDB_SHARED + bufferCol] = buff;
        }
        __syncthreads();
        wmma::load_matrix_sync(b_frag, buffer_shared_half + (threadTileCol * LDB_SHARED * B_FRAG_HEIGHT), LDB_SHARED);
        __syncthreads();

        wmma::load_matrix_sync(a_frag, A + loadARow * ld_A + loadACol, ld_A);

        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    int reduceRow = threadIdx.y * (THREADS_PER_WARP / B_FRAG_WIDTH) + threadIdx.x / B_FRAG_WIDTH;
    int reduceCol = threadIdx.x % B_FRAG_WIDTH;
    int cRow = (blockIdx.y * A_FRAG_HEIGHT * WARPS_PER_BLOCK / SPLIT_K + reduceRow);
    int cCol = (loadBCol + reduceCol);
#pragma unroll 4
    for (int tileRow = 0; tileRow < WARPS_PER_BLOCK / SPLIT_K; ++tileRow, cRow += A_FRAG_HEIGHT) {
        // I have SPLIT_K accumulators, the sum of which is the final sum for this tile of the result matrix
        // Store them to shared in preparation for reduction
        __syncthreads();
        // Shape is A_FRAG_HEIGHTxB_FRAG_WIDTH per tile, and there are SPLIT_K tiles horizontally in shared.
        if (threadTileRow == tileRow)
            wmma::store_matrix_sync(buffer_shared + threadTileCol * B_FRAG_WIDTH, acc_frag, SPLIT_K * B_FRAG_WIDTH, wmma::mem_row_major);
        __syncthreads();

        if (cRow < C_rows && cCol < C_cols) {
            float f = buffer_shared[reduceRow * (SPLIT_K * B_FRAG_WIDTH) + reduceCol];
#pragma unroll 2
            for (int i = 1; i < SPLIT_K; ++i)
                f += buffer_shared[reduceRow * (SPLIT_K * B_FRAG_WIDTH) + reduceCol + B_FRAG_WIDTH * i];
            half h = (half)f;

            C[cRow * ld_C + cCol] = h;
        }
    }
}
#undef SPLIT_K

__global__ void tensorCoreMatrixMultiplyKernel_Arow128_Bcol16_restrictionAMult16(const half *A,
                                                                                 const half *B,
                                                                                 half *C,
                                                                                 const int32_t A_rows,
                                                                                 const int32_t A_cols,
                                                                                 const int32_t B_cols,
                                                                                 const int32_t ld_A,
                                                                                 const int32_t ld_B,
                                                                                 const int32_t ld_C) {
    constexpr int SHARED_HALVES = B_FRAG_WIDTH * A_FRAG_HEIGHT * HALVES_PER_FLOAT > LDB_SHARED * B_FRAG_HEIGHT
                                      ? B_FRAG_WIDTH * A_FRAG_HEIGHT * HALVES_PER_FLOAT
                                      : LDB_SHARED * B_FRAG_HEIGHT;
    __shared__ half buffer_shared_half[SHARED_HALVES];
    float *buffer_shared = (float *)buffer_shared_half;

    wmma::fragment<wmma::matrix_a, A_FRAG_HEIGHT, B_FRAG_WIDTH, A_FRAG_WIDTH, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, A_FRAG_HEIGHT, B_FRAG_WIDTH, A_FRAG_WIDTH, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, A_FRAG_HEIGHT, B_FRAG_WIDTH, A_FRAG_WIDTH, float> acc_frag;

    int loadARow = blockIdx.y * (WARPS_PER_BLOCK * A_FRAG_HEIGHT) + threadIdx.y * A_FRAG_HEIGHT;
    int loadBCol = blockIdx.x * B_FRAG_WIDTH;
    int loadACol = 0;  // counts along k
    int loadBRow = 0;  // counts along k

    wmma::fill_fragment(acc_frag, 0.0f);

#pragma unroll
    while (loadACol < A_cols) {
        // Load B_FRAG_HEIGHT rows x B_FRAG_WIDTH cols of B, this will be sent to each warp
        int bufferCol = threadIdx.x % B_FRAG_WIDTH;
        int bufferRow = threadIdx.x / B_FRAG_WIDTH + threadIdx.y * (THREADS_PER_WARP / B_FRAG_WIDTH);
        half buff;
        if ((loadBRow + bufferRow) >= B_rows || (loadBCol + bufferCol) >= B_cols) {
            buff = half(0.0f);
        } else {
            buff = B[(loadBRow + bufferRow) * ld_B + loadBCol + bufferCol];
        }
        loadBRow += B_FRAG_HEIGHT;
        __syncthreads();
        buffer_shared_half[bufferRow * LDB_SHARED + bufferCol] = buff;
        __syncthreads();

        // Each warp loads the first A_FRAG_HEIGHT rows x A_FRAG_WIDTH cols of A and multipies with B
        wmma::load_matrix_sync(a_frag, A + loadARow * ld_A + loadACol, ld_A);
        wmma::load_matrix_sync(b_frag, buffer_shared_half, LDB_SHARED);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        loadACol += A_FRAG_WIDTH;
    }

    int bufferCol = threadIdx.x % B_FRAG_WIDTH;
    int cCol = loadBCol + bufferCol;
    int bufferRow = threadIdx.x / B_FRAG_WIDTH + threadIdx.y * (THREADS_PER_WARP / B_FRAG_WIDTH);
    int cRow = blockIdx.y * (WARPS_PER_BLOCK * A_FRAG_HEIGHT) + bufferRow;
#pragma unroll 8
    for (int tileRow = 0; tileRow < WARPS_PER_BLOCK; ++tileRow) {
        __syncthreads();
        if (threadIdx.y == tileRow)
            wmma::store_matrix_sync(buffer_shared, acc_frag, B_FRAG_WIDTH, wmma::mem_row_major);
        __syncthreads();
        if (cRow >= C_rows)
            return;
        if (cCol < C_cols)
            C[cRow * ld_C + cCol] = (half)(buffer_shared[bufferRow * B_FRAG_WIDTH + bufferCol]);
        cRow += A_FRAG_HEIGHT;
    }
}

#define SPLIT_K 8
__global__ void tensorCoreMatrixMultiplyKernel_Arow16_Bcol16_noRestriction(const half *A,
                                                                           const half *B,
                                                                           half *C,
                                                                           const int32_t A_rows,
                                                                           const int32_t A_cols,
                                                                           const int32_t B_cols,
                                                                           const int32_t ld_A,
                                                                           const int32_t ld_B,
                                                                           const int32_t ld_C) {
    __shared__ half buffer_shared[SPLIT_K * B_FRAG_WIDTH * A_FRAG_HEIGHT * HALVES_PER_FLOAT];
    float *buffer_shared_float = (float *)buffer_shared;

    wmma::fragment<wmma::matrix_a, A_FRAG_HEIGHT, B_FRAG_WIDTH, A_FRAG_WIDTH, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, A_FRAG_HEIGHT, B_FRAG_WIDTH, A_FRAG_WIDTH, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, A_FRAG_HEIGHT, B_FRAG_WIDTH, A_FRAG_WIDTH, float> acc_frag;

    int loadARow = blockIdx.y * A_FRAG_HEIGHT;
    int loadBCol = blockIdx.x * B_FRAG_WIDTH;
    int loadACol = threadIdx.y * A_FRAG_WIDTH;  // counts along k
    int loadBRow = 0;                           // counts along k

    wmma::fill_fragment(acc_frag, 0.0f);

    constexpr uint32_t MASK = ~(SPLIT_K * A_FRAG_WIDTH - 1);

#pragma unroll
    for (; (loadACol & MASK) < A_cols; loadACol += SPLIT_K * A_FRAG_WIDTH, loadBRow += SPLIT_K * B_FRAG_HEIGHT) {
        int bufferCol = threadIdx.x % A_FRAG_WIDTH;
#pragma unroll 8
        for (int bufferRow = threadIdx.x / A_FRAG_WIDTH; bufferRow < A_FRAG_HEIGHT; bufferRow += THREADS_PER_WARP / A_FRAG_WIDTH) {
            half buff;
            if ((loadARow + bufferRow) >= A_rows || (loadACol + bufferCol) >= A_cols) {
                buff = half(0.0f);
            } else {
                buff = A[(loadARow + bufferRow) * ld_A + loadACol + bufferCol];
            }
            buffer_shared[(threadIdx.y * LDA_SHARED * A_FRAG_HEIGHT) + bufferRow * LDA_SHARED + bufferCol] = buff;
        }
        wmma::load_matrix_sync(a_frag, buffer_shared + threadIdx.y * LDA_SHARED * A_FRAG_HEIGHT, LDA_SHARED);
        __syncthreads();

        bufferCol = threadIdx.x % B_FRAG_WIDTH;
#pragma unroll 4
        for (int bufferRow = threadIdx.x / B_FRAG_WIDTH + threadIdx.y * B_FRAG_HEIGHT; bufferRow < (threadIdx.y + 1) * B_FRAG_HEIGHT;
             bufferRow += (THREADS_PER_WARP / B_FRAG_WIDTH)) {
            half buff;
            if ((loadBRow + bufferRow) >= B_rows || (loadBCol + bufferCol) >= B_cols) {
                buff = half(0.0f);
            } else {
                buff = B[(loadBRow + bufferRow) * ld_B + loadBCol + bufferCol];
            }
            buffer_shared[(threadIdx.y * LDB_SHARED * B_FRAG_HEIGHT) + (bufferRow % B_FRAG_HEIGHT) * LDB_SHARED + bufferCol] = buff;
        }
        wmma::load_matrix_sync(b_frag, buffer_shared + (threadIdx.y * LDB_SHARED * B_FRAG_HEIGHT), LDB_SHARED);
        __syncthreads();

        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    // I have SPLIT_K accumulators, the sum of which is the final sum for this tile of the result matrix
    // Store them to shared in preparation for reduction
    __syncthreads();
    // Shape is A_FRAG_HEIGHTxB_FRAG_WIDTH per tile, and there are SPLIT_K tiles horizontally in shared.
    wmma::store_matrix_sync(buffer_shared_float + threadIdx.y * B_FRAG_WIDTH, acc_frag, SPLIT_K * B_FRAG_WIDTH, wmma::mem_row_major);
    __syncthreads();

    int reduceRow = threadIdx.y * (THREADS_PER_WARP / B_FRAG_WIDTH) + threadIdx.x / B_FRAG_WIDTH;
    int reduceCol = threadIdx.x % B_FRAG_WIDTH;
    float f = buffer_shared_float[reduceRow * (SPLIT_K * B_FRAG_WIDTH) + reduceCol];
#pragma unroll 7
    for (int i = 1; i < SPLIT_K; ++i)
        f += buffer_shared_float[reduceRow * (SPLIT_K * B_FRAG_WIDTH) + reduceCol + B_FRAG_WIDTH * i];
    half h = (half)f;

    int cRow = (loadARow + reduceRow);
    int cCol = (loadBCol + reduceCol);
    if (cRow < C_rows && cCol < C_cols) {
        C[cRow * ld_C + cCol] = h;
    }
}
#undef SPLIT_K

#define SPLIT_K 4
__global__ void tensorCoreMatrixMultiplyKernel_Arow32_Bcol16_noRestriction(const half *A,
                                                                           const half *B,
                                                                           half *C,
                                                                           const int32_t A_rows,
                                                                           const int32_t A_cols,
                                                                           const int32_t B_cols,
                                                                           const int32_t ld_A,
                                                                           const int32_t ld_B,
                                                                           const int32_t ld_C) {
    constexpr int SHARED_HALVES_FIRST = SPLIT_K * B_FRAG_WIDTH * A_FRAG_HEIGHT * HALVES_PER_FLOAT > SPLIT_K * LDB_SHARED * B_FRAG_HEIGHT
                                            ? SPLIT_K * B_FRAG_WIDTH * A_FRAG_HEIGHT * HALVES_PER_FLOAT
                                            : SPLIT_K * LDB_SHARED * B_FRAG_HEIGHT;
    constexpr int SHARED_HALVES = SHARED_HALVES_FIRST > WARPS_PER_BLOCK * LDA_SHARED * A_FRAG_HEIGHT
                                      ? SHARED_HALVES_FIRST
                                      : WARPS_PER_BLOCK * LDA_SHARED * A_FRAG_HEIGHT;
    __shared__ half buffer_shared_half[SHARED_HALVES];
    float *buffer_shared = (float *)buffer_shared_half;

    wmma::fragment<wmma::matrix_a, A_FRAG_HEIGHT, B_FRAG_WIDTH, A_FRAG_WIDTH, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, A_FRAG_HEIGHT, B_FRAG_WIDTH, A_FRAG_WIDTH, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, A_FRAG_HEIGHT, B_FRAG_WIDTH, A_FRAG_WIDTH, float> acc_frag;

    int threadTileCol = threadIdx.y % SPLIT_K;
    int threadTileRow = threadIdx.y / SPLIT_K;

    int loadARow = blockIdx.y * ((WARPS_PER_BLOCK / SPLIT_K) * A_FRAG_HEIGHT) + threadTileRow * A_FRAG_HEIGHT;
    int loadBCol = blockIdx.x * B_FRAG_WIDTH;
    int loadACol = threadTileCol * A_FRAG_WIDTH;   // counts along k
    int loadBRow = threadTileCol * B_FRAG_HEIGHT;  // counts along k

    wmma::fill_fragment(acc_frag, 0.0f);

    constexpr uint32_t MASK = ~(SPLIT_K * A_FRAG_WIDTH - 1);

#pragma unroll
    for (; (loadACol & MASK) < A_cols; loadACol += SPLIT_K * A_FRAG_WIDTH, loadBRow += SPLIT_K * B_FRAG_HEIGHT) {
        int bufferCol = threadIdx.x % B_FRAG_WIDTH;
#pragma unroll 4
        for (int bufferRow = threadIdx.x / B_FRAG_WIDTH + threadTileRow * (THREADS_PER_WARP / B_FRAG_WIDTH); bufferRow < B_FRAG_HEIGHT;
             bufferRow += (THREADS_PER_WARP / B_FRAG_WIDTH) * (WARPS_PER_BLOCK / SPLIT_K)) {
            half buff;
            if ((loadBRow + bufferRow) >= B_rows || (loadBCol + bufferCol) >= B_cols) {
                buff = half(0.0f);
            } else {
                buff = B[(loadBRow + bufferRow) * ld_B + loadBCol + bufferCol];
            }
            buffer_shared_half[(threadTileCol * LDB_SHARED * B_FRAG_HEIGHT) + bufferRow * LDB_SHARED + bufferCol] = buff;
        }
        __syncthreads();
        wmma::load_matrix_sync(b_frag, buffer_shared_half + (threadTileCol * LDB_SHARED * B_FRAG_HEIGHT), LDB_SHARED);
        __syncthreads();

        bufferCol = threadIdx.x % A_FRAG_WIDTH;
#pragma unroll 8
        for (int bufferRow = threadIdx.x / A_FRAG_WIDTH; bufferRow < A_FRAG_HEIGHT; bufferRow += THREADS_PER_WARP / A_FRAG_WIDTH) {
            half buff;
            if ((loadARow + bufferRow) >= A_rows || (loadACol + bufferCol) >= A_cols) {
                buff = half(0.0f);
            } else {
                buff = A[(loadARow + bufferRow) * ld_A + loadACol + bufferCol];
            }
            buffer_shared_half[(threadIdx.y * LDA_SHARED * A_FRAG_HEIGHT) + bufferRow * LDA_SHARED + bufferCol] = buff;
        }
        wmma::load_matrix_sync(a_frag, buffer_shared_half + threadIdx.y * LDA_SHARED * A_FRAG_HEIGHT, LDA_SHARED);
        __syncthreads();

        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    int reduceRow = threadIdx.y * (THREADS_PER_WARP / B_FRAG_WIDTH) + threadIdx.x / B_FRAG_WIDTH;
    int reduceCol = threadIdx.x % B_FRAG_WIDTH;
    int cRow = (blockIdx.y * A_FRAG_HEIGHT * WARPS_PER_BLOCK / SPLIT_K + reduceRow);
    int cCol = (loadBCol + reduceCol);
#pragma unroll 2
    for (int tileRow = 0; tileRow < WARPS_PER_BLOCK / SPLIT_K; ++tileRow, cRow += A_FRAG_HEIGHT) {
        // I have SPLIT_K accumulators, the sum of which is the final sum for this tile of the result matrix
        // Store them to shared in preparation for reduction
        __syncthreads();
        // Shape is A_FRAG_HEIGHTxB_FRAG_WIDTH per tile, and there are SPLIT_K tiles horizontally in shared.
        if (threadTileRow == tileRow)
            wmma::store_matrix_sync(buffer_shared + threadTileCol * B_FRAG_WIDTH, acc_frag, SPLIT_K * B_FRAG_WIDTH, wmma::mem_row_major);
        __syncthreads();

        if (cRow < C_rows && cCol < C_cols) {
            float f = buffer_shared[reduceRow * (SPLIT_K * B_FRAG_WIDTH) + reduceCol];
#pragma unroll 3
            for (int i = 1; i < SPLIT_K; ++i)
                f += buffer_shared[reduceRow * (SPLIT_K * B_FRAG_WIDTH) + reduceCol + B_FRAG_WIDTH * i];
            half h = (half)f;

            C[cRow * ld_C + cCol] = h;
        }
    }
}
#undef SPLIT_K

#define SPLIT_K 2
__global__ void tensorCoreMatrixMultiplyKernel_Arow64_Bcol16_noRestriction(const half *A,
                                                                           const half *B,
                                                                           half *C,
                                                                           const int32_t A_rows,
                                                                           const int32_t A_cols,
                                                                           const int32_t B_cols,
                                                                           const int32_t ld_A,
                                                                           const int32_t ld_B,
                                                                           const int32_t ld_C) {
    constexpr int SHARED_HALVES_FIRST = SPLIT_K * B_FRAG_WIDTH * A_FRAG_HEIGHT * HALVES_PER_FLOAT > SPLIT_K * LDB_SHARED * B_FRAG_HEIGHT
                                            ? SPLIT_K * B_FRAG_WIDTH * A_FRAG_HEIGHT * HALVES_PER_FLOAT
                                            : SPLIT_K * LDB_SHARED * B_FRAG_HEIGHT;
    constexpr int SHARED_HALVES = SHARED_HALVES_FIRST > WARPS_PER_BLOCK * LDA_SHARED * A_FRAG_HEIGHT
                                      ? SHARED_HALVES_FIRST
                                      : WARPS_PER_BLOCK * LDA_SHARED * A_FRAG_HEIGHT;
    __shared__ half buffer_shared_half[SHARED_HALVES];
    float *buffer_shared = (float *)buffer_shared_half;

    wmma::fragment<wmma::matrix_a, A_FRAG_HEIGHT, B_FRAG_WIDTH, A_FRAG_WIDTH, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, A_FRAG_HEIGHT, B_FRAG_WIDTH, A_FRAG_WIDTH, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, A_FRAG_HEIGHT, B_FRAG_WIDTH, A_FRAG_WIDTH, float> acc_frag;

    int threadTileCol = threadIdx.y % SPLIT_K;
    int threadTileRow = threadIdx.y / SPLIT_K;

    int loadARow = blockIdx.y * ((WARPS_PER_BLOCK / SPLIT_K) * A_FRAG_HEIGHT) + threadTileRow * A_FRAG_HEIGHT;
    int loadBCol = blockIdx.x * B_FRAG_WIDTH;
    int loadACol = threadTileCol * A_FRAG_WIDTH;   // counts along k
    int loadBRow = threadTileCol * B_FRAG_HEIGHT;  // counts along k

    wmma::fill_fragment(acc_frag, 0.0f);

    constexpr uint32_t MASK = ~(SPLIT_K * A_FRAG_WIDTH - 1);

#pragma unroll
    for (; (loadACol & MASK) < A_cols; loadACol += SPLIT_K * A_FRAG_WIDTH, loadBRow += SPLIT_K * B_FRAG_HEIGHT) {
        int bufferCol = threadIdx.x % B_FRAG_WIDTH;
#pragma unroll 4
        for (int bufferRow = threadIdx.x / B_FRAG_WIDTH + threadTileRow * (THREADS_PER_WARP / B_FRAG_WIDTH); bufferRow < B_FRAG_HEIGHT;
             bufferRow += (THREADS_PER_WARP / B_FRAG_WIDTH) * (WARPS_PER_BLOCK / SPLIT_K)) {
            half buff;
            if ((loadBRow + bufferRow) >= B_rows || (loadBCol + bufferCol) >= B_cols) {
                buff = half(0.0f);
            } else {
                buff = B[(loadBRow + bufferRow) * ld_B + loadBCol + bufferCol];
            }
            buffer_shared_half[(threadTileCol * LDB_SHARED * B_FRAG_HEIGHT) + bufferRow * LDB_SHARED + bufferCol] = buff;
        }
        __syncthreads();
        wmma::load_matrix_sync(b_frag, buffer_shared_half + (threadTileCol * LDB_SHARED * B_FRAG_HEIGHT), LDB_SHARED);
        __syncthreads();

        bufferCol = threadIdx.x % A_FRAG_WIDTH;
#pragma unroll 8
        for (int bufferRow = threadIdx.x / A_FRAG_WIDTH; bufferRow < A_FRAG_HEIGHT; bufferRow += THREADS_PER_WARP / A_FRAG_WIDTH) {
            half buff;
            if ((loadARow + bufferRow) >= A_rows || (loadACol + bufferCol) >= A_cols) {
                buff = half(0.0f);
            } else {
                buff = A[(loadARow + bufferRow) * ld_A + loadACol + bufferCol];
            }
            buffer_shared_half[(threadIdx.y * LDA_SHARED * A_FRAG_HEIGHT) + bufferRow * LDA_SHARED + bufferCol] = buff;
        }
        wmma::load_matrix_sync(a_frag, buffer_shared_half + threadIdx.y * LDA_SHARED * A_FRAG_HEIGHT, LDA_SHARED);
        __syncthreads();

        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    int reduceRow = threadIdx.y * (THREADS_PER_WARP / B_FRAG_WIDTH) + threadIdx.x / B_FRAG_WIDTH;
    int reduceCol = threadIdx.x % B_FRAG_WIDTH;
    int cRow = (blockIdx.y * A_FRAG_HEIGHT * WARPS_PER_BLOCK / SPLIT_K + reduceRow);
    int cCol = (loadBCol + reduceCol);
#pragma unroll 2
    for (int tileRow = 0; tileRow < WARPS_PER_BLOCK / SPLIT_K; ++tileRow, cRow += A_FRAG_HEIGHT) {
        // I have SPLIT_K accumulators, the sum of which is the final sum for this tile of the result matrix
        // Store them to shared in preparation for reduction
        __syncthreads();
        // Shape is A_FRAG_HEIGHTxB_FRAG_WIDTH per tile, and there are SPLIT_K tiles horizontally in shared.
        if (threadTileRow == tileRow)
            wmma::store_matrix_sync(buffer_shared + threadTileCol * B_FRAG_WIDTH, acc_frag, SPLIT_K * B_FRAG_WIDTH, wmma::mem_row_major);
        __syncthreads();

        if (cRow < C_rows && cCol < C_cols) {
            float f = buffer_shared[reduceRow * (SPLIT_K * B_FRAG_WIDTH) + reduceCol];
#pragma unroll 3
            for (int i = 1; i < SPLIT_K; ++i)
                f += buffer_shared[reduceRow * (SPLIT_K * B_FRAG_WIDTH) + reduceCol + B_FRAG_WIDTH * i];
            half h = (half)f;

            C[cRow * ld_C + cCol] = h;
        }
    }
}
#undef SPLIT_K

__global__ void tensorCoreMatrixMultiplyKernel_Arow128_Bcol16_noRestriction(const half *A,
                                                                            const half *B,
                                                                            half *C,
                                                                            const int32_t A_rows,
                                                                            const int32_t A_cols,
                                                                            const int32_t B_cols,
                                                                            const int32_t ld_A,
                                                                            const int32_t ld_B,
                                                                            const int32_t ld_C) {
    constexpr int SHARED_HALVES_FIRST = B_FRAG_WIDTH * A_FRAG_HEIGHT * HALVES_PER_FLOAT > LDB_SHARED * B_FRAG_HEIGHT
                                            ? B_FRAG_WIDTH * A_FRAG_HEIGHT * HALVES_PER_FLOAT
                                            : LDB_SHARED * B_FRAG_HEIGHT;
    constexpr int SHARED_HALVES = SHARED_HALVES_FIRST > WARPS_PER_BLOCK * LDA_SHARED * A_FRAG_HEIGHT
                                      ? SHARED_HALVES_FIRST
                                      : WARPS_PER_BLOCK * LDA_SHARED * A_FRAG_HEIGHT;
    __shared__ half buffer_shared_half[SHARED_HALVES];
    float *buffer_shared = (float *)buffer_shared_half;

    wmma::fragment<wmma::matrix_a, A_FRAG_HEIGHT, B_FRAG_WIDTH, A_FRAG_WIDTH, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, A_FRAG_HEIGHT, B_FRAG_WIDTH, A_FRAG_WIDTH, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, A_FRAG_HEIGHT, B_FRAG_WIDTH, A_FRAG_WIDTH, float> acc_frag;

    int loadARow = blockIdx.y * (WARPS_PER_BLOCK * A_FRAG_HEIGHT) + threadIdx.y * A_FRAG_HEIGHT;
    int loadBCol = blockIdx.x * B_FRAG_WIDTH;
    int loadACol = 0;  // counts along k
    int loadBRow = 0;  // counts along k

    wmma::fill_fragment(acc_frag, 0.0f);

#pragma unroll
    while ((loadACol & 0xFFFFFFF0) < A_cols) {
        // Load B_FRAG_HEIGHT rows x B_FRAG_WIDTH cols of B, this will be sent to each warp
        int bufferCol = threadIdx.x % B_FRAG_WIDTH;
        int bufferRow = threadIdx.x / B_FRAG_WIDTH + threadIdx.y * (THREADS_PER_WARP / B_FRAG_WIDTH);
        half buff;
        if ((loadBRow + bufferRow) >= B_rows || (loadBCol + bufferCol) >= B_cols) {
            buff = half(0.0f);
        } else {
            buff = B[(loadBRow + bufferRow) * ld_B + loadBCol + bufferCol];
        }
        loadBRow += B_FRAG_HEIGHT;
        __syncthreads();
        buffer_shared_half[bufferRow * LDB_SHARED + bufferCol] = buff;
        __syncthreads();
        wmma::load_matrix_sync(b_frag, buffer_shared_half, LDB_SHARED);
        __syncthreads();

        // Each warp loads the first A_FRAG_HEIGHT rows x A_FRAG_WIDTH cols of A and multipies with B
        bufferCol = threadIdx.x % A_FRAG_WIDTH;
#pragma unroll 8
        for (int bufferRow = threadIdx.x / A_FRAG_WIDTH; bufferRow < A_FRAG_HEIGHT; bufferRow += THREADS_PER_WARP / A_FRAG_WIDTH) {
            half buff;
            if ((loadARow + bufferRow) >= A_rows || (loadACol + bufferCol) >= A_cols) {
                buff = half(0.0f);
            } else {
                buff = A[(loadARow + bufferRow) * ld_A + loadACol + bufferCol];
            }
            buffer_shared_half[(threadIdx.y * LDA_SHARED * A_FRAG_HEIGHT) + bufferRow * LDA_SHARED + bufferCol] = buff;
        }
        wmma::load_matrix_sync(a_frag, buffer_shared_half + threadIdx.y * LDA_SHARED * A_FRAG_HEIGHT, LDA_SHARED);
        __syncthreads();

        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        loadACol += A_FRAG_WIDTH;
    }

    int bufferCol = threadIdx.x % B_FRAG_WIDTH;
    int cCol = loadBCol + bufferCol;
    int bufferRow = threadIdx.x / B_FRAG_WIDTH + threadIdx.y * (THREADS_PER_WARP / B_FRAG_WIDTH);
    int cRow = blockIdx.y * (WARPS_PER_BLOCK * A_FRAG_HEIGHT) + bufferRow;
#pragma unroll 8
    for (int tileRow = 0; tileRow < WARPS_PER_BLOCK; ++tileRow) {
        __syncthreads();
        if (threadIdx.y == tileRow)
            wmma::store_matrix_sync(buffer_shared, acc_frag, B_FRAG_WIDTH, wmma::mem_row_major);
        __syncthreads();
        if (cRow >= C_rows)
            return;
        if (cCol < C_cols)
            C[cRow * ld_C + cCol] = (half)(buffer_shared[bufferRow * B_FRAG_WIDTH + bufferCol]);
        cRow += A_FRAG_HEIGHT;
    }
}

void launchTensorCoreMatrixMultiplyKernel_Arow16_Bcol16_restrictionABMult16(const half *A,
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
    dim3 gridSize((B_cols + 15) / 16, (A_rows + 15) / 16);
    tensorCoreMatrixMultiplyKernel_Arow16_Bcol16_restrictionABMult16<<<gridSize, blockSize, 0, stream.getStream()>>>(
        A, B, C, A_rows, A_cols, B_cols, ld_A, ld_B, ld_C);
}

void launchTensorCoreMatrixMultiplyKernel_Arow16_Bcol16_restrictionAMult16(const half *A,
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
    dim3 gridSize((B_cols + 15) / 16, (A_rows + 15) / 16);
    tensorCoreMatrixMultiplyKernel_Arow16_Bcol16_restrictionAMult16<<<gridSize, blockSize, 0, stream.getStream()>>>(
        A, B, C, A_rows, A_cols, B_cols, ld_A, ld_B, ld_C);
}

void launchTensorCoreMatrixMultiplyKernel_Arow32_Bcol16_restrictionAMult16(const half *A,
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
    dim3 gridSize((B_cols + 15) / 16, (A_rows + 31) / 32);
    tensorCoreMatrixMultiplyKernel_Arow32_Bcol16_restrictionAMult16<<<gridSize, blockSize, 0, stream.getStream()>>>(
        A, B, C, A_rows, A_cols, B_cols, ld_A, ld_B, ld_C);
}

void launchTensorCoreMatrixMultiplyKernel_Arow64_Bcol16_restrictionAMult16(const half *A,
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
    dim3 gridSize((B_cols + 15) / 16, (A_rows + 63) / 64);
    tensorCoreMatrixMultiplyKernel_Arow64_Bcol16_restrictionAMult16<<<gridSize, blockSize, 0, stream.getStream()>>>(
        A, B, C, A_rows, A_cols, B_cols, ld_A, ld_B, ld_C);
}

void launchTensorCoreMatrixMultiplyKernel_Arow128_Bcol16_restrictionAMult16(const half *A,
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
    dim3 gridSize((B_cols + 15) / 16, (A_rows + 127) / 128);
    tensorCoreMatrixMultiplyKernel_Arow128_Bcol16_restrictionAMult16<<<gridSize, blockSize, 0, stream.getStream()>>>(
        A, B, C, A_rows, A_cols, B_cols, ld_A, ld_B, ld_C);
}

void launchTensorCoreMatrixMultiplyKernel_Arow16_Bcol16_noRestriction(const half *A,
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
    dim3 gridSize((B_cols + 15) / 16, (A_rows + 15) / 16);
    tensorCoreMatrixMultiplyKernel_Arow16_Bcol16_noRestriction<<<gridSize, blockSize, 0, stream.getStream()>>>(
        A, B, C, A_rows, A_cols, B_cols, ld_A, ld_B, ld_C);
}

void launchTensorCoreMatrixMultiplyKernel_Arow32_Bcol16_noRestriction(const half *A,
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
    dim3 gridSize((B_cols + 15) / 16, (A_rows + 31) / 32);
    tensorCoreMatrixMultiplyKernel_Arow32_Bcol16_noRestriction<<<gridSize, blockSize, 0, stream.getStream()>>>(
        A, B, C, A_rows, A_cols, B_cols, ld_A, ld_B, ld_C);
}

void launchTensorCoreMatrixMultiplyKernel_Arow64_Bcol16_noRestriction(const half *A,
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
    dim3 gridSize((B_cols + 15) / 16, (A_rows + 63) / 64);
    tensorCoreMatrixMultiplyKernel_Arow64_Bcol16_noRestriction<<<gridSize, blockSize, 0, stream.getStream()>>>(
        A, B, C, A_rows, A_cols, B_cols, ld_A, ld_B, ld_C);
}

void launchTensorCoreMatrixMultiplyKernel_Arow128_Bcol16_noRestriction(const half *A,
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
    dim3 gridSize((B_cols + 15) / 16, (A_rows + 127) / 128);
    tensorCoreMatrixMultiplyKernel_Arow128_Bcol16_noRestriction<<<gridSize, blockSize, 0, stream.getStream()>>>(
        A, B, C, A_rows, A_cols, B_cols, ld_A, ld_B, ld_C);
}

vector<KernelWithSpec> TensorCoreMatrixMultiply::getBCol16Kernels() {
    vector<KernelWithSpec> kernels;
    KernelWithSpec kernel;

    kernel.aRowsPerBlock = 16;
    kernel.bColsPerBlock = 16;
    kernel.aRowSizeModulusRequirement = 16;
    kernel.aColSizeModulusRequirement = 16;
    kernel.bRowSizeModulusRequirement = 16;
    kernel.bColSizeModulusRequirement = 16;
    kernel.id = KernelWithSpec::KernelIndex::_16_16_ABRestrict;
    kernel.executeKernel = launchTensorCoreMatrixMultiplyKernel_Arow16_Bcol16_restrictionABMult16;
    kernels.push_back(kernel);

    kernel.aRowsPerBlock = 16;
    kernel.bColsPerBlock = 16;
    kernel.aRowSizeModulusRequirement = 16;
    kernel.aColSizeModulusRequirement = 16;
    kernel.bRowSizeModulusRequirement = 0;
    kernel.bColSizeModulusRequirement = 0;
    kernel.id = KernelWithSpec::KernelIndex::_16_16_ARestrict;
    kernel.executeKernel = launchTensorCoreMatrixMultiplyKernel_Arow16_Bcol16_restrictionAMult16;
    kernels.push_back(kernel);

    kernel.aRowsPerBlock = 32;
    kernel.bColsPerBlock = 16;
    kernel.aRowSizeModulusRequirement = 16;
    kernel.aColSizeModulusRequirement = 16;
    kernel.bRowSizeModulusRequirement = 0;
    kernel.bColSizeModulusRequirement = 0;
    kernel.id = KernelWithSpec::KernelIndex::_32_16_ARestrict;
    kernel.executeKernel = launchTensorCoreMatrixMultiplyKernel_Arow32_Bcol16_restrictionAMult16;
    kernels.push_back(kernel);

    kernel.aRowsPerBlock = 64;
    kernel.bColsPerBlock = 16;
    kernel.aRowSizeModulusRequirement = 16;
    kernel.aColSizeModulusRequirement = 16;
    kernel.bRowSizeModulusRequirement = 0;
    kernel.bColSizeModulusRequirement = 0;
    kernel.id = KernelWithSpec::KernelIndex::_64_16_ARestrict;
    kernel.executeKernel = launchTensorCoreMatrixMultiplyKernel_Arow64_Bcol16_restrictionAMult16;
    kernels.push_back(kernel);

    kernel.aRowsPerBlock = 128;
    kernel.bColsPerBlock = 16;
    kernel.aRowSizeModulusRequirement = 16;
    kernel.aColSizeModulusRequirement = 16;
    kernel.bRowSizeModulusRequirement = 0;
    kernel.bColSizeModulusRequirement = 0;
    kernel.id = KernelWithSpec::KernelIndex::_128_16_ARestrict;
    kernel.executeKernel = launchTensorCoreMatrixMultiplyKernel_Arow128_Bcol16_restrictionAMult16;
    kernels.push_back(kernel);

    kernel.aRowsPerBlock = 16;
    kernel.bColsPerBlock = 16;
    kernel.aRowSizeModulusRequirement = 0;
    kernel.aColSizeModulusRequirement = 0;
    kernel.bRowSizeModulusRequirement = 0;
    kernel.bColSizeModulusRequirement = 0;
    kernel.id = KernelWithSpec::KernelIndex::_16_16_loadToShared;
    kernel.executeKernel = launchTensorCoreMatrixMultiplyKernel_Arow16_Bcol16_noRestriction;
    kernels.push_back(kernel);

    kernel.aRowsPerBlock = 32;
    kernel.bColsPerBlock = 16;
    kernel.aRowSizeModulusRequirement = 0;
    kernel.aColSizeModulusRequirement = 0;
    kernel.bRowSizeModulusRequirement = 0;
    kernel.bColSizeModulusRequirement = 0;
    kernel.id = KernelWithSpec::KernelIndex::_32_16_loadToShared;
    kernel.executeKernel = launchTensorCoreMatrixMultiplyKernel_Arow32_Bcol16_noRestriction;
    kernels.push_back(kernel);

    kernel.aRowsPerBlock = 64;
    kernel.bColsPerBlock = 16;
    kernel.aRowSizeModulusRequirement = 0;
    kernel.aColSizeModulusRequirement = 0;
    kernel.bRowSizeModulusRequirement = 0;
    kernel.bColSizeModulusRequirement = 0;
    kernel.id = KernelWithSpec::KernelIndex::_64_16_loadToShared;
    kernel.executeKernel = launchTensorCoreMatrixMultiplyKernel_Arow64_Bcol16_noRestriction;
    kernels.push_back(kernel);

    kernel.aRowsPerBlock = 128;
    kernel.bColsPerBlock = 16;
    kernel.aRowSizeModulusRequirement = 0;
    kernel.aColSizeModulusRequirement = 0;
    kernel.bRowSizeModulusRequirement = 0;
    kernel.bColSizeModulusRequirement = 0;
    kernel.id = KernelWithSpec::KernelIndex::_128_16_loadToShared;
    kernel.executeKernel = launchTensorCoreMatrixMultiplyKernel_Arow128_Bcol16_noRestriction;
    kernels.push_back(kernel);

    return kernels;
}

#undef A_FRAG_HEIGHT
#undef B_FRAG_WIDTH
#undef A_FRAG_WIDTH
#undef B_FRAG_HEIGHT
#undef B_PAD
#undef LDB_SHARED
#undef SPLIT_K
#undef THREADS_PER_WARP
#undef WARPS_PER_BLOCK
#undef HALVES_PER_FLOAT
