#include "TensorCoreMatrixMultiply.h"

using namespace nvcuda;

#define ACC_WIDTH 3
#define ACC_HEIGHT 3

__global__ void tensorCoreMatrixMultiplyKernel_Arow48_Bcol48_restrictionABMult16x16_loadToReg(const half *A,
                                                                                              const half *B,
                                                                                              half *C,
                                                                                              const int32_t A_rows,
                                                                                              const int32_t A_cols,
                                                                                              const int32_t B_cols,
                                                                                              const int32_t ld_A,
                                                                                              const int32_t ld_B,
                                                                                              const int32_t ld_C) {
    __shared__ half buffer_shared[8][ACC_WIDTH * 16 * (16 + 8)];  // 8*1024*2 = 16KB
    float *buffer_shared_float = (float *)buffer_shared;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[ACC_WIDTH];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag[ACC_HEIGHT][ACC_WIDTH];

    int loadARow = blockIdx.y * ACC_HEIGHT * 16;
    int loadBCol = blockIdx.x * ACC_WIDTH * 16;
    int loadACol = threadIdx.y * 16;  // counts along k
    int loadBRow = threadIdx.y * 16;  // counts along k

    for (int aTile = 0; aTile < ACC_HEIGHT; ++aTile) {
        for (int bTile = 0; bTile < ACC_WIDTH; ++bTile) {
            wmma::fill_fragment(acc_frag[aTile][bTile], 0.0f);
        }
    }

#pragma unroll
    for (; (loadACol & 0xFFFFFF80) < A_cols; loadACol += 128, loadBRow += 128) {
        if (loadACol < A_cols) {
#pragma unroll
            for (int bTile = 0; bTile < ACC_WIDTH; ++bTile) {
                if (loadBCol + bTile * 16 >= B_cols)
                    break;
                wmma::load_matrix_sync(b_frag[bTile], B + loadBRow * ld_B + loadBCol + bTile * 16, ld_B);
            }

#pragma unroll
            for (int aRowOffset = 0, aTile = 0; aTile < ACC_HEIGHT; ++aTile, aRowOffset += 16) {
                if (loadARow + aRowOffset < A_rows) {
                    // load A from main mem to frag
                    wmma::load_matrix_sync(a_frag, A + (loadARow + aRowOffset) * ld_A + loadACol, ld_A);
#pragma unroll
                    for (int bTile = 0; bTile < ACC_WIDTH; ++bTile) {
                        if (loadBCol + bTile * 16 >= B_cols)
                            break;
                        wmma::mma_sync(acc_frag[aTile][bTile], a_frag, b_frag[bTile], acc_frag[aTile][bTile]);
                    }
                }
            }
        }
        __syncthreads();  // improves performance
    }

    // Each of the 8 warps in the block have ACC_WIDTH accumulators, the sum of which is the final sum for this tile of the result
    // matrix Store them to shared in preparation for reduction
#pragma unroll
    for (int aTile = 0; aTile < ACC_HEIGHT; ++aTile) {
#pragma unroll
        for (int bTile = 0; bTile < ACC_WIDTH; ++bTile) {
            if (bTile * 16 + loadBCol >= B_cols)
                break;

            __syncthreads();
            wmma::store_matrix_sync(
                &(buffer_shared_float[threadIdx.y * (ACC_WIDTH * 16 * (8 + 4))]), acc_frag[aTile][bTile], (16 + 4), wmma::mem_row_major);
            __syncthreads();

            int reduceRow = threadIdx.y * 2 + threadIdx.x / 16;
            int reduceCol = threadIdx.x % 16;
            float f = buffer_shared_float[reduceRow * (16 + 4) + reduceCol];
#pragma unroll
            for (int i = 1; i < 8; ++i) {
                f += buffer_shared_float[i * (ACC_WIDTH * 16 * (8 + 4)) + reduceRow * (16 + 4) + reduceCol];
            }
            half h = (half)f;

            // if(threadIdx.y == 1) printf("t.y = 1, C[%d, %d] = %f\n", (loadARow + 16 * aTile + reduceRow), (loadBCol + bTile * 16 +
            // reduceCol), float(h));
            C[(loadARow + 16 * aTile + reduceRow) * ld_C + (loadBCol + bTile * 16 + reduceCol)] = h;
        }
    }
}

__global__ void tensorCoreMatrixMultiplyKernel_Arow48_Bcol48_restrictionAMult16x16_loadToReg(const half *A,
                                                                                             const half *B,
                                                                                             half *C,
                                                                                             const int32_t A_rows,
                                                                                             const int32_t A_cols,
                                                                                             const int32_t B_cols,
                                                                                             const int32_t ld_A,
                                                                                             const int32_t ld_B,
                                                                                             const int32_t ld_C) {
    __shared__ half buffer_shared[8][ACC_WIDTH * 16 * (16 + 8)];  // 8*1024*2 = 16KB
    float *buffer_shared_float = (float *)buffer_shared;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[ACC_WIDTH];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag[ACC_HEIGHT][ACC_WIDTH];

    int loadARow = blockIdx.y * ACC_HEIGHT * 16;
    int loadBCol = blockIdx.x * ACC_WIDTH * 16;
    int loadACol = threadIdx.y * 16;  // counts along k
    int loadBRow = threadIdx.y * 16;  // counts along k

    for (int aTile = 0; aTile < ACC_HEIGHT; ++aTile) {
        for (int bTile = 0; bTile < ACC_WIDTH; ++bTile) {
            wmma::fill_fragment(acc_frag[aTile][bTile], 0.0f);
        }
    }

#pragma unroll 8
    for (; (loadACol & 0xFFFFFF80) < A_cols; loadACol += 128, loadBRow += 128) {
        if (loadACol < A_cols) {
            // Load B to shared
#pragma unroll
            for (int bRowOffset = threadIdx.x / 16; bRowOffset < 16; bRowOffset += 2) {
                if (((loadBRow + bRowOffset) & 0xFFFFFFF0) >= B_rows)
                    break;
                for (int bColOffset = threadIdx.x % 16; bColOffset < ACC_WIDTH * 16; bColOffset += 16) {
                    half h;
                    if ((loadBRow + bRowOffset) >= B_rows || (loadBCol + bColOffset) >= B_cols) {
                        // this causes slowdown:
                        // if (((loadBCol + bColOffset) & 0xFFFFFFF0) >= B_cols) {
                        //    break;
                        //} else {
                        //    buffer_shared[threadIdx.y][bRowOffset * (16 * ACC_WIDTH + 8) + bColOffset] = (half)0.0f;
                        //}
                        h = (half)0.0f;
                    } else {
                        h = B[(loadBRow + bRowOffset) * ld_B + (loadBCol + bColOffset)];
                    }
                    buffer_shared[threadIdx.y][bRowOffset * (16 * ACC_WIDTH + 8) + bColOffset] = h;
                }
            }

#pragma unroll
            for (int bTile = 0; bTile < ACC_WIDTH; ++bTile) {
                if (loadBCol + bTile * 16 >= B_cols)
                    break;
                wmma::load_matrix_sync(b_frag[bTile], &(buffer_shared[threadIdx.y][bTile * 16]), (16 * ACC_WIDTH + 8));
            }

            /*
            if(threadIdx.y == 0 && threadIdx.x == 0) {
                printf("A GPU:\n");
                for(int i = 0; i < 16; ++i) {
                    for(int j = 0; j < 16; ++j) {
                        printf("%5.2f ", (float)A_shared[threadIdx.y][i*24 + j]);
                    }
                    printf("\n");
                }
                printf("\n");
            }
            __syncthreads();
            */

            /*
            if(threadIdx.y == 1 && threadIdx.x == 0) {
                printf("B GPU:\n");
                for(int i = 0; i < 16; ++i) {
                    for(int j = 0; j < 16; ++j) {
                        printf("%5.2f ", (float)buffer_shared[threadIdx.y][i*(16 * ACC_WIDTH + 8) + j]);
                    }
                    printf("\n");
                }
                printf("\n");
            }
            __syncthreads();
            */

#pragma unroll
            for (int aRowOffset = 0, aTile = 0; aTile < ACC_HEIGHT; ++aTile, aRowOffset += 16) {
                if (loadARow + aRowOffset < A_rows) {
                    // load A from main mem to frag
                    wmma::load_matrix_sync(a_frag, A + (loadARow + aRowOffset) * ld_A + loadACol, ld_A);
#pragma unroll
                    for (int bTile = 0; bTile < ACC_WIDTH; ++bTile) {
                        if (loadBCol + bTile * 16 >= B_cols)
                            break;
                        wmma::mma_sync(acc_frag[aTile][bTile], a_frag, b_frag[bTile], acc_frag[aTile][bTile]);
                    }
                }
            }
        }
        __syncthreads();  // improves performance
    }

    // Each of the 8 warps in the block have ACC_WIDTH accumulators, the sum of which is the final sum for this tile of the result
    // matrix Store them to shared in preparation for reduction
#pragma unroll
    for (int aTile = 0; aTile < ACC_HEIGHT; ++aTile) {
#pragma unroll
        for (int bTile = 0; bTile < ACC_WIDTH; ++bTile) {
            if (bTile * 16 + loadBCol >= B_cols)
                break;

            __syncthreads();
            wmma::store_matrix_sync(
                &(buffer_shared_float[threadIdx.y * (ACC_WIDTH * 16 * (8 + 4))]), acc_frag[aTile][bTile], (16 + 4), wmma::mem_row_major);
            __syncthreads();

            int reduceRow = threadIdx.y * 2 + threadIdx.x / 16;
            int reduceCol = threadIdx.x % 16;
            float f = buffer_shared_float[reduceRow * (16 + 4) + reduceCol];
#pragma unroll
            for (int i = 1; i < 8; ++i) {
                f += buffer_shared_float[i * (ACC_WIDTH * 16 * (8 + 4)) + reduceRow * (16 + 4) + reduceCol];
            }
            half h = (half)f;

            int CRow = loadARow + 16 * aTile + reduceRow;
            int CCol = loadBCol + bTile * 16 + reduceCol;

            // if(threadIdx.y == 0) printf("t.y = 1, C[%d, %d] = %f\n", CRow, CCol, float(h));
            if (CRow < C_rows && CCol < C_cols) {
                C[CRow * ld_C + CCol] = h;
            }
        }
    }
}

__global__ void tensorCoreMatrixMultiplyKernel_Arow48_Bcol48_noRestriction_loadToReg(const half *A,
                                                                                     const half *B,
                                                                                     half *C,
                                                                                     const int32_t A_rows,
                                                                                     const int32_t A_cols,
                                                                                     const int32_t B_cols,
                                                                                     const int32_t ld_A,
                                                                                     const int32_t ld_B,
                                                                                     const int32_t ld_C) {
    __shared__ half buffer_shared[8][ACC_WIDTH * 16 * (16 + 8)];  // 8*1024*2 = 16KB
    float *buffer_shared_float = (float *)buffer_shared;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[ACC_WIDTH];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag[ACC_HEIGHT][ACC_WIDTH];

    int loadARow = blockIdx.y * ACC_HEIGHT * 16;
    int loadBCol = blockIdx.x * ACC_WIDTH * 16;
    int loadACol = threadIdx.y * 16;  // counts along k
    int loadBRow = threadIdx.y * 16;  // counts along k

    for (int aTile = 0; aTile < ACC_HEIGHT; ++aTile) {
        for (int bTile = 0; bTile < ACC_WIDTH; ++bTile) {
            wmma::fill_fragment(acc_frag[aTile][bTile], 0.0f);
        }
    }

#pragma unroll 8
    for (; (loadACol & 0xFFFFFF80) < A_cols; loadACol += 128, loadBRow += 128) {
        if (loadACol < A_cols) {
            // Load B to shared
#pragma unroll
            for (int bRowOffset = threadIdx.x / 16; bRowOffset < 16; bRowOffset += 2) {
                if (((loadBRow + bRowOffset) & 0xFFFFFFF0) >= B_rows)
                    break;
                for (int bColOffset = threadIdx.x % 16; bColOffset < ACC_WIDTH * 16; bColOffset += 16) {
                    half h;
                    if ((loadBRow + bRowOffset) >= B_rows || (loadBCol + bColOffset) >= B_cols) {
                        // this causes slowdown:
                        // if (((loadBCol + bColOffset) & 0xFFFFFFF0) >= B_cols) {
                        //    break;
                        //} else {
                        //    buffer_shared[threadIdx.y][bRowOffset * (16 * ACC_WIDTH + 8) + bColOffset] = (half)0.0f;
                        //}
                        h = (half)0.0f;
                    } else {
                        h = B[(loadBRow + bRowOffset) * ld_B + (loadBCol + bColOffset)];
                    }
                    buffer_shared[threadIdx.y][bRowOffset * (16 * ACC_WIDTH + 8) + bColOffset] = h;
                }
            }

#pragma unroll
            for (int bTile = 0; bTile < ACC_WIDTH; ++bTile) {
                if (loadBCol + bTile * 16 >= B_cols)
                    break;
                wmma::load_matrix_sync(b_frag[bTile], &(buffer_shared[threadIdx.y][bTile * 16]), (16 * ACC_WIDTH + 8));
            }

            /*
            if(threadIdx.y == 1 && threadIdx.x == 0) {
                printf("B GPU:\n");
                for(int i = 0; i < 16; ++i) {
                    for(int j = 0; j < 16; ++j) {
                        printf("%5.2f ", (float)buffer_shared[threadIdx.y][i*(16 * ACC_WIDTH + 8) + j]);
                    }
                    printf("\n");
                }
                printf("\n");
            }
            __syncthreads();
            */

#pragma unroll
            for (int aRowOffset = 0, aTile = 0; aTile < ACC_HEIGHT; ++aTile, aRowOffset += 16) {
                // load A from main mem to frag
                int bufferCol = threadIdx.x % 16;
                int bufferRow = threadIdx.x / 16;
                int curARow = loadARow + aRowOffset + bufferRow;
                int curACol = loadACol + bufferCol;
#pragma unroll 8
                for (int i = 0; i < 16; i += 2) {
                    half h;
                    if (curARow < A_rows && curACol < A_cols)
                        h = A[curARow * ld_A + curACol];
                    else
                        h = half(0.0f);
                    buffer_shared[threadIdx.y][bufferRow * (16 + 8) + bufferCol] = h;
                    curARow += 2;
                    bufferRow += 2;
                }
                wmma::load_matrix_sync(a_frag, buffer_shared[threadIdx.y], (16 + 8));

                /*
                if(threadIdx.y == 0 && threadIdx.x == 0) {
                    printf("A GPU:\n");
                    for(int i = 0; i < 16; ++i) {
                        for(int j = 0; j < 16; ++j) {
                            printf("%5.2f ", (float)buffer_shared[threadIdx.y][i*(16+8) + j]);
                        }
                        printf("\n");
                    }
                    printf("\n");
                }
                __syncthreads();
                */

#pragma unroll
                for (int bTile = 0; bTile < ACC_WIDTH; ++bTile) {
                    if (loadBCol + bTile * 16 >= B_cols)
                        break;
                    wmma::mma_sync(acc_frag[aTile][bTile], a_frag, b_frag[bTile], acc_frag[aTile][bTile]);
                }
            }
        }
        __syncthreads();  // improves performance
    }

    // Each of the 8 warps in the block have ACC_WIDTH accumulators, the sum of which is the final sum for this tile of the result matrix
    // Store them to shared in preparation for reduction
#pragma unroll
    for (int aTile = 0; aTile < ACC_HEIGHT; ++aTile) {
#pragma unroll
        for (int bTile = 0; bTile < ACC_WIDTH; ++bTile) {
            if (bTile * 16 + loadBCol >= B_cols)
                break;

            __syncthreads();
            wmma::store_matrix_sync(
                &(buffer_shared_float[threadIdx.y * (ACC_WIDTH * 16 * (8 + 4))]), acc_frag[aTile][bTile], (16 + 4), wmma::mem_row_major);
            __syncthreads();

            int reduceRow = threadIdx.y * 2 + threadIdx.x / 16;
            int reduceCol = threadIdx.x % 16;
            float f = buffer_shared_float[reduceRow * (16 + 4) + reduceCol];
#pragma unroll
            for (int i = 1; i < 8; ++i) {
                f += buffer_shared_float[i * (ACC_WIDTH * 16 * (8 + 4)) + reduceRow * (16 + 4) + reduceCol];
            }
            half h = (half)f;

            int CRow = loadARow + 16 * aTile + reduceRow;
            int CCol = loadBCol + bTile * 16 + reduceCol;

            // if(threadIdx.y == 0) printf("t.y = 1, C[%d, %d] = %f\n", CRow, CCol, float(h));
            if (CRow < C_rows && CCol < C_cols) {
                C[CRow * ld_C + CCol] = h;
            }
        }
    }
}

#undef ACC_WIDTH
#undef ACC_HEIGHT

#define ACC_WIDTH 3
#define ACC_HEIGHT 2

__global__ void tensorCoreMatrixMultiplyKernel_Arow32_Bcol48_restrictionABMult16x16_loadToReg(const half *A,
                                                                                              const half *B,
                                                                                              half *C,
                                                                                              const int32_t A_rows,
                                                                                              const int32_t A_cols,
                                                                                              const int32_t B_cols,
                                                                                              const int32_t ld_A,
                                                                                              const int32_t ld_B,
                                                                                              const int32_t ld_C) {
    __shared__ half buffer_shared[8][ACC_WIDTH * 16 * (16 + 8)];  // 8*1024*2 = 16KB
    float *buffer_shared_float = (float *)buffer_shared;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[ACC_WIDTH];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag[ACC_HEIGHT][ACC_WIDTH];

    int loadARow = blockIdx.y * ACC_HEIGHT * 16;
    int loadBCol = blockIdx.x * ACC_WIDTH * 16;
    int loadACol = threadIdx.y * 16;  // counts along k
    int loadBRow = threadIdx.y * 16;  // counts along k

    for (int aTile = 0; aTile < ACC_HEIGHT; ++aTile) {
        for (int bTile = 0; bTile < ACC_WIDTH; ++bTile) {
            wmma::fill_fragment(acc_frag[aTile][bTile], 0.0f);
        }
    }

#pragma unroll
    for (; (loadACol & 0xFFFFFF80) < A_cols; loadACol += 128, loadBRow += 128) {
        if (loadACol < A_cols) {
#pragma unroll
            for (int bTile = 0; bTile < ACC_WIDTH; ++bTile) {
                if (loadBCol + bTile * 16 >= B_cols)
                    break;
                wmma::load_matrix_sync(b_frag[bTile], B + loadBRow * ld_B + loadBCol + bTile * 16, ld_B);
            }

#pragma unroll
            for (int aRowOffset = 0, aTile = 0; aTile < ACC_HEIGHT; ++aTile, aRowOffset += 16) {
                if (loadARow + aRowOffset < A_rows) {
                    // load A from main mem to frag
                    wmma::load_matrix_sync(a_frag, A + (loadARow + aRowOffset) * ld_A + loadACol, ld_A);
#pragma unroll
                    for (int bTile = 0; bTile < ACC_WIDTH; ++bTile) {
                        if (loadBCol + bTile * 16 >= B_cols)
                            break;
                        wmma::mma_sync(acc_frag[aTile][bTile], a_frag, b_frag[bTile], acc_frag[aTile][bTile]);
                    }
                }
            }
        }
        __syncthreads();  // improves performance
    }

    // Each of the 8 warps in the block have ACC_WIDTH accumulators, the sum of which is the final sum for this tile of the result
    // matrix Store them to shared in preparation for reduction
#pragma unroll
    for (int aTile = 0; aTile < ACC_HEIGHT; ++aTile) {
#pragma unroll
        for (int bTile = 0; bTile < ACC_WIDTH; ++bTile) {
            if (bTile * 16 + loadBCol >= B_cols)
                break;

            __syncthreads();
            wmma::store_matrix_sync(
                &(buffer_shared_float[threadIdx.y * (ACC_WIDTH * 16 * (8 + 4))]), acc_frag[aTile][bTile], (16 + 4), wmma::mem_row_major);
            __syncthreads();

            int reduceRow = threadIdx.y * 2 + threadIdx.x / 16;
            int reduceCol = threadIdx.x % 16;
            float f = buffer_shared_float[reduceRow * (16 + 4) + reduceCol];
#pragma unroll
            for (int i = 1; i < 8; ++i) {
                f += buffer_shared_float[i * (ACC_WIDTH * 16 * (8 + 4)) + reduceRow * (16 + 4) + reduceCol];
            }
            half h = (half)f;

            // if(threadIdx.y == 1) printf("t.y = 1, C[%d, %d] = %f\n", (loadARow + 16 * aTile + reduceRow), (loadBCol + bTile * 16 +
            // reduceCol), float(h));
            C[(loadARow + 16 * aTile + reduceRow) * ld_C + (loadBCol + bTile * 16 + reduceCol)] = h;
        }
    }
}

#undef ACC_WIDTH
#undef ACC_HEIGHT

#define ACC_WIDTH 3
#define ACC_HEIGHT 1

__global__ void tensorCoreMatrixMultiplyKernel_Arow16_Bcol48_restrictionABMult16x16_loadToReg(const half *A,
                                                                                              const half *B,
                                                                                              half *C,
                                                                                              const int32_t A_rows,
                                                                                              const int32_t A_cols,
                                                                                              const int32_t B_cols,
                                                                                              const int32_t ld_A,
                                                                                              const int32_t ld_B,
                                                                                              const int32_t ld_C) {
    __shared__ half buffer_shared[8][ACC_WIDTH * 16 * (16 + 8)];  // 8*1024*2 = 16KB
    float *buffer_shared_float = (float *)buffer_shared;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[ACC_WIDTH];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag[ACC_HEIGHT][ACC_WIDTH];

    int loadARow = blockIdx.y * ACC_HEIGHT * 16;
    int loadBCol = blockIdx.x * ACC_WIDTH * 16;
    int loadACol = threadIdx.y * 16;  // counts along k
    int loadBRow = threadIdx.y * 16;  // counts along k

    for (int aTile = 0; aTile < ACC_HEIGHT; ++aTile) {
        for (int bTile = 0; bTile < ACC_WIDTH; ++bTile) {
            wmma::fill_fragment(acc_frag[aTile][bTile], 0.0f);
        }
    }

#pragma unroll
    for (; (loadACol & 0xFFFFFF80) < A_cols; loadACol += 128, loadBRow += 128) {
        if (loadACol < A_cols) {
#pragma unroll
            for (int bTile = 0; bTile < ACC_WIDTH; ++bTile) {
                if (loadBCol + bTile * 16 >= B_cols)
                    break;
                wmma::load_matrix_sync(b_frag[bTile], B + loadBRow * ld_B + loadBCol + bTile * 16, ld_B);
            }

#pragma unroll
            for (int aRowOffset = 0, aTile = 0; aTile < ACC_HEIGHT; ++aTile, aRowOffset += 16) {
                if (loadARow + aRowOffset < A_rows) {
                    // load A from main mem to frag
                    wmma::load_matrix_sync(a_frag, A + (loadARow + aRowOffset) * ld_A + loadACol, ld_A);
#pragma unroll
                    for (int bTile = 0; bTile < ACC_WIDTH; ++bTile) {
                        if (loadBCol + bTile * 16 >= B_cols)
                            break;
                        wmma::mma_sync(acc_frag[aTile][bTile], a_frag, b_frag[bTile], acc_frag[aTile][bTile]);
                    }
                }
            }
        }
        __syncthreads();  // improves performance
    }

    // Each of the 8 warps in the block have ACC_WIDTH accumulators, the sum of which is the final sum for this tile of the result
    // matrix Store them to shared in preparation for reduction
#pragma unroll
    for (int aTile = 0; aTile < ACC_HEIGHT; ++aTile) {
#pragma unroll
        for (int bTile = 0; bTile < ACC_WIDTH; ++bTile) {
            if (bTile * 16 + loadBCol >= B_cols)
                break;

            __syncthreads();
            wmma::store_matrix_sync(
                &(buffer_shared_float[threadIdx.y * (ACC_WIDTH * 16 * (8 + 4))]), acc_frag[aTile][bTile], (16 + 4), wmma::mem_row_major);
            __syncthreads();

            int reduceRow = threadIdx.y * 2 + threadIdx.x / 16;
            int reduceCol = threadIdx.x % 16;
            float f = buffer_shared_float[reduceRow * (16 + 4) + reduceCol];
#pragma unroll
            for (int i = 1; i < 8; ++i) {
                f += buffer_shared_float[i * (ACC_WIDTH * 16 * (8 + 4)) + reduceRow * (16 + 4) + reduceCol];
            }
            half h = (half)f;

            // if(threadIdx.y == 1) printf("t.y = 1, C[%d, %d] = %f\n", (loadARow + 16 * aTile + reduceRow), (loadBCol + bTile * 16 +
            // reduceCol), float(h));
            C[(loadARow + 16 * aTile + reduceRow) * ld_C + (loadBCol + bTile * 16 + reduceCol)] = h;
        }
    }
}

#undef ACC_WIDTH
#undef ACC_HEIGHT

#define ACC_WIDTH 3
#define ACC_HEIGHT 4

__global__ void tensorCoreMatrixMultiplyKernel_Arow64_Bcol48_restrictionABMult16x16_loadToReg(const half *A,
                                                                                              const half *B,
                                                                                              half *C,
                                                                                              const int32_t A_rows,
                                                                                              const int32_t A_cols,
                                                                                              const int32_t B_cols,
                                                                                              const int32_t ld_A,
                                                                                              const int32_t ld_B,
                                                                                              const int32_t ld_C) {
    __shared__ half buffer_shared[8][ACC_WIDTH * 16 * (16 + 8)];  // 8*1024*2 = 16KB
    float *buffer_shared_float = (float *)buffer_shared;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[ACC_WIDTH];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag[ACC_HEIGHT][ACC_WIDTH];

    int loadARow = blockIdx.y * ACC_HEIGHT * 16;
    int loadBCol = blockIdx.x * ACC_WIDTH * 16;
    int loadACol = threadIdx.y * 16;  // counts along k
    int loadBRow = threadIdx.y * 16;  // counts along k

    for (int aTile = 0; aTile < ACC_HEIGHT; ++aTile) {
        for (int bTile = 0; bTile < ACC_WIDTH; ++bTile) {
            wmma::fill_fragment(acc_frag[aTile][bTile], 0.0f);
        }
    }

#pragma unroll
    for (; (loadACol & 0xFFFFFF80) < A_cols; loadACol += 128, loadBRow += 128) {
        if (loadACol < A_cols) {
#pragma unroll
            for (int bTile = 0; bTile < ACC_WIDTH; ++bTile) {
                if (loadBCol + bTile * 16 >= B_cols)
                    break;
                wmma::load_matrix_sync(b_frag[bTile], B + loadBRow * ld_B + loadBCol + bTile * 16, ld_B);
            }

#pragma unroll
            for (int aRowOffset = 0, aTile = 0; aTile < ACC_HEIGHT; ++aTile, aRowOffset += 16) {
                if (loadARow + aRowOffset < A_rows) {
                    // load A from main mem to frag
                    wmma::load_matrix_sync(a_frag, A + (loadARow + aRowOffset) * ld_A + loadACol, ld_A);
#pragma unroll
                    for (int bTile = 0; bTile < ACC_WIDTH; ++bTile) {
                        if (loadBCol + bTile * 16 >= B_cols)
                            break;
                        wmma::mma_sync(acc_frag[aTile][bTile], a_frag, b_frag[bTile], acc_frag[aTile][bTile]);
                    }
                }
            }
        }
        __syncthreads();  // improves performance
    }

    // Each of the 8 warps in the block have ACC_WIDTH accumulators, the sum of which is the final sum for this tile of the result
    // matrix Store them to shared in preparation for reduction
#pragma unroll
    for (int aTile = 0; aTile < ACC_HEIGHT; ++aTile) {
#pragma unroll
        for (int bTile = 0; bTile < ACC_WIDTH; ++bTile) {
            if (bTile * 16 + loadBCol >= B_cols)
                break;

            __syncthreads();
            wmma::store_matrix_sync(
                &(buffer_shared_float[threadIdx.y * (ACC_WIDTH * 16 * (8 + 4))]), acc_frag[aTile][bTile], (16 + 4), wmma::mem_row_major);
            __syncthreads();

            int reduceRow = threadIdx.y * 2 + threadIdx.x / 16;
            int reduceCol = threadIdx.x % 16;
            float f = buffer_shared_float[reduceRow * (16 + 4) + reduceCol];
#pragma unroll
            for (int i = 1; i < 8; ++i) {
                f += buffer_shared_float[i * (ACC_WIDTH * 16 * (8 + 4)) + reduceRow * (16 + 4) + reduceCol];
            }
            half h = (half)f;

            // if(threadIdx.y == 1) printf("t.y = 1, C[%d, %d] = %f\n", (loadARow + 16 * aTile + reduceRow), (loadBCol + bTile * 16 +
            // reduceCol), float(h));
            C[(loadARow + 16 * aTile + reduceRow) * ld_C + (loadBCol + bTile * 16 + reduceCol)] = h;
        }
    }
}

#undef ACC_WIDTH
#undef ACC_HEIGHT

#define ACC_WIDTH 3
#define ACC_HEIGHT 2
#define ldb_shared (32 * ((ACC_WIDTH * 16 + 31) / 32) + 8)

__global__ void tensorCoreMatrixMultiplyKernel_Arow256_Bcol48_restrictionAMult16x16(const half *A,
                                                                                    const half *B,
                                                                                    half *C,
                                                                                    const int32_t A_rows,
                                                                                    const int32_t A_cols,
                                                                                    const int32_t B_cols,
                                                                                    const int32_t ld_A,
                                                                                    const int32_t ld_B,
                                                                                    const int32_t ld_C) {
    __shared__ half B_shared[128 * ldb_shared];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag[ACC_HEIGHT][ACC_WIDTH];

    int loadARow = blockIdx.y * 128 * ACC_HEIGHT + threadIdx.y * 16;
    int loadACol = 0;                                          // counts along k
    int loadBCol = blockIdx.x * ACC_WIDTH * 16 + threadIdx.x;  // iterates to fill ACC_WIDTH*16
    int loadBRowShared = threadIdx.y;
    int loadBRow = loadBRowShared;  // counts along k

    for (int i = 0; i < ACC_HEIGHT; ++i) {
        for (int j = 0; j < ACC_WIDTH; ++j) {
            wmma::fill_fragment(acc_frag[i][j], 0.0f);
        }
    }

#pragma unroll 8
    for (; loadACol < A_cols; loadACol += 128, loadBRow += 128) {
        // Load a chunk of B to shared
        __syncthreads();
#pragma unroll 16
        for (int bRowOffset = 0; bRowOffset < 128; bRowOffset += 8) {
#pragma unroll 5
            for (int bColOffset = 0; bColOffset < ACC_WIDTH * 16; bColOffset += 32) {
                // Note: detecting this:
                // if ( ((loadBRow + bRowOffset)&0xFFFFFFF0) >= A_cols || ((loadBCol + bColOffset)&0xFFFFFFF0) >= B_cols ) break;
                // causes fifty percent kernel slowdown

                int bSharedIndex = (loadBRowShared + bRowOffset) * ldb_shared + (threadIdx.x + bColOffset);
                half buff;
                if (loadBRow + bRowOffset >= A_cols || loadBCol + bColOffset >= B_cols) {
                    buff = half(0.0f);
                } else {
                    buff = B[(loadBRow + bRowOffset) * ld_B + (loadBCol + bColOffset)];
                }
                B_shared[bSharedIndex] = buff;
            }
        }
        __syncthreads();

        // Iterate through the memory doing mma, with ratio of 1 load of A to ACC_WIDTH loads of shared_B for ACC_WIDTH mma's
#pragma unroll 2
        for (int aRowOffset = 0; aRowOffset < ACC_HEIGHT * 128; aRowOffset += 128) {
            if (loadARow + aRowOffset >= A_rows)
                break;
#pragma unroll 8
            for (int aColOffset = 0; aColOffset < 128; aColOffset += 16) {
                if (loadACol + aColOffset >= A_cols)
                    break;

                wmma::load_matrix_sync(a_frag, A + (loadARow + aRowOffset) * ld_A + loadACol + aColOffset, ld_A);

#pragma unroll 6
                for (int bColTileOffset = 0; bColTileOffset < ACC_WIDTH; bColTileOffset += 1) {
                    // Note: detecting this:
                    // if(bColStart >= B_cols) break;
                    // causes a five percent slowdown

                    // FIXME: for ACC_WIDTH < 6, swap the order in which aRowOffset is incremented and B is loaded and below, so 1 load from
                    // shared not 2 for both lines of A
                    wmma::load_matrix_sync(b_frag, B_shared + aColOffset * ldb_shared + (bColTileOffset * 16), ldb_shared);
                    if (aRowOffset == 0)
                        wmma::mma_sync(acc_frag[0][bColTileOffset], a_frag, b_frag, acc_frag[0][bColTileOffset]);
                    else
                        wmma::mma_sync(acc_frag[1][bColTileOffset], a_frag, b_frag, acc_frag[1][bColTileOffset]);
                }
            }
        }
    }
    __syncthreads();

    // Convert to half and write the result to memory
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> acc_frag_half;
    const int CSharedCol = threadIdx.x % 16;
#pragma unroll 2
    for (int accLine = 0; accLine < ACC_HEIGHT; ++accLine) {
#pragma unroll 6
        for (int acc = 0; acc < ACC_WIDTH; ++acc) {
            for (int8_t i = 0; i < acc_frag_half.num_elements; ++i)
                acc_frag_half.x[i] = (half)acc_frag[accLine][acc].x[i];
            __syncthreads();
            wmma::store_matrix_sync(B_shared + threadIdx.y * (16 * 24), acc_frag_half, 24, wmma::mem_row_major);

            int warpStartCCol = (blockIdx.x * ACC_WIDTH + acc) * 16;
            if (warpStartCCol >= B_cols)
                break;
            int CSharedRow = threadIdx.y * 2 + threadIdx.x / 16;
            int CRow = blockIdx.y * 128 * ACC_HEIGHT + accLine * 128 + CSharedRow;
            int CCol = warpStartCCol + CSharedCol;

            __syncthreads();

            if (CCol < B_cols) {
#pragma unroll 8
                for (int i = 0; i < 8 && CRow < A_rows; ++i) {
                    C[CRow * ld_C + CCol] = B_shared[i * (16 * 24) + CSharedRow * 24 + CSharedCol];
                    CRow += 16;
                }
            }
        }
    }
}

__global__ void tensorCoreMatrixMultiplyKernel_Arow256_Bcol48_noRestriction(const half *A,
                                                                            const half *B,
                                                                            half *C,
                                                                            const int32_t A_rows,
                                                                            const int32_t A_cols,
                                                                            const int32_t B_cols,
                                                                            const int32_t ld_A,
                                                                            const int32_t ld_B,
                                                                            const int32_t ld_C) {
    __shared__ half B_shared[128 * ldb_shared];
    __shared__ half buffer_shared[8 * 16 * 24];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag[ACC_HEIGHT][ACC_WIDTH];

    int loadARow = blockIdx.y * 128 * ACC_HEIGHT + threadIdx.y * 16;
    int loadACol = 0;                                          // counts along k
    int loadBCol = blockIdx.x * ACC_WIDTH * 16 + threadIdx.x;  // iterates to fill ACC_WIDTH*16
    int loadBRowShared = threadIdx.y;
    int loadBRow = loadBRowShared;  // counts along k

    for (int i = 0; i < ACC_HEIGHT; ++i) {
        for (int j = 0; j < ACC_WIDTH; ++j) {
            wmma::fill_fragment(acc_frag[i][j], 0.0f);
        }
    }

    for (; loadACol < A_cols; loadACol += 128, loadBRow += 128) {
        // Load a chunk of B to shared
        __syncthreads();
#pragma unroll 16
        for (int bRowOffset = 0; bRowOffset < 128; bRowOffset += 8) {
#pragma unroll 5
            for (int bColOffset = 0; bColOffset < ACC_WIDTH * 16; bColOffset += 32) {
                int bSharedIndex = (loadBRowShared + bRowOffset) * ldb_shared + (threadIdx.x + bColOffset);
                half buff;
                if (loadBRow + bRowOffset >= A_cols || loadBCol + bColOffset >= B_cols) {
                    buff = half(0.0f);
                } else {
                    buff = B[(loadBRow + bRowOffset) * ld_B + (loadBCol + bColOffset)];
                }
                B_shared[bSharedIndex] = buff;
            }
        }
        __syncthreads();

        // Iterate through the memory doing mma, with ratio of 1 load of A to ACC_WIDTH loads of shared_B for ACC_WIDTH mma's
        int bColStart = blockIdx.x * ACC_WIDTH * 16;
#pragma unroll 2
        for (int aRowOffset = 0; aRowOffset < ACC_HEIGHT * 128; aRowOffset += 128) {
            if (loadARow + aRowOffset >= A_rows)
                break;
#pragma unroll 8
            for (int aColOffset = 0; aColOffset < 128; aColOffset += 16) {
                if (loadACol + aColOffset >= A_cols)
                    break;

#pragma unroll 8
                for (int bufferRow = threadIdx.x / 16; bufferRow < 16; bufferRow += 2) {
                    half buff;
                    if ((loadARow + aRowOffset + bufferRow) >= A_rows || (loadACol + aColOffset + (threadIdx.x % 16)) >= A_cols) {
                        buff = half(0.0f);
                    } else {
                        buff = A[(loadARow + aRowOffset + bufferRow) * ld_A + loadACol + aColOffset + threadIdx.x % 16];
                    }
                    buffer_shared[threadIdx.y * (16 * 24) + bufferRow * 24 + threadIdx.x % 16] = buff;
                }
                wmma::load_matrix_sync(a_frag, buffer_shared + threadIdx.y * (16 * 24), 24);

#pragma unroll 6
                for (int bColTileOffset = 0; bColTileOffset < ACC_WIDTH; bColTileOffset += 1, bColStart += 16) {
                    wmma::load_matrix_sync(b_frag, B_shared + aColOffset * ldb_shared + (bColTileOffset * 16), ldb_shared);
                    if (aRowOffset == 0)
                        wmma::mma_sync(acc_frag[0][bColTileOffset], a_frag, b_frag, acc_frag[0][bColTileOffset]);
                    else
                        wmma::mma_sync(acc_frag[1][bColTileOffset], a_frag, b_frag, acc_frag[1][bColTileOffset]);
                }
            }
        }
    }

    // Convert to half and write the result to memory
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> acc_frag_half;
    const int CSharedCol = threadIdx.x % 16;
    for (int accLine = 0; accLine < ACC_HEIGHT; ++accLine) {
#pragma unroll 6
        for (int acc = 0; acc < ACC_WIDTH; ++acc) {
            for (int8_t i = 0; i < acc_frag_half.num_elements; ++i)
                acc_frag_half.x[i] = (half)acc_frag[accLine][acc].x[i];
            __syncthreads();
            wmma::store_matrix_sync(buffer_shared + threadIdx.y * (16 * 24), acc_frag_half, 24, wmma::mem_row_major);

            int warpStartCCol = (blockIdx.x * ACC_WIDTH + acc) * 16;
            if (warpStartCCol >= B_cols)
                break;
            int CSharedRow = threadIdx.y * 2 + threadIdx.x / 16;
            int CRow = blockIdx.y * 128 * ACC_HEIGHT + accLine * 128 + CSharedRow;
            int CCol = warpStartCCol + CSharedCol;

            __syncthreads();

            if (CCol < B_cols) {
#pragma unroll 8
                for (int i = 0; i < 8 && CRow < A_rows; ++i) {
                    C[CRow * ld_C + CCol] = buffer_shared[i * (16 * 24) + CSharedRow * 24 + CSharedCol];
                    CRow += 16;
                }
            }
        }
    }
}

__global__ void tensorCoreMatrixMultiplyKernel_Arow256_Bcol48_withReduce_restrictionAMult16x16(const half *A,
                                                                                               const half *B,
                                                                                               half *C,
                                                                                               half *C_workspace,
                                                                                               const int32_t A_rows,
                                                                                               const int32_t A_cols,
                                                                                               const int32_t B_cols,
                                                                                               const int32_t ld_A,
                                                                                               const int32_t ld_B,
                                                                                               const int32_t ld_C) {
    __shared__ half B_shared[128 * ldb_shared];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag[ACC_HEIGHT][ACC_WIDTH];

    int kDistanceRaw = (A_cols + (gridDim.z - 1)) / gridDim.z;
    int kDistanceMod16 = 16 * ((kDistanceRaw + 15) / 16);
    int kEnd = ((blockIdx.z + 1) * kDistanceMod16) - 1;
    if (kEnd >= A_cols)
        kEnd = A_cols - 1;

    int loadARow = blockIdx.y * 128 * ACC_HEIGHT + threadIdx.y * 16;
    int loadACol = blockIdx.z * kDistanceMod16;                // counts along k
    int loadBCol = blockIdx.x * ACC_WIDTH * 16 + threadIdx.x;  // iterates to fill ACC_WIDTH*16
    int loadBRowShared = threadIdx.y;
    int loadBRow = loadBRowShared + blockIdx.z * kDistanceMod16;  // counts along k

    for (int i = 0; i < ACC_HEIGHT; ++i) {
        for (int j = 0; j < ACC_WIDTH; ++j) {
            wmma::fill_fragment(acc_frag[i][j], 0.0f);
        }
    }

#pragma unroll 8
    for (; loadACol <= kEnd; loadACol += 128, loadBRow += 128) {
        // Load a chunk of B to shared
        __syncthreads();
#pragma unroll 16
        for (int bRowOffset = 0; bRowOffset < 128; bRowOffset += 8) {
#pragma unroll 5
            for (int bColOffset = 0; bColOffset < ACC_WIDTH * 16; bColOffset += 32) {
                // Note: detecting this:
                // if ( ((loadBRow + bRowOffset)&0xFFFFFFF0) > kEnd || ((loadBCol + bColOffset)&0xFFFFFFF0) >= B_cols ) break;
                // causes fifty percent kernel slowdown

                int bSharedIndex = (loadBRowShared + bRowOffset) * ldb_shared + (threadIdx.x + bColOffset);
                half buff;
                if (loadBRow + bRowOffset > kEnd || loadBCol + bColOffset >= B_cols) {
                    buff = half(0.0f);
                } else {
                    buff = B[(loadBRow + bRowOffset) * ld_B + (loadBCol + bColOffset)];
                }
                B_shared[bSharedIndex] = buff;
            }
        }
        __syncthreads();

        // Iterate through the memory doing mma, with ratio of 1 load of A to ACC_WIDTH loads of shared_B for ACC_WIDTH mma's
#pragma unroll 2
        for (int aRowOffset = 0; aRowOffset < ACC_HEIGHT * 128; aRowOffset += 128) {
            if (loadARow + aRowOffset >= A_rows)
                break;
#pragma unroll 8
            for (int aColOffset = 0; aColOffset < 128; aColOffset += 16) {
                if (loadACol + aColOffset > kEnd)
                    break;

                wmma::load_matrix_sync(a_frag, A + (loadARow + aRowOffset) * ld_A + loadACol + aColOffset, ld_A);

#pragma unroll 6
                for (int bColTileOffset = 0; bColTileOffset < ACC_WIDTH; bColTileOffset += 1) {
                    // Note: detecting this:
                    // if(bColStart >= B_cols) break;
                    // causes a five percent slowdown

                    // FIXME: for ACC_WIDTH < 6, swap the order in which aRowOffset is incremented and B is loaded and below, so 1 load from
                    // shared not 2 for both lines of A
                    wmma::load_matrix_sync(b_frag, B_shared + aColOffset * ldb_shared + (bColTileOffset * 16), ldb_shared);
                    if (aRowOffset == 0)
                        wmma::mma_sync(acc_frag[0][bColTileOffset], a_frag, b_frag, acc_frag[0][bColTileOffset]);
                    else
                        wmma::mma_sync(acc_frag[1][bColTileOffset], a_frag, b_frag, acc_frag[1][bColTileOffset]);
                }
            }
        }
    }
    __syncthreads();

    // Convert to half and write the result to memory
    long workspaceOffset;
    if (blockIdx.z != 0)
        workspaceOffset = (blockIdx.z - 1) * (A_rows * ld_C);
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> acc_frag_half;
    const int CSharedCol = threadIdx.x % 16;
#pragma unroll 2
    for (int accLine = 0; accLine < ACC_HEIGHT; ++accLine) {
#pragma unroll 6
        for (int acc = 0; acc < ACC_WIDTH; ++acc) {
            for (int8_t i = 0; i < acc_frag_half.num_elements; ++i)
                acc_frag_half.x[i] = (half)acc_frag[accLine][acc].x[i];
            __syncthreads();
            wmma::store_matrix_sync(B_shared + threadIdx.y * (16 * 24), acc_frag_half, 24, wmma::mem_row_major);

            int warpStartCCol = (blockIdx.x * ACC_WIDTH + acc) * 16;
            if (warpStartCCol >= B_cols)
                break;
            int CSharedRow = threadIdx.y * 2 + threadIdx.x / 16;
            int CRow = blockIdx.y * 128 * ACC_HEIGHT + accLine * 128 + CSharedRow;
            int CCol = warpStartCCol + CSharedCol;

            __syncthreads();

            if (CCol < B_cols) {
                if (blockIdx.z == 0) {
#pragma unroll 8
                    for (int i = 0; i < 8 && CRow < A_rows; ++i) {
                        C[CRow * ld_C + CCol] = B_shared[i * (16 * 24) + CSharedRow * 24 + CSharedCol];
                        CRow += 16;
                    }
                } else {
#pragma unroll 8
                    for (int i = 0; i < 8 && CRow < A_rows; ++i) {
                        C_workspace[workspaceOffset + CRow * ld_C + CCol] = B_shared[i * (16 * 24) + CSharedRow * 24 + CSharedCol];
                        CRow += 16;
                    }
                }
            }
        }
    }
}

__global__ void tensorCoreMatrixMultiplyKernel_Arow256_Bcol48_withReduce_noRestriction(const half *A,
                                                                                       const half *B,
                                                                                       half *C,
                                                                                       half *C_workspace,
                                                                                       const int32_t A_rows,
                                                                                       const int32_t A_cols,
                                                                                       const int32_t B_cols,
                                                                                       const int32_t ld_A,
                                                                                       const int32_t ld_B,
                                                                                       const int32_t ld_C) {
    __shared__ half B_shared[128 * ldb_shared];
    __shared__ half buffer_shared[8 * 16 * 24];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag[ACC_HEIGHT][ACC_WIDTH];

    int kDistanceRaw = (A_cols + (gridDim.z - 1)) / gridDim.z;
    int kDistanceMod16 = 16 * ((kDistanceRaw + 15) / 16);
    int kEnd = ((blockIdx.z + 1) * kDistanceMod16) - 1;
    if (kEnd >= A_cols)
        kEnd = A_cols - 1;

    int loadARow = blockIdx.y * 128 * ACC_HEIGHT + threadIdx.y * 16;
    int loadACol = blockIdx.z * kDistanceMod16;                // counts along k
    int loadBCol = blockIdx.x * ACC_WIDTH * 16 + threadIdx.x;  // iterates to fill ACC_WIDTH*16
    int loadBRowShared = threadIdx.y;
    int loadBRow = loadBRowShared + blockIdx.z * kDistanceMod16;  // counts along k

    for (int i = 0; i < ACC_HEIGHT; ++i) {
        for (int j = 0; j < ACC_WIDTH; ++j) {
            wmma::fill_fragment(acc_frag[i][j], 0.0f);
        }
    }

    for (; loadACol < A_cols; loadACol += 128, loadBRow += 128) {
        // Load a chunk of B to shared
        __syncthreads();
#pragma unroll 16
        for (int bRowOffset = 0; bRowOffset < 128; bRowOffset += 8) {
#pragma unroll 5
            for (int bColOffset = 0; bColOffset < ACC_WIDTH * 16; bColOffset += 32) {
                int bSharedIndex = (loadBRowShared + bRowOffset) * ldb_shared + (threadIdx.x + bColOffset);
                half buff;
                if (loadBRow + bRowOffset > kEnd || loadBCol + bColOffset >= B_cols) {
                    buff = half(0.0f);
                } else {
                    buff = B[(loadBRow + bRowOffset) * ld_B + (loadBCol + bColOffset)];
                }
                B_shared[bSharedIndex] = buff;
            }
        }
        __syncthreads();

        // Iterate through the memory doing mma, with ratio of 1 load of A to ACC_WIDTH loads of shared_B for ACC_WIDTH mma's
        int bColStart = blockIdx.x * ACC_WIDTH * 16;
#pragma unroll 2
        for (int aRowOffset = 0; aRowOffset < ACC_HEIGHT * 128; aRowOffset += 128) {
            if (loadARow + aRowOffset >= A_rows)
                break;
#pragma unroll 8
            for (int aColOffset = 0; aColOffset < 128; aColOffset += 16) {
                if (loadACol + aColOffset > kEnd)
                    break;

#pragma unroll 8
                for (int bufferRow = threadIdx.x / 16; bufferRow < 16; bufferRow += 2) {
                    half buff;
                    if ((loadARow + aRowOffset + bufferRow) >= A_rows || (loadACol + aColOffset + (threadIdx.x % 16)) > kEnd) {
                        buff = half(0.0f);
                    } else {
                        buff = A[(loadARow + aRowOffset + bufferRow) * ld_A + loadACol + aColOffset + threadIdx.x % 16];
                    }
                    buffer_shared[threadIdx.y * (16 * 24) + bufferRow * 24 + threadIdx.x % 16] = buff;
                }
                wmma::load_matrix_sync(a_frag, buffer_shared + threadIdx.y * (16 * 24), 24);

#pragma unroll 6
                for (int bColTileOffset = 0; bColTileOffset < ACC_WIDTH; bColTileOffset += 1, bColStart += 16) {
                    wmma::load_matrix_sync(b_frag, B_shared + aColOffset * ldb_shared + (bColTileOffset * 16), ldb_shared);
                    if (aRowOffset == 0)
                        wmma::mma_sync(acc_frag[0][bColTileOffset], a_frag, b_frag, acc_frag[0][bColTileOffset]);
                    else
                        wmma::mma_sync(acc_frag[1][bColTileOffset], a_frag, b_frag, acc_frag[1][bColTileOffset]);
                }
            }
        }
    }

    // Convert to half and write the result to memory
    long workspaceOffset;
    if (blockIdx.z != 0)
        workspaceOffset = (blockIdx.z - 1) * (A_rows * ld_C);
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> acc_frag_half;
    const int CSharedCol = threadIdx.x % 16;
    for (int accLine = 0; accLine < ACC_HEIGHT; ++accLine) {
#pragma unroll 6
        for (int acc = 0; acc < ACC_WIDTH; ++acc) {
            for (int8_t i = 0; i < acc_frag_half.num_elements; ++i)
                acc_frag_half.x[i] = (half)acc_frag[accLine][acc].x[i];
            __syncthreads();
            wmma::store_matrix_sync(buffer_shared + threadIdx.y * (16 * 24), acc_frag_half, 24, wmma::mem_row_major);

            int warpStartCCol = (blockIdx.x * ACC_WIDTH + acc) * 16;
            if (warpStartCCol >= B_cols)
                break;
            int CSharedRow = threadIdx.y * 2 + threadIdx.x / 16;
            int CRow = blockIdx.y * 128 * ACC_HEIGHT + accLine * 128 + CSharedRow;
            int CCol = warpStartCCol + CSharedCol;

            __syncthreads();

            if (CCol < B_cols) {
                if (blockIdx.z == 0) {
#pragma unroll 8
                    for (int i = 0; i < 8 && CRow < A_rows; ++i) {
                        C[CRow * ld_C + CCol] = buffer_shared[i * (16 * 24) + CSharedRow * 24 + CSharedCol];
                        CRow += 16;
                    }
                } else {
                    for (int i = 0; i < 8 && CRow < A_rows; ++i) {
                        C_workspace[workspaceOffset + CRow * ld_C + CCol] = buffer_shared[i * (16 * 24) + CSharedRow * 24 + CSharedCol];
                        CRow += 16;
                    }
                }
            }
        }
    }
}

#undef ACC_HEIGHT
#undef ACC_WIDTH
#undef ldb_shared

#define ACC_WIDTH 5
#define ACC_HEIGHT "hard coded to 1"
#define ldb_shared (32 * ((ACC_WIDTH * 16 + 31) / 32) + 8)

__global__ void tensorCoreMatrixMultiplyKernel_Arow128_Bcol48_restrictionAMult16x16(const half *A,
                                                                                    const half *B,
                                                                                    half *C,
                                                                                    const int32_t A_rows,
                                                                                    const int32_t A_cols,
                                                                                    const int32_t B_cols,
                                                                                    const int32_t ld_A,
                                                                                    const int32_t ld_B,
                                                                                    const int32_t ld_C) {
    __shared__ half B_shared[128 * ldb_shared];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag[ACC_WIDTH];

    int loadARow = blockIdx.y * 128 + threadIdx.y * 16;
    int loadACol = 0;                                          // counts along k
    int loadBCol = blockIdx.x * ACC_WIDTH * 16 + threadIdx.x;  // iterates to fill ACC_WIDTH*16
    int loadBRowShared = threadIdx.y;
    int loadBRow = loadBRowShared;  // counts along k

    for (int j = 0; j < ACC_WIDTH; ++j) {
        wmma::fill_fragment(acc_frag[j], 0.0f);
    }

#pragma unroll 8
    for (; loadACol < A_cols; loadACol += 128, loadBRow += 128) {
        // Load a chunk of B to shared
        __syncthreads();
#pragma unroll 16
        for (int bRowOffset = 0; bRowOffset < 128; bRowOffset += 8) {
#pragma unroll 5
            for (int bColOffset = 0; bColOffset < ACC_WIDTH * 16; bColOffset += 32) {
                // Note: detecting this:
                // if ( ((loadBRow + bRowOffset)&0xFFFFFFF0) >= A_cols || ((loadBCol + bColOffset)&0xFFFFFFF0) >= B_cols ) break;
                // causes fifty percent kernel slowdown

                int bSharedIndex = (loadBRowShared + bRowOffset) * ldb_shared + (threadIdx.x + bColOffset);
                half buff;
                if (loadBRow + bRowOffset >= A_cols || loadBCol + bColOffset >= B_cols) {
                    buff = half(0.0f);
                } else {
                    buff = B[(loadBRow + bRowOffset) * ld_B + (loadBCol + bColOffset)];
                }
                B_shared[bSharedIndex] = buff;
            }
        }
        __syncthreads();

        // Iterate through the memory doing mma, with ratio of 1 load of A to ACC_WIDTH loads of shared_B for ACC_WIDTH mma's
        if (loadARow < A_rows) {
#pragma unroll 8
            for (int aColOffset = 0; aColOffset < 128; aColOffset += 16) {
                if (loadACol + aColOffset >= A_cols)
                    break;

                wmma::load_matrix_sync(a_frag, A + loadARow * ld_A + loadACol + aColOffset, ld_A);

#pragma unroll 6
                for (int bColTileOffset = 0; bColTileOffset < ACC_WIDTH; bColTileOffset += 1) {
                    // Note: detecting this:
                    // if(bColStart >= B_cols) break;
                    // causes a five percent slowdown

                    wmma::load_matrix_sync(b_frag, B_shared + aColOffset * ldb_shared + (bColTileOffset * 16), ldb_shared);
                    wmma::mma_sync(acc_frag[bColTileOffset], a_frag, b_frag, acc_frag[bColTileOffset]);
                }
            }
        }
    }
    __syncthreads();

    // Convert to half and write the result to memory
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> acc_frag_half;
    const int CSharedCol = threadIdx.x % 16;
#pragma unroll 6
    for (int acc = 0; acc < ACC_WIDTH; ++acc) {
        for (int8_t i = 0; i < acc_frag_half.num_elements; ++i)
            acc_frag_half.x[i] = (half)acc_frag[acc].x[i];
        __syncthreads();
        wmma::store_matrix_sync(B_shared + threadIdx.y * (16 * 24), acc_frag_half, 24, wmma::mem_row_major);

        int warpStartCCol = (blockIdx.x * ACC_WIDTH + acc) * 16;
        if (warpStartCCol >= B_cols)
            break;
        int CSharedRow = threadIdx.y * 2 + threadIdx.x / 16;
        int CRow = blockIdx.y * 128 + CSharedRow;
        int CCol = warpStartCCol + CSharedCol;

        __syncthreads();

        if (CCol < B_cols) {
#pragma unroll 8
            for (int i = 0; i < 8 && CRow < A_rows; ++i) {
                C[CRow * ld_C + CCol] = B_shared[i * (16 * 24) + CSharedRow * 24 + CSharedCol];
                CRow += 16;
            }
        }
    }
}

__global__ void tensorCoreMatrixMultiplyKernel_Arow128_Bcol48_noRestriction(const half *A,
                                                                            const half *B,
                                                                            half *C,
                                                                            const int32_t A_rows,
                                                                            const int32_t A_cols,
                                                                            const int32_t B_cols,
                                                                            const int32_t ld_A,
                                                                            const int32_t ld_B,
                                                                            const int32_t ld_C) {
    __shared__ half B_shared[128 * ldb_shared];
    __shared__ half buffer_shared[8 * 16 * 24];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag[ACC_WIDTH];

    int loadARow = blockIdx.y * 128 + threadIdx.y * 16;
    int loadACol = 0;                                          // counts along k
    int loadBCol = blockIdx.x * ACC_WIDTH * 16 + threadIdx.x;  // iterates to fill ACC_WIDTH*16
    int loadBRowShared = threadIdx.y;
    int loadBRow = loadBRowShared;  // counts along k

    for (int j = 0; j < ACC_WIDTH; ++j) {
        wmma::fill_fragment(acc_frag[j], 0.0f);
    }

    for (; loadACol < A_cols; loadACol += 128, loadBRow += 128) {
        // Load a chunk of B to shared
        __syncthreads();
#pragma unroll 16
        for (int bRowOffset = 0; bRowOffset < 128; bRowOffset += 8) {
#pragma unroll 5
            for (int bColOffset = 0; bColOffset < ACC_WIDTH * 16; bColOffset += 32) {
                int bSharedIndex = (loadBRowShared + bRowOffset) * ldb_shared + (threadIdx.x + bColOffset);
                half buff;
                if (loadBRow + bRowOffset >= A_cols || loadBCol + bColOffset >= B_cols) {
                    buff = half(0.0f);
                } else {
                    buff = B[(loadBRow + bRowOffset) * ld_B + (loadBCol + bColOffset)];
                }
                B_shared[bSharedIndex] = buff;
            }
        }
        __syncthreads();

        // Iterate through the memory doing mma, with ratio of 1 load of A to ACC_WIDTH loads of shared_B for ACC_WIDTH mma's
        int bColStart = blockIdx.x * ACC_WIDTH * 16;
        if (loadARow < A_rows) {
#pragma unroll 8
            for (int aColOffset = 0; aColOffset < 128; aColOffset += 16) {
                if (loadACol + aColOffset >= A_cols)
                    break;

#pragma unroll 8
                for (int bufferRow = threadIdx.x / 16; bufferRow < 16; bufferRow += 2) {
                    half buff;
                    if ((loadARow + bufferRow) >= A_rows || (loadACol + aColOffset + (threadIdx.x % 16)) >= A_cols) {
                        buff = half(0.0f);
                    } else {
                        buff = A[(loadARow + bufferRow) * ld_A + loadACol + aColOffset + threadIdx.x % 16];
                    }
                    buffer_shared[threadIdx.y * (16 * 24) + bufferRow * 24 + threadIdx.x % 16] = buff;
                }
                wmma::load_matrix_sync(a_frag, buffer_shared + threadIdx.y * (16 * 24), 24);

#pragma unroll 6
                for (int bColTileOffset = 0; bColTileOffset < ACC_WIDTH; bColTileOffset += 1, bColStart += 16) {
                    wmma::load_matrix_sync(b_frag, B_shared + aColOffset * ldb_shared + (bColTileOffset * 16), ldb_shared);
                    wmma::mma_sync(acc_frag[bColTileOffset], a_frag, b_frag, acc_frag[bColTileOffset]);
                }
            }
        }
    }

    // Convert to half and write the result to memory
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> acc_frag_half;
    const int CSharedCol = threadIdx.x % 16;
#pragma unroll 6
    for (int acc = 0; acc < ACC_WIDTH; ++acc) {
        for (int8_t i = 0; i < acc_frag_half.num_elements; ++i)
            acc_frag_half.x[i] = (half)acc_frag[acc].x[i];
        __syncthreads();
        wmma::store_matrix_sync(buffer_shared + threadIdx.y * (16 * 24), acc_frag_half, 24, wmma::mem_row_major);

        int warpStartCCol = (blockIdx.x * ACC_WIDTH + acc) * 16;
        if (warpStartCCol >= B_cols)
            break;
        int CSharedRow = threadIdx.y * 2 + threadIdx.x / 16;
        int CRow = blockIdx.y * 128 + CSharedRow;
        int CCol = warpStartCCol + CSharedCol;

        __syncthreads();

        if (CCol < B_cols) {
#pragma unroll 8
            for (int i = 0; i < 8 && CRow < A_rows; ++i) {
                C[CRow * ld_C + CCol] = buffer_shared[i * (16 * 24) + CSharedRow * 24 + CSharedCol];
                CRow += 16;
            }
        }
    }
}

__global__ void tensorCoreMatrixMultiplyKernel_Arow128_Bcol48_withReduce_restrictionAMult16x16(const half *A,
                                                                                               const half *B,
                                                                                               half *C,
                                                                                               half *C_workspace,
                                                                                               const int32_t A_rows,
                                                                                               const int32_t A_cols,
                                                                                               const int32_t B_cols,
                                                                                               const int32_t ld_A,
                                                                                               const int32_t ld_B,
                                                                                               const int32_t ld_C) {
    __shared__ half B_shared[128 * ldb_shared];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag[ACC_WIDTH];

    int kDistanceRaw = (A_cols + (gridDim.z - 1)) / gridDim.z;
    int kDistanceMod16 = 16 * ((kDistanceRaw + 15) / 16);
    int kEnd = ((blockIdx.z + 1) * kDistanceMod16) - 1;
    if (kEnd >= A_cols)
        kEnd = A_cols - 1;

    int loadARow = blockIdx.y * 128 + threadIdx.y * 16;
    int loadACol = blockIdx.z * kDistanceMod16;                // counts along k
    int loadBCol = blockIdx.x * ACC_WIDTH * 16 + threadIdx.x;  // iterates to fill ACC_WIDTH*16
    int loadBRowShared = threadIdx.y;
    int loadBRow = loadBRowShared + blockIdx.z * kDistanceMod16;  // counts along k

    for (int j = 0; j < ACC_WIDTH; ++j) {
        wmma::fill_fragment(acc_frag[j], 0.0f);
    }

#pragma unroll 8
    for (; loadACol < A_cols; loadACol += 128, loadBRow += 128) {
        // Load a chunk of B to shared
        __syncthreads();
#pragma unroll 16
        for (int bRowOffset = 0; bRowOffset < 128; bRowOffset += 8) {
#pragma unroll 5
            for (int bColOffset = 0; bColOffset < ACC_WIDTH * 16; bColOffset += 32) {
                // Note: detecting this:
                // if ( ((loadBRow + bRowOffset)&0xFFFFFFF0) > kEnd || ((loadBCol + bColOffset)&0xFFFFFFF0) >= B_cols ) break;
                // causes fifty percent kernel slowdown

                int bSharedIndex = (loadBRowShared + bRowOffset) * ldb_shared + (threadIdx.x + bColOffset);
                half buff;
                if (loadBRow + bRowOffset > kEnd || loadBCol + bColOffset >= B_cols) {
                    buff = half(0.0f);
                } else {
                    buff = B[(loadBRow + bRowOffset) * ld_B + (loadBCol + bColOffset)];
                }
                B_shared[bSharedIndex] = buff;
            }
        }
        __syncthreads();

        // Iterate through the memory doing mma, with ratio of 1 load of A to ACC_WIDTH loads of shared_B for ACC_WIDTH mma's
        if (loadARow < A_rows) {
#pragma unroll 8
            for (int aColOffset = 0; aColOffset < 128; aColOffset += 16) {
                if (loadACol + aColOffset > kEnd)
                    break;

                wmma::load_matrix_sync(a_frag, A + loadARow * ld_A + loadACol + aColOffset, ld_A);

#pragma unroll 6
                for (int bColTileOffset = 0; bColTileOffset < ACC_WIDTH; bColTileOffset += 1) {
                    // Note: detecting this:
                    // if(bColStart >= B_cols) break;
                    // causes a five percent slowdown

                    wmma::load_matrix_sync(b_frag, B_shared + aColOffset * ldb_shared + (bColTileOffset * 16), ldb_shared);
                    wmma::mma_sync(acc_frag[bColTileOffset], a_frag, b_frag, acc_frag[bColTileOffset]);
                }
            }
        }
    }
    __syncthreads();

    // Convert to half and write the result to memory
    long workspaceOffset;
    if (blockIdx.z != 0)
        workspaceOffset = (blockIdx.z - 1) * (A_rows * ld_C);
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> acc_frag_half;
    const int CSharedCol = threadIdx.x % 16;
#pragma unroll 6
    for (int acc = 0; acc < ACC_WIDTH; ++acc) {
        for (int8_t i = 0; i < acc_frag_half.num_elements; ++i)
            acc_frag_half.x[i] = (half)acc_frag[acc].x[i];
        __syncthreads();
        wmma::store_matrix_sync(B_shared + threadIdx.y * (16 * 24), acc_frag_half, 24, wmma::mem_row_major);

        int warpStartCCol = (blockIdx.x * ACC_WIDTH + acc) * 16;
        if (warpStartCCol >= B_cols)
            break;
        int CSharedRow = threadIdx.y * 2 + threadIdx.x / 16;
        int CRow = blockIdx.y * 128 + CSharedRow;
        int CCol = warpStartCCol + CSharedCol;

        __syncthreads();

        if (CCol < B_cols) {
            if (blockIdx.z == 0) {
#pragma unroll 8
                for (int i = 0; i < 8 && CRow < A_rows; ++i) {
                    C[CRow * ld_C + CCol] = B_shared[i * (16 * 24) + CSharedRow * 24 + CSharedCol];
                    CRow += 16;
                }
            } else {
                for (int i = 0; i < 8 && CRow < A_rows; ++i) {
                    C_workspace[workspaceOffset + CRow * ld_C + CCol] = B_shared[i * (16 * 24) + CSharedRow * 24 + CSharedCol];
                    CRow += 16;
                }
            }
        }
    }
}

__global__ void tensorCoreMatrixMultiplyKernel_Arow128_Bcol48_withReduce_noRestriction(const half *A,
                                                                                       const half *B,
                                                                                       half *C,
                                                                                       half *C_workspace,
                                                                                       const int32_t A_rows,
                                                                                       const int32_t A_cols,
                                                                                       const int32_t B_cols,
                                                                                       const int32_t ld_A,
                                                                                       const int32_t ld_B,
                                                                                       const int32_t ld_C) {
    __shared__ half B_shared[128 * ldb_shared];
    __shared__ half buffer_shared[8 * 16 * 24];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag[ACC_WIDTH];

    int kDistanceRaw = (A_cols + (gridDim.z - 1)) / gridDim.z;
    int kDistanceMod16 = 16 * ((kDistanceRaw + 15) / 16);
    int kEnd = ((blockIdx.z + 1) * kDistanceMod16) - 1;
    if (kEnd >= A_cols)
        kEnd = A_cols - 1;

    int loadARow = blockIdx.y * 128 + threadIdx.y * 16;
    int loadACol = blockIdx.z * kDistanceMod16;                // counts along k
    int loadBCol = blockIdx.x * ACC_WIDTH * 16 + threadIdx.x;  // iterates to fill ACC_WIDTH*16
    int loadBRowShared = threadIdx.y;
    int loadBRow = loadBRowShared + blockIdx.z * kDistanceMod16;  // counts along k

    for (int j = 0; j < ACC_WIDTH; ++j) {
        wmma::fill_fragment(acc_frag[j], 0.0f);
    }

    for (; loadACol < A_cols; loadACol += 128, loadBRow += 128) {
        // Load a chunk of B to shared
        __syncthreads();
#pragma unroll 16
        for (int bRowOffset = 0; bRowOffset < 128; bRowOffset += 8) {
#pragma unroll 5
            for (int bColOffset = 0; bColOffset < ACC_WIDTH * 16; bColOffset += 32) {
                int bSharedIndex = (loadBRowShared + bRowOffset) * ldb_shared + (threadIdx.x + bColOffset);
                half buff;
                if (loadBRow + bRowOffset > kEnd || loadBCol + bColOffset >= B_cols) {
                    buff = half(0.0f);
                } else {
                    buff = B[(loadBRow + bRowOffset) * ld_B + (loadBCol + bColOffset)];
                }
                B_shared[bSharedIndex] = buff;
            }
        }
        __syncthreads();

        // Iterate through the memory doing mma, with ratio of 1 load of A to ACC_WIDTH loads of shared_B for ACC_WIDTH mma's
        int bColStart = blockIdx.x * ACC_WIDTH * 16;
        if (loadARow < A_rows) {
#pragma unroll 8
            for (int aColOffset = 0; aColOffset < 128; aColOffset += 16) {
                if (loadACol + aColOffset > kEnd)
                    break;

#pragma unroll 8
                for (int bufferRow = threadIdx.x / 16; bufferRow < 16; bufferRow += 2) {
                    half buff;
                    if ((loadARow + bufferRow) >= A_rows || (loadACol + aColOffset + (threadIdx.x % 16)) > kEnd) {
                        buff = half(0.0f);
                    } else {
                        buff = A[(loadARow + bufferRow) * ld_A + loadACol + aColOffset + threadIdx.x % 16];
                    }
                    buffer_shared[threadIdx.y * (16 * 24) + bufferRow * 24 + threadIdx.x % 16] = buff;
                }
                wmma::load_matrix_sync(a_frag, buffer_shared + threadIdx.y * (16 * 24), 24);

#pragma unroll 6
                for (int bColTileOffset = 0; bColTileOffset < ACC_WIDTH; bColTileOffset += 1, bColStart += 16) {
                    wmma::load_matrix_sync(b_frag, B_shared + aColOffset * ldb_shared + (bColTileOffset * 16), ldb_shared);
                    wmma::mma_sync(acc_frag[bColTileOffset], a_frag, b_frag, acc_frag[bColTileOffset]);
                }
            }
        }
    }

    // Convert to half and write the result to memory
    long workspaceOffset;
    if (blockIdx.z != 0)
        workspaceOffset = (blockIdx.z - 1) * (A_rows * ld_C);
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> acc_frag_half;
    const int CSharedCol = threadIdx.x % 16;
#pragma unroll 6
    for (int acc = 0; acc < ACC_WIDTH; ++acc) {
        for (int8_t i = 0; i < acc_frag_half.num_elements; ++i)
            acc_frag_half.x[i] = (half)acc_frag[acc].x[i];
        __syncthreads();
        wmma::store_matrix_sync(buffer_shared + threadIdx.y * (16 * 24), acc_frag_half, 24, wmma::mem_row_major);

        int warpStartCCol = (blockIdx.x * ACC_WIDTH + acc) * 16;
        if (warpStartCCol >= B_cols)
            break;
        int CSharedRow = threadIdx.y * 2 + threadIdx.x / 16;
        int CRow = blockIdx.y * 128 + CSharedRow;
        int CCol = warpStartCCol + CSharedCol;

        __syncthreads();

        if (CCol < B_cols) {
            if (blockIdx.z == 0) {
#pragma unroll 8
                for (int i = 0; i < 8 && CRow < A_rows; ++i) {
                    C[CRow * ld_C + CCol] = buffer_shared[i * (16 * 24) + CSharedRow * 24 + CSharedCol];
                    CRow += 16;
                }
            } else {
                for (int i = 0; i < 8 && CRow < A_rows; ++i) {
                    C_workspace[workspaceOffset + CRow * ld_C + CCol] = buffer_shared[i * (16 * 24) + CSharedRow * 24 + CSharedCol];
                    CRow += 16;
                }
            }
        }
    }
}

#undef ACC_HEIGHT
#undef ACC_WIDTH
#undef ldb_shared

#define ACC_WIDTH 3
#define SPLIT_K 2
#define ACC_HEIGHT "hard coded to 1"
#define ldb_shared (32 * ((ACC_WIDTH * 16 + 31) / 32) + 8)
#define BLOCK_A_HEIGHT (128 / SPLIT_K)

__global__ void tensorCoreMatrixMultiplyKernel_Arow64_Bcol48_restrictionAMult16x16(const half *A,
                                                                                   const half *B,
                                                                                   half *C,
                                                                                   const int32_t A_rows,
                                                                                   const int32_t A_cols,
                                                                                   const int32_t B_cols,
                                                                                   const int32_t ld_A,
                                                                                   const int32_t ld_B,
                                                                                   const int32_t ld_C) {
    __shared__ half B_shared[128 * ldb_shared];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag[ACC_WIDTH];

    int threadTileCol = threadIdx.y % SPLIT_K;
    int threadTileRow = threadIdx.y / SPLIT_K;

    int loadARow = blockIdx.y * BLOCK_A_HEIGHT + threadTileRow * 16;
    int loadACol = 0;                                          // counts along k
    int loadBCol = blockIdx.x * ACC_WIDTH * 16 + threadIdx.x;  // iterates to fill ACC_WIDTH*16
    int loadBRow = threadIdx.y;                                // counts along k

    for (int j = 0; j < ACC_WIDTH; ++j) {
        wmma::fill_fragment(acc_frag[j], 0.0f);
    }

#pragma unroll 8
    for (; loadACol < A_cols; loadACol += 128, loadBRow += 128) {
        // Load a chunk of B to shared
        __syncthreads();
#pragma unroll 16
        for (int bRowOffset = 0; bRowOffset < 128; bRowOffset += 8) {
#pragma unroll 5
            for (int bColOffset = 0; bColOffset < ACC_WIDTH * 16; bColOffset += 32) {
                // Note: detecting this:
                // if ( ((loadBRow + bRowOffset)&0xFFFFFFF0) >= A_cols || ((loadBCol + bColOffset)&0xFFFFFFF0) >= B_cols ) break;
                // causes fifty percent kernel slowdown

                int bSharedIndex = (threadIdx.y + bRowOffset) * ldb_shared + (threadIdx.x + bColOffset);
                half buff;
                if (loadBRow + bRowOffset >= A_cols || loadBCol + bColOffset >= B_cols) {
                    buff = half(0.0f);
                } else {
                    buff = B[(loadBRow + bRowOffset) * ld_B + (loadBCol + bColOffset)];
                }
                B_shared[bSharedIndex] = buff;
            }
        }
        __syncthreads();

        // Iterate through the memory doing mma, with ratio of 1 load of A to ACC_WIDTH loads of shared_B for ACC_WIDTH mma's
        if (loadARow < A_rows) {
#pragma unroll 2
            for (int aColOffset = threadTileCol * 16; aColOffset < 128; aColOffset += 16 * SPLIT_K) {
                if (loadACol + aColOffset >= A_cols)
                    break;
                // printf("t.y %d t.x %d ARow %d ACol %d\n", threadIdx.y, threadIdx.x, loadARow, loadACol + aColOffset);

                wmma::load_matrix_sync(a_frag, A + loadARow * ld_A + loadACol + aColOffset, ld_A);

#pragma unroll 6
                for (int bColTileOffset = 0; bColTileOffset < ACC_WIDTH; bColTileOffset += 1) {
                    // Note: detecting this:
                    // if(bColStart >= B_cols) break;
                    // causes a five percent slowdown

                    wmma::load_matrix_sync(b_frag, B_shared + aColOffset * ldb_shared + (bColTileOffset * 16), ldb_shared);
                    wmma::mma_sync(acc_frag[bColTileOffset], a_frag, b_frag, acc_frag[bColTileOffset]);
                }
            }
        }
    }

    float *sharedBuffer = (float *)B_shared;

    // Reduce, writing the result as a half to C
    // I have SPLIT_K frags to combine to form the final result for the frag's part of C
#pragma unroll 6
    for (int i = 0; i < ACC_WIDTH; ++i) {
        int CCol = blockIdx.x * ACC_WIDTH * 16 + i * 16 + threadIdx.x % 16;
        if ((CCol & 0xFFFFFFF0) >= C_cols)
            break;

        __syncthreads();
        wmma::store_matrix_sync(sharedBuffer + threadTileRow * 16 * (SPLIT_K * 16 + 4) + threadTileCol * 16,
                                acc_frag[i],
                                SPLIT_K * 16 + 4,
                                wmma::mem_row_major);
        __syncthreads();

        int CRowStart = blockIdx.y * BLOCK_A_HEIGHT;
#pragma unroll 4
        for (int reduceRow = threadIdx.y * 2 + threadIdx.x / 16; reduceRow < BLOCK_A_HEIGHT; reduceRow += 16) {
            if (CRowStart + reduceRow >= C_rows)
                break;

            // Sum SPLIT_K instances of rows of 16 threads, then write the result to memory
            float f = 0.0f;
            for (int reduceCol = threadIdx.x % 16; reduceCol < SPLIT_K * 16; reduceCol += 16) {
                f += sharedBuffer[reduceRow * (SPLIT_K * 16 + 4) + reduceCol];
            }

            if ((CRowStart + reduceRow) < C_rows && CCol < C_cols)
                C[(CRowStart + reduceRow) * ld_C + CCol] = (half)f;
        }
    }
}

__global__ void tensorCoreMatrixMultiplyKernel_Arow64_Bcol48_noRestriction(const half *A,
                                                                           const half *B,
                                                                           half *C,
                                                                           const int32_t A_rows,
                                                                           const int32_t A_cols,
                                                                           const int32_t B_cols,
                                                                           const int32_t ld_A,
                                                                           const int32_t ld_B,
                                                                           const int32_t ld_C) {
    __shared__ half B_shared[128 * ldb_shared];
    __shared__ half A_shared[8 * 16 * 24];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag[ACC_WIDTH];

    int threadTileCol = threadIdx.y % SPLIT_K;
    int threadTileRow = threadIdx.y / SPLIT_K;

    int loadARow = blockIdx.y * BLOCK_A_HEIGHT + threadTileRow * 16;
    int loadACol = 0;                                          // counts along k
    int loadBCol = blockIdx.x * ACC_WIDTH * 16 + threadIdx.x;  // iterates to fill ACC_WIDTH*16
    int loadBRow = threadIdx.y;                                // counts along k

    for (int j = 0; j < ACC_WIDTH; ++j) {
        wmma::fill_fragment(acc_frag[j], 0.0f);
    }

#pragma unroll 8
    for (; loadACol < A_cols; loadACol += 128, loadBRow += 128) {
        // Load a chunk of B to shared
        __syncthreads();
#pragma unroll 16
        for (int bRowOffset = 0; bRowOffset < 128; bRowOffset += 8) {
#pragma unroll 5
            for (int bColOffset = 0; bColOffset < ACC_WIDTH * 16; bColOffset += 32) {
                // Note: detecting this:
                // if ( ((loadBRow + bRowOffset)&0xFFFFFFF0) >= A_cols || ((loadBCol + bColOffset)&0xFFFFFFF0) >= B_cols ) break;
                // causes fifty percent kernel slowdown

                int bSharedIndex = (threadIdx.y + bRowOffset) * ldb_shared + (threadIdx.x + bColOffset);
                half buff;
                if (loadBRow + bRowOffset >= A_cols || loadBCol + bColOffset >= B_cols) {
                    buff = half(0.0f);
                } else {
                    buff = B[(loadBRow + bRowOffset) * ld_B + (loadBCol + bColOffset)];
                }
                B_shared[bSharedIndex] = buff;
            }
        }
        __syncthreads();

        // Iterate through the memory doing mma, with ratio of 1 load of A to ACC_WIDTH loads of shared_B for ACC_WIDTH mma's
        if (loadARow < A_rows) {
#pragma unroll 8
            for (int aColOffset = threadTileCol * 16; aColOffset < 128; aColOffset += 16 * SPLIT_K) {
                if (loadACol + aColOffset >= A_cols)
                    break;

                for (int bufferRow = threadIdx.x / 16; bufferRow < 16; bufferRow += 2) {
                    half buff;
                    if ((loadARow + bufferRow) >= A_rows || (loadACol + aColOffset + (threadIdx.x % 16)) >= A_cols) {
                        buff = half(0.0f);
                    } else {
                        buff = A[(loadARow + bufferRow) * ld_A + loadACol + aColOffset + threadIdx.x % 16];
                    }
                    A_shared[threadIdx.y * (16 * 24) + bufferRow * 24 + threadIdx.x % 16] = buff;
                }
                wmma::load_matrix_sync(a_frag, A_shared + threadIdx.y * (16 * 24), 24);

#pragma unroll 6
                for (int bColTileOffset = 0; bColTileOffset < ACC_WIDTH; bColTileOffset += 1) {
                    // Note: detecting this:
                    // if(bColStart >= B_cols) break;
                    // causes a five percent slowdown

                    wmma::load_matrix_sync(b_frag, B_shared + aColOffset * ldb_shared + (bColTileOffset * 16), ldb_shared);
                    wmma::mma_sync(acc_frag[bColTileOffset], a_frag, b_frag, acc_frag[bColTileOffset]);
                }
            }
        }
    }

    float *sharedBuffer = (float *)B_shared;

    // Reduce, writing the result as a half to C
    // I have SPLIT_K frags to combine to form the final result for the frag's part of C
#pragma unroll 6
    for (int i = 0; i < ACC_WIDTH; ++i) {
        int CCol = blockIdx.x * ACC_WIDTH * 16 + i * 16 + threadIdx.x % 16;
        if ((CCol & 0xFFFFFFF0) >= C_cols)
            break;

        __syncthreads();
        wmma::store_matrix_sync(sharedBuffer + threadTileRow * 16 * (SPLIT_K * 16 + 4) + threadTileCol * 16,
                                acc_frag[i],
                                SPLIT_K * 16 + 4,
                                wmma::mem_row_major);
        __syncthreads();

        int CRowStart = blockIdx.y * BLOCK_A_HEIGHT;
#pragma unroll 4
        for (int reduceRow = threadIdx.y * 2 + threadIdx.x / 16; reduceRow < BLOCK_A_HEIGHT; reduceRow += 16) {
            if (CRowStart + reduceRow >= C_rows)
                break;

            // Sum SPLIT_K instances of rows of 16 threads, then write the result to memory
            float f = 0.0f;
            for (int reduceCol = threadIdx.x % 16; reduceCol < SPLIT_K * 16; reduceCol += 16) {
                f += sharedBuffer[reduceRow * (SPLIT_K * 16 + 4) + reduceCol];
            }

            if ((CRowStart + reduceRow) < C_rows && CCol < C_cols)
                C[(CRowStart + reduceRow) * ld_C + CCol] = (half)f;
        }
    }
}

#undef ACC_HEIGHT
#undef SPLIT_K
#undef ACC_WIDTH
#undef ldb_shared
#undef BLOCK_A_HEIGHT

#define ACC_WIDTH 3
#define SPLIT_K 4
#define ACC_HEIGHT "hard coded to 1"
#define ldb_shared (32 * ((ACC_WIDTH * 16 + 31) / 32) + 8)
#define BLOCK_A_HEIGHT (128 / SPLIT_K)

__global__ void tensorCoreMatrixMultiplyKernel_Arow32_Bcol48_restrictionAMult16x16(const half *A,
                                                                                   const half *B,
                                                                                   half *C,
                                                                                   const int32_t A_rows,
                                                                                   const int32_t A_cols,
                                                                                   const int32_t B_cols,
                                                                                   const int32_t ld_A,
                                                                                   const int32_t ld_B,
                                                                                   const int32_t ld_C) {
    __shared__ half B_shared[128 * ldb_shared];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag[ACC_WIDTH];

    int threadTileCol = threadIdx.y % SPLIT_K;
    int threadTileRow = threadIdx.y / SPLIT_K;

    int loadARow = blockIdx.y * BLOCK_A_HEIGHT + threadTileRow * 16;
    int loadACol = 0;                                          // counts along k
    int loadBCol = blockIdx.x * ACC_WIDTH * 16 + threadIdx.x;  // iterates to fill ACC_WIDTH*16
    int loadBRow = threadIdx.y;                                // counts along k

    for (int j = 0; j < ACC_WIDTH; ++j) {
        wmma::fill_fragment(acc_frag[j], 0.0f);
    }

#pragma unroll 8
    for (; loadACol < A_cols; loadACol += 128, loadBRow += 128) {
        // Load a chunk of B to shared
        __syncthreads();
#pragma unroll 16
        for (int bRowOffset = 0; bRowOffset < 128; bRowOffset += 8) {
#pragma unroll 5
            for (int bColOffset = 0; bColOffset < ACC_WIDTH * 16; bColOffset += 32) {
                // Note: detecting this:
                // if ( ((loadBRow + bRowOffset)&0xFFFFFFF0) >= A_cols || ((loadBCol + bColOffset)&0xFFFFFFF0) >= B_cols ) break;
                // causes fifty percent kernel slowdown

                int bSharedIndex = (threadIdx.y + bRowOffset) * ldb_shared + (threadIdx.x + bColOffset);
                half buff;
                if (loadBRow + bRowOffset >= A_cols || loadBCol + bColOffset >= B_cols) {
                    buff = half(0.0f);
                } else {
                    buff = B[(loadBRow + bRowOffset) * ld_B + (loadBCol + bColOffset)];
                }
                B_shared[bSharedIndex] = buff;
            }
        }
        __syncthreads();

        // Iterate through the memory doing mma, with ratio of 1 load of A to ACC_WIDTH loads of shared_B for ACC_WIDTH mma's
        if (loadARow < A_rows) {
#pragma unroll 2
            for (int aColOffset = threadTileCol * 16; aColOffset < 128; aColOffset += 16 * SPLIT_K) {
                if (loadACol + aColOffset >= A_cols)
                    break;
                // printf("t.y %d t.x %d ARow %d ACol %d\n", threadIdx.y, threadIdx.x, loadARow, loadACol + aColOffset);

                wmma::load_matrix_sync(a_frag, A + loadARow * ld_A + loadACol + aColOffset, ld_A);

#pragma unroll 6
                for (int bColTileOffset = 0; bColTileOffset < ACC_WIDTH; bColTileOffset += 1) {
                    // Note: detecting this:
                    // if(bColStart >= B_cols) break;
                    // causes a five percent slowdown

                    wmma::load_matrix_sync(b_frag, B_shared + aColOffset * ldb_shared + (bColTileOffset * 16), ldb_shared);
                    wmma::mma_sync(acc_frag[bColTileOffset], a_frag, b_frag, acc_frag[bColTileOffset]);
                }
            }
        }
    }

    float *sharedBuffer = (float *)B_shared;

    // Reduce, writing the result as a half to C
    // I have SPLIT_K frags to combine to form the final result for the frag's part of C
#pragma unroll 6
    for (int i = 0; i < ACC_WIDTH; ++i) {
        int CCol = blockIdx.x * ACC_WIDTH * 16 + i * 16 + threadIdx.x % 16;
        if ((CCol & 0xFFFFFFF0) >= C_cols)
            break;

        __syncthreads();
        wmma::store_matrix_sync(sharedBuffer + threadTileRow * 16 * (SPLIT_K * 16 + 4) + threadTileCol * 16,
                                acc_frag[i],
                                SPLIT_K * 16 + 4,
                                wmma::mem_row_major);
        __syncthreads();

        int CRowStart = blockIdx.y * BLOCK_A_HEIGHT;
#pragma unroll 4
        for (int reduceRow = threadIdx.y * 2 + threadIdx.x / 16; reduceRow < BLOCK_A_HEIGHT; reduceRow += 16) {
            if (CRowStart + reduceRow >= C_rows)
                break;

            // Sum SPLIT_K instances of rows of 16 threads, then write the result to memory
            float f = 0.0f;
            for (int reduceCol = threadIdx.x % 16; reduceCol < SPLIT_K * 16; reduceCol += 16) {
                f += sharedBuffer[reduceRow * (SPLIT_K * 16 + 4) + reduceCol];
            }

            if ((CRowStart + reduceRow) < C_rows && CCol < C_cols)
                C[(CRowStart + reduceRow) * ld_C + CCol] = (half)f;
        }
    }
}

__global__ void tensorCoreMatrixMultiplyKernel_Arow32_Bcol48_noRestriction(const half *A,
                                                                           const half *B,
                                                                           half *C,
                                                                           const int32_t A_rows,
                                                                           const int32_t A_cols,
                                                                           const int32_t B_cols,
                                                                           const int32_t ld_A,
                                                                           const int32_t ld_B,
                                                                           const int32_t ld_C) {
    __shared__ half B_shared[128 * ldb_shared];
    __shared__ half A_shared[8 * 16 * 24];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag[ACC_WIDTH];

    int threadTileCol = threadIdx.y % SPLIT_K;
    int threadTileRow = threadIdx.y / SPLIT_K;

    int loadARow = blockIdx.y * BLOCK_A_HEIGHT + threadTileRow * 16;
    int loadACol = 0;                                          // counts along k
    int loadBCol = blockIdx.x * ACC_WIDTH * 16 + threadIdx.x;  // iterates to fill ACC_WIDTH*16
    int loadBRow = threadIdx.y;                                // counts along k

    for (int j = 0; j < ACC_WIDTH; ++j) {
        wmma::fill_fragment(acc_frag[j], 0.0f);
    }

#pragma unroll 8
    for (; loadACol < A_cols; loadACol += 128, loadBRow += 128) {
        // Load a chunk of B to shared
        __syncthreads();
#pragma unroll 16
        for (int bRowOffset = 0; bRowOffset < 128; bRowOffset += 8) {
#pragma unroll 5
            for (int bColOffset = 0; bColOffset < ACC_WIDTH * 16; bColOffset += 32) {
                // Note: detecting this:
                // if ( ((loadBRow + bRowOffset)&0xFFFFFFF0) >= A_cols || ((loadBCol + bColOffset)&0xFFFFFFF0) >= B_cols ) break;
                // causes fifty percent kernel slowdown

                int bSharedIndex = (threadIdx.y + bRowOffset) * ldb_shared + (threadIdx.x + bColOffset);
                half buff;
                if (loadBRow + bRowOffset >= A_cols || loadBCol + bColOffset >= B_cols) {
                    buff = half(0.0f);
                } else {
                    buff = B[(loadBRow + bRowOffset) * ld_B + (loadBCol + bColOffset)];
                }
                B_shared[bSharedIndex] = buff;
            }
        }
        __syncthreads();

        // Iterate through the memory doing mma, with ratio of 1 load of A to ACC_WIDTH loads of shared_B for ACC_WIDTH mma's
        if (loadARow < A_rows) {
#pragma unroll 8
            for (int aColOffset = threadTileCol * 16; aColOffset < 128; aColOffset += 16 * SPLIT_K) {
                if (loadACol + aColOffset >= A_cols)
                    break;

                for (int bufferRow = threadIdx.x / 16; bufferRow < 16; bufferRow += 2) {
                    half buff;
                    if ((loadARow + bufferRow) >= A_rows || (loadACol + aColOffset + (threadIdx.x % 16)) >= A_cols) {
                        buff = half(0.0f);
                    } else {
                        buff = A[(loadARow + bufferRow) * ld_A + loadACol + aColOffset + threadIdx.x % 16];
                    }
                    A_shared[threadIdx.y * (16 * 24) + bufferRow * 24 + threadIdx.x % 16] = buff;
                }
                wmma::load_matrix_sync(a_frag, A_shared + threadIdx.y * (16 * 24), 24);

#pragma unroll 6
                for (int bColTileOffset = 0; bColTileOffset < ACC_WIDTH; bColTileOffset += 1) {
                    // Note: detecting this:
                    // if(bColStart >= B_cols) break;
                    // causes a five percent slowdown

                    wmma::load_matrix_sync(b_frag, B_shared + aColOffset * ldb_shared + (bColTileOffset * 16), ldb_shared);
                    wmma::mma_sync(acc_frag[bColTileOffset], a_frag, b_frag, acc_frag[bColTileOffset]);
                }
            }
        }
    }

    float *sharedBuffer = (float *)B_shared;

    // Reduce, writing the result as a half to C
    // I have SPLIT_K frags to combine to form the final result for the frag's part of C
#pragma unroll 6
    for (int i = 0; i < ACC_WIDTH; ++i) {
        int CCol = blockIdx.x * ACC_WIDTH * 16 + i * 16 + threadIdx.x % 16;
        if ((CCol & 0xFFFFFFF0) >= C_cols)
            break;

        __syncthreads();
        wmma::store_matrix_sync(sharedBuffer + threadTileRow * 16 * (SPLIT_K * 16 + 4) + threadTileCol * 16,
                                acc_frag[i],
                                SPLIT_K * 16 + 4,
                                wmma::mem_row_major);
        __syncthreads();

        int CRowStart = blockIdx.y * BLOCK_A_HEIGHT;
#pragma unroll 4
        for (int reduceRow = threadIdx.y * 2 + threadIdx.x / 16; reduceRow < BLOCK_A_HEIGHT; reduceRow += 16) {
            if (CRowStart + reduceRow >= C_rows)
                break;

            // Sum SPLIT_K instances of rows of 16 threads, then write the result to memory
            float f = 0.0f;
            for (int reduceCol = threadIdx.x % 16; reduceCol < SPLIT_K * 16; reduceCol += 16) {
                f += sharedBuffer[reduceRow * (SPLIT_K * 16 + 4) + reduceCol];
            }

            if ((CRowStart + reduceRow) < C_rows && CCol < C_cols)
                C[(CRowStart + reduceRow) * ld_C + CCol] = (half)f;
        }
    }
}

#undef ACC_HEIGHT
#undef SPLIT_K
#undef ACC_WIDTH
#undef ldb_shared
#undef BLOCK_A_HEIGHT

#define ACC_WIDTH 3
#define SPLIT_K 8
#define ACC_HEIGHT "hard coded to 1"
#define ldb_shared (32 * ((ACC_WIDTH * 16 + 31) / 32) + 8)
#define BLOCK_A_HEIGHT (128 / SPLIT_K)

__global__ void tensorCoreMatrixMultiplyKernel_Arow16_Bcol48_restrictionAMult16x16(const half *A,
                                                                                   const half *B,
                                                                                   half *C,
                                                                                   const int32_t A_rows,
                                                                                   const int32_t A_cols,
                                                                                   const int32_t B_cols,
                                                                                   const int32_t ld_A,
                                                                                   const int32_t ld_B,
                                                                                   const int32_t ld_C) {
    __shared__ half B_shared[128 * ldb_shared];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag[ACC_WIDTH];

    int threadTileCol = threadIdx.y % SPLIT_K;
    int threadTileRow = threadIdx.y / SPLIT_K;

    int loadARow = blockIdx.y * BLOCK_A_HEIGHT + threadTileRow * 16;
    int loadACol = 0;                                          // counts along k
    int loadBCol = blockIdx.x * ACC_WIDTH * 16 + threadIdx.x;  // iterates to fill ACC_WIDTH*16
    int loadBRow = threadIdx.y;                                // counts along k

    for (int j = 0; j < ACC_WIDTH; ++j) {
        wmma::fill_fragment(acc_frag[j], 0.0f);
    }

#pragma unroll 8
    for (; loadACol < A_cols; loadACol += 128, loadBRow += 128) {
        // Load a chunk of B to shared
        __syncthreads();
#pragma unroll 16
        for (int bRowOffset = 0; bRowOffset < 128; bRowOffset += 8) {
#pragma unroll 5
            for (int bColOffset = 0; bColOffset < ACC_WIDTH * 16; bColOffset += 32) {
                // Note: detecting this:
                // if ( ((loadBRow + bRowOffset)&0xFFFFFFF0) >= A_cols || ((loadBCol + bColOffset)&0xFFFFFFF0) >= B_cols ) break;
                // causes fifty percent kernel slowdown

                int bSharedIndex = (threadIdx.y + bRowOffset) * ldb_shared + (threadIdx.x + bColOffset);
                half buff;
                if (loadBRow + bRowOffset >= A_cols || loadBCol + bColOffset >= B_cols) {
                    buff = half(0.0f);
                } else {
                    buff = B[(loadBRow + bRowOffset) * ld_B + (loadBCol + bColOffset)];
                }
                B_shared[bSharedIndex] = buff;
            }
        }
        __syncthreads();

        // Iterate through the memory doing mma, with ratio of 1 load of A to ACC_WIDTH loads of shared_B for ACC_WIDTH mma's
        if (loadARow < A_rows) {
#pragma unroll 2
            for (int aColOffset = threadTileCol * 16; aColOffset < 128; aColOffset += 16 * SPLIT_K) {
                if (loadACol + aColOffset >= A_cols)
                    break;
                // printf("t.y %d t.x %d ARow %d ACol %d\n", threadIdx.y, threadIdx.x, loadARow, loadACol + aColOffset);

                wmma::load_matrix_sync(a_frag, A + loadARow * ld_A + loadACol + aColOffset, ld_A);

#pragma unroll 6
                for (int bColTileOffset = 0; bColTileOffset < ACC_WIDTH; bColTileOffset += 1) {
                    // Note: detecting this:
                    // if(bColStart >= B_cols) break;
                    // causes a five percent slowdown

                    wmma::load_matrix_sync(b_frag, B_shared + aColOffset * ldb_shared + (bColTileOffset * 16), ldb_shared);
                    wmma::mma_sync(acc_frag[bColTileOffset], a_frag, b_frag, acc_frag[bColTileOffset]);
                }
            }
        }
    }

    float *sharedBuffer = (float *)B_shared;

    // Reduce, writing the result as a half to C
    // I have SPLIT_K frags to combine to form the final result for the frag's part of C
#pragma unroll 6
    for (int i = 0; i < ACC_WIDTH; ++i) {
        int CCol = blockIdx.x * ACC_WIDTH * 16 + i * 16 + threadIdx.x % 16;
        if ((CCol & 0xFFFFFFF0) >= C_cols)
            break;

        __syncthreads();
        wmma::store_matrix_sync(sharedBuffer + threadTileRow * 16 * (SPLIT_K * 16 + 4) + threadTileCol * 16,
                                acc_frag[i],
                                SPLIT_K * 16 + 4,
                                wmma::mem_row_major);
        __syncthreads();

        int CRowStart = blockIdx.y * BLOCK_A_HEIGHT;
#pragma unroll 4
        for (int reduceRow = threadIdx.y * 2 + threadIdx.x / 16; reduceRow < BLOCK_A_HEIGHT; reduceRow += 16) {
            if (CRowStart + reduceRow >= C_rows)
                break;

            // Sum SPLIT_K instances of rows of 16 threads, then write the result to memory
            float f = 0.0f;
            for (int reduceCol = threadIdx.x % 16; reduceCol < SPLIT_K * 16; reduceCol += 16) {
                f += sharedBuffer[reduceRow * (SPLIT_K * 16 + 4) + reduceCol];
            }

            if ((CRowStart + reduceRow) < C_rows && CCol < C_cols)
                C[(CRowStart + reduceRow) * ld_C + CCol] = (half)f;
        }
    }
}

__global__ void tensorCoreMatrixMultiplyKernel_Arow16_Bcol48_noRestriction(const half *A,
                                                                           const half *B,
                                                                           half *C,
                                                                           const int32_t A_rows,
                                                                           const int32_t A_cols,
                                                                           const int32_t B_cols,
                                                                           const int32_t ld_A,
                                                                           const int32_t ld_B,
                                                                           const int32_t ld_C) {
    __shared__ half B_shared[128 * ldb_shared];
    __shared__ half A_shared[8 * 16 * 24];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag[ACC_WIDTH];

    int threadTileCol = threadIdx.y % SPLIT_K;
    int threadTileRow = threadIdx.y / SPLIT_K;

    int loadARow = blockIdx.y * BLOCK_A_HEIGHT + threadTileRow * 16;
    int loadACol = 0;                                          // counts along k
    int loadBCol = blockIdx.x * ACC_WIDTH * 16 + threadIdx.x;  // iterates to fill ACC_WIDTH*16
    int loadBRow = threadIdx.y;                                // counts along k

    for (int j = 0; j < ACC_WIDTH; ++j) {
        wmma::fill_fragment(acc_frag[j], 0.0f);
    }

#pragma unroll 8
    for (; loadACol < A_cols; loadACol += 128, loadBRow += 128) {
        // Load a chunk of B to shared
        __syncthreads();
#pragma unroll 16
        for (int bRowOffset = 0; bRowOffset < 128; bRowOffset += 8) {
#pragma unroll 5
            for (int bColOffset = 0; bColOffset < ACC_WIDTH * 16; bColOffset += 32) {
                // Note: detecting this:
                // if ( ((loadBRow + bRowOffset)&0xFFFFFFF0) >= A_cols || ((loadBCol + bColOffset)&0xFFFFFFF0) >= B_cols ) break;
                // causes fifty percent kernel slowdown

                int bSharedIndex = (threadIdx.y + bRowOffset) * ldb_shared + (threadIdx.x + bColOffset);
                half buff;
                if (loadBRow + bRowOffset >= A_cols || loadBCol + bColOffset >= B_cols) {
                    buff = half(0.0f);
                } else {
                    buff = B[(loadBRow + bRowOffset) * ld_B + (loadBCol + bColOffset)];
                }
                B_shared[bSharedIndex] = buff;
            }
        }
        __syncthreads();

        // Iterate through the memory doing mma, with ratio of 1 load of A to ACC_WIDTH loads of shared_B for ACC_WIDTH mma's
        if (loadARow < A_rows) {
#pragma unroll 8
            for (int aColOffset = threadTileCol * 16; aColOffset < 128; aColOffset += 16 * SPLIT_K) {
                if (loadACol + aColOffset >= A_cols)
                    break;

                for (int bufferRow = threadIdx.x / 16; bufferRow < 16; bufferRow += 2) {
                    half buff;
                    if ((loadARow + bufferRow) >= A_rows || (loadACol + aColOffset + (threadIdx.x % 16)) >= A_cols) {
                        buff = half(0.0f);
                    } else {
                        buff = A[(loadARow + bufferRow) * ld_A + loadACol + aColOffset + threadIdx.x % 16];
                    }
                    A_shared[threadIdx.y * (16 * 24) + bufferRow * 24 + threadIdx.x % 16] = buff;
                }
                wmma::load_matrix_sync(a_frag, A_shared + threadIdx.y * (16 * 24), 24);

#pragma unroll 6
                for (int bColTileOffset = 0; bColTileOffset < ACC_WIDTH; bColTileOffset += 1) {
                    // Note: detecting this:
                    // if(bColStart >= B_cols) break;
                    // causes a five percent slowdown

                    wmma::load_matrix_sync(b_frag, B_shared + aColOffset * ldb_shared + (bColTileOffset * 16), ldb_shared);
                    wmma::mma_sync(acc_frag[bColTileOffset], a_frag, b_frag, acc_frag[bColTileOffset]);
                }
            }
        }
    }

    float *sharedBuffer = (float *)B_shared;

    // Reduce, writing the result as a half to C
    // I have SPLIT_K frags to combine to form the final result for the frag's part of C
#pragma unroll 6
    for (int i = 0; i < ACC_WIDTH; ++i) {
        int CCol = blockIdx.x * ACC_WIDTH * 16 + i * 16 + threadIdx.x % 16;
        if ((CCol & 0xFFFFFFF0) >= C_cols)
            break;

        __syncthreads();
        wmma::store_matrix_sync(sharedBuffer + threadTileRow * 16 * (SPLIT_K * 16 + 4) + threadTileCol * 16,
                                acc_frag[i],
                                SPLIT_K * 16 + 4,
                                wmma::mem_row_major);
        __syncthreads();

        int CRowStart = blockIdx.y * BLOCK_A_HEIGHT;
#pragma unroll 4
        for (int reduceRow = threadIdx.y * 2 + threadIdx.x / 16; reduceRow < BLOCK_A_HEIGHT; reduceRow += 16) {
            if (CRowStart + reduceRow >= C_rows)
                break;

            // Sum SPLIT_K instances of rows of 16 threads, then write the result to memory
            float f = 0.0f;
            for (int reduceCol = threadIdx.x % 16; reduceCol < SPLIT_K * 16; reduceCol += 16) {
                f += sharedBuffer[reduceRow * (SPLIT_K * 16 + 4) + reduceCol];
            }

            if ((CRowStart + reduceRow) < C_rows && CCol < C_cols)
                C[(CRowStart + reduceRow) * ld_C + CCol] = (half)f;
        }
    }
}

#undef ACC_HEIGHT
#undef SPLIT_K
#undef ACC_WIDTH
#undef ldb_shared
#undef BLOCK_A_HEIGHT

void launchTensorCoreMatrixMultiplyKernel_Arow16_Bcol48_restrictionABMult16x16_loadToReg(const half *A,
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
    dim3 gridSize((B_cols + 47) / 48, (A_rows + 15) / 16);
    tensorCoreMatrixMultiplyKernel_Arow16_Bcol48_restrictionABMult16x16_loadToReg<<<gridSize, blockSize, 0, stream.getStream()>>>(
        A, B, C, A_rows, A_cols, B_cols, ld_A, ld_B, ld_C);
}

void launchTensorCoreMatrixMultiplyKernel_Arow32_Bcol48_restrictionABMult16x16_loadToReg(const half *A,
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
    dim3 gridSize((B_cols + 47) / 48, (A_rows + 31) / 32);
    tensorCoreMatrixMultiplyKernel_Arow32_Bcol48_restrictionABMult16x16_loadToReg<<<gridSize, blockSize, 0, stream.getStream()>>>(
        A, B, C, A_rows, A_cols, B_cols, ld_A, ld_B, ld_C);
}

void launchTensorCoreMatrixMultiplyKernel_Arow48_Bcol48_restrictionABMult16x16_loadToReg(const half *A,
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
    dim3 gridSize((B_cols + 47) / 48, (A_rows + 47) / 48);
    tensorCoreMatrixMultiplyKernel_Arow48_Bcol48_restrictionABMult16x16_loadToReg<<<gridSize, blockSize, 0, stream.getStream()>>>(
        A, B, C, A_rows, A_cols, B_cols, ld_A, ld_B, ld_C);
}

void launchTensorCoreMatrixMultiplyKernel_Arow48_Bcol48_restrictionAMult16x16_loadToReg(const half *A,
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
    dim3 gridSize((B_cols + 47) / 48, (A_rows + 47) / 48);
    tensorCoreMatrixMultiplyKernel_Arow48_Bcol48_restrictionAMult16x16_loadToReg<<<gridSize, blockSize, 0, stream.getStream()>>>(
        A, B, C, A_rows, A_cols, B_cols, ld_A, ld_B, ld_C);
}

void launchTensorCoreMatrixMultiplyKernel_Arow48_Bcol48_noRestriction_loadToReg(const half *A,
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
    dim3 gridSize((B_cols + 47) / 48, (A_rows + 47) / 48);
    tensorCoreMatrixMultiplyKernel_Arow48_Bcol48_noRestriction_loadToReg<<<gridSize, blockSize, 0, stream.getStream()>>>(
        A, B, C, A_rows, A_cols, B_cols, ld_A, ld_B, ld_C);
}

void launchTensorCoreMatrixMultiplyKernel_Arow64_Bcol48_restrictionABMult16x16_loadToReg(const half *A,
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
    dim3 gridSize((B_cols + 47) / 48, (A_rows + 63) / 64);
    tensorCoreMatrixMultiplyKernel_Arow64_Bcol48_restrictionABMult16x16_loadToReg<<<gridSize, blockSize, 0, stream.getStream()>>>(
        A, B, C, A_rows, A_cols, B_cols, ld_A, ld_B, ld_C);
}

void launchTensorCoreMatrixMultiplyKernel_Arow256_Bcol48_restrictionAMult16x16(const half *A,
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
    dim3 gridSize((B_cols + 47) / 48, (A_rows + 255) / 256);
    tensorCoreMatrixMultiplyKernel_Arow256_Bcol48_restrictionAMult16x16<<<gridSize, blockSize, 0, stream.getStream()>>>(
        A, B, C, A_rows, A_cols, B_cols, ld_A, ld_B, ld_C);
}

void launchTensorCoreMatrixMultiplyKernel_Arow256_Bcol48_noRestriction(const half *A,
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
    dim3 gridSize((B_cols + 47) / 48, (A_rows + 255) / 256);
    tensorCoreMatrixMultiplyKernel_Arow256_Bcol48_noRestriction<<<gridSize, blockSize, 0, stream.getStream()>>>(
        A, B, C, A_rows, A_cols, B_cols, ld_A, ld_B, ld_C);
}

void launchTensorCoreMatrixMultiplyKernel_Arow128_Bcol48_restrictionAMult16x16(const half *A,
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
    dim3 gridSize((B_cols + 47) / 48, (A_rows + 127) / 128);
    tensorCoreMatrixMultiplyKernel_Arow128_Bcol48_restrictionAMult16x16<<<gridSize, blockSize, 0, stream.getStream()>>>(
        A, B, C, A_rows, A_cols, B_cols, ld_A, ld_B, ld_C);
}

void launchTensorCoreMatrixMultiplyKernel_Arow128_Bcol48_noRestriction(const half *A,
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
    dim3 gridSize((B_cols + 47) / 48, (A_rows + 127) / 128);
    tensorCoreMatrixMultiplyKernel_Arow128_Bcol48_noRestriction<<<gridSize, blockSize, 0, stream.getStream()>>>(
        A, B, C, A_rows, A_cols, B_cols, ld_A, ld_B, ld_C);
}

void launchTensorCoreMatrixMultiplyKernel_Arow64_Bcol48_restrictionAMult16x16(const half *A,
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
    dim3 gridSize((B_cols + 47) / 48, (A_rows + 63) / 64);
    tensorCoreMatrixMultiplyKernel_Arow64_Bcol48_restrictionAMult16x16<<<gridSize, blockSize, 0, stream.getStream()>>>(
        A, B, C, A_rows, A_cols, B_cols, ld_A, ld_B, ld_C);
}

void launchTensorCoreMatrixMultiplyKernel_Arow64_Bcol48_noRestriction(const half *A,
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
    dim3 gridSize((B_cols + 47) / 48, (A_rows + 63) / 64);
    tensorCoreMatrixMultiplyKernel_Arow64_Bcol48_noRestriction<<<gridSize, blockSize, 0, stream.getStream()>>>(
        A, B, C, A_rows, A_cols, B_cols, ld_A, ld_B, ld_C);
}

void launchTensorCoreMatrixMultiplyKernel_Arow32_Bcol48_restrictionAMult16x16(const half *A,
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
    dim3 gridSize((B_cols + 47) / 48, (A_rows + 31) / 32);
    tensorCoreMatrixMultiplyKernel_Arow32_Bcol48_restrictionAMult16x16<<<gridSize, blockSize, 0, stream.getStream()>>>(
        A, B, C, A_rows, A_cols, B_cols, ld_A, ld_B, ld_C);
}

void launchTensorCoreMatrixMultiplyKernel_Arow32_Bcol48_noRestriction(const half *A,
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
    dim3 gridSize((B_cols + 47) / 48, (A_rows + 31) / 32);
    tensorCoreMatrixMultiplyKernel_Arow32_Bcol48_noRestriction<<<gridSize, blockSize, 0, stream.getStream()>>>(
        A, B, C, A_rows, A_cols, B_cols, ld_A, ld_B, ld_C);
}

void launchTensorCoreMatrixMultiplyKernel_Arow16_Bcol48_restrictionAMult16x16(const half *A,
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
    dim3 gridSize((B_cols + 47) / 48, (A_rows + 15) / 16);
    tensorCoreMatrixMultiplyKernel_Arow16_Bcol48_restrictionAMult16x16<<<gridSize, blockSize, 0, stream.getStream()>>>(
        A, B, C, A_rows, A_cols, B_cols, ld_A, ld_B, ld_C);
}

void launchTensorCoreMatrixMultiplyKernel_Arow16_Bcol48_noRestriction(const half *A,
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
    dim3 gridSize((B_cols + 47) / 48, (A_rows + 15) / 16);
    tensorCoreMatrixMultiplyKernel_Arow16_Bcol48_noRestriction<<<gridSize, blockSize, 0, stream.getStream()>>>(
        A, B, C, A_rows, A_cols, B_cols, ld_A, ld_B, ld_C);
}

//---------------------------------------------------
//
// Algorithms with reduce
//
//---------------------------------------------------

void launchTensorCoreMatrixMultiplyKernel_Arow256_Bcol48_reduce2_restrictionAMult16x16(const half *A,
                                                                                       const half *B,
                                                                                       half *C,
                                                                                       half *workspace_d,
                                                                                       const int32_t A_rows,
                                                                                       const int32_t A_cols,
                                                                                       const int32_t B_cols,
                                                                                       const int32_t ld_A,
                                                                                       const int32_t ld_B,
                                                                                       const int32_t ld_C,
                                                                                       Stream stream) {
    // Perform the multiply
    dim3 blockSize(32, 8);
    dim3 gridSize((B_cols + 47) / 48, (A_rows + 255) / 256, 2);
    tensorCoreMatrixMultiplyKernel_Arow256_Bcol48_withReduce_restrictionAMult16x16<<<gridSize, blockSize, 0, stream.getStream()>>>(
        A, B, C, workspace_d, A_rows, A_cols, B_cols, ld_A, ld_B, ld_C);

    // Reduce into C
    launchReduce2(A_rows, B_cols, ld_C, C, workspace_d, stream);
}

void launchTensorCoreMatrixMultiplyKernel_Arow256_Bcol48_reduce4_restrictionAMult16x16(const half *A,
                                                                                       const half *B,
                                                                                       half *C,
                                                                                       half *workspace_d,
                                                                                       const int32_t A_rows,
                                                                                       const int32_t A_cols,
                                                                                       const int32_t B_cols,
                                                                                       const int32_t ld_A,
                                                                                       const int32_t ld_B,
                                                                                       const int32_t ld_C,
                                                                                       Stream stream) {
    // Perform the multiply
    dim3 blockSize(32, 8);
    dim3 gridSize((B_cols + 47) / 48, (A_rows + 255) / 256, 4);
    tensorCoreMatrixMultiplyKernel_Arow256_Bcol48_withReduce_restrictionAMult16x16<<<gridSize, blockSize, 0, stream.getStream()>>>(
        A, B, C, workspace_d, A_rows, A_cols, B_cols, ld_A, ld_B, ld_C);

    // Reduce into C
    launchReduce4(A_rows, B_cols, ld_C, C, workspace_d, stream);
}

void launchTensorCoreMatrixMultiplyKernel_Arow256_Bcol48_reduce8_restrictionAMult16x16(const half *A,
                                                                                       const half *B,
                                                                                       half *C,
                                                                                       half *workspace_d,
                                                                                       const int32_t A_rows,
                                                                                       const int32_t A_cols,
                                                                                       const int32_t B_cols,
                                                                                       const int32_t ld_A,
                                                                                       const int32_t ld_B,
                                                                                       const int32_t ld_C,
                                                                                       Stream stream) {
    // Perform the multiply
    dim3 blockSize(32, 8);
    dim3 gridSize((B_cols + 47) / 48, (A_rows + 255) / 256, 8);
    tensorCoreMatrixMultiplyKernel_Arow256_Bcol48_withReduce_restrictionAMult16x16<<<gridSize, blockSize, 0, stream.getStream()>>>(
        A, B, C, workspace_d, A_rows, A_cols, B_cols, ld_A, ld_B, ld_C);

    // Reduce into C
    launchReduce8(A_rows, B_cols, ld_C, C, workspace_d, stream);
}

void launchTensorCoreMatrixMultiplyKernel_Arow256_Bcol48_reduce6_restrictionAMult16x16(const half *A,
                                                                                       const half *B,
                                                                                       half *C,
                                                                                       half *workspace_d,
                                                                                       const int32_t A_rows,
                                                                                       const int32_t A_cols,
                                                                                       const int32_t B_cols,
                                                                                       const int32_t ld_A,
                                                                                       const int32_t ld_B,
                                                                                       const int32_t ld_C,
                                                                                       Stream stream) {
    // Perform the multiply
    dim3 blockSize(32, 8);
    dim3 gridSize((B_cols + 47) / 48, (A_rows + 255) / 256, 6);
    tensorCoreMatrixMultiplyKernel_Arow256_Bcol48_withReduce_restrictionAMult16x16<<<gridSize, blockSize, 0, stream.getStream()>>>(
        A, B, C, workspace_d, A_rows, A_cols, B_cols, ld_A, ld_B, ld_C);

    // Reduce into C
    launchReduce6(A_rows, B_cols, ld_C, C, workspace_d, stream);
}

void launchTensorCoreMatrixMultiplyKernel_Arow256_Bcol48_reduce2_noRestriction(const half *A,
                                                                               const half *B,
                                                                               half *C,
                                                                               half *workspace_d,
                                                                               const int32_t A_rows,
                                                                               const int32_t A_cols,
                                                                               const int32_t B_cols,
                                                                               const int32_t ld_A,
                                                                               const int32_t ld_B,
                                                                               const int32_t ld_C,
                                                                               Stream stream) {
    // Perform the multiply
    dim3 blockSize(32, 8);
    dim3 gridSize((B_cols + 47) / 48, (A_rows + 255) / 256, 2);
    tensorCoreMatrixMultiplyKernel_Arow256_Bcol48_withReduce_noRestriction<<<gridSize, blockSize, 0, stream.getStream()>>>(
        A, B, C, workspace_d, A_rows, A_cols, B_cols, ld_A, ld_B, ld_C);

    // Reduce into C
    launchReduce2(A_rows, B_cols, ld_C, C, workspace_d, stream);
}

void launchTensorCoreMatrixMultiplyKernel_Arow256_Bcol48_reduce4_noRestriction(const half *A,
                                                                               const half *B,
                                                                               half *C,
                                                                               half *workspace_d,
                                                                               const int32_t A_rows,
                                                                               const int32_t A_cols,
                                                                               const int32_t B_cols,
                                                                               const int32_t ld_A,
                                                                               const int32_t ld_B,
                                                                               const int32_t ld_C,
                                                                               Stream stream) {
    // Perform the multiply
    dim3 blockSize(32, 8);
    dim3 gridSize((B_cols + 47) / 48, (A_rows + 255) / 256, 4);
    tensorCoreMatrixMultiplyKernel_Arow256_Bcol48_withReduce_noRestriction<<<gridSize, blockSize, 0, stream.getStream()>>>(
        A, B, C, workspace_d, A_rows, A_cols, B_cols, ld_A, ld_B, ld_C);

    // Reduce into C
    launchReduce4(A_rows, B_cols, ld_C, C, workspace_d, stream);
}

void launchTensorCoreMatrixMultiplyKernel_Arow256_Bcol48_reduce8_noRestriction(const half *A,
                                                                               const half *B,
                                                                               half *C,
                                                                               half *workspace_d,
                                                                               const int32_t A_rows,
                                                                               const int32_t A_cols,
                                                                               const int32_t B_cols,
                                                                               const int32_t ld_A,
                                                                               const int32_t ld_B,
                                                                               const int32_t ld_C,
                                                                               Stream stream) {
    // Perform the multiply
    dim3 blockSize(32, 8);
    dim3 gridSize((B_cols + 47) / 48, (A_rows + 255) / 256, 8);
    tensorCoreMatrixMultiplyKernel_Arow256_Bcol48_withReduce_noRestriction<<<gridSize, blockSize, 0, stream.getStream()>>>(
        A, B, C, workspace_d, A_rows, A_cols, B_cols, ld_A, ld_B, ld_C);

    // Reduce into C
    launchReduce8(A_rows, B_cols, ld_C, C, workspace_d, stream);
}

void launchTensorCoreMatrixMultiplyKernel_Arow256_Bcol48_reduce6_noRestriction(const half *A,
                                                                               const half *B,
                                                                               half *C,
                                                                               half *workspace_d,
                                                                               const int32_t A_rows,
                                                                               const int32_t A_cols,
                                                                               const int32_t B_cols,
                                                                               const int32_t ld_A,
                                                                               const int32_t ld_B,
                                                                               const int32_t ld_C,
                                                                               Stream stream) {
    // Perform the multiply
    dim3 blockSize(32, 8);
    dim3 gridSize((B_cols + 47) / 48, (A_rows + 255) / 256, 6);
    tensorCoreMatrixMultiplyKernel_Arow256_Bcol48_withReduce_noRestriction<<<gridSize, blockSize, 0, stream.getStream()>>>(
        A, B, C, workspace_d, A_rows, A_cols, B_cols, ld_A, ld_B, ld_C);

    // Reduce into C
    launchReduce6(A_rows, B_cols, ld_C, C, workspace_d, stream);
}

void launchTensorCoreMatrixMultiplyKernel_Arow128_Bcol48_reduce2_restrictionAMult16x16(const half *A,
                                                                                       const half *B,
                                                                                       half *C,
                                                                                       half *workspace_d,
                                                                                       const int32_t A_rows,
                                                                                       const int32_t A_cols,
                                                                                       const int32_t B_cols,
                                                                                       const int32_t ld_A,
                                                                                       const int32_t ld_B,
                                                                                       const int32_t ld_C,
                                                                                       Stream stream) {
    // Perform the multiply
    dim3 blockSize(32, 8);
    dim3 gridSize((B_cols + 47) / 48, (A_rows + 127) / 128, 2);
    tensorCoreMatrixMultiplyKernel_Arow128_Bcol48_withReduce_restrictionAMult16x16<<<gridSize, blockSize, 0, stream.getStream()>>>(
        A, B, C, workspace_d, A_rows, A_cols, B_cols, ld_A, ld_B, ld_C);

    // Reduce into C
    launchReduce2(A_rows, B_cols, ld_C, C, workspace_d, stream);
}

void launchTensorCoreMatrixMultiplyKernel_Arow128_Bcol48_reduce4_restrictionAMult16x16(const half *A,
                                                                                       const half *B,
                                                                                       half *C,
                                                                                       half *workspace_d,
                                                                                       const int32_t A_rows,
                                                                                       const int32_t A_cols,
                                                                                       const int32_t B_cols,
                                                                                       const int32_t ld_A,
                                                                                       const int32_t ld_B,
                                                                                       const int32_t ld_C,
                                                                                       Stream stream) {
    // Perform the multiply
    dim3 blockSize(32, 8);
    dim3 gridSize((B_cols + 47) / 48, (A_rows + 127) / 128, 4);
    tensorCoreMatrixMultiplyKernel_Arow128_Bcol48_withReduce_restrictionAMult16x16<<<gridSize, blockSize, 0, stream.getStream()>>>(
        A, B, C, workspace_d, A_rows, A_cols, B_cols, ld_A, ld_B, ld_C);

    // Reduce into C
    launchReduce4(A_rows, B_cols, ld_C, C, workspace_d, stream);
}

void launchTensorCoreMatrixMultiplyKernel_Arow128_Bcol48_reduce8_restrictionAMult16x16(const half *A,
                                                                                       const half *B,
                                                                                       half *C,
                                                                                       half *workspace_d,
                                                                                       const int32_t A_rows,
                                                                                       const int32_t A_cols,
                                                                                       const int32_t B_cols,
                                                                                       const int32_t ld_A,
                                                                                       const int32_t ld_B,
                                                                                       const int32_t ld_C,
                                                                                       Stream stream) {
    // Perform the multiply
    dim3 blockSize(32, 8);
    dim3 gridSize((B_cols + 47) / 48, (A_rows + 127) / 128, 8);
    tensorCoreMatrixMultiplyKernel_Arow128_Bcol48_withReduce_restrictionAMult16x16<<<gridSize, blockSize, 0, stream.getStream()>>>(
        A, B, C, workspace_d, A_rows, A_cols, B_cols, ld_A, ld_B, ld_C);

    // Reduce into C
    launchReduce8(A_rows, B_cols, ld_C, C, workspace_d, stream);
}

void launchTensorCoreMatrixMultiplyKernel_Arow128_Bcol48_reduce6_restrictionAMult16x16(const half *A,
                                                                                       const half *B,
                                                                                       half *C,
                                                                                       half *workspace_d,
                                                                                       const int32_t A_rows,
                                                                                       const int32_t A_cols,
                                                                                       const int32_t B_cols,
                                                                                       const int32_t ld_A,
                                                                                       const int32_t ld_B,
                                                                                       const int32_t ld_C,
                                                                                       Stream stream) {
    // Perform the multiply
    dim3 blockSize(32, 8);
    dim3 gridSize((B_cols + 47) / 48, (A_rows + 127) / 128, 6);
    tensorCoreMatrixMultiplyKernel_Arow128_Bcol48_withReduce_restrictionAMult16x16<<<gridSize, blockSize, 0, stream.getStream()>>>(
        A, B, C, workspace_d, A_rows, A_cols, B_cols, ld_A, ld_B, ld_C);

    // Reduce into C
    launchReduce6(A_rows, B_cols, ld_C, C, workspace_d, stream);
}

void launchTensorCoreMatrixMultiplyKernel_Arow128_Bcol48_reduce2_noRestriction(const half *A,
                                                                               const half *B,
                                                                               half *C,
                                                                               half *workspace_d,
                                                                               const int32_t A_rows,
                                                                               const int32_t A_cols,
                                                                               const int32_t B_cols,
                                                                               const int32_t ld_A,
                                                                               const int32_t ld_B,
                                                                               const int32_t ld_C,
                                                                               Stream stream) {
    // Perform the multiply
    dim3 blockSize(32, 8);
    dim3 gridSize((B_cols + 47) / 48, (A_rows + 127) / 128, 2);
    tensorCoreMatrixMultiplyKernel_Arow128_Bcol48_withReduce_noRestriction<<<gridSize, blockSize, 0, stream.getStream()>>>(
        A, B, C, workspace_d, A_rows, A_cols, B_cols, ld_A, ld_B, ld_C);

    // Reduce into C
    launchReduce2(A_rows, B_cols, ld_C, C, workspace_d, stream);
}

void launchTensorCoreMatrixMultiplyKernel_Arow128_Bcol48_reduce4_noRestriction(const half *A,
                                                                               const half *B,
                                                                               half *C,
                                                                               half *workspace_d,
                                                                               const int32_t A_rows,
                                                                               const int32_t A_cols,
                                                                               const int32_t B_cols,
                                                                               const int32_t ld_A,
                                                                               const int32_t ld_B,
                                                                               const int32_t ld_C,
                                                                               Stream stream) {
    // Perform the multiply
    dim3 blockSize(32, 8);
    dim3 gridSize((B_cols + 47) / 48, (A_rows + 127) / 128, 4);
    tensorCoreMatrixMultiplyKernel_Arow128_Bcol48_withReduce_noRestriction<<<gridSize, blockSize, 0, stream.getStream()>>>(
        A, B, C, workspace_d, A_rows, A_cols, B_cols, ld_A, ld_B, ld_C);

    // Reduce into C
    launchReduce4(A_rows, B_cols, ld_C, C, workspace_d, stream);
}

void launchTensorCoreMatrixMultiplyKernel_Arow128_Bcol48_reduce8_noRestriction(const half *A,
                                                                               const half *B,
                                                                               half *C,
                                                                               half *workspace_d,
                                                                               const int32_t A_rows,
                                                                               const int32_t A_cols,
                                                                               const int32_t B_cols,
                                                                               const int32_t ld_A,
                                                                               const int32_t ld_B,
                                                                               const int32_t ld_C,
                                                                               Stream stream) {
    // Perform the multiply
    dim3 blockSize(32, 8);
    dim3 gridSize((B_cols + 47) / 48, (A_rows + 127) / 128, 8);
    tensorCoreMatrixMultiplyKernel_Arow128_Bcol48_withReduce_noRestriction<<<gridSize, blockSize, 0, stream.getStream()>>>(
        A, B, C, workspace_d, A_rows, A_cols, B_cols, ld_A, ld_B, ld_C);

    // Reduce into C
    launchReduce8(A_rows, B_cols, ld_C, C, workspace_d, stream);
}

void launchTensorCoreMatrixMultiplyKernel_Arow128_Bcol48_reduce6_noRestriction(const half *A,
                                                                               const half *B,
                                                                               half *C,
                                                                               half *workspace_d,
                                                                               const int32_t A_rows,
                                                                               const int32_t A_cols,
                                                                               const int32_t B_cols,
                                                                               const int32_t ld_A,
                                                                               const int32_t ld_B,
                                                                               const int32_t ld_C,
                                                                               Stream stream) {
    // Perform the multiply
    dim3 blockSize(32, 8);
    dim3 gridSize((B_cols + 47) / 48, (A_rows + 127) / 128, 6);
    tensorCoreMatrixMultiplyKernel_Arow128_Bcol48_withReduce_noRestriction<<<gridSize, blockSize, 0, stream.getStream()>>>(
        A, B, C, workspace_d, A_rows, A_cols, B_cols, ld_A, ld_B, ld_C);

    // Reduce into C
    launchReduce6(A_rows, B_cols, ld_C, C, workspace_d, stream);
}

vector<KernelWithSpec> TensorCoreMatrixMultiply::getBCol48Kernels() {
    vector<KernelWithSpec> kernels;
    KernelWithSpec kernel;

    kernel.bColsPerBlock = 48;
    kernel.aRowsPerBlock = 16;
    kernel.aRowSizeModulusRequirement = 16;
    kernel.aColSizeModulusRequirement = 16;
    kernel.bRowSizeModulusRequirement = 16;
    kernel.bColSizeModulusRequirement = 16;
    kernel.id = KernelWithSpec::KernelIndex::_16_48_AB16Restrict;
    kernel.executeKernel = launchTensorCoreMatrixMultiplyKernel_Arow16_Bcol48_restrictionABMult16x16_loadToReg;
    kernels.push_back(kernel);

    kernel.bColsPerBlock = 48;
    kernel.aRowsPerBlock = 32;
    kernel.aRowSizeModulusRequirement = 16;
    kernel.aColSizeModulusRequirement = 16;
    kernel.bRowSizeModulusRequirement = 16;
    kernel.bColSizeModulusRequirement = 16;
    kernel.id = KernelWithSpec::KernelIndex::_32_48_AB16Restrict;
    kernel.executeKernel = launchTensorCoreMatrixMultiplyKernel_Arow32_Bcol48_restrictionABMult16x16_loadToReg;
    kernels.push_back(kernel);

    kernel.bColsPerBlock = 48;
    kernel.aRowsPerBlock = 48;
    kernel.aRowSizeModulusRequirement = 16;
    kernel.aColSizeModulusRequirement = 16;
    kernel.bRowSizeModulusRequirement = 16;
    kernel.bColSizeModulusRequirement = 16;
    kernel.id = KernelWithSpec::KernelIndex::_48_48_AB16Restrict;
    kernel.executeKernel = launchTensorCoreMatrixMultiplyKernel_Arow48_Bcol48_restrictionABMult16x16_loadToReg;
    kernels.push_back(kernel);

    kernel.bColsPerBlock = 48;
    kernel.aRowsPerBlock = 48;
    kernel.aRowSizeModulusRequirement = 16;
    kernel.aColSizeModulusRequirement = 16;
    kernel.bRowSizeModulusRequirement = 0;
    kernel.bColSizeModulusRequirement = 0;
    kernel.id = KernelWithSpec::KernelIndex::_48_48_A16Restrict;
    kernel.executeKernel = launchTensorCoreMatrixMultiplyKernel_Arow48_Bcol48_restrictionAMult16x16_loadToReg;
    kernels.push_back(kernel);

    kernel.bColsPerBlock = 48;
    kernel.aRowsPerBlock = 48;
    kernel.aRowSizeModulusRequirement = 0;
    kernel.aColSizeModulusRequirement = 0;
    kernel.bRowSizeModulusRequirement = 0;
    kernel.bColSizeModulusRequirement = 0;
    kernel.id = KernelWithSpec::KernelIndex::_48_48_noRestrict;
    kernel.executeKernel = launchTensorCoreMatrixMultiplyKernel_Arow48_Bcol48_noRestriction_loadToReg;
    kernels.push_back(kernel);

    kernel.bColsPerBlock = 48;
    kernel.aRowsPerBlock = 64;
    kernel.aRowSizeModulusRequirement = 16;
    kernel.aColSizeModulusRequirement = 16;
    kernel.bRowSizeModulusRequirement = 16;
    kernel.bColSizeModulusRequirement = 16;
    kernel.id = KernelWithSpec::KernelIndex::_64_48_AB16Restrict;
    kernel.executeKernel = launchTensorCoreMatrixMultiplyKernel_Arow64_Bcol48_restrictionABMult16x16_loadToReg;
    kernels.push_back(kernel);

    kernel.bColsPerBlock = 48;
    kernel.aRowsPerBlock = 128;
    kernel.aRowSizeModulusRequirement = 16;
    kernel.aColSizeModulusRequirement = 16;
    kernel.bRowSizeModulusRequirement = 0;
    kernel.bColSizeModulusRequirement = 0;
    kernel.id = KernelWithSpec::KernelIndex::_128_48_A16Restrict;
    kernel.executeKernel = launchTensorCoreMatrixMultiplyKernel_Arow128_Bcol48_restrictionAMult16x16;
    kernels.push_back(kernel);

    kernel.bColsPerBlock = 48;
    kernel.aRowsPerBlock = 128;
    kernel.aRowSizeModulusRequirement = 0;
    kernel.aColSizeModulusRequirement = 0;
    kernel.bRowSizeModulusRequirement = 0;
    kernel.bColSizeModulusRequirement = 0;
    kernel.id = KernelWithSpec::KernelIndex::_128_48_noRestrict;
    kernel.executeKernel = launchTensorCoreMatrixMultiplyKernel_Arow128_Bcol48_noRestriction;
    kernels.push_back(kernel);

    kernel.bColsPerBlock = 48;
    kernel.aRowsPerBlock = 256;
    kernel.aRowSizeModulusRequirement = 16;
    kernel.aColSizeModulusRequirement = 16;
    kernel.bRowSizeModulusRequirement = 0;
    kernel.bColSizeModulusRequirement = 0;
    kernel.id = KernelWithSpec::KernelIndex::_256_48_A16Restrict;
    kernel.executeKernel = launchTensorCoreMatrixMultiplyKernel_Arow256_Bcol48_restrictionAMult16x16;
    kernels.push_back(kernel);

    kernel.bColsPerBlock = 48;
    kernel.aRowsPerBlock = 256;
    kernel.aRowSizeModulusRequirement = 0;
    kernel.aColSizeModulusRequirement = 0;
    kernel.bRowSizeModulusRequirement = 0;
    kernel.bColSizeModulusRequirement = 0;
    kernel.id = KernelWithSpec::KernelIndex::_256_48_noRestrict;
    kernel.executeKernel = launchTensorCoreMatrixMultiplyKernel_Arow256_Bcol48_noRestriction;
    kernels.push_back(kernel);

    kernel.bColsPerBlock = 48;
    kernel.aRowsPerBlock = 64;
    kernel.aRowSizeModulusRequirement = 16;
    kernel.aColSizeModulusRequirement = 16;
    kernel.bRowSizeModulusRequirement = 0;
    kernel.bColSizeModulusRequirement = 0;
    kernel.id = KernelWithSpec::KernelIndex::_64_48_A16Restrict;
    kernel.executeKernel = launchTensorCoreMatrixMultiplyKernel_Arow64_Bcol48_restrictionAMult16x16;
    kernels.push_back(kernel);

    kernel.bColsPerBlock = 48;
    kernel.aRowsPerBlock = 64;
    kernel.aRowSizeModulusRequirement = 0;
    kernel.aColSizeModulusRequirement = 0;
    kernel.bRowSizeModulusRequirement = 0;
    kernel.bColSizeModulusRequirement = 0;
    kernel.id = KernelWithSpec::KernelIndex::_64_48_noRestrict;
    kernel.executeKernel = launchTensorCoreMatrixMultiplyKernel_Arow64_Bcol48_noRestriction;
    kernels.push_back(kernel);

    kernel.bColsPerBlock = 48;
    kernel.aRowsPerBlock = 32;
    kernel.aRowSizeModulusRequirement = 16;
    kernel.aColSizeModulusRequirement = 16;
    kernel.bRowSizeModulusRequirement = 0;
    kernel.bColSizeModulusRequirement = 0;
    kernel.id = KernelWithSpec::KernelIndex::_32_48_A16Restrict;
    kernel.executeKernel = launchTensorCoreMatrixMultiplyKernel_Arow32_Bcol48_restrictionAMult16x16;
    kernels.push_back(kernel);

    kernel.bColsPerBlock = 48;
    kernel.aRowsPerBlock = 32;
    kernel.aRowSizeModulusRequirement = 0;
    kernel.aColSizeModulusRequirement = 0;
    kernel.bRowSizeModulusRequirement = 0;
    kernel.bColSizeModulusRequirement = 0;
    kernel.id = KernelWithSpec::KernelIndex::_32_48_noRestrict;
    kernel.executeKernel = launchTensorCoreMatrixMultiplyKernel_Arow32_Bcol48_noRestriction;
    kernels.push_back(kernel);

    kernel.bColsPerBlock = 48;
    kernel.aRowsPerBlock = 16;
    kernel.aRowSizeModulusRequirement = 16;
    kernel.aColSizeModulusRequirement = 16;
    kernel.bRowSizeModulusRequirement = 0;
    kernel.bColSizeModulusRequirement = 0;
    kernel.id = KernelWithSpec::KernelIndex::_16_48_A16Restrict;
    kernel.executeKernel = launchTensorCoreMatrixMultiplyKernel_Arow16_Bcol48_restrictionAMult16x16;
    kernels.push_back(kernel);

    kernel.bColsPerBlock = 48;
    kernel.aRowsPerBlock = 16;
    kernel.aRowSizeModulusRequirement = 0;
    kernel.aColSizeModulusRequirement = 0;
    kernel.bRowSizeModulusRequirement = 0;
    kernel.bColSizeModulusRequirement = 0;
    kernel.id = KernelWithSpec::KernelIndex::_16_48_noRestrict;
    kernel.executeKernel = launchTensorCoreMatrixMultiplyKernel_Arow16_Bcol48_noRestriction;
    kernels.push_back(kernel);

    //----------------------------------------
    // Kernels with reductions and workspaces
    //----------------------------------------

    kernel.aRowsPerBlock = 256;
    kernel.bColsPerBlock = 48;
    kernel.aRowSizeModulusRequirement = 16;
    kernel.aColSizeModulusRequirement = 16;
    kernel.bRowSizeModulusRequirement = 0;
    kernel.bColSizeModulusRequirement = 0;
    kernel.getWorkspaceSize = getReductionWorkspaceSize<half, 2>;
    kernel.id = KernelWithSpec::KernelIndex::_256_48_bigSharedBlockA16Restrict_reduce2;
    kernel.executeKernel = launchTensorCoreMatrixMultiplyKernel_Arow256_Bcol48_reduce2_restrictionAMult16x16;
    kernels.push_back(kernel);

    kernel.aRowsPerBlock = 256;
    kernel.bColsPerBlock = 48;
    kernel.aRowSizeModulusRequirement = 16;
    kernel.aColSizeModulusRequirement = 16;
    kernel.bRowSizeModulusRequirement = 0;
    kernel.bColSizeModulusRequirement = 0;
    kernel.getWorkspaceSize = getReductionWorkspaceSize<half, 4>;
    kernel.id = KernelWithSpec::KernelIndex::_256_48_bigSharedBlockA16Restrict_reduce4;
    kernel.executeKernel = launchTensorCoreMatrixMultiplyKernel_Arow256_Bcol48_reduce4_restrictionAMult16x16;
    kernels.push_back(kernel);

    kernel.aRowsPerBlock = 256;
    kernel.bColsPerBlock = 48;
    kernel.aRowSizeModulusRequirement = 16;
    kernel.aColSizeModulusRequirement = 16;
    kernel.bRowSizeModulusRequirement = 0;
    kernel.bColSizeModulusRequirement = 0;
    kernel.getWorkspaceSize = getReductionWorkspaceSize<half, 8>;
    kernel.id = KernelWithSpec::KernelIndex::_256_48_bigSharedBlockA16Restrict_reduce8;
    kernel.executeKernel = launchTensorCoreMatrixMultiplyKernel_Arow256_Bcol48_reduce8_restrictionAMult16x16;
    kernels.push_back(kernel);

    kernel.aRowsPerBlock = 256;
    kernel.bColsPerBlock = 48;
    kernel.aRowSizeModulusRequirement = 16;
    kernel.aColSizeModulusRequirement = 16;
    kernel.bRowSizeModulusRequirement = 0;
    kernel.bColSizeModulusRequirement = 0;
    kernel.getWorkspaceSize = getReductionWorkspaceSize<half, 6>;
    kernel.id = KernelWithSpec::KernelIndex::_256_48_bigSharedBlockA16Restrict_reduce6;
    kernel.executeKernel = launchTensorCoreMatrixMultiplyKernel_Arow256_Bcol48_reduce6_restrictionAMult16x16;
    kernels.push_back(kernel);

    kernel.aRowsPerBlock = 256;
    kernel.bColsPerBlock = 48;
    kernel.aRowSizeModulusRequirement = 0;
    kernel.aColSizeModulusRequirement = 0;
    kernel.bRowSizeModulusRequirement = 0;
    kernel.bColSizeModulusRequirement = 0;
    kernel.getWorkspaceSize = getReductionWorkspaceSize<half, 2>;
    kernel.id = KernelWithSpec::KernelIndex::_256_48_bigSharedBlock_reduce2;
    kernel.executeKernel = launchTensorCoreMatrixMultiplyKernel_Arow256_Bcol48_reduce2_noRestriction;
    kernels.push_back(kernel);

    kernel.aRowsPerBlock = 256;
    kernel.bColsPerBlock = 48;
    kernel.aRowSizeModulusRequirement = 0;
    kernel.aColSizeModulusRequirement = 0;
    kernel.bRowSizeModulusRequirement = 0;
    kernel.bColSizeModulusRequirement = 0;
    kernel.getWorkspaceSize = getReductionWorkspaceSize<half, 4>;
    kernel.id = KernelWithSpec::KernelIndex::_256_48_bigSharedBlock_reduce4;
    kernel.executeKernel = launchTensorCoreMatrixMultiplyKernel_Arow256_Bcol48_reduce4_noRestriction;
    kernels.push_back(kernel);

    kernel.aRowsPerBlock = 256;
    kernel.bColsPerBlock = 48;
    kernel.aRowSizeModulusRequirement = 0;
    kernel.aColSizeModulusRequirement = 0;
    kernel.bRowSizeModulusRequirement = 0;
    kernel.bColSizeModulusRequirement = 0;
    kernel.getWorkspaceSize = getReductionWorkspaceSize<half, 8>;
    kernel.id = KernelWithSpec::KernelIndex::_256_48_bigSharedBlock_reduce8;
    kernel.executeKernel = launchTensorCoreMatrixMultiplyKernel_Arow256_Bcol48_reduce8_noRestriction;
    kernels.push_back(kernel);

    kernel.aRowsPerBlock = 256;
    kernel.bColsPerBlock = 48;
    kernel.aRowSizeModulusRequirement = 0;
    kernel.aColSizeModulusRequirement = 0;
    kernel.bRowSizeModulusRequirement = 0;
    kernel.bColSizeModulusRequirement = 0;
    kernel.getWorkspaceSize = getReductionWorkspaceSize<half, 6>;
    kernel.id = KernelWithSpec::KernelIndex::_256_48_bigSharedBlock_reduce6;
    kernel.executeKernel = launchTensorCoreMatrixMultiplyKernel_Arow256_Bcol48_reduce6_noRestriction;
    kernels.push_back(kernel);

    kernel.aRowsPerBlock = 128;
    kernel.bColsPerBlock = 48;
    kernel.aRowSizeModulusRequirement = 16;
    kernel.aColSizeModulusRequirement = 16;
    kernel.bRowSizeModulusRequirement = 0;
    kernel.bColSizeModulusRequirement = 0;
    kernel.getWorkspaceSize = getReductionWorkspaceSize<half, 2>;
    kernel.id = KernelWithSpec::KernelIndex::_128_48_bigSharedBlockA16Restrict_reduce2;
    kernel.executeKernel = launchTensorCoreMatrixMultiplyKernel_Arow128_Bcol48_reduce2_restrictionAMult16x16;
    kernels.push_back(kernel);

    kernel.aRowsPerBlock = 128;
    kernel.bColsPerBlock = 48;
    kernel.aRowSizeModulusRequirement = 16;
    kernel.aColSizeModulusRequirement = 16;
    kernel.bRowSizeModulusRequirement = 0;
    kernel.bColSizeModulusRequirement = 0;
    kernel.getWorkspaceSize = getReductionWorkspaceSize<half, 4>;
    kernel.id = KernelWithSpec::KernelIndex::_128_48_bigSharedBlockA16Restrict_reduce4;
    kernel.executeKernel = launchTensorCoreMatrixMultiplyKernel_Arow128_Bcol48_reduce4_restrictionAMult16x16;
    kernels.push_back(kernel);

    kernel.aRowsPerBlock = 128;
    kernel.bColsPerBlock = 48;
    kernel.aRowSizeModulusRequirement = 16;
    kernel.aColSizeModulusRequirement = 16;
    kernel.bRowSizeModulusRequirement = 0;
    kernel.bColSizeModulusRequirement = 0;
    kernel.getWorkspaceSize = getReductionWorkspaceSize<half, 8>;
    kernel.id = KernelWithSpec::KernelIndex::_128_48_bigSharedBlockA16Restrict_reduce8;
    kernel.executeKernel = launchTensorCoreMatrixMultiplyKernel_Arow128_Bcol48_reduce8_restrictionAMult16x16;
    kernels.push_back(kernel);

    kernel.aRowsPerBlock = 128;
    kernel.bColsPerBlock = 48;
    kernel.aRowSizeModulusRequirement = 16;
    kernel.aColSizeModulusRequirement = 16;
    kernel.bRowSizeModulusRequirement = 0;
    kernel.bColSizeModulusRequirement = 0;
    kernel.getWorkspaceSize = getReductionWorkspaceSize<half, 6>;
    kernel.id = KernelWithSpec::KernelIndex::_128_48_bigSharedBlockA16Restrict_reduce6;
    kernel.executeKernel = launchTensorCoreMatrixMultiplyKernel_Arow128_Bcol48_reduce6_restrictionAMult16x16;
    kernels.push_back(kernel);

    kernel.aRowsPerBlock = 128;
    kernel.bColsPerBlock = 48;
    kernel.aRowSizeModulusRequirement = 0;
    kernel.aColSizeModulusRequirement = 0;
    kernel.bRowSizeModulusRequirement = 0;
    kernel.bColSizeModulusRequirement = 0;
    kernel.getWorkspaceSize = getReductionWorkspaceSize<half, 2>;
    kernel.id = KernelWithSpec::KernelIndex::_128_48_bigSharedBlock_reduce2;
    kernel.executeKernel = launchTensorCoreMatrixMultiplyKernel_Arow128_Bcol48_reduce2_noRestriction;
    kernels.push_back(kernel);

    kernel.aRowsPerBlock = 128;
    kernel.bColsPerBlock = 48;
    kernel.aRowSizeModulusRequirement = 0;
    kernel.aColSizeModulusRequirement = 0;
    kernel.bRowSizeModulusRequirement = 0;
    kernel.bColSizeModulusRequirement = 0;
    kernel.getWorkspaceSize = getReductionWorkspaceSize<half, 4>;
    kernel.id = KernelWithSpec::KernelIndex::_128_48_bigSharedBlock_reduce4;
    kernel.executeKernel = launchTensorCoreMatrixMultiplyKernel_Arow128_Bcol48_reduce4_noRestriction;
    kernels.push_back(kernel);

    kernel.aRowsPerBlock = 128;
    kernel.bColsPerBlock = 48;
    kernel.aRowSizeModulusRequirement = 0;
    kernel.aColSizeModulusRequirement = 0;
    kernel.bRowSizeModulusRequirement = 0;
    kernel.bColSizeModulusRequirement = 0;
    kernel.getWorkspaceSize = getReductionWorkspaceSize<half, 8>;
    kernel.id = KernelWithSpec::KernelIndex::_128_48_bigSharedBlock_reduce8;
    kernel.executeKernel = launchTensorCoreMatrixMultiplyKernel_Arow128_Bcol48_reduce8_noRestriction;
    kernels.push_back(kernel);

    kernel.aRowsPerBlock = 128;
    kernel.bColsPerBlock = 48;
    kernel.aRowSizeModulusRequirement = 0;
    kernel.aColSizeModulusRequirement = 0;
    kernel.bRowSizeModulusRequirement = 0;
    kernel.bColSizeModulusRequirement = 0;
    kernel.getWorkspaceSize = getReductionWorkspaceSize<half, 6>;
    kernel.id = KernelWithSpec::KernelIndex::_128_48_bigSharedBlock_reduce6;
    kernel.executeKernel = launchTensorCoreMatrixMultiplyKernel_Arow128_Bcol48_reduce6_noRestriction;
    kernels.push_back(kernel);

    return kernels;
}
