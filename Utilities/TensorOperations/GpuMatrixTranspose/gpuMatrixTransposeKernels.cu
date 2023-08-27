
#include "gpuMatrixTranspose.h"

#define TILE_DIM 32
#define TWICE_TILE_DIM 64
#define QUAD_TILE_DIM 128
#define BLOCK_ROWS 8
#define TWICE_BLOCK_ROWS 16
#define QUAD_BLOCK_ROWS 32

__global__ void matrixTransposeKernel(float *transposedMatrix, const float *matrix, int memRows, int memCols) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = memCols;

    if (x >= memCols && y >= memRows)
        return;

    if (x < memCols) {
        // 32 line reads:
        for (int j = 0; threadIdx.y + j < TILE_DIM && y + j < memRows; j += BLOCK_ROWS) {
            // 32 contiguous element reads horizontal - one line:
            tile[threadIdx.y + j][threadIdx.x] = matrix[(y + j) * width + x];
        }
    }

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    int transposedWidth = memRows;

    __syncthreads();

    if (x >= memRows)
        return;

    for (int j = 0; j < TILE_DIM && y + j < memCols; j += BLOCK_ROWS) {
        transposedMatrix[(y + j) * transposedWidth + x] = tile[threadIdx.x][threadIdx.y + j];
    }
}

__global__ void matrixTransposeKernel(float2 *transposedMatrix, const float2 *matrix, int numRows, int numCols) {
    __shared__ float2 tile[2 * TILE_DIM][TILE_DIM + 1];

    int tileRow = threadIdx.y;
    int tileCol = threadIdx.x;
    int matrixCol = threadIdx.x + blockIdx.x * TILE_DIM;
    int matrixMemoryCols = (numCols + 1) / 2;
    int matrixRowStart = blockIdx.y * TWICE_TILE_DIM;
    int matrixRowStartNextBlock = matrixRowStart + TWICE_TILE_DIM;
    if (matrixCol < matrixMemoryCols) {
        for (int matrixRow = threadIdx.y + matrixRowStart;
             matrixRow < matrixRowStartNextBlock && matrixRow < numRows && tileRow < TWICE_TILE_DIM;
             matrixRow += BLOCK_ROWS, tileRow += BLOCK_ROWS) {
            tile[tileRow][tileCol] = matrix[matrixRow * matrixMemoryCols + matrixCol];
        }
    }

    int transposedMatrixRows = numCols;
    int transposedMatrixMemoryCols = (numRows + 1) / 2;
    matrixRowStart = blockIdx.x * TWICE_TILE_DIM;
    matrixRowStartNextBlock = matrixRowStart + TWICE_TILE_DIM;
    matrixCol = threadIdx.x + blockIdx.y * TILE_DIM;
    if (matrixCol >= transposedMatrixMemoryCols)
        return;
    tileRow = 2 * threadIdx.x;
    tileCol = threadIdx.y;

    __syncthreads();

    for (int matrixRow = matrixRowStart + 2 * threadIdx.y;
         matrixRow < matrixRowStartNextBlock && matrixRow < transposedMatrixRows && tileCol < TILE_DIM;
         matrixRow += TWICE_BLOCK_ROWS, tileCol += BLOCK_ROWS) {
        float2 buffer0 = tile[tileRow][tileCol];
        float2 buffer1 = tile[tileRow + 1][tileCol];
        float2 outputBuffer;

        outputBuffer.x = buffer0.x;
        outputBuffer.y = buffer1.x;
        transposedMatrix[matrixRow * transposedMatrixMemoryCols + matrixCol] = outputBuffer;

        outputBuffer.x = buffer0.y;
        outputBuffer.y = buffer1.y;
        transposedMatrix[(matrixRow + 1) * transposedMatrixMemoryCols + matrixCol] = outputBuffer;
    }
}

__global__ void matrixTransposeKernel(half *transposedMatrix, const half *matrix, int memRows, int memCols) {
    __shared__ half tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = memCols;

    if (x >= memCols && y >= memRows)
        return;

    if (x < memCols) {
        for (int j = 0; threadIdx.y + j < TILE_DIM && y + j < memRows; j += BLOCK_ROWS)
            tile[threadIdx.y + j][threadIdx.x] = matrix[(y + j) * width + x];
    }

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    int transposedWidth = memRows;

    __syncthreads();

    if (x >= memRows)
        return;

    for (int j = 0; j < TILE_DIM && y + j < memCols; j += BLOCK_ROWS) {
        transposedMatrix[(y + j) * transposedWidth + x] = tile[threadIdx.x][threadIdx.y + j];
    }
}

__global__ void matrixTransposeKernel(half2 *transposedMatrix, const half2 *matrix, int numRows, int numCols) {
    __shared__ half2 tile[2 * TILE_DIM][TILE_DIM + 1];

    int tileRow = threadIdx.y;
    int tileCol = threadIdx.x;
    int matrixCol = threadIdx.x + blockIdx.x * TILE_DIM;
    int matrixMemoryCols = (numCols + 1) / 2;
    int matrixRowStart = blockIdx.y * TWICE_TILE_DIM;
    int matrixRowStartNextBlock = matrixRowStart + TWICE_TILE_DIM;
    if (matrixCol < matrixMemoryCols) {
        for (int matrixRow = threadIdx.y + matrixRowStart;
             matrixRow < matrixRowStartNextBlock && matrixRow < numRows && tileRow < TWICE_TILE_DIM;
             matrixRow += BLOCK_ROWS, tileRow += BLOCK_ROWS) {
            tile[tileRow][tileCol] = matrix[matrixRow * matrixMemoryCols + matrixCol];
        }
    }

    int transposedMatrixRows = numCols;
    int transposedMatrixMemoryCols = (numRows + 1) / 2;
    matrixRowStart = blockIdx.x * TWICE_TILE_DIM;
    matrixRowStartNextBlock = matrixRowStart + TWICE_TILE_DIM;
    matrixCol = threadIdx.x + blockIdx.y * TILE_DIM;
    if (matrixCol >= transposedMatrixMemoryCols)
        return;
    tileRow = 2 * threadIdx.x;
    tileCol = threadIdx.y;

    __syncthreads();

    for (int matrixRow = matrixRowStart + 2 * threadIdx.y;
         matrixRow < matrixRowStartNextBlock && matrixRow < transposedMatrixRows && tileCol < TILE_DIM;
         matrixRow += TWICE_BLOCK_ROWS, tileCol += BLOCK_ROWS) {
        half2 buffer0 = tile[tileRow][tileCol];
        half2 buffer1 = tile[tileRow + 1][tileCol];
        half2 outputBuffer;

        outputBuffer.x = buffer0.x;
        outputBuffer.y = buffer1.x;
        transposedMatrix[matrixRow * transposedMatrixMemoryCols + matrixCol] = outputBuffer;

        outputBuffer.x = buffer0.y;
        outputBuffer.y = buffer1.y;
        transposedMatrix[(matrixRow + 1) * transposedMatrixMemoryCols + matrixCol] = outputBuffer;
    }
}

// This one does not seem faster
__global__ void matrixTransposeKernel(short4 *transposedMatrix, const short4 *matrix, int numRows, int numCols) {
    __shared__ short4 tile[4 * TILE_DIM][TILE_DIM + 1];

    int tileRow = threadIdx.y;
    int tileCol = threadIdx.x;
    int matrixCol = threadIdx.x + blockIdx.x * TILE_DIM;
    int matrixMemoryCols = (numCols + 3) / 4;
    int matrixRowStart = blockIdx.y * QUAD_TILE_DIM;
    int matrixRowStartNextBlock = matrixRowStart + QUAD_TILE_DIM;
    if (matrixCol < matrixMemoryCols) {
        for (int matrixRow = threadIdx.y + matrixRowStart;
             matrixRow < matrixRowStartNextBlock && matrixRow < numRows && tileRow < QUAD_TILE_DIM;
             matrixRow += BLOCK_ROWS, tileRow += BLOCK_ROWS) {
            tile[tileRow][tileCol] = matrix[matrixRow * matrixMemoryCols + matrixCol];
        }
    }

    int transposedMatrixRows = numCols;
    int transposedMatrixMemoryCols = (numRows + 3) / 4;
    matrixRowStart = blockIdx.x * QUAD_TILE_DIM;
    matrixRowStartNextBlock = matrixRowStart + QUAD_TILE_DIM;
    matrixCol = threadIdx.x + blockIdx.y * TILE_DIM;
    if (matrixCol >= transposedMatrixMemoryCols)
        return;
    tileRow = 4 * threadIdx.x;
    tileCol = threadIdx.y;

    __syncthreads();

    for (int matrixRow = matrixRowStart + 4 * threadIdx.y;
         matrixRow < matrixRowStartNextBlock && matrixRow < transposedMatrixRows && tileCol < TILE_DIM;
         matrixRow += QUAD_BLOCK_ROWS, tileCol += BLOCK_ROWS) {
        short4 buffer0 = tile[tileRow][tileCol];
        short4 buffer1 = tile[tileRow + 1][tileCol];
        short4 buffer2 = tile[tileRow + 2][tileCol];
        short4 buffer3 = tile[tileRow + 3][tileCol];
        short4 outputBuffer;

        outputBuffer.x = buffer0.x;
        outputBuffer.y = buffer1.x;
        outputBuffer.z = buffer2.x;
        outputBuffer.w = buffer3.x;
        transposedMatrix[matrixRow * transposedMatrixMemoryCols + matrixCol] = outputBuffer;

        outputBuffer.x = buffer0.y;
        outputBuffer.y = buffer1.y;
        outputBuffer.z = buffer2.y;
        outputBuffer.w = buffer3.y;
        transposedMatrix[(matrixRow + 1) * transposedMatrixMemoryCols + matrixCol] = outputBuffer;

        outputBuffer.x = buffer0.z;
        outputBuffer.y = buffer1.z;
        outputBuffer.z = buffer2.z;
        outputBuffer.w = buffer3.z;
        transposedMatrix[(matrixRow + 2) * transposedMatrixMemoryCols + matrixCol] = outputBuffer;

        outputBuffer.x = buffer0.w;
        outputBuffer.y = buffer1.w;
        outputBuffer.z = buffer2.w;
        outputBuffer.w = buffer3.w;
        transposedMatrix[(matrixRow + 3) * transposedMatrixMemoryCols + matrixCol] = outputBuffer;
    }
}

// Allows in-place transpose
__global__ void matrixTransposeSquareKernel(float *transposedMatrix, const float *matrix, int width) {
    __shared__ float tile1[TILE_DIM][TILE_DIM + 1];
    __shared__ float tile2[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    if (x >= width && y >= width)
        return;

    if (x < width) {
        // 32 line reads - 4 per warp:
        for (int j = 0; threadIdx.y + j < TILE_DIM && y + j < width; j += BLOCK_ROWS) {
            // 32 contiguous element reads horizontal - one line:
            if (x > y + j) {
                tile1[threadIdx.y + j][threadIdx.x] = matrix[(y + j) * width + x];
            }
        }
    }

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    if (x < width) {
        for (int j = 0; j < TILE_DIM && y + j < width; j += BLOCK_ROWS) {
            if (x < y + j) {
                tile2[threadIdx.x][threadIdx.y + j] = matrix[(y + j) * width + x];
            }
        }
    }

    __syncthreads();

    x = blockIdx.x * TILE_DIM + threadIdx.x;
    y = blockIdx.y * TILE_DIM + threadIdx.y;

    if (x < width) {
        // 32 line writes:
        for (int j = 0; threadIdx.y + j < TILE_DIM && y + j < width; j += BLOCK_ROWS) {
            // 32 contiguous element writes horizontal - one line:
            if (x > y + j) {
                transposedMatrix[(y + j) * width + x] = tile2[threadIdx.y + j][threadIdx.x];
            }
        }
    }

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    if (x < width) {
        for (int j = 0; j < TILE_DIM && y + j < width; j += BLOCK_ROWS) {
            if (x < y + j) {
                transposedMatrix[(y + j) * width + x] = tile1[threadIdx.x][threadIdx.y + j];
            }
        }
    }
}

// Allows in-place transpose
__global__ void matrixTransposeSquareKernel(half *transposedMatrix, const half *matrix, int width) {
    __shared__ half tile1[TILE_DIM][TILE_DIM + 1];
    __shared__ half tile2[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    if (x >= width && y >= width)
        return;

    if (x < width) {
        // 32 line reads - 4 per warp:
        for (int j = 0; threadIdx.y + j < TILE_DIM && y + j < width; j += BLOCK_ROWS) {
            // 32 contiguous element reads horizontal - one line:
            if (x > y + j) {
                tile1[threadIdx.y + j][threadIdx.x] = matrix[(y + j) * width + x];
            }
        }
    }

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    if (x < width) {
        for (int j = 0; j < TILE_DIM && y + j < width; j += BLOCK_ROWS) {
            if (x < y + j) {
                tile2[threadIdx.x][threadIdx.y + j] = matrix[(y + j) * width + x];
            }
        }
    }

    __syncthreads();

    x = blockIdx.x * TILE_DIM + threadIdx.x;
    y = blockIdx.y * TILE_DIM + threadIdx.y;

    if (x < width) {
        // 32 line writes:
        for (int j = 0; threadIdx.y + j < TILE_DIM && y + j < width; j += BLOCK_ROWS) {
            // 32 contiguous element writes horizontal - one line:
            if (x > y + j) {
                transposedMatrix[(y + j) * width + x] = tile2[threadIdx.y + j][threadIdx.x];
            }
        }
    }

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    if (x < width) {
        for (int j = 0; j < TILE_DIM && y + j < width; j += BLOCK_ROWS) {
            if (x < y + j) {
                transposedMatrix[(y + j) * width + x] = tile1[threadIdx.x][threadIdx.y + j];
            }
        }
    }
}

__global__ void matrixTransposeSquareKernel(half2 *transposedMatrix, const half2 *matrix, int width) {
    __shared__ half2 tile1[2 * TILE_DIM][TILE_DIM + 1];
    __shared__ half2 tile2[2 * TILE_DIM][TILE_DIM + 1];

    int tileRow = threadIdx.y;
    int tileCol = threadIdx.x;
    int matrixCol = threadIdx.x + blockIdx.x * TILE_DIM;
    int matrixMemoryCols = (width + 1) / 2;
    int matrixRowStart = blockIdx.y * TWICE_TILE_DIM;
    int matrixRowStartNextBlock = matrixRowStart + TWICE_TILE_DIM;
    if (matrixCol < matrixMemoryCols) {
        for (int matrixRow = threadIdx.y + matrixRowStart;
             matrixRow < matrixRowStartNextBlock && matrixRow < width && tileRow < TWICE_TILE_DIM;
             matrixRow += BLOCK_ROWS, tileRow += BLOCK_ROWS) {
            // >= because if on the diagonal and reading 2 elements, we need the next element
            if (matrixCol * 2 + 1 >= matrixRow) {
                tile1[tileRow][tileCol] = matrix[matrixRow * matrixMemoryCols + matrixCol];
            }
        }
    }

    int transposedMatrixRows = width;
    int transposedMatrixMemoryCols = (width + 1) / 2;
    matrixRowStart = blockIdx.x * TWICE_TILE_DIM;
    matrixRowStartNextBlock = matrixRowStart + TWICE_TILE_DIM;
    matrixCol = threadIdx.x + blockIdx.y * TILE_DIM;
    tileRow = 2 * threadIdx.x;
    tileCol = threadIdx.y;
    for (int matrixRow = matrixRowStart + 2 * threadIdx.y;
         matrixRow < matrixRowStartNextBlock && matrixRow < transposedMatrixRows && tileCol < TILE_DIM;
         matrixRow += TWICE_BLOCK_ROWS, tileCol += BLOCK_ROWS) {
        half2 buffer0;
        half2 buffer1;
        half2 inputBuffer;

        if (matrixCol * 2 <= matrixRow) {
            inputBuffer = matrix[matrixRow * transposedMatrixMemoryCols + matrixCol];
            buffer0.x = inputBuffer.x;
            buffer1.x = inputBuffer.y;
        }

        if (matrixCol * 2 <= matrixRow + 1) {
            inputBuffer = matrix[(matrixRow + 1) * transposedMatrixMemoryCols + matrixCol];
            buffer0.y = inputBuffer.x;
            buffer1.y = inputBuffer.y;
        }

        tile2[tileRow][tileCol] = buffer0;
        tile2[tileRow + 1][tileCol] = buffer1;
    }

    __syncthreads();

    tileRow = threadIdx.y;
    tileCol = threadIdx.x;
    matrixCol = threadIdx.x + blockIdx.x * TILE_DIM;
    matrixMemoryCols = (width + 1) / 2;
    matrixRowStart = blockIdx.y * TWICE_TILE_DIM;
    matrixRowStartNextBlock = matrixRowStart + TWICE_TILE_DIM;
    if (matrixCol < matrixMemoryCols) {
        for (int matrixRow = threadIdx.y + matrixRowStart;
             matrixRow < matrixRowStartNextBlock && matrixRow < width && tileRow < TWICE_TILE_DIM;
             matrixRow += BLOCK_ROWS, tileRow += BLOCK_ROWS) {
            // >= because if on the diagonal and writing 2 elements, we need to write the next element
            if (matrixCol * 2 >= matrixRow) {
                transposedMatrix[matrixRow * matrixMemoryCols + matrixCol] = tile2[tileRow][tileCol];
            }
        }
    }

    transposedMatrixRows = width;
    transposedMatrixMemoryCols = (width + 1) / 2;
    matrixRowStart = blockIdx.x * TWICE_TILE_DIM;
    matrixRowStartNextBlock = matrixRowStart + TWICE_TILE_DIM;
    matrixCol = threadIdx.x + blockIdx.y * TILE_DIM;
    tileRow = 2 * threadIdx.x;
    tileCol = threadIdx.y;
    for (int matrixRow = matrixRowStart + 2 * threadIdx.y;
         matrixRow < matrixRowStartNextBlock && matrixRow < transposedMatrixRows && tileCol < TILE_DIM;
         matrixRow += TWICE_BLOCK_ROWS, tileCol += BLOCK_ROWS) {
        half2 buffer0 = tile1[tileRow][tileCol];
        half2 buffer1 = tile1[tileRow + 1][tileCol];
        half2 outputBuffer;

        // Note: a col is 2 halfs, a row is one element
        if (matrixCol * 2 < matrixRow) {
            outputBuffer.x = buffer0.x;
            outputBuffer.y = buffer1.x;
            transposedMatrix[matrixRow * transposedMatrixMemoryCols + matrixCol] = outputBuffer;
        }

        if (matrixCol * 2 < matrixRow + 1) {
            outputBuffer.x = buffer0.y;
            outputBuffer.y = buffer1.y;
            transposedMatrix[(matrixRow + 1) * transposedMatrixMemoryCols + matrixCol] = outputBuffer;
        }
    }
}

void matrixTranspose(half *transposedMatrix_d, const half *matrix_d, int numRows, int numCols, cudaStream_t stream) {
    if (numRows % 2 == 0 && numCols % 2 == 0) {
        dim3 blockSize = dim3(TILE_DIM, BLOCK_ROWS);
        dim3 gridSize = dim3((numCols + (TWICE_TILE_DIM - 1)) / (TWICE_TILE_DIM), (numRows + (TWICE_TILE_DIM - 1)) / TWICE_TILE_DIM);
        matrixTransposeKernel<<<gridSize, blockSize, 0, stream>>>((half2 *)transposedMatrix_d, (half2 *)matrix_d, numRows, numCols);
    } else {
        dim3 blockSize = dim3(TILE_DIM, BLOCK_ROWS);
        dim3 gridSize = dim3((numCols + (TILE_DIM - 1)) / TILE_DIM, (numRows + (TILE_DIM - 1)) / TILE_DIM);
        matrixTransposeKernel<<<gridSize, blockSize, 0, stream>>>(transposedMatrix_d, matrix_d, numRows, numCols);
    }
}

void matrixTranspose(float *transposedMatrix_d, const float *matrix_d, int numRows, int numCols, cudaStream_t stream) {
    if (numRows % 2 == 0 && numCols % 2 == 0) {
        dim3 blockSize = dim3(TILE_DIM, BLOCK_ROWS);
        dim3 gridSize = dim3((numCols + (TWICE_TILE_DIM - 1)) / (TWICE_TILE_DIM), (numRows + (TWICE_TILE_DIM - 1)) / TWICE_TILE_DIM);
        matrixTransposeKernel<<<gridSize, blockSize, 0, stream>>>((float2 *)transposedMatrix_d, (float2 *)matrix_d, numRows, numCols);
    } else {
        dim3 blockSize = dim3(TILE_DIM, BLOCK_ROWS);
        dim3 gridSize = dim3((numCols + (TILE_DIM - 1)) / TILE_DIM, (numRows + (TILE_DIM - 1)) / TILE_DIM);
        matrixTransposeKernel<<<gridSize, blockSize, 0, stream>>>(transposedMatrix_d, matrix_d, numRows, numCols);
    }
}

void matrixTransposeSquare(half *transposedMatrix_d, const half *matrix_d, int width, cudaStream_t stream) {
    if (width % 2 == 0) {
        dim3 blockSize = dim3(TILE_DIM, BLOCK_ROWS);
        dim3 gridSize = dim3((width + (TWICE_TILE_DIM - 1)) / (TWICE_TILE_DIM), (width + (TWICE_TILE_DIM - 1)) / TWICE_TILE_DIM);
        matrixTransposeSquareKernel<<<gridSize, blockSize, 0, stream>>>((half2 *)transposedMatrix_d, (half2 *)matrix_d, width);
    } else {
        dim3 blockSize = dim3(TILE_DIM, BLOCK_ROWS);
        dim3 gridSize = dim3((width + (TILE_DIM - 1)) / TILE_DIM, (width + (TILE_DIM - 1)) / TILE_DIM);
        matrixTransposeSquareKernel<<<gridSize, blockSize, 0, stream>>>(transposedMatrix_d, matrix_d, width);
    }
}

void matrixTransposeSquare(float *transposedMatrix_d, const float *matrix_d, int width, cudaStream_t stream) {
    dim3 blockSize = dim3(TILE_DIM, BLOCK_ROWS);
    dim3 gridSize = dim3((width + (TILE_DIM - 1)) / TILE_DIM, (width + (TILE_DIM - 1)) / TILE_DIM);
    matrixTransposeSquareKernel<<<gridSize, blockSize, 0, stream>>>(transposedMatrix_d, matrix_d, width);
}
