
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
        for (int j = 0; threadIdx.y + j < TILE_DIM && y + j < memRows; j += BLOCK_ROWS) {
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
