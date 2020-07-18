#include "TensorCoreMatrixMultiply.h"

// Stores the element-wise sum of M0 and M1 to M0
__global__ void reduce2(half *M0, half *M1, int rows, int cols, int ld) {
    int col = blockIdx.x * 256 + threadIdx.x;
    int row = blockIdx.y * 8;

    if (col >= cols)
        return;
    if (row >= rows)
        return;

#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
        // read and sum
        half buff = M0[row * ld + col];
        float accum = (float)buff;
        buff = M1[row * ld + col];
        accum += (float)buff;

        // write back
        M0[row * ld + col] = (half)accum;

        row += 1;
        if (row >= rows)
            return;
    }
}

// Sum is stored to M0
__global__ void reduce4(half *M0, half *M1, half *M2, half *M3, int rows, int cols, int ld) {
    int col = blockIdx.x * 256 + threadIdx.x;
    int row = blockIdx.y * 4;

    if (col >= cols)
        return;
    if (row >= rows)
        return;

#pragma unroll 4
    for (int i = 0; i < 4; ++i) {
        // read and sum
        half buff = M0[row * ld + col];
        float accum = (float)buff;
        buff = M1[row * ld + col];
        accum += (float)buff;
        buff = M2[row * ld + col];
        accum += (float)buff;
        buff = M3[row * ld + col];
        accum += (float)buff;

        // write back
        M0[row * ld + col] = (half)accum;

        row += 1;
        if (row >= rows)
            return;
    }
}

// Sum is stored to M0
__global__ void reduce8(half *M0, half *M1, half *M2, half *M3, half *M4, half *M5, half *M6, half *M7, int rows, int cols, int ld) {
    int col = blockIdx.x * 256 + threadIdx.x;
    int row = blockIdx.y * 4;

    if (col >= cols)
        return;
    if (row >= rows)
        return;

#pragma unroll 4
    for (int i = 0; i < 4; ++i) {
        // read and sum
        half buff = M0[row * ld + col];
        float accum = (float)buff;
        buff = M1[row * ld + col];
        accum += (float)buff;
        buff = M2[row * ld + col];
        accum += (float)buff;
        buff = M3[row * ld + col];
        accum += (float)buff;
        buff = M4[row * ld + col];
        accum += (float)buff;
        buff = M5[row * ld + col];
        accum += (float)buff;
        buff = M6[row * ld + col];
        accum += (float)buff;
        buff = M7[row * ld + col];
        accum += (float)buff;

        // write back
        M0[row * ld + col] = (half)accum;

        row += 1;
        if (row >= rows)
            return;
    }
}

// Sum is stored to M0
__global__ void reduce6(half *M0, half *M1, half *M2, half *M3, half *M4, half *M5, int rows, int cols, int ld) {
    int col = blockIdx.x * 256 + threadIdx.x;
    int row = blockIdx.y * 4;

    if (col >= cols)
        return;
    if (row >= rows)
        return;

#pragma unroll 4
    for (int i = 0; i < 4; ++i) {
        // read and sum
        half buff = M0[row * ld + col];
        float accum = (float)buff;
        buff = M1[row * ld + col];
        accum += (float)buff;
        buff = M2[row * ld + col];
        accum += (float)buff;
        buff = M3[row * ld + col];
        accum += (float)buff;
        buff = M4[row * ld + col];
        accum += (float)buff;
        buff = M5[row * ld + col];
        accum += (float)buff;

        // write back
        M0[row * ld + col] = (half)accum;

        row += 1;
        if (row >= rows)
            return;
    }
}

void launchReduce2(int rows, int cols, int ld, half *C, half *workspace, Stream stream) {
    dim3 blockSize(256);
    dim3 gridSize((cols + 255) / 256, (rows + 7) / 8);
    reduce2<<<gridSize, blockSize, 0, stream.getStream()>>>(C, workspace, rows, cols, ld);
}

void launchReduce4(int rows, int cols, int ld, half *C, half *workspace, Stream stream) {
    dim3 blockSize(256);
    dim3 gridSize((cols + 255) / 256, (rows + 3) / 4);
    long workspaceIncrement = rows * ld;
    reduce4<<<gridSize, blockSize, 0, stream.getStream()>>>(
        C, workspace, workspace + workspaceIncrement, workspace + 2 * workspaceIncrement, rows, cols, ld);
}

void launchReduce6(int rows, int cols, int ld, half *C, half *workspace, Stream stream) {
    dim3 blockSize(256);
    dim3 gridSize((cols + 255) / 256, (rows + 3) / 4);
    long workspaceIncrement = rows * ld;
    reduce6<<<gridSize, blockSize, 0, stream.getStream()>>>(C,
                                                            workspace,
                                                            workspace + workspaceIncrement,
                                                            workspace + 2 * workspaceIncrement,
                                                            workspace + 3 * workspaceIncrement,
                                                            workspace + 4 * workspaceIncrement,
                                                            rows,
                                                            cols,
                                                            ld);
}

void launchReduce8(int rows, int cols, int ld, half *C, half *workspace, Stream stream) {
    dim3 blockSize(256);
    dim3 gridSize((cols + 255) / 256, (rows + 3) / 4);
    long workspaceIncrement = rows * ld;
    reduce8<<<gridSize, blockSize, 0, stream.getStream()>>>(C,
                                                            workspace,
                                                            workspace + workspaceIncrement,
                                                            workspace + 2 * workspaceIncrement,
                                                            workspace + 3 * workspaceIncrement,
                                                            workspace + 4 * workspaceIncrement,
                                                            workspace + 5 * workspaceIncrement,
                                                            workspace + 6 * workspaceIncrement,
                                                            rows,
                                                            cols,
                                                            ld);
}
