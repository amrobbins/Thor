#include "Thor.h"

#include <stdio.h>
#include "cuda.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "gtest/gtest.h"
#include "omp.h"

void printMatrixFp32(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%7.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

void printMatrixFp16(half *matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%7.2f ", __half2float(matrix[i * cols + j]));
        }
        printf("\n");
    }
}

void testMatrixTransposeFp32(int rows, int cols, bool print = false, bool verify = true) {
    float *matrix;
    float *matrix_gpu;
    float *matrix_d;
    float *transposedMatrix_d;
    cudaStream_t stream;
    cudaError_t cudaStatus;

    cudaStatus = cudaHostAlloc(&matrix, rows * cols * sizeof(float), cudaHostAllocPortable);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaHostAlloc(&matrix_gpu, rows * cols * sizeof(float), cudaHostAllocPortable);
    assert(cudaStatus == cudaSuccess);

    if (verify) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                matrix[i * cols + j] = i + 0.1 * j;
            }
        }
    }

    cudaStatus = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&matrix_d, rows * cols * sizeof(float));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&transposedMatrix_d, rows * cols * sizeof(float));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMemcpyAsync(matrix_d, matrix, rows * cols * sizeof(float), cudaMemcpyHostToDevice, stream);
    assert(cudaStatus == cudaSuccess);

    if (print) {
        printf("rows: %d cols %d\n", rows, cols);
        if (rows > 25 && rows < 40 && cols > 25 && cols < 40) {
            printMatrixFp32(matrix, rows, cols);
            printf("\n\n");
        }
    }

    matrixTranspose(transposedMatrix_d, matrix_d, rows, cols, stream);

    cudaStatus = cudaMemcpyAsync(matrix_gpu, transposedMatrix_d, rows * cols * sizeof(float), cudaMemcpyDeviceToHost, stream);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaStreamSynchronize(stream);
    if (print) {
        printf("cudaStatus %d\n", cudaStatus);
    }
    assert(cudaStatus == cudaSuccess);

    if (print) {
        if (rows > 25 && rows < 40 && cols > 25 && cols < 40)
            printMatrixFp32(matrix_gpu, cols, rows);
        fflush(stdout);
    }

    if (verify) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                assert(matrix[i * cols + j] == matrix_gpu[j * rows + i]);
            }
        }
    }

    cudaStatus = cudaFreeHost(matrix);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaFreeHost(matrix_gpu);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaFree(matrix_d);
    assert(cudaStatus == cudaSuccess);

    assert(cudaStatus == cudaSuccess);

    if (print) {
        printf("\n\nTest passed.\n");
    }
}

void testMatrixTransposeFp16(int rows, int cols, bool print = false, bool verify = true) {
    half *matrix;
    half *matrix_gpu;
    half *matrix_d;
    half *transposedMatrix_d;
    cudaStream_t stream;
    cudaError_t cudaStatus;

    cudaStatus = cudaHostAlloc(&matrix, rows * cols * sizeof(half), cudaHostAllocPortable);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaHostAlloc(&matrix_gpu, rows * cols * sizeof(half), cudaHostAllocPortable);
    assert(cudaStatus == cudaSuccess);

    if (verify) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                matrix[i * cols + j] = i + 0.1 * j;
            }
        }
    }

    cudaStatus = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    assert(cudaStatus == cudaSuccess);

    cudaStatus = cudaMalloc(&matrix_d, rows * cols * sizeof(half));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&transposedMatrix_d, rows * cols * sizeof(half));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMemcpyAsync(matrix_d, matrix, rows * cols * sizeof(half), cudaMemcpyHostToDevice, stream);
    assert(cudaStatus == cudaSuccess);

    if (print) {
        printf("rows: %d cols %d\n", rows, cols);
        if (rows < 100 && cols < 100) {
            printMatrixFp16(matrix, rows, cols);
            printf("\n\n");
        }
    }

    matrixTranspose(transposedMatrix_d, matrix_d, rows, cols, stream);

    cudaStatus = cudaMemcpyAsync(matrix_gpu, transposedMatrix_d, rows * cols * sizeof(half), cudaMemcpyDeviceToHost, stream);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaStreamSynchronize(stream);
    if (print) {
        printf("cudaStatus %d\n", cudaStatus);
    }
    assert(cudaStatus == cudaSuccess);

    if (print) {
        if (rows < 100 && cols < 100)
            printMatrixFp16(matrix_gpu, cols, rows);
        fflush(stdout);
    }

    if (verify) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                assert(__half2float(matrix[i * cols + j]) == __half2float(matrix_gpu[j * rows + i]));
            }
        }
    }

    cudaStatus = cudaFreeHost(matrix);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaFreeHost(matrix_gpu);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaFree(matrix_d);
    assert(cudaStatus == cudaSuccess);

    if (print) {
        printf("\n\nTest passed.\n");
    }
}

void testSquareMatrixTransposeFp32(int width, bool print = false, bool verify = true) {
    float *matrix;
    float *matrix_gpu;
    float *matrix_d;
    cudaStream_t stream;
    cudaError_t cudaStatus;

    cudaStatus = cudaHostAlloc(&matrix, width * width * sizeof(float), cudaHostAllocPortable);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaHostAlloc(&matrix_gpu, width * width * sizeof(float), cudaHostAllocPortable);
    assert(cudaStatus == cudaSuccess);

    if (verify) {
        for (int i = 0; i < width; ++i) {
            for (int j = 0; j < width; ++j) {
                matrix[i * width + j] = i + 0.1 * j;
            }
        }
    }

    cudaStatus = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&matrix_d, width * width * sizeof(float));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMemcpyAsync(matrix_d, matrix, width * width * sizeof(float), cudaMemcpyHostToDevice, stream);
    assert(cudaStatus == cudaSuccess);

    if (print) {
        printf("width: %d width %d\n", width, width);
        if (width > 25 && width < 40 && width > 25 && width < 40) {
            printMatrixFp32(matrix, width, width);
            printf("\n\n");
        }
    }

    matrixTransposeSquare(matrix_d, matrix_d, width, stream);

    cudaStatus = cudaMemcpyAsync(matrix_gpu, matrix_d, width * width * sizeof(float), cudaMemcpyDeviceToHost, stream);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaStreamSynchronize(stream);
    if (print) {
        printf("cudaStatus %d\n", cudaStatus);
    }
    assert(cudaStatus == cudaSuccess);

    if (print) {
        if (width > 25 && width < 40 && width > 25 && width < 40)
            printMatrixFp32(matrix_gpu, width, width);
        fflush(stdout);
    }

    if (verify) {
        for (int i = 0; i < width; ++i) {
            for (int j = 0; j < width; ++j) {
                assert(matrix[i * width + j] == matrix_gpu[j * width + i]);
            }
        }
    }

    cudaStatus = cudaFreeHost(matrix);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaFreeHost(matrix_gpu);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaFree(matrix_d);
    assert(cudaStatus == cudaSuccess);

    assert(cudaStatus == cudaSuccess);

    if (print) {
        printf("\n\nTest passed.\n");
    }
}

void testSquareMatrixTransposeFp16(int width, bool print = false, bool verify = true) {
    half *matrix;
    half *matrix_gpu;
    half *matrix_d;
    cudaStream_t stream;
    cudaError_t cudaStatus;

    cudaStatus = cudaHostAlloc(&matrix, width * width * sizeof(half), cudaHostAllocPortable);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaHostAlloc(&matrix_gpu, width * width * sizeof(half), cudaHostAllocPortable);
    assert(cudaStatus == cudaSuccess);

    if (verify) {
        for (int i = 0; i < width; ++i) {
            for (int j = 0; j < width; ++j) {
                matrix[i * width + j] = i + 0.1 * j;
            }
        }
    }

    cudaStatus = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    assert(cudaStatus == cudaSuccess);

    cudaStatus = cudaMalloc(&matrix_d, width * width * sizeof(half));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMemcpyAsync(matrix_d, matrix, width * width * sizeof(half), cudaMemcpyHostToDevice, stream);
    assert(cudaStatus == cudaSuccess);

    if (print) {
        printf("width: %d width %d\n", width, width);
        if (width < 100 && width < 100) {
            printMatrixFp16(matrix, width, width);
            printf("\n\n");
        }
    }

    matrixTransposeSquare(matrix_d, matrix_d, width, stream);

    cudaStatus = cudaMemcpyAsync(matrix_gpu, matrix_d, width * width * sizeof(half), cudaMemcpyDeviceToHost, stream);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaStreamSynchronize(stream);
    if (print) {
        printf("cudaStatus %d\n", cudaStatus);
    }
    assert(cudaStatus == cudaSuccess);

    if (print) {
        if (width < 100 && width < 100)
            printMatrixFp16(matrix_gpu, width, width);
        fflush(stdout);
    }

    if (verify) {
        for (int i = 0; i < width; ++i) {
            for (int j = 0; j < width; ++j) {
                assert(__half2float(matrix[i * width + j]) == __half2float(matrix_gpu[j * width + i]));
            }
        }
    }

    cudaStatus = cudaFreeHost(matrix);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaFreeHost(matrix_gpu);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaFree(matrix_d);
    assert(cudaStatus == cudaSuccess);

    if (print) {
        printf("\n\nTest passed.\n");
    }
}

TEST(MatrixTransposeFp32, OutputIsCorrect) {
    srand(time(nullptr));

    if (omp_get_num_procs() > 1)
        omp_set_num_threads(omp_get_num_procs() - 1);

#pragma omp parallel for schedule(dynamic, 10)
    for (int rows = 1; rows < 150; rows += (rand() % 5) + 1) {
        for (int cols = 1; cols < 150; cols += (rand() % 5) + 1) {
            testMatrixTransposeFp32(rows, cols);
        }
    }
    testMatrixTransposeFp32(5000, 1000);
    testMatrixTransposeFp32(2000, 4000);
}

TEST(MatrixTransposeFp16, OutputIsCorrect) {
    srand(time(nullptr));

    if (omp_get_num_procs() > 1)
        omp_set_num_threads(omp_get_num_procs() - 1);

#pragma omp parallel for schedule(dynamic, 10)
    for (int rows = 1; rows < 150; rows += (rand() % 5) + 1) {
        for (int cols = 1; cols < 150; cols += (rand() % 5) + 1) {
            testMatrixTransposeFp16(rows, cols);
        }
    }
    testMatrixTransposeFp16(5000, 1000);
    testMatrixTransposeFp16(2000, 4001);
}

TEST(MatrixTransposeSquareFp32, OutputIsCorrect) {
    srand(time(nullptr));

    if (omp_get_num_procs() > 1)
        omp_set_num_threads(omp_get_num_procs() - 1);

#pragma omp parallel for schedule(dynamic, 10)
    for (int width = 1; width < 150; width += (rand() % 5) + 1) {
        testSquareMatrixTransposeFp32(width);
    }
    testSquareMatrixTransposeFp32(2000);
}

TEST(MatrixTransposeSquareFp16, OutputIsCorrect) {
    srand(time(nullptr));

    if (omp_get_num_procs() > 1)
        omp_set_num_threads(omp_get_num_procs() - 1);

#pragma omp parallel for schedule(dynamic, 10)
    for (int width = 1; width < 150; width += (rand() % 5) + 1) {
        testSquareMatrixTransposeFp16(width);
    }
    testSquareMatrixTransposeFp16(3000);
    testSquareMatrixTransposeFp16(3001);
}

/*
TEST(MatrixTransposeFp32, Speed) {
    for(int i = 0; i < 200; i++) {
        testMatrixTransposeFp32(2048, 1024, false, false);
        testMatrixTransposeFp32(2048, 1023, false, false);
    }
}


TEST(MatrixTransposeFp16, Speed) {
    for(int i = 0; i < 200; i++) {
        testMatrixTransposeFp16(2048, 1024, false, false);
        testMatrixTransposeFp16(2048, 1023, false, false);
    }
}
*/

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
