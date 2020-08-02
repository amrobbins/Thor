#include "MLDev.h"

#include "gtest/gtest.h"
#include "omp.h"

using std::string;

#define INTEREST_KERNEL KernelWithSpec::KernelIndex::_256_96_bigSharedBlockA16Restrict

// computes result = AB, where A is an mxk matrix and B is an kxn matrix. This makes result a mxn matrix.
void matrixMultiplyCpu(float *A, float *B, float *C, int m, int n, int k, int lda, int ldb, int ldc) {
    for (int ra = 0; ra < m; ra++) {
        for (int cb = 0; cb < n; cb++) {
            float accum = 0.0;
            for (int carb = 0; carb < k; carb++) {
                accum += A[ra * lda + carb] * B[carb * ldb + cb];
            }
            C[ra * ldc + cb] = accum;
        }
    }
}

void printMatrix(float *matrix, int rows, int cols, int ld) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%5.2f ", matrix[i * ld + j]);
        }
        printf("\n");
    }
}

void printMatrixWide(float *matrix, int rows, int cols, int ld) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%7.2f ", matrix[i * ld + j]);
        }
        printf("\n");
    }
}

void printMatrix(half *matrix, int rows, int cols, int ld) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%5.2f ", (float)matrix[i * ld + j]);
        }
        printf("\n");
    }
}

void printMatrixWide(half *matrix, int rows, int cols, int ld) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%7.2f ", (float)matrix[i * ld + j]);
        }
        printf("\n");
    }
}

void printMatrices(float *matrixCpu, float *matrixGpu, int rows, int cols, int ld) {
    for (int i = 0; i < rows; ++i) {
        printf("CPU: ");
        for (int j = 0; j < cols; ++j) {
            printf("%5.2f:%-5.2f", matrixCpu[i * cols + j], matrixGpu[i * cols + j]);
        }
        printf(" :GPU\n");
    }
}

inline void checkCudaErrors(cudaError_t cudaStatus) {
    if (cudaStatus != cudaSuccess) {
        printf("cudaStatus == %d\n", cudaStatus);
        fflush(stdout);
    }
    assert(cudaStatus == cudaSuccess);
}

TEST(CublasMatrixMultiply, TestOptimize) {
    int rowsA = 1307;  // 128 + (rand() % 1024);
    int colsA = 2001;  // 128 + (rand() % 1024);
    int colsB = 1607;  // 128 + (rand() % 1024);

    /*
        int rowsA = 23 * 256 + 16;
        int colsA = 47 * 128 + 16;
        int colsB = 48 * 96;
    */
    CublasMatrixMultiply::instance().chooseOptimalKernel(0, rowsA, colsA, colsB, TensorDescriptor::DataType::FP16, true);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
