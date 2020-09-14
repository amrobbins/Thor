#include "Thor.h"

#include "gtest/gtest.h"

// computes result = AB, where A is an mxk matrix and B is an kxn matrix. This makes result a mxn matrix.
void matrixMultiplyCpu(float *A, float *B, float *C, int m, int n, int k) {
    for (int ra = 0; ra < m; ra++) {
        for (int cb = 0; cb < n; cb++) {
            float accum = 0.0;
            for (int carb = 0; carb < k; carb++) {
                accum += A[ra * k + carb] * B[carb * n + cb];
            }
            C[ra * n + cb] = accum;
        }
    }
}

void printMatrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%7.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

void printMatrixes(float *matrixCpu, float *matrixGpu, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        printf("CPU: ");
        for (int j = 0; j < cols; ++j) {
            printf("%7.2f:%-7.2f", matrixCpu[i * cols + j], matrixGpu[i * cols + j]);
        }
        printf(" :GPU\n");
    }
}

void matrixMultiplyCpu(half *A, half *B, half *C, int m, int n, int k) {
    for (int ra = 0; ra < m; ra++) {
        for (int cb = 0; cb < n; cb++) {
            half accum = 0.0;
            for (int carb = 0; carb < k; carb++) {
                accum = accum + (A[ra * k + carb] * B[carb * n + cb]);
            }
            C[ra * n + cb] = accum;
        }
    }
}

void printMatrix(half *matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%7.2f ", (float)matrix[i * cols + j]);
        }
        printf("\n");
    }
}

void printMatrixes(half *matrixCpu, half *matrixGpu, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        printf("CPU: ");
        for (int j = 0; j < cols; ++j) {
            printf("%7.2f:%-7.2f", (float)matrixCpu[i * cols + j], (float)matrixGpu[i * cols + j]);
        }
        printf(" :GPU\n");
    }
}

TEST(GpuMatrixMultiplyFp32, ProducesTheCorrectResult) {
    bool VERIFY = true;

    srand(time(NULL));

    int M = ((rand() % 128) + 7) * 8 + 128;
    int N = ((rand() % 128) + 7) * 8 + 128;
    int K = ((rand() % 128) + 7) * 8 + 128;
    // int M = 2*1024;
    // int K = 2*1024;
    // int N = 2*1056;
    // printf("%d %d %d\n", M, N, K);

    int numDevices;
    cudaError_t cudaStatus;
    cudaStatus = cudaGetDeviceCount(&numDevices);
    ASSERT_EQ(cudaStatus, cudaSuccess);
    ASSERT_GT(numDevices, 0);

    vector<MatrixMultiplyKernelInfo> kernelInfoArray;

    int maxDeviceToTest = 1;

    for (int deviceNum = 0; deviceNum <= maxDeviceToTest; ++deviceNum) {
        // printf("GPU %d\n", deviceNum);
        // fflush(stdout);
        kernelInfoArray.push_back(getBestGemmKernel(M,
                                                    N,
                                                    K,
                                                    deviceNum,
                                                    DataType::FP32,
                                                    ElementOrder::CPP_ROW_MAJOR,
                                                    ElementOrder::CPP_ROW_MAJOR,
                                                    ElementOrder::CPP_ROW_MAJOR,
                                                    true));
        // printf("\n\n");
        // fflush(stdout);
    }

    float *A;
    A = new float[M * K];
    float *B;
    B = new float[K * N];
    float *C_cpu;
    C_cpu = new float[M * N];

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            A[i * K + j] = randRange(-10.0, 10.0);
        }
    }
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < N; ++j) {
            B[i * N + j] = randRange(-10.0, 10.0);
        }
    }

    if (VERIFY)
        matrixMultiplyCpu(A, B, C_cpu, M, N, K);

    for (int deviceNum = 0; deviceNum <= maxDeviceToTest; ++deviceNum) {
        // printf("device %d\n", deviceNum);

        cudaStream_t stream;
        cudaStatus = cudaSetDevice(deviceNum);
        ASSERT_EQ(cudaStatus, cudaSuccess);
        cudaStatus = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        ASSERT_EQ(cudaStatus, cudaSuccess);

        cublasLtHandle_t cublasLtHandle;
        cublasStatus_t cublasStatus;
        cublasStatus = cublasLtCreate(&cublasLtHandle);
        assert(cublasStatus == CUBLAS_STATUS_SUCCESS);

        float *A_d = NULL;
        float *B_d = NULL;
        float *C_d = NULL;
        float *C_gpu = NULL;
        float *workspace_d = NULL;
        float *transposeBuffer_d = NULL;
        cudaStatus = cudaMalloc(&A_d, M * K * sizeof(float));
        ASSERT_EQ(cudaStatus, cudaSuccess);
        cudaStatus = cudaMalloc(&B_d, K * N * sizeof(float));
        ASSERT_EQ(cudaStatus, cudaSuccess);
        cudaStatus = cudaMalloc(&C_d, M * N * sizeof(float));
        ASSERT_EQ(cudaStatus, cudaSuccess);
        cudaStatus = cudaMalloc(&transposeBuffer_d, M * N * sizeof(float));
        ASSERT_EQ(cudaStatus, cudaSuccess);
        cudaStatus = cudaHostAlloc(&C_gpu, M * N * sizeof(float), cudaHostAllocPortable);
        ASSERT_EQ(cudaStatus, cudaSuccess);
        cudaStatus = cudaMemcpyAsync(A_d, A, M * K * sizeof(float), cudaMemcpyHostToDevice, stream);
        ASSERT_EQ(cudaStatus, cudaSuccess);
        cudaStatus = cudaMemcpyAsync(B_d, B, K * N * sizeof(float), cudaMemcpyHostToDevice, stream);
        ASSERT_EQ(cudaStatus, cudaSuccess);
        if (kernelInfoArray[0].workspaceSizeInBytes > 0) {
            cudaStatus = cudaMalloc(&workspace_d, kernelInfoArray[deviceNum].workspaceSizeInBytes);
            ASSERT_EQ(cudaStatus, cudaSuccess);
        }

        cublasStatus = matrixMultiply(cublasLtHandle, stream, A_d, B_d, C_d, workspace_d, kernelInfoArray[deviceNum], transposeBuffer_d);
        ASSERT_EQ(cublasStatus, CUBLAS_STATUS_SUCCESS);

        cudaStatus = cudaMemcpyAsync(C_gpu, C_d, M * N * sizeof(float), cudaMemcpyDeviceToHost, stream);
        ASSERT_EQ(cudaStatus, cudaSuccess);
        cudaStatus = cudaStreamSynchronize(stream);
        ASSERT_EQ(cudaStatus, cudaSuccess);

        if (N < 100 && M < 100 && K < 100) {
            printf("\nA:\n");
            printMatrix(A, M, K);
            printf("\nB:\n");
            printMatrix(B, K, N);
            printf("\nC:\n");
            printMatrix(C_cpu, M, N);

            printf("\n\n");
            printf("C_CPU:\n");
            printMatrix(C_cpu, M, N);
            printf("C_GPU:\n");
            printMatrix(C_gpu, M, N);
            fflush(stdout);
        }

        if (VERIFY) {
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N; ++j) {
                    // if(abs(C_cpu[i * N + j] - C_gpu[i * N + j]) >= 1.0) {
                    //    printf("[%d, %d] %f : %f\n", i, j, C_cpu[i * N + j], C_gpu[i * N + j]);
                    //    fflush(stdout);
                    //}
                    ASSERT_LT(abs(C_cpu[i * N + j] - C_gpu[i * N + j]), 2.0);
                }
            }
        }

        cudaStatus = cudaFree(A_d);
        ASSERT_EQ(cudaStatus, cudaSuccess);
        cudaStatus = cudaFree(B_d);
        ASSERT_EQ(cudaStatus, cudaSuccess);
        cudaStatus = cudaFree(C_d);
        ASSERT_EQ(cudaStatus, cudaSuccess);
        cudaStatus = cudaFree(transposeBuffer_d);
        ASSERT_EQ(cudaStatus, cudaSuccess);
        cudaStatus = cudaFreeHost(C_gpu);
        ASSERT_EQ(cudaStatus, cudaSuccess);
        if (workspace_d != NULL) {
            cudaStatus = cudaFree(workspace_d);
            ASSERT_EQ(cudaStatus, cudaSuccess);
        }

        cudaStreamDestroy(stream);
        cublasLtDestroy(cublasLtHandle);
        assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
    }

    delete[] A;
    delete[] B;
    delete[] C_cpu;
}

/* currently getting all 0's from gpu for fp16
TEST(GpuMatrixMultiplyFp16, ProducesTheCorrectResult) {
    bool VERIFY = true;

    srand(time(NULL));

    //int M = ((rand() % 128) + 7) * 8 + 128;
    //int N = ((rand() % 128) + 7) * 8 + 128;
    //int K = ((rand() % 128) + 7) * 8 + 128;
    int M = 1024;
    int N = 512;
    int K = 768;
    printf("%d %d %d\n", M, N, K);

    int numDevices;
    cudaError_t cudaStatus;
    cudaStatus = cudaGetDeviceCount(&numDevices);
    ASSERT_EQ(cudaStatus, cudaSuccess);
    ASSERT_GT(numDevices, 0);

    vector<MatrixMultiplyKernelInfo> kernelInfoArray;

    int maxDeviceToTest = 0;

    for (int deviceNum = 0; deviceNum <= maxDeviceToTest; ++deviceNum) {
        // printf("GPU %d\n", deviceNum);
        // fflush(stdout);
        kernelInfoArray.push_back(getBestGemmKernel(M,
                                                    N,
                                                    K,
                                                    deviceNum,
                                                    DataType::FP16,
                                                    ElementOrder::CPP_ROW_MAJOR,
                                                    ElementOrder::CPP_ROW_MAJOR,
                                                    ElementOrder::CPP_ROW_MAJOR,
                                                    true));
        // printf("\n\n");
        // fflush(stdout);
    }

    half *A;
    A = new half[M * K];
    half *B;
    B = new half[K * N];
    half *C_cpu;
    C_cpu = new half[M * N];

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            A[i * K + j] = randRange(-10.0, 10.0);
        }
    }
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < N; ++j) {
            B[i * N + j] = randRange(-10.0, 10.0);
        }
    }

    if (VERIFY)
        matrixMultiplyCpu(A, B, C_cpu, M, N, K);

    for (int deviceNum = 0; deviceNum <= maxDeviceToTest; ++deviceNum) {
        printf("device %d\n", deviceNum);

        cudaStream_t stream;
        cudaStatus = cudaSetDevice(deviceNum);
        ASSERT_EQ(cudaStatus, cudaSuccess);
        cudaStatus = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        ASSERT_EQ(cudaStatus, cudaSuccess);

        cublasLtHandle_t cublasLtHandle;
        cublasStatus_t cublasStatus;
        cublasStatus = cublasLtCreate(&cublasLtHandle);
        assert(cublasStatus == CUBLAS_STATUS_SUCCESS);

        half *A_d = NULL;
        half *B_d = NULL;
        half *C_d = NULL;
        half *C_gpu = NULL;
        half *workspace_d = NULL;
        half *transposeBuffer_d = NULL;
        cudaStatus = cudaMalloc(&A_d, M * K * sizeof(half));
        ASSERT_EQ(cudaStatus, cudaSuccess);
        cudaStatus = cudaMalloc(&B_d, K * N * sizeof(half));
        ASSERT_EQ(cudaStatus, cudaSuccess);
        cudaStatus = cudaMalloc(&C_d, M * N * sizeof(half));
        ASSERT_EQ(cudaStatus, cudaSuccess);
        cudaStatus = cudaMalloc(&transposeBuffer_d, M * N * sizeof(half));
        ASSERT_EQ(cudaStatus, cudaSuccess);
        cudaStatus = cudaHostAlloc(&C_gpu, M * N * sizeof(half), cudaHostAllocPortable);
        ASSERT_EQ(cudaStatus, cudaSuccess);
        cudaStatus = cudaMemcpyAsync(A_d, A, M * K * sizeof(half), cudaMemcpyHostToDevice, stream);
        ASSERT_EQ(cudaStatus, cudaSuccess);
        cudaStatus = cudaMemcpyAsync(B_d, B, K * N * sizeof(half), cudaMemcpyHostToDevice, stream);
        ASSERT_EQ(cudaStatus, cudaSuccess);
        if (kernelInfoArray[0].workspaceSizeInBytes > 0) {
            cudaStatus = cudaMalloc(&workspace_d, kernelInfoArray[deviceNum].workspaceSizeInBytes);
            ASSERT_EQ(cudaStatus, cudaSuccess);
        }

        cublasStatus = matrixMultiply(cublasLtHandle, stream, A_d, B_d, C_d, workspace_d, kernelInfoArray[deviceNum], transposeBuffer_d);
        ASSERT_EQ(cublasStatus, CUBLAS_STATUS_SUCCESS);

        cudaStatus = cudaMemcpyAsync(C_gpu, C_d, M * N * sizeof(half), cudaMemcpyDeviceToHost, stream);
        ASSERT_EQ(cudaStatus, cudaSuccess);
        cudaStatus = cudaStreamSynchronize(stream);
        ASSERT_EQ(cudaStatus, cudaSuccess);

        if (N < 100 && M < 100 && K < 100) {
            printf("\nA:\n");
            printMatrix(A, M, K);
            printf("\nB:\n");
            printMatrix(B, K, N);
            printf("\nC:\n");
            printMatrix(C_cpu, M, N);

            printf("\n\n");
            printf("C_CPU:\n");
            printMatrix(C_cpu, M, N);
            printf("C_GPU:\n");
            printMatrix(C_gpu, M, N);
            fflush(stdout);
        }

        if (VERIFY) {
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N; ++j) {
                    if(abs(C_cpu[i * N + j] - C_gpu[i * N + j]) >= 1.0) {
                        printf("[%d, %d] %f : %f\n", i, j, (float)C_cpu[i * N + j], (float)C_gpu[i * N + j]);
                        fflush(stdout);
                    }
                    EXPECT_LT(abs(C_cpu[i * N + j] - C_gpu[i * N + j]), 2.0);
                }
            }
        }

        cudaStatus = cudaFree(A_d);
        ASSERT_EQ(cudaStatus, cudaSuccess);
        cudaStatus = cudaFree(B_d);
        ASSERT_EQ(cudaStatus, cudaSuccess);
        cudaStatus = cudaFree(C_d);
        ASSERT_EQ(cudaStatus, cudaSuccess);
        cudaStatus = cudaFree(transposeBuffer_d);
        ASSERT_EQ(cudaStatus, cudaSuccess);
        cudaStatus = cudaFreeHost(C_gpu);
        ASSERT_EQ(cudaStatus, cudaSuccess);
        if (workspace_d != NULL) {
            cudaStatus = cudaFree(workspace_d);
            ASSERT_EQ(cudaStatus, cudaSuccess);
        }

        cudaStreamDestroy(stream);
        cublasLtDestroy(cublasLtHandle);
        assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
    }

    delete[] A;
    delete[] B;
    delete[] C_cpu;
}
*/

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
