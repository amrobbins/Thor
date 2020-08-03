#include "MLDev.h"

#include "gtest/gtest.h"
#include "omp.h"

using std::string;

#define INTEREST_KERNEL KernelWithSpec::KernelIndex::_256_96_bigSharedBlockA16Restrict

// computes result = AB, where A is an mxk matrix and B is an kxn matrix. This makes result a mxn matrix.
void matrixMultiplyCpu(float *A, float *B, float *C, int rowsA, int colsA, int colsB, int lda, int ldb, int ldc, bool accumulate) {
    for (int ra = 0; ra < rowsA; ra++) {
        for (int cb = 0; cb < colsB; cb++) {
            float accum = 0.0;
            for (int carb = 0; carb < colsA; carb++)
                accum += A[ra * lda + carb] * B[carb * ldb + cb];

            if (accumulate)
                C[ra * ldc + cb] += accum;
            else
                C[ra * ldc + cb] = accum;
        }
    }
}

void matrixMultiplyCpuHalf(half *A, half *B, half *C, int rowsA, int colsA, int colsB, int lda, int ldb, int ldc, bool accumulate) {
    for (int ra = 0; ra < rowsA; ra++) {
        for (int cb = 0; cb < colsB; cb++) {
            float accum = 0.0;
            for (int carb = 0; carb < colsA; carb++)
                accum += A[ra * lda + carb] * B[carb * ldb + cb];

            if (accumulate)
                C[ra * ldc + cb] = (float)(C[ra * ldc + cb]) + accum;
            else
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

TEST(CublasMatrixMultiply, ChooseOptimalKernelWorksFP16) {
    srand(time(nullptr));

    ScopedGpu scopedGpu(0);
    Stream stream(0);

    for (int i = 0; i < 100; ++i) {
        int rowsA = 128 + (rand() % 1500);
        int colsA = 128 + (rand() % 1500);
        int colsB = 128 + (rand() % 1500);

        int ldA = colsA;
        int ldB = colsB;
        int ldC = colsB;
        bool useLdVersion;
        if (rand() % 2) {
            useLdVersion = true;
            ldA += rand() % 10;
            ldB += rand() % 10;
            ldC += rand() % 10;
        }

        TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU, 0);
        TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

        TensorDescriptor ADescriptor(TensorDescriptor::DataType::FP16, rowsA, ldA);
        TensorDescriptor BDescriptor(TensorDescriptor::DataType::FP16, colsA, ldB);
        TensorDescriptor CDescriptor(TensorDescriptor::DataType::FP16, rowsA, ldC);

        Tensor A(cpuPlacement, ADescriptor);
        Tensor B(cpuPlacement, BDescriptor);
        Tensor C(cpuPlacement, CDescriptor);
        Tensor A_d(gpuPlacement, ADescriptor);
        Tensor B_d(gpuPlacement, BDescriptor);
        Tensor C_d(gpuPlacement, CDescriptor);
        Tensor C_gpu_h(cpuPlacement, CDescriptor);

        half *AMem = (half *)A.getMemPtr();
        for (int row = 0; row < rowsA; ++row) {
            for (int col = 0; col < colsA; ++col) {
                AMem[row * ldA + col] = ((rand() % 100) - 50) / 10.0f;
            }
        }
        half *BMem = (half *)B.getMemPtr();
        for (int row = 0; row < colsA; ++row) {
            for (int col = 0; col < colsB; ++col) {
                BMem[row * ldB + col] = ((rand() % 100) - 50) / 10.0f;
            }
        }
        half *CMem = (half *)C.getMemPtr();
        for (int row = 0; row < rowsA; ++row) {
            for (int col = 0; col < colsB; ++col) {
                CMem[row * ldC + col] = ((rand() % 100) - 50) / 10.0f;
            }
        }

        /*
                printf("\n\nA:\n");
                printMatrix(AMem, rowsA, colsA, ldA);
                printf("\n\nB:\n");
                printMatrix(BMem, colsA, colsB, ldB);
                printf("\n\nC before:\n");
                printMatrix(CMem, colsA, colsB, ldC);
        */

        // CublasLt currently takes D = Alpha*(AB) + Beta*(AB+C), I believe this is a bug, will try to get it fixed. Until then no
        // accumulate.
        // bool accumulate = rand() % 2 ? true : false;
        bool accumulate = false;

        std::thread cpuWorker(matrixMultiplyCpuHalf,
                              (half *)A.getMemPtr(),
                              (half *)B.getMemPtr(),
                              (half *)C.getMemPtr(),
                              rowsA,
                              colsA,
                              colsB,
                              ldA,
                              ldB,
                              ldC,
                              accumulate);

        if (useLdVersion)
            CublasMatrixMultiply::instance().chooseOptimalKernel(
                0, rowsA, colsA, colsB, ldA, ldB, ldC, TensorDescriptor::DataType::FP16, false);
        else
            CublasMatrixMultiply::instance().chooseOptimalKernel(0, rowsA, colsA, colsB, TensorDescriptor::DataType::FP16, false);

        bool useWorkspace = rand() % 2;

        Tensor workspace_d;

        if (useWorkspace) {
            bool kernelWillRunOnGpu;
            int workspaceSizeInBytes = CublasMatrixMultiply::instance().getWorkspaceSizeInBytes(
                0, rowsA, colsA, colsB, ldA, ldB, ldC, TensorDescriptor::DataType::FP16, kernelWillRunOnGpu);
            assert(kernelWillRunOnGpu);

            if (workspaceSizeInBytes > 0) {
                TensorDescriptor workspaceDescriptor(TensorDescriptor::DataType::UINT8, workspaceSizeInBytes);
                workspace_d = Tensor(gpuPlacement, workspaceDescriptor);
            }
        }

        A_d.copyFromAsync(A, stream);
        B_d.copyFromAsync(B, stream);
        C_d.copyFromAsync(C, stream);

        if (useWorkspace) {
            if (useLdVersion) {
                CublasMatrixMultiply::instance().multiply(
                    A_d, B_d, C_d, workspace_d, rowsA, colsA, colsB, ldA, ldB, ldC, accumulate, TensorDescriptor::DataType::FP16, stream);
            } else {
                CublasMatrixMultiply::instance().multiply(
                    A_d, B_d, C_d, workspace_d, rowsA, colsA, colsB, accumulate, TensorDescriptor::DataType::FP16, stream);
            }
        } else {
            if (useLdVersion) {
                CublasMatrixMultiply::instance().multiply(
                    A_d, B_d, C_d, rowsA, colsA, colsB, ldA, ldB, ldC, accumulate, TensorDescriptor::DataType::FP16, stream);
            } else {
                CublasMatrixMultiply::instance().multiply(
                    A_d, B_d, C_d, rowsA, colsA, colsB, accumulate, TensorDescriptor::DataType::FP16, stream);
            }
        }

        C_gpu_h.copyFromAsync(C_d, stream);
        cpuWorker.join();
        stream.synchronize();

        float maxDiff = colsA * 0.005;

        half *CMemGpu = (half *)C_gpu_h.getMemPtr();

        /*
                printf("\n\nCPU C:\n");
                printMatrix(CMem, rowsA, colsB, ldC);
                printf("\n\nGPU C:\n");
                printMatrix(CMemGpu, rowsA, colsB, ldC);
        */

        for (int i = 0; i < rowsA; ++i) {
            for (int j = 0; j < colsB; ++j) {
                float diff = abs((float)(CMem[i * ldC + j]) - (float)(CMemGpu[i * ldC + j]));

                if (diff >= maxDiff) {
                    printf("arows %d acols %d bcols %d\n", rowsA, colsA, colsB);
                    printf("row %d col %d : CPU %f vs %f GPU\n", i, j, float(CMem[i * ldC + j]), float(CMemGpu[i * ldC + j]));
                    fflush(stdout);
                }
                ASSERT_LT(diff, maxDiff);
            }
        }
    }
}

TEST(CublasMatrixMultiply, ChooseOptimalKernelWorksFP32) {
    srand(time(nullptr));

    ScopedGpu scopedGpu(0);
    Stream stream(0);

    for (int i = 0; i < 10; ++i) {
        int rowsA = 128 + (rand() % 1500);
        int colsA = 128 + (rand() % 1500);
        int colsB = 128 + (rand() % 1500);

        int ldA = colsA;
        int ldB = colsB;
        int ldC = colsB;
        bool useLdVersion;
        if (rand() % 2) {
            useLdVersion = true;
            ldA += rand() % 10;
            ldB += rand() % 10;
            ldC += rand() % 10;
        }

        TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU, 0);
        TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

        TensorDescriptor ADescriptor(TensorDescriptor::DataType::FP32, rowsA, ldA);
        TensorDescriptor BDescriptor(TensorDescriptor::DataType::FP32, colsA, ldB);
        TensorDescriptor CDescriptor(TensorDescriptor::DataType::FP32, rowsA, ldC);

        Tensor A(cpuPlacement, ADescriptor);
        Tensor B(cpuPlacement, BDescriptor);
        Tensor C(cpuPlacement, CDescriptor);
        Tensor A_d(gpuPlacement, ADescriptor);
        Tensor B_d(gpuPlacement, BDescriptor);
        Tensor C_d(gpuPlacement, CDescriptor);
        Tensor C_gpu_h(cpuPlacement, CDescriptor);

        float *AMem = (float *)A.getMemPtr();
        for (int row = 0; row < rowsA; ++row) {
            for (int col = 0; col < colsA; ++col) {
                AMem[row * ldA + col] = ((rand() % 100) - 50) / 10.0f;
            }
        }
        float *BMem = (float *)B.getMemPtr();
        for (int row = 0; row < colsA; ++row) {
            for (int col = 0; col < colsB; ++col) {
                BMem[row * ldB + col] = ((rand() % 100) - 50) / 10.0f;
            }
        }
        float *CMem = (float *)C.getMemPtr();
        for (int row = 0; row < rowsA; ++row) {
            for (int col = 0; col < colsB; ++col) {
                CMem[row * ldC + col] = ((rand() % 100) - 50) / 10.0f;
            }
        }

        /*
                printf("\n\nA:\n");
                printMatrix(AMem, rowsA, colsA, ldA);
                printf("\n\nB:\n");
                printMatrix(BMem, colsA, colsB, ldB);
                printf("\n\nC before:\n");
                printMatrix(CMem, colsA, colsB, ldC);
        */

        // CublasLt currently takes D = Alpha*(AB) + Beta*(AB+C), I believe this is a bug, will try to get it fixed. Until then no
        // accumulate.
        // bool accumulate = rand() % 2 ? true : false;
        bool accumulate = false;

        std::thread cpuWorker(matrixMultiplyCpu,
                              (float *)A.getMemPtr(),
                              (float *)B.getMemPtr(),
                              (float *)C.getMemPtr(),
                              rowsA,
                              colsA,
                              colsB,
                              ldA,
                              ldB,
                              ldC,
                              accumulate);

        if (useLdVersion)
            CublasMatrixMultiply::instance().chooseOptimalKernel(
                0, rowsA, colsA, colsB, ldA, ldB, ldC, TensorDescriptor::DataType::FP32, false);
        else
            CublasMatrixMultiply::instance().chooseOptimalKernel(0, rowsA, colsA, colsB, TensorDescriptor::DataType::FP32, false);

        bool useWorkspace = rand() % 2;

        Tensor workspace_d;

        if (useWorkspace) {
            bool kernelWillRunOnGpu;
            int workspaceSizeInBytes = CublasMatrixMultiply::instance().getWorkspaceSizeInBytes(
                0, rowsA, colsA, colsB, ldA, ldB, ldC, TensorDescriptor::DataType::FP32, kernelWillRunOnGpu);
            assert(kernelWillRunOnGpu);

            if (workspaceSizeInBytes > 0) {
                TensorDescriptor workspaceDescriptor(TensorDescriptor::DataType::UINT8, workspaceSizeInBytes);
                workspace_d = Tensor(gpuPlacement, workspaceDescriptor);
            }
        }

        A_d.copyFromAsync(A, stream);
        B_d.copyFromAsync(B, stream);
        C_d.copyFromAsync(C, stream);

        if (useWorkspace) {
            if (useLdVersion) {
                CublasMatrixMultiply::instance().multiply(
                    A_d, B_d, C_d, workspace_d, rowsA, colsA, colsB, ldA, ldB, ldC, accumulate, TensorDescriptor::DataType::FP32, stream);
            } else {
                CublasMatrixMultiply::instance().multiply(
                    A_d, B_d, C_d, workspace_d, rowsA, colsA, colsB, accumulate, TensorDescriptor::DataType::FP32, stream);
            }
        } else {
            if (useLdVersion) {
                CublasMatrixMultiply::instance().multiply(
                    A_d, B_d, C_d, rowsA, colsA, colsB, ldA, ldB, ldC, accumulate, TensorDescriptor::DataType::FP32, stream);
            } else {
                CublasMatrixMultiply::instance().multiply(
                    A_d, B_d, C_d, rowsA, colsA, colsB, accumulate, TensorDescriptor::DataType::FP32, stream);
            }
        }

        C_gpu_h.copyFromAsync(C_d, stream);
        cpuWorker.join();
        stream.synchronize();

        float maxDiff = colsA * 0.0001;

        float *CMemGpu = (float *)C_gpu_h.getMemPtr();

        /*
                printf("\n\nCPU C:\n");
                printMatrix(CMem, rowsA, colsB, ldC);
                printf("\n\nGPU C:\n");
                printMatrix(CMemGpu, rowsA, colsB, ldC);
        */

        for (int i = 0; i < rowsA; ++i) {
            for (int j = 0; j < colsB; ++j) {
                float diff = abs(CMem[i * ldC + j] - CMemGpu[i * ldC + j]);

                if (diff >= maxDiff) {
                    printf("arows %d acols %d bcols %d\n", rowsA, colsA, colsB);
                    printf("row %d col %d : CPU %f vs %f GPU\n", i, j, CMem[i * ldC + j], CMemGpu[i * ldC + j]);
                    fflush(stdout);
                }
                ASSERT_LT(diff, maxDiff);
            }
        }
    }
}

/*
FIXME:

TEST(CublasMatrixMultiply, HeuristicKernelWorks) {
    int rowsA = 10 * 1024;  // 128 + (rand() % 1024);
    int colsA = 8 * 128;    // 128 + (rand() % 1024);
    int colsB = 10 * 1024;  // 128 + (rand() % 1024);

    CublasMatrixMultiply::instance().chooseOptimalKernel(0, rowsA, colsA, colsB, TensorDescriptor::DataType::FP32, true);
}
*/
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
