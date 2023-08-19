#include "Thor.h"

#include "test/Utilities/TensorOperations/GpuMatrixMultiply/MatrixMultiplyTestHelper.h"

#include "gtest/gtest.h"
#include "omp.h"

using std::string;

using namespace ThorImplementation;

#define INTEREST_KERNEL KernelWithSpec::KernelIndex::_256_96_bigSharedBlockA16Restrict

inline void checkCudaErrors(cudaError_t cudaStatus) {
    if (cudaStatus != cudaSuccess) {
        printf("cudaStatus == %d\n", cudaStatus);
        fflush(stdout);
    }
    assert(cudaStatus == cudaSuccess);
}

TEST(CublasMatrixMultiply, ChooseOptimalKernelWorksFP32) {
    srand(time(nullptr));

    ScopedGpu scopedGpu(0);
    Stream stream(0);

    for (uint64_t i = 0; i < 4; ++i) {
        bool transposeA = rand() % 2;
        bool transposeB = rand() % 2;

        uint64_t rowsA = 128 + (rand() % 1500);
        uint64_t colsA = 128 + (rand() % 1500);
        uint64_t rowsB = 128 + (rand() % 1500);
        uint64_t colsB = 128 + (rand() % 1500);

        // Now make the operation legal
        if (!transposeA && !transposeB)
            rowsB = colsA;
        if (!transposeA && transposeB)
            colsB = colsA;
        if (transposeA && !transposeB)
            rowsB = rowsA;
        if (transposeA && transposeB)
            colsB = rowsA;

        uint64_t ldA = colsA;
        uint64_t ldB = colsB;
        uint64_t ldC = transposeB == false ? colsB : rowsB;
        bool useLdVersion = false;
        if (rand() % 2) {
            useLdVersion = true;
            ldA += rand() % 10;
            ldB += rand() % 10;
            ldC += rand() % 10;
        }

        uint64_t rowsC = transposeA == false ? rowsA : colsA;
        uint64_t colsC = transposeB == false ? colsB : rowsB;

        TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU, 0);
        TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

        TensorDescriptor ADescriptor(TensorDescriptor::DataType::FP32, {rowsA, ldA});
        TensorDescriptor BDescriptor(TensorDescriptor::DataType::FP32, {rowsB, ldB});
        TensorDescriptor CDescriptor(TensorDescriptor::DataType::FP32, {rowsC, ldC});

        Tensor A(cpuPlacement, ADescriptor);
        Tensor B(cpuPlacement, BDescriptor);
        Tensor C(cpuPlacement, CDescriptor);
        Tensor A_d(gpuPlacement, ADescriptor);
        Tensor B_d(gpuPlacement, BDescriptor);
        Tensor C_d(gpuPlacement, CDescriptor);
        Tensor C_gpu_h(cpuPlacement, CDescriptor);

        float *AMem = (float *)A.getMemPtr();
        for (uint64_t row = 0; row < rowsA; ++row) {
            for (uint64_t col = 0; col < colsA; ++col) {
                AMem[row * ldA + col] = ((rand() % 100) - 50) / 10.0f;
            }
        }
        float *BMem = (float *)B.getMemPtr();
        for (uint64_t row = 0; row < rowsB; ++row) {
            for (uint64_t col = 0; col < colsB; ++col) {
                BMem[row * ldB + col] = ((rand() % 100) - 50) / 10.0f;
            }
        }
        float *CMem = (float *)C.getMemPtr();
        for (uint64_t row = 0; row < rowsC; ++row) {
            for (uint64_t col = 0; col < colsC; ++col) {
                CMem[row * ldC + col] = ((rand() % 100) - 50) / 10.0f;
            }
        }

        // printf("\n\nA:\n");
        // printMatrix(AMem, rowsA, colsA, ldA);
        // printf("\n\nB:\n");
        // printMatrix(BMem, colsA, colsB, ldB);
        // printf("\n\nC before:\n");
        // printMatrix(CMem, colsA, colsB, ldC);

        // CublasLt currently takes D = Alpha*(AB) + Beta*(AB+C), I believe this is a bug, will try to get it fixed. Until then no
        // accumulate.
        bool accumulate = rand() % 2 ? true : false;
        bool negate = rand() % 2 ? true : false;

        verifyOperationIsLegal(rowsA, colsA, rowsB, colsB, ldA, ldB, ldC, transposeA, transposeB);

        if (useLdVersion)
            CublasMatrixMultiply::instance().chooseOptimalKernel(
                0, rowsA, colsA, rowsB, colsB, ldA, ldB, ldC, transposeA, transposeB, TensorDescriptor::DataType::FP32, false);
        else
            CublasMatrixMultiply::instance().chooseOptimalKernel(
                0, rowsA, colsA, rowsB, colsB, transposeA, transposeB, TensorDescriptor::DataType::FP32, false);

        bool useWorkspace = rand() % 2;

        Optional<Tensor> workspace_d;

        if (useWorkspace) {
            bool kernelWillRunOnGpu;
            uint64_t workspaceSizeInBytes = CublasMatrixMultiply::instance().getWorkspaceSizeInBytes(
                0, rowsA, colsA, rowsB, colsB, ldA, ldB, ldC, transposeA, transposeB, TensorDescriptor::DataType::FP32, kernelWillRunOnGpu);
            assert(kernelWillRunOnGpu);

            if (workspaceSizeInBytes > 0) {
                TensorDescriptor workspaceDescriptor(TensorDescriptor::DataType::UINT8, {workspaceSizeInBytes});
                workspace_d = Tensor(gpuPlacement, workspaceDescriptor);
            }
        }

        A_d.copyFromAsync(A, stream);
        B_d.copyFromAsync(B, stream);
        C_d.copyFromAsync(C, stream);
        stream.synchronize();

        std::thread cpuWorker(matrixMultiplyCpu,
                              (float *)A.getMemPtr(),
                              (float *)B.getMemPtr(),
                              (float *)C.getMemPtr(),
                              rowsA,
                              colsA,
                              rowsB,
                              colsB,
                              ldA,
                              ldB,
                              ldC,
                              transposeA,
                              transposeB,
                              accumulate,
                              negate);

        if (useLdVersion) {
            CublasMatrixMultiply::instance().multiply(A_d,
                                                      B_d,
                                                      C_d,
                                                      workspace_d,
                                                      rowsA,
                                                      colsA,
                                                      rowsB,
                                                      colsB,
                                                      ldA,
                                                      ldB,
                                                      ldC,
                                                      transposeA,
                                                      transposeB,
                                                      accumulate,
                                                      negate,
                                                      TensorDescriptor::DataType::FP32,
                                                      stream);
        } else {
            CublasMatrixMultiply::instance().multiply(A_d,
                                                      B_d,
                                                      C_d,
                                                      workspace_d,
                                                      rowsA,
                                                      colsA,
                                                      rowsB,
                                                      colsB,
                                                      transposeA,
                                                      transposeB,
                                                      accumulate,
                                                      negate,
                                                      TensorDescriptor::DataType::FP32,
                                                      stream);
        }

        C_gpu_h.copyFromAsync(C_d, stream);
        stream.synchronize();

        cpuWorker.join();

        float maxDiff = transposeA == false ? colsA * 0.0001 : rowsA * 0.0001;

        float *CMemGpu = (float *)C_gpu_h.getMemPtr();

        // printf("\n\nCPU C:\n");
        // printMatrix(CMem, rowsA, colsB, ldC);
        // printf("\n\nGPU C:\n");
        // printMatrix(CMemGpu, rowsA, colsB, ldC);

        for (uint64_t i = 0; i < rowsC; ++i) {
            for (uint64_t j = 0; j < colsC; ++j) {
                float diff = abs(CMem[i * ldC + j] - CMemGpu[i * ldC + j]);

                if (diff >= maxDiff) {
                    printf("arows %ld acols %ld brows %ld bcols %ld transposeA %i transposeB %i\n",
                           rowsA,
                           colsA,
                           rowsB,
                           colsB,
                           transposeA,
                           transposeB);
                    printf("row %ld col %ld : CPU %f vs %f GPU\n", i, j, CMem[i * ldC + j], CMemGpu[i * ldC + j]);
                    fflush(stdout);
                }
                ASSERT_LT(diff, maxDiff);
            }
        }
    }
}

TEST(CublasMatrixMultiply, ChooseOptimalKernelWorksFP16) {
    srand(time(nullptr));

    ScopedGpu scopedGpu(0);
    Stream stream(0);

    for (uint64_t i = 0; i < 4; ++i) {
        bool transposeA = rand() % 2;
        bool transposeB = rand() % 2;

        uint64_t rowsA = 128 + (rand() % 1500);
        uint64_t colsA = 128 + (rand() % 1500);
        uint64_t rowsB = 128 + (rand() % 1500);
        uint64_t colsB = 128 + (rand() % 1500);

        // Now make the operation legal
        if (!transposeA && !transposeB)
            rowsB = colsA;
        if (!transposeA && transposeB)
            colsB = colsA;
        if (transposeA && !transposeB)
            rowsB = rowsA;
        if (transposeA && transposeB)
            colsB = rowsA;

        uint64_t ldA = colsA;
        uint64_t ldB = colsB;
        uint64_t ldC = transposeB == false ? colsB : rowsB;
        bool useLdVersion = false;
        if (rand() % 2) {
            useLdVersion = true;
            ldA += rand() % 10;
            ldB += rand() % 10;
            ldC += rand() % 10;
        }

        uint64_t rowsC = transposeA == false ? rowsA : colsA;
        uint64_t colsC = transposeB == false ? colsB : rowsB;

        TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU, 0);
        TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

        TensorDescriptor ADescriptor(TensorDescriptor::DataType::FP16, {rowsA, ldA});
        TensorDescriptor BDescriptor(TensorDescriptor::DataType::FP16, {rowsB, ldB});
        TensorDescriptor CDescriptor(TensorDescriptor::DataType::FP16, {rowsC, ldC});

        Tensor A(cpuPlacement, ADescriptor);
        Tensor B(cpuPlacement, BDescriptor);
        Tensor C(cpuPlacement, CDescriptor);
        Tensor A_d(gpuPlacement, ADescriptor);
        Tensor B_d(gpuPlacement, BDescriptor);
        Tensor C_d(gpuPlacement, CDescriptor);
        Tensor C_gpu_h(cpuPlacement, CDescriptor);

        half *AMem = (half *)A.getMemPtr();
        for (uint64_t row = 0; row < rowsA; ++row) {
            for (uint64_t col = 0; col < colsA; ++col) {
                AMem[row * ldA + col] = ((rand() % 100) - 50) / 10.0f;
            }
        }
        half *BMem = (half *)B.getMemPtr();
        for (uint64_t row = 0; row < rowsB; ++row) {
            for (uint64_t col = 0; col < colsB; ++col) {
                BMem[row * ldB + col] = ((rand() % 100) - 50) / 10.0f;
            }
        }
        half *CMem = (half *)C.getMemPtr();
        for (uint64_t row = 0; row < rowsC; ++row) {
            for (uint64_t col = 0; col < colsC; ++col) {
                CMem[row * ldC + col] = ((rand() % 100) - 50) / 10.0f;
            }
        }

        // printf("\n\nA:\n");
        // printMatrix(AMem, rowsA, colsA, ldA);
        // printf("\n\nB:\n");
        // printMatrix(BMem, colsA, colsB, ldB);
        // printf("\n\nC before:\n");
        // printMatrix(CMem, colsA, colsB, ldC);

        // CublasLt currently takes D = Alpha*(AB) + Beta*(AB+C), I believe this is a bug, will try to get it fixed. Until then no
        // accumulate.
        bool accumulate = rand() % 2 ? true : false;
        bool negate = rand() % 2 ? true : false;

        verifyOperationIsLegal(rowsA, colsA, rowsB, colsB, ldA, ldB, ldC, transposeA, transposeB);

        if (useLdVersion)
            CublasMatrixMultiply::instance().chooseOptimalKernel(
                0, rowsA, colsA, rowsB, colsB, ldA, ldB, ldC, transposeA, transposeB, TensorDescriptor::DataType::FP16, false);
        else
            CublasMatrixMultiply::instance().chooseOptimalKernel(
                0, rowsA, colsA, rowsB, colsB, transposeA, transposeB, TensorDescriptor::DataType::FP16, false);

        bool useWorkspace = rand() % 2;

        Optional<Tensor> workspace_d;

        if (useWorkspace) {
            bool kernelWillRunOnGpu;
            uint64_t workspaceSizeInBytes = CublasMatrixMultiply::instance().getWorkspaceSizeInBytes(
                0, rowsA, colsA, rowsB, colsB, ldA, ldB, ldC, transposeA, transposeB, TensorDescriptor::DataType::FP16, kernelWillRunOnGpu);
            assert(kernelWillRunOnGpu);

            if (workspaceSizeInBytes > 0) {
                TensorDescriptor workspaceDescriptor(TensorDescriptor::DataType::UINT8, {workspaceSizeInBytes});
                workspace_d = Tensor(gpuPlacement, workspaceDescriptor);
            }
        }

        A_d.copyFromAsync(A, stream);
        B_d.copyFromAsync(B, stream);
        C_d.copyFromAsync(C, stream);
        stream.synchronize();

        std::thread cpuWorker(matrixMultiplyCpuHalf,
                              (half *)A.getMemPtr(),
                              (half *)B.getMemPtr(),
                              (half *)C.getMemPtr(),
                              rowsA,
                              colsA,
                              rowsB,
                              colsB,
                              ldA,
                              ldB,
                              ldC,
                              transposeA,
                              transposeB,
                              accumulate,
                              negate);

        if (useLdVersion) {
            CublasMatrixMultiply::instance().multiply(A_d,
                                                      B_d,
                                                      C_d,
                                                      workspace_d,
                                                      rowsA,
                                                      colsA,
                                                      rowsB,
                                                      colsB,
                                                      ldA,
                                                      ldB,
                                                      ldC,
                                                      transposeA,
                                                      transposeB,
                                                      accumulate,
                                                      negate,
                                                      TensorDescriptor::DataType::FP16,
                                                      stream);
        } else {
            CublasMatrixMultiply::instance().multiply(A_d,
                                                      B_d,
                                                      C_d,
                                                      workspace_d,
                                                      rowsA,
                                                      colsA,
                                                      rowsB,
                                                      colsB,
                                                      transposeA,
                                                      transposeB,
                                                      accumulate,
                                                      negate,
                                                      TensorDescriptor::DataType::FP16,
                                                      stream);
        }

        C_gpu_h.copyFromAsync(C_d, stream);
        stream.synchronize();

        cpuWorker.join();

        float maxDiff = transposeA == false ? colsA * 0.005 : rowsA * 0.005;

        half *CMemGpu = (half *)C_gpu_h.getMemPtr();

        // printf("\n\nCPU C:\n");
        // printMatrix(CMem, rowsA, colsB, ldC);
        // printf("\n\nGPU C:\n");
        // printMatrix(CMemGpu, rowsA, colsB, ldC);

        for (uint64_t i = 0; i < rowsC; ++i) {
            for (uint64_t j = 0; j < colsC; ++j) {
                float diff = abs((float)CMem[i * ldC + j] - (float)CMemGpu[i * ldC + j]);

                if (diff >= maxDiff) {
                    printf("arows %ld acols %ld brows %ld bcols %ld transposeA %i transposeB %i\n",
                           rowsA,
                           colsA,
                           rowsB,
                           colsB,
                           transposeA,
                           transposeB);
                    printf("row %ld col %ld : CPU %f vs %f GPU\n", i, j, (float)CMem[i * ldC + j], (float)CMemGpu[i * ldC + j]);
                    fflush(stdout);
                }
                ASSERT_LT(diff, maxDiff);
            }
        }
    }
}

TEST(CublasMatrixMultiply, HeuristicKernelWorksFP32) {
    srand(time(nullptr));

    ScopedGpu scopedGpu(0);
    Stream stream(0);

    for (uint64_t i = 0; i < 4; ++i) {
        bool transposeA = rand() % 2;
        bool transposeB = rand() % 2;

        uint64_t rowsA = 128 + (rand() % 1500);
        uint64_t colsA = 128 + (rand() % 1500);
        uint64_t rowsB = 128 + (rand() % 1500);
        uint64_t colsB = 128 + (rand() % 1500);

        // Now make the operation legal
        if (!transposeA && !transposeB)
            rowsB = colsA;
        if (!transposeA && transposeB)
            colsB = colsA;
        if (transposeA && !transposeB)
            rowsB = rowsA;
        if (transposeA && transposeB)
            colsB = rowsA;

        uint64_t ldA = colsA;
        uint64_t ldB = colsB;
        uint64_t ldC = transposeB == false ? colsB : rowsB;
        bool useLdVersion = false;
        if (rand() % 2) {
            useLdVersion = true;
            ldA += rand() % 10;
            ldB += rand() % 10;
            ldC += rand() % 10;
        }

        uint64_t rowsC = transposeA == false ? rowsA : colsA;
        uint64_t colsC = transposeB == false ? colsB : rowsB;

        TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU, 0);
        TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

        TensorDescriptor ADescriptor(TensorDescriptor::DataType::FP32, {rowsA, ldA});
        TensorDescriptor BDescriptor(TensorDescriptor::DataType::FP32, {rowsB, ldB});
        TensorDescriptor CDescriptor(TensorDescriptor::DataType::FP32, {rowsC, ldC});

        Tensor A(cpuPlacement, ADescriptor);
        Tensor B(cpuPlacement, BDescriptor);
        Tensor C(cpuPlacement, CDescriptor);
        Tensor A_d(gpuPlacement, ADescriptor);
        Tensor B_d(gpuPlacement, BDescriptor);
        Tensor C_d(gpuPlacement, CDescriptor);
        Tensor C_gpu_h(cpuPlacement, CDescriptor);

        float *AMem = (float *)A.getMemPtr();
        for (uint64_t row = 0; row < rowsA; ++row) {
            for (uint64_t col = 0; col < colsA; ++col) {
                AMem[row * ldA + col] = ((rand() % 100) - 50) / 10.0f;
            }
        }
        float *BMem = (float *)B.getMemPtr();
        for (uint64_t row = 0; row < rowsB; ++row) {
            for (uint64_t col = 0; col < colsB; ++col) {
                BMem[row * ldB + col] = ((rand() % 100) - 50) / 10.0f;
            }
        }
        float *CMem = (float *)C.getMemPtr();
        for (uint64_t row = 0; row < rowsC; ++row) {
            for (uint64_t col = 0; col < colsC; ++col) {
                CMem[row * ldC + col] = ((rand() % 100) - 50) / 10.0f;
            }
        }

        // printf("\n\nA:\n");
        // printMatrix(AMem, rowsA, colsA, ldA);
        // printf("\n\nB:\n");
        // printMatrix(BMem, colsA, colsB, ldB);
        // printf("\n\nC before:\n");
        // printMatrix(CMem, colsA, colsB, ldC);

        // CublasLt currently takes D = Alpha*(AB) + Beta*(AB+C), I believe this is a bug, will try to get it fixed. Until then no
        // accumulate.
        bool accumulate = rand() % 2 ? true : false;
        bool negate = rand() % 2 ? true : false;

        verifyOperationIsLegal(rowsA, colsA, rowsB, colsB, ldA, ldB, ldC, transposeA, transposeB);

        A_d.copyFromAsync(A, stream);
        B_d.copyFromAsync(B, stream);
        C_d.copyFromAsync(C, stream);
        stream.synchronize();

        std::thread cpuWorker(matrixMultiplyCpu,
                              (float *)A.getMemPtr(),
                              (float *)B.getMemPtr(),
                              (float *)C.getMemPtr(),
                              rowsA,
                              colsA,
                              rowsB,
                              colsB,
                              ldA,
                              ldB,
                              ldC,
                              transposeA,
                              transposeB,
                              accumulate,
                              negate);

        if (useLdVersion) {
            CublasMatrixMultiply::instance().multiplyUsingHeuristicKernelChoice(A_d,
                                                                                B_d,
                                                                                C_d,
                                                                                rowsA,
                                                                                colsA,
                                                                                rowsB,
                                                                                colsB,
                                                                                ldA,
                                                                                ldB,
                                                                                ldC,
                                                                                transposeA,
                                                                                transposeB,
                                                                                accumulate,
                                                                                negate,
                                                                                TensorDescriptor::DataType::FP32,
                                                                                stream);
        } else {
            CublasMatrixMultiply::instance().multiplyUsingHeuristicKernelChoice(A_d,
                                                                                B_d,
                                                                                C_d,
                                                                                rowsA,
                                                                                colsA,
                                                                                rowsB,
                                                                                colsB,
                                                                                transposeA,
                                                                                transposeB,
                                                                                accumulate,
                                                                                negate,
                                                                                TensorDescriptor::DataType::FP32,
                                                                                stream);
        }

        C_gpu_h.copyFromAsync(C_d, stream);
        stream.synchronize();

        cpuWorker.join();

        float maxDiff = transposeA == false ? colsA * 0.0001 : rowsA * 0.0001;

        float *CMemGpu = (float *)C_gpu_h.getMemPtr();

        // printf("\n\nCPU C:\n");
        // printMatrix(CMem, rowsA, colsB, ldC);
        // printf("\n\nGPU C:\n");
        // printMatrix(CMemGpu, rowsA, colsB, ldC);

        for (uint64_t i = 0; i < rowsC; ++i) {
            for (uint64_t j = 0; j < colsC; ++j) {
                float diff = abs(CMem[i * ldC + j] - CMemGpu[i * ldC + j]);

                if (diff >= maxDiff) {
                    printf("arows %ld acols %ld brows %ld bcols %ld transposeA %i transposeB %i\n",
                           rowsA,
                           colsA,
                           rowsB,
                           colsB,
                           transposeA,
                           transposeB);
                    printf("row %ld col %ld : CPU %f vs %f GPU\n", i, j, CMem[i * ldC + j], CMemGpu[i * ldC + j]);
                    fflush(stdout);
                }
                ASSERT_LT(diff, maxDiff);
            }
        }
    }
}

TEST(CublasMatrixMultiply, HeuristicKernelWorksFP16) {
    srand(time(nullptr));

    ScopedGpu scopedGpu(0);
    Stream stream(0);

    for (uint64_t i = 0; i < 4; ++i) {
        bool transposeA = rand() % 2;
        bool transposeB = rand() % 2;

        uint64_t rowsA = 128 + (rand() % 1500);
        uint64_t colsA = 128 + (rand() % 1500);
        uint64_t rowsB = 128 + (rand() % 1500);
        uint64_t colsB = 128 + (rand() % 1500);

        // Now make the operation legal
        if (!transposeA && !transposeB)
            rowsB = colsA;
        if (!transposeA && transposeB)
            colsB = colsA;
        if (transposeA && !transposeB)
            rowsB = rowsA;
        if (transposeA && transposeB)
            colsB = rowsA;

        uint64_t ldA = colsA;
        uint64_t ldB = colsB;
        uint64_t ldC = transposeB == false ? colsB : rowsB;
        bool useLdVersion = false;
        if (rand() % 2) {
            useLdVersion = true;
            ldA += rand() % 10;
            ldB += rand() % 10;
            ldC += rand() % 10;
        }

        uint64_t rowsC = transposeA == false ? rowsA : colsA;
        uint64_t colsC = transposeB == false ? colsB : rowsB;

        TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU, 0);
        TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

        TensorDescriptor ADescriptor(TensorDescriptor::DataType::FP16, {rowsA, ldA});
        TensorDescriptor BDescriptor(TensorDescriptor::DataType::FP16, {rowsB, ldB});
        TensorDescriptor CDescriptor(TensorDescriptor::DataType::FP16, {rowsC, ldC});

        Tensor A(cpuPlacement, ADescriptor);
        Tensor B(cpuPlacement, BDescriptor);
        Tensor C(cpuPlacement, CDescriptor);
        Tensor A_d(gpuPlacement, ADescriptor);
        Tensor B_d(gpuPlacement, BDescriptor);
        Tensor C_d(gpuPlacement, CDescriptor);
        Tensor C_gpu_h(cpuPlacement, CDescriptor);

        half *AMem = (half *)A.getMemPtr();
        for (uint64_t row = 0; row < rowsA; ++row) {
            for (uint64_t col = 0; col < colsA; ++col) {
                AMem[row * ldA + col] = ((rand() % 100) - 50) / 10.0f;
            }
        }
        half *BMem = (half *)B.getMemPtr();
        for (uint64_t row = 0; row < rowsB; ++row) {
            for (uint64_t col = 0; col < colsB; ++col) {
                BMem[row * ldB + col] = ((rand() % 100) - 50) / 10.0f;
            }
        }
        half *CMem = (half *)C.getMemPtr();
        for (uint64_t row = 0; row < rowsC; ++row) {
            for (uint64_t col = 0; col < colsC; ++col) {
                CMem[row * ldC + col] = ((rand() % 100) - 50) / 10.0f;
            }
        }

        // printf("\n\nA:\n");
        // printMatrix(AMem, rowsA, colsA, ldA);
        // printf("\n\nB:\n");
        // printMatrix(BMem, colsA, colsB, ldB);
        // printf("\n\nC before:\n");
        // printMatrix(CMem, colsA, colsB, ldC);

        // CublasLt currently takes D = Alpha*(AB) + Beta*(AB+C), I believe this is a bug, will try to get it fixed. Until then no
        // accumulate.
        bool accumulate = rand() % 2 ? true : false;
        bool negate = rand() % 2 ? true : false;

        verifyOperationIsLegal(rowsA, colsA, rowsB, colsB, ldA, ldB, ldC, transposeA, transposeB);

        if (useLdVersion)
            CublasMatrixMultiply::instance().chooseOptimalKernel(
                0, rowsA, colsA, rowsB, colsB, ldA, ldB, ldC, transposeA, transposeB, TensorDescriptor::DataType::FP16, false);
        else
            CublasMatrixMultiply::instance().chooseOptimalKernel(
                0, rowsA, colsA, rowsB, colsB, transposeA, transposeB, TensorDescriptor::DataType::FP16, false);

        A_d.copyFromAsync(A, stream);
        B_d.copyFromAsync(B, stream);
        C_d.copyFromAsync(C, stream);
        stream.synchronize();

        std::thread cpuWorker(matrixMultiplyCpuHalf,
                              (half *)A.getMemPtr(),
                              (half *)B.getMemPtr(),
                              (half *)C.getMemPtr(),
                              rowsA,
                              colsA,
                              rowsB,
                              colsB,
                              ldA,
                              ldB,
                              ldC,
                              transposeA,
                              transposeB,
                              accumulate,
                              negate);

        if (useLdVersion) {
            CublasMatrixMultiply::instance().multiplyUsingHeuristicKernelChoice(A_d,
                                                                                B_d,
                                                                                C_d,
                                                                                rowsA,
                                                                                colsA,
                                                                                rowsB,
                                                                                colsB,
                                                                                ldA,
                                                                                ldB,
                                                                                ldC,
                                                                                transposeA,
                                                                                transposeB,
                                                                                accumulate,
                                                                                negate,
                                                                                TensorDescriptor::DataType::FP16,
                                                                                stream);
        } else {
            CublasMatrixMultiply::instance().multiplyUsingHeuristicKernelChoice(A_d,
                                                                                B_d,
                                                                                C_d,
                                                                                rowsA,
                                                                                colsA,
                                                                                rowsB,
                                                                                colsB,
                                                                                transposeA,
                                                                                transposeB,
                                                                                accumulate,
                                                                                negate,
                                                                                TensorDescriptor::DataType::FP16,
                                                                                stream);
        }

        C_gpu_h.copyFromAsync(C_d, stream);
        stream.synchronize();

        cpuWorker.join();

        float maxDiff = transposeA == false ? colsA * 0.005 : rowsA * 0.005;

        half *CMemGpu = (half *)C_gpu_h.getMemPtr();

        // printf("\n\nCPU C:\n");
        // printMatrix(CMem, rowsA, colsB, ldC);
        // printf("\n\nGPU C:\n");
        // printMatrix(CMemGpu, rowsA, colsB, ldC);

        for (uint64_t i = 0; i < rowsC; ++i) {
            for (uint64_t j = 0; j < colsC; ++j) {
                float diff = abs((float)CMem[i * ldC + j] - (float)CMemGpu[i * ldC + j]);

                if (diff >= maxDiff) {
                    printf("arows %ld acols %ld brows %ld bcols %ld transposeA %i transposeB %i\n",
                           rowsA,
                           colsA,
                           rowsB,
                           colsB,
                           transposeA,
                           transposeB);
                    printf("row %ld col %ld : CPU %f vs %f GPU\n", i, j, (float)CMem[i * ldC + j], (float)CMemGpu[i * ldC + j]);
                    fflush(stdout);
                }
                ASSERT_LT(diff, maxDiff);
            }
        }
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
