#include "Utilities/TensorOperations/GpuMatrixMultiply/CublasMatrixMultiply.h"

#include "test/Utilities/TensorOperations/GpuMatrixMultiply/MatrixMultiplyTestHelper.h"

#include "gtest/gtest.h"
#include "omp.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

using std::string;

using namespace ThorImplementation;

static inline void checkCudaErrors(cudaError_t cudaStatus) {
    if (cudaStatus != cudaSuccess) {
        printf("cudaStatus == %d; %s\n", cudaStatus, cudaGetErrorString(cudaStatus));
        fflush(stdout);
    }
    assert(cudaStatus == cudaSuccess);
}

TEST(CublasMatrixMultiply, ChooseOptimalMatrixMultiplyKernelWorksFP32) {
    srand(time(nullptr));

    ScopedGpu scopedGpu(0);
    Stream stream(0);

    for (uint64_t i = 0; i < 1; ++i) {
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
            CublasMatrixMultiply::instance().chooseOptimalMatrixMultiplyKernel(
                0, rowsA, colsA, rowsB, colsB, ldA, ldB, ldC, transposeA, transposeB, TensorDescriptor::DataType::FP32, false);
        else
            CublasMatrixMultiply::instance().chooseOptimalMatrixMultiplyKernel(
                0, rowsA, colsA, rowsB, colsB, transposeA, transposeB, TensorDescriptor::DataType::FP32, false);

        bool useWorkspace = rand() % 2;

        Optional<Tensor> workspace_d;

        if (useWorkspace) {
            bool kernelWillRunOnGpu;
            uint64_t workspaceSizeInBytes;
            if (useLdVersion) {
                workspaceSizeInBytes =
                    CublasMatrixMultiply::instance().getMatrixMultiplyWorkspaceSizeInBytes(0,
                                                                                           rowsA,
                                                                                           colsA,
                                                                                           rowsB,
                                                                                           colsB,
                                                                                           ldA,
                                                                                           ldB,
                                                                                           ldC,
                                                                                           transposeA,
                                                                                           transposeB,
                                                                                           TensorDescriptor::DataType::FP32,
                                                                                           kernelWillRunOnGpu);
            } else {
                workspaceSizeInBytes = CublasMatrixMultiply::instance().getMatrixMultiplyWorkspaceSizeInBytes(
                    0, rowsA, colsA, rowsB, colsB, transposeA, transposeB, TensorDescriptor::DataType::FP32, kernelWillRunOnGpu);
            }
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

TEST(CublasMatrixMultiply, ChooseOptimalMatrixMultiplyKernelWorksFP16) {
    srand(time(nullptr));

    ScopedGpu scopedGpu(0);
    Stream stream(0);

    for (uint64_t i = 0; i < 1; ++i) {
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
            CublasMatrixMultiply::instance().chooseOptimalMatrixMultiplyKernel(
                0, rowsA, colsA, rowsB, colsB, ldA, ldB, ldC, transposeA, transposeB, TensorDescriptor::DataType::FP16, false);
        else
            CublasMatrixMultiply::instance().chooseOptimalMatrixMultiplyKernel(
                0, rowsA, colsA, rowsB, colsB, transposeA, transposeB, TensorDescriptor::DataType::FP16, false);

        bool useWorkspace = rand() % 2;

        Optional<Tensor> workspace_d;

        if (useWorkspace) {
            bool kernelWillRunOnGpu;
            uint64_t workspaceSizeInBytes;
            if (useLdVersion) {
                workspaceSizeInBytes =
                    CublasMatrixMultiply::instance().getMatrixMultiplyWorkspaceSizeInBytes(0,
                                                                                           rowsA,
                                                                                           colsA,
                                                                                           rowsB,
                                                                                           colsB,
                                                                                           ldA,
                                                                                           ldB,
                                                                                           ldC,
                                                                                           transposeA,
                                                                                           transposeB,
                                                                                           TensorDescriptor::DataType::FP16,
                                                                                           kernelWillRunOnGpu);
            } else {
                workspaceSizeInBytes = CublasMatrixMultiply::instance().getMatrixMultiplyWorkspaceSizeInBytes(
                    0, rowsA, colsA, rowsB, colsB, transposeA, transposeB, TensorDescriptor::DataType::FP16, kernelWillRunOnGpu);
            }
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

TEST(CublasMatrixMultiply, HeuristicMatrixMultiplyKernelWorksFP32) {
    srand(time(nullptr));

    ScopedGpu scopedGpu(0);
    Stream stream(0);

    for (uint64_t i = 0; i < 1; ++i) {
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
        if (rand() % 2) {
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

TEST(CublasMatrixMultiply, HeuristicMatrixMultiplyKernelWorksFP16) {
    srand(time(nullptr));

    ScopedGpu scopedGpu(0);
    Stream stream(0);

    for (uint64_t i = 0; i < 1; ++i) {
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
            CublasMatrixMultiply::instance().chooseOptimalMatrixMultiplyKernel(
                0, rowsA, colsA, rowsB, colsB, ldA, ldB, ldC, transposeA, transposeB, TensorDescriptor::DataType::FP16, false);
        else
            CublasMatrixMultiply::instance().chooseOptimalMatrixMultiplyKernel(
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

TEST(CublasMatrixMultiply, ChooseOptimalGemmKernelWorksFP32) {
    srand(time(nullptr));

    ScopedGpu scopedGpu(0);
    Stream stream(0);

    for (uint64_t i = 0; i < 1; ++i) {
        bool transposeA = rand() % 2;
        bool transposeB = rand() % 2;
        bool transposeC = rand() % 2;
        // FIXME: TEMP
        transposeC = false;
        bool CDInPlace = rand() % 2;
        if (CDInPlace)
            transposeC = false;

        uint64_t rowsA = 128 + (rand() % 1500);
        uint64_t colsA = 128 + (rand() % 1500);
        uint64_t rowsB = 128 + (rand() % 1500);
        uint64_t colsB = 128 + (rand() % 1500);
        uint64_t rowsC;
        uint64_t colsC;

        // Now make the operation legal
        if (!transposeA && !transposeB)
            rowsB = colsA;
        if (!transposeA && transposeB)
            colsB = colsA;
        if (transposeA && !transposeB)
            rowsB = rowsA;
        if (transposeA && transposeB)
            colsB = rowsA;
        if (transposeC) {
            rowsC = transposeB ? rowsB : colsB;
            colsC = transposeA ? colsA : rowsA;
        } else {
            rowsC = transposeA ? colsA : rowsA;
            colsC = transposeB ? rowsB : colsB;
        }
        uint64_t rowsD = transposeA ? colsA : rowsA;
        uint64_t colsD = transposeB ? rowsB : colsB;

        uint64_t ldA = colsA;
        uint64_t ldB = colsB;
        uint64_t ldC = colsC;
        uint64_t ldD = colsD;
        if (rand() % 2) {
            ldA += rand() % 10;
            ldB += rand() % 10;
            ldC += rand() % 10;
            ldD = ldC;
        }

        TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU, 0);
        TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

        TensorDescriptor ADescriptor(TensorDescriptor::DataType::FP32, {rowsA, ldA});
        TensorDescriptor BDescriptor(TensorDescriptor::DataType::FP32, {rowsB, ldB});
        TensorDescriptor CDescriptor(TensorDescriptor::DataType::FP32, {rowsC, ldC});
        TensorDescriptor DDescriptor(TensorDescriptor::DataType::FP32, {rowsD, ldD});

        Tensor A(cpuPlacement, ADescriptor);
        Tensor B(cpuPlacement, BDescriptor);
        Tensor C(cpuPlacement, CDescriptor);
        Tensor D;
        if (CDInPlace)
            D = C;
        else
            D = Tensor(cpuPlacement, DDescriptor);
        Tensor A_d(gpuPlacement, ADescriptor);
        Tensor B_d(gpuPlacement, BDescriptor);
        Tensor C_d(gpuPlacement, CDescriptor);
        Tensor D_d;
        if (CDInPlace)
            D_d = C_d;
        else
            D_d = Tensor(gpuPlacement, DDescriptor);
        Tensor D_gpu_h(cpuPlacement, DDescriptor);

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
        float *DMem;
        if (CDInPlace) {
            DMem = CMem;
        } else {
            DMem = (float *)D.getMemPtr();
            for (uint64_t row = 0; row < rowsD; ++row) {
                for (uint64_t col = 0; col < colsD; ++col) {
                    DMem[row * ldD + col] = ((rand() % 100) - 50) / 10.0f;
                }
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
        float alpha = (rand() % 1000) / 10.0f;
        float beta = (rand() % 1000) / 10.0f;

        verifyOperationIsLegal(rowsA,
                               colsA,
                               rowsB,
                               colsB,
                               rowsC,
                               colsC,
                               ldA,
                               ldB,
                               ldC,
                               ldD,
                               transposeA,
                               transposeB,
                               transposeC,
                               CDInPlace,
                               C.getMemPtr(),
                               D.getMemPtr(),
                               C_d.getMemPtr(),
                               D_d.getMemPtr());

        CublasMatrixMultiply::instance().chooseOptimalGemmKernel(
            0, rowsA, colsA, rowsB, colsB, ldA, ldB, ldC, ldD, transposeA, transposeB, transposeC, TensorDescriptor::DataType::FP32, false);

        bool useWorkspace = rand() % 2;

        Optional<Tensor> workspace_d;

        if (useWorkspace) {
            bool kernelWillRunOnGpu;
            uint64_t workspaceSizeInBytes = CublasMatrixMultiply::instance().getGemmWorkspaceSizeInBytes(0,
                                                                                                         rowsA,
                                                                                                         colsA,
                                                                                                         rowsB,
                                                                                                         colsB,
                                                                                                         ldA,
                                                                                                         ldB,
                                                                                                         ldC,
                                                                                                         ldD,
                                                                                                         transposeA,
                                                                                                         transposeB,
                                                                                                         transposeC,
                                                                                                         TensorDescriptor::DataType::FP32,
                                                                                                         kernelWillRunOnGpu);
            assert(kernelWillRunOnGpu);

            if (workspaceSizeInBytes > 0) {
                TensorDescriptor workspaceDescriptor(TensorDescriptor::DataType::UINT8, {workspaceSizeInBytes});
                workspace_d = Tensor(gpuPlacement, workspaceDescriptor);
            }
        }

        A_d.copyFromAsync(A, stream);
        B_d.copyFromAsync(B, stream);
        C_d.copyFromAsync(C, stream);
        if (!CDInPlace)
            D_d.copyFromAsync(D, stream);
        stream.synchronize();

        std::thread cpuWorker(gemmCpuFp32,
                              (float *)A.getMemPtr(),
                              (float *)B.getMemPtr(),
                              (float *)C.getMemPtr(),
                              (float *)D.getMemPtr(),
                              rowsA,
                              colsA,
                              rowsB,
                              colsB,
                              ldA,
                              ldB,
                              ldC,
                              ldD,
                              transposeA,
                              transposeB,
                              transposeC,
                              alpha,
                              beta);

        CublasMatrixMultiply::instance().gemm(A_d,
                                              B_d,
                                              C_d,
                                              D_d,
                                              workspace_d,
                                              rowsA,
                                              colsA,
                                              rowsB,
                                              colsB,
                                              transposeA,
                                              transposeB,
                                              transposeC,
                                              &alpha,
                                              &beta,
                                              TensorDescriptor::DataType::FP32,
                                              stream);

        D_gpu_h.copyFromAsync(D_d, stream);
        stream.synchronize();

        cpuWorker.join();

        float maxDiff = transposeA == false ? colsA * 0.0005 : rowsA * 0.0005;

        float *DMemGpu = (float *)D_gpu_h.getMemPtr();

        // printf("\n\nCPU C:\n");
        // printMatrix(CMem, rowsA, colsB, ldC);
        // printf("\n\nGPU C:\n");
        // printMatrix(CMemGpu, rowsA, colsB, ldC);

        for (uint64_t i = 0; i < rowsC; ++i) {
            for (uint64_t j = 0; j < colsC; ++j) {
                float diff = abs(DMem[i * ldD + j] - DMemGpu[i * ldD + j]);

                if (diff >= maxDiff) {
                    printf("arows %ld acols %ld brows %ld bcols %ld transposeA %i transposeB %i\n",
                           rowsA,
                           colsA,
                           rowsB,
                           colsB,
                           transposeA,
                           transposeB);
                    printf("row %ld col %ld : CPU %f vs %f GPU\n", i, j, DMem[i * ldD + j], DMemGpu[i * ldD + j]);
                    fflush(stdout);
                }
                ASSERT_LT(diff, maxDiff);
            }
        }
    }
}

TEST(CublasMatrixMultiply, ChooseOptimalGemmKernelWorksFP16) {
    srand(time(nullptr));

    ScopedGpu scopedGpu(0);
    Stream stream(0);

    for (uint64_t i = 0; i < 1; ++i) {
        bool transposeA = rand() % 2;
        bool transposeB = rand() % 2;
        bool transposeC = rand() % 2;
        // FIXME: TEMP
        transposeC = false;
        bool CDInPlace = rand() % 2;
        if (CDInPlace)
            transposeC = false;

        uint64_t rowsA = 128 + (rand() % 1500);
        uint64_t colsA = 128 + (rand() % 1500);
        uint64_t rowsB = 128 + (rand() % 1500);
        uint64_t colsB = 128 + (rand() % 1500);
        uint64_t rowsC;
        uint64_t colsC;

        // Now make the operation legal
        if (!transposeA && !transposeB)
            rowsB = colsA;
        if (!transposeA && transposeB)
            colsB = colsA;
        if (transposeA && !transposeB)
            rowsB = rowsA;
        if (transposeA && transposeB)
            colsB = rowsA;
        if (transposeC) {
            rowsC = transposeB ? rowsB : colsB;
            colsC = transposeA ? colsA : rowsA;
        } else {
            rowsC = transposeA ? colsA : rowsA;
            colsC = transposeB ? rowsB : colsB;
        }
        uint64_t rowsD = transposeA ? colsA : rowsA;
        uint64_t colsD = transposeB ? rowsB : colsB;

        uint64_t ldA = colsA;
        uint64_t ldB = colsB;
        uint64_t ldC = colsC;
        uint64_t ldD = colsD;
        if (rand() % 2) {
            ldA += rand() % 10;
            ldB += rand() % 10;
            ldC += rand() % 10;
            // It seems that cublas does not support this for FP16, but does for FP32:
            // if (CDInPlace)
            //    ldD = ldC;
            // else
            //    ldD += rand() % 10;
            ldD = ldC;
        }

        TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU, 0);
        TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

        TensorDescriptor ADescriptor(TensorDescriptor::DataType::FP16, {rowsA, ldA});
        TensorDescriptor BDescriptor(TensorDescriptor::DataType::FP16, {rowsB, ldB});
        TensorDescriptor CDescriptor(TensorDescriptor::DataType::FP16, {rowsC, ldC});
        TensorDescriptor DDescriptor(TensorDescriptor::DataType::FP16, {rowsD, ldD});

        Tensor A(cpuPlacement, ADescriptor);
        Tensor B(cpuPlacement, BDescriptor);
        Tensor C(cpuPlacement, CDescriptor);
        Tensor D;
        if (CDInPlace)
            D = C;
        else
            D = Tensor(cpuPlacement, DDescriptor);
        Tensor A_d(gpuPlacement, ADescriptor);
        Tensor B_d(gpuPlacement, BDescriptor);
        Tensor C_d(gpuPlacement, CDescriptor);
        Tensor D_d;
        if (CDInPlace)
            D_d = C_d;
        else
            D_d = Tensor(gpuPlacement, DDescriptor);
        Tensor D_gpu_h(cpuPlacement, DDescriptor);

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
        half *DMem;
        if (CDInPlace) {
            DMem = CMem;
        } else {
            DMem = (half *)D.getMemPtr();
            for (uint64_t row = 0; row < rowsD; ++row) {
                for (uint64_t col = 0; col < colsD; ++col) {
                    DMem[row * ldD + col] = ((rand() % 100) - 50) / 10.0f;
                }
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
        float alpha = (rand() % 100) / 10.0f;
        float beta = (rand() % 100) / 10.0f;

        verifyOperationIsLegal(rowsA,
                               colsA,
                               rowsB,
                               colsB,
                               rowsC,
                               colsC,
                               ldA,
                               ldB,
                               ldC,
                               ldD,
                               transposeA,
                               transposeB,
                               transposeC,
                               CDInPlace,
                               C.getMemPtr(),
                               D.getMemPtr(),
                               C_d.getMemPtr(),
                               D_d.getMemPtr());

        CublasMatrixMultiply::instance().chooseOptimalGemmKernel(
            0, rowsA, colsA, rowsB, colsB, ldA, ldB, ldC, ldD, transposeA, transposeB, transposeC, TensorDescriptor::DataType::FP16, false);

        bool useWorkspace = rand() % 2;

        Optional<Tensor> workspace_d;

        if (useWorkspace) {
            bool kernelWillRunOnGpu;
            uint64_t workspaceSizeInBytes = CublasMatrixMultiply::instance().getGemmWorkspaceSizeInBytes(0,
                                                                                                         rowsA,
                                                                                                         colsA,
                                                                                                         rowsB,
                                                                                                         colsB,
                                                                                                         ldA,
                                                                                                         ldB,
                                                                                                         ldC,
                                                                                                         ldD,
                                                                                                         transposeA,
                                                                                                         transposeB,
                                                                                                         transposeC,
                                                                                                         TensorDescriptor::DataType::FP16,
                                                                                                         kernelWillRunOnGpu);
            assert(kernelWillRunOnGpu);

            if (workspaceSizeInBytes > 0) {
                TensorDescriptor workspaceDescriptor(TensorDescriptor::DataType::UINT8, {workspaceSizeInBytes});
                workspace_d = Tensor(gpuPlacement, workspaceDescriptor);
            }
        }

        A_d.copyFromAsync(A, stream);
        B_d.copyFromAsync(B, stream);
        C_d.copyFromAsync(C, stream);
        if (!CDInPlace)
            D_d.copyFromAsync(D, stream);
        stream.synchronize();

        std::thread cpuWorker(gemmCpuFp16,
                              (half *)A.getMemPtr(),
                              (half *)B.getMemPtr(),
                              (half *)C.getMemPtr(),
                              (half *)D.getMemPtr(),
                              rowsA,
                              colsA,
                              rowsB,
                              colsB,
                              ldA,
                              ldB,
                              ldC,
                              ldD,
                              transposeA,
                              transposeB,
                              transposeC,
                              alpha,
                              beta);

        CublasMatrixMultiply::instance().gemm(A_d,
                                              B_d,
                                              C_d,
                                              D_d,
                                              workspace_d,
                                              rowsA,
                                              colsA,
                                              rowsB,
                                              colsB,
                                              transposeA,
                                              transposeB,
                                              transposeC,
                                              &alpha,
                                              &beta,
                                              TensorDescriptor::DataType::FP16,
                                              stream);

        D_gpu_h.copyFromAsync(D_d, stream);
        stream.synchronize();

        cpuWorker.join();

        float maxDiff = transposeA == false ? colsA * 0.05f : rowsA * 0.05f;

        half *DMemGpu = (half *)D_gpu_h.getMemPtr();

        // printf("\n\nCPU C:\n");
        // printMatrix(CMem, rowsA, colsB, ldC);
        // printf("\n\nGPU C:\n");
        // printMatrix(CMemGpu, rowsA, colsB, ldC);

        for (uint64_t i = 0; i < rowsC; ++i) {
            for (uint64_t j = 0; j < colsC; ++j) {
                float diff = abs((float)(DMem[i * ldD + j] - DMemGpu[i * ldD + j]));

                if (!(diff < maxDiff)) {
                    printf("arows %ld acols %ld brows %ld bcols %ld transposeA %i transposeB %i\n",
                           rowsA,
                           colsA,
                           rowsB,
                           colsB,
                           transposeA,
                           transposeB);
                    printf("row %ld col %ld : CPU %f vs %f GPU\n", i, j, (float)DMem[i * ldD + j], (float)DMemGpu[i * ldD + j]);
                    fflush(stdout);
                }
                ASSERT_LT(diff, maxDiff);
            }
        }
    }
}

TEST(CublasMatrixMultiply, HeuristicGemmKernelWorksFP32) {
    srand(time(nullptr));

    ScopedGpu scopedGpu(0);
    Stream stream(0);

    for (uint64_t i = 0; i < 1; ++i) {
        bool transposeA = rand() % 2;
        bool transposeB = rand() % 2;
        bool transposeC = rand() % 2;
        // FIXME: TEMP
        transposeC = false;
        bool CDInPlace = rand() % 2;
        if (CDInPlace)
            transposeC = false;

        uint64_t rowsA = 128 + (rand() % 1500);
        uint64_t colsA = 128 + (rand() % 1500);
        uint64_t rowsB = 128 + (rand() % 1500);
        uint64_t colsB = 128 + (rand() % 1500);
        uint64_t rowsC;
        uint64_t colsC;

        // Now make the operation legal
        if (!transposeA && !transposeB)
            rowsB = colsA;
        if (!transposeA && transposeB)
            colsB = colsA;
        if (transposeA && !transposeB)
            rowsB = rowsA;
        if (transposeA && transposeB)
            colsB = rowsA;
        if (transposeC) {
            rowsC = transposeB ? rowsB : colsB;
            colsC = transposeA ? colsA : rowsA;
        } else {
            rowsC = transposeA ? colsA : rowsA;
            colsC = transposeB ? rowsB : colsB;
        }
        uint64_t rowsD = transposeA ? colsA : rowsA;
        uint64_t colsD = transposeB ? rowsB : colsB;

        uint64_t ldA = colsA;
        uint64_t ldB = colsB;
        uint64_t ldC = colsC;
        uint64_t ldD = colsD;
        if (rand() % 2) {
            ldA += rand() % 10;
            ldB += rand() % 10;
            ldC += rand() % 10;
            if (CDInPlace)
                ldD = ldC;
            else
                ldD += rand() % 10;
        }

        TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU, 0);
        TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

        TensorDescriptor ADescriptor(TensorDescriptor::DataType::FP32, {rowsA, ldA});
        TensorDescriptor BDescriptor(TensorDescriptor::DataType::FP32, {rowsB, ldB});
        TensorDescriptor CDescriptor(TensorDescriptor::DataType::FP32, {rowsC, ldC});
        TensorDescriptor DDescriptor(TensorDescriptor::DataType::FP32, {rowsD, ldD});

        Tensor A(cpuPlacement, ADescriptor);
        Tensor B(cpuPlacement, BDescriptor);
        Tensor C(cpuPlacement, CDescriptor);
        Tensor D;
        if (CDInPlace)
            D = C;
        else
            D = Tensor(cpuPlacement, DDescriptor);
        Tensor A_d(gpuPlacement, ADescriptor);
        Tensor B_d(gpuPlacement, BDescriptor);
        Tensor C_d(gpuPlacement, CDescriptor);
        Tensor D_d;
        if (CDInPlace)
            D_d = C_d;
        else
            D_d = Tensor(gpuPlacement, DDescriptor);
        Tensor D_gpu_h(cpuPlacement, DDescriptor);

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
        float *DMem;
        if (CDInPlace) {
            DMem = CMem;
        } else {
            DMem = (float *)D.getMemPtr();
            for (uint64_t row = 0; row < rowsD; ++row) {
                for (uint64_t col = 0; col < colsD; ++col) {
                    DMem[row * ldD + col] = ((rand() % 100) - 50) / 10.0f;
                }
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
        float alpha = (rand() % 1000) / 10.0f;
        float beta = (rand() % 1000) / 10.0f;

        verifyOperationIsLegal(rowsA,
                               colsA,
                               rowsB,
                               colsB,
                               rowsC,
                               colsC,
                               ldA,
                               ldB,
                               ldC,
                               ldD,
                               transposeA,
                               transposeB,
                               transposeC,
                               CDInPlace,
                               C.getMemPtr(),
                               D.getMemPtr(),
                               C_d.getMemPtr(),
                               D_d.getMemPtr());

        A_d.copyFromAsync(A, stream);
        B_d.copyFromAsync(B, stream);
        C_d.copyFromAsync(C, stream);
        if (!CDInPlace)
            D_d.copyFromAsync(D, stream);
        stream.synchronize();

        std::thread cpuWorker(gemmCpuFp32,
                              (float *)A.getMemPtr(),
                              (float *)B.getMemPtr(),
                              (float *)C.getMemPtr(),
                              (float *)D.getMemPtr(),
                              rowsA,
                              colsA,
                              rowsB,
                              colsB,
                              ldA,
                              ldB,
                              ldC,
                              ldD,
                              transposeA,
                              transposeB,
                              transposeC,
                              alpha,
                              beta);

        CublasMatrixMultiply::instance().gemmUsingHeuristicKernelChoice(A_d,
                                                                        B_d,
                                                                        C_d,
                                                                        D_d,
                                                                        rowsA,
                                                                        colsA,
                                                                        rowsB,
                                                                        colsB,
                                                                        transposeA,
                                                                        transposeB,
                                                                        transposeC,
                                                                        &alpha,
                                                                        &beta,
                                                                        TensorDescriptor::DataType::FP32,
                                                                        stream);

        D_gpu_h.copyFromAsync(D_d, stream);
        stream.synchronize();

        cpuWorker.join();

        float maxDiff = transposeA == false ? colsA * 0.0005 : rowsA * 0.0005;

        float *DMemGpu = (float *)D_gpu_h.getMemPtr();

        // printf("\n\nCPU C:\n");
        // printMatrix(CMem, rowsA, colsB, ldC);
        // printf("\n\nGPU C:\n");
        // printMatrix(CMemGpu, rowsA, colsB, ldC);

        for (uint64_t i = 0; i < rowsC; ++i) {
            for (uint64_t j = 0; j < colsC; ++j) {
                float diff = abs(DMem[i * ldD + j] - DMemGpu[i * ldD + j]);

                if (diff >= maxDiff) {
                    printf("arows %ld acols %ld brows %ld bcols %ld transposeA %i transposeB %i\n",
                           rowsA,
                           colsA,
                           rowsB,
                           colsB,
                           transposeA,
                           transposeB);
                    printf("row %ld col %ld : CPU %f vs %f GPU\n", i, j, DMem[i * ldD + j], DMemGpu[i * ldD + j]);
                    fflush(stdout);
                }
                ASSERT_LT(diff, maxDiff);
            }
        }
    }
}

TEST(CublasMatrixMultiply, HeuristicGemmKernelWorksFP16) {
    srand(time(nullptr));

    ScopedGpu scopedGpu(0);
    Stream stream(0);

    for (uint64_t i = 0; i < 1; ++i) {
        bool transposeA = rand() % 2;
        bool transposeB = rand() % 2;
        bool transposeC = rand() % 2;
        // FIXME: TEMP
        transposeC = false;
        bool CDInPlace = rand() % 2;
        if (CDInPlace)
            transposeC = false;

        uint64_t rowsA = 128 + (rand() % 1500);
        uint64_t colsA = 128 + (rand() % 1500);
        uint64_t rowsB = 128 + (rand() % 1500);
        uint64_t colsB = 128 + (rand() % 1500);
        uint64_t rowsC;
        uint64_t colsC;

        // Now make the operation legal
        if (!transposeA && !transposeB)
            rowsB = colsA;
        if (!transposeA && transposeB)
            colsB = colsA;
        if (transposeA && !transposeB)
            rowsB = rowsA;
        if (transposeA && transposeB)
            colsB = rowsA;
        if (transposeC) {
            rowsC = transposeB ? rowsB : colsB;
            colsC = transposeA ? colsA : rowsA;
        } else {
            rowsC = transposeA ? colsA : rowsA;
            colsC = transposeB ? rowsB : colsB;
        }
        uint64_t rowsD = transposeA ? colsA : rowsA;
        uint64_t colsD = transposeB ? rowsB : colsB;

        uint64_t ldA = colsA;
        uint64_t ldB = colsB;
        uint64_t ldC = colsC;
        uint64_t ldD = colsD;
        if (rand() % 2) {
            ldA += rand() % 10;
            ldB += rand() % 10;
            ldC += rand() % 10;
            if (CDInPlace)
                ldD = ldC;
            else
                ldD += rand() % 10;
        }

        TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU, 0);
        TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

        TensorDescriptor ADescriptor(TensorDescriptor::DataType::FP16, {rowsA, ldA});
        TensorDescriptor BDescriptor(TensorDescriptor::DataType::FP16, {rowsB, ldB});
        TensorDescriptor CDescriptor(TensorDescriptor::DataType::FP16, {rowsC, ldC});
        TensorDescriptor DDescriptor(TensorDescriptor::DataType::FP16, {rowsD, ldD});

        Tensor A(cpuPlacement, ADescriptor);
        Tensor B(cpuPlacement, BDescriptor);
        Tensor C(cpuPlacement, CDescriptor);
        Tensor D;
        if (CDInPlace)
            D = C;
        else
            D = Tensor(cpuPlacement, DDescriptor);
        Tensor A_d(gpuPlacement, ADescriptor);
        Tensor B_d(gpuPlacement, BDescriptor);
        Tensor C_d(gpuPlacement, CDescriptor);
        Tensor D_d;
        if (CDInPlace)
            D_d = C_d;
        else
            D_d = Tensor(gpuPlacement, DDescriptor);
        Tensor D_gpu_h(cpuPlacement, DDescriptor);

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
        half *DMem;
        if (CDInPlace) {
            DMem = CMem;
        } else {
            DMem = (half *)D.getMemPtr();
            for (uint64_t row = 0; row < rowsD; ++row) {
                for (uint64_t col = 0; col < colsD; ++col) {
                    DMem[row * ldD + col] = ((rand() % 100) - 50) / 10.0f;
                }
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
        float alpha = (rand() % 100) / 10.0f;
        float beta = (rand() % 100) / 10.0f;

        verifyOperationIsLegal(rowsA,
                               colsA,
                               rowsB,
                               colsB,
                               rowsC,
                               colsC,
                               ldA,
                               ldB,
                               ldC,
                               ldD,
                               transposeA,
                               transposeB,
                               transposeC,
                               CDInPlace,
                               C.getMemPtr(),
                               D.getMemPtr(),
                               C_d.getMemPtr(),
                               D_d.getMemPtr());

        A_d.copyFromAsync(A, stream);
        B_d.copyFromAsync(B, stream);
        C_d.copyFromAsync(C, stream);
        if (!CDInPlace)
            D_d.copyFromAsync(D, stream);
        stream.synchronize();

        std::thread cpuWorker(gemmCpuFp16,
                              (half *)A.getMemPtr(),
                              (half *)B.getMemPtr(),
                              (half *)C.getMemPtr(),
                              (half *)D.getMemPtr(),
                              rowsA,
                              colsA,
                              rowsB,
                              colsB,
                              ldA,
                              ldB,
                              ldC,
                              ldD,
                              transposeA,
                              transposeB,
                              transposeC,
                              alpha,
                              beta);

        CublasMatrixMultiply::instance().gemmUsingHeuristicKernelChoice(A_d,
                                                                        B_d,
                                                                        C_d,
                                                                        D_d,
                                                                        rowsA,
                                                                        colsA,
                                                                        rowsB,
                                                                        colsB,
                                                                        transposeA,
                                                                        transposeB,
                                                                        transposeC,
                                                                        &alpha,
                                                                        &beta,
                                                                        TensorDescriptor::DataType::FP16,
                                                                        stream);

        D_gpu_h.copyFromAsync(D_d, stream);
        stream.synchronize();

        cpuWorker.join();

        float maxDiff = transposeA == false ? colsA * 0.05f : rowsA * 0.05f;

        half *DMemGpu = (half *)D_gpu_h.getMemPtr();

        // printf("\n\nCPU C:\n");
        // printMatrix(CMem, rowsA, colsB, ldC);
        // printf("\n\nGPU C:\n");
        // printMatrix(CMemGpu, rowsA, colsB, ldC);

        for (uint64_t i = 0; i < rowsC; ++i) {
            for (uint64_t j = 0; j < colsC; ++j) {
                float diff = abs((float)(DMem[i * ldD + j] - DMemGpu[i * ldD + j]));

                if (!(diff < maxDiff)) {
                    printf("arows %ld acols %ld brows %ld bcols %ld transposeA %i transposeB %i\n",
                           rowsA,
                           colsA,
                           rowsB,
                           colsB,
                           transposeA,
                           transposeB);
                    printf("row %ld col %ld : CPU %f vs %f GPU\n", i, j, (float)DMem[i * ldD + j], (float)DMemGpu[i * ldD + j]);
                    fflush(stdout);
                }
                ASSERT_LT(diff, maxDiff);
            }
        }
    }
}

namespace {

using CublasMatmulDTypes = CublasMatrixMultiply::MatmulDataTypes;
using DataType = TensorDescriptor::DataType;

bool gpuComputeCapabilityAtLeast(int gpuNum, int requiredMajor, int requiredMinor) {
    cudaDeviceProp prop;
    cudaError_t status = cudaGetDeviceProperties(&prop, gpuNum);
    if (status != cudaSuccess) {
        return false;
    }
    if (prop.major != requiredMajor) {
        return prop.major > requiredMajor;
    }
    return prop.minor >= requiredMinor;
}

bool gpuSupportsBf16TensorCores(int gpuNum) {
    // BF16 tensor core GEMM kernels are Ampere+.
    return gpuComputeCapabilityAtLeast(gpuNum, 8, 0);
}

bool gpuSupportsFp8TensorCores(int gpuNum) {
    // FP8 tensor core GEMM kernels are Hopper/Ada+ in practice. This keeps the
    // runtime tests from failing on older CI GPUs that still compile the FP8 types.
    cudaDeviceProp prop;
    cudaError_t status = cudaGetDeviceProperties(&prop, gpuNum);
    if (status != cudaSuccess) {
        return false;
    }
    return prop.major >= 9 || (prop.major == 8 && prop.minor >= 9);
}

template <typename T>
T testValueFromFloat(float value);

template <>
float testValueFromFloat<float>(float value) {
    return value;
}

template <>
half testValueFromFloat<half>(float value) {
    return __float2half(value);
}

template <>
__nv_bfloat16 testValueFromFloat<__nv_bfloat16>(float value) {
    return __float2bfloat16(value);
}

template <>
__nv_fp8_e4m3 testValueFromFloat<__nv_fp8_e4m3>(float value) {
    return __nv_fp8_e4m3(value);
}

template <>
__nv_fp8_e5m2 testValueFromFloat<__nv_fp8_e5m2>(float value) {
    return __nv_fp8_e5m2(value);
}

template <typename T>
float testValueToFloat(T value) {
    return static_cast<float>(value);
}

template <>
float testValueToFloat<float>(float value) {
    return value;
}

template <>
float testValueToFloat<half>(half value) {
    return __half2float(value);
}

template <>
float testValueToFloat<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename T>
void fillPaddedMatrix(Tensor &tensor, int rows, int cols, int ld, float rowScale, float colScale, float offset) {
    T *mem = reinterpret_cast<T *>(tensor.getMemPtr());
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < ld; ++col) {
            float value = 0.0f;
            if (col < cols) {
                value = offset + rowScale * static_cast<float>((row % 7) - 3) + colScale * static_cast<float>((col % 5) - 2);
            }
            mem[row * ld + col] = testValueFromFloat<T>(value);
        }
    }
}

template <typename TA, typename TB, typename TC>
std::vector<float> referenceGemm(const Tensor &A,
                                 const Tensor &B,
                                 const Tensor &C,
                                 int rowsA,
                                 int colsA,
                                 int rowsB,
                                 int colsB,
                                 int ldA,
                                 int ldB,
                                 int ldC,
                                 bool transposeA,
                                 bool transposeB,
                                 float alpha,
                                 float beta) {
    const TA *a = reinterpret_cast<const TA *>(A.getMemPtr());
    const TB *b = reinterpret_cast<const TB *>(B.getMemPtr());
    const TC *c = reinterpret_cast<const TC *>(C.getMemPtr());

    const int rowsD = transposeA ? colsA : rowsA;
    const int colsD = transposeB ? rowsB : colsB;
    const int inner = transposeA ? rowsA : colsA;

    std::vector<float> expected(static_cast<size_t>(rowsD) * static_cast<size_t>(colsD), 0.0f);

    for (int row = 0; row < rowsD; ++row) {
        for (int col = 0; col < colsD; ++col) {
            float accum = 0.0f;
            for (int k = 0; k < inner; ++k) {
                const float av = transposeA ? testValueToFloat(a[k * ldA + row]) : testValueToFloat(a[row * ldA + k]);
                const float bv = transposeB ? testValueToFloat(b[col * ldB + k]) : testValueToFloat(b[k * ldB + col]);
                accum += av * bv;
            }
            expected[static_cast<size_t>(row) * static_cast<size_t>(colsD) + static_cast<size_t>(col)] =
                alpha * accum + beta * testValueToFloat(c[row * ldC + col]);
        }
    }

    return expected;
}

void assertFp32MatrixClose(const Tensor &D, const std::vector<float> &expected, int rows, int cols, int ld, float tolerance) {
    const float *d = reinterpret_cast<const float *>(D.getMemPtr());
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            const float got = d[row * ld + col];
            const float want = expected[static_cast<size_t>(row) * static_cast<size_t>(cols) + static_cast<size_t>(col)];
            ASSERT_NEAR(got, want, tolerance) << "row=" << row << " col=" << col;
        }
    }
}

}  // namespace

TEST(CublasMatrixMultiply, OperationTypeTableIncludesExpandedFloatAndFp8Combinations) {
    EXPECT_TRUE(isSupportedCublasLtOperationType(CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F));

    EXPECT_TRUE(isSupportedCublasLtOperationType(CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F));
    EXPECT_TRUE(isSupportedCublasLtOperationType(CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F, CUDA_R_32F));
    EXPECT_TRUE(isSupportedCublasLtOperationType(CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF));
    EXPECT_TRUE(isSupportedCublasLtOperationType(CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_32F, CUDA_R_32F));

    EXPECT_FALSE(isSupportedCublasLtOperationType(CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, CUDA_R_32F, CUDA_R_32F));
    EXPECT_TRUE(
        isSupportedCublasLtOperationType(CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_8F_E4M3, CUDA_R_8F_E5M2, CUDA_R_16BF, CUDA_R_8F_E4M3));
    EXPECT_TRUE(
        isSupportedCublasLtOperationType(CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_8F_E5M2, CUDA_R_8F_E4M3, CUDA_R_16F, CUDA_R_8F_E5M2));

    EXPECT_FALSE(isSupportedCublasLtOperationType(CUBLAS_COMPUTE_64F, CUDA_R_64F, CUDA_R_64F, CUDA_R_64F, CUDA_R_64F, CUDA_R_64F));
    EXPECT_FALSE(isSupportedCublasLtOperationType(CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_8F_E5M2, CUDA_R_8F_E5M2, CUDA_R_16F, CUDA_R_16F));
    EXPECT_FALSE(isSupportedCublasLtOperationType(CUBLAS_COMPUTE_32F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F));
}

TEST(CublasMatrixMultiply, HeuristicGemmSupportsBf16InputsAndFp32Output) {
    constexpr int gpuNum = 0;
    if (!gpuSupportsBf16TensorCores(gpuNum)) {
        GTEST_SKIP() << "BF16 GEMM kernels require an Ampere-or-newer GPU.";
    }

    ScopedGpu scopedGpu(gpuNum);
    Stream stream(gpuNum);

    constexpr int m = 32;
    constexpr int n = 24;
    constexpr int k = 40;
    constexpr int rowsA = m;
    constexpr int colsA = k;
    constexpr int rowsB = k;
    constexpr int colsB = n;
    constexpr int ldA = colsA;
    constexpr int ldB = colsB;
    constexpr int ldC = colsB;
    constexpr int ldD = colsB;
    constexpr bool transposeA = false;
    constexpr bool transposeB = false;
    constexpr bool transposeC = false;

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU, 0);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, gpuNum);

    TensorDescriptor ADescriptor(DataType::BF16, {static_cast<uint64_t>(rowsA), static_cast<uint64_t>(ldA)});
    TensorDescriptor BDescriptor(DataType::BF16, {static_cast<uint64_t>(rowsB), static_cast<uint64_t>(ldB)});
    TensorDescriptor CDescriptor(DataType::FP32, {static_cast<uint64_t>(m), static_cast<uint64_t>(ldC)});
    TensorDescriptor DDescriptor(DataType::FP32, {static_cast<uint64_t>(m), static_cast<uint64_t>(ldD)});

    Tensor A(cpuPlacement, ADescriptor);
    Tensor B(cpuPlacement, BDescriptor);
    Tensor C(cpuPlacement, CDescriptor);
    Tensor D(cpuPlacement, DDescriptor);
    Tensor A_d(gpuPlacement, ADescriptor);
    Tensor B_d(gpuPlacement, BDescriptor);
    Tensor C_d(gpuPlacement, CDescriptor);
    Tensor D_d(gpuPlacement, DDescriptor);
    Tensor D_h(cpuPlacement, DDescriptor);

    fillPaddedMatrix<__nv_bfloat16>(A, rowsA, colsA, ldA, 0.125f, -0.0625f, 0.25f);
    fillPaddedMatrix<__nv_bfloat16>(B, rowsB, colsB, ldB, -0.09375f, 0.15625f, -0.125f);
    fillPaddedMatrix<float>(C, m, n, ldC, 0.03125f, -0.046875f, 0.5f);
    fillPaddedMatrix<float>(D, m, n, ldD, 0.0f, 0.0f, -999.0f);

    constexpr float alpha = 0.75f;
    constexpr float beta = -1.25f;
    std::vector<float> expected = referenceGemm<__nv_bfloat16, __nv_bfloat16, float>(
        A, B, C, rowsA, colsA, rowsB, colsB, ldA, ldB, ldC, transposeA, transposeB, alpha, beta);

    A_d.copyFromAsync(A, stream);
    B_d.copyFromAsync(B, stream);
    C_d.copyFromAsync(C, stream);
    D_d.copyFromAsync(D, stream);
    stream.synchronize();

    CublasMatrixMultiply::instance().gemmUsingHeuristicKernelChoice(
        A_d,
        B_d,
        C_d,
        D_d,
        rowsA,
        colsA,
        rowsB,
        colsB,
        transposeA,
        transposeB,
        transposeC,
        &alpha,
        &beta,
        CublasMatmulDTypes{DataType::BF16, DataType::BF16, DataType::FP32, DataType::FP32},
        stream);

    D_h.copyFromAsync(D_d, stream);
    stream.synchronize();

    assertFp32MatrixClose(D_h, expected, m, n, ldD, 0.20f);
}

TEST(CublasMatrixMultiply, HeuristicGemmRejectsFp8InputsAndFp32OutputInTNLayoutWithFirstClassError) {
    constexpr int gpuNum = 0;
    if (!gpuSupportsFp8TensorCores(gpuNum)) {
        GTEST_SKIP() << "FP8 tensors require an FP8-capable GPU for this runtime validation test.";
    }

    ScopedGpu scopedGpu(gpuNum);
    Stream stream(gpuNum);

    constexpr int m = 16;
    constexpr int n = 16;
    constexpr int k = 16;
    constexpr int rowsA = k;
    constexpr int colsA = m;
    constexpr int rowsB = k;
    constexpr int colsB = n;
    constexpr int ldA = colsA;
    constexpr int ldB = colsB;
    constexpr int ldC = colsB;
    constexpr int ldD = colsB;
    constexpr bool transposeA = true;
    constexpr bool transposeB = false;
    constexpr bool transposeC = false;

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU, 0);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, gpuNum);

    TensorDescriptor ADescriptor(DataType::FP8_E4M3, {static_cast<uint64_t>(rowsA), static_cast<uint64_t>(ldA)});
    TensorDescriptor BDescriptor(DataType::FP8_E5M2, {static_cast<uint64_t>(rowsB), static_cast<uint64_t>(ldB)});
    TensorDescriptor CDescriptor(DataType::FP32, {static_cast<uint64_t>(m), static_cast<uint64_t>(ldC)});
    TensorDescriptor DDescriptor(DataType::FP32, {static_cast<uint64_t>(m), static_cast<uint64_t>(ldD)});

    Tensor A(cpuPlacement, ADescriptor);
    Tensor B(cpuPlacement, BDescriptor);
    Tensor C(cpuPlacement, CDescriptor);
    Tensor D(cpuPlacement, DDescriptor);
    Tensor A_d(gpuPlacement, ADescriptor);
    Tensor B_d(gpuPlacement, BDescriptor);
    Tensor C_d(gpuPlacement, CDescriptor);
    Tensor D_d(gpuPlacement, DDescriptor);

    fillPaddedMatrix<__nv_fp8_e4m3>(A, rowsA, colsA, ldA, 0.125f, -0.125f, 0.0f);
    fillPaddedMatrix<__nv_fp8_e5m2>(B, rowsB, colsB, ldB, -0.125f, 0.0625f, 0.25f);
    fillPaddedMatrix<float>(C, m, n, ldC, 0.015625f, -0.03125f, 0.25f);
    fillPaddedMatrix<float>(D, m, n, ldD, 0.0f, 0.0f, -999.0f);

    A_d.copyFromAsync(A, stream);
    B_d.copyFromAsync(B, stream);
    C_d.copyFromAsync(C, stream);
    D_d.copyFromAsync(D, stream);
    stream.synchronize();

    constexpr float alpha = 1.0f;
    constexpr float beta = 0.5f;

    try {
        CublasMatrixMultiply::instance().gemmUsingHeuristicKernelChoice(
            A_d,
            B_d,
            C_d,
            D_d,
            rowsA,
            colsA,
            rowsB,
            colsB,
            transposeA,
            transposeB,
            transposeC,
            &alpha,
            &beta,
            CublasMatmulDTypes{DataType::FP8_E4M3, DataType::FP8_E5M2, DataType::FP32, DataType::FP32},
            stream);
        FAIL() << "Expected FP8-input / FP32-output GEMM to be rejected before cuBLASLt heuristic lookup.";
    } catch (const std::invalid_argument &e) {
        const std::string message(e.what());
        EXPECT_NE(message.find("FP8 input GEMM with FP32 C/D output"), std::string::npos) << message;
        EXPECT_NE(message.find("{A=FP8_E4M3, B=FP8_E5M2, C=FP32, D=FP32}"), std::string::npos) << message;
        EXPECT_NE(message.find("Cast the FP8 inputs to FP16 or BF16"), std::string::npos) << message;
    }
}
