#include "Utilities/TensorOperations/GpuMatrixMultiply/CublasMatrixMultiply.h"

#include "test/Utilities/TensorOperations/GpuMatrixMultiply/MatrixMultiplyTestHelper.h"

#include "gtest/gtest.h"
#include "omp.h"

#include <cmath>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "Utilities/Expression/CudaHelpers.h"

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
                                 float beta,
                                 float aScale = 1.0f,
                                 float bScale = 1.0f,
                                 float cScale = 1.0f) {
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
                alpha * aScale * bScale * accum + beta * cScale * testValueToFloat(c[row * ldC + col]);
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

bool isCublasLtHeuristicAvailabilityError(const std::exception &e) {
    const std::string message(e.what());
    return message.find("CUBLAS_STATUS_NOT_SUPPORTED") != std::string::npos ||
           message.find("CUBLAS_STATUS_ARCH_MISMATCH") != std::string::npos ||
           message.find("CUBLAS_STATUS_NOT_INITIALIZED") != std::string::npos;
}

struct ScopedDeviceFp8MatmulScales {
    float *aScale = nullptr;
    float *bScale = nullptr;
    float *cScale = nullptr;
    float *dScale = nullptr;
    float *dAmax = nullptr;

    ScopedDeviceFp8MatmulScales(float aScaleValue, float bScaleValue, float cScaleValue = 1.0f, float dScaleValue = 1.0f) {
        allocateAndCopy(aScale, aScaleValue);
        allocateAndCopy(bScale, bScaleValue);
        allocateAndCopy(cScale, cScaleValue);
        allocateAndCopy(dScale, dScaleValue);
        checkCudaErrors(cudaMalloc(&dAmax, sizeof(float)));
        checkCudaErrors(cudaMemset(dAmax, 0, sizeof(float)));
    }

    ~ScopedDeviceFp8MatmulScales() {
        if (aScale != nullptr) {
            cudaFree(aScale);
        }
        if (bScale != nullptr) {
            cudaFree(bScale);
        }
        if (cScale != nullptr) {
            cudaFree(cScale);
        }
        if (dScale != nullptr) {
            cudaFree(dScale);
        }
        if (dAmax != nullptr) {
            cudaFree(dAmax);
        }
    }

    CublasMatrixMultiply::Fp8MatmulScales inputScales() const { return CublasMatrixMultiply::Fp8MatmulScales::tensorwide(aScale, bScale); }

    CublasMatrixMultiply::Fp8MatmulScales inputAndCScales() const {
        return CublasMatrixMultiply::Fp8MatmulScales::tensorwide(aScale, bScale, cScale);
    }

    CublasMatrixMultiply::Fp8MatmulScales allScales() const {
        return CublasMatrixMultiply::Fp8MatmulScales::tensorwide(aScale, bScale, cScale, dScale, dAmax);
    }

   private:
    static void allocateAndCopy(float *&devicePointer, float value) {
        checkCudaErrors(cudaMalloc(&devicePointer, sizeof(float)));
        checkCudaErrors(cudaMemcpy(devicePointer, &value, sizeof(float), cudaMemcpyHostToDevice));
    }
};

}  // namespace

TEST(CublasMatrixMultiply, OperationTypeTableIncludesExpandedFloatAndFp8Combinations) {
    EXPECT_TRUE(isSupportedCublasLtOperationType(CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F));

    EXPECT_TRUE(isSupportedCublasLtOperationType(CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F));
    EXPECT_TRUE(isSupportedCublasLtOperationType(CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F, CUDA_R_32F));
    EXPECT_TRUE(isSupportedCublasLtOperationType(CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF));
    EXPECT_TRUE(isSupportedCublasLtOperationType(CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_32F, CUDA_R_32F));

    EXPECT_TRUE(isSupportedCublasLtOperationType(CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, CUDA_R_32F, CUDA_R_32F));
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

TEST(CublasMatrixMultiply, HeuristicGemmRejectsFp8InputsAndFp32OutputInTNLayoutWithoutExplicitScales) {
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
        FAIL() << "Expected FP8-input / FP32-output GEMM to require explicit FP8 scale pointers.";
    } catch (const std::invalid_argument &e) {
        const std::string message(e.what());
        EXPECT_NE(message.find("FP8 GEMM scale configuration is incomplete"), std::string::npos) << message;
        EXPECT_NE(message.find("{A=FP8_E4M3, B=FP8_E5M2, C=FP32, D=FP32}"), std::string::npos) << message;
        EXPECT_NE(message.find("A_SCALE_POINTER"), std::string::npos) << message;
        EXPECT_NE(message.find("B_SCALE_POINTER"), std::string::npos) << message;
    }
}

TEST(CublasMatrixMultiply, ChooseOptimalGemmSupportsFp8InputsAndFp32OutputInTNLayoutWithExplicitTensorwideScalesAndWorkspace) {
    constexpr int gpuNum = 0;
    if (!gpuSupportsFp8TensorCores(gpuNum)) {
        GTEST_SKIP() << "FP8 GEMM kernels require an FP8-capable GPU.";
    }

    ScopedGpu scopedGpu(gpuNum);
    Stream stream(gpuNum);

    constexpr int m = 128;
    constexpr int n = 128;
    constexpr int k = 128;
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
    Tensor D_h(cpuPlacement, DDescriptor);

    fillPaddedMatrix<__nv_fp8_e4m3>(A, rowsA, colsA, ldA, 0.0625f, -0.03125f, 0.125f);
    fillPaddedMatrix<__nv_fp8_e5m2>(B, rowsB, colsB, ldB, -0.03125f, 0.0625f, -0.0625f);
    fillPaddedMatrix<float>(C, m, n, ldC, 0.015625f, -0.03125f, 0.25f);
    fillPaddedMatrix<float>(D, m, n, ldD, 0.0f, 0.0f, -999.0f);

    constexpr float alpha = 0.75f;
    constexpr float beta = -0.5f;
    constexpr float aScale = 0.5f;
    constexpr float bScale = 0.25f;
    constexpr float cScale = 2.0f;

    std::vector<float> expected = referenceGemm<__nv_fp8_e4m3, __nv_fp8_e5m2, float>(
        A, B, C, rowsA, colsA, rowsB, colsB, ldA, ldB, ldC, transposeA, transposeB, alpha, beta, aScale, bScale, cScale);

    A_d.copyFromAsync(A, stream);
    B_d.copyFromAsync(B, stream);
    C_d.copyFromAsync(C, stream);
    D_d.copyFromAsync(D, stream);
    stream.synchronize();

    ScopedDeviceFp8MatmulScales scales(aScale, bScale, cScale);
    const CublasMatmulDTypes dataTypes{DataType::FP8_E4M3, DataType::FP8_E5M2, DataType::FP32, DataType::FP32};

    try {
        CublasMatrixMultiply::instance().chooseOptimalGemmKernel(gpuNum,
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
                                                                 dataTypes,
                                                                 scales.inputAndCScales(),
                                                                 false);
    } catch (const std::runtime_error &e) {
        if (isCublasLtHeuristicAvailabilityError(e) ||
            std::string(e.what()).find("could not find any cuBLASLt kernel candidates") != std::string::npos) {
            GTEST_SKIP() << "cuBLASLt did not expose a usable workspace-backed FP8 scaled kernel for this descriptor: " << e.what();
        }
        throw;
    }

    bool kernelWillRunOnGpu = false;
    const uint64_t workspaceSizeInBytes = CublasMatrixMultiply::instance().getGemmWorkspaceSizeInBytes(gpuNum,
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
                                                                                                       dataTypes,
                                                                                                       scales.inputAndCScales(),
                                                                                                       kernelWillRunOnGpu);
    ASSERT_TRUE(kernelWillRunOnGpu);

    Optional<Tensor> workspace_d;
    if (workspaceSizeInBytes > 0) {
        workspace_d = Tensor(gpuPlacement, TensorDescriptor(DataType::UINT8, {workspaceSizeInBytes}));
    }

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
                                          dataTypes,
                                          scales.inputAndCScales(),
                                          stream);

    D_h.copyFromAsync(D_d, stream);
    stream.synchronize();

    assertFp32MatrixClose(D_h, expected, m, n, ldD, 1.0f);
}

namespace {

template <typename T>
TensorDescriptor::DataType surfaceThorDataType();

template <>
TensorDescriptor::DataType surfaceThorDataType<float>() {
    return TensorDescriptor::DataType::FP32;
}

template <>
TensorDescriptor::DataType surfaceThorDataType<half>() {
    return TensorDescriptor::DataType::FP16;
}

template <>
TensorDescriptor::DataType surfaceThorDataType<__nv_bfloat16>() {
    return TensorDescriptor::DataType::BF16;
}

template <>
TensorDescriptor::DataType surfaceThorDataType<int8_t>() {
    return TensorDescriptor::DataType::INT8;
}

template <typename T>
T surfaceValueFromFloat(float value);

template <>
float surfaceValueFromFloat<float>(float value) {
    return value;
}

template <>
half surfaceValueFromFloat<half>(float value) {
    return __float2half(value);
}

template <>
__nv_bfloat16 surfaceValueFromFloat<__nv_bfloat16>(float value) {
    return __float2bfloat16(value);
}

template <>
int8_t surfaceValueFromFloat<int8_t>(float value) {
    value = std::round(value);
    value = std::max(-127.0f, std::min(127.0f, value));
    return static_cast<int8_t>(value);
}

template <typename T>
float surfaceValueToFloat(T value) {
    return static_cast<float>(value);
}

template <>
float surfaceValueToFloat<float>(float value) {
    return value;
}

template <>
float surfaceValueToFloat<half>(half value) {
    return __half2float(value);
}

template <>
float surfaceValueToFloat<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <>
float surfaceValueToFloat<int8_t>(int8_t value) {
    return static_cast<float>(value);
}

template <typename T>
void fillSurfacePaddedMatrix(Tensor &tensor, int rows, int cols, int ld, float rowScale, float colScale, float offset) {
    T *mem = reinterpret_cast<T *>(tensor.getMemPtr());
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < ld; ++col) {
            float value = 0.0f;
            if (col < cols) {
                value = offset + rowScale * static_cast<float>((row % 7) - 3) + colScale * static_cast<float>((col % 5) - 2);
            }
            mem[row * ld + col] = surfaceValueFromFloat<T>(value);
        }
    }
}

template <typename TA, typename TB, typename TC>
std::vector<float> referenceSurfaceGemm(const Tensor &A,
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
                const float av = transposeA ? surfaceValueToFloat(a[k * ldA + row]) : surfaceValueToFloat(a[row * ldA + k]);
                const float bv = transposeB ? surfaceValueToFloat(b[col * ldB + k]) : surfaceValueToFloat(b[k * ldB + col]);
                accum += av * bv;
            }

            expected[static_cast<size_t>(row) * static_cast<size_t>(colsD) + static_cast<size_t>(col)] =
                alpha * accum + beta * surfaceValueToFloat(c[row * ldC + col]);
        }
    }

    return expected;
}

template <typename TD>
void assertSurfaceMatrixClose(const Tensor &D, const std::vector<float> &expected, int rows, int cols, int ld, float tolerance) {
    const TD *d = reinterpret_cast<const TD *>(D.getMemPtr());

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            const float got = surfaceValueToFloat(d[row * ld + col]);
            const float want = expected[static_cast<size_t>(row) * static_cast<size_t>(cols) + static_cast<size_t>(col)];
            ASSERT_NEAR(got, want, tolerance) << "row=" << row << " col=" << col;
        }
    }
}

bool isCublasLtSurfaceAvailabilityError(const std::exception &e) {
    const std::string message(e.what());
    return message.find("CUBLAS_STATUS_NOT_SUPPORTED") != std::string::npos ||
           message.find("CUBLAS_STATUS_ARCH_MISMATCH") != std::string::npos ||
           message.find("CUBLAS_STATUS_NOT_INITIALIZED") != std::string::npos;
}

template <typename TA, typename TB, typename TD>
void runHeuristicMatmulSurfaceCase(
    int gpuNum, int rowsA, int colsA, int rowsB, int colsB, bool transposeA, bool transposeB, float tolerance) {
    ScopedGpu scopedGpu(gpuNum);
    Stream stream(gpuNum);

    const int rowsD = transposeA ? colsA : rowsA;
    const int colsD = transposeB ? rowsB : colsB;
    const int ldA = colsA;
    const int ldB = colsB;
    const int ldD = colsD;

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU, 0);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, gpuNum);

    TensorDescriptor ADescriptor(surfaceThorDataType<TA>(), {static_cast<uint64_t>(rowsA), static_cast<uint64_t>(ldA)});
    TensorDescriptor BDescriptor(surfaceThorDataType<TB>(), {static_cast<uint64_t>(rowsB), static_cast<uint64_t>(ldB)});
    TensorDescriptor DDescriptor(surfaceThorDataType<TD>(), {static_cast<uint64_t>(rowsD), static_cast<uint64_t>(ldD)});

    Tensor A(cpuPlacement, ADescriptor);
    Tensor B(cpuPlacement, BDescriptor);
    Tensor D(cpuPlacement, DDescriptor);
    Tensor A_d(gpuPlacement, ADescriptor);
    Tensor B_d(gpuPlacement, BDescriptor);
    Tensor D_d(gpuPlacement, DDescriptor);
    Tensor D_h(cpuPlacement, DDescriptor);

    fillSurfacePaddedMatrix<TA>(A, rowsA, colsA, ldA, 0.125f, -0.0625f, 0.25f);
    fillSurfacePaddedMatrix<TB>(B, rowsB, colsB, ldB, -0.09375f, 0.15625f, -0.125f);
    fillSurfacePaddedMatrix<TD>(D, rowsD, colsD, ldD, 0.0f, 0.0f, -7.0f);

    const std::vector<float> expected =
        referenceSurfaceGemm<TA, TB, TD>(A, B, D, rowsA, colsA, rowsB, colsB, ldA, ldB, ldD, transposeA, transposeB, 1.0f, 0.0f);

    A_d.copyFromAsync(A, stream);
    B_d.copyFromAsync(B, stream);
    D_d.copyFromAsync(D, stream);
    stream.synchronize();

    CublasMatrixMultiply::instance().multiplyUsingHeuristicKernelChoice(
        A_d,
        B_d,
        D_d,
        rowsA,
        colsA,
        rowsB,
        colsB,
        transposeA,
        transposeB,
        false,
        false,
        CublasMatrixMultiply::MatmulDataTypes{
            surfaceThorDataType<TA>(), surfaceThorDataType<TB>(), surfaceThorDataType<TD>(), surfaceThorDataType<TD>()},
        stream);

    D_h.copyFromAsync(D_d, stream);
    stream.synchronize();

    assertSurfaceMatrixClose<TD>(D_h, expected, rowsD, colsD, ldD, tolerance);
}

template <typename TA, typename TB, typename TC, typename TD>
void runHeuristicGemmSurfaceCase(
    int gpuNum, int rowsA, int colsA, int rowsB, int colsB, bool transposeA, bool transposeB, float alpha, float beta, float tolerance) {
    ScopedGpu scopedGpu(gpuNum);
    Stream stream(gpuNum);

    constexpr bool transposeC = false;

    const int rowsD = transposeA ? colsA : rowsA;
    const int colsD = transposeB ? rowsB : colsB;
    const int ldA = colsA;
    const int ldB = colsB;
    const int ldC = colsD;
    const int ldD = colsD;

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU, 0);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, gpuNum);

    TensorDescriptor ADescriptor(surfaceThorDataType<TA>(), {static_cast<uint64_t>(rowsA), static_cast<uint64_t>(ldA)});
    TensorDescriptor BDescriptor(surfaceThorDataType<TB>(), {static_cast<uint64_t>(rowsB), static_cast<uint64_t>(ldB)});
    TensorDescriptor CDescriptor(surfaceThorDataType<TC>(), {static_cast<uint64_t>(rowsD), static_cast<uint64_t>(ldC)});
    TensorDescriptor DDescriptor(surfaceThorDataType<TD>(), {static_cast<uint64_t>(rowsD), static_cast<uint64_t>(ldD)});

    Tensor A(cpuPlacement, ADescriptor);
    Tensor B(cpuPlacement, BDescriptor);
    Tensor C(cpuPlacement, CDescriptor);
    Tensor D(cpuPlacement, DDescriptor);
    Tensor A_d(gpuPlacement, ADescriptor);
    Tensor B_d(gpuPlacement, BDescriptor);
    Tensor C_d(gpuPlacement, CDescriptor);
    Tensor D_d(gpuPlacement, DDescriptor);
    Tensor D_h(cpuPlacement, DDescriptor);

    fillSurfacePaddedMatrix<TA>(A, rowsA, colsA, ldA, 0.125f, -0.0625f, 0.25f);
    fillSurfacePaddedMatrix<TB>(B, rowsB, colsB, ldB, -0.09375f, 0.15625f, -0.125f);
    fillSurfacePaddedMatrix<TC>(C, rowsD, colsD, ldC, 0.03125f, -0.046875f, 0.5f);
    fillSurfacePaddedMatrix<TD>(D, rowsD, colsD, ldD, 0.0f, 0.0f, -7.0f);

    const std::vector<float> expected =
        referenceSurfaceGemm<TA, TB, TC>(A, B, C, rowsA, colsA, rowsB, colsB, ldA, ldB, ldC, transposeA, transposeB, alpha, beta);

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
        CublasMatrixMultiply::MatmulDataTypes{
            surfaceThorDataType<TA>(), surfaceThorDataType<TB>(), surfaceThorDataType<TC>(), surfaceThorDataType<TD>()},
        stream);

    D_h.copyFromAsync(D_d, stream);
    stream.synchronize();

    assertSurfaceMatrixClose<TD>(D_h, expected, rowsD, colsD, ldD, tolerance);
}

}  // namespace

TEST(CublasMatrixMultiply, HeuristicMatmulSupportsFp16InputsAndFp32Output) {
    runHeuristicMatmulSurfaceCase<half, half, float>(0, 24, 32, 32, 20, false, false, 0.08f);
}

TEST(CublasMatrixMultiply, HeuristicMatmulSupportsBf16InputsAndFp32Output) {
    constexpr int gpuNum = 0;
    if (!gpuSupportsBf16TensorCores(gpuNum)) {
        GTEST_SKIP() << "BF16 GEMM kernels require an Ampere-or-newer GPU.";
    }

    runHeuristicMatmulSurfaceCase<__nv_bfloat16, __nv_bfloat16, float>(gpuNum, 24, 32, 32, 20, false, false, 0.25f);
}

TEST(CublasMatrixMultiply, HeuristicGemmSupportsFp16InputsAndFp32Output) {
    runHeuristicGemmSurfaceCase<half, half, float, float>(0, 24, 32, 32, 20, false, false, 0.75f, -1.25f, 0.10f);
}

TEST(CublasMatrixMultiply, HeuristicGemmSupportsBf16SameTypeCAndD) {
    constexpr int gpuNum = 0;
    if (!gpuSupportsBf16TensorCores(gpuNum)) {
        GTEST_SKIP() << "BF16 GEMM kernels require an Ampere-or-newer GPU.";
    }

    runHeuristicGemmSurfaceCase<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16>(
        gpuNum, 24, 32, 32, 20, false, false, 0.75f, -0.5f, 0.75f);
}

TEST(CublasMatrixMultiply, HeuristicGemmSupportsInt8InputsAndFp32OutputInTNLayout) {
    constexpr int gpuNum = 0;
    if (!gpuComputeCapabilityAtLeast(gpuNum, 7, 5)) {
        GTEST_SKIP() << "INT8 tensor core GEMM kernels require a Turing-or-newer GPU.";
    }

    // Regular-layout IMMA has stricter requirements than the floating-point path:
    // pointers aligned, leading dimensions multiples of 4, m/k multiples of 4, and TN layout.
    // Use square-ish multiples to avoid accidentally testing an unsupported descriptor shape.
    try {
        runHeuristicGemmSurfaceCase<int8_t, int8_t, float, float>(gpuNum,
                                                                  64,  // rowsA = k because transposeA=true
                                                                  32,  // colsA = m
                                                                  64,  // rowsB = k
                                                                  48,  // colsB = n
                                                                  true,
                                                                  false,
                                                                  1.0f,
                                                                  0.5f,
                                                                  1.0e-3f);
    } catch (const std::runtime_error &e) {
        if (isCublasLtSurfaceAvailabilityError(e)) {
            GTEST_SKIP() << "cuBLASLt did not expose a usable INT8->FP32 heuristic for this descriptor: " << e.what();
        }
        throw;
    }
}

TEST(CublasMatrixMultiply, HeuristicMatmulSupportsInt8InputsAndFp32OutputInTNLayout) {
    constexpr int gpuNum = 0;
    if (!gpuComputeCapabilityAtLeast(gpuNum, 7, 5)) {
        GTEST_SKIP() << "INT8 tensor core GEMM kernels require a Turing-or-newer GPU.";
    }

    try {
        runHeuristicMatmulSurfaceCase<int8_t, int8_t, float>(gpuNum,
                                                             64,  // rowsA = k because transposeA=true
                                                             32,  // colsA = m
                                                             64,  // rowsB = k
                                                             48,  // colsB = n
                                                             true,
                                                             false,
                                                             1.0e-3f);
    } catch (const std::runtime_error &e) {
        if (isCublasLtSurfaceAvailabilityError(e)) {
            GTEST_SKIP() << "cuBLASLt did not expose a usable INT8->FP32 heuristic for this descriptor: " << e.what();
        }
        throw;
    }
}

namespace {

template <typename T>
TensorDescriptor::DataType supportedFp8ThorDataType();

template <>
TensorDescriptor::DataType supportedFp8ThorDataType<float>() {
    return TensorDescriptor::DataType::FP32;
}

template <>
TensorDescriptor::DataType supportedFp8ThorDataType<half>() {
    return TensorDescriptor::DataType::FP16;
}

template <>
TensorDescriptor::DataType supportedFp8ThorDataType<__nv_bfloat16>() {
    return TensorDescriptor::DataType::BF16;
}

template <>
TensorDescriptor::DataType supportedFp8ThorDataType<__nv_fp8_e4m3>() {
    return TensorDescriptor::DataType::FP8_E4M3;
}

template <>
TensorDescriptor::DataType supportedFp8ThorDataType<__nv_fp8_e5m2>() {
    return TensorDescriptor::DataType::FP8_E5M2;
}

template <typename T>
T supportedFp8ValueFromFloat(float value);

template <>
float supportedFp8ValueFromFloat<float>(float value) {
    return value;
}

template <>
half supportedFp8ValueFromFloat<half>(float value) {
    return __float2half(value);
}

template <>
__nv_bfloat16 supportedFp8ValueFromFloat<__nv_bfloat16>(float value) {
    return __float2bfloat16(value);
}

template <>
__nv_fp8_e4m3 supportedFp8ValueFromFloat<__nv_fp8_e4m3>(float value) {
    return __nv_fp8_e4m3(value);
}

template <>
__nv_fp8_e5m2 supportedFp8ValueFromFloat<__nv_fp8_e5m2>(float value) {
    return __nv_fp8_e5m2(value);
}

template <typename T>
float supportedFp8ValueToFloat(T value) {
    return static_cast<float>(value);
}

template <>
float supportedFp8ValueToFloat<float>(float value) {
    return value;
}

template <>
float supportedFp8ValueToFloat<half>(half value) {
    return __half2float(value);
}

template <>
float supportedFp8ValueToFloat<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <>
float supportedFp8ValueToFloat<__nv_fp8_e4m3>(__nv_fp8_e4m3 value) {
    return static_cast<float>(value);
}

template <>
float supportedFp8ValueToFloat<__nv_fp8_e5m2>(__nv_fp8_e5m2 value) {
    return static_cast<float>(value);
}

template <typename T>
void fillSupportedFp8PaddedMatrix(Tensor &tensor, int rows, int cols, int ld, float rowScale, float colScale, float offset) {
    T *mem = reinterpret_cast<T *>(tensor.getMemPtr());
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < ld; ++col) {
            float value = 0.0f;
            if (col < cols) {
                value = offset + rowScale * static_cast<float>((row % 7) - 3) + colScale * static_cast<float>((col % 5) - 2);
            }
            mem[row * ld + col] = supportedFp8ValueFromFloat<T>(value);
        }
    }
}

template <typename TA, typename TB, typename TC>
std::vector<float> referenceSupportedFp8Gemm(const Tensor &A,
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
                const float av = transposeA ? supportedFp8ValueToFloat(a[k * ldA + row]) : supportedFp8ValueToFloat(a[row * ldA + k]);
                const float bv = transposeB ? supportedFp8ValueToFloat(b[col * ldB + k]) : supportedFp8ValueToFloat(b[k * ldB + col]);
                accum += av * bv;
            }

            expected[static_cast<size_t>(row) * static_cast<size_t>(colsD) + static_cast<size_t>(col)] =
                alpha * accum + beta * supportedFp8ValueToFloat(c[row * ldC + col]);
        }
    }

    return expected;
}

template <typename TD>
void assertSupportedFp8MatrixClose(const Tensor &D, const std::vector<float> &expected, int rows, int cols, int ld, float tolerance) {
    const TD *d = reinterpret_cast<const TD *>(D.getMemPtr());

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            const float got = supportedFp8ValueToFloat(d[row * ld + col]);
            const float want = expected[static_cast<size_t>(row) * static_cast<size_t>(cols) + static_cast<size_t>(col)];
            ASSERT_NEAR(got, want, tolerance) << "row=" << row << " col=" << col;
        }
    }
}

bool isSupportedFp8CublasLtAvailabilityError(const std::exception &e) {
    const std::string message(e.what());
    return message.find("CUBLAS_STATUS_NOT_SUPPORTED") != std::string::npos ||
           message.find("CUBLAS_STATUS_ARCH_MISMATCH") != std::string::npos ||
           message.find("CUBLAS_STATUS_NOT_INITIALIZED") != std::string::npos;
}

template <typename TA, typename TB, typename TD>
void runSupportedFp8OptimalMatmulCase(int gpuNum, float tolerance) {
    if (!gpuSupportsFp8TensorCores(gpuNum)) {
        GTEST_SKIP() << "FP8 GEMM kernels require an FP8-capable GPU.";
    }

    ScopedGpu scopedGpu(gpuNum);
    Stream stream(gpuNum);

    constexpr int m = 256;
    constexpr int n = 256;
    constexpr int k = 256;

    // FP8 cuBLASLt path: TN.
    constexpr int rowsA = k;
    constexpr int colsA = m;
    constexpr int rowsB = k;
    constexpr int colsB = n;
    constexpr int ldA = colsA;
    constexpr int ldB = colsB;
    constexpr int ldD = colsB;
    constexpr bool transposeA = true;
    constexpr bool transposeB = false;

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU, 0);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, gpuNum);

    TensorDescriptor ADescriptor(supportedFp8ThorDataType<TA>(), {static_cast<uint64_t>(rowsA), static_cast<uint64_t>(ldA)});
    TensorDescriptor BDescriptor(supportedFp8ThorDataType<TB>(), {static_cast<uint64_t>(rowsB), static_cast<uint64_t>(ldB)});
    TensorDescriptor DDescriptor(supportedFp8ThorDataType<TD>(), {static_cast<uint64_t>(m), static_cast<uint64_t>(ldD)});

    Tensor A(cpuPlacement, ADescriptor);
    Tensor B(cpuPlacement, BDescriptor);
    Tensor D(cpuPlacement, DDescriptor);
    Tensor A_d(gpuPlacement, ADescriptor);
    Tensor B_d(gpuPlacement, BDescriptor);
    Tensor D_d(gpuPlacement, DDescriptor);
    Tensor D_h(cpuPlacement, DDescriptor);

    fillSupportedFp8PaddedMatrix<TA>(A, rowsA, colsA, ldA, 0.03125f, -0.015625f, 0.0625f);
    fillSupportedFp8PaddedMatrix<TB>(B, rowsB, colsB, ldB, -0.015625f, 0.03125f, -0.03125f);
    fillSupportedFp8PaddedMatrix<TD>(D, m, n, ldD, 0.0f, 0.0f, -7.0f);

    const std::vector<float> expected =
        referenceSupportedFp8Gemm<TA, TB, TD>(A, B, D, rowsA, colsA, rowsB, colsB, ldA, ldB, ldD, transposeA, transposeB, 1.0f, 0.0f);

    A_d.copyFromAsync(A, stream);
    B_d.copyFromAsync(B, stream);
    D_d.copyFromAsync(D, stream);
    stream.synchronize();

    ScopedDeviceFp8MatmulScales scales(1.0f, 1.0f);

    const auto dataTypes = CublasMatrixMultiply::MatmulDataTypes{
        supportedFp8ThorDataType<TA>(), supportedFp8ThorDataType<TB>(), supportedFp8ThorDataType<TD>(), supportedFp8ThorDataType<TD>()};
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;

    try {
        CublasMatrixMultiply::instance().chooseOptimalGemmKernel(
            gpuNum, rowsA, colsA, rowsB, colsB, ldA, ldB, ldD, ldD, transposeA, transposeB, false, dataTypes, scales.inputScales(), false);

        bool kernelWillRunOnGpu = false;
        const uint64_t workspaceSizeInBytes = CublasMatrixMultiply::instance().getGemmWorkspaceSizeInBytes(gpuNum,
                                                                                                           rowsA,
                                                                                                           colsA,
                                                                                                           rowsB,
                                                                                                           colsB,
                                                                                                           ldA,
                                                                                                           ldB,
                                                                                                           ldD,
                                                                                                           ldD,
                                                                                                           transposeA,
                                                                                                           transposeB,
                                                                                                           false,
                                                                                                           dataTypes,
                                                                                                           scales.inputScales(),
                                                                                                           kernelWillRunOnGpu);
        ASSERT_TRUE(kernelWillRunOnGpu);

        Optional<Tensor> workspace_d;
        if (workspaceSizeInBytes > 0) {
            workspace_d = Tensor(gpuPlacement, TensorDescriptor(TensorDescriptor::DataType::UINT8, {workspaceSizeInBytes}));
        }

        CublasMatrixMultiply::instance().gemm(A_d,
                                              B_d,
                                              D_d,
                                              D_d,
                                              workspace_d,
                                              rowsA,
                                              colsA,
                                              rowsB,
                                              colsB,
                                              transposeA,
                                              transposeB,
                                              false,
                                              &alpha,
                                              &beta,
                                              dataTypes,
                                              scales.inputScales(),
                                              stream);
    } catch (const std::runtime_error &e) {
        if (isSupportedFp8CublasLtAvailabilityError(e) ||
            std::string(e.what()).find("could not find any cuBLASLt kernel candidates") != std::string::npos) {
            GTEST_SKIP() << "cuBLASLt did not expose a usable workspace-backed kernel for this supported FP8 descriptor: " << e.what();
        }
        throw;
    }

    D_h.copyFromAsync(D_d, stream);
    stream.synchronize();

    assertSupportedFp8MatrixClose<TD>(D_h, expected, m, n, ldD, tolerance);
}

template <typename TA, typename TB, typename TC, typename TD>
void runSupportedFp8OptimalGemmCase(int gpuNum, float alpha, float beta, float tolerance) {
    if (!gpuSupportsFp8TensorCores(gpuNum)) {
        GTEST_SKIP() << "FP8 GEMM kernels require an FP8-capable GPU.";
    }

    ScopedGpu scopedGpu(gpuNum);
    Stream stream(gpuNum);

    constexpr int m = 129;
    constexpr int n = 128;
    constexpr int k = 128;

    // FP8 cuBLASLt path: TN.
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

    TensorDescriptor ADescriptor(supportedFp8ThorDataType<TA>(), {static_cast<uint64_t>(rowsA), static_cast<uint64_t>(ldA)});
    TensorDescriptor BDescriptor(supportedFp8ThorDataType<TB>(), {static_cast<uint64_t>(rowsB), static_cast<uint64_t>(ldB)});
    TensorDescriptor CDescriptor(supportedFp8ThorDataType<TC>(), {static_cast<uint64_t>(m), static_cast<uint64_t>(ldC)});
    TensorDescriptor DDescriptor(supportedFp8ThorDataType<TD>(), {static_cast<uint64_t>(m), static_cast<uint64_t>(ldD)});

    Tensor A(cpuPlacement, ADescriptor);
    Tensor B(cpuPlacement, BDescriptor);
    Tensor C(cpuPlacement, CDescriptor);
    Tensor D(cpuPlacement, DDescriptor);
    Tensor A_d(gpuPlacement, ADescriptor);
    Tensor B_d(gpuPlacement, BDescriptor);
    Tensor C_d(gpuPlacement, CDescriptor);
    Tensor D_d(gpuPlacement, DDescriptor);
    Tensor D_h(cpuPlacement, DDescriptor);

    fillSupportedFp8PaddedMatrix<TA>(A, rowsA, colsA, ldA, 0.03125f, -0.015625f, 0.0625f);
    fillSupportedFp8PaddedMatrix<TB>(B, rowsB, colsB, ldB, -0.015625f, 0.03125f, -0.03125f);
    fillSupportedFp8PaddedMatrix<TC>(C, m, n, ldC, 0.0078125f, -0.015625f, 0.125f);
    fillSupportedFp8PaddedMatrix<TD>(D, m, n, ldD, 0.0f, 0.0f, -7.0f);

    const std::vector<float> expected =
        referenceSupportedFp8Gemm<TA, TB, TC>(A, B, C, rowsA, colsA, rowsB, colsB, ldA, ldB, ldC, transposeA, transposeB, alpha, beta);

    A_d.copyFromAsync(A, stream);
    B_d.copyFromAsync(B, stream);
    C_d.copyFromAsync(C, stream);
    D_d.copyFromAsync(D, stream);
    stream.synchronize();

    ScopedDeviceFp8MatmulScales scales(1.0f, 1.0f);
    const CublasMatrixMultiply::Fp8MatmulScales fp8Scales = (supportedFp8ThorDataType<TD>() == TensorDescriptor::DataType::FP8_E4M3 ||
                                                             supportedFp8ThorDataType<TD>() == TensorDescriptor::DataType::FP8_E5M2)
                                                                ? scales.allScales()
                                                                : scales.inputScales();

    const auto dataTypes = CublasMatrixMultiply::MatmulDataTypes{
        supportedFp8ThorDataType<TA>(), supportedFp8ThorDataType<TB>(), supportedFp8ThorDataType<TC>(), supportedFp8ThorDataType<TD>()};

    try {
        CublasMatrixMultiply::instance().chooseOptimalGemmKernel(
            gpuNum, rowsA, colsA, rowsB, colsB, ldA, ldB, ldC, ldD, transposeA, transposeB, transposeC, dataTypes, fp8Scales, false);

        bool kernelWillRunOnGpu = false;
        const uint64_t workspaceSizeInBytes = CublasMatrixMultiply::instance().getGemmWorkspaceSizeInBytes(gpuNum,
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
                                                                                                           dataTypes,
                                                                                                           fp8Scales,
                                                                                                           kernelWillRunOnGpu);
        ASSERT_TRUE(kernelWillRunOnGpu);

        Optional<Tensor> workspace_d;
        if (workspaceSizeInBytes > 0) {
            workspace_d = Tensor(gpuPlacement, TensorDescriptor(TensorDescriptor::DataType::UINT8, {workspaceSizeInBytes}));
        }

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
                                              dataTypes,
                                              fp8Scales,
                                              stream);
    } catch (const std::runtime_error &e) {
        if (isSupportedFp8CublasLtAvailabilityError(e) ||
            std::string(e.what()).find("could not find any cuBLASLt kernel candidates") != std::string::npos) {
            GTEST_SKIP() << "cuBLASLt did not expose a usable workspace-backed kernel for this supported FP8 descriptor: " << e.what();
        }
        throw;
    }

    D_h.copyFromAsync(D_d, stream);
    stream.synchronize();

    assertSupportedFp8MatrixClose<TD>(D_h, expected, m, n, ldD, tolerance);
}

struct PackedFp8GemmShape {
    int m;
    int n;
    int k;
    int rowsA;
    int colsA;
    int rowsB;
    int colsB;
    int ldA;
    int ldB;
    int ldC;
    int ldD;
    bool transposeA;
    bool transposeB;
    bool transposeC;
};

PackedFp8GemmShape makePackedFp8GemmShape(int m, int n, int k, bool transposeA, bool transposeB) {
    PackedFp8GemmShape shape{};
    shape.m = m;
    shape.n = n;
    shape.k = k;
    shape.transposeA = transposeA;
    shape.transposeB = transposeB;
    shape.transposeC = false;
    shape.rowsA = transposeA ? k : m;
    shape.colsA = transposeA ? m : k;
    shape.rowsB = transposeB ? n : k;
    shape.colsB = transposeB ? k : n;
    shape.ldA = shape.colsA;
    shape.ldB = shape.colsB;
    shape.ldC = n;
    shape.ldD = n;
    return shape;
}

const char *fp8LayoutName(bool transposeA, bool transposeB) {
    if (!transposeA && !transposeB) {
        return "NN";
    }
    if (!transposeA && transposeB) {
        return "NT";
    }
    if (transposeA && !transposeB) {
        return "TN";
    }
    return "TT";
}

bool fp8LayoutNeedsTemporaryTransposeWorkspace(bool transposeA, bool transposeB) { return transposeA || !transposeB; }

void expectRuntimeErrorContains(const std::function<void()> &fn, const std::string &expectedText) {
    try {
        fn();
        FAIL() << "Expected std::runtime_error containing: " << expectedText;
    } catch (const std::runtime_error &e) {
        const std::string message(e.what());
        EXPECT_NE(message.find(expectedText), std::string::npos) << message;
        EXPECT_EQ(message.find("CUBLAS_STATUS"), std::string::npos) << message;
    }
}

void expectFp8GemmSetupRejectsShape(const PackedFp8GemmShape &shape, const std::string &expectedText) {
    ScopedDeviceFp8MatmulScales scales(1.0f, 1.0f);
    const CublasMatrixMultiply::MatmulDataTypes dataTypes{
        TensorDescriptor::DataType::FP8_E4M3,
        TensorDescriptor::DataType::FP8_E5M2,
        TensorDescriptor::DataType::FP32,
        TensorDescriptor::DataType::FP32,
    };

    expectRuntimeErrorContains(
        [&]() {
            CublasMatrixMultiply::instance().chooseOptimalGemmKernel(0,
                                                                     shape.rowsA,
                                                                     shape.colsA,
                                                                     shape.rowsB,
                                                                     shape.colsB,
                                                                     shape.ldA,
                                                                     shape.ldB,
                                                                     shape.ldC,
                                                                     shape.ldD,
                                                                     shape.transposeA,
                                                                     shape.transposeB,
                                                                     shape.transposeC,
                                                                     dataTypes,
                                                                     scales.inputScales(),
                                                                     false);
        },
        expectedText);

    expectRuntimeErrorContains(
        [&]() {
            bool kernelWillRunOnGpu = false;
            (void)CublasMatrixMultiply::instance().getGemmWorkspaceSizeInBytes(0,
                                                                               shape.rowsA,
                                                                               shape.colsA,
                                                                               shape.rowsB,
                                                                               shape.colsB,
                                                                               shape.ldA,
                                                                               shape.ldB,
                                                                               shape.ldC,
                                                                               shape.ldD,
                                                                               shape.transposeA,
                                                                               shape.transposeB,
                                                                               shape.transposeC,
                                                                               dataTypes,
                                                                               scales.inputScales(),
                                                                               kernelWillRunOnGpu);
        },
        expectedText);
}

void runPackedFp8ToFp32OptimalGemmLayoutCase(int gpuNum, bool transposeA, bool transposeB) {
    if (!gpuSupportsFp8TensorCores(gpuNum)) {
        GTEST_SKIP() << "FP8 GEMM kernels require an FP8-capable GPU.";
    }

    ScopedGpu scopedGpu(gpuNum);
    Stream stream(gpuNum);

    const PackedFp8GemmShape shape = makePackedFp8GemmShape(64, 64, 64, transposeA, transposeB);
    constexpr float alpha = 0.75f;
    constexpr float beta = -0.25f;

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU, 0);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, gpuNum);

    TensorDescriptor ADescriptor(TensorDescriptor::DataType::FP8_E4M3,
                                 {static_cast<uint64_t>(shape.rowsA), static_cast<uint64_t>(shape.ldA)});
    TensorDescriptor BDescriptor(TensorDescriptor::DataType::FP8_E5M2,
                                 {static_cast<uint64_t>(shape.rowsB), static_cast<uint64_t>(shape.ldB)});
    TensorDescriptor CDescriptor(TensorDescriptor::DataType::FP32, {static_cast<uint64_t>(shape.m), static_cast<uint64_t>(shape.ldC)});
    TensorDescriptor DDescriptor(TensorDescriptor::DataType::FP32, {static_cast<uint64_t>(shape.m), static_cast<uint64_t>(shape.ldD)});

    Tensor A(cpuPlacement, ADescriptor);
    Tensor B(cpuPlacement, BDescriptor);
    Tensor C(cpuPlacement, CDescriptor);
    Tensor D(cpuPlacement, DDescriptor);
    Tensor A_d(gpuPlacement, ADescriptor);
    Tensor B_d(gpuPlacement, BDescriptor);
    Tensor C_d(gpuPlacement, CDescriptor);
    Tensor D_d(gpuPlacement, DDescriptor);
    Tensor D_h(cpuPlacement, DDescriptor);

    fillSupportedFp8PaddedMatrix<__nv_fp8_e4m3>(A, shape.rowsA, shape.colsA, shape.ldA, 0.03125f, -0.015625f, 0.0625f);
    fillSupportedFp8PaddedMatrix<__nv_fp8_e5m2>(B, shape.rowsB, shape.colsB, shape.ldB, -0.015625f, 0.03125f, -0.03125f);
    fillSupportedFp8PaddedMatrix<float>(C, shape.m, shape.n, shape.ldC, 0.0078125f, -0.015625f, 0.125f);
    fillSupportedFp8PaddedMatrix<float>(D, shape.m, shape.n, shape.ldD, 0.0f, 0.0f, -7.0f);

    const std::vector<float> expected = referenceSupportedFp8Gemm<__nv_fp8_e4m3, __nv_fp8_e5m2, float>(A,
                                                                                                       B,
                                                                                                       C,
                                                                                                       shape.rowsA,
                                                                                                       shape.colsA,
                                                                                                       shape.rowsB,
                                                                                                       shape.colsB,
                                                                                                       shape.ldA,
                                                                                                       shape.ldB,
                                                                                                       shape.ldC,
                                                                                                       shape.transposeA,
                                                                                                       shape.transposeB,
                                                                                                       alpha,
                                                                                                       beta);

    A_d.copyFromAsync(A, stream);
    B_d.copyFromAsync(B, stream);
    C_d.copyFromAsync(C, stream);
    D_d.copyFromAsync(D, stream);
    stream.synchronize();

    ScopedDeviceFp8MatmulScales scales(1.0f, 1.0f, 1.0f);
    const CublasMatrixMultiply::MatmulDataTypes dataTypes{
        TensorDescriptor::DataType::FP8_E4M3,
        TensorDescriptor::DataType::FP8_E5M2,
        TensorDescriptor::DataType::FP32,
        TensorDescriptor::DataType::FP32,
    };

    try {
        CublasMatrixMultiply::instance().chooseOptimalGemmKernel(gpuNum,
                                                                 shape.rowsA,
                                                                 shape.colsA,
                                                                 shape.rowsB,
                                                                 shape.colsB,
                                                                 shape.ldA,
                                                                 shape.ldB,
                                                                 shape.ldC,
                                                                 shape.ldD,
                                                                 shape.transposeA,
                                                                 shape.transposeB,
                                                                 shape.transposeC,
                                                                 dataTypes,
                                                                 scales.inputAndCScales(),
                                                                 false);

        bool kernelWillRunOnGpu = false;
        const uint64_t workspaceSizeInBytes = CublasMatrixMultiply::instance().getGemmWorkspaceSizeInBytes(gpuNum,
                                                                                                           shape.rowsA,
                                                                                                           shape.colsA,
                                                                                                           shape.rowsB,
                                                                                                           shape.colsB,
                                                                                                           shape.ldA,
                                                                                                           shape.ldB,
                                                                                                           shape.ldC,
                                                                                                           shape.ldD,
                                                                                                           shape.transposeA,
                                                                                                           shape.transposeB,
                                                                                                           shape.transposeC,
                                                                                                           dataTypes,
                                                                                                           scales.inputAndCScales(),
                                                                                                           kernelWillRunOnGpu);
        ASSERT_TRUE(kernelWillRunOnGpu);

        Optional<Tensor> workspace_d;
        if (workspaceSizeInBytes > 0) {
            workspace_d = Tensor(gpuPlacement, TensorDescriptor(TensorDescriptor::DataType::UINT8, {workspaceSizeInBytes}));
        }

        CublasMatrixMultiply::instance().gemm(A_d,
                                              B_d,
                                              C_d,
                                              D_d,
                                              workspace_d,
                                              shape.rowsA,
                                              shape.colsA,
                                              shape.rowsB,
                                              shape.colsB,
                                              shape.transposeA,
                                              shape.transposeB,
                                              shape.transposeC,
                                              &alpha,
                                              &beta,
                                              dataTypes,
                                              scales.inputAndCScales(),
                                              stream);
    } catch (const std::runtime_error &e) {
        if (isSupportedFp8CublasLtAvailabilityError(e) ||
            std::string(e.what()).find("could not find any cuBLASLt kernel candidates") != std::string::npos) {
            GTEST_SKIP() << "cuBLASLt did not expose a usable FP8 kernel for external " << fp8LayoutName(transposeA, transposeB)
                         << " layout: " << e.what();
        }
        throw;
    }

    D_h.copyFromAsync(D_d, stream);
    stream.synchronize();

    assertFp32MatrixClose(D_h, expected, shape.m, shape.n, shape.ldD, 1.0f);
}

void expectFp8HeuristicGemmRejectsTemporaryTransposeLayout(int gpuNum, bool transposeA, bool transposeB) {
    ASSERT_TRUE(fp8LayoutNeedsTemporaryTransposeWorkspace(transposeA, transposeB));

    ScopedGpu scopedGpu(gpuNum);
    Stream stream(gpuNum);

    const PackedFp8GemmShape shape = makePackedFp8GemmShape(32, 32, 32, transposeA, transposeB);
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;

    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, gpuNum);
    TensorDescriptor ADescriptor(TensorDescriptor::DataType::FP8_E4M3,
                                 {static_cast<uint64_t>(shape.rowsA), static_cast<uint64_t>(shape.ldA)});
    TensorDescriptor BDescriptor(TensorDescriptor::DataType::FP8_E5M2,
                                 {static_cast<uint64_t>(shape.rowsB), static_cast<uint64_t>(shape.ldB)});
    TensorDescriptor CDescriptor(TensorDescriptor::DataType::FP32, {static_cast<uint64_t>(shape.m), static_cast<uint64_t>(shape.ldC)});
    TensorDescriptor DDescriptor(TensorDescriptor::DataType::FP32, {static_cast<uint64_t>(shape.m), static_cast<uint64_t>(shape.ldD)});

    Tensor A_d(gpuPlacement, ADescriptor);
    Tensor B_d(gpuPlacement, BDescriptor);
    Tensor C_d(gpuPlacement, CDescriptor);
    Tensor D_d(gpuPlacement, DDescriptor);

    ScopedDeviceFp8MatmulScales scales(1.0f, 1.0f);
    const CublasMatrixMultiply::MatmulDataTypes dataTypes{
        TensorDescriptor::DataType::FP8_E4M3,
        TensorDescriptor::DataType::FP8_E5M2,
        TensorDescriptor::DataType::FP32,
        TensorDescriptor::DataType::FP32,
    };

    expectRuntimeErrorContains(
        [&]() {
            CublasMatrixMultiply::instance().gemmUsingHeuristicKernelChoice(A_d,
                                                                            B_d,
                                                                            C_d,
                                                                            D_d,
                                                                            shape.rowsA,
                                                                            shape.colsA,
                                                                            shape.rowsB,
                                                                            shape.colsB,
                                                                            shape.transposeA,
                                                                            shape.transposeB,
                                                                            shape.transposeC,
                                                                            &alpha,
                                                                            &beta,
                                                                            dataTypes,
                                                                            scales.inputScales(),
                                                                            stream);
        },
        "FP8 row-major path requires chooseOptimalGemmKernel/gemm with workspace");
}

}  // namespace

TEST(CublasMatrixMultiply, ChooseOptimalMatmulSupportsFp8E4m3InputsAndFp16OutputInTNLayout) {
    runSupportedFp8OptimalMatmulCase<__nv_fp8_e4m3, __nv_fp8_e4m3, half>(0, 0.75f);
}

TEST(CublasMatrixMultiply, ChooseOptimalGemmSupportsFp8MixedInputsAndBf16OutputInTNLayout) {
    runSupportedFp8OptimalGemmCase<__nv_fp8_e4m3, __nv_fp8_e5m2, __nv_bfloat16, __nv_bfloat16>(0, 0.75f, -0.5f, 1.0f);
}

TEST(CublasMatrixMultiply, ChooseOptimalGemmSupportsFp8InputsAndFp8E4m3OutputInTNLayout) {
    runSupportedFp8OptimalGemmCase<__nv_fp8_e4m3, __nv_fp8_e5m2, half, __nv_fp8_e4m3>(0, 0.75f, 0.25f, 1.0f);
}

TEST(CublasMatrixMultiply, ChooseOptimalGemmSupportsFp8InputsAndFp8E5m2OutputWhenInputPairIsNotBothE4m3InTNLayout) {
    runSupportedFp8OptimalGemmCase<__nv_fp8_e5m2, __nv_fp8_e4m3, __nv_bfloat16, __nv_fp8_e5m2>(0, 0.75f, 0.25f, 1.5f);
}

TEST(CublasMatrixMultiply, ChooseOptimalGemmSupportsFp8InputsAndFp32OutputInNNLayoutWithWorkspace) {
    runPackedFp8ToFp32OptimalGemmLayoutCase(0, false, false);  // NN materializes B for the internal FP8 COL/TN path.
}

TEST(CublasMatrixMultiply, ChooseOptimalGemmSupportsFp8InputsAndFp32OutputInNTLayoutWithoutHelperWorkspace) {
    runPackedFp8ToFp32OptimalGemmLayoutCase(0, false, true);  // NT is the zero-copy external layout.
}

TEST(CublasMatrixMultiply, ChooseOptimalGemmSupportsFp8InputsAndFp32OutputInTNLayoutWithWorkspace) {
    runPackedFp8ToFp32OptimalGemmLayoutCase(0, true, false);  // TN materializes A and B.
}

TEST(CublasMatrixMultiply, ChooseOptimalGemmSupportsFp8InputsAndFp32OutputInTTLayoutWithWorkspace) {
    runPackedFp8ToFp32OptimalGemmLayoutCase(0, true, true);  // TT materializes A.
}

TEST(CublasMatrixMultiply, Fp8GemmRejectsOddNAndOddKBeforeCublasLt) {
    constexpr int gpuNum = 0;
    if (!gpuSupportsFp8TensorCores(gpuNum)) {
        GTEST_SKIP() << "FP8 tensors require an FP8-capable GPU for this validation test.";
    }

    PackedFp8GemmShape oddN = makePackedFp8GemmShape(64, 63, 64, true, false);
    expectFp8GemmSetupRejectsShape(oddN, "requires even N");

    PackedFp8GemmShape oddK = makePackedFp8GemmShape(64, 64, 63, true, false);
    expectFp8GemmSetupRejectsShape(oddK, "requires even K");
}

TEST(CublasMatrixMultiply, Fp8GemmRejectsUnpackedLeadingDimensionsBeforeCublasLt) {
    constexpr int gpuNum = 0;
    if (!gpuSupportsFp8TensorCores(gpuNum)) {
        GTEST_SKIP() << "FP8 tensors require an FP8-capable GPU for this validation test.";
    }

    PackedFp8GemmShape unpackedA = makePackedFp8GemmShape(64, 64, 64, true, false);
    unpackedA.ldA += 1;
    expectFp8GemmSetupRejectsShape(unpackedA, "requires packed A");

    PackedFp8GemmShape unpackedB = makePackedFp8GemmShape(64, 64, 64, true, false);
    unpackedB.ldB += 1;
    expectFp8GemmSetupRejectsShape(unpackedB, "requires packed B");

    PackedFp8GemmShape unpackedC = makePackedFp8GemmShape(64, 64, 64, true, false);
    unpackedC.ldC += 1;
    expectFp8GemmSetupRejectsShape(unpackedC, "requires packed C");

    PackedFp8GemmShape unpackedD = makePackedFp8GemmShape(64, 64, 64, true, false);
    unpackedD.ldD += 1;
    expectFp8GemmSetupRejectsShape(unpackedD, "requires packed D");
}

TEST(CublasMatrixMultiply, HeuristicGemmRejectsFp8LayoutsThatNeedTemporaryTransposeWorkspace) {
    constexpr int gpuNum = 0;
    if (!gpuSupportsFp8TensorCores(gpuNum)) {
        GTEST_SKIP() << "FP8 tensors require an FP8-capable GPU for this validation test.";
    }

    expectFp8HeuristicGemmRejectsTemporaryTransposeLayout(gpuNum, false, false);  // NN needs B materialization.
    expectFp8HeuristicGemmRejectsTemporaryTransposeLayout(gpuNum, true, false);   // TN needs A and B materialization.
    expectFp8HeuristicGemmRejectsTemporaryTransposeLayout(gpuNum, true, true);    // TT needs A materialization.
}
