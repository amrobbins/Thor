#include "Thor.h"

#include "gtest/gtest.h"
#include "omp.h"

using namespace std;

#define INTEREST_KERNEL KernelWithSpec::KernelIndex::_256_96_bigSharedBlockA16Restrict

// This helper class changes the visibility of its member functions from protected to public, so that the test can call them.
class TensorCoreMatrixMultiplyTestHelper : public TensorCoreMatrixMultiply {
   public:
    static TensorCoreMatrixMultiplyTestHelper &instance() {
        static TensorCoreMatrixMultiplyTestHelper singletonInstance;
        return singletonInstance;
    }

    vector<KernelWithSpec> getAllKernels() { return TensorCoreMatrixMultiply::getAllKernels(); }
    static bool checkPreRequisitesBatch8() { return TensorCoreMatrixMultiply::checkPreRequisitesBatch8(); }
    static bool checkPreRequisitesBatch16() { return TensorCoreMatrixMultiply::checkPreRequisitesBatch16(); }

   private:
    TensorCoreMatrixMultiplyTestHelper() {}
};

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

TEST(TensorCoreMatrixMultiply, MeasureReductionPerformance) {
    bool print = false;
    if (!print)
        return;

    srand(time(NULL));

    constexpr int MAX_REDUCTIONS = 8;

    half *C;
    half *workspace;

    Stream stream(0);

    checkCudaErrors(cudaMalloc(&C, 4096 * 96 * sizeof(half)));
    checkCudaErrors(cudaMalloc(&workspace, (MAX_REDUCTIONS - 1) * 4096 * 96 * sizeof(half)));

    for (int reduceSize = 2; reduceSize <= MAX_REDUCTIONS; reduceSize += 2) {
        int rows = 96;
        int cols = 4096;
        int ld = cols;

        cudaEvent_t start, stop;
        checkCudaErrors(cudaEventCreate(&start));
        checkCudaErrors(cudaEventCreate(&stop));

        // Warm up
        for (int i = 0; i < 100; ++i) {
            if (reduceSize == 2)
                launchReduce2(rows, cols, ld, C, workspace, stream);
            else if (reduceSize == 4)
                launchReduce4(rows, cols, ld, C, workspace, stream);
            else if (reduceSize == 8)
                launchReduce8(rows, cols, ld, C, workspace, stream);
            else if (reduceSize == 6)
                launchReduce6(rows, cols, ld, C, workspace, stream);
            else
                assert(false);
        }
        // Measure
        checkCudaErrors(cudaEventRecord(start, stream.getStream()));
        for (int i = 0; i < 1000; ++i) {
            if (reduceSize == 2)
                launchReduce2(rows, cols, ld, C, workspace, stream);
            else if (reduceSize == 4)
                launchReduce4(rows, cols, ld, C, workspace, stream);
            else if (reduceSize == 8)
                launchReduce8(rows, cols, ld, C, workspace, stream);
            else if (reduceSize == 6)
                launchReduce6(rows, cols, ld, C, workspace, stream);
            else
                assert(false);
        }
        checkCudaErrors(cudaEventRecord(stop, stream.getStream()));
        checkCudaErrors(cudaEventSynchronize(stop));
        float milliseconds = 0;
        checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));

        printf("Reduce%d - %f millisecond per kernel\n", reduceSize, milliseconds / 1000.0f);

        checkCudaErrors(cudaEventDestroy(start));
        checkCudaErrors(cudaEventDestroy(stop));
    }

    checkCudaErrors(cudaFree(C));
    checkCudaErrors(cudaFree(workspace));
}

TEST(TensorCoreMatrixMultiply, ReductionsWork) {
    srand(time(NULL));

    constexpr int MAX_REDUCTIONS = 8;
    constexpr int MAX_SIZE = 700;
    constexpr int NUM_TESTS = 5;

    bool print = false;
    // bool printSourceToo = true;

    half *C_h;
    half *workspace_h;
    half *C_d;
    half *workspace_d;
    float result[MAX_SIZE * MAX_SIZE];
    half *gpuResult_h;

    Stream stream(0);

    checkCudaErrors(cudaHostAlloc(&C_h, MAX_SIZE * MAX_SIZE * sizeof(half), cudaHostAllocPortable));
    checkCudaErrors(cudaHostAlloc(&workspace_h, (MAX_REDUCTIONS - 1) * MAX_SIZE * MAX_SIZE * sizeof(half), cudaHostAllocPortable));
    checkCudaErrors(cudaMalloc(&C_d, MAX_SIZE * MAX_SIZE * sizeof(half)));
    checkCudaErrors(cudaMalloc(&workspace_d, (MAX_REDUCTIONS - 1) * MAX_SIZE * MAX_SIZE * sizeof(half)));

    checkCudaErrors(cudaHostAlloc(&gpuResult_h, MAX_SIZE * MAX_SIZE * sizeof(half), cudaHostAllocPortable));

    for (int i = 0; i < NUM_TESTS; ++i) {
        for (int reduceSize = 2; reduceSize <= MAX_REDUCTIONS; reduceSize += 2) {
            int rows = (rand() % MAX_SIZE) + 1;
            int cols = (rand() % MAX_SIZE) + 1;
            int ld = cols + (rand() % 40);
            if (ld > MAX_SIZE)
                ld = MAX_SIZE;

            // Fill the memInstances with random values
            for (int memInstance = 0; memInstance < reduceSize; ++memInstance) {
                if (memInstance == 0) {
                    for (int r = 0; r < rows; ++r) {
                        for (int c = 0; c < cols; ++c) {
                            half val = (rand() % 50) / 10.0f;
                            C_h[r * ld + c] = val;
                            result[r * ld + c] = (float)val;
                        }
                    }
                } else {
                    for (int r = 0; r < rows; ++r) {
                        for (int c = 0; c < cols; ++c) {
                            half val = (rand() % 50) / 10.0f;
                            workspace_h[((memInstance - 1) * rows * ld) + r * ld + c] = val;
                            result[r * ld + c] += (float)val;
                        }
                    }
                }

                // if (print && printSourceToo) {
                //    printf("Mem %d:\n", memInstance);
                //    for (int r = 0; r < rows; ++r) {
                //        for (int c = 0; c < cols; ++c) {
                //            printf("%3.2f ", (float)(memInstances_h[memInstance][r * ld + c]));
                //        }
                //        printf("\n");
                //    }
                //    printf("\n");
                //}

                if (memInstance == 0) {
                    checkCudaErrors(cudaMemcpyAsync(C_d, C_h, rows * ld * sizeof(half), cudaMemcpyHostToDevice, stream.getStream()));
                } else {
                    checkCudaErrors(cudaMemcpyAsync(workspace_d + (memInstance - 1) * rows * ld,
                                                    workspace_h + (memInstance - 1) * rows * ld,
                                                    rows * ld * sizeof(half),
                                                    cudaMemcpyHostToDevice,
                                                    stream.getStream()));
                }
            }

            // Launch reduction
            if (reduceSize == 2)
                launchReduce2(rows, cols, ld, C_d, workspace_d, stream);
            else if (reduceSize == 4)
                launchReduce4(rows, cols, ld, C_d, workspace_d, stream);
            else if (reduceSize == 8)
                launchReduce8(rows, cols, ld, C_d, workspace_d, stream);
            else if (reduceSize == 6)
                launchReduce6(rows, cols, ld, C_d, workspace_d, stream);
            else
                assert(false);

            checkCudaErrors(cudaMemcpyAsync(gpuResult_h, C_d, rows * ld * sizeof(half), cudaMemcpyDeviceToHost, stream.getStream()));
            checkCudaErrors(cudaStreamSynchronize(stream.getStream()));

            if (print) {
                printf("Result CPU:\n");
                for (int r = 0; r < rows; ++r) {
                    for (int c = 0; c < cols; ++c) {
                        printf("%3.2f ", (float)((half)result[r * ld + c]));
                    }
                    printf("\n");
                }
                printf("\n");

                printf("Result GPU:\n");
                for (int r = 0; r < rows; ++r) {
                    for (int c = 0; c < cols; ++c) {
                        printf("%3.2f ", (float)((half)gpuResult_h[r * ld + c]));
                    }
                    printf("\n");
                }
                printf("\n");
            }

            // Check result
            for (int r = 0; r < rows; ++r) {
                for (int c = 0; c < cols; ++c) {
                    ASSERT_LT(abs((float)gpuResult_h[r * ld + c] - (float)((half)(result[r * ld + c]))), 0.001);
                }
            }
        }
    }

    checkCudaErrors(cudaFreeHost(C_h));
    checkCudaErrors(cudaFreeHost(workspace_h));
    checkCudaErrors(cudaFree(C_d));
    checkCudaErrors(cudaFree(workspace_d));
    checkCudaErrors(cudaFreeHost(gpuResult_h));
}

TEST(TensorCoreMatrixMultiply, CheckPerformanceOfTargetKernel) {
    srand(time(NULL));

    KernelWithSpec::KernelIndex targetKernelId = INTEREST_KERNEL;

    vector<KernelWithSpec> kernels = TensorCoreMatrixMultiplyTestHelper::instance().getAllKernels();

    KernelWithSpec targetKernel;
    unsigned int i;
    for (i = 0; i < kernels.size(); ++i) {
        if (kernels[i].id == targetKernelId) {
            targetKernel = kernels[i];
            break;
        }
    }
    ASSERT_LT(i, kernels.size());

    int aRows = 23 * 256 + 16;
    int aCols = 47 * 128 + 16;
    int bCols = 48 * 96;
    int lda = aCols;
    int ldb = bCols;
    int ldc = bCols;

    KernelRequirement kernelRequirement;
    kernelRequirement.gpuType = MachineEvaluator::instance().getGpuType();
    kernelRequirement.rowsA = aRows;
    kernelRequirement.colsA = aCols;
    kernelRequirement.colsB = bCols;
    kernelRequirement.ldA = lda + 100;
    kernelRequirement.ldB = ldb + 100;
    kernelRequirement.ldC = ldc + 100;
    kernelRequirement.allowWorkspace = true;
    unsigned int workspaceSize = targetKernel.getWorkspaceSize(kernelRequirement);

    cudaError_t cudaStatus;

    Stream stream(0);

    int ldMax = 128 * 100;

    half *A_h;
    half *B_h;
    half *CGpu_h;

    float *A_h_float;
    float *B_h_float;
    float *C_h_float;

    checkCudaErrors(cudaHostAlloc(&A_h, ldMax * ldMax * sizeof(half), cudaHostAllocPortable));
    checkCudaErrors(cudaHostAlloc(&B_h, ldMax * ldMax * sizeof(half), cudaHostAllocPortable));
    checkCudaErrors(cudaHostAlloc(&CGpu_h, ldMax * ldMax * sizeof(half), cudaHostAllocPortable));

    checkCudaErrors(cudaHostAlloc(&A_h_float, ldMax * ldMax * sizeof(float), cudaHostAllocPortable));
    checkCudaErrors(cudaHostAlloc(&B_h_float, ldMax * ldMax * sizeof(float), cudaHostAllocPortable));
    checkCudaErrors(cudaHostAlloc(&C_h_float, ldMax * ldMax * sizeof(float), cudaHostAllocPortable));

    half *A;
    half *B;
    half *C;
    half *workspace_d = nullptr;
    cudaStatus = cudaMalloc(&A, ldMax * ldMax * sizeof(half));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&B, ldMax * ldMax * sizeof(half));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&C, ldMax * ldMax * sizeof(half));
    assert(cudaStatus == cudaSuccess);
    if (workspaceSize > 0) {
        cudaStatus = cudaMalloc(&workspace_d, workspaceSize);
        assert(cudaStatus == cudaSuccess);
    }

    checkCudaErrors(cudaMemcpy(A, A_h, ldMax * ldMax * sizeof(half), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(B, B_h, ldMax * ldMax * sizeof(half), cudaMemcpyHostToDevice));

    // Enforce kernel restrictions
    if (targetKernel.aColSizeModulusRequirement > 1 && targetKernel.bRowSizeModulusRequirement > 1)
        assert(targetKernel.aColSizeModulusRequirement % targetKernel.bRowSizeModulusRequirement == 0 ||
               targetKernel.bRowSizeModulusRequirement % targetKernel.aColSizeModulusRequirement == 0);
    if (targetKernel.aRowSizeModulusRequirement > 1) {
        aRows = (aRows / targetKernel.aRowSizeModulusRequirement) * targetKernel.aRowSizeModulusRequirement;
        if (aRows == 0)
            aRows = targetKernel.aRowSizeModulusRequirement;
    }
    if (targetKernel.aColSizeModulusRequirement > 1) {
        aCols = (aCols / targetKernel.aColSizeModulusRequirement) * targetKernel.aColSizeModulusRequirement;
        if (aCols == 0)
            aCols = targetKernel.aColSizeModulusRequirement;
    }
    if (targetKernel.bRowSizeModulusRequirement > 1) {
        aCols = (aCols / targetKernel.bRowSizeModulusRequirement) * targetKernel.bRowSizeModulusRequirement;
        if (aCols == 0)
            aCols = targetKernel.bRowSizeModulusRequirement;
    }
    if (targetKernel.bColSizeModulusRequirement > 1) {
        bCols = (bCols / targetKernel.bColSizeModulusRequirement) * targetKernel.bColSizeModulusRequirement;
        if (bCols == 0)
            bCols = targetKernel.bColSizeModulusRequirement;
    }
    lda = (rand() % 2) ? aCols : aCols + (rand() % 50);
    ldb = (rand() % 2) ? bCols : bCols + (rand() % 50);
    ldc = (rand() % 2) ? bCols : bCols + (rand() % 50);
    if (targetKernel.aColSizeModulusRequirement > 1)
        lda = (lda / targetKernel.aColSizeModulusRequirement) * targetKernel.aColSizeModulusRequirement;
    if (targetKernel.bColSizeModulusRequirement > 1)
        ldb = (ldb / targetKernel.bColSizeModulusRequirement) * targetKernel.bColSizeModulusRequirement;
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    int REPEAT = 100;
    // warmup
    for (int i = 0; i < 20; ++i)
        targetKernel.executeKernel(A, B, C, workspace_d, aRows, aCols, bCols, lda, ldb, ldc, stream);
    checkCudaErrors(cudaEventRecord(start, stream.getStream()));
    for (int i = 0; i < REPEAT; ++i)
        targetKernel.executeKernel(A, B, C, workspace_d, aRows, aCols, bCols, lda, ldb, ldc, stream);
    checkCudaErrors(cudaEventRecord(stop, stream.getStream()));
    cudaStatus = cudaEventSynchronize(stop);
    assert(cudaStatus == cudaSuccess);

    float milliseconds = 0;
    checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("%lf TFLOPS   (%f milliseconds per kernel)\n",
           ((2.0 * aCols - 1.0) * aRows * bCols * REPEAT) / (milliseconds * 1.0e9),
           milliseconds / REPEAT);

    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    checkCudaErrors(cudaStreamSynchronize(stream.getStream()));

    checkCudaErrors(cudaFree((void *)A));
    checkCudaErrors(cudaFree((void *)B));
    checkCudaErrors(cudaFree((void *)C));
    if (workspaceSize > 0)
        checkCudaErrors(cudaFree((void *)workspace_d));

    checkCudaErrors(cudaFreeHost(A_h));
    checkCudaErrors(cudaFreeHost(B_h));
    checkCudaErrors(cudaFreeHost(CGpu_h));
    checkCudaErrors(cudaFreeHost(A_h_float));
    checkCudaErrors(cudaFreeHost(B_h_float));
    checkCudaErrors(cudaFreeHost(C_h_float));
}

TEST(TensorCoreMatrixMultiply, TargetKernelMultipliesCorrectly) {
    srand(time(NULL));

    KernelWithSpec::KernelIndex targetKernelId = INTEREST_KERNEL;

    vector<KernelWithSpec> kernels = TensorCoreMatrixMultiplyTestHelper::instance().getAllKernels();

    KernelWithSpec targetKernel;
    unsigned int i;
    for (i = 0; i < kernels.size(); ++i) {
        if (kernels[i].id == targetKernelId) {
            targetKernel = kernels[i];
            break;
        }
    }
    ASSERT_LT(i, kernels.size());

    cudaError_t cudaStatus;

    Stream stream(0);

    int print = true;
    int ldMax = 1000;

    KernelRequirement kernelRequirement;
    kernelRequirement.gpuType = MachineEvaluator::instance().getGpuType();
    kernelRequirement.rowsA = ldMax;
    kernelRequirement.colsA = ldMax;
    kernelRequirement.colsB = ldMax;
    kernelRequirement.ldA = ldMax;
    kernelRequirement.ldB = ldMax;
    kernelRequirement.ldC = ldMax;
    kernelRequirement.allowWorkspace = true;
    unsigned int workspaceSize = targetKernel.getWorkspaceSize(kernelRequirement);

    half *A_h;
    half *B_h;
    half *CGpu_h;

    float *A_h_float;
    float *B_h_float;
    float *C_h_float;

    checkCudaErrors(cudaHostAlloc(&A_h, ldMax * ldMax * sizeof(half), cudaHostAllocPortable));
    checkCudaErrors(cudaHostAlloc(&B_h, ldMax * ldMax * sizeof(half), cudaHostAllocPortable));
    checkCudaErrors(cudaHostAlloc(&CGpu_h, ldMax * ldMax * sizeof(half), cudaHostAllocPortable));

    checkCudaErrors(cudaHostAlloc(&A_h_float, ldMax * ldMax * sizeof(float), cudaHostAllocPortable));
    checkCudaErrors(cudaHostAlloc(&B_h_float, ldMax * ldMax * sizeof(float), cudaHostAllocPortable));
    checkCudaErrors(cudaHostAlloc(&C_h_float, ldMax * ldMax * sizeof(float), cudaHostAllocPortable));

    for (int i = 0; i < ldMax; ++i) {
        for (int j = 0; j < ldMax; ++j) {
            A_h[i * ldMax + j] = half(((rand() % 100) - 50) / 10.0f);
            A_h_float[i * ldMax + j] = (float)A_h[i * ldMax + j];
            B_h[i * ldMax + j] = half(((rand() % 100) - 50) / 10.0f);
            B_h_float[i * ldMax + j] = (float)B_h[i * ldMax + j];
        }
    }

    half *A;
    half *B;
    half *C;
    half *workspace_d = nullptr;
    cudaStatus = cudaMalloc(&A, ldMax * ldMax * sizeof(half));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&B, ldMax * ldMax * sizeof(half));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&C, ldMax * ldMax * sizeof(half));
    assert(cudaStatus == cudaSuccess);
    if (workspaceSize > 0) {
        cudaStatus = cudaMalloc(&workspace_d, workspaceSize);
        assert(cudaStatus == cudaSuccess);
    }

    checkCudaErrors(cudaMemcpy(A, A_h, ldMax * ldMax * sizeof(half), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(B, B_h, ldMax * ldMax * sizeof(half), cudaMemcpyHostToDevice));

    constexpr int TESTS_PER_KERNEL = 20;

    for (int i = 0; i < TESTS_PER_KERNEL; ++i) {
        int aRows, aCols, bCols;
        int lda, ldb, ldc;
        if (i < TESTS_PER_KERNEL / 2) {
            aRows = (rand() % 100) + 1;
            aCols = (rand() % 100) + 1;
            bCols = (rand() % 100) + 1;
        } else {
            aRows = (rand() % (ldMax - 50)) + 1;
            aCols = (rand() % (ldMax - 50)) + 1;
            bCols = (rand() % (ldMax - 50)) + 1;
        }
        // Enforce kernel restrictions
        if (targetKernel.aColSizeModulusRequirement > 1 && targetKernel.bRowSizeModulusRequirement > 1)
            assert(targetKernel.aColSizeModulusRequirement % targetKernel.bRowSizeModulusRequirement == 0 ||
                   targetKernel.bRowSizeModulusRequirement % targetKernel.aColSizeModulusRequirement == 0);
        if (targetKernel.aRowSizeModulusRequirement > 1) {
            aRows = (aRows / targetKernel.aRowSizeModulusRequirement) * targetKernel.aRowSizeModulusRequirement;
            if (aRows == 0)
                aRows = targetKernel.aRowSizeModulusRequirement;
        }
        if (targetKernel.aColSizeModulusRequirement > 1) {
            aCols = (aCols / targetKernel.aColSizeModulusRequirement) * targetKernel.aColSizeModulusRequirement;
            if (aCols == 0)
                aCols = targetKernel.aColSizeModulusRequirement;
        }
        if (targetKernel.bRowSizeModulusRequirement > 1) {
            aCols = (aCols / targetKernel.bRowSizeModulusRequirement) * targetKernel.bRowSizeModulusRequirement;
            if (aCols == 0)
                aCols = targetKernel.bRowSizeModulusRequirement;
        }
        if (targetKernel.bColSizeModulusRequirement > 1) {
            bCols = (bCols / targetKernel.bColSizeModulusRequirement) * targetKernel.bColSizeModulusRequirement;
            if (bCols == 0)
                bCols = targetKernel.bColSizeModulusRequirement;
        }
        lda = (rand() % 2) ? aCols : aCols + (rand() % 50);
        ldb = (rand() % 2) ? bCols : bCols + (rand() % 50);
        ldc = (rand() % 2) ? bCols : bCols + (rand() % 50);
        if (targetKernel.aColSizeModulusRequirement > 1)
            lda = (lda / targetKernel.aColSizeModulusRequirement) * targetKernel.aColSizeModulusRequirement;
        if (targetKernel.bColSizeModulusRequirement > 1)
            ldb = (ldb / targetKernel.bColSizeModulusRequirement) * targetKernel.bColSizeModulusRequirement;
        int bRows = aCols;
        int cRows = aRows;
        int cCols = bCols;

        kernelRequirement.gpuType = MachineEvaluator::instance().getGpuType();
        kernelRequirement.rowsA = ldMax;
        kernelRequirement.colsA = ldMax;
        kernelRequirement.colsB = ldMax;
        kernelRequirement.ldA = ldMax;
        kernelRequirement.ldB = ldMax;
        kernelRequirement.ldC = ldMax;
        kernelRequirement.allowWorkspace = true;
        // Waring: I am assuming that the largest possible matrix multiply uses the largest possible workspace.
        // This is currently true but not necessarily true, if it becomes untrue then this test would have to be
        // updated to support testing the new kernel.
        assert(workspaceSize <= targetKernel.getWorkspaceSize(kernelRequirement));

        matrixMultiplyCpu(A_h_float, B_h_float, C_h_float, aRows, bCols, aCols, lda, ldb, ldc);

        targetKernel.executeKernel(A, B, C, workspace_d, aRows, aCols, bCols, lda, ldb, ldc, stream);

        checkCudaErrors(cudaMemcpyAsync(CGpu_h, C, cRows * ldc * sizeof(half), cudaMemcpyDeviceToHost, stream.getStream()));
        checkCudaErrors(cudaStreamSynchronize(stream.getStream()));

        float maxDiff = aCols * 0.0125;

        for (int i = 0; i < cRows; ++i) {
            for (int j = 0; j < cCols; ++j) {
                float diff = abs(C_h_float[i * ldc + j] - float(CGpu_h[i * ldc + j]));

                if (diff >= maxDiff) {
                    if (print) {
                        printf("A:\n");
                        printMatrix(A_h_float, aRows, aCols, lda);
                        printf("\n\n");

                        printf("B:\n");
                        printMatrix(B_h_float, bRows, bCols, ldb);
                        printf("\n\n");

                        printf("C:\n");
                        printMatrixWide(C_h_float, cRows, cCols, ldc);
                        printf("\n\n");

                        printf("C_GPU:\n");
                        printMatrixWide(CGpu_h, cRows, cCols, ldc);
                        printf("\n\n");
                    }

                    printf("arows %d acols %d bcols %d\n", aRows, aCols, bCols);
                    printf("row %d col %d : CPU %f vs %f GPU\n", i, j, C_h_float[i * ldc + j], float(CGpu_h[i * ldc + j]));
                    printf("kernel %d failed\n", (int)targetKernel.id);
                    fflush(stdout);
                }
                ASSERT_LT(diff, maxDiff);
            }
        }
    }

    checkCudaErrors(cudaFree((void *)A));
    checkCudaErrors(cudaFree((void *)B));
    checkCudaErrors(cudaFree((void *)C));
    if (workspaceSize > 0)
        checkCudaErrors(cudaFree((void *)workspace_d));

    checkCudaErrors(cudaFreeHost(A_h));
    checkCudaErrors(cudaFreeHost(B_h));
    checkCudaErrors(cudaFreeHost(CGpu_h));
    checkCudaErrors(cudaFreeHost(A_h_float));
    checkCudaErrors(cudaFreeHost(B_h_float));
    checkCudaErrors(cudaFreeHost(C_h_float));
}

TEST(TensorCoreMatrixMultiply, PreprocessorPrerequisitesMet) {
    ASSERT_TRUE(TensorCoreMatrixMultiplyTestHelper::checkPreRequisitesBatch8());
    ASSERT_TRUE(TensorCoreMatrixMultiplyTestHelper::checkPreRequisitesBatch16());
}

TEST(TensorCoreMatrixMultiply, ChooseOptimalKernelFindsAKernel) {
    srand(0);
    TensorCoreMatrixMultiply::instance().chooseOptimalKernel(0, 100, 210, 5);
    ASSERT_GT(TensorCoreMatrixMultiply::instance().getOptimalKernelTime(0, 100, 210, 5, true), 0.0f);

    srand(time(NULL));
}

TEST(TensorCoreMatrixMultiply, HeuristicMutliplyProducesCorrectResult) {
    srand(time(NULL));

    cudaError_t cudaStatus;

    Stream stream(0);

    int ldMax = 1400;

    half *A_h;
    half *B_h;
    half *CGpu_h;

    float *A_h_float;
    float *B_h_float;
    float *C_h_float;

    checkCudaErrors(cudaHostAlloc(&A_h, ldMax * ldMax * sizeof(half), cudaHostAllocPortable));
    checkCudaErrors(cudaHostAlloc(&B_h, ldMax * ldMax * sizeof(half), cudaHostAllocPortable));
    checkCudaErrors(cudaHostAlloc(&CGpu_h, ldMax * ldMax * sizeof(half), cudaHostAllocPortable));

    checkCudaErrors(cudaHostAlloc(&A_h_float, ldMax * ldMax * sizeof(float), cudaHostAllocPortable));
    checkCudaErrors(cudaHostAlloc(&B_h_float, ldMax * ldMax * sizeof(float), cudaHostAllocPortable));
    checkCudaErrors(cudaHostAlloc(&C_h_float, ldMax * ldMax * sizeof(float), cudaHostAllocPortable));

    for (int i = 0; i < ldMax; ++i) {
        for (int j = 0; j < ldMax; ++j) {
            A_h[i * ldMax + j] = half(((rand() % 100) - 50) / 10.0f);
            A_h_float[i * ldMax + j] = (float)A_h[i * ldMax + j];
            B_h[i * ldMax + j] = half(((rand() % 100) - 50) / 10.0f);
            B_h_float[i * ldMax + j] = (float)B_h[i * ldMax + j];
        }
    }

    half *A;
    half *B;
    half *C;
    cudaStatus = cudaMalloc(&A, ldMax * ldMax * sizeof(half));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&B, ldMax * ldMax * sizeof(half));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&C, ldMax * ldMax * sizeof(half));
    assert(cudaStatus == cudaSuccess);

    checkCudaErrors(cudaMemcpy(A, A_h, ldMax * ldMax * sizeof(half), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(B, B_h, ldMax * ldMax * sizeof(half), cudaMemcpyHostToDevice));

    for (int iter = 0; iter < 11; ++iter) {
        unsigned int aRows = (rand() % (ldMax - 200)) + 1;
        unsigned int aCols = (rand() % (ldMax - 200)) + 1;
        unsigned int bCols = (rand() % (ldMax - 200)) + 1;
        unsigned int cRows = aRows;
        unsigned int cCols = bCols;

        unsigned int lda;
        unsigned int ldb;
        unsigned int ldc;

        if (rand() % 2) {
            lda = aCols + (rand() % 150);
            ldb = bCols + (rand() % 150);
            ldc = cCols + (rand() % 150);
            TensorCoreMatrixMultiply::instance().multiplyUsingHeuristicKernelChoice(A, B, C, aRows, aCols, bCols, lda, ldb, ldc, stream);
        } else {
            lda = aCols;
            ldb = bCols;
            ldc = cCols;
            TensorCoreMatrixMultiply::instance().multiplyUsingHeuristicKernelChoice(A, B, C, aRows, aCols, bCols, stream);
        }

        matrixMultiplyCpu(A_h_float, B_h_float, C_h_float, aRows, bCols, aCols, lda, ldb, ldc);

        checkCudaErrors(cudaMemcpyAsync(CGpu_h, C, cRows * ldc * sizeof(half), cudaMemcpyDeviceToHost, stream.getStream()));
        checkCudaErrors(cudaStreamSynchronize(stream.getStream()));

        float maxDiff = aCols * 0.0125;

        for (unsigned int i = 0; i < cRows; ++i) {
            for (unsigned int j = 0; j < cCols; ++j) {
                float diff = abs(C_h_float[i * ldc + j] - float(CGpu_h[i * ldc + j]));
                if (diff >= maxDiff) {
                    printf("arows %d acols %d bcols %d\n", aRows, aCols, bCols);
                    printf("row %d col %d : CPU %f vs %f GPU\n", i, j, C_h_float[i * ldc + j], float(CGpu_h[i * ldc + j]));
                    fflush(stdout);
                }
                ASSERT_LT(diff, maxDiff);
            }
        }
    }

    checkCudaErrors(cudaFree((void *)A));
    checkCudaErrors(cudaFree((void *)B));
    checkCudaErrors(cudaFree((void *)C));

    checkCudaErrors(cudaFreeHost(A_h));
    checkCudaErrors(cudaFreeHost(B_h));
    checkCudaErrors(cudaFreeHost(CGpu_h));
    checkCudaErrors(cudaFreeHost(A_h_float));
    checkCudaErrors(cudaFreeHost(B_h_float));
    checkCudaErrors(cudaFreeHost(C_h_float));
}

TEST(TensorCoreMatrixMultiply, OptimalMutliplyWithoutWorkspaceProducesCorrectResult) {
    srand(time(NULL));

    cudaError_t cudaStatus;

    Stream stream(0);

    int ldMax = 4096;

    half *A_h;
    half *B_h;
    half *CGpu_h;

    float *A_h_float;
    float *B_h_float;
    float *C_h_float;

    checkCudaErrors(cudaHostAlloc(&A_h, ldMax * ldMax * sizeof(half), cudaHostAllocPortable));
    checkCudaErrors(cudaHostAlloc(&B_h, ldMax * ldMax * sizeof(half), cudaHostAllocPortable));
    checkCudaErrors(cudaHostAlloc(&CGpu_h, ldMax * ldMax * sizeof(half), cudaHostAllocPortable));

    checkCudaErrors(cudaHostAlloc(&A_h_float, ldMax * ldMax * sizeof(float), cudaHostAllocPortable));
    checkCudaErrors(cudaHostAlloc(&B_h_float, ldMax * ldMax * sizeof(float), cudaHostAllocPortable));
    checkCudaErrors(cudaHostAlloc(&C_h_float, ldMax * ldMax * sizeof(float), cudaHostAllocPortable));

    for (int i = 0; i < ldMax; ++i) {
        for (int j = 0; j < ldMax; ++j) {
            A_h[i * ldMax + j] = half(((rand() % 100) - 50) / 10.0f);
            A_h_float[i * ldMax + j] = (float)A_h[i * ldMax + j];
            B_h[i * ldMax + j] = half(((rand() % 100) - 50) / 10.0f);
            B_h_float[i * ldMax + j] = (float)B_h[i * ldMax + j];
        }
    }

    half *A;
    half *B;
    half *C;
    cudaStatus = cudaMalloc(&A, ldMax * ldMax * sizeof(half));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&B, ldMax * ldMax * sizeof(half));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&C, ldMax * ldMax * sizeof(half));
    assert(cudaStatus == cudaSuccess);

    checkCudaErrors(cudaMemcpy(A, A_h, ldMax * ldMax * sizeof(half), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(B, B_h, ldMax * ldMax * sizeof(half), cudaMemcpyHostToDevice));

    unsigned int aRows;
    unsigned int aCols;
    unsigned int bCols;
    unsigned int cRows;
    unsigned int cCols;
    unsigned int lda;
    unsigned int ldb;
    unsigned int ldc;

    constexpr int TEST_RUNS = 10;

    for (int iter = 0; iter < TEST_RUNS; ++iter) {
        // On iter 1, do the same multiply as on iter0
        if (iter != 1) {
            aRows = (rand() % (ldMax - 200)) + 1;
            aCols = (rand() % (ldMax - 200)) + 1;
            bCols = (rand() % 256) + 1;
            cRows = aRows;
            cCols = bCols;
        }

        if (rand() % 2) {
            lda = aCols + (rand() % 150);
            ldb = bCols + (rand() % 150);
            ldc = cCols + (rand() % 150);
            TensorCoreMatrixMultiply::instance().chooseOptimalKernel(0, aRows, aCols, bCols, lda, ldb, ldc);
            TensorCoreMatrixMultiply::instance().multiply(A, B, C, aRows, aCols, bCols, lda, ldb, ldc, stream);
        } else {
            lda = aCols;
            ldb = bCols;
            ldc = cCols;
            TensorCoreMatrixMultiply::instance().chooseOptimalKernel(0, aRows, aCols, bCols);
            TensorCoreMatrixMultiply::instance().multiply(A, B, C, aRows, aCols, bCols, stream);
        }

        matrixMultiplyCpu(A_h_float, B_h_float, C_h_float, aRows, bCols, aCols, lda, ldb, ldc);

        checkCudaErrors(cudaMemcpyAsync(CGpu_h, C, cRows * ldc * sizeof(half), cudaMemcpyDeviceToHost, stream.getStream()));
        checkCudaErrors(cudaStreamSynchronize(stream.getStream()));

        float maxDiff = aCols * 0.0125;

        for (unsigned int i = 0; i < cRows; ++i) {
            for (unsigned int j = 0; j < cCols; ++j) {
                float diff = abs(C_h_float[i * ldc + j] - float(CGpu_h[i * ldc + j]));
                if (diff >= maxDiff) {
                    printf("arows %d acols %d bcols %d\n", aRows, aCols, bCols);
                    printf("row %d col %d : CPU %f vs %f GPU\n", i, j, C_h_float[i * ldc + j], float(CGpu_h[i * ldc + j]));
                    fflush(stdout);
                }
                ASSERT_LT(diff, maxDiff);
            }
        }
    }

    checkCudaErrors(cudaFree((void *)A));
    checkCudaErrors(cudaFree((void *)B));
    checkCudaErrors(cudaFree((void *)C));

    checkCudaErrors(cudaFreeHost(A_h));
    checkCudaErrors(cudaFreeHost(B_h));
    checkCudaErrors(cudaFreeHost(CGpu_h));
    checkCudaErrors(cudaFreeHost(A_h_float));
    checkCudaErrors(cudaFreeHost(B_h_float));
    checkCudaErrors(cudaFreeHost(C_h_float));
}

TEST(TensorCoreMatrixMultiply, OptimalMutliplyWithWorkspaceProducesCorrectResult) {
    srand(time(NULL));

    cudaError_t cudaStatus;

    Stream stream(0);

    int ldMax = 4096;

    half *A_h;
    half *B_h;
    half *CGpu_h;

    float *A_h_float;
    float *B_h_float;
    float *C_h_float;

    checkCudaErrors(cudaHostAlloc(&A_h, ldMax * ldMax * sizeof(half), cudaHostAllocPortable));
    checkCudaErrors(cudaHostAlloc(&B_h, ldMax * ldMax * sizeof(half), cudaHostAllocPortable));
    checkCudaErrors(cudaHostAlloc(&CGpu_h, ldMax * ldMax * sizeof(half), cudaHostAllocPortable));

    checkCudaErrors(cudaHostAlloc(&A_h_float, ldMax * ldMax * sizeof(float), cudaHostAllocPortable));
    checkCudaErrors(cudaHostAlloc(&B_h_float, ldMax * ldMax * sizeof(float), cudaHostAllocPortable));
    checkCudaErrors(cudaHostAlloc(&C_h_float, ldMax * ldMax * sizeof(float), cudaHostAllocPortable));

    for (int i = 0; i < ldMax; ++i) {
        for (int j = 0; j < ldMax; ++j) {
            A_h[i * ldMax + j] = half(((rand() % 100) - 50) / 10.0f);
            A_h_float[i * ldMax + j] = (float)A_h[i * ldMax + j];
            B_h[i * ldMax + j] = half(((rand() % 100) - 50) / 10.0f);
            B_h_float[i * ldMax + j] = (float)B_h[i * ldMax + j];
        }
    }

    half *A;
    half *B;
    half *C;
    cudaStatus = cudaMalloc(&A, ldMax * ldMax * sizeof(half));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&B, ldMax * ldMax * sizeof(half));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&C, ldMax * ldMax * sizeof(half));
    assert(cudaStatus == cudaSuccess);

    checkCudaErrors(cudaMemcpy(A, A_h, ldMax * ldMax * sizeof(half), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(B, B_h, ldMax * ldMax * sizeof(half), cudaMemcpyHostToDevice));

    unsigned int aRows;
    unsigned int aCols;
    unsigned int bCols;
    unsigned int cRows;
    unsigned int cCols;
    unsigned int lda;
    unsigned int ldb;
    unsigned int ldc;

    constexpr int TEST_RUNS = 11;

    for (int iter = 0; iter < TEST_RUNS; ++iter) {
        // On iter 1, do the same multiply as on iter0
        if (iter != 1) {
            aRows = (rand() % (ldMax - 200)) + 1;
            aCols = (rand() % (ldMax - 200)) + 1;
            bCols = (rand() % 256) + 1;
            cRows = aRows;
            cCols = bCols;
        }

        if (rand() % 2) {
            lda = aCols + (rand() % 150);
            ldb = bCols + (rand() % 150);
            ldc = cCols + (rand() % 150);
        } else {
            lda = aCols;
            ldb = bCols;
            ldc = cCols;
        }

        TensorCoreMatrixMultiply::instance().chooseOptimalKernel(0, aRows, aCols, bCols, lda, ldb, ldc);

        unsigned int workspaceSize;
        if (aCols == lda && bCols == ldb && cCols == ldc)
            workspaceSize = TensorCoreMatrixMultiply::instance().getWorkspaceSizeInBytes(0, aRows, aCols, bCols);
        else
            workspaceSize = TensorCoreMatrixMultiply::instance().getWorkspaceSizeInBytes(0, aRows, aCols, bCols, lda, ldb, ldc);
        half *workspace_d = nullptr;
        if (workspaceSize > 0) {
            cudaStatus = cudaMalloc(&workspace_d, workspaceSize);
            assert(cudaStatus == cudaSuccess);
        }

        // printf("workspace size %d aRows %d aCols %d bCols %d lda %d ldb %d ldc %d\n", workspaceSize, aRows, aCols, bCols, lda, ldb, ldc);
        // fflush(stdout);

        if (lda == aCols && ldb == bCols && ldc == cCols) {
            TensorCoreMatrixMultiply::instance().multiply(A, B, C, workspace_d, aRows, aCols, bCols, stream);
        } else {
            TensorCoreMatrixMultiply::instance().multiply(A, B, C, workspace_d, aRows, aCols, bCols, lda, ldb, ldc, stream);
        }

        matrixMultiplyCpu(A_h_float, B_h_float, C_h_float, aRows, bCols, aCols, lda, ldb, ldc);

        checkCudaErrors(cudaMemcpyAsync(CGpu_h, C, cRows * ldc * sizeof(half), cudaMemcpyDeviceToHost, stream.getStream()));
        cudaStatus = cudaStreamSynchronize(stream.getStream());
        assert(cudaStatus == cudaSuccess);

        if (workspaceSize > 0)
            checkCudaErrors(cudaFree((void *)workspace_d));

        float maxDiff = aCols * 0.0125;

        for (unsigned int i = 0; i < cRows; ++i) {
            for (unsigned int j = 0; j < cCols; ++j) {
                float diff = abs(C_h_float[i * ldc + j] - float(CGpu_h[i * ldc + j]));
                if (diff >= maxDiff) {
                    printf("arows %d acols %d bcols %d\n", aRows, aCols, bCols);
                    printf("row %d col %d : CPU %f vs %f GPU\n", i, j, C_h_float[i * ldc + j], float(CGpu_h[i * ldc + j]));
                    fflush(stdout);
                }
                ASSERT_LT(diff, maxDiff);
            }
        }
    }

    checkCudaErrors(cudaFree((void *)A));
    checkCudaErrors(cudaFree((void *)B));
    checkCudaErrors(cudaFree((void *)C));

    checkCudaErrors(cudaFreeHost(A_h));
    checkCudaErrors(cudaFreeHost(B_h));
    checkCudaErrors(cudaFreeHost(CGpu_h));
    checkCudaErrors(cudaFreeHost(A_h_float));
    checkCudaErrors(cudaFreeHost(B_h_float));
    checkCudaErrors(cudaFreeHost(C_h_float));
}

TEST(TensorCoreMatrixMultiply, MultiThreadedOptimizationWorks) {
    srand(time(NULL));

    omp_set_num_threads(10);

    KernelRequirement kernelRequirement0;
    kernelRequirement0.gpuType = MachineEvaluator::instance().getGpuType();
    kernelRequirement0.rowsA = 100;
    kernelRequirement0.colsA = 200;
    kernelRequirement0.colsB = 300;
    kernelRequirement0.ldA = 100;
    kernelRequirement0.ldB = 200;
    kernelRequirement0.ldC = 300;
    kernelRequirement0.allowWorkspace = true;

    KernelRequirement kernelRequirement1;
    kernelRequirement1.gpuType = MachineEvaluator::instance().getGpuType();
    kernelRequirement1.rowsA = 400;
    kernelRequirement1.colsA = 500;
    kernelRequirement1.colsB = 600;
    kernelRequirement0.ldA = 400;
    kernelRequirement0.ldB = 500;
    kernelRequirement0.ldC = 600;
    kernelRequirement0.allowWorkspace = false;

    TensorCoreMatrixMultiply::instance().startingMultiThreadedKernelOptimization();

#pragma omp parallel for schedule(static, 1)
    for (int i = 0; i < 1000; ++i) {
        if (rand() % 2)
            TensorCoreMatrixMultiply::instance().chooseOptimalKernel(0, 100, 200, 300);
        else
            TensorCoreMatrixMultiply::instance().chooseOptimalKernel(0, 400, 500, 600);
    }

    TensorCoreMatrixMultiply::instance().finishedMultiThreadedKernelOptimization();

    ASSERT_GT(TensorCoreMatrixMultiply::instance().getOptimalKernelTime(0, 100, 200, 300, true), 0.0f);
    ASSERT_GT(TensorCoreMatrixMultiply::instance().getOptimalKernelTime(0, 100, 200, 300, false), 0.0f);
    ASSERT_GT(TensorCoreMatrixMultiply::instance().getOptimalKernelTime(0, 400, 500, 600, true), 0.0f);
    ASSERT_GT(TensorCoreMatrixMultiply::instance().getOptimalKernelTime(0, 400, 500, 600, false), 0.0f);
}

TEST(TensorCoreMatrixMultiply, AllKernelsGiveCorrectAnswer) {
    srand(time(NULL));

    if (omp_get_num_procs() > 1)
        omp_set_num_threads(omp_get_num_procs() - 1);

    volatile std::atomic<bool> breakout;
    breakout = false;

    vector<KernelWithSpec> kernels = TensorCoreMatrixMultiplyTestHelper::instance().getAllKernels();
    vector<KernelWithSpec> someKernels;

    int numKernels = kernels.size();

#pragma omp parallel for schedule(static, 3)
    for (int kernelIndex = 0; kernelIndex < numKernels; ++kernelIndex) {
        cudaError_t cudaStatus;

        Stream stream(0);

        int print = false;
        int ldMax = 1000;

        half *A_h;
        half *B_h;
        half *CGpu_h;

        float *A_h_float;
        float *B_h_float;
        float *C_h_float;

        checkCudaErrors(cudaHostAlloc(&A_h, ldMax * ldMax * sizeof(half), cudaHostAllocPortable));
        checkCudaErrors(cudaHostAlloc(&B_h, ldMax * ldMax * sizeof(half), cudaHostAllocPortable));
        checkCudaErrors(cudaHostAlloc(&CGpu_h, ldMax * ldMax * sizeof(half), cudaHostAllocPortable));

        checkCudaErrors(cudaHostAlloc(&A_h_float, ldMax * ldMax * sizeof(float), cudaHostAllocPortable));
        checkCudaErrors(cudaHostAlloc(&B_h_float, ldMax * ldMax * sizeof(float), cudaHostAllocPortable));
        checkCudaErrors(cudaHostAlloc(&C_h_float, ldMax * ldMax * sizeof(float), cudaHostAllocPortable));

        for (int i = 0; i < ldMax; ++i) {
            for (int j = 0; j < ldMax; ++j) {
                A_h[i * ldMax + j] = half(((rand() % 100) - 50) / 10.0f);
                A_h_float[i * ldMax + j] = (float)A_h[i * ldMax + j];
                B_h[i * ldMax + j] = half(((rand() % 100) - 50) / 10.0f);
                B_h_float[i * ldMax + j] = (float)B_h[i * ldMax + j];
            }
        }

        half *A;
        half *B;
        half *C;
        cudaStatus = cudaMalloc(&A, ldMax * ldMax * sizeof(half));
        assert(cudaStatus == cudaSuccess);
        cudaStatus = cudaMalloc(&B, ldMax * ldMax * sizeof(half));
        assert(cudaStatus == cudaSuccess);
        cudaStatus = cudaMalloc(&C, ldMax * ldMax * sizeof(half));
        assert(cudaStatus == cudaSuccess);

        checkCudaErrors(cudaMemcpy(A, A_h, ldMax * ldMax * sizeof(half), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(B, B_h, ldMax * ldMax * sizeof(half), cudaMemcpyHostToDevice));

        constexpr int TESTS_PER_KERNEL = 10;

        for (int i = 0; i < TESTS_PER_KERNEL; ++i) {
            int aRows, aCols, bCols;
            int lda, ldb, ldc;
            if (i < TESTS_PER_KERNEL / 2) {
                aRows = (rand() % 100) + 1;
                aCols = (rand() % 100) + 1;
                bCols = (rand() % 100) + 1;
            } else {
                aRows = (rand() % (ldMax - 50)) + 1;
                aCols = (rand() % (ldMax - 50)) + 1;
                bCols = (rand() % (ldMax - 50)) + 1;
            }
            // Enforce kernel restrictions
            if (kernels[kernelIndex].aColSizeModulusRequirement > 1 && kernels[kernelIndex].bRowSizeModulusRequirement > 1)
                assert(kernels[kernelIndex].aColSizeModulusRequirement % kernels[kernelIndex].bRowSizeModulusRequirement == 0 ||
                       kernels[kernelIndex].bRowSizeModulusRequirement % kernels[kernelIndex].aColSizeModulusRequirement == 0);
            if (kernels[kernelIndex].aRowSizeModulusRequirement > 1) {
                aRows = (aRows / kernels[kernelIndex].aRowSizeModulusRequirement) * kernels[kernelIndex].aRowSizeModulusRequirement;
                if (aRows == 0)
                    aRows = kernels[kernelIndex].aRowSizeModulusRequirement;
            }
            if (kernels[kernelIndex].aColSizeModulusRequirement > 1) {
                aCols = (aCols / kernels[kernelIndex].aColSizeModulusRequirement) * kernels[kernelIndex].aColSizeModulusRequirement;
                if (aCols == 0)
                    aCols = kernels[kernelIndex].aColSizeModulusRequirement;
            }
            if (kernels[kernelIndex].bRowSizeModulusRequirement > 1) {
                aCols = (aCols / kernels[kernelIndex].bRowSizeModulusRequirement) * kernels[kernelIndex].bRowSizeModulusRequirement;
                if (aCols == 0)
                    aCols = kernels[kernelIndex].bRowSizeModulusRequirement;
            }
            if (kernels[kernelIndex].bColSizeModulusRequirement > 1) {
                bCols = (bCols / kernels[kernelIndex].bColSizeModulusRequirement) * kernels[kernelIndex].bColSizeModulusRequirement;
                if (bCols == 0)
                    bCols = kernels[kernelIndex].bColSizeModulusRequirement;
            }
            lda = (rand() % 2) ? aCols : aCols + (rand() % 50);
            ldb = (rand() % 2) ? bCols : bCols + (rand() % 50);
            ldc = (rand() % 2) ? bCols : bCols + (rand() % 50);
            if (kernels[kernelIndex].aColSizeModulusRequirement > 1)
                lda = (lda / kernels[kernelIndex].aColSizeModulusRequirement) * kernels[kernelIndex].aColSizeModulusRequirement;
            if (kernels[kernelIndex].bColSizeModulusRequirement > 1)
                ldb = (ldb / kernels[kernelIndex].bColSizeModulusRequirement) * kernels[kernelIndex].bColSizeModulusRequirement;

            unsigned int bRows = aCols;
            unsigned int cRows = aRows;
            unsigned int cCols = bCols;

            matrixMultiplyCpu(A_h_float, B_h_float, C_h_float, aRows, bCols, aCols, lda, ldb, ldc);

            if (print) {
                printf("A:\n");
                printMatrix(A_h_float, aRows, aCols, lda);
                printf("\n\n");

                printf("B:\n");
                printMatrix(B_h_float, bRows, bCols, ldb);
                printf("\n\n");

                printf("C:\n");
                printMatrixWide(C_h_float, cRows, cCols, ldc);
                printf("\n\n");
            }

            KernelRequirement kernelRequirement;
            kernelRequirement.gpuType = MachineEvaluator::instance().getGpuType();
            kernelRequirement.rowsA = aRows;
            kernelRequirement.colsA = aCols;
            kernelRequirement.colsB = bCols;
            kernelRequirement.ldA = lda;
            kernelRequirement.ldB = ldb;
            kernelRequirement.ldC = ldc;
            kernelRequirement.allowWorkspace = true;
            unsigned int workspaceSize = kernels[kernelIndex].getWorkspaceSize(kernelRequirement);
            half *workspace_d = nullptr;
            if (workspaceSize > 0) {
                cudaStatus = cudaMalloc(&workspace_d, workspaceSize);
                assert(cudaStatus == cudaSuccess);
            }

            kernels[kernelIndex].executeKernel(A, B, C, workspace_d, aRows, aCols, bCols, lda, ldb, ldc, stream);

            checkCudaErrors(cudaMemcpyAsync(CGpu_h, C, cRows * ldc * sizeof(half), cudaMemcpyDeviceToHost, stream.getStream()));
            checkCudaErrors(cudaStreamSynchronize(stream.getStream()));

            if (workspaceSize > 0)
                checkCudaErrors(cudaFree((void *)workspace_d));

            if (print) {
                printMatrixWide(CGpu_h, cRows, cCols, ldc);
                printf("\n\n");
            }

            float maxDiff = aCols * 0.0125;

            for (unsigned int i = 0; i < cRows; ++i) {
                for (unsigned int j = 0; j < cCols; ++j) {
                    float diff = abs(C_h_float[i * ldc + j] - float(CGpu_h[i * ldc + j]));
                    if (diff >= maxDiff) {
                        printf("arows %d acols %d bcols %d\n", aRows, aCols, bCols);
                        printf("row %d col %d : CPU %f vs %f GPU\n", i, j, C_h_float[i * ldc + j], float(CGpu_h[i * ldc + j]));
                        printf("kernel %d failed\n", (int)kernels[kernelIndex].id);
                        fflush(stdout);
                        breakout = true;
                    }
                    if (breakout)
                        break;
                }
                if (breakout)
                    break;
            }
            if (breakout)
                break;
        }

        checkCudaErrors(cudaFree((void *)A));
        checkCudaErrors(cudaFree((void *)B));
        checkCudaErrors(cudaFree((void *)C));

        checkCudaErrors(cudaFreeHost(A_h));
        checkCudaErrors(cudaFreeHost(B_h));
        checkCudaErrors(cudaFreeHost(CGpu_h));
        checkCudaErrors(cudaFreeHost(A_h_float));
        checkCudaErrors(cudaFreeHost(B_h_float));
        checkCudaErrors(cudaFreeHost(C_h_float));
    }

    ASSERT_FALSE(breakout);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
