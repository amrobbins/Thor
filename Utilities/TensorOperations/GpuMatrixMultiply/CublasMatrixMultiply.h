#pragma once

#include "Utilities/TensorOperations/GpuMatrixMultiply/CublasKernel.h"
#include "Utilities/TensorOperations/GpuMatrixMultiply/CublasKernelRequirement.h"

#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/ComputeTopology/MachineEvaluator.h"

#include <cublas.h>
#include <cublasLt.h>
#include <cuda.h>
#include <cuda_fp16.h>

#include <assert.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <unordered_map>
#include <utility>

using std::exception;
using std::make_pair;
using std::mutex;
using std::pair;
using std::unordered_map;

/**
 * CublasMatrixMultiply is a singleton object that can find the optimal matrix multiply kernel for a matrix multiply operation
 * of a given set of dimensions on a specific type of GPU.
 *
 * Once the optimal kernel is found, it will be use on all subsequent matching matrix multiply calls.
 *
 * It is not required to first evaluate the optimal kernel, when the optimal kernel is not known then a pretty-good-fit kernel can be used.
 *
 * The recommendation is that if you plan to do a matrix multiply of a certain size many times, first find the optimal kernel.
 *
 * The singleton object is accessed as CublasMatrixMultiply::instance().multiply(...)
 *
 * Note: By having the member functions be non-static, they cannot be called unless the constructor has already been called.
 *       Member variables should be non-static so that they are only initialized upon creation of the singleton instance.
 */
class CublasMatrixMultiply {
   public:
    static CublasMatrixMultiply &instance() {
        static CublasMatrixMultiply singletonInstance;  // Guaranteed to be destroyed. Instantiated on first use.
        return singletonInstance;
    }

    virtual ~CublasMatrixMultiply() {}

    // fills C as C = A * B, where A, B and C are all matrices whose memory is allocated on the GPU that will be performing the computation.
    //
    // accumulate=true computes C += A * B. accumulate=false computes C = A * B.
    //
    // Prerequisites to using this version of multiply:
    //  1. You have previously called chooseOptimalKernel for a matrix multiply with the same dimensions (i.e. A rows, A cols, B cols).
    //  2. You provide the workspace, which is memory allocated on the target gpu of size getWorkspaceSizeInBytes(same parameters).
    //     When the workspace is 0 bytes, set it to nullptr.
    //
    //  Note: don't allocate a workspace before each call to multiply, you would be better off using the version of multiply that doesn't
    //  use a workspace in terms of performance. Use workspaces only when you can allocate it once and use it a number of times, i.e. when
    //  you have to do a matrix multiplication of the same size over and over.
    void multiply(Tensor A,
                  Tensor B,
                  Tensor C,
                  Tensor workspace,
                  const int32_t A_rows,
                  const int32_t A_cols,
                  const int32_t B_rows,
                  const int32_t B_cols,
                  bool transposeA,
                  bool transposeB,
                  const bool accumulate,
                  const TensorDescriptor::DataType ABCDataType,
                  Stream stream) {
        int ldC = transposeB == false ? B_cols : B_rows;
        multiply(A,
                 B,
                 C,
                 workspace,
                 A_rows,
                 A_cols,
                 B_rows,
                 B_cols,
                 A_cols,
                 B_cols,
                 ldC,
                 transposeA,
                 transposeB,
                 accumulate,
                 ABCDataType,
                 stream);
    }

    // This variant allows non-packed matrices
    void multiply(Tensor A,
                  Tensor B,
                  Tensor C,
                  Tensor workspace,
                  const int32_t A_rows,
                  const int32_t A_cols,
                  const int32_t B_rows,
                  const int32_t B_cols,
                  // Leading dimension of A, i.e. number of elements (not bytes) that separate the beginning of two adjacent rows in
                  // memory. Some slots at the end of a row may be unused.
                  const int32_t ld_A,
                  const int32_t ld_B,
                  const int32_t ld_C,
                  bool transposeA,
                  bool transposeB,
                  const bool accumulate,
                  const TensorDescriptor::DataType ABCDataType,
                  Stream stream);

    // fills C as C = A * B, where A, B and C are all matrices whose memory is allocated on the GPU that will be performing the computation.
    //
    // Prerequisites to using this version of multiply:
    //  1. You have previously called chooseOptimalKernel for a matrix multiply with the same dimensions (i.e. A rows, A cols, B cols).
    void multiply(Tensor A,
                  Tensor B,
                  Tensor C,
                  const int32_t A_rows,
                  const int32_t A_cols,
                  const int32_t B_rows,
                  const int32_t B_cols,
                  bool transposeA,
                  bool transposeB,
                  const bool accumulate,
                  const TensorDescriptor::DataType ABCDataType,
                  Stream stream) {
        int ldC = transposeB == false ? B_cols : B_rows;
        multiply(A, B, C, A_rows, A_cols, B_rows, B_cols, A_cols, B_cols, ldC, transposeA, transposeB, accumulate, ABCDataType, stream);
    }

    // This variant allows non-packed matrices
    void multiply(Tensor A,
                  Tensor B,
                  Tensor C,
                  const int32_t A_rows,
                  const int32_t A_cols,
                  const int32_t B_rows,
                  const int32_t B_cols,
                  // Leading dimension of A, i.e. number of elements (not bytes) that separate the beginning of two adjacent rows in
                  // memory. Some slots at the end of a row may be unused.
                  const int32_t ld_A,
                  const int32_t ld_B,
                  const int32_t ld_C,
                  bool transposeA,
                  bool transposeB,
                  const bool accumulate,
                  const TensorDescriptor::DataType ABCDataType,
                  Stream stream);

    // fills C as C = A * B, where A, B and C are all matrices whose memory is allocated on the GPU that will be performing the computation.
    //
    // This variant performs a multiplication and does not use a workspace. If the optimal kernel is not already known,
    // then a guess is made at what is a good kernel and the multiplication is performed using it.
    //
    // This version of multiply should be used when you can't predict in advance the dimensions of the matrix multiplications that you will
    // need to do.
    //
    // Performance is not as good as the other 2 variants, but you are getting a GPU tensor core matrix multiply kernel that fits your
    // computation size pretty well, so it is very fast, i.e. sub millisecond for large matrices.
    void multiplyUsingHeuristicKernelChoice(Tensor A,
                                            Tensor B,
                                            Tensor C,
                                            const int32_t A_rows,
                                            const int32_t A_cols,
                                            const int32_t B_rows,
                                            const int32_t B_cols,
                                            bool transposeA,
                                            bool transposeB,
                                            const bool accumulate,
                                            const TensorDescriptor::DataType ABCDataType,
                                            Stream stream) {
        int ldC = transposeB == false ? B_cols : B_rows;
        multiplyUsingHeuristicKernelChoice(
            A, B, C, A_rows, A_cols, B_rows, B_cols, A_cols, B_cols, ldC, transposeA, transposeB, accumulate, ABCDataType, stream);
    }

    // This variant allows non-packed matrices
    void multiplyUsingHeuristicKernelChoice(Tensor A,
                                            Tensor B,
                                            Tensor C,
                                            const int32_t A_rows,
                                            const int32_t A_cols,
                                            const int32_t B_rows,
                                            const int32_t B_cols,
                                            // Leading dimension of A, i.e. number of elements (not bytes) that separate the beginning of
                                            // two adjacent rows in memory. Some slots at the end of a row may be unused.
                                            const int32_t ld_A,
                                            const int32_t ld_B,
                                            const int32_t ld_C,
                                            bool transposeA,
                                            bool transposeB,
                                            const bool accumulate,
                                            const TensorDescriptor::DataType ABCDataType,
                                            Stream stream);

    // Find any gpu of the specififed type and measure the optimal kernel for the matrix multiply operation
    // Find any gpu of the specififed type and measure the optimal kernel for the matrix multiply operation
    inline void chooseOptimalKernel(string gpuType,
                                    int rowsA,
                                    int colsA,
                                    int rowsB,
                                    int colsB,
                                    bool transposeA,
                                    bool transposeB,
                                    TensorDescriptor::DataType ABCDataType,
                                    bool printResults = false) {
        int ldC = transposeB == false ? colsB : rowsB;
        chooseOptimalKernel(gpuType, rowsA, colsA, rowsB, colsB, colsA, colsB, ldC, transposeA, transposeB, ABCDataType, printResults);
    }

    inline void chooseOptimalKernel(string gpuType,
                                    int rowsA,
                                    int colsA,
                                    int rowsB,
                                    int colsB,
                                    int ldA,
                                    int ldB,
                                    int ldC,
                                    bool transposeA,
                                    bool transposeB,
                                    TensorDescriptor::DataType ABCDataType,
                                    bool printResults = false) {
        // Find a gpu of the proper type or fail
        int gpuNum = -1;
        for (int i = 0; i < (int)MachineEvaluator::instance().getNumGpus(); ++i) {
            if (MachineEvaluator::instance().getGpuType(i) == gpuType) {
                gpuNum = i;
                break;
            }
        }
        assert(gpuNum >= 0);

        chooseOptimalKernel(gpuNum, rowsA, colsA, rowsB, colsB, ldA, ldB, ldC, transposeA, transposeB, ABCDataType, printResults);
    }

    // Find the optimal kernel for the matrix multiply operation for type of the gpuNum GPU, by measuring it specifically on that gpu.
    //
    // This is can be used to optimize kernels across multiple GPU's, but you must start and end this process by calling
    // startingMultiThreadedOptimization and finishedMultiThreadedOptimization, which adds locking around reading and writing
    // of the map that holds the optimal kernels. This is only needed during optimization because then there is a writer,
    // multiple readers are fine lock free so locks are only used during multi-threaded optimization.
    //
    // Note: never run more than one optimization at a time on any gpu.
    inline void chooseOptimalKernel(int gpuNum,
                                    int rowsA,
                                    int colsA,
                                    int rowsB,
                                    int colsB,
                                    bool transposeA,
                                    bool transposeB,
                                    TensorDescriptor::DataType ABCDataType,
                                    bool printResults = false) {
        int ldC = transposeB == false ? colsB : rowsB;
        chooseOptimalKernel(gpuNum, rowsA, colsA, rowsB, colsB, colsA, colsB, ldC, transposeA, transposeB, ABCDataType, printResults);
    }

    void chooseOptimalKernel(int gpuNum,
                             int rowsA,
                             int colsA,
                             int rowsB,
                             int colsB,
                             int ldA,
                             int ldB,
                             int ldC,
                             bool transposeA,
                             bool transposeB,
                             TensorDescriptor::DataType ABCDataType,
                             bool printResults = false);

    inline unsigned int getWorkspaceSizeInBytes(int gpuNum,
                                                int rowsA,
                                                int colsA,
                                                int rowsB,
                                                int colsB,
                                                bool transposeA,
                                                bool transposeB,
                                                TensorDescriptor::DataType ABCDataType,
                                                bool &kernelWillRunOnGpu) {
        int ldC = transposeB == false ? colsB : rowsB;
        return getWorkspaceSizeInBytes(
            gpuNum, rowsA, colsA, rowsB, colsB, colsA, colsB, ldC, transposeA, transposeB, ABCDataType, kernelWillRunOnGpu);
    }

    unsigned int getWorkspaceSizeInBytes(int gpuNum,
                                         int rowsA,
                                         int colsA,
                                         int rowsB,
                                         int colsB,
                                         int ldA,
                                         int ldB,
                                         int ldC,
                                         bool transposeA,
                                         bool transposeB,
                                         TensorDescriptor::DataType ABCDataType,
                                         bool &kernelWillRunOnGpu);

    // getOptimalKernelTime(...) will give you the average time the kernel took when chooseOptimalKernel was called for the
    // matrix multiply operation with those dimensions on that GPU. It is an error to call this for an operation where the
    // optimal kernel was not measured.
    //
    // This can be useful if you need to balance the execution time between multiple GPUs.
    double getOptimalKernelTime(string gpuType,
                                int rowsA,
                                int colsA,
                                int rowsB,
                                int colsB,
                                int ldA,
                                int ldB,
                                int ldC,
                                bool transposeA,
                                bool transposeB,
                                TensorDescriptor::DataType ABCDataType,
                                bool workspaceAllowed);

    double getOptimalKernelTime(int gpuNum,
                                int rowsA,
                                int colsA,
                                int rowsB,
                                int colsB,
                                int ldA,
                                int ldB,
                                int ldC,
                                bool transposeA,
                                bool transposeB,
                                TensorDescriptor::DataType ABCDataType,
                                bool workspaceAllowed);

   private:
    unordered_map<CublasKernelRequirement, CublasKernel> optimalKernels;
    unordered_map<CublasKernelRequirement, cublasLtMatmulAlgo_t> knownHeuristicAlgorithms;

    mutex mtx;

    class Youreusingitwrong : public exception {
       public:
        Youreusingitwrong(string message) { this->message = message; }

        virtual const char *what() const throw() { return message.c_str(); }

       private:
        string message;
    };

    CublasMatrixMultiply() {}

    cudaDataType_t mapToCublasDataType(TensorDescriptor::DataType dataType);

    bool chooseOptimalKernel(int gpuNum,
                             int rowsA,
                             int colsA,
                             int rowsB,
                             int colsB,
                             int ldA,
                             int ldB,
                             int ldC,
                             bool transposeA,
                             bool transposeB,
                             TensorDescriptor::DataType ABCDataType,
                             bool allowWorkspaces,
                             bool printResults);

    void getSupportedCublasAlgorithms(const OperationType &operationType,
                                      vector<cublasLtMatmulAlgo_t> &supportedAlgorithms,
                                      vector<int> &supportedAlgorithmIds,
                                      CublasKernelRequirement cublasKernelRequirement,
                                      int gpuNum);
    vector<cublasLtMatmulTile_t> getSupportedTileSizes(cublasLtMatmulAlgo_t algo);
    bool isSplitKSupported(cublasLtMatmulAlgo_t algo);
    uint32_t getReductionSupportMask(cublasLtMatmulAlgo_t algo);
    int getSwizzleMaxValue(cublasLtMatmulAlgo_t algo);
    int getCustomKernelOptionMaxValue(cublasLtMatmulAlgo_t algo);

    void multiply(Tensor A,
                  Tensor B,
                  Tensor C,
                  Optional<Tensor> workspace,
                  const int32_t A_rows,
                  const int32_t A_cols,
                  const int32_t B_rows,
                  const int32_t B_cols,
                  // Leading dimension of A, i.e. number of elements (not bytes) that separate the beginning of two
                  // adjacent rows in memory. Some slots at the end of a row may be unused.
                  const int32_t ld_A,
                  const int32_t ld_B,
                  const int32_t ld_C,
                  bool transposeA,
                  bool transposeB,
                  const bool accumulate,
                  const TensorDescriptor::DataType ABCDataType,
                  Stream stream);
};
