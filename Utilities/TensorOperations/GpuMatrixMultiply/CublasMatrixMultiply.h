#pragma once

#include "Utilities/TensorOperations/GpuMatrixMultiply/CublasKernel.h"
#include "Utilities/TensorOperations/GpuMatrixMultiply/CublasKernelRequirement.h"

#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/ComputeTopology/MachineEvaluator.h"

#include <cublasLt.h>
#include <cublas_v2.h>
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

namespace ThorImplementation {

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
    // with negate=true:
    //   accumulate=true computes C -= A * B. accumulate=false computes C = -A * B.
    //
    // Prerequisites to using this version of multiply:
    //  1. You have previously called chooseOptimalMatrixMultiplyKernel for a matrix multiply with the same dimensions (i.e. A rows, A cols,
    //  B cols).
    //  2. You provide the workspace, which is memory allocated on the target gpu of size getWorkspaceSizeInBytes(same parameters).
    //     When the workspace is 0 bytes, set it to nullptr.
    //
    //  Note: don't allocate a workspace before each call to multiply, you would be better off using the version of multiply that doesn't
    //  use a workspace in terms of performance. Use workspaces only when you can allocate it once and use it a number of times, i.e. when
    //  you have to do a matrix multiplication of the same size over and over.
    //
    // Note that the number of columns in an input matrix does not have to be the same size as its leading dimension i.e. A.getDimensions[0]
    // in this case the ldA/ldB etc facitilies of GEMM are used. The matrix size is instead specified by the number of columns.
    void multiply(Tensor A,
                  Tensor B,
                  Tensor C,
                  Optional<Tensor> workspace,
                  const int32_t A_rows,
                  const int32_t A_cols,
                  const int32_t B_rows,
                  const int32_t B_cols,
                  bool transposeA,
                  bool transposeB,
                  const bool accumulate,
                  const bool negate,
                  const TensorDescriptor::DataType ABCDataType,
                  Stream stream);

    // This exposes the full GEMM functionality using an optimal kernel
    // D = alpha*(A*B) + beta*(C)
    // C and D may be the same tensor. It should be this way for beta = 0 because then no tensor will be loaded for the addition stage.
    // When C and D are the same tensor transposeC must be false:
    // https://docs.nvidia.com/cuda/archive/10.2/cublas/index.html#cublasLtMatmulDescAttributes_t
    void gemm(Tensor A,
              Tensor B,
              Tensor C,
              Tensor D,
              Optional<Tensor> workspace,
              const int32_t A_rows,
              const int32_t A_cols,
              const int32_t B_rows,
              const int32_t B_cols,
              // Leading dimension of A, i.e. number of elements (not bytes) that separate the beginning of two adjacent rows in
              // memory. Some slots at the end of a row may be unused.
              bool transposeA,
              bool transposeB,
              bool transposeC,
              float alpha,
              float beta,
              const TensorDescriptor::DataType ABCDDataType,
              Stream stream);

    // fills C as C = A * B, where A, B and C are all matrices whose memory is allocated on the GPU that will be performing the computation.
    //
    // This variant performs a multiplication and does not use a workspace. If the optimal kernel is not already known,
    // then a guess is made at what is a good kernel and the multiplication is performed using it.
    //
    // This version of multiply should be used when you can't predict in advance the dimensions of the matrix multiplications that you will
    // need to do.
    //
    // Performance is not as good as the other variant, but you are getting a GPU tensor core matrix multiply kernel that fits your
    // computation size and hardware pretty well, so it is very fast, i.e. sub millisecond for large matrices.
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
                                            const bool negate,
                                            const TensorDescriptor::DataType ABCDataType,
                                            Stream stream);

    // This exposes the full GEMM functionality using an optimal kernel
    // D = alpha*(A*B) + beta*(C)
    // C and D may be the same tensor. It should be this way for beta = 0 because then no tensor will be loaded for the addition stage.
    // When C and D are the same tensor transposeC must be false:
    // https://docs.nvidia.com/cuda/archive/10.2/cublas/index.html#cublasLtMatmulDescAttributes_t
    //
    // The uses a kernel that will fit the hardware well, though perhaps not the optimal kernel. For that we need to test a bunch
    // of them and measure the speed. Still this is way faster than any non-cublas based GEMM, as an understatement.
    void gemmUsingHeuristicKernelChoice(Tensor A,
                                        Tensor B,
                                        Tensor C,
                                        Tensor D,
                                        const int32_t A_rows,
                                        const int32_t A_cols,
                                        const int32_t B_rows,
                                        const int32_t B_cols,
                                        // Leading dimension of A, i.e. number of elements (not bytes) that separate the beginning of two
                                        // adjacent rows in memory. Some slots at the end of a row may be unused.
                                        bool transposeA,
                                        bool transposeB,
                                        bool transposeC,
                                        float alpha,
                                        float beta,
                                        const TensorDescriptor::DataType ABCDDataType,
                                        Stream stream);

    // Find any gpu of the specififed type and measure the optimal kernel for the matrix multiply operation
    // Find any gpu of the specififed type and measure the optimal kernel for the matrix multiply operation
    inline void chooseOptimalMatrixMultiplyKernel(std::string gpuType,
                                                  int rowsA,
                                                  int colsA,
                                                  int rowsB,
                                                  int colsB,
                                                  bool transposeA,
                                                  bool transposeB,
                                                  TensorDescriptor::DataType ABCDataType,
                                                  bool printResults = false) {
        int ldC = transposeB == false ? colsB : rowsB;
        chooseOptimalMatrixMultiplyKernel(
            gpuType, rowsA, colsA, rowsB, colsB, colsA, colsB, ldC, transposeA, transposeB, ABCDataType, printResults);
    }

    inline void chooseOptimalMatrixMultiplyKernel(std::string gpuType,
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
        int ldD = ldC;
        chooseOptimalGemmKernel(
            gpuType, rowsA, colsA, rowsB, colsB, ldA, ldB, ldC, ldD, transposeA, transposeB, false, ABCDataType, printResults);
    }

    inline void chooseOptimalGemmKernel(std::string gpuType,
                                        int rowsA,
                                        int colsA,
                                        int rowsB,
                                        int colsB,
                                        int ldA,
                                        int ldB,
                                        int ldC,
                                        int ldD,
                                        bool transposeA,
                                        bool transposeB,
                                        bool transposeC,
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

        chooseOptimalGemmKernel(
            gpuNum, rowsA, colsA, rowsB, colsB, ldA, ldB, ldC, ldD, transposeA, transposeB, transposeC, ABCDataType, printResults);
    }

    // Find the optimal kernel for the matrix multiply operation for type of the gpuNum GPU, by measuring it specifically on that gpu.
    //
    // This is can be used to optimize kernels across multiple GPU's, but you must start and end this process by calling
    // startingMultiThreadedOptimization and finishedMultiThreadedOptimization, which adds locking around reading and writing
    // of the map that holds the optimal kernels. This is only needed during optimization because then there is a writer,
    // multiple readers are fine lock free so locks are only used during multi-threaded optimization.
    //
    // Note: never run more than one optimization at a time on any gpu.
    inline void chooseOptimalMatrixMultiplyKernel(int gpuNum,
                                                  int rowsA,
                                                  int colsA,
                                                  int rowsB,
                                                  int colsB,
                                                  bool transposeA,
                                                  bool transposeB,
                                                  TensorDescriptor::DataType ABCDataType,
                                                  bool printResults = false) {
        int ldC = transposeB == false ? colsB : rowsB;
        chooseOptimalMatrixMultiplyKernel(
            gpuNum, rowsA, colsA, rowsB, colsB, colsA, colsB, ldC, transposeA, transposeB, ABCDataType, printResults);
    }

    inline void chooseOptimalMatrixMultiplyKernel(int gpuNum,
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
        int ldD = ldC;
        chooseOptimalGemmKernel(
            gpuNum, rowsA, colsA, rowsB, colsB, ldA, ldB, ldC, ldD, transposeA, transposeB, false, ABCDataType, printResults);
    }

    void chooseOptimalGemmKernel(int gpuNum,
                                 int rowsA,
                                 int colsA,
                                 int rowsB,
                                 int colsB,
                                 int ldA,
                                 int ldB,
                                 int ldC,
                                 int ldD,
                                 bool transposeA,
                                 bool transposeB,
                                 bool transposeC,
                                 TensorDescriptor::DataType ABCDataType,
                                 bool printResults = false);

    inline unsigned int getMatrixMultiplyWorkspaceSizeInBytes(int gpuNum,
                                                              int rowsA,
                                                              int colsA,
                                                              int rowsB,
                                                              int colsB,
                                                              bool transposeA,
                                                              bool transposeB,
                                                              TensorDescriptor::DataType ABCDataType,
                                                              bool &kernelWillRunOnGpu) {
        int ldC = transposeB == false ? colsB : rowsB;
        return getMatrixMultiplyWorkspaceSizeInBytes(
            gpuNum, rowsA, colsA, rowsB, colsB, colsA, colsB, ldC, transposeA, transposeB, ABCDataType, kernelWillRunOnGpu);
    }

    inline unsigned int getMatrixMultiplyWorkspaceSizeInBytes(int gpuNum,
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
                                                              bool &kernelWillRunOnGpu) {
        int ldD = ldC;  // They are the same memory
        return getGemmWorkspaceSizeInBytes(
            gpuNum, rowsA, colsA, rowsB, colsB, ldA, ldB, ldC, ldD, transposeA, transposeB, false, ABCDataType, kernelWillRunOnGpu);
    }

    unsigned int getGemmWorkspaceSizeInBytes(int gpuNum,
                                             int rowsA,
                                             int colsA,
                                             int rowsB,
                                             int colsB,
                                             int ldA,
                                             int ldB,
                                             int ldC,
                                             int ldD,
                                             bool transposeA,
                                             bool transposeB,
                                             bool transposeC,
                                             TensorDescriptor::DataType ABCDataType,
                                             bool &kernelWillRunOnGpu);

    // getOptimalKernelTime(...) will give you the average time the kernel took when chooseOptimalMatrixMultiplyKernel was called for the
    // matrix multiply operation with those dimensions on that GPU. It is an error to call this for an operation where the
    // optimal kernel was not measured.
    //
    // This can be useful if you need to balance the execution time between multiple GPUs.
    double getOptimalKernelTime(std::string gpuType,
                                int rowsA,
                                int colsA,
                                int rowsB,
                                int colsB,
                                int ldA,
                                int ldB,
                                int ldC,
                                int ldD,
                                bool transposeA,
                                bool transposeB,
                                bool transposeC,
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
                                int ldD,
                                bool transposeA,
                                bool transposeB,
                                bool transposeC,
                                TensorDescriptor::DataType ABCDataType,
                                bool workspaceAllowed);

   private:
    static const float ALPHA_NO_SCALE;
    static const float ALPHA_NEGATE;
    static const float BETA_ACCUMULATE;
    static const float BETA_CLEAR;

    std::unordered_map<CublasKernelRequirement, CublasKernel> optimalKernels;
    std::unordered_map<CublasKernelRequirement, cublasLtMatmulAlgo_t> knownHeuristicAlgorithms;

    std::mutex mtx;

    class Youreusingitwrong : public std::exception {
       public:
        Youreusingitwrong(std::string message) { this->message = message; }

        virtual const char *what() const throw() { return message.c_str(); }

       private:
        std::string message;
    };

    CublasMatrixMultiply() {}

    cudaDataType_t mapToCublasDataType(TensorDescriptor::DataType dataType);

    bool chooseOptimalGemmKernel(int gpuNum,
                                 int rowsA,
                                 int colsA,
                                 int rowsB,
                                 int colsB,
                                 int ldA,
                                 int ldB,
                                 int ldC,
                                 int ldD,
                                 bool transposeA,
                                 bool transposeB,
                                 bool transposeC,
                                 TensorDescriptor::DataType ABCDataType,
                                 bool allowWorkspaces,
                                 bool printResults);

    void getSupportedCublasAlgorithms(const OperationType &operationType,
                                      std::vector<cublasLtMatmulAlgo_t> &supportedAlgorithms,
                                      std::vector<int> &supportedAlgorithmIds,
                                      CublasKernelRequirement cublasKernelRequirement,
                                      int gpuNum);
    std::vector<cublasLtMatmulTile_t> getSupportedTileSizes(cublasLtMatmulAlgo_t algo);
    bool isSplitKSupported(cublasLtMatmulAlgo_t algo);
    uint32_t getReductionSupportMask(cublasLtMatmulAlgo_t algo);
    int getSwizzleMaxValue(cublasLtMatmulAlgo_t algo);
    int getCustomKernelOptionMaxValue(cublasLtMatmulAlgo_t algo);
};

}  // namespace ThorImplementation
