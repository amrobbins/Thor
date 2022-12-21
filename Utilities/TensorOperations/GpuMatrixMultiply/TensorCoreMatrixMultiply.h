#pragma once

#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/ComputeTopology/MachineEvaluator.h"
#include "Utilities/TensorOperations/GpuMatrixMultiply/KernelSpec.h"

#include <assert.h>
#include <cuda.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <unordered_map>
#include <utility>

#include <boost/interprocess/containers/map.hpp>
#include <boost/interprocess/managed_mapped_file.hpp>

#define B_rows A_cols
#define C_rows A_rows
#define C_cols B_cols

/**
 * TensorCoreMatrixMultiply is a singleton object that can find the optimal matrix multiply kernel for a matrix multiply operation
 * of a given set of dimensions on a specific type of GPU.
 *
 * Once the optimal kernel is found, it will be use on all subsequent matching matrix multiply calls.
 *
 * It is not required to first evaluate the optimal kernel, when the optimal kernel is not known then a pretty-good-fit kernel can be used.
 *
 * The recommendation is that if you plan to do a matrix multiply of a certain size many times, first find the optimal kernel.
 *
 * The singleton object is accessed as TensorCoreMatrixMultiply::instance().multiply(...)
 *
 * Note: By having the member functions be non-static, they cannot be called unless the constructor has already been called.
 *       Member variables should be non-static so that they are only initialized upon creation of the singleton instance.
 */
class TensorCoreMatrixMultiply {
   public:
    static TensorCoreMatrixMultiply &instance() {
        static TensorCoreMatrixMultiply singletonInstance;  // Guaranteed to be destroyed. Instantiated on first use.
        return singletonInstance;
    }

    virtual ~TensorCoreMatrixMultiply() {}

    // fills C as C = A * B, where A, B and C are all matrices whose memory is allocated on the GPU that will be performing the computation.
    //
    // Prerequisites to using this version of multiply:
    //  1. You have previously called chooseOptimalKernel for a matrix multiply with the same dimensions (i.e. A rows, A cols, B cols).
    //  2. You provide the workspace, which is memory allocated on the target gpu of size getWorkspaceSizeInBytes(same parameters).
    //     When the workspace is 0 bytes, set it to nullptr.
    //
    //  Note: don't allocate a workspace before each call to multiply, you would be better off using the version of multiply that doesn't
    //  use a workspace in terms of performance. Use workspaces only when you can allocate it once and use it a number of times, i.e. when
    //  you have to do a matrix multiplication of the same size over and over.
    void multiply(const half *A,
                  const half *B,
                  half *C,
                  half *workspace,
                  const int32_t A_rows,
                  const int32_t A_cols,
                  const int32_t B_cols,
                  const Stream stream) {
        multiply(A, B, C, workspace, A_rows, A_cols, B_cols, A_cols, B_cols, B_cols, stream);
    }

    // This variant allows non-packed matrices
    void multiply(const half *A,
                  const half *B,
                  half *C,
                  half *workspace,
                  const int32_t A_rows,
                  const int32_t A_cols,
                  const int32_t B_cols,
                  // Leading dimension of A, i.e. number of elements (not bytes) that separate the beginning of two adjacent rows in
                  // memory. Some slots at the end of a row may be unused.
                  const int32_t ld_A,
                  const int32_t ld_B,
                  const int32_t ld_C,
                  const Stream stream);

    // fills C as C = A * B, where A, B and C are all matrices whose memory is allocated on the GPU that will be performing the computation.
    //
    // Prerequisites to using this version of multiply:
    //  1. You have previously called chooseOptimalKernel for a matrix multiply with the same dimensions (i.e. A rows, A cols, B cols).
    void multiply(
        const half *A, const half *B, half *C, const int32_t A_rows, const int32_t A_cols, const int32_t B_cols, const Stream stream) {
        multiply(A, B, C, A_rows, A_cols, B_cols, A_cols, B_cols, B_cols, stream);
    }

    // This variant allows non-packed matrices
    void multiply(const half *A,
                  const half *B,
                  half *C,
                  const int32_t A_rows,
                  const int32_t A_cols,
                  const int32_t B_cols,
                  // Leading dimension of A, i.e. number of elements (not bytes) that separate the beginning of two adjacent rows in
                  // memory. Some slots at the end of a row may be unused.
                  const int32_t ld_A,
                  const int32_t ld_B,
                  const int32_t ld_C,
                  const Stream stream);

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
    void multiplyUsingHeuristicKernelChoice(
        const half *A, const half *B, half *C, const int32_t A_rows, const int32_t A_cols, const int32_t B_cols, const Stream stream) {
        multiplyUsingHeuristicKernelChoice(A, B, C, A_rows, A_cols, B_cols, A_cols, B_cols, B_cols, stream);
    }

    // This variant allows non-packed matrices
    void multiplyUsingHeuristicKernelChoice(const half *A,
                                            const half *B,
                                            half *C,
                                            const int32_t A_rows,
                                            const int32_t A_cols,
                                            const int32_t B_cols,
                                            // Leading dimension of A, i.e. number of elements (not bytes) that separate the beginning of
                                            // two adjacent rows in memory. Some slots at the end of a row may be unused.
                                            const int32_t ld_A,
                                            const int32_t ld_B,
                                            const int32_t ld_C,
                                            const Stream stream);

    // Find any gpu of the specififed type and measure the optimal kernel for the matrix multiply operation
    void chooseOptimalKernel(std::string gpuType, int rowsA, int colsA, int colsB) {
        chooseOptimalKernel(gpuType, rowsA, colsA, colsB, colsA, colsB, colsB);
    }

    void chooseOptimalKernel(std::string gpuType, int rowsA, int colsA, int colsB, int ldA, int ldB, int ldC) {
        // Find a gpu of the proper type or fail, switch current gpu to that one while in this scope
        int gpuNum = -1;
        for (int i = 0; i < (int)MachineEvaluator::instance().getNumGpus(); ++i) {
            if (MachineEvaluator::instance().getGpuType(i) == gpuType) {
                gpuNum = i;
                break;
            }
        }
        assert(gpuNum >= 0);

        chooseOptimalKernel(gpuNum, rowsA, colsA, colsB, ldA, ldB, ldC);
    }

    // Find the optimal kernel for the matrix multiply operation for type of the gpuNum GPU, by measuring it specifically on that gpu.
    //
    // This is can be used to optimize kernels across multiple GPU's, but you must start and end this process by calling
    // startingMultiThreadedOptimization and finishedMultiThreadedOptimization, which adds locking around reading and writing
    // of the map that holds the optimal kernels. This is only needed during optimization because then there is a writer,
    // multiple readers are fine lock free so locks are only used during multi-threaded optimization.
    //
    // Note: never run more than one optimization at a time on any gpu.
    void chooseOptimalKernel(int gpuNum, int rowsA, int colsA, int colsB) {
        chooseOptimalKernel(gpuNum, rowsA, colsA, colsB, colsA, colsB, colsB);
    }

    void chooseOptimalKernel(int gpuNum, int rowsA, int colsA, int colsB, int ldA, int ldB, int ldC);

    unsigned int getWorkspaceSizeInBytes(std::string gpuType, int rowsA, int colsA, int colsB) {
        return getWorkspaceSizeInBytes(gpuType, rowsA, colsA, colsB, colsA, colsB, colsB);
    }
    unsigned int getWorkspaceSizeInBytes(std::string gpuType, int rowsA, int colsA, int colsB, int ldA, int ldB, int ldC);

    unsigned int getWorkspaceSizeInBytes(int gpuNum, int rowsA, int colsA, int colsB) {
        return getWorkspaceSizeInBytes(gpuNum, rowsA, colsA, colsB, colsA, colsB, colsB);
    }
    unsigned int getWorkspaceSizeInBytes(int gpuNum, int rowsA, int colsA, int colsB, int ldA, int ldB, int ldC);

    // getOptimalKernelTime(...) will give you the average time the kernel took when chooseOptimalKernel was called for the
    // matrix multiply operation with those dimensions on that GPU. It is an error to call this for an operation where the
    // optimal kernel was not measured.
    //
    // This can be useful if you need to balance the execution time between multiple GPUs.
    inline float getOptimalKernelTime(int gpuNum, int rowsA, int colsA, int colsB, bool workspaceAllowed) {
        return getOptimalKernelTime(
            MachineEvaluator::instance().getGpuType(gpuNum), rowsA, colsA, colsB, colsA, colsB, colsB, workspaceAllowed);
    }

    inline float getOptimalKernelTime(int gpuNum, int rowsA, int colsA, int colsB, int ldA, int ldB, int ldC, bool workspaceAllowed) {
        return getOptimalKernelTime(MachineEvaluator::instance().getGpuType(gpuNum), rowsA, colsA, colsB, ldA, ldB, ldC, workspaceAllowed);
    }

    inline float getOptimalKernelTime(std::string gpuType, int rowsA, int colsA, int colsB, bool workspaceAllowed) {
        return getOptimalKernelTime(gpuType, rowsA, colsA, colsB, colsA, colsB, colsB, workspaceAllowed);
    }

    float getOptimalKernelTime(std::string gpuType, int rowsA, int colsA, int colsB, int ldA, int ldB, int ldC, bool workspaceAllowed);

    // You must call startingMultiThreadedOptimization() if you are optimizing kernels using multiple threads (so that you can use multiple
    // GPUs). Call it before you begin.
    // After you call startingMultiThreadedKernelOptimization(), the only method of this class that you can call
    // is chooseOptimalKernel(), until you call finishedMultiThreadedKernelOptimization().
    //
    // Remember: never run more than one optimzation at a time on any gpu. So use 1 thread per GPU for this.
    void startingMultiThreadedKernelOptimization();

    // Call finishingMultiThreadedOptimization() once you are done with the multi-threaded kernel optimization.
    void finishedMultiThreadedKernelOptimization();

   protected:
    // The following functions are not private solely for testing purposes.
    static bool checkPreRequisitesBatch8();
    static bool checkPreRequisitesBatch16();
    TensorCoreMatrixMultiply();
    std::vector<KernelWithSpec> getAllKernels() { return TensorCoreMatrixMultiply::kernels; }

   private:
    std::unordered_map<KernelRequirement, KernelWithSpec> optimalKernels;
    std::unordered_map<KernelRequirement, float> optimalKernelMeasuredTime;

    std::vector<KernelWithSpec> kernels;

    bool useLocks;
    std::mutex mtx;

    std::string diskIndexFileName;
    const int CURRENT_KERNEL_VERSION;

    typedef boost::interprocess::allocator<std::pair<const KernelRequirement, std::pair<int, float>>,
                                           boost::interprocess::managed_mapped_file::segment_manager>
        kernel_listing_map_allocator_t;
    typedef boost::interprocess::map<KernelRequirement, std::pair<int, float>, std::less<KernelRequirement>, kernel_listing_map_allocator_t>
        kernel_listing_map_t;
    boost::interprocess::managed_mapped_file optimalKernelListingFile;
    kernel_listing_map_t *optimalKernelListing;

    class Youreusingitwrong : public std::exception {
       public:
        Youreusingitwrong(std::string message) { this->message = message; }

        virtual const char *what() const throw() { return message.c_str(); }

       private:
        std::string message;
    };

    std::vector<KernelWithSpec> getBCol8Kernels();
    std::vector<KernelWithSpec> getBCol16Kernels();
    std::vector<KernelWithSpec> getBCol32Kernels();
    std::vector<KernelWithSpec> getBCol48Kernels();
    std::vector<KernelWithSpec> getBCol64Kernels();
    std::vector<KernelWithSpec> getBCol80Kernels();
    std::vector<KernelWithSpec> getBCol96Kernels();
    std::vector<KernelWithSpec> getBCol112Kernels();
    std::vector<KernelWithSpec> getBCol128Kernels();

    static float computeWaves(KernelRequirement kernelRequirement, KernelWithSpec kernel, int gpuNum);

    std::vector<KernelWithSpec> getEligibleKernels(KernelRequirement kernelRequirement);
    KernelWithSpec getHeuristicKernel(KernelRequirement kernelRequirement);

    kernel_listing_map_t *createDiskIndex();
    void removeDiskIndex();
};

void launchReduce2(int rows, int cols, int ld, half *C, half *workspace, Stream stream);
void launchReduce4(int rows, int cols, int ld, half *C, half *workspace, Stream stream);
void launchReduce8(int rows, int cols, int ld, half *C, half *workspace, Stream stream);
void launchReduce6(int rows, int cols, int ld, half *C, half *workspace, Stream stream);
void launchFreeWorkspace(half *workspace, Stream stream);

template <class DataType, unsigned int numberOfTiles>
unsigned int getReductionWorkspaceSize(KernelRequirement kernelRequirement) {
    assert(numberOfTiles > 0);
    return sizeof(DataType) * (numberOfTiles - 1) * kernelRequirement.rowsA * kernelRequirement.ldC;
}
