#include "Utilities/TensorOperations/GpuMatrixMultiply/CublasKernel.h"

#include "gtest/gtest.h"

#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

using namespace ThorImplementation;

namespace {

CublasKernel makeTestKernel() {
    KernelRequirement kernelRequirement("shared-ownership-test-gpu",
                                        2,
                                        3,
                                        3,
                                        4,
                                        false,
                                        false,
                                        false,
                                        3,
                                        4,
                                        4,
                                        4,
                                        false);
    OperationType operationType(CUBLAS_COMPUTE_32F,
                                CUDA_R_32F,
                                CUDA_R_32F,
                                CUDA_R_32F,
                                CUDA_R_32F,
                                CUDA_R_32F);
    CublasKernelRequirement requirement(kernelRequirement, operationType);

    cublasLtMatmulAlgo_t algorithm{};
    CublasKernelOptions options(algorithm,
                                17,
                                CUBLASLT_MATMUL_TILE_UNDEFINED,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                1.0f);

    return CublasKernel(requirement, options, "shared-ownership-test-gpu");
}

}  // namespace

TEST(CublasKernelSharedOwnership, CopiesShareDescriptorsAndRunStats) {
    CublasKernel kernel = makeTestKernel();
    CublasKernel copy = kernel;

    EXPECT_TRUE(kernel == copy);
    EXPECT_EQ(kernel.getOperationDesc(CublasScalarPointerMode::Host), copy.getOperationDesc(CublasScalarPointerMode::Host));
    EXPECT_EQ(kernel.getOperationDesc(CublasScalarPointerMode::Device), copy.getOperationDesc(CublasScalarPointerMode::Device));
    EXPECT_EQ(kernel.getADesc(), copy.getADesc());
    EXPECT_EQ(kernel.getBDesc(), copy.getBDesc());
    EXPECT_EQ(kernel.getCDesc(), copy.getCDesc());
    EXPECT_EQ(kernel.getDDesc(), copy.getDDesc());

    kernel.recordRun(1.25);
    EXPECT_EQ(copy.getMeasuredRunCount(), 1);
    copy.recordRun(2.75);
    EXPECT_EQ(kernel.getMeasuredRunCount(), 2);
    EXPECT_DOUBLE_EQ(kernel.getAverageRunTimeMilliseconds(), 2.0);
}

TEST(CublasKernelSharedOwnership, MoveTransfersHandleAndSharedStateSurvivesOriginalReset) {
    CublasKernel original = makeTestKernel();
    CublasKernel survivor = original;
    const cublasLtMatrixLayout_t expectedADesc = original.getADesc();

    CublasKernel moved = std::move(original);

    EXPECT_THROW(original.getAlgorithmId(), std::logic_error);
    EXPECT_TRUE(moved == survivor);
    EXPECT_EQ(moved.getADesc(), expectedADesc);

    moved = CublasKernel();
    EXPECT_EQ(survivor.getADesc(), expectedADesc);
    EXPECT_EQ(survivor.getAlgorithmId(), 17);
}

TEST(CublasKernelSharedOwnership, SeparatelyConstructedEquivalentKernelsRemainDistinct) {
    CublasKernel lhs = makeTestKernel();
    CublasKernel rhs = makeTestKernel();

    EXPECT_FALSE(lhs == rhs);
    EXPECT_EQ(lhs.getCublasKernelRequirement(), rhs.getCublasKernelRequirement());
    EXPECT_EQ(lhs.getCublasKernelOptions(), rhs.getCublasKernelOptions());
}

TEST(CublasKernelSharedOwnership, DistinctHandlesMayBeCopiedMovedAndDestroyedConcurrently) {
    constexpr int kThreadCount = 8;
    constexpr int kIterationsPerThread = 5000;

    CublasKernel root = makeTestKernel();
    const cublasLtMatmulDesc_t expectedOperationDesc = root.getOperationDesc();
    const cublasLtMatrixLayout_t expectedADesc = root.getADesc();

    std::vector<CublasKernel> stableSources(kThreadCount, root);
    std::atomic<bool> failed{false};
    std::vector<std::thread> workers;
    workers.reserve(kThreadCount);

    for (int threadIndex = 0; threadIndex < kThreadCount; ++threadIndex) {
        workers.emplace_back([&, threadIndex]() {
            for (int iteration = 0; iteration < kIterationsPerThread; ++iteration) {
                CublasKernel local = stableSources[threadIndex];
                CublasKernel moved = std::move(local);
                CublasKernel assigned;
                assigned = moved;

                if (!(assigned == root) || assigned.getOperationDesc() != expectedOperationDesc || assigned.getADesc() != expectedADesc ||
                    assigned.getAlgorithmId() != 17) {
                    failed.store(true, std::memory_order_relaxed);
                    return;
                }
            }
        });
    }

    for (std::thread &worker : workers)
        worker.join();

    EXPECT_FALSE(failed.load(std::memory_order_relaxed));
    EXPECT_EQ(root.getOperationDesc(), expectedOperationDesc);
    EXPECT_EQ(root.getADesc(), expectedADesc);
}
