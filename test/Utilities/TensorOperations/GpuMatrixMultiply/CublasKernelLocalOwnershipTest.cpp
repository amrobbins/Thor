#include "Utilities/TensorOperations/GpuMatrixMultiply/CublasKernel.h"

#include "gtest/gtest.h"

#include <type_traits>
#include <utility>

using namespace ThorImplementation;

namespace {

CublasKernelRequirement makeTestRequirement() {
    KernelRequirement kernelRequirement("local-ownership-test-gpu",
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
    return CublasKernelRequirement(kernelRequirement, operationType);
}

CublasKernelSelection makeTestSelection() {
    CublasKernelSelection selection;
    selection.algorithm = cublasLtMatmulAlgo_t{};
    selection.algorithmId = 17;
    selection.tileSize = CUBLASLT_MATMUL_TILE_UNDEFINED;
    selection.workspaceSizeInBytes = 0;
    selection.wavesCount = 1.0f;
    selection.measuredRunCount = 2;
    selection.measuredTotalExecutionTimeMilliseconds = 4.0;
    return selection;
}

CublasKernel makeTestKernel() {
    CublasKernelRequirement requirement = makeTestRequirement();
    return CublasKernel(requirement, makeTestSelection(), requirement.kernelRequirement.gpuType);
}

}  // namespace

static_assert(isAcceleratorBackendSelectionRecipeV<CublasKernelSelection>);
static_assert(std::is_copy_constructible_v<CublasKernelSelection>);
static_assert(std::is_copy_assignable_v<CublasKernelSelection>);
static_assert(isAcceleratorBackendLocalExecutionStateV<CublasKernel>);
static_assert(!std::is_copy_constructible_v<CublasKernel>);
static_assert(!std::is_copy_assignable_v<CublasKernel>);
static_assert(std::is_move_constructible_v<CublasKernel>);
static_assert(std::is_move_assignable_v<CublasKernel>);

TEST(CublasKernelLocalOwnership, EquivalentSelectionsMaterializeDistinctDescriptorsAndRunStats) {
    const CublasKernelSelection selection = makeTestSelection();
    CublasKernelRequirement lhsRequirement = makeTestRequirement();
    CublasKernelRequirement rhsRequirement = makeTestRequirement();

    CublasKernel lhs(lhsRequirement, selection, lhsRequirement.kernelRequirement.gpuType);
    CublasKernel rhs(rhsRequirement, selection, rhsRequirement.kernelRequirement.gpuType);

    EXPECT_NE(lhs.executionStateId(), rhs.executionStateId());
    EXPECT_NE(lhs.getOperationDesc(CublasScalarPointerMode::Host), rhs.getOperationDesc(CublasScalarPointerMode::Host));
    EXPECT_NE(lhs.getOperationDesc(CublasScalarPointerMode::Device), rhs.getOperationDesc(CublasScalarPointerMode::Device));
    EXPECT_NE(lhs.getADesc(), rhs.getADesc());
    EXPECT_NE(lhs.getBDesc(), rhs.getBDesc());
    EXPECT_NE(lhs.getCDesc(), rhs.getCDesc());
    EXPECT_NE(lhs.getDDesc(), rhs.getDDesc());

    EXPECT_EQ(lhs.getMeasuredRunCount(), 2);
    EXPECT_EQ(rhs.getMeasuredRunCount(), 2);
    lhs.recordRun(2.0);
    EXPECT_EQ(lhs.getMeasuredRunCount(), 3);
    EXPECT_EQ(rhs.getMeasuredRunCount(), 2);
}

TEST(CublasKernelLocalOwnership, SelectionSnapshotContainsNoSharedMutableRunStats) {
    CublasKernel kernel = makeTestKernel();
    kernel.recordRun(2.0);

    const CublasKernelSelection first = kernel.getSelectionRecipe();
    CublasKernelSelection copy = first;

    EXPECT_EQ(first, copy);
    EXPECT_EQ(first.measuredRunCount, 3);
    EXPECT_DOUBLE_EQ(first.measuredTotalExecutionTimeMilliseconds, 6.0);

    kernel.recordRun(4.0);
    EXPECT_EQ(copy.measuredRunCount, 3);
    EXPECT_DOUBLE_EQ(copy.measuredTotalExecutionTimeMilliseconds, 6.0);
}

TEST(CublasKernelLocalOwnership, MoveTransfersSoleExecutionStateOwnership) {
    CublasKernel original = makeTestKernel();
    const uintptr_t expectedState = original.executionStateId();
    const cublasLtMatrixLayout_t expectedADesc = original.getADesc();

    CublasKernel moved = std::move(original);

    EXPECT_THROW(original.getAlgorithmId(), std::logic_error);
    EXPECT_EQ(moved.executionStateId(), expectedState);
    EXPECT_EQ(moved.getADesc(), expectedADesc);
    EXPECT_EQ(moved.getAlgorithmId(), 17);
}

TEST(CublasKernelLocalOwnership, RepeatedMaterializationNeverAliasesDescriptorState) {
    const CublasKernelSelection selection = makeTestSelection();
    for (int i = 0; i < 128; ++i) {
        CublasKernelRequirement lhsRequirement = makeTestRequirement();
        CublasKernelRequirement rhsRequirement = makeTestRequirement();
        CublasKernel lhs(lhsRequirement, selection, lhsRequirement.kernelRequirement.gpuType);
        CublasKernel rhs(rhsRequirement, selection, rhsRequirement.kernelRequirement.gpuType);
        EXPECT_NE(lhs.executionStateId(), rhs.executionStateId());
        EXPECT_NE(lhs.getOperationDesc(), rhs.getOperationDesc());
        EXPECT_NE(lhs.getADesc(), rhs.getADesc());
    }
}
