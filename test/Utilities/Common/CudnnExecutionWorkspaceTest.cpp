#include "Utilities/Common/CudnnExecutionWorkspace.h"

#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <optional>
#include <stdexcept>
#include <string>

using namespace ThorImplementation;
using namespace std;

namespace {

int cudaDeviceCount() {
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess)
        return 0;
    return count;
}

}  // namespace

TEST(CudnnExecutionWorkspace, CheckedSizeAcceptsNonNegativeAndRejectsNegativeValues) {
    EXPECT_EQ(checkedCudnnWorkspaceSizeInBytes(0, "test"), 0U);
    EXPECT_EQ(checkedCudnnWorkspaceSizeInBytes(4096, "test"), 4096U);

    try {
        (void)checkedCudnnWorkspaceSizeInBytes(-1, "RMSNorm forward");
        FAIL() << "expected negative cuDNN workspace size to be rejected";
    } catch (const runtime_error& error) {
        EXPECT_NE(string(error.what()).find("RMSNorm forward"), string::npos);
        EXPECT_NE(string(error.what()).find("negative"), string::npos);
    }
}

TEST(CudnnExecutionWorkspace, ZeroBytesAllowOmittedWorkspaceAndReturnNullPointer) {
    optional<Tensor> workspace;
    EXPECT_NO_THROW(validateCudnnExecutionWorkspace(workspace, 0, 0, "RMSNorm forward"));
    EXPECT_EQ(cudnnExecutionWorkspacePointer(workspace, 0, 0, "RMSNorm forward"), nullptr);
}

TEST(CudnnExecutionWorkspace, NonzeroRequirementRejectsMissingOrUninitializedWorkspace) {
    EXPECT_THROW(validateCudnnExecutionWorkspace(nullopt, 64, 0, "LayerNorm forward"), invalid_argument);

    optional<Tensor> uninitialized = Tensor();
    EXPECT_THROW(validateCudnnExecutionWorkspace(uninitialized, 64, 0, "LayerNorm forward"), invalid_argument);
}

TEST(CudnnExecutionWorkspace, RejectsCpuWorkspace) {
    optional<Tensor> workspace = Tensor(TensorPlacement(TensorPlacement::MemDevices::CPU), TensorDescriptor(DataType::UINT8, {64}));
    EXPECT_THROW(validateCudnnExecutionWorkspace(workspace, 64, 0, "InstanceNorm backward"), invalid_argument);
}

TEST(CudnnExecutionWorkspace, RejectsWrongDtypeAndUndersizedGpuWorkspaceAndAcceptsOversizedWorkspace) {
    if (cudaDeviceCount() < 1)
        GTEST_SKIP() << "CUDA device is required for cuDNN execution-workspace validation tests.";

    const TensorPlacement gpu0(TensorPlacement::MemDevices::GPU, 0);

    optional<Tensor> wrong_dtype = Tensor(gpu0, TensorDescriptor(DataType::FP32, {64}));
    EXPECT_THROW(validateCudnnExecutionWorkspace(wrong_dtype, 64, 0, "RMSNorm backward"), invalid_argument);

    optional<Tensor> too_small = Tensor(gpu0, TensorDescriptor(DataType::UINT8, {63}));
    EXPECT_THROW(validateCudnnExecutionWorkspace(too_small, 64, 0, "RMSNorm backward"), invalid_argument);

    optional<Tensor> oversized = Tensor(gpu0, TensorDescriptor(DataType::UINT8, {256}));
    EXPECT_NO_THROW(validateCudnnExecutionWorkspace(oversized, 64, 0, "RMSNorm backward"));
    EXPECT_EQ(cudnnExecutionWorkspacePointer(oversized, 64, 0, "RMSNorm backward"), oversized->getMemPtr<void>());
}

TEST(CudnnExecutionWorkspace, SuppliedZeroByteWorkspaceStillMustSatisfyWorkspaceContract) {
    optional<Tensor> cpu_workspace =
        Tensor(TensorPlacement(TensorPlacement::MemDevices::CPU), TensorDescriptor(DataType::UINT8, {1}));
    EXPECT_THROW(validateCudnnExecutionWorkspace(cpu_workspace, 0, 0, "SDPA forward"), invalid_argument);
}

TEST(CudnnExecutionWorkspace, RejectsWorkspaceOnDifferentGpuWhenMultipleDevicesAreAvailable) {
    if (cudaDeviceCount() < 2)
        GTEST_SKIP() << "Two CUDA devices are required for wrong-GPU workspace validation.";

    optional<Tensor> workspace =
        Tensor(TensorPlacement(TensorPlacement::MemDevices::GPU, 1), TensorDescriptor(DataType::UINT8, {64}));
    EXPECT_THROW(validateCudnnExecutionWorkspace(workspace, 64, 0, "SDPA backward"), invalid_argument);
}
