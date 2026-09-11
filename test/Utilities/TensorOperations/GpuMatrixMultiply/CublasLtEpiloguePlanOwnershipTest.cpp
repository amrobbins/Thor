#include "Utilities/TensorOperations/GpuMatrixMultiply/CublasMatrixMultiply.h"

#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <cmath>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

using namespace ThorImplementation;

namespace {

#define REQUIRE_CUDA_DEVICE()                                                                                          \
    do {                                                                                                                \
        int cuda_device_count_for_test = 0;                                                                             \
        const cudaError_t cuda_status_for_test = cudaGetDeviceCount(&cuda_device_count_for_test);                       \
        if (cuda_status_for_test != cudaSuccess || cuda_device_count_for_test <= 0) {                                    \
            GTEST_SKIP() << "CUDA device is required for cuBLASLt epilogue ownership tests.";                           \
        }                                                                                                               \
    } while (false)

static_assert(isAcceleratorBackendSelectionRecipeV<CublasMatrixMultiply::LtMatmulAlgorithmSelection>);
static_assert(std::is_copy_constructible_v<CublasMatrixMultiply::LtMatmulAlgorithmSelection>);
static_assert(std::is_copy_assignable_v<CublasMatrixMultiply::LtMatmulAlgorithmSelection>);
static_assert(isAcceleratorBackendLocalExecutionStateV<CublasMatrixMultiply::LtMatmulPlan>);
static_assert(!std::is_copy_constructible_v<CublasMatrixMultiply::LtMatmulPlan>);
static_assert(!std::is_copy_assignable_v<CublasMatrixMultiply::LtMatmulPlan>);
static_assert(std::is_move_constructible_v<CublasMatrixMultiply::LtMatmulPlan>);
static_assert(std::is_move_assignable_v<CublasMatrixMultiply::LtMatmulPlan>);

std::vector<float> copyFp32ToHost(const Tensor& gpuTensor, Stream& stream) {
    Tensor cpu = gpuTensor.clone(TensorPlacement(TensorPlacement::MemDevices::CPU));
    cpu.copyFromAsync(gpuTensor, stream);
    stream.synchronize();
    const float* values = cpu.getMemPtr<float>();
    return std::vector<float>(values, values + cpu.getTotalNumElements());
}

float cublasLtGeluApprox(float x) {
    constexpr float sqrtTwoOverPi = 0.7978845608028654f;
    return 0.5f * x * (1.0f + std::tanh(sqrtTwoOverPi * (x + 0.044715f * x * x * x)));
}

}  // namespace

TEST(CublasLtEpiloguePlanOwnership, GeluAuxForwardPlanWritesPreactivationAndResult) {
    REQUIRE_CUDA_DEVICE();
    constexpr int gpuNum = 0;
    constexpr int m = 8;
    constexpr int k = 8;
    constexpr int n = 8;
    constexpr float aValue = 0.25f;
    constexpr float bValue = 0.5f;
    constexpr float biasValue = 0.1f;

    ScopedGpu scopedGpu(gpuNum);
    Stream stream(gpuNum);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, gpuNum);
    Tensor A(gpuPlacement, TensorDescriptor(DataType::FP32, {m, k}));
    Tensor B(gpuPlacement, TensorDescriptor(DataType::FP32, {k, n}));
    Tensor bias(gpuPlacement, TensorDescriptor(DataType::FP32, {n}));
    Tensor D(gpuPlacement, TensorDescriptor(DataType::FP32, {m, n}));
    Tensor aux(gpuPlacement, TensorDescriptor(DataType::FP32, {m, n}));
    A.fill(aValue, stream);
    B.fill(bValue, stream);
    bias.fill(biasValue, stream);
    D.memsetAsync(stream, 0);
    aux.memsetAsync(stream, 0);
    stream.synchronize();

    auto& cublas = CublasMatrixMultiply::instance();
    const CublasMatrixMultiply::MatmulDataTypes dataTypes =
        CublasMatrixMultiply::MatmulDataTypes::same(DataType::FP32);
    auto plan = cublas.buildGemmWithEpiloguePlan(gpuNum,
                                                  m,
                                                  k,
                                                  k,
                                                  n,
                                                  k,
                                                  n,
                                                  n,
                                                  n,
                                                  false,
                                                  false,
                                                  dataTypes,
                                                  CublasMatrixMultiply::EpilogueFusion::Gelu,
                                                  std::optional<Tensor>(bias),
                                                  true,
                                                  std::nullopt,
                                                  std::nullopt,
                                                  std::optional<Tensor>(aux));
    if (!plan) {
        GTEST_SKIP() << "This GPU/cuBLASLt combination has no qualified GELU_AUX_BIAS algorithm for the test shape.";
    }

    std::optional<Tensor> workspace;
    if (plan->algorithm.workspace_size_in_bytes != 0) {
        workspace = Tensor(gpuPlacement, TensorDescriptor(DataType::UINT8, {plan->algorithm.workspace_size_in_bytes}));
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    plan->runGemmWithEpilogue(A,
                              B,
                              std::optional<Tensor>(bias),
                              D,
                              &alpha,
                              &beta,
                              stream,
                              CublasScalarPointerMode::Host,
                              workspace,
                              true,
                              std::optional<Tensor>(aux));
    stream.synchronize();

    const float expectedPreactivation = static_cast<float>(k) * aValue * bValue + biasValue;
    for (float value : copyFp32ToHost(aux, stream)) {
        EXPECT_NEAR(value, expectedPreactivation, 2.0e-4f)
            << "GELU_AUX must contain the input to GELU after the bias epilogue is applied.";
    }
    const float expectedOutput = cublasLtGeluApprox(expectedPreactivation);
    for (float value : copyFp32ToHost(D, stream)) {
        EXPECT_NEAR(value, expectedOutput, 3.0e-4f);
    }
}

TEST(CublasLtEpiloguePlanOwnership, EquivalentPlansShareSelectionOnlyAndSurviveSelectionCacheClear) {
    REQUIRE_CUDA_DEVICE();
    constexpr int gpuNum = 0;
    constexpr int m = 32;
    constexpr int k = 32;
    constexpr int n = 32;

    ScopedGpu scopedGpu(gpuNum);
    Stream streamA(gpuNum);
    Stream streamB(gpuNum);
    auto& cublas = CublasMatrixMultiply::instance();
    cublas.clearLtMatmulAlgorithmSelectionCacheForTests();
    const uint64_t buildsBefore = cublas.ltMatmulPlanBuildCountForTests();

    const CublasMatrixMultiply::MatmulDataTypes dataTypes =
        CublasMatrixMultiply::MatmulDataTypes::same(DataType::FP32);

    auto first = cublas.buildGemmWithEpiloguePlan(gpuNum,
                                                   m,
                                                   k,
                                                   k,
                                                   n,
                                                   k,
                                                   n,
                                                   n,
                                                   n,
                                                   false,
                                                   false,
                                                   dataTypes,
                                                   CublasMatrixMultiply::EpilogueFusion::Relu,
                                                   std::nullopt,
                                                   false);
    ASSERT_NE(first, nullptr);
    EXPECT_EQ(cublas.ltMatmulPlanBuildCountForTests(), buildsBefore + 1);
    ASSERT_EQ(cublas.cachedLtMatmulAlgorithmSelectionCountForTests(), 1u);
    ASSERT_EQ(cublas.ltMatmulAlgorithmSelectionMissCountForTests(), 1u);
    ASSERT_EQ(cublas.ltMatmulAlgorithmSelectionTuneCountForTests(), 1u);

    // Workspace availability is a lookup constraint, not part of immutable
    // algorithm identity.  A cache miss preselects both the preferred algorithm
    // and a zero-workspace fallback for this stable matmul key.
    auto zeroWorkspace = cublas.buildGemmWithEpiloguePlan(gpuNum,
                                                           m,
                                                           k,
                                                           k,
                                                           n,
                                                           k,
                                                           n,
                                                           n,
                                                           n,
                                                           false,
                                                           false,
                                                           dataTypes,
                                                           CublasMatrixMultiply::EpilogueFusion::Relu,
                                                           std::nullopt,
                                                           false,
                                                           std::nullopt,
                                                           0u);
    ASSERT_NE(zeroWorkspace, nullptr);
    EXPECT_EQ(zeroWorkspace->algorithm.workspace_size_in_bytes, 0u);
    EXPECT_EQ(cublas.cachedLtMatmulAlgorithmSelectionCountForTests(), 1u);
    EXPECT_EQ(cublas.ltMatmulAlgorithmSelectionTuneCountForTests(), 1u);
    EXPECT_GE(cublas.ltMatmulAlgorithmSelectionHitCountForTests(), 1u);

    auto second = cublas.buildGemmWithEpiloguePlan(gpuNum,
                                                    m,
                                                    k,
                                                    k,
                                                    n,
                                                    k,
                                                    n,
                                                    n,
                                                    n,
                                                    false,
                                                    false,
                                                    dataTypes,
                                                    CublasMatrixMultiply::EpilogueFusion::Relu,
                                                    std::nullopt,
                                                    false);
    ASSERT_NE(second, nullptr);
    EXPECT_EQ(cublas.ltMatmulPlanBuildCountForTests(), buildsBefore + 3);
    EXPECT_EQ(cublas.cachedLtMatmulAlgorithmSelectionCountForTests(), 1u);
    EXPECT_EQ(cublas.ltMatmulAlgorithmSelectionTuneCountForTests(), 1u);
    EXPECT_GE(cublas.ltMatmulAlgorithmSelectionHitCountForTests(), 2u);
    EXPECT_EQ(first->algorithm.workspace_size_in_bytes, second->algorithm.workspace_size_in_bytes);

    EXPECT_NE(first->executionStateId(), 0u);
    EXPECT_NE(second->executionStateId(), 0u);
    EXPECT_NE(first->executionStateId(), second->executionStateId());
    EXPECT_NE(first->operation_desc_host, second->operation_desc_host);
    EXPECT_NE(first->operation_desc_device, second->operation_desc_device);
    EXPECT_NE(first->a_desc, second->a_desc);
    EXPECT_NE(first->b_desc, second->b_desc);
    EXPECT_NE(first->c_desc, second->c_desc);
    EXPECT_NE(first->d_desc, second->d_desc);

    // Move-only local state must transfer the sole descriptor ownership without
    // duplicating or dropping any of the retained cuBLASLt layouts.
    CublasMatrixMultiply::LtMatmulPlan moved(std::move(*second));
    EXPECT_EQ(second->executionStateId(), 0u);
    EXPECT_NE(moved.executionStateId(), 0u);
    EXPECT_NE(moved.a_desc, nullptr);
    EXPECT_NE(moved.b_desc, nullptr);
    EXPECT_NE(moved.c_desc, nullptr);
    EXPECT_NE(moved.d_desc, nullptr);

    CublasMatrixMultiply::LtMatmulPlan movedAssigned;
    movedAssigned = std::move(moved);
    EXPECT_EQ(moved.executionStateId(), 0u);
    EXPECT_NE(movedAssigned.executionStateId(), 0u);
    EXPECT_NE(movedAssigned.a_desc, nullptr);
    EXPECT_NE(movedAssigned.b_desc, nullptr);
    EXPECT_NE(movedAssigned.c_desc, nullptr);
    EXPECT_NE(movedAssigned.d_desc, nullptr);

    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, gpuNum);
    Tensor A(gpuPlacement, TensorDescriptor(DataType::FP32, {m, k}));
    Tensor B(gpuPlacement, TensorDescriptor(DataType::FP32, {k, n}));
    Tensor D1(gpuPlacement, TensorDescriptor(DataType::FP32, {m, n}));
    Tensor D2(gpuPlacement, TensorDescriptor(DataType::FP32, {m, n}));
    A.fill(1.0, streamA);
    B.fill(1.0, streamA);
    D1.memsetAsync(streamA, 0);
    D2.memsetAsync(streamA, 0);
    streamA.synchronize();

    std::optional<Tensor> workspace1;
    std::optional<Tensor> workspace2;
    const uint64_t workspaceBytes = first->algorithm.workspace_size_in_bytes;
    if (workspaceBytes != 0) {
        workspace1 = Tensor(gpuPlacement, TensorDescriptor(DataType::UINT8, {workspaceBytes}));
        workspace2 = Tensor(gpuPlacement, TensorDescriptor(DataType::UINT8, {workspaceBytes}));
        EXPECT_NE(workspace1->getMemPtr(), workspace2->getMemPtr());
    }

    cublas.clearLtMatmulAlgorithmSelectionCacheForTests();
    ASSERT_EQ(cublas.cachedLtMatmulAlgorithmSelectionCountForTests(), 0u);
    ASSERT_EQ(cublas.ltMatmulAlgorithmSelectionTuneCountForTests(), 0u);
    const uint64_t buildsAfterStamping = cublas.ltMatmulPlanBuildCountForTests();

    const float alpha = 1.0f;
    const float beta = 0.0f;
    for (int repetition = 0; repetition < 8; ++repetition) {
        first->runGemmWithEpilogue(A,
                                   B,
                                   std::nullopt,
                                   D1,
                                   &alpha,
                                   &beta,
                                   streamA,
                                   CublasScalarPointerMode::Host,
                                   workspace1,
                                   false);
        movedAssigned.runGemmWithEpilogue(A,
                                          B,
                                          std::nullopt,
                                          D2,
                                          &alpha,
                                          &beta,
                                          streamB,
                                          CublasScalarPointerMode::Host,
                                          workspace2,
                                          false);
    }
    streamA.synchronize();
    streamB.synchronize();

    EXPECT_EQ(cublas.cachedLtMatmulAlgorithmSelectionCountForTests(), 0u);
    EXPECT_EQ(cublas.ltMatmulAlgorithmSelectionTuneCountForTests(), 0u);
    EXPECT_EQ(cublas.ltMatmulPlanBuildCountForTests(), buildsAfterStamping)
        << "cuBLASLt epilogue runtime must not rebuild descriptor-bearing LtMatmulPlan state.";

    for (const float value : copyFp32ToHost(D1, streamA)) {
        EXPECT_NEAR(value, static_cast<float>(k), 1.0e-5f);
    }
    for (const float value : copyFp32ToHost(D2, streamB)) {
        EXPECT_NEAR(value, static_cast<float>(k), 1.0e-5f);
    }
}
