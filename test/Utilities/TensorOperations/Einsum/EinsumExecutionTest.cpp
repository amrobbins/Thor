#include "Utilities/TensorOperations/Einsum/Einsum.h"

#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ThorImplementation;

namespace {

#define REQUIRE_CUDA_DEVICE()                                                                                          \
    do {                                                                                                                \
        int cuda_device_count_for_test = 0;                                                                             \
        const cudaError_t cuda_status_for_test = cudaGetDeviceCount(&cuda_device_count_for_test);                       \
        if (cuda_status_for_test != cudaSuccess || cuda_device_count_for_test <= 0) {                                    \
            GTEST_SKIP() << "CUDA device is required for Einsum execution tests.";                                      \
        }                                                                                                               \
    } while (false)

TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

Tensor makeGpuTensor(const std::vector<uint64_t>& dims, const std::vector<float>& values, Stream& stream) {
    Tensor cpu(cpuPlacement, TensorDescriptor(DataType::FP32, dims));
    if (cpu.getTotalNumElements() != values.size()) {
        throw std::runtime_error("makeGpuTensor value count mismatch.");
    }
    auto* ptr = cpu.getMemPtr<float>();
    for (size_t i = 0; i < values.size(); ++i) {
        ptr[i] = values[i];
    }

    Tensor gpu(gpuPlacement, TensorDescriptor(DataType::FP32, dims));
    gpu.copyFromAsync(cpu, stream);
    stream.synchronize();
    return gpu;
}

std::vector<float> copyToCpu(const Tensor& gpu, Stream& stream) {
    Tensor cpu(cpuPlacement, gpu.getDescriptor());
    cpu.copyFromAsync(gpu, stream);
    stream.synchronize();

    std::vector<float> values(cpu.getTotalNumElements());
    const auto* ptr = cpu.getMemPtr<float>();
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = ptr[i];
    }
    return values;
}

void expectNear(const std::vector<float>& actual, const std::vector<float>& expected, float atol = 1.0e-5f) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], atol) << "index " << i;
    }
}

std::vector<float> randomSmallValues(size_t count, std::mt19937& rng) {
    std::uniform_int_distribution<int> distribution(-8, 8);
    std::vector<float> values(count);
    for (float& value : values) {
        value = static_cast<float>(distribution(rng)) * 0.125f;
    }
    return values;
}

}  // namespace

TEST(EinsumExecution, DirectGemmUsesMatrixPathWithoutStandaloneReduction) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor lhs = makeGpuTensor({2, 3}, {1, 2, 3, 4, 5, 6}, stream);
    Tensor rhs = makeGpuTensor({3, 2}, {7, 8, 9, 10, 11, 12}, stream);

    auto einsum = Einsum("ik,kj->ij").stamp({lhs, rhs}, stream);
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::GEMM);
    EXPECT_FALSE(einsum->usesStandaloneReduction());
    EXPECT_EQ(einsum->getExpressionStageKindNames(), (std::vector<std::string>{"Matmul"}));
    EXPECT_EQ(einsum->getOutputTensor().getDimensions(), (std::vector<uint64_t>{2, 2}));

    einsum->run();
    expectNear(copyToCpu(einsum->getOutputTensor(), stream), {58, 64, 139, 154});
}

TEST(EinsumExecution, BlasAddressableDiagonalFeedsGemmAsZeroCopyView) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor lhs = makeGpuTensor({2, 2, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, stream);
    Tensor rhs = makeGpuTensor({3, 2}, {1, 2, 3, 4, 5, 6}, stream);

    auto einsum = Einsum("iik,kj->ij").stamp({lhs, rhs}, stream);
    ASSERT_TRUE(einsum->getPlan().matrix_multiply.has_value());
    EXPECT_TRUE(einsum->getPlan().operands[0].requiresDiagonalExtraction());
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::GEMM);
    EXPECT_FALSE(einsum->usesStandaloneReduction());
    EXPECT_EQ(einsum->getExpressionStageKindNames(), (std::vector<std::string>{"Matmul"}));

    einsum->run();
    expectNear(copyToCpu(einsum->getOutputTensor(), stream), {22, 28, 103, 136});
}

TEST(EinsumExecution, NonBlasAddressableDiagonalMaterializesOnlyOperandBeforeGemm) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor lhs = makeGpuTensor({2, 3, 3},
                               {1, 2, 3, 4, 5, 6, 7, 8, 9,
                                10, 11, 12, 13, 14, 15, 16, 17, 18},
                               stream);
    Tensor rhs = makeGpuTensor({3, 2}, {1, 2, 3, 4, 5, 6}, stream);

    auto einsum = Einsum("ikk,kj->ij").stamp({lhs, rhs}, stream);
    ASSERT_TRUE(einsum->getPlan().matrix_multiply.has_value());
    EXPECT_TRUE(einsum->getPlan().operands[0].requiresDiagonalExtraction());
    // The label order itself is already canonical. The materialization is
    // required only because the diagonal K stride is not a BLAS matrix plane.
    EXPECT_FALSE(einsum->getPlan().matrix_multiply->lhs.requires_materialized_permutation);
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::GEMM);
    EXPECT_FALSE(einsum->usesStandaloneReduction());
    const std::vector<std::string> stage_kinds = einsum->getExpressionStageKindNames();
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "FusedKernel"), 1);
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Matmul"), 1);
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Reduction"), 0);

    einsum->run();
    expectNear(copyToCpu(einsum->getOutputTensor(), stream), {61, 76, 142, 184});
}

TEST(EinsumExecution, NonBlasAddressableDiagonalMaterializationComposesWithSwappedOutputGemm) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor lhs = makeGpuTensor({2, 3, 3},
                               {1, 2, 3, 4, 5, 6, 7, 8, 9,
                                10, 11, 12, 13, 14, 15, 16, 17, 18},
                               stream);
    Tensor rhs = makeGpuTensor({3, 2}, {1, 2, 3, 4, 5, 6}, stream);

    auto einsum = Einsum("ikk,kj->ji").stamp({lhs, rhs}, stream);
    ASSERT_TRUE(einsum->getPlan().matrix_multiply.has_value());
    EXPECT_TRUE(einsum->getPlan().matrix_multiply->requires_output_permutation);
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::GEMM);
    const std::vector<std::string> stage_kinds = einsum->getExpressionStageKindNames();
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "FusedKernel"), 1);
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Matmul"), 1);
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Reduction"), 0);

    einsum->run();
    expectNear(copyToCpu(einsum->getOutputTensor(), stream), {61, 142, 76, 184});
}

TEST(EinsumExecution, RhsBlasAddressableDiagonalFeedsGemmWithoutMaterialization) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor lhs = makeGpuTensor({2, 3}, {1, 2, 3, 4, 5, 6}, stream);
    Tensor rhs = makeGpuTensor({3, 3, 2},
                               {1, 2, 3, 4, 5, 6,
                                7, 8, 9, 10, 11, 12,
                                13, 14, 15, 16, 17, 18},
                               stream);

    auto einsum = Einsum("ik,kkj->ij").stamp({lhs, rhs}, stream);
    ASSERT_TRUE(einsum->getPlan().matrix_multiply.has_value());
    EXPECT_TRUE(einsum->getPlan().operands[1].requiresDiagonalExtraction());
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::GEMM);
    EXPECT_FALSE(einsum->usesStandaloneReduction());
    EXPECT_EQ(einsum->getExpressionStageKindNames(), (std::vector<std::string>{"Matmul"}));

    einsum->run();
    // rhs diagonal rows are [1,2], [9,10], [17,18].
    expectNear(copyToCpu(einsum->getOutputTensor(), stream), {70, 76, 151, 166});
}

TEST(EinsumExecution, BlasAddressableDiagonalFeedsStridedBatchedGemm) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor lhs = makeGpuTensor({2, 2, 2, 2},
                               {1, 2, 3, 4, 5, 6, 7, 8,
                                9, 10, 11, 12, 13, 14, 15, 16},
                               stream);
    Tensor rhs = makeGpuTensor({2, 2, 1}, {1, 2, 3, 4}, stream);

    auto einsum = Einsum("biik,bkj->bij").stamp({lhs, rhs}, stream);
    ASSERT_TRUE(einsum->getPlan().matrix_multiply.has_value());
    EXPECT_TRUE(einsum->getPlan().operands[0].requiresDiagonalExtraction());
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::BATCHED_GEMM);
    EXPECT_TRUE(einsum->usesStridedBatchedGemm());
    EXPECT_FALSE(einsum->usesStandaloneReduction());
    EXPECT_EQ(einsum->getExpressionStageKindNames(), (std::vector<std::string>{"Matmul"}));

    einsum->run();
    expectNear(copyToCpu(einsum->getOutputTensor(), stream), {5, 23, 67, 109});
}

TEST(EinsumExecution, DiagonalPreReductionHappensBeforeGemmWithoutOperandMaterialization) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor lhs = makeGpuTensor({2, 2, 2, 3},
                               {1, 2, 3, 4, 5, 6,
                                7, 8, 9, 10, 11, 12,
                                13, 14, 15, 16, 17, 18,
                                19, 20, 21, 22, 23, 24},
                               stream);
    Tensor rhs = makeGpuTensor({3, 2}, {1, 2, 3, 4, 5, 6}, stream);

    auto einsum = Einsum("iirk,kj->ij").stamp({lhs, rhs}, stream);
    ASSERT_TRUE(einsum->getPlan().matrix_multiply.has_value());
    EXPECT_TRUE(einsum->getPlan().operands[0].requiresDiagonalExtraction());
    EXPECT_FALSE(einsum->getPlan().matrix_multiply->lhs_reduction_labels.empty());
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::GEMM);
    EXPECT_TRUE(einsum->usesStandaloneReduction());
    const std::vector<std::string> stage_kinds = einsum->getExpressionStageKindNames();
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Reduction"), 1);
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Matmul"), 1);
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "FusedKernel"), 0);

    einsum->run();
    expectNear(copyToCpu(einsum->getOutputTensor(), stream), {71, 92, 395, 524});
}

TEST(EinsumExecution, DotProductUsesGemmIntrinsicReductionAndPhysicalScalarOutput) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor lhs = makeGpuTensor({3}, {1, 2, 3}, stream);
    Tensor rhs = makeGpuTensor({3}, {4, 5, 6}, stream);

    auto einsum = Einsum("i,i->").stamp({lhs, rhs}, stream);
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::GEMM);
    EXPECT_FALSE(einsum->usesStandaloneReduction());
    EXPECT_TRUE(einsum->getPlan().equation.output_dimensions.empty());
    EXPECT_EQ(einsum->getOutputTensor().getDimensions(), (std::vector<uint64_t>{1}));

    einsum->run();
    expectNear(copyToCpu(einsum->getOutputTensor(), stream), {32});
}

TEST(EinsumExecution, FlattenedMatrixGroupsLowerThroughExpressionMatmul) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor lhs = makeGpuTensor({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8}, stream);
    Tensor rhs = makeGpuTensor({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8}, stream);

    auto einsum = Einsum("abc,cde->abde").stamp({lhs, rhs}, stream);
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::GEMM);
    EXPECT_EQ(einsum->getExpressionStageKindNames(), (std::vector<std::string>{"Matmul"}));
    EXPECT_TRUE(einsum->getOutputTensor().isDenseContiguous());
    EXPECT_EQ(einsum->getOutputTensor().getDimensions(), (std::vector<uint64_t>{2, 2, 2, 2}));

    einsum->run();
    expectNear(copyToCpu(einsum->getOutputTensor(), stream),
               {11, 14, 17, 20, 23, 30, 37, 44, 35, 46, 57, 68, 47, 62, 77, 92});
}

TEST(EinsumExecution, DirectGemmUsesTransposeFlagsWithoutMaterialization) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    // Equation ki,jk->ij.  Stored lhs is KxI and rhs is JxK.
    Tensor lhs = makeGpuTensor({3, 2}, {1, 4, 2, 5, 3, 6}, stream);
    Tensor rhs = makeGpuTensor({2, 3}, {7, 9, 11, 8, 10, 12}, stream);

    auto einsum = Einsum("ki,jk->ij").stamp({lhs, rhs}, stream);
    ASSERT_TRUE(einsum->getPlan().matrix_multiply.has_value());
    EXPECT_TRUE(einsum->getPlan().matrix_multiply->lhs.transpose);
    EXPECT_TRUE(einsum->getPlan().matrix_multiply->rhs.transpose);
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::GEMM);
    EXPECT_FALSE(einsum->usesStandaloneReduction());

    einsum->run();
    expectNear(copyToCpu(einsum->getOutputTensor(), stream), {58, 64, 139, 154});
}


TEST(EinsumExecution, InterleavedLhsMatrixGroupsMaterializeOnlyOperandThenGemm) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor lhs = makeGpuTensor({2, 3, 2},
                               {1, 2,
                                3, 4,
                                5, 6,
                                7, 8,
                                9, 10,
                                11, 12},
                               stream);
    Tensor rhs = makeGpuTensor({3, 2}, {1, 2, 3, 4, 5, 6}, stream);

    auto einsum = Einsum("acb,cd->abd").stamp({lhs, rhs}, stream);
    ASSERT_TRUE(einsum->getPlan().matrix_multiply.has_value());
    EXPECT_TRUE(einsum->getPlan().matrix_multiply->lhs.requires_materialized_permutation);
    EXPECT_FALSE(einsum->getPlan().matrix_multiply->rhs.requires_materialized_permutation);
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::GEMM);
    EXPECT_FALSE(einsum->usesStandaloneReduction());
    const std::vector<std::string> stage_kinds = einsum->getExpressionStageKindNames();
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "FusedKernel"), 1);
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Matmul"), 1);
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Reduction"), 0);

    einsum->run();
    expectNear(copyToCpu(einsum->getOutputTensor(), stream), {35, 44, 44, 56, 89, 116, 98, 128});
}

TEST(EinsumExecution, BothInterleavedMatrixOperandsMaterializeIndependentlyBeforeOneGemm) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor lhs = makeGpuTensor({2, 3, 2},
                               {1, 2,
                                3, 4,
                                5, 6,
                                7, 8,
                                9, 10,
                                11, 12},
                               stream);
    Tensor rhs = makeGpuTensor({2, 3, 2},
                               {1, 2,
                                3, 4,
                                5, 6,
                                7, 8,
                                9, 10,
                                11, 12},
                               stream);

    auto einsum = Einsum("acb,dce->abde").stamp({lhs, rhs}, stream);
    ASSERT_TRUE(einsum->getPlan().matrix_multiply.has_value());
    EXPECT_TRUE(einsum->getPlan().matrix_multiply->lhs.requires_materialized_permutation);
    EXPECT_TRUE(einsum->getPlan().matrix_multiply->rhs.requires_materialized_permutation);
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::GEMM);
    EXPECT_FALSE(einsum->usesStandaloneReduction());
    const std::vector<std::string> stage_kinds = einsum->getExpressionStageKindNames();
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "FusedKernel"), 2);
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Matmul"), 1);
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Reduction"), 0);

    einsum->run();
    expectNear(copyToCpu(einsum->getOutputTensor(), stream),
               {35, 44, 89, 98,
                44, 56, 116, 128,
                89, 116, 251, 278,
                98, 128, 278, 308});
}

TEST(EinsumExecution, PreReductionCanExposeInterleavedOperandMaterializationBeforeGemm) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor lhs = makeGpuTensor({2, 2, 3, 2},
                               {1, 2, 3, 4, 5, 6,
                                7, 8, 9, 10, 11, 12,
                                13, 14, 15, 16, 17, 18,
                                19, 20, 21, 22, 23, 24},
                               stream);
    Tensor rhs = makeGpuTensor({3, 2}, {1, 2, 3, 4, 5, 6}, stream);

    auto einsum = Einsum("arcb,cd->abd").stamp({lhs, rhs}, stream);
    ASSERT_TRUE(einsum->getPlan().matrix_multiply.has_value());
    EXPECT_TRUE(einsum->getPlan().matrix_multiply->lhs.requires_materialized_permutation);
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::GEMM);
    EXPECT_TRUE(einsum->usesStandaloneReduction());
    const std::vector<std::string> stage_kinds = einsum->getExpressionStageKindNames();
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Reduction"), 1);
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "FusedKernel"), 1);
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Matmul"), 1);

    einsum->run();
    expectNear(copyToCpu(einsum->getOutputTensor(), stream), {124, 160, 142, 184, 340, 448, 358, 472});
}


TEST(EinsumExecution, MaterializedOperandStillUsesBatchBroadcastMatmul) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor lhs = makeGpuTensor({2, 2, 3, 2},
                               {1, 2, 3, 4, 5, 6,
                                7, 8, 9, 10, 11, 12,
                                13, 14, 15, 16, 17, 18,
                                19, 20, 21, 22, 23, 24},
                               stream);
    Tensor rhs = makeGpuTensor({1, 3, 2}, {1, 2, 3, 4, 5, 6}, stream);

    auto einsum = Einsum("xacb,xcd->xabd").stamp({lhs, rhs}, stream);
    ASSERT_TRUE(einsum->getPlan().matrix_multiply.has_value());
    EXPECT_TRUE(einsum->getPlan().matrix_multiply->lhs.requires_materialized_permutation);
    EXPECT_FALSE(einsum->getPlan().matrix_multiply->rhs.requires_materialized_permutation);
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::BATCHED_GEMM);
    EXPECT_TRUE(einsum->usesStridedBatchedGemm());
    EXPECT_FALSE(einsum->usesStandaloneReduction());
    const std::vector<std::string> stage_kinds = einsum->getExpressionStageKindNames();
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "FusedKernel"), 1);
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Matmul"), 1);

    einsum->run();
    expectNear(copyToCpu(einsum->getOutputTensor(), stream),
               {35, 44, 44, 56,
                89, 116, 98, 128,
                143, 188, 152, 200,
                197, 260, 206, 272});
}

TEST(EinsumExecution, MaterializedOperandCanFeedSwappedOutputGemmOrientation) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor lhs = makeGpuTensor({2, 3, 2},
                               {1, 2,
                                3, 4,
                                5, 6,
                                7, 8,
                                9, 10,
                                11, 12},
                               stream);
    Tensor rhs = makeGpuTensor({3, 2}, {1, 2, 3, 4, 5, 6}, stream);

    auto einsum = Einsum("acb,cd->dab").stamp({lhs, rhs}, stream);
    ASSERT_TRUE(einsum->getPlan().matrix_multiply.has_value());
    EXPECT_TRUE(einsum->getPlan().matrix_multiply->lhs.requires_materialized_permutation);
    EXPECT_TRUE(einsum->getPlan().matrix_multiply->requires_output_permutation);
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::GEMM);
    EXPECT_FALSE(einsum->usesStandaloneReduction());
    const std::vector<std::string> stage_kinds = einsum->getExpressionStageKindNames();
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "FusedKernel"), 1);
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Matmul"), 1);

    einsum->run();
    expectNear(copyToCpu(einsum->getOutputTensor(), stream), {35, 44, 89, 98, 44, 56, 116, 128});
}

TEST(EinsumExecution, RequestedMatrixOutputTransposeIsFoldedIntoGemmOrientation) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor lhs = makeGpuTensor({2, 3}, {1, 2, 3, 4, 5, 6}, stream);
    Tensor rhs = makeGpuTensor({3, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, stream);

    auto einsum = Einsum("ik,kj->ji").stamp({lhs, rhs}, stream);
    ASSERT_TRUE(einsum->getPlan().matrix_multiply.has_value());
    EXPECT_TRUE(einsum->getPlan().matrix_multiply->requires_output_permutation);
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::GEMM);
    EXPECT_FALSE(einsum->usesStandaloneReduction());
    EXPECT_EQ(einsum->getExpressionStageKindNames(), (std::vector<std::string>{"Matmul"}));
    EXPECT_EQ(einsum->getOutputTensor().getDimensions(), (std::vector<uint64_t>{4, 2}));
    EXPECT_TRUE(einsum->getOutputTensor().isDenseContiguous());

    einsum->run();
    expectNear(copyToCpu(einsum->getOutputTensor(), stream), {38, 83, 44, 98, 50, 113, 56, 128});
}

TEST(EinsumExecution, SwappedOutputInvertsExistingOperandTransposeFlagsInsideGemm) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    // Equation ki,jk->ji. The normal ij contraction requires both stored
    // operands transposed; the swapped-output GEMM must therefore consume both
    // stored tensors without transpose after reversing operand order.
    Tensor lhs = makeGpuTensor({3, 2}, {1, 4, 2, 5, 3, 6}, stream);
    Tensor rhs = makeGpuTensor({2, 3}, {7, 9, 11, 8, 10, 12}, stream);

    auto einsum = Einsum("ki,jk->ji").stamp({lhs, rhs}, stream);
    ASSERT_TRUE(einsum->getPlan().matrix_multiply.has_value());
    EXPECT_TRUE(einsum->getPlan().matrix_multiply->lhs.transpose);
    EXPECT_TRUE(einsum->getPlan().matrix_multiply->rhs.transpose);
    EXPECT_TRUE(einsum->getPlan().matrix_multiply->requires_output_permutation);
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::GEMM);
    EXPECT_FALSE(einsum->usesStandaloneReduction());
    EXPECT_EQ(einsum->getExpressionStageKindNames(), (std::vector<std::string>{"Matmul"}));

    einsum->run();
    expectNear(copyToCpu(einsum->getOutputTensor(), stream), {58, 139, 64, 154});
}

TEST(EinsumExecution, FlattenedFreeGroupsSwapThroughGemmOrientation) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor lhs = makeGpuTensor({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8}, stream);
    Tensor rhs = makeGpuTensor({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8}, stream);

    auto einsum = Einsum("abc,cde->deab").stamp({lhs, rhs}, stream);
    ASSERT_TRUE(einsum->getPlan().matrix_multiply.has_value());
    EXPECT_TRUE(einsum->getPlan().matrix_multiply->requires_output_permutation);
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::GEMM);
    EXPECT_FALSE(einsum->usesStandaloneReduction());
    EXPECT_EQ(einsum->getExpressionStageKindNames(), (std::vector<std::string>{"Matmul"}));
    EXPECT_EQ(einsum->getOutputTensor().getDimensions(), (std::vector<uint64_t>{2, 2, 2, 2}));
    EXPECT_TRUE(einsum->getOutputTensor().isDenseContiguous());

    einsum->run();
    expectNear(copyToCpu(einsum->getOutputTensor(), stream),
               {11, 23, 35, 47, 14, 30, 46, 62, 17, 37, 57, 77, 20, 44, 68, 92});
}

TEST(EinsumExecution, InterleavedBatchOutputPermutationMaterializesOnlyCompactPostGemmResult) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor lhs = makeGpuTensor({2, 2, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, stream);
    Tensor rhs = makeGpuTensor({2, 3, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, stream);

    auto einsum = Einsum("bia,bac->ibc").stamp({lhs, rhs}, stream);
    ASSERT_TRUE(einsum->getPlan().matrix_multiply.has_value());
    EXPECT_TRUE(einsum->getPlan().matrix_multiply->requires_output_permutation);
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::BATCHED_GEMM);
    EXPECT_FALSE(einsum->usesStandaloneReduction());
    EXPECT_EQ(einsum->getExpressionStageKindNames(), (std::vector<std::string>{"Matmul", "FusedKernel"}));
    EXPECT_EQ(einsum->getOutputTensor().getDimensions(), (std::vector<uint64_t>{2, 2, 2}));
    EXPECT_TRUE(einsum->getOutputTensor().isDenseContiguous());

    einsum->run();
    expectNear(copyToCpu(einsum->getOutputTensor(), stream), {22, 28, 220, 244, 49, 64, 301, 334});
}

TEST(EinsumExecution, InterleavedFreeGroupsUseCompactPostGemmMaterialization) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor lhs = makeGpuTensor({2, 2, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, stream);
    Tensor rhs = makeGpuTensor({3, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, stream);

    auto einsum = Einsum("abc,cde->adbe").stamp({lhs, rhs}, stream);
    ASSERT_TRUE(einsum->getPlan().matrix_multiply.has_value());
    EXPECT_TRUE(einsum->getPlan().matrix_multiply->requires_output_permutation);
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::GEMM);
    EXPECT_FALSE(einsum->usesStandaloneReduction());
    EXPECT_EQ(einsum->getExpressionStageKindNames(), (std::vector<std::string>{"Matmul", "FusedKernel"}));
    EXPECT_EQ(einsum->getOutputTensor().getDimensions(), (std::vector<uint64_t>{2, 2, 2, 2}));
    EXPECT_TRUE(einsum->getOutputTensor().isDenseContiguous());

    einsum->run();
    expectNear(copyToCpu(einsum->getOutputTensor(), stream),
               {38, 44, 83, 98, 50, 56, 113, 128, 128, 152, 173, 206, 176, 200, 239, 272});
}

TEST(EinsumExecution, OperandAndOutputPermutationsMaterializeIndependentlyAroundGemm) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor lhs = makeGpuTensor({2, 3, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, stream);
    Tensor rhs = makeGpuTensor({3, 2}, {1, 2, 3, 4, 5, 6}, stream);

    auto einsum = Einsum("acb,cd->bda").stamp({lhs, rhs}, stream);
    ASSERT_TRUE(einsum->getPlan().matrix_multiply.has_value());
    EXPECT_TRUE(einsum->getPlan().matrix_multiply->lhs.requires_materialized_permutation);
    EXPECT_TRUE(einsum->getPlan().matrix_multiply->requires_output_permutation);
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::GEMM);
    EXPECT_FALSE(einsum->usesStandaloneReduction());
    EXPECT_EQ(einsum->getExpressionStageKindNames(),
              (std::vector<std::string>{"FusedKernel", "Matmul", "FusedKernel"}));

    einsum->run();
    expectNear(copyToCpu(einsum->getOutputTensor(), stream), {35, 89, 44, 116, 44, 98, 56, 128});
}

TEST(EinsumExecution, GeneralPostGemmOutputPermutationWritesPreallocatedOutput) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor lhs = makeGpuTensor({2, 2, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, stream);
    Tensor rhs = makeGpuTensor({2, 3, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, stream);
    Tensor output(gpuPlacement, TensorDescriptor(DataType::FP32, {2, 2, 2}));
    void* expected_ptr = output.getMemPtr<void>();

    auto einsum = Einsum("bia,bac->ibc").stamp({lhs, rhs}, output, stream);
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::BATCHED_GEMM);
    EXPECT_EQ(einsum->getExpressionStageKindNames(), (std::vector<std::string>{"Matmul", "FusedKernel"}));
    EXPECT_EQ(einsum->getOutputTensor().getMemPtr<void>(), expected_ptr);

    einsum->run();
    expectNear(copyToCpu(einsum->getOutputTensor(), stream), {22, 28, 220, 244, 49, 64, 301, 334});
}

TEST(EinsumExecution, RequestedMatrixOutputTransposeWritesPreallocatedOutput) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor lhs = makeGpuTensor({2, 3}, {1, 2, 3, 4, 5, 6}, stream);
    Tensor rhs = makeGpuTensor({3, 2}, {7, 8, 9, 10, 11, 12}, stream);
    Tensor output(gpuPlacement, TensorDescriptor(DataType::FP32, {2, 2}));
    void* expected_ptr = output.getMemPtr<void>();

    auto einsum = Einsum("ik,kj->ji").stamp({lhs, rhs}, output, stream);
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::GEMM);
    EXPECT_EQ(einsum->getExpressionStageKindNames(), (std::vector<std::string>{"Matmul"}));
    EXPECT_EQ(einsum->getOutputTensor().getMemPtr<void>(), expected_ptr);

    einsum->run();
    expectNear(copyToCpu(einsum->getOutputTensor(), stream), {58, 139, 64, 154});
}

TEST(EinsumExecution, BatchedGemmCanSwapFinalMatrixGroupsAfterContraction) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor lhs = makeGpuTensor({2, 2, 3},
                               {1, 2, 3,
                                4, 5, 6,
                                2, 0, 1,
                                1, 3, 2},
                               stream);
    Tensor rhs = makeGpuTensor({2, 3, 2},
                               {7, 8,
                                9, 10,
                                11, 12,
                                1, 2,
                                3, 4,
                                5, 6},
                               stream);

    auto einsum = Einsum("bij,bjk->bki").stamp({lhs, rhs}, stream);
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::BATCHED_GEMM);
    EXPECT_TRUE(einsum->usesStridedBatchedGemm());
    EXPECT_FALSE(einsum->usesStandaloneReduction());
    EXPECT_EQ(einsum->getExpressionStageKindNames(), (std::vector<std::string>{"Matmul"}));
    EXPECT_EQ(einsum->getOutputTensor().getDimensions(), (std::vector<uint64_t>{2, 2, 2}));

    einsum->run();
    expectNear(copyToCpu(einsum->getOutputTensor(), stream),
               {58, 139,
                64, 154,
                7, 20,
                10, 26});
}

TEST(EinsumExecution, DirectBatchedGemmUsesOneStridedBatchedKernelWithoutStandaloneReduction) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor lhs = makeGpuTensor({2, 2, 3},
                               {1, 2, 3,
                                4, 5, 6,
                                2, 0, 1,
                                1, 3, 2},
                               stream);
    Tensor rhs = makeGpuTensor({2, 3, 2},
                               {7, 8,
                                9, 10,
                                11, 12,
                                1, 2,
                                3, 4,
                                5, 6},
                               stream);

    auto einsum = Einsum("bij,bjk->bik").stamp({lhs, rhs}, stream);
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::BATCHED_GEMM);
    EXPECT_TRUE(einsum->usesStridedBatchedGemm());
    EXPECT_FALSE(einsum->usesStandaloneReduction());
    EXPECT_EQ(einsum->getExpressionStageKindNames(), (std::vector<std::string>{"Matmul"}));
    EXPECT_EQ(einsum->getOutputTensor().getDimensions(), (std::vector<uint64_t>{2, 2, 2}));

    einsum->run();
    expectNear(copyToCpu(einsum->getOutputTensor(), stream),
               {58, 64,
                139, 154,
                7, 10,
                20, 26});
}

TEST(EinsumExecution, EllipsisBatchBroadcastUsesExpressionMatmulZeroStrideWithoutReduction) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor lhs = makeGpuTensor({2, 2, 3},
                               {1, 2, 3,
                                4, 5, 6,
                                7, 8, 9,
                                10, 11, 12},
                               stream);
    Tensor rhs = makeGpuTensor({3, 2},
                               {1, 0,
                                0, 1,
                                1, 1},
                               stream);

    auto einsum = Einsum("...ik,...kj->...ij").stamp({lhs, rhs}, stream);
    ASSERT_TRUE(einsum->getPlan().matrix_multiply.has_value());
    EXPECT_FALSE(einsum->getPlan().matrix_multiply->direct);
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::BATCHED_GEMM);
    EXPECT_TRUE(einsum->usesStridedBatchedGemm());
    EXPECT_FALSE(einsum->usesStandaloneReduction());
    EXPECT_EQ(einsum->getExpressionStageKindNames(), (std::vector<std::string>{"Matmul"}));

    einsum->run();
    expectNear(copyToCpu(einsum->getOutputTensor(), stream), {4, 5, 10, 11, 16, 17, 22, 23});
}

TEST(EinsumExecution, CrossBatchBroadcastUsesExpressionGroupedMatmulWithoutReduction) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor lhs = makeGpuTensor({2, 1, 2, 3},
                               {1, 2, 3,
                                4, 5, 6,
                                7, 8, 9,
                                10, 11, 12},
                               stream);
    Tensor rhs = makeGpuTensor({1, 3, 3, 2},
                               {1, 2, 3, 4, 5, 6,
                                7, 8, 9, 10, 11, 12,
                                13, 14, 15, 16, 17, 18},
                               stream);

    auto einsum = Einsum("abik,abkj->abij").stamp({lhs, rhs}, stream);
    ASSERT_TRUE(einsum->getPlan().matrix_multiply.has_value());
    EXPECT_FALSE(einsum->getPlan().matrix_multiply->direct);
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::BATCHED_GEMM);
    EXPECT_TRUE(einsum->usesStridedBatchedGemm());
    EXPECT_FALSE(einsum->usesStandaloneReduction());
    EXPECT_EQ(einsum->getExpressionStageKindNames(),
              (std::vector<std::string>{"Matmul", "Matmul", "DependencyBarrier"}));

    einsum->run();
    expectNear(copyToCpu(einsum->getOutputTensor(), stream),
               {22, 28, 49, 64,
                58, 64, 139, 154,
                94, 100, 229, 244,
                76, 100, 103, 136,
                220, 244, 301, 334,
                364, 388, 499, 532});
}

TEST(EinsumExecution, CrossBatchBroadcastCanSwapOutputGroupsInsideGroupedMatmul) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor lhs = makeGpuTensor({2, 1, 2, 3},
                               {1, 2, 3,
                                4, 5, 6,
                                7, 8, 9,
                                10, 11, 12},
                               stream);
    Tensor rhs = makeGpuTensor({1, 3, 3, 2},
                               {1, 2, 3, 4, 5, 6,
                                7, 8, 9, 10, 11, 12,
                                13, 14, 15, 16, 17, 18},
                               stream);

    auto einsum = Einsum("abik,abkj->abji").stamp({lhs, rhs}, stream);
    ASSERT_TRUE(einsum->getPlan().matrix_multiply.has_value());
    EXPECT_TRUE(einsum->getPlan().matrix_multiply->requires_output_permutation);
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::BATCHED_GEMM);
    EXPECT_TRUE(einsum->usesStridedBatchedGemm());
    EXPECT_FALSE(einsum->usesStandaloneReduction());
    EXPECT_EQ(einsum->getExpressionStageKindNames(),
              (std::vector<std::string>{"Matmul", "Matmul", "DependencyBarrier"}));

    einsum->run();
    expectNear(copyToCpu(einsum->getOutputTensor(), stream),
               {22, 49, 28, 64,
                58, 139, 64, 154,
                94, 229, 100, 244,
                76, 103, 100, 136,
                220, 301, 244, 334,
                364, 499, 388, 532});
}

TEST(EinsumExecution, DirectBatchedGemmPreservesTransposeFlagsInStridedBatchedKernel) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    // Equation bki,bjk->bij. Stored lhs is BxKxI and rhs is BxJxK.
    Tensor lhs = makeGpuTensor({2, 3, 2},
                               {1, 4,
                                2, 5,
                                3, 6,
                                2, 1,
                                0, 3,
                                1, 2},
                               stream);
    Tensor rhs = makeGpuTensor({2, 2, 3},
                               {7, 9, 11,
                                8, 10, 12,
                                1, 3, 5,
                                2, 4, 6},
                               stream);

    auto einsum = Einsum("bki,bjk->bij").stamp({lhs, rhs}, stream);
    ASSERT_TRUE(einsum->getPlan().matrix_multiply.has_value());
    EXPECT_TRUE(einsum->getPlan().matrix_multiply->lhs.transpose);
    EXPECT_TRUE(einsum->getPlan().matrix_multiply->rhs.transpose);
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::BATCHED_GEMM);
    EXPECT_TRUE(einsum->usesStridedBatchedGemm());
    EXPECT_FALSE(einsum->usesStandaloneReduction());

    einsum->run();
    expectNear(copyToCpu(einsum->getOutputTensor(), stream),
               {58, 64,
                139, 154,
                7, 10,
                20, 26});
}

TEST(EinsumExecution, UnaryPermutationMaterializesIntoThorOutputWithoutReduction) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor input = makeGpuTensor({2, 3}, {1, 2, 3, 4, 5, 6}, stream);

    auto einsum = Einsum("ij->ji").stamp({input}, stream);
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::GENERIC);
    EXPECT_FALSE(einsum->usesStandaloneReduction());
    EXPECT_EQ(einsum->getOutputTensor().getDimensions(), (std::vector<uint64_t>{3, 2}));
    EXPECT_NE(einsum->getOutputTensor().getMemPtr<void>(), input.getMemPtr<void>());
    EXPECT_TRUE(einsum->getOutputTensor().isDenseContiguous());

    einsum->run();
    expectNear(copyToCpu(einsum->getOutputTensor(), stream), {1, 4, 2, 5, 3, 6});
}

TEST(EinsumExecution, PairProductOuterProductNeedsNoReduction) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor lhs = makeGpuTensor({2}, {2, 3}, stream);
    Tensor rhs = makeGpuTensor({3}, {5, 7, 11}, stream);

    auto einsum = Einsum("i,j->ij").stamp({lhs, rhs}, stream);
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::PAIR_PRODUCT);
    EXPECT_FALSE(einsum->usesStandaloneReduction());
    const std::vector<std::string> stage_kinds = einsum->getExpressionStageKindNames();
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Reduction"), 0);
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Matmul"), 0);
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "FusedKernel"), 1);

    einsum->run();
    expectNear(copyToCpu(einsum->getOutputTensor(), stream), {10, 14, 22, 15, 21, 33});
}

TEST(EinsumExecution, IndependentReductionsLowerBeforePairProduct) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor lhs = makeGpuTensor({2, 3}, {1, 2, 3, 4, 5, 6}, stream);
    Tensor rhs = makeGpuTensor({2, 2}, {1, 2, 3, 4}, stream);

    auto einsum = Einsum("ir,sj->ij").stamp({lhs, rhs}, stream);
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::PAIR_PRODUCT);
    EXPECT_TRUE(einsum->usesStandaloneReduction());
    EXPECT_EQ(einsum->getStandaloneReductionPaths().size(), 2u);
    const std::vector<std::string> stage_kinds = einsum->getExpressionStageKindNames();
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Reduction"), 2);
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Matmul"), 0);
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "FusedKernel"), 1);

    einsum->run();
    // sum_r(lhs)=[6,15], sum_s(rhs)=[4,6], then their outer product.
    expectNear(copyToCpu(einsum->getOutputTensor(), stream), {24, 36, 60, 90});
}

TEST(EinsumExecution, OneSidedReductionLowersBeforePairProduct) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor lhs = makeGpuTensor({2, 3}, {1, 2, 3, 4, 5, 6}, stream);
    Tensor rhs = makeGpuTensor({2}, {7, 11}, stream);

    auto einsum = Einsum("ir,j->ij").stamp({lhs, rhs}, stream);
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::PAIR_PRODUCT);
    EXPECT_TRUE(einsum->usesStandaloneReduction());
    const std::vector<std::string> stage_kinds = einsum->getExpressionStageKindNames();
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Reduction"), 1);
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Matmul"), 0);
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "FusedKernel"), 1);

    einsum->run();
    expectNear(copyToCpu(einsum->getOutputTensor(), stream), {42, 66, 105, 165});
}

TEST(EinsumExecution, PairProductPreservesSharedOutputBatchLabels) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor lhs = makeGpuTensor({2, 2, 2},
                               {1, 2, 3, 4,
                                5, 6, 7, 8},
                               stream);
    Tensor rhs = makeGpuTensor({2, 2}, {2, 3, 5, 7}, stream);

    auto einsum = Einsum("bir,bj->bij").stamp({lhs, rhs}, stream);
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::PAIR_PRODUCT);
    EXPECT_TRUE(einsum->usesStandaloneReduction());
    const std::vector<std::string> stage_kinds = einsum->getExpressionStageKindNames();
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Reduction"), 1);
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Matmul"), 0);
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "FusedKernel"), 1);

    einsum->run();
    expectNear(copyToCpu(einsum->getOutputTensor(), stream), {6, 9, 14, 21, 55, 77, 75, 105});
}

TEST(EinsumExecution, DiagonalViewCanPreReduceBeforePairProduct) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor lhs = makeGpuTensor({2, 2, 2},
                               {1, 2, 3, 4,
                                5, 6, 7, 8},
                               stream);
    Tensor rhs = makeGpuTensor({2}, {3, 5}, stream);

    auto einsum = Einsum("iir,j->ij").stamp({lhs, rhs}, stream);
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::PAIR_PRODUCT);
    EXPECT_TRUE(einsum->usesStandaloneReduction());
    const std::vector<std::string> stage_kinds = einsum->getExpressionStageKindNames();
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Reduction"), 1);
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Matmul"), 0);
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "FusedKernel"), 1);

    einsum->run();
    // Diagonal i selects [1,2] and [7,8]; reduce r -> [3,15].
    expectNear(copyToCpu(einsum->getOutputTensor(), stream), {9, 15, 45, 75});
}

TEST(EinsumExecution, FullyReducedOperandsPairProductProducesPhysicalScalar) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor lhs = makeGpuTensor({3}, {1, 2, 3}, stream);
    Tensor rhs = makeGpuTensor({2}, {4, 5}, stream);

    auto einsum = Einsum("i,j->").stamp({lhs, rhs}, stream);
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::PAIR_PRODUCT);
    EXPECT_TRUE(einsum->usesStandaloneReduction());
    EXPECT_EQ(einsum->getStandaloneReductionPaths().size(), 2u);
    EXPECT_EQ(einsum->getOutputTensor().getDimensions(), (std::vector<uint64_t>{1}));

    einsum->run();
    expectNear(copyToCpu(einsum->getOutputTensor(), stream), {54});
}

TEST(EinsumExecution, OneSidedReductionLowersToCentralReductionThenGemm) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor lhs = makeGpuTensor({2, 2, 3},
                               {1, 2, 3,
                                4, 5, 6,
                                7, 8, 9,
                                10, 11, 12},
                               stream);
    Tensor rhs = makeGpuTensor({3, 2}, {1, 2, 3, 4, 5, 6}, stream);

    auto einsum = Einsum("irk,kj->ij").stamp({lhs, rhs}, stream);
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::GEMM);
    EXPECT_TRUE(einsum->usesStandaloneReduction());
    ASSERT_TRUE(einsum->getStandaloneReductionPath().has_value());
    EXPECT_EQ(einsum->getStandaloneReductionPath().value(), CubReductionPath::TiledFixedSegment);
    const std::vector<std::string> stage_kinds = einsum->getExpressionStageKindNames();
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Reduction"), 1);
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Matmul"), 1);
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "FusedKernel"), 0);
    ASSERT_TRUE(einsum->getPlan().matrix_multiply.has_value());
    EXPECT_FALSE(einsum->getPlan().matrix_multiply->lhs_reduction_labels.empty());

    einsum->run();
    // i=0: sum_r(lhs)=[5,7,9], then [5,7,9] @ rhs = [71,92].
    // i=1: sum_r(lhs)=[17,19,21] -> [179,236].
    expectNear(copyToCpu(einsum->getOutputTensor(), stream), {71, 92, 179, 236});
}

TEST(EinsumExecution, PreReductionCanFeedSwappedOrientationGemm) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor lhs = makeGpuTensor({2, 2, 3},
                               {1, 2, 3,
                                4, 5, 6,
                                7, 8, 9,
                                10, 11, 12},
                               stream);
    Tensor rhs = makeGpuTensor({3, 2}, {1, 2, 3, 4, 5, 6}, stream);

    auto einsum = Einsum("irk,kj->ji").stamp({lhs, rhs}, stream);
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::GEMM);
    EXPECT_TRUE(einsum->usesStandaloneReduction());
    const std::vector<std::string> stage_kinds = einsum->getExpressionStageKindNames();
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Reduction"), 1);
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Matmul"), 1);
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "FusedKernel"), 0);

    einsum->run();
    expectNear(copyToCpu(einsum->getOutputTensor(), stream), {71, 179, 92, 236});
}

TEST(EinsumExecution, PreReductionFeedsBatchedGemm) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor lhs = makeGpuTensor({2, 1, 2, 2},
                               {1, 2, 3, 4,
                                5, 6, 7, 8},
                               stream);
    Tensor rhs = makeGpuTensor({2, 2, 1}, {1, 2, 3, 4}, stream);

    auto einsum = Einsum("birk,bkj->bij").stamp({lhs, rhs}, stream);
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::BATCHED_GEMM);
    EXPECT_TRUE(einsum->usesStandaloneReduction());
    EXPECT_TRUE(einsum->usesStridedBatchedGemm());
    const std::vector<std::string> stage_kinds = einsum->getExpressionStageKindNames();
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Reduction"), 1);
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Matmul"), 1);
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "FusedKernel"), 0);

    einsum->run();
    expectNear(copyToCpu(einsum->getOutputTensor(), stream), {16, 92});
}

TEST(EinsumExecution, BothOperandPreReductionsFeedOneGemm) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor lhs = makeGpuTensor({2, 2, 3},
                               {1, 2, 3,
                                4, 5, 6,
                                7, 8, 9,
                                10, 11, 12},
                               stream);
    Tensor rhs = makeGpuTensor({3, 2, 2},
                               {1, 2,
                                3, 4,
                                5, 6,
                                7, 8,
                                9, 10,
                                11, 12},
                               stream);

    auto einsum = Einsum("irk,ksj->ij").stamp({lhs, rhs}, stream);
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::GEMM);
    EXPECT_TRUE(einsum->usesStandaloneReduction());
    EXPECT_EQ(einsum->getStandaloneReductionPaths().size(), 2u);
    const std::vector<std::string> stage_kinds = einsum->getExpressionStageKindNames();
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Reduction"), 2);
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Matmul"), 1);
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "FusedKernel"), 0);

    einsum->run();
    expectNear(copyToCpu(einsum->getOutputTensor(), stream), {284, 326, 716, 830});
}

TEST(EinsumExecution, ExactThreeOperandElementwiseThenDotUsesSelectedPairTree) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor a = makeGpuTensor({2, 3}, {1, 2, 3, 4, 5, 6}, stream);
    Tensor b = makeGpuTensor({2, 3}, {2, 3, 4, 5, 6, 7}, stream);
    Tensor c = makeGpuTensor({2, 3}, {1, 1, 2, 2, 1, 1}, stream);

    auto einsum = Einsum("ij,ij,ij->i").stamp({a, b, c}, stream);
    ASSERT_TRUE(einsum->getPlan().exact_contraction.has_value());
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::EXACT_CONTRACTION);
    EXPECT_FALSE(einsum->usesStandaloneReduction());
    const std::vector<std::string> stage_kinds = einsum->getExpressionStageKindNames();
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "FusedKernel"), 1);
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Matmul"), 1);
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Reduction"), 0);

    einsum->run();
    expectNear(copyToCpu(einsum->getOutputTensor(), stream), {32, 112});
}

TEST(EinsumExecution, ExactThreeOperandPlannerExecutesRightToLeftGemmChainWithZeroCopyIntermediateView) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor a = makeGpuTensor({4, 2}, {1, 2, 3, 4, 5, 6, 7, 8}, stream);
    Tensor b = makeGpuTensor({2, 4}, {1, 2, 3, 4, 5, 6, 7, 8}, stream);
    Tensor c = makeGpuTensor({4, 2}, {1, 2, 3, 4, 5, 6, 7, 8}, stream);

    auto einsum = Einsum("ab,bc,cd->ad").stamp({a, b, c}, stream);
    ASSERT_TRUE(einsum->getPlan().exact_contraction.has_value());
    const EinsumExactContractionPlan& exact = *einsum->getPlan().exact_contraction;
    ASSERT_EQ(exact.steps.size(), 2u);
    EXPECT_EQ(exact.steps[0].lhs_source_mask, 2u);
    EXPECT_EQ(exact.steps[0].rhs_source_mask, 4u);
    EXPECT_FALSE(exact.steps[0].physical_candidate.result.dense_storage);
    EXPECT_FALSE(exact.steps[0].physical_candidate.output_materialized);
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::EXACT_CONTRACTION);
    EXPECT_FALSE(einsum->usesStandaloneReduction());
    const std::vector<std::string> stage_kinds = einsum->getExpressionStageKindNames();
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Matmul"), 2);
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "FusedKernel"), 0);
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Reduction"), 0);

    einsum->run();
    expectNear(copyToCpu(einsum->getOutputTensor(), stream),
               {278, 340, 606, 740, 934, 1140, 1262, 1540});
}

TEST(EinsumExecution, ExactThreeOperandTreeComposesKBroadcastPairProductWithGemm) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor lhs = makeGpuTensor({2, 1}, {2, 3}, stream);
    Tensor middle = makeGpuTensor({3, 2}, {1, 2, 3, 4, 5, 6}, stream);
    Tensor rhs = makeGpuTensor({2, 2}, {1, 2, 3, 4}, stream);

    auto einsum = Einsum("ik,kj,jl->il").stamp({lhs, middle, rhs}, stream);
    ASSERT_TRUE(einsum->getPlan().exact_contraction.has_value());
    const EinsumExactContractionPlan& exact = *einsum->getPlan().exact_contraction;
    ASSERT_EQ(exact.steps.size(), 2u);
    EXPECT_EQ(exact.steps[0].physical_candidate.kind, EinsumPlanKind::PAIR_PRODUCT);
    EXPECT_EQ(exact.steps[1].physical_candidate.kind, EinsumPlanKind::GEMM);
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::EXACT_CONTRACTION);
    EXPECT_TRUE(einsum->usesStandaloneReduction());
    EXPECT_EQ(einsum->getStandaloneReductionPaths().size(), 1u);
    const std::vector<std::string> stage_kinds = einsum->getExpressionStageKindNames();
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Reduction"), 1);
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "FusedKernel"), 1);
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Matmul"), 1);

    einsum->run();
    expectNear(copyToCpu(einsum->getOutputTensor(), stream), {90, 132, 135, 198});
}

TEST(EinsumExecution, ExactFourOperandGemmChainExecutesSelectedTree) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor a = makeGpuTensor({2, 2}, {1, 2, 3, 4}, stream);
    Tensor b = makeGpuTensor({2, 2}, {1, 0, 0, 1}, stream);
    Tensor c = makeGpuTensor({2, 2}, {2, 0, 0, 3}, stream);
    Tensor d = makeGpuTensor({2, 2}, {1, 1, 1, 0}, stream);

    auto einsum = Einsum("ij,jk,kl,lm->im").stamp({a, b, c, d}, stream);
    ASSERT_TRUE(einsum->getPlan().exact_contraction.has_value());
    ASSERT_EQ(einsum->getPlan().exact_contraction->steps.size(), 3u);
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::EXACT_CONTRACTION);
    EXPECT_FALSE(einsum->usesStandaloneReduction());
    const std::vector<std::string> stage_kinds = einsum->getExpressionStageKindNames();
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Matmul"), 3);
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "FusedKernel"), 0);
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Reduction"), 0);

    einsum->run();
    expectNear(copyToCpu(einsum->getOutputTensor(), stream), {8, 2, 18, 6});
}

TEST(EinsumExecution, ExactFiveOperandGemmChainExecutesSelectedTree) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor a = makeGpuTensor({2, 2}, {1, 2, 3, 4}, stream);
    Tensor b = makeGpuTensor({2, 2}, {1, 0, 0, 1}, stream);
    Tensor c = makeGpuTensor({2, 2}, {2, 0, 0, 3}, stream);
    Tensor d = makeGpuTensor({2, 2}, {1, 1, 1, 0}, stream);
    Tensor e = makeGpuTensor({2, 2}, {1, 2, 0, 1}, stream);

    auto einsum = Einsum("ij,jk,kl,lm,mn->in").stamp({a, b, c, d, e}, stream);
    ASSERT_TRUE(einsum->getPlan().exact_contraction.has_value());
    ASSERT_EQ(einsum->getPlan().exact_contraction->steps.size(), 4u);
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::EXACT_CONTRACTION);
    EXPECT_FALSE(einsum->usesStandaloneReduction());
    const std::vector<std::string> stage_kinds = einsum->getExpressionStageKindNames();
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Matmul"), 4);
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "FusedKernel"), 0);
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Reduction"), 0);

    einsum->run();
    expectNear(copyToCpu(einsum->getOutputTensor(), stream), {8, 18, 18, 42});
}

TEST(EinsumExecution, SixOperandBridgeGemmChainExecutesSelectedTree) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor a = makeGpuTensor({2, 2}, {1, 2, 3, 4}, stream);
    Tensor b = makeGpuTensor({2, 2}, {1, 0, 0, 1}, stream);
    Tensor c = makeGpuTensor({2, 2}, {2, 0, 0, 3}, stream);
    Tensor d = makeGpuTensor({2, 2}, {1, 0, 0, 1}, stream);
    Tensor e = makeGpuTensor({2, 2}, {1, 1, 1, 0}, stream);
    Tensor f = makeGpuTensor({2, 2}, {1, 2, 0, 1}, stream);

    auto einsum = Einsum("ij,jk,kl,lm,mn,no->io").stamp({a, b, c, d, e, f}, stream);
    ASSERT_TRUE(einsum->getPlan().exact_contraction.has_value());
    const EinsumExactContractionPlan& bridged = *einsum->getPlan().exact_contraction;
    EXPECT_EQ(bridged.planning_mode, EinsumContractionPlanningMode::SIX_OPERAND_BRIDGE);
    ASSERT_EQ(bridged.steps.size(), 5u);
    EXPECT_EQ(bridged.steps.back().result_source_mask, 63u);
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::EXACT_CONTRACTION);
    EXPECT_FALSE(einsum->usesStandaloneReduction());
    const std::vector<std::string> stage_kinds = einsum->getExpressionStageKindNames();
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Matmul"), 5);
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "FusedKernel"), 0);
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Reduction"), 0);

    einsum->run();
    expectNear(copyToCpu(einsum->getOutputTensor(), stream), {8, 18, 18, 42});
}

TEST(EinsumExecution, SevenOperandBeamPlanExecutesSelectedContractionTree) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor a = makeGpuTensor({2, 2}, {1, 2, 3, 4}, stream);
    Tensor b = makeGpuTensor({2, 2}, {1, 0, 0, 1}, stream);
    Tensor c = makeGpuTensor({2, 2}, {2, 0, 0, 3}, stream);
    Tensor d = makeGpuTensor({2, 2}, {1, 0, 0, 1}, stream);
    Tensor e = makeGpuTensor({2, 2}, {1, 1, 1, 0}, stream);
    Tensor f = makeGpuTensor({2, 2}, {1, 2, 0, 1}, stream);
    Tensor g = makeGpuTensor({2, 2}, {1, 0, 0, 1}, stream);

    auto einsum = Einsum("ij,jk,kl,lm,mn,no,op->ip").stamp({a, b, c, d, e, f, g}, stream);
    EXPECT_FALSE(einsum->getPlan().exact_contraction.has_value());
    ASSERT_TRUE(einsum->getPlan().beam_contraction.has_value());
    ASSERT_EQ(einsum->getPlan().beam_contraction->steps.size(), 6u);
    EXPECT_EQ(einsum->getPlan().beam_contraction->steps.back().result_source_mask, 127u);

    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::BEAM_CONTRACTION);
    EXPECT_FALSE(einsum->usesStandaloneReduction());
    const std::vector<std::string> stage_kinds = einsum->getExpressionStageKindNames();
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Matmul"), 6);
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "FusedKernel"), 0);
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Reduction"), 0);

    einsum->run();
    expectNear(copyToCpu(einsum->getOutputTensor(), stream), {8, 18, 18, 42});
}

TEST(EinsumExecution, EightOperandBeamPlanExecutesBeyondFirstBeamDepth) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    std::vector<Tensor> inputs;
    inputs.reserve(8);
    for (int operand = 0; operand < 8; ++operand) {
        inputs.push_back(makeGpuTensor({2, 2}, {1, 0, 0, 1}, stream));
    }

    auto einsum = Einsum("ij,jk,kl,lm,mn,no,op,pq->iq").stamp(inputs, stream);
    ASSERT_TRUE(einsum->getPlan().beam_contraction.has_value());
    const EinsumBeamContractionPlan& beam = *einsum->getPlan().beam_contraction;
    EXPECT_EQ(beam.beam_levels, 3u);
    ASSERT_EQ(beam.steps.size(), 7u);
    EXPECT_EQ(beam.steps.back().result_source_mask, 255u);
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::BEAM_CONTRACTION);
    const std::vector<std::string> stage_kinds = einsum->getExpressionStageKindNames();
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Matmul"), 7);

    einsum->run();
    expectNear(copyToCpu(einsum->getOutputTensor(), stream), {1, 0, 0, 1});
}

TEST(EinsumExecution, SevenOperandBeamBranchingTreeUsesExpressionHelperLane) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor ab = makeGpuTensor({2, 2}, std::vector<float>(4, 1.0f), stream);
    Tensor bc = makeGpuTensor({2, 2}, std::vector<float>(4, 1.0f), stream);
    Tensor de = makeGpuTensor({2, 2}, std::vector<float>(4, 1.0f), stream);
    Tensor ef = makeGpuTensor({2, 2}, std::vector<float>(4, 1.0f), stream);
    Tensor gh = makeGpuTensor({2, 2}, std::vector<float>(4, 1.0f), stream);
    Tensor hi = makeGpuTensor({2, 2}, std::vector<float>(4, 1.0f), stream);
    Tensor cdfg = makeGpuTensor({2, 2, 2, 2}, std::vector<float>(16, 1.0f), stream);

    Einsum operation("ab,bc,de,ef,gh,hi,cdfg->ai");
    auto optimized = operation.stamp({ab, bc, de, ef, gh, hi, cdfg}, stream);
    auto reference = operation.stampGenericReference({ab, bc, de, ef, gh, hi, cdfg}, stream);

    ASSERT_TRUE(optimized->getPlan().beam_contraction.has_value());
    const EinsumBeamContractionPlan& beam = *optimized->getPlan().beam_contraction;
    ASSERT_EQ(beam.steps.size(), 6u);
    // The selected tree builds three independent two-operand branches before
    // joining them through the rank-four connector. Expression owns the actual
    // concurrency and assigns at least one branch to a helper lane.
    EXPECT_EQ(beam.steps[0].result_source_mask, 3u);
    EXPECT_EQ(beam.steps[1].result_source_mask, 12u);
    EXPECT_EQ(beam.steps[2].result_source_mask, 48u);
    EXPECT_EQ(optimized->getExecutionPath(), EinsumExecutionPath::BEAM_CONTRACTION);
    const std::vector<StampedMatmulStageDiagnostic> matmul_stages =
        optimized->getExpressionMatmulStageDiagnostics();
    ASSERT_EQ(matmul_stages.size(), 6u);
    EXPECT_TRUE(std::any_of(matmul_stages.begin(), matmul_stages.end(), [](const auto& stage) {
        return stage.lane_index > 0;
    }));
    EXPECT_EQ(reference->getExecutionPath(), EinsumExecutionPath::GENERIC);

    optimized->run();
    reference->run();
    expectNear(copyToCpu(optimized->getOutputTensor(), stream),
               copyToCpu(reference->getOutputTensor(), stream));
}

TEST(EinsumExecution, SevenOperandBeamTreeExecutesLocalReductionAndBatchBroadcast) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor bxij = makeGpuTensor({2, 2, 2, 2}, std::vector<float>(16, 1.0f), stream);
    Tensor bjk = makeGpuTensor({2, 2, 2}, std::vector<float>(8, 1.0f), stream);
    Tensor kl = makeGpuTensor({2, 2}, std::vector<float>(4, 1.0f), stream);
    Tensor lm = makeGpuTensor({2, 2}, std::vector<float>(4, 1.0f), stream);
    Tensor mn = makeGpuTensor({2, 2}, std::vector<float>(4, 1.0f), stream);
    Tensor no = makeGpuTensor({2, 2}, std::vector<float>(4, 1.0f), stream);
    Tensor op = makeGpuTensor({2, 2}, std::vector<float>(4, 1.0f), stream);

    Einsum operation("bxij,bjk,kl,lm,mn,no,op->bip");
    auto optimized = operation.stamp({bxij, bjk, kl, lm, mn, no, op}, stream);
    auto reference = operation.stampGenericReference({bxij, bjk, kl, lm, mn, no, op}, stream);

    ASSERT_TRUE(optimized->getPlan().beam_contraction.has_value());
    EXPECT_EQ(optimized->getExecutionPath(), EinsumExecutionPath::BEAM_CONTRACTION);
    EXPECT_TRUE(optimized->usesStandaloneReduction());
    EXPECT_TRUE(optimized->usesStridedBatchedGemm());
    const std::vector<std::string> stage_kinds = optimized->getExpressionStageKindNames();
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Matmul"), 6);
    EXPECT_GT(std::count(stage_kinds.begin(), stage_kinds.end(), "Reduction"), 0);
    EXPECT_EQ(reference->getExecutionPath(), EinsumExecutionPath::GENERIC);

    optimized->run();
    reference->run();
    expectNear(copyToCpu(optimized->getOutputTensor(), stream),
               copyToCpu(reference->getOutputTensor(), stream));
}

TEST(EinsumExecution, ExactFiveOperandBranchingTreeExecutesIndependentBranchesBeforeJoin) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor a = makeGpuTensor({2, 3}, {1, 2, 3, 4, 5, 6}, stream);
    Tensor b = makeGpuTensor({3, 2}, {1, 0, 2, 1, 0, 1}, stream);
    Tensor d_to_e = makeGpuTensor({2, 3}, {1, 2, 0, 0, 1, 3}, stream);
    Tensor e_to_f = makeGpuTensor({3, 2}, {1, 2, 0, 1, 2, 0}, stream);
    Tensor c_to_d = makeGpuTensor({2, 2}, {1, 1, 0, 2}, stream);

    auto einsum = Einsum("ab,bc,de,ef,cd->af").stamp({a, b, d_to_e, e_to_f, c_to_d}, stream);
    ASSERT_TRUE(einsum->getPlan().exact_contraction.has_value());
    const EinsumExactContractionPlan& exact = *einsum->getPlan().exact_contraction;
    ASSERT_EQ(exact.steps.size(), 4u);
    // The selected postorder tree first builds the two independent chain
    // branches {de,ef} and {ab,bc}, then joins them through cd.
    EXPECT_EQ(exact.steps[0].result_source_mask, 12u);
    EXPECT_EQ(exact.steps[1].result_source_mask, 3u);
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::EXACT_CONTRACTION);
    const std::vector<std::string> stage_kinds = einsum->getExpressionStageKindNames();
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Matmul"), 4);
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Reduction"), 0);

    einsum->run();
    expectNear(copyToCpu(einsum->getOutputTensor(), stream), {95, 35, 230, 92});
}

TEST(EinsumExecution, ExactFourOperandBatchedGemmChainUsesStridedBatchedExecution) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor a = makeGpuTensor({2, 2, 2},
                             {1, 2, 3, 4,
                              2, 0, 1, 3},
                             stream);
    Tensor b = makeGpuTensor({2, 2, 2},
                             {1, 0, 0, 2,
                              2, 1, 0, 1},
                             stream);
    Tensor c = makeGpuTensor({2, 2, 2},
                             {1, 1, 0, 1,
                              1, 0, 1, 1},
                             stream);
    Tensor d = makeGpuTensor({2, 2, 2},
                             {2, 0, 1, 1,
                              1, 2, 0, 1},
                             stream);

    auto einsum = Einsum("bij,bjk,bkl,blm->bim").stamp({a, b, c, d}, stream);
    ASSERT_TRUE(einsum->getPlan().exact_contraction.has_value());
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::EXACT_CONTRACTION);
    EXPECT_TRUE(einsum->usesStridedBatchedGemm());
    EXPECT_FALSE(einsum->usesStandaloneReduction());
    const std::vector<std::string> stage_kinds = einsum->getExpressionStageKindNames();
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Matmul"), 3);
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Reduction"), 0);

    einsum->run();
    expectNear(copyToCpu(einsum->getOutputTensor(), stream), {7, 5, 17, 11, 6, 14, 6, 16});
}

TEST(EinsumExecution, ExactFourOperandDiagonalChainPreservesZeroCopyDiagonalWhenBlasAddressable) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor diagonal = makeGpuTensor({2, 2, 3},
                                    {1, 2, 3, 4, 5, 6,
                                     7, 8, 9, 10, 11, 12},
                                    stream);
    Tensor b = makeGpuTensor({3, 2}, {1, 2, 3, 4, 5, 6}, stream);
    Tensor c = makeGpuTensor({2, 2}, {1, 0, 0, 1}, stream);
    Tensor d = makeGpuTensor({2, 2}, {2, 1, 1, 3}, stream);

    auto einsum = Einsum("iik,kj,jl,lm->im").stamp({diagonal, b, c, d}, stream);
    ASSERT_TRUE(einsum->getPlan().exact_contraction.has_value());
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::EXACT_CONTRACTION);
    EXPECT_FALSE(einsum->usesStandaloneReduction());
    const std::vector<std::string> stage_kinds = einsum->getExpressionStageKindNames();
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Matmul"), 3);
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "FusedKernel"), 0);

    einsum->run();
    expectNear(copyToCpu(einsum->getOutputTensor(), stream), {72, 106, 342, 511});
}

TEST(EinsumExecution, ExactFourOperandTreeComposesOperandLocalCubReductionWithGemmChain) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor a = makeGpuTensor({2, 2, 3},
                             {1, 2, 3, 4, 5, 6,
                              7, 8, 9, 10, 11, 12},
                             stream);
    Tensor b = makeGpuTensor({3, 2}, {1, 2, 3, 4, 5, 6}, stream);
    Tensor c = makeGpuTensor({2, 2}, {1, 0, 0, 1}, stream);
    Tensor d = makeGpuTensor({2, 2}, {2, 1, 1, 3}, stream);

    auto einsum = Einsum("irk,kj,jl,lm->im").stamp({a, b, c, d}, stream);
    ASSERT_TRUE(einsum->getPlan().exact_contraction.has_value());
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::EXACT_CONTRACTION);
    EXPECT_TRUE(einsum->usesStandaloneReduction());
    const std::vector<std::string> stage_kinds = einsum->getExpressionStageKindNames();
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Reduction"), 1);
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Matmul"), 3);
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "FusedKernel"), 0);

    einsum->run();
    expectNear(copyToCpu(einsum->getOutputTensor(), stream), {234, 347, 594, 887});
}

TEST(EinsumExecution, ExactFourOperandTreeComposesSharedKBroadcastPairProductAndGemmChain) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor a = makeGpuTensor({2, 1}, {2, 3}, stream);
    Tensor b = makeGpuTensor({3, 2}, {1, 2, 3, 4, 5, 6}, stream);
    Tensor c = makeGpuTensor({2, 2}, {1, 0, 0, 1}, stream);
    Tensor d = makeGpuTensor({2, 2}, {2, 1, 1, 3}, stream);

    auto einsum = Einsum("ik,kj,jl,lm->im").stamp({a, b, c, d}, stream);
    ASSERT_TRUE(einsum->getPlan().exact_contraction.has_value());
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::EXACT_CONTRACTION);
    EXPECT_TRUE(einsum->usesStandaloneReduction());
    const std::vector<std::string> stage_kinds = einsum->getExpressionStageKindNames();
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Reduction"), 1);
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "FusedKernel"), 1);
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Matmul"), 2);

    einsum->run();
    expectNear(copyToCpu(einsum->getOutputTensor(), stream), {60, 90, 90, 135});
}

TEST(EinsumExecution, ExactFourOperandTreeHonorsNonCanonicalFinalOutputOrder) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor a = makeGpuTensor({2, 3}, {1, 2, 3, 4, 5, 6}, stream);
    Tensor b = makeGpuTensor({3, 2}, {1, 0, 2, 1, 0, 1}, stream);
    Tensor c = makeGpuTensor({2, 3}, {1, 2, 0, 0, 1, 3}, stream);
    Tensor d = makeGpuTensor({3, 2}, {1, 2, 0, 1, 2, 0}, stream);

    auto einsum = Einsum("ab,bc,cd,de->ea").stamp({a, b, c, d}, stream);
    ASSERT_TRUE(einsum->getPlan().exact_contraction.has_value());
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::EXACT_CONTRACTION);
    EXPECT_EQ(einsum->getOutputTensor().getDimensions(), (std::vector<uint64_t>{2, 2}));
    const std::vector<std::string> stage_kinds = einsum->getExpressionStageKindNames();
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Matmul"), 3);

    einsum->run();
    // Requested e,a order is the transpose of the natural a,e chain result.
    expectNear(copyToCpu(einsum->getOutputTensor(), stream), {35, 80, 25, 67});
}

TEST(EinsumExecution, UnaryPermutationAndReductionUsesCentralPermutationAwareTiledCubWithoutInputMaterialization) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor input = makeGpuTensor({2, 3, 2},
                                 {1, 2,
                                  3, 4,
                                  5, 6,
                                  7, 8,
                                  9, 10,
                                  11, 12},
                                 stream);

    auto einsum = Einsum("ijk->ki").stamp({input}, stream);
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::GENERIC);
    EXPECT_TRUE(einsum->usesStandaloneReduction());
    ASSERT_TRUE(einsum->getStandaloneReductionPath().has_value());
    EXPECT_EQ(einsum->getStandaloneReductionPath().value(), CubReductionPath::TiledFixedSegment);
    EXPECT_EQ(einsum->getOutputTensor().getDimensions(), (std::vector<uint64_t>{2, 2}));

    einsum->run();
    // k-major, i-minor output after summing j.
    expectNear(copyToCpu(einsum->getOutputTensor(), stream), {9, 27, 12, 30});
}

TEST(EinsumExecution, DiagonalFullReductionUsesCubAndMapsLogicalScalarToPhysicalOne) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor input = makeGpuTensor({3, 3},
                                 {1, 2, 3,
                                  4, 5, 6,
                                  7, 8, 9},
                                 stream);

    auto einsum = Einsum("ii->").stamp({input}, stream);
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::GENERIC);
    EXPECT_TRUE(einsum->usesStandaloneReduction());
    ASSERT_TRUE(einsum->getStandaloneReductionPath().has_value());
    EXPECT_EQ(einsum->getStandaloneReductionPath().value(), CubReductionPath::DeviceTransformReduce);
    EXPECT_TRUE(einsum->getPlan().equation.output_dimensions.empty());
    EXPECT_EQ(einsum->getOutputTensor().getDimensions(), (std::vector<uint64_t>{1}));

    einsum->run();
    expectNear(copyToCpu(einsum->getOutputTensor(), stream), {15});
}

TEST(EinsumExecution, FullyBroadcastSharedContractionReducesBeforePairProduct) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor lhs = makeGpuTensor({2, 1}, {2, 3}, stream);
    Tensor rhs = makeGpuTensor({3, 2}, {1, 2, 3, 4, 5, 6}, stream);

    auto einsum = Einsum("ik,kj->ij").stamp({lhs, rhs}, stream);
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::PAIR_PRODUCT);
    EXPECT_TRUE(einsum->usesStandaloneReduction());
    EXPECT_EQ(einsum->getStandaloneReductionPaths().size(), 1u);
    const std::vector<std::string> stage_kinds = einsum->getExpressionStageKindNames();
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Reduction"), 1);
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Matmul"), 0);
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "FusedKernel"), 1);

    einsum->run();
    expectNear(copyToCpu(einsum->getOutputTensor(), stream), {18, 24, 27, 36});
}

TEST(EinsumExecution, PartialSharedContractionBroadcastReducesBeforeRemainingGemm) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor lhs = makeGpuTensor({2, 1, 2}, {1, 2, 3, 4}, stream);
    Tensor rhs = makeGpuTensor({3, 2, 2},
                               {1, 2, 3, 4,
                                5, 6, 7, 8,
                                9, 10, 11, 12},
                               stream);

    auto einsum = Einsum("ikl,klj->ij").stamp({lhs, rhs}, stream);
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::GEMM);
    EXPECT_TRUE(einsum->usesStandaloneReduction());
    EXPECT_EQ(einsum->getStandaloneReductionPaths().size(), 1u);
    const std::vector<std::string> stage_kinds = einsum->getExpressionStageKindNames();
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Reduction"), 1);
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Matmul"), 1);

    einsum->run();
    // sum_k(rhs)=[[15,18],[21,24]], then lhs[:,0,:] @ sum_k(rhs).
    expectNear(copyToCpu(einsum->getOutputTensor(), stream), {57, 66, 129, 150});
}

TEST(EinsumExecution, OppositeSharedBroadcastsNormalizeEveryKBeforePairProduct) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor lhs = makeGpuTensor({2, 1, 2}, {1, 2, 3, 4}, stream);
    Tensor rhs = makeGpuTensor({3, 1, 2}, {1, 2, 3, 4, 5, 6}, stream);

    auto einsum = Einsum("ikl,klj->ij").stamp({lhs, rhs}, stream);
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::PAIR_PRODUCT);
    EXPECT_TRUE(einsum->usesStandaloneReduction());
    EXPECT_EQ(einsum->getStandaloneReductionPaths().size(), 2u);
    const std::vector<std::string> stage_kinds = einsum->getExpressionStageKindNames();
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Reduction"), 2);
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Matmul"), 0);
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "FusedKernel"), 1);

    einsum->run();
    // lhs: reduce l -> [3,7], rhs: reduce k -> [9,12], then outer product.
    expectNear(copyToCpu(einsum->getOutputTensor(), stream), {27, 36, 63, 84});
}

TEST(EinsumExecution, WritesIntoPreallocatedThorOutput) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor lhs = makeGpuTensor({2, 3}, {1, 2, 3, 4, 5, 6}, stream);
    Tensor rhs = makeGpuTensor({3, 2}, {7, 8, 9, 10, 11, 12}, stream);
    Tensor output(gpuPlacement, TensorDescriptor(DataType::FP32, {2, 2}));
    void* expected_ptr = output.getMemPtr<void>();

    auto einsum = Einsum("ik,kj->ij").stamp({lhs, rhs}, output, stream);
    EXPECT_EQ(einsum->getOutputTensor().getMemPtr<void>(), expected_ptr);

    einsum->run();
    expectNear(copyToCpu(output, stream), {58, 64, 139, 154});
}

TEST(EinsumExecution, RejectsOutputThatOverlapsAnInput) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor lhs = makeGpuTensor({2, 2}, {1, 2, 3, 4}, stream);
    Tensor rhs = makeGpuTensor({2, 2}, {5, 6, 7, 8}, stream);

    EXPECT_THROW((void)Einsum("ik,kj->ij").stamp({lhs, rhs}, lhs, stream), std::invalid_argument);
}

TEST(EinsumExecution, GenericReferenceExplicitlyBypassesSelectedExactTree) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor a = makeGpuTensor({2, 3}, {1, 2, 3, 4, 5, 6}, stream);
    Tensor b = makeGpuTensor({3, 2}, {1, 0, 2, 1, 0, 1}, stream);
    Tensor c = makeGpuTensor({2, 2}, {2, 1, 1, 3}, stream);

    Einsum operation("ab,bc,cd->ad");
    auto optimized = operation.stamp({a, b, c}, stream);
    auto reference = operation.stampGenericReference({a, b, c}, stream);
    ASSERT_TRUE(optimized->getPlan().exact_contraction.has_value());
    ASSERT_TRUE(reference->getPlan().exact_contraction.has_value());
    EXPECT_EQ(optimized->getExecutionPath(), EinsumExecutionPath::EXACT_CONTRACTION);
    EXPECT_EQ(reference->getExecutionPath(), EinsumExecutionPath::GENERIC);

    optimized->run();
    reference->run();
    expectNear(copyToCpu(optimized->getOutputTensor(), stream),
               copyToCpu(reference->getOutputTensor(), stream));
}

TEST(EinsumExecution, RandomizedExactThreeToFiveOperandChainsMatchWholeEquationGenericReference) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    std::mt19937 rng(0x7A11C0DEu);
    std::uniform_int_distribution<uint64_t> dimension_distribution(1, 4);

    struct ChainCase {
        const char* equation;
        size_t operand_count;
    };
    const std::vector<ChainCase> cases = {
        {"ab,bc,cd->ad", 3},
        {"ab,bc,cd,de->ae", 4},
        {"ab,bc,cd,de,ef->af", 5},
    };

    for (const ChainCase& test_case : cases) {
        for (int trial = 0; trial < 4; ++trial) {
            std::vector<uint64_t> chain_dimensions(test_case.operand_count + 1);
            for (uint64_t& dimension : chain_dimensions) {
                dimension = dimension_distribution(rng);
            }

            std::vector<Tensor> inputs;
            inputs.reserve(test_case.operand_count);
            for (size_t operand = 0; operand < test_case.operand_count; ++operand) {
                const uint64_t rows = chain_dimensions[operand];
                const uint64_t columns = chain_dimensions[operand + 1];
                inputs.push_back(makeGpuTensor({rows, columns},
                                               randomSmallValues(static_cast<size_t>(rows * columns), rng),
                                               stream));
            }

            Einsum operation(test_case.equation);
            auto optimized = operation.stamp(inputs, stream);
            auto reference = operation.stampGenericReference(inputs, stream);
            ASSERT_TRUE(optimized->getPlan().exact_contraction.has_value());
            EXPECT_EQ(optimized->getExecutionPath(), EinsumExecutionPath::EXACT_CONTRACTION);
            EXPECT_EQ(reference->getExecutionPath(), EinsumExecutionPath::GENERIC);

            optimized->run();
            reference->run();
            const std::vector<float> optimized_values = copyToCpu(optimized->getOutputTensor(), stream);
            const std::vector<float> reference_values = copyToCpu(reference->getOutputTensor(), stream);
            ASSERT_EQ(optimized_values.size(), reference_values.size());
            for (size_t element = 0; element < optimized_values.size(); ++element) {
                EXPECT_NEAR(optimized_values[element], reference_values[element], 2.0e-4f)
                    << "equation=" << test_case.equation << " trial=" << trial << " element=" << element;
            }
        }
    }
}
