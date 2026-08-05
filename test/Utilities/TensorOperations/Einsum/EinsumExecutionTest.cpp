#include "Utilities/TensorOperations/Einsum/Einsum.h"

#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <stdexcept>
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

}  // namespace

TEST(EinsumExecution, DirectGemmUsesMatrixPathWithoutStandaloneReduction) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor lhs = makeGpuTensor({2, 3}, {1, 2, 3, 4, 5, 6}, stream);
    Tensor rhs = makeGpuTensor({3, 2}, {7, 8, 9, 10, 11, 12}, stream);

    auto einsum = Einsum("ik,kj->ij").stamp({lhs, rhs}, stream);
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::GEMM);
    EXPECT_FALSE(einsum->usesStandaloneReduction());
    EXPECT_EQ(einsum->getOutputTensor().getDimensions(), (std::vector<uint64_t>{2, 2}));

    einsum->run();
    expectNear(copyToCpu(einsum->getOutputTensor(), stream), {58, 64, 139, 154});
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
    EXPECT_EQ(einsum->getOutputTensor().getDimensions(), (std::vector<uint64_t>{2, 2, 2}));

    einsum->run();
    expectNear(copyToCpu(einsum->getOutputTensor(), stream),
               {58, 64,
                139, 154,
                7, 10,
                20, 26});
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

    einsum->run();
    expectNear(copyToCpu(einsum->getOutputTensor(), stream), {1, 4, 2, 5, 3, 6});
}

TEST(EinsumExecution, GenericOuterProductNeedsNoReduction) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor lhs = makeGpuTensor({2}, {2, 3}, stream);
    Tensor rhs = makeGpuTensor({3}, {5, 7, 11}, stream);

    auto einsum = Einsum("i,j->ij").stamp({lhs, rhs}, stream);
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::GENERIC);
    EXPECT_FALSE(einsum->usesStandaloneReduction());

    einsum->run();
    expectNear(copyToCpu(einsum->getOutputTensor(), stream), {10, 14, 22, 15, 21, 33});
}

TEST(EinsumExecution, OneSidedGenericReductionUsesCentralCubReduction) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    // r is present only in lhs, so this cannot be represented solely by GEMM.
    Tensor lhs = makeGpuTensor({2, 2, 3},
                               {1, 2, 3,
                                4, 5, 6,
                                7, 8, 9,
                                10, 11, 12},
                               stream);
    Tensor rhs = makeGpuTensor({3, 2}, {1, 2, 3, 4, 5, 6}, stream);

    auto einsum = Einsum("irk,kj->ij").stamp({lhs, rhs}, stream);
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::GENERIC);
    EXPECT_TRUE(einsum->usesStandaloneReduction());
    EXPECT_TRUE(einsum->getStandaloneReductionPath().has_value());
    EXPECT_EQ(einsum->getPlan().reduction_axes, (std::vector<uint32_t>{2, 3}));

    einsum->run();
    // i=0 sums rows [1,2,3] and [4,5,6] over r before contraction.
    // k totals are [5,7,9], giving [71,92]. i=1 -> [17,19,21], giving [179,236].
    expectNear(copyToCpu(einsum->getOutputTensor(), stream), {71, 92, 179, 236});
}

TEST(EinsumExecution, MoreThanTwoOperandsMultiplyThenReduceThroughCentralCubReduction) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor a = makeGpuTensor({2, 3}, {1, 2, 3, 4, 5, 6}, stream);
    Tensor b = makeGpuTensor({2, 3}, {2, 3, 4, 5, 6, 7}, stream);
    Tensor c = makeGpuTensor({2, 3}, {1, 1, 2, 2, 1, 1}, stream);

    auto einsum = Einsum("ij,ij,ij->i").stamp({a, b, c}, stream);
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::GENERIC);
    EXPECT_TRUE(einsum->usesStandaloneReduction());
    ASSERT_TRUE(einsum->getStandaloneReductionPath().has_value());

    einsum->run();
    expectNear(copyToCpu(einsum->getOutputTensor(), stream), {32, 112});
}

TEST(EinsumExecution, UnaryPermutationAndReductionMaterializesThenUsesCentralCubReduction) {
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
    EXPECT_TRUE(einsum->getPlan().equation.output_dimensions.empty());
    EXPECT_EQ(einsum->getOutputTensor().getDimensions(), (std::vector<uint64_t>{1}));

    einsum->run();
    expectNear(copyToCpu(einsum->getOutputTensor(), stream), {15});
}

TEST(EinsumExecution, SharedContractionBroadcastFallsBackToGenericCubReduction) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor lhs = makeGpuTensor({2, 1}, {2, 3}, stream);
    Tensor rhs = makeGpuTensor({3, 2}, {1, 2, 3, 4, 5, 6}, stream);

    auto einsum = Einsum("ik,kj->ij").stamp({lhs, rhs}, stream);
    ASSERT_TRUE(einsum->getPlan().matrix_multiply.has_value());
    EXPECT_FALSE(einsum->getPlan().matrix_multiply->direct);
    EXPECT_EQ(einsum->getExecutionPath(), EinsumExecutionPath::GENERIC);
    EXPECT_TRUE(einsum->usesStandaloneReduction());

    einsum->run();
    expectNear(copyToCpu(einsum->getOutputTensor(), stream), {18, 24, 27, 36});
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
