#include "Utilities/TensorOperations/GpuMatrixMultiply/BucketedCublasGemm.h"

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Common/Stream.h"

#include "gtest/gtest.h"

#include <cmath>
#include <cstdint>
#include <optional>
#include <vector>

using namespace ThorImplementation;

namespace {

constexpr uint64_t kFullRows = 66;
constexpr uint64_t kInner = 16;
constexpr uint64_t kOutput = 8;
constexpr float kSentinel = -777.0f;

BucketedCublasGemmShape forwardShape() {
    return BucketedCublasGemmShape{static_cast<int>(kFullRows),
                                   static_cast<int>(kInner),
                                   static_cast<int>(kInner),
                                   static_cast<int>(kOutput),
                                   static_cast<int>(kInner),
                                   static_cast<int>(kOutput),
                                   static_cast<int>(kOutput),
                                   static_cast<int>(kOutput),
                                   false,
                                   false,
                                   false};
}

void fillInputs(float *A, float *B) {
    for (uint64_t row = 0; row < kFullRows; ++row) {
        for (uint64_t col = 0; col < kInner; ++col) {
            A[row * kInner + col] = static_cast<float>(static_cast<int>(row % 11) - 5) * 0.125f + static_cast<float>(col) * 0.03125f;
        }
    }
    for (uint64_t row = 0; row < kInner; ++row) {
        for (uint64_t col = 0; col < kOutput; ++col) {
            B[row * kOutput + col] = static_cast<float>(static_cast<int>((row + 2 * col) % 7) - 3) * 0.0625f;
        }
    }
}

float referenceValue(const float *A, const float *B, uint64_t row, uint64_t col) {
    float value = 0.0f;
    for (uint64_t k = 0; k < kInner; ++k) {
        value += A[row * kInner + k] * B[k * kOutput + col];
    }
    return value;
}

}  // namespace

TEST(BucketedCublasGemm, RowsABindingUsesSelectedCachedDescriptorAgainstFullCapacityAllocation) {
    ScopedGpu scopedGpu(0);
    Stream stream(0);
    auto& cublas = CublasMatrixMultiply::instance();
    cublas.clearOptimalKernelSelectionCacheForTests();

    BucketedCublasGemm gemm = BucketedCublasGemm::build(0,
                                                        kFullRows,
                                                        forwardShape(),
                                                        BucketedCublasGemmRowBinding::RowsA,
                                                        CublasMatrixMultiply::MatmulDataTypes::same(DataType::FP32));
    const size_t selectionsAfterFirstFamily = cublas.cachedOptimalKernelSelectionCountForTests();
    ASSERT_GT(selectionsAfterFirstFamily, 0u);

    BucketedCublasGemm peer = BucketedCublasGemm::build(0,
                                                        kFullRows,
                                                        forwardShape(),
                                                        BucketedCublasGemmRowBinding::RowsA,
                                                        CublasMatrixMultiply::MatmulDataTypes::same(DataType::FP32));
    EXPECT_EQ(cublas.cachedOptimalKernelSelectionCountForTests(), selectionsAfterFirstFamily);
    for (const uint64_t activeRows : {0u, 20u, 33u, 65u}) {
        EXPECT_EQ(gemm.getSelectedKernelSelectionForTests(activeRows), peer.getSelectedKernelSelectionForTests(activeRows));
        EXPECT_NE(gemm.getSelectedExecutionStateIdForTests(activeRows), peer.getSelectedExecutionStateIdForTests(activeRows));
    }
    const uint64_t materializationsAfterFamilies = CublasKernel::materializationCountForTests();
    cublas.clearOptimalKernelSelectionCacheForTests();
    ASSERT_EQ(cublas.cachedOptimalKernelSelectionCountForTests(), 0u);

    EXPECT_EQ(gemm.getCapacityBuckets(), (std::vector<uint64_t>{8, 16, 32, 64, 66}));
    EXPECT_EQ(gemm.getSelectedCapacityRows(0), 8U);
    EXPECT_EQ(gemm.getSelectedCapacityRows(7), 8U);
    EXPECT_EQ(gemm.getSelectedCapacityRows(9), 16U);
    EXPECT_EQ(gemm.getSelectedCapacityRows(20), 32U);
    EXPECT_EQ(gemm.getSelectedCapacityRows(32), 32U);
    EXPECT_EQ(gemm.getSelectedCapacityRows(33), 64U);

    const CublasKernelRequirement smallRequirement = gemm.getSelectedKernelRequirement(20);
    EXPECT_EQ(smallRequirement.kernelRequirement.rowsA, 32);
    EXPECT_EQ(smallRequirement.kernelRequirement.rowsB, static_cast<int>(kInner));

    const CublasKernelRequirement largeRequirement = gemm.getSelectedKernelRequirement(33);
    EXPECT_EQ(largeRequirement.kernelRequirement.rowsA, 64);
    const CublasKernelRequirement fullRequirement = gemm.getSelectedKernelRequirement(65);
    EXPECT_EQ(fullRequirement.kernelRequirement.rowsA, static_cast<int>(kFullRows));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU, 0);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    TensorDescriptor ADescriptor(DataType::FP32, {kFullRows, kInner});
    TensorDescriptor BDescriptor(DataType::FP32, {kInner, kOutput});
    TensorDescriptor outputDescriptor(DataType::FP32, {kFullRows, kOutput});

    Tensor A(cpuPlacement, ADescriptor);
    Tensor B(cpuPlacement, BDescriptor);
    Tensor C(cpuPlacement, outputDescriptor);
    Tensor D(cpuPlacement, outputDescriptor);
    Tensor DAfter(cpuPlacement, outputDescriptor);
    Tensor ADevice(gpuPlacement, ADescriptor);
    Tensor BDevice(gpuPlacement, BDescriptor);
    Tensor CDevice(gpuPlacement, outputDescriptor);
    Tensor DDevice(gpuPlacement, outputDescriptor);

    float *AMem = static_cast<float *>(A.getMemPtr());
    float *BMem = static_cast<float *>(B.getMemPtr());
    float *CMem = static_cast<float *>(C.getMemPtr());
    float *DMem = static_cast<float *>(D.getMemPtr());
    fillInputs(AMem, BMem);
    for (uint64_t i = 0; i < kFullRows * kOutput; ++i) {
        CMem[i] = 0.0f;
        DMem[i] = kSentinel;
    }

    ADevice.copyFromAsync(A, stream);
    BDevice.copyFromAsync(B, stream);
    CDevice.copyFromAsync(C, stream);
    DDevice.copyFromAsync(D, stream);

    std::optional<Tensor> workspace;
    if (gemm.getWorkspaceSizeInBytes() > 0) {
        workspace = Tensor(gpuPlacement, TensorDescriptor(DataType::UINT8, {gemm.getWorkspaceSizeInBytes()}));
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    ASSERT_EQ(gemm.launchUncheckedPrevalidated(20, ADevice, BDevice, CDevice, DDevice, workspace, &alpha, &beta, stream),
              CUBLAS_STATUS_SUCCESS);

    DAfter.copyFromAsync(DDevice, stream);
    stream.synchronize();

    const float *afterSmall = static_cast<const float *>(DAfter.getMemPtr());
    for (uint64_t row = 0; row < 32; ++row) {
        for (uint64_t col = 0; col < kOutput; ++col) {
            EXPECT_NEAR(afterSmall[row * kOutput + col], referenceValue(AMem, BMem, row, col), 1.0e-4f);
        }
    }
    for (uint64_t row = 32; row < kFullRows; ++row) {
        for (uint64_t col = 0; col < kOutput; ++col) {
            EXPECT_EQ(afterSmall[row * kOutput + col], kSentinel)
                << "row " << row << " col " << col << " changed beyond the selected M=32 descriptor";
        }
    }

    DDevice.copyFromAsync(D, stream);
    ASSERT_EQ(gemm.launchUncheckedPrevalidated(33, ADevice, BDevice, CDevice, DDevice, workspace, &alpha, &beta, stream),
              CUBLAS_STATUS_SUCCESS);
    DAfter.copyFromAsync(DDevice, stream);
    stream.synchronize();

    const float *afterLarge = static_cast<const float *>(DAfter.getMemPtr());
    for (uint64_t row = 0; row < 64; ++row) {
        for (uint64_t col = 0; col < kOutput; ++col) {
            EXPECT_NEAR(afterLarge[row * kOutput + col], referenceValue(AMem, BMem, row, col), 1.0e-4f);
        }
    }
    for (uint64_t row = 64; row < kFullRows; ++row) {
        for (uint64_t col = 0; col < kOutput; ++col) {
            EXPECT_EQ(afterLarge[row * kOutput + col], kSentinel)
                << "row " << row << " col " << col << " changed beyond the selected M=64 descriptor";
        }
    }

    EXPECT_EQ(CublasKernel::materializationCountForTests(), materializationsAfterFamilies)
        << "Bucketed cuBLASLt runtime must use the prebuilt executable family without rematerializing kernels.";
    EXPECT_EQ(cublas.cachedOptimalKernelSelectionCountForTests(), 0u)
        << "Bucketed cuBLASLt runtime must not consult/repopulate the global selection family.";
}

TEST(BucketedCublasGemm, RowsAAndRowsBBindingBucketsTheRawReductionRowsTogether) {
    ScopedGpu scopedGpu(0);

    const BucketedCublasGemmShape weightGradientShape{static_cast<int>(kFullRows),
                                                       static_cast<int>(kInner),
                                                       static_cast<int>(kFullRows),
                                                       static_cast<int>(kOutput),
                                                       static_cast<int>(kInner),
                                                       static_cast<int>(kOutput),
                                                       static_cast<int>(kOutput),
                                                       static_cast<int>(kOutput),
                                                       true,
                                                       false,
                                                       false};

    BucketedCublasGemm gemm = BucketedCublasGemm::build(0,
                                                        kFullRows,
                                                        weightGradientShape,
                                                        BucketedCublasGemmRowBinding::RowsAAndRowsB,
                                                        CublasMatrixMultiply::MatmulDataTypes::same(DataType::FP32));

    const BucketedCublasGemmShape smallShape = gemm.getSelectedShape(20);
    EXPECT_EQ(smallShape.rowsA, 32);
    EXPECT_EQ(smallShape.rowsB, 32);
    EXPECT_TRUE(smallShape.transposeA);

    const CublasKernelRequirement smallRequirement = gemm.getSelectedKernelRequirement(20);
    EXPECT_EQ(smallRequirement.kernelRequirement.rowsA, 32);
    EXPECT_EQ(smallRequirement.kernelRequirement.rowsB, 32);
    EXPECT_TRUE(smallRequirement.kernelRequirement.transposeA);

    const BucketedCublasGemmShape largeShape = gemm.getSelectedShape(33);
    EXPECT_EQ(largeShape.rowsA, 64);
    EXPECT_EQ(largeShape.rowsB, 64);

    const BucketedCublasGemmShape fullShape = gemm.getSelectedShape(65);
    EXPECT_EQ(fullShape.rowsA, static_cast<int>(kFullRows));
    EXPECT_EQ(fullShape.rowsB, static_cast<int>(kFullRows));
}

TEST(BucketedCublasGemm, RejectsFullShapeThatDoesNotMatchBoundCapacityRowsBeforeKernelSelection) {
    BucketedCublasGemmShape shape = forwardShape();
    shape.rowsA = 65;

    EXPECT_THROW((void)BucketedCublasGemm::build(0,
                                                66,
                                                shape,
                                                BucketedCublasGemmRowBinding::RowsA,
                                                CublasMatrixMultiply::MatmulDataTypes::same(DataType::FP32)),
                 std::invalid_argument);
}
