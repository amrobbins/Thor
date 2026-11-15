#include "Utilities/TensorOperations/GpuMatrixMultiply/CublasMatrixMultiply.h"

#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <optional>
#include <unordered_set>
#include <vector>

using namespace ThorImplementation;

namespace {

bool haveCudaDevice() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

const TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
const TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

// Small integers: every product and sum below is exact in FP16 and BF16, so the device result must
// match the host reference exactly.
float inputValue(uint64_t i, uint64_t salt) { return static_cast<float>(static_cast<int64_t>((i * 7 + salt * 3) % 7) - 3); }

template <typename T>
T toDevice(float value);
template <>
half toDevice<half>(float value) {
    return __float2half(value);
}
template <>
__nv_bfloat16 toDevice<__nv_bfloat16>(float value) {
    return __float2bfloat16(value);
}
float toHost(half value) { return __half2float(value); }
float toHost(__nv_bfloat16 value) { return __bfloat162float(value); }

template <typename T>
Tensor makeGpuStorage(DataType dtype, uint64_t elements, uint64_t salt, Stream &stream) {
    Tensor cpu(cpuPlacement, TensorDescriptor(dtype, {elements}));
    T *values = cpu.getMemPtr<T>();
    for (uint64_t i = 0; i < elements; ++i) {
        values[i] = toDevice<T>(inputValue(i, salt));
    }
    Tensor gpu(gpuPlacement, TensorDescriptor(dtype, {elements}));
    gpu.copyFromAsync(cpu, stream);
    stream.synchronize();
    return gpu;
}

template <typename T>
std::vector<float> copyStorageToHost(const Tensor &gpu, Stream &stream) {
    Tensor cpu(cpuPlacement, gpu.getDescriptor());
    cpu.copyFromAsync(gpu, stream);
    stream.synchronize();
    std::vector<float> values(gpu.getTotalNumElements());
    const T *data = cpu.getMemPtr<T>();
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = toHost(data[i]);
    }
    return values;
}

// D[b] = A[b] * B[b] for row-major A (m x k), B (k x n), D (m x n), each batch `stride` elements
// apart, with every operand starting `offset` elements into its storage.
template <typename T>
void runMisalignedStridedBatchedGemm(DataType dtype) {
    ScopedGpu scopedGpu(0);
    Stream stream(0);

    // Odd sizes: every matrix and every batch stride is an odd number of 2-byte elements, and each
    // operand starts at an odd element offset, so no operand pointer is more than 2-byte aligned.
    constexpr int32_t batchCount = 3;
    constexpr int32_t m = 3;
    constexpr int32_t k = 5;
    constexpr int32_t n = 7;
    constexpr int64_t strideA = m * k;
    constexpr int64_t strideB = k * n;
    constexpr int64_t strideD = m * n;
    constexpr uint64_t offsetA = 1;
    constexpr uint64_t offsetB = 3;
    constexpr uint64_t offsetD = 5;

    const uint64_t elementsA = static_cast<uint64_t>(strideA) * batchCount;
    const uint64_t elementsB = static_cast<uint64_t>(strideB) * batchCount;
    const uint64_t elementsD = static_cast<uint64_t>(strideD) * batchCount;
    Tensor storageA = makeGpuStorage<T>(dtype, offsetA + elementsA, 1, stream);
    Tensor storageB = makeGpuStorage<T>(dtype, offsetB + elementsB, 2, stream);
    Tensor storageD(gpuPlacement, TensorDescriptor(dtype, {offsetD + elementsD}));
    storageD.memsetAsync(stream, 0);
    const TensorSpan A(storageA, offsetA, elementsA);
    const TensorSpan B(storageB, offsetB, elementsB);
    const TensorSpan D(storageD, offsetD, elementsD);

    const CublasOperandAlignment alignment = CublasKernel::operandAlignmentOf(A, B, D, D);
    ASSERT_EQ(alignment.a, 2u);
    ASSERT_EQ(alignment.b, 2u);
    ASSERT_EQ(alignment.d, 2u);

    const CublasStridedBatchConfig batchConfig = CublasStridedBatchConfig::strided(batchCount, strideA, strideB, strideD, strideD);
    const CublasMatrixMultiply::MatmulDataTypes dataTypes{dtype, dtype, dtype, dtype, DataType::FP32};
    CublasMatrixMultiply &cublas = CublasMatrixMultiply::instance();
    cublas.chooseOptimalStridedBatchedGemmKernel(0, m, k, k, n, k, n, n, n, false, false, false, dataTypes, batchConfig, false, alignment);

    CublasKernel kernel =
        cublas.materializeSelectedGemmKernel(0, m, k, k, n, k, n, n, n, false, false, false, dataTypes, true, batchConfig, alignment);
    EXPECT_EQ(kernel.getCublasKernelRequirement().kernelRequirement.alignment, alignment);
    EXPECT_TRUE(kernel.getAlgorithmAlignment().satisfiedBy(alignment))
        << "cuBLASLt returned an algorithm needing A/B/C/D = " << kernel.getAlgorithmAlignment().a << "/"
        << kernel.getAlgorithmAlignment().b << "/" << kernel.getAlgorithmAlignment().c << "/" << kernel.getAlgorithmAlignment().d
        << " bytes for 2-byte-aligned operands.";
    EXPECT_NO_THROW(kernel.requireOperandAlignment(A, B, D, D));

    std::optional<Tensor> workspace;
    const uint64_t workspaceBytes = kernel.getWorkspaceSizeInBytes(0);
    if (workspaceBytes > 0) {
        workspace = Tensor(gpuPlacement, TensorDescriptor(DataType::UINT8, {workspaceBytes}), 256);
    }
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublas.stridedBatchedGemm(
        A, B, D, D, workspace, m, k, k, n, k, n, n, n, false, false, false, &alpha, &beta, dataTypes, batchConfig, stream);

    const std::vector<float> a = copyStorageToHost<T>(storageA, stream);
    const std::vector<float> b = copyStorageToHost<T>(storageB, stream);
    const std::vector<float> d = copyStorageToHost<T>(storageD, stream);
    for (uint64_t i = 0; i < offsetD; ++i) {
        EXPECT_EQ(d[i], 0.0f) << "write before the output view at element " << i;
    }
    for (int32_t batch = 0; batch < batchCount; ++batch) {
        for (int32_t row = 0; row < m; ++row) {
            for (int32_t col = 0; col < n; ++col) {
                float expected = 0.0f;
                for (int32_t inner = 0; inner < k; ++inner) {
                    expected += a[offsetA + batch * strideA + row * k + inner] * b[offsetB + batch * strideB + inner * n + col];
                }
                EXPECT_EQ(d[offsetD + batch * strideD + row * n + col], expected) << "batch " << batch << " row " << row << " col " << col;
            }
        }
    }
}

}  // namespace

TEST(CublasOperandAlignment, IsTheLowestSetBitOfThePointerCappedAt256) {
    EXPECT_EQ(CublasOperandAlignment::of(0), 256u);
    EXPECT_EQ(CublasOperandAlignment::of(0x10000), 256u);
    EXPECT_EQ(CublasOperandAlignment::of(0x10100), 256u);
    EXPECT_EQ(CublasOperandAlignment::of(0x10080), 128u);
    EXPECT_EQ(CublasOperandAlignment::of(0x10010), 16u);
    EXPECT_EQ(CublasOperandAlignment::of(0x1001e), 2u);
    EXPECT_EQ(CublasOperandAlignment::of(0x10001), 1u);

    const CublasOperandAlignment dense{};
    EXPECT_EQ(dense.a, 256u);
    EXPECT_EQ(dense.d, 256u);
    const CublasOperandAlignment needs16{16, 16, 16, 16};
    EXPECT_TRUE(needs16.satisfiedBy(dense));
    EXPECT_TRUE(needs16.satisfiedBy(needs16));
    EXPECT_FALSE(needs16.satisfiedBy(CublasOperandAlignment{16, 8, 16, 16}));
    EXPECT_FALSE(dense.satisfiedBy(needs16));

    const CublasOperandAlignment mixed{2, 4, 8, 16};
    EXPECT_EQ(mixed.swappedAB(), (CublasOperandAlignment{4, 2, 8, 16}));
    EXPECT_EQ(mixed.swappedAB().swappedAB(), mixed);
}

TEST(CublasOperandAlignment, ParticipatesInKernelRequirementIdentity) {
    const KernelRequirement dense("test-gpu", 3, 5, 5, 7, false, false, false, 5, 7, 7, 7, true);
    const KernelRequirement misaligned = dense.withAlignment(CublasOperandAlignment{2, 2, 256, 2});

    EXPECT_FALSE(dense == misaligned);
    EXPECT_TRUE(dense == dense.withAlignment(CublasOperandAlignment{}));
    EXPECT_EQ(misaligned.alignment, (CublasOperandAlignment{2, 2, 256, 2}));
    EXPECT_EQ(misaligned.rowsA, dense.rowsA);
    EXPECT_EQ(misaligned.ldD, dense.ldD);

    std::unordered_set<KernelRequirement> requirements{dense, misaligned};
    EXPECT_EQ(requirements.size(), 2u);
}

TEST(CublasOperandAlignment, SpanAlignmentIsItsPointerAlignment) {
    Tensor storage(cpuPlacement, TensorDescriptor(DataType::FP16, {64}), 256);
    const TensorSpan whole(storage, 0, 64);
    const TensorSpan odd(storage, 1, 32);
    const TensorSpan eight(storage, 4, 32);
    const CublasOperandAlignment alignment = CublasKernel::operandAlignmentOf(whole, odd, eight, storage);
    EXPECT_EQ(alignment.a, 256u);
    EXPECT_EQ(alignment.b, 2u);
    EXPECT_EQ(alignment.c, 8u);
    EXPECT_EQ(alignment.d, 256u);
}

TEST(CublasOperandAlignment, MisalignedOddShapeStridedBatchedGemmFp16MatchesHostReference) {
    if (!haveCudaDevice())
        GTEST_SKIP() << "CUDA device is required.";
    runMisalignedStridedBatchedGemm<half>(DataType::FP16);
}

TEST(CublasOperandAlignment, MisalignedOddShapeStridedBatchedGemmBf16MatchesHostReference) {
    if (!haveCudaDevice())
        GTEST_SKIP() << "CUDA device is required.";
    runMisalignedStridedBatchedGemm<__nv_bfloat16>(DataType::BF16);
}
