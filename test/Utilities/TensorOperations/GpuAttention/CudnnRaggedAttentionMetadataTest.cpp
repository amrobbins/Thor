#include "Utilities/TensorOperations/GpuAttention/CudnnRaggedAttentionMetadata.h"

#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <limits>
#include <vector>

using namespace ThorImplementation;

namespace {

#define REQUIRE_CUDA_DEVICE()                                                                                           \
    do {                                                                                                                 \
        int cudaDeviceCountForTest = 0;                                                                                  \
        const cudaError_t cudaStatusForTest = cudaGetDeviceCount(&cudaDeviceCountForTest);                              \
        if (cudaStatusForTest != cudaSuccess || cudaDeviceCountForTest <= 0) {                                           \
            GTEST_SKIP() << "CUDA device is required for cuDNN ragged attention metadata tests.";                       \
        }                                                                                                                \
    } while (false)

TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

template <typename T>
DataType dtypeFor();

template <>
DataType dtypeFor<uint32_t>() {
    return DataType::UINT32;
}

template <>
DataType dtypeFor<uint64_t>() {
    return DataType::UINT64;
}

template <>
DataType dtypeFor<int32_t>() {
    return DataType::INT32;
}

template <typename T>
Tensor makeGpuTensor(const std::vector<uint64_t>& dimensions, const std::vector<T>& values, Stream& stream) {
    Tensor cpu(cpuPlacement, TensorDescriptor(dtypeFor<T>(), dimensions));
    EXPECT_EQ(cpu.getTotalNumElements(), values.size());
    T* cpuValues = cpu.getMemPtr<T>();
    for (size_t i = 0; i < values.size(); ++i) {
        cpuValues[i] = values[i];
    }

    Tensor gpu(gpuPlacement, TensorDescriptor(dtypeFor<T>(), dimensions));
    gpu.copyFromAsync(cpu, stream);
    stream.synchronize();
    return gpu;
}

template <typename T>
std::vector<T> copyGpuTensor(const Tensor& gpu, Stream& stream) {
    Tensor cpu = gpu.clone(cpuPlacement);
    cpu.copyFromAsync(gpu, stream);
    stream.synchronize();

    std::vector<T> values(cpu.getTotalNumElements());
    const T* cpuValues = cpu.getMemPtr<T>();
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = cpuValues[i];
    }
    return values;
}

template <typename OffsetT>
void convertsCanonicalOffsetsToLengthsAndIndependentElementOffsets() {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t batchSize = 4;
    constexpr uint64_t maxTotalValues = 8;
    Tensor canonicalOffsets = makeGpuTensor<OffsetT>({batchSize + 1}, {0, 2, 2, 5, 6}, stream);
    Tensor sequenceLengths(gpuPlacement, TensorDescriptor(DataType::INT32, {batchSize}));
    Tensor qElementOffsets(gpuPlacement, TensorDescriptor(DataType::INT32, {batchSize + 1}));
    Tensor oElementOffsets(gpuPlacement, TensorDescriptor(DataType::INT32, {batchSize + 1}));

    // Deliberately use different scales. This is the important Dqk != Dv case
    // that the old shared-offset ragged attention plumbing could not express.
    convertCanonicalRowPartitionForCudnnAttention(canonicalOffsets,
                                                   batchSize,
                                                   maxTotalValues,
                                                   12,
                                                   20,
                                                   sequenceLengths,
                                                   qElementOffsets,
                                                   oElementOffsets,
                                                   stream);

    EXPECT_EQ(copyGpuTensor<int32_t>(sequenceLengths, stream), (std::vector<int32_t>{2, 0, 3, 1}));
    EXPECT_EQ(copyGpuTensor<int32_t>(qElementOffsets, stream), (std::vector<int32_t>{0, 24, 24, 60, 72}));
    EXPECT_EQ(copyGpuTensor<int32_t>(oElementOffsets, stream), (std::vector<int32_t>{0, 40, 40, 100, 120}));
}

}  // namespace

TEST(CudnnRaggedAttentionMetadata, ConvertsUint32CanonicalOffsets) {
    convertsCanonicalOffsetsToLengthsAndIndependentElementOffsets<uint32_t>();
}

TEST(CudnnRaggedAttentionMetadata, ConvertsUint64CanonicalOffsets) {
    convertsCanonicalOffsetsToLengthsAndIndependentElementOffsets<uint64_t>();
}

TEST(CudnnRaggedAttentionMetadata, RejectsNonCanonicalOffsetDType) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor canonicalOffsets = makeGpuTensor<int32_t>({3}, {0, 1, 2}, stream);
    Tensor sequenceLengths(gpuPlacement, TensorDescriptor(DataType::INT32, {2}));
    Tensor firstOffsets(gpuPlacement, TensorDescriptor(DataType::INT32, {3}));
    Tensor secondOffsets(gpuPlacement, TensorDescriptor(DataType::INT32, {3}));

    EXPECT_THROW(convertCanonicalRowPartitionForCudnnAttention(
                     canonicalOffsets, 2, 2, 4, 4, sequenceLengths, firstOffsets, secondOffsets, stream),
                 std::invalid_argument);
}

TEST(CudnnRaggedAttentionMetadata, RejectsMismatchedScratchShapes) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor canonicalOffsets = makeGpuTensor<uint32_t>({4}, {0, 1, 1, 2}, stream);
    Tensor badSequenceLengths(gpuPlacement, TensorDescriptor(DataType::INT32, {4}));
    Tensor firstOffsets(gpuPlacement, TensorDescriptor(DataType::INT32, {4}));
    Tensor secondOffsets(gpuPlacement, TensorDescriptor(DataType::INT32, {4}));

    EXPECT_THROW(convertCanonicalRowPartitionForCudnnAttention(
                     canonicalOffsets, 3, 3, 4, 4, badSequenceLengths, firstOffsets, secondOffsets, stream),
                 std::invalid_argument);
}

TEST(CudnnRaggedAttentionMetadata, RejectsElementOffsetCapacityThatCannotFitInt32) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor canonicalOffsets = makeGpuTensor<uint32_t>({2}, {0, 1}, stream);
    Tensor sequenceLengths(gpuPlacement, TensorDescriptor(DataType::INT32, {1}));
    Tensor firstOffsets(gpuPlacement, TensorDescriptor(DataType::INT32, {2}));
    Tensor secondOffsets(gpuPlacement, TensorDescriptor(DataType::INT32, {2}));
    const uint64_t maxTotalValues = static_cast<uint64_t>(std::numeric_limits<int32_t>::max());

    EXPECT_THROW(convertCanonicalRowPartitionForCudnnAttention(
                     canonicalOffsets, 1, maxTotalValues, 2, 1, sequenceLengths, firstOffsets, secondOffsets, stream),
                 std::invalid_argument);
}
