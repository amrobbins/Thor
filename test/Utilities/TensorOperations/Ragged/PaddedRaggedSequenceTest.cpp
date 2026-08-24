#include "Utilities/TensorOperations/Ragged/PaddedRaggedSequence.h"

#include "DeepLearning/Implementation/Tensor/RowPartitionDescriptor.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"
#include "gtest/gtest.h"
#include "cuda_runtime.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

using namespace ThorImplementation;

namespace {

#define REQUIRE_CUDA_DEVICE()                                                                                          \
    do {                                                                                                                \
        int cudaDeviceCountForTest = 0;                                                                                 \
        const cudaError_t cudaStatusForTest = cudaGetDeviceCount(&cudaDeviceCountForTest);                              \
        if (cudaStatusForTest != cudaSuccess || cudaDeviceCountForTest <= 0) {                                           \
            GTEST_SKIP() << "CUDA device is required for padded ragged sequence adapter tests.";                       \
        }                                                                                                               \
    } while (false)

const TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
const TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

template <typename T>
DataType dtypeFor();
template <> DataType dtypeFor<float>() { return DataType::FP32; }
template <> DataType dtypeFor<uint32_t>() { return DataType::UINT32; }

template <typename T>
Tensor makeGpuTensor(const std::vector<uint64_t>& dims, const std::vector<T>& values, Stream& stream) {
    Tensor cpu(cpuPlacement, TensorDescriptor(dtypeFor<T>(), dims));
    if (cpu.getTotalNumElements() != values.size()) throw std::runtime_error("PaddedRaggedSequence test value count mismatch.");
    T* ptr = cpu.getMemPtr<T>();
    for (size_t i = 0; i < values.size(); ++i) ptr[i] = values[i];
    Tensor gpu(gpuPlacement, TensorDescriptor(dtypeFor<T>(), dims));
    gpu.copyFromAsync(cpu, stream);
    stream.synchronize();
    return gpu;
}

std::vector<float> copyGpuFloatTensor(const Tensor& gpu, Stream& stream) {
    Tensor cpu(cpuPlacement, TensorDescriptor(DataType::FP32, gpu.getDimensions()));
    cpu.copyFromAsync(gpu, stream);
    stream.synchronize();
    const float* ptr = cpu.getMemPtr<float>();
    return std::vector<float>(ptr, ptr + cpu.getTotalNumElements());
}


template <typename WordT>
Tensor makeGpuRawTensor(DataType dtype, const std::vector<uint64_t>& dims, const std::vector<WordT>& words, Stream& stream) {
    Tensor cpu(cpuPlacement, TensorDescriptor(dtype, dims));
    if (cpu.getTotalNumElements() != words.size() || cpu.getArraySizeInBytes() != words.size() * sizeof(WordT)) {
        throw std::runtime_error("PaddedRaggedSequence raw test tensor size mismatch.");
    }
    std::memcpy(cpu.getMemPtr<void>(), words.data(), words.size() * sizeof(WordT));
    Tensor gpu(gpuPlacement, TensorDescriptor(dtype, dims));
    gpu.copyFromAsync(cpu, stream);
    stream.synchronize();
    return gpu;
}

template <typename WordT>
std::vector<WordT> copyGpuRawTensor(const Tensor& gpu, Stream& stream) {
    Tensor cpu(cpuPlacement, TensorDescriptor(gpu.getDataType(), gpu.getDimensions()));
    cpu.copyFromAsync(gpu, stream);
    stream.synchronize();
    if (cpu.getArraySizeInBytes() != cpu.getTotalNumElements() * sizeof(WordT)) {
        throw std::runtime_error("PaddedRaggedSequence raw test copy size mismatch.");
    }
    std::vector<WordT> words(cpu.getTotalNumElements());
    std::memcpy(words.data(), cpu.getMemPtr<void>(), words.size() * sizeof(WordT));
    return words;
}

template <typename WordT>
void expectBitPreservingAdapterRoundTrip(DataType dtype, WordT storagePoison, WordT outputSentinel) {
    constexpr uint64_t batchSize = 4;
    constexpr uint64_t maxTotalValues = 11;
    constexpr uint64_t maxValuesPerRow = 6;
    constexpr uint64_t channels = 2;
    constexpr uint64_t selectedWidth = 4;
    const std::vector<uint32_t> offsets32{0, 2, 2, 5, 9};
    const std::vector<uint64_t> offsets{0, 2, 2, 5, 9};

    std::vector<WordT> values(maxTotalValues * channels, outputSentinel);
    for (uint64_t i = 0; i < offsets.back() * channels; ++i) {
        values[i] = static_cast<WordT>(static_cast<uint64_t>(0x101u) + i * 37u);
        if (values[i] == WordT{}) values[i] = static_cast<WordT>(1);
    }

    Stream stream(0);
    Tensor gpuValues = makeGpuRawTensor<WordT>(dtype, {maxTotalValues, channels}, values, stream);
    Tensor gpuOffsets = makeGpuTensor<uint32_t>({batchSize + 1}, offsets32, stream);
    RowPartitionRuntime partition(
        gpuOffsets, RowPartitionDescriptor(batchSize, maxTotalValues, DataType::UINT32, maxValuesPerRow));
    partition.setHostActiveValueCount(offsets.back());
    partition.setHostMaxActiveRowLength(selectedWidth);
    const PaddedRaggedSequencePlan plan =
        preparePaddedRaggedSequencePlan(partition, channels, dtype, selectedWidth);

    PaddedRaggedSequence padded(plan, gpuOffsets, gpuPlacement, maxValuesPerRow);
    Tensor storage = padded.getPaddedValuesStorage();
    Tensor poison = makeGpuRawTensor<WordT>(
        dtype, {storage.getTotalNumElements()}, std::vector<WordT>(storage.getTotalNumElements(), storagePoison), stream);
    storage.copyFromAsync(poison, stream);
    stream.synchronize();

    padded.packFrom(gpuValues, stream);
    const std::vector<WordT> actual = copyGpuRawTensor<WordT>(storage, stream);
    std::vector<WordT> expected(plan.valueElements, WordT{});
    for (uint64_t row = 0; row < batchSize; ++row) {
        const uint64_t length = offsets[row + 1] - offsets[row];
        for (uint64_t channel = 0; channel < channels; ++channel) {
            for (uint64_t timestep = 0; timestep < length; ++timestep) {
                expected[(row * channels + channel) * selectedWidth + timestep] =
                    values[(offsets[row] + timestep) * channels + channel];
            }
        }
    }
    for (uint64_t i = 0; i < plan.valueElements; ++i) EXPECT_EQ(actual[i], expected[i]) << "selected raw element " << i;
    for (uint64_t i = plan.valueElements; i < actual.size(); ++i) EXPECT_EQ(actual[i], storagePoison) << "reserved raw element " << i;

    Tensor output = makeGpuRawTensor<WordT>(
        dtype, {maxTotalValues, channels}, std::vector<WordT>(maxTotalValues * channels, outputSentinel), stream);
    padded.unpackTo(output, stream);
    const std::vector<WordT> roundTrip = copyGpuRawTensor<WordT>(output, stream);
    const uint64_t activeElements = offsets.back() * channels;
    for (uint64_t i = 0; i < activeElements; ++i) EXPECT_EQ(roundTrip[i], values[i]) << "round-trip raw element " << i;
    for (uint64_t i = activeElements; i < roundTrip.size(); ++i) EXPECT_EQ(roundTrip[i], outputSentinel) << "packed raw spare element " << i;
}

}  // namespace

TEST(PaddedRaggedSequence, PlanDescribesOneCompactDenseBatchAtSelectedWidth) {
    REQUIRE_CUDA_DEVICE();
    constexpr uint64_t batchSize = 5;
    constexpr uint64_t maxTotalValues = 24;
    constexpr uint64_t maxValuesPerRow = 12;
    constexpr uint64_t channels = 3;
    constexpr uint64_t selectedWidth = 9;
    Stream stream(0);
    Tensor offsets = makeGpuTensor<uint32_t>({batchSize + 1}, {0, 3, 3, 8, 10, 19}, stream);
    RowPartitionRuntime partition(offsets, RowPartitionDescriptor(batchSize, maxTotalValues, DataType::UINT32, maxValuesPerRow));
    partition.setHostActiveValueCount(19);
    partition.setHostMaxActiveRowLength(9);

    const PaddedRaggedSequencePlan plan =
        preparePaddedRaggedSequencePlan(partition, channels, DataType::FP32, selectedWidth);
    EXPECT_EQ(plan.batchSize, batchSize);
    EXPECT_EQ(plan.activeValues, 19u);
    EXPECT_EQ(plan.maxValuesPerRow, maxValuesPerRow);
    EXPECT_EQ(plan.widthCapacity, selectedWidth);
    EXPECT_EQ(plan.denseCapacityValues(), batchSize * selectedWidth);
    EXPECT_EQ(plan.paddingValueCapacity(), batchSize * selectedWidth - 19u);
    EXPECT_EQ(plan.valueElements, batchSize * channels * selectedWidth);
    EXPECT_EQ(plan.valueBytes, batchSize * channels * selectedWidth * sizeof(float));
    EXPECT_EQ(plan.totalWorkspaceBytes(), plan.valueBytes);
}

TEST(PaddedRaggedSequence, PlanRequiresOnlyPublishedScalarsNotAHostOffsetsMirror) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor offsets = makeGpuTensor<uint32_t>({4}, {0, 3, 8, 10}, stream);
    RowPartitionRuntime partition(offsets, RowPartitionDescriptor(3, 12, DataType::UINT32, 8));
    EXPECT_FALSE(partition.getHostOffsetsIfAvailable().has_value());
    partition.setHostActiveValueCount(10);
    partition.setHostMaxActiveRowLength(5);

    const PaddedRaggedSequencePlan plan = preparePaddedRaggedSequencePlan(partition, 4, DataType::FP32, 5);
    EXPECT_EQ(plan.activeValues, 10u);
    EXPECT_EQ(plan.widthCapacity, 5u);
    EXPECT_FALSE(partition.getHostOffsetsIfAvailable().has_value());
}

TEST(PaddedRaggedSequence, AdaptersCanonicalizeSelectedDenseTailsAndRoundTripLogicalPackedPositions) {
    REQUIRE_CUDA_DEVICE();
    constexpr uint64_t batchSize = 5;
    constexpr uint64_t maxTotalValues = 22;
    constexpr uint64_t maxValuesPerRow = 12;
    constexpr uint64_t channels = 2;
    constexpr uint64_t selectedWidth = 9;
    constexpr float packedSparePoison = 9001.0F;
    constexpr float outputSpareSentinel = -777.0F;
    const std::vector<uint32_t> offsets32{0, 3, 3, 8, 10, 19};
    const std::vector<uint64_t> offsets{0, 3, 3, 8, 10, 19};

    std::vector<float> values(maxTotalValues * channels, packedSparePoison);
    for (uint64_t value = 0; value < offsets.back(); ++value) {
        for (uint64_t channel = 0; channel < channels; ++channel) {
            values[value * channels + channel] = static_cast<float>(100 * value + 10 * channel + 1);
        }
    }

    Stream stream(0);
    Tensor gpuValues = makeGpuTensor<float>({maxTotalValues, channels}, values, stream);
    Tensor gpuOffsets = makeGpuTensor<uint32_t>({batchSize + 1}, offsets32, stream);
    RowPartitionRuntime partition(gpuOffsets, RowPartitionDescriptor(batchSize, maxTotalValues, DataType::UINT32, maxValuesPerRow));
    partition.setHostActiveValueCount(offsets.back());
    partition.setHostMaxActiveRowLength(selectedWidth);
    const PaddedRaggedSequencePlan plan = preparePaddedRaggedSequencePlan(partition, channels, DataType::FP32, selectedWidth);

    PaddedRaggedSequence padded(plan, gpuOffsets, gpuPlacement, maxValuesPerRow);
    Tensor storage = padded.getPaddedValuesStorage();
    Tensor hostPoison(cpuPlacement, TensorDescriptor(DataType::FP32, {storage.getTotalNumElements()}));
    float* poison = hostPoison.getMemPtr<float>();
    for (uint64_t i = 0; i < hostPoison.getTotalNumElements(); ++i) poison[i] = -12345.0F;
    storage.copyFromAsync(hostPoison, stream);

    padded.packFrom(gpuValues, stream);
    EXPECT_EQ(padded.paddedTensor().getDimensions(), (std::vector<uint64_t>{batchSize, channels, 1, selectedWidth}));
    const std::vector<float> actual = copyGpuFloatTensor(storage, stream);
    std::vector<float> expected(plan.valueElements, 0.0F);
    for (uint64_t row = 0; row < batchSize; ++row) {
        const uint64_t length = offsets[row + 1] - offsets[row];
        for (uint64_t channel = 0; channel < channels; ++channel) {
            for (uint64_t timestep = 0; timestep < length; ++timestep) {
                expected[(row * channels + channel) * selectedWidth + timestep] =
                    values[(offsets[row] + timestep) * channels + channel];
            }
        }
    }
    for (uint64_t i = 0; i < plan.valueElements; ++i) EXPECT_EQ(actual[i], expected[i]) << "selected padded element " << i;
    for (uint64_t i = plan.valueElements; i < actual.size(); ++i) EXPECT_EQ(actual[i], -12345.0F) << "unused reserved element " << i;

    Tensor output = makeGpuTensor<float>({maxTotalValues, channels}, std::vector<float>(maxTotalValues * channels, outputSpareSentinel), stream);
    padded.unpackTo(output, stream);
    const std::vector<float> roundTrip = copyGpuFloatTensor(output, stream);
    const uint64_t activeElements = offsets.back() * channels;
    for (uint64_t i = 0; i < activeElements; ++i) EXPECT_EQ(roundTrip[i], values[i]) << "active element " << i;
    for (uint64_t i = activeElements; i < roundTrip.size(); ++i) EXPECT_EQ(roundTrip[i], outputSpareSentinel) << "packed spare element " << i;
}


TEST(PaddedRaggedSequence, AdaptersPreserveFp16Bf16AndFp32BitsWhileZeroingTails) {
    REQUIRE_CUDA_DEVICE();
    expectBitPreservingAdapterRoundTrip<uint16_t>(DataType::FP16, 0xD55Du, 0xA33Au);
    expectBitPreservingAdapterRoundTrip<uint16_t>(DataType::BF16, 0xC44Cu, 0xB22Bu);
    expectBitPreservingAdapterRoundTrip<uint32_t>(DataType::FP32, 0xDEADBEEFu, 0xA5A5A5A5u);
}
