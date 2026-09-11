#include "Utilities/TensorOperations/Ragged/PaddedRaggedSequence.h"
#include "Utilities/TensorOperations/Ragged/PaddedRaggedSequenceKernel.h"
#include "Utilities/Expression/StampedEquation.h"

#include "DeepLearning/Implementation/Tensor/RowPartitionDescriptor.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"
#include "gtest/gtest.h"
#include "cuda_runtime.h"

#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
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
template <> DataType dtypeFor<uint64_t>() { return DataType::UINT64; }

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

template <typename WordT, typename OffsetT>
void expectZeroPaddedTailPrimitive(DataType valuesDtype) {
    constexpr uint64_t batchSize = 4;
    constexpr uint64_t channels = 3;
    constexpr uint64_t selectedWidth = 40;
    constexpr uint64_t reservedWidth = 48;
    constexpr uint64_t selectedElements = batchSize * channels * selectedWidth;
    constexpr uint64_t allocatedElements = batchSize * channels * reservedWidth;
    // Exercise an empty row, misaligned long tails that require exact prefix
    // handling before 32-byte bulk stores, and one full-width row.
    const std::vector<uint64_t> offsets64{0, 3, 3, 8, 48};
    std::vector<OffsetT> offsets;
    offsets.reserve(offsets64.size());
    for (uint64_t offset : offsets64) offsets.push_back(static_cast<OffsetT>(offset));

    std::vector<WordT> original(allocatedElements);
    for (uint64_t i = 0; i < original.size(); ++i) {
        WordT word = static_cast<WordT>(static_cast<uint64_t>(0x1357u) + i * 73u);
        if (word == WordT{}) word = static_cast<WordT>(1);
        original[i] = word;
    }

    Stream stream(0);
    Tensor padded = makeGpuRawTensor<WordT>(valuesDtype, {allocatedElements}, original, stream);
    Tensor gpuOffsets = makeGpuTensor<OffsetT>({batchSize + 1}, offsets, stream);
    const PaddedRaggedTailZeroLaunchPlan plan = preparePaddedRaggedTailZeroLaunchPlan(
        batchSize, channels, selectedWidth, valuesDtype, dtypeFor<OffsetT>());

    EXPECT_EQ(plan.selectedValueElements, selectedElements);
    EXPECT_EQ(plan.elementBytes, sizeof(WordT));
    EXPECT_EQ(plan.batchSize, batchSize);
    EXPECT_EQ(plan.channels, channels);
    EXPECT_EQ(plan.widthCapacity, selectedWidth);
    EXPECT_GT(plan.channelBlocks, 0u);
    EXPECT_GT(plan.rowBlocks, 0u);
    EXPECT_GT(plan.threadsPerBlock, 0u);

    launchZeroPaddedRaggedSequenceTail(padded, gpuOffsets, plan, stream);
    const std::vector<WordT> actual = copyGpuRawTensor<WordT>(padded, stream);
    std::vector<WordT> expected = original;
    for (uint64_t row = 0; row < batchSize; ++row) {
        const uint64_t rowLength = offsets64[row + 1] - offsets64[row];
        for (uint64_t channel = 0; channel < channels; ++channel) {
            for (uint64_t timestep = rowLength; timestep < selectedWidth; ++timestep) {
                expected[(row * channels + channel) * selectedWidth + timestep] = WordT{};
            }
        }
    }

    for (uint64_t i = 0; i < selectedElements; ++i) {
        EXPECT_EQ(actual[i], expected[i]) << "selected element " << i;
    }
    for (uint64_t i = selectedElements; i < actual.size(); ++i) {
        EXPECT_EQ(actual[i], original[i]) << "reserved suffix element " << i;
    }
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
    partition.setHostOffsets(offsets);
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
    std::vector<WordT> expected(plan.valueElements, storagePoison);
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


template <typename WordT, typename OffsetT>
void expectPackedToPaddedCase(DataType dtype,
                                   uint64_t channels,
                                   uint64_t selectedWidth,
                                   const std::vector<uint64_t>& rowLengths,
                                   WordT storagePoison) {
    const uint64_t batchSize = rowLengths.size();
    uint64_t activeValues = 0;
    for (uint64_t length : rowLengths) {
        ASSERT_LE(length, selectedWidth);
        activeValues += length;
    }
    const uint64_t maxTotalValues = activeValues + 7;
    const uint64_t reservedWidth = selectedWidth + 7;

    std::vector<uint64_t> offsets64(batchSize + 1, 0);
    for (uint64_t row = 0; row < batchSize; ++row) {
        offsets64[row + 1] = offsets64[row] + rowLengths[row];
    }
    std::vector<OffsetT> offsets;
    offsets.reserve(offsets64.size());
    for (uint64_t offset : offsets64) offsets.push_back(static_cast<OffsetT>(offset));

    const uint64_t packedElements = maxTotalValues * channels;
    std::vector<WordT> packedWords(packedElements, static_cast<WordT>(0x5Au));
    for (uint64_t i = 0; i < activeValues * channels; ++i) {
        WordT word = static_cast<WordT>(static_cast<uint64_t>(0x101u) + i * 131u);
        if (word == WordT{}) word = static_cast<WordT>(1);
        packedWords[i] = word;
    }

    Stream stream(0);
    Tensor gpuPacked = makeGpuRawTensor<WordT>(dtype, {maxTotalValues, channels}, packedWords, stream);
    Tensor gpuOffsets = makeGpuTensor<OffsetT>({batchSize + 1}, offsets, stream);
    RowPartitionRuntime partition(
        gpuOffsets, RowPartitionDescriptor(batchSize, maxTotalValues, dtypeFor<OffsetT>(), reservedWidth));
    partition.setHostOffsets(offsets64);
    const PaddedRaggedSequencePlan paddedPlan =
        preparePaddedRaggedSequencePlan(partition, channels, dtype, selectedWidth);
    PaddedRaggedSequence padded(paddedPlan, gpuOffsets, gpuPlacement, reservedWidth);

    Tensor storage = padded.getPaddedValuesStorage();
    const uint64_t allocatedElements = storage.getTotalNumElements();
    Tensor poison = makeGpuRawTensor<WordT>(
        dtype, {allocatedElements}, std::vector<WordT>(allocatedElements, storagePoison), stream);
    storage.copyFromAsync(poison, stream);
    stream.synchronize();

    const PaddedRaggedPackLaunchPlan launchPlan = preparePaddedRaggedPackLaunchPlan(
        batchSize, maxTotalValues, channels, selectedWidth, dtype, dtypeFor<OffsetT>());
    padded.packFrom(gpuPacked, launchPlan, stream);

    const std::vector<WordT> actual = copyGpuRawTensor<WordT>(storage, stream);
    std::vector<WordT> expected(allocatedElements, storagePoison);
    for (uint64_t row = 0; row < batchSize; ++row) {
        for (uint64_t channel = 0; channel < channels; ++channel) {
            for (uint64_t timestep = 0; timestep < rowLengths[row]; ++timestep) {
                expected[(row * channels + channel) * selectedWidth + timestep] =
                    packedWords[(offsets64[row] + timestep) * channels + channel];
            }
        }
    }
    EXPECT_EQ(actual, expected);
}


template <typename WordT, typename OffsetT>
void expectPaddedToPackedCase(DataType dtype,
                                   uint64_t channels,
                                   uint64_t selectedWidth,
                                   const std::vector<uint64_t>& rowLengths,
                                   WordT storagePoison,
                                   WordT packedSpareSentinel) {
    const uint64_t batchSize = rowLengths.size();
    uint64_t activeValues = 0;
    for (uint64_t length : rowLengths) {
        ASSERT_LE(length, selectedWidth);
        activeValues += length;
    }
    const uint64_t maxTotalValues = activeValues + 11;
    const uint64_t reservedWidth = selectedWidth + 7;

    std::vector<uint64_t> offsets64(batchSize + 1, 0);
    for (uint64_t row = 0; row < batchSize; ++row) {
        offsets64[row + 1] = offsets64[row] + rowLengths[row];
    }
    std::vector<OffsetT> offsets;
    offsets.reserve(offsets64.size());
    for (uint64_t offset : offsets64) offsets.push_back(static_cast<OffsetT>(offset));

    Stream stream(0);
    Tensor gpuOffsets = makeGpuTensor<OffsetT>({batchSize + 1}, offsets, stream);
    RowPartitionRuntime partition(
        gpuOffsets, RowPartitionDescriptor(batchSize, maxTotalValues, dtypeFor<OffsetT>(), reservedWidth));
    partition.setHostOffsets(offsets64);
    const PaddedRaggedSequencePlan paddedPlan =
        preparePaddedRaggedSequencePlan(partition, channels, dtype, selectedWidth);
    PaddedRaggedSequence padded(paddedPlan, gpuOffsets, gpuPlacement, reservedWidth);

    Tensor storage = padded.getPaddedValuesStorage();
    std::vector<WordT> paddedWords(storage.getTotalNumElements(), storagePoison);
    std::vector<WordT> expectedPacked(maxTotalValues * channels, packedSpareSentinel);
    for (uint64_t row = 0; row < batchSize; ++row) {
        for (uint64_t timestep = 0; timestep < rowLengths[row]; ++timestep) {
            for (uint64_t channel = 0; channel < channels; ++channel) {
                WordT word = static_cast<WordT>(
                    static_cast<uint64_t>(0x101u) +
                    (offsets64[row] + timestep) * channels * 131u + channel * 17u);
                if (word == WordT{}) word = static_cast<WordT>(1);
                paddedWords[(row * channels + channel) * selectedWidth + timestep] = word;
                expectedPacked[(offsets64[row] + timestep) * channels + channel] = word;
            }
        }
    }
    Tensor paddedSource = makeGpuRawTensor<WordT>(
        dtype, {storage.getTotalNumElements()}, paddedWords, stream);
    storage.copyFromAsync(paddedSource, stream);
    stream.synchronize();

    Tensor packed = makeGpuRawTensor<WordT>(
        dtype, {maxTotalValues, channels}, expectedPacked, stream);
    // Restore only the active expected words after creating a destination whose
    // spare capacity has a strong sentinel. This keeps expectedPacked as the
    // complete postcondition while independently poisoning the actual output.
    std::vector<WordT> initialPacked(maxTotalValues * channels, packedSpareSentinel);
    Tensor packedPoison = makeGpuRawTensor<WordT>(dtype, {maxTotalValues, channels}, initialPacked, stream);
    packed.copyFromAsync(packedPoison, stream);
    stream.synchronize();

    const PaddedRaggedUnpackLaunchPlan launchPlan = preparePaddedRaggedUnpackLaunchPlan(
        batchSize, maxTotalValues, channels, selectedWidth, dtype, dtypeFor<OffsetT>());
    padded.unpackTo(packed, launchPlan, stream);

    EXPECT_EQ(copyGpuRawTensor<WordT>(packed, stream), expectedPacked);
    EXPECT_EQ(copyGpuRawTensor<WordT>(storage, stream), paddedWords);
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
    partition.setHostOffsets({0, 3, 3, 8, 10, 19});

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

TEST(PaddedRaggedSequence, PlanUsesScalarsDerivedFromAuthoritativeHostOffsets) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor offsets = makeGpuTensor<uint32_t>({4}, {0, 3, 8, 10}, stream);
    RowPartitionRuntime partition(offsets, RowPartitionDescriptor(3, 12, DataType::UINT32, 8));
    EXPECT_FALSE(partition.getHostOffsetsIfAvailable().has_value());
    partition.setHostOffsets({0, 5, 10, 10});

    const PaddedRaggedSequencePlan plan = preparePaddedRaggedSequencePlan(partition, 4, DataType::FP32, 5);
    EXPECT_EQ(plan.activeValues, 10u);
    EXPECT_EQ(plan.widthCapacity, 5u);
    EXPECT_EQ(partition.requireHostOffsets(), (std::vector<uint64_t>{0, 5, 10, 10}));
}

TEST(PaddedRaggedSequence, AdaptersLeaveSelectedDenseTailsUntouchedAndRoundTripLogicalPackedPositions) {
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
    partition.setHostOffsets(offsets);
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
    std::vector<float> expected(plan.valueElements, -12345.0F);
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


TEST(PaddedRaggedSequence, AdaptersPreserveFp16Bf16AndFp32BitsWhileLeavingTailsUntouched) {
    REQUIRE_CUDA_DEVICE();
    expectBitPreservingAdapterRoundTrip<uint16_t>(DataType::FP16, 0xD55Du, 0xA33Au);
    expectBitPreservingAdapterRoundTrip<uint16_t>(DataType::BF16, 0xC44Cu, 0xB22Bu);
    expectBitPreservingAdapterRoundTrip<uint32_t>(DataType::FP32, 0xDEADBEEFu, 0xA5A5A5A5u);
}



TEST(PaddedRaggedSequence, ChannelOnePlannerSelectsDirectCopyGroupingForPackAndUnpack) {
    struct Case {
        uint64_t width;
        uint32_t expectedRowsPerBlock;
    };
    // At batch 16384 the parallelism floor permits the full payload-selected
    // 1..256 row-group ladder. FP32 turns these widths into the first byte range
    // at each ~32-bytes/lane threshold.
    const std::vector<Case> cases{{8, 256}, {9, 128}, {17, 64}, {33, 32}, {65, 16},
                                  {129, 8}, {257, 4}, {513, 2}, {1025, 1}};
    for (const Case& testCase : cases) {
        const PaddedRaggedPackLaunchPlan packPlan = preparePaddedRaggedPackLaunchPlan(
            16384, 20000, 1, testCase.width, DataType::FP32, DataType::UINT32);
        const PaddedRaggedUnpackLaunchPlan unpackPlan = preparePaddedRaggedUnpackLaunchPlan(
            16384, 20000, 1, testCase.width, DataType::FP32, DataType::UINT32);
        EXPECT_EQ(packPlan, unpackPlan);
        EXPECT_TRUE(packPlan.useChannelOneDirectCopy);
        EXPECT_EQ(packPlan.directCopyElementBytes, 4u);
        EXPECT_EQ(packPlan.directCopyElementShift, 2u);
        EXPECT_EQ(packPlan.directCopyMaxRowBytes, testCase.width * 4u);
        EXPECT_EQ(packPlan.directCopyRowsPerBlock, testCase.expectedRowsPerBlock) << "width=" << testCase.width;
        EXPECT_GT(packPlan.directCopyRowBlocks, 0u);
        EXPECT_EQ(packPlan.channelTileBlocks, 0u);
        EXPECT_EQ(packPlan.timestepTileBlocks, 0u);
        EXPECT_EQ(packPlan.rowBlocks, 0u);
        EXPECT_TRUE(packPlan.directCopyUse32BitRowIndexing);
        EXPECT_TRUE(packPlan.directCopyUse32BitSpanIndexing);
        EXPECT_TRUE(packPlan.use32BitCoordinateIndexing);
        EXPECT_TRUE(packPlan.use32BitIndexing);
        EXPECT_EQ(packPlan.selectedValueBytes, 16384u * testCase.width * 4u);
        EXPECT_EQ(packPlan.packedValueBytes, 20000u * 4u);
    }

    const PaddedRaggedPackLaunchPlan hugeRow = preparePaddedRaggedPackLaunchPlan(
        1,
        1,
        1,
        static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) / 4u + 1u,
        DataType::FP32,
        DataType::UINT64);
    EXPECT_TRUE(hugeRow.useChannelOneDirectCopy);
    EXPECT_EQ(hugeRow.directCopyElementShift, 2u);
    EXPECT_TRUE(hugeRow.use32BitCoordinateIndexing);
    EXPECT_TRUE(hugeRow.use32BitIndexing);
    EXPECT_FALSE(hugeRow.directCopyUse32BitSpanIndexing);

    // A byte span can require 64 bits while the element offsets remain 32-bit.
    // Conversely, a very large batch can require 64-bit element/row addressing
    // while each row's byte traversal remains 32-bit. PRS10 keeps those hot
    // arithmetic widths independent.
    const PaddedRaggedPackLaunchPlan hugeWidth = preparePaddedRaggedPackLaunchPlan(
        1,
        1,
        1,
        static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1u,
        DataType::FP32,
        DataType::UINT32);
    EXPECT_FALSE(hugeWidth.use32BitCoordinateIndexing);
    EXPECT_FALSE(hugeWidth.use32BitIndexing);
    EXPECT_TRUE(hugeWidth.directCopyUse32BitRowIndexing);
    EXPECT_FALSE(hugeWidth.directCopyUse32BitSpanIndexing);

    const PaddedRaggedPackLaunchPlan hugeBatch = preparePaddedRaggedPackLaunchPlan(
        static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1u,
        1,
        1,
        1,
        DataType::FP32,
        DataType::UINT32);
    EXPECT_FALSE(hugeBatch.use32BitCoordinateIndexing);
    EXPECT_FALSE(hugeBatch.use32BitIndexing);
    EXPECT_FALSE(hugeBatch.directCopyUse32BitRowIndexing);
    EXPECT_TRUE(hugeBatch.directCopyUse32BitSpanIndexing);
}

TEST(PaddedRaggedSequence, ChannelOneDirectCopyPackAndUnpackPreserveBitsAndInactiveStorage) {
    REQUIRE_CUDA_DEVICE();
    const std::vector<uint64_t> lengths{0, 1, 7, 31, 32, 33, 64, 65};

    expectPackedToPaddedCase<uint16_t, uint32_t>(DataType::FP16, 1, 65, lengths, 0xD55Du);
    expectPackedToPaddedCase<uint16_t, uint64_t>(DataType::BF16, 1, 65, lengths, 0xC44Cu);
    expectPackedToPaddedCase<uint32_t, uint64_t>(DataType::FP32, 1, 65, lengths, 0xDEADBEEFu);
    // Width 64 with FP32 gives a 256-byte padded row stride; the selected
    // offsets below include 32-byte-aligned packed row starts, exercising the
    // ulonglong4_32a bulk path in addition to width-65 alignment fallbacks.
    expectPackedToPaddedCase<uint32_t, uint32_t>(
        DataType::FP32, 1, 64, {0, 8, 16, 32, 64}, 0xD00DFEEDu);

    expectPaddedToPackedCase<uint16_t, uint32_t>(
        DataType::FP16, 1, 65, lengths, 0xD55Du, 0xA33Au);
    expectPaddedToPackedCase<uint16_t, uint64_t>(
        DataType::BF16, 1, 65, lengths, 0xC44Cu, 0xB22Bu);
    expectPaddedToPackedCase<uint32_t, uint64_t>(
        DataType::FP32, 1, 65, lengths, 0xDEADBEEFu, 0xA5A5A5A5u);
    expectPaddedToPackedCase<uint32_t, uint32_t>(
        DataType::FP32, 1, 64, {0, 8, 16, 32, 64}, 0xC001D00Du, 0x94949494u);
}

TEST(PaddedRaggedSequence, PackedToPaddedTiledPlannerCoversTransposeBoundariesAndIndexWidths) {
    const std::vector<std::pair<uint64_t, uint32_t>> tileBoundaries{
        {31, 1}, {32, 1}, {33, 2}, {63, 2}, {64, 2}, {65, 3}};
    for (const auto& [extent, expectedTiles] : tileBoundaries) {
        const PaddedRaggedPackLaunchPlan plan = preparePaddedRaggedPackLaunchPlan(
            7, 512, extent, extent, DataType::FP32, DataType::UINT32);
        EXPECT_FALSE(plan.useChannelOneDirectCopy);
        EXPECT_EQ(plan.channelTileBlocks, expectedTiles) << "channels=" << extent;
        EXPECT_EQ(plan.timestepTileBlocks, expectedTiles) << "width=" << extent;
        EXPECT_EQ(plan.rowBlocks, 7u);
        EXPECT_EQ(plan.directCopyRowsPerBlock, 0u);
        EXPECT_EQ(plan.directCopyRowBlocks, 0u);
        EXPECT_TRUE(plan.use32BitCoordinateIndexing);
        EXPECT_TRUE(plan.use32BitIndexing);
    }

    const PaddedRaggedPackLaunchPlan huge = preparePaddedRaggedPackLaunchPlan(
        65536, 65536, 65536, 2, DataType::FP16, DataType::UINT64);
    EXPECT_TRUE(huge.use32BitCoordinateIndexing);
    EXPECT_FALSE(huge.use32BitIndexing);

    const PaddedRaggedPackLaunchPlan hugeCoordinate = preparePaddedRaggedPackLaunchPlan(
        static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1u,
        1,
        2,
        1,
        DataType::FP16,
        DataType::UINT64);
    EXPECT_FALSE(hugeCoordinate.use32BitCoordinateIndexing);
    EXPECT_FALSE(hugeCoordinate.use32BitIndexing);
}

TEST(PaddedRaggedSequence, PackPlannerChecksOffsetCapacityAndByteExtentOverflowAtStampTime) {
    EXPECT_THROW(
        (void)preparePaddedRaggedPackLaunchPlan(
            1,
            static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1u,
            2,
            1,
            DataType::FP16,
            DataType::UINT32),
        std::invalid_argument);

    const uint64_t maxTotalValues = std::numeric_limits<uint64_t>::max() / 4u + 1u;
    EXPECT_THROW(
        (void)preparePaddedRaggedPackLaunchPlan(
            1, maxTotalValues, 2, 1, DataType::FP32, DataType::UINT64),
        std::invalid_argument);
}

TEST(PaddedRaggedSequence, PackedToPaddedTiledTransposeHandlesEmptyRowsAndTileEdges) {
    REQUIRE_CUDA_DEVICE();
    expectPackedToPaddedCase<uint32_t, uint32_t>(
        DataType::FP32, 31, 33, {0, 31, 32, 33}, 0xDEADBEEFu);
    expectPackedToPaddedCase<uint32_t, uint32_t>(
        DataType::FP32, 33, 31, {0, 1, 30, 31}, 0xD00DFEEDu);
    expectPackedToPaddedCase<uint32_t, uint32_t>(
        DataType::FP32, 65, 65, {0, 31, 32, 33, 63, 64, 65}, 0xC001D00Du);
}

TEST(PaddedRaggedSequence, PackedToPaddedTiledTransposeSupportsUint64OffsetsAndPreservesRawBits) {
    REQUIRE_CUDA_DEVICE();
    const std::vector<uint64_t> lengths{0, 31, 32, 33, 63, 64, 65};
    expectPackedToPaddedCase<uint16_t, uint64_t>(DataType::FP16, 33, 65, lengths, 0xD55Du);
    expectPackedToPaddedCase<uint16_t, uint64_t>(DataType::BF16, 33, 65, lengths, 0xC44Cu);
    expectPackedToPaddedCase<uint32_t, uint64_t>(DataType::FP32, 33, 65, lengths, 0xDEADBEEFu);
}

TEST(PaddedRaggedSequence, PaddedToPackedTiledPlannerMatchesForwardTransposeGeometryAndIndexWidths) {
    const std::vector<std::pair<uint64_t, uint32_t>> tileBoundaries{
        {31, 1}, {32, 1}, {33, 2}, {63, 2}, {64, 2}, {65, 3}};
    for (const auto& [extent, expectedTiles] : tileBoundaries) {
        const PaddedRaggedUnpackLaunchPlan unpackPlan = preparePaddedRaggedUnpackLaunchPlan(
            7, 512, extent, extent, DataType::FP32, DataType::UINT32);
        const PaddedRaggedPackLaunchPlan packPlan = preparePaddedRaggedPackLaunchPlan(
            7, 512, extent, extent, DataType::FP32, DataType::UINT32);
        EXPECT_EQ(unpackPlan, packPlan);
        EXPECT_EQ(unpackPlan.channelTileBlocks, expectedTiles) << "channels=" << extent;
        EXPECT_EQ(unpackPlan.timestepTileBlocks, expectedTiles) << "width=" << extent;
        EXPECT_TRUE(unpackPlan.use32BitCoordinateIndexing);
        EXPECT_TRUE(unpackPlan.use32BitIndexing);
    }

    const PaddedRaggedUnpackLaunchPlan huge = preparePaddedRaggedUnpackLaunchPlan(
        65536, 65536, 65536, 2, DataType::FP16, DataType::UINT64);
    EXPECT_TRUE(huge.use32BitCoordinateIndexing);
    EXPECT_FALSE(huge.use32BitIndexing);
}

TEST(PaddedRaggedSequence, PaddedToPackedAdapterHandlesDirectC1AndTiledEdges) {
    REQUIRE_CUDA_DEVICE();
    expectPaddedToPackedCase<uint32_t, uint32_t>(
        DataType::FP32, 1, 33, {0, 31, 32, 33}, 0xFEEDBEEFu, 0x94949494u);
    expectPaddedToPackedCase<uint32_t, uint32_t>(
        DataType::FP32, 31, 33, {0, 31, 32, 33}, 0xDEADBEEFu, 0xA5A5A5A5u);
    expectPaddedToPackedCase<uint32_t, uint32_t>(
        DataType::FP32, 33, 31, {0, 1, 30, 31}, 0xD00DFEEDu, 0xB6B6B6B6u);
    expectPaddedToPackedCase<uint32_t, uint32_t>(
        DataType::FP32, 65, 65, {0, 31, 32, 33, 63, 64, 65}, 0xC001D00Du, 0xC7C7C7C7u);
}

TEST(PaddedRaggedSequence, PaddedToPackedTiledTransposeSupportsUint64OffsetsAndPreservesRawBits) {
    REQUIRE_CUDA_DEVICE();
    const std::vector<uint64_t> lengths{0, 31, 32, 33, 63, 64, 65};
    expectPaddedToPackedCase<uint16_t, uint64_t>(
        DataType::FP16, 33, 65, lengths, 0xD55Du, 0xA33Au);
    expectPaddedToPackedCase<uint16_t, uint64_t>(
        DataType::BF16, 33, 65, lengths, 0xC44Cu, 0xB22Bu);
    expectPaddedToPackedCase<uint32_t, uint64_t>(
        DataType::FP32, 33, 65, lengths, 0xDEADBEEFu, 0xA5A5A5A5u);
}

TEST(PaddedRaggedSequence, StampedPackPrecomputesOneTiledLaunchPlanPerWidth) {
    REQUIRE_CUDA_DEVICE();
    constexpr uint64_t batchSize = 4;
    constexpr uint64_t maxTotalValues = 20;
    constexpr uint64_t maxValuesPerRow = 16;
    constexpr uint64_t channels = 33;
    const std::vector<uint32_t> offsets32{0, 3, 3, 8, 10};
    const std::vector<uint64_t> offsets64{0, 3, 3, 8, 10};

    Stream stream(0);
    Tensor gpuOffsets = makeGpuTensor<uint32_t>({batchSize + 1}, offsets32, stream);
    RowPartitionRuntime partition(
        gpuOffsets, RowPartitionDescriptor(batchSize, maxTotalValues, DataType::UINT32, maxValuesPerRow));
    partition.setHostOffsets(offsets64);
    auto padded = std::make_shared<PaddedRaggedSequence>(
        preparePaddedRaggedSequencePlan(partition, channels, DataType::FP32, 8),
        gpuOffsets,
        gpuPlacement,
        maxValuesPerRow);
    Tensor packed(gpuPlacement, TensorDescriptor(DataType::FP32, {maxTotalValues, channels}));
    CompiledPaddedRaggedSequenceLayout layout{
        DataType::FP32, DataType::UINT32, batchSize, maxTotalValues, maxValuesPerRow, channels};

    StampedPaddedRaggedPack stamped(layout, {8, 16}, packed, padded, stream);
    EXPECT_EQ(stamped.preStampedWidthCount(), 2u);
}

TEST(PaddedRaggedSequence, StampedUnpackPrecomputesOneTiledLaunchPlanPerWidth) {
    REQUIRE_CUDA_DEVICE();
    constexpr uint64_t batchSize = 4;
    constexpr uint64_t maxTotalValues = 20;
    constexpr uint64_t maxValuesPerRow = 16;
    constexpr uint64_t channels = 33;
    const std::vector<uint32_t> offsets32{0, 3, 3, 8, 10};
    const std::vector<uint64_t> offsets64{0, 3, 3, 8, 10};

    Stream stream(0);
    Tensor gpuOffsets = makeGpuTensor<uint32_t>({batchSize + 1}, offsets32, stream);
    RowPartitionRuntime partition(
        gpuOffsets, RowPartitionDescriptor(batchSize, maxTotalValues, DataType::UINT32, maxValuesPerRow));
    partition.setHostOffsets(offsets64);
    auto padded = std::make_shared<PaddedRaggedSequence>(
        preparePaddedRaggedSequencePlan(partition, channels, DataType::FP32, 8),
        gpuOffsets,
        gpuPlacement,
        maxValuesPerRow);
    Tensor packed(gpuPlacement, TensorDescriptor(DataType::FP32, {maxTotalValues, channels}));
    CompiledPaddedRaggedSequenceLayout layout{
        DataType::FP32, DataType::UINT32, batchSize, maxTotalValues, maxValuesPerRow, channels};

    StampedPaddedRaggedUnpack stamped(layout, {8, 16}, padded, packed, stream);
    EXPECT_EQ(stamped.preStampedWidthCount(), 2u);
}

TEST(PaddedRaggedSequence, TailZeroPlannerPrecomputesGroupingLaneAndIndexPolicy) {
    const PaddedRaggedTailZeroLaunchPlan small = preparePaddedRaggedTailZeroLaunchPlan(
        64, 3, 8, DataType::FP32, DataType::UINT32);
    EXPECT_EQ(small.selectedValueElements, 64u * 3u * 8u);
    EXPECT_EQ(small.selectedValueBytes, 64u * 3u * 8u * 4u);
    EXPECT_EQ(small.maxTailBytes, 32u);
    EXPECT_EQ(small.elementShift, 2u);
    EXPECT_EQ(small.lanesPerChannel, 1u);
    EXPECT_EQ(small.laneShift, 0u);
    EXPECT_EQ(small.channelsPerBlock, 3u);
    EXPECT_EQ(small.channelBlocks, 1u);
    EXPECT_EQ(small.rowBlocks, 64u);
    EXPECT_EQ(small.threadsPerBlock, 32u);
    EXPECT_TRUE(small.use32BitCoordinateIndexing);
    EXPECT_TRUE(small.use32BitIndexing);
    EXPECT_TRUE(small.use32BitSpanIndexing);

    const PaddedRaggedTailZeroLaunchPlan grouped = preparePaddedRaggedTailZeroLaunchPlan(
        64, 100, 64, DataType::FP32, DataType::UINT64);
    EXPECT_EQ(grouped.maxTailBytes, 256u);
    EXPECT_EQ(grouped.elementShift, 2u);
    EXPECT_EQ(grouped.lanesPerChannel, 8u);
    EXPECT_EQ(grouped.laneShift, 3u);
    EXPECT_EQ(grouped.channelsPerBlock, 32u);
    EXPECT_EQ(grouped.channelBlocks, 4u);
    EXPECT_EQ(grouped.threadsPerBlock, 256u);
    EXPECT_TRUE(grouped.use32BitCoordinateIndexing);
    EXPECT_TRUE(grouped.use32BitIndexing);
    EXPECT_TRUE(grouped.use32BitSpanIndexing);

    constexpr uint64_t widthPastFourGiBBytes =
        static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) / 4u + 1u;
    const PaddedRaggedTailZeroLaunchPlan wideSpan = preparePaddedRaggedTailZeroLaunchPlan(
        1, 1, widthPastFourGiBBytes, DataType::FP32, DataType::UINT64);
    EXPECT_TRUE(wideSpan.use32BitCoordinateIndexing);
    EXPECT_TRUE(wideSpan.use32BitIndexing);
    EXPECT_FALSE(wideSpan.use32BitSpanIndexing);
    EXPECT_EQ(wideSpan.lanesPerChannel, 256u);
    EXPECT_EQ(wideSpan.laneShift, 8u);
    EXPECT_EQ(wideSpan.channelsPerBlock, 1u);
    EXPECT_EQ(wideSpan.threadsPerBlock, 256u);

    const PaddedRaggedTailZeroLaunchPlan hugePhysical = preparePaddedRaggedTailZeroLaunchPlan(
        65536, 65536, 2, DataType::FP16, DataType::UINT64);
    EXPECT_TRUE(hugePhysical.use32BitCoordinateIndexing);
    EXPECT_FALSE(hugePhysical.use32BitIndexing);
    EXPECT_TRUE(hugePhysical.use32BitSpanIndexing);
}

TEST(PaddedRaggedSequence, ZeroTailPrimitiveSupportsUint32AndUint64OffsetsEmptyRowsAndReservedSuffix) {
    REQUIRE_CUDA_DEVICE();
    expectZeroPaddedTailPrimitive<uint32_t, uint32_t>(DataType::FP32);
    expectZeroPaddedTailPrimitive<uint32_t, uint64_t>(DataType::FP32);
}

TEST(PaddedRaggedSequence, ZeroTailPrimitivePreservesActiveFp16Bf16AndFp32Bits) {
    REQUIRE_CUDA_DEVICE();
    expectZeroPaddedTailPrimitive<uint16_t, uint32_t>(DataType::FP16);
    expectZeroPaddedTailPrimitive<uint16_t, uint32_t>(DataType::BF16);
    expectZeroPaddedTailPrimitive<uint32_t, uint32_t>(DataType::FP32);
}

TEST(PaddedRaggedSequence, ZeroTailPrimitiveWidthZeroIsNoOp) {
    REQUIRE_CUDA_DEVICE();
    constexpr uint64_t batchSize = 3;
    constexpr uint64_t channels = 2;
    const std::vector<uint32_t> offsets{0, 0, 0, 0};
    const std::vector<uint32_t> original{
        0x10000001u, 0x10000002u, 0x10000003u, 0x10000004u,
        0x10000005u, 0x10000006u, 0x10000007u, 0x10000008u};

    Stream stream(0);
    Tensor padded = makeGpuRawTensor<uint32_t>(DataType::FP32, {original.size()}, original, stream);
    Tensor gpuOffsets = makeGpuTensor<uint32_t>({batchSize + 1}, offsets, stream);
    const PaddedRaggedTailZeroLaunchPlan plan = preparePaddedRaggedTailZeroLaunchPlan(
        batchSize, channels, 0, DataType::FP32, DataType::UINT32);

    EXPECT_TRUE(plan.empty());
    EXPECT_EQ(plan.selectedValueElements, 0u);
    EXPECT_EQ(plan.channelBlocks, 0u);
    EXPECT_EQ(plan.rowBlocks, 0u);
    EXPECT_EQ(plan.threadsPerBlock, 0u);

    launchZeroPaddedRaggedSequenceTail(padded, gpuOffsets, plan, stream);
    EXPECT_EQ(copyGpuRawTensor<uint32_t>(padded, stream), original);
}


TEST(PaddedRaggedSequence, StampedTailSanitizerSwitchesPrecomputedWidthsWithoutAllocating) {
    REQUIRE_CUDA_DEVICE();
    constexpr uint64_t batchSize = 4;
    constexpr uint64_t maxTotalValues = 20;
    constexpr uint64_t maxValuesPerRow = 16;
    constexpr uint64_t channels = 2;
    constexpr uint64_t reservedWidth = 16;
    const std::vector<uint32_t> offsets32{0, 3, 3, 8, 10};
    const std::vector<uint64_t> offsets{0, 3, 3, 8, 10};

    Stream stream(0);
    Tensor gpuOffsets = makeGpuTensor<uint32_t>({batchSize + 1}, offsets32, stream);
    RowPartitionRuntime partition(
        gpuOffsets, RowPartitionDescriptor(batchSize, maxTotalValues, DataType::UINT32, maxValuesPerRow));
    partition.setHostOffsets(offsets);

    auto padded = std::make_shared<PaddedRaggedSequence>(
        preparePaddedRaggedSequencePlan(partition, channels, DataType::FP32, 8),
        gpuOffsets,
        gpuPlacement,
        reservedWidth);
    auto sanitizer = std::make_shared<StampedSanitizePaddedRaggedTail>(
        padded, std::vector<uint64_t>{8, 16}, stream);
    EXPECT_EQ(sanitizer->preStampedWidthCount(), 2u);

    StampedExecutionPlan execution_plan(
        {StampedExecutionStage(sanitizer)}, std::unordered_map<std::string, Tensor>{}, stream);
    EXPECT_EQ(execution_plan.stageKindNames(), (std::vector<std::string>{"SanitizePaddedRaggedTail"}));

    Tensor storage = padded->getPaddedValuesStorage();
    void* const storage_address = storage.getMemPtr<void>();
    const uint64_t allocated_bytes = padded->allocatedValueBytes();

    auto verify_width = [&](uint64_t width, uint32_t seed) {
        const uint64_t allocated_elements = storage.getTotalNumElements();
        std::vector<uint32_t> original(allocated_elements);
        for (uint64_t i = 0; i < allocated_elements; ++i) {
            original[i] = seed + static_cast<uint32_t>(i * 17u + 1u);
            if (original[i] == 0u) original[i] = 1u;
        }
        Tensor poison = makeGpuRawTensor<uint32_t>(DataType::FP32, {allocated_elements}, original, stream);
        storage.copyFromAsync(poison, stream);
        stream.synchronize();

        padded->reconfigure(preparePaddedRaggedSequencePlan(partition, channels, DataType::FP32, width));
        execution_plan.run();

        EXPECT_EQ(padded->getPaddedValuesStorage().getMemPtr<void>(), storage_address);
        EXPECT_EQ(padded->allocatedValueBytes(), allocated_bytes);
        const std::vector<uint32_t> actual = copyGpuRawTensor<uint32_t>(storage, stream);
        std::vector<uint32_t> expected = original;
        for (uint64_t row = 0; row < batchSize; ++row) {
            const uint64_t row_length = offsets[row + 1] - offsets[row];
            for (uint64_t channel = 0; channel < channels; ++channel) {
                for (uint64_t timestep = row_length; timestep < width; ++timestep) {
                    expected[(row * channels + channel) * width + timestep] = 0u;
                }
            }
        }
        const uint64_t selected_elements = batchSize * channels * width;
        for (uint64_t i = 0; i < selected_elements; ++i) {
            EXPECT_EQ(actual[i], expected[i]) << "width=" << width << " selected element=" << i;
        }
        for (uint64_t i = selected_elements; i < actual.size(); ++i) {
            EXPECT_EQ(actual[i], original[i]) << "width=" << width << " reserved suffix element=" << i;
        }
    };

    verify_width(8, 0x31000000u);
    verify_width(16, 0x52000000u);
}

TEST(PaddedRaggedSequence, StampedTailSanitizerWidthZeroIsNoOp) {
    REQUIRE_CUDA_DEVICE();
    constexpr uint64_t batchSize = 3;
    constexpr uint64_t maxTotalValues = 12;
    constexpr uint64_t maxValuesPerRow = 8;
    constexpr uint64_t channels = 2;
    constexpr uint64_t reservedWidth = 8;
    const std::vector<uint32_t> offsets32{0, 0, 0, 0};
    const std::vector<uint64_t> offsets{0, 0, 0, 0};

    Stream stream(0);
    Tensor gpuOffsets = makeGpuTensor<uint32_t>({batchSize + 1}, offsets32, stream);
    RowPartitionRuntime partition(
        gpuOffsets, RowPartitionDescriptor(batchSize, maxTotalValues, DataType::UINT32, maxValuesPerRow));
    partition.setHostOffsets(offsets);
    auto padded = std::make_shared<PaddedRaggedSequence>(
        preparePaddedRaggedSequencePlan(partition, channels, DataType::FP32, 0),
        gpuOffsets,
        gpuPlacement,
        reservedWidth);
    auto sanitizer = std::make_shared<StampedSanitizePaddedRaggedTail>(
        padded, std::vector<uint64_t>{8}, stream);

    Tensor storage = padded->getPaddedValuesStorage();
    std::vector<uint32_t> original(storage.getTotalNumElements());
    for (uint64_t i = 0; i < original.size(); ++i) original[i] = 0x71000000u + static_cast<uint32_t>(i + 1);
    Tensor poison = makeGpuRawTensor<uint32_t>(DataType::FP32, {original.size()}, original, stream);
    storage.copyFromAsync(poison, stream);
    stream.synchronize();

    sanitizer->run();
    EXPECT_EQ(copyGpuRawTensor<uint32_t>(storage, stream), original);
}
