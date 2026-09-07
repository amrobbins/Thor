#include "Utilities/TensorOperations/Ragged/RaggedSequenceConcatenate.h"
#include "Utilities/TensorOperations/Ragged/RowPartition.h"

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/Expression/CudaHelpers.h"
#include "gtest/gtest.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace ThorImplementation;

namespace {

#define REQUIRE_CUDA_DEVICE()                                                                                           \
    do {                                                                                                                 \
        int cudaDeviceCountForTest = 0;                                                                                  \
        const cudaError_t cudaStatusForTest = cudaGetDeviceCount(&cudaDeviceCountForTest);                               \
        if (cudaStatusForTest != cudaSuccess || cudaDeviceCountForTest <= 0) {                                            \
            GTEST_SKIP() << "CUDA device is required for RaggedSequenceConcatenate kernel tests.";                      \
        }                                                                                                                \
    } while (false)

const TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
const TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

template <typename T>
DataType dtypeFor();
template <>
DataType dtypeFor<float>() { return DataType::FP32; }
template <>
DataType dtypeFor<uint8_t>() { return DataType::UINT8; }
template <>
DataType dtypeFor<uint32_t>() { return DataType::UINT32; }
template <>
DataType dtypeFor<uint64_t>() { return DataType::UINT64; }

template <typename T>
Tensor makeGpuTensor(const std::vector<uint64_t>& dimensions, const std::vector<T>& values, Stream& stream) {
    Tensor host(cpuPlacement, TensorDescriptor(dtypeFor<T>(), dimensions));
    if (host.getTotalNumElements() != values.size()) {
        throw std::runtime_error("RaggedSequenceConcatenate test tensor value count mismatch.");
    }
    T* hostValues = host.getMemPtr<T>();
    for (size_t i = 0; i < values.size(); ++i) hostValues[i] = values[i];
    Tensor device(gpuPlacement, TensorDescriptor(dtypeFor<T>(), dimensions));
    device.copyFromAsync(host, stream);
    stream.synchronize();
    return device;
}

template <typename T>
std::vector<T> copyGpuTensor(const Tensor& device, Stream& stream) {
    Tensor host(cpuPlacement, TensorDescriptor(dtypeFor<T>(), device.getDimensions()));
    host.copyFromAsync(device, stream);
    stream.synchronize();
    const T* hostValues = host.getMemPtr<T>();
    return std::vector<T>(hostValues, hostValues + host.getTotalNumElements());
}

class DeviceAllocation {
   public:
    explicit DeviceAllocation(std::size_t bytes) { CUDA_CHECK(cudaMalloc(&pointer, bytes)); }
    DeviceAllocation(const DeviceAllocation&) = delete;
    DeviceAllocation& operator=(const DeviceAllocation&) = delete;
    DeviceAllocation(DeviceAllocation&& other) noexcept : pointer(other.pointer) { other.pointer = nullptr; }
    DeviceAllocation& operator=(DeviceAllocation&& other) noexcept {
        if (this == &other) return *this;
        if (pointer != nullptr) cudaFree(pointer);
        pointer = other.pointer;
        other.pointer = nullptr;
        return *this;
    }
    ~DeviceAllocation() {
        if (pointer != nullptr) cudaFree(pointer);
    }
    void* get() const { return pointer; }

   private:
    void* pointer = nullptr;
};

DeviceAllocation makePointerTable(const std::vector<void*>& pointers, Stream& stream) {
    DeviceAllocation allocation(pointers.size() * sizeof(void*));
    CUDA_CHECK(cudaMemcpyAsync(
        allocation.get(), pointers.data(), pointers.size() * sizeof(void*), cudaMemcpyHostToDevice, stream.getStream()));
    stream.synchronize();
    return allocation;
}

template <typename OffsetT>
void runForwardBackwardCase(DataType expectedOffsetsDataType) {
    constexpr uint64_t batchSize = 3;
    constexpr uint64_t leftCapacity = 6;
    constexpr uint64_t rightCapacity = 7;
    constexpr uint64_t outputCapacity = leftCapacity + rightCapacity;
    constexpr uint64_t width = 2;
    constexpr uint64_t activeOutputValues = 9;
    constexpr float outputSentinel = -7001.0F;
    constexpr float gradientSentinel = -8001.0F;
    const float poison = std::numeric_limits<float>::quiet_NaN();

    Stream stream(0);
    Tensor leftOffsets = makeGpuTensor<OffsetT>({batchSize + 1}, {0, 2, 2, 5}, stream);
    Tensor rightOffsets = makeGpuTensor<OffsetT>({batchSize + 1}, {0, 1, 4, 4}, stream);
    ASSERT_EQ(leftOffsets.getDataType(), expectedOffsetsDataType);
    ASSERT_EQ(rightOffsets.getDataType(), expectedOffsetsDataType);

    std::vector<float> leftValues(leftCapacity * width, poison);
    std::vector<float> rightValues(rightCapacity * width, poison);
    for (uint64_t token = 0; token < 5; ++token) {
        for (uint64_t d = 0; d < width; ++d) leftValues[token * width + d] = 1000.0F + 10.0F * token + d;
    }
    for (uint64_t token = 0; token < 4; ++token) {
        for (uint64_t d = 0; d < width; ++d) rightValues[token * width + d] = 2000.0F + 10.0F * token + d;
    }

    Tensor left = makeGpuTensor<float>({leftCapacity, width}, leftValues, stream);
    Tensor right = makeGpuTensor<float>({rightCapacity, width}, rightValues, stream);
    Tensor output = makeGpuTensor<float>(
        {outputCapacity, width}, std::vector<float>(outputCapacity * width, outputSentinel), stream);
    Tensor outputOffsets = makeGpuTensor<OffsetT>({batchSize + 1}, std::vector<OffsetT>(batchSize + 1, 99), stream);

    DeviceAllocation valueTable = makePointerTable({left.getMemPtr(), right.getMemPtr()}, stream);
    DeviceAllocation offsetsTable = makePointerTable({leftOffsets.getMemPtr(), rightOffsets.getMemPtr()}, stream);

    rowPartitionUploadHostOffsets({0, 3, 6, 9}, outputOffsets, batchSize, stream);
    launchRaggedSequenceConcatenate(output.getMemPtr(),
                                    reinterpret_cast<void**>(valueTable.get()),
                                    reinterpret_cast<void**>(offsetsTable.get()),
                                    2,
                                    sizeof(float),
                                    width,
                                    sizeof(OffsetT),
                                    batchSize,
                                    stream);
    stream.synchronize();

    EXPECT_EQ(copyGpuTensor<OffsetT>(outputOffsets, stream), (std::vector<OffsetT>{0, 3, 6, 9}));
    const std::vector<float> actual = copyGpuTensor<float>(output, stream);
    const std::vector<std::pair<const std::vector<float>*, uint64_t>> expectedTokenSources{
        {&leftValues, 0}, {&leftValues, 1}, {&rightValues, 0},
        {&rightValues, 1}, {&rightValues, 2}, {&rightValues, 3},
        {&leftValues, 2}, {&leftValues, 3}, {&leftValues, 4},
    };
    for (uint64_t outputToken = 0; outputToken < expectedTokenSources.size(); ++outputToken) {
        const auto& [source, sourceToken] = expectedTokenSources[outputToken];
        for (uint64_t d = 0; d < width; ++d) {
            EXPECT_EQ(actual[outputToken * width + d], (*source)[sourceToken * width + d]);
        }
    }
    for (uint64_t scalar = activeOutputValues * width; scalar < actual.size(); ++scalar) {
        EXPECT_EQ(actual[scalar], outputSentinel) << "inactive output scalar " << scalar;
    }

    std::vector<float> upstream(outputCapacity * width, 9999.0F);
    for (uint64_t scalar = 0; scalar < activeOutputValues * width; ++scalar) {
        upstream[scalar] = 3000.0F + static_cast<float>(scalar);
    }
    Tensor upstreamGpu = makeGpuTensor<float>({outputCapacity, width}, upstream, stream);
    Tensor leftGradient = makeGpuTensor<float>(
        {leftCapacity, width}, std::vector<float>(leftCapacity * width, gradientSentinel), stream);
    Tensor rightGradient = makeGpuTensor<float>(
        {rightCapacity, width}, std::vector<float>(rightCapacity * width, gradientSentinel), stream);
    DeviceAllocation gradientTable = makePointerTable({leftGradient.getMemPtr(), rightGradient.getMemPtr()}, stream);

    launchRaggedSequenceConcatenateBackward(reinterpret_cast<void**>(gradientTable.get()),
                                            upstreamGpu.getMemPtr(),
                                            reinterpret_cast<void**>(offsetsTable.get()),
                                            2,
                                            sizeof(float),
                                            width,
                                            sizeof(OffsetT),
                                            batchSize,
                                            stream);
    stream.synchronize();

    const std::vector<float> actualLeftGradient = copyGpuTensor<float>(leftGradient, stream);
    const std::vector<float> actualRightGradient = copyGpuTensor<float>(rightGradient, stream);
    const std::vector<uint64_t> leftOutputToken{0, 1, 6, 7, 8};
    for (uint64_t token = 0; token < leftOutputToken.size(); ++token) {
        for (uint64_t d = 0; d < width; ++d) {
            EXPECT_EQ(actualLeftGradient[token * width + d], upstream[leftOutputToken[token] * width + d]);
        }
    }
    for (uint64_t scalar = 5 * width; scalar < actualLeftGradient.size(); ++scalar) {
        EXPECT_EQ(actualLeftGradient[scalar], gradientSentinel) << "inactive left gradient scalar " << scalar;
    }

    const std::vector<uint64_t> rightOutputToken{2, 3, 4, 5};
    for (uint64_t token = 0; token < rightOutputToken.size(); ++token) {
        for (uint64_t d = 0; d < width; ++d) {
            EXPECT_EQ(actualRightGradient[token * width + d], upstream[rightOutputToken[token] * width + d]);
        }
    }
    for (uint64_t scalar = 4 * width; scalar < actualRightGradient.size(); ++scalar) {
        EXPECT_EQ(actualRightGradient[scalar], gradientSentinel) << "inactive right gradient scalar " << scalar;
    }
}

template <typename OffsetT>
void runAllEmptyCase() {
    constexpr uint64_t batchSize = 3;
    constexpr uint64_t leftCapacity = 2;
    constexpr uint64_t rightCapacity = 3;
    constexpr float sentinel = -9191.0F;

    Stream stream(0);
    Tensor leftOffsets = makeGpuTensor<OffsetT>({batchSize + 1}, {0, 0, 0, 0}, stream);
    Tensor rightOffsets = makeGpuTensor<OffsetT>({batchSize + 1}, {0, 0, 0, 0}, stream);
    Tensor left = makeGpuTensor<float>({leftCapacity}, std::vector<float>(leftCapacity, 11.0F), stream);
    Tensor right = makeGpuTensor<float>({rightCapacity}, std::vector<float>(rightCapacity, 22.0F), stream);
    Tensor output = makeGpuTensor<float>({leftCapacity + rightCapacity},
                                         std::vector<float>(leftCapacity + rightCapacity, sentinel),
                                         stream);
    Tensor outputOffsets = makeGpuTensor<OffsetT>({batchSize + 1}, std::vector<OffsetT>(batchSize + 1, 77), stream);
    DeviceAllocation valueTable = makePointerTable({left.getMemPtr(), right.getMemPtr()}, stream);
    DeviceAllocation offsetsTable = makePointerTable({leftOffsets.getMemPtr(), rightOffsets.getMemPtr()}, stream);

    rowPartitionUploadHostOffsets({0, 0, 0, 0}, outputOffsets, batchSize, stream);
    launchRaggedSequenceConcatenate(output.getMemPtr(),
                                    reinterpret_cast<void**>(valueTable.get()),
                                    reinterpret_cast<void**>(offsetsTable.get()),
                                    2,
                                    sizeof(float),
                                    1,
                                    sizeof(OffsetT),
                                    batchSize,
                                    stream);
    stream.synchronize();
    EXPECT_EQ(copyGpuTensor<OffsetT>(outputOffsets, stream), (std::vector<OffsetT>{0, 0, 0, 0}));
    for (float value : copyGpuTensor<float>(output, stream)) EXPECT_EQ(value, sentinel);

    Tensor leftGradient = makeGpuTensor<float>({leftCapacity}, std::vector<float>(leftCapacity, sentinel), stream);
    Tensor rightGradient = makeGpuTensor<float>({rightCapacity}, std::vector<float>(rightCapacity, sentinel), stream);
    DeviceAllocation gradientTable = makePointerTable({leftGradient.getMemPtr(), rightGradient.getMemPtr()}, stream);
    launchRaggedSequenceConcatenateBackward(reinterpret_cast<void**>(gradientTable.get()),
                                            output.getMemPtr(),
                                            reinterpret_cast<void**>(offsetsTable.get()),
                                            2,
                                            sizeof(float),
                                            1,
                                            sizeof(OffsetT),
                                            batchSize,
                                            stream);
    stream.synchronize();
    for (float value : copyGpuTensor<float>(leftGradient, stream)) EXPECT_EQ(value, sentinel);
    for (float value : copyGpuTensor<float>(rightGradient, stream)) EXPECT_EQ(value, sentinel);
}

template <typename OffsetT>
void runExactPairGroupingBoundaryCase(uint32_t pairCount) {
    ASSERT_GE(pairCount, 2U);
    constexpr uint64_t batchSize = 1;
    constexpr uint8_t gradientSentinel = 0xA7;

    Stream stream(0);
    Tensor sharedOffsets = makeGpuTensor<OffsetT>({2}, {0, 1}, stream);

    std::vector<uint8_t> valuesHost(pairCount);
    std::vector<uint8_t> upstreamHost(pairCount);
    for (uint32_t input = 0; input < pairCount; ++input) {
        valuesHost[input] = static_cast<uint8_t>((input * 41U + 7U) % 251U);
        upstreamHost[input] = static_cast<uint8_t>((input * 23U + 11U) % 251U);
    }

    Tensor values = makeGpuTensor<uint8_t>({pairCount}, valuesHost, stream);
    Tensor output = makeGpuTensor<uint8_t>({pairCount}, std::vector<uint8_t>(pairCount, 0xD3), stream);

    std::vector<void *> valuePointers(pairCount);
    std::vector<void *> offsetPointers(pairCount, sharedOffsets.getMemPtr());
    uint8_t *valuesBase = values.getMemPtr<uint8_t>();
    for (uint32_t input = 0; input < pairCount; ++input) {
        valuePointers[input] = valuesBase + input;
    }
    DeviceAllocation valueTable = makePointerTable(valuePointers, stream);
    DeviceAllocation offsetsTable = makePointerTable(offsetPointers, stream);

    launchRaggedSequenceConcatenate(output.getMemPtr(),
                                    reinterpret_cast<void **>(valueTable.get()),
                                    reinterpret_cast<void **>(offsetsTable.get()),
                                    pairCount,
                                    sizeof(uint8_t),
                                    1,
                                    sizeof(OffsetT),
                                    batchSize,
                                    stream);
    stream.synchronize();
    EXPECT_EQ(copyGpuTensor<uint8_t>(output, stream), valuesHost);

    Tensor upstream = makeGpuTensor<uint8_t>({pairCount}, upstreamHost, stream);
    Tensor gradients = makeGpuTensor<uint8_t>(
        {pairCount}, std::vector<uint8_t>(pairCount, gradientSentinel), stream);
    std::vector<void *> gradientPointers(pairCount);
    uint8_t *gradientsBase = gradients.getMemPtr<uint8_t>();
    for (uint32_t input = 0; input < pairCount; ++input) {
        gradientPointers[input] = gradientsBase + input;
    }
    DeviceAllocation gradientTable = makePointerTable(gradientPointers, stream);

    launchRaggedSequenceConcatenateBackward(reinterpret_cast<void **>(gradientTable.get()),
                                            upstream.getMemPtr(),
                                            reinterpret_cast<void **>(offsetsTable.get()),
                                            pairCount,
                                            sizeof(uint8_t),
                                            1,
                                            sizeof(OffsetT),
                                            batchSize,
                                            stream);
    stream.synchronize();
    EXPECT_EQ(copyGpuTensor<uint8_t>(gradients, stream), upstreamHost);
}

template <typename OffsetT>
void runManyInputsOddWidthCase(uint64_t batchSize, uint32_t numInputs = 9) {
    ASSERT_GT(numInputs, 0U);
    constexpr uint64_t widthBytes = 17;
    constexpr uint8_t valueSentinel = 0xD3;
    constexpr uint8_t gradientSentinel = 0xA7;

    Stream stream(0);
    std::vector<std::vector<OffsetT>> offsetsHost(numInputs);
    std::vector<std::vector<uint8_t>> valuesHost(numInputs);
    std::vector<uint64_t> capacities(numInputs, 0);
    std::vector<uint64_t> activeCounts(numInputs, 0);
    std::vector<Tensor> offsets;
    std::vector<Tensor> values;
    offsets.reserve(numInputs);
    values.reserve(numInputs);

    uint64_t outputCapacity = 0;
    for (uint32_t input = 0; input < numInputs; ++input) {
        auto &inputOffsets = offsetsHost[input];
        inputOffsets.resize(batchSize + 1, 0);
        for (uint64_t row = 0; row < batchSize; ++row) {
            const uint64_t rowLength = (input * 3 + row * 2 + 1) % 4;
            inputOffsets[row + 1] = static_cast<OffsetT>(
                static_cast<uint64_t>(inputOffsets[row]) + rowLength);
        }

        activeCounts[input] = static_cast<uint64_t>(inputOffsets.back());
        capacities[input] = activeCounts[input] + 1 + (input & 1U);
        outputCapacity += capacities[input];

        auto &inputValues = valuesHost[input];
        inputValues.assign(capacities[input] * widthBytes, valueSentinel);
        for (uint64_t token = 0; token < activeCounts[input]; ++token) {
            for (uint64_t byte = 0; byte < widthBytes; ++byte) {
                inputValues[token * widthBytes + byte] =
                    static_cast<uint8_t>((input * 41 + token * 19 + byte * 7) % 251);
            }
        }

        offsets.emplace_back(makeGpuTensor<OffsetT>({batchSize + 1}, inputOffsets, stream));
        values.emplace_back(makeGpuTensor<uint8_t>({capacities[input], widthBytes}, inputValues, stream));
    }

    std::vector<void *> valuePointers;
    std::vector<void *> offsetPointers;
    valuePointers.reserve(numInputs);
    offsetPointers.reserve(numInputs);
    for (uint32_t input = 0; input < numInputs; ++input) {
        valuePointers.push_back(values[input].getMemPtr());
        offsetPointers.push_back(offsets[input].getMemPtr());
    }
    DeviceAllocation valueTable = makePointerTable(valuePointers, stream);
    DeviceAllocation offsetsTable = makePointerTable(offsetPointers, stream);

    Tensor output = makeGpuTensor<uint8_t>(
        {outputCapacity, widthBytes}, std::vector<uint8_t>(outputCapacity * widthBytes, valueSentinel), stream);
    launchRaggedSequenceConcatenate(output.getMemPtr(),
                                    reinterpret_cast<void **>(valueTable.get()),
                                    reinterpret_cast<void **>(offsetsTable.get()),
                                    numInputs,
                                    sizeof(uint8_t),
                                    widthBytes,
                                    sizeof(OffsetT),
                                    batchSize,
                                    stream);
    stream.synchronize();

    std::vector<uint8_t> expected(outputCapacity * widthBytes, valueSentinel);
    std::vector<std::vector<uint64_t>> inputTokenToOutput(numInputs);
    for (uint32_t input = 0; input < numInputs; ++input) {
        inputTokenToOutput[input].resize(activeCounts[input]);
    }

    uint64_t outputToken = 0;
    for (uint64_t row = 0; row < batchSize; ++row) {
        for (uint32_t input = 0; input < numInputs; ++input) {
            const uint64_t begin = static_cast<uint64_t>(offsetsHost[input][row]);
            const uint64_t end = static_cast<uint64_t>(offsetsHost[input][row + 1]);
            for (uint64_t token = begin; token < end; ++token, ++outputToken) {
                inputTokenToOutput[input][token] = outputToken;
                for (uint64_t byte = 0; byte < widthBytes; ++byte) {
                    expected[outputToken * widthBytes + byte] = valuesHost[input][token * widthBytes + byte];
                }
            }
        }
    }
    EXPECT_EQ(copyGpuTensor<uint8_t>(output, stream), expected);

    std::vector<uint8_t> upstream(outputCapacity * widthBytes, 0xEF);
    for (uint64_t token = 0; token < outputToken; ++token) {
        for (uint64_t byte = 0; byte < widthBytes; ++byte) {
            upstream[token * widthBytes + byte] = static_cast<uint8_t>((token * 23 + byte * 11 + 5) % 251);
        }
    }
    Tensor upstreamGpu = makeGpuTensor<uint8_t>({outputCapacity, widthBytes}, upstream, stream);

    std::vector<Tensor> gradients;
    gradients.reserve(numInputs);
    std::vector<void *> gradientPointers;
    gradientPointers.reserve(numInputs);
    for (uint32_t input = 0; input < numInputs; ++input) {
        gradients.emplace_back(makeGpuTensor<uint8_t>(
            {capacities[input], widthBytes},
            std::vector<uint8_t>(capacities[input] * widthBytes, gradientSentinel),
            stream));
        // Exercise nullable gradient destinations while neighboring pair slots in
        // the same CTA remain active.
        gradientPointers.push_back((input == 3 || input == 7) ? nullptr : gradients.back().getMemPtr());
    }
    DeviceAllocation gradientTable = makePointerTable(gradientPointers, stream);

    launchRaggedSequenceConcatenateBackward(reinterpret_cast<void **>(gradientTable.get()),
                                            upstreamGpu.getMemPtr(),
                                            reinterpret_cast<void **>(offsetsTable.get()),
                                            numInputs,
                                            sizeof(uint8_t),
                                            widthBytes,
                                            sizeof(OffsetT),
                                            batchSize,
                                            stream);
    stream.synchronize();

    for (uint32_t input = 0; input < numInputs; ++input) {
        std::vector<uint8_t> expectedGradient(capacities[input] * widthBytes, gradientSentinel);
        if (input != 3 && input != 7) {
            for (uint64_t token = 0; token < activeCounts[input]; ++token) {
                const uint64_t sourceToken = inputTokenToOutput[input][token];
                for (uint64_t byte = 0; byte < widthBytes; ++byte) {
                    expectedGradient[token * widthBytes + byte] = upstream[sourceToken * widthBytes + byte];
                }
            }
        }
        EXPECT_EQ(copyGpuTensor<uint8_t>(gradients[input], stream), expectedGradient) << "input " << input;
    }
}

}  // namespace

TEST(RaggedSequenceConcatenate, ForwardBackwardUint32UsesHostDerivedPartitionAndIgnoresInactiveCapacity) {
    REQUIRE_CUDA_DEVICE();
    runForwardBackwardCase<uint32_t>(DataType::UINT32);
}

TEST(RaggedSequenceConcatenate, ForwardBackwardUint64UsesHostDerivedPartitionAndIgnoresInactiveCapacity) {
    REQUIRE_CUDA_DEVICE();
    runForwardBackwardCase<uint64_t>(DataType::UINT64);
}

TEST(RaggedSequenceConcatenate, AllEmptyRowsUseHostDerivedZeroOffsetsWithoutTouchingValueCapacity) {
    REQUIRE_CUDA_DEVICE();
    runAllEmptyCase<uint32_t>();
    runAllEmptyCase<uint64_t>();
}

TEST(RaggedSequenceConcatenate, ManyInputsOddByteWidthAndNullableGradientsUseSharedPairMetadata) {
    REQUIRE_CUDA_DEVICE();
    runManyInputsOddWidthCase<uint32_t>(5);
    runManyInputsOddWidthCase<uint64_t>(5);
}

TEST(RaggedSequenceConcatenate, AdaptivePairGroupingCoversEverySpecializationForBothOffsetWidths) {
    REQUIRE_CUDA_DEVICE();
    // The 5-row case above covers 45 pairs -> 1/CTA for both offset widths.
    // Nine inputs cross the remaining launch ranges at batches 15, 29 and 57:
    // 135 pairs -> 2/CTA, 261 pairs -> 4/CTA, 513 pairs -> 8/CTA.
    for (uint64_t batchSize : {15ULL, 29ULL, 57ULL}) {
        SCOPED_TRACE(::testing::Message() << "UINT32 batchSize=" << batchSize);
        runManyInputsOddWidthCase<uint32_t>(batchSize);
    }
    for (uint64_t batchSize : {15ULL, 29ULL, 57ULL}) {
        SCOPED_TRACE(::testing::Message() << "UINT64 batchSize=" << batchSize);
        runManyInputsOddWidthCase<uint64_t>(batchSize);
    }
}

TEST(RaggedSequenceConcatenate, AdaptivePairGroupingThresholdBoundariesCoverBothOffsetWidths) {
    REQUIRE_CUDA_DEVICE();
    // One row with one active value per input makes numInputs == totalPairs while
    // respecting the public contract that concatenation has at least two inputs.
    // Reuse one offsets tensor and contiguous value/gradient storage so the large
    // 127/511-input cases remain lightweight. These hit both sides of every
    // launch-policy boundary exactly:
    //   <128 -> 1 pair/CTA, [128,256) -> 2, [256,512) -> 4, >=512 -> 8.
    constexpr uint32_t pairCounts[] = {127, 128, 255, 256, 511, 512};
    for (uint32_t pairCount : pairCounts) {
        SCOPED_TRACE(::testing::Message() << "UINT32 totalPairs=" << pairCount);
        runExactPairGroupingBoundaryCase<uint32_t>(pairCount);
    }
    for (uint32_t pairCount : pairCounts) {
        SCOPED_TRACE(::testing::Message() << "UINT64 totalPairs=" << pairCount);
        runExactPairGroupingBoundaryCase<uint64_t>(pairCount);
    }
}
