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
#include <type_traits>
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
using CopySpanT = std::conditional_t<sizeof(OffsetT) == sizeof(uint32_t),
                                     RaggedSequenceCopySpan32,
                                     RaggedSequenceCopySpan64>;

template <typename OffsetT>
std::vector<CopySpanT<OffsetT>> makeCopySpans(const std::vector<std::vector<OffsetT>>& inputOffsets,
                                               uint64_t validBatchSize) {
    static_assert(sizeof(OffsetT) == sizeof(uint32_t) || sizeof(OffsetT) == sizeof(uint64_t));
    if (inputOffsets.empty()) throw std::invalid_argument("RaggedSequenceConcatenate test copy plan needs inputs.");
    const uint64_t batchSize = inputOffsets.front().size() - 1;
    if (validBatchSize > batchSize) throw std::invalid_argument("RaggedSequenceConcatenate test valid batch is too large.");

    std::vector<CopySpanT<OffsetT>> spans;
    spans.reserve(static_cast<std::size_t>(validBatchSize) * inputOffsets.size());
    uint64_t outputBegin = 0;
    for (uint64_t row = 0; row < validBatchSize; ++row) {
        for (uint32_t input = 0; input < inputOffsets.size(); ++input) {
            const auto& offsets = inputOffsets[input];
            if (offsets.size() != batchSize + 1) throw std::invalid_argument("RaggedSequenceConcatenate test partition size mismatch.");
            const uint64_t sourceBegin = static_cast<uint64_t>(offsets[row]);
            const uint64_t sourceEnd = static_cast<uint64_t>(offsets[row + 1]);
            if (sourceBegin > sourceEnd) throw std::invalid_argument("RaggedSequenceConcatenate test partition is not monotonic.");
            const uint64_t valueCount = sourceEnd - sourceBegin;
            if (valueCount != 0) {
                if constexpr (sizeof(OffsetT) == sizeof(uint32_t)) {
                    if (sourceBegin > std::numeric_limits<uint32_t>::max() ||
                        outputBegin > std::numeric_limits<uint32_t>::max() ||
                        valueCount > std::numeric_limits<uint32_t>::max()) {
                        throw std::overflow_error("RaggedSequenceConcatenate UINT32 test copy span overflow.");
                    }
                    spans.push_back(RaggedSequenceCopySpan32{
                        input,
                        static_cast<uint32_t>(sourceBegin),
                        static_cast<uint32_t>(outputBegin),
                        static_cast<uint32_t>(valueCount)});
                } else {
                    spans.push_back(RaggedSequenceCopySpan64{input, 0, sourceBegin, outputBegin, valueCount});
                }
            }
            if (outputBegin > std::numeric_limits<uint64_t>::max() - valueCount)
                throw std::overflow_error("RaggedSequenceConcatenate test output placement overflow.");
            outputBegin += valueCount;
        }
    }
    return spans;
}

template <typename SpanT>
DeviceAllocation makeCopySpanTable(const std::vector<SpanT>& spans, Stream& stream) {
    const std::size_t bytes = spans.size() * sizeof(SpanT);
    DeviceAllocation allocation(bytes == 0 ? sizeof(SpanT) : bytes);
    if (bytes != 0) {
        CUDA_CHECK(cudaMemcpyAsync(
            allocation.get(), spans.data(), bytes, cudaMemcpyHostToDevice, stream.getStream()));
        stream.synchronize();
    }
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
    const std::vector<std::vector<OffsetT>> offsetsHost{{0, 2, 2, 5}, {0, 1, 4, 4}};
    EXPECT_EQ(dtypeFor<OffsetT>(), expectedOffsetsDataType);
    const auto copySpansHost = makeCopySpans(offsetsHost, batchSize);
    ASSERT_EQ(copySpansHost.size(), 4U);  // Exactly the four non-empty (row,input) spans.
    DeviceAllocation copySpans = makeCopySpanTable(copySpansHost, stream);

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

    rowPartitionUploadHostOffsets({0, 3, 6, 9}, outputOffsets, batchSize, stream);
    launchRaggedSequenceConcatenate(output.getMemPtr(),
                                    reinterpret_cast<void**>(valueTable.get()),
                                    copySpans.get(),
                                    copySpansHost.size(),
                                    sizeof(float),
                                    width,
                                    sizeof(OffsetT),
                                    activeOutputValues,
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
                                            copySpans.get(),
                                            copySpansHost.size(),
                                            sizeof(float),
                                            width,
                                            sizeof(OffsetT),
                                            activeOutputValues,
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
    const std::vector<std::vector<OffsetT>> offsetsHost{{0, 0, 0, 0}, {0, 0, 0, 0}};
    const auto copySpansHost = makeCopySpans(offsetsHost, batchSize);
    ASSERT_TRUE(copySpansHost.empty());
    DeviceAllocation copySpans = makeCopySpanTable(copySpansHost, stream);

    Tensor left = makeGpuTensor<float>({leftCapacity}, std::vector<float>(leftCapacity, 11.0F), stream);
    Tensor right = makeGpuTensor<float>({rightCapacity}, std::vector<float>(rightCapacity, 22.0F), stream);
    Tensor output = makeGpuTensor<float>({leftCapacity + rightCapacity},
                                         std::vector<float>(leftCapacity + rightCapacity, sentinel),
                                         stream);
    Tensor outputOffsets = makeGpuTensor<OffsetT>({batchSize + 1}, std::vector<OffsetT>(batchSize + 1, 77), stream);
    DeviceAllocation valueTable = makePointerTable({left.getMemPtr(), right.getMemPtr()}, stream);

    rowPartitionUploadHostOffsets({0, 0, 0, 0}, outputOffsets, batchSize, stream);
    launchRaggedSequenceConcatenate(output.getMemPtr(),
                                    reinterpret_cast<void**>(valueTable.get()),
                                    copySpans.get(),
                                    copySpansHost.size(),
                                    sizeof(float),
                                    1,
                                    sizeof(OffsetT),
                                    0,
                                    stream);
    stream.synchronize();
    EXPECT_EQ(copyGpuTensor<OffsetT>(outputOffsets, stream), (std::vector<OffsetT>{0, 0, 0, 0}));
    for (float value : copyGpuTensor<float>(output, stream)) EXPECT_EQ(value, sentinel);

    Tensor leftGradient = makeGpuTensor<float>({leftCapacity}, std::vector<float>(leftCapacity, sentinel), stream);
    Tensor rightGradient = makeGpuTensor<float>({rightCapacity}, std::vector<float>(rightCapacity, sentinel), stream);
    DeviceAllocation gradientTable = makePointerTable({leftGradient.getMemPtr(), rightGradient.getMemPtr()}, stream);
    launchRaggedSequenceConcatenateBackward(reinterpret_cast<void**>(gradientTable.get()),
                                            output.getMemPtr(),
                                            copySpans.get(),
                                            copySpansHost.size(),
                                            sizeof(float),
                                            1,
                                            sizeof(OffsetT),
                                            0,
                                            stream);
    stream.synchronize();
    for (float value : copyGpuTensor<float>(leftGradient, stream)) EXPECT_EQ(value, sentinel);
    for (float value : copyGpuTensor<float>(rightGradient, stream)) EXPECT_EQ(value, sentinel);
}

template <typename OffsetT>
void runExactSpanGroupingBoundaryCase(uint32_t spanCount) {
    ASSERT_GE(spanCount, 2U);
    constexpr uint64_t batchSize = 1;
    constexpr uint8_t gradientSentinel = 0xA7;

    Stream stream(0);
    std::vector<std::vector<OffsetT>> offsetsHost(spanCount, std::vector<OffsetT>{0, 1});
    const auto copySpansHost = makeCopySpans(offsetsHost, batchSize);
    ASSERT_EQ(copySpansHost.size(), spanCount);
    DeviceAllocation copySpans = makeCopySpanTable(copySpansHost, stream);

    std::vector<uint8_t> valuesHost(spanCount);
    std::vector<uint8_t> upstreamHost(spanCount);
    for (uint32_t input = 0; input < spanCount; ++input) {
        valuesHost[input] = static_cast<uint8_t>((input * 41U + 7U) % 251U);
        upstreamHost[input] = static_cast<uint8_t>((input * 23U + 11U) % 251U);
    }

    Tensor values = makeGpuTensor<uint8_t>({spanCount}, valuesHost, stream);
    Tensor output = makeGpuTensor<uint8_t>({spanCount}, std::vector<uint8_t>(spanCount, 0xD3), stream);

    std::vector<void *> valuePointers(spanCount);
    uint8_t *valuesBase = values.getMemPtr<uint8_t>();
    for (uint32_t input = 0; input < spanCount; ++input) valuePointers[input] = valuesBase + input;
    DeviceAllocation valueTable = makePointerTable(valuePointers, stream);

    launchRaggedSequenceConcatenate(output.getMemPtr(),
                                    reinterpret_cast<void **>(valueTable.get()),
                                    copySpans.get(),
                                    copySpansHost.size(),
                                    sizeof(uint8_t),
                                    1,
                                    sizeof(OffsetT),
                                    spanCount,
                                    stream);
    stream.synchronize();
    EXPECT_EQ(copyGpuTensor<uint8_t>(output, stream), valuesHost);

    Tensor upstream = makeGpuTensor<uint8_t>({spanCount}, upstreamHost, stream);
    Tensor gradients = makeGpuTensor<uint8_t>(
        {spanCount}, std::vector<uint8_t>(spanCount, gradientSentinel), stream);
    std::vector<void *> gradientPointers(spanCount);
    uint8_t *gradientsBase = gradients.getMemPtr<uint8_t>();
    for (uint32_t input = 0; input < spanCount; ++input) gradientPointers[input] = gradientsBase + input;
    DeviceAllocation gradientTable = makePointerTable(gradientPointers, stream);

    launchRaggedSequenceConcatenateBackward(reinterpret_cast<void **>(gradientTable.get()),
                                            upstream.getMemPtr(),
                                            copySpans.get(),
                                            copySpansHost.size(),
                                            sizeof(uint8_t),
                                            1,
                                            sizeof(OffsetT),
                                            spanCount,
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
    std::vector<Tensor> values;
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

        values.emplace_back(makeGpuTensor<uint8_t>({capacities[input], widthBytes}, inputValues, stream));
    }

    std::vector<void *> valuePointers;
    valuePointers.reserve(numInputs);
    for (uint32_t input = 0; input < numInputs; ++input) valuePointers.push_back(values[input].getMemPtr());
    DeviceAllocation valueTable = makePointerTable(valuePointers, stream);
    const auto copySpansHost = makeCopySpans(offsetsHost, batchSize);
    uint64_t nonEmptySpans = 0;
    for (uint64_t row = 0; row < batchSize; ++row)
        for (uint32_t input = 0; input < numInputs; ++input)
            nonEmptySpans += offsetsHost[input][row] != offsetsHost[input][row + 1] ? 1ULL : 0ULL;
    ASSERT_EQ(copySpansHost.size(), nonEmptySpans);
    DeviceAllocation copySpans = makeCopySpanTable(copySpansHost, stream);
    uint64_t activeOutputValues = 0;
    for (uint64_t activeCount : activeCounts) activeOutputValues += activeCount;

    Tensor output = makeGpuTensor<uint8_t>(
        {outputCapacity, widthBytes}, std::vector<uint8_t>(outputCapacity * widthBytes, valueSentinel), stream);
    launchRaggedSequenceConcatenate(output.getMemPtr(),
                                    reinterpret_cast<void **>(valueTable.get()),
                                    copySpans.get(),
                                    copySpansHost.size(),
                                    sizeof(uint8_t),
                                    widthBytes,
                                    sizeof(OffsetT),
                                    activeOutputValues,
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
        // Exercise nullable gradient destinations while neighboring span slots in
        // the same CTA remain active.
        gradientPointers.push_back((input == 3 || input == 7) ? nullptr : gradients.back().getMemPtr());
    }
    DeviceAllocation gradientTable = makePointerTable(gradientPointers, stream);

    launchRaggedSequenceConcatenateBackward(reinterpret_cast<void **>(gradientTable.get()),
                                            upstreamGpu.getMemPtr(),
                                            copySpans.get(),
                                            copySpansHost.size(),
                                            sizeof(uint8_t),
                                            widthBytes,
                                            sizeof(OffsetT),
                                            activeOutputValues,
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


template <typename OffsetT>
void runUniformSpanPayloadCase(uint64_t expectedSpanBytes, uint32_t expectedSpansPerBlock) {
    ASSERT_GT(expectedSpanBytes, 0U);
    ASSERT_GE(expectedSpansPerBlock, 1U);
    ASSERT_LE(expectedSpansPerBlock, 256U);
    ASSERT_EQ(expectedSpansPerBlock & (expectedSpansPerBlock - 1U), 0U);

    constexpr uint32_t numInputs = 2;
    const uint64_t totalSpans = static_cast<uint64_t>(expectedSpansPerBlock) * 64ULL;
    ASSERT_EQ(totalSpans % numInputs, 0U);
    const uint64_t batchSize = totalSpans / numInputs;
    const uint64_t activePerInput = batchSize * expectedSpanBytes;
    const uint64_t activeOutputValues = activePerInput * numInputs;
    constexpr uint8_t outputSentinel = 0xD3;
    constexpr uint8_t gradientSentinel = 0xA7;

    Stream stream(0);
    std::vector<OffsetT> offsetsHost(batchSize + 1, 0);
    for (uint64_t row = 0; row < batchSize; ++row) {
        const uint64_t next = (row + 1) * expectedSpanBytes;
        ASSERT_LE(next, static_cast<uint64_t>(std::numeric_limits<OffsetT>::max()));
        offsetsHost[row + 1] = static_cast<OffsetT>(next);
    }
    const std::vector<std::vector<OffsetT>> allOffsetsHost{offsetsHost, offsetsHost};
    const auto copySpansHost = makeCopySpans(allOffsetsHost, batchSize);
    ASSERT_EQ(copySpansHost.size(), totalSpans);
    DeviceAllocation copySpans = makeCopySpanTable(copySpansHost, stream);

    std::vector<uint8_t> leftHost(activePerInput);
    std::vector<uint8_t> rightHost(activePerInput);
    for (uint64_t i = 0; i < activePerInput; ++i) {
        leftHost[i] = static_cast<uint8_t>((i * 17U + 3U) % 251U);
        rightHost[i] = static_cast<uint8_t>((i * 29U + 11U) % 251U);
    }
    Tensor left = makeGpuTensor<uint8_t>({activePerInput}, leftHost, stream);
    Tensor right = makeGpuTensor<uint8_t>({activePerInput}, rightHost, stream);
    Tensor output = makeGpuTensor<uint8_t>(
        {activeOutputValues + 1}, std::vector<uint8_t>(activeOutputValues + 1, outputSentinel), stream);

    DeviceAllocation valueTable = makePointerTable({left.getMemPtr(), right.getMemPtr()}, stream);
    launchRaggedSequenceConcatenate(output.getMemPtr(),
                                    reinterpret_cast<void **>(valueTable.get()),
                                    copySpans.get(),
                                    copySpansHost.size(),
                                    sizeof(uint8_t),
                                    1,
                                    sizeof(OffsetT),
                                    activeOutputValues,
                                    stream);
    stream.synchronize();

    const std::vector<uint8_t> actual = copyGpuTensor<uint8_t>(output, stream);
    for (uint64_t row = 0; row < batchSize; ++row) {
        const uint64_t inputBase = row * expectedSpanBytes;
        const uint64_t outputBase = row * numInputs * expectedSpanBytes;
        for (uint64_t byte = 0; byte < expectedSpanBytes; ++byte) {
            EXPECT_EQ(actual[outputBase + byte], leftHost[inputBase + byte]);
            EXPECT_EQ(actual[outputBase + expectedSpanBytes + byte], rightHost[inputBase + byte]);
        }
    }
    EXPECT_EQ(actual[activeOutputValues], outputSentinel);

    std::vector<uint8_t> upstreamHost(activeOutputValues);
    for (uint64_t i = 0; i < activeOutputValues; ++i)
        upstreamHost[i] = static_cast<uint8_t>((i * 13U + 5U) % 251U);
    Tensor upstream = makeGpuTensor<uint8_t>({activeOutputValues}, upstreamHost, stream);
    Tensor leftGradient = makeGpuTensor<uint8_t>(
        {activePerInput + 1}, std::vector<uint8_t>(activePerInput + 1, gradientSentinel), stream);
    Tensor rightGradient = makeGpuTensor<uint8_t>(
        {activePerInput + 1}, std::vector<uint8_t>(activePerInput + 1, gradientSentinel), stream);
    DeviceAllocation gradientTable = makePointerTable({leftGradient.getMemPtr(), rightGradient.getMemPtr()}, stream);

    launchRaggedSequenceConcatenateBackward(reinterpret_cast<void **>(gradientTable.get()),
                                            upstream.getMemPtr(),
                                            copySpans.get(),
                                            copySpansHost.size(),
                                            sizeof(uint8_t),
                                            1,
                                            sizeof(OffsetT),
                                            activeOutputValues,
                                            stream);
    stream.synchronize();

    const std::vector<uint8_t> actualLeft = copyGpuTensor<uint8_t>(leftGradient, stream);
    const std::vector<uint8_t> actualRight = copyGpuTensor<uint8_t>(rightGradient, stream);
    for (uint64_t row = 0; row < batchSize; ++row) {
        const uint64_t inputBase = row * expectedSpanBytes;
        const uint64_t outputBase = row * numInputs * expectedSpanBytes;
        for (uint64_t byte = 0; byte < expectedSpanBytes; ++byte) {
            EXPECT_EQ(actualLeft[inputBase + byte], upstreamHost[outputBase + byte]);
            EXPECT_EQ(actualRight[inputBase + byte], upstreamHost[outputBase + expectedSpanBytes + byte]);
        }
    }
    EXPECT_EQ(actualLeft[activePerInput], gradientSentinel);
    EXPECT_EQ(actualRight[activePerInput], gradientSentinel);
}

template <typename OffsetT>
void runExactTailCase(uint32_t copyWidth, uint32_t tailBytes) {
    ASSERT_GT(copyWidth, 1U);
    ASSERT_GT(tailBytes, 0U);
    ASSERT_LT(tailBytes, copyWidth);

    uint64_t alignmentOffset = 0;
    switch (copyWidth) {
        case 32: alignmentOffset = 0; break;
        case 16: alignmentOffset = 16; break;
        case 8: alignmentOffset = 8; break;
        case 4: alignmentOffset = 4; break;
        case 2: alignmentOffset = 2; break;
        default: FAIL() << "unsupported copy width"; return;
    }

    constexpr uint64_t batchSize = 1;
    const uint64_t valueBytes = copyWidth + tailBytes;
    const uint64_t sourceLogicalBytes = alignmentOffset + valueBytes;
    const uint64_t destinationLogicalBytes = alignmentOffset + valueBytes + copyWidth;
    constexpr uint8_t sentinel = 0xD7;

    Stream stream(0);
    const std::vector<std::vector<OffsetT>> offsetsHost{{0, 1}, {0, 0}};
    const auto copySpansHost = makeCopySpans(offsetsHost, batchSize);
    ASSERT_EQ(copySpansHost.size(), 1U);
    DeviceAllocation copySpans = makeCopySpanTable(copySpansHost, stream);

    std::vector<uint8_t> sourceHost(sourceLogicalBytes);
    for (uint64_t i = 0; i < sourceLogicalBytes; ++i)
        sourceHost[i] = static_cast<uint8_t>((37U * i + 11U) & 0xFFU);
    Tensor source = makeGpuTensor<uint8_t>({sourceLogicalBytes}, sourceHost, stream);
    auto *sourceStart = source.getMemPtr<uint8_t>() + alignmentOffset;

    Tensor output = makeGpuTensor<uint8_t>(
        {destinationLogicalBytes}, std::vector<uint8_t>(destinationLogicalBytes, sentinel), stream);
    auto *outputStart = output.getMemPtr<uint8_t>() + alignmentOffset;
    DeviceAllocation valueTable = makePointerTable({sourceStart, sourceStart}, stream);

    launchRaggedSequenceConcatenate(outputStart,
                                    reinterpret_cast<void **>(valueTable.get()),
                                    copySpans.get(),
                                    copySpansHost.size(),
                                    sizeof(uint8_t),
                                    valueBytes,
                                    sizeof(OffsetT),
                                    1,
                                    stream);
    stream.synchronize();

    const std::vector<uint8_t> actual = copyGpuTensor<uint8_t>(output, stream);
    for (uint64_t byte = 0; byte < valueBytes; ++byte)
        EXPECT_EQ(actual[alignmentOffset + byte], sourceHost[alignmentOffset + byte]);
    for (uint64_t byte = valueBytes; byte < valueBytes + copyWidth; ++byte)
        EXPECT_EQ(actual[alignmentOffset + byte], sentinel) << "forward tail over-store byte=" << byte;

    // Backward uses an output-gradient tensor that ends exactly at the source
    // tail. The full-width final load therefore has to use Thor's physical
    // 128-byte tensor padding, while the destination must remain exact-sized.
    std::vector<uint8_t> upstreamHost(sourceLogicalBytes);
    for (uint64_t i = 0; i < sourceLogicalBytes; ++i)
        upstreamHost[i] = static_cast<uint8_t>((19U * i + 7U) & 0xFFU);
    Tensor upstream = makeGpuTensor<uint8_t>({sourceLogicalBytes}, upstreamHost, stream);
    auto *upstreamStart = upstream.getMemPtr<uint8_t>() + alignmentOffset;
    Tensor gradient = makeGpuTensor<uint8_t>(
        {destinationLogicalBytes}, std::vector<uint8_t>(destinationLogicalBytes, sentinel), stream);
    auto *gradientStart = gradient.getMemPtr<uint8_t>() + alignmentOffset;
    DeviceAllocation gradientTable = makePointerTable({gradientStart, nullptr}, stream);

    launchRaggedSequenceConcatenateBackward(reinterpret_cast<void **>(gradientTable.get()),
                                            upstreamStart,
                                            copySpans.get(),
                                            copySpansHost.size(),
                                            sizeof(uint8_t),
                                            valueBytes,
                                            sizeof(OffsetT),
                                            1,
                                            stream);
    stream.synchronize();

    const std::vector<uint8_t> actualGradient = copyGpuTensor<uint8_t>(gradient, stream);
    for (uint64_t byte = 0; byte < valueBytes; ++byte)
        EXPECT_EQ(actualGradient[alignmentOffset + byte], upstreamHost[alignmentOffset + byte]);
    for (uint64_t byte = valueBytes; byte < valueBytes + copyWidth; ++byte)
        EXPECT_EQ(actualGradient[alignmentOffset + byte], sentinel) << "backward tail over-store byte=" << byte;
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

TEST(RaggedSequenceConcatenate, ManyInputsOddByteWidthBuildsOnlyNonEmptySpansAndHandlesNullableGradients) {
    REQUIRE_CUDA_DEVICE();
    runManyInputsOddWidthCase<uint32_t>(5);
    runManyInputsOddWidthCase<uint64_t>(5);
}

TEST(RaggedSequenceConcatenate, SparseNonEmptySpanPlansRemainCorrectAcrossLaunchScales) {
    REQUIRE_CUDA_DEVICE();
    // Row/input pairs include many zero-length rows. The host plan omits them,
    // so these sizes exercise sparse span counts across increasingly large batches.
    for (uint64_t batchSize : {15ULL, 29ULL, 57ULL}) {
        SCOPED_TRACE(::testing::Message() << "UINT32 batchSize=" << batchSize);
        runManyInputsOddWidthCase<uint32_t>(batchSize);
    }
    for (uint64_t batchSize : {15ULL, 29ULL, 57ULL}) {
        SCOPED_TRACE(::testing::Message() << "UINT64 batchSize=" << batchSize);
        runManyInputsOddWidthCase<uint64_t>(batchSize);
    }
}

TEST(RaggedSequenceConcatenate, AdaptiveSpanGroupingThresholdBoundariesCoverBothOffsetWidths) {
    REQUIRE_CUDA_DEVICE();
    // One row with one active value per input makes numInputs == totalSpans while
    // respecting the public contract that concatenation has at least two inputs.
    // Contiguous value/gradient storage keeps the 127/511-input cases lightweight.
    // These hit both sides of every launch-policy boundary exactly:
    //   <128 -> 1 span/CTA, [128,256) -> 2, [256,512) -> 4, >=512 -> 8.
    constexpr uint32_t spanCounts[] = {127, 128, 255, 256, 511, 512};
    for (uint32_t spanCount : spanCounts) {
        SCOPED_TRACE(::testing::Message() << "UINT32 totalSpans=" << spanCount);
        runExactSpanGroupingBoundaryCase<uint32_t>(spanCount);
    }
    for (uint32_t spanCount : spanCounts) {
        SCOPED_TRACE(::testing::Message() << "UINT64 totalSpans=" << spanCount);
        runExactSpanGroupingBoundaryCase<uint64_t>(spanCount);
    }
}
TEST(RaggedSequenceConcatenate, PayloadAwareSpanGroupingCoversFullOneThrough256SpecializationLadder) {
    REQUIRE_CUDA_DEVICE();
    // One byte per span lets payload geometry permit the widest grouping. The
    // helper chooses exactly 64 CTAs worth of spans for each requested rung, so
    // the parallelism floor forces every 1/2/4/.../256 spans-per-CTA template.
    for (const uint32_t spansPerBlock : {1U, 2U, 4U, 8U, 16U, 32U, 64U, 128U, 256U}) {
        SCOPED_TRACE(::testing::Message() << "UINT32 spansPerBlock=" << spansPerBlock);
        runUniformSpanPayloadCase<uint32_t>(1, spansPerBlock);
    }
    for (const uint32_t spansPerBlock : {1U, 2U, 4U, 8U, 16U, 32U, 64U, 128U, 256U}) {
        SCOPED_TRACE(::testing::Message() << "UINT64 spansPerBlock=" << spansPerBlock);
        runUniformSpanPayloadCase<uint64_t>(1, spansPerBlock);
    }
}

TEST(RaggedSequenceConcatenate, PayloadThresholdTransitionsCoverEveryLaneAllocation) {
    REQUIRE_CUDA_DEVICE();
    struct LaunchCase {
        uint64_t expectedSpanBytes;
        uint32_t spansPerBlock;
    };
    // The first byte above every 32-bytes/lane boundary halves spans/CTA. Each
    // helper invocation supplies enough spans that the 64-CTA parallelism floor
    // permits exactly the payload-selected specialization.
    const std::vector<LaunchCase> cases{
        {32, 256}, {33, 128}, {65, 64}, {129, 32}, {257, 16},
        {513, 8}, {1025, 4}, {2049, 2}, {4097, 1},
    };
    for (const LaunchCase& launchCase : cases) {
        SCOPED_TRACE(::testing::Message() << "bytes=" << launchCase.expectedSpanBytes
                                          << " spansPerBlock=" << launchCase.spansPerBlock);
        runUniformSpanPayloadCase<uint32_t>(launchCase.expectedSpanBytes, launchCase.spansPerBlock);
    }
}

TEST(RaggedSequenceConcatenate, AlignedBulkCopiesUseExactTailForEverySupportedWidthAndTailSize) {
    REQUIRE_CUDA_DEVICE();
    for (const uint32_t copyWidth : {32U, 16U, 8U, 4U, 2U}) {
        for (uint32_t tailBytes = 1; tailBytes < copyWidth; ++tailBytes) {
            SCOPED_TRACE(::testing::Message() << "copyWidth=" << copyWidth << " tailBytes=" << tailBytes);
            runExactTailCase<uint32_t>(copyWidth, tailBytes);
        }
    }
    // Offset width is orthogonal to the byte-copy tail machinery, but retain a
    // wide-tail UINT64 case so both structural dtypes cross that path.
    runExactTailCase<uint64_t>(32, 31);
}
