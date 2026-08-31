#include "Utilities/TensorOperations/Ragged/RaggedSequenceSlice.h"

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/TensorOperations/Ragged/RowPartition.h"
#include "gtest/gtest.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace ThorImplementation;

namespace {

#define REQUIRE_CUDA_DEVICE()                                                                                           \
    do {                                                                                                                 \
        int cudaDeviceCountForTest = 0;                                                                                  \
        const cudaError_t cudaStatusForTest = cudaGetDeviceCount(&cudaDeviceCountForTest);                               \
        if (cudaStatusForTest != cudaSuccess || cudaDeviceCountForTest <= 0) {                                            \
            GTEST_SKIP() << "CUDA device is required for RaggedSequenceSlice kernel tests.";                            \
        }                                                                                                                \
    } while (false)

const TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
const TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

template <typename T>
DataType dtypeFor();
template <>
DataType dtypeFor<float>() { return DataType::FP32; }
template <>
DataType dtypeFor<uint32_t>() { return DataType::UINT32; }
template <>
DataType dtypeFor<uint64_t>() { return DataType::UINT64; }

template <typename T>
Tensor makeGpuTensor(const std::vector<uint64_t>& dimensions, const std::vector<T>& values, Stream& stream) {
    Tensor host(cpuPlacement, TensorDescriptor(dtypeFor<T>(), dimensions));
    if (host.getTotalNumElements() != values.size()) {
        throw std::runtime_error("RaggedSequenceSlice test tensor value count mismatch.");
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

template <typename OffsetT>
void runForwardBackwardCase(DataType expectedOffsetsDataType) {
    constexpr uint64_t batchSize = 4;
    constexpr uint64_t inputCapacity = 10;
    constexpr uint64_t outputCapacity = 8;
    constexpr uint64_t width = 2;
    constexpr uint64_t start = 1;
    constexpr uint64_t length = 2;
    constexpr uint64_t activeInputValues = 8;
    constexpr uint64_t activeOutputValues = 4;
    constexpr float outputSentinel = -7001.0F;
    constexpr float gradientSentinel = -8001.0F;
    const float poison = std::numeric_limits<float>::quiet_NaN();

    Stream stream(0);
    Tensor inputOffsets = makeGpuTensor<OffsetT>({batchSize + 1}, {0, 4, 5, 8, 8}, stream);
    ASSERT_EQ(inputOffsets.getDataType(), expectedOffsetsDataType);

    std::vector<float> inputValues(inputCapacity * width, poison);
    for (uint64_t token = 0; token < activeInputValues; ++token) {
        for (uint64_t d = 0; d < width; ++d) inputValues[token * width + d] = 1000.0F + 10.0F * token + d;
    }
    Tensor input = makeGpuTensor<float>({inputCapacity, width}, inputValues, stream);
    Tensor output = makeGpuTensor<float>(
        {outputCapacity, width}, std::vector<float>(outputCapacity * width, outputSentinel), stream);
    Tensor rowLengths(gpuPlacement, TensorDescriptor(expectedOffsetsDataType, {batchSize}));
    Tensor outputOffsets = makeGpuTensor<OffsetT>({batchSize + 1}, std::vector<OffsetT>(batchSize + 1, 99), stream);
    const RowPartitionLengthsToOffsetsPlan scanPlan =
        prepareRowPartitionLengthsToOffsets(rowLengths, outputOffsets, batchSize);
    Tensor scanTemp(gpuPlacement, TensorDescriptor(DataType::UINT8, {std::max<size_t>(scanPlan.temp_storage_bytes, 1)}));

    launchRaggedSequenceSliceRowLengths(inputOffsets, rowLengths, start, length, batchSize, stream);
    rowPartitionLengthsToOffsets(scanPlan, scanTemp, rowLengths, outputOffsets, stream);
    launchRaggedSequenceSliceValues(input, inputOffsets, outputOffsets, output, start, length, batchSize, stream);
    stream.synchronize();

    EXPECT_EQ(copyGpuTensor<OffsetT>(rowLengths, stream), (std::vector<OffsetT>{2, 0, 2, 0}));
    EXPECT_EQ(copyGpuTensor<OffsetT>(outputOffsets, stream), (std::vector<OffsetT>{0, 2, 2, 4, 4}));
    const std::vector<float> actual = copyGpuTensor<float>(output, stream);
    const std::vector<uint64_t> expectedInputTokens{1, 2, 6, 7};
    for (uint64_t outputToken = 0; outputToken < expectedInputTokens.size(); ++outputToken) {
        for (uint64_t d = 0; d < width; ++d) {
            EXPECT_EQ(actual[outputToken * width + d], inputValues[expectedInputTokens[outputToken] * width + d]);
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
    Tensor inputGradient = makeGpuTensor<float>(
        {inputCapacity, width}, std::vector<float>(inputCapacity * width, gradientSentinel), stream);

    launchRaggedSequenceSliceBackward(
        inputOffsets, outputOffsets, upstreamGpu, inputGradient, start, length, batchSize, stream);
    stream.synchronize();

    const std::vector<float> actualGradient = copyGpuTensor<float>(inputGradient, stream);
    const std::vector<int64_t> outputTokenForInput{-1, 0, 1, -1, -1, -1, 2, 3};
    for (uint64_t token = 0; token < activeInputValues; ++token) {
        for (uint64_t d = 0; d < width; ++d) {
            const int64_t outputToken = outputTokenForInput[token];
            const float expected = outputToken < 0 ? 0.0F : upstream[static_cast<uint64_t>(outputToken) * width + d];
            EXPECT_EQ(actualGradient[token * width + d], expected) << "active input token " << token;
        }
    }
    for (uint64_t scalar = activeInputValues * width; scalar < actualGradient.size(); ++scalar) {
        EXPECT_EQ(actualGradient[scalar], gradientSentinel) << "inactive input-gradient scalar " << scalar;
    }
}

template <typename OffsetT>
void runAllRowsClippedCase() {
    constexpr uint64_t batchSize = 3;
    constexpr uint64_t inputCapacity = 5;
    constexpr uint64_t outputCapacity = 1;
    constexpr uint64_t start = 2;
    constexpr uint64_t length = 4;
    constexpr float sentinel = -9191.0F;

    Stream stream(0);
    Tensor inputOffsets = makeGpuTensor<OffsetT>({batchSize + 1}, {0, 1, 2, 2}, stream);
    Tensor input = makeGpuTensor<float>({inputCapacity}, {10.0F, 20.0F, 30.0F, 40.0F, 50.0F}, stream);
    Tensor output = makeGpuTensor<float>({outputCapacity}, {sentinel}, stream);
    Tensor rowLengths(gpuPlacement, TensorDescriptor(dtypeFor<OffsetT>(), {batchSize}));
    Tensor outputOffsets = makeGpuTensor<OffsetT>({batchSize + 1}, std::vector<OffsetT>(batchSize + 1, 77), stream);
    const RowPartitionLengthsToOffsetsPlan scanPlan =
        prepareRowPartitionLengthsToOffsets(rowLengths, outputOffsets, batchSize);
    Tensor scanTemp(gpuPlacement, TensorDescriptor(DataType::UINT8, {std::max<size_t>(scanPlan.temp_storage_bytes, 1)}));

    launchRaggedSequenceSliceRowLengths(inputOffsets, rowLengths, start, length, batchSize, stream);
    rowPartitionLengthsToOffsets(scanPlan, scanTemp, rowLengths, outputOffsets, stream);
    launchRaggedSequenceSliceValues(input, inputOffsets, outputOffsets, output, start, length, batchSize, stream);
    stream.synchronize();

    EXPECT_EQ(copyGpuTensor<OffsetT>(outputOffsets, stream), (std::vector<OffsetT>{0, 0, 0, 0}));
    EXPECT_EQ(copyGpuTensor<float>(output, stream), (std::vector<float>{sentinel}));

    Tensor upstream = makeGpuTensor<float>({outputCapacity}, {1234.0F}, stream);
    Tensor inputGradient = makeGpuTensor<float>(
        {inputCapacity}, std::vector<float>(inputCapacity, sentinel), stream);
    launchRaggedSequenceSliceBackward(
        inputOffsets, outputOffsets, upstream, inputGradient, start, length, batchSize, stream);
    stream.synchronize();
    const std::vector<float> gradient = copyGpuTensor<float>(inputGradient, stream);
    EXPECT_EQ(gradient[0], 0.0F);
    EXPECT_EQ(gradient[1], 0.0F);
    for (uint64_t token = 2; token < inputCapacity; ++token) EXPECT_EQ(gradient[token], sentinel);
}

}  // namespace

TEST(RaggedSequenceSlice, ForwardBackwardUint32ClipRowsAndIgnoreInactiveCapacity) {
    REQUIRE_CUDA_DEVICE();
    runForwardBackwardCase<uint32_t>(DataType::UINT32);
}

TEST(RaggedSequenceSlice, ForwardBackwardUint64ClipRowsAndIgnoreInactiveCapacity) {
    REQUIRE_CUDA_DEVICE();
    runForwardBackwardCase<uint64_t>(DataType::UINT64);
}

TEST(RaggedSequenceSlice, AllRowsClippedProduceZeroOffsetsAndZeroOnlyActiveInputGradients) {
    REQUIRE_CUDA_DEVICE();
    runAllRowsClippedCase<uint32_t>();
    runAllRowsClippedCase<uint64_t>();
}
