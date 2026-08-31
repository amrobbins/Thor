#include "Utilities/TensorOperations/Ragged/RaggedFilter.h"

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
            GTEST_SKIP() << "CUDA device is required for RaggedFilter kernel tests.";                                   \
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
        throw std::runtime_error("RaggedFilter test tensor value count mismatch.");
    }
    T* hostValues = host.getMemPtr<T>();
    for (size_t i = 0; i < values.size(); ++i) hostValues[i] = values[i];
    Tensor device(gpuPlacement, TensorDescriptor(dtypeFor<T>(), dimensions));
    device.copyFromAsync(host, stream);
    stream.synchronize();
    return device;
}

Tensor makeBooleanGpuTensor(const std::vector<uint8_t>& values, Stream& stream) {
    Tensor host(cpuPlacement, TensorDescriptor(DataType::BOOLEAN, {values.size()}));
    bool* hostValues = host.getMemPtr<bool>();
    for (size_t i = 0; i < values.size(); ++i) hostValues[i] = values[i] != 0;
    Tensor device(gpuPlacement, TensorDescriptor(DataType::BOOLEAN, {values.size()}));
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
    constexpr uint64_t capacity = 10;
    constexpr uint64_t width = 2;
    constexpr uint64_t activeInputValues = 8;
    constexpr uint64_t activeOutputValues = 5;
    constexpr float outputSentinel = -7001.0F;
    constexpr float gradientSentinel = -8001.0F;
    const float poison = std::numeric_limits<float>::quiet_NaN();

    Stream stream(0);
    Tensor inputOffsets = makeGpuTensor<OffsetT>({batchSize + 1}, {0, 4, 5, 8, 8}, stream);
    ASSERT_EQ(inputOffsets.getDataType(), expectedOffsetsDataType);

    std::vector<float> inputValues(capacity * width, poison);
    for (uint64_t token = 0; token < activeInputValues; ++token) {
        for (uint64_t d = 0; d < width; ++d) inputValues[token * width + d] = 1000.0F + 10.0F * token + d;
    }
    Tensor input = makeGpuTensor<float>({capacity, width}, inputValues, stream);

    // The inactive mask tail is deliberately true. Reading it would incorrectly
    // retain poisoned inactive values and violate the canonical active-prefix contract.
    Tensor mask = makeBooleanGpuTensor({1, 0, 1, 1, 0, 1, 0, 1, 1, 1}, stream);
    Tensor output = makeGpuTensor<float>(
        {capacity, width}, std::vector<float>(capacity * width, outputSentinel), stream);
    Tensor rowLengths(gpuPlacement, TensorDescriptor(expectedOffsetsDataType, {batchSize}));
    Tensor outputOffsets = makeGpuTensor<OffsetT>({batchSize + 1}, std::vector<OffsetT>(batchSize + 1, 99), stream);
    const RowPartitionLengthsToOffsetsPlan scanPlan =
        prepareRowPartitionLengthsToOffsets(rowLengths, outputOffsets, batchSize);
    Tensor scanTemp(gpuPlacement, TensorDescriptor(DataType::UINT8, {std::max<size_t>(scanPlan.temp_storage_bytes, 1)}));

    launchRaggedFilterRowLengths(mask, inputOffsets, rowLengths, batchSize, stream);
    rowPartitionLengthsToOffsets(scanPlan, scanTemp, rowLengths, outputOffsets, stream);
    launchRaggedFilterValues(input, mask, inputOffsets, outputOffsets, output, batchSize, stream);
    stream.synchronize();

    EXPECT_EQ(copyGpuTensor<OffsetT>(rowLengths, stream), (std::vector<OffsetT>{3, 0, 2, 0}));
    EXPECT_EQ(copyGpuTensor<OffsetT>(outputOffsets, stream), (std::vector<OffsetT>{0, 3, 3, 5, 5}));
    const std::vector<float> actual = copyGpuTensor<float>(output, stream);
    const std::vector<uint64_t> expectedInputTokens{0, 2, 3, 5, 7};
    for (uint64_t outputToken = 0; outputToken < expectedInputTokens.size(); ++outputToken) {
        for (uint64_t d = 0; d < width; ++d) {
            EXPECT_EQ(actual[outputToken * width + d], inputValues[expectedInputTokens[outputToken] * width + d]);
        }
    }
    for (uint64_t scalar = activeOutputValues * width; scalar < actual.size(); ++scalar) {
        EXPECT_EQ(actual[scalar], outputSentinel) << "inactive output scalar " << scalar;
    }

    std::vector<float> upstream(capacity * width, 9999.0F);
    for (uint64_t scalar = 0; scalar < activeOutputValues * width; ++scalar) {
        upstream[scalar] = 3000.0F + static_cast<float>(scalar);
    }
    Tensor upstreamGpu = makeGpuTensor<float>({capacity, width}, upstream, stream);
    Tensor inputGradient = makeGpuTensor<float>(
        {capacity, width}, std::vector<float>(capacity * width, gradientSentinel), stream);

    launchRaggedFilterBackward(mask, inputOffsets, outputOffsets, upstreamGpu, inputGradient, batchSize, stream);
    stream.synchronize();

    const std::vector<float> actualGradient = copyGpuTensor<float>(inputGradient, stream);
    const std::vector<int64_t> outputTokenForInput{0, -1, 1, 2, -1, 3, -1, 4};
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
void runAllFalseCase() {
    constexpr uint64_t batchSize = 3;
    constexpr uint64_t capacity = 6;
    constexpr float sentinel = -9191.0F;

    Stream stream(0);
    Tensor inputOffsets = makeGpuTensor<OffsetT>({batchSize + 1}, {0, 2, 5, 5}, stream);
    Tensor input = makeGpuTensor<float>({capacity}, {10, 20, 30, 40, 50, 60}, stream);
    Tensor mask = makeBooleanGpuTensor({0, 0, 0, 0, 0, 1}, stream);
    Tensor output = makeGpuTensor<float>({capacity}, std::vector<float>(capacity, sentinel), stream);
    Tensor rowLengths(gpuPlacement, TensorDescriptor(dtypeFor<OffsetT>(), {batchSize}));
    Tensor outputOffsets = makeGpuTensor<OffsetT>({batchSize + 1}, std::vector<OffsetT>(batchSize + 1, 77), stream);
    const RowPartitionLengthsToOffsetsPlan scanPlan =
        prepareRowPartitionLengthsToOffsets(rowLengths, outputOffsets, batchSize);
    Tensor scanTemp(gpuPlacement, TensorDescriptor(DataType::UINT8, {std::max<size_t>(scanPlan.temp_storage_bytes, 1)}));

    launchRaggedFilterRowLengths(mask, inputOffsets, rowLengths, batchSize, stream);
    rowPartitionLengthsToOffsets(scanPlan, scanTemp, rowLengths, outputOffsets, stream);
    launchRaggedFilterValues(input, mask, inputOffsets, outputOffsets, output, batchSize, stream);
    stream.synchronize();

    EXPECT_EQ(copyGpuTensor<OffsetT>(outputOffsets, stream), (std::vector<OffsetT>{0, 0, 0, 0}));
    EXPECT_EQ(copyGpuTensor<float>(output, stream), (std::vector<float>(capacity, sentinel)));

    Tensor upstream = makeGpuTensor<float>({capacity}, std::vector<float>(capacity, 1234.0F), stream);
    Tensor inputGradient = makeGpuTensor<float>({capacity}, std::vector<float>(capacity, sentinel), stream);
    launchRaggedFilterBackward(mask, inputOffsets, outputOffsets, upstream, inputGradient, batchSize, stream);
    stream.synchronize();
    const std::vector<float> gradient = copyGpuTensor<float>(inputGradient, stream);
    for (uint64_t token = 0; token < 5; ++token) EXPECT_EQ(gradient[token], 0.0F);
    EXPECT_EQ(gradient[5], sentinel);
}

}  // namespace

TEST(RaggedFilter, ForwardBackwardUint32StableCompactionIgnoresInactiveCapacity) {
    REQUIRE_CUDA_DEVICE();
    runForwardBackwardCase<uint32_t>(DataType::UINT32);
}

TEST(RaggedFilter, ForwardBackwardUint64StableCompactionIgnoresInactiveCapacity) {
    REQUIRE_CUDA_DEVICE();
    runForwardBackwardCase<uint64_t>(DataType::UINT64);
}

TEST(RaggedFilter, AllFalseMaskProducesZeroOffsetsAndZerosOnlyActiveInputGradient) {
    REQUIRE_CUDA_DEVICE();
    runAllFalseCase<uint32_t>();
    runAllFalseCase<uint64_t>();
}

TEST(RaggedFilter, StableCompactionPreservesOrderAcrossMultipleBlockTiles) {
    REQUIRE_CUDA_DEVICE();
    constexpr uint64_t batchSize = 1;
    constexpr uint64_t active = 520;
    constexpr uint64_t capacity = 540;
    Stream stream(0);

    Tensor inputOffsets = makeGpuTensor<uint32_t>({2}, {0, active}, stream);
    std::vector<float> values(capacity, -1.0F);
    std::vector<uint8_t> maskValues(capacity, 1);
    std::vector<float> expected;
    for (uint64_t token = 0; token < active; ++token) {
        values[token] = static_cast<float>(token);
        maskValues[token] = (token % 5 == 1 || token % 7 == 3) ? 1 : 0;
        if (maskValues[token]) expected.push_back(static_cast<float>(token));
    }
    Tensor input = makeGpuTensor<float>({capacity}, values, stream);
    Tensor mask = makeBooleanGpuTensor(maskValues, stream);
    Tensor output = makeGpuTensor<float>({capacity}, std::vector<float>(capacity, -777.0F), stream);
    Tensor lengths(gpuPlacement, TensorDescriptor(DataType::UINT32, {batchSize}));
    Tensor outputOffsets = makeGpuTensor<uint32_t>({2}, {99, 99}, stream);
    const RowPartitionLengthsToOffsetsPlan scanPlan =
        prepareRowPartitionLengthsToOffsets(lengths, outputOffsets, batchSize);
    Tensor scanTemp(gpuPlacement, TensorDescriptor(DataType::UINT8, {std::max<size_t>(scanPlan.temp_storage_bytes, 1)}));

    launchRaggedFilterRowLengths(mask, inputOffsets, lengths, batchSize, stream);
    rowPartitionLengthsToOffsets(scanPlan, scanTemp, lengths, outputOffsets, stream);
    launchRaggedFilterValues(input, mask, inputOffsets, outputOffsets, output, batchSize, stream);
    stream.synchronize();

    const std::vector<uint32_t> offsets = copyGpuTensor<uint32_t>(outputOffsets, stream);
    ASSERT_EQ(offsets[1], expected.size());
    const std::vector<float> actual = copyGpuTensor<float>(output, stream);
    for (size_t i = 0; i < expected.size(); ++i) EXPECT_EQ(actual[i], expected[i]) << "output token " << i;
    for (size_t i = expected.size(); i < actual.size(); ++i) EXPECT_EQ(actual[i], -777.0F);
}
