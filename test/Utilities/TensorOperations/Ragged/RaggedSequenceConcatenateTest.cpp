#include "Utilities/TensorOperations/Ragged/RaggedSequenceConcatenate.h"

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

    launchRaggedSequenceConcatenate(output.getMemPtr(),
                                    outputOffsets.getMemPtr(),
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

    launchRaggedSequenceConcatenate(output.getMemPtr(),
                                    outputOffsets.getMemPtr(),
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

}  // namespace

TEST(RaggedSequenceConcatenate, ForwardBackwardUint32ProducePartitionAndIgnoreInactiveCapacity) {
    REQUIRE_CUDA_DEVICE();
    runForwardBackwardCase<uint32_t>(DataType::UINT32);
}

TEST(RaggedSequenceConcatenate, ForwardBackwardUint64ProducePartitionAndIgnoreInactiveCapacity) {
    REQUIRE_CUDA_DEVICE();
    runForwardBackwardCase<uint64_t>(DataType::UINT64);
}

TEST(RaggedSequenceConcatenate, AllEmptyRowsProduceZeroOffsetsWithoutTouchingValueCapacity) {
    REQUIRE_CUDA_DEVICE();
    runAllEmptyCase<uint32_t>();
    runAllEmptyCase<uint64_t>();
}
