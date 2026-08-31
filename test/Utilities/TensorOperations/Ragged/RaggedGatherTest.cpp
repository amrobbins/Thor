#include "Utilities/TensorOperations/Ragged/RaggedGather.h"

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"
#include "gtest/gtest.h"

#include <cuda_runtime.h>

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
            GTEST_SKIP() << "CUDA device is required for RaggedGather kernel tests.";                                   \
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
        throw std::runtime_error("RaggedGather test tensor value count mismatch.");
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
    const T* values = host.getMemPtr<T>();
    return std::vector<T>(values, values + host.getTotalNumElements());
}

template <typename SourceOffsetT, typename IndexOffsetT, typename IndexT>
void runForwardBackwardCase() {
    constexpr uint64_t batchSize = 4;
    constexpr uint64_t sourceCapacity = 10;
    constexpr uint64_t indexCapacity = 9;
    constexpr uint64_t width = 2;
    constexpr float outputSentinel = -7001.0F;
    constexpr float gradientSentinel = -8001.0F;
    const float poison = std::numeric_limits<float>::quiet_NaN();

    Stream stream(0);
    // Source rows: [0,1,2], [3,4], [], [5,6,7,8]. Token 9 is inactive poison.
    Tensor sourceOffsets = makeGpuTensor<SourceOffsetT>({batchSize + 1}, {0, 3, 5, 5, 9}, stream);
    std::vector<float> sourceValues(sourceCapacity * width, poison);
    for (uint64_t token = 0; token < 9; ++token) {
        for (uint64_t d = 0; d < width; ++d) sourceValues[token * width + d] = 1000.0F + 10.0F * token + d;
    }
    Tensor source = makeGpuTensor<float>({sourceCapacity, width}, sourceValues, stream);

    // Q rows have lengths [3, 2, 0, 3]. Indices are row-local and include
    // duplicates in rows 0 and 3. Final capacity slot is inactive poison.
    Tensor indexOffsets = makeGpuTensor<IndexOffsetT>({batchSize + 1}, {0, 3, 5, 5, 8}, stream);
    Tensor indices = makeGpuTensor<IndexT>({indexCapacity}, {2, 0, 2, 1, 0, 3, 1, 3, static_cast<IndexT>(999)}, stream);
    Tensor output = makeGpuTensor<float>(
        {indexCapacity, width}, std::vector<float>(indexCapacity * width, outputSentinel), stream);

    launchRaggedGather(source, sourceOffsets, indices, indexOffsets, output, batchSize, stream);
    stream.synchronize();

    const std::vector<float> actual = copyGpuTensor<float>(output, stream);
    const std::vector<uint64_t> expectedSourceTokens{2, 0, 2, 4, 3, 8, 6, 8};
    for (uint64_t outToken = 0; outToken < expectedSourceTokens.size(); ++outToken) {
        for (uint64_t d = 0; d < width; ++d) {
            EXPECT_EQ(actual[outToken * width + d], sourceValues[expectedSourceTokens[outToken] * width + d]);
        }
    }
    for (uint64_t scalar = expectedSourceTokens.size() * width; scalar < actual.size(); ++scalar) {
        EXPECT_EQ(actual[scalar], outputSentinel) << "inactive output scalar " << scalar;
    }

    std::vector<float> upstream(indexCapacity * width, 9999.0F);
    for (uint64_t scalar = 0; scalar < expectedSourceTokens.size() * width; ++scalar) {
        upstream[scalar] = 3000.0F + static_cast<float>(scalar);
    }
    Tensor upstreamGpu = makeGpuTensor<float>({indexCapacity, width}, upstream, stream);
    Tensor sourceGradient = makeGpuTensor<float>(
        {sourceCapacity, width}, std::vector<float>(sourceCapacity * width, gradientSentinel), stream);

    launchRaggedGatherBackward(sourceOffsets, indices, indexOffsets, upstreamGpu, sourceGradient, batchSize, stream);
    stream.synchronize();

    const std::vector<float> gradient = copyGpuTensor<float>(sourceGradient, stream);
    for (uint64_t token = 0; token < 9; ++token) {
        for (uint64_t d = 0; d < width; ++d) {
            float expected = 0.0F;
            for (uint64_t outToken = 0; outToken < expectedSourceTokens.size(); ++outToken) {
                if (expectedSourceTokens[outToken] == token) expected += upstream[outToken * width + d];
            }
            EXPECT_EQ(gradient[token * width + d], expected) << "source token " << token << " component " << d;
        }
    }
    for (uint64_t scalar = 9 * width; scalar < gradient.size(); ++scalar) {
        EXPECT_EQ(gradient[scalar], gradientSentinel) << "inactive source-gradient scalar " << scalar;
    }
}

}  // namespace

TEST(RaggedGather, ForwardBackwardMixedPartitionsUint32) {
    REQUIRE_CUDA_DEVICE();
    runForwardBackwardCase<uint32_t, uint32_t, uint32_t>();
}

TEST(RaggedGather, ForwardBackwardMixedOffsetAndIndexDtypes) {
    REQUIRE_CUDA_DEVICE();
    runForwardBackwardCase<uint32_t, uint64_t, uint64_t>();
    runForwardBackwardCase<uint64_t, uint32_t, uint32_t>();
    runForwardBackwardCase<uint64_t, uint64_t, uint64_t>();
}

TEST(RaggedGather, EmptyIndexRowsAndAllEmptyQLeaveOutputCapacityUntouchedAndZeroActiveSourceGradient) {
    REQUIRE_CUDA_DEVICE();
    constexpr uint64_t batchSize = 3;
    Stream stream(0);
    Tensor sourceOffsets = makeGpuTensor<uint32_t>({4}, {0, 2, 5, 5}, stream);
    Tensor source = makeGpuTensor<float>({6}, {10, 20, 30, 40, 50, -999}, stream);
    Tensor indexOffsets = makeGpuTensor<uint64_t>({4}, {0, 0, 0, 0}, stream);
    Tensor indices = makeGpuTensor<uint32_t>({4}, {99, 99, 99, 99}, stream);
    Tensor output = makeGpuTensor<float>({4}, {-7, -7, -7, -7}, stream);

    launchRaggedGather(source, sourceOffsets, indices, indexOffsets, output, batchSize, stream);
    stream.synchronize();
    EXPECT_EQ(copyGpuTensor<float>(output, stream), (std::vector<float>{-7, -7, -7, -7}));

    Tensor upstream = makeGpuTensor<float>({4}, {1, 2, 3, 4}, stream);
    Tensor sourceGradient = makeGpuTensor<float>({6}, {-8, -8, -8, -8, -8, -8}, stream);
    launchRaggedGatherBackward(sourceOffsets, indices, indexOffsets, upstream, sourceGradient, batchSize, stream);
    stream.synchronize();
    EXPECT_EQ(copyGpuTensor<float>(sourceGradient, stream), (std::vector<float>{0, 0, 0, 0, 0, -8}));
}

TEST(RaggedGather, InvalidRowLocalIndexIsGuardedAndCannotReadAcrossRows) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor sourceOffsets = makeGpuTensor<uint32_t>({3}, {0, 1, 3}, stream);
    Tensor source = makeGpuTensor<float>({4}, {10, 20, 30, 9999}, stream);
    Tensor indexOffsets = makeGpuTensor<uint32_t>({3}, {0, 1, 2}, stream);
    // Row 0 has length 1, so local index 1 would incorrectly read row 1's first
    // token if the kernel failed to enforce row locality.
    Tensor indices = makeGpuTensor<uint32_t>({2}, {1, 1}, stream);
    Tensor output = makeGpuTensor<float>({2}, {-1, -1}, stream);

    launchRaggedGather(source, sourceOffsets, indices, indexOffsets, output, 2, stream);
    stream.synchronize();
    EXPECT_EQ(copyGpuTensor<float>(output, stream), (std::vector<float>{0, 30}));
}
