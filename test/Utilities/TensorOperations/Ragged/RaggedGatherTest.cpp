#include "Utilities/TensorOperations/Ragged/RaggedGather.h"

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"
#include "gtest/gtest.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
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
DataType dtypeFor<__half>() { return DataType::FP16; }
template <>
DataType dtypeFor<__nv_bfloat16>() { return DataType::BF16; }
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


template <typename SourceOffsetT, typename IndexOffsetT, typename IndexT>
void runWideVectorCase() {
    constexpr uint64_t batchSize = 3;
    constexpr uint64_t sourceCapacity = 10;
    constexpr uint64_t indexCapacity = 16;
    constexpr uint64_t width = 128;
    constexpr float outputSentinel = -9101.0F;
    constexpr float gradientSentinel = -9201.0F;

    Stream stream(0);
    Tensor sourceOffsets = makeGpuTensor<SourceOffsetT>({batchSize + 1}, {0, 3, 7, 9}, stream);
    std::vector<float> sourceValues(sourceCapacity * width, -9999.0F);
    for (uint64_t token = 0; token < 9; ++token) {
        for (uint64_t feature = 0; feature < width; ++feature) {
            sourceValues[token * width + feature] = static_cast<float>(10000 + token * 256 + feature);
        }
    }
    Tensor source = makeGpuTensor<float>({sourceCapacity, width}, sourceValues, stream);

    Tensor indexOffsets = makeGpuTensor<IndexOffsetT>({batchSize + 1}, {0, 5, 11, 15}, stream);
    Tensor indices = makeGpuTensor<IndexT>(
        {indexCapacity},
        {2, 0, 2, 1, 2, 3, 0, 2, 3, 1, 3, 1, 0, 1, 1, static_cast<IndexT>(999)},
        stream);
    const std::vector<uint64_t> expectedSourceTokens{2, 0, 2, 1, 2, 6, 3, 5, 6, 4, 6, 8, 7, 8, 8};

    Tensor output = makeGpuTensor<float>(
        {indexCapacity, width}, std::vector<float>(indexCapacity * width, outputSentinel), stream);
    launchRaggedGather(source, sourceOffsets, indices, indexOffsets, output, batchSize, stream);
    stream.synchronize();

    const std::vector<float> actual = copyGpuTensor<float>(output, stream);
    for (uint64_t outputToken = 0; outputToken < expectedSourceTokens.size(); ++outputToken) {
        for (uint64_t feature = 0; feature < width; ++feature) {
            EXPECT_EQ(actual[outputToken * width + feature],
                      sourceValues[expectedSourceTokens[outputToken] * width + feature]);
        }
    }
    for (uint64_t feature = 0; feature < width; ++feature) {
        EXPECT_EQ(actual[15 * width + feature], outputSentinel);
    }

    std::vector<float> upstream(indexCapacity * width, 7777.0F);
    for (uint64_t outputToken = 0; outputToken < expectedSourceTokens.size(); ++outputToken) {
        for (uint64_t feature = 0; feature < width; ++feature) {
            upstream[outputToken * width + feature] = static_cast<float>(1 + outputToken % 5 + feature % 3);
        }
    }
    Tensor upstreamGpu = makeGpuTensor<float>({indexCapacity, width}, upstream, stream);
    Tensor sourceGradient = makeGpuTensor<float>(
        {sourceCapacity, width}, std::vector<float>(sourceCapacity * width, gradientSentinel), stream);
    launchRaggedGatherBackward(sourceOffsets, indices, indexOffsets, upstreamGpu, sourceGradient, batchSize, stream);
    stream.synchronize();

    const std::vector<float> gradient = copyGpuTensor<float>(sourceGradient, stream);
    for (uint64_t token = 0; token < 9; ++token) {
        for (uint64_t feature = 0; feature < width; ++feature) {
            float expected = 0.0F;
            for (uint64_t outputToken = 0; outputToken < expectedSourceTokens.size(); ++outputToken) {
                if (expectedSourceTokens[outputToken] == token) expected += upstream[outputToken * width + feature];
            }
            EXPECT_EQ(gradient[token * width + feature], expected)
                << "source token " << token << " feature " << feature;
        }
    }
    for (uint64_t feature = 0; feature < width; ++feature) {
        EXPECT_EQ(gradient[9 * width + feature], gradientSentinel);
    }
}


void runBackwardIntermediateLaneGroupCase(uint64_t width) {
    constexpr uint64_t batchSize = 1;
    constexpr uint64_t sourceCapacity = 4;
    constexpr uint64_t indexCapacity = 5;
    constexpr float gradientSentinel = -9301.0F;
    Stream stream(0);

    Tensor sourceOffsets = makeGpuTensor<uint32_t>({2}, {0, 3}, stream);
    Tensor indexOffsets = makeGpuTensor<uint32_t>({2}, {0, 4}, stream);
    Tensor indices = makeGpuTensor<uint32_t>({indexCapacity}, {1, 1, 2, 0, 99}, stream);

    std::vector<float> upstream(indexCapacity * width, 7777.0F);
    for (uint64_t outputToken = 0; outputToken < 4; ++outputToken) {
        for (uint64_t feature = 0; feature < width; ++feature) {
            upstream[outputToken * width + feature] = static_cast<float>(1 + outputToken + feature);
        }
    }
    Tensor upstreamGpu = makeGpuTensor<float>({indexCapacity, width}, upstream, stream);
    Tensor sourceGradient = makeGpuTensor<float>(
        {sourceCapacity, width}, std::vector<float>(sourceCapacity * width, gradientSentinel), stream);

    launchRaggedGatherBackward(sourceOffsets, indices, indexOffsets, upstreamGpu, sourceGradient, batchSize, stream);
    stream.synchronize();
    const std::vector<float> actual = copyGpuTensor<float>(sourceGradient, stream);

    // Output tokens 0/1 both target source token 1, so this also verifies
    // duplicate scatter-add while the trailing feature group is parallelized.
    for (uint64_t feature = 0; feature < width; ++feature) {
        EXPECT_EQ(actual[0 * width + feature], upstream[3 * width + feature]) << "width " << width;
        EXPECT_EQ(actual[1 * width + feature],
                  upstream[0 * width + feature] + upstream[1 * width + feature])
            << "width " << width;
        EXPECT_EQ(actual[2 * width + feature], upstream[2 * width + feature]) << "width " << width;
        EXPECT_EQ(actual[3 * width + feature], gradientSentinel) << "width " << width;
    }
}

template <typename T>
T lowPrecisionValue(float value);

template <>
__half lowPrecisionValue(float value) {
    return __float2half_rn(value);
}

template <>
__nv_bfloat16 lowPrecisionValue(float value) {
    return __float2bfloat16_rn(value);
}

template <typename T>
float lowPrecisionToFloat(T value);

template <>
float lowPrecisionToFloat(__half value) {
    return __half2float(value);
}

template <>
float lowPrecisionToFloat(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename ValueT>
void runLowPrecisionDuplicateBackwardCase() {
    constexpr uint64_t batchSize = 2;
    constexpr uint64_t sourceCapacity = 5;
    constexpr uint64_t indexCapacity = 5;
    constexpr uint64_t width = 3;
    Stream stream(0);

    Tensor sourceOffsets = makeGpuTensor<uint32_t>({3}, {0, 2, 4}, stream);
    Tensor indexOffsets = makeGpuTensor<uint32_t>({3}, {0, 3, 5}, stream);
    Tensor indices = makeGpuTensor<uint32_t>({5}, {1, 1, 0, 0, 0}, stream);

    std::vector<ValueT> upstream(indexCapacity * width);
    for (uint64_t outputToken = 0; outputToken < indexCapacity; ++outputToken) {
        for (uint64_t feature = 0; feature < width; ++feature) {
            upstream[outputToken * width + feature] =
                lowPrecisionValue<ValueT>(static_cast<float>(outputToken + feature + 1));
        }
    }
    Tensor upstreamGpu = makeGpuTensor<ValueT>({indexCapacity, width}, upstream, stream);
    Tensor sourceGradient = makeGpuTensor<ValueT>(
        {sourceCapacity, width}, std::vector<ValueT>(sourceCapacity * width, lowPrecisionValue<ValueT>(7.0F)), stream);

    launchRaggedGatherBackward(sourceOffsets, indices, indexOffsets, upstreamGpu, sourceGradient, batchSize, stream);
    stream.synchronize();
    const std::vector<ValueT> actual = copyGpuTensor<ValueT>(sourceGradient, stream);

    const std::vector<std::vector<uint64_t>> gatheredByToken{{2}, {0, 1}, {3, 4}, {}};
    for (uint64_t sourceToken = 0; sourceToken < 4; ++sourceToken) {
        for (uint64_t feature = 0; feature < width; ++feature) {
            float expected = 0.0F;
            for (uint64_t outputToken : gatheredByToken[sourceToken]) {
                expected += static_cast<float>(outputToken + feature + 1);
            }
            EXPECT_EQ(lowPrecisionToFloat(actual[sourceToken * width + feature]), expected)
                << "source token " << sourceToken << " feature " << feature;
        }
    }
    for (uint64_t feature = 0; feature < width; ++feature) {
        EXPECT_EQ(lowPrecisionToFloat(actual[4 * width + feature]), 7.0F);
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

TEST(RaggedGather, InvalidRowLocalIndexIsGuardedAndCannotCrossRowsForwardOrBackward) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor sourceOffsets = makeGpuTensor<uint32_t>({3}, {0, 1, 3}, stream);
    Tensor source = makeGpuTensor<float>({4}, {10, 20, 30, 9999}, stream);
    Tensor indexOffsets = makeGpuTensor<uint32_t>({3}, {0, 1, 2}, stream);
    // Row 0 has length 1, so local index 1 would incorrectly address row 1's
    // first token if either kernel failed to enforce row locality.
    Tensor indices = makeGpuTensor<uint32_t>({2}, {1, 1}, stream);
    Tensor output = makeGpuTensor<float>({2}, {-1, -1}, stream);

    launchRaggedGather(source, sourceOffsets, indices, indexOffsets, output, 2, stream);
    stream.synchronize();
    EXPECT_EQ(copyGpuTensor<float>(output, stream), (std::vector<float>{0, 30}));

    Tensor upstream = makeGpuTensor<float>({2}, {5, 7}, stream);
    Tensor sourceGradient = makeGpuTensor<float>({4}, {-8, -8, -8, -8}, stream);
    launchRaggedGatherBackward(sourceOffsets, indices, indexOffsets, upstream, sourceGradient, 2, stream);
    stream.synchronize();
    EXPECT_EQ(copyGpuTensor<float>(sourceGradient, stream), (std::vector<float>{0, 0, 7, -8}));
}


TEST(RaggedGather, WideTrailingVectorParallelizesForwardAndBackwardAcrossFeatures) {
    REQUIRE_CUDA_DEVICE();
    runWideVectorCase<uint32_t, uint32_t, uint32_t>();
    runWideVectorCase<uint64_t, uint64_t, uint64_t>();
}

TEST(RaggedGather, ForwardCopyWidthsAndLaneGroupsPreserveOddWidthsAndInactiveCapacity) {
    REQUIRE_CUDA_DEVICE();
    constexpr uint64_t batchSize = 2;
    Stream stream(0);

    // These byte widths select copy transactions of 1/2/4/8/16 bytes and,
    // correspondingly, token groups of 32/16/8/4/2 lanes. Existing scalar
    // and width-2 FP32 cases cover the remaining one-lane token group.
    for (uint64_t width : {17ULL, 18ULL, 20ULL, 24ULL, 32ULL}) {
        Tensor sourceOffsets = makeGpuTensor<uint32_t>({3}, {0, 2, 4}, stream);
        std::vector<uint8_t> sourceValues(5 * width, 0xEE);
        for (uint64_t token = 0; token < 4; ++token) {
            for (uint64_t feature = 0; feature < width; ++feature) {
                sourceValues[token * width + feature] = static_cast<uint8_t>(token * 48 + feature);
            }
        }
        Tensor source = makeGpuTensor<uint8_t>({5, width}, sourceValues, stream);
        Tensor indexOffsets = makeGpuTensor<uint32_t>({3}, {0, 2, 4}, stream);
        Tensor indices = makeGpuTensor<uint32_t>({5}, {1, 0, 2, 1, 99}, stream);
        Tensor output = makeGpuTensor<uint8_t>({5, width}, std::vector<uint8_t>(5 * width, 0xAB), stream);

        launchRaggedGather(source, sourceOffsets, indices, indexOffsets, output, batchSize, stream);
        stream.synchronize();
        const std::vector<uint8_t> actual = copyGpuTensor<uint8_t>(output, stream);
        for (uint64_t feature = 0; feature < width; ++feature) {
            EXPECT_EQ(actual[0 * width + feature], sourceValues[1 * width + feature]) << "width " << width;
            EXPECT_EQ(actual[1 * width + feature], sourceValues[0 * width + feature]) << "width " << width;
            EXPECT_EQ(actual[2 * width + feature], 0U) << "width " << width;  // row-local index 2 is invalid.
            EXPECT_EQ(actual[3 * width + feature], sourceValues[3 * width + feature]) << "width " << width;
            EXPECT_EQ(actual[4 * width + feature], 0xABU) << "width " << width;  // inactive output capacity
        }
    }
}


TEST(RaggedGather, BackwardIntermediateLaneGroupsPreserveDuplicateAccumulationAndInactiveCapacity) {
    REQUIRE_CUDA_DEVICE();
    // widths 5 and 9 select 8- and 16-lane token groups. Existing backward
    // cases cover 1, 2, 4, and 32 lanes.
    runBackwardIntermediateLaneGroupCase(5);
    runBackwardIntermediateLaneGroupCase(9);
}

TEST(RaggedGather, DuplicateBackwardSupportsFp16AndBf16WithParallelFeatureAccumulation) {
    REQUIRE_CUDA_DEVICE();
    runLowPrecisionDuplicateBackwardCase<__half>();
    runLowPrecisionDuplicateBackwardCase<__nv_bfloat16>();
}
