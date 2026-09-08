#include "DeepLearning/Implementation/Data/Residency/DeviceResidentWindowMaterializationKernel.h"

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"
#include "gtest/gtest.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

using namespace ThorImplementation;

namespace {

#define REQUIRE_CUDA_DEVICE()                                                                                           \
    do {                                                                                                                 \
        int cudaDeviceCountForTest = 0;                                                                                  \
        const cudaError_t cudaStatusForTest = cudaGetDeviceCount(&cudaDeviceCountForTest);                               \
        if (cudaStatusForTest != cudaSuccess || cudaDeviceCountForTest <= 0) {                                            \
            GTEST_SKIP() << "CUDA device is required for device-resident window materialization tests.";                \
        }                                                                                                                \
    } while (false)

const TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
const TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

template <typename T>
DataType dtypeFor();
template <>
DataType dtypeFor<uint8_t>() { return DataType::UINT8; }
template <>
DataType dtypeFor<uint16_t>() { return DataType::UINT16; }

template <typename T>
Tensor makeGpuTensor(
    const std::vector<uint64_t> &dimensions,
    const std::vector<T> &values,
    Stream &stream) {
    Tensor host(cpuPlacement, TensorDescriptor(dtypeFor<T>(), dimensions));
    if (host.getTotalNumElements() != values.size()) {
        throw std::runtime_error(
            "DeviceResidentWindowMaterializationKernel test tensor value count mismatch.");
    }
    std::memcpy(host.getMemPtr(), values.data(), values.size() * sizeof(T));
    Tensor device(gpuPlacement, host.getDescriptor());
    device.copyFromAsync(host, stream);
    stream.synchronize();
    return device;
}

template <typename T>
std::vector<T> copyGpuTensor(const Tensor &device, Stream &stream) {
    Tensor host(cpuPlacement, TensorDescriptor(dtypeFor<T>(), device.getDimensions()));
    host.copyFromAsync(device, stream);
    stream.synchronize();
    const T *values = host.getMemPtr<T>();
    return std::vector<T>(values, values + host.getTotalNumElements());
}

Tensor makeGpuPlans(
    const std::vector<DeviceResidentWindowRowPlan32> &plans,
    Stream &stream) {
    const uint64_t bytes = plans.size() * sizeof(DeviceResidentWindowRowPlan32);
    Tensor host(cpuPlacement, TensorDescriptor(DataType::UINT8, {bytes}));
    std::memcpy(host.getMemPtr(), plans.data(), static_cast<size_t>(bytes));
    Tensor device(gpuPlacement, host.getDescriptor());
    device.copyFromAsync(host, stream);
    stream.synchronize();
    return device;
}

Tensor makeGpuPlansAsWords(
    const std::vector<DeviceResidentWindowRowPlan32> &plans,
    Stream &stream) {
    const uint64_t bytes = plans.size() * sizeof(DeviceResidentWindowRowPlan32);
    if (bytes % sizeof(uint64_t) != 0) {
        throw std::runtime_error(
            "DeviceResidentWindowMaterializationKernel plan-word tensor must be UINT64 aligned.");
    }
    Tensor host(cpuPlacement, TensorDescriptor(DataType::UINT64, {bytes / sizeof(uint64_t)}));
    std::memcpy(host.getMemPtr(), plans.data(), static_cast<size_t>(bytes));
    Tensor device(gpuPlacement, host.getDescriptor());
    device.copyFromAsync(host, stream);
    stream.synchronize();
    return device;
}

TEST(DeviceResidentWindowMaterializationKernelTest,
     ResolvedPlanCopiesContiguousMiddleAndFillsExactPadding) {
    REQUIRE_CUDA_DEVICE();

    constexpr uint64_t batchCapacity = 3;
    constexpr uint64_t windowLength = 5;
    constexpr uint64_t stepBytes = sizeof(uint16_t);
    constexpr uint16_t pad = 0x1234;
    std::vector<uint16_t> source{10, 11, 12, 13, 14, 15, 16, 17};
    std::vector<DeviceResidentWindowRowPlan32> plans{
        {.sourceOffsetBytes = 2 * stepBytes, .validStepBegin = 1, .validStepCount = 3},
        {.sourceOffsetBytes = 0, .validStepBegin = 0, .validStepCount = 0},
        {.sourceOffsetBytes = 5 * stepBytes, .validStepBegin = 0, .validStepCount = 2},
    };
    const std::vector<uint16_t> expected{
        pad, 12, 13, 14, pad,
        pad, pad, pad, pad, pad,
        15, 16, pad, pad, pad,
    };

    Stream stream(0);
    Tensor sourceDevice = makeGpuTensor<uint16_t>({source.size()}, source, stream);
    // File-backed sessions expose plan sections as UINT64 alias views into the
    // consolidated selection upload allocation.
    Tensor planDevice = makeGpuPlansAsWords(plans, stream);
    Tensor destination = makeGpuTensor<uint16_t>(
        {batchCapacity, windowLength},
        std::vector<uint16_t>(batchCapacity * windowLength, 0xffffU),
        stream);

    DeviceResidentWindowMaterializationSpec spec;
    spec.dataType = DataType::UINT16;
    spec.windowLength = windowLength;
    spec.sourceStepBytes = stepBytes;
    spec.padValue = static_cast<double>(pad);
    launchDeviceResidentWindowMaterializationKernel(
        sourceDevice, planDevice, batchCapacity, spec, destination, stream);
    stream.synchronize();

    EXPECT_EQ(copyGpuTensor<uint16_t>(destination, stream), expected);
}

TEST(DeviceResidentWindowMaterializationKernelTest,
     MaskUsesSameResolvedPlanAndLeavesInactiveCapacityUntouched) {
    REQUIRE_CUDA_DEVICE();

    constexpr uint64_t batchCapacity = 4;
    constexpr uint64_t logicalRows = 2;
    constexpr uint64_t windowLength = 7;
    constexpr uint8_t sentinel = 0xcdU;
    // Only live plan rows are transferred by the file-backed session; inactive
    // destination capacity is intentionally not represented in the plan table.
    std::vector<DeviceResidentWindowRowPlan32> plans{
        {.sourceOffsetBytes = 0, .validStepBegin = 2, .validStepCount = 3},
        {.sourceOffsetBytes = 0, .validStepBegin = 0, .validStepCount = 5},
    };
    std::vector<uint8_t> expected(batchCapacity * windowLength, sentinel);
    for (uint64_t step = 0; step < windowLength; ++step) {
        expected[step] = step >= 2 && step < 5 ? 1 : 0;
        expected[windowLength + step] = step < 5 ? 1 : 0;
    }

    Stream stream(0);
    Tensor sourceDevice = makeGpuTensor<uint8_t>({1}, {0}, stream);
    Tensor planDevice = makeGpuPlans(plans, stream);
    Tensor destination = makeGpuTensor<uint8_t>(
        {batchCapacity, windowLength},
        std::vector<uint8_t>(batchCapacity * windowLength, sentinel),
        stream);

    DeviceResidentWindowMaterializationSpec spec;
    spec.dataType = DataType::UINT8;
    spec.windowLength = windowLength;
    spec.sourceStepBytes = 1;
    spec.materializeMask = true;
    launchDeviceResidentWindowMaterializationKernel(
        sourceDevice, planDevice, logicalRows, spec, destination, stream);
    stream.synchronize();

    EXPECT_EQ(copyGpuTensor<uint8_t>(destination, stream), expected);
}

void runExactTailCase(uint32_t copyWidth, uint32_t tailBytes) {
    ASSERT_TRUE(copyWidth == 32 || copyWidth == 16 || copyWidth == 8 ||
                copyWidth == 4 || copyWidth == 2);
    ASSERT_GT(tailBytes, 0U);
    ASSERT_LT(tailBytes, copyWidth);

    constexpr uint64_t batchCapacity = 2;
    constexpr uint8_t sentinel = 0xcdU;
    const uint64_t sourceOffset = copyWidth == 32 ? 0 : copyWidth;
    const uint64_t validBytes = copyWidth + tailBytes;
    std::vector<uint8_t> source(sourceOffset + validBytes, 0x5aU);
    for (uint64_t byte = 0; byte < validBytes; ++byte) {
        source[sourceOffset + byte] =
            static_cast<uint8_t>((byte * 31 + copyWidth * 7 + tailBytes) & 0xffU);
    }
    std::vector<DeviceResidentWindowRowPlan32> plans{
        {.sourceOffsetBytes = sourceOffset,
         .validStepBegin = 0,
         .validStepCount = static_cast<uint32_t>(validBytes)},
    };
    std::vector<uint8_t> expected(batchCapacity * validBytes, sentinel);
    std::memcpy(expected.data(), source.data() + sourceOffset, static_cast<size_t>(validBytes));

    Stream stream(0);
    Tensor sourceDevice = makeGpuTensor<uint8_t>({source.size()}, source, stream);
    Tensor planDevice = makeGpuPlans(plans, stream);
    Tensor destination = makeGpuTensor<uint8_t>(
        {batchCapacity, validBytes},
        std::vector<uint8_t>(batchCapacity * validBytes, sentinel),
        stream);

    DeviceResidentWindowMaterializationSpec spec;
    spec.dataType = DataType::UINT8;
    spec.windowLength = validBytes;
    spec.sourceStepBytes = 1;
    launchDeviceResidentWindowMaterializationKernel(
        sourceDevice, planDevice, /*logicalRows=*/1, spec, destination, stream);
    stream.synchronize();

    const std::vector<uint8_t> actual = copyGpuTensor<uint8_t>(destination, stream);
    ASSERT_EQ(actual.size(), expected.size());
    for (uint64_t byte = 0; byte < expected.size(); ++byte) {
        EXPECT_EQ(actual[byte], expected[byte])
            << "copyWidth=" << copyWidth << " tailBytes=" << tailBytes
            << " byte=" << byte;
    }
}

TEST(DeviceResidentWindowMaterializationKernelTest,
     WideBulkPathsUseEveryExactTailAndNeverOverwriteFollowingRow) {
    REQUIRE_CUDA_DEVICE();
    for (const uint32_t copyWidth : {32U, 16U, 8U, 4U, 2U}) {
        for (uint32_t tailBytes = 1; tailBytes < copyWidth; ++tailBytes) {
            runExactTailCase(copyWidth, tailBytes);
        }
    }
}

void runGroupingCase(uint64_t rowBytes, uint64_t logicalRows) {
    constexpr uint8_t sentinel = 0xcdU;
    std::vector<uint8_t> source(rowBytes);
    for (uint64_t byte = 0; byte < rowBytes; ++byte) {
        source[byte] = static_cast<uint8_t>((byte * 19 + 3) & 0xffU);
    }
    std::vector<DeviceResidentWindowRowPlan32> plans(logicalRows);
    for (auto &plan : plans) {
        plan.sourceOffsetBytes = 0;
        plan.validStepBegin = 0;
        plan.validStepCount = static_cast<uint32_t>(rowBytes);
    }

    Stream stream(0);
    Tensor sourceDevice = makeGpuTensor<uint8_t>({source.size()}, source, stream);
    Tensor planDevice = makeGpuPlans(plans, stream);
    Tensor destination = makeGpuTensor<uint8_t>(
        {logicalRows, rowBytes},
        std::vector<uint8_t>(logicalRows * rowBytes, sentinel),
        stream);
    DeviceResidentWindowMaterializationSpec spec;
    spec.dataType = DataType::UINT8;
    spec.windowLength = rowBytes;
    spec.sourceStepBytes = 1;
    launchDeviceResidentWindowMaterializationKernel(
        sourceDevice, planDevice, logicalRows, spec, destination, stream);
    stream.synchronize();

    const std::vector<uint8_t> actual = copyGpuTensor<uint8_t>(destination, stream);
    for (uint64_t row = 0; row < logicalRows; ++row) {
        for (uint64_t byte = 0; byte < rowBytes; ++byte) {
            EXPECT_EQ(actual[row * rowBytes + byte], source[byte]);
        }
    }
}

TEST(DeviceResidentWindowMaterializationKernelTest, PayloadAwareGroupingCoversFullLaneLadder) {
    REQUIRE_CUDA_DEVICE();
    struct LaunchCase {
        uint64_t rowBytes;
        uint64_t logicalRows;
    };
    constexpr std::array<LaunchCase, 9> cases{{
        {4097, 64},
        {4096, 128},
        {2048, 256},
        {1024, 512},
        {512, 1024},
        {256, 2048},
        {128, 4096},
        {64, 8192},
        {32, 16384},
    }};
    for (const LaunchCase launchCase : cases) {
        runGroupingCase(launchCase.rowBytes, launchCase.logicalRows);
    }
}

TEST(DeviceResidentWindowMaterializationKernelTest, PayloadThresholdTransitionsAreCovered) {
    REQUIRE_CUDA_DEVICE();
    constexpr std::array<uint64_t, 8> rowBytes{{33, 65, 129, 257, 513, 1025, 2049, 4097}};
    constexpr std::array<uint64_t, 8> rows{{16384, 8192, 4096, 2048, 1024, 512, 256, 128}};
    for (size_t i = 0; i < rowBytes.size(); ++i) runGroupingCase(rowBytes[i], rows[i]);
}

}  // namespace
