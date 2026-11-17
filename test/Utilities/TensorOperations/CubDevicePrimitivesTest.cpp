#include "Utilities/TensorOperations/Cub/CubDevicePrimitives.h"

#include "cuda_runtime.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <cstdint>
#include <vector>

using namespace ThorImplementation;

namespace {

#define REQUIRE_CUDA_DEVICE()                                                                                          \
    do {                                                                                                                \
        int cuda_device_count_for_test = 0;                                                                             \
        const cudaError_t cuda_status_for_test = cudaGetDeviceCount(&cuda_device_count_for_test);                       \
        if (cuda_status_for_test != cudaSuccess || cuda_device_count_for_test <= 0) {                                    \
            GTEST_SKIP() << "CUDA device is required for CUB primitive tests.";                                         \
        }                                                                                                               \
    } while (false)

TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

template <typename T>
DataType dtypeFor();

template <>
DataType dtypeFor<uint32_t>() {
    return DataType::UINT32;
}

template <>
DataType dtypeFor<uint64_t>() {
    return DataType::UINT64;
}

template <>
DataType dtypeFor<float>() {
    return DataType::FP32;
}

template <typename T>
Tensor makeGpuVector(const std::vector<T>& values, Stream& stream) {
    Tensor cpu(cpuPlacement, TensorDescriptor(dtypeFor<T>(), {static_cast<uint64_t>(values.size())}));
    auto* cpu_ptr = cpu.getMemPtr<T>();
    for (size_t i = 0; i < values.size(); ++i) {
        cpu_ptr[i] = values[i];
    }

    Tensor gpu(gpuPlacement, TensorDescriptor(dtypeFor<T>(), {static_cast<uint64_t>(values.size())}));
    gpu.copyFromAsync(cpu, stream);
    stream.synchronize();
    return gpu;
}

template <typename T>
std::vector<T> copyGpuVector(const Tensor& gpu, Stream& stream) {
    Tensor cpu = gpu.clone(cpuPlacement);
    cpu.copyFromAsync(gpu, stream);
    stream.synchronize();

    std::vector<T> values(cpu.getTotalNumElements());
    const auto* ptr = cpu.getMemPtr<T>();
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = ptr[i];
    }
    return values;
}

}  // namespace

TEST(CubDevicePrimitives, ScanDTypeSupportPolicyMatchesFeatureDefines) {
    EXPECT_TRUE(isCubScanDTypeSupported(DataType::UINT32));
    EXPECT_TRUE(isCubScanDTypeSupported(DataType::FP16));
    EXPECT_TRUE(isCubScanDTypeSupported(DataType::BF16));
    EXPECT_TRUE(isCubScanDTypeSupported(DataType::FP32));
    EXPECT_FALSE(isCubScanDTypeSupported(DataType::INT32));
    EXPECT_FALSE(isCubScanDTypeSupported(DataType::FP8_E4M3));
    EXPECT_FALSE(isCubScanDTypeSupported(DataType::FP8_E5M2));
#if THOR_CUB_ENABLE_64BIT_TYPES
    EXPECT_TRUE(isCubScanDTypeSupported(DataType::UINT64));
    EXPECT_TRUE(isCubScanDTypeSupported(DataType::FP64));
#else
    EXPECT_FALSE(isCubScanDTypeSupported(DataType::UINT64));
    EXPECT_FALSE(isCubScanDTypeSupported(DataType::FP64));
#endif
    EXPECT_FALSE(isCubScanDTypeSupported(DataType::INT64));

    EXPECT_TRUE(isCubSegmentOffsetDTypeSupported(DataType::UINT32));
#if THOR_CUB_ENABLE_64BIT_SEGMENT_OFFSETS
    EXPECT_TRUE(isCubSegmentOffsetDTypeSupported(DataType::UINT64));
#else
    EXPECT_FALSE(isCubSegmentOffsetDTypeSupported(DataType::UINT64));
#endif
    EXPECT_FALSE(isCubSegmentOffsetDTypeSupported(DataType::INT32));
    EXPECT_FALSE(isCubSegmentOffsetDTypeSupported(DataType::INT64));
    EXPECT_FALSE(isCubSegmentOffsetDTypeSupported(DataType::FP32));
}

TEST(CubDevicePrimitives, DeviceScanSupportsReverseDirection) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor input = makeGpuVector<uint32_t>({1U, 2U, 3U, 4U}, stream);
    Tensor inclusive_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {4}));
    Tensor exclusive_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {4}));

    const CubDeviceScanPlan inclusive_plan =
        prepareCubDeviceScan(input, inclusive_out, 4, CubScanOp::Sum, CubScanMode::Inclusive, CubScanDirection::Reverse);
    const CubDeviceScanPlan exclusive_plan =
        prepareCubDeviceScan(input, exclusive_out, 4, CubScanOp::Sum, CubScanMode::Exclusive, CubScanDirection::Reverse);
    Tensor temp = allocateCubTemporaryStorage(
        gpuPlacement, std::max({inclusive_plan.temp_storage_bytes, exclusive_plan.temp_storage_bytes}));

    cubDeviceScan(inclusive_plan, temp, input, inclusive_out, stream);
    cubDeviceScan(exclusive_plan, temp, input, exclusive_out, stream);
    stream.synchronize();

    EXPECT_EQ(copyGpuVector<uint32_t>(inclusive_out, stream), (std::vector<uint32_t>{10U, 9U, 7U, 4U}));
    EXPECT_EQ(copyGpuVector<uint32_t>(exclusive_out, stream), (std::vector<uint32_t>{9U, 7U, 4U, 0U}));
}

TEST(CubDevicePrimitives, SegmentedUniformScanSupportsReverseWithinSegmentDirection) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor input = makeGpuVector<uint32_t>({1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U}, stream);
    Tensor inclusive_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {8}));
    Tensor exclusive_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {8}));

    const CubDeviceSegmentedUniformScanPlan inclusive_plan = prepareCubDeviceSegmentedUniformScan(
        input, inclusive_out, 8, 2, 4, CubScanOp::Sum, CubScanMode::Inclusive, CubScanDirection::Reverse);
    const CubDeviceSegmentedUniformScanPlan exclusive_plan = prepareCubDeviceSegmentedUniformScan(
        input, exclusive_out, 8, 2, 4, CubScanOp::Sum, CubScanMode::Exclusive, CubScanDirection::Reverse);
    Tensor temp = allocateCubTemporaryStorage(
        gpuPlacement, std::max({inclusive_plan.temp_storage_bytes, exclusive_plan.temp_storage_bytes}));

    cubDeviceSegmentedUniformScan(inclusive_plan, temp, input, inclusive_out, stream);
    cubDeviceSegmentedUniformScan(exclusive_plan, temp, input, exclusive_out, stream);
    stream.synchronize();

    EXPECT_EQ(copyGpuVector<uint32_t>(inclusive_out, stream),
              (std::vector<uint32_t>{10U, 9U, 7U, 4U, 26U, 21U, 15U, 8U}));
    EXPECT_EQ(copyGpuVector<uint32_t>(exclusive_out, stream),
              (std::vector<uint32_t>{9U, 7U, 4U, 0U, 21U, 15U, 8U, 0U}));
}

TEST(CubDevicePrimitives, DeviceArgScanProducesFlattenedPrefixWinnerIndices) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor input = makeGpuVector<uint32_t>({3U, 1U, 4U, 1U}, stream);
    Tensor min_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {4}));
    Tensor max_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {4}));

    const CubDeviceArgScanPlan min_plan =
        prepareCubDeviceArgScan(input, min_out, 4, CubArgScanOp::ArgMin, CubScanMode::Inclusive);
    const CubDeviceArgScanPlan max_plan =
        prepareCubDeviceArgScan(input, max_out, 4, CubArgScanOp::ArgMax, CubScanMode::Inclusive);
    Tensor temp = allocateCubTemporaryStorage(
        gpuPlacement, std::max({min_plan.temp_storage_bytes, max_plan.temp_storage_bytes}));

    cubDeviceArgScan(min_plan, temp, input, min_out, stream);
    cubDeviceArgScan(max_plan, temp, input, max_out, stream);
    stream.synchronize();

    EXPECT_EQ(copyGpuVector<uint32_t>(min_out, stream), (std::vector<uint32_t>{0U, 1U, 1U, 1U}));
    EXPECT_EQ(copyGpuVector<uint32_t>(max_out, stream), (std::vector<uint32_t>{0U, 0U, 2U, 2U}));
}

TEST(CubDevicePrimitives, SegmentedUniformArgScanSupportsReverseWithinSegmentDirection) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor input = makeGpuVector<uint32_t>({3U, 1U, 4U, 1U, 2U, 5U, 0U, 0U}, stream);
    Tensor max_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {8}));

    const CubDeviceSegmentedUniformArgScanPlan max_plan = prepareCubDeviceSegmentedUniformArgScan(
        input, max_out, 8, 2, 4, CubArgScanOp::ArgMax, CubScanMode::Inclusive, CubScanDirection::Reverse);
    Tensor temp = allocateCubTemporaryStorage(gpuPlacement, max_plan.temp_storage_bytes);

    cubDeviceSegmentedUniformArgScan(max_plan, temp, input, max_out, stream);
    stream.synchronize();

    EXPECT_EQ(copyGpuVector<uint32_t>(max_out, stream), (std::vector<uint32_t>{2U, 2U, 2U, 3U, 5U, 5U, 7U, 7U}));
}

TEST(CubDevicePrimitives, SegmentedArgScanUsesContiguousOffsets) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor input = makeGpuVector<uint32_t>({8U, 6U, 7U, 5U, 3U, 2U, 9U}, stream);
    Tensor offsets = makeGpuVector<uint32_t>({0U, 2U, 5U, 7U}, stream);
    Tensor min_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {7}));
    Tensor max_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {7}));

    const CubDeviceSegmentedArgScanPlan min_plan =
        prepareCubDeviceSegmentedArgScan(input, min_out, offsets, 7, 3, CubArgScanOp::ArgMin, CubScanMode::Inclusive);
    const CubDeviceSegmentedArgScanPlan max_plan =
        prepareCubDeviceSegmentedArgScan(input, max_out, offsets, 7, 3, CubArgScanOp::ArgMax, CubScanMode::Inclusive);
    Tensor temp = allocateCubTemporaryStorage(
        gpuPlacement, std::max({min_plan.temp_storage_bytes, max_plan.temp_storage_bytes}));

    cubDeviceSegmentedArgScan(min_plan, temp, input, min_out, offsets, stream);
    cubDeviceSegmentedArgScan(max_plan, temp, input, max_out, offsets, stream);
    stream.synchronize();

    EXPECT_EQ(copyGpuVector<uint32_t>(min_out, stream), (std::vector<uint32_t>{0U, 1U, 2U, 3U, 4U, 5U, 5U}));
    EXPECT_EQ(copyGpuVector<uint32_t>(max_out, stream), (std::vector<uint32_t>{0U, 0U, 2U, 2U, 2U, 5U, 6U}));
}

TEST(CubDevicePrimitives, SegmentedArgScanMarksUnusedCapacityInvalidEvenWithDirtyWorkspace) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor input = makeGpuVector<uint32_t>({8U, 6U, 7U, 5U, 3U, 99U, 100U, 101U}, stream);
    Tensor offsets = makeGpuVector<uint32_t>({0U, 2U, 2U, 5U}, stream);
    Tensor forward_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {8}));
    Tensor reverse_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {8}));

    const CubDeviceSegmentedArgScanPlan forward_plan = prepareCubDeviceSegmentedArgScan(
        input, forward_out, offsets, 8, 3, CubArgScanOp::ArgMax, CubScanMode::Inclusive, CubScanDirection::Forward);
    const CubDeviceSegmentedArgScanPlan reverse_plan = prepareCubDeviceSegmentedArgScan(
        input, reverse_out, offsets, 8, 3, CubArgScanOp::ArgMax, CubScanMode::Inclusive, CubScanDirection::Reverse);
    Tensor temp = allocateCubTemporaryStorage(
        gpuPlacement, std::max({forward_plan.temp_storage_bytes, reverse_plan.temp_storage_bytes}));

    // Deliberately poison the reusable workspace with zeroes. Before the
    // inactive-pair initialization fix, unused pair indices were then read as
    // valid index 0 and could contaminate a later scatter-add.
    temp.memsetAsync(stream, 0);
    cubDeviceSegmentedArgScan(forward_plan, temp, input, forward_out, offsets, stream);
    temp.memsetAsync(stream, 0);
    cubDeviceSegmentedArgScan(reverse_plan, temp, input, reverse_out, offsets, stream);
    stream.synchronize();

    EXPECT_EQ(copyGpuVector<uint32_t>(forward_out, stream),
              (std::vector<uint32_t>{0U, 0U, 2U, 2U, 2U, UINT32_MAX, UINT32_MAX, UINT32_MAX}));
    EXPECT_EQ(copyGpuVector<uint32_t>(reverse_out, stream),
              (std::vector<uint32_t>{0U, 1U, 2U, 3U, 4U, UINT32_MAX, UINT32_MAX, UINT32_MAX}));
}

TEST(CubDevicePrimitives, SegmentedScanSupportsGenericOpsWithContiguousOffsets) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor input = makeGpuVector<uint32_t>({8U, 6U, 7U, 5U, 3U, 2U, 9U}, stream);
    Tensor offsets = makeGpuVector<uint32_t>({0U, 2U, 5U, 7U}, stream);
    Tensor min_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {7}));
    Tensor max_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {7}));
    Tensor product_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {7}));

    const CubDeviceSegmentedScanPlan min_plan =
        prepareCubDeviceSegmentedScan(input, min_out, offsets, 7, 3, CubScanOp::Min, CubScanMode::Inclusive);
    const CubDeviceSegmentedScanPlan max_plan =
        prepareCubDeviceSegmentedScan(input, max_out, offsets, 7, 3, CubScanOp::Max, CubScanMode::Inclusive);
    const CubDeviceSegmentedScanPlan product_plan =
        prepareCubDeviceSegmentedScan(input, product_out, offsets, 7, 3, CubScanOp::Product, CubScanMode::Exclusive);
    Tensor temp = allocateCubTemporaryStorage(
        gpuPlacement,
        std::max({min_plan.temp_storage_bytes,
                  max_plan.temp_storage_bytes,
                  product_plan.temp_storage_bytes}));

    cubDeviceSegmentedScan(min_plan, temp, input, min_out, offsets, stream);
    cubDeviceSegmentedScan(max_plan, temp, input, max_out, offsets, stream);
    cubDeviceSegmentedScan(product_plan, temp, input, product_out, offsets, stream);
    stream.synchronize();

    EXPECT_EQ(copyGpuVector<uint32_t>(min_out, stream), (std::vector<uint32_t>{8U, 6U, 7U, 5U, 3U, 2U, 2U}));
    EXPECT_EQ(copyGpuVector<uint32_t>(max_out, stream), (std::vector<uint32_t>{8U, 8U, 7U, 7U, 7U, 2U, 9U}));
    EXPECT_EQ(copyGpuVector<uint32_t>(product_out, stream), (std::vector<uint32_t>{1U, 8U, 1U, 7U, 35U, 1U, 2U}));
}

#if THOR_CUB_ENABLE_64BIT_SEGMENT_OFFSETS
TEST(CubDevicePrimitives, SegmentedScanSupportsUint64OffsetsIndependentOf64BitValueTypes) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor input = makeGpuVector<float>({1.0F, 2.0F, 3.0F, 4.0F, 5.0F}, stream);
    Tensor offsets = makeGpuVector<uint64_t>({0ULL, 2ULL, 2ULL, 5ULL}, stream);
    Tensor output(gpuPlacement, TensorDescriptor(DataType::FP32, {5}));

    const CubDeviceSegmentedScanPlan plan =
        prepareCubDeviceSegmentedScan(input, output, offsets, 5, 3, CubScanOp::Sum, CubScanMode::Inclusive);
    Tensor temp = allocateCubTemporaryStorage(gpuPlacement, plan.temp_storage_bytes);
    cubDeviceSegmentedScan(plan, temp, input, output, offsets, stream);
    stream.synchronize();

    EXPECT_EQ(copyGpuVector<float>(output, stream), (std::vector<float>{1.0F, 3.0F, 3.0F, 7.0F, 12.0F}));
}
#endif

TEST(CubDevicePrimitives, SegmentedScanSupportsRaggedReverseDirection) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor input = makeGpuVector<uint32_t>({1U, 2U, 3U, 4U, 5U, 6U, 7U}, stream);
    Tensor offsets = makeGpuVector<uint32_t>({0U, 3U, 3U, 7U}, stream);
    Tensor inclusive_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {7}));
    Tensor exclusive_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {7}));

    const CubDeviceSegmentedScanPlan inclusive_plan =
        prepareCubDeviceSegmentedScan(input, inclusive_out, offsets, 7, 3, CubScanOp::Sum, CubScanMode::Inclusive, CubScanDirection::Reverse);
    const CubDeviceSegmentedScanPlan exclusive_plan =
        prepareCubDeviceSegmentedScan(input, exclusive_out, offsets, 7, 3, CubScanOp::Sum, CubScanMode::Exclusive, CubScanDirection::Reverse);
    Tensor temp = allocateCubTemporaryStorage(
        gpuPlacement, std::max({inclusive_plan.temp_storage_bytes, exclusive_plan.temp_storage_bytes}));

    cubDeviceSegmentedScan(inclusive_plan, temp, input, inclusive_out, offsets, stream);
    cubDeviceSegmentedScan(exclusive_plan, temp, input, exclusive_out, offsets, stream);
    stream.synchronize();

    EXPECT_EQ(copyGpuVector<uint32_t>(inclusive_out, stream), (std::vector<uint32_t>{6U, 5U, 3U, 22U, 18U, 13U, 7U}));
    EXPECT_EQ(copyGpuVector<uint32_t>(exclusive_out, stream), (std::vector<uint32_t>{5U, 3U, 0U, 18U, 13U, 7U, 0U}));
}
