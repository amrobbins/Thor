#include "Utilities/TensorOperations/Cub/CubDevicePrimitives.h"

#include "cuda_runtime.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include "gtest/gtest.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <utility>
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
DataType dtypeFor<uint8_t>() {
    return DataType::UINT8;
}

template <>
DataType dtypeFor<int8_t>() {
    return DataType::INT8;
}

template <>
DataType dtypeFor<bool>() {
    return DataType::BOOLEAN;
}

template <>
DataType dtypeFor<uint32_t>() {
    return DataType::UINT32;
}

template <>
DataType dtypeFor<int32_t>() {
    return DataType::INT32;
}

template <>
DataType dtypeFor<uint64_t>() {
    return DataType::UINT64;
}

template <>
DataType dtypeFor<int64_t>() {
    return DataType::INT64;
}

template <>
DataType dtypeFor<double>() {
    return DataType::FP64;
}

template <>
DataType dtypeFor<__half>() {
    return DataType::FP16;
}

template <>
DataType dtypeFor<__nv_bfloat16>() {
    return DataType::BF16;
}

template <>
DataType dtypeFor<__nv_fp8_e4m3>() {
    return DataType::FP8_E4M3;
}

template <>
DataType dtypeFor<__nv_fp8_e5m2>() {
    return DataType::FP8_E5M2;
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

template <typename T>
std::vector<float> copyGpuVectorAsFloat(const Tensor& gpu, Stream& stream) {
    const std::vector<T> typed = copyGpuVector<T>(gpu, stream);
    std::vector<float> values;
    values.reserve(typed.size());
    for (const T& value : typed) {
        values.push_back(static_cast<float>(value));
    }
    return values;
}

template <typename T>
T fp8FromBytePattern(uint8_t byte) {
    static_assert(sizeof(T) == sizeof(uint8_t));
    T value{};
    std::memcpy(&value, &byte, sizeof(uint8_t));
    return value;
}

template <typename T>
uint8_t fp8BytePattern(const T& value) {
    static_assert(sizeof(T) == sizeof(uint8_t));
    uint8_t byte = 0;
    std::memcpy(&byte, &value, sizeof(uint8_t));
    return byte;
}

template <typename T>
std::vector<uint8_t> fp8BytePatterns(const std::vector<T>& values) {
    std::vector<uint8_t> bytes;
    bytes.reserve(values.size());
    for (const T& value : values) {
        bytes.push_back(fp8BytePattern(value));
    }
    return bytes;
}

template <typename T>
void expectRunLengthEncodeFp8BitwiseDType() {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    const std::vector<uint8_t> input_bytes = {0x00U, 0x00U, 0x80U, 0x38U, 0x38U, 0x7FU, 0x7FU};
    std::vector<T> typed_input;
    typed_input.reserve(input_bytes.size());
    for (const uint8_t byte : input_bytes) {
        typed_input.push_back(fp8FromBytePattern<T>(byte));
    }

    Tensor input = makeGpuVector<T>(typed_input, stream);
    Tensor unique_out(gpuPlacement, TensorDescriptor(dtypeFor<T>(), {static_cast<uint64_t>(typed_input.size())}));
    Tensor counts_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {static_cast<uint64_t>(typed_input.size())}));
    Tensor num_runs_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {1}));

    const CubDeviceRunLengthEncodePlan rle_plan =
        prepareCubDeviceRunLengthEncode(input, unique_out, counts_out, num_runs_out, typed_input.size());
    Tensor temp = allocateCubTemporaryStorage(cubTemporaryStoragePlan(gpuPlacement, rle_plan.temp_storage_bytes));
    cubDeviceRunLengthEncode(rle_plan, temp, input, unique_out, counts_out, num_runs_out, stream);
    stream.synchronize();

    std::vector<uint8_t> expected_unique_bytes;
    std::vector<uint32_t> expected_counts;
    for (const uint8_t byte : input_bytes) {
        if (expected_unique_bytes.empty() || expected_unique_bytes.back() != byte) {
            expected_unique_bytes.push_back(byte);
            expected_counts.push_back(1U);
        } else {
            ++expected_counts.back();
        }
    }

    const auto unique_typed = copyGpuVector<T>(unique_out, stream);
    const std::vector<uint8_t> unique_bytes = fp8BytePatterns(unique_typed);
    const auto counts = copyGpuVector<uint32_t>(counts_out, stream);

    EXPECT_EQ(copyGpuVector<uint32_t>(num_runs_out, stream), (std::vector<uint32_t>{static_cast<uint32_t>(expected_unique_bytes.size())}));
    EXPECT_EQ(std::vector<uint8_t>(unique_bytes.begin(), unique_bytes.begin() + expected_unique_bytes.size()), expected_unique_bytes);
    EXPECT_EQ(std::vector<uint32_t>(counts.begin(), counts.begin() + expected_counts.size()), expected_counts);
}

void expectFloatVectorNear(const std::vector<float>& actual, const std::vector<float>& expected, float atol = 0.0f) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], atol) << "index " << i;
    }
}

template <typename KeyT>
void expectSortKeysDescendingFloatingDType(const std::vector<float>& input_values,
                                           const std::vector<float>& expected_values,
                                           float atol = 0.0f) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    std::vector<KeyT> typed_input;
    typed_input.reserve(input_values.size());
    for (float value : input_values) {
        typed_input.emplace_back(value);
    }

    Tensor keys_in = makeGpuVector<KeyT>(typed_input, stream);
    Tensor keys_out(gpuPlacement, TensorDescriptor(dtypeFor<KeyT>(), {static_cast<uint64_t>(input_values.size())}));

    const CubDeviceRadixSortKeysPlan sort_plan =
        prepareCubDeviceRadixSortKeys(keys_in, keys_out, input_values.size(), CubSortOrder::Descending);
    Tensor temp = allocateCubTemporaryStorage(cubTemporaryStoragePlan(gpuPlacement, sort_plan.temp_storage_bytes));
    cubDeviceRadixSortKeys(sort_plan, temp, keys_in, keys_out, stream);
    stream.synchronize();

    expectFloatVectorNear(copyGpuVectorAsFloat<KeyT>(keys_out, stream), expected_values, atol);
}


template <typename T>
void expectRunLengthEncodeFloatingDType(const std::vector<float>& input_values,
                                        const std::vector<float>& expected_unique,
                                        const std::vector<uint32_t>& expected_counts,
                                        float atol = 0.0f) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    std::vector<T> typed_input;
    typed_input.reserve(input_values.size());
    for (float value : input_values) {
        typed_input.emplace_back(value);
    }

    Tensor input = makeGpuVector<T>(typed_input, stream);
    Tensor unique_out(gpuPlacement, TensorDescriptor(dtypeFor<T>(), {static_cast<uint64_t>(input_values.size())}));
    Tensor counts_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {static_cast<uint64_t>(input_values.size())}));
    Tensor num_runs_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {1}));

    const CubDeviceRunLengthEncodePlan rle_plan =
        prepareCubDeviceRunLengthEncode(input, unique_out, counts_out, num_runs_out, input_values.size());
    Tensor temp = allocateCubTemporaryStorage(cubTemporaryStoragePlan(gpuPlacement, rle_plan.temp_storage_bytes));
    cubDeviceRunLengthEncode(rle_plan, temp, input, unique_out, counts_out, num_runs_out, stream);
    stream.synchronize();

    EXPECT_EQ(copyGpuVector<uint32_t>(num_runs_out, stream), (std::vector<uint32_t>{static_cast<uint32_t>(expected_unique.size())}));

    const auto unique = copyGpuVectorAsFloat<T>(unique_out, stream);
    const auto counts = copyGpuVector<uint32_t>(counts_out, stream);
    expectFloatVectorNear(std::vector<float>(unique.begin(), unique.begin() + expected_unique.size()), expected_unique, atol);
    EXPECT_EQ(std::vector<uint32_t>(counts.begin(), counts.begin() + expected_counts.size()), expected_counts);
}

template <typename T>
void expectExclusiveScanFloatingDType(const std::vector<float>& input_values,
                                      const std::vector<float>& expected_values,
                                      float atol) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    std::vector<T> typed_input;
    typed_input.reserve(input_values.size());
    for (float value : input_values) {
        typed_input.emplace_back(value);
    }

    Tensor input = makeGpuVector<T>(typed_input, stream);
    Tensor output(gpuPlacement, TensorDescriptor(dtypeFor<T>(), {static_cast<uint64_t>(input_values.size())}));

    const CubDeviceExclusiveSumPlan scan_plan = prepareCubDeviceExclusiveSum(input, output, input_values.size());
    Tensor temp = allocateCubTemporaryStorage(cubTemporaryStoragePlan(gpuPlacement, scan_plan.temp_storage_bytes));
    cubDeviceExclusiveSum(scan_plan, temp, input, output, stream);
    stream.synchronize();

    expectFloatVectorNear(copyGpuVectorAsFloat<T>(output, stream), expected_values, atol);
}

template <typename T>
void expectSegmentedExclusiveScanFloatingDType(const std::vector<float>& input_values,
                                               const std::vector<uint32_t>& offsets_values,
                                               const std::vector<float>& expected_values,
                                               float atol) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    std::vector<T> typed_input;
    typed_input.reserve(input_values.size());
    for (float value : input_values) {
        typed_input.emplace_back(value);
    }

    Tensor input = makeGpuVector<T>(typed_input, stream);
    Tensor offsets = makeGpuVector<uint32_t>(offsets_values, stream);
    Tensor output(gpuPlacement, TensorDescriptor(dtypeFor<T>(), {static_cast<uint64_t>(input_values.size())}));

    const CubDeviceSegmentedExclusiveSumPlan scan_plan =
        prepareCubDeviceSegmentedExclusiveSum(input, output, offsets, input_values.size(), offsets_values.size() - 1);
    Tensor temp = allocateCubTemporaryStorage(cubTemporaryStoragePlan(gpuPlacement, scan_plan.temp_storage_bytes));
    cubDeviceSegmentedExclusiveSum(scan_plan, temp, input, output, offsets, stream);
    stream.synchronize();

    expectFloatVectorNear(copyGpuVectorAsFloat<T>(output, stream), expected_values, atol);
}

template <typename T>
void expectSegmentedReduceFloatingDType(const std::vector<float>& input_values,
                                        const std::vector<uint32_t>& offsets_values,
                                        const std::vector<float>& expected_sums,
                                        const std::vector<float>& expected_maxima,
                                        float atol) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    std::vector<T> typed_input;
    typed_input.reserve(input_values.size());
    for (float value : input_values) {
        typed_input.emplace_back(value);
    }

    Tensor input = makeGpuVector<T>(typed_input, stream);
    Tensor offsets = makeGpuVector<uint32_t>(offsets_values, stream);
    Tensor sum_output(gpuPlacement, TensorDescriptor(dtypeFor<T>(), {static_cast<uint64_t>(offsets_values.size() - 1)}));
    Tensor max_output(gpuPlacement, TensorDescriptor(dtypeFor<T>(), {static_cast<uint64_t>(offsets_values.size() - 1)}));

    const CubDeviceSegmentedReduceSumPlan sum_plan =
        prepareCubDeviceSegmentedReduceSum(input, sum_output, offsets, input_values.size(), offsets_values.size() - 1);
    const CubDeviceSegmentedReduceMaxPlan max_plan =
        prepareCubDeviceSegmentedReduceMax(input, max_output, offsets, input_values.size(), offsets_values.size() - 1);
    const CubTemporaryStoragePlan workspace_plan = cubMaxTemporaryStoragePlan(
        {cubTemporaryStoragePlan(gpuPlacement, sum_plan.temp_storage_bytes),
         cubTemporaryStoragePlan(gpuPlacement, max_plan.temp_storage_bytes)});
    Tensor temp = allocateCubTemporaryStorage(workspace_plan);

    cubDeviceSegmentedReduceSum(sum_plan, temp, input, sum_output, offsets, stream);
    cubDeviceSegmentedReduceMax(max_plan, temp, input, max_output, offsets, stream);
    stream.synchronize();

    expectFloatVectorNear(copyGpuVectorAsFloat<T>(sum_output, stream), expected_sums, atol);
    expectFloatVectorNear(copyGpuVectorAsFloat<T>(max_output, stream), expected_maxima, atol);
}

template <typename T>
void expectDeviceReduceFloatingDType(const std::vector<float>& input_values,
                                     float expected_sum,
                                     float expected_max,
                                     float expected_min,
                                     float atol) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    std::vector<T> typed_input;
    typed_input.reserve(input_values.size());
    for (float value : input_values) {
        typed_input.emplace_back(value);
    }

    Tensor input = makeGpuVector<T>(typed_input, stream);
    Tensor sum_output(gpuPlacement, TensorDescriptor(dtypeFor<T>(), {1}));
    Tensor max_output(gpuPlacement, TensorDescriptor(dtypeFor<T>(), {1}));
    Tensor min_output(gpuPlacement, TensorDescriptor(dtypeFor<T>(), {1}));

    const CubDeviceReduceSumPlan sum_plan = prepareCubDeviceReduceSum(input, sum_output, input_values.size());
    const CubDeviceReduceMaxPlan max_plan = prepareCubDeviceReduceMax(input, max_output, input_values.size());
    const CubDeviceReduceMinPlan min_plan = prepareCubDeviceReduceMin(input, min_output, input_values.size());
    const CubTemporaryStoragePlan workspace_plan = cubMaxTemporaryStoragePlan(
        {cubTemporaryStoragePlan(gpuPlacement, sum_plan.temp_storage_bytes),
         cubTemporaryStoragePlan(gpuPlacement, max_plan.temp_storage_bytes),
         cubTemporaryStoragePlan(gpuPlacement, min_plan.temp_storage_bytes)});
    Tensor temp = allocateCubTemporaryStorage(workspace_plan);

    cubDeviceReduceSum(sum_plan, temp, input, sum_output, stream);
    cubDeviceReduceMax(max_plan, temp, input, max_output, stream);
    cubDeviceReduceMin(min_plan, temp, input, min_output, stream);
    stream.synchronize();

    expectFloatVectorNear(copyGpuVectorAsFloat<T>(sum_output, stream), {expected_sum}, atol);
    expectFloatVectorNear(copyGpuVectorAsFloat<T>(max_output, stream), {expected_max}, atol);
    expectFloatVectorNear(copyGpuVectorAsFloat<T>(min_output, stream), {expected_min}, atol);
}

template <typename T>
void expectTopKKeysFloatingDType(const std::vector<float>& input_values,
                                 const std::vector<float>& expected_values,
                                 CubTopKOrder order,
                                 float atol) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    std::vector<T> typed_input;
    typed_input.reserve(input_values.size());
    for (float value : input_values) {
        typed_input.emplace_back(value);
    }

    Tensor keys_in = makeGpuVector<T>(typed_input, stream);
    Tensor keys_out(gpuPlacement, TensorDescriptor(dtypeFor<T>(), {static_cast<uint64_t>(expected_values.size())}));

    const CubDeviceTopKKeysPlan topk_plan =
        prepareCubDeviceTopKKeys(keys_in, keys_out, input_values.size(), expected_values.size(), order);
    Tensor temp = allocateCubTemporaryStorage(cubTemporaryStoragePlan(gpuPlacement, topk_plan.temp_storage_bytes));
    cubDeviceTopKKeys(topk_plan, temp, keys_in, keys_out, stream);
    stream.synchronize();

    auto actual = copyGpuVectorAsFloat<T>(keys_out, stream);
    if (order == CubTopKOrder::Largest) {
        std::sort(actual.begin(), actual.end(), std::greater<float>());
    } else {
        std::sort(actual.begin(), actual.end());
    }
    expectFloatVectorNear(actual, expected_values, atol);
}

template <typename T>
void expectSegmentedTopKKeysFloatingDType(const std::vector<float>& input_values,
                                          uint64_t num_segments,
                                          uint64_t segment_size,
                                          uint64_t k,
                                          const std::vector<float>& expected_values,
                                          CubTopKOrder order,
                                          float atol) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    std::vector<T> typed_input;
    typed_input.reserve(input_values.size());
    for (float value : input_values) {
        typed_input.emplace_back(value);
    }

    Tensor keys_in = makeGpuVector<T>(typed_input, stream);
    Tensor keys_out(gpuPlacement, TensorDescriptor(dtypeFor<T>(), {num_segments * k}));

    const CubDeviceSegmentedTopKKeysPlan topk_plan =
        prepareCubDeviceSegmentedTopKKeys(keys_in, keys_out, num_segments, segment_size, k, order);
    Tensor temp = allocateCubTemporaryStorage(cubTemporaryStoragePlan(gpuPlacement, topk_plan.temp_storage_bytes));
    cubDeviceSegmentedTopKKeys(topk_plan, temp, keys_in, keys_out, stream);
    stream.synchronize();

    auto actual = copyGpuVectorAsFloat<T>(keys_out, stream);
    for (uint64_t segment = 0; segment < num_segments; ++segment) {
        auto begin = actual.begin() + static_cast<std::ptrdiff_t>(segment * k);
        auto end = begin + static_cast<std::ptrdiff_t>(k);
        if (order == CubTopKOrder::Largest) {
            std::sort(begin, end, std::greater<float>());
        } else {
            std::sort(begin, end);
        }
    }
    expectFloatVectorNear(actual, expected_values, atol);
}

}  // namespace

TEST(CubDevicePrimitives, DTypeSupportPolicyMatchesFeatureDefines) {
    EXPECT_TRUE(isCubRadixSortKeyDTypeSupported(DataType::UINT8));
    EXPECT_TRUE(isCubRadixSortKeyDTypeSupported(DataType::INT8));
    EXPECT_TRUE(isCubRadixSortKeyDTypeSupported(DataType::UINT16));
    EXPECT_TRUE(isCubRadixSortKeyDTypeSupported(DataType::INT16));
    EXPECT_TRUE(isCubRadixSortKeyDTypeSupported(DataType::UINT32));
    EXPECT_TRUE(isCubRadixSortKeyDTypeSupported(DataType::INT32));
    EXPECT_TRUE(isCubRadixSortKeyDTypeSupported(DataType::FP16));
    EXPECT_TRUE(isCubRadixSortKeyDTypeSupported(DataType::BF16));
    EXPECT_TRUE(isCubRadixSortKeyDTypeSupported(DataType::FP32));

    EXPECT_TRUE(isCubRadixSortValueDTypeSupported(DataType::UINT32));
    EXPECT_FALSE(isCubRadixSortValueDTypeSupported(DataType::FP16));
    EXPECT_FALSE(isCubRadixSortValueDTypeSupported(DataType::BF16));
    EXPECT_FALSE(isCubRadixSortValueDTypeSupported(DataType::FP8_E4M3));

    EXPECT_TRUE(isCubTopKKeyDTypeSupported(DataType::UINT8));
    EXPECT_TRUE(isCubTopKKeyDTypeSupported(DataType::INT8));
    EXPECT_TRUE(isCubTopKKeyDTypeSupported(DataType::UINT16));
    EXPECT_TRUE(isCubTopKKeyDTypeSupported(DataType::INT16));
    EXPECT_TRUE(isCubTopKKeyDTypeSupported(DataType::UINT32));
    EXPECT_TRUE(isCubTopKKeyDTypeSupported(DataType::INT32));
    EXPECT_TRUE(isCubTopKKeyDTypeSupported(DataType::FP16));
    EXPECT_TRUE(isCubTopKKeyDTypeSupported(DataType::BF16));
    EXPECT_TRUE(isCubTopKKeyDTypeSupported(DataType::FP32));
    EXPECT_FALSE(isCubTopKKeyDTypeSupported(DataType::FP8_E4M3));
    EXPECT_FALSE(isCubTopKKeyDTypeSupported(DataType::FP8_E5M2));
    EXPECT_TRUE(isCubTopKValueDTypeSupported(DataType::UINT32));
    EXPECT_FALSE(isCubTopKValueDTypeSupported(DataType::FP32));

    EXPECT_TRUE(isCubSelectValueDTypeSupported(DataType::UINT8));
    EXPECT_TRUE(isCubSelectValueDTypeSupported(DataType::INT8));
    EXPECT_TRUE(isCubSelectValueDTypeSupported(DataType::UINT16));
    EXPECT_TRUE(isCubSelectValueDTypeSupported(DataType::INT16));
    EXPECT_TRUE(isCubSelectValueDTypeSupported(DataType::UINT32));
    EXPECT_TRUE(isCubSelectValueDTypeSupported(DataType::INT32));
    EXPECT_TRUE(isCubSelectValueDTypeSupported(DataType::FP16));
    EXPECT_TRUE(isCubSelectValueDTypeSupported(DataType::BF16));
    EXPECT_TRUE(isCubSelectValueDTypeSupported(DataType::FP32));
    EXPECT_TRUE(isCubSelectFlagDTypeSupported(DataType::BOOLEAN));
    EXPECT_TRUE(isCubSelectFlagDTypeSupported(DataType::UINT8));
    EXPECT_FALSE(isCubSelectFlagDTypeSupported(DataType::INT32));

    EXPECT_TRUE(isCubRunLengthEncodeDTypeSupported(DataType::UINT8));
    EXPECT_TRUE(isCubRunLengthEncodeDTypeSupported(DataType::INT8));
    EXPECT_TRUE(isCubRunLengthEncodeDTypeSupported(DataType::UINT16));
    EXPECT_TRUE(isCubRunLengthEncodeDTypeSupported(DataType::INT16));
    EXPECT_TRUE(isCubRunLengthEncodeDTypeSupported(DataType::UINT32));
    EXPECT_TRUE(isCubRunLengthEncodeDTypeSupported(DataType::INT32));
    EXPECT_TRUE(isCubRunLengthEncodeDTypeSupported(DataType::FP16));
    EXPECT_TRUE(isCubRunLengthEncodeDTypeSupported(DataType::BF16));
    EXPECT_TRUE(isCubRunLengthEncodeDTypeSupported(DataType::FP32));

    EXPECT_TRUE(isCubExclusiveSumDTypeSupported(DataType::UINT32));
    EXPECT_FALSE(isCubExclusiveSumDTypeSupported(DataType::INT32));
    EXPECT_TRUE(isCubExclusiveSumDTypeSupported(DataType::FP16));
    EXPECT_TRUE(isCubExclusiveSumDTypeSupported(DataType::BF16));
    EXPECT_TRUE(isCubExclusiveSumDTypeSupported(DataType::FP32));
    EXPECT_FALSE(isCubExclusiveSumDTypeSupported(DataType::FP8_E4M3));
    EXPECT_FALSE(isCubExclusiveSumDTypeSupported(DataType::FP8_E5M2));

    EXPECT_TRUE(isCubReduceSumDTypeSupported(DataType::UINT32));
    EXPECT_FALSE(isCubReduceSumDTypeSupported(DataType::INT32));
    EXPECT_TRUE(isCubReduceSumDTypeSupported(DataType::FP16));
    EXPECT_TRUE(isCubReduceSumDTypeSupported(DataType::BF16));
    EXPECT_TRUE(isCubReduceSumDTypeSupported(DataType::FP32));
    EXPECT_FALSE(isCubReduceSumDTypeSupported(DataType::FP8_E4M3));
    EXPECT_FALSE(isCubReduceSumDTypeSupported(DataType::FP8_E5M2));

    EXPECT_TRUE(isCubReduceMaxDTypeSupported(DataType::UINT32));
    EXPECT_FALSE(isCubReduceMaxDTypeSupported(DataType::INT32));
    EXPECT_TRUE(isCubReduceMaxDTypeSupported(DataType::FP16));
    EXPECT_TRUE(isCubReduceMaxDTypeSupported(DataType::BF16));
    EXPECT_TRUE(isCubReduceMaxDTypeSupported(DataType::FP32));
    EXPECT_FALSE(isCubReduceMaxDTypeSupported(DataType::FP8_E4M3));
    EXPECT_FALSE(isCubReduceMaxDTypeSupported(DataType::FP8_E5M2));

    EXPECT_TRUE(isCubReduceMinDTypeSupported(DataType::UINT32));
    EXPECT_FALSE(isCubReduceMinDTypeSupported(DataType::INT32));
    EXPECT_TRUE(isCubReduceMinDTypeSupported(DataType::FP16));
    EXPECT_TRUE(isCubReduceMinDTypeSupported(DataType::BF16));
    EXPECT_TRUE(isCubReduceMinDTypeSupported(DataType::FP32));
    EXPECT_FALSE(isCubReduceMinDTypeSupported(DataType::FP8_E4M3));
    EXPECT_FALSE(isCubReduceMinDTypeSupported(DataType::FP8_E5M2));

    EXPECT_TRUE(isCubSegmentedExclusiveSumDTypeSupported(DataType::UINT32));
    EXPECT_FALSE(isCubSegmentedExclusiveSumDTypeSupported(DataType::INT32));
    EXPECT_TRUE(isCubSegmentedExclusiveSumDTypeSupported(DataType::FP16));
    EXPECT_TRUE(isCubSegmentedExclusiveSumDTypeSupported(DataType::BF16));
    EXPECT_TRUE(isCubSegmentedExclusiveSumDTypeSupported(DataType::FP32));
    EXPECT_FALSE(isCubSegmentedExclusiveSumDTypeSupported(DataType::FP8_E4M3));
    EXPECT_FALSE(isCubSegmentedExclusiveSumDTypeSupported(DataType::FP8_E5M2));

    EXPECT_TRUE(isCubSegmentedReduceSumDTypeSupported(DataType::UINT32));
    EXPECT_FALSE(isCubSegmentedReduceSumDTypeSupported(DataType::INT32));
    EXPECT_TRUE(isCubSegmentedReduceSumDTypeSupported(DataType::FP16));
    EXPECT_TRUE(isCubSegmentedReduceSumDTypeSupported(DataType::BF16));
    EXPECT_TRUE(isCubSegmentedReduceSumDTypeSupported(DataType::FP32));
    EXPECT_FALSE(isCubSegmentedReduceSumDTypeSupported(DataType::FP8_E4M3));
    EXPECT_FALSE(isCubSegmentedReduceSumDTypeSupported(DataType::FP8_E5M2));

    EXPECT_TRUE(isCubSegmentedReduceMaxDTypeSupported(DataType::UINT32));
    EXPECT_FALSE(isCubSegmentedReduceMaxDTypeSupported(DataType::INT32));
    EXPECT_TRUE(isCubSegmentedReduceMaxDTypeSupported(DataType::FP16));
    EXPECT_TRUE(isCubSegmentedReduceMaxDTypeSupported(DataType::BF16));
    EXPECT_TRUE(isCubSegmentedReduceMaxDTypeSupported(DataType::FP32));
    EXPECT_FALSE(isCubSegmentedReduceMaxDTypeSupported(DataType::FP8_E4M3));
    EXPECT_FALSE(isCubSegmentedReduceMaxDTypeSupported(DataType::FP8_E5M2));

    EXPECT_TRUE(isCubSegmentOffsetDTypeSupported(DataType::UINT32));
    EXPECT_FALSE(isCubSegmentOffsetDTypeSupported(DataType::INT32));
    EXPECT_FALSE(isCubSegmentOffsetDTypeSupported(DataType::FP32));

#if THOR_CUB_ENABLE_64BIT_TYPES
    EXPECT_TRUE(isCubRadixSortKeyDTypeSupported(DataType::UINT64));
    EXPECT_TRUE(isCubRadixSortKeyDTypeSupported(DataType::INT64));
    EXPECT_TRUE(isCubRadixSortKeyDTypeSupported(DataType::FP64));
    EXPECT_TRUE(isCubRadixSortValueDTypeSupported(DataType::UINT64));
    EXPECT_TRUE(isCubTopKKeyDTypeSupported(DataType::UINT64));
    EXPECT_TRUE(isCubTopKKeyDTypeSupported(DataType::INT64));
    EXPECT_TRUE(isCubTopKKeyDTypeSupported(DataType::FP64));
    EXPECT_TRUE(isCubTopKValueDTypeSupported(DataType::UINT64));
    EXPECT_FALSE(isCubTopKValueDTypeSupported(DataType::INT64));
    EXPECT_FALSE(isCubTopKValueDTypeSupported(DataType::FP64));
    EXPECT_TRUE(isCubSelectValueDTypeSupported(DataType::UINT64));
    EXPECT_TRUE(isCubSelectValueDTypeSupported(DataType::INT64));
    EXPECT_TRUE(isCubSelectValueDTypeSupported(DataType::FP64));
    EXPECT_TRUE(isCubRunLengthEncodeDTypeSupported(DataType::UINT64));
    EXPECT_TRUE(isCubRunLengthEncodeDTypeSupported(DataType::INT64));
    EXPECT_TRUE(isCubRunLengthEncodeDTypeSupported(DataType::FP64));
    EXPECT_TRUE(isCubExclusiveSumDTypeSupported(DataType::UINT64));
    EXPECT_FALSE(isCubExclusiveSumDTypeSupported(DataType::INT64));
    EXPECT_TRUE(isCubExclusiveSumDTypeSupported(DataType::FP64));
    EXPECT_TRUE(isCubReduceSumDTypeSupported(DataType::UINT64));
    EXPECT_FALSE(isCubReduceSumDTypeSupported(DataType::INT64));
    EXPECT_TRUE(isCubReduceSumDTypeSupported(DataType::FP64));
    EXPECT_TRUE(isCubReduceMaxDTypeSupported(DataType::UINT64));
    EXPECT_FALSE(isCubReduceMaxDTypeSupported(DataType::INT64));
    EXPECT_TRUE(isCubReduceMaxDTypeSupported(DataType::FP64));
    EXPECT_TRUE(isCubReduceMinDTypeSupported(DataType::UINT64));
    EXPECT_FALSE(isCubReduceMinDTypeSupported(DataType::INT64));
    EXPECT_TRUE(isCubReduceMinDTypeSupported(DataType::FP64));
    EXPECT_TRUE(isCubSegmentedExclusiveSumDTypeSupported(DataType::UINT64));
    EXPECT_FALSE(isCubSegmentedExclusiveSumDTypeSupported(DataType::INT64));
    EXPECT_TRUE(isCubSegmentedExclusiveSumDTypeSupported(DataType::FP64));
    EXPECT_TRUE(isCubSegmentedReduceSumDTypeSupported(DataType::UINT64));
    EXPECT_FALSE(isCubSegmentedReduceSumDTypeSupported(DataType::INT64));
    EXPECT_TRUE(isCubSegmentedReduceSumDTypeSupported(DataType::FP64));
    EXPECT_TRUE(isCubSegmentedReduceMaxDTypeSupported(DataType::UINT64));
    EXPECT_FALSE(isCubSegmentedReduceMaxDTypeSupported(DataType::INT64));
    EXPECT_TRUE(isCubSegmentedReduceMaxDTypeSupported(DataType::FP64));
    EXPECT_TRUE(isCubSegmentOffsetDTypeSupported(DataType::UINT64));
    EXPECT_FALSE(isCubSegmentOffsetDTypeSupported(DataType::INT64));
#else
    EXPECT_FALSE(isCubRadixSortKeyDTypeSupported(DataType::UINT64));
    EXPECT_FALSE(isCubRadixSortKeyDTypeSupported(DataType::INT64));
    EXPECT_FALSE(isCubRadixSortKeyDTypeSupported(DataType::FP64));
    EXPECT_FALSE(isCubRadixSortValueDTypeSupported(DataType::UINT64));
    EXPECT_FALSE(isCubTopKKeyDTypeSupported(DataType::UINT64));
    EXPECT_FALSE(isCubTopKKeyDTypeSupported(DataType::INT64));
    EXPECT_FALSE(isCubTopKKeyDTypeSupported(DataType::FP64));
    EXPECT_FALSE(isCubTopKValueDTypeSupported(DataType::UINT64));
    EXPECT_FALSE(isCubSelectValueDTypeSupported(DataType::UINT64));
    EXPECT_FALSE(isCubSelectValueDTypeSupported(DataType::INT64));
    EXPECT_FALSE(isCubSelectValueDTypeSupported(DataType::FP64));
    EXPECT_FALSE(isCubRunLengthEncodeDTypeSupported(DataType::UINT64));
    EXPECT_FALSE(isCubRunLengthEncodeDTypeSupported(DataType::INT64));
    EXPECT_FALSE(isCubRunLengthEncodeDTypeSupported(DataType::FP64));
    EXPECT_FALSE(isCubExclusiveSumDTypeSupported(DataType::UINT64));
    EXPECT_FALSE(isCubExclusiveSumDTypeSupported(DataType::INT64));
    EXPECT_FALSE(isCubExclusiveSumDTypeSupported(DataType::FP64));
    EXPECT_FALSE(isCubReduceSumDTypeSupported(DataType::UINT64));
    EXPECT_FALSE(isCubReduceSumDTypeSupported(DataType::INT64));
    EXPECT_FALSE(isCubReduceSumDTypeSupported(DataType::FP64));
    EXPECT_FALSE(isCubReduceMaxDTypeSupported(DataType::UINT64));
    EXPECT_FALSE(isCubReduceMaxDTypeSupported(DataType::INT64));
    EXPECT_FALSE(isCubReduceMaxDTypeSupported(DataType::FP64));
    EXPECT_FALSE(isCubReduceMinDTypeSupported(DataType::UINT64));
    EXPECT_FALSE(isCubReduceMinDTypeSupported(DataType::INT64));
    EXPECT_FALSE(isCubReduceMinDTypeSupported(DataType::FP64));
    EXPECT_FALSE(isCubSegmentedExclusiveSumDTypeSupported(DataType::UINT64));
    EXPECT_FALSE(isCubSegmentedExclusiveSumDTypeSupported(DataType::INT64));
    EXPECT_FALSE(isCubSegmentedExclusiveSumDTypeSupported(DataType::FP64));
    EXPECT_FALSE(isCubSegmentedReduceSumDTypeSupported(DataType::UINT64));
    EXPECT_FALSE(isCubSegmentedReduceSumDTypeSupported(DataType::INT64));
    EXPECT_FALSE(isCubSegmentedReduceSumDTypeSupported(DataType::FP64));
    EXPECT_FALSE(isCubSegmentedReduceMaxDTypeSupported(DataType::UINT64));
    EXPECT_FALSE(isCubSegmentedReduceMaxDTypeSupported(DataType::INT64));
    EXPECT_FALSE(isCubSegmentedReduceMaxDTypeSupported(DataType::FP64));
    EXPECT_FALSE(isCubSegmentOffsetDTypeSupported(DataType::UINT64));
    EXPECT_FALSE(isCubSegmentOffsetDTypeSupported(DataType::INT64));
#endif

#if THOR_CUB_ENABLE_FP8_TYPES
    EXPECT_TRUE(isCubRadixSortKeyDTypeSupported(DataType::FP8_E4M3));
    EXPECT_TRUE(isCubRadixSortKeyDTypeSupported(DataType::FP8_E5M2));
    EXPECT_TRUE(isCubSelectValueDTypeSupported(DataType::FP8_E4M3));
    EXPECT_TRUE(isCubSelectValueDTypeSupported(DataType::FP8_E5M2));
    EXPECT_TRUE(isCubRunLengthEncodeDTypeSupported(DataType::FP8_E4M3));
    EXPECT_TRUE(isCubRunLengthEncodeDTypeSupported(DataType::FP8_E5M2));
#else
    EXPECT_FALSE(isCubRadixSortKeyDTypeSupported(DataType::FP8_E4M3));
    EXPECT_FALSE(isCubRadixSortKeyDTypeSupported(DataType::FP8_E5M2));
    EXPECT_FALSE(isCubSelectValueDTypeSupported(DataType::FP8_E4M3));
    EXPECT_FALSE(isCubSelectValueDTypeSupported(DataType::FP8_E5M2));
    EXPECT_FALSE(isCubRunLengthEncodeDTypeSupported(DataType::FP8_E4M3));
    EXPECT_FALSE(isCubRunLengthEncodeDTypeSupported(DataType::FP8_E5M2));
#endif
}

TEST(CubDevicePrimitives, SortKeysDescendingSupportsFp16AndBf16Keys) {
    expectSortKeysDescendingFloatingDType<__half>({0.5f, -1.0f, 3.0f, 2.0f}, {3.0f, 2.0f, 0.5f, -1.0f});
    expectSortKeysDescendingFloatingDType<__nv_bfloat16>({0.5f, -1.0f, 3.0f, 2.0f}, {3.0f, 2.0f, 0.5f, -1.0f});
}

#if THOR_CUB_ENABLE_FP8_TYPES
TEST(CubDevicePrimitives, SortKeysDescendingSupportsFp8KeysWhenEnabled) {
    expectSortKeysDescendingFloatingDType<__nv_fp8_e4m3>({0.5f, -1.0f, 3.0f, 2.0f}, {3.0f, 2.0f, 0.5f, -1.0f});
    expectSortKeysDescendingFloatingDType<__nv_fp8_e5m2>({0.5f, -1.0f, 3.0f, 2.0f}, {3.0f, 2.0f, 0.5f, -1.0f});
}
#else
TEST(CubDevicePrimitives, SortKeysRejectsFp8KeysWhenDisabled) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor keys_in = makeGpuVector<__nv_fp8_e4m3>({__nv_fp8_e4m3(2.0f), __nv_fp8_e4m3(1.0f)}, stream);
    Tensor keys_out(gpuPlacement, TensorDescriptor(DataType::FP8_E4M3, {2}));
    EXPECT_THROW((void)prepareCubDeviceRadixSortKeys(keys_in, keys_out, 2, CubSortOrder::Descending), std::invalid_argument);
}
#endif

#if THOR_CUB_ENABLE_64BIT_TYPES
TEST(CubDevicePrimitives, SortKeysDescendingSupports64BitKeysWhenEnabled) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor uint_keys_in = makeGpuVector<uint64_t>({2ULL, 9ULL, 1ULL, 5ULL}, stream);
    Tensor uint_keys_out(gpuPlacement, TensorDescriptor(DataType::UINT64, {4}));
    const CubDeviceRadixSortKeysPlan uint_plan =
        prepareCubDeviceRadixSortKeys(uint_keys_in, uint_keys_out, 4, CubSortOrder::Descending);
    Tensor uint_temp = allocateCubTemporaryStorage(cubTemporaryStoragePlan(gpuPlacement, uint_plan.temp_storage_bytes));
    cubDeviceRadixSortKeys(uint_plan, uint_temp, uint_keys_in, uint_keys_out, stream);
    stream.synchronize();
    EXPECT_EQ(copyGpuVector<uint64_t>(uint_keys_out, stream), (std::vector<uint64_t>{9ULL, 5ULL, 2ULL, 1ULL}));

    expectSortKeysDescendingFloatingDType<double>({0.25f, -3.0f, 9.0f, 1.0f}, {9.0f, 1.0f, 0.25f, -3.0f});
}
#else
TEST(CubDevicePrimitives, SortKeysRejects64BitKeysWhenDisabled) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor uint_keys_in = makeGpuVector<uint64_t>({2ULL, 1ULL}, stream);
    Tensor uint_keys_out(gpuPlacement, TensorDescriptor(DataType::UINT64, {2}));
    EXPECT_THROW((void)prepareCubDeviceRadixSortKeys(uint_keys_in, uint_keys_out, 2, CubSortOrder::Descending),
                 std::invalid_argument);

    Tensor fp_keys_in = makeGpuVector<double>({2.0, 1.0}, stream);
    Tensor fp_keys_out(gpuPlacement, TensorDescriptor(DataType::FP64, {2}));
    EXPECT_THROW((void)prepareCubDeviceRadixSortKeys(fp_keys_in, fp_keys_out, 2, CubSortOrder::Descending),
                 std::invalid_argument);
}
#endif

TEST(CubDevicePrimitives, SortPairsDescendingFloatKeysCarriesUint32Values) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor keys_in = makeGpuVector<float>({0.25f, 3.0f, -1.0f, 3.0f, 2.0f}, stream);
    Tensor values_in = makeGpuVector<uint32_t>({10U, 11U, 12U, 13U, 14U}, stream);
    Tensor keys_out(gpuPlacement, TensorDescriptor(DataType::FP32, {5}));
    Tensor values_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {5}));

    const CubDeviceRadixSortPairsPlan sort_plan = prepareCubDeviceRadixSortPairs(
        keys_in, keys_out, values_in, values_out, 5, CubSortOrder::Descending);
    Tensor temp = allocateCubTemporaryStorage(cubTemporaryStoragePlan(gpuPlacement, sort_plan.temp_storage_bytes));
    cubDeviceRadixSortPairs(sort_plan, temp, keys_in, keys_out, values_in, values_out, stream);
    stream.synchronize();

    EXPECT_EQ(copyGpuVector<float>(keys_out, stream), (std::vector<float>{3.0f, 3.0f, 2.0f, 0.25f, -1.0f}));
    EXPECT_EQ(copyGpuVector<uint32_t>(values_out, stream), (std::vector<uint32_t>{11U, 13U, 14U, 10U, 12U}));
}


#if THOR_CUB_ENABLE_64BIT_TYPES
TEST(CubDevicePrimitives, SortPairsDescendingFloatKeysCarriesUint64IndexValuesWhen64BitEnabled) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor keys_in = makeGpuVector<float>({2.0f, -3.0f, 4.0f, 1.0f}, stream);
    Tensor values_in = makeGpuVector<uint64_t>({100U, 101U, 102U, 103U}, stream);
    Tensor keys_out(gpuPlacement, TensorDescriptor(DataType::FP32, {4}));
    Tensor values_out(gpuPlacement, TensorDescriptor(DataType::UINT64, {4}));

    const CubDeviceRadixSortPairsPlan sort_plan = prepareCubDeviceRadixSortPairs(
        keys_in, keys_out, values_in, values_out, 4, CubSortOrder::Descending);
    Tensor temp = allocateCubTemporaryStorage(cubTemporaryStoragePlan(gpuPlacement, sort_plan.temp_storage_bytes));
    cubDeviceRadixSortPairs(sort_plan, temp, keys_in, keys_out, values_in, values_out, stream);
    stream.synchronize();

    EXPECT_EQ(copyGpuVector<float>(keys_out, stream), (std::vector<float>{4.0f, 2.0f, 1.0f, -3.0f}));
    EXPECT_EQ(copyGpuVector<uint64_t>(values_out, stream), (std::vector<uint64_t>{102U, 100U, 103U, 101U}));
}
#else
TEST(CubDevicePrimitives, SortPairsRejectsUint64IndexValuesWhen64BitDisabled) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor keys_in = makeGpuVector<float>({2.0f, 1.0f}, stream);
    Tensor values_in = makeGpuVector<uint64_t>({20U, 10U}, stream);
    Tensor keys_out(gpuPlacement, TensorDescriptor(DataType::FP32, {2}));
    Tensor values_out(gpuPlacement, TensorDescriptor(DataType::UINT64, {2}));

    EXPECT_THROW((void)prepareCubDeviceRadixSortPairs(keys_in, keys_out, values_in, values_out, 2, CubSortOrder::Descending),
                 std::invalid_argument);
}
#endif

TEST(CubDevicePrimitives, SortPairsRejectsNonIndexValueDType) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor keys_in = makeGpuVector<float>({2.0f, 1.0f}, stream);
    Tensor values_in = makeGpuVector<float>({20.0f, 10.0f}, stream);
    Tensor keys_out(gpuPlacement, TensorDescriptor(DataType::FP32, {2}));
    Tensor values_out(gpuPlacement, TensorDescriptor(DataType::FP32, {2}));

    EXPECT_THROW((void)prepareCubDeviceRadixSortPairs(keys_in, keys_out, values_in, values_out, 2, CubSortOrder::Descending),
                 std::invalid_argument);
}

TEST(CubDevicePrimitives, TopKKeysLargestAndSmallestUseCubDeviceTopK) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor keys_in = makeGpuVector<uint32_t>({5U, 3U, 1U, 7U, 8U, 2U, 4U, 6U}, stream);
    Tensor largest_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {3}));
    Tensor smallest_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {4}));

    const CubDeviceTopKKeysPlan largest_plan = prepareCubDeviceTopKKeys(keys_in, largest_out, 8, 3, CubTopKOrder::Largest);
    const CubDeviceTopKKeysPlan smallest_plan = prepareCubDeviceTopKKeys(keys_in, smallest_out, 8, 4, CubTopKOrder::Smallest);
    const CubTemporaryStoragePlan workspace_plan = cubMaxTemporaryStoragePlan(
        {cubTemporaryStoragePlan(gpuPlacement, largest_plan.temp_storage_bytes),
         cubTemporaryStoragePlan(gpuPlacement, smallest_plan.temp_storage_bytes)});
    Tensor temp = allocateCubTemporaryStorage(workspace_plan);

    cubDeviceTopKKeys(largest_plan, temp, keys_in, largest_out, stream);
    cubDeviceTopKKeys(smallest_plan, temp, keys_in, smallest_out, stream);
    stream.synchronize();

    auto largest = copyGpuVector<uint32_t>(largest_out, stream);
    auto smallest = copyGpuVector<uint32_t>(smallest_out, stream);
    std::sort(largest.begin(), largest.end(), std::greater<uint32_t>());
    std::sort(smallest.begin(), smallest.end());
    EXPECT_EQ(largest, (std::vector<uint32_t>{8U, 7U, 6U}));
    EXPECT_EQ(smallest, (std::vector<uint32_t>{1U, 2U, 3U, 4U}));
}

TEST(CubDevicePrimitives, TopKPairsCarriesUint32IndexValues) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor keys_in = makeGpuVector<float>({5.0f, -3.0f, 1.0f, 7.0f, 8.0f, 2.0f, 4.0f, 6.0f}, stream);
    Tensor values_in = makeGpuVector<uint32_t>({0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U}, stream);
    Tensor keys_out(gpuPlacement, TensorDescriptor(DataType::FP32, {4}));
    Tensor values_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {4}));

    const CubDeviceTopKPairsPlan topk_plan =
        prepareCubDeviceTopKPairs(keys_in, keys_out, values_in, values_out, 8, 4, CubTopKOrder::Largest);
    Tensor temp = allocateCubTemporaryStorage(cubTemporaryStoragePlan(gpuPlacement, topk_plan.temp_storage_bytes));
    cubDeviceTopKPairs(topk_plan, temp, keys_in, keys_out, values_in, values_out, stream);
    stream.synchronize();

    const auto keys = copyGpuVector<float>(keys_out, stream);
    const auto values = copyGpuVector<uint32_t>(values_out, stream);
    std::vector<std::pair<float, uint32_t>> pairs;
    for (size_t i = 0; i < keys.size(); ++i) {
        pairs.emplace_back(keys[i], values[i]);
    }
    std::sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) { return a.first > b.first; });

    EXPECT_EQ(pairs, (std::vector<std::pair<float, uint32_t>>{{8.0f, 4U}, {7.0f, 3U}, {6.0f, 7U}, {5.0f, 0U}}));
}

TEST(CubDevicePrimitives, TopKSupportsFp16AndBf16Keys) {
    expectTopKKeysFloatingDType<__half>({0.5f, -1.0f, 3.0f, 2.0f, 4.0f}, {4.0f, 3.0f, 2.0f}, CubTopKOrder::Largest, 0.0f);
    expectTopKKeysFloatingDType<__nv_bfloat16>(
        {0.5f, -1.0f, 3.0f, 2.0f, 4.0f}, {4.0f, 3.0f, 2.0f}, CubTopKOrder::Largest, 0.01f);
}

TEST(CubDevicePrimitives, TopKCapsKToNumItems) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor keys_in = makeGpuVector<uint32_t>({5U, 3U, 7U}, stream);
    Tensor keys_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {3}));

    const CubDeviceTopKKeysPlan topk_plan = prepareCubDeviceTopKKeys(keys_in, keys_out, 3, 99, CubTopKOrder::Largest);
    Tensor temp = allocateCubTemporaryStorage(cubTemporaryStoragePlan(gpuPlacement, topk_plan.temp_storage_bytes));
    cubDeviceTopKKeys(topk_plan, temp, keys_in, keys_out, stream);
    stream.synchronize();

    auto keys = copyGpuVector<uint32_t>(keys_out, stream);
    std::sort(keys.begin(), keys.end(), std::greater<uint32_t>());
    EXPECT_EQ(keys, (std::vector<uint32_t>{7U, 5U, 3U}));
}

TEST(CubDevicePrimitives, TopKRejectsUnsupportedDTypesAndMismatches) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor keys_in = makeGpuVector<uint32_t>({1U, 2U, 3U}, stream);
    Tensor fp_keys_out(gpuPlacement, TensorDescriptor(DataType::FP32, {2}));
    EXPECT_THROW((void)prepareCubDeviceTopKKeys(keys_in, fp_keys_out, 3, 2), std::invalid_argument);

    Tensor fp8_keys_in = makeGpuVector<__nv_fp8_e4m3>({__nv_fp8_e4m3(1.0f), __nv_fp8_e4m3(2.0f)}, stream);
    Tensor fp8_keys_out(gpuPlacement, TensorDescriptor(DataType::FP8_E4M3, {1}));
    EXPECT_THROW((void)prepareCubDeviceTopKKeys(fp8_keys_in, fp8_keys_out, 2, 1), std::invalid_argument);

    Tensor values_in = makeGpuVector<float>({10.0f, 20.0f, 30.0f}, stream);
    Tensor values_out(gpuPlacement, TensorDescriptor(DataType::FP32, {2}));
    Tensor keys_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {2}));
    EXPECT_THROW((void)prepareCubDeviceTopKPairs(keys_in, keys_out, values_in, values_out, 3, 2), std::invalid_argument);

    Tensor small_keys_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {1}));
    EXPECT_THROW((void)prepareCubDeviceTopKKeys(keys_in, small_keys_out, 3, 2), std::invalid_argument);
}

TEST(CubDevicePrimitives, SelectFlaggedCompactsUint32WithBooleanFlags) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor input = makeGpuVector<uint32_t>({10U, 20U, 30U, 40U, 50U, 60U}, stream);
    Tensor flags = makeGpuVector<bool>({true, false, true, false, false, true}, stream);
    Tensor output(gpuPlacement, TensorDescriptor(DataType::UINT32, {6}));
    Tensor num_selected_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {1}));

    const CubDeviceSelectFlaggedPlan select_plan = prepareCubDeviceSelectFlagged(input, flags, output, num_selected_out, 6);
    Tensor temp = allocateCubTemporaryStorage(cubTemporaryStoragePlan(gpuPlacement, select_plan.temp_storage_bytes));
    cubDeviceSelectFlagged(select_plan, temp, input, flags, output, num_selected_out, stream);
    stream.synchronize();

    EXPECT_EQ(copyGpuVector<uint32_t>(num_selected_out, stream), (std::vector<uint32_t>{3U}));
    const auto selected = copyGpuVector<uint32_t>(output, stream);
    EXPECT_EQ(std::vector<uint32_t>(selected.begin(), selected.begin() + 3), (std::vector<uint32_t>{10U, 30U, 60U}));
}

TEST(CubDevicePrimitives, SelectFlaggedWritesZeroCountForZeroItems) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor input = makeGpuVector<uint32_t>({123U}, stream);
    Tensor flags = makeGpuVector<uint8_t>({1U}, stream);
    Tensor output(gpuPlacement, TensorDescriptor(DataType::UINT32, {1}));
    Tensor num_selected_out = makeGpuVector<uint32_t>({77U}, stream);

    const CubDeviceSelectFlaggedPlan select_plan = prepareCubDeviceSelectFlagged(input, flags, output, num_selected_out, 0);
    Tensor temp = allocateCubTemporaryStorage(cubTemporaryStoragePlan(gpuPlacement, select_plan.temp_storage_bytes));
    cubDeviceSelectFlagged(select_plan, temp, input, flags, output, num_selected_out, stream);
    stream.synchronize();

    EXPECT_EQ(copyGpuVector<uint32_t>(num_selected_out, stream), (std::vector<uint32_t>{0U}));
}

TEST(CubDevicePrimitives, SelectFlaggedCompactsFloatValuesWithUint8Flags) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor input = makeGpuVector<float>({1.5f, -2.0f, 3.25f, 4.5f, -5.0f}, stream);
    Tensor flags = makeGpuVector<uint8_t>({0U, 1U, 1U, 0U, 1U}, stream);
    Tensor output(gpuPlacement, TensorDescriptor(DataType::FP32, {5}));
    Tensor num_selected_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {1}));

    const CubDeviceSelectFlaggedPlan select_plan = prepareCubDeviceSelectFlagged(input, flags, output, num_selected_out, 5);
    Tensor temp = allocateCubTemporaryStorage(cubTemporaryStoragePlan(gpuPlacement, select_plan.temp_storage_bytes));
    cubDeviceSelectFlagged(select_plan, temp, input, flags, output, num_selected_out, stream);
    stream.synchronize();

    EXPECT_EQ(copyGpuVector<uint32_t>(num_selected_out, stream), (std::vector<uint32_t>{3U}));
    const auto selected = copyGpuVector<float>(output, stream);
    EXPECT_EQ(std::vector<float>(selected.begin(), selected.begin() + 3), (std::vector<float>{-2.0f, 3.25f, -5.0f}));
}

TEST(CubDevicePrimitives, SelectFlaggedSupportsFp16AndBf16Values) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor half_input = makeGpuVector<__half>({__half(1.0f), __half(2.0f), __half(3.0f), __half(4.0f)}, stream);
    Tensor half_flags = makeGpuVector<uint8_t>({1U, 0U, 1U, 0U}, stream);
    Tensor half_output(gpuPlacement, TensorDescriptor(DataType::FP16, {4}));
    Tensor half_count(gpuPlacement, TensorDescriptor(DataType::UINT32, {1}));
    const CubDeviceSelectFlaggedPlan half_plan = prepareCubDeviceSelectFlagged(half_input, half_flags, half_output, half_count, 4);

    Tensor bf16_input = makeGpuVector<__nv_bfloat16>(
        {__nv_bfloat16(5.0f), __nv_bfloat16(6.0f), __nv_bfloat16(7.0f), __nv_bfloat16(8.0f)}, stream);
    Tensor bf16_flags = makeGpuVector<uint8_t>({0U, 1U, 0U, 1U}, stream);
    Tensor bf16_output(gpuPlacement, TensorDescriptor(DataType::BF16, {4}));
    Tensor bf16_count(gpuPlacement, TensorDescriptor(DataType::UINT32, {1}));
    const CubDeviceSelectFlaggedPlan bf16_plan = prepareCubDeviceSelectFlagged(bf16_input, bf16_flags, bf16_output, bf16_count, 4);

    const CubTemporaryStoragePlan workspace_plan = cubMaxTemporaryStoragePlan(
        {cubTemporaryStoragePlan(gpuPlacement, half_plan.temp_storage_bytes),
         cubTemporaryStoragePlan(gpuPlacement, bf16_plan.temp_storage_bytes)});
    Tensor temp = allocateCubTemporaryStorage(workspace_plan);
    cubDeviceSelectFlagged(half_plan, temp, half_input, half_flags, half_output, half_count, stream);
    cubDeviceSelectFlagged(bf16_plan, temp, bf16_input, bf16_flags, bf16_output, bf16_count, stream);
    stream.synchronize();

    EXPECT_EQ(copyGpuVector<uint32_t>(half_count, stream), (std::vector<uint32_t>{2U}));
    EXPECT_EQ(copyGpuVector<uint32_t>(bf16_count, stream), (std::vector<uint32_t>{2U}));
    const auto half_selected = copyGpuVectorAsFloat<__half>(half_output, stream);
    expectFloatVectorNear(std::vector<float>(half_selected.begin(), half_selected.begin() + 2), {1.0f, 3.0f}, 0.0f);
    const auto bf16_selected = copyGpuVectorAsFloat<__nv_bfloat16>(bf16_output, stream);
    expectFloatVectorNear(std::vector<float>(bf16_selected.begin(), bf16_selected.begin() + 2), {6.0f, 8.0f}, 0.01f);
}

#if THOR_CUB_ENABLE_FP8_TYPES
TEST(CubDevicePrimitives, SelectFlaggedSupportsFp8ValuesWhenEnabled) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    const std::vector<uint8_t> input_bytes = {0x00U, 0x38U, 0x80U, 0x7FU, 0x41U};
    std::vector<__nv_fp8_e4m3> typed_input;
    typed_input.reserve(input_bytes.size());
    for (uint8_t byte : input_bytes) {
        typed_input.push_back(fp8FromBytePattern<__nv_fp8_e4m3>(byte));
    }

    Tensor input = makeGpuVector<__nv_fp8_e4m3>(typed_input, stream);
    Tensor flags = makeGpuVector<uint8_t>({1U, 0U, 1U, 1U, 0U}, stream);
    Tensor output(gpuPlacement, TensorDescriptor(DataType::FP8_E4M3, {5}));
    Tensor num_selected_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {1}));

    const CubDeviceSelectFlaggedPlan select_plan = prepareCubDeviceSelectFlagged(input, flags, output, num_selected_out, 5);
    Tensor temp = allocateCubTemporaryStorage(cubTemporaryStoragePlan(gpuPlacement, select_plan.temp_storage_bytes));
    cubDeviceSelectFlagged(select_plan, temp, input, flags, output, num_selected_out, stream);
    stream.synchronize();

    EXPECT_EQ(copyGpuVector<uint32_t>(num_selected_out, stream), (std::vector<uint32_t>{3U}));
    const std::vector<uint8_t> selected_bytes = fp8BytePatterns(copyGpuVector<__nv_fp8_e4m3>(output, stream));
    EXPECT_EQ(std::vector<uint8_t>(selected_bytes.begin(), selected_bytes.begin() + 3), (std::vector<uint8_t>{0x00U, 0x80U, 0x7FU}));
}
#else
TEST(CubDevicePrimitives, SelectFlaggedRejectsFp8ValuesWhenDisabled) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor input = makeGpuVector<__nv_fp8_e4m3>({__nv_fp8_e4m3(1.0f), __nv_fp8_e4m3(2.0f)}, stream);
    Tensor flags = makeGpuVector<uint8_t>({1U, 0U}, stream);
    Tensor output(gpuPlacement, TensorDescriptor(DataType::FP8_E4M3, {2}));
    Tensor num_selected_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {1}));
    EXPECT_THROW((void)prepareCubDeviceSelectFlagged(input, flags, output, num_selected_out, 2), std::invalid_argument);
}
#endif

TEST(CubDevicePrimitives, SelectFlaggedRejectsUnsupportedDTypesAndMismatches) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor input = makeGpuVector<uint32_t>({1U, 2U, 3U}, stream);
    Tensor flags = makeGpuVector<uint8_t>({1U, 0U, 1U}, stream);
    Tensor output(gpuPlacement, TensorDescriptor(DataType::UINT32, {3}));
    Tensor count(gpuPlacement, TensorDescriptor(DataType::UINT32, {1}));

    Tensor fp_output(gpuPlacement, TensorDescriptor(DataType::FP32, {3}));
    EXPECT_THROW((void)prepareCubDeviceSelectFlagged(input, flags, fp_output, count, 3), std::invalid_argument);

    Tensor bad_flags = makeGpuVector<int32_t>({1, 0, 1}, stream);
    EXPECT_THROW((void)prepareCubDeviceSelectFlagged(input, bad_flags, output, count, 3), std::invalid_argument);

    Tensor bad_count(gpuPlacement, TensorDescriptor(DataType::FP32, {1}));
    EXPECT_THROW((void)prepareCubDeviceSelectFlagged(input, flags, output, bad_count, 3), std::invalid_argument);

    Tensor small_output(gpuPlacement, TensorDescriptor(DataType::UINT32, {2}));
    EXPECT_THROW((void)prepareCubDeviceSelectFlagged(input, flags, small_output, count, 3), std::invalid_argument);

}

TEST(CubDevicePrimitives, SegmentedTopKKeysLargestAndSmallestUseFixedSizeSegments) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor keys_in = makeGpuVector<uint32_t>(
        {1U, 5U, 3U, 9U, 2U, 10U, 4U, 8U, 6U, 7U, 11U, 15U, 12U, 14U, 13U}, stream);
    Tensor largest_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {6}));
    Tensor smallest_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {6}));

    const CubDeviceSegmentedTopKKeysPlan largest_plan =
        prepareCubDeviceSegmentedTopKKeys(keys_in, largest_out, 3, 5, 2, CubTopKOrder::Largest);
    const CubDeviceSegmentedTopKKeysPlan smallest_plan =
        prepareCubDeviceSegmentedTopKKeys(keys_in, smallest_out, 3, 5, 2, CubTopKOrder::Smallest);
    const CubTemporaryStoragePlan workspace_plan = cubMaxTemporaryStoragePlan(
        {cubTemporaryStoragePlan(gpuPlacement, largest_plan.temp_storage_bytes),
         cubTemporaryStoragePlan(gpuPlacement, smallest_plan.temp_storage_bytes)});
    Tensor temp = allocateCubTemporaryStorage(workspace_plan);

    cubDeviceSegmentedTopKKeys(largest_plan, temp, keys_in, largest_out, stream);
    cubDeviceSegmentedTopKKeys(smallest_plan, temp, keys_in, smallest_out, stream);
    stream.synchronize();

    auto largest = copyGpuVector<uint32_t>(largest_out, stream);
    auto smallest = copyGpuVector<uint32_t>(smallest_out, stream);
    for (size_t offset = 0; offset < largest.size(); offset += 2) {
        std::sort(largest.begin() + static_cast<std::ptrdiff_t>(offset),
                  largest.begin() + static_cast<std::ptrdiff_t>(offset + 2),
                  std::greater<uint32_t>());
        std::sort(smallest.begin() + static_cast<std::ptrdiff_t>(offset),
                  smallest.begin() + static_cast<std::ptrdiff_t>(offset + 2));
    }

    EXPECT_EQ(largest, (std::vector<uint32_t>{9U, 5U, 10U, 8U, 15U, 14U}));
    EXPECT_EQ(smallest, (std::vector<uint32_t>{1U, 2U, 4U, 6U, 11U, 12U}));
}

TEST(CubDevicePrimitives, SegmentedTopKPairsCarriesUint32IndexValues) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor keys_in = makeGpuVector<float>(
        {1.0f, 5.0f, 3.0f, 9.0f, 2.0f, 10.0f, 4.0f, 8.0f, 6.0f, 7.0f, 11.0f, 15.0f, 12.0f, 14.0f, 13.0f},
        stream);
    Tensor values_in = makeGpuVector<uint32_t>(
        {100U, 101U, 102U, 103U, 104U, 200U, 201U, 202U, 203U, 204U, 300U, 301U, 302U, 303U, 304U}, stream);
    Tensor keys_out(gpuPlacement, TensorDescriptor(DataType::FP32, {6}));
    Tensor values_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {6}));

    const CubDeviceSegmentedTopKPairsPlan topk_plan =
        prepareCubDeviceSegmentedTopKPairs(keys_in, keys_out, values_in, values_out, 3, 5, 2, CubTopKOrder::Largest);
    Tensor temp = allocateCubTemporaryStorage(cubTemporaryStoragePlan(gpuPlacement, topk_plan.temp_storage_bytes));
    cubDeviceSegmentedTopKPairs(topk_plan, temp, keys_in, keys_out, values_in, values_out, stream);
    stream.synchronize();

    const auto keys = copyGpuVector<float>(keys_out, stream);
    const auto values = copyGpuVector<uint32_t>(values_out, stream);
    std::vector<std::pair<float, uint32_t>> pairs;
    for (size_t i = 0; i < keys.size(); ++i) {
        pairs.emplace_back(keys[i], values[i]);
    }
    for (size_t offset = 0; offset < pairs.size(); offset += 2) {
        std::sort(pairs.begin() + static_cast<std::ptrdiff_t>(offset),
                  pairs.begin() + static_cast<std::ptrdiff_t>(offset + 2),
                  [](const auto& a, const auto& b) { return a.first > b.first; });
    }

    EXPECT_EQ(pairs,
              (std::vector<std::pair<float, uint32_t>>{{9.0f, 103U},
                                                       {5.0f, 101U},
                                                       {10.0f, 200U},
                                                       {8.0f, 202U},
                                                       {15.0f, 301U},
                                                       {14.0f, 303U}}));
}

TEST(CubDevicePrimitives, SegmentedTopKSupportsFp16AndBf16Keys) {
    expectSegmentedTopKKeysFloatingDType<__half>({1.0f, 5.0f, 3.0f, 9.0f, 2.0f, 10.0f, 4.0f, 8.0f, 6.0f, 7.0f},
                                                2,
                                                5,
                                                2,
                                                {9.0f, 5.0f, 10.0f, 8.0f},
                                                CubTopKOrder::Largest,
                                                0.0f);
    expectSegmentedTopKKeysFloatingDType<__nv_bfloat16>(
        {1.0f, 5.0f, 3.0f, 9.0f, 2.0f, 10.0f, 4.0f, 8.0f, 6.0f, 7.0f},
        2,
        5,
        2,
        {9.0f, 5.0f, 10.0f, 8.0f},
        CubTopKOrder::Largest,
        0.01f);
}

TEST(CubDevicePrimitives, SegmentedTopKRejectsUnsupportedDTypesAndMismatches) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor keys_in = makeGpuVector<uint32_t>({1U, 2U, 3U, 4U, 5U}, stream);
    Tensor fp_keys_out(gpuPlacement, TensorDescriptor(DataType::FP32, {2}));
    EXPECT_THROW((void)prepareCubDeviceSegmentedTopKKeys(keys_in, fp_keys_out, 1, 5, 2), std::invalid_argument);

    Tensor fp8_keys_in = makeGpuVector<__nv_fp8_e4m3>({__nv_fp8_e4m3(1.0f), __nv_fp8_e4m3(2.0f), __nv_fp8_e4m3(3.0f)}, stream);
    Tensor fp8_keys_out(gpuPlacement, TensorDescriptor(DataType::FP8_E4M3, {1}));
    EXPECT_THROW((void)prepareCubDeviceSegmentedTopKKeys(fp8_keys_in, fp8_keys_out, 1, 3, 1), std::invalid_argument);

    Tensor keys_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {2}));
    Tensor values_in = makeGpuVector<float>({10.0f, 20.0f, 30.0f, 40.0f, 50.0f}, stream);
    Tensor values_out(gpuPlacement, TensorDescriptor(DataType::FP32, {2}));
    EXPECT_THROW((void)prepareCubDeviceSegmentedTopKPairs(keys_in, keys_out, values_in, values_out, 1, 5, 2),
                 std::invalid_argument);

    Tensor small_keys_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {1}));
    EXPECT_THROW((void)prepareCubDeviceSegmentedTopKKeys(keys_in, small_keys_out, 1, 5, 2), std::invalid_argument);
    EXPECT_THROW((void)prepareCubDeviceSegmentedTopKKeys(keys_in, keys_out, 1, 5, 5), std::invalid_argument);
    EXPECT_THROW((void)prepareCubDeviceSegmentedTopKKeys(keys_in, keys_out, 1, 0, 1), std::invalid_argument);

    Tensor large_keys_in(gpuPlacement, TensorDescriptor(DataType::UINT32, {8193}));
    Tensor large_keys_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {1}));
    EXPECT_THROW((void)prepareCubDeviceSegmentedTopKKeys(large_keys_in, large_keys_out, 1, 8193, 1), std::invalid_argument);
}

TEST(CubDevicePrimitives, SegmentedSortKeysAscendingUsesContiguousOffsets) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor keys_in = makeGpuVector<uint32_t>({8U, 6U, 7U, 5U, 3U, 0U, 9U}, stream);
    Tensor offsets = makeGpuVector<uint32_t>({0U, 3U, 3U, 7U}, stream);
    Tensor keys_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {7}));

    const CubDeviceSegmentedRadixSortKeysPlan sort_plan =
        prepareCubDeviceSegmentedRadixSortKeys(keys_in, keys_out, offsets, 7, 3, CubSortOrder::Ascending);
    Tensor temp = allocateCubTemporaryStorage(cubTemporaryStoragePlan(gpuPlacement, sort_plan.temp_storage_bytes));
    cubDeviceSegmentedRadixSortKeys(sort_plan, temp, keys_in, keys_out, offsets, stream);
    stream.synchronize();

    EXPECT_EQ(copyGpuVector<uint32_t>(keys_out, stream), (std::vector<uint32_t>{6U, 7U, 8U, 0U, 3U, 5U, 9U}));
}

TEST(CubDevicePrimitives, SegmentedSortPairsDescendingCarriesValues) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor keys_in = makeGpuVector<uint32_t>({8U, 6U, 7U, 5U, 3U, 0U, 9U}, stream);
    Tensor values_in = makeGpuVector<uint32_t>({0U, 1U, 2U, 3U, 4U, 5U, 6U}, stream);
    Tensor offsets = makeGpuVector<uint32_t>({0U, 3U, 3U, 7U}, stream);
    Tensor keys_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {7}));
    Tensor values_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {7}));

    const CubDeviceSegmentedRadixSortPairsPlan sort_plan = prepareCubDeviceSegmentedRadixSortPairs(
        keys_in, keys_out, values_in, values_out, offsets, 7, 3, CubSortOrder::Descending);
    Tensor temp = allocateCubTemporaryStorage(cubTemporaryStoragePlan(gpuPlacement, sort_plan.temp_storage_bytes));
    cubDeviceSegmentedRadixSortPairs(sort_plan, temp, keys_in, keys_out, values_in, values_out, offsets, stream);
    stream.synchronize();

    EXPECT_EQ(copyGpuVector<uint32_t>(keys_out, stream), (std::vector<uint32_t>{8U, 7U, 6U, 9U, 5U, 3U, 0U}));
    EXPECT_EQ(copyGpuVector<uint32_t>(values_out, stream), (std::vector<uint32_t>{0U, 2U, 1U, 6U, 3U, 4U, 5U}));
}

TEST(CubDevicePrimitives, SegmentedSortRejectsUnsupportedOffsetDTypes) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor keys_in = makeGpuVector<uint32_t>({2U, 1U}, stream);
    Tensor keys_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {2}));
    Tensor float_offsets = makeGpuVector<float>({0.0f, 2.0f}, stream);
    Tensor signed_offsets = makeGpuVector<int32_t>({0, 2}, stream);

    EXPECT_THROW(
        (void)prepareCubDeviceSegmentedRadixSortKeys(keys_in, keys_out, float_offsets, 2, 1, CubSortOrder::Ascending),
        std::invalid_argument);
    EXPECT_THROW(
        (void)prepareCubDeviceSegmentedRadixSortKeys(keys_in, keys_out, signed_offsets, 2, 1, CubSortOrder::Ascending),
        std::invalid_argument);
}

#if THOR_CUB_ENABLE_64BIT_TYPES
TEST(CubDevicePrimitives, SegmentedSortSupportsUint64OffsetsWhen64BitEnabled) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor keys_in = makeGpuVector<uint32_t>({3U, 1U, 4U, 2U}, stream);
    Tensor offsets = makeGpuVector<uint64_t>({0ULL, 2ULL, 4ULL}, stream);
    Tensor keys_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {4}));

    const CubDeviceSegmentedRadixSortKeysPlan sort_plan =
        prepareCubDeviceSegmentedRadixSortKeys(keys_in, keys_out, offsets, 4, 2, CubSortOrder::Ascending);
    Tensor temp = allocateCubTemporaryStorage(cubTemporaryStoragePlan(gpuPlacement, sort_plan.temp_storage_bytes));
    cubDeviceSegmentedRadixSortKeys(sort_plan, temp, keys_in, keys_out, offsets, stream);
    stream.synchronize();

    EXPECT_EQ(copyGpuVector<uint32_t>(keys_out, stream), (std::vector<uint32_t>{1U, 3U, 2U, 4U}));
}
#else
TEST(CubDevicePrimitives, SegmentedSortRejectsUint64OffsetsWhen64BitDisabled) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor keys_in = makeGpuVector<uint32_t>({3U, 1U}, stream);
    Tensor offsets = makeGpuVector<uint64_t>({0ULL, 2ULL}, stream);
    Tensor keys_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {2}));

    EXPECT_THROW(
        (void)prepareCubDeviceSegmentedRadixSortKeys(keys_in, keys_out, offsets, 2, 1, CubSortOrder::Ascending),
        std::invalid_argument);
}
#endif

TEST(CubDevicePrimitives, DeviceReduceSumMaxAndMinUseSingleOutput) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor input = makeGpuVector<uint32_t>({8U, 1U, 7U, 5U}, stream);
    Tensor sum_output(gpuPlacement, TensorDescriptor(DataType::UINT32, {1}));
    Tensor max_output(gpuPlacement, TensorDescriptor(DataType::UINT32, {1}));
    Tensor min_output(gpuPlacement, TensorDescriptor(DataType::UINT32, {1}));

    const CubDeviceReduceSumPlan sum_plan = prepareCubDeviceReduceSum(input, sum_output, 4);
    const CubDeviceReduceMaxPlan max_plan = prepareCubDeviceReduceMax(input, max_output, 4);
    const CubDeviceReduceMinPlan min_plan = prepareCubDeviceReduceMin(input, min_output, 4);
    Tensor temp = allocateCubTemporaryStorage(cubMaxTemporaryStoragePlan(
        {cubTemporaryStoragePlan(gpuPlacement, sum_plan.temp_storage_bytes),
         cubTemporaryStoragePlan(gpuPlacement, max_plan.temp_storage_bytes),
         cubTemporaryStoragePlan(gpuPlacement, min_plan.temp_storage_bytes)}));

    cubDeviceReduceSum(sum_plan, temp, input, sum_output, stream);
    cubDeviceReduceMax(max_plan, temp, input, max_output, stream);
    cubDeviceReduceMin(min_plan, temp, input, min_output, stream);
    stream.synchronize();

    EXPECT_EQ(copyGpuVector<uint32_t>(sum_output, stream), (std::vector<uint32_t>{21U}));
    EXPECT_EQ(copyGpuVector<uint32_t>(max_output, stream), (std::vector<uint32_t>{8U}));
    EXPECT_EQ(copyGpuVector<uint32_t>(min_output, stream), (std::vector<uint32_t>{1U}));
}

TEST(CubDevicePrimitives, DeviceReduceSupportsFp16AndBf16Values) {
    expectDeviceReduceFloatingDType<__half>({1.0f, -2.0f, 3.0f, 0.5f}, 2.5f, 3.0f, -2.0f, 0.0f);
    expectDeviceReduceFloatingDType<__nv_bfloat16>({1.0f, -2.0f, 3.0f, 0.5f}, 2.5f, 3.0f, -2.0f, 0.01f);
}

TEST(CubDevicePrimitives, DeviceReduceRejectsUnsupportedDTypesAndMismatches) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor signed_input = makeGpuVector<int32_t>({1, 2, 3}, stream);
    Tensor signed_output(gpuPlacement, TensorDescriptor(DataType::INT32, {1}));
    EXPECT_THROW((void)prepareCubDeviceReduceSum(signed_input, signed_output, 3), std::invalid_argument);
    EXPECT_THROW((void)prepareCubDeviceReduceMax(signed_input, signed_output, 3), std::invalid_argument);
    EXPECT_THROW((void)prepareCubDeviceReduceMin(signed_input, signed_output, 3), std::invalid_argument);

    Tensor input = makeGpuVector<uint32_t>({1U, 2U, 3U}, stream);
    Tensor fp_output(gpuPlacement, TensorDescriptor(DataType::FP32, {1}));
    EXPECT_THROW((void)prepareCubDeviceReduceSum(input, fp_output, 3), std::invalid_argument);
    EXPECT_THROW((void)prepareCubDeviceReduceMax(input, fp_output, 3), std::invalid_argument);
    EXPECT_THROW((void)prepareCubDeviceReduceMin(input, fp_output, 3), std::invalid_argument);


    Tensor output(gpuPlacement, TensorDescriptor(DataType::UINT32, {1}));
    EXPECT_THROW((void)prepareCubDeviceReduceSum(input, output, 0), std::invalid_argument);
    EXPECT_THROW((void)prepareCubDeviceReduceMax(input, output, 0), std::invalid_argument);
    EXPECT_THROW((void)prepareCubDeviceReduceMin(input, output, 0), std::invalid_argument);

    Tensor fp8_input = makeGpuVector<__nv_fp8_e4m3>({__nv_fp8_e4m3(1.0f), __nv_fp8_e4m3(2.0f)}, stream);
    Tensor fp8_output(gpuPlacement, TensorDescriptor(DataType::FP8_E4M3, {1}));
    EXPECT_THROW((void)prepareCubDeviceReduceSum(fp8_input, fp8_output, 2), std::invalid_argument);
    EXPECT_THROW((void)prepareCubDeviceReduceMax(fp8_input, fp8_output, 2), std::invalid_argument);
    EXPECT_THROW((void)prepareCubDeviceReduceMin(fp8_input, fp8_output, 2), std::invalid_argument);
}

#if THOR_CUB_ENABLE_64BIT_TYPES
TEST(CubDevicePrimitives, DeviceReduceSupports64BitTypesWhenEnabled) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor uint_input = makeGpuVector<uint64_t>({9ULL, 2ULL, 5ULL}, stream);
    Tensor uint_sum(gpuPlacement, TensorDescriptor(DataType::UINT64, {1}));
    Tensor uint_max(gpuPlacement, TensorDescriptor(DataType::UINT64, {1}));
    Tensor uint_min(gpuPlacement, TensorDescriptor(DataType::UINT64, {1}));

    const CubDeviceReduceSumPlan sum_plan = prepareCubDeviceReduceSum(uint_input, uint_sum, 3);
    const CubDeviceReduceMaxPlan max_plan = prepareCubDeviceReduceMax(uint_input, uint_max, 3);
    const CubDeviceReduceMinPlan min_plan = prepareCubDeviceReduceMin(uint_input, uint_min, 3);
    Tensor temp = allocateCubTemporaryStorage(cubMaxTemporaryStoragePlan(
        {cubTemporaryStoragePlan(gpuPlacement, sum_plan.temp_storage_bytes),
         cubTemporaryStoragePlan(gpuPlacement, max_plan.temp_storage_bytes),
         cubTemporaryStoragePlan(gpuPlacement, min_plan.temp_storage_bytes)}));

    cubDeviceReduceSum(sum_plan, temp, uint_input, uint_sum, stream);
    cubDeviceReduceMax(max_plan, temp, uint_input, uint_max, stream);
    cubDeviceReduceMin(min_plan, temp, uint_input, uint_min, stream);
    stream.synchronize();

    EXPECT_EQ(copyGpuVector<uint64_t>(uint_sum, stream), (std::vector<uint64_t>{16ULL}));
    EXPECT_EQ(copyGpuVector<uint64_t>(uint_max, stream), (std::vector<uint64_t>{9ULL}));
    EXPECT_EQ(copyGpuVector<uint64_t>(uint_min, stream), (std::vector<uint64_t>{2ULL}));
}
#else
TEST(CubDevicePrimitives, DeviceReduceRejects64BitTypesWhenDisabled) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor input = makeGpuVector<uint64_t>({9ULL, 2ULL, 5ULL}, stream);
    Tensor output(gpuPlacement, TensorDescriptor(DataType::UINT64, {1}));

    EXPECT_THROW((void)prepareCubDeviceReduceSum(input, output, 3), std::invalid_argument);
    EXPECT_THROW((void)prepareCubDeviceReduceMax(input, output, 3), std::invalid_argument);
    EXPECT_THROW((void)prepareCubDeviceReduceMin(input, output, 3), std::invalid_argument);
}
#endif


TEST(CubDevicePrimitives, DeviceFindLowerAndUpperBoundUseAscendingRange) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor range = makeGpuVector<uint32_t>({1U, 2U, 2U, 4U, 7U}, stream);
    Tensor values = makeGpuVector<uint32_t>({0U, 2U, 3U, 8U}, stream);
    Tensor lower_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {4}));
    Tensor upper_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {4}));

    const CubDeviceLowerBoundPlan lower_plan = prepareCubDeviceLowerBound(range, values, lower_out, 5, 4);
    const CubDeviceUpperBoundPlan upper_plan = prepareCubDeviceUpperBound(range, values, upper_out, 5, 4);
    Tensor temp = allocateCubTemporaryStorage(cubMaxTemporaryStoragePlan(
        {cubTemporaryStoragePlan(gpuPlacement, lower_plan.temp_storage_bytes),
         cubTemporaryStoragePlan(gpuPlacement, upper_plan.temp_storage_bytes)}));

    cubDeviceLowerBound(lower_plan, temp, range, values, lower_out, stream);
    cubDeviceUpperBound(upper_plan, temp, range, values, upper_out, stream);
    stream.synchronize();

    EXPECT_EQ(copyGpuVector<uint32_t>(lower_out, stream), (std::vector<uint32_t>{0U, 1U, 3U, 5U}));
    EXPECT_EQ(copyGpuVector<uint32_t>(upper_out, stream), (std::vector<uint32_t>{0U, 3U, 3U, 5U}));
}

TEST(CubDevicePrimitives, DeviceFindLowerAndUpperBoundUseDescendingRange) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor range = makeGpuVector<uint32_t>({9U, 7U, 7U, 3U, 1U}, stream);
    Tensor values = makeGpuVector<uint32_t>({10U, 7U, 5U, 0U}, stream);
    Tensor lower_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {4}));
    Tensor upper_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {4}));

    const CubDeviceLowerBoundPlan lower_plan =
        prepareCubDeviceLowerBound(range, values, lower_out, 5, 4, CubSortOrder::Descending);
    const CubDeviceUpperBoundPlan upper_plan =
        prepareCubDeviceUpperBound(range, values, upper_out, 5, 4, CubSortOrder::Descending);
    Tensor temp = allocateCubTemporaryStorage(cubMaxTemporaryStoragePlan(
        {cubTemporaryStoragePlan(gpuPlacement, lower_plan.temp_storage_bytes),
         cubTemporaryStoragePlan(gpuPlacement, upper_plan.temp_storage_bytes)}));

    cubDeviceLowerBound(lower_plan, temp, range, values, lower_out, stream);
    cubDeviceUpperBound(upper_plan, temp, range, values, upper_out, stream);
    stream.synchronize();

    EXPECT_EQ(copyGpuVector<uint32_t>(lower_out, stream), (std::vector<uint32_t>{0U, 1U, 3U, 5U}));
    EXPECT_EQ(copyGpuVector<uint32_t>(upper_out, stream), (std::vector<uint32_t>{0U, 3U, 3U, 5U}));
}

TEST(CubDevicePrimitives, DeviceFindBoundsSupportFp16AndBf16Values) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor half_range = makeGpuVector<__half>({__half(1.0f), __half(2.0f), __half(4.0f)}, stream);
    Tensor half_values = makeGpuVector<__half>({__half(0.5f), __half(3.0f), __half(5.0f)}, stream);
    Tensor half_lower(gpuPlacement, TensorDescriptor(DataType::UINT32, {3}));
    Tensor half_upper(gpuPlacement, TensorDescriptor(DataType::UINT32, {3}));

    Tensor bf16_range = makeGpuVector<__nv_bfloat16>(
        {__nv_bfloat16(1.0f), __nv_bfloat16(2.0f), __nv_bfloat16(4.0f)}, stream);
    Tensor bf16_values = makeGpuVector<__nv_bfloat16>(
        {__nv_bfloat16(0.5f), __nv_bfloat16(3.0f), __nv_bfloat16(5.0f)}, stream);
    Tensor bf16_lower(gpuPlacement, TensorDescriptor(DataType::UINT32, {3}));
    Tensor bf16_upper(gpuPlacement, TensorDescriptor(DataType::UINT32, {3}));

    const CubDeviceLowerBoundPlan half_lower_plan = prepareCubDeviceLowerBound(half_range, half_values, half_lower, 3, 3);
    const CubDeviceUpperBoundPlan half_upper_plan = prepareCubDeviceUpperBound(half_range, half_values, half_upper, 3, 3);
    const CubDeviceLowerBoundPlan bf16_lower_plan = prepareCubDeviceLowerBound(bf16_range, bf16_values, bf16_lower, 3, 3);
    const CubDeviceUpperBoundPlan bf16_upper_plan = prepareCubDeviceUpperBound(bf16_range, bf16_values, bf16_upper, 3, 3);

    Tensor temp = allocateCubTemporaryStorage(cubMaxTemporaryStoragePlan(
        {cubTemporaryStoragePlan(gpuPlacement, half_lower_plan.temp_storage_bytes),
         cubTemporaryStoragePlan(gpuPlacement, half_upper_plan.temp_storage_bytes),
         cubTemporaryStoragePlan(gpuPlacement, bf16_lower_plan.temp_storage_bytes),
         cubTemporaryStoragePlan(gpuPlacement, bf16_upper_plan.temp_storage_bytes)}));

    cubDeviceLowerBound(half_lower_plan, temp, half_range, half_values, half_lower, stream);
    cubDeviceUpperBound(half_upper_plan, temp, half_range, half_values, half_upper, stream);
    cubDeviceLowerBound(bf16_lower_plan, temp, bf16_range, bf16_values, bf16_lower, stream);
    cubDeviceUpperBound(bf16_upper_plan, temp, bf16_range, bf16_values, bf16_upper, stream);
    stream.synchronize();

    EXPECT_EQ(copyGpuVector<uint32_t>(half_lower, stream), (std::vector<uint32_t>{0U, 2U, 3U}));
    EXPECT_EQ(copyGpuVector<uint32_t>(half_upper, stream), (std::vector<uint32_t>{0U, 2U, 3U}));
    EXPECT_EQ(copyGpuVector<uint32_t>(bf16_lower, stream), (std::vector<uint32_t>{0U, 2U, 3U}));
    EXPECT_EQ(copyGpuVector<uint32_t>(bf16_upper, stream), (std::vector<uint32_t>{0U, 2U, 3U}));
}

TEST(CubDevicePrimitives, DeviceFindIfFlaggedFindsFirstTrueFlagAndNoMatchReturnsNumItems) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor flags = makeGpuVector<bool>({false, false, true, true}, stream);
    Tensor index_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {1}));
    const CubDeviceFindIfFlaggedPlan plan = prepareCubDeviceFindIfFlagged(flags, index_out, 4);
    Tensor temp = allocateCubTemporaryStorage(cubTemporaryStoragePlan(gpuPlacement, plan.temp_storage_bytes));

    cubDeviceFindIfFlagged(plan, temp, flags, index_out, stream);
    stream.synchronize();
    EXPECT_EQ(copyGpuVector<uint32_t>(index_out, stream), (std::vector<uint32_t>{2U}));

    Tensor no_match_flags = makeGpuVector<uint8_t>({0U, 0U, 0U, 0U}, stream);
    Tensor no_match_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {1}));
    const CubDeviceFindIfFlaggedPlan no_match_plan = prepareCubDeviceFindIfFlagged(no_match_flags, no_match_out, 4);
    Tensor no_match_temp = allocateCubTemporaryStorage(cubTemporaryStoragePlan(gpuPlacement, no_match_plan.temp_storage_bytes));
    cubDeviceFindIfFlagged(no_match_plan, no_match_temp, no_match_flags, no_match_out, stream);
    stream.synchronize();
    EXPECT_EQ(copyGpuVector<uint32_t>(no_match_out, stream), (std::vector<uint32_t>{4U}));
}

TEST(CubDevicePrimitives, DeviceFindIfFlaggedWritesZeroForZeroItems) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor flags = makeGpuVector<bool>({true}, stream);
    Tensor index_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {1}));
    const CubDeviceFindIfFlaggedPlan plan = prepareCubDeviceFindIfFlagged(flags, index_out, 0);
    Tensor temp = allocateCubTemporaryStorage(cubTemporaryStoragePlan(gpuPlacement, plan.temp_storage_bytes));

    cubDeviceFindIfFlagged(plan, temp, flags, index_out, stream);
    stream.synchronize();
    EXPECT_EQ(copyGpuVector<uint32_t>(index_out, stream), (std::vector<uint32_t>{0U}));
}

TEST(CubDevicePrimitives, DeviceFindRejectsUnsupportedDTypesAndMismatches) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor range = makeGpuVector<uint32_t>({1U, 2U, 3U}, stream);
    Tensor values = makeGpuVector<uint32_t>({2U, 3U}, stream);
    Tensor output(gpuPlacement, TensorDescriptor(DataType::UINT32, {2}));

    Tensor fp_values = makeGpuVector<float>({2.0f, 3.0f}, stream);
    EXPECT_THROW((void)prepareCubDeviceLowerBound(range, fp_values, output, 3, 2), std::invalid_argument);
    EXPECT_THROW((void)prepareCubDeviceUpperBound(range, fp_values, output, 3, 2), std::invalid_argument);

    Tensor bad_output(gpuPlacement, TensorDescriptor(DataType::FP32, {2}));
    EXPECT_THROW((void)prepareCubDeviceLowerBound(range, values, bad_output, 3, 2), std::invalid_argument);
    EXPECT_THROW((void)prepareCubDeviceUpperBound(range, values, bad_output, 3, 2), std::invalid_argument);

    Tensor small_output(gpuPlacement, TensorDescriptor(DataType::UINT32, {1}));
    EXPECT_THROW((void)prepareCubDeviceLowerBound(range, values, small_output, 3, 2), std::invalid_argument);
    EXPECT_THROW((void)prepareCubDeviceUpperBound(range, values, small_output, 3, 2), std::invalid_argument);

    Tensor fp8_range = makeGpuVector<__nv_fp8_e4m3>({__nv_fp8_e4m3(1.0f), __nv_fp8_e4m3(2.0f)}, stream);
    Tensor fp8_values = makeGpuVector<__nv_fp8_e4m3>({__nv_fp8_e4m3(1.0f)}, stream);
    Tensor fp8_output(gpuPlacement, TensorDescriptor(DataType::UINT32, {1}));
    EXPECT_THROW((void)prepareCubDeviceLowerBound(fp8_range, fp8_values, fp8_output, 2, 1), std::invalid_argument);
    EXPECT_THROW((void)prepareCubDeviceUpperBound(fp8_range, fp8_values, fp8_output, 2, 1), std::invalid_argument);

    Tensor bad_flags = makeGpuVector<int32_t>({0, 1, 0}, stream);
    Tensor index_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {1}));
    EXPECT_THROW((void)prepareCubDeviceFindIfFlagged(bad_flags, index_out, 3), std::invalid_argument);

    Tensor bad_index_out(gpuPlacement, TensorDescriptor(DataType::FP32, {1}));
    Tensor bool_flags = makeGpuVector<bool>({false, true}, stream);
    EXPECT_THROW((void)prepareCubDeviceFindIfFlagged(bool_flags, bad_index_out, 2), std::invalid_argument);
}

#if THOR_CUB_ENABLE_64BIT_TYPES
TEST(CubDevicePrimitives, DeviceFindBoundsSupport64BitTypesWhenEnabled) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor range = makeGpuVector<uint64_t>({1ULL, 4ULL, 9ULL}, stream);
    Tensor values = makeGpuVector<uint64_t>({0ULL, 4ULL, 10ULL}, stream);
    Tensor lower_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {3}));
    Tensor upper_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {3}));

    const CubDeviceLowerBoundPlan lower_plan = prepareCubDeviceLowerBound(range, values, lower_out, 3, 3);
    const CubDeviceUpperBoundPlan upper_plan = prepareCubDeviceUpperBound(range, values, upper_out, 3, 3);
    Tensor temp = allocateCubTemporaryStorage(cubMaxTemporaryStoragePlan(
        {cubTemporaryStoragePlan(gpuPlacement, lower_plan.temp_storage_bytes),
         cubTemporaryStoragePlan(gpuPlacement, upper_plan.temp_storage_bytes)}));

    cubDeviceLowerBound(lower_plan, temp, range, values, lower_out, stream);
    cubDeviceUpperBound(upper_plan, temp, range, values, upper_out, stream);
    stream.synchronize();

    EXPECT_EQ(copyGpuVector<uint32_t>(lower_out, stream), (std::vector<uint32_t>{0U, 1U, 3U}));
    EXPECT_EQ(copyGpuVector<uint32_t>(upper_out, stream), (std::vector<uint32_t>{0U, 2U, 3U}));
}
#else
TEST(CubDevicePrimitives, DeviceFindBoundsReject64BitTypesWhenDisabled) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor range = makeGpuVector<uint64_t>({1ULL, 4ULL, 9ULL}, stream);
    Tensor values = makeGpuVector<uint64_t>({0ULL, 4ULL, 10ULL}, stream);
    Tensor output(gpuPlacement, TensorDescriptor(DataType::UINT32, {3}));

    EXPECT_THROW((void)prepareCubDeviceLowerBound(range, values, output, 3, 3), std::invalid_argument);
    EXPECT_THROW((void)prepareCubDeviceUpperBound(range, values, output, 3, 3), std::invalid_argument);
}
#endif

TEST(CubDevicePrimitives, SegmentedExclusiveScanUsesContiguousOffsets) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor input = makeGpuVector<uint32_t>({8U, 6U, 7U, 5U, 3U, 2U, 9U}, stream);
    Tensor offsets = makeGpuVector<uint32_t>({0U, 2U, 5U, 7U}, stream);
    Tensor output(gpuPlacement, TensorDescriptor(DataType::UINT32, {7}));

    const CubDeviceSegmentedExclusiveSumPlan scan_plan = prepareCubDeviceSegmentedExclusiveSum(input, output, offsets, 7, 3);
    Tensor temp = allocateCubTemporaryStorage(cubTemporaryStoragePlan(gpuPlacement, scan_plan.temp_storage_bytes));
    cubDeviceSegmentedExclusiveSum(scan_plan, temp, input, output, offsets, stream);
    stream.synchronize();

    EXPECT_EQ(copyGpuVector<uint32_t>(output, stream), (std::vector<uint32_t>{0U, 8U, 0U, 7U, 12U, 0U, 2U}));
}

TEST(CubDevicePrimitives, SegmentedExclusiveScanAllowsEmptySegments) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor input = makeGpuVector<uint32_t>({8U, 6U, 7U, 5U, 3U, 2U, 9U}, stream);
    Tensor offsets = makeGpuVector<uint32_t>({0U, 3U, 3U, 7U}, stream);
    Tensor output(gpuPlacement, TensorDescriptor(DataType::UINT32, {7}));

    const CubDeviceSegmentedExclusiveSumPlan scan_plan = prepareCubDeviceSegmentedExclusiveSum(input, output, offsets, 7, 3);
    Tensor temp = allocateCubTemporaryStorage(cubTemporaryStoragePlan(gpuPlacement, scan_plan.temp_storage_bytes));
    cubDeviceSegmentedExclusiveSum(scan_plan, temp, input, output, offsets, stream);
    stream.synchronize();

    EXPECT_EQ(copyGpuVector<uint32_t>(output, stream), (std::vector<uint32_t>{0U, 8U, 14U, 0U, 5U, 8U, 10U}));
}

TEST(CubDevicePrimitives, SegmentedExclusiveScanSupportsFp16AndBf16Values) {
    expectSegmentedExclusiveScanFloatingDType<__half>(
        {1.0f, 2.0f, -0.5f, 3.0f, 4.0f}, {0U, 3U, 5U}, {0.0f, 1.0f, 3.0f, 0.0f, 3.0f}, 0.0f);
    expectSegmentedExclusiveScanFloatingDType<__nv_bfloat16>(
        {1.0f, 2.0f, -0.5f, 3.0f, 4.0f}, {0U, 3U, 5U}, {0.0f, 1.0f, 3.0f, 0.0f, 3.0f}, 0.01f);
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
    Tensor temp = allocateCubTemporaryStorage(cubMaxTemporaryStoragePlan(
        {cubTemporaryStoragePlan(gpuPlacement, inclusive_plan.temp_storage_bytes),
         cubTemporaryStoragePlan(gpuPlacement, exclusive_plan.temp_storage_bytes)}));

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
    Tensor temp = allocateCubTemporaryStorage(cubMaxTemporaryStoragePlan(
        {cubTemporaryStoragePlan(gpuPlacement, inclusive_plan.temp_storage_bytes),
         cubTemporaryStoragePlan(gpuPlacement, exclusive_plan.temp_storage_bytes)}));

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
    Tensor temp = allocateCubTemporaryStorage(cubMaxTemporaryStoragePlan(
        {cubTemporaryStoragePlan(gpuPlacement, min_plan.temp_storage_bytes),
         cubTemporaryStoragePlan(gpuPlacement, max_plan.temp_storage_bytes)}));

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
    Tensor temp = allocateCubTemporaryStorage(cubTemporaryStoragePlan(gpuPlacement, max_plan.temp_storage_bytes));

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
    Tensor temp = allocateCubTemporaryStorage(cubMaxTemporaryStoragePlan(
        {cubTemporaryStoragePlan(gpuPlacement, min_plan.temp_storage_bytes),
         cubTemporaryStoragePlan(gpuPlacement, max_plan.temp_storage_bytes)}));

    cubDeviceSegmentedArgScan(min_plan, temp, input, min_out, offsets, stream);
    cubDeviceSegmentedArgScan(max_plan, temp, input, max_out, offsets, stream);
    stream.synchronize();

    EXPECT_EQ(copyGpuVector<uint32_t>(min_out, stream), (std::vector<uint32_t>{0U, 1U, 2U, 3U, 4U, 5U, 5U}));
    EXPECT_EQ(copyGpuVector<uint32_t>(max_out, stream), (std::vector<uint32_t>{0U, 0U, 2U, 2U, 2U, 5U, 6U}));
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
    Tensor temp = allocateCubTemporaryStorage(cubMaxTemporaryStoragePlan(
        {cubTemporaryStoragePlan(gpuPlacement, min_plan.temp_storage_bytes),
         cubTemporaryStoragePlan(gpuPlacement, max_plan.temp_storage_bytes),
         cubTemporaryStoragePlan(gpuPlacement, product_plan.temp_storage_bytes)}));

    cubDeviceSegmentedScan(min_plan, temp, input, min_out, offsets, stream);
    cubDeviceSegmentedScan(max_plan, temp, input, max_out, offsets, stream);
    cubDeviceSegmentedScan(product_plan, temp, input, product_out, offsets, stream);
    stream.synchronize();

    EXPECT_EQ(copyGpuVector<uint32_t>(min_out, stream), (std::vector<uint32_t>{8U, 6U, 7U, 5U, 3U, 2U, 2U}));
    EXPECT_EQ(copyGpuVector<uint32_t>(max_out, stream), (std::vector<uint32_t>{8U, 8U, 7U, 7U, 7U, 2U, 9U}));
    EXPECT_EQ(copyGpuVector<uint32_t>(product_out, stream), (std::vector<uint32_t>{1U, 8U, 1U, 7U, 35U, 1U, 2U}));
}

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
    Tensor temp = allocateCubTemporaryStorage(cubMaxTemporaryStoragePlan(
        {cubTemporaryStoragePlan(gpuPlacement, inclusive_plan.temp_storage_bytes),
         cubTemporaryStoragePlan(gpuPlacement, exclusive_plan.temp_storage_bytes)}));

    cubDeviceSegmentedScan(inclusive_plan, temp, input, inclusive_out, offsets, stream);
    cubDeviceSegmentedScan(exclusive_plan, temp, input, exclusive_out, offsets, stream);
    stream.synchronize();

    EXPECT_EQ(copyGpuVector<uint32_t>(inclusive_out, stream), (std::vector<uint32_t>{6U, 5U, 3U, 22U, 18U, 13U, 7U}));
    EXPECT_EQ(copyGpuVector<uint32_t>(exclusive_out, stream), (std::vector<uint32_t>{5U, 3U, 0U, 18U, 13U, 7U, 0U}));
}

TEST(CubDevicePrimitives, SegmentedReduceSumAndMaxUseContiguousOffsets) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor input = makeGpuVector<uint32_t>({8U, 6U, 7U, 5U, 3U, 2U, 9U}, stream);
    Tensor offsets = makeGpuVector<uint32_t>({0U, 2U, 5U, 7U}, stream);
    Tensor sum_output(gpuPlacement, TensorDescriptor(DataType::UINT32, {3}));
    Tensor max_output(gpuPlacement, TensorDescriptor(DataType::UINT32, {3}));

    const CubDeviceSegmentedReduceSumPlan sum_plan = prepareCubDeviceSegmentedReduceSum(input, sum_output, offsets, 7, 3);
    const CubDeviceSegmentedReduceMaxPlan max_plan = prepareCubDeviceSegmentedReduceMax(input, max_output, offsets, 7, 3);
    const CubTemporaryStoragePlan workspace_plan = cubMaxTemporaryStoragePlan(
        {cubTemporaryStoragePlan(gpuPlacement, sum_plan.temp_storage_bytes),
         cubTemporaryStoragePlan(gpuPlacement, max_plan.temp_storage_bytes)});
    Tensor temp = allocateCubTemporaryStorage(workspace_plan);

    cubDeviceSegmentedReduceSum(sum_plan, temp, input, sum_output, offsets, stream);
    cubDeviceSegmentedReduceMax(max_plan, temp, input, max_output, offsets, stream);
    stream.synchronize();

    EXPECT_EQ(copyGpuVector<uint32_t>(sum_output, stream), (std::vector<uint32_t>{14U, 15U, 11U}));
    EXPECT_EQ(copyGpuVector<uint32_t>(max_output, stream), (std::vector<uint32_t>{8U, 7U, 9U}));
}

TEST(CubDevicePrimitives, SegmentedReduceAllowsEmptySegments) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor input = makeGpuVector<uint32_t>({8U, 6U, 7U, 5U, 3U, 2U, 9U}, stream);
    Tensor offsets = makeGpuVector<uint32_t>({0U, 3U, 3U, 7U}, stream);
    Tensor sum_output(gpuPlacement, TensorDescriptor(DataType::UINT32, {3}));
    Tensor max_output(gpuPlacement, TensorDescriptor(DataType::UINT32, {3}));

    const CubDeviceSegmentedReduceSumPlan sum_plan = prepareCubDeviceSegmentedReduceSum(input, sum_output, offsets, 7, 3);
    const CubDeviceSegmentedReduceMaxPlan max_plan = prepareCubDeviceSegmentedReduceMax(input, max_output, offsets, 7, 3);
    const CubTemporaryStoragePlan workspace_plan = cubMaxTemporaryStoragePlan(
        {cubTemporaryStoragePlan(gpuPlacement, sum_plan.temp_storage_bytes),
         cubTemporaryStoragePlan(gpuPlacement, max_plan.temp_storage_bytes)});
    Tensor temp = allocateCubTemporaryStorage(workspace_plan);

    cubDeviceSegmentedReduceSum(sum_plan, temp, input, sum_output, offsets, stream);
    cubDeviceSegmentedReduceMax(max_plan, temp, input, max_output, offsets, stream);
    stream.synchronize();

    EXPECT_EQ(copyGpuVector<uint32_t>(sum_output, stream), (std::vector<uint32_t>{21U, 0U, 19U}));
    EXPECT_EQ(copyGpuVector<uint32_t>(max_output, stream), (std::vector<uint32_t>{8U, 0U, 9U}));
}

TEST(CubDevicePrimitives, SegmentedReduceSupportsFp16AndBf16Values) {
    expectSegmentedReduceFloatingDType<__half>(
        {1.0f, 2.0f, -0.5f, 3.0f, 4.0f}, {0U, 3U, 5U}, {2.5f, 7.0f}, {2.0f, 4.0f}, 0.0f);
    expectSegmentedReduceFloatingDType<__nv_bfloat16>(
        {1.0f, 2.0f, -0.5f, 3.0f, 4.0f}, {0U, 3U, 5U}, {2.5f, 7.0f}, {2.0f, 4.0f}, 0.01f);
}

TEST(CubDevicePrimitives, SegmentedReduceRejectsUnsupportedDTypesAndMismatches) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor signed_input = makeGpuVector<int32_t>({1, 2, 3}, stream);
    Tensor signed_output(gpuPlacement, TensorDescriptor(DataType::INT32, {1}));
    Tensor offsets = makeGpuVector<uint32_t>({0U, 3U}, stream);
    EXPECT_THROW((void)prepareCubDeviceSegmentedReduceSum(signed_input, signed_output, offsets, 3, 1), std::invalid_argument);
    EXPECT_THROW((void)prepareCubDeviceSegmentedReduceMax(signed_input, signed_output, offsets, 3, 1), std::invalid_argument);

    Tensor input = makeGpuVector<uint32_t>({1U, 2U, 3U}, stream);
    Tensor fp_output(gpuPlacement, TensorDescriptor(DataType::FP32, {1}));
    EXPECT_THROW((void)prepareCubDeviceSegmentedReduceSum(input, fp_output, offsets, 3, 1), std::invalid_argument);
    EXPECT_THROW((void)prepareCubDeviceSegmentedReduceMax(input, fp_output, offsets, 3, 1), std::invalid_argument);

    Tensor signed_offsets = makeGpuVector<int32_t>({0, 3}, stream);
    Tensor output(gpuPlacement, TensorDescriptor(DataType::UINT32, {1}));
    EXPECT_THROW((void)prepareCubDeviceSegmentedReduceSum(input, output, signed_offsets, 3, 1), std::invalid_argument);
    EXPECT_THROW((void)prepareCubDeviceSegmentedReduceMax(input, output, signed_offsets, 3, 1), std::invalid_argument);

    Tensor too_small_output(gpuPlacement, TensorDescriptor(DataType::UINT32, {1}));
    Tensor two_segment_offsets = makeGpuVector<uint32_t>({0U, 1U, 3U}, stream);
    EXPECT_THROW((void)prepareCubDeviceSegmentedReduceSum(input, too_small_output, two_segment_offsets, 3, 2),
                 std::invalid_argument);
    EXPECT_THROW((void)prepareCubDeviceSegmentedReduceMax(input, too_small_output, two_segment_offsets, 3, 2),
                 std::invalid_argument);

    Tensor fp8_input = makeGpuVector<__nv_fp8_e4m3>({__nv_fp8_e4m3(1.0f), __nv_fp8_e4m3(2.0f)}, stream);
    Tensor fp8_output(gpuPlacement, TensorDescriptor(DataType::FP8_E4M3, {1}));
    Tensor fp8_offsets = makeGpuVector<uint32_t>({0U, 2U}, stream);
    EXPECT_THROW((void)prepareCubDeviceSegmentedReduceSum(fp8_input, fp8_output, fp8_offsets, 2, 1), std::invalid_argument);
    EXPECT_THROW((void)prepareCubDeviceSegmentedReduceMax(fp8_input, fp8_output, fp8_offsets, 2, 1), std::invalid_argument);
}

TEST(CubDevicePrimitives, SegmentedExclusiveScanRejectsUnsupportedDTypesAndMismatches) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor signed_input = makeGpuVector<int32_t>({1, 2, 3}, stream);
    Tensor signed_output(gpuPlacement, TensorDescriptor(DataType::INT32, {3}));
    Tensor offsets = makeGpuVector<uint32_t>({0U, 3U}, stream);
    EXPECT_THROW((void)prepareCubDeviceSegmentedExclusiveSum(signed_input, signed_output, offsets, 3, 1), std::invalid_argument);

    Tensor input = makeGpuVector<uint32_t>({1U, 2U, 3U}, stream);
    Tensor fp_output(gpuPlacement, TensorDescriptor(DataType::FP32, {3}));
    EXPECT_THROW((void)prepareCubDeviceSegmentedExclusiveSum(input, fp_output, offsets, 3, 1), std::invalid_argument);

    Tensor signed_offsets = makeGpuVector<int32_t>({0, 3}, stream);
    Tensor output(gpuPlacement, TensorDescriptor(DataType::UINT32, {3}));
    EXPECT_THROW((void)prepareCubDeviceSegmentedExclusiveSum(input, output, signed_offsets, 3, 1), std::invalid_argument);

    Tensor fp8_input = makeGpuVector<__nv_fp8_e4m3>({__nv_fp8_e4m3(1.0f), __nv_fp8_e4m3(2.0f)}, stream);
    Tensor fp8_output(gpuPlacement, TensorDescriptor(DataType::FP8_E4M3, {2}));
    Tensor fp8_offsets = makeGpuVector<uint32_t>({0U, 2U}, stream);
    EXPECT_THROW((void)prepareCubDeviceSegmentedExclusiveSum(fp8_input, fp8_output, fp8_offsets, 2, 1), std::invalid_argument);
}

TEST(CubDevicePrimitives, RunLengthEncodeUint32RowsAndExclusiveScanCounts) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor input = makeGpuVector<uint32_t>({2U, 2U, 2U, 5U, 5U, 9U}, stream);
    Tensor unique_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {6}));
    Tensor counts_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {6}));
    Tensor num_runs_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {1}));

    const CubDeviceRunLengthEncodePlan rle_plan = prepareCubDeviceRunLengthEncode(input, unique_out, counts_out, num_runs_out, 6);
    Tensor rle_temp = allocateCubTemporaryStorage(cubTemporaryStoragePlan(gpuPlacement, rle_plan.temp_storage_bytes));
    cubDeviceRunLengthEncode(rle_plan, rle_temp, input, unique_out, counts_out, num_runs_out, stream);
    stream.synchronize();

    EXPECT_EQ(copyGpuVector<uint32_t>(num_runs_out, stream), (std::vector<uint32_t>{3U}));

    const auto unique = copyGpuVector<uint32_t>(unique_out, stream);
    const auto counts = copyGpuVector<uint32_t>(counts_out, stream);
    EXPECT_EQ(std::vector<uint32_t>(unique.begin(), unique.begin() + 3), (std::vector<uint32_t>{2U, 5U, 9U}));
    EXPECT_EQ(std::vector<uint32_t>(counts.begin(), counts.begin() + 3), (std::vector<uint32_t>{3U, 2U, 1U}));

    Tensor offsets_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {6}));
    const CubDeviceExclusiveSumPlan scan_plan = prepareCubDeviceExclusiveSum(counts_out, offsets_out, 3);
    Tensor scan_temp = allocateCubTemporaryStorage(cubTemporaryStoragePlan(gpuPlacement, scan_plan.temp_storage_bytes));
    cubDeviceExclusiveSum(scan_plan, scan_temp, counts_out, offsets_out, stream);
    stream.synchronize();

    const auto offsets = copyGpuVector<uint32_t>(offsets_out, stream);
    EXPECT_EQ(std::vector<uint32_t>(offsets.begin(), offsets.begin() + 3), (std::vector<uint32_t>{0U, 3U, 5U}));
}


TEST(CubDevicePrimitives, RunLengthEncodeSupportsFp16AndBf16Values) {
    expectRunLengthEncodeFloatingDType<__half>({1.0f, 1.0f, -2.0f, 4.0f, 4.0f}, {1.0f, -2.0f, 4.0f}, {2U, 1U, 2U});
    expectRunLengthEncodeFloatingDType<__nv_bfloat16>({1.0f, 1.0f, -2.0f, 4.0f, 4.0f}, {1.0f, -2.0f, 4.0f}, {2U, 1U, 2U});
}

#if THOR_CUB_ENABLE_FP8_TYPES
TEST(CubDevicePrimitives, RunLengthEncodeSupportsFp8ValuesWhenEnabled) {
    expectRunLengthEncodeFloatingDType<__nv_fp8_e4m3>({1.0f, 1.0f, -2.0f, 4.0f, 4.0f}, {1.0f, -2.0f, 4.0f}, {2U, 1U, 2U});
    expectRunLengthEncodeFloatingDType<__nv_fp8_e5m2>({1.0f, 1.0f, -2.0f, 4.0f, 4.0f}, {1.0f, -2.0f, 4.0f}, {2U, 1U, 2U});
}

TEST(CubDevicePrimitives, RunLengthEncodeFp8UsesBitwiseEqualityWhenEnabled) {
    expectRunLengthEncodeFp8BitwiseDType<__nv_fp8_e4m3>();
    expectRunLengthEncodeFp8BitwiseDType<__nv_fp8_e5m2>();
}
#else
TEST(CubDevicePrimitives, RunLengthEncodeRejectsFp8ValuesWhenDisabled) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor input = makeGpuVector<__nv_fp8_e4m3>({__nv_fp8_e4m3(1.0f), __nv_fp8_e4m3(1.0f)}, stream);
    Tensor unique_out(gpuPlacement, TensorDescriptor(DataType::FP8_E4M3, {2}));
    Tensor counts_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {2}));
    Tensor num_runs_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {1}));
    EXPECT_THROW((void)prepareCubDeviceRunLengthEncode(input, unique_out, counts_out, num_runs_out, 2), std::invalid_argument);
}
#endif

#if THOR_CUB_ENABLE_64BIT_TYPES
TEST(CubDevicePrimitives, RunLengthEncodeAndExclusiveScanSupport64BitTypesWhenEnabled) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor input = makeGpuVector<uint64_t>({7ULL, 7ULL, 2ULL}, stream);
    Tensor unique_out(gpuPlacement, TensorDescriptor(DataType::UINT64, {3}));
    Tensor counts_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {3}));
    Tensor num_runs_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {1}));
    const CubDeviceRunLengthEncodePlan rle_plan = prepareCubDeviceRunLengthEncode(input, unique_out, counts_out, num_runs_out, 3);
    Tensor temp = allocateCubTemporaryStorage(cubTemporaryStoragePlan(gpuPlacement, rle_plan.temp_storage_bytes));
    cubDeviceRunLengthEncode(rle_plan, temp, input, unique_out, counts_out, num_runs_out, stream);
    stream.synchronize();
    const auto unique = copyGpuVector<uint64_t>(unique_out, stream);
    EXPECT_EQ(std::vector<uint64_t>(unique.begin(), unique.begin() + 2), (std::vector<uint64_t>{7ULL, 2ULL}));

    Tensor scan_input = makeGpuVector<uint64_t>({1ULL, 2ULL, 3ULL}, stream);
    Tensor scan_output(gpuPlacement, TensorDescriptor(DataType::UINT64, {3}));
    const CubDeviceExclusiveSumPlan scan_plan = prepareCubDeviceExclusiveSum(scan_input, scan_output, 3);
    Tensor scan_temp = allocateCubTemporaryStorage(cubTemporaryStoragePlan(gpuPlacement, scan_plan.temp_storage_bytes));
    cubDeviceExclusiveSum(scan_plan, scan_temp, scan_input, scan_output, stream);
    stream.synchronize();
    EXPECT_EQ(copyGpuVector<uint64_t>(scan_output, stream), (std::vector<uint64_t>{0ULL, 1ULL, 3ULL}));

    Tensor segmented_scan_input = makeGpuVector<uint64_t>({1ULL, 2ULL, 3ULL, 4ULL}, stream);
    Tensor segmented_scan_offsets = makeGpuVector<uint64_t>({0ULL, 2ULL, 4ULL}, stream);
    Tensor segmented_scan_output(gpuPlacement, TensorDescriptor(DataType::UINT64, {4}));
    const CubDeviceSegmentedExclusiveSumPlan segmented_scan_plan =
        prepareCubDeviceSegmentedExclusiveSum(segmented_scan_input, segmented_scan_output, segmented_scan_offsets, 4, 2);
    Tensor segmented_scan_temp = allocateCubTemporaryStorage(cubTemporaryStoragePlan(gpuPlacement, segmented_scan_plan.temp_storage_bytes));
    cubDeviceSegmentedExclusiveSum(segmented_scan_plan, segmented_scan_temp, segmented_scan_input, segmented_scan_output, segmented_scan_offsets, stream);
    stream.synchronize();
    EXPECT_EQ(copyGpuVector<uint64_t>(segmented_scan_output, stream), (std::vector<uint64_t>{0ULL, 1ULL, 0ULL, 3ULL}));

    Tensor segmented_reduce_input = makeGpuVector<uint64_t>({7ULL, 2ULL, 3ULL, 4ULL}, stream);
    Tensor segmented_reduce_offsets = makeGpuVector<uint64_t>({0ULL, 2ULL, 4ULL}, stream);
    Tensor segmented_reduce_sum_output(gpuPlacement, TensorDescriptor(DataType::UINT64, {2}));
    Tensor segmented_reduce_max_output(gpuPlacement, TensorDescriptor(DataType::UINT64, {2}));
    const CubDeviceSegmentedReduceSumPlan segmented_reduce_sum_plan =
        prepareCubDeviceSegmentedReduceSum(segmented_reduce_input, segmented_reduce_sum_output, segmented_reduce_offsets, 4, 2);
    const CubDeviceSegmentedReduceMaxPlan segmented_reduce_max_plan =
        prepareCubDeviceSegmentedReduceMax(segmented_reduce_input, segmented_reduce_max_output, segmented_reduce_offsets, 4, 2);
    const CubTemporaryStoragePlan segmented_reduce_workspace_plan = cubMaxTemporaryStoragePlan(
        {cubTemporaryStoragePlan(gpuPlacement, segmented_reduce_sum_plan.temp_storage_bytes),
         cubTemporaryStoragePlan(gpuPlacement, segmented_reduce_max_plan.temp_storage_bytes)});
    Tensor segmented_reduce_temp = allocateCubTemporaryStorage(segmented_reduce_workspace_plan);
    cubDeviceSegmentedReduceSum(
        segmented_reduce_sum_plan, segmented_reduce_temp, segmented_reduce_input, segmented_reduce_sum_output, segmented_reduce_offsets, stream);
    cubDeviceSegmentedReduceMax(
        segmented_reduce_max_plan, segmented_reduce_temp, segmented_reduce_input, segmented_reduce_max_output, segmented_reduce_offsets, stream);
    stream.synchronize();
    EXPECT_EQ(copyGpuVector<uint64_t>(segmented_reduce_sum_output, stream), (std::vector<uint64_t>{9ULL, 7ULL}));
    EXPECT_EQ(copyGpuVector<uint64_t>(segmented_reduce_max_output, stream), (std::vector<uint64_t>{7ULL, 4ULL}));

    Tensor topk_keys_in = makeGpuVector<uint64_t>({7ULL, 2ULL, 9ULL, 4ULL}, stream);
    Tensor topk_keys_out(gpuPlacement, TensorDescriptor(DataType::UINT64, {2}));
    const CubDeviceTopKKeysPlan topk_plan = prepareCubDeviceTopKKeys(topk_keys_in, topk_keys_out, 4, 2);
    Tensor topk_temp = allocateCubTemporaryStorage(cubTemporaryStoragePlan(gpuPlacement, topk_plan.temp_storage_bytes));
    cubDeviceTopKKeys(topk_plan, topk_temp, topk_keys_in, topk_keys_out, stream);
    stream.synchronize();
    auto topk_keys = copyGpuVector<uint64_t>(topk_keys_out, stream);
    std::sort(topk_keys.begin(), topk_keys.end(), std::greater<uint64_t>());
    EXPECT_EQ(topk_keys, (std::vector<uint64_t>{9ULL, 7ULL}));

    Tensor segmented_topk_keys_in = makeGpuVector<uint64_t>({7ULL, 2ULL, 9ULL, 4ULL}, stream);
    Tensor segmented_topk_keys_out(gpuPlacement, TensorDescriptor(DataType::UINT64, {2}));
    const CubDeviceSegmentedTopKKeysPlan segmented_topk_plan =
        prepareCubDeviceSegmentedTopKKeys(segmented_topk_keys_in, segmented_topk_keys_out, 2, 2, 1);
    Tensor segmented_topk_temp = allocateCubTemporaryStorage(cubTemporaryStoragePlan(gpuPlacement, segmented_topk_plan.temp_storage_bytes));
    cubDeviceSegmentedTopKKeys(segmented_topk_plan, segmented_topk_temp, segmented_topk_keys_in, segmented_topk_keys_out, stream);
    stream.synchronize();
    EXPECT_EQ(copyGpuVector<uint64_t>(segmented_topk_keys_out, stream), (std::vector<uint64_t>{7ULL, 9ULL}));

    Tensor select_input = makeGpuVector<uint64_t>({11ULL, 22ULL, 33ULL, 44ULL}, stream);
    Tensor select_flags = makeGpuVector<uint8_t>({0U, 1U, 1U, 0U}, stream);
    Tensor select_output(gpuPlacement, TensorDescriptor(DataType::UINT64, {4}));
    Tensor select_count(gpuPlacement, TensorDescriptor(DataType::UINT32, {1}));
    const CubDeviceSelectFlaggedPlan select_plan =
        prepareCubDeviceSelectFlagged(select_input, select_flags, select_output, select_count, 4);
    Tensor select_temp = allocateCubTemporaryStorage(cubTemporaryStoragePlan(gpuPlacement, select_plan.temp_storage_bytes));
    cubDeviceSelectFlagged(select_plan, select_temp, select_input, select_flags, select_output, select_count, stream);
    stream.synchronize();
    EXPECT_EQ(copyGpuVector<uint32_t>(select_count, stream), (std::vector<uint32_t>{2U}));
    const auto selected = copyGpuVector<uint64_t>(select_output, stream);
    EXPECT_EQ(std::vector<uint64_t>(selected.begin(), selected.begin() + 2), (std::vector<uint64_t>{22ULL, 33ULL}));
}
#else
TEST(CubDevicePrimitives, RunLengthEncodeAndExclusiveScanReject64BitTypesWhenDisabled) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor input = makeGpuVector<uint64_t>({7ULL, 7ULL, 2ULL}, stream);
    Tensor unique_out(gpuPlacement, TensorDescriptor(DataType::UINT64, {3}));
    Tensor counts_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {3}));
    Tensor num_runs_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {1}));
    EXPECT_THROW((void)prepareCubDeviceRunLengthEncode(input, unique_out, counts_out, num_runs_out, 3), std::invalid_argument);

    Tensor scan_output(gpuPlacement, TensorDescriptor(DataType::UINT64, {3}));
    EXPECT_THROW((void)prepareCubDeviceExclusiveSum(input, scan_output, 3), std::invalid_argument);

    Tensor topk_output(gpuPlacement, TensorDescriptor(DataType::UINT64, {2}));
    EXPECT_THROW((void)prepareCubDeviceTopKKeys(input, topk_output, 3, 2), std::invalid_argument);

    Tensor select_flags = makeGpuVector<uint8_t>({1U, 0U, 1U}, stream);
    Tensor select_output(gpuPlacement, TensorDescriptor(DataType::UINT64, {3}));
    Tensor select_count(gpuPlacement, TensorDescriptor(DataType::UINT32, {1}));
    EXPECT_THROW((void)prepareCubDeviceSelectFlagged(input, select_flags, select_output, select_count, 3), std::invalid_argument);

    Tensor segmented_topk_output(gpuPlacement, TensorDescriptor(DataType::UINT64, {1}));
    EXPECT_THROW((void)prepareCubDeviceSegmentedTopKKeys(input, segmented_topk_output, 1, 3, 1), std::invalid_argument);

    Tensor offsets = makeGpuVector<uint32_t>({0U, 3U}, stream);
    EXPECT_THROW((void)prepareCubDeviceSegmentedExclusiveSum(input, scan_output, offsets, 3, 1), std::invalid_argument);

    Tensor reduce_output(gpuPlacement, TensorDescriptor(DataType::UINT64, {1}));
    EXPECT_THROW((void)prepareCubDeviceSegmentedReduceSum(input, reduce_output, offsets, 3, 1), std::invalid_argument);
    EXPECT_THROW((void)prepareCubDeviceSegmentedReduceMax(input, reduce_output, offsets, 3, 1), std::invalid_argument);
}
#endif

TEST(CubDevicePrimitives, ExclusiveScanSupportsFp16AndBf16Values) {
    expectExclusiveScanFloatingDType<__half>({1.0f, 2.0f, -0.5f, 3.0f}, {0.0f, 1.0f, 3.0f, 2.5f}, 0.0f);
    expectExclusiveScanFloatingDType<__nv_bfloat16>({1.0f, 2.0f, -0.5f, 3.0f}, {0.0f, 1.0f, 3.0f, 2.5f}, 0.01f);
}

TEST(CubDevicePrimitives, ExclusiveScanRejectsSignedIntegerDTypes) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor input = makeGpuVector<int32_t>({1, 2, 3}, stream);
    Tensor output(gpuPlacement, TensorDescriptor(DataType::INT32, {3}));

    EXPECT_THROW((void)prepareCubDeviceExclusiveSum(input, output, 3), std::invalid_argument);
}

TEST(CubDevicePrimitives, RejectsMismatchedSortKeyDTypes) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor keys_in = makeGpuVector<float>({1.0f, 0.0f}, stream);
    Tensor keys_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {2}));

    EXPECT_THROW((void)cubDeviceRadixSortKeysTempBytes(keys_in, keys_out, 2), std::invalid_argument);
}

TEST(CubDevicePrimitives, PreparedPlansCanShareOnePreallocatedWorkspace) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor keys_in = makeGpuVector<uint32_t>({4U, 1U, 3U, 1U}, stream);
    Tensor values_in = makeGpuVector<uint32_t>({40U, 10U, 30U, 11U}, stream);
    Tensor keys_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {4}));
    Tensor values_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {4}));

    Tensor rle_unique_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {4}));
    Tensor rle_counts_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {4}));
    Tensor rle_num_runs_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {1}));
    Tensor offsets_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {4}));
    Tensor reduce_sum_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {1}));
    Tensor reduce_max_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {1}));
    Tensor reduce_min_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {1}));
    Tensor segmented_offsets = makeGpuVector<uint32_t>({0U, 2U, 4U}, stream);
    Tensor segmented_scan_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {4}));
    Tensor segmented_reduce_sum_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {2}));
    Tensor segmented_reduce_max_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {2}));
    Tensor topk_keys_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {2}));
    Tensor segmented_topk_keys_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {2}));
    Tensor select_flags = makeGpuVector<bool>({false, true, true, false}, stream);
    Tensor select_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {4}));
    Tensor select_count(gpuPlacement, TensorDescriptor(DataType::UINT32, {1}));
    Tensor find_values = makeGpuVector<uint32_t>({1U, 2U, 5U}, stream);
    Tensor lower_bound_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {3}));
    Tensor upper_bound_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {3}));
    Tensor find_if_index_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {1}));

    const CubDeviceRadixSortPairsPlan sort_plan =
        prepareCubDeviceRadixSortPairs(keys_in, keys_out, values_in, values_out, 4, CubSortOrder::Ascending);
    const CubDeviceRunLengthEncodePlan rle_plan =
        prepareCubDeviceRunLengthEncode(keys_out, rle_unique_out, rle_counts_out, rle_num_runs_out, 4);
    const CubDeviceReduceSumPlan reduce_sum_plan = prepareCubDeviceReduceSum(keys_in, reduce_sum_out, 4);
    const CubDeviceReduceMaxPlan reduce_max_plan = prepareCubDeviceReduceMax(keys_in, reduce_max_out, 4);
    const CubDeviceReduceMinPlan reduce_min_plan = prepareCubDeviceReduceMin(keys_in, reduce_min_out, 4);
    const CubDeviceExclusiveSumPlan scan_plan = prepareCubDeviceExclusiveSum(rle_counts_out, offsets_out, 3);
    const CubDeviceSegmentedExclusiveSumPlan segmented_scan_plan =
        prepareCubDeviceSegmentedExclusiveSum(keys_out, segmented_scan_out, segmented_offsets, 4, 2);
    const CubDeviceSegmentedReduceSumPlan segmented_reduce_sum_plan =
        prepareCubDeviceSegmentedReduceSum(keys_out, segmented_reduce_sum_out, segmented_offsets, 4, 2);
    const CubDeviceSegmentedReduceMaxPlan segmented_reduce_max_plan =
        prepareCubDeviceSegmentedReduceMax(keys_out, segmented_reduce_max_out, segmented_offsets, 4, 2);
    const CubDeviceTopKKeysPlan topk_plan = prepareCubDeviceTopKKeys(keys_in, topk_keys_out, 4, 2);
    const CubDeviceSegmentedTopKKeysPlan segmented_topk_plan = prepareCubDeviceSegmentedTopKKeys(keys_in, segmented_topk_keys_out, 2, 2, 1);
    const CubDeviceSelectFlaggedPlan select_plan = prepareCubDeviceSelectFlagged(keys_in, select_flags, select_out, select_count, 4);
    const CubDeviceLowerBoundPlan lower_bound_plan = prepareCubDeviceLowerBound(keys_out, find_values, lower_bound_out, 4, 3);
    const CubDeviceUpperBoundPlan upper_bound_plan = prepareCubDeviceUpperBound(keys_out, find_values, upper_bound_out, 4, 3);
    const CubDeviceFindIfFlaggedPlan find_if_plan = prepareCubDeviceFindIfFlagged(select_flags, find_if_index_out, 4);

    const CubTemporaryStoragePlan sort_workspace_plan = cubTemporaryStoragePlan(gpuPlacement, sort_plan.temp_storage_bytes);
    const CubTemporaryStoragePlan rle_workspace_plan = cubTemporaryStoragePlan(gpuPlacement, rle_plan.temp_storage_bytes);
    const CubTemporaryStoragePlan reduce_sum_workspace_plan = cubTemporaryStoragePlan(gpuPlacement, reduce_sum_plan.temp_storage_bytes);
    const CubTemporaryStoragePlan reduce_max_workspace_plan = cubTemporaryStoragePlan(gpuPlacement, reduce_max_plan.temp_storage_bytes);
    const CubTemporaryStoragePlan reduce_min_workspace_plan = cubTemporaryStoragePlan(gpuPlacement, reduce_min_plan.temp_storage_bytes);
    const CubTemporaryStoragePlan scan_workspace_plan = cubTemporaryStoragePlan(gpuPlacement, scan_plan.temp_storage_bytes);
    const CubTemporaryStoragePlan segmented_scan_workspace_plan =
        cubTemporaryStoragePlan(gpuPlacement, segmented_scan_plan.temp_storage_bytes);
    const CubTemporaryStoragePlan segmented_reduce_sum_workspace_plan =
        cubTemporaryStoragePlan(gpuPlacement, segmented_reduce_sum_plan.temp_storage_bytes);
    const CubTemporaryStoragePlan segmented_reduce_max_workspace_plan =
        cubTemporaryStoragePlan(gpuPlacement, segmented_reduce_max_plan.temp_storage_bytes);
    const CubTemporaryStoragePlan topk_workspace_plan = cubTemporaryStoragePlan(gpuPlacement, topk_plan.temp_storage_bytes);
    const CubTemporaryStoragePlan segmented_topk_workspace_plan = cubTemporaryStoragePlan(gpuPlacement, segmented_topk_plan.temp_storage_bytes);
    const CubTemporaryStoragePlan select_workspace_plan = cubTemporaryStoragePlan(gpuPlacement, select_plan.temp_storage_bytes);
    const CubTemporaryStoragePlan lower_bound_workspace_plan = cubTemporaryStoragePlan(gpuPlacement, lower_bound_plan.temp_storage_bytes);
    const CubTemporaryStoragePlan upper_bound_workspace_plan = cubTemporaryStoragePlan(gpuPlacement, upper_bound_plan.temp_storage_bytes);
    const CubTemporaryStoragePlan find_if_workspace_plan = cubTemporaryStoragePlan(gpuPlacement, find_if_plan.temp_storage_bytes);
    const CubTemporaryStoragePlan workspace_plan = cubMaxTemporaryStoragePlan(
        {sort_workspace_plan,
         rle_workspace_plan,
         reduce_sum_workspace_plan,
         reduce_max_workspace_plan,
         reduce_min_workspace_plan,
         scan_workspace_plan,
         segmented_scan_workspace_plan,
         segmented_reduce_sum_workspace_plan,
         segmented_reduce_max_workspace_plan,
         topk_workspace_plan,
         segmented_topk_workspace_plan,
         select_workspace_plan,
         lower_bound_workspace_plan,
         upper_bound_workspace_plan,
         find_if_workspace_plan});
    Tensor workspace = allocateCubTemporaryStorage(workspace_plan);

    cubDeviceRadixSortPairs(sort_plan, workspace, keys_in, keys_out, values_in, values_out, stream);
    cubDeviceRunLengthEncode(rle_plan, workspace, keys_out, rle_unique_out, rle_counts_out, rle_num_runs_out, stream);
    cubDeviceReduceSum(reduce_sum_plan, workspace, keys_in, reduce_sum_out, stream);
    cubDeviceReduceMax(reduce_max_plan, workspace, keys_in, reduce_max_out, stream);
    cubDeviceReduceMin(reduce_min_plan, workspace, keys_in, reduce_min_out, stream);
    cubDeviceExclusiveSum(scan_plan, workspace, rle_counts_out, offsets_out, stream);
    cubDeviceSegmentedExclusiveSum(segmented_scan_plan, workspace, keys_out, segmented_scan_out, segmented_offsets, stream);
    cubDeviceSegmentedReduceSum(
        segmented_reduce_sum_plan, workspace, keys_out, segmented_reduce_sum_out, segmented_offsets, stream);
    cubDeviceSegmentedReduceMax(
        segmented_reduce_max_plan, workspace, keys_out, segmented_reduce_max_out, segmented_offsets, stream);
    cubDeviceTopKKeys(topk_plan, workspace, keys_in, topk_keys_out, stream);
    cubDeviceSegmentedTopKKeys(segmented_topk_plan, workspace, keys_in, segmented_topk_keys_out, stream);
    cubDeviceSelectFlagged(select_plan, workspace, keys_in, select_flags, select_out, select_count, stream);
    cubDeviceLowerBound(lower_bound_plan, workspace, keys_out, find_values, lower_bound_out, stream);
    cubDeviceUpperBound(upper_bound_plan, workspace, keys_out, find_values, upper_bound_out, stream);
    cubDeviceFindIfFlagged(find_if_plan, workspace, select_flags, find_if_index_out, stream);
    stream.synchronize();

    EXPECT_EQ(copyGpuVector<uint32_t>(keys_out, stream), (std::vector<uint32_t>{1U, 1U, 3U, 4U}));
    EXPECT_EQ(copyGpuVector<uint32_t>(values_out, stream), (std::vector<uint32_t>{10U, 11U, 30U, 40U}));
    EXPECT_EQ(copyGpuVector<uint32_t>(rle_num_runs_out, stream), (std::vector<uint32_t>{3U}));

    const auto unique = copyGpuVector<uint32_t>(rle_unique_out, stream);
    const auto counts = copyGpuVector<uint32_t>(rle_counts_out, stream);
    const auto offsets = copyGpuVector<uint32_t>(offsets_out, stream);
    const auto reduce_sum = copyGpuVector<uint32_t>(reduce_sum_out, stream);
    const auto reduce_max = copyGpuVector<uint32_t>(reduce_max_out, stream);
    const auto reduce_min = copyGpuVector<uint32_t>(reduce_min_out, stream);
    const auto segmented_scan = copyGpuVector<uint32_t>(segmented_scan_out, stream);
    const auto segmented_reduce_sum = copyGpuVector<uint32_t>(segmented_reduce_sum_out, stream);
    const auto segmented_reduce_max = copyGpuVector<uint32_t>(segmented_reduce_max_out, stream);
    auto topk_keys = copyGpuVector<uint32_t>(topk_keys_out, stream);
    const auto segmented_topk_keys = copyGpuVector<uint32_t>(segmented_topk_keys_out, stream);
    const auto select_count_values = copyGpuVector<uint32_t>(select_count, stream);
    const auto selected = copyGpuVector<uint32_t>(select_out, stream);
    const auto lower_bound_values = copyGpuVector<uint32_t>(lower_bound_out, stream);
    const auto upper_bound_values = copyGpuVector<uint32_t>(upper_bound_out, stream);
    const auto find_if_index = copyGpuVector<uint32_t>(find_if_index_out, stream);
    std::sort(topk_keys.begin(), topk_keys.end(), std::greater<uint32_t>());
    EXPECT_EQ(std::vector<uint32_t>(unique.begin(), unique.begin() + 3), (std::vector<uint32_t>{1U, 3U, 4U}));
    EXPECT_EQ(std::vector<uint32_t>(counts.begin(), counts.begin() + 3), (std::vector<uint32_t>{2U, 1U, 1U}));
    EXPECT_EQ(reduce_sum, (std::vector<uint32_t>{9U}));
    EXPECT_EQ(reduce_max, (std::vector<uint32_t>{4U}));
    EXPECT_EQ(reduce_min, (std::vector<uint32_t>{1U}));
    EXPECT_EQ(std::vector<uint32_t>(offsets.begin(), offsets.begin() + 3), (std::vector<uint32_t>{0U, 2U, 3U}));
    EXPECT_EQ(segmented_scan, (std::vector<uint32_t>{0U, 1U, 0U, 3U}));
    EXPECT_EQ(segmented_reduce_sum, (std::vector<uint32_t>{2U, 7U}));
    EXPECT_EQ(segmented_reduce_max, (std::vector<uint32_t>{1U, 4U}));
    EXPECT_EQ(topk_keys, (std::vector<uint32_t>{4U, 3U}));
    EXPECT_EQ(segmented_topk_keys, (std::vector<uint32_t>{4U, 3U}));
    EXPECT_EQ(select_count_values, (std::vector<uint32_t>{2U}));
    EXPECT_EQ(std::vector<uint32_t>(selected.begin(), selected.begin() + 2), (std::vector<uint32_t>{1U, 3U}));
    EXPECT_EQ(lower_bound_values, (std::vector<uint32_t>{0U, 2U, 4U}));
    EXPECT_EQ(upper_bound_values, (std::vector<uint32_t>{2U, 2U, 4U}));
    EXPECT_EQ(find_if_index, (std::vector<uint32_t>{1U}));
}

TEST(CubDevicePrimitives, PreparedPlanRejectsUndersizedWorkspace) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor keys_in = makeGpuVector<uint32_t>({2U, 1U, 0U}, stream);
    Tensor keys_out(gpuPlacement, TensorDescriptor(DataType::UINT32, {3}));

    // CUB is allowed to report a one-byte temporary-storage requirement for tiny inputs.
    // Exercise Thor's hot-path workspace guard directly rather than depending on a
    // particular CUB query size.
    CubDeviceRadixSortKeysPlan sort_plan = prepareCubDeviceRadixSortKeys(keys_in, keys_out, 3);
    sort_plan.temp_storage_bytes = std::max<size_t>(sort_plan.temp_storage_bytes, 2U);

    Tensor too_small(gpuPlacement, TensorDescriptor(DataType::UINT8, {static_cast<uint64_t>(sort_plan.temp_storage_bytes - 1)}));
    EXPECT_THROW(cubDeviceRadixSortKeys(sort_plan, too_small, keys_in, keys_out, stream), std::invalid_argument);
}
