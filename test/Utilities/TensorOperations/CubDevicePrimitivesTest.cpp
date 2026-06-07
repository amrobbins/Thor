#include "Utilities/TensorOperations/Cub/CubDevicePrimitives.h"

#include "cuda_runtime.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include "gtest/gtest.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <type_traits>
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

    EXPECT_TRUE(isCubSegmentOffsetDTypeSupported(DataType::UINT32));
    EXPECT_FALSE(isCubSegmentOffsetDTypeSupported(DataType::INT32));
    EXPECT_FALSE(isCubSegmentOffsetDTypeSupported(DataType::FP32));

#if THOR_CUB_ENABLE_64BIT_TYPES
    EXPECT_TRUE(isCubRadixSortKeyDTypeSupported(DataType::UINT64));
    EXPECT_TRUE(isCubRadixSortKeyDTypeSupported(DataType::INT64));
    EXPECT_TRUE(isCubRadixSortKeyDTypeSupported(DataType::FP64));
    EXPECT_TRUE(isCubRadixSortValueDTypeSupported(DataType::UINT64));
    EXPECT_TRUE(isCubRunLengthEncodeDTypeSupported(DataType::UINT64));
    EXPECT_TRUE(isCubRunLengthEncodeDTypeSupported(DataType::INT64));
    EXPECT_TRUE(isCubRunLengthEncodeDTypeSupported(DataType::FP64));
    EXPECT_TRUE(isCubExclusiveSumDTypeSupported(DataType::UINT64));
    EXPECT_FALSE(isCubExclusiveSumDTypeSupported(DataType::INT64));
    EXPECT_TRUE(isCubExclusiveSumDTypeSupported(DataType::FP64));
    EXPECT_TRUE(isCubSegmentOffsetDTypeSupported(DataType::UINT64));
    EXPECT_FALSE(isCubSegmentOffsetDTypeSupported(DataType::INT64));
#else
    EXPECT_FALSE(isCubRadixSortKeyDTypeSupported(DataType::UINT64));
    EXPECT_FALSE(isCubRadixSortKeyDTypeSupported(DataType::INT64));
    EXPECT_FALSE(isCubRadixSortKeyDTypeSupported(DataType::FP64));
    EXPECT_FALSE(isCubRadixSortValueDTypeSupported(DataType::UINT64));
    EXPECT_FALSE(isCubRunLengthEncodeDTypeSupported(DataType::UINT64));
    EXPECT_FALSE(isCubRunLengthEncodeDTypeSupported(DataType::INT64));
    EXPECT_FALSE(isCubRunLengthEncodeDTypeSupported(DataType::FP64));
    EXPECT_FALSE(isCubExclusiveSumDTypeSupported(DataType::UINT64));
    EXPECT_FALSE(isCubExclusiveSumDTypeSupported(DataType::INT64));
    EXPECT_FALSE(isCubExclusiveSumDTypeSupported(DataType::FP64));
    EXPECT_FALSE(isCubSegmentOffsetDTypeSupported(DataType::UINT64));
    EXPECT_FALSE(isCubSegmentOffsetDTypeSupported(DataType::INT64));
#endif

#if THOR_CUB_ENABLE_FP8_TYPES
    EXPECT_TRUE(isCubRadixSortKeyDTypeSupported(DataType::FP8_E4M3));
    EXPECT_TRUE(isCubRadixSortKeyDTypeSupported(DataType::FP8_E5M2));
    EXPECT_TRUE(isCubRunLengthEncodeDTypeSupported(DataType::FP8_E4M3));
    EXPECT_TRUE(isCubRunLengthEncodeDTypeSupported(DataType::FP8_E5M2));
#else
    EXPECT_FALSE(isCubRadixSortKeyDTypeSupported(DataType::FP8_E4M3));
    EXPECT_FALSE(isCubRadixSortKeyDTypeSupported(DataType::FP8_E5M2));
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

    const CubDeviceRadixSortPairsPlan sort_plan =
        prepareCubDeviceRadixSortPairs(keys_in, keys_out, values_in, values_out, 4, CubSortOrder::Ascending);
    const CubDeviceRunLengthEncodePlan rle_plan =
        prepareCubDeviceRunLengthEncode(keys_out, rle_unique_out, rle_counts_out, rle_num_runs_out, 4);
    const CubDeviceExclusiveSumPlan scan_plan = prepareCubDeviceExclusiveSum(rle_counts_out, offsets_out, 3);

    const CubTemporaryStoragePlan sort_workspace_plan = cubTemporaryStoragePlan(gpuPlacement, sort_plan.temp_storage_bytes);
    const CubTemporaryStoragePlan rle_workspace_plan = cubTemporaryStoragePlan(gpuPlacement, rle_plan.temp_storage_bytes);
    const CubTemporaryStoragePlan scan_workspace_plan = cubTemporaryStoragePlan(gpuPlacement, scan_plan.temp_storage_bytes);
    const CubTemporaryStoragePlan workspace_plan =
        cubMaxTemporaryStoragePlan({sort_workspace_plan, rle_workspace_plan, scan_workspace_plan});
    Tensor workspace = allocateCubTemporaryStorage(workspace_plan);

    cubDeviceRadixSortPairs(sort_plan, workspace, keys_in, keys_out, values_in, values_out, stream);
    cubDeviceRunLengthEncode(rle_plan, workspace, keys_out, rle_unique_out, rle_counts_out, rle_num_runs_out, stream);
    cubDeviceExclusiveSum(scan_plan, workspace, rle_counts_out, offsets_out, stream);
    stream.synchronize();

    EXPECT_EQ(copyGpuVector<uint32_t>(keys_out, stream), (std::vector<uint32_t>{1U, 1U, 3U, 4U}));
    EXPECT_EQ(copyGpuVector<uint32_t>(values_out, stream), (std::vector<uint32_t>{10U, 11U, 30U, 40U}));
    EXPECT_EQ(copyGpuVector<uint32_t>(rle_num_runs_out, stream), (std::vector<uint32_t>{3U}));

    const auto unique = copyGpuVector<uint32_t>(rle_unique_out, stream);
    const auto counts = copyGpuVector<uint32_t>(rle_counts_out, stream);
    const auto offsets = copyGpuVector<uint32_t>(offsets_out, stream);
    EXPECT_EQ(std::vector<uint32_t>(unique.begin(), unique.begin() + 3), (std::vector<uint32_t>{1U, 3U, 4U}));
    EXPECT_EQ(std::vector<uint32_t>(counts.begin(), counts.begin() + 3), (std::vector<uint32_t>{2U, 1U, 1U}));
    EXPECT_EQ(std::vector<uint32_t>(offsets.begin(), offsets.begin() + 3), (std::vector<uint32_t>{0U, 2U, 3U}));
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
