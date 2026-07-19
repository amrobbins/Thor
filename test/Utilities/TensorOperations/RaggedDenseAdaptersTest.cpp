#include "Utilities/TensorOperations/Ragged/RaggedDenseAdapters.h"

#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace ThorImplementation;

namespace {

#define REQUIRE_CUDA_DEVICE()                                                                                          \
    do {                                                                                                                \
        int cuda_device_count_for_test = 0;                                                                             \
        const cudaError_t cuda_status_for_test = cudaGetDeviceCount(&cuda_device_count_for_test);                       \
        if (cuda_status_for_test != cudaSuccess || cuda_device_count_for_test <= 0) {                                    \
            GTEST_SKIP() << "CUDA device is required for ragged dense adapter tests.";                                  \
        }                                                                                                               \
    } while (false)

TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

template <typename T>
DataType dtypeFor();

template <>
DataType dtypeFor<float>() {
    return DataType::FP32;
}

template <>
DataType dtypeFor<uint32_t>() {
    return DataType::UINT32;
}

template <>
DataType dtypeFor<uint64_t>() {
    return DataType::UINT64;
}

template <>
DataType dtypeFor<int32_t>() {
    return DataType::INT32;
}

template <typename T>
Tensor makeGpuTensor(const std::vector<uint64_t>& dims, const std::vector<T>& values, Stream& stream) {
    Tensor cpu(cpuPlacement, TensorDescriptor(dtypeFor<T>(), dims));
    EXPECT_EQ(cpu.getTotalNumElements(), values.size());
    T* cpu_ptr = cpu.getMemPtr<T>();
    for (size_t i = 0; i < values.size(); ++i) {
        cpu_ptr[i] = values[i];
    }

    Tensor gpu(gpuPlacement, TensorDescriptor(dtypeFor<T>(), dims));
    gpu.copyFromAsync(cpu, stream);
    stream.synchronize();
    return gpu;
}

template <typename T>
std::vector<T> copyGpuTensor(const Tensor& gpu, Stream& stream) {
    Tensor cpu = gpu.clone(cpuPlacement);
    cpu.copyFromAsync(gpu, stream);
    stream.synchronize();

    std::vector<T> values(cpu.getTotalNumElements());
    const T* ptr = cpu.getMemPtr<T>();
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = ptr[i];
    }
    return values;
}

Tensor makeFilledGpuTensor(const std::vector<uint64_t>& dims, float value, Stream& stream) {
    Tensor gpu(gpuPlacement, TensorDescriptor(DataType::FP32, dims));
    gpu.fill(value, stream);
    stream.synchronize();
    return gpu;
}

Tensor makeValidationErrorBits() { return Tensor(gpuPlacement, TensorDescriptor(DataType::UINT32, {1})); }

uint32_t readValidationErrorBits(const Tensor& validation_error_bits, Stream& stream) {
    return copyGpuTensor<uint32_t>(validation_error_bits, stream).at(0);
}

}  // namespace

TEST(RaggedDenseAdapters, FromDenseWithOffsetsCopiesLogicalRowsAndLeavesUnusedCapacityUntouched) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    // Dense shape [B=4, max_length=4, width=2].
    Tensor dense = makeGpuTensor<float>({4, 4, 2},
                                        {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F,
                                         10.0F, 11.0F, 12.0F, 13.0F, 14.0F, 15.0F, 16.0F, 17.0F,
                                         20.0F, 21.0F, 22.0F, 23.0F, 24.0F, 25.0F, 26.0F, 27.0F,
                                         30.0F, 31.0F, 32.0F, 33.0F, 34.0F, 35.0F, 36.0F, 37.0F},
                                        stream);
    Tensor offsets = makeGpuTensor<uint64_t>({5}, {0ULL, 2ULL, 2ULL, 5ULL, 6ULL}, stream);
    Tensor values = makeFilledGpuTensor({8, 2}, -777.0F, stream);

    Tensor validation_error_bits = makeValidationErrorBits();
    RaggedTensor ragged = raggedFromDense(dense, offsets, values, validation_error_bits, stream);
    stream.synchronize();

    EXPECT_EQ(readValidationErrorBits(validation_error_bits, stream), ROW_PARTITION_VALID);
    EXPECT_TRUE(ragged.isInitialized());
    EXPECT_EQ(ragged.getBatchSize(), 4ULL);
    EXPECT_EQ(ragged.getMaxTotalValues(), 8ULL);
    EXPECT_EQ(ragged.getOffsetsDataType(), DataType::UINT64);
    EXPECT_EQ(copyGpuTensor<float>(values, stream),
              (std::vector<float>{0.0F, 1.0F, 2.0F, 3.0F,
                                  20.0F, 21.0F, 22.0F, 23.0F, 24.0F, 25.0F,
                                  30.0F, 31.0F,
                                  -777.0F, -777.0F, -777.0F, -777.0F}));
}

TEST(RaggedDenseAdapters, FromDenseWithLengthsBuildsOffsetsAndCopiesValuesWithoutAllocatingTempInternally) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor dense = makeGpuTensor<float>({3, 3}, {1.0F, 2.0F, 3.0F, 10.0F, 11.0F, 12.0F, 20.0F, 21.0F, 22.0F}, stream);
    Tensor lengths = makeGpuTensor<uint32_t>({3}, {2U, 0U, 1U}, stream);
    Tensor values = makeFilledGpuTensor({5}, -9.0F, stream);
    Tensor offsets(gpuPlacement, TensorDescriptor(DataType::UINT32, {4}));

    const RaggedFromDenseWithLengthsPlan plan = prepareRaggedFromDenseWithLengths(dense, lengths, values, offsets);
    Tensor temp_storage(gpuPlacement, TensorDescriptor(DataType::UINT8, {std::max<uint64_t>(static_cast<uint64_t>(plan.tempStorageBytes), 1ULL)}));

    Tensor validation_error_bits = makeValidationErrorBits();
    RaggedTensor ragged = raggedFromDense(plan, temp_storage, dense, lengths, values, offsets, validation_error_bits, stream);
    stream.synchronize();

    EXPECT_EQ(readValidationErrorBits(validation_error_bits, stream), ROW_PARTITION_VALID);
    EXPECT_TRUE(ragged.isInitialized());
    EXPECT_EQ(plan.batchSize, 3ULL);
    EXPECT_EQ(plan.maxLength, 3ULL);
    EXPECT_EQ(plan.maxTotalValues, 5ULL);
    EXPECT_EQ(copyGpuTensor<uint32_t>(offsets, stream), (std::vector<uint32_t>{0U, 2U, 2U, 3U}));
    EXPECT_EQ(copyGpuTensor<float>(values, stream), (std::vector<float>{1.0F, 2.0F, 20.0F, -9.0F, -9.0F}));
}

TEST(RaggedDenseAdapters, RaggedToDensePadsAndCopiesLogicalValues) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor values = makeGpuTensor<float>({6, 2}, {0.0F, 1.0F, 2.0F, 3.0F,
                                                20.0F, 21.0F, 22.0F, 23.0F, 24.0F, 25.0F,
                                                30.0F, 31.0F},
                                        stream);
    Tensor offsets = makeGpuTensor<uint32_t>({5}, {0U, 2U, 2U, 5U, 6U}, stream);
    RaggedTensor ragged(values, offsets);
    Tensor dense(gpuPlacement, TensorDescriptor(DataType::FP32, {4, 4, 2}));

    Tensor validation_error_bits = makeValidationErrorBits();
    raggedToDense(ragged, dense, -5.0, validation_error_bits, stream);
    stream.synchronize();

    EXPECT_EQ(readValidationErrorBits(validation_error_bits, stream), ROW_PARTITION_VALID);
    EXPECT_EQ(copyGpuTensor<float>(dense, stream),
              (std::vector<float>{0.0F, 1.0F, 2.0F, 3.0F, -5.0F, -5.0F, -5.0F, -5.0F,
                                  -5.0F, -5.0F, -5.0F, -5.0F, -5.0F, -5.0F, -5.0F, -5.0F,
                                  20.0F, 21.0F, 22.0F, 23.0F, 24.0F, 25.0F, -5.0F, -5.0F,
                                  30.0F, 31.0F, -5.0F, -5.0F, -5.0F, -5.0F, -5.0F, -5.0F}));
}

TEST(RaggedDenseAdapters, RoundTripDenseOffsetsRaggedDensePreservesLogicalRowsAndPadding) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor dense = makeGpuTensor<float>({3, 3}, {1.0F, 2.0F, 3.0F, 10.0F, 11.0F, 12.0F, 20.0F, 21.0F, 22.0F}, stream);
    Tensor offsets = makeGpuTensor<uint32_t>({4}, {0U, 2U, 2U, 3U}, stream);
    Tensor values = makeFilledGpuTensor({5}, -1.0F, stream);

    Tensor validation_error_bits = makeValidationErrorBits();
    RaggedTensor ragged = raggedFromDense(dense, offsets, values, validation_error_bits, stream);
    Tensor dense_out(gpuPlacement, TensorDescriptor(DataType::FP32, {3, 3}));
    raggedToDense(ragged, dense_out, 99.0, validation_error_bits, stream);
    stream.synchronize();

    EXPECT_EQ(readValidationErrorBits(validation_error_bits, stream), ROW_PARTITION_VALID);
    EXPECT_EQ(copyGpuTensor<float>(dense_out, stream),
              (std::vector<float>{1.0F, 2.0F, 99.0F, 99.0F, 99.0F, 99.0F, 20.0F, 99.0F, 99.0F}));
}

TEST(RaggedDenseAdapters, RejectsSignedLengthDType) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor dense = makeGpuTensor<float>({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F}, stream);
    Tensor lengths = makeGpuTensor<int32_t>({2}, {1, 1}, stream);
    Tensor values(gpuPlacement, TensorDescriptor(DataType::FP32, {2}));
    Tensor offsets(gpuPlacement, TensorDescriptor(DataType::UINT32, {3}));

    EXPECT_THROW(static_cast<void>(prepareRaggedFromDenseWithLengths(dense, lengths, values, offsets)), std::invalid_argument);
}

TEST(RaggedDenseAdapters, RejectsImplicitDenseShapeMismatch) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor dense = makeGpuTensor<float>({2, 2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F,
                                                 7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F},
                                        stream);
    Tensor offsets = makeGpuTensor<uint32_t>({3}, {0U, 1U, 2U}, stream);
    Tensor wrong_values(gpuPlacement, TensorDescriptor(DataType::FP32, {2, 4}));

    Tensor validation_error_bits = makeValidationErrorBits();
    EXPECT_THROW(static_cast<void>(raggedFromDense(dense, offsets, wrong_values, validation_error_bits, stream)), std::invalid_argument);
}


TEST(RaggedDenseAdapters, InvalidOffsetsSetDeviceStatusAndDoNotWriteValues) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor dense = makeGpuTensor<float>({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F}, stream);
    Tensor offsets = makeGpuTensor<uint32_t>({3}, {0U, 3U, 2U}, stream);
    Tensor values = makeFilledGpuTensor({4}, -17.0F, stream);
    Tensor validation_error_bits = makeValidationErrorBits();

    static_cast<void>(raggedFromDense(dense, offsets, values, validation_error_bits, stream));
    stream.synchronize();

    const uint32_t errors = readValidationErrorBits(validation_error_bits, stream);
    EXPECT_NE(errors & ROW_PARTITION_ROW_LENGTH_EXCEEDS_MAX, 0U);
    EXPECT_NE(errors & ROW_PARTITION_OFFSETS_MUST_BE_MONOTONIC, 0U);
    EXPECT_EQ(copyGpuTensor<float>(values, stream), (std::vector<float>{-17.0F, -17.0F, -17.0F, -17.0F}));
}

TEST(RaggedDenseAdapters, LengthsExceedingDenseRowsSetDeviceStatusAndDoNotWriteValues) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor dense = makeGpuTensor<float>({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F}, stream);
    Tensor lengths = makeGpuTensor<uint32_t>({2}, {3U, 1U}, stream);
    Tensor values = makeFilledGpuTensor({4}, -19.0F, stream);
    Tensor offsets(gpuPlacement, TensorDescriptor(DataType::UINT32, {3}));
    const RaggedFromDenseWithLengthsPlan plan = prepareRaggedFromDenseWithLengths(dense, lengths, values, offsets);
    Tensor temp_storage(gpuPlacement, TensorDescriptor(DataType::UINT8, {std::max<uint64_t>(static_cast<uint64_t>(plan.tempStorageBytes), 1ULL)}));
    Tensor validation_error_bits = makeValidationErrorBits();

    static_cast<void>(raggedFromDense(plan, temp_storage, dense, lengths, values, offsets, validation_error_bits, stream));
    stream.synchronize();

    EXPECT_NE(readValidationErrorBits(validation_error_bits, stream) & ROW_PARTITION_ROW_LENGTH_EXCEEDS_MAX, 0U);
    EXPECT_EQ(copyGpuTensor<float>(values, stream), (std::vector<float>{-19.0F, -19.0F, -19.0F, -19.0F}));
}

TEST(RaggedDenseAdapters, RaggedToDenseInvalidOffsetsLeavePaddingOnly) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor values = makeGpuTensor<float>({4}, {1.0F, 2.0F, 3.0F, 4.0F}, stream);
    Tensor offsets = makeGpuTensor<uint32_t>({3}, {1U, 2U, 4U}, stream);
    RaggedTensor ragged(values, offsets);
    Tensor dense(gpuPlacement, TensorDescriptor(DataType::FP32, {2, 2}));
    Tensor validation_error_bits = makeValidationErrorBits();

    raggedToDense(ragged, dense, 41.0, validation_error_bits, stream);
    stream.synchronize();

    EXPECT_NE(readValidationErrorBits(validation_error_bits, stream) & ROW_PARTITION_OFFSETS_MUST_START_AT_ZERO, 0U);
    EXPECT_EQ(copyGpuTensor<float>(dense, stream), (std::vector<float>{41.0F, 41.0F, 41.0F, 41.0F}));
}
