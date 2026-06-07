#pragma once

#include "Utilities/TensorOperations/Cub/CubDevicePrimitives.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ThorImplementation::CubDevicePrimitiveSupport {

[[nodiscard]] std::string dtypeName(DataType dtype);

void requireDenseContiguousGpuTensor(const Tensor& tensor, const char* name);
void requireSameGpuPlacement(const Tensor& a, const Tensor& b, const char* a_name, const char* b_name);
void requireStorageForNumItems(const Tensor& tensor, const char* name, uint64_t num_items);
[[nodiscard]] int checkedCubNumItems(uint64_t num_items);
[[nodiscard]] int64_t checkedCubInt64Count(uint64_t count, const char* name);
[[nodiscard]] int fullEndBitFor(DataType dtype, int end_bit);
void validateBitRange(DataType dtype, int begin_bit, int end_bit);
void requireTempStorage(const Tensor& temp_storage, const TensorPlacement& placement, size_t temp_storage_bytes);
[[nodiscard]] void* mutableCubTempStoragePtr(const Tensor& temp_storage);

void validateSortKeys(const Tensor& keys_in, const Tensor& keys_out, uint64_t num_items, int begin_bit, int end_bit);
void validateSortPairs(const Tensor& keys_in,
                       const Tensor& keys_out,
                       const Tensor& values_in,
                       const Tensor& values_out,
                       uint64_t num_items,
                       int begin_bit,
                       int end_bit);
void validateSegmentOffsets(const Tensor& keys_in, const Tensor& segment_offsets, uint64_t num_items, uint64_t num_segments);
void validateSegmentedSortKeys(const Tensor& keys_in,
                               const Tensor& keys_out,
                               const Tensor& segment_offsets,
                               uint64_t num_items,
                               uint64_t num_segments,
                               int begin_bit,
                               int end_bit);
void validateSegmentedSortPairs(const Tensor& keys_in,
                                const Tensor& keys_out,
                                const Tensor& values_in,
                                const Tensor& values_out,
                                const Tensor& segment_offsets,
                                uint64_t num_items,
                                uint64_t num_segments,
                                int begin_bit,
                                int end_bit);
void validateRle(const Tensor& input, const Tensor& unique_out, const Tensor& counts_out, const Tensor& num_runs_out, uint64_t num_items);
void validateExclusiveSum(const Tensor& input, const Tensor& output, uint64_t num_items);

template <typename Fn>
decltype(auto) dispatchSortKeyDType(DataType dtype, Fn&& fn) {
    switch (dtype) {
        case DataType::UINT8:
            return fn.template operator()<uint8_t>();
        case DataType::INT8:
            return fn.template operator()<int8_t>();
        case DataType::UINT16:
            return fn.template operator()<uint16_t>();
        case DataType::INT16:
            return fn.template operator()<int16_t>();
        case DataType::UINT32:
            return fn.template operator()<uint32_t>();
        case DataType::INT32:
            return fn.template operator()<int32_t>();
#if THOR_CUB_ENABLE_64BIT_TYPES
        case DataType::UINT64:
            return fn.template operator()<uint64_t>();
        case DataType::INT64:
            return fn.template operator()<int64_t>();
#endif
        case DataType::FP16:
            return fn.template operator()<__half>();
        case DataType::BF16:
            return fn.template operator()<__nv_bfloat16>();
        case DataType::FP32:
            return fn.template operator()<float>();
#if THOR_CUB_ENABLE_64BIT_TYPES
        case DataType::FP64:
            return fn.template operator()<double>();
#endif
#if THOR_CUB_ENABLE_FP8_TYPES
        case DataType::FP8_E4M3:
            return fn.template operator()<__nv_fp8_e4m3>();
        case DataType::FP8_E5M2:
            return fn.template operator()<__nv_fp8_e5m2>();
#endif
        default:
            throw std::invalid_argument("Unsupported CUB radix-sort key dtype " + dtypeName(dtype) + ".");
    }
}

template <typename Fn>
decltype(auto) dispatchSortValueDType(DataType dtype, Fn&& fn) {
    switch (dtype) {
        case DataType::UINT32:
            return fn.template operator()<uint32_t>();
#if THOR_CUB_ENABLE_64BIT_TYPES
        case DataType::UINT64:
            return fn.template operator()<uint64_t>();
#endif
        default:
            throw std::invalid_argument("Unsupported CUB radix-sort index value dtype " + dtypeName(dtype) + ".");
    }
}

template <typename Fn>
decltype(auto) dispatchRleDType(DataType dtype, Fn&& fn) {
    switch (dtype) {
        case DataType::UINT8:
            return fn.template operator()<uint8_t>();
        case DataType::INT8:
            return fn.template operator()<int8_t>();
        case DataType::UINT16:
            return fn.template operator()<uint16_t>();
        case DataType::INT16:
            return fn.template operator()<int16_t>();
        case DataType::UINT32:
            return fn.template operator()<uint32_t>();
        case DataType::INT32:
            return fn.template operator()<int32_t>();
#if THOR_CUB_ENABLE_64BIT_TYPES
        case DataType::UINT64:
            return fn.template operator()<uint64_t>();
        case DataType::INT64:
            return fn.template operator()<int64_t>();
#endif
        case DataType::FP16:
            return fn.template operator()<__half>();
        case DataType::BF16:
            return fn.template operator()<__nv_bfloat16>();
        case DataType::FP32:
            return fn.template operator()<float>();
#if THOR_CUB_ENABLE_64BIT_TYPES
        case DataType::FP64:
            return fn.template operator()<double>();
#endif
        default:
            throw std::invalid_argument("Unsupported direct CUB run-length-encode dtype " + dtypeName(dtype) + ".");
    }
}

template <typename Fn>
decltype(auto) dispatchScanDType(DataType dtype, Fn&& fn) {
    switch (dtype) {
        case DataType::UINT32:
            return fn.template operator()<uint32_t>();
#if THOR_CUB_ENABLE_64BIT_TYPES
        case DataType::UINT64:
            return fn.template operator()<uint64_t>();
#endif
        case DataType::FP16:
            return fn.template operator()<__half>();
        case DataType::BF16:
            return fn.template operator()<__nv_bfloat16>();
        case DataType::FP32:
            return fn.template operator()<float>();
#if THOR_CUB_ENABLE_64BIT_TYPES
        case DataType::FP64:
            return fn.template operator()<double>();
#endif
        default:
            throw std::invalid_argument("Unsupported CUB exclusive-sum dtype " + dtypeName(dtype) + ".");
    }
}

template <typename Fn>
decltype(auto) dispatchSegmentOffsetDType(DataType dtype, Fn&& fn) {
    switch (dtype) {
        case DataType::UINT32:
            return fn.template operator()<uint32_t>();
#if THOR_CUB_ENABLE_64BIT_TYPES
        case DataType::UINT64:
            return fn.template operator()<uint64_t>();
#endif
        default:
            throw std::invalid_argument("Unsupported CUB segment-offset dtype " + dtypeName(dtype) + ".");
    }
}

}  // namespace ThorImplementation::CubDevicePrimitiveSupport
