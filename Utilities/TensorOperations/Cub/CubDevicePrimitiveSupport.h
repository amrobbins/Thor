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
void requireTempStorage(const Tensor& temp_storage, const TensorPlacement& placement, size_t temp_storage_bytes);
[[nodiscard]] void* mutableCubTempStoragePtr(const Tensor& temp_storage);

void validateSegmentOffsets(const Tensor& reference,
                            const Tensor& segment_offsets,
                            uint64_t num_items,
                            uint64_t num_segments,
                            const char* reference_name);
void validateScan(const Tensor& input, const Tensor& output, uint64_t num_items);
void validateSegmentedScan(const Tensor& input,
                           const Tensor& output,
                           const Tensor& segment_offsets,
                           uint64_t num_items,
                           uint64_t num_segments);

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
            throw std::invalid_argument("Unsupported CUB scan dtype " + dtypeName(dtype) + ".");
    }
}

template <typename Fn>
decltype(auto) dispatchSegmentOffsetDType(DataType dtype, Fn&& fn) {
    switch (dtype) {
        case DataType::UINT32:
            return fn.template operator()<uint32_t>();
#if THOR_CUB_ENABLE_64BIT_SEGMENT_OFFSETS
        case DataType::UINT64:
            return fn.template operator()<uint64_t>();
#endif
        default:
            throw std::invalid_argument("Unsupported CUB segment-offset dtype " + dtypeName(dtype) + ".");
    }
}

}  // namespace ThorImplementation::CubDevicePrimitiveSupport
