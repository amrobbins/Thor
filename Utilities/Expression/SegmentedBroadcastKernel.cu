#include "Utilities/Expression/SegmentedBroadcastKernel.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <stdexcept>

#include "Utilities/Common/LowPrecisionFloat.h"
#include "Utilities/Expression/CudaHelpers.h"

namespace ThorImplementation {
namespace {

template <typename T>
__device__ inline float toFloat(T value);

template <>
__device__ inline float toFloat<float>(float value) {
    return value;
}

template <>
__device__ inline float toFloat<__half>(__half value) {
    return __half2float(value);
}

template <>
__device__ inline float toFloat<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <>
__device__ inline float toFloat<__nv_fp8_e4m3>(__nv_fp8_e4m3 value) {
    return __half2float(__half(value));
}

template <>
__device__ inline float toFloat<__nv_fp8_e5m2>(__nv_fp8_e5m2 value) {
    return __half2float(__half(value));
}

template <typename T>
__device__ inline T fromFloat(float value);

template <>
__device__ inline float fromFloat<float>(float value) {
    return value;
}

template <>
__device__ inline __half fromFloat<__half>(float value) {
    return __float2half_rn(value);
}

template <>
__device__ inline __nv_bfloat16 fromFloat<__nv_bfloat16>(float value) {
    return __float2bfloat16(value);
}

template <>
__device__ inline __nv_fp8_e4m3 fromFloat<__nv_fp8_e4m3>(float value) {
    return ThorLowPrecision::toFp8E4M3Satfinite(value);
}

template <>
__device__ inline __nv_fp8_e5m2 fromFloat<__nv_fp8_e5m2>(float value) {
    return __nv_fp8_e5m2(__float2half_rn(value));
}

template <typename T>
__device__ inline T normalizeByLength(T value, uint64_t row_length) {
    return fromFloat<T>(toFloat(value) / static_cast<float>(row_length));
}

template <>
__device__ inline double normalizeByLength<double>(double value, uint64_t row_length) {
    return value / static_cast<double>(row_length);
}

template <typename T, typename OffsetT>
__global__ void segmentedBroadcastKernel(const T* per_segment_values,
                                         const OffsetT* offsets,
                                         T* output,
                                         uint64_t batch_size,
                                         uint64_t output_capacity,
                                         bool normalize_by_segment_length) {
    const uint64_t row = blockIdx.x;
    if (row >= batch_size) {
        return;
    }

    const uint64_t begin = static_cast<uint64_t>(offsets[row]);
    const uint64_t end = static_cast<uint64_t>(offsets[row + 1]);
    if (begin > end || end > output_capacity) {
        return;
    }

    const uint64_t row_length = end - begin;
    if (row_length == 0) {
        return;
    }

    T value = per_segment_values[row];
    if (normalize_by_segment_length) {
        value = normalizeByLength<T>(value, row_length);
    }

    for (uint64_t index = begin + threadIdx.x; index < end; index += blockDim.x) {
        output[index] = value;
    }
}

template <typename T>
void dispatchOffsets(const Tensor& per_segment_values,
                     const Tensor& segment_offsets,
                     Tensor& output,
                     bool normalize_by_segment_length,
                     Stream& stream) {
    const uint64_t batch_size = per_segment_values.getTotalNumElements();
    if (batch_size == 0) {
        return;
    }

    constexpr uint32_t threads_per_block = 256;
    switch (segment_offsets.getDataType()) {
        case DataType::UINT32:
            segmentedBroadcastKernel<T, uint32_t><<<static_cast<uint32_t>(batch_size), threads_per_block, 0, stream.getStream()>>>(
                per_segment_values.getMemPtr<T>(),
                segment_offsets.getMemPtr<uint32_t>(),
                output.getMemPtr<T>(),
                batch_size,
                output.getTotalNumElements(),
                normalize_by_segment_length);
            break;
        case DataType::UINT64:
            segmentedBroadcastKernel<T, uint64_t><<<static_cast<uint32_t>(batch_size), threads_per_block, 0, stream.getStream()>>>(
                per_segment_values.getMemPtr<T>(),
                segment_offsets.getMemPtr<uint64_t>(),
                output.getMemPtr<T>(),
                batch_size,
                output.getTotalNumElements(),
                normalize_by_segment_length);
            break;
        default:
            throw std::runtime_error("Segmented broadcast offsets must use UINT32 or UINT64.");
    }
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace

void launchSegmentedBroadcast(const Tensor& per_segment_values,
                              const Tensor& segment_offsets,
                              Tensor& output,
                              bool normalize_by_segment_length,
                              Stream& stream) {
    if (per_segment_values.getPlacement() != segment_offsets.getPlacement() ||
        per_segment_values.getPlacement() != output.getPlacement()) {
        throw std::runtime_error("Segmented broadcast tensors must share one placement.");
    }
    if (per_segment_values.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU ||
        per_segment_values.getPlacement().getDeviceNum() != stream.getGpuNum()) {
        throw std::runtime_error("Segmented broadcast requires tensors on the execution GPU.");
    }
    if (per_segment_values.getDimensions().size() != 1 || segment_offsets.getDimensions().size() != 1 ||
        output.getDimensions().size() != 1) {
        throw std::runtime_error("Segmented broadcast currently supports rank-1 scalar ragged values only.");
    }
    const uint64_t batch_size = per_segment_values.getDimensions()[0];
    if (segment_offsets.getDimensions()[0] != batch_size + 1) {
        throw std::runtime_error("Segmented broadcast offsets must have shape [batch_size + 1].");
    }
    if (per_segment_values.getDataType() != output.getDataType()) {
        throw std::runtime_error("Segmented broadcast input and output dtypes must match.");
    }

    switch (per_segment_values.getDataType()) {
        case DataType::FP8_E4M3:
            dispatchOffsets<__nv_fp8_e4m3>(per_segment_values, segment_offsets, output, normalize_by_segment_length, stream);
            break;
        case DataType::FP8_E5M2:
            dispatchOffsets<__nv_fp8_e5m2>(per_segment_values, segment_offsets, output, normalize_by_segment_length, stream);
            break;
        case DataType::FP16:
            dispatchOffsets<__half>(per_segment_values, segment_offsets, output, normalize_by_segment_length, stream);
            break;
        case DataType::BF16:
            dispatchOffsets<__nv_bfloat16>(per_segment_values, segment_offsets, output, normalize_by_segment_length, stream);
            break;
        case DataType::FP32:
            dispatchOffsets<float>(per_segment_values, segment_offsets, output, normalize_by_segment_length, stream);
            break;
        case DataType::FP64:
            dispatchOffsets<double>(per_segment_values, segment_offsets, output, normalize_by_segment_length, stream);
            break;
        default:
            throw std::runtime_error("Segmented broadcast requires a floating-point expression dtype.");
    }
}

}  // namespace ThorImplementation
