#include "Utilities/Expression/SegmentedBroadcastKernel.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <limits>
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
                                         uint64_t elements_per_value,
                                         bool normalize_by_segment_length) {
    const uint64_t row = static_cast<uint64_t>(blockIdx.x);
    const uint64_t begin = static_cast<uint64_t>(offsets[row]);
    const uint64_t end = static_cast<uint64_t>(offsets[row + 1]);
    const uint64_t row_length = end - begin;
    if (row_length == 0) {
        return;
    }

    const uint64_t row_elements = row_length * elements_per_value;
    for (uint64_t flat = threadIdx.x; flat < row_elements; flat += blockDim.x) {
        const uint64_t value_offset = flat / elements_per_value;
        const uint64_t component = flat - value_offset * elements_per_value;
        T value = per_segment_values[row * elements_per_value + component];
        if (normalize_by_segment_length) {
            value = normalizeByLength<T>(value, row_length);
        }
        output[(begin + value_offset) * elements_per_value + component] = value;
    }
}

template <typename T>
void dispatchOffsets(const Tensor& per_segment_values,
                     const Tensor& segment_offsets,
                     Tensor& output,
                     uint64_t elements_per_value,
                     bool normalize_by_segment_length,
                     Stream& stream) {
    const uint64_t batch_size = per_segment_values.getDimensions()[0];
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
                elements_per_value,
                normalize_by_segment_length);
            break;
        case DataType::UINT64:
            segmentedBroadcastKernel<T, uint64_t><<<static_cast<uint32_t>(batch_size), threads_per_block, 0, stream.getStream()>>>(
                per_segment_values.getMemPtr<T>(),
                segment_offsets.getMemPtr<uint64_t>(),
                output.getMemPtr<T>(),
                elements_per_value,
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
    if (per_segment_values.getDimensions().empty() || segment_offsets.getDimensions().size() != 1 || output.getDimensions().empty()) {
        throw std::runtime_error("Segmented broadcast requires per-segment values [B,D...], offsets [B+1], and output [N,D...].");
    }
    const uint64_t batch_size = per_segment_values.getDimensions()[0];
    if (batch_size == 0 || batch_size > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
        throw std::runtime_error("Segmented broadcast batch size exceeds CUDA grid.x capacity.");
    }
    if (segment_offsets.getDimensions()[0] != batch_size + 1) {
        throw std::runtime_error("Segmented broadcast offsets must have shape [batch_size + 1].");
    }
    if (per_segment_values.getDataType() != output.getDataType()) {
        throw std::runtime_error("Segmented broadcast input and output dtypes must match.");
    }

    // Internal expression aliases may reshape the trailing dimensions while preserving
    // the dense row-major per-value storage extent (for example [B, 4] <-> [B, 2, 2],
    // or [B, 1] <-> [B]). The broadcast kernel is intentionally flat over those
    // trailing elements, so elements-per-value is the execution invariant. Logical
    // output shape preservation is validated by the compiler/stamping layer.
    const uint64_t elements_per_value = per_segment_values.getTotalNumElements() / batch_size;
    if (elements_per_value == 0 || output.getDimensions()[0] == 0 ||
        output.getTotalNumElements() % output.getDimensions()[0] != 0 ||
        output.getTotalNumElements() / output.getDimensions()[0] != elements_per_value) {
        throw std::runtime_error("Segmented broadcast trailing dimensions produce an invalid elements-per-value extent.");
    }

    switch (per_segment_values.getDataType()) {
        case DataType::FP8_E4M3:
            dispatchOffsets<__nv_fp8_e4m3>(per_segment_values, segment_offsets, output, elements_per_value, normalize_by_segment_length, stream);
            break;
        case DataType::FP8_E5M2:
            dispatchOffsets<__nv_fp8_e5m2>(per_segment_values, segment_offsets, output, elements_per_value, normalize_by_segment_length, stream);
            break;
        case DataType::FP16:
            dispatchOffsets<__half>(per_segment_values, segment_offsets, output, elements_per_value, normalize_by_segment_length, stream);
            break;
        case DataType::BF16:
            dispatchOffsets<__nv_bfloat16>(per_segment_values, segment_offsets, output, elements_per_value, normalize_by_segment_length, stream);
            break;
        case DataType::FP32:
            dispatchOffsets<float>(per_segment_values, segment_offsets, output, elements_per_value, normalize_by_segment_length, stream);
            break;
        case DataType::FP64:
            dispatchOffsets<double>(per_segment_values, segment_offsets, output, elements_per_value, normalize_by_segment_length, stream);
            break;
        default:
            throw std::runtime_error("Segmented broadcast requires a floating-point expression dtype.");
    }
}

}  // namespace ThorImplementation
