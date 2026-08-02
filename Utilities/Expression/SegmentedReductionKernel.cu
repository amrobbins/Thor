#include "Utilities/Expression/SegmentedReductionKernel.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <math_constants.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "Utilities/Common/LowPrecisionFloat.h"
#include "Utilities/Expression/CudaHelpers.h"

namespace ThorImplementation {
namespace {

constexpr uint32_t kThreadsPerBlock = 256;

enum class ReductionKind : uint8_t {
    Sum,
    Min,
    Max,
    Mean,
};

template <typename T>
__device__ inline float toFloat(T value);

template <>
__device__ inline float toFloat<float>(float value) {
    return value;
}

template <>
__device__ inline float toFloat<double>(double value) {
    return static_cast<float>(value);
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
__device__ inline double fromFloat<double>(float value) {
    return static_cast<double>(value);
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

template <ReductionKind Kind>
__device__ inline float reductionIdentity() {
    if constexpr (Kind == ReductionKind::Min) {
        return CUDART_INF_F;
    } else if constexpr (Kind == ReductionKind::Max) {
        return -CUDART_INF_F;
    } else {
        return 0.0F;
    }
}

template <ReductionKind Kind>
__device__ inline float combine(float lhs, float rhs) {
    if constexpr (Kind == ReductionKind::Min) {
        if (isnan(lhs)) {
            return lhs;
        }
        if (isnan(rhs)) {
            return rhs;
        }
        return fminf(lhs, rhs);
    } else if constexpr (Kind == ReductionKind::Max) {
        if (isnan(lhs)) {
            return lhs;
        }
        if (isnan(rhs)) {
            return rhs;
        }
        return fmaxf(lhs, rhs);
    } else {
        return lhs + rhs;
    }
}

template <typename T, typename OffsetT, ReductionKind Kind>
__global__ void vectorSegmentedReductionKernel(const T* values,
                                               const OffsetT* offsets,
                                               T* output,
                                               uint64_t elements_per_value) {
    const uint64_t output_index = static_cast<uint64_t>(blockIdx.x);
    const uint64_t row = output_index / elements_per_value;
    const uint64_t component = output_index - row * elements_per_value;

    const uint64_t begin = static_cast<uint64_t>(offsets[row]);
    const uint64_t end = static_cast<uint64_t>(offsets[row + 1]);
    const uint64_t row_length = end - begin;

    float local = reductionIdentity<Kind>();
    for (uint64_t value_index = begin + threadIdx.x; value_index < end; value_index += blockDim.x) {
        local = combine<Kind>(local, toFloat(values[value_index * elements_per_value + component]));
    }

    __shared__ float partial[kThreadsPerBlock];
    partial[threadIdx.x] = local;
    __syncthreads();

    for (uint32_t stride = kThreadsPerBlock / 2; stride != 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] = combine<Kind>(partial[threadIdx.x], partial[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float result = partial[0];
        if constexpr (Kind == ReductionKind::Mean) {
            result = row_length == 0 ? 0.0F : result / static_cast<float>(row_length);
        }
        output[row * elements_per_value + component] = fromFloat<T>(result);
    }
}

template <typename T, typename OffsetT, ReductionKind Kind>
void launchTyped(const Tensor& values,
                 const Tensor& segment_offsets,
                 Tensor& output,
                 uint64_t batch_size,
                 uint64_t elements_per_value,
                 Stream& stream) {
    if (elements_per_value > std::numeric_limits<uint64_t>::max() / batch_size) {
        throw std::invalid_argument("Vector segmented reduction launch block count overflows uint64_t.");
    }
    const uint64_t blocks = batch_size * elements_per_value;
    if (blocks > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
        throw std::invalid_argument("Vector segmented reduction launch grid exceeds CUDA grid.x capacity.");
    }
    vectorSegmentedReductionKernel<T, OffsetT, Kind><<<static_cast<uint32_t>(blocks), kThreadsPerBlock, 0, stream.getStream()>>>(
        values.getMemPtr<T>(),
        segment_offsets.getMemPtr<OffsetT>(),
        output.getMemPtr<T>(),
        elements_per_value);
    CUDA_CHECK(cudaGetLastError());
}

template <typename T, typename OffsetT>
void dispatchOp(ExprOp op,
                const Tensor& values,
                const Tensor& segment_offsets,
                Tensor& output,
                uint64_t batch_size,
                uint64_t elements_per_value,
                Stream& stream) {
    switch (op) {
        case ExprOp::SEGMENTED_REDUCE_SUM:
            launchTyped<T, OffsetT, ReductionKind::Sum>(values, segment_offsets, output, batch_size, elements_per_value, stream);
            return;
        case ExprOp::SEGMENTED_REDUCE_MIN:
            launchTyped<T, OffsetT, ReductionKind::Min>(values, segment_offsets, output, batch_size, elements_per_value, stream);
            return;
        case ExprOp::SEGMENTED_REDUCE_MAX:
            launchTyped<T, OffsetT, ReductionKind::Max>(values, segment_offsets, output, batch_size, elements_per_value, stream);
            return;
        case ExprOp::SEGMENTED_REDUCE_MEAN:
            launchTyped<T, OffsetT, ReductionKind::Mean>(values, segment_offsets, output, batch_size, elements_per_value, stream);
            return;
        default:
            throw std::invalid_argument("Unsupported vector segmented reduction op.");
    }
}

template <typename T>
void dispatchOffsets(ExprOp op,
                     const Tensor& values,
                     const Tensor& segment_offsets,
                     Tensor& output,
                     uint64_t batch_size,
                     uint64_t elements_per_value,
                     Stream& stream) {
    switch (segment_offsets.getDataType()) {
        case DataType::UINT32:
            dispatchOp<T, uint32_t>(op, values, segment_offsets, output, batch_size, elements_per_value, stream);
            return;
        case DataType::UINT64:
            dispatchOp<T, uint64_t>(op, values, segment_offsets, output, batch_size, elements_per_value, stream);
            return;
        default:
            throw std::invalid_argument("Vector segmented reduction offsets must use UINT32 or UINT64.");
    }
}

}  // namespace

void launchVectorSegmentedReduction(ExprOp op,
                                    const Tensor& values,
                                    const Tensor& segment_offsets,
                                    Tensor& output,
                                    uint64_t elements_per_value,
                                    Stream& stream) {
    if (elements_per_value == 0) {
        throw std::invalid_argument("Vector segmented reduction requires non-zero elements_per_value.");
    }
    if (values.getPlacement() != segment_offsets.getPlacement() || values.getPlacement() != output.getPlacement()) {
        throw std::invalid_argument("Vector segmented reduction tensors must share one placement.");
    }
    if (values.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU ||
        values.getPlacement().getDeviceNum() != stream.getGpuNum()) {
        throw std::invalid_argument("Vector segmented reduction requires tensors on the execution GPU.");
    }
    if (values.getDimensions().empty() || output.getDimensions().empty() || segment_offsets.getDimensions().size() != 1 ||
        segment_offsets.getDimensions()[0] == 0) {
        throw std::invalid_argument("Vector segmented reduction requires values [N,D...], output [B,D...], and offsets [B+1].");
    }
    if (values.getDataType() != output.getDataType()) {
        throw std::invalid_argument("Vector segmented reduction input and output dtypes must match.");
    }

    const uint64_t batch_size = segment_offsets.getDimensions()[0] - 1;
    if (batch_size == 0) {
        throw std::invalid_argument("Vector segmented reduction requires a non-zero batch size.");
    }
    std::vector<uint64_t> expected_output_dims = values.getDimensions();
    expected_output_dims[0] = batch_size;
    if (output.getDimensions() != expected_output_dims) {
        throw std::invalid_argument("Vector segmented reduction output shape must be [B,D...].");
    }
    if (values.getDimensions()[0] == 0 || values.getTotalNumElements() % values.getDimensions()[0] != 0 ||
        values.getTotalNumElements() / values.getDimensions()[0] != elements_per_value ||
        output.getTotalNumElements() % batch_size != 0 || output.getTotalNumElements() / batch_size != elements_per_value) {
        throw std::invalid_argument("Vector segmented reduction elements_per_value does not match tensor shapes.");
    }

    switch (values.getDataType()) {
        case DataType::FP8_E4M3:
            dispatchOffsets<__nv_fp8_e4m3>(op, values, segment_offsets, output, batch_size, elements_per_value, stream);
            return;
        case DataType::FP8_E5M2:
            dispatchOffsets<__nv_fp8_e5m2>(op, values, segment_offsets, output, batch_size, elements_per_value, stream);
            return;
        case DataType::FP16:
            dispatchOffsets<__half>(op, values, segment_offsets, output, batch_size, elements_per_value, stream);
            return;
        case DataType::BF16:
            dispatchOffsets<__nv_bfloat16>(op, values, segment_offsets, output, batch_size, elements_per_value, stream);
            return;
        case DataType::FP32:
            dispatchOffsets<float>(op, values, segment_offsets, output, batch_size, elements_per_value, stream);
            return;
        case DataType::FP64:
            dispatchOffsets<double>(op, values, segment_offsets, output, batch_size, elements_per_value, stream);
            return;
        default:
            throw std::invalid_argument("Vector segmented reduction requires a supported floating-point dtype.");
    }
}


namespace {

template <ReductionKind Kind>
__device__ inline bool candidateWins(bool lhs_valid,
                                     bool lhs_nan,
                                     float lhs_value,
                                     uint64_t lhs_index,
                                     bool rhs_valid,
                                     bool rhs_nan,
                                     float rhs_value,
                                     uint64_t rhs_index) {
    if (!lhs_valid) return false;
    if (!rhs_valid) return true;
    if (lhs_nan != rhs_nan) return lhs_nan;
    if (lhs_nan) return lhs_index < rhs_index;
    if constexpr (Kind == ReductionKind::Min) {
        if (lhs_value < rhs_value) return true;
        if (rhs_value < lhs_value) return false;
    } else {
        if (lhs_value > rhs_value) return true;
        if (rhs_value > lhs_value) return false;
    }
    return lhs_index < rhs_index;
}

template <typename ValueT, typename GradT, typename OffsetT, ReductionKind Kind>
__global__ void vectorSegmentedMinMaxBackwardKernel(const ValueT* values,
                                                     const OffsetT* offsets,
                                                     const GradT* grad_output,
                                                     GradT* grad_input,
                                                     uint64_t elements_per_value) {
    const uint64_t output_index = static_cast<uint64_t>(blockIdx.x);
    const uint64_t row = output_index / elements_per_value;
    const uint64_t component = output_index - row * elements_per_value;
    const uint64_t begin = static_cast<uint64_t>(offsets[row]);
    const uint64_t end = static_cast<uint64_t>(offsets[row + 1]);

    bool local_valid = false;
    bool local_nan = false;
    float local_value = 0.0F;
    uint64_t local_index = 0;
    for (uint64_t value_index = begin + threadIdx.x; value_index < end; value_index += blockDim.x) {
        const uint64_t element_index = value_index * elements_per_value + component;
        grad_input[element_index] = fromFloat<GradT>(0.0F);
        const float value = toFloat(values[element_index]);
        const bool value_nan = isnan(value);
        if (candidateWins<Kind>(true,
                                value_nan,
                                value,
                                value_index,
                                local_valid,
                                local_nan,
                                local_value,
                                local_index)) {
            local_valid = true;
            local_nan = value_nan;
            local_value = value;
            local_index = value_index;
        }
    }

    __shared__ float shared_values[kThreadsPerBlock];
    __shared__ uint64_t shared_indices[kThreadsPerBlock];
    __shared__ uint8_t shared_valid[kThreadsPerBlock];
    __shared__ uint8_t shared_nan[kThreadsPerBlock];
    shared_values[threadIdx.x] = local_value;
    shared_indices[threadIdx.x] = local_index;
    shared_valid[threadIdx.x] = local_valid ? 1U : 0U;
    shared_nan[threadIdx.x] = local_nan ? 1U : 0U;
    __syncthreads();

    for (uint32_t stride = kThreadsPerBlock / 2; stride != 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const uint32_t rhs = threadIdx.x + stride;
            if (candidateWins<Kind>(shared_valid[rhs] != 0,
                                    shared_nan[rhs] != 0,
                                    shared_values[rhs],
                                    shared_indices[rhs],
                                    shared_valid[threadIdx.x] != 0,
                                    shared_nan[threadIdx.x] != 0,
                                    shared_values[threadIdx.x],
                                    shared_indices[threadIdx.x])) {
                shared_values[threadIdx.x] = shared_values[rhs];
                shared_indices[threadIdx.x] = shared_indices[rhs];
                shared_valid[threadIdx.x] = shared_valid[rhs];
                shared_nan[threadIdx.x] = shared_nan[rhs];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0 && shared_valid[0] != 0) {
        grad_input[shared_indices[0] * elements_per_value + component] =
            grad_output[row * elements_per_value + component];
    }
}

template <typename ValueT, typename GradT, typename OffsetT>
void launchVectorMinMaxBackwardTyped(ExprOp op,
                                     const Tensor& values,
                                     const Tensor& segment_offsets,
                                     const Tensor& grad_output,
                                     Tensor& grad_input,
                                     uint64_t batch_size,
                                     uint64_t elements_per_value,
                                     Stream& stream) {
    if (elements_per_value > std::numeric_limits<uint64_t>::max() / batch_size) {
        throw std::invalid_argument("Vector segmented min/max backward launch block count overflows uint64_t.");
    }
    const uint64_t blocks = batch_size * elements_per_value;
    if (blocks > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
        throw std::invalid_argument("Vector segmented min/max backward launch grid exceeds CUDA grid.x capacity.");
    }

    if (op == ExprOp::SEGMENTED_REDUCE_MIN_BACKWARD) {
        vectorSegmentedMinMaxBackwardKernel<ValueT, GradT, OffsetT, ReductionKind::Min>
            <<<static_cast<uint32_t>(blocks), kThreadsPerBlock, 0, stream.getStream()>>>(
                values.getMemPtr<ValueT>(),
                segment_offsets.getMemPtr<OffsetT>(),
                grad_output.getMemPtr<GradT>(),
                grad_input.getMemPtr<GradT>(),
                elements_per_value);
    } else if (op == ExprOp::SEGMENTED_REDUCE_MAX_BACKWARD) {
        vectorSegmentedMinMaxBackwardKernel<ValueT, GradT, OffsetT, ReductionKind::Max>
            <<<static_cast<uint32_t>(blocks), kThreadsPerBlock, 0, stream.getStream()>>>(
                values.getMemPtr<ValueT>(),
                segment_offsets.getMemPtr<OffsetT>(),
                grad_output.getMemPtr<GradT>(),
                grad_input.getMemPtr<GradT>(),
                elements_per_value);
    } else {
        throw std::invalid_argument("Vector segmented min/max backward requires min-backward or max-backward op.");
    }
    CUDA_CHECK(cudaGetLastError());
}

template <typename ValueT, typename GradT>
void dispatchVectorMinMaxBackwardOffsets(ExprOp op,
                                         const Tensor& values,
                                         const Tensor& segment_offsets,
                                         const Tensor& grad_output,
                                         Tensor& grad_input,
                                         uint64_t batch_size,
                                         uint64_t elements_per_value,
                                         Stream& stream) {
    switch (segment_offsets.getDataType()) {
        case DataType::UINT32:
            launchVectorMinMaxBackwardTyped<ValueT, GradT, uint32_t>(
                op, values, segment_offsets, grad_output, grad_input, batch_size, elements_per_value, stream);
            return;
        case DataType::UINT64:
            launchVectorMinMaxBackwardTyped<ValueT, GradT, uint64_t>(
                op, values, segment_offsets, grad_output, grad_input, batch_size, elements_per_value, stream);
            return;
        default:
            throw std::invalid_argument("Vector segmented min/max backward offsets must use UINT32 or UINT64.");
    }
}

template <typename ValueT>
void dispatchVectorMinMaxBackwardGrad(ExprOp op,
                                      const Tensor& values,
                                      const Tensor& segment_offsets,
                                      const Tensor& grad_output,
                                      Tensor& grad_input,
                                      uint64_t batch_size,
                                      uint64_t elements_per_value,
                                      Stream& stream) {
    switch (grad_output.getDataType()) {
        case DataType::FP8_E4M3:
            dispatchVectorMinMaxBackwardOffsets<ValueT, __nv_fp8_e4m3>(
                op, values, segment_offsets, grad_output, grad_input, batch_size, elements_per_value, stream);
            return;
        case DataType::FP8_E5M2:
            dispatchVectorMinMaxBackwardOffsets<ValueT, __nv_fp8_e5m2>(
                op, values, segment_offsets, grad_output, grad_input, batch_size, elements_per_value, stream);
            return;
        case DataType::FP16:
            dispatchVectorMinMaxBackwardOffsets<ValueT, __half>(
                op, values, segment_offsets, grad_output, grad_input, batch_size, elements_per_value, stream);
            return;
        case DataType::BF16:
            dispatchVectorMinMaxBackwardOffsets<ValueT, __nv_bfloat16>(
                op, values, segment_offsets, grad_output, grad_input, batch_size, elements_per_value, stream);
            return;
        case DataType::FP32:
            dispatchVectorMinMaxBackwardOffsets<ValueT, float>(
                op, values, segment_offsets, grad_output, grad_input, batch_size, elements_per_value, stream);
            return;
        case DataType::FP64:
            dispatchVectorMinMaxBackwardOffsets<ValueT, double>(
                op, values, segment_offsets, grad_output, grad_input, batch_size, elements_per_value, stream);
            return;
        default:
            throw std::invalid_argument(
                "Vector segmented min/max backward requires FP8/FP16/BF16/FP32/FP64 gradients.");
    }
}

}  // namespace

void launchVectorSegmentedReduceMinMaxBackward(ExprOp op,
                                                const Tensor& values,
                                                const Tensor& segment_offsets,
                                                const Tensor& grad_output,
                                                Tensor& grad_input,
                                                uint64_t elements_per_value,
                                                Stream& stream) {
    if (op != ExprOp::SEGMENTED_REDUCE_MIN_BACKWARD && op != ExprOp::SEGMENTED_REDUCE_MAX_BACKWARD) {
        throw std::invalid_argument("Vector segmented min/max backward received an unsupported op.");
    }
    if (elements_per_value <= 1) {
        throw std::invalid_argument("Vector segmented min/max backward requires elements_per_value > 1.");
    }
    if (values.getPlacement() != segment_offsets.getPlacement() || values.getPlacement() != grad_output.getPlacement() ||
        values.getPlacement() != grad_input.getPlacement()) {
        throw std::invalid_argument("Vector segmented min/max backward tensors must share one placement.");
    }
    if (values.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU ||
        values.getPlacement().getDeviceNum() != stream.getGpuNum()) {
        throw std::invalid_argument("Vector segmented min/max backward requires tensors on the execution GPU.");
    }
    if (values.getDimensions().empty() || grad_output.getDimensions().empty() || grad_input.getDimensions().empty() ||
        segment_offsets.getDimensions().size() != 1 || segment_offsets.getDimensions()[0] == 0) {
        throw std::invalid_argument(
            "Vector segmented min/max backward requires values [N,D...], grad [B,D...], output [N,D...], and offsets [B+1].");
    }
    if (grad_output.getDataType() != grad_input.getDataType()) {
        throw std::invalid_argument("Vector segmented min/max backward grad-output and grad-input dtypes must match.");
    }
    if (grad_input.getDimensions() != values.getDimensions()) {
        throw std::invalid_argument("Vector segmented min/max backward output shape must match packed input shape.");
    }

    const uint64_t batch_size = segment_offsets.getDimensions()[0] - 1;
    if (batch_size == 0 || values.getDimensions()[0] == 0 || grad_output.getDimensions()[0] != batch_size) {
        throw std::invalid_argument("Vector segmented min/max backward received invalid batch dimensions.");
    }
    if (values.getTotalNumElements() / values.getDimensions()[0] != elements_per_value ||
        grad_output.getTotalNumElements() / batch_size != elements_per_value) {
        throw std::invalid_argument("Vector segmented min/max backward elements_per_value does not match tensor shapes.");
    }

    switch (values.getDataType()) {
        case DataType::FP8_E4M3:
            dispatchVectorMinMaxBackwardGrad<__nv_fp8_e4m3>(
                op, values, segment_offsets, grad_output, grad_input, batch_size, elements_per_value, stream);
            return;
        case DataType::FP8_E5M2:
            dispatchVectorMinMaxBackwardGrad<__nv_fp8_e5m2>(
                op, values, segment_offsets, grad_output, grad_input, batch_size, elements_per_value, stream);
            return;
        case DataType::FP16:
            dispatchVectorMinMaxBackwardGrad<__half>(
                op, values, segment_offsets, grad_output, grad_input, batch_size, elements_per_value, stream);
            return;
        case DataType::BF16:
            dispatchVectorMinMaxBackwardGrad<__nv_bfloat16>(
                op, values, segment_offsets, grad_output, grad_input, batch_size, elements_per_value, stream);
            return;
        case DataType::FP32:
            dispatchVectorMinMaxBackwardGrad<float>(
                op, values, segment_offsets, grad_output, grad_input, batch_size, elements_per_value, stream);
            return;
        case DataType::FP64:
            dispatchVectorMinMaxBackwardGrad<double>(
                op, values, segment_offsets, grad_output, grad_input, batch_size, elements_per_value, stream);
            return;
        default:
            throw std::invalid_argument("Vector segmented min/max backward requires a supported floating-point values dtype.");
    }
}

}  // namespace ThorImplementation
