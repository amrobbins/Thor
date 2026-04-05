#include "Utilities/TensorMathFusion/ReduceMinMaxBackwardKernel.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ThorImplementation {
namespace {

constexpr int32_t MAX_REDUCE_BACKWARD_RANK = 8;

struct ReduceMinMaxBackwardMeta {
    int32_t input_rank = 0;
    int32_t reduction_rank = 0;
    int64_t input_dims[MAX_REDUCE_BACKWARD_RANK]{};
    int64_t input_strides[MAX_REDUCE_BACKWARD_RANK]{};
    int32_t reduced_axes[MAX_REDUCE_BACKWARD_RANK]{};
    int64_t visible_strides_by_full_axis[MAX_REDUCE_BACKWARD_RANK]{};
};

static std::vector<uint64_t> normalizeAxes(std::vector<uint64_t> axes) {
    std::sort(axes.begin(), axes.end());
    axes.erase(std::unique(axes.begin(), axes.end()), axes.end());
    return axes;
}

template <typename T>
__device__ inline float reduceBwToFloat(T v);

template <>
__device__ inline float reduceBwToFloat<float>(float v) {
    return v;
}

template <>
__device__ inline float reduceBwToFloat<__half>(__half v) {
    return __half2float(v);
}

template <>
__device__ inline float reduceBwToFloat<__nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

template <>
__device__ inline float reduceBwToFloat<__nv_fp8_e4m3>(__nv_fp8_e4m3 v) {
    return __half2float(__half(v));
}

template <>
__device__ inline float reduceBwToFloat<__nv_fp8_e5m2>(__nv_fp8_e5m2 v) {
    return __half2float(__half(v));
}

template <typename T>
__device__ inline T reduceBwFromFloat(float v);

template <>
__device__ inline float reduceBwFromFloat<float>(float v) {
    return v;
}

template <>
__device__ inline __half reduceBwFromFloat<__half>(float v) {
    return __float2half_rn(v);
}

template <>
__device__ inline __nv_bfloat16 reduceBwFromFloat<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

template <>
__device__ inline __nv_fp8_e4m3 reduceBwFromFloat<__nv_fp8_e4m3>(float v) {
    return __nv_fp8_e4m3(__float2half_rn(v));
}

template <>
__device__ inline __nv_fp8_e5m2 reduceBwFromFloat<__nv_fp8_e5m2>(float v) {
    return __nv_fp8_e5m2(__float2half_rn(v));
}

static std::vector<uint64_t> computeVisibleOutputDims(const std::vector<uint64_t>& input_dims,
                                                      const std::vector<uint64_t>& reduction_axes,
                                                      const std::vector<uint64_t>& squeeze_axes,
                                                      std::array<int32_t, MAX_REDUCE_BACKWARD_RANK>& full_to_visible_axis) {
    std::vector<uint64_t> unsqueezed = input_dims;
    for (uint64_t axis : reduction_axes) {
        if (axis >= unsqueezed.size()) {
            throw std::runtime_error("Reduction axis out of range in computeVisibleOutputDims.");
        }
        unsqueezed[axis] = 1;
    }

    full_to_visible_axis.fill(-1);

    if (squeeze_axes.empty()) {
        std::vector<uint64_t> visible = unsqueezed;
        for (int32_t axis = 0, vis = 0; axis < static_cast<int32_t>(unsqueezed.size()); ++axis, ++vis) {
            full_to_visible_axis[axis] = vis;
        }
        return visible;
    }

    std::vector<uint64_t> normalized_squeeze = normalizeAxes(squeeze_axes);
    const bool squeeze_all_singletons = normalized_squeeze.size() == 1 && normalized_squeeze[0] == UINT64_MAX;

    std::vector<uint64_t> visible;
    visible.reserve(unsqueezed.size());

    int32_t vis_axis = 0;
    size_t squeeze_i = 0;
    for (int32_t axis = 0; axis < static_cast<int32_t>(unsqueezed.size()); ++axis) {
        const bool should_squeeze = squeeze_all_singletons ? (unsqueezed[axis] == 1)
                                                           : (squeeze_i < normalized_squeeze.size() &&
                                                              normalized_squeeze[squeeze_i] == static_cast<uint32_t>(axis));
        if (should_squeeze) {
            if (unsqueezed[axis] != 1) {
                throw std::runtime_error("Squeezed axis must be singleton in computeVisibleOutputDims.");
            }
            if (!squeeze_all_singletons && squeeze_i < normalized_squeeze.size() &&
                normalized_squeeze[squeeze_i] == static_cast<uint32_t>(axis)) {
                ++squeeze_i;
            }
            continue;
        }

        visible.push_back(unsqueezed[axis]);
        full_to_visible_axis[axis] = vis_axis++;
    }

    if (!squeeze_all_singletons && squeeze_i != normalized_squeeze.size()) {
        throw std::runtime_error("Squeeze axis out of range in computeVisibleOutputDims.");
    }

    return visible;
}

template <typename GradT, typename OutT>
__global__ void reduceMinMaxBackwardScatterKernel(
    const GradT* grad_output, const uint32_t* arg_indices, OutT* grad_input, ReduceMinMaxBackwardMeta meta, int64_t output_numel) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    if (idx >= output_numel) {
        return;
    }

    int64_t tmp = idx;
    int64_t base_input_offset = 0;
    int64_t grad_output_offset = 0;

    for (int32_t axis = static_cast<int32_t>(meta.input_rank) - 1; axis >= 0; --axis) {
        bool is_reduced = false;
        for (int32_t i = 0; i < meta.reduction_rank; ++i) {
            if (meta.reduced_axes[i] == axis) {
                is_reduced = true;
                break;
            }
        }

        const int64_t output_dim = is_reduced ? 1LL : meta.input_dims[axis];
        const int64_t coord = tmp % output_dim;
        tmp /= output_dim;

        base_input_offset += coord * meta.input_strides[axis];
        grad_output_offset += coord * meta.visible_strides_by_full_axis[axis];
    }

    uint32_t local_index = arg_indices[idx];
    int64_t winner_offset = base_input_offset;
    for (int32_t red_i = static_cast<int32_t>(meta.reduction_rank) - 1; red_i >= 0; --red_i) {
        const int32_t axis = meta.reduced_axes[red_i];
        const int64_t dim = meta.input_dims[axis];
        const int64_t coord = local_index % static_cast<uint32_t>(dim);
        local_index /= static_cast<uint32_t>(dim);
        winner_offset += coord * meta.input_strides[axis];
    }

    const float grad_value = reduceBwToFloat<GradT>(grad_output[grad_output_offset]);
    grad_input[winner_offset] = reduceBwFromFloat<OutT>(grad_value);
}

template <typename GradT, typename OutT>
void launchTypedReduceMinMaxBackwardScatter(const void* grad_output,
                                            const uint32_t* arg_indices,
                                            void* grad_input,
                                            const ReduceMinMaxBackwardMeta& meta,
                                            int64_t output_numel,
                                            cudaStream_t stream) {
    if (output_numel == 0) {
        return;
    }

    constexpr int32_t threads_per_block = 256;
    const int32_t blocks = static_cast<int32_t>((output_numel + threads_per_block - 1) / threads_per_block);
    reduceMinMaxBackwardScatterKernel<GradT, OutT><<<blocks, threads_per_block, 0, stream>>>(
        static_cast<const GradT*>(grad_output), arg_indices, static_cast<OutT*>(grad_input), meta, output_numel);
}

template <typename GradT>
void dispatchReduceMinMaxBackwardScatterOutput(const void* grad_output,
                                               const uint32_t* arg_indices,
                                               void* grad_input,
                                               const ReduceMinMaxBackwardMeta& meta,
                                               int64_t output_numel,
                                               TensorDescriptor::DataType grad_input_dtype,
                                               cudaStream_t stream) {
    switch (grad_input_dtype) {
        case TensorDescriptor::DataType::FP32:
            launchTypedReduceMinMaxBackwardScatter<GradT, float>(grad_output, arg_indices, grad_input, meta, output_numel, stream);
            break;
        case TensorDescriptor::DataType::FP16:
            launchTypedReduceMinMaxBackwardScatter<GradT, __half>(grad_output, arg_indices, grad_input, meta, output_numel, stream);
            break;
        case TensorDescriptor::DataType::BF16:
            launchTypedReduceMinMaxBackwardScatter<GradT, __nv_bfloat16>(grad_output, arg_indices, grad_input, meta, output_numel, stream);
            break;
        case TensorDescriptor::DataType::FP8_E4M3:
            launchTypedReduceMinMaxBackwardScatter<GradT, __nv_fp8_e4m3>(grad_output, arg_indices, grad_input, meta, output_numel, stream);
            break;
        case TensorDescriptor::DataType::FP8_E5M2:
            launchTypedReduceMinMaxBackwardScatter<GradT, __nv_fp8_e5m2>(grad_output, arg_indices, grad_input, meta, output_numel, stream);
            break;
        default:
            throw std::runtime_error("launchReduceMinMaxBackwardScatter received unsupported grad-input dtype.");
    }
}

}  // namespace

void launchReduceMinMaxBackwardScatter(const void* grad_output,
                                       const uint32_t* arg_indices,
                                       void* grad_input,
                                       const std::vector<uint64_t>& input_dims,
                                       const std::vector<uint64_t>& reduction_axes,
                                       const std::vector<uint64_t>& squeeze_axes,
                                       TensorDescriptor::DataType grad_output_dtype,
                                       TensorDescriptor::DataType grad_input_dtype,
                                       cudaStream_t stream) {
    if (input_dims.size() > MAX_REDUCE_BACKWARD_RANK) {
        throw std::runtime_error("launchReduceMinMaxBackwardScatter supports rank <= 8.");
    }

    const std::vector<uint64_t> normalized_reduction_axes = normalizeAxes(reduction_axes);
    ReduceMinMaxBackwardMeta meta{};
    meta.input_rank = static_cast<int32_t>(input_dims.size());
    meta.reduction_rank = static_cast<int32_t>(normalized_reduction_axes.size());

    for (int32_t axis = 0; axis < meta.input_rank; ++axis) {
        if (input_dims[axis] > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            throw std::runtime_error("launchReduceMinMaxBackwardScatter requires int64_t-compatible input dims.");
        }
        meta.input_dims[axis] = static_cast<int64_t>(input_dims[axis]);
    }

    if (meta.input_rank > 0) {
        meta.input_strides[meta.input_rank - 1] = 1;
        for (int32_t axis = static_cast<int32_t>(meta.input_rank) - 2; axis >= 0; --axis) {
            meta.input_strides[axis] = meta.input_strides[axis + 1] * meta.input_dims[axis + 1];
        }
    }

    for (int32_t i = 0; i < meta.reduction_rank; ++i) {
        if (normalized_reduction_axes[static_cast<size_t>(i)] >= static_cast<uint64_t>(meta.input_rank)) {
            throw std::runtime_error("Reduction axis out of range in launchReduceMinMaxBackwardScatter.");
        }
        meta.reduced_axes[i] = static_cast<int32_t>(normalized_reduction_axes[static_cast<size_t>(i)]);
    }

    std::array<int32_t, MAX_REDUCE_BACKWARD_RANK> full_to_visible_axis{};
    const std::vector<uint64_t> visible_dims =
        computeVisibleOutputDims(input_dims, normalized_reduction_axes, squeeze_axes, full_to_visible_axis);

    std::vector<int64_t> visible_strides(visible_dims.size(), 1LL);
    if (!visible_dims.empty()) {
        visible_strides.back() = 1LL;
        for (int32_t axis = static_cast<int32_t>(visible_dims.size()) - 2; axis >= 0; --axis) {
            visible_strides[axis] = visible_strides[axis + 1] * visible_dims[static_cast<size_t>(axis) + 1];
        }
    }

    for (int32_t axis = 0; axis < meta.input_rank; ++axis) {
        const int32_t vis_axis = full_to_visible_axis[axis];
        meta.visible_strides_by_full_axis[axis] = (vis_axis < 0) ? 0LL : visible_strides[vis_axis];
    }

    int64_t output_numel = 1;
    for (int32_t axis = 0; axis < meta.input_rank; ++axis) {
        bool is_reduced = false;
        for (int32_t i = 0; i < meta.reduction_rank; ++i) {
            if (meta.reduced_axes[i] == axis) {
                is_reduced = true;
                break;
            }
        }
        output_numel *= is_reduced ? 1LL : meta.input_dims[axis];
    }

    switch (grad_output_dtype) {
        case TensorDescriptor::DataType::FP32:
            dispatchReduceMinMaxBackwardScatterOutput<float>(
                grad_output, arg_indices, grad_input, meta, output_numel, grad_input_dtype, stream);
            break;
        case TensorDescriptor::DataType::FP16:
            dispatchReduceMinMaxBackwardScatterOutput<__half>(
                grad_output, arg_indices, grad_input, meta, output_numel, grad_input_dtype, stream);
            break;
        case TensorDescriptor::DataType::BF16:
            dispatchReduceMinMaxBackwardScatterOutput<__nv_bfloat16>(
                grad_output, arg_indices, grad_input, meta, output_numel, grad_input_dtype, stream);
            break;
        case TensorDescriptor::DataType::FP8_E4M3:
            dispatchReduceMinMaxBackwardScatterOutput<__nv_fp8_e4m3>(
                grad_output, arg_indices, grad_input, meta, output_numel, grad_input_dtype, stream);
            break;
        case TensorDescriptor::DataType::FP8_E5M2:
            dispatchReduceMinMaxBackwardScatterOutput<__nv_fp8_e5m2>(
                grad_output, arg_indices, grad_input, meta, output_numel, grad_input_dtype, stream);
            break;
        default:
            throw std::runtime_error("launchReduceMinMaxBackwardScatter received unsupported grad-output dtype.");
    }
}

}  // namespace ThorImplementation
