#include "Utilities/TensorMathFusion/ReduceMinMaxBackwardKernel.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ThorImplementation {
namespace {

constexpr uint32_t MAX_REDUCE_BACKWARD_RANK = 8;

struct ReduceMinMaxBackwardMeta {
    uint32_t input_rank = 0;
    uint32_t reduction_rank = 0;
    uint64_t input_dims[MAX_REDUCE_BACKWARD_RANK]{};
    uint64_t input_strides[MAX_REDUCE_BACKWARD_RANK]{};
    uint32_t reduced_axes[MAX_REDUCE_BACKWARD_RANK]{};
    uint64_t visible_strides_by_full_axis[MAX_REDUCE_BACKWARD_RANK]{};
};

static std::vector<uint64_t> normalizeAxes(std::vector<uint64_t> axes) {
    std::sort(axes.begin(), axes.end());
    axes.erase(std::unique(axes.begin(), axes.end()), axes.end());
    return axes;
}

static std::vector<uint64_t> computeVisibleOutputDims(const std::vector<uint64_t>& input_dims,
                                                      const std::vector<uint64_t>& reduction_axes,
                                                      const std::vector<uint64_t>& squeeze_axes,
                                                      std::array<uint32_t, MAX_REDUCE_BACKWARD_RANK>& full_to_visible_axis) {
    std::vector<uint64_t> unsqueezed = input_dims;
    for (uint64_t axis : reduction_axes) {
        if (axis >= unsqueezed.size()) {
            throw std::runtime_error("Reduction axis out of range in computeVisibleOutputDims.");
        }
        unsqueezed[axis] = 1;
    }

    full_to_visible_axis.fill(UINT32_MAX);

    if (squeeze_axes.empty()) {
        std::vector<uint64_t> visible = unsqueezed;
        for (uint32_t axis = 0, vis = 0; axis < unsqueezed.size(); ++axis, ++vis) {
            full_to_visible_axis[axis] = vis;
        }
        return visible;
    }

    std::vector<uint64_t> normalized_squeeze = normalizeAxes(squeeze_axes);
    const bool squeeze_all_singletons = normalized_squeeze.size() == 1 && normalized_squeeze[0] == UINT64_MAX;

    std::vector<uint64_t> visible;
    visible.reserve(unsqueezed.size());

    uint32_t vis_axis = 0;
    size_t squeeze_i = 0;
    for (uint32_t axis = 0; axis < unsqueezed.size(); ++axis) {
        const bool should_squeeze = squeeze_all_singletons
                                        ? (unsqueezed[axis] == 1)
                                        : (squeeze_i < normalized_squeeze.size() && normalized_squeeze[squeeze_i] == axis);
        if (should_squeeze) {
            if (unsqueezed[axis] != 1) {
                throw std::runtime_error("Squeezed axis must be singleton in computeVisibleOutputDims.");
            }
            if (!squeeze_all_singletons && squeeze_i < normalized_squeeze.size() && normalized_squeeze[squeeze_i] == axis) {
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

template <typename T>
__global__ void reduceMinMaxBackwardScatterKernel(
    const T* grad_output, const uint32_t* arg_indices, T* grad_input, ReduceMinMaxBackwardMeta meta, uint64_t output_numel) {
    const uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= output_numel) {
        return;
    }

    uint64_t tmp = idx;
    uint64_t base_input_offset = 0;
    uint64_t grad_output_offset = 0;

    for (int32_t axis = static_cast<int32_t>(meta.input_rank) - 1; axis >= 0; --axis) {
        bool is_reduced = false;
        for (uint32_t i = 0; i < meta.reduction_rank; ++i) {
            if (meta.reduced_axes[i] == static_cast<uint32_t>(axis)) {
                is_reduced = true;
                break;
            }
        }

        const uint64_t output_dim = is_reduced ? 1ULL : meta.input_dims[axis];
        const uint64_t coord = tmp % output_dim;
        tmp /= output_dim;

        base_input_offset += coord * meta.input_strides[axis];
        grad_output_offset += coord * meta.visible_strides_by_full_axis[axis];
    }

    uint32_t local_index = arg_indices[idx];
    uint64_t winner_offset = base_input_offset;
    for (int32_t red_i = static_cast<int32_t>(meta.reduction_rank) - 1; red_i >= 0; --red_i) {
        const uint32_t axis = meta.reduced_axes[red_i];
        const uint64_t dim = meta.input_dims[axis];
        const uint64_t coord = local_index % static_cast<uint32_t>(dim);
        local_index /= static_cast<uint32_t>(dim);
        winner_offset += coord * meta.input_strides[axis];
    }

    grad_input[winner_offset] = grad_output[grad_output_offset];
}

template <typename T>
void launchTypedReduceMinMaxBackwardScatter(const void* grad_output,
                                            const uint32_t* arg_indices,
                                            void* grad_input,
                                            const ReduceMinMaxBackwardMeta& meta,
                                            uint64_t output_numel,
                                            cudaStream_t stream) {
    if (output_numel == 0) {
        return;
    }

    constexpr uint32_t threads_per_block = 256;
    const uint32_t blocks = static_cast<uint32_t>((output_numel + threads_per_block - 1) / threads_per_block);
    reduceMinMaxBackwardScatterKernel<T><<<blocks, threads_per_block, 0, stream>>>(
        static_cast<const T*>(grad_output), arg_indices, static_cast<T*>(grad_input), meta, output_numel);
}

}  // namespace

void launchReduceMinMaxBackwardScatter(const void* grad_output,
                                       const uint32_t* arg_indices,
                                       void* grad_input,
                                       const std::vector<uint64_t>& input_dims,
                                       const std::vector<uint64_t>& reduction_axes,
                                       const std::vector<uint64_t>& squeeze_axes,
                                       TensorDescriptor::DataType grad_dtype,
                                       cudaStream_t stream) {
    if (input_dims.size() > MAX_REDUCE_BACKWARD_RANK) {
        throw std::runtime_error("launchReduceMinMaxBackwardScatter supports rank <= 8.");
    }

    const std::vector<uint64_t> normalized_reduction_axes = normalizeAxes(reduction_axes);
    ReduceMinMaxBackwardMeta meta{};
    meta.input_rank = static_cast<uint32_t>(input_dims.size());
    meta.reduction_rank = static_cast<uint32_t>(normalized_reduction_axes.size());

    for (uint32_t axis = 0; axis < meta.input_rank; ++axis) {
        meta.input_dims[axis] = input_dims[axis];
    }

    if (meta.input_rank > 0) {
        meta.input_strides[meta.input_rank - 1] = 1;
        for (int32_t axis = static_cast<int32_t>(meta.input_rank) - 2; axis >= 0; --axis) {
            meta.input_strides[axis] = meta.input_strides[axis + 1] * meta.input_dims[axis + 1];
        }
    }

    for (uint32_t i = 0; i < meta.reduction_rank; ++i) {
        if (normalized_reduction_axes[i] >= meta.input_rank) {
            throw std::runtime_error("Reduction axis out of range in launchReduceMinMaxBackwardScatter.");
        }
        meta.reduced_axes[i] = static_cast<uint32_t>(normalized_reduction_axes[i]);
    }

    std::array<uint32_t, MAX_REDUCE_BACKWARD_RANK> full_to_visible_axis{};
    const std::vector<uint64_t> visible_dims =
        computeVisibleOutputDims(input_dims, normalized_reduction_axes, squeeze_axes, full_to_visible_axis);

    std::vector<uint64_t> visible_strides(visible_dims.size(), 1ULL);
    if (!visible_dims.empty()) {
        visible_strides.back() = 1ULL;
        for (int32_t axis = static_cast<int32_t>(visible_dims.size()) - 2; axis >= 0; --axis) {
            visible_strides[axis] = visible_strides[axis + 1] * visible_dims[static_cast<size_t>(axis) + 1];
        }
    }

    for (uint32_t axis = 0; axis < meta.input_rank; ++axis) {
        const uint32_t vis_axis = full_to_visible_axis[axis];
        meta.visible_strides_by_full_axis[axis] = (vis_axis == UINT32_MAX) ? 0ULL : visible_strides[vis_axis];
    }

    uint64_t output_numel = 1;
    for (uint32_t axis = 0; axis < meta.input_rank; ++axis) {
        bool is_reduced = false;
        for (uint32_t i = 0; i < meta.reduction_rank; ++i) {
            if (meta.reduced_axes[i] == axis) {
                is_reduced = true;
                break;
            }
        }
        output_numel *= is_reduced ? 1ULL : meta.input_dims[axis];
    }

    switch (grad_dtype) {
        case TensorDescriptor::DataType::FP32:
            launchTypedReduceMinMaxBackwardScatter<float>(grad_output, arg_indices, grad_input, meta, output_numel, stream);
            break;
        case TensorDescriptor::DataType::FP16:
            launchTypedReduceMinMaxBackwardScatter<__half>(grad_output, arg_indices, grad_input, meta, output_numel, stream);
            break;
        case TensorDescriptor::DataType::BF16:
            launchTypedReduceMinMaxBackwardScatter<__nv_bfloat16>(grad_output, arg_indices, grad_input, meta, output_numel, stream);
            break;
        case TensorDescriptor::DataType::FP8_E4M3:
            launchTypedReduceMinMaxBackwardScatter<__nv_fp8_e4m3>(grad_output, arg_indices, grad_input, meta, output_numel, stream);
            break;
        case TensorDescriptor::DataType::FP8_E5M2:
            launchTypedReduceMinMaxBackwardScatter<__nv_fp8_e5m2>(grad_output, arg_indices, grad_input, meta, output_numel, stream);
            break;
        default:
            throw std::runtime_error("launchReduceMinMaxBackwardScatter received unsupported gradient dtype.");
    }
}

}  // namespace ThorImplementation
