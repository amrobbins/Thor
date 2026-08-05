#include "Utilities/Expression/ReduceMinMaxBackwardKernel.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "Utilities/Common/LowPrecisionFloat.h"
#include "Utilities/Expression/CudaHelpers.h"

namespace ThorImplementation {
namespace {

struct ReduceMinMaxBackwardMeta {
    uint32_t input_rank = 0;
    uint32_t reduction_rank = 0;
    const uint64_t* input_dims = nullptr;
    const uint64_t* input_strides = nullptr;
    const uint64_t* reduced_axes = nullptr;
    const uint64_t* visible_strides_by_full_axis = nullptr;
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
__device__ inline double reduceBwFromFloat<double>(float v) {
    return static_cast<double>(v);
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
    return ThorLowPrecision::toFp8E4M3Satfinite(v);
}

template <>
__device__ inline __nv_fp8_e5m2 reduceBwFromFloat<__nv_fp8_e5m2>(float v) {
    return __nv_fp8_e5m2(__float2half_rn(v));
}

static std::vector<uint64_t> computeVisibleOutputDims(const std::vector<uint64_t>& input_dims,
                                                      const std::vector<uint64_t>& reduction_axes,
                                                      const std::vector<uint64_t>& squeeze_axes,
                                                      std::vector<uint32_t>& full_to_visible_axis) {
    std::vector<uint64_t> unsqueezed = input_dims;
    for (uint64_t axis : reduction_axes) {
        if (axis >= unsqueezed.size()) {
            throw std::runtime_error("Reduction axis out of range in computeVisibleOutputDims.");
        }
        unsqueezed[axis] = 1;
    }

    full_to_visible_axis.assign(input_dims.size(), UINT32_MAX);

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

template <typename GradT, typename OutT>
__global__ void reduceMinMaxBackwardScatterKernel(
    const GradT* grad_output, const uint32_t* arg_indices, OutT* grad_input, ReduceMinMaxBackwardMeta meta, uint64_t output_numel) {
    const uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= output_numel) {
        return;
    }

    uint64_t tmp = idx;
    uint64_t base_input_offset = 0;
    uint64_t grad_output_offset = 0;

    for (uint64_t axis = meta.input_rank; axis-- > 0;) {
        bool is_reduced = false;
        for (uint32_t i = 0; i < meta.reduction_rank; ++i) {
            if (meta.reduced_axes[i] == axis) {
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
    for (uint64_t red_i = meta.reduction_rank; red_i-- > 0;) {
        const uint64_t axis = meta.reduced_axes[red_i];
        const uint64_t dim = meta.input_dims[axis];
        const uint64_t coord = local_index % static_cast<uint32_t>(dim);
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
                                            uint64_t output_numel,
                                            cudaStream_t stream) {
    if (output_numel == 0) {
        return;
    }

    constexpr uint32_t threads_per_block = 256;
    constexpr uint64_t max_blocks = 4096;
    const uint64_t required_blocks = (output_numel + threads_per_block - 1) / threads_per_block;
    const uint32_t blocks = static_cast<uint32_t>(std::min<uint64_t>(required_blocks, max_blocks));
    reduceMinMaxBackwardScatterKernel<GradT, OutT><<<blocks, threads_per_block, 0, stream>>>(
        static_cast<const GradT*>(grad_output), arg_indices, static_cast<OutT*>(grad_input), meta, output_numel);
}

template <typename GradT>
void dispatchReduceMinMaxBackwardScatterOutput(const void* grad_output,
                                               const uint32_t* arg_indices,
                                               void* grad_input,
                                               const ReduceMinMaxBackwardMeta& meta,
                                               uint64_t output_numel,
                                               DataType grad_input_dtype,
                                               cudaStream_t stream) {
    switch (grad_input_dtype) {
        case DataType::FP32:
            launchTypedReduceMinMaxBackwardScatter<GradT, float>(grad_output, arg_indices, grad_input, meta, output_numel, stream);
            break;
        case DataType::FP16:
            launchTypedReduceMinMaxBackwardScatter<GradT, __half>(grad_output, arg_indices, grad_input, meta, output_numel, stream);
            break;
        case DataType::BF16:
            launchTypedReduceMinMaxBackwardScatter<GradT, __nv_bfloat16>(grad_output, arg_indices, grad_input, meta, output_numel, stream);
            break;
        case DataType::FP8_E4M3:
            launchTypedReduceMinMaxBackwardScatter<GradT, __nv_fp8_e4m3>(grad_output, arg_indices, grad_input, meta, output_numel, stream);
            break;
        case DataType::FP8_E5M2:
            launchTypedReduceMinMaxBackwardScatter<GradT, __nv_fp8_e5m2>(grad_output, arg_indices, grad_input, meta, output_numel, stream);
            break;
        default:
            throw std::runtime_error("launchReduceMinMaxBackwardScatter received unsupported grad-input dtype.");
    }
}

template <typename OffsetT, typename StorageT>
__global__ void segmentedReduceMinMaxBackwardActivePrefixZeroKernel(
    const OffsetT* offsets, uint64_t num_segments, StorageT* grad_input, uint64_t grad_input_numel, uint64_t elements_per_value) {
    const uint64_t max_active_values = grad_input_numel / elements_per_value;
    const uint64_t requested_active_values = static_cast<uint64_t>(offsets[num_segments]);
    const uint64_t active_values = requested_active_values < max_active_values ? requested_active_values : max_active_values;
    const uint64_t active_elements = active_values * elements_per_value;
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < active_elements; index += stride) {
        grad_input[index] = reduceBwFromFloat<StorageT>(0.0f);
    }
}

template <typename OffsetT, typename StorageT>
void launchTypedSegmentedReduceMinMaxBackwardActivePrefixZero(const void* segment_offsets,
                                                              uint64_t num_segments,
                                                              void* grad_input,
                                                              uint64_t grad_input_numel,
                                                              uint64_t elements_per_value,
                                                              cudaStream_t stream) {
    if (grad_input_numel == 0) {
        return;
    }
    constexpr uint32_t threads_per_block = 256;
    constexpr uint64_t max_blocks = 4096;
    const uint64_t required_blocks = (grad_input_numel + threads_per_block - 1) / threads_per_block;
    const uint32_t blocks = static_cast<uint32_t>(std::min<uint64_t>(required_blocks, max_blocks));
    segmentedReduceMinMaxBackwardActivePrefixZeroKernel<OffsetT, StorageT>
        <<<blocks, threads_per_block, 0, stream>>>(static_cast<const OffsetT*>(segment_offsets),
                                                   num_segments,
                                                   static_cast<StorageT*>(grad_input),
                                                   grad_input_numel,
                                                   elements_per_value);
    CUDA_CHECK(cudaGetLastError());
}

template <typename OffsetT>
void dispatchSegmentedReduceMinMaxBackwardActivePrefixZeroStorage(const void* segment_offsets,
                                                                  uint64_t num_segments,
                                                                  void* grad_input,
                                                                  uint64_t grad_input_numel,
                                                                  uint64_t elements_per_value,
                                                                  DataType grad_input_dtype,
                                                                  cudaStream_t stream) {
    switch (grad_input_dtype) {
        case DataType::FP8_E4M3:
            launchTypedSegmentedReduceMinMaxBackwardActivePrefixZero<OffsetT, __nv_fp8_e4m3>(
                segment_offsets, num_segments, grad_input, grad_input_numel, elements_per_value, stream);
            return;
        case DataType::FP8_E5M2:
            launchTypedSegmentedReduceMinMaxBackwardActivePrefixZero<OffsetT, __nv_fp8_e5m2>(
                segment_offsets, num_segments, grad_input, grad_input_numel, elements_per_value, stream);
            return;
        case DataType::FP16:
            launchTypedSegmentedReduceMinMaxBackwardActivePrefixZero<OffsetT, __half>(
                segment_offsets, num_segments, grad_input, grad_input_numel, elements_per_value, stream);
            return;
        case DataType::BF16:
            launchTypedSegmentedReduceMinMaxBackwardActivePrefixZero<OffsetT, __nv_bfloat16>(
                segment_offsets, num_segments, grad_input, grad_input_numel, elements_per_value, stream);
            return;
        case DataType::FP32:
            launchTypedSegmentedReduceMinMaxBackwardActivePrefixZero<OffsetT, float>(
                segment_offsets, num_segments, grad_input, grad_input_numel, elements_per_value, stream);
            return;
        case DataType::FP64:
            launchTypedSegmentedReduceMinMaxBackwardActivePrefixZero<OffsetT, double>(
                segment_offsets, num_segments, grad_input, grad_input_numel, elements_per_value, stream);
            return;
        default:
            throw std::runtime_error("Segmented reduce-min/max backward active-prefix zero received unsupported dtype.");
    }
}

template <typename StorageT, typename IndexT>
__global__ void segmentedReduceMinMaxBackwardScatterKernel(
    const StorageT* grad_output, const IndexT* winner_indices, StorageT* grad_input, uint64_t output_numel, uint64_t grad_input_numel) {
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t output_index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; output_index < output_numel;
         output_index += stride) {
        const IndexT winner = winner_indices[output_index];
        if (winner == static_cast<IndexT>(~IndexT{0}) || static_cast<uint64_t>(winner) >= grad_input_numel) {
            continue;
        }
        grad_input[static_cast<uint64_t>(winner)] = grad_output[output_index];
    }
}

template <typename StorageT, typename IndexT>
void launchTypedSegmentedReduceMinMaxBackwardScatter(const void* grad_output,
                                                     const void* winner_indices,
                                                     void* grad_input,
                                                     uint64_t output_numel,
                                                     uint64_t grad_input_numel,
                                                     cudaStream_t stream) {
    if (output_numel == 0) {
        return;
    }
    constexpr uint32_t threads_per_block = 256;
    constexpr uint64_t max_blocks = 4096;
    const uint64_t required_blocks = (output_numel + threads_per_block - 1) / threads_per_block;
    const uint32_t blocks = static_cast<uint32_t>(std::min<uint64_t>(required_blocks, max_blocks));
    segmentedReduceMinMaxBackwardScatterKernel<StorageT, IndexT>
        <<<blocks, threads_per_block, 0, stream>>>(static_cast<const StorageT*>(grad_output),
                                                   static_cast<const IndexT*>(winner_indices),
                                                   static_cast<StorageT*>(grad_input),
                                                   output_numel,
                                                   grad_input_numel);
    CUDA_CHECK(cudaGetLastError());
}

template <typename StorageT>
void dispatchSegmentedReduceMinMaxBackwardScatterIndex(const void* grad_output,
                                                       const void* winner_indices,
                                                       DataType winner_index_dtype,
                                                       void* grad_input,
                                                       uint64_t output_numel,
                                                       uint64_t grad_input_numel,
                                                       cudaStream_t stream) {
    switch (winner_index_dtype) {
        case DataType::UINT32:
            launchTypedSegmentedReduceMinMaxBackwardScatter<StorageT, uint32_t>(
                grad_output, winner_indices, grad_input, output_numel, grad_input_numel, stream);
            return;
        case DataType::UINT64:
            launchTypedSegmentedReduceMinMaxBackwardScatter<StorageT, uint64_t>(
                grad_output, winner_indices, grad_input, output_numel, grad_input_numel, stream);
            return;
        default:
            throw std::runtime_error("Segmented reduce-min/max backward winner indices must be UINT32 or UINT64.");
    }
}

}  // namespace

ReduceMinMaxBackwardScatterPlan prepareReduceMinMaxBackwardScatter(const std::vector<uint64_t>& input_dims,
                                                                   const std::vector<uint64_t>& reduction_axes,
                                                                   const std::vector<uint64_t>& squeeze_axes,
                                                                   const TensorPlacement& placement,
                                                                   const Stream& stream) {
    if (input_dims.empty()) {
        throw std::runtime_error("Reduce-min/max backward scatter requires a non-empty input rank.");
    }
    if (input_dims.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::runtime_error("Reduce-min/max backward scatter rank exceeds the uint32 axis representation limit.");
    }
    if (placement.getMemDevice() != TensorPlacement::MemDevices::GPU || placement.getDeviceNum() != stream.getGpuNum()) {
        throw std::runtime_error("Reduce-min/max backward scatter metadata must be stamped on the execution GPU.");
    }

    const std::vector<uint64_t> normalized_reduction_axes = normalizeAxes(reduction_axes);
    if (normalized_reduction_axes.empty()) {
        throw std::runtime_error("Reduce-min/max backward scatter requires at least one reduction axis.");
    }
    if (normalized_reduction_axes.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::runtime_error("Reduce-min/max backward scatter reduction rank exceeds uint32 limits.");
    }

    const uint32_t input_rank = static_cast<uint32_t>(input_dims.size());
    const uint32_t reduction_rank = static_cast<uint32_t>(normalized_reduction_axes.size());
    std::vector<uint64_t> input_strides(input_dims.size(), 1);
    for (uint64_t dim : input_dims) {
        if (dim == 0) {
            throw std::runtime_error("Reduce-min/max backward scatter does not support zero-sized dimensions.");
        }
    }
    for (int64_t axis = static_cast<int64_t>(input_dims.size()) - 2; axis >= 0; --axis) {
        const size_t index = static_cast<size_t>(axis);
        if (input_strides[index + 1] > std::numeric_limits<uint64_t>::max() / input_dims[index + 1]) {
            throw std::runtime_error("Reduce-min/max backward scatter input strides overflow uint64_t.");
        }
        input_strides[index] = input_strides[index + 1] * input_dims[index + 1];
    }

    for (uint64_t axis : normalized_reduction_axes) {
        if (axis >= input_dims.size()) {
            throw std::runtime_error("Reduction axis out of range in prepareReduceMinMaxBackwardScatter.");
        }
    }

    std::vector<uint32_t> full_to_visible_axis;
    const std::vector<uint64_t> visible_dims =
        computeVisibleOutputDims(input_dims, normalized_reduction_axes, squeeze_axes, full_to_visible_axis);
    std::vector<uint64_t> visible_strides(visible_dims.size(), 1);
    for (int64_t axis = static_cast<int64_t>(visible_dims.size()) - 2; axis >= 0; --axis) {
        const size_t index = static_cast<size_t>(axis);
        if (visible_strides[index + 1] > std::numeric_limits<uint64_t>::max() / visible_dims[index + 1]) {
            throw std::runtime_error("Reduce-min/max backward scatter visible strides overflow uint64_t.");
        }
        visible_strides[index] = visible_strides[index + 1] * visible_dims[index + 1];
    }

    std::vector<uint64_t> visible_strides_by_full_axis(input_dims.size(), 0);
    for (size_t axis = 0; axis < input_dims.size(); ++axis) {
        const uint32_t visible_axis = full_to_visible_axis[axis];
        visible_strides_by_full_axis[axis] = visible_axis == UINT32_MAX ? 0 : visible_strides[visible_axis];
    }

    std::vector<bool> reduced(input_dims.size(), false);
    for (uint64_t axis : normalized_reduction_axes) {
        reduced[axis] = true;
    }
    uint64_t output_numel = 1;
    for (size_t axis = 0; axis < input_dims.size(); ++axis) {
        const uint64_t factor = reduced[axis] ? 1 : input_dims[axis];
        if (output_numel > std::numeric_limits<uint64_t>::max() / factor) {
            throw std::runtime_error("Reduce-min/max backward scatter output size overflows uint64_t.");
        }
        output_numel *= factor;
    }

    std::vector<uint64_t> packed;
    packed.reserve(3 * input_dims.size() + normalized_reduction_axes.size());
    packed.insert(packed.end(), input_dims.begin(), input_dims.end());
    packed.insert(packed.end(), input_strides.begin(), input_strides.end());
    packed.insert(packed.end(), normalized_reduction_axes.begin(), normalized_reduction_axes.end());
    packed.insert(packed.end(), visible_strides_by_full_axis.begin(), visible_strides_by_full_axis.end());

    Tensor host_metadata(TensorPlacement(TensorPlacement::MemDevices::CPU),
                         TensorDescriptor(DataType::UINT64, {static_cast<uint64_t>(packed.size())}));
    std::memcpy(host_metadata.getMemPtr<uint64_t>(), packed.data(), packed.size() * sizeof(uint64_t));
    Tensor device_metadata(placement, TensorDescriptor(DataType::UINT64, {static_cast<uint64_t>(packed.size())}));
    device_metadata.copyFromAsync(host_metadata, stream);
    stream.synchronize();

    return ReduceMinMaxBackwardScatterPlan{
        .metadata = device_metadata,
        .input_rank = input_rank,
        .reduction_rank = reduction_rank,
        .output_numel = output_numel,
    };
}

void launchReduceMinMaxBackwardScatter(const void* grad_output,
                                       const uint32_t* arg_indices,
                                       void* grad_input,
                                       const ReduceMinMaxBackwardScatterPlan& plan,
                                       DataType grad_output_dtype,
                                       DataType grad_input_dtype,
                                       cudaStream_t stream) {
    if (!plan.metadata.isInitialized() || plan.metadata.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU ||
        plan.metadata.getDataType() != DataType::UINT64) {
        throw std::runtime_error("Reduce-min/max backward scatter received invalid stamped metadata.");
    }
    const uint64_t* base = plan.metadata.getMemPtr<uint64_t>();
    const ReduceMinMaxBackwardMeta meta{
        .input_rank = plan.input_rank,
        .reduction_rank = plan.reduction_rank,
        .input_dims = base,
        .input_strides = base + plan.input_rank,
        .reduced_axes = base + 2ULL * plan.input_rank,
        .visible_strides_by_full_axis = base + 2ULL * plan.input_rank + plan.reduction_rank,
    };

    switch (grad_output_dtype) {
        case DataType::FP32:
            dispatchReduceMinMaxBackwardScatterOutput<float>(
                grad_output, arg_indices, grad_input, meta, plan.output_numel, grad_input_dtype, stream);
            break;
        case DataType::FP16:
            dispatchReduceMinMaxBackwardScatterOutput<__half>(
                grad_output, arg_indices, grad_input, meta, plan.output_numel, grad_input_dtype, stream);
            break;
        case DataType::BF16:
            dispatchReduceMinMaxBackwardScatterOutput<__nv_bfloat16>(
                grad_output, arg_indices, grad_input, meta, plan.output_numel, grad_input_dtype, stream);
            break;
        case DataType::FP8_E4M3:
            dispatchReduceMinMaxBackwardScatterOutput<__nv_fp8_e4m3>(
                grad_output, arg_indices, grad_input, meta, plan.output_numel, grad_input_dtype, stream);
            break;
        case DataType::FP8_E5M2:
            dispatchReduceMinMaxBackwardScatterOutput<__nv_fp8_e5m2>(
                grad_output, arg_indices, grad_input, meta, plan.output_numel, grad_input_dtype, stream);
            break;
        default:
            throw std::runtime_error("launchReduceMinMaxBackwardScatter received unsupported grad-output dtype.");
    }
}

void launchSegmentedReduceMinMaxBackwardActivePrefixZero(const void* segment_offsets,
                                                         DataType offset_dtype,
                                                         uint64_t num_segments,
                                                         void* grad_input,
                                                         uint64_t grad_input_numel,
                                                         uint64_t elements_per_value,
                                                         DataType grad_input_dtype,
                                                         cudaStream_t stream) {
    if (segment_offsets == nullptr || grad_input == nullptr) {
        throw std::runtime_error("Segmented reduce-min/max backward active-prefix zero received a null tensor pointer.");
    }
    if (elements_per_value == 0) {
        throw std::runtime_error("Segmented reduce-min/max backward active-prefix zero requires elements_per_value > 0.");
    }
    switch (offset_dtype) {
        case DataType::UINT32:
            dispatchSegmentedReduceMinMaxBackwardActivePrefixZeroStorage<uint32_t>(
                segment_offsets, num_segments, grad_input, grad_input_numel, elements_per_value, grad_input_dtype, stream);
            return;
        case DataType::UINT64:
            dispatchSegmentedReduceMinMaxBackwardActivePrefixZeroStorage<uint64_t>(
                segment_offsets, num_segments, grad_input, grad_input_numel, elements_per_value, grad_input_dtype, stream);
            return;
        default:
            throw std::runtime_error("Segmented reduce-min/max backward offsets must be UINT32 or UINT64.");
    }
}

void launchSegmentedReduceMinMaxBackwardScatter(const void* grad_output,
                                                const void* winner_indices,
                                                DataType winner_index_dtype,
                                                void* grad_input,
                                                uint64_t output_numel,
                                                uint64_t grad_input_numel,
                                                DataType grad_output_dtype,
                                                DataType grad_input_dtype,
                                                cudaStream_t stream) {
    if (grad_output == nullptr || winner_indices == nullptr || grad_input == nullptr) {
        throw std::runtime_error("Segmented reduce-min/max backward scatter received a null tensor pointer.");
    }
    if (grad_output_dtype != grad_input_dtype) {
        throw std::runtime_error("Segmented reduce-min/max backward scatter requires matching gradient dtypes.");
    }
    switch (grad_output_dtype) {
        case DataType::FP8_E4M3:
            dispatchSegmentedReduceMinMaxBackwardScatterIndex<__nv_fp8_e4m3>(
                grad_output, winner_indices, winner_index_dtype, grad_input, output_numel, grad_input_numel, stream);
            return;
        case DataType::FP8_E5M2:
            dispatchSegmentedReduceMinMaxBackwardScatterIndex<__nv_fp8_e5m2>(
                grad_output, winner_indices, winner_index_dtype, grad_input, output_numel, grad_input_numel, stream);
            return;
        case DataType::FP16:
            dispatchSegmentedReduceMinMaxBackwardScatterIndex<__half>(
                grad_output, winner_indices, winner_index_dtype, grad_input, output_numel, grad_input_numel, stream);
            return;
        case DataType::BF16:
            dispatchSegmentedReduceMinMaxBackwardScatterIndex<__nv_bfloat16>(
                grad_output, winner_indices, winner_index_dtype, grad_input, output_numel, grad_input_numel, stream);
            return;
        case DataType::FP32:
            dispatchSegmentedReduceMinMaxBackwardScatterIndex<float>(
                grad_output, winner_indices, winner_index_dtype, grad_input, output_numel, grad_input_numel, stream);
            return;
        case DataType::FP64:
            dispatchSegmentedReduceMinMaxBackwardScatterIndex<double>(
                grad_output, winner_indices, winner_index_dtype, grad_input, output_numel, grad_input_numel, stream);
            return;
        default:
            throw std::runtime_error("Segmented reduce-min/max backward scatter received unsupported gradient dtype.");
    }
}

}  // namespace ThorImplementation
