#pragma once

#include "Utilities/Expression/CudaHelpers.h"
#include "Utilities/TensorOperations/Cub/CubReductionIndexing.cuh"
#include "Utilities/TensorOperations/Cub/CubReductionOperation.cuh"

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_segmented_reduce.cuh>
#include <cuda/iterator>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace ThorImplementation::CubReductionInternal {

struct ArgReductionCandidateFp32 {
    uint64_t index;
    float value;
};

template <typename IndexT>
struct DenseArgReductionCandidateFp32 {
    IndexT index;
    float value;
};

static_assert(std::is_trivially_copyable_v<ArgReductionCandidateFp32>);
static_assert(std::is_trivially_copyable_v<DenseArgReductionCandidateFp32<uint32_t>>);
static_assert(std::is_trivially_copyable_v<DenseArgReductionCandidateFp32<uint64_t>>);

struct ArgMinimumCandidateFp32 {
    template <typename CandidateT>
    __host__ __device__ CandidateT operator()(const CandidateT& lhs, const CandidateT& rhs) const {
        const bool lhs_nan = lhs.value != lhs.value;
        const bool rhs_nan = rhs.value != rhs.value;
        if (lhs_nan || rhs_nan) {
            if (lhs_nan && rhs_nan) {
                return lhs.index <= rhs.index ? lhs : rhs;
            }
            return lhs_nan ? lhs : rhs;
        }
        if (lhs.value < rhs.value) {
            return lhs;
        }
        if (rhs.value < lhs.value) {
            return rhs;
        }
        return lhs.index <= rhs.index ? lhs : rhs;
    }
};

struct ArgMaximumCandidateFp32 {
    template <typename CandidateT>
    __host__ __device__ CandidateT operator()(const CandidateT& lhs, const CandidateT& rhs) const {
        const bool lhs_nan = lhs.value != lhs.value;
        const bool rhs_nan = rhs.value != rhs.value;
        if (lhs_nan || rhs_nan) {
            if (lhs_nan && rhs_nan) {
                return lhs.index <= rhs.index ? lhs : rhs;
            }
            return lhs_nan ? lhs : rhs;
        }
        if (lhs.value > rhs.value) {
            return lhs;
        }
        if (rhs.value > lhs.value) {
            return rhs;
        }
        return lhs.index <= rhs.index ? lhs : rhs;
    }
};

inline __host__ __device__ void storeArgIndexAsRuntimeDType(void* output, DataType output_dtype, uint64_t output_index, uint64_t value) {
    switch (output_dtype) {
        case DataType::UINT32:
            static_cast<uint32_t*>(output)[output_index] = static_cast<uint32_t>(value);
            return;
        case DataType::UINT64:
            static_cast<uint64_t*>(output)[output_index] = value;
            return;
        default:
            return;
    }
}

struct StoreArgReductionResultRuntime {
    void* value_output;
    DataType value_output_dtype;
    void* index_output;
    DataType index_output_dtype;

    template <typename OutputIndexT, typename CandidateT>
    __host__ __device__ void operator()(OutputIndexT output_index, CandidateT candidate) const {
        const uint64_t index = static_cast<uint64_t>(output_index);
        if (value_output != nullptr) {
            storeFp32AsRuntimeDType(value_output, value_output_dtype, index, candidate.value);
        }
        if (index_output != nullptr) {
            storeArgIndexAsRuntimeDType(index_output, index_output_dtype, index, static_cast<uint64_t>(candidate.index));
        }
    }
};

inline auto makeRuntimeArgReductionOutputIterator(Tensor* value_output, Tensor* index_output) {
    return cuda::make_tabulate_output_iterator(
        StoreArgReductionResultRuntime{value_output == nullptr ? nullptr : value_output->getMemPtr<void>(),
                                       value_output == nullptr ? DataType::FP32 : value_output->getDataType(),
                                       index_output == nullptr ? nullptr : index_output->getMemPtr<void>(),
                                       index_output == nullptr ? DataType::UINT32 : index_output->getDataType()});
}

inline auto makeRuntimeArgReductionOutputIterator(std::optional<DataType> value_output_dtype,
                                                   std::optional<DataType> index_output_dtype) {
    return cuda::make_tabulate_output_iterator(
        StoreArgReductionResultRuntime{nullptr,
                                       value_output_dtype.value_or(DataType::FP32),
                                       nullptr,
                                       index_output_dtype.value_or(DataType::UINT32)});
}

template <typename InputT>
struct DeviceArgCandidateInput {
    const InputT* input;

    __host__ __device__ ArgReductionCandidateFp32 operator()(int64_t logical_index) const {
        const uint64_t index = static_cast<uint64_t>(logical_index);
        return ArgReductionCandidateFp32{index, ToFp32<InputT>{}(input[index])};
    }
};

template <typename InputT>
struct ContiguousArgCandidateInput {
    const InputT* input;
    uint64_t reduction_size;

    __host__ __device__ ArgReductionCandidateFp32 operator()(int64_t logical_index) const {
        const uint64_t index = static_cast<uint64_t>(logical_index);
        return ArgReductionCandidateFp32{index % reduction_size, ToFp32<InputT>{}(input[index])};
    }
};

template <typename InputT>
struct StridedArgCandidateInput {
    const InputT* input;
    uint64_t reduction_size;
    CubReductionDeviceIndexing indexing;

    __host__ __device__ ArgReductionCandidateFp32 operator()(int64_t logical_index) const {
        const uint64_t index = static_cast<uint64_t>(logical_index);
        const uint64_t output_index = index / reduction_size;
        const uint64_t reduction_index = index - output_index * reduction_size;
        const uint64_t physical_index = mapLogicalReductionIndex(indexing, output_index, reduction_index);
        return ArgReductionCandidateFp32{reduction_index, ToFp32<InputT>{}(input[physical_index])};
    }
};

template <typename InputT>
auto makeDeviceArgCandidateIterator(const Tensor& input) {
    return thrust::make_transform_iterator(thrust::counting_iterator<int64_t>(0),
                                           DeviceArgCandidateInput<InputT>{input.getMemPtr<InputT>()});
}

template <typename InputT>
auto makeContiguousArgCandidateIterator(const Tensor& input, const CubReductionGeometry& geometry) {
    return thrust::make_transform_iterator(thrust::counting_iterator<int64_t>(0),
                                           ContiguousArgCandidateInput<InputT>{input.getMemPtr<InputT>(), geometry.reduction_size});
}

template <typename InputT, typename IndexT>
struct CompactDeviceArgCandidateInput {
    const InputT* input;

    __host__ __device__ DenseArgReductionCandidateFp32<IndexT> operator()(int64_t logical_index) const {
        const uint64_t index = static_cast<uint64_t>(logical_index);
        return DenseArgReductionCandidateFp32<IndexT>{static_cast<IndexT>(index), ToFp32<InputT>{}(input[index])};
    }
};

template <typename InputT, typename IndexT>
struct CompactContiguousArgCandidateInput {
    const InputT* input;
    uint64_t reduction_size;

    __host__ __device__ DenseArgReductionCandidateFp32<IndexT> operator()(int64_t logical_index) const {
        const uint64_t index = static_cast<uint64_t>(logical_index);
        return DenseArgReductionCandidateFp32<IndexT>{static_cast<IndexT>(index % reduction_size), ToFp32<InputT>{}(input[index])};
    }
};

template <typename InputT, typename IndexT>
auto makeCompactDeviceArgCandidateIterator(const Tensor& input) {
    return thrust::make_transform_iterator(thrust::counting_iterator<int64_t>(0),
                                           CompactDeviceArgCandidateInput<InputT, IndexT>{input.getMemPtr<InputT>()});
}

template <typename InputT, typename IndexT>
auto makeCompactContiguousArgCandidateIterator(const Tensor& input, const CubReductionGeometry& geometry) {
    return thrust::make_transform_iterator(
        thrust::counting_iterator<int64_t>(0),
        CompactContiguousArgCandidateInput<InputT, IndexT>{input.getMemPtr<InputT>(), geometry.reduction_size});
}

template <typename InputT, typename IndexT>
auto makeCompactDeviceArgCandidateIterator() {
    return thrust::make_transform_iterator(thrust::counting_iterator<int64_t>(0),
                                           CompactDeviceArgCandidateInput<InputT, IndexT>{nullptr});
}

template <typename InputT, typename IndexT>
auto makeCompactContiguousArgCandidateIterator(const CubReductionGeometry& geometry) {
    return thrust::make_transform_iterator(
        thrust::counting_iterator<int64_t>(0),
        CompactContiguousArgCandidateInput<InputT, IndexT>{nullptr, geometry.reduction_size});
}

template <typename InputT>
auto makeStridedArgCandidateIterator(const Tensor& input, const CubReductionGeometry& geometry) {
    return thrust::make_transform_iterator(
        thrust::counting_iterator<int64_t>(0),
        StridedArgCandidateInput<InputT>{input.getMemPtr<InputT>(), geometry.reduction_size, geometry.device_indexing});
}

template <typename Fn>
decltype(auto) dispatchComposedArgIndexDType(DataType dtype, Fn&& fn) {
    switch (dtype) {
        case DataType::UINT32:
            return fn.template operator()<uint32_t>();
        case DataType::UINT64:
            return fn.template operator()<uint64_t>();
        default:
            throw std::invalid_argument("Composed CUB arg reduction requires UINT32 or UINT64 carried indices.");
    }
}

template <typename InputT, typename IndexT>
struct ComposedDeviceArgCandidateInput {
    const InputT* values;
    const IndexT* carried_indices;
    uint64_t domain_stride;

    __host__ __device__ DenseArgReductionCandidateFp32<IndexT> operator()(int64_t logical_index) const {
        const uint64_t index = static_cast<uint64_t>(logical_index);
        const uint64_t previous = carried_indices == nullptr ? 0 : static_cast<uint64_t>(carried_indices[index]);
        const uint64_t composed = previous + index * domain_stride;
        return DenseArgReductionCandidateFp32<IndexT>{static_cast<IndexT>(composed), ToFp32<InputT>{}(values[index])};
    }
};

template <typename InputT, typename IndexT>
struct ComposedContiguousArgCandidateInput {
    const InputT* values;
    const IndexT* carried_indices;
    uint64_t reduction_size;
    uint64_t domain_stride;

    __host__ __device__ DenseArgReductionCandidateFp32<IndexT> operator()(int64_t logical_index) const {
        const uint64_t index = static_cast<uint64_t>(logical_index);
        const uint64_t local_index = index % reduction_size;
        const uint64_t previous = carried_indices == nullptr ? 0 : static_cast<uint64_t>(carried_indices[index]);
        const uint64_t composed = previous + local_index * domain_stride;
        return DenseArgReductionCandidateFp32<IndexT>{static_cast<IndexT>(composed), ToFp32<InputT>{}(values[index])};
    }
};

template <typename InputT, typename IndexT>
auto makeComposedDeviceArgCandidateIterator(const Tensor& value_input, const Tensor* carried_index_input, uint64_t domain_stride) {
    return thrust::make_transform_iterator(
        thrust::counting_iterator<int64_t>(0),
        ComposedDeviceArgCandidateInput<InputT, IndexT>{value_input.getMemPtr<InputT>(),
                                                        carried_index_input == nullptr ? nullptr : carried_index_input->getMemPtr<IndexT>(),
                                                        domain_stride});
}

template <typename InputT, typename IndexT>
auto makeComposedContiguousArgCandidateIterator(const Tensor& value_input,
                                                const Tensor* carried_index_input,
                                                const CubReductionGeometry& geometry,
                                                uint64_t domain_stride) {
    return thrust::make_transform_iterator(thrust::counting_iterator<int64_t>(0),
                                           ComposedContiguousArgCandidateInput<InputT, IndexT>{
                                               value_input.getMemPtr<InputT>(),
                                               carried_index_input == nullptr ? nullptr : carried_index_input->getMemPtr<IndexT>(),
                                               geometry.reduction_size,
                                               domain_stride});
}

template <typename InputT, typename IndexT>
auto makeComposedDeviceArgCandidateIterator(uint64_t domain_stride) {
    return thrust::make_transform_iterator(
        thrust::counting_iterator<int64_t>(0),
        ComposedDeviceArgCandidateInput<InputT, IndexT>{nullptr, nullptr, domain_stride});
}

template <typename InputT, typename IndexT>
auto makeComposedContiguousArgCandidateIterator(const CubReductionGeometry& geometry, uint64_t domain_stride) {
    return thrust::make_transform_iterator(
        thrust::counting_iterator<int64_t>(0),
        ComposedContiguousArgCandidateInput<InputT, IndexT>{nullptr, nullptr, geometry.reduction_size, domain_stride});
}

// ARG reductions carry both a value and an index for every live accumulator. The dense value reducer can profitably
// carry up to 16 scalar accumulators per lane, but ARG's doubled accumulator state becomes register/dependency heavy
// for long reductions. Keep the exact-width ARG fast path to at most four candidate pairs per lane and scale width
// horizontally across warps/blocks instead.
constexpr int DENSE_ARG_MAX_ITEMS_PER_LANE = 4;
constexpr uint64_t DENSE_ARG_COMPONENTS_PER_BLOCK = static_cast<uint64_t>(TILED_REDUCTION_BLOCK_THREADS) * DENSE_ARG_MAX_ITEMS_PER_LANE;

template <typename IndexT>
[[nodiscard]] __host__ __device__ inline DenseArgReductionCandidateFp32<IndexT> makeDenseArgReductionInit(ArgReductionCandidateFp32 init) {
    return DenseArgReductionCandidateFp32<IndexT>{static_cast<IndexT>(~IndexT{0}), init.value};
}

template <typename InputT, typename IndexT>
[[nodiscard]] __device__ inline DenseArgReductionCandidateFp32<IndexT> loadDenseArgCandidate(
    const InputT* values, const IndexT* carried_indices, uint64_t linear_index, uint64_t local_run_index, uint64_t domain_stride) {
    const uint64_t previous = carried_indices == nullptr ? 0 : static_cast<uint64_t>(carried_indices[linear_index]);
    const uint64_t original_index = previous + local_run_index * domain_stride;
    return DenseArgReductionCandidateFp32<IndexT>{static_cast<IndexT>(original_index), ToFp32<InputT>{}(values[linear_index])};
}

template <typename InputT, typename IndexT, int ItemsPerLane>
__device__ inline void loadVectorizedDenseArgCandidates(const InputT* values,
                                                        const IndexT* carried_indices,
                                                        uint64_t linear_index,
                                                        uint64_t local_run_index,
                                                        uint64_t domain_stride,
                                                        DenseArgReductionCandidateFp32<IndexT> (&candidates)[ItemsPerLane]) {
    const PackedInputValues<InputT, ItemsPerLane> value_packet = loadVectorizedInputPacket<InputT, ItemsPerLane>(values + linear_index);
    PackedInputValues<IndexT, ItemsPerLane> index_packet{};
    if (carried_indices != nullptr) {
        index_packet = loadVectorizedInputPacket<IndexT, ItemsPerLane>(carried_indices + linear_index);
    }
    const uint64_t contribution = local_run_index * domain_stride;
#pragma unroll
    for (int item = 0; item < ItemsPerLane; ++item) {
        const uint64_t previous = carried_indices == nullptr ? 0 : static_cast<uint64_t>(index_packet.values[item]);
        candidates[item] = DenseArgReductionCandidateFp32<IndexT>{static_cast<IndexT>(previous + contribution),
                                                                  ToFp32<InputT>{}(value_packet.values[item])};
    }
}

template <typename IndexT>
__device__ inline void storeDenseArgReductionResult(void* value_output,
                                                    DataType value_output_dtype,
                                                    void* index_output,
                                                    DataType index_output_dtype,
                                                    uint64_t output_index,
                                                    DenseArgReductionCandidateFp32<IndexT> candidate) {
    if (value_output != nullptr) {
        storeFp32AsRuntimeDType(value_output, value_output_dtype, output_index, candidate.value);
    }
    if (index_output != nullptr) {
        storeArgIndexAsRuntimeDType(index_output, index_output_dtype, output_index, static_cast<uint64_t>(candidate.index));
    }
}

// Dense tiled arg reductions normally need only a 32-bit local row index even when the requested output is UINT64.
// Keeping the hot candidate state to {float,uint32_t} materially reduces the register cost of the multi-component
// kernels. UINT64 candidate state is selected only for the exceptional reduction domain that cannot be represented by
// UINT32.
template <typename Fn>
decltype(auto) dispatchDenseArgAccumulatorIndexDType(uint64_t reduction_size, Fn&& fn) {
    if (reduction_size <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        return fn.template operator()<uint32_t>();
    }
    return fn.template operator()<uint64_t>();
}

// ARG-DIRECT-1: dense middle-axis reductions use one deliberately shaped cooperative kernel instead of a collection of
// exact-width special cases.  A warp owns one contiguous component tile for a subset of reduction rows; 1/2/4/8 warps
// cooperate on the same tile for progressively longer reductions.  Cross-warp candidate exchange is exclusively through
// shared memory -- no warp-shuffle reconstruction is used.
//
// The hot global-read path is 16 bytes/thread whenever the storage type permits it.  For awkward row strides, the chosen
// warps-per-tile makes each warp revisit rows at a byte stride divisible by 16.  Consequently that warp has one stable
// row-head alignment for its entire reduction slice.  It peels at most ItemsPerLane-1 scalar components at the row edges;
// the other lanes issue directly-owned, naturally aligned 16-byte packets.  Per-warp ownership is canonicalized once
// through shared memory, after which the combining warp reads components in output order and emits coalesced stores.
constexpr size_t DENSE_ARG_DIRECT_VECTOR_BYTES = 16;

template <typename InputT>
[[nodiscard]] constexpr int denseArgDirectItemsPerLane() {
    static_assert(DENSE_ARG_DIRECT_VECTOR_BYTES % sizeof(InputT) == 0);
    constexpr int items = static_cast<int>(DENSE_ARG_DIRECT_VECTOR_BYTES / sizeof(InputT));
    static_assert(items == 2 || items == 4 || items == 8 || items == 16);
    return items;
}

[[nodiscard]] inline int denseArgWarpsForOutputParallelism(uint64_t output_tiles, uint64_t reduction_size) {
    // Reuse the same launch-sizing target as the successful value reducer: if the output geometry alone cannot expose
    // TARGET_ACTIVE_WARPS, spend otherwise-idle warps cooperatively on the reduction dimension.  This is deliberately
    // tied to actual output parallelism rather than a fixed reduction-length threshold.
    const uint64_t desired_warps = ceilDivideU64(TILED_REDUCTION_TARGET_ACTIVE_WARPS, std::max<uint64_t>(output_tiles, 1));
    int warps = 1;
    while (warps < TILED_REDUCTION_WARPS_PER_BLOCK && static_cast<uint64_t>(warps) < desired_warps &&
           static_cast<uint64_t>(warps) < reduction_size) {
        warps <<= 1;
    }
    return warps;
}

template <typename ElementT>
[[nodiscard]] inline bool denseArgRowStrideIs16ByteAligned(uint64_t inner_size, int warps_per_tile) {
    const uint64_t row_bytes = inner_size * static_cast<uint64_t>(sizeof(ElementT));
    return (row_bytes * static_cast<uint64_t>(warps_per_tile)) % DENSE_ARG_DIRECT_VECTOR_BYTES == 0;
}

template <typename InputT, bool HasCarriedIndex>
[[nodiscard]] inline int chooseAlignedCooperativeArgWarpsPerTile(uint64_t outer_size,
                                                                 uint64_t reduction_size,
                                                                 uint64_t inner_size,
                                                                 uint64_t tile_components) {
    const uint64_t component_tiles = ceilDivideU64(inner_size, tile_components);
    const uint64_t output_tiles = outer_size * component_tiles;
    int warps = denseArgWarpsForOutputParallelism(output_tiles, reduction_size);

    // If each cooperating warp receives at most one row, there is no revisited-row stride to constrain. Otherwise
    // enlarge the power-of-two group until every warp's next row starts at the same 16-byte alignment.  FP32 carried
    // indices have their own stream and must satisfy the same invariant.
    while (warps <= TILED_REDUCTION_WARPS_PER_BLOCK) {
        const bool one_row_per_warp = reduction_size <= static_cast<uint64_t>(warps);
        const bool values_aligned = one_row_per_warp || denseArgRowStrideIs16ByteAligned<InputT>(inner_size, warps);
        bool indices_aligned = true;
        if constexpr (HasCarriedIndex) {
            indices_aligned = one_row_per_warp || denseArgRowStrideIs16ByteAligned<uint32_t>(inner_size, warps);
        }
        if (values_aligned && indices_aligned) {
            return warps;
        }
        warps <<= 1;
    }
    return 0;
}

template <typename ElementT, int ItemsPerLane>
[[nodiscard]] __device__ inline int denseArgAlignedHeadItems(const ElementT* source) {
    static_assert(sizeof(ElementT) * ItemsPerLane == DENSE_ARG_DIRECT_VECTOR_BYTES);
    const uintptr_t misalignment = reinterpret_cast<uintptr_t>(source) & uintptr_t{DENSE_ARG_DIRECT_VECTOR_BYTES - 1};
    if (misalignment == 0) {
        return 0;
    }
    const size_t head_bytes = DENSE_ARG_DIRECT_VECTOR_BYTES - static_cast<size_t>(misalignment);
    return static_cast<int>(head_bytes / sizeof(ElementT));
}

template <int ItemsPerLane>
[[nodiscard]] __device__ inline uint64_t denseArgOwnedComponentOffset(int lane, int item, int head_items) {
    if (head_items == 0) {
        return static_cast<uint64_t>(lane * ItemsPerLane + item);
    }
    if (lane < TILED_REDUCTION_WARP_THREADS - 1) {
        return static_cast<uint64_t>(head_items + lane * ItemsPerLane + item);
    }
    // The final lane owns both bounded scalar edges.  Their total size is exactly ItemsPerLane:
    // head_items prefix values plus ItemsPerLane-head_items suffix values.
    if (item < head_items) {
        return static_cast<uint64_t>(item);
    }
    return static_cast<uint64_t>(head_items + (TILED_REDUCTION_WARP_THREADS - 1) * ItemsPerLane + (item - head_items));
}

template <typename InputT, typename ReductionOpT, int ItemsPerLane, bool HasCarriedIndex>
__global__ void alignedContiguousSegmentArgReductionKernel(const InputT* input,
                                                           const uint32_t* carried_index_input,
                                                           uint64_t domain_stride,
                                                           void* value_output,
                                                           DataType value_output_dtype,
                                                           void* index_output,
                                                           DataType index_output_dtype,
                                                           uint64_t output_elements,
                                                           uint64_t reduction_size,
                                                           ReductionOpT reduction_op,
                                                           ArgReductionCandidateFp32 init) {
    static_assert(sizeof(InputT) * ItemsPerLane == DENSE_ARG_DIRECT_VECTOR_BYTES);
    if constexpr (HasCarriedIndex) {
        static_assert(std::is_same_v<InputT, float>);
        static_assert(ItemsPerLane == 4);
    }

    using CandidateT = DenseArgReductionCandidateFp32<uint32_t>;
    // Keep shared candidate state SoA. Adjacent lanes then touch adjacent 32-bit banks instead of storing an
    // 8-byte AoS candidate per lane, which would introduce avoidable shared-memory bank pressure.
    __shared__ float lane_partial_values[TILED_REDUCTION_BLOCK_THREADS];
    __shared__ uint32_t lane_partial_indices[TILED_REDUCTION_BLOCK_THREADS];

    const int physical_warp = static_cast<int>(threadIdx.x) / TILED_REDUCTION_WARP_THREADS;
    const int lane = static_cast<int>(threadIdx.x) % TILED_REDUCTION_WARP_THREADS;
    const uint64_t block_work_stride = static_cast<uint64_t>(gridDim.x) * static_cast<uint64_t>(TILED_REDUCTION_WARPS_PER_BLOCK);
    const CandidateT init_candidate = makeDenseArgReductionInit<uint32_t>(init);

    for (uint64_t block_work_base = static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(TILED_REDUCTION_WARPS_PER_BLOCK);
         block_work_base < output_elements;
         block_work_base += block_work_stride) {
        const uint64_t segment = block_work_base + static_cast<uint64_t>(physical_warp);
        const bool active = segment < output_elements;
        CandidateT local = init_candidate;

        if (active) {
            const uint64_t segment_base = segment * reduction_size;
            const int aligned_head = denseArgAlignedHeadItems<InputT, ItemsPerLane>(input + segment_base);
            const uint64_t head_count = minU64(static_cast<uint64_t>(aligned_head), reduction_size);

            if (lane == TILED_REDUCTION_WARP_THREADS - 1) {
                for (uint64_t local_index = 0; local_index < head_count; ++local_index) {
                    const uint64_t linear_index = segment_base + local_index;
                    const uint64_t previous = HasCarriedIndex ? static_cast<uint64_t>(carried_index_input[linear_index]) : 0;
                    const CandidateT candidate{static_cast<uint32_t>(previous + local_index * domain_stride),
                                               ToFp32<InputT>{}(input[linear_index])};
                    local = reduction_op(local, candidate);
                }
            }

            const uint64_t bulk_elements =
                ((reduction_size - head_count) / static_cast<uint64_t>(ItemsPerLane)) * static_cast<uint64_t>(ItemsPerLane);
            const uint64_t bulk_packets = bulk_elements / static_cast<uint64_t>(ItemsPerLane);
            for (uint64_t packet = static_cast<uint64_t>(lane); packet < bulk_packets;
                 packet += static_cast<uint64_t>(TILED_REDUCTION_WARP_THREADS)) {
                const uint64_t local_begin = head_count + packet * static_cast<uint64_t>(ItemsPerLane);
                const uint64_t linear_begin = segment_base + local_begin;
                const PackedInputValues<InputT, ItemsPerLane> values =
                    loadVectorizedInputPacket<InputT, ItemsPerLane>(input + linear_begin);
                PackedInputValues<uint32_t, ItemsPerLane> carried{};
                if constexpr (HasCarriedIndex) {
                    carried = loadVectorizedInputPacket<uint32_t, ItemsPerLane>(carried_index_input + linear_begin);
                }
#pragma unroll
                for (int item = 0; item < ItemsPerLane; ++item) {
                    const uint64_t local_index = local_begin + static_cast<uint64_t>(item);
                    const uint64_t previous = HasCarriedIndex ? static_cast<uint64_t>(carried.values[item]) : 0;
                    const CandidateT candidate{static_cast<uint32_t>(previous + local_index * domain_stride),
                                               ToFp32<InputT>{}(values.values[item])};
                    local = reduction_op(local, candidate);
                }
            }

            if (lane == TILED_REDUCTION_WARP_THREADS - 1) {
                const uint64_t tail_begin = head_count + bulk_elements;
                for (uint64_t local_index = tail_begin; local_index < reduction_size; ++local_index) {
                    const uint64_t linear_index = segment_base + local_index;
                    const uint64_t previous = HasCarriedIndex ? static_cast<uint64_t>(carried_index_input[linear_index]) : 0;
                    const CandidateT candidate{static_cast<uint32_t>(previous + local_index * domain_stride),
                                               ToFp32<InputT>{}(input[linear_index])};
                    local = reduction_op(local, candidate);
                }
            }
        }

        lane_partial_values[threadIdx.x] = local.value;
        lane_partial_indices[threadIdx.x] = local.index;
        __syncwarp();

        // Shared-memory tree reduction. This deliberately avoids warp shuffle: candidate exchange remains in the
        // same shared-memory style used by Thor's successful reduction kernels.
        const int warp_base = physical_warp * TILED_REDUCTION_WARP_THREADS;
#pragma unroll
        for (int offset = TILED_REDUCTION_WARP_THREADS / 2; offset != 0; offset >>= 1) {
            if (lane < offset) {
                const int lhs_slot = warp_base + lane;
                const int rhs_slot = lhs_slot + offset;
                const CandidateT lhs{lane_partial_indices[lhs_slot], lane_partial_values[lhs_slot]};
                const CandidateT rhs{lane_partial_indices[rhs_slot], lane_partial_values[rhs_slot]};
                const CandidateT reduced = reduction_op(lhs, rhs);
                lane_partial_values[lhs_slot] = reduced.value;
                lane_partial_indices[lhs_slot] = reduced.index;
            }
            __syncwarp();
        }

        if (active && lane == 0) {
            const CandidateT aggregate{lane_partial_indices[warp_base], lane_partial_values[warp_base]};
            storeDenseArgReductionResult(value_output, value_output_dtype, index_output, index_output_dtype, segment, aggregate);
        }
        __syncwarp();
    }
}

template <typename InputT, typename ReductionOpT, bool HasCarriedIndex>
void launchAlignedContiguousSegmentArgReduction(const InputT* input,
                                                const uint32_t* carried_index_input,
                                                uint64_t domain_stride,
                                                void* value_output,
                                                DataType value_output_dtype,
                                                void* index_output,
                                                DataType index_output_dtype,
                                                const CubReductionGeometry& geometry,
                                                ReductionOpT reduction_op,
                                                ArgReductionCandidateFp32 init,
                                                cudaStream_t stream) {
    constexpr int items_per_lane = denseArgDirectItemsPerLane<InputT>();
    if constexpr (HasCarriedIndex) {
        static_assert(std::is_same_v<InputT, float>);
        static_assert(items_per_lane == 4);
    }
    const uint64_t required_blocks = ceilDivideU64(geometry.output_elements, static_cast<uint64_t>(TILED_REDUCTION_WARPS_PER_BLOCK));
    const unsigned int grid_blocks = static_cast<unsigned int>(std::min<uint64_t>(required_blocks, TILED_REDUCTION_MAX_GRID_BLOCKS));

    alignedContiguousSegmentArgReductionKernel<InputT, ReductionOpT, items_per_lane, HasCarriedIndex>
        <<<grid_blocks, TILED_REDUCTION_BLOCK_THREADS, 0, stream>>>(input,
                                                                    carried_index_input,
                                                                    domain_stride,
                                                                    value_output,
                                                                    value_output_dtype,
                                                                    index_output,
                                                                    index_output_dtype,
                                                                    geometry.output_elements,
                                                                    geometry.reduction_size,
                                                                    reduction_op,
                                                                    init);
    CUDA_CHECK(cudaGetLastError());
}

// Narrow trailing rows need a different memory geometry from the packet-owned wide path.  When inner_size <= 32, a
// physical warp can cover every retained component and use its otherwise-idle lanes to split reduction rows.  The
// successful value reducer already solves the corresponding global-memory problem: peel a bounded number of complete
// rows until the source reaches the strongest useful alignment, stage the contiguous bulk through 16/8/4-byte aligned
// async copies, then consume a bounded suffix directly.  ARG uses the same staging geometry, but carries an original
// index beside every local value and combines row-lane partials through shared memory rather than warp shuffles/CUB.
//
// A composed stage has two equally-sized FP32/UINT32 source streams.  Split the existing 2 KiB-per-warp stage budget
// evenly between them so ARG composition does not silently double the shared-memory footprint or reduce occupancy.
template <bool HasCarriedIndex>
inline constexpr size_t DENSE_ARG_NARROW_VALUE_STAGE_BYTES =
    HasCarriedIndex ? ASYNC_TILED_REDUCTION_STAGE_BYTES_PER_WARP / 2 : ASYNC_TILED_REDUCTION_STAGE_BYTES_PER_WARP;

template <bool HasCarriedIndex>
inline constexpr size_t DENSE_ARG_NARROW_INDEX_STAGE_BYTES = HasCarriedIndex ? ASYNC_TILED_REDUCTION_STAGE_BYTES_PER_WARP / 2 : 0;

template <typename InputT, bool HasCarriedIndex, typename GroupT, typename PipelineT>
__device__ inline void enqueueAlignedNarrowArgStage(GroupT warp,
                                                    PipelineT& pipeline,
                                                    unsigned char* stage_bytes,
                                                    const InputT* input,
                                                    const uint32_t* carried_indices,
                                                    uint64_t outer_index,
                                                    uint64_t row_begin,
                                                    uint64_t row_count,
                                                    uint64_t reduction_size,
                                                    uint64_t inner_size) {
    constexpr size_t value_stage_bytes = DENSE_ARG_NARROW_VALUE_STAGE_BYTES<HasCarriedIndex>;
    constexpr size_t index_stage_bytes = DENSE_ARG_NARROW_INDEX_STAGE_BYTES<HasCarriedIndex>;
    static_assert(value_stage_bytes + index_stage_bytes == ASYNC_TILED_REDUCTION_STAGE_BYTES_PER_WARP);
    if constexpr (HasCarriedIndex) {
        static_assert(std::is_same_v<InputT, float>);
        static_assert(value_stage_bytes == index_stage_bytes);
    }

    const uint64_t linear_begin = (outer_index * reduction_size + row_begin) * inner_size;
    const size_t value_bytes = static_cast<size_t>(row_count * inner_size) * sizeof(InputT);
    pipeline.producer_acquire();
    memcpyAsyncPreferAligned(warp, stage_bytes, input + linear_begin, value_bytes, pipeline);
    if constexpr (HasCarriedIndex) {
        const size_t index_bytes = static_cast<size_t>(row_count * inner_size) * sizeof(uint32_t);
        memcpyAsyncPreferAligned(warp, stage_bytes + value_stage_bytes, carried_indices + linear_begin, index_bytes, pipeline);
    }
    pipeline.producer_commit();
}

template <typename InputT, typename ReductionOpT, int RowLanes, bool HasCarriedIndex>
__device__ inline void reduceDirectNarrowArgRowRange(const InputT* outer_input,
                                                     const uint32_t* outer_carried_indices,
                                                     uint64_t row_begin,
                                                     uint64_t row_end,
                                                     uint64_t inner_size,
                                                     uint64_t component,
                                                     int row_lane,
                                                     uint64_t domain_stride,
                                                     ReductionOpT reduction_op,
                                                     DenseArgReductionCandidateFp32<uint32_t>& local) {
    static_assert(RowLanes == 1 || RowLanes == 2 || RowLanes == 4 || RowLanes == 8 || RowLanes == 16);
    if constexpr (HasCarriedIndex) {
        static_assert(std::is_same_v<InputT, float>);
    }

    const uint64_t first_row = row_begin + static_cast<uint64_t>(row_lane);
    if (first_row >= row_end) {
        return;
    }

    uint64_t linear_index = first_row * inner_size + component;
    const uint64_t linear_stride = static_cast<uint64_t>(RowLanes) * inner_size;
    for (uint64_t row = first_row; row < row_end; row += static_cast<uint64_t>(RowLanes)) {
        const uint64_t previous = HasCarriedIndex ? static_cast<uint64_t>(outer_carried_indices[linear_index]) : 0;
        const DenseArgReductionCandidateFp32<uint32_t> candidate{static_cast<uint32_t>(previous + row * domain_stride),
                                                                 ToFp32<InputT>{}(outer_input[linear_index])};
        local = reduction_op(local, candidate);
        linear_index += linear_stride;
    }
}

template <typename InputT, typename ReductionOpT, int RowLanes, bool HasCarriedIndex>
__global__ void alignedAsyncNarrowTiledArgReductionKernel(const InputT* input,
                                                          const uint32_t* carried_index_input,
                                                          uint64_t domain_stride,
                                                          void* value_output,
                                                          DataType value_output_dtype,
                                                          void* index_output,
                                                          DataType index_output_dtype,
                                                          uint64_t outer_size,
                                                          uint64_t reduction_size,
                                                          uint64_t inner_size,
                                                          uint64_t rows_per_stage,
                                                          size_t bulk_copy_alignment,
                                                          ReductionOpT reduction_op,
                                                          ArgReductionCandidateFp32 init) {
    static_assert(RowLanes == 1 || RowLanes == 2 || RowLanes == 4 || RowLanes == 8 || RowLanes == 16);
    static_assert(TILED_REDUCTION_WARP_THREADS % RowLanes == 0);
    constexpr size_t value_stage_bytes = DENSE_ARG_NARROW_VALUE_STAGE_BYTES<HasCarriedIndex>;
    constexpr size_t index_stage_bytes = DENSE_ARG_NARROW_INDEX_STAGE_BYTES<HasCarriedIndex>;
    constexpr size_t bytes_per_pipeline_stage = value_stage_bytes + index_stage_bytes;
    static_assert(bytes_per_pipeline_stage == ASYNC_TILED_REDUCTION_STAGE_BYTES_PER_WARP);
    static_assert(value_stage_bytes % sizeof(InputT) == 0);
    if constexpr (HasCarriedIndex) {
        static_assert(std::is_same_v<InputT, float>);
        static_assert(value_stage_bytes % sizeof(uint32_t) == 0);
    }

    using CandidateT = DenseArgReductionCandidateFp32<uint32_t>;
    const CandidateT init_candidate = makeDenseArgReductionInit<uint32_t>(init);

    // SoA shared candidate state avoids 8-byte AoS bank conflicts.  The staged global input remains in the dynamic
    // shared buffer below; the two regions are independent and retain the value reducer's fixed 32 KiB staging budget.
    __shared__ float lane_partial_values[TILED_REDUCTION_BLOCK_THREADS];
    __shared__ uint32_t lane_partial_indices[TILED_REDUCTION_BLOCK_THREADS];
#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, ASYNC_TILED_REDUCTION_PIPELINE_STAGES>
        pipeline_states[TILED_REDUCTION_WARPS_PER_BLOCK];
    extern __shared__ __align__(16) unsigned char async_shared_bytes[];

    const auto block = cg::this_thread_block();
    const auto warp = cg::tiled_partition<TILED_REDUCTION_WARP_THREADS>(block);
    const int physical_warp = static_cast<int>(threadIdx.x) / TILED_REDUCTION_WARP_THREADS;
    const int lane = static_cast<int>(threadIdx.x) % TILED_REDUCTION_WARP_THREADS;
    const int component = lane / RowLanes;
    const int row_lane = lane % RowLanes;
    auto pipeline = cuda::make_pipeline(warp, &pipeline_states[physical_warp]);

    unsigned char* warp_shared =
        async_shared_bytes + static_cast<size_t>(physical_warp * ASYNC_TILED_REDUCTION_PIPELINE_STAGES) * bytes_per_pipeline_stage;

    const uint64_t block_work_stride = static_cast<uint64_t>(gridDim.x) * static_cast<uint64_t>(TILED_REDUCTION_WARPS_PER_BLOCK);
    for (uint64_t block_work_base = static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(TILED_REDUCTION_WARPS_PER_BLOCK);
         block_work_base < outer_size;
         block_work_base += block_work_stride) {
        const uint64_t outer_index = block_work_base + static_cast<uint64_t>(physical_warp);
        if (outer_index >= outer_size) {
            continue;
        }

        const bool component_active = static_cast<uint64_t>(component) < inner_size;
        CandidateT local = init_candidate;
        const InputT* outer_input = input + outer_index * reduction_size * inner_size;
        const uint32_t* outer_carried_indices = HasCarriedIndex ? carried_index_input + outer_index * reduction_size * inner_size : nullptr;

        const uint64_t head_rows = chooseAsyncFullRowHeadRows(outer_input, reduction_size, inner_size, bulk_copy_alignment);
        const uint64_t remaining_rows = reduction_size - head_rows;
        const uint64_t async_rows = chooseAsyncFullRowBulkRows<InputT>(remaining_rows, inner_size);
        const uint64_t async_end = head_rows + async_rows;

        uint64_t current_row_begin = head_rows;
        uint64_t current_rows = 0;
        int current_stage = 0;
        if (async_rows != 0) {
            current_rows = minU64(rows_per_stage, async_rows);
            enqueueAlignedNarrowArgStage<InputT, HasCarriedIndex>(warp,
                                                                  pipeline,
                                                                  warp_shared,
                                                                  input,
                                                                  carried_index_input,
                                                                  outer_index,
                                                                  current_row_begin,
                                                                  current_rows,
                                                                  reduction_size,
                                                                  inner_size);
        }

        // The prologue is bounded by the requested copy alignment.  Consume it directly while the first staged bulk
        // copy is in flight, exactly as the successful value reducer does.
        if (component_active) {
            reduceDirectNarrowArgRowRange<InputT, ReductionOpT, RowLanes, HasCarriedIndex>(outer_input,
                                                                                           outer_carried_indices,
                                                                                           0,
                                                                                           head_rows,
                                                                                           inner_size,
                                                                                           static_cast<uint64_t>(component),
                                                                                           row_lane,
                                                                                           domain_stride,
                                                                                           reduction_op,
                                                                                           local);
        }

        if (async_rows != 0) {
            uint64_t next_row = current_row_begin + current_rows;
            while (true) {
                uint64_t next_rows = 0;
                if (next_row < async_end) {
                    next_rows = minU64(rows_per_stage, async_end - next_row);
                    unsigned char* next_stage = warp_shared + static_cast<size_t>(current_stage ^ 1) * bytes_per_pipeline_stage;
                    enqueueAlignedNarrowArgStage<InputT, HasCarriedIndex>(warp,
                                                                          pipeline,
                                                                          next_stage,
                                                                          input,
                                                                          carried_index_input,
                                                                          outer_index,
                                                                          next_row,
                                                                          next_rows,
                                                                          reduction_size,
                                                                          inner_size);
                }

                pipeline.consumer_wait();
                if (component_active) {
                    unsigned char* current_stage_bytes = warp_shared + static_cast<size_t>(current_stage) * bytes_per_pipeline_stage;
                    const InputT* stage_values = reinterpret_cast<const InputT*>(current_stage_bytes);
                    const uint32_t* stage_indices =
                        HasCarriedIndex ? reinterpret_cast<const uint32_t*>(current_stage_bytes + value_stage_bytes) : nullptr;
                    for (uint64_t stage_row = static_cast<uint64_t>(row_lane); stage_row < current_rows;
                         stage_row += static_cast<uint64_t>(RowLanes)) {
                        const uint64_t shared_index = stage_row * inner_size + static_cast<uint64_t>(component);
                        const uint64_t previous = HasCarriedIndex ? static_cast<uint64_t>(stage_indices[shared_index]) : 0;
                        const uint64_t original_index = previous + (current_row_begin + stage_row) * domain_stride;
                        const CandidateT candidate{static_cast<uint32_t>(original_index), ToFp32<InputT>{}(stage_values[shared_index])};
                        local = reduction_op(local, candidate);
                    }
                }
                pipeline.consumer_release();

                if (next_rows == 0) {
                    break;
                }
                current_row_begin = next_row;
                current_rows = next_rows;
                next_row += next_rows;
                current_stage ^= 1;
            }
        }

        if (component_active) {
            reduceDirectNarrowArgRowRange<InputT, ReductionOpT, RowLanes, HasCarriedIndex>(outer_input,
                                                                                           outer_carried_indices,
                                                                                           async_end,
                                                                                           reduction_size,
                                                                                           inner_size,
                                                                                           static_cast<uint64_t>(component),
                                                                                           row_lane,
                                                                                           domain_stride,
                                                                                           reduction_op,
                                                                                           local);
        }

        lane_partial_values[threadIdx.x] = local.value;
        lane_partial_indices[threadIdx.x] = local.index;
        __syncwarp();

        // Each contiguous RowLanes group belongs to one retained component.  Reduce that group with a shared-memory
        // tree, never a shuffle/CUB warp primitive.  All lanes execute every synchronization point.
        const int group_base = physical_warp * TILED_REDUCTION_WARP_THREADS + component * RowLanes;
#pragma unroll
        for (int offset = RowLanes / 2; offset != 0; offset >>= 1) {
            if (row_lane < offset) {
                const int lhs_slot = group_base + row_lane;
                const int rhs_slot = lhs_slot + offset;
                const CandidateT lhs{lane_partial_indices[lhs_slot], lane_partial_values[lhs_slot]};
                const CandidateT rhs{lane_partial_indices[rhs_slot], lane_partial_values[rhs_slot]};
                const CandidateT reduced = reduction_op(lhs, rhs);
                lane_partial_values[lhs_slot] = reduced.value;
                lane_partial_indices[lhs_slot] = reduced.index;
            }
            __syncwarp();
        }

        if (component_active && row_lane == 0) {
            const CandidateT aggregate{lane_partial_indices[group_base], lane_partial_values[group_base]};
            storeDenseArgReductionResult(value_output,
                                         value_output_dtype,
                                         index_output,
                                         index_output_dtype,
                                         outer_index * inner_size + static_cast<uint64_t>(component),
                                         aggregate);
        }
        __syncwarp();
    }
}

template <typename InputT, typename ReductionOpT, int RowLanes, bool HasCarriedIndex>
void launchAlignedAsyncNarrowTiledArgReductionForRowLanes(const InputT* input,
                                                          const uint32_t* carried_index_input,
                                                          uint64_t domain_stride,
                                                          void* value_output,
                                                          DataType value_output_dtype,
                                                          void* index_output,
                                                          DataType index_output_dtype,
                                                          const CubReductionGeometry& geometry,
                                                          ReductionOpT reduction_op,
                                                          ArgReductionCandidateFp32 init,
                                                          cudaStream_t stream) {
    constexpr size_t value_stage_bytes = DENSE_ARG_NARROW_VALUE_STAGE_BYTES<HasCarriedIndex>;
    const AsyncFullRowStagePlan async_plan =
        chooseAsyncFullRowStagePlan<InputT>(geometry.inner_size, geometry.reduction_size, value_stage_bytes);
    const uint64_t required_blocks = ceilDivideU64(geometry.outer_size, static_cast<uint64_t>(TILED_REDUCTION_WARPS_PER_BLOCK));
    const unsigned int grid_blocks = static_cast<unsigned int>(std::min<uint64_t>(required_blocks, TILED_REDUCTION_MAX_GRID_BLOCKS));

    alignedAsyncNarrowTiledArgReductionKernel<InputT, ReductionOpT, RowLanes, HasCarriedIndex>
        <<<grid_blocks, TILED_REDUCTION_BLOCK_THREADS, ASYNC_TILED_REDUCTION_SHARED_BYTES, stream>>>(input,
                                                                                                     carried_index_input,
                                                                                                     domain_stride,
                                                                                                     value_output,
                                                                                                     value_output_dtype,
                                                                                                     index_output,
                                                                                                     index_output_dtype,
                                                                                                     geometry.outer_size,
                                                                                                     geometry.reduction_size,
                                                                                                     geometry.inner_size,
                                                                                                     async_plan.rows_per_stage,
                                                                                                     async_plan.bulk_copy_alignment,
                                                                                                     reduction_op,
                                                                                                     init);
    CUDA_CHECK(cudaGetLastError());
}

template <typename InputT, typename ReductionOpT, bool HasCarriedIndex>
void launchAlignedAsyncNarrowTiledArgReduction(const InputT* input,
                                               const uint32_t* carried_index_input,
                                               uint64_t domain_stride,
                                               void* value_output,
                                               DataType value_output_dtype,
                                               void* index_output,
                                               DataType index_output_dtype,
                                               const CubReductionGeometry& geometry,
                                               ReductionOpT reduction_op,
                                               ArgReductionCandidateFp32 init,
                                               cudaStream_t stream) {
    if (geometry.inner_size == 0 || geometry.inner_size > 32) {
        throw std::logic_error("Narrow ARG-DIRECT-1 tiled reduction requires 1..32 retained components.");
    }
    if constexpr (HasCarriedIndex) {
        static_assert(std::is_same_v<InputT, float>, "Composed narrow ARG-DIRECT-1 stages use FP32 values with UINT32 carried indices.");
    }

    if (geometry.inner_size <= 2) {
        launchAlignedAsyncNarrowTiledArgReductionForRowLanes<InputT, ReductionOpT, 16, HasCarriedIndex>(input,
                                                                                                        carried_index_input,
                                                                                                        domain_stride,
                                                                                                        value_output,
                                                                                                        value_output_dtype,
                                                                                                        index_output,
                                                                                                        index_output_dtype,
                                                                                                        geometry,
                                                                                                        reduction_op,
                                                                                                        init,
                                                                                                        stream);
    } else if (geometry.inner_size <= 4) {
        launchAlignedAsyncNarrowTiledArgReductionForRowLanes<InputT, ReductionOpT, 8, HasCarriedIndex>(input,
                                                                                                       carried_index_input,
                                                                                                       domain_stride,
                                                                                                       value_output,
                                                                                                       value_output_dtype,
                                                                                                       index_output,
                                                                                                       index_output_dtype,
                                                                                                       geometry,
                                                                                                       reduction_op,
                                                                                                       init,
                                                                                                       stream);
    } else if (geometry.inner_size <= 8) {
        launchAlignedAsyncNarrowTiledArgReductionForRowLanes<InputT, ReductionOpT, 4, HasCarriedIndex>(input,
                                                                                                       carried_index_input,
                                                                                                       domain_stride,
                                                                                                       value_output,
                                                                                                       value_output_dtype,
                                                                                                       index_output,
                                                                                                       index_output_dtype,
                                                                                                       geometry,
                                                                                                       reduction_op,
                                                                                                       init,
                                                                                                       stream);
    } else if (geometry.inner_size <= 16) {
        launchAlignedAsyncNarrowTiledArgReductionForRowLanes<InputT, ReductionOpT, 2, HasCarriedIndex>(input,
                                                                                                       carried_index_input,
                                                                                                       domain_stride,
                                                                                                       value_output,
                                                                                                       value_output_dtype,
                                                                                                       index_output,
                                                                                                       index_output_dtype,
                                                                                                       geometry,
                                                                                                       reduction_op,
                                                                                                       init,
                                                                                                       stream);
    } else {
        launchAlignedAsyncNarrowTiledArgReductionForRowLanes<InputT, ReductionOpT, 1, HasCarriedIndex>(input,
                                                                                                       carried_index_input,
                                                                                                       domain_stride,
                                                                                                       value_output,
                                                                                                       value_output_dtype,
                                                                                                       index_output,
                                                                                                       index_output_dtype,
                                                                                                       geometry,
                                                                                                       reduction_op,
                                                                                                       init,
                                                                                                       stream);
    }
}

template <typename InputT, typename ReductionOpT, int ItemsPerLane, int WarpsPerTile, bool HasCarriedIndex>
__global__ void alignedCooperativeTiledArgReductionKernel(const InputT* input,
                                                          const uint32_t* carried_index_input,
                                                          uint64_t domain_stride,
                                                          void* value_output,
                                                          DataType value_output_dtype,
                                                          void* index_output,
                                                          DataType index_output_dtype,
                                                          uint64_t outer_size,
                                                          uint64_t reduction_size,
                                                          uint64_t inner_size,
                                                          ReductionOpT reduction_op,
                                                          ArgReductionCandidateFp32 init) {
    static_assert(WarpsPerTile == 1 || WarpsPerTile == 2 || WarpsPerTile == 4 || WarpsPerTile == 8);
    static_assert(TILED_REDUCTION_WARPS_PER_BLOCK % WarpsPerTile == 0);
    static_assert(sizeof(InputT) * ItemsPerLane == DENSE_ARG_DIRECT_VECTOR_BYTES);
    if constexpr (HasCarriedIndex) {
        static_assert(std::is_same_v<InputT, float>);
        static_assert(ItemsPerLane == 4);
    }

    using CandidateT = DenseArgReductionCandidateFp32<uint32_t>;
    constexpr int groups_per_block = TILED_REDUCTION_WARPS_PER_BLOCK / WarpsPerTile;
    constexpr uint64_t tile_components = static_cast<uint64_t>(TILED_REDUCTION_WARP_THREADS) * static_cast<uint64_t>(ItemsPerLane);

    // Keep one canonical component-ordered partial tile per physical warp, with values and indices in separate 32-bit
    // arrays.  Each warp writes these partials only once after all global reads; canonical order lets the combining warp
    // read adjacent components and issue adjacent/coalesced output stores without any cross-lane reconstruction.
    __shared__ float warp_partial_values[TILED_REDUCTION_WARPS_PER_BLOCK * tile_components];
    __shared__ uint32_t warp_partial_indices[TILED_REDUCTION_WARPS_PER_BLOCK * tile_components];

    const int physical_warp = static_cast<int>(threadIdx.x) / TILED_REDUCTION_WARP_THREADS;
    const int lane = static_cast<int>(threadIdx.x) % TILED_REDUCTION_WARP_THREADS;
    const int group_in_block = physical_warp / WarpsPerTile;
    const int warp_in_tile = physical_warp % WarpsPerTile;

    const uint64_t component_tiles = ceilDivideU64(inner_size, tile_components);
    const uint64_t total_work = outer_size * component_tiles;
    const uint64_t block_work_stride = static_cast<uint64_t>(gridDim.x) * static_cast<uint64_t>(groups_per_block);
    const CandidateT init_candidate = makeDenseArgReductionInit<uint32_t>(init);

    for (uint64_t block_work_base = static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(groups_per_block);
         block_work_base < total_work;
         block_work_base += block_work_stride) {
        const uint64_t work_index = block_work_base + static_cast<uint64_t>(group_in_block);
        const bool work_active = work_index < total_work;
        const uint64_t outer_index = work_active ? work_index / component_tiles : 0;
        const uint64_t component_tile = work_active ? work_index - outer_index * component_tiles : 0;
        const uint64_t tile_begin = component_tile * tile_components;
        const uint64_t tile_width = work_active ? minU64(tile_components, inner_size - tile_begin) : 0;

        CandidateT local[ItemsPerLane];
#pragma unroll
        for (int item = 0; item < ItemsPerLane; ++item) {
            local[item] = init_candidate;
        }

        const uint64_t first_row = static_cast<uint64_t>(warp_in_tile);
        int head_items = 0;
        if (work_active && first_row < reduction_size) {
            uint64_t row_base = ((outer_index * reduction_size + first_row) * inner_size) + tile_begin;
            head_items = denseArgAlignedHeadItems<InputT, ItemsPerLane>(input + row_base);
            if constexpr (HasCarriedIndex) {
                // Both allocations are naturally >=16-byte aligned and both streams are FP32/UINT32, so their row and
                // tile offsets are byte-identical and therefore have the same head.  Do not add a runtime branch here:
                // this is the steady-state path and every thread in the block must reach the shared-memory barriers.
            }

            const uint64_t row_stride = static_cast<uint64_t>(WarpsPerTile) * inner_size;
            uint64_t row = first_row;
            uint64_t contribution = first_row * domain_stride;
            const uint64_t contribution_stride = static_cast<uint64_t>(WarpsPerTile) * domain_stride;

            for (; row < reduction_size;
                 row += static_cast<uint64_t>(WarpsPerTile), row_base += row_stride, contribution += contribution_stride) {
                const bool contiguous_packet_owner = head_items == 0 || lane < TILED_REDUCTION_WARP_THREADS - 1;
                const uint64_t component_base =
                    head_items == 0 ? static_cast<uint64_t>(lane * ItemsPerLane) : static_cast<uint64_t>(head_items + lane * ItemsPerLane);
                const bool complete_vector_packet =
                    contiguous_packet_owner && component_base + static_cast<uint64_t>(ItemsPerLane) <= tile_width;

                if (complete_vector_packet) {
                    const PackedInputValues<InputT, ItemsPerLane> values =
                        loadVectorizedInputPacket<InputT, ItemsPerLane>(input + row_base + component_base);
                    PackedInputValues<uint32_t, ItemsPerLane> carried{};
                    if constexpr (HasCarriedIndex) {
                        carried = loadVectorizedInputPacket<uint32_t, ItemsPerLane>(carried_index_input + row_base + component_base);
                    }
#pragma unroll
                    for (int item = 0; item < ItemsPerLane; ++item) {
                        const uint64_t previous = HasCarriedIndex ? static_cast<uint64_t>(carried.values[item]) : 0;
                        const CandidateT candidate{static_cast<uint32_t>(previous + contribution), ToFp32<InputT>{}(values.values[item])};
                        local[item] = reduction_op(local[item], candidate);
                    }
                } else {
                    // The only scalar work is bounded edge work: the peeled alignment head/suffix, or one partial
                    // packet at the final component tile.  Ownership stays exactly the same as the aligned bulk path,
                    // so no loaded value has to move between lanes.
#pragma unroll
                    for (int item = 0; item < ItemsPerLane; ++item) {
                        const uint64_t component = denseArgOwnedComponentOffset<ItemsPerLane>(lane, item, head_items);
                        if (component < tile_width) {
                            const uint64_t linear_index = row_base + component;
                            const uint64_t previous = HasCarriedIndex ? static_cast<uint64_t>(carried_index_input[linear_index]) : 0;
                            const CandidateT candidate{static_cast<uint32_t>(previous + contribution),
                                                       ToFp32<InputT>{}(input[linear_index])};
                            local[item] = reduction_op(local[item], candidate);
                        }
                    }
                }
            }
        }

        // Every physical warp writes one complete logical tile worth of partials.  Warps with no assigned rows write
        // identity candidates, which keeps the combine path branch-free for short reductions.
#pragma unroll
        for (int item = 0; item < ItemsPerLane; ++item) {
            const uint64_t component = work_active ? denseArgOwnedComponentOffset<ItemsPerLane>(lane, item, head_items) : 0;
            if (work_active && component < tile_width) {
                const uint64_t shared_slot = static_cast<uint64_t>(physical_warp) * tile_components + component;
                warp_partial_values[shared_slot] = local[item].value;
                warp_partial_indices[shared_slot] = local[item].index;
            }
        }
        __syncthreads();

        if (work_active && warp_in_tile == 0) {
            const int first_group_warp = group_in_block * WarpsPerTile;
#pragma unroll
            for (int item = 0; item < ItemsPerLane; ++item) {
                // item-major output ownership makes the active lanes write adjacent retained components on each store
                // instruction.  Candidate partials are already canonicalized by component in shared memory.
                const uint64_t component = static_cast<uint64_t>(item * TILED_REDUCTION_WARP_THREADS + lane);
                if (component < tile_width) {
                    CandidateT aggregate = init_candidate;
#pragma unroll
                    for (int cooperating_warp = 0; cooperating_warp < WarpsPerTile; ++cooperating_warp) {
                        const uint64_t shared_slot =
                            static_cast<uint64_t>(first_group_warp + cooperating_warp) * tile_components + component;
                        aggregate =
                            reduction_op(aggregate, CandidateT{warp_partial_indices[shared_slot], warp_partial_values[shared_slot]});
                    }
                    storeDenseArgReductionResult(value_output,
                                                 value_output_dtype,
                                                 index_output,
                                                 index_output_dtype,
                                                 outer_index * inner_size + tile_begin + component,
                                                 aggregate);
                }
            }
        }
        __syncthreads();
    }
}

template <typename InputT, typename ReductionOpT, int ItemsPerLane, int WarpsPerTile, bool HasCarriedIndex>
void launchAlignedCooperativeTiledArgReductionForWarps(const InputT* input,
                                                       const uint32_t* carried_index_input,
                                                       uint64_t domain_stride,
                                                       void* value_output,
                                                       DataType value_output_dtype,
                                                       void* index_output,
                                                       DataType index_output_dtype,
                                                       const CubReductionGeometry& geometry,
                                                       ReductionOpT reduction_op,
                                                       ArgReductionCandidateFp32 init,
                                                       cudaStream_t stream) {
    constexpr int groups_per_block = TILED_REDUCTION_WARPS_PER_BLOCK / WarpsPerTile;
    constexpr uint64_t tile_components = static_cast<uint64_t>(TILED_REDUCTION_WARP_THREADS) * static_cast<uint64_t>(ItemsPerLane);
    const uint64_t component_tiles = ceilDivideU64(geometry.inner_size, tile_components);
    const uint64_t total_work = geometry.outer_size * component_tiles;
    const uint64_t required_blocks = ceilDivideU64(total_work, static_cast<uint64_t>(groups_per_block));
    const unsigned int grid_blocks = static_cast<unsigned int>(std::min<uint64_t>(required_blocks, TILED_REDUCTION_MAX_GRID_BLOCKS));

    alignedCooperativeTiledArgReductionKernel<InputT, ReductionOpT, ItemsPerLane, WarpsPerTile, HasCarriedIndex>
        <<<grid_blocks, TILED_REDUCTION_BLOCK_THREADS, 0, stream>>>(input,
                                                                    carried_index_input,
                                                                    domain_stride,
                                                                    value_output,
                                                                    value_output_dtype,
                                                                    index_output,
                                                                    index_output_dtype,
                                                                    geometry.outer_size,
                                                                    geometry.reduction_size,
                                                                    geometry.inner_size,
                                                                    reduction_op,
                                                                    init);
    CUDA_CHECK(cudaGetLastError());
}

template <typename InputT, typename ReductionOpT, bool HasCarriedIndex>
void launchAlignedCooperativeTiledArgReduction(const InputT* input,
                                               const uint32_t* carried_index_input,
                                               uint64_t domain_stride,
                                               void* value_output,
                                               DataType value_output_dtype,
                                               void* index_output,
                                               DataType index_output_dtype,
                                               const CubReductionGeometry& geometry,
                                               ReductionOpT reduction_op,
                                               ArgReductionCandidateFp32 init,
                                               cudaStream_t stream) {
    constexpr int items_per_lane = denseArgDirectItemsPerLane<InputT>();
    if constexpr (HasCarriedIndex) {
        static_assert(std::is_same_v<InputT, float>, "Composed ARG-DIRECT-1 stages use FP32 values with UINT32 carried indices.");
        static_assert(items_per_lane == 4);
    }

    constexpr uint64_t tile_components = static_cast<uint64_t>(TILED_REDUCTION_WARP_THREADS) * static_cast<uint64_t>(items_per_lane);
    const int warps_per_tile = chooseAlignedCooperativeArgWarpsPerTile<InputT, HasCarriedIndex>(
        geometry.outer_size, geometry.reduction_size, geometry.inner_size, tile_components);
    switch (warps_per_tile) {
        case 1:
            launchAlignedCooperativeTiledArgReductionForWarps<InputT, ReductionOpT, items_per_lane, 1, HasCarriedIndex>(input,
                                                                                                                        carried_index_input,
                                                                                                                        domain_stride,
                                                                                                                        value_output,
                                                                                                                        value_output_dtype,
                                                                                                                        index_output,
                                                                                                                        index_output_dtype,
                                                                                                                        geometry,
                                                                                                                        reduction_op,
                                                                                                                        init,
                                                                                                                        stream);
            return;
        case 2:
            launchAlignedCooperativeTiledArgReductionForWarps<InputT, ReductionOpT, items_per_lane, 2, HasCarriedIndex>(input,
                                                                                                                        carried_index_input,
                                                                                                                        domain_stride,
                                                                                                                        value_output,
                                                                                                                        value_output_dtype,
                                                                                                                        index_output,
                                                                                                                        index_output_dtype,
                                                                                                                        geometry,
                                                                                                                        reduction_op,
                                                                                                                        init,
                                                                                                                        stream);
            return;
        case 4:
            launchAlignedCooperativeTiledArgReductionForWarps<InputT, ReductionOpT, items_per_lane, 4, HasCarriedIndex>(input,
                                                                                                                        carried_index_input,
                                                                                                                        domain_stride,
                                                                                                                        value_output,
                                                                                                                        value_output_dtype,
                                                                                                                        index_output,
                                                                                                                        index_output_dtype,
                                                                                                                        geometry,
                                                                                                                        reduction_op,
                                                                                                                        init,
                                                                                                                        stream);
            return;
        case 8:
            launchAlignedCooperativeTiledArgReductionForWarps<InputT, ReductionOpT, items_per_lane, 8, HasCarriedIndex>(input,
                                                                                                                        carried_index_input,
                                                                                                                        domain_stride,
                                                                                                                        value_output,
                                                                                                                        value_output_dtype,
                                                                                                                        index_output,
                                                                                                                        index_output_dtype,
                                                                                                                        geometry,
                                                                                                                        reduction_op,
                                                                                                                        init,
                                                                                                                        stream);
            return;
        default:
            throw std::logic_error("ARG-DIRECT-1 selected an unsupported cooperative warp count.");
    }
}

template <typename InputT, typename ReductionOpT, typename IndexT, int ItemsPerLane>
__global__ void vectorizedDirectFullRowArgReductionKernel(const InputT* input,
                                                          const IndexT* carried_index_input,
                                                          uint64_t domain_stride,
                                                          void* value_output,
                                                          DataType value_output_dtype,
                                                          void* index_output,
                                                          DataType index_output_dtype,
                                                          uint64_t outer_size,
                                                          uint64_t reduction_size,
                                                          uint64_t inner_size,
                                                          ReductionOpT reduction_op,
                                                          ArgReductionCandidateFp32 init) {
    static_assert(ItemsPerLane == 2 || ItemsPerLane == 4 || ItemsPerLane == 8 || ItemsPerLane == 16);
    constexpr uint64_t expected_inner_size = TILED_REDUCTION_WARP_THREADS * ItemsPerLane;
    using CandidateT = DenseArgReductionCandidateFp32<IndexT>;

    const int physical_warp = static_cast<int>(threadIdx.x) / TILED_REDUCTION_WARP_THREADS;
    const int lane = static_cast<int>(threadIdx.x) % TILED_REDUCTION_WARP_THREADS;
    const uint64_t block_work_stride = static_cast<uint64_t>(gridDim.x) * static_cast<uint64_t>(TILED_REDUCTION_WARPS_PER_BLOCK);

    if (inner_size != expected_inner_size) {
        return;
    }

    for (uint64_t block_work_base = static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(TILED_REDUCTION_WARPS_PER_BLOCK);
         block_work_base < outer_size;
         block_work_base += block_work_stride) {
        const uint64_t outer_index = block_work_base + static_cast<uint64_t>(physical_warp);
        if (outer_index >= outer_size) {
            continue;
        }

        CandidateT local[ItemsPerLane];
#pragma unroll
        for (int item = 0; item < ItemsPerLane; ++item) {
            local[item] = makeDenseArgReductionInit<IndexT>(init);
        }

        uint64_t row_base = outer_index * reduction_size * inner_size;
        const uint64_t lane_component_begin = static_cast<uint64_t>(lane * ItemsPerLane);
        for (uint64_t row = 0; row < reduction_size; ++row) {
            CandidateT candidates[ItemsPerLane];
            loadVectorizedDenseArgCandidates<InputT, IndexT, ItemsPerLane>(
                input, carried_index_input, row_base + lane_component_begin, row, domain_stride, candidates);
#pragma unroll
            for (int item = 0; item < ItemsPerLane; ++item) {
                local[item] = reduction_op(local[item], candidates[item]);
            }
            row_base += inner_size;
        }

#pragma unroll
        for (int item = 0; item < ItemsPerLane; ++item) {
            const uint64_t component = lane_component_begin + static_cast<uint64_t>(item);
            storeDenseArgReductionResult(
                value_output, value_output_dtype, index_output, index_output_dtype, outer_index * inner_size + component, local[item]);
        }
    }
}

template <typename InputT, typename ReductionOpT, typename IndexT, int ItemsPerLane>
void launchVectorizedDirectFullRowArgReduction(const InputT* input,
                                               const IndexT* carried_index_input,
                                               uint64_t domain_stride,
                                               void* value_output,
                                               DataType value_output_dtype,
                                               void* index_output,
                                               DataType index_output_dtype,
                                               const CubReductionGeometry& geometry,
                                               ReductionOpT reduction_op,
                                               ArgReductionCandidateFp32 init,
                                               cudaStream_t stream) {
    const uint64_t required_blocks = ceilDivideU64(geometry.outer_size, static_cast<uint64_t>(TILED_REDUCTION_WARPS_PER_BLOCK));
    const unsigned int grid_blocks = static_cast<unsigned int>(std::min<uint64_t>(required_blocks, TILED_REDUCTION_MAX_GRID_BLOCKS));

    vectorizedDirectFullRowArgReductionKernel<InputT, ReductionOpT, IndexT, ItemsPerLane>
        <<<grid_blocks, TILED_REDUCTION_BLOCK_THREADS, 0, stream>>>(input,
                                                                    carried_index_input,
                                                                    domain_stride,
                                                                    value_output,
                                                                    value_output_dtype,
                                                                    index_output,
                                                                    index_output_dtype,
                                                                    geometry.outer_size,
                                                                    geometry.reduction_size,
                                                                    geometry.inner_size,
                                                                    reduction_op,
                                                                    init);
    CUDA_CHECK(cudaGetLastError());
}

template <typename InputT, typename ReductionOpT, typename IndexT, int WarpsPerOutput, int ItemsPerLane>
__global__ void vectorizedDirectGroupedFullRowArgReductionKernel(const InputT* input,
                                                                 const IndexT* carried_index_input,
                                                                 uint64_t domain_stride,
                                                                 void* value_output,
                                                                 DataType value_output_dtype,
                                                                 void* index_output,
                                                                 DataType index_output_dtype,
                                                                 uint64_t outer_size,
                                                                 uint64_t reduction_size,
                                                                 uint64_t inner_size,
                                                                 ReductionOpT reduction_op,
                                                                 ArgReductionCandidateFp32 init) {
    static_assert(WarpsPerOutput == 2 || WarpsPerOutput == 4 || WarpsPerOutput == 8);
    static_assert(ItemsPerLane == DENSE_ARG_MAX_ITEMS_PER_LANE);
    static_assert(TILED_REDUCTION_WARPS_PER_BLOCK % WarpsPerOutput == 0);
    using CandidateT = DenseArgReductionCandidateFp32<IndexT>;

    constexpr int group_threads = TILED_REDUCTION_WARP_THREADS * WarpsPerOutput;
    constexpr int groups_per_block = TILED_REDUCTION_WARPS_PER_BLOCK / WarpsPerOutput;
    constexpr uint64_t expected_inner_size = static_cast<uint64_t>(group_threads) * static_cast<uint64_t>(ItemsPerLane);

    if (inner_size != expected_inner_size) {
        return;
    }

    const int group_index = static_cast<int>(threadIdx.x) / group_threads;
    const int group_lane = static_cast<int>(threadIdx.x) % group_threads;
    const uint64_t block_work_stride = static_cast<uint64_t>(gridDim.x) * static_cast<uint64_t>(groups_per_block);

    for (uint64_t block_work_base = static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(groups_per_block);
         block_work_base < outer_size;
         block_work_base += block_work_stride) {
        const uint64_t outer_index = block_work_base + static_cast<uint64_t>(group_index);
        if (outer_index >= outer_size) {
            continue;
        }

        CandidateT local[ItemsPerLane];
#pragma unroll
        for (int item = 0; item < ItemsPerLane; ++item) {
            local[item] = makeDenseArgReductionInit<IndexT>(init);
        }

        uint64_t row_base = outer_index * reduction_size * inner_size;
        const uint64_t component_begin = static_cast<uint64_t>(group_lane * ItemsPerLane);
        for (uint64_t row = 0; row < reduction_size; ++row) {
            CandidateT candidates[ItemsPerLane];
            loadVectorizedDenseArgCandidates<InputT, IndexT, ItemsPerLane>(
                input, carried_index_input, row_base + component_begin, row, domain_stride, candidates);
#pragma unroll
            for (int item = 0; item < ItemsPerLane; ++item) {
                local[item] = reduction_op(local[item], candidates[item]);
            }
            row_base += inner_size;
        }

#pragma unroll
        for (int item = 0; item < ItemsPerLane; ++item) {
            const uint64_t component = component_begin + static_cast<uint64_t>(item);
            storeDenseArgReductionResult(
                value_output, value_output_dtype, index_output, index_output_dtype, outer_index * inner_size + component, local[item]);
        }
    }
}

template <typename InputT, typename ReductionOpT, typename IndexT, int WarpsPerOutput, int ItemsPerLane>
void launchVectorizedDirectGroupedFullRowArgReduction(const InputT* input,
                                                      const IndexT* carried_index_input,
                                                      uint64_t domain_stride,
                                                      void* value_output,
                                                      DataType value_output_dtype,
                                                      void* index_output,
                                                      DataType index_output_dtype,
                                                      const CubReductionGeometry& geometry,
                                                      ReductionOpT reduction_op,
                                                      ArgReductionCandidateFp32 init,
                                                      cudaStream_t stream) {
    static_assert(TILED_REDUCTION_WARPS_PER_BLOCK % WarpsPerOutput == 0);
    constexpr int groups_per_block = TILED_REDUCTION_WARPS_PER_BLOCK / WarpsPerOutput;
    const uint64_t required_blocks = ceilDivideU64(geometry.outer_size, static_cast<uint64_t>(groups_per_block));
    const unsigned int grid_blocks = static_cast<unsigned int>(std::min<uint64_t>(required_blocks, TILED_REDUCTION_MAX_GRID_BLOCKS));

    vectorizedDirectGroupedFullRowArgReductionKernel<InputT, ReductionOpT, IndexT, WarpsPerOutput, ItemsPerLane>
        <<<grid_blocks, TILED_REDUCTION_BLOCK_THREADS, 0, stream>>>(input,
                                                                    carried_index_input,
                                                                    domain_stride,
                                                                    value_output,
                                                                    value_output_dtype,
                                                                    index_output,
                                                                    index_output_dtype,
                                                                    geometry.outer_size,
                                                                    geometry.reduction_size,
                                                                    geometry.inner_size,
                                                                    reduction_op,
                                                                    init);
    CUDA_CHECK(cudaGetLastError());
}

template <typename InputT, typename ReductionOpT, typename IndexT, int ItemsPerLane>
__global__ void vectorizedDirectBlockShardedArgReductionKernel(const InputT* input,
                                                               const IndexT* carried_index_input,
                                                               uint64_t domain_stride,
                                                               void* value_output,
                                                               DataType value_output_dtype,
                                                               void* index_output,
                                                               DataType index_output_dtype,
                                                               uint64_t outer_size,
                                                               uint64_t reduction_size,
                                                               uint64_t inner_size,
                                                               ReductionOpT reduction_op,
                                                               ArgReductionCandidateFp32 init) {
    static_assert(ItemsPerLane == DENSE_ARG_MAX_ITEMS_PER_LANE);
    using CandidateT = DenseArgReductionCandidateFp32<IndexT>;
    constexpr uint64_t components_per_block = static_cast<uint64_t>(TILED_REDUCTION_BLOCK_THREADS) * static_cast<uint64_t>(ItemsPerLane);
    static_assert(components_per_block == DENSE_ARG_COMPONENTS_PER_BLOCK);

    if (inner_size % components_per_block != 0) {
        return;
    }

    const uint64_t component_shards = inner_size / components_per_block;
    const uint64_t total_work = outer_size * component_shards;
    const uint64_t grid_stride = static_cast<uint64_t>(gridDim.x);

    for (uint64_t work_index = static_cast<uint64_t>(blockIdx.x); work_index < total_work; work_index += grid_stride) {
        const uint64_t outer_index = work_index / component_shards;
        const uint64_t shard_index = work_index - outer_index * component_shards;
        const uint64_t shard_begin = shard_index * components_per_block;
        const uint64_t component_begin = shard_begin + static_cast<uint64_t>(threadIdx.x) * static_cast<uint64_t>(ItemsPerLane);

        CandidateT local[ItemsPerLane];
#pragma unroll
        for (int item = 0; item < ItemsPerLane; ++item) {
            local[item] = makeDenseArgReductionInit<IndexT>(init);
        }

        uint64_t row_base = outer_index * reduction_size * inner_size + component_begin;
        for (uint64_t row = 0; row < reduction_size; ++row) {
            CandidateT candidates[ItemsPerLane];
            loadVectorizedDenseArgCandidates<InputT, IndexT, ItemsPerLane>(
                input, carried_index_input, row_base, row, domain_stride, candidates);
#pragma unroll
            for (int item = 0; item < ItemsPerLane; ++item) {
                local[item] = reduction_op(local[item], candidates[item]);
            }
            row_base += inner_size;
        }

#pragma unroll
        for (int item = 0; item < ItemsPerLane; ++item) {
            const uint64_t component = component_begin + static_cast<uint64_t>(item);
            storeDenseArgReductionResult(
                value_output, value_output_dtype, index_output, index_output_dtype, outer_index * inner_size + component, local[item]);
        }
    }
}

template <typename InputT, typename ReductionOpT, typename IndexT, int ItemsPerLane>
void launchVectorizedDirectBlockShardedArgReduction(const InputT* input,
                                                    const IndexT* carried_index_input,
                                                    uint64_t domain_stride,
                                                    void* value_output,
                                                    DataType value_output_dtype,
                                                    void* index_output,
                                                    DataType index_output_dtype,
                                                    const CubReductionGeometry& geometry,
                                                    ReductionOpT reduction_op,
                                                    ArgReductionCandidateFp32 init,
                                                    cudaStream_t stream) {
    static_assert(ItemsPerLane == DENSE_ARG_MAX_ITEMS_PER_LANE);
    constexpr uint64_t components_per_block = static_cast<uint64_t>(TILED_REDUCTION_BLOCK_THREADS) * static_cast<uint64_t>(ItemsPerLane);
    if (geometry.inner_size % components_per_block != 0) {
        throw std::logic_error("Vectorized block-sharded arg reduction requires an exact 1024-component shard width.");
    }
    const uint64_t component_shards = geometry.inner_size / components_per_block;
    const uint64_t total_work = geometry.outer_size * component_shards;
    const unsigned int grid_blocks = static_cast<unsigned int>(std::min<uint64_t>(total_work, TILED_REDUCTION_MAX_GRID_BLOCKS));

    vectorizedDirectBlockShardedArgReductionKernel<InputT, ReductionOpT, IndexT, ItemsPerLane>
        <<<grid_blocks, TILED_REDUCTION_BLOCK_THREADS, 0, stream>>>(input,
                                                                    carried_index_input,
                                                                    domain_stride,
                                                                    value_output,
                                                                    value_output_dtype,
                                                                    index_output,
                                                                    index_output_dtype,
                                                                    geometry.outer_size,
                                                                    geometry.reduction_size,
                                                                    geometry.inner_size,
                                                                    reduction_op,
                                                                    init);
    CUDA_CHECK(cudaGetLastError());
}

template <typename InputT, int ItemsPerLane, int ElementOffset>
[[nodiscard]] __device__ inline PackedInputValues<InputT, ItemsPerLane> loadPacketAlignedArgInputPacket(const InputT* source) {
    constexpr size_t packet_bytes = sizeof(InputT) * ItemsPerLane;
    static_assert(ItemsPerLane == DENSE_ARG_MAX_ITEMS_PER_LANE);
    static_assert(packet_bytes == 4 || packet_bytes == 8);
    static_assert(ElementOffset >= 0 && ElementOffset < ItemsPerLane);

    if constexpr (ElementOffset == 0) {
        return loadVectorizedInputPacket<InputT, ItemsPerLane>(source);
    } else {
        using RawPacketT = RawVectorPacket<packet_bytes>;
        using PacketValuesT = PackedInputValues<InputT, ItemsPerLane>;
        static_assert(sizeof(RawPacketT) == sizeof(PacketValuesT));

        const uintptr_t source_address = reinterpret_cast<uintptr_t>(source);
        const uintptr_t aligned_address = source_address & ~uintptr_t{packet_bytes - 1};
        const auto* aligned_source = reinterpret_cast<const InputT*>(aligned_address);

        // Keep the two global accesses naturally aligned to the logical packet width. Do not combine them into a
        // wider load: the first address is guaranteed packet-aligned, not necessarily 8/16-byte aligned.
        const RawPacketT raw_lo = *reinterpret_cast<const RawPacketT*>(aligned_source);
        const RawPacketT raw_hi = *reinterpret_cast<const RawPacketT*>(aligned_source + ItemsPerLane);
        const PacketValuesT lo = cuda::std::bit_cast<PacketValuesT>(raw_lo);
        const PacketValuesT hi = cuda::std::bit_cast<PacketValuesT>(raw_hi);

        PackedInputValues<InputT, ItemsPerLane> values;
        if constexpr (ElementOffset == 1) {
            values.values[0] = lo.values[1];
            values.values[1] = lo.values[2];
            values.values[2] = lo.values[3];
            values.values[3] = hi.values[0];
        } else if constexpr (ElementOffset == 2) {
            values.values[0] = lo.values[2];
            values.values[1] = lo.values[3];
            values.values[2] = hi.values[0];
            values.values[3] = hi.values[1];
        } else {
            static_assert(ElementOffset == 3);
            values.values[0] = lo.values[3];
            values.values[1] = hi.values[0];
            values.values[2] = hi.values[1];
            values.values[3] = hi.values[2];
        }
        return values;
    }
}

template <typename InputT, int ItemsPerLane>
[[nodiscard]] __device__ inline PackedInputValues<InputT, ItemsPerLane> loadAlignmentSafeArgInputPacket(const InputT* source) {
    constexpr size_t packet_bytes = sizeof(InputT) * ItemsPerLane;
    static_assert(ItemsPerLane == DENSE_ARG_MAX_ITEMS_PER_LANE);

    if constexpr (packet_bytes >= 16) {
        return loadAlignmentSafeInputPacket<InputT, ItemsPerLane>(source);
    } else {
        // ARG's x4 geometry gives FP8 a 4-byte packet and FP16/BF16 an 8-byte packet. Use that packet width as the
        // alignment primitive instead of inflating every lane to a 32-byte window. Neighboring lanes begin exactly
        // packet_bytes apart, so every lane in the CTA sees the same ElementOffset for a given reduction row. The
        // switch is therefore lockstep. An aligned packet needs one naturally aligned vector load; a shifted packet
        // needs exactly two naturally aligned packet-width loads followed by register-only reconstruction. Tensor
        // allocations carry 128 bytes of trailing padding, so the final logical packet needs no scalar tail path.
        static_assert(packet_bytes == 4 || packet_bytes == 8);
        const int element_offset = static_cast<int>((reinterpret_cast<uintptr_t>(source) & uintptr_t{packet_bytes - 1}) / sizeof(InputT));

        switch (element_offset) {
            case 0:
                return loadPacketAlignedArgInputPacket<InputT, ItemsPerLane, 0>(source);
            case 1:
                return loadPacketAlignedArgInputPacket<InputT, ItemsPerLane, 1>(source);
            case 2:
                return loadPacketAlignedArgInputPacket<InputT, ItemsPerLane, 2>(source);
            default:
                return loadPacketAlignedArgInputPacket<InputT, ItemsPerLane, 3>(source);
        }
    }
}

template <typename InputT, typename IndexT, int ItemsPerLane>
__device__ inline void loadAlignmentSafeDenseArgCandidates(const InputT* values,
                                                           const IndexT* carried_indices,
                                                           uint64_t linear_index,
                                                           uint64_t local_run_index,
                                                           uint64_t domain_stride,
                                                           DenseArgReductionCandidateFp32<IndexT> (&candidates)[ItemsPerLane]) {
    const PackedInputValues<InputT, ItemsPerLane> value_packet =
        loadAlignmentSafeArgInputPacket<InputT, ItemsPerLane>(values + linear_index);
    PackedInputValues<IndexT, ItemsPerLane> index_packet{};
    if (carried_indices != nullptr) {
        index_packet = loadAlignmentSafeInputPacket<IndexT, ItemsPerLane>(carried_indices + linear_index);
    }
    const uint64_t contribution = local_run_index * domain_stride;
#pragma unroll
    for (int item = 0; item < ItemsPerLane; ++item) {
        const uint64_t previous = carried_indices == nullptr ? 0 : static_cast<uint64_t>(index_packet.values[item]);
        candidates[item] = DenseArgReductionCandidateFp32<IndexT>{static_cast<IndexT>(previous + contribution),
                                                                  ToFp32<InputT>{}(value_packet.values[item])};
    }
}

template <typename InputT, typename ReductionOpT, typename IndexT, int ItemsPerLane>
__global__ void alignmentSafeVectorizedShardRangeArgReductionKernel(const InputT* input,
                                                                    const IndexT* carried_index_input,
                                                                    uint64_t domain_stride,
                                                                    void* value_output,
                                                                    DataType value_output_dtype,
                                                                    void* index_output,
                                                                    DataType index_output_dtype,
                                                                    uint64_t outer_size,
                                                                    uint64_t reduction_size,
                                                                    uint64_t inner_size,
                                                                    uint64_t first_shard_begin,
                                                                    uint64_t shard_width,
                                                                    uint64_t shard_stride,
                                                                    uint64_t shard_count,
                                                                    ReductionOpT reduction_op,
                                                                    ArgReductionCandidateFp32 init) {
    static_assert(ItemsPerLane == DENSE_ARG_MAX_ITEMS_PER_LANE);
    using CandidateT = DenseArgReductionCandidateFp32<IndexT>;

    const uint64_t total_work = outer_size * shard_count;
    const uint64_t grid_stride = static_cast<uint64_t>(gridDim.x);

    for (uint64_t work_index = static_cast<uint64_t>(blockIdx.x); work_index < total_work; work_index += grid_stride) {
        const uint64_t outer_index = work_index / shard_count;
        const uint64_t shard_index = work_index - outer_index * shard_count;
        const uint64_t shard_begin = first_shard_begin + shard_index * shard_stride;
        const uint64_t component_in_shard = static_cast<uint64_t>(threadIdx.x) * static_cast<uint64_t>(ItemsPerLane);
        const uint64_t component_begin = shard_begin + component_in_shard;

        CandidateT local[ItemsPerLane];
#pragma unroll
        for (int item = 0; item < ItemsPerLane; ++item) {
            local[item] = makeDenseArgReductionInit<IndexT>(init);
        }

        uint64_t row_base = outer_index * reduction_size * inner_size + component_begin;
        for (uint64_t row = 0; row < reduction_size; ++row) {
            CandidateT candidates[ItemsPerLane];
            loadAlignmentSafeDenseArgCandidates<InputT, IndexT, ItemsPerLane>(
                input, carried_index_input, row_base, row, domain_stride, candidates);
#pragma unroll
            for (int item = 0; item < ItemsPerLane; ++item) {
                local[item] = reduction_op(local[item], candidates[item]);
            }
            row_base += inner_size;
        }

#pragma unroll
        for (int item = 0; item < ItemsPerLane; ++item) {
            const uint64_t item_in_shard = component_in_shard + static_cast<uint64_t>(item);
            if (item_in_shard < shard_width) {
                const uint64_t component = shard_begin + item_in_shard;
                storeDenseArgReductionResult(
                    value_output, value_output_dtype, index_output, index_output_dtype, outer_index * inner_size + component, local[item]);
            }
        }
    }
}

template <typename InputT, typename ReductionOpT, typename IndexT, int ItemsPerLane>
void launchAlignmentSafeVectorizedArgShardRange(const InputT* input,
                                                const IndexT* carried_index_input,
                                                uint64_t domain_stride,
                                                void* value_output,
                                                DataType value_output_dtype,
                                                void* index_output,
                                                DataType index_output_dtype,
                                                const CubReductionGeometry& geometry,
                                                uint64_t first_shard_begin,
                                                uint64_t shard_width,
                                                uint64_t shard_stride,
                                                uint64_t shard_count,
                                                ReductionOpT reduction_op,
                                                ArgReductionCandidateFp32 init,
                                                cudaStream_t stream) {
    static_assert(ItemsPerLane == DENSE_ARG_MAX_ITEMS_PER_LANE);
    if (shard_count == 0) {
        return;
    }

    const uint64_t block_threads_u64 = ceilDivideU64(shard_width, static_cast<uint64_t>(ItemsPerLane));
    if (block_threads_u64 == 0 || block_threads_u64 > static_cast<uint64_t>(TILED_REDUCTION_BLOCK_THREADS)) {
        throw std::logic_error("Alignment-safe vectorized arg-reduction shard width exceeds one block.");
    }
    const unsigned int block_threads = static_cast<unsigned int>(block_threads_u64);
    const uint64_t total_work = geometry.outer_size * shard_count;
    const unsigned int grid_blocks = static_cast<unsigned int>(std::min<uint64_t>(total_work, TILED_REDUCTION_MAX_GRID_BLOCKS));

    alignmentSafeVectorizedShardRangeArgReductionKernel<InputT, ReductionOpT, IndexT, ItemsPerLane>
        <<<grid_blocks, block_threads, 0, stream>>>(input,
                                                    carried_index_input,
                                                    domain_stride,
                                                    value_output,
                                                    value_output_dtype,
                                                    index_output,
                                                    index_output_dtype,
                                                    geometry.outer_size,
                                                    geometry.reduction_size,
                                                    geometry.inner_size,
                                                    first_shard_begin,
                                                    shard_width,
                                                    shard_stride,
                                                    shard_count,
                                                    reduction_op,
                                                    init);
    CUDA_CHECK(cudaGetLastError());
}

template <typename InputT, typename ReductionOpT, typename IndexT, int ItemsPerLane>
void launchAlignmentSafeVectorizedBlockShardedArgReduction(const InputT* input,
                                                           const IndexT* carried_index_input,
                                                           uint64_t domain_stride,
                                                           void* value_output,
                                                           DataType value_output_dtype,
                                                           void* index_output,
                                                           DataType index_output_dtype,
                                                           const CubReductionGeometry& geometry,
                                                           ReductionOpT reduction_op,
                                                           ArgReductionCandidateFp32 init,
                                                           cudaStream_t stream) {
    static_assert(ItemsPerLane == DENSE_ARG_MAX_ITEMS_PER_LANE);
    constexpr uint64_t components_per_block = DENSE_ARG_COMPONENTS_PER_BLOCK;

    const uint64_t full_shards = geometry.inner_size / components_per_block;
    const uint64_t remainder = geometry.inner_size % components_per_block;

    // ARG deliberately keeps the arbitrary-width scheduler simple while evaluating the x4 candidate geometry:
    // preserve every proven 1024-component shard and emit at most one remainder shard. The reduction loop itself
    // remains width-branch-free; only the final output stores mask the unused slots in the last x4 packet.
    launchAlignmentSafeVectorizedArgShardRange<InputT, ReductionOpT, IndexT, ItemsPerLane>(input,
                                                                                           carried_index_input,
                                                                                           domain_stride,
                                                                                           value_output,
                                                                                           value_output_dtype,
                                                                                           index_output,
                                                                                           index_output_dtype,
                                                                                           geometry,
                                                                                           0,
                                                                                           components_per_block,
                                                                                           components_per_block,
                                                                                           full_shards,
                                                                                           reduction_op,
                                                                                           init,
                                                                                           stream);

    if (remainder != 0) {
        launchAlignmentSafeVectorizedArgShardRange<InputT, ReductionOpT, IndexT, ItemsPerLane>(input,
                                                                                               carried_index_input,
                                                                                               domain_stride,
                                                                                               value_output,
                                                                                               value_output_dtype,
                                                                                               index_output,
                                                                                               index_output_dtype,
                                                                                               geometry,
                                                                                               full_shards * components_per_block,
                                                                                               remainder,
                                                                                               0,
                                                                                               1,
                                                                                               reduction_op,
                                                                                               init,
                                                                                               stream);
    }
}

template <typename InputT, typename ReductionOpT, typename IndexT, int RowLanes>
__global__ void directTiledFixedSegmentArgReductionKernel(const InputT* input,
                                                          const IndexT* carried_index_input,
                                                          uint64_t domain_stride,
                                                          void* value_output,
                                                          DataType value_output_dtype,
                                                          void* index_output,
                                                          DataType index_output_dtype,
                                                          uint64_t outer_size,
                                                          uint64_t reduction_size,
                                                          uint64_t inner_size,
                                                          int warps_per_tile,
                                                          ReductionOpT reduction_op,
                                                          ArgReductionCandidateFp32 init) {
    static_assert(RowLanes == 1 || RowLanes == 2 || RowLanes == 4 || RowLanes == 8 || RowLanes == 16);
    constexpr int components_per_warp = TILED_REDUCTION_WARP_THREADS / RowLanes;
    using CandidateT = DenseArgReductionCandidateFp32<IndexT>;

    // Conservative UINT64-domain fallback still uses the old scalar global-load geometry, but candidate exchange is
    // shared-memory-only. Keep the candidate fields SoA to avoid 8/16-byte AoS bank pressure.
    __shared__ float lane_partial_values[TILED_REDUCTION_BLOCK_THREADS];
    __shared__ IndexT lane_partial_indices[TILED_REDUCTION_BLOCK_THREADS];

    const int groups_per_block = TILED_REDUCTION_WARPS_PER_BLOCK / warps_per_tile;
    const int physical_warp = static_cast<int>(threadIdx.x) / TILED_REDUCTION_WARP_THREADS;
    const int lane = static_cast<int>(threadIdx.x) % TILED_REDUCTION_WARP_THREADS;
    const int group_in_block = physical_warp / warps_per_tile;
    const int warp_in_tile = physical_warp % warps_per_tile;
    const int component_in_tile = lane / RowLanes;
    const int row_lane = lane % RowLanes;

    const uint64_t component_tiles = ceilDivideU64(inner_size, static_cast<uint64_t>(components_per_warp));
    const uint64_t total_work = outer_size * component_tiles;
    const uint64_t block_work_stride = static_cast<uint64_t>(gridDim.x) * static_cast<uint64_t>(groups_per_block);
    const CandidateT init_candidate = makeDenseArgReductionInit<IndexT>(init);

    for (uint64_t block_work_base = static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(groups_per_block);
         block_work_base < total_work;
         block_work_base += block_work_stride) {
        const uint64_t work_index = block_work_base + static_cast<uint64_t>(group_in_block);
        const bool work_active = work_index < total_work;
        const uint64_t outer_index = work_active ? work_index / component_tiles : 0;
        const uint64_t component_tile = work_active ? work_index - outer_index * component_tiles : 0;
        const uint64_t component = component_tile * static_cast<uint64_t>(components_per_warp) + static_cast<uint64_t>(component_in_tile);
        const bool component_active = work_active && component < inner_size;

        CandidateT local = init_candidate;
        if (component_active) {
            const uint64_t first_row = static_cast<uint64_t>(warp_in_tile * RowLanes + row_lane);
            const uint64_t row_stride = static_cast<uint64_t>(warps_per_tile * RowLanes);
            if (first_row < reduction_size) {
                uint64_t input_index = ((outer_index * reduction_size + first_row) * inner_size) + component;
                const uint64_t input_stride = row_stride * inner_size;
                for (uint64_t row = first_row; row < reduction_size; row += row_stride) {
                    local = reduction_op(
                        local, loadDenseArgCandidate<InputT, IndexT>(input, carried_index_input, input_index, row, domain_stride));
                    input_index += input_stride;
                }
            }
        }

        lane_partial_values[threadIdx.x] = local.value;
        lane_partial_indices[threadIdx.x] = local.index;
        __syncwarp();

        const int logical_group_base = physical_warp * TILED_REDUCTION_WARP_THREADS + component_in_tile * RowLanes;
        if constexpr (RowLanes > 1) {
#pragma unroll
            for (int offset = RowLanes / 2; offset != 0; offset >>= 1) {
                if (row_lane < offset) {
                    const int lhs_slot = logical_group_base + row_lane;
                    const int rhs_slot = lhs_slot + offset;
                    const CandidateT lhs{lane_partial_indices[lhs_slot], lane_partial_values[lhs_slot]};
                    const CandidateT rhs{lane_partial_indices[rhs_slot], lane_partial_values[rhs_slot]};
                    const CandidateT reduced = reduction_op(lhs, rhs);
                    lane_partial_values[lhs_slot] = reduced.value;
                    lane_partial_indices[lhs_slot] = reduced.index;
                }
                __syncwarp();
            }
        }

        if (warps_per_tile == 1) {
            if (component_active && row_lane == 0) {
                const CandidateT aggregate{lane_partial_indices[logical_group_base], lane_partial_values[logical_group_base]};
                storeDenseArgReductionResult(
                    value_output, value_output_dtype, index_output, index_output_dtype, outer_index * inner_size + component, aggregate);
            }
            __syncwarp();
        } else {
            __syncthreads();

            if (warp_in_tile == 0 && row_lane == 0 && component_active) {
                CandidateT aggregate = init_candidate;
                const int first_warp_in_group = group_in_block * warps_per_tile;
                for (int cooperating_warp = 0; cooperating_warp < warps_per_tile; ++cooperating_warp) {
                    const int source_slot =
                        (first_warp_in_group + cooperating_warp) * TILED_REDUCTION_WARP_THREADS + component_in_tile * RowLanes;
                    aggregate = reduction_op(aggregate, CandidateT{lane_partial_indices[source_slot], lane_partial_values[source_slot]});
                }
                storeDenseArgReductionResult(
                    value_output, value_output_dtype, index_output, index_output_dtype, outer_index * inner_size + component, aggregate);
            }
            __syncthreads();
        }
    }
}

template <typename InputT, typename ReductionOpT, typename IndexT, int RowLanes>
void launchDirectTiledFixedSegmentArgReductionForRowLanes(const InputT* input,
                                                          const IndexT* carried_index_input,
                                                          uint64_t domain_stride,
                                                          void* value_output,
                                                          DataType value_output_dtype,
                                                          void* index_output,
                                                          DataType index_output_dtype,
                                                          const CubReductionGeometry& geometry,
                                                          ReductionOpT reduction_op,
                                                          ArgReductionCandidateFp32 init,
                                                          cudaStream_t stream) {
    constexpr uint64_t components_per_warp = TILED_REDUCTION_WARP_THREADS / RowLanes;
    const uint64_t component_tiles = ceilDivideU64(geometry.inner_size, components_per_warp);
    const int warps_per_tile = chooseDirectTiledReductionWarpsPerTile<RowLanes>(geometry);
    const int groups_per_block = TILED_REDUCTION_WARPS_PER_BLOCK / warps_per_tile;
    const uint64_t total_work = geometry.outer_size * component_tiles;
    const uint64_t required_blocks = ceilDivideU64(total_work, static_cast<uint64_t>(groups_per_block));
    const unsigned int grid_blocks = static_cast<unsigned int>(std::min<uint64_t>(required_blocks, TILED_REDUCTION_MAX_GRID_BLOCKS));

    directTiledFixedSegmentArgReductionKernel<InputT, ReductionOpT, IndexT, RowLanes>
        <<<grid_blocks, TILED_REDUCTION_BLOCK_THREADS, 0, stream>>>(input,
                                                                    carried_index_input,
                                                                    domain_stride,
                                                                    value_output,
                                                                    value_output_dtype,
                                                                    index_output,
                                                                    index_output_dtype,
                                                                    geometry.outer_size,
                                                                    geometry.reduction_size,
                                                                    geometry.inner_size,
                                                                    warps_per_tile,
                                                                    reduction_op,
                                                                    init);
    CUDA_CHECK(cudaGetLastError());
}

template <typename InputT, typename ReductionOpT, typename IndexT>
void launchTiledFixedSegmentArgReductionForIndex(const InputT* input,
                                                 const IndexT* carried_index_input,
                                                 uint64_t domain_stride,
                                                 void* value_output,
                                                 DataType value_output_dtype,
                                                 void* index_output,
                                                 DataType index_output_dtype,
                                                 const CubReductionGeometry& geometry,
                                                 ReductionOpT reduction_op,
                                                 ArgReductionCandidateFp32 init,
                                                 cudaStream_t stream) {
    if (!geometry.reduced_axes_are_contiguous || geometry.inner_size <= 1) {
        throw std::logic_error("Tiled CUB arg reduction requires contiguous reduced axes and trailing width > 1.");
    }

    // ARG-DIRECT-1 owns the complete normal UINT32 dense hot path. Narrow rows use the same aligned staged-copy
    // geometry proven by the value reducer and reduce row-lane candidates through shared memory. Medium/wide rows
    // use directly-owned aligned 16-byte packets and cooperative whole-warps. Very large public reduction domains that
    // genuinely require UINT64 candidate state retain the conservative backend until separately benchmarked.
    if constexpr (std::is_same_v<IndexT, uint32_t>) {
        if (geometry.inner_size <= 32) {
            if (carried_index_input == nullptr) {
                launchAlignedAsyncNarrowTiledArgReduction<InputT, ReductionOpT, false>(input,
                                                                                       nullptr,
                                                                                       domain_stride,
                                                                                       value_output,
                                                                                       value_output_dtype,
                                                                                       index_output,
                                                                                       index_output_dtype,
                                                                                       geometry,
                                                                                       reduction_op,
                                                                                       init,
                                                                                       stream);
                return;
            }
            if constexpr (std::is_same_v<InputT, float>) {
                launchAlignedAsyncNarrowTiledArgReduction<InputT, ReductionOpT, true>(input,
                                                                                      carried_index_input,
                                                                                      domain_stride,
                                                                                      value_output,
                                                                                      value_output_dtype,
                                                                                      index_output,
                                                                                      index_output_dtype,
                                                                                      geometry,
                                                                                      reduction_op,
                                                                                      init,
                                                                                      stream);
                return;
            }
        } else if constexpr (sizeof(InputT) >= 2) {
            // With at least two bytes/element, multiplying the row stride by at most the eight warps in one block can
            // always make a warp's revisited-row stride divisible by sixteen. FP8 odd-width rows need sixteen warps
            // for that invariant and therefore retain the existing alignment-safe fallback.
            if (carried_index_input == nullptr) {
                launchAlignedCooperativeTiledArgReduction<InputT, ReductionOpT, false>(input,
                                                                                       nullptr,
                                                                                       domain_stride,
                                                                                       value_output,
                                                                                       value_output_dtype,
                                                                                       index_output,
                                                                                       index_output_dtype,
                                                                                       geometry,
                                                                                       reduction_op,
                                                                                       init,
                                                                                       stream);
                return;
            }
            if constexpr (std::is_same_v<InputT, float>) {
                launchAlignedCooperativeTiledArgReduction<InputT, ReductionOpT, true>(input,
                                                                                      carried_index_input,
                                                                                      domain_stride,
                                                                                      value_output,
                                                                                      value_output_dtype,
                                                                                      index_output,
                                                                                      index_output_dtype,
                                                                                      geometry,
                                                                                      reduction_op,
                                                                                      init,
                                                                                      stream);
                return;
            }
        }
    }

    if (geometry.inner_size == 64) {
        launchVectorizedDirectFullRowArgReduction<InputT, ReductionOpT, IndexT, 2>(input,
                                                                                   carried_index_input,
                                                                                   domain_stride,
                                                                                   value_output,
                                                                                   value_output_dtype,
                                                                                   index_output,
                                                                                   index_output_dtype,
                                                                                   geometry,
                                                                                   reduction_op,
                                                                                   init,
                                                                                   stream);
    } else if (geometry.inner_size == 128) {
        launchVectorizedDirectFullRowArgReduction<InputT, ReductionOpT, IndexT, 4>(input,
                                                                                   carried_index_input,
                                                                                   domain_stride,
                                                                                   value_output,
                                                                                   value_output_dtype,
                                                                                   index_output,
                                                                                   index_output_dtype,
                                                                                   geometry,
                                                                                   reduction_op,
                                                                                   init,
                                                                                   stream);
    } else if (geometry.inner_size == 256) {
        launchVectorizedDirectGroupedFullRowArgReduction<InputT, ReductionOpT, IndexT, 2, 4>(input,
                                                                                             carried_index_input,
                                                                                             domain_stride,
                                                                                             value_output,
                                                                                             value_output_dtype,
                                                                                             index_output,
                                                                                             index_output_dtype,
                                                                                             geometry,
                                                                                             reduction_op,
                                                                                             init,
                                                                                             stream);
    } else if (geometry.inner_size == 512) {
        launchVectorizedDirectGroupedFullRowArgReduction<InputT, ReductionOpT, IndexT, 4, 4>(input,
                                                                                             carried_index_input,
                                                                                             domain_stride,
                                                                                             value_output,
                                                                                             value_output_dtype,
                                                                                             index_output,
                                                                                             index_output_dtype,
                                                                                             geometry,
                                                                                             reduction_op,
                                                                                             init,
                                                                                             stream);
    } else if (geometry.inner_size == 1024) {
        launchVectorizedDirectGroupedFullRowArgReduction<InputT, ReductionOpT, IndexT, 8, 4>(input,
                                                                                             carried_index_input,
                                                                                             domain_stride,
                                                                                             value_output,
                                                                                             value_output_dtype,
                                                                                             index_output,
                                                                                             index_output_dtype,
                                                                                             geometry,
                                                                                             reduction_op,
                                                                                             init,
                                                                                             stream);
    } else if (geometry.inner_size > 1024 && geometry.inner_size % DENSE_ARG_COMPONENTS_PER_BLOCK == 0) {
        launchVectorizedDirectBlockShardedArgReduction<InputT, ReductionOpT, IndexT, 4>(input,
                                                                                        carried_index_input,
                                                                                        domain_stride,
                                                                                        value_output,
                                                                                        value_output_dtype,
                                                                                        index_output,
                                                                                        index_output_dtype,
                                                                                        geometry,
                                                                                        reduction_op,
                                                                                        init,
                                                                                        stream);
    } else if (geometry.inner_size > FULL_ROW_GROUP_MAX_INNER_SIZE) {
        launchAlignmentSafeVectorizedBlockShardedArgReduction<InputT, ReductionOpT, IndexT, 4>(input,
                                                                                               carried_index_input,
                                                                                               domain_stride,
                                                                                               value_output,
                                                                                               value_output_dtype,
                                                                                               index_output,
                                                                                               index_output_dtype,
                                                                                               geometry,
                                                                                               reduction_op,
                                                                                               init,
                                                                                               stream);
    } else if (geometry.inner_size <= 2) {
        launchDirectTiledFixedSegmentArgReductionForRowLanes<InputT, ReductionOpT, IndexT, 16>(input,
                                                                                               carried_index_input,
                                                                                               domain_stride,
                                                                                               value_output,
                                                                                               value_output_dtype,
                                                                                               index_output,
                                                                                               index_output_dtype,
                                                                                               geometry,
                                                                                               reduction_op,
                                                                                               init,
                                                                                               stream);
    } else if (geometry.inner_size <= 4) {
        launchDirectTiledFixedSegmentArgReductionForRowLanes<InputT, ReductionOpT, IndexT, 8>(input,
                                                                                              carried_index_input,
                                                                                              domain_stride,
                                                                                              value_output,
                                                                                              value_output_dtype,
                                                                                              index_output,
                                                                                              index_output_dtype,
                                                                                              geometry,
                                                                                              reduction_op,
                                                                                              init,
                                                                                              stream);
    } else if (geometry.inner_size <= 8) {
        launchDirectTiledFixedSegmentArgReductionForRowLanes<InputT, ReductionOpT, IndexT, 4>(input,
                                                                                              carried_index_input,
                                                                                              domain_stride,
                                                                                              value_output,
                                                                                              value_output_dtype,
                                                                                              index_output,
                                                                                              index_output_dtype,
                                                                                              geometry,
                                                                                              reduction_op,
                                                                                              init,
                                                                                              stream);
    } else if (geometry.inner_size <= 16) {
        launchDirectTiledFixedSegmentArgReductionForRowLanes<InputT, ReductionOpT, IndexT, 2>(input,
                                                                                              carried_index_input,
                                                                                              domain_stride,
                                                                                              value_output,
                                                                                              value_output_dtype,
                                                                                              index_output,
                                                                                              index_output_dtype,
                                                                                              geometry,
                                                                                              reduction_op,
                                                                                              init,
                                                                                              stream);
    } else {
        launchDirectTiledFixedSegmentArgReductionForRowLanes<InputT, ReductionOpT, IndexT, 1>(input,
                                                                                              carried_index_input,
                                                                                              domain_stride,
                                                                                              value_output,
                                                                                              value_output_dtype,
                                                                                              index_output,
                                                                                              index_output_dtype,
                                                                                              geometry,
                                                                                              reduction_op,
                                                                                              init,
                                                                                              stream);
    }
}

template <typename InputT, typename ReductionOpT>
void launchTiledFixedSegmentArgReduction(const InputT* input,
                                         void* value_output,
                                         DataType value_output_dtype,
                                         void* index_output,
                                         DataType index_output_dtype,
                                         const CubReductionGeometry& geometry,
                                         ReductionOpT reduction_op,
                                         ArgReductionCandidateFp32 init,
                                         cudaStream_t stream) {
    auto dispatch_index = [&]<typename IndexT>() -> void {
        launchTiledFixedSegmentArgReductionForIndex<InputT, ReductionOpT, IndexT>(
            input, nullptr, 1, value_output, value_output_dtype, index_output, index_output_dtype, geometry, reduction_op, init, stream);
    };
    dispatchDenseArgAccumulatorIndexDType(geometry.reduction_size, dispatch_index);
}

template <typename InputT, typename ReductionOpT, typename IndexT>
void launchComposedTiledFixedSegmentArgReduction(const InputT* input,
                                                 const IndexT* carried_index_input,
                                                 uint64_t domain_stride,
                                                 void* value_output,
                                                 DataType value_output_dtype,
                                                 void* index_output,
                                                 DataType index_output_dtype,
                                                 const CubReductionGeometry& geometry,
                                                 ReductionOpT reduction_op,
                                                 ArgReductionCandidateFp32 init,
                                                 cudaStream_t stream) {
    launchTiledFixedSegmentArgReductionForIndex<InputT, ReductionOpT, IndexT>(input,
                                                                              carried_index_input,
                                                                              domain_stride,
                                                                              value_output,
                                                                              value_output_dtype,
                                                                              index_output,
                                                                              index_output_dtype,
                                                                              geometry,
                                                                              reduction_op,
                                                                              init,
                                                                              stream);
}

template <typename InputT, typename ReductionOpT>
size_t queryArgReductionBytesForInput(DataType value_output_dtype,
                                      bool produce_value,
                                      DataType index_output_dtype,
                                      bool produce_index,
                                      const CubReductionGeometry& geometry,
                                      ReductionOpT reduction_op,
                                      ArgReductionCandidateFp32 init,
                                      cudaStream_t stream);

template <typename InputT, typename ReductionOpT>
size_t queryArgReductionBytesForInput(const Tensor& input,
                                      Tensor* value_output,
                                      Tensor* index_output,
                                      const CubReductionGeometry& geometry,
                                      ReductionOpT reduction_op,
                                      ArgReductionCandidateFp32 init,
                                      cudaStream_t stream) {
    if (geometry.path == CubReductionPath::StridedFixedSegment) {
        size_t queried_bytes = 0;
        auto output_iterator = makeRuntimeArgReductionOutputIterator(value_output, index_output);
        auto input_iterator = makeStridedArgCandidateIterator<InputT>(input, geometry);
        CUDA_CHECK(cub::DeviceSegmentedReduce::Reduce(nullptr,
                                                      queried_bytes,
                                                      input_iterator,
                                                      output_iterator,
                                                      static_cast<int64_t>(geometry.output_elements),
                                                      static_cast<int>(geometry.reduction_size),
                                                      reduction_op,
                                                      init,
                                                      stream));
        return std::max<size_t>(queried_bytes, 1);
    }
    return queryArgReductionBytesForInput<InputT>(value_output == nullptr ? DataType::FP32 : value_output->getDataType(),
                                                   value_output != nullptr,
                                                   index_output == nullptr ? DataType::UINT32 : index_output->getDataType(),
                                                   index_output != nullptr,
                                                   geometry,
                                                   reduction_op,
                                                   init,
                                                   stream);
}

template <typename InputT, typename ReductionOpT>
size_t queryArgReductionBytesForInput(DataType value_output_dtype,
                                      bool produce_value,
                                      DataType index_output_dtype,
                                      bool produce_index,
                                      const CubReductionGeometry& geometry,
                                      ReductionOpT reduction_op,
                                      ArgReductionCandidateFp32 init,
                                      cudaStream_t stream) {
    using AccumulatorT = std::decay_t<decltype(std::declval<ReductionOpT>()(std::declval<ArgReductionCandidateFp32>(),
                                                                            std::declval<ArgReductionCandidateFp32>()))>;
    static_assert(std::is_same_v<AccumulatorT, ArgReductionCandidateFp32>,
                  "CUB arg reductions must preserve the FP32 candidate state.");

    size_t queried_bytes = 0;
    auto output_iterator = makeRuntimeArgReductionOutputIterator(
        produce_value ? std::optional<DataType>(value_output_dtype) : std::nullopt,
        produce_index ? std::optional<DataType>(index_output_dtype) : std::nullopt);

    switch (geometry.path) {
        case CubReductionPath::DeviceTransformReduce: {
            auto dispatch_index = [&]<typename IndexT>() -> size_t {
                size_t bytes = 0;
                auto input_iterator = makeCompactDeviceArgCandidateIterator<InputT, IndexT>();
                const auto compact_init = makeDenseArgReductionInit<IndexT>(init);
                CUDA_CHECK(cub::DeviceReduce::Reduce(nullptr,
                                                     bytes,
                                                     input_iterator,
                                                     output_iterator,
                                                     static_cast<int64_t>(geometry.input_elements),
                                                     reduction_op,
                                                     compact_init,
                                                     stream));
                return bytes;
            };
            queried_bytes = dispatchDenseArgAccumulatorIndexDType(geometry.input_elements, dispatch_index);
            break;
        }
        case CubReductionPath::ContiguousFixedSegment: {
            if (geometry.reduction_size <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
                queried_bytes = 1;
                break;
            }
            auto input_iterator = makeCompactContiguousArgCandidateIterator<InputT, uint64_t>(geometry);
            const auto compact_init = makeDenseArgReductionInit<uint64_t>(init);
            CUDA_CHECK(cub::DeviceSegmentedReduce::Reduce(nullptr,
                                                          queried_bytes,
                                                          input_iterator,
                                                          output_iterator,
                                                          static_cast<int64_t>(geometry.output_elements),
                                                          static_cast<int>(geometry.reduction_size),
                                                          reduction_op,
                                                          compact_init,
                                                          stream));
            break;
        }
        case CubReductionPath::TiledFixedSegment:
            queried_bytes = 1;
            break;
        case CubReductionPath::StridedFixedSegment:
            throw std::logic_error("Descriptor-only CUB arg workspace query requires ordinary dense geometry.");
        case CubReductionPath::OffsetSegmented:
            throw std::logic_error("Dense CUB arg reduction received offset-segmented geometry.");
        case CubReductionPath::ComposedDense:
            throw std::logic_error("Composed dense arg reduction must query its stamped direct stages.");
    }

    return std::max<size_t>(queried_bytes, 1);
}

template <typename InputT, typename ReductionOpT>
void launchArgReductionForInput(const Tensor& temp_storage,
                                size_t temp_storage_bytes,
                                const Tensor& input,
                                Tensor* value_output,
                                Tensor* index_output,
                                const CubReductionGeometry& geometry,
                                ReductionOpT reduction_op,
                                ArgReductionCandidateFp32 init,
                                cudaStream_t stream) {
    void* temp_storage_ptr = const_cast<void*>(static_cast<const void*>(temp_storage.getMemPtr<void>()));
    auto output_iterator = makeRuntimeArgReductionOutputIterator(value_output, index_output);

    switch (geometry.path) {
        case CubReductionPath::DeviceTransformReduce: {
            auto dispatch_index = [&]<typename IndexT>() -> void {
                auto input_iterator = makeCompactDeviceArgCandidateIterator<InputT, IndexT>(input);
                const auto compact_init = makeDenseArgReductionInit<IndexT>(init);
                CUDA_CHECK(cub::DeviceReduce::Reduce(temp_storage_ptr,
                                                     temp_storage_bytes,
                                                     input_iterator,
                                                     output_iterator,
                                                     static_cast<int64_t>(geometry.input_elements),
                                                     reduction_op,
                                                     compact_init,
                                                     stream));
            };
            dispatchDenseArgAccumulatorIndexDType(geometry.input_elements, dispatch_index);
            break;
        }
        case CubReductionPath::ContiguousFixedSegment: {
            if (geometry.reduction_size <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
                launchAlignedContiguousSegmentArgReduction<InputT, ReductionOpT, false>(
                    input.getMemPtr<InputT>(),
                    nullptr,
                    1,
                    value_output == nullptr ? nullptr : value_output->getMemPtr<void>(),
                    value_output == nullptr ? DataType::FP32 : value_output->getDataType(),
                    index_output == nullptr ? nullptr : index_output->getMemPtr<void>(),
                    index_output == nullptr ? DataType::UINT32 : index_output->getDataType(),
                    geometry,
                    reduction_op,
                    init,
                    stream);
                break;
            }
            auto input_iterator = makeCompactContiguousArgCandidateIterator<InputT, uint64_t>(input, geometry);
            const auto compact_init = makeDenseArgReductionInit<uint64_t>(init);
            CUDA_CHECK(cub::DeviceSegmentedReduce::Reduce(temp_storage_ptr,
                                                          temp_storage_bytes,
                                                          input_iterator,
                                                          output_iterator,
                                                          static_cast<int64_t>(geometry.output_elements),
                                                          static_cast<int>(geometry.reduction_size),
                                                          reduction_op,
                                                          compact_init,
                                                          stream));
            break;
        }
        case CubReductionPath::TiledFixedSegment:
            launchTiledFixedSegmentArgReduction<InputT>(input.getMemPtr<InputT>(),
                                                        value_output == nullptr ? nullptr : value_output->getMemPtr<void>(),
                                                        value_output == nullptr ? DataType::FP32 : value_output->getDataType(),
                                                        index_output == nullptr ? nullptr : index_output->getMemPtr<void>(),
                                                        index_output == nullptr ? DataType::UINT32 : index_output->getDataType(),
                                                        geometry,
                                                        reduction_op,
                                                        init,
                                                        stream);
            break;
        case CubReductionPath::StridedFixedSegment: {
            auto input_iterator = makeStridedArgCandidateIterator<InputT>(input, geometry);
            CUDA_CHECK(cub::DeviceSegmentedReduce::Reduce(temp_storage_ptr,
                                                          temp_storage_bytes,
                                                          input_iterator,
                                                          output_iterator,
                                                          static_cast<int64_t>(geometry.output_elements),
                                                          static_cast<int>(geometry.reduction_size),
                                                          reduction_op,
                                                          init,
                                                          stream));
            break;
        }
        case CubReductionPath::OffsetSegmented:
            throw std::logic_error("Dense CUB arg reduction received offset-segmented geometry.");
        case CubReductionPath::ComposedDense:
            throw std::logic_error("Composed dense arg reduction must execute through its stamped direct stages.");
    }
}

template <typename ReductionOpT>
size_t queryOperationArgReductionBytes(const Tensor& input,
                                       Tensor* value_output,
                                       Tensor* index_output,
                                       const CubReductionGeometry& geometry,
                                       ReductionOpT reduction_op,
                                       ArgReductionCandidateFp32 init,
                                       const Stream& stream) {
    auto dispatch_input = [&]<typename InputT>() -> size_t {
        return queryArgReductionBytesForInput<InputT>(input, value_output, index_output, geometry, reduction_op, init, stream.getStream());
    };
    return dispatchReductionInputDType(input.getDataType(), dispatch_input);
}

template <typename ReductionOpT>
size_t queryOperationArgReductionBytes(DataType input_dtype,
                                       std::optional<DataType> value_output_dtype,
                                       std::optional<DataType> index_output_dtype,
                                       const CubReductionGeometry& geometry,
                                       ReductionOpT reduction_op,
                                       ArgReductionCandidateFp32 init,
                                       const Stream& stream) {
    auto dispatch_input = [&]<typename InputT>() -> size_t {
        return queryArgReductionBytesForInput<InputT>(value_output_dtype.value_or(DataType::FP32),
                                                       value_output_dtype.has_value(),
                                                       index_output_dtype.value_or(DataType::UINT32),
                                                       index_output_dtype.has_value(),
                                                       geometry,
                                                       reduction_op,
                                                       init,
                                                       stream.getStream());
    };
    return dispatchReductionInputDType(input_dtype, dispatch_input);
}

template <typename ReductionOpT>
void launchOperationArgReduction(const Tensor& temp_storage,
                                 size_t temp_storage_bytes,
                                 const Tensor& input,
                                 Tensor* value_output,
                                 Tensor* index_output,
                                 const CubReductionGeometry& geometry,
                                 ReductionOpT reduction_op,
                                 ArgReductionCandidateFp32 init,
                                 Stream& stream) {
    auto dispatch_input = [&]<typename InputT>() -> void {
        launchArgReductionForInput<InputT>(
            temp_storage, temp_storage_bytes, input, value_output, index_output, geometry, reduction_op, init, stream.getStream());
    };
    dispatchReductionInputDType(input.getDataType(), dispatch_input);
}

template <typename InputT, typename IndexT, typename ReductionOpT>
size_t queryComposedArgReductionStageBytesForInput(bool has_carried_index_input,
                                                   std::optional<DataType> value_output_dtype,
                                                   std::optional<DataType> index_output_dtype,
                                                   const CubReductionGeometry& geometry,
                                                   uint64_t domain_stride,
                                                   ReductionOpT reduction_op,
                                                   ArgReductionCandidateFp32 init,
                                                   cudaStream_t stream);

template <typename InputT, typename IndexT, typename ReductionOpT>
size_t queryComposedArgReductionStageBytesForInput(const Tensor& value_input,
                                                   const Tensor* carried_index_input,
                                                   Tensor* value_output,
                                                   Tensor* index_output,
                                                   const CubReductionGeometry& geometry,
                                                   uint64_t domain_stride,
                                                   ReductionOpT reduction_op,
                                                   ArgReductionCandidateFp32 init,
                                                   cudaStream_t stream) {
    return queryComposedArgReductionStageBytesForInput<InputT, IndexT>(
        carried_index_input != nullptr,
        value_output == nullptr ? std::nullopt : std::optional<DataType>(value_output->getDataType()),
        index_output == nullptr ? std::nullopt : std::optional<DataType>(index_output->getDataType()),
        geometry,
        domain_stride,
        reduction_op,
        init,
        stream);
}

template <typename InputT, typename IndexT, typename ReductionOpT>
size_t queryComposedArgReductionStageBytesForInput(bool has_carried_index_input,
                                                   std::optional<DataType> value_output_dtype,
                                                   std::optional<DataType> index_output_dtype,
                                                   const CubReductionGeometry& geometry,
                                                   uint64_t domain_stride,
                                                   ReductionOpT reduction_op,
                                                   ArgReductionCandidateFp32 init,
                                                   cudaStream_t stream) {
    using CandidateT = DenseArgReductionCandidateFp32<IndexT>;
    const CandidateT typed_init = makeDenseArgReductionInit<IndexT>(init);
    size_t queried_bytes = 0;
    auto output_iterator = makeRuntimeArgReductionOutputIterator(value_output_dtype, index_output_dtype);

    switch (geometry.path) {
        case CubReductionPath::DeviceTransformReduce: {
            auto input_iterator = makeComposedDeviceArgCandidateIterator<InputT, IndexT>(domain_stride);
            CUDA_CHECK(cub::DeviceReduce::Reduce(nullptr,
                                                 queried_bytes,
                                                 input_iterator,
                                                 output_iterator,
                                                 static_cast<int64_t>(geometry.input_elements),
                                                 reduction_op,
                                                 typed_init,
                                                 stream));
            break;
        }
        case CubReductionPath::ContiguousFixedSegment: {
            if constexpr (std::is_same_v<IndexT, uint32_t>) {
                if (!has_carried_index_input || std::is_same_v<InputT, float>) {
                    queried_bytes = 1;
                    break;
                }
            }
            auto input_iterator =
                makeComposedContiguousArgCandidateIterator<InputT, IndexT>(geometry, domain_stride);
            CUDA_CHECK(cub::DeviceSegmentedReduce::Reduce(nullptr,
                                                          queried_bytes,
                                                          input_iterator,
                                                          output_iterator,
                                                          static_cast<int64_t>(geometry.output_elements),
                                                          static_cast<int>(geometry.reduction_size),
                                                          reduction_op,
                                                          typed_init,
                                                          stream));
            break;
        }
        case CubReductionPath::TiledFixedSegment:
            queried_bytes = 1;
            break;
        case CubReductionPath::StridedFixedSegment:
        case CubReductionPath::OffsetSegmented:
        case CubReductionPath::ComposedDense:
            throw std::logic_error("Composed dense CUB arg stage must resolve to a direct dense reducer family.");
    }
    return std::max<size_t>(queried_bytes, 1);
}

template <typename InputT, typename IndexT, typename ReductionOpT>
void launchComposedArgReductionStageForInput(const Tensor& temp_storage,
                                             size_t temp_storage_bytes,
                                             const Tensor& value_input,
                                             const Tensor* carried_index_input,
                                             Tensor* value_output,
                                             Tensor* index_output,
                                             const CubReductionGeometry& geometry,
                                             uint64_t domain_stride,
                                             ReductionOpT reduction_op,
                                             ArgReductionCandidateFp32 init,
                                             cudaStream_t stream) {
    using CandidateT = DenseArgReductionCandidateFp32<IndexT>;
    const CandidateT typed_init = makeDenseArgReductionInit<IndexT>(init);
    void* temp_storage_ptr = const_cast<void*>(static_cast<const void*>(temp_storage.getMemPtr<void>()));
    auto output_iterator = makeRuntimeArgReductionOutputIterator(value_output, index_output);

    switch (geometry.path) {
        case CubReductionPath::DeviceTransformReduce: {
            auto input_iterator = makeComposedDeviceArgCandidateIterator<InputT, IndexT>(value_input, carried_index_input, domain_stride);
            CUDA_CHECK(cub::DeviceReduce::Reduce(temp_storage_ptr,
                                                 temp_storage_bytes,
                                                 input_iterator,
                                                 output_iterator,
                                                 static_cast<int64_t>(geometry.input_elements),
                                                 reduction_op,
                                                 typed_init,
                                                 stream));
            break;
        }
        case CubReductionPath::ContiguousFixedSegment: {
            if constexpr (std::is_same_v<IndexT, uint32_t>) {
                if (carried_index_input == nullptr) {
                    launchAlignedContiguousSegmentArgReduction<InputT, ReductionOpT, false>(
                        value_input.getMemPtr<InputT>(),
                        nullptr,
                        domain_stride,
                        value_output == nullptr ? nullptr : value_output->getMemPtr<void>(),
                        value_output == nullptr ? DataType::FP32 : value_output->getDataType(),
                        index_output == nullptr ? nullptr : index_output->getMemPtr<void>(),
                        index_output == nullptr ? DataType::UINT32 : index_output->getDataType(),
                        geometry,
                        reduction_op,
                        init,
                        stream);
                    break;
                }
                if constexpr (std::is_same_v<InputT, float>) {
                    launchAlignedContiguousSegmentArgReduction<InputT, ReductionOpT, true>(
                        value_input.getMemPtr<InputT>(),
                        carried_index_input->getMemPtr<uint32_t>(),
                        domain_stride,
                        value_output == nullptr ? nullptr : value_output->getMemPtr<void>(),
                        value_output == nullptr ? DataType::FP32 : value_output->getDataType(),
                        index_output == nullptr ? nullptr : index_output->getMemPtr<void>(),
                        index_output == nullptr ? DataType::UINT32 : index_output->getDataType(),
                        geometry,
                        reduction_op,
                        init,
                        stream);
                    break;
                }
            }
            auto input_iterator =
                makeComposedContiguousArgCandidateIterator<InputT, IndexT>(value_input, carried_index_input, geometry, domain_stride);
            CUDA_CHECK(cub::DeviceSegmentedReduce::Reduce(temp_storage_ptr,
                                                          temp_storage_bytes,
                                                          input_iterator,
                                                          output_iterator,
                                                          static_cast<int64_t>(geometry.output_elements),
                                                          static_cast<int>(geometry.reduction_size),
                                                          reduction_op,
                                                          typed_init,
                                                          stream));
            break;
        }
        case CubReductionPath::TiledFixedSegment:
            launchComposedTiledFixedSegmentArgReduction<InputT, ReductionOpT, IndexT>(
                value_input.getMemPtr<InputT>(),
                carried_index_input == nullptr ? nullptr : carried_index_input->getMemPtr<IndexT>(),
                domain_stride,
                value_output == nullptr ? nullptr : value_output->getMemPtr<void>(),
                value_output == nullptr ? DataType::FP32 : value_output->getDataType(),
                index_output == nullptr ? nullptr : index_output->getMemPtr<void>(),
                index_output == nullptr ? DataType::UINT32 : index_output->getDataType(),
                geometry,
                reduction_op,
                init,
                stream);
            break;
        case CubReductionPath::StridedFixedSegment:
        case CubReductionPath::OffsetSegmented:
        case CubReductionPath::ComposedDense:
            throw std::logic_error("Composed dense CUB arg stage must resolve to a direct dense reducer family.");
    }
}

template <typename ReductionOpT>
size_t queryComposedOperationArgReductionStageBytes(const Tensor& value_input,
                                                    const Tensor* carried_index_input,
                                                    Tensor* value_output,
                                                    Tensor* index_output,
                                                    const CubReductionGeometry& geometry,
                                                    uint64_t domain_stride,
                                                    DataType carried_index_dtype,
                                                    ReductionOpT reduction_op,
                                                    ArgReductionCandidateFp32 init,
                                                    const Stream& stream) {
    auto dispatch_index = [&]<typename IndexT>() -> size_t {
        auto dispatch_input = [&]<typename InputT>() -> size_t {
            return queryComposedArgReductionStageBytesForInput<InputT, IndexT>(value_input,
                                                                               carried_index_input,
                                                                               value_output,
                                                                               index_output,
                                                                               geometry,
                                                                               domain_stride,
                                                                               reduction_op,
                                                                               init,
                                                                               stream.getStream());
        };
        return dispatchReductionInputDType(value_input.getDataType(), dispatch_input);
    };
    return dispatchComposedArgIndexDType(carried_index_dtype, dispatch_index);
}

template <typename ReductionOpT>
size_t queryComposedOperationArgReductionStageBytes(DataType value_input_dtype,
                                                    bool has_carried_index_input,
                                                    std::optional<DataType> value_output_dtype,
                                                    std::optional<DataType> index_output_dtype,
                                                    const CubReductionGeometry& geometry,
                                                    uint64_t domain_stride,
                                                    DataType carried_index_dtype,
                                                    ReductionOpT reduction_op,
                                                    ArgReductionCandidateFp32 init,
                                                    const Stream& stream) {
    auto dispatch_index = [&]<typename IndexT>() -> size_t {
        auto dispatch_input = [&]<typename InputT>() -> size_t {
            return queryComposedArgReductionStageBytesForInput<InputT, IndexT>(has_carried_index_input,
                                                                               value_output_dtype,
                                                                               index_output_dtype,
                                                                               geometry,
                                                                               domain_stride,
                                                                               reduction_op,
                                                                               init,
                                                                               stream.getStream());
        };
        return dispatchReductionInputDType(value_input_dtype, dispatch_input);
    };
    return dispatchComposedArgIndexDType(carried_index_dtype, dispatch_index);
}

template <typename ReductionOpT>
void launchComposedOperationArgReductionStage(const Tensor& temp_storage,
                                              size_t temp_storage_bytes,
                                              const Tensor& value_input,
                                              const Tensor* carried_index_input,
                                              Tensor* value_output,
                                              Tensor* index_output,
                                              const CubReductionGeometry& geometry,
                                              uint64_t domain_stride,
                                              DataType carried_index_dtype,
                                              ReductionOpT reduction_op,
                                              ArgReductionCandidateFp32 init,
                                              Stream& stream) {
    auto dispatch_index = [&]<typename IndexT>() -> void {
        auto dispatch_input = [&]<typename InputT>() -> void {
            launchComposedArgReductionStageForInput<InputT, IndexT>(temp_storage,
                                                                    temp_storage_bytes,
                                                                    value_input,
                                                                    carried_index_input,
                                                                    value_output,
                                                                    index_output,
                                                                    geometry,
                                                                    domain_stride,
                                                                    reduction_op,
                                                                    init,
                                                                    stream.getStream());
        };
        dispatchReductionInputDType(value_input.getDataType(), dispatch_input);
    };
    dispatchComposedArgIndexDType(carried_index_dtype, dispatch_index);
}

}  // namespace ThorImplementation::CubReductionInternal
