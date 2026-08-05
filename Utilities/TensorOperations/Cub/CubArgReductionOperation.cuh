#pragma once

#include "Utilities/Expression/CudaHelpers.h"
#include "Utilities/TensorOperations/Cub/CubReductionIndexing.cuh"
#include "Utilities/TensorOperations/Cub/CubReductionOperation.cuh"

#include <cub/device/device_reduce.cuh>
#include <cub/device/device_segmented_reduce.cuh>
#include <cuda/iterator>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace ThorImplementation::CubReductionInternal {

struct ArgReductionCandidateFp32 {
    uint64_t index;
    float value;
};

static_assert(std::is_trivially_copyable_v<ArgReductionCandidateFp32>);

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

inline __host__ __device__ void storeArgIndexAsRuntimeDType(void* output,
                                                            DataType output_dtype,
                                                            uint64_t output_index,
                                                            uint64_t value) {
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

    template <typename IndexT>
    __host__ __device__ void operator()(IndexT output_index, ArgReductionCandidateFp32 candidate) const {
        const uint64_t index = static_cast<uint64_t>(output_index);
        if (value_output != nullptr) {
            storeFp32AsRuntimeDType(value_output, value_output_dtype, index, candidate.value);
        }
        if (index_output != nullptr) {
            storeArgIndexAsRuntimeDType(index_output, index_output_dtype, index, candidate.index);
        }
    }
};

inline auto makeRuntimeArgReductionOutputIterator(Tensor* value_output, Tensor* index_output) {
    return cuda::make_tabulate_output_iterator(StoreArgReductionResultRuntime{
        value_output == nullptr ? nullptr : value_output->getMemPtr<void>(),
        value_output == nullptr ? DataType::FP32 : value_output->getDataType(),
        index_output == nullptr ? nullptr : index_output->getMemPtr<void>(),
        index_output == nullptr ? DataType::UINT32 : index_output->getDataType()});
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
    return thrust::make_transform_iterator(
        thrust::counting_iterator<int64_t>(0),
        ContiguousArgCandidateInput<InputT>{input.getMemPtr<InputT>(), geometry.reduction_size});
}

template <typename InputT>
auto makeStridedArgCandidateIterator(const Tensor& input, const CubReductionGeometry& geometry) {
    return thrust::make_transform_iterator(
        thrust::counting_iterator<int64_t>(0),
        StridedArgCandidateInput<InputT>{input.getMemPtr<InputT>(), geometry.reduction_size, geometry.device_indexing});
}


template <typename IndexT>
struct DenseArgReductionCandidateFp32 {
    IndexT index;
    float value;
};

// ARG reductions carry both a value and an index for every live accumulator. The dense value reducer can profitably
// carry up to 16 scalar accumulators per lane, but ARG's doubled accumulator state becomes register/dependency heavy
// for long reductions. Keep the exact-width ARG fast path to at most four candidate pairs per lane and scale width
// horizontally across warps/blocks instead.
constexpr int DENSE_ARG_MAX_ITEMS_PER_LANE = 4;
constexpr uint64_t DENSE_ARG_COMPONENTS_PER_BLOCK =
    static_cast<uint64_t>(TILED_REDUCTION_BLOCK_THREADS) * DENSE_ARG_MAX_ITEMS_PER_LANE;

template <typename IndexT>
[[nodiscard]] __host__ __device__ inline DenseArgReductionCandidateFp32<IndexT> makeDenseArgReductionInit(
    ArgReductionCandidateFp32 init) {
    return DenseArgReductionCandidateFp32<IndexT>{static_cast<IndexT>(~IndexT{0}), init.value};
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
        storeArgIndexAsRuntimeDType(
            index_output, index_output_dtype, output_index, static_cast<uint64_t>(candidate.index));
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

template <typename InputT, typename ReductionOpT, typename IndexT, int ItemsPerLane>
__global__ void vectorizedDirectFullRowArgReductionKernel(const InputT* input,
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
    const uint64_t block_work_stride =
        static_cast<uint64_t>(gridDim.x) * static_cast<uint64_t>(TILED_REDUCTION_WARPS_PER_BLOCK);

    if (inner_size != expected_inner_size) {
        return;
    }

    for (uint64_t block_work_base =
             static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(TILED_REDUCTION_WARPS_PER_BLOCK);
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
            const PackedInputValues<InputT, ItemsPerLane> values =
                loadVectorizedInputPacket<InputT, ItemsPerLane>(input + row_base + lane_component_begin);
            const IndexT row_index = static_cast<IndexT>(row);
#pragma unroll
            for (int item = 0; item < ItemsPerLane; ++item) {
                local[item] = reduction_op(local[item], CandidateT{row_index, ToFp32<InputT>{}(values.values[item])});
            }
            row_base += inner_size;
        }

#pragma unroll
        for (int item = 0; item < ItemsPerLane; ++item) {
            const uint64_t component = lane_component_begin + static_cast<uint64_t>(item);
            storeDenseArgReductionResult(value_output,
                                         value_output_dtype,
                                         index_output,
                                         index_output_dtype,
                                         outer_index * inner_size + component,
                                         local[item]);
        }
    }
}

template <typename InputT, typename ReductionOpT, typename IndexT, int ItemsPerLane>
void launchVectorizedDirectFullRowArgReduction(const InputT* input,
                                               void* value_output,
                                               DataType value_output_dtype,
                                               void* index_output,
                                               DataType index_output_dtype,
                                               const CubReductionGeometry& geometry,
                                               ReductionOpT reduction_op,
                                               ArgReductionCandidateFp32 init,
                                               cudaStream_t stream) {
    const uint64_t required_blocks =
        ceilDivideU64(geometry.outer_size, static_cast<uint64_t>(TILED_REDUCTION_WARPS_PER_BLOCK));
    const unsigned int grid_blocks = static_cast<unsigned int>(
        std::min<uint64_t>(required_blocks, TILED_REDUCTION_MAX_GRID_BLOCKS));

    vectorizedDirectFullRowArgReductionKernel<InputT, ReductionOpT, IndexT, ItemsPerLane>
        <<<grid_blocks, TILED_REDUCTION_BLOCK_THREADS, 0, stream>>>(input,
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
    constexpr uint64_t expected_inner_size =
        static_cast<uint64_t>(group_threads) * static_cast<uint64_t>(ItemsPerLane);

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
            const PackedInputValues<InputT, ItemsPerLane> values =
                loadVectorizedInputPacket<InputT, ItemsPerLane>(input + row_base + component_begin);
            const IndexT row_index = static_cast<IndexT>(row);
#pragma unroll
            for (int item = 0; item < ItemsPerLane; ++item) {
                local[item] = reduction_op(local[item], CandidateT{row_index, ToFp32<InputT>{}(values.values[item])});
            }
            row_base += inner_size;
        }

#pragma unroll
        for (int item = 0; item < ItemsPerLane; ++item) {
            const uint64_t component = component_begin + static_cast<uint64_t>(item);
            storeDenseArgReductionResult(value_output,
                                         value_output_dtype,
                                         index_output,
                                         index_output_dtype,
                                         outer_index * inner_size + component,
                                         local[item]);
        }
    }
}

template <typename InputT, typename ReductionOpT, typename IndexT, int WarpsPerOutput, int ItemsPerLane>
void launchVectorizedDirectGroupedFullRowArgReduction(const InputT* input,
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
    const unsigned int grid_blocks = static_cast<unsigned int>(
        std::min<uint64_t>(required_blocks, TILED_REDUCTION_MAX_GRID_BLOCKS));

    vectorizedDirectGroupedFullRowArgReductionKernel<InputT, ReductionOpT, IndexT, WarpsPerOutput, ItemsPerLane>
        <<<grid_blocks, TILED_REDUCTION_BLOCK_THREADS, 0, stream>>>(input,
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
    constexpr uint64_t components_per_block =
        static_cast<uint64_t>(TILED_REDUCTION_BLOCK_THREADS) * static_cast<uint64_t>(ItemsPerLane);
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
        const uint64_t component_begin =
            shard_begin + static_cast<uint64_t>(threadIdx.x) * static_cast<uint64_t>(ItemsPerLane);

        CandidateT local[ItemsPerLane];
#pragma unroll
        for (int item = 0; item < ItemsPerLane; ++item) {
            local[item] = makeDenseArgReductionInit<IndexT>(init);
        }

        uint64_t row_base = outer_index * reduction_size * inner_size + component_begin;
        for (uint64_t row = 0; row < reduction_size; ++row) {
            const PackedInputValues<InputT, ItemsPerLane> values =
                loadVectorizedInputPacket<InputT, ItemsPerLane>(input + row_base);
            const IndexT row_index = static_cast<IndexT>(row);
#pragma unroll
            for (int item = 0; item < ItemsPerLane; ++item) {
                local[item] = reduction_op(local[item], CandidateT{row_index, ToFp32<InputT>{}(values.values[item])});
            }
            row_base += inner_size;
        }

#pragma unroll
        for (int item = 0; item < ItemsPerLane; ++item) {
            const uint64_t component = component_begin + static_cast<uint64_t>(item);
            storeDenseArgReductionResult(value_output,
                                         value_output_dtype,
                                         index_output,
                                         index_output_dtype,
                                         outer_index * inner_size + component,
                                         local[item]);
        }
    }
}

template <typename InputT, typename ReductionOpT, typename IndexT, int ItemsPerLane>
void launchVectorizedDirectBlockShardedArgReduction(const InputT* input,
                                                    void* value_output,
                                                    DataType value_output_dtype,
                                                    void* index_output,
                                                    DataType index_output_dtype,
                                                    const CubReductionGeometry& geometry,
                                                    ReductionOpT reduction_op,
                                                    ArgReductionCandidateFp32 init,
                                                    cudaStream_t stream) {
    static_assert(ItemsPerLane == DENSE_ARG_MAX_ITEMS_PER_LANE);
    constexpr uint64_t components_per_block =
        static_cast<uint64_t>(TILED_REDUCTION_BLOCK_THREADS) * static_cast<uint64_t>(ItemsPerLane);
    if (geometry.inner_size % components_per_block != 0) {
        throw std::logic_error("Vectorized block-sharded arg reduction requires an exact 1024-component shard width.");
    }
    const uint64_t component_shards = geometry.inner_size / components_per_block;
    const uint64_t total_work = geometry.outer_size * component_shards;
    const unsigned int grid_blocks = static_cast<unsigned int>(
        std::min<uint64_t>(total_work, TILED_REDUCTION_MAX_GRID_BLOCKS));

    vectorizedDirectBlockShardedArgReductionKernel<InputT, ReductionOpT, IndexT, ItemsPerLane>
        <<<grid_blocks, TILED_REDUCTION_BLOCK_THREADS, 0, stream>>>(input,
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
[[nodiscard]] __device__ inline PackedInputValues<InputT, ItemsPerLane> loadPacketAlignedArgInputPacket(
    const InputT* source) {
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
[[nodiscard]] __device__ inline PackedInputValues<InputT, ItemsPerLane> loadAlignmentSafeArgInputPacket(
    const InputT* source) {
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
        const int element_offset = static_cast<int>(
            (reinterpret_cast<uintptr_t>(source) & uintptr_t{packet_bytes - 1}) / sizeof(InputT));

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

template <typename InputT, typename ReductionOpT, typename IndexT, int ItemsPerLane>
__global__ void alignmentSafeVectorizedShardRangeArgReductionKernel(const InputT* input,
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
        const uint64_t component_in_shard =
            static_cast<uint64_t>(threadIdx.x) * static_cast<uint64_t>(ItemsPerLane);
        const uint64_t component_begin = shard_begin + component_in_shard;

        CandidateT local[ItemsPerLane];
#pragma unroll
        for (int item = 0; item < ItemsPerLane; ++item) {
            local[item] = makeDenseArgReductionInit<IndexT>(init);
        }

        uint64_t row_base = outer_index * reduction_size * inner_size + component_begin;
        for (uint64_t row = 0; row < reduction_size; ++row) {
            const PackedInputValues<InputT, ItemsPerLane> values =
                loadAlignmentSafeArgInputPacket<InputT, ItemsPerLane>(input + row_base);
            const IndexT row_index = static_cast<IndexT>(row);
#pragma unroll
            for (int item = 0; item < ItemsPerLane; ++item) {
                local[item] = reduction_op(local[item], CandidateT{row_index, ToFp32<InputT>{}(values.values[item])});
            }
            row_base += inner_size;
        }

#pragma unroll
        for (int item = 0; item < ItemsPerLane; ++item) {
            const uint64_t item_in_shard = component_in_shard + static_cast<uint64_t>(item);
            if (item_in_shard < shard_width) {
                const uint64_t component = shard_begin + item_in_shard;
                storeDenseArgReductionResult(value_output,
                                             value_output_dtype,
                                             index_output,
                                             index_output_dtype,
                                             outer_index * inner_size + component,
                                             local[item]);
            }
        }
    }
}

template <typename InputT, typename ReductionOpT, typename IndexT, int ItemsPerLane>
void launchAlignmentSafeVectorizedArgShardRange(const InputT* input,
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
    const unsigned int grid_blocks = static_cast<unsigned int>(
        std::min<uint64_t>(total_work, TILED_REDUCTION_MAX_GRID_BLOCKS));

    alignmentSafeVectorizedShardRangeArgReductionKernel<InputT, ReductionOpT, IndexT, ItemsPerLane>
        <<<grid_blocks, block_threads, 0, stream>>>(input,
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
        launchAlignmentSafeVectorizedArgShardRange<InputT, ReductionOpT, IndexT, ItemsPerLane>(
            input,
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
    using WarpReduceT = cub::WarpReduce<CandidateT, RowLanes>;

    __shared__ typename WarpReduceT::TempStorage
        logical_warp_storage[TILED_REDUCTION_BLOCK_THREADS / RowLanes];
    __shared__ CandidateT warp_partials[TILED_REDUCTION_WARPS_PER_BLOCK * components_per_warp];

    const int groups_per_block = TILED_REDUCTION_WARPS_PER_BLOCK / warps_per_tile;
    const int physical_warp = static_cast<int>(threadIdx.x) / TILED_REDUCTION_WARP_THREADS;
    const int lane = static_cast<int>(threadIdx.x) % TILED_REDUCTION_WARP_THREADS;
    const int group_in_block = physical_warp / warps_per_tile;
    const int warp_in_tile = physical_warp % warps_per_tile;
    const int component_in_tile = lane / RowLanes;
    const int row_lane = lane % RowLanes;
    const int logical_warp_in_block = static_cast<int>(threadIdx.x) / RowLanes;

    const uint64_t component_tiles = ceilDivideU64(inner_size, static_cast<uint64_t>(components_per_warp));
    const uint64_t total_work = outer_size * component_tiles;
    const uint64_t block_work_stride = static_cast<uint64_t>(gridDim.x) * static_cast<uint64_t>(groups_per_block);
    const CandidateT init_candidate = makeDenseArgReductionInit<IndexT>(init);

    for (uint64_t block_work_base =
             static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(groups_per_block);
         block_work_base < total_work;
         block_work_base += block_work_stride) {
        const uint64_t work_index = block_work_base + static_cast<uint64_t>(group_in_block);
        const bool work_active = work_index < total_work;
        const uint64_t outer_index = work_active ? work_index / component_tiles : 0;
        const uint64_t component_tile = work_active ? work_index - outer_index * component_tiles : 0;
        const uint64_t component =
            component_tile * static_cast<uint64_t>(components_per_warp) + static_cast<uint64_t>(component_in_tile);
        const bool component_active = work_active && component < inner_size;

        CandidateT local = init_candidate;
        if (component_active) {
            const uint64_t first_row = static_cast<uint64_t>(warp_in_tile * RowLanes + row_lane);
            const uint64_t row_stride = static_cast<uint64_t>(warps_per_tile * RowLanes);
            if (first_row < reduction_size) {
                uint64_t input_index = ((outer_index * reduction_size + first_row) * inner_size) + component;
                const uint64_t input_stride = row_stride * inner_size;
                for (uint64_t row = first_row; row < reduction_size; row += row_stride) {
                    local = reduction_op(local,
                                         CandidateT{static_cast<IndexT>(row), ToFp32<InputT>{}(input[input_index])});
                    input_index += input_stride;
                }
            }
        }

        CandidateT warp_partial = local;
        if constexpr (RowLanes > 1) {
            warp_partial = WarpReduceT(logical_warp_storage[logical_warp_in_block]).Reduce(local, reduction_op);
        }

        if (warps_per_tile == 1) {
            if (component_active && row_lane == 0) {
                storeDenseArgReductionResult(value_output,
                                             value_output_dtype,
                                             index_output,
                                             index_output_dtype,
                                             outer_index * inner_size + component,
                                             warp_partial);
            }
            if constexpr (RowLanes > 1) {
                __syncwarp();
            }
        } else {
            if (row_lane == 0) {
                warp_partials[physical_warp * components_per_warp + component_in_tile] =
                    component_active ? warp_partial : init_candidate;
            }
            __syncthreads();

            if (warp_in_tile == 0 && row_lane == 0 && component_active) {
                CandidateT aggregate = init_candidate;
                const int first_warp_in_group = group_in_block * warps_per_tile;
                for (int cooperating_warp = 0; cooperating_warp < warps_per_tile; ++cooperating_warp) {
                    aggregate = reduction_op(
                        aggregate,
                        warp_partials[(first_warp_in_group + cooperating_warp) * components_per_warp
                                      + component_in_tile]);
                }
                storeDenseArgReductionResult(value_output,
                                             value_output_dtype,
                                             index_output,
                                             index_output_dtype,
                                             outer_index * inner_size + component,
                                             aggregate);
            }
            __syncthreads();
        }
    }
}

template <typename InputT, typename ReductionOpT, typename IndexT, int RowLanes>
void launchDirectTiledFixedSegmentArgReductionForRowLanes(const InputT* input,
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
    const unsigned int grid_blocks = static_cast<unsigned int>(
        std::min<uint64_t>(required_blocks, TILED_REDUCTION_MAX_GRID_BLOCKS));

    directTiledFixedSegmentArgReductionKernel<InputT, ReductionOpT, IndexT, RowLanes>
        <<<grid_blocks, TILED_REDUCTION_BLOCK_THREADS, 0, stream>>>(input,
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

    if (geometry.inner_size == 64) {
        launchVectorizedDirectFullRowArgReduction<InputT, ReductionOpT, IndexT, 2>(input,
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
                                                                                        value_output,
                                                                                        value_output_dtype,
                                                                                        index_output,
                                                                                        index_output_dtype,
                                                                                        geometry,
                                                                                        reduction_op,
                                                                                        init,
                                                                                        stream);
    } else if (geometry.inner_size > FULL_ROW_GROUP_MAX_INNER_SIZE) {
        launchAlignmentSafeVectorizedBlockShardedArgReduction<InputT, ReductionOpT, IndexT, 4>(
            input,
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
        launchTiledFixedSegmentArgReductionForIndex<InputT, ReductionOpT, IndexT>(input,
                                                                                  value_output,
                                                                                  value_output_dtype,
                                                                                  index_output,
                                                                                  index_output_dtype,
                                                                                  geometry,
                                                                                  reduction_op,
                                                                                  init,
                                                                                  stream);
    };
    dispatchDenseArgAccumulatorIndexDType(geometry.reduction_size, dispatch_index);
}

template <typename InputT, typename ReductionOpT>
size_t queryArgReductionBytesForInput(const Tensor& input,
                                      Tensor* value_output,
                                      Tensor* index_output,
                                      const CubReductionGeometry& geometry,
                                      ReductionOpT reduction_op,
                                      ArgReductionCandidateFp32 init,
                                      cudaStream_t stream) {
    using AccumulatorT = std::decay_t<decltype(std::declval<ReductionOpT>()(
        std::declval<ArgReductionCandidateFp32>(), std::declval<ArgReductionCandidateFp32>()))>;
    static_assert(std::is_same_v<AccumulatorT, ArgReductionCandidateFp32>,
                  "CUB arg reductions must preserve the FP32 candidate state.");

    size_t queried_bytes = 0;
    auto output_iterator = makeRuntimeArgReductionOutputIterator(value_output, index_output);

    switch (geometry.path) {
        case CubReductionPath::DeviceTransformReduce: {
            auto input_iterator = makeDeviceArgCandidateIterator<InputT>(input);
            CUDA_CHECK(cub::DeviceReduce::Reduce(nullptr,
                                                 queried_bytes,
                                                 input_iterator,
                                                 output_iterator,
                                                 static_cast<int64_t>(geometry.input_elements),
                                                 reduction_op,
                                                 init,
                                                 stream));
            break;
        }
        case CubReductionPath::ContiguousFixedSegment: {
            auto input_iterator = makeContiguousArgCandidateIterator<InputT>(input, geometry);
            CUDA_CHECK(cub::DeviceSegmentedReduce::Reduce(nullptr,
                                                          queried_bytes,
                                                          input_iterator,
                                                          output_iterator,
                                                          static_cast<int64_t>(geometry.output_elements),
                                                          static_cast<int>(geometry.reduction_size),
                                                          reduction_op,
                                                          init,
                                                          stream));
            break;
        }
        case CubReductionPath::TiledFixedSegment:
            // Thor-owned tiled arg kernels use no stamped dynamic workspace.
            queried_bytes = 1;
            break;
        case CubReductionPath::StridedFixedSegment: {
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
            break;
        }
        case CubReductionPath::OffsetSegmented:
            throw std::logic_error("Dense CUB arg reduction received offset-segmented geometry.");
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
            auto input_iterator = makeDeviceArgCandidateIterator<InputT>(input);
            CUDA_CHECK(cub::DeviceReduce::Reduce(temp_storage_ptr,
                                                 temp_storage_bytes,
                                                 input_iterator,
                                                 output_iterator,
                                                 static_cast<int64_t>(geometry.input_elements),
                                                 reduction_op,
                                                 init,
                                                 stream));
            break;
        }
        case CubReductionPath::ContiguousFixedSegment: {
            auto input_iterator = makeContiguousArgCandidateIterator<InputT>(input, geometry);
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
        return queryArgReductionBytesForInput<InputT>(input,
                                                      value_output,
                                                      index_output,
                                                      geometry,
                                                      reduction_op,
                                                      init,
                                                      stream.getStream());
    };
    return dispatchReductionInputDType(input.getDataType(), dispatch_input);
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
        launchArgReductionForInput<InputT>(temp_storage,
                                           temp_storage_bytes,
                                           input,
                                           value_output,
                                           index_output,
                                           geometry,
                                           reduction_op,
                                           init,
                                           stream.getStream());
    };
    dispatchReductionInputDType(input.getDataType(), dispatch_input);
}

}  // namespace ThorImplementation::CubReductionInternal
