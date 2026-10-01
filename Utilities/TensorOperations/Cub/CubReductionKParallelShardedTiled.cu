#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"
#include "Utilities/TensorOperations/Cub/CubReductionOperation.cuh"

#include "Utilities/Exceptions.h"

#include <cuda/std/bit>
#include <cuda/std/functional>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace ThorImplementation::CubReductionInternal {
namespace {

template <typename ReductionOpT>
struct KParallelReductionIdentityFp32;

template <>
struct KParallelReductionIdentityFp32<cuda::std::plus<float>> {
    static constexpr float value = 0.0f;
};

template <>
struct KParallelReductionIdentityFp32<cuda::std::multiplies<float>> {
    static constexpr float value = 1.0f;
};

template <>
struct KParallelReductionIdentityFp32<PropagatingMinimumFp32> {
    static constexpr float value = std::numeric_limits<float>::infinity();
};

template <>
struct KParallelReductionIdentityFp32<PropagatingMaximumFp32> {
    static constexpr float value = -std::numeric_limits<float>::infinity();
};

template <int ItemsPerLane>
__device__ inline void storeKParallelFp32Packet(float* destination,
                                                const float (&values)[ItemsPerLane],
                                                uint64_t valid_items) {
    static_assert(ItemsPerLane == 1 || ItemsPerLane == 2 || ItemsPerLane == 4 || ItemsPerLane == 8);
    constexpr size_t store_bytes = sizeof(float) * ItemsPerLane;
    using RawT = RawVectorPacket<store_bytes>;

    if (valid_items == static_cast<uint64_t>(ItemsPerLane)
        && (reinterpret_cast<uintptr_t>(destination) % alignof(RawT)) == 0) {
        PackedInputValues<float, ItemsPerLane> packed{};
#pragma unroll
        for (int item = 0; item < ItemsPerLane; ++item) {
            packed.values[item] = values[item];
        }
        static_assert(sizeof(RawT) == sizeof(packed));
        *reinterpret_cast<RawT*>(destination) = cuda::std::bit_cast<RawT>(packed);
        return;
    }

#pragma unroll
    for (int item = 0; item < ItemsPerLane; ++item) {
        if (static_cast<uint64_t>(item) < valid_items) {
            destination[item] = values[item];
        }
    }
}

template <typename OutputT>
__device__ inline OutputT convertKParallelFp32(float value) {
    if constexpr (std::is_same_v<OutputT, __half>) {
        return __float2half_rn(value);
    } else if constexpr (std::is_same_v<OutputT, __nv_bfloat16>) {
        return __float2bfloat16_rn(value);
    } else {
        static_assert(std::is_same_v<OutputT, float>);
        return value;
    }
}

template <typename OutputT, typename OutputFinalizeT, int ItemsPerLane>
__device__ inline void storeKParallelFinalPacketTyped(OutputT* output,
                                                       uint64_t outer_index,
                                                       uint64_t packet_component_begin,
                                                       uint64_t output_outer_stride,
                                                       uint64_t output_inner_stride,
                                                       const float (&values)[ItemsPerLane],
                                                       uint64_t valid_items,
                                                       OutputFinalizeT output_finalize,
                                                       float output_scale) {
    static_assert(ItemsPerLane == 1 || ItemsPerLane == 2 || ItemsPerLane == 4 || ItemsPerLane == 8);
    float finalized[ItemsPerLane];
#pragma unroll
    for (int item = 0; item < ItemsPerLane; ++item) {
        finalized[item] = output_finalize(values[item]) * output_scale;
    }

    // Preserve the same successful packet-first epilogue pattern used by Thor's modern tiled reducers. The common
    // natural [outer,K] output emits one aligned packet per lane. Non-contiguous output layouts or a misaligned row
    // head retain correctness through scalar stores; the TiledRK planner can reject those geometries when a wide
    // packet store is required for the fast path.
    if (output_inner_stride == 1 && valid_items == static_cast<uint64_t>(ItemsPerLane)) {
        const uint64_t output_index = tiledReductionOutputIndex(
            outer_index, packet_component_begin, output_outer_stride, output_inner_stride);
        OutputT* destination = output + output_index;
        constexpr size_t store_bytes = sizeof(OutputT) * ItemsPerLane;
        using RawT = RawVectorPacket<store_bytes>;
        if ((reinterpret_cast<uintptr_t>(destination) % alignof(RawT)) == 0) {
            PackedInputValues<OutputT, ItemsPerLane> packed{};
#pragma unroll
            for (int item = 0; item < ItemsPerLane; ++item) {
                packed.values[item] = convertKParallelFp32<OutputT>(finalized[item]);
            }
            static_assert(sizeof(RawT) == sizeof(packed));
            *reinterpret_cast<RawT*>(destination) = cuda::std::bit_cast<RawT>(packed);
            return;
        }
    }

#pragma unroll
    for (int item = 0; item < ItemsPerLane; ++item) {
        if (static_cast<uint64_t>(item) < valid_items) {
            const uint64_t component = packet_component_begin + static_cast<uint64_t>(item);
            const uint64_t output_index = tiledReductionOutputIndex(
                outer_index, component, output_outer_stride, output_inner_stride);
            output[output_index] = convertKParallelFp32<OutputT>(finalized[item]);
        }
    }
}

template <typename OutputFinalizeT, int ItemsPerLane>
__device__ inline void storeKParallelFinalPacket(void* output,
                                                  DataType output_dtype,
                                                  uint64_t outer_index,
                                                  uint64_t packet_component_begin,
                                                  uint64_t output_outer_stride,
                                                  uint64_t output_inner_stride,
                                                  const float (&values)[ItemsPerLane],
                                                  uint64_t valid_items,
                                                  OutputFinalizeT output_finalize,
                                                  float output_scale) {
    switch (output_dtype) {
        case DataType::FP16:
            storeKParallelFinalPacketTyped<__half>(static_cast<__half*>(output),
                                                   outer_index,
                                                   packet_component_begin,
                                                   output_outer_stride,
                                                   output_inner_stride,
                                                   values,
                                                   valid_items,
                                                   output_finalize,
                                                   output_scale);
            return;
        case DataType::BF16:
            storeKParallelFinalPacketTyped<__nv_bfloat16>(static_cast<__nv_bfloat16*>(output),
                                                           outer_index,
                                                           packet_component_begin,
                                                           output_outer_stride,
                                                           output_inner_stride,
                                                           values,
                                                           valid_items,
                                                           output_finalize,
                                                           output_scale);
            return;
        case DataType::FP32:
            storeKParallelFinalPacketTyped<float>(static_cast<float*>(output),
                                                   outer_index,
                                                   packet_component_begin,
                                                   output_outer_stride,
                                                   output_inner_stride,
                                                   values,
                                                   valid_items,
                                                   output_finalize,
                                                   output_scale);
            return;
        default:
            break;
    }

    // Keep the same runtime-dtype semantic coverage as the complete R-cooperative reducer. FP16/BF16/FP32 use the
    // packetized path above; less common enabled output dtypes remain correct through the shared runtime store helper.
#pragma unroll
    for (int item = 0; item < ItemsPerLane; ++item) {
        if (static_cast<uint64_t>(item) < valid_items) {
            const uint64_t component = packet_component_begin + static_cast<uint64_t>(item);
            const uint64_t output_index = tiledReductionOutputIndex(
                outer_index, component, output_outer_stride, output_inner_stride);
            storeFp32AsRuntimeDType(
                output, output_dtype, output_index, output_finalize(values[item]) * output_scale);
        }
    }
}



template <typename InputT,
          typename InputTransformT,
          typename ReductionOpT,
          typename OutputFinalizeT,
          int ItemsPerLane,
          int BlockThreads>
__global__ __launch_bounds__(BlockThreads) void kParallelLeanCompleteKernel(const InputT* input,
                                                                           void* output,
                                                                           DataType output_dtype,
                                                                           uint64_t reduction_size,
                                                                           uint64_t inner_size,
                                                                           uint64_t output_outer_stride,
                                                                           uint64_t output_inner_stride,
                                                                           uint64_t component_tiles,
                                                                           InputTransformT input_transform,
                                                                           OutputFinalizeT output_finalize,
                                                                           float output_scale) {
    constexpr size_t packet_bytes = sizeof(InputT) * ItemsPerLane;
    static_assert(packet_bytes == 4 || packet_bytes == 8 || packet_bytes == 16);
    static_assert(BlockThreads == 32 || BlockThreads == 64 || BlockThreads == 128 || BlockThreads == 256);
    constexpr uint64_t components_per_cta =
        static_cast<uint64_t>(BlockThreads) * static_cast<uint64_t>(ItemsPerLane);
    constexpr float Init = KParallelReductionIdentityFp32<ReductionOpT>::value;
    const ReductionOpT reduction_op{};

    const uint64_t outer_component = static_cast<uint64_t>(blockIdx.x);
    const uint64_t outer_index = outer_component / component_tiles;
    const uint64_t component_tile = outer_component - outer_index * component_tiles;
    const uint64_t packet_component_begin =
        component_tile * components_per_cta
        + static_cast<uint64_t>(threadIdx.x) * static_cast<uint64_t>(ItemsPerLane);
    if (packet_component_begin >= inner_size) {
        return;
    }

    float local[ItemsPerLane];
#pragma unroll
    for (int item = 0; item < ItemsPerLane; ++item) {
        local[item] = Init;
    }

    uint64_t input_index = outer_index * reduction_size * inner_size + packet_component_begin;
    for (uint64_t row = 0; row < reduction_size; ++row) {
        const PackedInputValues<InputT, ItemsPerLane> values =
            loadVectorizedInputPacket<InputT, ItemsPerLane>(input + input_index);
#pragma unroll
        for (int item = 0; item < ItemsPerLane; ++item) {
            local[item] = reduction_op(local[item], input_transform(ToFp32<InputT>{}(values.values[item])));
        }
        input_index += inner_size;
    }

    const uint64_t remaining = inner_size - packet_component_begin;
    const uint64_t valid_items =
        remaining < static_cast<uint64_t>(ItemsPerLane) ? remaining : static_cast<uint64_t>(ItemsPerLane);
    storeKParallelFinalPacket(output,
                              output_dtype,
                              outer_index,
                              packet_component_begin,
                              output_outer_stride,
                              output_inner_stride,
                              local,
                              valid_items,
                              output_finalize,
                              output_scale);
}

template <typename InputT,
          typename InputTransformT,
          typename ReductionOpT,
          typename OutputFinalizeT,
          int ItemsPerLane,
          int BlockThreads,
          TiledRKStageTopology Topology>
__global__ __launch_bounds__(BlockThreads) void kParallelTiledStageKernel(const InputT* input,
                                                                         void* stage_output,
                                                                         DataType output_dtype,
                                                                         uint64_t outer_size,
                                                                         uint64_t reduction_size,
                                                                         uint64_t inner_size,
                                                                         uint64_t output_outer_stride,
                                                                         uint64_t output_inner_stride,
                                                                         uint64_t shards_per_output,
                                                                         InputTransformT input_transform,
                                                                         OutputFinalizeT output_finalize,
                                                                         float output_scale) {
    constexpr size_t packet_bytes = sizeof(InputT) * ItemsPerLane;
    static_assert(packet_bytes == 4 || packet_bytes == 8 || packet_bytes == 16);
    static_assert(BlockThreads == 32 || BlockThreads == 64 || BlockThreads == 128 || BlockThreads == 256);
    constexpr uint64_t components_per_cta =
        static_cast<uint64_t>(BlockThreads) * static_cast<uint64_t>(ItemsPerLane);
    constexpr float Init = KParallelReductionIdentityFp32<ReductionOpT>::value;
    const ReductionOpT reduction_op{};

    const uint64_t component_tiles = ceilDivideU64(inner_size, components_per_cta);
    const uint64_t total_work = outer_size * component_tiles * shards_per_output;
    const uint64_t grid_stride = static_cast<uint64_t>(gridDim.x);

    for (uint64_t work_index = static_cast<uint64_t>(blockIdx.x); work_index < total_work; work_index += grid_stride) {
        const uint64_t shard = work_index % shards_per_output;
        const uint64_t outer_component_work = work_index / shards_per_output;
        const uint64_t component_tile = outer_component_work % component_tiles;
        const uint64_t outer_index = outer_component_work / component_tiles;

        const uint64_t packet_component_begin =
            component_tile * components_per_cta
            + static_cast<uint64_t>(threadIdx.x) * static_cast<uint64_t>(ItemsPerLane);
        const bool owns_packet = packet_component_begin < inner_size;

        // Split R evenly instead of coupling the physical kernel to a particular rows/shard policy. The planner owns
        // shard depth. Complete topology is the exact same reduction core with one shard and a final-output epilogue.
        const uint64_t base_rows = reduction_size / shards_per_output;
        const uint64_t extra_rows = reduction_size % shards_per_output;
        const uint64_t row_begin = shard * base_rows + (shard < extra_rows ? shard : extra_rows);
        const uint64_t row_count = base_rows + static_cast<uint64_t>(shard < extra_rows);
        const uint64_t row_end = row_begin + row_count;

        float local[ItemsPerLane];
#pragma unroll
        for (int item = 0; item < ItemsPerLane; ++item) {
            local[item] = Init;
        }

        if (owns_packet) {
            uint64_t input_index =
                (outer_index * reduction_size + row_begin) * inner_size + packet_component_begin;
            for (uint64_t row = row_begin; row < row_end; ++row) {
                const PackedInputValues<InputT, ItemsPerLane> values =
                    loadVectorizedInputPacket<InputT, ItemsPerLane>(input + input_index);
#pragma unroll
                for (int item = 0; item < ItemsPerLane; ++item) {
                    local[item] = reduction_op(
                        local[item], input_transform(ToFp32<InputT>{}(values.values[item])));
                }
                input_index += inner_size;
            }

            const uint64_t remaining = inner_size - packet_component_begin;
            const uint64_t valid_items =
                remaining < static_cast<uint64_t>(ItemsPerLane) ? remaining : static_cast<uint64_t>(ItemsPerLane);
            if constexpr (Topology == TiledRKStageTopology::Staged) {
                float* fp32_partials = static_cast<float*>(stage_output);
                float* destination = fp32_partials
                                     + (outer_index * shards_per_output + shard) * inner_size
                                     + packet_component_begin;
                storeKParallelFp32Packet<ItemsPerLane>(destination, local, valid_items);
            } else {
                static_assert(Topology == TiledRKStageTopology::Complete);
                storeKParallelFinalPacket(stage_output,
                                          output_dtype,
                                          outer_index,
                                          packet_component_begin,
                                          output_outer_stride,
                                          output_inner_stride,
                                          local,
                                          valid_items,
                                          output_finalize,
                                          output_scale);
            }
        }
    }
}

template <typename InputT, size_t PacketBytes, int BlockThreads, TiledRKStageTopology Topology>
[[nodiscard]] CubKernelOccupancyInfo queryKParallelOccupancyForPacketAndBlock() {
    static_assert(PacketBytes == 4 || PacketBytes == 8 || PacketBytes == 16);
    static_assert(PacketBytes % sizeof(InputT) == 0);
    constexpr int ItemsPerLane = static_cast<int>(PacketBytes / sizeof(InputT));
    using InputTransformT = IdentityFp32;
    using ReductionOpT = cuda::std::plus<float>;
    using OutputFinalizeT = AdditiveFinalizeFp32;
    const auto kernel = [] {
        if constexpr (Topology == TiledRKStageTopology::Complete) {
            return kParallelLeanCompleteKernel<InputT,
                                               InputTransformT,
                                               ReductionOpT,
                                               OutputFinalizeT,
                                               ItemsPerLane,
                                               BlockThreads>;
        } else {
            return kParallelTiledStageKernel<InputT,
                                             InputTransformT,
                                             ReductionOpT,
                                             OutputFinalizeT,
                                             ItemsPerLane,
                                             BlockThreads,
                                             Topology>;
        }
    }();
    cudaFuncAttributes attributes{};
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
    int active_blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks, kernel, BlockThreads, 0));
    return CubKernelOccupancyInfo{attributes.numRegs,
                                  attributes.sharedSizeBytes,
                                  0,
                                  active_blocks,
                                  active_blocks * (BlockThreads / TILED_REDUCTION_WARP_THREADS)};
}

template <typename InputT, size_t PacketBytes, TiledRKStageTopology Topology>
[[nodiscard]] CubKernelOccupancyInfo queryKParallelOccupancyForPacket(uint32_t block_threads) {
    switch (block_threads) {
        case 32:
            return queryKParallelOccupancyForPacketAndBlock<InputT, PacketBytes, 32, Topology>();
        case 64:
            return queryKParallelOccupancyForPacketAndBlock<InputT, PacketBytes, 64, Topology>();
        case 128:
            return queryKParallelOccupancyForPacketAndBlock<InputT, PacketBytes, 128, Topology>();
        case 256:
            return queryKParallelOccupancyForPacketAndBlock<InputT, PacketBytes, 256, Topology>();
        default:
            throw std::logic_error("K-parallel occupancy query requires a supported CTA width.");
    }
}

template <typename InputT, TiledRKStageTopology Topology>
[[nodiscard]] CubKernelOccupancyInfo queryKParallelOccupancyForInput(size_t packet_bytes, uint32_t block_threads) {
    switch (packet_bytes) {
        case 4:
            return queryKParallelOccupancyForPacket<InputT, 4, Topology>(block_threads);
        case 8:
            return queryKParallelOccupancyForPacket<InputT, 8, Topology>(block_threads);
        case 16:
            return queryKParallelOccupancyForPacket<InputT, 16, Topology>(block_threads);
        default:
            throw std::logic_error("K-parallel occupancy query requires a 4-, 8-, or 16-byte packet.");
    }
}

template <typename InputT,
          typename InputTransformT,
          typename ReductionOpT,
          typename OutputFinalizeT,
          size_t PacketBytes,
          int BlockThreads,
          TiledRKStageTopology Topology>
void launchKParallelForPacketAndBlock(const Tensor& input,
                                      Tensor& stage_output,
                                      const CubReductionGeometry& geometry,
                                      uint64_t shards_per_output,
                                      InputTransformT input_transform,
                                      OutputFinalizeT output_finalize,
                                      float output_scale,
                                      Stream& stream) {
    static_assert(PacketBytes == 4 || PacketBytes == 8 || PacketBytes == 16);
    static_assert(PacketBytes % sizeof(InputT) == 0);
    static_assert(BlockThreads == 32 || BlockThreads == 64 || BlockThreads == 128 || BlockThreads == 256);
    constexpr int items_per_lane = static_cast<int>(PacketBytes / sizeof(InputT));
    constexpr uint64_t components_per_cta =
        static_cast<uint64_t>(BlockThreads) * static_cast<uint64_t>(items_per_lane);

    const uint64_t component_tiles = ceilDivideU64(geometry.inner_size, components_per_cta);
    if (geometry.outer_size > std::numeric_limits<uint64_t>::max() / component_tiles
        || geometry.outer_size * component_tiles > std::numeric_limits<uint64_t>::max() / shards_per_output) {
        throw std::overflow_error("K-parallel tiled stage work count overflows uint64_t.");
    }
    const uint64_t total_work = geometry.outer_size * component_tiles * shards_per_output;
    const unsigned int grid_blocks = static_cast<unsigned int>(
        std::min<uint64_t>(total_work, TILED_REDUCTION_MAX_GRID_BLOCKS));
    if (grid_blocks == 0) {
        throw std::logic_error("K-parallel tiled stage requires non-zero work.");
    }

    if constexpr (Topology == TiledRKStageTopology::Complete) {
        if (shards_per_output != 1) {
            throw std::logic_error("Lean K-parallel Complete requires exactly one R shard.");
        }
        kParallelLeanCompleteKernel<InputT,
                                    InputTransformT,
                                    ReductionOpT,
                                    OutputFinalizeT,
                                    items_per_lane,
                                    BlockThreads>
            <<<grid_blocks, BlockThreads, 0, stream.getStream()>>>(input.getMemPtr<InputT>(),
                                                                   stage_output.getMemPtr(),
                                                                   stage_output.getDataType(),
                                                                   geometry.reduction_size,
                                                                   geometry.inner_size,
                                                                   geometry.tiled_output_outer_stride,
                                                                   geometry.tiled_output_inner_stride,
                                                                   component_tiles,
                                                                   input_transform,
                                                                   output_finalize,
                                                                   output_scale);
    } else {
        kParallelTiledStageKernel<InputT,
                                  InputTransformT,
                                  ReductionOpT,
                                  OutputFinalizeT,
                                  items_per_lane,
                                  BlockThreads,
                                  Topology>
            <<<grid_blocks, BlockThreads, 0, stream.getStream()>>>(input.getMemPtr<InputT>(),
                                                                   stage_output.getMemPtr(),
                                                                   stage_output.getDataType(),
                                                                   geometry.outer_size,
                                                                   geometry.reduction_size,
                                                                   geometry.inner_size,
                                                                   geometry.tiled_output_outer_stride,
                                                                   geometry.tiled_output_inner_stride,
                                                                   shards_per_output,
                                                                   input_transform,
                                                                   output_finalize,
                                                                   output_scale);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <typename InputT,
          typename InputTransformT,
          typename ReductionOpT,
          typename OutputFinalizeT,
          size_t PacketBytes,
          TiledRKStageTopology Topology>
void launchKParallelForPacket(const Tensor& input,
                              Tensor& stage_output,
                              const CubReductionGeometry& geometry,
                              uint32_t block_threads,
                              uint64_t shards_per_output,
                              InputTransformT input_transform,
                              OutputFinalizeT output_finalize,
                              float output_scale,
                              Stream& stream) {
    switch (block_threads) {
        case 32:
            launchKParallelForPacketAndBlock<InputT,
                                             InputTransformT,
                                             ReductionOpT,
                                             OutputFinalizeT,
                                             PacketBytes,
                                             32,
                                             Topology>(input,
                                                       stage_output,
                                                       geometry,
                                                       shards_per_output,
                                                       input_transform,
                                                       output_finalize,
                                                       output_scale,
                                                       stream);
            return;
        case 64:
            launchKParallelForPacketAndBlock<InputT,
                                             InputTransformT,
                                             ReductionOpT,
                                             OutputFinalizeT,
                                             PacketBytes,
                                             64,
                                             Topology>(input,
                                                       stage_output,
                                                       geometry,
                                                       shards_per_output,
                                                       input_transform,
                                                       output_finalize,
                                                       output_scale,
                                                       stream);
            return;
        case 128:
            launchKParallelForPacketAndBlock<InputT,
                                             InputTransformT,
                                             ReductionOpT,
                                             OutputFinalizeT,
                                             PacketBytes,
                                             128,
                                             Topology>(input,
                                                       stage_output,
                                                       geometry,
                                                       shards_per_output,
                                                       input_transform,
                                                       output_finalize,
                                                       output_scale,
                                                       stream);
            return;
        case 256:
            launchKParallelForPacketAndBlock<InputT,
                                             InputTransformT,
                                             ReductionOpT,
                                             OutputFinalizeT,
                                             PacketBytes,
                                             256,
                                             Topology>(input,
                                                       stage_output,
                                                       geometry,
                                                       shards_per_output,
                                                       input_transform,
                                                       output_finalize,
                                                       output_scale,
                                                       stream);
            return;
        default:
            throw std::logic_error("K-parallel tiled stage selected an unsupported CTA width.");
    }
}

template <typename InputT,
          typename InputTransformT,
          typename ReductionOpT,
          typename OutputFinalizeT,
          TiledRKStageTopology Topology>
void launchKParallelForInput(const Tensor& input,
                             Tensor& stage_output,
                             const CubReductionGeometry& geometry,
                             size_t packet_bytes,
                             uint32_t block_threads,
                             uint64_t shards_per_output,
                             InputTransformT input_transform,
                             OutputFinalizeT output_finalize,
                             float output_scale,
                             Stream& stream) {
    switch (packet_bytes) {
        case 16:
            launchKParallelForPacket<InputT,
                                     InputTransformT,
                                     ReductionOpT,
                                     OutputFinalizeT,
                                     16,
                                     Topology>(input,
                                               stage_output,
                                               geometry,
                                               block_threads,
                                               shards_per_output,
                                               input_transform,
                                               output_finalize,
                                               output_scale,
                                               stream);
            return;
        case 8:
            launchKParallelForPacket<InputT,
                                     InputTransformT,
                                     ReductionOpT,
                                     OutputFinalizeT,
                                     8,
                                     Topology>(input,
                                               stage_output,
                                               geometry,
                                               block_threads,
                                               shards_per_output,
                                               input_transform,
                                               output_finalize,
                                               output_scale,
                                               stream);
            return;
        case 4:
            launchKParallelForPacket<InputT,
                                     InputTransformT,
                                     ReductionOpT,
                                     OutputFinalizeT,
                                     4,
                                     Topology>(input,
                                               stage_output,
                                               geometry,
                                               block_threads,
                                               shards_per_output,
                                               input_transform,
                                               output_finalize,
                                               output_scale,
                                               stream);
            return;
        default:
            throw std::logic_error("K-parallel tiled stage selected an invalid packet width.");
    }
}

template <typename InputTransformT,
          typename ReductionOpT,
          typename OutputFinalizeT,
          TiledRKStageTopology Topology>
void dispatchInputType(const Tensor& input,
                       Tensor& stage_output,
                       const CubReductionGeometry& geometry,
                       size_t packet_bytes,
                       uint32_t block_threads,
                       uint64_t shards_per_output,
                       InputTransformT input_transform,
                       OutputFinalizeT output_finalize,
                       float output_scale,
                       Stream& stream) {
    switch (input.getDataType()) {
        case DataType::FP16:
            launchKParallelForInput<__half, InputTransformT, ReductionOpT, OutputFinalizeT, Topology>(input,
                                                                                                      stage_output,
                                                                                                      geometry,
                                                                                                      packet_bytes,
                                                                                                      block_threads,
                                                                                                      shards_per_output,
                                                                                                      input_transform,
                                                                                                      output_finalize,
                                                                                                      output_scale,
                                                                                                      stream);
            return;
        case DataType::BF16:
            launchKParallelForInput<__nv_bfloat16, InputTransformT, ReductionOpT, OutputFinalizeT, Topology>(input,
                                                                                                              stage_output,
                                                                                                              geometry,
                                                                                                              packet_bytes,
                                                                                                              block_threads,
                                                                                                              shards_per_output,
                                                                                                              input_transform,
                                                                                                              output_finalize,
                                                                                                              output_scale,
                                                                                                              stream);
            return;
        case DataType::FP32:
            launchKParallelForInput<float, InputTransformT, ReductionOpT, OutputFinalizeT, Topology>(input,
                                                                                                     stage_output,
                                                                                                     geometry,
                                                                                                     packet_bytes,
                                                                                                     block_threads,
                                                                                                     shards_per_output,
                                                                                                     input_transform,
                                                                                                     output_finalize,
                                                                                                     output_scale,
                                                                                                     stream);
            return;
        default:
            throw std::logic_error("K-parallel tiled stage received an unsupported input dtype.");
    }
}

[[nodiscard]] size_t elementBytes(DataType dtype) {
    switch (dtype) {
        case DataType::FP16:
        case DataType::BF16:
            return 2;
        case DataType::FP32:
            return 4;
        default:
            throw std::logic_error("K-parallel tiled stage received an unsupported input dtype.");
    }
}

[[nodiscard]] AdditiveFinalizeFp32 additiveFinalizeFromSemantics(const CubReductionStageSemantics& semantics) {
    switch (semantics.finalize) {
        case CubReductionStageFinalize::Identity:
            return AdditiveFinalizeFp32{1.0f, false};
        case CubReductionStageFinalize::Divide:
            return AdditiveFinalizeFp32{static_cast<float>(semantics.finalize_divisor), false};
        case CubReductionStageFinalize::SquareRoot:
            return AdditiveFinalizeFp32{1.0f, true};
    }
    throw std::logic_error("Unsupported K-parallel tiled output finalizer.");
}

template <TiledRKStageTopology Topology, typename InputTransformT, typename ReductionOpT, typename OutputFinalizeT>
void dispatchStage(const Tensor& input,
                   Tensor& stage_output,
                   const CubReductionGeometry& geometry,
                   const KParallelTiledStagePlan& plan,
                   InputTransformT input_transform,
                   OutputFinalizeT output_finalize,
                   float output_scale,
                   Stream& stream) {
    dispatchInputType<InputTransformT, ReductionOpT, OutputFinalizeT, Topology>(input,
                                                                                stage_output,
                                                                                geometry,
                                                                                plan.packet_bytes,
                                                                                plan.block_threads,
                                                                                plan.shards_per_output,
                                                                                input_transform,
                                                                                output_finalize,
                                                                                output_scale,
                                                                                stream);
}

template <TiledRKStageTopology Topology>
void dispatchStageSemantics(const CubReductionStageSemantics& semantics,
                            const Tensor& input,
                            Tensor& stage_output,
                            const CubReductionGeometry& geometry,
                            const KParallelTiledStagePlan& plan,
                            float output_scale,
                            Stream& stream) {
    switch (semantics.combine) {
        case CubReductionStageCombine::Sum: {
            const AdditiveFinalizeFp32 output_finalize = additiveFinalizeFromSemantics(semantics);
            switch (semantics.input_transform) {
                case CubReductionStageInputTransform::Identity:
                    dispatchStage<Topology, IdentityFp32, cuda::std::plus<float>>(input,
                                                                                  stage_output,
                                                                                  geometry,
                                                                                  plan,
                                                                                  IdentityFp32{},
                                                                                  output_finalize,
                                                                                  output_scale,
                                                                                  stream);
                    return;
                case CubReductionStageInputTransform::AbsoluteValue:
                    dispatchStage<Topology, AbsoluteValueFp32, cuda::std::plus<float>>(input,
                                                                                       stage_output,
                                                                                       geometry,
                                                                                       plan,
                                                                                       AbsoluteValueFp32{},
                                                                                       output_finalize,
                                                                                       output_scale,
                                                                                       stream);
                    return;
                case CubReductionStageInputTransform::Square:
                    dispatchStage<Topology, SquareFp32, cuda::std::plus<float>>(input,
                                                                                stage_output,
                                                                                geometry,
                                                                                plan,
                                                                                SquareFp32{},
                                                                                output_finalize,
                                                                                output_scale,
                                                                                stream);
                    return;
            }
            break;
        }
        case CubReductionStageCombine::Product:
            if (semantics.input_transform == CubReductionStageInputTransform::Identity
                && semantics.finalize == CubReductionStageFinalize::Identity) {
                dispatchStage<Topology, IdentityFp32, cuda::std::multiplies<float>>(input,
                                                                                     stage_output,
                                                                                     geometry,
                                                                                     plan,
                                                                                     IdentityFp32{},
                                                                                     IdentityFp32{},
                                                                                     output_scale,
                                                                                     stream);
                return;
            }
            break;
        case CubReductionStageCombine::Minimum:
            if (semantics.input_transform == CubReductionStageInputTransform::Identity
                && semantics.finalize == CubReductionStageFinalize::Identity) {
                dispatchStage<Topology, IdentityFp32, PropagatingMinimumFp32>(input,
                                                                              stage_output,
                                                                              geometry,
                                                                              plan,
                                                                              IdentityFp32{},
                                                                              IdentityFp32{},
                                                                              output_scale,
                                                                              stream);
                return;
            }
            break;
        case CubReductionStageCombine::Maximum:
            if (semantics.input_transform == CubReductionStageInputTransform::Identity
                && semantics.finalize == CubReductionStageFinalize::Identity) {
                dispatchStage<Topology, IdentityFp32, PropagatingMaximumFp32>(input,
                                                                              stage_output,
                                                                              geometry,
                                                                              plan,
                                                                              IdentityFp32{},
                                                                              IdentityFp32{},
                                                                              output_scale,
                                                                              stream);
                return;
            }
            break;
    }
    throw std::logic_error("Unsupported K-parallel tiled stage semantics.");
}

void validateCompleteOutput(const Tensor& output, const CubReductionGeometry& geometry) {
    if (geometry.outer_size > std::numeric_limits<uint64_t>::max() / geometry.inner_size) {
        throw std::overflow_error("K-parallel complete tiled output element count overflows uint64_t.");
    }
    const uint64_t expected_elements = geometry.outer_size * geometry.inner_size;
    if (output.getTotalNumElements() != expected_elements) {
        throw std::logic_error("K-parallel complete tiled output element count does not match stamped geometry.");
    }
    if (geometry.outer_size == 0 || geometry.inner_size == 0) {
        return;
    }
    const uint64_t max_outer = geometry.outer_size - 1;
    const uint64_t max_inner = geometry.inner_size - 1;
    if (max_outer != 0
        && geometry.tiled_output_outer_stride > std::numeric_limits<uint64_t>::max() / max_outer) {
        throw std::overflow_error("K-parallel complete tiled output outer index overflows uint64_t.");
    }
    const uint64_t outer_offset = max_outer * geometry.tiled_output_outer_stride;
    if (max_inner != 0
        && geometry.tiled_output_inner_stride > std::numeric_limits<uint64_t>::max() / max_inner) {
        throw std::overflow_error("K-parallel complete tiled output inner index overflows uint64_t.");
    }
    const uint64_t inner_offset = max_inner * geometry.tiled_output_inner_stride;
    if (outer_offset > std::numeric_limits<uint64_t>::max() - inner_offset
        || outer_offset + inner_offset >= output.getTotalNumElements()) {
        throw std::logic_error("K-parallel complete tiled output strides exceed the supplied output tensor.");
    }
}

}  // namespace

CubKernelOccupancyInfo queryKParallelTiledStageOccupancy(DataType input_dtype,
                                                          const KParallelTiledStagePlan& plan) {
    if (plan.packet_bytes != 4 && plan.packet_bytes != 8 && plan.packet_bytes != 16) {
        throw std::logic_error("K-parallel occupancy query requires a 4-, 8-, or 16-byte packet.");
    }
    if (plan.block_threads != 32 && plan.block_threads != 64 && plan.block_threads != 128
        && plan.block_threads != 256) {
        throw std::logic_error("K-parallel occupancy query requires a compile-time-specialized CTA width.");
    }
    const auto query_for = [&]<typename InputT>() -> CubKernelOccupancyInfo {
        if (plan.packet_bytes < sizeof(InputT) || plan.packet_bytes % sizeof(InputT) != 0) {
            throw std::logic_error("K-parallel occupancy query packet is incompatible with the input dtype.");
        }
        if (plan.topology == TiledRKStageTopology::Complete) {
            return queryKParallelOccupancyForInput<InputT, TiledRKStageTopology::Complete>(
                plan.packet_bytes, plan.block_threads);
        }
        return queryKParallelOccupancyForInput<InputT, TiledRKStageTopology::Staged>(
            plan.packet_bytes, plan.block_threads);
    };
    switch (input_dtype) {
        case DataType::FP16:
            return query_for.template operator()<__half>();
        case DataType::BF16:
            return query_for.template operator()<__nv_bfloat16>();
        case DataType::FP32:
            return query_for.template operator()<float>();
        default:
            throw std::logic_error("K-parallel occupancy query received an unsupported input dtype.");
    }
}

void launchKParallelTiledStage(const CubReductionStageSemantics& semantics,
                               const Tensor& input,
                               Tensor& stage_output,
                               const CubReductionGeometry& geometry,
                               const KParallelTiledStagePlan& plan,
                               float output_scale,
                               Stream& stream) {
    if (geometry.path != CubReductionPath::TiledFixedSegment) {
        throw std::logic_error("K-parallel tiled stage requires TiledFixedSegment geometry.");
    }
    if (geometry.outer_size == 0 || geometry.reduction_size == 0 || geometry.inner_size == 0
        || plan.shards_per_output == 0 || plan.shards_per_output > geometry.reduction_size) {
        throw std::logic_error("K-parallel tiled stage requires valid non-zero geometry.");
    }
    if (plan.packet_bytes != 4 && plan.packet_bytes != 8 && plan.packet_bytes != 16) {
        throw std::logic_error("K-parallel tiled stage requires a 4-, 8-, or 16-byte input packet.");
    }
    const size_t input_element_bytes = elementBytes(input.getDataType());
    if (plan.packet_bytes < input_element_bytes || plan.packet_bytes % input_element_bytes != 0
        || geometry.inner_size % static_cast<uint64_t>(plan.packet_bytes / input_element_bytes) != 0) {
        throw std::logic_error("K-parallel tiled stage requires naturally aligned whole-row input packets.");
    }
    if (plan.block_threads != 32 && plan.block_threads != 64 && plan.block_threads != 128
        && plan.block_threads != 256) {
        throw std::logic_error("K-parallel tiled stage requires a compile-time-specialized CTA width.");
    }

    switch (plan.topology) {
        case TiledRKStageTopology::Staged: {
            if (plan.shards_per_output <= 1) {
                throw std::logic_error("K-parallel staged topology requires more than one R shard.");
            }
            if (stage_output.getDataType() != DataType::FP32) {
                throw std::logic_error("K-parallel staged topology requires FP32 partial storage.");
            }
            const std::vector<uint64_t> expected_partial_dimensions = {
                geometry.outer_size, plan.shards_per_output, geometry.inner_size};
            if (stage_output.getDimensions() != expected_partial_dimensions) {
                throw std::logic_error("K-parallel staged FP32 partial shape does not match stamped geometry.");
            }
            // Exactly like the successful R-cooperative staged path, the physical first pass consumes only the input
            // transform and associative combine. A mathematically Final role may still arrive here; its finalizer and
            // output scale intentionally remain deferred to the independently stamped continuation.
            CubReductionStageSemantics partial_semantics = semantics;
            partial_semantics.finalize = CubReductionStageFinalize::Identity;
            partial_semantics.finalize_divisor = 1;
            dispatchStageSemantics<TiledRKStageTopology::Staged>(
                partial_semantics, input, stage_output, geometry, plan, 1.0f, stream);
            return;
        }
        case TiledRKStageTopology::Complete:
            if (plan.shards_per_output != 1) {
                throw std::logic_error("K-parallel complete topology requires exactly one R shard.");
            }
            validateCompleteOutput(stage_output, geometry);
            dispatchStageSemantics<TiledRKStageTopology::Complete>(
                semantics, input, stage_output, geometry, plan, output_scale, stream);
            return;
    }
    throw std::logic_error("K-parallel tiled stage selected an invalid topology.");
}

}  // namespace ThorImplementation::CubReductionInternal
