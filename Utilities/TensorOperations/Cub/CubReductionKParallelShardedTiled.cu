#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"
#include "Utilities/TensorOperations/Cub/CubReductionOperation.cuh"
#include "Utilities/TensorOperations/Cub/Fp8RetainedKernelPolicy.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8AlignedKSelector.h"

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

#if THOR_CUB_ENABLE_FP8_TYPES
template <typename InputT, typename TransformT, typename ReductionOpT,
          size_t PacketBytes, int BlockThreads, TiledRKStageTopology Topology>
constexpr bool prunedFp8AlignedInstantiation() {
    constexpr unsigned dtype = std::is_same_v<InputT, __nv_fp8_e4m3> ? 0u : 1u;
    constexpr unsigned combine = std::is_same_v<ReductionOpT, cuda::std::plus<float>> ? 0u :
        std::is_same_v<ReductionOpT, cuda::std::multiplies<float>> ? 1u :
        std::is_same_v<ReductionOpT, PropagatingMinimumFp32> ? 2u : 3u;
    constexpr unsigned transform = std::is_same_v<TransformT, IdentityFp32> ? 0u :
        std::is_same_v<TransformT, AbsoluteValueFp32> ? 1u : 2u;
    return fp8AlignedTypedPruned(dtype, combine, transform, PacketBytes, BlockThreads,
                                 Topology == TiledRKStageTopology::Staged);
}

// Retained FP8 AlignedK physical configurations, owned by Fp8RK::Fp8AlignedKPolicy:
// Staged at every retained packet x CTA pair, Complete only where supportedComplete().
// FP8 launch and occupancy dispatch instantiate exactly these configurations.
template <TiledRKStageTopology Topology, size_t PacketBytes, int BlockThreads>
[[nodiscard]] constexpr bool fp8AlignedKRetainedConfiguration() {
    namespace Policy = Fp8RK::Fp8AlignedKPolicy;
    bool packet = false;
    for (uint32_t retained : Policy::packet_bytes) packet = packet || retained == PacketBytes;
    bool threads = false;
    for (uint32_t retained : Policy::cta_threads) threads = threads || retained == static_cast<uint32_t>(BlockThreads);
    if constexpr (Topology == TiledRKStageTopology::Staged) {
        return packet && threads;
    } else {
        return packet && threads &&
               Policy::supportedComplete(static_cast<uint32_t>(PacketBytes), static_cast<uint32_t>(BlockThreads));
    }
}

// Invokes callback.template operator()<PacketBytes, BlockThreads>() for the retained
// configuration matching the runtime request; returns false for any other request.
template <TiledRKStageTopology Topology, typename Callback>
[[nodiscard]] bool dispatchFp8AlignedKConfiguration(size_t packet_bytes, uint32_t block_threads, Callback&& callback) {
    bool dispatched = false;
    const auto configuration = [&]<size_t PacketBytes, int BlockThreads>() {
        if constexpr (fp8AlignedKRetainedConfiguration<Topology, PacketBytes, BlockThreads>()) {
            callback.template operator()<PacketBytes, BlockThreads>();
            dispatched = true;
        }
    };
    const auto for_packet = [&]<size_t PacketBytes>() {
        switch (block_threads) {
            case 32: configuration.template operator()<PacketBytes, 32>(); return;
            case 64: configuration.template operator()<PacketBytes, 64>(); return;
            case 128: configuration.template operator()<PacketBytes, 128>(); return;
            default: return;
        }
    };
    switch (packet_bytes) {
        case 2: for_packet.template operator()<2>(); break;
        case 8: for_packet.template operator()<8>(); break;
        default: break;
    }
    return dispatched;
}
#endif

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

#if THOR_CUB_ENABLE_FP8_TYPES

// DENSE-FP8-VALUE-1 keeps the established KParallel reduction topology and extends one-byte storage across the
// naturally coalesced 16/8/4/2-byte packet widths. Each lane owns one contiguous FP8 packet, performs its complete R
// reduction in FP32 registers, and only changes physical ownership at writeback when those FP32 results would exceed
// Thor's 16-byte-per-lane global transaction ceiling. The shared writeback transpose is therefore required for staged
// packet16/packet8, but packet4 and packet2 emit their FP32 partials directly as coalesced 16-byte/8-byte lane stores.
// No other lane ever contributes to a lane's K accumulators; the transpose remains a KParallel layout epilogue, not
// RCooperative reduction work.
constexpr int K_PARALLEL_FP8_SHARED_STRIPE_PITCH = 34;

template <size_t PacketBytes>
inline constexpr int K_PARALLEL_FP8_ITEMS_PER_LANE = static_cast<int>(PacketBytes);

template <size_t PacketBytes>
inline constexpr int K_PARALLEL_FP8_COMPONENTS_PER_WARP =
    TILED_REDUCTION_WARP_THREADS * K_PARALLEL_FP8_ITEMS_PER_LANE<PacketBytes>;

template <size_t PacketBytes>
inline constexpr size_t K_PARALLEL_FP8_SHARED_BYTES_PER_WARP =
    static_cast<size_t>(K_PARALLEL_FP8_ITEMS_PER_LANE<PacketBytes>)
    * static_cast<size_t>(K_PARALLEL_FP8_SHARED_STRIPE_PITCH) * sizeof(float);

static_assert(sizeof(__nv_fp8_e4m3) == 1);
static_assert(sizeof(__nv_fp8_e5m2) == 1);
static_assert(sizeof(RawVectorPacket<2>) == 2);
static_assert(sizeof(RawVectorPacket<4>) == 4);
static_assert(sizeof(RawVectorPacket<8>) == 8);
static_assert(sizeof(RawVectorPacket<16>) == 16);

template <typename T>
inline constexpr bool IS_K_PARALLEL_FP8_TYPE =
    std::is_same_v<T, __nv_fp8_e4m3> || std::is_same_v<T, __nv_fp8_e5m2>;

[[nodiscard]] bool kParallelFp8OutputDTypeSupported(DataType dtype) {
    switch (dtype) {
        case DataType::FP8_E4M3:
        case DataType::FP8_E5M2:
        case DataType::FP16:
        case DataType::BF16:
        case DataType::FP32:
            return true;
#if THOR_CUB_ENABLE_64BIT_TYPES
        case DataType::FP64:
            return true;
#endif
        default:
            return false;
    }
}

template <typename OutputT>
__device__ inline OutputT convertKParallelFp8Writeback(float value) {
    if constexpr (std::is_same_v<OutputT, __nv_fp8_e4m3>) {
        return ThorLowPrecision::toFp8E4M3Satfinite(value);
    } else if constexpr (std::is_same_v<OutputT, __nv_fp8_e5m2>) {
        return ThorLowPrecision::toFp8E5M2Satfinite(value);
    } else if constexpr (std::is_same_v<OutputT, __half>) {
        return __float2half_rn(value);
    } else if constexpr (std::is_same_v<OutputT, __nv_bfloat16>) {
        return __float2bfloat16_rn(value);
    } else if constexpr (std::is_same_v<OutputT, float>) {
        return value;
#if THOR_CUB_ENABLE_64BIT_TYPES
    } else if constexpr (std::is_same_v<OutputT, double>) {
        return static_cast<double>(value);
#endif
    } else {
        static_assert(!sizeof(OutputT), "Unsupported FP8 KParallel output dtype.");
    }
}

template <typename OutputT, int ItemsPerLane, typename OutputFinalizeT>
__device__ inline void storeKParallelFp8DirectFinalPacket(OutputT* output,
                                                          uint64_t outer_index,
                                                          uint64_t packet_component_begin,
                                                          uint64_t output_outer_stride,
                                                          const float (&values)[ItemsPerLane],
                                                          OutputFinalizeT output_finalize,
                                                          float output_scale) {
    constexpr size_t store_bytes = sizeof(OutputT) * static_cast<size_t>(ItemsPerLane);
    static_assert(store_bytes == 2 || store_bytes == 4 || store_bytes == 8 || store_bytes == 16);
    using RawT = RawVectorPacket<store_bytes>;

    PackedInputValues<OutputT, ItemsPerLane> packed{};
#pragma unroll
    for (int item = 0; item < ItemsPerLane; ++item) {
        packed.values[item] = convertKParallelFp8Writeback<OutputT>(output_finalize(values[item]) * output_scale);
    }
    static_assert(sizeof(RawT) == sizeof(packed));
    const uint64_t output_index = outer_index * output_outer_stride + packet_component_begin;
    OutputT* destination = output + output_index;
    *reinterpret_cast<RawT*>(destination) = cuda::std::bit_cast<RawT>(packed);
}

template <typename OutputT, int ItemsPerLane>
__device__ inline void storeKParallelFp8TransposedPackets(OutputT* output,
                                                          uint64_t output_row_base,
                                                          uint64_t warp_component_begin,
                                                          uint64_t valid_components,
                                                          const float* warp_tile) {
    static_assert(sizeof(OutputT) == 1 || sizeof(OutputT) == 2 || sizeof(OutputT) == 4
#if THOR_CUB_ENABLE_64BIT_TYPES
                  || sizeof(OutputT) == 8
#endif
    );
    constexpr int items_per_store = 16 / static_cast<int>(sizeof(OutputT));
    constexpr int store_rounds = ItemsPerLane / items_per_store;
    static_assert(items_per_store == 2 || items_per_store == 4 || items_per_store == 8 || items_per_store == 16);
    static_assert(ItemsPerLane * static_cast<int>(sizeof(OutputT)) > 16);
    static_assert(store_rounds * items_per_store == ItemsPerLane);
    static_assert(sizeof(PackedInputValues<OutputT, items_per_store>) == 16);

    const uint64_t lane = static_cast<uint64_t>(threadIdx.x) % TILED_REDUCTION_WARP_THREADS;
#pragma unroll
    for (int round = 0; round < store_rounds; ++round) {
        const uint64_t component_offset =
            static_cast<uint64_t>(round) * TILED_REDUCTION_WARP_THREADS * items_per_store
            + lane * static_cast<uint64_t>(items_per_store);
        if (component_offset >= valid_components) {
            continue;
        }

        PackedInputValues<OutputT, items_per_store> packed{};
#pragma unroll
        for (int item = 0; item < items_per_store; ++item) {
            const uint64_t transposed_component = component_offset + static_cast<uint64_t>(item);
            const uint64_t producer_lane = transposed_component / static_cast<uint64_t>(ItemsPerLane);
            const uint64_t producer_item = transposed_component % static_cast<uint64_t>(ItemsPerLane);
            const float value = warp_tile[producer_item * K_PARALLEL_FP8_SHARED_STRIPE_PITCH + producer_lane];
            packed.values[item] = convertKParallelFp8Writeback<OutputT>(value);
        }

        OutputT* destination = output + output_row_base + warp_component_begin + component_offset;
        *reinterpret_cast<RawVectorPacket<16>*>(destination) =
            cuda::std::bit_cast<RawVectorPacket<16>>(packed);
    }
}

template <size_t PacketBytes>
inline constexpr bool K_PARALLEL_FP8_STAGED_NEEDS_TRANSPOSE =
    K_PARALLEL_FP8_ITEMS_PER_LANE<PacketBytes> * static_cast<int>(sizeof(float)) > 16;

template <size_t PacketBytes>
[[nodiscard]] bool kParallelFp8CompleteNeedsTranspose(DataType output_dtype) {
    static_assert(PacketBytes == 2 || PacketBytes == 4 || PacketBytes == 8 || PacketBytes == 16);
    const size_t output_bytes = TensorDescriptor::getElementSizeInBytes(output_dtype);
    if (output_bytes == 0) {
        throw std::logic_error("FP8 K-parallel writeback received an invalid output dtype.");
    }
    return PacketBytes * output_bytes > 16;
}

template <typename InputT,
          typename InputTransformT,
          typename ReductionOpT,
          typename OutputFinalizeT,
          size_t PacketBytes,
          int BlockThreads,
          TiledRKStageTopology Topology>
__global__ __launch_bounds__(BlockThreads) void kParallelFp8PacketKernel(const InputT* input,
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
    static_assert(IS_K_PARALLEL_FP8_TYPE<InputT>);
    static_assert(PacketBytes == 2 || PacketBytes == 4 || PacketBytes == 8 || PacketBytes == 16);
    constexpr int ItemsPerLane = K_PARALLEL_FP8_ITEMS_PER_LANE<PacketBytes>;
    static_assert(sizeof(InputT) * ItemsPerLane == PacketBytes);
    static_assert(BlockThreads == 32 || BlockThreads == 64 || BlockThreads == 128 || BlockThreads == 256);
    static_assert(BlockThreads % TILED_REDUCTION_WARP_THREADS == 0);
    static_cast<void>(output_inner_stride);

    constexpr uint64_t components_per_cta =
        static_cast<uint64_t>(BlockThreads) * static_cast<uint64_t>(ItemsPerLane);
    constexpr float Init = KParallelReductionIdentityFp32<ReductionOpT>::value;
    const ReductionOpT reduction_op{};

    extern __shared__ float fp8_writeback_shared[];
    const int warp = static_cast<int>(threadIdx.x) / TILED_REDUCTION_WARP_THREADS;
    const int lane = static_cast<int>(threadIdx.x) % TILED_REDUCTION_WARP_THREADS;
    float* warp_tile = fp8_writeback_shared
                       + static_cast<size_t>(warp) * static_cast<size_t>(ItemsPerLane)
                             * K_PARALLEL_FP8_SHARED_STRIPE_PITCH;

    const uint64_t component_tiles = ceilDivideU64(inner_size, components_per_cta);
    const uint64_t total_work = outer_size * component_tiles * shards_per_output;
    const uint64_t grid_stride = static_cast<uint64_t>(gridDim.x);

    for (uint64_t work_index = static_cast<uint64_t>(blockIdx.x); work_index < total_work; work_index += grid_stride) {
        const uint64_t shard = work_index % shards_per_output;
        const uint64_t outer_component_work = work_index / shards_per_output;
        const uint64_t component_tile = outer_component_work % component_tiles;
        const uint64_t outer_index = outer_component_work / component_tiles;
        const uint64_t tile_component_begin = component_tile * components_per_cta;
        const uint64_t packet_component_begin =
            tile_component_begin + static_cast<uint64_t>(threadIdx.x) * static_cast<uint64_t>(ItemsPerLane);
        const bool owns_packet = packet_component_begin < inner_size;

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

            // Whole-row packet divisibility means every row start preserves PacketBytes alignment whenever the tensor
            // base does. That is the normal allocator path for all 16/8/4/2-byte specializations. Packet16 retains the
            // existing shifted-alias correctness path because its 16-byte aligned-window loader is specifically built
            // for that geometry; narrower packets use their native coalesced vector load directly.
            if constexpr (PacketBytes == 16) {
                if ((reinterpret_cast<uintptr_t>(input) & uintptr_t{15}) == 0) {
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
                } else {
                    for (uint64_t row = row_begin; row < row_end; ++row) {
                        const PackedInputValues<InputT, ItemsPerLane> values =
                            loadAlignmentSafeInputPacket<InputT, ItemsPerLane>(input + input_index);
#pragma unroll
                        for (int item = 0; item < ItemsPerLane; ++item) {
                            local[item] = reduction_op(
                                local[item], input_transform(ToFp32<InputT>{}(values.values[item])));
                        }
                        input_index += inner_size;
                    }
                }
            } else {
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
            }
        }

        if constexpr (Topology == TiledRKStageTopology::Staged && !K_PARALLEL_FP8_STAGED_NEEDS_TRANSPOSE<PacketBytes>) {
            if (owns_packet) {
                float* fp32_partials = static_cast<float*>(stage_output);
                float* destination = fp32_partials
                                     + (outer_index * shards_per_output + shard) * inner_size
                                     + packet_component_begin;
                // packet4 => one coalesced 16-byte FP32 store/lane; packet2 => one coalesced 8-byte store/lane.
                storeKParallelFp32Packet<ItemsPerLane>(destination, local, ItemsPerLane);
            }
            continue;
        }

        if constexpr (Topology == TiledRKStageTopology::Complete) {
            if (owns_packet) {
                switch (output_dtype) {
                    case DataType::FP8_E4M3:
                        if constexpr (sizeof(__nv_fp8_e4m3) * static_cast<size_t>(ItemsPerLane) <= 16) {
                            storeKParallelFp8DirectFinalPacket(static_cast<__nv_fp8_e4m3*>(stage_output),
                                                               outer_index,
                                                               packet_component_begin,
                                                               output_outer_stride,
                                                               local,
                                                               output_finalize,
                                                               output_scale);
                        }
                        break;
                    case DataType::FP8_E5M2:
                        if constexpr (sizeof(__nv_fp8_e5m2) * static_cast<size_t>(ItemsPerLane) <= 16) {
                            storeKParallelFp8DirectFinalPacket(static_cast<__nv_fp8_e5m2*>(stage_output),
                                                               outer_index,
                                                               packet_component_begin,
                                                               output_outer_stride,
                                                               local,
                                                               output_finalize,
                                                               output_scale);
                        }
                        break;
                    case DataType::FP16:
                        if constexpr (sizeof(__half) * static_cast<size_t>(ItemsPerLane) <= 16) {
                            storeKParallelFp8DirectFinalPacket(static_cast<__half*>(stage_output),
                                                               outer_index,
                                                               packet_component_begin,
                                                               output_outer_stride,
                                                               local,
                                                               output_finalize,
                                                               output_scale);
                        }
                        break;
                    case DataType::BF16:
                        if constexpr (sizeof(__nv_bfloat16) * static_cast<size_t>(ItemsPerLane) <= 16) {
                            storeKParallelFp8DirectFinalPacket(static_cast<__nv_bfloat16*>(stage_output),
                                                               outer_index,
                                                               packet_component_begin,
                                                               output_outer_stride,
                                                               local,
                                                               output_finalize,
                                                               output_scale);
                        }
                        break;
                    case DataType::FP32:
                        if constexpr (sizeof(float) * static_cast<size_t>(ItemsPerLane) <= 16) {
                            storeKParallelFp8DirectFinalPacket(static_cast<float*>(stage_output),
                                                               outer_index,
                                                               packet_component_begin,
                                                               output_outer_stride,
                                                               local,
                                                               output_finalize,
                                                               output_scale);
                        }
                        break;
#if THOR_CUB_ENABLE_64BIT_TYPES
                    case DataType::FP64:
                        if constexpr (sizeof(double) * static_cast<size_t>(ItemsPerLane) <= 16) {
                            storeKParallelFp8DirectFinalPacket(static_cast<double*>(stage_output),
                                                               outer_index,
                                                               packet_component_begin,
                                                               output_outer_stride,
                                                               local,
                                                               output_finalize,
                                                               output_scale);
                        }
                        break;
#endif
                    default:
                        break;
                }
            }
            // The output dtype is warp-uniform. When direct storage is legal every active packet owner takes it, while
            // inactive tail lanes have no value to stage. No synchronization is required on this path.
            if ((output_dtype == DataType::FP8_E4M3 || output_dtype == DataType::FP8_E5M2)
                || (ItemsPerLane <= 8 && (output_dtype == DataType::FP16 || output_dtype == DataType::BF16))
                || (ItemsPerLane <= 4 && output_dtype == DataType::FP32)
#if THOR_CUB_ENABLE_64BIT_TYPES
                || (ItemsPerLane <= 2 && output_dtype == DataType::FP64)
#endif
            ) {
                continue;
            }
        }

        if (owns_packet) {
#pragma unroll
            for (int item = 0; item < ItemsPerLane; ++item) {
                const float value = Topology == TiledRKStageTopology::Staged
                                        ? local[item]
                                        : output_finalize(local[item]) * output_scale;
                // Pitch 34 is conflict-free for both producer [item][lane] stores and every 16-byte transposed
                // consumer packet used by the packet8/packet16 staged paths and wider complete-output conversions.
                warp_tile[item * K_PARALLEL_FP8_SHARED_STRIPE_PITCH + lane] = value;
            }
        }
        __syncwarp();

        constexpr uint64_t ComponentsPerWarp =
            static_cast<uint64_t>(K_PARALLEL_FP8_COMPONENTS_PER_WARP<PacketBytes>);
        const uint64_t warp_component_begin =
            tile_component_begin + static_cast<uint64_t>(warp) * ComponentsPerWarp;
        const uint64_t valid_components =
            warp_component_begin < inner_size ? minU64(ComponentsPerWarp, inner_size - warp_component_begin) : 0;

        if (valid_components != 0) {
            if constexpr (Topology == TiledRKStageTopology::Staged) {
                if constexpr (K_PARALLEL_FP8_STAGED_NEEDS_TRANSPOSE<PacketBytes>) {
                    float* fp32_partials = static_cast<float*>(stage_output);
                    const uint64_t output_row_base = (outer_index * shards_per_output + shard) * inner_size;
                    storeKParallelFp8TransposedPackets<float, ItemsPerLane>(
                        fp32_partials, output_row_base, warp_component_begin, valid_components, warp_tile);
                }
            } else {
                static_assert(Topology == TiledRKStageTopology::Complete);
                const uint64_t output_row_base = outer_index * output_outer_stride;
                switch (output_dtype) {
                    case DataType::FP16:
                        if constexpr (sizeof(__half) * static_cast<size_t>(ItemsPerLane) > 16) {
                            storeKParallelFp8TransposedPackets<__half, ItemsPerLane>(
                                static_cast<__half*>(stage_output),
                                output_row_base,
                                warp_component_begin,
                                valid_components,
                                warp_tile);
                        }
                        break;
                    case DataType::BF16:
                        if constexpr (sizeof(__nv_bfloat16) * static_cast<size_t>(ItemsPerLane) > 16) {
                            storeKParallelFp8TransposedPackets<__nv_bfloat16, ItemsPerLane>(
                                static_cast<__nv_bfloat16*>(stage_output),
                                output_row_base,
                                warp_component_begin,
                                valid_components,
                                warp_tile);
                        }
                        break;
                    case DataType::FP32:
                        if constexpr (sizeof(float) * static_cast<size_t>(ItemsPerLane) > 16) {
                            storeKParallelFp8TransposedPackets<float, ItemsPerLane>(
                                static_cast<float*>(stage_output),
                                output_row_base,
                                warp_component_begin,
                                valid_components,
                                warp_tile);
                        }
                        break;
#if THOR_CUB_ENABLE_64BIT_TYPES
                    case DataType::FP64:
                        if constexpr (sizeof(double) * static_cast<size_t>(ItemsPerLane) > 16) {
                            storeKParallelFp8TransposedPackets<double, ItemsPerLane>(
                                static_cast<double*>(stage_output),
                                output_row_base,
                                warp_component_begin,
                                valid_components,
                                warp_tile);
                        }
                        break;
#endif
                    default:
                        break;
                }
            }
        }

        // A grid-stride iteration may reuse this warp's private shared tile. No inter-warp reduction/cooperation is
        // involved, so a warp-local barrier is sufficient and keeps KParallel distinct from RCooperative.
        __syncwarp();
    }
}


// AlignedK Staged: one operation-unified kernel per retained FP8 packet/CTA configuration.
// SUM/PRODUCT/MIN/MAX are selected once per output tile outside the R accumulation loop.
// Other input transforms retain their typed specialization. Identity Complete
// has its own mixed typed/unified operation policy below.
// Covers the six retained P2/P8 x CTA32/64/128 physical configurations.
// It replaces the four compiled Identity-operation variants for these configurations.
// The launch-uniform operation branch is outside the R row loop and is executed once
// per owned (outer, K-tile, shard) work item, rather than once per input packet.
// Input-transformed staged variants continue using typed kernels.
template <typename InputT, typename ReductionOpT, int PacketBytes>
__device__ inline void kParallelFp8HoistedAccumulateRows(const InputT* input,
                                                          uint64_t input_index,
                                                          uint64_t row_begin,
                                                          uint64_t row_end,
                                                          uint64_t inner_size,
                                                          float (&local)[PacketBytes]) {
    constexpr float Init = KParallelReductionIdentityFp32<ReductionOpT>::value;
    const ReductionOpT reduction_op{};
#pragma unroll
    for (int item = 0; item < PacketBytes; ++item) {
        local[item] = Init;
    }
    for (uint64_t row = row_begin; row < row_end; ++row) {
        const PackedInputValues<InputT, PacketBytes> values =
            loadVectorizedInputPacket<InputT, PacketBytes>(input + input_index);
#pragma unroll
        for (int item = 0; item < PacketBytes; ++item) {
            local[item] = reduction_op(local[item], ToFp32<InputT>{}(values.values[item]));
        }
        input_index += inner_size;
    }
}

template <typename InputT, int PacketBytes, int BlockThreads>
__global__ __launch_bounds__(BlockThreads) void kParallelFp8StagedHoistedCombineKernel(
    const InputT* input, float* fp32_partials, uint64_t outer_size,
    uint64_t reduction_size, uint64_t inner_size, uint64_t shards_per_output,
    CubReductionStageCombine operation) {
    static_assert(IS_K_PARALLEL_FP8_TYPE<InputT>);
    static_assert(PacketBytes == 2 || PacketBytes == 8);
    static_assert(BlockThreads == 32 || BlockThreads == 64 || BlockThreads == 128);
    constexpr uint64_t ComponentsPerCTA = static_cast<uint64_t>(PacketBytes) * BlockThreads;
    extern __shared__ float fp8_writeback_shared[];
    const int warp = static_cast<int>(threadIdx.x) / TILED_REDUCTION_WARP_THREADS;
    const int lane = static_cast<int>(threadIdx.x) % TILED_REDUCTION_WARP_THREADS;
    float* warp_tile = fp8_writeback_shared + static_cast<size_t>(warp) * PacketBytes
                                            * K_PARALLEL_FP8_SHARED_STRIPE_PITCH;
    const uint64_t component_tiles = ceilDivideU64(inner_size, ComponentsPerCTA);
    const uint64_t total_work = outer_size * component_tiles * shards_per_output;

    for (uint64_t work_index = blockIdx.x; work_index < total_work; work_index += gridDim.x) {
        const uint64_t shard = work_index % shards_per_output;
        const uint64_t outer_component_work = work_index / shards_per_output;
        const uint64_t component_tile = outer_component_work % component_tiles;
        const uint64_t outer_index = outer_component_work / component_tiles;
        const uint64_t tile_component_begin = component_tile * ComponentsPerCTA;
        const uint64_t packet_component_begin =
            tile_component_begin + static_cast<uint64_t>(threadIdx.x) * PacketBytes;
        const bool owns_packet = packet_component_begin < inner_size;
        const uint64_t base_rows = reduction_size / shards_per_output;
        const uint64_t extra_rows = reduction_size % shards_per_output;
        const uint64_t row_begin = shard * base_rows + minU64(shard, extra_rows);
        const uint64_t row_end = row_begin + base_rows + static_cast<uint64_t>(shard < extra_rows);
        float local[PacketBytes]{};
        if (owns_packet) {
            const uint64_t input_index =
                (outer_index * reduction_size + row_begin) * inner_size + packet_component_begin;
            // One uniform switch PER shard/tile, not per R row or decoded packet.
            switch (operation) {
                case CubReductionStageCombine::Sum:
                    kParallelFp8HoistedAccumulateRows<InputT, cuda::std::plus<float>, PacketBytes>(
                        input, input_index, row_begin, row_end, inner_size, local);
                    break;
                case CubReductionStageCombine::Product:
                    kParallelFp8HoistedAccumulateRows<InputT, cuda::std::multiplies<float>, PacketBytes>(
                        input, input_index, row_begin, row_end, inner_size, local);
                    break;
                case CubReductionStageCombine::Minimum:
                    kParallelFp8HoistedAccumulateRows<InputT, PropagatingMinimumFp32, PacketBytes>(
                        input, input_index, row_begin, row_end, inner_size, local);
                    break;
                case CubReductionStageCombine::Maximum:
                    kParallelFp8HoistedAccumulateRows<InputT, PropagatingMaximumFp32, PacketBytes>(
                        input, input_index, row_begin, row_end, inner_size, local);
                    break;
            }
        }
        const uint64_t output_row_base = (outer_index * shards_per_output + shard) * inner_size;
        if constexpr (PacketBytes == 2) {
            if (owns_packet) {
                storeKParallelFp32Packet<PacketBytes>(
                    fp32_partials + output_row_base + packet_component_begin, local, PacketBytes);
            }
        } else {
            if (owns_packet) {
#pragma unroll
                for (int item = 0; item < PacketBytes; ++item) {
                    warp_tile[item * K_PARALLEL_FP8_SHARED_STRIPE_PITCH + lane] = local[item];
                }
            }
            __syncwarp();
            constexpr uint64_t ComponentsPerWarp =
                static_cast<uint64_t>(K_PARALLEL_FP8_COMPONENTS_PER_WARP<PacketBytes>);
            const uint64_t warp_component_begin =
                tile_component_begin + static_cast<uint64_t>(warp) * ComponentsPerWarp;
            const uint64_t valid_components = warp_component_begin < inner_size
                ? minU64(ComponentsPerWarp, inner_size - warp_component_begin) : 0;
            if (valid_components != 0) {
                storeKParallelFp8TransposedPackets<float, PacketBytes>(
                    fp32_partials, output_row_base, warp_component_begin, valid_components, warp_tile);
            }
            __syncwarp();
        }
    }
}

// Retained FP8 Complete operation-unified physical kernels. The two finalization
// placements are intentional: P2/CTA32 uses a post-switch join, while P8/CTA32
// and P8/CTA64 retain the in-branch schedule. In particular the P8/CTA64
// nonadditive dispatch MUST NOT strip the additive code from the physical kernel:
// that apparently harmless simplification regressed MAX at high bandwidth.
enum class Fp8CompleteFinalizePolicy { InBranch, Joined };

template <typename InputT, size_t PacketBytes, int BlockThreads, Fp8CompleteFinalizePolicy Policy>
__global__ __launch_bounds__(BlockThreads) void kParallelFp8CompleteUnifiedKernel(const InputT* input,
                                                                        void* stage_output,
                                                                        DataType output_dtype,
                                                                        uint64_t outer_size,
                                                                        uint64_t reduction_size,
                                                                        uint64_t inner_size,
                                                                        uint64_t output_outer_stride,
                                                                        uint64_t output_inner_stride,
                                                                        uint64_t shards_per_output,
                                                                        CubReductionStageCombine operation,
                                                                        AdditiveFinalizeFp32 output_finalize,
                                                                        float output_scale) {
    static_assert(IS_K_PARALLEL_FP8_TYPE<InputT>);
    static_assert(PacketBytes == 2 || PacketBytes == 8);
    static_assert((PacketBytes == 2 && BlockThreads == 32) ||
                  (PacketBytes == 8 && (BlockThreads == 32 || BlockThreads == 64)));
    constexpr int ItemsPerLane = K_PARALLEL_FP8_ITEMS_PER_LANE<PacketBytes>;
    static_assert(sizeof(InputT) * ItemsPerLane == PacketBytes);
    static_assert(BlockThreads == 32 || BlockThreads == 64 || BlockThreads == 128 || BlockThreads == 256);
    static_assert(BlockThreads % TILED_REDUCTION_WARP_THREADS == 0);
    static_cast<void>(output_inner_stride);

    constexpr uint64_t components_per_cta =
        static_cast<uint64_t>(BlockThreads) * static_cast<uint64_t>(ItemsPerLane);
    constexpr TiledRKStageTopology Topology = TiledRKStageTopology::Complete;
    // Finalization is applied to SUM/MEAN once per decoded packet. Storage
    // always receives already-finalized values (and applies runtime output scale).
    const IdentityFp32 store_finalize{};

    extern __shared__ float fp8_writeback_shared[];
    const int warp = static_cast<int>(threadIdx.x) / TILED_REDUCTION_WARP_THREADS;
    const int lane = static_cast<int>(threadIdx.x) % TILED_REDUCTION_WARP_THREADS;
    float* warp_tile = fp8_writeback_shared
                       + static_cast<size_t>(warp) * static_cast<size_t>(ItemsPerLane)
                             * K_PARALLEL_FP8_SHARED_STRIPE_PITCH;

    const uint64_t component_tiles = ceilDivideU64(inner_size, components_per_cta);
    const uint64_t total_work = outer_size * component_tiles * shards_per_output;
    const uint64_t grid_stride = static_cast<uint64_t>(gridDim.x);

    for (uint64_t work_index = static_cast<uint64_t>(blockIdx.x); work_index < total_work; work_index += grid_stride) {
        const uint64_t shard = work_index % shards_per_output;
        const uint64_t outer_component_work = work_index / shards_per_output;
        const uint64_t component_tile = outer_component_work % component_tiles;
        const uint64_t outer_index = outer_component_work / component_tiles;
        const uint64_t tile_component_begin = component_tile * components_per_cta;
        const uint64_t packet_component_begin =
            tile_component_begin + static_cast<uint64_t>(threadIdx.x) * static_cast<uint64_t>(ItemsPerLane);
        const bool owns_packet = packet_component_begin < inner_size;

        const uint64_t base_rows = reduction_size / shards_per_output;
        const uint64_t extra_rows = reduction_size % shards_per_output;
        const uint64_t row_begin = shard * base_rows + (shard < extra_rows ? shard : extra_rows);
        const uint64_t row_count = base_rows + static_cast<uint64_t>(shard < extra_rows);
        const uint64_t row_end = row_begin + row_count;

        float local[ItemsPerLane]{};

        if (owns_packet) {
            uint64_t input_index =
                (outer_index * reduction_size + row_begin) * inner_size + packet_component_begin;

            // Uniform operation selection outside the R loop. Both policies
            // retain all four operation bodies, including the additive branch
            // in the P8/CTA64 nonadditive physical instance.
            switch (operation) {
                case CubReductionStageCombine::Sum:
                    kParallelFp8HoistedAccumulateRows<InputT, cuda::std::plus<float>, ItemsPerLane>(
                        input, input_index, row_begin, row_end, inner_size, local);
                    if constexpr (Policy == Fp8CompleteFinalizePolicy::InBranch) {
#pragma unroll
                        for (int item = 0; item < ItemsPerLane; ++item)
                            local[item] = output_finalize(local[item]);
                    }
                    break;
                case CubReductionStageCombine::Product:
                    kParallelFp8HoistedAccumulateRows<InputT, cuda::std::multiplies<float>, ItemsPerLane>(
                        input, input_index, row_begin, row_end, inner_size, local);
                    break;
                case CubReductionStageCombine::Minimum:
                    kParallelFp8HoistedAccumulateRows<InputT, PropagatingMinimumFp32, ItemsPerLane>(
                        input, input_index, row_begin, row_end, inner_size, local);
                    break;
                case CubReductionStageCombine::Maximum:
                    kParallelFp8HoistedAccumulateRows<InputT, PropagatingMaximumFp32, ItemsPerLane>(
                        input, input_index, row_begin, row_end, inner_size, local);
                    break;
            }
            if constexpr (Policy == Fp8CompleteFinalizePolicy::Joined) {
                if (operation == CubReductionStageCombine::Sum) {
#pragma unroll
                    for (int item = 0; item < ItemsPerLane; ++item)
                        local[item] = output_finalize(local[item]);
                }
            }
        }

        if constexpr (Topology == TiledRKStageTopology::Staged && !K_PARALLEL_FP8_STAGED_NEEDS_TRANSPOSE<PacketBytes>) {
            if (owns_packet) {
                float* fp32_partials = static_cast<float*>(stage_output);
                float* destination = fp32_partials
                                     + (outer_index * shards_per_output + shard) * inner_size
                                     + packet_component_begin;
                // packet4 => one coalesced 16-byte FP32 store/lane; packet2 => one coalesced 8-byte store/lane.
                storeKParallelFp32Packet<ItemsPerLane>(destination, local, ItemsPerLane);
            }
            continue;
        }

        if constexpr (Topology == TiledRKStageTopology::Complete) {
            if (owns_packet) {
                switch (output_dtype) {
                    case DataType::FP8_E4M3:
                        if constexpr (sizeof(__nv_fp8_e4m3) * static_cast<size_t>(ItemsPerLane) <= 16) {
                            storeKParallelFp8DirectFinalPacket(static_cast<__nv_fp8_e4m3*>(stage_output),
                                                               outer_index,
                                                               packet_component_begin,
                                                               output_outer_stride,
                                                               local,
                                                               store_finalize,
                                                               output_scale);
                        }
                        break;
                    case DataType::FP8_E5M2:
                        if constexpr (sizeof(__nv_fp8_e5m2) * static_cast<size_t>(ItemsPerLane) <= 16) {
                            storeKParallelFp8DirectFinalPacket(static_cast<__nv_fp8_e5m2*>(stage_output),
                                                               outer_index,
                                                               packet_component_begin,
                                                               output_outer_stride,
                                                               local,
                                                               store_finalize,
                                                               output_scale);
                        }
                        break;
                    case DataType::FP16:
                        if constexpr (sizeof(__half) * static_cast<size_t>(ItemsPerLane) <= 16) {
                            storeKParallelFp8DirectFinalPacket(static_cast<__half*>(stage_output),
                                                               outer_index,
                                                               packet_component_begin,
                                                               output_outer_stride,
                                                               local,
                                                               store_finalize,
                                                               output_scale);
                        }
                        break;
                    case DataType::BF16:
                        if constexpr (sizeof(__nv_bfloat16) * static_cast<size_t>(ItemsPerLane) <= 16) {
                            storeKParallelFp8DirectFinalPacket(static_cast<__nv_bfloat16*>(stage_output),
                                                               outer_index,
                                                               packet_component_begin,
                                                               output_outer_stride,
                                                               local,
                                                               store_finalize,
                                                               output_scale);
                        }
                        break;
                    case DataType::FP32:
                        if constexpr (sizeof(float) * static_cast<size_t>(ItemsPerLane) <= 16) {
                            storeKParallelFp8DirectFinalPacket(static_cast<float*>(stage_output),
                                                               outer_index,
                                                               packet_component_begin,
                                                               output_outer_stride,
                                                               local,
                                                               store_finalize,
                                                               output_scale);
                        }
                        break;
#if THOR_CUB_ENABLE_64BIT_TYPES
                    case DataType::FP64:
                        if constexpr (sizeof(double) * static_cast<size_t>(ItemsPerLane) <= 16) {
                            storeKParallelFp8DirectFinalPacket(static_cast<double*>(stage_output),
                                                               outer_index,
                                                               packet_component_begin,
                                                               output_outer_stride,
                                                               local,
                                                               store_finalize,
                                                               output_scale);
                        }
                        break;
#endif
                    default:
                        break;
                }
            }
            // The output dtype is warp-uniform. When direct storage is legal every active packet owner takes it, while
            // inactive tail lanes have no value to stage. No synchronization is required on this path.
            if ((output_dtype == DataType::FP8_E4M3 || output_dtype == DataType::FP8_E5M2)
                || (ItemsPerLane <= 8 && (output_dtype == DataType::FP16 || output_dtype == DataType::BF16))
                || (ItemsPerLane <= 4 && output_dtype == DataType::FP32)
#if THOR_CUB_ENABLE_64BIT_TYPES
                || (ItemsPerLane <= 2 && output_dtype == DataType::FP64)
#endif
            ) {
                continue;
            }
        }

        if (owns_packet) {
#pragma unroll
            for (int item = 0; item < ItemsPerLane; ++item) {
                const float value = Topology == TiledRKStageTopology::Staged
                                        ? local[item]
                                        : store_finalize(local[item]) * output_scale;
                // Pitch 34 is conflict-free for both producer [item][lane] stores and every 16-byte transposed
                // consumer packet used by the packet8/packet16 staged paths and wider complete-output conversions.
                warp_tile[item * K_PARALLEL_FP8_SHARED_STRIPE_PITCH + lane] = value;
            }
        }
        __syncwarp();

        constexpr uint64_t ComponentsPerWarp =
            static_cast<uint64_t>(K_PARALLEL_FP8_COMPONENTS_PER_WARP<PacketBytes>);
        const uint64_t warp_component_begin =
            tile_component_begin + static_cast<uint64_t>(warp) * ComponentsPerWarp;
        const uint64_t valid_components =
            warp_component_begin < inner_size ? minU64(ComponentsPerWarp, inner_size - warp_component_begin) : 0;

        if (valid_components != 0) {
            if constexpr (Topology == TiledRKStageTopology::Staged) {
                if constexpr (K_PARALLEL_FP8_STAGED_NEEDS_TRANSPOSE<PacketBytes>) {
                    float* fp32_partials = static_cast<float*>(stage_output);
                    const uint64_t output_row_base = (outer_index * shards_per_output + shard) * inner_size;
                    storeKParallelFp8TransposedPackets<float, ItemsPerLane>(
                        fp32_partials, output_row_base, warp_component_begin, valid_components, warp_tile);
                }
            } else {
                static_assert(Topology == TiledRKStageTopology::Complete);
                const uint64_t output_row_base = outer_index * output_outer_stride;
                switch (output_dtype) {
                    case DataType::FP16:
                        if constexpr (sizeof(__half) * static_cast<size_t>(ItemsPerLane) > 16) {
                            storeKParallelFp8TransposedPackets<__half, ItemsPerLane>(
                                static_cast<__half*>(stage_output),
                                output_row_base,
                                warp_component_begin,
                                valid_components,
                                warp_tile);
                        }
                        break;
                    case DataType::BF16:
                        if constexpr (sizeof(__nv_bfloat16) * static_cast<size_t>(ItemsPerLane) > 16) {
                            storeKParallelFp8TransposedPackets<__nv_bfloat16, ItemsPerLane>(
                                static_cast<__nv_bfloat16*>(stage_output),
                                output_row_base,
                                warp_component_begin,
                                valid_components,
                                warp_tile);
                        }
                        break;
                    case DataType::FP32:
                        if constexpr (sizeof(float) * static_cast<size_t>(ItemsPerLane) > 16) {
                            storeKParallelFp8TransposedPackets<float, ItemsPerLane>(
                                static_cast<float*>(stage_output),
                                output_row_base,
                                warp_component_begin,
                                valid_components,
                                warp_tile);
                        }
                        break;
#if THOR_CUB_ENABLE_64BIT_TYPES
                    case DataType::FP64:
                        if constexpr (sizeof(double) * static_cast<size_t>(ItemsPerLane) > 16) {
                            storeKParallelFp8TransposedPackets<double, ItemsPerLane>(
                                static_cast<double*>(stage_output),
                                output_row_base,
                                warp_component_begin,
                                valid_components,
                                warp_tile);
                        }
                        break;
#endif
                    default:
                        break;
                }
            }
        }

        // A grid-stride iteration may reuse this warp's private shared tile. No inter-warp reduction/cooperation is
        // involved, so a warp-local barrier is sufficient and keeps KParallel distinct from RCooperative.
        __syncwarp();
    }
}


#endif  // THOR_CUB_ENABLE_FP8_TYPES

template <typename InputT,
          typename InputTransformT,
          typename ReductionOpT,
          typename OutputFinalizeT,
          int ItemsPerLane,
          int BlockThreads>
__global__ __launch_bounds__(BlockThreads) void kParallelLeanCompleteKernel(const InputT* input,
                                                                           void* output,
                                                                           DataType output_dtype,
                                                                           uint64_t outer_size,
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

    // Grid X is deliberately capped at 65535 CTAs. A Complete reduction must
    // therefore stride over *all* outer/component work, not just process one
    // tile per CTA. In particular, FP32 continuations of FP8 staged reductions
    // commonly have more than 65535 outer rows with narrow K.
    const uint64_t total_work = outer_size * component_tiles;
    const uint64_t grid_stride = static_cast<uint64_t>(gridDim.x);
    for (uint64_t outer_component = static_cast<uint64_t>(blockIdx.x);
         outer_component < total_work;
         outer_component += grid_stride) {
        const uint64_t outer_index = outer_component / component_tiles;
        const uint64_t component_tile = outer_component - outer_index * component_tiles;
        const uint64_t packet_component_begin =
            component_tile * components_per_cta
            + static_cast<uint64_t>(threadIdx.x) * static_cast<uint64_t>(ItemsPerLane);
        if (packet_component_begin >= inner_size) {
            continue;
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

#if THOR_CUB_ENABLE_FP8_TYPES

template <typename InputT, size_t PacketBytes, int BlockThreads, TiledRKStageTopology Topology>
[[nodiscard]] CubKernelOccupancyInfo queryKParallelFp8OccupancyForBlock() {
    static_assert(IS_K_PARALLEL_FP8_TYPE<InputT>);
    static_assert(PacketBytes == 2 || PacketBytes == 4 || PacketBytes == 8 || PacketBytes == 16);
    static_assert(BlockThreads == 32 || BlockThreads == 64 || BlockThreads == 128 || BlockThreads == 256);
    using InputTransformT = IdentityFp32;
    using ReductionOpT = cuda::std::plus<float>;
    using OutputFinalizeT = AdditiveFinalizeFp32;

    // Staged output is always FP32, so packet4/packet2 have an exact zero-shared occupancy query. Complete occupancy is
    // conservatively queried against FP32 writeback; packet8/packet16 therefore include the transpose footprint while
    // packet4/packet2 remain direct. Runtime Complete launches may use still less shared memory for narrower outputs.
    constexpr bool NeedsTranspose =
        Topology == TiledRKStageTopology::Staged ? K_PARALLEL_FP8_STAGED_NEEDS_TRANSPOSE<PacketBytes>
                                                 : (PacketBytes * sizeof(float) > 16);
    constexpr size_t dynamic_shared_bytes =
        NeedsTranspose
            ? static_cast<size_t>(BlockThreads / TILED_REDUCTION_WARP_THREADS)
                  * K_PARALLEL_FP8_SHARED_BYTES_PER_WARP<PacketBytes>
            : 0;
    const auto kernel = [] {
        if constexpr (Topology == TiledRKStageTopology::Staged
                      && (PacketBytes == 2 || PacketBytes == 8)
                      && (BlockThreads == 32 || BlockThreads == 64 || BlockThreads == 128)) {
            return kParallelFp8StagedHoistedCombineKernel<InputT, PacketBytes, BlockThreads>;
        } else if constexpr (Topology == TiledRKStageTopology::Complete
                             && PacketBytes == 2 && BlockThreads == 32) {
            return kParallelFp8CompleteUnifiedKernel<InputT, 2, 32, Fp8CompleteFinalizePolicy::Joined>;
        } else if constexpr (Topology == TiledRKStageTopology::Complete
                             && PacketBytes == 8 && BlockThreads == 32) {
            return kParallelFp8CompleteUnifiedKernel<InputT, 8, 32, Fp8CompleteFinalizePolicy::InBranch>;
        } else {
            return kParallelFp8PacketKernel<InputT,
                                            InputTransformT,
                                            ReductionOpT,
                                            OutputFinalizeT,
                                            PacketBytes,
                                            BlockThreads,
                                            Topology>;
        }
    }();
    cudaFuncAttributes attributes{};
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
    int active_blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks, kernel, BlockThreads, dynamic_shared_bytes));
    return CubKernelOccupancyInfo{attributes.numRegs,
                                  attributes.sharedSizeBytes,
                                  dynamic_shared_bytes,
                                  active_blocks,
                                  active_blocks * (BlockThreads / TILED_REDUCTION_WARP_THREADS)};
}

template <typename InputT, TiledRKStageTopology Topology>
[[nodiscard]] CubKernelOccupancyInfo queryKParallelFp8Occupancy(size_t packet_bytes, uint32_t block_threads) {
    static_assert(IS_K_PARALLEL_FP8_TYPE<InputT>);
    CubKernelOccupancyInfo occupancy{};
    const bool queried = dispatchFp8AlignedKConfiguration<Topology>(
        packet_bytes, block_threads, [&]<size_t PacketBytes, int BlockThreads>() {
            occupancy = queryKParallelFp8OccupancyForBlock<InputT, PacketBytes, BlockThreads, Topology>();
        });
    if (!queried) {
        // Not a retained compiled AlignedK configuration; the selector treats this as absent.
        throw std::logic_error("FP8 AlignedK occupancy requires a retained packet/CTA configuration.");
    }
    return occupancy;
}

#endif  // THOR_CUB_ENABLE_FP8_TYPES

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
                                                                   geometry.outer_size,
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

#if THOR_CUB_ENABLE_FP8_TYPES

template <typename InputT,
          typename InputTransformT,
          typename ReductionOpT,
          typename OutputFinalizeT,
          size_t PacketBytes,
          int BlockThreads,
          TiledRKStageTopology Topology>
void launchKParallelFp8ForBlock(const Tensor& input,
                                Tensor& stage_output,
                                const CubReductionGeometry& geometry,
                                uint64_t shards_per_output,
                                InputTransformT input_transform,
                                OutputFinalizeT output_finalize,
                                float output_scale,
                                Stream& stream) {
    static_assert(IS_K_PARALLEL_FP8_TYPE<InputT>);
    static_assert(PacketBytes == 2 || PacketBytes == 4 || PacketBytes == 8 || PacketBytes == 16);
    static_assert(BlockThreads == 32 || BlockThreads == 64 || BlockThreads == 128 || BlockThreads == 256);
    constexpr int ItemsPerLane = K_PARALLEL_FP8_ITEMS_PER_LANE<PacketBytes>;
    constexpr uint64_t components_per_cta =
        static_cast<uint64_t>(BlockThreads) * static_cast<uint64_t>(ItemsPerLane);
    constexpr size_t max_dynamic_shared_bytes =
        static_cast<size_t>(BlockThreads / TILED_REDUCTION_WARP_THREADS)
        * K_PARALLEL_FP8_SHARED_BYTES_PER_WARP<PacketBytes>;

    const bool needs_transpose = Topology == TiledRKStageTopology::Staged
                                     ? K_PARALLEL_FP8_STAGED_NEEDS_TRANSPOSE<PacketBytes>
                                     : kParallelFp8CompleteNeedsTranspose<PacketBytes>(stage_output.getDataType());
    const size_t dynamic_shared_bytes = needs_transpose ? max_dynamic_shared_bytes : 0;

    const uint64_t component_tiles = ceilDivideU64(geometry.inner_size, components_per_cta);
    if (geometry.outer_size > std::numeric_limits<uint64_t>::max() / component_tiles
        || geometry.outer_size * component_tiles > std::numeric_limits<uint64_t>::max() / shards_per_output) {
        throw std::overflow_error("FP8 K-parallel tiled stage work count overflows uint64_t.");
    }
    const uint64_t total_work = geometry.outer_size * component_tiles * shards_per_output;
    const unsigned int grid_blocks = static_cast<unsigned int>(
        std::min<uint64_t>(total_work, TILED_REDUCTION_MAX_GRID_BLOCKS));
    if (grid_blocks == 0) {
        throw std::logic_error("FP8 K-parallel tiled stage requires non-zero work.");
    }

    // Production Staged AlignedK: remove the operation dimension from the six retained
    // P2/P8 x CTA32/64/128 kernels. Compile-time routing is important: a runtime
    // pre-dispatch alone still instantiates all four legacy operation kernels.
    if constexpr (Topology == TiledRKStageTopology::Staged
                  && std::is_same_v<InputTransformT, IdentityFp32>
                  && (PacketBytes == 2 || PacketBytes == 8)
                  && (BlockThreads == 32 || BlockThreads == 64 || BlockThreads == 128)
                  && (std::is_same_v<ReductionOpT, cuda::std::plus<float>>
                      || std::is_same_v<ReductionOpT, cuda::std::multiplies<float>>
                      || std::is_same_v<ReductionOpT, PropagatingMinimumFp32>
                      || std::is_same_v<ReductionOpT, PropagatingMaximumFp32>)) {
        constexpr CubReductionStageCombine operation = [] {
            if constexpr (std::is_same_v<ReductionOpT, cuda::std::plus<float>>) {
                return CubReductionStageCombine::Sum;
            } else if constexpr (std::is_same_v<ReductionOpT, cuda::std::multiplies<float>>) {
                return CubReductionStageCombine::Product;
            } else if constexpr (std::is_same_v<ReductionOpT, PropagatingMinimumFp32>) {
                return CubReductionStageCombine::Minimum;
            } else {
                return CubReductionStageCombine::Maximum;
            }
        }();
        kParallelFp8StagedHoistedCombineKernel<InputT, PacketBytes, BlockThreads>
            <<<grid_blocks, BlockThreads, dynamic_shared_bytes, stream.getStream()>>>(
                input.getMemPtr<InputT>(), stage_output.getMemPtr<float>(), geometry.outer_size,
                geometry.reduction_size, geometry.inner_size, shards_per_output, operation);
        CUDA_CHECK(cudaGetLastError());
    } else if constexpr (Topology == TiledRKStageTopology::Complete
                         && std::is_same_v<InputTransformT, IdentityFp32>
                         && ((PacketBytes == 2 && BlockThreads == 32)
                             || (PacketBytes == 8 && (BlockThreads == 32 || BlockThreads == 64)))
                         && (std::is_same_v<ReductionOpT, cuda::std::multiplies<float>>
                             || std::is_same_v<ReductionOpT, PropagatingMinimumFp32>
                             || std::is_same_v<ReductionOpT, PropagatingMaximumFp32>
                             || (std::is_same_v<ReductionOpT, cuda::std::plus<float>>
                                 && !(PacketBytes == 8 && BlockThreads == 64)))) {
        // Compile-time routing prevents instantiation of the retired typed
        // Identity Complete operations at these physical configurations.
        // P8/CTA64 SUM/MEAN deliberately remain on the original typed kernel.
        constexpr CubReductionStageCombine operation = [] {
            if constexpr (std::is_same_v<ReductionOpT, cuda::std::plus<float>>)
                return CubReductionStageCombine::Sum;
            else if constexpr (std::is_same_v<ReductionOpT, cuda::std::multiplies<float>>)
                return CubReductionStageCombine::Product;
            else if constexpr (std::is_same_v<ReductionOpT, PropagatingMinimumFp32>)
                return CubReductionStageCombine::Minimum;
            else
                return CubReductionStageCombine::Maximum;
        }();
        constexpr Fp8CompleteFinalizePolicy policy = PacketBytes == 2
            ? Fp8CompleteFinalizePolicy::Joined : Fp8CompleteFinalizePolicy::InBranch;
        const AdditiveFinalizeFp32 additive_finalize = [&] {
            if constexpr (std::is_same_v<ReductionOpT, cuda::std::plus<float>>)
                return output_finalize;
            else
                return AdditiveFinalizeFp32{1.0f, false};
        }();
        kParallelFp8CompleteUnifiedKernel<InputT, PacketBytes, BlockThreads, policy>
            <<<grid_blocks, BlockThreads, dynamic_shared_bytes, stream.getStream()>>>(
                input.getMemPtr<InputT>(), stage_output.getMemPtr<void>(), stage_output.getDataType(),
                geometry.outer_size, geometry.reduction_size, geometry.inner_size,
                geometry.tiled_output_outer_stride, geometry.tiled_output_inner_stride,
                shards_per_output, operation, additive_finalize, output_scale);
        CUDA_CHECK(cudaGetLastError());
    } else {
        if constexpr (prunedFp8AlignedInstantiation<InputT, InputTransformT, ReductionOpT,
                                                    PacketBytes, BlockThreads, Topology>()) {
            throw std::logic_error("Pruned FP8 AlignedK specialization requested by launcher.");
        } else {
            kParallelFp8PacketKernel<InputT,
                                     InputTransformT,
                                     ReductionOpT,
                                     OutputFinalizeT,
                                     PacketBytes,
                                     BlockThreads,
                                     Topology>
                <<<grid_blocks, BlockThreads, dynamic_shared_bytes, stream.getStream()>>>(input.getMemPtr<InputT>(),
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
            CUDA_CHECK(cudaGetLastError());
        }
    }
}


#endif  // THOR_CUB_ENABLE_FP8_TYPES

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
    // FP8 never reaches this generic grid: dispatchFp8AlignedKStageSemantics owns its retained configurations.
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
#if THOR_CUB_ENABLE_FP8_TYPES
        case DataType::FP8_E4M3:
        case DataType::FP8_E5M2:
            return 1;
#endif
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

#if THOR_CUB_ENABLE_FP8_TYPES

// Retained FP8 AlignedK stage dispatch. Every FP8 (transform, combine, finalizer)
// composition is routed only through the retained packet/CTA configurations; the
// generic dispatchStage() grid is never instantiated for FP8 input. Within a retained
// configuration, launchKParallelFp8ForBlock selects the hoisted staged, unified
// complete or typed packet kernel, and honors the accepted typed pruning.
template <TiledRKStageTopology Topology, typename TransformT, typename ReductionOpT, typename OutputFinalizeT>
void launchFp8AlignedKRetained(const Tensor& input,
                               Tensor& stage_output,
                               const CubReductionGeometry& geometry,
                               const KParallelTiledStagePlan& plan,
                               OutputFinalizeT output_finalize,
                               float output_scale,
                               Stream& stream) {
    const auto for_type = [&]<typename InputT>() {
        return dispatchFp8AlignedKConfiguration<Topology>(
            plan.packet_bytes, plan.block_threads, [&]<size_t PacketBytes, int BlockThreads>() {
                launchKParallelFp8ForBlock<InputT, TransformT, ReductionOpT, OutputFinalizeT,
                                           PacketBytes, BlockThreads, Topology>(
                    input, stage_output, geometry, plan.shards_per_output, TransformT{}, output_finalize,
                    output_scale, stream);
            });
    };
    bool launched = false;
    switch (input.getDataType()) {
        case DataType::FP8_E4M3: launched = for_type.template operator()<__nv_fp8_e4m3>(); break;
        case DataType::FP8_E5M2: launched = for_type.template operator()<__nv_fp8_e5m2>(); break;
        default: throw std::logic_error("Retained AlignedK stage requires FP8 input.");
    }
    if (!launched) {
        throw std::logic_error("FP8 AlignedK stage requires a retained packet/CTA configuration.");
    }
}

template <TiledRKStageTopology Topology, typename ReductionOpT>
void launchFp8AlignedKNonadditive(const CubReductionStageSemantics& semantics,
                                  const Tensor& input,
                                  Tensor& stage_output,
                                  const CubReductionGeometry& geometry,
                                  const KParallelTiledStagePlan& plan,
                                  float output_scale,
                                  Stream& stream) {
    if (semantics.finalize != CubReductionStageFinalize::Identity)
        throw std::logic_error("Nonadditive AlignedK combine requires identity finalization.");
    switch (semantics.input_transform) {
        case CubReductionStageInputTransform::Identity:
            return launchFp8AlignedKRetained<Topology, IdentityFp32, ReductionOpT>(
                input, stage_output, geometry, plan, IdentityFp32{}, output_scale, stream);
        case CubReductionStageInputTransform::AbsoluteValue:
            return launchFp8AlignedKRetained<Topology, AbsoluteValueFp32, ReductionOpT>(
                input, stage_output, geometry, plan, IdentityFp32{}, output_scale, stream);
        case CubReductionStageInputTransform::Square:
            return launchFp8AlignedKRetained<Topology, SquareFp32, ReductionOpT>(
                input, stage_output, geometry, plan, IdentityFp32{}, output_scale, stream);
    }
    throw std::logic_error("Unsupported FP8 AlignedK input transform.");
}

template <TiledRKStageTopology Topology>
void dispatchFp8AlignedKStageSemantics(const CubReductionStageSemantics& semantics,
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
                    return launchFp8AlignedKRetained<Topology, IdentityFp32, cuda::std::plus<float>>(
                        input, stage_output, geometry, plan, output_finalize, output_scale, stream);
                case CubReductionStageInputTransform::AbsoluteValue:
                    return launchFp8AlignedKRetained<Topology, AbsoluteValueFp32, cuda::std::plus<float>>(
                        input, stage_output, geometry, plan, output_finalize, output_scale, stream);
                case CubReductionStageInputTransform::Square:
                    return launchFp8AlignedKRetained<Topology, SquareFp32, cuda::std::plus<float>>(
                        input, stage_output, geometry, plan, output_finalize, output_scale, stream);
            }
            break;
        }
        case CubReductionStageCombine::Product:
            return launchFp8AlignedKNonadditive<Topology, cuda::std::multiplies<float>>(
                semantics, input, stage_output, geometry, plan, output_scale, stream);
        case CubReductionStageCombine::Minimum:
            return launchFp8AlignedKNonadditive<Topology, PropagatingMinimumFp32>(
                semantics, input, stage_output, geometry, plan, output_scale, stream);
        case CubReductionStageCombine::Maximum:
            return launchFp8AlignedKNonadditive<Topology, PropagatingMaximumFp32>(
                semantics, input, stage_output, geometry, plan, output_scale, stream);
    }
    throw std::logic_error("Unsupported FP8 AlignedK stage semantics.");
}

#endif  // THOR_CUB_ENABLE_FP8_TYPES

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
    if (plan.packet_bytes != 2 && plan.packet_bytes != 4 && plan.packet_bytes != 8 && plan.packet_bytes != 16) {
        throw std::logic_error("K-parallel occupancy query requires a 2-, 4-, 8-, or 16-byte packet.");
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
#if THOR_CUB_ENABLE_FP8_TYPES
        case DataType::FP8_E4M3:
            if (plan.topology == TiledRKStageTopology::Complete) {
                return queryKParallelFp8Occupancy<__nv_fp8_e4m3, TiledRKStageTopology::Complete>(
                    plan.packet_bytes, plan.block_threads);
            }
            return queryKParallelFp8Occupancy<__nv_fp8_e4m3, TiledRKStageTopology::Staged>(
                plan.packet_bytes, plan.block_threads);
        case DataType::FP8_E5M2:
            if (plan.topology == TiledRKStageTopology::Complete) {
                return queryKParallelFp8Occupancy<__nv_fp8_e5m2, TiledRKStageTopology::Complete>(
                    plan.packet_bytes, plan.block_threads);
            }
            return queryKParallelFp8Occupancy<__nv_fp8_e5m2, TiledRKStageTopology::Staged>(
                plan.packet_bytes, plan.block_threads);
#endif
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
    if (plan.packet_bytes != 2 && plan.packet_bytes != 4 && plan.packet_bytes != 8 && plan.packet_bytes != 16) {
        throw std::logic_error("K-parallel tiled stage requires a 2-, 4-, 8-, or 16-byte input packet.");
    }
    const size_t input_element_bytes = elementBytes(input.getDataType());
#if THOR_CUB_ENABLE_FP8_TYPES
    const bool fp8_input =
        input.getDataType() == DataType::FP8_E4M3 || input.getDataType() == DataType::FP8_E5M2;
    if (!fp8_input && plan.packet_bytes == 2) {
        throw std::logic_error("Two-byte K-parallel packets are currently specialized only for FP8 input.");
    }
    if (fp8_input && !kParallelFp8OutputDTypeSupported(stage_output.getDataType())) {
        throw std::logic_error("FP8 K-parallel writeback requires a floating-point output dtype.");
    }
    if (fp8_input && plan.packet_bytes < 16
        && (reinterpret_cast<uintptr_t>(input.getMemPtr()) % plan.packet_bytes) != 0) {
        throw std::logic_error(
            "Narrow FP8 K-parallel packets require the input base to be naturally packet-aligned.");
    }
#endif
    if (plan.packet_bytes < input_element_bytes || plan.packet_bytes % input_element_bytes != 0
        || geometry.inner_size % static_cast<uint64_t>(plan.packet_bytes / input_element_bytes) != 0) {
        throw std::logic_error("K-parallel tiled stage requires naturally aligned whole-row input packets.");
    }
    if (plan.block_threads != 32 && plan.block_threads != 64 && plan.block_threads != 128
        && plan.block_threads != 256) {
        throw std::logic_error("K-parallel tiled stage requires a compile-time-specialized CTA width.");
    }
#if THOR_CUB_ENABLE_FP8_TYPES
    if (fp8_input && (reinterpret_cast<uintptr_t>(stage_output.getMemPtr()) & uintptr_t{15}) != 0) {
        throw std::logic_error("FP8 K-parallel writeback requires a 16-byte-aligned output base.");
    }
#endif

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
#if THOR_CUB_ENABLE_FP8_TYPES
            if (fp8_input) {
                dispatchFp8AlignedKStageSemantics<TiledRKStageTopology::Staged>(
                    partial_semantics, input, stage_output, geometry, plan, 1.0f, stream);
                return;
            }
#endif
            dispatchStageSemantics<TiledRKStageTopology::Staged>(
                partial_semantics, input, stage_output, geometry, plan, 1.0f, stream);
            return;
        }
        case TiledRKStageTopology::Complete:
            if (plan.shards_per_output != 1) {
                throw std::logic_error("K-parallel complete topology requires exactly one R shard.");
            }
#if THOR_CUB_ENABLE_FP8_TYPES
            if (fp8_input) {
                if (geometry.tiled_output_inner_stride != 1) {
                    throw std::logic_error("FP8 K-parallel complete writeback requires contiguous inner output.");
                }
                const size_t output_element_bytes = TensorDescriptor::getElementSizeInBytes(stage_output.getDataType());
                if (output_element_bytes == 0) {
                    throw std::logic_error("FP8 K-parallel complete writeback requires a valid output dtype.");
                }
                const size_t producer_output_bytes = plan.packet_bytes * output_element_bytes;
                const size_t store_alignment = std::min<size_t>(producer_output_bytes, 16U);
                if (((geometry.tiled_output_outer_stride % store_alignment) * output_element_bytes)
                    % store_alignment != 0) {
                    throw std::logic_error(
                        "FP8 K-parallel complete writeback requires rows aligned to its physical store width.");
                }
            }
#endif
            validateCompleteOutput(stage_output, geometry);
#if THOR_CUB_ENABLE_FP8_TYPES
            if (fp8_input) {
                dispatchFp8AlignedKStageSemantics<TiledRKStageTopology::Complete>(
                    semantics, input, stage_output, geometry, plan, output_scale, stream);
                return;
            }
#endif
            dispatchStageSemantics<TiledRKStageTopology::Complete>(
                semantics, input, stage_output, geometry, plan, output_scale, stream);
            return;
    }
    throw std::logic_error("K-parallel tiled stage selected an invalid topology.");
}


// Exact operation-aware Complete occupancy; unlike the generic KParallel
// occupancy interface this accounts for P8/CTA64's mixed additive policy.
CubKernelOccupancyInfo queryFp8AlignedKCompleteOccupancy(
    DataType input_dtype, DataType output_dtype, CubReductionStageCombine operation,
    const KParallelTiledStagePlan& plan) {
#if THOR_CUB_ENABLE_FP8_TYPES
    if (plan.topology != TiledRKStageTopology::Complete || plan.shards_per_output != 1 ||
        !((plan.packet_bytes == 2 && plan.block_threads == 32) ||
          (plan.packet_bytes == 8 && (plan.block_threads == 32 || plan.block_threads == 64))) ||
        !kParallelFp8OutputDTypeSupported(output_dtype))
        throw std::logic_error("AlignedK Complete occupancy requires a retained physical configuration.");
    switch (operation) {
        case CubReductionStageCombine::Sum:
        case CubReductionStageCombine::Product:
        case CubReductionStageCombine::Minimum:
        case CubReductionStageCombine::Maximum:
            break;
        default:
            throw std::logic_error("AlignedK Complete occupancy requires a supported combine operation.");
    }
    const auto per_dtype = [&]<typename T>() -> CubKernelOccupancyInfo {
        const auto query_kernel = [&]<typename KernelT>(KernelT kernel, int threads, size_t smem) {
            cudaFuncAttributes attrs{};
            CUDA_CHECK(cudaFuncGetAttributes(&attrs, kernel));
            int active = 0;
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, kernel, threads, smem));
            return CubKernelOccupancyInfo{attrs.numRegs, attrs.sharedSizeBytes, smem,
                                          active, active * (threads / TILED_REDUCTION_WARP_THREADS)};
        };
        if (plan.packet_bytes == 2) {
            const size_t smem = kParallelFp8CompleteNeedsTranspose<2>(output_dtype)
                ? K_PARALLEL_FP8_SHARED_BYTES_PER_WARP<2> : 0;
            return query_kernel(kParallelFp8CompleteUnifiedKernel<T, 2, 32, Fp8CompleteFinalizePolicy::Joined>,
                                32, smem);
        }
        if (plan.block_threads == 32) {
            const size_t smem = kParallelFp8CompleteNeedsTranspose<8>(output_dtype)
                ? K_PARALLEL_FP8_SHARED_BYTES_PER_WARP<8> : 0;
            return query_kernel(kParallelFp8CompleteUnifiedKernel<T, 8, 32, Fp8CompleteFinalizePolicy::InBranch>,
                                32, smem);
        }
        const size_t smem = kParallelFp8CompleteNeedsTranspose<8>(output_dtype)
            ? 2 * K_PARALLEL_FP8_SHARED_BYTES_PER_WARP<8> : 0;
        if (operation == CubReductionStageCombine::Sum)
            return query_kernel(kParallelFp8PacketKernel<T, IdentityFp32, cuda::std::plus<float>,
                                AdditiveFinalizeFp32, 8, 64, TiledRKStageTopology::Complete>, 64, smem);
        return query_kernel(kParallelFp8CompleteUnifiedKernel<T, 8, 64, Fp8CompleteFinalizePolicy::InBranch>,
                            64, smem);
    };
    switch (input_dtype) {
        case DataType::FP8_E4M3: return per_dtype.template operator()<__nv_fp8_e4m3>();
        case DataType::FP8_E5M2: return per_dtype.template operator()<__nv_fp8_e5m2>();
        default: throw std::logic_error("AlignedK Complete occupancy requires FP8 input.");
    }
#else
    static_cast<void>(input_dtype); static_cast<void>(output_dtype);
    static_cast<void>(operation); static_cast<void>(plan);
    throw std::logic_error("AlignedK Complete occupancy requires FP8 support.");
#endif
}


#if THOR_CUB_ENABLE_FP8_TYPES

template <TiledRKStageTopology Topology, typename InputT, typename TransformT,
          typename ReductionOpT, typename FinalizeT>
[[nodiscard]] CubKernelOccupancyInfo queryFp8AlignedKTransformedTyped(
    DataType output_dtype, const KParallelTiledStagePlan& plan) {
    const auto query = [&]<size_t PacketBytes, int BlockThreads>() -> CubKernelOccupancyInfo {
        if constexpr (prunedFp8AlignedInstantiation<InputT, TransformT, ReductionOpT,
                                                    PacketBytes, BlockThreads, Topology>()) {
            return {}; // Physical kernel removed; the planner must choose another configuration.
        } else {
            const bool needs_transpose = Topology == TiledRKStageTopology::Staged
                ? K_PARALLEL_FP8_STAGED_NEEDS_TRANSPOSE<PacketBytes>
                : kParallelFp8CompleteNeedsTranspose<PacketBytes>(output_dtype);
            const size_t smem = needs_transpose
                ? (BlockThreads / TILED_REDUCTION_WARP_THREADS) * K_PARALLEL_FP8_SHARED_BYTES_PER_WARP<PacketBytes>
                : 0;
            const auto kernel = kParallelFp8PacketKernel<InputT, TransformT, ReductionOpT, FinalizeT,
                                                         PacketBytes, BlockThreads, Topology>;
            cudaFuncAttributes attributes{};
            CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
            int active = 0;
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, kernel, BlockThreads, smem));
            return CubKernelOccupancyInfo{attributes.numRegs, attributes.sharedSizeBytes, smem,
                                          active, active * (BlockThreads / TILED_REDUCTION_WARP_THREADS)};
        }
    };
    switch (plan.packet_bytes) {
        case 2:
            switch (plan.block_threads) {
                case 32: return query.template operator()<2, 32>();
                case 64:
                    if constexpr (Topology == TiledRKStageTopology::Staged)
                        return query.template operator()<2, 64>();
                    break;
                case 128:
                    if constexpr (Topology == TiledRKStageTopology::Staged)
                        return query.template operator()<2, 128>();
                    break;
            }
            break;
        case 8:
            switch (plan.block_threads) {
                case 32: return query.template operator()<8, 32>();
                case 64: return query.template operator()<8, 64>();
                case 128:
                    if constexpr (Topology == TiledRKStageTopology::Staged)
                        return query.template operator()<8, 128>();
                    break;
            }
            break;
    }
    throw std::logic_error("Transformed AlignedK occupancy requires a retained packet/CTA configuration.");
}

template <TiledRKStageTopology Topology, typename InputT, typename TransformT>
[[nodiscard]] CubKernelOccupancyInfo queryFp8AlignedKTransformedCombine(
    DataType output_dtype, CubReductionStageCombine operation, const KParallelTiledStagePlan& plan) {
    switch (operation) {
        case CubReductionStageCombine::Sum:
            return queryFp8AlignedKTransformedTyped<Topology, InputT, TransformT,
                cuda::std::plus<float>, AdditiveFinalizeFp32>(output_dtype, plan);
        case CubReductionStageCombine::Product:
            return queryFp8AlignedKTransformedTyped<Topology, InputT, TransformT,
                cuda::std::multiplies<float>, IdentityFp32>(output_dtype, plan);
        case CubReductionStageCombine::Minimum:
            return queryFp8AlignedKTransformedTyped<Topology, InputT, TransformT,
                PropagatingMinimumFp32, IdentityFp32>(output_dtype, plan);
        case CubReductionStageCombine::Maximum:
            return queryFp8AlignedKTransformedTyped<Topology, InputT, TransformT,
                PropagatingMaximumFp32, IdentityFp32>(output_dtype, plan);
    }
    throw std::logic_error("Transformed AlignedK occupancy requires a supported combine operation.");
}

template <TiledRKStageTopology Topology, typename InputT>
[[nodiscard]] CubKernelOccupancyInfo queryFp8AlignedKTransform(
    DataType output_dtype, CubReductionStageInputTransform transform,
    CubReductionStageCombine operation, const KParallelTiledStagePlan& plan) {
    switch (transform) {
        case CubReductionStageInputTransform::AbsoluteValue:
            return queryFp8AlignedKTransformedCombine<Topology, InputT, AbsoluteValueFp32>(
                output_dtype, operation, plan);
        case CubReductionStageInputTransform::Square:
            return queryFp8AlignedKTransformedCombine<Topology, InputT, SquareFp32>(
                output_dtype, operation, plan);
        case CubReductionStageInputTransform::Identity:
            break;
    }
    throw std::logic_error("Transformed AlignedK occupancy requires ABS or square transform.");
}

#endif  // THOR_CUB_ENABLE_FP8_TYPES

// Occupancy of the ACTUALLY selected transformed FP8 AlignedK packet kernel.
// The legacy Identity occupancy queries remain responsible for the hoisted
// Staged and mixed Complete implementations; they must not be substituted
// for the transformed typed packet kernel.
CubKernelOccupancyInfo queryFp8AlignedKStageOccupancy(
    DataType input_dtype, DataType output_dtype, CubReductionStageInputTransform transform,
    CubReductionStageCombine operation, const KParallelTiledStagePlan& plan) {
    if (transform == CubReductionStageInputTransform::Identity)
        return plan.topology == TiledRKStageTopology::Complete
            ? queryFp8AlignedKCompleteOccupancy(input_dtype, output_dtype, operation, plan)
            : queryKParallelTiledStageOccupancy(input_dtype, plan);
#if THOR_CUB_ENABLE_FP8_TYPES
    if (!kParallelFp8OutputDTypeSupported(output_dtype))
        throw std::logic_error("Transformed AlignedK occupancy requires a floating-point output.");
    const auto typed = [&]<typename InputT>() -> CubKernelOccupancyInfo {
        if (plan.topology == TiledRKStageTopology::Staged) {
            if (plan.shards_per_output <= 1 || output_dtype != DataType::FP32)
                throw std::logic_error("Transformed AlignedK staged occupancy requires FP32 partials.");
            return queryFp8AlignedKTransform<TiledRKStageTopology::Staged, InputT>(
                output_dtype, transform, operation, plan);
        }
        if (plan.topology == TiledRKStageTopology::Complete && plan.shards_per_output == 1)
            return queryFp8AlignedKTransform<TiledRKStageTopology::Complete, InputT>(
                output_dtype, transform, operation, plan);
        throw std::logic_error("Transformed AlignedK occupancy requires valid staging topology.");
    };
    switch (input_dtype) {
        case DataType::FP8_E4M3: return typed.template operator()<__nv_fp8_e4m3>();
        case DataType::FP8_E5M2: return typed.template operator()<__nv_fp8_e5m2>();
        default: throw std::logic_error("Transformed AlignedK occupancy requires FP8 input.");
    }
#else
    static_cast<void>(input_dtype); static_cast<void>(output_dtype);
    static_cast<void>(operation); static_cast<void>(plan);
    throw std::logic_error("Transformed AlignedK occupancy requires FP8 support.");
#endif
}

}  // namespace ThorImplementation::CubReductionInternal
