#include "Utilities/TensorOperations/Reduction/Kernels/KernelLaunchers.h"
#include "Utilities/TensorOperations/Reduction/Kernels/ReductionOperation.cuh"
#include "Utilities/TensorOperations/Reduction/Kernels/FastDivU32.h"
#include "Utilities/TensorOperations/Reduction/Arg/Fp32ArgKey.cuh"
#include "Utilities/TensorOperations/Reduction/Arg/Fp8ArgKey.h"
#include "Utilities/TensorOperations/Reduction/Kernels/Fp8KernelPolicy.h"
#include "Utilities/TensorOperations/Reduction/Kernels/KernelCommon.cuh"
#include "Utilities/TensorOperations/Reduction/RK/Fp8/Fp8AlignedKSelector.h"

#include "Utilities/Exceptions.h"

#include <cuda/std/bit>
#include <cuda/std/functional>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace ThorImplementation::ReductionInternal {
namespace {

#if THOR_CUB_ENABLE_FP8_TYPES
template <typename InputT, typename TransformT, typename ReductionOpT,
          size_t PacketBytes, int BlockThreads, RKStageTopology Topology>
constexpr bool prunedFp8AlignedInstantiation() {
    constexpr unsigned dtype = std::is_same_v<InputT, __nv_fp8_e4m3> ? 0u : 1u;
    constexpr unsigned combine = std::is_same_v<ReductionOpT, cuda::std::plus<float>> ? 0u :
        std::is_same_v<ReductionOpT, cuda::std::multiplies<float>> ? 1u :
        std::is_same_v<ReductionOpT, PropagatingMinimumFp32> ? 2u : 3u;
    constexpr unsigned transform = std::is_same_v<TransformT, IdentityFp32> ? 0u :
        std::is_same_v<TransformT, AbsoluteValueFp32> ? 1u : 2u;
    return fp8AlignedTypedPruned(dtype, combine, transform, PacketBytes, BlockThreads,
                                 Topology == RKStageTopology::Staged);
}

// The FP8 AlignedK physical configurations, owned by RK::Fp8::Fp8AlignedKPolicy: Staged at every policy packet x CTA
// pair, Complete only where supportedComplete().
// FP8 launch and occupancy dispatch instantiate exactly these configurations.
template <RKStageTopology Topology, size_t PacketBytes, int BlockThreads>
[[nodiscard]] constexpr bool fp8AlignedKConfiguration() {
    namespace Policy = RK::Fp8::Fp8AlignedKPolicy;
    bool packet = false;
    for (uint32_t retained : Policy::packet_bytes) packet = packet || retained == PacketBytes;
    bool threads = false;
    for (uint32_t retained : Policy::cta_threads) threads = threads || retained == static_cast<uint32_t>(BlockThreads);
    if constexpr (Topology == RKStageTopology::Staged) {
        return packet && threads;
    } else {
        return packet && threads &&
               Policy::supportedComplete(static_cast<uint32_t>(PacketBytes), static_cast<uint32_t>(BlockThreads));
    }
}

// Invokes callback.template operator()<PacketBytes, BlockThreads>() for the compiled configuration matching the runtime
// request; returns false for any other request.
template <RKStageTopology Topology, typename Callback>
[[nodiscard]] bool dispatchFp8AlignedKConfiguration(size_t packet_bytes, uint32_t block_threads, Callback&& callback) {
    bool dispatched = false;
    const auto configuration = [&]<size_t PacketBytes, int BlockThreads>() {
        if constexpr (fp8AlignedKConfiguration<Topology, PacketBytes, BlockThreads>()) {
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
struct AlignedKReductionIdentityFp32;

template <>
struct AlignedKReductionIdentityFp32<cuda::std::plus<float>> {
    static constexpr float value = 0.0f;
};

template <>
struct AlignedKReductionIdentityFp32<cuda::std::multiplies<float>> {
    static constexpr float value = 1.0f;
};

template <>
struct AlignedKReductionIdentityFp32<PropagatingMinimumFp32> {
    static constexpr float value = std::numeric_limits<float>::infinity();
};

template <>
struct AlignedKReductionIdentityFp32<PropagatingMaximumFp32> {
    static constexpr float value = -std::numeric_limits<float>::infinity();
};

template <int ItemsPerLane>
__device__ inline void storeAlignedKFp32Packet(float* destination,
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
__device__ inline OutputT convertAlignedKFp32(float value) {
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
__device__ inline void storeAlignedKFinalPacketTyped(OutputT* output,
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

    // Packet-first epilogue: the common natural [outer,K] output emits one aligned packet per lane. Non-contiguous
    // output layouts or a misaligned row head stay correct through scalar stores; the RK planner can reject those
    // geometries when a wide packet store is required for the fast path.
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
                packed.values[item] = convertAlignedKFp32<OutputT>(finalized[item]);
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
            output[output_index] = convertAlignedKFp32<OutputT>(finalized[item]);
        }
    }
}

template <typename OutputFinalizeT, int ItemsPerLane>
__device__ inline void storeAlignedKFinalPacket(void* output,
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
            storeAlignedKFinalPacketTyped<__half>(static_cast<__half*>(output),
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
            storeAlignedKFinalPacketTyped<__nv_bfloat16>(static_cast<__nv_bfloat16*>(output),
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
            storeAlignedKFinalPacketTyped<float>(static_cast<float*>(output),
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

// The FP8 AlignedK kernels: the K-parallel reduction topology over one-byte storage, at the naturally coalesced
// 16/8/4/2-byte packet widths. Each lane owns one contiguous FP8 packet, performs its complete R reduction in FP32
// registers, and only changes physical ownership at writeback when those FP32 results would exceed Thor's
// 16-byte-per-lane global transaction ceiling. The shared writeback transpose is therefore required for staged
// packet16/packet8, but packet4 and packet2 emit their FP32 partials directly as coalesced 16-byte/8-byte lane stores.
// No other lane ever contributes to a lane's K accumulators; the transpose is a layout epilogue, not cross-lane
// reduction work.
constexpr int FP8_ALIGNED_K_SHARED_STRIPE_PITCH = 34;

template <size_t PacketBytes>
inline constexpr int FP8_ALIGNED_K_ITEMS_PER_LANE = static_cast<int>(PacketBytes);

template <size_t PacketBytes>
inline constexpr int FP8_ALIGNED_K_COMPONENTS_PER_WARP =
    TILED_REDUCTION_WARP_THREADS * FP8_ALIGNED_K_ITEMS_PER_LANE<PacketBytes>;

template <size_t PacketBytes>
inline constexpr size_t FP8_ALIGNED_K_SHARED_BYTES_PER_WARP =
    static_cast<size_t>(FP8_ALIGNED_K_ITEMS_PER_LANE<PacketBytes>)
    * static_cast<size_t>(FP8_ALIGNED_K_SHARED_STRIPE_PITCH) * sizeof(float);

static_assert(sizeof(__nv_fp8_e4m3) == 1);
static_assert(sizeof(__nv_fp8_e5m2) == 1);
static_assert(sizeof(RawVectorPacket<2>) == 2);
static_assert(sizeof(RawVectorPacket<4>) == 4);
static_assert(sizeof(RawVectorPacket<8>) == 8);
static_assert(sizeof(RawVectorPacket<16>) == 16);

template <typename T>
inline constexpr bool IS_FP8_ALIGNED_K_TYPE =
    std::is_same_v<T, __nv_fp8_e4m3> || std::is_same_v<T, __nv_fp8_e5m2>;

// The AlignedK R loop, with explicit load-ahead and paired FP8 decode.
//
// Each lane owns one packet per row and walks its shard's rows with stride inner_size. The loop is software-pipelined
// in batches of FP8_ALIGNED_K_LOAD_AHEAD_ROWS rows: the next batch is issued before the current batch is decoded, so up
// to two batches of loads are in flight while the lane converts and accumulates. P8 lanes keep 4 rows (2 x 32 bytes of
// packets in registers), P2 lanes 8 rows (2 x 16 bytes). Leaving this to nvcc's unroller produced 4-row trips with no
// cross-trip overlap, and the factor flipped with unrelated edits. Rows are still accumulated in increasing order per
// component, so results are bit-identical to the plain row loop.
template <size_t PacketBytes>
inline constexpr uint32_t FP8_ALIGNED_K_LOAD_AHEAD_ROWS = PacketBytes >= 8 ? 4U : 8U;

// The Staged hoisted (Identity, operation-unified) P8/CTA64 kernel batches 2 rows. With 4 rows it compiled at 72
// registers (14 blocks of 64 threads per SM) and was 1-5% slower than the plain row loop; 2 rows halve the in-flight
// packet registers and measured 0.944x the 4-row batch. The typed ABS/SQUARE Staged P8/CTA64 kernels were 4-5% slower
// with 2 rows and keep 4.
template <size_t PacketBytes, int BlockThreads>
inline constexpr uint32_t FP8_ALIGNED_K_HOISTED_STAGED_LOAD_AHEAD_ROWS =
    PacketBytes == 8 && BlockThreads == 64 ? 2U : FP8_ALIGNED_K_LOAD_AHEAD_ROWS<PacketBytes>;

template <typename InputT, int ItemsPerLane, uint32_t Rows>
struct Fp8AlignedKRowBatch {
    PackedInputValues<InputT, ItemsPerLane> rows[Rows];
};

template <typename InputT, int ItemsPerLane, uint32_t Rows>
__device__ __forceinline__ void loadFp8AlignedKRowBatch(Fp8AlignedKRowBatch<InputT, ItemsPerLane, Rows>& batch,
                                                        const InputT* row_input,
                                                        uint64_t inner_size) {
#pragma unroll
    for (uint32_t row = 0; row < Rows; ++row) {
        batch.rows[row] = loadVectorizedInputPacket<InputT, ItemsPerLane>(row_input + row * inner_size);
    }
}

template <typename InputT, int ItemsPerLane, uint32_t Rows, typename ReductionOpT, typename InputTransformT>
__device__ __forceinline__ void reduceFp8AlignedKRowBatch(float (&local)[ItemsPerLane],
                                                          const Fp8AlignedKRowBatch<InputT, ItemsPerLane, Rows>& batch,
                                                          const ReductionOpT& reduction_op,
                                                          const InputTransformT& input_transform) {
#pragma unroll
    for (uint32_t row = 0; row < Rows; ++row) {
        reducePackedFp8ValuesToFp32<InputT>(local, batch.rows[row], reduction_op, input_transform);
    }
}

// Accumulates row_count packets starting at row_input (stride inner_size) into local, which the caller initialized.
// Rows is the load-ahead batch (FP8_ALIGNED_K_LOAD_AHEAD_ROWS unless a caller sets it).
template <typename InputT,
          int ItemsPerLane,
          uint32_t Rows = FP8_ALIGNED_K_LOAD_AHEAD_ROWS<sizeof(InputT) * ItemsPerLane>,
          typename ReductionOpT,
          typename InputTransformT>
__device__ __forceinline__ void accumulateFp8AlignedKRows(float (&local)[ItemsPerLane],
                                                          const InputT* row_input,
                                                          uint32_t row_count,
                                                          uint64_t inner_size,
                                                          const ReductionOpT& reduction_op,
                                                          const InputTransformT& input_transform) {
    static_assert(IS_FP8_ALIGNED_K_TYPE<InputT>);
    static_assert(Rows >= 1);
    using BatchT = Fp8AlignedKRowBatch<InputT, ItemsPerLane, Rows>;
    const uint64_t batch_stride = static_cast<uint64_t>(Rows) * inner_size;
    uint32_t remaining = row_count;
    if (remaining >= Rows) {
        BatchT current;
        loadFp8AlignedKRowBatch(current, row_input, inner_size);
        row_input += batch_stride;
        remaining -= Rows;
        while (remaining >= Rows) {
            BatchT next;
            loadFp8AlignedKRowBatch(next, row_input, inner_size);
            row_input += batch_stride;
            remaining -= Rows;
            reduceFp8AlignedKRowBatch(local, current, reduction_op, input_transform);
            current = next;
        }
        reduceFp8AlignedKRowBatch(local, current, reduction_op, input_transform);
    }
    for (; remaining != 0; --remaining) {
        const PackedInputValues<InputT, ItemsPerLane> values =
            loadVectorizedInputPacket<InputT, ItemsPerLane>(row_input);
        reducePackedFp8ValuesToFp32<InputT>(local, values, reduction_op, input_transform);
        row_input += inner_size;
    }
}

[[nodiscard]] bool fp8AlignedKOutputDTypeSupported(DataType dtype) {
    switch (dtype) {
        case DataType::FP8_E4M3:
        case DataType::FP8_E5M2:
        case DataType::FP16:
        case DataType::BF16:
        case DataType::FP32:
            return true;
        default:
            return false;
    }
}

template <typename OutputT>
__device__ inline OutputT convertFp8AlignedKWriteback(float value) {
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
    } else {
        static_assert(!sizeof(OutputT), "Unsupported FP8 AlignedK output dtype.");
    }
}

template <typename OutputT, int ItemsPerLane, typename OutputFinalizeT>
__device__ inline void storeFp8AlignedKDirectFinalPacket(OutputT* output,
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
        packed.values[item] = convertFp8AlignedKWriteback<OutputT>(output_finalize(values[item]) * output_scale);
    }
    static_assert(sizeof(RawT) == sizeof(packed));
    const uint64_t output_index = outer_index * output_outer_stride + packet_component_begin;
    OutputT* destination = output + output_index;
    *reinterpret_cast<RawT*>(destination) = cuda::std::bit_cast<RawT>(packed);
}

template <typename OutputT, int ItemsPerLane>
__device__ inline void storeFp8AlignedKTransposedPackets(OutputT* output,
                                                         uint64_t output_row_base,
                                                         uint64_t warp_component_begin,
                                                         uint64_t valid_components,
                                                         const float* warp_tile) {
    static_assert(sizeof(OutputT) == 1 || sizeof(OutputT) == 2 || sizeof(OutputT) == 4
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
            const float value = warp_tile[producer_item * FP8_ALIGNED_K_SHARED_STRIPE_PITCH + producer_lane];
            packed.values[item] = convertFp8AlignedKWriteback<OutputT>(value);
        }

        OutputT* destination = output + output_row_base + warp_component_begin + component_offset;
        *reinterpret_cast<RawVectorPacket<16>*>(destination) =
            cuda::std::bit_cast<RawVectorPacket<16>>(packed);
    }
}

template <size_t PacketBytes>
inline constexpr bool FP8_ALIGNED_K_STAGED_NEEDS_TRANSPOSE =
    FP8_ALIGNED_K_ITEMS_PER_LANE<PacketBytes> * static_cast<int>(sizeof(float)) > 16;

template <size_t PacketBytes>
[[nodiscard]] bool fp8AlignedKCompleteNeedsTranspose(DataType output_dtype) {
    static_assert(PacketBytes == 2 || PacketBytes == 4 || PacketBytes == 8 || PacketBytes == 16);
    const size_t output_bytes = TensorDescriptor::getElementSizeInBytes(output_dtype);
    if (output_bytes == 0) {
        throw std::logic_error("FP8 AlignedK writeback received an invalid output dtype.");
    }
    return PacketBytes * output_bytes > 16;
}

template <typename InputT,
          typename InputTransformT,
          typename ReductionOpT,
          typename OutputFinalizeT,
          size_t PacketBytes,
          int BlockThreads,
          RKStageTopology Topology>
__global__ __launch_bounds__(BlockThreads) void fp8AlignedKValueKernel(const InputT* input,
                                                                        void* stage_output,
                                                                        DataType output_dtype,
                                                                        uint64_t outer_size,
                                                                        uint64_t reduction_size,
                                                                        uint64_t inner_size,
                                                                        uint64_t output_outer_stride,
                                                                        uint64_t output_inner_stride,
                                                                        uint64_t shards_per_output,
                                                                        uint64_t component_tiles,
                                                                        ReductionShardRows shard_rows,
                                                                        InputTransformT input_transform,
                                                                        OutputFinalizeT output_finalize,
                                                                        float output_scale) {
    static_assert(IS_FP8_ALIGNED_K_TYPE<InputT>);
    static_assert(PacketBytes == 2 || PacketBytes == 4 || PacketBytes == 8 || PacketBytes == 16);
    constexpr int ItemsPerLane = FP8_ALIGNED_K_ITEMS_PER_LANE<PacketBytes>;
    static_assert(sizeof(InputT) * ItemsPerLane == PacketBytes);
    static_assert(BlockThreads == 32 || BlockThreads == 64 || BlockThreads == 128 || BlockThreads == 256);
    static_assert(BlockThreads % TILED_REDUCTION_WARP_THREADS == 0);
    static_cast<void>(output_inner_stride);

    constexpr uint64_t components_per_cta =
        static_cast<uint64_t>(BlockThreads) * static_cast<uint64_t>(ItemsPerLane);
    constexpr float Init = AlignedKReductionIdentityFp32<ReductionOpT>::value;
    const ReductionOpT reduction_op{};
    static_cast<void>(output_finalize);  // Staged instantiations write unfinalized partials.

    extern __shared__ float fp8_writeback_shared[];
    const int warp = static_cast<int>(threadIdx.x) / TILED_REDUCTION_WARP_THREADS;
    const int lane = static_cast<int>(threadIdx.x) % TILED_REDUCTION_WARP_THREADS;
    float* warp_tile = fp8_writeback_shared
                       + static_cast<size_t>(warp) * static_cast<size_t>(ItemsPerLane)
                             * FP8_ALIGNED_K_SHARED_STRIPE_PITCH;

    // Grid: x = shard, y = component tile, z = outer (see reductionWorkGrid). The balanced row split is precomputed on
    // the host; the per-shard row loop runs on a 32-bit counter and a 64-bit row pointer.
    // reductionWorkGrid() limits every work extent to 2^31 - 1, so 32-bit work coordinates never wrap.
    const uint32_t outer_count = static_cast<uint32_t>(outer_size);
    const uint32_t tile_count = static_cast<uint32_t>(component_tiles);
    const uint32_t shard_count = static_cast<uint32_t>(shards_per_output);
    for (uint32_t outer_index = blockIdx.z; outer_index < outer_count; outer_index += gridDim.z)
    for (uint32_t component_tile = blockIdx.y; component_tile < tile_count; component_tile += gridDim.y)
    for (uint32_t shard = blockIdx.x; shard < shard_count; shard += gridDim.x) {
        const uint64_t tile_component_begin = component_tile * components_per_cta;
        const uint64_t packet_component_begin =
            tile_component_begin + static_cast<uint64_t>(threadIdx.x) * static_cast<uint64_t>(ItemsPerLane);
        const bool owns_packet = packet_component_begin < inner_size;

        const uint64_t row_begin = reductionShardRowBegin(shard_rows, shard);
        const uint32_t row_count = reductionShardRowCount(shard_rows, shard);

        float local[ItemsPerLane];
#pragma unroll
        for (int item = 0; item < ItemsPerLane; ++item) {
            local[item] = Init;
        }

        if (owns_packet) {
            const InputT* row_input =
                input + ((outer_index * reduction_size + row_begin) * inner_size + packet_component_begin);

            // Whole-row packet divisibility means every row start preserves PacketBytes alignment whenever the tensor
            // base does. That is the normal allocator path for all 16/8/4/2-byte specializations. Packet16 retains the
            // existing shifted-alias correctness path because its 16-byte aligned-window loader is specifically built
            // for that geometry; narrower packets use their native coalesced vector load directly.
            if constexpr (PacketBytes == 16) {
                if ((reinterpret_cast<uintptr_t>(input) & uintptr_t{15}) == 0) {
                    for (uint32_t row = 0; row < row_count; ++row) {
                        const PackedInputValues<InputT, ItemsPerLane> values =
                            loadVectorizedInputPacket<InputT, ItemsPerLane>(row_input);
#pragma unroll
                        for (int item = 0; item < ItemsPerLane; ++item) {
                            local[item] = reduction_op(
                                local[item], input_transform(ToFp32<InputT>{}(values.values[item])));
                        }
                        row_input += inner_size;
                    }
                } else {
                    for (uint32_t row = 0; row < row_count; ++row) {
                        const PackedInputValues<InputT, ItemsPerLane> values =
                            loadAlignmentSafeInputPacket<InputT, ItemsPerLane>(row_input);
#pragma unroll
                        for (int item = 0; item < ItemsPerLane; ++item) {
                            local[item] = reduction_op(
                                local[item], input_transform(ToFp32<InputT>{}(values.values[item])));
                        }
                        row_input += inner_size;
                    }
                }
            } else {
                accumulateFp8AlignedKRows<InputT, ItemsPerLane>(
                    local, row_input, row_count, inner_size, reduction_op, input_transform);
            }
        }

        if constexpr (Topology == RKStageTopology::Staged && !FP8_ALIGNED_K_STAGED_NEEDS_TRANSPOSE<PacketBytes>) {
            if (owns_packet) {
                float* fp32_partials = static_cast<float*>(stage_output);
                float* destination = fp32_partials
                                     + (outer_index * shards_per_output + shard) * inner_size
                                     + packet_component_begin;
                // packet4 => one coalesced 16-byte FP32 store/lane; packet2 => one coalesced 8-byte store/lane.
                storeAlignedKFp32Packet<ItemsPerLane>(destination, local, ItemsPerLane);
            }
            continue;
        }

        if constexpr (Topology == RKStageTopology::Complete) {
            // Complete values are finalized once before storage; the stores only scale and convert.
            const IdentityFp32 store_finalize{};
            if (owns_packet) {
                // Finalize once, before the uniform output-dtype switch, instead of inside every dtype's store (nvcc
                // kept one inline divide/sqrt finalizer per item per dtype). Stores below receive finalized values and
                // still apply output_scale, as in fp8AlignedKValueCompleteUnifiedKernel.
#pragma unroll
                for (int item = 0; item < ItemsPerLane; ++item) {
                    local[item] = finalizeFp32Epilogue(output_finalize, local[item]);
                }
                switch (output_dtype) {
                    case DataType::FP8_E4M3:
                        if constexpr (sizeof(__nv_fp8_e4m3) * static_cast<size_t>(ItemsPerLane) <= 16) {
                            storeFp8AlignedKDirectFinalPacket(static_cast<__nv_fp8_e4m3*>(stage_output),
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
                            storeFp8AlignedKDirectFinalPacket(static_cast<__nv_fp8_e5m2*>(stage_output),
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
                            storeFp8AlignedKDirectFinalPacket(static_cast<__half*>(stage_output),
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
                            storeFp8AlignedKDirectFinalPacket(static_cast<__nv_bfloat16*>(stage_output),
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
                            storeFp8AlignedKDirectFinalPacket(static_cast<float*>(stage_output),
                                                              outer_index,
                                                              packet_component_begin,
                                                              output_outer_stride,
                                                              local,
                                                              store_finalize,
                                                              output_scale);
                        }
                        break;
                    default:
                        break;
                }
            }
            // The output dtype is warp-uniform. When direct storage is legal every active packet owner takes it, while
            // inactive tail lanes have no value to stage. No synchronization is required on this path.
            if ((output_dtype == DataType::FP8_E4M3 || output_dtype == DataType::FP8_E5M2)
                || (ItemsPerLane <= 8 && (output_dtype == DataType::FP16 || output_dtype == DataType::BF16))
                || (ItemsPerLane <= 4 && output_dtype == DataType::FP32)
            ) {
                continue;
            }
        }

        if (owns_packet) {
#pragma unroll
            for (int item = 0; item < ItemsPerLane; ++item) {
                const float value = Topology == RKStageTopology::Staged
                                        ? local[item]
                                        : local[item] * output_scale;
                // Pitch 34 is conflict-free for both producer [item][lane] stores and every 16-byte transposed
                // consumer packet used by the packet8/packet16 staged paths and wider complete-output conversions.
                warp_tile[item * FP8_ALIGNED_K_SHARED_STRIPE_PITCH + lane] = value;
            }
        }
        __syncwarp();

        constexpr uint64_t ComponentsPerWarp =
            static_cast<uint64_t>(FP8_ALIGNED_K_COMPONENTS_PER_WARP<PacketBytes>);
        const uint64_t warp_component_begin =
            tile_component_begin + static_cast<uint64_t>(warp) * ComponentsPerWarp;
        const uint64_t valid_components =
            warp_component_begin < inner_size ? minU64(ComponentsPerWarp, inner_size - warp_component_begin) : 0;

        if (valid_components != 0) {
            if constexpr (Topology == RKStageTopology::Staged) {
                if constexpr (FP8_ALIGNED_K_STAGED_NEEDS_TRANSPOSE<PacketBytes>) {
                    float* fp32_partials = static_cast<float*>(stage_output);
                    const uint64_t output_row_base = (outer_index * shards_per_output + shard) * inner_size;
                    storeFp8AlignedKTransposedPackets<float, ItemsPerLane>(
                        fp32_partials, output_row_base, warp_component_begin, valid_components, warp_tile);
                }
            } else {
                static_assert(Topology == RKStageTopology::Complete);
                const uint64_t output_row_base = outer_index * output_outer_stride;
                switch (output_dtype) {
                    case DataType::FP16:
                        if constexpr (sizeof(__half) * static_cast<size_t>(ItemsPerLane) > 16) {
                            storeFp8AlignedKTransposedPackets<__half, ItemsPerLane>(
                                static_cast<__half*>(stage_output),
                                output_row_base,
                                warp_component_begin,
                                valid_components,
                                warp_tile);
                        }
                        break;
                    case DataType::BF16:
                        if constexpr (sizeof(__nv_bfloat16) * static_cast<size_t>(ItemsPerLane) > 16) {
                            storeFp8AlignedKTransposedPackets<__nv_bfloat16, ItemsPerLane>(
                                static_cast<__nv_bfloat16*>(stage_output),
                                output_row_base,
                                warp_component_begin,
                                valid_components,
                                warp_tile);
                        }
                        break;
                    case DataType::FP32:
                        if constexpr (sizeof(float) * static_cast<size_t>(ItemsPerLane) > 16) {
                            storeFp8AlignedKTransposedPackets<float, ItemsPerLane>(
                                static_cast<float*>(stage_output),
                                output_row_base,
                                warp_component_begin,
                                valid_components,
                                warp_tile);
                        }
                        break;
                    default:
                        break;
                }
            }
        }

        // A later work item of this block may reuse this warp's private shared tile. No inter-warp
        // reduction/cooperation is involved, so a warp-local barrier is sufficient.
        __syncwarp();
    }
}


// AlignedK Staged: one operation-unified kernel per FP8 packet/CTA configuration.
// SUM/PRODUCT/MIN/MAX are selected once per output tile outside the R accumulation loop.
// Other input transforms retain their typed specialization. Identity Complete
// has its own mixed typed/unified operation policy below.
// Covers the six P2/P8 x CTA32/64/128 physical configurations, in place of four Identity-operation variants each.
// The launch-uniform operation branch is outside the R row loop and is executed once
// per owned (outer, K-tile, shard) work item, rather than once per input packet.
// Input-transformed staged variants continue using typed kernels.
// Force-inlined: the hoisted staged and unified complete kernels call this from four operation branches. As an
// out-of-line call, `local` is passed by reference and lives on the stack (STACK > 0 for the hoisted kernels).
template <typename InputT, typename ReductionOpT, int PacketBytes,
          uint32_t LoadAheadRows = FP8_ALIGNED_K_LOAD_AHEAD_ROWS<static_cast<size_t>(PacketBytes)>>
__device__ __forceinline__ void fp8AlignedKHoistedAccumulateRows(const InputT* row_input,
                                                          uint32_t row_count,
                                                          uint64_t inner_size,
                                                          float (&local)[PacketBytes]) {
    constexpr float Init = AlignedKReductionIdentityFp32<ReductionOpT>::value;
    const ReductionOpT reduction_op{};
#pragma unroll
    for (int item = 0; item < PacketBytes; ++item) {
        local[item] = Init;
    }
    accumulateFp8AlignedKRows<InputT, PacketBytes, LoadAheadRows>(
        local, row_input, row_count, inner_size, reduction_op, IdentityFp32{});
}

template <typename InputT, int PacketBytes, int BlockThreads>
__global__ __launch_bounds__(BlockThreads) void fp8AlignedKValueStagedUnifiedKernel(
    const InputT* input, float* fp32_partials, uint64_t outer_size,
    uint64_t reduction_size, uint64_t inner_size, uint64_t shards_per_output,
    uint64_t component_tiles, ReductionShardRows shard_rows,
    ReductionStageCombine operation) {
    static_assert(IS_FP8_ALIGNED_K_TYPE<InputT>);
    static_assert(PacketBytes == 2 || PacketBytes == 8);
    static_assert(BlockThreads == 32 || BlockThreads == 64 || BlockThreads == 128);
    constexpr uint32_t StagedLoadAheadRows =
        FP8_ALIGNED_K_HOISTED_STAGED_LOAD_AHEAD_ROWS<static_cast<size_t>(PacketBytes), BlockThreads>;
    constexpr uint64_t ComponentsPerCTA = static_cast<uint64_t>(PacketBytes) * BlockThreads;
    extern __shared__ float fp8_writeback_shared[];
    const int warp = static_cast<int>(threadIdx.x) / TILED_REDUCTION_WARP_THREADS;
    const int lane = static_cast<int>(threadIdx.x) % TILED_REDUCTION_WARP_THREADS;
    float* warp_tile = fp8_writeback_shared + static_cast<size_t>(warp) * PacketBytes
                                            * FP8_ALIGNED_K_SHARED_STRIPE_PITCH;
    // Grid: x = shard, y = component tile, z = outer (see reductionWorkGrid).
    // reductionWorkGrid() limits every work extent to 2^31 - 1, so 32-bit work coordinates never wrap.
    const uint32_t outer_count = static_cast<uint32_t>(outer_size);
    const uint32_t tile_count = static_cast<uint32_t>(component_tiles);
    const uint32_t shard_count = static_cast<uint32_t>(shards_per_output);
    for (uint32_t outer_index = blockIdx.z; outer_index < outer_count; outer_index += gridDim.z)
    for (uint32_t component_tile = blockIdx.y; component_tile < tile_count; component_tile += gridDim.y)
    for (uint32_t shard = blockIdx.x; shard < shard_count; shard += gridDim.x) {
        const uint64_t tile_component_begin = component_tile * ComponentsPerCTA;
        const uint64_t packet_component_begin =
            tile_component_begin + static_cast<uint64_t>(threadIdx.x) * PacketBytes;
        const bool owns_packet = packet_component_begin < inner_size;
        const uint64_t row_begin = reductionShardRowBegin(shard_rows, shard);
        const uint32_t row_count = reductionShardRowCount(shard_rows, shard);
        float local[PacketBytes]{};
        if (owns_packet) {
            const InputT* row_input =
                input + ((outer_index * reduction_size + row_begin) * inner_size + packet_component_begin);
            // One uniform switch PER shard/tile, not per R row or decoded packet.
            switch (operation) {
                case ReductionStageCombine::Sum:
                    fp8AlignedKHoistedAccumulateRows<InputT, cuda::std::plus<float>, PacketBytes,
                                                     StagedLoadAheadRows>(
                        row_input, row_count, inner_size, local);
                    break;
                case ReductionStageCombine::Product:
                    fp8AlignedKHoistedAccumulateRows<InputT, cuda::std::multiplies<float>, PacketBytes,
                                                     StagedLoadAheadRows>(
                        row_input, row_count, inner_size, local);
                    break;
                case ReductionStageCombine::Minimum:
                    fp8AlignedKHoistedAccumulateRows<InputT, PropagatingMinimumFp32, PacketBytes,
                                                     StagedLoadAheadRows>(
                        row_input, row_count, inner_size, local);
                    break;
                case ReductionStageCombine::Maximum:
                    fp8AlignedKHoistedAccumulateRows<InputT, PropagatingMaximumFp32, PacketBytes,
                                                     StagedLoadAheadRows>(
                        row_input, row_count, inner_size, local);
                    break;
            }
        }
        const uint64_t output_row_base = (outer_index * shards_per_output + shard) * inner_size;
        if constexpr (PacketBytes == 2) {
            if (owns_packet) {
                storeAlignedKFp32Packet<PacketBytes>(
                    fp32_partials + output_row_base + packet_component_begin, local, PacketBytes);
            }
        } else {
            if (owns_packet) {
#pragma unroll
                for (int item = 0; item < PacketBytes; ++item) {
                    warp_tile[item * FP8_ALIGNED_K_SHARED_STRIPE_PITCH + lane] = local[item];
                }
            }
            __syncwarp();
            constexpr uint64_t ComponentsPerWarp =
                static_cast<uint64_t>(FP8_ALIGNED_K_COMPONENTS_PER_WARP<PacketBytes>);
            const uint64_t warp_component_begin =
                tile_component_begin + static_cast<uint64_t>(warp) * ComponentsPerWarp;
            const uint64_t valid_components = warp_component_begin < inner_size
                ? minU64(ComponentsPerWarp, inner_size - warp_component_begin) : 0;
            if (valid_components != 0) {
                storeFp8AlignedKTransposedPackets<float, PacketBytes>(
                    fp32_partials, output_row_base, warp_component_begin, valid_components, warp_tile);
            }
            __syncwarp();
        }
    }
}

// The FP8 Complete operation-unified physical kernels. The two finalization placements are intentional: P2/CTA32 uses a
// post-switch join, while P8/CTA32 and P8/CTA64 retain the in-branch schedule. In particular the P8/CTA64 nonadditive
// dispatch MUST NOT strip the additive code from the physical kernel: that apparently harmless simplification regressed
// MAX at high bandwidth.
enum class Fp8CompleteFinalizePolicy { InBranch, Joined };

template <typename InputT, size_t PacketBytes, int BlockThreads, Fp8CompleteFinalizePolicy Policy>
__global__ __launch_bounds__(BlockThreads) void fp8AlignedKValueCompleteUnifiedKernel(const InputT* input,
                                                                        void* stage_output,
                                                                        DataType output_dtype,
                                                                        uint64_t outer_size,
                                                                        uint64_t reduction_size,
                                                                        uint64_t inner_size,
                                                                        uint64_t output_outer_stride,
                                                                        uint64_t output_inner_stride,
                                                                        uint64_t shards_per_output,
                                                                        uint64_t component_tiles,
                                                                        ReductionShardRows shard_rows,
                                                                        ReductionStageCombine operation,
                                                                        AdditiveFinalizeFp32 output_finalize,
                                                                        float output_scale) {
    static_assert(IS_FP8_ALIGNED_K_TYPE<InputT>);
    static_assert(PacketBytes == 2 || PacketBytes == 8);
    static_assert((PacketBytes == 2 && BlockThreads == 32) ||
                  (PacketBytes == 8 && (BlockThreads == 32 || BlockThreads == 64)));
    constexpr int ItemsPerLane = FP8_ALIGNED_K_ITEMS_PER_LANE<PacketBytes>;
    static_assert(sizeof(InputT) * ItemsPerLane == PacketBytes);
    static_assert(BlockThreads == 32 || BlockThreads == 64 || BlockThreads == 128 || BlockThreads == 256);
    static_assert(BlockThreads % TILED_REDUCTION_WARP_THREADS == 0);
    static_cast<void>(output_inner_stride);

    constexpr uint64_t components_per_cta =
        static_cast<uint64_t>(BlockThreads) * static_cast<uint64_t>(ItemsPerLane);
    constexpr RKStageTopology Topology = RKStageTopology::Complete;
    // Finalization is applied to SUM/MEAN once per decoded packet. Storage
    // always receives already-finalized values (and applies runtime output scale).
    const IdentityFp32 store_finalize{};

    extern __shared__ float fp8_writeback_shared[];
    const int warp = static_cast<int>(threadIdx.x) / TILED_REDUCTION_WARP_THREADS;
    const int lane = static_cast<int>(threadIdx.x) % TILED_REDUCTION_WARP_THREADS;
    float* warp_tile = fp8_writeback_shared
                       + static_cast<size_t>(warp) * static_cast<size_t>(ItemsPerLane)
                             * FP8_ALIGNED_K_SHARED_STRIPE_PITCH;

    // Grid: x = shard, y = component tile, z = outer (see reductionWorkGrid).
    // reductionWorkGrid() limits every work extent to 2^31 - 1, so 32-bit work coordinates never wrap.
    const uint32_t outer_count = static_cast<uint32_t>(outer_size);
    const uint32_t tile_count = static_cast<uint32_t>(component_tiles);
    const uint32_t shard_count = static_cast<uint32_t>(shards_per_output);
    for (uint32_t outer_index = blockIdx.z; outer_index < outer_count; outer_index += gridDim.z)
    for (uint32_t component_tile = blockIdx.y; component_tile < tile_count; component_tile += gridDim.y)
    for (uint32_t shard = blockIdx.x; shard < shard_count; shard += gridDim.x) {
        const uint64_t tile_component_begin = component_tile * components_per_cta;
        const uint64_t packet_component_begin =
            tile_component_begin + static_cast<uint64_t>(threadIdx.x) * static_cast<uint64_t>(ItemsPerLane);
        const bool owns_packet = packet_component_begin < inner_size;

        const uint64_t row_begin = reductionShardRowBegin(shard_rows, shard);
        const uint32_t row_count = reductionShardRowCount(shard_rows, shard);

        float local[ItemsPerLane]{};

        if (owns_packet) {
            const InputT* row_input =
                input + ((outer_index * reduction_size + row_begin) * inner_size + packet_component_begin);

            // Uniform operation selection outside the R loop. Both policies
            // retain all four operation bodies, including the additive branch
            // in the P8/CTA64 nonadditive physical instance.
            switch (operation) {
                case ReductionStageCombine::Sum:
                    fp8AlignedKHoistedAccumulateRows<InputT, cuda::std::plus<float>, ItemsPerLane>(
                        row_input, row_count, inner_size, local);
                    if constexpr (Policy == Fp8CompleteFinalizePolicy::InBranch) {
                        // Deliberately the inline finalizer, not finalizeFp32Epilogue(): the out-of-line one raised the
                        // P8/CTA32 instance from 79 to 82 registers and slowed identity-transform shapes by up to 48%
                        // (3511x224x32). Its finalizer is not replicated per output dtype, so little SASS is at stake.
#pragma unroll
                        for (int item = 0; item < ItemsPerLane; ++item)
                            local[item] = output_finalize(local[item]);
                    }
                    break;
                case ReductionStageCombine::Product:
                    fp8AlignedKHoistedAccumulateRows<InputT, cuda::std::multiplies<float>, ItemsPerLane>(
                        row_input, row_count, inner_size, local);
                    break;
                case ReductionStageCombine::Minimum:
                    fp8AlignedKHoistedAccumulateRows<InputT, PropagatingMinimumFp32, ItemsPerLane>(
                        row_input, row_count, inner_size, local);
                    break;
                case ReductionStageCombine::Maximum:
                    fp8AlignedKHoistedAccumulateRows<InputT, PropagatingMaximumFp32, ItemsPerLane>(
                        row_input, row_count, inner_size, local);
                    break;
            }
            if constexpr (Policy == Fp8CompleteFinalizePolicy::Joined) {
                if (operation == ReductionStageCombine::Sum) {
#pragma unroll
                    for (int item = 0; item < ItemsPerLane; ++item)
                        local[item] = finalizeFp32Epilogue(output_finalize, local[item]);
                }
            }
        }

        if constexpr (Topology == RKStageTopology::Staged && !FP8_ALIGNED_K_STAGED_NEEDS_TRANSPOSE<PacketBytes>) {
            if (owns_packet) {
                float* fp32_partials = static_cast<float*>(stage_output);
                float* destination = fp32_partials
                                     + (outer_index * shards_per_output + shard) * inner_size
                                     + packet_component_begin;
                // packet4 => one coalesced 16-byte FP32 store/lane; packet2 => one coalesced 8-byte store/lane.
                storeAlignedKFp32Packet<ItemsPerLane>(destination, local, ItemsPerLane);
            }
            continue;
        }

        if constexpr (Topology == RKStageTopology::Complete) {
            if (owns_packet) {
                switch (output_dtype) {
                    case DataType::FP8_E4M3:
                        if constexpr (sizeof(__nv_fp8_e4m3) * static_cast<size_t>(ItemsPerLane) <= 16) {
                            storeFp8AlignedKDirectFinalPacket(static_cast<__nv_fp8_e4m3*>(stage_output),
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
                            storeFp8AlignedKDirectFinalPacket(static_cast<__nv_fp8_e5m2*>(stage_output),
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
                            storeFp8AlignedKDirectFinalPacket(static_cast<__half*>(stage_output),
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
                            storeFp8AlignedKDirectFinalPacket(static_cast<__nv_bfloat16*>(stage_output),
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
                            storeFp8AlignedKDirectFinalPacket(static_cast<float*>(stage_output),
                                                              outer_index,
                                                              packet_component_begin,
                                                              output_outer_stride,
                                                              local,
                                                              store_finalize,
                                                              output_scale);
                        }
                        break;
                    default:
                        break;
                }
            }
            // The output dtype is warp-uniform. When direct storage is legal every active packet owner takes it, while
            // inactive tail lanes have no value to stage. No synchronization is required on this path.
            if ((output_dtype == DataType::FP8_E4M3 || output_dtype == DataType::FP8_E5M2)
                || (ItemsPerLane <= 8 && (output_dtype == DataType::FP16 || output_dtype == DataType::BF16))
                || (ItemsPerLane <= 4 && output_dtype == DataType::FP32)
            ) {
                continue;
            }
        }

        if (owns_packet) {
#pragma unroll
            for (int item = 0; item < ItemsPerLane; ++item) {
                const float value = Topology == RKStageTopology::Staged
                                        ? local[item]
                                        : store_finalize(local[item]) * output_scale;
                // Pitch 34 is conflict-free for both producer [item][lane] stores and every 16-byte transposed
                // consumer packet used by the packet8/packet16 staged paths and wider complete-output conversions.
                warp_tile[item * FP8_ALIGNED_K_SHARED_STRIPE_PITCH + lane] = value;
            }
        }
        __syncwarp();

        constexpr uint64_t ComponentsPerWarp =
            static_cast<uint64_t>(FP8_ALIGNED_K_COMPONENTS_PER_WARP<PacketBytes>);
        const uint64_t warp_component_begin =
            tile_component_begin + static_cast<uint64_t>(warp) * ComponentsPerWarp;
        const uint64_t valid_components =
            warp_component_begin < inner_size ? minU64(ComponentsPerWarp, inner_size - warp_component_begin) : 0;

        if (valid_components != 0) {
            if constexpr (Topology == RKStageTopology::Staged) {
                if constexpr (FP8_ALIGNED_K_STAGED_NEEDS_TRANSPOSE<PacketBytes>) {
                    float* fp32_partials = static_cast<float*>(stage_output);
                    const uint64_t output_row_base = (outer_index * shards_per_output + shard) * inner_size;
                    storeFp8AlignedKTransposedPackets<float, ItemsPerLane>(
                        fp32_partials, output_row_base, warp_component_begin, valid_components, warp_tile);
                }
            } else {
                static_assert(Topology == RKStageTopology::Complete);
                const uint64_t output_row_base = outer_index * output_outer_stride;
                switch (output_dtype) {
                    case DataType::FP16:
                        if constexpr (sizeof(__half) * static_cast<size_t>(ItemsPerLane) > 16) {
                            storeFp8AlignedKTransposedPackets<__half, ItemsPerLane>(
                                static_cast<__half*>(stage_output),
                                output_row_base,
                                warp_component_begin,
                                valid_components,
                                warp_tile);
                        }
                        break;
                    case DataType::BF16:
                        if constexpr (sizeof(__nv_bfloat16) * static_cast<size_t>(ItemsPerLane) > 16) {
                            storeFp8AlignedKTransposedPackets<__nv_bfloat16, ItemsPerLane>(
                                static_cast<__nv_bfloat16*>(stage_output),
                                output_row_base,
                                warp_component_begin,
                                valid_components,
                                warp_tile);
                        }
                        break;
                    case DataType::FP32:
                        if constexpr (sizeof(float) * static_cast<size_t>(ItemsPerLane) > 16) {
                            storeFp8AlignedKTransposedPackets<float, ItemsPerLane>(
                                static_cast<float*>(stage_output),
                                output_row_base,
                                warp_component_begin,
                                valid_components,
                                warp_tile);
                        }
                        break;
                    default:
                        break;
                }
            }
        }

        // A later work item of this block may reuse this warp's private shared tile. No inter-warp
        // reduction/cooperation is involved, so a warp-local barrier is sufficient.
        __syncwarp();
    }
}


#endif  // THOR_CUB_ENABLE_FP8_TYPES


// ---- AlignedK on FP32 input. A copy of the FP8 AlignedK
// mapping rather than a template of its kernels: FP8 lanes own PacketBytes one-byte components and decode pairs,
// transpose FP32 writeback above 16 bytes and keep typed / unified / pruned kernel splits calibrated on FP8. An FP32
// lane owns ItemsPerLane = 2 or 4 components (an 8- or 16-byte packet), so every FP32 partial packet is <= 16 bytes
// and stores directly (no transpose), and one kernel per (items, CTA, topology) selects the operation and transform
// once per work item, outside the R loop. Grid x = shard, y = component tile, z = outer, the balanced row split and
// the load-ahead row batches are FP8's; rows accumulate in increasing order per component.
// ItemsPerLane = 1 is the scalar packet (4 bytes: any K, any 4-byte aligned base), for odd K. Without it, odd K at
// small R ran only cooperative passes (RuntimeK, OneWindow), 0.2-0.6 us slower on 2-4 us reductions. Every load-ahead
// batch is 64 bytes per lane (16 rows of one float); a short tail loads all its rows before reducing any.
// FP16 / BF16 input goes through the InputT parameter: a lane owns a 2-, 4-, 8- or 16-byte packet (ItemsPerLane
// 1 / 2 / 4 / 8), widens each component to FP32 before the transform and accumulates in FP32 exactly as on FP32 input.
// The batch depth is set by the packet's bytes (16 rows up to 4 bytes, 8 rows of 8, 4 rows of 16: FP32's depths
// unchanged), and the load-all tail serves the packets up to 4 bytes (FP32's scalar packet, the 2-byte packets with one
// or two components).
template <size_t PacketBytes>
inline constexpr uint32_t ALIGNED_K_LOAD_AHEAD_ROWS = PacketBytes >= 16 ? 4U : (PacketBytes == 8 ? 8U : 16U);

template <int ItemsPerLane>
inline constexpr uint32_t FP32_ALIGNED_K_LOAD_AHEAD_ROWS = ALIGNED_K_LOAD_AHEAD_ROWS<sizeof(float) * ItemsPerLane>;

template <typename InputT, int ItemsPerLane, typename ReductionOpT, typename InputTransformT>
__device__ __forceinline__ void fp32AlignedKReducePacket(float (&local)[ItemsPerLane],
                                                         const PackedInputValues<InputT, ItemsPerLane>& packet,
                                                         const ReductionOpT& reduction_op,
                                                         const InputTransformT& input_transform) {
#pragma unroll
    for (int item = 0; item < ItemsPerLane; ++item)
        local[item] = reduction_op(local[item], input_transform(ToFp32<InputT>{}(packet.values[item])));
}

template <typename InputT, int ItemsPerLane, typename ReductionOpT, typename InputTransformT>
__device__ __forceinline__ void fp32AlignedKAccumulateRows(float (&local)[ItemsPerLane], const InputT* row_input,
                                                           uint32_t row_count, uint64_t inner_size) {
    constexpr size_t PacketBytes = sizeof(InputT) * ItemsPerLane;
    constexpr uint32_t Rows = ALIGNED_K_LOAD_AHEAD_ROWS<PacketBytes>;
    using PacketT = PackedInputValues<InputT, ItemsPerLane>;
    const ReductionOpT reduction_op{};
    const InputTransformT input_transform{};
#pragma unroll
    for (int item = 0; item < ItemsPerLane; ++item) local[item] = AlignedKReductionIdentityFp32<ReductionOpT>::value;
    uint32_t remaining = row_count;
    if (remaining >= Rows) {
        PacketT current[Rows];
#pragma unroll
        for (uint32_t row = 0; row < Rows; ++row)
            current[row] = loadVectorizedInputPacket<InputT, ItemsPerLane>(row_input + row * inner_size);
        row_input += static_cast<uint64_t>(Rows) * inner_size;
        remaining -= Rows;
        while (remaining >= Rows) {
            PacketT next[Rows];
#pragma unroll
            for (uint32_t row = 0; row < Rows; ++row)
                next[row] = loadVectorizedInputPacket<InputT, ItemsPerLane>(row_input + row * inner_size);
            row_input += static_cast<uint64_t>(Rows) * inner_size;
            remaining -= Rows;
#pragma unroll
            for (uint32_t row = 0; row < Rows; ++row)
                fp32AlignedKReducePacket(local, current[row], reduction_op, input_transform);
#pragma unroll
            for (uint32_t row = 0; row < Rows; ++row) current[row] = next[row];
        }
#pragma unroll
        for (uint32_t row = 0; row < Rows; ++row)
            fp32AlignedKReducePacket(local, current[row], reduction_op, input_transform);
    }
    if constexpr (PacketBytes <= 4) {
        // Fewer than Rows rows remain (all of them at the small R the scalar packet serves): load them all, then reduce
        // in row order, so the loads overlap instead of each waiting for the one before.
        if (remaining != 0) {
            PacketT tail[Rows - 1]{};
#pragma unroll
            for (uint32_t row = 0; row + 1 < Rows; ++row)
                if (row < remaining)
                    tail[row] = loadVectorizedInputPacket<InputT, ItemsPerLane>(row_input + row * inner_size);
#pragma unroll
            for (uint32_t row = 0; row + 1 < Rows; ++row)
                if (row < remaining) fp32AlignedKReducePacket(local, tail[row], reduction_op, input_transform);
        }
    } else {
        for (; remaining != 0; --remaining, row_input += inner_size)
            fp32AlignedKReducePacket(local, loadVectorizedInputPacket<InputT, ItemsPerLane>(row_input), reduction_op,
                                     input_transform);
    }
}

template <typename InputT, int ItemsPerLane, typename ReductionOpT>
__device__ __forceinline__ void fp32AlignedKAccumulateTransform(float (&local)[ItemsPerLane], const InputT* row_input,
                                                                uint32_t row_count, uint64_t inner_size,
                                                                ReductionStageInputTransform transform) {
    switch (transform) {
        case ReductionStageInputTransform::Identity:
            fp32AlignedKAccumulateRows<InputT, ItemsPerLane, ReductionOpT, IdentityFp32>(local, row_input, row_count,
                                                                                        inner_size);
            return;
        case ReductionStageInputTransform::AbsoluteValue:
            fp32AlignedKAccumulateRows<InputT, ItemsPerLane, ReductionOpT, AbsoluteValueFp32>(local, row_input,
                                                                                             row_count, inner_size);
            return;
        default:
            fp32AlignedKAccumulateRows<InputT, ItemsPerLane, ReductionOpT, SquareFp32>(local, row_input, row_count,
                                                                                      inner_size);
            return;
    }
}

template <typename InputT, int ItemsPerLane, int BlockThreads, RKStageTopology Topology>
__global__ __launch_bounds__(BlockThreads) void alignedKValueKernel(const InputT* input,
                                                                    void* stage_output,
                                                                    DataType output_dtype,
                                                                    uint64_t outer_size,
                                                                    uint64_t reduction_size,
                                                                    uint64_t inner_size,
                                                                    uint64_t output_outer_stride,
                                                                    uint64_t output_inner_stride,
                                                                    uint64_t shards_per_output,
                                                                    uint64_t component_tiles,
                                                                    ReductionShardRows shard_rows,
                                                                    ReductionStageCombine operation,
                                                                   ReductionStageInputTransform transform,
                                                                   AdditiveFinalizeFp32 output_finalize,
                                                                   float output_scale) {
    static_assert(std::is_same_v<InputT, float> || std::is_same_v<InputT, __half> ||
                  std::is_same_v<InputT, __nv_bfloat16>);
    static_assert(ItemsPerLane == 1 || ItemsPerLane == 2 || ItemsPerLane == 4 ||
                  (ItemsPerLane == 8 && sizeof(InputT) == 2));
    static_assert(BlockThreads == 32 || BlockThreads == 64 || BlockThreads == 128);
    constexpr uint64_t ComponentsPerCta = static_cast<uint64_t>(BlockThreads) * ItemsPerLane;
    const uint32_t outer_count = static_cast<uint32_t>(outer_size);
    const uint32_t tile_count = static_cast<uint32_t>(component_tiles);
    const uint32_t shard_count = static_cast<uint32_t>(shards_per_output);
    for (uint32_t outer_index = blockIdx.z; outer_index < outer_count; outer_index += gridDim.z)
    for (uint32_t component_tile = blockIdx.y; component_tile < tile_count; component_tile += gridDim.y)
    for (uint32_t shard = blockIdx.x; shard < shard_count; shard += gridDim.x) {
        const uint64_t packet_component_begin = static_cast<uint64_t>(component_tile) * ComponentsPerCta +
                                                static_cast<uint64_t>(threadIdx.x) * ItemsPerLane;
        if (packet_component_begin >= inner_size) continue;  // K % ItemsPerLane == 0: owned packets are whole
        const uint64_t row_begin = reductionShardRowBegin(shard_rows, shard);
        const uint32_t row_count = reductionShardRowCount(shard_rows, shard);
        const InputT* row_input =
            input + ((outer_index * reduction_size + row_begin) * inner_size + packet_component_begin);
        float local[ItemsPerLane];
        switch (operation) {
            case ReductionStageCombine::Sum:
                fp32AlignedKAccumulateTransform<InputT, ItemsPerLane, cuda::std::plus<float>>(
                    local, row_input, row_count, inner_size, transform);
                break;
            case ReductionStageCombine::Product:
                fp32AlignedKAccumulateRows<InputT, ItemsPerLane, cuda::std::multiplies<float>, IdentityFp32>(
                    local, row_input, row_count, inner_size);
                break;
            case ReductionStageCombine::Minimum:
                fp32AlignedKAccumulateTransform<InputT, ItemsPerLane, PropagatingMinimumFp32>(
                    local, row_input, row_count, inner_size, transform);
                break;
            default:
                fp32AlignedKAccumulateTransform<InputT, ItemsPerLane, PropagatingMaximumFp32>(
                    local, row_input, row_count, inner_size, transform);
                break;
        }
        if constexpr (Topology == RKStageTopology::Staged) {
            static_cast<void>(output_dtype);
            static_cast<void>(output_outer_stride);
            static_cast<void>(output_inner_stride);
            static_cast<void>(output_finalize);
            static_cast<void>(output_scale);
            float* partials = static_cast<float*>(stage_output) +
                              (static_cast<uint64_t>(outer_index) * shards_per_output + shard) * inner_size +
                              packet_component_begin;
            storeAlignedKFp32Packet<ItemsPerLane>(partials, local, ItemsPerLane);
        } else {
            if (operation == ReductionStageCombine::Sum) {
#pragma unroll
                for (int item = 0; item < ItemsPerLane; ++item)
                    local[item] = finalizeFp32Epilogue(output_finalize, local[item]);
            }
            storeAlignedKFinalPacket<IdentityFp32, ItemsPerLane>(stage_output, output_dtype, outer_index,
                                                                 packet_component_begin, output_outer_stride,
                                                                 output_inner_stride, local, ItemsPerLane,
                                                                 IdentityFp32{}, output_scale);
        }
    }
}

// ---- FP32 RK ARG, the AlignedK twin: alignedKValueKernel's
// ownership, row split and load-ahead batches, with one FP32 ARG candidate (Fp32ArgKey.cuh) per owned component in
// place of the accumulator. A lane's rows only increase, so a strictly smaller key replaces its best and ties keep the
// lower row. Complete (one shard) writes the requested outputs; Staged writes packed partials (key << 32 | shard row
// << 1 | negative zero) that fp32ArgPackedContinuationKernel finishes in the same pass. Rows per shard are at most
// FP32_ARG_MAX_SHARD_ROWS.
// FP16 / BF16 input: each component widens to FP32 (exact and order-preserving: NaN stays NaN, -0 stays -0) and takes
// the FP32 key, so the lane state, the packed partials and the continuation are FP32's. 4-, 8- and 16-byte packets (2,
// 4 or 8 components; no 2-byte packet, as FP32 ARG has no 4-byte one).
template <typename InputT, int ItemsPerLane>
__device__ __forceinline__ void fp32AlignedKArgKeyPacket(uint32_t (&best_key)[ItemsPerLane],
                                                         uint32_t (&best_tail)[ItemsPerLane],
                                                         const PackedInputValues<InputT, ItemsPerLane>& packet,
                                                         uint32_t row, uint32_t xor_mask) {
#pragma unroll
    for (int item = 0; item < ItemsPerLane; ++item) {
        uint32_t negative_zero;
        const uint32_t key =
            fp32ArgKey(__float_as_uint(ToFp32<InputT>{}(packet.values[item])), xor_mask, negative_zero);
        if (key < best_key[item]) {
            best_key[item] = key;
            best_tail[item] = (row << 1) | negative_zero;
        }
    }
}

template <typename InputT, int ItemsPerLane, int BlockThreads, bool Staged>
__global__ __launch_bounds__(BlockThreads) void alignedKArgKernel(const InputT* input,
                                                                  Fp32ArgOutputs outputs,
                                                                  uint64_t* packed,
                                                                  uint64_t outer_size,
                                                                  uint64_t reduction_size,
                                                                  uint64_t inner_size,
                                                                  uint64_t shards_per_output,
                                                                  uint64_t component_tiles,
                                                                  ReductionShardRows shard_rows) {
    static_assert(std::is_same_v<InputT, float> || std::is_same_v<InputT, __half> ||
                  std::is_same_v<InputT, __nv_bfloat16>);
    static_assert(ItemsPerLane == 2 || ItemsPerLane == 4 || (ItemsPerLane == 8 && sizeof(InputT) == 2));
    static_assert(BlockThreads == 32 || BlockThreads == 64 || BlockThreads == 128);
    constexpr uint64_t ComponentsPerCta = static_cast<uint64_t>(BlockThreads) * ItemsPerLane;
    constexpr uint32_t Rows = ALIGNED_K_LOAD_AHEAD_ROWS<sizeof(InputT) * ItemsPerLane>;
    using PacketT = PackedInputValues<InputT, ItemsPerLane>;
    const uint32_t outer_count = static_cast<uint32_t>(outer_size);
    const uint32_t tile_count = static_cast<uint32_t>(component_tiles);
    const uint32_t shard_count = static_cast<uint32_t>(shards_per_output);
    for (uint32_t outer_index = blockIdx.z; outer_index < outer_count; outer_index += gridDim.z)
    for (uint32_t component_tile = blockIdx.y; component_tile < tile_count; component_tile += gridDim.y)
    for (uint32_t shard = blockIdx.x; shard < shard_count; shard += gridDim.x) {
        const uint64_t packet_component_begin = static_cast<uint64_t>(component_tile) * ComponentsPerCta +
                                                static_cast<uint64_t>(threadIdx.x) * ItemsPerLane;
        if (packet_component_begin >= inner_size) continue;  // K % ItemsPerLane == 0: owned packets are whole
        const uint64_t row_begin = reductionShardRowBegin(shard_rows, shard);
        const uint32_t row_count = reductionShardRowCount(shard_rows, shard);
        const InputT* row_input =
            input + ((outer_index * reduction_size + row_begin) * inner_size + packet_component_begin);
        uint32_t best_key[ItemsPerLane], best_tail[ItemsPerLane];
#pragma unroll
        for (int item = 0; item < ItemsPerLane; ++item) {
            best_key[item] = FP32_ARG_KEY_IDENTITY;
            best_tail[item] = 0xFFFFFFFFU;
        }
        uint32_t row = 0;
        if (row_count >= Rows) {
            PacketT current[Rows];
#pragma unroll
            for (uint32_t r = 0; r < Rows; ++r)
                current[r] = loadVectorizedInputPacket<InputT, ItemsPerLane>(row_input + r * inner_size);
            row_input += static_cast<uint64_t>(Rows) * inner_size;
            while (row + 2 * Rows <= row_count) {
                PacketT next[Rows];
#pragma unroll
                for (uint32_t r = 0; r < Rows; ++r)
                    next[r] = loadVectorizedInputPacket<InputT, ItemsPerLane>(row_input + r * inner_size);
                row_input += static_cast<uint64_t>(Rows) * inner_size;
#pragma unroll
                for (uint32_t r = 0; r < Rows; ++r)
                    fp32AlignedKArgKeyPacket(best_key, best_tail, current[r], row + r, outputs.xor_mask);
                row += Rows;
#pragma unroll
                for (uint32_t r = 0; r < Rows; ++r) current[r] = next[r];
            }
#pragma unroll
            for (uint32_t r = 0; r < Rows; ++r)
                fp32AlignedKArgKeyPacket(best_key, best_tail, current[r], row + r, outputs.xor_mask);
            row += Rows;
        }
        for (; row < row_count; ++row, row_input += inner_size)
            fp32AlignedKArgKeyPacket(best_key, best_tail, loadVectorizedInputPacket<InputT, ItemsPerLane>(row_input),
                                     row, outputs.xor_mask);
        if constexpr (Staged) {
            uint64_t* partial =
                packed + (static_cast<uint64_t>(outer_index) * shards_per_output + shard) * inner_size +
                packet_component_begin;
#pragma unroll
            for (int item = 0; item < ItemsPerLane; ++item)
                partial[item] = (static_cast<uint64_t>(best_key[item]) << 32) | best_tail[item];
        } else {
            static_cast<void>(packed);
#pragma unroll
            for (int item = 0; item < ItemsPerLane; ++item)
                storeFp32ArgResult(outputs, static_cast<uint64_t>(outer_index) * inner_size + packet_component_begin +
                                                item,
                                   Fp32ArgCandidate{best_key[item], best_tail[item]});
        }
    }
}

// Reduces packed partials [outer, shards, K] over shards (shard s begins at reductionShardRowBegin) and writes the
// requested outputs: FP8's per-output continuation (one thread per output component, coalesced reads per shard).
template <int Unused>
__global__ __launch_bounds__(256) void fp32ArgPackedContinuationKernel(const uint64_t* packed, Fp32ArgOutputs outputs,
                                                                       uint64_t outer_size, uint64_t shards,
                                                                       uint64_t inner_size,
                                                                       ReductionShardRows shard_rows) {
    const uint64_t outputs_count = outer_size * inner_size;
    for (uint64_t output = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; output < outputs_count;
         output += static_cast<uint64_t>(gridDim.x) * blockDim.x) {
        const uint64_t outer_index = output / inner_size;
        const uint64_t component = output - outer_index * inner_size;
        const uint64_t* column = packed + outer_index * shards * inner_size + component;
        Fp32ArgCandidate best = fp32ArgIdentity();
        for (uint32_t shard = 0; shard < static_cast<uint32_t>(shards); ++shard)
            best = fp32ArgMin(best, fp32ArgWiden(column[static_cast<uint64_t>(shard) * inner_size],
                                                 reductionShardRowBegin(shard_rows, shard)));
        storeFp32ArgResult(outputs, output, best);
    }
}

// FP8's shard-split continuation (fp8ArgPackedSplitLayout: `columns` components x `groups` shard groups per block),
// for few outputs with many shards; the groups combine in shared memory with the same (key, tail) minimum.
constexpr uint32_t FP32_ARG_PACKED_SPLIT_MAX_THREADS = 1024;
template <int Unused>
__global__ __launch_bounds__(FP32_ARG_PACKED_SPLIT_MAX_THREADS) void fp32ArgPackedContinuationSplitKernel(
    const uint64_t* packed, Fp32ArgOutputs outputs, uint64_t outer_size, uint64_t shards, uint64_t inner_size,
    ReductionShardRows shard_rows, uint32_t columns, uint32_t groups) {
    __shared__ uint32_t partial_keys[FP32_ARG_PACKED_SPLIT_MAX_THREADS];
    __shared__ uint64_t partial_tails[FP32_ARG_PACKED_SPLIT_MAX_THREADS];
    const uint32_t lane_column = threadIdx.x % columns;
    const uint32_t group = threadIdx.x / columns;
    const uint32_t shard_count = static_cast<uint32_t>(shards);
    const uint64_t tiles = (inner_size + columns - 1) / columns;
    const uint64_t items = outer_size * tiles;
    for (uint64_t item = blockIdx.x; item < items; item += gridDim.x) {
        const uint64_t outer_index = item / tiles;
        const uint64_t component = (item - outer_index * tiles) * columns + lane_column;
        Fp32ArgCandidate best = fp32ArgIdentity();
        if (group < groups && component < inner_size) {
            const uint64_t* column = packed + outer_index * shards * inner_size + component;
            Fp32ArgCandidate other = fp32ArgIdentity();
            uint32_t shard = group;
            for (; shard + 3 * groups < shard_count; shard += 4 * groups) {
                const uint32_t s1 = shard + groups, s2 = s1 + groups, s3 = s2 + groups;
                const uint64_t c0 = column[static_cast<uint64_t>(shard) * inner_size];
                const uint64_t c1 = column[static_cast<uint64_t>(s1) * inner_size];
                const uint64_t c2 = column[static_cast<uint64_t>(s2) * inner_size];
                const uint64_t c3 = column[static_cast<uint64_t>(s3) * inner_size];
                best = fp32ArgMin(best, fp32ArgWiden(c0, reductionShardRowBegin(shard_rows, shard)));
                other = fp32ArgMin(other, fp32ArgWiden(c1, reductionShardRowBegin(shard_rows, s1)));
                best = fp32ArgMin(best, fp32ArgWiden(c2, reductionShardRowBegin(shard_rows, s2)));
                other = fp32ArgMin(other, fp32ArgWiden(c3, reductionShardRowBegin(shard_rows, s3)));
            }
            for (; shard < shard_count; shard += groups)
                best = fp32ArgMin(best, fp32ArgWiden(column[static_cast<uint64_t>(shard) * inner_size],
                                                     reductionShardRowBegin(shard_rows, shard)));
            best = fp32ArgMin(best, other);
        }
        partial_keys[threadIdx.x] = best.key;
        partial_tails[threadIdx.x] = best.tail;
        __syncthreads();
        for (uint32_t active = groups; active > 1;) {
            const uint32_t half = (active + 1) / 2;
            if (group < active - half) {
                const uint32_t other = threadIdx.x + half * columns;
                const Fp32ArgCandidate merged = fp32ArgMin({partial_keys[threadIdx.x], partial_tails[threadIdx.x]},
                                                           {partial_keys[other], partial_tails[other]});
                partial_keys[threadIdx.x] = merged.key;
                partial_tails[threadIdx.x] = merged.tail;
            }
            __syncthreads();
            active = half;
        }
        if (group == 0 && component < inner_size)
            storeFp32ArgResult(outputs, outer_index * inner_size + component,
                               Fp32ArgCandidate{partial_keys[threadIdx.x], partial_tails[threadIdx.x]});
        __syncthreads();  // the partials are rewritten by the next item
    }
}

// ---- The carried-index AlignedK ARG twin (see Fp32ArgWidePartials): alignedKArgKernel's ownership,
// balanced rows and load-ahead batches, reading the carried index packet alongside each value packet. Complete (one
// shard) writes the requested outputs (index = the composed index); Staged writes wide partials that
// fp32ArgWideContinuationKernel finishes in the same pass.
template <int ItemsPerLane, typename IndexT>
__device__ __forceinline__ void fp32AlignedKCarriedArgKeyPacket(uint32_t (&best_key)[ItemsPerLane],
                                                                uint64_t (&best_tail)[ItemsPerLane],
                                                                const PackedInputValues<float, ItemsPerLane>& packet,
                                                                const PackedInputValues<IndexT, ItemsPerLane>& carried,
                                                                uint64_t row_index, uint32_t xor_mask) {
#pragma unroll
    for (int item = 0; item < ItemsPerLane; ++item) {
        uint32_t negative_zero;
        const uint32_t key = fp32ArgKey(__float_as_uint(packet.values[item]), xor_mask, negative_zero);
        const uint64_t composed = static_cast<uint64_t>(carried.values[item]) + row_index;
        fp32ArgKeepMin(best_key[item], best_tail[item], key, (composed << 1) | negative_zero);
    }
}

template <int ItemsPerLane, int BlockThreads, bool Staged, typename IndexT>
__global__ __launch_bounds__(BlockThreads) void alignedKCarriedArgKernel(const float* input,
                                                                         const IndexT* carried,
                                                                         Fp32ArgOutputs outputs,
                                                                         Fp32ArgWidePartials partials,
                                                                         uint64_t outer_size,
                                                                         uint64_t reduction_size,
                                                                         uint64_t inner_size,
                                                                         uint64_t shards_per_output,
                                                                         uint64_t component_tiles,
                                                                         ReductionShardRows shard_rows,
                                                                         uint64_t row_stride) {
    static_assert(ItemsPerLane == 2 || ItemsPerLane == 4);
    static_assert(BlockThreads == 32 || BlockThreads == 64 || BlockThreads == 128);
    static_assert(std::is_same_v<IndexT, uint32_t> || std::is_same_v<IndexT, uint64_t>);
    constexpr uint64_t ComponentsPerCta = static_cast<uint64_t>(BlockThreads) * ItemsPerLane;
    constexpr uint32_t Rows = FP32_ALIGNED_K_LOAD_AHEAD_ROWS<ItemsPerLane>;
    using PacketT = PackedInputValues<float, ItemsPerLane>;
    using IndexPacketT = PackedInputValues<IndexT, ItemsPerLane>;
    const uint32_t outer_count = static_cast<uint32_t>(outer_size);
    const uint32_t tile_count = static_cast<uint32_t>(component_tiles);
    const uint32_t shard_count = static_cast<uint32_t>(shards_per_output);
    for (uint32_t outer_index = blockIdx.z; outer_index < outer_count; outer_index += gridDim.z)
    for (uint32_t component_tile = blockIdx.y; component_tile < tile_count; component_tile += gridDim.y)
    for (uint32_t shard = blockIdx.x; shard < shard_count; shard += gridDim.x) {
        const uint64_t packet_component_begin = static_cast<uint64_t>(component_tile) * ComponentsPerCta +
                                                static_cast<uint64_t>(threadIdx.x) * ItemsPerLane;
        if (packet_component_begin >= inner_size) continue;  // K % ItemsPerLane == 0: owned packets are whole
        const uint64_t row_begin = reductionShardRowBegin(shard_rows, shard);
        const uint32_t row_count = reductionShardRowCount(shard_rows, shard);
        const uint64_t element_begin = (outer_index * reduction_size + row_begin) * inner_size + packet_component_begin;
        const float* row_input = input + element_begin;
        const IndexT* row_carried = carried + element_begin;
        uint64_t row_index = row_begin * row_stride;  // the row's contribution to the composed index
        uint32_t best_key[ItemsPerLane];
        uint64_t best_tail[ItemsPerLane];
#pragma unroll
        for (int item = 0; item < ItemsPerLane; ++item) {
            best_key[item] = FP32_ARG_KEY_IDENTITY;
            best_tail[item] = FP32_ARG_TAIL_IDENTITY;
        }
        uint32_t row = 0;
        if (row_count >= Rows) {
            PacketT current[Rows];
            IndexPacketT current_index[Rows];
#pragma unroll
            for (uint32_t r = 0; r < Rows; ++r) {
                current[r] = loadVectorizedInputPacket<float, ItemsPerLane>(row_input + r * inner_size);
                current_index[r] = loadVectorizedInputPacket<IndexT, ItemsPerLane>(row_carried + r * inner_size);
            }
            row_input += static_cast<uint64_t>(Rows) * inner_size;
            row_carried += static_cast<uint64_t>(Rows) * inner_size;
            while (row + 2 * Rows <= row_count) {
                PacketT next[Rows];
                IndexPacketT next_index[Rows];
#pragma unroll
                for (uint32_t r = 0; r < Rows; ++r) {
                    next[r] = loadVectorizedInputPacket<float, ItemsPerLane>(row_input + r * inner_size);
                    next_index[r] = loadVectorizedInputPacket<IndexT, ItemsPerLane>(row_carried + r * inner_size);
                }
                row_input += static_cast<uint64_t>(Rows) * inner_size;
                row_carried += static_cast<uint64_t>(Rows) * inner_size;
#pragma unroll
                for (uint32_t r = 0; r < Rows; ++r) {
                    fp32AlignedKCarriedArgKeyPacket(best_key, best_tail, current[r], current_index[r], row_index,
                                                    outputs.xor_mask);
                    row_index += row_stride;
                }
                row += Rows;
#pragma unroll
                for (uint32_t r = 0; r < Rows; ++r) {
                    current[r] = next[r];
                    current_index[r] = next_index[r];
                }
            }
#pragma unroll
            for (uint32_t r = 0; r < Rows; ++r) {
                fp32AlignedKCarriedArgKeyPacket(best_key, best_tail, current[r], current_index[r], row_index,
                                                outputs.xor_mask);
                row_index += row_stride;
            }
            row += Rows;
        }
        for (; row < row_count; ++row, row_input += inner_size, row_carried += inner_size, row_index += row_stride) {
            fp32AlignedKCarriedArgKeyPacket(best_key, best_tail,
                                            loadVectorizedInputPacket<float, ItemsPerLane>(row_input),
                                            loadVectorizedInputPacket<IndexT, ItemsPerLane>(row_carried), row_index,
                                            outputs.xor_mask);
        }
        if constexpr (Staged) {
            const uint64_t partial =
                (static_cast<uint64_t>(outer_index) * shards_per_output + shard) * inner_size + packet_component_begin;
#pragma unroll
            for (int item = 0; item < ItemsPerLane; ++item) {
                partials.keys[partial + item] = best_key[item];
                partials.tails[partial + item] = best_tail[item];
            }
        } else {
            static_cast<void>(partials);
#pragma unroll
            for (int item = 0; item < ItemsPerLane; ++item)
                storeFp32ArgResult(outputs, static_cast<uint64_t>(outer_index) * inner_size + packet_component_begin +
                                                item,
                                   Fp32ArgCandidate{best_key[item], best_tail[item]});
        }
    }
}

// Reduces wide partials [outer, shards, K] over shards and writes the requested outputs (the tails hold the composed
// index): the packed continuation's per-output form.
template <int Unused>
__global__ __launch_bounds__(256) void fp32ArgWideContinuationKernel(Fp32ArgWidePartials partials,
                                                                     Fp32ArgOutputs outputs, uint64_t outer_size,
                                                                     uint64_t shards, uint64_t inner_size) {
    const uint64_t outputs_count = outer_size * inner_size;
    for (uint64_t output = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; output < outputs_count;
         output += static_cast<uint64_t>(gridDim.x) * blockDim.x) {
        const uint64_t outer_index = output / inner_size;
        const uint64_t component = output - outer_index * inner_size;
        const uint64_t column = outer_index * shards * inner_size + component;
        uint32_t best_key = FP32_ARG_KEY_IDENTITY;
        uint64_t best_tail = FP32_ARG_TAIL_IDENTITY;
        for (uint32_t shard = 0; shard < static_cast<uint32_t>(shards); ++shard) {
            const uint64_t partial = column + static_cast<uint64_t>(shard) * inner_size;
            fp32ArgKeepMin(best_key, best_tail, partials.keys[partial], partials.tails[partial]);
        }
        storeFp32ArgResult(outputs, output, Fp32ArgCandidate{best_key, best_tail});
    }
}

// The shard-split form (fp8ArgPackedSplitLayout), for few outputs with many shards.
template <int Unused>
__global__ __launch_bounds__(FP32_ARG_PACKED_SPLIT_MAX_THREADS) void fp32ArgWideContinuationSplitKernel(
    Fp32ArgWidePartials partials, Fp32ArgOutputs outputs, uint64_t outer_size, uint64_t shards, uint64_t inner_size,
    uint32_t columns, uint32_t groups) {
    __shared__ uint32_t partial_keys[FP32_ARG_PACKED_SPLIT_MAX_THREADS];
    __shared__ uint64_t partial_tails[FP32_ARG_PACKED_SPLIT_MAX_THREADS];
    const uint32_t lane_column = threadIdx.x % columns;
    const uint32_t group = threadIdx.x / columns;
    const uint32_t shard_count = static_cast<uint32_t>(shards);
    const uint64_t tiles = (inner_size + columns - 1) / columns;
    const uint64_t items = outer_size * tiles;
    for (uint64_t item = blockIdx.x; item < items; item += gridDim.x) {
        const uint64_t outer_index = item / tiles;
        const uint64_t component = (item - outer_index * tiles) * columns + lane_column;
        uint32_t best_key = FP32_ARG_KEY_IDENTITY, other_key = FP32_ARG_KEY_IDENTITY;
        uint64_t best_tail = FP32_ARG_TAIL_IDENTITY, other_tail = FP32_ARG_TAIL_IDENTITY;
        if (group < groups && component < inner_size) {
            const uint64_t column = outer_index * shards * inner_size + component;
            uint32_t shard = group;
            for (; shard + groups < shard_count; shard += 2 * groups) {
                const uint64_t p0 = column + static_cast<uint64_t>(shard) * inner_size;
                const uint64_t p1 = p0 + static_cast<uint64_t>(groups) * inner_size;
                fp32ArgKeepMin(best_key, best_tail, partials.keys[p0], partials.tails[p0]);
                fp32ArgKeepMin(other_key, other_tail, partials.keys[p1], partials.tails[p1]);
            }
            if (shard < shard_count) {
                const uint64_t p0 = column + static_cast<uint64_t>(shard) * inner_size;
                fp32ArgKeepMin(best_key, best_tail, partials.keys[p0], partials.tails[p0]);
            }
            fp32ArgKeepMin(best_key, best_tail, other_key, other_tail);
        }
        partial_keys[threadIdx.x] = best_key;
        partial_tails[threadIdx.x] = best_tail;
        __syncthreads();
        for (uint32_t active = groups; active > 1;) {
            const uint32_t half = (active + 1) / 2;
            if (group < active - half) {
                const uint32_t other = threadIdx.x + half * columns;
                uint32_t key = partial_keys[threadIdx.x];
                uint64_t tail = partial_tails[threadIdx.x];
                fp32ArgKeepMin(key, tail, partial_keys[other], partial_tails[other]);
                partial_keys[threadIdx.x] = key;
                partial_tails[threadIdx.x] = tail;
            }
            __syncthreads();
            active = half;
        }
        if (group == 0 && component < inner_size)
            storeFp32ArgResult(outputs, outer_index * inner_size + component,
                               Fp32ArgCandidate{partial_keys[threadIdx.x], partial_tails[threadIdx.x]});
        __syncthreads();  // the partials are rewritten by the next item
    }
}

// The FP8 AlignedK argmin/argmax twin: the K-parallel ownership of fp8AlignedKValue{,StagedUnified,
// CompleteUnified}Kernel (one thread owns one aligned 2- or 8-byte packet of components and walks its shard's rows; no
// cross-thread reduction; grid x = shard, y = K tile, z = outer; balanced shard rows). The CTA size is a launch
// operand, so two kernels serve VALUE's six packet/CTA pairs. One shard is Complete (requested outputs); several write
// FP32 value + carried index partials. The index is (shard row base + row) * stride.
// VALUE's R loop and writeback. Rows are loaded in batches one batch ahead (P8: 4 rows, P2: 8 rows,
// FP8_ALIGNED_K_LOAD_AHEAD_ROWS); eight-byte packets key two words per row, two-byte packets pack two rows per key
// word. Results leave as 16-byte (P8, through a per-warp shared transpose, as VALUE's P8 writeback) or 8-byte (P2)
// lane-contiguous stores (one scalar per component, 8 bytes apart per lane, measured 1.5x VALUE, 4.4x at R = 4).
constexpr int FP8_ARG_ALIGNED_MAX_THREADS = 128;
// VALUE's AlignedK load-ahead depth (as FP8_ALIGNED_K_LOAD_AHEAD_ROWS above, which only FP8 builds define).
template <int PacketBytes>
inline constexpr uint32_t FP8_ALIGNED_K_ARG_LOAD_AHEAD_ROWS = PacketBytes >= 8 ? 4U : 8U;
constexpr int FP8_ARG_ALIGNED_STRIPE_PITCH = 33;

// Stores N (2 or 4) consecutive results starting at output_index: one vector store per output when the storage is
// FP32 values / UINT32 or UINT64 indices and the address is aligned, element stores otherwise.
template <int N>
__device__ __forceinline__ void storeFp8ArgResults(const Fp8ArgOutputs& outputs,
                                                   uint64_t output_index,
                                                   const uint32_t (&candidates)[N],
                                                   uint64_t row_base) {
    static_assert(N == 2 || N == 4);
    if (outputs.packed != nullptr) {
        uint32_t* destination = outputs.packed + output_index;
        if ((reinterpret_cast<uintptr_t>(destination) & uintptr_t{N * sizeof(uint32_t) - 1}) == 0) {
            if constexpr (N == 4) {
                *reinterpret_cast<uint4*>(destination) =
                    make_uint4(candidates[0], candidates[1], candidates[2], candidates[3]);
            } else {
                *reinterpret_cast<uint2*>(destination) = make_uint2(candidates[0], candidates[1]);
            }
        } else {
#pragma unroll
            for (int i = 0; i < N; ++i) destination[i] = candidates[i];
        }
        return;
    }
    if (outputs.value != nullptr) {
        float values[N];
#pragma unroll
        for (int i = 0; i < N; ++i) {
            values[i] = decodeFp8ArgByte(fp8ArgValueByte(candidates[i] >> 24, candidates[i] & 1u, outputs.fold),
                                         outputs.e5m2);
        }
        float* destination = static_cast<float*>(outputs.value) + output_index;
        if (outputs.value_dtype == DataType::FP32 &&
            (reinterpret_cast<uintptr_t>(destination) & uintptr_t{N * sizeof(float) - 1}) == 0) {
            if constexpr (N == 4) {
                *reinterpret_cast<float4*>(destination) = make_float4(values[0], values[1], values[2], values[3]);
            } else {
                *reinterpret_cast<float2*>(destination) = make_float2(values[0], values[1]);
            }
        } else {
#pragma unroll
            for (int i = 0; i < N; ++i) {
                storeFp32AsRuntimeDType(outputs.value, outputs.value_dtype, output_index + i, values[i]);
            }
        }
    }
    if (outputs.index != nullptr) {
        uint64_t indices[N];
#pragma unroll
        for (int i = 0; i < N; ++i) {
            indices[i] = (row_base + ((candidates[i] >> 1) & (FP8_ARG_MAX_LANE_ROWS - 1u))) * outputs.domain_stride;
        }
        if (outputs.index_dtype == DataType::UINT32) {
            uint32_t* destination = static_cast<uint32_t*>(outputs.index) + output_index;
            if ((reinterpret_cast<uintptr_t>(destination) & uintptr_t{N * sizeof(uint32_t) - 1}) == 0) {
                if constexpr (N == 4) {
                    *reinterpret_cast<uint4*>(destination) =
                        make_uint4(static_cast<uint32_t>(indices[0]), static_cast<uint32_t>(indices[1]),
                                   static_cast<uint32_t>(indices[2]), static_cast<uint32_t>(indices[3]));
                } else {
                    *reinterpret_cast<uint2*>(destination) =
                        make_uint2(static_cast<uint32_t>(indices[0]), static_cast<uint32_t>(indices[1]));
                }
            } else {
#pragma unroll
                for (int i = 0; i < N; ++i) destination[i] = static_cast<uint32_t>(indices[i]);
            }
        } else {
            uint64_t* destination = static_cast<uint64_t*>(outputs.index) + output_index;
            if ((reinterpret_cast<uintptr_t>(destination) & uintptr_t{15}) == 0) {
#pragma unroll
                for (int i = 0; i < N; i += 2) {
                    *reinterpret_cast<uint4*>(destination + i) =
                        make_uint4(static_cast<uint32_t>(indices[i]), static_cast<uint32_t>(indices[i] >> 32),
                                   static_cast<uint32_t>(indices[i + 1]), static_cast<uint32_t>(indices[i + 1] >> 32));
                }
            } else {
#pragma unroll
                for (int i = 0; i < N; ++i) destination[i] = indices[i];
            }
        }
    }
}

template <int PacketBytes>
__global__ __launch_bounds__(FP8_ARG_ALIGNED_MAX_THREADS) void fp8AlignedKArgKernel(const uint8_t* input,
                                                                                   Fp8ArgOutputs outputs,
                                                                                   uint64_t outer_size,
                                                                                   uint64_t reduction_size,
                                                                                   uint64_t inner_size,
                                                                                   uint64_t shards_per_output,
                                                                                   uint64_t component_tiles,
                                                                                   ReductionShardRows shard_rows) {
    static_assert(PacketBytes == 2 || PacketBytes == 8);
    constexpr uint32_t Identity = FP8_ARG_LANE_IDENTITY;
    // VALUE's load-ahead depth; a P2 batch is an even number of rows (two rows per key word).
    constexpr uint32_t Rows = FP8_ALIGNED_K_ARG_LOAD_AHEAD_ROWS<PacketBytes>;
    static_assert(PacketBytes == 8 || Rows % 2 == 0);
    using Word = std::conditional_t<PacketBytes == 8, uint2, uint32_t>;
    // P8 writeback transpose: warp_tiles[warp][item][lane] holds lane's candidate for component lane * 8 + item.
    __shared__ uint32_t warp_tiles[FP8_ARG_ALIGNED_MAX_THREADS / TILED_REDUCTION_WARP_THREADS][PacketBytes]
                                  [FP8_ARG_ALIGNED_STRIPE_PITCH];
    const uint32_t warp = threadIdx.x / TILED_REDUCTION_WARP_THREADS;
    const uint32_t lane = threadIdx.x % TILED_REDUCTION_WARP_THREADS;
    const uint64_t components_per_cta = static_cast<uint64_t>(blockDim.x) * PacketBytes;
    const uint32_t outer_count = static_cast<uint32_t>(outer_size);
    const uint32_t tile_count = static_cast<uint32_t>(component_tiles);
    const uint32_t shard_count = static_cast<uint32_t>(shards_per_output);
    for (uint32_t outer_index = blockIdx.z; outer_index < outer_count; outer_index += gridDim.z)
    for (uint32_t component_tile = blockIdx.y; component_tile < tile_count; component_tile += gridDim.y)
    for (uint32_t shard = blockIdx.x; shard < shard_count; shard += gridDim.x) {
        const uint64_t tile_begin = component_tile * components_per_cta;
        const uint64_t component = tile_begin + static_cast<uint64_t>(threadIdx.x) * PacketBytes;
        const bool owns_packet = component < inner_size;
        const uint64_t row_begin = reductionShardRowBegin(shard_rows, shard);
        const uint32_t row_count = reductionShardRowCount(shard_rows, shard);
        uint32_t local[PacketBytes];
#pragma unroll
        for (int item = 0; item < PacketBytes; ++item) local[item] = Identity;
        if (owns_packet) {
            const uint8_t* packet = input + (outer_index * reduction_size + row_begin) * inner_size + component;
            const auto load = [&](const uint8_t* at) -> Word {
                if constexpr (PacketBytes == 8) {
                    return reinterpret_cast<const RawVectorPacket<8>*>(at)->value;
                } else {
                    return reinterpret_cast<const RawVectorPacket<2>*>(at)->value;
                }
            };
            const auto key_batch = [&](const Word (&batch)[Rows], uint32_t first_row) {
                if constexpr (PacketBytes == 8) {
#pragma unroll
                    for (uint32_t r = 0; r < Rows; ++r) {
                        const Fp8ArgKeys4 low = fp8ArgKeys4(batch[r].x, outputs.fold);
                        const Fp8ArgKeys4 high = fp8ArgKeys4(batch[r].y, outputs.fold);
#pragma unroll
                        for (uint32_t b = 0; b < 4U; ++b) {
                            local[b] = fp8ArgMin(local[b], fp8ArgLaneCandidate(low, b, first_row + r));
                            local[4 + b] = fp8ArgMin(local[4 + b], fp8ArgLaneCandidate(high, b, first_row + r));
                        }
                    }
                } else {
#pragma unroll
                    for (uint32_t r = 0; r < Rows; r += 2) {
                        const Fp8ArgKeys4 keys = fp8ArgKeys4(batch[r] | (batch[r + 1] << 16), outputs.fold);
                        local[0] = fp8ArgMin(local[0], fp8ArgLaneCandidate(keys, 0, first_row + r));
                        local[1] = fp8ArgMin(local[1], fp8ArgLaneCandidate(keys, 1, first_row + r));
                        local[0] = fp8ArgMin(local[0], fp8ArgLaneCandidate(keys, 2, first_row + r + 1));
                        local[1] = fp8ArgMin(local[1], fp8ArgLaneCandidate(keys, 3, first_row + r + 1));
                    }
                }
            };
            const auto load_batch = [&](Word (&batch)[Rows]) {
#pragma unroll
                for (uint32_t r = 0; r < Rows; ++r) batch[r] = load(packet + r * inner_size);
                packet += Rows * inner_size;
            };
            // Load-ahead as accumulateFp8AlignedKRows: the next batch is in flight while the current one is keyed.
            uint32_t row = 0;
            if (row_count >= Rows) {
                Word current[Rows];
                load_batch(current);
                for (row = Rows; row + Rows <= row_count; row += Rows) {
                    Word next[Rows];
                    load_batch(next);
                    key_batch(current, row - Rows);
#pragma unroll
                    for (uint32_t r = 0; r < Rows; ++r) current[r] = next[r];
                }
                key_batch(current, row - Rows);
            }
            for (; row < row_count; ++row) {
                const Word word = load(packet);
                packet += inner_size;
                if constexpr (PacketBytes == 8) {
                    const Fp8ArgKeys4 low = fp8ArgKeys4(word.x, outputs.fold);
                    const Fp8ArgKeys4 high = fp8ArgKeys4(word.y, outputs.fold);
#pragma unroll
                    for (uint32_t b = 0; b < 4U; ++b) {
                        local[b] = fp8ArgMin(local[b], fp8ArgLaneCandidate(low, b, row));
                        local[4 + b] = fp8ArgMin(local[4 + b], fp8ArgLaneCandidate(high, b, row));
                    }
                } else {
                    const Fp8ArgKeys4 keys = fp8ArgKeys4(word, outputs.fold);
                    local[0] = fp8ArgMin(local[0], fp8ArgLaneCandidate(keys, 0, row));
                    local[1] = fp8ArgMin(local[1], fp8ArgLaneCandidate(keys, 1, row));
                }
            }
        }
        const uint64_t output_row = (static_cast<uint64_t>(outer_index) * shards_per_output + shard) * inner_size;
        if constexpr (PacketBytes == 2) {
            if (owns_packet) storeFp8ArgResults<2>(outputs, output_row + component, local, row_begin);
        } else {
            // K is a multiple of 8, so a warp's valid components are whole packets and whole 4-component chunks.
#pragma unroll
            for (int item = 0; item < PacketBytes; ++item) warp_tiles[warp][item][lane] = local[item];
            __syncwarp();
            const uint64_t warp_begin = tile_begin + static_cast<uint64_t>(warp) * TILED_REDUCTION_WARP_THREADS *
                                                         PacketBytes;
#pragma unroll
            for (uint32_t round = 0; round < 2U; ++round) {
                const uint32_t offset = round * TILED_REDUCTION_WARP_THREADS * 4U + lane * 4U;
                if (warp_begin + offset < inner_size) {
                    uint32_t chunk[4];
#pragma unroll
                    for (uint32_t i = 0; i < 4U; ++i) {
                        const uint32_t c = offset + i;
                        chunk[i] = warp_tiles[warp][c % PacketBytes][c / PacketBytes];
                    }
                    storeFp8ArgResults<4>(outputs, output_row + warp_begin + offset, chunk, row_begin);
                }
            }
            __syncwarp();
        }
    }
}

// Reduces packed staged partials [outer, shards, K] (shard-relative lane candidates) over shards and writes the
// requested outputs. Shard s begins at s * base_rows + min(s, extra_rows) (uniform depths: extra_rows = 0).
// One thread per output component, consecutive threads on consecutive components (coalesced 4-byte reads per shard).
// Used while there are enough outputs to cover the shard loop's latency (fp8ArgPackedContinuationSplitsShards).
template <int Unused>
__global__ __launch_bounds__(256) void fp8ArgPackedContinuationKernel(const uint32_t* packed,
                                                                     Fp8ArgOutputs outputs,
                                                                     uint64_t outer_size,
                                                                     uint64_t shards,
                                                                     uint64_t inner_size,
                                                                     ReductionShardRows shard_rows) {
    const uint64_t outputs_count = outer_size * inner_size;
    for (uint64_t output = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; output < outputs_count;
         output += static_cast<uint64_t>(gridDim.x) * blockDim.x) {
        const uint64_t outer_index = output / inner_size;
        const uint64_t component = output - outer_index * inner_size;
        const uint32_t* column = packed + outer_index * shards * inner_size + component;
        uint64_t best = FP8_ARG_WIDE_IDENTITY;
        for (uint32_t shard = 0; shard < static_cast<uint32_t>(shards); ++shard) {
            const uint32_t candidate = column[static_cast<uint64_t>(shard) * inner_size];
            best = fp8ArgMin(best, fp8ArgWiden(candidate, reductionShardRowBegin(shard_rows, shard)));
        }
        storeFp8ArgPackedResult(outputs, output, best);
    }
}

// The shard-split packed continuation, for few outputs with many shards (the per-output kernel ran 6-16x a copy of its
// partials at >= 256 shards and < 64 Ki outputs, up to half of the ARG time). A block owns one (outer, column tile)
// work item: `columns` consecutive components (K when K < 32, else 32) times `groups` shard groups; group g walks
// shards g, g + groups, ... with four independent loads in flight. For K < 32 a block's threads cover a contiguous
// [shards, K] slab, so reads stay coalesced. The groups combine in shared memory with the same order-free wide minimum
// (key, then index, then -0), so the result equals the per-output kernel's.
constexpr uint32_t FP8_ARG_PACKED_SPLIT_MAX_THREADS = 1024;
template <int Unused>
__global__ __launch_bounds__(FP8_ARG_PACKED_SPLIT_MAX_THREADS) void fp8ArgPackedContinuationSplitKernel(
    const uint32_t* packed,
    Fp8ArgOutputs outputs,
    uint64_t outer_size,
    uint64_t shards,
    uint64_t inner_size,
    ReductionShardRows shard_rows,
    uint32_t columns,
    uint32_t groups) {
    __shared__ uint64_t partial[FP8_ARG_PACKED_SPLIT_MAX_THREADS];
    const uint32_t lane_column = threadIdx.x % columns;
    const uint32_t group = threadIdx.x / columns;
    const uint32_t shard_count = static_cast<uint32_t>(shards);
    const uint64_t tiles = (inner_size + columns - 1) / columns;
    const uint64_t items = outer_size * tiles;
    for (uint64_t item = blockIdx.x; item < items; item += gridDim.x) {
        const uint64_t outer_index = item / tiles;
        const uint64_t component = (item - outer_index * tiles) * columns + lane_column;
        uint64_t best = FP8_ARG_WIDE_IDENTITY;
        if (group < groups && component < inner_size) {
            const uint32_t* column = packed + outer_index * shards * inner_size + component;
            uint64_t other = FP8_ARG_WIDE_IDENTITY;
            uint32_t shard = group;
            for (; shard + 3 * groups < shard_count; shard += 4 * groups) {
                const uint32_t s1 = shard + groups, s2 = s1 + groups, s3 = s2 + groups;
                const uint32_t c0 = column[static_cast<uint64_t>(shard) * inner_size];
                const uint32_t c1 = column[static_cast<uint64_t>(s1) * inner_size];
                const uint32_t c2 = column[static_cast<uint64_t>(s2) * inner_size];
                const uint32_t c3 = column[static_cast<uint64_t>(s3) * inner_size];
                best = fp8ArgMin(best, fp8ArgWiden(c0, reductionShardRowBegin(shard_rows, shard)));
                other = fp8ArgMin(other, fp8ArgWiden(c1, reductionShardRowBegin(shard_rows, s1)));
                best = fp8ArgMin(best, fp8ArgWiden(c2, reductionShardRowBegin(shard_rows, s2)));
                other = fp8ArgMin(other, fp8ArgWiden(c3, reductionShardRowBegin(shard_rows, s3)));
            }
            for (; shard < shard_count; shard += groups) {
                best = fp8ArgMin(best, fp8ArgWiden(column[static_cast<uint64_t>(shard) * inner_size],
                                                   reductionShardRowBegin(shard_rows, shard)));
            }
            best = fp8ArgMin(best, other);
        }
        partial[threadIdx.x] = best;
        __syncthreads();
        for (uint32_t active = groups; active > 1;) {
            const uint32_t half = (active + 1) / 2;
            if (group < active - half)
                partial[threadIdx.x] = fp8ArgMin(partial[threadIdx.x], partial[threadIdx.x + half * columns]);
            __syncthreads();
            active = half;
        }
        if (group == 0 && component < inner_size)
            storeFp8ArgPackedResult(outputs, outer_index * inner_size + component, partial[threadIdx.x]);
        __syncthreads();  // partial[] is rewritten by the next item
    }
}

// The FP32 AlignedK configurations: 8- and 16-byte packets x CTA 32 / 64 / 128 x Staged / Complete, each one kernel.
// ScalarPacket (VALUE only) adds the 4-byte packet; the ARG twins have no scalar form.
// ElementBytes 2 is the FP16 / BF16 VALUE kernel's: 2- (ScalarPacket), 4-, 8- and 16-byte packets; and the FP16 / BF16
// ARG twin's (no ScalarPacket): 4-, 8- and 16-byte packets.
template <RKStageTopology Topology, bool ScalarPacket = false, uint32_t ElementBytes = 4, typename Callback>
[[nodiscard]] bool dispatchFp32AlignedKConfiguration(uint32_t packet_bytes, uint32_t block_threads,
                                                     Callback&& callback) {
    static_assert(ElementBytes == 4 || ElementBytes == 2);
    const auto for_items = [&]<int ItemsPerLane>() {
        switch (block_threads) {
            case 32: callback.template operator()<ItemsPerLane, 32>(); return true;
            case 64: callback.template operator()<ItemsPerLane, 64>(); return true;
            case 128: callback.template operator()<ItemsPerLane, 128>(); return true;
            default: return false;
        }
    };
    if constexpr (ElementBytes == 2) {
        switch (packet_bytes) {
            case 2:
                if constexpr (ScalarPacket) {
                    return for_items.template operator()<1>();
                } else {
                    return false;
                }
            case 4: return for_items.template operator()<2>();
            case 8: return for_items.template operator()<4>();
            case 16: return for_items.template operator()<8>();
            default: return false;
        }
    } else {
        switch (packet_bytes) {
            case 4:
                if constexpr (ScalarPacket) {
                    return for_items.template operator()<1>();
                } else {
                    return false;
                }
            case 8: return for_items.template operator()<2>();
            case 16: return for_items.template operator()<4>();
            default: return false;
        }
    }
}

#if THOR_CUB_ENABLE_FP8_TYPES

template <typename InputT, size_t PacketBytes, int BlockThreads, RKStageTopology Topology>
[[nodiscard]] KernelOccupancyInfo queryFp8AlignedKOccupancyForBlock() {
    static_assert(IS_FP8_ALIGNED_K_TYPE<InputT>);
    static_assert(PacketBytes == 2 || PacketBytes == 4 || PacketBytes == 8 || PacketBytes == 16);
    static_assert(BlockThreads == 32 || BlockThreads == 64 || BlockThreads == 128 || BlockThreads == 256);
    using InputTransformT = IdentityFp32;
    using ReductionOpT = cuda::std::plus<float>;
    using OutputFinalizeT = AdditiveFinalizeFp32;

    // Staged output is always FP32, so packet4/packet2 have an exact zero-shared occupancy query. Complete occupancy is
    // conservatively queried against FP32 writeback; packet8/packet16 therefore include the transpose footprint while
    // packet4/packet2 remain direct. Runtime Complete launches may use still less shared memory for narrower outputs.
    constexpr bool NeedsTranspose =
        Topology == RKStageTopology::Staged ? FP8_ALIGNED_K_STAGED_NEEDS_TRANSPOSE<PacketBytes>
                                                 : (PacketBytes * sizeof(float) > 16);
    constexpr size_t dynamic_shared_bytes =
        NeedsTranspose
            ? static_cast<size_t>(BlockThreads / TILED_REDUCTION_WARP_THREADS)
                  * FP8_ALIGNED_K_SHARED_BYTES_PER_WARP<PacketBytes>
            : 0;
    const auto kernel = [] {
        if constexpr (Topology == RKStageTopology::Staged
                      && (PacketBytes == 2 || PacketBytes == 8)
                      && (BlockThreads == 32 || BlockThreads == 64 || BlockThreads == 128)) {
            return fp8AlignedKValueStagedUnifiedKernel<InputT, PacketBytes, BlockThreads>;
        } else if constexpr (Topology == RKStageTopology::Complete
                             && PacketBytes == 2 && BlockThreads == 32) {
            return fp8AlignedKValueCompleteUnifiedKernel<InputT, 2, 32, Fp8CompleteFinalizePolicy::Joined>;
        } else if constexpr (Topology == RKStageTopology::Complete
                             && PacketBytes == 8 && BlockThreads == 32) {
            return fp8AlignedKValueCompleteUnifiedKernel<InputT, 8, 32, Fp8CompleteFinalizePolicy::InBranch>;
        } else {
            return fp8AlignedKValueKernel<InputT,
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
    return KernelOccupancyInfo{attributes.numRegs,
                               attributes.sharedSizeBytes,
                               dynamic_shared_bytes,
                               active_blocks,
                               active_blocks * (BlockThreads / TILED_REDUCTION_WARP_THREADS)};
}

template <typename InputT, RKStageTopology Topology>
[[nodiscard]] KernelOccupancyInfo queryFp8AlignedKOccupancy(size_t packet_bytes, uint32_t block_threads) {
    static_assert(IS_FP8_ALIGNED_K_TYPE<InputT>);
    KernelOccupancyInfo occupancy{};
    const bool queried = dispatchFp8AlignedKConfiguration<Topology>(
        packet_bytes, block_threads, [&]<size_t PacketBytes, int BlockThreads>() {
            occupancy = queryFp8AlignedKOccupancyForBlock<InputT, PacketBytes, BlockThreads, Topology>();
        });
    if (!queried) {
        // Not a compiled AlignedK configuration; the selector treats this as absent.
        throw std::logic_error("FP8 AlignedK occupancy requires a retained packet/CTA configuration.");
    }
    return occupancy;
}

#endif  // THOR_CUB_ENABLE_FP8_TYPES

#if THOR_CUB_ENABLE_FP8_TYPES

template <typename InputT,
          typename InputTransformT,
          typename ReductionOpT,
          typename OutputFinalizeT,
          size_t PacketBytes,
          int BlockThreads,
          RKStageTopology Topology>
void launchFp8AlignedKForBlock(const StridedSpan& input,
                               const StridedSpan& stage_output,
                               const ReductionGeometry& geometry,
                               uint64_t shards_per_output,
                               InputTransformT input_transform,
                               OutputFinalizeT output_finalize,
                               float output_scale,
                               Stream& stream) {
    static_assert(IS_FP8_ALIGNED_K_TYPE<InputT>);
    static_assert(PacketBytes == 2 || PacketBytes == 4 || PacketBytes == 8 || PacketBytes == 16);
    static_assert(BlockThreads == 32 || BlockThreads == 64 || BlockThreads == 128 || BlockThreads == 256);
    constexpr int ItemsPerLane = FP8_ALIGNED_K_ITEMS_PER_LANE<PacketBytes>;
    constexpr uint64_t components_per_cta =
        static_cast<uint64_t>(BlockThreads) * static_cast<uint64_t>(ItemsPerLane);
    constexpr size_t max_dynamic_shared_bytes =
        static_cast<size_t>(BlockThreads / TILED_REDUCTION_WARP_THREADS)
        * FP8_ALIGNED_K_SHARED_BYTES_PER_WARP<PacketBytes>;

    const bool needs_transpose = Topology == RKStageTopology::Staged
                                     ? FP8_ALIGNED_K_STAGED_NEEDS_TRANSPOSE<PacketBytes>
                                     : fp8AlignedKCompleteNeedsTranspose<PacketBytes>(stage_output.getDataType());
    const size_t dynamic_shared_bytes = needs_transpose ? max_dynamic_shared_bytes : 0;

    // Division-free launch: x = shard, y = component tile, z = outer. The balanced row split is computed here once
    // instead of by every block, and every shard's row count must fit the kernels' 32-bit row counter.
    const uint64_t component_tiles = ceilDivideU64(geometry.inner_size, components_per_cta);
    if (geometry.outer_size == 0 || component_tiles == 0 || shards_per_output == 0) {
        throw std::logic_error("FP8 AlignedK stage requires non-zero work.");
    }
    const ReductionShardRows shard_rows = balancedReductionShardRows(geometry.reduction_size, shards_per_output);
    // Complete items of <= 8 rows and <= 512 bytes keep a persistent grid (see REDUCTION_PERSISTENT_MAX_BLOCKS).
    const uint64_t rows_per_work_item =
        static_cast<uint64_t>(shard_rows.base_rows) + static_cast<uint64_t>(shard_rows.extra_rows != 0);
    const uint64_t bytes_per_work_item =
        rows_per_work_item * std::min<uint64_t>(geometry.inner_size, components_per_cta) * sizeof(InputT);
    const bool persistent = Topology == RKStageTopology::Complete && rows_per_work_item <= 8
                            && bytes_per_work_item <= 512;
    const dim3 grid_blocks = reductionWorkGrid(shards_per_output, component_tiles, geometry.outer_size,
                                               persistent ? REDUCTION_PERSISTENT_MAX_BLOCKS : 0);

    // Production Staged AlignedK: remove the operation dimension from the six P2/P8 x CTA32/64/128 kernels.
    // Compile-time routing is important: a runtime pre-dispatch alone would still instantiate all four per-operation
    // kernels.
    if constexpr (Topology == RKStageTopology::Staged
                  && std::is_same_v<InputTransformT, IdentityFp32>
                  && (PacketBytes == 2 || PacketBytes == 8)
                  && (BlockThreads == 32 || BlockThreads == 64 || BlockThreads == 128)
                  && (std::is_same_v<ReductionOpT, cuda::std::plus<float>>
                      || std::is_same_v<ReductionOpT, cuda::std::multiplies<float>>
                      || std::is_same_v<ReductionOpT, PropagatingMinimumFp32>
                      || std::is_same_v<ReductionOpT, PropagatingMaximumFp32>)) {
        constexpr ReductionStageCombine operation = [] {
            if constexpr (std::is_same_v<ReductionOpT, cuda::std::plus<float>>) {
                return ReductionStageCombine::Sum;
            } else if constexpr (std::is_same_v<ReductionOpT, cuda::std::multiplies<float>>) {
                return ReductionStageCombine::Product;
            } else if constexpr (std::is_same_v<ReductionOpT, PropagatingMinimumFp32>) {
                return ReductionStageCombine::Minimum;
            } else {
                return ReductionStageCombine::Maximum;
            }
        }();
        fp8AlignedKValueStagedUnifiedKernel<InputT, PacketBytes, BlockThreads>
            <<<grid_blocks, BlockThreads, dynamic_shared_bytes, stream.getStream()>>>(
                input.getMemPtr<InputT>(), stage_output.getMemPtr<float>(), geometry.outer_size,
                geometry.reduction_size, geometry.inner_size, shards_per_output, component_tiles, shard_rows,
                operation);
        CUDA_CHECK(cudaGetLastError());
    } else if constexpr (Topology == RKStageTopology::Complete
                         && std::is_same_v<InputTransformT, IdentityFp32>
                         && ((PacketBytes == 2 && BlockThreads == 32)
                             || (PacketBytes == 8 && (BlockThreads == 32 || BlockThreads == 64)))
                         && (std::is_same_v<ReductionOpT, cuda::std::multiplies<float>>
                             || std::is_same_v<ReductionOpT, PropagatingMinimumFp32>
                             || std::is_same_v<ReductionOpT, PropagatingMaximumFp32>
                             || (std::is_same_v<ReductionOpT, cuda::std::plus<float>>
                                 && !(PacketBytes == 8 && BlockThreads == 64)))) {
        // Compile-time routing prevents instantiation of the pruned typed Identity Complete operations at these
        // physical configurations.
        // P8/CTA64 SUM/MEAN deliberately remain on the original typed kernel.
        constexpr ReductionStageCombine operation = [] {
            if constexpr (std::is_same_v<ReductionOpT, cuda::std::plus<float>>)
                return ReductionStageCombine::Sum;
            else if constexpr (std::is_same_v<ReductionOpT, cuda::std::multiplies<float>>)
                return ReductionStageCombine::Product;
            else if constexpr (std::is_same_v<ReductionOpT, PropagatingMinimumFp32>)
                return ReductionStageCombine::Minimum;
            else
                return ReductionStageCombine::Maximum;
        }();
        constexpr Fp8CompleteFinalizePolicy policy = PacketBytes == 2
            ? Fp8CompleteFinalizePolicy::Joined : Fp8CompleteFinalizePolicy::InBranch;
        const AdditiveFinalizeFp32 additive_finalize = [&] {
            if constexpr (std::is_same_v<ReductionOpT, cuda::std::plus<float>>)
                return output_finalize;
            else
                return AdditiveFinalizeFp32{1.0f, false};
        }();
        fp8AlignedKValueCompleteUnifiedKernel<InputT, PacketBytes, BlockThreads, policy>
            <<<grid_blocks, BlockThreads, dynamic_shared_bytes, stream.getStream()>>>(
                input.getMemPtr<InputT>(), stage_output.getMemPtr<void>(), stage_output.getDataType(),
                geometry.outer_size, geometry.reduction_size, geometry.inner_size,
                geometry.tiled_output_outer_stride, geometry.tiled_output_inner_stride,
                shards_per_output, component_tiles, shard_rows, operation, additive_finalize, output_scale);
        CUDA_CHECK(cudaGetLastError());
    } else {
        if constexpr (prunedFp8AlignedInstantiation<InputT, InputTransformT, ReductionOpT,
                                                    PacketBytes, BlockThreads, Topology>()) {
            throw std::logic_error("Pruned FP8 AlignedK specialization requested by launcher.");
        } else {
            fp8AlignedKValueKernel<InputT,
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
                                                                                          component_tiles,
                                                                                          shard_rows,
                                                                                          input_transform,
                                                                                          output_finalize,
                                                                                          output_scale);
            CUDA_CHECK(cudaGetLastError());
        }
    }
}


#endif  // THOR_CUB_ENABLE_FP8_TYPES

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
            throw std::logic_error("AlignedK stage received an unsupported input dtype.");
    }
}

[[nodiscard]] AdditiveFinalizeFp32 additiveFinalizeFromSemantics(const ReductionStageSemantics& semantics) {
    switch (semantics.finalize) {
        case ReductionStageFinalize::Identity:
            return AdditiveFinalizeFp32{1.0f, false};
        case ReductionStageFinalize::Divide:
            return AdditiveFinalizeFp32{static_cast<float>(semantics.finalize_divisor), false};
        case ReductionStageFinalize::SquareRoot:
            return AdditiveFinalizeFp32{1.0f, true};
    }
    throw std::logic_error("Unsupported AlignedK output finalizer.");
}

#if THOR_CUB_ENABLE_FP8_TYPES

// The FP8 AlignedK stage dispatch. Every FP8 (transform, combine, finalizer) composition is routed only through the
// policy's packet/CTA configurations (FP16 / BF16 / FP32 use their own AlignedK kernels). Within a configuration,
// launchFp8AlignedKForBlock selects the hoisted staged, unified complete or typed packet kernel, and honors the
// accepted typed pruning.
template <RKStageTopology Topology, typename TransformT, typename ReductionOpT, typename OutputFinalizeT>
void launchFp8AlignedK(const StridedSpan& input,
                       const StridedSpan& stage_output,
                       const ReductionGeometry& geometry,
                       const Fp8AlignedKStagePlan& plan,
                       OutputFinalizeT output_finalize,
                       float output_scale,
                       Stream& stream) {
    const auto for_type = [&]<typename InputT>() {
        return dispatchFp8AlignedKConfiguration<Topology>(
            plan.packet_bytes, plan.block_threads, [&]<size_t PacketBytes, int BlockThreads>() {
                launchFp8AlignedKForBlock<InputT, TransformT, ReductionOpT, OutputFinalizeT,
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

template <RKStageTopology Topology, typename ReductionOpT>
void launchFp8AlignedKNonadditive(const ReductionStageSemantics& semantics,
                                  const StridedSpan& input,
                                  const StridedSpan& stage_output,
                                  const ReductionGeometry& geometry,
                                  const Fp8AlignedKStagePlan& plan,
                                  float output_scale,
                                  Stream& stream) {
    if (semantics.finalize != ReductionStageFinalize::Identity)
        throw std::logic_error("Nonadditive AlignedK combine requires identity finalization.");
    switch (semantics.input_transform) {
        case ReductionStageInputTransform::Identity:
            return launchFp8AlignedK<Topology, IdentityFp32, ReductionOpT>(
                input, stage_output, geometry, plan, IdentityFp32{}, output_scale, stream);
        case ReductionStageInputTransform::AbsoluteValue:
            return launchFp8AlignedK<Topology, AbsoluteValueFp32, ReductionOpT>(
                input, stage_output, geometry, plan, IdentityFp32{}, output_scale, stream);
        case ReductionStageInputTransform::Square:
            return launchFp8AlignedK<Topology, SquareFp32, ReductionOpT>(
                input, stage_output, geometry, plan, IdentityFp32{}, output_scale, stream);
    }
    throw std::logic_error("Unsupported FP8 AlignedK input transform.");
}

template <RKStageTopology Topology>
void dispatchFp8AlignedKStageSemantics(const ReductionStageSemantics& semantics,
                                       const StridedSpan& input,
                                       const StridedSpan& stage_output,
                                       const ReductionGeometry& geometry,
                                       const Fp8AlignedKStagePlan& plan,
                                       float output_scale,
                                       Stream& stream) {
    switch (semantics.combine) {
        case ReductionStageCombine::Sum: {
            const AdditiveFinalizeFp32 output_finalize = additiveFinalizeFromSemantics(semantics);
            switch (semantics.input_transform) {
                case ReductionStageInputTransform::Identity:
                    return launchFp8AlignedK<Topology, IdentityFp32, cuda::std::plus<float>>(
                        input, stage_output, geometry, plan, output_finalize, output_scale, stream);
                case ReductionStageInputTransform::AbsoluteValue:
                    return launchFp8AlignedK<Topology, AbsoluteValueFp32, cuda::std::plus<float>>(
                        input, stage_output, geometry, plan, output_finalize, output_scale, stream);
                case ReductionStageInputTransform::Square:
                    return launchFp8AlignedK<Topology, SquareFp32, cuda::std::plus<float>>(
                        input, stage_output, geometry, plan, output_finalize, output_scale, stream);
            }
            break;
        }
        case ReductionStageCombine::Product:
            return launchFp8AlignedKNonadditive<Topology, cuda::std::multiplies<float>>(
                semantics, input, stage_output, geometry, plan, output_scale, stream);
        case ReductionStageCombine::Minimum:
            return launchFp8AlignedKNonadditive<Topology, PropagatingMinimumFp32>(
                semantics, input, stage_output, geometry, plan, output_scale, stream);
        case ReductionStageCombine::Maximum:
            return launchFp8AlignedKNonadditive<Topology, PropagatingMaximumFp32>(
                semantics, input, stage_output, geometry, plan, output_scale, stream);
    }
    throw std::logic_error("Unsupported FP8 AlignedK stage semantics.");
}

#endif  // THOR_CUB_ENABLE_FP8_TYPES

void validateCompleteOutput(const StridedSpan& output, const ReductionGeometry& geometry) {
    if (geometry.outer_size > std::numeric_limits<uint64_t>::max() / geometry.inner_size) {
        throw std::overflow_error("AlignedK complete output element count overflows uint64_t.");
    }
    const uint64_t expected_elements = geometry.outer_size * geometry.inner_size;
    if (output.getTotalNumElements() != expected_elements) {
        throw std::logic_error("AlignedK complete output element count does not match stamped geometry.");
    }
    if (geometry.outer_size == 0 || geometry.inner_size == 0) {
        return;
    }
    const uint64_t max_outer = geometry.outer_size - 1;
    const uint64_t max_inner = geometry.inner_size - 1;
    if (max_outer != 0
        && geometry.tiled_output_outer_stride > std::numeric_limits<uint64_t>::max() / max_outer) {
        throw std::overflow_error("AlignedK complete output outer index overflows uint64_t.");
    }
    const uint64_t outer_offset = max_outer * geometry.tiled_output_outer_stride;
    if (max_inner != 0
        && geometry.tiled_output_inner_stride > std::numeric_limits<uint64_t>::max() / max_inner) {
        throw std::overflow_error("AlignedK complete output inner index overflows uint64_t.");
    }
    const uint64_t inner_offset = max_inner * geometry.tiled_output_inner_stride;
    if (outer_offset > std::numeric_limits<uint64_t>::max() - inner_offset
        || outer_offset + inner_offset >= output.getTotalNumElements()) {
        throw std::logic_error("AlignedK complete output strides exceed the supplied output tensor.");
    }
}

}  // namespace

KernelOccupancyInfo queryFp8AlignedKIdentityOccupancy(DataType input_dtype,
                                                          const Fp8AlignedKStagePlan& plan) {
    if (plan.packet_bytes != 2 && plan.packet_bytes != 4 && plan.packet_bytes != 8 && plan.packet_bytes != 16) {
        throw std::logic_error("AlignedK occupancy query requires a 2-, 4-, 8-, or 16-byte packet.");
    }
    if (plan.block_threads != 32 && plan.block_threads != 64 && plan.block_threads != 128
        && plan.block_threads != 256) {
        throw std::logic_error("AlignedK occupancy query requires a compile-time-specialized CTA width.");
    }
    switch (input_dtype) {
#if THOR_CUB_ENABLE_FP8_TYPES
        case DataType::FP8_E4M3:
            if (plan.topology == RKStageTopology::Complete) {
                return queryFp8AlignedKOccupancy<__nv_fp8_e4m3, RKStageTopology::Complete>(
                    plan.packet_bytes, plan.block_threads);
            }
            return queryFp8AlignedKOccupancy<__nv_fp8_e4m3, RKStageTopology::Staged>(
                plan.packet_bytes, plan.block_threads);
        case DataType::FP8_E5M2:
            if (plan.topology == RKStageTopology::Complete) {
                return queryFp8AlignedKOccupancy<__nv_fp8_e5m2, RKStageTopology::Complete>(
                    plan.packet_bytes, plan.block_threads);
            }
            return queryFp8AlignedKOccupancy<__nv_fp8_e5m2, RKStageTopology::Staged>(
                plan.packet_bytes, plan.block_threads);
#endif
        default:
            // FP16 / BF16 / FP32 use their own AlignedK passes.
            throw std::logic_error("AlignedK occupancy query requires FP8 input.");
    }
}

void launchFp8AlignedKStage(const ReductionStageSemantics& semantics,
                            const StridedSpan& input,
                            const StridedSpan& stage_output,
                            const ReductionGeometry& geometry,
                            const Fp8AlignedKStagePlan& plan,
                            float output_scale,
                            Stream& stream) {
    if (geometry.path != ReductionLayout::MiddleBlock) {
        throw std::logic_error("AlignedK stage requires MiddleBlock geometry.");
    }
    if (geometry.outer_size == 0 || geometry.reduction_size == 0 || geometry.inner_size == 0
        || plan.shards_per_output == 0 || plan.shards_per_output > geometry.reduction_size) {
        throw std::logic_error("AlignedK stage requires valid non-zero geometry.");
    }
    if (plan.packet_bytes != 2 && plan.packet_bytes != 4 && plan.packet_bytes != 8 && plan.packet_bytes != 16) {
        throw std::logic_error("AlignedK stage requires a 2-, 4-, 8-, or 16-byte input packet.");
    }
#if THOR_CUB_ENABLE_FP8_TYPES
    const bool fp8_input =
        input.getDataType() == DataType::FP8_E4M3 || input.getDataType() == DataType::FP8_E5M2;
#else
    const bool fp8_input = false;
#endif
    if (!fp8_input) {
        // FP16 / BF16 / FP32 use their own AlignedK passes.
        throw std::logic_error("AlignedK stage requires FP8 input.");
    }
    const size_t input_element_bytes = elementBytes(input.getDataType());
#if THOR_CUB_ENABLE_FP8_TYPES
    if (!fp8AlignedKOutputDTypeSupported(stage_output.getDataType())) {
        throw std::logic_error("FP8 AlignedK writeback requires a floating-point output dtype.");
    }
    if (fp8_input && plan.packet_bytes < 16
        && (reinterpret_cast<uintptr_t>(input.getMemPtr()) % plan.packet_bytes) != 0) {
        throw std::logic_error(
            "Narrow FP8 AlignedK packets require the input base to be naturally packet-aligned.");
    }
#endif
    if (plan.packet_bytes < input_element_bytes || plan.packet_bytes % input_element_bytes != 0
        || geometry.inner_size % static_cast<uint64_t>(plan.packet_bytes / input_element_bytes) != 0) {
        throw std::logic_error("AlignedK stage requires naturally aligned whole-row input packets.");
    }
    if (plan.block_threads != 32 && plan.block_threads != 64 && plan.block_threads != 128
        && plan.block_threads != 256) {
        throw std::logic_error("AlignedK stage requires a compile-time-specialized CTA width.");
    }
#if THOR_CUB_ENABLE_FP8_TYPES
    if (fp8_input && (reinterpret_cast<uintptr_t>(stage_output.getMemPtr()) & uintptr_t{15}) != 0) {
        throw std::logic_error("FP8 AlignedK writeback requires a 16-byte-aligned output base.");
    }
#endif

    switch (plan.topology) {
        case RKStageTopology::Staged: {
            if (plan.shards_per_output <= 1) {
                throw std::logic_error("AlignedK staged topology requires more than one R shard.");
            }
            if (stage_output.getDataType() != DataType::FP32) {
                throw std::logic_error("AlignedK staged topology requires FP32 partial storage.");
            }
            const std::vector<uint64_t> expected_partial_dimensions = {
                geometry.outer_size, plan.shards_per_output, geometry.inner_size};
            if (stage_output.getDimensions() != expected_partial_dimensions) {
                throw std::logic_error("AlignedK staged FP32 partial shape does not match stamped geometry.");
            }
            // Exactly like the successful R-cooperative staged path, the physical first pass consumes only the input
            // transform and associative combine. A mathematically Final role may still arrive here; its finalizer and
            // output scale intentionally remain deferred to the independently stamped continuation.
            ReductionStageSemantics partial_semantics = semantics;
            partial_semantics.finalize = ReductionStageFinalize::Identity;
            partial_semantics.finalize_divisor = 1;
#if THOR_CUB_ENABLE_FP8_TYPES
            dispatchFp8AlignedKStageSemantics<RKStageTopology::Staged>(
                partial_semantics, input, stage_output, geometry, plan, 1.0f, stream);
#endif
            return;
        }
        case RKStageTopology::Complete:
            if (plan.shards_per_output != 1) {
                throw std::logic_error("AlignedK complete topology requires exactly one R shard.");
            }
#if THOR_CUB_ENABLE_FP8_TYPES
            if (fp8_input) {
                if (geometry.tiled_output_inner_stride != 1) {
                    throw std::logic_error("FP8 AlignedK complete writeback requires contiguous inner output.");
                }
                const size_t output_element_bytes = TensorDescriptor::getElementSizeInBytes(stage_output.getDataType());
                if (output_element_bytes == 0) {
                    throw std::logic_error("FP8 AlignedK complete writeback requires a valid output dtype.");
                }
                const size_t producer_output_bytes = plan.packet_bytes * output_element_bytes;
                const size_t store_alignment = std::min<size_t>(producer_output_bytes, 16U);
                if (((geometry.tiled_output_outer_stride % store_alignment) * output_element_bytes)
                    % store_alignment != 0) {
                    throw std::logic_error(
                        "FP8 AlignedK complete writeback requires rows aligned to its physical store width.");
                }
            }
#endif
            validateCompleteOutput(stage_output, geometry);
#if THOR_CUB_ENABLE_FP8_TYPES
            dispatchFp8AlignedKStageSemantics<RKStageTopology::Complete>(
                semantics, input, stage_output, geometry, plan, output_scale, stream);
#endif
            return;
    }
    throw std::logic_error("AlignedK stage selected an invalid topology.");
}


// Exact operation-aware Complete occupancy; unlike the generic AlignedK occupancy interface this accounts for
// P8/CTA64's mixed additive policy.
KernelOccupancyInfo queryFp8AlignedKCompleteOccupancy(
    DataType input_dtype, DataType output_dtype, ReductionStageCombine operation,
    const Fp8AlignedKStagePlan& plan) {
#if THOR_CUB_ENABLE_FP8_TYPES
    if (plan.topology != RKStageTopology::Complete || plan.shards_per_output != 1 ||
        !((plan.packet_bytes == 2 && plan.block_threads == 32) ||
          (plan.packet_bytes == 8 && (plan.block_threads == 32 || plan.block_threads == 64))) ||
        !fp8AlignedKOutputDTypeSupported(output_dtype))
        throw std::logic_error("AlignedK Complete occupancy requires a retained physical configuration.");
    switch (operation) {
        case ReductionStageCombine::Sum:
        case ReductionStageCombine::Product:
        case ReductionStageCombine::Minimum:
        case ReductionStageCombine::Maximum:
            break;
        default:
            throw std::logic_error("AlignedK Complete occupancy requires a supported combine operation.");
    }
    const auto per_dtype = [&]<typename T>() -> KernelOccupancyInfo {
        const auto query_kernel = [&]<typename KernelT>(KernelT kernel, int threads, size_t smem) {
            cudaFuncAttributes attrs{};
            CUDA_CHECK(cudaFuncGetAttributes(&attrs, kernel));
            int active = 0;
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, kernel, threads, smem));
            return KernelOccupancyInfo{attrs.numRegs, attrs.sharedSizeBytes, smem,
                                       active, active * (threads / TILED_REDUCTION_WARP_THREADS)};
        };
        if (plan.packet_bytes == 2) {
            const size_t smem = fp8AlignedKCompleteNeedsTranspose<2>(output_dtype)
                ? FP8_ALIGNED_K_SHARED_BYTES_PER_WARP<2> : 0;
            return query_kernel(fp8AlignedKValueCompleteUnifiedKernel<T, 2, 32, Fp8CompleteFinalizePolicy::Joined>,
                                32, smem);
        }
        if (plan.block_threads == 32) {
            const size_t smem = fp8AlignedKCompleteNeedsTranspose<8>(output_dtype)
                ? FP8_ALIGNED_K_SHARED_BYTES_PER_WARP<8> : 0;
            return query_kernel(fp8AlignedKValueCompleteUnifiedKernel<T, 8, 32, Fp8CompleteFinalizePolicy::InBranch>,
                                32, smem);
        }
        const size_t smem = fp8AlignedKCompleteNeedsTranspose<8>(output_dtype)
            ? 2 * FP8_ALIGNED_K_SHARED_BYTES_PER_WARP<8> : 0;
        if (operation == ReductionStageCombine::Sum)
            return query_kernel(fp8AlignedKValueKernel<T, IdentityFp32, cuda::std::plus<float>,
                                AdditiveFinalizeFp32, 8, 64, RKStageTopology::Complete>, 64, smem);
        return query_kernel(fp8AlignedKValueCompleteUnifiedKernel<T, 8, 64, Fp8CompleteFinalizePolicy::InBranch>,
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

template <RKStageTopology Topology, typename InputT, typename TransformT,
          typename ReductionOpT, typename FinalizeT>
[[nodiscard]] KernelOccupancyInfo queryFp8AlignedKTransformedTyped(
    DataType output_dtype, const Fp8AlignedKStagePlan& plan) {
    const auto query = [&]<size_t PacketBytes, int BlockThreads>() -> KernelOccupancyInfo {
        if constexpr (prunedFp8AlignedInstantiation<InputT, TransformT, ReductionOpT,
                                                    PacketBytes, BlockThreads, Topology>()) {
            return {}; // Physical kernel removed; the planner must choose another configuration.
        } else {
            const bool needs_transpose = Topology == RKStageTopology::Staged
                ? FP8_ALIGNED_K_STAGED_NEEDS_TRANSPOSE<PacketBytes>
                : fp8AlignedKCompleteNeedsTranspose<PacketBytes>(output_dtype);
            const size_t smem = needs_transpose
                ? (BlockThreads / TILED_REDUCTION_WARP_THREADS) * FP8_ALIGNED_K_SHARED_BYTES_PER_WARP<PacketBytes>
                : 0;
            const auto kernel = fp8AlignedKValueKernel<InputT, TransformT, ReductionOpT, FinalizeT,
                                                       PacketBytes, BlockThreads, Topology>;
            cudaFuncAttributes attributes{};
            CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
            int active = 0;
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, kernel, BlockThreads, smem));
            return KernelOccupancyInfo{attributes.numRegs, attributes.sharedSizeBytes, smem,
                                       active, active * (BlockThreads / TILED_REDUCTION_WARP_THREADS)};
        }
    };
    switch (plan.packet_bytes) {
        case 2:
            switch (plan.block_threads) {
                case 32: return query.template operator()<2, 32>();
                case 64:
                    if constexpr (Topology == RKStageTopology::Staged)
                        return query.template operator()<2, 64>();
                    break;
                case 128:
                    if constexpr (Topology == RKStageTopology::Staged)
                        return query.template operator()<2, 128>();
                    break;
            }
            break;
        case 8:
            switch (plan.block_threads) {
                case 32: return query.template operator()<8, 32>();
                case 64: return query.template operator()<8, 64>();
                case 128:
                    if constexpr (Topology == RKStageTopology::Staged)
                        return query.template operator()<8, 128>();
                    break;
            }
            break;
    }
    throw std::logic_error("Transformed AlignedK occupancy requires a retained packet/CTA configuration.");
}

template <RKStageTopology Topology, typename InputT, typename TransformT>
[[nodiscard]] KernelOccupancyInfo queryFp8AlignedKTransformedCombine(
    DataType output_dtype, ReductionStageCombine operation, const Fp8AlignedKStagePlan& plan) {
    switch (operation) {
        case ReductionStageCombine::Sum:
            return queryFp8AlignedKTransformedTyped<Topology, InputT, TransformT,
                cuda::std::plus<float>, AdditiveFinalizeFp32>(output_dtype, plan);
        case ReductionStageCombine::Product:
            return queryFp8AlignedKTransformedTyped<Topology, InputT, TransformT,
                cuda::std::multiplies<float>, IdentityFp32>(output_dtype, plan);
        case ReductionStageCombine::Minimum:
            return queryFp8AlignedKTransformedTyped<Topology, InputT, TransformT,
                PropagatingMinimumFp32, IdentityFp32>(output_dtype, plan);
        case ReductionStageCombine::Maximum:
            return queryFp8AlignedKTransformedTyped<Topology, InputT, TransformT,
                PropagatingMaximumFp32, IdentityFp32>(output_dtype, plan);
    }
    throw std::logic_error("Transformed AlignedK occupancy requires a supported combine operation.");
}

template <RKStageTopology Topology, typename InputT>
[[nodiscard]] KernelOccupancyInfo queryFp8AlignedKTransform(
    DataType output_dtype, ReductionStageInputTransform transform,
    ReductionStageCombine operation, const Fp8AlignedKStagePlan& plan) {
    switch (transform) {
        case ReductionStageInputTransform::AbsoluteValue:
            return queryFp8AlignedKTransformedCombine<Topology, InputT, AbsoluteValueFp32>(
                output_dtype, operation, plan);
        case ReductionStageInputTransform::Square:
            return queryFp8AlignedKTransformedCombine<Topology, InputT, SquareFp32>(
                output_dtype, operation, plan);
        case ReductionStageInputTransform::Identity:
            break;
    }
    throw std::logic_error("Transformed AlignedK occupancy requires ABS or square transform.");
}

#endif  // THOR_CUB_ENABLE_FP8_TYPES

// Occupancy of the ACTUALLY selected transformed FP8 AlignedK packet kernel.
// The Identity occupancy queries are responsible for the hoisted Staged and mixed Complete implementations; they must
// not be substituted for the transformed typed packet kernel.
KernelOccupancyInfo queryFp8AlignedKStageOccupancy(
    DataType input_dtype, DataType output_dtype, ReductionStageInputTransform transform,
    ReductionStageCombine operation, const Fp8AlignedKStagePlan& plan) {
    if (transform == ReductionStageInputTransform::Identity)
        return plan.topology == RKStageTopology::Complete
            ? queryFp8AlignedKCompleteOccupancy(input_dtype, output_dtype, operation, plan)
            : queryFp8AlignedKIdentityOccupancy(input_dtype, plan);
#if THOR_CUB_ENABLE_FP8_TYPES
    if (!fp8AlignedKOutputDTypeSupported(output_dtype))
        throw std::logic_error("Transformed AlignedK occupancy requires a floating-point output.");
    const auto typed = [&]<typename InputT>() -> KernelOccupancyInfo {
        if (plan.topology == RKStageTopology::Staged) {
            if (plan.shards_per_output <= 1 || output_dtype != DataType::FP32)
                throw std::logic_error("Transformed AlignedK staged occupancy requires FP32 partials.");
            return queryFp8AlignedKTransform<RKStageTopology::Staged, InputT>(
                output_dtype, transform, operation, plan);
        }
        if (plan.topology == RKStageTopology::Complete && plan.shards_per_output == 1)
            return queryFp8AlignedKTransform<RKStageTopology::Complete, InputT>(
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


// ---- FP32 and FP16 / BF16 AlignedK, with the 4-byte packet for VALUE (see alignedKValueKernel).
namespace {
[[nodiscard]] bool alignedKSupportsElement(uint32_t element_bytes, uint64_t outer_size, uint64_t reduction_size,
                                           uint64_t inner_size, uint32_t packet_bytes, uint32_t block_threads,
                                           uint64_t shards_per_output) {
    if (packet_bytes < element_bytes || packet_bytes > 16 || (packet_bytes & (packet_bytes - 1)) != 0 ||
        (block_threads != 32 && block_threads != 64 && block_threads != 128) || outer_size == 0 ||
        reduction_size == 0 || inner_size == 0 || shards_per_output == 0 || shards_per_output > reduction_size) {
        return false;
    }
    const uint64_t items = packet_bytes / element_bytes;
    if (inner_size % items != 0) return false;
    const uint64_t tiles = ceilDivideU64(inner_size, items * block_threads);
    return RK::rkWorkGridFits(shards_per_output, tiles, outer_size) &&
           ceilDivideU64(reduction_size, shards_per_output) <= std::numeric_limits<uint32_t>::max();
}
}  // namespace

bool fp32AlignedKSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size, uint32_t packet_bytes,
                          uint32_t block_threads, uint64_t shards_per_output) {
    return alignedKSupportsElement(4, outer_size, reduction_size, inner_size, packet_bytes, block_threads,
                                   shards_per_output);
}

bool f16AlignedKSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size, uint32_t packet_bytes,
                         uint32_t block_threads, uint64_t shards_per_output) {
    return alignedKSupportsElement(2, outer_size, reduction_size, inner_size, packet_bytes, block_threads,
                                   shards_per_output);
}

namespace {
[[nodiscard]] bool fp32AlignedKOutputSupported(DataType dtype) {
    switch (dtype) {
#if THOR_CUB_ENABLE_FP8_TYPES
        case DataType::FP8_E4M3:
        case DataType::FP8_E5M2:
#endif
        case DataType::FP16:
        case DataType::BF16:
        case DataType::FP32:
            return true;
        default:
            return false;
    }
}

[[nodiscard]] bool fp32AlignedKSemanticsSupported(const ReductionStageSemantics& semantics) {
    const bool transform_ok = semantics.input_transform == ReductionStageInputTransform::Identity ||
                              semantics.input_transform == ReductionStageInputTransform::AbsoluteValue ||
                              semantics.input_transform == ReductionStageInputTransform::Square;
    if (!transform_ok) return false;
    if (semantics.combine == ReductionStageCombine::Product)
        return semantics.input_transform == ReductionStageInputTransform::Identity;
    return true;
}

// One input type's AlignedK stage: FP32 (packets 4 / 8 / 16) or FP16 / BF16 (packets 2 / 4 / 8 / 16). The FP16 / BF16
// launcher checks the input dtype and picks InputT; `name` labels the errors.
template <typename InputT>
void launchAlignedKStageTyped(const ReductionStageSemantics& semantics,
                              const StridedSpan& input,
                              const StridedSpan& stage_output,
                              const ReductionGeometry& geometry,
                              uint32_t packet_bytes,
                              uint32_t block_threads,
                              uint64_t shards_per_output,
                              float output_scale,
                              Stream& stream,
                              const char* name) {
    constexpr uint32_t ElementBytes = sizeof(InputT);
    const bool staged = shards_per_output > 1;
    if (!fp32AlignedKSemanticsSupported(semantics) ||
        !alignedKSupportsElement(ElementBytes, geometry.outer_size, geometry.reduction_size, geometry.inner_size,
                                 packet_bytes, block_threads, shards_per_output) ||
        input.getTotalNumElements() != geometry.outer_size * geometry.reduction_size * geometry.inner_size ||
        (reinterpret_cast<uintptr_t>(input.getMemPtr<void>()) & uintptr_t{packet_bytes - 1}) != 0) {
        throw std::logic_error(std::string(name) + " AlignedK received an unsupported input, geometry or semantics.");
    }
    if (staged ? (stage_output.getDataType() != DataType::FP32 ||
                  stage_output.getTotalNumElements() != geometry.outer_size * shards_per_output * geometry.inner_size)
               : (!fp32AlignedKOutputSupported(stage_output.getDataType()) ||
                  stage_output.getTotalNumElements() != geometry.outer_size * geometry.inner_size)) {
        throw std::logic_error(std::string(name) + " AlignedK received an unsupported stage output.");
    }
    const uint64_t items = packet_bytes / ElementBytes;
    const uint64_t component_tiles = ceilDivideU64(geometry.inner_size, items * block_threads);
    const ReductionShardRows shard_rows = balancedReductionShardRows(geometry.reduction_size, shards_per_output);
    const dim3 grid_blocks = reductionWorkGrid(shards_per_output, component_tiles, geometry.outer_size, 0);
    // Complete: the semantics' finalizer (SUM family) and the runtime scale apply once; Staged writes raw partials.
    const AdditiveFinalizeFp32 finalize = staged ? AdditiveFinalizeFp32{1.0f, false}
                                                 : additiveFinalizeFromSemantics(semantics);
    // A natural [outer, K] output unless the geometry says otherwise (Staged partials are always natural).
    const uint64_t outer_stride = geometry.tiled_output_outer_stride != 0 ? geometry.tiled_output_outer_stride
                                                                          : geometry.inner_size;
    const uint64_t inner_stride = geometry.tiled_output_inner_stride != 0 ? geometry.tiled_output_inner_stride : 1;
    const auto launch = [&]<RKStageTopology Topology>() {
        return dispatchFp32AlignedKConfiguration<Topology, true, ElementBytes>(
            packet_bytes, block_threads, [&]<int ItemsPerLane, int BlockThreads>() {
                alignedKValueKernel<InputT, ItemsPerLane, BlockThreads, Topology>
                    <<<grid_blocks, BlockThreads, 0, stream.getStream()>>>(
                        input.getMemPtr<InputT>(), stage_output.getMemPtr<void>(), stage_output.getDataType(),
                        geometry.outer_size, geometry.reduction_size, geometry.inner_size, outer_stride,
                        inner_stride, shards_per_output, component_tiles, shard_rows, semantics.combine,
                        semantics.input_transform, finalize, staged ? 1.0f : output_scale);
                CUDA_CHECK(cudaGetLastError());
            });
    };
    const bool launched = staged ? launch.template operator()<RKStageTopology::Staged>()
                                 : launch.template operator()<RKStageTopology::Complete>();
    if (!launched) throw std::logic_error(std::string(name) + " AlignedK has no kernel for this configuration.");
}

template <typename InputT>
[[nodiscard]] bool queryAlignedKOccupancyTyped(uint32_t packet_bytes, uint32_t block_threads, bool staged,
                                               KernelOccupancyInfo& occupancy) {
    const auto query = [&]<RKStageTopology Topology>() {
        return dispatchFp32AlignedKConfiguration<Topology, true, sizeof(InputT)>(
            packet_bytes, block_threads, [&]<int ItemsPerLane, int BlockThreads>() {
                const auto kernel = alignedKValueKernel<InputT, ItemsPerLane, BlockThreads, Topology>;
                cudaFuncAttributes attributes{};
                CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
                int active_blocks = 0;
                CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks, kernel, BlockThreads, 0));
                occupancy = KernelOccupancyInfo{attributes.numRegs, attributes.sharedSizeBytes, 0, active_blocks,
                                                active_blocks * (BlockThreads / TILED_REDUCTION_WARP_THREADS)};
            });
    };
    return staged ? query.template operator()<RKStageTopology::Staged>()
                  : query.template operator()<RKStageTopology::Complete>();
}
}  // namespace

void launchFp32AlignedKStage(const ReductionStageSemantics& semantics,
                             const StridedSpan& input,
                             const StridedSpan& stage_output,
                             const ReductionGeometry& geometry,
                             uint32_t packet_bytes,
                             uint32_t block_threads,
                             uint64_t shards_per_output,
                             float output_scale,
                             Stream& stream) {
    if (input.getDataType() != DataType::FP32)
        throw std::logic_error("FP32 AlignedK received an unsupported input, geometry or semantics.");
    launchAlignedKStageTyped<float>(semantics, input, stage_output, geometry, packet_bytes, block_threads,
                                    shards_per_output, output_scale, stream, "FP32");
}

KernelOccupancyInfo queryFp32AlignedKOccupancy(uint32_t packet_bytes, uint32_t block_threads, bool staged) {
    KernelOccupancyInfo occupancy{};
    if (!queryAlignedKOccupancyTyped<float>(packet_bytes, block_threads, staged, occupancy))
        throw std::logic_error("FP32 AlignedK occupancy requires a 4-, 8- or 16-byte packet and CTA 32/64/128.");
    return occupancy;
}

void launchF16AlignedKStage(const ReductionStageSemantics& semantics,
                            const StridedSpan& input,
                            const StridedSpan& stage_output,
                            const ReductionGeometry& geometry,
                            uint32_t packet_bytes,
                            uint32_t block_threads,
                            uint64_t shards_per_output,
                            float output_scale,
                            Stream& stream) {
    switch (input.getDataType()) {
        case DataType::FP16:
            launchAlignedKStageTyped<__half>(semantics, input, stage_output, geometry, packet_bytes, block_threads,
                                             shards_per_output, output_scale, stream, "FP16");
            return;
        case DataType::BF16:
            launchAlignedKStageTyped<__nv_bfloat16>(semantics, input, stage_output, geometry, packet_bytes,
                                                    block_threads, shards_per_output, output_scale, stream, "BF16");
            return;
        default:
            throw std::logic_error("FP16 / BF16 AlignedK received a non-FP16 / BF16 input.");
    }
}

KernelOccupancyInfo queryF16AlignedKOccupancy(DataType input_dtype, uint32_t packet_bytes, uint32_t block_threads,
                                              bool staged) {
    KernelOccupancyInfo occupancy{};
    bool queried = false;
    if (input_dtype == DataType::FP16)
        queried = queryAlignedKOccupancyTyped<__half>(packet_bytes, block_threads, staged, occupancy);
    else if (input_dtype == DataType::BF16)
        queried = queryAlignedKOccupancyTyped<__nv_bfloat16>(packet_bytes, block_threads, staged, occupancy);
    if (!queried) {
        throw std::logic_error(
            "FP16 / BF16 AlignedK occupancy requires FP16 / BF16, a 2-, 4-, 8- or 16-byte packet and CTA 32/64/128.");
    }
    return occupancy;
}


// ---- FP32 RK ARG, the AlignedK twin and the packed continuation (see alignedKArgKernel).
bool fp32AlignedKArgSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size, uint32_t packet_bytes,
                             uint32_t block_threads, uint64_t shards_per_output) {
    // The 4-byte packet is VALUE only.
    return packet_bytes != 4 &&
           fp32AlignedKSupports(outer_size, reduction_size, inner_size, packet_bytes, block_threads,
                                shards_per_output) &&
           ceilDivideU64(reduction_size, shards_per_output) <= FP32_ARG_MAX_SHARD_ROWS;
}

// The FP16 / BF16 twin: 4-, 8- and 16-byte packets (the 2-byte packet is VALUE only).
bool f16AlignedKArgSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size, uint32_t packet_bytes,
                            uint32_t block_threads, uint64_t shards_per_output) {
    return packet_bytes != 2 &&
           f16AlignedKSupports(outer_size, reduction_size, inner_size, packet_bytes, block_threads,
                               shards_per_output) &&
           ceilDivideU64(reduction_size, shards_per_output) <= FP32_ARG_MAX_SHARD_ROWS;
}

uint64_t fp32ArgPackedScratchBytes(uint64_t outer_size, uint64_t shards, uint64_t inner_size) {
    return shards > 1 ? outer_size * shards * inner_size * sizeof(uint64_t) : 0;
}

namespace {
// The outputs of an FP32 ARG pass: value (any float storage) and / or index (UINT32 or UINT64; UINT32 must hold the
// largest original-domain index).
[[nodiscard]] Fp32ArgOutputs fp32ArgOutputsFor(const StridedSpan* value_output, const StridedSpan* index_output, uint64_t outputs,
                                               uint64_t reduction_size, bool maximum, uint64_t domain_stride,
                                               const char* who) {
    if ((value_output == nullptr && index_output == nullptr) ||
        (value_output != nullptr && (value_output->getTotalNumElements() != outputs ||
                                     !fp32AlignedKOutputSupported(value_output->getDataType()))) ||
        (index_output != nullptr && (index_output->getTotalNumElements() != outputs ||
                                     (index_output->getDataType() != DataType::UINT32 &&
                                      index_output->getDataType() != DataType::UINT64)))) {
        throw std::logic_error(std::string(who) + " received unsupported outputs.");
    }
    if (index_output != nullptr && index_output->getDataType() == DataType::UINT32 &&
        (reduction_size - 1) * domain_stride > std::numeric_limits<uint32_t>::max()) {
        throw std::logic_error(std::string(who) + " UINT32 index cannot hold the original-domain index.");
    }
    return Fp32ArgOutputs{value_output == nullptr ? nullptr : value_output->getMemPtr<void>(),
                          value_output == nullptr ? DataType::FP32 : value_output->getDataType(),
                          index_output == nullptr ? nullptr : index_output->getMemPtr<void>(),
                          index_output == nullptr ? DataType::UINT32 : index_output->getDataType(),
                          domain_stride,
                          maximum ? 0x80000000U : 0U};
}

void launchFp32ArgPackedContinuationRows(const uint64_t* packed, const Fp32ArgOutputs& outputs, uint64_t outer_size,
                                         uint64_t shards, uint64_t inner_size, ReductionShardRows rows,
                                         Stream& stream) {
    constexpr uint64_t MaxGridBlocks = 65535;
    if (fp8ArgPackedContinuationSplitsShards(outer_size, inner_size, shards)) {
        const Fp8ArgPackedSplitLayout layout = fp8ArgPackedSplitLayout(inner_size, shards);
        const uint64_t items = outer_size * ceilDivideU64(inner_size, layout.columns);
        fp32ArgPackedContinuationSplitKernel<0>
            <<<static_cast<unsigned int>(std::min<uint64_t>(items, MaxGridBlocks)), layout.threads, 0,
               stream.getStream()>>>(packed, outputs, outer_size, shards, inner_size, rows, layout.columns,
                                     layout.groups);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    constexpr uint32_t Threads = 256;
    const uint64_t blocks = std::min<uint64_t>(ceilDivideU64(outer_size * inner_size, Threads), MaxGridBlocks);
    fp32ArgPackedContinuationKernel<0><<<static_cast<unsigned int>(blocks), Threads, 0, stream.getStream()>>>(
        packed, outputs, outer_size, shards, inner_size, rows);
    CUDA_CHECK(cudaGetLastError());
}
}  // namespace

void launchFp32ArgPackedContinuation(const uint64_t* packed_partials,
                                     const StridedSpan* value_output,
                                     const StridedSpan* index_output,
                                     uint64_t outer_size,
                                     uint64_t reduction_size,
                                     uint64_t inner_size,
                                     uint64_t shards,
                                     uint64_t uniform_rows_per_shard,
                                     bool maximum,
                                     uint64_t domain_stride,
                                     Stream& stream) {
    const uint64_t outputs = outer_size * inner_size;
    if (packed_partials == nullptr || outputs == 0 || shards < 2 || shards > reduction_size ||
        shards > REDUCTION_WORK_GRID_MAX_EXTENT ||
        (uniform_rows_per_shard != 0 && (shards != ceilDivideU64(reduction_size, uniform_rows_per_shard) ||
                                         uniform_rows_per_shard > FP32_ARG_MAX_SHARD_ROWS))) {
        throw std::logic_error("FP32 packed ARG continuation received unsupported geometry.");
    }
    const Fp32ArgOutputs outputs_spec = fp32ArgOutputsFor(value_output, index_output, outputs, reduction_size, maximum,
                                                          domain_stride, "FP32 packed ARG continuation");
    // Uniform depths: shard s begins at s * depth. Balanced (AlignedK): VALUE's split.
    const ReductionShardRows rows = uniform_rows_per_shard != 0
                                        ? ReductionShardRows{static_cast<uint32_t>(uniform_rows_per_shard), 0U}
                                        : balancedReductionShardRows(reduction_size, shards);
    launchFp32ArgPackedContinuationRows(packed_partials, outputs_spec, outer_size, shards, inner_size, rows, stream);
}

namespace {
// One input type's AlignedK ARG pass (FP32, or FP16 / BF16 widened to the FP32 key); `name` labels the errors.
template <typename InputT>
void launchAlignedKArgTyped(const StridedSpan& input,
                            const StridedSpan* value_output,
                            const StridedSpan* index_output,
                            const ReductionGeometry& geometry,
                            uint32_t packet_bytes,
                            uint32_t block_threads,
                            uint64_t shards_per_output,
                            uint64_t* packed_scratch,
                            bool maximum,
                            uint64_t domain_stride,
                            Stream& stream,
                            const char* name) {
    constexpr uint32_t ElementBytes = sizeof(InputT);
    const bool staged = shards_per_output > 1;
    const bool supported =
        ElementBytes == 4 ? fp32AlignedKArgSupports(geometry.outer_size, geometry.reduction_size, geometry.inner_size,
                                                    packet_bytes, block_threads, shards_per_output)
                          : f16AlignedKArgSupports(geometry.outer_size, geometry.reduction_size, geometry.inner_size,
                                                   packet_bytes, block_threads, shards_per_output);
    if (!supported ||
        input.getTotalNumElements() != geometry.outer_size * geometry.reduction_size * geometry.inner_size ||
        (reinterpret_cast<uintptr_t>(input.getMemPtr<void>()) & uintptr_t{packet_bytes - 1}) != 0 ||
        (staged && packed_scratch == nullptr)) {
        throw std::logic_error(std::string(name) +
                               " AlignedK ARG received an unsupported input, geometry or scratch.");
    }
    const std::string who = std::string(name) + " AlignedK ARG";
    const Fp32ArgOutputs outputs =
        fp32ArgOutputsFor(value_output, index_output, geometry.outer_size * geometry.inner_size,
                          geometry.reduction_size, maximum, domain_stride, who.c_str());
    const uint64_t items = packet_bytes / ElementBytes;
    const uint64_t component_tiles = ceilDivideU64(geometry.inner_size, items * block_threads);
    const ReductionShardRows shard_rows = balancedReductionShardRows(geometry.reduction_size, shards_per_output);
    const dim3 grid_blocks = reductionWorkGrid(shards_per_output, component_tiles, geometry.outer_size, 0);
    const auto launch = [&]<bool Staged>() {
        return dispatchFp32AlignedKConfiguration<RKStageTopology::Complete, false, ElementBytes>(
            packet_bytes, block_threads, [&]<int ItemsPerLane, int BlockThreads>() {
                alignedKArgKernel<InputT, ItemsPerLane, BlockThreads, Staged>
                    <<<grid_blocks, BlockThreads, 0, stream.getStream()>>>(
                        input.getMemPtr<InputT>(), outputs, packed_scratch, geometry.outer_size,
                        geometry.reduction_size, geometry.inner_size, shards_per_output, component_tiles,
                        shard_rows);
                CUDA_CHECK(cudaGetLastError());
            });
    };
    const bool launched = staged ? launch.template operator()<true>() : launch.template operator()<false>();
    if (!launched) throw std::logic_error(std::string(name) + " AlignedK ARG has no kernel for this configuration.");
    if (staged) {
        launchFp32ArgPackedContinuationRows(packed_scratch, outputs, geometry.outer_size, shards_per_output,
                                            geometry.inner_size, shard_rows, stream);
    }
}

template <typename InputT>
[[nodiscard]] bool queryAlignedKArgOccupancyTyped(uint32_t packet_bytes, uint32_t block_threads, bool staged,
                                                  KernelOccupancyInfo& occupancy) {
    const auto query = [&]<bool Staged>() {
        return dispatchFp32AlignedKConfiguration<RKStageTopology::Complete, false, sizeof(InputT)>(
            packet_bytes, block_threads, [&]<int ItemsPerLane, int BlockThreads>() {
                const auto kernel = alignedKArgKernel<InputT, ItemsPerLane, BlockThreads, Staged>;
                cudaFuncAttributes attributes{};
                CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
                int active_blocks = 0;
                CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks, kernel, BlockThreads, 0));
                occupancy = KernelOccupancyInfo{attributes.numRegs, attributes.sharedSizeBytes, 0, active_blocks,
                                                active_blocks * (BlockThreads / TILED_REDUCTION_WARP_THREADS)};
            });
    };
    return staged ? query.template operator()<true>() : query.template operator()<false>();
}
}  // namespace

void launchFp32AlignedKArg(const StridedSpan& input,
                           const StridedSpan* value_output,
                           const StridedSpan* index_output,
                           const ReductionGeometry& geometry,
                           uint32_t packet_bytes,
                           uint32_t block_threads,
                           uint64_t shards_per_output,
                           uint64_t* packed_scratch,
                           bool maximum,
                           uint64_t domain_stride,
                           Stream& stream) {
    if (input.getDataType() != DataType::FP32)
        throw std::logic_error("FP32 AlignedK ARG received an unsupported input, geometry or scratch.");
    launchAlignedKArgTyped<float>(input, value_output, index_output, geometry, packet_bytes, block_threads,
                                  shards_per_output, packed_scratch, maximum, domain_stride, stream, "FP32");
}

void launchF16AlignedKArg(const StridedSpan& input,
                          const StridedSpan* value_output,
                          const StridedSpan* index_output,
                          const ReductionGeometry& geometry,
                          uint32_t packet_bytes,
                          uint32_t block_threads,
                          uint64_t shards_per_output,
                          uint64_t* packed_scratch,
                          bool maximum,
                          uint64_t domain_stride,
                          Stream& stream) {
    switch (input.getDataType()) {
        case DataType::FP16:
            return launchAlignedKArgTyped<__half>(input, value_output, index_output, geometry, packet_bytes,
                                                  block_threads, shards_per_output, packed_scratch, maximum,
                                                  domain_stride, stream, "FP16");
        case DataType::BF16:
            return launchAlignedKArgTyped<__nv_bfloat16>(input, value_output, index_output, geometry, packet_bytes,
                                                         block_threads, shards_per_output, packed_scratch, maximum,
                                                         domain_stride, stream, "BF16");
        default:
            throw std::logic_error("FP16 / BF16 AlignedK ARG received a non-FP16 / BF16 input.");
    }
}

KernelOccupancyInfo queryF16AlignedKArgOccupancy(DataType input_dtype, uint32_t packet_bytes,
                                                 uint32_t block_threads, bool staged) {
    KernelOccupancyInfo occupancy{};
    bool queried = false;
    if (input_dtype == DataType::FP16)
        queried = queryAlignedKArgOccupancyTyped<__half>(packet_bytes, block_threads, staged, occupancy);
    else if (input_dtype == DataType::BF16)
        queried = queryAlignedKArgOccupancyTyped<__nv_bfloat16>(packet_bytes, block_threads, staged, occupancy);
    if (!queried) {
        throw std::logic_error(
            "FP16 / BF16 AlignedK ARG occupancy requires FP16 / BF16, a 4-, 8- or 16-byte packet and CTA 32/64/128.");
    }
    return occupancy;
}

// ---- Carried-index FP32 ARG (see Fp32ArgWidePartials).
uint64_t fp32ArgWideScratchBytes(uint64_t outer_size, uint64_t shards, uint64_t inner_size) {
    if (shards < 2) return 0;
    const uint64_t partials = outer_size * shards * inner_size;
    return ((partials * sizeof(uint32_t) + 255) & ~uint64_t{255}) + partials * sizeof(uint64_t);
}

namespace {
[[nodiscard]] Fp32ArgWidePartials fp32ArgWidePartialsAt(void* scratch, uint64_t outer_size, uint64_t shards,
                                                        uint64_t inner_size) {
    if (scratch == nullptr) return {nullptr, nullptr};
    if ((reinterpret_cast<uintptr_t>(scratch) & uintptr_t{255}) != 0)
        throw std::logic_error("FP32 wide ARG partials require a 256-byte aligned scratch.");
    const uint64_t partials = outer_size * shards * inner_size;
    auto* bytes = static_cast<uint8_t*>(scratch);
    return {reinterpret_cast<uint32_t*>(bytes),
            reinterpret_cast<uint64_t*>(bytes + ((partials * sizeof(uint32_t) + 255) & ~uint64_t{255}))};
}

// The outputs of a carried-index pass: the tails hold the composed index, so the index stride is 1. A UINT32 index
// output is the caller's choice (the planner sizes it from the whole original domain).
[[nodiscard]] Fp32ArgOutputs fp32CarriedArgOutputsFor(const StridedSpan* value_output, const StridedSpan* index_output, uint64_t outputs,
                                                      bool maximum, const char* who) {
    return fp32ArgOutputsFor(value_output, index_output, outputs, 1, maximum, 1, who);
}

void launchFp32ArgWideContinuationPartials(const Fp32ArgWidePartials& partials, const Fp32ArgOutputs& outputs,
                                           uint64_t outer_size, uint64_t shards, uint64_t inner_size,
                                           Stream& stream) {
    constexpr uint64_t MaxGridBlocks = 65535;
    if (fp8ArgPackedContinuationSplitsShards(outer_size, inner_size, shards)) {
        const Fp8ArgPackedSplitLayout layout = fp8ArgPackedSplitLayout(inner_size, shards);
        const uint64_t items = outer_size * ceilDivideU64(inner_size, layout.columns);
        fp32ArgWideContinuationSplitKernel<0>
            <<<static_cast<unsigned int>(std::min<uint64_t>(items, MaxGridBlocks)), layout.threads, 0,
               stream.getStream()>>>(partials, outputs, outer_size, shards, inner_size, layout.columns,
                                     layout.groups);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    constexpr uint32_t Threads = 256;
    const uint64_t blocks = std::min<uint64_t>(ceilDivideU64(outer_size * inner_size, Threads), MaxGridBlocks);
    fp32ArgWideContinuationKernel<0><<<static_cast<unsigned int>(blocks), Threads, 0, stream.getStream()>>>(
        partials, outputs, outer_size, shards, inner_size);
    CUDA_CHECK(cudaGetLastError());
}

// A carried index tensor of the input's element count, UINT32 or UINT64, aligned to `alignment` bytes.
void checkFp32CarriedIndex(const StridedSpan& carried_index, uint64_t elements, uint64_t alignment, const char* who) {
    if ((carried_index.getDataType() != DataType::UINT32 && carried_index.getDataType() != DataType::UINT64) ||
        carried_index.getTotalNumElements() != elements ||
        (reinterpret_cast<uintptr_t>(carried_index.getMemPtr<void>()) & uintptr_t{alignment - 1}) != 0) {
        throw std::logic_error(std::string(who) + " received an unsupported carried index.");
    }
}
}  // namespace

void launchFp32ArgWideContinuation(void* wide_scratch,
                                   const StridedSpan* value_output,
                                   const StridedSpan* index_output,
                                   uint64_t outer_size,
                                   uint64_t shards,
                                   uint64_t inner_size,
                                   bool maximum,
                                   Stream& stream) {
    if (wide_scratch == nullptr || outer_size == 0 || inner_size == 0 || shards < 2 ||
        shards > REDUCTION_WORK_GRID_MAX_EXTENT) {
        throw std::logic_error("FP32 wide ARG continuation received unsupported geometry.");
    }
    const Fp32ArgOutputs outputs = fp32CarriedArgOutputsFor(value_output, index_output, outer_size * inner_size,
                                                            maximum, "FP32 wide ARG continuation");
    launchFp32ArgWideContinuationPartials(fp32ArgWidePartialsAt(wide_scratch, outer_size, shards, inner_size), outputs,
                                          outer_size, shards, inner_size, stream);
}

void launchFp32AlignedKCarriedArg(const StridedSpan& input,
                                  const StridedSpan& carried_index,
                                  const StridedSpan* value_output,
                                  const StridedSpan* index_output,
                                  const ReductionGeometry& geometry,
                                  uint32_t packet_bytes,
                                  uint32_t block_threads,
                                  uint64_t shards_per_output,
                                  void* wide_scratch,
                                  bool maximum,
                                  uint64_t row_stride,
                                  Stream& stream) {
    const bool staged = shards_per_output > 1;
    const uint64_t elements = geometry.outer_size * geometry.reduction_size * geometry.inner_size;
    if (input.getDataType() != DataType::FP32 || packet_bytes == 4 ||  // the 4-byte packet is VALUE only
        !fp32AlignedKSupports(geometry.outer_size, geometry.reduction_size, geometry.inner_size, packet_bytes,
                              block_threads, shards_per_output) ||
        input.getTotalNumElements() != elements ||
        (reinterpret_cast<uintptr_t>(input.getMemPtr<void>()) & uintptr_t{packet_bytes - 1}) != 0 ||
        (staged && wide_scratch == nullptr)) {
        throw std::logic_error("FP32 AlignedK carried ARG received an unsupported input, geometry or scratch.");
    }
    const uint64_t items = packet_bytes / 4;
    const uint64_t index_bytes = carried_index.getDataType() == DataType::UINT64 ? 8 : 4;
    checkFp32CarriedIndex(carried_index, elements, std::min<uint64_t>(16, items * index_bytes),
                          "FP32 AlignedK carried ARG");
    const Fp32ArgOutputs outputs = fp32CarriedArgOutputsFor(value_output, index_output,
                                                            geometry.outer_size * geometry.inner_size, maximum,
                                                            "FP32 AlignedK carried ARG");
    const Fp32ArgWidePartials partials =
        fp32ArgWidePartialsAt(staged ? wide_scratch : nullptr, geometry.outer_size, shards_per_output,
                              geometry.inner_size);
    const uint64_t component_tiles = ceilDivideU64(geometry.inner_size, items * block_threads);
    const ReductionShardRows shard_rows = balancedReductionShardRows(geometry.reduction_size, shards_per_output);
    const dim3 grid_blocks = reductionWorkGrid(shards_per_output, component_tiles, geometry.outer_size, 0);
    const auto launch = [&]<bool Staged, typename IndexT>() {
        return dispatchFp32AlignedKConfiguration<RKStageTopology::Complete>(
            packet_bytes, block_threads, [&]<int ItemsPerLane, int BlockThreads>() {
                alignedKCarriedArgKernel<ItemsPerLane, BlockThreads, Staged, IndexT>
                    <<<grid_blocks, BlockThreads, 0, stream.getStream()>>>(
                        input.getMemPtr<float>(), carried_index.getMemPtr<IndexT>(), outputs, partials,
                        geometry.outer_size, geometry.reduction_size, geometry.inner_size, shards_per_output,
                        component_tiles, shard_rows, row_stride);
                CUDA_CHECK(cudaGetLastError());
            });
    };
    const bool wide = index_bytes == 8;
    const bool launched = staged ? (wide ? launch.template operator()<true, uint64_t>()
                                         : launch.template operator()<true, uint32_t>())
                                 : (wide ? launch.template operator()<false, uint64_t>()
                                         : launch.template operator()<false, uint32_t>());
    if (!launched) throw std::logic_error("FP32 AlignedK carried ARG has no kernel for this configuration.");
    if (staged) {
        launchFp32ArgWideContinuationPartials(partials, outputs, geometry.outer_size, shards_per_output,
                                              geometry.inner_size, stream);
    }
}

KernelOccupancyInfo queryFp32AlignedKCarriedArgOccupancy(uint32_t packet_bytes, uint32_t block_threads,
                                                         bool staged, DataType index_dtype) {
    if (index_dtype != DataType::UINT32 && index_dtype != DataType::UINT64)
        throw std::logic_error("FP32 AlignedK carried ARG occupancy requires a UINT32 or UINT64 carried index.");
    KernelOccupancyInfo occupancy{};
    const auto query = [&]<bool Staged, typename IndexT>() {
        return dispatchFp32AlignedKConfiguration<RKStageTopology::Complete>(
            packet_bytes, block_threads, [&]<int ItemsPerLane, int BlockThreads>() {
                const auto kernel = alignedKCarriedArgKernel<ItemsPerLane, BlockThreads, Staged, IndexT>;
                cudaFuncAttributes attributes{};
                CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
                int active_blocks = 0;
                CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks, kernel, BlockThreads, 0));
                occupancy = KernelOccupancyInfo{attributes.numRegs, attributes.sharedSizeBytes, 0, active_blocks,
                                                active_blocks * (BlockThreads / TILED_REDUCTION_WARP_THREADS)};
            });
    };
    const bool wide = index_dtype == DataType::UINT64;
    const bool queried = staged ? (wide ? query.template operator()<true, uint64_t>()
                                        : query.template operator()<true, uint32_t>())
                                : (wide ? query.template operator()<false, uint64_t>()
                                        : query.template operator()<false, uint32_t>());
    if (!queried)
        throw std::logic_error(
            "FP32 AlignedK carried ARG occupancy requires an 8- or 16-byte packet and CTA 32/64/128.");
    return occupancy;
}

KernelOccupancyInfo queryFp32AlignedKArgOccupancy(uint32_t packet_bytes, uint32_t block_threads, bool staged) {
    KernelOccupancyInfo occupancy{};
    const bool queried = queryAlignedKArgOccupancyTyped<float>(packet_bytes, block_threads, staged, occupancy);
    if (!queried)
        throw std::logic_error("FP32 AlignedK ARG occupancy requires an 8- or 16-byte packet and CTA 32/64/128.");
    return occupancy;
}

// FP8 AlignedK ARG: the support test and the launcher.
bool fp8AlignedKArgSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size,
                            uint32_t packet_bytes, uint32_t cta_threads, uint64_t shards) {
    if (outer_size == 0 || reduction_size == 0 || inner_size == 0 || shards == 0 || shards > reduction_size ||
        (packet_bytes != 2 && packet_bytes != 8) || inner_size % packet_bytes != 0 ||
        (cta_threads != 32 && cta_threads != 64 && cta_threads != 128)) {
        return false;
    }
    // Rows of one shard carry 23-bit candidate rows.
    if (ceilDivideU64(reduction_size, shards) >= FP8_ARG_MAX_LANE_ROWS) return false;
    const uint64_t tiles = ceilDivideU64(inner_size, static_cast<uint64_t>(cta_threads) * packet_bytes);
    constexpr uint64_t MaxExtent = (uint64_t{1} << 31) - 1;  // reductionWorkGrid
    return shards <= MaxExtent && tiles <= MaxExtent && outer_size <= MaxExtent;
}

void launchFp8AlignedKArgPass(const StridedSpan& input,
                              const StridedSpan* value_output,
                              const StridedSpan* index_output,
                              const ReductionGeometry& geometry,
                              uint32_t packet_bytes,
                              uint32_t cta_threads,
                              uint64_t shards,
                              bool maximum,
                              uint64_t domain_stride,
                              Stream& stream,
                              uint32_t* packed_partials) {
#if THOR_CUB_ENABLE_FP8_TYPES
    const DataType input_dtype = input.getDataType();
    const uint64_t outputs = geometry.outer_size * shards * geometry.inner_size;
    const bool packed = packed_partials != nullptr;
    if ((input_dtype != DataType::FP8_E4M3 && input_dtype != DataType::FP8_E5M2) ||
        !fp8AlignedKArgSupports(geometry.outer_size, geometry.reduction_size, geometry.inner_size, packet_bytes,
                                cta_threads, shards) ||
        input.getTotalNumElements() != geometry.outer_size * geometry.reduction_size * geometry.inner_size ||
        (packed && (value_output != nullptr || index_output != nullptr || shards < 2)) ||
        (!packed && value_output == nullptr && index_output == nullptr) ||
        (value_output != nullptr && (value_output->getTotalNumElements() != outputs ||
                                     !fp8CompleteSupportsOutputStorage(value_output->getDataType()))) ||
        (index_output != nullptr && (index_output->getTotalNumElements() != outputs ||
                                     (index_output->getDataType() != DataType::UINT32 &&
                                      index_output->getDataType() != DataType::UINT64))) ||
        (!packed && shards > 1 && (value_output == nullptr || value_output->getDataType() != DataType::FP32 ||
                                   index_output == nullptr))) {
        throw std::logic_error("FP8 AlignedK ARG received unsupported geometry or outputs.");
    }
    if ((reinterpret_cast<uintptr_t>(input.getMemPtr<void>()) & uintptr_t{packet_bytes - 1}) != 0) {
        throw std::logic_error("FP8 AlignedK ARG requires a packet-aligned input base.");
    }
    if (index_output != nullptr && index_output->getDataType() == DataType::UINT32 &&
        (geometry.reduction_size - 1) * domain_stride > std::numeric_limits<uint32_t>::max()) {
        throw std::logic_error("FP8 AlignedK ARG UINT32 index cannot hold the original-domain index.");
    }
    Fp8ArgOutputs arg_outputs{
        value_output == nullptr ? nullptr : value_output->getMemPtr<void>(),
        value_output == nullptr ? DataType::FP32 : value_output->getDataType(),
        index_output == nullptr ? nullptr : index_output->getMemPtr<void>(),
        index_output == nullptr ? DataType::UINT32 : index_output->getDataType(),
        domain_stride,
        fp8ArgFold(maximum, input_dtype == DataType::FP8_E5M2),
        input_dtype == DataType::FP8_E5M2};
    arg_outputs.packed = packed_partials;
    const uint64_t tiles = ceilDivideU64(geometry.inner_size, static_cast<uint64_t>(cta_threads) * packet_bytes);
    const ReductionShardRows rows = balancedReductionShardRows(geometry.reduction_size, shards);
    // VALUE's persistent-grid rule for tiny Complete work items (REDUCTION_PERSISTENT_MAX_BLOCKS).
    const uint64_t item_rows = static_cast<uint64_t>(rows.base_rows) + static_cast<uint64_t>(rows.extra_rows != 0);
    const uint64_t item_bytes =
        item_rows * std::min<uint64_t>(geometry.inner_size, static_cast<uint64_t>(cta_threads) * packet_bytes);
    const bool persistent = shards == 1 && item_rows <= 8 && item_bytes <= 512;
    const dim3 grid = reductionWorkGrid(shards, tiles, geometry.outer_size,
                                        persistent ? REDUCTION_PERSISTENT_MAX_BLOCKS : 0);
    const auto* bytes = static_cast<const uint8_t*>(input.getMemPtr<void>());
    if (packet_bytes == 8) {
        fp8AlignedKArgKernel<8><<<grid, cta_threads, 0, stream.getStream()>>>(
            bytes, arg_outputs, geometry.outer_size, geometry.reduction_size, geometry.inner_size, shards, tiles, rows);
    } else {
        fp8AlignedKArgKernel<2><<<grid, cta_threads, 0, stream.getStream()>>>(
            bytes, arg_outputs, geometry.outer_size, geometry.reduction_size, geometry.inner_size, shards, tiles, rows);
    }
    CUDA_CHECK(cudaGetLastError());
#else
    static_cast<void>(input);
    static_cast<void>(value_output);
    static_cast<void>(index_output);
    static_cast<void>(geometry);
    static_cast<void>(packet_bytes);
    static_cast<void>(cta_threads);
    static_cast<void>(shards);
    static_cast<void>(maximum);
    static_cast<void>(domain_stride);
    static_cast<void>(stream);
    static_cast<void>(packed_partials);
    throw std::logic_error("FP8 AlignedK ARG requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}

// FP8 ARG packed continuations: the shard-split test and layout, and the launcher.
bool fp8ArgPackedContinuationSplitsShards(uint64_t outer_size, uint64_t inner_size, uint64_t shards) {
    if (shards < FP8_ARG_PACKED_SPLIT_MIN_SHARDS || inner_size == 0 ||
        outer_size > std::numeric_limits<uint64_t>::max() / inner_size) {
        return false;
    }
    const uint64_t outputs = outer_size * inner_size;
    if (outputs >= FP8_ARG_PACKED_SPLIT_MAX_OUTPUTS &&
        (shards < FP8_ARG_PACKED_SPLIT_WIDE_MIN_SHARDS || outputs >= FP8_ARG_PACKED_SPLIT_WIDE_MAX_OUTPUTS)) {
        return false;
    }
    const Fp8ArgPackedSplitLayout layout = fp8ArgPackedSplitLayout(inner_size, shards);
    return uint64_t{layout.columns} * layout.groups >= 32;
}

Fp8ArgPackedSplitLayout fp8ArgPackedSplitLayout(uint64_t inner_size, uint64_t shards) {
    Fp8ArgPackedSplitLayout layout;
    layout.columns = inner_size < 32 ? static_cast<uint32_t>(inner_size) : 32U;
    // At least four shards per group (one batch of loads in flight), at most 1024 threads.
    const uint64_t by_shards = std::max<uint64_t>(1, ceilDivideU64(shards, 4));
    layout.groups = static_cast<uint32_t>(std::min<uint64_t>(1024U / layout.columns, by_shards));
    layout.threads = static_cast<uint32_t>(ceilDivideU64(uint64_t{layout.columns} * layout.groups, 32) * 32);
    return layout;
}

void launchFp8ArgPackedContinuation(const uint32_t* packed_partials,
                                    const StridedSpan* value_output,
                                    const StridedSpan* index_output,
                                    DataType input_dtype,
                                    uint64_t outer_size,
                                    uint64_t reduction_size,
                                    uint64_t inner_size,
                                    uint64_t shards,
                                    uint64_t uniform_rows_per_shard,
                                    bool maximum,
                                    uint64_t domain_stride,
                                    Stream& stream) {
#if THOR_CUB_ENABLE_FP8_TYPES
    const uint64_t outputs = outer_size * inner_size;
    if ((input_dtype != DataType::FP8_E4M3 && input_dtype != DataType::FP8_E5M2) || packed_partials == nullptr ||
        // 55-bit widened indices; 32-bit shard loop.
        outputs == 0 || shards < 2 || shards > reduction_size || shards > REDUCTION_WORK_GRID_MAX_EXTENT ||
        reduction_size - 1 > FP8_ARG_MAX_WIDE_INDEX ||
        (uniform_rows_per_shard != 0 && shards != ceilDivideU64(reduction_size, uniform_rows_per_shard)) ||
        (value_output == nullptr && index_output == nullptr) ||
        (value_output != nullptr && (value_output->getTotalNumElements() != outputs ||
                                     !fp8CompleteSupportsOutputStorage(value_output->getDataType()))) ||
        (index_output != nullptr && (index_output->getTotalNumElements() != outputs ||
                                     (index_output->getDataType() != DataType::UINT32 &&
                                      index_output->getDataType() != DataType::UINT64)))) {
        throw std::logic_error("FP8 packed ARG continuation received unsupported geometry or outputs.");
    }
    if (index_output != nullptr && index_output->getDataType() == DataType::UINT32 &&
        (reduction_size - 1) * domain_stride > std::numeric_limits<uint32_t>::max()) {
        throw std::logic_error("FP8 packed ARG continuation UINT32 index cannot hold the original-domain index.");
    }
    const Fp8ArgOutputs arg_outputs{
        value_output == nullptr ? nullptr : value_output->getMemPtr<void>(),
        value_output == nullptr ? DataType::FP32 : value_output->getDataType(),
        index_output == nullptr ? nullptr : index_output->getMemPtr<void>(),
        index_output == nullptr ? DataType::UINT32 : index_output->getDataType(),
        domain_stride,
        fp8ArgFold(maximum, input_dtype == DataType::FP8_E5M2),
        input_dtype == DataType::FP8_E5M2};
    // Uniform depths (DeepRSmallK, rotated RuntimeK): shard s begins at s * depth. Balanced (AlignedK): VALUE's split.
    const ReductionShardRows rows = uniform_rows_per_shard != 0
                                        ? ReductionShardRows{static_cast<uint32_t>(uniform_rows_per_shard), 0U}
                                        : balancedReductionShardRows(reduction_size, shards);
    if (fp8ArgPackedContinuationSplitsShards(outer_size, inner_size, shards)) {
        const Fp8ArgPackedSplitLayout layout = fp8ArgPackedSplitLayout(inner_size, shards);
        const uint64_t items = outer_size * ceilDivideU64(inner_size, layout.columns);
        const uint64_t blocks = std::min<uint64_t>(items, PERSISTENT_MAX_GRID_BLOCKS);
        fp8ArgPackedContinuationSplitKernel<0><<<static_cast<unsigned int>(blocks), layout.threads, 0,
                                                 stream.getStream()>>>(packed_partials, arg_outputs, outer_size,
                                                                       shards, inner_size, rows, layout.columns,
                                                                       layout.groups);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    constexpr uint32_t Threads = 256;
    const uint64_t blocks = std::min<uint64_t>(ceilDivideU64(outputs, Threads), PERSISTENT_MAX_GRID_BLOCKS);
    fp8ArgPackedContinuationKernel<0><<<static_cast<unsigned int>(blocks), Threads, 0, stream.getStream()>>>(
        packed_partials, arg_outputs, outer_size, shards, inner_size, rows);
    CUDA_CHECK(cudaGetLastError());
#else
    static_cast<void>(packed_partials);
    static_cast<void>(value_output);
    static_cast<void>(index_output);
    static_cast<void>(input_dtype);
    static_cast<void>(outer_size);
    static_cast<void>(reduction_size);
    static_cast<void>(inner_size);
    static_cast<void>(shards);
    static_cast<void>(uniform_rows_per_shard);
    static_cast<void>(maximum);
    static_cast<void>(domain_stride);
    static_cast<void>(stream);
    throw std::logic_error("FP8 packed ARG continuation requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}

}  // namespace ThorImplementation::ReductionInternal
