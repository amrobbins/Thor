#include "Utilities/TensorOperations/Reduction/Kernels/KernelLaunchers.h"
#include "Utilities/TensorOperations/Reduction/Kernels/ReductionOperation.cuh"
#include "Utilities/TensorOperations/Reduction/Arg/Fp8ArgKey.h"
#include "Utilities/TensorOperations/Reduction/Arg/Fp32ArgKey.cuh"
#include "Utilities/TensorOperations/Reduction/Kernels/KernelCommon.cuh"
#include "Utilities/TensorOperations/Reduction/RK/Fp8/Fp8RuntimeKSelector.h"

#include "Utilities/Exceptions.h"

#include <cuda/std/bit>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace ThorImplementation::ReductionInternal {
namespace {

// RuntimeK rotated: the deep-R, low-output value reduction at a K that is not a multiple of the packet. It shards both
// R and K, and one cooperative warp group owns each (R shard, K tile), so no few CTAs walk all of R alone.
//
// Such a K shifts successive row starts relative to packet boundaries. Rows owned by one warp differ by one
// cooperative-group width in K strides, so the warp aligns its row window once and every revisited row stays
// packet-aligned. FP8 runs 16-byte packets with sixteen warps, PrefixReserved (497 logical components) or Exact (512,
// with distributed 4-byte tail ownership); FP16 / BF16 and FP32 run 16-byte packets with eight and four warps (see
// their launchers). Packets are decoded immediately (FP8 in native fp8x2 pairs) and accumulated in FP32 registers;
// shared memory carries only the post-R-reduction FP32 warp partials, realigned by logical component before coalesced
// FP32 partial stores.
constexpr int RUNTIME_K_ROTATED_WARPS = 8;
constexpr int RUNTIME_K_ROTATED_THREADS =
    RUNTIME_K_ROTATED_WARPS * TILED_REDUCTION_WARP_THREADS;
constexpr int FP8_RUNTIME_K_ROTATED_WARPS = 16;
constexpr int FP8_RUNTIME_K_ROTATED_THREADS =
    FP8_RUNTIME_K_ROTATED_WARPS * TILED_REDUCTION_WARP_THREADS;
static_assert(RUNTIME_K_ROTATED_THREADS == 256);
static_assert(FP8_RUNTIME_K_ROTATED_THREADS == 512);

template <typename InputT>
inline constexpr int runtimeKRotatedCooperativeWarps = sizeof(InputT) == 1 ? FP8_RUNTIME_K_ROTATED_WARPS
                                                                    : RUNTIME_K_ROTATED_WARPS;

template <typename InputT, int PacketBytes>
inline constexpr int runtimeKRotatedItemsPerLaneForPacket = PacketBytes / static_cast<int>(sizeof(InputT));

template <typename InputT, int PacketBytes>
inline constexpr int runtimeKRotatedPacketSpanElementsForPacket =
    TILED_REDUCTION_WARP_THREADS * runtimeKRotatedItemsPerLaneForPacket<InputT, PacketBytes>;

template <typename InputT, int PacketBytes, bool ExactFp8Tile = true>
inline constexpr int runtimeKRotatedLogicalTileElementsForPacket =
    sizeof(InputT) == 1 && ExactFp8Tile
        ? runtimeKRotatedPacketSpanElementsForPacket<InputT, PacketBytes>
        : runtimeKRotatedPacketSpanElementsForPacket<InputT, PacketBytes>
              - (runtimeKRotatedItemsPerLaneForPacket<InputT, PacketBytes> - 1);

template <int PacketBytes, typename InputT>
[[nodiscard]] __device__ inline const InputT* alignPacketWindowDown(const InputT* source) {
    static_assert(PacketBytes == 4 || PacketBytes == 8 || PacketBytes == 16);
    const uintptr_t address = reinterpret_cast<uintptr_t>(source);
    return reinterpret_cast<const InputT*>(address & ~uintptr_t{PacketBytes - 1});
}

template <typename ReductionOpT>
struct RuntimeKRotatedReductionIdentityFp32;

template <>
struct RuntimeKRotatedReductionIdentityFp32<cuda::std::plus<float>> {
    static constexpr float value = 0.0f;
};

template <>
struct RuntimeKRotatedReductionIdentityFp32<cuda::std::multiplies<float>> {
    static constexpr float value = 1.0f;
};

template <>
struct RuntimeKRotatedReductionIdentityFp32<PropagatingMinimumFp32> {
    static constexpr float value = std::numeric_limits<float>::infinity();
};

template <>
struct RuntimeKRotatedReductionIdentityFp32<PropagatingMaximumFp32> {
    static constexpr float value = -std::numeric_limits<float>::infinity();
};

// Occupancy contract for the rotated first stage. FP8 PrefixReserved (16 warps, 512 threads, 33,856 B static shared
// including the 1 KiB per-block reserve) fits three blocks = 48 warps per SM only at <= 40 registers/thread; at 41..64
// it drops to two. Unpinned it compiled at 43 registers, 15-23% slower on one-tile shapes, so the 40-register budget is
// pinned here. FP8 Exact compiles at exactly 64 registers (two blocks per SM; 65+ leaves one). Its MIN/MAX kernels,
// which carry the transform fold (Fp8ExtremumFold), are pinned to two blocks: an unpinned fold went to 67 registers and
// 1.47x slower. FP8 Exact SUM/PRODUCT and the 8-warp FP16/BF16/FP32 kernels keep their natural allocation; maxThreads
// is their exact launch block size.
template <typename InputT, bool ExactFp8Tile, typename ReductionOpT = cuda::std::plus<float>>
inline constexpr int runtimeKRotatedMinBlocksPerSm =
    sizeof(InputT) != 1 ? 1
    : !ExactFp8Tile     ? 3
    : (std::is_same_v<ReductionOpT, PropagatingMinimumFp32> || std::is_same_v<ReductionOpT, PropagatingMaximumFp32>)
        ? 2
        : 1;

template <typename InputT,
          typename ReductionOpT,
          typename InputTransformT,
          int RowsPerShard,
          int PacketBytes,
          int CooperativeWarps,
          bool ExactFp8Tile = true>
__global__ __launch_bounds__(CooperativeWarps * TILED_REDUCTION_WARP_THREADS,
                             (runtimeKRotatedMinBlocksPerSm<InputT, ExactFp8Tile, ReductionOpT>))
void runtimeKRotatedValueKernel(const InputT* input,
                                                                float* fp32_partials,
                                                                uint64_t outer_size,
                                                                uint64_t reduction_size,
                                                                uint64_t inner_size,
                                                                uint64_t shards_per_output,
                                                                uint64_t component_tiles,
                                                                Fp8ExtremumFold extremum_fold) {
    // FP8 MIN/MAX (see Fp8ExtremumFold) fold the ABS/SQUARE transform into the FP8 bytes. PrefixReserved also folds the
    // MAX sign (its MIN kernel serves MAX); Exact keeps separate MIN and MAX kernels and folds only the transform (no
    // XOR operand, see runtimeKRotatedMinBlocksPerSm). The stored FP32 partial is unfolded, so staged partials are
    // those of the requested operation.
    constexpr bool Fp8MinimumOp = sizeof(InputT) == 1 && std::is_same_v<ReductionOpT, PropagatingMinimumFp32>;
    constexpr bool Fp8MaximumOp = sizeof(InputT) == 1 && std::is_same_v<ReductionOpT, PropagatingMaximumFp32>;
    constexpr bool FoldExtremum = Fp8MinimumOp || (ExactFp8Tile && Fp8MaximumOp);
    constexpr bool SignFold = !ExactFp8Tile;
    static_assert(!FoldExtremum || std::is_same_v<InputTransformT, IdentityFp32>);
    static_assert(!Fp8MaximumOp || ExactFp8Tile, "FP8 PrefixReserved MAX launches the folded MIN kernel.");
    static_assert(RowsPerShard == 8 || RowsPerShard == 16 || RowsPerShard == 32 || RowsPerShard == 64
                  || RowsPerShard == 128 || RowsPerShard == 256 || RowsPerShard == 512 || RowsPerShard == 1024
                  || RowsPerShard == 2048 || RowsPerShard == 4096 || RowsPerShard == 8192
                  || RowsPerShard == 16384);
    constexpr int ItemsPerLane = runtimeKRotatedItemsPerLaneForPacket<InputT, PacketBytes>;
    constexpr int PacketSpanElements = runtimeKRotatedPacketSpanElementsForPacket<InputT, PacketBytes>;
    constexpr int LogicalTileElements = runtimeKRotatedLogicalTileElementsForPacket<InputT, PacketBytes, ExactFp8Tile>;
    ReductionOpT reduction_op{};
    const float Init = [&] {
        if constexpr (requires { reduction_op.identity(); }) return reduction_op.identity();
        else return RuntimeKRotatedReductionIdentityFp32<ReductionOpT>::value;
    }();
    static_assert(sizeof(InputT) == 1 || sizeof(InputT) == 2 || sizeof(InputT) == 4);
    static_assert(PacketBytes == 4 || PacketBytes == 8 || PacketBytes == 16);
    static_assert(PacketBytes % sizeof(InputT) == 0);
    static_assert(sizeof(InputT) * ItemsPerLane == PacketBytes);
    static_assert(CooperativeWarps == 4 || CooperativeWarps == 8 || CooperativeWarps == 16);
    // Revisited rows differ by CooperativeWarps*K elements. Requiring the cooperative storage-width stride to be an
    // integer packet guarantees every warp can align once and keep issuing naturally aligned packets for arbitrary K.
    static_assert((CooperativeWarps * sizeof(InputT)) % PacketBytes == 0);
    static_assert(CooperativeWarps * TILED_REDUCTION_WARP_THREADS >= LogicalTileElements);
    static_assert(sizeof(InputT) != 1
                  || CooperativeWarps * TILED_REDUCTION_WARP_THREADS == PacketSpanElements);
    static_assert(LogicalTileElements > 0);

    __shared__ float warp_partials[CooperativeWarps][PacketSpanElements];
    __shared__ uint32_t warp_prefix_elements[CooperativeWarps];

    const InputTransformT input_transform{};
    const int warp = static_cast<int>(threadIdx.x) / TILED_REDUCTION_WARP_THREADS;
    const int lane = static_cast<int>(threadIdx.x) % TILED_REDUCTION_WARP_THREADS;
    // Grid: x = shard, y = component tile, z = outer (see reductionWorkGrid). The launcher computes the component tile
    // count; rows inside one shard (<= RowsPerShard) use 32-bit counters and a 64-bit row pointer.
    // reductionWorkGrid() limits every work extent to 2^31 - 1, so 32-bit work coordinates never wrap.
    const uint32_t outer_count = static_cast<uint32_t>(outer_size);
    const uint32_t tile_count = static_cast<uint32_t>(component_tiles);
    const uint32_t shard_count = static_cast<uint32_t>(shards_per_output);
    for (uint32_t outer_index = blockIdx.z; outer_index < outer_count; outer_index += gridDim.z)
    for (uint32_t component_tile = blockIdx.y; component_tile < tile_count; component_tile += gridDim.y)
    for (uint32_t shard = blockIdx.x; shard < shard_count; shard += gridDim.x) {
        const uint64_t tile_begin = component_tile * static_cast<uint64_t>(LogicalTileElements);
        const uint64_t valid_components =
            minU64(static_cast<uint64_t>(LogicalTileElements), inner_size - tile_begin);
        const uint64_t row_begin = shard * static_cast<uint64_t>(RowsPerShard);
        const uint32_t shard_rows = row_begin < reduction_size
            ? static_cast<uint32_t>(minU64(static_cast<uint64_t>(RowsPerShard), reduction_size - row_begin))
            : 0U;
        const uint32_t first_row = static_cast<uint32_t>(warp);

        float local[ItemsPerLane];
#pragma unroll
        for (int item = 0; item < ItemsPerLane; ++item) {
            local[item] = Init;
        }

        // FP8 exact-tile tails are distributed over one 4-byte chunk per participating lane instead of serializing
        // a full extra packet through lane 0. For the calibrated p16 Exact layout, the first four lanes each own one
        // four-byte tail chunk. The compiler removes this storage entirely for PrefixReserved and non-FP8 kernels.
        float exact_tail_local[4] = {Init, Init, Init, Init};
        uint32_t exact_tail_elements = 0;
        uint32_t prefix_elements = 0;
        if (first_row < shard_rows) {
            const InputT* logical_row_tile =
                input + (outer_index * reduction_size + row_begin + first_row) * inner_size + tile_begin;
            const uintptr_t logical_address = reinterpret_cast<uintptr_t>(logical_row_tile);
            prefix_elements =
                static_cast<uint32_t>((logical_address & uintptr_t{PacketBytes - 1}) / sizeof(InputT));
            const InputT* aligned_row_window = alignPacketWindowDown<PacketBytes>(logical_row_tile);

            constexpr uint32_t WarpRows = static_cast<uint32_t>(CooperativeWarps);
            const uint64_t row_stride_elements = static_cast<uint64_t>(WarpRows) * inner_size;
            if constexpr (sizeof(InputT) == 1 && ExactFp8Tile) {
                // Compute tail demand before per-lane packet bounds so it remains warp-uniform. Tail owners are always
                // low lanes, but the synchronization used for shared-memory overwrite must be reached by the full warp.
                const uint64_t exact_tail_begin = static_cast<uint64_t>(PacketSpanElements)
                                                  - static_cast<uint64_t>(prefix_elements);
                exact_tail_elements =
                    exact_tail_begin < valid_components
                        ? static_cast<uint32_t>(minU64(static_cast<uint64_t>(prefix_elements),
                                                       valid_components - exact_tail_begin))
                        : 0U;
            }
            const int64_t lane_packet_component =
                static_cast<int64_t>(tile_begin) - static_cast<int64_t>(prefix_elements)
                + static_cast<int64_t>(lane * ItemsPerLane);

            if (lane_packet_component < static_cast<int64_t>(inner_size)) {
                const InputT* lane_packet = aligned_row_window + lane * ItemsPerLane;
                if constexpr (sizeof(InputT) == 1 && ExactFp8Tile) {
                    // FP8 Exact keeps the full packet-span logical tile (512 components for the calibrated p16 layout). The
                    // ordinary 32 packets cover [tile_begin-prefix, tile_begin+PacketSpan-prefix). If that already
                    // contains every valid logical component, run the original branchless packet loop unchanged.
                    // Otherwise the low tail-owner lanes each load one naturally aligned 4-byte tail chunk
                    // and accumulate four tail components in parallel. Those partials overwrite the otherwise-invalid
                    // prefix slots in shared memory below, so logical readback can still treat the packet span as a
                    // circular window. The normal packet loop never carries per-item prefix/tail predicates.
                    constexpr int TailItemsPerOwnerLane = 4;
                    constexpr int TailOwnerLanes = PacketBytes / TailItemsPerOwnerLane;
                    static_assert(TailOwnerLanes == 1 || TailOwnerLanes == 2 || TailOwnerLanes == 4);

                    if (exact_tail_elements == 0) {
                        // Established rotated fast path: one packet per lane, no exact-tail bookkeeping in the R loop.
                        for (uint32_t row = first_row; row < shard_rows; row += WarpRows) {
                            const PackedInputValues<InputT, ItemsPerLane> values = foldFp8ExtremumPacket<FoldExtremum, SignFold>(
                                loadVectorizedInputPacket<InputT, ItemsPerLane>(lane_packet), extremum_fold);
                            reducePackedFp8ValuesToFp32<InputT>(local, values, reduction_op, input_transform);
                            lane_packet += row_stride_elements;
                        }
                    } else {
                        const InputT* extra_tail_packet = aligned_row_window + PacketSpanElements;
                        for (uint32_t row = first_row; row < shard_rows; row += WarpRows) {
                            const PackedInputValues<InputT, ItemsPerLane> values = foldFp8ExtremumPacket<FoldExtremum, SignFold>(
                                loadVectorizedInputPacket<InputT, ItemsPerLane>(lane_packet), extremum_fold);
                            reducePackedFp8ValuesToFp32<InputT>(local, values, reduction_op, input_transform);

                            if (lane < TailOwnerLanes) {
                                const PackedInputValues<InputT, TailItemsPerOwnerLane> tail_values =
                                    foldFp8ExtremumPacket<FoldExtremum, SignFold>(
                                        loadVectorizedInputPacket<InputT, TailItemsPerOwnerLane>(
                                            extra_tail_packet + lane * TailItemsPerOwnerLane),
                                        extremum_fold);
#pragma unroll
                                for (int item = 0; item < TailItemsPerOwnerLane; item += 2) {
                                    const uint16_t packed = static_cast<uint16_t>(tail_values.values[item].__x)
                                                            | (static_cast<uint16_t>(tail_values.values[item + 1].__x)
                                                               << 8);
                                    const float2 decoded = decodeFp8x2ToFp32<InputT>(packed);
                                    const uint32_t first_tail_component =
                                        static_cast<uint32_t>(lane * TailItemsPerOwnerLane + item);
                                    if (first_tail_component < exact_tail_elements) {
                                        exact_tail_local[item] = reduction_op(
                                            exact_tail_local[item], input_transform(decoded.x));
                                    }
                                    if (first_tail_component + 1 < exact_tail_elements) {
                                        exact_tail_local[item + 1] = reduction_op(
                                            exact_tail_local[item + 1], input_transform(decoded.y));
                                    }
                                }
                            }

                            lane_packet += row_stride_elements;
                            extra_tail_packet += row_stride_elements;
                        }
                    }
                } else if constexpr (sizeof(InputT) == 1) {
                    // PrefixReserved FP8 keeps the same rotated ownership and register accumulation, but decodes each
                    // loaded p4/p8/p16 packet in native fp8x2 pairs before applying the input transform.
                    for (uint32_t row = first_row; row < shard_rows; row += WarpRows) {
                        const PackedInputValues<InputT, ItemsPerLane> values = foldFp8ExtremumPacket<FoldExtremum, SignFold>(
                            loadVectorizedInputPacket<InputT, ItemsPerLane>(lane_packet), extremum_fold);
                        reducePackedFp8ValuesToFp32<InputT>(local, values, reduction_op, input_transform);
                        lane_packet += row_stride_elements;
                    }
                } else {
                    // Preserve the established FP16/BF16/FP32 rotated kernel exactly: one aligned packet per lane and
                    // the prefix-reserved logical tile width.
                    for (uint32_t row = first_row; row < shard_rows; row += WarpRows) {
                        const PackedInputValues<InputT, ItemsPerLane> values =
                            loadVectorizedInputPacket<InputT, ItemsPerLane>(lane_packet);
#pragma unroll
                        for (int item = 0; item < ItemsPerLane; ++item) {
                            local[item] = reduction_op(
                                local[item], input_transform(ToFp32<InputT>{}(values.values[item])));
                        }
                        lane_packet += row_stride_elements;
                    }
                }
            }
        }

        if (lane == 0) {
            warp_prefix_elements[warp] = prefix_elements;
        }
#pragma unroll
        for (int item = 0; item < ItemsPerLane; ++item) {
            warp_partials[warp][lane * ItemsPerLane + item] = local[item];
        }
        if constexpr (sizeof(InputT) == 1 && ExactFp8Tile) {
            constexpr int TailItemsPerOwnerLane = 4;
            constexpr int TailOwnerLanes = PacketBytes / TailItemsPerOwnerLane;
            if (exact_tail_elements != 0) {
                // The regular lane-0 prefix slots are outside the logical tile. Once every lane has published its
                // ordinary packet partials, tail owners replace exactly those slots with the missing logical tail.
                // __syncwarp orders the overwrite before the block-wide cooperative reduction consumes shared memory.
                __syncwarp();
                if (lane < TailOwnerLanes) {
#pragma unroll
                    for (int item = 0; item < TailItemsPerOwnerLane; ++item) {
                        const uint32_t tail_component =
                            static_cast<uint32_t>(lane * TailItemsPerOwnerLane + item);
                        if (tail_component < exact_tail_elements) {
                            warp_partials[warp][tail_component] = exact_tail_local[item];
                        }
                    }
                }
            }
        }
        __syncthreads();

        const uint64_t logical_component = static_cast<uint64_t>(threadIdx.x);
        if (logical_component < valid_components) {
            float reduced = Init;
#pragma unroll
            for (int source_warp = 0; source_warp < CooperativeWarps; ++source_warp) {
                uint32_t physical_component =
                    static_cast<uint32_t>(logical_component) + warp_prefix_elements[source_warp];
                if constexpr (sizeof(InputT) == 1 && ExactFp8Tile) {
                    static_assert((PacketSpanElements & (PacketSpanElements - 1)) == 0);
                    physical_component &= static_cast<uint32_t>(PacketSpanElements - 1);
                }
                reduced = reduction_op(reduced, warp_partials[source_warp][physical_component]);
            }
            fp32_partials[(outer_index * shards_per_output + shard) * inner_size + tile_begin + logical_component] =
                unfoldFp8ExtremumPartial<FoldExtremum, SignFold>(reduced, extremum_fold);
        }
        __syncthreads();
    }
}

template <typename InputT,
          typename ReductionOpT,
          typename InputTransformT,
          int RowsPerShard,
          int PacketBytes,
          int CooperativeWarps,
          bool ExactFp8Tile = true>
void launchRuntimeKRotatedFirstStageForTransformPacket(const StridedSpan& input,
                                                       const StridedSpan& fp32_partials,
                                                       const ReductionGeometry& geometry,
                                                       uint64_t shards_per_output,
                                                       cudaStream_t stream,
                                                       Fp8ExtremumFold extremum_fold = FP8_EXTREMUM_FOLD_NONE) {
    constexpr int LogicalTileElements = runtimeKRotatedLogicalTileElementsForPacket<InputT, PacketBytes, ExactFp8Tile>;
    const uint64_t component_tiles =
        ceilDivideU64(geometry.inner_size, static_cast<uint64_t>(LogicalTileElements));
    if (geometry.outer_size == 0 || component_tiles == 0 || shards_per_output == 0) {
        throw std::logic_error("Awkward rotated sharded first stage requires non-zero work.");
    }
    // Division-free launch: x = shard, y = component tile, z = outer.
    const dim3 grid_blocks = reductionWorkGrid(shards_per_output, component_tiles, geometry.outer_size);

    runtimeKRotatedValueKernel<InputT,
                                                     ReductionOpT,
                                                     InputTransformT,
                                                     RowsPerShard,
                                                     PacketBytes,
                                                     CooperativeWarps,
                                                     ExactFp8Tile>
        <<<grid_blocks, CooperativeWarps * TILED_REDUCTION_WARP_THREADS, 0, stream>>>(input.getMemPtr<InputT>(),
                                                                      fp32_partials.getMemPtr<float>(),
                                                                      geometry.outer_size,
                                                                      geometry.reduction_size,
                                                                      geometry.inner_size,
                                                                      shards_per_output,
                                                                      component_tiles,
                                                                      extremum_fold);
    CUDA_CHECK(cudaGetLastError());
}

// PacketComplete, RuntimeK's Complete topology over K tiles (the large-K continuation of OneWindow at O1): one work
// item owns [outer, k_tile], K tiles are mathematically independent, and W1/W2/W4/W8 cooperate only over R.
// Total K and tile width are runtime values; only the p16 physical packet shape is specialized. A tile is capped at
// 496 logical bytes so an arbitrary per-row prefix (0..15 bytes) still fits in one 32-lane p16 packet window.
// Pinned to four 256-thread blocks per SM (<= 64 registers): 65+ would lose a block, and RuntimeK SUM measured up to
// 1.9x slower at 80 registers. With the row load-ahead it compiles at 56 or fewer, no spills (0.79x the time of the
// loop without it, at 62 registers).
// Shared with FP32 input. The mapping is in components: a p16 packet holds 4 FP32 components, so the staged window is
// 128 components and an FP32 tile is capped at 124 (FP8's 496 is 31 packets; 124 is 31 FP32 packets), four slots per
// lane. FP32 reads its components from the staged words directly (no byte gather or packed conversion); MIN/MAX combine
// the raw values (no fold).
// FP16 / BF16 input: 8 components per packet, a 256-component staged window and a tile cap of 248 (31 packets), eight
// slots per lane. Each component is read as its half-word of the staged word and widened to FP32 (widenF16Bits);
// MIN/MAX combine the widened values, as FP32.
// Rows are read at `row_stride` and slabs at `outer_stride` elements (K and R x K for a dense input; a pitched view's
// row and slab pitches), since each row is loaded from its own address. The output stays dense [outer, K].
template <typename InputT, int FixedInputTransform = 0, int FixedOp = 0>
__global__ __launch_bounds__(SHALLOW_COMPLETE_THREADS, 4) void packetCompleteValueKernel(const InputT* input,
                                                          void* output,
                                                          DataType output_dtype,
                                                          uint32_t outer_size,
                                                          uint32_t reduction_size,
                                                          uint32_t inner_size,
                                                          uint64_t row_stride,
                                                          uint64_t outer_stride,
                                                          uint32_t tile_components,
                                                          uint32_t tile_count,
                                                          uint32_t warps_per_output,
                                                          AdditiveFinalizeFp32 output_finalize,
                                                          ReductionStageInputTransform input_transform,
                                                          float output_scale) {
    constexpr bool Fp32Input = std::is_same_v<InputT, float>;
    constexpr bool F16Input = isF16InputType<InputT>;
    static_assert(sizeof(InputT) == 1 || Fp32Input || F16Input);
    static_assert(FixedInputTransform >= 0 && FixedInputTransform <= 2,
                  "RuntimeK Complete requires a compile-time input transform.");
    constexpr uint32_t PacketBytes = 16;
    // Components per packet (16 FP8 / 8 FP16 / 4 FP32); prefixes, packet counts and offsets below are in components.
    constexpr uint32_t PacketShift = Fp32Input ? 2 : (F16Input ? 3 : 4);
    constexpr uint32_t PacketMask = (PacketBytes / sizeof(InputT)) - 1;
    constexpr uint32_t PacketWords = 4;
    constexpr uint32_t MaxTileComponents = Fp32Input ? 124 : (F16Input ? 248 : 496);
    constexpr uint32_t MaxComponentsPerLane =
        (MaxTileComponents + TILED_REDUCTION_WARP_THREADS - 1) / TILED_REDUCTION_WARP_THREADS;
    static_assert(MaxComponentsPerLane == (Fp32Input ? 4 : (F16Input ? 8 : 16)));
    static_assert((1U << PacketShift) == PacketMask + 1);

    const Fp8TypedReductionOp<FixedOp> combine{};
    const float initial = combine.identity();
    __shared__ uint32_t staged_packet_words[SHALLOW_COMPLETE_WARPS]
                                            [PacketWords * TILED_REDUCTION_WARP_THREADS];
    __shared__ float warp_partials[SHALLOW_COMPLETE_WARPS][MaxTileComponents];

    const uint32_t warp = threadIdx.x / TILED_REDUCTION_WARP_THREADS;
    const uint32_t lane = threadIdx.x % TILED_REDUCTION_WARP_THREADS;
    const uint32_t outputs_per_block = SHALLOW_COMPLETE_WARPS / warps_per_output;
    const uint32_t output_group = warp / warps_per_output;
    const uint32_t warp_in_output = warp - output_group * warps_per_output;
    const uint32_t first_warp_in_output = output_group * warps_per_output;
    const uint64_t work_count = static_cast<uint64_t>(outer_size) * tile_count;
    const uint64_t block_work_stride = static_cast<uint64_t>(gridDim.x) * outputs_per_block;
    // The base's place in its 16-byte packet, in components (0 for an aligned base). Prefixes and packet addresses
    // follow the input's address, not its element index, so the kernel reads any element address: a row's first packet
    // starts at most 15 bytes before the base, inside the base's own 16-byte granule.
    const uint32_t base_shift = static_cast<uint32_t>(
        (reinterpret_cast<uintptr_t>(input) & uintptr_t{PacketBytes - 1}) / sizeof(InputT));
    const uint8_t* const aligned_input = reinterpret_cast<const uint8_t*>(input) - base_shift * sizeof(InputT);

    for (uint64_t block_work_begin = static_cast<uint64_t>(blockIdx.x) * outputs_per_block;
         block_work_begin < work_count;
         block_work_begin += block_work_stride) {
        const uint64_t work_index = block_work_begin + output_group;
        const bool valid_work = work_index < work_count;
        uint32_t outer_index = 0;
        uint32_t tile_index = 0;
        uint32_t tile_begin = 0;
        uint32_t tile_width = 0;
        if (valid_work) {
            outer_index = static_cast<uint32_t>(work_index / tile_count);
            tile_index = static_cast<uint32_t>(work_index - static_cast<uint64_t>(outer_index) * tile_count);
            tile_begin = tile_index * tile_components;
            tile_width = tile_components < (inner_size - tile_begin) ? tile_components : (inner_size - tile_begin);
        }

        float local[MaxComponentsPerLane];
#pragma unroll
        for (uint32_t slot = 0; slot < MaxComponentsPerLane; ++slot) {
            local[slot] = initial;
        }

        // Decodes the staged row (tile_width components after prefix_bytes alignment components) into local[].
        const auto decode_row = [&](uint32_t prefix_bytes) {
            if constexpr (Fp32Input) {
                // Component c sits at staged position prefix + c: packet (position >> 2), word (position & 3).
                if (lane < tile_width) {
                    float gathered[4];
#pragma unroll
                    for (uint32_t slot = 0; slot < 4; ++slot) {
                        const uint32_t component = lane + slot * TILED_REDUCTION_WARP_THREADS;
                        const uint32_t staged = prefix_bytes + component;
                        gathered[slot] =
                            component < tile_width
                                ? __uint_as_float(staged_packet_words[warp][(staged & PacketMask) *
                                                                                TILED_REDUCTION_WARP_THREADS +
                                                                            (staged >> PacketShift)])
                                : 0.0f;
                    }
                    const float4 decoded = transformFp8CompleteDecodedPacket<FixedInputTransform>(
                        make_float4(gathered[0], gathered[1], gathered[2], gathered[3]), input_transform);
                    local[0] = combine(local[0], decoded.x);
                    if (lane + TILED_REDUCTION_WARP_THREADS < tile_width) {
                        local[1] = combine(local[1], decoded.y);
                    }
                    if (lane + 2U * TILED_REDUCTION_WARP_THREADS < tile_width) {
                        local[2] = combine(local[2], decoded.z);
                    }
                    if (lane + 3U * TILED_REDUCTION_WARP_THREADS < tile_width) {
                        local[3] = combine(local[3], decoded.w);
                    }
                }
            } else if constexpr (F16Input) {
                // FP16 / BF16: component c sits at staged position prefix + c: packet (position >> 3), word ((position
                // & 7) >> 1), half-word (position & 1). Four components per transform, as FP8.
#pragma unroll
                for (uint32_t slot_base = 0; slot_base < MaxComponentsPerLane; slot_base += 4) {
                    const uint32_t first_component = lane + slot_base * TILED_REDUCTION_WARP_THREADS;
                    if (first_component < tile_width) {
                        float gathered[4];
#pragma unroll
                        for (uint32_t packed_slot = 0; packed_slot < 4; ++packed_slot) {
                            const uint32_t component = lane + (slot_base + packed_slot) * TILED_REDUCTION_WARP_THREADS;
                            gathered[packed_slot] = 0.0f;
                            if (component < tile_width) {  // the window holds positions < prefix + tile_width only
                                const uint32_t staged = prefix_bytes + component;
                                const uint32_t word =
                                    staged_packet_words[warp][((staged & PacketMask) >> 1) *
                                                                  TILED_REDUCTION_WARP_THREADS +
                                                              (staged >> PacketShift)];
                                gathered[packed_slot] = widenF16Bits<InputT>((staged & 1U) != 0 ? word >> 16 : word);
                            }
                        }
                        const float4 decoded = transformFp8CompleteDecodedPacket<FixedInputTransform>(
                            make_float4(gathered[0], gathered[1], gathered[2], gathered[3]), input_transform);
                        local[slot_base] = combine(local[slot_base], decoded.x);
                        if (first_component + TILED_REDUCTION_WARP_THREADS < tile_width) {
                            local[slot_base + 1] = combine(local[slot_base + 1], decoded.y);
                        }
                        if (first_component + 2U * TILED_REDUCTION_WARP_THREADS < tile_width) {
                            local[slot_base + 2] = combine(local[slot_base + 2], decoded.z);
                        }
                        if (first_component + 3U * TILED_REDUCTION_WARP_THREADS < tile_width) {
                            local[slot_base + 3] = combine(local[slot_base + 3], decoded.w);
                        }
                    }
                }
            } else {
                // Packet loading is lane-contiguous, while accumulation is component-owned across the warp. Gather four
                // component bytes from the staged packet transpose, repack them into one FP8x4 value, and let CUDA use
                // the packed conversion path instead of issuing four independent scalar FP8 conversions.
#pragma unroll
                for (uint32_t slot_base = 0; slot_base < MaxComponentsPerLane; slot_base += 4) {
                    const uint32_t first_component = lane + slot_base * TILED_REDUCTION_WARP_THREADS;
                    if (first_component < tile_width) {
                        uint32_t packed_component_values = 0;
#pragma unroll
                        for (uint32_t packed_slot = 0; packed_slot < 4; ++packed_slot) {
                            const uint32_t slot = slot_base + packed_slot;
                            const uint32_t component = lane + slot * TILED_REDUCTION_WARP_THREADS;
                            if (component < tile_width) {
                                const uint32_t staged_byte = prefix_bytes + component;
                                const uint32_t source_packet = staged_byte >> PacketShift;
                                const uint32_t packet_byte = staged_byte & PacketMask;
                                const uint32_t source_word =
                                    staged_packet_words[warp][
                                        (packet_byte >> 2) * TILED_REDUCTION_WARP_THREADS + source_packet];
                                const uint32_t source_value =
                                    (source_word >> ((packet_byte & 3U) * 8U)) & 0xffU;
                                packed_component_values |= source_value << (packed_slot * 8U);
                            }
                        }

                        const float4 decoded = transformFp8CompleteDecodedPacket<FixedInputTransform>(
                            decodeFp8x4ToFp32<InputT>(packed_component_values), input_transform);
                        local[slot_base] = combine(local[slot_base], decoded.x);
                        if (first_component + TILED_REDUCTION_WARP_THREADS < tile_width) {
                            local[slot_base + 1] = combine(local[slot_base + 1], decoded.y);
                        }
                        if (first_component + 2U * TILED_REDUCTION_WARP_THREADS < tile_width) {
                            local[slot_base + 2] = combine(local[slot_base + 2], decoded.z);
                        }
                        if (first_component + 3U * TILED_REDUCTION_WARP_THREADS < tile_width) {
                            local[slot_base + 3] = combine(local[slot_base + 3], decoded.w);
                        }
                    }
                }
            }
        };
        // Computes a row's alignment prefix and packet count and loads this lane's packet. A p16 load extends at
        // most 15 bytes past the logical tensor end; Tensor/view allocation provides >=128 trailing bytes, and
        // component masks prevent out-of-range values from entering the reduction.
        const auto load_row = [&](uint32_t row_index, uint32_t& prefix_bytes, uint32_t& packet_count,
                                  RawVectorPacket<PacketBytes>& packet) {
            const uint64_t logical_begin = base_shift + static_cast<uint64_t>(outer_index) * outer_stride +
                                           static_cast<uint64_t>(row_index) * row_stride + tile_begin;
            prefix_bytes = static_cast<uint32_t>(logical_begin & PacketMask);
            packet_count = (prefix_bytes + tile_width + PacketMask) >> PacketShift;
            if (lane < packet_count) {
                packet = *reinterpret_cast<const RawVectorPacket<PacketBytes>*>(
                    aligned_input + (logical_begin - prefix_bytes) * sizeof(InputT) +
                    static_cast<uint64_t>(lane) * PacketBytes);
            }
        };
        // Load-ahead: the next row's packet is loaded into registers before the current row is staged and decoded, so
        // one global load per lane is always in flight while the warp decodes. Row order is unchanged.
        uint32_t row = warp_in_output;
        uint32_t current_prefix = 0;
        uint32_t current_packet_count = 0;
        RawVectorPacket<PacketBytes> current_packet{};
        if (valid_work && row < reduction_size) {
            load_row(row, current_prefix, current_packet_count, current_packet);
        }
        while (valid_work && row < reduction_size) {
            const uint32_t next_row = row + warps_per_output;
            uint32_t next_prefix = 0;
            uint32_t next_packet_count = 0;
            RawVectorPacket<PacketBytes> next_packet{};
            if (next_row < reduction_size) {
                load_row(next_row, next_prefix, next_packet_count, next_packet);
            }
            if (lane < current_packet_count) {
                staged_packet_words[warp][0 * TILED_REDUCTION_WARP_THREADS + lane] = current_packet.value.x;
                staged_packet_words[warp][1 * TILED_REDUCTION_WARP_THREADS + lane] = current_packet.value.y;
                staged_packet_words[warp][2 * TILED_REDUCTION_WARP_THREADS + lane] = current_packet.value.z;
                staged_packet_words[warp][3 * TILED_REDUCTION_WARP_THREADS + lane] = current_packet.value.w;
            }
            __syncwarp();
            decode_row(current_prefix);
            __syncwarp();
            current_prefix = next_prefix;
            current_packet_count = next_packet_count;
            current_packet = next_packet;
            row = next_row;
        }

        if (warps_per_output == 1) {
#pragma unroll
            for (uint32_t slot = 0; slot < MaxComponentsPerLane; ++slot) {
                const uint32_t component = lane + slot * TILED_REDUCTION_WARP_THREADS;
                if (valid_work && component < tile_width) {
                    storeFp8CompleteResult<FixedOp == 0>(output, output_dtype, static_cast<uint64_t>(outer_index) * inner_size + tile_begin + component, local[slot], output_finalize, output_scale);
                }
            }
            continue;
        }

#pragma unroll
        for (uint32_t slot = 0; slot < MaxComponentsPerLane; ++slot) {
            const uint32_t component = lane + slot * TILED_REDUCTION_WARP_THREADS;
            if (component < tile_width) {
                warp_partials[warp][component] = valid_work ? local[slot] : initial;
            }
        }
        __syncthreads();

        if (warp_in_output == 0) {
#pragma unroll
            for (uint32_t slot = 0; slot < MaxComponentsPerLane; ++slot) {
                const uint32_t component = lane + slot * TILED_REDUCTION_WARP_THREADS;
                if (valid_work && component < tile_width) {
                    float reduced = initial;
                    for (uint32_t source_warp = 0; source_warp < warps_per_output; ++source_warp) {
                        reduced = combine(reduced, warp_partials[first_warp_in_output + source_warp][component]);
                    }
                    storeFp8CompleteResult<FixedOp == 0>(output, output_dtype, static_cast<uint64_t>(outer_index) * inner_size + tile_begin + component, reduced, output_finalize, output_scale);
                }
            }
        }
        __syncthreads();
    }
}



// The row and slab strides a PacketComplete launch reads, in elements: a pitched view's, given as the 3-D span [outer,
// R, K] with strides [slab pitch, row pitch, 1] (Reducer's pitched stamp), else the dense input's K and R x K (every
// other caller passes its dense [outer, R, K] in whatever dimensions).
struct PacketCompleteRowStrides {
    uint64_t row = 0;
    uint64_t outer = 0;
};

[[nodiscard]] PacketCompleteRowStrides packetCompleteRowStrides(const StridedSpan& input,
                                                                const ReductionGeometry& geometry) {
    const std::vector<uint64_t>& dims = input.getDimensions();
    const std::vector<uint64_t>& strides = input.getStridesElements();
    if (dims.size() == 3 && dims[0] == geometry.outer_size && dims[1] == geometry.reduction_size &&
        dims[2] == geometry.inner_size && (strides[2] == 1 || dims[2] == 1) &&
        (strides[1] >= dims[2] || dims[1] == 1)) {
        return {strides[1], strides[0]};
    }
    if (!input.isDenseContiguous() && !input.isCompactPermutation()) {
        throw std::logic_error("RuntimeK PacketComplete reads a dense input or a pitched [outer, R, K] span.");
    }
    return {geometry.inner_size, geometry.reduction_size * geometry.inner_size};
}

template <typename InputT, int FixedInputTransform = 0, int FixedOp = 0>
void launchPacketCompleteValueKernel(const StridedSpan& input,
                                     const StridedSpan& output,
                                     const ReductionGeometry& geometry,
                                     uint32_t warps_per_output,
                                     uint32_t tile_components,
                                     cudaStream_t stream,
                                     AdditiveFinalizeFp32 output_finalize = {1.0f, false},
                                     ReductionStageInputTransform input_transform = ReductionStageInputTransform::Identity,
                                     float output_scale = 1.0f) {
    static_assert(FixedInputTransform >= 0 && FixedInputTransform <= 2);
    static_assert(sizeof(InputT) == 1 || std::is_same_v<InputT, float>);
    constexpr uint32_t MaxTileComponents = std::is_same_v<InputT, float> ? 124 : 496;
    if (tile_components == 0 || tile_components > MaxTileComponents ||
        (warps_per_output != 1 && warps_per_output != 2 && warps_per_output != 4 && warps_per_output != 8)) {
        throw std::logic_error("RuntimeK PacketComplete received illegal tile/W geometry.");
    }
    // Any element address (the kernel takes its prefixes from the base address).
    if ((reinterpret_cast<uintptr_t>(input.getMemPtr<InputT>()) & uintptr_t{sizeof(InputT) - 1}) != 0) {
        throw std::logic_error("RuntimeK PacketComplete requires an element-aligned input base.");
    }
    const uint32_t tile_count = static_cast<uint32_t>(
        ceilDivideU64(geometry.inner_size, static_cast<uint64_t>(tile_components)));
    const uint32_t outputs_per_block = SHALLOW_COMPLETE_WARPS / warps_per_output;
    const uint64_t work_count = geometry.outer_size * static_cast<uint64_t>(tile_count);
    const uint64_t block_count = std::min<uint64_t>(
        ceilDivideU64(work_count, static_cast<uint64_t>(outputs_per_block)),
        PERSISTENT_MAX_GRID_BLOCKS);
    const PacketCompleteRowStrides row_strides = packetCompleteRowStrides(input, geometry);
    packetCompleteValueKernel<InputT, FixedInputTransform, FixedOp>
        <<<static_cast<unsigned int>(block_count), SHALLOW_COMPLETE_THREADS, 0, stream>>>(
            input.getMemPtr<InputT>(),
            output.getMemPtr<void>(),
            output.getDataType(),
            static_cast<uint32_t>(geometry.outer_size),
            static_cast<uint32_t>(geometry.reduction_size),
            static_cast<uint32_t>(geometry.inner_size),
            row_strides.row,
            row_strides.outer,
            tile_components,
            tile_count,
            warps_per_output,
            output_finalize,
            input_transform,
            output_scale);
    CUDA_CHECK(cudaGetLastError());
}


// The FP32 RK ARG PacketComplete twin: the FP32 path of packetCompleteValueKernel (persistent 256-thread blocks; a work
// item is [outer, K tile] of at most 124 components; W1/W2/W4/W8 warps per item take rows round-robin; a warp loads one
// row's tile window with its 0..3 prefix, stages it plane-major and reads its four slots, with the next row's packet
// loaded ahead) with one FP32 ARG candidate (Fp32ArgKey.cuh) per owned component. A warp's rows only increase, so a
// strictly smaller key replaces its best; the warps of an item merge in shared memory with the (key, tail) minimum.
// Complete: writes the requested outputs. R <= 2^31 - 1.
// FP16 / BF16 input runs at the 2-byte VALUE kernel's mapping (8 components per packet, tiles of at most 248, eight
// slots per lane): each component is its half-word of the staged word, widened to FP32 and given the FP32 key.
template <typename InputT>
__global__ __launch_bounds__(SHALLOW_COMPLETE_THREADS, 4) void packetCompleteArgKernel(const InputT* input,
                                                                                    Fp32ArgOutputs outputs,
                                                                                    uint32_t outer_size,
                                                                                    uint32_t reduction_size,
                                                                                    uint32_t inner_size,
                                                                                    uint32_t tile_components,
                                                                                    uint32_t tile_count,
                                                                                    uint32_t warps_per_output) {
    static_assert(std::is_same_v<InputT, float> || isF16InputType<InputT>);
    constexpr bool F16Input = isF16InputType<InputT>;
    constexpr uint32_t PacketBytes = 16;
    constexpr uint32_t PacketWords = 4;
    constexpr uint32_t PacketShift = F16Input ? 3 : 2;  // components per packet: 8 or 4
    constexpr uint32_t PacketMask = (1U << PacketShift) - 1;
    constexpr uint32_t MaxTileComponents = F16Input ? 248 : 124;
    constexpr uint32_t Slots = F16Input ? 8 : 4;
    __shared__ uint32_t staged_packet_words[SHALLOW_COMPLETE_WARPS][PacketWords * TILED_REDUCTION_WARP_THREADS];
    __shared__ uint32_t warp_keys[SHALLOW_COMPLETE_WARPS][MaxTileComponents];
    __shared__ uint32_t warp_tails[SHALLOW_COMPLETE_WARPS][MaxTileComponents];

    const uint32_t warp = threadIdx.x / TILED_REDUCTION_WARP_THREADS;
    const uint32_t lane = threadIdx.x % TILED_REDUCTION_WARP_THREADS;
    const uint32_t outputs_per_block = SHALLOW_COMPLETE_WARPS / warps_per_output;
    const uint32_t output_group = warp / warps_per_output;
    const uint32_t warp_in_output = warp - output_group * warps_per_output;
    const uint32_t first_warp_in_output = output_group * warps_per_output;
    const uint64_t work_count = static_cast<uint64_t>(outer_size) * tile_count;
    const uint64_t block_work_stride = static_cast<uint64_t>(gridDim.x) * outputs_per_block;
    const uint64_t outer_input_stride = static_cast<uint64_t>(reduction_size) * inner_size;
    // As in the VALUE kernel, the base's place in its 16-byte packet, in components (0 for an aligned base): prefixes
    // and packet addresses follow the address, so the kernel reads any element address. A row's first packet starts at
    // most 15 bytes before the base, inside the base's own 16-byte granule; the tile cap (124 / 248) already leaves
    // room for a 3 / 7 component prefix. Keys carry rows, so the indices do not change.
    const uint32_t base_shift = static_cast<uint32_t>(
        (reinterpret_cast<uintptr_t>(input) & uintptr_t{PacketBytes - 1}) / sizeof(InputT));
    const uint8_t* const aligned_input = reinterpret_cast<const uint8_t*>(input) - base_shift * sizeof(InputT);

    for (uint64_t block_work_begin = static_cast<uint64_t>(blockIdx.x) * outputs_per_block;
         block_work_begin < work_count;
         block_work_begin += block_work_stride) {
        const uint64_t work_index = block_work_begin + output_group;
        const bool valid_work = work_index < work_count;
        uint32_t outer_index = 0;
        uint32_t tile_begin = 0;
        uint32_t tile_width = 0;
        if (valid_work) {
            outer_index = static_cast<uint32_t>(work_index / tile_count);
            const uint32_t tile_index =
                static_cast<uint32_t>(work_index - static_cast<uint64_t>(outer_index) * tile_count);
            tile_begin = tile_index * tile_components;
            tile_width = tile_components < (inner_size - tile_begin) ? tile_components : (inner_size - tile_begin);
        }
        uint32_t best_key[Slots], best_tail[Slots];
#pragma unroll
        for (uint32_t slot = 0; slot < Slots; ++slot) {
            best_key[slot] = FP32_ARG_KEY_IDENTITY;
            best_tail[slot] = 0xFFFFFFFFU;
        }
        const auto load_row = [&](uint32_t row_index, uint32_t& prefix, uint32_t& packet_count,
                                  RawVectorPacket<PacketBytes>& packet) {
            const uint64_t logical_begin = base_shift + static_cast<uint64_t>(outer_index) * outer_input_stride +
                                           static_cast<uint64_t>(row_index) * inner_size + tile_begin;
            prefix = static_cast<uint32_t>(logical_begin & PacketMask);
            packet_count = (prefix + tile_width + PacketMask) >> PacketShift;
            if (lane < packet_count) {
                packet = *reinterpret_cast<const RawVectorPacket<PacketBytes>*>(
                    aligned_input + (logical_begin - prefix) * sizeof(InputT) +
                    static_cast<uint64_t>(lane) * PacketBytes);
            }
        };
        // Component c of the staged row sits at position prefix + c: packet (position >> 2), word (position & 3).
        // FP16 / BF16: at packet (position >> 3), word ((position & 7) >> 1), half-word (position & 1).
        const auto key_row = [&](uint32_t row, uint32_t prefix) {
#pragma unroll
            for (uint32_t slot = 0; slot < Slots; ++slot) {
                const uint32_t component = lane + slot * TILED_REDUCTION_WARP_THREADS;
                if (component < tile_width) {
                    const uint32_t staged = prefix + component;
                    uint32_t bits;
                    if constexpr (F16Input) {
                        const uint32_t word = staged_packet_words[warp][((staged & PacketMask) >> 1) *
                                                                            TILED_REDUCTION_WARP_THREADS +
                                                                        (staged >> PacketShift)];
                        bits = __float_as_uint(widenF16Bits<InputT>((staged & 1U) != 0 ? word >> 16 : word));
                    } else {
                        bits = staged_packet_words[warp][(staged & 3U) * TILED_REDUCTION_WARP_THREADS + (staged >> 2)];
                    }
                    uint32_t negative_zero;
                    const uint32_t key = fp32ArgKey(bits, outputs.xor_mask, negative_zero);
                    if (key < best_key[slot]) {
                        best_key[slot] = key;
                        best_tail[slot] = (row << 1) | negative_zero;
                    }
                }
            }
        };
        uint32_t row = warp_in_output;
        uint32_t current_prefix = 0;
        uint32_t current_packet_count = 0;
        RawVectorPacket<PacketBytes> current_packet{};
        if (valid_work && row < reduction_size) load_row(row, current_prefix, current_packet_count, current_packet);
        while (valid_work && row < reduction_size) {
            const uint32_t next_row = row + warps_per_output;
            uint32_t next_prefix = 0;
            uint32_t next_packet_count = 0;
            RawVectorPacket<PacketBytes> next_packet{};
            if (next_row < reduction_size) load_row(next_row, next_prefix, next_packet_count, next_packet);
            if (lane < current_packet_count) {
                staged_packet_words[warp][0 * TILED_REDUCTION_WARP_THREADS + lane] = current_packet.value.x;
                staged_packet_words[warp][1 * TILED_REDUCTION_WARP_THREADS + lane] = current_packet.value.y;
                staged_packet_words[warp][2 * TILED_REDUCTION_WARP_THREADS + lane] = current_packet.value.z;
                staged_packet_words[warp][3 * TILED_REDUCTION_WARP_THREADS + lane] = current_packet.value.w;
            }
            __syncwarp();
            key_row(row, current_prefix);
            __syncwarp();
            current_prefix = next_prefix;
            current_packet_count = next_packet_count;
            current_packet = next_packet;
            row = next_row;
        }

        if (warps_per_output == 1) {
#pragma unroll
            for (uint32_t slot = 0; slot < Slots; ++slot) {
                const uint32_t component = lane + slot * TILED_REDUCTION_WARP_THREADS;
                if (valid_work && component < tile_width)
                    storeFp32ArgResult(outputs, static_cast<uint64_t>(outer_index) * inner_size + tile_begin + component,
                                       Fp32ArgCandidate{best_key[slot], best_tail[slot]});
            }
            continue;
        }
#pragma unroll
        for (uint32_t slot = 0; slot < Slots; ++slot) {
            const uint32_t component = lane + slot * TILED_REDUCTION_WARP_THREADS;
            if (component < tile_width) {
                warp_keys[warp][component] = best_key[slot];
                warp_tails[warp][component] = best_tail[slot];
            }
        }
        __syncthreads();
        if (warp_in_output == 0) {
#pragma unroll
            for (uint32_t slot = 0; slot < Slots; ++slot) {
                const uint32_t component = lane + slot * TILED_REDUCTION_WARP_THREADS;
                if (valid_work && component < tile_width) {
                    Fp32ArgCandidate reduced = fp32ArgIdentity();
                    for (uint32_t source_warp = 0; source_warp < warps_per_output; ++source_warp) {
                        const uint32_t source = first_warp_in_output + source_warp;
                        const uint32_t key = warp_keys[source][component];
                        if (key != FP32_ARG_KEY_IDENTITY)
                            reduced = fp32ArgMin(reduced, Fp32ArgCandidate{key, warp_tails[source][component]});
                    }
                    storeFp32ArgResult(outputs, static_cast<uint64_t>(outer_index) * inner_size + tile_begin + component,
                                       reduced);
                }
            }
        }
        __syncthreads();
    }
}

// The FP8 RuntimeK PacketComplete argmin/argmax twin: packetCompleteValueKernel's topology (persistent 256-thread
// blocks; one work item = [outer, K tile]; W1/W2/W4/W8 warps per item split R round-robin; a warp loads one row's tile
// window with an alignment prefix and stages it through shared memory) with uint32 candidates.
// Complete: writes the requested outputs, index = row * stride.
//
// 16-byte packets (32 lanes cover 512 bytes, prefix <= 15), so a tile holds up to 256 components (8 per lane). The
// window is staged linearly (one vector store per loading lane) and lane L owns four adjacent components per quad q
// (components q * 128 + 4L .. +3): one word of keys is two LDS.32 and a funnel shift by the row's prefix (prefix & 3 is
// uniform across the warp), instead of four plane-major byte gathers of ~10 instructions each. The tile is the fewest
// 128-component quads per row (fp8PacketCompleteArgTileComponents).
constexpr uint32_t FP8_ARG_PACKET_COMPLETE_MAX_TILE = 256;

template <int PacketBytes>
__global__ __launch_bounds__(SHALLOW_COMPLETE_THREADS, 4) void fp8PacketCompleteArgKernel(const uint8_t* input,
                                                                                   Fp8ArgOutputs outputs,
                                                                                   uint32_t outer_size,
                                                                                   uint32_t reduction_size,
                                                                                   uint32_t inner_size,
                                                                                   uint32_t tile_components,
                                                                                   uint32_t tile_count,
                                                                                   uint32_t warps_per_output) {
    static_assert(PacketBytes == 16);
    constexpr uint32_t PacketShift = 4;
    constexpr uint32_t PacketMask = PacketBytes - 1;
    constexpr uint32_t Quads = 2;
    constexpr uint32_t Slots = 4 * Quads;
    constexpr uint32_t QuadComponents = 4 * TILED_REDUCTION_WARP_THREADS;
    constexpr uint32_t MaxTile = FP8_ARG_PACKET_COMPLETE_MAX_TILE;
    constexpr uint32_t WindowWords = PacketBytes / 4 * TILED_REDUCTION_WARP_THREADS;
    static_assert(Quads * QuadComponents == MaxTile);
    static_assert(PacketMask + MaxTile <= PacketBytes * TILED_REDUCTION_WARP_THREADS);
    // The highest word read: (prefix + last quad's last lane base) / 4 + 1.
    static_assert((PacketMask + (Quads - 1) * QuadComponents + 4 * (TILED_REDUCTION_WARP_THREADS - 1)) / 4 + 1 <
                  WindowWords);
    constexpr uint32_t Identity = FP8_ARG_LANE_IDENTITY;

    __shared__ __align__(16) uint32_t staged_packet_words[SHALLOW_COMPLETE_WARPS][WindowWords];
    __shared__ uint32_t warp_partials[SHALLOW_COMPLETE_WARPS][MaxTile];

    const uint32_t warp = threadIdx.x / TILED_REDUCTION_WARP_THREADS;
    const uint32_t lane = threadIdx.x % TILED_REDUCTION_WARP_THREADS;
    const uint32_t lane_component = 4U * lane;
    const uint32_t outputs_per_block = SHALLOW_COMPLETE_WARPS / warps_per_output;
    const uint32_t output_group = warp / warps_per_output;
    const uint32_t warp_in_output = warp - output_group * warps_per_output;
    const uint32_t first_warp_in_output = output_group * warps_per_output;
    const uint64_t work_count = static_cast<uint64_t>(outer_size) * tile_count;
    const uint64_t block_work_stride = static_cast<uint64_t>(gridDim.x) * outputs_per_block;
    const uint64_t outer_input_stride = static_cast<uint64_t>(reduction_size) * inner_size;
    // As in the VALUE kernel, the base's place in its 16-byte packet (0 for an aligned base): prefixes and packet
    // addresses follow the address, so the kernel reads any byte address. A row's first packet starts at most 15 bytes
    // before the base, inside the base's own 16-byte granule; the window (512 bytes) holds a 15-byte prefix with a
    // 256-component tile. Keys carry rows, so the indices do not change.
    const uint32_t base_shift = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(input) & uintptr_t{PacketMask});
    const uint8_t* const aligned_input = input - base_shift;

    for (uint64_t block_work_begin = static_cast<uint64_t>(blockIdx.x) * outputs_per_block; block_work_begin < work_count;
         block_work_begin += block_work_stride) {
        const uint64_t work_index = block_work_begin + output_group;
        const bool valid_work = work_index < work_count;
        uint32_t outer_index = 0;
        uint32_t tile_begin = 0;
        uint32_t tile_width = 0;
        if (valid_work) {
            outer_index = static_cast<uint32_t>(work_index / tile_count);
            const uint32_t tile_index = static_cast<uint32_t>(work_index - static_cast<uint64_t>(outer_index) * tile_count);
            tile_begin = tile_index * tile_components;
            tile_width = tile_components < (inner_size - tile_begin) ? tile_components : (inner_size - tile_begin);
        }

        uint32_t local[Slots];
#pragma unroll
        for (uint32_t slot = 0; slot < Slots; ++slot) local[slot] = Identity;

        // Keys the lane's four adjacent components of each quad in one word.
        const auto key_row = [&](uint32_t prefix_bytes, uint32_t row_index) {
            const uint32_t* words = staged_packet_words[warp];
            const uint32_t shift = (prefix_bytes & 3U) * 8U;
#pragma unroll
            for (uint32_t quad = 0; quad < Quads; ++quad) {
                const uint32_t first = quad * QuadComponents + lane_component;
                if (first < tile_width) {
                    const uint32_t word = (prefix_bytes + first) >> 2;
                    const uint32_t packed = __funnelshift_r(words[word], words[word + 1], shift);
                    const Fp8ArgKeys4 keys = fp8ArgKeys4(packed, outputs.fold);
#pragma unroll
                    for (uint32_t j = 0; j < 4; ++j) {
                        if (first + j < tile_width) {
                            local[quad * 4 + j] =
                                fp8ArgMin(local[quad * 4 + j], fp8ArgLaneCandidate(keys, j, row_index));
                        }
                    }
                }
            }
        };
        // A p16 load extends at most 15 bytes past the logical tensor end (Tensor allocations keep >= 128 trailing
        // bytes); component masks keep those bytes out of the reduction.
        const auto load_row = [&](uint32_t row_index, uint32_t& prefix_bytes, uint32_t& packet_count,
                                  RawVectorPacket<PacketBytes>& packet) {
            const uint64_t logical_begin = base_shift + static_cast<uint64_t>(outer_index) * outer_input_stride +
                                           static_cast<uint64_t>(row_index) * inner_size + tile_begin;
            prefix_bytes = static_cast<uint32_t>(logical_begin & PacketMask);
            packet_count = (prefix_bytes + tile_width + PacketMask) >> PacketShift;
            if (lane < packet_count) {
                packet = *reinterpret_cast<const RawVectorPacket<PacketBytes>*>(
                    aligned_input + (logical_begin - prefix_bytes) + static_cast<uint64_t>(lane) * PacketBytes);
            }
        };
        // Load-ahead as in the VALUE kernel: the next row's packet is in flight while this row is keyed.
        uint32_t row = warp_in_output;
        uint32_t current_prefix = 0;
        uint32_t current_packet_count = 0;
        RawVectorPacket<PacketBytes> current_packet{};
        if (valid_work && row < reduction_size) load_row(row, current_prefix, current_packet_count, current_packet);
        while (valid_work && row < reduction_size) {
            const uint32_t next_row = row + warps_per_output;
            uint32_t next_prefix = 0;
            uint32_t next_packet_count = 0;
            RawVectorPacket<PacketBytes> next_packet{};
            if (next_row < reduction_size) load_row(next_row, next_prefix, next_packet_count, next_packet);
            if (lane < current_packet_count) {
                reinterpret_cast<uint4*>(staged_packet_words[warp])[lane] = current_packet.value;
            }
            __syncwarp();
            key_row(current_prefix, row);
            __syncwarp();
            current_prefix = next_prefix;
            current_packet_count = next_packet_count;
            current_packet = next_packet;
            row = next_row;
        }

        if (warps_per_output == 1) {
#pragma unroll
            for (uint32_t slot = 0; slot < Slots; ++slot) {
                const uint32_t component = (slot / 4U) * QuadComponents + lane_component + slot % 4U;
                if (valid_work && component < tile_width) {
                    storeFp8ArgResult(outputs, static_cast<uint64_t>(outer_index) * inner_size + tile_begin + component,
                                      local[slot]);
                }
            }
            continue;
        }

#pragma unroll
        for (uint32_t slot = 0; slot < Slots; ++slot) {
            const uint32_t component = (slot / 4U) * QuadComponents + lane_component + slot % 4U;
            if (component < tile_width) warp_partials[warp][component] = valid_work ? local[slot] : Identity;
        }
        __syncthreads();
        if (warp_in_output == 0) {
#pragma unroll
            for (uint32_t slot = 0; slot < Slots; ++slot) {
                const uint32_t component = (slot / 4U) * QuadComponents + lane_component + slot % 4U;
                if (valid_work && component < tile_width) {
                    uint32_t reduced = Identity;
                    for (uint32_t source_warp = 0; source_warp < warps_per_output; ++source_warp) {
                        reduced = fp8ArgMin(reduced, warp_partials[first_warp_in_output + source_warp][component]);
                    }
                    storeFp8ArgResult(outputs, static_cast<uint64_t>(outer_index) * inner_size + tile_begin + component,
                                      reduced);
                }
            }
        }
        __syncthreads();
    }
}

// The FP8 RuntimeK argmin/argmax twin: the rotated sharded layout of runtimeKRotatedValueKernel (16 cooperative warps;
// grid x = shard, y = component tile, z = outer; each warp aligns its first row's tile window once and every revisited
// row, 16 rows later, stays packet-aligned) with uint32 candidates and a runtime shard depth.
// The logical tile is the warp's packet span less the alignment reserve (PacketBytes 16: 497 components, 8: 249), so
// one kernel serves both VALUE tile layouts. One shard (depth >= R) is a Complete pass; several shards write FP32
// value + carried index partials. The index is (row base + row) * stride. Only the 8-byte packet is compiled: the
// 16-byte packet measured 1.56x slower (best p16 / best p8 over 2408 cells; p16 faster on 78).
constexpr int FP8_ARG_ROTATED_WARPS = 16;
constexpr int FP8_ARG_ROTATED_THREADS = FP8_ARG_ROTATED_WARPS * TILED_REDUCTION_WARP_THREADS;

template <int PacketBytes>
inline constexpr uint32_t fp8ArgRotatedTileComponents =
    static_cast<uint32_t>(TILED_REDUCTION_WARP_THREADS * PacketBytes - (PacketBytes - 1));

template <int PacketBytes>
__global__ __launch_bounds__(FP8_ARG_ROTATED_THREADS) void fp8RuntimeKRotatedArgKernel(const uint8_t* input,
                                                                              Fp8ArgOutputs outputs,
                                                                              uint64_t outer_size,
                                                                              uint64_t reduction_size,
                                                                              uint64_t inner_size,
                                                                              uint64_t shards_per_output,
                                                                              uint64_t component_tiles,
                                                                              uint32_t rows_per_shard) {
    static_assert(PacketBytes == 8 || PacketBytes == 16);
    constexpr int Words = PacketBytes / 4;
    constexpr uint32_t SpanComponents = TILED_REDUCTION_WARP_THREADS * PacketBytes;
    constexpr uint32_t TileComponents = fp8ArgRotatedTileComponents<PacketBytes>;
    constexpr uint32_t Identity = FP8_ARG_LANE_IDENTITY;
    // Revisited rows are FP8_ARG_ROTATED_WARPS * K bytes apart: a multiple of the packet for every K.
    static_assert(FP8_ARG_ROTATED_WARPS % PacketBytes == 0);

    __shared__ uint32_t warp_partials[FP8_ARG_ROTATED_WARPS][SpanComponents];
    __shared__ uint32_t warp_prefix[FP8_ARG_ROTATED_WARPS];

    const uint32_t warp = threadIdx.x / TILED_REDUCTION_WARP_THREADS;
    const uint32_t lane = threadIdx.x % TILED_REDUCTION_WARP_THREADS;
    const uint32_t outer_count = static_cast<uint32_t>(outer_size);
    const uint32_t tile_count = static_cast<uint32_t>(component_tiles);
    const uint32_t shard_count = static_cast<uint32_t>(shards_per_output);
    for (uint32_t outer_index = blockIdx.z; outer_index < outer_count; outer_index += gridDim.z)
    for (uint32_t component_tile = blockIdx.y; component_tile < tile_count; component_tile += gridDim.y)
    for (uint32_t shard = blockIdx.x; shard < shard_count; shard += gridDim.x) {
        const uint64_t tile_begin = static_cast<uint64_t>(component_tile) * TileComponents;
        const uint32_t valid_components =
            static_cast<uint32_t>(minU64(TileComponents, inner_size - tile_begin));
        const uint64_t row_begin = static_cast<uint64_t>(shard) * rows_per_shard;
        const uint32_t shard_rows = row_begin < reduction_size
            ? static_cast<uint32_t>(minU64(rows_per_shard, reduction_size - row_begin))
            : 0U;

        uint32_t local[PacketBytes];
#pragma unroll
        for (int item = 0; item < PacketBytes; ++item) local[item] = Identity;
        uint32_t prefix = 0;
        if (warp < shard_rows) {
            const uint8_t* logical = input + (outer_index * reduction_size + row_begin + warp) * inner_size + tile_begin;
            const uintptr_t address = reinterpret_cast<uintptr_t>(logical);
            prefix = static_cast<uint32_t>(address & uintptr_t{PacketBytes - 1});
            const uint8_t* lane_packet =
                reinterpret_cast<const uint8_t*>(address & ~uintptr_t{PacketBytes - 1}) + lane * PacketBytes;
            const int64_t lane_component = static_cast<int64_t>(tile_begin) - static_cast<int64_t>(prefix) +
                                           static_cast<int64_t>(lane * PacketBytes);
            if (lane_component < static_cast<int64_t>(inner_size)) {
                const uint64_t row_stride = static_cast<uint64_t>(FP8_ARG_ROTATED_WARPS) * inner_size;
                for (uint32_t row = warp; row < shard_rows; row += FP8_ARG_ROTATED_WARPS) {
                    const RawVectorPacket<PacketBytes> packet = *reinterpret_cast<const RawVectorPacket<PacketBytes>*>(lane_packet);
                    uint32_t words[Words];
                    if constexpr (PacketBytes == 16) {
                        words[0] = packet.value.x; words[1] = packet.value.y;
                        words[2] = packet.value.z; words[3] = packet.value.w;
                    } else {
                        words[0] = packet.value.x; words[1] = packet.value.y;
                    }
#pragma unroll
                    for (int w = 0; w < Words; ++w) {
                        const Fp8ArgKeys4 keys = fp8ArgKeys4(words[w], outputs.fold);
#pragma unroll
                        for (int b = 0; b < 4; ++b) {
                            local[4 * w + b] = fp8ArgMin(local[4 * w + b], fp8ArgLaneCandidate(keys, b, row));
                        }
                    }
                    lane_packet += row_stride;
                }
            }
        }
        if (lane == 0) warp_prefix[warp] = prefix;
#pragma unroll
        for (int item = 0; item < PacketBytes; ++item) warp_partials[warp][lane * PacketBytes + item] = local[item];
        __syncthreads();
        for (uint32_t component = threadIdx.x; component < valid_components; component += FP8_ARG_ROTATED_THREADS) {
            uint32_t reduced = Identity;
#pragma unroll
            for (int source_warp = 0; source_warp < FP8_ARG_ROTATED_WARPS; ++source_warp) {
                // Warps whose first row is beyond the shard published identities at prefix 0.
                reduced = fp8ArgMin(reduced, warp_partials[source_warp][component + warp_prefix[source_warp]]);
            }
            storeFp8ArgResult<true>(outputs,
                              (static_cast<uint64_t>(outer_index) * shards_per_output + shard) * inner_size +
                                  tile_begin + component,
                              reduced, row_begin);
        }
        __syncthreads();
    }
}

}  // namespace

namespace {

template <typename InputT, int RowsPerShard, int PacketBytes, int CooperativeWarps,
          bool ExactFp8Tile = true, typename InputTransformT = IdentityFp32,
          typename ReductionOpT = cuda::std::plus<float>>
[[nodiscard]] KernelOccupancyInfo queryRuntimeKRotatedOccupancyForRows() {
    const auto kernel = runtimeKRotatedValueKernel<InputT,
                                                                        ReductionOpT,
                                                                        InputTransformT,
                                                                        RowsPerShard,
                                                                        PacketBytes,
                                                                        CooperativeWarps,
                                                                        ExactFp8Tile>;
    cudaFuncAttributes attributes{};
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
    int active_blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks, kernel, CooperativeWarps * TILED_REDUCTION_WARP_THREADS, 0));
    return KernelOccupancyInfo{attributes.numRegs,
                               attributes.sharedSizeBytes,
                               0,
                               active_blocks,
                               active_blocks * CooperativeWarps};
}

// The RuntimeK rotated first stage on FP32 input, PrefixReserved only (Exact's tail logic is FP8 byte chunks and would
// gain 3 components). The shared kernel at p16 with 4 cooperative warps: 128 threads cover the 125 component logical
// tile in the readback, and 4 rows of K floats are a whole number of packets for any K, so each warp aligns once. FP32
// has no extremum fold (raw MIN/MAX kernels, every transform). FP8's row depths.
constexpr int FP32_RUNTIME_K_ROTATED_PACKET_BYTES = 16;
constexpr int FP32_RUNTIME_K_ROTATED_WARPS = 4;
static_assert(runtimeKRotatedLogicalTileElementsForPacket<float, FP32_RUNTIME_K_ROTATED_PACKET_BYTES, false> ==
              static_cast<int>(FP32_RUNTIME_K_ROTATED_TILE_COMPONENTS));
static_assert(RK::Fp8::Fp8RuntimeKPolicy::rotated_row_depths == std::array<uint64_t, 6>{32, 64, 128, 256, 512, 1024},
              "dispatchFp32RuntimeKRotated's cases restate the retained depths");

// Invokes callback.template operator()<InputTransformT, RowsPerShard>() for the compiled FP32 rotated specialization
// matching the runtime request; returns false otherwise.
template <typename Callback>
[[nodiscard]] bool dispatchFp32RuntimeKRotated(uint64_t rows_per_shard, ReductionStageInputTransform transform,
                                               Callback&& callback) {
    bool dispatched = false;
    const auto for_rows = [&]<typename InputTransformT>() {
        // The cases are Fp8RuntimeKPolicy::rotated_row_depths (fp32RuntimeKRotatedSupports checks the same list).
        const auto rows = [&]<int RowsPerShard>() {
            callback.template operator()<InputTransformT, RowsPerShard>();
            dispatched = true;
        };
        switch (rows_per_shard) {
            case 32: rows.template operator()<32>(); return;
            case 64: rows.template operator()<64>(); return;
            case 128: rows.template operator()<128>(); return;
            case 256: rows.template operator()<256>(); return;
            case 512: rows.template operator()<512>(); return;
            case 1024: rows.template operator()<1024>(); return;
            default: return;
        }
    };
    switch (transform) {
        case ReductionStageInputTransform::Identity: for_rows.template operator()<IdentityFp32>(); break;
        case ReductionStageInputTransform::AbsoluteValue: for_rows.template operator()<AbsoluteValueFp32>(); break;
        case ReductionStageInputTransform::Square: for_rows.template operator()<SquareFp32>(); break;
    }
    return dispatched;
}

template <typename ReductionOpT>
void launchFp32RuntimeKRotatedFirstStageTyped(const StridedSpan& input,
                                              const StridedSpan& fp32_partials,
                                              const ReductionGeometry& geometry,
                                              uint64_t rows_per_shard,
                                              uint64_t shards_per_output,
                                              Stream& stream,
                                              ReductionStageInputTransform transform) {
    if (input.getDataType() != DataType::FP32 || input.getTotalNumElements() != geometry.input_elements ||
        !fp32RuntimeKRotatedSupports(geometry.outer_size, geometry.reduction_size, geometry.inner_size,
                                     rows_per_shard) ||
        shards_per_output != ceilDivideU64(geometry.reduction_size, rows_per_shard) ||
        fp32_partials.getDataType() != DataType::FP32 ||
        fp32_partials.getDimensions() !=
            std::vector<uint64_t>{geometry.outer_size, shards_per_output, geometry.inner_size}) {
        throw std::logic_error("FP32 RuntimeK rotated first stage received invalid staged geometry.");
    }
    if ((reinterpret_cast<uintptr_t>(input.getMemPtr<void>()) & uintptr_t{3}) != 0) {
        throw std::logic_error("FP32 RuntimeK rotated first stage requires a 4-byte aligned input base.");
    }
    const bool launched = dispatchFp32RuntimeKRotated(
        rows_per_shard, transform, [&]<typename InputTransformT, int RowsPerShard>() {
            launchRuntimeKRotatedFirstStageForTransformPacket<float, ReductionOpT, InputTransformT, RowsPerShard,
                                                              FP32_RUNTIME_K_ROTATED_PACKET_BYTES,
                                                              FP32_RUNTIME_K_ROTATED_WARPS, false>(
                input, fp32_partials, geometry, shards_per_output, stream.getStream());
        });
    if (!launched) {
        throw std::logic_error("FP32 RuntimeK rotated first stage requires a retained compiled specialization.");
    }
}

template <typename ReductionOpT>
KernelOccupancyInfo queryFp32RuntimeKRotatedOccupancyTyped(uint64_t rows_per_shard,
                                                           ReductionStageInputTransform transform) {
    KernelOccupancyInfo occupancy{};
    const bool queried = dispatchFp32RuntimeKRotated(
        rows_per_shard, transform, [&]<typename InputTransformT, int RowsPerShard>() {
            occupancy = queryRuntimeKRotatedOccupancyForRows<float, RowsPerShard,
                                                                        FP32_RUNTIME_K_ROTATED_PACKET_BYTES,
                                                                        FP32_RUNTIME_K_ROTATED_WARPS, false,
                                                                        InputTransformT, ReductionOpT>();
        });
    if (!queried) {
        throw std::logic_error("FP32 RuntimeK rotated occupancy requires a retained compiled specialization.");
    }
    return occupancy;
}

// The RuntimeK rotated first stage on FP16 / BF16 input: the 2-byte layout (p16 packets of 8 components, 8 cooperative
// warps, whose revisit stride of 8 rows of K half-words is a whole number of packets for any K, and the 249-component
// PrefixReserved tile) at FP8's row depths (ExactFp8Tile at its default, which only FP8 reads). Every
// combine x {identity, |x|, x^2}, no fold, any 2-byte aligned input.
constexpr int F16_RUNTIME_K_ROTATED_PACKET_BYTES = 16;
constexpr int F16_RUNTIME_K_ROTATED_WARPS = RUNTIME_K_ROTATED_WARPS;
static_assert(F16_RUNTIME_K_ROTATED_WARPS == runtimeKRotatedCooperativeWarps<__half>);
static_assert(runtimeKRotatedLogicalTileElementsForPacket<__half, F16_RUNTIME_K_ROTATED_PACKET_BYTES> ==
              static_cast<int>(F16_RUNTIME_K_ROTATED_TILE_COMPONENTS));

template <typename InputT, typename ReductionOpT>
void launchF16RuntimeKRotatedFirstStageTyped(const StridedSpan& input,
                                             const StridedSpan& fp32_partials,
                                             const ReductionGeometry& geometry,
                                             uint64_t rows_per_shard,
                                             uint64_t shards_per_output,
                                             Stream& stream,
                                             ReductionStageInputTransform transform) {
    if (input.getTotalNumElements() != geometry.input_elements ||
        !f16RuntimeKRotatedSupports(geometry.outer_size, geometry.reduction_size, geometry.inner_size,
                                    rows_per_shard) ||
        shards_per_output != ceilDivideU64(geometry.reduction_size, rows_per_shard) ||
        fp32_partials.getDataType() != DataType::FP32 ||
        fp32_partials.getDimensions() !=
            std::vector<uint64_t>{geometry.outer_size, shards_per_output, geometry.inner_size}) {
        throw std::logic_error("FP16 / BF16 RuntimeK rotated first stage received invalid staged geometry.");
    }
    if ((reinterpret_cast<uintptr_t>(input.getMemPtr<void>()) & uintptr_t{1}) != 0) {
        throw std::logic_error("FP16 / BF16 RuntimeK rotated first stage requires a 2-byte aligned input base.");
    }
    const bool launched = dispatchFp32RuntimeKRotated(
        rows_per_shard, transform, [&]<typename InputTransformT, int RowsPerShard>() {
            launchRuntimeKRotatedFirstStageForTransformPacket<InputT, ReductionOpT, InputTransformT, RowsPerShard,
                                                              F16_RUNTIME_K_ROTATED_PACKET_BYTES,
                                                              F16_RUNTIME_K_ROTATED_WARPS>(
                input, fp32_partials, geometry, shards_per_output, stream.getStream());
        });
    if (!launched) {
        throw std::logic_error("FP16 / BF16 RuntimeK rotated first stage requires a retained compiled specialization.");
    }
}

template <typename InputT, typename ReductionOpT>
KernelOccupancyInfo queryF16RuntimeKRotatedOccupancyTyped(uint64_t rows_per_shard,
                                                          ReductionStageInputTransform transform) {
    KernelOccupancyInfo occupancy{};
    const bool queried = dispatchFp32RuntimeKRotated(
        rows_per_shard, transform, [&]<typename InputTransformT, int RowsPerShard>() {
            occupancy = queryRuntimeKRotatedOccupancyForRows<InputT, RowsPerShard,
                                                                        F16_RUNTIME_K_ROTATED_PACKET_BYTES,
                                                                        F16_RUNTIME_K_ROTATED_WARPS, true,
                                                                        InputTransformT, ReductionOpT>();
        });
    if (!queried) {
        throw std::logic_error("FP16 / BF16 RuntimeK rotated occupancy requires a retained compiled specialization.");
    }
    return occupancy;
}

// Invokes callback.template operator()<InputT, ReductionOpT>() for FP16 / BF16 and a combine; false otherwise.
template <typename Callback>
[[nodiscard]] bool dispatchF16RuntimeKRotatedTypes(DataType input_dtype, ReductionStageCombine combine,
                                                   Callback&& callback) {
    const auto for_input = [&]<typename InputT>() {
        switch (combine) {
            case ReductionStageCombine::Sum:
                callback.template operator()<InputT, cuda::std::plus<float>>();
                return true;
            case ReductionStageCombine::Product:
                callback.template operator()<InputT, cuda::std::multiplies<float>>();
                return true;
            case ReductionStageCombine::Minimum:
                callback.template operator()<InputT, PropagatingMinimumFp32>();
                return true;
            case ReductionStageCombine::Maximum:
                callback.template operator()<InputT, PropagatingMaximumFp32>();
                return true;
        }
        return false;
    };
    switch (input_dtype) {
        case DataType::FP16: return for_input.template operator()<__half>();
        case DataType::BF16: return for_input.template operator()<__nv_bfloat16>();
        default: return false;
    }
}

// The FP32 RK ARG rotated twin: FP8's fp8RuntimeKRotatedArgKernel layout at FP32 VALUE rotated's geometry (16-byte
// packets of 4 components, 4 cooperative warps, the 125-component logical tile; each warp aligns its first row's window
// once and every revisited row, 4 rows later, stays aligned) with one FP32 ARG candidate (Fp32ArgKey.cuh) per component
// and a runtime shard depth. A warp's rows only increase, so a strictly smaller key replaces its best; the warps merge
// in shared memory with the (key, tail) minimum. One shard (depth >= R) writes the requested outputs; several write
// packed partials, finished by the continuation.
constexpr int FP32_ARG_ROTATED_WARPS = 4;
constexpr int FP32_ARG_ROTATED_THREADS = FP32_ARG_ROTATED_WARPS * TILED_REDUCTION_WARP_THREADS;
constexpr uint32_t FP32_ARG_ROTATED_SPAN = TILED_REDUCTION_WARP_THREADS * 4;
static_assert(FP32_ARG_ROTATED_SPAN - 3 == FP32_RUNTIME_K_ROTATED_TILE_COMPONENTS);
static_assert(FP32_ARG_ROTATED_THREADS >= static_cast<int>(FP32_RUNTIME_K_ROTATED_TILE_COMPONENTS));

// FP16 / BF16 input runs at the 2-byte VALUE rotated geometry (16-byte packets of 8 components, 8 cooperative warps,
// whose revisit stride of 8 rows of K half-words is a whole number of packets, and the 249-component tile): each
// component widens to FP32 and takes the FP32 key, so the lane state, the packed partials and the continuation are
// FP32's.
template <typename InputT>
inline constexpr int ARG_ROTATED_ITEMS = 16 / static_cast<int>(sizeof(InputT));
template <typename InputT>
inline constexpr int ARG_ROTATED_WARPS = std::is_same_v<InputT, float> ? FP32_ARG_ROTATED_WARPS : 8;
template <typename InputT>
inline constexpr int ARG_ROTATED_THREADS = ARG_ROTATED_WARPS<InputT> * TILED_REDUCTION_WARP_THREADS;
template <typename InputT>
inline constexpr uint32_t ARG_ROTATED_SPAN = TILED_REDUCTION_WARP_THREADS * ARG_ROTATED_ITEMS<InputT>;
template <typename InputT>
inline constexpr uint32_t ARG_ROTATED_TILE_COMPONENTS = ARG_ROTATED_SPAN<InputT> - (ARG_ROTATED_ITEMS<InputT> - 1);
static_assert(ARG_ROTATED_TILE_COMPONENTS<float> == FP32_RUNTIME_K_ROTATED_TILE_COMPONENTS);
static_assert(ARG_ROTATED_TILE_COMPONENTS<__half> == F16_RUNTIME_K_ROTATED_TILE_COMPONENTS);
static_assert(ARG_ROTATED_THREADS<__half> >= static_cast<int>(F16_RUNTIME_K_ROTATED_TILE_COMPONENTS));
static_assert((ARG_ROTATED_WARPS<__half> * 2) % 16 == 0 && (ARG_ROTATED_WARPS<float> * 4) % 16 == 0);

template <typename InputT, bool Staged>
__global__ __launch_bounds__(ARG_ROTATED_THREADS<InputT>) void runtimeKRotatedArgKernel(const InputT* input,
                                                                                Fp32ArgOutputs outputs,
                                                                                uint64_t* packed,
                                                                                uint64_t outer_size,
                                                                                uint64_t reduction_size,
                                                                                uint64_t inner_size,
                                                                                uint64_t shards_per_output,
                                                                                uint64_t component_tiles,
                                                                                uint32_t rows_per_shard) {
    static_assert(std::is_same_v<InputT, float> || std::is_same_v<InputT, __half> ||
                  std::is_same_v<InputT, __nv_bfloat16>);
    constexpr int Items = ARG_ROTATED_ITEMS<InputT>;
    constexpr int Warps = ARG_ROTATED_WARPS<InputT>;
    constexpr int Threads = ARG_ROTATED_THREADS<InputT>;
    constexpr uint32_t Span = ARG_ROTATED_SPAN<InputT>;
    constexpr uint32_t TileComponents = ARG_ROTATED_TILE_COMPONENTS<InputT>;
    __shared__ uint32_t warp_keys[Warps][Span];
    __shared__ uint32_t warp_tails[Warps][Span];
    __shared__ uint32_t warp_prefix[Warps];

    const uint32_t warp = threadIdx.x / TILED_REDUCTION_WARP_THREADS;
    const uint32_t lane = threadIdx.x % TILED_REDUCTION_WARP_THREADS;
    const uint32_t outer_count = static_cast<uint32_t>(outer_size);
    const uint32_t tile_count = static_cast<uint32_t>(component_tiles);
    const uint32_t shard_count = static_cast<uint32_t>(shards_per_output);
    for (uint32_t outer_index = blockIdx.z; outer_index < outer_count; outer_index += gridDim.z)
    for (uint32_t component_tile = blockIdx.y; component_tile < tile_count; component_tile += gridDim.y)
    for (uint32_t shard = blockIdx.x; shard < shard_count; shard += gridDim.x) {
        const uint64_t tile_begin = static_cast<uint64_t>(component_tile) * TileComponents;
        const uint32_t valid_components = static_cast<uint32_t>(minU64(TileComponents, inner_size - tile_begin));
        const uint64_t row_begin = static_cast<uint64_t>(shard) * rows_per_shard;
        const uint32_t shard_rows = row_begin < reduction_size
            ? static_cast<uint32_t>(minU64(rows_per_shard, reduction_size - row_begin))
            : 0U;

        uint32_t best_key[Items], best_tail[Items];
#pragma unroll
        for (int item = 0; item < Items; ++item) {
            best_key[item] = FP32_ARG_KEY_IDENTITY;
            best_tail[item] = 0xFFFFFFFFU;
        }
        uint32_t prefix = 0;
        if (warp < shard_rows) {
            const InputT* logical =
                input + (outer_index * reduction_size + row_begin + warp) * inner_size + tile_begin;
            const uintptr_t address = reinterpret_cast<uintptr_t>(logical);
            prefix = static_cast<uint32_t>((address & uintptr_t{15}) / sizeof(InputT));
            const InputT* lane_packet = reinterpret_cast<const InputT*>(address & ~uintptr_t{15}) + lane * Items;
            const int64_t lane_component = static_cast<int64_t>(tile_begin) - static_cast<int64_t>(prefix) +
                                           static_cast<int64_t>(lane * Items);
            if (lane_component < static_cast<int64_t>(inner_size)) {
                const uint64_t row_stride = static_cast<uint64_t>(Warps) * inner_size;
                for (uint32_t row = warp; row < shard_rows; row += Warps) {
                    const PackedInputValues<InputT, Items> values =
                        loadVectorizedInputPacket<InputT, Items>(lane_packet);
#pragma unroll
                    for (int item = 0; item < Items; ++item) {
                        uint32_t negative_zero;
                        const uint32_t key = fp32ArgKey(__float_as_uint(ToFp32<InputT>{}(values.values[item])),
                                                        outputs.xor_mask, negative_zero);
                        if (key < best_key[item]) {
                            best_key[item] = key;
                            best_tail[item] = (row << 1) | negative_zero;
                        }
                    }
                    lane_packet += row_stride;
                }
            }
        }
        if (lane == 0) warp_prefix[warp] = prefix;
#pragma unroll
        for (int item = 0; item < Items; ++item) {
            warp_keys[warp][lane * Items + item] = best_key[item];
            warp_tails[warp][lane * Items + item] = best_tail[item];
        }
        __syncthreads();
        for (uint32_t component = threadIdx.x; component < valid_components; component += Threads) {
            Fp32ArgCandidate reduced = fp32ArgIdentity();
#pragma unroll
            for (int source_warp = 0; source_warp < Warps; ++source_warp) {
                // Warps whose first row is beyond the shard published identities at prefix 0.
                const uint32_t slot = component + warp_prefix[source_warp];
                const uint32_t key = warp_keys[source_warp][slot];
                if (key != FP32_ARG_KEY_IDENTITY)
                    reduced = fp32ArgMin(reduced, Fp32ArgCandidate{key, warp_tails[source_warp][slot]});
            }
            const uint64_t output =
                (static_cast<uint64_t>(outer_index) * shards_per_output + shard) * inner_size + tile_begin + component;
            if constexpr (Staged) {
                packed[output] = fp32ArgPack(reduced);
            } else {
                static_cast<void>(packed);
                storeFp32ArgResult(outputs, output, reduced);
            }
        }
        __syncthreads();
    }
}

// The carried-index rotated ARG twin (see Fp32ArgWidePartials): runtimeKRotatedArgKernel's layout, reading the carried
// index packet at each value packet's element offset (both bases 16-byte aligned, so the windows match). Composed
// indices are not monotone in the row, so every merge is the full (key, tail) minimum. One shard writes the requested
// outputs; several write wide partials, finished by the wide continuation.
template <bool Staged, typename IndexT>
__global__ __launch_bounds__(FP32_ARG_ROTATED_THREADS) void runtimeKRotatedCarriedArgKernel(const float* input,
                                                                                       const IndexT* carried,
                                                                                       Fp32ArgOutputs outputs,
                                                                                       Fp32ArgWidePartials partials,
                                                                                       uint64_t outer_size,
                                                                                       uint64_t reduction_size,
                                                                                       uint64_t inner_size,
                                                                                       uint64_t shards_per_output,
                                                                                       uint64_t component_tiles,
                                                                                       uint32_t rows_per_shard,
                                                                                       uint64_t row_stride) {
    static_assert(std::is_same_v<IndexT, uint32_t> || std::is_same_v<IndexT, uint64_t>);
    constexpr uint32_t TileComponents = FP32_RUNTIME_K_ROTATED_TILE_COMPONENTS;
    __shared__ uint32_t warp_keys[FP32_ARG_ROTATED_WARPS][FP32_ARG_ROTATED_SPAN];
    __shared__ uint64_t warp_tails[FP32_ARG_ROTATED_WARPS][FP32_ARG_ROTATED_SPAN];
    __shared__ uint32_t warp_prefix[FP32_ARG_ROTATED_WARPS];

    const uint32_t warp = threadIdx.x / TILED_REDUCTION_WARP_THREADS;
    const uint32_t lane = threadIdx.x % TILED_REDUCTION_WARP_THREADS;
    const uint32_t outer_count = static_cast<uint32_t>(outer_size);
    const uint32_t tile_count = static_cast<uint32_t>(component_tiles);
    const uint32_t shard_count = static_cast<uint32_t>(shards_per_output);
    for (uint32_t outer_index = blockIdx.z; outer_index < outer_count; outer_index += gridDim.z)
    for (uint32_t component_tile = blockIdx.y; component_tile < tile_count; component_tile += gridDim.y)
    for (uint32_t shard = blockIdx.x; shard < shard_count; shard += gridDim.x) {
        const uint64_t tile_begin = static_cast<uint64_t>(component_tile) * TileComponents;
        const uint32_t valid_components = static_cast<uint32_t>(minU64(TileComponents, inner_size - tile_begin));
        const uint64_t row_begin = static_cast<uint64_t>(shard) * rows_per_shard;
        const uint32_t shard_rows = row_begin < reduction_size
            ? static_cast<uint32_t>(minU64(rows_per_shard, reduction_size - row_begin))
            : 0U;

        uint32_t best_key[4];
        uint64_t best_tail[4];
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            best_key[item] = FP32_ARG_KEY_IDENTITY;
            best_tail[item] = FP32_ARG_TAIL_IDENTITY;
        }
        uint32_t prefix = 0;
        if (warp < shard_rows) {
            const uint64_t logical_element =
                (outer_index * reduction_size + row_begin + warp) * inner_size + tile_begin;
            prefix = static_cast<uint32_t>(logical_element % 4);  // the bases are 16-byte aligned
            const uint64_t lane_element = logical_element - prefix + lane * 4;
            const float* lane_packet = input + lane_element;
            const IndexT* lane_carried = carried + lane_element;
            const int64_t lane_component = static_cast<int64_t>(tile_begin) - static_cast<int64_t>(prefix) +
                                           static_cast<int64_t>(lane * 4);
            if (lane_component < static_cast<int64_t>(inner_size)) {
                const uint64_t element_step = static_cast<uint64_t>(FP32_ARG_ROTATED_WARPS) * inner_size;
                uint64_t row_index = (row_begin + warp) * row_stride;
                const uint64_t row_index_step = static_cast<uint64_t>(FP32_ARG_ROTATED_WARPS) * row_stride;
                for (uint32_t row = warp; row < shard_rows; row += FP32_ARG_ROTATED_WARPS) {
                    const PackedInputValues<float, 4> values = loadVectorizedInputPacket<float, 4>(lane_packet);
                    const PackedInputValues<IndexT, 4> indices = loadVectorizedInputPacket<IndexT, 4>(lane_carried);
#pragma unroll
                    for (int item = 0; item < 4; ++item) {
                        uint32_t negative_zero;
                        const uint32_t key =
                            fp32ArgKey(__float_as_uint(values.values[item]), outputs.xor_mask, negative_zero);
                        const uint64_t composed = static_cast<uint64_t>(indices.values[item]) + row_index;
                        fp32ArgKeepMin(best_key[item], best_tail[item], key, (composed << 1) | negative_zero);
                    }
                    lane_packet += element_step;
                    lane_carried += element_step;
                    row_index += row_index_step;
                }
            }
        }
        if (lane == 0) warp_prefix[warp] = prefix;
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            warp_keys[warp][lane * 4 + item] = best_key[item];
            warp_tails[warp][lane * 4 + item] = best_tail[item];
        }
        __syncthreads();
        for (uint32_t component = threadIdx.x; component < valid_components; component += FP32_ARG_ROTATED_THREADS) {
            uint32_t reduced_key = FP32_ARG_KEY_IDENTITY;
            uint64_t reduced_tail = FP32_ARG_TAIL_IDENTITY;
#pragma unroll
            for (int source_warp = 0; source_warp < FP32_ARG_ROTATED_WARPS; ++source_warp) {
                // Warps whose first row is beyond the shard published identities at prefix 0.
                const uint32_t slot = component + warp_prefix[source_warp];
                fp32ArgKeepMin(reduced_key, reduced_tail, warp_keys[source_warp][slot], warp_tails[source_warp][slot]);
            }
            const uint64_t output =
                (static_cast<uint64_t>(outer_index) * shards_per_output + shard) * inner_size + tile_begin + component;
            if constexpr (Staged) {
                partials.keys[output] = reduced_key;
                partials.tails[output] = reduced_tail;
            } else {
                static_cast<void>(partials);
                storeFp32ArgResult(outputs, output, Fp32ArgCandidate{reduced_key, reduced_tail});
            }
        }
        __syncthreads();
    }
}

#if THOR_CUB_ENABLE_FP8_TYPES

// The FP8 RuntimeK rotated first stage, owned by RK::Fp8::Fp8RuntimeKPolicy: p16 packets with 16 cooperative warps,
// exact (512) or prefix-reserved (497) K tiles, and only the policy's row depths. Packet width is not a runtime axis
// and no other row depth is compiled.
constexpr int FP8_RUNTIME_K_ROTATED_PACKET_BYTES = 16;
static_assert(FP8_RUNTIME_K_ROTATED_WARPS == FP8_RUNTIME_K_ROTATED_PACKET_BYTES);

template <int RowsPerShard>
[[nodiscard]] constexpr bool fp8RuntimeKRotatedRows() {
    for (uint64_t retained : RK::Fp8::Fp8RuntimeKPolicy::rotated_row_depths) {
        if (retained == static_cast<uint64_t>(RowsPerShard)) return true;
    }
    return false;
}

// Invokes callback.template operator()<InputT, InputTransformT, RowsPerShard, ExactTile>() for the compiled rotated
// specialization matching the runtime request; returns false otherwise.
// IdentityOnly (folded FP8 MIN/MAX, see Fp8ExtremumFold) instantiates only the identity transform.
template <bool IdentityOnly = false, typename Callback>
[[nodiscard]] bool dispatchFp8RuntimeKRotated(DataType input_dtype,
                                              bool exact_tile,
                                              uint64_t rows_per_shard,
                                              ReductionStageInputTransform transform,
                                              Callback&& callback) {
    bool dispatched = false;
    const auto for_rows = [&]<typename InputT, typename InputTransformT, bool ExactTile>() {
        const auto rows = [&]<int RowsPerShard>() {
            if constexpr (fp8RuntimeKRotatedRows<RowsPerShard>()) {
                callback.template operator()<InputT, InputTransformT, RowsPerShard, ExactTile>();
                dispatched = true;
            }
        };
        switch (rows_per_shard) {
            case 32: rows.template operator()<32>(); return;
            case 64: rows.template operator()<64>(); return;
            case 128: rows.template operator()<128>(); return;
            case 256: rows.template operator()<256>(); return;
            case 512: rows.template operator()<512>(); return;
            case 1024: rows.template operator()<1024>(); return;
            default: return;
        }
    };
    const auto for_tile = [&]<typename InputT, typename InputTransformT>() {
        if (exact_tile) {
            for_rows.template operator()<InputT, InputTransformT, true>();
        } else {
            for_rows.template operator()<InputT, InputTransformT, false>();
        }
    };
    const auto for_transform = [&]<typename InputT>() {
        if constexpr (IdentityOnly) {
            if (transform == ReductionStageInputTransform::Identity) {
                for_tile.template operator()<InputT, IdentityFp32>();
            }
        } else {
            switch (transform) {
                case ReductionStageInputTransform::Identity:
                    for_tile.template operator()<InputT, IdentityFp32>(); return;
                case ReductionStageInputTransform::AbsoluteValue:
                    for_tile.template operator()<InputT, AbsoluteValueFp32>(); return;
                case ReductionStageInputTransform::Square:
                    for_tile.template operator()<InputT, SquareFp32>(); return;
            }
        }
    };
    switch (input_dtype) {
        case DataType::FP8_E4M3: for_transform.template operator()<__nv_fp8_e4m3>(); break;
        case DataType::FP8_E5M2: for_transform.template operator()<__nv_fp8_e5m2>(); break;
        default: break;
    }
    return dispatched;
}

#endif  // THOR_CUB_ENABLE_FP8_TYPES

}  // namespace

// FP8 MIN and MAX launch identity-transform kernels with the transform folded into the bytes. PrefixReserved MAX also
// folds its sign and launches the MIN kernel; Exact keeps its MAX kernel (transform-only fold).
template <typename ReductionOpT>
inline constexpr bool fp8RotatedFoldedExtremum =
    std::is_same_v<ReductionOpT, PropagatingMinimumFp32> || std::is_same_v<ReductionOpT, PropagatingMaximumFp32>;

template <typename ReductionOpT, bool ExactTile>
using Fp8RotatedPhysicalReductionOp =
    std::conditional_t<fp8RotatedFoldedExtremum<ReductionOpT> && !ExactTile, PropagatingMinimumFp32, ReductionOpT>;

template <typename ReductionOpT>
[[nodiscard]] Fp8ExtremumFold fp8RotatedExtremumFold(bool exact_tile, ReductionStageInputTransform transform) {
    if constexpr (!fp8RotatedFoldedExtremum<ReductionOpT>) {
        static_cast<void>(exact_tile);
        static_cast<void>(transform);
        return FP8_EXTREMUM_FOLD_NONE;
    } else {
        return fp8ExtremumFold(!exact_tile && std::is_same_v<ReductionOpT, PropagatingMaximumFp32>, transform);
    }
}

template <typename ReductionOpT>
void launchFp8RuntimeKRotatedFirstStageTyped(const StridedSpan& input,
                                             const StridedSpan& fp32_partials,
                                             const ReductionGeometry& geometry,
                                             bool exact_tile,
                                             uint64_t rows_per_shard,
                                             uint64_t shards_per_output,
                                             Stream& stream,
                                             ReductionStageInputTransform transform) {
#if THOR_CUB_ENABLE_FP8_TYPES
    if (rows_per_shard == 0 || shards_per_output <= 1
        || shards_per_output != ceilDivideU64(geometry.reduction_size, rows_per_shard)
        || fp32_partials.getDataType() != DataType::FP32
        || fp32_partials.getDimensions()
               != std::vector<uint64_t>{geometry.outer_size, shards_per_output, geometry.inner_size}) {
        throw std::logic_error("FP8 RuntimeK rotated first stage received invalid staged geometry.");
    }
    constexpr bool Fold = fp8RotatedFoldedExtremum<ReductionOpT>;
    const Fp8ExtremumFold fold = fp8RotatedExtremumFold<ReductionOpT>(exact_tile, transform);
    const bool launched = dispatchFp8RuntimeKRotated<Fold>(
        input.getDataType(), exact_tile, rows_per_shard, Fold ? ReductionStageInputTransform::Identity : transform,
        [&]<typename InputT, typename InputTransformT, int RowsPerShard, bool ExactTile>() {
            launchRuntimeKRotatedFirstStageForTransformPacket<
                InputT, Fp8RotatedPhysicalReductionOp<ReductionOpT, ExactTile>, InputTransformT, RowsPerShard,
                FP8_RUNTIME_K_ROTATED_PACKET_BYTES, FP8_RUNTIME_K_ROTATED_WARPS, ExactTile>(
                input, fp32_partials, geometry, shards_per_output, stream.getStream(), fold);
        });
    if (!launched) {
        throw std::logic_error("FP8 RuntimeK rotated first stage requires a retained compiled specialization.");
    }
#else
    static_cast<void>(input);
    static_cast<void>(fp32_partials);
    static_cast<void>(geometry);
    static_cast<void>(exact_tile);
    static_cast<void>(rows_per_shard);
    static_cast<void>(shards_per_output);
    static_cast<void>(stream);
    static_cast<void>(transform);
    throw std::logic_error("FP8 RuntimeK rotated first stage requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}

template <typename ReductionOpT>
KernelOccupancyInfo queryFp8RuntimeKRotatedOccupancyTyped(DataType input_dtype,
                                                          bool exact_tile,
                                                          uint64_t rows_per_shard,
                                                          ReductionStageInputTransform transform) {
#if THOR_CUB_ENABLE_FP8_TYPES
    constexpr bool Fold = fp8RotatedFoldedExtremum<ReductionOpT>;
    static_cast<void>(fp8RotatedExtremumFold<ReductionOpT>(exact_tile, transform));  // validates the transform
    KernelOccupancyInfo occupancy{};
    const bool queried = dispatchFp8RuntimeKRotated<Fold>(
        input_dtype, exact_tile, rows_per_shard, Fold ? ReductionStageInputTransform::Identity : transform,
        [&]<typename InputT, typename InputTransformT, int RowsPerShard, bool ExactTile>() {
            occupancy = queryRuntimeKRotatedOccupancyForRows<InputT, RowsPerShard,
                                                                        FP8_RUNTIME_K_ROTATED_PACKET_BYTES,
                                                                        FP8_RUNTIME_K_ROTATED_WARPS,
                                                                        ExactTile, InputTransformT,
                                                                        Fp8RotatedPhysicalReductionOp<ReductionOpT, ExactTile>>();
        });
    if (!queried) {
        // Not a compiled specialization; the selector treats this as an absent candidate.
        throw std::logic_error("FP8 RuntimeK rotated occupancy requires a retained compiled specialization.");
    }
    return occupancy;
#else
    static_cast<void>(input_dtype);
    static_cast<void>(exact_tile);
    static_cast<void>(rows_per_shard);
    static_cast<void>(transform);
    throw std::logic_error("FP8 RuntimeK rotated occupancy requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}

void launchFp8RuntimeKRotatedFirstStage(
    const StridedSpan& input,
    const StridedSpan& fp32_partials,
    const ReductionGeometry& geometry,
    bool exact_tile,
    uint64_t rows_per_shard,
    uint64_t shards_per_output,
    Stream& stream,
    ReductionStageInputTransform transform, ReductionStageCombine combine) {
    switch (combine) {
        case ReductionStageCombine::Sum:
            return launchFp8RuntimeKRotatedFirstStageTyped<cuda::std::plus<float>>(
                input, fp32_partials, geometry, exact_tile, rows_per_shard, shards_per_output, stream, transform);
        case ReductionStageCombine::Product:
            return launchFp8RuntimeKRotatedFirstStageTyped<cuda::std::multiplies<float>>(
                input, fp32_partials, geometry, exact_tile, rows_per_shard, shards_per_output, stream, transform);
        case ReductionStageCombine::Minimum:
            return launchFp8RuntimeKRotatedFirstStageTyped<PropagatingMinimumFp32>(
                input, fp32_partials, geometry, exact_tile, rows_per_shard, shards_per_output, stream, transform);
        case ReductionStageCombine::Maximum:
            return launchFp8RuntimeKRotatedFirstStageTyped<PropagatingMaximumFp32>(
                input, fp32_partials, geometry, exact_tile, rows_per_shard, shards_per_output, stream, transform);
    }
    throw std::logic_error("Invalid RuntimeK Rotated combine operation.");
}

KernelOccupancyInfo queryFp8RuntimeKRotatedOccupancy(
    DataType input_dtype, bool exact_tile, uint64_t rows_per_shard,
    ReductionStageInputTransform transform, ReductionStageCombine combine) {
    switch (combine) {
        case ReductionStageCombine::Sum:
            return queryFp8RuntimeKRotatedOccupancyTyped<cuda::std::plus<float>>(
                input_dtype, exact_tile, rows_per_shard, transform);
        case ReductionStageCombine::Product:
            return queryFp8RuntimeKRotatedOccupancyTyped<cuda::std::multiplies<float>>(
                input_dtype, exact_tile, rows_per_shard, transform);
        case ReductionStageCombine::Minimum:
            return queryFp8RuntimeKRotatedOccupancyTyped<PropagatingMinimumFp32>(
                input_dtype, exact_tile, rows_per_shard, transform);
        case ReductionStageCombine::Maximum:
            return queryFp8RuntimeKRotatedOccupancyTyped<PropagatingMaximumFp32>(
                input_dtype, exact_tile, rows_per_shard, transform);
    }
    throw std::logic_error("Invalid RuntimeK Rotated occupancy combine operation.");
}

bool fp32RuntimeKRotatedSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size,
                                 uint64_t rows_per_shard) {
    const auto& depths = RK::Fp8::Fp8RuntimeKPolicy::rotated_row_depths;
    if (outer_size == 0 || inner_size == 0 || rows_per_shard >= reduction_size ||
        std::find(depths.begin(), depths.end(), rows_per_shard) == depths.end()) {
        return false;
    }
    const uint64_t tiles = ceilDivideU64(inner_size, FP32_RUNTIME_K_ROTATED_TILE_COMPONENTS);
    return RK::rkWorkGridFits(ceilDivideU64(reduction_size, rows_per_shard), tiles, outer_size);
}

void launchFp32RuntimeKRotatedFirstStage(const StridedSpan& input,
                                         const StridedSpan& fp32_partials,
                                         const ReductionGeometry& geometry,
                                         uint64_t rows_per_shard,
                                         uint64_t shards_per_output,
                                         Stream& stream,
                                         ReductionStageInputTransform transform,
                                         ReductionStageCombine combine) {
    switch (combine) {
        case ReductionStageCombine::Sum:
            return launchFp32RuntimeKRotatedFirstStageTyped<cuda::std::plus<float>>(
                input, fp32_partials, geometry, rows_per_shard, shards_per_output, stream, transform);
        case ReductionStageCombine::Product:
            return launchFp32RuntimeKRotatedFirstStageTyped<cuda::std::multiplies<float>>(
                input, fp32_partials, geometry, rows_per_shard, shards_per_output, stream, transform);
        case ReductionStageCombine::Minimum:
            return launchFp32RuntimeKRotatedFirstStageTyped<PropagatingMinimumFp32>(
                input, fp32_partials, geometry, rows_per_shard, shards_per_output, stream, transform);
        case ReductionStageCombine::Maximum:
            return launchFp32RuntimeKRotatedFirstStageTyped<PropagatingMaximumFp32>(
                input, fp32_partials, geometry, rows_per_shard, shards_per_output, stream, transform);
    }
    throw std::logic_error("Invalid FP32 RuntimeK rotated combine operation.");
}

KernelOccupancyInfo queryFp32RuntimeKRotatedOccupancy(uint64_t rows_per_shard,
                                                      ReductionStageInputTransform transform,
                                                      ReductionStageCombine combine) {
    switch (combine) {
        case ReductionStageCombine::Sum:
            return queryFp32RuntimeKRotatedOccupancyTyped<cuda::std::plus<float>>(rows_per_shard, transform);
        case ReductionStageCombine::Product:
            return queryFp32RuntimeKRotatedOccupancyTyped<cuda::std::multiplies<float>>(rows_per_shard, transform);
        case ReductionStageCombine::Minimum:
            return queryFp32RuntimeKRotatedOccupancyTyped<PropagatingMinimumFp32>(rows_per_shard, transform);
        case ReductionStageCombine::Maximum:
            return queryFp32RuntimeKRotatedOccupancyTyped<PropagatingMaximumFp32>(rows_per_shard, transform);
    }
    throw std::logic_error("Invalid FP32 RuntimeK rotated occupancy combine operation.");
}

bool f16RuntimeKRotatedSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size,
                                uint64_t rows_per_shard) {
    const auto& depths = RK::Fp8::Fp8RuntimeKPolicy::rotated_row_depths;
    if (outer_size == 0 || inner_size == 0 || rows_per_shard >= reduction_size ||
        std::find(depths.begin(), depths.end(), rows_per_shard) == depths.end()) {
        return false;
    }
    const uint64_t tiles = ceilDivideU64(inner_size, F16_RUNTIME_K_ROTATED_TILE_COMPONENTS);
    return RK::rkWorkGridFits(ceilDivideU64(reduction_size, rows_per_shard), tiles, outer_size);
}

void launchF16RuntimeKRotatedFirstStage(const StridedSpan& input,
                                        const StridedSpan& fp32_partials,
                                        const ReductionGeometry& geometry,
                                        uint64_t rows_per_shard,
                                        uint64_t shards_per_output,
                                        Stream& stream,
                                        ReductionStageInputTransform transform,
                                        ReductionStageCombine combine) {
    const bool launched = dispatchF16RuntimeKRotatedTypes(
        input.getDataType(), combine, [&]<typename InputT, typename ReductionOpT>() {
            launchF16RuntimeKRotatedFirstStageTyped<InputT, ReductionOpT>(input, fp32_partials, geometry,
                                                                          rows_per_shard, shards_per_output, stream,
                                                                          transform);
        });
    if (!launched) throw std::logic_error("FP16 / BF16 RuntimeK rotated first stage received an unsupported input.");
}

KernelOccupancyInfo queryF16RuntimeKRotatedOccupancy(DataType input_dtype, uint64_t rows_per_shard,
                                                     ReductionStageInputTransform transform,
                                                     ReductionStageCombine combine) {
    KernelOccupancyInfo occupancy{};
    const bool queried = dispatchF16RuntimeKRotatedTypes(
        input_dtype, combine, [&]<typename InputT, typename ReductionOpT>() {
            occupancy = queryF16RuntimeKRotatedOccupancyTyped<InputT, ReductionOpT>(rows_per_shard, transform);
        });
    if (!queried) throw std::logic_error("FP16 / BF16 RuntimeK rotated occupancy requires FP16 / BF16 input.");
    return occupancy;
}

namespace {
// The value dtypes storeFp32AsRuntimeDType writes.
[[nodiscard]] bool fp32ArgValueStorageSupported(DataType dtype) {
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
}  // namespace

namespace {
// One input type's rotated ARG pass (FP32, or FP16 / BF16); `name` labels the errors.
[[nodiscard]] bool runtimeKArgSupportsTile(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size,
                                           uint64_t rows_per_shard, uint64_t tile_components) {
    if (outer_size == 0 || reduction_size == 0 || inner_size == 0 || rows_per_shard == 0 ||
        std::min(rows_per_shard, reduction_size) > FP32_ARG_MAX_SHARD_ROWS) {
        return false;
    }
    const uint64_t shards = ceilDivideU64(reduction_size, rows_per_shard);
    return RK::rkWorkGridFits(shards, ceilDivideU64(inner_size, tile_components), outer_size);
}

template <typename InputT>
void launchRuntimeKArgTyped(const StridedSpan& input,
                            const StridedSpan* value_output,
                            const StridedSpan* index_output,
                            const ReductionGeometry& geometry,
                            uint64_t rows_per_shard,
                            uint64_t* packed_scratch,
                            bool maximum,
                            uint64_t domain_stride,
                            Stream& stream,
                            const char* name) {
    constexpr uint64_t Tile = ARG_ROTATED_TILE_COMPONENTS<InputT>;
    const uint64_t shards = rows_per_shard == 0 ? 0 : ceilDivideU64(geometry.reduction_size, rows_per_shard);
    const uint64_t outputs = geometry.outer_size * geometry.inner_size;
    if (!runtimeKArgSupportsTile(geometry.outer_size, geometry.reduction_size, geometry.inner_size, rows_per_shard,
                                 Tile) ||
        input.getTotalNumElements() != geometry.outer_size * geometry.reduction_size * geometry.inner_size ||
        (reinterpret_cast<uintptr_t>(input.getMemPtr<void>()) & uintptr_t{sizeof(InputT) - 1}) != 0 ||
        (shards > 1 && packed_scratch == nullptr) || (value_output == nullptr && index_output == nullptr) ||
        (value_output != nullptr && (value_output->getTotalNumElements() != outputs ||
                                     !fp32ArgValueStorageSupported(value_output->getDataType()))) ||
        (index_output != nullptr && (index_output->getTotalNumElements() != outputs ||
                                     (index_output->getDataType() != DataType::UINT32 &&
                                      index_output->getDataType() != DataType::UINT64)))) {
        throw std::logic_error(std::string(name) + " RuntimeK ARG received unsupported geometry, outputs or scratch.");
    }
    if (index_output != nullptr && index_output->getDataType() == DataType::UINT32 &&
        (geometry.reduction_size - 1) * domain_stride > std::numeric_limits<uint32_t>::max()) {
        throw std::logic_error(std::string(name) + " RuntimeK ARG UINT32 index cannot hold the original-domain index.");
    }
    const uint64_t tiles = ceilDivideU64(geometry.inner_size, Tile);
    const dim3 grid = reductionWorkGrid(shards, tiles, geometry.outer_size);
    const uint32_t depth = static_cast<uint32_t>(std::min(rows_per_shard, geometry.reduction_size));
    const Fp32ArgOutputs arg_outputs{value_output == nullptr ? nullptr : value_output->getMemPtr<void>(),
                                     value_output == nullptr ? DataType::FP32 : value_output->getDataType(),
                                     index_output == nullptr ? nullptr : index_output->getMemPtr<void>(),
                                     index_output == nullptr ? DataType::UINT32 : index_output->getDataType(),
                                     domain_stride,
                                     maximum ? 0x80000000U : 0U};
    if (shards == 1) {
        runtimeKRotatedArgKernel<InputT, false><<<grid, ARG_ROTATED_THREADS<InputT>, 0, stream.getStream()>>>(
            input.getMemPtr<InputT>(), arg_outputs, nullptr, geometry.outer_size, geometry.reduction_size,
            geometry.inner_size, shards, tiles, depth);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    runtimeKRotatedArgKernel<InputT, true><<<grid, ARG_ROTATED_THREADS<InputT>, 0, stream.getStream()>>>(
        input.getMemPtr<InputT>(), arg_outputs, packed_scratch, geometry.outer_size, geometry.reduction_size,
        geometry.inner_size, shards, tiles, depth);
    CUDA_CHECK(cudaGetLastError());
    launchFp32ArgPackedContinuation(packed_scratch, value_output, index_output, geometry.outer_size,
                                    geometry.reduction_size, geometry.inner_size, shards, rows_per_shard, maximum,
                                    domain_stride, stream);
}

template <typename InputT>
[[nodiscard]] KernelOccupancyInfo queryRuntimeKArgOccupancyTyped(bool staged) {
    const auto query = [&](auto kernel) {
        cudaFuncAttributes attributes{};
        CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
        int active_blocks = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks, kernel, ARG_ROTATED_THREADS<InputT>,
                                                                 0));
        return KernelOccupancyInfo{attributes.numRegs, attributes.sharedSizeBytes, 0, active_blocks,
                                   active_blocks * ARG_ROTATED_WARPS<InputT>};
    };
    return staged ? query(runtimeKRotatedArgKernel<InputT, true>) : query(runtimeKRotatedArgKernel<InputT, false>);
}
}  // namespace

bool fp32RuntimeKArgSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size,
                             uint64_t rows_per_shard) {
    return runtimeKArgSupportsTile(outer_size, reduction_size, inner_size, rows_per_shard,
                                   FP32_RUNTIME_K_ROTATED_TILE_COMPONENTS);
}

bool f16RuntimeKArgSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size,
                            uint64_t rows_per_shard) {
    return runtimeKArgSupportsTile(outer_size, reduction_size, inner_size, rows_per_shard,
                                   F16_RUNTIME_K_ROTATED_TILE_COMPONENTS);
}

void launchFp32RuntimeKArg(const StridedSpan& input,
                           const StridedSpan* value_output,
                           const StridedSpan* index_output,
                           const ReductionGeometry& geometry,
                           uint64_t rows_per_shard,
                           uint64_t* packed_scratch,
                           bool maximum,
                           uint64_t domain_stride,
                           Stream& stream) {
    if (input.getDataType() != DataType::FP32)
        throw std::logic_error("FP32 RuntimeK ARG received unsupported geometry, outputs or scratch.");
    launchRuntimeKArgTyped<float>(input, value_output, index_output, geometry, rows_per_shard, packed_scratch, maximum,
                                  domain_stride, stream, "FP32");
}

void launchF16RuntimeKArg(const StridedSpan& input,
                          const StridedSpan* value_output,
                          const StridedSpan* index_output,
                          const ReductionGeometry& geometry,
                          uint64_t rows_per_shard,
                          uint64_t* packed_scratch,
                          bool maximum,
                          uint64_t domain_stride,
                          Stream& stream) {
    switch (input.getDataType()) {
        case DataType::FP16:
            return launchRuntimeKArgTyped<__half>(input, value_output, index_output, geometry, rows_per_shard,
                                                  packed_scratch, maximum, domain_stride, stream, "FP16");
        case DataType::BF16:
            return launchRuntimeKArgTyped<__nv_bfloat16>(input, value_output, index_output, geometry, rows_per_shard,
                                                         packed_scratch, maximum, domain_stride, stream, "BF16");
        default:
            throw std::logic_error("FP16 / BF16 RuntimeK ARG received a non-FP16 / BF16 input.");
    }
}

KernelOccupancyInfo queryF16RuntimeKArgOccupancy(DataType input_dtype, bool staged) {
    switch (input_dtype) {
        case DataType::FP16: return queryRuntimeKArgOccupancyTyped<__half>(staged);
        case DataType::BF16: return queryRuntimeKArgOccupancyTyped<__nv_bfloat16>(staged);
        default: throw std::logic_error("FP16 / BF16 RuntimeK ARG occupancy requires FP16 / BF16 input.");
    }
}

// ---- The carried-index rotated ARG twin (see runtimeKRotatedCarriedArgKernel).
bool fp32RuntimeKCarriedArgSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size,
                                    uint64_t rows_per_shard) {
    if (outer_size == 0 || reduction_size == 0 || inner_size == 0 || rows_per_shard == 0 ||
        std::min(rows_per_shard, reduction_size) > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    const uint64_t shards = ceilDivideU64(reduction_size, rows_per_shard);
    return RK::rkWorkGridFits(shards, ceilDivideU64(inner_size, FP32_RUNTIME_K_ROTATED_TILE_COMPONENTS),
                              outer_size);
}

void launchFp32RuntimeKCarriedArg(const StridedSpan& input,
                                  const StridedSpan& carried_index,
                                  const StridedSpan* value_output,
                                  const StridedSpan* index_output,
                                  const ReductionGeometry& geometry,
                                  uint64_t rows_per_shard,
                                  void* wide_scratch,
                                  bool maximum,
                                  uint64_t row_stride,
                                  Stream& stream) {
    const uint64_t shards = rows_per_shard == 0 ? 0 : ceilDivideU64(geometry.reduction_size, rows_per_shard);
    const uint64_t outputs = geometry.outer_size * geometry.inner_size;
    const uint64_t elements = geometry.outer_size * geometry.reduction_size * geometry.inner_size;
    if (input.getDataType() != DataType::FP32 ||
        !fp32RuntimeKCarriedArgSupports(geometry.outer_size, geometry.reduction_size, geometry.inner_size,
                                        rows_per_shard) ||
        input.getTotalNumElements() != elements ||
        (reinterpret_cast<uintptr_t>(input.getMemPtr<void>()) & uintptr_t{15}) != 0 ||
        (carried_index.getDataType() != DataType::UINT32 && carried_index.getDataType() != DataType::UINT64) ||
        carried_index.getTotalNumElements() != elements ||
        (reinterpret_cast<uintptr_t>(carried_index.getMemPtr<void>()) & uintptr_t{15}) != 0 ||
        (shards > 1 && wide_scratch == nullptr) ||
        (shards > 1 && (reinterpret_cast<uintptr_t>(wide_scratch) & uintptr_t{255}) != 0) ||
        (value_output == nullptr && index_output == nullptr) ||
        (value_output != nullptr && (value_output->getTotalNumElements() != outputs ||
                                     !fp32ArgValueStorageSupported(value_output->getDataType()))) ||
        (index_output != nullptr && (index_output->getTotalNumElements() != outputs ||
                                     (index_output->getDataType() != DataType::UINT32 &&
                                      index_output->getDataType() != DataType::UINT64)))) {
        throw std::logic_error("FP32 RuntimeK carried ARG received unsupported geometry, inputs, outputs or scratch.");
    }
    const uint64_t tiles = ceilDivideU64(geometry.inner_size, FP32_RUNTIME_K_ROTATED_TILE_COMPONENTS);
    const dim3 grid = reductionWorkGrid(shards, tiles, geometry.outer_size);
    const uint32_t depth = static_cast<uint32_t>(std::min(rows_per_shard, geometry.reduction_size));
    // The tails hold the composed index: the index stride is 1.
    const Fp32ArgOutputs arg_outputs{value_output == nullptr ? nullptr : value_output->getMemPtr<void>(),
                                     value_output == nullptr ? DataType::FP32 : value_output->getDataType(),
                                     index_output == nullptr ? nullptr : index_output->getMemPtr<void>(),
                                     index_output == nullptr ? DataType::UINT32 : index_output->getDataType(),
                                     1,
                                     maximum ? 0x80000000U : 0U};
    Fp32ArgWidePartials partials{nullptr, nullptr};
    if (shards > 1) {
        auto* bytes = static_cast<uint8_t*>(wide_scratch);
        const uint64_t count = geometry.outer_size * shards * geometry.inner_size;
        partials = {reinterpret_cast<uint32_t*>(bytes),
                    reinterpret_cast<uint64_t*>(bytes + ((count * sizeof(uint32_t) + 255) & ~uint64_t{255}))};
    }
    const auto launch = [&]<bool Staged, typename IndexT>() {
        runtimeKRotatedCarriedArgKernel<Staged, IndexT><<<grid, FP32_ARG_ROTATED_THREADS, 0, stream.getStream()>>>(
            input.getMemPtr<float>(), carried_index.getMemPtr<IndexT>(), arg_outputs, partials, geometry.outer_size,
            geometry.reduction_size, geometry.inner_size, shards, tiles, depth, row_stride);
        CUDA_CHECK(cudaGetLastError());
    };
    const bool wide = carried_index.getDataType() == DataType::UINT64;
    if (shards == 1) {
        if (wide) launch.template operator()<false, uint64_t>();
        else launch.template operator()<false, uint32_t>();
        return;
    }
    if (wide) launch.template operator()<true, uint64_t>();
    else launch.template operator()<true, uint32_t>();
    launchFp32ArgWideContinuation(wide_scratch, value_output, index_output, geometry.outer_size, shards,
                                  geometry.inner_size, maximum, stream);
}

KernelOccupancyInfo queryFp32RuntimeKCarriedArgOccupancy(bool staged, DataType index_dtype) {
    if (index_dtype != DataType::UINT32 && index_dtype != DataType::UINT64)
        throw std::logic_error("FP32 RuntimeK carried ARG occupancy requires a UINT32 or UINT64 carried index.");
    const auto query = [&](auto kernel) {
        cudaFuncAttributes attributes{};
        CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
        int active_blocks = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks, kernel, FP32_ARG_ROTATED_THREADS, 0));
        return KernelOccupancyInfo{attributes.numRegs, attributes.sharedSizeBytes, 0, active_blocks,
                                   active_blocks * FP32_ARG_ROTATED_WARPS};
    };
    const bool wide = index_dtype == DataType::UINT64;
    if (staged) return wide ? query(runtimeKRotatedCarriedArgKernel<true, uint64_t>)
                            : query(runtimeKRotatedCarriedArgKernel<true, uint32_t>);
    return wide ? query(runtimeKRotatedCarriedArgKernel<false, uint64_t>)
                : query(runtimeKRotatedCarriedArgKernel<false, uint32_t>);
}

KernelOccupancyInfo queryFp32RuntimeKArgOccupancy(bool staged) {
    return queryRuntimeKArgOccupancyTyped<float>(staged);
}

uint32_t fp8RuntimeKArgTileComponents(uint32_t packet_bytes) {
    if (packet_bytes == 8) return fp8ArgRotatedTileComponents<8>;
    throw std::logic_error("FP8 RuntimeK ARG compiles only the 8-byte packet.");
}

bool fp8RuntimeKArgSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size,
                            uint64_t rows_per_shard, uint32_t packet_bytes) {
    if (outer_size == 0 || reduction_size == 0 || inner_size == 0 || rows_per_shard == 0 ||
        rows_per_shard >= FP8_ARG_MAX_LANE_ROWS || rows_per_shard > std::numeric_limits<uint32_t>::max() ||
        packet_bytes != 8) {
        return false;
    }
    const uint64_t shards = ceilDivideU64(reduction_size, rows_per_shard);
    const uint64_t tiles = ceilDivideU64(inner_size, fp8RuntimeKArgTileComponents(packet_bytes));
    constexpr uint64_t MaxExtent = (uint64_t{1} << 31) - 1;  // reductionWorkGrid
    return shards <= MaxExtent && tiles <= MaxExtent && outer_size <= MaxExtent;
}

void launchFp8RuntimeKArgPass(const StridedSpan& input,
                              const StridedSpan* value_output,
                              const StridedSpan* index_output,
                              const ReductionGeometry& geometry,
                              uint64_t rows_per_shard,
                              uint32_t packet_bytes,
                              bool maximum,
                              uint64_t domain_stride,
                              Stream& stream,
                              uint32_t* packed_partials) {
#if THOR_CUB_ENABLE_FP8_TYPES
    const DataType input_dtype = input.getDataType();
    const uint64_t shards = rows_per_shard == 0 ? 0 : ceilDivideU64(geometry.reduction_size, rows_per_shard);
    const uint64_t outputs = geometry.outer_size * shards * geometry.inner_size;
    const bool packed = packed_partials != nullptr;
    if ((input_dtype != DataType::FP8_E4M3 && input_dtype != DataType::FP8_E5M2) ||
        !fp8RuntimeKArgSupports(geometry.outer_size, geometry.reduction_size, geometry.inner_size, rows_per_shard,
                                packet_bytes) ||
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
        throw std::logic_error("FP8 RuntimeK ARG received unsupported geometry or outputs.");
    }
    if (index_output != nullptr && index_output->getDataType() == DataType::UINT32 &&
        (geometry.reduction_size - 1) * domain_stride > std::numeric_limits<uint32_t>::max()) {
        throw std::logic_error("FP8 RuntimeK ARG UINT32 index cannot hold the original-domain index.");
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
    const uint64_t tiles = ceilDivideU64(geometry.inner_size, fp8RuntimeKArgTileComponents(packet_bytes));
    const dim3 grid = reductionWorkGrid(shards, tiles, geometry.outer_size);
    const auto* bytes = static_cast<const uint8_t*>(input.getMemPtr<void>());
    fp8RuntimeKRotatedArgKernel<8><<<grid, FP8_ARG_ROTATED_THREADS, 0, stream.getStream()>>>(
        bytes, arg_outputs, geometry.outer_size, geometry.reduction_size, geometry.inner_size, shards, tiles,
        static_cast<uint32_t>(rows_per_shard));
    CUDA_CHECK(cudaGetLastError());
#else
    static_cast<void>(input);
    static_cast<void>(value_output);
    static_cast<void>(index_output);
    static_cast<void>(geometry);
    static_cast<void>(rows_per_shard);
    static_cast<void>(packet_bytes);
    static_cast<void>(maximum);
    static_cast<void>(domain_stride);
    static_cast<void>(stream);
    static_cast<void>(packed_partials);
    throw std::logic_error("FP8 RuntimeK ARG requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}

// RuntimeK PacketComplete: the VALUE launchers (FP8, FP32, FP16 / BF16) and the ARG launchers.
template <int FixedOp>
void launchFp8PacketCompleteTyped(
    const StridedSpan& input,
    const StridedSpan& output,
    const ReductionGeometry& geometry,
    uint32_t warps_per_output,
    uint32_t tile_components,
    Stream& stream,
    const ReductionStageSemantics& semantics,
    float output_scale) {
#if THOR_CUB_ENABLE_FP8_TYPES
    const AdditiveFinalizeFp32 output_finalize = fp8CompleteFinalizerFromSemantics(semantics);
    if (geometry.outer_size == 0 || geometry.reduction_size == 0 || geometry.inner_size == 0 ||
        geometry.inner_size > RK::RK_MAX_UINT32 ||
        // The kernel's input and output offsets are 64-bit; only outer, R and K are 32-bit operands, and the 32-bit row
        // counter steps by W <= 8.
        geometry.outer_size > RK::RK_MAX_UINT32 ||
        geometry.reduction_size > RK::RK_PACKET_COMPLETE_MAX_ROWS ||
        !fp8CompleteSupportsOutputStorage(output.getDataType()) || output.getDimensions() != geometry.output_dimensions ||
        tile_components == 0 || tile_components > 496) {
        throw std::logic_error("FP8 RuntimeK PacketComplete received unsupported geometry.");
    }
    // RuntimeK packet-complete deliberately spends three compiled input-transform
    // variants per input dtype. Dispatch only once on the host; no uniform transform
    // branch remains in the hot device accumulation loop. Output dtype and additive
    // finalization stay runtime parameters and do not multiply this inventory.
    const auto dispatch = [&]<typename InputT>() {
        switch (semantics.input_transform) {
            case ReductionStageInputTransform::Identity:
                launchPacketCompleteValueKernel<InputT, 0, FixedOp>(
                    input, output, geometry, warps_per_output, tile_components,
                    stream.getStream(), output_finalize,
                    ReductionStageInputTransform::Identity, output_scale);
                return;
            case ReductionStageInputTransform::AbsoluteValue:
                launchPacketCompleteValueKernel<InputT, 1, FixedOp>(
                    input, output, geometry, warps_per_output, tile_components,
                    stream.getStream(), output_finalize,
                    ReductionStageInputTransform::Identity, output_scale);
                return;
            case ReductionStageInputTransform::Square:
                launchPacketCompleteValueKernel<InputT, 2, FixedOp>(
                    input, output, geometry, warps_per_output, tile_components,
                    stream.getStream(), output_finalize,
                    ReductionStageInputTransform::Identity, output_scale);
                return;
        }
        throw std::logic_error("FP8 RuntimeK PacketComplete received unsupported input transform.");
    };
    switch (input.getDataType()) {
        case DataType::FP8_E4M3: dispatch.template operator()<__nv_fp8_e4m3>(); return;
        case DataType::FP8_E5M2: dispatch.template operator()<__nv_fp8_e5m2>(); return;
        default:
            throw std::logic_error("FP8 RuntimeK PacketComplete requires FP8 input.");
    }
#else
    static_cast<void>(input);
    static_cast<void>(output);
    static_cast<void>(geometry);
    static_cast<void>(warps_per_output);
    static_cast<void>(tile_components);
    static_cast<void>(stream);
    static_cast<void>(semantics);
    static_cast<void>(output_scale);
    throw std::logic_error("FP8 RuntimeK PacketComplete requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}


void launchFp8PacketComplete(
    const StridedSpan& input,
    const StridedSpan& output,
    const ReductionGeometry& geometry,
    uint32_t warps_per_output,
    uint32_t tile_components,
    Stream& stream,
    const ReductionStageSemantics& semantics,
    float output_scale) {
    switch (semantics.combine) {
        case ReductionStageCombine::Sum:
            return launchFp8PacketCompleteTyped<0>(input, output, geometry, warps_per_output, tile_components, stream, semantics, output_scale);
        case ReductionStageCombine::Product:
            return launchFp8PacketCompleteTyped<1>(input, output, geometry, warps_per_output, tile_components, stream, semantics, output_scale);
        case ReductionStageCombine::Minimum:
            return launchFp8PacketCompleteTyped<2>(input, output, geometry, warps_per_output, tile_components, stream, semantics, output_scale);
        case ReductionStageCombine::Maximum:
            return launchFp8PacketCompleteTyped<3>(input, output, geometry, warps_per_output, tile_components, stream, semantics, output_scale);
    }
    throw std::logic_error("Invalid RuntimeK Complete combine operation.");
}

// Query the transform-specific Complete specialization; its register usage and
// occupancy may differ across input transforms.
template <int FixedOp>
KernelOccupancyInfo queryFp8PacketCompleteOccupancyTyped(
    DataType input_dtype, ReductionStageInputTransform transform) {
#if THOR_CUB_ENABLE_FP8_TYPES
    const auto dispatch = [&]<typename InputT>() -> KernelOccupancyInfo {
        switch (transform) {
            case ReductionStageInputTransform::Identity:
                return queryShallowCompleteOccupancy(packetCompleteValueKernel<InputT, 0, FixedOp>);
            case ReductionStageInputTransform::AbsoluteValue:
                return queryShallowCompleteOccupancy(packetCompleteValueKernel<InputT, 1, FixedOp>);
            case ReductionStageInputTransform::Square:
                return queryShallowCompleteOccupancy(packetCompleteValueKernel<InputT, 2, FixedOp>);
        }
        throw std::logic_error("RuntimeK Complete occupancy requires a supported input transform.");
    };
    switch (input_dtype) {
        case DataType::FP8_E4M3: return dispatch.template operator()<__nv_fp8_e4m3>();
        case DataType::FP8_E5M2: return dispatch.template operator()<__nv_fp8_e5m2>();
        default:
            throw std::logic_error("FP8 RuntimeK PacketComplete occupancy requires FP8 input.");
    }
#else
    static_cast<void>(input_dtype); static_cast<void>(transform);
    throw std::logic_error("FP8 RuntimeK PacketComplete occupancy requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}


KernelOccupancyInfo queryFp8PacketCompleteOccupancy(
    DataType input_dtype, ReductionStageInputTransform transform, ReductionStageCombine combine) {
    switch (combine) {
        case ReductionStageCombine::Sum:
            return queryFp8PacketCompleteOccupancyTyped<0>(input_dtype, transform);
        case ReductionStageCombine::Product:
            return queryFp8PacketCompleteOccupancyTyped<1>(input_dtype, transform);
        case ReductionStageCombine::Minimum:
            return queryFp8PacketCompleteOccupancyTyped<2>(input_dtype, transform);
        case ReductionStageCombine::Maximum:
            return queryFp8PacketCompleteOccupancyTyped<3>(input_dtype, transform);
    }
    throw std::logic_error("Invalid RuntimeK Complete occupancy combine operation.");
}

static_assert(FP32_PACKET_COMPLETE_MAX_TILE_COMPONENTS == 124, "the FP32 PacketComplete kernel's tile cap");
static_assert(F16_PACKET_COMPLETE_MAX_TILE_COMPONENTS == 248, "the FP16 / BF16 PacketComplete kernel's tile cap");

// RuntimeK PacketComplete on FP32 input: the shared kernel at InputT = float (tile <= 124 components), the same
// W1/W2/W4/W8 ownership and three compiled transforms per combine (12 kernels); FP16 / BF16 (tile <= 248 components),
// 12 kernels per format through the same typed launcher.
namespace {
[[nodiscard]] bool packetCompleteSupportsTile(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size,
                                              uint32_t warps_per_output, uint32_t tile_components,
                                              uint32_t max_tile_components) {
    return outer_size != 0 && reduction_size != 0 && inner_size != 0 && outer_size <= RK::RK_MAX_UINT32 &&
           reduction_size <= RK::RK_PACKET_COMPLETE_MAX_ROWS && inner_size <= RK::RK_MAX_UINT32 &&
           (warps_per_output == 1 || warps_per_output == 2 || warps_per_output == 4 || warps_per_output == 8) &&
           tile_components != 0 && tile_components <= max_tile_components;
}
}  // namespace

bool fp32PacketCompleteSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size,
                                uint32_t warps_per_output, uint32_t tile_components) {
    return packetCompleteSupportsTile(outer_size, reduction_size, inner_size, warps_per_output, tile_components,
                                      FP32_PACKET_COMPLETE_MAX_TILE_COMPONENTS);
}

bool f16PacketCompleteSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size,
                               uint32_t warps_per_output, uint32_t tile_components) {
    return packetCompleteSupportsTile(outer_size, reduction_size, inner_size, warps_per_output, tile_components,
                                      F16_PACKET_COMPLETE_MAX_TILE_COMPONENTS);
}

namespace {

template <typename InputT, int FixedOp, typename Callback>
[[nodiscard]] bool dispatchPacketCompleteTransform(ReductionStageInputTransform transform, Callback&& callback) {
    switch (transform) {
        case ReductionStageInputTransform::Identity:
            callback(packetCompleteValueKernel<InputT, 0, FixedOp>);
            return true;
        case ReductionStageInputTransform::AbsoluteValue:
            callback(packetCompleteValueKernel<InputT, 1, FixedOp>);
            return true;
        case ReductionStageInputTransform::Square:
            callback(packetCompleteValueKernel<InputT, 2, FixedOp>);
            return true;
    }
    return false;
}

// One input type's PacketComplete (FP32 or FP16 / BF16; `name` labels the errors).
template <typename InputT, int FixedOp>
void launchPacketCompleteTyped(const StridedSpan& input,
                               const StridedSpan& output,
                               const ReductionGeometry& geometry,
                               uint32_t warps_per_output,
                               uint32_t tile_components,
                               Stream& stream,
                               const ReductionStageSemantics& semantics,
                               float output_scale,
                               const char* name) {
    const AdditiveFinalizeFp32 output_finalize = fp8CompleteFinalizerFromSemantics(semantics);
    const uint32_t max_tile = std::is_same_v<InputT, float> ? FP32_PACKET_COMPLETE_MAX_TILE_COMPONENTS
                                                            : F16_PACKET_COMPLETE_MAX_TILE_COMPONENTS;
    if (!packetCompleteSupportsTile(geometry.outer_size, geometry.reduction_size, geometry.inner_size,
                                    warps_per_output, tile_components, max_tile) ||
        input.getTotalNumElements() != geometry.input_elements ||
        !fp8CompleteSupportsOutputStorage(output.getDataType()) || output.getDimensions() != geometry.output_dimensions) {
        throw std::logic_error(std::string(name) +
                               " RuntimeK PacketComplete received unsupported geometry or outputs.");
    }
    // Any element address (the kernel takes its prefixes from the base address).
    if ((reinterpret_cast<uintptr_t>(input.getMemPtr<void>()) & uintptr_t{sizeof(InputT) - 1}) != 0) {
        throw std::logic_error(std::string(name) + " RuntimeK PacketComplete requires an element-aligned input base.");
    }
    const uint32_t tile_count =
        static_cast<uint32_t>(ceilDivideU64(geometry.inner_size, static_cast<uint64_t>(tile_components)));
    const uint32_t outputs_per_block = SHALLOW_COMPLETE_WARPS / warps_per_output;
    const uint64_t work_count = geometry.outer_size * static_cast<uint64_t>(tile_count);
    const uint64_t block_count = std::min<uint64_t>(
        ceilDivideU64(work_count, static_cast<uint64_t>(outputs_per_block)), PERSISTENT_MAX_GRID_BLOCKS);
    const PacketCompleteRowStrides row_strides = packetCompleteRowStrides(input, geometry);
    const bool launched =
        dispatchPacketCompleteTransform<InputT, FixedOp>(semantics.input_transform, [&](auto kernel) {
            kernel<<<static_cast<unsigned int>(block_count), SHALLOW_COMPLETE_THREADS, 0, stream.getStream()>>>(
                input.getMemPtr<InputT>(),
                output.getMemPtr<void>(),
                output.getDataType(),
                static_cast<uint32_t>(geometry.outer_size),
                static_cast<uint32_t>(geometry.reduction_size),
                static_cast<uint32_t>(geometry.inner_size),
                row_strides.row,
                row_strides.outer,
                tile_components,
                tile_count,
                warps_per_output,
                output_finalize,
                ReductionStageInputTransform::Identity,
                output_scale);
            CUDA_CHECK(cudaGetLastError());
        });
    if (!launched) {
        throw std::logic_error(std::string(name) + " RuntimeK PacketComplete received an unsupported input transform.");
    }
}

template <typename InputT>
void launchPacketCompleteCombine(const StridedSpan& input,
                                 const StridedSpan& output,
                                 const ReductionGeometry& geometry,
                                 uint32_t warps_per_output,
                                 uint32_t tile_components,
                                 Stream& stream,
                                 const ReductionStageSemantics& semantics,
                                 float output_scale,
                                 const char* name) {
    switch (semantics.combine) {
        case ReductionStageCombine::Sum:
            return launchPacketCompleteTyped<InputT, 0>(input, output, geometry, warps_per_output, tile_components,
                                                        stream, semantics, output_scale, name);
        case ReductionStageCombine::Product:
            return launchPacketCompleteTyped<InputT, 1>(input, output, geometry, warps_per_output, tile_components,
                                                        stream, semantics, output_scale, name);
        case ReductionStageCombine::Minimum:
            return launchPacketCompleteTyped<InputT, 2>(input, output, geometry, warps_per_output, tile_components,
                                                        stream, semantics, output_scale, name);
        case ReductionStageCombine::Maximum:
            return launchPacketCompleteTyped<InputT, 3>(input, output, geometry, warps_per_output, tile_components,
                                                        stream, semantics, output_scale, name);
    }
    throw std::logic_error(std::string("Invalid ") + name + " RuntimeK PacketComplete combine operation.");
}

template <typename InputT, int FixedOp>
KernelOccupancyInfo queryPacketCompleteOccupancyTyped(ReductionStageInputTransform transform) {
    KernelOccupancyInfo occupancy{};
    const bool queried = dispatchPacketCompleteTransform<InputT, FixedOp>(
        transform, [&](auto kernel) { occupancy = queryShallowCompleteOccupancy(kernel); });
    if (!queried) {
        throw std::logic_error("RuntimeK PacketComplete occupancy requires a supported input transform.");
    }
    return occupancy;
}

template <typename InputT>
KernelOccupancyInfo queryPacketCompleteOccupancyCombine(ReductionStageInputTransform transform,
                                                        ReductionStageCombine combine) {
    switch (combine) {
        case ReductionStageCombine::Sum: return queryPacketCompleteOccupancyTyped<InputT, 0>(transform);
        case ReductionStageCombine::Product: return queryPacketCompleteOccupancyTyped<InputT, 1>(transform);
        case ReductionStageCombine::Minimum: return queryPacketCompleteOccupancyTyped<InputT, 2>(transform);
        case ReductionStageCombine::Maximum: return queryPacketCompleteOccupancyTyped<InputT, 3>(transform);
    }
    throw std::logic_error("Invalid RuntimeK PacketComplete occupancy combine operation.");
}

}  // namespace

void launchFp32PacketComplete(const StridedSpan& input,
                              const StridedSpan& output,
                              const ReductionGeometry& geometry,
                              uint32_t warps_per_output,
                              uint32_t tile_components,
                              Stream& stream,
                              const ReductionStageSemantics& semantics,
                              float output_scale) {
    if (input.getDataType() != DataType::FP32)
        throw std::logic_error("FP32 RuntimeK PacketComplete received unsupported geometry or outputs.");
    launchPacketCompleteCombine<float>(input, output, geometry, warps_per_output, tile_components, stream, semantics,
                                       output_scale, "FP32");
}

void launchF16PacketComplete(const StridedSpan& input,
                             const StridedSpan& output,
                             const ReductionGeometry& geometry,
                             uint32_t warps_per_output,
                             uint32_t tile_components,
                             Stream& stream,
                             const ReductionStageSemantics& semantics,
                             float output_scale) {
    switch (input.getDataType()) {
        case DataType::FP16:
            return launchPacketCompleteCombine<__half>(input, output, geometry, warps_per_output, tile_components,
                                                       stream, semantics, output_scale, "FP16");
        case DataType::BF16:
            return launchPacketCompleteCombine<__nv_bfloat16>(input, output, geometry, warps_per_output,
                                                              tile_components, stream, semantics, output_scale,
                                                              "BF16");
        default:
            throw std::logic_error("FP16 / BF16 RuntimeK PacketComplete received a non-FP16 / BF16 input.");
    }
}

KernelOccupancyInfo queryF16PacketCompleteOccupancy(DataType input_dtype,
                                                    ReductionStageInputTransform transform,
                                                    ReductionStageCombine combine) {
    switch (input_dtype) {
        case DataType::FP16: return queryPacketCompleteOccupancyCombine<__half>(transform, combine);
        case DataType::BF16: return queryPacketCompleteOccupancyCombine<__nv_bfloat16>(transform, combine);
        default: throw std::logic_error("FP16 / BF16 RuntimeK PacketComplete occupancy requires FP16 / BF16.");
    }
}

// ---- The FP32 and FP16 / BF16 RK ARG PacketComplete twins (see packetCompleteArgKernel).
bool fp32PacketCompleteArgSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size,
                                   uint32_t warps_per_output, uint32_t tile_components) {
    return fp32PacketCompleteSupports(outer_size, reduction_size, inner_size, warps_per_output, tile_components) &&
           reduction_size <= FP32_ARG_MAX_SHARD_ROWS;
}

bool f16PacketCompleteArgSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size,
                                  uint32_t warps_per_output, uint32_t tile_components) {
    return f16PacketCompleteSupports(outer_size, reduction_size, inner_size, warps_per_output, tile_components) &&
           reduction_size <= FP32_ARG_MAX_SHARD_ROWS;
}

namespace {
template <typename InputT>
void launchPacketCompleteArgTyped(const StridedSpan& input,
                                  const StridedSpan* value_output,
                                  const StridedSpan* index_output,
                                  const ReductionGeometry& geometry,
                                  uint32_t warps_per_output,
                                  uint32_t tile_components,
                                  bool maximum,
                                  uint64_t domain_stride,
                                  Stream& stream,
                                  const char* name) {
    const uint64_t outputs = geometry.outer_size * geometry.inner_size;
    const bool supported =
        std::is_same_v<InputT, float>
            ? fp32PacketCompleteArgSupports(geometry.outer_size, geometry.reduction_size, geometry.inner_size,
                                            warps_per_output, tile_components)
            : f16PacketCompleteArgSupports(geometry.outer_size, geometry.reduction_size, geometry.inner_size,
                                           warps_per_output, tile_components);
    if (!supported ||
        input.getTotalNumElements() != geometry.outer_size * geometry.reduction_size * geometry.inner_size ||
        (value_output == nullptr && index_output == nullptr) ||
        (value_output != nullptr && (value_output->getTotalNumElements() != outputs ||
                                     !fp8CompleteSupportsOutputStorage(value_output->getDataType()))) ||
        (index_output != nullptr && (index_output->getTotalNumElements() != outputs ||
                                     (index_output->getDataType() != DataType::UINT32 &&
                                      index_output->getDataType() != DataType::UINT64)))) {
        throw std::logic_error(std::string(name) + " PacketComplete ARG received unsupported geometry or outputs.");
    }
    // Any element-aligned base (the kernel's prefixes follow the address).
    if ((reinterpret_cast<uintptr_t>(input.getMemPtr<void>()) & uintptr_t{sizeof(InputT) - 1}) != 0) {
        throw std::logic_error(std::string(name) + " PacketComplete ARG requires an element-aligned input base.");
    }
    if (index_output != nullptr && index_output->getDataType() == DataType::UINT32 &&
        (geometry.reduction_size - 1) * domain_stride > std::numeric_limits<uint32_t>::max()) {
        throw std::logic_error(std::string(name) +
                               " PacketComplete ARG UINT32 index cannot hold the original-domain index.");
    }
    const Fp32ArgOutputs arg_outputs{value_output == nullptr ? nullptr : value_output->getMemPtr<void>(),
                                     value_output == nullptr ? DataType::FP32 : value_output->getDataType(),
                                     index_output == nullptr ? nullptr : index_output->getMemPtr<void>(),
                                     index_output == nullptr ? DataType::UINT32 : index_output->getDataType(),
                                     domain_stride,
                                     maximum ? 0x80000000U : 0U};
    const uint32_t tile_count =
        static_cast<uint32_t>(ceilDivideU64(geometry.inner_size, static_cast<uint64_t>(tile_components)));
    const uint64_t outputs_per_block = SHALLOW_COMPLETE_WARPS / warps_per_output;
    const uint64_t block_count =
        std::min<uint64_t>(ceilDivideU64(geometry.outer_size * tile_count, outputs_per_block),
                           PERSISTENT_MAX_GRID_BLOCKS);
    packetCompleteArgKernel<InputT><<<static_cast<unsigned int>(block_count), SHALLOW_COMPLETE_THREADS, 0,
                                          stream.getStream()>>>(
        input.getMemPtr<InputT>(), arg_outputs, static_cast<uint32_t>(geometry.outer_size),
        static_cast<uint32_t>(geometry.reduction_size), static_cast<uint32_t>(geometry.inner_size), tile_components,
        tile_count, warps_per_output);
    CUDA_CHECK(cudaGetLastError());
}
}  // namespace

void launchFp32PacketCompleteArg(const StridedSpan& input,
                                 const StridedSpan* value_output,
                                 const StridedSpan* index_output,
                                 const ReductionGeometry& geometry,
                                 uint32_t warps_per_output,
                                 uint32_t tile_components,
                                 bool maximum,
                                 uint64_t domain_stride,
                                 Stream& stream) {
    if (input.getDataType() != DataType::FP32)
        throw std::logic_error("FP32 PacketComplete ARG received unsupported geometry or outputs.");
    launchPacketCompleteArgTyped<float>(input, value_output, index_output, geometry, warps_per_output,
                                        tile_components, maximum, domain_stride, stream, "FP32");
}

void launchF16PacketCompleteArg(const StridedSpan& input,
                                const StridedSpan* value_output,
                                const StridedSpan* index_output,
                                const ReductionGeometry& geometry,
                                uint32_t warps_per_output,
                                uint32_t tile_components,
                                bool maximum,
                                uint64_t domain_stride,
                                Stream& stream) {
    switch (input.getDataType()) {
        case DataType::FP16:
            return launchPacketCompleteArgTyped<__half>(input, value_output, index_output, geometry, warps_per_output,
                                                        tile_components, maximum, domain_stride, stream, "FP16");
        case DataType::BF16:
            return launchPacketCompleteArgTyped<__nv_bfloat16>(input, value_output, index_output, geometry,
                                                               warps_per_output, tile_components, maximum,
                                                               domain_stride, stream, "BF16");
        default:
            throw std::logic_error("FP16 / BF16 PacketComplete ARG received a non-FP16 / BF16 input.");
    }
}

KernelOccupancyInfo queryFp32PacketCompleteArgOccupancy() {
    return queryShallowCompleteOccupancy(packetCompleteArgKernel<float>);
}

KernelOccupancyInfo queryF16PacketCompleteArgOccupancy(DataType input_dtype) {
    switch (input_dtype) {
        case DataType::FP16: return queryShallowCompleteOccupancy(packetCompleteArgKernel<__half>);
        case DataType::BF16: return queryShallowCompleteOccupancy(packetCompleteArgKernel<__nv_bfloat16>);
        default: throw std::logic_error("FP16 / BF16 PacketComplete ARG occupancy requires FP16 / BF16 input.");
    }
}

KernelOccupancyInfo queryFp32PacketCompleteOccupancy(ReductionStageInputTransform transform,
                                                     ReductionStageCombine combine) {
    return queryPacketCompleteOccupancyCombine<float>(transform, combine);
}

uint32_t fp8PacketCompleteArgTileComponents(uint64_t inner_size, uint32_t value_tile_components) {
    // A lane keys four adjacent components per word, so a row of a tile costs ceil(tile / 128) key words and a row of
    // the whole K costs sum over tiles. A 256-component tile reaches the minimum ceil(K / 128); VALUE's tile (capped at
    // 256) is kept when it is as cheap (it may carry the planner's parallelism choice for small outer).
    if (inner_size == 0 || value_tile_components == 0) return 0;
    const auto quads_per_row = [inner_size](uint64_t tile) {
        return (inner_size / tile) * ceilDivideU64(tile, 128) + ceilDivideU64(inner_size % tile, 128);
    };
    const uint64_t value_tile =
        std::min<uint64_t>({value_tile_components, FP8_ARG_PACKET_COMPLETE_MAX_TILE, inner_size});
    const uint64_t widest_tile = std::min<uint64_t>(FP8_ARG_PACKET_COMPLETE_MAX_TILE, inner_size);
    return static_cast<uint32_t>(quads_per_row(widest_tile) < quads_per_row(value_tile) ? widest_tile : value_tile);
}

bool fp8PacketCompleteArgSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size,
                                  uint32_t warps_per_output, uint32_t tile_components) {
    if (outer_size == 0 || reduction_size == 0 || inner_size == 0 || tile_components == 0 ||
        tile_components > FP8_ARG_PACKET_COMPLETE_MAX_TILE ||
        (warps_per_output != 1 && warps_per_output != 2 && warps_per_output != 4 && warps_per_output != 8)) {
        return false;
    }
    constexpr uint64_t U32 = std::numeric_limits<uint32_t>::max();
    return reduction_size < FP8_ARG_MAX_LANE_ROWS && outer_size <= U32 && inner_size <= U32 &&
           outer_size * ceilDivideU64(inner_size, tile_components) <= U32;
}

void launchFp8PacketCompleteArg(const StridedSpan& input,
                                const StridedSpan* value_output,
                                const StridedSpan* index_output,
                                const ReductionGeometry& geometry,
                                uint32_t warps_per_output,
                                uint32_t tile_components,
                                bool maximum,
                                uint64_t domain_stride,
                                Stream& stream) {
#if THOR_CUB_ENABLE_FP8_TYPES
    const DataType input_dtype = input.getDataType();
    const uint64_t outputs = geometry.outer_size * geometry.inner_size;
    if ((input_dtype != DataType::FP8_E4M3 && input_dtype != DataType::FP8_E5M2) ||
        !fp8PacketCompleteArgSupports(geometry.outer_size, geometry.reduction_size, geometry.inner_size,
                                      warps_per_output, tile_components) ||
        input.getTotalNumElements() != geometry.outer_size * geometry.reduction_size * geometry.inner_size ||
        (value_output == nullptr && index_output == nullptr) ||
        (value_output != nullptr && (value_output->getTotalNumElements() != outputs ||
                                     !fp8CompleteSupportsOutputStorage(value_output->getDataType()))) ||
        (index_output != nullptr && (index_output->getTotalNumElements() != outputs ||
                                     (index_output->getDataType() != DataType::UINT32 &&
                                      index_output->getDataType() != DataType::UINT64)))) {
        throw std::logic_error("FP8 PacketComplete ARG received unsupported geometry or outputs.");
    }
    // Any byte address (the kernel's prefixes follow the address).
    if (index_output != nullptr && index_output->getDataType() == DataType::UINT32 &&
        (geometry.reduction_size - 1) * domain_stride > std::numeric_limits<uint32_t>::max()) {
        throw std::logic_error("FP8 PacketComplete ARG UINT32 index cannot hold the original-domain index.");
    }
    const Fp8ArgOutputs arg_outputs{
        value_output == nullptr ? nullptr : value_output->getMemPtr<void>(),
        value_output == nullptr ? DataType::FP32 : value_output->getDataType(),
        index_output == nullptr ? nullptr : index_output->getMemPtr<void>(),
        index_output == nullptr ? DataType::UINT32 : index_output->getDataType(),
        domain_stride,
        fp8ArgFold(maximum, input_dtype == DataType::FP8_E5M2),
        input_dtype == DataType::FP8_E5M2};
    const uint32_t tile_count = static_cast<uint32_t>(ceilDivideU64(geometry.inner_size, tile_components));
    const uint32_t outputs_per_block = SHALLOW_COMPLETE_WARPS / warps_per_output;
    const uint64_t work_count = geometry.outer_size * static_cast<uint64_t>(tile_count);
    const uint64_t block_count =
        std::min<uint64_t>(ceilDivideU64(work_count, static_cast<uint64_t>(outputs_per_block)),
                           PERSISTENT_MAX_GRID_BLOCKS);
    fp8PacketCompleteArgKernel<16><<<static_cast<unsigned int>(block_count), SHALLOW_COMPLETE_THREADS, 0,
                                     stream.getStream()>>>(
        static_cast<const uint8_t*>(input.getMemPtr<void>()), arg_outputs,
        static_cast<uint32_t>(geometry.outer_size), static_cast<uint32_t>(geometry.reduction_size),
        static_cast<uint32_t>(geometry.inner_size), tile_components, tile_count, warps_per_output);
    CUDA_CHECK(cudaGetLastError());
#else
    static_cast<void>(input);
    static_cast<void>(value_output);
    static_cast<void>(index_output);
    static_cast<void>(geometry);
    static_cast<void>(warps_per_output);
    static_cast<void>(tile_components);
    static_cast<void>(maximum);
    static_cast<void>(domain_stride);
    static_cast<void>(stream);
    throw std::logic_error("FP8 PacketComplete ARG requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}

}  // namespace ThorImplementation::ReductionInternal
