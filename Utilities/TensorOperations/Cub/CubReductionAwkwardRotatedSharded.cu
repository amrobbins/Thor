#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"
#include "Utilities/TensorOperations/Cub/CubReductionOperation.cuh"
#include "Utilities/TensorOperations/Cub/Fp32ArgKey.cuh"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RuntimeKSelector.h"

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

namespace ThorImplementation::CubReductionInternal {
namespace {

// AWKWARD-ROTATED-SHARDED: production replacement for the old alignment-safe shaped-block kernel in the
// low-output/deep value-reduction regime. The old shaped-block ownership shards only K, leaving a few CTAs to serially walk
// all R. This design shards both R and K, and one cooperative warp group owns each (R shard, K tile).
//
// Awkward K shifts successive row starts relative to packet boundaries. Rows owned by one warp differ by one
// cooperative-group width in K strides. Production FP16/BF16/FP32 keeps the established 16-byte/eight-warp path; the
// currently policy-gated FP8 production-direction specialization is 16-byte/sixteen-warp. Full-space calibration also
// exposes natural p4/four-warp and p8/eight-warp PrefixReserved layouts for shallow/underfilled problems, while p16
// compares PrefixReserved (497 logical components) against Exact (512 logical components with distributed 4-byte tail
// ownership). FP8 p4/p8/p16 packets are decoded immediately in native fp8x2 pairs and accumulated in FP32 registers;
// shared memory still carries only post-R-reduction FP32 warp partials. In every layout the warp aligns its row window
// once and every revisited row remains naturally packet-aligned. Shifted per-warp partial windows
// are then realigned by logical component in shared memory before coalesced FP32 partial stores.
constexpr int AWKWARD_ROTATED_SHARDED_WARPS = 8;
constexpr int AWKWARD_ROTATED_SHARDED_THREADS =
    AWKWARD_ROTATED_SHARDED_WARPS * TILED_REDUCTION_WARP_THREADS;
constexpr int AWKWARD_ROTATED_FP8_SHARDED_WARPS = 16;
constexpr int AWKWARD_ROTATED_FP8_SHARDED_THREADS =
    AWKWARD_ROTATED_FP8_SHARDED_WARPS * TILED_REDUCTION_WARP_THREADS;
static_assert(AWKWARD_ROTATED_SHARDED_THREADS == 256);
static_assert(AWKWARD_ROTATED_FP8_SHARDED_THREADS == 512);

template <typename InputT>
inline constexpr int awkwardCooperativeWarps = sizeof(InputT) == 1 ? AWKWARD_ROTATED_FP8_SHARDED_WARPS
                                                                    : AWKWARD_ROTATED_SHARDED_WARPS;

template <typename InputT, int PacketBytes>
inline constexpr int awkwardItemsPerLaneForPacket = PacketBytes / static_cast<int>(sizeof(InputT));

template <typename InputT, int PacketBytes>
inline constexpr int awkwardPacketSpanElementsForPacket =
    TILED_REDUCTION_WARP_THREADS * awkwardItemsPerLaneForPacket<InputT, PacketBytes>;

template <typename InputT, int PacketBytes, bool ExactFp8Tile = true>
inline constexpr int awkwardLogicalTileElementsForPacket =
    sizeof(InputT) == 1 && ExactFp8Tile
        ? awkwardPacketSpanElementsForPacket<InputT, PacketBytes>
        : awkwardPacketSpanElementsForPacket<InputT, PacketBytes>
              - (awkwardItemsPerLaneForPacket<InputT, PacketBytes> - 1);

template <int PacketBytes, typename InputT>
[[nodiscard]] __device__ inline const InputT* alignPacketWindowDown(const InputT* source) {
    static_assert(PacketBytes == 4 || PacketBytes == 8 || PacketBytes == 16);
    const uintptr_t address = reinterpret_cast<uintptr_t>(source);
    return reinterpret_cast<const InputT*>(address & ~uintptr_t{PacketBytes - 1});
}

template <typename ReductionOpT>
struct AwkwardReductionIdentityFp32;

template <>
struct AwkwardReductionIdentityFp32<cuda::std::plus<float>> {
    static constexpr float value = 0.0f;
};

template <>
struct AwkwardReductionIdentityFp32<cuda::std::multiplies<float>> {
    static constexpr float value = 1.0f;
};

template <>
struct AwkwardReductionIdentityFp32<PropagatingMinimumFp32> {
    static constexpr float value = std::numeric_limits<float>::infinity();
};

template <>
struct AwkwardReductionIdentityFp32<PropagatingMaximumFp32> {
    static constexpr float value = -std::numeric_limits<float>::infinity();
};

// Occupancy contract for the rotated first stage. FP8 PrefixReserved (16 warps, 512 threads, 33,856 B static shared
// including the 1 KiB per-block reserve) fits three blocks = 48 warps per SM only at <= 40 registers/thread; at 41..64
// it drops to two. Thor1067 compiled it at 36 registers; P5-B's loop rework moved it to 43 and cost 15-23% on
// one-tile shapes, so the 40-register budget is pinned here. FP8 Exact compiles at exactly 64 registers (two blocks
// per SM; 65+ leaves one). Its MIN/MAX kernels, which carry the P5-F transform fold, are pinned to two blocks: an
// unpinned fold went to 67 registers and 1.47x slower (1084). FP8 Exact SUM/PRODUCT and the 8-warp FP16/BF16/FP32
// kernels keep their natural allocation; maxThreads is their exact launch block size.
template <typename InputT, bool ExactFp8Tile, typename ReductionOpT = cuda::std::plus<float>>
inline constexpr int awkwardRotatedMinBlocksPerSm =
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
                             (awkwardRotatedMinBlocksPerSm<InputT, ExactFp8Tile, ReductionOpT>))
void awkwardAlignmentRotatedShardedFirstStageKernel(const InputT* input,
                                                                float* fp32_partials,
                                                                uint64_t outer_size,
                                                                uint64_t reduction_size,
                                                                uint64_t inner_size,
                                                                uint64_t shards_per_output,
                                                                uint64_t component_tiles,
                                                                Fp8ExtremumFold extremum_fold) {
    // P5-F (see Fp8ExtremumFold): FP8 MIN/MAX fold the ABS/SQUARE transform into the FP8 bytes. PrefixReserved also
    // folds the MAX sign (its MIN kernel serves MAX); Exact keeps separate MIN and MAX kernels and folds only the
    // transform (no XOR operand, see awkwardRotatedMinBlocksPerSm). The stored FP32 partial is unfolded, so staged
    // partials are those of the requested operation.
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
    constexpr int ItemsPerLane = awkwardItemsPerLaneForPacket<InputT, PacketBytes>;
    constexpr int PacketSpanElements = awkwardPacketSpanElementsForPacket<InputT, PacketBytes>;
    constexpr int LogicalTileElements = awkwardLogicalTileElementsForPacket<InputT, PacketBytes, ExactFp8Tile>;
    ReductionOpT reduction_op{};
    const float Init = [&] {
        if constexpr (requires { reduction_op.identity(); }) return reduction_op.identity();
        else return AwkwardReductionIdentityFp32<ReductionOpT>::value;
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
void launchAwkwardFirstStageForTransformPacket(const Tensor& input,
                                               Tensor& fp32_partials,
                                               const CubReductionGeometry& geometry,
                                               uint64_t shards_per_output,
                                               cudaStream_t stream,
                                               Fp8ExtremumFold extremum_fold = FP8_EXTREMUM_FOLD_NONE) {
    constexpr int LogicalTileElements = awkwardLogicalTileElementsForPacket<InputT, PacketBytes, ExactFp8Tile>;
    const uint64_t component_tiles =
        ceilDivideU64(geometry.inner_size, static_cast<uint64_t>(LogicalTileElements));
    if (geometry.outer_size == 0 || component_tiles == 0 || shards_per_output == 0) {
        throw std::logic_error("Awkward rotated sharded first stage requires non-zero work.");
    }
    // Division-free launch: x = shard, y = component tile, z = outer.
    const dim3 grid_blocks = reductionWorkGrid(shards_per_output, component_tiles, geometry.outer_size);

    awkwardAlignmentRotatedShardedFirstStageKernel<InputT,
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


}  // namespace

namespace {

template <typename InputT, int RowsPerShard, int PacketBytes, int CooperativeWarps,
          bool ExactFp8Tile = true, typename InputTransformT = IdentityFp32,
          typename ReductionOpT = cuda::std::plus<float>>
[[nodiscard]] CubKernelOccupancyInfo queryRotatedRCooperativePacketOccupancyForRows() {
    const auto kernel = awkwardAlignmentRotatedShardedFirstStageKernel<InputT,
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
    return CubKernelOccupancyInfo{attributes.numRegs,
                                  attributes.sharedSizeBytes,
                                  0,
                                  active_blocks,
                                  active_blocks * CooperativeWarps};
}

// 1190: the retained RuntimeK rotated first stage on FP32 input, PrefixReserved only (Exact's tail logic is FP8 byte
// chunks and would gain 3 components). The shared kernel at p16 with 4 cooperative warps: 128 threads cover the 125
// component logical tile in the readback, and 4 rows of K floats are a whole number of packets for any K, so each warp
// aligns once. FP32 has no extremum fold (raw MIN/MAX kernels, every transform). FP8's retained row depths.
constexpr int FP32_RUNTIME_K_ROTATED_PACKET_BYTES = 16;
constexpr int FP32_RUNTIME_K_ROTATED_WARPS = 4;
static_assert(awkwardLogicalTileElementsForPacket<float, FP32_RUNTIME_K_ROTATED_PACKET_BYTES, false> ==
              static_cast<int>(FP32_RUNTIME_K_ROTATED_TILE_COMPONENTS));
static_assert(Fp8RK::Fp8RuntimeKPolicy::rotated_row_depths == std::array<uint64_t, 6>{32, 64, 128, 256, 512, 1024},
              "dispatchFp32RuntimeKRotated's cases restate the retained depths");

// Invokes callback.template operator()<InputTransformT, RowsPerShard>() for the retained FP32 rotated specialization
// matching the runtime request; returns false otherwise.
template <typename Callback>
[[nodiscard]] bool dispatchFp32RuntimeKRotated(uint64_t rows_per_shard, CubReductionStageInputTransform transform,
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
        case CubReductionStageInputTransform::Identity: for_rows.template operator()<IdentityFp32>(); break;
        case CubReductionStageInputTransform::AbsoluteValue: for_rows.template operator()<AbsoluteValueFp32>(); break;
        case CubReductionStageInputTransform::Square: for_rows.template operator()<SquareFp32>(); break;
    }
    return dispatched;
}

template <typename ReductionOpT>
void launchFp32RuntimeKRotatedFirstStageTyped(const Tensor& input,
                                              Tensor& fp32_partials,
                                              const CubReductionGeometry& geometry,
                                              uint64_t rows_per_shard,
                                              uint64_t shards_per_output,
                                              Stream& stream,
                                              CubReductionStageInputTransform transform) {
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
            launchAwkwardFirstStageForTransformPacket<float, ReductionOpT, InputTransformT, RowsPerShard,
                                                       FP32_RUNTIME_K_ROTATED_PACKET_BYTES,
                                                       FP32_RUNTIME_K_ROTATED_WARPS, false>(
                input, fp32_partials, geometry, shards_per_output, stream.getStream());
        });
    if (!launched) {
        throw std::logic_error("FP32 RuntimeK rotated first stage requires a retained compiled specialization.");
    }
}

template <typename ReductionOpT>
CubKernelOccupancyInfo queryFp32RuntimeKRotatedOccupancyTyped(uint64_t rows_per_shard,
                                                              CubReductionStageInputTransform transform) {
    CubKernelOccupancyInfo occupancy{};
    const bool queried = dispatchFp32RuntimeKRotated(
        rows_per_shard, transform, [&]<typename InputTransformT, int RowsPerShard>() {
            occupancy = queryRotatedRCooperativePacketOccupancyForRows<float, RowsPerShard,
                                                                        FP32_RUNTIME_K_ROTATED_PACKET_BYTES,
                                                                        FP32_RUNTIME_K_ROTATED_WARPS, false,
                                                                        InputTransformT, ReductionOpT>();
        });
    if (!queried) {
        throw std::logic_error("FP32 RuntimeK rotated occupancy requires a retained compiled specialization.");
    }
    return occupancy;
}

// 1245: the retained RuntimeK rotated first stage on FP16 / BF16 input (plans/Thor_FP16_BF16_Dense_Reduction_Plan,
// phase 5): the established 2-byte layout, unchanged (p16 packets of 8 components, 8 cooperative warps, whose revisit
// stride of 8 rows of K half-words is a whole number of packets for any K, and the 249-component PrefixReserved tile)
// at FP8's retained row depths. The instantiations are the legacy production pass's (same template arguments,
// ExactFp8Tile at its default, which only FP8 reads), so depths 128..1024 add no kernels; 32 and 64 are new. Every
// combine x {identity, |x|, x^2}, no fold, any 2-byte aligned input.
constexpr int F16_RUNTIME_K_ROTATED_PACKET_BYTES = 16;
constexpr int F16_RUNTIME_K_ROTATED_WARPS = AWKWARD_ROTATED_SHARDED_WARPS;
static_assert(F16_RUNTIME_K_ROTATED_WARPS == awkwardCooperativeWarps<__half>);
static_assert(awkwardLogicalTileElementsForPacket<__half, F16_RUNTIME_K_ROTATED_PACKET_BYTES> ==
              static_cast<int>(F16_RUNTIME_K_ROTATED_TILE_COMPONENTS));

template <typename InputT, typename ReductionOpT>
void launchF16RuntimeKRotatedFirstStageTyped(const Tensor& input,
                                             Tensor& fp32_partials,
                                             const CubReductionGeometry& geometry,
                                             uint64_t rows_per_shard,
                                             uint64_t shards_per_output,
                                             Stream& stream,
                                             CubReductionStageInputTransform transform) {
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
            launchAwkwardFirstStageForTransformPacket<InputT, ReductionOpT, InputTransformT, RowsPerShard,
                                                       F16_RUNTIME_K_ROTATED_PACKET_BYTES,
                                                       F16_RUNTIME_K_ROTATED_WARPS>(
                input, fp32_partials, geometry, shards_per_output, stream.getStream());
        });
    if (!launched) {
        throw std::logic_error("FP16 / BF16 RuntimeK rotated first stage requires a retained compiled specialization.");
    }
}

template <typename InputT, typename ReductionOpT>
CubKernelOccupancyInfo queryF16RuntimeKRotatedOccupancyTyped(uint64_t rows_per_shard,
                                                             CubReductionStageInputTransform transform) {
    CubKernelOccupancyInfo occupancy{};
    const bool queried = dispatchFp32RuntimeKRotated(
        rows_per_shard, transform, [&]<typename InputTransformT, int RowsPerShard>() {
            occupancy = queryRotatedRCooperativePacketOccupancyForRows<InputT, RowsPerShard,
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
[[nodiscard]] bool dispatchF16RuntimeKRotatedTypes(DataType input_dtype, CubReductionStageCombine combine,
                                                   Callback&& callback) {
    const auto for_input = [&]<typename InputT>() {
        switch (combine) {
            case CubReductionStageCombine::Sum:
                callback.template operator()<InputT, cuda::std::plus<float>>();
                return true;
            case CubReductionStageCombine::Product:
                callback.template operator()<InputT, cuda::std::multiplies<float>>();
                return true;
            case CubReductionStageCombine::Minimum:
                callback.template operator()<InputT, PropagatingMinimumFp32>();
                return true;
            case CubReductionStageCombine::Maximum:
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

// 1198-1200: the FP32 RK ARG rotated twin (plans/Thor_FP32_Dense_Reduction_Plan, phase 6): FP8's fp8RotatedArgKernel
// layout at FP32 VALUE rotated's geometry (16-byte packets of 4 components, 4 cooperative warps, the 125-component
// logical tile; each warp aligns its first row's window once and every revisited row, 4 rows later, stays aligned)
// with one FP32 ARG candidate (Fp32ArgKey.cuh) per component and a runtime shard depth. A warp's rows only increase,
// so a strictly smaller key replaces its best; the warps merge in shared memory with the (key, tail) minimum. One
// shard (depth >= R) writes the requested outputs; several write packed partials, finished by the continuation.
constexpr int FP32_ARG_ROTATED_WARPS = 4;
constexpr int FP32_ARG_ROTATED_THREADS = FP32_ARG_ROTATED_WARPS * TILED_REDUCTION_WARP_THREADS;
constexpr uint32_t FP32_ARG_ROTATED_SPAN = TILED_REDUCTION_WARP_THREADS * 4;
static_assert(FP32_ARG_ROTATED_SPAN - 3 == FP32_RUNTIME_K_ROTATED_TILE_COMPONENTS);
static_assert(FP32_ARG_ROTATED_THREADS >= static_cast<int>(FP32_RUNTIME_K_ROTATED_TILE_COMPONENTS));

// 1248 / 1249: and FP16 / BF16 input (plans/Thor_FP16_BF16_Dense_Reduction_Plan, phase 6), at the 2-byte VALUE rotated
// geometry (16-byte packets of 8 components, 8 cooperative warps, whose revisit stride of 8 rows of K half-words is a
// whole number of packets, and the 249-component tile): each component widens to FP32 and takes the FP32 key (user
// decision 2026-11-05), so the lane state, the packed partials and the continuation are FP32's.
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
__global__ __launch_bounds__(ARG_ROTATED_THREADS<InputT>) void fp32RotatedArgKernel(const InputT* input,
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

// 1210: the carried-index rotated ARG twin (phase 8; see Fp32ArgWidePartials): fp32RotatedArgKernel's layout, reading
// the carried index packet at each value packet's element offset (both bases 16-byte aligned, so the windows match).
// Composed indices are not monotone in the row, so every merge is the full (key, tail) minimum. One shard writes the
// requested outputs; several write wide partials, finished by the wide continuation.
template <bool Staged, typename IndexT>
__global__ __launch_bounds__(FP32_ARG_ROTATED_THREADS) void fp32RotatedCarriedArgKernel(const float* input,
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

// Retained FP8 RuntimeK rotated first stage, owned by Fp8RK::Fp8RuntimeKPolicy: p16 packets with
// 16 cooperative warps, exact (512) or prefix-reserved (497) K tiles, and only the retained row
// depths. Packet width is not a runtime axis and no other row depth is compiled.
constexpr int FP8_RUNTIME_K_ROTATED_PACKET_BYTES = 16;
static_assert(AWKWARD_ROTATED_FP8_SHARDED_WARPS == FP8_RUNTIME_K_ROTATED_PACKET_BYTES);

template <int RowsPerShard>
[[nodiscard]] constexpr bool fp8RuntimeKRotatedRetainedRows() {
    for (uint64_t retained : Fp8RK::Fp8RuntimeKPolicy::rotated_row_depths) {
        if (retained == static_cast<uint64_t>(RowsPerShard)) return true;
    }
    return false;
}

// Invokes callback.template operator()<InputT, InputTransformT, RowsPerShard, ExactTile>() for the
// retained compiled rotated specialization matching the runtime request; returns false otherwise.
// IdentityOnly (folded FP8 MIN/MAX, P5-F) instantiates only the identity transform.
template <bool IdentityOnly = false, typename Callback>
[[nodiscard]] bool dispatchRetainedFp8RuntimeKRotated(DataType input_dtype,
                                                      bool exact_tile,
                                                      uint64_t rows_per_shard,
                                                      CubReductionStageInputTransform transform,
                                                      Callback&& callback) {
    bool dispatched = false;
    const auto for_rows = [&]<typename InputT, typename InputTransformT, bool ExactTile>() {
        const auto rows = [&]<int RowsPerShard>() {
            if constexpr (fp8RuntimeKRotatedRetainedRows<RowsPerShard>()) {
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
            if (transform == CubReductionStageInputTransform::Identity) {
                for_tile.template operator()<InputT, IdentityFp32>();
            }
        } else {
            switch (transform) {
                case CubReductionStageInputTransform::Identity:
                    for_tile.template operator()<InputT, IdentityFp32>(); return;
                case CubReductionStageInputTransform::AbsoluteValue:
                    for_tile.template operator()<InputT, AbsoluteValueFp32>(); return;
                case CubReductionStageInputTransform::Square:
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

// P5-F: FP8 MIN and MAX launch identity-transform kernels with the transform folded into the bytes. PrefixReserved
// MAX also folds its sign and launches the MIN kernel; Exact keeps its MAX kernel (transform-only fold).
template <typename ReductionOpT>
inline constexpr bool fp8RotatedFoldedExtremum =
    std::is_same_v<ReductionOpT, PropagatingMinimumFp32> || std::is_same_v<ReductionOpT, PropagatingMaximumFp32>;

template <typename ReductionOpT, bool ExactTile>
using Fp8RotatedPhysicalReductionOp =
    std::conditional_t<fp8RotatedFoldedExtremum<ReductionOpT> && !ExactTile, PropagatingMinimumFp32, ReductionOpT>;

template <typename ReductionOpT>
[[nodiscard]] Fp8ExtremumFold fp8RotatedExtremumFold(bool exact_tile, CubReductionStageInputTransform transform) {
    if constexpr (!fp8RotatedFoldedExtremum<ReductionOpT>) {
        static_cast<void>(exact_tile);
        static_cast<void>(transform);
        return FP8_EXTREMUM_FOLD_NONE;
    } else {
        return fp8ExtremumFold(!exact_tile && std::is_same_v<ReductionOpT, PropagatingMaximumFp32>, transform);
    }
}

template <typename ReductionOpT>
void launchFp8RuntimeKRotatedFirstStageTyped(const Tensor& input,
                                             Tensor& fp32_partials,
                                             const CubReductionGeometry& geometry,
                                             bool exact_tile,
                                             uint64_t rows_per_shard,
                                             uint64_t shards_per_output,
                                             Stream& stream,
                                             CubReductionStageInputTransform transform) {
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
    const bool launched = dispatchRetainedFp8RuntimeKRotated<Fold>(
        input.getDataType(), exact_tile, rows_per_shard, Fold ? CubReductionStageInputTransform::Identity : transform,
        [&]<typename InputT, typename InputTransformT, int RowsPerShard, bool ExactTile>() {
            launchAwkwardFirstStageForTransformPacket<InputT, Fp8RotatedPhysicalReductionOp<ReductionOpT, ExactTile>,
                                                       InputTransformT, RowsPerShard,
                                                       FP8_RUNTIME_K_ROTATED_PACKET_BYTES,
                                                       AWKWARD_ROTATED_FP8_SHARDED_WARPS, ExactTile>(
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
CubKernelOccupancyInfo queryFp8RuntimeKRotatedOccupancyTyped(DataType input_dtype,
                                                             bool exact_tile,
                                                             uint64_t rows_per_shard,
                                                             CubReductionStageInputTransform transform) {
#if THOR_CUB_ENABLE_FP8_TYPES
    constexpr bool Fold = fp8RotatedFoldedExtremum<ReductionOpT>;
    static_cast<void>(fp8RotatedExtremumFold<ReductionOpT>(exact_tile, transform));  // validates the transform
    CubKernelOccupancyInfo occupancy{};
    const bool queried = dispatchRetainedFp8RuntimeKRotated<Fold>(
        input_dtype, exact_tile, rows_per_shard, Fold ? CubReductionStageInputTransform::Identity : transform,
        [&]<typename InputT, typename InputTransformT, int RowsPerShard, bool ExactTile>() {
            occupancy = queryRotatedRCooperativePacketOccupancyForRows<InputT, RowsPerShard,
                                                                        FP8_RUNTIME_K_ROTATED_PACKET_BYTES,
                                                                        AWKWARD_ROTATED_FP8_SHARDED_WARPS,
                                                                        ExactTile, InputTransformT,
                                                                        Fp8RotatedPhysicalReductionOp<ReductionOpT, ExactTile>>();
        });
    if (!queried) {
        // Not a retained compiled specialization; the selector treats this as an absent candidate.
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

void launchFp8GeneralRParallelKTileRotatedCalibrationFirstStage(
    const Tensor& input,
    Tensor& fp32_partials,
    const CubReductionGeometry& geometry,
    bool exact_tile,
    uint64_t rows_per_shard,
    uint64_t shards_per_output,
    Stream& stream,
    CubReductionStageInputTransform transform, CubReductionStageCombine combine) {
    switch (combine) {
        case CubReductionStageCombine::Sum:
            return launchFp8RuntimeKRotatedFirstStageTyped<cuda::std::plus<float>>(
                input, fp32_partials, geometry, exact_tile, rows_per_shard, shards_per_output, stream, transform);
        case CubReductionStageCombine::Product:
            return launchFp8RuntimeKRotatedFirstStageTyped<cuda::std::multiplies<float>>(
                input, fp32_partials, geometry, exact_tile, rows_per_shard, shards_per_output, stream, transform);
        case CubReductionStageCombine::Minimum:
            return launchFp8RuntimeKRotatedFirstStageTyped<PropagatingMinimumFp32>(
                input, fp32_partials, geometry, exact_tile, rows_per_shard, shards_per_output, stream, transform);
        case CubReductionStageCombine::Maximum:
            return launchFp8RuntimeKRotatedFirstStageTyped<PropagatingMaximumFp32>(
                input, fp32_partials, geometry, exact_tile, rows_per_shard, shards_per_output, stream, transform);
    }
    throw std::logic_error("Invalid RuntimeK Rotated combine operation.");
}

CubKernelOccupancyInfo queryFp8GeneralRParallelKTileRotatedCalibrationOccupancy(
    DataType input_dtype, bool exact_tile, uint64_t rows_per_shard,
    CubReductionStageInputTransform transform, CubReductionStageCombine combine) {
    switch (combine) {
        case CubReductionStageCombine::Sum:
            return queryFp8RuntimeKRotatedOccupancyTyped<cuda::std::plus<float>>(
                input_dtype, exact_tile, rows_per_shard, transform);
        case CubReductionStageCombine::Product:
            return queryFp8RuntimeKRotatedOccupancyTyped<cuda::std::multiplies<float>>(
                input_dtype, exact_tile, rows_per_shard, transform);
        case CubReductionStageCombine::Minimum:
            return queryFp8RuntimeKRotatedOccupancyTyped<PropagatingMinimumFp32>(
                input_dtype, exact_tile, rows_per_shard, transform);
        case CubReductionStageCombine::Maximum:
            return queryFp8RuntimeKRotatedOccupancyTyped<PropagatingMaximumFp32>(
                input_dtype, exact_tile, rows_per_shard, transform);
    }
    throw std::logic_error("Invalid RuntimeK Rotated occupancy combine operation.");
}

bool fp32RuntimeKRotatedSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size,
                                 uint64_t rows_per_shard) {
    const auto& depths = Fp8RK::Fp8RuntimeKPolicy::rotated_row_depths;
    if (outer_size == 0 || inner_size == 0 || rows_per_shard >= reduction_size ||
        std::find(depths.begin(), depths.end(), rows_per_shard) == depths.end()) {
        return false;
    }
    const uint64_t tiles = ceilDivideU64(inner_size, FP32_RUNTIME_K_ROTATED_TILE_COMPONENTS);
    return Fp8RK::fp8RKWorkGridFits(ceilDivideU64(reduction_size, rows_per_shard), tiles, outer_size);
}

void launchFp32RuntimeKRotatedFirstStage(const Tensor& input,
                                         Tensor& fp32_partials,
                                         const CubReductionGeometry& geometry,
                                         uint64_t rows_per_shard,
                                         uint64_t shards_per_output,
                                         Stream& stream,
                                         CubReductionStageInputTransform transform,
                                         CubReductionStageCombine combine) {
    switch (combine) {
        case CubReductionStageCombine::Sum:
            return launchFp32RuntimeKRotatedFirstStageTyped<cuda::std::plus<float>>(
                input, fp32_partials, geometry, rows_per_shard, shards_per_output, stream, transform);
        case CubReductionStageCombine::Product:
            return launchFp32RuntimeKRotatedFirstStageTyped<cuda::std::multiplies<float>>(
                input, fp32_partials, geometry, rows_per_shard, shards_per_output, stream, transform);
        case CubReductionStageCombine::Minimum:
            return launchFp32RuntimeKRotatedFirstStageTyped<PropagatingMinimumFp32>(
                input, fp32_partials, geometry, rows_per_shard, shards_per_output, stream, transform);
        case CubReductionStageCombine::Maximum:
            return launchFp32RuntimeKRotatedFirstStageTyped<PropagatingMaximumFp32>(
                input, fp32_partials, geometry, rows_per_shard, shards_per_output, stream, transform);
    }
    throw std::logic_error("Invalid FP32 RuntimeK rotated combine operation.");
}

CubKernelOccupancyInfo queryFp32RuntimeKRotatedOccupancy(uint64_t rows_per_shard,
                                                         CubReductionStageInputTransform transform,
                                                         CubReductionStageCombine combine) {
    switch (combine) {
        case CubReductionStageCombine::Sum:
            return queryFp32RuntimeKRotatedOccupancyTyped<cuda::std::plus<float>>(rows_per_shard, transform);
        case CubReductionStageCombine::Product:
            return queryFp32RuntimeKRotatedOccupancyTyped<cuda::std::multiplies<float>>(rows_per_shard, transform);
        case CubReductionStageCombine::Minimum:
            return queryFp32RuntimeKRotatedOccupancyTyped<PropagatingMinimumFp32>(rows_per_shard, transform);
        case CubReductionStageCombine::Maximum:
            return queryFp32RuntimeKRotatedOccupancyTyped<PropagatingMaximumFp32>(rows_per_shard, transform);
    }
    throw std::logic_error("Invalid FP32 RuntimeK rotated occupancy combine operation.");
}

bool f16RuntimeKRotatedSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size,
                                uint64_t rows_per_shard) {
    const auto& depths = Fp8RK::Fp8RuntimeKPolicy::rotated_row_depths;
    if (outer_size == 0 || inner_size == 0 || rows_per_shard >= reduction_size ||
        std::find(depths.begin(), depths.end(), rows_per_shard) == depths.end()) {
        return false;
    }
    const uint64_t tiles = ceilDivideU64(inner_size, F16_RUNTIME_K_ROTATED_TILE_COMPONENTS);
    return Fp8RK::fp8RKWorkGridFits(ceilDivideU64(reduction_size, rows_per_shard), tiles, outer_size);
}

void launchF16RuntimeKRotatedFirstStage(const Tensor& input,
                                        Tensor& fp32_partials,
                                        const CubReductionGeometry& geometry,
                                        uint64_t rows_per_shard,
                                        uint64_t shards_per_output,
                                        Stream& stream,
                                        CubReductionStageInputTransform transform,
                                        CubReductionStageCombine combine) {
    const bool launched = dispatchF16RuntimeKRotatedTypes(
        input.getDataType(), combine, [&]<typename InputT, typename ReductionOpT>() {
            launchF16RuntimeKRotatedFirstStageTyped<InputT, ReductionOpT>(input, fp32_partials, geometry,
                                                                          rows_per_shard, shards_per_output, stream,
                                                                          transform);
        });
    if (!launched) throw std::logic_error("FP16 / BF16 RuntimeK rotated first stage received an unsupported input.");
}

CubKernelOccupancyInfo queryF16RuntimeKRotatedOccupancy(DataType input_dtype, uint64_t rows_per_shard,
                                                        CubReductionStageInputTransform transform,
                                                        CubReductionStageCombine combine) {
    CubKernelOccupancyInfo occupancy{};
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
// One input type's rotated ARG pass (FP32, or 1249 FP16 / BF16); `name` labels the errors.
[[nodiscard]] bool runtimeKArgSupportsTile(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size,
                                           uint64_t rows_per_shard, uint64_t tile_components) {
    if (outer_size == 0 || reduction_size == 0 || inner_size == 0 || rows_per_shard == 0 ||
        std::min(rows_per_shard, reduction_size) > FP32_ARG_MAX_SHARD_ROWS) {
        return false;
    }
    const uint64_t shards = ceilDivideU64(reduction_size, rows_per_shard);
    return Fp8RK::fp8RKWorkGridFits(shards, ceilDivideU64(inner_size, tile_components), outer_size);
}

template <typename InputT>
void launchRuntimeKArgTyped(const Tensor& input,
                            Tensor* value_output,
                            Tensor* index_output,
                            const CubReductionGeometry& geometry,
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
        fp32RotatedArgKernel<InputT, false><<<grid, ARG_ROTATED_THREADS<InputT>, 0, stream.getStream()>>>(
            input.getMemPtr<InputT>(), arg_outputs, nullptr, geometry.outer_size, geometry.reduction_size,
            geometry.inner_size, shards, tiles, depth);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    fp32RotatedArgKernel<InputT, true><<<grid, ARG_ROTATED_THREADS<InputT>, 0, stream.getStream()>>>(
        input.getMemPtr<InputT>(), arg_outputs, packed_scratch, geometry.outer_size, geometry.reduction_size,
        geometry.inner_size, shards, tiles, depth);
    CUDA_CHECK(cudaGetLastError());
    launchFp32ArgPackedContinuation(packed_scratch, value_output, index_output, geometry.outer_size,
                                    geometry.reduction_size, geometry.inner_size, shards, rows_per_shard, maximum,
                                    domain_stride, stream);
}

template <typename InputT>
[[nodiscard]] CubKernelOccupancyInfo queryRuntimeKArgOccupancyTyped(bool staged) {
    const auto query = [&](auto kernel) {
        cudaFuncAttributes attributes{};
        CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
        int active_blocks = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks, kernel, ARG_ROTATED_THREADS<InputT>,
                                                                 0));
        return CubKernelOccupancyInfo{attributes.numRegs, attributes.sharedSizeBytes, 0, active_blocks,
                                      active_blocks * ARG_ROTATED_WARPS<InputT>};
    };
    return staged ? query(fp32RotatedArgKernel<InputT, true>) : query(fp32RotatedArgKernel<InputT, false>);
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

void launchFp32RuntimeKArg(const Tensor& input,
                           Tensor* value_output,
                           Tensor* index_output,
                           const CubReductionGeometry& geometry,
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

void launchF16RuntimeKArg(const Tensor& input,
                          Tensor* value_output,
                          Tensor* index_output,
                          const CubReductionGeometry& geometry,
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

CubKernelOccupancyInfo queryF16RuntimeKArgOccupancy(DataType input_dtype, bool staged) {
    switch (input_dtype) {
        case DataType::FP16: return queryRuntimeKArgOccupancyTyped<__half>(staged);
        case DataType::BF16: return queryRuntimeKArgOccupancyTyped<__nv_bfloat16>(staged);
        default: throw std::logic_error("FP16 / BF16 RuntimeK ARG occupancy requires FP16 / BF16 input.");
    }
}

// ---- 1210: the carried-index rotated ARG twin (see fp32RotatedCarriedArgKernel).
bool fp32RuntimeKCarriedArgSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size,
                                    uint64_t rows_per_shard) {
    if (outer_size == 0 || reduction_size == 0 || inner_size == 0 || rows_per_shard == 0 ||
        std::min(rows_per_shard, reduction_size) > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    const uint64_t shards = ceilDivideU64(reduction_size, rows_per_shard);
    return Fp8RK::fp8RKWorkGridFits(shards, ceilDivideU64(inner_size, FP32_RUNTIME_K_ROTATED_TILE_COMPONENTS),
                                    outer_size);
}

void launchFp32RuntimeKCarriedArg(const Tensor& input,
                                  const Tensor& carried_index,
                                  Tensor* value_output,
                                  Tensor* index_output,
                                  const CubReductionGeometry& geometry,
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
        fp32RotatedCarriedArgKernel<Staged, IndexT><<<grid, FP32_ARG_ROTATED_THREADS, 0, stream.getStream()>>>(
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

CubKernelOccupancyInfo queryFp32RuntimeKCarriedArgOccupancy(bool staged, DataType index_dtype) {
    if (index_dtype != DataType::UINT32 && index_dtype != DataType::UINT64)
        throw std::logic_error("FP32 RuntimeK carried ARG occupancy requires a UINT32 or UINT64 carried index.");
    const auto query = [&](auto kernel) {
        cudaFuncAttributes attributes{};
        CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
        int active_blocks = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks, kernel, FP32_ARG_ROTATED_THREADS, 0));
        return CubKernelOccupancyInfo{attributes.numRegs, attributes.sharedSizeBytes, 0, active_blocks,
                                      active_blocks * FP32_ARG_ROTATED_WARPS};
    };
    const bool wide = index_dtype == DataType::UINT64;
    if (staged) return wide ? query(fp32RotatedCarriedArgKernel<true, uint64_t>)
                            : query(fp32RotatedCarriedArgKernel<true, uint32_t>);
    return wide ? query(fp32RotatedCarriedArgKernel<false, uint64_t>)
                : query(fp32RotatedCarriedArgKernel<false, uint32_t>);
}

CubKernelOccupancyInfo queryFp32RuntimeKArgOccupancy(bool staged) {
    return queryRuntimeKArgOccupancyTyped<float>(staged);
}


}  // namespace ThorImplementation::CubReductionInternal
