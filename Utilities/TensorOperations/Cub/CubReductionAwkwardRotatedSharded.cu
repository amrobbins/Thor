#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"
#include "Utilities/TensorOperations/Cub/CubReductionOperation.cuh"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RuntimeKSelector.h"

#include "Utilities/Exceptions.h"

#include <cuda/std/bit>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
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
// one-tile shapes, so the 40-register budget is pinned here. FP8 Exact (46+ registers, distributed tail state) and
// the 8-warp FP16/BF16/FP32 kernels keep their natural allocation; maxThreads is their exact launch block size.
template <typename InputT, bool ExactFp8Tile>
inline constexpr int awkwardRotatedMinBlocksPerSm = sizeof(InputT) == 1 && !ExactFp8Tile ? 3 : 1;

template <typename InputT,
          typename ReductionOpT,
          typename InputTransformT,
          int RowsPerShard,
          int PacketBytes,
          int CooperativeWarps,
          bool ExactFp8Tile = true>
__global__ __launch_bounds__(CooperativeWarps * TILED_REDUCTION_WARP_THREADS,
                             (awkwardRotatedMinBlocksPerSm<InputT, ExactFp8Tile>))
void awkwardAlignmentRotatedShardedFirstStageKernel(const InputT* input,
                                                                float* fp32_partials,
                                                                uint64_t outer_size,
                                                                uint64_t reduction_size,
                                                                uint64_t inner_size,
                                                                uint64_t shards_per_output,
                                                                uint64_t component_tiles) {
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
                            const PackedInputValues<InputT, ItemsPerLane> values =
                                loadVectorizedInputPacket<InputT, ItemsPerLane>(lane_packet);
                            reducePackedFp8ValuesToFp32<InputT>(local, values, reduction_op, input_transform);
                            lane_packet += row_stride_elements;
                        }
                    } else {
                        const InputT* extra_tail_packet = aligned_row_window + PacketSpanElements;
                        for (uint32_t row = first_row; row < shard_rows; row += WarpRows) {
                            const PackedInputValues<InputT, ItemsPerLane> values =
                                loadVectorizedInputPacket<InputT, ItemsPerLane>(lane_packet);
                            reducePackedFp8ValuesToFp32<InputT>(local, values, reduction_op, input_transform);

                            if (lane < TailOwnerLanes) {
                                const PackedInputValues<InputT, TailItemsPerOwnerLane> tail_values =
                                    loadVectorizedInputPacket<InputT, TailItemsPerOwnerLane>(
                                        extra_tail_packet + lane * TailItemsPerOwnerLane);
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
                        const PackedInputValues<InputT, ItemsPerLane> values =
                            loadVectorizedInputPacket<InputT, ItemsPerLane>(lane_packet);
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
                reduced;
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
                                               cudaStream_t stream) {
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
                                                                      component_tiles);
    CUDA_CHECK(cudaGetLastError());
}


template <typename InputT, typename ReductionOpT, typename InputTransformT, int RowsPerShard>
void launchAwkwardFirstStageForTransform(const Tensor& input,
                                         Tensor& fp32_partials,
                                         const CubReductionGeometry& geometry,
                                         uint64_t shards_per_output,
                                         cudaStream_t stream) {
    launchAwkwardFirstStageForTransformPacket<InputT,
                                               ReductionOpT,
                                               InputTransformT,
                                               RowsPerShard,
                                               16,
                                               awkwardCooperativeWarps<InputT>>(
        input, fp32_partials, geometry, shards_per_output, stream);
}

template <typename ReductionOpT, typename InputTransformT, int RowsPerShard>
void dispatchAwkwardFirstStageInput(const Tensor& input,
                                    Tensor& fp32_partials,
                                    const CubReductionGeometry& geometry,
                                    uint64_t shards_per_output,
                                    cudaStream_t stream) {
    switch (input.getDataType()) {
        case DataType::FP16:
            launchAwkwardFirstStageForTransform<__half, ReductionOpT, InputTransformT, RowsPerShard>(
                input, fp32_partials, geometry, shards_per_output, stream);
            return;
        case DataType::BF16:
            launchAwkwardFirstStageForTransform<__nv_bfloat16, ReductionOpT, InputTransformT, RowsPerShard>(
                input, fp32_partials, geometry, shards_per_output, stream);
            return;
        case DataType::FP32:
            launchAwkwardFirstStageForTransform<float, ReductionOpT, InputTransformT, RowsPerShard>(
                input, fp32_partials, geometry, shards_per_output, stream);
            return;
        default:
            // Original FP8 input uses the retained RuntimeK rotated first stage, never this generic RK family.
            throw std::logic_error("Awkward rotated sharded first stage received an unsupported input dtype.");
    }
}

template <typename ReductionOpT, typename InputTransformT>
void dispatchAwkwardFirstStageRows(const Tensor& input,
                                   Tensor& fp32_partials,
                                   const CubReductionGeometry& geometry,
                                   uint64_t rows_per_shard,
                                   uint64_t shards_per_output,
                                   cudaStream_t stream) {
    switch (rows_per_shard) {
        case 128:
            dispatchAwkwardFirstStageInput<ReductionOpT, InputTransformT, 128>(
                input, fp32_partials, geometry, shards_per_output, stream);
            return;
        case 256:
            dispatchAwkwardFirstStageInput<ReductionOpT, InputTransformT, 256>(
                input, fp32_partials, geometry, shards_per_output, stream);
            return;
        case 512:
            dispatchAwkwardFirstStageInput<ReductionOpT, InputTransformT, 512>(
                input, fp32_partials, geometry, shards_per_output, stream);
            return;
        case 1024:
            dispatchAwkwardFirstStageInput<ReductionOpT, InputTransformT, 1024>(
                input, fp32_partials, geometry, shards_per_output, stream);
            return;
        default:
            throw std::logic_error("Unsupported awkward rotated rows_per_shard production specialization.");
    }
}

}  // namespace

namespace {

template <typename ReductionOpT>
void launchAwkwardAlignmentRotatedFirstStageForCombine(const CubReductionStageSemantics& semantics,
                                                        const Tensor& input,
                                                        Tensor& fp32_partials,
                                                        const CubReductionGeometry& geometry,
                                                        uint64_t rows_per_shard,
                                                        uint64_t shards_per_output,
                                                        Stream& stream) {
    switch (semantics.input_transform) {
        case CubReductionStageInputTransform::Identity:
            dispatchAwkwardFirstStageRows<ReductionOpT, IdentityFp32>(input,
                                                                      fp32_partials,
                                                                      geometry,
                                                                      rows_per_shard,
                                                                      shards_per_output,
                                                                      stream.getStream());
            return;
        case CubReductionStageInputTransform::AbsoluteValue:
            dispatchAwkwardFirstStageRows<ReductionOpT, AbsoluteValueFp32>(input,
                                                                           fp32_partials,
                                                                           geometry,
                                                                           rows_per_shard,
                                                                           shards_per_output,
                                                                           stream.getStream());
            return;
        case CubReductionStageInputTransform::Square:
            dispatchAwkwardFirstStageRows<ReductionOpT, SquareFp32>(input,
                                                                    fp32_partials,
                                                                    geometry,
                                                                    rows_per_shard,
                                                                    shards_per_output,
                                                                    stream.getStream());
            return;
    }
    throw std::logic_error("Unsupported awkward rotated sharded input transform.");
}

void validateAwkwardAlignmentRotatedShardedFirstStage(const Tensor& fp32_partials,
                                                       const CubReductionGeometry& geometry,
                                                       uint64_t rows_per_shard,
                                                       uint64_t shards_per_output) {
    if ((rows_per_shard != 128 && rows_per_shard != 256 && rows_per_shard != 512 && rows_per_shard != 1024)
        || shards_per_output <= 1) {
        throw std::logic_error(
            "Awkward rotated sharded first stage requires rows_per_shard in {128,256,512,1024} and >1 shard.");
    }
    if (fp32_partials.getDataType() != DataType::FP32) {
        throw std::logic_error("Awkward rotated sharded first stage requires FP32 partial storage.");
    }
    const std::vector<uint64_t> expected_partial_dimensions = {
        geometry.outer_size, shards_per_output, geometry.inner_size};
    if (fp32_partials.getDimensions() != expected_partial_dimensions) {
        throw std::logic_error("Awkward rotated sharded FP32 partial shape does not match stamped geometry.");
    }
}

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

template <typename InputT, int RowsPerShard>
[[nodiscard]] CubKernelOccupancyInfo queryRotatedRCooperativeOccupancyForRows() {
    return queryRotatedRCooperativePacketOccupancyForRows<InputT,
                                                           RowsPerShard,
                                                           16,
                                                           awkwardCooperativeWarps<InputT>>();
}

template <typename InputT>
[[nodiscard]] CubKernelOccupancyInfo queryRotatedRCooperativeOccupancy(uint64_t rows_per_shard) {
    switch (rows_per_shard) {
        case 128:
            return queryRotatedRCooperativeOccupancyForRows<InputT, 128>();
        case 256:
            return queryRotatedRCooperativeOccupancyForRows<InputT, 256>();
        case 512:
            return queryRotatedRCooperativeOccupancyForRows<InputT, 512>();
        case 1024:
            return queryRotatedRCooperativeOccupancyForRows<InputT, 1024>();
        default:
            throw std::logic_error("Rotated R-cooperative occupancy query received unsupported rows-per-shard.");
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
template <typename Callback>
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
        switch (transform) {
            case CubReductionStageInputTransform::Identity:
                for_tile.template operator()<InputT, IdentityFp32>(); return;
            case CubReductionStageInputTransform::AbsoluteValue:
                for_tile.template operator()<InputT, AbsoluteValueFp32>(); return;
            case CubReductionStageInputTransform::Square:
                for_tile.template operator()<InputT, SquareFp32>(); return;
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
    const bool launched = dispatchRetainedFp8RuntimeKRotated(
        input.getDataType(), exact_tile, rows_per_shard, transform,
        [&]<typename InputT, typename InputTransformT, int RowsPerShard, bool ExactTile>() {
            launchAwkwardFirstStageForTransformPacket<InputT, ReductionOpT, InputTransformT, RowsPerShard,
                                                       FP8_RUNTIME_K_ROTATED_PACKET_BYTES,
                                                       AWKWARD_ROTATED_FP8_SHARDED_WARPS, ExactTile>(
                input, fp32_partials, geometry, shards_per_output, stream.getStream());
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
    CubKernelOccupancyInfo occupancy{};
    const bool queried = dispatchRetainedFp8RuntimeKRotated(
        input_dtype, exact_tile, rows_per_shard, transform,
        [&]<typename InputT, typename InputTransformT, int RowsPerShard, bool ExactTile>() {
            occupancy = queryRotatedRCooperativePacketOccupancyForRows<InputT, RowsPerShard,
                                                                        FP8_RUNTIME_K_ROTATED_PACKET_BYTES,
                                                                        AWKWARD_ROTATED_FP8_SHARDED_WARPS,
                                                                        ExactTile, InputTransformT, ReductionOpT>();
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

CubKernelOccupancyInfo queryAwkwardAlignmentRotatedShardedFirstStageOccupancy(
    DataType input_dtype, uint64_t rows_per_shard) {
    switch (input_dtype) {
        case DataType::FP16:
            return queryRotatedRCooperativeOccupancy<__half>(rows_per_shard);
        case DataType::BF16:
            return queryRotatedRCooperativeOccupancy<__nv_bfloat16>(rows_per_shard);
        case DataType::FP32:
            return queryRotatedRCooperativeOccupancy<float>(rows_per_shard);
        default:
            throw std::logic_error("Rotated R-cooperative occupancy query received unsupported input dtype.");
    }
}

void launchAwkwardAlignmentRotatedShardedFirstStage(const CubReductionStageSemantics& semantics,
                                                     const Tensor& input,
                                                     Tensor& fp32_partials,
                                                     const CubReductionGeometry& geometry,
                                                     uint64_t rows_per_shard,
                                                     uint64_t shards_per_output,
                                                     Stream& stream) {
    validateAwkwardAlignmentRotatedShardedFirstStage(
        fp32_partials, geometry, rows_per_shard, shards_per_output);

    switch (semantics.combine) {
        case CubReductionStageCombine::Sum:
            launchAwkwardAlignmentRotatedFirstStageForCombine<cuda::std::plus<float>>(semantics,
                                                                                      input,
                                                                                      fp32_partials,
                                                                                      geometry,
                                                                                      rows_per_shard,
                                                                                      shards_per_output,
                                                                                      stream);
            return;
        case CubReductionStageCombine::Product:
            launchAwkwardAlignmentRotatedFirstStageForCombine<cuda::std::multiplies<float>>(semantics,
                                                                                            input,
                                                                                            fp32_partials,
                                                                                            geometry,
                                                                                            rows_per_shard,
                                                                                            shards_per_output,
                                                                                            stream);
            return;
        case CubReductionStageCombine::Minimum:
            launchAwkwardAlignmentRotatedFirstStageForCombine<PropagatingMinimumFp32>(semantics,
                                                                                      input,
                                                                                      fp32_partials,
                                                                                      geometry,
                                                                                      rows_per_shard,
                                                                                      shards_per_output,
                                                                                      stream);
            return;
        case CubReductionStageCombine::Maximum:
            launchAwkwardAlignmentRotatedFirstStageForCombine<PropagatingMaximumFp32>(semantics,
                                                                                      input,
                                                                                      fp32_partials,
                                                                                      geometry,
                                                                                      rows_per_shard,
                                                                                      shards_per_output,
                                                                                      stream);
            return;
    }
    throw std::logic_error("Unsupported awkward rotated sharded reduction combine.");
}


}  // namespace ThorImplementation::CubReductionInternal
