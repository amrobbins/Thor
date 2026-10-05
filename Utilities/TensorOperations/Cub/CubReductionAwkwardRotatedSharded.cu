#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"
#include "Utilities/TensorOperations/Cub/CubReductionOperation.cuh"

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
constexpr uint64_t AWKWARD_ROTATED_MAX_GRID_BLOCKS = 65535;
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

template <typename InputT,
          typename ReductionOpT,
          typename InputTransformT,
          int RowsPerShard,
          int PacketBytes,
          int CooperativeWarps,
          bool ExactFp8Tile = true>
__global__ void awkwardAlignmentRotatedShardedFirstStageKernel(const InputT* input,
                                                                float* fp32_partials,
                                                                uint64_t outer_size,
                                                                uint64_t reduction_size,
                                                                uint64_t inner_size,
                                                                uint64_t shards_per_output) {
    static_assert(RowsPerShard == 8 || RowsPerShard == 16 || RowsPerShard == 32 || RowsPerShard == 64
                  || RowsPerShard == 128 || RowsPerShard == 256 || RowsPerShard == 512 || RowsPerShard == 1024
                  || RowsPerShard == 2048 || RowsPerShard == 4096 || RowsPerShard == 8192
                  || RowsPerShard == 16384);
    constexpr int ItemsPerLane = awkwardItemsPerLaneForPacket<InputT, PacketBytes>;
    constexpr int PacketSpanElements = awkwardPacketSpanElementsForPacket<InputT, PacketBytes>;
    constexpr int LogicalTileElements = awkwardLogicalTileElementsForPacket<InputT, PacketBytes, ExactFp8Tile>;
    constexpr float Init = AwkwardReductionIdentityFp32<ReductionOpT>::value;
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

    const ReductionOpT reduction_op{};
    const InputTransformT input_transform{};
    const int warp = static_cast<int>(threadIdx.x) / TILED_REDUCTION_WARP_THREADS;
    const int lane = static_cast<int>(threadIdx.x) % TILED_REDUCTION_WARP_THREADS;
    const uint64_t component_tiles = ceilDivideU64(inner_size, static_cast<uint64_t>(LogicalTileElements));
    const uint64_t total_work = outer_size * component_tiles * shards_per_output;
    const uint64_t grid_stride = static_cast<uint64_t>(gridDim.x);

    for (uint64_t work_index = static_cast<uint64_t>(blockIdx.x); work_index < total_work; work_index += grid_stride) {
        const uint64_t shard = work_index % shards_per_output;
        const uint64_t outer_component_work = work_index / shards_per_output;
        const uint64_t component_tile = outer_component_work % component_tiles;
        const uint64_t outer_index = outer_component_work / component_tiles;
        const uint64_t tile_begin = component_tile * static_cast<uint64_t>(LogicalTileElements);
        const uint64_t valid_components =
            minU64(static_cast<uint64_t>(LogicalTileElements), inner_size - tile_begin);
        const uint64_t row_begin = shard * static_cast<uint64_t>(RowsPerShard);
        const uint64_t row_end = minU64(row_begin + static_cast<uint64_t>(RowsPerShard), reduction_size);
        const uint64_t first_row = row_begin + static_cast<uint64_t>(warp);

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
        if (first_row < row_end) {
            const InputT* logical_row_tile =
                input + (outer_index * reduction_size + first_row) * inner_size + tile_begin;
            const uintptr_t logical_address = reinterpret_cast<uintptr_t>(logical_row_tile);
            prefix_elements =
                static_cast<uint32_t>((logical_address & uintptr_t{PacketBytes - 1}) / sizeof(InputT));
            const InputT* aligned_row_window = alignPacketWindowDown<PacketBytes>(logical_row_tile);

            constexpr uint64_t WarpRows = static_cast<uint64_t>(CooperativeWarps);
            const uint64_t row_stride_elements = WarpRows * inner_size;
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
                        for (uint64_t row = first_row; row < row_end; row += WarpRows) {
                            const PackedInputValues<InputT, ItemsPerLane> values =
                                loadVectorizedInputPacket<InputT, ItemsPerLane>(lane_packet);
                            reducePackedFp8ValuesToFp32<InputT>(local, values, reduction_op, input_transform);
                            lane_packet += row_stride_elements;
                        }
                    } else {
                        const InputT* extra_tail_packet = aligned_row_window + PacketSpanElements;
                        for (uint64_t row = first_row; row < row_end; row += WarpRows) {
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
                    for (uint64_t row = first_row; row < row_end; row += WarpRows) {
                        const PackedInputValues<InputT, ItemsPerLane> values =
                            loadVectorizedInputPacket<InputT, ItemsPerLane>(lane_packet);
                        reducePackedFp8ValuesToFp32<InputT>(local, values, reduction_op, input_transform);
                        lane_packet += row_stride_elements;
                    }
                } else {
                    // Preserve the established FP16/BF16/FP32 rotated kernel exactly: one aligned packet per lane and
                    // the prefix-reserved logical tile width.
                    for (uint64_t row = first_row; row < row_end; row += WarpRows) {
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
    if (geometry.outer_size > std::numeric_limits<uint64_t>::max() / component_tiles
        || geometry.outer_size * component_tiles > std::numeric_limits<uint64_t>::max() / shards_per_output) {
        throw std::overflow_error("Awkward rotated sharded first-stage work count overflows uint64_t.");
    }
    const uint64_t total_work = geometry.outer_size * component_tiles * shards_per_output;
    const unsigned int grid_blocks = static_cast<unsigned int>(
        std::min<uint64_t>(total_work, AWKWARD_ROTATED_MAX_GRID_BLOCKS));
    if (grid_blocks == 0) {
        throw std::logic_error("Awkward rotated sharded first stage requires non-zero work.");
    }

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
                                                                      shards_per_output);
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
#if THOR_CUB_ENABLE_FP8_TYPES
        case DataType::FP8_E4M3:
            launchAwkwardFirstStageForTransform<__nv_fp8_e4m3, ReductionOpT, InputTransformT, RowsPerShard>(
                input, fp32_partials, geometry, shards_per_output, stream);
            return;
        case DataType::FP8_E5M2:
            launchAwkwardFirstStageForTransform<__nv_fp8_e5m2, ReductionOpT, InputTransformT, RowsPerShard>(
                input, fp32_partials, geometry, shards_per_output, stream);
            return;
#endif
        default:
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

template <typename InputT, int RowsPerShard, int PacketBytes, int CooperativeWarps, bool ExactFp8Tile = true>
[[nodiscard]] CubKernelOccupancyInfo queryRotatedRCooperativePacketOccupancyForRows() {
    const auto kernel = awkwardAlignmentRotatedShardedFirstStageKernel<InputT,
                                                                        cuda::std::plus<float>,
                                                                        IdentityFp32,
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

template <typename InputT, int PacketBytes, int CooperativeWarps, bool ExactFp8Tile>
void launchFp8RotatedCalibrationRows(const Tensor& input,
                                     Tensor& fp32_partials,
                                     const CubReductionGeometry& geometry,
                                     uint64_t rows_per_shard,
                                     uint64_t shards_per_output,
                                     cudaStream_t stream) {
    static_assert(sizeof(InputT) == 1);
    static_assert(PacketBytes == 4 || PacketBytes == 8 || PacketBytes == 16);
    static_assert(CooperativeWarps == PacketBytes);  // FP8 natural p4/4w, p8/8w, p16/16w calibration geometry.
    switch (rows_per_shard) {
        case 8:
            launchAwkwardFirstStageForTransformPacket<InputT, cuda::std::plus<float>, IdentityFp32,
                                                       8, PacketBytes, CooperativeWarps, ExactFp8Tile>(
                input, fp32_partials, geometry, shards_per_output, stream);
            return;
        case 16:
            launchAwkwardFirstStageForTransformPacket<InputT, cuda::std::plus<float>, IdentityFp32,
                                                       16, PacketBytes, CooperativeWarps, ExactFp8Tile>(
                input, fp32_partials, geometry, shards_per_output, stream);
            return;
        case 32:
            launchAwkwardFirstStageForTransformPacket<InputT, cuda::std::plus<float>, IdentityFp32,
                                                       32, PacketBytes, CooperativeWarps, ExactFp8Tile>(
                input, fp32_partials, geometry, shards_per_output, stream);
            return;
        case 64:
            launchAwkwardFirstStageForTransformPacket<InputT, cuda::std::plus<float>, IdentityFp32,
                                                       64, PacketBytes, CooperativeWarps, ExactFp8Tile>(
                input, fp32_partials, geometry, shards_per_output, stream);
            return;
        case 128:
            launchAwkwardFirstStageForTransformPacket<InputT, cuda::std::plus<float>, IdentityFp32,
                                                       128, PacketBytes, CooperativeWarps, ExactFp8Tile>(
                input, fp32_partials, geometry, shards_per_output, stream);
            return;
        case 256:
            launchAwkwardFirstStageForTransformPacket<InputT, cuda::std::plus<float>, IdentityFp32,
                                                       256, PacketBytes, CooperativeWarps, ExactFp8Tile>(
                input, fp32_partials, geometry, shards_per_output, stream);
            return;
        case 512:
            launchAwkwardFirstStageForTransformPacket<InputT, cuda::std::plus<float>, IdentityFp32,
                                                       512, PacketBytes, CooperativeWarps, ExactFp8Tile>(
                input, fp32_partials, geometry, shards_per_output, stream);
            return;
        case 1024:
            launchAwkwardFirstStageForTransformPacket<InputT, cuda::std::plus<float>, IdentityFp32,
                                                       1024, PacketBytes, CooperativeWarps, ExactFp8Tile>(
                input, fp32_partials, geometry, shards_per_output, stream);
            return;
        case 2048:
            launchAwkwardFirstStageForTransformPacket<InputT, cuda::std::plus<float>, IdentityFp32,
                                                       2048, PacketBytes, CooperativeWarps, ExactFp8Tile>(
                input, fp32_partials, geometry, shards_per_output, stream);
            return;
        case 4096:
            launchAwkwardFirstStageForTransformPacket<InputT, cuda::std::plus<float>, IdentityFp32,
                                                       4096, PacketBytes, CooperativeWarps, ExactFp8Tile>(
                input, fp32_partials, geometry, shards_per_output, stream);
            return;
        case 8192:
            launchAwkwardFirstStageForTransformPacket<InputT, cuda::std::plus<float>, IdentityFp32,
                                                       8192, PacketBytes, CooperativeWarps, ExactFp8Tile>(
                input, fp32_partials, geometry, shards_per_output, stream);
            return;
        case 16384:
            launchAwkwardFirstStageForTransformPacket<InputT, cuda::std::plus<float>, IdentityFp32,
                                                       16384, PacketBytes, CooperativeWarps, ExactFp8Tile>(
                input, fp32_partials, geometry, shards_per_output, stream);
            return;
        default:
            throw std::logic_error("FP8 rotated calibration received unsupported rows-per-shard.");
    }
}

template <typename InputT, int PacketBytes, int CooperativeWarps, bool ExactFp8Tile>
[[nodiscard]] CubKernelOccupancyInfo queryFp8RotatedCalibrationOccupancyForInput(uint64_t rows_per_shard) {
    static_assert(sizeof(InputT) == 1);
    static_assert(PacketBytes == 4 || PacketBytes == 8 || PacketBytes == 16);
    static_assert(CooperativeWarps == PacketBytes);
    switch (rows_per_shard) {
        case 8: return queryRotatedRCooperativePacketOccupancyForRows<InputT, 8, PacketBytes, CooperativeWarps, ExactFp8Tile>();
        case 16: return queryRotatedRCooperativePacketOccupancyForRows<InputT, 16, PacketBytes, CooperativeWarps, ExactFp8Tile>();
        case 32: return queryRotatedRCooperativePacketOccupancyForRows<InputT, 32, PacketBytes, CooperativeWarps, ExactFp8Tile>();
        case 64: return queryRotatedRCooperativePacketOccupancyForRows<InputT, 64, PacketBytes, CooperativeWarps, ExactFp8Tile>();
        case 128: return queryRotatedRCooperativePacketOccupancyForRows<InputT, 128, PacketBytes, CooperativeWarps, ExactFp8Tile>();
        case 256: return queryRotatedRCooperativePacketOccupancyForRows<InputT, 256, PacketBytes, CooperativeWarps, ExactFp8Tile>();
        case 512: return queryRotatedRCooperativePacketOccupancyForRows<InputT, 512, PacketBytes, CooperativeWarps, ExactFp8Tile>();
        case 1024: return queryRotatedRCooperativePacketOccupancyForRows<InputT, 1024, PacketBytes, CooperativeWarps, ExactFp8Tile>();
        case 2048: return queryRotatedRCooperativePacketOccupancyForRows<InputT, 2048, PacketBytes, CooperativeWarps, ExactFp8Tile>();
        case 4096: return queryRotatedRCooperativePacketOccupancyForRows<InputT, 4096, PacketBytes, CooperativeWarps, ExactFp8Tile>();
        case 8192: return queryRotatedRCooperativePacketOccupancyForRows<InputT, 8192, PacketBytes, CooperativeWarps, ExactFp8Tile>();
        case 16384: return queryRotatedRCooperativePacketOccupancyForRows<InputT, 16384, PacketBytes, CooperativeWarps, ExactFp8Tile>();
        default:
            throw std::logic_error("FP8 rotated calibration occupancy received unsupported rows-per-shard.");
    }
}

}  // namespace

void launchFp8AwkwardAlignmentRotatedShardedCalibrationFirstStage(
    const Tensor& input,
    Tensor& fp32_partials,
    const CubReductionGeometry& geometry,
    size_t packet_bytes,
    bool exact_tile,
    uint64_t rows_per_shard,
    uint64_t shards_per_output,
    Stream& stream) {
#if THOR_CUB_ENABLE_FP8_TYPES
    if ((packet_bytes != 4 && packet_bytes != 8 && packet_bytes != 16)
        || (exact_tile && packet_bytes != 16)
        || shards_per_output <= 1
        || shards_per_output != ceilDivideU64(geometry.reduction_size, rows_per_shard)
        || fp32_partials.getDataType() != DataType::FP32
        || fp32_partials.getDimensions()
               != std::vector<uint64_t>{geometry.outer_size, shards_per_output, geometry.inner_size}) {
        throw std::logic_error("FP8 rotated calibration received invalid staged geometry.");
    }
    const auto dispatch = [&]<typename InputT>() {
        if (packet_bytes == 4) {
            launchFp8RotatedCalibrationRows<InputT, 4, 4, false>(
                input, fp32_partials, geometry, rows_per_shard, shards_per_output, stream.getStream());
        } else if (packet_bytes == 8) {
            launchFp8RotatedCalibrationRows<InputT, 8, 8, false>(
                input, fp32_partials, geometry, rows_per_shard, shards_per_output, stream.getStream());
        } else if (exact_tile) {
            launchFp8RotatedCalibrationRows<InputT, 16, 16, true>(
                input, fp32_partials, geometry, rows_per_shard, shards_per_output, stream.getStream());
        } else {
            launchFp8RotatedCalibrationRows<InputT, 16, 16, false>(
                input, fp32_partials, geometry, rows_per_shard, shards_per_output, stream.getStream());
        }
    };
    switch (input.getDataType()) {
        case DataType::FP8_E4M3: dispatch.template operator()<__nv_fp8_e4m3>(); return;
        case DataType::FP8_E5M2: dispatch.template operator()<__nv_fp8_e5m2>(); return;
        default: throw std::logic_error("FP8 rotated calibration requires FP8 input.");
    }
#else
    static_cast<void>(input);
    static_cast<void>(fp32_partials);
    static_cast<void>(geometry);
    static_cast<void>(packet_bytes);
    static_cast<void>(exact_tile);
    static_cast<void>(rows_per_shard);
    static_cast<void>(shards_per_output);
    static_cast<void>(stream);
    throw std::logic_error("FP8 rotated calibration requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}

CubKernelOccupancyInfo queryFp8AwkwardAlignmentRotatedShardedCalibrationOccupancy(
    DataType input_dtype, size_t packet_bytes, bool exact_tile, uint64_t rows_per_shard) {
#if THOR_CUB_ENABLE_FP8_TYPES
    const auto dispatch = [&]<typename InputT>() -> CubKernelOccupancyInfo {
        if (packet_bytes == 4) return queryFp8RotatedCalibrationOccupancyForInput<InputT, 4, 4, false>(rows_per_shard);
        if (packet_bytes == 8) return queryFp8RotatedCalibrationOccupancyForInput<InputT, 8, 8, false>(rows_per_shard);
        if (packet_bytes == 16 && exact_tile) return queryFp8RotatedCalibrationOccupancyForInput<InputT, 16, 16, true>(rows_per_shard);
        if (packet_bytes == 16) return queryFp8RotatedCalibrationOccupancyForInput<InputT, 16, 16, false>(rows_per_shard);
        throw std::logic_error("FP8 rotated calibration occupancy received unsupported packet/layout combination.");
    };
    switch (input_dtype) {
        case DataType::FP8_E4M3: return dispatch.template operator()<__nv_fp8_e4m3>();
        case DataType::FP8_E5M2: return dispatch.template operator()<__nv_fp8_e5m2>();
        default: throw std::logic_error("FP8 rotated calibration occupancy requires FP8 input.");
    }
#else
    static_cast<void>(input_dtype);
    static_cast<void>(packet_bytes);
    static_cast<void>(exact_tile);
    static_cast<void>(rows_per_shard);
    throw std::logic_error("FP8 rotated calibration requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
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
#if THOR_CUB_ENABLE_FP8_TYPES
        case DataType::FP8_E4M3:
            return queryRotatedRCooperativeOccupancy<__nv_fp8_e4m3>(rows_per_shard);
        case DataType::FP8_E5M2:
            return queryRotatedRCooperativeOccupancy<__nv_fp8_e5m2>(rows_per_shard);
#endif
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
