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
// all R. This design shards both R and K, and eight physical warps cooperate on each (R shard, K tile).
//
// Awkward K shifts successive row starts relative to 16-byte boundaries. Rows owned by one warp differ by eight K
// strides, which is always a multiple of 16 bytes for FP16/BF16/FP32. Each warp can therefore align its row window
// down once and issue only aligned 16-byte packets throughout the hot loop. The shifted per-warp partial windows are
// realigned by logical component in shared memory before coalesced FP32 partial stores.
constexpr int AWKWARD_ROTATED_SHARDED_WARPS = 8;
constexpr int AWKWARD_ROTATED_SHARDED_THREADS =
    AWKWARD_ROTATED_SHARDED_WARPS * TILED_REDUCTION_WARP_THREADS;
constexpr uint64_t AWKWARD_ROTATED_MAX_GRID_BLOCKS = 65535;
static_assert(AWKWARD_ROTATED_SHARDED_THREADS == 256);

template <typename InputT>
inline constexpr int awkwardItemsPerLane = 16 / static_cast<int>(sizeof(InputT));

template <typename InputT>
inline constexpr int awkwardPacketSpanElements =
    TILED_REDUCTION_WARP_THREADS * awkwardItemsPerLane<InputT>;

template <typename InputT>
inline constexpr int awkwardLogicalTileElements =
    awkwardPacketSpanElements<InputT> - (awkwardItemsPerLane<InputT> - 1);

template <typename InputT>
[[nodiscard]] __device__ inline const InputT* alignPacketWindowDown16(const InputT* source) {
    const uintptr_t address = reinterpret_cast<uintptr_t>(source);
    return reinterpret_cast<const InputT*>(address & ~uintptr_t{15});
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

template <typename InputT, typename ReductionOpT, typename InputTransformT, int RowsPerShard>
__global__ void awkwardAlignmentRotatedShardedFirstStageKernel(const InputT* input,
                                                                float* fp32_partials,
                                                                uint64_t outer_size,
                                                                uint64_t reduction_size,
                                                                uint64_t inner_size,
                                                                uint64_t shards_per_output) {
    static_assert(RowsPerShard == 128 || RowsPerShard == 256 || RowsPerShard == 512 || RowsPerShard == 1024);
    constexpr int ItemsPerLane = awkwardItemsPerLane<InputT>;
    constexpr int PacketSpanElements = awkwardPacketSpanElements<InputT>;
    constexpr int LogicalTileElements = awkwardLogicalTileElements<InputT>;
    constexpr float Init = AwkwardReductionIdentityFp32<ReductionOpT>::value;
    static_assert(sizeof(InputT) == 2 || sizeof(InputT) == 4);
    static_assert(sizeof(InputT) * ItemsPerLane == 16);
    static_assert(LogicalTileElements > 0);

    __shared__ float warp_partials[AWKWARD_ROTATED_SHARDED_WARPS][PacketSpanElements];
    __shared__ uint32_t warp_prefix_elements[AWKWARD_ROTATED_SHARDED_WARPS];

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

        uint32_t prefix_elements = 0;
        if (first_row < row_end) {
            const InputT* logical_row_tile =
                input + (outer_index * reduction_size + first_row) * inner_size + tile_begin;
            const uintptr_t logical_address = reinterpret_cast<uintptr_t>(logical_row_tile);
            prefix_elements = static_cast<uint32_t>((logical_address & uintptr_t{15}) / sizeof(InputT));
            const InputT* aligned_row_window = alignPacketWindowDown16(logical_row_tile);

            constexpr uint64_t WarpRows = static_cast<uint64_t>(AWKWARD_ROTATED_SHARDED_WARPS);
            const uint64_t row_stride_elements = WarpRows * inner_size;
            const int64_t lane_packet_component =
                static_cast<int64_t>(tile_begin) - static_cast<int64_t>(prefix_elements)
                + static_cast<int64_t>(lane * ItemsPerLane);

            if (lane_packet_component < static_cast<int64_t>(inner_size)) {
                const InputT* lane_packet = aligned_row_window + lane * ItemsPerLane;
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

        if (lane == 0) {
            warp_prefix_elements[warp] = prefix_elements;
        }
#pragma unroll
        for (int item = 0; item < ItemsPerLane; ++item) {
            warp_partials[warp][lane * ItemsPerLane + item] = local[item];
        }
        __syncthreads();

        const uint64_t logical_component = static_cast<uint64_t>(threadIdx.x);
        if (logical_component < valid_components) {
            float reduced = Init;
#pragma unroll
            for (int source_warp = 0; source_warp < AWKWARD_ROTATED_SHARDED_WARPS; ++source_warp) {
                reduced = reduction_op(
                    reduced, warp_partials[source_warp][logical_component + warp_prefix_elements[source_warp]]);
            }
            fp32_partials[(outer_index * shards_per_output + shard) * inner_size + tile_begin + logical_component] =
                reduced;
        }
        __syncthreads();
    }
}

template <typename InputT, typename ReductionOpT, typename InputTransformT, int RowsPerShard>
void launchAwkwardFirstStageForTransform(const Tensor& input,
                                         Tensor& fp32_partials,
                                         const CubReductionGeometry& geometry,
                                         uint64_t shards_per_output,
                                         cudaStream_t stream) {
    constexpr int LogicalTileElements = awkwardLogicalTileElements<InputT>;
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

    awkwardAlignmentRotatedShardedFirstStageKernel<InputT, ReductionOpT, InputTransformT, RowsPerShard>
        <<<grid_blocks, AWKWARD_ROTATED_SHARDED_THREADS, 0, stream>>>(input.getMemPtr<InputT>(),
                                                                      fp32_partials.getMemPtr<float>(),
                                                                      geometry.outer_size,
                                                                      geometry.reduction_size,
                                                                      geometry.inner_size,
                                                                      shards_per_output);
    CUDA_CHECK(cudaGetLastError());
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

template <typename InputT, int RowsPerShard>
[[nodiscard]] CubKernelOccupancyInfo queryRotatedRCooperativeOccupancyForRows() {
    const auto kernel = awkwardAlignmentRotatedShardedFirstStageKernel<InputT,
                                                                        cuda::std::plus<float>,
                                                                        IdentityFp32,
                                                                        RowsPerShard>;
    cudaFuncAttributes attributes{};
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
    int active_blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks, kernel, AWKWARD_ROTATED_SHARDED_THREADS, 0));
    return CubKernelOccupancyInfo{attributes.numRegs,
                                  attributes.sharedSizeBytes,
                                  0,
                                  active_blocks,
                                  active_blocks * AWKWARD_ROTATED_SHARDED_WARPS};
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

}  // namespace

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
