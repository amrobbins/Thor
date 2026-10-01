#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"
#include "Utilities/TensorOperations/Cub/CubReductionOperation.cuh"

#include "Utilities/Exceptions.h"

#include <cuda/std/bit>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace ThorImplementation::CubReductionInternal {
namespace {

constexpr int SHARDED_TILED_MIDDLE_WARPS = 8;
constexpr int SHARDED_TILED_MIDDLE_BLOCK_THREADS =
    SHARDED_TILED_MIDDLE_WARPS * TILED_REDUCTION_WARP_THREADS;

template <int ItemsPerLane>
__device__ inline void storeFp32PacketVectorized(float* destination, const float (&values)[ItemsPerLane]) {
    static_assert(ItemsPerLane == 4 || ItemsPerLane == 8);
    static_assert((ItemsPerLane * sizeof(float)) % 16 == 0);
    PackedInputValues<float, ItemsPerLane> packed{};
#pragma unroll
    for (int item = 0; item < ItemsPerLane; ++item) {
        packed.values[item] = values[item];
    }
    using RawT = RawVectorPacket<sizeof(float) * ItemsPerLane>;
    *reinterpret_cast<RawT*>(destination) = cuda::std::bit_cast<RawT>(packed);
}

template <typename ReductionOpT>
struct ShardedTiledReductionIdentityFp32;

template <>
struct ShardedTiledReductionIdentityFp32<cuda::std::plus<float>> {
    static constexpr float value = 0.0f;
};

template <>
struct ShardedTiledReductionIdentityFp32<cuda::std::multiplies<float>> {
    static constexpr float value = 1.0f;
};

template <>
struct ShardedTiledReductionIdentityFp32<PropagatingMinimumFp32> {
    static constexpr float value = std::numeric_limits<float>::infinity();
};

template <>
struct ShardedTiledReductionIdentityFp32<PropagatingMaximumFp32> {
    static constexpr float value = -std::numeric_limits<float>::infinity();
};

template <typename InputT, typename ReductionOpT, int ItemsPerLane, int RowsPerShard>
__global__ void shardedCooperativeTiledMiddleFirstStageKernel(const InputT* input,
                                                               float* workspace,
                                                               uint64_t outer_size,
                                                               uint64_t reduction_size,
                                                               uint64_t inner_size,
                                                               uint64_t shards_per_output) {
    static_assert(RowsPerShard == 128 || RowsPerShard == 256 || RowsPerShard == 512 || RowsPerShard == 1024);
    static_assert(sizeof(InputT) * ItemsPerLane == 16, "first stage requires one 16-byte input packet per thread");
    constexpr uint64_t components_per_tile =
        static_cast<uint64_t>(TILED_REDUCTION_WARP_THREADS) * static_cast<uint64_t>(ItemsPerLane);
    constexpr float Init = ShardedTiledReductionIdentityFp32<ReductionOpT>::value;
    const ReductionOpT reduction_op{};

    __shared__ float partials[SHARDED_TILED_MIDDLE_WARPS][ItemsPerLane][TILED_REDUCTION_WARP_THREADS];

    const int warp_in_output = static_cast<int>(threadIdx.x) / TILED_REDUCTION_WARP_THREADS;
    const int lane = static_cast<int>(threadIdx.x) % TILED_REDUCTION_WARP_THREADS;
    const uint64_t component_tiles = ceilDivideU64(inner_size, components_per_tile);
    const uint64_t total_work = outer_size * component_tiles * shards_per_output;
    const uint64_t grid_stride = static_cast<uint64_t>(gridDim.x);

    for (uint64_t work_index = static_cast<uint64_t>(blockIdx.x); work_index < total_work; work_index += grid_stride) {
        const uint64_t shard = work_index % shards_per_output;
        const uint64_t outer_component_work = work_index / shards_per_output;
        const uint64_t component_tile = outer_component_work % component_tiles;
        const uint64_t outer_index = outer_component_work / component_tiles;
        const uint64_t packet_component_begin =
            component_tile * components_per_tile
            + static_cast<uint64_t>(lane) * static_cast<uint64_t>(ItemsPerLane);
        const uint64_t row_begin = shard * static_cast<uint64_t>(RowsPerShard);
        const uint64_t shard_limit = row_begin + static_cast<uint64_t>(RowsPerShard);
        const uint64_t row_end = shard_limit < reduction_size ? shard_limit : reduction_size;

        float local[ItemsPerLane];
#pragma unroll
        for (int item = 0; item < ItemsPerLane; ++item) {
            local[item] = Init;
        }

        if (packet_component_begin + static_cast<uint64_t>(ItemsPerLane) <= inner_size) {
            const uint64_t first_row = row_begin + static_cast<uint64_t>(warp_in_output);
            if (first_row < row_end) {
                uint64_t input_index =
                    (outer_index * reduction_size + first_row) * inner_size + packet_component_begin;
                const uint64_t row_stride = static_cast<uint64_t>(SHARDED_TILED_MIDDLE_WARPS) * inner_size;
                for (uint64_t row = first_row; row < row_end;
                     row += static_cast<uint64_t>(SHARDED_TILED_MIDDLE_WARPS)) {
                    const PackedInputValues<InputT, ItemsPerLane> values =
                        loadVectorizedInputPacket<InputT, ItemsPerLane>(input + input_index);
#pragma unroll
                    for (int item = 0; item < ItemsPerLane; ++item) {
                        local[item] = reduction_op(local[item], ToFp32<InputT>{}(values.values[item]));
                    }
                    input_index += row_stride;
                }
            }
        }

#pragma unroll
        for (int item = 0; item < ItemsPerLane; ++item) {
            partials[warp_in_output][item][lane] = local[item];
        }
        __syncthreads();

#pragma unroll
        for (int stride = SHARDED_TILED_MIDDLE_WARPS / 2; stride > 0; stride >>= 1) {
            if (warp_in_output < stride) {
#pragma unroll
                for (int item = 0; item < ItemsPerLane; ++item) {
                    partials[warp_in_output][item][lane] = reduction_op(
                        partials[warp_in_output][item][lane],
                        partials[warp_in_output + stride][item][lane]);
                }
            }
            __syncthreads();
        }

        if (warp_in_output == 0 && packet_component_begin + static_cast<uint64_t>(ItemsPerLane) <= inner_size) {
            float reduced[ItemsPerLane];
#pragma unroll
            for (int item = 0; item < ItemsPerLane; ++item) {
                reduced[item] = partials[0][item][lane];
            }
            float* destination = workspace
                                 + (outer_index * shards_per_output + shard) * inner_size
                                 + packet_component_begin;
            storeFp32PacketVectorized<ItemsPerLane>(destination, reduced);
        }
        __syncthreads();
    }
}

constexpr int COOPERATIVE_SHARDED_TILED_WARPS = 8;
constexpr int COOPERATIVE_SHARDED_TILED_BLOCK_THREADS =
    COOPERATIVE_SHARDED_TILED_WARPS * TILED_REDUCTION_WARP_THREADS;

template <int ItemsPerLane>
__device__ inline void storeCooperativeShardedFp32Packet(float* destination,
                                                          const float (&values)[ItemsPerLane]) {
    static_assert(ItemsPerLane == 4 || ItemsPerLane == 8);
    static_assert((ItemsPerLane * sizeof(float)) % 16 == 0);
    PackedInputValues<float, ItemsPerLane> packed{};
#pragma unroll
    for (int item = 0; item < ItemsPerLane; ++item) {
        packed.values[item] = values[item];
    }
    using RawT = RawVectorPacket<sizeof(float) * ItemsPerLane>;
    *reinterpret_cast<RawT*>(destination) = cuda::std::bit_cast<RawT>(packed);
}

template <typename InputT, typename InputTransformT, int ItemsPerLane, int RowsPerShard>
__global__ void cooperativeShardedTiledFirstStageKernel(const InputT* input,
                                                         float* fp32_partials,
                                                         uint64_t outer_size,
                                                         uint64_t reduction_size,
                                                         uint64_t inner_size,
                                                         uint64_t shards_per_output,
                                                         InputTransformT input_transform) {
    static_assert(RowsPerShard == 128 || RowsPerShard == 256 || RowsPerShard == 512 || RowsPerShard == 1024);
    static_assert(sizeof(InputT) * ItemsPerLane == 16,
                  "cooperative sharded first stage requires one 16-byte input packet per lane");
    constexpr uint64_t components_per_tile =
        static_cast<uint64_t>(TILED_REDUCTION_WARP_THREADS) * static_cast<uint64_t>(ItemsPerLane);

    __shared__ float partials[COOPERATIVE_SHARDED_TILED_WARPS][ItemsPerLane][TILED_REDUCTION_WARP_THREADS];

    const int warp_in_tile = static_cast<int>(threadIdx.x) / TILED_REDUCTION_WARP_THREADS;
    const int lane = static_cast<int>(threadIdx.x) % TILED_REDUCTION_WARP_THREADS;
    const uint64_t component_tiles = ceilDivideU64(inner_size, components_per_tile);
    const uint64_t total_work = outer_size * component_tiles * shards_per_output;
    const uint64_t grid_stride = static_cast<uint64_t>(gridDim.x);

    for (uint64_t work_index = static_cast<uint64_t>(blockIdx.x); work_index < total_work; work_index += grid_stride) {
        const uint64_t shard = work_index % shards_per_output;
        const uint64_t outer_component_work = work_index / shards_per_output;
        const uint64_t component_tile = outer_component_work % component_tiles;
        const uint64_t outer_index = outer_component_work / component_tiles;
        const uint64_t packet_component_begin =
            component_tile * components_per_tile
            + static_cast<uint64_t>(lane) * static_cast<uint64_t>(ItemsPerLane);
        const uint64_t row_begin = shard * static_cast<uint64_t>(RowsPerShard);
        const uint64_t shard_limit = row_begin + static_cast<uint64_t>(RowsPerShard);
        const uint64_t row_end = shard_limit < reduction_size ? shard_limit : reduction_size;

        float local[ItemsPerLane];
#pragma unroll
        for (int item = 0; item < ItemsPerLane; ++item) {
            local[item] = 0.0f;
        }

        if (packet_component_begin + static_cast<uint64_t>(ItemsPerLane) <= inner_size) {
            const uint64_t first_row = row_begin + static_cast<uint64_t>(warp_in_tile);
            if (first_row < row_end) {
                uint64_t input_index =
                    (outer_index * reduction_size + first_row) * inner_size + packet_component_begin;
                const uint64_t row_stride = static_cast<uint64_t>(COOPERATIVE_SHARDED_TILED_WARPS) * inner_size;
                for (uint64_t row = first_row; row < row_end;
                     row += static_cast<uint64_t>(COOPERATIVE_SHARDED_TILED_WARPS)) {
                    const PackedInputValues<InputT, ItemsPerLane> values =
                        loadVectorizedInputPacket<InputT, ItemsPerLane>(input + input_index);
#pragma unroll
                    for (int item = 0; item < ItemsPerLane; ++item) {
                        local[item] += input_transform(ToFp32<InputT>{}(values.values[item]));
                    }
                    input_index += row_stride;
                }
            }
        }

        cooperativeTiledCombineWarpPartials<
            COOPERATIVE_SHARDED_TILED_WARPS, COOPERATIVE_SHARDED_TILED_WARPS, ItemsPerLane>(
            partials,
            local,
            warp_in_tile,
            warp_in_tile,
            lane,
            cuda::std::plus<float>{});

        if (warp_in_tile == 0 && packet_component_begin + static_cast<uint64_t>(ItemsPerLane) <= inner_size) {
            float reduced[ItemsPerLane];
#pragma unroll
            for (int item = 0; item < ItemsPerLane; ++item) {
                reduced[item] = partials[0][item][lane];
            }
            float* destination = fp32_partials
                                 + (outer_index * shards_per_output + shard) * inner_size
                                 + packet_component_begin;
            storeCooperativeShardedFp32Packet<ItemsPerLane>(destination, reduced);
        }
        __syncthreads();
    }
}

template <typename InputT, typename InputTransformT, typename ReductionOpT, int ItemsPerLane>
void launchForInputAndTransform(const Tensor& input,
                                Tensor& fp32_partials,
                                const CubReductionGeometry& geometry,
                                uint64_t rows_per_shard,
                                uint64_t shards_per_output,
                                InputTransformT input_transform,
                                cudaStream_t stream) {
    constexpr uint64_t components_per_tile =
        static_cast<uint64_t>(TILED_REDUCTION_WARP_THREADS) * static_cast<uint64_t>(ItemsPerLane);
    const uint64_t component_tiles = ceilDivideU64(geometry.inner_size, components_per_tile);
    if (geometry.outer_size > std::numeric_limits<uint64_t>::max() / component_tiles
        || geometry.outer_size * component_tiles > std::numeric_limits<uint64_t>::max() / shards_per_output) {
        throw std::overflow_error("Cooperative sharded tiled first-stage work count overflows uint64_t.");
    }
    const uint64_t total_work = geometry.outer_size * component_tiles * shards_per_output;
    const unsigned int grid_blocks = static_cast<unsigned int>(
        std::min<uint64_t>(total_work, TILED_REDUCTION_MAX_GRID_BLOCKS));
    if (grid_blocks == 0) {
        throw std::logic_error("Cooperative sharded tiled first stage requires non-zero work.");
    }

    const auto launch_rows = [&]<int RowsPerShard>() {
        if constexpr (std::is_same_v<InputTransformT, IdentityFp32>) {
            shardedCooperativeTiledMiddleFirstStageKernel<InputT, ReductionOpT, ItemsPerLane, RowsPerShard>
                <<<grid_blocks, SHARDED_TILED_MIDDLE_BLOCK_THREADS, 0, stream>>>(
                    input.getMemPtr<InputT>(),
                    fp32_partials.getMemPtr<float>(),
                    geometry.outer_size,
                    geometry.reduction_size,
                    geometry.inner_size,
                    shards_per_output);
        } else {
            static_assert(std::is_same_v<ReductionOpT, cuda::std::plus<float>>,
                          "transformed cooperative sharded reductions are additive only");
            cooperativeShardedTiledFirstStageKernel<InputT, InputTransformT, ItemsPerLane, RowsPerShard>
                <<<grid_blocks, COOPERATIVE_SHARDED_TILED_BLOCK_THREADS, 0, stream>>>(
                    input.getMemPtr<InputT>(),
                    fp32_partials.getMemPtr<float>(),
                    geometry.outer_size,
                    geometry.reduction_size,
                    geometry.inner_size,
                    shards_per_output,
                    input_transform);
        }
    };

    switch (rows_per_shard) {
        case 128:
            launch_rows.template operator()<128>();
            break;
        case 256:
            launch_rows.template operator()<256>();
            break;
        case 512:
            launch_rows.template operator()<512>();
            break;
        case 1024:
            launch_rows.template operator()<1024>();
            break;
        default:
            throw std::logic_error("Unsupported cooperative sharded tiled rows-per-shard value.");
    }
    CUDA_CHECK(cudaGetLastError());
}

template <typename InputTransformT, typename ReductionOpT>
void dispatchInputType(const Tensor& input,
                       Tensor& fp32_partials,
                       const CubReductionGeometry& geometry,
                       uint64_t rows_per_shard,
                       uint64_t shards_per_output,
                       InputTransformT input_transform,
                       cudaStream_t stream) {
    switch (input.getDataType()) {
        case DataType::FP16:
            launchForInputAndTransform<__half, InputTransformT, ReductionOpT, 8>(
                input, fp32_partials, geometry, rows_per_shard, shards_per_output, input_transform, stream);
            return;
        case DataType::BF16:
            launchForInputAndTransform<__nv_bfloat16, InputTransformT, ReductionOpT, 8>(
                input, fp32_partials, geometry, rows_per_shard, shards_per_output, input_transform, stream);
            return;
        case DataType::FP32:
            launchForInputAndTransform<float, InputTransformT, ReductionOpT, 4>(
                input, fp32_partials, geometry, rows_per_shard, shards_per_output, input_transform, stream);
            return;
        default:
            throw std::logic_error("Cooperative sharded tiled first stage received an unsupported input dtype.");
    }
}

template <typename InputT, int ItemsPerLane, int RowsPerShard>
[[nodiscard]] CubKernelOccupancyInfo queryAlignedRCooperativeOccupancyForRows() {
    const auto kernel = shardedCooperativeTiledMiddleFirstStageKernel<InputT,
                                                                       cuda::std::plus<float>,
                                                                       ItemsPerLane,
                                                                       RowsPerShard>;
    cudaFuncAttributes attributes{};
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
    int active_blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks, kernel, SHARDED_TILED_MIDDLE_BLOCK_THREADS, 0));
    return CubKernelOccupancyInfo{attributes.numRegs,
                                  attributes.sharedSizeBytes,
                                  0,
                                  active_blocks,
                                  active_blocks * SHARDED_TILED_MIDDLE_WARPS};
}

template <typename InputT, int ItemsPerLane>
[[nodiscard]] CubKernelOccupancyInfo queryAlignedRCooperativeOccupancy(uint64_t rows_per_shard) {
    switch (rows_per_shard) {
        case 128:
            return queryAlignedRCooperativeOccupancyForRows<InputT, ItemsPerLane, 128>();
        case 256:
            return queryAlignedRCooperativeOccupancyForRows<InputT, ItemsPerLane, 256>();
        case 512:
            return queryAlignedRCooperativeOccupancyForRows<InputT, ItemsPerLane, 512>();
        case 1024:
            return queryAlignedRCooperativeOccupancyForRows<InputT, ItemsPerLane, 1024>();
        default:
            throw std::logic_error("Aligned R-cooperative occupancy query received unsupported rows-per-shard.");
    }
}

}  // namespace

CubKernelOccupancyInfo queryCooperativeShardedTiledFirstStageOccupancy(DataType input_dtype,
                                                                        uint64_t rows_per_shard) {
    switch (input_dtype) {
        case DataType::FP16:
            return queryAlignedRCooperativeOccupancy<__half, 8>(rows_per_shard);
        case DataType::BF16:
            return queryAlignedRCooperativeOccupancy<__nv_bfloat16, 8>(rows_per_shard);
        case DataType::FP32:
            return queryAlignedRCooperativeOccupancy<float, 4>(rows_per_shard);
        default:
            throw std::logic_error("Aligned R-cooperative occupancy query received unsupported input dtype.");
    }
}

void launchCooperativeShardedTiledFirstStage(const CubReductionStageSemantics& semantics,
                                             const Tensor& input,
                                             Tensor& fp32_partials,
                                             const CubReductionGeometry& geometry,
                                             uint64_t rows_per_shard,
                                             uint64_t shards_per_output,
                                             Stream& stream) {
    if (fp32_partials.getDataType() != DataType::FP32) {
        throw std::logic_error("Cooperative sharded tiled first stage requires FP32 partial storage.");
    }
    const std::vector<uint64_t> expected_partial_dimensions = {
        geometry.outer_size, shards_per_output, geometry.inner_size};
    if (fp32_partials.getDimensions() != expected_partial_dimensions) {
        throw std::logic_error("Cooperative sharded tiled FP32 partial shape does not match stamped geometry.");
    }

    switch (semantics.combine) {
        case CubReductionStageCombine::Sum:
            switch (semantics.input_transform) {
                case CubReductionStageInputTransform::Identity:
                    dispatchInputType<IdentityFp32, cuda::std::plus<float>>(input,
                                                                           fp32_partials,
                                                                           geometry,
                                                                           rows_per_shard,
                                                                           shards_per_output,
                                                                           IdentityFp32{},
                                                                           stream.getStream());
                    return;
                case CubReductionStageInputTransform::AbsoluteValue:
                    dispatchInputType<AbsoluteValueFp32, cuda::std::plus<float>>(input,
                                                                                fp32_partials,
                                                                                geometry,
                                                                                rows_per_shard,
                                                                                shards_per_output,
                                                                                AbsoluteValueFp32{},
                                                                                stream.getStream());
                    return;
                case CubReductionStageInputTransform::Square:
                    dispatchInputType<SquareFp32, cuda::std::plus<float>>(input,
                                                                         fp32_partials,
                                                                         geometry,
                                                                         rows_per_shard,
                                                                         shards_per_output,
                                                                         SquareFp32{},
                                                                         stream.getStream());
                    return;
            }
            break;
        case CubReductionStageCombine::Product:
            if (semantics.input_transform != CubReductionStageInputTransform::Identity) {
                break;
            }
            dispatchInputType<IdentityFp32, cuda::std::multiplies<float>>(input,
                                                                          fp32_partials,
                                                                          geometry,
                                                                          rows_per_shard,
                                                                          shards_per_output,
                                                                          IdentityFp32{},
                                                                          stream.getStream());
            return;
        case CubReductionStageCombine::Minimum:
            if (semantics.input_transform != CubReductionStageInputTransform::Identity) {
                break;
            }
            dispatchInputType<IdentityFp32, PropagatingMinimumFp32>(input,
                                                                    fp32_partials,
                                                                    geometry,
                                                                    rows_per_shard,
                                                                    shards_per_output,
                                                                    IdentityFp32{},
                                                                    stream.getStream());
            return;
        case CubReductionStageCombine::Maximum:
            if (semantics.input_transform != CubReductionStageInputTransform::Identity) {
                break;
            }
            dispatchInputType<IdentityFp32, PropagatingMaximumFp32>(input,
                                                                    fp32_partials,
                                                                    geometry,
                                                                    rows_per_shard,
                                                                    shards_per_output,
                                                                    IdentityFp32{},
                                                                    stream.getStream());
            return;
    }
    throw std::logic_error("Unsupported cooperative sharded tiled stage semantics.");
}

}  // namespace ThorImplementation::CubReductionInternal
