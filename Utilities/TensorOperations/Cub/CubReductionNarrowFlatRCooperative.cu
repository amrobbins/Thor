#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"
#include "Utilities/TensorOperations/Cub/CubReductionOperation.cuh"

#include "Utilities/Exceptions.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace ThorImplementation::CubReductionInternal {
namespace {

// NARROW-FLAT-R-COOPERATIVE targets the low-precision K<=32 hole where a logical row is too narrow / awkwardly
// aligned for efficient row-at-a-time 16-byte ownership. Each warp instead reads eight complete rows as one flat,
// contiguous packet window. For FP16/BF16 this window is exactly 16*K bytes, so K<=32 needs at most 32 aligned
// 16-byte lane packets (plus an alignment prefix that still fits in one warp). Shared memory reconstructs [8,K]
// logical ownership after the coalesced load, then lane k reduces component k. Eight warps process independent
// 8-row groups and finally cooperate through a small FP32 [warp,K] scratch array.
constexpr int NARROW_FLAT_WARPS = 8;
constexpr int NARROW_FLAT_THREADS = NARROW_FLAT_WARPS * TILED_REDUCTION_WARP_THREADS;
constexpr int NARROW_FLAT_ROWS_PER_WARP_GROUP = 8;
constexpr int NARROW_FLAT_MAX_K = 32;
constexpr int NARROW_FLAT_MAX_GROUP_ELEMENTS = NARROW_FLAT_ROWS_PER_WARP_GROUP * NARROW_FLAT_MAX_K;
constexpr uint64_t NARROW_FLAT_MAX_GRID_BLOCKS = 65535;
static_assert(NARROW_FLAT_THREADS == 256);

// This kernel deliberately supports only 2-byte input storage. Eight rows then contain 16*K bytes, which is the
// geometry that lets one warp cover K<=32 using at most 32 native 16-byte packets.
template <typename InputT>
inline constexpr bool narrowFlatInputType =
    std::is_same_v<InputT, __half> || std::is_same_v<InputT, __nv_bfloat16>;

template <typename ReductionOpT>
struct NarrowFlatReductionIdentityFp32;

template <>
struct NarrowFlatReductionIdentityFp32<cuda::std::plus<float>> {
    static constexpr float value = 0.0f;
};

template <>
struct NarrowFlatReductionIdentityFp32<cuda::std::multiplies<float>> {
    static constexpr float value = 1.0f;
};

template <>
struct NarrowFlatReductionIdentityFp32<PropagatingMinimumFp32> {
    static constexpr float value = std::numeric_limits<float>::infinity();
};

template <>
struct NarrowFlatReductionIdentityFp32<PropagatingMaximumFp32> {
    static constexpr float value = -std::numeric_limits<float>::infinity();
};

template <typename InputT, typename ReductionOpT, typename InputTransformT, int RowsPerShard>
__global__ void narrowLowPrecisionFlatRCooperativeFirstStageKernel(const InputT* input,
                                                                    float* fp32_partials,
                                                                    uint64_t outer_size,
                                                                    uint64_t reduction_size,
                                                                    uint64_t inner_size,
                                                                    uint64_t shards_per_output) {
    static_assert(narrowFlatInputType<InputT>);
    static_assert(sizeof(InputT) == 2);
    static_assert(RowsPerShard == 128 || RowsPerShard == 256 || RowsPerShard == 512 || RowsPerShard == 1024);
    constexpr int ItemsPerPacket = 16 / static_cast<int>(sizeof(InputT));
    constexpr float Init = NarrowFlatReductionIdentityFp32<ReductionOpT>::value;
    static_assert(ItemsPerPacket == 8);

    // Store transformed FP32 values rather than 2-byte input values. Besides keeping the reduction phase type-simple,
    // this avoids two low-precision values contending for each 4-byte shared-memory bank during logical reconstruction.
    __shared__ float staged_rows[NARROW_FLAT_WARPS][NARROW_FLAT_MAX_GROUP_ELEMENTS];
    __shared__ float warp_partials[NARROW_FLAT_WARPS][NARROW_FLAT_MAX_K];

    const ReductionOpT reduction_op{};
    const InputTransformT input_transform{};
    const int warp = static_cast<int>(threadIdx.x) / TILED_REDUCTION_WARP_THREADS;
    const int lane = static_cast<int>(threadIdx.x) % TILED_REDUCTION_WARP_THREADS;
    const uint64_t total_work = outer_size * shards_per_output;
    const uint64_t grid_stride = static_cast<uint64_t>(gridDim.x);

    for (uint64_t work_index = static_cast<uint64_t>(blockIdx.x); work_index < total_work; work_index += grid_stride) {
        const uint64_t shard = work_index % shards_per_output;
        const uint64_t outer_index = work_index / shards_per_output;
        const uint64_t row_begin = shard * static_cast<uint64_t>(RowsPerShard);
        const uint64_t row_end = minU64(row_begin + static_cast<uint64_t>(RowsPerShard), reduction_size);

        float local = Init;
        const uint64_t first_group_row =
            row_begin + static_cast<uint64_t>(warp * NARROW_FLAT_ROWS_PER_WARP_GROUP);
        constexpr uint64_t RowsPerBlockRound =
            static_cast<uint64_t>(NARROW_FLAT_WARPS * NARROW_FLAT_ROWS_PER_WARP_GROUP);

        for (uint64_t group_row = first_group_row; group_row < row_end; group_row += RowsPerBlockRound) {
            const uint64_t group_rows = minU64(static_cast<uint64_t>(NARROW_FLAT_ROWS_PER_WARP_GROUP),
                                               row_end - group_row);

            if (group_rows == NARROW_FLAT_ROWS_PER_WARP_GROUP) {
                const InputT* group_begin = input + (outer_index * reduction_size + group_row) * inner_size;
                const uintptr_t group_address = reinterpret_cast<uintptr_t>(group_begin);
                const uint32_t prefix_elements =
                    static_cast<uint32_t>((group_address & uintptr_t{15}) / sizeof(InputT));
                const InputT* aligned_begin = reinterpret_cast<const InputT*>(group_address & ~uintptr_t{15});
                const uint64_t logical_group_elements =
                    static_cast<uint64_t>(NARROW_FLAT_ROWS_PER_WARP_GROUP) * inner_size;
                const uint64_t packet_count = ceilDivideU64(prefix_elements + logical_group_elements,
                                                            static_cast<uint64_t>(ItemsPerPacket));

                // For 2-byte K<=32, an eight-row group plus at most seven prefix elements always fits in 32 packets.
                if (static_cast<uint64_t>(lane) < packet_count) {
                    const PackedInputValues<InputT, ItemsPerPacket> values =
                        loadVectorizedInputPacket<InputT, ItemsPerPacket>(aligned_begin + lane * ItemsPerPacket);
#pragma unroll
                    for (int item = 0; item < ItemsPerPacket; ++item) {
                        const int64_t logical_index =
                            static_cast<int64_t>(lane * ItemsPerPacket + item)
                            - static_cast<int64_t>(prefix_elements);
                        if (logical_index >= 0
                            && static_cast<uint64_t>(logical_index) < logical_group_elements) {
                            staged_rows[warp][logical_index] =
                                input_transform(ToFp32<InputT>{}(values.values[item]));
                        }
                    }
                }
                __syncwarp();

                if (static_cast<uint64_t>(lane) < inner_size) {
#pragma unroll
                    for (int row = 0; row < NARROW_FLAT_ROWS_PER_WARP_GROUP; ++row) {
                        local = reduction_op(local,
                                             staged_rows[warp][static_cast<uint64_t>(row) * inner_size + lane]);
                    }
                }
                __syncwarp();
            } else if (static_cast<uint64_t>(lane) < inner_size) {
                // Only the final group of a shard can be short. Keep the common path entirely on contiguous 16-byte
                // packets and handle this at-most-seven-row tail with naturally aligned scalar low-precision accesses.
                for (uint64_t row = 0; row < group_rows; ++row) {
                    const InputT value = input[((outer_index * reduction_size + group_row + row) * inner_size) + lane];
                    local = reduction_op(local, input_transform(ToFp32<InputT>{}(value)));
                }
            }
        }

        if (lane < NARROW_FLAT_MAX_K) {
            warp_partials[warp][lane] = static_cast<uint64_t>(lane) < inner_size ? local : Init;
        }
        __syncthreads();

        if (threadIdx.x < inner_size) {
            const uint64_t component = static_cast<uint64_t>(threadIdx.x);
            float reduced = Init;
#pragma unroll
            for (int source_warp = 0; source_warp < NARROW_FLAT_WARPS; ++source_warp) {
                reduced = reduction_op(reduced, warp_partials[source_warp][component]);
            }
            fp32_partials[(outer_index * shards_per_output + shard) * inner_size + component] = reduced;
        }
        __syncthreads();
    }
}

template <typename InputT, typename ReductionOpT, typename InputTransformT, int RowsPerShard>
void launchNarrowFlatForTransform(const Tensor& input,
                                  Tensor& fp32_partials,
                                  const CubReductionGeometry& geometry,
                                  uint64_t shards_per_output,
                                  cudaStream_t stream) {
    if (geometry.outer_size > std::numeric_limits<uint64_t>::max() / shards_per_output) {
        throw std::overflow_error("Narrow flat R-cooperative first-stage work count overflows uint64_t.");
    }
    const uint64_t total_work = geometry.outer_size * shards_per_output;
    const unsigned int grid_blocks =
        static_cast<unsigned int>(std::min<uint64_t>(total_work, NARROW_FLAT_MAX_GRID_BLOCKS));
    if (grid_blocks == 0) {
        throw std::logic_error("Narrow flat R-cooperative first stage requires non-zero work.");
    }

    narrowLowPrecisionFlatRCooperativeFirstStageKernel<InputT, ReductionOpT, InputTransformT, RowsPerShard>
        <<<grid_blocks, NARROW_FLAT_THREADS, 0, stream>>>(input.getMemPtr<InputT>(),
                                                          fp32_partials.getMemPtr<float>(),
                                                          geometry.outer_size,
                                                          geometry.reduction_size,
                                                          geometry.inner_size,
                                                          shards_per_output);
    CUDA_CHECK(cudaGetLastError());
}

template <typename InputT, typename ReductionOpT, typename InputTransformT>
void dispatchNarrowFlatRows(const Tensor& input,
                            Tensor& fp32_partials,
                            const CubReductionGeometry& geometry,
                            uint64_t rows_per_shard,
                            uint64_t shards_per_output,
                            cudaStream_t stream) {
    switch (rows_per_shard) {
        case 128:
            launchNarrowFlatForTransform<InputT, ReductionOpT, InputTransformT, 128>(
                input, fp32_partials, geometry, shards_per_output, stream);
            return;
        case 256:
            launchNarrowFlatForTransform<InputT, ReductionOpT, InputTransformT, 256>(
                input, fp32_partials, geometry, shards_per_output, stream);
            return;
        case 512:
            launchNarrowFlatForTransform<InputT, ReductionOpT, InputTransformT, 512>(
                input, fp32_partials, geometry, shards_per_output, stream);
            return;
        case 1024:
            launchNarrowFlatForTransform<InputT, ReductionOpT, InputTransformT, 1024>(
                input, fp32_partials, geometry, shards_per_output, stream);
            return;
        default:
            throw std::logic_error("Unsupported narrow flat R-cooperative rows_per_shard specialization.");
    }
}

template <typename ReductionOpT, typename InputTransformT>
void dispatchNarrowFlatInput(const Tensor& input,
                             Tensor& fp32_partials,
                             const CubReductionGeometry& geometry,
                             uint64_t rows_per_shard,
                             uint64_t shards_per_output,
                             cudaStream_t stream) {
    switch (input.getDataType()) {
        case DataType::FP16:
            dispatchNarrowFlatRows<__half, ReductionOpT, InputTransformT>(
                input, fp32_partials, geometry, rows_per_shard, shards_per_output, stream);
            return;
        case DataType::BF16:
            dispatchNarrowFlatRows<__nv_bfloat16, ReductionOpT, InputTransformT>(
                input, fp32_partials, geometry, rows_per_shard, shards_per_output, stream);
            return;
        default:
            throw std::logic_error("Narrow flat R-cooperative first stage requires FP16 or BF16 input.");
    }
}

template <typename ReductionOpT>
void launchNarrowFlatForCombine(const CubReductionStageSemantics& semantics,
                                const Tensor& input,
                                Tensor& fp32_partials,
                                const CubReductionGeometry& geometry,
                                uint64_t rows_per_shard,
                                uint64_t shards_per_output,
                                cudaStream_t stream) {
    switch (semantics.input_transform) {
        case CubReductionStageInputTransform::Identity:
            dispatchNarrowFlatInput<ReductionOpT, IdentityFp32>(
                input, fp32_partials, geometry, rows_per_shard, shards_per_output, stream);
            return;
        case CubReductionStageInputTransform::AbsoluteValue:
            dispatchNarrowFlatInput<ReductionOpT, AbsoluteValueFp32>(
                input, fp32_partials, geometry, rows_per_shard, shards_per_output, stream);
            return;
        case CubReductionStageInputTransform::Square:
            dispatchNarrowFlatInput<ReductionOpT, SquareFp32>(
                input, fp32_partials, geometry, rows_per_shard, shards_per_output, stream);
            return;
    }
    throw std::logic_error("Unsupported narrow flat R-cooperative input transform.");
}

template <typename InputT, int RowsPerShard>
[[nodiscard]] CubKernelOccupancyInfo queryNarrowFlatOccupancyForRows() {
    const auto kernel = narrowLowPrecisionFlatRCooperativeFirstStageKernel<InputT,
                                                                           cuda::std::plus<float>,
                                                                           IdentityFp32,
                                                                           RowsPerShard>;
    cudaFuncAttributes attributes{};
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
    int active_blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks, kernel, NARROW_FLAT_THREADS, 0));
    return CubKernelOccupancyInfo{attributes.numRegs,
                                  attributes.sharedSizeBytes,
                                  0,
                                  active_blocks,
                                  active_blocks * NARROW_FLAT_WARPS};
}

template <typename InputT>
[[nodiscard]] CubKernelOccupancyInfo queryNarrowFlatOccupancy(uint64_t rows_per_shard) {
    switch (rows_per_shard) {
        case 128:
            return queryNarrowFlatOccupancyForRows<InputT, 128>();
        case 256:
            return queryNarrowFlatOccupancyForRows<InputT, 256>();
        case 512:
            return queryNarrowFlatOccupancyForRows<InputT, 512>();
        case 1024:
            return queryNarrowFlatOccupancyForRows<InputT, 1024>();
        default:
            throw std::logic_error("Narrow flat R-cooperative occupancy query received unsupported rows-per-shard.");
    }
}

void validateNarrowFlatFirstStage(const Tensor& input,
                                  const Tensor& fp32_partials,
                                  const CubReductionGeometry& geometry,
                                  uint64_t rows_per_shard,
                                  uint64_t shards_per_output) {
    if (input.getDataType() != DataType::FP16 && input.getDataType() != DataType::BF16) {
        throw std::logic_error("Narrow flat R-cooperative first stage requires FP16 or BF16 input.");
    }
    if (geometry.inner_size == 0 || geometry.inner_size > NARROW_FLAT_MAX_K) {
        throw std::logic_error("Narrow flat R-cooperative first stage requires 1 <= K <= 32.");
    }
    if ((rows_per_shard != 128 && rows_per_shard != 256 && rows_per_shard != 512 && rows_per_shard != 1024)
        || shards_per_output <= 1) {
        throw std::logic_error(
            "Narrow flat R-cooperative first stage requires rows_per_shard in {128,256,512,1024} and >1 shard.");
    }
    if (shards_per_output != ceilDivideU64(geometry.reduction_size, rows_per_shard)) {
        throw std::logic_error(
            "Narrow flat R-cooperative first stage requires fixed-row sharding consistent with rows_per_shard.");
    }
    if (fp32_partials.getDataType() != DataType::FP32
        || fp32_partials.getDimensions()
               != std::vector<uint64_t>{geometry.outer_size, shards_per_output, geometry.inner_size}) {
        throw std::logic_error("Narrow flat R-cooperative FP32 partial shape does not match stamped geometry.");
    }
}

}  // namespace

CubKernelOccupancyInfo queryNarrowLowPrecisionFlatRCooperativeFirstStageOccupancy(DataType input_dtype,
                                                                                   uint64_t rows_per_shard) {
    switch (input_dtype) {
        case DataType::FP16:
            return queryNarrowFlatOccupancy<__half>(rows_per_shard);
        case DataType::BF16:
            return queryNarrowFlatOccupancy<__nv_bfloat16>(rows_per_shard);
        default:
            throw std::logic_error("Narrow flat R-cooperative occupancy query requires FP16 or BF16 input.");
    }
}

void launchNarrowLowPrecisionFlatRCooperativeFirstStage(const CubReductionStageSemantics& semantics,
                                                         const Tensor& input,
                                                         Tensor& fp32_partials,
                                                         const CubReductionGeometry& geometry,
                                                         uint64_t rows_per_shard,
                                                         uint64_t shards_per_output,
                                                         Stream& stream) {
    validateNarrowFlatFirstStage(input, fp32_partials, geometry, rows_per_shard, shards_per_output);

    switch (semantics.combine) {
        case CubReductionStageCombine::Sum:
            launchNarrowFlatForCombine<cuda::std::plus<float>>(
                semantics, input, fp32_partials, geometry, rows_per_shard, shards_per_output, stream.getStream());
            return;
        case CubReductionStageCombine::Product:
            launchNarrowFlatForCombine<cuda::std::multiplies<float>>(
                semantics, input, fp32_partials, geometry, rows_per_shard, shards_per_output, stream.getStream());
            return;
        case CubReductionStageCombine::Minimum:
            launchNarrowFlatForCombine<PropagatingMinimumFp32>(
                semantics, input, fp32_partials, geometry, rows_per_shard, shards_per_output, stream.getStream());
            return;
        case CubReductionStageCombine::Maximum:
            launchNarrowFlatForCombine<PropagatingMaximumFp32>(
                semantics, input, fp32_partials, geometry, rows_per_shard, shards_per_output, stream.getStream());
            return;
    }
    throw std::logic_error("Unsupported narrow flat R-cooperative reduction combine.");
}

}  // namespace ThorImplementation::CubReductionInternal
