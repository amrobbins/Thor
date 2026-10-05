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

// NARROW-FLAT-R-COOPERATIVE targets the low-precision geometry where a logical row is too narrow / awkwardly aligned
// for efficient row-at-a-time ownership. FP16/BF16 retain the established K<=32 fixed-group implementation. FP8 keeps
// p16 global packets but calibrates a broader physical shape below: adapt row-group depth toward one 512-byte warp
// window, preserve all eight warps across the R shard, and let each lane own multiple components when K>32. Production
// policy remains unchanged until that FP8 calibration establishes the useful niche and handoff to Rotated.
constexpr int NARROW_FLAT_WARPS = 8;
constexpr int NARROW_FLAT_THREADS = NARROW_FLAT_WARPS * TILED_REDUCTION_WARP_THREADS;
constexpr int NARROW_FLAT_MAX_K = 32;
constexpr uint64_t NARROW_FLAT_MAX_GRID_BLOCKS = 65535;
static_assert(NARROW_FLAT_THREADS == 256);

template <typename InputT>
inline constexpr bool narrowFlatInputType = std::is_same_v<InputT, __half> || std::is_same_v<InputT, __nv_bfloat16>
#if THOR_CUB_ENABLE_FP8_TYPES
                                            || std::is_same_v<InputT, __nv_fp8_e4m3> || std::is_same_v<InputT, __nv_fp8_e5m2>
#endif
    ;

template <typename InputT>
inline constexpr int narrowFlatRowsPerWarpGroup = 16 / static_cast<int>(sizeof(InputT));

template <typename InputT>
inline constexpr int narrowFlatMaxGroupElements = narrowFlatRowsPerWarpGroup<InputT> * NARROW_FLAT_MAX_K;

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
    static_assert(sizeof(InputT) == 1 || sizeof(InputT) == 2);
    static_assert(RowsPerShard == 128 || RowsPerShard == 256 || RowsPerShard == 512 || RowsPerShard == 1024 || RowsPerShard == 2048 ||
                  RowsPerShard == 4096 || RowsPerShard == 8192 || RowsPerShard == 16384);
    constexpr int ItemsPerPacket = 16 / static_cast<int>(sizeof(InputT));
    constexpr int RowsPerWarpGroup = narrowFlatRowsPerWarpGroup<InputT>;
    constexpr int MaxGroupElements = narrowFlatMaxGroupElements<InputT>;
    constexpr float Init = NarrowFlatReductionIdentityFp32<ReductionOpT>::value;
    static_assert(ItemsPerPacket == 8 || ItemsPerPacket == 16);
    static_assert(RowsPerWarpGroup == 8 || RowsPerWarpGroup == 16);
    static_assert(MaxGroupElements <= TILED_REDUCTION_WARP_THREADS * ItemsPerPacket);

    // Store transformed FP32 values rather than low-precision input values. This keeps the reduction phase type-simple
    // and avoids packed low-precision shared-memory bank contention during logical reconstruction.
    __shared__ float staged_rows[NARROW_FLAT_WARPS][MaxGroupElements];
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
        const uint64_t first_group_row = row_begin + static_cast<uint64_t>(warp * RowsPerWarpGroup);
        constexpr uint64_t RowsPerBlockRound = static_cast<uint64_t>(NARROW_FLAT_WARPS * RowsPerWarpGroup);

        for (uint64_t group_row = first_group_row; group_row < row_end; group_row += RowsPerBlockRound) {
            const uint64_t group_rows = minU64(static_cast<uint64_t>(RowsPerWarpGroup), row_end - group_row);
            const InputT* group_begin = input + (outer_index * reduction_size + group_row) * inner_size;
            const uintptr_t group_address = reinterpret_cast<uintptr_t>(group_begin);
            const uint32_t prefix_elements = static_cast<uint32_t>((group_address & uintptr_t{15}) / sizeof(InputT));
            const InputT* aligned_begin = reinterpret_cast<const InputT*>(group_address & ~uintptr_t{15});
            const uint64_t logical_group_elements = group_rows * inner_size;
            const uint64_t packet_count = ceilDivideU64(prefix_elements + logical_group_elements, static_cast<uint64_t>(ItemsPerPacket));

            // The full FP16/BF16 group is 8*K*2 bytes and the full FP8 group is 16*K*1 byte: both are 16*K bytes.
            // K<=31 plus the largest possible alignment prefix is <512 bytes; K=32 starts naturally 16-byte aligned.
            // Short final groups are smaller still. Therefore all supported groups fit in one warp of 16-byte packets,
            // including the tail; there is deliberately no scalar FP8 global-memory path.
            if (static_cast<uint64_t>(lane) < packet_count) {
                const PackedInputValues<InputT, ItemsPerPacket> values =
                    loadVectorizedInputPacket<InputT, ItemsPerPacket>(aligned_begin + lane * ItemsPerPacket);
                for (int item = 0; item < ItemsPerPacket; ++item) {
                    const int64_t logical_index =
                        static_cast<int64_t>(lane * ItemsPerPacket + item) - static_cast<int64_t>(prefix_elements);
                    if (logical_index >= 0 && static_cast<uint64_t>(logical_index) < logical_group_elements) {
                        staged_rows[warp][logical_index] = input_transform(ToFp32<InputT>{}(values.values[item]));
                    }
                }
            }
            __syncwarp();

            if (static_cast<uint64_t>(lane) < inner_size) {
                for (uint64_t row = 0; row < group_rows; ++row) {
                    local = reduction_op(local, staged_rows[warp][row * inner_size + lane]);
                }
            }
            __syncwarp();
        }

        if (lane < NARROW_FLAT_MAX_K) {
            warp_partials[warp][lane] = static_cast<uint64_t>(lane) < inner_size ? local : Init;
        }
        __syncthreads();

        if (threadIdx.x < inner_size) {
            const uint64_t component = static_cast<uint64_t>(threadIdx.x);
            float reduced = Init;
            for (int source_warp = 0; source_warp < NARROW_FLAT_WARPS; ++source_warp) {
                reduced = reduction_op(reduced, warp_partials[source_warp][component]);
            }
            fp32_partials[(outer_index * shards_per_output + shard) * inner_size + component] = reduced;
        }
        __syncthreads();
    }
}

// FP8 FlatRows calibration exposes 4/8/16-byte global packets while preserving one coherent eight-warp topology.
// Each packet width adaptively fills one warp-wide packet window and reserves its worst-case alignment prefix, so the
// logical group limits are p4=125, p8=249, and p16=497 FP8 bytes. K<=8 retains the dedicated same-component
// cooperative ownership specialization; K>=9 keeps simple lane ownership with multiple components per lane above K=32.
// Packet-width variation is calibration-only; production remains on the established p16 path.
constexpr int NARROW_FP8_PACKETS_PER_WARP = TILED_REDUCTION_WARP_THREADS;
constexpr int NARROW_FP8_CALIBRATION_MAX_K = 256;
static_assert(NARROW_FP8_CALIBRATION_MAX_K == NARROW_FLAT_THREADS);

template <int PacketBytes>
inline constexpr int narrowFp8PacketWindowBytes = PacketBytes * NARROW_FP8_PACKETS_PER_WARP;

template <int PacketBytes>
inline constexpr int narrowFp8MaxLogicalGroupBytes = narrowFp8PacketWindowBytes<PacketBytes> - (PacketBytes - 1);

static_assert(narrowFp8PacketWindowBytes<4> == 128 && narrowFp8MaxLogicalGroupBytes<4> == 125);
static_assert(narrowFp8PacketWindowBytes<8> == 256 && narrowFp8MaxLogicalGroupBytes<8> == 249);
static_assert(narrowFp8PacketWindowBytes<16> == 512 && narrowFp8MaxLogicalGroupBytes<16> == 497);

template <typename InputT>
[[nodiscard]] __device__ inline InputT narrowFp8FromBits(uint8_t bits) {
    static_assert(sizeof(InputT) == 1);
    if constexpr (std::is_same_v<InputT, __nv_fp8_e4m3>) {
        return ThorLowPrecision::fp8E4M3FromBits(static_cast<__nv_fp8_storage_t>(bits));
    } else {
        static_assert(std::is_same_v<InputT, __nv_fp8_e5m2>);
        return ThorLowPrecision::fp8E5M2FromBits(static_cast<__nv_fp8_storage_t>(bits));
    }
}

// DEPRECATED EXPERIMENT ONLY -- NOT A PRODUCTION CANDIDATE. This one-output-per-warp prototype established that
// simplifying the shallow-R / narrow-K geometry matters, but the focused census showed that it still leaves the warp
// badly underutilized and reaches no line-rate cells. Keep it only as a calibration baseline while the adaptive
// multi-output/R-parallel replacement below is developed. Eight independent warps share one 256-thread CTA;
// each warp owns one complete outer output and therefore needs no inter-warp reduction or block-wide barrier. Global
// input traffic is always 4-byte packetized. Thor guarantees 128 bytes of allocation padding after every tensor, so
// the final aligned p4 load may safely overread by at most three bytes; only logical rows/components are accumulated.
// This geometry is intentionally narrow: it exists to test the small-R x small/medium-K hole exposed by the full-space
// census, not to overlap the established deep-R FlatRows/Rotated inventory.
constexpr int OUTER_GROUPED_FP8_WARPS = 8;
constexpr int OUTER_GROUPED_FP8_THREADS = OUTER_GROUPED_FP8_WARPS * TILED_REDUCTION_WARP_THREADS;
constexpr int OUTER_GROUPED_FP8_PACKET_BYTES = 4;
constexpr int OUTER_GROUPED_FP8_MAX_K = 64;
constexpr uint64_t OUTER_GROUPED_FP8_MAX_R = 1024;
constexpr int OUTER_GROUPED_FP8_PACKETS_PER_WARP = TILED_REDUCTION_WARP_THREADS;
constexpr int OUTER_GROUPED_FP8_PACKET_WINDOW_BYTES = OUTER_GROUPED_FP8_PACKET_BYTES * OUTER_GROUPED_FP8_PACKETS_PER_WARP;
constexpr int OUTER_GROUPED_FP8_MAX_LOGICAL_GROUP_BYTES = OUTER_GROUPED_FP8_PACKET_WINDOW_BYTES - (OUTER_GROUPED_FP8_PACKET_BYTES - 1);
static_assert(OUTER_GROUPED_FP8_THREADS == 256);
static_assert(OUTER_GROUPED_FP8_PACKET_WINDOW_BYTES == 128);
static_assert(OUTER_GROUPED_FP8_MAX_LOGICAL_GROUP_BYTES == 125);

template <typename InputT, int MaxComponentsPerLane>
__global__ void fp8OuterGroupedP4RCooperativeCompleteKernel(
    const InputT* input, InputT* output, uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size) {
    static_assert(sizeof(InputT) == 1);
    static_assert(MaxComponentsPerLane == 1 || MaxComponentsPerLane == 2);
    constexpr int MaxSupportedK = TILED_REDUCTION_WARP_THREADS * MaxComponentsPerLane;
    static_assert(MaxSupportedK <= OUTER_GROUPED_FP8_MAX_K);

    __shared__ RawVectorPacket<OUTER_GROUPED_FP8_PACKET_BYTES> staged_packets[OUTER_GROUPED_FP8_WARPS][OUTER_GROUPED_FP8_PACKETS_PER_WARP];

    const int warp = static_cast<int>(threadIdx.x) / TILED_REDUCTION_WARP_THREADS;
    const int lane = static_cast<int>(threadIdx.x) % TILED_REDUCTION_WARP_THREADS;
    const uint64_t rows_per_group = static_cast<uint64_t>(OUTER_GROUPED_FP8_MAX_LOGICAL_GROUP_BYTES) / inner_size;
    const uint64_t outer_group_stride = static_cast<uint64_t>(gridDim.x);

    for (uint64_t outer_group = static_cast<uint64_t>(blockIdx.x);; outer_group += outer_group_stride) {
        const uint64_t outer_index = outer_group * static_cast<uint64_t>(OUTER_GROUPED_FP8_WARPS) + static_cast<uint64_t>(warp);
        if (outer_index >= outer_size) {
            return;
        }

        float local[MaxComponentsPerLane];
        for (int slot = 0; slot < MaxComponentsPerLane; ++slot) {
            local[slot] = 0.0f;
        }

        for (uint64_t group_row = 0; group_row < reduction_size; group_row += rows_per_group) {
            const uint64_t group_rows = minU64(rows_per_group, reduction_size - group_row);
            const InputT* group_begin = input + (outer_index * reduction_size + group_row) * inner_size;
            const uintptr_t group_address = reinterpret_cast<uintptr_t>(group_begin);
            const uint32_t prefix_elements = static_cast<uint32_t>(group_address & uintptr_t{OUTER_GROUPED_FP8_PACKET_BYTES - 1});
            const auto* aligned_begin = reinterpret_cast<const RawVectorPacket<OUTER_GROUPED_FP8_PACKET_BYTES>*>(
                group_address & ~uintptr_t{OUTER_GROUPED_FP8_PACKET_BYTES - 1});
            const uint64_t logical_group_elements = group_rows * inner_size;
            const uint64_t packet_count = ceilDivideU64(static_cast<uint64_t>(prefix_elements) + logical_group_elements,
                                                        static_cast<uint64_t>(OUTER_GROUPED_FP8_PACKET_BYTES));

            if (static_cast<uint64_t>(lane) < packet_count) {
                // Deliberately unconditional. Thor tensor allocations provide 128 bytes of trailing padding, while a
                // p4 packet can overread the logical tensor end by at most three bytes. Logical masking below prevents
                // those padding bytes (or neighboring logical rows) from contributing to the reduction.
                staged_packets[warp][lane] = aligned_begin[lane];
            }
            __syncwarp();

            const auto* staged_bytes = reinterpret_cast<const uint8_t*>(&staged_packets[warp][0]);
            for (int slot = 0; slot < MaxComponentsPerLane; ++slot) {
                const uint64_t component = static_cast<uint64_t>(lane + slot * TILED_REDUCTION_WARP_THREADS);
                if (component < inner_size) {
                    for (uint64_t row = 0; row < group_rows; ++row) {
                        const uint64_t staged_index = static_cast<uint64_t>(prefix_elements) + row * inner_size + component;
                        const InputT value = narrowFp8FromBits<InputT>(staged_bytes[staged_index]);
                        local[slot] += ToFp32<InputT>{}(value);
                    }
                }
            }
            __syncwarp();
        }

        for (int slot = 0; slot < MaxComponentsPerLane; ++slot) {
            const uint64_t component = static_cast<uint64_t>(lane + slot * TILED_REDUCTION_WARP_THREADS);
            if (component < inner_size) {
                // Consecutive lanes own consecutive FP8 outputs, so these scalar stores coalesce naturally without a
                // shared-memory ownership transpose. A second slot covers components 32..63 when K>32.
                output[outer_index * inner_size + component] = ThorLowPrecision::castToStorage<InputT>(local[slot]);
            }
        }
    }
}

template <typename InputT, int MaxComponentsPerLane>
void launchFp8OuterGroupedP4ForComponents(const Tensor& input, Tensor& output, const CubReductionGeometry& geometry, cudaStream_t stream) {
    static_assert(sizeof(InputT) == 1);
    static_assert(MaxComponentsPerLane == 1 || MaxComponentsPerLane == 2);
    const uint64_t block_count =
        std::min<uint64_t>(ceilDivideU64(geometry.outer_size, static_cast<uint64_t>(OUTER_GROUPED_FP8_WARPS)), NARROW_FLAT_MAX_GRID_BLOCKS);
    fp8OuterGroupedP4RCooperativeCompleteKernel<InputT, MaxComponentsPerLane>
        <<<static_cast<unsigned int>(block_count), OUTER_GROUPED_FP8_THREADS, 0, stream>>>(
            input.getMemPtr<InputT>(), output.getMemPtr<InputT>(), geometry.outer_size, geometry.reduction_size, geometry.inner_size);
    CUDA_CHECK(cudaGetLastError());
}

template <typename InputT, int MaxComponentsPerLane>
[[nodiscard]] CubKernelOccupancyInfo queryFp8OuterGroupedP4OccupancyForComponents() {
    static_assert(sizeof(InputT) == 1);
    const auto kernel = fp8OuterGroupedP4RCooperativeCompleteKernel<InputT, MaxComponentsPerLane>;
    cudaFuncAttributes attributes{};
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
    int active_blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks, kernel, OUTER_GROUPED_FP8_THREADS, 0));
    return CubKernelOccupancyInfo{
        attributes.numRegs, attributes.sharedSizeBytes, 0, active_blocks, active_blocks * OUTER_GROUPED_FP8_WARPS};
}

// Shallow-R / narrow-K calibration family. The original adaptive kernel proved that multiple outputs per warp is
// the missing geometry for tiny reductions, but one monolithic kernel also made the wider-K crossover harder to
// optimize. Keep the three physical mechanisms separate so calibration can choose rather than forcing every shape
// through the same hot loop:
//   * CompactMultiOutput: several complete adjacent outputs per warp, with p4-aligned group strides and p4 writeback.
//   * RParallel: one output per warp, otherwise-idle lanes cooperate across R for each K component.
//   * DirectComponent: one output per warp, one lane/component, used when K itself already supplies enough parallelism.
// DirectComponent remains p4-only. CompactMultiOutput retains its established p4/FP32-shared control and calibrates
// p8/p16 producer packets through packed shared planes. One-output aligned RParallel keeps the ownership remap as raw
// p4 shared words, while MultiOutput RParallel likewise calibrates general p8/p16 packed shared staging. All paths
// accumulate in FP32. The candidates remain
// calibration-only until the focused census establishes a production selector.
constexpr int SHALLOW_FP8_WARPS = 8;
constexpr int SHALLOW_FP8_THREADS = SHALLOW_FP8_WARPS * TILED_REDUCTION_WARP_THREADS;
constexpr int SHALLOW_FP8_PACKET_BYTES = 4;
constexpr int SHALLOW_FP8_MAX_K = 32;
constexpr int RPARALLEL_FP8_MAX_K = 256;
constexpr uint64_t SHALLOW_FP8_MAX_R = 1024;
constexpr int SHALLOW_FP8_PACKET_WINDOW_BYTES = SHALLOW_FP8_PACKET_BYTES * TILED_REDUCTION_WARP_THREADS;
constexpr int SHALLOW_FP8_MAX_UNALIGNED_LOGICAL_BYTES = SHALLOW_FP8_PACKET_WINDOW_BYTES - (SHALLOW_FP8_PACKET_BYTES - 1);
static_assert(SHALLOW_FP8_THREADS == 256);
static_assert(SHALLOW_FP8_PACKET_WINDOW_BYTES == 128);
static_assert(SHALLOW_FP8_MAX_UNALIGNED_LOGICAL_BYTES == 125);

[[nodiscard]] inline bool shallowFp8SupportedCalibrationK(uint64_t inner_size) {
    switch (inner_size) {
        case 2:
        case 3:
        case 4:
        case 5:
        case 7:
        case 8:
        case 9:
        case 11:
        case 13:
        case 15:
        case 16:
        case 17:
        case 19:
        case 31:
        case 32:
            return true;
        default:
            return false;
    }
}

[[nodiscard]] inline bool rParallelFp8SupportedCalibrationK(uint64_t inner_size) {
    if (inner_size == 1 || shallowFp8SupportedCalibrationK(inner_size)) {
        return true;
    }
    switch (inner_size) {
        case 63:
        case 64:
        case 127:
        case 128:
        case 129:
        case 255:
        case 256:
            return true;
        default:
            return false;
    }
}

[[nodiscard]] inline bool rParallelFp8PacketSupportsOwnership(
    uint64_t inner_size, uint32_t outputs_per_warp, uint32_t packet_bytes) {
    if ((outputs_per_warp != 1 && outputs_per_warp != 2 && outputs_per_warp != 4 && outputs_per_warp != 8) ||
        inner_size == 0 || inner_size > RPARALLEL_FP8_MAX_K ||
        (packet_bytes != 4 && packet_bytes != 8 && packet_bytes != 16)) {
        return false;
    }
    const uint64_t lanes_per_output = TILED_REDUCTION_WARP_THREADS / outputs_per_warp;
    const uint64_t packet_window = lanes_per_output * packet_bytes;
    const bool row_aligned = (inner_size % packet_bytes) == 0;
    const uint64_t logical_capacity = row_aligned ? packet_window : packet_window - (packet_bytes - 1);
    return inner_size <= logical_capacity;
}

[[nodiscard]] inline bool shallowFp8FitsUint32Indexing(const CubReductionGeometry& geometry) {
    constexpr uint64_t MaxIndex = std::numeric_limits<uint32_t>::max();
    return geometry.outer_size <= MaxIndex && geometry.reduction_size <= MaxIndex && geometry.input_elements <= MaxIndex &&
           geometry.output_elements <= MaxIndex;
}

[[nodiscard]] inline uint32_t shallowFp8CompactOutputsPerWarp(uint64_t reduction_size, uint64_t inner_size) {
    if (reduction_size == 0 || inner_size == 0) {
        return 1;
    }
    const uint64_t bytes_per_output = reduction_size * inner_size;
    if (bytes_per_output == 0 || bytes_per_output > SHALLOW_FP8_PACKET_WINDOW_BYTES) {
        return 1;
    }
    const uint32_t max_outputs = static_cast<uint32_t>(
        std::min<uint64_t>(TILED_REDUCTION_WARP_THREADS, static_cast<uint64_t>(SHALLOW_FP8_PACKET_WINDOW_BYTES) / bytes_per_output));

    // Keep both the input group stride and output group stride naturally p4 aligned. Powers of two make ownership
    // compile-time once dispatched below. Odd K therefore needs >=4 outputs/group, K==2 mod 4 needs >=2, and K
    // divisible by four can use any power-of-two group size.
    const uint32_t alignment_quantum = (inner_size % 4 == 0) ? 1u : ((inner_size % 2 == 0) ? 2u : 4u);
    uint32_t outputs = 32;
    while (outputs > max_outputs) {
        outputs >>= 1;
    }
    while (outputs > 1 && (outputs % alignment_quantum) != 0) {
        outputs >>= 1;
    }
    return outputs >= 2 ? outputs : 1;
}


template <int PacketBytes>
[[nodiscard]] inline uint32_t shallowFp8CompactPackedOutputsPerWarp(uint64_t reduction_size, uint64_t inner_size) {
    static_assert(PacketBytes == 8 || PacketBytes == 16);
    if (reduction_size < 2 || inner_size == 0) {
        return 1;
    }
    const uint64_t bytes_per_output = reduction_size * inner_size;
    if (bytes_per_output == 0 || bytes_per_output > static_cast<uint64_t>(narrowFp8MaxLogicalGroupBytes<PacketBytes>)) {
        return 1;
    }
    const uint32_t max_outputs = static_cast<uint32_t>(std::min<uint64_t>(
        TILED_REDUCTION_WARP_THREADS,
        static_cast<uint64_t>(narrowFp8MaxLogicalGroupBytes<PacketBytes>) / bytes_per_output));

    // Input groups may start at any byte alignment for p8/p16; the packed kernel reserves PacketBytes-1 prefix bytes
    // and aligns the producer window down. Keep only the output group p4 aligned so full groups retain packed p4
    // writeback. As with p4 Compact, powers of two keep ownership compile-time after dispatch.
    const uint32_t output_alignment_quantum = (inner_size % 4 == 0) ? 1u : ((inner_size % 2 == 0) ? 2u : 4u);
    uint32_t outputs = 32;
    while (outputs > max_outputs) {
        outputs >>= 1;
    }
    while (outputs > 1 && (outputs % output_alignment_quantum) != 0) {
        outputs >>= 1;
    }
    return outputs >= 2 ? outputs : 1;
}

template <typename InputT, int K, int OutputsPerWarp>
__global__ void fp8CompactMultiOutputWarpP4CompleteKernel(const InputT* input,
                                                          InputT* output,
                                                          uint64_t outer_size,
                                                          uint32_t reduction_size) {
    static_assert(sizeof(InputT) == 1);
    static_assert(K >= 2 && K <= SHALLOW_FP8_MAX_K);
    static_assert(OutputsPerWarp == 2 || OutputsPerWarp == 4 || OutputsPerWarp == 8 || OutputsPerWarp == 16 || OutputsPerWarp == 32);
    static_assert((OutputsPerWarp * K) % SHALLOW_FP8_PACKET_BYTES == 0);
    static_assert(OutputsPerWarp * K <= 64);  // R>=2 and one 128-byte input window.

    __shared__ float staged_values[SHALLOW_FP8_WARPS][SHALLOW_FP8_PACKET_WINDOW_BYTES];
    __shared__ float lane_partials[SHALLOW_FP8_WARPS][TILED_REDUCTION_WARP_THREADS];
    __shared__ float writeback_values[SHALLOW_FP8_WARPS][64];

    const int warp = static_cast<int>(threadIdx.x) / TILED_REDUCTION_WARP_THREADS;
    const int lane = static_cast<int>(threadIdx.x) % TILED_REDUCTION_WARP_THREADS;
    constexpr int FullOutputComponents = OutputsPerWarp * K;
    constexpr int FullOutputPackets = FullOutputComponents / SHALLOW_FP8_PACKET_BYTES;
    constexpr int WorkersPerComponent =
        FullOutputComponents < TILED_REDUCTION_WARP_THREADS ? TILED_REDUCTION_WARP_THREADS / FullOutputComponents : 1;
    constexpr int FullTaskCount = FullOutputComponents * WorkersPerComponent;
    const uint64_t grid_stride = static_cast<uint64_t>(gridDim.x);

    for (uint64_t block_round = static_cast<uint64_t>(blockIdx.x);; block_round += grid_stride) {
        const uint64_t warp_group_index = block_round * static_cast<uint64_t>(SHALLOW_FP8_WARPS) + static_cast<uint64_t>(warp);
        const uint64_t outer_begin = warp_group_index * static_cast<uint64_t>(OutputsPerWarp);
        if (outer_begin >= outer_size) {
            return;
        }
        const uint32_t actual_outputs = static_cast<uint32_t>(minU64(static_cast<uint64_t>(OutputsPerWarp), outer_size - outer_begin));
        const uint32_t logical_input_elements = actual_outputs * reduction_size * static_cast<uint32_t>(K);
        const uint32_t packet_count = (logical_input_elements + SHALLOW_FP8_PACKET_BYTES - 1) / SHALLOW_FP8_PACKET_BYTES;
        const InputT* group_begin = input + outer_begin * static_cast<uint64_t>(reduction_size) * K;

        // Full groups are p4 aligned by construction. The final short group starts at the same aligned stride and may
        // overread only its final packet; Thor's 128-byte tensor tail padding makes that safe.
        if (lane < static_cast<int>(packet_count)) {
            const PackedInputValues<InputT, SHALLOW_FP8_PACKET_BYTES> values = loadVectorizedInputPacket<InputT, SHALLOW_FP8_PACKET_BYTES>(
                group_begin + static_cast<uint64_t>(lane) * SHALLOW_FP8_PACKET_BYTES);
            for (int item = 0; item < SHALLOW_FP8_PACKET_BYTES; ++item) {
                const uint32_t logical_index = static_cast<uint32_t>(lane * SHALLOW_FP8_PACKET_BYTES + item);
                if (logical_index < logical_input_elements) {
                    staged_values[warp][logical_index] = ToFp32<InputT>{}(values.values[item]);
                }
            }
        }
        __syncwarp();

        if (actual_outputs == OutputsPerWarp) {
            if constexpr (FullOutputComponents >= TILED_REDUCTION_WARP_THREADS) {
                for (int output_component = lane; output_component < FullOutputComponents;
                     output_component += TILED_REDUCTION_WARP_THREADS) {
                    const int output_slot = output_component / K;
                    const int component = output_component % K;
                    float reduced = 0.0f;
                    const uint32_t output_offset = static_cast<uint32_t>(output_slot) * reduction_size * static_cast<uint32_t>(K);
                    for (uint32_t row = 0; row < reduction_size; ++row) {
                        reduced += staged_values[warp][output_offset + row * K + component];
                    }
                    writeback_values[warp][output_component] = reduced;
                }
            } else {
                float local = 0.0f;
                if (lane < FullTaskCount) {
                    constexpr int W = WorkersPerComponent;
                    const int output_component = lane / W;
                    const int worker = lane % W;
                    const int output_slot = output_component / K;
                    const int component = output_component % K;
                    const uint32_t output_offset = static_cast<uint32_t>(output_slot) * reduction_size * static_cast<uint32_t>(K);
                    for (uint32_t row = static_cast<uint32_t>(worker); row < reduction_size; row += W) {
                        local += staged_values[warp][output_offset + row * K + component];
                    }
                }
                lane_partials[warp][lane] = local;
                __syncwarp();
                if (lane < FullOutputComponents) {
                    float reduced = 0.0f;
                    for (int worker = 0; worker < WorkersPerComponent; ++worker) {
                        reduced += lane_partials[warp][lane * WorkersPerComponent + worker];
                    }
                    writeback_values[warp][lane] = reduced;
                }
            }
            __syncwarp();

            // OutputsPerWarp*K is guaranteed p4 aligned. One lane owns each contiguous four-byte output packet.
            // Convert four FP32 results as two native fp8x2 operations, matching the hardware conversion granularity.
            if (lane < FullOutputPackets) {
                const int component_offset = lane * SHALLOW_FP8_PACKET_BYTES;
                const int component_offset_float_4 = component_offset >> 2;
                float2 values[2];
                float4* warpWritebackValues = (float4*)writeback_values[warp];
                ((float4*)values)[0] = warpWritebackValues[component_offset_float_4];
                uint16_t outputReg[2];
                if constexpr (std::is_same_v<InputT, __nv_fp8_e4m3>) {
                    outputReg[0] = __nv_cvt_float2_to_fp8x2(values[0], __NV_SATFINITE, __NV_E4M3);
                    outputReg[1] = __nv_cvt_float2_to_fp8x2(values[1], __NV_SATFINITE, __NV_E4M3);
                } else {
                    outputReg[0] = __nv_cvt_float2_to_fp8x2(values[0], __NV_SATFINITE, __NV_E5M2);
                    outputReg[1] = __nv_cvt_float2_to_fp8x2(values[1], __NV_SATFINITE, __NV_E5M2);
                }
                uint32_t* destination = reinterpret_cast<uint32_t*>(output + outer_begin * static_cast<uint64_t>(K));
                destination[lane] = ((uint32_t*)outputReg)[0];
            }
            // if (lane < FullOutputPackets) {
            //     const int component_offset = lane * SHALLOW_FP8_PACKET_BYTES;
            //     const float2 low_values = make_float2(
            //         writeback_values[warp][component_offset],
            //         writeback_values[warp][component_offset + 1]);
            //     const float2 high_values = make_float2(
            //         writeback_values[warp][component_offset + 2],
            //         writeback_values[warp][component_offset + 3]);
            //     __nv_fp8x2_storage_t low_bits;
            //     __nv_fp8x2_storage_t high_bits;
            //     if constexpr (std::is_same_v<InputT, __nv_fp8_e4m3>) {
            //         low_bits = __nv_cvt_float2_to_fp8x2(low_values, __NV_SATFINITE, __NV_E4M3);
            //         high_bits = __nv_cvt_float2_to_fp8x2(high_values, __NV_SATFINITE, __NV_E4M3);
            //     } else {
            //         low_bits = __nv_cvt_float2_to_fp8x2(low_values, __NV_SATFINITE, __NV_E5M2);
            //         high_bits = __nv_cvt_float2_to_fp8x2(high_values, __NV_SATFINITE, __NV_E5M2);
            //     }
            //     RawVectorPacket<SHALLOW_FP8_PACKET_BYTES> packet{};
            //     packet.value = static_cast<uint32_t>(low_bits)
            //         | (static_cast<uint32_t>(high_bits) << 16);
            //     auto* destination = reinterpret_cast<RawVectorPacket<SHALLOW_FP8_PACKET_BYTES>*>(
            //         output + outer_begin * static_cast<uint64_t>(K));
            //     destination[lane] = packet;
            // }
            __syncwarp();
            continue;
        }

        // Only the final outer group can be short. Keep its slow path simple; its cost is negligible in the large-work
        // calibration shapes and avoids burdening the steady-state full-group loop with dynamic division/modulo.
        for (uint32_t output_slot = 0; output_slot < actual_outputs; ++output_slot) {
            if (lane < K) {
                float reduced = 0.0f;
                const uint32_t output_offset = output_slot * reduction_size * static_cast<uint32_t>(K);
                for (uint32_t row = 0; row < reduction_size; ++row) {
                    reduced += staged_values[warp][output_offset + row * K + lane];
                }
                output[(outer_begin + output_slot) * static_cast<uint64_t>(K) + static_cast<uint64_t>(lane)] =
                    ThorLowPrecision::castToStorage<InputT>(reduced);
            }
        }
        __syncwarp();
    }
}


// Wide-packet CompactMultiOutput keeps Compact's one contiguous multi-output warp window, but stages FP8 bytes in
// packed shared memory instead of expanding every input byte to FP32. One p8/p16 packet is loaded per producer lane
// and written as two/four 32-bit shared planes. Component owners then gather up to four same-component R values at a
// time and use native packed FP8x4 decode. R remains runtime; PacketBytes and OutputsPerWarp are physical calibration
// dimensions, not K x R specializations.
template <typename InputT, int K, int OutputsPerWarp, int PacketBytes>
__global__ void fp8CompactMultiOutputWarpPackedCompleteKernel(const InputT* input,
                                                               InputT* output,
                                                               uint32_t outer_size,
                                                               uint32_t reduction_size) {
    static_assert(sizeof(InputT) == 1);
    static_assert(K >= 2 && K <= SHALLOW_FP8_MAX_K);
    static_assert(PacketBytes == 8 || PacketBytes == 16);
    static_assert(OutputsPerWarp == 2 || OutputsPerWarp == 4 || OutputsPerWarp == 8 || OutputsPerWarp == 16 || OutputsPerWarp == 32);
    constexpr uint32_t PacketWords = PacketBytes / static_cast<int>(sizeof(uint32_t));
    constexpr uint32_t PacketShift = PacketBytes == 8 ? 3 : 4;
    constexpr uint32_t PacketMask = PacketBytes - 1;
    constexpr int FullOutputComponents = OutputsPerWarp * K;
    constexpr int FullOutputPackets = FullOutputComponents / SHALLOW_FP8_PACKET_BYTES;
    constexpr int WorkersPerComponent =
        FullOutputComponents < TILED_REDUCTION_WARP_THREADS ? TILED_REDUCTION_WARP_THREADS / FullOutputComponents : 1;
    constexpr int FullTaskCount = FullOutputComponents * WorkersPerComponent;
    static_assert(PacketWords == 2 || PacketWords == 4);
    static_assert((FullOutputComponents % SHALLOW_FP8_PACKET_BYTES) == 0);
    // Compact calibration starts at R=2. Reserving PacketBytes-1 prefix bytes leaves these exact maximum logical
    // windows: p8=249 bytes and p16=497 bytes.
    static_assert(FullOutputComponents <= narrowFp8MaxLogicalGroupBytes<PacketBytes> / 2);

    __shared__ uint32_t staged_packet_words[SHALLOW_FP8_WARPS]
                                            [PacketWords * TILED_REDUCTION_WARP_THREADS];
    __shared__ float lane_partials[SHALLOW_FP8_WARPS][TILED_REDUCTION_WARP_THREADS];
    __shared__ float writeback_values[SHALLOW_FP8_WARPS][FullOutputComponents];

    const uint32_t warp = threadIdx.x / TILED_REDUCTION_WARP_THREADS;
    const uint32_t lane = threadIdx.x % TILED_REDUCTION_WARP_THREADS;
    const uint64_t grid_stride = static_cast<uint64_t>(gridDim.x);
    const uint32_t outer_input_stride = reduction_size * static_cast<uint32_t>(K);
    const uint32_t total_input_bytes = outer_size * outer_input_stride;

    for (uint64_t block_round = static_cast<uint64_t>(blockIdx.x);; block_round += grid_stride) {
        const uint64_t warp_group_index =
            block_round * static_cast<uint64_t>(SHALLOW_FP8_WARPS) + static_cast<uint64_t>(warp);
        const uint64_t outer_begin = warp_group_index * static_cast<uint64_t>(OutputsPerWarp);
        if (outer_begin >= outer_size) {
            return;
        }
        const uint32_t actual_outputs = static_cast<uint32_t>(
            minU64(static_cast<uint64_t>(OutputsPerWarp), static_cast<uint64_t>(outer_size) - outer_begin));
        const uint32_t logical_input_bytes = actual_outputs * outer_input_stride;
        const uint32_t group_byte_begin = static_cast<uint32_t>(outer_begin) * outer_input_stride;
        const uint32_t prefix_bytes = group_byte_begin & PacketMask;
        const uint32_t aligned_byte_begin = group_byte_begin - prefix_bytes;
        const uint32_t packet_count = (prefix_bytes + logical_input_bytes + PacketBytes - 1) >> PacketShift;

        if (lane < packet_count) {
            const uint32_t packet_byte_begin = aligned_byte_begin + lane * PacketBytes;
            RawVectorPacket<PacketBytes> packet{};
            if (total_input_bytes >= static_cast<uint32_t>(PacketBytes) &&
                packet_byte_begin <= total_input_bytes - static_cast<uint32_t>(PacketBytes)) {
                packet = *reinterpret_cast<const RawVectorPacket<PacketBytes>*>(
                    reinterpret_cast<const uint8_t*>(input) + packet_byte_begin);
            } else {
                // Only the final tensor packet can reach this path. The steady-state full groups remain one native
                // p8/p16 load per active producer lane.
                auto* packet_bytes = reinterpret_cast<uint8_t*>(&packet.value);
#pragma unroll
                for (uint32_t byte = 0; byte < PacketBytes; ++byte) {
                    const uint32_t byte_index = packet_byte_begin + byte;
                    if (byte_index < total_input_bytes) {
                        packet_bytes[byte] = reinterpret_cast<const uint8_t*>(input)[byte_index];
                    }
                }
            }
            if constexpr (PacketBytes == 8) {
                staged_packet_words[warp][0 * TILED_REDUCTION_WARP_THREADS + lane] = packet.value.x;
                staged_packet_words[warp][1 * TILED_REDUCTION_WARP_THREADS + lane] = packet.value.y;
            } else {
                staged_packet_words[warp][0 * TILED_REDUCTION_WARP_THREADS + lane] = packet.value.x;
                staged_packet_words[warp][1 * TILED_REDUCTION_WARP_THREADS + lane] = packet.value.y;
                staged_packet_words[warp][2 * TILED_REDUCTION_WARP_THREADS + lane] = packet.value.z;
                staged_packet_words[warp][3 * TILED_REDUCTION_WARP_THREADS + lane] = packet.value.w;
            }
        }
        __syncwarp();

        if (actual_outputs == OutputsPerWarp) {
            if constexpr (FullOutputComponents >= TILED_REDUCTION_WARP_THREADS) {
                for (int output_component = static_cast<int>(lane); output_component < FullOutputComponents;
                     output_component += TILED_REDUCTION_WARP_THREADS) {
                    const int output_slot = output_component / K;
                    const int component = output_component % K;
                    const uint32_t output_offset =
                        static_cast<uint32_t>(output_slot) * reduction_size * static_cast<uint32_t>(K);
                    float reduced = 0.0f;
                    for (uint32_t row = 0; row < reduction_size; row += 4) {
                        uint32_t packed_component_values = 0;
                        uint32_t valid_values = 0;
#pragma unroll
                        for (uint32_t item = 0; item < 4; ++item) {
                            const uint32_t source_row = row + item;
                            if (source_row < reduction_size) {
                                const uint32_t logical_byte = output_offset + source_row * K + component;
                                const uint32_t staged_byte = prefix_bytes + logical_byte;
                                const uint32_t source_packet = staged_byte >> PacketShift;
                                const uint32_t packet_byte = staged_byte & PacketMask;
                                const uint32_t source_word = staged_packet_words[warp]
                                    [(packet_byte >> 2) * TILED_REDUCTION_WARP_THREADS + source_packet];
                                packed_component_values |=
                                    ((source_word >> ((packet_byte & 3U) * 8U)) & 0xffU) << (item * 8U);
                                ++valid_values;
                            }
                        }
                        const float4 decoded = decodeFp8x4ToFp32<InputT>(packed_component_values);
                        if (valid_values > 0) reduced += decoded.x;
                        if (valid_values > 1) reduced += decoded.y;
                        if (valid_values > 2) reduced += decoded.z;
                        if (valid_values > 3) reduced += decoded.w;
                    }
                    writeback_values[warp][output_component] = reduced;
                }
            } else {
                float local = 0.0f;
                if (lane < FullTaskCount) {
                    constexpr int W = WorkersPerComponent;
                    const int output_component = static_cast<int>(lane) / W;
                    const int worker = static_cast<int>(lane) % W;
                    const int output_slot = output_component / K;
                    const int component = output_component % K;
                    const uint32_t output_offset =
                        static_cast<uint32_t>(output_slot) * reduction_size * static_cast<uint32_t>(K);
                    for (uint32_t row = static_cast<uint32_t>(worker); row < reduction_size; row += W * 4) {
                        uint32_t packed_component_values = 0;
                        uint32_t valid_values = 0;
#pragma unroll
                        for (uint32_t item = 0; item < 4; ++item) {
                            const uint32_t source_row = row + item * W;
                            if (source_row < reduction_size) {
                                const uint32_t logical_byte = output_offset + source_row * K + component;
                                const uint32_t staged_byte = prefix_bytes + logical_byte;
                                const uint32_t source_packet = staged_byte >> PacketShift;
                                const uint32_t packet_byte = staged_byte & PacketMask;
                                const uint32_t source_word = staged_packet_words[warp]
                                    [(packet_byte >> 2) * TILED_REDUCTION_WARP_THREADS + source_packet];
                                packed_component_values |=
                                    ((source_word >> ((packet_byte & 3U) * 8U)) & 0xffU) << (item * 8U);
                                ++valid_values;
                            }
                        }
                        const float4 decoded = decodeFp8x4ToFp32<InputT>(packed_component_values);
                        if (valid_values > 0) local += decoded.x;
                        if (valid_values > 1) local += decoded.y;
                        if (valid_values > 2) local += decoded.z;
                        if (valid_values > 3) local += decoded.w;
                    }
                }
                lane_partials[warp][lane] = local;
                __syncwarp();
                if (lane < FullOutputComponents) {
                    float reduced = 0.0f;
                    for (int source_worker = 0; source_worker < WorkersPerComponent; ++source_worker) {
                        reduced += lane_partials[warp][lane * WorkersPerComponent + source_worker];
                    }
                    writeback_values[warp][lane] = reduced;
                }
            }
            __syncwarp();

            // Compact's output grouping remains p4 aligned even when the input packet is p8/p16. More than 128
            // output bytes simply give each lane a second packed p4 store.
            for (int output_packet = static_cast<int>(lane); output_packet < FullOutputPackets;
                 output_packet += TILED_REDUCTION_WARP_THREADS) {
                const int component_offset = output_packet * SHALLOW_FP8_PACKET_BYTES;
                const int component_offset_float_4 = component_offset >> 2;
                float2 values[2];
                const float4* warp_writeback_values = reinterpret_cast<const float4*>(writeback_values[warp]);
                reinterpret_cast<float4*>(values)[0] = warp_writeback_values[component_offset_float_4];
                uint16_t output_reg[2];
                if constexpr (std::is_same_v<InputT, __nv_fp8_e4m3>) {
                    output_reg[0] = __nv_cvt_float2_to_fp8x2(values[0], __NV_SATFINITE, __NV_E4M3);
                    output_reg[1] = __nv_cvt_float2_to_fp8x2(values[1], __NV_SATFINITE, __NV_E4M3);
                } else {
                    output_reg[0] = __nv_cvt_float2_to_fp8x2(values[0], __NV_SATFINITE, __NV_E5M2);
                    output_reg[1] = __nv_cvt_float2_to_fp8x2(values[1], __NV_SATFINITE, __NV_E5M2);
                }
                auto* destination = reinterpret_cast<uint32_t*>(
                    output + outer_begin * static_cast<uint64_t>(K));
                destination[output_packet] = reinterpret_cast<uint32_t*>(output_reg)[0];
            }
            __syncwarp();
            continue;
        }

        // Only the final outer group can be short. Keep it scalar on output; packed input staging is still reused so
        // the steady-state kernel does not carry a second producer path.
        for (uint32_t output_slot = 0; output_slot < actual_outputs; ++output_slot) {
            if (lane < K) {
                float reduced = 0.0f;
                const uint32_t output_offset = output_slot * reduction_size * static_cast<uint32_t>(K);
                for (uint32_t row = 0; row < reduction_size; row += 4) {
                    uint32_t packed_component_values = 0;
                    uint32_t valid_values = 0;
#pragma unroll
                    for (uint32_t item = 0; item < 4; ++item) {
                        const uint32_t source_row = row + item;
                        if (source_row < reduction_size) {
                            const uint32_t logical_byte = output_offset + source_row * K + lane;
                            const uint32_t staged_byte = prefix_bytes + logical_byte;
                            const uint32_t source_packet = staged_byte >> PacketShift;
                            const uint32_t packet_byte = staged_byte & PacketMask;
                            const uint32_t source_word = staged_packet_words[warp]
                                [(packet_byte >> 2) * TILED_REDUCTION_WARP_THREADS + source_packet];
                            packed_component_values |=
                                ((source_word >> ((packet_byte & 3U) * 8U)) & 0xffU) << (item * 8U);
                            ++valid_values;
                        }
                    }
                    const float4 decoded = decodeFp8x4ToFp32<InputT>(packed_component_values);
                    if (valid_values > 0) reduced += decoded.x;
                    if (valid_values > 1) reduced += decoded.y;
                    if (valid_values > 2) reduced += decoded.z;
                    if (valid_values > 3) reduced += decoded.w;
                }
                output[(outer_begin + output_slot) * static_cast<uint64_t>(K) + lane] =
                    ThorLowPrecision::castToStorage<InputT>(reduced);
            }
        }
        __syncwarp();
    }
}

template <typename InputT, int K, bool P4Aligned>
__global__ void fp8RParallelWarpP4CompleteKernel(const InputT* input, InputT* output, uint32_t outer_size, uint32_t reduction_size) {
    static_assert(sizeof(InputT) == 1);
    static_assert(K >= 2 && K <= SHALLOW_FP8_MAX_K);
    static_assert(!P4Aligned || K % SHALLOW_FP8_PACKET_BYTES == 0);
    constexpr uint32_t WorkersPerComponent = TILED_REDUCTION_WARP_THREADS / K;
    constexpr uint32_t TaskCount = K * WorkersPerComponent;
    constexpr uint32_t RowsPerGroup = (P4Aligned ? SHALLOW_FP8_PACKET_WINDOW_BYTES : SHALLOW_FP8_MAX_UNALIGNED_LOGICAL_BYTES) / K;
    static_assert(WorkersPerComponent >= 1);
    static_assert(RowsPerGroup >= 1);
    static_assert(!P4Aligned || RowsPerGroup * K == SHALLOW_FP8_PACKET_WINDOW_BYTES);

    // The aligned specialization keeps the ownership remap in packed FP8 shared memory. The generic awkward-K path
    // retains the existing FP32 staging. Compile-time extents avoid carrying both full staging footprints.
    __shared__ uint32_t staged_packets[P4Aligned ? SHALLOW_FP8_WARPS : 1][P4Aligned ? TILED_REDUCTION_WARP_THREADS : 1];
    __shared__ float staged_values[P4Aligned ? 1 : SHALLOW_FP8_WARPS][P4Aligned ? 1 : SHALLOW_FP8_PACKET_WINDOW_BYTES];
    __shared__ float lane_partials[SHALLOW_FP8_WARPS][TILED_REDUCTION_WARP_THREADS];

    const uint32_t warp = threadIdx.x / TILED_REDUCTION_WARP_THREADS;
    const uint32_t lane = threadIdx.x % TILED_REDUCTION_WARP_THREADS;
    const uint32_t component = lane / WorkersPerComponent;
    const uint32_t worker = lane % WorkersPerComponent;
    const uint32_t global_warp = blockIdx.x * SHALLOW_FP8_WARPS + warp;
    const uint32_t grid_warps = gridDim.x * SHALLOW_FP8_WARPS;
    const uint32_t outer_input_stride = reduction_size * K;

    for (uint32_t outer_index = global_warp; outer_index < outer_size; outer_index += grid_warps) {
        const InputT* outer_input = input + outer_index * outer_input_stride;
        InputT* outer_output = output + outer_index * K;
        float local = 0.0f;

        if constexpr (P4Aligned) {
            // K=4/8/16/32: the output stride, row stride, and every 128-byte full group are naturally p4 aligned.
            // Keep the ownership remap in the native representation: 32 coalesced p4 loads become 32 raw p4 shared
            // stores. Each consumer lane then gathers its same component byte from four broadcast-friendly shared
            // words and performs two packed FP8x2 decodes immediately before FP32 accumulation.
            static_assert(RowsPerGroup / WorkersPerComponent == SHALLOW_FP8_PACKET_BYTES);
            constexpr uint32_t ComponentPacketsPerRow = K / SHALLOW_FP8_PACKET_BYTES;
            constexpr uint32_t ConsumerPacketStride = WorkersPerComponent * ComponentPacketsPerRow;
            static_assert(ConsumerPacketStride == 8);
            const uint32_t component_packet = component / SHALLOW_FP8_PACKET_BYTES;
            const uint32_t component_shift = (component % SHALLOW_FP8_PACKET_BYTES) * 8;

            const InputT* group_begin = outer_input;
            uint32_t remaining_rows = reduction_size;
            while (remaining_rows >= RowsPerGroup) {
                const auto* source_packets = reinterpret_cast<const RawVectorPacket<SHALLOW_FP8_PACKET_BYTES>*>(group_begin);
                staged_packets[warp][lane] = source_packets[lane].value;
                __syncwarp();

                const uint32_t first_packet = worker * ComponentPacketsPerRow + component_packet;
                uint32_t packed_component_values = 0;
                for (uint32_t item = 0; item < SHALLOW_FP8_PACKET_BYTES; ++item) {
                    const uint32_t source_word = staged_packets[warp][first_packet + item * ConsumerPacketStride];
                    packed_component_values |= ((source_word >> component_shift) & 0xffU) << (item * 8);
                }
                const float4 decoded = decodeFp8x4ToFp32<InputT>(packed_component_values);
                local += decoded.x;
                local += decoded.y;
                local += decoded.z;
                local += decoded.w;
                __syncwarp();

                group_begin += SHALLOW_FP8_PACKET_WINDOW_BYTES;
                remaining_rows -= RowsPerGroup;
            }

            // At most one tail group. K is p4 aligned, so every loaded p4 packet is entirely valid. Keep the same raw
            // shared-memory remap and zero-fill absent rows before the two packed conversions.
            if (remaining_rows != 0) {
                const uint32_t packet_count = remaining_rows * ComponentPacketsPerRow;
                if (lane < packet_count) {
                    const auto* source_packets = reinterpret_cast<const RawVectorPacket<SHALLOW_FP8_PACKET_BYTES>*>(group_begin);
                    staged_packets[warp][lane] = source_packets[lane].value;
                }
                __syncwarp();

                const uint32_t first_packet = worker * ComponentPacketsPerRow + component_packet;
                uint32_t packed_component_values = 0;
                uint32_t valid_values = 0;
                for (uint32_t item = 0; item < SHALLOW_FP8_PACKET_BYTES; ++item) {
                    const uint32_t row = worker + item * WorkersPerComponent;
                    if (row < remaining_rows) {
                        const uint32_t source_word = staged_packets[warp][first_packet + item * ConsumerPacketStride];
                        packed_component_values |= ((source_word >> component_shift) & 0xffU) << (item * 8);
                        ++valid_values;
                    }
                }
                const float4 decoded = decodeFp8x4ToFp32<InputT>(packed_component_values);
                if (valid_values > 0)
                    local += decoded.x;
                if (valid_values > 1)
                    local += decoded.y;
                if (valid_values > 2)
                    local += decoded.z;
                if (valid_values > 3)
                    local += decoded.w;
                __syncwarp();
            }
        } else {
            // Generic calibration fallback for supported non-p4 K, and for an unusual unaligned tensor-view base.
            for (uint32_t group_row = 0; group_row < reduction_size; group_row += RowsPerGroup) {
                const uint32_t remaining_rows = reduction_size - group_row;
                const uint32_t group_rows = remaining_rows < RowsPerGroup ? remaining_rows : RowsPerGroup;
                const InputT* group_begin = outer_input + group_row * K;
                const uintptr_t group_address = reinterpret_cast<uintptr_t>(group_begin);
                const uint32_t prefix_elements = static_cast<uint32_t>(group_address & uintptr_t{SHALLOW_FP8_PACKET_BYTES - 1});
                const InputT* aligned_begin = reinterpret_cast<const InputT*>(group_address & ~uintptr_t{SHALLOW_FP8_PACKET_BYTES - 1});
                const uint32_t logical_elements = group_rows * K;
                const uint32_t packet_count =
                    (prefix_elements + logical_elements + SHALLOW_FP8_PACKET_BYTES - 1) / SHALLOW_FP8_PACKET_BYTES;

                if (lane < packet_count) {
                    const PackedInputValues<InputT, SHALLOW_FP8_PACKET_BYTES> values =
                        loadVectorizedInputPacket<InputT, SHALLOW_FP8_PACKET_BYTES>(aligned_begin + lane * SHALLOW_FP8_PACKET_BYTES);
                    for (uint32_t item = 0; item < SHALLOW_FP8_PACKET_BYTES; ++item) {
                        const int32_t logical_index =
                            static_cast<int32_t>(lane * SHALLOW_FP8_PACKET_BYTES + item) - static_cast<int32_t>(prefix_elements);
                        if (logical_index >= 0 && logical_index < static_cast<int32_t>(logical_elements)) {
                            staged_values[warp][logical_index] = ToFp32<InputT>{}(values.values[item]);
                        }
                    }
                }
                __syncwarp();

                if constexpr (TaskCount == TILED_REDUCTION_WARP_THREADS) {
                    for (uint32_t row = worker; row < group_rows; row += WorkersPerComponent) {
                        local += staged_values[warp][row * K + component];
                    }
                } else if (lane < TaskCount) {
                    for (uint32_t row = worker; row < group_rows; row += WorkersPerComponent) {
                        local += staged_values[warp][row * K + component];
                    }
                }
                __syncwarp();
            }
        }

        lane_partials[warp][lane] = local;
        __syncwarp();
        if (lane < K) {
            float reduced = 0.0f;
            for (uint32_t source_worker = 0; source_worker < WorkersPerComponent; ++source_worker) {
                reduced += lane_partials[warp][lane * WorkersPerComponent + source_worker];
            }
            if constexpr (P4Aligned) {
                // The worker partials are dead once the fold completes. Reuse their first K slots for contiguous FP32
                // writeback values so p4 owners can narrow four results with two native fp8x2 conversions.
                lane_partials[warp][lane] = reduced;
            } else {
                outer_output[lane] = ThorLowPrecision::castToStorage<InputT>(reduced);
            }
        }
        if constexpr (P4Aligned) {
            __syncwarp();
            if (lane < K / SHALLOW_FP8_PACKET_BYTES) {
                const int component_offset = lane * SHALLOW_FP8_PACKET_BYTES;
                const int component_offset_float_4 = component_offset >> 2;
                float2 values[2];
                float4* lanePartialsWarp = (float4*)lane_partials[warp];
                ((float4*)values)[0] = lanePartialsWarp[component_offset_float_4];
                uint16_t outputReg[2];
                if constexpr (std::is_same_v<InputT, __nv_fp8_e4m3>) {
                    outputReg[0] = __nv_cvt_float2_to_fp8x2(values[0], __NV_SATFINITE, __NV_E4M3);
                    outputReg[1] = __nv_cvt_float2_to_fp8x2(values[1], __NV_SATFINITE, __NV_E4M3);
                } else {
                    outputReg[0] = __nv_cvt_float2_to_fp8x2(values[0], __NV_SATFINITE, __NV_E5M2);
                    outputReg[1] = __nv_cvt_float2_to_fp8x2(values[1], __NV_SATFINITE, __NV_E5M2);
                }
                uint32_t* destination = reinterpret_cast<uint32_t*>(outer_output);
                destination[lane] = ((uint32_t*)outputReg)[0];
            }

            // if (lane < K / SHALLOW_FP8_PACKET_BYTES) {
            //     const uint32_t component_offset = lane * SHALLOW_FP8_PACKET_BYTES;
            //     const float2 low_values = make_float2(
            //         lane_partials[warp][component_offset],
            //         lane_partials[warp][component_offset + 1]);
            //     const float2 high_values = make_float2(
            //         lane_partials[warp][component_offset + 2],
            //         lane_partials[warp][component_offset + 3]);
            //     __nv_fp8x2_storage_t low_bits;
            //     __nv_fp8x2_storage_t high_bits;
            //     if constexpr (std::is_same_v<InputT, __nv_fp8_e4m3>) {
            //         low_bits = __nv_cvt_float2_to_fp8x2(low_values, __NV_SATFINITE, __NV_E4M3);
            //         high_bits = __nv_cvt_float2_to_fp8x2(high_values, __NV_SATFINITE, __NV_E4M3);
            //     } else {
            //         low_bits = __nv_cvt_float2_to_fp8x2(low_values, __NV_SATFINITE, __NV_E5M2);
            //         high_bits = __nv_cvt_float2_to_fp8x2(high_values, __NV_SATFINITE, __NV_E5M2);
            //     }
            //     RawVectorPacket<SHALLOW_FP8_PACKET_BYTES> packet{};
            //     packet.value = static_cast<uint32_t>(low_bits)
            //         | (static_cast<uint32_t>(high_bits) << 16);
            //     auto* destination =
            //         reinterpret_cast<RawVectorPacket<SHALLOW_FP8_PACKET_BYTES>*>(outer_output);
            //     destination[lane] = packet;
            // }
        }
        __syncwarp();
    }
}


template <typename InputT, int K, int OutputsPerWarp>
__global__ void fp8MultiOutputRParallelWarpP4CompleteKernel(const InputT* input,
                                                            InputT* output,
                                                            uint32_t outer_size,
                                                            uint32_t reduction_size) {
    static_assert(sizeof(InputT) == 1);
    static_assert(K >= 1 && K <= SHALLOW_FP8_MAX_K);
    static_assert(OutputsPerWarp == 2 || OutputsPerWarp == 4 || OutputsPerWarp == 8 ||
                  OutputsPerWarp == 16 || OutputsPerWarp == 32);
    constexpr uint32_t LanesPerOutput = TILED_REDUCTION_WARP_THREADS / OutputsPerWarp;
    static_assert(LanesPerOutput * OutputsPerWarp == TILED_REDUCTION_WARP_THREADS);
    // Preserve the existing RParallel ownership exactly: every retained K component has at least one lane, and
    // otherwise-idle lanes become additional R workers for that component. Shapes narrower than K would be a
    // different ownership model (DirectComponent-like), so they are deliberately not represented here.
    static_assert(K <= static_cast<int>(LanesPerOutput));
    constexpr uint32_t WorkersPerComponent = LanesPerOutput / K;
    constexpr uint32_t TaskCountPerOutput = K * WorkersPerComponent;
    constexpr uint32_t StagingValuesPerOutput = LanesPerOutput * SHALLOW_FP8_PACKET_BYTES;
    constexpr uint32_t MaxLogicalValuesPerGroup = StagingValuesPerOutput - (SHALLOW_FP8_PACKET_BYTES - 1);
    constexpr uint32_t RowsPerGroup = MaxLogicalValuesPerGroup / K;
    static_assert(WorkersPerComponent >= 1);
    static_assert(TaskCountPerOutput <= LanesPerOutput);
    static_assert(RowsPerGroup >= 1);
    static_assert(StagingValuesPerOutput * OutputsPerWarp == SHALLOW_FP8_PACKET_WINDOW_BYTES);

    // This is the existing generic RParallel transpose, partitioned into independent subwarp slices. Each subwarp
    // issues contiguous p4 loads for one output, decodes the packet bytes into its shared slice, and consumes that
    // slice with the same component/worker ownership as the one-output kernel. Total shared staging per warp remains
    // exactly one 128-value window; only ownership is subdivided.
    __shared__ float staged_values[SHALLOW_FP8_WARPS][SHALLOW_FP8_PACKET_WINDOW_BYTES];
    __shared__ float lane_partials[SHALLOW_FP8_WARPS][TILED_REDUCTION_WARP_THREADS];

    const uint32_t warp = threadIdx.x / TILED_REDUCTION_WARP_THREADS;
    const uint32_t lane = threadIdx.x % TILED_REDUCTION_WARP_THREADS;
    const uint32_t output_slot = lane / LanesPerOutput;
    const uint32_t subgroup_lane = lane % LanesPerOutput;
    const uint32_t component = subgroup_lane / WorkersPerComponent;
    const uint32_t worker = subgroup_lane % WorkersPerComponent;
    const uint32_t staging_base = output_slot * StagingValuesPerOutput;
    const uint32_t global_warp = blockIdx.x * SHALLOW_FP8_WARPS + warp;
    const uint32_t grid_warps = gridDim.x * SHALLOW_FP8_WARPS;
    const uint32_t outer_input_stride = reduction_size * K;

    for (uint32_t warp_group = global_warp;; warp_group += grid_warps) {
        const uint64_t outer_begin = static_cast<uint64_t>(warp_group) * OutputsPerWarp;
        if (outer_begin >= outer_size) {
            return;
        }
        const uint64_t outer_index = outer_begin + output_slot;
        const bool valid_output = outer_index < outer_size;
        const InputT* outer_input = valid_output
            ? input + outer_index * static_cast<uint64_t>(outer_input_stride)
            : input;
        float local = 0.0f;

        for (uint32_t group_row = 0; group_row < reduction_size; group_row += RowsPerGroup) {
            const uint32_t remaining_rows = reduction_size - group_row;
            const uint32_t group_rows = remaining_rows < RowsPerGroup ? remaining_rows : RowsPerGroup;
            uint32_t prefix_elements = 0;
            uint32_t packet_count = 0;
            const InputT* aligned_begin = outer_input;
            if (valid_output) {
                const InputT* group_begin = outer_input + static_cast<uint64_t>(group_row) * K;
                const uintptr_t group_address = reinterpret_cast<uintptr_t>(group_begin);
                prefix_elements = static_cast<uint32_t>(
                    group_address & uintptr_t{SHALLOW_FP8_PACKET_BYTES - 1});
                aligned_begin = reinterpret_cast<const InputT*>(
                    group_address & ~uintptr_t{SHALLOW_FP8_PACKET_BYTES - 1});
                const uint32_t logical_elements = group_rows * K;
                packet_count = (prefix_elements + logical_elements + SHALLOW_FP8_PACKET_BYTES - 1) /
                               SHALLOW_FP8_PACKET_BYTES;
            }

            if (valid_output && subgroup_lane < packet_count) {
                const PackedInputValues<InputT, SHALLOW_FP8_PACKET_BYTES> values =
                    loadVectorizedInputPacket<InputT, SHALLOW_FP8_PACKET_BYTES>(
                        aligned_begin + subgroup_lane * SHALLOW_FP8_PACKET_BYTES);
                const uint32_t logical_elements = group_rows * K;
                for (uint32_t item = 0; item < SHALLOW_FP8_PACKET_BYTES; ++item) {
                    const int32_t logical_index =
                        static_cast<int32_t>(subgroup_lane * SHALLOW_FP8_PACKET_BYTES + item) -
                        static_cast<int32_t>(prefix_elements);
                    if (logical_index >= 0 && logical_index < static_cast<int32_t>(logical_elements)) {
                        staged_values[warp][staging_base + static_cast<uint32_t>(logical_index)] =
                            ToFp32<InputT>{}(values.values[item]);
                    }
                }
            }
            __syncwarp();

            if (valid_output && subgroup_lane < TaskCountPerOutput) {
                for (uint32_t row = worker; row < group_rows; row += WorkersPerComponent) {
                    local += staged_values[warp][staging_base + row * K + component];
                }
            }
            // Keep the existing RParallel shared-transpose lifetime: every consumer is finished before any subgroup
            // reuses its staging slice for the next R group.
            __syncwarp();
        }

        lane_partials[warp][lane] = (valid_output && subgroup_lane < TaskCountPerOutput) ? local : 0.0f;
        __syncwarp();
        if (valid_output && subgroup_lane < K) {
            float reduced = 0.0f;
            const uint32_t partial_base = output_slot * LanesPerOutput + subgroup_lane * WorkersPerComponent;
            for (uint32_t source_worker = 0; source_worker < WorkersPerComponent; ++source_worker) {
                reduced += lane_partials[warp][partial_base + source_worker];
            }
            output[outer_index * static_cast<uint64_t>(K) + subgroup_lane] =
                ThorLowPrecision::castToStorage<InputT>(reduced);
        }
        __syncwarp();
    }
}


// Wide-packet MultiOutput RParallel keeps the same component/R-worker ownership as the established p4 kernel, but
// decouples global packet width from the shared-memory ownership transpose. Each lane loads one p8/p16 packet. The
// packet is then written to shared as 32-bit planes, so each plane store is a conflict-free warp-wide sequence of
// 4-byte words. Consumers reconstruct their component bytes from those planes, gather four R-worker values, and use
// the native packed FP8x4 decode immediately before FP32 accumulation. R remains runtime and K only determines the
// existing component-worker ownership; there is no K x R specialization here.
template <typename InputT, int K, int OutputsPerWarp, int PacketBytes>
__global__ void fp8MultiOutputRParallelWarpPackedCompleteKernel(const InputT* input,
                                                                InputT* output,
                                                                uint32_t outer_size,
                                                                uint32_t reduction_size) {
    static_assert(sizeof(InputT) == 1);
    static_assert(K >= 1 && K <= SHALLOW_FP8_MAX_K);
    static_assert(PacketBytes == 8 || PacketBytes == 16);
    static_assert(OutputsPerWarp == 1 || OutputsPerWarp == 2 || OutputsPerWarp == 4 || OutputsPerWarp == 8 ||
                  OutputsPerWarp == 16 || OutputsPerWarp == 32);
    constexpr uint32_t LanesPerOutput = TILED_REDUCTION_WARP_THREADS / OutputsPerWarp;
    constexpr uint32_t PacketWords = PacketBytes / static_cast<int>(sizeof(uint32_t));
    constexpr uint32_t PacketShift = PacketBytes == 8 ? 3 : 4;
    constexpr uint32_t PacketMask = PacketBytes - 1;
    static_assert(LanesPerOutput * OutputsPerWarp == TILED_REDUCTION_WARP_THREADS);
    static_assert(PacketWords == 2 || PacketWords == 4);
    static_assert(K <= static_cast<int>(LanesPerOutput));

    constexpr uint32_t WorkersPerComponent = LanesPerOutput / K;
    constexpr uint32_t TaskCountPerOutput = K * WorkersPerComponent;
    constexpr uint32_t StagingBytesPerOutput = LanesPerOutput * PacketBytes;
    constexpr uint32_t MaxLogicalBytesPerGroup = StagingBytesPerOutput - (PacketBytes - 1);
    constexpr uint32_t RowsPerGroup = MaxLogicalBytesPerGroup / K;
    static_assert(WorkersPerComponent >= 1);
    static_assert(TaskCountPerOutput <= LanesPerOutput);
    static_assert(RowsPerGroup >= 1);
    static_assert(StagingBytesPerOutput * OutputsPerWarp ==
                  static_cast<uint32_t>(PacketBytes * TILED_REDUCTION_WARP_THREADS));

    // Plane-major shared layout. For every plane, lane N writes one uint32_t to bank N. p8 therefore performs two
    // conflict-free 128-byte shared-store rounds per warp and p16 performs four. Keeping the bytes packed avoids the
    // 4x FP32 shared expansion in the original p4 MultiOutput path.
    __shared__ uint32_t staged_packet_words[SHALLOW_FP8_WARPS]
                                            [PacketWords * TILED_REDUCTION_WARP_THREADS];
    __shared__ float lane_partials[SHALLOW_FP8_WARPS][TILED_REDUCTION_WARP_THREADS];

    const uint32_t warp = threadIdx.x / TILED_REDUCTION_WARP_THREADS;
    const uint32_t lane = threadIdx.x % TILED_REDUCTION_WARP_THREADS;
    const uint32_t output_slot = lane / LanesPerOutput;
    const uint32_t subgroup_lane = lane % LanesPerOutput;
    const uint32_t component = subgroup_lane / WorkersPerComponent;
    const uint32_t worker = subgroup_lane % WorkersPerComponent;
    const uint32_t subgroup_lane_base = output_slot * LanesPerOutput;
    const uint32_t global_warp = blockIdx.x * SHALLOW_FP8_WARPS + warp;
    const uint32_t grid_warps = gridDim.x * SHALLOW_FP8_WARPS;
    const uint32_t outer_input_stride = reduction_size * K;

    for (uint32_t warp_group = global_warp;; warp_group += grid_warps) {
        const uint64_t outer_begin = static_cast<uint64_t>(warp_group) * OutputsPerWarp;
        if (outer_begin >= outer_size) {
            return;
        }
        const uint64_t outer_index = outer_begin + output_slot;
        const bool valid_output = outer_index < outer_size;
        float local = 0.0f;

        // The launch requires the tensor base to satisfy PacketBytes alignment. Individual outputs and R groups may
        // still be arbitrarily misaligned; align their byte offsets down inside the tensor so packet addressing stays
        // in 32-bit arithmetic and the common path remains one native p8/p16 load.
        const uint32_t total_input_bytes = outer_size * outer_input_stride;
        const uint32_t outer_byte_begin =
            valid_output ? static_cast<uint32_t>(outer_index) * outer_input_stride : 0;

        for (uint32_t group_row = 0; group_row < reduction_size; group_row += RowsPerGroup) {
            const uint32_t remaining_rows = reduction_size - group_row;
            const uint32_t group_rows = remaining_rows < RowsPerGroup ? remaining_rows : RowsPerGroup;
            uint32_t prefix_bytes = 0;
            uint32_t packet_count = 0;
            uint32_t aligned_byte_begin = 0;
            if (valid_output) {
                const uint32_t group_byte_begin = outer_byte_begin + group_row * K;
                prefix_bytes = group_byte_begin & PacketMask;
                aligned_byte_begin = group_byte_begin - prefix_bytes;
                const uint32_t logical_bytes = group_rows * K;
                packet_count = (prefix_bytes + logical_bytes + PacketBytes - 1) >> PacketShift;
            }

            if (valid_output && subgroup_lane < packet_count) {
                const uint32_t packet_byte_begin =
                    aligned_byte_begin + subgroup_lane * PacketBytes;
                RawVectorPacket<PacketBytes> packet{};
                if (total_input_bytes >= static_cast<uint32_t>(PacketBytes) &&
                    packet_byte_begin <= total_input_bytes - static_cast<uint32_t>(PacketBytes)) {
                    packet = *reinterpret_cast<const RawVectorPacket<PacketBytes>*>(
                        reinterpret_cast<const uint8_t*>(input) + packet_byte_begin);
                } else {
                    // Only the final tensor packet can reach this path. Keep the common producer as one native
                    // p8/p16 load while making the last partial packet fully bounds-safe.
                    auto* packet_bytes = reinterpret_cast<uint8_t*>(&packet.value);
#pragma unroll
                    for (uint32_t byte = 0; byte < PacketBytes; ++byte) {
                        const uint32_t byte_index = packet_byte_begin + byte;
                        if (byte_index < total_input_bytes) {
                            packet_bytes[byte] = reinterpret_cast<const uint8_t*>(input)[byte_index];
                        }
                    }
                }
                if constexpr (PacketBytes == 8) {
                    staged_packet_words[warp][0 * TILED_REDUCTION_WARP_THREADS + lane] = packet.value.x;
                    staged_packet_words[warp][1 * TILED_REDUCTION_WARP_THREADS + lane] = packet.value.y;
                } else {
                    staged_packet_words[warp][0 * TILED_REDUCTION_WARP_THREADS + lane] = packet.value.x;
                    staged_packet_words[warp][1 * TILED_REDUCTION_WARP_THREADS + lane] = packet.value.y;
                    staged_packet_words[warp][2 * TILED_REDUCTION_WARP_THREADS + lane] = packet.value.z;
                    staged_packet_words[warp][3 * TILED_REDUCTION_WARP_THREADS + lane] = packet.value.w;
                }
            }
            __syncwarp();

            if (valid_output && subgroup_lane < TaskCountPerOutput) {
                // Gather four same-component rows per conversion. WorkersPerComponent is compile-time, so row stride
                // and the power-of-two packet remap lower to adds/shifts/masks rather than runtime division/modulo.
                constexpr uint32_t RowsPerDecode = 4;
                constexpr uint32_t WorkerRowStride = WorkersPerComponent;
                for (uint32_t row = worker; row < group_rows;
                     row += WorkerRowStride * RowsPerDecode) {
                    uint32_t packed_component_values = 0;
                    uint32_t valid_values = 0;
#pragma unroll
                    for (uint32_t item = 0; item < RowsPerDecode; ++item) {
                        const uint32_t source_row = row + item * WorkerRowStride;
                        if (source_row < group_rows) {
                            const uint32_t logical_byte = source_row * K + component;
                            const uint32_t staged_byte = prefix_bytes + logical_byte;
                            const uint32_t source_packet = staged_byte >> PacketShift;
                            const uint32_t packet_byte = staged_byte & PacketMask;
                            const uint32_t source_lane = subgroup_lane_base + source_packet;
                            const uint32_t source_word =
                                staged_packet_words[warp]
                                                   [(packet_byte >> 2) * TILED_REDUCTION_WARP_THREADS +
                                                    source_lane];
                            const uint32_t source_value =
                                (source_word >> ((packet_byte & 3U) * 8U)) & 0xffU;
                            packed_component_values |= source_value << (item * 8U);
                            ++valid_values;
                        }
                    }
                    const float4 decoded = decodeFp8x4ToFp32<InputT>(packed_component_values);
                    if (valid_values > 0)
                        local += decoded.x;
                    if (valid_values > 1)
                        local += decoded.y;
                    if (valid_values > 2)
                        local += decoded.z;
                    if (valid_values > 3)
                        local += decoded.w;
                }
            }
            __syncwarp();
        }

        lane_partials[warp][lane] =
            (valid_output && subgroup_lane < TaskCountPerOutput) ? local : 0.0f;
        __syncwarp();
        if (valid_output && subgroup_lane < K) {
            float reduced = 0.0f;
            const uint32_t partial_base =
                output_slot * LanesPerOutput + subgroup_lane * WorkersPerComponent;
            for (uint32_t source_worker = 0; source_worker < WorkersPerComponent; ++source_worker) {
                reduced += lane_partials[warp][partial_base + source_worker];
            }
            output[outer_index * static_cast<uint64_t>(K) + subgroup_lane] =
                ThorLowPrecision::castToStorage<InputT>(reduced);
        }
        __syncwarp();
    }
}


// General Complete RParallel spans both output and R ownership with one packed FP8 physical family. O1 may use
// a 1/2/4/8-warp subgroup per output. O2/O4/O8 instead partition one warp into independent output subgroups and
// therefore require W1. Within each output subgroup, K<=lanes-per-output spends spare lanes as independent R
// workers/component; larger K assigns multiple components/lane. Every subgroup gets its own p4/p8/p16 packet slice
// in the warp-wide plane-major shared staging buffer and streams runtime R through that slice. This subsumes the
// ownership mechanics of the earlier MultiOutput path while extending them through medium K; output multiplicity,
// warps/output, and packet width are axes of one family rather than distinct physical kernel families.
template <typename InputT, int K, int PacketBytes, int OutputsPerWarp, bool LinearPackets = false>
__global__ void fp8GeneralRParallelPackedCompleteKernel(const InputT* input,
                                                        InputT* output,
                                                        uint32_t outer_size,
                                                        uint32_t reduction_size,
                                                        uint32_t warps_per_output) {
    static_assert(sizeof(InputT) == 1);
    static_assert(K >= 1 && K <= RPARALLEL_FP8_MAX_K);
    static_assert(PacketBytes == 4 || PacketBytes == 8 || PacketBytes == 16);
    static_assert(OutputsPerWarp == 1 || OutputsPerWarp == 2 || OutputsPerWarp == 4 || OutputsPerWarp == 8);
    static_assert(!LinearPackets || OutputsPerWarp == 1, "LinearPackets is calibrated only for O1 ownership.");
    constexpr uint32_t PacketWords = PacketBytes / static_cast<int>(sizeof(uint32_t));
    constexpr uint32_t PacketShift = PacketBytes == 4 ? 2 : (PacketBytes == 8 ? 3 : 4);
    constexpr uint32_t PacketMask = PacketBytes - 1;
    constexpr uint32_t LanesPerOutput = TILED_REDUCTION_WARP_THREADS / OutputsPerWarp;
    constexpr bool MultipleComponentsPerLane = K > static_cast<int>(LanesPerOutput);
    constexpr uint32_t WorkersPerComponent = MultipleComponentsPerLane ? 1U : LanesPerOutput / K;
    constexpr uint32_t TaskCountPerOutput =
        MultipleComponentsPerLane ? LanesPerOutput : K * WorkersPerComponent;
    constexpr uint32_t ComponentsPerLane =
        MultipleComponentsPerLane ? (K + LanesPerOutput - 1) / LanesPerOutput : 1U;
    constexpr uint32_t StagingBytesPerOutput = LanesPerOutput * PacketBytes;
    constexpr bool PacketAlignedRows = (K % PacketBytes) == 0;
    constexpr uint32_t MaxLogicalBytesPerGroup =
        PacketAlignedRows ? StagingBytesPerOutput : StagingBytesPerOutput - (PacketBytes - 1);
    constexpr uint32_t RowsPerGroup = MaxLogicalBytesPerGroup / K;
    constexpr uint32_t LanePartialColumns = MultipleComponentsPerLane ? 1U : TILED_REDUCTION_WARP_THREADS;
    constexpr uint32_t WarpPartialRows = OutputsPerWarp == 1 ? SHALLOW_FP8_WARPS : 1U;
    constexpr uint32_t WarpPartialColumns = OutputsPerWarp == 1 ? K : 1U;
    static_assert(LanesPerOutput * OutputsPerWarp == TILED_REDUCTION_WARP_THREADS);
    static_assert(PacketWords == 1 || PacketWords == 2 || PacketWords == 4);
    static_assert(WorkersPerComponent >= 1);
    static_assert(TaskCountPerOutput <= LanesPerOutput);
    static_assert(ComponentsPerLane >= 1 && ComponentsPerLane <= 16);
    static_assert(RowsPerGroup >= 1, "RParallel packet slice cannot hold one logical K row.");

    __shared__ uint32_t staged_packet_words[SHALLOW_FP8_WARPS]
                                            [PacketWords * TILED_REDUCTION_WARP_THREADS];
    __shared__ float lane_partials[SHALLOW_FP8_WARPS][LanePartialColumns];
    __shared__ float warp_partials[WarpPartialRows][WarpPartialColumns];

    const uint32_t warp = threadIdx.x / TILED_REDUCTION_WARP_THREADS;
    const uint32_t lane = threadIdx.x % TILED_REDUCTION_WARP_THREADS;
    const uint32_t outer_input_stride = reduction_size * K;
    const uint32_t total_input_bytes = outer_size * outer_input_stride;

    if constexpr (OutputsPerWarp > 1) {
        static_cast<void>(warps_per_output);
        // Multi-output ownership is warp-local. Each subgroup owns one output and one independent packet slice in
        // shared memory. The subgroup may either spend lanes across R workers/component or assign several K
        // components to each lane. Runtime R is streamed through RowsPerGroup just like O1.
        const uint32_t output_slot = lane / LanesPerOutput;
        const uint32_t subgroup_lane = lane % LanesPerOutput;
        const uint32_t subgroup_lane_base = output_slot * LanesPerOutput;
        const uint32_t component = MultipleComponentsPerLane ? subgroup_lane : subgroup_lane / WorkersPerComponent;
        const uint32_t worker = MultipleComponentsPerLane ? 0U : subgroup_lane % WorkersPerComponent;
        const uint32_t global_warp = blockIdx.x * SHALLOW_FP8_WARPS + warp;
        const uint32_t grid_warps = gridDim.x * SHALLOW_FP8_WARPS;

        for (uint32_t warp_group = global_warp;; warp_group += grid_warps) {
            const uint64_t outer_begin = static_cast<uint64_t>(warp_group) * OutputsPerWarp;
            if (outer_begin >= outer_size) {
                return;
            }
            const uint64_t outer_index = outer_begin + output_slot;
            const bool valid_output = outer_index < outer_size;
            float local[ComponentsPerLane];
#pragma unroll
            for (uint32_t slot = 0; slot < ComponentsPerLane; ++slot) {
                local[slot] = 0.0f;
            }
            const uint32_t outer_byte_begin =
                valid_output ? static_cast<uint32_t>(outer_index) * outer_input_stride : 0;

            for (uint32_t group_row = 0; group_row < reduction_size; group_row += RowsPerGroup) {
                const uint32_t remaining_rows = reduction_size - group_row;
                const uint32_t group_rows = remaining_rows < RowsPerGroup ? remaining_rows : RowsPerGroup;
                uint32_t prefix_bytes = 0;
                uint32_t packet_count = 0;
                uint32_t aligned_byte_begin = 0;
                if (valid_output) {
                    const uint32_t group_byte_begin = outer_byte_begin + group_row * K;
                    if constexpr (PacketAlignedRows) {
                        aligned_byte_begin = group_byte_begin;
                        packet_count = (group_rows * K) >> PacketShift;
                    } else {
                        prefix_bytes = group_byte_begin & PacketMask;
                        aligned_byte_begin = group_byte_begin - prefix_bytes;
                        const uint32_t logical_bytes = group_rows * K;
                        packet_count = (prefix_bytes + logical_bytes + PacketBytes - 1) >> PacketShift;
                    }
                }

                if (valid_output && subgroup_lane < packet_count) {
                    const uint32_t packet_byte_begin = aligned_byte_begin + subgroup_lane * PacketBytes;
                    RawVectorPacket<PacketBytes> packet{};
                    if (total_input_bytes >= static_cast<uint32_t>(PacketBytes) &&
                        packet_byte_begin <= total_input_bytes - static_cast<uint32_t>(PacketBytes)) {
                        packet = *reinterpret_cast<const RawVectorPacket<PacketBytes>*>(
                            reinterpret_cast<const uint8_t*>(input) + packet_byte_begin);
                    } else {
                        auto* packet_bytes = reinterpret_cast<uint8_t*>(&packet.value);
#pragma unroll
                        for (uint32_t byte = 0; byte < PacketBytes; ++byte) {
                            const uint32_t byte_index = packet_byte_begin + byte;
                            if (byte_index < total_input_bytes) {
                                packet_bytes[byte] = reinterpret_cast<const uint8_t*>(input)[byte_index];
                            }
                        }
                    }

                    if constexpr (PacketBytes == 4) {
                        staged_packet_words[warp][lane] = packet.value;
                    } else if constexpr (PacketBytes == 8) {
                        staged_packet_words[warp][0 * TILED_REDUCTION_WARP_THREADS + lane] = packet.value.x;
                        staged_packet_words[warp][1 * TILED_REDUCTION_WARP_THREADS + lane] = packet.value.y;
                    } else {
                        staged_packet_words[warp][0 * TILED_REDUCTION_WARP_THREADS + lane] = packet.value.x;
                        staged_packet_words[warp][1 * TILED_REDUCTION_WARP_THREADS + lane] = packet.value.y;
                        staged_packet_words[warp][2 * TILED_REDUCTION_WARP_THREADS + lane] = packet.value.z;
                        staged_packet_words[warp][3 * TILED_REDUCTION_WARP_THREADS + lane] = packet.value.w;
                    }
                }
                __syncwarp();

                if (valid_output && subgroup_lane < TaskCountPerOutput) {
                    constexpr uint32_t RowsPerDecode = 4;
#pragma unroll
                    for (uint32_t slot = 0; slot < ComponentsPerLane; ++slot) {
                        const uint32_t owned_component =
                            MultipleComponentsPerLane ? subgroup_lane + slot * LanesPerOutput : component;
                        if constexpr (!MultipleComponentsPerLane) {
                            if (slot != 0) {
                                continue;
                            }
                        }
                        if (owned_component >= K) {
                            continue;
                        }
                        for (uint32_t row = worker; row < group_rows;
                             row += WorkersPerComponent * RowsPerDecode) {
                            uint32_t packed_component_values = 0;
                            uint32_t valid_values = 0;
#pragma unroll
                            for (uint32_t item = 0; item < RowsPerDecode; ++item) {
                                const uint32_t source_row = row + item * WorkersPerComponent;
                                if (source_row < group_rows) {
                                    const uint32_t logical_byte = source_row * K + owned_component;
                                    const uint32_t staged_byte = prefix_bytes + logical_byte;
                                    const uint32_t source_packet = staged_byte >> PacketShift;
                                    const uint32_t packet_byte = staged_byte & PacketMask;
                                    const uint32_t source_lane = subgroup_lane_base + source_packet;
                                    const uint32_t source_word =
                                        staged_packet_words[warp]
                                                           [(packet_byte >> 2) * TILED_REDUCTION_WARP_THREADS +
                                                            source_lane];
                                    const uint32_t source_value =
                                        (source_word >> ((packet_byte & 3U) * 8U)) & 0xffU;
                                    packed_component_values |= source_value << (item * 8U);
                                    ++valid_values;
                                }
                            }
                            const float4 decoded = decodeFp8x4ToFp32<InputT>(packed_component_values);
                            if (valid_values > 0) local[slot] += decoded.x;
                            if (valid_values > 1) local[slot] += decoded.y;
                            if (valid_values > 2) local[slot] += decoded.z;
                            if (valid_values > 3) local[slot] += decoded.w;
                        }
                    }
                }
                __syncwarp();
            }

            if constexpr (MultipleComponentsPerLane) {
#pragma unroll
                for (uint32_t slot = 0; slot < ComponentsPerLane; ++slot) {
                    const uint32_t owned_component = subgroup_lane + slot * LanesPerOutput;
                    if (valid_output && owned_component < K) {
                        output[outer_index * static_cast<uint64_t>(K) + owned_component] =
                            ThorLowPrecision::castToStorage<InputT>(local[slot]);
                    }
                }
            } else {
                lane_partials[warp][lane] =
                    (valid_output && subgroup_lane < TaskCountPerOutput) ? local[0] : 0.0f;
                __syncwarp();
                if (valid_output && subgroup_lane < K) {
                    float reduced = 0.0f;
                    const uint32_t partial_base =
                        output_slot * LanesPerOutput + subgroup_lane * WorkersPerComponent;
#pragma unroll
                    for (uint32_t source_worker = 0; source_worker < WorkersPerComponent; ++source_worker) {
                        reduced += lane_partials[warp][partial_base + source_worker];
                    }
                    output[outer_index * static_cast<uint64_t>(K) + subgroup_lane] =
                        ThorLowPrecision::castToStorage<InputT>(reduced);
                }
                __syncwarp();
            }
        }
    } else {

    // O1 owns one output with one or more complete warps. W1 remains warp-local; W2/W4/W8 consume disjoint R
    // groups and fold K FP32 warp partials in CTA shared memory.
    const uint32_t outputs_per_block = SHALLOW_FP8_WARPS / warps_per_output;
    const uint32_t output_group = warp / warps_per_output;
    const uint32_t warp_in_output = warp - output_group * warps_per_output;
    const uint32_t first_warp_in_output = output_group * warps_per_output;
    const uint32_t component = MultipleComponentsPerLane ? lane : lane / WorkersPerComponent;
    const uint32_t worker = MultipleComponentsPerLane ? 0U : lane % WorkersPerComponent;
    const uint32_t rows_per_output_round = RowsPerGroup * warps_per_output;
    const uint64_t block_output_stride = static_cast<uint64_t>(gridDim.x) * outputs_per_block;

    for (uint64_t block_outer_begin = static_cast<uint64_t>(blockIdx.x) * outputs_per_block;
         block_outer_begin < outer_size;
         block_outer_begin += block_output_stride) {
        const uint64_t outer_index = block_outer_begin + output_group;
        const bool valid_output = outer_index < outer_size;
        float local[ComponentsPerLane];
#pragma unroll
        for (uint32_t slot = 0; slot < ComponentsPerLane; ++slot) {
            local[slot] = 0.0f;
        }
        const uint32_t outer_byte_begin =
            valid_output ? static_cast<uint32_t>(outer_index) * outer_input_stride : 0;

        for (uint32_t group_row = warp_in_output * RowsPerGroup;
             group_row < reduction_size;
             group_row += rows_per_output_round) {
            const uint32_t remaining_rows = reduction_size - group_row;
            const uint32_t group_rows = remaining_rows < RowsPerGroup ? remaining_rows : RowsPerGroup;
            uint32_t prefix_bytes = 0;
            uint32_t packet_count = 0;
            uint32_t aligned_byte_begin = 0;
            if (valid_output) {
                const uint32_t group_byte_begin = outer_byte_begin + group_row * K;
                if constexpr (PacketAlignedRows) {
                    aligned_byte_begin = group_byte_begin;
                    packet_count = (group_rows * K) >> PacketShift;
                } else {
                    prefix_bytes = group_byte_begin & PacketMask;
                    aligned_byte_begin = group_byte_begin - prefix_bytes;
                    const uint32_t logical_bytes = group_rows * K;
                    packet_count = (prefix_bytes + logical_bytes + PacketBytes - 1) >> PacketShift;
                }
            }

            if (valid_output && lane < packet_count) {
                const uint32_t packet_byte_begin = aligned_byte_begin + lane * PacketBytes;
                RawVectorPacket<PacketBytes> packet{};
                if (total_input_bytes >= static_cast<uint32_t>(PacketBytes) &&
                    packet_byte_begin <= total_input_bytes - static_cast<uint32_t>(PacketBytes)) {
                    packet = *reinterpret_cast<const RawVectorPacket<PacketBytes>*>(
                        reinterpret_cast<const uint8_t*>(input) + packet_byte_begin);
                } else {
                    auto* packet_bytes = reinterpret_cast<uint8_t*>(&packet.value);
#pragma unroll
                    for (uint32_t byte = 0; byte < PacketBytes; ++byte) {
                        const uint32_t byte_index = packet_byte_begin + byte;
                        if (byte_index < total_input_bytes) {
                            packet_bytes[byte] = reinterpret_cast<const uint8_t*>(input)[byte_index];
                        }
                    }
                }

                if constexpr (PacketBytes == 4) {
                    staged_packet_words[warp][lane] = packet.value;
                } else if constexpr (PacketBytes == 8) {
                    if constexpr (LinearPackets) {
                        staged_packet_words[warp][lane * PacketWords + 0] = packet.value.x;
                        staged_packet_words[warp][lane * PacketWords + 1] = packet.value.y;
                    } else {
                        staged_packet_words[warp][0 * TILED_REDUCTION_WARP_THREADS + lane] = packet.value.x;
                        staged_packet_words[warp][1 * TILED_REDUCTION_WARP_THREADS + lane] = packet.value.y;
                    }
                } else {
                    if constexpr (LinearPackets) {
                        staged_packet_words[warp][lane * PacketWords + 0] = packet.value.x;
                        staged_packet_words[warp][lane * PacketWords + 1] = packet.value.y;
                        staged_packet_words[warp][lane * PacketWords + 2] = packet.value.z;
                        staged_packet_words[warp][lane * PacketWords + 3] = packet.value.w;
                    } else {
                        staged_packet_words[warp][0 * TILED_REDUCTION_WARP_THREADS + lane] = packet.value.x;
                        staged_packet_words[warp][1 * TILED_REDUCTION_WARP_THREADS + lane] = packet.value.y;
                        staged_packet_words[warp][2 * TILED_REDUCTION_WARP_THREADS + lane] = packet.value.z;
                        staged_packet_words[warp][3 * TILED_REDUCTION_WARP_THREADS + lane] = packet.value.w;
                    }
                }
            }
            __syncwarp();

            if (valid_output && lane < TaskCountPerOutput) {
                constexpr uint32_t RowsPerDecode = 4;
#pragma unroll
                for (uint32_t slot = 0; slot < ComponentsPerLane; ++slot) {
                    const uint32_t owned_component =
                        MultipleComponentsPerLane ? lane + slot * TILED_REDUCTION_WARP_THREADS : component;
                    if constexpr (!MultipleComponentsPerLane) {
                        if (slot != 0) {
                            continue;
                        }
                    }
                    if (owned_component >= K) {
                        continue;
                    }
                    for (uint32_t row = worker; row < group_rows;
                         row += WorkersPerComponent * RowsPerDecode) {
                        uint32_t packed_component_values = 0;
                        uint32_t valid_values = 0;
#pragma unroll
                        for (uint32_t item = 0; item < RowsPerDecode; ++item) {
                            const uint32_t source_row = row + item * WorkersPerComponent;
                            if (source_row < group_rows) {
                                const uint32_t logical_byte = source_row * K + owned_component;
                                const uint32_t staged_byte = prefix_bytes + logical_byte;
                                uint32_t source_value = 0;
                                if constexpr (LinearPackets) {
                                    const auto* staged_bytes =
                                        reinterpret_cast<const uint8_t*>(&staged_packet_words[warp][0]);
                                    source_value = staged_bytes[staged_byte];
                                } else {
                                    const uint32_t source_packet = staged_byte >> PacketShift;
                                    const uint32_t packet_byte = staged_byte & PacketMask;
                                    const uint32_t source_word =
                                        staged_packet_words[warp]
                                                           [(packet_byte >> 2) * TILED_REDUCTION_WARP_THREADS +
                                                            source_packet];
                                    source_value = (source_word >> ((packet_byte & 3U) * 8U)) & 0xffU;
                                }
                                packed_component_values |= source_value << (item * 8U);
                                ++valid_values;
                            }
                        }
                        const float4 decoded = decodeFp8x4ToFp32<InputT>(packed_component_values);
                        if (valid_values > 0) local[slot] += decoded.x;
                        if (valid_values > 1) local[slot] += decoded.y;
                        if (valid_values > 2) local[slot] += decoded.z;
                        if (valid_values > 3) local[slot] += decoded.w;
                    }
                }
            }
            __syncwarp();
        }

        if (warps_per_output == 1) {
            if constexpr (MultipleComponentsPerLane) {
#pragma unroll
                for (uint32_t slot = 0; slot < ComponentsPerLane; ++slot) {
                    const uint32_t owned_component = lane + slot * TILED_REDUCTION_WARP_THREADS;
                    if (valid_output && owned_component < K) {
                        output[outer_index * static_cast<uint64_t>(K) + owned_component] =
                            ThorLowPrecision::castToStorage<InputT>(local[slot]);
                    }
                }
            } else if constexpr (LinearPackets && WorkersPerComponent == 1) {
                if (valid_output && lane < K) {
                    output[outer_index * static_cast<uint64_t>(K) + lane] =
                        ThorLowPrecision::castToStorage<InputT>(local[0]);
                }
                __syncwarp();
            } else {
                lane_partials[warp][lane] = (valid_output && lane < TaskCountPerOutput) ? local[0] : 0.0f;
                __syncwarp();
                if (valid_output && lane < K) {
                    float reduced = 0.0f;
                    const uint32_t partial_base = lane * WorkersPerComponent;
#pragma unroll
                    for (uint32_t source_worker = 0; source_worker < WorkersPerComponent; ++source_worker) {
                        reduced += lane_partials[warp][partial_base + source_worker];
                    }
                    output[outer_index * static_cast<uint64_t>(K) + lane] =
                        ThorLowPrecision::castToStorage<InputT>(reduced);
                }
                __syncwarp();
            }
            continue;
        }

        if constexpr (MultipleComponentsPerLane) {
#pragma unroll
            for (uint32_t slot = 0; slot < ComponentsPerLane; ++slot) {
                const uint32_t owned_component = lane + slot * TILED_REDUCTION_WARP_THREADS;
                if (owned_component < K) {
                    warp_partials[warp][owned_component] = valid_output ? local[slot] : 0.0f;
                }
            }
        } else if constexpr (LinearPackets && WorkersPerComponent == 1) {
            if (lane < K) {
                warp_partials[warp][lane] = valid_output ? local[0] : 0.0f;
            }
        } else {
            lane_partials[warp][lane] = (valid_output && lane < TaskCountPerOutput) ? local[0] : 0.0f;
            __syncwarp();
            if (lane < K) {
                float warp_reduced = 0.0f;
                const uint32_t partial_base = lane * WorkersPerComponent;
#pragma unroll
                for (uint32_t source_worker = 0; source_worker < WorkersPerComponent; ++source_worker) {
                    warp_reduced += lane_partials[warp][partial_base + source_worker];
                }
                warp_partials[warp][lane] = valid_output ? warp_reduced : 0.0f;
            }
        }
        __syncthreads();

        if (warp_in_output == 0) {
#pragma unroll
            for (uint32_t slot = 0; slot < ComponentsPerLane; ++slot) {
                const uint32_t owned_component =
                    MultipleComponentsPerLane ? lane + slot * TILED_REDUCTION_WARP_THREADS : lane;
                if constexpr (!MultipleComponentsPerLane) {
                    if (slot != 0) {
                        continue;
                    }
                }
                if (valid_output && owned_component < K) {
                    float reduced = 0.0f;
                    for (uint32_t source_warp = 0; source_warp < warps_per_output; ++source_warp) {
                        reduced += warp_partials[first_warp_in_output + source_warp][owned_component];
                    }
                    output[outer_index * static_cast<uint64_t>(K) + owned_component] =
                        ThorLowPrecision::castToStorage<InputT>(reduced);
                }
            }
        }
        __syncthreads();
    }
    }
}


// Staged topology of the same generalized RParallel family. S>1 shards one logical output across independent CTAs;
// each shard retains the O1 W1/W2/W4/W8 p4/p8/p16 packed producer and component/R-worker ownership used by the
// Complete kernel above, but writes FP32 [outer, shard, K] partials. Shards are balanced across R so S is an
// orthogonal amount-of-progress axis rather than another rows-per-shard kernel specialization. Output parallelism O>1
// is intentionally not combined with S>1: shard work already supplies independent CTA/warp work, while W remains the
// useful intra-shard cooperation axis.
template <typename InputT, int K, int PacketBytes, bool LinearPackets = false>
__global__ void fp8GeneralRParallelPackedFirstStageKernel(const InputT* input,
                                                          float* fp32_partials,
                                                          uint32_t outer_size,
                                                          uint32_t reduction_size,
                                                          uint32_t warps_per_output,
                                                          uint32_t shards_per_output) {
    static_assert(sizeof(InputT) == 1);
    static_assert(K >= 1 && K <= RPARALLEL_FP8_MAX_K);
    static_assert(PacketBytes == 4 || PacketBytes == 8 || PacketBytes == 16);
    constexpr uint32_t PacketWords = PacketBytes / static_cast<int>(sizeof(uint32_t));
    constexpr uint32_t PacketShift = PacketBytes == 4 ? 2 : (PacketBytes == 8 ? 3 : 4);
    constexpr uint32_t PacketMask = PacketBytes - 1;
    constexpr bool MultipleComponentsPerLane = K > TILED_REDUCTION_WARP_THREADS;
    constexpr uint32_t WorkersPerComponent = MultipleComponentsPerLane ? 1U : TILED_REDUCTION_WARP_THREADS / K;
    constexpr uint32_t TaskCountPerOutput =
        MultipleComponentsPerLane ? TILED_REDUCTION_WARP_THREADS : K * WorkersPerComponent;
    constexpr uint32_t ComponentsPerLane =
        MultipleComponentsPerLane ? (K + TILED_REDUCTION_WARP_THREADS - 1) / TILED_REDUCTION_WARP_THREADS : 1U;
    constexpr uint32_t StagingBytesPerOutput = TILED_REDUCTION_WARP_THREADS * PacketBytes;
    constexpr bool PacketAlignedRows = (K % PacketBytes) == 0;
    constexpr uint32_t MaxLogicalBytesPerGroup =
        PacketAlignedRows ? StagingBytesPerOutput : StagingBytesPerOutput - (PacketBytes - 1);
    constexpr uint32_t RowsPerGroup = MaxLogicalBytesPerGroup / K;
    constexpr uint32_t LanePartialColumns = MultipleComponentsPerLane ? 1U : TILED_REDUCTION_WARP_THREADS;
    static_assert(PacketWords == 1 || PacketWords == 2 || PacketWords == 4);
    static_assert(WorkersPerComponent >= 1);
    static_assert(TaskCountPerOutput <= TILED_REDUCTION_WARP_THREADS);
    static_assert(ComponentsPerLane >= 1 && ComponentsPerLane <= 8);
    static_assert(RowsPerGroup >= 1, "Staged RParallel packet window cannot hold one logical K row.");

    __shared__ uint32_t staged_packet_words[SHALLOW_FP8_WARPS]
                                            [PacketWords * TILED_REDUCTION_WARP_THREADS];
    __shared__ float lane_partials[SHALLOW_FP8_WARPS][LanePartialColumns];
    __shared__ float warp_partials[SHALLOW_FP8_WARPS][K];

    const uint32_t warp = threadIdx.x / TILED_REDUCTION_WARP_THREADS;
    const uint32_t lane = threadIdx.x % TILED_REDUCTION_WARP_THREADS;
    const uint32_t outputs_per_block = SHALLOW_FP8_WARPS / warps_per_output;
    const uint32_t output_group = warp / warps_per_output;
    const uint32_t warp_in_output = warp - output_group * warps_per_output;
    const uint32_t first_warp_in_output = output_group * warps_per_output;
    const uint32_t component = MultipleComponentsPerLane ? lane : lane / WorkersPerComponent;
    const uint32_t worker = MultipleComponentsPerLane ? 0U : lane % WorkersPerComponent;
    const uint32_t rows_per_output_round = RowsPerGroup * warps_per_output;
    const uint32_t outer_input_stride = reduction_size * K;
    const uint32_t total_input_bytes = outer_size * outer_input_stride;
    const uint64_t work_count = static_cast<uint64_t>(outer_size) * shards_per_output;
    const uint64_t block_work_stride = static_cast<uint64_t>(gridDim.x) * outputs_per_block;
    const uint32_t base_rows = reduction_size / shards_per_output;
    const uint32_t extra_rows = reduction_size - base_rows * shards_per_output;

    for (uint64_t block_work_begin = static_cast<uint64_t>(blockIdx.x) * outputs_per_block;
         block_work_begin < work_count;
         block_work_begin += block_work_stride) {
        const uint64_t work_index = block_work_begin + output_group;
        const bool valid_work = work_index < work_count;
        uint32_t outer_index = 0;
        uint32_t shard_index = 0;
        uint32_t shard_row_begin = 0;
        uint32_t shard_row_end = 0;
        if (valid_work) {
            outer_index = static_cast<uint32_t>(work_index / shards_per_output);
            shard_index = static_cast<uint32_t>(work_index - static_cast<uint64_t>(outer_index) * shards_per_output);
            const uint32_t prior_extra = shard_index < extra_rows ? shard_index : extra_rows;
            shard_row_begin = shard_index * base_rows + prior_extra;
            shard_row_end = shard_row_begin + base_rows + static_cast<uint32_t>(shard_index < extra_rows);
        }

        float local[ComponentsPerLane];
#pragma unroll
        for (uint32_t slot = 0; slot < ComponentsPerLane; ++slot) {
            local[slot] = 0.0f;
        }
        const uint32_t outer_byte_begin = valid_work ? outer_index * outer_input_stride : 0;

        for (uint32_t group_row = shard_row_begin + warp_in_output * RowsPerGroup;
             valid_work && group_row < shard_row_end;
             group_row += rows_per_output_round) {
            const uint32_t remaining_rows = shard_row_end - group_row;
            const uint32_t group_rows = remaining_rows < RowsPerGroup ? remaining_rows : RowsPerGroup;
            uint32_t prefix_bytes = 0;
            uint32_t packet_count = 0;
            uint32_t aligned_byte_begin = 0;
            const uint32_t group_byte_begin = outer_byte_begin + group_row * K;
            if constexpr (PacketAlignedRows) {
                aligned_byte_begin = group_byte_begin;
                packet_count = (group_rows * K) >> PacketShift;
            } else {
                prefix_bytes = group_byte_begin & PacketMask;
                aligned_byte_begin = group_byte_begin - prefix_bytes;
                const uint32_t logical_bytes = group_rows * K;
                packet_count = (prefix_bytes + logical_bytes + PacketBytes - 1) >> PacketShift;
            }

            if (lane < packet_count) {
                const uint32_t packet_byte_begin = aligned_byte_begin + lane * PacketBytes;
                RawVectorPacket<PacketBytes> packet{};
                if (total_input_bytes >= static_cast<uint32_t>(PacketBytes) &&
                    packet_byte_begin <= total_input_bytes - static_cast<uint32_t>(PacketBytes)) {
                    packet = *reinterpret_cast<const RawVectorPacket<PacketBytes>*>(
                        reinterpret_cast<const uint8_t*>(input) + packet_byte_begin);
                } else {
                    auto* packet_bytes = reinterpret_cast<uint8_t*>(&packet.value);
#pragma unroll
                    for (uint32_t byte = 0; byte < PacketBytes; ++byte) {
                        const uint32_t byte_index = packet_byte_begin + byte;
                        if (byte_index < total_input_bytes) {
                            packet_bytes[byte] = reinterpret_cast<const uint8_t*>(input)[byte_index];
                        }
                    }
                }

                if constexpr (PacketBytes == 4) {
                    staged_packet_words[warp][lane] = packet.value;
                } else if constexpr (PacketBytes == 8) {
                    if constexpr (LinearPackets) {
                        staged_packet_words[warp][lane * PacketWords + 0] = packet.value.x;
                        staged_packet_words[warp][lane * PacketWords + 1] = packet.value.y;
                    } else {
                        staged_packet_words[warp][0 * TILED_REDUCTION_WARP_THREADS + lane] = packet.value.x;
                        staged_packet_words[warp][1 * TILED_REDUCTION_WARP_THREADS + lane] = packet.value.y;
                    }
                } else {
                    if constexpr (LinearPackets) {
                        staged_packet_words[warp][lane * PacketWords + 0] = packet.value.x;
                        staged_packet_words[warp][lane * PacketWords + 1] = packet.value.y;
                        staged_packet_words[warp][lane * PacketWords + 2] = packet.value.z;
                        staged_packet_words[warp][lane * PacketWords + 3] = packet.value.w;
                    } else {
                        staged_packet_words[warp][0 * TILED_REDUCTION_WARP_THREADS + lane] = packet.value.x;
                        staged_packet_words[warp][1 * TILED_REDUCTION_WARP_THREADS + lane] = packet.value.y;
                        staged_packet_words[warp][2 * TILED_REDUCTION_WARP_THREADS + lane] = packet.value.z;
                        staged_packet_words[warp][3 * TILED_REDUCTION_WARP_THREADS + lane] = packet.value.w;
                    }
                }
            }
            __syncwarp();

            if (lane < TaskCountPerOutput) {
                constexpr uint32_t RowsPerDecode = 4;
#pragma unroll
                for (uint32_t slot = 0; slot < ComponentsPerLane; ++slot) {
                    const uint32_t owned_component =
                        MultipleComponentsPerLane ? lane + slot * TILED_REDUCTION_WARP_THREADS : component;
                    if constexpr (!MultipleComponentsPerLane) {
                        if (slot != 0) continue;
                    }
                    if (owned_component >= K) continue;
                    for (uint32_t row = worker; row < group_rows;
                         row += WorkersPerComponent * RowsPerDecode) {
                        uint32_t packed_component_values = 0;
                        uint32_t valid_values = 0;
#pragma unroll
                        for (uint32_t item = 0; item < RowsPerDecode; ++item) {
                            const uint32_t source_row = row + item * WorkersPerComponent;
                            if (source_row < group_rows) {
                                const uint32_t logical_byte = source_row * K + owned_component;
                                const uint32_t staged_byte = prefix_bytes + logical_byte;
                                uint32_t source_value = 0;
                                if constexpr (LinearPackets) {
                                    const auto* staged_bytes =
                                        reinterpret_cast<const uint8_t*>(&staged_packet_words[warp][0]);
                                    source_value = staged_bytes[staged_byte];
                                } else {
                                    const uint32_t source_packet = staged_byte >> PacketShift;
                                    const uint32_t packet_byte = staged_byte & PacketMask;
                                    const uint32_t source_word =
                                        staged_packet_words[warp]
                                                           [(packet_byte >> 2) * TILED_REDUCTION_WARP_THREADS +
                                                            source_packet];
                                    source_value = (source_word >> ((packet_byte & 3U) * 8U)) & 0xffU;
                                }
                                packed_component_values |= source_value << (item * 8U);
                                ++valid_values;
                            }
                        }
                        const float4 decoded = decodeFp8x4ToFp32<InputT>(packed_component_values);
                        if (valid_values > 0) local[slot] += decoded.x;
                        if (valid_values > 1) local[slot] += decoded.y;
                        if (valid_values > 2) local[slot] += decoded.z;
                        if (valid_values > 3) local[slot] += decoded.w;
                    }
                }
            }
            __syncwarp();
        }

        const uint64_t partial_base = work_index * static_cast<uint64_t>(K);
        if (warps_per_output == 1) {
            if constexpr (MultipleComponentsPerLane) {
#pragma unroll
                for (uint32_t slot = 0; slot < ComponentsPerLane; ++slot) {
                    const uint32_t owned_component = lane + slot * TILED_REDUCTION_WARP_THREADS;
                    if (valid_work && owned_component < K) {
                        fp32_partials[partial_base + owned_component] = local[slot];
                    }
                }
            } else if constexpr (LinearPackets && WorkersPerComponent == 1) {
                if (valid_work && lane < K) {
                    fp32_partials[partial_base + lane] = local[0];
                }
                __syncwarp();
            } else {
                lane_partials[warp][lane] = (valid_work && lane < TaskCountPerOutput) ? local[0] : 0.0f;
                __syncwarp();
                if (valid_work && lane < K) {
                    float reduced = 0.0f;
                    const uint32_t lane_base = lane * WorkersPerComponent;
#pragma unroll
                    for (uint32_t source_worker = 0; source_worker < WorkersPerComponent; ++source_worker) {
                        reduced += lane_partials[warp][lane_base + source_worker];
                    }
                    fp32_partials[partial_base + lane] = reduced;
                }
                __syncwarp();
            }
            continue;
        }

        if constexpr (MultipleComponentsPerLane) {
#pragma unroll
            for (uint32_t slot = 0; slot < ComponentsPerLane; ++slot) {
                const uint32_t owned_component = lane + slot * TILED_REDUCTION_WARP_THREADS;
                if (owned_component < K) {
                    warp_partials[warp][owned_component] = valid_work ? local[slot] : 0.0f;
                }
            }
        } else if constexpr (LinearPackets && WorkersPerComponent == 1) {
            if (lane < K) {
                warp_partials[warp][lane] = valid_work ? local[0] : 0.0f;
            }
        } else {
            lane_partials[warp][lane] = (valid_work && lane < TaskCountPerOutput) ? local[0] : 0.0f;
            __syncwarp();
            if (lane < K) {
                float warp_reduced = 0.0f;
                const uint32_t lane_base = lane * WorkersPerComponent;
#pragma unroll
                for (uint32_t source_worker = 0; source_worker < WorkersPerComponent; ++source_worker) {
                    warp_reduced += lane_partials[warp][lane_base + source_worker];
                }
                warp_partials[warp][lane] = valid_work ? warp_reduced : 0.0f;
            }
        }
        __syncthreads();

        if (warp_in_output == 0) {
#pragma unroll
            for (uint32_t slot = 0; slot < ComponentsPerLane; ++slot) {
                const uint32_t owned_component =
                    MultipleComponentsPerLane ? lane + slot * TILED_REDUCTION_WARP_THREADS : lane;
                if constexpr (!MultipleComponentsPerLane) {
                    if (slot != 0) continue;
                }
                if (valid_work && owned_component < K) {
                    float reduced = 0.0f;
                    for (uint32_t source_warp = 0; source_warp < warps_per_output; ++source_warp) {
                        reduced += warp_partials[first_warp_in_output + source_warp][owned_component];
                    }
                    fp32_partials[partial_base + owned_component] = reduced;
                }
            }
        }
        __syncthreads();
    }
}

template <typename KernelT>
[[nodiscard]] CubKernelOccupancyInfo queryShallowFp8Occupancy(KernelT kernel);

// K=2 has a natural p4 orientation across R rather than across K: two adjacent rows contain exactly four FP8
// values. This benchmark-only specialization gives each lane one aligned p4 packet, reduces both retained components
// in registers, and uses 16 lanes/output for R=31/32 or one full warp/output for R=63/64. Odd R deliberately shares
// one boundary packet between adjacent outputs; each output consumes only its own fp8x2 half. There is no FP32 input
// staging and no byte-wise ownership remap.
template <typename InputT, int R>
__global__ void fp8K2PairedRowsRParallelP4CompleteKernel(const InputT* input, InputT* output, uint64_t outer_size) {
    static_assert(sizeof(InputT) == 1);
    static_assert(R == 31 || R == 32 || R == 63 || R == 64);
    constexpr int LanesPerOutput = R <= 32 ? 16 : 32;
    constexpr int OutputsPerWarp = TILED_REDUCTION_WARP_THREADS / LanesPerOutput;
    constexpr uint32_t FullWarpMask = 0xffffffffu;
    static_assert(OutputsPerWarp == 1 || OutputsPerWarp == 2);
    static_assert((R + 1) / 2 == LanesPerOutput);

    // Only the one-output/warp R=63/64 shape needs CTA-local writeback pairing so every logical output still uses a
    // p4 store. Input reduction itself is entirely register/warp local.
    __shared__ float2 warp_results[SHALLOW_FP8_WARPS];

    const int warp = static_cast<int>(threadIdx.x) / TILED_REDUCTION_WARP_THREADS;
    const int lane = static_cast<int>(threadIdx.x) % TILED_REDUCTION_WARP_THREADS;

    if constexpr (OutputsPerWarp == 2) {
        const int subgroup = lane / LanesPerOutput;
        const int subgroup_lane = lane % LanesPerOutput;
        const uint64_t global_warp = static_cast<uint64_t>(blockIdx.x) * SHALLOW_FP8_WARPS + static_cast<uint64_t>(warp);
        const uint64_t grid_warps = static_cast<uint64_t>(gridDim.x) * SHALLOW_FP8_WARPS;

        for (uint64_t warp_group = global_warp;; warp_group += grid_warps) {
            const uint64_t outer_begin = warp_group * OutputsPerWarp;
            if (outer_begin >= outer_size) {
                return;
            }
            const uint64_t outer_index = outer_begin + static_cast<uint64_t>(subgroup);
            const bool valid_output = outer_index < outer_size;
            float k0 = 0.0f;
            float k1 = 0.0f;

            if (valid_output) {
                const InputT* outer_input = input + outer_index * static_cast<uint64_t>(R * 2);
                const uintptr_t address = reinterpret_cast<uintptr_t>(outer_input);
                const auto* packets = reinterpret_cast<const RawVectorPacket<SHALLOW_FP8_PACKET_BYTES>*>(
                    address & ~uintptr_t{SHALLOW_FP8_PACKET_BYTES - 1});
                const float4 values = decodeFp8x4ToFp32<InputT>(packets[subgroup_lane].value);

                if constexpr ((R & 1) == 0) {
                    k0 = values.x + values.z;
                    k1 = values.y + values.w;
                } else {
                    // R*2 is 2 (mod 4), so consecutive outputs alternate between an aligned start and a two-byte
                    // prefix. The aligned output's final packet contributes only its low fp8x2; the misaligned
                    // output's first packet contributes only its high fp8x2. All interior packets contribute two rows.
                    if ((outer_index & 1ULL) == 0) {
                        if (subgroup_lane == LanesPerOutput - 1) {
                            k0 = values.x;
                            k1 = values.y;
                        } else {
                            k0 = values.x + values.z;
                            k1 = values.y + values.w;
                        }
                    } else {
                        if (subgroup_lane == 0) {
                            k0 = values.z;
                            k1 = values.w;
                        } else {
                            k0 = values.x + values.z;
                            k1 = values.y + values.w;
                        }
                    }
                }
            }

            for (int offset = LanesPerOutput / 2; offset > 0; offset >>= 1) {
                k0 += __shfl_down_sync(FullWarpMask, k0, offset, LanesPerOutput);
                k1 += __shfl_down_sync(FullWarpMask, k1, offset, LanesPerOutput);
            }

            const float output0_k0 = __shfl_sync(FullWarpMask, k0, 0);
            const float output0_k1 = __shfl_sync(FullWarpMask, k1, 0);
            const float output1_k0 = __shfl_sync(FullWarpMask, k0, LanesPerOutput);
            const float output1_k1 = __shfl_sync(FullWarpMask, k1, LanesPerOutput);
            if (lane == 0) {
                const float2 low_values = make_float2(output0_k0, output0_k1);
                const float2 high_values = outer_begin + 1 < outer_size
                    ? make_float2(output1_k0, output1_k1)
                    : make_float2(0.0f, 0.0f);
                uint16_t low_bits;
                uint16_t high_bits;
                if constexpr (std::is_same_v<InputT, __nv_fp8_e4m3>) {
                    low_bits = __nv_cvt_float2_to_fp8x2(low_values, __NV_SATFINITE, __NV_E4M3);
                    high_bits = __nv_cvt_float2_to_fp8x2(high_values, __NV_SATFINITE, __NV_E4M3);
                } else {
                    low_bits = __nv_cvt_float2_to_fp8x2(low_values, __NV_SATFINITE, __NV_E5M2);
                    high_bits = __nv_cvt_float2_to_fp8x2(high_values, __NV_SATFINITE, __NV_E5M2);
                }
                RawVectorPacket<SHALLOW_FP8_PACKET_BYTES> packet{};
                packet.value = static_cast<uint32_t>(low_bits) | (static_cast<uint32_t>(high_bits) << 16);
                reinterpret_cast<RawVectorPacket<SHALLOW_FP8_PACKET_BYTES>*>(output)[warp_group] = packet;
            }
        }
    } else {
        const uint64_t grid_stride = static_cast<uint64_t>(gridDim.x);
        for (uint64_t block_round = static_cast<uint64_t>(blockIdx.x);; block_round += grid_stride) {
            const uint64_t outer_begin = block_round * SHALLOW_FP8_WARPS;
            if (outer_begin >= outer_size) {
                return;
            }
            const uint64_t outer_index = outer_begin + static_cast<uint64_t>(warp);
            const bool valid_output = outer_index < outer_size;
            float k0 = 0.0f;
            float k1 = 0.0f;

            if (valid_output) {
                const InputT* outer_input = input + outer_index * static_cast<uint64_t>(R * 2);
                const uintptr_t address = reinterpret_cast<uintptr_t>(outer_input);
                const auto* packets = reinterpret_cast<const RawVectorPacket<SHALLOW_FP8_PACKET_BYTES>*>(
                    address & ~uintptr_t{SHALLOW_FP8_PACKET_BYTES - 1});
                const float4 values = decodeFp8x4ToFp32<InputT>(packets[lane].value);

                if constexpr ((R & 1) == 0) {
                    k0 = values.x + values.z;
                    k1 = values.y + values.w;
                } else if ((outer_index & 1ULL) == 0) {
                    if (lane == TILED_REDUCTION_WARP_THREADS - 1) {
                        k0 = values.x;
                        k1 = values.y;
                    } else {
                        k0 = values.x + values.z;
                        k1 = values.y + values.w;
                    }
                } else {
                    if (lane == 0) {
                        k0 = values.z;
                        k1 = values.w;
                    } else {
                        k0 = values.x + values.z;
                        k1 = values.y + values.w;
                    }
                }
            }

            for (int offset = TILED_REDUCTION_WARP_THREADS / 2; offset > 0; offset >>= 1) {
                k0 += __shfl_down_sync(FullWarpMask, k0, offset);
                k1 += __shfl_down_sync(FullWarpMask, k1, offset);
            }
            if (lane == 0) {
                warp_results[warp] = valid_output ? make_float2(k0, k1) : make_float2(0.0f, 0.0f);
            }
            __syncthreads();

            // Pair adjacent one-output warps into one aligned p4 output store. If the entire tensor has an odd output
            // count, the final even-indexed output writes its unused high fp8x2 into Thor's physical tensor tail pad;
            // no logical neighboring output is overwritten.
            if (threadIdx.x < SHALLOW_FP8_WARPS / 2) {
                const uint64_t pair = static_cast<uint64_t>(threadIdx.x);
                const uint64_t pair_outer = outer_begin + pair * 2;
                if (pair_outer < outer_size) {
                    const float2 low_values = warp_results[pair * 2];
                    const float2 high_values = pair_outer + 1 < outer_size
                        ? warp_results[pair * 2 + 1]
                        : make_float2(0.0f, 0.0f);
                    uint16_t low_bits;
                    uint16_t high_bits;
                    if constexpr (std::is_same_v<InputT, __nv_fp8_e4m3>) {
                        low_bits = __nv_cvt_float2_to_fp8x2(low_values, __NV_SATFINITE, __NV_E4M3);
                        high_bits = __nv_cvt_float2_to_fp8x2(high_values, __NV_SATFINITE, __NV_E4M3);
                    } else {
                        low_bits = __nv_cvt_float2_to_fp8x2(low_values, __NV_SATFINITE, __NV_E5M2);
                        high_bits = __nv_cvt_float2_to_fp8x2(high_values, __NV_SATFINITE, __NV_E5M2);
                    }
                    RawVectorPacket<SHALLOW_FP8_PACKET_BYTES> packet{};
                    packet.value = static_cast<uint32_t>(low_bits) | (static_cast<uint32_t>(high_bits) << 16);
                    reinterpret_cast<RawVectorPacket<SHALLOW_FP8_PACKET_BYTES>*>(output)[pair_outer / 2] = packet;
                }
            }
            __syncthreads();
        }
    }
}


// K=3 repeats cleanly every four rows: 4 rows * 3 FP8 bytes = 12 bytes = three aligned p4 packets. Each lane loads
// one consecutive p4 packet, canonicalizes its phase locally into (k0,k1,k2) FP32 partials, and only then
// participates in a subwarp/warp reduction. Odd-R outputs may share their aligned boundary packet with a neighbor,
// but no raw values move between lanes and there is no FP32 input staging.
__host__ __device__ constexpr uint32_t k3PhaseLaneMask(int packet_base, int prefix_bytes, int phase) {
    uint32_t mask = 0;
    for (int lane = 0; lane < TILED_REDUCTION_WARP_THREADS; ++lane) {
        int packet_phase = (packet_base + lane) % 3;
        int canonical_phase = packet_phase - (prefix_bytes % 3);
        if (canonical_phase < 0) canonical_phase += 3;
        if (canonical_phase == phase) mask |= uint32_t{1} << lane;
    }
    return mask;
}

template <typename InputT, int PacketBase, int PrefixBytes>
__device__ __forceinline__ void accumulateFp8K3FixedPhasePacket(uint32_t packed,
                                                                 int subgroup_lane,
                                                                 float& k0,
                                                                 float& k1,
                                                                 float& k2) {
    static_assert(PacketBase == 0 || PacketBase == 32);
    static_assert(PrefixBytes >= 0 && PrefixBytes < 4);
    constexpr uint32_t Phase0Mask = k3PhaseLaneMask(PacketBase, PrefixBytes, 0);
    constexpr uint32_t Phase1Mask = k3PhaseLaneMask(PacketBase, PrefixBytes, 1);
    const uint32_t lane_bit = uint32_t{1} << subgroup_lane;
    const float4 decoded = decodeFp8x4ToFp32<InputT>(packed);

    if ((Phase0Mask & lane_bit) != 0) {
        k0 += decoded.x + decoded.w;
        k1 += decoded.y;
        k2 += decoded.z;
    } else if ((Phase1Mask & lane_bit) != 0) {
        k0 += decoded.z;
        k1 += decoded.x + decoded.w;
        k2 += decoded.y;
    } else {
        k0 += decoded.y;
        k1 += decoded.z;
        k2 += decoded.x + decoded.w;
    }
}

template <typename InputT, int PacketBase>
__device__ __forceinline__ void accumulateFp8K3FixedPhasePacketForPrefix(uint32_t packed,
                                                                          int subgroup_lane,
                                                                          int prefix_bytes,
                                                                          float& k0,
                                                                          float& k1,
                                                                          float& k2) {
    switch (prefix_bytes) {
        case 0: accumulateFp8K3FixedPhasePacket<InputT, PacketBase, 0>(packed, subgroup_lane, k0, k1, k2); return;
        case 1: accumulateFp8K3FixedPhasePacket<InputT, PacketBase, 1>(packed, subgroup_lane, k0, k1, k2); return;
        case 2: accumulateFp8K3FixedPhasePacket<InputT, PacketBase, 2>(packed, subgroup_lane, k0, k1, k2); return;
        default: accumulateFp8K3FixedPhasePacket<InputT, PacketBase, 3>(packed, subgroup_lane, k0, k1, k2); return;
    }
}


// Even-R K=3 p8 calibration. Eight K=3 rows occupy exactly 24 bytes == three p8 packets, so each output starts
// p8-aligned and packet phases repeat A,C,B. This candidate intentionally keeps the proven p4 output packing below;
// it isolates whether doubling only the input transaction/decode packet is profitable.
__host__ __device__ constexpr uint32_t k3P8PhaseLaneMask(int phase) {
    uint32_t mask = 0;
    for (int lane = 0; lane < TILED_REDUCTION_WARP_THREADS; ++lane) {
        const int first_component = (2 * lane) % 3;  // 8 bytes advances the K=3 stream by 2 components.
        if (first_component == phase) mask |= uint32_t{1} << lane;
    }
    return mask;
}

template <typename InputT>
__device__ __forceinline__ void accumulateFp8K3P8FixedPhasePacket(uint2 packed,
                                                                  int subgroup_lane,
                                                                  float& k0,
                                                                  float& k1,
                                                                  float& k2) {
    constexpr uint32_t Phase0Mask = k3P8PhaseLaneMask(0);
    constexpr uint32_t Phase1Mask = k3P8PhaseLaneMask(1);
    const uint32_t lane_bit = uint32_t{1} << subgroup_lane;
    const float4 low = decodeFp8x4ToFp32<InputT>(packed.x);
    const float4 high = decodeFp8x4ToFp32<InputT>(packed.y);

    // phase 0: A B C A B C A B
    if ((Phase0Mask & lane_bit) != 0) {
        k0 += low.x + low.w + high.z;
        k1 += low.y + high.x + high.w;
        k2 += low.z + high.y;
    // phase 1: B C A B C A B C
    } else if ((Phase1Mask & lane_bit) != 0) {
        k0 += low.z + high.y;
        k1 += low.x + low.w + high.z;
        k2 += low.y + high.x + high.w;
    // phase 2: C A B C A B C A
    } else {
        k0 += low.y + high.x + high.w;
        k1 += low.z + high.y;
        k2 += low.x + low.w + high.z;
    }
}

template <typename InputT, int R>
__global__ void fp8K3PhaseStreamRParallelP4CompleteKernel(const InputT* input, InputT* output, uint64_t outer_size) {
    static_assert(sizeof(InputT) == 1);
    static_assert(R == 7 || R == 8 || R == 15 || R == 16 || R == 31 || R == 32 || R == 63 || R == 64);
    constexpr int BytesPerOutput = R * 3;
    constexpr int PacketsPerOutput = R <= 8 ? 6 : (R <= 16 ? 12 : (R <= 32 ? 24 : 48));
    constexpr int LanesPerOutput = R <= 8 ? 8 : (R <= 16 ? 16 : 32);
    constexpr int OutputsPerWarp = TILED_REDUCTION_WARP_THREADS / LanesPerOutput;
    constexpr int OutputsPerBlock = SHALLOW_FP8_WARPS * OutputsPerWarp;
    constexpr int OutputPacketsPerBlock = (OutputsPerBlock * 3) / SHALLOW_FP8_PACKET_BYTES;
    constexpr uint32_t FullWarpMask = 0xffffffffu;
    static_assert(OutputsPerWarp == 1 || OutputsPerWarp == 2 || OutputsPerWarp == 4);
    static_assert((OutputsPerBlock * 3) % SHALLOW_FP8_PACKET_BYTES == 0);
    static_assert(PacketsPerOutput <= LanesPerOutput || (PacketsPerOutput == 48 && LanesPerOutput == 32));

    __shared__ float output_values[OutputsPerBlock][3];

    const int warp = static_cast<int>(threadIdx.x) / TILED_REDUCTION_WARP_THREADS;
    const int lane = static_cast<int>(threadIdx.x) % TILED_REDUCTION_WARP_THREADS;
    const int subgroup = lane / LanesPerOutput;
    const int subgroup_lane = lane % LanesPerOutput;
    const uint64_t grid_stride = static_cast<uint64_t>(gridDim.x);

    for (uint64_t block_round = static_cast<uint64_t>(blockIdx.x);; block_round += grid_stride) {
        const uint64_t outer_begin = block_round * static_cast<uint64_t>(OutputsPerBlock);
        if (outer_begin >= outer_size) {
            return;
        }
        const int output_slot = warp * OutputsPerWarp + subgroup;
        const uint64_t outer_index = outer_begin + static_cast<uint64_t>(output_slot);
        const bool valid_output = outer_index < outer_size;
        float k0 = 0.0f;
        float k1 = 0.0f;
        float k2 = 0.0f;

        if (valid_output) {
            // OutputsPerBlock is a multiple of four, so every CTA round starts p4-aligned. Even R therefore has
            // no prefix/tail handling at all. For the odd R values in this calibration (all R == 3 mod 4), each
            // output advances the p4 prefix by exactly one byte, so the prefix is just output_slot mod 4.
            constexpr bool OddR = (R & 1) != 0;
            const int prefix_bytes = OddR ? (output_slot & (SHALLOW_FP8_PACKET_BYTES - 1)) : 0;
            const uint64_t logical_byte_start = outer_index * static_cast<uint64_t>(BytesPerOutput);
            const uint64_t aligned_byte_start = logical_byte_start - static_cast<uint64_t>(prefix_bytes);
            const auto* packets = reinterpret_cast<const RawVectorPacket<SHALLOW_FP8_PACKET_BYTES>*>(input) +
                                  aligned_byte_start / SHALLOW_FP8_PACKET_BYTES;
            constexpr int FirstRoundPackets = PacketsPerOutput < LanesPerOutput ? PacketsPerOutput : LanesPerOutput;

            if (subgroup_lane < FirstRoundPackets) {
                const int packet_index = subgroup_lane;
                uint32_t packed;
                if constexpr (OddR) {
                    // For odd R, the final aligned p4 may cross the tensor end. This is the only load that cannot
                    // safely use the normal p4 path. The useful byte count in that packet is prefix_bytes + 1.
                    if (outer_index + 1 == outer_size && packet_index == PacketsPerOutput - 1 && prefix_bytes < 3) {
                        const auto* bytes = reinterpret_cast<const uint8_t*>(packets + packet_index);
                        packed = static_cast<uint32_t>(bytes[0]);
                        if (prefix_bytes >= 1) packed |= static_cast<uint32_t>(bytes[1]) << 8;
                        if (prefix_bytes >= 2) packed |= static_cast<uint32_t>(bytes[2]) << 16;
                    } else {
                        packed = packets[packet_index].value;
                    }

                    // Zero neighboring-output bytes in the two boundary packets before FP8 decode. FP8 +0 is all
                    // zero bits for both E4M3 and E5M2, so this avoids decoded-float masking in the hot path.
                    if (packet_index == 0) {
                        packed &= 0xffffffffu << (prefix_bytes * 8);
                    }
                    if (packet_index == PacketsPerOutput - 1) {
                        packed &= 0xffffffffu >> ((3 - prefix_bytes) * 8);
                    }
                    accumulateFp8K3FixedPhasePacketForPrefix<InputT, 0>(
                        packed, subgroup_lane, prefix_bytes, k0, k1, k2);
                } else {
                    packed = packets[packet_index].value;
                    accumulateFp8K3FixedPhasePacket<InputT, 0, 0>(packed, subgroup_lane, k0, k1, k2);
                }
            }

            if constexpr (PacketsPerOutput > LanesPerOutput) {
                constexpr int SecondRoundPackets = PacketsPerOutput - LanesPerOutput;
                static_assert(LanesPerOutput == TILED_REDUCTION_WARP_THREADS);
                if (subgroup_lane < SecondRoundPackets) {
                    constexpr int PacketBase = TILED_REDUCTION_WARP_THREADS;
                    const int packet_index = PacketBase + subgroup_lane;
                    uint32_t packed;
                    if constexpr (OddR) {
                        if (outer_index + 1 == outer_size && packet_index == PacketsPerOutput - 1 && prefix_bytes < 3) {
                            const auto* bytes = reinterpret_cast<const uint8_t*>(packets + packet_index);
                            packed = static_cast<uint32_t>(bytes[0]);
                            if (prefix_bytes >= 1) packed |= static_cast<uint32_t>(bytes[1]) << 8;
                            if (prefix_bytes >= 2) packed |= static_cast<uint32_t>(bytes[2]) << 16;
                        } else {
                            packed = packets[packet_index].value;
                        }
                        if (packet_index == PacketsPerOutput - 1) {
                            packed &= 0xffffffffu >> ((3 - prefix_bytes) * 8);
                        }
                        accumulateFp8K3FixedPhasePacketForPrefix<InputT, PacketBase>(
                            packed, subgroup_lane, prefix_bytes, k0, k1, k2);
                    } else {
                        packed = packets[packet_index].value;
                        accumulateFp8K3FixedPhasePacket<InputT, PacketBase, 0>(packed, subgroup_lane, k0, k1, k2);
                    }
                }
            }
        }

        for (int offset = LanesPerOutput / 2; offset > 0; offset >>= 1) {
            k0 += __shfl_down_sync(FullWarpMask, k0, offset, LanesPerOutput);
            k1 += __shfl_down_sync(FullWarpMask, k1, offset, LanesPerOutput);
            k2 += __shfl_down_sync(FullWarpMask, k2, offset, LanesPerOutput);
        }
        if (subgroup_lane == 0) {
            output_values[output_slot][0] = valid_output ? k0 : 0.0f;
            output_values[output_slot][1] = valid_output ? k1 : 0.0f;
            output_values[output_slot][2] = valid_output ? k2 : 0.0f;
        }
        __syncthreads();

        // OutputsPerBlock is a multiple of four, so each CTA round begins on a p4 output boundary. Flatten the final
        // K=3 triples and write four FP8 results per aligned p4. Shared memory is used only for this tiny final
        // writeback packing; the reduction itself remains register/subwarp local.
        if (threadIdx.x < OutputPacketsPerBlock) {
            const int packet = static_cast<int>(threadIdx.x);
            const int scalar0 = packet * SHALLOW_FP8_PACKET_BYTES;
            const uint64_t valid_scalars = (outer_size - outer_begin) * 3 < static_cast<uint64_t>(OutputsPerBlock * 3)
                ? (outer_size - outer_begin) * 3
                : static_cast<uint64_t>(OutputsPerBlock * 3);
            if (static_cast<uint64_t>(scalar0) < valid_scalars) {
                const int output_group = packet / 3;
                const int phase = packet - output_group * 3;
                const int slot = output_group * 4;
                float2 low_values;
                float2 high_values;
                if (phase == 0) {
                    low_values = make_float2(output_values[slot][0], output_values[slot][1]);
                    high_values = make_float2(output_values[slot][2], output_values[slot + 1][0]);
                } else if (phase == 1) {
                    low_values = make_float2(output_values[slot + 1][1], output_values[slot + 1][2]);
                    high_values = make_float2(output_values[slot + 2][0], output_values[slot + 2][1]);
                } else {
                    low_values = make_float2(output_values[slot + 2][2], output_values[slot + 3][0]);
                    high_values = make_float2(output_values[slot + 3][1], output_values[slot + 3][2]);
                }
                uint16_t low_bits;
                uint16_t high_bits;
                if constexpr (std::is_same_v<InputT, __nv_fp8_e4m3>) {
                    low_bits = __nv_cvt_float2_to_fp8x2(low_values, __NV_SATFINITE, __NV_E4M3);
                    high_bits = __nv_cvt_float2_to_fp8x2(high_values, __NV_SATFINITE, __NV_E4M3);
                } else {
                    low_bits = __nv_cvt_float2_to_fp8x2(low_values, __NV_SATFINITE, __NV_E5M2);
                    high_bits = __nv_cvt_float2_to_fp8x2(high_values, __NV_SATFINITE, __NV_E5M2);
                }
                RawVectorPacket<SHALLOW_FP8_PACKET_BYTES> packed_output{};
                packed_output.value = static_cast<uint32_t>(low_bits) | (static_cast<uint32_t>(high_bits) << 16);
                auto* output_bytes = reinterpret_cast<uint8_t*>(output) + outer_begin * 3 + scalar0;
                const uint64_t remaining_scalars = valid_scalars - static_cast<uint64_t>(scalar0);
                if (remaining_scalars >= SHALLOW_FP8_PACKET_BYTES) {
                    *reinterpret_cast<RawVectorPacket<SHALLOW_FP8_PACKET_BYTES>*>(output_bytes) = packed_output;
                } else {
                    // Exact final tensor tail: avoid relying on allocation padding for the last partial p4 store.
                    const auto* packed_bytes = reinterpret_cast<const uint8_t*>(&packed_output.value);
                    if (remaining_scalars >= 1) output_bytes[0] = packed_bytes[0];
                    if (remaining_scalars >= 2) output_bytes[1] = packed_bytes[1];
                    if (remaining_scalars >= 3) output_bytes[2] = packed_bytes[2];
                }
            }
        }
        __syncthreads();
    }
}

template <typename InputT, int R>
void launchFp8K3PhaseStreamRParallelP4(const Tensor& input,
                                       Tensor& output,
                                       const CubReductionGeometry& geometry,
                                       cudaStream_t stream) {
    static_assert(R == 7 || R == 8 || R == 15 || R == 16 || R == 31 || R == 32 || R == 63 || R == 64);
    constexpr uint64_t LanesPerOutput = R <= 8 ? 8 : (R <= 16 ? 16 : 32);
    constexpr uint64_t OutputsPerWarp = TILED_REDUCTION_WARP_THREADS / LanesPerOutput;
    constexpr uint64_t OutputsPerBlock = SHALLOW_FP8_WARPS * OutputsPerWarp;
    const uint64_t block_count = std::min<uint64_t>(
        ceilDivideU64(geometry.outer_size, OutputsPerBlock), NARROW_FLAT_MAX_GRID_BLOCKS);
    fp8K3PhaseStreamRParallelP4CompleteKernel<InputT, R>
        <<<static_cast<unsigned int>(block_count), SHALLOW_FP8_THREADS, 0, stream>>>(
            input.getMemPtr<InputT>(), output.getMemPtr<InputT>(), geometry.outer_size);
    CUDA_CHECK(cudaGetLastError());
}

template <typename InputT, int R>
[[nodiscard]] CubKernelOccupancyInfo queryFp8K3PhaseStreamRParallelP4Occupancy() {
    static_assert(R == 7 || R == 8 || R == 15 || R == 16 || R == 31 || R == 32 || R == 63 || R == 64);
    return queryShallowFp8Occupancy(fp8K3PhaseStreamRParallelP4CompleteKernel<InputT, R>);
}

template <typename InputT, int R>
__global__ void fp8K3PhaseStreamRParallelP8InputCompleteKernel(const InputT* input, InputT* output, uint64_t outer_size) {
    static_assert(sizeof(InputT) == 1);
    static_assert(R == 8 || R == 16 || R == 32 || R == 64);
    constexpr int InputPacketBytes = 8;
    constexpr int BytesPerOutput = R * 3;
    constexpr int PacketsPerOutput = BytesPerOutput / InputPacketBytes;
    constexpr int LanesPerOutput = R <= 8 ? 4 : (R <= 16 ? 8 : (R <= 32 ? 16 : 32));
    constexpr int OutputsPerWarp = TILED_REDUCTION_WARP_THREADS / LanesPerOutput;
    constexpr int OutputsPerBlock = SHALLOW_FP8_WARPS * OutputsPerWarp;
    constexpr int OutputPacketsPerBlock = (OutputsPerBlock * 3) / SHALLOW_FP8_PACKET_BYTES;
    constexpr uint32_t FullWarpMask = 0xffffffffu;
    static_assert(BytesPerOutput % InputPacketBytes == 0);
    static_assert(PacketsPerOutput == (R / 8) * 3);
    static_assert(PacketsPerOutput < LanesPerOutput);
    static_assert(OutputsPerWarp == 1 || OutputsPerWarp == 2 || OutputsPerWarp == 4 || OutputsPerWarp == 8);
    static_assert((OutputsPerBlock * 3) % SHALLOW_FP8_PACKET_BYTES == 0);

    __shared__ float output_values[OutputsPerBlock][3];

    const int warp = static_cast<int>(threadIdx.x) / TILED_REDUCTION_WARP_THREADS;
    const int lane = static_cast<int>(threadIdx.x) % TILED_REDUCTION_WARP_THREADS;
    const int subgroup = lane / LanesPerOutput;
    const int subgroup_lane = lane % LanesPerOutput;
    const uint64_t grid_stride = static_cast<uint64_t>(gridDim.x);

    for (uint64_t block_round = static_cast<uint64_t>(blockIdx.x);; block_round += grid_stride) {
        const uint64_t outer_begin = block_round * static_cast<uint64_t>(OutputsPerBlock);
        if (outer_begin >= outer_size) return;

        const int output_slot = warp * OutputsPerWarp + subgroup;
        const uint64_t outer_index = outer_begin + static_cast<uint64_t>(output_slot);
        const bool valid_output = outer_index < outer_size;
        float k0 = 0.0f;
        float k1 = 0.0f;
        float k2 = 0.0f;

        if (valid_output && subgroup_lane < PacketsPerOutput) {
            const auto* packets = reinterpret_cast<const RawVectorPacket<InputPacketBytes>*>(input) +
                                  outer_index * static_cast<uint64_t>(PacketsPerOutput);
            const uint2 packed = packets[subgroup_lane].value;
            accumulateFp8K3P8FixedPhasePacket<InputT>(packed, subgroup_lane, k0, k1, k2);
        }

        for (int offset = LanesPerOutput / 2; offset > 0; offset >>= 1) {
            k0 += __shfl_down_sync(FullWarpMask, k0, offset, LanesPerOutput);
            k1 += __shfl_down_sync(FullWarpMask, k1, offset, LanesPerOutput);
            k2 += __shfl_down_sync(FullWarpMask, k2, offset, LanesPerOutput);
        }
        if (subgroup_lane == 0) {
            output_values[output_slot][0] = valid_output ? k0 : 0.0f;
            output_values[output_slot][1] = valid_output ? k1 : 0.0f;
            output_values[output_slot][2] = valid_output ? k2 : 0.0f;
        }
        __syncthreads();

        // Deliberately retain the established p4 writeback so this A/B changes only the input packet width and its
        // natural subgroup width. The output path is byte-for-byte the same packing pattern as the p4 phase stream.
        if (threadIdx.x < OutputPacketsPerBlock) {
            const int packet = static_cast<int>(threadIdx.x);
            const int scalar0 = packet * SHALLOW_FP8_PACKET_BYTES;
            const uint64_t valid_scalars = (outer_size - outer_begin) * 3 < static_cast<uint64_t>(OutputsPerBlock * 3)
                ? (outer_size - outer_begin) * 3
                : static_cast<uint64_t>(OutputsPerBlock * 3);
            if (static_cast<uint64_t>(scalar0) < valid_scalars) {
                const int output_group = packet / 3;
                const int phase = packet - output_group * 3;
                const int slot = output_group * 4;
                float2 low_values;
                float2 high_values;
                if (phase == 0) {
                    low_values = make_float2(output_values[slot][0], output_values[slot][1]);
                    high_values = make_float2(output_values[slot][2], output_values[slot + 1][0]);
                } else if (phase == 1) {
                    low_values = make_float2(output_values[slot + 1][1], output_values[slot + 1][2]);
                    high_values = make_float2(output_values[slot + 2][0], output_values[slot + 2][1]);
                } else {
                    low_values = make_float2(output_values[slot + 2][2], output_values[slot + 3][0]);
                    high_values = make_float2(output_values[slot + 3][1], output_values[slot + 3][2]);
                }
                uint16_t low_bits;
                uint16_t high_bits;
                if constexpr (std::is_same_v<InputT, __nv_fp8_e4m3>) {
                    low_bits = __nv_cvt_float2_to_fp8x2(low_values, __NV_SATFINITE, __NV_E4M3);
                    high_bits = __nv_cvt_float2_to_fp8x2(high_values, __NV_SATFINITE, __NV_E4M3);
                } else {
                    low_bits = __nv_cvt_float2_to_fp8x2(low_values, __NV_SATFINITE, __NV_E5M2);
                    high_bits = __nv_cvt_float2_to_fp8x2(high_values, __NV_SATFINITE, __NV_E5M2);
                }
                RawVectorPacket<SHALLOW_FP8_PACKET_BYTES> packed_output{};
                packed_output.value = static_cast<uint32_t>(low_bits) | (static_cast<uint32_t>(high_bits) << 16);
                auto* output_bytes = reinterpret_cast<uint8_t*>(output) + outer_begin * 3 + scalar0;
                const uint64_t remaining_scalars = valid_scalars - static_cast<uint64_t>(scalar0);
                if (remaining_scalars >= SHALLOW_FP8_PACKET_BYTES) {
                    *reinterpret_cast<RawVectorPacket<SHALLOW_FP8_PACKET_BYTES>*>(output_bytes) = packed_output;
                } else {
                    const auto* packed_bytes = reinterpret_cast<const uint8_t*>(&packed_output.value);
                    if (remaining_scalars >= 1) output_bytes[0] = packed_bytes[0];
                    if (remaining_scalars >= 2) output_bytes[1] = packed_bytes[1];
                    if (remaining_scalars >= 3) output_bytes[2] = packed_bytes[2];
                }
            }
        }
        __syncthreads();
    }
}

template <typename InputT, int R>
void launchFp8K3PhaseStreamRParallelP8Input(const Tensor& input,
                                            Tensor& output,
                                            const CubReductionGeometry& geometry,
                                            cudaStream_t stream) {
    static_assert(R == 8 || R == 16 || R == 32 || R == 64);
    constexpr uint64_t LanesPerOutput = R <= 8 ? 4 : (R <= 16 ? 8 : (R <= 32 ? 16 : 32));
    constexpr uint64_t OutputsPerWarp = TILED_REDUCTION_WARP_THREADS / LanesPerOutput;
    constexpr uint64_t OutputsPerBlock = SHALLOW_FP8_WARPS * OutputsPerWarp;
    const uint64_t block_count = std::min<uint64_t>(
        ceilDivideU64(geometry.outer_size, OutputsPerBlock), NARROW_FLAT_MAX_GRID_BLOCKS);
    fp8K3PhaseStreamRParallelP8InputCompleteKernel<InputT, R>
        <<<static_cast<unsigned int>(block_count), SHALLOW_FP8_THREADS, 0, stream>>>(
            input.getMemPtr<InputT>(), output.getMemPtr<InputT>(), geometry.outer_size);
    CUDA_CHECK(cudaGetLastError());
}

template <typename InputT, int R>
[[nodiscard]] CubKernelOccupancyInfo queryFp8K3PhaseStreamRParallelP8InputOccupancy() {
    static_assert(R == 8 || R == 16 || R == 32 || R == 64);
    return queryShallowFp8Occupancy(fp8K3PhaseStreamRParallelP8InputCompleteKernel<InputT, R>);
}


// Benchmark-only K=3 runtime-R phase-stream experiment. Unlike the calibrated kernels above, R is not a template
// dimension. PacketBytes is the only physical phase-stream template dimension (besides FP8 dtype). A uniform runtime
// choice selects one of four compile-time subgroup widths, so shuffle geometry remains constant inside each body while
// arbitrary R values share the same compiled kernel. Misaligned outputs use aligned packet loads with only their first
// and last packet masked; naturally aligned outputs take the direct-load fast path.
template <int PacketBytes>
[[nodiscard]] __host__ __device__ constexpr uint64_t k3RuntimeRMaxPacketsPerOutput(uint64_t reduction_size) {
    static_assert(PacketBytes == 4 || PacketBytes == 8);
    const uint64_t bytes_per_output = reduction_size * 3;
    // Worst possible aligned-prefix is PacketBytes-1. ceil((bytes + prefix) / PacketBytes).
    return (bytes_per_output + 2 * PacketBytes - 2) / PacketBytes;
}

template <int PacketBytes>
[[nodiscard]] __host__ __device__ constexpr int k3RuntimeRLanesPerOutput(uint64_t reduction_size) {
    const uint64_t max_packets = k3RuntimeRMaxPacketsPerOutput<PacketBytes>(reduction_size);
    if (max_packets <= 4) return 4;
    if (max_packets <= 8) return 8;
    if (max_packets <= 16) return 16;
    return 32;
}

template <typename InputT>
__device__ __forceinline__ void accumulateFp8K3P4RuntimePhasePacket(uint32_t packed,
                                                                    int phase,
                                                                    float& k0,
                                                                    float& k1,
                                                                    float& k2) {
    const float4 decoded = decodeFp8x4ToFp32<InputT>(packed);
    if (phase == 0) {
        k0 += decoded.x + decoded.w;
        k1 += decoded.y;
        k2 += decoded.z;
    } else if (phase == 1) {
        k0 += decoded.z;
        k1 += decoded.x + decoded.w;
        k2 += decoded.y;
    } else {
        k0 += decoded.y;
        k1 += decoded.z;
        k2 += decoded.x + decoded.w;
    }
}

template <typename InputT>
__device__ __forceinline__ void accumulateFp8K3P8RuntimePhasePacket(uint2 packed,
                                                                    int phase,
                                                                    float& k0,
                                                                    float& k1,
                                                                    float& k2) {
    const float4 low = decodeFp8x4ToFp32<InputT>(packed.x);
    const float4 high = decodeFp8x4ToFp32<InputT>(packed.y);
    if (phase == 0) {
        k0 += low.x + low.w + high.z;
        k1 += low.y + high.x + high.w;
        k2 += low.z + high.y;
    } else if (phase == 1) {
        k0 += low.z + high.y;
        k1 += low.x + low.w + high.z;
        k2 += low.y + high.x + high.w;
    } else {
        k0 += low.y + high.x + high.w;
        k1 += low.z + high.y;
        k2 += low.x + low.w + high.z;
    }
}

template <int PacketBase, int PrefixMod3>
[[nodiscard]] __host__ __device__ constexpr uint32_t k3RuntimeP8PhaseLaneMask(int phase) {
    static_assert(PacketBase == 0 || PacketBase == 32);
    static_assert(PrefixMod3 >= 0 && PrefixMod3 < 3);
    uint32_t mask = 0;
    for (int lane = 0; lane < TILED_REDUCTION_WARP_THREADS; ++lane) {
        int canonical_phase = (2 * (PacketBase + lane) - PrefixMod3) % 3;
        if (canonical_phase < 0) canonical_phase += 3;
        if (canonical_phase == phase) mask |= uint32_t{1} << lane;
    }
    return mask;
}

template <typename InputT, int PacketBase, int PrefixMod3>
__device__ __forceinline__ void accumulateFp8K3P8FixedPhasePacketForPrefixMod(uint2 packed,
                                                                              int subgroup_lane,
                                                                              float& k0,
                                                                              float& k1,
                                                                              float& k2) {
    constexpr uint32_t Phase0Mask = k3RuntimeP8PhaseLaneMask<PacketBase, PrefixMod3>(0);
    constexpr uint32_t Phase1Mask = k3RuntimeP8PhaseLaneMask<PacketBase, PrefixMod3>(1);
    const uint32_t lane_bit = uint32_t{1} << subgroup_lane;
    const float4 low = decodeFp8x4ToFp32<InputT>(packed.x);
    const float4 high = decodeFp8x4ToFp32<InputT>(packed.y);
    if ((Phase0Mask & lane_bit) != 0) {
        k0 += low.x + low.w + high.z;
        k1 += low.y + high.x + high.w;
        k2 += low.z + high.y;
    } else if ((Phase1Mask & lane_bit) != 0) {
        k0 += low.z + high.y;
        k1 += low.x + low.w + high.z;
        k2 += low.y + high.x + high.w;
    } else {
        k0 += low.y + high.x + high.w;
        k1 += low.z + high.y;
        k2 += low.x + low.w + high.z;
    }
}

template <typename InputT, int PacketBase>
__device__ __forceinline__ void accumulateFp8K3P8FixedPhasePacketForPrefix(uint2 packed,
                                                                           int subgroup_lane,
                                                                           int prefix_bytes,
                                                                           float& k0,
                                                                           float& k1,
                                                                           float& k2) {
    switch (prefix_bytes % 3) {
        case 0:
            accumulateFp8K3P8FixedPhasePacketForPrefixMod<InputT, PacketBase, 0>(
                packed, subgroup_lane, k0, k1, k2);
            return;
        case 1:
            accumulateFp8K3P8FixedPhasePacketForPrefixMod<InputT, PacketBase, 1>(
                packed, subgroup_lane, k0, k1, k2);
            return;
        default:
            accumulateFp8K3P8FixedPhasePacketForPrefixMod<InputT, PacketBase, 2>(
                packed, subgroup_lane, k0, k1, k2);
            return;
    }
}

template <int PacketBytes>
__device__ __forceinline__ RawVectorPacket<PacketBytes> loadFp8K3RuntimeRMaskedPacket(const uint8_t* input_bytes,
                                                                                       uint64_t packet_byte_start,
                                                                                       uint64_t total_input_bytes,
                                                                                       bool first_packet,
                                                                                       bool last_packet,
                                                                                       int prefix_bytes,
                                                                                       int last_valid_bytes) {
    static_assert(PacketBytes == 4 || PacketBytes == 8);
    RawVectorPacket<PacketBytes> packet{};
    if (packet_byte_start + PacketBytes <= total_input_bytes) {
        packet = *reinterpret_cast<const RawVectorPacket<PacketBytes>*>(input_bytes + packet_byte_start);
    } else {
        auto* packet_bytes = reinterpret_cast<uint8_t*>(&packet.value);
#pragma unroll
        for (int byte = 0; byte < PacketBytes; ++byte) {
            if (packet_byte_start + static_cast<uint64_t>(byte) < total_input_bytes) {
                packet_bytes[byte] = input_bytes[packet_byte_start + static_cast<uint64_t>(byte)];
            }
        }
    }

    auto* packet_bytes = reinterpret_cast<uint8_t*>(&packet.value);
    if (first_packet) {
#pragma unroll
        for (int byte = 0; byte < PacketBytes; ++byte) {
            if (byte < prefix_bytes) packet_bytes[byte] = 0;
        }
    }
    if (last_packet) {
#pragma unroll
        for (int byte = 0; byte < PacketBytes; ++byte) {
            if (byte >= last_valid_bytes) packet_bytes[byte] = 0;
        }
    }
    return packet;
}

template <int PacketBytes>
[[nodiscard]] __device__ __forceinline__ int k3RuntimeRCanonicalPhase(uint64_t packet_index, int prefix_bytes) {
    static_assert(PacketBytes == 4 || PacketBytes == 8);
    const int packet_mod3 = static_cast<int>(packet_index % 3);
    int phase = ((PacketBytes % 3) * packet_mod3) - (prefix_bytes % 3);
    phase %= 3;
    if (phase < 0) phase += 3;
    return phase;
}

template <typename InputT, int PacketBytes>
__device__ __forceinline__ void accumulateFp8K3RuntimeRPacket(const RawVectorPacket<PacketBytes>& packet,
                                                               int phase,
                                                               float& k0,
                                                               float& k1,
                                                               float& k2) {
    static_assert(PacketBytes == 4 || PacketBytes == 8);
    if constexpr (PacketBytes == 4) {
        accumulateFp8K3P4RuntimePhasePacket<InputT>(packet.value, phase, k0, k1, k2);
    } else {
        accumulateFp8K3P8RuntimePhasePacket<InputT>(packet.value, phase, k0, k1, k2);
    }
}

template <typename InputT, int PacketBytes, int LanesPerOutput>
__device__ __forceinline__ void runFp8K3RuntimeRPhaseStream(const InputT* input,
                                                            InputT* output,
                                                            uint64_t outer_size,
                                                            uint64_t reduction_size,
                                                            float (*output_values)[3]) {
    static_assert(PacketBytes == 4 || PacketBytes == 8);
    static_assert(LanesPerOutput == 4 || LanesPerOutput == 8 || LanesPerOutput == 16 || LanesPerOutput == 32);
    constexpr int OutputsPerWarp = TILED_REDUCTION_WARP_THREADS / LanesPerOutput;
    constexpr int OutputsPerBlock = SHALLOW_FP8_WARPS * OutputsPerWarp;
    constexpr int OutputPacketsPerBlock = (OutputsPerBlock * 3) / SHALLOW_FP8_PACKET_BYTES;
    constexpr uint32_t FullWarpMask = 0xffffffffu;
    static_assert((OutputsPerBlock * 3) % SHALLOW_FP8_PACKET_BYTES == 0);

    const uint64_t bytes_per_output = reduction_size * 3;
    const uint64_t total_input_bytes = outer_size * bytes_per_output;
    const bool naturally_aligned = (bytes_per_output & static_cast<uint64_t>(PacketBytes - 1)) == 0;
    const auto* input_bytes = reinterpret_cast<const uint8_t*>(input);
    const int warp = static_cast<int>(threadIdx.x) / TILED_REDUCTION_WARP_THREADS;
    const int lane = static_cast<int>(threadIdx.x) % TILED_REDUCTION_WARP_THREADS;
    const int subgroup = lane / LanesPerOutput;
    const int subgroup_lane = lane % LanesPerOutput;
    const uint64_t grid_stride = static_cast<uint64_t>(gridDim.x);

    for (uint64_t block_round = static_cast<uint64_t>(blockIdx.x);; block_round += grid_stride) {
        const uint64_t outer_begin = block_round * static_cast<uint64_t>(OutputsPerBlock);
        if (outer_begin >= outer_size) return;

        const int output_slot = warp * OutputsPerWarp + subgroup;
        const uint64_t outer_index = outer_begin + static_cast<uint64_t>(output_slot);
        const bool valid_output = outer_index < outer_size;
        float k0 = 0.0f;
        float k1 = 0.0f;
        float k2 = 0.0f;

        if (valid_output) {
            const uint64_t logical_byte_start = outer_index * bytes_per_output;
            const int prefix_bytes = naturally_aligned
                ? 0
                : static_cast<int>(logical_byte_start & static_cast<uint64_t>(PacketBytes - 1));
            const uint64_t aligned_byte_start = logical_byte_start - static_cast<uint64_t>(prefix_bytes);
            const uint64_t packet_count =
                (static_cast<uint64_t>(prefix_bytes) + bytes_per_output + PacketBytes - 1) / PacketBytes;
            const int last_valid_bytes = static_cast<int>(
                static_cast<uint64_t>(prefix_bytes) + bytes_per_output - (packet_count - 1) * PacketBytes);

            if (subgroup_lane < static_cast<int>(packet_count < LanesPerOutput ? packet_count : LanesPerOutput)) {
                const uint64_t packet_index = static_cast<uint64_t>(subgroup_lane);
                RawVectorPacket<PacketBytes> packet;
                if (naturally_aligned) {
                    packet = *reinterpret_cast<const RawVectorPacket<PacketBytes>*>(
                        input_bytes + aligned_byte_start + packet_index * PacketBytes);
                    if constexpr (PacketBytes == 4) {
                        accumulateFp8K3FixedPhasePacket<InputT, 0, 0>(
                            packet.value, subgroup_lane, k0, k1, k2);
                    } else {
                        accumulateFp8K3P8FixedPhasePacket<InputT>(packet.value, subgroup_lane, k0, k1, k2);
                    }
                } else {
                    packet = loadFp8K3RuntimeRMaskedPacket<PacketBytes>(
                        input_bytes,
                        aligned_byte_start + packet_index * PacketBytes,
                        total_input_bytes,
                        packet_index == 0,
                        packet_index + 1 == packet_count,
                        prefix_bytes,
                        last_valid_bytes);
                    if constexpr (PacketBytes == 4) {
                        accumulateFp8K3FixedPhasePacketForPrefix<InputT, 0>(
                            packet.value, subgroup_lane, prefix_bytes, k0, k1, k2);
                    } else {
                        accumulateFp8K3P8FixedPhasePacketForPrefix<InputT, 0>(
                            packet.value, subgroup_lane, prefix_bytes, k0, k1, k2);
                    }
                }
            }

            // The selected subgroup width is the next power of two through 16 lanes, then a full warp. Therefore a
            // second packet round can occur only for the 32-lane body. Keep that common deep-R case phase-specialized.
            if constexpr (LanesPerOutput == 32) {
                if (packet_count > 32 && subgroup_lane < static_cast<int>((packet_count - 32) < 32 ? (packet_count - 32) : 32)) {
                    constexpr uint64_t PacketBase = 32;
                    const uint64_t packet_index = PacketBase + static_cast<uint64_t>(subgroup_lane);
                    RawVectorPacket<PacketBytes> packet;
                    if (naturally_aligned) {
                        packet = *reinterpret_cast<const RawVectorPacket<PacketBytes>*>(
                            input_bytes + aligned_byte_start + packet_index * PacketBytes);
                        if constexpr (PacketBytes == 4) {
                            accumulateFp8K3FixedPhasePacket<InputT, 32, 0>(
                                packet.value, subgroup_lane, k0, k1, k2);
                        } else {
                            accumulateFp8K3P8FixedPhasePacketForPrefix<InputT, 32>(
                                packet.value, subgroup_lane, 0, k0, k1, k2);
                        }
                    } else {
                        packet = loadFp8K3RuntimeRMaskedPacket<PacketBytes>(
                            input_bytes,
                            aligned_byte_start + packet_index * PacketBytes,
                            total_input_bytes,
                            false,
                            packet_index + 1 == packet_count,
                            prefix_bytes,
                            last_valid_bytes);
                        if constexpr (PacketBytes == 4) {
                            accumulateFp8K3FixedPhasePacketForPrefix<InputT, 32>(
                                packet.value, subgroup_lane, prefix_bytes, k0, k1, k2);
                        } else {
                            accumulateFp8K3P8FixedPhasePacketForPrefix<InputT, 32>(
                                packet.value, subgroup_lane, prefix_bytes, k0, k1, k2);
                        }
                    }
                }

                // Functional fallback for arbitrarily large R. It is deliberately outside the <=64 calibration hot
                // path; if this experiment succeeds we can later decide whether very deep R deserves another family.
                for (uint64_t packet_index = 64 + static_cast<uint64_t>(subgroup_lane);
                     packet_index < packet_count;
                     packet_index += 32) {
                    RawVectorPacket<PacketBytes> packet;
                    if (naturally_aligned) {
                        packet = *reinterpret_cast<const RawVectorPacket<PacketBytes>*>(
                            input_bytes + aligned_byte_start + packet_index * PacketBytes);
                    } else {
                        packet = loadFp8K3RuntimeRMaskedPacket<PacketBytes>(
                            input_bytes,
                            aligned_byte_start + packet_index * PacketBytes,
                            total_input_bytes,
                            false,
                            packet_index + 1 == packet_count,
                            prefix_bytes,
                            last_valid_bytes);
                    }
                    const int phase = k3RuntimeRCanonicalPhase<PacketBytes>(packet_index, prefix_bytes);
                    accumulateFp8K3RuntimeRPacket<InputT, PacketBytes>(packet, phase, k0, k1, k2);
                }
            }
        }

#pragma unroll
        for (int offset = LanesPerOutput / 2; offset > 0; offset >>= 1) {
            k0 += __shfl_down_sync(FullWarpMask, k0, offset, LanesPerOutput);
            k1 += __shfl_down_sync(FullWarpMask, k1, offset, LanesPerOutput);
            k2 += __shfl_down_sync(FullWarpMask, k2, offset, LanesPerOutput);
        }
        if (subgroup_lane == 0) {
            output_values[output_slot][0] = valid_output ? k0 : 0.0f;
            output_values[output_slot][1] = valid_output ? k1 : 0.0f;
            output_values[output_slot][2] = valid_output ? k2 : 0.0f;
        }
        __syncthreads();

        if (threadIdx.x < OutputPacketsPerBlock) {
            const int packet = static_cast<int>(threadIdx.x);
            const int scalar0 = packet * SHALLOW_FP8_PACKET_BYTES;
            const uint64_t valid_scalars = (outer_size - outer_begin) * 3 < static_cast<uint64_t>(OutputsPerBlock * 3)
                ? (outer_size - outer_begin) * 3
                : static_cast<uint64_t>(OutputsPerBlock * 3);
            if (static_cast<uint64_t>(scalar0) < valid_scalars) {
                const int output_group = packet / 3;
                const int phase = packet - output_group * 3;
                const int slot = output_group * 4;
                float2 low_values;
                float2 high_values;
                if (phase == 0) {
                    low_values = make_float2(output_values[slot][0], output_values[slot][1]);
                    high_values = make_float2(output_values[slot][2], output_values[slot + 1][0]);
                } else if (phase == 1) {
                    low_values = make_float2(output_values[slot + 1][1], output_values[slot + 1][2]);
                    high_values = make_float2(output_values[slot + 2][0], output_values[slot + 2][1]);
                } else {
                    low_values = make_float2(output_values[slot + 2][2], output_values[slot + 3][0]);
                    high_values = make_float2(output_values[slot + 3][1], output_values[slot + 3][2]);
                }
                uint16_t low_bits;
                uint16_t high_bits;
                if constexpr (std::is_same_v<InputT, __nv_fp8_e4m3>) {
                    low_bits = __nv_cvt_float2_to_fp8x2(low_values, __NV_SATFINITE, __NV_E4M3);
                    high_bits = __nv_cvt_float2_to_fp8x2(high_values, __NV_SATFINITE, __NV_E4M3);
                } else {
                    low_bits = __nv_cvt_float2_to_fp8x2(low_values, __NV_SATFINITE, __NV_E5M2);
                    high_bits = __nv_cvt_float2_to_fp8x2(high_values, __NV_SATFINITE, __NV_E5M2);
                }
                RawVectorPacket<SHALLOW_FP8_PACKET_BYTES> packed_output{};
                packed_output.value = static_cast<uint32_t>(low_bits) | (static_cast<uint32_t>(high_bits) << 16);
                auto* output_bytes = reinterpret_cast<uint8_t*>(output) + outer_begin * 3 + scalar0;
                const uint64_t remaining_scalars = valid_scalars - static_cast<uint64_t>(scalar0);
                if (remaining_scalars >= SHALLOW_FP8_PACKET_BYTES) {
                    *reinterpret_cast<RawVectorPacket<SHALLOW_FP8_PACKET_BYTES>*>(output_bytes) = packed_output;
                } else {
                    const auto* packed_bytes = reinterpret_cast<const uint8_t*>(&packed_output.value);
                    if (remaining_scalars >= 1) output_bytes[0] = packed_bytes[0];
                    if (remaining_scalars >= 2) output_bytes[1] = packed_bytes[1];
                    if (remaining_scalars >= 3) output_bytes[2] = packed_bytes[2];
                }
            }
        }
        __syncthreads();
    }
}

template <typename InputT, int PacketBytes>
__global__ void fp8K3RuntimeRPhaseStreamCompleteKernel(const InputT* input,
                                                        InputT* output,
                                                        uint64_t outer_size,
                                                        uint64_t reduction_size) {
    static_assert(sizeof(InputT) == 1);
    static_assert(PacketBytes == 4 || PacketBytes == 8);
    __shared__ float output_values[SHALLOW_FP8_WARPS * 8][3];

    const int lanes_per_output = k3RuntimeRLanesPerOutput<PacketBytes>(reduction_size);
    if (lanes_per_output == 4) {
        runFp8K3RuntimeRPhaseStream<InputT, PacketBytes, 4>(
            input, output, outer_size, reduction_size, output_values);
    } else if (lanes_per_output == 8) {
        runFp8K3RuntimeRPhaseStream<InputT, PacketBytes, 8>(
            input, output, outer_size, reduction_size, output_values);
    } else if (lanes_per_output == 16) {
        runFp8K3RuntimeRPhaseStream<InputT, PacketBytes, 16>(
            input, output, outer_size, reduction_size, output_values);
    } else {
        runFp8K3RuntimeRPhaseStream<InputT, PacketBytes, 32>(
            input, output, outer_size, reduction_size, output_values);
    }
}

template <typename InputT, int PacketBytes>
void launchFp8K3RuntimeRPhaseStream(const Tensor& input,
                                    Tensor& output,
                                    const CubReductionGeometry& geometry,
                                    cudaStream_t stream) {
    static_assert(PacketBytes == 4 || PacketBytes == 8);
    const uint64_t lanes_per_output = static_cast<uint64_t>(
        k3RuntimeRLanesPerOutput<PacketBytes>(geometry.reduction_size));
    const uint64_t outputs_per_warp = TILED_REDUCTION_WARP_THREADS / lanes_per_output;
    const uint64_t outputs_per_block = SHALLOW_FP8_WARPS * outputs_per_warp;
    const uint64_t block_count = std::min<uint64_t>(
        ceilDivideU64(geometry.outer_size, outputs_per_block), NARROW_FLAT_MAX_GRID_BLOCKS);
    fp8K3RuntimeRPhaseStreamCompleteKernel<InputT, PacketBytes>
        <<<static_cast<unsigned int>(block_count), SHALLOW_FP8_THREADS, 0, stream>>>(
            input.getMemPtr<InputT>(), output.getMemPtr<InputT>(), geometry.outer_size, geometry.reduction_size);
    CUDA_CHECK(cudaGetLastError());
}

template <typename InputT, int PacketBytes>
[[nodiscard]] CubKernelOccupancyInfo queryFp8K3RuntimeRPhaseStreamOccupancy() {
    static_assert(PacketBytes == 4 || PacketBytes == 8);
    return queryShallowFp8Occupancy(fp8K3RuntimeRPhaseStreamCompleteKernel<InputT, PacketBytes>);
}


// Benchmark-only K=7 phase-stream candidates. As with the K=3 specialization, consecutive packet loads stay
// physically contiguous and are canonicalized lane-locally into K=7 FP32 partials before subgroup reduction. The
// p8 candidate is naturally aligned for R={8,16,32,64}; p16 is naturally aligned for R={16,32,64}. Both retain the
// established p4 output path so the calibration isolates input packet width and the resulting subgroup geometry.
template <int Phase>
__device__ __forceinline__ void accumulateFp8K7EightValues(const float4& low,
                                                            const float4& high,
                                                            float& k0,
                                                            float& k1,
                                                            float& k2,
                                                            float& k3,
                                                            float& k4,
                                                            float& k5,
                                                            float& k6) {
    static_assert(Phase >= 0 && Phase < 7);
    if constexpr (Phase == 0) {
        k0 += low.x + high.w; k1 += low.y; k2 += low.z; k3 += low.w; k4 += high.x; k5 += high.y; k6 += high.z;
    } else if constexpr (Phase == 1) {
        k1 += low.x + high.w; k2 += low.y; k3 += low.z; k4 += low.w; k5 += high.x; k6 += high.y; k0 += high.z;
    } else if constexpr (Phase == 2) {
        k2 += low.x + high.w; k3 += low.y; k4 += low.z; k5 += low.w; k6 += high.x; k0 += high.y; k1 += high.z;
    } else if constexpr (Phase == 3) {
        k3 += low.x + high.w; k4 += low.y; k5 += low.z; k6 += low.w; k0 += high.x; k1 += high.y; k2 += high.z;
    } else if constexpr (Phase == 4) {
        k4 += low.x + high.w; k5 += low.y; k6 += low.z; k0 += low.w; k1 += high.x; k2 += high.y; k3 += high.z;
    } else if constexpr (Phase == 5) {
        k5 += low.x + high.w; k6 += low.y; k0 += low.z; k1 += low.w; k2 += high.x; k3 += high.y; k4 += high.z;
    } else {
        k6 += low.x + high.w; k0 += low.y; k1 += low.z; k2 += low.w; k3 += high.x; k4 += high.y; k5 += high.z;
    }
}

template <int PhaseStep, int PacketBase, int PhaseOffset>
__host__ __device__ constexpr uint32_t k7PhaseLaneMask(int phase) {
    uint32_t mask = 0;
    for (int lane = 0; lane < TILED_REDUCTION_WARP_THREADS; ++lane) {
        const int first_component = (PhaseStep * (PacketBase + lane) + PhaseOffset) % 7;
        if (first_component == phase) mask |= uint32_t{1} << lane;
    }
    return mask;
}

template <int PhaseStep, int PacketBase, int PhaseOffset>
__device__ __forceinline__ void accumulateFp8K7EightValuesForLane(const float4& low,
                                                                   const float4& high,
                                                                   int subgroup_lane,
                                                                   float& k0,
                                                                   float& k1,
                                                                   float& k2,
                                                                   float& k3,
                                                                   float& k4,
                                                                   float& k5,
                                                                   float& k6) {
    constexpr uint32_t Phase0Mask = k7PhaseLaneMask<PhaseStep, PacketBase, PhaseOffset>(0);
    constexpr uint32_t Phase1Mask = k7PhaseLaneMask<PhaseStep, PacketBase, PhaseOffset>(1);
    constexpr uint32_t Phase2Mask = k7PhaseLaneMask<PhaseStep, PacketBase, PhaseOffset>(2);
    constexpr uint32_t Phase3Mask = k7PhaseLaneMask<PhaseStep, PacketBase, PhaseOffset>(3);
    constexpr uint32_t Phase4Mask = k7PhaseLaneMask<PhaseStep, PacketBase, PhaseOffset>(4);
    constexpr uint32_t Phase5Mask = k7PhaseLaneMask<PhaseStep, PacketBase, PhaseOffset>(5);
    const uint32_t lane_bit = uint32_t{1} << subgroup_lane;

    if ((Phase0Mask & lane_bit) != 0) {
        accumulateFp8K7EightValues<0>(low, high, k0, k1, k2, k3, k4, k5, k6);
    } else if ((Phase1Mask & lane_bit) != 0) {
        accumulateFp8K7EightValues<1>(low, high, k0, k1, k2, k3, k4, k5, k6);
    } else if ((Phase2Mask & lane_bit) != 0) {
        accumulateFp8K7EightValues<2>(low, high, k0, k1, k2, k3, k4, k5, k6);
    } else if ((Phase3Mask & lane_bit) != 0) {
        accumulateFp8K7EightValues<3>(low, high, k0, k1, k2, k3, k4, k5, k6);
    } else if ((Phase4Mask & lane_bit) != 0) {
        accumulateFp8K7EightValues<4>(low, high, k0, k1, k2, k3, k4, k5, k6);
    } else if ((Phase5Mask & lane_bit) != 0) {
        accumulateFp8K7EightValues<5>(low, high, k0, k1, k2, k3, k4, k5, k6);
    } else {
        accumulateFp8K7EightValues<6>(low, high, k0, k1, k2, k3, k4, k5, k6);
    }
}

template <typename InputT, int PacketBytes, int PacketBase>
__device__ __forceinline__ void accumulateFp8K7PhasePacket(const RawVectorPacket<PacketBytes>& packet,
                                                            int subgroup_lane,
                                                            float& k0,
                                                            float& k1,
                                                            float& k2,
                                                            float& k3,
                                                            float& k4,
                                                            float& k5,
                                                            float& k6) {
    static_assert(PacketBytes == 8 || PacketBytes == 16);
    if constexpr (PacketBytes == 8) {
        const float4 low = decodeFp8x4ToFp32<InputT>(packet.value.x);
        const float4 high = decodeFp8x4ToFp32<InputT>(packet.value.y);
        // 8 mod 7 == 1, so packet n starts one component later than packet n-1.
        accumulateFp8K7EightValuesForLane<1, PacketBase, 0>(
            low, high, subgroup_lane, k0, k1, k2, k3, k4, k5, k6);
    } else {
        // Treat p16 as two consecutive p8 halves. 16 mod 7 == 2, so the first half begins at phase 2*n and the
        // second half at phase 2*n+1. Decoding/accumulating the halves sequentially keeps temporary state smaller.
        const float4 first_low = decodeFp8x4ToFp32<InputT>(packet.value.x);
        const float4 first_high = decodeFp8x4ToFp32<InputT>(packet.value.y);
        accumulateFp8K7EightValuesForLane<2, PacketBase, 0>(
            first_low, first_high, subgroup_lane, k0, k1, k2, k3, k4, k5, k6);
        const float4 second_low = decodeFp8x4ToFp32<InputT>(packet.value.z);
        const float4 second_high = decodeFp8x4ToFp32<InputT>(packet.value.w);
        accumulateFp8K7EightValuesForLane<2, PacketBase, 1>(
            second_low, second_high, subgroup_lane, k0, k1, k2, k3, k4, k5, k6);
    }
}

template <typename InputT, int R, int InputPacketBytes>
__global__ void fp8K7PhaseStreamRParallelCompleteKernel(const InputT* input, InputT* output, uint64_t outer_size) {
    static_assert(sizeof(InputT) == 1);
    static_assert(InputPacketBytes == 8 || InputPacketBytes == 16);
    static_assert((InputPacketBytes == 8 && (R == 8 || R == 16 || R == 32 || R == 64)) ||
                  (InputPacketBytes == 16 && (R == 16 || R == 32 || R == 64)));
    constexpr int BytesPerOutput = R * 7;
    constexpr int PacketsPerOutput = BytesPerOutput / InputPacketBytes;
    constexpr int LanesPerOutput = PacketsPerOutput <= 7 ? 8 : (PacketsPerOutput <= 14 ? 16 : 32);
    constexpr int OutputsPerWarp = TILED_REDUCTION_WARP_THREADS / LanesPerOutput;
    constexpr int OutputsPerBlock = SHALLOW_FP8_WARPS * OutputsPerWarp;
    constexpr int OutputPacketsPerBlock = (OutputsPerBlock * 7) / SHALLOW_FP8_PACKET_BYTES;
    constexpr uint32_t FullWarpMask = 0xffffffffu;
    static_assert(BytesPerOutput % InputPacketBytes == 0);
    static_assert(PacketsPerOutput <= 2 * LanesPerOutput);
    static_assert(OutputsPerWarp == 1 || OutputsPerWarp == 2 || OutputsPerWarp == 4);
    static_assert((OutputsPerBlock * 7) % SHALLOW_FP8_PACKET_BYTES == 0);

    __shared__ float output_values[OutputsPerBlock][7];

    const int warp = static_cast<int>(threadIdx.x) / TILED_REDUCTION_WARP_THREADS;
    const int lane = static_cast<int>(threadIdx.x) % TILED_REDUCTION_WARP_THREADS;
    const int subgroup = lane / LanesPerOutput;
    const int subgroup_lane = lane % LanesPerOutput;
    const uint64_t grid_stride = static_cast<uint64_t>(gridDim.x);

    for (uint64_t block_round = static_cast<uint64_t>(blockIdx.x);; block_round += grid_stride) {
        const uint64_t outer_begin = block_round * static_cast<uint64_t>(OutputsPerBlock);
        if (outer_begin >= outer_size) return;

        const int output_slot = warp * OutputsPerWarp + subgroup;
        const uint64_t outer_index = outer_begin + static_cast<uint64_t>(output_slot);
        const bool valid_output = outer_index < outer_size;
        float k0 = 0.0f;
        float k1 = 0.0f;
        float k2 = 0.0f;
        float k3 = 0.0f;
        float k4 = 0.0f;
        float k5 = 0.0f;
        float k6 = 0.0f;

        if (valid_output) {
            const auto* packets = reinterpret_cast<const RawVectorPacket<InputPacketBytes>*>(input) +
                                  outer_index * static_cast<uint64_t>(PacketsPerOutput);
            constexpr int FirstRoundPackets = PacketsPerOutput < LanesPerOutput ? PacketsPerOutput : LanesPerOutput;
            if (subgroup_lane < FirstRoundPackets) {
                const RawVectorPacket<InputPacketBytes> packet = packets[subgroup_lane];
                accumulateFp8K7PhasePacket<InputT, InputPacketBytes, 0>(
                    packet, subgroup_lane, k0, k1, k2, k3, k4, k5, k6);
            }
            if constexpr (PacketsPerOutput > LanesPerOutput) {
                constexpr int SecondRoundPackets = PacketsPerOutput - LanesPerOutput;
                static_assert(LanesPerOutput == TILED_REDUCTION_WARP_THREADS);
                if (subgroup_lane < SecondRoundPackets) {
                    const RawVectorPacket<InputPacketBytes> packet = packets[LanesPerOutput + subgroup_lane];
                    accumulateFp8K7PhasePacket<InputT, InputPacketBytes, LanesPerOutput>(
                        packet, subgroup_lane, k0, k1, k2, k3, k4, k5, k6);
                }
            }
        }

        for (int offset = LanesPerOutput / 2; offset > 0; offset >>= 1) {
            k0 += __shfl_down_sync(FullWarpMask, k0, offset, LanesPerOutput);
            k1 += __shfl_down_sync(FullWarpMask, k1, offset, LanesPerOutput);
            k2 += __shfl_down_sync(FullWarpMask, k2, offset, LanesPerOutput);
            k3 += __shfl_down_sync(FullWarpMask, k3, offset, LanesPerOutput);
            k4 += __shfl_down_sync(FullWarpMask, k4, offset, LanesPerOutput);
            k5 += __shfl_down_sync(FullWarpMask, k5, offset, LanesPerOutput);
            k6 += __shfl_down_sync(FullWarpMask, k6, offset, LanesPerOutput);
        }
        if (subgroup_lane == 0) {
            output_values[output_slot][0] = valid_output ? k0 : 0.0f;
            output_values[output_slot][1] = valid_output ? k1 : 0.0f;
            output_values[output_slot][2] = valid_output ? k2 : 0.0f;
            output_values[output_slot][3] = valid_output ? k3 : 0.0f;
            output_values[output_slot][4] = valid_output ? k4 : 0.0f;
            output_values[output_slot][5] = valid_output ? k5 : 0.0f;
            output_values[output_slot][6] = valid_output ? k6 : 0.0f;
        }
        __syncthreads();

        // Four adjacent K=7 outputs contain 28 bytes == seven p4 output packets. Keep this writeback identical for
        // p8 and p16 input candidates so timing differences reflect only the input packet/decode geometry.
        if (threadIdx.x < OutputPacketsPerBlock) {
            const int packet = static_cast<int>(threadIdx.x);
            const int scalar0 = packet * SHALLOW_FP8_PACKET_BYTES;
            const uint64_t valid_scalars = (outer_size - outer_begin) * 7 < static_cast<uint64_t>(OutputsPerBlock * 7)
                ? (outer_size - outer_begin) * 7
                : static_cast<uint64_t>(OutputsPerBlock * 7);
            if (static_cast<uint64_t>(scalar0) < valid_scalars) {
                const int output_group = packet / 7;
                const int phase = packet - output_group * 7;
                const int slot = output_group * 4;
                float2 low_values;
                float2 high_values;
                if (phase == 0) {
                    low_values = make_float2(output_values[slot][0], output_values[slot][1]);
                    high_values = make_float2(output_values[slot][2], output_values[slot][3]);
                } else if (phase == 1) {
                    low_values = make_float2(output_values[slot][4], output_values[slot][5]);
                    high_values = make_float2(output_values[slot][6], output_values[slot + 1][0]);
                } else if (phase == 2) {
                    low_values = make_float2(output_values[slot + 1][1], output_values[slot + 1][2]);
                    high_values = make_float2(output_values[slot + 1][3], output_values[slot + 1][4]);
                } else if (phase == 3) {
                    low_values = make_float2(output_values[slot + 1][5], output_values[slot + 1][6]);
                    high_values = make_float2(output_values[slot + 2][0], output_values[slot + 2][1]);
                } else if (phase == 4) {
                    low_values = make_float2(output_values[slot + 2][2], output_values[slot + 2][3]);
                    high_values = make_float2(output_values[slot + 2][4], output_values[slot + 2][5]);
                } else if (phase == 5) {
                    low_values = make_float2(output_values[slot + 2][6], output_values[slot + 3][0]);
                    high_values = make_float2(output_values[slot + 3][1], output_values[slot + 3][2]);
                } else {
                    low_values = make_float2(output_values[slot + 3][3], output_values[slot + 3][4]);
                    high_values = make_float2(output_values[slot + 3][5], output_values[slot + 3][6]);
                }
                uint16_t low_bits;
                uint16_t high_bits;
                if constexpr (std::is_same_v<InputT, __nv_fp8_e4m3>) {
                    low_bits = __nv_cvt_float2_to_fp8x2(low_values, __NV_SATFINITE, __NV_E4M3);
                    high_bits = __nv_cvt_float2_to_fp8x2(high_values, __NV_SATFINITE, __NV_E4M3);
                } else {
                    low_bits = __nv_cvt_float2_to_fp8x2(low_values, __NV_SATFINITE, __NV_E5M2);
                    high_bits = __nv_cvt_float2_to_fp8x2(high_values, __NV_SATFINITE, __NV_E5M2);
                }
                RawVectorPacket<SHALLOW_FP8_PACKET_BYTES> packed_output{};
                packed_output.value = static_cast<uint32_t>(low_bits) | (static_cast<uint32_t>(high_bits) << 16);
                auto* output_bytes = reinterpret_cast<uint8_t*>(output) + outer_begin * 7 + scalar0;
                const uint64_t remaining_scalars = valid_scalars - static_cast<uint64_t>(scalar0);
                if (remaining_scalars >= SHALLOW_FP8_PACKET_BYTES) {
                    *reinterpret_cast<RawVectorPacket<SHALLOW_FP8_PACKET_BYTES>*>(output_bytes) = packed_output;
                } else {
                    const auto* packed_bytes = reinterpret_cast<const uint8_t*>(&packed_output.value);
                    if (remaining_scalars >= 1) output_bytes[0] = packed_bytes[0];
                    if (remaining_scalars >= 2) output_bytes[1] = packed_bytes[1];
                    if (remaining_scalars >= 3) output_bytes[2] = packed_bytes[2];
                }
            }
        }
        __syncthreads();
    }
}

template <typename InputT, int R, int InputPacketBytes>
void launchFp8K7PhaseStreamRParallel(const Tensor& input,
                                     Tensor& output,
                                     const CubReductionGeometry& geometry,
                                     cudaStream_t stream) {
    static_assert((InputPacketBytes == 8 && (R == 8 || R == 16 || R == 32 || R == 64)) ||
                  (InputPacketBytes == 16 && (R == 16 || R == 32 || R == 64)));
    constexpr uint64_t PacketsPerOutput = static_cast<uint64_t>(R * 7 / InputPacketBytes);
    constexpr uint64_t LanesPerOutput = PacketsPerOutput <= 7 ? 8 : (PacketsPerOutput <= 14 ? 16 : 32);
    constexpr uint64_t OutputsPerWarp = TILED_REDUCTION_WARP_THREADS / LanesPerOutput;
    constexpr uint64_t OutputsPerBlock = SHALLOW_FP8_WARPS * OutputsPerWarp;
    const uint64_t block_count = std::min<uint64_t>(
        ceilDivideU64(geometry.outer_size, OutputsPerBlock), NARROW_FLAT_MAX_GRID_BLOCKS);
    fp8K7PhaseStreamRParallelCompleteKernel<InputT, R, InputPacketBytes>
        <<<static_cast<unsigned int>(block_count), SHALLOW_FP8_THREADS, 0, stream>>>(
            input.getMemPtr<InputT>(), output.getMemPtr<InputT>(), geometry.outer_size);
    CUDA_CHECK(cudaGetLastError());
}

template <typename InputT, int R, int InputPacketBytes>
[[nodiscard]] CubKernelOccupancyInfo queryFp8K7PhaseStreamRParallelOccupancy() {
    static_assert((InputPacketBytes == 8 && (R == 8 || R == 16 || R == 32 || R == 64)) ||
                  (InputPacketBytes == 16 && (R == 16 || R == 32 || R == 64)));
    return queryShallowFp8Occupancy(fp8K7PhaseStreamRParallelCompleteKernel<InputT, R, InputPacketBytes>);
}

template <typename InputT, int R>
void launchFp8K2PairedRowsRParallelP4(const Tensor& input,
                                      Tensor& output,
                                      const CubReductionGeometry& geometry,
                                      cudaStream_t stream) {
    static_assert(R == 31 || R == 32 || R == 63 || R == 64);
    constexpr uint64_t OutputsPerWarp = R <= 32 ? 2 : 1;
    const uint64_t warp_groups = ceilDivideU64(geometry.outer_size, OutputsPerWarp);
    const uint64_t block_count = std::min<uint64_t>(
        ceilDivideU64(warp_groups, static_cast<uint64_t>(SHALLOW_FP8_WARPS)), NARROW_FLAT_MAX_GRID_BLOCKS);
    fp8K2PairedRowsRParallelP4CompleteKernel<InputT, R>
        <<<static_cast<unsigned int>(block_count), SHALLOW_FP8_THREADS, 0, stream>>>(
            input.getMemPtr<InputT>(), output.getMemPtr<InputT>(), geometry.outer_size);
    CUDA_CHECK(cudaGetLastError());
}

template <typename InputT, int R>
[[nodiscard]] CubKernelOccupancyInfo queryFp8K2PairedRowsRParallelP4Occupancy() {
    static_assert(R == 31 || R == 32 || R == 63 || R == 64);
    return queryShallowFp8Occupancy(fp8K2PairedRowsRParallelP4CompleteKernel<InputT, R>);
}

template <typename InputT, int K>
__global__ void fp8DirectComponentWarpP4CompleteKernel(const InputT* input, InputT* output, uint64_t outer_size, uint32_t reduction_size) {
    static_assert(sizeof(InputT) == 1);
    static_assert(K >= 2 && K <= SHALLOW_FP8_MAX_K);
    constexpr uint32_t RowsPerGroup = SHALLOW_FP8_MAX_UNALIGNED_LOGICAL_BYTES / K;
    static_assert(RowsPerGroup >= 1);

    __shared__ float staged_values[SHALLOW_FP8_WARPS][SHALLOW_FP8_PACKET_WINDOW_BYTES];

    const int warp = static_cast<int>(threadIdx.x) / TILED_REDUCTION_WARP_THREADS;
    const int lane = static_cast<int>(threadIdx.x) % TILED_REDUCTION_WARP_THREADS;
    const uint64_t grid_stride = static_cast<uint64_t>(gridDim.x);

    for (uint64_t block_round = static_cast<uint64_t>(blockIdx.x);; block_round += grid_stride) {
        const uint64_t outer_index = block_round * static_cast<uint64_t>(SHALLOW_FP8_WARPS) + static_cast<uint64_t>(warp);
        if (outer_index >= outer_size) {
            return;
        }

        float local = 0.0f;
        for (uint32_t group_row = 0; group_row < reduction_size; group_row += RowsPerGroup) {
            const uint32_t group_rows = static_cast<uint32_t>(minU64(RowsPerGroup, static_cast<uint64_t>(reduction_size - group_row)));
            const InputT* group_begin =
                input + (outer_index * static_cast<uint64_t>(reduction_size) + group_row) * static_cast<uint64_t>(K);
            const uintptr_t group_address = reinterpret_cast<uintptr_t>(group_begin);
            const uint32_t prefix_elements = static_cast<uint32_t>(group_address & uintptr_t{SHALLOW_FP8_PACKET_BYTES - 1});
            const InputT* aligned_begin = reinterpret_cast<const InputT*>(group_address & ~uintptr_t{SHALLOW_FP8_PACKET_BYTES - 1});
            const uint32_t logical_elements = group_rows * static_cast<uint32_t>(K);
            const uint32_t packet_count = (prefix_elements + logical_elements + SHALLOW_FP8_PACKET_BYTES - 1) / SHALLOW_FP8_PACKET_BYTES;

            if (lane < static_cast<int>(packet_count)) {
                const PackedInputValues<InputT, SHALLOW_FP8_PACKET_BYTES> values =
                    loadVectorizedInputPacket<InputT, SHALLOW_FP8_PACKET_BYTES>(aligned_begin +
                                                                                static_cast<uint64_t>(lane) * SHALLOW_FP8_PACKET_BYTES);
                for (int item = 0; item < SHALLOW_FP8_PACKET_BYTES; ++item) {
                    const int logical_index = lane * SHALLOW_FP8_PACKET_BYTES + item - static_cast<int>(prefix_elements);
                    if (logical_index >= 0 && logical_index < static_cast<int>(logical_elements)) {
                        staged_values[warp][logical_index] = ToFp32<InputT>{}(values.values[item]);
                    }
                }
            }
            __syncwarp();

            if (lane < K) {
                for (uint32_t row = 0; row < group_rows; ++row) {
                    local += staged_values[warp][row * K + lane];
                }
            }
            __syncwarp();
        }

        if (lane < K) {
            if constexpr (K % SHALLOW_FP8_PACKET_BYTES == 0) {
                // Input staging is dead after the reduction; stage final FP32 values contiguously for packed narrow.
                staged_values[warp][lane] = local;
            } else {
                output[outer_index * static_cast<uint64_t>(K) + static_cast<uint64_t>(lane)] =
                    ThorLowPrecision::castToStorage<InputT>(local);
            }
        }
        if constexpr (K % SHALLOW_FP8_PACKET_BYTES == 0) {
            __syncwarp();
            
            if (lane < K / SHALLOW_FP8_PACKET_BYTES) {
                const int component_offset = lane * SHALLOW_FP8_PACKET_BYTES;
                const float2 low_values = make_float2(staged_values[warp][component_offset], staged_values[warp][component_offset + 1]);
                const float2 high_values =
                    make_float2(staged_values[warp][component_offset + 2], staged_values[warp][component_offset + 3]);
                __nv_fp8x2_storage_t low_bits;
                __nv_fp8x2_storage_t high_bits;
                if constexpr (std::is_same_v<InputT, __nv_fp8_e4m3>) {
                    low_bits = __nv_cvt_float2_to_fp8x2(low_values, __NV_SATFINITE, __NV_E4M3);
                    high_bits = __nv_cvt_float2_to_fp8x2(high_values, __NV_SATFINITE, __NV_E4M3);
                } else {
                    low_bits = __nv_cvt_float2_to_fp8x2(low_values, __NV_SATFINITE, __NV_E5M2);
                    high_bits = __nv_cvt_float2_to_fp8x2(high_values, __NV_SATFINITE, __NV_E5M2);
                }
                RawVectorPacket<SHALLOW_FP8_PACKET_BYTES> packet{};
                packet.value = static_cast<uint32_t>(low_bits) | (static_cast<uint32_t>(high_bits) << 16);
                auto* destination =
                    reinterpret_cast<RawVectorPacket<SHALLOW_FP8_PACKET_BYTES>*>(output + outer_index * static_cast<uint64_t>(K));
                destination[lane] = packet;
            }
        }
        __syncwarp();
    }
}

__global__ void fp8P4ReadWriteRooflineKernel(const RawVectorPacket<SHALLOW_FP8_PACKET_BYTES>* input_packets,
                                             RawVectorPacket<SHALLOW_FP8_PACKET_BYTES>* output_packets,
                                             uint64_t output_packet_count,
                                             uint32_t reduction_size,
                                             const uint8_t* input_bytes,
                                             uint8_t* output_bytes,
                                             uint32_t tail_output_bytes) {
    const int warp = static_cast<int>(threadIdx.x) / TILED_REDUCTION_WARP_THREADS;
    const int lane = static_cast<int>(threadIdx.x) % TILED_REDUCTION_WARP_THREADS;
    const uint64_t global_warp = static_cast<uint64_t>(blockIdx.x) * SHALLOW_FP8_WARPS + static_cast<uint64_t>(warp);
    const uint64_t full_groups = output_packet_count / TILED_REDUCTION_WARP_THREADS;
    const uint64_t grid_warps = static_cast<uint64_t>(gridDim.x) * SHALLOW_FP8_WARPS;

    for (uint64_t group = global_warp; group < full_groups; group += grid_warps) {
        const uint64_t output_packet = group * TILED_REDUCTION_WARP_THREADS + static_cast<uint64_t>(lane);
        const uint64_t input_group_base = group * static_cast<uint64_t>(TILED_REDUCTION_WARP_THREADS) * reduction_size;
        uint32_t reduced_bits = 0;
        for (uint32_t row = 0; row < reduction_size; ++row) {
            reduced_bits ^= input_packets[input_group_base + static_cast<uint64_t>(row) * TILED_REDUCTION_WARP_THREADS + lane].value;
        }
        output_packets[output_packet].value = reduced_bits;
    }

    // One partial packet group at most. Keep its lanes contiguous for every R read so the reference remains a true
    // p4 coalesced streaming path even when the output packet count is not warp aligned.
    const uint32_t tail_packets = static_cast<uint32_t>(output_packet_count % TILED_REDUCTION_WARP_THREADS);
    if (global_warp == 0 && lane < static_cast<int>(tail_packets)) {
        const uint64_t output_packet = full_groups * TILED_REDUCTION_WARP_THREADS + static_cast<uint64_t>(lane);
        const uint64_t input_group_base = full_groups * static_cast<uint64_t>(TILED_REDUCTION_WARP_THREADS) * reduction_size;
        uint32_t reduced_bits = 0;
        for (uint32_t row = 0; row < reduction_size; ++row) {
            reduced_bits ^= input_packets[input_group_base + static_cast<uint64_t>(row) * tail_packets + lane].value;
        }
        output_packets[output_packet].value = reduced_bits;
    }

    // Exact scalar tail after all complete output p4 packets. This is <=3 output bytes and does not affect the roofline.
    if (blockIdx.x == 0 && threadIdx.x < tail_output_bytes) {
        const uint64_t consumed_input_bytes = output_packet_count * SHALLOW_FP8_PACKET_BYTES * reduction_size;
        uint8_t reduced_bits = 0;
        for (uint32_t row = 0; row < reduction_size; ++row) {
            reduced_bits ^= input_bytes[consumed_input_bytes + static_cast<uint64_t>(row) * tail_output_bytes + threadIdx.x];
        }
        output_bytes[output_packet_count * SHALLOW_FP8_PACKET_BYTES + threadIdx.x] = reduced_bits;
    }
}

template <typename InputT, int K, int OutputsPerWarp>
void launchFp8CompactMultiOutputP4(const Tensor& input, Tensor& output, const CubReductionGeometry& geometry, cudaStream_t stream) {
    static_assert(sizeof(InputT) == 1);
    const uint64_t warp_groups = ceilDivideU64(geometry.outer_size, static_cast<uint64_t>(OutputsPerWarp));
    const uint64_t block_count =
        std::min<uint64_t>(ceilDivideU64(warp_groups, static_cast<uint64_t>(SHALLOW_FP8_WARPS)), NARROW_FLAT_MAX_GRID_BLOCKS);
    fp8CompactMultiOutputWarpP4CompleteKernel<InputT, K, OutputsPerWarp>
        <<<static_cast<unsigned int>(block_count), SHALLOW_FP8_THREADS, 0, stream>>>(
            input.getMemPtr<InputT>(), output.getMemPtr<InputT>(), geometry.outer_size, static_cast<uint32_t>(geometry.reduction_size));
    CUDA_CHECK(cudaGetLastError());
}


template <typename InputT, int K, int OutputsPerWarp, int PacketBytes>
void launchFp8CompactMultiOutputPacked(const Tensor& input,
                                       Tensor& output,
                                       const CubReductionGeometry& geometry,
                                       cudaStream_t stream) {
    static_assert(sizeof(InputT) == 1);
    static_assert(PacketBytes == 8 || PacketBytes == 16);
    if ((reinterpret_cast<uintptr_t>(input.getMemPtr<InputT>()) & uintptr_t{PacketBytes - 1}) != 0) {
        throw std::logic_error("FP8 compact multi-output packed calibration requires packet-aligned input base.");
    }
    const uint64_t warp_groups = ceilDivideU64(geometry.outer_size, static_cast<uint64_t>(OutputsPerWarp));
    const uint64_t block_count =
        std::min<uint64_t>(ceilDivideU64(warp_groups, static_cast<uint64_t>(SHALLOW_FP8_WARPS)), NARROW_FLAT_MAX_GRID_BLOCKS);
    fp8CompactMultiOutputWarpPackedCompleteKernel<InputT, K, OutputsPerWarp, PacketBytes>
        <<<static_cast<unsigned int>(block_count), SHALLOW_FP8_THREADS, 0, stream>>>(
            input.getMemPtr<InputT>(),
            output.getMemPtr<InputT>(),
            static_cast<uint32_t>(geometry.outer_size),
            static_cast<uint32_t>(geometry.reduction_size));
    CUDA_CHECK(cudaGetLastError());
}

template <typename InputT, int K, int PacketBytes>
void launchFp8CompactMultiOutputPackedForOutputs(const Tensor& input,
                                                 Tensor& output,
                                                 const CubReductionGeometry& geometry,
                                                 uint32_t outputs_per_warp,
                                                 cudaStream_t stream) {
    static_assert(PacketBytes == 8 || PacketBytes == 16);
    constexpr int MaxOutputComponents = narrowFp8MaxLogicalGroupBytes<PacketBytes> / 2;
    switch (outputs_per_warp) {
        case 2:
            if constexpr ((2 * K) % 4 == 0 && 2 * K <= MaxOutputComponents) {
                launchFp8CompactMultiOutputPacked<InputT, K, 2, PacketBytes>(input, output, geometry, stream);
                return;
            }
            break;
        case 4:
            if constexpr ((4 * K) % 4 == 0 && 4 * K <= MaxOutputComponents) {
                launchFp8CompactMultiOutputPacked<InputT, K, 4, PacketBytes>(input, output, geometry, stream);
                return;
            }
            break;
        case 8:
            if constexpr ((8 * K) % 4 == 0 && 8 * K <= MaxOutputComponents) {
                launchFp8CompactMultiOutputPacked<InputT, K, 8, PacketBytes>(input, output, geometry, stream);
                return;
            }
            break;
        case 16:
            if constexpr ((16 * K) % 4 == 0 && 16 * K <= MaxOutputComponents) {
                launchFp8CompactMultiOutputPacked<InputT, K, 16, PacketBytes>(input, output, geometry, stream);
                return;
            }
            break;
        case 32:
            if constexpr ((32 * K) % 4 == 0 && 32 * K <= MaxOutputComponents) {
                launchFp8CompactMultiOutputPacked<InputT, K, 32, PacketBytes>(input, output, geometry, stream);
                return;
            }
            break;
        default:
            break;
    }
    throw std::logic_error("FP8 compact multi-output packed calibration has invalid outputs-per-warp for K.");
}

template <typename InputT, int K>
void launchFp8RParallelP4(const Tensor& input, Tensor& output, const CubReductionGeometry& geometry, cudaStream_t stream) {
    static_assert(sizeof(InputT) == 1);
    const uint64_t block_count =
        std::min<uint64_t>(ceilDivideU64(geometry.outer_size, static_cast<uint64_t>(SHALLOW_FP8_WARPS)), NARROW_FLAT_MAX_GRID_BLOCKS);
    const InputT* input_ptr = input.getMemPtr<InputT>();
    InputT* output_ptr = output.getMemPtr<InputT>();
    const uint32_t outer_size = static_cast<uint32_t>(geometry.outer_size);
    const uint32_t reduction_size = static_cast<uint32_t>(geometry.reduction_size);

    if constexpr (K % SHALLOW_FP8_PACKET_BYTES == 0) {
        const uintptr_t pointer_bits = reinterpret_cast<uintptr_t>(input_ptr) | reinterpret_cast<uintptr_t>(output_ptr);
        if ((pointer_bits & uintptr_t{SHALLOW_FP8_PACKET_BYTES - 1}) == 0) {
            fp8RParallelWarpP4CompleteKernel<InputT, K, true><<<static_cast<unsigned int>(block_count), SHALLOW_FP8_THREADS, 0, stream>>>(
                input_ptr, output_ptr, outer_size, reduction_size);
        } else {
            fp8RParallelWarpP4CompleteKernel<InputT, K, false><<<static_cast<unsigned int>(block_count), SHALLOW_FP8_THREADS, 0, stream>>>(
                input_ptr, output_ptr, outer_size, reduction_size);
        }
    } else {
        fp8RParallelWarpP4CompleteKernel<InputT, K, false>
            <<<static_cast<unsigned int>(block_count), SHALLOW_FP8_THREADS, 0, stream>>>(input_ptr, output_ptr, outer_size, reduction_size);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <typename InputT, int K, int OutputsPerWarp>
void launchFp8MultiOutputRParallelP4(const Tensor& input,
                                     Tensor& output,
                                     const CubReductionGeometry& geometry,
                                     cudaStream_t stream) {
    static_assert(sizeof(InputT) == 1);
    constexpr uint32_t LanesPerOutput = TILED_REDUCTION_WARP_THREADS / OutputsPerWarp;
    static_assert(K <= static_cast<int>(LanesPerOutput));
    const uint64_t warp_groups = ceilDivideU64(geometry.outer_size, static_cast<uint64_t>(OutputsPerWarp));
    const uint64_t block_count = std::min<uint64_t>(
        ceilDivideU64(warp_groups, static_cast<uint64_t>(SHALLOW_FP8_WARPS)),
        NARROW_FLAT_MAX_GRID_BLOCKS);
    fp8MultiOutputRParallelWarpP4CompleteKernel<InputT, K, OutputsPerWarp>
        <<<static_cast<unsigned int>(block_count), SHALLOW_FP8_THREADS, 0, stream>>>(
            input.getMemPtr<InputT>(),
            output.getMemPtr<InputT>(),
            static_cast<uint32_t>(geometry.outer_size),
            static_cast<uint32_t>(geometry.reduction_size));
    CUDA_CHECK(cudaGetLastError());
}


template <typename InputT, int K, int OutputsPerWarp, int PacketBytes>
void launchFp8MultiOutputRParallelPacked(const Tensor& input,
                                         Tensor& output,
                                         const CubReductionGeometry& geometry,
                                         cudaStream_t stream) {
    static_assert(sizeof(InputT) == 1);
    static_assert(PacketBytes == 8 || PacketBytes == 16);
    constexpr uint32_t LanesPerOutput = TILED_REDUCTION_WARP_THREADS / OutputsPerWarp;
    static_assert(K <= static_cast<int>(LanesPerOutput));
    if ((reinterpret_cast<uintptr_t>(input.getMemPtr<InputT>()) & uintptr_t{PacketBytes - 1}) != 0) {
        throw std::logic_error("FP8 multi-output RParallel packed calibration requires packet-aligned input base.");
    }
    const uint64_t warp_groups = ceilDivideU64(geometry.outer_size, static_cast<uint64_t>(OutputsPerWarp));
    const uint64_t block_count = std::min<uint64_t>(
        ceilDivideU64(warp_groups, static_cast<uint64_t>(SHALLOW_FP8_WARPS)),
        NARROW_FLAT_MAX_GRID_BLOCKS);
    fp8MultiOutputRParallelWarpPackedCompleteKernel<InputT, K, OutputsPerWarp, PacketBytes>
        <<<static_cast<unsigned int>(block_count), SHALLOW_FP8_THREADS, 0, stream>>>(
            input.getMemPtr<InputT>(),
            output.getMemPtr<InputT>(),
            static_cast<uint32_t>(geometry.outer_size),
            static_cast<uint32_t>(geometry.reduction_size));
    CUDA_CHECK(cudaGetLastError());
}

template <typename InputT, int K, int PacketBytes>
void launchFp8MultiOutputRParallelPackedForOutputs(const Tensor& input,
                                                   Tensor& output,
                                                   const CubReductionGeometry& geometry,
                                                   uint32_t outputs_per_warp,
                                                   cudaStream_t stream) {
    static_assert(PacketBytes == 8 || PacketBytes == 16);
    switch (outputs_per_warp) {
        case 1:
            if constexpr (K <= 32) {
                launchFp8MultiOutputRParallelPacked<InputT, K, 1, PacketBytes>(input, output, geometry, stream);
                return;
            }
            break;
        case 2:
            if constexpr (K <= 16) {
                launchFp8MultiOutputRParallelPacked<InputT, K, 2, PacketBytes>(input, output, geometry, stream);
                return;
            }
            break;
        case 4:
            if constexpr (K <= 8) {
                launchFp8MultiOutputRParallelPacked<InputT, K, 4, PacketBytes>(input, output, geometry, stream);
                return;
            }
            break;
        case 8:
            if constexpr (K <= 4) {
                launchFp8MultiOutputRParallelPacked<InputT, K, 8, PacketBytes>(input, output, geometry, stream);
                return;
            }
            break;
        case 16:
            if constexpr (K <= 2) {
                launchFp8MultiOutputRParallelPacked<InputT, K, 16, PacketBytes>(input, output, geometry, stream);
                return;
            }
            break;
        case 32:
            if constexpr (K <= 1) {
                launchFp8MultiOutputRParallelPacked<InputT, K, 32, PacketBytes>(input, output, geometry, stream);
                return;
            }
            break;
        default:
            break;
    }
    throw std::logic_error("FP8 multi-output RParallel packed calibration received illegal K/outputs-per-warp ownership.");
}

template <typename InputT, int K>
void launchFp8MultiOutputRParallelP4ForOutputs(const Tensor& input,
                                               Tensor& output,
                                               const CubReductionGeometry& geometry,
                                               uint32_t outputs_per_warp,
                                               cudaStream_t stream) {
    switch (outputs_per_warp) {
        case 1:
            if constexpr (K >= 2 && K <= 32) {
                launchFp8RParallelP4<InputT, K>(input, output, geometry, stream);
                return;
            }
            break;
        case 2:
            if constexpr (K <= 16) {
                launchFp8MultiOutputRParallelP4<InputT, K, 2>(input, output, geometry, stream);
                return;
            }
            break;
        case 4:
            if constexpr (K <= 8) {
                launchFp8MultiOutputRParallelP4<InputT, K, 4>(input, output, geometry, stream);
                return;
            }
            break;
        case 8:
            if constexpr (K <= 4) {
                launchFp8MultiOutputRParallelP4<InputT, K, 8>(input, output, geometry, stream);
                return;
            }
            break;
        case 16:
            if constexpr (K <= 2) {
                launchFp8MultiOutputRParallelP4<InputT, K, 16>(input, output, geometry, stream);
                return;
            }
            break;
        case 32:
            if constexpr (K <= 1) {
                launchFp8MultiOutputRParallelP4<InputT, K, 32>(input, output, geometry, stream);
                return;
            }
            break;
        default:
            break;
    }
    throw std::logic_error("FP8 multi-output RParallel p4 calibration received illegal K/outputs-per-warp ownership.");
}


template <typename InputT, int K, int PacketBytes, int OutputsPerWarp, bool LinearPackets = false>
void launchFp8GeneralRParallelPacked(const Tensor& input,
                                     Tensor& output,
                                     const CubReductionGeometry& geometry,
                                     uint32_t warps_per_output,
                                     cudaStream_t stream) {
    static_assert(sizeof(InputT) == 1);
    static_assert(PacketBytes == 4 || PacketBytes == 8 || PacketBytes == 16);
    static_assert(OutputsPerWarp == 1 || OutputsPerWarp == 2 || OutputsPerWarp == 4 || OutputsPerWarp == 8);
    constexpr uint32_t LanesPerOutput = TILED_REDUCTION_WARP_THREADS / OutputsPerWarp;
    constexpr uint32_t PacketWindow = LanesPerOutput * PacketBytes;
    constexpr uint32_t LogicalCapacity =
        (K % PacketBytes) == 0 ? PacketWindow : PacketWindow - (PacketBytes - 1);
    static_assert(K <= static_cast<int>(LogicalCapacity));
    if constexpr (OutputsPerWarp > 1) {
        if (warps_per_output != 1) {
            throw std::logic_error("FP8 general RParallel output-parallel ownership requires W1.");
        }
    }
    const uint32_t outputs_per_block =
        (SHALLOW_FP8_WARPS / warps_per_output) * static_cast<uint32_t>(OutputsPerWarp);
    if ((reinterpret_cast<uintptr_t>(input.getMemPtr<InputT>()) & uintptr_t{PacketBytes - 1}) != 0) {
        throw std::logic_error("FP8 general RParallel calibration requires packet-aligned input base.");
    }
    const uint64_t block_count = std::min<uint64_t>(
        ceilDivideU64(geometry.outer_size, static_cast<uint64_t>(outputs_per_block)),
        NARROW_FLAT_MAX_GRID_BLOCKS);
    fp8GeneralRParallelPackedCompleteKernel<InputT, K, PacketBytes, OutputsPerWarp, LinearPackets>
        <<<static_cast<unsigned int>(block_count), SHALLOW_FP8_THREADS, 0, stream>>>(
            input.getMemPtr<InputT>(),
            output.getMemPtr<InputT>(),
            static_cast<uint32_t>(geometry.outer_size),
            static_cast<uint32_t>(geometry.reduction_size),
            warps_per_output);
    CUDA_CHECK(cudaGetLastError());
}


template <typename InputT, int K, int PacketBytes, bool LinearPackets = false>
void launchFp8GeneralRParallelPackedFirstStage(const Tensor& input,
                                                Tensor& fp32_partials,
                                                const CubReductionGeometry& geometry,
                                                uint32_t warps_per_output,
                                                uint32_t shards_per_output,
                                                cudaStream_t stream) {
    static_assert(sizeof(InputT) == 1);
    static_assert(PacketBytes == 4 || PacketBytes == 8 || PacketBytes == 16);
    constexpr uint32_t PacketWindow = TILED_REDUCTION_WARP_THREADS * PacketBytes;
    constexpr uint32_t LogicalCapacity =
        (K % PacketBytes) == 0 ? PacketWindow : PacketWindow - (PacketBytes - 1);
    static_assert(K <= static_cast<int>(LogicalCapacity));
    const uint32_t outputs_per_block = SHALLOW_FP8_WARPS / warps_per_output;
    if ((reinterpret_cast<uintptr_t>(input.getMemPtr<InputT>()) & uintptr_t{PacketBytes - 1}) != 0) {
        throw std::logic_error("FP8 staged general RParallel calibration requires packet-aligned input base.");
    }
    const uint64_t work_count = geometry.outer_size * static_cast<uint64_t>(shards_per_output);
    const uint64_t block_count = std::min<uint64_t>(
        ceilDivideU64(work_count, static_cast<uint64_t>(outputs_per_block)),
        NARROW_FLAT_MAX_GRID_BLOCKS);
    fp8GeneralRParallelPackedFirstStageKernel<InputT, K, PacketBytes, LinearPackets>
        <<<static_cast<unsigned int>(block_count), SHALLOW_FP8_THREADS, 0, stream>>>(
            input.getMemPtr<InputT>(),
            fp32_partials.getMemPtr<float>(),
            static_cast<uint32_t>(geometry.outer_size),
            static_cast<uint32_t>(geometry.reduction_size),
            warps_per_output,
            shards_per_output);
    CUDA_CHECK(cudaGetLastError());
}

template <typename InputT, int K, int PacketBytes, int OutputsPerWarp, bool LinearPackets = false>
void launchFp8GeneralRParallelPackedForOwnership(const Tensor& input,
                                                  Tensor& output,
                                                  const CubReductionGeometry& geometry,
                                                  uint32_t warps_per_output,
                                                  cudaStream_t stream) {
    static_assert(PacketBytes == 4 || PacketBytes == 8 || PacketBytes == 16);
    launchFp8GeneralRParallelPacked<InputT, K, PacketBytes, OutputsPerWarp, LinearPackets>(
        input, output, geometry, warps_per_output, stream);
}

template <typename InputT, int K>
void launchFp8DirectComponentP4(const Tensor& input, Tensor& output, const CubReductionGeometry& geometry, cudaStream_t stream) {
    static_assert(sizeof(InputT) == 1);
    const uint64_t block_count =
        std::min<uint64_t>(ceilDivideU64(geometry.outer_size, static_cast<uint64_t>(SHALLOW_FP8_WARPS)), NARROW_FLAT_MAX_GRID_BLOCKS);
    fp8DirectComponentWarpP4CompleteKernel<InputT, K><<<static_cast<unsigned int>(block_count), SHALLOW_FP8_THREADS, 0, stream>>>(
        input.getMemPtr<InputT>(), output.getMemPtr<InputT>(), geometry.outer_size, static_cast<uint32_t>(geometry.reduction_size));
    CUDA_CHECK(cudaGetLastError());
}

template <typename KernelT>
[[nodiscard]] CubKernelOccupancyInfo queryShallowFp8Occupancy(KernelT kernel) {
    cudaFuncAttributes attributes{};
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
    int active_blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks, kernel, SHALLOW_FP8_THREADS, 0));
    return CubKernelOccupancyInfo{attributes.numRegs, attributes.sharedSizeBytes, 0, active_blocks, active_blocks * SHALLOW_FP8_WARPS};
}

template <typename InputT, int K, int OutputsPerWarp>
[[nodiscard]] CubKernelOccupancyInfo queryFp8CompactMultiOutputP4Occupancy() {
    return queryShallowFp8Occupancy(fp8CompactMultiOutputWarpP4CompleteKernel<InputT, K, OutputsPerWarp>);
}


template <typename InputT, int K, int OutputsPerWarp, int PacketBytes>
[[nodiscard]] CubKernelOccupancyInfo queryFp8CompactMultiOutputPackedOccupancy() {
    static_assert(PacketBytes == 8 || PacketBytes == 16);
    return queryShallowFp8Occupancy(
        fp8CompactMultiOutputWarpPackedCompleteKernel<InputT, K, OutputsPerWarp, PacketBytes>);
}

template <typename InputT, int K, int PacketBytes>
[[nodiscard]] CubKernelOccupancyInfo queryFp8CompactMultiOutputPackedOccupancyForOutputs(uint32_t outputs_per_warp) {
    static_assert(PacketBytes == 8 || PacketBytes == 16);
    constexpr int MaxOutputComponents = narrowFp8MaxLogicalGroupBytes<PacketBytes> / 2;
    switch (outputs_per_warp) {
        case 2:
            if constexpr ((2 * K) % 4 == 0 && 2 * K <= MaxOutputComponents)
                return queryFp8CompactMultiOutputPackedOccupancy<InputT, K, 2, PacketBytes>();
            break;
        case 4:
            if constexpr ((4 * K) % 4 == 0 && 4 * K <= MaxOutputComponents)
                return queryFp8CompactMultiOutputPackedOccupancy<InputT, K, 4, PacketBytes>();
            break;
        case 8:
            if constexpr ((8 * K) % 4 == 0 && 8 * K <= MaxOutputComponents)
                return queryFp8CompactMultiOutputPackedOccupancy<InputT, K, 8, PacketBytes>();
            break;
        case 16:
            if constexpr ((16 * K) % 4 == 0 && 16 * K <= MaxOutputComponents)
                return queryFp8CompactMultiOutputPackedOccupancy<InputT, K, 16, PacketBytes>();
            break;
        case 32:
            if constexpr ((32 * K) % 4 == 0 && 32 * K <= MaxOutputComponents)
                return queryFp8CompactMultiOutputPackedOccupancy<InputT, K, 32, PacketBytes>();
            break;
        default:
            break;
    }
    throw std::logic_error("FP8 compact multi-output packed occupancy has invalid outputs-per-warp for K.");
}

template <typename InputT, int K>
[[nodiscard]] CubKernelOccupancyInfo queryFp8RParallelP4Occupancy() {
    if constexpr (K % SHALLOW_FP8_PACKET_BYTES == 0) {
        return queryShallowFp8Occupancy(fp8RParallelWarpP4CompleteKernel<InputT, K, true>);
    }
    return queryShallowFp8Occupancy(fp8RParallelWarpP4CompleteKernel<InputT, K, false>);
}

template <typename InputT, int K, int OutputsPerWarp>
[[nodiscard]] CubKernelOccupancyInfo queryFp8MultiOutputRParallelP4Occupancy() {
    constexpr uint32_t LanesPerOutput = TILED_REDUCTION_WARP_THREADS / OutputsPerWarp;
    static_assert(K <= static_cast<int>(LanesPerOutput));
    return queryShallowFp8Occupancy(
        fp8MultiOutputRParallelWarpP4CompleteKernel<InputT, K, OutputsPerWarp>);
}


template <typename InputT, int K, int OutputsPerWarp, int PacketBytes>
[[nodiscard]] CubKernelOccupancyInfo queryFp8MultiOutputRParallelPackedOccupancy() {
    static_assert(PacketBytes == 8 || PacketBytes == 16);
    constexpr uint32_t LanesPerOutput = TILED_REDUCTION_WARP_THREADS / OutputsPerWarp;
    static_assert(K <= static_cast<int>(LanesPerOutput));
    return queryShallowFp8Occupancy(
        fp8MultiOutputRParallelWarpPackedCompleteKernel<InputT, K, OutputsPerWarp, PacketBytes>);
}

template <typename InputT, int K, int PacketBytes>
[[nodiscard]] CubKernelOccupancyInfo queryFp8MultiOutputRParallelPackedOccupancyForOutputs(
    uint32_t outputs_per_warp) {
    static_assert(PacketBytes == 8 || PacketBytes == 16);
    switch (outputs_per_warp) {
        case 1:
            if constexpr (K <= 32)
                return queryFp8MultiOutputRParallelPackedOccupancy<InputT, K, 1, PacketBytes>();
            break;
        case 2:
            if constexpr (K <= 16)
                return queryFp8MultiOutputRParallelPackedOccupancy<InputT, K, 2, PacketBytes>();
            break;
        case 4:
            if constexpr (K <= 8)
                return queryFp8MultiOutputRParallelPackedOccupancy<InputT, K, 4, PacketBytes>();
            break;
        case 8:
            if constexpr (K <= 4)
                return queryFp8MultiOutputRParallelPackedOccupancy<InputT, K, 8, PacketBytes>();
            break;
        case 16:
            if constexpr (K <= 2)
                return queryFp8MultiOutputRParallelPackedOccupancy<InputT, K, 16, PacketBytes>();
            break;
        case 32:
            if constexpr (K <= 1)
                return queryFp8MultiOutputRParallelPackedOccupancy<InputT, K, 32, PacketBytes>();
            break;
        default:
            break;
    }
    throw std::logic_error("FP8 multi-output RParallel packed occupancy received illegal K/outputs-per-warp ownership.");
}

template <typename InputT, int K>
[[nodiscard]] CubKernelOccupancyInfo queryFp8MultiOutputRParallelP4OccupancyForOutputs(
    uint32_t outputs_per_warp) {
    switch (outputs_per_warp) {
        case 1:
            if constexpr (K >= 2 && K <= 32)
                return queryFp8RParallelP4Occupancy<InputT, K>();
            break;
        case 2:
            if constexpr (K <= 16)
                return queryFp8MultiOutputRParallelP4Occupancy<InputT, K, 2>();
            break;
        case 4:
            if constexpr (K <= 8)
                return queryFp8MultiOutputRParallelP4Occupancy<InputT, K, 4>();
            break;
        case 8:
            if constexpr (K <= 4)
                return queryFp8MultiOutputRParallelP4Occupancy<InputT, K, 8>();
            break;
        case 16:
            if constexpr (K <= 2)
                return queryFp8MultiOutputRParallelP4Occupancy<InputT, K, 16>();
            break;
        case 32:
            if constexpr (K <= 1)
                return queryFp8MultiOutputRParallelP4Occupancy<InputT, K, 32>();
            break;
        default:
            break;
    }
    throw std::logic_error("FP8 multi-output RParallel p4 occupancy received illegal K/outputs-per-warp ownership.");
}


template <typename InputT, int K, int PacketBytes, int OutputsPerWarp, bool LinearPackets = false>
[[nodiscard]] CubKernelOccupancyInfo queryFp8GeneralRParallelPackedOccupancy() {
    static_assert(PacketBytes == 4 || PacketBytes == 8 || PacketBytes == 16);
    static_assert(OutputsPerWarp == 1 || OutputsPerWarp == 2 || OutputsPerWarp == 4 || OutputsPerWarp == 8);
    constexpr uint32_t LanesPerOutput = TILED_REDUCTION_WARP_THREADS / OutputsPerWarp;
    constexpr uint32_t PacketWindow = LanesPerOutput * PacketBytes;
    constexpr uint32_t LogicalCapacity =
        (K % PacketBytes) == 0 ? PacketWindow : PacketWindow - (PacketBytes - 1);
    static_assert(K <= static_cast<int>(LogicalCapacity));
    return queryShallowFp8Occupancy(
        fp8GeneralRParallelPackedCompleteKernel<InputT, K, PacketBytes, OutputsPerWarp, LinearPackets>);
}

template <typename InputT, int K, int PacketBytes, int OutputsPerWarp, bool LinearPackets = false>
[[nodiscard]] CubKernelOccupancyInfo queryFp8GeneralRParallelPackedOccupancyForOwnership(
    uint32_t warps_per_output) {
    static_assert(PacketBytes == 4 || PacketBytes == 8 || PacketBytes == 16);
    if (warps_per_output != 1 && warps_per_output != 2 && warps_per_output != 4 && warps_per_output != 8) {
        throw std::logic_error("FP8 general RParallel occupancy received invalid warps-per-output.");
    }
    if constexpr (OutputsPerWarp > 1) {
        if (warps_per_output != 1) {
            throw std::logic_error("FP8 general RParallel output-parallel occupancy requires W1.");
        }
    }
    return queryFp8GeneralRParallelPackedOccupancy<InputT, K, PacketBytes, OutputsPerWarp, LinearPackets>();
}


template <typename InputT, int K, int PacketBytes, bool LinearPackets = false>
[[nodiscard]] CubKernelOccupancyInfo queryFp8GeneralRParallelPackedFirstStageOccupancy() {
    static_assert(PacketBytes == 4 || PacketBytes == 8 || PacketBytes == 16);
    constexpr uint32_t PacketWindow = TILED_REDUCTION_WARP_THREADS * PacketBytes;
    constexpr uint32_t LogicalCapacity =
        (K % PacketBytes) == 0 ? PacketWindow : PacketWindow - (PacketBytes - 1);
    static_assert(K <= static_cast<int>(LogicalCapacity));
    return queryShallowFp8Occupancy(fp8GeneralRParallelPackedFirstStageKernel<InputT, K, PacketBytes, LinearPackets>);
}

template <typename InputT, int K>
[[nodiscard]] CubKernelOccupancyInfo queryFp8DirectComponentP4Occupancy() {
    return queryShallowFp8Occupancy(fp8DirectComponentWarpP4CompleteKernel<InputT, K>);
}

[[nodiscard]] CubKernelOccupancyInfo queryFp8P4ReadWriteRooflineOccupancy() {
    return queryShallowFp8Occupancy(fp8P4ReadWriteRooflineKernel);
}

template <typename InputT, typename ReductionOpT, typename InputTransformT, int RowsPerShard, int PacketBytes, int MaxComponentsPerLane>
__global__ void narrowFp8PackedFlatRCooperativeFirstStageKernel(const InputT* input,
                                                                float* fp32_partials,
                                                                uint64_t outer_size,
                                                                uint64_t reduction_size,
                                                                uint64_t inner_size,
                                                                uint64_t shards_per_output) {
    static_assert(sizeof(InputT) == 1);
    static_assert(PacketBytes == 4 || PacketBytes == 8 || PacketBytes == 16);
    static_assert(RowsPerShard == 8 || RowsPerShard == 16 || RowsPerShard == 32 || RowsPerShard == 64 || RowsPerShard == 128 ||
                  RowsPerShard == 256 || RowsPerShard == 512 || RowsPerShard == 1024 || RowsPerShard == 2048 || RowsPerShard == 4096 ||
                  RowsPerShard == 8192 || RowsPerShard == 16384);
    static_assert(MaxComponentsPerLane == 1 || MaxComponentsPerLane == 2 || MaxComponentsPerLane == 4 || MaxComponentsPerLane == 5 ||
                  MaxComponentsPerLane == 8);
    constexpr int MaxSupportedK = TILED_REDUCTION_WARP_THREADS * MaxComponentsPerLane;
    constexpr uint64_t MaxRowsPerWarpFromShard = static_cast<uint64_t>(RowsPerShard / NARROW_FLAT_WARPS);
    constexpr float Init = NarrowFlatReductionIdentityFp32<ReductionOpT>::value;
    static_assert(MaxSupportedK <= NARROW_FP8_CALIBRATION_MAX_K);
    static_assert(MaxRowsPerWarpFromShard >= 1);

    __shared__ RawVectorPacket<PacketBytes> staged_packets[NARROW_FLAT_WARPS][NARROW_FP8_PACKETS_PER_WARP];
    __shared__ float warp_partials[NARROW_FLAT_WARPS][MaxSupportedK];

    const ReductionOpT reduction_op{};
    const InputTransformT input_transform{};
    const int warp = static_cast<int>(threadIdx.x) / TILED_REDUCTION_WARP_THREADS;
    const int lane = static_cast<int>(threadIdx.x) % TILED_REDUCTION_WARP_THREADS;
    const uint64_t total_work = outer_size * shards_per_output;
    const uint64_t grid_stride = static_cast<uint64_t>(gridDim.x);

    // Fill toward the packet window, but never choose a group so deep that fewer than eight warps can participate in a
    // full RowsPerShard shard. RowsPerShard is always divisible by eight in the calibration ladder.
    const uint64_t rows_from_packet_window = static_cast<uint64_t>(narrowFp8MaxLogicalGroupBytes<PacketBytes>) / inner_size;
    const uint64_t rows_per_warp_group = minU64(rows_from_packet_window, MaxRowsPerWarpFromShard);
    const uint64_t rows_per_block_round = rows_per_warp_group * static_cast<uint64_t>(NARROW_FLAT_WARPS);

    for (uint64_t work_index = static_cast<uint64_t>(blockIdx.x); work_index < total_work; work_index += grid_stride) {
        const uint64_t shard = work_index % shards_per_output;
        const uint64_t outer_index = work_index / shards_per_output;
        const uint64_t row_begin = shard * static_cast<uint64_t>(RowsPerShard);
        const uint64_t row_end = minU64(row_begin + static_cast<uint64_t>(RowsPerShard), reduction_size);

        float local[MaxComponentsPerLane];
        for (int slot = 0; slot < MaxComponentsPerLane; ++slot) {
            local[slot] = Init;
        }

        const uint64_t first_group_row = row_begin + static_cast<uint64_t>(warp) * rows_per_warp_group;
        for (uint64_t group_row = first_group_row; group_row < row_end; group_row += rows_per_block_round) {
            const uint64_t group_rows = minU64(rows_per_warp_group, row_end - group_row);
            const InputT* group_begin = input + (outer_index * reduction_size + group_row) * inner_size;
            const uintptr_t group_address = reinterpret_cast<uintptr_t>(group_begin);
            const uint32_t prefix_elements = static_cast<uint32_t>(group_address & uintptr_t{PacketBytes - 1});
            const auto* aligned_begin = reinterpret_cast<const RawVectorPacket<PacketBytes>*>(group_address & ~uintptr_t{PacketBytes - 1});
            const uint64_t logical_group_elements = group_rows * inner_size;
            const uint64_t packet_count =
                ceilDivideU64(static_cast<uint64_t>(prefix_elements) + logical_group_elements, static_cast<uint64_t>(PacketBytes));

            if (static_cast<uint64_t>(lane) < packet_count) {
                staged_packets[warp][lane] = aligned_begin[lane];
            }
            __syncwarp();

            const auto* staged_bytes = reinterpret_cast<const uint8_t*>(&staged_packets[warp][0]);
            for (int slot = 0; slot < MaxComponentsPerLane; ++slot) {
                const uint64_t component = static_cast<uint64_t>(lane + slot * TILED_REDUCTION_WARP_THREADS);
                if (component < inner_size) {
                    for (uint64_t row = 0; row < group_rows; ++row) {
                        const uint64_t staged_index = static_cast<uint64_t>(prefix_elements) + row * inner_size + component;
                        const InputT value = narrowFp8FromBits<InputT>(staged_bytes[staged_index]);
                        local[slot] = reduction_op(local[slot], input_transform(ToFp32<InputT>{}(value)));
                    }
                }
            }
            __syncwarp();
        }

        for (int slot = 0; slot < MaxComponentsPerLane; ++slot) {
            const uint64_t component = static_cast<uint64_t>(lane + slot * TILED_REDUCTION_WARP_THREADS);
            if (component < inner_size) {
                warp_partials[warp][component] = local[slot];
            }
        }
        __syncthreads();

        if (static_cast<uint64_t>(threadIdx.x) < inner_size) {
            const uint64_t component = static_cast<uint64_t>(threadIdx.x);
            float reduced = Init;
            for (int source_warp = 0; source_warp < NARROW_FLAT_WARPS; ++source_warp) {
                reduced = reduction_op(reduced, warp_partials[source_warp][component]);
            }
            fp32_partials[(outer_index * shards_per_output + shard) * inner_size + component] = reduced;
        }
        __syncthreads();
    }
}

template <typename InputT, typename ReductionOpT, typename InputTransformT, int RowsPerShard, int PacketBytes, int LanesPerComponent>
__global__ void narrowFp8PackedFlatRCooperativeTinyKFirstStageKernel(const InputT* input,
                                                                     float* fp32_partials,
                                                                     uint64_t outer_size,
                                                                     uint64_t reduction_size,
                                                                     uint64_t inner_size,
                                                                     uint64_t shards_per_output) {
    static_assert(sizeof(InputT) == 1);
    static_assert(PacketBytes == 4 || PacketBytes == 8 || PacketBytes == 16);
    static_assert(RowsPerShard == 8 || RowsPerShard == 16 || RowsPerShard == 32 || RowsPerShard == 64 || RowsPerShard == 128 ||
                  RowsPerShard == 256 || RowsPerShard == 512 || RowsPerShard == 1024 || RowsPerShard == 2048 || RowsPerShard == 4096 ||
                  RowsPerShard == 8192 || RowsPerShard == 16384);
    static_assert(LanesPerComponent == 4 || LanesPerComponent == 8);
    static_assert(TILED_REDUCTION_WARP_THREADS % LanesPerComponent == 0);
    constexpr int MaxSupportedK = TILED_REDUCTION_WARP_THREADS / LanesPerComponent;
    constexpr uint64_t MaxRowsPerWarpFromShard = static_cast<uint64_t>(RowsPerShard / NARROW_FLAT_WARPS);
    constexpr float Init = NarrowFlatReductionIdentityFp32<ReductionOpT>::value;
    static_assert(MaxSupportedK == 4 || MaxSupportedK == 8);
    static_assert(MaxRowsPerWarpFromShard >= 1);

    __shared__ RawVectorPacket<PacketBytes> staged_packets[NARROW_FLAT_WARPS][NARROW_FP8_PACKETS_PER_WARP];
    __shared__ float cooperative_lane_partials[NARROW_FLAT_WARPS][TILED_REDUCTION_WARP_THREADS];
    __shared__ float warp_partials[NARROW_FLAT_WARPS][MaxSupportedK];

    const ReductionOpT reduction_op{};
    const InputTransformT input_transform{};
    const int warp = static_cast<int>(threadIdx.x) / TILED_REDUCTION_WARP_THREADS;
    const int lane = static_cast<int>(threadIdx.x) % TILED_REDUCTION_WARP_THREADS;
    const int component = lane / LanesPerComponent;
    const int component_lane = lane % LanesPerComponent;
    const uint64_t total_work = outer_size * shards_per_output;
    const uint64_t grid_stride = static_cast<uint64_t>(gridDim.x);

    const uint64_t rows_from_packet_window = static_cast<uint64_t>(narrowFp8MaxLogicalGroupBytes<PacketBytes>) / inner_size;
    const uint64_t rows_per_warp_group = minU64(rows_from_packet_window, MaxRowsPerWarpFromShard);
    const uint64_t rows_per_block_round = rows_per_warp_group * static_cast<uint64_t>(NARROW_FLAT_WARPS);

    for (uint64_t work_index = static_cast<uint64_t>(blockIdx.x); work_index < total_work; work_index += grid_stride) {
        const uint64_t shard = work_index % shards_per_output;
        const uint64_t outer_index = work_index / shards_per_output;
        const uint64_t row_begin = shard * static_cast<uint64_t>(RowsPerShard);
        const uint64_t row_end = minU64(row_begin + static_cast<uint64_t>(RowsPerShard), reduction_size);

        float local = Init;
        const uint64_t first_group_row = row_begin + static_cast<uint64_t>(warp) * rows_per_warp_group;
        for (uint64_t group_row = first_group_row; group_row < row_end; group_row += rows_per_block_round) {
            const uint64_t group_rows = minU64(rows_per_warp_group, row_end - group_row);
            const InputT* group_begin = input + (outer_index * reduction_size + group_row) * inner_size;
            const uintptr_t group_address = reinterpret_cast<uintptr_t>(group_begin);
            const uint32_t prefix_elements = static_cast<uint32_t>(group_address & uintptr_t{PacketBytes - 1});
            const auto* aligned_begin = reinterpret_cast<const RawVectorPacket<PacketBytes>*>(group_address & ~uintptr_t{PacketBytes - 1});
            const uint64_t logical_group_elements = group_rows * inner_size;
            const uint64_t packet_count =
                ceilDivideU64(static_cast<uint64_t>(prefix_elements) + logical_group_elements, static_cast<uint64_t>(PacketBytes));

            if (static_cast<uint64_t>(lane) < packet_count) {
                staged_packets[warp][lane] = aligned_begin[lane];
            }
            __syncwarp();

            if (static_cast<uint64_t>(component) < inner_size) {
                const auto* staged_bytes = reinterpret_cast<const uint8_t*>(&staged_packets[warp][0]);
                for (uint64_t row = static_cast<uint64_t>(component_lane); row < group_rows;
                     row += static_cast<uint64_t>(LanesPerComponent)) {
                    const uint64_t staged_index =
                        static_cast<uint64_t>(prefix_elements) + row * inner_size + static_cast<uint64_t>(component);
                    const InputT value = narrowFp8FromBits<InputT>(staged_bytes[staged_index]);
                    local = reduction_op(local, input_transform(ToFp32<InputT>{}(value)));
                }
            }
            __syncwarp();
        }

        cooperative_lane_partials[warp][lane] = local;
        __syncwarp();
        if (component_lane == 0 && static_cast<uint64_t>(component) < inner_size) {
            float reduced = Init;
            for (int source_lane = 0; source_lane < LanesPerComponent; ++source_lane) {
                reduced = reduction_op(reduced, cooperative_lane_partials[warp][component * LanesPerComponent + source_lane]);
            }
            warp_partials[warp][component] = reduced;
        }
        __syncthreads();

        if (static_cast<uint64_t>(threadIdx.x) < inner_size) {
            const uint64_t output_component = static_cast<uint64_t>(threadIdx.x);
            float reduced = Init;
            for (int source_warp = 0; source_warp < NARROW_FLAT_WARPS; ++source_warp) {
                reduced = reduction_op(reduced, warp_partials[source_warp][output_component]);
            }
            fp32_partials[(outer_index * shards_per_output + shard) * inner_size + output_component] = reduced;
        }
        __syncthreads();
    }
}

template <typename InputT, typename ReductionOpT, typename InputTransformT, int RowsPerShard, int PacketBytes, int LanesPerComponent>
void launchNarrowFp8PackedTinyKForTransform(
    const Tensor& input, Tensor& fp32_partials, const CubReductionGeometry& geometry, uint64_t shards_per_output, cudaStream_t stream) {
    static_assert(sizeof(InputT) == 1);
    static_assert(PacketBytes == 4 || PacketBytes == 8 || PacketBytes == 16);
    static_assert(LanesPerComponent == 4 || LanesPerComponent == 8);
    constexpr uint64_t MaxSupportedK = static_cast<uint64_t>(TILED_REDUCTION_WARP_THREADS / LanesPerComponent);
    if (geometry.inner_size == 0 || geometry.inner_size > MaxSupportedK) {
        throw std::logic_error("Cooperative tiny-K packed FP8 narrow-flat launch received K outside its specialization.");
    }
    if (geometry.outer_size > std::numeric_limits<uint64_t>::max() / shards_per_output) {
        throw std::overflow_error("Cooperative tiny-K packed FP8 narrow-flat first-stage work count overflows uint64_t.");
    }
    const uint64_t total_work = geometry.outer_size * shards_per_output;
    const unsigned int grid_blocks = static_cast<unsigned int>(std::min<uint64_t>(total_work, NARROW_FLAT_MAX_GRID_BLOCKS));
    if (grid_blocks == 0) {
        throw std::logic_error("Cooperative tiny-K packed FP8 narrow-flat first stage requires non-zero work.");
    }

    narrowFp8PackedFlatRCooperativeTinyKFirstStageKernel<InputT,
                                                         ReductionOpT,
                                                         InputTransformT,
                                                         RowsPerShard,
                                                         PacketBytes,
                                                         LanesPerComponent>
        <<<grid_blocks, NARROW_FLAT_THREADS, 0, stream>>>(input.getMemPtr<InputT>(),
                                                          fp32_partials.getMemPtr<float>(),
                                                          geometry.outer_size,
                                                          geometry.reduction_size,
                                                          geometry.inner_size,
                                                          shards_per_output);
    CUDA_CHECK(cudaGetLastError());
}

template <typename InputT, typename ReductionOpT, typename InputTransformT, int RowsPerShard, int PacketBytes, int MaxComponentsPerLane>
void launchNarrowFp8PackedForTransform(
    const Tensor& input, Tensor& fp32_partials, const CubReductionGeometry& geometry, uint64_t shards_per_output, cudaStream_t stream) {
    static_assert(sizeof(InputT) == 1);
    static_assert(PacketBytes == 4 || PacketBytes == 8 || PacketBytes == 16);
    constexpr uint64_t MaxSupportedK = static_cast<uint64_t>(TILED_REDUCTION_WARP_THREADS * MaxComponentsPerLane);
    if (geometry.inner_size == 0 || geometry.inner_size > MaxSupportedK) {
        throw std::logic_error("Packed FP8 narrow-flat launch received K outside its component-per-lane specialization.");
    }
    if (geometry.outer_size > std::numeric_limits<uint64_t>::max() / shards_per_output) {
        throw std::overflow_error("Packed FP8 narrow-flat first-stage work count overflows uint64_t.");
    }
    const uint64_t total_work = geometry.outer_size * shards_per_output;
    const unsigned int grid_blocks = static_cast<unsigned int>(std::min<uint64_t>(total_work, NARROW_FLAT_MAX_GRID_BLOCKS));
    if (grid_blocks == 0) {
        throw std::logic_error("Packed FP8 narrow-flat first stage requires non-zero work.");
    }

    narrowFp8PackedFlatRCooperativeFirstStageKernel<InputT, ReductionOpT, InputTransformT, RowsPerShard, PacketBytes, MaxComponentsPerLane>
        <<<grid_blocks, NARROW_FLAT_THREADS, 0, stream>>>(input.getMemPtr<InputT>(),
                                                          fp32_partials.getMemPtr<float>(),
                                                          geometry.outer_size,
                                                          geometry.reduction_size,
                                                          geometry.inner_size,
                                                          shards_per_output);
    CUDA_CHECK(cudaGetLastError());
}

template <typename InputT, typename ReductionOpT, typename InputTransformT, int RowsPerShard>
void launchNarrowFlatForTransform(
    const Tensor& input, Tensor& fp32_partials, const CubReductionGeometry& geometry, uint64_t shards_per_output, cudaStream_t stream) {
    if (geometry.outer_size > std::numeric_limits<uint64_t>::max() / shards_per_output) {
        throw std::overflow_error("Narrow flat R-cooperative first-stage work count overflows uint64_t.");
    }
    const uint64_t total_work = geometry.outer_size * shards_per_output;
    const unsigned int grid_blocks = static_cast<unsigned int>(std::min<uint64_t>(total_work, NARROW_FLAT_MAX_GRID_BLOCKS));
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

template <typename InputT, typename ReductionOpT, typename InputTransformT>
void dispatchNarrowFlatFp8PackedRows(const Tensor& input,
                                     Tensor& fp32_partials,
                                     const CubReductionGeometry& geometry,
                                     uint64_t rows_per_shard,
                                     uint64_t shards_per_output,
                                     cudaStream_t stream) {
    static_assert(sizeof(InputT) == 1);
    switch (rows_per_shard) {
        case 128:
            launchNarrowFp8PackedForTransform<InputT, ReductionOpT, InputTransformT, 128, 16, 1>(
                input, fp32_partials, geometry, shards_per_output, stream);
            return;
        case 256:
            launchNarrowFp8PackedForTransform<InputT, ReductionOpT, InputTransformT, 256, 16, 1>(
                input, fp32_partials, geometry, shards_per_output, stream);
            return;
        case 512:
            launchNarrowFp8PackedForTransform<InputT, ReductionOpT, InputTransformT, 512, 16, 1>(
                input, fp32_partials, geometry, shards_per_output, stream);
            return;
        case 1024:
            launchNarrowFp8PackedForTransform<InputT, ReductionOpT, InputTransformT, 1024, 16, 1>(
                input, fp32_partials, geometry, shards_per_output, stream);
            return;
        default:
            throw std::logic_error("Unsupported packed FP8 narrow-flat rows_per_shard production specialization.");
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
#if THOR_CUB_ENABLE_FP8_TYPES
        case DataType::FP8_E4M3:
            dispatchNarrowFlatFp8PackedRows<__nv_fp8_e4m3, ReductionOpT, InputTransformT>(
                input, fp32_partials, geometry, rows_per_shard, shards_per_output, stream);
            return;
        case DataType::FP8_E5M2:
            dispatchNarrowFlatFp8PackedRows<__nv_fp8_e5m2, ReductionOpT, InputTransformT>(
                input, fp32_partials, geometry, rows_per_shard, shards_per_output, stream);
            return;
#endif
        default:
            throw std::logic_error("Narrow flat R-cooperative first stage requires FP8, FP16, or BF16 input.");
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
            dispatchNarrowFlatInput<ReductionOpT, IdentityFp32>(input, fp32_partials, geometry, rows_per_shard, shards_per_output, stream);
            return;
        case CubReductionStageInputTransform::AbsoluteValue:
            dispatchNarrowFlatInput<ReductionOpT, AbsoluteValueFp32>(
                input, fp32_partials, geometry, rows_per_shard, shards_per_output, stream);
            return;
        case CubReductionStageInputTransform::Square:
            dispatchNarrowFlatInput<ReductionOpT, SquareFp32>(input, fp32_partials, geometry, rows_per_shard, shards_per_output, stream);
            return;
    }
    throw std::logic_error("Unsupported narrow flat R-cooperative input transform.");
}

template <typename InputT, int RowsPerShard>
[[nodiscard]] CubKernelOccupancyInfo queryNarrowFlatOccupancyForRows() {
    const auto kernel = narrowLowPrecisionFlatRCooperativeFirstStageKernel<InputT, cuda::std::plus<float>, IdentityFp32, RowsPerShard>;
    cudaFuncAttributes attributes{};
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
    int active_blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks, kernel, NARROW_FLAT_THREADS, 0));
    return CubKernelOccupancyInfo{attributes.numRegs, attributes.sharedSizeBytes, 0, active_blocks, active_blocks * NARROW_FLAT_WARPS};
}

template <typename InputT, int RowsPerShard, int PacketBytes, int LanesPerComponent>
[[nodiscard]] CubKernelOccupancyInfo queryNarrowFp8PackedTinyKOccupancyForRows() {
    static_assert(sizeof(InputT) == 1);
    static_assert(PacketBytes == 4 || PacketBytes == 8 || PacketBytes == 16);
    static_assert(LanesPerComponent == 4 || LanesPerComponent == 8);
    const auto kernel = narrowFp8PackedFlatRCooperativeTinyKFirstStageKernel<InputT,
                                                                             cuda::std::plus<float>,
                                                                             IdentityFp32,
                                                                             RowsPerShard,
                                                                             PacketBytes,
                                                                             LanesPerComponent>;
    cudaFuncAttributes attributes{};
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
    int active_blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks, kernel, NARROW_FLAT_THREADS, 0));
    return CubKernelOccupancyInfo{attributes.numRegs, attributes.sharedSizeBytes, 0, active_blocks, active_blocks * NARROW_FLAT_WARPS};
}

template <typename InputT, int RowsPerShard, int PacketBytes, int MaxComponentsPerLane>
[[nodiscard]] CubKernelOccupancyInfo queryNarrowFp8PackedOccupancyForRows() {
    static_assert(sizeof(InputT) == 1);
    const auto kernel = narrowFp8PackedFlatRCooperativeFirstStageKernel<InputT,
                                                                        cuda::std::plus<float>,
                                                                        IdentityFp32,
                                                                        RowsPerShard,
                                                                        PacketBytes,
                                                                        MaxComponentsPerLane>;
    cudaFuncAttributes attributes{};
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
    int active_blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks, kernel, NARROW_FLAT_THREADS, 0));
    return CubKernelOccupancyInfo{attributes.numRegs, attributes.sharedSizeBytes, 0, active_blocks, active_blocks * NARROW_FLAT_WARPS};
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

template <typename InputT, int RowsPerShard, int PacketBytes>
void launchFp8NarrowFlatCalibrationForRows(
    const Tensor& input, Tensor& fp32_partials, const CubReductionGeometry& geometry, uint64_t shards_per_output, cudaStream_t stream) {
    static_assert(sizeof(InputT) == 1);
    static_assert(PacketBytes == 4 || PacketBytes == 8 || PacketBytes == 16);
    if (geometry.inner_size > static_cast<uint64_t>(narrowFp8MaxLogicalGroupBytes<PacketBytes>)) {
        throw std::logic_error("FP8 narrow-flat calibration K does not fit one packet-width-specific warp window.");
    }
    if (geometry.inner_size <= 4) {
        launchNarrowFp8PackedTinyKForTransform<InputT, cuda::std::plus<float>, IdentityFp32, RowsPerShard, PacketBytes, 8>(
            input, fp32_partials, geometry, shards_per_output, stream);
    } else if (geometry.inner_size <= 8) {
        launchNarrowFp8PackedTinyKForTransform<InputT, cuda::std::plus<float>, IdentityFp32, RowsPerShard, PacketBytes, 4>(
            input, fp32_partials, geometry, shards_per_output, stream);
    } else if (geometry.inner_size <= 32) {
        launchNarrowFp8PackedForTransform<InputT, cuda::std::plus<float>, IdentityFp32, RowsPerShard, PacketBytes, 1>(
            input, fp32_partials, geometry, shards_per_output, stream);
    } else if (geometry.inner_size <= 64) {
        launchNarrowFp8PackedForTransform<InputT, cuda::std::plus<float>, IdentityFp32, RowsPerShard, PacketBytes, 2>(
            input, fp32_partials, geometry, shards_per_output, stream);
    } else if (geometry.inner_size <= 128) {
        launchNarrowFp8PackedForTransform<InputT, cuda::std::plus<float>, IdentityFp32, RowsPerShard, PacketBytes, 4>(
            input, fp32_partials, geometry, shards_per_output, stream);
    } else if constexpr (PacketBytes >= 8) {
        if (geometry.inner_size <= 160) {
            launchNarrowFp8PackedForTransform<InputT, cuda::std::plus<float>, IdentityFp32, RowsPerShard, PacketBytes, 5>(
                input, fp32_partials, geometry, shards_per_output, stream);
        } else if (geometry.inner_size <= NARROW_FP8_CALIBRATION_MAX_K) {
            launchNarrowFp8PackedForTransform<InputT, cuda::std::plus<float>, IdentityFp32, RowsPerShard, PacketBytes, 8>(
                input, fp32_partials, geometry, shards_per_output, stream);
        } else {
            throw std::logic_error("FP8 narrow-flat calibration K exceeds 256.");
        }
    } else {
        throw std::logic_error("FP8 narrow-flat p4 calibration K exceeds its packet window.");
    }
}

template <typename InputT, int PacketBytes>
void launchFp8NarrowFlatCalibrationRows(const Tensor& input,
                                        Tensor& fp32_partials,
                                        const CubReductionGeometry& geometry,
                                        uint64_t rows_per_shard,
                                        uint64_t shards_per_output,
                                        cudaStream_t stream) {
    static_assert(sizeof(InputT) == 1);
    switch (rows_per_shard) {
        case 8:
            launchFp8NarrowFlatCalibrationForRows<InputT, 8, PacketBytes>(input, fp32_partials, geometry, shards_per_output, stream);
            return;
        case 16:
            launchFp8NarrowFlatCalibrationForRows<InputT, 16, PacketBytes>(input, fp32_partials, geometry, shards_per_output, stream);
            return;
        case 32:
            launchFp8NarrowFlatCalibrationForRows<InputT, 32, PacketBytes>(input, fp32_partials, geometry, shards_per_output, stream);
            return;
        case 64:
            launchFp8NarrowFlatCalibrationForRows<InputT, 64, PacketBytes>(input, fp32_partials, geometry, shards_per_output, stream);
            return;
        case 128:
            launchFp8NarrowFlatCalibrationForRows<InputT, 128, PacketBytes>(input, fp32_partials, geometry, shards_per_output, stream);
            return;
        case 256:
            launchFp8NarrowFlatCalibrationForRows<InputT, 256, PacketBytes>(input, fp32_partials, geometry, shards_per_output, stream);
            return;
        case 512:
            launchFp8NarrowFlatCalibrationForRows<InputT, 512, PacketBytes>(input, fp32_partials, geometry, shards_per_output, stream);
            return;
        case 1024:
            launchFp8NarrowFlatCalibrationForRows<InputT, 1024, PacketBytes>(input, fp32_partials, geometry, shards_per_output, stream);
            return;
        case 2048:
            launchFp8NarrowFlatCalibrationForRows<InputT, 2048, PacketBytes>(input, fp32_partials, geometry, shards_per_output, stream);
            return;
        case 4096:
            launchFp8NarrowFlatCalibrationForRows<InputT, 4096, PacketBytes>(input, fp32_partials, geometry, shards_per_output, stream);
            return;
        case 8192:
            launchFp8NarrowFlatCalibrationForRows<InputT, 8192, PacketBytes>(input, fp32_partials, geometry, shards_per_output, stream);
            return;
        case 16384:
            launchFp8NarrowFlatCalibrationForRows<InputT, 16384, PacketBytes>(input, fp32_partials, geometry, shards_per_output, stream);
            return;
        default:
            throw std::logic_error("FP8 narrow-flat calibration received unsupported rows-per-shard.");
    }
}

template <typename InputT, int RowsPerShard, int PacketBytes>
[[nodiscard]] CubKernelOccupancyInfo queryFp8NarrowFlatCalibrationOccupancyForRows(uint64_t inner_size) {
    static_assert(sizeof(InputT) == 1);
    if (inner_size > static_cast<uint64_t>(narrowFp8MaxLogicalGroupBytes<PacketBytes>)) {
        throw std::logic_error("FP8 narrow-flat calibration occupancy K does not fit the packet window.");
    }
    if (inner_size <= 4)
        return queryNarrowFp8PackedTinyKOccupancyForRows<InputT, RowsPerShard, PacketBytes, 8>();
    if (inner_size <= 8)
        return queryNarrowFp8PackedTinyKOccupancyForRows<InputT, RowsPerShard, PacketBytes, 4>();
    if (inner_size <= 32)
        return queryNarrowFp8PackedOccupancyForRows<InputT, RowsPerShard, PacketBytes, 1>();
    if (inner_size <= 64)
        return queryNarrowFp8PackedOccupancyForRows<InputT, RowsPerShard, PacketBytes, 2>();
    if (inner_size <= 128)
        return queryNarrowFp8PackedOccupancyForRows<InputT, RowsPerShard, PacketBytes, 4>();
    if constexpr (PacketBytes >= 8) {
        if (inner_size <= 160)
            return queryNarrowFp8PackedOccupancyForRows<InputT, RowsPerShard, PacketBytes, 5>();
        if (inner_size <= NARROW_FP8_CALIBRATION_MAX_K)
            return queryNarrowFp8PackedOccupancyForRows<InputT, RowsPerShard, PacketBytes, 8>();
    }
    throw std::logic_error("FP8 narrow-flat calibration occupancy K exceeds the packet-specific window.");
}

template <typename InputT, int PacketBytes>
[[nodiscard]] CubKernelOccupancyInfo queryFp8NarrowFlatCalibrationOccupancyForInput(uint64_t rows_per_shard, uint64_t inner_size) {
    static_assert(sizeof(InputT) == 1);
    switch (rows_per_shard) {
        case 8:
            return queryFp8NarrowFlatCalibrationOccupancyForRows<InputT, 8, PacketBytes>(inner_size);
        case 16:
            return queryFp8NarrowFlatCalibrationOccupancyForRows<InputT, 16, PacketBytes>(inner_size);
        case 32:
            return queryFp8NarrowFlatCalibrationOccupancyForRows<InputT, 32, PacketBytes>(inner_size);
        case 64:
            return queryFp8NarrowFlatCalibrationOccupancyForRows<InputT, 64, PacketBytes>(inner_size);
        case 128:
            return queryFp8NarrowFlatCalibrationOccupancyForRows<InputT, 128, PacketBytes>(inner_size);
        case 256:
            return queryFp8NarrowFlatCalibrationOccupancyForRows<InputT, 256, PacketBytes>(inner_size);
        case 512:
            return queryFp8NarrowFlatCalibrationOccupancyForRows<InputT, 512, PacketBytes>(inner_size);
        case 1024:
            return queryFp8NarrowFlatCalibrationOccupancyForRows<InputT, 1024, PacketBytes>(inner_size);
        case 2048:
            return queryFp8NarrowFlatCalibrationOccupancyForRows<InputT, 2048, PacketBytes>(inner_size);
        case 4096:
            return queryFp8NarrowFlatCalibrationOccupancyForRows<InputT, 4096, PacketBytes>(inner_size);
        case 8192:
            return queryFp8NarrowFlatCalibrationOccupancyForRows<InputT, 8192, PacketBytes>(inner_size);
        case 16384:
            return queryFp8NarrowFlatCalibrationOccupancyForRows<InputT, 16384, PacketBytes>(inner_size);
        default:
            throw std::logic_error("FP8 narrow-flat calibration occupancy received unsupported rows-per-shard.");
    }
}

template <typename InputT>
[[nodiscard]] CubKernelOccupancyInfo queryNarrowFp8PackedOccupancy(uint64_t rows_per_shard) {
    static_assert(sizeof(InputT) == 1);
    switch (rows_per_shard) {
        case 128:
            return queryNarrowFp8PackedOccupancyForRows<InputT, 128, 16, 1>();
        case 256:
            return queryNarrowFp8PackedOccupancyForRows<InputT, 256, 16, 1>();
        case 512:
            return queryNarrowFp8PackedOccupancyForRows<InputT, 512, 16, 1>();
        case 1024:
            return queryNarrowFp8PackedOccupancyForRows<InputT, 1024, 16, 1>();
        default:
            throw std::logic_error("Packed FP8 narrow-flat occupancy query received unsupported rows-per-shard.");
    }
}

void validateNarrowFlatFirstStage(const Tensor& input,
                                  const Tensor& fp32_partials,
                                  const CubReductionGeometry& geometry,
                                  uint64_t rows_per_shard,
                                  uint64_t shards_per_output) {
    const bool supported_input = input.getDataType() == DataType::FP16 || input.getDataType() == DataType::BF16
#if THOR_CUB_ENABLE_FP8_TYPES
                                 || input.getDataType() == DataType::FP8_E4M3 || input.getDataType() == DataType::FP8_E5M2
#endif
        ;
    if (!supported_input) {
        throw std::logic_error("Narrow flat R-cooperative first stage requires FP8, FP16, or BF16 input.");
    }
    if (geometry.inner_size == 0 || geometry.inner_size > NARROW_FLAT_MAX_K) {
        throw std::logic_error("Narrow flat R-cooperative first stage requires 1 <= K <= 32.");
    }
    if ((rows_per_shard != 128 && rows_per_shard != 256 && rows_per_shard != 512 && rows_per_shard != 1024) || shards_per_output <= 1) {
        throw std::logic_error("Narrow flat R-cooperative first stage requires rows_per_shard in {128,256,512,1024} and >1 shard.");
    }
    if (shards_per_output != ceilDivideU64(geometry.reduction_size, rows_per_shard)) {
        throw std::logic_error("Narrow flat R-cooperative first stage requires fixed-row sharding consistent with rows_per_shard.");
    }
    if (fp32_partials.getDataType() != DataType::FP32 ||
        fp32_partials.getDimensions() != std::vector<uint64_t>{geometry.outer_size, shards_per_output, geometry.inner_size}) {
        throw std::logic_error("Narrow flat R-cooperative FP32 partial shape does not match stamped geometry.");
    }
}

}  // namespace

void launchFp8NarrowLowPrecisionFlatRCooperativeCalibrationFirstStage(const Tensor& input,
                                                                      Tensor& fp32_partials,
                                                                      const CubReductionGeometry& geometry,
                                                                      size_t packet_bytes,
                                                                      uint64_t rows_per_shard,
                                                                      uint64_t shards_per_output,
                                                                      Stream& stream) {
#if THOR_CUB_ENABLE_FP8_TYPES
    if ((packet_bytes != 4 && packet_bytes != 8 && packet_bytes != 16) || geometry.inner_size == 0 ||
        geometry.inner_size > NARROW_FP8_CALIBRATION_MAX_K || shards_per_output <= 1 ||
        shards_per_output != ceilDivideU64(geometry.reduction_size, rows_per_shard) || fp32_partials.getDataType() != DataType::FP32 ||
        fp32_partials.getDimensions() != std::vector<uint64_t>{geometry.outer_size, shards_per_output, geometry.inner_size}) {
        throw std::logic_error("FP8 narrow-flat calibration received invalid staged geometry.");
    }
    const uint64_t max_logical_group_bytes = packet_bytes == 4   ? narrowFp8MaxLogicalGroupBytes<4>
                                             : packet_bytes == 8 ? narrowFp8MaxLogicalGroupBytes<8>
                                                                 : narrowFp8MaxLogicalGroupBytes<16>;
    if (geometry.inner_size > max_logical_group_bytes) {
        throw std::logic_error("FP8 narrow-flat calibration K does not fit the selected packet window.");
    }
    const auto dispatch = [&]<typename InputT>() {
        switch (packet_bytes) {
            case 4:
                launchFp8NarrowFlatCalibrationRows<InputT, 4>(
                    input, fp32_partials, geometry, rows_per_shard, shards_per_output, stream.getStream());
                return;
            case 8:
                launchFp8NarrowFlatCalibrationRows<InputT, 8>(
                    input, fp32_partials, geometry, rows_per_shard, shards_per_output, stream.getStream());
                return;
            case 16:
                launchFp8NarrowFlatCalibrationRows<InputT, 16>(
                    input, fp32_partials, geometry, rows_per_shard, shards_per_output, stream.getStream());
                return;
        }
        throw std::logic_error("FP8 narrow-flat calibration received unsupported packet width.");
    };
    switch (input.getDataType()) {
        case DataType::FP8_E4M3:
            dispatch.template operator()<__nv_fp8_e4m3>();
            return;
        case DataType::FP8_E5M2:
            dispatch.template operator()<__nv_fp8_e5m2>();
            return;
        default:
            throw std::logic_error("FP8 narrow-flat calibration requires FP8 input.");
    }
#else
    static_cast<void>(input);
    static_cast<void>(fp32_partials);
    static_cast<void>(geometry);
    static_cast<void>(packet_bytes);
    static_cast<void>(rows_per_shard);
    static_cast<void>(shards_per_output);
    static_cast<void>(stream);
    throw std::logic_error("FP8 narrow-flat calibration requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}

CubKernelOccupancyInfo queryFp8NarrowLowPrecisionFlatRCooperativeCalibrationOccupancy(DataType input_dtype,
                                                                                      size_t packet_bytes,
                                                                                      uint64_t rows_per_shard,
                                                                                      uint64_t inner_size) {
#if THOR_CUB_ENABLE_FP8_TYPES
    const auto dispatch = [&]<typename InputT>() -> CubKernelOccupancyInfo {
        switch (packet_bytes) {
            case 4:
                return queryFp8NarrowFlatCalibrationOccupancyForInput<InputT, 4>(rows_per_shard, inner_size);
            case 8:
                return queryFp8NarrowFlatCalibrationOccupancyForInput<InputT, 8>(rows_per_shard, inner_size);
            case 16:
                return queryFp8NarrowFlatCalibrationOccupancyForInput<InputT, 16>(rows_per_shard, inner_size);
            default:
                throw std::logic_error("FP8 narrow-flat calibration occupancy received unsupported packet width.");
        }
    };
    switch (input_dtype) {
        case DataType::FP8_E4M3:
            return dispatch.template operator()<__nv_fp8_e4m3>();
        case DataType::FP8_E5M2:
            return dispatch.template operator()<__nv_fp8_e5m2>();
        default:
            throw std::logic_error("FP8 narrow-flat calibration occupancy requires FP8 input.");
    }
#else
    static_cast<void>(input_dtype);
    static_cast<void>(packet_bytes);
    static_cast<void>(rows_per_shard);
    static_cast<void>(inner_size);
    throw std::logic_error("FP8 narrow-flat calibration requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}

void launchDeprecatedExperimentalFp8OuterGroupedP4RCooperativeCalibrationComplete(const Tensor& input,
                                                                                  Tensor& output,
                                                                                  const CubReductionGeometry& geometry,
                                                                                  Stream& stream) {
#if THOR_CUB_ENABLE_FP8_TYPES
    if (geometry.outer_size == 0 || geometry.reduction_size == 0 || geometry.reduction_size > OUTER_GROUPED_FP8_MAX_R ||
        geometry.inner_size == 0 || geometry.inner_size > OUTER_GROUPED_FP8_MAX_K || output.getDataType() != input.getDataType() ||
        output.getDimensions() != geometry.output_dimensions) {
        throw std::logic_error("FP8 outer-grouped p4 calibration received unsupported complete geometry.");
    }
    const auto dispatch = [&]<typename InputT>() {
        if (geometry.inner_size <= TILED_REDUCTION_WARP_THREADS) {
            launchFp8OuterGroupedP4ForComponents<InputT, 1>(input, output, geometry, stream.getStream());
        } else {
            launchFp8OuterGroupedP4ForComponents<InputT, 2>(input, output, geometry, stream.getStream());
        }
    };
    switch (input.getDataType()) {
        case DataType::FP8_E4M3:
            dispatch.template operator()<__nv_fp8_e4m3>();
            return;
        case DataType::FP8_E5M2:
            dispatch.template operator()<__nv_fp8_e5m2>();
            return;
        default:
            throw std::logic_error("FP8 outer-grouped p4 calibration requires FP8 input.");
    }
#else
    static_cast<void>(input);
    static_cast<void>(output);
    static_cast<void>(geometry);
    static_cast<void>(stream);
    throw std::logic_error("FP8 outer-grouped p4 calibration requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}

CubKernelOccupancyInfo queryDeprecatedExperimentalFp8OuterGroupedP4RCooperativeCalibrationOccupancy(DataType input_dtype,
                                                                                                    uint64_t inner_size) {
#if THOR_CUB_ENABLE_FP8_TYPES
    if (inner_size == 0 || inner_size > OUTER_GROUPED_FP8_MAX_K) {
        throw std::logic_error("FP8 outer-grouped p4 calibration occupancy requires 1 <= K <= 64.");
    }
    const auto dispatch = [&]<typename InputT>() -> CubKernelOccupancyInfo {
        if (inner_size <= TILED_REDUCTION_WARP_THREADS) {
            return queryFp8OuterGroupedP4OccupancyForComponents<InputT, 1>();
        }
        return queryFp8OuterGroupedP4OccupancyForComponents<InputT, 2>();
    };
    switch (input_dtype) {
        case DataType::FP8_E4M3:
            return dispatch.template operator()<__nv_fp8_e4m3>();
        case DataType::FP8_E5M2:
            return dispatch.template operator()<__nv_fp8_e5m2>();
        default:
            throw std::logic_error("FP8 outer-grouped p4 calibration occupancy requires FP8 input.");
    }
#else
    static_cast<void>(input_dtype);
    static_cast<void>(inner_size);
    throw std::logic_error("FP8 outer-grouped p4 calibration requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}

void launchFp8CompactMultiOutputWarpP4RCooperativeCalibrationComplete(const Tensor& input,
                                                                      Tensor& output,
                                                                      const CubReductionGeometry& geometry,
                                                                      Stream& stream) {
#if THOR_CUB_ENABLE_FP8_TYPES
    const uint32_t outputs_per_warp = shallowFp8CompactOutputsPerWarp(geometry.reduction_size, geometry.inner_size);
    if (geometry.outer_size == 0 || geometry.reduction_size == 0 || geometry.reduction_size > SHALLOW_FP8_MAX_R ||
        !shallowFp8SupportedCalibrationK(geometry.inner_size) || outputs_per_warp <= 1 || output.getDataType() != input.getDataType() ||
        output.getDimensions() != geometry.output_dimensions) {
        throw std::logic_error("FP8 compact multi-output p4 calibration received unsupported complete geometry.");
    }
    const auto dispatch_outputs = [&]<typename InputT, int K>() {
        switch (outputs_per_warp) {
            case 2:
                if constexpr (2 * K <= 64 && (2 * K) % 4 == 0) {
                    launchFp8CompactMultiOutputP4<InputT, K, 2>(input, output, geometry, stream.getStream());
                    return;
                }
                break;
            case 4:
                if constexpr (4 * K <= 64 && (4 * K) % 4 == 0) {
                    launchFp8CompactMultiOutputP4<InputT, K, 4>(input, output, geometry, stream.getStream());
                    return;
                }
                break;
            case 8:
                if constexpr (8 * K <= 64 && (8 * K) % 4 == 0) {
                    launchFp8CompactMultiOutputP4<InputT, K, 8>(input, output, geometry, stream.getStream());
                    return;
                }
                break;
            case 16:
                if constexpr (16 * K <= 64 && (16 * K) % 4 == 0) {
                    launchFp8CompactMultiOutputP4<InputT, K, 16>(input, output, geometry, stream.getStream());
                    return;
                }
                break;
            case 32:
                if constexpr (32 * K <= 64 && (32 * K) % 4 == 0) {
                    launchFp8CompactMultiOutputP4<InputT, K, 32>(input, output, geometry, stream.getStream());
                    return;
                }
                break;
            default:
                break;
        }
        throw std::logic_error("FP8 compact multi-output p4 calibration has invalid outputs-per-warp for K.");
    };
    const auto dispatch_k = [&]<typename InputT>() {
        switch (geometry.inner_size) {
            case 2:
                dispatch_outputs.template operator()<InputT, 2>();
                return;
            case 3:
                dispatch_outputs.template operator()<InputT, 3>();
                return;
            case 4:
                dispatch_outputs.template operator()<InputT, 4>();
                return;
            case 5:
                dispatch_outputs.template operator()<InputT, 5>();
                return;
            case 7:
                dispatch_outputs.template operator()<InputT, 7>();
                return;
            case 8:
                dispatch_outputs.template operator()<InputT, 8>();
                return;
            case 9:
                dispatch_outputs.template operator()<InputT, 9>();
                return;
            case 11:
                dispatch_outputs.template operator()<InputT, 11>();
                return;
            case 13:
                dispatch_outputs.template operator()<InputT, 13>();
                return;
            case 15:
                dispatch_outputs.template operator()<InputT, 15>();
                return;
            case 16:
                dispatch_outputs.template operator()<InputT, 16>();
                return;
            case 17:
                dispatch_outputs.template operator()<InputT, 17>();
                return;
            case 19:
                dispatch_outputs.template operator()<InputT, 19>();
                return;
            case 31:
                dispatch_outputs.template operator()<InputT, 31>();
                return;
            case 32:
                dispatch_outputs.template operator()<InputT, 32>();
                return;
            default:
                throw std::logic_error("FP8 compact multi-output p4 calibration received unsupported K.");
        }
    };
    switch (input.getDataType()) {
        case DataType::FP8_E4M3:
            dispatch_k.template operator()<__nv_fp8_e4m3>();
            return;
        case DataType::FP8_E5M2:
            dispatch_k.template operator()<__nv_fp8_e5m2>();
            return;
        default:
            throw std::logic_error("FP8 compact multi-output p4 calibration requires FP8 input.");
    }
#else
    static_cast<void>(input);
    static_cast<void>(output);
    static_cast<void>(geometry);
    static_cast<void>(stream);
    throw std::logic_error("FP8 compact multi-output p4 calibration requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}

CubKernelOccupancyInfo queryFp8CompactMultiOutputWarpP4RCooperativeCalibrationOccupancy(DataType input_dtype,
                                                                                        uint64_t inner_size,
                                                                                        uint32_t outputs_per_warp) {
#if THOR_CUB_ENABLE_FP8_TYPES
    const auto dispatch_outputs = [&]<typename InputT, int K>() -> CubKernelOccupancyInfo {
        switch (outputs_per_warp) {
            case 2:
                if constexpr (2 * K <= 64 && (2 * K) % 4 == 0)
                    return queryFp8CompactMultiOutputP4Occupancy<InputT, K, 2>();
                break;
            case 4:
                if constexpr (4 * K <= 64 && (4 * K) % 4 == 0)
                    return queryFp8CompactMultiOutputP4Occupancy<InputT, K, 4>();
                break;
            case 8:
                if constexpr (8 * K <= 64 && (8 * K) % 4 == 0)
                    return queryFp8CompactMultiOutputP4Occupancy<InputT, K, 8>();
                break;
            case 16:
                if constexpr (16 * K <= 64 && (16 * K) % 4 == 0)
                    return queryFp8CompactMultiOutputP4Occupancy<InputT, K, 16>();
                break;
            case 32:
                if constexpr (32 * K <= 64 && (32 * K) % 4 == 0)
                    return queryFp8CompactMultiOutputP4Occupancy<InputT, K, 32>();
                break;
            default:
                break;
        }
        throw std::logic_error("FP8 compact multi-output p4 occupancy has invalid outputs-per-warp for K.");
    };
    const auto dispatch_k = [&]<typename InputT>() -> CubKernelOccupancyInfo {
        switch (inner_size) {
            case 2:
                return dispatch_outputs.template operator()<InputT, 2>();
            case 3:
                return dispatch_outputs.template operator()<InputT, 3>();
            case 4:
                return dispatch_outputs.template operator()<InputT, 4>();
            case 5:
                return dispatch_outputs.template operator()<InputT, 5>();
            case 7:
                return dispatch_outputs.template operator()<InputT, 7>();
            case 8:
                return dispatch_outputs.template operator()<InputT, 8>();
            case 9:
                return dispatch_outputs.template operator()<InputT, 9>();
            case 11:
                return dispatch_outputs.template operator()<InputT, 11>();
            case 13:
                return dispatch_outputs.template operator()<InputT, 13>();
            case 15:
                return dispatch_outputs.template operator()<InputT, 15>();
            case 16:
                return dispatch_outputs.template operator()<InputT, 16>();
            case 17:
                return dispatch_outputs.template operator()<InputT, 17>();
            case 19:
                return dispatch_outputs.template operator()<InputT, 19>();
            case 31:
                return dispatch_outputs.template operator()<InputT, 31>();
            case 32:
                return dispatch_outputs.template operator()<InputT, 32>();
            default:
                throw std::logic_error("FP8 compact multi-output p4 occupancy received unsupported K.");
        }
    };
    switch (input_dtype) {
        case DataType::FP8_E4M3:
            return dispatch_k.template operator()<__nv_fp8_e4m3>();
        case DataType::FP8_E5M2:
            return dispatch_k.template operator()<__nv_fp8_e5m2>();
        default:
            throw std::logic_error("FP8 compact multi-output p4 occupancy requires FP8 input.");
    }
#else
    static_cast<void>(input_dtype);
    static_cast<void>(inner_size);
    static_cast<void>(outputs_per_warp);
    throw std::logic_error("FP8 compact multi-output p4 calibration requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}

uint32_t fp8CompactMultiOutputWarpP4RCooperativeCalibrationOutputsPerWarp(uint64_t reduction_size, uint64_t inner_size) {
    if (reduction_size == 0 || reduction_size > SHALLOW_FP8_MAX_R || !shallowFp8SupportedCalibrationK(inner_size)) {
        throw std::logic_error("FP8 compact multi-output p4 calibration grouping requires supported R/K.");
    }
    return shallowFp8CompactOutputsPerWarp(reduction_size, inner_size);
}


uint32_t fp8CompactMultiOutputWarpRCooperativeCalibrationOutputsPerWarp(uint64_t reduction_size,
                                                                        uint64_t inner_size,
                                                                        uint32_t packet_bytes) {
    if (reduction_size < 2 || reduction_size > SHALLOW_FP8_MAX_R || !shallowFp8SupportedCalibrationK(inner_size)) {
        throw std::logic_error("FP8 compact multi-output calibration grouping requires supported R>=2/K.");
    }
    switch (packet_bytes) {
        case 4:
            return shallowFp8CompactOutputsPerWarp(reduction_size, inner_size);
        case 8:
            return shallowFp8CompactPackedOutputsPerWarp<8>(reduction_size, inner_size);
        case 16:
            return shallowFp8CompactPackedOutputsPerWarp<16>(reduction_size, inner_size);
        default:
            throw std::logic_error("FP8 compact multi-output calibration grouping requires packet width 4/8/16.");
    }
}

void launchFp8CompactMultiOutputWarpRCooperativeCalibrationComplete(const Tensor& input,
                                                                    Tensor& output,
                                                                    const CubReductionGeometry& geometry,
                                                                    uint32_t packet_bytes,
                                                                    Stream& stream) {
#if THOR_CUB_ENABLE_FP8_TYPES
    if (packet_bytes == 4) {
        launchFp8CompactMultiOutputWarpP4RCooperativeCalibrationComplete(input, output, geometry, stream);
        return;
    }
    const uint32_t outputs_per_warp =
        fp8CompactMultiOutputWarpRCooperativeCalibrationOutputsPerWarp(
            geometry.reduction_size, geometry.inner_size, packet_bytes);
    if (geometry.outer_size == 0 || !shallowFp8FitsUint32Indexing(geometry) || outputs_per_warp <= 1 ||
        output.getDataType() != input.getDataType() || output.getDimensions() != geometry.output_dimensions ||
        (packet_bytes != 8 && packet_bytes != 16)) {
        throw std::logic_error("FP8 compact multi-output packed calibration received unsupported complete geometry.");
    }

    const auto dispatch_k = [&]<typename InputT>() {
        const auto launch_k = [&]<int K>() {
            switch (packet_bytes) {
                case 8:
                    launchFp8CompactMultiOutputPackedForOutputs<InputT, K, 8>(
                        input, output, geometry, outputs_per_warp, stream.getStream());
                    return;
                case 16:
                    launchFp8CompactMultiOutputPackedForOutputs<InputT, K, 16>(
                        input, output, geometry, outputs_per_warp, stream.getStream());
                    return;
                default:
                    throw std::logic_error("FP8 compact multi-output packed calibration received unsupported packet width.");
            }
        };
        switch (geometry.inner_size) {
            case 2: launch_k.template operator()<2>(); return;
            case 3: launch_k.template operator()<3>(); return;
            case 4: launch_k.template operator()<4>(); return;
            case 5: launch_k.template operator()<5>(); return;
            case 7: launch_k.template operator()<7>(); return;
            case 8: launch_k.template operator()<8>(); return;
            case 9: launch_k.template operator()<9>(); return;
            case 11: launch_k.template operator()<11>(); return;
            case 13: launch_k.template operator()<13>(); return;
            case 15: launch_k.template operator()<15>(); return;
            case 16: launch_k.template operator()<16>(); return;
            case 17: launch_k.template operator()<17>(); return;
            case 19: launch_k.template operator()<19>(); return;
            case 31: launch_k.template operator()<31>(); return;
            case 32: launch_k.template operator()<32>(); return;
            default:
                throw std::logic_error("FP8 compact multi-output packed calibration received unsupported K.");
        }
    };
    switch (input.getDataType()) {
        case DataType::FP8_E4M3:
            dispatch_k.template operator()<__nv_fp8_e4m3>();
            return;
        case DataType::FP8_E5M2:
            dispatch_k.template operator()<__nv_fp8_e5m2>();
            return;
        default:
            throw std::logic_error("FP8 compact multi-output packed calibration requires FP8 input.");
    }
#else
    static_cast<void>(input);
    static_cast<void>(output);
    static_cast<void>(geometry);
    static_cast<void>(packet_bytes);
    static_cast<void>(stream);
    throw std::logic_error("FP8 compact multi-output packed calibration requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}

CubKernelOccupancyInfo queryFp8CompactMultiOutputWarpRCooperativeCalibrationOccupancy(
    DataType input_dtype,
    uint64_t inner_size,
    uint32_t outputs_per_warp,
    uint32_t packet_bytes) {
#if THOR_CUB_ENABLE_FP8_TYPES
    if (packet_bytes == 4) {
        return queryFp8CompactMultiOutputWarpP4RCooperativeCalibrationOccupancy(
            input_dtype, inner_size, outputs_per_warp);
    }
    if ((packet_bytes != 8 && packet_bytes != 16) || outputs_per_warp <= 1 ||
        !shallowFp8SupportedCalibrationK(inner_size)) {
        throw std::logic_error("FP8 compact multi-output packed occupancy received unsupported geometry.");
    }
    const auto dispatch_k = [&]<typename InputT>() -> CubKernelOccupancyInfo {
        const auto query_k = [&]<int K>() -> CubKernelOccupancyInfo {
            switch (packet_bytes) {
                case 8:
                    return queryFp8CompactMultiOutputPackedOccupancyForOutputs<InputT, K, 8>(outputs_per_warp);
                case 16:
                    return queryFp8CompactMultiOutputPackedOccupancyForOutputs<InputT, K, 16>(outputs_per_warp);
                default:
                    throw std::logic_error("FP8 compact multi-output packed occupancy received unsupported packet width.");
            }
        };
        switch (inner_size) {
            case 2: return query_k.template operator()<2>();
            case 3: return query_k.template operator()<3>();
            case 4: return query_k.template operator()<4>();
            case 5: return query_k.template operator()<5>();
            case 7: return query_k.template operator()<7>();
            case 8: return query_k.template operator()<8>();
            case 9: return query_k.template operator()<9>();
            case 11: return query_k.template operator()<11>();
            case 13: return query_k.template operator()<13>();
            case 15: return query_k.template operator()<15>();
            case 16: return query_k.template operator()<16>();
            case 17: return query_k.template operator()<17>();
            case 19: return query_k.template operator()<19>();
            case 31: return query_k.template operator()<31>();
            case 32: return query_k.template operator()<32>();
            default:
                throw std::logic_error("FP8 compact multi-output packed occupancy received unsupported K.");
        }
    };
    switch (input_dtype) {
        case DataType::FP8_E4M3:
            return dispatch_k.template operator()<__nv_fp8_e4m3>();
        case DataType::FP8_E5M2:
            return dispatch_k.template operator()<__nv_fp8_e5m2>();
        default:
            throw std::logic_error("FP8 compact multi-output packed occupancy requires FP8 input.");
    }
#else
    static_cast<void>(input_dtype);
    static_cast<void>(inner_size);
    static_cast<void>(outputs_per_warp);
    static_cast<void>(packet_bytes);
    throw std::logic_error("FP8 compact multi-output packed occupancy requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}

void launchFp8RParallelWarpP4RCooperativeCalibrationComplete(const Tensor& input,
                                                             Tensor& output,
                                                             const CubReductionGeometry& geometry,
                                                             Stream& stream) {
#if THOR_CUB_ENABLE_FP8_TYPES
    if (geometry.outer_size == 0 || geometry.reduction_size == 0 || geometry.reduction_size > SHALLOW_FP8_MAX_R ||
        !shallowFp8SupportedCalibrationK(geometry.inner_size) || !shallowFp8FitsUint32Indexing(geometry) ||
        output.getDataType() != input.getDataType() || output.getDimensions() != geometry.output_dimensions) {
        throw std::logic_error("FP8 R-parallel p4 calibration received unsupported complete geometry.");
    }
    const auto dispatch_k = [&]<typename InputT>() {
        switch (geometry.inner_size) {
            case 2:
                launchFp8RParallelP4<InputT, 2>(input, output, geometry, stream.getStream());
                return;
            case 3:
                launchFp8RParallelP4<InputT, 3>(input, output, geometry, stream.getStream());
                return;
            case 4:
                launchFp8RParallelP4<InputT, 4>(input, output, geometry, stream.getStream());
                return;
            case 5:
                launchFp8RParallelP4<InputT, 5>(input, output, geometry, stream.getStream());
                return;
            case 7:
                launchFp8RParallelP4<InputT, 7>(input, output, geometry, stream.getStream());
                return;
            case 8:
                launchFp8RParallelP4<InputT, 8>(input, output, geometry, stream.getStream());
                return;
            case 9:
                launchFp8RParallelP4<InputT, 9>(input, output, geometry, stream.getStream());
                return;
            case 11:
                launchFp8RParallelP4<InputT, 11>(input, output, geometry, stream.getStream());
                return;
            case 13:
                launchFp8RParallelP4<InputT, 13>(input, output, geometry, stream.getStream());
                return;
            case 15:
                launchFp8RParallelP4<InputT, 15>(input, output, geometry, stream.getStream());
                return;
            case 16:
                launchFp8RParallelP4<InputT, 16>(input, output, geometry, stream.getStream());
                return;
            case 17:
                launchFp8RParallelP4<InputT, 17>(input, output, geometry, stream.getStream());
                return;
            case 19:
                launchFp8RParallelP4<InputT, 19>(input, output, geometry, stream.getStream());
                return;
            case 31:
                launchFp8RParallelP4<InputT, 31>(input, output, geometry, stream.getStream());
                return;
            case 32:
                launchFp8RParallelP4<InputT, 32>(input, output, geometry, stream.getStream());
                return;
            default:
                throw std::logic_error("FP8 R-parallel p4 calibration received unsupported K.");
        }
    };
    switch (input.getDataType()) {
        case DataType::FP8_E4M3:
            dispatch_k.template operator()<__nv_fp8_e4m3>();
            return;
        case DataType::FP8_E5M2:
            dispatch_k.template operator()<__nv_fp8_e5m2>();
            return;
        default:
            throw std::logic_error("FP8 R-parallel p4 calibration requires FP8 input.");
    }
#else
    static_cast<void>(input);
    static_cast<void>(output);
    static_cast<void>(geometry);
    static_cast<void>(stream);
    throw std::logic_error("FP8 R-parallel p4 calibration requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}

CubKernelOccupancyInfo queryFp8RParallelWarpP4RCooperativeCalibrationOccupancy(DataType input_dtype, uint64_t inner_size) {
#if THOR_CUB_ENABLE_FP8_TYPES
    const auto dispatch_k = [&]<typename InputT>() -> CubKernelOccupancyInfo {
        switch (inner_size) {
            case 2:
                return queryFp8RParallelP4Occupancy<InputT, 2>();
            case 3:
                return queryFp8RParallelP4Occupancy<InputT, 3>();
            case 4:
                return queryFp8RParallelP4Occupancy<InputT, 4>();
            case 5:
                return queryFp8RParallelP4Occupancy<InputT, 5>();
            case 7:
                return queryFp8RParallelP4Occupancy<InputT, 7>();
            case 8:
                return queryFp8RParallelP4Occupancy<InputT, 8>();
            case 9:
                return queryFp8RParallelP4Occupancy<InputT, 9>();
            case 11:
                return queryFp8RParallelP4Occupancy<InputT, 11>();
            case 13:
                return queryFp8RParallelP4Occupancy<InputT, 13>();
            case 15:
                return queryFp8RParallelP4Occupancy<InputT, 15>();
            case 16:
                return queryFp8RParallelP4Occupancy<InputT, 16>();
            case 17:
                return queryFp8RParallelP4Occupancy<InputT, 17>();
            case 19:
                return queryFp8RParallelP4Occupancy<InputT, 19>();
            case 31:
                return queryFp8RParallelP4Occupancy<InputT, 31>();
            case 32:
                return queryFp8RParallelP4Occupancy<InputT, 32>();
            default:
                throw std::logic_error("FP8 R-parallel p4 occupancy received unsupported K.");
        }
    };
    switch (input_dtype) {
        case DataType::FP8_E4M3:
            return dispatch_k.template operator()<__nv_fp8_e4m3>();
        case DataType::FP8_E5M2:
            return dispatch_k.template operator()<__nv_fp8_e5m2>();
        default:
            throw std::logic_error("FP8 R-parallel p4 occupancy requires FP8 input.");
    }
#else
    static_cast<void>(input_dtype);
    static_cast<void>(inner_size);
    throw std::logic_error("FP8 R-parallel p4 calibration requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}


void launchFp8MultiOutputRParallelWarpRCooperativeCalibrationComplete(
    const Tensor& input,
    Tensor& output,
    const CubReductionGeometry& geometry,
    uint32_t outputs_per_warp,
    uint32_t packet_bytes,
    Stream& stream) {
#if THOR_CUB_ENABLE_FP8_TYPES
    const uint32_t lanes_per_output =
        outputs_per_warp == 0 ? 0 : TILED_REDUCTION_WARP_THREADS / outputs_per_warp;
    if (geometry.outer_size == 0 || geometry.reduction_size == 0 || geometry.reduction_size > SHALLOW_FP8_MAX_R ||
        (geometry.inner_size != 1 && !shallowFp8SupportedCalibrationK(geometry.inner_size)) ||
        !shallowFp8FitsUint32Indexing(geometry) ||
        output.getDataType() != input.getDataType() || output.getDimensions() != geometry.output_dimensions ||
        (outputs_per_warp != 1 && outputs_per_warp != 2 && outputs_per_warp != 4 && outputs_per_warp != 8 &&
         outputs_per_warp != 16 && outputs_per_warp != 32) ||
        (packet_bytes != 4 && packet_bytes != 8 && packet_bytes != 16) ||
        (outputs_per_warp == 1 && packet_bytes == 4 && geometry.inner_size < 2) ||
        geometry.inner_size > lanes_per_output) {
        throw std::logic_error("FP8 multi-output RParallel calibration received unsupported complete geometry.");
    }

    const auto dispatch_k = [&]<typename InputT>() {
        const auto launch_k = [&]<int K>() {
            switch (packet_bytes) {
                case 4:
                    launchFp8MultiOutputRParallelP4ForOutputs<InputT, K>(
                        input, output, geometry, outputs_per_warp, stream.getStream());
                    return;
                case 8:
                    launchFp8MultiOutputRParallelPackedForOutputs<InputT, K, 8>(
                        input, output, geometry, outputs_per_warp, stream.getStream());
                    return;
                case 16:
                    launchFp8MultiOutputRParallelPackedForOutputs<InputT, K, 16>(
                        input, output, geometry, outputs_per_warp, stream.getStream());
                    return;
                default:
                    throw std::logic_error("FP8 multi-output RParallel calibration received unsupported packet width.");
            }
        };
        switch (geometry.inner_size) {
            case 1: launch_k.template operator()<1>(); return;
            case 2: launch_k.template operator()<2>(); return;
            case 3: launch_k.template operator()<3>(); return;
            case 4: launch_k.template operator()<4>(); return;
            case 5: launch_k.template operator()<5>(); return;
            case 7: launch_k.template operator()<7>(); return;
            case 8: launch_k.template operator()<8>(); return;
            case 9: launch_k.template operator()<9>(); return;
            case 11: launch_k.template operator()<11>(); return;
            case 13: launch_k.template operator()<13>(); return;
            case 15: launch_k.template operator()<15>(); return;
            case 16: launch_k.template operator()<16>(); return;
            case 17: launch_k.template operator()<17>(); return;
            case 19: launch_k.template operator()<19>(); return;
            case 31: launch_k.template operator()<31>(); return;
            case 32: launch_k.template operator()<32>(); return;
            default:
                throw std::logic_error("FP8 multi-output RParallel calibration received unsupported K.");
        }
    };
    switch (input.getDataType()) {
        case DataType::FP8_E4M3:
            dispatch_k.template operator()<__nv_fp8_e4m3>();
            return;
        case DataType::FP8_E5M2:
            dispatch_k.template operator()<__nv_fp8_e5m2>();
            return;
        default:
            throw std::logic_error("FP8 multi-output RParallel calibration requires FP8 input.");
    }
#else
    static_cast<void>(input);
    static_cast<void>(output);
    static_cast<void>(geometry);
    static_cast<void>(outputs_per_warp);
    static_cast<void>(packet_bytes);
    static_cast<void>(stream);
    throw std::logic_error("FP8 multi-output RParallel calibration requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}


void launchFp8GeneralRParallelRCooperativeCalibrationComplete(
    const Tensor& input,
    Tensor& output,
    const CubReductionGeometry& geometry,
    uint32_t outputs_per_warp,
    uint32_t warps_per_output,
    uint32_t packet_bytes,
    Stream& stream,
    bool linear_packets) {
#if THOR_CUB_ENABLE_FP8_TYPES
    const bool valid_outputs_per_warp =
        outputs_per_warp == 1 || outputs_per_warp == 2 || outputs_per_warp == 4 || outputs_per_warp == 8;
    const bool valid_warps_per_output =
        warps_per_output == 1 || warps_per_output == 2 || warps_per_output == 4 || warps_per_output == 8;
    const bool ownership_axes_are_exclusive = outputs_per_warp == 1 || warps_per_output == 1;
    if (geometry.outer_size == 0 || geometry.reduction_size == 0 || geometry.reduction_size > SHALLOW_FP8_MAX_R ||
        !rParallelFp8SupportedCalibrationK(geometry.inner_size) ||
        !rParallelFp8PacketSupportsOwnership(geometry.inner_size, outputs_per_warp, packet_bytes) ||
        !shallowFp8FitsUint32Indexing(geometry) ||
        output.getDataType() != input.getDataType() || output.getDimensions() != geometry.output_dimensions ||
        !valid_outputs_per_warp || !valid_warps_per_output || !ownership_axes_are_exclusive ||
        (linear_packets && outputs_per_warp != 1)) {
        throw std::logic_error("FP8 general RParallel calibration received unsupported complete geometry.");
    }

    const auto dispatch_k = [&]<typename InputT>() {
        const auto launch_k = [&]<int K>() {
            const auto launch_outputs = [&]<int OutputsPerWarp>() {
                constexpr uint32_t LanesPerOutput = TILED_REDUCTION_WARP_THREADS / OutputsPerWarp;
                switch (packet_bytes) {
                    case 4: {
                        constexpr uint32_t PacketWindow = LanesPerOutput * 4U;
                        constexpr uint32_t Capacity = (K % 4) == 0 ? PacketWindow : PacketWindow - 3U;
                        if constexpr (K <= static_cast<int>(Capacity)) {
                            if constexpr (OutputsPerWarp == 1) {
                                if (linear_packets) {
                                    launchFp8GeneralRParallelPackedForOwnership<InputT, K, 4, OutputsPerWarp, true>(
                                        input, output, geometry, warps_per_output, stream.getStream());
                                } else {
                                    launchFp8GeneralRParallelPackedForOwnership<InputT, K, 4, OutputsPerWarp, false>(
                                        input, output, geometry, warps_per_output, stream.getStream());
                                }
                            } else {
                                launchFp8GeneralRParallelPackedForOwnership<InputT, K, 4, OutputsPerWarp, false>(
                                    input, output, geometry, warps_per_output, stream.getStream());
                            }
                            return;
                        }
                        break;
                    }
                    case 8: {
                        constexpr uint32_t PacketWindow = LanesPerOutput * 8U;
                        constexpr uint32_t Capacity = (K % 8) == 0 ? PacketWindow : PacketWindow - 7U;
                        if constexpr (K <= static_cast<int>(Capacity)) {
                            if constexpr (OutputsPerWarp == 1) {
                                if (linear_packets) {
                                    launchFp8GeneralRParallelPackedForOwnership<InputT, K, 8, OutputsPerWarp, true>(
                                        input, output, geometry, warps_per_output, stream.getStream());
                                } else {
                                    launchFp8GeneralRParallelPackedForOwnership<InputT, K, 8, OutputsPerWarp, false>(
                                        input, output, geometry, warps_per_output, stream.getStream());
                                }
                            } else {
                                launchFp8GeneralRParallelPackedForOwnership<InputT, K, 8, OutputsPerWarp, false>(
                                    input, output, geometry, warps_per_output, stream.getStream());
                            }
                            return;
                        }
                        break;
                    }
                    case 16: {
                        constexpr uint32_t PacketWindow = LanesPerOutput * 16U;
                        constexpr uint32_t Capacity = (K % 16) == 0 ? PacketWindow : PacketWindow - 15U;
                        if constexpr (K <= static_cast<int>(Capacity)) {
                            if constexpr (OutputsPerWarp == 1) {
                                if (linear_packets) {
                                    launchFp8GeneralRParallelPackedForOwnership<InputT, K, 16, OutputsPerWarp, true>(
                                        input, output, geometry, warps_per_output, stream.getStream());
                                } else {
                                    launchFp8GeneralRParallelPackedForOwnership<InputT, K, 16, OutputsPerWarp, false>(
                                        input, output, geometry, warps_per_output, stream.getStream());
                                }
                            } else {
                                launchFp8GeneralRParallelPackedForOwnership<InputT, K, 16, OutputsPerWarp, false>(
                                    input, output, geometry, warps_per_output, stream.getStream());
                            }
                            return;
                        }
                        break;
                    }
                    default:
                        break;
                }
                throw std::logic_error("FP8 general RParallel calibration packet width cannot cover K/outputs ownership.");
            };
            switch (outputs_per_warp) {
                case 1: launch_outputs.template operator()<1>(); return;
                case 2: launch_outputs.template operator()<2>(); return;
                case 4: launch_outputs.template operator()<4>(); return;
                case 8: launch_outputs.template operator()<8>(); return;
                default:
                    throw std::logic_error("FP8 general RParallel calibration received invalid outputs-per-warp.");
            }
        };
        switch (geometry.inner_size) {
            case 1: launch_k.template operator()<1>(); return;
            case 2: launch_k.template operator()<2>(); return;
            case 3: launch_k.template operator()<3>(); return;
            case 4: launch_k.template operator()<4>(); return;
            case 5: launch_k.template operator()<5>(); return;
            case 7: launch_k.template operator()<7>(); return;
            case 8: launch_k.template operator()<8>(); return;
            case 9: launch_k.template operator()<9>(); return;
            case 11: launch_k.template operator()<11>(); return;
            case 13: launch_k.template operator()<13>(); return;
            case 15: launch_k.template operator()<15>(); return;
            case 16: launch_k.template operator()<16>(); return;
            case 17: launch_k.template operator()<17>(); return;
            case 19: launch_k.template operator()<19>(); return;
            case 31: launch_k.template operator()<31>(); return;
            case 32: launch_k.template operator()<32>(); return;
            case 63: launch_k.template operator()<63>(); return;
            case 64: launch_k.template operator()<64>(); return;
            case 127: launch_k.template operator()<127>(); return;
            case 128: launch_k.template operator()<128>(); return;
            case 129: launch_k.template operator()<129>(); return;
            case 255: launch_k.template operator()<255>(); return;
            case 256: launch_k.template operator()<256>(); return;
            default:
                throw std::logic_error("FP8 general RParallel calibration received unsupported K.");
        }
    };
    switch (input.getDataType()) {
        case DataType::FP8_E4M3:
            dispatch_k.template operator()<__nv_fp8_e4m3>();
            return;
        case DataType::FP8_E5M2:
            dispatch_k.template operator()<__nv_fp8_e5m2>();
            return;
        default:
            throw std::logic_error("FP8 general RParallel calibration requires FP8 input.");
    }
#else
    static_cast<void>(input);
    static_cast<void>(output);
    static_cast<void>(geometry);
    static_cast<void>(outputs_per_warp);
    static_cast<void>(warps_per_output);
    static_cast<void>(packet_bytes);
    static_cast<void>(stream);
    static_cast<void>(linear_packets);
    throw std::logic_error("FP8 general RParallel calibration requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}

CubKernelOccupancyInfo queryFp8GeneralRParallelRCooperativeCalibrationOccupancy(
    DataType input_dtype,
    uint64_t inner_size,
    uint32_t outputs_per_warp,
    uint32_t warps_per_output,
    uint32_t packet_bytes,
    bool linear_packets) {
#if THOR_CUB_ENABLE_FP8_TYPES
    const bool valid_outputs_per_warp =
        outputs_per_warp == 1 || outputs_per_warp == 2 || outputs_per_warp == 4 || outputs_per_warp == 8;
    const bool valid_warps_per_output =
        warps_per_output == 1 || warps_per_output == 2 || warps_per_output == 4 || warps_per_output == 8;
    if (!valid_outputs_per_warp || !valid_warps_per_output ||
        (outputs_per_warp > 1 && warps_per_output > 1) ||
        (linear_packets && outputs_per_warp != 1) ||
        !rParallelFp8SupportedCalibrationK(inner_size) ||
        !rParallelFp8PacketSupportsOwnership(inner_size, outputs_per_warp, packet_bytes)) {
        throw std::logic_error("FP8 general RParallel occupancy received unsupported ownership.");
    }

    const auto dispatch_k = [&]<typename InputT>() -> CubKernelOccupancyInfo {
        const auto query_k = [&]<int K>() -> CubKernelOccupancyInfo {
            const auto query_outputs = [&]<int OutputsPerWarp>() -> CubKernelOccupancyInfo {
                constexpr uint32_t LanesPerOutput = TILED_REDUCTION_WARP_THREADS / OutputsPerWarp;
                switch (packet_bytes) {
                    case 4: {
                        constexpr uint32_t PacketWindow = LanesPerOutput * 4U;
                        constexpr uint32_t Capacity = (K % 4) == 0 ? PacketWindow : PacketWindow - 3U;
                        if constexpr (K <= static_cast<int>(Capacity))
                            if constexpr (OutputsPerWarp == 1) {
                                return linear_packets
                                    ? queryFp8GeneralRParallelPackedOccupancyForOwnership<InputT, K, 4, OutputsPerWarp, true>(
                                          warps_per_output)
                                    : queryFp8GeneralRParallelPackedOccupancyForOwnership<InputT, K, 4, OutputsPerWarp, false>(
                                          warps_per_output);
                            } else {
                                return queryFp8GeneralRParallelPackedOccupancyForOwnership<InputT, K, 4, OutputsPerWarp, false>(
                                    warps_per_output);
                            }
                        break;
                    }
                    case 8: {
                        constexpr uint32_t PacketWindow = LanesPerOutput * 8U;
                        constexpr uint32_t Capacity = (K % 8) == 0 ? PacketWindow : PacketWindow - 7U;
                        if constexpr (K <= static_cast<int>(Capacity))
                            if constexpr (OutputsPerWarp == 1) {
                                return linear_packets
                                    ? queryFp8GeneralRParallelPackedOccupancyForOwnership<InputT, K, 8, OutputsPerWarp, true>(
                                          warps_per_output)
                                    : queryFp8GeneralRParallelPackedOccupancyForOwnership<InputT, K, 8, OutputsPerWarp, false>(
                                          warps_per_output);
                            } else {
                                return queryFp8GeneralRParallelPackedOccupancyForOwnership<InputT, K, 8, OutputsPerWarp, false>(
                                    warps_per_output);
                            }
                        break;
                    }
                    case 16: {
                        constexpr uint32_t PacketWindow = LanesPerOutput * 16U;
                        constexpr uint32_t Capacity = (K % 16) == 0 ? PacketWindow : PacketWindow - 15U;
                        if constexpr (K <= static_cast<int>(Capacity))
                            if constexpr (OutputsPerWarp == 1) {
                                return linear_packets
                                    ? queryFp8GeneralRParallelPackedOccupancyForOwnership<InputT, K, 16, OutputsPerWarp, true>(
                                          warps_per_output)
                                    : queryFp8GeneralRParallelPackedOccupancyForOwnership<InputT, K, 16, OutputsPerWarp, false>(
                                          warps_per_output);
                            } else {
                                return queryFp8GeneralRParallelPackedOccupancyForOwnership<InputT, K, 16, OutputsPerWarp, false>(
                                    warps_per_output);
                            }
                        break;
                    }
                    default:
                        break;
                }
                throw std::logic_error("FP8 general RParallel occupancy packet width cannot cover K/outputs ownership.");
            };
            switch (outputs_per_warp) {
                case 1: return query_outputs.template operator()<1>();
                case 2: return query_outputs.template operator()<2>();
                case 4: return query_outputs.template operator()<4>();
                case 8: return query_outputs.template operator()<8>();
                default:
                    throw std::logic_error("FP8 general RParallel occupancy received invalid outputs-per-warp.");
            }
        };
        switch (inner_size) {
            case 1: return query_k.template operator()<1>();
            case 2: return query_k.template operator()<2>();
            case 3: return query_k.template operator()<3>();
            case 4: return query_k.template operator()<4>();
            case 5: return query_k.template operator()<5>();
            case 7: return query_k.template operator()<7>();
            case 8: return query_k.template operator()<8>();
            case 9: return query_k.template operator()<9>();
            case 11: return query_k.template operator()<11>();
            case 13: return query_k.template operator()<13>();
            case 15: return query_k.template operator()<15>();
            case 16: return query_k.template operator()<16>();
            case 17: return query_k.template operator()<17>();
            case 19: return query_k.template operator()<19>();
            case 31: return query_k.template operator()<31>();
            case 32: return query_k.template operator()<32>();
            case 63: return query_k.template operator()<63>();
            case 64: return query_k.template operator()<64>();
            case 127: return query_k.template operator()<127>();
            case 128: return query_k.template operator()<128>();
            case 129: return query_k.template operator()<129>();
            case 255: return query_k.template operator()<255>();
            case 256: return query_k.template operator()<256>();
            default:
                throw std::logic_error("FP8 general RParallel occupancy received unsupported K.");
        }
    };
    switch (input_dtype) {
        case DataType::FP8_E4M3:
            return dispatch_k.template operator()<__nv_fp8_e4m3>();
        case DataType::FP8_E5M2:
            return dispatch_k.template operator()<__nv_fp8_e5m2>();
        default:
            throw std::logic_error("FP8 general RParallel occupancy requires FP8 input.");
    }
#else
    static_cast<void>(input_dtype);
    static_cast<void>(inner_size);
    static_cast<void>(outputs_per_warp);
    static_cast<void>(warps_per_output);
    static_cast<void>(packet_bytes);
    static_cast<void>(linear_packets);
    throw std::logic_error("FP8 general RParallel calibration requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}


void launchFp8GeneralRParallelRCooperativeCalibrationFirstStage(
    const Tensor& input,
    Tensor& fp32_partials,
    const CubReductionGeometry& geometry,
    uint32_t warps_per_output,
    uint32_t packet_bytes,
    uint32_t shards_per_output,
    Stream& stream,
    bool linear_packets) {
#if THOR_CUB_ENABLE_FP8_TYPES
    const bool valid_warps_per_output =
        warps_per_output == 1 || warps_per_output == 2 || warps_per_output == 4 || warps_per_output == 8;
    const bool valid_shards_per_output =
        shards_per_output == 2 || shards_per_output == 4 || shards_per_output == 8;
    const std::vector<uint64_t> expected_dimensions = {
        geometry.outer_size, shards_per_output, geometry.inner_size};
    if (geometry.outer_size == 0 || geometry.reduction_size == 0 || geometry.reduction_size > SHALLOW_FP8_MAX_R ||
        shards_per_output > geometry.reduction_size || !valid_warps_per_output || !valid_shards_per_output ||
        !rParallelFp8SupportedCalibrationK(geometry.inner_size) ||
        !rParallelFp8PacketSupportsOwnership(geometry.inner_size, 1, packet_bytes) ||
        !shallowFp8FitsUint32Indexing(geometry) || fp32_partials.getDataType() != DataType::FP32 ||
        fp32_partials.getDimensions() != expected_dimensions) {
        throw std::logic_error("FP8 staged general RParallel calibration received unsupported geometry.");
    }

    const auto dispatch_k = [&]<typename InputT>() {
        const auto launch_k = [&]<int K>() {
            switch (packet_bytes) {
                case 4: {
                    constexpr uint32_t Capacity = (K % 4) == 0 ? 128U : 125U;
                    if constexpr (K <= static_cast<int>(Capacity)) {
                        if (linear_packets) {
                            launchFp8GeneralRParallelPackedFirstStage<InputT, K, 4, true>(
                                input, fp32_partials, geometry, warps_per_output, shards_per_output, stream.getStream());
                        } else {
                            launchFp8GeneralRParallelPackedFirstStage<InputT, K, 4, false>(
                                input, fp32_partials, geometry, warps_per_output, shards_per_output, stream.getStream());
                        }
                        return;
                    }
                    break;
                }
                case 8: {
                    constexpr uint32_t Capacity = (K % 8) == 0 ? 256U : 249U;
                    if constexpr (K <= static_cast<int>(Capacity)) {
                        if (linear_packets) {
                            launchFp8GeneralRParallelPackedFirstStage<InputT, K, 8, true>(
                                input, fp32_partials, geometry, warps_per_output, shards_per_output, stream.getStream());
                        } else {
                            launchFp8GeneralRParallelPackedFirstStage<InputT, K, 8, false>(
                                input, fp32_partials, geometry, warps_per_output, shards_per_output, stream.getStream());
                        }
                        return;
                    }
                    break;
                }
                case 16: {
                    constexpr uint32_t Capacity = (K % 16) == 0 ? 512U : 497U;
                    if constexpr (K <= static_cast<int>(Capacity)) {
                        if (linear_packets) {
                            launchFp8GeneralRParallelPackedFirstStage<InputT, K, 16, true>(
                                input, fp32_partials, geometry, warps_per_output, shards_per_output, stream.getStream());
                        } else {
                            launchFp8GeneralRParallelPackedFirstStage<InputT, K, 16, false>(
                                input, fp32_partials, geometry, warps_per_output, shards_per_output, stream.getStream());
                        }
                        return;
                    }
                    break;
                }
                default:
                    break;
            }
            throw std::logic_error("FP8 staged general RParallel packet width cannot cover K.");
        };
        switch (geometry.inner_size) {
            case 1: launch_k.template operator()<1>(); return;
            case 2: launch_k.template operator()<2>(); return;
            case 3: launch_k.template operator()<3>(); return;
            case 4: launch_k.template operator()<4>(); return;
            case 5: launch_k.template operator()<5>(); return;
            case 7: launch_k.template operator()<7>(); return;
            case 8: launch_k.template operator()<8>(); return;
            case 9: launch_k.template operator()<9>(); return;
            case 11: launch_k.template operator()<11>(); return;
            case 13: launch_k.template operator()<13>(); return;
            case 15: launch_k.template operator()<15>(); return;
            case 16: launch_k.template operator()<16>(); return;
            case 17: launch_k.template operator()<17>(); return;
            case 19: launch_k.template operator()<19>(); return;
            case 31: launch_k.template operator()<31>(); return;
            case 32: launch_k.template operator()<32>(); return;
            case 63: launch_k.template operator()<63>(); return;
            case 64: launch_k.template operator()<64>(); return;
            case 127: launch_k.template operator()<127>(); return;
            case 128: launch_k.template operator()<128>(); return;
            case 129: launch_k.template operator()<129>(); return;
            case 255: launch_k.template operator()<255>(); return;
            case 256: launch_k.template operator()<256>(); return;
            default:
                throw std::logic_error("FP8 staged general RParallel received unsupported K.");
        }
    };
    switch (input.getDataType()) {
        case DataType::FP8_E4M3: dispatch_k.template operator()<__nv_fp8_e4m3>(); return;
        case DataType::FP8_E5M2: dispatch_k.template operator()<__nv_fp8_e5m2>(); return;
        default:
            throw std::logic_error("FP8 staged general RParallel calibration requires FP8 input.");
    }
#else
    static_cast<void>(input);
    static_cast<void>(fp32_partials);
    static_cast<void>(geometry);
    static_cast<void>(warps_per_output);
    static_cast<void>(packet_bytes);
    static_cast<void>(shards_per_output);
    static_cast<void>(stream);
    static_cast<void>(linear_packets);
    throw std::logic_error("FP8 staged general RParallel calibration requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}

CubKernelOccupancyInfo queryFp8GeneralRParallelRCooperativeCalibrationFirstStageOccupancy(
    DataType input_dtype,
    uint64_t inner_size,
    uint32_t warps_per_output,
    uint32_t packet_bytes,
    bool linear_packets) {
#if THOR_CUB_ENABLE_FP8_TYPES
    const bool valid_warps_per_output =
        warps_per_output == 1 || warps_per_output == 2 || warps_per_output == 4 || warps_per_output == 8;
    if (!valid_warps_per_output || !rParallelFp8SupportedCalibrationK(inner_size) ||
        !rParallelFp8PacketSupportsOwnership(inner_size, 1, packet_bytes)) {
        throw std::logic_error("FP8 staged general RParallel occupancy received unsupported ownership.");
    }
    static_cast<void>(warps_per_output);  // Runtime W changes work grouping but not kernel resource usage.

    const auto dispatch_k = [&]<typename InputT>() -> CubKernelOccupancyInfo {
        const auto query_k = [&]<int K>() -> CubKernelOccupancyInfo {
            switch (packet_bytes) {
                case 4:
                    if constexpr (K <= ((K % 4) == 0 ? 128 : 125))
                        return linear_packets
                            ? queryFp8GeneralRParallelPackedFirstStageOccupancy<InputT, K, 4, true>()
                            : queryFp8GeneralRParallelPackedFirstStageOccupancy<InputT, K, 4, false>();
                    break;
                case 8:
                    if constexpr (K <= ((K % 8) == 0 ? 256 : 249))
                        return linear_packets
                            ? queryFp8GeneralRParallelPackedFirstStageOccupancy<InputT, K, 8, true>()
                            : queryFp8GeneralRParallelPackedFirstStageOccupancy<InputT, K, 8, false>();
                    break;
                case 16:
                    if constexpr (K <= ((K % 16) == 0 ? 512 : 497))
                        return linear_packets
                            ? queryFp8GeneralRParallelPackedFirstStageOccupancy<InputT, K, 16, true>()
                            : queryFp8GeneralRParallelPackedFirstStageOccupancy<InputT, K, 16, false>();
                    break;
                default:
                    break;
            }
            throw std::logic_error("FP8 staged general RParallel occupancy packet width cannot cover K.");
        };
        switch (inner_size) {
            case 1: return query_k.template operator()<1>();
            case 2: return query_k.template operator()<2>();
            case 3: return query_k.template operator()<3>();
            case 4: return query_k.template operator()<4>();
            case 5: return query_k.template operator()<5>();
            case 7: return query_k.template operator()<7>();
            case 8: return query_k.template operator()<8>();
            case 9: return query_k.template operator()<9>();
            case 11: return query_k.template operator()<11>();
            case 13: return query_k.template operator()<13>();
            case 15: return query_k.template operator()<15>();
            case 16: return query_k.template operator()<16>();
            case 17: return query_k.template operator()<17>();
            case 19: return query_k.template operator()<19>();
            case 31: return query_k.template operator()<31>();
            case 32: return query_k.template operator()<32>();
            case 63: return query_k.template operator()<63>();
            case 64: return query_k.template operator()<64>();
            case 127: return query_k.template operator()<127>();
            case 128: return query_k.template operator()<128>();
            case 129: return query_k.template operator()<129>();
            case 255: return query_k.template operator()<255>();
            case 256: return query_k.template operator()<256>();
            default:
                throw std::logic_error("FP8 staged general RParallel occupancy received unsupported K.");
        }
    };
    switch (input_dtype) {
        case DataType::FP8_E4M3: return dispatch_k.template operator()<__nv_fp8_e4m3>();
        case DataType::FP8_E5M2: return dispatch_k.template operator()<__nv_fp8_e5m2>();
        default:
            throw std::logic_error("FP8 staged general RParallel occupancy requires FP8 input.");
    }
#else
    static_cast<void>(input_dtype);
    static_cast<void>(inner_size);
    static_cast<void>(warps_per_output);
    static_cast<void>(packet_bytes);
    throw std::logic_error("FP8 staged general RParallel calibration requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}

// Compatibility entry points for the earlier small-K W2/W4/W8 calibration. They now route through the same general
// physical kernel as W1 and medium K so calibration does not preserve a second multi-warp implementation.
void launchFp8MultiWarpRParallelRCooperativeCalibrationComplete(
    const Tensor& input,
    Tensor& output,
    const CubReductionGeometry& geometry,
    uint32_t warps_per_output,
    uint32_t packet_bytes,
    Stream& stream) {
    if (warps_per_output == 1 || geometry.inner_size > SHALLOW_FP8_MAX_K) {
        throw std::logic_error("FP8 multi-warp compatibility entry point only supports small-K W2/W4/W8.");
    }
    launchFp8GeneralRParallelRCooperativeCalibrationComplete(
        input, output, geometry, 1, warps_per_output, packet_bytes, stream);
}

CubKernelOccupancyInfo queryFp8MultiWarpRParallelRCooperativeCalibrationOccupancy(
    DataType input_dtype,
    uint64_t inner_size,
    uint32_t warps_per_output,
    uint32_t packet_bytes) {
    if (warps_per_output == 1 || inner_size > SHALLOW_FP8_MAX_K) {
        throw std::logic_error("FP8 multi-warp compatibility occupancy only supports small-K W2/W4/W8.");
    }
    return queryFp8GeneralRParallelRCooperativeCalibrationOccupancy(
        input_dtype, inner_size, 1, warps_per_output, packet_bytes);
}

void launchFp8MultiOutputRParallelWarpP4RCooperativeCalibrationComplete(
    const Tensor& input,
    Tensor& output,
    const CubReductionGeometry& geometry,
    uint32_t outputs_per_warp,
    Stream& stream) {
    launchFp8MultiOutputRParallelWarpRCooperativeCalibrationComplete(
        input, output, geometry, outputs_per_warp, 4, stream);
}

CubKernelOccupancyInfo queryFp8MultiOutputRParallelWarpRCooperativeCalibrationOccupancy(
    DataType input_dtype,
    uint64_t inner_size,
    uint32_t outputs_per_warp,
    uint32_t packet_bytes) {
#if THOR_CUB_ENABLE_FP8_TYPES
    const uint32_t lanes_per_output =
        outputs_per_warp == 0 ? 0 : TILED_REDUCTION_WARP_THREADS / outputs_per_warp;
    if ((outputs_per_warp != 1 && outputs_per_warp != 2 && outputs_per_warp != 4 && outputs_per_warp != 8 &&
         outputs_per_warp != 16 && outputs_per_warp != 32) ||
        (packet_bytes != 4 && packet_bytes != 8 && packet_bytes != 16) ||
        inner_size == 0 ||
        (outputs_per_warp == 1 && packet_bytes == 4 && inner_size < 2) ||
        inner_size > lanes_per_output) {
        throw std::logic_error("FP8 multi-output RParallel occupancy received unsupported ownership.");
    }

    const auto dispatch_k = [&]<typename InputT>() -> CubKernelOccupancyInfo {
        const auto query_k = [&]<int K>() -> CubKernelOccupancyInfo {
            switch (packet_bytes) {
                case 4:
                    return queryFp8MultiOutputRParallelP4OccupancyForOutputs<InputT, K>(outputs_per_warp);
                case 8:
                    return queryFp8MultiOutputRParallelPackedOccupancyForOutputs<InputT, K, 8>(outputs_per_warp);
                case 16:
                    return queryFp8MultiOutputRParallelPackedOccupancyForOutputs<InputT, K, 16>(outputs_per_warp);
                default:
                    throw std::logic_error("FP8 multi-output RParallel occupancy received unsupported packet width.");
            }
        };
        switch (inner_size) {
            case 1: return query_k.template operator()<1>();
            case 2: return query_k.template operator()<2>();
            case 3: return query_k.template operator()<3>();
            case 4: return query_k.template operator()<4>();
            case 5: return query_k.template operator()<5>();
            case 7: return query_k.template operator()<7>();
            case 8: return query_k.template operator()<8>();
            case 9: return query_k.template operator()<9>();
            case 11: return query_k.template operator()<11>();
            case 13: return query_k.template operator()<13>();
            case 15: return query_k.template operator()<15>();
            case 16: return query_k.template operator()<16>();
            case 17: return query_k.template operator()<17>();
            case 19: return query_k.template operator()<19>();
            case 31: return query_k.template operator()<31>();
            case 32: return query_k.template operator()<32>();
            default:
                throw std::logic_error("FP8 multi-output RParallel occupancy received unsupported K.");
        }
    };
    switch (input_dtype) {
        case DataType::FP8_E4M3:
            return dispatch_k.template operator()<__nv_fp8_e4m3>();
        case DataType::FP8_E5M2:
            return dispatch_k.template operator()<__nv_fp8_e5m2>();
        default:
            throw std::logic_error("FP8 multi-output RParallel occupancy requires FP8 input.");
    }
#else
    static_cast<void>(input_dtype);
    static_cast<void>(inner_size);
    static_cast<void>(outputs_per_warp);
    static_cast<void>(packet_bytes);
    throw std::logic_error("FP8 multi-output RParallel calibration requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}

CubKernelOccupancyInfo queryFp8MultiOutputRParallelWarpP4RCooperativeCalibrationOccupancy(
    DataType input_dtype,
    uint64_t inner_size,
    uint32_t outputs_per_warp) {
    return queryFp8MultiOutputRParallelWarpRCooperativeCalibrationOccupancy(
        input_dtype, inner_size, outputs_per_warp, 4);
}

void launchFp8K2PairedRowsRParallelP4RCooperativeCalibrationComplete(const Tensor& input,
                                                                     Tensor& output,
                                                                     const CubReductionGeometry& geometry,
                                                                     Stream& stream) {
#if THOR_CUB_ENABLE_FP8_TYPES
    if (geometry.outer_size == 0 || geometry.inner_size != 2 || output.getDataType() != input.getDataType() ||
        output.getDimensions() != geometry.output_dimensions ||
        (geometry.reduction_size != 31 && geometry.reduction_size != 32 &&
         geometry.reduction_size != 63 && geometry.reduction_size != 64)) {
        throw std::logic_error("FP8 K=2 paired-row R-parallel p4 calibration requires R in {31,32,63,64} and K=2.");
    }
    const uintptr_t pointer_bits = reinterpret_cast<uintptr_t>(input.getMemPtr<void>()) |
                                   reinterpret_cast<uintptr_t>(output.getMemPtr<void>());
    if ((pointer_bits & uintptr_t{SHALLOW_FP8_PACKET_BYTES - 1}) != 0) {
        throw std::logic_error("FP8 K=2 paired-row R-parallel p4 calibration requires p4-aligned tensor bases.");
    }
    const auto dispatch_r = [&]<typename InputT>() {
        switch (geometry.reduction_size) {
            case 31: launchFp8K2PairedRowsRParallelP4<InputT, 31>(input, output, geometry, stream.getStream()); return;
            case 32: launchFp8K2PairedRowsRParallelP4<InputT, 32>(input, output, geometry, stream.getStream()); return;
            case 63: launchFp8K2PairedRowsRParallelP4<InputT, 63>(input, output, geometry, stream.getStream()); return;
            case 64: launchFp8K2PairedRowsRParallelP4<InputT, 64>(input, output, geometry, stream.getStream()); return;
            default: throw std::logic_error("Unsupported R for FP8 K=2 paired-row R-parallel calibration.");
        }
    };
    switch (input.getDataType()) {
        case DataType::FP8_E4M3: dispatch_r.template operator()<__nv_fp8_e4m3>(); return;
        case DataType::FP8_E5M2: dispatch_r.template operator()<__nv_fp8_e5m2>(); return;
        default: throw std::logic_error("FP8 K=2 paired-row R-parallel calibration requires FP8 input.");
    }
#else
    static_cast<void>(input);
    static_cast<void>(output);
    static_cast<void>(geometry);
    static_cast<void>(stream);
    throw std::logic_error("FP8 K=2 paired-row R-parallel calibration requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}

CubKernelOccupancyInfo queryFp8K2PairedRowsRParallelP4RCooperativeCalibrationOccupancy(DataType input_dtype,
                                                                                       uint64_t reduction_size) {
#if THOR_CUB_ENABLE_FP8_TYPES
    const auto dispatch_r = [&]<typename InputT>() -> CubKernelOccupancyInfo {
        switch (reduction_size) {
            case 31: return queryFp8K2PairedRowsRParallelP4Occupancy<InputT, 31>();
            case 32: return queryFp8K2PairedRowsRParallelP4Occupancy<InputT, 32>();
            case 63: return queryFp8K2PairedRowsRParallelP4Occupancy<InputT, 63>();
            case 64: return queryFp8K2PairedRowsRParallelP4Occupancy<InputT, 64>();
            default: throw std::logic_error("Unsupported R for FP8 K=2 paired-row R-parallel occupancy.");
        }
    };
    switch (input_dtype) {
        case DataType::FP8_E4M3: return dispatch_r.template operator()<__nv_fp8_e4m3>();
        case DataType::FP8_E5M2: return dispatch_r.template operator()<__nv_fp8_e5m2>();
        default: throw std::logic_error("FP8 K=2 paired-row R-parallel occupancy requires FP8 input.");
    }
#else
    static_cast<void>(input_dtype);
    static_cast<void>(reduction_size);
    throw std::logic_error("FP8 K=2 paired-row R-parallel calibration requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}


void launchFp8K3PhaseStreamRParallelP4RCooperativeCalibrationComplete(const Tensor& input,
                                                                      Tensor& output,
                                                                      const CubReductionGeometry& geometry,
                                                                      Stream& stream) {
#if THOR_CUB_ENABLE_FP8_TYPES
    if (geometry.outer_size == 0 || geometry.inner_size != 3 || output.getDataType() != input.getDataType() ||
        output.getDimensions() != geometry.output_dimensions ||
        (geometry.reduction_size != 7 && geometry.reduction_size != 8 &&
         geometry.reduction_size != 15 && geometry.reduction_size != 16 &&
         geometry.reduction_size != 31 && geometry.reduction_size != 32 &&
         geometry.reduction_size != 63 && geometry.reduction_size != 64)) {
        throw std::logic_error("FP8 K=3 phase-stream R-parallel p4 calibration requires R in {7,8,15,16,31,32,63,64} and K=3.");
    }
    const uintptr_t pointer_bits = reinterpret_cast<uintptr_t>(input.getMemPtr<void>()) |
                                   reinterpret_cast<uintptr_t>(output.getMemPtr<void>());
    if ((pointer_bits & uintptr_t{SHALLOW_FP8_PACKET_BYTES - 1}) != 0) {
        throw std::logic_error("FP8 K=3 phase-stream R-parallel p4 calibration requires p4-aligned tensor bases.");
    }
    const auto dispatch_r = [&]<typename InputT>() {
        switch (geometry.reduction_size) {
            case 7: launchFp8K3PhaseStreamRParallelP4<InputT, 7>(input, output, geometry, stream.getStream()); return;
            case 8: launchFp8K3PhaseStreamRParallelP4<InputT, 8>(input, output, geometry, stream.getStream()); return;
            case 15: launchFp8K3PhaseStreamRParallelP4<InputT, 15>(input, output, geometry, stream.getStream()); return;
            case 16: launchFp8K3PhaseStreamRParallelP4<InputT, 16>(input, output, geometry, stream.getStream()); return;
            case 31: launchFp8K3PhaseStreamRParallelP4<InputT, 31>(input, output, geometry, stream.getStream()); return;
            case 32: launchFp8K3PhaseStreamRParallelP4<InputT, 32>(input, output, geometry, stream.getStream()); return;
            case 63: launchFp8K3PhaseStreamRParallelP4<InputT, 63>(input, output, geometry, stream.getStream()); return;
            case 64: launchFp8K3PhaseStreamRParallelP4<InputT, 64>(input, output, geometry, stream.getStream()); return;
            default: throw std::logic_error("Unsupported R for FP8 K=3 phase-stream R-parallel calibration.");
        }
    };
    switch (input.getDataType()) {
        case DataType::FP8_E4M3: dispatch_r.template operator()<__nv_fp8_e4m3>(); return;
        case DataType::FP8_E5M2: dispatch_r.template operator()<__nv_fp8_e5m2>(); return;
        default: throw std::logic_error("FP8 K=3 phase-stream R-parallel calibration requires FP8 input.");
    }
#else
    static_cast<void>(input);
    static_cast<void>(output);
    static_cast<void>(geometry);
    static_cast<void>(stream);
    throw std::logic_error("FP8 K=3 phase-stream R-parallel calibration requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}

CubKernelOccupancyInfo queryFp8K3PhaseStreamRParallelP4RCooperativeCalibrationOccupancy(DataType input_dtype,
                                                                                        uint64_t reduction_size) {
#if THOR_CUB_ENABLE_FP8_TYPES
    const auto dispatch_r = [&]<typename InputT>() -> CubKernelOccupancyInfo {
        switch (reduction_size) {
            case 7: return queryFp8K3PhaseStreamRParallelP4Occupancy<InputT, 7>();
            case 8: return queryFp8K3PhaseStreamRParallelP4Occupancy<InputT, 8>();
            case 15: return queryFp8K3PhaseStreamRParallelP4Occupancy<InputT, 15>();
            case 16: return queryFp8K3PhaseStreamRParallelP4Occupancy<InputT, 16>();
            case 31: return queryFp8K3PhaseStreamRParallelP4Occupancy<InputT, 31>();
            case 32: return queryFp8K3PhaseStreamRParallelP4Occupancy<InputT, 32>();
            case 63: return queryFp8K3PhaseStreamRParallelP4Occupancy<InputT, 63>();
            case 64: return queryFp8K3PhaseStreamRParallelP4Occupancy<InputT, 64>();
            default: throw std::logic_error("Unsupported R for FP8 K=3 phase-stream R-parallel occupancy.");
        }
    };
    switch (input_dtype) {
        case DataType::FP8_E4M3: return dispatch_r.template operator()<__nv_fp8_e4m3>();
        case DataType::FP8_E5M2: return dispatch_r.template operator()<__nv_fp8_e5m2>();
        default: throw std::logic_error("FP8 K=3 phase-stream R-parallel occupancy requires FP8 input.");
    }
#else
    static_cast<void>(input_dtype);
    static_cast<void>(reduction_size);
    throw std::logic_error("FP8 K=3 phase-stream R-parallel calibration requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}


void launchFp8K3PhaseStreamRParallelP8InputRCooperativeCalibrationComplete(const Tensor& input,
                                                                           Tensor& output,
                                                                           const CubReductionGeometry& geometry,
                                                                           Stream& stream) {
#if THOR_CUB_ENABLE_FP8_TYPES
    if (geometry.outer_size == 0 || geometry.inner_size != 3 || output.getDataType() != input.getDataType() ||
        output.getDimensions() != geometry.output_dimensions ||
        (geometry.reduction_size != 8 && geometry.reduction_size != 16 &&
         geometry.reduction_size != 32 && geometry.reduction_size != 64)) {
        throw std::logic_error("FP8 K=3 phase-stream p8-input calibration requires R in {8,16,32,64} and K=3.");
    }
    const uintptr_t pointer_bits = reinterpret_cast<uintptr_t>(input.getMemPtr<void>());
    if ((pointer_bits & uintptr_t{7}) != 0) {
        throw std::logic_error("FP8 K=3 phase-stream p8-input calibration requires a p8-aligned input tensor base.");
    }
    const auto dispatch_r = [&]<typename InputT>() {
        switch (geometry.reduction_size) {
            case 8: launchFp8K3PhaseStreamRParallelP8Input<InputT, 8>(input, output, geometry, stream.getStream()); return;
            case 16: launchFp8K3PhaseStreamRParallelP8Input<InputT, 16>(input, output, geometry, stream.getStream()); return;
            case 32: launchFp8K3PhaseStreamRParallelP8Input<InputT, 32>(input, output, geometry, stream.getStream()); return;
            case 64: launchFp8K3PhaseStreamRParallelP8Input<InputT, 64>(input, output, geometry, stream.getStream()); return;
            default: throw std::logic_error("Unsupported R for FP8 K=3 phase-stream p8-input calibration.");
        }
    };
    switch (input.getDataType()) {
        case DataType::FP8_E4M3: dispatch_r.template operator()<__nv_fp8_e4m3>(); return;
        case DataType::FP8_E5M2: dispatch_r.template operator()<__nv_fp8_e5m2>(); return;
        default: throw std::logic_error("FP8 K=3 phase-stream p8-input calibration requires FP8 input.");
    }
#else
    static_cast<void>(input);
    static_cast<void>(output);
    static_cast<void>(geometry);
    static_cast<void>(stream);
    throw std::logic_error("FP8 K=3 phase-stream p8-input calibration requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}

CubKernelOccupancyInfo queryFp8K3PhaseStreamRParallelP8InputRCooperativeCalibrationOccupancy(
    DataType input_dtype, uint64_t reduction_size) {
#if THOR_CUB_ENABLE_FP8_TYPES
    const auto dispatch_r = [&]<typename InputT>() -> CubKernelOccupancyInfo {
        switch (reduction_size) {
            case 8: return queryFp8K3PhaseStreamRParallelP8InputOccupancy<InputT, 8>();
            case 16: return queryFp8K3PhaseStreamRParallelP8InputOccupancy<InputT, 16>();
            case 32: return queryFp8K3PhaseStreamRParallelP8InputOccupancy<InputT, 32>();
            case 64: return queryFp8K3PhaseStreamRParallelP8InputOccupancy<InputT, 64>();
            default: throw std::logic_error("Unsupported R for FP8 K=3 phase-stream p8-input occupancy.");
        }
    };
    switch (input_dtype) {
        case DataType::FP8_E4M3: return dispatch_r.template operator()<__nv_fp8_e4m3>();
        case DataType::FP8_E5M2: return dispatch_r.template operator()<__nv_fp8_e5m2>();
        default: throw std::logic_error("FP8 K=3 phase-stream p8-input occupancy requires FP8 input.");
    }
#else
    static_cast<void>(input_dtype);
    static_cast<void>(reduction_size);
    throw std::logic_error("FP8 K=3 phase-stream p8-input calibration requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}


void launchFp8K3RuntimeRPhaseStreamRParallelP4RCooperativeCalibrationComplete(const Tensor& input,
                                                                              Tensor& output,
                                                                              const CubReductionGeometry& geometry,
                                                                              Stream& stream) {
#if THOR_CUB_ENABLE_FP8_TYPES
    if (geometry.outer_size == 0 || geometry.reduction_size == 0 || geometry.inner_size != 3 ||
        output.getDataType() != input.getDataType() || output.getDimensions() != geometry.output_dimensions) {
        throw std::logic_error("FP8 K=3 runtime-R phase-stream p4 calibration requires non-empty R and K=3.");
    }
    const uintptr_t pointer_bits = reinterpret_cast<uintptr_t>(input.getMemPtr<void>()) |
                                   reinterpret_cast<uintptr_t>(output.getMemPtr<void>());
    if ((pointer_bits & uintptr_t{3}) != 0) {
        throw std::logic_error("FP8 K=3 runtime-R phase-stream p4 calibration requires p4-aligned tensor bases.");
    }
    switch (input.getDataType()) {
        case DataType::FP8_E4M3:
            launchFp8K3RuntimeRPhaseStream<__nv_fp8_e4m3, 4>(input, output, geometry, stream.getStream());
            return;
        case DataType::FP8_E5M2:
            launchFp8K3RuntimeRPhaseStream<__nv_fp8_e5m2, 4>(input, output, geometry, stream.getStream());
            return;
        default:
            throw std::logic_error("FP8 K=3 runtime-R phase-stream p4 calibration requires FP8 input.");
    }
#else
    static_cast<void>(input);
    static_cast<void>(output);
    static_cast<void>(geometry);
    static_cast<void>(stream);
    throw std::logic_error("FP8 K=3 runtime-R phase-stream p4 calibration requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}

void launchFp8K3RuntimeRPhaseStreamRParallelP8InputRCooperativeCalibrationComplete(const Tensor& input,
                                                                                   Tensor& output,
                                                                                   const CubReductionGeometry& geometry,
                                                                                   Stream& stream) {
#if THOR_CUB_ENABLE_FP8_TYPES
    if (geometry.outer_size == 0 || geometry.reduction_size == 0 || geometry.inner_size != 3 ||
        output.getDataType() != input.getDataType() || output.getDimensions() != geometry.output_dimensions) {
        throw std::logic_error("FP8 K=3 runtime-R phase-stream p8 calibration requires non-empty R and K=3.");
    }
    const uintptr_t pointer_bits = reinterpret_cast<uintptr_t>(input.getMemPtr<void>()) |
                                   reinterpret_cast<uintptr_t>(output.getMemPtr<void>());
    if ((pointer_bits & uintptr_t{7}) != 0) {
        throw std::logic_error("FP8 K=3 runtime-R phase-stream p8 calibration requires p8-aligned tensor bases.");
    }
    switch (input.getDataType()) {
        case DataType::FP8_E4M3:
            launchFp8K3RuntimeRPhaseStream<__nv_fp8_e4m3, 8>(input, output, geometry, stream.getStream());
            return;
        case DataType::FP8_E5M2:
            launchFp8K3RuntimeRPhaseStream<__nv_fp8_e5m2, 8>(input, output, geometry, stream.getStream());
            return;
        default:
            throw std::logic_error("FP8 K=3 runtime-R phase-stream p8 calibration requires FP8 input.");
    }
#else
    static_cast<void>(input);
    static_cast<void>(output);
    static_cast<void>(geometry);
    static_cast<void>(stream);
    throw std::logic_error("FP8 K=3 runtime-R phase-stream p8 calibration requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}

CubKernelOccupancyInfo queryFp8K3RuntimeRPhaseStreamRParallelP4RCooperativeCalibrationOccupancy(DataType input_dtype) {
#if THOR_CUB_ENABLE_FP8_TYPES
    switch (input_dtype) {
        case DataType::FP8_E4M3: return queryFp8K3RuntimeRPhaseStreamOccupancy<__nv_fp8_e4m3, 4>();
        case DataType::FP8_E5M2: return queryFp8K3RuntimeRPhaseStreamOccupancy<__nv_fp8_e5m2, 4>();
        default: throw std::logic_error("FP8 K=3 runtime-R phase-stream p4 occupancy requires FP8 input.");
    }
#else
    static_cast<void>(input_dtype);
    throw std::logic_error("FP8 K=3 runtime-R phase-stream p4 calibration requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}

CubKernelOccupancyInfo queryFp8K3RuntimeRPhaseStreamRParallelP8InputRCooperativeCalibrationOccupancy(
    DataType input_dtype) {
#if THOR_CUB_ENABLE_FP8_TYPES
    switch (input_dtype) {
        case DataType::FP8_E4M3: return queryFp8K3RuntimeRPhaseStreamOccupancy<__nv_fp8_e4m3, 8>();
        case DataType::FP8_E5M2: return queryFp8K3RuntimeRPhaseStreamOccupancy<__nv_fp8_e5m2, 8>();
        default: throw std::logic_error("FP8 K=3 runtime-R phase-stream p8 occupancy requires FP8 input.");
    }
#else
    static_cast<void>(input_dtype);
    throw std::logic_error("FP8 K=3 runtime-R phase-stream p8 calibration requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}


void launchFp8K7PhaseStreamRParallelP8InputRCooperativeCalibrationComplete(const Tensor& input,
                                                                           Tensor& output,
                                                                           const CubReductionGeometry& geometry,
                                                                           Stream& stream) {
#if THOR_CUB_ENABLE_FP8_TYPES
    if (geometry.outer_size == 0 || geometry.inner_size != 7 || output.getDataType() != input.getDataType() ||
        output.getDimensions() != geometry.output_dimensions ||
        (geometry.reduction_size != 8 && geometry.reduction_size != 16 &&
         geometry.reduction_size != 32 && geometry.reduction_size != 64)) {
        throw std::logic_error("FP8 K=7 phase-stream p8-input calibration requires R in {8,16,32,64} and K=7.");
    }
    const uintptr_t pointer_bits = reinterpret_cast<uintptr_t>(input.getMemPtr<void>());
    if ((pointer_bits & uintptr_t{7}) != 0) {
        throw std::logic_error("FP8 K=7 phase-stream p8-input calibration requires a p8-aligned input tensor base.");
    }
    const auto dispatch_r = [&]<typename InputT>() {
        switch (geometry.reduction_size) {
            case 8: launchFp8K7PhaseStreamRParallel<InputT, 8, 8>(input, output, geometry, stream.getStream()); return;
            case 16: launchFp8K7PhaseStreamRParallel<InputT, 16, 8>(input, output, geometry, stream.getStream()); return;
            case 32: launchFp8K7PhaseStreamRParallel<InputT, 32, 8>(input, output, geometry, stream.getStream()); return;
            case 64: launchFp8K7PhaseStreamRParallel<InputT, 64, 8>(input, output, geometry, stream.getStream()); return;
            default: throw std::logic_error("Unsupported R for FP8 K=7 phase-stream p8-input calibration.");
        }
    };
    switch (input.getDataType()) {
        case DataType::FP8_E4M3: dispatch_r.template operator()<__nv_fp8_e4m3>(); return;
        case DataType::FP8_E5M2: dispatch_r.template operator()<__nv_fp8_e5m2>(); return;
        default: throw std::logic_error("FP8 K=7 phase-stream p8-input calibration requires FP8 input.");
    }
#else
    static_cast<void>(input);
    static_cast<void>(output);
    static_cast<void>(geometry);
    static_cast<void>(stream);
    throw std::logic_error("FP8 K=7 phase-stream p8-input calibration requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}

CubKernelOccupancyInfo queryFp8K7PhaseStreamRParallelP8InputRCooperativeCalibrationOccupancy(
    DataType input_dtype, uint64_t reduction_size) {
#if THOR_CUB_ENABLE_FP8_TYPES
    const auto dispatch_r = [&]<typename InputT>() -> CubKernelOccupancyInfo {
        switch (reduction_size) {
            case 8: return queryFp8K7PhaseStreamRParallelOccupancy<InputT, 8, 8>();
            case 16: return queryFp8K7PhaseStreamRParallelOccupancy<InputT, 16, 8>();
            case 32: return queryFp8K7PhaseStreamRParallelOccupancy<InputT, 32, 8>();
            case 64: return queryFp8K7PhaseStreamRParallelOccupancy<InputT, 64, 8>();
            default: throw std::logic_error("Unsupported R for FP8 K=7 phase-stream p8-input occupancy.");
        }
    };
    switch (input_dtype) {
        case DataType::FP8_E4M3: return dispatch_r.template operator()<__nv_fp8_e4m3>();
        case DataType::FP8_E5M2: return dispatch_r.template operator()<__nv_fp8_e5m2>();
        default: throw std::logic_error("FP8 K=7 phase-stream p8-input occupancy requires FP8 input.");
    }
#else
    static_cast<void>(input_dtype);
    static_cast<void>(reduction_size);
    throw std::logic_error("FP8 K=7 phase-stream p8-input calibration requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}

void launchFp8K7PhaseStreamRParallelP16InputRCooperativeCalibrationComplete(const Tensor& input,
                                                                            Tensor& output,
                                                                            const CubReductionGeometry& geometry,
                                                                            Stream& stream) {
#if THOR_CUB_ENABLE_FP8_TYPES
    if (geometry.outer_size == 0 || geometry.inner_size != 7 || output.getDataType() != input.getDataType() ||
        output.getDimensions() != geometry.output_dimensions ||
        (geometry.reduction_size != 16 && geometry.reduction_size != 32 && geometry.reduction_size != 64)) {
        throw std::logic_error("FP8 K=7 phase-stream p16-input calibration requires R in {16,32,64} and K=7.");
    }
    const uintptr_t pointer_bits = reinterpret_cast<uintptr_t>(input.getMemPtr<void>());
    if ((pointer_bits & uintptr_t{15}) != 0) {
        throw std::logic_error("FP8 K=7 phase-stream p16-input calibration requires a p16-aligned input tensor base.");
    }
    const auto dispatch_r = [&]<typename InputT>() {
        switch (geometry.reduction_size) {
            case 16: launchFp8K7PhaseStreamRParallel<InputT, 16, 16>(input, output, geometry, stream.getStream()); return;
            case 32: launchFp8K7PhaseStreamRParallel<InputT, 32, 16>(input, output, geometry, stream.getStream()); return;
            case 64: launchFp8K7PhaseStreamRParallel<InputT, 64, 16>(input, output, geometry, stream.getStream()); return;
            default: throw std::logic_error("Unsupported R for FP8 K=7 phase-stream p16-input calibration.");
        }
    };
    switch (input.getDataType()) {
        case DataType::FP8_E4M3: dispatch_r.template operator()<__nv_fp8_e4m3>(); return;
        case DataType::FP8_E5M2: dispatch_r.template operator()<__nv_fp8_e5m2>(); return;
        default: throw std::logic_error("FP8 K=7 phase-stream p16-input calibration requires FP8 input.");
    }
#else
    static_cast<void>(input);
    static_cast<void>(output);
    static_cast<void>(geometry);
    static_cast<void>(stream);
    throw std::logic_error("FP8 K=7 phase-stream p16-input calibration requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}

CubKernelOccupancyInfo queryFp8K7PhaseStreamRParallelP16InputRCooperativeCalibrationOccupancy(
    DataType input_dtype, uint64_t reduction_size) {
#if THOR_CUB_ENABLE_FP8_TYPES
    const auto dispatch_r = [&]<typename InputT>() -> CubKernelOccupancyInfo {
        switch (reduction_size) {
            case 16: return queryFp8K7PhaseStreamRParallelOccupancy<InputT, 16, 16>();
            case 32: return queryFp8K7PhaseStreamRParallelOccupancy<InputT, 32, 16>();
            case 64: return queryFp8K7PhaseStreamRParallelOccupancy<InputT, 64, 16>();
            default: throw std::logic_error("Unsupported R for FP8 K=7 phase-stream p16-input occupancy.");
        }
    };
    switch (input_dtype) {
        case DataType::FP8_E4M3: return dispatch_r.template operator()<__nv_fp8_e4m3>();
        case DataType::FP8_E5M2: return dispatch_r.template operator()<__nv_fp8_e5m2>();
        default: throw std::logic_error("FP8 K=7 phase-stream p16-input occupancy requires FP8 input.");
    }
#else
    static_cast<void>(input_dtype);
    static_cast<void>(reduction_size);
    throw std::logic_error("FP8 K=7 phase-stream p16-input calibration requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}

void launchFp8DirectComponentWarpP4RCooperativeCalibrationComplete(const Tensor& input,
                                                                   Tensor& output,
                                                                   const CubReductionGeometry& geometry,
                                                                   Stream& stream) {
#if THOR_CUB_ENABLE_FP8_TYPES
    if (geometry.outer_size == 0 || geometry.reduction_size == 0 || geometry.reduction_size > SHALLOW_FP8_MAX_R ||
        !shallowFp8SupportedCalibrationK(geometry.inner_size) || output.getDataType() != input.getDataType() ||
        output.getDimensions() != geometry.output_dimensions) {
        throw std::logic_error("FP8 direct-component p4 calibration received unsupported complete geometry.");
    }
    const auto dispatch_k = [&]<typename InputT>() {
        switch (geometry.inner_size) {
            case 2:
                launchFp8DirectComponentP4<InputT, 2>(input, output, geometry, stream.getStream());
                return;
            case 3:
                launchFp8DirectComponentP4<InputT, 3>(input, output, geometry, stream.getStream());
                return;
            case 4:
                launchFp8DirectComponentP4<InputT, 4>(input, output, geometry, stream.getStream());
                return;
            case 5:
                launchFp8DirectComponentP4<InputT, 5>(input, output, geometry, stream.getStream());
                return;
            case 7:
                launchFp8DirectComponentP4<InputT, 7>(input, output, geometry, stream.getStream());
                return;
            case 8:
                launchFp8DirectComponentP4<InputT, 8>(input, output, geometry, stream.getStream());
                return;
            case 9:
                launchFp8DirectComponentP4<InputT, 9>(input, output, geometry, stream.getStream());
                return;
            case 11:
                launchFp8DirectComponentP4<InputT, 11>(input, output, geometry, stream.getStream());
                return;
            case 13:
                launchFp8DirectComponentP4<InputT, 13>(input, output, geometry, stream.getStream());
                return;
            case 15:
                launchFp8DirectComponentP4<InputT, 15>(input, output, geometry, stream.getStream());
                return;
            case 16:
                launchFp8DirectComponentP4<InputT, 16>(input, output, geometry, stream.getStream());
                return;
            case 17:
                launchFp8DirectComponentP4<InputT, 17>(input, output, geometry, stream.getStream());
                return;
            case 19:
                launchFp8DirectComponentP4<InputT, 19>(input, output, geometry, stream.getStream());
                return;
            case 31:
                launchFp8DirectComponentP4<InputT, 31>(input, output, geometry, stream.getStream());
                return;
            case 32:
                launchFp8DirectComponentP4<InputT, 32>(input, output, geometry, stream.getStream());
                return;
            default:
                throw std::logic_error("FP8 direct-component p4 calibration received unsupported K.");
        }
    };
    switch (input.getDataType()) {
        case DataType::FP8_E4M3:
            dispatch_k.template operator()<__nv_fp8_e4m3>();
            return;
        case DataType::FP8_E5M2:
            dispatch_k.template operator()<__nv_fp8_e5m2>();
            return;
        default:
            throw std::logic_error("FP8 direct-component p4 calibration requires FP8 input.");
    }
#else
    static_cast<void>(input);
    static_cast<void>(output);
    static_cast<void>(geometry);
    static_cast<void>(stream);
    throw std::logic_error("FP8 direct-component p4 calibration requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}

CubKernelOccupancyInfo queryFp8DirectComponentWarpP4RCooperativeCalibrationOccupancy(DataType input_dtype, uint64_t inner_size) {
#if THOR_CUB_ENABLE_FP8_TYPES
    const auto dispatch_k = [&]<typename InputT>() -> CubKernelOccupancyInfo {
        switch (inner_size) {
            case 2:
                return queryFp8DirectComponentP4Occupancy<InputT, 2>();
            case 3:
                return queryFp8DirectComponentP4Occupancy<InputT, 3>();
            case 4:
                return queryFp8DirectComponentP4Occupancy<InputT, 4>();
            case 5:
                return queryFp8DirectComponentP4Occupancy<InputT, 5>();
            case 7:
                return queryFp8DirectComponentP4Occupancy<InputT, 7>();
            case 8:
                return queryFp8DirectComponentP4Occupancy<InputT, 8>();
            case 9:
                return queryFp8DirectComponentP4Occupancy<InputT, 9>();
            case 11:
                return queryFp8DirectComponentP4Occupancy<InputT, 11>();
            case 13:
                return queryFp8DirectComponentP4Occupancy<InputT, 13>();
            case 15:
                return queryFp8DirectComponentP4Occupancy<InputT, 15>();
            case 16:
                return queryFp8DirectComponentP4Occupancy<InputT, 16>();
            case 17:
                return queryFp8DirectComponentP4Occupancy<InputT, 17>();
            case 19:
                return queryFp8DirectComponentP4Occupancy<InputT, 19>();
            case 31:
                return queryFp8DirectComponentP4Occupancy<InputT, 31>();
            case 32:
                return queryFp8DirectComponentP4Occupancy<InputT, 32>();
            default:
                throw std::logic_error("FP8 direct-component p4 occupancy received unsupported K.");
        }
    };
    switch (input_dtype) {
        case DataType::FP8_E4M3:
            return dispatch_k.template operator()<__nv_fp8_e4m3>();
        case DataType::FP8_E5M2:
            return dispatch_k.template operator()<__nv_fp8_e5m2>();
        default:
            throw std::logic_error("FP8 direct-component p4 occupancy requires FP8 input.");
    }
#else
    static_cast<void>(input_dtype);
    static_cast<void>(inner_size);
    throw std::logic_error("FP8 direct-component p4 calibration requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}

void launchFp8P4ReadWriteRooflineCalibrationReference(const Tensor& input,
                                                      Tensor& output,
                                                      const CubReductionGeometry& geometry,
                                                      Stream& stream) {
#if THOR_CUB_ENABLE_FP8_TYPES
    if (geometry.outer_size == 0 || geometry.reduction_size == 0 || geometry.reduction_size > SHALLOW_FP8_MAX_R ||
        output.getDataType() != input.getDataType() || output.getDimensions() != geometry.output_dimensions ||
        (input.getDataType() != DataType::FP8_E4M3 && input.getDataType() != DataType::FP8_E5M2)) {
        throw std::logic_error("FP8 p4 read/write roofline reference received unsupported geometry.");
    }
    const uint64_t output_elements = geometry.outer_size * geometry.inner_size;
    const uint64_t output_packet_count = output_elements / SHALLOW_FP8_PACKET_BYTES;
    const uint32_t tail_output_bytes = static_cast<uint32_t>(output_elements % SHALLOW_FP8_PACKET_BYTES);
    const uint64_t full_warp_groups = output_packet_count / TILED_REDUCTION_WARP_THREADS;
    const uint64_t block_count = std::max<uint64_t>(
        1, std::min<uint64_t>(ceilDivideU64(full_warp_groups, static_cast<uint64_t>(SHALLOW_FP8_WARPS)), NARROW_FLAT_MAX_GRID_BLOCKS));
    const auto* input_bytes = reinterpret_cast<const uint8_t*>(input.getMemPtr<void>());
    auto* output_bytes = reinterpret_cast<uint8_t*>(output.getMemPtr<void>());
    fp8P4ReadWriteRooflineKernel<<<static_cast<unsigned int>(block_count), SHALLOW_FP8_THREADS, 0, stream.getStream()>>>(
        reinterpret_cast<const RawVectorPacket<SHALLOW_FP8_PACKET_BYTES>*>(input_bytes),
        reinterpret_cast<RawVectorPacket<SHALLOW_FP8_PACKET_BYTES>*>(output_bytes),
        output_packet_count,
        static_cast<uint32_t>(geometry.reduction_size),
        input_bytes,
        output_bytes,
        tail_output_bytes);
    CUDA_CHECK(cudaGetLastError());
#else
    static_cast<void>(input);
    static_cast<void>(output);
    static_cast<void>(geometry);
    static_cast<void>(stream);
    throw std::logic_error("FP8 p4 read/write roofline reference requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}

CubKernelOccupancyInfo queryFp8P4ReadWriteRooflineCalibrationOccupancy() {
#if THOR_CUB_ENABLE_FP8_TYPES
    return queryFp8P4ReadWriteRooflineOccupancy();
#else
    throw std::logic_error("FP8 p4 read/write roofline reference requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}

CubKernelOccupancyInfo queryNarrowLowPrecisionFlatRCooperativeFirstStageOccupancy(DataType input_dtype, uint64_t rows_per_shard) {
    switch (input_dtype) {
        case DataType::FP16:
            return queryNarrowFlatOccupancy<__half>(rows_per_shard);
        case DataType::BF16:
            return queryNarrowFlatOccupancy<__nv_bfloat16>(rows_per_shard);
#if THOR_CUB_ENABLE_FP8_TYPES
        case DataType::FP8_E4M3:
            return queryNarrowFp8PackedOccupancy<__nv_fp8_e4m3>(rows_per_shard);
        case DataType::FP8_E5M2:
            return queryNarrowFp8PackedOccupancy<__nv_fp8_e5m2>(rows_per_shard);
#endif
        default:
            throw std::logic_error("Narrow flat R-cooperative occupancy query requires FP8, FP16, or BF16 input.");
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
