#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"
#include "Utilities/TensorOperations/Cub/CubReductionOperation.cuh"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8OneWindowSelector.h"

#include "Utilities/Exceptions.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace ThorImplementation::CubReductionInternal {
namespace {

// FP16/BF16 narrow-flat fixed-row stages and the retained FP8 DeepRSmallK P16 stage
// share this CUDA translation unit. The FP8 retained specialization owns its
// selected row depths and K component layout independently of generic RK policy.
constexpr int NARROW_FLAT_WARPS = 8;
constexpr int NARROW_FLAT_THREADS = NARROW_FLAT_WARPS * TILED_REDUCTION_WARP_THREADS;
constexpr int NARROW_FLAT_MAX_K = 32;
constexpr uint64_t NARROW_FLAT_MAX_GRID_BLOCKS = 65535;
static_assert(NARROW_FLAT_THREADS == 256);

template <typename InputT>
inline constexpr bool narrowFlatInputType =
    std::is_same_v<InputT, __half> || std::is_same_v<InputT, __nv_bfloat16>;

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

            // FP16/BF16 group is 8*K*2 bytes. For K<=31, even the
            // largest alignment prefix fits in one warp of 16-byte packets;
            // K=32 starts naturally 16-byte aligned. Short tails fit too.
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

// Retained shallow-R FP8 Complete families (OneWindow and RuntimeK packet-complete) launch eight-warp CTAs.
// Their compiled physical inventory is owned by the Fp8RK selector policy headers; the dispatchers below
// instantiate only those retained variants.
constexpr int SHALLOW_FP8_WARPS = 8;
constexpr int SHALLOW_FP8_THREADS = SHALLOW_FP8_WARPS * TILED_REDUCTION_WARP_THREADS;
constexpr int RPARALLEL_FP8_MAX_K = 512;
static_assert(SHALLOW_FP8_THREADS == 256);

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
        case 497:
        case 512:
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


// Match the additive finalization semantics of the production dense reducer. Input
// transformation now belongs to each decoded value; finalization belongs only to
// the complete FP32 aggregate before runtime output conversion.
[[nodiscard]] AdditiveFinalizeFp32 fp8CompleteFinalizerFromSemantics(
    const CubReductionStageSemantics& semantics) {
    if (semantics.combine != CubReductionStageCombine::Sum &&
        semantics.finalize != CubReductionStageFinalize::Identity)
        throw std::logic_error("Nonadditive FP8 Complete cannot have an additive finalizer.");
    switch (semantics.input_transform) {
        case CubReductionStageInputTransform::Identity:
        case CubReductionStageInputTransform::AbsoluteValue:
        case CubReductionStageInputTransform::Square:
            break;
        default:
            throw std::logic_error("FP8 Complete received an unsupported input transform.");
    }
    switch (semantics.finalize) {
        case CubReductionStageFinalize::Identity:
            return {1.0f, false};
        case CubReductionStageFinalize::Divide:
            if (semantics.finalize_divisor == 0) {
                throw std::logic_error("FP8 Complete DIVIDE requires a nonzero original reduction size.");
            }
            return {static_cast<float>(semantics.finalize_divisor), false};
        case CubReductionStageFinalize::SquareRoot:
            return {1.0f, true};
        default:
            throw std::logic_error("FP8 Complete received an unsupported finalization mode.");
    }
}

// Transform a decoded packet under one launch-uniform branch, before the
// per-component tail guards. This avoids a separate transform-selection branch
// inside each valid-value guard. OneWindow passes FixedInputTransform=-1 (runtime
// transform operand); RuntimeK packet-complete compiles the three fixed transforms.
template <int FixedInputTransform = -1>
__device__ __forceinline__ float4 transformFp8CompleteDecodedPacket(
    float4 values, CubReductionStageInputTransform transform) {
    if constexpr (FixedInputTransform >= 0) {
        static_cast<void>(transform);
    }
    if constexpr (FixedInputTransform == 0) {
        return values;
    } else if constexpr (FixedInputTransform == 1) {
        values.x = ::fabsf(values.x);
        values.y = ::fabsf(values.y);
        values.z = ::fabsf(values.z);
        values.w = ::fabsf(values.w);
    } else if constexpr (FixedInputTransform == 2) {
        values.x *= values.x;
        values.y *= values.y;
        values.z *= values.z;
        values.w *= values.w;
    } else {
        if (transform == CubReductionStageInputTransform::AbsoluteValue) {
            values.x = ::fabsf(values.x);
            values.y = ::fabsf(values.y);
            values.z = ::fabsf(values.z);
            values.w = ::fabsf(values.w);
        } else if (transform == CubReductionStageInputTransform::Square) {
            values.x *= values.x;
            values.y *= values.y;
            values.z *= values.z;
            values.w *= values.w;
        }
    }
    return values;
}

// The runtime-dtype store is deliberately shared by retained FP8 Complete
// kernels. Reject unsupported destinations on the host: the device store's
// default arm does not write anything.
[[nodiscard]] bool fp8CompleteSupportsOutputStorage(DataType dtype) {
    switch (dtype) {
#if THOR_CUB_ENABLE_FP8_TYPES
        case DataType::FP8_E4M3:
        case DataType::FP8_E5M2:
#endif
        case DataType::FP16:
        case DataType::BF16:
        case DataType::FP32:
#if THOR_CUB_ENABLE_64BIT_TYPES
        case DataType::FP64:
#endif
            return true;
        default:
            return false;
    }
}

// The Complete families share a uniform runtime-dtype store. Finalization and
// scaling happen once, on the final mathematical reduction pass only.
__device__ __forceinline__ void storeFp8CompleteResult(void* output,
                                                        DataType output_dtype,
                                                        uint64_t index,
                                                        float value,
                                                        AdditiveFinalizeFp32 finalizer,
                                                        float output_scale) {
    storeFp32AsRuntimeDType(output, output_dtype, index, finalizer(value) * output_scale);
}

// Apply the selected compile-time FP32 combine over one decoded packet.
// Output ownership, accumulation order and valid component masking are unchanged.
template <int FixedOp>
__device__ __forceinline__ float reduceOneWindowDecoded4(float acc, float4 values,
                                                         uint32_t valid) {
    const Fp8TypedReductionOp<FixedOp> combine{};
    if (valid > 0) acc = combine(acc, values.x);
    if (valid > 1) acc = combine(acc, values.y);
    if (valid > 2) acc = combine(acc, values.z);
    if (valid > 3) acc = combine(acc, values.w);
    return acc;
}

// General Complete RParallel spans both output and R ownership with one packed FP8 physical family. O1 may use
// a 1/2/4/8-warp subgroup per output. O2/O4/O8 instead partition one warp into independent output subgroups and
// therefore require W1. Within each output subgroup, K<=lanes-per-output spends spare lanes as independent R
// workers/component; larger K assigns multiple components/lane. Every subgroup gets its own p4/p8/p16 packet slice
// in the warp-wide plane-major shared staging buffer and streams runtime R through that slice. This subsumes the
// ownership mechanics of the earlier MultiOutput path while extending them through the current one-window K range; output multiplicity,
// warps/output, and packet width are axes of one family rather than distinct physical kernel families.
template <typename InputT, int K, int PacketBytes, int OutputsPerWarp,
          int FixedInputTransform = -1, int FixedOp = 0>
__global__ void fp8GeneralRParallelPackedCompleteKernel(const InputT* input,
                                                        void* output,
                                                        DataType output_dtype,
                                                        uint32_t outer_size,
                                                        uint32_t reduction_size,
                                                        uint32_t warps_per_output,
                                                        AdditiveFinalizeFp32 output_finalize,
                                                        CubReductionStageInputTransform input_transform,
                                                        float output_scale) {
    const Fp8TypedReductionOp<FixedOp> combine{};
    static_assert(sizeof(InputT) == 1);
    static_assert(K >= 1 && K <= RPARALLEL_FP8_MAX_K);
    static_assert(PacketBytes == 4 || PacketBytes == 8 || PacketBytes == 16);
    static_assert(OutputsPerWarp == 1 || OutputsPerWarp == 2 || OutputsPerWarp == 4 || OutputsPerWarp == 8);
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
                local[slot] = combine.identity();
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
                            const float4 decoded = transformFp8CompleteDecodedPacket<FixedInputTransform>(
                                decodeFp8x4ToFp32<InputT>(packed_component_values), input_transform);
                            local[slot] = reduceOneWindowDecoded4<FixedOp>(
                                local[slot], decoded, valid_values);
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
                        storeFp8CompleteResult(output, output_dtype, outer_index * static_cast<uint64_t>(K) + owned_component, local[slot], output_finalize, output_scale);
                    }
                }
            } else {
                lane_partials[warp][lane] =
                    (valid_output && subgroup_lane < TaskCountPerOutput) ? local[0] : combine.identity();
                __syncwarp();
                if (valid_output && subgroup_lane < K) {
                    float reduced = combine.identity();
                    const uint32_t partial_base =
                        output_slot * LanesPerOutput + subgroup_lane * WorkersPerComponent;
#pragma unroll
                    for (uint32_t source_worker = 0; source_worker < WorkersPerComponent; ++source_worker) {
                        reduced = combine(reduced, lane_partials[warp][partial_base + source_worker]);
                    }
                    storeFp8CompleteResult(output, output_dtype, outer_index * static_cast<uint64_t>(K) + subgroup_lane, reduced, output_finalize, output_scale);
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
            local[slot] = combine.identity();
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
                                const uint32_t source_packet = staged_byte >> PacketShift;
                                const uint32_t packet_byte = staged_byte & PacketMask;
                                const uint32_t source_word =
                                    staged_packet_words[warp]
                                                       [(packet_byte >> 2) * TILED_REDUCTION_WARP_THREADS +
                                                        source_packet];
                                const uint32_t source_value = (source_word >> ((packet_byte & 3U) * 8U)) & 0xffU;
                                packed_component_values |= source_value << (item * 8U);
                                ++valid_values;
                            }
                        }
                        const float4 decoded = transformFp8CompleteDecodedPacket<FixedInputTransform>(
                            decodeFp8x4ToFp32<InputT>(packed_component_values), input_transform);
                        local[slot] = reduceOneWindowDecoded4<FixedOp>(
                            local[slot], decoded, valid_values);
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
                        storeFp8CompleteResult(output, output_dtype, outer_index * static_cast<uint64_t>(K) + owned_component, local[slot], output_finalize, output_scale);
                    }
                }
            } else {
                lane_partials[warp][lane] = (valid_output && lane < TaskCountPerOutput) ? local[0] : combine.identity();
                __syncwarp();
                if (valid_output && lane < K) {
                    float reduced = combine.identity();
                    const uint32_t partial_base = lane * WorkersPerComponent;
#pragma unroll
                    for (uint32_t source_worker = 0; source_worker < WorkersPerComponent; ++source_worker) {
                        reduced = combine(reduced, lane_partials[warp][partial_base + source_worker]);
                    }
                    storeFp8CompleteResult(output, output_dtype, outer_index * static_cast<uint64_t>(K) + lane, reduced, output_finalize, output_scale);
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
                    warp_partials[warp][owned_component] = valid_output ? local[slot] : combine.identity();
                }
            }
        } else {
            lane_partials[warp][lane] = (valid_output && lane < TaskCountPerOutput) ? local[0] : combine.identity();
            __syncwarp();
            if (lane < K) {
                float warp_reduced = combine.identity();
                const uint32_t partial_base = lane * WorkersPerComponent;
#pragma unroll
                for (uint32_t source_worker = 0; source_worker < WorkersPerComponent; ++source_worker) {
                    warp_reduced = combine(warp_reduced, lane_partials[warp][partial_base + source_worker]);
                }
                warp_partials[warp][lane] = valid_output ? warp_reduced : combine.identity();
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
                    float reduced = combine.identity();
                    for (uint32_t source_warp = 0; source_warp < warps_per_output; ++source_warp) {
                        reduced = combine(reduced, warp_partials[first_warp_in_output + source_warp][owned_component]);
                    }
                    storeFp8CompleteResult(output, output_dtype, outer_index * static_cast<uint64_t>(K) + owned_component, reduced, output_finalize, output_scale);
                }
            }
        }
        __syncthreads();
    }
    }
}


// Runtime-K tiled Complete topology for generalized RParallel. This is the large-K continuation of the O1 family:
// one work item owns [outer, k_tile], K tiles are mathematically independent, and W1/W2/W4/W8 cooperate only over R.
// Total K and tile width are runtime values; only the p16 physical packet shape is specialized. A tile is capped at
// 496 logical bytes so an arbitrary per-row prefix (0..15 bytes) still fits in one 32-lane p16 packet window.
template <typename InputT, int FixedInputTransform = 0, int FixedOp = 0>
__global__ void fp8GeneralRParallelKTileP16CompleteKernel(const InputT* input,
                                                          void* output,
                                                          DataType output_dtype,
                                                          uint32_t outer_size,
                                                          uint32_t reduction_size,
                                                          uint32_t inner_size,
                                                          uint32_t tile_components,
                                                          uint32_t tile_count,
                                                          uint32_t warps_per_output,
                                                          AdditiveFinalizeFp32 output_finalize,
                                                          CubReductionStageInputTransform input_transform,
                                                          float output_scale) {
    static_assert(sizeof(InputT) == 1);
    static_assert(FixedInputTransform >= 0 && FixedInputTransform <= 2,
                  "RuntimeK Complete requires a compile-time FP8 input transform.");
    constexpr uint32_t PacketBytes = 16;
    constexpr uint32_t PacketShift = 4;
    constexpr uint32_t PacketMask = 15;
    constexpr uint32_t PacketWords = 4;
    constexpr uint32_t MaxTileComponents = 496;
    constexpr uint32_t MaxComponentsPerLane =
        (MaxTileComponents + TILED_REDUCTION_WARP_THREADS - 1) / TILED_REDUCTION_WARP_THREADS;
    static_assert(MaxComponentsPerLane == 16);

    const Fp8TypedReductionOp<FixedOp> combine{};
    const float initial = combine.identity();
    __shared__ uint32_t staged_packet_words[SHALLOW_FP8_WARPS]
                                            [PacketWords * TILED_REDUCTION_WARP_THREADS];
    __shared__ float warp_partials[SHALLOW_FP8_WARPS][MaxTileComponents];

    const uint32_t warp = threadIdx.x / TILED_REDUCTION_WARP_THREADS;
    const uint32_t lane = threadIdx.x % TILED_REDUCTION_WARP_THREADS;
    const uint32_t outputs_per_block = SHALLOW_FP8_WARPS / warps_per_output;
    const uint32_t output_group = warp / warps_per_output;
    const uint32_t warp_in_output = warp - output_group * warps_per_output;
    const uint32_t first_warp_in_output = output_group * warps_per_output;
    const uint64_t work_count = static_cast<uint64_t>(outer_size) * tile_count;
    const uint64_t block_work_stride = static_cast<uint64_t>(gridDim.x) * outputs_per_block;
    const uint64_t outer_input_stride = static_cast<uint64_t>(reduction_size) * inner_size;

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

        for (uint32_t row = warp_in_output; valid_work && row < reduction_size; row += warps_per_output) {
            const uint64_t logical_begin =
                static_cast<uint64_t>(outer_index) * outer_input_stride +
                static_cast<uint64_t>(row) * inner_size + tile_begin;
            const uint32_t prefix_bytes = static_cast<uint32_t>(logical_begin & PacketMask);
            const uint64_t aligned_begin = logical_begin - prefix_bytes;
            const uint32_t packet_count =
                (prefix_bytes + tile_width + PacketBytes - 1) >> PacketShift;

            if (lane < packet_count) {
                const uint64_t packet_byte_begin = aligned_begin + static_cast<uint64_t>(lane) * PacketBytes;
                // A p16 load extends at most 15 bytes past the logical tensor end.
                // Tensor/view allocation provides >=128 trailing bytes; component masks
                // below still prevent out-of-range values from entering the reduction.
                const RawVectorPacket<PacketBytes> packet =
                    *reinterpret_cast<const RawVectorPacket<PacketBytes>*>(
                        reinterpret_cast<const uint8_t*>(input) + packet_byte_begin);
                staged_packet_words[warp][0 * TILED_REDUCTION_WARP_THREADS + lane] = packet.value.x;
                staged_packet_words[warp][1 * TILED_REDUCTION_WARP_THREADS + lane] = packet.value.y;
                staged_packet_words[warp][2 * TILED_REDUCTION_WARP_THREADS + lane] = packet.value.z;
                staged_packet_words[warp][3 * TILED_REDUCTION_WARP_THREADS + lane] = packet.value.w;
            }
            __syncwarp();

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
            __syncwarp();
        }

        if (warps_per_output == 1) {
#pragma unroll
            for (uint32_t slot = 0; slot < MaxComponentsPerLane; ++slot) {
                const uint32_t component = lane + slot * TILED_REDUCTION_WARP_THREADS;
                if (valid_work && component < tile_width) {
                    storeFp8CompleteResult(output, output_dtype, static_cast<uint64_t>(outer_index) * inner_size + tile_begin + component, local[slot], output_finalize, output_scale);
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
                    storeFp8CompleteResult(output, output_dtype, static_cast<uint64_t>(outer_index) * inner_size + tile_begin + component, reduced, output_finalize, output_scale);
                }
            }
        }
        __syncthreads();
    }
}



template <typename InputT, int K, int PacketBytes, int OutputsPerWarp,
          int FixedInputTransform = -1, int FixedOp = 0>
void launchFp8GeneralRParallelPacked(const Tensor& input,
                                     Tensor& output,
                                     const CubReductionGeometry& geometry,
                                     uint32_t warps_per_output,
                                     cudaStream_t stream,
                                     AdditiveFinalizeFp32 output_finalize = {1.0f, false},
                                     CubReductionStageInputTransform input_transform = CubReductionStageInputTransform::Identity,
                                     float output_scale = 1.0f) {
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
    fp8GeneralRParallelPackedCompleteKernel<InputT, K, PacketBytes, OutputsPerWarp, FixedInputTransform, FixedOp>
        <<<static_cast<unsigned int>(block_count), SHALLOW_FP8_THREADS, 0, stream>>>(
            input.getMemPtr<InputT>(),
            output.getMemPtr<void>(),
            output.getDataType(),
            static_cast<uint32_t>(geometry.outer_size),
            static_cast<uint32_t>(geometry.reduction_size),
            warps_per_output,
            output_finalize,
            input_transform,
            output_scale);
    CUDA_CHECK(cudaGetLastError());
}


template <typename InputT, int FixedInputTransform = 0, int FixedOp = 0>
void launchFp8GeneralRParallelKTileP16Complete(const Tensor& input,
                                               Tensor& output,
                                               const CubReductionGeometry& geometry,
                                               uint32_t warps_per_output,
                                               uint32_t tile_components,
                                               cudaStream_t stream,
                                               AdditiveFinalizeFp32 output_finalize = {1.0f, false},
                                               CubReductionStageInputTransform input_transform = CubReductionStageInputTransform::Identity,
                                               float output_scale = 1.0f) {
    static_assert(FixedInputTransform >= 0 && FixedInputTransform <= 2);
    static_assert(sizeof(InputT) == 1);
    if (tile_components == 0 || tile_components > 496 ||
        (warps_per_output != 1 && warps_per_output != 2 && warps_per_output != 4 && warps_per_output != 8)) {
        throw std::logic_error("FP8 tiled general RParallel Complete received illegal tile/W geometry.");
    }
    if ((reinterpret_cast<uintptr_t>(input.getMemPtr<InputT>()) & uintptr_t{15}) != 0) {
        throw std::logic_error("FP8 tiled general RParallel Complete requires p16-aligned input base.");
    }
    const uint32_t tile_count = static_cast<uint32_t>(
        ceilDivideU64(geometry.inner_size, static_cast<uint64_t>(tile_components)));
    const uint32_t outputs_per_block = SHALLOW_FP8_WARPS / warps_per_output;
    const uint64_t work_count = geometry.outer_size * static_cast<uint64_t>(tile_count);
    const uint64_t block_count = std::min<uint64_t>(
        ceilDivideU64(work_count, static_cast<uint64_t>(outputs_per_block)),
        NARROW_FLAT_MAX_GRID_BLOCKS);
    fp8GeneralRParallelKTileP16CompleteKernel<InputT, FixedInputTransform, FixedOp>
        <<<static_cast<unsigned int>(block_count), SHALLOW_FP8_THREADS, 0, stream>>>(
            input.getMemPtr<InputT>(),
            output.getMemPtr<void>(),
            output.getDataType(),
            static_cast<uint32_t>(geometry.outer_size),
            static_cast<uint32_t>(geometry.reduction_size),
            static_cast<uint32_t>(geometry.inner_size),
            tile_components,
            tile_count,
            warps_per_output,
            output_finalize,
            input_transform,
            output_scale);
    CUDA_CHECK(cudaGetLastError());
}


template <typename InputT, int K, int PacketBytes, int OutputsPerWarp, int FixedOp = 0>
void launchFp8GeneralRParallelPackedForOwnership(const Tensor& input,
                                                  Tensor& output,
                                                  const CubReductionGeometry& geometry,
                                                  uint32_t warps_per_output,
                                                  cudaStream_t stream,
                                                  AdditiveFinalizeFp32 output_finalize = {1.0f, false},
                                                  CubReductionStageInputTransform input_transform = CubReductionStageInputTransform::Identity,
                                                  float output_scale = 1.0f) {
    static_assert(PacketBytes == 4 || PacketBytes == 8 || PacketBytes == 16);
    launchFp8GeneralRParallelPacked<InputT, K, PacketBytes, OutputsPerWarp, -1, FixedOp>(
        input, output, geometry, warps_per_output, stream, output_finalize, input_transform, output_scale);
}


template <typename KernelT>
[[nodiscard]] CubKernelOccupancyInfo queryShallowFp8Occupancy(KernelT kernel) {
    cudaFuncAttributes attributes{};
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
    int active_blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks, kernel, SHALLOW_FP8_THREADS, 0));
    return CubKernelOccupancyInfo{attributes.numRegs, attributes.sharedSizeBytes, 0, active_blocks, active_blocks * SHALLOW_FP8_WARPS};
}


template <typename InputT, int K, int PacketBytes, int OutputsPerWarp, int FixedOp = 0>
[[nodiscard]] CubKernelOccupancyInfo queryFp8GeneralRParallelPackedOccupancy() {
    static_assert(PacketBytes == 4 || PacketBytes == 8 || PacketBytes == 16);
    static_assert(OutputsPerWarp == 1 || OutputsPerWarp == 2 || OutputsPerWarp == 4 || OutputsPerWarp == 8);
    constexpr uint32_t LanesPerOutput = TILED_REDUCTION_WARP_THREADS / OutputsPerWarp;
    constexpr uint32_t PacketWindow = LanesPerOutput * PacketBytes;
    constexpr uint32_t LogicalCapacity =
        (K % PacketBytes) == 0 ? PacketWindow : PacketWindow - (PacketBytes - 1);
    static_assert(K <= static_cast<int>(LogicalCapacity));
    return queryShallowFp8Occupancy(
        fp8GeneralRParallelPackedCompleteKernel<InputT, K, PacketBytes, OutputsPerWarp, -1, FixedOp>);
}

template <typename InputT, int K, int PacketBytes, int OutputsPerWarp, int FixedOp = 0>
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
    return queryFp8GeneralRParallelPackedOccupancy<InputT, K, PacketBytes, OutputsPerWarp, FixedOp>();
}


template <typename InputT>
struct NarrowFp8TypedFormatDecoder {
    __device__ float operator()(uint8_t bits) const {
        return ToFp32<InputT>{}(narrowFp8FromBits<InputT>(bits));
    }
};

template <typename InputT, typename ReductionOpT, typename InputTransformT, int RowsPerShard, int PacketBytes, int MaxComponentsPerLane,
          typename DecoderT = NarrowFp8TypedFormatDecoder<InputT>>
__global__ void narrowFp8PackedFlatRCooperativeFirstStageKernel(const InputT* input,
                                                                float* fp32_partials,
                                                                uint64_t outer_size,
                                                                uint64_t reduction_size,
                                                                uint64_t inner_size,
                                                                uint64_t shards_per_output,
                                                                InputTransformT input_transform, DecoderT decoder = DecoderT{}) {
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
    const int warp = static_cast<int>(threadIdx.x) / TILED_REDUCTION_WARP_THREADS;
    const int lane = static_cast<int>(threadIdx.x) % TILED_REDUCTION_WARP_THREADS;
    const uint64_t total_work = outer_size * shards_per_output;
    const uint64_t grid_stride = static_cast<uint64_t>(gridDim.x);

    // DeepRSmallK's retained launcher guarantees 1 <= inner_size <= 256.
    // One warp stages at most 32 * PacketBytes <= 512 bytes, including an
    // alignment prefix < PacketBytes. Keep global tensor addressing in 64 bits,
    // but use 32-bit arithmetic for every offset within this packet window.
    const uint32_t inner_size_32 = static_cast<uint32_t>(inner_size);

    // Fill toward the packet window, but never choose a group so deep that fewer than eight warps can participate in a
    // full RowsPerShard shard. RowsPerShard is always divisible by eight in the calibration ladder.
    const uint32_t rows_from_packet_window = static_cast<uint32_t>(narrowFp8MaxLogicalGroupBytes<PacketBytes>) / inner_size_32;
    const uint32_t rows_per_warp_group =
        rows_from_packet_window < static_cast<uint32_t>(MaxRowsPerWarpFromShard)
            ? rows_from_packet_window : static_cast<uint32_t>(MaxRowsPerWarpFromShard);
    const uint64_t rows_per_block_round = static_cast<uint64_t>(rows_per_warp_group) * NARROW_FLAT_WARPS;

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
            const uint32_t group_rows = static_cast<uint32_t>(minU64(rows_per_warp_group, row_end - group_row));
            const InputT* group_begin = input + (outer_index * reduction_size + group_row) * inner_size;
            const uintptr_t group_address = reinterpret_cast<uintptr_t>(group_begin);
            const uint32_t prefix_elements = static_cast<uint32_t>(group_address & uintptr_t{PacketBytes - 1});
            const auto* aligned_begin = reinterpret_cast<const RawVectorPacket<PacketBytes>*>(group_address & ~uintptr_t{PacketBytes - 1});
            const uint32_t logical_group_elements = group_rows * inner_size_32;
            const uint32_t packet_count = (prefix_elements + logical_group_elements + PacketBytes - 1) / PacketBytes;

            if (static_cast<uint32_t>(lane) < packet_count) {
                staged_packets[warp][lane] = aligned_begin[lane];
            }
            __syncwarp();

            const auto* staged_bytes = reinterpret_cast<const uint8_t*>(&staged_packets[warp][0]);
            for (int slot = 0; slot < MaxComponentsPerLane; ++slot) {
                const uint32_t component = static_cast<uint32_t>(lane + slot * TILED_REDUCTION_WARP_THREADS);
                if (component < inner_size_32) {
                    uint32_t staged_index = prefix_elements + component;
                    for (uint32_t row = 0; row < group_rows; ++row, staged_index += inner_size_32) {
                        const float value = decoder(staged_bytes[staged_index]);
                        local[slot] = reduction_op(local[slot], input_transform(value));
                    }
                }
            }
            __syncwarp();
        }

        for (int slot = 0; slot < MaxComponentsPerLane; ++slot) {
            const uint32_t component = static_cast<uint32_t>(lane + slot * TILED_REDUCTION_WARP_THREADS);
            if (component < inner_size_32) {
                warp_partials[warp][component] = local[slot];
            }
        }
        __syncthreads();

        if (static_cast<uint32_t>(threadIdx.x) < inner_size_32) {
            const uint32_t component = static_cast<uint32_t>(threadIdx.x);
            float reduced = Init;
            for (int source_warp = 0; source_warp < NARROW_FLAT_WARPS; ++source_warp) {
                reduced = reduction_op(reduced, warp_partials[source_warp][component]);
            }
            fp32_partials[(outer_index * shards_per_output + shard) * inner_size + component] = reduced;
        }
        __syncthreads();
    }
}

template <typename InputT, typename ReductionOpT, typename InputTransformT, int RowsPerShard, int PacketBytes, int LanesPerComponent,
          typename DecoderT = NarrowFp8TypedFormatDecoder<InputT>>
__global__ void narrowFp8PackedFlatRCooperativeTinyKFirstStageKernel(const InputT* input,
                                                                     float* fp32_partials,
                                                                     uint64_t outer_size,
                                                                     uint64_t reduction_size,
                                                                     uint64_t inner_size,
                                                                     uint64_t shards_per_output,
                                                                InputTransformT input_transform, DecoderT decoder = DecoderT{}) {
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
    const int warp = static_cast<int>(threadIdx.x) / TILED_REDUCTION_WARP_THREADS;
    const int lane = static_cast<int>(threadIdx.x) % TILED_REDUCTION_WARP_THREADS;
    const int component = lane / LanesPerComponent;
    const int component_lane = lane % LanesPerComponent;
    const uint64_t total_work = outer_size * shards_per_output;
    const uint64_t grid_stride = static_cast<uint64_t>(gridDim.x);

    // DeepRSmallK's retained launcher guarantees 1 <= inner_size <= 256.
    // One warp stages at most 32 * PacketBytes <= 512 bytes, including an
    // alignment prefix < PacketBytes. Keep global tensor addressing in 64 bits,
    // but use 32-bit arithmetic for every offset within this packet window.
    const uint32_t inner_size_32 = static_cast<uint32_t>(inner_size);

    const uint32_t rows_from_packet_window = static_cast<uint32_t>(narrowFp8MaxLogicalGroupBytes<PacketBytes>) / inner_size_32;
    const uint32_t rows_per_warp_group =
        rows_from_packet_window < static_cast<uint32_t>(MaxRowsPerWarpFromShard)
            ? rows_from_packet_window : static_cast<uint32_t>(MaxRowsPerWarpFromShard);
    const uint64_t rows_per_block_round = static_cast<uint64_t>(rows_per_warp_group) * NARROW_FLAT_WARPS;

    for (uint64_t work_index = static_cast<uint64_t>(blockIdx.x); work_index < total_work; work_index += grid_stride) {
        const uint64_t shard = work_index % shards_per_output;
        const uint64_t outer_index = work_index / shards_per_output;
        const uint64_t row_begin = shard * static_cast<uint64_t>(RowsPerShard);
        const uint64_t row_end = minU64(row_begin + static_cast<uint64_t>(RowsPerShard), reduction_size);

        float local = Init;
        const uint64_t first_group_row = row_begin + static_cast<uint64_t>(warp) * rows_per_warp_group;
        for (uint64_t group_row = first_group_row; group_row < row_end; group_row += rows_per_block_round) {
            const uint32_t group_rows = static_cast<uint32_t>(minU64(rows_per_warp_group, row_end - group_row));
            const InputT* group_begin = input + (outer_index * reduction_size + group_row) * inner_size;
            const uintptr_t group_address = reinterpret_cast<uintptr_t>(group_begin);
            const uint32_t prefix_elements = static_cast<uint32_t>(group_address & uintptr_t{PacketBytes - 1});
            const auto* aligned_begin = reinterpret_cast<const RawVectorPacket<PacketBytes>*>(group_address & ~uintptr_t{PacketBytes - 1});
            const uint32_t logical_group_elements = group_rows * inner_size_32;
            const uint32_t packet_count = (prefix_elements + logical_group_elements + PacketBytes - 1) / PacketBytes;

            if (static_cast<uint32_t>(lane) < packet_count) {
                staged_packets[warp][lane] = aligned_begin[lane];
            }
            __syncwarp();

            if (static_cast<uint32_t>(component) < inner_size_32) {
                const auto* staged_bytes = reinterpret_cast<const uint8_t*>(&staged_packets[warp][0]);
                uint32_t staged_index = prefix_elements + static_cast<uint32_t>(component_lane) * inner_size_32 +
                                        static_cast<uint32_t>(component);
                const uint32_t staged_step = LanesPerComponent * inner_size_32;
                for (uint32_t row = static_cast<uint32_t>(component_lane); row < group_rows;
                     row += LanesPerComponent, staged_index += staged_step) {
                    const float value = decoder(staged_bytes[staged_index]);
                    local = reduction_op(local, input_transform(value));
                }
            }
            __syncwarp();
        }

        cooperative_lane_partials[warp][lane] = local;
        __syncwarp();
        if (component_lane == 0 && static_cast<uint32_t>(component) < inner_size_32) {
            float reduced = Init;
            for (int source_lane = 0; source_lane < LanesPerComponent; ++source_lane) {
                reduced = reduction_op(reduced, cooperative_lane_partials[warp][component * LanesPerComponent + source_lane]);
            }
            warp_partials[warp][component] = reduced;
        }
        __syncthreads();

        if (static_cast<uint32_t>(threadIdx.x) < inner_size_32) {
            const uint32_t output_component = static_cast<uint32_t>(threadIdx.x);
            float reduced = Init;
            for (int source_warp = 0; source_warp < NARROW_FLAT_WARPS; ++source_warp) {
                reduced = reduction_op(reduced, warp_partials[source_warp][output_component]);
            }
            fp32_partials[(outer_index * shards_per_output + shard) * inner_size + output_component] = reduced;
        }
        __syncthreads();
    }
}

template <typename InputT, typename ReductionOpT, typename InputTransformT, int RowsPerShard, int PacketBytes, int LanesPerComponent,
          typename DecoderT = NarrowFp8TypedFormatDecoder<InputT>>
void launchNarrowFp8PackedTinyKForTransform(
    const Tensor& input, Tensor& fp32_partials, const CubReductionGeometry& geometry, uint64_t shards_per_output, cudaStream_t stream,
    InputTransformT input_transform = InputTransformT{}, DecoderT decoder = DecoderT{}) {
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
                                                         LanesPerComponent,
                                                         DecoderT>
        <<<grid_blocks, NARROW_FLAT_THREADS, 0, stream>>>(([](const Tensor& src) -> const InputT* {
            if constexpr (std::is_same_v<InputT, uint8_t>) return reinterpret_cast<const InputT*>(src.getMemPtr<void>());
            else return src.getMemPtr<InputT>();
        })(input),
                                                          fp32_partials.getMemPtr<float>(),
                                                          geometry.outer_size,
                                                          geometry.reduction_size,
                                                          geometry.inner_size,
                                                          shards_per_output,
                                                          input_transform, decoder);
    CUDA_CHECK(cudaGetLastError());
}

template <typename InputT, typename ReductionOpT, typename InputTransformT, int RowsPerShard, int PacketBytes, int MaxComponentsPerLane,
          typename DecoderT = NarrowFp8TypedFormatDecoder<InputT>>
void launchNarrowFp8PackedForTransform(
    const Tensor& input, Tensor& fp32_partials, const CubReductionGeometry& geometry, uint64_t shards_per_output, cudaStream_t stream,
    InputTransformT input_transform = InputTransformT{}, DecoderT decoder = DecoderT{}) {
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

    narrowFp8PackedFlatRCooperativeFirstStageKernel<InputT, ReductionOpT, InputTransformT, RowsPerShard, PacketBytes, MaxComponentsPerLane, DecoderT>
        <<<grid_blocks, NARROW_FLAT_THREADS, 0, stream>>>(([](const Tensor& src) -> const InputT* {
            if constexpr (std::is_same_v<InputT, uint8_t>) return reinterpret_cast<const InputT*>(src.getMemPtr<void>());
            else return src.getMemPtr<InputT>();
        })(input),
                                                          fp32_partials.getMemPtr<float>(),
                                                          geometry.outer_size,
                                                          geometry.reduction_size,
                                                          geometry.inner_size,
                                                          shards_per_output,
                                                          input_transform, decoder);
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

template <typename InputT, int RowsPerShard, int PacketBytes, int LanesPerComponent, typename InputTransformT = IdentityFp32,
          typename DecoderT = NarrowFp8TypedFormatDecoder<InputT>, typename ReductionOpT = cuda::std::plus<float>>
[[nodiscard]] CubKernelOccupancyInfo queryNarrowFp8PackedTinyKOccupancyForRows() {
    static_assert(sizeof(InputT) == 1);
    static_assert(PacketBytes == 4 || PacketBytes == 8 || PacketBytes == 16);
    static_assert(LanesPerComponent == 4 || LanesPerComponent == 8);
    const auto kernel = narrowFp8PackedFlatRCooperativeTinyKFirstStageKernel<InputT,
                                                                             ReductionOpT,
                                                                             InputTransformT,
                                                                             RowsPerShard,
                                                                             PacketBytes,
                                                                             LanesPerComponent,
                                                                         DecoderT>;
    cudaFuncAttributes attributes{};
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
    int active_blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks, kernel, NARROW_FLAT_THREADS, 0));
    return CubKernelOccupancyInfo{attributes.numRegs, attributes.sharedSizeBytes, 0, active_blocks, active_blocks * NARROW_FLAT_WARPS};
}

template <typename InputT, int RowsPerShard, int PacketBytes, int MaxComponentsPerLane, typename InputTransformT = IdentityFp32,
          typename DecoderT = NarrowFp8TypedFormatDecoder<InputT>, typename ReductionOpT = cuda::std::plus<float>>
[[nodiscard]] CubKernelOccupancyInfo queryNarrowFp8PackedOccupancyForRows() {
    static_assert(sizeof(InputT) == 1);
    const auto kernel = narrowFp8PackedFlatRCooperativeFirstStageKernel<InputT,
                                                                        ReductionOpT,
                                                                        InputTransformT,
                                                                        RowsPerShard,
                                                                        PacketBytes,
                                                                        MaxComponentsPerLane,
                                                                        DecoderT>;
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

// Retained DeepRSmallK P16 geometry helpers: five compiled depths and seven K
// ownership choices, shared by all four typed FP32 reduction operations.
#if THOR_CUB_ENABLE_FP8_TYPES
template <typename InputT, typename TransformT, int Rows, typename ReductionOpT = cuda::std::plus<float>>
void launchDeepRRetainedRows(const Tensor& input, Tensor& partials,
                                const CubReductionGeometry& geometry, uint64_t shards,
                                cudaStream_t stream, TransformT transform) {
    if (geometry.inner_size <= 4)
        launchNarrowFp8PackedTinyKForTransform<InputT, ReductionOpT, TransformT, Rows, 16, 8>(
            input, partials, geometry, shards, stream, transform);
    else if (geometry.inner_size <= 8)
        launchNarrowFp8PackedTinyKForTransform<InputT, ReductionOpT, TransformT, Rows, 16, 4>(
            input, partials, geometry, shards, stream, transform);
    else if (geometry.inner_size <= 32)
        launchNarrowFp8PackedForTransform<InputT, ReductionOpT, TransformT, Rows, 16, 1>(
            input, partials, geometry, shards, stream, transform);
    else if (geometry.inner_size <= 64)
        launchNarrowFp8PackedForTransform<InputT, ReductionOpT, TransformT, Rows, 16, 2>(
            input, partials, geometry, shards, stream, transform);
    else if (geometry.inner_size <= 128)
        launchNarrowFp8PackedForTransform<InputT, ReductionOpT, TransformT, Rows, 16, 4>(
            input, partials, geometry, shards, stream, transform);
    else if (geometry.inner_size <= 160)
        launchNarrowFp8PackedForTransform<InputT, ReductionOpT, TransformT, Rows, 16, 5>(
            input, partials, geometry, shards, stream, transform);
    else
        launchNarrowFp8PackedForTransform<InputT, ReductionOpT, TransformT, Rows, 16, 8>(
            input, partials, geometry, shards, stream, transform);
}

template <typename InputT, typename TransformT, typename ReductionOpT = cuda::std::plus<float>>
void dispatchDeepRRetainedRows(const Tensor& input, Tensor& partials,
                                  const CubReductionGeometry& geometry, uint64_t rows,
                                  uint64_t shards, cudaStream_t stream, TransformT transform) {
    switch (rows) {
        case 64: launchDeepRRetainedRows<InputT, TransformT, 64, ReductionOpT>(input, partials, geometry, shards, stream, transform); return;
        case 128: launchDeepRRetainedRows<InputT, TransformT, 128, ReductionOpT>(input, partials, geometry, shards, stream, transform); return;
        case 256: launchDeepRRetainedRows<InputT, TransformT, 256, ReductionOpT>(input, partials, geometry, shards, stream, transform); return;
        case 512: launchDeepRRetainedRows<InputT, TransformT, 512, ReductionOpT>(input, partials, geometry, shards, stream, transform); return;
        case 1024: launchDeepRRetainedRows<InputT, TransformT, 1024, ReductionOpT>(input, partials, geometry, shards, stream, transform); return;
    }
    throw std::logic_error("DeepRSmallK transform A/B requires retained row depth.");
}

template <typename InputT, typename TransformT, int Rows, typename ReductionOpT = cuda::std::plus<float>>
CubKernelOccupancyInfo queryDeepRRetainedRows(uint64_t k) {
    if (k <= 4) return queryNarrowFp8PackedTinyKOccupancyForRows<InputT, Rows, 16, 8, TransformT, NarrowFp8TypedFormatDecoder<InputT>, ReductionOpT>();
    if (k <= 8) return queryNarrowFp8PackedTinyKOccupancyForRows<InputT, Rows, 16, 4, TransformT, NarrowFp8TypedFormatDecoder<InputT>, ReductionOpT>();
    if (k <= 32) return queryNarrowFp8PackedOccupancyForRows<InputT, Rows, 16, 1, TransformT, NarrowFp8TypedFormatDecoder<InputT>, ReductionOpT>();
    if (k <= 64) return queryNarrowFp8PackedOccupancyForRows<InputT, Rows, 16, 2, TransformT, NarrowFp8TypedFormatDecoder<InputT>, ReductionOpT>();
    if (k <= 128) return queryNarrowFp8PackedOccupancyForRows<InputT, Rows, 16, 4, TransformT, NarrowFp8TypedFormatDecoder<InputT>, ReductionOpT>();
    if (k <= 160) return queryNarrowFp8PackedOccupancyForRows<InputT, Rows, 16, 5, TransformT, NarrowFp8TypedFormatDecoder<InputT>, ReductionOpT>();
    return queryNarrowFp8PackedOccupancyForRows<InputT, Rows, 16, 8, TransformT, NarrowFp8TypedFormatDecoder<InputT>, ReductionOpT>();
}

template <typename InputT, typename TransformT, typename ReductionOpT = cuda::std::plus<float>>
CubKernelOccupancyInfo queryDeepRRetainedRows(uint64_t rows, uint64_t k) {
    switch (rows) {
        case 64: return queryDeepRRetainedRows<InputT, TransformT, 64, ReductionOpT>(k);
        case 128: return queryDeepRRetainedRows<InputT, TransformT, 128, ReductionOpT>(k);
        case 256: return queryDeepRRetainedRows<InputT, TransformT, 256, ReductionOpT>(k);
        case 512: return queryDeepRRetainedRows<InputT, TransformT, 512, ReductionOpT>(k);
        case 1024: return queryDeepRRetainedRows<InputT, TransformT, 1024, ReductionOpT>(k);
    }
    throw std::logic_error("DeepRSmallK transform A/B occupancy requires retained row depth.");
}

// Retained DeepRSmallK P16 kernels. The only compile-time axes are dtype,
// input transform, reduction combine, depth and the existing K ownership.
// These wrappers deliberately share the benchmark-proven device kernels with
// the earlier SUM calibration; no new geometry or runtime operation switch.
template <typename InputT, typename ReductionOpT, typename TransformT>
void launchDeepRRetainedTyped(const Tensor& input, Tensor& partials,
                              const CubReductionGeometry& geometry, uint64_t rows,
                              uint64_t shards, cudaStream_t stream) {
    dispatchDeepRRetainedRows<InputT, TransformT, ReductionOpT>(
        input, partials, geometry, rows, shards, stream, TransformT{});
}

template <typename InputT, typename ReductionOpT, typename TransformT>
CubKernelOccupancyInfo queryDeepRRetainedTyped(uint64_t rows, uint64_t k) {
    return queryDeepRRetainedRows<InputT, TransformT, ReductionOpT>(rows, k);
}

template <typename InputT, typename ReductionOpT, typename Callback>
auto dispatchDeepRRetainedTransform(CubReductionStageInputTransform transform, Callback&& cb) {
    switch (transform) {
        case CubReductionStageInputTransform::Identity:
            return cb.template operator()<InputT, ReductionOpT, IdentityFp32>();
        case CubReductionStageInputTransform::AbsoluteValue:
            return cb.template operator()<InputT, ReductionOpT, AbsoluteValueFp32>();
        case CubReductionStageInputTransform::Square:
            return cb.template operator()<InputT, ReductionOpT, SquareFp32>();
    }
    throw std::logic_error("DeepRSmallK retained kernel received invalid input transform.");
}

template <typename InputT, typename Callback>
auto dispatchDeepRRetainedCombine(const CubReductionStageSemantics& semantics, Callback&& cb) {
    switch (semantics.combine) {
        case CubReductionStageCombine::Sum:
            return dispatchDeepRRetainedTransform<InputT, cuda::std::plus<float>>(
                semantics.input_transform, cb);
        case CubReductionStageCombine::Product:
            return dispatchDeepRRetainedTransform<InputT, cuda::std::multiplies<float>>(
                semantics.input_transform, cb);
        case CubReductionStageCombine::Minimum:
            return dispatchDeepRRetainedTransform<InputT, PropagatingMinimumFp32>(
                semantics.input_transform, cb);
        case CubReductionStageCombine::Maximum:
            return dispatchDeepRRetainedTransform<InputT, PropagatingMaximumFp32>(
                semantics.input_transform, cb);
    }
    throw std::logic_error("DeepRSmallK retained kernel received invalid reduction combine.");
}

// Retained OneWindow physical inventory. Warps per output is a runtime operand;
// (outputs per warp, packet bytes) and K are compile-time kernel axes. The selector
// policy table Fp8OneWindowPolicy::candidates is the single source of truth: the
// launcher and occupancy query instantiate exactly those pairs, at each compiled K
// whose packet window covers K, and nothing else.
template <int OutputsPerWarp, int PacketBytes>
[[nodiscard]] constexpr bool oneWindowRetainedOwnership() {
    for (const auto& variant : Fp8RK::Fp8OneWindowPolicy::candidates) {
        if (variant.outputs_per_warp == static_cast<uint32_t>(OutputsPerWarp) &&
            variant.packet_bytes == static_cast<uint32_t>(PacketBytes)) {
            return true;
        }
    }
    return false;
}

template <int K, int OutputsPerWarp, int PacketBytes>
[[nodiscard]] constexpr bool oneWindowCompiledSpecialization() {
    constexpr uint32_t PacketWindow =
        (TILED_REDUCTION_WARP_THREADS / static_cast<uint32_t>(OutputsPerWarp)) * static_cast<uint32_t>(PacketBytes);
    constexpr uint32_t Capacity =
        (K % PacketBytes) == 0 ? PacketWindow : PacketWindow - static_cast<uint32_t>(PacketBytes - 1);
    return oneWindowRetainedOwnership<OutputsPerWarp, PacketBytes>() && K <= static_cast<int>(Capacity);
}

[[nodiscard]] inline bool oneWindowValidWarpsPerOutput(uint32_t outputs_per_warp, uint32_t warps_per_output) {
    const bool valid_warps =
        warps_per_output == 1 || warps_per_output == 2 || warps_per_output == 4 || warps_per_output == 8;
    // O2/O4/O8 partition one warp among outputs and therefore require W1.
    return valid_warps && (outputs_per_warp == 1 || warps_per_output == 1);
}

// Invokes callback.template operator()<InputT, K, OutputsPerWarp, PacketBytes>() for the one
// retained compiled OneWindow specialization matching the runtime request. Returns false,
// without invoking the callback, when that specialization is not part of the retained inventory.
template <typename Callback>
[[nodiscard]] bool dispatchRetainedOneWindow(DataType input_dtype,
                                             uint64_t inner_size,
                                             uint32_t outputs_per_warp,
                                             uint32_t packet_bytes,
                                             Callback&& callback) {
    bool dispatched = false;
    const auto for_k = [&]<typename InputT, int K>() {
        const auto for_packet = [&]<int OutputsPerWarp, int PacketBytes>() {
            if constexpr (oneWindowCompiledSpecialization<K, OutputsPerWarp, PacketBytes>()) {
                callback.template operator()<InputT, K, OutputsPerWarp, PacketBytes>();
                dispatched = true;
            }
        };
        const auto for_outputs = [&]<int OutputsPerWarp>() {
            switch (packet_bytes) {
                case 4: for_packet.template operator()<OutputsPerWarp, 4>(); return;
                case 8: for_packet.template operator()<OutputsPerWarp, 8>(); return;
                case 16: for_packet.template operator()<OutputsPerWarp, 16>(); return;
                default: return;
            }
        };
        switch (outputs_per_warp) {
            case 1: for_outputs.template operator()<1>(); return;
            case 2: for_outputs.template operator()<2>(); return;
            case 4: for_outputs.template operator()<4>(); return;
            case 8: for_outputs.template operator()<8>(); return;
            default: return;
        }
    };
    const auto for_type = [&]<typename InputT>() {
        switch (inner_size) {
            case 1: for_k.template operator()<InputT, 1>(); return;
            case 2: for_k.template operator()<InputT, 2>(); return;
            case 3: for_k.template operator()<InputT, 3>(); return;
            case 4: for_k.template operator()<InputT, 4>(); return;
            case 5: for_k.template operator()<InputT, 5>(); return;
            case 7: for_k.template operator()<InputT, 7>(); return;
            case 8: for_k.template operator()<InputT, 8>(); return;
            case 9: for_k.template operator()<InputT, 9>(); return;
            case 11: for_k.template operator()<InputT, 11>(); return;
            case 13: for_k.template operator()<InputT, 13>(); return;
            case 15: for_k.template operator()<InputT, 15>(); return;
            case 16: for_k.template operator()<InputT, 16>(); return;
            case 17: for_k.template operator()<InputT, 17>(); return;
            case 19: for_k.template operator()<InputT, 19>(); return;
            case 31: for_k.template operator()<InputT, 31>(); return;
            case 32: for_k.template operator()<InputT, 32>(); return;
            case 63: for_k.template operator()<InputT, 63>(); return;
            case 64: for_k.template operator()<InputT, 64>(); return;
            case 127: for_k.template operator()<InputT, 127>(); return;
            case 128: for_k.template operator()<InputT, 128>(); return;
            case 129: for_k.template operator()<InputT, 129>(); return;
            case 255: for_k.template operator()<InputT, 255>(); return;
            case 256: for_k.template operator()<InputT, 256>(); return;
            case 497: for_k.template operator()<InputT, 497>(); return;
            case 512: for_k.template operator()<InputT, 512>(); return;
            default: return;
        }
    };
    switch (input_dtype) {
        case DataType::FP8_E4M3: for_type.template operator()<__nv_fp8_e4m3>(); break;
        case DataType::FP8_E5M2: for_type.template operator()<__nv_fp8_e5m2>(); break;
        default: break;
    }
    return dispatched;
}

#endif

}  // namespace

void launchFp8DeepRSmallKRetainedFirstStage(const CubReductionStageSemantics& semantics,
                                            const Tensor& input, Tensor& fp32_partials,
                                            const CubReductionGeometry& geometry, uint64_t rows,
                                            uint64_t shards, Stream& stream) {
#if THOR_CUB_ENABLE_FP8_TYPES
    if (rows != 64 && rows != 128 && rows != 256 && rows != 512 && rows != 1024)
        throw std::logic_error("DeepRSmallK retained first stage requires a selected depth.");
    if (geometry.outer_size == 0 || geometry.reduction_size <= rows ||
        geometry.inner_size == 0 || geometry.inner_size > 256 ||
        shards != ceilDivideU64(geometry.reduction_size, rows) ||
        input.getDimensions() != std::vector<uint64_t>{geometry.outer_size, geometry.reduction_size, geometry.inner_size} ||
        fp32_partials.getDataType() != DataType::FP32 ||
        fp32_partials.getDimensions() != std::vector<uint64_t>{geometry.outer_size, shards, geometry.inner_size}) {
        throw std::logic_error("DeepRSmallK retained first stage received illegal geometry or partials.");
    }
    // All partials are raw FP32. Finalization (including MEAN division and L2
    // square root) is the responsibility of the independently planned final stage.
    const auto launch = [&]<typename InputT, typename ReductionOpT, typename TransformT>() {
        launchDeepRRetainedTyped<InputT, ReductionOpT, TransformT>(
            input, fp32_partials, geometry, rows, shards, stream.getStream());
    };
    switch (input.getDataType()) {
        case DataType::FP8_E4M3:
            dispatchDeepRRetainedCombine<__nv_fp8_e4m3>(semantics, launch);
            return;
        case DataType::FP8_E5M2:
            dispatchDeepRRetainedCombine<__nv_fp8_e5m2>(semantics, launch);
            return;
        default:
            throw std::logic_error("DeepRSmallK retained first stage requires E4M3 or E5M2.");
    }
#else
    static_cast<void>(semantics);
    static_cast<void>(input);
    static_cast<void>(fp32_partials);
    static_cast<void>(geometry);
    static_cast<void>(rows);
    static_cast<void>(shards);
    static_cast<void>(stream);
    throw std::logic_error("DeepRSmallK retained first stage requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}

CubKernelOccupancyInfo queryFp8DeepRSmallKRetainedOccupancy(
    DataType dtype, uint64_t rows, uint64_t k, const CubReductionStageSemantics& semantics) {
#if THOR_CUB_ENABLE_FP8_TYPES
    if (k == 0 || k > 256 || (rows != 64 && rows != 128 && rows != 256 && rows != 512 && rows != 1024))
        throw std::logic_error("DeepRSmallK retained occupancy requires selected P16 depth and K <= 256.");
    const auto query = [&]<typename InputT, typename ReductionOpT, typename TransformT>() -> CubKernelOccupancyInfo {
        return queryDeepRRetainedTyped<InputT, ReductionOpT, TransformT>(rows, k);
    };
    switch (dtype) {
        case DataType::FP8_E4M3: return dispatchDeepRRetainedCombine<__nv_fp8_e4m3>(semantics, query);
        case DataType::FP8_E5M2: return dispatchDeepRRetainedCombine<__nv_fp8_e5m2>(semantics, query);
        default: throw std::logic_error("DeepRSmallK retained occupancy requires E4M3 or E5M2.");
    }
#else
    static_cast<void>(dtype);
    static_cast<void>(rows);
    static_cast<void>(k);
    static_cast<void>(semantics);
    throw std::logic_error("DeepRSmallK retained occupancy requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}

template <int FixedOp>
void launchFp8GeneralRParallelKTileRCooperativeCalibrationCompleteTyped(
    const Tensor& input,
    Tensor& output,
    const CubReductionGeometry& geometry,
    uint32_t warps_per_output,
    uint32_t tile_components,
    Stream& stream,
    const CubReductionStageSemantics& semantics,
    float output_scale) {
#if THOR_CUB_ENABLE_FP8_TYPES
    const AdditiveFinalizeFp32 output_finalize = fp8CompleteFinalizerFromSemantics(semantics);
    if (geometry.outer_size == 0 || geometry.reduction_size == 0 || geometry.inner_size == 0 ||
        geometry.inner_size > std::numeric_limits<uint32_t>::max() ||
        !shallowFp8FitsUint32Indexing(geometry) ||
        !fp8CompleteSupportsOutputStorage(output.getDataType()) || output.getDimensions() != geometry.output_dimensions ||
        tile_components == 0 || tile_components > 496) {
        throw std::logic_error("FP8 tiled general RParallel calibration received unsupported Complete geometry.");
    }
    // RuntimeK packet-complete deliberately spends three compiled input-transform
    // variants per input dtype. Dispatch only once on the host; no uniform transform
    // branch remains in the hot device accumulation loop. Output dtype and additive
    // finalization stay runtime parameters and do not multiply this inventory.
    const auto dispatch = [&]<typename InputT>() {
        switch (semantics.input_transform) {
            case CubReductionStageInputTransform::Identity:
                launchFp8GeneralRParallelKTileP16Complete<InputT, 0, FixedOp>(
                    input, output, geometry, warps_per_output, tile_components,
                    stream.getStream(), output_finalize,
                    CubReductionStageInputTransform::Identity, output_scale);
                return;
            case CubReductionStageInputTransform::AbsoluteValue:
                launchFp8GeneralRParallelKTileP16Complete<InputT, 1, FixedOp>(
                    input, output, geometry, warps_per_output, tile_components,
                    stream.getStream(), output_finalize,
                    CubReductionStageInputTransform::Identity, output_scale);
                return;
            case CubReductionStageInputTransform::Square:
                launchFp8GeneralRParallelKTileP16Complete<InputT, 2, FixedOp>(
                    input, output, geometry, warps_per_output, tile_components,
                    stream.getStream(), output_finalize,
                    CubReductionStageInputTransform::Identity, output_scale);
                return;
        }
        throw std::logic_error("FP8 tiled general RParallel received unsupported input transform.");
    };
    switch (input.getDataType()) {
        case DataType::FP8_E4M3: dispatch.template operator()<__nv_fp8_e4m3>(); return;
        case DataType::FP8_E5M2: dispatch.template operator()<__nv_fp8_e5m2>(); return;
        default:
            throw std::logic_error("FP8 tiled general RParallel calibration requires FP8 input.");
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
    throw std::logic_error("FP8 tiled general RParallel calibration requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}


void launchFp8GeneralRParallelKTileRCooperativeCalibrationComplete(
    const Tensor& input,
    Tensor& output,
    const CubReductionGeometry& geometry,
    uint32_t warps_per_output,
    uint32_t tile_components,
    Stream& stream,
    const CubReductionStageSemantics& semantics,
    float output_scale) {
    switch (semantics.combine) {
        case CubReductionStageCombine::Sum:
            return launchFp8GeneralRParallelKTileRCooperativeCalibrationCompleteTyped<0>(input, output, geometry, warps_per_output, tile_components, stream, semantics, output_scale);
        case CubReductionStageCombine::Product:
            return launchFp8GeneralRParallelKTileRCooperativeCalibrationCompleteTyped<1>(input, output, geometry, warps_per_output, tile_components, stream, semantics, output_scale);
        case CubReductionStageCombine::Minimum:
            return launchFp8GeneralRParallelKTileRCooperativeCalibrationCompleteTyped<2>(input, output, geometry, warps_per_output, tile_components, stream, semantics, output_scale);
        case CubReductionStageCombine::Maximum:
            return launchFp8GeneralRParallelKTileRCooperativeCalibrationCompleteTyped<3>(input, output, geometry, warps_per_output, tile_components, stream, semantics, output_scale);
    }
    throw std::logic_error("Invalid RuntimeK Complete combine operation.");
}

// Query the transform-specific Complete specialization; its register usage and
// occupancy may differ across input transforms.
template <int FixedOp>
CubKernelOccupancyInfo queryFp8GeneralRParallelKTileRCooperativeCalibrationOccupancyTyped(
    DataType input_dtype, CubReductionStageInputTransform transform) {
#if THOR_CUB_ENABLE_FP8_TYPES
    const auto dispatch = [&]<typename InputT>() -> CubKernelOccupancyInfo {
        switch (transform) {
            case CubReductionStageInputTransform::Identity:
                return queryShallowFp8Occupancy(fp8GeneralRParallelKTileP16CompleteKernel<InputT, 0, FixedOp>);
            case CubReductionStageInputTransform::AbsoluteValue:
                return queryShallowFp8Occupancy(fp8GeneralRParallelKTileP16CompleteKernel<InputT, 1, FixedOp>);
            case CubReductionStageInputTransform::Square:
                return queryShallowFp8Occupancy(fp8GeneralRParallelKTileP16CompleteKernel<InputT, 2, FixedOp>);
        }
        throw std::logic_error("RuntimeK Complete occupancy requires a supported input transform.");
    };
    switch (input_dtype) {
        case DataType::FP8_E4M3: return dispatch.template operator()<__nv_fp8_e4m3>();
        case DataType::FP8_E5M2: return dispatch.template operator()<__nv_fp8_e5m2>();
        default:
            throw std::logic_error("FP8 tiled general RParallel occupancy requires FP8 input.");
    }
#else
    static_cast<void>(input_dtype); static_cast<void>(transform);
    throw std::logic_error("FP8 tiled general RParallel occupancy requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}


CubKernelOccupancyInfo queryFp8GeneralRParallelKTileRCooperativeCalibrationOccupancy(
    DataType input_dtype, CubReductionStageInputTransform transform, CubReductionStageCombine combine) {
    switch (combine) {
        case CubReductionStageCombine::Sum:
            return queryFp8GeneralRParallelKTileRCooperativeCalibrationOccupancyTyped<0>(input_dtype, transform);
        case CubReductionStageCombine::Product:
            return queryFp8GeneralRParallelKTileRCooperativeCalibrationOccupancyTyped<1>(input_dtype, transform);
        case CubReductionStageCombine::Minimum:
            return queryFp8GeneralRParallelKTileRCooperativeCalibrationOccupancyTyped<2>(input_dtype, transform);
        case CubReductionStageCombine::Maximum:
            return queryFp8GeneralRParallelKTileRCooperativeCalibrationOccupancyTyped<3>(input_dtype, transform);
    }
    throw std::logic_error("Invalid RuntimeK Complete occupancy combine operation.");
}

template <int FixedOp>
void launchFp8GeneralRParallelRCooperativeCalibrationCompleteTyped(
    const Tensor& input,
    Tensor& output,
    const CubReductionGeometry& geometry,
    uint32_t outputs_per_warp,
    uint32_t warps_per_output,
    uint32_t packet_bytes,
    Stream& stream,
    const CubReductionStageSemantics& semantics,
    float output_scale) {
#if THOR_CUB_ENABLE_FP8_TYPES
    const AdditiveFinalizeFp32 output_finalize = fp8CompleteFinalizerFromSemantics(semantics);
    if (geometry.outer_size == 0 || geometry.reduction_size == 0 ||
        !rParallelFp8SupportedCalibrationK(geometry.inner_size) ||
        !rParallelFp8PacketSupportsOwnership(geometry.inner_size, outputs_per_warp, packet_bytes) ||
        !oneWindowValidWarpsPerOutput(outputs_per_warp, warps_per_output) ||
        !shallowFp8FitsUint32Indexing(geometry) ||
        !fp8CompleteSupportsOutputStorage(output.getDataType()) || output.getDimensions() != geometry.output_dimensions) {
        throw std::logic_error("FP8 OneWindow received unsupported complete geometry.");
    }
    const bool launched = dispatchRetainedOneWindow(
        input.getDataType(), geometry.inner_size, outputs_per_warp, packet_bytes,
        [&]<typename InputT, int K, int OutputsPerWarp, int PacketBytes>() {
            launchFp8GeneralRParallelPackedForOwnership<InputT, K, PacketBytes, OutputsPerWarp, FixedOp>(
                input, output, geometry, warps_per_output, stream.getStream(), output_finalize,
                semantics.input_transform, output_scale);
        });
    if (!launched) {
        throw std::logic_error("FP8 OneWindow request is not a retained compiled specialization.");
    }
#else
    static_cast<void>(input);
    static_cast<void>(output);
    static_cast<void>(geometry);
    static_cast<void>(outputs_per_warp);
    static_cast<void>(warps_per_output);
    static_cast<void>(packet_bytes);
    static_cast<void>(stream);
    static_cast<void>(semantics);
    static_cast<void>(output_scale);
    throw std::logic_error("FP8 OneWindow requires THOR_CUB_ENABLE_FP8_TYPES.");
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
    const CubReductionStageSemantics& semantics,
    float output_scale) {
    switch (semantics.combine) {
        case CubReductionStageCombine::Sum:
            return launchFp8GeneralRParallelRCooperativeCalibrationCompleteTyped<0>(input, output, geometry, outputs_per_warp, warps_per_output, packet_bytes, stream, semantics, output_scale);
        case CubReductionStageCombine::Product:
            return launchFp8GeneralRParallelRCooperativeCalibrationCompleteTyped<1>(input, output, geometry, outputs_per_warp, warps_per_output, packet_bytes, stream, semantics, output_scale);
        case CubReductionStageCombine::Minimum:
            return launchFp8GeneralRParallelRCooperativeCalibrationCompleteTyped<2>(input, output, geometry, outputs_per_warp, warps_per_output, packet_bytes, stream, semantics, output_scale);
        case CubReductionStageCombine::Maximum:
            return launchFp8GeneralRParallelRCooperativeCalibrationCompleteTyped<3>(input, output, geometry, outputs_per_warp, warps_per_output, packet_bytes, stream, semantics, output_scale);
    }
    throw std::logic_error("Invalid OneWindow combine operation.");
}

template <int FixedOp>
CubKernelOccupancyInfo queryFp8GeneralRParallelRCooperativeCalibrationOccupancyTyped(
    DataType input_dtype,
    uint64_t inner_size,
    uint32_t outputs_per_warp,
    uint32_t warps_per_output,
    uint32_t packet_bytes) {
#if THOR_CUB_ENABLE_FP8_TYPES
    if (!rParallelFp8SupportedCalibrationK(inner_size) ||
        !rParallelFp8PacketSupportsOwnership(inner_size, outputs_per_warp, packet_bytes) ||
        !oneWindowValidWarpsPerOutput(outputs_per_warp, warps_per_output)) {
        throw std::logic_error("FP8 OneWindow occupancy received unsupported ownership.");
    }
    CubKernelOccupancyInfo occupancy{};
    const bool queried = dispatchRetainedOneWindow(
        input_dtype, inner_size, outputs_per_warp, packet_bytes,
        [&]<typename InputT, int K, int OutputsPerWarp, int PacketBytes>() {
            occupancy = queryFp8GeneralRParallelPackedOccupancyForOwnership<InputT, K, PacketBytes, OutputsPerWarp, FixedOp>(
                warps_per_output);
        });
    if (!queried) {
        // Absent physical specialization: the selector treats this as an excluded candidate.
        throw std::logic_error("FP8 OneWindow occupancy request is not a retained compiled specialization.");
    }
    return occupancy;
#else
    static_cast<void>(input_dtype);
    static_cast<void>(inner_size);
    static_cast<void>(outputs_per_warp);
    static_cast<void>(warps_per_output);
    static_cast<void>(packet_bytes);
    throw std::logic_error("FP8 OneWindow requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}



CubKernelOccupancyInfo queryFp8GeneralRParallelRCooperativeCalibrationOccupancy(
    DataType input_dtype,
    uint64_t inner_size,
    uint32_t outputs_per_warp,
    uint32_t warps_per_output,
    uint32_t packet_bytes,
    CubReductionStageCombine combine) {
    switch (combine) {
        case CubReductionStageCombine::Sum:
            return queryFp8GeneralRParallelRCooperativeCalibrationOccupancyTyped<0>(input_dtype, inner_size, outputs_per_warp, warps_per_output, packet_bytes);
        case CubReductionStageCombine::Product:
            return queryFp8GeneralRParallelRCooperativeCalibrationOccupancyTyped<1>(input_dtype, inner_size, outputs_per_warp, warps_per_output, packet_bytes);
        case CubReductionStageCombine::Minimum:
            return queryFp8GeneralRParallelRCooperativeCalibrationOccupancyTyped<2>(input_dtype, inner_size, outputs_per_warp, warps_per_output, packet_bytes);
        case CubReductionStageCombine::Maximum:
            return queryFp8GeneralRParallelRCooperativeCalibrationOccupancyTyped<3>(input_dtype, inner_size, outputs_per_warp, warps_per_output, packet_bytes);
    }
    throw std::logic_error("Invalid OneWindow occupancy combine operation.");
}

CubKernelOccupancyInfo queryNarrowLowPrecisionFlatRCooperativeFirstStageOccupancy(DataType input_dtype, uint64_t rows_per_shard) {
    switch (input_dtype) {
        case DataType::FP16:
            return queryNarrowFlatOccupancy<__half>(rows_per_shard);
        case DataType::BF16:
            return queryNarrowFlatOccupancy<__nv_bfloat16>(rows_per_shard);
        default:
            throw std::logic_error("Narrow flat R-cooperative occupancy query requires FP16 or BF16 input.");
    }
}

// The general narrow-flat RK stage remains a production FP16/BF16 primitive.
// FP8 original-input RK is served exclusively by the retained FP8 selector;
// DeepRSmallK has a separate validated retained launch and inventory.
static void validateNarrowFlatFirstStage(const Tensor& input,
                                         const Tensor& fp32_partials,
                                         const CubReductionGeometry& geometry,
                                         uint64_t rows_per_shard,
                                         uint64_t shards_per_output) {
    if (input.getDataType() != DataType::FP16 && input.getDataType() != DataType::BF16)
        throw std::logic_error("Narrow-flat RK first stage requires FP16 or BF16 input.");
    if (rows_per_shard != 128 && rows_per_shard != 256 &&
        rows_per_shard != 512 && rows_per_shard != 1024)
        throw std::logic_error("Narrow-flat RK first stage requires a compiled row depth.");
    if (geometry.outer_size == 0 || geometry.reduction_size <= rows_per_shard ||
        geometry.inner_size == 0 || geometry.inner_size > NARROW_FLAT_MAX_K ||
        shards_per_output != ceilDivideU64(geometry.reduction_size, rows_per_shard))
        throw std::logic_error("Narrow-flat RK first stage requires legal staged geometry.");
    // The ordinary dense planner may present contiguous [O,R,K] storage as an
    // equivalent compact rank-two tensor (or reshape a stamped intermediate).
    // This kernel addresses contiguous storage using the physical O/R/K extents,
    // not Tensor's logical rank. Validate the allocation sizes without imposing
    // a redundant, rank-specific representation on the executor. Avoid
    // getDimensions() here: execution must not allocate temporary vectors.
    if (input.getTotalNumElements() != geometry.input_elements ||
        fp32_partials.getDataType() != DataType::FP32 ||
        fp32_partials.getTotalNumElements() % shards_per_output != 0 ||
        fp32_partials.getTotalNumElements() / shards_per_output != geometry.output_elements)
        throw std::logic_error("Narrow-flat RK first-stage tensor storage does not match geometry.");
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
