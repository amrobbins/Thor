#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"
#include "Utilities/TensorOperations/Cub/CubReductionOperation.cuh"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8DeepRSmallKSelector.h"

#include "Utilities/Exceptions.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
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
// OneWindow is the bucketed runtime-K kernel (ONE_WINDOW_RUNTIME_K_UNITS); RuntimeK packet-complete is runtime-K tiled.
constexpr int SHALLOW_FP8_WARPS = 8;
constexpr int SHALLOW_FP8_THREADS = SHALLOW_FP8_WARPS * TILED_REDUCTION_WARP_THREADS;
static_assert(SHALLOW_FP8_THREADS == 256);

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
// AdditiveFinalize is FixedOp == 0: fp8CompleteFinalizerFromSemantics() only
// hands PRODUCT/MIN/MAX the identity finalizer, so their kernels compile none.
// SUM keeps the identity test inline and calls the shared out-of-line divide /
// square-root finalizer only for MEAN and the square-root operations (P5-A).
template <bool AdditiveFinalize>
__device__ __forceinline__ void storeFp8CompleteResult(void* output,
                                                        DataType output_dtype,
                                                        uint64_t index,
                                                        float value,
                                                        AdditiveFinalizeFp32 finalizer,
                                                        float output_scale) {
    if constexpr (AdditiveFinalize) {
        value = finalizeFp32Epilogue(finalizer, value);
    } else {
        static_cast<void>(finalizer);
    }
    storeFp32AsRuntimeDType(output, output_dtype, index, value * output_scale);
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

// One OneWindow R group: group_rows logical rows starting at a row, staged as packet_count packets beginning
// prefix_bytes before the group's first byte.
struct OneWindowGroupWindow {
    uint32_t rows;
    uint32_t prefix_bytes;
    uint32_t packet_count;
    uint32_t aligned_byte_begin;
};

// Loads one packet; a packet that would cross the end of the input tensor is assembled bytewise.
template <int PacketBytes>
__device__ __forceinline__ RawVectorPacket<PacketBytes> loadOneWindowPacket(const uint8_t* input_bytes,
                                                                            uint32_t packet_byte_begin,
                                                                            uint32_t total_input_bytes) {
    RawVectorPacket<PacketBytes> packet{};
    if (total_input_bytes >= static_cast<uint32_t>(PacketBytes) &&
        packet_byte_begin <= total_input_bytes - static_cast<uint32_t>(PacketBytes)) {
        packet = *reinterpret_cast<const RawVectorPacket<PacketBytes>*>(input_bytes + packet_byte_begin);
    } else {
        auto* packet_bytes = reinterpret_cast<uint8_t*>(&packet.value);
#pragma unroll
        for (uint32_t byte = 0; byte < PacketBytes; ++byte) {
            const uint32_t byte_index = packet_byte_begin + byte;
            if (byte_index < total_input_bytes) {
                packet_bytes[byte] = input_bytes[byte_index];
            }
        }
    }
    return packet;
}

// Plane-major staging: word w of lane's packet goes to warp_words[w * 32 + lane].
template <int PacketBytes>
__device__ __forceinline__ void stageOneWindowPacket(uint32_t* warp_words,
                                                     uint32_t lane,
                                                     const RawVectorPacket<PacketBytes>& packet) {
    if constexpr (PacketBytes == 4) {
        warp_words[lane] = packet.value;
    } else if constexpr (PacketBytes == 8) {
        warp_words[0 * TILED_REDUCTION_WARP_THREADS + lane] = packet.value.x;
        warp_words[1 * TILED_REDUCTION_WARP_THREADS + lane] = packet.value.y;
    } else {
        warp_words[0 * TILED_REDUCTION_WARP_THREADS + lane] = packet.value.x;
        warp_words[1 * TILED_REDUCTION_WARP_THREADS + lane] = packet.value.y;
        warp_words[2 * TILED_REDUCTION_WARP_THREADS + lane] = packet.value.z;
        warp_words[3 * TILED_REDUCTION_WARP_THREADS + lane] = packet.value.w;
    }
}

// OneWindow (P5-I bucketed, the only OneWindow kernel since P5-J2): the former compile-time-K OneWindow algorithm
// with K as a runtime operand. Packet width and ownership (outputs per warp) stay compile-time axes, and K is
// bucketed by the maximum number of components per lane (1, 2, 4, 8, 16), so local[] and the slot loop stay
// compile-time (unused slots are skipped by the owned_component < K guard). Everything compile-time K provided is
// computed once on the host (Fp8OneWindowRuntimeKGeometry): rows per staged group, R workers per component and the
// accumulating lane count. The group loop is either plain or load-ahead (LoadAhead), fixed per unit. Decode packing
// differs by bucket (decodeOneWindowRuntimeKGroup); everything is force-inlined functions, no lambdas holding
// local[] (P5-I2).
struct Fp8OneWindowRuntimeKGeometry {
    uint32_t k;                      // components per output
    uint32_t rows_per_group;         // logical rows staged per packet window
    uint32_t workers_per_component;  // R workers per component (1 when a lane owns several components)
    uint32_t task_count;             // accumulating lanes per output
};

template <uint32_t PacketBytes>
__device__ __forceinline__ OneWindowGroupWindow oneWindowRuntimeKGroupWindow(bool valid_output,
                                                                             uint32_t outer_byte_begin,
                                                                             uint32_t group_row,
                                                                             uint32_t reduction_size,
                                                                             uint32_t k,
                                                                             uint32_t rows_per_group) {
    const uint32_t remaining_rows = reduction_size - group_row;
    OneWindowGroupWindow window{remaining_rows < rows_per_group ? remaining_rows : rows_per_group, 0U, 0U, 0U};
    if (valid_output) {
        const uint32_t group_byte_begin = outer_byte_begin + group_row * k;
        constexpr uint32_t PacketShift = PacketBytes == 4 ? 2U : (PacketBytes == 8 ? 3U : 4U);
        window.prefix_bytes = group_byte_begin & (PacketBytes - 1U);
        window.aligned_byte_begin = group_byte_begin - window.prefix_bytes;
        window.packet_count = (window.prefix_bytes + window.rows * k + PacketBytes - 1U) >> PacketShift;
    }
    return window;
}

// One staged FP8 byte of a subgroup slice (plane-major: word w of the subgroup's packet p at w * 32 + base + p).
template <uint32_t PacketBytes>
__device__ __forceinline__ uint32_t oneWindowStagedByte(const uint32_t* warp_words,
                                                        uint32_t subgroup_lane_base,
                                                        uint32_t staged_byte) {
    constexpr uint32_t PacketShift = PacketBytes == 4 ? 2U : (PacketBytes == 8 ? 3U : 4U);
    const uint32_t packet_byte = staged_byte & (PacketBytes - 1U);
    const uint32_t word = warp_words[(packet_byte >> 2) * TILED_REDUCTION_WARP_THREADS + subgroup_lane_base +
                                     (staged_byte >> PacketShift)];
    return (word >> ((packet_byte & 3U) * 8U)) & 0xffU;
}

template <typename InputT, int FixedOp>
__device__ __forceinline__ float4 decodeOneWindowRuntimeKPacked(uint32_t packed,
                                                                CubReductionStageInputTransform input_transform,
                                                                const Fp8ExtremumFold& extremum_fold) {
    return transformFp8CompleteDecodedPacket<FixedOp == 2 ? 0 : -1>(
        decodeFp8x4ToFp32<InputT>(foldFp8ExtremumBytes<FixedOp == 2>(packed, extremum_fold)), input_transform);
}

// Decodes one staged group into local[]. Rows of each component are accumulated in increasing order, exactly as in
// the former compile-time-K kernel (1092-1097 checked bit identity against it).
//  - Buckets 1 and 2 pack four rows of one component per FP8x4 decode (the compile-time-K scheme), with
//    unpredicated full steps and one predicated tail.
//  - Buckets 4..16 pack four of the lane's components of one row per decode. Large K leaves few rows per group
//    (one row at K 497), where packing rows would decode one valid value per conversion; this packs four, with the
//    same byte reads.
template <typename InputT, int FixedOp, int MaxComponentsPerLane, uint32_t LanesPerOutput, uint32_t PacketBytes>
__device__ __forceinline__ void decodeOneWindowRuntimeKGroup(float (&local)[MaxComponentsPerLane],
                                                             const uint32_t* warp_words,
                                                             uint32_t subgroup_lane,
                                                             uint32_t subgroup_lane_base,
                                                             uint32_t component,
                                                             uint32_t worker,
                                                             uint32_t k,
                                                             uint32_t workers,
                                                             uint32_t group_rows,
                                                             uint32_t prefix_bytes,
                                                             CubReductionStageInputTransform input_transform,
                                                             const Fp8ExtremumFold& extremum_fold) {
    if constexpr (MaxComponentsPerLane <= 2) {
        constexpr bool MultipleComponentsPerLane = MaxComponentsPerLane > 1;
        const uint32_t row_stride = workers * k;  // bytes between a worker's consecutive rows
#pragma unroll
        for (uint32_t slot = 0; slot < static_cast<uint32_t>(MaxComponentsPerLane); ++slot) {
            const uint32_t owned_component =
                MultipleComponentsPerLane ? subgroup_lane + slot * LanesPerOutput : component;
            if (owned_component >= k) {
                continue;
            }
            uint32_t row = worker;
            uint32_t byte = prefix_bytes + worker * k + owned_component;
            for (; row + 3U * workers < group_rows; row += 4U * workers, byte += 4U * row_stride) {
                const auto staged = [&](uint32_t staged_byte) {
                    return oneWindowStagedByte<PacketBytes>(warp_words, subgroup_lane_base, staged_byte);
                };
                const uint32_t packed = staged(byte) | (staged(byte + row_stride) << 8) |
                                        (staged(byte + 2U * row_stride) << 16) |
                                        (staged(byte + 3U * row_stride) << 24);
                local[slot] = reduceOneWindowDecoded4<FixedOp>(
                    local[slot], decodeOneWindowRuntimeKPacked<InputT, FixedOp>(packed, input_transform, extremum_fold),
                    4U);
            }
            if (row < group_rows) {
                uint32_t packed = 0;
                uint32_t valid_values = 0;
#pragma unroll
                for (uint32_t item = 0; item < 4U; ++item) {
                    if (row + item * workers < group_rows) {
                        packed |= oneWindowStagedByte<PacketBytes>(warp_words, subgroup_lane_base,
                                                                   byte + item * row_stride)
                                  << (item * 8U);
                        ++valid_values;
                    }
                }
                local[slot] = reduceOneWindowDecoded4<FixedOp>(
                    local[slot], decodeOneWindowRuntimeKPacked<InputT, FixedOp>(packed, input_transform, extremum_fold),
                    valid_values);
            }
        }
    } else {
        static_cast<void>(component);
        static_cast<void>(worker);
        static_cast<void>(workers);
        const Fp8TypedReductionOp<FixedOp> combine{};
        for (uint32_t row = 0; row < group_rows; ++row) {
            const uint32_t row_byte = prefix_bytes + row * k + subgroup_lane;
#pragma unroll
            for (uint32_t slot_base = 0; slot_base < static_cast<uint32_t>(MaxComponentsPerLane); slot_base += 4U) {
                if (subgroup_lane + slot_base * LanesPerOutput >= k) {
                    continue;
                }
                uint32_t packed = 0;
                uint32_t valid_slots = 0;
#pragma unroll
                for (uint32_t j = 0; j < 4U; ++j) {
                    if (subgroup_lane + (slot_base + j) * LanesPerOutput < k) {
                        packed |= oneWindowStagedByte<PacketBytes>(warp_words, subgroup_lane_base,
                                                      row_byte + (slot_base + j) * LanesPerOutput)
                                  << (j * 8U);
                        ++valid_slots;
                    }
                }
                const float4 decoded =
                    decodeOneWindowRuntimeKPacked<InputT, FixedOp>(packed, input_transform, extremum_fold);
                // Valid slots are a prefix: owned components increase with j.
                local[slot_base] = combine(local[slot_base], decoded.x);
                if (valid_slots > 1) local[slot_base + 1] = combine(local[slot_base + 1], decoded.y);
                if (valid_slots > 2) local[slot_base + 2] = combine(local[slot_base + 2], decoded.z);
                if (valid_slots > 3) local[slot_base + 3] = combine(local[slot_base + 3], decoded.w);
            }
        }
    }
}

// Streams one output's R through packet windows first_row, first_row + row_step, ... and decodes them into local[].
template <typename InputT, int FixedOp, int MaxComponentsPerLane, uint32_t LanesPerOutput, uint32_t PacketBytes,
          bool LoadAhead>
__device__ __forceinline__ void streamOneWindowRuntimeKOutput(float (&local)[MaxComponentsPerLane],
                                                              uint32_t* warp_words,
                                                              const uint8_t* input_bytes,
                                                              uint32_t total_input_bytes,
                                                              bool valid_output,
                                                              uint32_t outer_byte_begin,
                                                              uint32_t first_row,
                                                              uint32_t row_step,
                                                              uint32_t reduction_size,
                                                              const Fp8OneWindowRuntimeKGeometry& geometry,
                                                              uint32_t lane,
                                                              uint32_t subgroup_lane,
                                                              uint32_t subgroup_lane_base,
                                                              uint32_t component,
                                                              uint32_t worker,
                                                              CubReductionStageInputTransform input_transform,
                                                              const Fp8ExtremumFold& extremum_fold) {
    const bool decodes = valid_output && subgroup_lane < geometry.task_count;
    const auto window_at = [&](uint32_t group_row) {
        return oneWindowRuntimeKGroupWindow<PacketBytes>(valid_output, outer_byte_begin, group_row, reduction_size,
                                                         geometry.k, geometry.rows_per_group);
    };
    if constexpr (LoadAhead) {
        uint32_t group_row = first_row;
        OneWindowGroupWindow current{};
        RawVectorPacket<PacketBytes> current_packet{};
        if (group_row < reduction_size) {
            current = window_at(group_row);
            if (valid_output && subgroup_lane < current.packet_count) {
                current_packet = loadOneWindowPacket<PacketBytes>(
                    input_bytes, current.aligned_byte_begin + subgroup_lane * PacketBytes, total_input_bytes);
            }
        }
        while (group_row < reduction_size) {
            const uint32_t next_row = group_row + row_step;
            OneWindowGroupWindow next{};
            RawVectorPacket<PacketBytes> next_packet{};
            if (next_row < reduction_size) {
                next = window_at(next_row);
                if (valid_output && subgroup_lane < next.packet_count) {
                    next_packet = loadOneWindowPacket<PacketBytes>(
                        input_bytes, next.aligned_byte_begin + subgroup_lane * PacketBytes, total_input_bytes);
                }
            }
            if (valid_output && subgroup_lane < current.packet_count) {
                stageOneWindowPacket<PacketBytes>(warp_words, lane, current_packet);
            }
            __syncwarp();
            if (decodes) {
                decodeOneWindowRuntimeKGroup<InputT, FixedOp, MaxComponentsPerLane, LanesPerOutput, PacketBytes>(
                    local, warp_words, subgroup_lane, subgroup_lane_base, component, worker, geometry.k,
                    geometry.workers_per_component, current.rows, current.prefix_bytes, input_transform,
                    extremum_fold);
            }
            __syncwarp();
            current = next;
            current_packet = next_packet;
            group_row = next_row;
        }
    } else {
        for (uint32_t group_row = first_row; group_row < reduction_size; group_row += row_step) {
            const OneWindowGroupWindow window = window_at(group_row);
            if (valid_output && subgroup_lane < window.packet_count) {
                stageOneWindowPacket<PacketBytes>(
                    warp_words, lane,
                    loadOneWindowPacket<PacketBytes>(
                        input_bytes, window.aligned_byte_begin + subgroup_lane * PacketBytes,
                        total_input_bytes));
            }
            __syncwarp();
            if (decodes) {
                decodeOneWindowRuntimeKGroup<InputT, FixedOp, MaxComponentsPerLane, LanesPerOutput, PacketBytes>(
                    local, warp_words, subgroup_lane, subgroup_lane_base, component, worker, geometry.k,
                    geometry.workers_per_component, window.rows, window.prefix_bytes, input_transform, extremum_fold);
            }
            __syncwarp();
        }
    }
}

// One-component buckets: the R workers of a component sit in consecutive lanes (component * workers + worker).
// P5-K1: a power-of-two worker count (always at O4/O8; at O1/O2 when lanes / K is a power of two, including one
// worker) folds with butterfly shuffles inside the aligned worker group, with no shared memory and no warp barrier;
// every lane of the group ends with the total. Other counts keep the shared lane_partials fold. Warp-uniform: every
// lane calls it with the same workers.
template <int FixedOp, uint32_t LanesPerOutput>
__device__ __forceinline__ float shuffleFoldOneWindowWorkers(float partial, uint32_t workers) {
    const Fp8TypedReductionOp<FixedOp> combine{};
#pragma unroll
    for (uint32_t offset = 1; offset < LanesPerOutput; offset <<= 1) {
        if (offset < workers) {
            partial = combine(partial, __shfl_xor_sync(0xffffffffU, partial, offset));
        }
    }
    return partial;
}

template <typename InputT, int OutputsPerWarp, int PacketBytes, int MaxComponentsPerLane, bool LoadAhead,
          int FixedOp = 0>
__global__ void fp8OneWindowRuntimeKCompleteKernel(const InputT* input,
                                                   void* output,
                                                   DataType output_dtype,
                                                   uint32_t outer_size,
                                                   uint32_t reduction_size,
                                                   uint32_t warps_per_output,
                                                   Fp8OneWindowRuntimeKGeometry geometry,
                                                   AdditiveFinalizeFp32 output_finalize,
                                                   CubReductionStageInputTransform input_transform,
                                                   float output_scale,
                                                   Fp8ExtremumFold extremum_fold) {
    static_assert(sizeof(InputT) == 1);
    static_assert(FixedOp == 0 || FixedOp == 1 || FixedOp == 2);
    static_assert(OutputsPerWarp == 1 || OutputsPerWarp == 2 || OutputsPerWarp == 4 || OutputsPerWarp == 8);
    static_assert(MaxComponentsPerLane == 1 || MaxComponentsPerLane == 2 || MaxComponentsPerLane == 4 ||
                  MaxComponentsPerLane == 8 || MaxComponentsPerLane == 16);
    static_assert(MaxComponentsPerLane > 1 || !LoadAhead, "One-component buckets use the plain group loop.");
    static_assert(PacketBytes == 4 || PacketBytes == 8 || PacketBytes == 16);
    constexpr uint32_t PacketWords = PacketBytes / 4;
    constexpr uint32_t LanesPerOutput = TILED_REDUCTION_WARP_THREADS / OutputsPerWarp;
    constexpr bool MultipleComponentsPerLane = MaxComponentsPerLane > 1;
    constexpr uint32_t MaxK = LanesPerOutput * static_cast<uint32_t>(MaxComponentsPerLane);
    constexpr uint32_t LanePartialColumns = MultipleComponentsPerLane ? 1U : TILED_REDUCTION_WARP_THREADS;
    constexpr uint32_t WarpPartialRows = OutputsPerWarp == 1 ? SHALLOW_FP8_WARPS : 1U;
    constexpr uint32_t WarpPartialColumns = OutputsPerWarp == 1 ? MaxK : 1U;
    const Fp8TypedReductionOp<FixedOp> combine{};

    __shared__ uint32_t staged_packet_words[SHALLOW_FP8_WARPS][PacketWords * TILED_REDUCTION_WARP_THREADS];
    __shared__ float lane_partials[SHALLOW_FP8_WARPS][LanePartialColumns];
    __shared__ float warp_partials[WarpPartialRows][WarpPartialColumns];

    const uint32_t k = geometry.k;
    const uint32_t workers = geometry.workers_per_component;
    const uint32_t warp = threadIdx.x / TILED_REDUCTION_WARP_THREADS;
    const uint32_t lane = threadIdx.x % TILED_REDUCTION_WARP_THREADS;
    const uint32_t outer_input_stride = reduction_size * k;
    const uint32_t total_input_bytes = outer_size * outer_input_stride;
    const uint8_t* const input_bytes = reinterpret_cast<const uint8_t*>(input);

    // O2/O4/O8: each lane subgroup owns one output (warp-local). O1: the subgroup is the whole warp.
    const uint32_t output_slot = lane / LanesPerOutput;
    const uint32_t subgroup_lane = lane % LanesPerOutput;
    const uint32_t subgroup_lane_base = output_slot * LanesPerOutput;
    // One component per lane: component and R worker, one runtime division per thread.
    const uint32_t component = MultipleComponentsPerLane ? subgroup_lane : subgroup_lane / workers;
    const uint32_t worker = MultipleComponentsPerLane ? 0U : subgroup_lane - component * workers;

    float local[MaxComponentsPerLane];

    if constexpr (OutputsPerWarp > 1) {
        static_cast<void>(warps_per_output);
        static_cast<void>(warp_partials);
        const uint32_t global_warp = blockIdx.x * SHALLOW_FP8_WARPS + warp;
        const uint32_t grid_warps = gridDim.x * SHALLOW_FP8_WARPS;
        for (uint32_t warp_group = global_warp;; warp_group += grid_warps) {
            const uint64_t outer_begin = static_cast<uint64_t>(warp_group) * OutputsPerWarp;
            if (outer_begin >= outer_size) {
                return;
            }
            const uint64_t outer_index = outer_begin + output_slot;
            const bool valid_output = outer_index < outer_size;
#pragma unroll
            for (uint32_t slot = 0; slot < static_cast<uint32_t>(MaxComponentsPerLane); ++slot) {
                local[slot] = combine.identity();
            }
            const uint32_t outer_byte_begin =
                valid_output ? static_cast<uint32_t>(outer_index) * outer_input_stride : 0;
            streamOneWindowRuntimeKOutput<InputT, FixedOp, MaxComponentsPerLane, LanesPerOutput, PacketBytes,
                                          LoadAhead>(
                local, staged_packet_words[warp], input_bytes, total_input_bytes, valid_output, outer_byte_begin, 0U,
                geometry.rows_per_group, reduction_size, geometry, lane, subgroup_lane, subgroup_lane_base, component,
                worker, input_transform, extremum_fold);

            if constexpr (MultipleComponentsPerLane) {
#pragma unroll
                for (uint32_t slot = 0; slot < static_cast<uint32_t>(MaxComponentsPerLane); ++slot) {
                    const uint32_t owned_component = subgroup_lane + slot * LanesPerOutput;
                    if (valid_output && owned_component < k) {
                        storeFp8CompleteResult<FixedOp == 0>(
                            output, output_dtype, outer_index * static_cast<uint64_t>(k) + owned_component,
                            unfoldFp8ExtremumResult<FixedOp == 2>(local[slot], extremum_fold), output_finalize,
                            output_scale);
                    }
                }
            } else if ((workers & (workers - 1U)) == 0U) {
                const float reduced = shuffleFoldOneWindowWorkers<FixedOp, LanesPerOutput>(
                    (valid_output && subgroup_lane < geometry.task_count) ? local[0] : combine.identity(), workers);
                if (valid_output && worker == 0U && component < k) {
                    storeFp8CompleteResult<FixedOp == 0>(
                        output, output_dtype, outer_index * static_cast<uint64_t>(k) + component,
                        unfoldFp8ExtremumResult<FixedOp == 2>(reduced, extremum_fold), output_finalize, output_scale);
                }
            } else {
                lane_partials[warp][lane] =
                    (valid_output && subgroup_lane < geometry.task_count) ? local[0] : combine.identity();
                __syncwarp();
                if (valid_output && subgroup_lane < k) {
                    float reduced = combine.identity();
                    const uint32_t partial_base = subgroup_lane_base + subgroup_lane * workers;
                    for (uint32_t source_worker = 0; source_worker < workers; ++source_worker) {
                        reduced = combine(reduced, lane_partials[warp][partial_base + source_worker]);
                    }
                    storeFp8CompleteResult<FixedOp == 0>(
                        output, output_dtype, outer_index * static_cast<uint64_t>(k) + subgroup_lane,
                        unfoldFp8ExtremumResult<FixedOp == 2>(reduced, extremum_fold), output_finalize, output_scale);
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
        const uint32_t rows_per_output_round = geometry.rows_per_group * warps_per_output;
        const uint64_t block_output_stride = static_cast<uint64_t>(gridDim.x) * outputs_per_block;

        for (uint64_t block_outer_begin = static_cast<uint64_t>(blockIdx.x) * outputs_per_block;
             block_outer_begin < outer_size;
             block_outer_begin += block_output_stride) {
            const uint64_t outer_index = block_outer_begin + output_group;
            const bool valid_output = outer_index < outer_size;
#pragma unroll
            for (uint32_t slot = 0; slot < static_cast<uint32_t>(MaxComponentsPerLane); ++slot) {
                local[slot] = combine.identity();
            }
            const uint32_t outer_byte_begin =
                valid_output ? static_cast<uint32_t>(outer_index) * outer_input_stride : 0;
            streamOneWindowRuntimeKOutput<InputT, FixedOp, MaxComponentsPerLane, LanesPerOutput, PacketBytes,
                                          LoadAhead>(
                local, staged_packet_words[warp], input_bytes, total_input_bytes, valid_output, outer_byte_begin,
                warp_in_output * geometry.rows_per_group, rows_per_output_round, reduction_size, geometry, lane,
                subgroup_lane, subgroup_lane_base, component, worker, input_transform, extremum_fold);

            if (warps_per_output == 1) {
                if constexpr (MultipleComponentsPerLane) {
#pragma unroll
                    for (uint32_t slot = 0; slot < static_cast<uint32_t>(MaxComponentsPerLane); ++slot) {
                        const uint32_t owned_component = lane + slot * TILED_REDUCTION_WARP_THREADS;
                        if (valid_output && owned_component < k) {
                            storeFp8CompleteResult<FixedOp == 0>(
                                output, output_dtype, outer_index * static_cast<uint64_t>(k) + owned_component,
                                unfoldFp8ExtremumResult<FixedOp == 2>(local[slot], extremum_fold), output_finalize,
                                output_scale);
                        }
                    }
                } else if ((workers & (workers - 1U)) == 0U) {
                    const float reduced = shuffleFoldOneWindowWorkers<FixedOp, LanesPerOutput>(
                        (valid_output && lane < geometry.task_count) ? local[0] : combine.identity(), workers);
                    if (valid_output && worker == 0U && component < k) {
                        storeFp8CompleteResult<FixedOp == 0>(
                            output, output_dtype, outer_index * static_cast<uint64_t>(k) + component,
                            unfoldFp8ExtremumResult<FixedOp == 2>(reduced, extremum_fold), output_finalize,
                            output_scale);
                    }
                } else {
                    lane_partials[warp][lane] =
                        (valid_output && lane < geometry.task_count) ? local[0] : combine.identity();
                    __syncwarp();
                    if (valid_output && lane < k) {
                        float reduced = combine.identity();
                        const uint32_t partial_base = lane * workers;
                        for (uint32_t source_worker = 0; source_worker < workers; ++source_worker) {
                            reduced = combine(reduced, lane_partials[warp][partial_base + source_worker]);
                        }
                        storeFp8CompleteResult<FixedOp == 0>(
                            output, output_dtype, outer_index * static_cast<uint64_t>(k) + lane,
                            unfoldFp8ExtremumResult<FixedOp == 2>(reduced, extremum_fold), output_finalize,
                            output_scale);
                    }
                    __syncwarp();
                }
                continue;
            }

            if constexpr (MultipleComponentsPerLane) {
#pragma unroll
                for (uint32_t slot = 0; slot < static_cast<uint32_t>(MaxComponentsPerLane); ++slot) {
                    const uint32_t owned_component = lane + slot * TILED_REDUCTION_WARP_THREADS;
                    if (owned_component < k) {
                        warp_partials[warp][owned_component] = valid_output ? local[slot] : combine.identity();
                    }
                }
            } else if ((workers & (workers - 1U)) == 0U) {
                const float warp_reduced = shuffleFoldOneWindowWorkers<FixedOp, LanesPerOutput>(
                    (valid_output && lane < geometry.task_count) ? local[0] : combine.identity(), workers);
                if (worker == 0U && component < k) {
                    warp_partials[warp][component] = valid_output ? warp_reduced : combine.identity();
                }
            } else {
                lane_partials[warp][lane] =
                    (valid_output && lane < geometry.task_count) ? local[0] : combine.identity();
                __syncwarp();
                if (lane < k) {
                    float warp_reduced = combine.identity();
                    const uint32_t partial_base = lane * workers;
                    for (uint32_t source_worker = 0; source_worker < workers; ++source_worker) {
                        warp_reduced = combine(warp_reduced, lane_partials[warp][partial_base + source_worker]);
                    }
                    warp_partials[warp][lane] = valid_output ? warp_reduced : combine.identity();
                }
            }
            __syncthreads();

            if (warp_in_output == 0) {
#pragma unroll
                for (uint32_t slot = 0; slot < static_cast<uint32_t>(MaxComponentsPerLane); ++slot) {
                    const uint32_t owned_component = lane + slot * TILED_REDUCTION_WARP_THREADS;
                    if (valid_output && owned_component < k) {
                        float reduced = combine.identity();
                        for (uint32_t source_warp = 0; source_warp < warps_per_output; ++source_warp) {
                            reduced = combine(reduced,
                                              warp_partials[first_warp_in_output + source_warp][owned_component]);
                        }
                        storeFp8CompleteResult<FixedOp == 0>(
                            output, output_dtype, outer_index * static_cast<uint64_t>(k) + owned_component,
                            unfoldFp8ExtremumResult<FixedOp == 2>(reduced, extremum_fold), output_finalize,
                            output_scale);
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
// P5-H: pinned to four 256-thread blocks per SM (<= 64 registers). SUM compiled at 62 registers before the row
// load-ahead; 65+ would lose a block, and 1077 showed RuntimeK SUM losing up to 1.9x at 80 registers. With the
// load-ahead it compiles at 56 or fewer, no spills, and runs 0.79x (1089).
template <typename InputT, int FixedInputTransform = 0, int FixedOp = 0>
__global__ __launch_bounds__(SHALLOW_FP8_THREADS, 4) void fp8GeneralRParallelKTileP16CompleteKernel(const InputT* input,
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

        // Decodes the staged row (tile_width components after prefix_bytes alignment bytes) into local[].
        const auto decode_row = [&](uint32_t prefix_bytes) {
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
        };
        // Computes a row's alignment prefix and packet count and loads this lane's packet. A p16 load extends at
        // most 15 bytes past the logical tensor end; Tensor/view allocation provides >=128 trailing bytes, and
        // component masks prevent out-of-range values from entering the reduction.
        const auto load_row = [&](uint32_t row_index, uint32_t& prefix_bytes, uint32_t& packet_count,
                                  RawVectorPacket<PacketBytes>& packet) {
            const uint64_t logical_begin = static_cast<uint64_t>(outer_index) * outer_input_stride +
                                           static_cast<uint64_t>(row_index) * inner_size + tile_begin;
            prefix_bytes = static_cast<uint32_t>(logical_begin & PacketMask);
            packet_count = (prefix_bytes + tile_width + PacketBytes - 1) >> PacketShift;
            if (lane < packet_count) {
                packet = *reinterpret_cast<const RawVectorPacket<PacketBytes>*>(
                    reinterpret_cast<const uint8_t*>(input) + (logical_begin - prefix_bytes) +
                    static_cast<uint64_t>(lane) * PacketBytes);
            }
        };
        // P5-H load-ahead: the next row's packet is loaded into registers before the current row is staged and decoded,
        // so one global load per lane is always in flight while the warp decodes. Row order is unchanged.
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


template <typename KernelT>
[[nodiscard]] CubKernelOccupancyInfo queryShallowFp8Occupancy(KernelT kernel) {
    cudaFuncAttributes attributes{};
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
    int active_blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks, kernel, SHALLOW_FP8_THREADS, 0));
    return CubKernelOccupancyInfo{attributes.numRegs, attributes.sharedSizeBytes, 0, active_blocks, active_blocks * SHALLOW_FP8_WARPS};
}


template <typename InputT>
struct NarrowFp8TypedFormatDecoder {
    __device__ float operator()(uint8_t bits) const {
        return ToFp32<InputT>{}(narrowFp8FromBits<InputT>(bits));
    }
};

// P5-E: DeepRSmallK warp-group window. Each warp stages one group of whole rows (group_rows * K bytes plus an
// alignment prefix < PacketBytes) as one packet per lane. The group loops below are software-pipelined: the packet of
// the warp's next group is loaded into a register before the current group is decoded from shared memory, so one
// group's global load is always in flight while the warp works. Row order per component is unchanged.
template <int PacketBytes>
struct NarrowFp8GroupWindow {
    const RawVectorPacket<PacketBytes>* aligned_begin;
    uint32_t prefix_elements;
    uint32_t group_rows;
    uint32_t packet_count;
};

template <int PacketBytes, typename InputT>
[[nodiscard]] __device__ __forceinline__ NarrowFp8GroupWindow<PacketBytes> narrowFp8GroupWindow(
    const InputT* shard_input, uint32_t group_row, uint32_t shard_rows, uint32_t rows_per_warp_group,
    uint32_t inner_size_32) {
    static_assert(sizeof(InputT) == 1);
    NarrowFp8GroupWindow<PacketBytes> window;
    window.group_rows = rows_per_warp_group < shard_rows - group_row ? rows_per_warp_group : shard_rows - group_row;
    const uintptr_t group_address = reinterpret_cast<uintptr_t>(shard_input + group_row * inner_size_32);
    window.prefix_elements = static_cast<uint32_t>(group_address & uintptr_t{PacketBytes - 1});
    window.aligned_begin =
        reinterpret_cast<const RawVectorPacket<PacketBytes>*>(group_address & ~uintptr_t{PacketBytes - 1});
    window.packet_count =
        (window.prefix_elements + window.group_rows * inner_size_32 + PacketBytes - 1) / PacketBytes;
    return window;
}

// P5-C: the shard depth is a runtime operand. It only places and clamps each shard; the row loops already ran on the
// runtime rows_per_warp_group / shard_rows, and the five compiled depths produced identical SASS apart from that
// constant (sass-1069). Retained depths are powers of two and the kernel receives log2(depth): a runtime 64-bit
// multiply for the shard origin (1079) cost 2-4% on shards of 4 KB or less, where the per-shard latency before the
// first load dominates; a shift keeps the compiled kernels' instruction shape. The launcher validates the depth
// against Fp8DeepRSmallKPolicy::compiled_rows.
// FixedRowsPerShardLog2 >= 0 compiles the depth back in (rows_per_shard_log2 is then ignored). Only the one-component
// kernel (K 9..32) uses it, at every retained depth: with the runtime depth it lost 4.8% at rows64 and 1.9% at
// rows128 (1080), the only ownership whose loss survived the shift. Every other ownership kind keeps one kernel.
template <typename InputT, typename ReductionOpT, typename InputTransformT, int PacketBytes, int MaxComponentsPerLane,
          int FixedRowsPerShardLog2 = -1, typename DecoderT = NarrowFp8TypedFormatDecoder<InputT>>
__global__ void narrowFp8PackedFlatRCooperativeFirstStageKernel(const InputT* input,
                                                                float* fp32_partials,
                                                                uint64_t outer_size,
                                                                uint64_t reduction_size,
                                                                uint64_t inner_size,
                                                                uint64_t shards_per_output,
                                                                uint32_t rows_per_shard_log2,
                                                                uint32_t rows_per_warp_group,
                                                                InputTransformT input_transform, DecoderT decoder = DecoderT{}) {
    static_assert(sizeof(InputT) == 1);
    static_assert(PacketBytes == 4 || PacketBytes == 8 || PacketBytes == 16);
    static_assert(MaxComponentsPerLane == 1 || MaxComponentsPerLane == 2 || MaxComponentsPerLane == 4 || MaxComponentsPerLane == 5 ||
                  MaxComponentsPerLane == 8);
    constexpr int MaxSupportedK = TILED_REDUCTION_WARP_THREADS * MaxComponentsPerLane;
    constexpr float Init = NarrowFlatReductionIdentityFp32<ReductionOpT>::value;
    static_assert(MaxSupportedK <= NARROW_FP8_CALIBRATION_MAX_K);
    static_assert(FixedRowsPerShardLog2 == -1 || (FixedRowsPerShardLog2 >= 3 && FixedRowsPerShardLog2 <= 14));
    const uint32_t shard_log2 =
        FixedRowsPerShardLog2 >= 0 ? static_cast<uint32_t>(FixedRowsPerShardLog2) : rows_per_shard_log2;

    __shared__ RawVectorPacket<PacketBytes> staged_packets[NARROW_FLAT_WARPS][NARROW_FP8_PACKETS_PER_WARP];
    __shared__ float warp_partials[NARROW_FLAT_WARPS][MaxSupportedK];

    const ReductionOpT reduction_op{};
    const int warp = static_cast<int>(threadIdx.x) / TILED_REDUCTION_WARP_THREADS;
    const int lane = static_cast<int>(threadIdx.x) % TILED_REDUCTION_WARP_THREADS;

    // DeepRSmallK's retained launcher guarantees 1 <= inner_size <= 256, and rows_per_shard <= 16384, so every offset inside
    // one shard (< 2^22 bytes) is 32-bit. One warp stages at most 32 * PacketBytes <= 512 bytes, including an alignment
    // prefix < PacketBytes. The shard base pointer is formed once per work item in 64 bits.
    const uint32_t inner_size_32 = static_cast<uint32_t>(inner_size);

    // rows_per_warp_group is computed by the launcher: fill toward the packet window, but never choose a group so deep
    // that fewer than eight warps can participate in a full rows_per_shard shard.
    const uint32_t rows_per_block_round = rows_per_warp_group * NARROW_FLAT_WARPS;

    // Grid: x = shard, z = outer (see reductionWorkGrid); this family has one component tile.
    // reductionWorkGrid() limits every work extent to 2^31 - 1, so 32-bit work coordinates never wrap.
    const uint32_t outer_count = static_cast<uint32_t>(outer_size);
    const uint32_t shard_count = static_cast<uint32_t>(shards_per_output);
    for (uint32_t outer_index = blockIdx.z; outer_index < outer_count; outer_index += gridDim.z)
    for (uint32_t shard = blockIdx.x; shard < shard_count; shard += gridDim.x) {
        const uint64_t row_begin = static_cast<uint64_t>(shard) << shard_log2;
        const uint32_t shard_rows = row_begin < reduction_size
            ? static_cast<uint32_t>(minU64(uint64_t{1} << shard_log2, reduction_size - row_begin))
            : 0U;
        const InputT* shard_input = input + (outer_index * reduction_size + row_begin) * inner_size;

        float local[MaxComponentsPerLane];
        for (int slot = 0; slot < MaxComponentsPerLane; ++slot) {
            local[slot] = Init;
        }

        // Prefetch the next group only when lanes own two or more components (K > 32). Measured on RTX 5090 against
        // the plain loop: 0.98-0.89x time for 2..8 components per lane, but 1.01-1.05x for one component per lane, whose
        // short per-group decode leaves too little work to hide the next load behind.
        if constexpr (MaxComponentsPerLane >= 2) {
            uint32_t group_row = static_cast<uint32_t>(warp) * rows_per_warp_group;
            NarrowFp8GroupWindow<PacketBytes> window{};
            RawVectorPacket<PacketBytes> pending{};
            if (group_row < shard_rows) {
                window = narrowFp8GroupWindow<PacketBytes>(shard_input, group_row, shard_rows, rows_per_warp_group,
                                                           inner_size_32);
                if (static_cast<uint32_t>(lane) < window.packet_count) {
                    pending = window.aligned_begin[lane];
                }
            }
            for (; group_row < shard_rows; group_row += rows_per_block_round) {
                const uint32_t group_rows = window.group_rows;
                const uint32_t prefix_elements = window.prefix_elements;
                if (static_cast<uint32_t>(lane) < window.packet_count) {
                    staged_packets[warp][lane] = pending;
                }
                __syncwarp();

                // Issue the next group's load before decoding this one.
                const uint32_t next_group_row = group_row + rows_per_block_round;
                if (next_group_row < shard_rows) {
                    window = narrowFp8GroupWindow<PacketBytes>(shard_input, next_group_row, shard_rows,
                                                               rows_per_warp_group, inner_size_32);
                    if (static_cast<uint32_t>(lane) < window.packet_count) {
                        pending = window.aligned_begin[lane];
                    }
                }

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
        } else {
            for (uint32_t group_row = static_cast<uint32_t>(warp) * rows_per_warp_group; group_row < shard_rows;
                 group_row += rows_per_block_round) {
                const uint32_t group_rows =
                    rows_per_warp_group < shard_rows - group_row ? rows_per_warp_group : shard_rows - group_row;
                const InputT* group_begin = shard_input + group_row * inner_size_32;
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

// P5-C: runtime log2(rows per shard), as in narrowFp8PackedFlatRCooperativeFirstStageKernel.
template <typename InputT, typename ReductionOpT, typename InputTransformT, int PacketBytes, int LanesPerComponent,
          typename DecoderT = NarrowFp8TypedFormatDecoder<InputT>>
__global__ void narrowFp8PackedFlatRCooperativeTinyKFirstStageKernel(const InputT* input,
                                                                     float* fp32_partials,
                                                                     uint64_t outer_size,
                                                                     uint64_t reduction_size,
                                                                     uint64_t inner_size,
                                                                     uint64_t shards_per_output,
                                                                     uint32_t rows_per_shard_log2,
                                                                     uint32_t rows_per_warp_group,
                                                                InputTransformT input_transform, DecoderT decoder = DecoderT{}) {
    static_assert(sizeof(InputT) == 1);
    static_assert(PacketBytes == 4 || PacketBytes == 8 || PacketBytes == 16);
    static_assert(LanesPerComponent == 4 || LanesPerComponent == 8);
    static_assert(TILED_REDUCTION_WARP_THREADS % LanesPerComponent == 0);
    constexpr int MaxSupportedK = TILED_REDUCTION_WARP_THREADS / LanesPerComponent;
    constexpr float Init = NarrowFlatReductionIdentityFp32<ReductionOpT>::value;
    static_assert(MaxSupportedK == 4 || MaxSupportedK == 8);

    __shared__ RawVectorPacket<PacketBytes> staged_packets[NARROW_FLAT_WARPS][NARROW_FP8_PACKETS_PER_WARP];
    __shared__ float cooperative_lane_partials[NARROW_FLAT_WARPS][TILED_REDUCTION_WARP_THREADS];
    __shared__ float warp_partials[NARROW_FLAT_WARPS][MaxSupportedK];

    const ReductionOpT reduction_op{};
    const int warp = static_cast<int>(threadIdx.x) / TILED_REDUCTION_WARP_THREADS;
    const int lane = static_cast<int>(threadIdx.x) % TILED_REDUCTION_WARP_THREADS;
    const int component = lane / LanesPerComponent;
    const int component_lane = lane % LanesPerComponent;

    // inner_size <= 8 here and rows_per_shard <= 16384, so every offset inside one shard is 32-bit. One warp stages at most
    // 32 * PacketBytes <= 512 bytes, including an alignment prefix < PacketBytes. The shard base pointer is formed once
    // per work item in 64 bits.
    const uint32_t inner_size_32 = static_cast<uint32_t>(inner_size);

    // rows_per_warp_group is computed by the launcher (see narrowFp8PackedRowsPerWarpGroup).
    const uint32_t rows_per_block_round = rows_per_warp_group * NARROW_FLAT_WARPS;

    // Grid: x = shard, z = outer (see reductionWorkGrid); this family has one component tile.
    // reductionWorkGrid() limits every work extent to 2^31 - 1, so 32-bit work coordinates never wrap.
    const uint32_t outer_count = static_cast<uint32_t>(outer_size);
    const uint32_t shard_count = static_cast<uint32_t>(shards_per_output);
    for (uint32_t outer_index = blockIdx.z; outer_index < outer_count; outer_index += gridDim.z)
    for (uint32_t shard = blockIdx.x; shard < shard_count; shard += gridDim.x) {
        const uint64_t row_begin = static_cast<uint64_t>(shard) << rows_per_shard_log2;
        const uint32_t shard_rows = row_begin < reduction_size
            ? static_cast<uint32_t>(minU64(uint64_t{1} << rows_per_shard_log2, reduction_size - row_begin))
            : 0U;
        const InputT* shard_input = input + (outer_index * reduction_size + row_begin) * inner_size;

        float local = Init;
        for (uint32_t group_row = static_cast<uint32_t>(warp) * rows_per_warp_group; group_row < shard_rows;
             group_row += rows_per_block_round) {
            const uint32_t group_rows =
                rows_per_warp_group < shard_rows - group_row ? rows_per_warp_group : shard_rows - group_row;
            const InputT* group_begin = shard_input + group_row * inner_size_32;
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

// Packed DeepRSmallK shard depths: a power of two from the eight warps up to 16384 rows (32-bit in-shard offsets).
[[nodiscard]] uint32_t narrowFp8PackedCheckedRowsPerShard(uint64_t rows_per_shard) {
    if (rows_per_shard < static_cast<uint64_t>(NARROW_FLAT_WARPS) || rows_per_shard > 16384 ||
        (rows_per_shard & (rows_per_shard - 1)) != 0) {
        throw std::logic_error("Packed FP8 narrow-flat launch received an unsupported rows-per-shard depth.");
    }
    return static_cast<uint32_t>(rows_per_shard);
}

[[nodiscard]] uint32_t narrowFp8PackedRowsPerShardLog2(uint32_t rows_per_shard) {
    uint32_t log2 = 0;
    while ((uint32_t{1} << log2) < rows_per_shard) ++log2;
    return log2;
}

// Rows per warp group for the packed DeepRSmallK kernels, computed once on the host: fill toward the packet window, but
// never choose a group so deep that fewer than eight warps can participate in a full rows_per_shard shard.
template <int PacketBytes>
[[nodiscard]] uint32_t narrowFp8PackedRowsPerWarpGroup(uint64_t inner_size, uint32_t rows_per_shard) {
    const uint32_t max_rows_per_warp_from_shard = rows_per_shard / static_cast<uint32_t>(NARROW_FLAT_WARPS);
    if (inner_size == 0 || inner_size > static_cast<uint64_t>(narrowFp8MaxLogicalGroupBytes<PacketBytes>)) {
        throw std::logic_error("Packed FP8 narrow-flat launch received K outside one packet window.");
    }
    const uint32_t rows_from_packet_window =
        static_cast<uint32_t>(narrowFp8MaxLogicalGroupBytes<PacketBytes>) / static_cast<uint32_t>(inner_size);
    return rows_from_packet_window < max_rows_per_warp_from_shard ? rows_from_packet_window : max_rows_per_warp_from_shard;
}

template <typename InputT, typename ReductionOpT, typename InputTransformT, int PacketBytes, int LanesPerComponent,
          typename DecoderT = NarrowFp8TypedFormatDecoder<InputT>>
void launchNarrowFp8PackedTinyKForTransform(
    const Tensor& input, Tensor& fp32_partials, const CubReductionGeometry& geometry, uint64_t rows_per_shard,
    uint64_t shards_per_output, cudaStream_t stream,
    InputTransformT input_transform = InputTransformT{}, DecoderT decoder = DecoderT{}) {
    static_assert(sizeof(InputT) == 1);
    static_assert(PacketBytes == 4 || PacketBytes == 8 || PacketBytes == 16);
    static_assert(LanesPerComponent == 4 || LanesPerComponent == 8);
    constexpr uint64_t MaxSupportedK = static_cast<uint64_t>(TILED_REDUCTION_WARP_THREADS / LanesPerComponent);
    if (geometry.inner_size == 0 || geometry.inner_size > MaxSupportedK) {
        throw std::logic_error("Cooperative tiny-K packed FP8 narrow-flat launch received K outside its specialization.");
    }
    if (geometry.outer_size == 0 || shards_per_output == 0) {
        throw std::logic_error("Cooperative tiny-K packed FP8 narrow-flat first stage requires non-zero work.");
    }
    // Division-free launch: x = shard, z = outer. Shards of <= 256 bytes keep a persistent grid
    // (see REDUCTION_PERSISTENT_MAX_BLOCKS).
    const uint32_t shard_depth = narrowFp8PackedCheckedRowsPerShard(rows_per_shard);
    const uint64_t bytes_per_work_item =
        std::min<uint64_t>(shard_depth, geometry.reduction_size) * geometry.inner_size * sizeof(InputT);
    const dim3 grid_blocks = reductionWorkGrid(shards_per_output, 1, geometry.outer_size,
                                               bytes_per_work_item <= 256 ? REDUCTION_PERSISTENT_MAX_BLOCKS : 0);
    const uint32_t rows_per_warp_group =
        narrowFp8PackedRowsPerWarpGroup<PacketBytes>(geometry.inner_size, shard_depth);

    narrowFp8PackedFlatRCooperativeTinyKFirstStageKernel<InputT,
                                                         ReductionOpT,
                                                         InputTransformT,
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
                                                          narrowFp8PackedRowsPerShardLog2(shard_depth),
                                                          rows_per_warp_group,
                                                          input_transform, decoder);
    CUDA_CHECK(cudaGetLastError());
}

template <typename InputT, typename ReductionOpT, typename InputTransformT, int PacketBytes, int MaxComponentsPerLane,
          int FixedRowsPerShardLog2 = -1, typename DecoderT = NarrowFp8TypedFormatDecoder<InputT>>
void launchNarrowFp8PackedForTransform(
    const Tensor& input, Tensor& fp32_partials, const CubReductionGeometry& geometry, uint64_t rows_per_shard,
    uint64_t shards_per_output, cudaStream_t stream,
    InputTransformT input_transform = InputTransformT{}, DecoderT decoder = DecoderT{}) {
    static_assert(sizeof(InputT) == 1);
    static_assert(PacketBytes == 4 || PacketBytes == 8 || PacketBytes == 16);
    constexpr uint64_t MaxSupportedK = static_cast<uint64_t>(TILED_REDUCTION_WARP_THREADS * MaxComponentsPerLane);
    if (geometry.inner_size == 0 || geometry.inner_size > MaxSupportedK) {
        throw std::logic_error("Packed FP8 narrow-flat launch received K outside its component-per-lane specialization.");
    }
    if (geometry.outer_size == 0 || shards_per_output == 0) {
        throw std::logic_error("Packed FP8 narrow-flat first stage requires non-zero work.");
    }
    // Division-free launch: x = shard, z = outer. Shards of <= 256 bytes keep a persistent grid
    // (see REDUCTION_PERSISTENT_MAX_BLOCKS).
    const uint32_t shard_depth = narrowFp8PackedCheckedRowsPerShard(rows_per_shard);
    const uint64_t bytes_per_work_item =
        std::min<uint64_t>(shard_depth, geometry.reduction_size) * geometry.inner_size * sizeof(InputT);
    const dim3 grid_blocks = reductionWorkGrid(shards_per_output, 1, geometry.outer_size,
                                               bytes_per_work_item <= 256 ? REDUCTION_PERSISTENT_MAX_BLOCKS : 0);
    const uint32_t rows_per_warp_group =
        narrowFp8PackedRowsPerWarpGroup<PacketBytes>(geometry.inner_size, shard_depth);
    const uint32_t shard_log2 = narrowFp8PackedRowsPerShardLog2(shard_depth);
    if constexpr (FixedRowsPerShardLog2 >= 0) {
        if (shard_log2 != static_cast<uint32_t>(FixedRowsPerShardLog2)) {
            throw std::logic_error("Packed FP8 narrow-flat launch depth does not match its compiled depth.");
        }
    }

    narrowFp8PackedFlatRCooperativeFirstStageKernel<InputT, ReductionOpT, InputTransformT, PacketBytes, MaxComponentsPerLane,
                                                    FixedRowsPerShardLog2, DecoderT>
        <<<grid_blocks, NARROW_FLAT_THREADS, 0, stream>>>(([](const Tensor& src) -> const InputT* {
            if constexpr (std::is_same_v<InputT, uint8_t>) return reinterpret_cast<const InputT*>(src.getMemPtr<void>());
            else return src.getMemPtr<InputT>();
        })(input),
                                                          fp32_partials.getMemPtr<float>(),
                                                          geometry.outer_size,
                                                          geometry.reduction_size,
                                                          geometry.inner_size,
                                                          shards_per_output,
                                                          shard_log2,
                                                          rows_per_warp_group,
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

template <typename InputT, int PacketBytes, int LanesPerComponent, typename InputTransformT = IdentityFp32,
          typename DecoderT = NarrowFp8TypedFormatDecoder<InputT>, typename ReductionOpT = cuda::std::plus<float>>
[[nodiscard]] CubKernelOccupancyInfo queryNarrowFp8PackedTinyKOccupancy() {
    static_assert(sizeof(InputT) == 1);
    static_assert(PacketBytes == 4 || PacketBytes == 8 || PacketBytes == 16);
    static_assert(LanesPerComponent == 4 || LanesPerComponent == 8);
    const auto kernel = narrowFp8PackedFlatRCooperativeTinyKFirstStageKernel<InputT,
                                                                             ReductionOpT,
                                                                             InputTransformT,
                                                                             PacketBytes,
                                                                             LanesPerComponent,
                                                                         DecoderT>;
    cudaFuncAttributes attributes{};
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
    int active_blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks, kernel, NARROW_FLAT_THREADS, 0));
    return CubKernelOccupancyInfo{attributes.numRegs, attributes.sharedSizeBytes, 0, active_blocks, active_blocks * NARROW_FLAT_WARPS};
}

template <typename InputT, int PacketBytes, int MaxComponentsPerLane, typename InputTransformT = IdentityFp32,
          typename DecoderT = NarrowFp8TypedFormatDecoder<InputT>, typename ReductionOpT = cuda::std::plus<float>,
          int FixedRowsPerShardLog2 = -1>
[[nodiscard]] CubKernelOccupancyInfo queryNarrowFp8PackedOccupancy() {
    static_assert(sizeof(InputT) == 1);
    const auto kernel = narrowFp8PackedFlatRCooperativeFirstStageKernel<InputT,
                                                                        ReductionOpT,
                                                                        InputTransformT,
                                                                        PacketBytes,
                                                                        MaxComponentsPerLane,
                                                                        FixedRowsPerShardLog2,
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

// Retained DeepRSmallK P16 geometry helpers: five retained depths (a runtime operand since P5-C, validated
// against Fp8DeepRSmallKPolicy::compiled_rows, and compiled in for the one-component ownership only) and seven K
// ownership choices, shared by all four typed FP32 reduction operations.
#if THOR_CUB_ENABLE_FP8_TYPES
[[nodiscard]] uint32_t deepRRetainedRowDepth(uint64_t rows) {
    const auto& depths = Fp8RK::Fp8DeepRSmallKPolicy::compiled_rows;
    if (std::find(depths.begin(), depths.end(), rows) == depths.end()) {
        throw std::logic_error("DeepRSmallK retained kernel requires a retained row depth.");
    }
    return static_cast<uint32_t>(rows);
}

// One-component kernels keep a compiled depth (see narrowFp8PackedFlatRCooperativeFirstStageKernel).
template <typename Callback>
void dispatchDeepRRetainedFixedDepth(uint32_t depth, Callback&& callback) {
    switch (depth) {
        case 64: callback.template operator()<6>(); return;
        case 128: callback.template operator()<7>(); return;
        case 256: callback.template operator()<8>(); return;
        case 512: callback.template operator()<9>(); return;
        case 1024: callback.template operator()<10>(); return;
    }
    throw std::logic_error("DeepRSmallK one-component kernel requires a retained row depth.");
}

template <typename InputT, typename TransformT, typename ReductionOpT = cuda::std::plus<float>>
void dispatchDeepRRetainedRows(const Tensor& input, Tensor& partials,
                                  const CubReductionGeometry& geometry, uint64_t rows,
                                  uint64_t shards, cudaStream_t stream, TransformT transform) {
    const uint32_t depth = deepRRetainedRowDepth(rows);
    if (geometry.inner_size <= 4)
        launchNarrowFp8PackedTinyKForTransform<InputT, ReductionOpT, TransformT, 16, 8>(
            input, partials, geometry, depth, shards, stream, transform);
    else if (geometry.inner_size <= 8)
        launchNarrowFp8PackedTinyKForTransform<InputT, ReductionOpT, TransformT, 16, 4>(
            input, partials, geometry, depth, shards, stream, transform);
    else if (geometry.inner_size <= 32)
        dispatchDeepRRetainedFixedDepth(depth, [&]<int DepthLog2>() {
            launchNarrowFp8PackedForTransform<InputT, ReductionOpT, TransformT, 16, 1, DepthLog2>(
                input, partials, geometry, depth, shards, stream, transform);
        });
    else if (geometry.inner_size <= 64)
        launchNarrowFp8PackedForTransform<InputT, ReductionOpT, TransformT, 16, 2>(
            input, partials, geometry, depth, shards, stream, transform);
    else if (geometry.inner_size <= 128)
        launchNarrowFp8PackedForTransform<InputT, ReductionOpT, TransformT, 16, 4>(
            input, partials, geometry, depth, shards, stream, transform);
    else if (geometry.inner_size <= 160)
        launchNarrowFp8PackedForTransform<InputT, ReductionOpT, TransformT, 16, 5>(
            input, partials, geometry, depth, shards, stream, transform);
    else
        launchNarrowFp8PackedForTransform<InputT, ReductionOpT, TransformT, 16, 8>(
            input, partials, geometry, depth, shards, stream, transform);
}

// The row depth selects a kernel only for the one-component (K 9..32) ownership; it is always validated.
template <typename InputT, typename TransformT, typename ReductionOpT = cuda::std::plus<float>>
CubKernelOccupancyInfo queryDeepRRetainedRows(uint64_t rows, uint64_t k) {
    static_cast<void>(deepRRetainedRowDepth(rows));
    if (k <= 4) return queryNarrowFp8PackedTinyKOccupancy<InputT, 16, 8, TransformT, NarrowFp8TypedFormatDecoder<InputT>, ReductionOpT>();
    if (k <= 8) return queryNarrowFp8PackedTinyKOccupancy<InputT, 16, 4, TransformT, NarrowFp8TypedFormatDecoder<InputT>, ReductionOpT>();
    if (k <= 32) {
        CubKernelOccupancyInfo info{};
        dispatchDeepRRetainedFixedDepth(deepRRetainedRowDepth(rows), [&]<int DepthLog2>() {
            info = queryNarrowFp8PackedOccupancy<InputT, 16, 1, TransformT, NarrowFp8TypedFormatDecoder<InputT>,
                                                 ReductionOpT, DepthLog2>();
        });
        return info;
    }
    if (k <= 64) return queryNarrowFp8PackedOccupancy<InputT, 16, 2, TransformT, NarrowFp8TypedFormatDecoder<InputT>, ReductionOpT>();
    if (k <= 128) return queryNarrowFp8PackedOccupancy<InputT, 16, 4, TransformT, NarrowFp8TypedFormatDecoder<InputT>, ReductionOpT>();
    if (k <= 160) return queryNarrowFp8PackedOccupancy<InputT, 16, 5, TransformT, NarrowFp8TypedFormatDecoder<InputT>, ReductionOpT>();
    return queryNarrowFp8PackedOccupancy<InputT, 16, 8, TransformT, NarrowFp8TypedFormatDecoder<InputT>, ReductionOpT>();
}

// Retained DeepRSmallK P16 kernels. The only compile-time axes are dtype,
// input transform, reduction combine and the existing K ownership; depth is a runtime operand.
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

#endif

[[nodiscard]] inline bool oneWindowValidWarpsPerOutput(uint32_t outputs_per_warp, uint32_t warps_per_output) {
    const bool valid_warps =
        warps_per_output == 1 || warps_per_output == 2 || warps_per_output == 4 || warps_per_output == 8;
    // O2/O4/O8 partition one warp among outputs and therefore require W1.
    return valid_warps && (outputs_per_warp == 1 || warps_per_output == 1);
}

// OneWindow host plan. Mirrors the former compile-time-K kernel's constexpr geometry for a runtime K: the packet
// window of a lane subgroup, rows per staged group, R workers per component, accumulating lanes, and the
// components-per-lane bucket (next power of two, at most 16).
struct OneWindowRuntimeKPlan {
    Fp8OneWindowRuntimeKGeometry geometry;
    uint32_t max_components_per_lane;
};

[[nodiscard]] inline std::optional<OneWindowRuntimeKPlan> oneWindowRuntimeKPlan(uint64_t k,
                                                                                uint32_t outputs_per_warp,
                                                                                uint32_t packet_bytes) {
    if (k == 0 || (outputs_per_warp != 1 && outputs_per_warp != 2 && outputs_per_warp != 4 && outputs_per_warp != 8) ||
        (packet_bytes != 4 && packet_bytes != 8 && packet_bytes != 16)) {
        return std::nullopt;
    }
    const uint32_t lanes = TILED_REDUCTION_WARP_THREADS / outputs_per_warp;
    const uint64_t window = static_cast<uint64_t>(lanes) * packet_bytes;
    const uint64_t capacity = k % packet_bytes == 0 ? window : window - (packet_bytes - 1);
    if (k > capacity) {
        return std::nullopt;
    }
    const bool multiple = k > lanes;
    const uint64_t components = multiple ? (k + lanes - 1) / lanes : 1;
    uint32_t bucket = 1;
    while (bucket < components) {
        bucket *= 2;
    }
    if (bucket > 16) {
        return std::nullopt;
    }
    const uint32_t k32 = static_cast<uint32_t>(k);
    const uint32_t workers = multiple ? 1U : lanes / k32;
    return OneWindowRuntimeKPlan{{k32, static_cast<uint32_t>(capacity / k), workers, multiple ? lanes : k32 * workers},
                                 bucket};
}

// The OneWindow inventory (P5-J2): 15 compiled units, each (outputs per warp, packet bytes, bucket) with exactly one
// group loop form, so a selector OneWindow variant (O, W, P16) maps to at most one kernel at a given K. Chosen by
// oracle pruning over the 1094 focused and 1095 uncovered-K sweeps with no compiled-K OneWindow present; 1097
// measured the 8-byte packet (every ownership x bucket 1-8, both loops) at a geometric mean of 1.0005 and dropped it.
// Every K <= 497, and K = 512, has a unit (oneWindowCoversK).
inline constexpr std::array<Fp8OneWindowRuntimeKUnit, 15> ONE_WINDOW_RUNTIME_K_UNITS{{
    {1, 16, 1, false}, {1, 16, 2, true}, {1, 16, 4, false}, {1, 16, 8, true}, {1, 16, 16, true},
    {2, 16, 1, false}, {2, 16, 2, false}, {2, 16, 4, false}, {2, 16, 8, true},
    {4, 16, 2, false}, {4, 16, 4, false}, {4, 16, 8, true},
    {8, 16, 1, false}, {8, 16, 2, false}, {8, 16, 4, false},
}};

[[nodiscard]] constexpr bool oneWindowRuntimeKUnitCompiled(const Fp8OneWindowRuntimeKUnit& unit) {
    for (const Fp8OneWindowRuntimeKUnit& compiled : ONE_WINDOW_RUNTIME_K_UNITS) {
        if (compiled == unit) {
            return true;
        }
    }
    return false;
}

// Invokes callback.template operator()<InputT, OutputsPerWarp, PacketBytes, MaxComponentsPerLane, LoadAhead>() for
// the compiled unit equal to `unit`. Returns false when that unit is not compiled.
template <typename Callback>
[[nodiscard]] bool dispatchOneWindowRuntimeK(DataType input_dtype,
                                             const Fp8OneWindowRuntimeKUnit& unit,
                                             Callback&& callback) {
    bool dispatched = false;
    const auto for_type = [&]<typename InputT>() {
        [&]<size_t... I>(std::index_sequence<I...>) {
            static_cast<void>(((ONE_WINDOW_RUNTIME_K_UNITS[I] == unit &&
                                (callback.template operator()<
                                     InputT,
                                     static_cast<int>(ONE_WINDOW_RUNTIME_K_UNITS[I].outputs_per_warp),
                                     static_cast<int>(ONE_WINDOW_RUNTIME_K_UNITS[I].packet_bytes),
                                     static_cast<int>(ONE_WINDOW_RUNTIME_K_UNITS[I].max_components_per_lane),
                                     ONE_WINDOW_RUNTIME_K_UNITS[I].load_ahead>(),
                                 dispatched = true)) ||
                               ...));
        }(std::make_index_sequence<ONE_WINDOW_RUNTIME_K_UNITS.size()>{});
    };
#if THOR_CUB_ENABLE_FP8_TYPES
    switch (input_dtype) {
        case DataType::FP8_E4M3: for_type.template operator()<__nv_fp8_e4m3>(); break;
        case DataType::FP8_E5M2: for_type.template operator()<__nv_fp8_e5m2>(); break;
        default: break;
    }
#else
    static_cast<void>(input_dtype);
    static_cast<void>(for_type);
#endif
    return dispatched;
}

// The plan for `unit` at K, when the unit is compiled and K falls in its bucket.
[[nodiscard]] std::optional<OneWindowRuntimeKPlan> oneWindowRuntimeKUnitPlan(uint64_t k,
                                                                             const Fp8OneWindowRuntimeKUnit& unit) {
    const auto plan = oneWindowRuntimeKPlan(k, unit.outputs_per_warp, unit.packet_bytes);
    if (!plan || plan->max_components_per_lane != unit.max_components_per_lane ||
        !oneWindowRuntimeKUnitCompiled(unit)) {
        return std::nullopt;
    }
    return plan;
}

// The compiled unit a selector OneWindow variant (outputs per warp, packet bytes) runs at K: the one unit with that
// ownership, packet width and K's bucket (each such triple has exactly one loop form).
[[nodiscard]] std::optional<Fp8OneWindowRuntimeKUnit> oneWindowUnitForVariant(uint64_t k,
                                                                              uint32_t outputs_per_warp,
                                                                              uint32_t packet_bytes) {
    const auto plan = oneWindowRuntimeKPlan(k, outputs_per_warp, packet_bytes);
    if (!plan) {
        return std::nullopt;
    }
    for (const Fp8OneWindowRuntimeKUnit& unit : ONE_WINDOW_RUNTIME_K_UNITS) {
        if (unit.outputs_per_warp == outputs_per_warp && unit.packet_bytes == packet_bytes &&
            unit.max_components_per_lane == plan->max_components_per_lane) {
            return unit;
        }
    }
    return std::nullopt;
}

template <int FixedOp>
void launchFp8OneWindowRuntimeKCompleteTyped(const Tensor& input,
                                             Tensor& output,
                                             const CubReductionGeometry& geometry,
                                             const Fp8OneWindowRuntimeKUnit& unit,
                                             uint32_t warps_per_output,
                                             Stream& stream,
                                             const CubReductionStageSemantics& semantics,
                                             float output_scale) {
#if THOR_CUB_ENABLE_FP8_TYPES
    const AdditiveFinalizeFp32 output_finalize = fp8CompleteFinalizerFromSemantics(semantics);
    const auto plan = oneWindowRuntimeKUnitPlan(geometry.inner_size, unit);
    if (!plan || geometry.outer_size == 0 || geometry.reduction_size == 0 ||
        !oneWindowValidWarpsPerOutput(unit.outputs_per_warp, warps_per_output) ||
        !shallowFp8FitsUint32Indexing(geometry) || !fp8CompleteSupportsOutputStorage(output.getDataType()) ||
        output.getTotalNumElements() != geometry.outer_size * geometry.inner_size) {
        throw std::logic_error("FP8 bucketed OneWindow received unsupported complete geometry.");
    }
    if ((reinterpret_cast<uintptr_t>(input.getMemPtr<void>()) & uintptr_t{15}) != 0) {
        throw std::logic_error("FP8 bucketed OneWindow requires a 16-byte aligned input base.");
    }
    float folded_scale = output_scale;
    const Fp8ExtremumFold extremum_fold =
        fp8CompleteExtremumFold(FixedOp, semantics.input_transform, folded_scale);
    const uint64_t outputs_per_block = (SHALLOW_FP8_WARPS / warps_per_output) * unit.outputs_per_warp;
    const uint64_t block_count = std::min<uint64_t>(ceilDivideU64(geometry.outer_size, outputs_per_block),
                                                    NARROW_FLAT_MAX_GRID_BLOCKS);
    const bool launched = dispatchOneWindowRuntimeK(
        input.getDataType(), unit,
        [&]<typename InputT, int OutputsPerWarp, int PacketBytes, int MaxComponentsPerLane, bool LoadAhead>() {
            fp8OneWindowRuntimeKCompleteKernel<InputT, OutputsPerWarp, PacketBytes, MaxComponentsPerLane, LoadAhead,
                                               fp8PhysicalExtremumOp(FixedOp)>
                <<<static_cast<unsigned int>(block_count), SHALLOW_FP8_THREADS, 0, stream.getStream()>>>(
                    input.getMemPtr<InputT>(),
                    output.getMemPtr<void>(),
                    output.getDataType(),
                    static_cast<uint32_t>(geometry.outer_size),
                    static_cast<uint32_t>(geometry.reduction_size),
                    warps_per_output,
                    plan->geometry,
                    output_finalize,
                    semantics.input_transform,
                    folded_scale,
                    extremum_fold);
            CUDA_CHECK(cudaGetLastError());
        });
    if (!launched) {
        throw std::logic_error("FP8 bucketed OneWindow has no kernel for this request.");
    }
#else
    static_cast<void>(input);
    static_cast<void>(output);
    static_cast<void>(geometry);
    static_cast<void>(unit);
    static_cast<void>(warps_per_output);
    static_cast<void>(stream);
    static_cast<void>(semantics);
    static_cast<void>(output_scale);
    throw std::logic_error("FP8 bucketed OneWindow requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}

template <int FixedOp>
CubKernelOccupancyInfo queryFp8OneWindowRuntimeKOccupancyTyped(DataType input_dtype,
                                                               uint64_t inner_size,
                                                               const Fp8OneWindowRuntimeKUnit& unit,
                                                               uint32_t warps_per_output) {
    const auto plan = oneWindowRuntimeKUnitPlan(inner_size, unit);
    if (!plan || !oneWindowValidWarpsPerOutput(unit.outputs_per_warp, warps_per_output)) {
        throw std::logic_error("FP8 bucketed OneWindow occupancy received unsupported ownership.");
    }
    CubKernelOccupancyInfo occupancy{};
    const bool queried = dispatchOneWindowRuntimeK(
        input_dtype, unit,
        [&]<typename InputT, int OutputsPerWarp, int PacketBytes, int MaxComponentsPerLane, bool LoadAhead>() {
            occupancy = queryShallowFp8Occupancy(
                fp8OneWindowRuntimeKCompleteKernel<InputT, OutputsPerWarp, PacketBytes, MaxComponentsPerLane,
                                                   LoadAhead, fp8PhysicalExtremumOp(FixedOp)>);
        });
    if (!queried) {
        throw std::logic_error("FP8 bucketed OneWindow occupancy has no kernel for this request.");
    }
    return occupancy;
}

}  // namespace

void launchFp8DeepRSmallKRetainedFirstStage(const CubReductionStageSemantics& semantics,
                                            const Tensor& input, Tensor& fp32_partials,
                                            const CubReductionGeometry& geometry, uint64_t rows,
                                            uint64_t shards, Stream& stream) {
#if THOR_CUB_ENABLE_FP8_TYPES
    const auto& depths = Fp8RK::Fp8DeepRSmallKPolicy::compiled_rows;
    if (std::find(depths.begin(), depths.end(), rows) == depths.end())
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
    const auto& depths = Fp8RK::Fp8DeepRSmallKPolicy::compiled_rows;
    if (k == 0 || k > 256 || std::find(depths.begin(), depths.end(), rows) == depths.end())
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

// The production OneWindow launcher and occupancy query (P5-J2): the selector's OneWindow variant (O, W, P) runs
// the bucketed unit for (O, P, bucket(K)). Only P16 variants have units; a request without a unit throws
// std::logic_error, which the selector treats as an excluded candidate.
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
    const auto unit = oneWindowUnitForVariant(geometry.inner_size, outputs_per_warp, packet_bytes);
    if (!unit) {
        throw std::logic_error("FP8 OneWindow has no bucketed kernel for this K, ownership and packet width.");
    }
    if (output.getDimensions() != geometry.output_dimensions) {
        throw std::logic_error("FP8 OneWindow received unsupported complete geometry.");
    }
    launchFp8OneWindowRuntimeKComplete(input, output, geometry, *unit, warps_per_output, stream, semantics,
                                       output_scale);
}

CubKernelOccupancyInfo queryFp8GeneralRParallelRCooperativeCalibrationOccupancy(
    DataType input_dtype,
    uint64_t inner_size,
    uint32_t outputs_per_warp,
    uint32_t warps_per_output,
    uint32_t packet_bytes,
    CubReductionStageCombine combine) {
    const auto unit = oneWindowUnitForVariant(inner_size, outputs_per_warp, packet_bytes);
    if (!unit) {
        throw std::logic_error("FP8 OneWindow occupancy: no bucketed kernel for this K, ownership and packet width.");
    }
    return queryFp8OneWindowRuntimeKOccupancy(input_dtype, inner_size, *unit, warps_per_output, combine);
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


std::vector<Fp8OneWindowRuntimeKUnit> fp8OneWindowRuntimeKUnits(uint64_t inner_size) {
    std::vector<Fp8OneWindowRuntimeKUnit> units;
    for (const Fp8OneWindowRuntimeKUnit& unit : ONE_WINDOW_RUNTIME_K_UNITS) {
        if (oneWindowRuntimeKUnitPlan(inner_size, unit)) {
            units.push_back(unit);
        }
    }
    return units;
}

void launchFp8OneWindowRuntimeKComplete(const Tensor& input,
                                        Tensor& output,
                                        const CubReductionGeometry& geometry,
                                        const Fp8OneWindowRuntimeKUnit& unit,
                                        uint32_t warps_per_output,
                                        Stream& stream,
                                        const CubReductionStageSemantics& semantics,
                                        float output_scale) {
    switch (semantics.combine) {
        case CubReductionStageCombine::Sum:
            return launchFp8OneWindowRuntimeKCompleteTyped<0>(input, output, geometry, unit, warps_per_output, stream,
                                                              semantics, output_scale);
        case CubReductionStageCombine::Product:
            return launchFp8OneWindowRuntimeKCompleteTyped<1>(input, output, geometry, unit, warps_per_output, stream,
                                                              semantics, output_scale);
        case CubReductionStageCombine::Minimum:
            return launchFp8OneWindowRuntimeKCompleteTyped<2>(input, output, geometry, unit, warps_per_output, stream,
                                                              semantics, output_scale);
        case CubReductionStageCombine::Maximum:
            return launchFp8OneWindowRuntimeKCompleteTyped<3>(input, output, geometry, unit, warps_per_output, stream,
                                                              semantics, output_scale);
    }
    throw std::logic_error("Invalid bucketed OneWindow combine operation.");
}

CubKernelOccupancyInfo queryFp8OneWindowRuntimeKOccupancy(DataType input_dtype,
                                                          uint64_t inner_size,
                                                          const Fp8OneWindowRuntimeKUnit& unit,
                                                          uint32_t warps_per_output,
                                                          CubReductionStageCombine combine) {
    switch (combine) {
        case CubReductionStageCombine::Sum:
            return queryFp8OneWindowRuntimeKOccupancyTyped<0>(input_dtype, inner_size, unit, warps_per_output);
        case CubReductionStageCombine::Product:
            return queryFp8OneWindowRuntimeKOccupancyTyped<1>(input_dtype, inner_size, unit, warps_per_output);
        case CubReductionStageCombine::Minimum:
            return queryFp8OneWindowRuntimeKOccupancyTyped<2>(input_dtype, inner_size, unit, warps_per_output);
        case CubReductionStageCombine::Maximum:
            return queryFp8OneWindowRuntimeKOccupancyTyped<3>(input_dtype, inner_size, unit, warps_per_output);
    }
    throw std::logic_error("Invalid bucketed OneWindow occupancy combine operation.");
}

}  // namespace ThorImplementation::CubReductionInternal
