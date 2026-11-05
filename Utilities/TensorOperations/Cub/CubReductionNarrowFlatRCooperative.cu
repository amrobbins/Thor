#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"
#include "Utilities/TensorOperations/Cub/CubReductionOperation.cuh"
#include "Utilities/TensorOperations/Cub/Fp8ArgKey.h"
#include "Utilities/TensorOperations/Cub/Fp32ArgKey.cuh"
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

static_assert(Fp8RK::FP8_RK_MAX_GRID_EXTENT == REDUCTION_WORK_GRID_MAX_EXTENT,
              "Fp8RKLaunchLimits.h must restate reductionWorkGrid's extent cap");

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

// 1185: ElementBytes 4 for FP32 input (a row is k elements of 4 bytes); FP8 is ElementBytes 1.
template <uint32_t PacketBytes, uint32_t ElementBytes = 1>
__device__ __forceinline__ OneWindowGroupWindow oneWindowRuntimeKGroupWindow(bool valid_output,
                                                                             uint32_t outer_byte_begin,
                                                                             uint32_t group_row,
                                                                             uint32_t reduction_size,
                                                                             uint32_t k,
                                                                             uint32_t rows_per_group) {
    const uint32_t remaining_rows = reduction_size - group_row;
    OneWindowGroupWindow window{remaining_rows < rows_per_group ? remaining_rows : rows_per_group, 0U, 0U, 0U};
    if (valid_output) {
        const uint32_t row_bytes = k * ElementBytes;
        const uint32_t group_byte_begin = outer_byte_begin + group_row * row_bytes;
        constexpr uint32_t PacketShift = PacketBytes == 4 ? 2U : (PacketBytes == 8 ? 3U : 4U);
        window.prefix_bytes = group_byte_begin & (PacketBytes - 1U);
        window.aligned_byte_begin = group_byte_begin - window.prefix_bytes;
        window.packet_count = (window.prefix_bytes + window.rows * row_bytes + PacketBytes - 1U) >> PacketShift;
    }
    return window;
}

// 1185: one staged FP32 element, MIN / MAX folded and the SUM-family transform applied. Staging is plane-major: element
// e of the window, counted from the first staged packet, is word e % PacketWords of the subgroup's packet
// e / PacketWords.
template <int FixedOp, uint32_t PacketBytes>
__device__ __forceinline__ float oneWindowStagedFp32(const uint32_t* warp_words,
                                                     uint32_t subgroup_lane_base,
                                                     uint32_t element,
                                                     CubReductionStageInputTransform input_transform,
                                                     const Fp8ExtremumFold& extremum_fold) {
    constexpr uint32_t PacketWords = PacketBytes / 4U;
    const uint32_t word =
        warp_words[(element % PacketWords) * TILED_REDUCTION_WARP_THREADS + subgroup_lane_base + element / PacketWords];
    float value = __uint_as_float(foldFp8ExtremumBytes<FixedOp == 2>(word, extremum_fold));
    if constexpr (FixedOp != 2) {
        if (input_transform == CubReductionStageInputTransform::AbsoluteValue) {
            value = ::fabsf(value);
        } else if (input_transform == CubReductionStageInputTransform::Square) {
            value *= value;
        }
    } else {
        static_cast<void>(input_transform);
    }
    return value;
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
    if constexpr (std::is_same_v<InputT, float>) {
        // 1185: FP32, one element per staged word, rows of each component accumulated in increasing order. Buckets 1
        // and 2: a lane walks its component's rows with stride `workers`; buckets 4 (the largest FP32 reaches: K <=
        // 125 at O1) walk the rows and take the lane's components of each.
        const Fp8TypedReductionOp<FixedOp> combine{};
        const uint32_t first_element = prefix_bytes / 4U;
        if constexpr (MaxComponentsPerLane <= 2) {
            constexpr bool MultipleComponentsPerLane = MaxComponentsPerLane > 1;
            const uint32_t row_stride = workers * k;  // elements between a worker's consecutive rows
#pragma unroll
            for (uint32_t slot = 0; slot < static_cast<uint32_t>(MaxComponentsPerLane); ++slot) {
                const uint32_t owned_component =
                    MultipleComponentsPerLane ? subgroup_lane + slot * LanesPerOutput : component;
                if (owned_component >= k) {
                    continue;
                }
                uint32_t element = first_element + worker * k + owned_component;
                for (uint32_t row = worker; row < group_rows; row += workers, element += row_stride) {
                    local[slot] = combine(local[slot], oneWindowStagedFp32<FixedOp, PacketBytes>(
                                                           warp_words, subgroup_lane_base, element, input_transform,
                                                           extremum_fold));
                }
            }
        } else {
            static_cast<void>(component);
            static_cast<void>(worker);
            static_cast<void>(workers);
            for (uint32_t row = 0; row < group_rows; ++row) {
                const uint32_t row_element = first_element + row * k + subgroup_lane;
#pragma unroll
                for (uint32_t slot = 0; slot < static_cast<uint32_t>(MaxComponentsPerLane); ++slot) {
                    if (subgroup_lane + slot * LanesPerOutput < k) {
                        local[slot] = combine(local[slot], oneWindowStagedFp32<FixedOp, PacketBytes>(
                                                               warp_words, subgroup_lane_base,
                                                               row_element + slot * LanesPerOutput, input_transform,
                                                               extremum_fold));
                    }
                }
            }
        }
    } else if constexpr (MaxComponentsPerLane <= 2) {
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
        return oneWindowRuntimeKGroupWindow<PacketBytes, static_cast<uint32_t>(sizeof(InputT))>(
            valid_output, outer_byte_begin, group_row, reduction_size, geometry.k, geometry.rows_per_group);
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
    static_assert(sizeof(InputT) == 1 || std::is_same_v<InputT, float>);  // 1185: FP32 (buckets <= 4)
    static_assert(sizeof(InputT) == 1 || MaxComponentsPerLane <= 4);
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
    const uint32_t outer_input_stride = reduction_size * k * static_cast<uint32_t>(sizeof(InputT));  // bytes
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
// 1190: shared with FP32 input (plans/Thor_FP32_Dense_Reduction_Plan, phase 5). The mapping is in components: a p16
// packet holds 4 FP32 components, so the staged window is 128 components and an FP32 tile is capped at 124 (FP8's 496 is
// 31 packets; 124 is 31 FP32 packets), four slots per lane. FP32 reads its components from the staged words directly
// (no byte gather or packed conversion); MIN/MAX combine the raw values (no fold).
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
    constexpr bool Fp32Input = std::is_same_v<InputT, float>;
    static_assert(sizeof(InputT) == 1 || Fp32Input);
    static_assert(FixedInputTransform >= 0 && FixedInputTransform <= 2,
                  "RuntimeK Complete requires a compile-time input transform.");
    constexpr uint32_t PacketBytes = 16;
    // Components per packet (16 FP8 / 4 FP32); prefixes, packet counts and offsets below are in components.
    constexpr uint32_t PacketShift = Fp32Input ? 2 : 4;
    constexpr uint32_t PacketMask = (PacketBytes / sizeof(InputT)) - 1;
    constexpr uint32_t PacketWords = 4;
    constexpr uint32_t MaxTileComponents = Fp32Input ? 124 : 496;
    constexpr uint32_t MaxComponentsPerLane =
        (MaxTileComponents + TILED_REDUCTION_WARP_THREADS - 1) / TILED_REDUCTION_WARP_THREADS;
    static_assert(MaxComponentsPerLane == (Fp32Input ? 4 : 16));
    static_assert((1U << PacketShift) == PacketMask + 1);

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
            const uint64_t logical_begin = static_cast<uint64_t>(outer_index) * outer_input_stride +
                                           static_cast<uint64_t>(row_index) * inner_size + tile_begin;
            prefix_bytes = static_cast<uint32_t>(logical_begin & PacketMask);
            packet_count = (prefix_bytes + tile_width + PacketMask) >> PacketShift;
            if (lane < packet_count) {
                packet = *reinterpret_cast<const RawVectorPacket<PacketBytes>*>(
                    reinterpret_cast<const uint8_t*>(input) + (logical_begin - prefix_bytes) * sizeof(InputT) +
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
    static_assert(sizeof(InputT) == 1 || std::is_same_v<InputT, float>);
    constexpr uint32_t MaxTileComponents = std::is_same_v<InputT, float> ? 124 : 496;
    if (tile_components == 0 || tile_components > MaxTileComponents ||
        (warps_per_output != 1 && warps_per_output != 2 && warps_per_output != 4 && warps_per_output != 8)) {
        throw std::logic_error("RuntimeK PacketComplete received illegal tile/W geometry.");
    }
    if ((reinterpret_cast<uintptr_t>(input.getMemPtr<InputT>()) & uintptr_t{15}) != 0) {
        throw std::logic_error("RuntimeK PacketComplete requires p16-aligned input base.");
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


// 1201: the FP32 RK ARG PacketComplete twin (plans/Thor_FP32_Dense_Reduction_Plan, phase 6): the FP32 path of
// fp8GeneralRParallelKTileP16CompleteKernel (persistent 256-thread blocks; a work item is [outer, K tile] of at most 124
// components; W1/W2/W4/W8 warps per item take rows round-robin; a warp loads one row's tile window with its 0..3
// prefix, stages it plane-major and reads its four slots, with the next row's packet loaded ahead) with one FP32 ARG
// candidate (Fp32ArgKey.cuh) per owned component. A warp's rows only increase, so a strictly smaller key replaces its
// best; the warps of an item merge in shared memory with the (key, tail) minimum. Complete: writes the requested
// outputs. R <= 2^31 - 1.
template <int Unused>
__global__ __launch_bounds__(SHALLOW_FP8_THREADS, 4) void fp32PacketCompleteArgKernel(const float* input,
                                                                                    Fp32ArgOutputs outputs,
                                                                                    uint32_t outer_size,
                                                                                    uint32_t reduction_size,
                                                                                    uint32_t inner_size,
                                                                                    uint32_t tile_components,
                                                                                    uint32_t tile_count,
                                                                                    uint32_t warps_per_output) {
    constexpr uint32_t PacketBytes = 16;
    constexpr uint32_t PacketWords = 4;
    constexpr uint32_t MaxTileComponents = 124;
    constexpr uint32_t Slots = 4;
    __shared__ uint32_t staged_packet_words[SHALLOW_FP8_WARPS][PacketWords * TILED_REDUCTION_WARP_THREADS];
    __shared__ uint32_t warp_keys[SHALLOW_FP8_WARPS][MaxTileComponents];
    __shared__ uint32_t warp_tails[SHALLOW_FP8_WARPS][MaxTileComponents];

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
            const uint64_t logical_begin = static_cast<uint64_t>(outer_index) * outer_input_stride +
                                           static_cast<uint64_t>(row_index) * inner_size + tile_begin;
            prefix = static_cast<uint32_t>(logical_begin & 3U);
            packet_count = (prefix + tile_width + 3U) >> 2;
            if (lane < packet_count) {
                packet = *reinterpret_cast<const RawVectorPacket<PacketBytes>*>(
                    reinterpret_cast<const uint8_t*>(input) + (logical_begin - prefix) * sizeof(float) +
                    static_cast<uint64_t>(lane) * PacketBytes);
            }
        };
        // Component c of the staged row sits at position prefix + c: packet (position >> 2), word (position & 3).
        const auto key_row = [&](uint32_t row, uint32_t prefix) {
#pragma unroll
            for (uint32_t slot = 0; slot < Slots; ++slot) {
                const uint32_t component = lane + slot * TILED_REDUCTION_WARP_THREADS;
                if (component < tile_width) {
                    const uint32_t staged = prefix + component;
                    const uint32_t bits =
                        staged_packet_words[warp][(staged & 3U) * TILED_REDUCTION_WARP_THREADS + (staged >> 2)];
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

// 1186: FP32 DeepRSmallK reads staged floats directly (decodeNarrowFp8DeepRGroup's FP32 branch); this empty decoder
// type only names the FP32 instantiations and is never called.
struct Fp32StagedDecoder {};

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
    static_assert(sizeof(InputT) == 1 || std::is_same_v<InputT, float>);  // 1186: FP32 (4-byte elements)
    constexpr uint32_t ElementBytes = static_cast<uint32_t>(sizeof(InputT));
    NarrowFp8GroupWindow<PacketBytes> window;
    window.group_rows = rows_per_warp_group < shard_rows - group_row ? rows_per_warp_group : shard_rows - group_row;
    const uintptr_t group_address = reinterpret_cast<uintptr_t>(shard_input + group_row * inner_size_32);
    window.prefix_elements = static_cast<uint32_t>(group_address & uintptr_t{PacketBytes - 1}) / ElementBytes;
    window.aligned_begin =
        reinterpret_cast<const RawVectorPacket<PacketBytes>*>(group_address & ~uintptr_t{PacketBytes - 1});
    window.packet_count =
        ((window.prefix_elements + window.group_rows * inner_size_32) * ElementBytes + PacketBytes - 1) / PacketBytes;
    return window;
}

// VALUE-P1: packed FP8x4 decode in the DeepRSmallK first stage. The scalar loop decoded one staged byte per element;
// the ARG twin's packed keying (1109) ran 0.93x VALUE's time at eight components per lane. With the typed decoder,
// lanes owning four or more components decode four components of one row per decodeFp8x4ToFp32 (rows still in
// increasing order per component: bit-identical). 1120 A/B (MIN/MAX, cold L2): c8 0.83x (rows1024 0.69x), c5 0.96x,
// c4 0.985x; DeepRSmallK SASS -4%. Packing four rows of one component for one or two components per lane lost
// (c1 1.034x, c2 1.010x), so those keep the scalar loop (1122). THOR_FP8_DEEP_R_VALUE_SCALAR_DECODE=1 restores it.
#ifndef THOR_FP8_DEEP_R_VALUE_SCALAR_DECODE
#define THOR_FP8_DEEP_R_VALUE_SCALAR_DECODE 0
#endif
template <typename InputT, int MaxComponentsPerLane, typename ReductionOpT, typename InputTransformT, typename DecoderT>
__device__ __forceinline__ void decodeNarrowFp8DeepRGroup(float (&local)[MaxComponentsPerLane],
                                                          const uint8_t* staged_bytes,
                                                          uint32_t prefix_elements,
                                                          uint32_t lane,
                                                          uint32_t group_rows,
                                                          uint32_t inner_size_32,
                                                          const ReductionOpT& reduction_op,
                                                          const InputTransformT& input_transform,
                                                          const DecoderT& decoder) {
    constexpr bool Packed = !THOR_FP8_DEEP_R_VALUE_SCALAR_DECODE &&
                            std::is_same_v<DecoderT, NarrowFp8TypedFormatDecoder<InputT>>;
    static_cast<void>(decoder);
    if constexpr (std::is_same_v<InputT, float>) {
        // 1186: FP32, one staged float per element, rows of each component in increasing order.
        const auto* staged = reinterpret_cast<const float*>(staged_bytes);
#pragma unroll
        for (int slot = 0; slot < MaxComponentsPerLane; ++slot) {
            const uint32_t component = lane + static_cast<uint32_t>(slot) * TILED_REDUCTION_WARP_THREADS;
            if (component < inner_size_32) {
                uint32_t staged_index = prefix_elements + component;
                for (uint32_t row = 0; row < group_rows; ++row, staged_index += inner_size_32) {
                    local[slot] = reduction_op(local[slot], input_transform(staged[staged_index]));
                }
            }
        }
    } else if constexpr (Packed && MaxComponentsPerLane >= 4) {
#pragma unroll
        for (int slot_base = 0; slot_base < MaxComponentsPerLane; slot_base += 4) {
            const uint32_t first_component = lane + static_cast<uint32_t>(slot_base) * TILED_REDUCTION_WARP_THREADS;
            if (first_component >= inner_size_32) continue;
            uint32_t staged_index = prefix_elements + first_component;
            for (uint32_t row = 0; row < group_rows; ++row, staged_index += inner_size_32) {
                uint32_t packed = 0;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    if (slot_base + j < MaxComponentsPerLane &&
                        first_component + static_cast<uint32_t>(j) * TILED_REDUCTION_WARP_THREADS < inner_size_32) {
                        packed |= static_cast<uint32_t>(staged_bytes[staged_index + j * TILED_REDUCTION_WARP_THREADS])
                                  << (8 * j);
                    }
                }
                const float4 decoded = decodeFp8x4ToFp32<InputT>(packed);
                const float values[4] = {decoded.x, decoded.y, decoded.z, decoded.w};
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    if (slot_base + j < MaxComponentsPerLane &&
                        first_component + static_cast<uint32_t>(j) * TILED_REDUCTION_WARP_THREADS < inner_size_32) {
                        local[slot_base + j] = reduction_op(local[slot_base + j], input_transform(values[j]));
                    }
                }
            }
        }
    } else {
        for (int slot = 0; slot < MaxComponentsPerLane; ++slot) {
            const uint32_t component = lane + static_cast<uint32_t>(slot) * TILED_REDUCTION_WARP_THREADS;
            if (component < inner_size_32) {
                uint32_t staged_index = prefix_elements + component;
                for (uint32_t row = 0; row < group_rows; ++row, staged_index += inner_size_32) {
                    const float value = decoder(staged_bytes[staged_index]);
                    local[slot] = reduction_op(local[slot], input_transform(value));
                }
            }
        }
    }
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
    static_assert(sizeof(InputT) == 1 || std::is_same_v<InputT, float>);  // 1186: FP32 (components <= 4)
    static_assert(sizeof(InputT) == 1 || MaxComponentsPerLane <= 4);
    static_assert(PacketBytes == 4 || PacketBytes == 8 || PacketBytes == 16);
    static_assert(MaxComponentsPerLane == 1 || MaxComponentsPerLane == 2 || MaxComponentsPerLane == 4 || MaxComponentsPerLane == 5 ||
                  MaxComponentsPerLane == 8);
    constexpr uint32_t ElementBytes = static_cast<uint32_t>(sizeof(InputT));
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
                decodeNarrowFp8DeepRGroup<InputT, MaxComponentsPerLane>(local, staged_bytes, prefix_elements,
                                                                        static_cast<uint32_t>(lane), group_rows,
                                                                        inner_size_32, reduction_op, input_transform,
                                                                        decoder);
                __syncwarp();
            }
        } else {
            for (uint32_t group_row = static_cast<uint32_t>(warp) * rows_per_warp_group; group_row < shard_rows;
                 group_row += rows_per_block_round) {
                const uint32_t group_rows =
                    rows_per_warp_group < shard_rows - group_row ? rows_per_warp_group : shard_rows - group_row;
                const InputT* group_begin = shard_input + group_row * inner_size_32;
                const uintptr_t group_address = reinterpret_cast<uintptr_t>(group_begin);
                const uint32_t prefix_elements =
                    static_cast<uint32_t>(group_address & uintptr_t{PacketBytes - 1}) / ElementBytes;
                const auto* aligned_begin = reinterpret_cast<const RawVectorPacket<PacketBytes>*>(group_address & ~uintptr_t{PacketBytes - 1});
                const uint32_t logical_group_elements = group_rows * inner_size_32;
                const uint32_t packet_count =
                    ((prefix_elements + logical_group_elements) * ElementBytes + PacketBytes - 1) / PacketBytes;

                if (static_cast<uint32_t>(lane) < packet_count) {
                    staged_packets[warp][lane] = aligned_begin[lane];
                }
                __syncwarp();

                const auto* staged_bytes = reinterpret_cast<const uint8_t*>(&staged_packets[warp][0]);
                decodeNarrowFp8DeepRGroup<InputT, MaxComponentsPerLane>(local, staged_bytes, prefix_elements,
                                                                        static_cast<uint32_t>(lane), group_rows,
                                                                        inner_size_32, reduction_op, input_transform,
                                                                        decoder);
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
    static_assert(sizeof(InputT) == 1 || std::is_same_v<InputT, float>);  // 1186: FP32
    constexpr uint32_t ElementBytes = static_cast<uint32_t>(sizeof(InputT));
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
            const uint32_t prefix_elements =
                static_cast<uint32_t>(group_address & uintptr_t{PacketBytes - 1}) / ElementBytes;
            const auto* aligned_begin = reinterpret_cast<const RawVectorPacket<PacketBytes>*>(group_address & ~uintptr_t{PacketBytes - 1});
            const uint32_t logical_group_elements = group_rows * inner_size_32;
            const uint32_t packet_count =
                ((prefix_elements + logical_group_elements) * ElementBytes + PacketBytes - 1) / PacketBytes;

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
                    float value;
                    if constexpr (std::is_same_v<InputT, float>) {
                        static_cast<void>(decoder);
                        value = reinterpret_cast<const float*>(staged_bytes)[staged_index];  // 1186: FP32
                    } else {
                        value = decoder(staged_bytes[staged_index]);
                    }
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
// 1186: ElementBytes 4 for FP32, whose window holds (32 * PacketBytes - (PacketBytes - 4)) / 4 = 125 elements at P16
// after a worst-case prefix (FP8: 497).
template <int PacketBytes, uint32_t ElementBytes = 1>
inline constexpr uint32_t narrowPackedMaxLogicalGroupElements =
    (static_cast<uint32_t>(narrowFp8PacketWindowBytes<PacketBytes>) - (PacketBytes - ElementBytes)) / ElementBytes;
static_assert(narrowPackedMaxLogicalGroupElements<16> == 497 && narrowPackedMaxLogicalGroupElements<16, 4> == 125);

template <int PacketBytes, uint32_t ElementBytes = 1>
[[nodiscard]] uint32_t narrowFp8PackedRowsPerWarpGroup(uint64_t inner_size, uint32_t rows_per_shard) {
    constexpr uint32_t MaxElements = narrowPackedMaxLogicalGroupElements<PacketBytes, ElementBytes>;
    const uint32_t max_rows_per_warp_from_shard = rows_per_shard / static_cast<uint32_t>(NARROW_FLAT_WARPS);
    if (inner_size == 0 || inner_size > static_cast<uint64_t>(MaxElements)) {
        throw std::logic_error("Packed FP8 narrow-flat launch received K outside one packet window.");
    }
    const uint32_t rows_from_packet_window = MaxElements / static_cast<uint32_t>(inner_size);
    return rows_from_packet_window < max_rows_per_warp_from_shard ? rows_from_packet_window : max_rows_per_warp_from_shard;
}

template <typename InputT, typename ReductionOpT, typename InputTransformT, int PacketBytes, int LanesPerComponent,
          typename DecoderT = NarrowFp8TypedFormatDecoder<InputT>>
void launchNarrowFp8PackedTinyKForTransform(
    const Tensor& input, Tensor& fp32_partials, const CubReductionGeometry& geometry, uint64_t rows_per_shard,
    uint64_t shards_per_output, cudaStream_t stream,
    InputTransformT input_transform = InputTransformT{}, DecoderT decoder = DecoderT{}) {
    static_assert(sizeof(InputT) == 1 || std::is_same_v<InputT, float>);  // 1186: FP32
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
        narrowFp8PackedRowsPerWarpGroup<PacketBytes, static_cast<uint32_t>(sizeof(InputT))>(geometry.inner_size,
                                                                                           shard_depth);

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
    static_assert(sizeof(InputT) == 1 || std::is_same_v<InputT, float>);  // 1186: FP32
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
        narrowFp8PackedRowsPerWarpGroup<PacketBytes, static_cast<uint32_t>(sizeof(InputT))>(geometry.inner_size,
                                                                                           shard_depth);
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
    static_assert(sizeof(InputT) == 1 || std::is_same_v<InputT, float>);
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
    static_assert(sizeof(InputT) == 1 || std::is_same_v<InputT, float>);
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

// 1186: the FP32 DeepRSmallK ownerships by K, the FP8 kernels at 4-byte elements: a P16 window holds 125 FP32 after a
// worst-case prefix, so K <= 125 and lanes own at most four components (FP8's five- and eight-component kernels do
// not exist for FP32). The one-component kernel keeps its compiled depth, as for FP8.
static_assert(narrowPackedMaxLogicalGroupElements<16, 4> == 125, "fp32DeepRSmallKSupports restates K <= 125");

template <typename TransformT, typename ReductionOpT>
void dispatchFp32DeepRRetainedRows(const Tensor& input, Tensor& partials, const CubReductionGeometry& geometry,
                                   uint64_t rows, uint64_t shards, cudaStream_t stream) {
    const uint32_t depth = deepRRetainedRowDepth(rows);
    const TransformT transform{};
    const Fp32StagedDecoder decoder{};
    if (geometry.inner_size <= 4)
        launchNarrowFp8PackedTinyKForTransform<float, ReductionOpT, TransformT, 16, 8, Fp32StagedDecoder>(
            input, partials, geometry, depth, shards, stream, transform, decoder);
    else if (geometry.inner_size <= 8)
        launchNarrowFp8PackedTinyKForTransform<float, ReductionOpT, TransformT, 16, 4, Fp32StagedDecoder>(
            input, partials, geometry, depth, shards, stream, transform, decoder);
    else if (geometry.inner_size <= 32)
        dispatchDeepRRetainedFixedDepth(depth, [&]<int DepthLog2>() {
            launchNarrowFp8PackedForTransform<float, ReductionOpT, TransformT, 16, 1, DepthLog2, Fp32StagedDecoder>(
                input, partials, geometry, depth, shards, stream, transform, decoder);
        });
    else if (geometry.inner_size <= 64)
        launchNarrowFp8PackedForTransform<float, ReductionOpT, TransformT, 16, 2, -1, Fp32StagedDecoder>(
            input, partials, geometry, depth, shards, stream, transform, decoder);
    else
        launchNarrowFp8PackedForTransform<float, ReductionOpT, TransformT, 16, 4, -1, Fp32StagedDecoder>(
            input, partials, geometry, depth, shards, stream, transform, decoder);
}

template <typename TransformT, typename ReductionOpT>
CubKernelOccupancyInfo queryFp32DeepRRetainedRows(uint64_t rows, uint64_t k) {
    static_cast<void>(deepRRetainedRowDepth(rows));
    if (k <= 4) return queryNarrowFp8PackedTinyKOccupancy<float, 16, 8, TransformT, Fp32StagedDecoder, ReductionOpT>();
    if (k <= 8) return queryNarrowFp8PackedTinyKOccupancy<float, 16, 4, TransformT, Fp32StagedDecoder, ReductionOpT>();
    if (k <= 32) {
        CubKernelOccupancyInfo info{};
        dispatchDeepRRetainedFixedDepth(deepRRetainedRowDepth(rows), [&]<int DepthLog2>() {
            info = queryNarrowFp8PackedOccupancy<float, 16, 1, TransformT, Fp32StagedDecoder, ReductionOpT,
                                                 DepthLog2>();
        });
        return info;
    }
    if (k <= 64) return queryNarrowFp8PackedOccupancy<float, 16, 2, TransformT, Fp32StagedDecoder, ReductionOpT>();
    return queryNarrowFp8PackedOccupancy<float, 16, 4, TransformT, Fp32StagedDecoder, ReductionOpT>();
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

// 1185: element_bytes 4 for FP32. The window and the worst-case prefix are counted in elements (a packet holds
// packet_bytes / element_bytes of them), so FP32 rows fit K <= 125 / 61 / 29 / 13 at O1 / O2 / O4 / O8 (P16).
[[nodiscard]] inline std::optional<OneWindowRuntimeKPlan> oneWindowRuntimeKPlan(uint64_t k,
                                                                                uint32_t outputs_per_warp,
                                                                                uint32_t packet_bytes,
                                                                                uint32_t element_bytes = 1) {
    if (k == 0 || (outputs_per_warp != 1 && outputs_per_warp != 2 && outputs_per_warp != 4 && outputs_per_warp != 8) ||
        (packet_bytes != 4 && packet_bytes != 8 && packet_bytes != 16) || (element_bytes != 1 && element_bytes != 4)) {
        return std::nullopt;
    }
    const uint32_t lanes = TILED_REDUCTION_WARP_THREADS / outputs_per_warp;
    const uint32_t packet_elements = packet_bytes / element_bytes;
    const uint64_t window = static_cast<uint64_t>(lanes) * packet_elements;
    const uint64_t capacity = k % packet_elements == 0 ? window : window - (packet_elements - 1);
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
                                                                             const Fp8OneWindowRuntimeKUnit& unit,
                                                                             uint32_t element_bytes = 1) {
    const auto plan = oneWindowRuntimeKPlan(k, unit.outputs_per_warp, unit.packet_bytes, element_bytes);
    if (!plan || plan->max_components_per_lane != unit.max_components_per_lane ||
        !oneWindowRuntimeKUnitCompiled(unit)) {
        return std::nullopt;
    }
    return plan;
}

// 1185: the FP32 OneWindow inventory is FP8's units whose bucket FP32 can reach (components per lane <= 4: a 16-byte
// lane packet holds 4 FP32), the same group loop form per unit. Invokes callback.template operator()<float,
// OutputsPerWarp, PacketBytes, MaxComponentsPerLane, LoadAhead>() for the unit equal to `unit`; false when FP32 has no
// such unit. Only this dispatch instantiates FP32 OneWindow kernels.
[[nodiscard]] constexpr bool oneWindowFp32Unit(const Fp8OneWindowRuntimeKUnit& unit) {
    return unit.max_components_per_lane <= 4;
}

template <typename Callback>
[[nodiscard]] bool dispatchFp32OneWindowRuntimeK(const Fp8OneWindowRuntimeKUnit& unit, Callback&& callback) {
    bool dispatched = false;
    const auto try_unit = [&]<size_t I>() {
        constexpr Fp8OneWindowRuntimeKUnit compiled = ONE_WINDOW_RUNTIME_K_UNITS[I];
        if constexpr (oneWindowFp32Unit(compiled)) {
            if (!dispatched && compiled == unit) {
                callback.template operator()<float, static_cast<int>(compiled.outputs_per_warp),
                                             static_cast<int>(compiled.packet_bytes),
                                             static_cast<int>(compiled.max_components_per_lane),
                                             compiled.load_ahead>();
                dispatched = true;
            }
        }
    };
    [&]<size_t... I>(std::index_sequence<I...>) {
        (try_unit.template operator()<I>(), ...);
    }(std::make_index_sequence<ONE_WINDOW_RUNTIME_K_UNITS.size()>{});
    return dispatched;
}

[[nodiscard]] std::optional<OneWindowRuntimeKPlan> fp32OneWindowRuntimeKUnitPlan(uint64_t k,
                                                                                 const Fp8OneWindowRuntimeKUnit& unit) {
    if (!oneWindowFp32Unit(unit)) return std::nullopt;
    return oneWindowRuntimeKUnitPlan(k, unit, 4);
}

// FP32 OneWindow indexes input bytes in 32 bits (the kernel's byte offsets), so the input holds at most 2^32 - 1
// bytes; outputs and rows as FP8.
[[nodiscard]] inline bool fp32OneWindowFitsIndexing(const CubReductionGeometry& geometry) {
    constexpr uint64_t MaxIndex = std::numeric_limits<uint32_t>::max();
    return shallowFp8FitsUint32Indexing(geometry) && geometry.input_elements <= MaxIndex / 4;
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

#if THOR_CUB_ENABLE_FP8_TYPES  // the retained kernels' gate (FP32 shares the FP8 OneWindow kernel)
template <int FixedOp>
void launchFp32OneWindowRuntimeKCompleteTyped(const Tensor& input,
                                              Tensor& output,
                                              const CubReductionGeometry& geometry,
                                              const Fp8OneWindowRuntimeKUnit& unit,
                                              uint32_t warps_per_output,
                                              Stream& stream,
                                              const CubReductionStageSemantics& semantics,
                                              float output_scale) {
    const AdditiveFinalizeFp32 output_finalize = fp8CompleteFinalizerFromSemantics(semantics);
    const auto plan = fp32OneWindowRuntimeKUnitPlan(geometry.inner_size, unit);
    if (input.getDataType() != DataType::FP32 || !plan || geometry.outer_size == 0 || geometry.reduction_size == 0 ||
        !oneWindowValidWarpsPerOutput(unit.outputs_per_warp, warps_per_output) ||
        !fp32OneWindowFitsIndexing(geometry) || !fp8CompleteSupportsOutputStorage(output.getDataType()) ||
        input.getTotalNumElements() != geometry.input_elements ||
        output.getTotalNumElements() != geometry.outer_size * geometry.inner_size) {
        throw std::logic_error("FP32 OneWindow received unsupported complete geometry.");
    }
    if ((reinterpret_cast<uintptr_t>(input.getMemPtr<void>()) & uintptr_t{15}) != 0) {
        throw std::logic_error("FP32 OneWindow requires a 16-byte aligned input base.");
    }
    float folded_scale = output_scale;
    const Fp8ExtremumFold extremum_fold = fp32CompleteExtremumFold(FixedOp, semantics.input_transform, folded_scale);
    const uint64_t outputs_per_block = (SHALLOW_FP8_WARPS / warps_per_output) * unit.outputs_per_warp;
    const uint64_t block_count = std::min<uint64_t>(ceilDivideU64(geometry.outer_size, outputs_per_block),
                                                    NARROW_FLAT_MAX_GRID_BLOCKS);
    const bool launched = dispatchFp32OneWindowRuntimeK(
        unit, [&]<typename InputT, int OutputsPerWarp, int PacketBytes, int MaxComponentsPerLane, bool LoadAhead>() {
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
        throw std::logic_error("FP32 OneWindow has no kernel for this request.");
    }
}

template <int FixedOp>
CubKernelOccupancyInfo queryFp32OneWindowRuntimeKOccupancyTyped(uint64_t inner_size,
                                                                const Fp8OneWindowRuntimeKUnit& unit,
                                                                uint32_t warps_per_output) {
    const auto plan = fp32OneWindowRuntimeKUnitPlan(inner_size, unit);
    if (!plan || !oneWindowValidWarpsPerOutput(unit.outputs_per_warp, warps_per_output)) {
        throw std::logic_error("FP32 OneWindow occupancy received unsupported ownership.");
    }
    CubKernelOccupancyInfo occupancy{};
    const bool queried = dispatchFp32OneWindowRuntimeK(
        unit, [&]<typename InputT, int OutputsPerWarp, int PacketBytes, int MaxComponentsPerLane, bool LoadAhead>() {
            occupancy = queryShallowFp8Occupancy(
                fp8OneWindowRuntimeKCompleteKernel<InputT, OutputsPerWarp, PacketBytes, MaxComponentsPerLane,
                                                   LoadAhead, fp8PhysicalExtremumOp(FixedOp)>);
        });
    if (!queried) {
        throw std::logic_error("FP32 OneWindow occupancy has no kernel for this request.");
    }
    return occupancy;
}
#endif  // THOR_CUB_ENABLE_FP8_TYPES

// ---------------------------------------------------------------------------------------------------------------
// 1202: the FP32 RK ARG OneWindow twin (plans/Thor_FP32_Dense_Reduction_Plan, phase 6): the FP32 path of
// fp8OneWindowRuntimeKCompleteKernel (the FP32 units, buckets <= 4: packet windows of rows_per_group rows staged
// plane-major, O1/O2/O4/O8 outputs per warp, R workers per component in one-component buckets, O1 W1/W2/W4/W8) with
// one FP32 ARG candidate (Fp32ArgKey.cuh) per owned component; the tail is the absolute row << 1 | negative zero. A
// lane's rows only increase, so a strictly smaller key replaces its best; workers fold with butterfly shuffles
// (power-of-two counts) or shared memory, O1 warps in shared memory, all with the (key, tail) minimum. Complete:
// writes the requested outputs. R <= 2^31 - 1.
template <int OutputsPerWarp, int MaxComponentsPerLane>
__device__ __forceinline__ void keyOneWindowFp32ArgGroup(uint32_t (&best_key)[MaxComponentsPerLane],
                                                         uint32_t (&best_tail)[MaxComponentsPerLane],
                                                         const uint32_t* warp_words,
                                                         uint32_t subgroup_lane,
                                                         uint32_t subgroup_lane_base,
                                                         uint32_t component,
                                                         uint32_t worker,
                                                         uint32_t k,
                                                         uint32_t workers,
                                                         uint32_t group_row,
                                                         uint32_t group_rows,
                                                         uint32_t prefix_bytes,
                                                         uint32_t xor_mask) {
    constexpr uint32_t LanesPerOutput = TILED_REDUCTION_WARP_THREADS / OutputsPerWarp;
    const auto key_element = [&](uint32_t slot, uint32_t element, uint32_t row) {
        const uint32_t bits =
            warp_words[(element % 4U) * TILED_REDUCTION_WARP_THREADS + subgroup_lane_base + element / 4U];
        uint32_t negative_zero;
        const uint32_t key = fp32ArgKey(bits, xor_mask, negative_zero);
        if (key < best_key[slot]) {
            best_key[slot] = key;
            best_tail[slot] = ((group_row + row) << 1) | negative_zero;
        }
    };
    const uint32_t first_element = prefix_bytes / 4U;
    if constexpr (MaxComponentsPerLane <= 2) {
        constexpr bool MultipleComponentsPerLane = MaxComponentsPerLane > 1;
        const uint32_t row_stride = workers * k;
#pragma unroll
        for (uint32_t slot = 0; slot < static_cast<uint32_t>(MaxComponentsPerLane); ++slot) {
            const uint32_t owned_component =
                MultipleComponentsPerLane ? subgroup_lane + slot * LanesPerOutput : component;
            if (owned_component >= k) continue;
            uint32_t element = first_element + worker * k + owned_component;
            for (uint32_t row = worker; row < group_rows; row += workers, element += row_stride)
                key_element(slot, element, row);
        }
    } else {
        static_cast<void>(component);
        static_cast<void>(worker);
        static_cast<void>(workers);
        for (uint32_t row = 0; row < group_rows; ++row) {
            const uint32_t row_element = first_element + row * k + subgroup_lane;
#pragma unroll
            for (uint32_t slot = 0; slot < static_cast<uint32_t>(MaxComponentsPerLane); ++slot) {
                if (subgroup_lane + slot * LanesPerOutput < k)
                    key_element(slot, row_element + slot * LanesPerOutput, row);
            }
        }
    }
}

// Streams one output's R through packet windows first_row, first_row + row_step, ... into the lane's candidates
// (identity first). Force-inlined like streamOneWindowRuntimeKOutput: no lambda holds the candidate arrays.
template <int OutputsPerWarp, int MaxComponentsPerLane, bool LoadAhead>
__device__ __forceinline__ void streamOneWindowFp32ArgOutput(uint32_t (&best_key)[MaxComponentsPerLane],
                                                             uint32_t (&best_tail)[MaxComponentsPerLane],
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
                                                             uint32_t xor_mask) {
    constexpr uint32_t PacketBytes = 16;
#pragma unroll
    for (uint32_t slot = 0; slot < static_cast<uint32_t>(MaxComponentsPerLane); ++slot) {
        best_key[slot] = FP32_ARG_KEY_IDENTITY;
        best_tail[slot] = 0xFFFFFFFFU;
    }
    const bool keys = valid_output && subgroup_lane < geometry.task_count;
    if constexpr (LoadAhead) {
        uint32_t group_row = first_row;
        OneWindowGroupWindow current{};
        RawVectorPacket<PacketBytes> current_packet{};
        if (group_row < reduction_size) {
            current = oneWindowRuntimeKGroupWindow<PacketBytes, 4U>(valid_output, outer_byte_begin, group_row,
                                                                    reduction_size, geometry.k,
                                                                    geometry.rows_per_group);
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
                next = oneWindowRuntimeKGroupWindow<PacketBytes, 4U>(valid_output, outer_byte_begin, next_row,
                                                                     reduction_size, geometry.k,
                                                                     geometry.rows_per_group);
                if (valid_output && subgroup_lane < next.packet_count) {
                    next_packet = loadOneWindowPacket<PacketBytes>(
                        input_bytes, next.aligned_byte_begin + subgroup_lane * PacketBytes, total_input_bytes);
                }
            }
            if (valid_output && subgroup_lane < current.packet_count) {
                stageOneWindowPacket<PacketBytes>(warp_words, lane, current_packet);
            }
            __syncwarp();
            if (keys) {
                keyOneWindowFp32ArgGroup<OutputsPerWarp, MaxComponentsPerLane>(
                    best_key, best_tail, warp_words, subgroup_lane, subgroup_lane_base, component, worker, geometry.k,
                    geometry.workers_per_component, group_row, current.rows, current.prefix_bytes, xor_mask);
            }
            __syncwarp();
            current = next;
            current_packet = next_packet;
            group_row = next_row;
        }
    } else {
        for (uint32_t group_row = first_row; group_row < reduction_size; group_row += row_step) {
            const OneWindowGroupWindow window = oneWindowRuntimeKGroupWindow<PacketBytes, 4U>(
                valid_output, outer_byte_begin, group_row, reduction_size, geometry.k, geometry.rows_per_group);
            if (valid_output && subgroup_lane < window.packet_count) {
                stageOneWindowPacket<PacketBytes>(
                    warp_words, lane,
                    loadOneWindowPacket<PacketBytes>(input_bytes,
                                                     window.aligned_byte_begin + subgroup_lane * PacketBytes,
                                                     total_input_bytes));
            }
            __syncwarp();
            if (keys) {
                keyOneWindowFp32ArgGroup<OutputsPerWarp, MaxComponentsPerLane>(
                    best_key, best_tail, warp_words, subgroup_lane, subgroup_lane_base, component, worker, geometry.k,
                    geometry.workers_per_component, group_row, window.rows, window.prefix_bytes, xor_mask);
            }
            __syncwarp();
        }
    }
}

// One-component buckets: folds a component's R workers (consecutive lanes component * workers + worker). Power-of-two
// counts use butterfly shuffles (the result is valid in the worker-0 lane, component `component`); other counts fold
// through the warp's shared lane words (valid in lane subgroup_lane < K, component subgroup_lane). Warp-uniform.
// Returns the folded candidate; `ready` and `result_component` say which lanes hold one.
template <uint32_t LanesPerOutput>
__device__ __forceinline__ Fp32ArgCandidate foldOneWindowFp32ArgWorkers(Fp32ArgCandidate mine,
                                                                        uint32_t* lane_keys,
                                                                        uint32_t* lane_tails,
                                                                        uint32_t lane,
                                                                        uint32_t subgroup_lane,
                                                                        uint32_t subgroup_lane_base,
                                                                        uint32_t component,
                                                                        uint32_t worker,
                                                                        uint32_t k,
                                                                        uint32_t workers,
                                                                        bool& ready,
                                                                        uint32_t& result_component) {
    if ((workers & (workers - 1U)) == 0U) {
#pragma unroll
        for (uint32_t offset = 1; offset < LanesPerOutput; offset <<= 1) {
            if (offset < workers) mine = fp32ArgMin(mine, fp32ArgShuffleXor(mine, static_cast<int>(offset)));
        }
        ready = worker == 0U && component < k;
        result_component = component;
        return mine;
    }
    lane_keys[lane] = mine.key;
    lane_tails[lane] = static_cast<uint32_t>(mine.tail);
    __syncwarp();
    Fp32ArgCandidate reduced = fp32ArgIdentity();
    ready = subgroup_lane < k;
    result_component = subgroup_lane;
    if (ready) {
        const uint32_t partial_base = subgroup_lane_base + subgroup_lane * workers;
        for (uint32_t source_worker = 0; source_worker < workers; ++source_worker) {
            const uint32_t key = lane_keys[partial_base + source_worker];
            if (key != FP32_ARG_KEY_IDENTITY) {
                reduced = fp32ArgMin(reduced, Fp32ArgCandidate{key, lane_tails[partial_base + source_worker]});
            }
        }
    }
    __syncwarp();
    return reduced;
}

template <int OutputsPerWarp, int MaxComponentsPerLane, bool LoadAhead>
__global__ __launch_bounds__(SHALLOW_FP8_THREADS) void fp32OneWindowArgKernel(const float* input,
                                                                             Fp32ArgOutputs outputs,
                                                                             uint32_t outer_size,
                                                                             uint32_t reduction_size,
                                                                             uint32_t warps_per_output,
                                                                             Fp8OneWindowRuntimeKGeometry geometry) {
    static_assert(OutputsPerWarp == 1 || OutputsPerWarp == 2 || OutputsPerWarp == 4 || OutputsPerWarp == 8);
    static_assert(MaxComponentsPerLane == 1 || MaxComponentsPerLane == 2 || MaxComponentsPerLane == 4);
    constexpr uint32_t PacketWords = 4;
    constexpr uint32_t LanesPerOutput = TILED_REDUCTION_WARP_THREADS / OutputsPerWarp;
    constexpr bool MultipleComponentsPerLane = MaxComponentsPerLane > 1;
    constexpr uint32_t MaxK = LanesPerOutput * static_cast<uint32_t>(MaxComponentsPerLane);
    constexpr uint32_t LanePartialColumns = MultipleComponentsPerLane ? 1U : TILED_REDUCTION_WARP_THREADS;
    constexpr uint32_t WarpPartialRows = OutputsPerWarp == 1 ? SHALLOW_FP8_WARPS : 1U;
    constexpr uint32_t WarpPartialColumns = OutputsPerWarp == 1 ? MaxK : 1U;

    __shared__ uint32_t staged_packet_words[SHALLOW_FP8_WARPS][PacketWords * TILED_REDUCTION_WARP_THREADS];
    __shared__ uint32_t lane_keys[SHALLOW_FP8_WARPS][LanePartialColumns];
    __shared__ uint32_t lane_tails[SHALLOW_FP8_WARPS][LanePartialColumns];
    __shared__ uint32_t warp_keys[WarpPartialRows][WarpPartialColumns];
    __shared__ uint32_t warp_tails[WarpPartialRows][WarpPartialColumns];

    const uint32_t k = geometry.k;
    const uint32_t workers = geometry.workers_per_component;
    const uint32_t warp = threadIdx.x / TILED_REDUCTION_WARP_THREADS;
    const uint32_t lane = threadIdx.x % TILED_REDUCTION_WARP_THREADS;
    const uint32_t outer_input_stride = reduction_size * k * 4U;  // bytes
    const uint32_t total_input_bytes = outer_size * outer_input_stride;
    const uint8_t* const input_bytes = reinterpret_cast<const uint8_t*>(input);
    const uint32_t output_slot = lane / LanesPerOutput;
    const uint32_t subgroup_lane = lane % LanesPerOutput;
    const uint32_t subgroup_lane_base = output_slot * LanesPerOutput;
    const uint32_t component = MultipleComponentsPerLane ? subgroup_lane : subgroup_lane / workers;
    const uint32_t worker = MultipleComponentsPerLane ? 0U : subgroup_lane - component * workers;

    uint32_t best_key[MaxComponentsPerLane];
    uint32_t best_tail[MaxComponentsPerLane];

    if constexpr (OutputsPerWarp > 1) {
        static_cast<void>(warps_per_output);
        static_cast<void>(warp_keys);
        static_cast<void>(warp_tails);
        const uint32_t global_warp = blockIdx.x * SHALLOW_FP8_WARPS + warp;
        const uint32_t grid_warps = gridDim.x * SHALLOW_FP8_WARPS;
        for (uint32_t warp_group = global_warp;; warp_group += grid_warps) {
            const uint64_t outer_begin = static_cast<uint64_t>(warp_group) * OutputsPerWarp;
            if (outer_begin >= outer_size) {
                return;
            }
            const uint64_t outer_index = outer_begin + output_slot;
            const bool valid_output = outer_index < outer_size;
            streamOneWindowFp32ArgOutput<OutputsPerWarp, MaxComponentsPerLane, LoadAhead>(
                best_key, best_tail, staged_packet_words[warp], input_bytes, total_input_bytes, valid_output,
                valid_output ? static_cast<uint32_t>(outer_index) * outer_input_stride : 0U, 0U,
                geometry.rows_per_group, reduction_size, geometry, lane, subgroup_lane, subgroup_lane_base, component,
                worker, outputs.xor_mask);
            if constexpr (MultipleComponentsPerLane) {
#pragma unroll
                for (uint32_t slot = 0; slot < static_cast<uint32_t>(MaxComponentsPerLane); ++slot) {
                    const uint32_t owned_component = subgroup_lane + slot * LanesPerOutput;
                    if (valid_output && owned_component < k) {
                        storeFp32ArgResult(outputs, outer_index * k + owned_component,
                                           Fp32ArgCandidate{best_key[slot], best_tail[slot]});
                    }
                }
            } else {
                const Fp32ArgCandidate mine = (valid_output && subgroup_lane < geometry.task_count)
                                                  ? Fp32ArgCandidate{best_key[0], best_tail[0]}
                                                  : fp32ArgIdentity();
                bool ready = false;
                uint32_t result_component = 0;
                const Fp32ArgCandidate reduced = foldOneWindowFp32ArgWorkers<LanesPerOutput>(
                    mine, lane_keys[warp], lane_tails[warp], lane, subgroup_lane, subgroup_lane_base, component,
                    worker, k, workers, ready, result_component);
                if (valid_output && ready) {
                    storeFp32ArgResult(outputs, outer_index * k + result_component, reduced);
                }
            }
        }
    } else {
        // O1: W1 stays warp-local; W2/W4/W8 take disjoint row groups and fold in shared memory.
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
            streamOneWindowFp32ArgOutput<OutputsPerWarp, MaxComponentsPerLane, LoadAhead>(
                best_key, best_tail, staged_packet_words[warp], input_bytes, total_input_bytes, valid_output,
                valid_output ? static_cast<uint32_t>(outer_index) * outer_input_stride : 0U,
                warp_in_output * geometry.rows_per_group, rows_per_output_round, reduction_size, geometry, lane,
                subgroup_lane, subgroup_lane_base, component, worker, outputs.xor_mask);
            if constexpr (MultipleComponentsPerLane) {
#pragma unroll
                for (uint32_t slot = 0; slot < static_cast<uint32_t>(MaxComponentsPerLane); ++slot) {
                    const uint32_t owned_component = lane + slot * TILED_REDUCTION_WARP_THREADS;
                    if (owned_component < k) {
                        if (warps_per_output == 1) {
                            if (valid_output) {
                                storeFp32ArgResult(outputs, outer_index * k + owned_component,
                                                   Fp32ArgCandidate{best_key[slot], best_tail[slot]});
                            }
                        } else {
                            warp_keys[warp][owned_component] = valid_output ? best_key[slot] : FP32_ARG_KEY_IDENTITY;
                            warp_tails[warp][owned_component] = best_tail[slot];
                        }
                    }
                }
            } else {
                const Fp32ArgCandidate mine = (valid_output && lane < geometry.task_count)
                                                  ? Fp32ArgCandidate{best_key[0], best_tail[0]}
                                                  : fp32ArgIdentity();
                bool ready = false;
                uint32_t result_component = 0;
                const Fp32ArgCandidate reduced = foldOneWindowFp32ArgWorkers<LanesPerOutput>(
                    mine, lane_keys[warp], lane_tails[warp], lane, subgroup_lane, subgroup_lane_base, component,
                    worker, k, workers, ready, result_component);
                if (ready) {
                    if (warps_per_output == 1) {
                        if (valid_output) storeFp32ArgResult(outputs, outer_index * k + result_component, reduced);
                    } else {
                        warp_keys[warp][result_component] = valid_output ? reduced.key : FP32_ARG_KEY_IDENTITY;
                        warp_tails[warp][result_component] = static_cast<uint32_t>(reduced.tail);
                    }
                }
            }
            if (warps_per_output == 1) {
                continue;
            }
            __syncthreads();
            if (warp_in_output == 0) {
#pragma unroll
                for (uint32_t slot = 0; slot < static_cast<uint32_t>(MaxComponentsPerLane); ++slot) {
                    const uint32_t owned_component = lane + slot * TILED_REDUCTION_WARP_THREADS;
                    if (valid_output && owned_component < k) {
                        Fp32ArgCandidate reduced = fp32ArgIdentity();
                        for (uint32_t source_warp = 0; source_warp < warps_per_output; ++source_warp) {
                            const uint32_t source = first_warp_in_output + source_warp;
                            const uint32_t key = warp_keys[source][owned_component];
                            if (key != FP32_ARG_KEY_IDENTITY) {
                                reduced = fp32ArgMin(reduced,
                                                     Fp32ArgCandidate{key, warp_tails[source][owned_component]});
                            }
                        }
                        storeFp32ArgResult(outputs, outer_index * k + owned_component, reduced);
                    }
                }
            }
            __syncthreads();
        }
    }
}

// 1212: the carried-index OneWindow ARG twin (phase 8; see Fp32ArgWidePartials): fp32OneWindowArgKernel's units,
// windows and merge topology, staging each lane's carried index packet (the same elements as its value packet,
// plane-major) next to the values. An element's tail is (carried + row * row_stride) << 1 | negative zero, and every
// merge is the full (key, tail) minimum (composed indices are not monotone in the row). Complete only; the input and
// the carried index are 16-byte aligned and hold at most 2^32 - 1 value bytes.
template <typename IndexT>
__device__ __forceinline__ PackedInputValues<IndexT, 4> loadOneWindowCarriedPacket(const IndexT* carried,
                                                                                  uint32_t element_begin,
                                                                                  uint32_t total_elements) {
    if (total_elements >= 4U && element_begin <= total_elements - 4U)
        return loadVectorizedInputPacket<IndexT, 4>(carried + element_begin);
    PackedInputValues<IndexT, 4> packet{};
#pragma unroll
    for (uint32_t item = 0; item < 4U; ++item) {
        if (element_begin + item < total_elements) packet.values[item] = carried[element_begin + item];
    }
    return packet;
}

// Plane-major, as stageOneWindowPacket: element w of lane's packet goes to warp_indices[w * 32 + lane].
template <typename IndexT>
__device__ __forceinline__ void stageOneWindowCarriedPacket(IndexT* warp_indices, uint32_t lane,
                                                            const PackedInputValues<IndexT, 4>& packet) {
#pragma unroll
    for (uint32_t item = 0; item < 4U; ++item) {
        warp_indices[item * TILED_REDUCTION_WARP_THREADS + lane] = packet.values[item];
    }
}

template <int OutputsPerWarp, int MaxComponentsPerLane, typename IndexT>
__device__ __forceinline__ void keyOneWindowFp32CarriedArgGroup(uint32_t (&best_key)[MaxComponentsPerLane],
                                                                uint64_t (&best_tail)[MaxComponentsPerLane],
                                                                const uint32_t* warp_words,
                                                                const IndexT* warp_indices,
                                                         uint32_t subgroup_lane,
                                                         uint32_t subgroup_lane_base,
                                                         uint32_t component,
                                                         uint32_t worker,
                                                         uint32_t k,
                                                         uint32_t workers,
                                                         uint32_t group_row,
                                                         uint32_t group_rows,
                                                         uint32_t prefix_bytes,
                                                         uint32_t xor_mask,
                                                         uint64_t row_stride_index) {
    constexpr uint32_t LanesPerOutput = TILED_REDUCTION_WARP_THREADS / OutputsPerWarp;
    const auto key_element = [&](uint32_t slot, uint32_t element, uint32_t row) {
        const uint32_t staged = (element % 4U) * TILED_REDUCTION_WARP_THREADS + subgroup_lane_base + element / 4U;
        uint32_t negative_zero;
        const uint32_t key = fp32ArgKey(warp_words[staged], xor_mask, negative_zero);
        const uint64_t composed = static_cast<uint64_t>(warp_indices[staged]) +
                                  static_cast<uint64_t>(group_row + row) * row_stride_index;
        fp32ArgKeepMin(best_key[slot], best_tail[slot], key, (composed << 1) | negative_zero);
    };
    const uint32_t first_element = prefix_bytes / 4U;
    if constexpr (MaxComponentsPerLane <= 2) {
        constexpr bool MultipleComponentsPerLane = MaxComponentsPerLane > 1;
        const uint32_t row_stride = workers * k;
#pragma unroll
        for (uint32_t slot = 0; slot < static_cast<uint32_t>(MaxComponentsPerLane); ++slot) {
            const uint32_t owned_component =
                MultipleComponentsPerLane ? subgroup_lane + slot * LanesPerOutput : component;
            if (owned_component >= k) continue;
            uint32_t element = first_element + worker * k + owned_component;
            for (uint32_t row = worker; row < group_rows; row += workers, element += row_stride)
                key_element(slot, element, row);
        }
    } else {
        static_cast<void>(component);
        static_cast<void>(worker);
        static_cast<void>(workers);
        for (uint32_t row = 0; row < group_rows; ++row) {
            const uint32_t row_element = first_element + row * k + subgroup_lane;
#pragma unroll
            for (uint32_t slot = 0; slot < static_cast<uint32_t>(MaxComponentsPerLane); ++slot) {
                if (subgroup_lane + slot * LanesPerOutput < k)
                    key_element(slot, row_element + slot * LanesPerOutput, row);
            }
        }
    }
}

// Streams one output's R through packet windows first_row, first_row + row_step, ... into the lane's candidates
// (identity first). Force-inlined like streamOneWindowRuntimeKOutput: no lambda holds the candidate arrays.
template <int OutputsPerWarp, int MaxComponentsPerLane, bool LoadAhead, typename IndexT>
__device__ __forceinline__ void streamOneWindowFp32CarriedArgOutput(uint32_t (&best_key)[MaxComponentsPerLane],
                                                                    uint64_t (&best_tail)[MaxComponentsPerLane],
                                                                    uint32_t* warp_words,
                                                                    IndexT* warp_indices,
                                                                    const IndexT* carried,
                                                                    uint64_t row_stride_index,
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
                                                             uint32_t xor_mask) {
    constexpr uint32_t PacketBytes = 16;
#pragma unroll
    for (uint32_t slot = 0; slot < static_cast<uint32_t>(MaxComponentsPerLane); ++slot) {
        best_key[slot] = FP32_ARG_KEY_IDENTITY;
        best_tail[slot] = FP32_ARG_TAIL_IDENTITY;
    }
    const bool keys = valid_output && subgroup_lane < geometry.task_count;
    const uint32_t total_elements = total_input_bytes / 4U;
    // The carried index packet of a value packet: the same elements (byte offset / 4), the same tail handling.
    const auto load_indices = [&](uint32_t packet_byte_begin) {
        return loadOneWindowCarriedPacket<IndexT>(carried, packet_byte_begin / 4U, total_elements);
    };
    if constexpr (LoadAhead) {
        uint32_t group_row = first_row;
        OneWindowGroupWindow current{};
        RawVectorPacket<PacketBytes> current_packet{};
        PackedInputValues<IndexT, 4> current_indices{};
        if (group_row < reduction_size) {
            current = oneWindowRuntimeKGroupWindow<PacketBytes, 4U>(valid_output, outer_byte_begin, group_row,
                                                                    reduction_size, geometry.k,
                                                                    geometry.rows_per_group);
            if (valid_output && subgroup_lane < current.packet_count) {
                current_packet = loadOneWindowPacket<PacketBytes>(
                    input_bytes, current.aligned_byte_begin + subgroup_lane * PacketBytes, total_input_bytes);
                current_indices = load_indices(current.aligned_byte_begin + subgroup_lane * PacketBytes);
            }
        }
        while (group_row < reduction_size) {
            const uint32_t next_row = group_row + row_step;
            OneWindowGroupWindow next{};
            RawVectorPacket<PacketBytes> next_packet{};
            PackedInputValues<IndexT, 4> next_indices{};
            if (next_row < reduction_size) {
                next = oneWindowRuntimeKGroupWindow<PacketBytes, 4U>(valid_output, outer_byte_begin, next_row,
                                                                     reduction_size, geometry.k,
                                                                     geometry.rows_per_group);
                if (valid_output && subgroup_lane < next.packet_count) {
                    next_packet = loadOneWindowPacket<PacketBytes>(
                        input_bytes, next.aligned_byte_begin + subgroup_lane * PacketBytes, total_input_bytes);
                    next_indices = load_indices(next.aligned_byte_begin + subgroup_lane * PacketBytes);
                }
            }
            if (valid_output && subgroup_lane < current.packet_count) {
                stageOneWindowPacket<PacketBytes>(warp_words, lane, current_packet);
                stageOneWindowCarriedPacket<IndexT>(warp_indices, lane, current_indices);
            }
            __syncwarp();
            if (keys) {
                keyOneWindowFp32CarriedArgGroup<OutputsPerWarp, MaxComponentsPerLane, IndexT>(
                    best_key, best_tail, warp_words, warp_indices, subgroup_lane, subgroup_lane_base, component,
                    worker, geometry.k, geometry.workers_per_component, group_row, current.rows, current.prefix_bytes,
                    xor_mask, row_stride_index);
            }
            __syncwarp();
            current = next;
            current_packet = next_packet;
            current_indices = next_indices;
            group_row = next_row;
        }
    } else {
        for (uint32_t group_row = first_row; group_row < reduction_size; group_row += row_step) {
            const OneWindowGroupWindow window = oneWindowRuntimeKGroupWindow<PacketBytes, 4U>(
                valid_output, outer_byte_begin, group_row, reduction_size, geometry.k, geometry.rows_per_group);
            if (valid_output && subgroup_lane < window.packet_count) {
                stageOneWindowPacket<PacketBytes>(
                    warp_words, lane,
                    loadOneWindowPacket<PacketBytes>(input_bytes,
                                                     window.aligned_byte_begin + subgroup_lane * PacketBytes,
                                                     total_input_bytes));
                stageOneWindowCarriedPacket<IndexT>(
                    warp_indices, lane, load_indices(window.aligned_byte_begin + subgroup_lane * PacketBytes));
            }
            __syncwarp();
            if (keys) {
                keyOneWindowFp32CarriedArgGroup<OutputsPerWarp, MaxComponentsPerLane, IndexT>(
                    best_key, best_tail, warp_words, warp_indices, subgroup_lane, subgroup_lane_base, component,
                    worker, geometry.k, geometry.workers_per_component, group_row, window.rows, window.prefix_bytes,
                    xor_mask, row_stride_index);
            }
            __syncwarp();
        }
    }
}

// One-component buckets: folds a component's R workers (consecutive lanes component * workers + worker). Power-of-two
// counts use butterfly shuffles (the result is valid in the worker-0 lane, component `component`); other counts fold
// through the warp's shared lane words (valid in lane subgroup_lane < K, component subgroup_lane). Warp-uniform.
// Returns the folded candidate; `ready` and `result_component` say which lanes hold one.
template <uint32_t LanesPerOutput>
__device__ __forceinline__ Fp32ArgCandidate foldOneWindowFp32CarriedArgWorkers(Fp32ArgCandidate mine,
                                                                        uint32_t* lane_keys,
                                                                               uint64_t* lane_tails,
                                                                        uint32_t lane,
                                                                        uint32_t subgroup_lane,
                                                                        uint32_t subgroup_lane_base,
                                                                        uint32_t component,
                                                                        uint32_t worker,
                                                                        uint32_t k,
                                                                        uint32_t workers,
                                                                        bool& ready,
                                                                        uint32_t& result_component) {
    if ((workers & (workers - 1U)) == 0U) {
#pragma unroll
        for (uint32_t offset = 1; offset < LanesPerOutput; offset <<= 1) {
            if (offset < workers) mine = fp32ArgMin(mine, fp32ArgShuffleXor(mine, static_cast<int>(offset)));
        }
        ready = worker == 0U && component < k;
        result_component = component;
        return mine;
    }
    lane_keys[lane] = mine.key;
    lane_tails[lane] = mine.tail;
    __syncwarp();
    Fp32ArgCandidate reduced = fp32ArgIdentity();
    ready = subgroup_lane < k;
    result_component = subgroup_lane;
    if (ready) {
        const uint32_t partial_base = subgroup_lane_base + subgroup_lane * workers;
        for (uint32_t source_worker = 0; source_worker < workers; ++source_worker) {
            const uint32_t key = lane_keys[partial_base + source_worker];
            if (key != FP32_ARG_KEY_IDENTITY) {
                reduced = fp32ArgMin(reduced, Fp32ArgCandidate{key, lane_tails[partial_base + source_worker]});
            }
        }
    }
    __syncwarp();
    return reduced;
}

template <int OutputsPerWarp, int MaxComponentsPerLane, bool LoadAhead, typename IndexT>
__global__ __launch_bounds__(SHALLOW_FP8_THREADS) void fp32OneWindowCarriedArgKernel(const float* input,
                                                                                    const IndexT* carried,
                                                                                    uint64_t row_stride_index,
                                                                                    Fp32ArgOutputs outputs,
                                                                             uint32_t outer_size,
                                                                             uint32_t reduction_size,
                                                                             uint32_t warps_per_output,
                                                                             Fp8OneWindowRuntimeKGeometry geometry) {
    static_assert(OutputsPerWarp == 1 || OutputsPerWarp == 2 || OutputsPerWarp == 4 || OutputsPerWarp == 8);
    static_assert(MaxComponentsPerLane == 1 || MaxComponentsPerLane == 2 || MaxComponentsPerLane == 4);
    static_assert(std::is_same_v<IndexT, uint32_t> || std::is_same_v<IndexT, uint64_t>);
    constexpr uint32_t PacketWords = 4;
    constexpr uint32_t LanesPerOutput = TILED_REDUCTION_WARP_THREADS / OutputsPerWarp;
    constexpr bool MultipleComponentsPerLane = MaxComponentsPerLane > 1;
    constexpr uint32_t MaxK = LanesPerOutput * static_cast<uint32_t>(MaxComponentsPerLane);
    constexpr uint32_t LanePartialColumns = MultipleComponentsPerLane ? 1U : TILED_REDUCTION_WARP_THREADS;
    constexpr uint32_t WarpPartialRows = OutputsPerWarp == 1 ? SHALLOW_FP8_WARPS : 1U;
    constexpr uint32_t WarpPartialColumns = OutputsPerWarp == 1 ? MaxK : 1U;

    __shared__ uint32_t staged_packet_words[SHALLOW_FP8_WARPS][PacketWords * TILED_REDUCTION_WARP_THREADS];
    __shared__ IndexT staged_indices[SHALLOW_FP8_WARPS][PacketWords * TILED_REDUCTION_WARP_THREADS];
    __shared__ uint32_t lane_keys[SHALLOW_FP8_WARPS][LanePartialColumns];
    __shared__ uint64_t lane_tails[SHALLOW_FP8_WARPS][LanePartialColumns];
    __shared__ uint32_t warp_keys[WarpPartialRows][WarpPartialColumns];
    __shared__ uint64_t warp_tails[WarpPartialRows][WarpPartialColumns];

    const uint32_t k = geometry.k;
    const uint32_t workers = geometry.workers_per_component;
    const uint32_t warp = threadIdx.x / TILED_REDUCTION_WARP_THREADS;
    const uint32_t lane = threadIdx.x % TILED_REDUCTION_WARP_THREADS;
    const uint32_t outer_input_stride = reduction_size * k * 4U;  // bytes
    const uint32_t total_input_bytes = outer_size * outer_input_stride;
    const uint8_t* const input_bytes = reinterpret_cast<const uint8_t*>(input);
    const uint32_t output_slot = lane / LanesPerOutput;
    const uint32_t subgroup_lane = lane % LanesPerOutput;
    const uint32_t subgroup_lane_base = output_slot * LanesPerOutput;
    const uint32_t component = MultipleComponentsPerLane ? subgroup_lane : subgroup_lane / workers;
    const uint32_t worker = MultipleComponentsPerLane ? 0U : subgroup_lane - component * workers;

    uint32_t best_key[MaxComponentsPerLane];
    uint64_t best_tail[MaxComponentsPerLane];

    if constexpr (OutputsPerWarp > 1) {
        static_cast<void>(warps_per_output);
        static_cast<void>(warp_keys);
        static_cast<void>(warp_tails);
        const uint32_t global_warp = blockIdx.x * SHALLOW_FP8_WARPS + warp;
        const uint32_t grid_warps = gridDim.x * SHALLOW_FP8_WARPS;
        for (uint32_t warp_group = global_warp;; warp_group += grid_warps) {
            const uint64_t outer_begin = static_cast<uint64_t>(warp_group) * OutputsPerWarp;
            if (outer_begin >= outer_size) {
                return;
            }
            const uint64_t outer_index = outer_begin + output_slot;
            const bool valid_output = outer_index < outer_size;
            streamOneWindowFp32CarriedArgOutput<OutputsPerWarp, MaxComponentsPerLane, LoadAhead, IndexT>(
                best_key, best_tail, staged_packet_words[warp], staged_indices[warp], carried, row_stride_index,
                input_bytes, total_input_bytes, valid_output,
                valid_output ? static_cast<uint32_t>(outer_index) * outer_input_stride : 0U, 0U,
                geometry.rows_per_group, reduction_size, geometry, lane, subgroup_lane, subgroup_lane_base, component,
                worker, outputs.xor_mask);
            if constexpr (MultipleComponentsPerLane) {
#pragma unroll
                for (uint32_t slot = 0; slot < static_cast<uint32_t>(MaxComponentsPerLane); ++slot) {
                    const uint32_t owned_component = subgroup_lane + slot * LanesPerOutput;
                    if (valid_output && owned_component < k) {
                        storeFp32ArgResult(outputs, outer_index * k + owned_component,
                                           Fp32ArgCandidate{best_key[slot], best_tail[slot]});
                    }
                }
            } else {
                const Fp32ArgCandidate mine = (valid_output && subgroup_lane < geometry.task_count)
                                                  ? Fp32ArgCandidate{best_key[0], best_tail[0]}
                                                  : fp32ArgIdentity();
                bool ready = false;
                uint32_t result_component = 0;
                const Fp32ArgCandidate reduced = foldOneWindowFp32CarriedArgWorkers<LanesPerOutput>(
                    mine, lane_keys[warp], lane_tails[warp], lane, subgroup_lane, subgroup_lane_base, component,
                    worker, k, workers, ready, result_component);
                if (valid_output && ready) {
                    storeFp32ArgResult(outputs, outer_index * k + result_component, reduced);
                }
            }
        }
    } else {
        // O1: W1 stays warp-local; W2/W4/W8 take disjoint row groups and fold in shared memory.
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
            streamOneWindowFp32CarriedArgOutput<OutputsPerWarp, MaxComponentsPerLane, LoadAhead, IndexT>(
                best_key, best_tail, staged_packet_words[warp], staged_indices[warp], carried, row_stride_index,
                input_bytes, total_input_bytes, valid_output,
                valid_output ? static_cast<uint32_t>(outer_index) * outer_input_stride : 0U,
                warp_in_output * geometry.rows_per_group, rows_per_output_round, reduction_size, geometry, lane,
                subgroup_lane, subgroup_lane_base, component, worker, outputs.xor_mask);
            if constexpr (MultipleComponentsPerLane) {
#pragma unroll
                for (uint32_t slot = 0; slot < static_cast<uint32_t>(MaxComponentsPerLane); ++slot) {
                    const uint32_t owned_component = lane + slot * TILED_REDUCTION_WARP_THREADS;
                    if (owned_component < k) {
                        if (warps_per_output == 1) {
                            if (valid_output) {
                                storeFp32ArgResult(outputs, outer_index * k + owned_component,
                                                   Fp32ArgCandidate{best_key[slot], best_tail[slot]});
                            }
                        } else {
                            warp_keys[warp][owned_component] = valid_output ? best_key[slot] : FP32_ARG_KEY_IDENTITY;
                            warp_tails[warp][owned_component] = best_tail[slot];
                        }
                    }
                }
            } else {
                const Fp32ArgCandidate mine = (valid_output && lane < geometry.task_count)
                                                  ? Fp32ArgCandidate{best_key[0], best_tail[0]}
                                                  : fp32ArgIdentity();
                bool ready = false;
                uint32_t result_component = 0;
                const Fp32ArgCandidate reduced = foldOneWindowFp32CarriedArgWorkers<LanesPerOutput>(
                    mine, lane_keys[warp], lane_tails[warp], lane, subgroup_lane, subgroup_lane_base, component,
                    worker, k, workers, ready, result_component);
                if (ready) {
                    if (warps_per_output == 1) {
                        if (valid_output) storeFp32ArgResult(outputs, outer_index * k + result_component, reduced);
                    } else {
                        warp_keys[warp][result_component] = valid_output ? reduced.key : FP32_ARG_KEY_IDENTITY;
                        warp_tails[warp][result_component] = reduced.tail;
                    }
                }
            }
            if (warps_per_output == 1) {
                continue;
            }
            __syncthreads();
            if (warp_in_output == 0) {
#pragma unroll
                for (uint32_t slot = 0; slot < static_cast<uint32_t>(MaxComponentsPerLane); ++slot) {
                    const uint32_t owned_component = lane + slot * TILED_REDUCTION_WARP_THREADS;
                    if (valid_output && owned_component < k) {
                        Fp32ArgCandidate reduced = fp32ArgIdentity();
                        for (uint32_t source_warp = 0; source_warp < warps_per_output; ++source_warp) {
                            const uint32_t source = first_warp_in_output + source_warp;
                            const uint32_t key = warp_keys[source][owned_component];
                            if (key != FP32_ARG_KEY_IDENTITY) {
                                reduced = fp32ArgMin(reduced,
                                                     Fp32ArgCandidate{key, warp_tails[source][owned_component]});
                            }
                        }
                        storeFp32ArgResult(outputs, outer_index * k + owned_component, reduced);
                    }
                }
            }
            __syncthreads();
        }
    }
}

// ARG-A2: OneWindow argmin/argmax twins (one kernel per OneWindow unit; the runtime Fp8ArgFold covers argmin/argmax x
// E4M3/E5M2). Same packet windows, staging, lane ownership, R workers and merge topology as
// fp8OneWindowRuntimeKCompleteKernel, with one uint32 lane candidate per owned component (Fp8ArgKey.h) in place of an
// FP32 accumulator. Candidates carry the absolute row, so lanes, workers and warps merge with a plain unsigned min;
// the launcher requires R < FP8_ARG_MAX_LANE_ROWS. Templates only: kernels are instantiated by the exported launcher
// after this anonymous namespace (the FP8 VALUE inventory TU compiles this prefix and must not gain ARG kernels).

__device__ __forceinline__ float decodeFp8ArgByte(uint32_t byte, bool e5m2) {
#if THOR_CUB_ENABLE_FP8_TYPES
    if (e5m2) {
        __nv_fp8_e5m2 value;
        value.__x = static_cast<__nv_fp8_storage_t>(byte);
        return static_cast<float>(value);
    }
    __nv_fp8_e4m3 value;
    value.__x = static_cast<__nv_fp8_storage_t>(byte);
    return static_cast<float>(value);
#else
    static_cast<void>(byte);
    static_cast<void>(e5m2);
    return 0.0f;
#endif
}

struct Fp8ArgOutputs {
    void* value;  // null: no value output
    DataType value_dtype;
    void* index;  // null: no index output
    DataType index_dtype;
    uint64_t domain_stride;  // the reduced run's original ARG-domain stride: index = row * domain_stride
    Fp8ArgFold fold;
    bool e5m2;
    // ARG-A3g (1116): staged passes may instead write the raw shard-relative lane candidate, 4 bytes per partial
    // (key << 24 | shard row << 1 | negative zero); fp8ArgPackedContinuationKernel finishes the reduction.
    uint32_t* packed = nullptr;
};

// MayPack: only the staged kernels (DeepRSmallK, rotated RuntimeK) test `packed`; Complete kernels (OneWindow,
// PacketComplete) compile without the branch (1116 measured the OneWindow O1 c16 twin 1.12x slower with it, 63 -> 64
// registers).
template <bool MayPack = false>
__device__ __forceinline__ void storeFp8ArgResult(const Fp8ArgOutputs& outputs, uint64_t output_index,
                                                  uint32_t candidate, uint64_t row_base = 0) {
    if constexpr (MayPack) {
        if (outputs.packed != nullptr) {
            outputs.packed[output_index] = candidate;
            return;
        }
    }
    if (outputs.value != nullptr) {
        const uint32_t byte = fp8ArgValueByte(candidate >> 24, candidate & 1u, outputs.fold);
        storeFp32AsRuntimeDType(outputs.value, outputs.value_dtype, output_index,
                                decodeFp8ArgByte(byte, outputs.e5m2));
    }
    if (outputs.index != nullptr) {
        const uint64_t index =
            (row_base + ((candidate >> 1) & (FP8_ARG_MAX_LANE_ROWS - 1u))) * outputs.domain_stride;
        if (outputs.index_dtype == DataType::UINT32) {
            static_cast<uint32_t*>(outputs.index)[output_index] = static_cast<uint32_t>(index);
        } else {
            static_cast<uint64_t*>(outputs.index)[output_index] = index;
        }
    }
}

// ARG-A3i (1126): linear staging for the O1 ARG units with one or two components per lane. The O1 subgroup is the
// whole warp, so lane's packet goes to bytes [lane * PacketBytes, +PacketBytes) of the warp's slice (one vector store)
// and staged byte b is simply byte b (one LDS.U8, no plane arithmetic). Plane-major staging cost ~10 SASS
// instructions per gathered byte (shift, mask, plane index, word load, extract), ~41 of the 72 per four rows; the
// O1 reads stay conflict-free (lanes read at most 32 contiguous bytes per row step).
// 1128: every c1/c2 unit stages linearly. Subgroup s's slice starts at s * SliceBytes; O2/O4/O8 pad each slice by
// LanesPerOutput words, so consecutive slices start 16/8/4 banks apart (5 * LanesPerOutput words mod 32) and the
// subgroups' reads (at most LanesPerOutput contiguous bytes per row step) never share a bank. O1 c4/c8 also stage
// linearly and own quads of adjacent components (ONE_WINDOW_ARG_QUAD_OWNERSHIP); their two-word reads may touch one
// word past the window, so O1 slices carry a 4-word pad.
// 1129: also O2/O4/O8 c4 (one quad of 4 * LanesPerOutput components). O2/O4 c8 (47 registers, next to the 49 cliff)
// and O1 c16 (63, next to 64) keep lane + slot * LanesPerOutput ownership.
template <uint32_t LanesPerOutput, int MaxComponentsPerLane>
inline constexpr bool ONE_WINDOW_ARG_QUAD_OWNERSHIP =
    MaxComponentsPerLane == 4 || (LanesPerOutput == TILED_REDUCTION_WARP_THREADS && MaxComponentsPerLane == 8);

template <uint32_t LanesPerOutput, int MaxComponentsPerLane>
inline constexpr bool ONE_WINDOW_ARG_LINEAR_STAGE =
    MaxComponentsPerLane <= 2 || ONE_WINDOW_ARG_QUAD_OWNERSHIP<LanesPerOutput, MaxComponentsPerLane>;

template <uint32_t LanesPerOutput>
inline constexpr uint32_t ONE_WINDOW_ARG_SLICE_PAD_WORDS =
    LanesPerOutput == TILED_REDUCTION_WARP_THREADS ? 4U : LanesPerOutput;

template <uint32_t LanesPerOutput, uint32_t PacketBytes>
inline constexpr uint32_t ONE_WINDOW_ARG_SLICE_BYTES =
    LanesPerOutput * PacketBytes + 4U * ONE_WINDOW_ARG_SLICE_PAD_WORDS<LanesPerOutput>;

// Words per warp of the ARG staging buffer (both layouts fit).
template <uint32_t LanesPerOutput, uint32_t PacketBytes>
inline constexpr uint32_t ONE_WINDOW_ARG_STAGE_WORDS =
    TILED_REDUCTION_WARP_THREADS / LanesPerOutput * ONE_WINDOW_ARG_SLICE_BYTES<LanesPerOutput, PacketBytes> / 4U;

// The component a lane's slot owns: lane + slot * LanesPerOutput, or with quad ownership (c4, O1 c8) the four
// adjacent components q * 4 * LanesPerOutput + 4 * lane .. +3 for slots 4q .. 4q + 3.
template <uint32_t LanesPerOutput, int MaxComponentsPerLane>
__device__ __forceinline__ uint32_t oneWindowArgOwnedComponent(uint32_t subgroup_lane, uint32_t slot) {
    if constexpr (ONE_WINDOW_ARG_QUAD_OWNERSHIP<LanesPerOutput, MaxComponentsPerLane>) {
        return (slot / 4U) * (4U * LanesPerOutput) + 4U * subgroup_lane + slot % 4U;
    } else {
        return subgroup_lane + slot * LanesPerOutput;
    }
}

template <uint32_t LanesPerOutput, uint32_t PacketBytes>
__device__ __forceinline__ void stageOneWindowPacketLinear(uint32_t* warp_words,
                                                           uint32_t lane,
                                                           const RawVectorPacket<PacketBytes>& packet) {
    static_assert(PacketBytes == 16 || LanesPerOutput == TILED_REDUCTION_WARP_THREADS);
    constexpr uint32_t SlicePackets = ONE_WINDOW_ARG_SLICE_BYTES<LanesPerOutput, PacketBytes> / PacketBytes;
    static_assert(ONE_WINDOW_ARG_SLICE_BYTES<LanesPerOutput, PacketBytes> % PacketBytes == 0);
    const uint32_t index = (lane / LanesPerOutput) * SlicePackets + lane % LanesPerOutput;
    if constexpr (PacketBytes == 4) {
        warp_words[index] = packet.value;
    } else if constexpr (PacketBytes == 8) {
        reinterpret_cast<uint2*>(warp_words)[index] = packet.value;
    } else {
        reinterpret_cast<uint4*>(warp_words)[index] = packet.value;
    }
}

// One staged byte of the subgroup whose first lane is subgroup_lane_base, in either layout.
template <bool Linear, uint32_t LanesPerOutput, uint32_t PacketBytes>
__device__ __forceinline__ uint32_t oneWindowArgStagedByte(const uint32_t* warp_words,
                                                           uint32_t subgroup_lane_base,
                                                           uint32_t staged_byte) {
    if constexpr (Linear) {
        const uint32_t slice =
            (subgroup_lane_base / LanesPerOutput) * ONE_WINDOW_ARG_SLICE_BYTES<LanesPerOutput, PacketBytes>;
        return reinterpret_cast<const uint8_t*>(warp_words)[slice + staged_byte];
    } else {
        return oneWindowStagedByte<PacketBytes>(warp_words, subgroup_lane_base, staged_byte);
    }
}

// decodeOneWindowRuntimeKGroup with keys: the same byte reads and packing; group_row is the group's first row.
// RowsCandidate (1125): reduce each four-row word with fp8ArgRowsCandidate4 before building the lane candidate. 1124
// measured it 0.96x at O1 c2 load-ahead and 0.99x at c1, but 1.02-1.04x (up to 1.17x at large R) for the O2/O4/O8 c2
// units without load-ahead, so those keep the per-byte candidates.
template <int MaxComponentsPerLane, uint32_t LanesPerOutput, uint32_t PacketBytes, bool RowsCandidate>
__device__ __forceinline__ void keyOneWindowArgGroup(uint32_t (&local)[MaxComponentsPerLane],
                                                     const uint32_t* warp_words,
                                                     uint32_t subgroup_lane,
                                                     uint32_t subgroup_lane_base,
                                                     uint32_t component,
                                                     uint32_t worker,
                                                     uint32_t k,
                                                     uint32_t workers,
                                                     uint32_t group_row,
                                                     uint32_t group_rows,
                                                     uint32_t prefix_bytes,
                                                     const Fp8ArgFold& fold) {
    if constexpr (MaxComponentsPerLane <= 2) {
        constexpr bool MultipleComponentsPerLane = MaxComponentsPerLane > 1;
        constexpr bool Linear = ONE_WINDOW_ARG_LINEAR_STAGE<LanesPerOutput, MaxComponentsPerLane>;
        const uint32_t row_stride = workers * k;
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
                uint32_t packed;
                if constexpr (Linear) {
                    const auto* staged = reinterpret_cast<const uint8_t*>(warp_words) +
                                         (subgroup_lane_base / LanesPerOutput) *
                                             ONE_WINDOW_ARG_SLICE_BYTES<LanesPerOutput, PacketBytes> +
                                         byte;
                    const uint32_t b0 = staged[0];
                    const uint32_t b1 = staged[row_stride];
                    const uint32_t b2 = staged[2U * row_stride];
                    const uint32_t b3 = staged[3U * row_stride];
                    packed = fp8ArgBytePerm(fp8ArgBytePerm(b0, b1, 0x0040u), fp8ArgBytePerm(b2, b3, 0x0040u),
                                            0x5410u);  // b0 | b1 << 8 | b2 << 16 | b3 << 24
                } else {
                    packed = oneWindowStagedByte<PacketBytes>(warp_words, subgroup_lane_base, byte) |
                             (oneWindowStagedByte<PacketBytes>(warp_words, subgroup_lane_base, byte + row_stride)
                              << 8) |
                             (oneWindowStagedByte<PacketBytes>(warp_words, subgroup_lane_base,
                                                               byte + 2U * row_stride) << 16) |
                             (oneWindowStagedByte<PacketBytes>(warp_words, subgroup_lane_base,
                                                               byte + 3U * row_stride) << 24);
                }
                const Fp8ArgKeys4 keys = fp8ArgKeys4(packed, fold);
                if constexpr (RowsCandidate) {
                    // Four rows of this component: reduce the word first (fp8ArgRowsCandidate4, 1124).
                    local[slot] = fp8ArgMin(local[slot], fp8ArgRowsCandidate4(keys, group_row + row, workers));
                } else {
                    const uint32_t first = group_row + row;
#pragma unroll
                    for (uint32_t j = 0; j < 4U; ++j) {
                        local[slot] = fp8ArgMin(local[slot], fp8ArgLaneCandidate(keys, j, first + j * workers));
                    }
                }
            }
            if (row < group_rows) {
                uint32_t packed = 0;
#pragma unroll
                for (uint32_t item = 0; item < 4U; ++item) {
                    if (row + item * workers < group_rows) {
                        packed |= oneWindowArgStagedByte<Linear, LanesPerOutput, PacketBytes>(
                                      warp_words, subgroup_lane_base, byte + item * row_stride)
                                  << (item * 8U);
                    }
                }
                const Fp8ArgKeys4 keys = fp8ArgKeys4(packed, fold);
                const uint32_t first = group_row + row;
#pragma unroll
                for (uint32_t item = 0; item < 4U; ++item) {
                    if (row + item * workers < group_rows) {
                        local[slot] = fp8ArgMin(local[slot], fp8ArgLaneCandidate(keys, item, first + item * workers));
                    }
                }
            }
        }
    } else if constexpr (ONE_WINDOW_ARG_QUAD_OWNERSHIP<LanesPerOutput, MaxComponentsPerLane>) {
        // 1128: O1 c4/c8 (1129: and O2/O4/O8 c4), linear staging. Lane owns components q * 4 * LanesPerOutput +
        // 4 * lane .. +3: one key word is two LDS.32 and a funnel shift by the row's byte offset (subgroup-uniform).
        // The second word stays inside the slice's pad.
        static_cast<void>(component);
        static_cast<void>(worker);
        static_cast<void>(workers);
        constexpr uint32_t Quads = static_cast<uint32_t>(MaxComponentsPerLane) / 4U;
        const uint32_t slice_word = (subgroup_lane_base / LanesPerOutput) *
                                    (ONE_WINDOW_ARG_SLICE_BYTES<LanesPerOutput, PacketBytes> / 4U);
        for (uint32_t row = 0; row < group_rows; ++row) {
            const uint32_t row_byte = prefix_bytes + row * k;
            const uint32_t shift = (row_byte & 3U) * 8U;
            const uint32_t absolute_row = group_row + row;
#pragma unroll
            for (uint32_t quad = 0; quad < Quads; ++quad) {
                const uint32_t first = quad * 4U * LanesPerOutput + 4U * subgroup_lane;
                if (first < k) {
                    const uint32_t word = slice_word + ((row_byte + first) >> 2);
                    const uint32_t packed = __funnelshift_r(warp_words[word], warp_words[word + 1], shift);
                    const Fp8ArgKeys4 keys = fp8ArgKeys4(packed, fold);
#pragma unroll
                    for (uint32_t j = 0; j < 4U; ++j) {
                        if (first + j < k) {
                            local[quad * 4U + j] =
                                fp8ArgMin(local[quad * 4U + j], fp8ArgLaneCandidate(keys, j, absolute_row));
                        }
                    }
                }
            }
        }
    } else {
        static_cast<void>(component);
        static_cast<void>(worker);
        static_cast<void>(workers);
        for (uint32_t row = 0; row < group_rows; ++row) {
            const uint32_t row_byte = prefix_bytes + row * k + subgroup_lane;
            const uint32_t absolute_row = group_row + row;
#pragma unroll
            for (uint32_t slot_base = 0; slot_base < static_cast<uint32_t>(MaxComponentsPerLane); slot_base += 4U) {
                if (subgroup_lane + slot_base * LanesPerOutput >= k) {
                    continue;
                }
                uint32_t packed = 0;
#pragma unroll
                for (uint32_t j = 0; j < 4U; ++j) {
                    if (subgroup_lane + (slot_base + j) * LanesPerOutput < k) {
                        packed |= oneWindowStagedByte<PacketBytes>(warp_words, subgroup_lane_base,
                                                                   row_byte + (slot_base + j) * LanesPerOutput)
                                  << (j * 8U);
                    }
                }
                const Fp8ArgKeys4 keys = fp8ArgKeys4(packed, fold);
#pragma unroll
                for (uint32_t j = 0; j < 4U; ++j) {
                    if (subgroup_lane + (slot_base + j) * LanesPerOutput < k) {
                        local[slot_base + j] =
                            fp8ArgMin(local[slot_base + j], fp8ArgLaneCandidate(keys, j, absolute_row));
                    }
                }
            }
        }
    }
}

// streamOneWindowRuntimeKOutput with keyOneWindowArgGroup (kept separate so the VALUE kernel's code is untouched).
template <int MaxComponentsPerLane, uint32_t LanesPerOutput, uint32_t PacketBytes, bool LoadAhead>
__device__ __forceinline__ void streamOneWindowArgOutput(uint32_t (&local)[MaxComponentsPerLane],
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
                                                         const Fp8ArgFold& fold) {
    const bool keys = valid_output && subgroup_lane < geometry.task_count;
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
                if constexpr (ONE_WINDOW_ARG_LINEAR_STAGE<LanesPerOutput, MaxComponentsPerLane>) {
                    stageOneWindowPacketLinear<LanesPerOutput, PacketBytes>(warp_words, lane, current_packet);
                } else {
                    stageOneWindowPacket<PacketBytes>(warp_words, lane, current_packet);
                }
            }
            __syncwarp();
            if (keys) {
                keyOneWindowArgGroup<MaxComponentsPerLane, LanesPerOutput, PacketBytes, true>(
                    local, warp_words, subgroup_lane, subgroup_lane_base, component, worker, geometry.k,
                    geometry.workers_per_component, group_row, current.rows, current.prefix_bytes, fold);
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
                const RawVectorPacket<PacketBytes> packet = loadOneWindowPacket<PacketBytes>(
                    input_bytes, window.aligned_byte_begin + subgroup_lane * PacketBytes, total_input_bytes);
                if constexpr (ONE_WINDOW_ARG_LINEAR_STAGE<LanesPerOutput, MaxComponentsPerLane>) {
                    stageOneWindowPacketLinear<LanesPerOutput, PacketBytes>(warp_words, lane, packet);
                } else {
                    stageOneWindowPacket<PacketBytes>(warp_words, lane, packet);
                }
            }
            __syncwarp();
            if (keys) {
                keyOneWindowArgGroup<MaxComponentsPerLane, LanesPerOutput, PacketBytes, MaxComponentsPerLane == 1>(
                    local, warp_words, subgroup_lane, subgroup_lane_base, component, worker, geometry.k,
                    geometry.workers_per_component, group_row, window.rows, window.prefix_bytes, fold);
            }
            __syncwarp();
        }
    }
}

template <uint32_t LanesPerOutput>
__device__ __forceinline__ uint32_t shuffleFoldOneWindowArgWorkers(uint32_t partial, uint32_t workers) {
#pragma unroll
    for (uint32_t offset = 1; offset < LanesPerOutput; offset <<= 1) {
        if (offset < workers) {
            partial = fp8ArgMin(partial, __shfl_xor_sync(0xffffffffU, partial, offset));
        }
    }
    return partial;
}

template <int OutputsPerWarp, int PacketBytes, int MaxComponentsPerLane, bool LoadAhead>
__global__ void fp8OneWindowArgCompleteKernel(const uint8_t* input,
                                              Fp8ArgOutputs outputs,
                                              uint32_t outer_size,
                                              uint32_t reduction_size,
                                              uint32_t warps_per_output,
                                              Fp8OneWindowRuntimeKGeometry geometry) {
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
    constexpr uint32_t Identity = FP8_ARG_LANE_IDENTITY;

    constexpr uint32_t StageWords = ONE_WINDOW_ARG_LINEAR_STAGE<LanesPerOutput, MaxComponentsPerLane>
                                        ? ONE_WINDOW_ARG_STAGE_WORDS<LanesPerOutput, PacketBytes>
                                        : PacketWords * TILED_REDUCTION_WARP_THREADS;
    __shared__ __align__(16) uint32_t staged_packet_words[SHALLOW_FP8_WARPS][StageWords];
    __shared__ uint32_t lane_partials[SHALLOW_FP8_WARPS][LanePartialColumns];
    __shared__ uint32_t warp_partials[WarpPartialRows][WarpPartialColumns];

    const uint32_t k = geometry.k;
    const uint32_t workers = geometry.workers_per_component;
    const uint32_t warp = threadIdx.x / TILED_REDUCTION_WARP_THREADS;
    const uint32_t lane = threadIdx.x % TILED_REDUCTION_WARP_THREADS;
    const uint32_t outer_input_stride = reduction_size * k;
    const uint32_t total_input_bytes = outer_size * outer_input_stride;

    const uint32_t output_slot = lane / LanesPerOutput;
    const uint32_t subgroup_lane = lane % LanesPerOutput;
    const uint32_t subgroup_lane_base = output_slot * LanesPerOutput;
    const uint32_t component = MultipleComponentsPerLane ? subgroup_lane : subgroup_lane / workers;
    const uint32_t worker = MultipleComponentsPerLane ? 0U : subgroup_lane - component * workers;

    uint32_t local[MaxComponentsPerLane];

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
                local[slot] = Identity;
            }
            const uint32_t outer_byte_begin = valid_output ? static_cast<uint32_t>(outer_index) * outer_input_stride : 0;
            streamOneWindowArgOutput<MaxComponentsPerLane, LanesPerOutput, PacketBytes, LoadAhead>(
                local, staged_packet_words[warp], input, total_input_bytes, valid_output, outer_byte_begin, 0U,
                geometry.rows_per_group, reduction_size, geometry, lane, subgroup_lane, subgroup_lane_base, component,
                worker, outputs.fold);

            if constexpr (MultipleComponentsPerLane) {
#pragma unroll
                for (uint32_t slot = 0; slot < static_cast<uint32_t>(MaxComponentsPerLane); ++slot) {
                    const uint32_t owned_component =
                        oneWindowArgOwnedComponent<LanesPerOutput, MaxComponentsPerLane>(subgroup_lane, slot);
                    if (valid_output && owned_component < k) {
                        storeFp8ArgResult(outputs, outer_index * static_cast<uint64_t>(k) + owned_component,
                                          local[slot]);
                    }
                }
            } else if ((workers & (workers - 1U)) == 0U) {
                const uint32_t reduced = shuffleFoldOneWindowArgWorkers<LanesPerOutput>(
                    (valid_output && subgroup_lane < geometry.task_count) ? local[0] : Identity, workers);
                if (valid_output && worker == 0U && component < k) {
                    storeFp8ArgResult(outputs, outer_index * static_cast<uint64_t>(k) + component, reduced);
                }
            } else {
                lane_partials[warp][lane] = (valid_output && subgroup_lane < geometry.task_count) ? local[0] : Identity;
                __syncwarp();
                if (valid_output && subgroup_lane < k) {
                    uint32_t reduced = Identity;
                    const uint32_t partial_base = subgroup_lane_base + subgroup_lane * workers;
                    for (uint32_t source_worker = 0; source_worker < workers; ++source_worker) {
                        reduced = fp8ArgMin(reduced, lane_partials[warp][partial_base + source_worker]);
                    }
                    storeFp8ArgResult(outputs, outer_index * static_cast<uint64_t>(k) + subgroup_lane, reduced);
                }
                __syncwarp();
            }
        }
    } else {
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
                local[slot] = Identity;
            }
            const uint32_t outer_byte_begin = valid_output ? static_cast<uint32_t>(outer_index) * outer_input_stride : 0;
            streamOneWindowArgOutput<MaxComponentsPerLane, LanesPerOutput, PacketBytes, LoadAhead>(
                local, staged_packet_words[warp], input, total_input_bytes, valid_output, outer_byte_begin,
                warp_in_output * geometry.rows_per_group, rows_per_output_round, reduction_size, geometry, lane,
                subgroup_lane, subgroup_lane_base, component, worker, outputs.fold);

            if (warps_per_output == 1) {
                if constexpr (MultipleComponentsPerLane) {
#pragma unroll
                    for (uint32_t slot = 0; slot < static_cast<uint32_t>(MaxComponentsPerLane); ++slot) {
                        const uint32_t owned_component =
                            oneWindowArgOwnedComponent<LanesPerOutput, MaxComponentsPerLane>(lane, slot);
                        if (valid_output && owned_component < k) {
                            storeFp8ArgResult(outputs, outer_index * static_cast<uint64_t>(k) + owned_component,
                                              local[slot]);
                        }
                    }
                } else if ((workers & (workers - 1U)) == 0U) {
                    const uint32_t reduced = shuffleFoldOneWindowArgWorkers<LanesPerOutput>(
                        (valid_output && lane < geometry.task_count) ? local[0] : Identity, workers);
                    if (valid_output && worker == 0U && component < k) {
                        storeFp8ArgResult(outputs, outer_index * static_cast<uint64_t>(k) + component, reduced);
                    }
                } else {
                    lane_partials[warp][lane] = (valid_output && lane < geometry.task_count) ? local[0] : Identity;
                    __syncwarp();
                    if (valid_output && lane < k) {
                        uint32_t reduced = Identity;
                        const uint32_t partial_base = lane * workers;
                        for (uint32_t source_worker = 0; source_worker < workers; ++source_worker) {
                            reduced = fp8ArgMin(reduced, lane_partials[warp][partial_base + source_worker]);
                        }
                        storeFp8ArgResult(outputs, outer_index * static_cast<uint64_t>(k) + lane, reduced);
                    }
                    __syncwarp();
                }
                continue;
            }

            if constexpr (MultipleComponentsPerLane) {
#pragma unroll
                for (uint32_t slot = 0; slot < static_cast<uint32_t>(MaxComponentsPerLane); ++slot) {
                    const uint32_t owned_component =
                        oneWindowArgOwnedComponent<LanesPerOutput, MaxComponentsPerLane>(lane, slot);
                    if (owned_component < k) {
                        warp_partials[warp][owned_component] = valid_output ? local[slot] : Identity;
                    }
                }
            } else if ((workers & (workers - 1U)) == 0U) {
                const uint32_t warp_reduced = shuffleFoldOneWindowArgWorkers<LanesPerOutput>(
                    (valid_output && lane < geometry.task_count) ? local[0] : Identity, workers);
                if (worker == 0U && component < k) {
                    warp_partials[warp][component] = valid_output ? warp_reduced : Identity;
                }
            } else {
                lane_partials[warp][lane] = (valid_output && lane < geometry.task_count) ? local[0] : Identity;
                __syncwarp();
                if (lane < k) {
                    uint32_t warp_reduced = Identity;
                    const uint32_t partial_base = lane * workers;
                    for (uint32_t source_worker = 0; source_worker < workers; ++source_worker) {
                        warp_reduced = fp8ArgMin(warp_reduced, lane_partials[warp][partial_base + source_worker]);
                    }
                    warp_partials[warp][lane] = valid_output ? warp_reduced : Identity;
                }
            }
            __syncthreads();

            if (warp_in_output == 0) {
#pragma unroll
                for (uint32_t slot = 0; slot < static_cast<uint32_t>(MaxComponentsPerLane); ++slot) {
                    const uint32_t owned_component =
                        oneWindowArgOwnedComponent<LanesPerOutput, MaxComponentsPerLane>(lane, slot);
                    if (valid_output && owned_component < k) {
                        uint32_t reduced = Identity;
                        for (uint32_t source_warp = 0; source_warp < warps_per_output; ++source_warp) {
                            reduced = fp8ArgMin(reduced, warp_partials[first_warp_in_output + source_warp][owned_component]);
                        }
                        storeFp8ArgResult(outputs, outer_index * static_cast<uint64_t>(k) + owned_component, reduced);
                    }
                }
            }
            __syncthreads();
        }
    }
}

// ARG-A3: DeepRSmallK argmin/argmax twins: the staged first stages of narrowFp8PackedFlatRCooperative{,TinyK}
// FirstStageKernel (P16, grid x = shard, z = outer, eight warps staging whole-row groups, the same rows per warp group
// and the same load-ahead rule) with uint32 candidates. They write FP32 value + carried index partials
// [outer, shards, K]; the index is (shard row base + row) * domain_stride. The planner reduces the shard axis with the
// carried-index ARG reducer. Rows of one component are packed four per key word (fp8ArgKeys4), the OneWindow
// small-bucket scheme. The shard depth is a runtime operand for every ownership (VALUE compiles it in for one
// component per lane; measure before copying that).

// Up to four rows of one component, `row_step_bytes` apart in the staged window, reduced into `local`. Rows are
// first_row, first_row + row_stride, ... (shard-local).
__device__ __forceinline__ uint32_t deepRArgRows4(uint32_t local, const uint8_t* staged_bytes, uint32_t staged_index,
                                                  uint32_t row_step_bytes, uint32_t first_row, uint32_t row_stride,
                                                  uint32_t count, const Fp8ArgFold& fold) {
    uint32_t packed = 0;
#pragma unroll
    for (uint32_t j = 0; j < 4U; ++j) {
        if (j < count) packed |= static_cast<uint32_t>(staged_bytes[staged_index + j * row_step_bytes]) << (8U * j);
    }
    const Fp8ArgKeys4 keys = fp8ArgKeys4(packed, fold);
    if (count == 4U) return fp8ArgMin(local, fp8ArgRowsCandidate4(keys, first_row, row_stride));  // 1124
#pragma unroll
    for (uint32_t j = 0; j < 4U; ++j) {
        if (j < count) local = fp8ArgMin(local, fp8ArgLaneCandidate(keys, j, first_row + j * row_stride));
    }
    return local;
}

// Packing (1109): one and two components per lane pack four rows of a component per key word (groups hold 15+ rows
// at K <= 32 and 7+ at K <= 64). Four or more components per lane pack four of the lane's components of one row, as
// OneWindow's large buckets do: at K 161..256 a 497-byte group holds one to three rows, so row packing filled one to
// three bytes of every key word (1108: c8 twin 1.92x its VALUE kernel).
// 1129: one component per lane at K 9..16 used only K of 32 lanes (ARG/VALUE 1.24, worst 1.41); `workers` =
// floor(32 / K) lanes now share a component, worker w keying rows w, w + workers, ... (the tiny-K scheme with a
// runtime lane count). workers = 1 at K 17..32 is the former loop.
template <int MaxComponentsPerLane>
__device__ __forceinline__ void deepRArgGroup(uint32_t (&local)[MaxComponentsPerLane], const uint8_t* staged_bytes,
                                              uint32_t lane, uint32_t prefix_elements, uint32_t inner_size_32,
                                              uint32_t group_row, uint32_t group_rows, const Fp8ArgFold& fold,
                                              uint32_t component = 0, uint32_t worker = 0, uint32_t workers = 1) {
    if constexpr (MaxComponentsPerLane == 1) {
        static_cast<void>(lane);
        if (component < inner_size_32) {
            const uint32_t row_step_bytes = workers * inner_size_32;
            uint32_t row = worker;
            uint32_t staged_index = prefix_elements + worker * inner_size_32 + component;
            for (; row + 3U * workers < group_rows; row += 4U * workers, staged_index += 4U * row_step_bytes) {
                local[0] = deepRArgRows4(local[0], staged_bytes, staged_index, row_step_bytes, group_row + row,
                                         workers, 4U, fold);
            }
            if (row < group_rows) {
                const uint32_t count = (group_rows - row + workers - 1U) / workers;
                local[0] = deepRArgRows4(local[0], staged_bytes, staged_index, row_step_bytes, group_row + row,
                                         workers, count, fold);
            }
        }
    } else if constexpr (MaxComponentsPerLane >= 4) {
        static_cast<void>(component);
        static_cast<void>(worker);
        static_cast<void>(workers);
        for (uint32_t row = 0; row < group_rows; ++row) {
            const uint32_t row_index = prefix_elements + row * inner_size_32 + lane;
#pragma unroll
            for (uint32_t slot_base = 0; slot_base < static_cast<uint32_t>(MaxComponentsPerLane); slot_base += 4U) {
                if (lane + slot_base * TILED_REDUCTION_WARP_THREADS >= inner_size_32) continue;
                uint32_t packed = 0;
#pragma unroll
                for (uint32_t j = 0; j < 4U; ++j) {
                    if (slot_base + j < static_cast<uint32_t>(MaxComponentsPerLane) &&
                        lane + (slot_base + j) * TILED_REDUCTION_WARP_THREADS < inner_size_32) {
                        packed |= static_cast<uint32_t>(
                                      staged_bytes[row_index + (slot_base + j) * TILED_REDUCTION_WARP_THREADS])
                                  << (8U * j);
                    }
                }
                const Fp8ArgKeys4 keys = fp8ArgKeys4(packed, fold);
#pragma unroll
                for (uint32_t j = 0; j < 4U; ++j) {
                    if (slot_base + j < static_cast<uint32_t>(MaxComponentsPerLane) &&
                        lane + (slot_base + j) * TILED_REDUCTION_WARP_THREADS < inner_size_32) {
                        // The guard is compile-time after unrolling; the clamp keeps every index in bounds.
                        const uint32_t slot = slot_base + j < static_cast<uint32_t>(MaxComponentsPerLane) ? slot_base + j : 0U;
                        local[slot] = fp8ArgMin(local[slot], fp8ArgLaneCandidate(keys, j, group_row + row));
                    }
                }
            }
        }
    } else {
        static_cast<void>(component);
        static_cast<void>(worker);
        static_cast<void>(workers);
#pragma unroll
        for (int slot = 0; slot < MaxComponentsPerLane; ++slot) {
            const uint32_t owned = lane + static_cast<uint32_t>(slot) * TILED_REDUCTION_WARP_THREADS;
            if (owned < inner_size_32) {
                uint32_t staged_index = prefix_elements + owned;
                uint32_t row = 0;
                for (; row + 4U <= group_rows; row += 4U, staged_index += 4U * inner_size_32) {
                    local[slot] = deepRArgRows4(local[slot], staged_bytes, staged_index, inner_size_32,
                                                group_row + row, 1U, 4U, fold);
                }
                if (row < group_rows) {
                    local[slot] = deepRArgRows4(local[slot], staged_bytes, staged_index, inner_size_32,
                                                group_row + row, 1U, group_rows - row, fold);
                }
            }
        }
    }
}

template <int MaxComponentsPerLane>
__global__ void fp8DeepRArgFirstStageKernel(const uint8_t* input,
                                            Fp8ArgOutputs outputs,
                                            uint64_t outer_size,
                                            uint64_t reduction_size,
                                            uint64_t inner_size,
                                            uint64_t shards_per_output,
                                            uint32_t rows_per_shard_log2,
                                            uint32_t rows_per_warp_group) {
    constexpr int PacketBytes = 16;
    static_assert(MaxComponentsPerLane == 1 || MaxComponentsPerLane == 2 || MaxComponentsPerLane == 4 ||
                  MaxComponentsPerLane == 5 || MaxComponentsPerLane == 8);
    constexpr int MaxSupportedK = TILED_REDUCTION_WARP_THREADS * MaxComponentsPerLane;
    constexpr uint32_t Identity = FP8_ARG_LANE_IDENTITY;

    __shared__ RawVectorPacket<PacketBytes> staged_packets[NARROW_FLAT_WARPS][NARROW_FP8_PACKETS_PER_WARP];
    __shared__ uint32_t warp_partials[NARROW_FLAT_WARPS][MaxSupportedK];

    const uint32_t warp = threadIdx.x / TILED_REDUCTION_WARP_THREADS;
    const uint32_t lane = threadIdx.x % TILED_REDUCTION_WARP_THREADS;
    const uint32_t inner_size_32 = static_cast<uint32_t>(inner_size);
    const uint32_t rows_per_block_round = rows_per_warp_group * NARROW_FLAT_WARPS;
    const uint32_t outer_count = static_cast<uint32_t>(outer_size);
    const uint32_t shard_count = static_cast<uint32_t>(shards_per_output);
    // One component per lane: floor(32 / K) workers per component (1129).
    const uint32_t workers =
        MaxComponentsPerLane == 1 && inner_size_32 <= TILED_REDUCTION_WARP_THREADS / 2 && inner_size_32 != 0
            ? TILED_REDUCTION_WARP_THREADS / inner_size_32
            : 1U;
    const uint32_t c1_component = lane / workers;
    const uint32_t c1_worker = lane - c1_component * workers;
    for (uint32_t outer_index = blockIdx.z; outer_index < outer_count; outer_index += gridDim.z)
    for (uint32_t shard = blockIdx.x; shard < shard_count; shard += gridDim.x) {
        const uint64_t row_begin = static_cast<uint64_t>(shard) << rows_per_shard_log2;
        const uint32_t shard_rows = row_begin < reduction_size
            ? static_cast<uint32_t>(minU64(uint64_t{1} << rows_per_shard_log2, reduction_size - row_begin))
            : 0U;
        const uint8_t* shard_input = input + (outer_index * reduction_size + row_begin) * inner_size;

        uint32_t local[MaxComponentsPerLane];
#pragma unroll
        for (int slot = 0; slot < MaxComponentsPerLane; ++slot) local[slot] = Identity;

        // The VALUE kernel's rule: prefetch the next group only when lanes own two or more components.
        if constexpr (MaxComponentsPerLane >= 2) {
            uint32_t group_row = warp * rows_per_warp_group;
            NarrowFp8GroupWindow<PacketBytes> window{};
            RawVectorPacket<PacketBytes> pending{};
            if (group_row < shard_rows) {
                window = narrowFp8GroupWindow<PacketBytes>(shard_input, group_row, shard_rows, rows_per_warp_group,
                                                           inner_size_32);
                if (lane < window.packet_count) pending = window.aligned_begin[lane];
            }
            for (; group_row < shard_rows; group_row += rows_per_block_round) {
                const uint32_t group_rows = window.group_rows;
                const uint32_t prefix_elements = window.prefix_elements;
                if (lane < window.packet_count) staged_packets[warp][lane] = pending;
                __syncwarp();
                const uint32_t next_group_row = group_row + rows_per_block_round;
                if (next_group_row < shard_rows) {
                    window = narrowFp8GroupWindow<PacketBytes>(shard_input, next_group_row, shard_rows,
                                                               rows_per_warp_group, inner_size_32);
                    if (lane < window.packet_count) pending = window.aligned_begin[lane];
                }
                deepRArgGroup<MaxComponentsPerLane>(local, reinterpret_cast<const uint8_t*>(&staged_packets[warp][0]),
                                                    lane, prefix_elements, inner_size_32, group_row, group_rows,
                                                    outputs.fold);
                __syncwarp();
            }
        } else {
            for (uint32_t group_row = warp * rows_per_warp_group; group_row < shard_rows;
                 group_row += rows_per_block_round) {
                const NarrowFp8GroupWindow<PacketBytes> window = narrowFp8GroupWindow<PacketBytes>(
                    shard_input, group_row, shard_rows, rows_per_warp_group, inner_size_32);
                if (lane < window.packet_count) staged_packets[warp][lane] = window.aligned_begin[lane];
                __syncwarp();
                deepRArgGroup<MaxComponentsPerLane>(local, reinterpret_cast<const uint8_t*>(&staged_packets[warp][0]),
                                                    lane, window.prefix_elements, inner_size_32, group_row,
                                                    window.group_rows, outputs.fold, c1_component, c1_worker, workers);
                __syncwarp();
            }
        }

        if constexpr (MaxComponentsPerLane == 1) {
            // Fold each component's workers (lanes c * workers .. + workers - 1) into its worker 0.
            uint32_t folded = local[0];
            for (uint32_t offset = 1; offset < workers; ++offset) {
                const uint32_t other = __shfl_down_sync(0xffffffffU, local[0], offset);
                if (c1_worker == 0U) folded = fp8ArgMin(folded, other);
            }
            if (c1_worker == 0U && c1_component < inner_size_32) warp_partials[warp][c1_component] = folded;
        } else {
            static_cast<void>(c1_component);
            static_cast<void>(c1_worker);
            static_cast<void>(workers);
#pragma unroll
            for (int slot = 0; slot < MaxComponentsPerLane; ++slot) {
                const uint32_t component = lane + static_cast<uint32_t>(slot) * TILED_REDUCTION_WARP_THREADS;
                if (component < inner_size_32) warp_partials[warp][component] = local[slot];
            }
        }
        __syncthreads();
        if (threadIdx.x < inner_size_32) {
            uint32_t reduced = Identity;
#pragma unroll
            for (int source_warp = 0; source_warp < NARROW_FLAT_WARPS; ++source_warp) {
                reduced = fp8ArgMin(reduced, warp_partials[source_warp][threadIdx.x]);
            }
            storeFp8ArgResult<true>(outputs, (static_cast<uint64_t>(outer_index) * shards_per_output + shard) * inner_size +
                                           threadIdx.x,
                              reduced, row_begin);
        }
        __syncthreads();
    }
}

template <int LanesPerComponent>
__global__ void fp8DeepRArgTinyKFirstStageKernel(const uint8_t* input,
                                                 Fp8ArgOutputs outputs,
                                                 uint64_t outer_size,
                                                 uint64_t reduction_size,
                                                 uint64_t inner_size,
                                                 uint64_t shards_per_output,
                                                 uint32_t rows_per_shard_log2,
                                                 uint32_t rows_per_warp_group) {
    constexpr int PacketBytes = 16;
    static_assert(LanesPerComponent == 4 || LanesPerComponent == 8);
    constexpr int MaxSupportedK = TILED_REDUCTION_WARP_THREADS / LanesPerComponent;
    constexpr uint32_t Identity = FP8_ARG_LANE_IDENTITY;

    __shared__ RawVectorPacket<PacketBytes> staged_packets[NARROW_FLAT_WARPS][NARROW_FP8_PACKETS_PER_WARP];
    __shared__ uint32_t warp_partials[NARROW_FLAT_WARPS][MaxSupportedK];

    const uint32_t warp = threadIdx.x / TILED_REDUCTION_WARP_THREADS;
    const uint32_t lane = threadIdx.x % TILED_REDUCTION_WARP_THREADS;
    const uint32_t component = lane / LanesPerComponent;
    const uint32_t component_lane = lane % LanesPerComponent;
    const uint32_t inner_size_32 = static_cast<uint32_t>(inner_size);
    const uint32_t rows_per_block_round = rows_per_warp_group * NARROW_FLAT_WARPS;
    const uint32_t outer_count = static_cast<uint32_t>(outer_size);
    const uint32_t shard_count = static_cast<uint32_t>(shards_per_output);
    for (uint32_t outer_index = blockIdx.z; outer_index < outer_count; outer_index += gridDim.z)
    for (uint32_t shard = blockIdx.x; shard < shard_count; shard += gridDim.x) {
        const uint64_t row_begin = static_cast<uint64_t>(shard) << rows_per_shard_log2;
        const uint32_t shard_rows = row_begin < reduction_size
            ? static_cast<uint32_t>(minU64(uint64_t{1} << rows_per_shard_log2, reduction_size - row_begin))
            : 0U;
        const uint8_t* shard_input = input + (outer_index * reduction_size + row_begin) * inner_size;

        uint32_t local = Identity;
        for (uint32_t group_row = warp * rows_per_warp_group; group_row < shard_rows;
             group_row += rows_per_block_round) {
            const NarrowFp8GroupWindow<PacketBytes> window = narrowFp8GroupWindow<PacketBytes>(
                shard_input, group_row, shard_rows, rows_per_warp_group, inner_size_32);
            if (lane < window.packet_count) staged_packets[warp][lane] = window.aligned_begin[lane];
            __syncwarp();
            if (component < inner_size_32) {
                const auto* staged_bytes = reinterpret_cast<const uint8_t*>(&staged_packets[warp][0]);
                const uint32_t row_step_bytes = LanesPerComponent * inner_size_32;
                uint32_t row = component_lane;
                uint32_t staged_index = window.prefix_elements + component_lane * inner_size_32 + component;
                // This lane's rows are component_lane, + L, + 2L, ...: four per key word.
                for (; row + 3U * LanesPerComponent < window.group_rows;
                     row += 4U * LanesPerComponent, staged_index += 4U * row_step_bytes) {
                    local = deepRArgRows4(local, staged_bytes, staged_index, row_step_bytes, group_row + row,
                                          LanesPerComponent, 4U, outputs.fold);
                }
                if (row < window.group_rows) {
                    const uint32_t count = (window.group_rows - row + LanesPerComponent - 1U) / LanesPerComponent;
                    local = deepRArgRows4(local, staged_bytes, staged_index, row_step_bytes, group_row + row,
                                          LanesPerComponent, count, outputs.fold);
                }
            }
            __syncwarp();
        }

        // Fold the component's lanes (aligned groups of LanesPerComponent) with butterfly shuffles.
#pragma unroll
        for (uint32_t offset = 1; offset < static_cast<uint32_t>(LanesPerComponent); offset <<= 1) {
            local = fp8ArgMin(local, __shfl_xor_sync(0xffffffffU, local, offset));
        }
        if (component_lane == 0 && component < inner_size_32) warp_partials[warp][component] = local;
        __syncthreads();
        if (threadIdx.x < inner_size_32) {
            uint32_t reduced = Identity;
#pragma unroll
            for (int source_warp = 0; source_warp < NARROW_FLAT_WARPS; ++source_warp) {
                reduced = fp8ArgMin(reduced, warp_partials[source_warp][threadIdx.x]);
            }
            storeFp8ArgResult<true>(outputs, (static_cast<uint64_t>(outer_index) * shards_per_output + shard) * inner_size +
                                           threadIdx.x,
                              reduced, row_begin);
        }
        __syncthreads();
    }
}

// 1203: the FP32 RK ARG DeepRSmallK twins (plans/Thor_FP32_Dense_Reduction_Plan, phase 6): the FP32 paths of
// narrowFp8PackedFlatRCooperative{,TinyK}FirstStageKernel (P16, grid x = shard, z = outer, eight warps staging
// whole-row groups of at most 125 FP32, the same rows per warp group and the same load-ahead rule) with one FP32 ARG
// candidate (Fp32ArgKey.cuh) per owned component, and FP8 ARG's lane sharing at K 9..16 (floor(32 / K) workers per
// component). Staged only (FP32 VALUE DeepRSmallK's R > depth): each shard writes packed partials [outer, shards, K],
// key << 32 | shard row << 1 | negative zero, which launchFp32ArgPackedContinuation finishes. A packed word orders as
// its (key, tail) candidate, so lanes, warps and workers merge with a 64-bit minimum; a lane's rows only increase, so
// a strictly smaller key replaces its best. The shard depth is a runtime operand for every ownership.
__device__ __forceinline__ void fp32DeepRArgKeyElement(uint32_t& best_key, uint32_t& best_tail, uint32_t bits,
                                                       uint32_t row, uint32_t xor_mask) {
    uint32_t negative_zero;
    const uint32_t key = fp32ArgKey(bits, xor_mask, negative_zero);
    if (key < best_key) {
        best_key = key;
        best_tail = (row << 1) | negative_zero;
    }
}

__device__ __forceinline__ uint64_t fp32DeepRArgPacked(uint32_t key, uint32_t tail) {
    return (static_cast<uint64_t>(key) << 32) | tail;
}

__device__ __forceinline__ uint64_t fp32DeepRArgMin(uint64_t a, uint64_t b) { return b < a ? b : a; }

template <int MaxComponentsPerLane>
__global__ __launch_bounds__(NARROW_FLAT_THREADS) void fp32DeepRArgKernel(const float* input,
                                                                         uint64_t* packed,
                                                                         uint64_t outer_size,
                                                                         uint64_t reduction_size,
                                                                         uint64_t inner_size,
                                                                         uint64_t shards_per_output,
                                                                         uint32_t rows_per_shard_log2,
                                                                         uint32_t rows_per_warp_group,
                                                                         uint32_t xor_mask) {
    constexpr int PacketBytes = 16;
    static_assert(MaxComponentsPerLane == 1 || MaxComponentsPerLane == 2 || MaxComponentsPerLane == 4);
    constexpr int MaxSupportedK = TILED_REDUCTION_WARP_THREADS * MaxComponentsPerLane;

    __shared__ RawVectorPacket<PacketBytes> staged_packets[NARROW_FLAT_WARPS][NARROW_FP8_PACKETS_PER_WARP];
    __shared__ uint64_t warp_partials[NARROW_FLAT_WARPS][MaxSupportedK];

    const uint32_t warp = threadIdx.x / TILED_REDUCTION_WARP_THREADS;
    const uint32_t lane = threadIdx.x % TILED_REDUCTION_WARP_THREADS;
    const uint32_t inner_size_32 = static_cast<uint32_t>(inner_size);
    const uint32_t rows_per_block_round = rows_per_warp_group * NARROW_FLAT_WARPS;
    const uint32_t outer_count = static_cast<uint32_t>(outer_size);
    const uint32_t shard_count = static_cast<uint32_t>(shards_per_output);
    // One component per lane: floor(32 / K) workers per component at K <= 16 (FP8 ARG's 1129 rule).
    const uint32_t workers =
        MaxComponentsPerLane == 1 && inner_size_32 <= TILED_REDUCTION_WARP_THREADS / 2 && inner_size_32 != 0
            ? TILED_REDUCTION_WARP_THREADS / inner_size_32
            : 1U;
    const uint32_t c1_component = lane / workers;
    const uint32_t c1_worker = lane - c1_component * workers;
    const auto* staged_words = reinterpret_cast<const uint32_t*>(&staged_packets[warp][0]);
    for (uint32_t outer_index = blockIdx.z; outer_index < outer_count; outer_index += gridDim.z)
    for (uint32_t shard = blockIdx.x; shard < shard_count; shard += gridDim.x) {
        const uint64_t row_begin = static_cast<uint64_t>(shard) << rows_per_shard_log2;
        const uint32_t shard_rows = row_begin < reduction_size
            ? static_cast<uint32_t>(minU64(uint64_t{1} << rows_per_shard_log2, reduction_size - row_begin))
            : 0U;
        const float* shard_input = input + (outer_index * reduction_size + row_begin) * inner_size;

        uint32_t best_key[MaxComponentsPerLane];
        uint32_t best_tail[MaxComponentsPerLane];
#pragma unroll
        for (int slot = 0; slot < MaxComponentsPerLane; ++slot) {
            best_key[slot] = FP32_ARG_KEY_IDENTITY;
            best_tail[slot] = 0xFFFFFFFFU;
        }

        // The VALUE kernel's rule: prefetch the next group only when lanes own two or more components.
        if constexpr (MaxComponentsPerLane >= 2) {
            uint32_t group_row = warp * rows_per_warp_group;
            NarrowFp8GroupWindow<PacketBytes> window{};
            RawVectorPacket<PacketBytes> pending{};
            if (group_row < shard_rows) {
                window = narrowFp8GroupWindow<PacketBytes>(shard_input, group_row, shard_rows, rows_per_warp_group,
                                                           inner_size_32);
                if (lane < window.packet_count) pending = window.aligned_begin[lane];
            }
            for (; group_row < shard_rows; group_row += rows_per_block_round) {
                const uint32_t group_rows = window.group_rows;
                const uint32_t prefix_elements = window.prefix_elements;
                if (lane < window.packet_count) staged_packets[warp][lane] = pending;
                __syncwarp();
                const uint32_t next_group_row = group_row + rows_per_block_round;
                if (next_group_row < shard_rows) {
                    window = narrowFp8GroupWindow<PacketBytes>(shard_input, next_group_row, shard_rows,
                                                               rows_per_warp_group, inner_size_32);
                    if (lane < window.packet_count) pending = window.aligned_begin[lane];
                }
#pragma unroll
                for (int slot = 0; slot < MaxComponentsPerLane; ++slot) {
                    const uint32_t owned = lane + static_cast<uint32_t>(slot) * TILED_REDUCTION_WARP_THREADS;
                    if (owned < inner_size_32) {
                        uint32_t staged_index = prefix_elements + owned;
                        for (uint32_t row = 0; row < group_rows; ++row, staged_index += inner_size_32) {
                            fp32DeepRArgKeyElement(best_key[slot], best_tail[slot], staged_words[staged_index],
                                                   group_row + row, xor_mask);
                        }
                    }
                }
                __syncwarp();
            }
        } else {
            for (uint32_t group_row = warp * rows_per_warp_group; group_row < shard_rows;
                 group_row += rows_per_block_round) {
                const NarrowFp8GroupWindow<PacketBytes> window = narrowFp8GroupWindow<PacketBytes>(
                    shard_input, group_row, shard_rows, rows_per_warp_group, inner_size_32);
                if (lane < window.packet_count) staged_packets[warp][lane] = window.aligned_begin[lane];
                __syncwarp();
                if (c1_component < inner_size_32) {
                    uint32_t staged_index = window.prefix_elements + c1_worker * inner_size_32 + c1_component;
                    const uint32_t staged_step = workers * inner_size_32;
                    for (uint32_t row = c1_worker; row < window.group_rows;
                         row += workers, staged_index += staged_step) {
                        fp32DeepRArgKeyElement(best_key[0], best_tail[0], staged_words[staged_index], group_row + row,
                                               xor_mask);
                    }
                }
                __syncwarp();
            }
        }

        if constexpr (MaxComponentsPerLane == 1) {
            // Fold each component's workers (lanes c * workers .. + workers - 1) into its worker 0.
            uint64_t folded = fp32DeepRArgPacked(best_key[0], best_tail[0]);
            const uint64_t mine = folded;
            for (uint32_t offset = 1; offset < workers; ++offset) {
                const uint64_t other =
                    __shfl_down_sync(0xffffffffU, static_cast<unsigned long long>(mine), static_cast<int>(offset));
                if (c1_worker == 0U) folded = fp32DeepRArgMin(folded, other);
            }
            if (c1_worker == 0U && c1_component < inner_size_32) warp_partials[warp][c1_component] = folded;
        } else {
            static_cast<void>(c1_component);
            static_cast<void>(c1_worker);
            static_cast<void>(workers);
#pragma unroll
            for (int slot = 0; slot < MaxComponentsPerLane; ++slot) {
                const uint32_t component = lane + static_cast<uint32_t>(slot) * TILED_REDUCTION_WARP_THREADS;
                if (component < inner_size_32) {
                    warp_partials[warp][component] = fp32DeepRArgPacked(best_key[slot], best_tail[slot]);
                }
            }
        }
        __syncthreads();
        if (threadIdx.x < inner_size_32) {
            uint64_t reduced = FP32_ARG_PACKED_IDENTITY;
#pragma unroll
            for (int source_warp = 0; source_warp < NARROW_FLAT_WARPS; ++source_warp) {
                reduced = fp32DeepRArgMin(reduced, warp_partials[source_warp][threadIdx.x]);
            }
            packed[(static_cast<uint64_t>(outer_index) * shards_per_output + shard) * inner_size + threadIdx.x] =
                reduced;
        }
        __syncthreads();
    }
}

template <int LanesPerComponent>
__global__ __launch_bounds__(NARROW_FLAT_THREADS) void fp32DeepRArgTinyKKernel(const float* input,
                                                                              uint64_t* packed,
                                                                              uint64_t outer_size,
                                                                              uint64_t reduction_size,
                                                                              uint64_t inner_size,
                                                                              uint64_t shards_per_output,
                                                                              uint32_t rows_per_shard_log2,
                                                                              uint32_t rows_per_warp_group,
                                                                              uint32_t xor_mask) {
    constexpr int PacketBytes = 16;
    static_assert(LanesPerComponent == 4 || LanesPerComponent == 8);
    constexpr int MaxSupportedK = TILED_REDUCTION_WARP_THREADS / LanesPerComponent;

    __shared__ RawVectorPacket<PacketBytes> staged_packets[NARROW_FLAT_WARPS][NARROW_FP8_PACKETS_PER_WARP];
    __shared__ uint64_t warp_partials[NARROW_FLAT_WARPS][MaxSupportedK];

    const uint32_t warp = threadIdx.x / TILED_REDUCTION_WARP_THREADS;
    const uint32_t lane = threadIdx.x % TILED_REDUCTION_WARP_THREADS;
    const uint32_t component = lane / LanesPerComponent;
    const uint32_t component_lane = lane % LanesPerComponent;
    const uint32_t inner_size_32 = static_cast<uint32_t>(inner_size);
    const uint32_t rows_per_block_round = rows_per_warp_group * NARROW_FLAT_WARPS;
    const uint32_t outer_count = static_cast<uint32_t>(outer_size);
    const uint32_t shard_count = static_cast<uint32_t>(shards_per_output);
    const auto* staged_words = reinterpret_cast<const uint32_t*>(&staged_packets[warp][0]);
    for (uint32_t outer_index = blockIdx.z; outer_index < outer_count; outer_index += gridDim.z)
    for (uint32_t shard = blockIdx.x; shard < shard_count; shard += gridDim.x) {
        const uint64_t row_begin = static_cast<uint64_t>(shard) << rows_per_shard_log2;
        const uint32_t shard_rows = row_begin < reduction_size
            ? static_cast<uint32_t>(minU64(uint64_t{1} << rows_per_shard_log2, reduction_size - row_begin))
            : 0U;
        const float* shard_input = input + (outer_index * reduction_size + row_begin) * inner_size;

        uint32_t best_key = FP32_ARG_KEY_IDENTITY;
        uint32_t best_tail = 0xFFFFFFFFU;
        for (uint32_t group_row = warp * rows_per_warp_group; group_row < shard_rows;
             group_row += rows_per_block_round) {
            const NarrowFp8GroupWindow<PacketBytes> window = narrowFp8GroupWindow<PacketBytes>(
                shard_input, group_row, shard_rows, rows_per_warp_group, inner_size_32);
            if (lane < window.packet_count) staged_packets[warp][lane] = window.aligned_begin[lane];
            __syncwarp();
            if (component < inner_size_32) {
                const uint32_t staged_step = LanesPerComponent * inner_size_32;
                uint32_t staged_index = window.prefix_elements + component_lane * inner_size_32 + component;
                for (uint32_t row = component_lane; row < window.group_rows;
                     row += LanesPerComponent, staged_index += staged_step) {
                    fp32DeepRArgKeyElement(best_key, best_tail, staged_words[staged_index], group_row + row,
                                           xor_mask);
                }
            }
            __syncwarp();
        }

        // Fold the component's lanes (aligned groups of LanesPerComponent) with butterfly shuffles.
        uint64_t local = fp32DeepRArgPacked(best_key, best_tail);
#pragma unroll
        for (uint32_t offset = 1; offset < static_cast<uint32_t>(LanesPerComponent); offset <<= 1) {
            local = fp32DeepRArgMin(
                local, __shfl_xor_sync(0xffffffffU, static_cast<unsigned long long>(local), static_cast<int>(offset)));
        }
        if (component_lane == 0 && component < inner_size_32) warp_partials[warp][component] = local;
        __syncthreads();
        if (threadIdx.x < inner_size_32) {
            uint64_t reduced = FP32_ARG_PACKED_IDENTITY;
#pragma unroll
            for (int source_warp = 0; source_warp < NARROW_FLAT_WARPS; ++source_warp) {
                reduced = fp32DeepRArgMin(reduced, warp_partials[source_warp][threadIdx.x]);
            }
            packed[(static_cast<uint64_t>(outer_index) * shards_per_output + shard) * inner_size + threadIdx.x] =
                reduced;
        }
        __syncthreads();
    }
}

// 1213: the carried-index DeepRSmallK ARG twins (phase 8; see Fp32ArgWidePartials): fp32DeepRArgKernel's and
// fp32DeepRArgTinyKKernel's ownership, groups and load-ahead rule, staging each lane's carried index packet (the same
// elements as its value packet; the last packet of the tensor assembled per element) next to the values. An element's
// tail is (carried + row * row_stride) << 1 | negative zero, with the row absolute in R; every merge (workers, lanes,
// warps) is the full (key, tail) minimum. Staged: wide partials [outer, shards, K], finished by the wide continuation.
// The input and the carried index are 16-byte aligned.
template <typename IndexT>
__device__ __forceinline__ PackedInputValues<IndexT, 4> loadDeepRCarriedPacket(const IndexT* carried,
                                                                              uint64_t element_begin,
                                                                              uint64_t total_elements) {
    if (element_begin + 4U <= total_elements) return loadVectorizedInputPacket<IndexT, 4>(carried + element_begin);
    PackedInputValues<IndexT, 4> packet{};
#pragma unroll
    for (uint32_t item = 0; item < 4U; ++item) {
        if (element_begin + item < total_elements) packet.values[item] = carried[element_begin + item];
    }
    return packet;
}

template <typename IndexT>
__device__ __forceinline__ void stageDeepRCarriedPacket(IndexT* warp_indices, uint32_t lane,
                                                        const PackedInputValues<IndexT, 4>& packet) {
#pragma unroll
    for (uint32_t item = 0; item < 4U; ++item) warp_indices[lane * 4U + item] = packet.values[item];
}

__device__ __forceinline__ void fp32DeepRCarriedArgKeyElement(uint32_t& best_key, uint64_t& best_tail, uint32_t bits,
                                                              uint64_t carried, uint64_t row_index,
                                                              uint32_t xor_mask) {
    uint32_t negative_zero;
    const uint32_t key = fp32ArgKey(bits, xor_mask, negative_zero);
    fp32ArgKeepMin(best_key, best_tail, key, ((carried + row_index) << 1) | negative_zero);
}

template <int MaxComponentsPerLane, typename IndexT>
__global__ __launch_bounds__(NARROW_FLAT_THREADS) void fp32DeepRCarriedArgKernel(const float* input,
                                                                                const IndexT* carried,
                                                                                Fp32ArgWidePartials partials,
                                                                                uint64_t outer_size,
                                                                                uint64_t reduction_size,
                                                                                uint64_t inner_size,
                                                                                uint64_t shards_per_output,
                                                                                uint32_t rows_per_shard_log2,
                                                                                uint32_t rows_per_warp_group,
                                                                                uint32_t xor_mask,
                                                                                uint64_t row_stride) {
    constexpr int PacketBytes = 16;
    static_assert(MaxComponentsPerLane == 1 || MaxComponentsPerLane == 2 || MaxComponentsPerLane == 4);
    static_assert(std::is_same_v<IndexT, uint32_t> || std::is_same_v<IndexT, uint64_t>);
    constexpr int MaxSupportedK = TILED_REDUCTION_WARP_THREADS * MaxComponentsPerLane;

    __shared__ RawVectorPacket<PacketBytes> staged_packets[NARROW_FLAT_WARPS][NARROW_FP8_PACKETS_PER_WARP];
    __shared__ IndexT staged_indices[NARROW_FLAT_WARPS][NARROW_FP8_PACKETS_PER_WARP * 4];
    __shared__ uint32_t warp_keys[NARROW_FLAT_WARPS][MaxSupportedK];
    __shared__ uint64_t warp_tails[NARROW_FLAT_WARPS][MaxSupportedK];

    const uint32_t warp = threadIdx.x / TILED_REDUCTION_WARP_THREADS;
    const uint32_t lane = threadIdx.x % TILED_REDUCTION_WARP_THREADS;
    const uint32_t inner_size_32 = static_cast<uint32_t>(inner_size);
    const uint32_t rows_per_block_round = rows_per_warp_group * NARROW_FLAT_WARPS;
    const uint32_t outer_count = static_cast<uint32_t>(outer_size);
    const uint32_t shard_count = static_cast<uint32_t>(shards_per_output);
    const uint64_t total_elements = outer_size * reduction_size * inner_size;
    // One component per lane: floor(32 / K) workers per component at K <= 16 (FP8 ARG's 1129 rule).
    const uint32_t workers =
        MaxComponentsPerLane == 1 && inner_size_32 <= TILED_REDUCTION_WARP_THREADS / 2 && inner_size_32 != 0
            ? TILED_REDUCTION_WARP_THREADS / inner_size_32
            : 1U;
    const uint32_t c1_component = lane / workers;
    const uint32_t c1_worker = lane - c1_component * workers;
    const auto* staged_words = reinterpret_cast<const uint32_t*>(&staged_packets[warp][0]);
    const IndexT* staged_carried = &staged_indices[warp][0];
    for (uint32_t outer_index = blockIdx.z; outer_index < outer_count; outer_index += gridDim.z)
    for (uint32_t shard = blockIdx.x; shard < shard_count; shard += gridDim.x) {
        const uint64_t row_begin = static_cast<uint64_t>(shard) << rows_per_shard_log2;
        const uint32_t shard_rows = row_begin < reduction_size
            ? static_cast<uint32_t>(minU64(uint64_t{1} << rows_per_shard_log2, reduction_size - row_begin))
            : 0U;
        const uint64_t shard_element = (outer_index * reduction_size + row_begin) * inner_size;
        const float* shard_input = input + shard_element;
        // The element offset of a window's first staged packet (both bases are 16-byte aligned).
        const auto packet_element = [&](const NarrowFp8GroupWindow<PacketBytes>& window) {
            return static_cast<uint64_t>(reinterpret_cast<const float*>(window.aligned_begin) - input) + lane * 4U;
        };

        uint32_t best_key[MaxComponentsPerLane];
        uint64_t best_tail[MaxComponentsPerLane];
#pragma unroll
        for (int slot = 0; slot < MaxComponentsPerLane; ++slot) {
            best_key[slot] = FP32_ARG_KEY_IDENTITY;
            best_tail[slot] = FP32_ARG_TAIL_IDENTITY;
        }

        // The VALUE kernel's rule: prefetch the next group only when lanes own two or more components.
        if constexpr (MaxComponentsPerLane >= 2) {
            uint32_t group_row = warp * rows_per_warp_group;
            NarrowFp8GroupWindow<PacketBytes> window{};
            RawVectorPacket<PacketBytes> pending{};
            PackedInputValues<IndexT, 4> pending_indices{};
            if (group_row < shard_rows) {
                window = narrowFp8GroupWindow<PacketBytes>(shard_input, group_row, shard_rows, rows_per_warp_group,
                                                           inner_size_32);
                if (lane < window.packet_count) {
                    pending = window.aligned_begin[lane];
                    pending_indices = loadDeepRCarriedPacket<IndexT>(carried, packet_element(window), total_elements);
                }
            }
            for (; group_row < shard_rows; group_row += rows_per_block_round) {
                const uint32_t group_rows = window.group_rows;
                const uint32_t prefix_elements = window.prefix_elements;
                if (lane < window.packet_count) {
                    staged_packets[warp][lane] = pending;
                    stageDeepRCarriedPacket<IndexT>(&staged_indices[warp][0], lane, pending_indices);
                }
                __syncwarp();
                const uint32_t next_group_row = group_row + rows_per_block_round;
                if (next_group_row < shard_rows) {
                    window = narrowFp8GroupWindow<PacketBytes>(shard_input, next_group_row, shard_rows,
                                                               rows_per_warp_group, inner_size_32);
                    if (lane < window.packet_count) {
                        pending = window.aligned_begin[lane];
                        pending_indices =
                            loadDeepRCarriedPacket<IndexT>(carried, packet_element(window), total_elements);
                    }
                }
                const uint64_t group_row_index = (row_begin + group_row) * row_stride;
#pragma unroll
                for (int slot = 0; slot < MaxComponentsPerLane; ++slot) {
                    const uint32_t owned = lane + static_cast<uint32_t>(slot) * TILED_REDUCTION_WARP_THREADS;
                    if (owned < inner_size_32) {
                        uint32_t staged_index = prefix_elements + owned;
                        uint64_t row_index = group_row_index;
                        for (uint32_t row = 0; row < group_rows;
                             ++row, staged_index += inner_size_32, row_index += row_stride) {
                            fp32DeepRCarriedArgKeyElement(best_key[slot], best_tail[slot], staged_words[staged_index],
                                                          static_cast<uint64_t>(staged_carried[staged_index]),
                                                          row_index, xor_mask);
                        }
                    }
                }
                __syncwarp();
            }
        } else {
            for (uint32_t group_row = warp * rows_per_warp_group; group_row < shard_rows;
                 group_row += rows_per_block_round) {
                const NarrowFp8GroupWindow<PacketBytes> window = narrowFp8GroupWindow<PacketBytes>(
                    shard_input, group_row, shard_rows, rows_per_warp_group, inner_size_32);
                if (lane < window.packet_count) {
                    staged_packets[warp][lane] = window.aligned_begin[lane];
                    stageDeepRCarriedPacket<IndexT>(
                        &staged_indices[warp][0], lane,
                        loadDeepRCarriedPacket<IndexT>(carried, packet_element(window), total_elements));
                }
                __syncwarp();
                if (c1_component < inner_size_32) {
                    uint32_t staged_index = window.prefix_elements + c1_worker * inner_size_32 + c1_component;
                    const uint32_t staged_step = workers * inner_size_32;
                    uint64_t row_index = (row_begin + group_row + c1_worker) * row_stride;
                    const uint64_t row_index_step = static_cast<uint64_t>(workers) * row_stride;
                    for (uint32_t row = c1_worker; row < window.group_rows;
                         row += workers, staged_index += staged_step, row_index += row_index_step) {
                        fp32DeepRCarriedArgKeyElement(best_key[0], best_tail[0], staged_words[staged_index],
                                                      static_cast<uint64_t>(staged_carried[staged_index]), row_index,
                                                      xor_mask);
                    }
                }
                __syncwarp();
            }
        }

        if constexpr (MaxComponentsPerLane == 1) {
            // Fold each component's workers (lanes c * workers .. + workers - 1) into its worker 0.
            uint32_t folded_key = best_key[0];
            uint64_t folded_tail = best_tail[0];
            for (uint32_t offset = 1; offset < workers; ++offset) {
                const uint32_t other_key = __shfl_down_sync(0xffffffffU, best_key[0], static_cast<int>(offset));
                const uint64_t other_tail = __shfl_down_sync(
                    0xffffffffU, static_cast<unsigned long long>(best_tail[0]), static_cast<int>(offset));
                if (c1_worker == 0U) fp32ArgKeepMin(folded_key, folded_tail, other_key, other_tail);
            }
            if (c1_worker == 0U && c1_component < inner_size_32) {
                warp_keys[warp][c1_component] = folded_key;
                warp_tails[warp][c1_component] = folded_tail;
            }
        } else {
            static_cast<void>(c1_component);
            static_cast<void>(c1_worker);
            static_cast<void>(workers);
#pragma unroll
            for (int slot = 0; slot < MaxComponentsPerLane; ++slot) {
                const uint32_t component = lane + static_cast<uint32_t>(slot) * TILED_REDUCTION_WARP_THREADS;
                if (component < inner_size_32) {
                    warp_keys[warp][component] = best_key[slot];
                    warp_tails[warp][component] = best_tail[slot];
                }
            }
        }
        __syncthreads();
        if (threadIdx.x < inner_size_32) {
            uint32_t reduced_key = FP32_ARG_KEY_IDENTITY;
            uint64_t reduced_tail = FP32_ARG_TAIL_IDENTITY;
#pragma unroll
            for (int source_warp = 0; source_warp < NARROW_FLAT_WARPS; ++source_warp) {
                fp32ArgKeepMin(reduced_key, reduced_tail, warp_keys[source_warp][threadIdx.x],
                               warp_tails[source_warp][threadIdx.x]);
            }
            const uint64_t partial =
                (static_cast<uint64_t>(outer_index) * shards_per_output + shard) * inner_size + threadIdx.x;
            partials.keys[partial] = reduced_key;
            partials.tails[partial] = reduced_tail;
        }
        __syncthreads();
    }
}

template <int LanesPerComponent, typename IndexT>
__global__ __launch_bounds__(NARROW_FLAT_THREADS) void fp32DeepRCarriedArgTinyKKernel(const float* input,
                                                                                     const IndexT* carried,
                                                                                     Fp32ArgWidePartials partials,
                                                                                     uint64_t outer_size,
                                                                                     uint64_t reduction_size,
                                                                                     uint64_t inner_size,
                                                                                     uint64_t shards_per_output,
                                                                                     uint32_t rows_per_shard_log2,
                                                                                     uint32_t rows_per_warp_group,
                                                                                     uint32_t xor_mask,
                                                                                     uint64_t row_stride) {
    constexpr int PacketBytes = 16;
    static_assert(LanesPerComponent == 4 || LanesPerComponent == 8);
    static_assert(std::is_same_v<IndexT, uint32_t> || std::is_same_v<IndexT, uint64_t>);
    constexpr int MaxSupportedK = TILED_REDUCTION_WARP_THREADS / LanesPerComponent;

    __shared__ RawVectorPacket<PacketBytes> staged_packets[NARROW_FLAT_WARPS][NARROW_FP8_PACKETS_PER_WARP];
    __shared__ IndexT staged_indices[NARROW_FLAT_WARPS][NARROW_FP8_PACKETS_PER_WARP * 4];
    __shared__ uint32_t warp_keys[NARROW_FLAT_WARPS][MaxSupportedK];
    __shared__ uint64_t warp_tails[NARROW_FLAT_WARPS][MaxSupportedK];

    const uint32_t warp = threadIdx.x / TILED_REDUCTION_WARP_THREADS;
    const uint32_t lane = threadIdx.x % TILED_REDUCTION_WARP_THREADS;
    const uint32_t component = lane / LanesPerComponent;
    const uint32_t component_lane = lane % LanesPerComponent;
    const uint32_t inner_size_32 = static_cast<uint32_t>(inner_size);
    const uint32_t rows_per_block_round = rows_per_warp_group * NARROW_FLAT_WARPS;
    const uint32_t outer_count = static_cast<uint32_t>(outer_size);
    const uint32_t shard_count = static_cast<uint32_t>(shards_per_output);
    const uint64_t total_elements = outer_size * reduction_size * inner_size;
    const auto* staged_words = reinterpret_cast<const uint32_t*>(&staged_packets[warp][0]);
    const IndexT* staged_carried = &staged_indices[warp][0];
    for (uint32_t outer_index = blockIdx.z; outer_index < outer_count; outer_index += gridDim.z)
    for (uint32_t shard = blockIdx.x; shard < shard_count; shard += gridDim.x) {
        const uint64_t row_begin = static_cast<uint64_t>(shard) << rows_per_shard_log2;
        const uint32_t shard_rows = row_begin < reduction_size
            ? static_cast<uint32_t>(minU64(uint64_t{1} << rows_per_shard_log2, reduction_size - row_begin))
            : 0U;
        const float* shard_input = input + (outer_index * reduction_size + row_begin) * inner_size;

        uint32_t best_key = FP32_ARG_KEY_IDENTITY;
        uint64_t best_tail = FP32_ARG_TAIL_IDENTITY;
        for (uint32_t group_row = warp * rows_per_warp_group; group_row < shard_rows;
             group_row += rows_per_block_round) {
            const NarrowFp8GroupWindow<PacketBytes> window = narrowFp8GroupWindow<PacketBytes>(
                shard_input, group_row, shard_rows, rows_per_warp_group, inner_size_32);
            if (lane < window.packet_count) {
                staged_packets[warp][lane] = window.aligned_begin[lane];
                const uint64_t element =
                    static_cast<uint64_t>(reinterpret_cast<const float*>(window.aligned_begin) - input) + lane * 4U;
                stageDeepRCarriedPacket<IndexT>(&staged_indices[warp][0], lane,
                                                loadDeepRCarriedPacket<IndexT>(carried, element, total_elements));
            }
            __syncwarp();
            if (component < inner_size_32) {
                const uint32_t staged_step = LanesPerComponent * inner_size_32;
                uint32_t staged_index = window.prefix_elements + component_lane * inner_size_32 + component;
                uint64_t row_index = (row_begin + group_row + component_lane) * row_stride;
                const uint64_t row_index_step = static_cast<uint64_t>(LanesPerComponent) * row_stride;
                for (uint32_t row = component_lane; row < window.group_rows;
                     row += LanesPerComponent, staged_index += staged_step, row_index += row_index_step) {
                    fp32DeepRCarriedArgKeyElement(best_key, best_tail, staged_words[staged_index],
                                                  static_cast<uint64_t>(staged_carried[staged_index]), row_index,
                                                  xor_mask);
                }
            }
            __syncwarp();
        }

        // Fold the component's lanes (aligned groups of LanesPerComponent) with butterfly shuffles.
#pragma unroll
        for (uint32_t offset = 1; offset < static_cast<uint32_t>(LanesPerComponent); offset <<= 1) {
            const uint32_t other_key = __shfl_xor_sync(0xffffffffU, best_key, static_cast<int>(offset));
            const uint64_t other_tail =
                __shfl_xor_sync(0xffffffffU, static_cast<unsigned long long>(best_tail), static_cast<int>(offset));
            fp32ArgKeepMin(best_key, best_tail, other_key, other_tail);
        }
        if (component_lane == 0 && component < inner_size_32) {
            warp_keys[warp][component] = best_key;
            warp_tails[warp][component] = best_tail;
        }
        __syncthreads();
        if (threadIdx.x < inner_size_32) {
            uint32_t reduced_key = FP32_ARG_KEY_IDENTITY;
            uint64_t reduced_tail = FP32_ARG_TAIL_IDENTITY;
#pragma unroll
            for (int source_warp = 0; source_warp < NARROW_FLAT_WARPS; ++source_warp) {
                fp32ArgKeepMin(reduced_key, reduced_tail, warp_keys[source_warp][threadIdx.x],
                               warp_tails[source_warp][threadIdx.x]);
            }
            const uint64_t partial =
                (static_cast<uint64_t>(outer_index) * shards_per_output + shard) * inner_size + threadIdx.x;
            partials.keys[partial] = reduced_key;
            partials.tails[partial] = reduced_tail;
        }
        __syncthreads();
    }
}

// ARG-A3: RuntimeK argmin/argmax twin: the rotated sharded layout of awkwardAlignmentRotatedShardedFirstStageKernel
// (16 cooperative warps; grid x = shard, y = component tile, z = outer; each warp aligns its first row's tile window
// once and every revisited row, 16 rows later, stays packet-aligned) with uint32 candidates and a runtime shard depth.
// The logical tile is the warp's packet span less the alignment reserve (PacketBytes 16: 497 components, 8: 249), so
// one kernel serves both VALUE tile layouts. One shard (depth >= R) is a Complete pass; several shards write FP32 value
// + carried index partials. The index is (row base + row) * stride. Only the 8-byte packet is compiled: 1110 measured
// the 16-byte packet 1.56x slower (best p16 / best p8 over 2408 cells; p16 faster on 78).
constexpr int FP8_ARG_ROTATED_WARPS = 16;
constexpr int FP8_ARG_ROTATED_THREADS = FP8_ARG_ROTATED_WARPS * TILED_REDUCTION_WARP_THREADS;

template <int PacketBytes>
inline constexpr uint32_t fp8ArgRotatedTileComponents =
    static_cast<uint32_t>(TILED_REDUCTION_WARP_THREADS * PacketBytes - (PacketBytes - 1));

template <int PacketBytes>
__global__ __launch_bounds__(FP8_ARG_ROTATED_THREADS) void fp8RotatedArgKernel(const uint8_t* input,
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

// ARG-A3: RuntimeK PacketComplete argmin/argmax twin: fp8GeneralRParallelKTileP16CompleteKernel's topology (persistent
// 256-thread blocks; one work item = [outer, K tile]; W1/W2/W4/W8 warps per item split R round-robin; a warp loads one
// row's tile window with an alignment prefix and stages it through shared memory) with uint32 candidates.
// Complete: writes the requested outputs, index = row * stride.
//
// ARG-A3j (1127): 16-byte packets (32 lanes cover 512 bytes, prefix <= 15), so a tile holds up to 256 components (8
// per lane, the 1105 rule). The window is staged linearly (one vector store per loading lane) and lane L owns four
// adjacent components per quad q (components q * 128 + 4L .. +3): one word of keys is two LDS.32 and a funnel shift by
// the row's prefix (prefix & 3 is uniform across the warp), instead of four plane-major byte gathers of ~10
// instructions each. The tile is the fewest 128-component quads per row (fp8PacketCompleteArgTileComponents).
constexpr uint32_t FP8_ARG_PACKET_COMPLETE_MAX_TILE = 256;

template <int PacketBytes>
__global__ __launch_bounds__(SHALLOW_FP8_THREADS, 4) void fp8PacketCompleteArgKernel(const uint8_t* input,
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

    __shared__ __align__(16) uint32_t staged_packet_words[SHALLOW_FP8_WARPS][WindowWords];
    __shared__ uint32_t warp_partials[SHALLOW_FP8_WARPS][MaxTile];

    const uint32_t warp = threadIdx.x / TILED_REDUCTION_WARP_THREADS;
    const uint32_t lane = threadIdx.x % TILED_REDUCTION_WARP_THREADS;
    const uint32_t lane_component = 4U * lane;
    const uint32_t outputs_per_block = SHALLOW_FP8_WARPS / warps_per_output;
    const uint32_t output_group = warp / warps_per_output;
    const uint32_t warp_in_output = warp - output_group * warps_per_output;
    const uint32_t first_warp_in_output = output_group * warps_per_output;
    const uint64_t work_count = static_cast<uint64_t>(outer_size) * tile_count;
    const uint64_t block_work_stride = static_cast<uint64_t>(gridDim.x) * outputs_per_block;
    const uint64_t outer_input_stride = static_cast<uint64_t>(reduction_size) * inner_size;

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
            const uint64_t logical_begin = static_cast<uint64_t>(outer_index) * outer_input_stride +
                                           static_cast<uint64_t>(row_index) * inner_size + tile_begin;
            prefix_bytes = static_cast<uint32_t>(logical_begin & PacketMask);
            packet_count = (prefix_bytes + tile_width + PacketMask) >> PacketShift;
            if (lane < packet_count) {
                packet = *reinterpret_cast<const RawVectorPacket<PacketBytes>*>(
                    input + (logical_begin - prefix_bytes) + static_cast<uint64_t>(lane) * PacketBytes);
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

// Writes one reduced packed candidate (widened, 55-bit index) to the requested outputs at `output`.
__device__ __forceinline__ void storeFp8ArgPackedResult(const Fp8ArgOutputs& outputs, uint64_t output, uint64_t best) {
    if (outputs.value != nullptr) {
        const uint32_t byte = fp8ArgValueByte(fp8ArgWideKey(best), fp8ArgWideNegativeZero(best), outputs.fold);
        storeFp32AsRuntimeDType(outputs.value, outputs.value_dtype, output, decodeFp8ArgByte(byte, outputs.e5m2));
    }
    if (outputs.index != nullptr) {
        const uint64_t index = fp8ArgWideIndex(best) * outputs.domain_stride;
        if (outputs.index_dtype == DataType::UINT32) {
            static_cast<uint32_t*>(outputs.index)[output] = static_cast<uint32_t>(index);
        } else {
            static_cast<uint64_t*>(outputs.index)[output] = index;
        }
    }
}

// ARG-A3g (1116): reduces packed staged partials [outer, shards, K] (shard-relative lane candidates) over shards and
// writes the requested outputs. Shard s begins at s * base_rows + min(s, extra_rows) (uniform depths: extra_rows = 0).
// One thread per output component, consecutive threads on consecutive components (coalesced 4-byte reads per shard).
// 1135: used while there are enough outputs to cover the shard loop's latency (fp8ArgPackedContinuationSplitsShards).
template <int Unused>
__global__ __launch_bounds__(256) void fp8ArgPackedContinuationKernel(const uint32_t* packed,
                                                                     Fp8ArgOutputs outputs,
                                                                     uint64_t outer_size,
                                                                     uint64_t shards,
                                                                     uint64_t inner_size,
                                                                     ReductionShardRows shard_rows) {
    const uint64_t outputs_count = outer_size * inner_size;
    for (uint64_t output = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; output < outputs_count;
         output += static_cast<uint64_t>(gridDim.x) * blockDim.x) {
        const uint64_t outer_index = output / inner_size;
        const uint64_t component = output - outer_index * inner_size;
        const uint32_t* column = packed + outer_index * shards * inner_size + component;
        uint64_t best = FP8_ARG_WIDE_IDENTITY;
        for (uint32_t shard = 0; shard < static_cast<uint32_t>(shards); ++shard) {
            const uint32_t candidate = column[static_cast<uint64_t>(shard) * inner_size];
            best = fp8ArgMin(best, fp8ArgWiden(candidate, reductionShardRowBegin(shard_rows, shard)));
        }
        storeFp8ArgPackedResult(outputs, output, best);
    }
}

// 1135: the shard-split packed continuation, for few outputs with many shards (1134 audit: the per-output kernel ran
// 6-16x a copy of its partials at >= 256 shards and < 64 Ki outputs, up to half of the ARG time). A block owns one
// (outer, column tile) work item: `columns` consecutive components (K when K < 32, else 32) times `groups` shard
// groups; group g walks shards g, g + groups, ... with four independent loads in flight. For K < 32 a block's
// threads cover a contiguous [shards, K] slab, so reads stay coalesced. The groups combine in shared memory with the
// same order-free wide minimum (key, then index, then -0), so the result equals the per-output kernel's.
constexpr uint32_t FP8_ARG_PACKED_SPLIT_MAX_THREADS = 1024;
template <int Unused>
__global__ __launch_bounds__(FP8_ARG_PACKED_SPLIT_MAX_THREADS) void fp8ArgPackedContinuationSplitKernel(
    const uint32_t* packed,
    Fp8ArgOutputs outputs,
    uint64_t outer_size,
    uint64_t shards,
    uint64_t inner_size,
    ReductionShardRows shard_rows,
    uint32_t columns,
    uint32_t groups) {
    __shared__ uint64_t partial[FP8_ARG_PACKED_SPLIT_MAX_THREADS];
    const uint32_t lane_column = threadIdx.x % columns;
    const uint32_t group = threadIdx.x / columns;
    const uint32_t shard_count = static_cast<uint32_t>(shards);
    const uint64_t tiles = (inner_size + columns - 1) / columns;
    const uint64_t items = outer_size * tiles;
    for (uint64_t item = blockIdx.x; item < items; item += gridDim.x) {
        const uint64_t outer_index = item / tiles;
        const uint64_t component = (item - outer_index * tiles) * columns + lane_column;
        uint64_t best = FP8_ARG_WIDE_IDENTITY;
        if (group < groups && component < inner_size) {
            const uint32_t* column = packed + outer_index * shards * inner_size + component;
            uint64_t other = FP8_ARG_WIDE_IDENTITY;
            uint32_t shard = group;
            for (; shard + 3 * groups < shard_count; shard += 4 * groups) {
                const uint32_t s1 = shard + groups, s2 = s1 + groups, s3 = s2 + groups;
                const uint32_t c0 = column[static_cast<uint64_t>(shard) * inner_size];
                const uint32_t c1 = column[static_cast<uint64_t>(s1) * inner_size];
                const uint32_t c2 = column[static_cast<uint64_t>(s2) * inner_size];
                const uint32_t c3 = column[static_cast<uint64_t>(s3) * inner_size];
                best = fp8ArgMin(best, fp8ArgWiden(c0, reductionShardRowBegin(shard_rows, shard)));
                other = fp8ArgMin(other, fp8ArgWiden(c1, reductionShardRowBegin(shard_rows, s1)));
                best = fp8ArgMin(best, fp8ArgWiden(c2, reductionShardRowBegin(shard_rows, s2)));
                other = fp8ArgMin(other, fp8ArgWiden(c3, reductionShardRowBegin(shard_rows, s3)));
            }
            for (; shard < shard_count; shard += groups) {
                best = fp8ArgMin(best, fp8ArgWiden(column[static_cast<uint64_t>(shard) * inner_size],
                                                   reductionShardRowBegin(shard_rows, shard)));
            }
            best = fp8ArgMin(best, other);
        }
        partial[threadIdx.x] = best;
        __syncthreads();
        for (uint32_t active = groups; active > 1;) {
            const uint32_t half = (active + 1) / 2;
            if (group < active - half)
                partial[threadIdx.x] = fp8ArgMin(partial[threadIdx.x], partial[threadIdx.x + half * columns]);
            __syncthreads();
            active = half;
        }
        if (group == 0 && component < inner_size)
            storeFp8ArgPackedResult(outputs, outer_index * inner_size + component, partial[threadIdx.x]);
        __syncthreads();  // partial[] is rewritten by the next item
    }
}

// ARG-A3: AlignedK argmin/argmax twin: the K-parallel ownership of kParallelFp8{Packet,StagedHoistedCombine,
// CompleteUnified}Kernel (one thread owns one aligned 2- or 8-byte packet of components and walks its shard's rows; no
// cross-thread reduction; grid x = shard, y = K tile, z = outer; balanced shard rows). The CTA size is a launch
// operand, so two kernels serve VALUE's six packet/CTA pairs. One shard is Complete (requested outputs); several write
// FP32 value + carried index partials. The index is (shard row base + row) * stride.
// ARG-A3e (1113): VALUE's R loop and writeback. Rows are loaded in batches one batch ahead (P8: 4 rows, P2: 8 rows,
// K_PARALLEL_FP8_LOAD_AHEAD_ROWS); eight-byte packets key two words per row, two-byte packets pack two rows per key
// word. Results leave as 16-byte (P8, through a per-warp shared transpose, as VALUE's P8 writeback) or 8-byte (P2)
// lane-contiguous stores; 1111 stored one scalar per component, 8 bytes apart per lane (1.5x VALUE, 4.4x at R = 4).
constexpr int FP8_ARG_ALIGNED_MAX_THREADS = 128;
// VALUE's AlignedK load-ahead depth (K_PARALLEL_FP8_LOAD_AHEAD_ROWS lives in the K-parallel TU).
template <int PacketBytes>
inline constexpr uint32_t K_PARALLEL_FP8_ARG_LOAD_AHEAD_ROWS = PacketBytes >= 8 ? 4U : 8U;
constexpr int FP8_ARG_ALIGNED_STRIPE_PITCH = 33;

// Stores N (2 or 4) consecutive results starting at output_index: one vector store per output when the storage is
// FP32 values / UINT32 or UINT64 indices and the address is aligned, element stores otherwise.
template <int N>
__device__ __forceinline__ void storeFp8ArgResults(const Fp8ArgOutputs& outputs,
                                                   uint64_t output_index,
                                                   const uint32_t (&candidates)[N],
                                                   uint64_t row_base) {
    static_assert(N == 2 || N == 4);
    if (outputs.packed != nullptr) {
        uint32_t* destination = outputs.packed + output_index;
        if ((reinterpret_cast<uintptr_t>(destination) & uintptr_t{N * sizeof(uint32_t) - 1}) == 0) {
            if constexpr (N == 4) {
                *reinterpret_cast<uint4*>(destination) =
                    make_uint4(candidates[0], candidates[1], candidates[2], candidates[3]);
            } else {
                *reinterpret_cast<uint2*>(destination) = make_uint2(candidates[0], candidates[1]);
            }
        } else {
#pragma unroll
            for (int i = 0; i < N; ++i) destination[i] = candidates[i];
        }
        return;
    }
    if (outputs.value != nullptr) {
        float values[N];
#pragma unroll
        for (int i = 0; i < N; ++i) {
            values[i] = decodeFp8ArgByte(fp8ArgValueByte(candidates[i] >> 24, candidates[i] & 1u, outputs.fold),
                                         outputs.e5m2);
        }
        float* destination = static_cast<float*>(outputs.value) + output_index;
        if (outputs.value_dtype == DataType::FP32 &&
            (reinterpret_cast<uintptr_t>(destination) & uintptr_t{N * sizeof(float) - 1}) == 0) {
            if constexpr (N == 4) {
                *reinterpret_cast<float4*>(destination) = make_float4(values[0], values[1], values[2], values[3]);
            } else {
                *reinterpret_cast<float2*>(destination) = make_float2(values[0], values[1]);
            }
        } else {
#pragma unroll
            for (int i = 0; i < N; ++i) {
                storeFp32AsRuntimeDType(outputs.value, outputs.value_dtype, output_index + i, values[i]);
            }
        }
    }
    if (outputs.index != nullptr) {
        uint64_t indices[N];
#pragma unroll
        for (int i = 0; i < N; ++i) {
            indices[i] = (row_base + ((candidates[i] >> 1) & (FP8_ARG_MAX_LANE_ROWS - 1u))) * outputs.domain_stride;
        }
        if (outputs.index_dtype == DataType::UINT32) {
            uint32_t* destination = static_cast<uint32_t*>(outputs.index) + output_index;
            if ((reinterpret_cast<uintptr_t>(destination) & uintptr_t{N * sizeof(uint32_t) - 1}) == 0) {
                if constexpr (N == 4) {
                    *reinterpret_cast<uint4*>(destination) =
                        make_uint4(static_cast<uint32_t>(indices[0]), static_cast<uint32_t>(indices[1]),
                                   static_cast<uint32_t>(indices[2]), static_cast<uint32_t>(indices[3]));
                } else {
                    *reinterpret_cast<uint2*>(destination) =
                        make_uint2(static_cast<uint32_t>(indices[0]), static_cast<uint32_t>(indices[1]));
                }
            } else {
#pragma unroll
                for (int i = 0; i < N; ++i) destination[i] = static_cast<uint32_t>(indices[i]);
            }
        } else {
            uint64_t* destination = static_cast<uint64_t*>(outputs.index) + output_index;
            if ((reinterpret_cast<uintptr_t>(destination) & uintptr_t{15}) == 0) {
#pragma unroll
                for (int i = 0; i < N; i += 2) {
                    *reinterpret_cast<uint4*>(destination + i) =
                        make_uint4(static_cast<uint32_t>(indices[i]), static_cast<uint32_t>(indices[i] >> 32),
                                   static_cast<uint32_t>(indices[i + 1]), static_cast<uint32_t>(indices[i + 1] >> 32));
                }
            } else {
#pragma unroll
                for (int i = 0; i < N; ++i) destination[i] = indices[i];
            }
        }
    }
}

template <int PacketBytes>
__global__ __launch_bounds__(FP8_ARG_ALIGNED_MAX_THREADS) void fp8AlignedKArgKernel(const uint8_t* input,
                                                                                   Fp8ArgOutputs outputs,
                                                                                   uint64_t outer_size,
                                                                                   uint64_t reduction_size,
                                                                                   uint64_t inner_size,
                                                                                   uint64_t shards_per_output,
                                                                                   uint64_t component_tiles,
                                                                                   ReductionShardRows shard_rows) {
    static_assert(PacketBytes == 2 || PacketBytes == 8);
    constexpr uint32_t Identity = FP8_ARG_LANE_IDENTITY;
    // VALUE's load-ahead depth; a P2 batch is an even number of rows (two rows per key word).
    constexpr uint32_t Rows = K_PARALLEL_FP8_ARG_LOAD_AHEAD_ROWS<PacketBytes>;
    static_assert(PacketBytes == 8 || Rows % 2 == 0);
    using Word = std::conditional_t<PacketBytes == 8, uint2, uint32_t>;
    // P8 writeback transpose: warp_tiles[warp][item][lane] holds lane's candidate for component lane * 8 + item.
    __shared__ uint32_t warp_tiles[FP8_ARG_ALIGNED_MAX_THREADS / TILED_REDUCTION_WARP_THREADS][PacketBytes]
                                  [FP8_ARG_ALIGNED_STRIPE_PITCH];
    const uint32_t warp = threadIdx.x / TILED_REDUCTION_WARP_THREADS;
    const uint32_t lane = threadIdx.x % TILED_REDUCTION_WARP_THREADS;
    const uint64_t components_per_cta = static_cast<uint64_t>(blockDim.x) * PacketBytes;
    const uint32_t outer_count = static_cast<uint32_t>(outer_size);
    const uint32_t tile_count = static_cast<uint32_t>(component_tiles);
    const uint32_t shard_count = static_cast<uint32_t>(shards_per_output);
    for (uint32_t outer_index = blockIdx.z; outer_index < outer_count; outer_index += gridDim.z)
    for (uint32_t component_tile = blockIdx.y; component_tile < tile_count; component_tile += gridDim.y)
    for (uint32_t shard = blockIdx.x; shard < shard_count; shard += gridDim.x) {
        const uint64_t tile_begin = component_tile * components_per_cta;
        const uint64_t component = tile_begin + static_cast<uint64_t>(threadIdx.x) * PacketBytes;
        const bool owns_packet = component < inner_size;
        const uint64_t row_begin = reductionShardRowBegin(shard_rows, shard);
        const uint32_t row_count = reductionShardRowCount(shard_rows, shard);
        uint32_t local[PacketBytes];
#pragma unroll
        for (int item = 0; item < PacketBytes; ++item) local[item] = Identity;
        if (owns_packet) {
            const uint8_t* packet = input + (outer_index * reduction_size + row_begin) * inner_size + component;
            const auto load = [&](const uint8_t* at) -> Word {
                if constexpr (PacketBytes == 8) {
                    return reinterpret_cast<const RawVectorPacket<8>*>(at)->value;
                } else {
                    return reinterpret_cast<const RawVectorPacket<2>*>(at)->value;
                }
            };
            const auto key_batch = [&](const Word (&batch)[Rows], uint32_t first_row) {
                if constexpr (PacketBytes == 8) {
#pragma unroll
                    for (uint32_t r = 0; r < Rows; ++r) {
                        const Fp8ArgKeys4 low = fp8ArgKeys4(batch[r].x, outputs.fold);
                        const Fp8ArgKeys4 high = fp8ArgKeys4(batch[r].y, outputs.fold);
#pragma unroll
                        for (uint32_t b = 0; b < 4U; ++b) {
                            local[b] = fp8ArgMin(local[b], fp8ArgLaneCandidate(low, b, first_row + r));
                            local[4 + b] = fp8ArgMin(local[4 + b], fp8ArgLaneCandidate(high, b, first_row + r));
                        }
                    }
                } else {
#pragma unroll
                    for (uint32_t r = 0; r < Rows; r += 2) {
                        const Fp8ArgKeys4 keys = fp8ArgKeys4(batch[r] | (batch[r + 1] << 16), outputs.fold);
                        local[0] = fp8ArgMin(local[0], fp8ArgLaneCandidate(keys, 0, first_row + r));
                        local[1] = fp8ArgMin(local[1], fp8ArgLaneCandidate(keys, 1, first_row + r));
                        local[0] = fp8ArgMin(local[0], fp8ArgLaneCandidate(keys, 2, first_row + r + 1));
                        local[1] = fp8ArgMin(local[1], fp8ArgLaneCandidate(keys, 3, first_row + r + 1));
                    }
                }
            };
            const auto load_batch = [&](Word (&batch)[Rows]) {
#pragma unroll
                for (uint32_t r = 0; r < Rows; ++r) batch[r] = load(packet + r * inner_size);
                packet += Rows * inner_size;
            };
            // Load-ahead as accumulateKParallelFp8Rows: the next batch is in flight while the current one is keyed.
            uint32_t row = 0;
            if (row_count >= Rows) {
                Word current[Rows];
                load_batch(current);
                for (row = Rows; row + Rows <= row_count; row += Rows) {
                    Word next[Rows];
                    load_batch(next);
                    key_batch(current, row - Rows);
#pragma unroll
                    for (uint32_t r = 0; r < Rows; ++r) current[r] = next[r];
                }
                key_batch(current, row - Rows);
            }
            for (; row < row_count; ++row) {
                const Word word = load(packet);
                packet += inner_size;
                if constexpr (PacketBytes == 8) {
                    const Fp8ArgKeys4 low = fp8ArgKeys4(word.x, outputs.fold);
                    const Fp8ArgKeys4 high = fp8ArgKeys4(word.y, outputs.fold);
#pragma unroll
                    for (uint32_t b = 0; b < 4U; ++b) {
                        local[b] = fp8ArgMin(local[b], fp8ArgLaneCandidate(low, b, row));
                        local[4 + b] = fp8ArgMin(local[4 + b], fp8ArgLaneCandidate(high, b, row));
                    }
                } else {
                    const Fp8ArgKeys4 keys = fp8ArgKeys4(word, outputs.fold);
                    local[0] = fp8ArgMin(local[0], fp8ArgLaneCandidate(keys, 0, row));
                    local[1] = fp8ArgMin(local[1], fp8ArgLaneCandidate(keys, 1, row));
                }
            }
        }
        const uint64_t output_row = (static_cast<uint64_t>(outer_index) * shards_per_output + shard) * inner_size;
        if constexpr (PacketBytes == 2) {
            if (owns_packet) storeFp8ArgResults<2>(outputs, output_row + component, local, row_begin);
        } else {
            // K is a multiple of 8, so a warp's valid components are whole packets and whole 4-component chunks.
#pragma unroll
            for (int item = 0; item < PacketBytes; ++item) warp_tiles[warp][item][lane] = local[item];
            __syncwarp();
            const uint64_t warp_begin = tile_begin + static_cast<uint64_t>(warp) * TILED_REDUCTION_WARP_THREADS *
                                                         PacketBytes;
#pragma unroll
            for (uint32_t round = 0; round < 2U; ++round) {
                const uint32_t offset = round * TILED_REDUCTION_WARP_THREADS * 4U + lane * 4U;
                if (warp_begin + offset < inner_size) {
                    uint32_t chunk[4];
#pragma unroll
                    for (uint32_t i = 0; i < 4U; ++i) {
                        const uint32_t c = offset + i;
                        chunk[i] = warp_tiles[warp][c % PacketBytes][c / PacketBytes];
                    }
                    storeFp8ArgResults<4>(outputs, output_row + warp_begin + offset, chunk, row_begin);
                }
            }
            __syncwarp();
        }
    }
}

// Invokes callback.template operator()<OutputsPerWarp, PacketBytes, MaxComponentsPerLane, LoadAhead>() for the
// OneWindow unit equal to `unit` (the ARG twins exist for exactly the compiled VALUE units).
template <typename Callback>
[[nodiscard]] bool dispatchOneWindowArg(const Fp8OneWindowRuntimeKUnit& unit, Callback&& callback) {
    bool dispatched = false;
    [&]<size_t... I>(std::index_sequence<I...>) {
        static_cast<void>(((ONE_WINDOW_RUNTIME_K_UNITS[I] == unit &&
                            (callback.template operator()<
                                 static_cast<int>(ONE_WINDOW_RUNTIME_K_UNITS[I].outputs_per_warp),
                                 static_cast<int>(ONE_WINDOW_RUNTIME_K_UNITS[I].packet_bytes),
                                 static_cast<int>(ONE_WINDOW_RUNTIME_K_UNITS[I].max_components_per_lane),
                                 ONE_WINDOW_RUNTIME_K_UNITS[I].load_ahead>(),
                             dispatched = true)) ||
                           ...));
    }(std::make_index_sequence<ONE_WINDOW_RUNTIME_K_UNITS.size()>{});
    return dispatched;
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

bool fp32DeepRSmallKSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size, uint64_t rows) {
    const auto& depths = Fp8RK::Fp8DeepRSmallKPolicy::compiled_rows;
    if (outer_size == 0 || inner_size == 0 || inner_size > 125 || reduction_size <= rows ||
        std::find(depths.begin(), depths.end(), rows) == depths.end()) {
        return false;
    }
    // reductionWorkGrid's work extents (shards and outers) stay below 2^31.
    const uint64_t shards = ceilDivideU64(reduction_size, rows);
    return shards <= Fp8RK::FP8_RK_MAX_GRID_EXTENT && outer_size <= Fp8RK::FP8_RK_MAX_GRID_EXTENT;
}

void launchFp32DeepRSmallKRetainedFirstStage(const CubReductionStageSemantics& semantics,
                                             const Tensor& input, Tensor& fp32_partials,
                                             const CubReductionGeometry& geometry, uint64_t rows,
                                             uint64_t shards, Stream& stream) {
#if THOR_CUB_ENABLE_FP8_TYPES
    if (input.getDataType() != DataType::FP32 ||
        !fp32DeepRSmallKSupports(geometry.outer_size, geometry.reduction_size, geometry.inner_size, rows) ||
        shards != ceilDivideU64(geometry.reduction_size, rows) ||
        input.getDimensions() !=
            std::vector<uint64_t>{geometry.outer_size, geometry.reduction_size, geometry.inner_size} ||
        (reinterpret_cast<uintptr_t>(input.getMemPtr<void>()) & uintptr_t{15}) != 0 ||
        fp32_partials.getDataType() != DataType::FP32 ||
        fp32_partials.getDimensions() != std::vector<uint64_t>{geometry.outer_size, shards, geometry.inner_size}) {
        throw std::logic_error("FP32 DeepRSmallK first stage received illegal geometry, input or partials.");
    }
    // Raw FP32 partials, as FP8's; the continuation finalizes.
    const auto launch = [&]<typename InputT, typename ReductionOpT, typename TransformT>() {
        static_assert(std::is_same_v<InputT, float>);
        dispatchFp32DeepRRetainedRows<TransformT, ReductionOpT>(input, fp32_partials, geometry, rows, shards,
                                                                stream.getStream());
    };
    dispatchDeepRRetainedCombine<float>(semantics, launch);
#else
    static_cast<void>(semantics);
    static_cast<void>(input);
    static_cast<void>(fp32_partials);
    static_cast<void>(geometry);
    static_cast<void>(rows);
    static_cast<void>(shards);
    static_cast<void>(stream);
    throw std::logic_error("FP32 DeepRSmallK requires THOR_CUB_ENABLE_FP8_TYPES (the retained kernels' gate).");
#endif
}

CubKernelOccupancyInfo queryFp32DeepRSmallKRetainedOccupancy(uint64_t rows, uint64_t k,
                                                            const CubReductionStageSemantics& semantics) {
#if THOR_CUB_ENABLE_FP8_TYPES
    const auto& depths = Fp8RK::Fp8DeepRSmallKPolicy::compiled_rows;
    if (k == 0 || k > 125 || std::find(depths.begin(), depths.end(), rows) == depths.end())
        throw std::logic_error("FP32 DeepRSmallK occupancy requires a retained depth and K <= 125.");
    const auto query = [&]<typename InputT, typename ReductionOpT, typename TransformT>() -> CubKernelOccupancyInfo {
        return queryFp32DeepRRetainedRows<TransformT, ReductionOpT>(rows, k);
    };
    return dispatchDeepRRetainedCombine<float>(semantics, query);
#else
    static_cast<void>(rows);
    static_cast<void>(k);
    static_cast<void>(semantics);
    throw std::logic_error("FP32 DeepRSmallK requires THOR_CUB_ENABLE_FP8_TYPES (the retained kernels' gate).");
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
        geometry.inner_size > Fp8RK::FP8_RK_MAX_UINT32 ||
        // 1131: the kernel's input and output offsets are 64-bit; only outer, R and K are 32-bit operands, and the
        // 32-bit row counter steps by W <= 8 (formerly shallowFp8FitsUint32Indexing: outer * R * K <= 2^32 - 1).
        geometry.outer_size > Fp8RK::FP8_RK_MAX_UINT32 ||
        geometry.reduction_size > Fp8RK::FP8_RK_PACKET_COMPLETE_MAX_ROWS ||
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

static_assert(FP32_PACKET_COMPLETE_MAX_TILE_COMPONENTS == 124, "the FP32 PacketComplete kernel's tile cap");

// 1190: RuntimeK PacketComplete on FP32 input: the shared kernel at InputT = float (tile <= 124 components), the same
// W1/W2/W4/W8 ownership and three compiled transforms per combine (12 kernels). Not routed yet.
bool fp32PacketCompleteSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size,
                                uint32_t warps_per_output, uint32_t tile_components) {
    return outer_size != 0 && reduction_size != 0 && inner_size != 0 && outer_size <= Fp8RK::FP8_RK_MAX_UINT32 &&
           reduction_size <= Fp8RK::FP8_RK_PACKET_COMPLETE_MAX_ROWS && inner_size <= Fp8RK::FP8_RK_MAX_UINT32 &&
           (warps_per_output == 1 || warps_per_output == 2 || warps_per_output == 4 || warps_per_output == 8) &&
           tile_components != 0 && tile_components <= FP32_PACKET_COMPLETE_MAX_TILE_COMPONENTS;
}

namespace {

template <int FixedOp, typename Callback>
[[nodiscard]] bool dispatchFp32PacketComplete(CubReductionStageInputTransform transform, Callback&& callback) {
    switch (transform) {
        case CubReductionStageInputTransform::Identity:
            callback(fp8GeneralRParallelKTileP16CompleteKernel<float, 0, FixedOp>);
            return true;
        case CubReductionStageInputTransform::AbsoluteValue:
            callback(fp8GeneralRParallelKTileP16CompleteKernel<float, 1, FixedOp>);
            return true;
        case CubReductionStageInputTransform::Square:
            callback(fp8GeneralRParallelKTileP16CompleteKernel<float, 2, FixedOp>);
            return true;
    }
    return false;
}

template <int FixedOp>
void launchFp32PacketCompleteTyped(const Tensor& input,
                                   Tensor& output,
                                   const CubReductionGeometry& geometry,
                                   uint32_t warps_per_output,
                                   uint32_t tile_components,
                                   Stream& stream,
                                   const CubReductionStageSemantics& semantics,
                                   float output_scale) {
    const AdditiveFinalizeFp32 output_finalize = fp8CompleteFinalizerFromSemantics(semantics);
    if (input.getDataType() != DataType::FP32 ||
        !fp32PacketCompleteSupports(geometry.outer_size, geometry.reduction_size, geometry.inner_size,
                                    warps_per_output, tile_components) ||
        input.getTotalNumElements() != geometry.input_elements ||
        !fp8CompleteSupportsOutputStorage(output.getDataType()) || output.getDimensions() != geometry.output_dimensions) {
        throw std::logic_error("FP32 RuntimeK PacketComplete received unsupported geometry or outputs.");
    }
    if ((reinterpret_cast<uintptr_t>(input.getMemPtr<void>()) & uintptr_t{15}) != 0) {
        throw std::logic_error("FP32 RuntimeK PacketComplete requires a 16-byte aligned input base.");
    }
    const uint32_t tile_count =
        static_cast<uint32_t>(ceilDivideU64(geometry.inner_size, static_cast<uint64_t>(tile_components)));
    const uint32_t outputs_per_block = SHALLOW_FP8_WARPS / warps_per_output;
    const uint64_t work_count = geometry.outer_size * static_cast<uint64_t>(tile_count);
    const uint64_t block_count = std::min<uint64_t>(
        ceilDivideU64(work_count, static_cast<uint64_t>(outputs_per_block)), NARROW_FLAT_MAX_GRID_BLOCKS);
    const bool launched = dispatchFp32PacketComplete<FixedOp>(semantics.input_transform, [&](auto kernel) {
        kernel<<<static_cast<unsigned int>(block_count), SHALLOW_FP8_THREADS, 0, stream.getStream()>>>(
            input.getMemPtr<float>(),
            output.getMemPtr<void>(),
            output.getDataType(),
            static_cast<uint32_t>(geometry.outer_size),
            static_cast<uint32_t>(geometry.reduction_size),
            static_cast<uint32_t>(geometry.inner_size),
            tile_components,
            tile_count,
            warps_per_output,
            output_finalize,
            CubReductionStageInputTransform::Identity,
            output_scale);
        CUDA_CHECK(cudaGetLastError());
    });
    if (!launched) {
        throw std::logic_error("FP32 RuntimeK PacketComplete received an unsupported input transform.");
    }
}

template <int FixedOp>
CubKernelOccupancyInfo queryFp32PacketCompleteOccupancyTyped(CubReductionStageInputTransform transform) {
    CubKernelOccupancyInfo occupancy{};
    const bool queried = dispatchFp32PacketComplete<FixedOp>(
        transform, [&](auto kernel) { occupancy = queryShallowFp8Occupancy(kernel); });
    if (!queried) {
        throw std::logic_error("FP32 RuntimeK PacketComplete occupancy requires a supported input transform.");
    }
    return occupancy;
}

}  // namespace

void launchFp32PacketComplete(const Tensor& input,
                              Tensor& output,
                              const CubReductionGeometry& geometry,
                              uint32_t warps_per_output,
                              uint32_t tile_components,
                              Stream& stream,
                              const CubReductionStageSemantics& semantics,
                              float output_scale) {
    switch (semantics.combine) {
        case CubReductionStageCombine::Sum:
            return launchFp32PacketCompleteTyped<0>(input, output, geometry, warps_per_output, tile_components, stream,
                                                    semantics, output_scale);
        case CubReductionStageCombine::Product:
            return launchFp32PacketCompleteTyped<1>(input, output, geometry, warps_per_output, tile_components, stream,
                                                    semantics, output_scale);
        case CubReductionStageCombine::Minimum:
            return launchFp32PacketCompleteTyped<2>(input, output, geometry, warps_per_output, tile_components, stream,
                                                    semantics, output_scale);
        case CubReductionStageCombine::Maximum:
            return launchFp32PacketCompleteTyped<3>(input, output, geometry, warps_per_output, tile_components, stream,
                                                    semantics, output_scale);
    }
    throw std::logic_error("Invalid FP32 RuntimeK PacketComplete combine operation.");
}

// ---- 1201: the FP32 RK ARG PacketComplete twin (see fp32PacketCompleteArgKernel).
bool fp32PacketCompleteArgSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size,
                                   uint32_t warps_per_output, uint32_t tile_components) {
    return fp32PacketCompleteSupports(outer_size, reduction_size, inner_size, warps_per_output, tile_components) &&
           reduction_size <= FP32_ARG_MAX_SHARD_ROWS;
}

void launchFp32PacketCompleteArg(const Tensor& input,
                                 Tensor* value_output,
                                 Tensor* index_output,
                                 const CubReductionGeometry& geometry,
                                 uint32_t warps_per_output,
                                 uint32_t tile_components,
                                 bool maximum,
                                 uint64_t domain_stride,
                                 Stream& stream) {
    const uint64_t outputs = geometry.outer_size * geometry.inner_size;
    if (input.getDataType() != DataType::FP32 ||
        !fp32PacketCompleteArgSupports(geometry.outer_size, geometry.reduction_size, geometry.inner_size,
                                       warps_per_output, tile_components) ||
        input.getTotalNumElements() != geometry.outer_size * geometry.reduction_size * geometry.inner_size ||
        (value_output == nullptr && index_output == nullptr) ||
        (value_output != nullptr && (value_output->getTotalNumElements() != outputs ||
                                     !fp8CompleteSupportsOutputStorage(value_output->getDataType()))) ||
        (index_output != nullptr && (index_output->getTotalNumElements() != outputs ||
                                     (index_output->getDataType() != DataType::UINT32 &&
                                      index_output->getDataType() != DataType::UINT64)))) {
        throw std::logic_error("FP32 PacketComplete ARG received unsupported geometry or outputs.");
    }
    if ((reinterpret_cast<uintptr_t>(input.getMemPtr<void>()) & uintptr_t{15}) != 0) {
        throw std::logic_error("FP32 PacketComplete ARG requires a 16-byte aligned input base.");
    }
    if (index_output != nullptr && index_output->getDataType() == DataType::UINT32 &&
        (geometry.reduction_size - 1) * domain_stride > std::numeric_limits<uint32_t>::max()) {
        throw std::logic_error("FP32 PacketComplete ARG UINT32 index cannot hold the original-domain index.");
    }
    const Fp32ArgOutputs arg_outputs{value_output == nullptr ? nullptr : value_output->getMemPtr<void>(),
                                     value_output == nullptr ? DataType::FP32 : value_output->getDataType(),
                                     index_output == nullptr ? nullptr : index_output->getMemPtr<void>(),
                                     index_output == nullptr ? DataType::UINT32 : index_output->getDataType(),
                                     domain_stride,
                                     maximum ? 0x80000000U : 0U};
    const uint32_t tile_count =
        static_cast<uint32_t>(ceilDivideU64(geometry.inner_size, static_cast<uint64_t>(tile_components)));
    const uint64_t outputs_per_block = SHALLOW_FP8_WARPS / warps_per_output;
    const uint64_t block_count =
        std::min<uint64_t>(ceilDivideU64(geometry.outer_size * tile_count, outputs_per_block),
                           NARROW_FLAT_MAX_GRID_BLOCKS);
    fp32PacketCompleteArgKernel<0><<<static_cast<unsigned int>(block_count), SHALLOW_FP8_THREADS, 0,
                                     stream.getStream()>>>(
        input.getMemPtr<float>(), arg_outputs, static_cast<uint32_t>(geometry.outer_size),
        static_cast<uint32_t>(geometry.reduction_size), static_cast<uint32_t>(geometry.inner_size), tile_components,
        tile_count, warps_per_output);
    CUDA_CHECK(cudaGetLastError());
}

CubKernelOccupancyInfo queryFp32PacketCompleteArgOccupancy() {
    return queryShallowFp8Occupancy(fp32PacketCompleteArgKernel<0>);
}

CubKernelOccupancyInfo queryFp32PacketCompleteOccupancy(CubReductionStageInputTransform transform,
                                                        CubReductionStageCombine combine) {
    switch (combine) {
        case CubReductionStageCombine::Sum: return queryFp32PacketCompleteOccupancyTyped<0>(transform);
        case CubReductionStageCombine::Product: return queryFp32PacketCompleteOccupancyTyped<1>(transform);
        case CubReductionStageCombine::Minimum: return queryFp32PacketCompleteOccupancyTyped<2>(transform);
        case CubReductionStageCombine::Maximum: return queryFp32PacketCompleteOccupancyTyped<3>(transform);
    }
    throw std::logic_error("Invalid FP32 RuntimeK PacketComplete occupancy combine operation.");
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

std::vector<Fp8OneWindowRuntimeKUnit> fp32OneWindowRuntimeKUnits(uint64_t inner_size) {
    std::vector<Fp8OneWindowRuntimeKUnit> units;
    for (const Fp8OneWindowRuntimeKUnit& unit : ONE_WINDOW_RUNTIME_K_UNITS) {
        if (fp32OneWindowRuntimeKUnitPlan(inner_size, unit)) {
            units.push_back(unit);
        }
    }
    return units;
}

void launchFp32OneWindowRuntimeKComplete(const Tensor& input,
                                         Tensor& output,
                                         const CubReductionGeometry& geometry,
                                         const Fp8OneWindowRuntimeKUnit& unit,
                                         uint32_t warps_per_output,
                                         Stream& stream,
                                         const CubReductionStageSemantics& semantics,
                                         float output_scale) {
#if !THOR_CUB_ENABLE_FP8_TYPES
    static_cast<void>(input);
    static_cast<void>(output);
    static_cast<void>(geometry);
    static_cast<void>(unit);
    static_cast<void>(warps_per_output);
    static_cast<void>(stream);
    static_cast<void>(output_scale);
    throw std::logic_error("FP32 OneWindow requires THOR_CUB_ENABLE_FP8_TYPES (the retained kernels' gate).");
#else
    switch (semantics.combine) {
        case CubReductionStageCombine::Sum:
            return launchFp32OneWindowRuntimeKCompleteTyped<0>(input, output, geometry, unit, warps_per_output, stream,
                                                               semantics, output_scale);
        case CubReductionStageCombine::Product:
            return launchFp32OneWindowRuntimeKCompleteTyped<1>(input, output, geometry, unit, warps_per_output, stream,
                                                               semantics, output_scale);
        case CubReductionStageCombine::Minimum:
            return launchFp32OneWindowRuntimeKCompleteTyped<2>(input, output, geometry, unit, warps_per_output, stream,
                                                               semantics, output_scale);
        case CubReductionStageCombine::Maximum:
            return launchFp32OneWindowRuntimeKCompleteTyped<3>(input, output, geometry, unit, warps_per_output, stream,
                                                               semantics, output_scale);
    }
    throw std::logic_error("Invalid FP32 OneWindow combine operation.");
#endif
}

CubKernelOccupancyInfo queryFp32OneWindowRuntimeKOccupancy(uint64_t inner_size,
                                                           const Fp8OneWindowRuntimeKUnit& unit,
                                                           uint32_t warps_per_output,
                                                           CubReductionStageCombine combine) {
#if !THOR_CUB_ENABLE_FP8_TYPES
    static_cast<void>(inner_size);
    static_cast<void>(unit);
    static_cast<void>(warps_per_output);
    static_cast<void>(combine);
    throw std::logic_error("FP32 OneWindow requires THOR_CUB_ENABLE_FP8_TYPES (the retained kernels' gate).");
#else
    switch (combine) {
        case CubReductionStageCombine::Sum:
            return queryFp32OneWindowRuntimeKOccupancyTyped<0>(inner_size, unit, warps_per_output);
        case CubReductionStageCombine::Product:
            return queryFp32OneWindowRuntimeKOccupancyTyped<1>(inner_size, unit, warps_per_output);
        case CubReductionStageCombine::Minimum:
            return queryFp32OneWindowRuntimeKOccupancyTyped<2>(inner_size, unit, warps_per_output);
        case CubReductionStageCombine::Maximum:
            return queryFp32OneWindowRuntimeKOccupancyTyped<3>(inner_size, unit, warps_per_output);
    }
    throw std::logic_error("Invalid FP32 OneWindow occupancy combine operation.");
#endif
}

// ---- 1202: the FP32 RK ARG OneWindow twin (see fp32OneWindowArgKernel).
bool fp32OneWindowArgSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size,
                              const Fp8OneWindowRuntimeKUnit& unit, uint32_t warps_per_output) {
    if (outer_size == 0 || reduction_size == 0 || reduction_size > FP32_ARG_MAX_SHARD_ROWS ||
        !fp32OneWindowRuntimeKUnitPlan(inner_size, unit) ||
        !oneWindowValidWarpsPerOutput(unit.outputs_per_warp, warps_per_output)) {
        return false;
    }
    constexpr uint64_t MaxIndex = std::numeric_limits<uint32_t>::max();
    // The kernel indexes input bytes in 32 bits (fp32OneWindowFitsIndexing).
    return outer_size <= MaxIndex && outer_size * reduction_size * inner_size <= MaxIndex / 4;
}

void launchFp32OneWindowArg(const Tensor& input,
                            Tensor* value_output,
                            Tensor* index_output,
                            const CubReductionGeometry& geometry,
                            const Fp8OneWindowRuntimeKUnit& unit,
                            uint32_t warps_per_output,
                            bool maximum,
                            uint64_t domain_stride,
                            Stream& stream) {
    const uint64_t outputs = geometry.outer_size * geometry.inner_size;
    const auto plan = fp32OneWindowRuntimeKUnitPlan(geometry.inner_size, unit);
    if (input.getDataType() != DataType::FP32 || !plan ||
        !fp32OneWindowArgSupports(geometry.outer_size, geometry.reduction_size, geometry.inner_size, unit,
                                  warps_per_output) ||
        input.getTotalNumElements() != geometry.outer_size * geometry.reduction_size * geometry.inner_size ||
        (value_output == nullptr && index_output == nullptr) ||
        (value_output != nullptr && (value_output->getTotalNumElements() != outputs ||
                                     !fp8CompleteSupportsOutputStorage(value_output->getDataType()))) ||
        (index_output != nullptr && (index_output->getTotalNumElements() != outputs ||
                                     (index_output->getDataType() != DataType::UINT32 &&
                                      index_output->getDataType() != DataType::UINT64)))) {
        throw std::logic_error("FP32 OneWindow ARG received unsupported geometry or outputs.");
    }
    if ((reinterpret_cast<uintptr_t>(input.getMemPtr<void>()) & uintptr_t{15}) != 0) {
        throw std::logic_error("FP32 OneWindow ARG requires a 16-byte aligned input base.");
    }
    if (index_output != nullptr && index_output->getDataType() == DataType::UINT32 &&
        (geometry.reduction_size - 1) * domain_stride > std::numeric_limits<uint32_t>::max()) {
        throw std::logic_error("FP32 OneWindow ARG UINT32 index cannot hold the original-domain index.");
    }
    const Fp32ArgOutputs arg_outputs{value_output == nullptr ? nullptr : value_output->getMemPtr<void>(),
                                     value_output == nullptr ? DataType::FP32 : value_output->getDataType(),
                                     index_output == nullptr ? nullptr : index_output->getMemPtr<void>(),
                                     index_output == nullptr ? DataType::UINT32 : index_output->getDataType(),
                                     domain_stride,
                                     maximum ? 0x80000000U : 0U};
    const uint64_t outputs_per_block = (SHALLOW_FP8_WARPS / warps_per_output) * unit.outputs_per_warp;
    const uint64_t block_count = std::min<uint64_t>(ceilDivideU64(geometry.outer_size, outputs_per_block),
                                                    NARROW_FLAT_MAX_GRID_BLOCKS);
    const bool launched = dispatchFp32OneWindowRuntimeK(
        unit, [&]<typename InputT, int OutputsPerWarp, int PacketBytes, int MaxComponentsPerLane, bool LoadAhead>() {
            static_assert(std::is_same_v<InputT, float> && PacketBytes == 16);
            fp32OneWindowArgKernel<OutputsPerWarp, MaxComponentsPerLane, LoadAhead>
                <<<static_cast<unsigned int>(block_count), SHALLOW_FP8_THREADS, 0, stream.getStream()>>>(
                    input.getMemPtr<float>(), arg_outputs, static_cast<uint32_t>(geometry.outer_size),
                    static_cast<uint32_t>(geometry.reduction_size), warps_per_output, plan->geometry);
            CUDA_CHECK(cudaGetLastError());
        });
    if (!launched) {
        throw std::logic_error("FP32 OneWindow ARG has no kernel for this request.");
    }
}

CubKernelOccupancyInfo queryFp32OneWindowArgOccupancy(uint64_t inner_size,
                                                      const Fp8OneWindowRuntimeKUnit& unit,
                                                      uint32_t warps_per_output) {
    if (!fp32OneWindowRuntimeKUnitPlan(inner_size, unit) ||
        !oneWindowValidWarpsPerOutput(unit.outputs_per_warp, warps_per_output)) {
        throw std::logic_error("FP32 OneWindow ARG occupancy received unsupported ownership.");
    }
    CubKernelOccupancyInfo occupancy{};
    const bool queried = dispatchFp32OneWindowRuntimeK(
        unit, [&]<typename InputT, int OutputsPerWarp, int PacketBytes, int MaxComponentsPerLane, bool LoadAhead>() {
            static_assert(std::is_same_v<InputT, float> && PacketBytes == 16);
            occupancy =
                queryShallowFp8Occupancy(fp32OneWindowArgKernel<OutputsPerWarp, MaxComponentsPerLane, LoadAhead>);
        });
    if (!queried) {
        throw std::logic_error("FP32 OneWindow ARG occupancy has no kernel for this request.");
    }
    return occupancy;
}

// ---- 1212: the carried-index OneWindow ARG twin (see fp32OneWindowCarriedArgKernel).
bool fp32OneWindowCarriedArgSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size,
                                     const Fp8OneWindowRuntimeKUnit& unit, uint32_t warps_per_output) {
    if (outer_size == 0 || reduction_size == 0 || !fp32OneWindowRuntimeKUnitPlan(inner_size, unit) ||
        !oneWindowValidWarpsPerOutput(unit.outputs_per_warp, warps_per_output)) {
        return false;
    }
    constexpr uint64_t MaxIndex = std::numeric_limits<uint32_t>::max();
    return outer_size <= MaxIndex && reduction_size <= MaxIndex &&
           outer_size * reduction_size * inner_size <= MaxIndex / 4;
}

void launchFp32OneWindowCarriedArg(const Tensor& input,
                                   const Tensor& carried_index,
                                   Tensor* value_output,
                                   Tensor* index_output,
                                   const CubReductionGeometry& geometry,
                                   const Fp8OneWindowRuntimeKUnit& unit,
                                   uint32_t warps_per_output,
                                   bool maximum,
                                   uint64_t row_stride,
                                   Stream& stream) {
    const uint64_t outputs = geometry.outer_size * geometry.inner_size;
    const uint64_t elements = geometry.outer_size * geometry.reduction_size * geometry.inner_size;
    const auto plan = fp32OneWindowRuntimeKUnitPlan(geometry.inner_size, unit);
    if (input.getDataType() != DataType::FP32 || !plan ||
        !fp32OneWindowCarriedArgSupports(geometry.outer_size, geometry.reduction_size, geometry.inner_size, unit,
                                         warps_per_output) ||
        input.getTotalNumElements() != elements ||
        (carried_index.getDataType() != DataType::UINT32 && carried_index.getDataType() != DataType::UINT64) ||
        carried_index.getTotalNumElements() != elements ||
        (value_output == nullptr && index_output == nullptr) ||
        (value_output != nullptr && (value_output->getTotalNumElements() != outputs ||
                                     !fp8CompleteSupportsOutputStorage(value_output->getDataType()))) ||
        (index_output != nullptr && (index_output->getTotalNumElements() != outputs ||
                                     (index_output->getDataType() != DataType::UINT32 &&
                                      index_output->getDataType() != DataType::UINT64)))) {
        throw std::logic_error("FP32 OneWindow carried ARG received unsupported geometry, inputs or outputs.");
    }
    if ((reinterpret_cast<uintptr_t>(input.getMemPtr<void>()) & uintptr_t{15}) != 0 ||
        (reinterpret_cast<uintptr_t>(carried_index.getMemPtr<void>()) & uintptr_t{15}) != 0) {
        throw std::logic_error("FP32 OneWindow carried ARG requires 16-byte aligned input and carried index bases.");
    }
    // The tails hold the composed index: the index stride is 1.
    const Fp32ArgOutputs arg_outputs{value_output == nullptr ? nullptr : value_output->getMemPtr<void>(),
                                     value_output == nullptr ? DataType::FP32 : value_output->getDataType(),
                                     index_output == nullptr ? nullptr : index_output->getMemPtr<void>(),
                                     index_output == nullptr ? DataType::UINT32 : index_output->getDataType(),
                                     1,
                                     maximum ? 0x80000000U : 0U};
    const uint64_t outputs_per_block = (SHALLOW_FP8_WARPS / warps_per_output) * unit.outputs_per_warp;
    const uint64_t block_count = std::min<uint64_t>(ceilDivideU64(geometry.outer_size, outputs_per_block),
                                                    NARROW_FLAT_MAX_GRID_BLOCKS);
    const bool wide = carried_index.getDataType() == DataType::UINT64;
    const bool launched = dispatchFp32OneWindowRuntimeK(
        unit, [&]<typename InputT, int OutputsPerWarp, int PacketBytes, int MaxComponentsPerLane, bool LoadAhead>() {
            static_assert(std::is_same_v<InputT, float> && PacketBytes == 16);
            const auto launch = [&]<typename IndexT>() {
                fp32OneWindowCarriedArgKernel<OutputsPerWarp, MaxComponentsPerLane, LoadAhead, IndexT>
                    <<<static_cast<unsigned int>(block_count), SHALLOW_FP8_THREADS, 0, stream.getStream()>>>(
                        input.getMemPtr<float>(), carried_index.getMemPtr<IndexT>(), row_stride, arg_outputs,
                        static_cast<uint32_t>(geometry.outer_size), static_cast<uint32_t>(geometry.reduction_size),
                        warps_per_output, plan->geometry);
                CUDA_CHECK(cudaGetLastError());
            };
            if (wide) launch.template operator()<uint64_t>();
            else launch.template operator()<uint32_t>();
        });
    if (!launched) {
        throw std::logic_error("FP32 OneWindow carried ARG has no kernel for this request.");
    }
}

CubKernelOccupancyInfo queryFp32OneWindowCarriedArgOccupancy(uint64_t inner_size,
                                                             const Fp8OneWindowRuntimeKUnit& unit,
                                                             uint32_t warps_per_output,
                                                             DataType index_dtype) {
    if (!fp32OneWindowRuntimeKUnitPlan(inner_size, unit) ||
        !oneWindowValidWarpsPerOutput(unit.outputs_per_warp, warps_per_output) ||
        (index_dtype != DataType::UINT32 && index_dtype != DataType::UINT64)) {
        throw std::logic_error("FP32 OneWindow carried ARG occupancy received unsupported ownership or index.");
    }
    CubKernelOccupancyInfo occupancy{};
    const bool wide = index_dtype == DataType::UINT64;
    const bool queried = dispatchFp32OneWindowRuntimeK(
        unit, [&]<typename InputT, int OutputsPerWarp, int PacketBytes, int MaxComponentsPerLane, bool LoadAhead>() {
            static_assert(std::is_same_v<InputT, float> && PacketBytes == 16);
            occupancy = wide ? queryShallowFp8Occupancy(
                                   fp32OneWindowCarriedArgKernel<OutputsPerWarp, MaxComponentsPerLane, LoadAhead,
                                                                 uint64_t>)
                             : queryShallowFp8Occupancy(
                                   fp32OneWindowCarriedArgKernel<OutputsPerWarp, MaxComponentsPerLane, LoadAhead,
                                                                 uint32_t>);
        });
    if (!queried) {
        throw std::logic_error("FP32 OneWindow carried ARG occupancy has no kernel for this request.");
    }
    return occupancy;
}

bool fp8OneWindowArgSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size,
                             const Fp8OneWindowRuntimeKUnit& unit, uint32_t warps_per_output) {
    if (outer_size == 0 || reduction_size == 0 || reduction_size >= FP8_ARG_MAX_LANE_ROWS ||
        !oneWindowRuntimeKUnitPlan(inner_size, unit) ||
        !oneWindowValidWarpsPerOutput(unit.outputs_per_warp, warps_per_output)) {
        return false;
    }
    constexpr uint64_t MaxIndex = std::numeric_limits<uint32_t>::max();
    return outer_size <= MaxIndex && outer_size * reduction_size * inner_size <= MaxIndex;
}

void launchFp8OneWindowArgComplete(const Tensor& input,
                                   Tensor* value_output,
                                   Tensor* index_output,
                                   const CubReductionGeometry& geometry,
                                   const Fp8OneWindowRuntimeKUnit& unit,
                                   uint32_t warps_per_output,
                                   bool maximum,
                                   uint64_t domain_stride,
                                   Stream& stream) {
#if THOR_CUB_ENABLE_FP8_TYPES
    const DataType input_dtype = input.getDataType();
    const uint64_t outputs = geometry.outer_size * geometry.inner_size;
    const auto plan = oneWindowRuntimeKUnitPlan(geometry.inner_size, unit);
    if ((input_dtype != DataType::FP8_E4M3 && input_dtype != DataType::FP8_E5M2) || !plan ||
        !fp8OneWindowArgSupports(geometry.outer_size, geometry.reduction_size, geometry.inner_size, unit,
                                 warps_per_output) ||
        input.getTotalNumElements() != geometry.outer_size * geometry.reduction_size * geometry.inner_size ||
        (value_output == nullptr && index_output == nullptr) ||
        (value_output != nullptr && (value_output->getTotalNumElements() != outputs ||
                                     !fp8CompleteSupportsOutputStorage(value_output->getDataType()))) ||
        (index_output != nullptr && (index_output->getTotalNumElements() != outputs ||
                                     (index_output->getDataType() != DataType::UINT32 &&
                                      index_output->getDataType() != DataType::UINT64)))) {
        throw std::logic_error("FP8 OneWindow ARG received unsupported complete geometry or outputs.");
    }
    if (index_output != nullptr && index_output->getDataType() == DataType::UINT32 &&
        (geometry.reduction_size - 1) * domain_stride > std::numeric_limits<uint32_t>::max()) {
        throw std::logic_error("FP8 OneWindow ARG UINT32 index output cannot hold the original-domain index.");
    }
    if ((reinterpret_cast<uintptr_t>(input.getMemPtr<void>()) & uintptr_t{15}) != 0) {
        throw std::logic_error("FP8 OneWindow ARG requires a 16-byte aligned input base.");
    }
    const Fp8ArgOutputs arg_outputs{
        value_output == nullptr ? nullptr : value_output->getMemPtr<void>(),
        value_output == nullptr ? DataType::FP32 : value_output->getDataType(),
        index_output == nullptr ? nullptr : index_output->getMemPtr<void>(),
        index_output == nullptr ? DataType::UINT32 : index_output->getDataType(),
        domain_stride,
        fp8ArgFold(maximum, input_dtype == DataType::FP8_E5M2),
        input_dtype == DataType::FP8_E5M2};
    const uint64_t outputs_per_block = (SHALLOW_FP8_WARPS / warps_per_output) * unit.outputs_per_warp;
    const uint64_t block_count = std::min<uint64_t>(ceilDivideU64(geometry.outer_size, outputs_per_block),
                                                    NARROW_FLAT_MAX_GRID_BLOCKS);
    const bool launched = dispatchOneWindowArg(
        unit, [&]<int OutputsPerWarp, int PacketBytes, int MaxComponentsPerLane, bool LoadAhead>() {
            fp8OneWindowArgCompleteKernel<OutputsPerWarp, PacketBytes, MaxComponentsPerLane, LoadAhead>
                <<<static_cast<unsigned int>(block_count), SHALLOW_FP8_THREADS, 0, stream.getStream()>>>(
                    static_cast<const uint8_t*>(input.getMemPtr<void>()),
                    arg_outputs,
                    static_cast<uint32_t>(geometry.outer_size),
                    static_cast<uint32_t>(geometry.reduction_size),
                    warps_per_output,
                    plan->geometry);
            CUDA_CHECK(cudaGetLastError());
        });
    if (!launched) {
        throw std::logic_error("FP8 OneWindow ARG has no kernel for this unit.");
    }
#else
    static_cast<void>(input);
    static_cast<void>(value_output);
    static_cast<void>(index_output);
    static_cast<void>(geometry);
    static_cast<void>(unit);
    static_cast<void>(warps_per_output);
    static_cast<void>(maximum);
    static_cast<void>(domain_stride);
    static_cast<void>(stream);
    throw std::logic_error("FP8 OneWindow ARG requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}


bool fp8DeepRSmallKArgSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size,
                               uint64_t rows_per_shard) {
    return Fp8RK::Fp8DeepRSmallKPolicy::legal({static_cast<uint32_t>(rows_per_shard)}, outer_size, reduction_size,
                                              inner_size) &&
           rows_per_shard < FP8_ARG_MAX_LANE_ROWS;
}

#if THOR_CUB_ENABLE_FP8_TYPES
namespace {
void dispatchFp8DeepRSmallKArgFirstStage(const Tensor& input,
                                         const Fp8ArgOutputs& arg_outputs,
                                         const CubReductionGeometry& geometry,
                                         uint64_t rows_per_shard,
                                         uint64_t shards,
                                         Stream& stream);
}  // namespace
#endif

void launchFp8DeepRSmallKArgFirstStage(const Tensor& input,
                                       Tensor& value_partials,
                                       Tensor& index_partials,
                                       const CubReductionGeometry& geometry,
                                       uint64_t rows_per_shard,
                                       uint64_t shards,
                                       bool maximum,
                                       uint64_t domain_stride,
                                       Stream& stream) {
#if THOR_CUB_ENABLE_FP8_TYPES
    const DataType input_dtype = input.getDataType();
    const uint64_t partials = geometry.outer_size * shards * geometry.inner_size;
    if ((input_dtype != DataType::FP8_E4M3 && input_dtype != DataType::FP8_E5M2) ||
        !fp8DeepRSmallKArgSupports(geometry.outer_size, geometry.reduction_size, geometry.inner_size,
                                   rows_per_shard) ||
        shards != ceilDivideU64(geometry.reduction_size, rows_per_shard) ||
        input.getTotalNumElements() != geometry.outer_size * geometry.reduction_size * geometry.inner_size ||
        value_partials.getDataType() != DataType::FP32 || value_partials.getTotalNumElements() != partials ||
        (index_partials.getDataType() != DataType::UINT32 && index_partials.getDataType() != DataType::UINT64) ||
        index_partials.getTotalNumElements() != partials) {
        throw std::logic_error("FP8 DeepRSmallK ARG first stage received illegal geometry or partials.");
    }
    if (index_partials.getDataType() == DataType::UINT32 &&
        (geometry.reduction_size - 1) * domain_stride > std::numeric_limits<uint32_t>::max()) {
        throw std::logic_error("FP8 DeepRSmallK ARG UINT32 partial index cannot hold the original-domain index.");
    }
    const Fp8ArgOutputs arg_outputs{value_partials.getMemPtr<void>(), DataType::FP32,
                                    index_partials.getMemPtr<void>(), index_partials.getDataType(), domain_stride,
                                    fp8ArgFold(maximum, input_dtype == DataType::FP8_E5M2),
                                    input_dtype == DataType::FP8_E5M2};
    dispatchFp8DeepRSmallKArgFirstStage(input, arg_outputs, geometry, rows_per_shard, shards, stream);
#else
    static_cast<void>(input);
    static_cast<void>(value_partials);
    static_cast<void>(index_partials);
    static_cast<void>(geometry);
    static_cast<void>(rows_per_shard);
    static_cast<void>(shards);
    static_cast<void>(maximum);
    static_cast<void>(domain_stride);
    static_cast<void>(stream);
    throw std::logic_error("FP8 DeepRSmallK ARG requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}

void launchFp8DeepRSmallKArgFirstStagePacked(const Tensor& input,
                                             uint32_t* packed_partials,
                                             const CubReductionGeometry& geometry,
                                             uint64_t rows_per_shard,
                                             uint64_t shards,
                                             bool maximum,
                                             Stream& stream) {
#if THOR_CUB_ENABLE_FP8_TYPES
    const DataType input_dtype = input.getDataType();
    if ((input_dtype != DataType::FP8_E4M3 && input_dtype != DataType::FP8_E5M2) || packed_partials == nullptr ||
        !fp8DeepRSmallKArgSupports(geometry.outer_size, geometry.reduction_size, geometry.inner_size,
                                   rows_per_shard) ||
        shards != ceilDivideU64(geometry.reduction_size, rows_per_shard) ||
        input.getTotalNumElements() != geometry.outer_size * geometry.reduction_size * geometry.inner_size) {
        throw std::logic_error("FP8 DeepRSmallK packed ARG first stage received illegal geometry or partials.");
    }
    Fp8ArgOutputs arg_outputs{nullptr, DataType::FP32, nullptr, DataType::UINT32, 1,
                              fp8ArgFold(maximum, input_dtype == DataType::FP8_E5M2),
                              input_dtype == DataType::FP8_E5M2};
    arg_outputs.packed = packed_partials;
    dispatchFp8DeepRSmallKArgFirstStage(input, arg_outputs, geometry, rows_per_shard, shards, stream);
#else
    static_cast<void>(input);
    static_cast<void>(packed_partials);
    static_cast<void>(geometry);
    static_cast<void>(rows_per_shard);
    static_cast<void>(shards);
    static_cast<void>(maximum);
    static_cast<void>(stream);
    throw std::logic_error("FP8 DeepRSmallK ARG requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}

#if THOR_CUB_ENABLE_FP8_TYPES
namespace {
void dispatchFp8DeepRSmallKArgFirstStage(const Tensor& input,
                                         const Fp8ArgOutputs& arg_outputs,
                                         const CubReductionGeometry& geometry,
                                         uint64_t rows_per_shard,
                                         uint64_t shards,
                                         Stream& stream) {
    const uint32_t depth = narrowFp8PackedCheckedRowsPerShard(rows_per_shard);
    const uint64_t bytes_per_work_item = std::min<uint64_t>(depth, geometry.reduction_size) * geometry.inner_size;
    const dim3 grid_blocks = reductionWorkGrid(shards, 1, geometry.outer_size,
                                               bytes_per_work_item <= 256 ? REDUCTION_PERSISTENT_MAX_BLOCKS : 0);
    const uint32_t rows_per_warp_group = narrowFp8PackedRowsPerWarpGroup<16>(geometry.inner_size, depth);
    const uint32_t depth_log2 = narrowFp8PackedRowsPerShardLog2(depth);
    const auto* bytes = static_cast<const uint8_t*>(input.getMemPtr<void>());
    const uint64_t k = geometry.inner_size;
    const auto packed = [&]<int Components>() {
        fp8DeepRArgFirstStageKernel<Components><<<grid_blocks, NARROW_FLAT_THREADS, 0, stream.getStream()>>>(
            bytes, arg_outputs, geometry.outer_size, geometry.reduction_size, k, shards, depth_log2,
            rows_per_warp_group);
    };
    const auto tiny = [&]<int Lanes>() {
        fp8DeepRArgTinyKFirstStageKernel<Lanes><<<grid_blocks, NARROW_FLAT_THREADS, 0, stream.getStream()>>>(
            bytes, arg_outputs, geometry.outer_size, geometry.reduction_size, k, shards, depth_log2,
            rows_per_warp_group);
    };
    // The VALUE DeepRSmallK ownership by K (dispatchDeepRRetainedRows).
    if (k <= 4) tiny.template operator()<8>();
    else if (k <= 8) tiny.template operator()<4>();
    else if (k <= 32) packed.template operator()<1>();
    else if (k <= 64) packed.template operator()<2>();
    else if (k <= 128) packed.template operator()<4>();
    else if (k <= 160) packed.template operator()<5>();
    else packed.template operator()<8>();
    CUDA_CHECK(cudaGetLastError());
}
}  // namespace
#endif


// ---- 1203: the FP32 RK ARG DeepRSmallK twins (see fp32DeepRArgKernel).
bool fp32DeepRSmallKArgSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size,
                                uint64_t rows_per_shard) {
    return fp32DeepRSmallKSupports(outer_size, reduction_size, inner_size, rows_per_shard);
}

void launchFp32DeepRSmallKArg(const Tensor& input,
                              Tensor* value_output,
                              Tensor* index_output,
                              const CubReductionGeometry& geometry,
                              uint64_t rows_per_shard,
                              uint64_t* packed_scratch,
                              bool maximum,
                              uint64_t domain_stride,
                              Stream& stream) {
    const uint64_t outputs = geometry.outer_size * geometry.inner_size;
    if (input.getDataType() != DataType::FP32 ||
        !fp32DeepRSmallKArgSupports(geometry.outer_size, geometry.reduction_size, geometry.inner_size,
                                    rows_per_shard) ||
        input.getTotalNumElements() != geometry.outer_size * geometry.reduction_size * geometry.inner_size ||
        packed_scratch == nullptr || (value_output == nullptr && index_output == nullptr) ||
        (value_output != nullptr && (value_output->getTotalNumElements() != outputs ||
                                     !fp8CompleteSupportsOutputStorage(value_output->getDataType()))) ||
        (index_output != nullptr && (index_output->getTotalNumElements() != outputs ||
                                     (index_output->getDataType() != DataType::UINT32 &&
                                      index_output->getDataType() != DataType::UINT64)))) {
        throw std::logic_error("FP32 DeepRSmallK ARG received unsupported geometry, outputs or scratch.");
    }
    if ((reinterpret_cast<uintptr_t>(input.getMemPtr<void>()) & uintptr_t{15}) != 0) {
        throw std::logic_error("FP32 DeepRSmallK ARG requires a 16-byte aligned input base.");
    }
    if (index_output != nullptr && index_output->getDataType() == DataType::UINT32 &&
        (geometry.reduction_size - 1) * domain_stride > std::numeric_limits<uint32_t>::max()) {
        throw std::logic_error("FP32 DeepRSmallK ARG UINT32 index cannot hold the original-domain index.");
    }
    const uint32_t depth = narrowFp8PackedCheckedRowsPerShard(rows_per_shard);
    const uint64_t shards = ceilDivideU64(geometry.reduction_size, depth);
    const uint64_t bytes_per_work_item =
        std::min<uint64_t>(depth, geometry.reduction_size) * geometry.inner_size * sizeof(float);
    const dim3 grid_blocks = reductionWorkGrid(shards, 1, geometry.outer_size,
                                               bytes_per_work_item <= 256 ? REDUCTION_PERSISTENT_MAX_BLOCKS : 0);
    const uint32_t rows_per_warp_group = narrowFp8PackedRowsPerWarpGroup<16, 4>(geometry.inner_size, depth);
    const uint32_t depth_log2 = narrowFp8PackedRowsPerShardLog2(depth);
    const uint32_t xor_mask = maximum ? 0x80000000U : 0U;
    const float* values = input.getMemPtr<float>();
    const uint64_t k = geometry.inner_size;
    const auto components = [&]<int Components>() {
        fp32DeepRArgKernel<Components><<<grid_blocks, NARROW_FLAT_THREADS, 0, stream.getStream()>>>(
            values, packed_scratch, geometry.outer_size, geometry.reduction_size, k, shards, depth_log2,
            rows_per_warp_group, xor_mask);
    };
    const auto tiny = [&]<int Lanes>() {
        fp32DeepRArgTinyKKernel<Lanes><<<grid_blocks, NARROW_FLAT_THREADS, 0, stream.getStream()>>>(
            values, packed_scratch, geometry.outer_size, geometry.reduction_size, k, shards, depth_log2,
            rows_per_warp_group, xor_mask);
    };
    // FP32 VALUE DeepRSmallK's ownership by K (dispatchFp32DeepRRetainedRows).
    if (k <= 4) tiny.template operator()<8>();
    else if (k <= 8) tiny.template operator()<4>();
    else if (k <= 32) components.template operator()<1>();
    else if (k <= 64) components.template operator()<2>();
    else components.template operator()<4>();
    CUDA_CHECK(cudaGetLastError());
    launchFp32ArgPackedContinuation(packed_scratch, value_output, index_output, geometry.outer_size,
                                    geometry.reduction_size, geometry.inner_size, shards, depth, maximum,
                                    domain_stride, stream);
}

CubKernelOccupancyInfo queryFp32DeepRSmallKArgOccupancy(uint64_t inner_size) {
    if (inner_size == 0 || inner_size > 125) {
        throw std::logic_error("FP32 DeepRSmallK ARG occupancy requires K <= 125.");
    }
    if (inner_size <= 4) return queryShallowFp8Occupancy(fp32DeepRArgTinyKKernel<8>);
    if (inner_size <= 8) return queryShallowFp8Occupancy(fp32DeepRArgTinyKKernel<4>);
    if (inner_size <= 32) return queryShallowFp8Occupancy(fp32DeepRArgKernel<1>);
    if (inner_size <= 64) return queryShallowFp8Occupancy(fp32DeepRArgKernel<2>);
    return queryShallowFp8Occupancy(fp32DeepRArgKernel<4>);
}

// ---- 1213: the carried-index DeepRSmallK ARG twins (see fp32DeepRCarriedArgKernel).
void launchFp32DeepRSmallKCarriedArg(const Tensor& input,
                                     const Tensor& carried_index,
                                     Tensor* value_output,
                                     Tensor* index_output,
                                     const CubReductionGeometry& geometry,
                                     uint64_t rows_per_shard,
                                     void* wide_scratch,
                                     bool maximum,
                                     uint64_t row_stride,
                                     Stream& stream) {
    const uint64_t elements = geometry.outer_size * geometry.reduction_size * geometry.inner_size;
    if (input.getDataType() != DataType::FP32 ||
        !fp32DeepRSmallKSupports(geometry.outer_size, geometry.reduction_size, geometry.inner_size,
                                 rows_per_shard) ||
        input.getTotalNumElements() != elements ||
        (carried_index.getDataType() != DataType::UINT32 && carried_index.getDataType() != DataType::UINT64) ||
        carried_index.getTotalNumElements() != elements || wide_scratch == nullptr ||
        (reinterpret_cast<uintptr_t>(wide_scratch) & uintptr_t{255}) != 0 ||
        (reinterpret_cast<uintptr_t>(input.getMemPtr<void>()) & uintptr_t{15}) != 0 ||
        (reinterpret_cast<uintptr_t>(carried_index.getMemPtr<void>()) & uintptr_t{15}) != 0) {
        throw std::logic_error("FP32 DeepRSmallK carried ARG received unsupported geometry, inputs or scratch.");
    }
    const uint32_t depth = narrowFp8PackedCheckedRowsPerShard(rows_per_shard);
    const uint64_t shards = ceilDivideU64(geometry.reduction_size, depth);
    const uint64_t bytes_per_work_item =
        std::min<uint64_t>(depth, geometry.reduction_size) * geometry.inner_size * sizeof(float);
    const dim3 grid_blocks = reductionWorkGrid(shards, 1, geometry.outer_size,
                                               bytes_per_work_item <= 256 ? REDUCTION_PERSISTENT_MAX_BLOCKS : 0);
    const uint32_t rows_per_warp_group = narrowFp8PackedRowsPerWarpGroup<16, 4>(geometry.inner_size, depth);
    const uint32_t depth_log2 = narrowFp8PackedRowsPerShardLog2(depth);
    const uint32_t xor_mask = maximum ? 0x80000000U : 0U;
    const uint64_t partial_count = geometry.outer_size * shards * geometry.inner_size;
    auto* bytes = static_cast<uint8_t*>(wide_scratch);
    const Fp32ArgWidePartials partials{
        reinterpret_cast<uint32_t*>(bytes),
        reinterpret_cast<uint64_t*>(bytes + ((partial_count * sizeof(uint32_t) + 255) & ~uint64_t{255}))};
    const float* values = input.getMemPtr<float>();
    const uint64_t k = geometry.inner_size;
    const auto launch = [&]<typename IndexT>() {
        const IndexT* carried = carried_index.getMemPtr<IndexT>();
        const auto components = [&]<int Components>() {
            fp32DeepRCarriedArgKernel<Components, IndexT>
                <<<grid_blocks, NARROW_FLAT_THREADS, 0, stream.getStream()>>>(
                values, carried, partials, geometry.outer_size, geometry.reduction_size, k, shards, depth_log2,
                rows_per_warp_group, xor_mask, row_stride);
        };
        const auto tiny = [&]<int Lanes>() {
            fp32DeepRCarriedArgTinyKKernel<Lanes, IndexT>
                <<<grid_blocks, NARROW_FLAT_THREADS, 0, stream.getStream()>>>(
                values, carried, partials, geometry.outer_size, geometry.reduction_size, k, shards, depth_log2,
                rows_per_warp_group, xor_mask, row_stride);
        };
        // FP32 VALUE DeepRSmallK's ownership by K (dispatchFp32DeepRRetainedRows).
        if (k <= 4) tiny.template operator()<8>();
        else if (k <= 8) tiny.template operator()<4>();
        else if (k <= 32) components.template operator()<1>();
        else if (k <= 64) components.template operator()<2>();
        else components.template operator()<4>();
        CUDA_CHECK(cudaGetLastError());
    };
    if (carried_index.getDataType() == DataType::UINT64) launch.template operator()<uint64_t>();
    else launch.template operator()<uint32_t>();
    launchFp32ArgWideContinuation(wide_scratch, value_output, index_output, geometry.outer_size, shards,
                                  geometry.inner_size, maximum, stream);
}

CubKernelOccupancyInfo queryFp32DeepRSmallKCarriedArgOccupancy(uint64_t inner_size, DataType index_dtype) {
    if (inner_size == 0 || inner_size > 125 || (index_dtype != DataType::UINT32 && index_dtype != DataType::UINT64)) {
        throw std::logic_error(
            "FP32 DeepRSmallK carried ARG occupancy requires K <= 125 and a UINT32 / UINT64 index.");
    }
    const auto query = [&]<typename IndexT>() {
        if (inner_size <= 4) return queryShallowFp8Occupancy(fp32DeepRCarriedArgTinyKKernel<8, IndexT>);
        if (inner_size <= 8) return queryShallowFp8Occupancy(fp32DeepRCarriedArgTinyKKernel<4, IndexT>);
        if (inner_size <= 32) return queryShallowFp8Occupancy(fp32DeepRCarriedArgKernel<1, IndexT>);
        if (inner_size <= 64) return queryShallowFp8Occupancy(fp32DeepRCarriedArgKernel<2, IndexT>);
        return queryShallowFp8Occupancy(fp32DeepRCarriedArgKernel<4, IndexT>);
    };
    return index_dtype == DataType::UINT64 ? query.template operator()<uint64_t>()
                                           : query.template operator()<uint32_t>();
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

void launchFp8RuntimeKArgPass(const Tensor& input,
                              Tensor* value_output,
                              Tensor* index_output,
                              const CubReductionGeometry& geometry,
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
    fp8RotatedArgKernel<8><<<grid, FP8_ARG_ROTATED_THREADS, 0, stream.getStream()>>>(
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


uint32_t fp8PacketCompleteArgTileComponents(uint64_t inner_size, uint32_t value_tile_components) {
    // 1127: a lane keys four adjacent components per word, so a row of a tile costs ceil(tile / 128) key words and a
    // row of the whole K costs sum over tiles. A 256-component tile reaches the minimum ceil(K / 128); VALUE's tile
    // (capped at 256) is kept when it is as cheap (it may carry the planner's parallelism choice for small outer).
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

void launchFp8PacketCompleteArg(const Tensor& input,
                                Tensor* value_output,
                                Tensor* index_output,
                                const CubReductionGeometry& geometry,
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
    if ((reinterpret_cast<uintptr_t>(input.getMemPtr<void>()) & uintptr_t{15}) != 0) {
        throw std::logic_error("FP8 PacketComplete ARG requires a 16-byte aligned input base.");
    }
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
    const uint32_t outputs_per_block = SHALLOW_FP8_WARPS / warps_per_output;
    const uint64_t work_count = geometry.outer_size * static_cast<uint64_t>(tile_count);
    const uint64_t block_count =
        std::min<uint64_t>(ceilDivideU64(work_count, static_cast<uint64_t>(outputs_per_block)),
                           NARROW_FLAT_MAX_GRID_BLOCKS);
    fp8PacketCompleteArgKernel<16><<<static_cast<unsigned int>(block_count), SHALLOW_FP8_THREADS, 0,
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

bool fp8AlignedKArgSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size,
                            uint32_t packet_bytes, uint32_t cta_threads, uint64_t shards) {
    if (outer_size == 0 || reduction_size == 0 || inner_size == 0 || shards == 0 || shards > reduction_size ||
        (packet_bytes != 2 && packet_bytes != 8) || inner_size % packet_bytes != 0 ||
        (cta_threads != 32 && cta_threads != 64 && cta_threads != 128)) {
        return false;
    }
    // Rows of one shard carry 23-bit candidate rows.
    if (ceilDivideU64(reduction_size, shards) >= FP8_ARG_MAX_LANE_ROWS) return false;
    const uint64_t tiles = ceilDivideU64(inner_size, static_cast<uint64_t>(cta_threads) * packet_bytes);
    constexpr uint64_t MaxExtent = (uint64_t{1} << 31) - 1;  // reductionWorkGrid
    return shards <= MaxExtent && tiles <= MaxExtent && outer_size <= MaxExtent;
}

void launchFp8AlignedKArgPass(const Tensor& input,
                              Tensor* value_output,
                              Tensor* index_output,
                              const CubReductionGeometry& geometry,
                              uint32_t packet_bytes,
                              uint32_t cta_threads,
                              uint64_t shards,
                              bool maximum,
                              uint64_t domain_stride,
                              Stream& stream,
                              uint32_t* packed_partials) {
#if THOR_CUB_ENABLE_FP8_TYPES
    const DataType input_dtype = input.getDataType();
    const uint64_t outputs = geometry.outer_size * shards * geometry.inner_size;
    const bool packed = packed_partials != nullptr;
    if ((input_dtype != DataType::FP8_E4M3 && input_dtype != DataType::FP8_E5M2) ||
        !fp8AlignedKArgSupports(geometry.outer_size, geometry.reduction_size, geometry.inner_size, packet_bytes,
                                cta_threads, shards) ||
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
        throw std::logic_error("FP8 AlignedK ARG received unsupported geometry or outputs.");
    }
    if ((reinterpret_cast<uintptr_t>(input.getMemPtr<void>()) & uintptr_t{packet_bytes - 1}) != 0) {
        throw std::logic_error("FP8 AlignedK ARG requires a packet-aligned input base.");
    }
    if (index_output != nullptr && index_output->getDataType() == DataType::UINT32 &&
        (geometry.reduction_size - 1) * domain_stride > std::numeric_limits<uint32_t>::max()) {
        throw std::logic_error("FP8 AlignedK ARG UINT32 index cannot hold the original-domain index.");
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
    const uint64_t tiles = ceilDivideU64(geometry.inner_size, static_cast<uint64_t>(cta_threads) * packet_bytes);
    const ReductionShardRows rows = balancedReductionShardRows(geometry.reduction_size, shards);
    // VALUE's persistent-grid rule for tiny Complete work items (REDUCTION_PERSISTENT_MAX_BLOCKS).
    const uint64_t item_rows = static_cast<uint64_t>(rows.base_rows) + static_cast<uint64_t>(rows.extra_rows != 0);
    const uint64_t item_bytes =
        item_rows * std::min<uint64_t>(geometry.inner_size, static_cast<uint64_t>(cta_threads) * packet_bytes);
    const bool persistent = shards == 1 && item_rows <= 8 && item_bytes <= 512;
    const dim3 grid = reductionWorkGrid(shards, tiles, geometry.outer_size,
                                        persistent ? REDUCTION_PERSISTENT_MAX_BLOCKS : 0);
    const auto* bytes = static_cast<const uint8_t*>(input.getMemPtr<void>());
    if (packet_bytes == 8) {
        fp8AlignedKArgKernel<8><<<grid, cta_threads, 0, stream.getStream()>>>(
            bytes, arg_outputs, geometry.outer_size, geometry.reduction_size, geometry.inner_size, shards, tiles, rows);
    } else {
        fp8AlignedKArgKernel<2><<<grid, cta_threads, 0, stream.getStream()>>>(
            bytes, arg_outputs, geometry.outer_size, geometry.reduction_size, geometry.inner_size, shards, tiles, rows);
    }
    CUDA_CHECK(cudaGetLastError());
#else
    static_cast<void>(input);
    static_cast<void>(value_output);
    static_cast<void>(index_output);
    static_cast<void>(geometry);
    static_cast<void>(packet_bytes);
    static_cast<void>(cta_threads);
    static_cast<void>(shards);
    static_cast<void>(maximum);
    static_cast<void>(domain_stride);
    static_cast<void>(stream);
    static_cast<void>(packed_partials);
    throw std::logic_error("FP8 AlignedK ARG requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}


bool fp8ArgPackedContinuationSplitsShards(uint64_t outer_size, uint64_t inner_size, uint64_t shards) {
    if (shards < FP8_ARG_PACKED_SPLIT_MIN_SHARDS || inner_size == 0 ||
        outer_size > std::numeric_limits<uint64_t>::max() / inner_size) {
        return false;
    }
    const uint64_t outputs = outer_size * inner_size;
    if (outputs >= FP8_ARG_PACKED_SPLIT_MAX_OUTPUTS &&
        (shards < FP8_ARG_PACKED_SPLIT_WIDE_MIN_SHARDS || outputs >= FP8_ARG_PACKED_SPLIT_WIDE_MAX_OUTPUTS)) {
        return false;
    }
    const Fp8ArgPackedSplitLayout layout = fp8ArgPackedSplitLayout(inner_size, shards);
    return uint64_t{layout.columns} * layout.groups >= 32;
}

Fp8ArgPackedSplitLayout fp8ArgPackedSplitLayout(uint64_t inner_size, uint64_t shards) {
    Fp8ArgPackedSplitLayout layout;
    layout.columns = inner_size < 32 ? static_cast<uint32_t>(inner_size) : 32U;
    // At least four shards per group (one batch of loads in flight), at most 1024 threads.
    const uint64_t by_shards = std::max<uint64_t>(1, ceilDivideU64(shards, 4));
    layout.groups = static_cast<uint32_t>(std::min<uint64_t>(1024U / layout.columns, by_shards));
    layout.threads = static_cast<uint32_t>(ceilDivideU64(uint64_t{layout.columns} * layout.groups, 32) * 32);
    return layout;
}

void launchFp8ArgPackedContinuation(const uint32_t* packed_partials,
                                    Tensor* value_output,
                                    Tensor* index_output,
                                    DataType input_dtype,
                                    uint64_t outer_size,
                                    uint64_t reduction_size,
                                    uint64_t inner_size,
                                    uint64_t shards,
                                    uint64_t uniform_rows_per_shard,
                                    bool maximum,
                                    uint64_t domain_stride,
                                    Stream& stream) {
#if THOR_CUB_ENABLE_FP8_TYPES
    const uint64_t outputs = outer_size * inner_size;
    if ((input_dtype != DataType::FP8_E4M3 && input_dtype != DataType::FP8_E5M2) || packed_partials == nullptr ||
        // 1132: 55-bit widened indices (formerly R <= 2^32 - 1); 32-bit shard loop.
        outputs == 0 || shards < 2 || shards > reduction_size || shards > REDUCTION_WORK_GRID_MAX_EXTENT ||
        reduction_size - 1 > FP8_ARG_MAX_WIDE_INDEX ||
        (uniform_rows_per_shard != 0 && shards != ceilDivideU64(reduction_size, uniform_rows_per_shard)) ||
        (value_output == nullptr && index_output == nullptr) ||
        (value_output != nullptr && (value_output->getTotalNumElements() != outputs ||
                                     !fp8CompleteSupportsOutputStorage(value_output->getDataType()))) ||
        (index_output != nullptr && (index_output->getTotalNumElements() != outputs ||
                                     (index_output->getDataType() != DataType::UINT32 &&
                                      index_output->getDataType() != DataType::UINT64)))) {
        throw std::logic_error("FP8 packed ARG continuation received unsupported geometry or outputs.");
    }
    if (index_output != nullptr && index_output->getDataType() == DataType::UINT32 &&
        (reduction_size - 1) * domain_stride > std::numeric_limits<uint32_t>::max()) {
        throw std::logic_error("FP8 packed ARG continuation UINT32 index cannot hold the original-domain index.");
    }
    const Fp8ArgOutputs arg_outputs{
        value_output == nullptr ? nullptr : value_output->getMemPtr<void>(),
        value_output == nullptr ? DataType::FP32 : value_output->getDataType(),
        index_output == nullptr ? nullptr : index_output->getMemPtr<void>(),
        index_output == nullptr ? DataType::UINT32 : index_output->getDataType(),
        domain_stride,
        fp8ArgFold(maximum, input_dtype == DataType::FP8_E5M2),
        input_dtype == DataType::FP8_E5M2};
    // Uniform depths (DeepRSmallK, rotated RuntimeK): shard s begins at s * depth. Balanced (AlignedK): VALUE's split.
    const ReductionShardRows rows = uniform_rows_per_shard != 0
                                        ? ReductionShardRows{static_cast<uint32_t>(uniform_rows_per_shard), 0U}
                                        : balancedReductionShardRows(reduction_size, shards);
    if (fp8ArgPackedContinuationSplitsShards(outer_size, inner_size, shards)) {
        const Fp8ArgPackedSplitLayout layout = fp8ArgPackedSplitLayout(inner_size, shards);
        const uint64_t items = outer_size * ceilDivideU64(inner_size, layout.columns);
        const uint64_t blocks = std::min<uint64_t>(items, NARROW_FLAT_MAX_GRID_BLOCKS);
        fp8ArgPackedContinuationSplitKernel<0><<<static_cast<unsigned int>(blocks), layout.threads, 0,
                                                 stream.getStream()>>>(packed_partials, arg_outputs, outer_size,
                                                                       shards, inner_size, rows, layout.columns,
                                                                       layout.groups);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    constexpr uint32_t Threads = 256;
    const uint64_t blocks = std::min<uint64_t>(ceilDivideU64(outputs, Threads), NARROW_FLAT_MAX_GRID_BLOCKS);
    fp8ArgPackedContinuationKernel<0><<<static_cast<unsigned int>(blocks), Threads, 0, stream.getStream()>>>(
        packed_partials, arg_outputs, outer_size, shards, inner_size, rows);
    CUDA_CHECK(cudaGetLastError());
#else
    static_cast<void>(packed_partials);
    static_cast<void>(value_output);
    static_cast<void>(index_output);
    static_cast<void>(input_dtype);
    static_cast<void>(outer_size);
    static_cast<void>(reduction_size);
    static_cast<void>(inner_size);
    static_cast<void>(shards);
    static_cast<void>(uniform_rows_per_shard);
    static_cast<void>(maximum);
    static_cast<void>(domain_stride);
    static_cast<void>(stream);
    throw std::logic_error("FP8 packed ARG continuation requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}

// ---------------------------------------------------------------------------------------------------------------------
// 1146-1148: retained FP8 R / KR reductions (Fp8KR/Fp8KRSegments.h): Pack (one thread, several short rows), Group (G lanes
// per row) and Split (one CTA per row chunk writing a partial, then a finishing kernel, one CTA per row). Every
// family reads 4-byte words on the tensor's word grid (FP8_KR_UNROLL loads in flight) and masks bytes outside its work.
// VALUE compiles only the reachable semantics: SUM x {identity, |x|, x^2}, PRODUCT, and MIN / MAX through RK's shared
// MIN kernel with the P5-F byte fold (MAX negates through output_scale; |x| / x^2 fold into the bytes).
#if THOR_CUB_ENABLE_FP8_TYPES
namespace {

constexpr uint32_t FP8_KR_UNROLL = 4;
constexpr uint32_t FP8_KR_WARP_LANES = 32;
constexpr uint32_t FP8_KR_WARPS = Fp8KR::FP8_KR_THREADS / FP8_KR_WARP_LANES;
constexpr uint64_t FP8_KR_MAX_GRID_BLOCKS = 2147483647ULL;
static_assert(Fp8KR::FP8_KR_THREADS == 256 && FP8_KR_WARPS == 8);

// Byte offsets on the word grid (the input's misalignment `head` included).
struct Fp8KRSegmentBounds {
    uint64_t begin;
    uint64_t end;
    uint64_t row_begin;
};

__device__ __forceinline__ Fp8KRSegmentBounds fp8KRSegmentBounds(uint64_t segment,
                                                                 uint64_t chunks,
                                                                 uint64_t row_bytes,
                                                                 uint64_t chunk_bytes,
                                                                 uint32_t head) {
    const uint64_t row = chunks == 1 ? segment : segment / chunks;
    const uint64_t offset = (segment - row * chunks) * chunk_bytes;
    const uint64_t remaining = row_bytes - offset;
    const uint64_t length = remaining < chunk_bytes ? remaining : chunk_bytes;
    const uint64_t row_begin = static_cast<uint64_t>(head) + row * row_bytes;
    return {row_begin + offset, row_begin + offset + length, row_begin};
}

__device__ __forceinline__ bool fp8KRByteInside(uint64_t byte, const Fp8KRSegmentBounds& bounds) {
    return byte >= bounds.begin && byte < bounds.end;
}

// One folded, decoded and transformed word.
template <typename InputT, int FixedOp, int FixedInputTransform>
__device__ __forceinline__ float4 fp8KRDecodeWord(uint32_t packed, const Fp8ExtremumFold& fold) {
    return transformFp8CompleteDecodedPacket<FixedInputTransform>(
        decodeFp8x4ToFp32<InputT>(foldFp8ExtremumBytes<FixedOp == 2>(packed, fold)),
        CubReductionStageInputTransform::Identity);
}

// 1179: FP32 input (Fp32 KR, sharing the Group / Pack / Block / Split kernels): one element per word. MIN / MAX fold
// the sign and |x| into the word as FP8 folds its bytes (fold masks 0x7fffffff / 0x80000000, fp32CompleteExtremumFold);
// x^2 for MIN / MAX is |x| squared at the end, exact because rounding is monotone in |x|.
template <int FixedOp, int FixedInputTransform>
__device__ __forceinline__ float fp32KRDecodeWord(uint32_t word, const Fp8ExtremumFold& fold) {
    float value = __uint_as_float(foldFp8ExtremumBytes<FixedOp == 2>(word, fold));
    if constexpr (FixedInputTransform == 1) {
        value = ::fabsf(value);
    } else if constexpr (FixedInputTransform == 2) {
        value *= value;
    }
    return value;
}

// Retained KR input element types: FP8 (four per word) and, 1179, FP32 (one per word).
template <typename InputT>
inline constexpr bool IS_RETAINED_KR_INPUT = sizeof(InputT) == 1 || std::is_same_v<InputT, float>;

template <int FixedOp>
__device__ __forceinline__ void fp8KRStoreValue(void* output, DataType output_dtype, uint64_t index, float value,
                                                const Fp8ExtremumFold& fold, AdditiveFinalizeFp32 output_finalize,
                                                float output_scale) {
    storeFp8CompleteResult<FixedOp == 0>(output, output_dtype, index,
                                         unfoldFp8ExtremumResult<FixedOp == 2>(value, fold), output_finalize,
                                         output_scale);
}

// Accumulates the words [first_word, word_end) a lane visits with stride `step`, masking bytes outside `bounds`.
// 1179: FP32 rows are whole words on the word grid (the launcher requires a 4-byte aligned input, so head is 0), so
// every visited word is inside and the FP32 branch needs no mask; consecutive loads alternate the two accumulators.
template <typename InputT, int FixedOp, int FixedInputTransform>
__device__ __forceinline__ void fp8KRAccumulateWords(const uint32_t* __restrict__ words, uint64_t first_word,
                                                     uint64_t word_end, uint64_t step, const Fp8KRSegmentBounds& bounds,
                                                     const Fp8ExtremumFold& fold, float& even, float& odd) {
    const Fp8TypedReductionOp<FixedOp> combine{};
    if constexpr (std::is_same_v<InputT, float>) {
        static_cast<void>(bounds);
        for (uint64_t word = first_word; word < word_end; word += static_cast<uint64_t>(FP8_KR_UNROLL) * step) {
            uint32_t packed[FP8_KR_UNROLL];
#pragma unroll
            for (uint32_t u = 0; u < FP8_KR_UNROLL; ++u) {
                const uint64_t w = word + static_cast<uint64_t>(u) * step;
                packed[u] = w < word_end ? __ldg(words + w) : 0U;
            }
#pragma unroll
            for (uint32_t u = 0; u < FP8_KR_UNROLL; ++u) {
                const uint64_t w = word + static_cast<uint64_t>(u) * step;
                if (w < word_end) {
                    const float v = fp32KRDecodeWord<FixedOp, FixedInputTransform>(packed[u], fold);
                    if ((u & 1U) == 0) {
                        even = combine(even, v);
                    } else {
                        odd = combine(odd, v);
                    }
                }
            }
        }
    } else {
        for (uint64_t word = first_word; word < word_end; word += static_cast<uint64_t>(FP8_KR_UNROLL) * step) {
            uint32_t packed[FP8_KR_UNROLL];
#pragma unroll
            for (uint32_t u = 0; u < FP8_KR_UNROLL; ++u) {
                const uint64_t w = word + static_cast<uint64_t>(u) * step;
                packed[u] = w < word_end ? __ldg(words + w) : 0U;
            }
#pragma unroll
            for (uint32_t u = 0; u < FP8_KR_UNROLL; ++u) {
                const uint64_t w = word + static_cast<uint64_t>(u) * step;
                if (w < word_end) {
                    const float4 v = fp8KRDecodeWord<InputT, FixedOp, FixedInputTransform>(packed[u], fold);
                    const uint64_t first = w << 2;
                    if (first >= bounds.begin && first + 4 <= bounds.end) {
                        even = combine(even, v.x);
                        odd = combine(odd, v.y);
                        even = combine(even, v.z);
                        odd = combine(odd, v.w);
                    } else {
                        if (fp8KRByteInside(first, bounds)) even = combine(even, v.x);
                        if (fp8KRByteInside(first + 1, bounds)) odd = combine(odd, v.y);
                        if (fp8KRByteInside(first + 2, bounds)) even = combine(even, v.z);
                        if (fp8KRByteInside(first + 3, bounds)) odd = combine(odd, v.w);
                    }
                }
            }
        }
    }
}

// ARG counterpart: the four bytes of a word are four consecutive indices of one row, so a word's best is
// fp8ArgRowsCandidate4 with row stride 1. Bytes outside the bounds get key 0xFF, which no FP8 byte has (Fp8ArgKey.h)
// and so never wins; every visited word holds at least one byte inside. The index is row-local.
__device__ __forceinline__ void fp8KRArgAccumulateWords(const uint32_t* __restrict__ words, uint64_t first_word,
                                                        uint64_t word_end, uint64_t step,
                                                        const Fp8KRSegmentBounds& bounds, Fp8ArgFold fold,
                                                        uint64_t& best) {
    for (uint64_t word = first_word; word < word_end; word += static_cast<uint64_t>(FP8_KR_UNROLL) * step) {
        uint32_t packed[FP8_KR_UNROLL];
#pragma unroll
        for (uint32_t u = 0; u < FP8_KR_UNROLL; ++u) {
            const uint64_t w = word + static_cast<uint64_t>(u) * step;
            packed[u] = w < word_end ? __ldg(words + w) : 0U;
        }
#pragma unroll
        for (uint32_t u = 0; u < FP8_KR_UNROLL; ++u) {
            const uint64_t w = word + static_cast<uint64_t>(u) * step;
            if (w < word_end) {
                Fp8ArgKeys4 keys = fp8ArgKeys4(packed[u], fold);
                const uint64_t first = w << 2;
                if (first < bounds.begin || first + 4 > bounds.end) {
                    uint32_t outside = 0;
#pragma unroll
                    for (uint32_t j = 0; j < 4; ++j) {
                        if (!fp8KRByteInside(first + j, bounds)) outside |= 0xFFU << (8U * j);
                    }
                    keys.keys |= outside;
                }
                const uint32_t word_best = fp8ArgRowsCandidate4(keys, 0U, 1U);
                const uint64_t index = first + ((word_best >> 1) & 3U) - bounds.row_begin;
                const uint64_t wide = (static_cast<uint64_t>(word_best >> 24) << (FP8_ARG_WIDE_INDEX_BITS + 1)) |
                                      (index << 1) | static_cast<uint64_t>(word_best & 1U);
                best = fp8ArgMin(best, wide);
            }
        }
    }
}

// ---- Group: G lanes per row. The loop bound is block-uniform, so every lane reaches the group shuffles.
template <typename InputT, int FixedOp, int FixedInputTransform>
__global__ __launch_bounds__(Fp8KR::FP8_KR_THREADS) void fp8KRValueGroupKernel(const uint32_t* __restrict__ words,
                                                                                uint32_t head,
                                                                                uint64_t rows,
                                                                                uint64_t row_bytes,
                                                                                uint32_t group_shift,
                                                                                Fp8ExtremumFold fold,
                                                                                void* output,
                                                                                DataType output_dtype,
                                                                                AdditiveFinalizeFp32 output_finalize,
                                                                                float output_scale) {
    static_assert(IS_RETAINED_KR_INPUT<InputT>);
    const Fp8TypedReductionOp<FixedOp> combine{};
    const uint32_t group_lanes = 1U << group_shift;
    const uint32_t lane = threadIdx.x & (group_lanes - 1U);
    const uint64_t group = threadIdx.x >> group_shift;
    const uint64_t groups_per_block = Fp8KR::FP8_KR_THREADS >> group_shift;
    const uint64_t block_stride = static_cast<uint64_t>(gridDim.x) * groups_per_block;
    for (uint64_t base = static_cast<uint64_t>(blockIdx.x) * groups_per_block; base < rows; base += block_stride) {
        const uint64_t row = base + group;
        float even = combine.identity();
        float odd = combine.identity();
        if (row < rows) {
            const Fp8KRSegmentBounds bounds = fp8KRSegmentBounds(row, 1, row_bytes, row_bytes, head);
            fp8KRAccumulateWords<InputT, FixedOp, FixedInputTransform>(words, (bounds.begin >> 2) + lane,
                                                                       (bounds.end + 3) >> 2, group_lanes, bounds,
                                                                       fold, even, odd);
        }
        float total = combine(even, odd);
        for (uint32_t offset = group_lanes >> 1; offset != 0; offset >>= 1) {
            total = combine(total, __shfl_xor_sync(0xffffffffU, total, static_cast<int>(offset)));
        }
        if (lane == 0 && row < rows) {
            fp8KRStoreValue<FixedOp>(output, output_dtype, row, total, fold, output_finalize, output_scale);
        }
    }
}

// ---- Pack family output staging (1150). The 1149 SASS showed every scalar result store expanding into the runtime
// output-dtype switch (FP8 / FP16 / BF16 / FP32 / FP64 stores and their branches) at every unrolled store site: the ARG
// Pack kernel had 416 STG and 1023 branches in 6612 instructions, a 16-row VALUE vector kernel 172 STG and 390 branches.
// Pack kernels now stage each block's results in shared memory (one word per row; pad one word per 32 so the
// row-strided per-thread writes do not bank-conflict) and write them from ONE store site in a block-strided loop, which
// also makes consecutive threads store consecutive rows.
constexpr uint32_t FP8_KR_PACK_MAX_ROWS = 16;  // Fp8KRSegments.h legality: rows_per_thread <= 16
constexpr uint32_t FP8_KR_PACK_STAGE_WORDS =
    Fp8KR::FP8_KR_THREADS * FP8_KR_PACK_MAX_ROWS + Fp8KR::FP8_KR_THREADS * FP8_KR_PACK_MAX_ROWS / 32;

__device__ __forceinline__ uint32_t fp8KRStageSlot(uint32_t index) { return index + (index >> 5); }

// Finalize (SUM family), unfold (MIN / MAX fold) and scale one result before its storage conversion.
template <int FixedOp>
__device__ __forceinline__ float fp8KRFinalValue(float value, const Fp8ExtremumFold& fold,
                                                 AdditiveFinalizeFp32 output_finalize, float output_scale) {
    value = unfoldFp8ExtremumResult<FixedOp == 2>(value, fold);
    if constexpr (FixedOp == 0) value = finalizeFp32Epilogue(output_finalize, value);
    return value * output_scale;
}

template <int FixedOp>
__device__ __forceinline__ void fp8KRWriteStagedValues(const float* staged, uint64_t first_row, uint32_t count,
                                                       const Fp8ExtremumFold& fold, void* output,
                                                       DataType output_dtype, AdditiveFinalizeFp32 output_finalize,
                                                       float output_scale) {
#pragma unroll 1
    for (uint32_t i = threadIdx.x; i < count; i += Fp8KR::FP8_KR_THREADS) {
        storeFp32AsRuntimeDType(output, output_dtype, first_row + i,
                                fp8KRFinalValue<FixedOp>(staged[fp8KRStageSlot(i)], fold, output_finalize,
                                                         output_scale));
    }
}

__device__ __forceinline__ void fp8KRWriteStagedArgs(const uint32_t* staged, uint64_t first_row, uint32_t count,
                                                     const Fp8ArgOutputs& outputs) {
#pragma unroll 1
    for (uint32_t i = threadIdx.x; i < count; i += Fp8KR::FP8_KR_THREADS) {
        storeFp8ArgPackedResult(outputs, first_row + i, fp8ArgWiden(staged[fp8KRStageSlot(i)], 0));
    }
}

// ---- Pack: one thread owns rows_per_thread (<= 16) consecutive short rows (row_bytes <= 32) and walks their bytes in
// order; each completed row goes to the block's staging buffer. The block loop is uniform, so every thread reaches the
// barriers.
template <typename InputT, int FixedOp, int FixedInputTransform>
__global__ __launch_bounds__(Fp8KR::FP8_KR_THREADS) void fp8KRValuePackKernel(const uint32_t* __restrict__ words,
                                                                               uint32_t head,
                                                                               uint64_t rows,
                                                                               uint32_t row_bytes,
                                                                               uint32_t rows_per_thread,
                                                                               Fp8ExtremumFold fold,
                                                                               void* output,
                                                                               DataType output_dtype,
                                                                               AdditiveFinalizeFp32 output_finalize,
                                                                               float output_scale) {
    static_assert(IS_RETAINED_KR_INPUT<InputT>);
    __shared__ float staged[FP8_KR_PACK_STAGE_WORDS];
    const Fp8TypedReductionOp<FixedOp> combine{};
    const float identity = combine.identity();
    const uint64_t owners = (rows + rows_per_thread - 1) / rows_per_thread;
    const uint64_t block_stride = static_cast<uint64_t>(gridDim.x) * Fp8KR::FP8_KR_THREADS;
    for (uint64_t block_owner = static_cast<uint64_t>(blockIdx.x) * Fp8KR::FP8_KR_THREADS; block_owner < owners;
         block_owner += block_stride) {
        const uint64_t block_first_row = block_owner * rows_per_thread;
        const uint64_t owner = block_owner + threadIdx.x;
        if (owner < owners) {
            uint64_t row = owner * rows_per_thread;
            uint32_t local = threadIdx.x * rows_per_thread;  // row - block_first_row
            const uint64_t count = rows - row < rows_per_thread ? rows - row : rows_per_thread;
            const uint64_t begin = static_cast<uint64_t>(head) + row * row_bytes;
            const uint64_t end = begin + count * row_bytes;
            const uint64_t word_end = (end + 3) >> 2;
            float acc = identity;
            uint32_t position = 0;
            for (uint64_t word = begin >> 2; word < word_end; word += FP8_KR_UNROLL) {
                uint32_t packed[FP8_KR_UNROLL];
#pragma unroll
                for (uint32_t u = 0; u < FP8_KR_UNROLL; ++u) {
                    packed[u] = word + u < word_end ? __ldg(words + word + u) : 0U;
                }
#pragma unroll
                for (uint32_t u = 0; u < FP8_KR_UNROLL; ++u) {
                    if constexpr (std::is_same_v<InputT, float>) {
                        // 1179: one element per word, every word of [begin, end) inside (head is 0); row_bytes / 4
                        // elements per row.
                        if (word + u < word_end) {
                            acc = combine(acc, fp32KRDecodeWord<FixedOp, FixedInputTransform>(packed[u], fold));
                            if (++position == (row_bytes >> 2)) {
                                staged[fp8KRStageSlot(local)] = acc;
                                acc = identity;
                                position = 0;
                                ++local;
                            }
                        }
                    } else if (word + u < word_end) {
                        const float4 v = fp8KRDecodeWord<InputT, FixedOp, FixedInputTransform>(packed[u], fold);
                        const float values[4] = {v.x, v.y, v.z, v.w};
                        const uint64_t first = (word + u) << 2;
#pragma unroll
                        for (uint32_t j = 0; j < 4; ++j) {
                            if (first + j >= begin && first + j < end) {
                                acc = combine(acc, values[j]);
                                if (++position == row_bytes) {
                                    staged[fp8KRStageSlot(local)] = acc;
                                    acc = identity;
                                    position = 0;
                                    ++local;
                                }
                            }
                        }
                    }
                }
            }
        }
        __syncthreads();
        const uint64_t block_rows = rows - block_first_row;
        const uint32_t count = static_cast<uint32_t>(
            block_rows < uint64_t{Fp8KR::FP8_KR_THREADS} * rows_per_thread ? block_rows
                                                                            : uint64_t{Fp8KR::FP8_KR_THREADS} *
                                                                                  rows_per_thread);
        fp8KRWriteStagedValues<FixedOp>(staged, block_first_row, count, fold, output, output_dtype, output_finalize,
                                        output_scale);
        __syncthreads();
    }
}

// ---- Vector Pack (1148): one thread owns fp8KRVectorRows(RowBytes) rows, a whole number of 16-byte vectors, loaded
// with 16-byte loads (the input is 16-byte aligned; the launcher otherwise runs the generic Pack kernel). Row boundaries
// are compile-time, so the accumulation is straight-line; results leave through the block's staging buffer (1150). The
// last rows % per-thread rows run byte by byte in the final block iteration.
template <int RowBytes>
inline constexpr int FP8_KR_VECTOR_ROWS = static_cast<int>(Fp8KR::fp8KRVectorRows(RowBytes));

template <typename InputT, int FixedOp, int FixedInputTransform, int RowBytes>
__global__ __launch_bounds__(Fp8KR::FP8_KR_THREADS) void fp8KRValueVectorKernel(const uint4* __restrict__ vectors,
                                                                                 uint64_t rows,
                                                                                 Fp8ExtremumFold fold,
                                                                                 void* output,
                                                                                 DataType output_dtype,
                                                                                 AdditiveFinalizeFp32 output_finalize,
                                                                                 float output_scale) {
    static_assert(sizeof(InputT) == 1);
    constexpr int Rows = FP8_KR_VECTOR_ROWS<RowBytes>;
    constexpr int Vectors = Rows * RowBytes / 16;
    constexpr int Words = 4 * Vectors;
    static_assert(Rows * RowBytes == 16 * Vectors && Rows <= static_cast<int>(FP8_KR_PACK_MAX_ROWS));
    __shared__ float staged[FP8_KR_PACK_STAGE_WORDS];
    const Fp8TypedReductionOp<FixedOp> combine{};
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(vectors);
    const uint64_t owners = (rows + Rows - 1) / Rows;  // the last owner may be partial (byte path)
    const uint64_t full_owners = rows / Rows;
    const uint64_t block_stride = static_cast<uint64_t>(gridDim.x) * Fp8KR::FP8_KR_THREADS;
    for (uint64_t block_owner = static_cast<uint64_t>(blockIdx.x) * Fp8KR::FP8_KR_THREADS; block_owner < owners;
         block_owner += block_stride) {
        const uint64_t block_first_row = block_owner * Rows;
        const uint64_t owner = block_owner + threadIdx.x;
        if (owner < full_owners) {
            uint32_t words[Words];
#pragma unroll
            for (int v = 0; v < Vectors; ++v) {
                const uint4 packet = __ldg(vectors + owner * Vectors + v);
                words[4 * v] = packet.x;
                words[4 * v + 1] = packet.y;
                words[4 * v + 2] = packet.z;
                words[4 * v + 3] = packet.w;
            }
            float acc[Rows];
#pragma unroll
            for (int r = 0; r < Rows; ++r) acc[r] = combine.identity();
#pragma unroll
            for (int w = 0; w < Words; ++w) {
                const float4 v = fp8KRDecodeWord<InputT, FixedOp, FixedInputTransform>(words[w], fold);
                acc[(4 * w) / RowBytes] = combine(acc[(4 * w) / RowBytes], v.x);
                acc[(4 * w + 1) / RowBytes] = combine(acc[(4 * w + 1) / RowBytes], v.y);
                acc[(4 * w + 2) / RowBytes] = combine(acc[(4 * w + 2) / RowBytes], v.z);
                acc[(4 * w + 3) / RowBytes] = combine(acc[(4 * w + 3) / RowBytes], v.w);
            }
#pragma unroll
            for (int r = 0; r < Rows; ++r) staged[fp8KRStageSlot(threadIdx.x * Rows + r)] = acc[r];
        } else if (owner < owners) {
            // The partial last owner: its rows byte by byte.
            for (uint64_t row = owner * Rows; row < rows; ++row) {
                float acc = combine.identity();
                for (int i = 0; i < RowBytes; ++i) {
                    const uint32_t byte = __ldg(bytes + row * RowBytes + i);
                    acc = combine(acc, fp8KRDecodeWord<InputT, FixedOp, FixedInputTransform>(byte, fold).x);
                }
                staged[fp8KRStageSlot(static_cast<uint32_t>(row - block_first_row))] = acc;
            }
        }
        __syncthreads();
        const uint64_t block_rows = rows - block_first_row;
        const uint32_t count = static_cast<uint32_t>(
            block_rows < uint64_t{Fp8KR::FP8_KR_THREADS} * Rows ? block_rows : uint64_t{Fp8KR::FP8_KR_THREADS} * Rows);
        fp8KRWriteStagedValues<FixedOp>(staged, block_first_row, count, fold, output, output_dtype, output_finalize,
                                        output_scale);
        __syncthreads();
    }
}

template <int RowBytes>
__global__ __launch_bounds__(Fp8KR::FP8_KR_THREADS) void fp8KRArgVectorKernel(const uint4* __restrict__ vectors,
                                                                               uint64_t rows,
                                                                               Fp8ArgFold fold,
                                                                               Fp8ArgOutputs outputs) {
    constexpr int Rows = FP8_KR_VECTOR_ROWS<RowBytes>;
    constexpr int Vectors = Rows * RowBytes / 16;
    constexpr int Words = 4 * Vectors;
    static_assert(Rows * RowBytes == 16 * Vectors && Rows <= static_cast<int>(FP8_KR_PACK_MAX_ROWS));
    __shared__ uint32_t staged[FP8_KR_PACK_STAGE_WORDS];
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(vectors);
    const uint64_t owners = (rows + Rows - 1) / Rows;
    const uint64_t full_owners = rows / Rows;
    const uint64_t block_stride = static_cast<uint64_t>(gridDim.x) * Fp8KR::FP8_KR_THREADS;
    for (uint64_t block_owner = static_cast<uint64_t>(blockIdx.x) * Fp8KR::FP8_KR_THREADS; block_owner < owners;
         block_owner += block_stride) {
        const uint64_t block_first_row = block_owner * Rows;
        const uint64_t owner = block_owner + threadIdx.x;
        if (owner < full_owners) {
            Fp8ArgKeys4 keys[Words];
#pragma unroll
            for (int v = 0; v < Vectors; ++v) {
                const uint4 packet = __ldg(vectors + owner * Vectors + v);
                keys[4 * v] = fp8ArgKeys4(packet.x, fold);
                keys[4 * v + 1] = fp8ArgKeys4(packet.y, fold);
                keys[4 * v + 2] = fp8ArgKeys4(packet.z, fold);
                keys[4 * v + 3] = fp8ArgKeys4(packet.w, fold);
            }
            uint32_t best[Rows];
#pragma unroll
            for (int r = 0; r < Rows; ++r) best[r] = FP8_ARG_LANE_IDENTITY;
#pragma unroll
            for (int b = 0; b < Rows * RowBytes; ++b) {
                best[b / RowBytes] = fp8ArgMin(best[b / RowBytes],
                                               fp8ArgLaneCandidate(keys[b / 4], static_cast<uint32_t>(b % 4),
                                                                   static_cast<uint32_t>(b % RowBytes)));
            }
#pragma unroll
            for (int r = 0; r < Rows; ++r) staged[fp8KRStageSlot(threadIdx.x * Rows + r)] = best[r];
        } else if (owner < owners) {
            for (uint64_t row = owner * Rows; row < rows; ++row) {
                uint32_t best = FP8_ARG_LANE_IDENTITY;
                for (int i = 0; i < RowBytes; ++i) {
                    const Fp8ArgKeys4 keys = fp8ArgKeys4(__ldg(bytes + row * RowBytes + i), fold);
                    best = fp8ArgMin(best, fp8ArgLaneCandidate(keys, 0U, static_cast<uint32_t>(i)));
                }
                staged[fp8KRStageSlot(static_cast<uint32_t>(row - block_first_row))] = best;
            }
        }
        __syncthreads();
        const uint64_t block_rows = rows - block_first_row;
        const uint32_t count = static_cast<uint32_t>(
            block_rows < uint64_t{Fp8KR::FP8_KR_THREADS} * Rows ? block_rows : uint64_t{Fp8KR::FP8_KR_THREADS} * Rows);
        fp8KRWriteStagedArgs(staged, block_first_row, count, outputs);
        __syncthreads();
    }
}

// The compiled vector row lengths (Fp8KR::FP8_KR_VECTOR_ROW_BYTES); false when RowBytes has no kernel.
template <typename LaunchT>
bool dispatchFp8KRVectorRowBytes(uint64_t row_bytes, const LaunchT& launch) {
    switch (row_bytes) {
        case 2: launch.template operator()<2>(); return true;
        case 3: launch.template operator()<3>(); return true;
        case 5: launch.template operator()<5>(); return true;
        case 6: launch.template operator()<6>(); return true;
        default: return false;
    }
}

// Block-wide combine of one value per thread (warp shuffles, then warp 0 over the warp totals). Thread 0 gets the
// total. Every thread of the block must call it; `scratch` is reused only after a __syncthreads.
template <int FixedOp>
__device__ __forceinline__ float fp8KRBlockCombine(float value, float (&scratch)[FP8_KR_WARPS]) {
    const Fp8TypedReductionOp<FixedOp> combine{};
    for (uint32_t offset = FP8_KR_WARP_LANES / 2; offset != 0; offset >>= 1) {
        value = combine(value, __shfl_xor_sync(0xffffffffU, value, static_cast<int>(offset)));
    }
    const uint32_t warp = threadIdx.x / FP8_KR_WARP_LANES;
    if ((threadIdx.x % FP8_KR_WARP_LANES) == 0) scratch[warp] = value;
    __syncthreads();
    float total = combine.identity();
    if (threadIdx.x == 0) {
#pragma unroll
        for (uint32_t w = 0; w < FP8_KR_WARPS; ++w) total = combine(total, scratch[w]);
    }
    return total;
}

__device__ __forceinline__ uint64_t fp8KRBlockArgMin(uint64_t value, uint64_t (&scratch)[FP8_KR_WARPS]) {
    for (uint32_t offset = FP8_KR_WARP_LANES / 2; offset != 0; offset >>= 1) {
        const uint64_t other =
            __shfl_xor_sync(0xffffffffU, static_cast<unsigned long long>(value), static_cast<int>(offset));
        value = fp8ArgMin(value, other);
    }
    const uint32_t warp = threadIdx.x / FP8_KR_WARP_LANES;
    if ((threadIdx.x % FP8_KR_WARP_LANES) == 0) scratch[warp] = value;
    __syncthreads();
    uint64_t total = FP8_ARG_WIDE_IDENTITY;
    if (threadIdx.x == 0) {
#pragma unroll
        for (uint32_t w = 0; w < FP8_KR_WARPS; ++w) total = fp8ArgMin(total, scratch[w]);
    }
    return total;
}

// ---- Split: one CTA per (row, chunk) writes the chunk's partial (MIN / MAX in the folded MIN domain); the finishing
// kernel gives each row one CTA, which combines the row's partials in chunk order (thread t takes chunks t, t + 256, ...,
// then the fixed block combine), unfolds, finalizes and stores once. No atomics; the result is deterministic.
template <typename InputT, int FixedOp, int FixedInputTransform>
__global__ __launch_bounds__(Fp8KR::FP8_KR_THREADS) void fp8KRValueSplitKernel(const uint32_t* __restrict__ words,
                                                                                uint32_t head,
                                                                                uint64_t rows,
                                                                                uint64_t row_bytes,
                                                                                uint64_t chunk_bytes,
                                                                                uint64_t chunks,
                                                                                Fp8ExtremumFold fold,
                                                                                float* __restrict__ partials) {
    static_assert(IS_RETAINED_KR_INPUT<InputT>);
    __shared__ float warp_totals[FP8_KR_WARPS];
    const Fp8TypedReductionOp<FixedOp> combine{};
    const uint64_t segments = rows * chunks;
    for (uint64_t segment = blockIdx.x; segment < segments; segment += gridDim.x) {
        const Fp8KRSegmentBounds bounds = fp8KRSegmentBounds(segment, chunks, row_bytes, chunk_bytes, head);
        float even = combine.identity();
        float odd = combine.identity();
        fp8KRAccumulateWords<InputT, FixedOp, FixedInputTransform>(words, (bounds.begin >> 2) + threadIdx.x,
                                                                   (bounds.end + 3) >> 2, Fp8KR::FP8_KR_THREADS,
                                                                   bounds, fold, even, odd);
        const float chunk_total = fp8KRBlockCombine<FixedOp>(combine(even, odd), warp_totals);
        if (threadIdx.x == 0) partials[segment] = chunk_total;
        __syncthreads();  // warp_totals is rewritten by the next segment
    }
}

template <int FixedOp>
__global__ __launch_bounds__(Fp8KR::FP8_KR_THREADS) void fp8KRValueFinishKernel(const float* __restrict__ partials,
                                                                                 uint64_t rows,
                                                                                 uint64_t chunks,
                                                                                 Fp8ExtremumFold fold,
                                                                                 void* output,
                                                                                 DataType output_dtype,
                                                                                 AdditiveFinalizeFp32 output_finalize,
                                                                                 float output_scale) {
    __shared__ float warp_totals[FP8_KR_WARPS];
    const Fp8TypedReductionOp<FixedOp> combine{};
    constexpr uint64_t Threads = Fp8KR::FP8_KR_THREADS;
    for (uint64_t row = blockIdx.x; row < rows; row += gridDim.x) {
        const float* row_partials = partials + row * chunks;
        float a = combine.identity();
        float b = combine.identity();
        uint64_t chunk = threadIdx.x;
        for (; chunk + 3 * Threads < chunks; chunk += 4 * Threads) {
            const float x0 = row_partials[chunk];
            const float x1 = row_partials[chunk + Threads];
            const float x2 = row_partials[chunk + 2 * Threads];
            const float x3 = row_partials[chunk + 3 * Threads];
            a = combine(combine(a, x0), x2);
            b = combine(combine(b, x1), x3);
        }
        for (; chunk < chunks; chunk += Threads) a = combine(a, row_partials[chunk]);
        const float row_total = fp8KRBlockCombine<FixedOp>(combine(a, b), warp_totals);
        if (threadIdx.x == 0) {
            fp8KRStoreValue<FixedOp>(output, output_dtype, row, row_total, fold, output_finalize, output_scale);
        }
        __syncthreads();
    }
}

// ---- Block (1156): one CTA per whole row, stored directly (the Split partial kernel's mapping with one chunk, no
// scratch, no finishing launch).
template <typename InputT, int FixedOp, int FixedInputTransform>
__global__ __launch_bounds__(Fp8KR::FP8_KR_THREADS) void fp8KRValueBlockKernel(const uint32_t* __restrict__ words,
                                                                                uint32_t head,
                                                                                uint64_t rows,
                                                                                uint64_t row_bytes,
                                                                                Fp8ExtremumFold fold,
                                                                                void* output,
                                                                                DataType output_dtype,
                                                                                AdditiveFinalizeFp32 output_finalize,
                                                                                float output_scale) {
    static_assert(IS_RETAINED_KR_INPUT<InputT>);
    __shared__ float warp_totals[FP8_KR_WARPS];
    const Fp8TypedReductionOp<FixedOp> combine{};
    for (uint64_t row = blockIdx.x; row < rows; row += gridDim.x) {
        const Fp8KRSegmentBounds bounds = fp8KRSegmentBounds(row, 1, row_bytes, row_bytes, head);
        float even = combine.identity();
        float odd = combine.identity();
        fp8KRAccumulateWords<InputT, FixedOp, FixedInputTransform>(words, (bounds.begin >> 2) + threadIdx.x,
                                                                   (bounds.end + 3) >> 2, Fp8KR::FP8_KR_THREADS,
                                                                   bounds, fold, even, odd);
        const float row_total = fp8KRBlockCombine<FixedOp>(combine(even, odd), warp_totals);
        if (threadIdx.x == 0) {
            fp8KRStoreValue<FixedOp>(output, output_dtype, row, row_total, fold, output_finalize, output_scale);
        }
        __syncthreads();  // warp_totals is rewritten by the next row
    }
}

__global__ __launch_bounds__(Fp8KR::FP8_KR_THREADS) void fp8KRArgBlockKernel(const uint32_t* __restrict__ words,
                                                                              uint32_t head,
                                                                              uint64_t rows,
                                                                              uint64_t row_bytes,
                                                                              Fp8ArgFold fold,
                                                                              Fp8ArgOutputs outputs) {
    __shared__ uint64_t warp_bests[FP8_KR_WARPS];
    for (uint64_t row = blockIdx.x; row < rows; row += gridDim.x) {
        const Fp8KRSegmentBounds bounds = fp8KRSegmentBounds(row, 1, row_bytes, row_bytes, head);
        uint64_t best = FP8_ARG_WIDE_IDENTITY;
        fp8KRArgAccumulateWords(words, (bounds.begin >> 2) + threadIdx.x, (bounds.end + 3) >> 2,
                                Fp8KR::FP8_KR_THREADS, bounds, fold, best);
        const uint64_t row_best = fp8KRBlockArgMin(best, warp_bests);
        if (threadIdx.x == 0) storeFp8ArgPackedResult(outputs, row, row_best);
        __syncthreads();
    }
}

// ---- ARG Group: G lanes per row.
__global__ __launch_bounds__(Fp8KR::FP8_KR_THREADS) void fp8KRArgGroupKernel(const uint32_t* __restrict__ words,
                                                                              uint32_t head,
                                                                              uint64_t rows,
                                                                              uint64_t row_bytes,
                                                                              uint32_t group_shift,
                                                                              Fp8ArgFold fold,
                                                                              Fp8ArgOutputs outputs) {
    const uint32_t group_lanes = 1U << group_shift;
    const uint32_t lane = threadIdx.x & (group_lanes - 1U);
    const uint64_t group = threadIdx.x >> group_shift;
    const uint64_t groups_per_block = Fp8KR::FP8_KR_THREADS >> group_shift;
    const uint64_t block_stride = static_cast<uint64_t>(gridDim.x) * groups_per_block;
    for (uint64_t base = static_cast<uint64_t>(blockIdx.x) * groups_per_block; base < rows; base += block_stride) {
        const uint64_t row = base + group;
        uint64_t best = FP8_ARG_WIDE_IDENTITY;
        if (row < rows) {
            const Fp8KRSegmentBounds bounds = fp8KRSegmentBounds(row, 1, row_bytes, row_bytes, head);
            fp8KRArgAccumulateWords(words, (bounds.begin >> 2) + lane, (bounds.end + 3) >> 2, group_lanes, bounds,
                                    fold, best);
        }
        for (uint32_t offset = group_lanes >> 1; offset != 0; offset >>= 1) {
            const uint64_t other =
                __shfl_xor_sync(0xffffffffU, static_cast<unsigned long long>(best), static_cast<int>(offset));
            best = fp8ArgMin(best, other);
        }
        if (lane == 0 && row < rows) storeFp8ArgPackedResult(outputs, row, best);
    }
}

// ---- ARG Pack: per row, the 32-bit lane candidate (key << 24 | row-local index << 1 | -0) of each byte; row_bytes
// <= 32 keeps the index far below FP8_ARG_MAX_LANE_ROWS. Completed rows go to the block's staging buffer (1150).
__global__ __launch_bounds__(Fp8KR::FP8_KR_THREADS) void fp8KRArgPackKernel(const uint32_t* __restrict__ words,
                                                                             uint32_t head,
                                                                             uint64_t rows,
                                                                             uint32_t row_bytes,
                                                                             uint32_t rows_per_thread,
                                                                             Fp8ArgFold fold,
                                                                             Fp8ArgOutputs outputs) {
    __shared__ uint32_t staged[FP8_KR_PACK_STAGE_WORDS];
    const uint64_t owners = (rows + rows_per_thread - 1) / rows_per_thread;
    const uint64_t block_stride = static_cast<uint64_t>(gridDim.x) * Fp8KR::FP8_KR_THREADS;
    for (uint64_t block_owner = static_cast<uint64_t>(blockIdx.x) * Fp8KR::FP8_KR_THREADS; block_owner < owners;
         block_owner += block_stride) {
        const uint64_t block_first_row = block_owner * rows_per_thread;
        const uint64_t owner = block_owner + threadIdx.x;
        if (owner < owners) {
            const uint64_t row = owner * rows_per_thread;
            uint32_t local = threadIdx.x * rows_per_thread;
            const uint64_t count = rows - row < rows_per_thread ? rows - row : rows_per_thread;
            const uint64_t begin = static_cast<uint64_t>(head) + row * row_bytes;
            const uint64_t end = begin + count * row_bytes;
            const uint64_t word_end = (end + 3) >> 2;
            uint32_t best = FP8_ARG_LANE_IDENTITY;
            uint32_t position = 0;
            for (uint64_t word = begin >> 2; word < word_end; word += FP8_KR_UNROLL) {
                uint32_t packed[FP8_KR_UNROLL];
#pragma unroll
                for (uint32_t u = 0; u < FP8_KR_UNROLL; ++u) {
                    packed[u] = word + u < word_end ? __ldg(words + word + u) : 0U;
                }
#pragma unroll
                for (uint32_t u = 0; u < FP8_KR_UNROLL; ++u) {
                    if (word + u < word_end) {
                        const Fp8ArgKeys4 keys = fp8ArgKeys4(packed[u], fold);
                        const uint64_t first = (word + u) << 2;
#pragma unroll
                        for (uint32_t j = 0; j < 4; ++j) {
                            if (first + j >= begin && first + j < end) {
                                best = fp8ArgMin(best, fp8ArgLaneCandidate(keys, j, position));
                                if (++position == row_bytes) {
                                    staged[fp8KRStageSlot(local)] = best;
                                    best = FP8_ARG_LANE_IDENTITY;
                                    position = 0;
                                    ++local;
                                }
                            }
                        }
                    }
                }
            }
        }
        __syncthreads();
        const uint64_t block_rows = rows - block_first_row;
        const uint32_t count = static_cast<uint32_t>(
            block_rows < uint64_t{Fp8KR::FP8_KR_THREADS} * rows_per_thread ? block_rows
                                                                            : uint64_t{Fp8KR::FP8_KR_THREADS} *
                                                                                  rows_per_thread);
        fp8KRWriteStagedArgs(staged, block_first_row, count, outputs);
        __syncthreads();
    }
}

// ---- ARG Split: as VALUE Split, with 64-bit wide candidates and the order-free wide minimum.
__global__ __launch_bounds__(Fp8KR::FP8_KR_THREADS) void fp8KRArgSplitKernel(const uint32_t* __restrict__ words,
                                                                              uint32_t head,
                                                                              uint64_t rows,
                                                                              uint64_t row_bytes,
                                                                              uint64_t chunk_bytes,
                                                                              uint64_t chunks,
                                                                              Fp8ArgFold fold,
                                                                              uint64_t* __restrict__ partials) {
    __shared__ uint64_t warp_bests[FP8_KR_WARPS];
    const uint64_t segments = rows * chunks;
    for (uint64_t segment = blockIdx.x; segment < segments; segment += gridDim.x) {
        const Fp8KRSegmentBounds bounds = fp8KRSegmentBounds(segment, chunks, row_bytes, chunk_bytes, head);
        uint64_t best = FP8_ARG_WIDE_IDENTITY;
        fp8KRArgAccumulateWords(words, (bounds.begin >> 2) + threadIdx.x, (bounds.end + 3) >> 2,
                                Fp8KR::FP8_KR_THREADS, bounds, fold, best);
        const uint64_t chunk_best = fp8KRBlockArgMin(best, warp_bests);
        if (threadIdx.x == 0) partials[segment] = chunk_best;
        __syncthreads();
    }
}

__global__ __launch_bounds__(Fp8KR::FP8_KR_THREADS) void fp8KRArgFinishKernel(const uint64_t* __restrict__ partials,
                                                                               uint64_t rows,
                                                                               uint64_t chunks,
                                                                               Fp8ArgOutputs outputs) {
    __shared__ uint64_t warp_bests[FP8_KR_WARPS];
    constexpr uint64_t Threads = Fp8KR::FP8_KR_THREADS;
    for (uint64_t row = blockIdx.x; row < rows; row += gridDim.x) {
        const uint64_t* row_partials = partials + row * chunks;
        uint64_t a = FP8_ARG_WIDE_IDENTITY;
        uint64_t b = FP8_ARG_WIDE_IDENTITY;
        uint64_t chunk = threadIdx.x;
        for (; chunk + 3 * Threads < chunks; chunk += 4 * Threads) {
            const uint64_t x0 = row_partials[chunk];
            const uint64_t x1 = row_partials[chunk + Threads];
            const uint64_t x2 = row_partials[chunk + 2 * Threads];
            const uint64_t x3 = row_partials[chunk + 3 * Threads];
            a = fp8ArgMin(fp8ArgMin(a, x0), x2);
            b = fp8ArgMin(fp8ArgMin(b, x1), x3);
        }
        for (; chunk < chunks; chunk += Threads) a = fp8ArgMin(a, row_partials[chunk]);
        const uint64_t row_best = fp8KRBlockArgMin(fp8ArgMin(a, b), warp_bests);
        if (threadIdx.x == 0) storeFp8ArgPackedResult(outputs, row, row_best);
        __syncthreads();
    }
}

// ---- 1181: FP32 R / KR ARG (plans/Thor_FP32_Dense_Reduction_Plan, phase 3). Copies of the FP8 ARG Group / Block /
// Pack / Split / Finish kernels (the candidate differs, not just the input type, so they live independently and FP8's
// kernels are untouched). One element per word; rows are whole words on a 4-byte aligned input.
//
// The candidate is two words: a 32-bit ordered key and a 64-bit tail = row-local index << 1 | negative zero, compared
// key first, then tail (user decision 2026-11-02: two words first, a 31-bit index only if this proves slow). The key
// of the folded value x (argmax = argmin of -x, so both share the kernels):
//   positive x: key = x | 0x80000000        (+0 -> 0x80000000, +inf -> 0xFF800000)
//   negative x: key = ~x                    (-inf -> 0x007FFFFF; -0 is given the +0 key, its sign in the tail)
//   NaN:        key = 0                     (NaN wins and propagates; no ordinary value has key 0)
// so unsigned order is value order, and key 0xFFFFFFFF never occurs (a strict identity). Within one lane the indices
// only increase, so a strictly smaller key replaces the lane's best (ties keep the lower index); across lanes, warps,
// CTAs and chunks the full (key, tail) minimum merges deterministically: Thor's ARG contract (NaNs propagate, the
// lowest index wins ties, NaN-NaN and -0/+0 ties included). Split partials pack to one 64-bit word, key << 32 |
// chunk-local index << 1 | negative zero (chunks of at most 2^31 elements, the launcher checks), so the Split scratch
// is FP8's (fp8KRScratchBytes(shape, true)); the finishing kernel restores the row-local index from the chunk.
// 1198: the candidate, its key, value, outputs and warp shuffle are in Fp32ArgKey.cuh (shared with FP32 RK ARG).

// The words [first_word, word_end) a lane visits with stride `step`; row_word is the row's first word.
__device__ __forceinline__ void fp32KRArgAccumulateWords(const uint32_t* __restrict__ words, uint64_t first_word,
                                                         uint64_t word_end, uint64_t step, uint64_t row_word,
                                                         uint32_t xor_mask, Fp32ArgCandidate& best) {
    for (uint64_t word = first_word; word < word_end; word += static_cast<uint64_t>(FP8_KR_UNROLL) * step) {
        uint32_t packed[FP8_KR_UNROLL];
#pragma unroll
        for (uint32_t u = 0; u < FP8_KR_UNROLL; ++u) {
            const uint64_t w = word + static_cast<uint64_t>(u) * step;
            packed[u] = w < word_end ? __ldg(words + w) : 0U;
        }
#pragma unroll
        for (uint32_t u = 0; u < FP8_KR_UNROLL; ++u) {
            const uint64_t w = word + static_cast<uint64_t>(u) * step;
            if (w < word_end) {
                uint32_t negative_zero;
                const uint32_t key = fp32ArgKey(packed[u], xor_mask, negative_zero);
                if (key < best.key) best = {key, ((w - row_word) << 1) | negative_zero};
            }
        }
    }
}


__device__ __forceinline__ Fp32ArgCandidate fp32KRBlockArgMin(Fp32ArgCandidate value,
                                                              uint32_t (&keys)[FP8_KR_WARPS],
                                                              uint64_t (&tails)[FP8_KR_WARPS]) {
    for (uint32_t offset = FP8_KR_WARP_LANES / 2; offset != 0; offset >>= 1) {
        value = fp32ArgMin(value, fp32ArgShuffleXor(value, static_cast<int>(offset)));
    }
    const uint32_t warp = threadIdx.x / FP8_KR_WARP_LANES;
    if ((threadIdx.x % FP8_KR_WARP_LANES) == 0) {
        keys[warp] = value.key;
        tails[warp] = value.tail;
    }
    __syncthreads();
    Fp32ArgCandidate total = fp32ArgIdentity();
    if (threadIdx.x == 0) {
#pragma unroll
        for (uint32_t w = 0; w < FP8_KR_WARPS; ++w) total = fp32ArgMin(total, {keys[w], tails[w]});
    }
    return total;
}

// ---- FP32 ARG Group: G lanes per row (the loop bound is block-uniform, so every lane reaches the shuffles).
__global__ __launch_bounds__(Fp8KR::FP8_KR_THREADS) void fp32KRArgGroupKernel(const uint32_t* __restrict__ words,
                                                                               uint64_t rows,
                                                                               uint64_t row_words,
                                                                               uint32_t group_shift,
                                                                               Fp32ArgOutputs outputs) {
    const uint32_t group_lanes = 1U << group_shift;
    const uint32_t lane = threadIdx.x & (group_lanes - 1U);
    const uint64_t group = threadIdx.x >> group_shift;
    const uint64_t groups_per_block = Fp8KR::FP8_KR_THREADS >> group_shift;
    const uint64_t block_stride = static_cast<uint64_t>(gridDim.x) * groups_per_block;
    for (uint64_t base = static_cast<uint64_t>(blockIdx.x) * groups_per_block; base < rows; base += block_stride) {
        const uint64_t row = base + group;
        Fp32ArgCandidate best = fp32ArgIdentity();
        if (row < rows) {
            const uint64_t row_word = row * row_words;
            fp32KRArgAccumulateWords(words, row_word + lane, row_word + row_words, group_lanes, row_word,
                                     outputs.xor_mask, best);
        }
        for (uint32_t offset = group_lanes >> 1; offset != 0; offset >>= 1) {
            best = fp32ArgMin(best, fp32ArgShuffleXor(best, static_cast<int>(offset)));
        }
        if (lane == 0 && row < rows) storeFp32ArgResult(outputs, row, best);
    }
}

// ---- FP32 ARG Block: one CTA per whole row.
__global__ __launch_bounds__(Fp8KR::FP8_KR_THREADS) void fp32KRArgBlockKernel(const uint32_t* __restrict__ words,
                                                                               uint64_t rows,
                                                                               uint64_t row_words,
                                                                               Fp32ArgOutputs outputs) {
    __shared__ uint32_t warp_keys[FP8_KR_WARPS];
    __shared__ uint64_t warp_tails[FP8_KR_WARPS];
    for (uint64_t row = blockIdx.x; row < rows; row += gridDim.x) {
        const uint64_t row_word = row * row_words;
        Fp32ArgCandidate best = fp32ArgIdentity();
        fp32KRArgAccumulateWords(words, row_word + threadIdx.x, row_word + row_words, Fp8KR::FP8_KR_THREADS,
                                 row_word, outputs.xor_mask, best);
        const Fp32ArgCandidate row_best = fp32KRBlockArgMin(best, warp_keys, warp_tails);
        if (threadIdx.x == 0) storeFp32ArgResult(outputs, row, row_best);
        __syncthreads();  // the warp slots are rewritten by the next row
    }
}

// ---- FP32 ARG Pack: one thread owns rows_per_thread consecutive rows of at most 8 elements. A row's best is one
// 64-bit word (key << 32 | row-local index << 1 | -0); completed rows go to the block's staging buffer (1150).
constexpr uint32_t FP32_KR_ARG_PACK_STAGE_WORDS = FP8_KR_PACK_STAGE_WORDS;
__global__ __launch_bounds__(Fp8KR::FP8_KR_THREADS) void fp32KRArgPackKernel(const uint32_t* __restrict__ words,
                                                                              uint64_t rows,
                                                                              uint32_t row_words,
                                                                              uint32_t rows_per_thread,
                                                                              Fp32ArgOutputs outputs) {
    __shared__ uint64_t staged[FP32_KR_ARG_PACK_STAGE_WORDS];
    const uint64_t owners = (rows + rows_per_thread - 1) / rows_per_thread;
    const uint64_t block_stride = static_cast<uint64_t>(gridDim.x) * Fp8KR::FP8_KR_THREADS;
    for (uint64_t block_owner = static_cast<uint64_t>(blockIdx.x) * Fp8KR::FP8_KR_THREADS; block_owner < owners;
         block_owner += block_stride) {
        const uint64_t block_first_row = block_owner * rows_per_thread;
        const uint64_t owner = block_owner + threadIdx.x;
        if (owner < owners) {
            const uint64_t row = owner * rows_per_thread;
            uint32_t local = threadIdx.x * rows_per_thread;
            const uint64_t count = rows - row < rows_per_thread ? rows - row : rows_per_thread;
            const uint64_t word_begin = row * row_words;
            const uint64_t word_end = word_begin + count * row_words;
            uint64_t best = ~uint64_t{0};
            uint32_t position = 0;
            for (uint64_t word = word_begin; word < word_end; word += FP8_KR_UNROLL) {
                uint32_t packed[FP8_KR_UNROLL];
#pragma unroll
                for (uint32_t u = 0; u < FP8_KR_UNROLL; ++u) {
                    packed[u] = word + u < word_end ? __ldg(words + word + u) : 0U;
                }
#pragma unroll
                for (uint32_t u = 0; u < FP8_KR_UNROLL; ++u) {
                    if (word + u < word_end) {
                        uint32_t negative_zero;
                        const uint32_t key = fp32ArgKey(packed[u], outputs.xor_mask, negative_zero);
                        const uint64_t candidate =
                            (static_cast<uint64_t>(key) << 32) | (position << 1) | negative_zero;
                        best = candidate < best ? candidate : best;
                        if (++position == row_words) {
                            staged[fp8KRStageSlot(local)] = best;
                            best = ~uint64_t{0};
                            position = 0;
                            ++local;
                        }
                    }
                }
            }
        }
        __syncthreads();
        const uint64_t block_rows = rows - block_first_row;
        const uint32_t count = static_cast<uint32_t>(
            block_rows < uint64_t{Fp8KR::FP8_KR_THREADS} * rows_per_thread ? block_rows
                                                                            : uint64_t{Fp8KR::FP8_KR_THREADS} *
                                                                                  rows_per_thread);
#pragma unroll 1
        for (uint32_t i = threadIdx.x; i < count; i += Fp8KR::FP8_KR_THREADS) {
            const uint64_t packed_best = staged[fp8KRStageSlot(i)];
            storeFp32ArgResult(outputs, block_first_row + i,
                               {static_cast<uint32_t>(packed_best >> 32), packed_best & 0xffffffffULL});
        }
        __syncthreads();
    }
}

// ---- FP32 ARG Split: one CTA per (row, chunk) writes the chunk's packed partial; the finishing kernel (one CTA per
// row) restores row-local indices from the chunk and takes the full minimum.
__global__ __launch_bounds__(Fp8KR::FP8_KR_THREADS) void fp32KRArgSplitKernel(const uint32_t* __restrict__ words,
                                                                               uint64_t rows,
                                                                               uint64_t row_words,
                                                                               uint64_t chunk_words,
                                                                               uint64_t chunks,
                                                                               uint32_t xor_mask,
                                                                               uint64_t* __restrict__ partials) {
    __shared__ uint32_t warp_keys[FP8_KR_WARPS];
    __shared__ uint64_t warp_tails[FP8_KR_WARPS];
    const uint64_t segments = rows * chunks;
    for (uint64_t segment = blockIdx.x; segment < segments; segment += gridDim.x) {
        const uint64_t row = segment / chunks;
        const uint64_t chunk_word = row * row_words + (segment - row * chunks) * chunk_words;
        const uint64_t row_end = (row + 1) * row_words;
        const uint64_t chunk_end = chunk_word + chunk_words < row_end ? chunk_word + chunk_words : row_end;
        Fp32ArgCandidate best = fp32ArgIdentity();
        // Chunk-local indices: the chunk's first word is index 0.
        fp32KRArgAccumulateWords(words, chunk_word + threadIdx.x, chunk_end, Fp8KR::FP8_KR_THREADS, chunk_word,
                                 xor_mask, best);
        const Fp32ArgCandidate chunk_best = fp32KRBlockArgMin(best, warp_keys, warp_tails);
        if (threadIdx.x == 0) partials[segment] = (static_cast<uint64_t>(chunk_best.key) << 32) | chunk_best.tail;
        __syncthreads();
    }
}

__global__ __launch_bounds__(Fp8KR::FP8_KR_THREADS) void fp32KRArgFinishKernel(const uint64_t* __restrict__ partials,
                                                                                uint64_t rows,
                                                                                uint64_t chunks,
                                                                                uint64_t chunk_words,
                                                                                Fp32ArgOutputs outputs) {
    __shared__ uint32_t warp_keys[FP8_KR_WARPS];
    __shared__ uint64_t warp_tails[FP8_KR_WARPS];
    for (uint64_t row = blockIdx.x; row < rows; row += gridDim.x) {
        const uint64_t* row_partials = partials + row * chunks;
        Fp32ArgCandidate best = fp32ArgIdentity();
        for (uint64_t chunk = threadIdx.x; chunk < chunks; chunk += Fp8KR::FP8_KR_THREADS) {
            const uint64_t partial = row_partials[chunk];
            best = fp32ArgMin(best, {static_cast<uint32_t>(partial >> 32),
                                     ((chunk * chunk_words) << 1) + (partial & 0xffffffffULL)});
        }
        const Fp32ArgCandidate row_best = fp32KRBlockArgMin(best, warp_keys, warp_tails);
        if (threadIdx.x == 0) storeFp32ArgResult(outputs, row, row_best);
        __syncthreads();
    }
}

// ---- 1214: carried-index FP32 R / KR ARG (plans/Thor_FP32_Dense_Reduction_Plan, phase 8). The FP32 KR ARG families
// and work mappings with a carried index (UINT32 / UINT64) per element: the element at row-local position p has the
// composed index carried + p * row_stride (row_stride 0 for an intermediate run) and the tail composed << 1 | negative
// zero. Composed indices are not monotone in the position, so every merge, within a lane included, is the full
// (key, tail) minimum. Split partials are wide (Fp32ArgWidePartials over [rows, chunks], fp32ArgWideScratchBytes) with
// final indices; the carried finishing kernel only merges. The Pack stage keeps FP32 KR ARG's one word per row, key
// << 32 | winning position << 1 | negative zero (wide staging would need 50 KiB of shared memory), and the block's
// store loop reloads the winner's carried index to compose its index.

// The words [first_word, word_end) a lane visits with stride `step`; row_word is the row's first word.
template <typename IndexT>
__device__ __forceinline__ void fp32KRCarriedArgAccumulateWords(const uint32_t* __restrict__ words,
                                                                const IndexT* __restrict__ carried,
                                                                uint64_t first_word, uint64_t word_end, uint64_t step,
                                                                uint64_t row_word, uint32_t xor_mask,
                                                                uint64_t row_stride, Fp32ArgCandidate& best) {
    for (uint64_t word = first_word; word < word_end; word += static_cast<uint64_t>(FP8_KR_UNROLL) * step) {
        uint32_t packed[FP8_KR_UNROLL];
        uint64_t indices[FP8_KR_UNROLL];
#pragma unroll
        for (uint32_t u = 0; u < FP8_KR_UNROLL; ++u) {
            const uint64_t w = word + static_cast<uint64_t>(u) * step;
            packed[u] = w < word_end ? __ldg(words + w) : 0U;
            indices[u] = w < word_end ? static_cast<uint64_t>(carried[w]) : 0U;
        }
#pragma unroll
        for (uint32_t u = 0; u < FP8_KR_UNROLL; ++u) {
            const uint64_t w = word + static_cast<uint64_t>(u) * step;
            if (w < word_end) {
                uint32_t negative_zero;
                const uint32_t key = fp32ArgKey(packed[u], xor_mask, negative_zero);
                fp32ArgKeepMin(best.key, best.tail, key,
                               ((indices[u] + (w - row_word) * row_stride) << 1) | negative_zero);
            }
        }
    }
}

// ---- Carried Group: G lanes per row (the loop bound is block-uniform, so every lane reaches the shuffles).
template <typename IndexT>
__global__ __launch_bounds__(Fp8KR::FP8_KR_THREADS) void fp32KRCarriedArgGroupKernel(
    const uint32_t* __restrict__ words, const IndexT* __restrict__ carried, uint64_t rows, uint64_t row_words,
    uint32_t group_shift, uint64_t row_stride, Fp32ArgOutputs outputs) {
    const uint32_t group_lanes = 1U << group_shift;
    const uint32_t lane = threadIdx.x & (group_lanes - 1U);
    const uint64_t group = threadIdx.x >> group_shift;
    const uint64_t groups_per_block = Fp8KR::FP8_KR_THREADS >> group_shift;
    const uint64_t block_stride = static_cast<uint64_t>(gridDim.x) * groups_per_block;
    for (uint64_t base = static_cast<uint64_t>(blockIdx.x) * groups_per_block; base < rows; base += block_stride) {
        const uint64_t row = base + group;
        Fp32ArgCandidate best = fp32ArgIdentity();
        if (row < rows) {
            const uint64_t row_word = row * row_words;
            fp32KRCarriedArgAccumulateWords<IndexT>(words, carried, row_word + lane, row_word + row_words,
                                                    group_lanes, row_word, outputs.xor_mask, row_stride, best);
        }
        for (uint32_t offset = group_lanes >> 1; offset != 0; offset >>= 1) {
            best = fp32ArgMin(best, fp32ArgShuffleXor(best, static_cast<int>(offset)));
        }
        if (lane == 0 && row < rows) storeFp32ArgResult(outputs, row, best);
    }
}

// ---- Carried Block: one CTA per whole row.
template <typename IndexT>
__global__ __launch_bounds__(Fp8KR::FP8_KR_THREADS) void fp32KRCarriedArgBlockKernel(
    const uint32_t* __restrict__ words, const IndexT* __restrict__ carried, uint64_t rows, uint64_t row_words,
    uint64_t row_stride, Fp32ArgOutputs outputs) {
    __shared__ uint32_t warp_keys[FP8_KR_WARPS];
    __shared__ uint64_t warp_tails[FP8_KR_WARPS];
    for (uint64_t row = blockIdx.x; row < rows; row += gridDim.x) {
        const uint64_t row_word = row * row_words;
        Fp32ArgCandidate best = fp32ArgIdentity();
        fp32KRCarriedArgAccumulateWords<IndexT>(words, carried, row_word + threadIdx.x, row_word + row_words,
                                                Fp8KR::FP8_KR_THREADS, row_word, outputs.xor_mask, row_stride, best);
        const Fp32ArgCandidate row_best = fp32KRBlockArgMin(best, warp_keys, warp_tails);
        if (threadIdx.x == 0) storeFp32ArgResult(outputs, row, row_best);
        __syncthreads();  // the warp slots are rewritten by the next row
    }
}

// ---- Carried Pack: one thread owns rows_per_thread consecutive rows of at most 8 elements and keeps each row's best
// (key, composed tail) and its position; the stage holds key << 32 | position << 1 | negative zero per row.
template <typename IndexT>
__global__ __launch_bounds__(Fp8KR::FP8_KR_THREADS) void fp32KRCarriedArgPackKernel(
    const uint32_t* __restrict__ words, const IndexT* __restrict__ carried, uint64_t rows, uint32_t row_words,
    uint32_t rows_per_thread, uint64_t row_stride, Fp32ArgOutputs outputs) {
    __shared__ uint64_t staged[FP32_KR_ARG_PACK_STAGE_WORDS];
    const uint64_t owners = (rows + rows_per_thread - 1) / rows_per_thread;
    const uint64_t block_stride = static_cast<uint64_t>(gridDim.x) * Fp8KR::FP8_KR_THREADS;
    for (uint64_t block_owner = static_cast<uint64_t>(blockIdx.x) * Fp8KR::FP8_KR_THREADS; block_owner < owners;
         block_owner += block_stride) {
        const uint64_t block_first_row = block_owner * rows_per_thread;
        const uint64_t owner = block_owner + threadIdx.x;
        if (owner < owners) {
            const uint64_t row = owner * rows_per_thread;
            uint32_t local = threadIdx.x * rows_per_thread;
            const uint64_t count = rows - row < rows_per_thread ? rows - row : rows_per_thread;
            const uint64_t word_begin = row * row_words;
            const uint64_t word_end = word_begin + count * row_words;
            uint32_t best_key = FP32_ARG_KEY_IDENTITY;
            uint64_t best_tail = FP32_ARG_TAIL_IDENTITY;
            uint32_t best_position = 0;
            uint32_t position = 0;
            for (uint64_t word = word_begin; word < word_end; word += FP8_KR_UNROLL) {
                uint32_t packed[FP8_KR_UNROLL];
                uint64_t indices[FP8_KR_UNROLL];
#pragma unroll
                for (uint32_t u = 0; u < FP8_KR_UNROLL; ++u) {
                    packed[u] = word + u < word_end ? __ldg(words + word + u) : 0U;
                    indices[u] = word + u < word_end ? static_cast<uint64_t>(carried[word + u]) : 0U;
                }
#pragma unroll
                for (uint32_t u = 0; u < FP8_KR_UNROLL; ++u) {
                    if (word + u < word_end) {
                        uint32_t negative_zero;
                        const uint32_t key = fp32ArgKey(packed[u], outputs.xor_mask, negative_zero);
                        const uint64_t tail = ((indices[u] + position * row_stride) << 1) | negative_zero;
                        if (key < best_key || (key == best_key && tail < best_tail)) {
                            best_key = key;
                            best_tail = tail;
                            best_position = position;
                        }
                        if (++position == row_words) {
                            staged[fp8KRStageSlot(local)] = (static_cast<uint64_t>(best_key) << 32) |
                                                            (best_position << 1) | (best_tail & 1U);
                            best_key = FP32_ARG_KEY_IDENTITY;
                            best_tail = FP32_ARG_TAIL_IDENTITY;
                            best_position = 0;
                            position = 0;
                            ++local;
                        }
                    }
                }
            }
        }
        __syncthreads();
        const uint64_t block_rows = rows - block_first_row;
        const uint32_t count = static_cast<uint32_t>(
            block_rows < uint64_t{Fp8KR::FP8_KR_THREADS} * rows_per_thread ? block_rows
                                                                            : uint64_t{Fp8KR::FP8_KR_THREADS} *
                                                                                  rows_per_thread);
#pragma unroll 1
        for (uint32_t i = threadIdx.x; i < count; i += Fp8KR::FP8_KR_THREADS) {
            const uint64_t packed_best = staged[fp8KRStageSlot(i)];
            const uint64_t row = block_first_row + i;
            const uint64_t position = (packed_best & 0xffffffffULL) >> 1;
            const uint64_t composed = static_cast<uint64_t>(carried[row * row_words + position]) + position * row_stride;
            storeFp32ArgResult(outputs, row,
                               {static_cast<uint32_t>(packed_best >> 32), (composed << 1) | (packed_best & 1U)});
        }
        __syncthreads();
    }
}

// ---- Carried Split: one CTA per (row, chunk) writes the chunk's wide partial (final composed tail); the carried
// finishing kernel (one CTA per row) takes the full minimum of the row's partials.
template <typename IndexT>
__global__ __launch_bounds__(Fp8KR::FP8_KR_THREADS) void fp32KRCarriedArgSplitKernel(
    const uint32_t* __restrict__ words, const IndexT* __restrict__ carried, uint64_t rows, uint64_t row_words,
    uint64_t chunk_words, uint64_t chunks, uint32_t xor_mask, uint64_t row_stride, Fp32ArgWidePartials partials) {
    __shared__ uint32_t warp_keys[FP8_KR_WARPS];
    __shared__ uint64_t warp_tails[FP8_KR_WARPS];
    const uint64_t segments = rows * chunks;
    for (uint64_t segment = blockIdx.x; segment < segments; segment += gridDim.x) {
        const uint64_t row = segment / chunks;
        const uint64_t row_word = row * row_words;
        const uint64_t chunk_word = row_word + (segment - row * chunks) * chunk_words;
        const uint64_t row_end = row_word + row_words;
        const uint64_t chunk_end = chunk_word + chunk_words < row_end ? chunk_word + chunk_words : row_end;
        Fp32ArgCandidate best = fp32ArgIdentity();
        fp32KRCarriedArgAccumulateWords<IndexT>(words, carried, chunk_word + threadIdx.x, chunk_end,
                                                Fp8KR::FP8_KR_THREADS, row_word, xor_mask, row_stride, best);
        const Fp32ArgCandidate chunk_best = fp32KRBlockArgMin(best, warp_keys, warp_tails);
        if (threadIdx.x == 0) {
            partials.keys[segment] = chunk_best.key;
            partials.tails[segment] = chunk_best.tail;
        }
        __syncthreads();
    }
}

__global__ __launch_bounds__(Fp8KR::FP8_KR_THREADS) void fp32KRCarriedArgFinishKernel(Fp32ArgWidePartials partials,
                                                                                       uint64_t rows,
                                                                                       uint64_t chunks,
                                                                                       Fp32ArgOutputs outputs) {
    __shared__ uint32_t warp_keys[FP8_KR_WARPS];
    __shared__ uint64_t warp_tails[FP8_KR_WARPS];
    for (uint64_t row = blockIdx.x; row < rows; row += gridDim.x) {
        const uint64_t first = row * chunks;
        Fp32ArgCandidate best = fp32ArgIdentity();
        for (uint64_t chunk = threadIdx.x; chunk < chunks; chunk += Fp8KR::FP8_KR_THREADS) {
            fp32ArgKeepMin(best.key, best.tail, partials.keys[first + chunk], partials.tails[first + chunk]);
        }
        const Fp32ArgCandidate row_best = fp32KRBlockArgMin(best, warp_keys, warp_tails);
        if (threadIdx.x == 0) storeFp32ArgResult(outputs, row, row_best);
        __syncthreads();
    }
}

struct Fp8KRLaunchGeometry {
    const uint32_t* words;
    uint32_t head;
    uint32_t group_shift;
    unsigned int blocks;
};

Fp8KRLaunchGeometry fp8KRLaunchGeometry(const Tensor& input, const Fp8KR::Fp8KRSegmentShape& shape) {
    const uintptr_t address = reinterpret_cast<uintptr_t>(input.getMemPtr<void>());
    uint32_t shift = 0;
    uint64_t blocks = 0;
    switch (shape.family()) {
        case Fp8KR::Fp8KRFamily::Pack:
            blocks = ceilDivideU64(ceilDivideU64(shape.rows, shape.rows_per_thread), Fp8KR::FP8_KR_THREADS);
            break;
        case Fp8KR::Fp8KRFamily::Group:
            while ((1U << shift) < shape.group_lanes) ++shift;
            blocks = ceilDivideU64(shape.rows, Fp8KR::FP8_KR_THREADS >> shift);
            break;
        case Fp8KR::Fp8KRFamily::Split:
            blocks = shape.segments();
            break;
        case Fp8KR::Fp8KRFamily::Block:
            blocks = shape.rows;
            break;
    }
    return {reinterpret_cast<const uint32_t*>(address & ~uintptr_t{3}), static_cast<uint32_t>(address & uintptr_t{3}),
            shift, static_cast<unsigned int>(std::min<uint64_t>(blocks, FP8_KR_MAX_GRID_BLOCKS))};
}

unsigned int fp8KRFinishBlocks(uint64_t rows) {
    return static_cast<unsigned int>(std::min<uint64_t>(rows, FP8_KR_MAX_GRID_BLOCKS));
}

void requireFp8KRInput(const Tensor& input, const Fp8KR::Fp8KRSegmentShape& shape, void* scratch,
                       uint64_t scratch_bytes, bool arg) {
    const DataType dtype = input.getDataType();
    if ((dtype != DataType::FP8_E4M3 && dtype != DataType::FP8_E5M2) || !Fp8KR::fp8KRSegmentShapeLegal(shape) ||
        input.getTotalNumElements() != shape.rows * shape.row_bytes) {
        throw std::logic_error("Retained FP8 KR pass received an unsupported input or segment shape.");
    }
    if (shape.family() == Fp8KR::Fp8KRFamily::Split &&
        (scratch == nullptr || scratch_bytes < Fp8KR::fp8KRScratchBytes(shape, arg) ||
         (reinterpret_cast<uintptr_t>(scratch) & uintptr_t{15}) != 0)) {
        throw std::logic_error("Retained FP8 KR Split pass requires its 16-byte aligned partial scratch.");
    }
}

// The VALUE launch for one input element type (FP8 E4M3 / E5M2, or 1179 FP32): the family kernel at the semantics.
template <typename InputT>
void launchRetainedKRValueTyped(const CubReductionStageSemantics& semantics, const Tensor& input, Tensor& output,
                                const Fp8KR::Fp8KRSegmentShape& shape, void* scratch, float output_scale,
                                Stream& stream, Fp8KR::Fp8KRSplitStages stages) {
    const AdditiveFinalizeFp32 output_finalize = fp8CompleteFinalizerFromSemantics(semantics);
    const Fp8KRLaunchGeometry launch = fp8KRLaunchGeometry(input, shape);
    const auto run = [&]<int FixedOp, int FixedInputTransform>(Fp8ExtremumFold fold, float scale) {
        void* out = output.getMemPtr<void>();
        const DataType out_dtype = output.getDataType();
        switch (shape.family()) {
            case Fp8KR::Fp8KRFamily::Pack:
                if constexpr (sizeof(InputT) == 1) {
                    if (shape.vector && (reinterpret_cast<uintptr_t>(input.getMemPtr<void>()) & uintptr_t{15}) == 0) {
                        const bool launched = dispatchFp8KRVectorRowBytes(shape.row_bytes, [&]<int RowBytes>() {
                            fp8KRValueVectorKernel<InputT, FixedOp, FixedInputTransform, RowBytes>
                                <<<launch.blocks, Fp8KR::FP8_KR_THREADS, 0, stream.getStream()>>>(
                                    static_cast<const uint4*>(input.getMemPtr<void>()), shape.rows, fold, out,
                                    out_dtype, output_finalize, scale);
                        });
                        if (!launched)
                            throw std::logic_error("Retained FP8 KR vector Pack has no kernel for this row length.");
                        break;
                    }
                } else if (shape.vector) {
                    throw std::logic_error("Retained FP32 KR has no vector Pack.");
                }
                fp8KRValuePackKernel<InputT, FixedOp, FixedInputTransform>
                    <<<launch.blocks, Fp8KR::FP8_KR_THREADS, 0, stream.getStream()>>>(
                        launch.words, launch.head, shape.rows, static_cast<uint32_t>(shape.row_bytes),
                        shape.rows_per_thread, fold, out, out_dtype, output_finalize, scale);
                break;
            case Fp8KR::Fp8KRFamily::Group:
                fp8KRValueGroupKernel<InputT, FixedOp, FixedInputTransform>
                    <<<launch.blocks, Fp8KR::FP8_KR_THREADS, 0, stream.getStream()>>>(
                        launch.words, launch.head, shape.rows, shape.row_bytes, launch.group_shift, fold, out,
                        out_dtype, output_finalize, scale);
                break;
            case Fp8KR::Fp8KRFamily::Block:
                fp8KRValueBlockKernel<InputT, FixedOp, FixedInputTransform>
                    <<<launch.blocks, Fp8KR::FP8_KR_THREADS, 0, stream.getStream()>>>(
                        launch.words, launch.head, shape.rows, shape.row_bytes, fold, out, out_dtype,
                        output_finalize, scale);
                break;
            case Fp8KR::Fp8KRFamily::Split: {
                auto* partials = static_cast<float*>(scratch);
                fp8KRValueSplitKernel<InputT, FixedOp, FixedInputTransform>
                    <<<launch.blocks, Fp8KR::FP8_KR_THREADS, 0, stream.getStream()>>>(
                        launch.words, launch.head, shape.rows, shape.row_bytes, shape.chunk_bytes, shape.chunks(),
                        fold, partials);
                CUDA_CHECK(cudaGetLastError());
                if (stages == Fp8KR::Fp8KRSplitStages::PartialsOnly) break;
                fp8KRValueFinishKernel<FixedOp><<<fp8KRFinishBlocks(shape.rows), Fp8KR::FP8_KR_THREADS, 0,
                                                  stream.getStream()>>>(partials, shape.rows, shape.chunks(), fold,
                                                                        out, out_dtype, output_finalize, scale);
                break;
            }
        }
        CUDA_CHECK(cudaGetLastError());
    };
    const auto by_semantics = [&]() {
        switch (semantics.combine) {
            case CubReductionStageCombine::Sum:
                switch (semantics.input_transform) {
                    case CubReductionStageInputTransform::Identity:
                        run.template operator()<0, 0>(FP8_EXTREMUM_FOLD_NONE, output_scale);
                        return;
                    case CubReductionStageInputTransform::AbsoluteValue:
                        run.template operator()<0, 1>(FP8_EXTREMUM_FOLD_NONE, output_scale);
                        return;
                    case CubReductionStageInputTransform::Square:
                        run.template operator()<0, 2>(FP8_EXTREMUM_FOLD_NONE, output_scale);
                        return;
                }
                break;
            case CubReductionStageCombine::Product:
                run.template operator()<1, 0>(FP8_EXTREMUM_FOLD_NONE, output_scale);
                return;
            case CubReductionStageCombine::Minimum:
            case CubReductionStageCombine::Maximum: {
                // RK's shared MIN kernel: the transform and MAX's sign fold into the bytes, MAX negates the scale.
                float scale = output_scale;
                const int fixed_op = semantics.combine == CubReductionStageCombine::Maximum ? 3 : 2;
                const Fp8ExtremumFold fold =
                    std::is_same_v<InputT, float>
                        ? fp32CompleteExtremumFold(fixed_op, semantics.input_transform, scale)
                        : fp8CompleteExtremumFold(fixed_op, semantics.input_transform, scale);
                run.template operator()<2, 0>(fold, scale);
                return;
            }
        }
        throw std::logic_error("Retained FP8 KR VALUE pass received unsupported semantics.");
    };
    by_semantics();
}

}  // namespace
#endif  // THOR_CUB_ENABLE_FP8_TYPES

bool fp8KRValueSupports(const CubReductionStageSemantics& semantics) {
    switch (semantics.combine) {
        case CubReductionStageCombine::Sum:
        case CubReductionStageCombine::Minimum:
        case CubReductionStageCombine::Maximum:
            return semantics.input_transform == CubReductionStageInputTransform::Identity ||
                   semantics.input_transform == CubReductionStageInputTransform::AbsoluteValue ||
                   semantics.input_transform == CubReductionStageInputTransform::Square;
        case CubReductionStageCombine::Product:
            return semantics.input_transform == CubReductionStageInputTransform::Identity;
    }
    return false;
}

void launchFp8KRValueSegments(const CubReductionStageSemantics& semantics,
                              const Tensor& input,
                              Tensor& output,
                              const Fp8KR::Fp8KRSegmentShape& shape,
                              void* scratch,
                              uint64_t scratch_bytes,
                              float output_scale,
                              Stream& stream,
                              Fp8KR::Fp8KRSplitStages stages) {
#if THOR_CUB_ENABLE_FP8_TYPES
    requireFp8KRInput(input, shape, scratch, scratch_bytes, false);
    if (output.getTotalNumElements() != shape.rows || !fp8CompleteSupportsOutputStorage(output.getDataType()) ||
        !fp8KRValueSupports(semantics)) {
        throw std::logic_error("Retained FP8 KR VALUE pass received an unsupported output or semantics.");
    }
    if (input.getDataType() == DataType::FP8_E4M3) {
        launchRetainedKRValueTyped<__nv_fp8_e4m3>(semantics, input, output, shape, scratch, output_scale, stream,
                                                  stages);
    } else {
        launchRetainedKRValueTyped<__nv_fp8_e5m2>(semantics, input, output, shape, scratch, output_scale, stream,
                                                  stages);
    }
#else
    static_cast<void>(semantics);
    static_cast<void>(input);
    static_cast<void>(output);
    static_cast<void>(shape);
    static_cast<void>(scratch);
    static_cast<void>(scratch_bytes);
    static_cast<void>(output_scale);
    static_cast<void>(stream);
    static_cast<void>(stages);
    throw std::logic_error("Retained FP8 KR VALUE requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}

void launchFp32KRValueSegments(const CubReductionStageSemantics& semantics,
                               const Tensor& input,
                               Tensor& output,
                               const Fp8KR::Fp8KRSegmentShape& shape,
                               void* scratch,
                               uint64_t scratch_bytes,
                               float output_scale,
                               Stream& stream,
                               Fp8KR::Fp8KRSplitStages stages) {
#if THOR_CUB_ENABLE_FP8_TYPES
    // 1179: the FP8 kernels at one element per word. Rows are whole elements on a 4-byte aligned input (head 0), so
    // no word holds bytes of two rows and the FP32 branches need no masks.
    if (input.getDataType() != DataType::FP32 || !Fp8KR::retainedKRShapeLegalForElement(shape, 4) ||
        input.getTotalNumElements() * 4 != shape.rows * shape.row_bytes ||
        (reinterpret_cast<uintptr_t>(input.getMemPtr<void>()) & uintptr_t{3}) != 0) {
        throw std::logic_error("Retained FP32 KR pass received an unsupported input or segment shape.");
    }
    if (shape.family() == Fp8KR::Fp8KRFamily::Split &&
        (scratch == nullptr || scratch_bytes < Fp8KR::fp8KRScratchBytes(shape, false) ||
         (reinterpret_cast<uintptr_t>(scratch) & uintptr_t{15}) != 0)) {
        throw std::logic_error("Retained FP32 KR Split pass requires its 16-byte aligned partial scratch.");
    }
    if (output.getTotalNumElements() != shape.rows || !fp8CompleteSupportsOutputStorage(output.getDataType()) ||
        !fp8KRValueSupports(semantics)) {
        throw std::logic_error("Retained FP32 KR VALUE pass received an unsupported output or semantics.");
    }
    launchRetainedKRValueTyped<float>(semantics, input, output, shape, scratch, output_scale, stream, stages);
#else
    static_cast<void>(semantics);
    static_cast<void>(input);
    static_cast<void>(output);
    static_cast<void>(shape);
    static_cast<void>(scratch);
    static_cast<void>(scratch_bytes);
    static_cast<void>(output_scale);
    static_cast<void>(stream);
    static_cast<void>(stages);
    throw std::logic_error("Retained FP32 KR VALUE requires THOR_CUB_ENABLE_FP8_TYPES (the retained kernels' gate).");
#endif
}

void launchFp8KRArgSegments(const Tensor& input,
                            Tensor* value_output,
                            Tensor* index_output,
                            const Fp8KR::Fp8KRSegmentShape& shape,
                            void* scratch,
                            uint64_t scratch_bytes,
                            bool maximum,
                            uint64_t domain_stride,
                            Stream& stream,
                            Fp8KR::Fp8KRSplitStages stages) {
#if THOR_CUB_ENABLE_FP8_TYPES
    requireFp8KRInput(input, shape, scratch, scratch_bytes, true);
    const uint64_t outputs = shape.rows;
    if (shape.row_bytes - 1 > FP8_ARG_MAX_WIDE_INDEX || (value_output == nullptr && index_output == nullptr) ||
        (value_output != nullptr && (value_output->getTotalNumElements() != outputs ||
                                     !fp8CompleteSupportsOutputStorage(value_output->getDataType()))) ||
        (index_output != nullptr && (index_output->getTotalNumElements() != outputs ||
                                     (index_output->getDataType() != DataType::UINT32 &&
                                      index_output->getDataType() != DataType::UINT64)))) {
        throw std::logic_error("Retained FP8 KR ARG pass received unsupported geometry or outputs.");
    }
    if (index_output != nullptr && index_output->getDataType() == DataType::UINT32 && domain_stride != 0 &&
        shape.row_bytes - 1 > std::numeric_limits<uint32_t>::max() / domain_stride) {
        throw std::logic_error("Retained FP8 KR ARG UINT32 index output cannot hold the original-domain index.");
    }
    const DataType dtype = input.getDataType();
    const Fp8ArgFold fold = fp8ArgFold(maximum, dtype == DataType::FP8_E5M2);
    const Fp8ArgOutputs arg_outputs{
        value_output == nullptr ? nullptr : value_output->getMemPtr<void>(),
        value_output == nullptr ? DataType::FP32 : value_output->getDataType(),
        index_output == nullptr ? nullptr : index_output->getMemPtr<void>(),
        index_output == nullptr ? DataType::UINT32 : index_output->getDataType(),
        domain_stride,
        fold,
        dtype == DataType::FP8_E5M2};
    const Fp8KRLaunchGeometry launch = fp8KRLaunchGeometry(input, shape);
    switch (shape.family()) {
        case Fp8KR::Fp8KRFamily::Pack:
            if (shape.vector && (reinterpret_cast<uintptr_t>(input.getMemPtr<void>()) & uintptr_t{15}) == 0) {
                const bool launched = dispatchFp8KRVectorRowBytes(shape.row_bytes, [&]<int RowBytes>() {
                    fp8KRArgVectorKernel<RowBytes><<<launch.blocks, Fp8KR::FP8_KR_THREADS, 0, stream.getStream()>>>(
                        static_cast<const uint4*>(input.getMemPtr<void>()), shape.rows, fold, arg_outputs);
                });
                if (!launched) throw std::logic_error("Retained FP8 KR vector Pack has no kernel for this row length.");
                break;
            }
            fp8KRArgPackKernel<<<launch.blocks, Fp8KR::FP8_KR_THREADS, 0, stream.getStream()>>>(
                launch.words, launch.head, shape.rows, static_cast<uint32_t>(shape.row_bytes), shape.rows_per_thread,
                fold, arg_outputs);
            break;
        case Fp8KR::Fp8KRFamily::Group:
            fp8KRArgGroupKernel<<<launch.blocks, Fp8KR::FP8_KR_THREADS, 0, stream.getStream()>>>(
                launch.words, launch.head, shape.rows, shape.row_bytes, launch.group_shift, fold, arg_outputs);
            break;
        case Fp8KR::Fp8KRFamily::Block:
            fp8KRArgBlockKernel<<<launch.blocks, Fp8KR::FP8_KR_THREADS, 0, stream.getStream()>>>(
                launch.words, launch.head, shape.rows, shape.row_bytes, fold, arg_outputs);
            break;
        case Fp8KR::Fp8KRFamily::Split: {
            auto* partials = static_cast<uint64_t*>(scratch);
            fp8KRArgSplitKernel<<<launch.blocks, Fp8KR::FP8_KR_THREADS, 0, stream.getStream()>>>(
                launch.words, launch.head, shape.rows, shape.row_bytes, shape.chunk_bytes, shape.chunks(), fold,
                partials);
            CUDA_CHECK(cudaGetLastError());
            if (stages == Fp8KR::Fp8KRSplitStages::PartialsOnly) break;
            fp8KRArgFinishKernel<<<fp8KRFinishBlocks(shape.rows), Fp8KR::FP8_KR_THREADS, 0, stream.getStream()>>>(
                partials, shape.rows, shape.chunks(), arg_outputs);
            break;
        }
    }
    CUDA_CHECK(cudaGetLastError());
#else
    static_cast<void>(input);
    static_cast<void>(value_output);
    static_cast<void>(index_output);
    static_cast<void>(shape);
    static_cast<void>(scratch);
    static_cast<void>(scratch_bytes);
    static_cast<void>(maximum);
    static_cast<void>(domain_stride);
    static_cast<void>(stream);
    static_cast<void>(stages);
    throw std::logic_error("Retained FP8 KR ARG requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}

void launchFp32KRArgSegments(const Tensor& input,
                             Tensor* value_output,
                             Tensor* index_output,
                             const Fp8KR::Fp8KRSegmentShape& shape,
                             void* scratch,
                             uint64_t scratch_bytes,
                             bool maximum,
                             uint64_t domain_stride,
                             Stream& stream,
                             Fp8KR::Fp8KRSplitStages stages) {
#if THOR_CUB_ENABLE_FP8_TYPES
    // 1181: whole-element rows on a 4-byte aligned input, as the FP32 VALUE launcher.
    if (input.getDataType() != DataType::FP32 || !Fp8KR::fp32KRArgShapeLegal(shape) ||
        input.getTotalNumElements() * 4 != shape.rows * shape.row_bytes ||
        (reinterpret_cast<uintptr_t>(input.getMemPtr<void>()) & uintptr_t{3}) != 0) {
        throw std::logic_error("Retained FP32 KR ARG pass received an unsupported input or segment shape.");
    }
    if (shape.family() == Fp8KR::Fp8KRFamily::Split &&
        (scratch == nullptr || scratch_bytes < Fp8KR::fp8KRScratchBytes(shape, true) ||
         (reinterpret_cast<uintptr_t>(scratch) & uintptr_t{15}) != 0)) {
        throw std::logic_error("Retained FP32 KR ARG Split pass requires its 16-byte aligned partial scratch.");
    }
    const uint64_t outputs = shape.rows;
    const uint64_t row_words = shape.row_bytes / 4;
    if ((value_output == nullptr && index_output == nullptr) ||
        (value_output != nullptr && (value_output->getTotalNumElements() != outputs ||
                                     !fp8CompleteSupportsOutputStorage(value_output->getDataType()))) ||
        (index_output != nullptr && (index_output->getTotalNumElements() != outputs ||
                                     (index_output->getDataType() != DataType::UINT32 &&
                                      index_output->getDataType() != DataType::UINT64)))) {
        throw std::logic_error("Retained FP32 KR ARG pass received unsupported outputs.");
    }
    if (index_output != nullptr && domain_stride != 0 &&
        row_words - 1 > (index_output->getDataType() == DataType::UINT32 ? std::numeric_limits<uint32_t>::max()
                                                                          : std::numeric_limits<uint64_t>::max()) /
                            domain_stride) {
        throw std::logic_error("Retained FP32 KR ARG index output cannot hold the original-domain index.");
    }
    const Fp32ArgOutputs arg_outputs{
        value_output == nullptr ? nullptr : value_output->getMemPtr<void>(),
        value_output == nullptr ? DataType::FP32 : value_output->getDataType(),
        index_output == nullptr ? nullptr : index_output->getMemPtr<void>(),
        index_output == nullptr ? DataType::UINT32 : index_output->getDataType(),
        domain_stride,
        maximum ? 0x80000000U : 0U};
    const Fp8KRLaunchGeometry launch = fp8KRLaunchGeometry(input, shape);
    switch (shape.family()) {
        case Fp8KR::Fp8KRFamily::Pack:
            fp32KRArgPackKernel<<<launch.blocks, Fp8KR::FP8_KR_THREADS, 0, stream.getStream()>>>(
                launch.words, shape.rows, static_cast<uint32_t>(row_words), shape.rows_per_thread, arg_outputs);
            break;
        case Fp8KR::Fp8KRFamily::Group:
            fp32KRArgGroupKernel<<<launch.blocks, Fp8KR::FP8_KR_THREADS, 0, stream.getStream()>>>(
                launch.words, shape.rows, row_words, launch.group_shift, arg_outputs);
            break;
        case Fp8KR::Fp8KRFamily::Block:
            fp32KRArgBlockKernel<<<launch.blocks, Fp8KR::FP8_KR_THREADS, 0, stream.getStream()>>>(
                launch.words, shape.rows, row_words, arg_outputs);
            break;
        case Fp8KR::Fp8KRFamily::Split: {
            auto* partials = static_cast<uint64_t*>(scratch);
            const uint64_t chunk_words = shape.chunk_bytes / 4;
            fp32KRArgSplitKernel<<<launch.blocks, Fp8KR::FP8_KR_THREADS, 0, stream.getStream()>>>(
                launch.words, shape.rows, row_words, chunk_words, shape.chunks(), arg_outputs.xor_mask, partials);
            CUDA_CHECK(cudaGetLastError());
            if (stages == Fp8KR::Fp8KRSplitStages::PartialsOnly) break;
            fp32KRArgFinishKernel<<<fp8KRFinishBlocks(shape.rows), Fp8KR::FP8_KR_THREADS, 0, stream.getStream()>>>(
                partials, shape.rows, shape.chunks(), chunk_words, arg_outputs);
            break;
        }
    }
    CUDA_CHECK(cudaGetLastError());
#else
    static_cast<void>(input);
    static_cast<void>(value_output);
    static_cast<void>(index_output);
    static_cast<void>(shape);
    static_cast<void>(scratch);
    static_cast<void>(scratch_bytes);
    static_cast<void>(maximum);
    static_cast<void>(domain_stride);
    static_cast<void>(stream);
    static_cast<void>(stages);
    throw std::logic_error("Retained FP32 KR ARG requires THOR_CUB_ENABLE_FP8_TYPES (the retained kernels' gate).");
#endif
}

uint64_t fp32KRCarriedArgScratchBytes(const Fp8KR::Fp8KRSegmentShape& shape) {
    return shape.family() == Fp8KR::Fp8KRFamily::Split ? fp32ArgWideScratchBytes(shape.rows, shape.chunks(), 1) : 0;
}

void launchFp32KRCarriedArgSegments(const Tensor& input,
                                    const Tensor& carried_index,
                                    Tensor* value_output,
                                    Tensor* index_output,
                                    const Fp8KR::Fp8KRSegmentShape& shape,
                                    void* scratch,
                                    uint64_t scratch_bytes,
                                    bool maximum,
                                    uint64_t row_stride,
                                    Stream& stream,
                                    Fp8KR::Fp8KRSplitStages stages) {
#if THOR_CUB_ENABLE_FP8_TYPES
    const uint64_t elements = shape.rows * (shape.row_bytes / 4);
    if (input.getDataType() != DataType::FP32 || !Fp8KR::fp32KRArgShapeLegal(shape) ||
        input.getTotalNumElements() * 4 != shape.rows * shape.row_bytes ||
        (reinterpret_cast<uintptr_t>(input.getMemPtr<void>()) & uintptr_t{3}) != 0) {
        throw std::logic_error("Retained FP32 KR carried ARG pass received an unsupported input or segment shape.");
    }
    const DataType index_dtype = carried_index.getDataType();
    const uintptr_t index_alignment = index_dtype == DataType::UINT64 ? 8 : 4;
    if ((index_dtype != DataType::UINT32 && index_dtype != DataType::UINT64) ||
        carried_index.getTotalNumElements() != elements ||
        (reinterpret_cast<uintptr_t>(carried_index.getMemPtr<void>()) & (index_alignment - 1)) != 0) {
        throw std::logic_error("Retained FP32 KR carried ARG pass received an unsupported carried index.");
    }
    if (shape.family() == Fp8KR::Fp8KRFamily::Split &&
        (scratch == nullptr || scratch_bytes < fp32KRCarriedArgScratchBytes(shape) ||
         (reinterpret_cast<uintptr_t>(scratch) & uintptr_t{255}) != 0)) {
        throw std::logic_error("Retained FP32 KR carried ARG Split pass requires its 256-byte aligned wide scratch.");
    }
    const uint64_t outputs = shape.rows;
    const uint64_t row_words = shape.row_bytes / 4;
    if ((value_output == nullptr && index_output == nullptr) ||
        (value_output != nullptr && (value_output->getTotalNumElements() != outputs ||
                                     !fp8CompleteSupportsOutputStorage(value_output->getDataType()))) ||
        (index_output != nullptr && (index_output->getTotalNumElements() != outputs ||
                                     (index_output->getDataType() != DataType::UINT32 &&
                                      index_output->getDataType() != DataType::UINT64)))) {
        throw std::logic_error("Retained FP32 KR carried ARG pass received unsupported outputs.");
    }
    // The tails hold the composed index: the index stride is 1. A UINT32 index output is the caller's choice (the
    // planner sizes it from the whole original domain).
    const Fp32ArgOutputs arg_outputs{
        value_output == nullptr ? nullptr : value_output->getMemPtr<void>(),
        value_output == nullptr ? DataType::FP32 : value_output->getDataType(),
        index_output == nullptr ? nullptr : index_output->getMemPtr<void>(),
        index_output == nullptr ? DataType::UINT32 : index_output->getDataType(),
        1,
        maximum ? 0x80000000U : 0U};
    const Fp8KRLaunchGeometry launch = fp8KRLaunchGeometry(input, shape);
    const auto run = [&]<typename IndexT>() {
        const IndexT* carried = carried_index.getMemPtr<IndexT>();
        switch (shape.family()) {
            case Fp8KR::Fp8KRFamily::Pack:
                fp32KRCarriedArgPackKernel<IndexT><<<launch.blocks, Fp8KR::FP8_KR_THREADS, 0, stream.getStream()>>>(
                    launch.words, carried, shape.rows, static_cast<uint32_t>(row_words), shape.rows_per_thread,
                    row_stride, arg_outputs);
                break;
            case Fp8KR::Fp8KRFamily::Group:
                fp32KRCarriedArgGroupKernel<IndexT><<<launch.blocks, Fp8KR::FP8_KR_THREADS, 0, stream.getStream()>>>(
                    launch.words, carried, shape.rows, row_words, launch.group_shift, row_stride, arg_outputs);
                break;
            case Fp8KR::Fp8KRFamily::Block:
                fp32KRCarriedArgBlockKernel<IndexT><<<launch.blocks, Fp8KR::FP8_KR_THREADS, 0, stream.getStream()>>>(
                    launch.words, carried, shape.rows, row_words, row_stride, arg_outputs);
                break;
            case Fp8KR::Fp8KRFamily::Split: {
                const uint64_t chunks = shape.chunks();
                auto* bytes = static_cast<uint8_t*>(scratch);
                const uint64_t tail_offset = (shape.segments() * sizeof(uint32_t) + 255) & ~uint64_t{255};
                const Fp32ArgWidePartials partials{reinterpret_cast<uint32_t*>(bytes),
                                                   reinterpret_cast<uint64_t*>(bytes + tail_offset)};
                fp32KRCarriedArgSplitKernel<IndexT><<<launch.blocks, Fp8KR::FP8_KR_THREADS, 0, stream.getStream()>>>(
                    launch.words, carried, shape.rows, row_words, shape.chunk_bytes / 4, chunks, arg_outputs.xor_mask,
                    row_stride, partials);
                CUDA_CHECK(cudaGetLastError());
                if (stages == Fp8KR::Fp8KRSplitStages::PartialsOnly) break;
                fp32KRCarriedArgFinishKernel<<<fp8KRFinishBlocks(shape.rows), Fp8KR::FP8_KR_THREADS, 0,
                                               stream.getStream()>>>(partials, shape.rows, chunks, arg_outputs);
                break;
            }
        }
        CUDA_CHECK(cudaGetLastError());
    };
    if (index_dtype == DataType::UINT64) run.template operator()<uint64_t>();
    else run.template operator()<uint32_t>();
#else
    static_cast<void>(input);
    static_cast<void>(carried_index);
    static_cast<void>(value_output);
    static_cast<void>(index_output);
    static_cast<void>(shape);
    static_cast<void>(scratch);
    static_cast<void>(scratch_bytes);
    static_cast<void>(maximum);
    static_cast<void>(row_stride);
    static_cast<void>(stream);
    static_cast<void>(stages);
    throw std::logic_error(
        "Retained FP32 KR carried ARG requires THOR_CUB_ENABLE_FP8_TYPES (the retained kernels' gate).");
#endif
}

}  // namespace ThorImplementation::CubReductionInternal
