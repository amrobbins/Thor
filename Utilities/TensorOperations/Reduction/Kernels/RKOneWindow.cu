#include "Utilities/TensorOperations/Reduction/Kernels/KernelLaunchers.h"
#include "Utilities/TensorOperations/Reduction/Kernels/ReductionOperation.cuh"
#include "Utilities/TensorOperations/Reduction/Arg/Fp8ArgKey.h"
#include "Utilities/TensorOperations/Reduction/Arg/Fp32ArgKey.cuh"
#include "Utilities/TensorOperations/Reduction/Kernels/KernelCommon.cuh"

#include "Utilities/Exceptions.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace ThorImplementation::ReductionInternal {
namespace {

[[nodiscard]] inline bool shallowFp8FitsUint32Indexing(const ReductionGeometry& geometry) {
    constexpr uint64_t MaxIndex = std::numeric_limits<uint32_t>::max();
    return geometry.outer_size <= MaxIndex && geometry.reduction_size <= MaxIndex && geometry.input_elements <= MaxIndex &&
           geometry.output_elements <= MaxIndex;
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

// OneWindow: one window of packets per row group, with K as a runtime operand. Packet width and ownership (outputs per
// warp) stay compile-time axes, and K is bucketed by the maximum number of components per lane (1, 2, 4, 8, 16), so
// local[] and the slot loop stay compile-time (unused slots are skipped by the owned_component < K guard). Everything
// compile-time K provided is computed once on the host (OneWindowGeometry): rows per staged group, R workers
// per component and the accumulating lane count. The group loop is either plain or load-ahead (LoadAhead), fixed per
// unit. Decode packing differs by bucket (decodeOneWindowGroup); everything is force-inlined functions, no
// lambdas holding local[].
struct OneWindowGeometry {
    uint32_t k;                      // components per output
    uint32_t rows_per_group;         // logical rows staged per packet window
    uint32_t workers_per_component;  // R workers per component (1 when a lane owns several components)
    uint32_t task_count;             // accumulating lanes per output
};

// ElementBytes 4 for FP32 input (a row is k elements of 4 bytes); FP8 is ElementBytes 1.
template <uint32_t PacketBytes, uint32_t ElementBytes = 1>
__device__ __forceinline__ OneWindowGroupWindow oneWindowGroupWindow(bool valid_output,
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

// One staged FP32 element, MIN / MAX folded and the SUM-family transform applied. Staging is plane-major: element e of
// the window, counted from the first staged packet, is word e % PacketWords of the subgroup's packet e / PacketWords.
template <int FixedOp, uint32_t PacketBytes>
__device__ __forceinline__ float oneWindowStagedFp32(const uint32_t* warp_words,
                                                     uint32_t subgroup_lane_base,
                                                     uint32_t element,
                                                     ReductionStageInputTransform input_transform,
                                                     const Fp8ExtremumFold& extremum_fold) {
    constexpr uint32_t PacketWords = PacketBytes / 4U;
    const uint32_t word =
        warp_words[(element % PacketWords) * TILED_REDUCTION_WARP_THREADS + subgroup_lane_base + element / PacketWords];
    float value = __uint_as_float(foldFp8ExtremumBytes<FixedOp == 2>(word, extremum_fold));
    if constexpr (FixedOp != 2) {
        if (input_transform == ReductionStageInputTransform::AbsoluteValue) {
            value = ::fabsf(value);
        } else if (input_transform == ReductionStageInputTransform::Square) {
            value *= value;
        }
    } else {
        static_cast<void>(input_transform);
    }
    return value;
}

// One staged FP16 / BF16 element, as oneWindowStagedFp32: element e of the window is half (e % 2) of word
// (e % PacketHalves) / 2 of the subgroup's packet e / PacketHalves. MIN / MAX fold both half-words of the word
// (fp16CompleteExtremumFold).
template <typename InputT, int FixedOp, uint32_t PacketBytes>
__device__ __forceinline__ float oneWindowStagedF16(const uint32_t* warp_words,
                                                    uint32_t subgroup_lane_base,
                                                    uint32_t element,
                                                    ReductionStageInputTransform input_transform,
                                                    const Fp8ExtremumFold& extremum_fold) {
    constexpr uint32_t PacketHalves = PacketBytes / 2U;
    const uint32_t within = element % PacketHalves;
    const uint32_t word =
        warp_words[(within >> 1) * TILED_REDUCTION_WARP_THREADS + subgroup_lane_base + element / PacketHalves];
    const uint32_t folded = foldFp8ExtremumBytes<FixedOp == 2>(word, extremum_fold);
    float value = widenF16Bits<InputT>((within & 1U) != 0 ? folded >> 16 : folded);
    if constexpr (FixedOp != 2) {
        if (input_transform == ReductionStageInputTransform::AbsoluteValue) {
            value = ::fabsf(value);
        } else if (input_transform == ReductionStageInputTransform::Square) {
            value *= value;
        }
    } else {
        static_cast<void>(input_transform);
    }
    return value;
}

// The staged element of a per-element input (FP32, FP16 / BF16).
template <typename InputT, int FixedOp, uint32_t PacketBytes>
__device__ __forceinline__ float oneWindowStagedElement(const uint32_t* warp_words,
                                                        uint32_t subgroup_lane_base,
                                                        uint32_t element,
                                                        ReductionStageInputTransform input_transform,
                                                        const Fp8ExtremumFold& extremum_fold) {
    if constexpr (std::is_same_v<InputT, float>) {
        return oneWindowStagedFp32<FixedOp, PacketBytes>(warp_words, subgroup_lane_base, element, input_transform,
                                                         extremum_fold);
    } else {
        return oneWindowStagedF16<InputT, FixedOp, PacketBytes>(warp_words, subgroup_lane_base, element,
                                                                input_transform, extremum_fold);
    }
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
__device__ __forceinline__ float4 decodeOneWindowPacked(uint32_t packed,
                                                        ReductionStageInputTransform input_transform,
                                                        const Fp8ExtremumFold& extremum_fold) {
    return transformFp8CompleteDecodedPacket<FixedOp == 2 ? 0 : -1>(
        decodeFp8x4ToFp32<InputT>(foldFp8ExtremumBytes<FixedOp == 2>(packed, extremum_fold)), input_transform);
}

// Decodes one staged group into local[]. Rows of each component are accumulated in increasing order.
//  - Buckets 1 and 2 pack four rows of one component per FP8x4 decode, with
//    unpredicated full steps and one predicated tail.
//  - Buckets 4..16 pack four of the lane's components of one row per decode. Large K leaves few rows per group
//    (one row at K 497), where packing rows would decode one valid value per conversion; this packs four, with the
//    same byte reads.
template <typename InputT, int FixedOp, int MaxComponentsPerLane, uint32_t LanesPerOutput, uint32_t PacketBytes>
__device__ __forceinline__ void decodeOneWindowGroup(float (&local)[MaxComponentsPerLane],
                                                     const uint32_t* warp_words,
                                                     uint32_t subgroup_lane,
                                                     uint32_t subgroup_lane_base,
                                                     uint32_t component,
                                                     uint32_t worker,
                                                     uint32_t k,
                                                     uint32_t workers,
                                                     uint32_t group_rows,
                                                     uint32_t prefix_bytes,
                                                     ReductionStageInputTransform input_transform,
                                                     const Fp8ExtremumFold& extremum_fold) {
    if constexpr (std::is_same_v<InputT, float> || isF16InputType<InputT>) {
        // FP32, one element per staged word, rows of each component accumulated in increasing order. Buckets 1 and 2: a
        // lane walks its component's rows with stride `workers`; buckets 4 (the largest FP32 reaches: K <= 125 at O1)
        // walk the rows and take the lane's components of each. FP16 / BF16 the same way, element by element
        // (buckets <= 8: K <= 249 at O1).
        const Fp8TypedReductionOp<FixedOp> combine{};
        const uint32_t first_element = prefix_bytes / static_cast<uint32_t>(sizeof(InputT));
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
                    local[slot] = combine(local[slot], oneWindowStagedElement<InputT, FixedOp, PacketBytes>(
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
                        local[slot] = combine(local[slot], oneWindowStagedElement<InputT, FixedOp, PacketBytes>(
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
                    local[slot], decodeOneWindowPacked<InputT, FixedOp>(packed, input_transform, extremum_fold),
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
                    local[slot], decodeOneWindowPacked<InputT, FixedOp>(packed, input_transform, extremum_fold),
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
                    decodeOneWindowPacked<InputT, FixedOp>(packed, input_transform, extremum_fold);
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
__device__ __forceinline__ void streamOneWindowOutput(float (&local)[MaxComponentsPerLane],
                                                      uint32_t* warp_words,
                                                      const uint8_t* input_bytes,
                                                      uint32_t total_input_bytes,
                                                      bool valid_output,
                                                      uint32_t outer_byte_begin,
                                                      uint32_t first_row,
                                                      uint32_t row_step,
                                                      uint32_t reduction_size,
                                                      const OneWindowGeometry& geometry,
                                                      uint32_t lane,
                                                      uint32_t subgroup_lane,
                                                              uint32_t subgroup_lane_base,
                                                              uint32_t component,
                                                              uint32_t worker,
                                                              ReductionStageInputTransform input_transform,
                                                              const Fp8ExtremumFold& extremum_fold) {
    const bool decodes = valid_output && subgroup_lane < geometry.task_count;
    const auto window_at = [&](uint32_t group_row) {
        return oneWindowGroupWindow<PacketBytes, static_cast<uint32_t>(sizeof(InputT))>(
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
                decodeOneWindowGroup<InputT, FixedOp, MaxComponentsPerLane, LanesPerOutput, PacketBytes>(
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
                decodeOneWindowGroup<InputT, FixedOp, MaxComponentsPerLane, LanesPerOutput, PacketBytes>(
                    local, warp_words, subgroup_lane, subgroup_lane_base, component, worker, geometry.k,
                    geometry.workers_per_component, window.rows, window.prefix_bytes, input_transform, extremum_fold);
            }
            __syncwarp();
        }
    }
}

// One-component buckets: the R workers of a component sit in consecutive lanes (component * workers + worker).
// A power-of-two worker count (always at O4/O8; at O1/O2 when lanes / K is a power of two, including one worker) folds
// with butterfly shuffles inside the aligned worker group, with no shared memory and no warp barrier; every lane of the
// group ends with the total. Other counts keep the shared lane_partials fold. Warp-uniform: every lane calls it with
// the same workers.
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
__global__ void oneWindowValueKernel(const InputT* input,
                                     void* output,
                                     DataType output_dtype,
                                     uint32_t outer_size,
                                     uint32_t reduction_size,
                                     uint32_t warps_per_output,
                                     OneWindowGeometry geometry,
                                     AdditiveFinalizeFp32 output_finalize,
                                     ReductionStageInputTransform input_transform,
                                     float output_scale,
                                     Fp8ExtremumFold extremum_fold) {
    // FP32: buckets <= 4; FP16 / BF16: buckets <= 8.
    static_assert(sizeof(InputT) == 1 || std::is_same_v<InputT, float> || isF16InputType<InputT>);
    static_assert(sizeof(InputT) == 1 || MaxComponentsPerLane <= 4 ||
                  (isF16InputType<InputT> && MaxComponentsPerLane <= 8));
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
    constexpr uint32_t WarpPartialRows = OutputsPerWarp == 1 ? SHALLOW_COMPLETE_WARPS : 1U;
    constexpr uint32_t WarpPartialColumns = OutputsPerWarp == 1 ? MaxK : 1U;
    const Fp8TypedReductionOp<FixedOp> combine{};

    __shared__ uint32_t staged_packet_words[SHALLOW_COMPLETE_WARPS][PacketWords * TILED_REDUCTION_WARP_THREADS];
    __shared__ float lane_partials[SHALLOW_COMPLETE_WARPS][LanePartialColumns];
    __shared__ float warp_partials[WarpPartialRows][WarpPartialColumns];

    const uint32_t k = geometry.k;
    const uint32_t workers = geometry.workers_per_component;
    const uint32_t warp = threadIdx.x / TILED_REDUCTION_WARP_THREADS;
    const uint32_t lane = threadIdx.x % TILED_REDUCTION_WARP_THREADS;
    const uint32_t outer_input_stride = reduction_size * k * static_cast<uint32_t>(sizeof(InputT));  // bytes
    // Byte offsets count from the packet boundary at or below the base (base_shift bytes before it; 0 for an aligned
    // base), so a group's prefix and packet addresses follow the input's address and the kernel reads any element
    // address. The first packet starts inside the base's own 16-byte granule; the last is still assembled bytewise at
    // the input's end. The launcher's plan leaves the window room for the prefix (aligned_base false).
    const uint32_t base_shift =
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(input) & static_cast<uintptr_t>(PacketBytes - 1));
    const uint32_t total_input_bytes = base_shift + outer_size * outer_input_stride;
    const uint8_t* const input_bytes = reinterpret_cast<const uint8_t*>(input) - base_shift;

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
        const uint32_t global_warp = blockIdx.x * SHALLOW_COMPLETE_WARPS + warp;
        const uint32_t grid_warps = gridDim.x * SHALLOW_COMPLETE_WARPS;
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
                valid_output ? base_shift + static_cast<uint32_t>(outer_index) * outer_input_stride : 0;
            streamOneWindowOutput<InputT, FixedOp, MaxComponentsPerLane, LanesPerOutput, PacketBytes,
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
        const uint32_t outputs_per_block = SHALLOW_COMPLETE_WARPS / warps_per_output;
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
                valid_output ? base_shift + static_cast<uint32_t>(outer_index) * outer_input_stride : 0;
            streamOneWindowOutput<InputT, FixedOp, MaxComponentsPerLane, LanesPerOutput, PacketBytes,
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

[[nodiscard]] inline bool oneWindowValidWarpsPerOutput(uint32_t outputs_per_warp, uint32_t warps_per_output) {
    const bool valid_warps =
        warps_per_output == 1 || warps_per_output == 2 || warps_per_output == 4 || warps_per_output == 8;
    // O2/O4/O8 partition one warp among outputs and therefore require W1.
    return valid_warps && (outputs_per_warp == 1 || warps_per_output == 1);
}

// OneWindow host plan: the geometry of one launch at a runtime K, computed once on the host: the packet window of a
// lane subgroup, rows per staged group, R workers per component, accumulating lanes, and the components-per-lane bucket
// (next power of two, at most 16).
struct OneWindowPlan {
    OneWindowGeometry geometry;
    uint32_t max_components_per_lane;
};

// element_bytes 4 for FP32. The window and the worst-case prefix are counted in elements (a packet holds
// packet_bytes / element_bytes of them), so FP32 rows fit K <= 125 / 61 / 29 / 13 at O1 / O2 / O4 / O8 (P16);
// element_bytes 2 for FP16 / BF16: K <= 249 / 121 / 57 / 25 (and the 8-element multiples 256 / 128 / 64 / 32).
// aligned_base false is an input base off a packet boundary (the VALUE kernel reads it: its windows follow the
// address). Every row group then carries a prefix, even at K a multiple of the packet, so the capacity is always the
// window less the worst prefix (RK::oneWindowUnalignedCapacity).
[[nodiscard]] inline std::optional<OneWindowPlan> oneWindowPlan(uint64_t k,
                                                                uint32_t outputs_per_warp,
                                                                uint32_t packet_bytes,
                                                                uint32_t element_bytes = 1,
                                                                bool aligned_base = true) {
    if (k == 0 || (outputs_per_warp != 1 && outputs_per_warp != 2 && outputs_per_warp != 4 && outputs_per_warp != 8) ||
        (packet_bytes != 4 && packet_bytes != 8 && packet_bytes != 16) ||
        (element_bytes != 1 && element_bytes != 2 && element_bytes != 4)) {
        return std::nullopt;
    }
    const uint32_t lanes = TILED_REDUCTION_WARP_THREADS / outputs_per_warp;
    const uint32_t packet_elements = packet_bytes / element_bytes;
    const uint64_t window = static_cast<uint64_t>(lanes) * packet_elements;
    const uint64_t capacity =
        aligned_base && k % packet_elements == 0 ? window : window - (packet_elements - 1);
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
    return OneWindowPlan{{k32, static_cast<uint32_t>(capacity / k), workers, multiple ? lanes : k32 * workers},
                         bucket};
}

// The OneWindow inventory: 15 compiled units, each (outputs per warp, packet bytes, bucket) with exactly one group loop
// form, so a selector OneWindow variant (O, W, P16) maps to at most one kernel at a given K. Chosen by oracle pruning
// over focused and uncovered-K sweeps; the 8-byte packet (every ownership x bucket 1-8, both loops) measured a
// geometric mean of 1.0005 and was dropped.
// Every K <= 497, and K = 512, has a unit (oneWindowCoversK).
inline constexpr std::array<OneWindowUnit, 15> ONE_WINDOW_UNITS{{
    {1, 16, 1, false}, {1, 16, 2, true}, {1, 16, 4, false}, {1, 16, 8, true}, {1, 16, 16, true},
    {2, 16, 1, false}, {2, 16, 2, false}, {2, 16, 4, false}, {2, 16, 8, true},
    {4, 16, 2, false}, {4, 16, 4, false}, {4, 16, 8, true},
    {8, 16, 1, false}, {8, 16, 2, false}, {8, 16, 4, false},
}};

[[nodiscard]] constexpr bool oneWindowUnitCompiled(const OneWindowUnit& unit) {
    for (const OneWindowUnit& compiled : ONE_WINDOW_UNITS) {
        if (compiled == unit) {
            return true;
        }
    }
    return false;
}

// Invokes callback.template operator()<InputT, OutputsPerWarp, PacketBytes, MaxComponentsPerLane, LoadAhead>() for
// the compiled unit equal to `unit`. Returns false when that unit is not compiled.
template <typename Callback>
[[nodiscard]] bool dispatchOneWindow(DataType input_dtype,
                                     const OneWindowUnit& unit,
                                     Callback&& callback) {
    bool dispatched = false;
    const auto for_type = [&]<typename InputT>() {
        [&]<size_t... I>(std::index_sequence<I...>) {
            static_cast<void>(((ONE_WINDOW_UNITS[I] == unit &&
                                (callback.template operator()<
                                     InputT,
                                     static_cast<int>(ONE_WINDOW_UNITS[I].outputs_per_warp),
                                     static_cast<int>(ONE_WINDOW_UNITS[I].packet_bytes),
                                     static_cast<int>(ONE_WINDOW_UNITS[I].max_components_per_lane),
                                     ONE_WINDOW_UNITS[I].load_ahead>(),
                                 dispatched = true)) ||
                               ...));
        }(std::make_index_sequence<ONE_WINDOW_UNITS.size()>{});
    };
    switch (input_dtype) {
        case DataType::FP8_E4M3: for_type.template operator()<__nv_fp8_e4m3>(); break;
        case DataType::FP8_E5M2: for_type.template operator()<__nv_fp8_e5m2>(); break;
        default: break;
    }
    return dispatched;
}

// The plan for `unit` at K, when the unit is compiled and K falls in its bucket.
[[nodiscard]] std::optional<OneWindowPlan> oneWindowUnitPlan(uint64_t k,
                                                             const OneWindowUnit& unit,
                                                             uint32_t element_bytes = 1,
                                                             bool aligned_base = true) {
    const auto plan =
        oneWindowPlan(k, unit.outputs_per_warp, unit.packet_bytes, element_bytes, aligned_base);
    if (!plan || plan->max_components_per_lane != unit.max_components_per_lane ||
        !oneWindowUnitCompiled(unit)) {
        return std::nullopt;
    }
    return plan;
}

// The FP32 OneWindow inventory is FP8's units whose bucket FP32 can reach (components per lane <= 4: a 16-byte lane
// packet holds 4 FP32), the same group loop form per unit. Invokes callback.template operator()<float, OutputsPerWarp,
// PacketBytes, MaxComponentsPerLane, LoadAhead>() for the unit equal to `unit`; false when FP32 has no such unit. Only
// this dispatch instantiates FP32 OneWindow kernels.
[[nodiscard]] constexpr bool oneWindowFp32Unit(const OneWindowUnit& unit) {
    return unit.max_components_per_lane <= 4;
}

template <typename Callback>
[[nodiscard]] bool dispatchFp32OneWindow(const OneWindowUnit& unit, Callback&& callback) {
    bool dispatched = false;
    const auto try_unit = [&]<size_t I>() {
        constexpr OneWindowUnit compiled = ONE_WINDOW_UNITS[I];
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
    }(std::make_index_sequence<ONE_WINDOW_UNITS.size()>{});
    return dispatched;
}

[[nodiscard]] std::optional<OneWindowPlan> fp32OneWindowUnitPlan(uint64_t k,
                                                                 const OneWindowUnit& unit,
                                                                 bool aligned_base = true) {
    if (!oneWindowFp32Unit(unit)) return std::nullopt;
    return oneWindowUnitPlan(k, unit, 4, aligned_base);
}

// FP32 OneWindow indexes input bytes in 32 bits (the kernel's byte offsets), so the input holds at most 2^32 - 1
// bytes; outputs and rows as FP8.
[[nodiscard]] inline bool fp32OneWindowFitsIndexing(const ReductionGeometry& geometry) {
    constexpr uint64_t MaxIndex = std::numeric_limits<uint32_t>::max();
    return shallowFp8FitsUint32Indexing(geometry) && geometry.input_elements <= MaxIndex / 4;
}

// The FP16 / BF16 OneWindow inventory is FP8's units of buckets <= 8 (a 16-byte lane packet holds 8 2-byte elements; 14
// units), dispatched for the runtime format. Only this dispatch instantiates 2-byte OneWindow kernels.
[[nodiscard]] constexpr bool oneWindowF16Unit(const OneWindowUnit& unit) {
    return unit.max_components_per_lane <= 8;
}

template <typename Callback>
[[nodiscard]] bool dispatchF16OneWindow(DataType input_dtype, const OneWindowUnit& unit,
                                        Callback&& callback) {
    bool dispatched = false;
    const auto for_type = [&]<typename InputT>() {
        const auto try_unit = [&]<size_t I>() {
            constexpr OneWindowUnit compiled = ONE_WINDOW_UNITS[I];
            if constexpr (oneWindowF16Unit(compiled)) {
                if (!dispatched && compiled == unit) {
                    callback.template operator()<InputT, static_cast<int>(compiled.outputs_per_warp),
                                                 static_cast<int>(compiled.packet_bytes),
                                                 static_cast<int>(compiled.max_components_per_lane),
                                                 compiled.load_ahead>();
                    dispatched = true;
                }
            }
        };
        [&]<size_t... I>(std::index_sequence<I...>) {
            (try_unit.template operator()<I>(), ...);
        }(std::make_index_sequence<ONE_WINDOW_UNITS.size()>{});
    };
    if (input_dtype == DataType::FP16) {
        for_type.template operator()<__half>();
    } else if (input_dtype == DataType::BF16) {
        for_type.template operator()<__nv_bfloat16>();
    }
    return dispatched;
}

[[nodiscard]] std::optional<OneWindowPlan> f16OneWindowUnitPlan(uint64_t k,
                                                                const OneWindowUnit& unit,
                                                                bool aligned_base = true) {
    if (!oneWindowF16Unit(unit)) return std::nullopt;
    return oneWindowUnitPlan(k, unit, 2, aligned_base);
}

// 2-byte OneWindow indexes input bytes in 32 bits, as FP32: at most 2^32 - 1 input bytes.
[[nodiscard]] inline bool f16OneWindowFitsIndexing(const ReductionGeometry& geometry) {
    constexpr uint64_t MaxIndex = std::numeric_limits<uint32_t>::max();
    return shallowFp8FitsUint32Indexing(geometry) && geometry.input_elements <= MaxIndex / 2;
}

// The compiled unit a selector OneWindow variant (outputs per warp, packet bytes) runs at K: the one unit with that
// ownership, packet width and K's bucket (each such triple has exactly one loop form).
[[nodiscard]] std::optional<OneWindowUnit> oneWindowUnitForVariant(uint64_t k,
                                                                   uint32_t outputs_per_warp,
                                                                   uint32_t packet_bytes) {
    const auto plan = oneWindowPlan(k, outputs_per_warp, packet_bytes);
    if (!plan) {
        return std::nullopt;
    }
    for (const OneWindowUnit& unit : ONE_WINDOW_UNITS) {
        if (unit.outputs_per_warp == outputs_per_warp && unit.packet_bytes == packet_bytes &&
            unit.max_components_per_lane == plan->max_components_per_lane) {
            return unit;
        }
    }
    return std::nullopt;
}

// The VALUE OneWindow kernel reads any element-aligned base. Off a 16-byte boundary its byte offsets start up to 15
// bytes before the base, so the input holds 15 bytes less than the 32-bit offset limit.
[[nodiscard]] inline bool oneWindowBaseIsAligned(const StridedSpan& input) {
    return (reinterpret_cast<uintptr_t>(input.getMemPtr<void>()) & uintptr_t{15}) == 0;
}
[[nodiscard]] inline bool oneWindowUnalignedBaseFits(const StridedSpan& input, uint64_t element_bytes,
                                                     uint64_t input_elements) {
    constexpr uint64_t MaxIndex = std::numeric_limits<uint32_t>::max();
    return (reinterpret_cast<uintptr_t>(input.getMemPtr<void>()) & static_cast<uintptr_t>(element_bytes - 1)) == 0 &&
           input_elements <= (MaxIndex - 15) / element_bytes;
}

template <int FixedOp>
void launchFp8OneWindowCompleteTyped(const StridedSpan& input,
                                     const StridedSpan& output,
                                     const ReductionGeometry& geometry,
                                     const OneWindowUnit& unit,
                                     uint32_t warps_per_output,
                                     Stream& stream,
                                     const ReductionStageSemantics& semantics,
                                     float output_scale) {
    const AdditiveFinalizeFp32 output_finalize = fp8CompleteFinalizerFromSemantics(semantics);
    // The plan of the base's alignment (rows per group leave room for the prefix off a 16-byte boundary).
    const bool aligned_base = oneWindowBaseIsAligned(input);
    const auto plan = oneWindowUnitPlan(geometry.inner_size, unit, 1, aligned_base);
    if (!plan || geometry.outer_size == 0 || geometry.reduction_size == 0 ||
        !oneWindowValidWarpsPerOutput(unit.outputs_per_warp, warps_per_output) ||
        !shallowFp8FitsUint32Indexing(geometry) || !fp8CompleteSupportsOutputStorage(output.getDataType()) ||
        output.getTotalNumElements() != geometry.outer_size * geometry.inner_size) {
        throw std::logic_error("FP8 bucketed OneWindow received unsupported complete geometry.");
    }
    if (!aligned_base && !oneWindowUnalignedBaseFits(input, 1, geometry.input_elements)) {
        throw std::logic_error("FP8 bucketed OneWindow at an unaligned base holds at most 2^32 - 16 input bytes.");
    }
    float folded_scale = output_scale;
    const Fp8ExtremumFold extremum_fold =
        fp8CompleteExtremumFold(FixedOp, semantics.input_transform, folded_scale);
    const uint64_t outputs_per_block = (SHALLOW_COMPLETE_WARPS / warps_per_output) * unit.outputs_per_warp;
    const uint64_t block_count = std::min<uint64_t>(ceilDivideU64(geometry.outer_size, outputs_per_block),
                                                    PERSISTENT_MAX_GRID_BLOCKS);
    const bool launched = dispatchOneWindow(
        input.getDataType(), unit,
        [&]<typename InputT, int OutputsPerWarp, int PacketBytes, int MaxComponentsPerLane, bool LoadAhead>() {
            oneWindowValueKernel<InputT, OutputsPerWarp, PacketBytes, MaxComponentsPerLane, LoadAhead,
                                 fp8PhysicalExtremumOp(FixedOp)>
                <<<static_cast<unsigned int>(block_count), SHALLOW_COMPLETE_THREADS, 0, stream.getStream()>>>(
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
}

template <int FixedOp>
KernelOccupancyInfo queryFp8OneWindowOccupancyTyped(DataType input_dtype,
                                                    uint64_t inner_size,
                                                    const OneWindowUnit& unit,
                                                    uint32_t warps_per_output) {
    const auto plan = oneWindowUnitPlan(inner_size, unit);
    if (!plan || !oneWindowValidWarpsPerOutput(unit.outputs_per_warp, warps_per_output)) {
        throw std::logic_error("FP8 bucketed OneWindow occupancy received unsupported ownership.");
    }
    KernelOccupancyInfo occupancy{};
    const bool queried = dispatchOneWindow(
        input_dtype, unit,
        [&]<typename InputT, int OutputsPerWarp, int PacketBytes, int MaxComponentsPerLane, bool LoadAhead>() {
            occupancy = queryShallowCompleteOccupancy(
                oneWindowValueKernel<InputT, OutputsPerWarp, PacketBytes, MaxComponentsPerLane,
                                     LoadAhead, fp8PhysicalExtremumOp(FixedOp)>);
        });
    if (!queried) {
        throw std::logic_error("FP8 bucketed OneWindow occupancy has no kernel for this request.");
    }
    return occupancy;
}

template <int FixedOp>
void launchFp32OneWindowCompleteTyped(const StridedSpan& input,
                                      const StridedSpan& output,
                                      const ReductionGeometry& geometry,
                                      const OneWindowUnit& unit,
                                      uint32_t warps_per_output,
                                      Stream& stream,
                                      const ReductionStageSemantics& semantics,
                                      float output_scale) {
    const AdditiveFinalizeFp32 output_finalize = fp8CompleteFinalizerFromSemantics(semantics);
    const bool aligned_base = oneWindowBaseIsAligned(input);
    const auto plan = fp32OneWindowUnitPlan(geometry.inner_size, unit, aligned_base);
    if (input.getDataType() != DataType::FP32 || !plan || geometry.outer_size == 0 || geometry.reduction_size == 0 ||
        !oneWindowValidWarpsPerOutput(unit.outputs_per_warp, warps_per_output) ||
        !fp32OneWindowFitsIndexing(geometry) || !fp8CompleteSupportsOutputStorage(output.getDataType()) ||
        input.getTotalNumElements() != geometry.input_elements ||
        output.getTotalNumElements() != geometry.outer_size * geometry.inner_size) {
        throw std::logic_error("FP32 OneWindow received unsupported complete geometry.");
    }
    if (!aligned_base && !oneWindowUnalignedBaseFits(input, 4, geometry.input_elements)) {
        throw std::logic_error("FP32 OneWindow requires a 4-byte aligned input base (and at most 2^32 - 16 input "
                               "bytes off a 16-byte boundary).");
    }
    float folded_scale = output_scale;
    const Fp8ExtremumFold extremum_fold = fp32CompleteExtremumFold(FixedOp, semantics.input_transform, folded_scale);
    const uint64_t outputs_per_block = (SHALLOW_COMPLETE_WARPS / warps_per_output) * unit.outputs_per_warp;
    const uint64_t block_count = std::min<uint64_t>(ceilDivideU64(geometry.outer_size, outputs_per_block),
                                                    PERSISTENT_MAX_GRID_BLOCKS);
    const bool launched = dispatchFp32OneWindow(
        unit, [&]<typename InputT, int OutputsPerWarp, int PacketBytes, int MaxComponentsPerLane, bool LoadAhead>() {
            oneWindowValueKernel<InputT, OutputsPerWarp, PacketBytes, MaxComponentsPerLane, LoadAhead,
                                 fp8PhysicalExtremumOp(FixedOp)>
                <<<static_cast<unsigned int>(block_count), SHALLOW_COMPLETE_THREADS, 0, stream.getStream()>>>(
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
KernelOccupancyInfo queryFp32OneWindowOccupancyTyped(uint64_t inner_size,
                                                     const OneWindowUnit& unit,
                                                     uint32_t warps_per_output) {
    const auto plan = fp32OneWindowUnitPlan(inner_size, unit);
    if (!plan || !oneWindowValidWarpsPerOutput(unit.outputs_per_warp, warps_per_output)) {
        throw std::logic_error("FP32 OneWindow occupancy received unsupported ownership.");
    }
    KernelOccupancyInfo occupancy{};
    const bool queried = dispatchFp32OneWindow(
        unit, [&]<typename InputT, int OutputsPerWarp, int PacketBytes, int MaxComponentsPerLane, bool LoadAhead>() {
            occupancy = queryShallowCompleteOccupancy(
                oneWindowValueKernel<InputT, OutputsPerWarp, PacketBytes, MaxComponentsPerLane,
                                     LoadAhead, fp8PhysicalExtremumOp(FixedOp)>);
        });
    if (!queried) {
        throw std::logic_error("FP32 OneWindow occupancy has no kernel for this request.");
    }
    return occupancy;
}

// FP16 / BF16 on the same kernel.
template <int FixedOp>
void launchF16OneWindowCompleteTyped(const StridedSpan& input,
                                     const StridedSpan& output,
                                     const ReductionGeometry& geometry,
                                     const OneWindowUnit& unit,
                                     uint32_t warps_per_output,
                                     Stream& stream,
                                     const ReductionStageSemantics& semantics,
                                     float output_scale) {
    const AdditiveFinalizeFp32 output_finalize = fp8CompleteFinalizerFromSemantics(semantics);
    const bool aligned_base = oneWindowBaseIsAligned(input);
    const auto plan = f16OneWindowUnitPlan(geometry.inner_size, unit, aligned_base);
    const DataType dtype = input.getDataType();
    if ((dtype != DataType::FP16 && dtype != DataType::BF16) || !plan || geometry.outer_size == 0 ||
        geometry.reduction_size == 0 || !oneWindowValidWarpsPerOutput(unit.outputs_per_warp, warps_per_output) ||
        !f16OneWindowFitsIndexing(geometry) || !fp8CompleteSupportsOutputStorage(output.getDataType()) ||
        input.getTotalNumElements() != geometry.input_elements ||
        output.getTotalNumElements() != geometry.outer_size * geometry.inner_size) {
        throw std::logic_error("FP16 / BF16 OneWindow received unsupported complete geometry.");
    }
    if (!aligned_base && !oneWindowUnalignedBaseFits(input, 2, geometry.input_elements)) {
        throw std::logic_error("FP16 / BF16 OneWindow requires a 2-byte aligned input base (and at most 2^32 - 16 "
                               "input bytes off a 16-byte boundary).");
    }
    float folded_scale = output_scale;
    const Fp8ExtremumFold extremum_fold = fp16CompleteExtremumFold(FixedOp, semantics.input_transform, folded_scale);
    const uint64_t outputs_per_block = (SHALLOW_COMPLETE_WARPS / warps_per_output) * unit.outputs_per_warp;
    const uint64_t block_count = std::min<uint64_t>(ceilDivideU64(geometry.outer_size, outputs_per_block),
                                                    PERSISTENT_MAX_GRID_BLOCKS);
    const bool launched = dispatchF16OneWindow(
        dtype, unit,
        [&]<typename InputT, int OutputsPerWarp, int PacketBytes, int MaxComponentsPerLane, bool LoadAhead>() {
            oneWindowValueKernel<InputT, OutputsPerWarp, PacketBytes, MaxComponentsPerLane, LoadAhead,
                                 fp8PhysicalExtremumOp(FixedOp)>
                <<<static_cast<unsigned int>(block_count), SHALLOW_COMPLETE_THREADS, 0, stream.getStream()>>>(
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
        throw std::logic_error("FP16 / BF16 OneWindow has no kernel for this request.");
    }
}

template <int FixedOp>
KernelOccupancyInfo queryF16OneWindowOccupancyTyped(DataType input_dtype,
                                                    uint64_t inner_size,
                                                    const OneWindowUnit& unit,
                                                    uint32_t warps_per_output) {
    const auto plan = f16OneWindowUnitPlan(inner_size, unit);
    if (!plan || !oneWindowValidWarpsPerOutput(unit.outputs_per_warp, warps_per_output)) {
        throw std::logic_error("FP16 / BF16 OneWindow occupancy received unsupported ownership.");
    }
    KernelOccupancyInfo occupancy{};
    const bool queried = dispatchF16OneWindow(
        input_dtype, unit,
        [&]<typename InputT, int OutputsPerWarp, int PacketBytes, int MaxComponentsPerLane, bool LoadAhead>() {
            occupancy = queryShallowCompleteOccupancy(
                oneWindowValueKernel<InputT, OutputsPerWarp, PacketBytes, MaxComponentsPerLane,
                                     LoadAhead, fp8PhysicalExtremumOp(FixedOp)>);
        });
    if (!queried) {
        throw std::logic_error("FP16 / BF16 OneWindow occupancy has no kernel for this request.");
    }
    return occupancy;
}

// ---------------------------------------------------------------------------------------------------------------
// The FP32 RK ARG OneWindow twin: the FP32 path of oneWindowValueKernel (the FP32 units, buckets <= 4: packet windows
// of rows_per_group rows staged plane-major, O1/O2/O4/O8 outputs per warp, R workers per component in one-component
// buckets, O1 W1/W2/W4/W8) with one FP32 ARG candidate (Fp32ArgKey.cuh) per owned component; the tail is the absolute
// row << 1 | negative zero. A lane's rows only increase, so a strictly smaller key replaces its best; workers fold with
// butterfly shuffles (power-of-two counts) or shared memory, O1 warps in shared memory, all with the (key, tail)
// minimum. Complete: writes the requested outputs. R <= 2^31 - 1.
// FP16 / BF16 input runs at the 2-byte VALUE units (buckets <= 8): element e of the window is half (e % 2) of word
// (e % 8) / 2 of packet e / 8, widened to FP32 and given the FP32 key.
template <typename InputT, int OutputsPerWarp, int MaxComponentsPerLane>
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
        uint32_t bits;
        if constexpr (std::is_same_v<InputT, float>) {
            bits = warp_words[(element % 4U) * TILED_REDUCTION_WARP_THREADS + subgroup_lane_base + element / 4U];
        } else {
            const uint32_t within = element % 8U;
            const uint32_t word =
                warp_words[(within >> 1) * TILED_REDUCTION_WARP_THREADS + subgroup_lane_base + element / 8U];
            bits = __float_as_uint(widenF16Bits<InputT>((within & 1U) != 0 ? word >> 16 : word));
        }
        uint32_t negative_zero;
        const uint32_t key = fp32ArgKey(bits, xor_mask, negative_zero);
        if (key < best_key[slot]) {
            best_key[slot] = key;
            best_tail[slot] = ((group_row + row) << 1) | negative_zero;
        }
    };
    const uint32_t first_element = prefix_bytes / static_cast<uint32_t>(sizeof(InputT));
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
// (identity first). Force-inlined like streamOneWindowOutput: no lambda holds the candidate arrays.
template <typename InputT, int OutputsPerWarp, int MaxComponentsPerLane, bool LoadAhead>
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
                                                             const OneWindowGeometry& geometry,
                                                             uint32_t lane,
                                                             uint32_t subgroup_lane,
                                                             uint32_t subgroup_lane_base,
                                                             uint32_t component,
                                                             uint32_t worker,
                                                             uint32_t xor_mask) {
    constexpr uint32_t PacketBytes = 16;
    constexpr uint32_t ElementBytes = static_cast<uint32_t>(sizeof(InputT));
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
            current = oneWindowGroupWindow<PacketBytes, ElementBytes>(valid_output, outer_byte_begin, group_row,
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
                next = oneWindowGroupWindow<PacketBytes, ElementBytes>(valid_output, outer_byte_begin, next_row,
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
                keyOneWindowFp32ArgGroup<InputT, OutputsPerWarp, MaxComponentsPerLane>(
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
            const OneWindowGroupWindow window = oneWindowGroupWindow<PacketBytes, ElementBytes>(
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
                keyOneWindowFp32ArgGroup<InputT, OutputsPerWarp, MaxComponentsPerLane>(
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

template <typename InputT, int OutputsPerWarp, int MaxComponentsPerLane, bool LoadAhead>
__global__ __launch_bounds__(SHALLOW_COMPLETE_THREADS) void oneWindowArgKernel(const InputT* input,
                                                                             Fp32ArgOutputs outputs,
                                                                             uint32_t outer_size,
                                                                             uint32_t reduction_size,
                                                                             uint32_t warps_per_output,
                                                                             OneWindowGeometry geometry) {
    static_assert(OutputsPerWarp == 1 || OutputsPerWarp == 2 || OutputsPerWarp == 4 || OutputsPerWarp == 8);
    static_assert(std::is_same_v<InputT, float> || isF16InputType<InputT>);
    static_assert(MaxComponentsPerLane == 1 || MaxComponentsPerLane == 2 || MaxComponentsPerLane == 4 ||
                  (MaxComponentsPerLane == 8 && sizeof(InputT) == 2));
    constexpr uint32_t PacketWords = 4;
    constexpr uint32_t LanesPerOutput = TILED_REDUCTION_WARP_THREADS / OutputsPerWarp;
    constexpr bool MultipleComponentsPerLane = MaxComponentsPerLane > 1;
    constexpr uint32_t MaxK = LanesPerOutput * static_cast<uint32_t>(MaxComponentsPerLane);
    constexpr uint32_t LanePartialColumns = MultipleComponentsPerLane ? 1U : TILED_REDUCTION_WARP_THREADS;
    constexpr uint32_t WarpPartialRows = OutputsPerWarp == 1 ? SHALLOW_COMPLETE_WARPS : 1U;
    constexpr uint32_t WarpPartialColumns = OutputsPerWarp == 1 ? MaxK : 1U;

    __shared__ uint32_t staged_packet_words[SHALLOW_COMPLETE_WARPS][PacketWords * TILED_REDUCTION_WARP_THREADS];
    __shared__ uint32_t lane_keys[SHALLOW_COMPLETE_WARPS][LanePartialColumns];
    __shared__ uint32_t lane_tails[SHALLOW_COMPLETE_WARPS][LanePartialColumns];
    __shared__ uint32_t warp_keys[WarpPartialRows][WarpPartialColumns];
    __shared__ uint32_t warp_tails[WarpPartialRows][WarpPartialColumns];

    const uint32_t k = geometry.k;
    const uint32_t workers = geometry.workers_per_component;
    const uint32_t warp = threadIdx.x / TILED_REDUCTION_WARP_THREADS;
    const uint32_t lane = threadIdx.x % TILED_REDUCTION_WARP_THREADS;
    const uint32_t outer_input_stride = reduction_size * k * static_cast<uint32_t>(sizeof(InputT));  // bytes
    // As in the VALUE kernel, byte offsets count from the 16-byte boundary at or below the base (base_shift bytes
    // before it; 0 for an aligned base), so the kernel reads any element address. The keys carry rows, not bytes, so
    // the indices are unchanged.
    const uint32_t base_shift = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(input) & uintptr_t{15});
    const uint32_t total_input_bytes = base_shift + outer_size * outer_input_stride;
    const uint8_t* const input_bytes = reinterpret_cast<const uint8_t*>(input) - base_shift;
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
        const uint32_t global_warp = blockIdx.x * SHALLOW_COMPLETE_WARPS + warp;
        const uint32_t grid_warps = gridDim.x * SHALLOW_COMPLETE_WARPS;
        for (uint32_t warp_group = global_warp;; warp_group += grid_warps) {
            const uint64_t outer_begin = static_cast<uint64_t>(warp_group) * OutputsPerWarp;
            if (outer_begin >= outer_size) {
                return;
            }
            const uint64_t outer_index = outer_begin + output_slot;
            const bool valid_output = outer_index < outer_size;
            streamOneWindowFp32ArgOutput<InputT, OutputsPerWarp, MaxComponentsPerLane, LoadAhead>(
                best_key, best_tail, staged_packet_words[warp], input_bytes, total_input_bytes, valid_output,
                valid_output ? base_shift + static_cast<uint32_t>(outer_index) * outer_input_stride : 0U, 0U,
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
        const uint32_t outputs_per_block = SHALLOW_COMPLETE_WARPS / warps_per_output;
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
            streamOneWindowFp32ArgOutput<InputT, OutputsPerWarp, MaxComponentsPerLane, LoadAhead>(
                best_key, best_tail, staged_packet_words[warp], input_bytes, total_input_bytes, valid_output,
                valid_output ? base_shift + static_cast<uint32_t>(outer_index) * outer_input_stride : 0U,
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

// The carried-index OneWindow ARG twin (see Fp32ArgWidePartials): oneWindowArgKernel's units, windows and merge
// topology, staging each lane's carried index packet (the same elements as its value packet, plane-major) next to the
// values. An element's tail is (carried + row * row_stride) << 1 | negative zero, and every merge is the full (key,
// tail) minimum (composed indices are not monotone in the row). Complete only; the input and the carried index are
// 16-byte aligned and hold at most 2^32 - 1 value bytes.
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
// (identity first). Force-inlined like streamOneWindowOutput: no lambda holds the candidate arrays.
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
                                                             const OneWindowGeometry& geometry,
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
            current = oneWindowGroupWindow<PacketBytes, 4U>(valid_output, outer_byte_begin, group_row,
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
                next = oneWindowGroupWindow<PacketBytes, 4U>(valid_output, outer_byte_begin, next_row,
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
            const OneWindowGroupWindow window = oneWindowGroupWindow<PacketBytes, 4U>(
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
__global__ __launch_bounds__(SHALLOW_COMPLETE_THREADS) void oneWindowCarriedArgKernel(const float* input,
                                                                                    const IndexT* carried,
                                                                                    uint64_t row_stride_index,
                                                                                    Fp32ArgOutputs outputs,
                                                                             uint32_t outer_size,
                                                                             uint32_t reduction_size,
                                                                             uint32_t warps_per_output,
                                                                             OneWindowGeometry geometry) {
    static_assert(OutputsPerWarp == 1 || OutputsPerWarp == 2 || OutputsPerWarp == 4 || OutputsPerWarp == 8);
    static_assert(MaxComponentsPerLane == 1 || MaxComponentsPerLane == 2 || MaxComponentsPerLane == 4);
    static_assert(std::is_same_v<IndexT, uint32_t> || std::is_same_v<IndexT, uint64_t>);
    constexpr uint32_t PacketWords = 4;
    constexpr uint32_t LanesPerOutput = TILED_REDUCTION_WARP_THREADS / OutputsPerWarp;
    constexpr bool MultipleComponentsPerLane = MaxComponentsPerLane > 1;
    constexpr uint32_t MaxK = LanesPerOutput * static_cast<uint32_t>(MaxComponentsPerLane);
    constexpr uint32_t LanePartialColumns = MultipleComponentsPerLane ? 1U : TILED_REDUCTION_WARP_THREADS;
    constexpr uint32_t WarpPartialRows = OutputsPerWarp == 1 ? SHALLOW_COMPLETE_WARPS : 1U;
    constexpr uint32_t WarpPartialColumns = OutputsPerWarp == 1 ? MaxK : 1U;

    __shared__ uint32_t staged_packet_words[SHALLOW_COMPLETE_WARPS][PacketWords * TILED_REDUCTION_WARP_THREADS];
    __shared__ IndexT staged_indices[SHALLOW_COMPLETE_WARPS][PacketWords * TILED_REDUCTION_WARP_THREADS];
    __shared__ uint32_t lane_keys[SHALLOW_COMPLETE_WARPS][LanePartialColumns];
    __shared__ uint64_t lane_tails[SHALLOW_COMPLETE_WARPS][LanePartialColumns];
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
        const uint32_t global_warp = blockIdx.x * SHALLOW_COMPLETE_WARPS + warp;
        const uint32_t grid_warps = gridDim.x * SHALLOW_COMPLETE_WARPS;
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
        const uint32_t outputs_per_block = SHALLOW_COMPLETE_WARPS / warps_per_output;
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

// The FP8 OneWindow argmin/argmax twins (one kernel per OneWindow unit; the runtime Fp8ArgFold covers
// argmin/argmax x E4M3/E5M2). Same packet windows, staging, lane ownership, R workers and merge topology as
// oneWindowValueKernel, with one uint32 lane candidate per owned component (Fp8ArgKey.h) in place of an FP32
// accumulator. Candidates carry the absolute row, so lanes, workers and warps merge with a plain unsigned min; the
// launcher requires R < FP8_ARG_MAX_LANE_ROWS. Templates only: kernels are instantiated by the exported launcher after
// this anonymous namespace (the FP8 VALUE inventory TU compiles this prefix and must not gain ARG kernels).

// Linear staging for the O1 ARG units with one or two components per lane. The O1 subgroup is the whole warp, so lane's
// packet goes to bytes [lane * PacketBytes, +PacketBytes) of the warp's slice (one vector store) and staged byte b is
// simply byte b (one LDS.U8, no plane arithmetic). Plane-major staging cost ~10 SASS instructions per gathered byte
// (shift, mask, plane index, word load, extract), ~41 of the 72 per four rows; the O1 reads stay conflict-free (lanes
// read at most 32 contiguous bytes per row step).
// Every c1/c2 unit stages linearly. Subgroup s's slice starts at s * SliceBytes; O2/O4/O8 pad each slice by
// LanesPerOutput words, so consecutive slices start 16/8/4 banks apart (5 * LanesPerOutput words mod 32) and the
// subgroups' reads (at most LanesPerOutput contiguous bytes per row step) never share a bank. O1 c4/c8 also stage
// linearly and own quads of adjacent components (ONE_WINDOW_ARG_QUAD_OWNERSHIP); their two-word reads may touch one
// word past the window, so O1 slices carry a 4-word pad.
// So do O2/O4/O8 c4 (one quad of 4 * LanesPerOutput components). O2/O4 c8 (47 registers, next to the 49 cliff) and O1
// c16 (63, next to 64) keep lane + slot * LanesPerOutput ownership.
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

// decodeOneWindowGroup with keys: the same byte reads and packing; group_row is the group's first row.
// RowsCandidate: reduce each four-row word with fp8ArgRowsCandidate4 before building the lane candidate. It measured
// 0.96x at O1 c2 load-ahead and 0.99x at c1, but 1.02-1.04x (up to 1.17x at large R) for the O2/O4/O8 c2 units without
// load-ahead, so those keep the per-byte candidates.
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
                    // Four rows of this component: reduce the word first (fp8ArgRowsCandidate4).
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
        // O1 c4/c8 and O2/O4/O8 c4, linear staging. Lane owns components q * 4 * LanesPerOutput + 4 * lane .. +3: one
        // key word is two LDS.32 and a funnel shift by the row's byte offset (subgroup-uniform).
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

// streamOneWindowOutput with keyOneWindowArgGroup (kept separate so the VALUE kernel's code is untouched).
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
                                                         const OneWindowGeometry& geometry,
                                                         uint32_t lane,
                                                         uint32_t subgroup_lane,
                                                         uint32_t subgroup_lane_base,
                                                         uint32_t component,
                                                         uint32_t worker,
                                                         const Fp8ArgFold& fold) {
    const bool keys = valid_output && subgroup_lane < geometry.task_count;
    const auto window_at = [&](uint32_t group_row) {
        return oneWindowGroupWindow<PacketBytes>(valid_output, outer_byte_begin, group_row, reduction_size,
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
__global__ void fp8OneWindowArgKernel(const uint8_t* input,
                                      Fp8ArgOutputs outputs,
                                      uint32_t outer_size,
                                      uint32_t reduction_size,
                                      uint32_t warps_per_output,
                                      OneWindowGeometry geometry) {
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
    constexpr uint32_t WarpPartialRows = OutputsPerWarp == 1 ? SHALLOW_COMPLETE_WARPS : 1U;
    constexpr uint32_t WarpPartialColumns = OutputsPerWarp == 1 ? MaxK : 1U;
    constexpr uint32_t Identity = FP8_ARG_LANE_IDENTITY;

    constexpr uint32_t StageWords = ONE_WINDOW_ARG_LINEAR_STAGE<LanesPerOutput, MaxComponentsPerLane>
                                        ? ONE_WINDOW_ARG_STAGE_WORDS<LanesPerOutput, PacketBytes>
                                        : PacketWords * TILED_REDUCTION_WARP_THREADS;
    __shared__ __align__(16) uint32_t staged_packet_words[SHALLOW_COMPLETE_WARPS][StageWords];
    __shared__ uint32_t lane_partials[SHALLOW_COMPLETE_WARPS][LanePartialColumns];
    __shared__ uint32_t warp_partials[WarpPartialRows][WarpPartialColumns];

    const uint32_t k = geometry.k;
    const uint32_t workers = geometry.workers_per_component;
    const uint32_t warp = threadIdx.x / TILED_REDUCTION_WARP_THREADS;
    const uint32_t lane = threadIdx.x % TILED_REDUCTION_WARP_THREADS;
    const uint32_t outer_input_stride = reduction_size * k;
    // As in the VALUE kernel, byte offsets count from the packet boundary at or below the base (base_shift bytes before
    // it; 0 for an aligned base), so the kernel reads any byte address. The keys carry rows, not bytes, so the indices
    // are unchanged.
    const uint32_t base_shift =
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(input) & static_cast<uintptr_t>(PacketBytes - 1));
    const uint32_t total_input_bytes = base_shift + outer_size * outer_input_stride;
    const uint8_t* const input_bytes = input - base_shift;

    const uint32_t output_slot = lane / LanesPerOutput;
    const uint32_t subgroup_lane = lane % LanesPerOutput;
    const uint32_t subgroup_lane_base = output_slot * LanesPerOutput;
    const uint32_t component = MultipleComponentsPerLane ? subgroup_lane : subgroup_lane / workers;
    const uint32_t worker = MultipleComponentsPerLane ? 0U : subgroup_lane - component * workers;

    uint32_t local[MaxComponentsPerLane];

    if constexpr (OutputsPerWarp > 1) {
        static_cast<void>(warps_per_output);
        static_cast<void>(warp_partials);
        const uint32_t global_warp = blockIdx.x * SHALLOW_COMPLETE_WARPS + warp;
        const uint32_t grid_warps = gridDim.x * SHALLOW_COMPLETE_WARPS;
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
            const uint32_t outer_byte_begin =
                valid_output ? base_shift + static_cast<uint32_t>(outer_index) * outer_input_stride : 0;
            streamOneWindowArgOutput<MaxComponentsPerLane, LanesPerOutput, PacketBytes, LoadAhead>(
                local, staged_packet_words[warp], input_bytes, total_input_bytes, valid_output, outer_byte_begin, 0U,
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
        const uint32_t outputs_per_block = SHALLOW_COMPLETE_WARPS / warps_per_output;
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
            const uint32_t outer_byte_begin =
                valid_output ? base_shift + static_cast<uint32_t>(outer_index) * outer_input_stride : 0;
            streamOneWindowArgOutput<MaxComponentsPerLane, LanesPerOutput, PacketBytes, LoadAhead>(
                local, staged_packet_words[warp], input_bytes, total_input_bytes, valid_output, outer_byte_begin,
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

// Invokes callback.template operator()<OutputsPerWarp, PacketBytes, MaxComponentsPerLane, LoadAhead>() for the
// OneWindow unit equal to `unit` (the ARG twins exist for exactly the compiled VALUE units).
template <typename Callback>
[[nodiscard]] bool dispatchOneWindowArg(const OneWindowUnit& unit, Callback&& callback) {
    bool dispatched = false;
    [&]<size_t... I>(std::index_sequence<I...>) {
        static_cast<void>(((ONE_WINDOW_UNITS[I] == unit &&
                            (callback.template operator()<
                                 static_cast<int>(ONE_WINDOW_UNITS[I].outputs_per_warp),
                                 static_cast<int>(ONE_WINDOW_UNITS[I].packet_bytes),
                                 static_cast<int>(ONE_WINDOW_UNITS[I].max_components_per_lane),
                                 ONE_WINDOW_UNITS[I].load_ahead>(),
                             dispatched = true)) ||
                           ...));
    }(std::make_index_sequence<ONE_WINDOW_UNITS.size()>{});
    return dispatched;
}

}  // namespace

// The production OneWindow launcher and occupancy query: the selector's OneWindow variant (O, W, P) runs the bucketed
// unit for (O, P, bucket(K)). Only P16 variants have units; a request without a unit throws std::logic_error, which the
// selector treats as an excluded candidate.
void launchFp8OneWindowForVariant(
    const StridedSpan& input,
    const StridedSpan& output,
    const ReductionGeometry& geometry,
    uint32_t outputs_per_warp,
    uint32_t warps_per_output,
    uint32_t packet_bytes,
    Stream& stream,
    const ReductionStageSemantics& semantics,
    float output_scale) {
    const auto unit = oneWindowUnitForVariant(geometry.inner_size, outputs_per_warp, packet_bytes);
    if (!unit) {
        throw std::logic_error("FP8 OneWindow has no bucketed kernel for this K, ownership and packet width.");
    }
    if (output.getDimensions() != geometry.output_dimensions) {
        throw std::logic_error("FP8 OneWindow received unsupported complete geometry.");
    }
    launchFp8OneWindow(input, output, geometry, *unit, warps_per_output, stream, semantics,
                       output_scale);
}

KernelOccupancyInfo queryFp8OneWindowOccupancyForVariant(
    DataType input_dtype,
    uint64_t inner_size,
    uint32_t outputs_per_warp,
    uint32_t warps_per_output,
    uint32_t packet_bytes,
    ReductionStageCombine combine) {
    const auto unit = oneWindowUnitForVariant(inner_size, outputs_per_warp, packet_bytes);
    if (!unit) {
        throw std::logic_error("FP8 OneWindow occupancy: no bucketed kernel for this K, ownership and packet width.");
    }
    return queryFp8OneWindowOccupancy(input_dtype, inner_size, *unit, warps_per_output, combine);
}

std::vector<OneWindowUnit> fp8OneWindowUnits(uint64_t inner_size) {
    std::vector<OneWindowUnit> units;
    for (const OneWindowUnit& unit : ONE_WINDOW_UNITS) {
        if (oneWindowUnitPlan(inner_size, unit)) {
            units.push_back(unit);
        }
    }
    return units;
}

void launchFp8OneWindow(const StridedSpan& input,
                        const StridedSpan& output,
                        const ReductionGeometry& geometry,
                        const OneWindowUnit& unit,
                        uint32_t warps_per_output,
                        Stream& stream,
                        const ReductionStageSemantics& semantics,
                        float output_scale) {
    switch (semantics.combine) {
        case ReductionStageCombine::Sum:
            return launchFp8OneWindowCompleteTyped<0>(input, output, geometry, unit, warps_per_output, stream,
                                                      semantics, output_scale);
        case ReductionStageCombine::Product:
            return launchFp8OneWindowCompleteTyped<1>(input, output, geometry, unit, warps_per_output, stream,
                                                      semantics, output_scale);
        case ReductionStageCombine::Minimum:
            return launchFp8OneWindowCompleteTyped<2>(input, output, geometry, unit, warps_per_output, stream,
                                                      semantics, output_scale);
        case ReductionStageCombine::Maximum:
            return launchFp8OneWindowCompleteTyped<3>(input, output, geometry, unit, warps_per_output, stream,
                                                      semantics, output_scale);
    }
    throw std::logic_error("Invalid bucketed OneWindow combine operation.");
}

KernelOccupancyInfo queryFp8OneWindowOccupancy(DataType input_dtype,
                                               uint64_t inner_size,
                                               const OneWindowUnit& unit,
                                               uint32_t warps_per_output,
                                               ReductionStageCombine combine) {
    switch (combine) {
        case ReductionStageCombine::Sum:
            return queryFp8OneWindowOccupancyTyped<0>(input_dtype, inner_size, unit, warps_per_output);
        case ReductionStageCombine::Product:
            return queryFp8OneWindowOccupancyTyped<1>(input_dtype, inner_size, unit, warps_per_output);
        case ReductionStageCombine::Minimum:
            return queryFp8OneWindowOccupancyTyped<2>(input_dtype, inner_size, unit, warps_per_output);
        case ReductionStageCombine::Maximum:
            return queryFp8OneWindowOccupancyTyped<3>(input_dtype, inner_size, unit, warps_per_output);
    }
    throw std::logic_error("Invalid bucketed OneWindow occupancy combine operation.");
}

std::vector<OneWindowUnit> fp32OneWindowUnits(uint64_t inner_size) {
    std::vector<OneWindowUnit> units;
    for (const OneWindowUnit& unit : ONE_WINDOW_UNITS) {
        if (fp32OneWindowUnitPlan(inner_size, unit)) {
            units.push_back(unit);
        }
    }
    return units;
}

void launchFp32OneWindow(const StridedSpan& input,
                         const StridedSpan& output,
                         const ReductionGeometry& geometry,
                         const OneWindowUnit& unit,
                         uint32_t warps_per_output,
                         Stream& stream,
                         const ReductionStageSemantics& semantics,
                         float output_scale) {
    switch (semantics.combine) {
        case ReductionStageCombine::Sum:
            return launchFp32OneWindowCompleteTyped<0>(input, output, geometry, unit, warps_per_output, stream,
                                                       semantics, output_scale);
        case ReductionStageCombine::Product:
            return launchFp32OneWindowCompleteTyped<1>(input, output, geometry, unit, warps_per_output, stream,
                                                       semantics, output_scale);
        case ReductionStageCombine::Minimum:
            return launchFp32OneWindowCompleteTyped<2>(input, output, geometry, unit, warps_per_output, stream,
                                                       semantics, output_scale);
        case ReductionStageCombine::Maximum:
            return launchFp32OneWindowCompleteTyped<3>(input, output, geometry, unit, warps_per_output, stream,
                                                       semantics, output_scale);
    }
    throw std::logic_error("Invalid FP32 OneWindow combine operation.");
}

KernelOccupancyInfo queryFp32OneWindowOccupancy(uint64_t inner_size,
                                                const OneWindowUnit& unit,
                                                uint32_t warps_per_output,
                                                ReductionStageCombine combine) {
    switch (combine) {
        case ReductionStageCombine::Sum:
            return queryFp32OneWindowOccupancyTyped<0>(inner_size, unit, warps_per_output);
        case ReductionStageCombine::Product:
            return queryFp32OneWindowOccupancyTyped<1>(inner_size, unit, warps_per_output);
        case ReductionStageCombine::Minimum:
            return queryFp32OneWindowOccupancyTyped<2>(inner_size, unit, warps_per_output);
        case ReductionStageCombine::Maximum:
            return queryFp32OneWindowOccupancyTyped<3>(inner_size, unit, warps_per_output);
    }
    throw std::logic_error("Invalid FP32 OneWindow occupancy combine operation.");
}

std::vector<OneWindowUnit> f16OneWindowUnits(uint64_t inner_size) {
    std::vector<OneWindowUnit> units;
    for (const OneWindowUnit& unit : ONE_WINDOW_UNITS) {
        if (f16OneWindowUnitPlan(inner_size, unit)) {
            units.push_back(unit);
        }
    }
    return units;
}

void launchF16OneWindow(const StridedSpan& input,
                        const StridedSpan& output,
                        const ReductionGeometry& geometry,
                        const OneWindowUnit& unit,
                        uint32_t warps_per_output,
                        Stream& stream,
                        const ReductionStageSemantics& semantics,
                        float output_scale) {
    switch (semantics.combine) {
        case ReductionStageCombine::Sum:
            return launchF16OneWindowCompleteTyped<0>(input, output, geometry, unit, warps_per_output, stream,
                                                      semantics, output_scale);
        case ReductionStageCombine::Product:
            return launchF16OneWindowCompleteTyped<1>(input, output, geometry, unit, warps_per_output, stream,
                                                      semantics, output_scale);
        case ReductionStageCombine::Minimum:
            return launchF16OneWindowCompleteTyped<2>(input, output, geometry, unit, warps_per_output, stream,
                                                      semantics, output_scale);
        case ReductionStageCombine::Maximum:
            return launchF16OneWindowCompleteTyped<3>(input, output, geometry, unit, warps_per_output, stream,
                                                      semantics, output_scale);
    }
    throw std::logic_error("Invalid FP16 / BF16 OneWindow combine operation.");
}

KernelOccupancyInfo queryF16OneWindowOccupancy(DataType input_dtype,
                                               uint64_t inner_size,
                                               const OneWindowUnit& unit,
                                               uint32_t warps_per_output,
                                               ReductionStageCombine combine) {
    switch (combine) {
        case ReductionStageCombine::Sum:
            return queryF16OneWindowOccupancyTyped<0>(input_dtype, inner_size, unit, warps_per_output);
        case ReductionStageCombine::Product:
            return queryF16OneWindowOccupancyTyped<1>(input_dtype, inner_size, unit, warps_per_output);
        case ReductionStageCombine::Minimum:
            return queryF16OneWindowOccupancyTyped<2>(input_dtype, inner_size, unit, warps_per_output);
        case ReductionStageCombine::Maximum:
            return queryF16OneWindowOccupancyTyped<3>(input_dtype, inner_size, unit, warps_per_output);
    }
    throw std::logic_error("Invalid FP16 / BF16 OneWindow occupancy combine operation.");
}

// ---- The FP32 and FP16 / BF16 RK ARG OneWindow twins (see oneWindowArgKernel).
namespace {
// aligned_base false plans for an input base off a 16-byte boundary (oneWindowPlan).
template <typename InputT>
[[nodiscard]] std::optional<OneWindowPlan> oneWindowArgUnitPlan(uint64_t k,
                                                                const OneWindowUnit& unit,
                                                                bool aligned_base = true) {
    if constexpr (std::is_same_v<InputT, float>) {
        return fp32OneWindowUnitPlan(k, unit, aligned_base);
    } else {
        return f16OneWindowUnitPlan(k, unit, aligned_base);
    }
}

template <typename InputT>
[[nodiscard]] bool oneWindowArgSupportsTyped(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size,
                                             const OneWindowUnit& unit, uint32_t warps_per_output) {
    if (outer_size == 0 || reduction_size == 0 || reduction_size > FP32_ARG_MAX_SHARD_ROWS ||
        !oneWindowArgUnitPlan<InputT>(inner_size, unit) ||
        !oneWindowValidWarpsPerOutput(unit.outputs_per_warp, warps_per_output)) {
        return false;
    }
    constexpr uint64_t MaxIndex = std::numeric_limits<uint32_t>::max();
    // The kernel indexes input bytes in 32 bits (fp32OneWindowFitsIndexing / f16OneWindowFitsIndexing).
    return outer_size <= MaxIndex && outer_size * reduction_size * inner_size <= MaxIndex / sizeof(InputT);
}

// Invokes callback.template operator()<InputT, OutputsPerWarp, MaxComponentsPerLane, LoadAhead>() for `unit` at the
// input type (FP32's units, or the 2-byte ones for FP16 / BF16).
template <typename InputT, typename Callback>
[[nodiscard]] bool dispatchOneWindowArgUnit(const OneWindowUnit& unit, Callback&& callback) {
    const auto forward = [&]<typename T, int OutputsPerWarp, int PacketBytes, int MaxComponentsPerLane,
                             bool LoadAhead>() {
        static_assert(PacketBytes == 16);
        callback.template operator()<OutputsPerWarp, MaxComponentsPerLane, LoadAhead>();
    };
    if constexpr (std::is_same_v<InputT, float>) {
        return dispatchFp32OneWindow(unit, forward);
    } else {
        return dispatchF16OneWindow(std::is_same_v<InputT, __half> ? DataType::FP16 : DataType::BF16, unit,
                                    forward);
    }
}

template <typename InputT>
void launchOneWindowArgTyped(const StridedSpan& input,
                             const StridedSpan* value_output,
                             const StridedSpan* index_output,
                             const ReductionGeometry& geometry,
                             const OneWindowUnit& unit,
                             uint32_t warps_per_output,
                             bool maximum,
                             uint64_t domain_stride,
                             Stream& stream,
                             const char* name) {
    const uint64_t outputs = geometry.outer_size * geometry.inner_size;
    // The plan of the base's alignment (rows per group leave room for the prefix off a 16-byte boundary).
    const bool aligned_base = oneWindowBaseIsAligned(input);
    const auto plan = oneWindowArgUnitPlan<InputT>(geometry.inner_size, unit, aligned_base);
    if (!plan ||
        !oneWindowArgSupportsTyped<InputT>(geometry.outer_size, geometry.reduction_size, geometry.inner_size, unit,
                                           warps_per_output) ||
        input.getTotalNumElements() != geometry.outer_size * geometry.reduction_size * geometry.inner_size ||
        (value_output == nullptr && index_output == nullptr) ||
        (value_output != nullptr && (value_output->getTotalNumElements() != outputs ||
                                     !fp8CompleteSupportsOutputStorage(value_output->getDataType()))) ||
        (index_output != nullptr && (index_output->getTotalNumElements() != outputs ||
                                     (index_output->getDataType() != DataType::UINT32 &&
                                      index_output->getDataType() != DataType::UINT64)))) {
        throw std::logic_error(std::string(name) + " OneWindow ARG received unsupported geometry or outputs.");
    }
    if (!aligned_base && !oneWindowUnalignedBaseFits(input, sizeof(InputT), input.getTotalNumElements())) {
        throw std::logic_error(std::string(name) + " OneWindow ARG requires an element-aligned input base (and at "
                                                   "most 2^32 - 16 input bytes off a 16-byte boundary).");
    }
    if (index_output != nullptr && index_output->getDataType() == DataType::UINT32 &&
        (geometry.reduction_size - 1) * domain_stride > std::numeric_limits<uint32_t>::max()) {
        throw std::logic_error(std::string(name) +
                               " OneWindow ARG UINT32 index cannot hold the original-domain index.");
    }
    const Fp32ArgOutputs arg_outputs{value_output == nullptr ? nullptr : value_output->getMemPtr<void>(),
                                     value_output == nullptr ? DataType::FP32 : value_output->getDataType(),
                                     index_output == nullptr ? nullptr : index_output->getMemPtr<void>(),
                                     index_output == nullptr ? DataType::UINT32 : index_output->getDataType(),
                                     domain_stride,
                                     maximum ? 0x80000000U : 0U};
    const uint64_t outputs_per_block = (SHALLOW_COMPLETE_WARPS / warps_per_output) * unit.outputs_per_warp;
    const uint64_t block_count = std::min<uint64_t>(ceilDivideU64(geometry.outer_size, outputs_per_block),
                                                    PERSISTENT_MAX_GRID_BLOCKS);
    const bool launched =
        dispatchOneWindowArgUnit<InputT>(unit, [&]<int OutputsPerWarp, int MaxComponentsPerLane, bool LoadAhead>() {
            oneWindowArgKernel<InputT, OutputsPerWarp, MaxComponentsPerLane, LoadAhead>
                <<<static_cast<unsigned int>(block_count), SHALLOW_COMPLETE_THREADS, 0, stream.getStream()>>>(
                    input.getMemPtr<InputT>(), arg_outputs, static_cast<uint32_t>(geometry.outer_size),
                    static_cast<uint32_t>(geometry.reduction_size), warps_per_output, plan->geometry);
            CUDA_CHECK(cudaGetLastError());
        });
    if (!launched) {
        throw std::logic_error(std::string(name) + " OneWindow ARG has no kernel for this request.");
    }
}

template <typename InputT>
[[nodiscard]] KernelOccupancyInfo queryOneWindowArgOccupancyTyped(uint64_t inner_size,
                                                                  const OneWindowUnit& unit,
                                                                  uint32_t warps_per_output, const char* name) {
    if (!oneWindowArgUnitPlan<InputT>(inner_size, unit) ||
        !oneWindowValidWarpsPerOutput(unit.outputs_per_warp, warps_per_output)) {
        throw std::logic_error(std::string(name) + " OneWindow ARG occupancy received unsupported ownership.");
    }
    KernelOccupancyInfo occupancy{};
    const bool queried =
        dispatchOneWindowArgUnit<InputT>(unit, [&]<int OutputsPerWarp, int MaxComponentsPerLane, bool LoadAhead>() {
            occupancy = queryShallowCompleteOccupancy(
                oneWindowArgKernel<InputT, OutputsPerWarp, MaxComponentsPerLane, LoadAhead>);
        });
    if (!queried) {
        throw std::logic_error(std::string(name) + " OneWindow ARG occupancy has no kernel for this request.");
    }
    return occupancy;
}
}  // namespace

bool fp32OneWindowArgSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size,
                              const OneWindowUnit& unit, uint32_t warps_per_output) {
    return oneWindowArgSupportsTyped<float>(outer_size, reduction_size, inner_size, unit, warps_per_output);
}

bool f16OneWindowArgSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size,
                             const OneWindowUnit& unit, uint32_t warps_per_output) {
    return oneWindowArgSupportsTyped<__half>(outer_size, reduction_size, inner_size, unit, warps_per_output);
}

void launchFp32OneWindowArg(const StridedSpan& input,
                            const StridedSpan* value_output,
                            const StridedSpan* index_output,
                            const ReductionGeometry& geometry,
                            const OneWindowUnit& unit,
                            uint32_t warps_per_output,
                            bool maximum,
                            uint64_t domain_stride,
                            Stream& stream) {
    if (input.getDataType() != DataType::FP32)
        throw std::logic_error("FP32 OneWindow ARG received unsupported geometry or outputs.");
    launchOneWindowArgTyped<float>(input, value_output, index_output, geometry, unit, warps_per_output, maximum,
                                   domain_stride, stream, "FP32");
}

void launchF16OneWindowArg(const StridedSpan& input,
                           const StridedSpan* value_output,
                           const StridedSpan* index_output,
                           const ReductionGeometry& geometry,
                           const OneWindowUnit& unit,
                           uint32_t warps_per_output,
                           bool maximum,
                           uint64_t domain_stride,
                           Stream& stream) {
    switch (input.getDataType()) {
        case DataType::FP16:
            return launchOneWindowArgTyped<__half>(input, value_output, index_output, geometry, unit,
                                                   warps_per_output, maximum, domain_stride, stream, "FP16");
        case DataType::BF16:
            return launchOneWindowArgTyped<__nv_bfloat16>(input, value_output, index_output, geometry, unit,
                                                          warps_per_output, maximum, domain_stride, stream, "BF16");
        default:
            throw std::logic_error("FP16 / BF16 OneWindow ARG received a non-FP16 / BF16 input.");
    }
}

KernelOccupancyInfo queryFp32OneWindowArgOccupancy(uint64_t inner_size,
                                                   const OneWindowUnit& unit,
                                                   uint32_t warps_per_output) {
    return queryOneWindowArgOccupancyTyped<float>(inner_size, unit, warps_per_output, "FP32");
}

KernelOccupancyInfo queryF16OneWindowArgOccupancy(DataType input_dtype, uint64_t inner_size,
                                                  const OneWindowUnit& unit,
                                                  uint32_t warps_per_output) {
    switch (input_dtype) {
        case DataType::FP16: return queryOneWindowArgOccupancyTyped<__half>(inner_size, unit, warps_per_output, "FP16");
        case DataType::BF16:
            return queryOneWindowArgOccupancyTyped<__nv_bfloat16>(inner_size, unit, warps_per_output, "BF16");
        default: throw std::logic_error("FP16 / BF16 OneWindow ARG occupancy requires FP16 / BF16 input.");
    }
}

// ---- The carried-index OneWindow ARG twin (see oneWindowCarriedArgKernel).
bool fp32OneWindowCarriedArgSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size,
                                     const OneWindowUnit& unit, uint32_t warps_per_output) {
    if (outer_size == 0 || reduction_size == 0 || !fp32OneWindowUnitPlan(inner_size, unit) ||
        !oneWindowValidWarpsPerOutput(unit.outputs_per_warp, warps_per_output)) {
        return false;
    }
    constexpr uint64_t MaxIndex = std::numeric_limits<uint32_t>::max();
    return outer_size <= MaxIndex && reduction_size <= MaxIndex &&
           outer_size * reduction_size * inner_size <= MaxIndex / 4;
}

void launchFp32OneWindowCarriedArg(const StridedSpan& input,
                                   const StridedSpan& carried_index,
                                   const StridedSpan* value_output,
                                   const StridedSpan* index_output,
                                   const ReductionGeometry& geometry,
                                   const OneWindowUnit& unit,
                                   uint32_t warps_per_output,
                                   bool maximum,
                                   uint64_t row_stride,
                                   Stream& stream) {
    const uint64_t outputs = geometry.outer_size * geometry.inner_size;
    const uint64_t elements = geometry.outer_size * geometry.reduction_size * geometry.inner_size;
    const auto plan = fp32OneWindowUnitPlan(geometry.inner_size, unit);
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
    const uint64_t outputs_per_block = (SHALLOW_COMPLETE_WARPS / warps_per_output) * unit.outputs_per_warp;
    const uint64_t block_count = std::min<uint64_t>(ceilDivideU64(geometry.outer_size, outputs_per_block),
                                                    PERSISTENT_MAX_GRID_BLOCKS);
    const bool wide = carried_index.getDataType() == DataType::UINT64;
    const bool launched = dispatchFp32OneWindow(
        unit, [&]<typename InputT, int OutputsPerWarp, int PacketBytes, int MaxComponentsPerLane, bool LoadAhead>() {
            static_assert(std::is_same_v<InputT, float> && PacketBytes == 16);
            const auto launch = [&]<typename IndexT>() {
                oneWindowCarriedArgKernel<OutputsPerWarp, MaxComponentsPerLane, LoadAhead, IndexT>
                    <<<static_cast<unsigned int>(block_count), SHALLOW_COMPLETE_THREADS, 0, stream.getStream()>>>(
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

KernelOccupancyInfo queryFp32OneWindowCarriedArgOccupancy(uint64_t inner_size,
                                                          const OneWindowUnit& unit,
                                                          uint32_t warps_per_output,
                                                          DataType index_dtype) {
    if (!fp32OneWindowUnitPlan(inner_size, unit) ||
        !oneWindowValidWarpsPerOutput(unit.outputs_per_warp, warps_per_output) ||
        (index_dtype != DataType::UINT32 && index_dtype != DataType::UINT64)) {
        throw std::logic_error("FP32 OneWindow carried ARG occupancy received unsupported ownership or index.");
    }
    KernelOccupancyInfo occupancy{};
    const bool wide = index_dtype == DataType::UINT64;
    const bool queried = dispatchFp32OneWindow(
        unit, [&]<typename InputT, int OutputsPerWarp, int PacketBytes, int MaxComponentsPerLane, bool LoadAhead>() {
            static_assert(std::is_same_v<InputT, float> && PacketBytes == 16);
            occupancy = wide ? queryShallowCompleteOccupancy(
                                   oneWindowCarriedArgKernel<OutputsPerWarp, MaxComponentsPerLane, LoadAhead,
                                                             uint64_t>)
                             : queryShallowCompleteOccupancy(
                                   oneWindowCarriedArgKernel<OutputsPerWarp, MaxComponentsPerLane, LoadAhead,
                                                             uint32_t>);
        });
    if (!queried) {
        throw std::logic_error("FP32 OneWindow carried ARG occupancy has no kernel for this request.");
    }
    return occupancy;
}

bool fp8OneWindowArgSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size,
                             const OneWindowUnit& unit, uint32_t warps_per_output) {
    if (outer_size == 0 || reduction_size == 0 || reduction_size >= FP8_ARG_MAX_LANE_ROWS ||
        !oneWindowUnitPlan(inner_size, unit) ||
        !oneWindowValidWarpsPerOutput(unit.outputs_per_warp, warps_per_output)) {
        return false;
    }
    constexpr uint64_t MaxIndex = std::numeric_limits<uint32_t>::max();
    return outer_size <= MaxIndex && outer_size * reduction_size * inner_size <= MaxIndex;
}

void launchFp8OneWindowArg(const StridedSpan& input,
                           const StridedSpan* value_output,
                           const StridedSpan* index_output,
                           const ReductionGeometry& geometry,
                           const OneWindowUnit& unit,
                           uint32_t warps_per_output,
                           bool maximum,
                           uint64_t domain_stride,
                           Stream& stream) {
    const DataType input_dtype = input.getDataType();
    const uint64_t outputs = geometry.outer_size * geometry.inner_size;
    // The plan of the base's alignment (rows per group leave room for the prefix off a 16-byte boundary).
    const bool aligned_base = oneWindowBaseIsAligned(input);
    const auto plan = oneWindowUnitPlan(geometry.inner_size, unit, 1, aligned_base);
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
    if (!aligned_base && !oneWindowUnalignedBaseFits(input, 1, input.getTotalNumElements())) {
        throw std::logic_error("FP8 OneWindow ARG at an unaligned base holds at most 2^32 - 16 input bytes.");
    }
    const Fp8ArgOutputs arg_outputs{
        value_output == nullptr ? nullptr : value_output->getMemPtr<void>(),
        value_output == nullptr ? DataType::FP32 : value_output->getDataType(),
        index_output == nullptr ? nullptr : index_output->getMemPtr<void>(),
        index_output == nullptr ? DataType::UINT32 : index_output->getDataType(),
        domain_stride,
        fp8ArgFold(maximum, input_dtype == DataType::FP8_E5M2),
        input_dtype == DataType::FP8_E5M2};
    const uint64_t outputs_per_block = (SHALLOW_COMPLETE_WARPS / warps_per_output) * unit.outputs_per_warp;
    const uint64_t block_count = std::min<uint64_t>(ceilDivideU64(geometry.outer_size, outputs_per_block),
                                                    PERSISTENT_MAX_GRID_BLOCKS);
    const bool launched = dispatchOneWindowArg(
        unit, [&]<int OutputsPerWarp, int PacketBytes, int MaxComponentsPerLane, bool LoadAhead>() {
            fp8OneWindowArgKernel<OutputsPerWarp, PacketBytes, MaxComponentsPerLane, LoadAhead>
                <<<static_cast<unsigned int>(block_count), SHALLOW_COMPLETE_THREADS, 0, stream.getStream()>>>(
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
}

}  // namespace ThorImplementation::ReductionInternal
