#include "Utilities/TensorOperations/Reduction/Kernels/KernelLaunchers.h"
#include "Utilities/TensorOperations/Reduction/Kernels/ReductionOperation.cuh"
#include "Utilities/TensorOperations/Reduction/Arg/Fp8ArgKey.h"
#include "Utilities/TensorOperations/Reduction/Arg/Fp32ArgKey.cuh"
#include "Utilities/TensorOperations/Reduction/Kernels/KernelCommon.cuh"
#include "Utilities/TensorOperations/Reduction/RK/Fp8/Fp8DeepRSmallKSelector.h"

#include "Utilities/Exceptions.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace ThorImplementation::ReductionInternal {
namespace {

// The DeepRSmallK stages (FP8 / FP16 / BF16 / FP32, VALUE and ARG) are this CUDA translation unit's; the other RK
// families and R / KR are in this directory (RKOneWindow.cu, RKRuntimeK.cu, RKAlignedK.cu, KR.cu). DeepRSmallK owns its
// row depths and K component layout independently of the generic RK policy.
constexpr int DEEP_R_SMALL_K_WARPS = 8;
constexpr int DEEP_R_SMALL_K_THREADS = DEEP_R_SMALL_K_WARPS * TILED_REDUCTION_WARP_THREADS;
static_assert(DEEP_R_SMALL_K_THREADS == 256);

template <typename ReductionOpT>
struct DeepRSmallKReductionIdentityFp32;

template <>
struct DeepRSmallKReductionIdentityFp32<cuda::std::plus<float>> {
    static constexpr float value = 0.0f;
};

template <>
struct DeepRSmallKReductionIdentityFp32<cuda::std::multiplies<float>> {
    static constexpr float value = 1.0f;
};

template <>
struct DeepRSmallKReductionIdentityFp32<PropagatingMinimumFp32> {
    static constexpr float value = std::numeric_limits<float>::infinity();
};

template <>
struct DeepRSmallKReductionIdentityFp32<PropagatingMaximumFp32> {
    static constexpr float value = -std::numeric_limits<float>::infinity();
};

// FP8 FlatRows calibration exposes 4/8/16-byte global packets while preserving one coherent eight-warp topology.
// Each packet width adaptively fills one warp-wide packet window and reserves its worst-case alignment prefix, so the
// logical group limits are p4=125, p8=249, and p16=497 FP8 bytes. K<=8 retains the dedicated same-component
// cooperative ownership specialization; K>=9 keeps simple lane ownership with multiple components per lane above K=32.
// Packet-width variation is calibration-only; production remains on the established p16 path.
constexpr int DEEP_R_SMALL_K_PACKETS_PER_WARP = TILED_REDUCTION_WARP_THREADS;
constexpr int DEEP_R_SMALL_K_MAX_K = 256;
static_assert(DEEP_R_SMALL_K_MAX_K == DEEP_R_SMALL_K_THREADS);

template <int PacketBytes>
inline constexpr int deepRSmallKPacketWindowBytes = PacketBytes * DEEP_R_SMALL_K_PACKETS_PER_WARP;

template <int PacketBytes>
inline constexpr int deepRSmallKMaxLogicalGroupBytes = deepRSmallKPacketWindowBytes<PacketBytes> - (PacketBytes - 1);

static_assert(deepRSmallKPacketWindowBytes<4> == 128 && deepRSmallKMaxLogicalGroupBytes<4> == 125);
static_assert(deepRSmallKPacketWindowBytes<8> == 256 && deepRSmallKMaxLogicalGroupBytes<8> == 249);
static_assert(deepRSmallKPacketWindowBytes<16> == 512 && deepRSmallKMaxLogicalGroupBytes<16> == 497);

template <typename InputT>
[[nodiscard]] __device__ inline InputT fp8FromBits(uint8_t bits) {
    static_assert(sizeof(InputT) == 1);
    if constexpr (std::is_same_v<InputT, __nv_fp8_e4m3>) {
        return ThorLowPrecision::fp8E4M3FromBits(static_cast<__nv_fp8_storage_t>(bits));
    } else {
        static_assert(std::is_same_v<InputT, __nv_fp8_e5m2>);
        return ThorLowPrecision::fp8E5M2FromBits(static_cast<__nv_fp8_storage_t>(bits));
    }
}

static_assert(RK::RK_MAX_GRID_EXTENT == REDUCTION_WORK_GRID_MAX_EXTENT,
              "RKLaunchLimits.h must restate reductionWorkGrid's extent cap");

template <typename InputT>
struct Fp8TypedFormatDecoder {
    __device__ float operator()(uint8_t bits) const {
        return ToFp32<InputT>{}(fp8FromBits<InputT>(bits));
    }
};

// FP32 DeepRSmallK reads staged floats directly (decodeDeepRSmallKGroup's FP32 branch); this empty decoder type only
// names the FP32 instantiations and is never called.
struct Fp32StagedDecoder {};
// Likewise for FP16 / BF16 (staged half-words, widened by deepRStagedElement).
struct F16StagedDecoder {};

// DeepRSmallK inputs read per staged element (FP32, FP16 / BF16), not as FP8 bytes.
template <typename InputT>
inline constexpr bool DEEP_R_PER_ELEMENT_INPUT = std::is_same_v<InputT, float> || isF16InputType<InputT>;

template <typename InputT>
__device__ __forceinline__ float deepRStagedElement(const uint8_t* staged_bytes, uint32_t staged_index) {
    if constexpr (std::is_same_v<InputT, float>) {
        return reinterpret_cast<const float*>(staged_bytes)[staged_index];
    } else {
        return widenF16Bits<InputT>(reinterpret_cast<const uint16_t*>(staged_bytes)[staged_index]);
    }
}

// The DeepRSmallK warp-group window. Each warp stages one group of whole rows (group_rows * K bytes plus an alignment
// prefix < PacketBytes) as one packet per lane. The group loops below are software-pipelined: the packet of the warp's
// next group is loaded into a register before the current group is decoded from shared memory, so one group's global
// load is always in flight while the warp works. Row order per component is unchanged.
template <int PacketBytes>
struct DeepRSmallKGroupWindow {
    const RawVectorPacket<PacketBytes>* aligned_begin;
    uint32_t prefix_elements;
    uint32_t group_rows;
    uint32_t packet_count;
};

template <int PacketBytes, typename InputT>
[[nodiscard]] __device__ __forceinline__ DeepRSmallKGroupWindow<PacketBytes> deepRSmallKGroupWindow(
    const InputT* shard_input, uint32_t group_row, uint32_t shard_rows, uint32_t rows_per_warp_group,
    uint32_t inner_size_32) {
    static_assert(sizeof(InputT) == 1 || DEEP_R_PER_ELEMENT_INPUT<InputT>);  // FP32, FP16 / BF16
    constexpr uint32_t ElementBytes = static_cast<uint32_t>(sizeof(InputT));
    DeepRSmallKGroupWindow<PacketBytes> window;
    window.group_rows = rows_per_warp_group < shard_rows - group_row ? rows_per_warp_group : shard_rows - group_row;
    const uintptr_t group_address = reinterpret_cast<uintptr_t>(shard_input + group_row * inner_size_32);
    window.prefix_elements = static_cast<uint32_t>(group_address & uintptr_t{PacketBytes - 1}) / ElementBytes;
    window.aligned_begin =
        reinterpret_cast<const RawVectorPacket<PacketBytes>*>(group_address & ~uintptr_t{PacketBytes - 1});
    window.packet_count =
        ((window.prefix_elements + window.group_rows * inner_size_32) * ElementBytes + PacketBytes - 1) / PacketBytes;
    return window;
}

// Packed FP8x4 decode in the DeepRSmallK first stage. Lanes owning four or more components decode four components of
// one row per decodeFp8x4ToFp32 (rows still in increasing order per component: bit-identical to decoding one staged
// byte per element). Against the scalar loop (MIN/MAX, cold L2): c8 0.83x (rows1024 0.69x), c5 0.96x, c4 0.985x;
// DeepRSmallK SASS -4%. Packing four rows of one component for one or two components per lane lost (c1 1.034x, c2
// 1.010x), so those keep the scalar loop. THOR_FP8_DEEP_R_VALUE_SCALAR_DECODE=1 restores it everywhere.
#ifndef THOR_FP8_DEEP_R_VALUE_SCALAR_DECODE
#define THOR_FP8_DEEP_R_VALUE_SCALAR_DECODE 0
#endif
template <typename InputT, int MaxComponentsPerLane, typename ReductionOpT, typename InputTransformT, typename DecoderT>
__device__ __forceinline__ void decodeDeepRSmallKGroup(float (&local)[MaxComponentsPerLane],
                                                       const uint8_t* staged_bytes,
                                                       uint32_t prefix_elements,
                                                       uint32_t lane,
                                                       uint32_t group_rows,
                                                       uint32_t inner_size_32,
                                                       const ReductionOpT& reduction_op,
                                                       const InputTransformT& input_transform,
                                                       const DecoderT& decoder) {
    constexpr bool Packed = !THOR_FP8_DEEP_R_VALUE_SCALAR_DECODE &&
                            std::is_same_v<DecoderT, Fp8TypedFormatDecoder<InputT>>;
    static_cast<void>(decoder);
    if constexpr (DEEP_R_PER_ELEMENT_INPUT<InputT>) {
        // FP32, one staged float per element, rows of each component in increasing order. FP16 / BF16 the same way, one
        // staged half-word per element.
#pragma unroll
        for (int slot = 0; slot < MaxComponentsPerLane; ++slot) {
            const uint32_t component = lane + static_cast<uint32_t>(slot) * TILED_REDUCTION_WARP_THREADS;
            if (component < inner_size_32) {
                uint32_t staged_index = prefix_elements + component;
                for (uint32_t row = 0; row < group_rows; ++row, staged_index += inner_size_32) {
                    local[slot] = reduction_op(local[slot],
                                               input_transform(deepRStagedElement<InputT>(staged_bytes, staged_index)));
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

// The shard depth is a runtime operand. It only places and clamps each shard; the row loops run on the runtime
// rows_per_warp_group / shard_rows, and compiled depths produced identical SASS apart from that constant. The depths
// are powers of two and the kernel receives log2(depth): a runtime 64-bit multiply for the shard origin cost 2-4% on
// shards of 4 KB or less, where the per-shard latency before the first load dominates; a shift keeps the compiled
// kernels' instruction shape. The launcher validates the depth against Fp8DeepRSmallKPolicy::compiled_rows.
// FixedRowsPerShardLog2 >= 0 compiles the depth back in (rows_per_shard_log2 is then ignored). Only the one-component
// kernel (K 9..32) uses it, at every depth: with the runtime depth it lost 4.8% at rows64 and 1.9% at rows128, the only
// ownership whose loss survived the shift. Every other ownership kind keeps one kernel.
template <typename InputT, typename ReductionOpT, typename InputTransformT, int PacketBytes, int MaxComponentsPerLane,
          int FixedRowsPerShardLog2 = -1, typename DecoderT = Fp8TypedFormatDecoder<InputT>>
__global__ void deepRSmallKValueKernel(const InputT* input,
                                       float* fp32_partials,
                                       uint64_t outer_size,
                                       uint64_t reduction_size,
                                       uint64_t inner_size,
                                       uint64_t shards_per_output,
                                       uint32_t rows_per_shard_log2,
                                       uint32_t rows_per_warp_group,
                                       InputTransformT input_transform, DecoderT decoder = DecoderT{}) {
    // FP32: components <= 4; FP16 / BF16: every ownership, K <= 249.
    static_assert(sizeof(InputT) == 1 || DEEP_R_PER_ELEMENT_INPUT<InputT>);
    static_assert(sizeof(InputT) == 1 || MaxComponentsPerLane <= 4 || isF16InputType<InputT>);
    static_assert(PacketBytes == 4 || PacketBytes == 8 || PacketBytes == 16);
    static_assert(MaxComponentsPerLane == 1 || MaxComponentsPerLane == 2 || MaxComponentsPerLane == 4 || MaxComponentsPerLane == 5 ||
                  MaxComponentsPerLane == 8);
    constexpr uint32_t ElementBytes = static_cast<uint32_t>(sizeof(InputT));
    constexpr int MaxSupportedK = TILED_REDUCTION_WARP_THREADS * MaxComponentsPerLane;
    constexpr float Init = DeepRSmallKReductionIdentityFp32<ReductionOpT>::value;
    static_assert(MaxSupportedK <= DEEP_R_SMALL_K_MAX_K);
    static_assert(FixedRowsPerShardLog2 == -1 || (FixedRowsPerShardLog2 >= 3 && FixedRowsPerShardLog2 <= 14));
    const uint32_t shard_log2 =
        FixedRowsPerShardLog2 >= 0 ? static_cast<uint32_t>(FixedRowsPerShardLog2) : rows_per_shard_log2;

    __shared__ RawVectorPacket<PacketBytes> staged_packets[DEEP_R_SMALL_K_WARPS][DEEP_R_SMALL_K_PACKETS_PER_WARP];
    __shared__ float warp_partials[DEEP_R_SMALL_K_WARPS][MaxSupportedK];

    const ReductionOpT reduction_op{};
    const int warp = static_cast<int>(threadIdx.x) / TILED_REDUCTION_WARP_THREADS;
    const int lane = static_cast<int>(threadIdx.x) % TILED_REDUCTION_WARP_THREADS;

    // DeepRSmallK's launcher guarantees 1 <= inner_size <= 256, and rows_per_shard <= 16384, so every offset inside one
    // shard (< 2^22 bytes) is 32-bit. One warp stages at most 32 * PacketBytes <= 512 bytes, including an alignment
    // prefix < PacketBytes. The shard base pointer is formed once per work item in 64 bits.
    const uint32_t inner_size_32 = static_cast<uint32_t>(inner_size);

    // rows_per_warp_group is computed by the launcher: fill toward the packet window, but never choose a group so deep
    // that fewer than eight warps can participate in a full rows_per_shard shard.
    const uint32_t rows_per_block_round = rows_per_warp_group * DEEP_R_SMALL_K_WARPS;

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
            DeepRSmallKGroupWindow<PacketBytes> window{};
            RawVectorPacket<PacketBytes> pending{};
            if (group_row < shard_rows) {
                window = deepRSmallKGroupWindow<PacketBytes>(shard_input, group_row, shard_rows, rows_per_warp_group,
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
                    window = deepRSmallKGroupWindow<PacketBytes>(shard_input, next_group_row, shard_rows,
                                                                 rows_per_warp_group, inner_size_32);
                    if (static_cast<uint32_t>(lane) < window.packet_count) {
                        pending = window.aligned_begin[lane];
                    }
                }

                const auto* staged_bytes = reinterpret_cast<const uint8_t*>(&staged_packets[warp][0]);
                decodeDeepRSmallKGroup<InputT, MaxComponentsPerLane>(local, staged_bytes, prefix_elements,
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
                decodeDeepRSmallKGroup<InputT, MaxComponentsPerLane>(local, staged_bytes, prefix_elements,
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
            for (int source_warp = 0; source_warp < DEEP_R_SMALL_K_WARPS; ++source_warp) {
                reduced = reduction_op(reduced, warp_partials[source_warp][component]);
            }
            fp32_partials[(outer_index * shards_per_output + shard) * inner_size + component] = reduced;
        }
        __syncthreads();
    }
}

// Runtime log2(rows per shard), as in deepRSmallKValueKernel.
template <typename InputT, typename ReductionOpT, typename InputTransformT, int PacketBytes, int LanesPerComponent,
          typename DecoderT = Fp8TypedFormatDecoder<InputT>>
__global__ void deepRSmallKValueTinyKKernel(const InputT* input,
                                            float* fp32_partials,
                                            uint64_t outer_size,
                                            uint64_t reduction_size,
                                            uint64_t inner_size,
                                            uint64_t shards_per_output,
                                            uint32_t rows_per_shard_log2,
                                            uint32_t rows_per_warp_group,
                                                                InputTransformT input_transform, DecoderT decoder = DecoderT{}) {
    static_assert(sizeof(InputT) == 1 || DEEP_R_PER_ELEMENT_INPUT<InputT>);  // FP32, FP16 / BF16
    constexpr uint32_t ElementBytes = static_cast<uint32_t>(sizeof(InputT));
    static_assert(PacketBytes == 4 || PacketBytes == 8 || PacketBytes == 16);
    static_assert(LanesPerComponent == 4 || LanesPerComponent == 8);
    static_assert(TILED_REDUCTION_WARP_THREADS % LanesPerComponent == 0);
    constexpr int MaxSupportedK = TILED_REDUCTION_WARP_THREADS / LanesPerComponent;
    constexpr float Init = DeepRSmallKReductionIdentityFp32<ReductionOpT>::value;
    static_assert(MaxSupportedK == 4 || MaxSupportedK == 8);

    __shared__ RawVectorPacket<PacketBytes> staged_packets[DEEP_R_SMALL_K_WARPS][DEEP_R_SMALL_K_PACKETS_PER_WARP];
    __shared__ float cooperative_lane_partials[DEEP_R_SMALL_K_WARPS][TILED_REDUCTION_WARP_THREADS];
    __shared__ float warp_partials[DEEP_R_SMALL_K_WARPS][MaxSupportedK];

    const ReductionOpT reduction_op{};
    const int warp = static_cast<int>(threadIdx.x) / TILED_REDUCTION_WARP_THREADS;
    const int lane = static_cast<int>(threadIdx.x) % TILED_REDUCTION_WARP_THREADS;
    const int component = lane / LanesPerComponent;
    const int component_lane = lane % LanesPerComponent;

    // inner_size <= 8 here and rows_per_shard <= 16384, so every offset inside one shard is 32-bit. One warp stages at most
    // 32 * PacketBytes <= 512 bytes, including an alignment prefix < PacketBytes. The shard base pointer is formed once
    // per work item in 64 bits.
    const uint32_t inner_size_32 = static_cast<uint32_t>(inner_size);

    // rows_per_warp_group is computed by the launcher (see deepRSmallKRowsPerWarpGroup).
    const uint32_t rows_per_block_round = rows_per_warp_group * DEEP_R_SMALL_K_WARPS;

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
                    if constexpr (DEEP_R_PER_ELEMENT_INPUT<InputT>) {
                        static_cast<void>(decoder);
                        value = deepRStagedElement<InputT>(staged_bytes, staged_index);  // FP32 or 2-byte
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
            for (int source_warp = 0; source_warp < DEEP_R_SMALL_K_WARPS; ++source_warp) {
                reduced = reduction_op(reduced, warp_partials[source_warp][output_component]);
            }
            fp32_partials[(outer_index * shards_per_output + shard) * inner_size + output_component] = reduced;
        }
        __syncthreads();
    }
}

// Packed DeepRSmallK shard depths: a power of two from the eight warps up to 16384 rows (32-bit in-shard offsets).
[[nodiscard]] uint32_t deepRSmallKCheckedRowsPerShard(uint64_t rows_per_shard) {
    if (rows_per_shard < static_cast<uint64_t>(DEEP_R_SMALL_K_WARPS) || rows_per_shard > 16384 ||
        (rows_per_shard & (rows_per_shard - 1)) != 0) {
        throw std::logic_error("Packed FP8 narrow-flat launch received an unsupported rows-per-shard depth.");
    }
    return static_cast<uint32_t>(rows_per_shard);
}

[[nodiscard]] uint32_t deepRSmallKRowsPerShardLog2(uint32_t rows_per_shard) {
    uint32_t log2 = 0;
    while ((uint32_t{1} << log2) < rows_per_shard) ++log2;
    return log2;
}

// Rows per warp group for the packed DeepRSmallK kernels, computed once on the host: fill toward the packet window, but
// never choose a group so deep that fewer than eight warps can participate in a full rows_per_shard shard.
// ElementBytes 4 for FP32, whose window holds (32 * PacketBytes - (PacketBytes - 4)) / 4 = 125 elements at P16 after a
// worst-case prefix (FP8: 497).
template <int PacketBytes, uint32_t ElementBytes = 1>
inline constexpr uint32_t deepRSmallKMaxLogicalGroupElements =
    (static_cast<uint32_t>(deepRSmallKPacketWindowBytes<PacketBytes>) - (PacketBytes - ElementBytes)) / ElementBytes;
static_assert(deepRSmallKMaxLogicalGroupElements<16> == 497 && deepRSmallKMaxLogicalGroupElements<16, 4> == 125);

template <int PacketBytes, uint32_t ElementBytes = 1>
[[nodiscard]] uint32_t deepRSmallKRowsPerWarpGroup(uint64_t inner_size, uint32_t rows_per_shard) {
    constexpr uint32_t MaxElements = deepRSmallKMaxLogicalGroupElements<PacketBytes, ElementBytes>;
    const uint32_t max_rows_per_warp_from_shard = rows_per_shard / static_cast<uint32_t>(DEEP_R_SMALL_K_WARPS);
    if (inner_size == 0 || inner_size > static_cast<uint64_t>(MaxElements)) {
        throw std::logic_error("Packed FP8 narrow-flat launch received K outside one packet window.");
    }
    const uint32_t rows_from_packet_window = MaxElements / static_cast<uint32_t>(inner_size);
    return rows_from_packet_window < max_rows_per_warp_from_shard ? rows_from_packet_window : max_rows_per_warp_from_shard;
}

template <typename InputT, typename ReductionOpT, typename InputTransformT, int PacketBytes, int LanesPerComponent,
          typename DecoderT = Fp8TypedFormatDecoder<InputT>>
void launchDeepRSmallKTinyKForTransform(
    const StridedSpan& input, const StridedSpan& fp32_partials, const ReductionGeometry& geometry, uint64_t rows_per_shard,
    uint64_t shards_per_output, cudaStream_t stream,
    InputTransformT input_transform = InputTransformT{}, DecoderT decoder = DecoderT{}) {
    static_assert(sizeof(InputT) == 1 || DEEP_R_PER_ELEMENT_INPUT<InputT>);  // FP32, FP16 / BF16
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
    const uint32_t shard_depth = deepRSmallKCheckedRowsPerShard(rows_per_shard);
    const uint64_t bytes_per_work_item =
        std::min<uint64_t>(shard_depth, geometry.reduction_size) * geometry.inner_size * sizeof(InputT);
    const dim3 grid_blocks = reductionWorkGrid(shards_per_output, 1, geometry.outer_size,
                                               bytes_per_work_item <= 256 ? REDUCTION_PERSISTENT_MAX_BLOCKS : 0);
    const uint32_t rows_per_warp_group =
        deepRSmallKRowsPerWarpGroup<PacketBytes, static_cast<uint32_t>(sizeof(InputT))>(geometry.inner_size,
                                                                                           shard_depth);

    deepRSmallKValueTinyKKernel<InputT,
                                ReductionOpT,
                                InputTransformT,
                                PacketBytes,
                                LanesPerComponent,
                                DecoderT>
        <<<grid_blocks, DEEP_R_SMALL_K_THREADS, 0, stream>>>(([](const StridedSpan& src) -> const InputT* {
            if constexpr (std::is_same_v<InputT, uint8_t>) return reinterpret_cast<const InputT*>(src.getMemPtr<void>());
            else return src.getMemPtr<InputT>();
        })(input),
                                                             fp32_partials.getMemPtr<float>(),
                                                             geometry.outer_size,
                                                             geometry.reduction_size,
                                                             geometry.inner_size,
                                                             shards_per_output,
                                                             deepRSmallKRowsPerShardLog2(shard_depth),
                                                             rows_per_warp_group,
                                                             input_transform, decoder);
    CUDA_CHECK(cudaGetLastError());
}

template <typename InputT, typename ReductionOpT, typename InputTransformT, int PacketBytes, int MaxComponentsPerLane,
          int FixedRowsPerShardLog2 = -1, typename DecoderT = Fp8TypedFormatDecoder<InputT>>
void launchDeepRSmallKForTransform(
    const StridedSpan& input, const StridedSpan& fp32_partials, const ReductionGeometry& geometry, uint64_t rows_per_shard,
    uint64_t shards_per_output, cudaStream_t stream,
    InputTransformT input_transform = InputTransformT{}, DecoderT decoder = DecoderT{}) {
    static_assert(sizeof(InputT) == 1 || DEEP_R_PER_ELEMENT_INPUT<InputT>);  // FP32, FP16 / BF16
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
    const uint32_t shard_depth = deepRSmallKCheckedRowsPerShard(rows_per_shard);
    const uint64_t bytes_per_work_item =
        std::min<uint64_t>(shard_depth, geometry.reduction_size) * geometry.inner_size * sizeof(InputT);
    const dim3 grid_blocks = reductionWorkGrid(shards_per_output, 1, geometry.outer_size,
                                               bytes_per_work_item <= 256 ? REDUCTION_PERSISTENT_MAX_BLOCKS : 0);
    const uint32_t rows_per_warp_group =
        deepRSmallKRowsPerWarpGroup<PacketBytes, static_cast<uint32_t>(sizeof(InputT))>(geometry.inner_size,
                                                                                           shard_depth);
    const uint32_t shard_log2 = deepRSmallKRowsPerShardLog2(shard_depth);
    if constexpr (FixedRowsPerShardLog2 >= 0) {
        if (shard_log2 != static_cast<uint32_t>(FixedRowsPerShardLog2)) {
            throw std::logic_error("Packed FP8 narrow-flat launch depth does not match its compiled depth.");
        }
    }

    deepRSmallKValueKernel<InputT, ReductionOpT, InputTransformT, PacketBytes, MaxComponentsPerLane,
                           FixedRowsPerShardLog2, DecoderT>
        <<<grid_blocks, DEEP_R_SMALL_K_THREADS, 0, stream>>>(([](const StridedSpan& src) -> const InputT* {
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

template <typename InputT, int PacketBytes, int LanesPerComponent, typename InputTransformT = IdentityFp32,
          typename DecoderT = Fp8TypedFormatDecoder<InputT>, typename ReductionOpT = cuda::std::plus<float>>
[[nodiscard]] KernelOccupancyInfo queryDeepRSmallKTinyKOccupancy() {
    static_assert(sizeof(InputT) == 1 || DEEP_R_PER_ELEMENT_INPUT<InputT>);
    static_assert(PacketBytes == 4 || PacketBytes == 8 || PacketBytes == 16);
    static_assert(LanesPerComponent == 4 || LanesPerComponent == 8);
    const auto kernel = deepRSmallKValueTinyKKernel<InputT,
                                                    ReductionOpT,
                                                    InputTransformT,
                                                    PacketBytes,
                                                    LanesPerComponent,
                                                                         DecoderT>;
    cudaFuncAttributes attributes{};
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
    int active_blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks, kernel, DEEP_R_SMALL_K_THREADS, 0));
    return KernelOccupancyInfo{attributes.numRegs, attributes.sharedSizeBytes, 0, active_blocks,
                               active_blocks * DEEP_R_SMALL_K_WARPS};
}

template <typename InputT, int PacketBytes, int MaxComponentsPerLane, typename InputTransformT = IdentityFp32,
          typename DecoderT = Fp8TypedFormatDecoder<InputT>, typename ReductionOpT = cuda::std::plus<float>,
          int FixedRowsPerShardLog2 = -1>
[[nodiscard]] KernelOccupancyInfo queryDeepRSmallKOccupancy() {
    static_assert(sizeof(InputT) == 1 || DEEP_R_PER_ELEMENT_INPUT<InputT>);
    const auto kernel = deepRSmallKValueKernel<InputT,
                                               ReductionOpT,
                                               InputTransformT,
                                               PacketBytes,
                                               MaxComponentsPerLane,
                                               FixedRowsPerShardLog2,
                                               DecoderT>;
    cudaFuncAttributes attributes{};
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
    int active_blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks, kernel, DEEP_R_SMALL_K_THREADS, 0));
    return KernelOccupancyInfo{attributes.numRegs, attributes.sharedSizeBytes, 0, active_blocks,
                               active_blocks * DEEP_R_SMALL_K_WARPS};
}

// DeepRSmallK P16 geometry helpers: five depths (a runtime operand, validated against
// Fp8DeepRSmallKPolicy::compiled_rows, and compiled in for the one-component ownership only) and seven K ownership
// choices, shared by all four typed FP32 reduction operations.
[[nodiscard]] uint32_t deepRSmallKRowDepth(uint64_t rows) {
    const auto& depths = RK::Fp8::Fp8DeepRSmallKPolicy::compiled_rows;
    if (std::find(depths.begin(), depths.end(), rows) == depths.end()) {
        throw std::logic_error("DeepRSmallK retained kernel requires a retained row depth.");
    }
    return static_cast<uint32_t>(rows);
}

// One-component kernels keep a compiled depth (see deepRSmallKValueKernel).
template <typename Callback>
void dispatchDeepRSmallKFixedDepth(uint32_t depth, Callback&& callback) {
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
void dispatchDeepRSmallKRows(const StridedSpan& input, const StridedSpan& partials,
                                  const ReductionGeometry& geometry, uint64_t rows,
                                  uint64_t shards, cudaStream_t stream, TransformT transform) {
    const uint32_t depth = deepRSmallKRowDepth(rows);
    if (geometry.inner_size <= 4)
        launchDeepRSmallKTinyKForTransform<InputT, ReductionOpT, TransformT, 16, 8>(
            input, partials, geometry, depth, shards, stream, transform);
    else if (geometry.inner_size <= 8)
        launchDeepRSmallKTinyKForTransform<InputT, ReductionOpT, TransformT, 16, 4>(
            input, partials, geometry, depth, shards, stream, transform);
    else if (geometry.inner_size <= 32)
        dispatchDeepRSmallKFixedDepth(depth, [&]<int DepthLog2>() {
            launchDeepRSmallKForTransform<InputT, ReductionOpT, TransformT, 16, 1, DepthLog2>(
                input, partials, geometry, depth, shards, stream, transform);
        });
    else if (geometry.inner_size <= 64)
        launchDeepRSmallKForTransform<InputT, ReductionOpT, TransformT, 16, 2>(
            input, partials, geometry, depth, shards, stream, transform);
    else if (geometry.inner_size <= 128)
        launchDeepRSmallKForTransform<InputT, ReductionOpT, TransformT, 16, 4>(
            input, partials, geometry, depth, shards, stream, transform);
    else if (geometry.inner_size <= 160)
        launchDeepRSmallKForTransform<InputT, ReductionOpT, TransformT, 16, 5>(
            input, partials, geometry, depth, shards, stream, transform);
    else
        launchDeepRSmallKForTransform<InputT, ReductionOpT, TransformT, 16, 8>(
            input, partials, geometry, depth, shards, stream, transform);
}

// The row depth selects a kernel only for the one-component (K 9..32) ownership; it is always validated.
template <typename InputT, typename TransformT, typename ReductionOpT = cuda::std::plus<float>>
KernelOccupancyInfo queryDeepRSmallKRows(uint64_t rows, uint64_t k) {
    static_cast<void>(deepRSmallKRowDepth(rows));
    if (k <= 4) return queryDeepRSmallKTinyKOccupancy<InputT, 16, 8, TransformT, Fp8TypedFormatDecoder<InputT>, ReductionOpT>();
    if (k <= 8) return queryDeepRSmallKTinyKOccupancy<InputT, 16, 4, TransformT, Fp8TypedFormatDecoder<InputT>, ReductionOpT>();
    if (k <= 32) {
        KernelOccupancyInfo info{};
        dispatchDeepRSmallKFixedDepth(deepRSmallKRowDepth(rows), [&]<int DepthLog2>() {
            info = queryDeepRSmallKOccupancy<InputT, 16, 1, TransformT, Fp8TypedFormatDecoder<InputT>,
                                             ReductionOpT, DepthLog2>();
        });
        return info;
    }
    if (k <= 64) return queryDeepRSmallKOccupancy<InputT, 16, 2, TransformT, Fp8TypedFormatDecoder<InputT>, ReductionOpT>();
    if (k <= 128) return queryDeepRSmallKOccupancy<InputT, 16, 4, TransformT, Fp8TypedFormatDecoder<InputT>, ReductionOpT>();
    if (k <= 160) return queryDeepRSmallKOccupancy<InputT, 16, 5, TransformT, Fp8TypedFormatDecoder<InputT>, ReductionOpT>();
    return queryDeepRSmallKOccupancy<InputT, 16, 8, TransformT, Fp8TypedFormatDecoder<InputT>, ReductionOpT>();
}

// The FP32 DeepRSmallK ownerships by K, the FP8 kernels at 4-byte elements: a P16 window holds 125 FP32 after a
// worst-case prefix, so K <= 125 and lanes own at most four components (FP8's five- and eight-component kernels do not
// exist for FP32). The one-component kernel keeps its compiled depth, as for FP8.
static_assert(deepRSmallKMaxLogicalGroupElements<16, 4> == 125, "fp32DeepRSmallKSupports restates K <= 125");

template <typename TransformT, typename ReductionOpT>
void dispatchFp32DeepRSmallKRows(const StridedSpan& input, const StridedSpan& partials, const ReductionGeometry& geometry,
                                 uint64_t rows, uint64_t shards, cudaStream_t stream) {
    const uint32_t depth = deepRSmallKRowDepth(rows);
    const TransformT transform{};
    const Fp32StagedDecoder decoder{};
    if (geometry.inner_size <= 4)
        launchDeepRSmallKTinyKForTransform<float, ReductionOpT, TransformT, 16, 8, Fp32StagedDecoder>(
            input, partials, geometry, depth, shards, stream, transform, decoder);
    else if (geometry.inner_size <= 8)
        launchDeepRSmallKTinyKForTransform<float, ReductionOpT, TransformT, 16, 4, Fp32StagedDecoder>(
            input, partials, geometry, depth, shards, stream, transform, decoder);
    else if (geometry.inner_size <= 32)
        dispatchDeepRSmallKFixedDepth(depth, [&]<int DepthLog2>() {
            launchDeepRSmallKForTransform<float, ReductionOpT, TransformT, 16, 1, DepthLog2, Fp32StagedDecoder>(
                input, partials, geometry, depth, shards, stream, transform, decoder);
        });
    else if (geometry.inner_size <= 64)
        launchDeepRSmallKForTransform<float, ReductionOpT, TransformT, 16, 2, -1, Fp32StagedDecoder>(
            input, partials, geometry, depth, shards, stream, transform, decoder);
    else
        launchDeepRSmallKForTransform<float, ReductionOpT, TransformT, 16, 4, -1, Fp32StagedDecoder>(
            input, partials, geometry, depth, shards, stream, transform, decoder);
}

template <typename TransformT, typename ReductionOpT>
KernelOccupancyInfo queryFp32DeepRSmallKRows(uint64_t rows, uint64_t k) {
    static_cast<void>(deepRSmallKRowDepth(rows));
    if (k <= 4) return queryDeepRSmallKTinyKOccupancy<float, 16, 8, TransformT, Fp32StagedDecoder, ReductionOpT>();
    if (k <= 8) return queryDeepRSmallKTinyKOccupancy<float, 16, 4, TransformT, Fp32StagedDecoder, ReductionOpT>();
    if (k <= 32) {
        KernelOccupancyInfo info{};
        dispatchDeepRSmallKFixedDepth(deepRSmallKRowDepth(rows), [&]<int DepthLog2>() {
            info = queryDeepRSmallKOccupancy<float, 16, 1, TransformT, Fp32StagedDecoder, ReductionOpT,
                                             DepthLog2>();
        });
        return info;
    }
    if (k <= 64) return queryDeepRSmallKOccupancy<float, 16, 2, TransformT, Fp32StagedDecoder, ReductionOpT>();
    return queryDeepRSmallKOccupancy<float, 16, 4, TransformT, Fp32StagedDecoder, ReductionOpT>();
}

// The FP16 / BF16 DeepRSmallK ownerships by K: FP8's seven (its thresholds), the kernels at 2-byte elements. A P16
// window holds 249 2-byte elements after a worst-case prefix, so K <= 249 (FP8's eight-component kernel serves K
// 161..249).
inline constexpr uint64_t F16_DEEP_R_MAX_K = deepRSmallKMaxLogicalGroupElements<16, 2>;
static_assert(F16_DEEP_R_MAX_K == 249, "f16DeepRSmallKSupports restates K <= 249");

template <typename InputT, typename TransformT, typename ReductionOpT>
void dispatchF16DeepRSmallKRows(const StridedSpan& input, const StridedSpan& partials, const ReductionGeometry& geometry,
                                uint64_t rows, uint64_t shards, cudaStream_t stream) {
    static_assert(isF16InputType<InputT>);
    const uint32_t depth = deepRSmallKRowDepth(rows);
    const TransformT transform{};
    const F16StagedDecoder decoder{};
    const uint64_t k = geometry.inner_size;
    if (k <= 4)
        launchDeepRSmallKTinyKForTransform<InputT, ReductionOpT, TransformT, 16, 8, F16StagedDecoder>(
            input, partials, geometry, depth, shards, stream, transform, decoder);
    else if (k <= 8)
        launchDeepRSmallKTinyKForTransform<InputT, ReductionOpT, TransformT, 16, 4, F16StagedDecoder>(
            input, partials, geometry, depth, shards, stream, transform, decoder);
    else if (k <= 32)
        dispatchDeepRSmallKFixedDepth(depth, [&]<int DepthLog2>() {
            launchDeepRSmallKForTransform<InputT, ReductionOpT, TransformT, 16, 1, DepthLog2, F16StagedDecoder>(
                input, partials, geometry, depth, shards, stream, transform, decoder);
        });
    else if (k <= 64)
        launchDeepRSmallKForTransform<InputT, ReductionOpT, TransformT, 16, 2, -1, F16StagedDecoder>(
            input, partials, geometry, depth, shards, stream, transform, decoder);
    else if (k <= 128)
        launchDeepRSmallKForTransform<InputT, ReductionOpT, TransformT, 16, 4, -1, F16StagedDecoder>(
            input, partials, geometry, depth, shards, stream, transform, decoder);
    else if (k <= 160)
        launchDeepRSmallKForTransform<InputT, ReductionOpT, TransformT, 16, 5, -1, F16StagedDecoder>(
            input, partials, geometry, depth, shards, stream, transform, decoder);
    else
        launchDeepRSmallKForTransform<InputT, ReductionOpT, TransformT, 16, 8, -1, F16StagedDecoder>(
            input, partials, geometry, depth, shards, stream, transform, decoder);
}

template <typename InputT, typename TransformT, typename ReductionOpT>
KernelOccupancyInfo queryF16DeepRSmallKRows(uint64_t rows, uint64_t k) {
    static_cast<void>(deepRSmallKRowDepth(rows));
    if (k <= 4) return queryDeepRSmallKTinyKOccupancy<InputT, 16, 8, TransformT, F16StagedDecoder, ReductionOpT>();
    if (k <= 8) return queryDeepRSmallKTinyKOccupancy<InputT, 16, 4, TransformT, F16StagedDecoder, ReductionOpT>();
    if (k <= 32) {
        KernelOccupancyInfo info{};
        dispatchDeepRSmallKFixedDepth(deepRSmallKRowDepth(rows), [&]<int DepthLog2>() {
            info = queryDeepRSmallKOccupancy<InputT, 16, 1, TransformT, F16StagedDecoder, ReductionOpT,
                                             DepthLog2>();
        });
        return info;
    }
    if (k <= 64) return queryDeepRSmallKOccupancy<InputT, 16, 2, TransformT, F16StagedDecoder, ReductionOpT>();
    if (k <= 128) return queryDeepRSmallKOccupancy<InputT, 16, 4, TransformT, F16StagedDecoder, ReductionOpT>();
    if (k <= 160) return queryDeepRSmallKOccupancy<InputT, 16, 5, TransformT, F16StagedDecoder, ReductionOpT>();
    return queryDeepRSmallKOccupancy<InputT, 16, 8, TransformT, F16StagedDecoder, ReductionOpT>();
}

// The DeepRSmallK P16 launches. The only compile-time axes are dtype, input transform, reduction combine and the K
// ownership; depth is a runtime operand.
template <typename InputT, typename ReductionOpT, typename TransformT>
void launchDeepRSmallKTyped(const StridedSpan& input, const StridedSpan& partials,
                            const ReductionGeometry& geometry, uint64_t rows,
                            uint64_t shards, cudaStream_t stream) {
    dispatchDeepRSmallKRows<InputT, TransformT, ReductionOpT>(
        input, partials, geometry, rows, shards, stream, TransformT{});
}

template <typename InputT, typename ReductionOpT, typename TransformT>
KernelOccupancyInfo queryDeepRSmallKTyped(uint64_t rows, uint64_t k) {
    return queryDeepRSmallKRows<InputT, TransformT, ReductionOpT>(rows, k);
}

template <typename InputT, typename ReductionOpT, typename Callback>
auto dispatchDeepRSmallKTransform(ReductionStageInputTransform transform, Callback&& cb) {
    switch (transform) {
        case ReductionStageInputTransform::Identity:
            return cb.template operator()<InputT, ReductionOpT, IdentityFp32>();
        case ReductionStageInputTransform::AbsoluteValue:
            return cb.template operator()<InputT, ReductionOpT, AbsoluteValueFp32>();
        case ReductionStageInputTransform::Square:
            return cb.template operator()<InputT, ReductionOpT, SquareFp32>();
    }
    throw std::logic_error("DeepRSmallK retained kernel received invalid input transform.");
}

template <typename InputT, typename Callback>
auto dispatchDeepRSmallKCombine(const ReductionStageSemantics& semantics, Callback&& cb) {
    switch (semantics.combine) {
        case ReductionStageCombine::Sum:
            return dispatchDeepRSmallKTransform<InputT, cuda::std::plus<float>>(
                semantics.input_transform, cb);
        case ReductionStageCombine::Product:
            return dispatchDeepRSmallKTransform<InputT, cuda::std::multiplies<float>>(
                semantics.input_transform, cb);
        case ReductionStageCombine::Minimum:
            return dispatchDeepRSmallKTransform<InputT, PropagatingMinimumFp32>(
                semantics.input_transform, cb);
        case ReductionStageCombine::Maximum:
            return dispatchDeepRSmallKTransform<InputT, PropagatingMaximumFp32>(
                semantics.input_transform, cb);
    }
    throw std::logic_error("DeepRSmallK retained kernel received invalid reduction combine.");
}


// The FP8 DeepRSmallK argmin/argmax twins: the staged first stages of deepRSmallKValue{,TinyK}Kernel (P16, grid x =
// shard, z = outer, eight warps staging whole-row groups, the same rows per warp group and the same load-ahead rule)
// with uint32 candidates. They write FP32 value + carried index partials [outer, shards, K]; the index is (shard row
// base + row) * domain_stride. The planner reduces the shard axis with the carried-index ARG reducer. Rows of one
// component are packed four per key word (fp8ArgKeys4), the OneWindow small-bucket scheme. The shard depth is a runtime
// operand for every ownership (VALUE compiles it in for one component per lane; measure before copying that).

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
    if (count == 4U) return fp8ArgMin(local, fp8ArgRowsCandidate4(keys, first_row, row_stride));
#pragma unroll
    for (uint32_t j = 0; j < 4U; ++j) {
        if (j < count) local = fp8ArgMin(local, fp8ArgLaneCandidate(keys, j, first_row + j * row_stride));
    }
    return local;
}

// Packing: one and two components per lane pack four rows of a component per key word (groups hold 15+ rows at K <= 32
// and 7+ at K <= 64). Four or more components per lane pack four of the lane's components of one row, as OneWindow's
// large buckets do: at K 161..256 a 497-byte group holds one to three rows, so row packing would fill one to three
// bytes of every key word (it measured the c8 twin at 1.92x its VALUE kernel).
// At K 9..16, one component per lane would use only K of 32 lanes (ARG/VALUE 1.24, worst 1.41), so
// `workers` = floor(32 / K) lanes share a component, worker w keying rows w, w + workers, ... (the tiny-K scheme with a
// runtime lane count). workers = 1 at K 17..32.
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
__global__ void fp8DeepRSmallKArgKernel(const uint8_t* input,
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

    __shared__ RawVectorPacket<PacketBytes> staged_packets[DEEP_R_SMALL_K_WARPS][DEEP_R_SMALL_K_PACKETS_PER_WARP];
    __shared__ uint32_t warp_partials[DEEP_R_SMALL_K_WARPS][MaxSupportedK];

    const uint32_t warp = threadIdx.x / TILED_REDUCTION_WARP_THREADS;
    const uint32_t lane = threadIdx.x % TILED_REDUCTION_WARP_THREADS;
    const uint32_t inner_size_32 = static_cast<uint32_t>(inner_size);
    const uint32_t rows_per_block_round = rows_per_warp_group * DEEP_R_SMALL_K_WARPS;
    const uint32_t outer_count = static_cast<uint32_t>(outer_size);
    const uint32_t shard_count = static_cast<uint32_t>(shards_per_output);
    // One component per lane: floor(32 / K) workers per component.
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
            DeepRSmallKGroupWindow<PacketBytes> window{};
            RawVectorPacket<PacketBytes> pending{};
            if (group_row < shard_rows) {
                window = deepRSmallKGroupWindow<PacketBytes>(shard_input, group_row, shard_rows, rows_per_warp_group,
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
                    window = deepRSmallKGroupWindow<PacketBytes>(shard_input, next_group_row, shard_rows,
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
                const DeepRSmallKGroupWindow<PacketBytes> window = deepRSmallKGroupWindow<PacketBytes>(
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
            for (int source_warp = 0; source_warp < DEEP_R_SMALL_K_WARPS; ++source_warp) {
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
__global__ void fp8DeepRSmallKArgTinyKKernel(const uint8_t* input,
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

    __shared__ RawVectorPacket<PacketBytes> staged_packets[DEEP_R_SMALL_K_WARPS][DEEP_R_SMALL_K_PACKETS_PER_WARP];
    __shared__ uint32_t warp_partials[DEEP_R_SMALL_K_WARPS][MaxSupportedK];

    const uint32_t warp = threadIdx.x / TILED_REDUCTION_WARP_THREADS;
    const uint32_t lane = threadIdx.x % TILED_REDUCTION_WARP_THREADS;
    const uint32_t component = lane / LanesPerComponent;
    const uint32_t component_lane = lane % LanesPerComponent;
    const uint32_t inner_size_32 = static_cast<uint32_t>(inner_size);
    const uint32_t rows_per_block_round = rows_per_warp_group * DEEP_R_SMALL_K_WARPS;
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
            const DeepRSmallKGroupWindow<PacketBytes> window = deepRSmallKGroupWindow<PacketBytes>(
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
            for (int source_warp = 0; source_warp < DEEP_R_SMALL_K_WARPS; ++source_warp) {
                reduced = fp8ArgMin(reduced, warp_partials[source_warp][threadIdx.x]);
            }
            storeFp8ArgResult<true>(outputs, (static_cast<uint64_t>(outer_index) * shards_per_output + shard) * inner_size +
                                           threadIdx.x,
                              reduced, row_begin);
        }
        __syncthreads();
    }
}

// The FP32 RK ARG DeepRSmallK twins: the FP32 paths of deepRSmallKValue{,TinyK}Kernel (P16, grid x = shard, z = outer,
// eight warps staging whole-row groups of at most 125 FP32, the same rows per warp group and the same load-ahead rule)
// with one FP32 ARG candidate (Fp32ArgKey.cuh) per owned component, and FP8 ARG's lane sharing at K 9..16
// (floor(32 / K) workers per component). Staged only (FP32 VALUE DeepRSmallK's R > depth): each shard writes packed
// partials [outer, shards, K], key << 32 | shard row << 1 | negative zero, which launchFp32ArgPackedContinuation
// finishes. A packed word orders as its (key, tail) candidate, so lanes, warps and workers merge with a 64-bit minimum;
// a lane's rows only increase, so a strictly smaller key replaces its best. The shard depth is a runtime operand for
// every ownership.
__device__ __forceinline__ void fp32DeepRArgKeyElement(uint32_t& best_key, uint32_t& best_tail, uint32_t bits,
                                                       uint32_t row, uint32_t xor_mask) {
    uint32_t negative_zero;
    const uint32_t key = fp32ArgKey(bits, xor_mask, negative_zero);
    if (key < best_key) {
        best_key = key;
        best_tail = (row << 1) | negative_zero;
    }
}

// FP16 / BF16 input runs at the 2-byte VALUE windows (K <= 249, TinyK at K <= 4 / 8, then one / two / four / eight
// components per lane): staged element i is the i-th half-word, widened to FP32 and given the FP32 key.
template <typename InputT>
__device__ __forceinline__ uint32_t fp32DeepRArgStagedBits(const uint32_t* staged_words, uint32_t staged_index) {
    if constexpr (std::is_same_v<InputT, float>) {
        return staged_words[staged_index];
    } else {
        return __float_as_uint(widenF16Bits<InputT>(reinterpret_cast<const uint16_t*>(staged_words)[staged_index]));
    }
}

__device__ __forceinline__ uint64_t fp32DeepRArgPacked(uint32_t key, uint32_t tail) {
    return (static_cast<uint64_t>(key) << 32) | tail;
}

__device__ __forceinline__ uint64_t fp32DeepRArgMin(uint64_t a, uint64_t b) { return b < a ? b : a; }

template <typename InputT, int MaxComponentsPerLane>
__global__ __launch_bounds__(DEEP_R_SMALL_K_THREADS) void deepRSmallKArgKernel(const InputT* input,
                                                                         uint64_t* packed,
                                                                         uint64_t outer_size,
                                                                         uint64_t reduction_size,
                                                                         uint64_t inner_size,
                                                                         uint64_t shards_per_output,
                                                                         uint32_t rows_per_shard_log2,
                                                                         uint32_t rows_per_warp_group,
                                                                         uint32_t xor_mask) {
    constexpr int PacketBytes = 16;
    static_assert(std::is_same_v<InputT, float> || isF16InputType<InputT>);
    static_assert(MaxComponentsPerLane == 1 || MaxComponentsPerLane == 2 || MaxComponentsPerLane == 4 ||
                  (MaxComponentsPerLane == 8 && sizeof(InputT) == 2));
    constexpr int MaxSupportedK = TILED_REDUCTION_WARP_THREADS * MaxComponentsPerLane;

    __shared__ RawVectorPacket<PacketBytes> staged_packets[DEEP_R_SMALL_K_WARPS][DEEP_R_SMALL_K_PACKETS_PER_WARP];
    __shared__ uint64_t warp_partials[DEEP_R_SMALL_K_WARPS][MaxSupportedK];

    const uint32_t warp = threadIdx.x / TILED_REDUCTION_WARP_THREADS;
    const uint32_t lane = threadIdx.x % TILED_REDUCTION_WARP_THREADS;
    const uint32_t inner_size_32 = static_cast<uint32_t>(inner_size);
    const uint32_t rows_per_block_round = rows_per_warp_group * DEEP_R_SMALL_K_WARPS;
    const uint32_t outer_count = static_cast<uint32_t>(outer_size);
    const uint32_t shard_count = static_cast<uint32_t>(shards_per_output);
    // One component per lane: floor(32 / K) workers per component at K <= 16 (FP8 ARG's rule).
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
        const InputT* shard_input = input + (outer_index * reduction_size + row_begin) * inner_size;

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
            DeepRSmallKGroupWindow<PacketBytes> window{};
            RawVectorPacket<PacketBytes> pending{};
            if (group_row < shard_rows) {
                window = deepRSmallKGroupWindow<PacketBytes>(shard_input, group_row, shard_rows, rows_per_warp_group,
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
                    window = deepRSmallKGroupWindow<PacketBytes>(shard_input, next_group_row, shard_rows,
                                                                 rows_per_warp_group, inner_size_32);
                    if (lane < window.packet_count) pending = window.aligned_begin[lane];
                }
#pragma unroll
                for (int slot = 0; slot < MaxComponentsPerLane; ++slot) {
                    const uint32_t owned = lane + static_cast<uint32_t>(slot) * TILED_REDUCTION_WARP_THREADS;
                    if (owned < inner_size_32) {
                        uint32_t staged_index = prefix_elements + owned;
                        for (uint32_t row = 0; row < group_rows; ++row, staged_index += inner_size_32) {
                            fp32DeepRArgKeyElement(best_key[slot], best_tail[slot],
                                                   fp32DeepRArgStagedBits<InputT>(staged_words, staged_index),
                                                   group_row + row, xor_mask);
                        }
                    }
                }
                __syncwarp();
            }
        } else {
            for (uint32_t group_row = warp * rows_per_warp_group; group_row < shard_rows;
                 group_row += rows_per_block_round) {
                const DeepRSmallKGroupWindow<PacketBytes> window = deepRSmallKGroupWindow<PacketBytes>(
                    shard_input, group_row, shard_rows, rows_per_warp_group, inner_size_32);
                if (lane < window.packet_count) staged_packets[warp][lane] = window.aligned_begin[lane];
                __syncwarp();
                if (c1_component < inner_size_32) {
                    uint32_t staged_index = window.prefix_elements + c1_worker * inner_size_32 + c1_component;
                    const uint32_t staged_step = workers * inner_size_32;
                    for (uint32_t row = c1_worker; row < window.group_rows;
                         row += workers, staged_index += staged_step) {
                        fp32DeepRArgKeyElement(best_key[0], best_tail[0],
                                               fp32DeepRArgStagedBits<InputT>(staged_words, staged_index),
                                               group_row + row, xor_mask);
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
            for (int source_warp = 0; source_warp < DEEP_R_SMALL_K_WARPS; ++source_warp) {
                reduced = fp32DeepRArgMin(reduced, warp_partials[source_warp][threadIdx.x]);
            }
            packed[(static_cast<uint64_t>(outer_index) * shards_per_output + shard) * inner_size + threadIdx.x] =
                reduced;
        }
        __syncthreads();
    }
}

template <typename InputT, int LanesPerComponent>
__global__ __launch_bounds__(DEEP_R_SMALL_K_THREADS) void deepRSmallKArgTinyKKernel(const InputT* input,
                                                                              uint64_t* packed,
                                                                              uint64_t outer_size,
                                                                              uint64_t reduction_size,
                                                                              uint64_t inner_size,
                                                                              uint64_t shards_per_output,
                                                                              uint32_t rows_per_shard_log2,
                                                                              uint32_t rows_per_warp_group,
                                                                              uint32_t xor_mask) {
    constexpr int PacketBytes = 16;
    static_assert(std::is_same_v<InputT, float> || isF16InputType<InputT>);
    static_assert(LanesPerComponent == 4 || LanesPerComponent == 8);
    constexpr int MaxSupportedK = TILED_REDUCTION_WARP_THREADS / LanesPerComponent;

    __shared__ RawVectorPacket<PacketBytes> staged_packets[DEEP_R_SMALL_K_WARPS][DEEP_R_SMALL_K_PACKETS_PER_WARP];
    __shared__ uint64_t warp_partials[DEEP_R_SMALL_K_WARPS][MaxSupportedK];

    const uint32_t warp = threadIdx.x / TILED_REDUCTION_WARP_THREADS;
    const uint32_t lane = threadIdx.x % TILED_REDUCTION_WARP_THREADS;
    const uint32_t component = lane / LanesPerComponent;
    const uint32_t component_lane = lane % LanesPerComponent;
    const uint32_t inner_size_32 = static_cast<uint32_t>(inner_size);
    const uint32_t rows_per_block_round = rows_per_warp_group * DEEP_R_SMALL_K_WARPS;
    const uint32_t outer_count = static_cast<uint32_t>(outer_size);
    const uint32_t shard_count = static_cast<uint32_t>(shards_per_output);
    const auto* staged_words = reinterpret_cast<const uint32_t*>(&staged_packets[warp][0]);
    for (uint32_t outer_index = blockIdx.z; outer_index < outer_count; outer_index += gridDim.z)
    for (uint32_t shard = blockIdx.x; shard < shard_count; shard += gridDim.x) {
        const uint64_t row_begin = static_cast<uint64_t>(shard) << rows_per_shard_log2;
        const uint32_t shard_rows = row_begin < reduction_size
            ? static_cast<uint32_t>(minU64(uint64_t{1} << rows_per_shard_log2, reduction_size - row_begin))
            : 0U;
        const InputT* shard_input = input + (outer_index * reduction_size + row_begin) * inner_size;

        uint32_t best_key = FP32_ARG_KEY_IDENTITY;
        uint32_t best_tail = 0xFFFFFFFFU;
        for (uint32_t group_row = warp * rows_per_warp_group; group_row < shard_rows;
             group_row += rows_per_block_round) {
            const DeepRSmallKGroupWindow<PacketBytes> window = deepRSmallKGroupWindow<PacketBytes>(
                shard_input, group_row, shard_rows, rows_per_warp_group, inner_size_32);
            if (lane < window.packet_count) staged_packets[warp][lane] = window.aligned_begin[lane];
            __syncwarp();
            if (component < inner_size_32) {
                const uint32_t staged_step = LanesPerComponent * inner_size_32;
                uint32_t staged_index = window.prefix_elements + component_lane * inner_size_32 + component;
                for (uint32_t row = component_lane; row < window.group_rows;
                     row += LanesPerComponent, staged_index += staged_step) {
                    fp32DeepRArgKeyElement(best_key, best_tail,
                                           fp32DeepRArgStagedBits<InputT>(staged_words, staged_index),
                                           group_row + row, xor_mask);
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
            for (int source_warp = 0; source_warp < DEEP_R_SMALL_K_WARPS; ++source_warp) {
                reduced = fp32DeepRArgMin(reduced, warp_partials[source_warp][threadIdx.x]);
            }
            packed[(static_cast<uint64_t>(outer_index) * shards_per_output + shard) * inner_size + threadIdx.x] =
                reduced;
        }
        __syncthreads();
    }
}

// The carried-index DeepRSmallK ARG twins (see Fp32ArgWidePartials): deepRSmallKArgKernel's and
// deepRSmallKArgTinyKKernel's ownership, groups and load-ahead rule, staging each lane's carried index packet (the same
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
__global__ __launch_bounds__(DEEP_R_SMALL_K_THREADS) void deepRSmallKCarriedArgKernel(const float* input,
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

    __shared__ RawVectorPacket<PacketBytes> staged_packets[DEEP_R_SMALL_K_WARPS][DEEP_R_SMALL_K_PACKETS_PER_WARP];
    __shared__ IndexT staged_indices[DEEP_R_SMALL_K_WARPS][DEEP_R_SMALL_K_PACKETS_PER_WARP * 4];
    __shared__ uint32_t warp_keys[DEEP_R_SMALL_K_WARPS][MaxSupportedK];
    __shared__ uint64_t warp_tails[DEEP_R_SMALL_K_WARPS][MaxSupportedK];

    const uint32_t warp = threadIdx.x / TILED_REDUCTION_WARP_THREADS;
    const uint32_t lane = threadIdx.x % TILED_REDUCTION_WARP_THREADS;
    const uint32_t inner_size_32 = static_cast<uint32_t>(inner_size);
    const uint32_t rows_per_block_round = rows_per_warp_group * DEEP_R_SMALL_K_WARPS;
    const uint32_t outer_count = static_cast<uint32_t>(outer_size);
    const uint32_t shard_count = static_cast<uint32_t>(shards_per_output);
    const uint64_t total_elements = outer_size * reduction_size * inner_size;
    // One component per lane: floor(32 / K) workers per component at K <= 16 (FP8 ARG's rule).
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
        const auto packet_element = [&](const DeepRSmallKGroupWindow<PacketBytes>& window) {
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
            DeepRSmallKGroupWindow<PacketBytes> window{};
            RawVectorPacket<PacketBytes> pending{};
            PackedInputValues<IndexT, 4> pending_indices{};
            if (group_row < shard_rows) {
                window = deepRSmallKGroupWindow<PacketBytes>(shard_input, group_row, shard_rows, rows_per_warp_group,
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
                    window = deepRSmallKGroupWindow<PacketBytes>(shard_input, next_group_row, shard_rows,
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
                const DeepRSmallKGroupWindow<PacketBytes> window = deepRSmallKGroupWindow<PacketBytes>(
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
            for (int source_warp = 0; source_warp < DEEP_R_SMALL_K_WARPS; ++source_warp) {
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
__global__ __launch_bounds__(DEEP_R_SMALL_K_THREADS) void deepRSmallKCarriedArgTinyKKernel(const float* input,
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

    __shared__ RawVectorPacket<PacketBytes> staged_packets[DEEP_R_SMALL_K_WARPS][DEEP_R_SMALL_K_PACKETS_PER_WARP];
    __shared__ IndexT staged_indices[DEEP_R_SMALL_K_WARPS][DEEP_R_SMALL_K_PACKETS_PER_WARP * 4];
    __shared__ uint32_t warp_keys[DEEP_R_SMALL_K_WARPS][MaxSupportedK];
    __shared__ uint64_t warp_tails[DEEP_R_SMALL_K_WARPS][MaxSupportedK];

    const uint32_t warp = threadIdx.x / TILED_REDUCTION_WARP_THREADS;
    const uint32_t lane = threadIdx.x % TILED_REDUCTION_WARP_THREADS;
    const uint32_t component = lane / LanesPerComponent;
    const uint32_t component_lane = lane % LanesPerComponent;
    const uint32_t inner_size_32 = static_cast<uint32_t>(inner_size);
    const uint32_t rows_per_block_round = rows_per_warp_group * DEEP_R_SMALL_K_WARPS;
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
            const DeepRSmallKGroupWindow<PacketBytes> window = deepRSmallKGroupWindow<PacketBytes>(
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
            for (int source_warp = 0; source_warp < DEEP_R_SMALL_K_WARPS; ++source_warp) {
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

}  // namespace

void launchFp8DeepRSmallKFirstStage(const ReductionStageSemantics& semantics,
                                    const StridedSpan& input, const StridedSpan& fp32_partials,
                                    const ReductionGeometry& geometry, uint64_t rows,
                                    uint64_t shards, Stream& stream) {
    const auto& depths = RK::Fp8::Fp8DeepRSmallKPolicy::compiled_rows;
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
        launchDeepRSmallKTyped<InputT, ReductionOpT, TransformT>(
            input, fp32_partials, geometry, rows, shards, stream.getStream());
    };
    switch (input.getDataType()) {
        case DataType::FP8_E4M3:
            dispatchDeepRSmallKCombine<__nv_fp8_e4m3>(semantics, launch);
            return;
        case DataType::FP8_E5M2:
            dispatchDeepRSmallKCombine<__nv_fp8_e5m2>(semantics, launch);
            return;
        default:
            throw std::logic_error("DeepRSmallK retained first stage requires E4M3 or E5M2.");
    }
}

bool fp32DeepRSmallKSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size, uint64_t rows) {
    const auto& depths = RK::Fp8::Fp8DeepRSmallKPolicy::compiled_rows;
    if (outer_size == 0 || inner_size == 0 || inner_size > 125 || reduction_size <= rows ||
        std::find(depths.begin(), depths.end(), rows) == depths.end()) {
        return false;
    }
    // reductionWorkGrid's work extents (shards and outers) stay below 2^31.
    const uint64_t shards = ceilDivideU64(reduction_size, rows);
    return shards <= RK::RK_MAX_GRID_EXTENT && outer_size <= RK::RK_MAX_GRID_EXTENT;
}

void launchFp32DeepRSmallKFirstStage(const ReductionStageSemantics& semantics,
                                     const StridedSpan& input, const StridedSpan& fp32_partials,
                                     const ReductionGeometry& geometry, uint64_t rows,
                                     uint64_t shards, Stream& stream) {
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
        dispatchFp32DeepRSmallKRows<TransformT, ReductionOpT>(input, fp32_partials, geometry, rows, shards,
                                                              stream.getStream());
    };
    dispatchDeepRSmallKCombine<float>(semantics, launch);
}

KernelOccupancyInfo queryFp32DeepRSmallKOccupancy(uint64_t rows, uint64_t k,
                                                            const ReductionStageSemantics& semantics) {
    const auto& depths = RK::Fp8::Fp8DeepRSmallKPolicy::compiled_rows;
    if (k == 0 || k > 125 || std::find(depths.begin(), depths.end(), rows) == depths.end())
        throw std::logic_error("FP32 DeepRSmallK occupancy requires a retained depth and K <= 125.");
    const auto query = [&]<typename InputT, typename ReductionOpT, typename TransformT>() -> KernelOccupancyInfo {
        return queryFp32DeepRSmallKRows<TransformT, ReductionOpT>(rows, k);
    };
    return dispatchDeepRSmallKCombine<float>(semantics, query);
}

bool f16DeepRSmallKSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size, uint64_t rows) {
    const auto& depths = RK::Fp8::Fp8DeepRSmallKPolicy::compiled_rows;
    if (outer_size == 0 || inner_size == 0 || inner_size > 249 || reduction_size <= rows ||
        std::find(depths.begin(), depths.end(), rows) == depths.end()) {
        return false;
    }
    const uint64_t shards = ceilDivideU64(reduction_size, rows);
    return shards <= RK::RK_MAX_GRID_EXTENT && outer_size <= RK::RK_MAX_GRID_EXTENT;
}

void launchF16DeepRSmallKFirstStage(const ReductionStageSemantics& semantics,
                                    const StridedSpan& input, const StridedSpan& fp32_partials,
                                    const ReductionGeometry& geometry, uint64_t rows,
                                    uint64_t shards, Stream& stream) {
    const DataType dtype = input.getDataType();
    if ((dtype != DataType::FP16 && dtype != DataType::BF16) ||
        !f16DeepRSmallKSupports(geometry.outer_size, geometry.reduction_size, geometry.inner_size, rows) ||
        shards != ceilDivideU64(geometry.reduction_size, rows) ||
        input.getDimensions() !=
            std::vector<uint64_t>{geometry.outer_size, geometry.reduction_size, geometry.inner_size} ||
        (reinterpret_cast<uintptr_t>(input.getMemPtr<void>()) & uintptr_t{15}) != 0 ||
        fp32_partials.getDataType() != DataType::FP32 ||
        fp32_partials.getDimensions() != std::vector<uint64_t>{geometry.outer_size, shards, geometry.inner_size}) {
        throw std::logic_error("FP16 / BF16 DeepRSmallK first stage received illegal geometry, input or partials.");
    }
    // Raw FP32 partials, as FP8's and FP32's; the continuation finalizes.
    const auto launch = [&]<typename InputT, typename ReductionOpT, typename TransformT>() {
        dispatchF16DeepRSmallKRows<InputT, TransformT, ReductionOpT>(input, fp32_partials, geometry, rows, shards,
                                                                     stream.getStream());
    };
    if (dtype == DataType::FP16) {
        dispatchDeepRSmallKCombine<__half>(semantics, launch);
    } else {
        dispatchDeepRSmallKCombine<__nv_bfloat16>(semantics, launch);
    }
}

KernelOccupancyInfo queryF16DeepRSmallKOccupancy(DataType input_dtype, uint64_t rows, uint64_t k,
                                                           const ReductionStageSemantics& semantics) {
    const auto& depths = RK::Fp8::Fp8DeepRSmallKPolicy::compiled_rows;
    if ((input_dtype != DataType::FP16 && input_dtype != DataType::BF16) || k == 0 || k > 249 ||
        std::find(depths.begin(), depths.end(), rows) == depths.end())
        throw std::logic_error("FP16 / BF16 DeepRSmallK occupancy requires a 2-byte input, a retained depth and K <= "
                               "249.");
    const auto query = [&]<typename InputT, typename ReductionOpT, typename TransformT>() -> KernelOccupancyInfo {
        return queryF16DeepRSmallKRows<InputT, TransformT, ReductionOpT>(rows, k);
    };
    if (input_dtype == DataType::FP16) return dispatchDeepRSmallKCombine<__half>(semantics, query);
    return dispatchDeepRSmallKCombine<__nv_bfloat16>(semantics, query);
}

KernelOccupancyInfo queryFp8DeepRSmallKOccupancy(
    DataType dtype, uint64_t rows, uint64_t k, const ReductionStageSemantics& semantics) {
    const auto& depths = RK::Fp8::Fp8DeepRSmallKPolicy::compiled_rows;
    if (k == 0 || k > 256 || std::find(depths.begin(), depths.end(), rows) == depths.end())
        throw std::logic_error("DeepRSmallK retained occupancy requires selected P16 depth and K <= 256.");
    const auto query = [&]<typename InputT, typename ReductionOpT, typename TransformT>() -> KernelOccupancyInfo {
        return queryDeepRSmallKTyped<InputT, ReductionOpT, TransformT>(rows, k);
    };
    switch (dtype) {
        case DataType::FP8_E4M3: return dispatchDeepRSmallKCombine<__nv_fp8_e4m3>(semantics, query);
        case DataType::FP8_E5M2: return dispatchDeepRSmallKCombine<__nv_fp8_e5m2>(semantics, query);
        default: throw std::logic_error("DeepRSmallK retained occupancy requires E4M3 or E5M2.");
    }
}


bool fp8DeepRSmallKArgSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size,
                               uint64_t rows_per_shard) {
    return RK::Fp8::Fp8DeepRSmallKPolicy::legal({static_cast<uint32_t>(rows_per_shard)}, outer_size, reduction_size,
                                                inner_size) &&
           rows_per_shard < FP8_ARG_MAX_LANE_ROWS;
}

namespace {
void dispatchFp8DeepRSmallKArgFirstStage(const StridedSpan& input,
                                         const Fp8ArgOutputs& arg_outputs,
                                         const ReductionGeometry& geometry,
                                         uint64_t rows_per_shard,
                                         uint64_t shards,
                                         Stream& stream);
}  // namespace

void launchFp8DeepRSmallKArgFirstStage(const StridedSpan& input,
                                       const StridedSpan& value_partials,
                                       const StridedSpan& index_partials,
                                       const ReductionGeometry& geometry,
                                       uint64_t rows_per_shard,
                                       uint64_t shards,
                                       bool maximum,
                                       uint64_t domain_stride,
                                       Stream& stream) {
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
}

void launchFp8DeepRSmallKArgFirstStagePacked(const StridedSpan& input,
                                             uint32_t* packed_partials,
                                             const ReductionGeometry& geometry,
                                             uint64_t rows_per_shard,
                                             uint64_t shards,
                                             bool maximum,
                                             Stream& stream) {
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
}

namespace {
void dispatchFp8DeepRSmallKArgFirstStage(const StridedSpan& input,
                                         const Fp8ArgOutputs& arg_outputs,
                                         const ReductionGeometry& geometry,
                                         uint64_t rows_per_shard,
                                         uint64_t shards,
                                         Stream& stream) {
    const uint32_t depth = deepRSmallKCheckedRowsPerShard(rows_per_shard);
    const uint64_t bytes_per_work_item = std::min<uint64_t>(depth, geometry.reduction_size) * geometry.inner_size;
    const dim3 grid_blocks = reductionWorkGrid(shards, 1, geometry.outer_size,
                                               bytes_per_work_item <= 256 ? REDUCTION_PERSISTENT_MAX_BLOCKS : 0);
    const uint32_t rows_per_warp_group = deepRSmallKRowsPerWarpGroup<16>(geometry.inner_size, depth);
    const uint32_t depth_log2 = deepRSmallKRowsPerShardLog2(depth);
    const auto* bytes = static_cast<const uint8_t*>(input.getMemPtr<void>());
    const uint64_t k = geometry.inner_size;
    const auto packed = [&]<int Components>() {
        fp8DeepRSmallKArgKernel<Components><<<grid_blocks, DEEP_R_SMALL_K_THREADS, 0, stream.getStream()>>>(
            bytes, arg_outputs, geometry.outer_size, geometry.reduction_size, k, shards, depth_log2,
            rows_per_warp_group);
    };
    const auto tiny = [&]<int Lanes>() {
        fp8DeepRSmallKArgTinyKKernel<Lanes><<<grid_blocks, DEEP_R_SMALL_K_THREADS, 0, stream.getStream()>>>(
            bytes, arg_outputs, geometry.outer_size, geometry.reduction_size, k, shards, depth_log2,
            rows_per_warp_group);
    };
    // The VALUE DeepRSmallK ownership by K (dispatchDeepRSmallKRows).
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


// ---- The FP32 and FP16 / BF16 RK ARG DeepRSmallK twins (see deepRSmallKArgKernel).
bool fp32DeepRSmallKArgSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size,
                                uint64_t rows_per_shard) {
    return fp32DeepRSmallKSupports(outer_size, reduction_size, inner_size, rows_per_shard);
}

bool f16DeepRSmallKArgSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size,
                               uint64_t rows_per_shard) {
    return f16DeepRSmallKSupports(outer_size, reduction_size, inner_size, rows_per_shard);
}

namespace {
// Invokes callback(kernel) with the DeepRSmallK ARG kernel owning K: FP32 VALUE DeepRSmallK's ownership by K
// (dispatchFp32DeepRSmallKRows: TinyK 8 / 4 lanes, then 1 / 2 / 4 components) and, at 2 bytes, one more step of 8
// components (K 129..249).
template <typename InputT, typename Callback>
void dispatchDeepRArgKernel(uint64_t k, Callback&& callback) {
    if (k <= 4) callback(deepRSmallKArgTinyKKernel<InputT, 8>);
    else if (k <= 8) callback(deepRSmallKArgTinyKKernel<InputT, 4>);
    else if (k <= 32) callback(deepRSmallKArgKernel<InputT, 1>);
    else if (k <= 64) callback(deepRSmallKArgKernel<InputT, 2>);
    else if (std::is_same_v<InputT, float> || k <= 128) callback(deepRSmallKArgKernel<InputT, 4>);
    else if constexpr (!std::is_same_v<InputT, float>) callback(deepRSmallKArgKernel<InputT, 8>);
}

template <typename InputT>
void launchDeepRSmallKArgTyped(const StridedSpan& input,
                               const StridedSpan* value_output,
                               const StridedSpan* index_output,
                               const ReductionGeometry& geometry,
                               uint64_t rows_per_shard,
                               uint64_t* packed_scratch,
                               bool maximum,
                               uint64_t domain_stride,
                               Stream& stream,
                               const char* name) {
    constexpr uint32_t ElementBytes = static_cast<uint32_t>(sizeof(InputT));
    const uint64_t outputs = geometry.outer_size * geometry.inner_size;
    const bool supported =
        ElementBytes == 4 ? fp32DeepRSmallKArgSupports(geometry.outer_size, geometry.reduction_size,
                                                       geometry.inner_size, rows_per_shard)
                          : f16DeepRSmallKArgSupports(geometry.outer_size, geometry.reduction_size,
                                                      geometry.inner_size, rows_per_shard);
    if (!supported ||
        input.getTotalNumElements() != geometry.outer_size * geometry.reduction_size * geometry.inner_size ||
        packed_scratch == nullptr || (value_output == nullptr && index_output == nullptr) ||
        (value_output != nullptr && (value_output->getTotalNumElements() != outputs ||
                                     !fp8CompleteSupportsOutputStorage(value_output->getDataType()))) ||
        (index_output != nullptr && (index_output->getTotalNumElements() != outputs ||
                                     (index_output->getDataType() != DataType::UINT32 &&
                                      index_output->getDataType() != DataType::UINT64)))) {
        throw std::logic_error(std::string(name) +
                               " DeepRSmallK ARG received unsupported geometry, outputs or scratch.");
    }
    if ((reinterpret_cast<uintptr_t>(input.getMemPtr<void>()) & uintptr_t{15}) != 0) {
        throw std::logic_error(std::string(name) + " DeepRSmallK ARG requires a 16-byte aligned input base.");
    }
    if (index_output != nullptr && index_output->getDataType() == DataType::UINT32 &&
        (geometry.reduction_size - 1) * domain_stride > std::numeric_limits<uint32_t>::max()) {
        throw std::logic_error(std::string(name) +
                               " DeepRSmallK ARG UINT32 index cannot hold the original-domain index.");
    }
    const uint32_t depth = deepRSmallKCheckedRowsPerShard(rows_per_shard);
    const uint64_t shards = ceilDivideU64(geometry.reduction_size, depth);
    const uint64_t bytes_per_work_item =
        std::min<uint64_t>(depth, geometry.reduction_size) * geometry.inner_size * ElementBytes;
    const dim3 grid_blocks = reductionWorkGrid(shards, 1, geometry.outer_size,
                                               bytes_per_work_item <= 256 ? REDUCTION_PERSISTENT_MAX_BLOCKS : 0);
    const uint32_t rows_per_warp_group =
        deepRSmallKRowsPerWarpGroup<16, ElementBytes>(geometry.inner_size, depth);
    const uint32_t depth_log2 = deepRSmallKRowsPerShardLog2(depth);
    const uint32_t xor_mask = maximum ? 0x80000000U : 0U;
    const InputT* values = input.getMemPtr<InputT>();
    const uint64_t k = geometry.inner_size;
    dispatchDeepRArgKernel<InputT>(k, [&](auto kernel) {
        kernel<<<grid_blocks, DEEP_R_SMALL_K_THREADS, 0, stream.getStream()>>>(
            values, packed_scratch, geometry.outer_size, geometry.reduction_size, k, shards, depth_log2,
            rows_per_warp_group, xor_mask);
    });
    CUDA_CHECK(cudaGetLastError());
    launchFp32ArgPackedContinuation(packed_scratch, value_output, index_output, geometry.outer_size,
                                    geometry.reduction_size, geometry.inner_size, shards, depth, maximum,
                                    domain_stride, stream);
}

template <typename InputT>
[[nodiscard]] KernelOccupancyInfo queryDeepRSmallKArgOccupancyTyped(uint64_t inner_size) {
    KernelOccupancyInfo occupancy{};
    dispatchDeepRArgKernel<InputT>(inner_size, [&](auto kernel) { occupancy = queryShallowCompleteOccupancy(kernel); });
    return occupancy;
}
}  // namespace

void launchFp32DeepRSmallKArg(const StridedSpan& input,
                              const StridedSpan* value_output,
                              const StridedSpan* index_output,
                              const ReductionGeometry& geometry,
                              uint64_t rows_per_shard,
                              uint64_t* packed_scratch,
                              bool maximum,
                              uint64_t domain_stride,
                              Stream& stream) {
    if (input.getDataType() != DataType::FP32)
        throw std::logic_error("FP32 DeepRSmallK ARG received unsupported geometry, outputs or scratch.");
    launchDeepRSmallKArgTyped<float>(input, value_output, index_output, geometry, rows_per_shard, packed_scratch,
                                     maximum, domain_stride, stream, "FP32");
}

void launchF16DeepRSmallKArg(const StridedSpan& input,
                             const StridedSpan* value_output,
                             const StridedSpan* index_output,
                             const ReductionGeometry& geometry,
                             uint64_t rows_per_shard,
                             uint64_t* packed_scratch,
                             bool maximum,
                             uint64_t domain_stride,
                             Stream& stream) {
    switch (input.getDataType()) {
        case DataType::FP16:
            return launchDeepRSmallKArgTyped<__half>(input, value_output, index_output, geometry, rows_per_shard,
                                                     packed_scratch, maximum, domain_stride, stream, "FP16");
        case DataType::BF16:
            return launchDeepRSmallKArgTyped<__nv_bfloat16>(input, value_output, index_output, geometry,
                                                            rows_per_shard, packed_scratch, maximum, domain_stride,
                                                            stream, "BF16");
        default:
            throw std::logic_error("FP16 / BF16 DeepRSmallK ARG received a non-FP16 / BF16 input.");
    }
}

KernelOccupancyInfo queryFp32DeepRSmallKArgOccupancy(uint64_t inner_size) {
    if (inner_size == 0 || inner_size > 125) {
        throw std::logic_error("FP32 DeepRSmallK ARG occupancy requires K <= 125.");
    }
    return queryDeepRSmallKArgOccupancyTyped<float>(inner_size);
}

KernelOccupancyInfo queryF16DeepRSmallKArgOccupancy(DataType input_dtype, uint64_t inner_size) {
    if (inner_size == 0 || inner_size > F16_DEEP_R_MAX_K) {
        throw std::logic_error("FP16 / BF16 DeepRSmallK ARG occupancy requires K <= 249.");
    }
    switch (input_dtype) {
        case DataType::FP16: return queryDeepRSmallKArgOccupancyTyped<__half>(inner_size);
        case DataType::BF16: return queryDeepRSmallKArgOccupancyTyped<__nv_bfloat16>(inner_size);
        default: throw std::logic_error("FP16 / BF16 DeepRSmallK ARG occupancy requires FP16 / BF16 input.");
    }
}

// ---- The carried-index DeepRSmallK ARG twins (see deepRSmallKCarriedArgKernel).
void launchFp32DeepRSmallKCarriedArg(const StridedSpan& input,
                                     const StridedSpan& carried_index,
                                     const StridedSpan* value_output,
                                     const StridedSpan* index_output,
                                     const ReductionGeometry& geometry,
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
    const uint32_t depth = deepRSmallKCheckedRowsPerShard(rows_per_shard);
    const uint64_t shards = ceilDivideU64(geometry.reduction_size, depth);
    const uint64_t bytes_per_work_item =
        std::min<uint64_t>(depth, geometry.reduction_size) * geometry.inner_size * sizeof(float);
    const dim3 grid_blocks = reductionWorkGrid(shards, 1, geometry.outer_size,
                                               bytes_per_work_item <= 256 ? REDUCTION_PERSISTENT_MAX_BLOCKS : 0);
    const uint32_t rows_per_warp_group = deepRSmallKRowsPerWarpGroup<16, 4>(geometry.inner_size, depth);
    const uint32_t depth_log2 = deepRSmallKRowsPerShardLog2(depth);
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
            deepRSmallKCarriedArgKernel<Components, IndexT>
                <<<grid_blocks, DEEP_R_SMALL_K_THREADS, 0, stream.getStream()>>>(
                values, carried, partials, geometry.outer_size, geometry.reduction_size, k, shards, depth_log2,
                rows_per_warp_group, xor_mask, row_stride);
        };
        const auto tiny = [&]<int Lanes>() {
            deepRSmallKCarriedArgTinyKKernel<Lanes, IndexT>
                <<<grid_blocks, DEEP_R_SMALL_K_THREADS, 0, stream.getStream()>>>(
                values, carried, partials, geometry.outer_size, geometry.reduction_size, k, shards, depth_log2,
                rows_per_warp_group, xor_mask, row_stride);
        };
        // FP32 VALUE DeepRSmallK's ownership by K (dispatchFp32DeepRSmallKRows).
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

KernelOccupancyInfo queryFp32DeepRSmallKCarriedArgOccupancy(uint64_t inner_size, DataType index_dtype) {
    if (inner_size == 0 || inner_size > 125 || (index_dtype != DataType::UINT32 && index_dtype != DataType::UINT64)) {
        throw std::logic_error(
            "FP32 DeepRSmallK carried ARG occupancy requires K <= 125 and a UINT32 / UINT64 index.");
    }
    const auto query = [&]<typename IndexT>() {
        if (inner_size <= 4) return queryShallowCompleteOccupancy(deepRSmallKCarriedArgTinyKKernel<8, IndexT>);
        if (inner_size <= 8) return queryShallowCompleteOccupancy(deepRSmallKCarriedArgTinyKKernel<4, IndexT>);
        if (inner_size <= 32) return queryShallowCompleteOccupancy(deepRSmallKCarriedArgKernel<1, IndexT>);
        if (inner_size <= 64) return queryShallowCompleteOccupancy(deepRSmallKCarriedArgKernel<2, IndexT>);
        return queryShallowCompleteOccupancy(deepRSmallKCarriedArgKernel<4, IndexT>);
    };
    return index_dtype == DataType::UINT64 ? query.template operator()<uint64_t>()
                                           : query.template operator()<uint32_t>();
}

}  // namespace ThorImplementation::ReductionInternal
