#include "Utilities/TensorOperations/Reduction/Kernels/KernelLaunchers.h"
#include "Utilities/TensorOperations/Reduction/Kernels/ReductionOperation.cuh"
#include "Utilities/TensorOperations/Reduction/Arg/F16ArgKey.h"
#include "Utilities/TensorOperations/Reduction/Arg/Fp8ArgKey.h"
#include "Utilities/TensorOperations/Reduction/Arg/Fp32ArgKey.cuh"
#include "Utilities/TensorOperations/Reduction/Kernels/KernelCommon.cuh"

#include "Utilities/Exceptions.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace ThorImplementation::ReductionInternal {

// ---------------------------------------------------------------------------------------------------------------------
// The FP8 R / KR reductions (KR/KRSegments.h): Pack (one thread, several short rows), Group (G lanes per row) and Split
// (one CTA per row chunk writing a partial, then a finishing kernel, one CTA per row). Every family reads 4-byte words
// on the tensor's word grid (FP8_KR_UNROLL loads in flight) and masks bytes outside its work.
// VALUE compiles only the reachable semantics: SUM x {identity, |x|, x^2}, PRODUCT, and MIN / MAX through RK's shared
// MIN kernel with the byte fold of Fp8ExtremumFold (MAX negates through output_scale; |x| / x^2 fold into the bytes).
#if THOR_CUB_ENABLE_FP8_TYPES
namespace {

constexpr uint32_t FP8_KR_UNROLL = 4;
constexpr uint32_t FP8_KR_WARP_LANES = 32;
constexpr uint32_t FP8_KR_WARPS = KR::KR_THREADS / FP8_KR_WARP_LANES;
constexpr uint64_t FP8_KR_MAX_GRID_BLOCKS = 2147483647ULL;
static_assert(KR::KR_THREADS == 256 && FP8_KR_WARPS == 8);

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
        ReductionStageInputTransform::Identity);
}

// FP32 input (Fp32 KR, sharing the Group / Pack / Block / Split kernels): one element per word. MIN / MAX fold the sign
// and |x| into the word as FP8 folds its bytes (fold masks 0x7fffffff / 0x80000000, fp32CompleteExtremumFold); x^2 for
// MIN / MAX is |x| squared at the end, exact because rounding is monotone in |x|.
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

// FP16 / BF16 input, two elements per word (the low half-word first). MIN / MAX fold both half-words at once
// (fp16CompleteExtremumFold: masks 0x7fff7fff / 0x80008000); both formats widen to FP32 exactly. x^2 for MIN / MAX is
// |x| squared at the end, as for FP32: the FP32 square is monotone in |x|, so it is the extremum of the per-element
// FP32 squares (exact for FP16; BF16 squares can overflow or underflow FP32, monotonically).
template <typename InputT>
__device__ __forceinline__ float f16KRWidenLow(uint32_t word) {
    if constexpr (std::is_same_v<InputT, __half>) {
        return __half2float(__ushort_as_half(static_cast<unsigned short>(word & 0xffffU)));
    } else {
        return __uint_as_float(word << 16);
    }
}

template <typename InputT>
__device__ __forceinline__ float f16KRWidenHigh(uint32_t word) {
    if constexpr (std::is_same_v<InputT, __half>) {
        return __half2float(__ushort_as_half(static_cast<unsigned short>(word >> 16)));
    } else {
        return __uint_as_float(word & 0xffff0000U);
    }
}

template <typename InputT, int FixedOp, int FixedInputTransform>
__device__ __forceinline__ float2 f16KRDecodeWord(uint32_t word, const Fp8ExtremumFold& fold) {
    const uint32_t folded = foldFp8ExtremumBytes<FixedOp == 2>(word, fold);
    float2 value{f16KRWidenLow<InputT>(folded), f16KRWidenHigh<InputT>(folded)};
    if constexpr (FixedInputTransform == 1) {
        value.x = ::fabsf(value.x);
        value.y = ::fabsf(value.y);
    } else if constexpr (FixedInputTransform == 2) {
        value.x *= value.x;
        value.y *= value.y;
    }
    return value;
}

template <typename InputT>
inline constexpr bool IS_F16_KR_INPUT = std::is_same_v<InputT, __half> || std::is_same_v<InputT, __nv_bfloat16>;

// The KR input element types: FP8 (four per word), FP32 (one per word) and FP16 / BF16 (two per word).
template <typename InputT>
inline constexpr bool IS_KR_INPUT =
    sizeof(InputT) == 1 || std::is_same_v<InputT, float> || IS_F16_KR_INPUT<InputT>;

template <int FixedOp>
__device__ __forceinline__ void fp8KRStoreValue(void* output, DataType output_dtype, uint64_t index, float value,
                                                const Fp8ExtremumFold& fold, AdditiveFinalizeFp32 output_finalize,
                                                float output_scale) {
    storeFp8CompleteResult<FixedOp == 0>(output, output_dtype, index,
                                         unfoldFp8ExtremumResult<FixedOp == 2>(value, fold), output_finalize,
                                         output_scale);
}

// Accumulates the words [first_word, word_end) a lane visits with stride `step`, masking bytes outside `bounds`.
// FP32 rows are whole words on the word grid (the launcher requires a 4-byte aligned input, so head is 0), so every
// visited word is inside and the FP32 branch needs no mask; consecutive loads alternate the two accumulators.
// FP16 / BF16 rows are whole half-words (bounds and head even), so a word straddling a bound has exactly one half-word
// inside: the low (byte `first`) or the high (`first + 2`) one.
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
    } else if constexpr (IS_F16_KR_INPUT<InputT>) {
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
                    const float2 v = f16KRDecodeWord<InputT, FixedOp, FixedInputTransform>(packed[u], fold);
                    const uint64_t first = w << 2;
                    if (first >= bounds.begin && first + 4 <= bounds.end) {
                        even = combine(even, v.x);
                        odd = combine(odd, v.y);
                    } else {
                        if (fp8KRByteInside(first, bounds)) even = combine(even, v.x);
                        if (fp8KRByteInside(first + 2, bounds)) odd = combine(odd, v.y);
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
__global__ __launch_bounds__(KR::KR_THREADS) void krValueGroupKernel(const uint32_t* __restrict__ words,
                                                                                uint32_t head,
                                                                                uint64_t rows,
                                                                                uint64_t row_bytes,
                                                                                uint32_t group_shift,
                                                                                Fp8ExtremumFold fold,
                                                                                void* output,
                                                                                DataType output_dtype,
                                                                                AdditiveFinalizeFp32 output_finalize,
                                                                                float output_scale) {
    static_assert(IS_KR_INPUT<InputT>);
    const Fp8TypedReductionOp<FixedOp> combine{};
    const uint32_t group_lanes = 1U << group_shift;
    const uint32_t lane = threadIdx.x & (group_lanes - 1U);
    const uint64_t group = threadIdx.x >> group_shift;
    const uint64_t groups_per_block = KR::KR_THREADS >> group_shift;
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

// ---- Pack family output staging. A scalar result store expands into the runtime output-dtype switch (FP8 / FP16 /
// BF16 / FP32 / FP64 stores and their branches) at every unrolled store site: storing directly, the ARG Pack kernel had
// 416 STG and 1023 branches in 6612 instructions, a 16-row VALUE vector kernel 172 STG and 390 branches. Pack kernels
// stage each block's results in shared memory (one word per row; pad one word per 32 so the row-strided per-thread
// writes do not bank-conflict) and write them from ONE store site in a block-strided loop, which also makes consecutive
// threads store consecutive rows.
constexpr uint32_t FP8_KR_PACK_MAX_ROWS = 16;  // KRSegments.h legality: rows_per_thread <= 16
constexpr uint32_t FP8_KR_PACK_STAGE_WORDS =
    KR::KR_THREADS * FP8_KR_PACK_MAX_ROWS + KR::KR_THREADS * FP8_KR_PACK_MAX_ROWS / 32;

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
    for (uint32_t i = threadIdx.x; i < count; i += KR::KR_THREADS) {
        storeFp32AsRuntimeDType(output, output_dtype, first_row + i,
                                fp8KRFinalValue<FixedOp>(staged[fp8KRStageSlot(i)], fold, output_finalize,
                                                         output_scale));
    }
}

__device__ __forceinline__ void fp8KRWriteStagedArgs(const uint32_t* staged, uint64_t first_row, uint32_t count,
                                                     const Fp8ArgOutputs& outputs) {
#pragma unroll 1
    for (uint32_t i = threadIdx.x; i < count; i += KR::KR_THREADS) {
        storeFp8ArgPackedResult(outputs, first_row + i, fp8ArgWiden(staged[fp8KRStageSlot(i)], 0));
    }
}

// ---- Pack: one thread owns rows_per_thread (<= 16) consecutive short rows (row_bytes <= 32) and walks their bytes in
// order; each completed row goes to the block's staging buffer. The block loop is uniform, so every thread reaches the
// barriers.
template <typename InputT, int FixedOp, int FixedInputTransform>
__global__ __launch_bounds__(KR::KR_THREADS) void krValuePackKernel(const uint32_t* __restrict__ words,
                                                                               uint32_t head,
                                                                               uint64_t rows,
                                                                               uint32_t row_bytes,
                                                                               uint32_t rows_per_thread,
                                                                               Fp8ExtremumFold fold,
                                                                               void* output,
                                                                               DataType output_dtype,
                                                                               AdditiveFinalizeFp32 output_finalize,
                                                                               float output_scale) {
    static_assert(IS_KR_INPUT<InputT>);
    __shared__ float staged[FP8_KR_PACK_STAGE_WORDS];
    const Fp8TypedReductionOp<FixedOp> combine{};
    const float identity = combine.identity();
    const uint64_t owners = (rows + rows_per_thread - 1) / rows_per_thread;
    const uint64_t block_stride = static_cast<uint64_t>(gridDim.x) * KR::KR_THREADS;
    for (uint64_t block_owner = static_cast<uint64_t>(blockIdx.x) * KR::KR_THREADS; block_owner < owners;
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
                        // FP32: one element per word, every word of [begin, end) inside (head is 0); row_bytes / 4
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
                    } else if constexpr (IS_F16_KR_INPUT<InputT>) {
                        // FP16 / BF16: two elements per word; rows of row_bytes / 2 elements may start mid-word (head 0
                        // or 2, odd element counts), so each half-word is checked against [begin, end).
                        if (word + u < word_end) {
                            const float2 v = f16KRDecodeWord<InputT, FixedOp, FixedInputTransform>(packed[u], fold);
                            const float values[2] = {v.x, v.y};
                            const uint64_t first = (word + u) << 2;
#pragma unroll
                            for (uint32_t j = 0; j < 2; ++j) {
                                if (first + 2 * j >= begin && first + 2 * j < end) {
                                    acc = combine(acc, values[j]);
                                    if (++position == (row_bytes >> 1)) {
                                        staged[fp8KRStageSlot(local)] = acc;
                                        acc = identity;
                                        position = 0;
                                        ++local;
                                    }
                                }
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
            block_rows < uint64_t{KR::KR_THREADS} * rows_per_thread ? block_rows
                                                                            : uint64_t{KR::KR_THREADS} *
                                                                                  rows_per_thread);
        fp8KRWriteStagedValues<FixedOp>(staged, block_first_row, count, fold, output, output_dtype, output_finalize,
                                        output_scale);
        __syncthreads();
    }
}

// ---- Vector Pack: one thread owns krVectorRows(RowBytes) rows, a whole number of 16-byte vectors, loaded
// with 16-byte loads (the input is 16-byte aligned; the launcher otherwise runs the generic Pack kernel). Row
// boundaries are compile-time, so the accumulation is straight-line; results leave through the block's staging buffer.
// The last rows % per-thread rows run byte by byte in the final block iteration. FP16 / BF16 too (two elements per
// word, row lengths KR::F16_KR_VECTOR_ROW_BYTES); their partial last owner runs element by element.
template <int RowBytes>
inline constexpr int FP8_KR_VECTOR_ROWS = static_cast<int>(KR::krVectorRows(RowBytes));

template <typename InputT, int FixedOp, int FixedInputTransform, int RowBytes>
__global__ __launch_bounds__(KR::KR_THREADS) void krValueVectorKernel(const uint4* __restrict__ vectors,
                                                                                 uint64_t rows,
                                                                                 Fp8ExtremumFold fold,
                                                                                 void* output,
                                                                                 DataType output_dtype,
                                                                                 AdditiveFinalizeFp32 output_finalize,
                                                                                 float output_scale) {
    static_assert(sizeof(InputT) == 1 || (IS_F16_KR_INPUT<InputT> && RowBytes % 2 == 0));
    constexpr int Rows = FP8_KR_VECTOR_ROWS<RowBytes>;
    constexpr int Vectors = Rows * RowBytes / 16;
    constexpr int Words = 4 * Vectors;
    static_assert(Rows * RowBytes == 16 * Vectors && Rows <= static_cast<int>(FP8_KR_PACK_MAX_ROWS));
    __shared__ float staged[FP8_KR_PACK_STAGE_WORDS];
    const Fp8TypedReductionOp<FixedOp> combine{};
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(vectors);
    const uint64_t owners = (rows + Rows - 1) / Rows;  // the last owner may be partial (byte path)
    const uint64_t full_owners = rows / Rows;
    const uint64_t block_stride = static_cast<uint64_t>(gridDim.x) * KR::KR_THREADS;
    for (uint64_t block_owner = static_cast<uint64_t>(blockIdx.x) * KR::KR_THREADS; block_owner < owners;
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
                if constexpr (IS_F16_KR_INPUT<InputT>) {
                    const float2 v = f16KRDecodeWord<InputT, FixedOp, FixedInputTransform>(words[w], fold);
                    acc[(4 * w) / RowBytes] = combine(acc[(4 * w) / RowBytes], v.x);
                    acc[(4 * w + 2) / RowBytes] = combine(acc[(4 * w + 2) / RowBytes], v.y);
                } else {
                    const float4 v = fp8KRDecodeWord<InputT, FixedOp, FixedInputTransform>(words[w], fold);
                    acc[(4 * w) / RowBytes] = combine(acc[(4 * w) / RowBytes], v.x);
                    acc[(4 * w + 1) / RowBytes] = combine(acc[(4 * w + 1) / RowBytes], v.y);
                    acc[(4 * w + 2) / RowBytes] = combine(acc[(4 * w + 2) / RowBytes], v.z);
                    acc[(4 * w + 3) / RowBytes] = combine(acc[(4 * w + 3) / RowBytes], v.w);
                }
            }
#pragma unroll
            for (int r = 0; r < Rows; ++r) staged[fp8KRStageSlot(threadIdx.x * Rows + r)] = acc[r];
        } else if (owner < owners) {
            // The partial last owner: its rows byte by byte (element by element for FP16 / BF16).
            for (uint64_t row = owner * Rows; row < rows; ++row) {
                float acc = combine.identity();
                if constexpr (IS_F16_KR_INPUT<InputT>) {
                    const uint16_t* elements = reinterpret_cast<const uint16_t*>(bytes + row * RowBytes);
                    for (int i = 0; i < RowBytes / 2; ++i) {
                        const uint32_t element = __ldg(elements + i);
                        acc = combine(acc, f16KRDecodeWord<InputT, FixedOp, FixedInputTransform>(element, fold).x);
                    }
                } else {
                    for (int i = 0; i < RowBytes; ++i) {
                        const uint32_t byte = __ldg(bytes + row * RowBytes + i);
                        acc = combine(acc, fp8KRDecodeWord<InputT, FixedOp, FixedInputTransform>(byte, fold).x);
                    }
                }
                staged[fp8KRStageSlot(static_cast<uint32_t>(row - block_first_row))] = acc;
            }
        }
        __syncthreads();
        const uint64_t block_rows = rows - block_first_row;
        const uint32_t count = static_cast<uint32_t>(
            block_rows < uint64_t{KR::KR_THREADS} * Rows ? block_rows : uint64_t{KR::KR_THREADS} * Rows);
        fp8KRWriteStagedValues<FixedOp>(staged, block_first_row, count, fold, output, output_dtype, output_finalize,
                                        output_scale);
        __syncthreads();
    }
}

template <int RowBytes>
__global__ __launch_bounds__(KR::KR_THREADS) void fp8KRArgVectorKernel(const uint4* __restrict__ vectors,
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
    const uint64_t block_stride = static_cast<uint64_t>(gridDim.x) * KR::KR_THREADS;
    for (uint64_t block_owner = static_cast<uint64_t>(blockIdx.x) * KR::KR_THREADS; block_owner < owners;
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
            block_rows < uint64_t{KR::KR_THREADS} * Rows ? block_rows : uint64_t{KR::KR_THREADS} * Rows);
        fp8KRWriteStagedArgs(staged, block_first_row, count, outputs);
        __syncthreads();
    }
}

// The compiled vector row lengths (KR::FP8_KR_VECTOR_ROW_BYTES); false when RowBytes has no kernel.
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

// The FP16 / BF16 vector row lengths (KR::F16_KR_VECTOR_ROW_BYTES: 1 to 8 elements).
template <typename LaunchT>
bool dispatchF16KRVectorRowBytes(uint64_t row_bytes, const LaunchT& launch) {
    switch (row_bytes) {
        case 2: launch.template operator()<2>(); return true;
        case 4: launch.template operator()<4>(); return true;
        case 6: launch.template operator()<6>(); return true;
        case 8: launch.template operator()<8>(); return true;
        case 10: launch.template operator()<10>(); return true;
        case 12: launch.template operator()<12>(); return true;
        case 14: launch.template operator()<14>(); return true;
        case 16: launch.template operator()<16>(); return true;
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
__global__ __launch_bounds__(KR::KR_THREADS) void krValueSplitKernel(const uint32_t* __restrict__ words,
                                                                                uint32_t head,
                                                                                uint64_t rows,
                                                                                uint64_t row_bytes,
                                                                                uint64_t chunk_bytes,
                                                                                uint64_t chunks,
                                                                                Fp8ExtremumFold fold,
                                                                                float* __restrict__ partials) {
    static_assert(IS_KR_INPUT<InputT>);
    __shared__ float warp_totals[FP8_KR_WARPS];
    const Fp8TypedReductionOp<FixedOp> combine{};
    const uint64_t segments = rows * chunks;
    for (uint64_t segment = blockIdx.x; segment < segments; segment += gridDim.x) {
        const Fp8KRSegmentBounds bounds = fp8KRSegmentBounds(segment, chunks, row_bytes, chunk_bytes, head);
        float even = combine.identity();
        float odd = combine.identity();
        fp8KRAccumulateWords<InputT, FixedOp, FixedInputTransform>(words, (bounds.begin >> 2) + threadIdx.x,
                                                                   (bounds.end + 3) >> 2, KR::KR_THREADS,
                                                                   bounds, fold, even, odd);
        const float chunk_total = fp8KRBlockCombine<FixedOp>(combine(even, odd), warp_totals);
        if (threadIdx.x == 0) partials[segment] = chunk_total;
        __syncthreads();  // warp_totals is rewritten by the next segment
    }
}

template <int FixedOp>
__global__ __launch_bounds__(KR::KR_THREADS) void krValueFinishKernel(const float* __restrict__ partials,
                                                                                 uint64_t rows,
                                                                                 uint64_t chunks,
                                                                                 Fp8ExtremumFold fold,
                                                                                 void* output,
                                                                                 DataType output_dtype,
                                                                                 AdditiveFinalizeFp32 output_finalize,
                                                                                 float output_scale) {
    __shared__ float warp_totals[FP8_KR_WARPS];
    const Fp8TypedReductionOp<FixedOp> combine{};
    constexpr uint64_t Threads = KR::KR_THREADS;
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

// ---- Block: one CTA per whole row, stored directly (the Split partial kernel's mapping with one chunk, no
// scratch, no finishing launch).
template <typename InputT, int FixedOp, int FixedInputTransform>
__global__ __launch_bounds__(KR::KR_THREADS) void krValueBlockKernel(const uint32_t* __restrict__ words,
                                                                                uint32_t head,
                                                                                uint64_t rows,
                                                                                uint64_t row_bytes,
                                                                                Fp8ExtremumFold fold,
                                                                                void* output,
                                                                                DataType output_dtype,
                                                                                AdditiveFinalizeFp32 output_finalize,
                                                                                float output_scale) {
    static_assert(IS_KR_INPUT<InputT>);
    __shared__ float warp_totals[FP8_KR_WARPS];
    const Fp8TypedReductionOp<FixedOp> combine{};
    for (uint64_t row = blockIdx.x; row < rows; row += gridDim.x) {
        const Fp8KRSegmentBounds bounds = fp8KRSegmentBounds(row, 1, row_bytes, row_bytes, head);
        float even = combine.identity();
        float odd = combine.identity();
        fp8KRAccumulateWords<InputT, FixedOp, FixedInputTransform>(words, (bounds.begin >> 2) + threadIdx.x,
                                                                   (bounds.end + 3) >> 2, KR::KR_THREADS,
                                                                   bounds, fold, even, odd);
        const float row_total = fp8KRBlockCombine<FixedOp>(combine(even, odd), warp_totals);
        if (threadIdx.x == 0) {
            fp8KRStoreValue<FixedOp>(output, output_dtype, row, row_total, fold, output_finalize, output_scale);
        }
        __syncthreads();  // warp_totals is rewritten by the next row
    }
}

// ---- FP16 / BF16 R / KR ARG shares FP8's Group / Block /
// Pack / Split / Finish kernels through an element-format trait (keys per word, the candidate words, the value store).
// Both formats use one 64-bit candidate (key << (index bits + 1) | row-local index << 1 | -0: FP8 key8 and a 55-bit
// index, 2-byte key16 and a 47-bit index, F16ArgKey.h) and the same unsigned minimum, so the kernels' mapping, warp and
// block combines and Split partials are shared; Split partials hold row-local indices, so Finish is a minimum only.

struct F16ArgOutputs {
    void* value;  // null: no value output
    DataType value_dtype;
    void* index;  // null: no index output
    DataType index_dtype;
    uint64_t domain_stride;  // the reduced run's original ARG-domain stride: index = row-local index * domain_stride
    F16ArgFold fold;
    bool bf16;
};

__device__ __forceinline__ float decodeF16ArgBits(uint32_t bits, bool bf16) {
    if (bf16) return __uint_as_float(bits << 16);
    return __half2float(__ushort_as_half(static_cast<unsigned short>(bits)));
}

// Writes one reduced 2-byte candidate to the requested outputs at `output`.
__device__ __forceinline__ void storeF16ArgPackedResult(const F16ArgOutputs& outputs, uint64_t output, uint64_t best) {
    if (outputs.value != nullptr) {
        const uint32_t bits = f16ArgValueBits(f16ArgWideKey(best), f16ArgWideNegativeZero(best), outputs.fold);
        storeFp32AsRuntimeDType(outputs.value, outputs.value_dtype, output, decodeF16ArgBits(bits, outputs.bf16));
    }
    if (outputs.index != nullptr) {
        const uint64_t index = f16ArgWideIndex(best) * outputs.domain_stride;
        if (outputs.index_dtype == DataType::UINT32) {
            static_cast<uint32_t*>(outputs.index)[output] = static_cast<uint32_t>(index);
        } else {
            static_cast<uint64_t*>(outputs.index)[output] = index;
        }
    }
}

// fp8KRArgAccumulateWords at two elements per word: half-words outside `bounds` are skipped (rows are whole
// half-words, so a word straddling a bound has one half inside); the index is row-local.
__device__ __forceinline__ void f16KRArgAccumulateWords(const uint32_t* __restrict__ words, uint64_t first_word,
                                                        uint64_t word_end, uint64_t step,
                                                        const Fp8KRSegmentBounds& bounds, F16ArgFold fold,
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
                const F16ArgKeys2 keys = f16ArgKeys2(packed[u], fold);
                const uint64_t first = w << 2;
                if (first >= bounds.begin && first + 4 <= bounds.end) {
                    const uint64_t index = (first - bounds.row_begin) >> 1;
                    best = fp8ArgMin(best, fp8ArgMin(f16ArgWideCandidate(keys, 0, index),
                                                     f16ArgWideCandidate(keys, 1, index + 1)));
                } else {
                    if (fp8KRByteInside(first, bounds))
                        best = fp8ArgMin(best, f16ArgWideCandidate(keys, 0, (first - bounds.row_begin) >> 1));
                    if (fp8KRByteInside(first + 2, bounds))
                        best = fp8ArgMin(best, f16ArgWideCandidate(keys, 1, (first + 2 - bounds.row_begin) >> 1));
                }
            }
        }
    }
}

// The element-format traits of the shared ARG kernels.
struct Fp8KRArgFormat {
    using Fold = Fp8ArgFold;
    using Outputs = Fp8ArgOutputs;
    using Keys = Fp8ArgKeys4;
    static constexpr uint32_t ELEMENT_BYTES = 1;
    static constexpr uint32_t ELEMENTS_PER_WORD = 4;
    static constexpr uint64_t WIDE_IDENTITY = FP8_ARG_WIDE_IDENTITY;
    static constexpr uint32_t LANE_IDENTITY = FP8_ARG_LANE_IDENTITY;
    __device__ __forceinline__ static Keys keys(uint32_t packed, Fold fold) { return fp8ArgKeys4(packed, fold); }
    __device__ __forceinline__ static uint32_t laneCandidate(Keys keys, uint32_t element, uint32_t index) {
        return fp8ArgLaneCandidate(keys, element, index);
    }
    __device__ __forceinline__ static uint64_t widenLane(uint32_t lane) { return fp8ArgWiden(lane, 0); }
    __device__ __forceinline__ static void accumulate(const uint32_t* __restrict__ words, uint64_t first_word,
                                                      uint64_t word_end, uint64_t step,
                                                      const Fp8KRSegmentBounds& bounds, Fold fold, uint64_t& best) {
        fp8KRArgAccumulateWords(words, first_word, word_end, step, bounds, fold, best);
    }
    __device__ __forceinline__ static void store(const Outputs& outputs, uint64_t output, uint64_t best) {
        storeFp8ArgPackedResult(outputs, output, best);
    }
};

struct F16KRArgFormat {
    using Fold = F16ArgFold;
    using Outputs = F16ArgOutputs;
    using Keys = F16ArgKeys2;
    static constexpr uint32_t ELEMENT_BYTES = 2;
    static constexpr uint32_t ELEMENTS_PER_WORD = 2;
    static constexpr uint64_t WIDE_IDENTITY = F16_ARG_WIDE_IDENTITY;
    static constexpr uint32_t LANE_IDENTITY = F16_ARG_LANE_IDENTITY;
    __device__ __forceinline__ static Keys keys(uint32_t packed, Fold fold) { return f16ArgKeys2(packed, fold); }
    __device__ __forceinline__ static uint32_t laneCandidate(Keys keys, uint32_t element, uint32_t index) {
        return f16ArgLaneCandidate(keys, element, index);
    }
    __device__ __forceinline__ static uint64_t widenLane(uint32_t lane) { return f16ArgWidenLane(lane); }
    __device__ __forceinline__ static void accumulate(const uint32_t* __restrict__ words, uint64_t first_word,
                                                      uint64_t word_end, uint64_t step,
                                                      const Fp8KRSegmentBounds& bounds, Fold fold, uint64_t& best) {
        f16KRArgAccumulateWords(words, first_word, word_end, step, bounds, fold, best);
    }
    __device__ __forceinline__ static void store(const Outputs& outputs, uint64_t output, uint64_t best) {
        storeF16ArgPackedResult(outputs, output, best);
    }
};
static_assert(FP8_ARG_WIDE_IDENTITY == F16_ARG_WIDE_IDENTITY);  // fp8KRBlockArgMin's identity serves both

template <typename ArgT>
__device__ __forceinline__ void retainedKRWriteStagedArgs(const uint32_t* staged, uint64_t first_row, uint32_t count,
                                                          const typename ArgT::Outputs& outputs) {
#pragma unroll 1
    for (uint32_t i = threadIdx.x; i < count; i += KR::KR_THREADS) {
        ArgT::store(outputs, first_row + i, ArgT::widenLane(staged[fp8KRStageSlot(i)]));
    }
}

template <typename ArgT>
__global__ __launch_bounds__(KR::KR_THREADS) void krArgBlockKernel(const uint32_t* __restrict__ words,
                                                                              uint32_t head,
                                                                              uint64_t rows,
                                                                              uint64_t row_bytes,
                                                                              typename ArgT::Fold fold,
                                                                              typename ArgT::Outputs outputs) {
    __shared__ uint64_t warp_bests[FP8_KR_WARPS];
    for (uint64_t row = blockIdx.x; row < rows; row += gridDim.x) {
        const Fp8KRSegmentBounds bounds = fp8KRSegmentBounds(row, 1, row_bytes, row_bytes, head);
        uint64_t best = ArgT::WIDE_IDENTITY;
        ArgT::accumulate(words, (bounds.begin >> 2) + threadIdx.x, (bounds.end + 3) >> 2, KR::KR_THREADS,
                         bounds, fold, best);
        const uint64_t row_best = fp8KRBlockArgMin(best, warp_bests);
        if (threadIdx.x == 0) ArgT::store(outputs, row, row_best);
        __syncthreads();
    }
}

// ---- ARG Group: G lanes per row.
template <typename ArgT>
__global__ __launch_bounds__(KR::KR_THREADS) void krArgGroupKernel(const uint32_t* __restrict__ words,
                                                                              uint32_t head,
                                                                              uint64_t rows,
                                                                              uint64_t row_bytes,
                                                                              uint32_t group_shift,
                                                                              typename ArgT::Fold fold,
                                                                              typename ArgT::Outputs outputs) {
    const uint32_t group_lanes = 1U << group_shift;
    const uint32_t lane = threadIdx.x & (group_lanes - 1U);
    const uint64_t group = threadIdx.x >> group_shift;
    const uint64_t groups_per_block = KR::KR_THREADS >> group_shift;
    const uint64_t block_stride = static_cast<uint64_t>(gridDim.x) * groups_per_block;
    for (uint64_t base = static_cast<uint64_t>(blockIdx.x) * groups_per_block; base < rows; base += block_stride) {
        const uint64_t row = base + group;
        uint64_t best = ArgT::WIDE_IDENTITY;
        if (row < rows) {
            const Fp8KRSegmentBounds bounds = fp8KRSegmentBounds(row, 1, row_bytes, row_bytes, head);
            ArgT::accumulate(words, (bounds.begin >> 2) + lane, (bounds.end + 3) >> 2, group_lanes, bounds, fold,
                             best);
        }
        for (uint32_t offset = group_lanes >> 1; offset != 0; offset >>= 1) {
            const uint64_t other =
                __shfl_xor_sync(0xffffffffU, static_cast<unsigned long long>(best), static_cast<int>(offset));
            best = fp8ArgMin(best, other);
        }
        if (lane == 0 && row < rows) ArgT::store(outputs, row, best);
    }
}

// ---- ARG Pack: per row, the 32-bit lane candidate (key << 24 | row-local index << 1 | -0) of each byte; row_bytes
// <= 32 keeps the index far below FP8_ARG_MAX_LANE_ROWS. Completed rows go to the block's staging buffer.
// Per element of the format (2-byte lanes: key << 16 | index << 1 | -0, at most 16 elements per row).
template <typename ArgT>
__global__ __launch_bounds__(KR::KR_THREADS) void krArgPackKernel(const uint32_t* __restrict__ words,
                                                                             uint32_t head,
                                                                             uint64_t rows,
                                                                             uint32_t row_bytes,
                                                                             uint32_t rows_per_thread,
                                                                             typename ArgT::Fold fold,
                                                                             typename ArgT::Outputs outputs) {
    __shared__ uint32_t staged[FP8_KR_PACK_STAGE_WORDS];
    const uint64_t owners = (rows + rows_per_thread - 1) / rows_per_thread;
    const uint64_t block_stride = static_cast<uint64_t>(gridDim.x) * KR::KR_THREADS;
    for (uint64_t block_owner = static_cast<uint64_t>(blockIdx.x) * KR::KR_THREADS; block_owner < owners;
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
            uint32_t best = ArgT::LANE_IDENTITY;
            uint32_t position = 0;
            const uint32_t row_elements = row_bytes / ArgT::ELEMENT_BYTES;
            for (uint64_t word = begin >> 2; word < word_end; word += FP8_KR_UNROLL) {
                uint32_t packed[FP8_KR_UNROLL];
#pragma unroll
                for (uint32_t u = 0; u < FP8_KR_UNROLL; ++u) {
                    packed[u] = word + u < word_end ? __ldg(words + word + u) : 0U;
                }
#pragma unroll
                for (uint32_t u = 0; u < FP8_KR_UNROLL; ++u) {
                    if (word + u < word_end) {
                        const typename ArgT::Keys keys = ArgT::keys(packed[u], fold);
                        const uint64_t first = (word + u) << 2;
#pragma unroll
                        for (uint32_t j = 0; j < ArgT::ELEMENTS_PER_WORD; ++j) {
                            const uint64_t byte = first + j * ArgT::ELEMENT_BYTES;
                            if (byte >= begin && byte < end) {
                                best = fp8ArgMin(best, ArgT::laneCandidate(keys, j, position));
                                if (++position == row_elements) {
                                    staged[fp8KRStageSlot(local)] = best;
                                    best = ArgT::LANE_IDENTITY;
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
            block_rows < uint64_t{KR::KR_THREADS} * rows_per_thread ? block_rows
                                                                            : uint64_t{KR::KR_THREADS} *
                                                                                  rows_per_thread);
        retainedKRWriteStagedArgs<ArgT>(staged, block_first_row, count, outputs);
        __syncthreads();
    }
}

// ---- ARG Split: as VALUE Split, with 64-bit wide candidates and the order-free wide minimum.
template <typename ArgT>
__global__ __launch_bounds__(KR::KR_THREADS) void krArgSplitKernel(const uint32_t* __restrict__ words,
                                                                              uint32_t head,
                                                                              uint64_t rows,
                                                                              uint64_t row_bytes,
                                                                              uint64_t chunk_bytes,
                                                                              uint64_t chunks,
                                                                              typename ArgT::Fold fold,
                                                                              uint64_t* __restrict__ partials) {
    __shared__ uint64_t warp_bests[FP8_KR_WARPS];
    const uint64_t segments = rows * chunks;
    for (uint64_t segment = blockIdx.x; segment < segments; segment += gridDim.x) {
        const Fp8KRSegmentBounds bounds = fp8KRSegmentBounds(segment, chunks, row_bytes, chunk_bytes, head);
        uint64_t best = ArgT::WIDE_IDENTITY;
        ArgT::accumulate(words, (bounds.begin >> 2) + threadIdx.x, (bounds.end + 3) >> 2, KR::KR_THREADS,
                         bounds, fold, best);
        const uint64_t chunk_best = fp8KRBlockArgMin(best, warp_bests);
        if (threadIdx.x == 0) partials[segment] = chunk_best;
        __syncthreads();
    }
}

template <typename ArgT>
__global__ __launch_bounds__(KR::KR_THREADS) void krArgFinishKernel(const uint64_t* __restrict__ partials,
                                                                               uint64_t rows,
                                                                               uint64_t chunks,
                                                                               typename ArgT::Outputs outputs) {
    __shared__ uint64_t warp_bests[FP8_KR_WARPS];
    constexpr uint64_t Threads = KR::KR_THREADS;
    for (uint64_t row = blockIdx.x; row < rows; row += gridDim.x) {
        const uint64_t* row_partials = partials + row * chunks;
        uint64_t a = ArgT::WIDE_IDENTITY;
        uint64_t b = ArgT::WIDE_IDENTITY;
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
        if (threadIdx.x == 0) ArgT::store(outputs, row, row_best);
        __syncthreads();
    }
}

// ---- FP32 R / KR ARG. Copies of the FP8 ARG Group / Block / Pack / Split / Finish kernels (the candidate differs, not
// just the input type, so they live independently). One element per word; rows are whole words on a 4-byte aligned
// input.
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
// is FP8's (krScratchBytes(shape, true)); the finishing kernel restores the row-local index from the chunk.
// The candidate, its key, value, outputs and warp shuffle are in Fp32ArgKey.cuh (shared with FP32 RK ARG).

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
__global__ __launch_bounds__(KR::KR_THREADS) void fp32KRArgGroupKernel(const uint32_t* __restrict__ words,
                                                                               uint64_t rows,
                                                                               uint64_t row_words,
                                                                               uint32_t group_shift,
                                                                               Fp32ArgOutputs outputs) {
    const uint32_t group_lanes = 1U << group_shift;
    const uint32_t lane = threadIdx.x & (group_lanes - 1U);
    const uint64_t group = threadIdx.x >> group_shift;
    const uint64_t groups_per_block = KR::KR_THREADS >> group_shift;
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
__global__ __launch_bounds__(KR::KR_THREADS) void fp32KRArgBlockKernel(const uint32_t* __restrict__ words,
                                                                               uint64_t rows,
                                                                               uint64_t row_words,
                                                                               Fp32ArgOutputs outputs) {
    __shared__ uint32_t warp_keys[FP8_KR_WARPS];
    __shared__ uint64_t warp_tails[FP8_KR_WARPS];
    for (uint64_t row = blockIdx.x; row < rows; row += gridDim.x) {
        const uint64_t row_word = row * row_words;
        Fp32ArgCandidate best = fp32ArgIdentity();
        fp32KRArgAccumulateWords(words, row_word + threadIdx.x, row_word + row_words, KR::KR_THREADS,
                                 row_word, outputs.xor_mask, best);
        const Fp32ArgCandidate row_best = fp32KRBlockArgMin(best, warp_keys, warp_tails);
        if (threadIdx.x == 0) storeFp32ArgResult(outputs, row, row_best);
        __syncthreads();  // the warp slots are rewritten by the next row
    }
}

// ---- FP32 ARG Pack: one thread owns rows_per_thread consecutive rows of at most 8 elements. A row's best is one
// 64-bit word (key << 32 | row-local index << 1 | -0); completed rows go to the block's staging buffer.
constexpr uint32_t FP32_KR_ARG_PACK_STAGE_WORDS = FP8_KR_PACK_STAGE_WORDS;
__global__ __launch_bounds__(KR::KR_THREADS) void fp32KRArgPackKernel(const uint32_t* __restrict__ words,
                                                                              uint64_t rows,
                                                                              uint32_t row_words,
                                                                              uint32_t rows_per_thread,
                                                                              Fp32ArgOutputs outputs) {
    __shared__ uint64_t staged[FP32_KR_ARG_PACK_STAGE_WORDS];
    const uint64_t owners = (rows + rows_per_thread - 1) / rows_per_thread;
    const uint64_t block_stride = static_cast<uint64_t>(gridDim.x) * KR::KR_THREADS;
    for (uint64_t block_owner = static_cast<uint64_t>(blockIdx.x) * KR::KR_THREADS; block_owner < owners;
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
            block_rows < uint64_t{KR::KR_THREADS} * rows_per_thread ? block_rows
                                                                            : uint64_t{KR::KR_THREADS} *
                                                                                  rows_per_thread);
#pragma unroll 1
        for (uint32_t i = threadIdx.x; i < count; i += KR::KR_THREADS) {
            const uint64_t packed_best = staged[fp8KRStageSlot(i)];
            storeFp32ArgResult(outputs, block_first_row + i,
                               {static_cast<uint32_t>(packed_best >> 32), packed_best & 0xffffffffULL});
        }
        __syncthreads();
    }
}

// ---- FP32 ARG Split: one CTA per (row, chunk) writes the chunk's packed partial; the finishing kernel (one CTA per
// row) restores row-local indices from the chunk and takes the full minimum.
__global__ __launch_bounds__(KR::KR_THREADS) void fp32KRArgSplitKernel(const uint32_t* __restrict__ words,
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
        fp32KRArgAccumulateWords(words, chunk_word + threadIdx.x, chunk_end, KR::KR_THREADS, chunk_word,
                                 xor_mask, best);
        const Fp32ArgCandidate chunk_best = fp32KRBlockArgMin(best, warp_keys, warp_tails);
        if (threadIdx.x == 0) partials[segment] = (static_cast<uint64_t>(chunk_best.key) << 32) | chunk_best.tail;
        __syncthreads();
    }
}

__global__ __launch_bounds__(KR::KR_THREADS) void fp32KRArgFinishKernel(const uint64_t* __restrict__ partials,
                                                                                uint64_t rows,
                                                                                uint64_t chunks,
                                                                                uint64_t chunk_words,
                                                                                Fp32ArgOutputs outputs) {
    __shared__ uint32_t warp_keys[FP8_KR_WARPS];
    __shared__ uint64_t warp_tails[FP8_KR_WARPS];
    for (uint64_t row = blockIdx.x; row < rows; row += gridDim.x) {
        const uint64_t* row_partials = partials + row * chunks;
        Fp32ArgCandidate best = fp32ArgIdentity();
        for (uint64_t chunk = threadIdx.x; chunk < chunks; chunk += KR::KR_THREADS) {
            const uint64_t partial = row_partials[chunk];
            best = fp32ArgMin(best, {static_cast<uint32_t>(partial >> 32),
                                     ((chunk * chunk_words) << 1) + (partial & 0xffffffffULL)});
        }
        const Fp32ArgCandidate row_best = fp32KRBlockArgMin(best, warp_keys, warp_tails);
        if (threadIdx.x == 0) storeFp32ArgResult(outputs, row, row_best);
        __syncthreads();
    }
}

// ---- Carried-index FP32 R / KR ARG. The FP32 KR ARG families
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
__global__ __launch_bounds__(KR::KR_THREADS) void krCarriedArgGroupKernel(
    const uint32_t* __restrict__ words, const IndexT* __restrict__ carried, uint64_t rows, uint64_t row_words,
    uint32_t group_shift, uint64_t row_stride, Fp32ArgOutputs outputs) {
    const uint32_t group_lanes = 1U << group_shift;
    const uint32_t lane = threadIdx.x & (group_lanes - 1U);
    const uint64_t group = threadIdx.x >> group_shift;
    const uint64_t groups_per_block = KR::KR_THREADS >> group_shift;
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
__global__ __launch_bounds__(KR::KR_THREADS) void krCarriedArgBlockKernel(
    const uint32_t* __restrict__ words, const IndexT* __restrict__ carried, uint64_t rows, uint64_t row_words,
    uint64_t row_stride, Fp32ArgOutputs outputs) {
    __shared__ uint32_t warp_keys[FP8_KR_WARPS];
    __shared__ uint64_t warp_tails[FP8_KR_WARPS];
    for (uint64_t row = blockIdx.x; row < rows; row += gridDim.x) {
        const uint64_t row_word = row * row_words;
        Fp32ArgCandidate best = fp32ArgIdentity();
        fp32KRCarriedArgAccumulateWords<IndexT>(words, carried, row_word + threadIdx.x, row_word + row_words,
                                                KR::KR_THREADS, row_word, outputs.xor_mask, row_stride, best);
        const Fp32ArgCandidate row_best = fp32KRBlockArgMin(best, warp_keys, warp_tails);
        if (threadIdx.x == 0) storeFp32ArgResult(outputs, row, row_best);
        __syncthreads();  // the warp slots are rewritten by the next row
    }
}

// ---- Carried Pack: one thread owns rows_per_thread consecutive rows of at most 8 elements and keeps each row's best
// (key, composed tail) and its position; the stage holds key << 32 | position << 1 | negative zero per row.
template <typename IndexT>
__global__ __launch_bounds__(KR::KR_THREADS) void krCarriedArgPackKernel(
    const uint32_t* __restrict__ words, const IndexT* __restrict__ carried, uint64_t rows, uint32_t row_words,
    uint32_t rows_per_thread, uint64_t row_stride, Fp32ArgOutputs outputs) {
    __shared__ uint64_t staged[FP32_KR_ARG_PACK_STAGE_WORDS];
    const uint64_t owners = (rows + rows_per_thread - 1) / rows_per_thread;
    const uint64_t block_stride = static_cast<uint64_t>(gridDim.x) * KR::KR_THREADS;
    for (uint64_t block_owner = static_cast<uint64_t>(blockIdx.x) * KR::KR_THREADS; block_owner < owners;
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
            block_rows < uint64_t{KR::KR_THREADS} * rows_per_thread ? block_rows
                                                                            : uint64_t{KR::KR_THREADS} *
                                                                                  rows_per_thread);
#pragma unroll 1
        for (uint32_t i = threadIdx.x; i < count; i += KR::KR_THREADS) {
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
__global__ __launch_bounds__(KR::KR_THREADS) void krCarriedArgSplitKernel(
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
                                                KR::KR_THREADS, row_word, xor_mask, row_stride, best);
        const Fp32ArgCandidate chunk_best = fp32KRBlockArgMin(best, warp_keys, warp_tails);
        if (threadIdx.x == 0) {
            partials.keys[segment] = chunk_best.key;
            partials.tails[segment] = chunk_best.tail;
        }
        __syncthreads();
    }
}

__global__ __launch_bounds__(KR::KR_THREADS) void krCarriedArgFinishKernel(Fp32ArgWidePartials partials,
                                                                                       uint64_t rows,
                                                                                       uint64_t chunks,
                                                                                       Fp32ArgOutputs outputs) {
    __shared__ uint32_t warp_keys[FP8_KR_WARPS];
    __shared__ uint64_t warp_tails[FP8_KR_WARPS];
    for (uint64_t row = blockIdx.x; row < rows; row += gridDim.x) {
        const uint64_t first = row * chunks;
        Fp32ArgCandidate best = fp32ArgIdentity();
        for (uint64_t chunk = threadIdx.x; chunk < chunks; chunk += KR::KR_THREADS) {
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

Fp8KRLaunchGeometry fp8KRLaunchGeometry(const StridedSpan& input, const KR::KRSegmentShape& shape) {
    const uintptr_t address = reinterpret_cast<uintptr_t>(input.getMemPtr<void>());
    uint32_t shift = 0;
    uint64_t blocks = 0;
    switch (shape.family()) {
        case KR::KRFamily::Pack:
            blocks = ceilDivideU64(ceilDivideU64(shape.rows, shape.rows_per_thread), KR::KR_THREADS);
            break;
        case KR::KRFamily::Group:
            while ((1U << shift) < shape.group_lanes) ++shift;
            blocks = ceilDivideU64(shape.rows, KR::KR_THREADS >> shift);
            break;
        case KR::KRFamily::Split:
            blocks = shape.segments();
            break;
        case KR::KRFamily::Block:
            blocks = shape.rows;
            break;
    }
    return {reinterpret_cast<const uint32_t*>(address & ~uintptr_t{3}), static_cast<uint32_t>(address & uintptr_t{3}),
            shift, static_cast<unsigned int>(std::min<uint64_t>(blocks, FP8_KR_MAX_GRID_BLOCKS))};
}

unsigned int fp8KRFinishBlocks(uint64_t rows) {
    return static_cast<unsigned int>(std::min<uint64_t>(rows, FP8_KR_MAX_GRID_BLOCKS));
}

void requireFp8KRInput(const StridedSpan& input, const KR::KRSegmentShape& shape, void* scratch,
                       uint64_t scratch_bytes, bool arg) {
    const DataType dtype = input.getDataType();
    if ((dtype != DataType::FP8_E4M3 && dtype != DataType::FP8_E5M2) || !KR::fp8KRSegmentShapeLegal(shape) ||
        input.getTotalNumElements() != shape.rows * shape.row_bytes) {
        throw std::logic_error("Retained FP8 KR pass received an unsupported input or segment shape.");
    }
    if (shape.family() == KR::KRFamily::Split &&
        (scratch == nullptr || scratch_bytes < KR::krScratchBytes(shape, arg) ||
         (reinterpret_cast<uintptr_t>(scratch) & uintptr_t{15}) != 0)) {
        throw std::logic_error("Retained FP8 KR Split pass requires its 16-byte aligned partial scratch.");
    }
}

// The VALUE launch for one input element type (FP8 E4M3 / E5M2, FP32, FP16 / BF16): the family kernel at the semantics.
template <typename InputT>
void launchKRValueTyped(const ReductionStageSemantics& semantics, const StridedSpan& input, const StridedSpan& output,
                        const KR::KRSegmentShape& shape, void* scratch, float output_scale,
                        Stream& stream, KR::KRSplitStages stages) {
    const AdditiveFinalizeFp32 output_finalize = fp8CompleteFinalizerFromSemantics(semantics);
    const Fp8KRLaunchGeometry launch = fp8KRLaunchGeometry(input, shape);
    const auto run = [&]<int FixedOp, int FixedInputTransform>(Fp8ExtremumFold fold, float scale) {
        void* out = output.getMemPtr<void>();
        const DataType out_dtype = output.getDataType();
        switch (shape.family()) {
            case KR::KRFamily::Pack:
                if constexpr (sizeof(InputT) == 1 || IS_F16_KR_INPUT<InputT>) {
                    // An input that is not 16-byte aligned runs the vector shape through the generic Pack kernel.
                    if (shape.vector && (reinterpret_cast<uintptr_t>(input.getMemPtr<void>()) & uintptr_t{15}) == 0) {
                        const auto launch_vector = [&]<int RowBytes>() {
                            krValueVectorKernel<InputT, FixedOp, FixedInputTransform, RowBytes>
                                <<<launch.blocks, KR::KR_THREADS, 0, stream.getStream()>>>(
                                    static_cast<const uint4*>(input.getMemPtr<void>()), shape.rows, fold, out,
                                    out_dtype, output_finalize, scale);
                        };
                        bool launched = false;
                        if constexpr (sizeof(InputT) == 1) {
                            launched = dispatchFp8KRVectorRowBytes(shape.row_bytes, launch_vector);
                        } else {
                            launched = dispatchF16KRVectorRowBytes(shape.row_bytes, launch_vector);
                        }
                        if (!launched)
                            throw std::logic_error("Retained KR vector Pack has no kernel for this row length.");
                        break;
                    }
                } else if (shape.vector) {
                    throw std::logic_error("Retained FP32 KR has no vector Pack.");
                }
                krValuePackKernel<InputT, FixedOp, FixedInputTransform>
                    <<<launch.blocks, KR::KR_THREADS, 0, stream.getStream()>>>(
                        launch.words, launch.head, shape.rows, static_cast<uint32_t>(shape.row_bytes),
                        shape.rows_per_thread, fold, out, out_dtype, output_finalize, scale);
                break;
            case KR::KRFamily::Group:
                krValueGroupKernel<InputT, FixedOp, FixedInputTransform>
                    <<<launch.blocks, KR::KR_THREADS, 0, stream.getStream()>>>(
                        launch.words, launch.head, shape.rows, shape.row_bytes, launch.group_shift, fold, out,
                        out_dtype, output_finalize, scale);
                break;
            case KR::KRFamily::Block:
                krValueBlockKernel<InputT, FixedOp, FixedInputTransform>
                    <<<launch.blocks, KR::KR_THREADS, 0, stream.getStream()>>>(
                        launch.words, launch.head, shape.rows, shape.row_bytes, fold, out, out_dtype,
                        output_finalize, scale);
                break;
            case KR::KRFamily::Split: {
                auto* partials = static_cast<float*>(scratch);
                krValueSplitKernel<InputT, FixedOp, FixedInputTransform>
                    <<<launch.blocks, KR::KR_THREADS, 0, stream.getStream()>>>(
                        launch.words, launch.head, shape.rows, shape.row_bytes, shape.chunk_bytes, shape.chunks(),
                        fold, partials);
                CUDA_CHECK(cudaGetLastError());
                if (stages == KR::KRSplitStages::PartialsOnly) break;
                krValueFinishKernel<FixedOp><<<fp8KRFinishBlocks(shape.rows), KR::KR_THREADS, 0,
                                                  stream.getStream()>>>(partials, shape.rows, shape.chunks(), fold,
                                                                        out, out_dtype, output_finalize, scale);
                break;
            }
        }
        CUDA_CHECK(cudaGetLastError());
    };
    const auto by_semantics = [&]() {
        switch (semantics.combine) {
            case ReductionStageCombine::Sum:
                switch (semantics.input_transform) {
                    case ReductionStageInputTransform::Identity:
                        run.template operator()<0, 0>(FP8_EXTREMUM_FOLD_NONE, output_scale);
                        return;
                    case ReductionStageInputTransform::AbsoluteValue:
                        run.template operator()<0, 1>(FP8_EXTREMUM_FOLD_NONE, output_scale);
                        return;
                    case ReductionStageInputTransform::Square:
                        run.template operator()<0, 2>(FP8_EXTREMUM_FOLD_NONE, output_scale);
                        return;
                }
                break;
            case ReductionStageCombine::Product:
                run.template operator()<1, 0>(FP8_EXTREMUM_FOLD_NONE, output_scale);
                return;
            case ReductionStageCombine::Minimum:
            case ReductionStageCombine::Maximum: {
                // RK's shared MIN kernel: the transform and MAX's sign fold into the bytes, MAX negates the scale.
                float scale = output_scale;
                const int fixed_op = semantics.combine == ReductionStageCombine::Maximum ? 3 : 2;
                Fp8ExtremumFold fold;
                if constexpr (std::is_same_v<InputT, float>) {
                    fold = fp32CompleteExtremumFold(fixed_op, semantics.input_transform, scale);
                } else if constexpr (IS_F16_KR_INPUT<InputT>) {
                    fold = fp16CompleteExtremumFold(fixed_op, semantics.input_transform, scale);
                } else {
                    fold = fp8CompleteExtremumFold(fixed_op, semantics.input_transform, scale);
                }
                run.template operator()<2, 0>(fold, scale);
                return;
            }
        }
        throw std::logic_error("Retained FP8 KR VALUE pass received unsupported semantics.");
    };
    by_semantics();
}

// The ARG launch of one element format (Fp8KRArgFormat, F16KRArgFormat) for a non-vector shape.
template <typename ArgT>
void launchKRArgFamily(const KR::KRSegmentShape& shape, const Fp8KRLaunchGeometry& launch,
                       typename ArgT::Fold fold, const typename ArgT::Outputs& outputs, void* scratch,
                       Stream& stream, KR::KRSplitStages stages) {
    switch (shape.family()) {
        case KR::KRFamily::Pack:
            krArgPackKernel<ArgT><<<launch.blocks, KR::KR_THREADS, 0, stream.getStream()>>>(
                launch.words, launch.head, shape.rows, static_cast<uint32_t>(shape.row_bytes), shape.rows_per_thread,
                fold, outputs);
            break;
        case KR::KRFamily::Group:
            krArgGroupKernel<ArgT><<<launch.blocks, KR::KR_THREADS, 0, stream.getStream()>>>(
                launch.words, launch.head, shape.rows, shape.row_bytes, launch.group_shift, fold, outputs);
            break;
        case KR::KRFamily::Block:
            krArgBlockKernel<ArgT><<<launch.blocks, KR::KR_THREADS, 0, stream.getStream()>>>(
                launch.words, launch.head, shape.rows, shape.row_bytes, fold, outputs);
            break;
        case KR::KRFamily::Split: {
            auto* partials = static_cast<uint64_t*>(scratch);
            krArgSplitKernel<ArgT><<<launch.blocks, KR::KR_THREADS, 0, stream.getStream()>>>(
                launch.words, launch.head, shape.rows, shape.row_bytes, shape.chunk_bytes, shape.chunks(), fold,
                partials);
            CUDA_CHECK(cudaGetLastError());
            if (stages == KR::KRSplitStages::PartialsOnly) break;
            krArgFinishKernel<ArgT><<<fp8KRFinishBlocks(shape.rows), KR::KR_THREADS, 0,
                                   stream.getStream()>>>(partials, shape.rows, shape.chunks(), outputs);
            break;
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace
#endif  // THOR_CUB_ENABLE_FP8_TYPES

bool fp8KRValueSupports(const ReductionStageSemantics& semantics) {
    switch (semantics.combine) {
        case ReductionStageCombine::Sum:
        case ReductionStageCombine::Minimum:
        case ReductionStageCombine::Maximum:
            return semantics.input_transform == ReductionStageInputTransform::Identity ||
                   semantics.input_transform == ReductionStageInputTransform::AbsoluteValue ||
                   semantics.input_transform == ReductionStageInputTransform::Square;
        case ReductionStageCombine::Product:
            return semantics.input_transform == ReductionStageInputTransform::Identity;
    }
    return false;
}

void launchFp8KRValueSegments(const ReductionStageSemantics& semantics,
                              const StridedSpan& input,
                              const StridedSpan& output,
                              const KR::KRSegmentShape& shape,
                              void* scratch,
                              uint64_t scratch_bytes,
                              float output_scale,
                              Stream& stream,
                              KR::KRSplitStages stages) {
#if THOR_CUB_ENABLE_FP8_TYPES
    requireFp8KRInput(input, shape, scratch, scratch_bytes, false);
    if (output.getTotalNumElements() != shape.rows || !fp8CompleteSupportsOutputStorage(output.getDataType()) ||
        !fp8KRValueSupports(semantics)) {
        throw std::logic_error("Retained FP8 KR VALUE pass received an unsupported output or semantics.");
    }
    if (input.getDataType() == DataType::FP8_E4M3) {
        launchKRValueTyped<__nv_fp8_e4m3>(semantics, input, output, shape, scratch, output_scale, stream,
                                          stages);
    } else {
        launchKRValueTyped<__nv_fp8_e5m2>(semantics, input, output, shape, scratch, output_scale, stream,
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

void launchFp32KRValueSegments(const ReductionStageSemantics& semantics,
                               const StridedSpan& input,
                               const StridedSpan& output,
                               const KR::KRSegmentShape& shape,
                               void* scratch,
                               uint64_t scratch_bytes,
                               float output_scale,
                               Stream& stream,
                               KR::KRSplitStages stages) {
#if THOR_CUB_ENABLE_FP8_TYPES
    // The FP8 kernels at one element per word. Rows are whole elements on a 4-byte aligned input (head 0), so no word
    // holds bytes of two rows and the FP32 branches need no masks.
    if (input.getDataType() != DataType::FP32 || !KR::retainedKRShapeLegalForElement(shape, 4) ||
        input.getTotalNumElements() * 4 != shape.rows * shape.row_bytes ||
        (reinterpret_cast<uintptr_t>(input.getMemPtr<void>()) & uintptr_t{3}) != 0) {
        throw std::logic_error("Retained FP32 KR pass received an unsupported input or segment shape.");
    }
    if (shape.family() == KR::KRFamily::Split &&
        (scratch == nullptr || scratch_bytes < KR::krScratchBytes(shape, false) ||
         (reinterpret_cast<uintptr_t>(scratch) & uintptr_t{15}) != 0)) {
        throw std::logic_error("Retained FP32 KR Split pass requires its 16-byte aligned partial scratch.");
    }
    if (output.getTotalNumElements() != shape.rows || !fp8CompleteSupportsOutputStorage(output.getDataType()) ||
        !fp8KRValueSupports(semantics)) {
        throw std::logic_error("Retained FP32 KR VALUE pass received an unsupported output or semantics.");
    }
    launchKRValueTyped<float>(semantics, input, output, shape, scratch, output_scale, stream, stages);
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

void launchF16KRValueSegments(const ReductionStageSemantics& semantics,
                              const StridedSpan& input,
                              const StridedSpan& output,
                              const KR::KRSegmentShape& shape,
                              void* scratch,
                              uint64_t scratch_bytes,
                              float output_scale,
                              Stream& stream,
                              KR::KRSplitStages stages) {
#if THOR_CUB_ENABLE_FP8_TYPES
    // The FP8 kernels at two elements per word. Rows are whole elements on a 2-byte aligned input, so a row may start
    // mid-word (head 2, or an odd element count before it) and the 2-byte branches mask half-words.
    const DataType dtype = input.getDataType();
    if ((dtype != DataType::FP16 && dtype != DataType::BF16) || !KR::retainedKRShapeLegalForElement(shape, 2) ||
        input.getTotalNumElements() * 2 != shape.rows * shape.row_bytes ||
        (reinterpret_cast<uintptr_t>(input.getMemPtr<void>()) & uintptr_t{1}) != 0) {
        throw std::logic_error("Retained FP16 / BF16 KR pass received an unsupported input or segment shape.");
    }
    if (shape.family() == KR::KRFamily::Split &&
        (scratch == nullptr || scratch_bytes < KR::krScratchBytes(shape, false) ||
         (reinterpret_cast<uintptr_t>(scratch) & uintptr_t{15}) != 0)) {
        throw std::logic_error("Retained FP16 / BF16 KR Split pass requires its 16-byte aligned partial scratch.");
    }
    if (output.getTotalNumElements() != shape.rows || !fp8CompleteSupportsOutputStorage(output.getDataType()) ||
        !fp8KRValueSupports(semantics)) {
        throw std::logic_error("Retained FP16 / BF16 KR VALUE pass received an unsupported output or semantics.");
    }
    if (dtype == DataType::FP16) {
        launchKRValueTyped<__half>(semantics, input, output, shape, scratch, output_scale, stream, stages);
    } else {
        launchKRValueTyped<__nv_bfloat16>(semantics, input, output, shape, scratch, output_scale, stream,
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
    throw std::logic_error(
        "Retained FP16 / BF16 KR VALUE requires THOR_CUB_ENABLE_FP8_TYPES (the retained kernels' gate).");
#endif
}

void launchFp8KRArgSegments(const StridedSpan& input,
                            const StridedSpan* value_output,
                            const StridedSpan* index_output,
                            const KR::KRSegmentShape& shape,
                            void* scratch,
                            uint64_t scratch_bytes,
                            bool maximum,
                            uint64_t domain_stride,
                            Stream& stream,
                            KR::KRSplitStages stages) {
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
    if (shape.family() == KR::KRFamily::Pack && shape.vector &&
        (reinterpret_cast<uintptr_t>(input.getMemPtr<void>()) & uintptr_t{15}) == 0) {
        const bool launched = dispatchFp8KRVectorRowBytes(shape.row_bytes, [&]<int RowBytes>() {
            fp8KRArgVectorKernel<RowBytes><<<launch.blocks, KR::KR_THREADS, 0, stream.getStream()>>>(
                static_cast<const uint4*>(input.getMemPtr<void>()), shape.rows, fold, arg_outputs);
        });
        if (!launched) throw std::logic_error("Retained FP8 KR vector Pack has no kernel for this row length.");
        CUDA_CHECK(cudaGetLastError());
    } else {
        launchKRArgFamily<Fp8KRArgFormat>(shape, launch, fold, arg_outputs, scratch, stream, stages);
    }
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

void launchF16KRArgSegments(const StridedSpan& input,
                            const StridedSpan* value_output,
                            const StridedSpan* index_output,
                            const KR::KRSegmentShape& shape,
                            void* scratch,
                            uint64_t scratch_bytes,
                            bool maximum,
                            uint64_t domain_stride,
                            Stream& stream,
                            KR::KRSplitStages stages) {
#if THOR_CUB_ENABLE_FP8_TYPES
    // FP8's ARG kernels at two elements per word (F16KRArgFormat), on a 2-byte aligned input.
    const DataType dtype = input.getDataType();
    if ((dtype != DataType::FP16 && dtype != DataType::BF16) || !KR::f16KRArgShapeLegal(shape) ||
        input.getTotalNumElements() * 2 != shape.rows * shape.row_bytes ||
        (reinterpret_cast<uintptr_t>(input.getMemPtr<void>()) & uintptr_t{1}) != 0) {
        throw std::logic_error("Retained FP16 / BF16 KR ARG pass received an unsupported input or segment shape.");
    }
    if (shape.family() == KR::KRFamily::Split &&
        (scratch == nullptr || scratch_bytes < KR::krScratchBytes(shape, true) ||
         (reinterpret_cast<uintptr_t>(scratch) & uintptr_t{15}) != 0)) {
        throw std::logic_error("Retained FP16 / BF16 KR ARG Split pass requires its 16-byte aligned partial scratch.");
    }
    const uint64_t outputs = shape.rows;
    const uint64_t last_index = shape.row_bytes / 2 - 1;
    if ((value_output == nullptr && index_output == nullptr) ||
        (value_output != nullptr && (value_output->getTotalNumElements() != outputs ||
                                     !fp8CompleteSupportsOutputStorage(value_output->getDataType()))) ||
        (index_output != nullptr && (index_output->getTotalNumElements() != outputs ||
                                     (index_output->getDataType() != DataType::UINT32 &&
                                      index_output->getDataType() != DataType::UINT64)))) {
        throw std::logic_error("Retained FP16 / BF16 KR ARG pass received unsupported outputs.");
    }
    if (index_output != nullptr && index_output->getDataType() == DataType::UINT32 && domain_stride != 0 &&
        last_index > std::numeric_limits<uint32_t>::max() / domain_stride) {
        throw std::logic_error(
            "Retained FP16 / BF16 KR ARG UINT32 index output cannot hold the original-domain index.");
    }
    const F16ArgFold fold = f16ArgFold(maximum, dtype == DataType::BF16);
    const F16ArgOutputs arg_outputs{value_output == nullptr ? nullptr : value_output->getMemPtr<void>(),
                                    value_output == nullptr ? DataType::FP32 : value_output->getDataType(),
                                    index_output == nullptr ? nullptr : index_output->getMemPtr<void>(),
                                    index_output == nullptr ? DataType::UINT32 : index_output->getDataType(),
                                    domain_stride,
                                    fold,
                                    dtype == DataType::BF16};
    launchKRArgFamily<F16KRArgFormat>(shape, fp8KRLaunchGeometry(input, shape), fold, arg_outputs, scratch,
                                      stream, stages);
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
    throw std::logic_error(
        "Retained FP16 / BF16 KR ARG requires THOR_CUB_ENABLE_FP8_TYPES (the retained kernels' gate).");
#endif
}

void launchFp32KRArgSegments(const StridedSpan& input,
                             const StridedSpan* value_output,
                             const StridedSpan* index_output,
                             const KR::KRSegmentShape& shape,
                             void* scratch,
                             uint64_t scratch_bytes,
                             bool maximum,
                             uint64_t domain_stride,
                             Stream& stream,
                             KR::KRSplitStages stages) {
#if THOR_CUB_ENABLE_FP8_TYPES
    // Whole-element rows on a 4-byte aligned input, as the FP32 VALUE launcher.
    if (input.getDataType() != DataType::FP32 || !KR::fp32KRArgShapeLegal(shape) ||
        input.getTotalNumElements() * 4 != shape.rows * shape.row_bytes ||
        (reinterpret_cast<uintptr_t>(input.getMemPtr<void>()) & uintptr_t{3}) != 0) {
        throw std::logic_error("Retained FP32 KR ARG pass received an unsupported input or segment shape.");
    }
    if (shape.family() == KR::KRFamily::Split &&
        (scratch == nullptr || scratch_bytes < KR::krScratchBytes(shape, true) ||
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
        case KR::KRFamily::Pack:
            fp32KRArgPackKernel<<<launch.blocks, KR::KR_THREADS, 0, stream.getStream()>>>(
                launch.words, shape.rows, static_cast<uint32_t>(row_words), shape.rows_per_thread, arg_outputs);
            break;
        case KR::KRFamily::Group:
            fp32KRArgGroupKernel<<<launch.blocks, KR::KR_THREADS, 0, stream.getStream()>>>(
                launch.words, shape.rows, row_words, launch.group_shift, arg_outputs);
            break;
        case KR::KRFamily::Block:
            fp32KRArgBlockKernel<<<launch.blocks, KR::KR_THREADS, 0, stream.getStream()>>>(
                launch.words, shape.rows, row_words, arg_outputs);
            break;
        case KR::KRFamily::Split: {
            auto* partials = static_cast<uint64_t*>(scratch);
            const uint64_t chunk_words = shape.chunk_bytes / 4;
            fp32KRArgSplitKernel<<<launch.blocks, KR::KR_THREADS, 0, stream.getStream()>>>(
                launch.words, shape.rows, row_words, chunk_words, shape.chunks(), arg_outputs.xor_mask, partials);
            CUDA_CHECK(cudaGetLastError());
            if (stages == KR::KRSplitStages::PartialsOnly) break;
            fp32KRArgFinishKernel<<<fp8KRFinishBlocks(shape.rows), KR::KR_THREADS, 0, stream.getStream()>>>(
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

uint64_t fp32KRCarriedArgScratchBytes(const KR::KRSegmentShape& shape) {
    return shape.family() == KR::KRFamily::Split ? fp32ArgWideScratchBytes(shape.rows, shape.chunks(), 1) : 0;
}

void launchFp32KRCarriedArgSegments(const StridedSpan& input,
                                    const StridedSpan& carried_index,
                                    const StridedSpan* value_output,
                                    const StridedSpan* index_output,
                                    const KR::KRSegmentShape& shape,
                                    void* scratch,
                                    uint64_t scratch_bytes,
                                    bool maximum,
                                    uint64_t row_stride,
                                    Stream& stream,
                                    KR::KRSplitStages stages) {
#if THOR_CUB_ENABLE_FP8_TYPES
    const uint64_t elements = shape.rows * (shape.row_bytes / 4);
    if (input.getDataType() != DataType::FP32 || !KR::fp32KRArgShapeLegal(shape) ||
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
    if (shape.family() == KR::KRFamily::Split &&
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
            case KR::KRFamily::Pack:
                krCarriedArgPackKernel<IndexT><<<launch.blocks, KR::KR_THREADS, 0, stream.getStream()>>>(
                    launch.words, carried, shape.rows, static_cast<uint32_t>(row_words), shape.rows_per_thread,
                    row_stride, arg_outputs);
                break;
            case KR::KRFamily::Group:
                krCarriedArgGroupKernel<IndexT><<<launch.blocks, KR::KR_THREADS, 0, stream.getStream()>>>(
                    launch.words, carried, shape.rows, row_words, launch.group_shift, row_stride, arg_outputs);
                break;
            case KR::KRFamily::Block:
                krCarriedArgBlockKernel<IndexT><<<launch.blocks, KR::KR_THREADS, 0, stream.getStream()>>>(
                    launch.words, carried, shape.rows, row_words, row_stride, arg_outputs);
                break;
            case KR::KRFamily::Split: {
                const uint64_t chunks = shape.chunks();
                auto* bytes = static_cast<uint8_t*>(scratch);
                const uint64_t tail_offset = (shape.segments() * sizeof(uint32_t) + 255) & ~uint64_t{255};
                const Fp32ArgWidePartials partials{reinterpret_cast<uint32_t*>(bytes),
                                                   reinterpret_cast<uint64_t*>(bytes + tail_offset)};
                krCarriedArgSplitKernel<IndexT><<<launch.blocks, KR::KR_THREADS, 0, stream.getStream()>>>(
                    launch.words, carried, shape.rows, row_words, shape.chunk_bytes / 4, chunks, arg_outputs.xor_mask,
                    row_stride, partials);
                CUDA_CHECK(cudaGetLastError());
                if (stages == KR::KRSplitStages::PartialsOnly) break;
                krCarriedArgFinishKernel<<<fp8KRFinishBlocks(shape.rows), KR::KR_THREADS, 0,
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

}  // namespace ThorImplementation::ReductionInternal
