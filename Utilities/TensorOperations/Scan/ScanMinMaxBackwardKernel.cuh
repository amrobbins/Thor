#pragma once

// The backward of a min or max scan in one kernel (ScanMinMaxBackward.h). It reads the scan's input x and the
// gradient dy of the scan's output and writes dx, the gradient of x: each element's dy goes to the winner of its
// prefix (the element its output was taken from), so dx[w] is the sum of dy over the run of elements whose winner is
// w, and 0 for an element that never wins. Design and host model: plans/Thor_Scan_MinMax_Backward_Plan_2026-11-20.md.
//
// It is the scan kernel's arg scan (SegmentedScanKernel.cuh) with a different last step, so its steps 1 to 4 are that
// kernel's, on the same tiles, slots and published runs, and it is bitwise reproducible for the same reason:
//   1-4. as the arg scan: each chunk's key run, the slots' runs scanned in slot order, the tile's run published and
//        the tile's prefix (the row's winner so far) folded from the tiles before it. dy is loaded and stashed with x;
//   5a.  each thread scans its chunks again with the prefix: the winners (a row's first element in scan order, or an
//        element whose key beats every key before it in its row) and the chunk's leading sum, the dy that belongs to
//        the winner arriving at the chunk;
//   5b.  the chunks' leading sums are scanned in shared memory from the tile's last slot to its first (the same fixed
//        tree as step 2, mirrored), so each chunk knows the sum that follows it up to the next winner; the tile
//        publishes the sum from its start to its first winner (its whole sum when it has none);
//   5c.  each thread walks its chunks from their last element to their first, adds up each winner's run and writes
//        the chunk: the run's sum at a winner, 0 elsewhere;
//   6.   a run that goes on past the tile's end cannot be finished by its winner's tile: the tiles after it have not
//        run (a tile that waited for later blocks could wait for blocks that never start). So that tile leaves the
//        one element unwritten and publishes the run's sum so far, its tail, and the tile where the run ends writes
//        the element: the tail, then the whole tiles between (tiles, groups, supergroups, groups, tiles: each level
//        folded left to right by the last tile of a group or supergroup), then its own leading sum. Every wait is
//        for a tile before it, and every element has one writer.
// An exclusive scan differs only in where a winner's own dy goes: to the winner before it (a row's first element:
// nowhere). A reverse scan takes the tiles and elements in scan order, as the scan kernel does. Max or min, the mode
// and the direction are launch arguments, not template parameters: a key flip, one select per element, and for the
// direction the tile order, where a row starts, and each chunk mirrored in registers as it is unpacked (and its
// outputs mirrored back), so every loop between runs over the chunk in scan order whichever way that is in memory.
//
// The formats of x, dy and dx are launch arguments too (FloatFormat: FP32, FP16, BF16, E4M3, E5M2, each on its own).
// Only the loads and the stores know them: x and dy become FP32 as they are loaded, which is exact for every format
// and keeps order, ties, NaN and the sign of zero, so the winners and the sums are those of FP32 tensors holding the
// same values; dx is converted from the FP32 sum as it is stored (to nearest; FP8 saturating to its finite range, as
// the sum scans write FP8). Everything between is one FP32 kernel, and no tensor is cast before or after it.

#include "Utilities/TensorOperations/Scan/SegmentedScanKernel.cuh"
#include "Utilities/TensorOperations/Scan/ScanMinMaxBackwardLaunch.h"

namespace ThorImplementation::ScanMinMaxBackwardKernels {

using namespace SegmentedScanKernels;

// An element's key: the forward arg scan's for a max (keyOf: the value's order above 2^31 - 1 - index, so the first
// of equal values wins), with the value's order flipped for a min (`flip`: the order bits, or 0) and the index bits
// flipped for a reverse scan (`place_flip`: bits 31 to 1, or 0), which leaves the index itself there, so the last of
// equal values wins: keyOf's reverse key. A NaN keeps order 0, below every number either way; no number has order 0.
__host__ __device__ inline uint64_t winnerKey(float value, uint64_t index, uint64_t flip, uint64_t place_flip) {
    const uint64_t key = keyOf<float, SegmentedScanOp::ArgMax, SegmentedScanDirection::Forward>(value, index) ^ place_flip;
    return (key >> 32) == 0U ? key : key ^ flip;
}

// The flat index a key was made from.
__host__ __device__ inline uint64_t winnerIndex(uint64_t key, bool reverse) {
    const uint32_t place = static_cast<uint32_t>(key) >> 1;
    return reverse ? place : 0x7FFFFFFFU - place;
}

// One chunk's head bits in memory order (bit j: element j starts a run of the scan: a row's first element in scan
// order). Dense rows take them from the chunk's first column, `col` (the scan kernel's denseHeads, with the direction
// at run time); ragged rows read the tile's bitmap at the chunk's place in the tile, `local`. The first element of all
// (forward) and the last before `end` (reverse) always start one.
template <bool RAGGED, uint32_t E>
__device__ inline uint32_t chunkHeads(bool reverse,
                                      uint64_t chunk_first,
                                      uint64_t end,
                                      uint64_t local,
                                      const uint32_t* s_heads,
                                      uint64_t cols,
                                      uint64_t col,
                                      uint32_t pattern) {
    uint32_t heads = 0;
    if constexpr (RAGGED) {
#pragma unroll
        for (uint32_t j = 0; j < E; ++j) {
            const uint64_t k = local + j;
            heads |= ((s_heads[k >> 5] >> (k & 31U)) & 1U) << j;
        }
    } else {
        const uint64_t first = reverse ? cols - col - 1 : (col == 0 ? 0 : cols - col);
        const uint32_t mask = ~0U >> (32 - E);
        heads = first < E ? (pattern << static_cast<uint32_t>(first)) & mask : 0U;
    }
    if (reverse) {
        heads |= endHead<E>(chunk_first, end);
    } else if (chunk_first == 0) {
        heads |= 1U;
    }
    return heads;
}

// A chunk's eight head bits from memory order to scan order (and back: it is its own inverse).
__host__ __device__ inline uint32_t mirrorBits(uint32_t bits, bool reverse) {
    if (!reverse) return bits;
    bits = ((bits & 0x0FU) << 4) | ((bits >> 4) & 0x0FU);
    bits = ((bits & 0x33U) << 2) | ((bits >> 2) & 0x33U);
    return ((bits & 0x55U) << 1) | ((bits >> 1) & 0x55U);
}

// A chunk's eight values from memory order (two 16-byte words) to scan order.
__device__ inline void unpackInScanOrder(const uint4* raw, bool reverse, float* out) {
    constexpr uint32_t E = kGradientItems;
    float value[E];
    memcpy(value, raw, sizeof(value));
#pragma unroll
    for (uint32_t jo = 0; jo < E; ++jo) out[jo] = reverse ? value[E - 1 - jo] : value[jo];
}

// ---- Formats: an element's bits as FP32 (exact), and FP32 as an element's bits (to nearest) ----

__host__ __device__ inline float halfBitsToFloat(uint32_t bits) {
    return __half2float(__ushort_as_half(static_cast<unsigned short>(bits)));
}

// BF16 is FP32's upper half.
__host__ __device__ inline float bf16BitsToFloat(uint32_t bits) { return cuda::std::bit_cast<float>(bits << 16); }

// E5M2 is FP16's upper byte: the same sign, exponent and bias, and the two highest fraction bits.
__host__ __device__ inline float e5m2BitsToFloat(uint32_t byte) { return halfBitsToFloat(byte << 8); }

// E4M3: sign, four exponent bits (bias 7), three fraction bits; no infinity, and S.1111.111 is its NaN. In units of
// 2^-9 every magnitude is an integer below 2^18: the fraction alone for a subnormal, 1.fraction shifted otherwise.
__host__ __device__ inline float e4m3BitsToFloat(uint32_t byte) {
    const uint32_t exponent = (byte >> 3) & 0xFU;
    const uint32_t fraction = byte & 0x7U;
    const uint32_t units = exponent == 0U ? fraction : (0x8U | fraction) << (exponent - 1U);
    uint32_t bits = cuda::std::bit_cast<uint32_t>(static_cast<float>(units) * 0x1p-9f);
    if ((byte & 0x7FU) == 0x7FU) bits = 0x7FC00000U;
    return cuda::std::bit_cast<float>(bits | ((byte & 0x80U) << 24));
}

__host__ __device__ inline uint32_t halfBitsOf(float value) {
    return __half_as_ushort(__float2half_rn(value));
}

__host__ __device__ inline uint32_t bf16BitsOf(float value) {
    return __bfloat16_as_ushort(__float2bfloat16_rn(value));
}

// One element of a tensor in `format`, as FP32.
__device__ inline float loadElement(const void* base, uint64_t index, FloatFormat format) {
    if (format == FloatFormat::Fp32) return static_cast<const float*>(base)[index];
    if (format == FloatFormat::Fp16) return halfBitsToFloat(static_cast<const uint16_t*>(base)[index]);
    if (format == FloatFormat::Bf16) return bf16BitsToFloat(static_cast<const uint16_t*>(base)[index]);
    const uint32_t byte = static_cast<const uint8_t*>(base)[index];
    return format == FloatFormat::E5m2 ? e5m2BitsToFloat(byte) : e4m3BitsToFloat(byte);
}

__device__ inline void storeElement(void* base, uint64_t index, float value, FloatFormat format) {
    if (format == FloatFormat::Fp32) {
        static_cast<float*>(base)[index] = value;
    } else if (format == FloatFormat::Fp16) {
        static_cast<uint16_t*>(base)[index] = static_cast<uint16_t>(halfBitsOf(value));
    } else if (format == FloatFormat::Bf16) {
        static_cast<uint16_t*>(base)[index] = static_cast<uint16_t>(bf16BitsOf(value));
    } else if (format == FloatFormat::E5m2) {
        static_cast<uint8_t*>(base)[index] = __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E5M2);
    } else {
        static_cast<uint8_t*>(base)[index] = __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
    }
}

// A chunk's eight elements as they are in memory: 32 bytes of FP32 (a and b), 16 of a 16-bit format (a), or 8 of an
// 8-bit one (a.x and a.y). The element at `first` is aligned to its chunk: every tile starts at a multiple of eight
// elements, and a tensor at 256 bytes. So an FP32 chunk is 32-byte aligned, and saying so lets nvcc make its two
// 16-byte words one 256-bit access where the GPU has one (sm_100 and up, sm_120 included: LDG.E.ENL2.256 /
// STG.E.ENL2.256); below sm_100 they stay two 16-byte accesses.
struct RawChunk {
    uint4 a;
    uint4 b;
};

__device__ inline RawChunk loadRawChunk(const void* base, uint64_t first, uint32_t bytes) {
    RawChunk raw{};
    const char* at = static_cast<const char*>(base) + first * bytes;
    if (bytes == 4) {
        const uint4* const words = static_cast<const uint4*>(__builtin_assume_aligned(at, 32));
        raw.a = words[0];
        raw.b = words[1];
    } else if (bytes == 2) {
        raw.a = *reinterpret_cast<const uint4*>(at);
    } else {
        const uint2 word = *reinterpret_cast<const uint2*>(at);
        raw.a.x = word.x;
        raw.a.y = word.y;
    }
    return raw;
}

// The chunk as eight FP32 values, in two 16-byte words. FP32 passes through.
__device__ inline void chunkToFloat(const RawChunk& raw, FloatFormat format, uint4* out) {
    constexpr uint32_t E = kGradientItems;
    if (format == FloatFormat::Fp32) {
        out[0] = raw.a;
        out[1] = raw.b;
        return;
    }
    const uint32_t word[4] = {raw.a.x, raw.a.y, raw.a.z, raw.a.w};
    float value[E];
    if (format == FloatFormat::Fp16) {
#pragma unroll
        for (uint32_t j = 0; j < E; ++j) value[j] = halfBitsToFloat((word[j >> 1] >> (16U * (j & 1U))) & 0xFFFFU);
    } else if (format == FloatFormat::Bf16) {
#pragma unroll
        for (uint32_t j = 0; j < E; ++j) value[j] = bf16BitsToFloat((word[j >> 1] >> (16U * (j & 1U))) & 0xFFFFU);
    } else if (format == FloatFormat::E5m2) {
#pragma unroll
        for (uint32_t j = 0; j < E; ++j) value[j] = e5m2BitsToFloat((word[j >> 2] >> (8U * (j & 3U))) & 0xFFU);
    } else {
#pragma unroll
        for (uint32_t j = 0; j < E; ++j) value[j] = e4m3BitsToFloat((word[j >> 2] >> (8U * (j & 3U))) & 0xFFU);
    }
    memcpy(out, value, sizeof(value));
}

// Eight FP32 values as a chunk of `format`, from the first byte of the two 16-byte words.
__device__ inline void chunkFromFloat(const float* value, FloatFormat format, uint4* out) {
    constexpr uint32_t E = kGradientItems;
    if (format == FloatFormat::Fp32) {
        memcpy(out, value, E * sizeof(float));
        return;
    }
    uint32_t word[4] = {0U, 0U, 0U, 0U};
    if (format == FloatFormat::Fp16) {
#pragma unroll
        for (uint32_t j = 0; j < E; ++j) word[j >> 1] |= halfBitsOf(value[j]) << (16U * (j & 1U));
    } else if (format == FloatFormat::Bf16) {
#pragma unroll
        for (uint32_t j = 0; j < E; ++j) word[j >> 1] |= bf16BitsOf(value[j]) << (16U * (j & 1U));
    } else if (format == FloatFormat::E5m2) {
#pragma unroll
        for (uint32_t j = 0; j < E; ++j) {
            word[j >> 2] |= static_cast<uint32_t>(__nv_cvt_float_to_fp8(value[j], __NV_SATFINITE, __NV_E5M2)) << (8U * (j & 3U));
        }
    } else {
#pragma unroll
        for (uint32_t j = 0; j < E; ++j) {
            word[j >> 2] |= static_cast<uint32_t>(__nv_cvt_float_to_fp8(value[j], __NV_SATFINITE, __NV_E4M3)) << (8U * (j & 3U));
        }
    }
    out[0] = make_uint4(word[0], word[1], word[2], word[3]);
    out[1] = make_uint4(0U, 0U, 0U, 0U);
}

// Step 2 of the scan kernel, on any runs: the slots' runs scanned in slot order. Thread t scans the CHUNKS adjacent
// slots from CHUNKS x t in registers, each warp scans its threads' totals in place, and warp 0 scans the warps'
// totals. Called by every thread, after a __syncthreads; ends with one.
template <typename A, SegmentedScanOp OP, uint32_t THREADS, uint32_t CHUNKS>
__device__ inline void scanSlotRuns(A* s_value, uint8_t* s_head, A* s_warp_value, uint32_t* s_warp_head) {
    constexpr uint32_t kWarps = THREADS / 32;
    const uint32_t t = threadIdx.x;
    const uint32_t lane = t & 31U;
    const uint32_t warp = t >> 5;
    const uint32_t own = CHUNKS * t;
    const uint32_t own_last = own + CHUNKS - 1;
    Run<A> mine{s_value[own], s_head[own], 1U};
#pragma unroll
    for (uint32_t k = 1; k < CHUNKS; ++k) {
        mine = combine<A, OP>(mine, Run<A>{s_value[own + k], s_head[own + k], 1U});
        s_value[own + k] = mine.value;
        s_head[own + k] = mine.head;
    }
    __syncwarp();
#pragma unroll
    for (uint32_t offset = 1; offset < 32; offset <<= 1) {
        Run<A> before{A{}, 0U, 0U};
        if (lane >= offset) {
            before = Run<A>{s_value[own_last - CHUNKS * offset], s_head[own_last - CHUNKS * offset], 1U};
        }
        __syncwarp();
        mine = combine<A, OP>(before, mine);
        s_value[own_last] = mine.value;
        s_head[own_last] = mine.head;
        __syncwarp();
    }
    if (lane == 31) {
        s_warp_value[warp] = mine.value;
        s_warp_head[warp] = mine.head;
    }
    __syncthreads();
    if (warp == 0) {
        Run<A> w{A{}, 0U, 0U};
        if (lane < kWarps) w = Run<A>{s_warp_value[lane], s_warp_head[lane], 1U};
#pragma unroll
        for (uint32_t offset = 1; offset < kWarps; offset <<= 1) {
            Run<A> before{A{}, 0U, 0U};
            if (lane < kWarps && lane >= offset) {
                before = Run<A>{s_warp_value[lane - offset], s_warp_head[lane - offset], 1U};
            }
            __syncwarp();
            w = combine<A, OP>(before, w);
            if (lane < kWarps) {
                s_warp_value[lane] = w.value;
                s_warp_head[lane] = w.head;
            }
            __syncwarp();
        }
    }
    __syncthreads();
}

// The run of the slots before slot `order` within the tile, from what scanSlotRuns left: the warps before its
// owner's, then that warp's threads before its owner, then the owner's slots before it.
template <typename A, SegmentedScanOp OP, uint32_t CHUNKS>
__device__ inline Run<A> runBefore(uint32_t order,
                                   const A* s_value,
                                   const uint8_t* s_head,
                                   const A* s_warp_value,
                                   const uint32_t* s_warp_head) {
    const uint32_t owner = order / CHUNKS;
    const uint32_t owner_warp = owner >> 5;
    Run<A> before{A{}, 0U, 0U};
    if (owner_warp > 0) before = Run<A>{s_warp_value[owner_warp - 1], s_warp_head[owner_warp - 1], 1U};
    if ((owner & 31U) > 0) {
        const uint32_t last = CHUNKS * owner - 1;
        before = combine<A, OP>(before, Run<A>{s_value[last], s_head[last], 1U});
    }
    if (order > CHUNKS * owner) {
        before = combine<A, OP>(before, Run<A>{s_value[order - 1], s_head[order - 1], 1U});
    }
    return before;
}

__device__ inline void publishSum(unsigned long long* slot, float sum, uint32_t head, uint32_t epoch) {
    publish(slot, packRun(cuda::std::bit_cast<uint32_t>(sum), head, epoch));
}

__device__ inline float awaitSum(unsigned long long* slot, uint32_t epoch) {
    return cuda::std::bit_cast<float>(static_cast<uint32_t>(await(slot, epoch, peek(slot))));
}

// Called by a whole warp: every lane leaves its value in s_fold, and lane 0 returns acc plus the values of lanes
// [first, last) in lane order (the other lanes return acc).
__device__ inline float foldLanes(float value, uint32_t lane, float* s_fold, uint32_t first, uint32_t last, float acc) {
    s_fold[lane] = value;
    __syncwarp();
    if (lane == 0) {
        for (uint32_t i = first; i < last; ++i) acc += s_fold[i];
    }
    __syncwarp();
    return acc;
}

// OFF: void for dense rows of columns.count elements, the offset type for ragged rows. input, grad and output hold n
// elements each, in x_format, dy_format and dx_format. is_min: a min scan's backward. exclusive: the scan was
// exclusive. reverse: it ran from each row's last element to its first.
template <typename OFF,
          uint32_t THREADS = kThreads,
          uint32_t GROUP = kGroup,
          uint32_t SUPER = kSuper,
          uint32_t CHUNKS = kGradientChunks>
__global__ void __launch_bounds__(THREADS, kGradientMinBlocks)
    scanMinMaxBackwardKernel(const void* __restrict__ input,
                             const void* __restrict__ grad,
                             void* __restrict__ output,
                             FloatFormat x_format,
                             FloatFormat dy_format,
                             FloatFormat dx_format,
                             uint64_t n,
                             Columns columns,
                             const OFF* __restrict__ offsets,
                             uint32_t rows,
                             Workspace ws,
                             GradientWorkspace gs,
                             bool is_min,
                             bool exclusive,
                             bool reverse) {
    using T = float;                                          // x and dy as the kernel holds them
    using A = uint64_t;                                       // a key
    constexpr SegmentedScanOp KEY = SegmentedScanOp::ArgMax;  // keys combine by one unsigned max
    constexpr SegmentedScanOp SUM = SegmentedScanOp::Sum;
    constexpr uint32_t W = 2;  // workspace words per key run
    constexpr bool RAGGED = !std::is_void_v<OFF>;
    const uint64_t cols = columns.count;
    constexpr uint32_t E = kGradientItems;
    constexpr uint64_t TILE = kTile<THREADS, CHUNKS, E>;
    using Raw = RawWord<E * sizeof(T)>;
    constexpr uint32_t kRaw = E * sizeof(T) / sizeof(Raw);          // raw words of x per chunk
    constexpr uint32_t kGradRaw = E * sizeof(float) / sizeof(uint4);  // raw words of dy per chunk
    constexpr uint32_t kWarps = THREADS / 32;
    constexpr uint32_t kSlots = THREADS * CHUNKS;
    constexpr uint32_t kPerWord = sizeof(uint4) / sizeof(float);  // outputs per store
    constexpr uint32_t kWords = E / kPerWord;
    static_assert(THREADS % 32 == 0 && kWarps <= 32, "whole warps; one lane of warp 0 per warp total");
    static_assert(GROUP <= 32 && SUPER <= 32, "one lane of warp 0 per tile of a group and per group of a supergroup");
    static_assert(E == 8 && kRaw == 2 && kGradRaw == 2 && kWords == 2, "a chunk is 32 bytes of FP32: two 16-byte words");

    __shared__ Raw s_data[kRaw * kSlots];        // x as FP32: slot p at kRaw * p
    __shared__ uint4 s_grad[kGradRaw * kSlots];  // dy as FP32
    __shared__ uint32_t s_heads[RAGGED ? TILE / 32 : 1];
    __shared__ A s_value[kSlots];  // the slots' key runs, in scan order
    __shared__ uint8_t s_head[kSlots];
    __shared__ A s_warp_value[kWarps];
    __shared__ uint32_t s_warp_head[kWarps];
    __shared__ float s_sum[kSlots];  // the slots' leading sums, from the tile's last slot in scan order to its first
    __shared__ uint8_t s_stop[kSlots];
    __shared__ float s_warp_sum[kWarps];
    __shared__ uint32_t s_warp_stop[kWarps];
    __shared__ uint8_t s_winners[kSlots];   // bit j: the slot's chunk's element j in scan order is a winner
    __shared__ uint8_t s_rowheads[kSlots];  // bit j: it starts a run of the scan (a row's first element in scan order)
    __shared__ uint64_t s_band_cols;
    __shared__ uint32_t s_head_pattern;
    __shared__ A s_fold_value[32];
    __shared__ uint32_t s_fold_head[32];
    __shared__ uint32_t s_fold_valid[32];
    __shared__ float s_fold_sum[32];
    __shared__ float s_group_tiles[32];  // the sums of the group's tiles before this one
    __shared__ uint32_t s_starts;        // the tile's first element in scan order starts a row
    __shared__ uint32_t s_after;         // ragged, forward: a row starts right after the tile
    __shared__ uint32_t s_closed;        // no run goes on past the tile's last element in scan order
    __shared__ float s_tail;             // the sum of the run that does
    __shared__ Run<A> s_prefix;
    __shared__ uint32_t s_epoch;
    __shared__ uint64_t s_end;

    const uint32_t t = threadIdx.x;
    const uint32_t lane = t & 31U;
    const uint32_t warp = t >> 5;
    const uint64_t flip = is_min ? 0xFFFFFFFF00000000ULL : 0ULL;
    const uint64_t place_flip = reverse ? 0xFFFFFFFEULL : 0ULL;
    if constexpr (!RAGGED) {
        if (t == 0) {
            uint32_t pattern = 1U;
            for (uint64_t k = cols; k != 0 && k < 32; k += cols) pattern |= 1U << static_cast<uint32_t>(k);
            s_head_pattern = pattern;
            if constexpr (CHUNKS > 1) s_band_cols = columns.mod(static_cast<uint64_t>(THREADS) * E);
        }
        __syncthreads();
    }
    const uint64_t tile = blockIdx.x;  // the tile's place in scan order, and so in the lookback
    const uint64_t tile_begin = (reverse ? ws.num_tiles - 1 - tile : tile) * TILE;  // where it is in memory
    const uint64_t thread_first = tile_begin + static_cast<uint64_t>(t) * E;
    constexpr uint64_t kBand = static_cast<uint64_t>(THREADS) * E;

    // 1. As the scan kernel: mark the tile's row heads (ragged), load and stash the thread's chunks of x, and of dy,
    // and fold each chunk's keys. `end` is where the scan ends.
    uint64_t end = n;
    bool row_after = false;  // thread 0: the next element in scan order after the tile starts a run of the scan
    if constexpr (RAGGED) {
        const uint64_t rows_end = offsets[rows];
        if (rows_end < end) end = rows_end;
        if (tile_begin >= end) {
            // A tile past the rows: it writes every word it owns, empty and with this launch's epoch (the scan kernel).
            if (t == 0) {
                const Place place = placeOf<GROUP, SUPER>(tile, ws.num_tiles);
                const uint32_t epoch = nextEpoch(static_cast<uint32_t>(peek(ws.tile_runs + W * tile) >> 33));
                const unsigned long long empty = packRun(0U, 0U, epoch);
                publish(ws.tile_runs + W * tile, empty);
                publish(ws.tile_runs + W * tile + 1, empty);
                publish(gs.tile_sums + tile, empty);
                publish(gs.tile_tails + tile, empty);
                if (place.group_duty) {
                    publish(ws.group_runs + W * place.group, empty);
                    publish(ws.group_runs + W * place.group + 1, empty);
                    publish(gs.group_sums + place.group, empty);
                }
                if (place.super_duty) {
                    publish(ws.super_runs + W * place.super, empty);
                    publish(ws.super_runs + W * place.super + 1, empty);
                    publish(gs.super_sums + place.super, empty);
                }
            }
            return;
        }
        if (t == 0) {
            s_end = end;
            s_after = 0U;
        }
        for (uint32_t i = t; i < TILE / 32; i += THREADS) s_heads[i] = 0;
        const uint64_t first_start = reverse ? tile_begin + 1 : tile_begin;
        const uint32_t r0 = lowerBoundOffsets<THREADS>(offsets, rows, first_start);  // syncs after the clear
        for (uint32_t r = r0 + t; r <= rows; r += THREADS) {
            const uint64_t start = offsets[r];
            if (start >= first_start + TILE) {
                // Forward: a row that starts right after the tile ends every run at the tile's end.
                if (!reverse && start == first_start + TILE) atomicOr(&s_after, 1U);
                break;
            }
            if (start < end) {
                const uint64_t at = start - first_start;  // reverse: start - 1 - tile_begin
                atomicOr(&s_heads[at >> 5], 1U << (at & 31U));
            }
        }
        __syncthreads();
        if (t == 0) {
            if (reverse) {
                // The element before the tile in memory comes next in scan order: it starts a run when the tile's
                // first element in memory starts a row.
                row_after = r0 > 0 && static_cast<uint64_t>(offsets[r0 - 1]) == tile_begin;
            } else {
                row_after = s_after != 0U;
            }
        }
    }
    // No run goes on past the tile when no element follows it in scan order, or the next one starts a run.
    const bool no_next = reverse ? tile_begin == 0 : tile_begin + TILE >= end;
    const uint64_t col_first = RAGGED ? 0 : columns.mod(thread_first);
    uint64_t band_cols = 0;
    uint32_t pattern = 0;
    if constexpr (!RAGGED) pattern = s_head_pattern;
    if constexpr (!RAGGED && CHUNKS > 1) band_cols = s_band_cols;

    // Every band's loads of x and of dy are issued, in their own formats, before any is used; then each chunk is
    // stashed as FP32. A partial chunk (only the tile holding the end has one) goes element by element, with zeros
    // past the end: zero dy adds nothing to any sum, and what the zero x win or lose reaches only elements that are
    // never stored (a reverse scan meets them first, and the last element after them starts a run).
    const uint32_t x_bytes = formatBytes(x_format);
    const uint32_t dy_bytes = formatBytes(dy_format);
    RawChunk loaded[CHUNKS];
    RawChunk loaded_grad[CHUNKS];
    bool whole[CHUNKS];
#pragma unroll
    for (uint32_t s = 0; s < CHUNKS; ++s) {
        const uint64_t chunk_first = thread_first + s * kBand;
        whole[s] = chunk_first + E <= end;
        if (whole[s]) {
            loaded[s] = loadRawChunk(input, chunk_first, x_bytes);
            loaded_grad[s] = loadRawChunk(grad, chunk_first, dy_bytes);
        }
    }
#pragma unroll
    for (uint32_t s = 0; s < CHUNKS; ++s) {
        const uint32_t slot = s * THREADS + t;
        if (whole[s]) {
            uint4 as_float[kRaw];
            chunkToFloat(loaded[s], x_format, as_float);
            s_data[kRaw * slot] = as_float[0];
            s_data[kRaw * slot + 1] = as_float[1];
            chunkToFloat(loaded_grad[s], dy_format, as_float);
            s_grad[kGradRaw * slot] = as_float[0];
            s_grad[kGradRaw * slot + 1] = as_float[1];
        } else {
            const uint64_t chunk_first = thread_first + s * kBand;
            float* const stash = reinterpret_cast<float*>(s_data) + E * slot;
            float* const grad_stash = reinterpret_cast<float*>(s_grad) + E * slot;
#pragma unroll 1
            for (uint32_t j = 0; j < E; ++j) {
                const bool inside = chunk_first + j < end;
                stash[j] = inside ? loadElement(input, chunk_first + j, x_format) : 0.0f;
                grad_stash[j] = inside ? loadElement(grad, chunk_first + j, dy_format) : 0.0f;
            }
        }
    }

    // Dense rows that divide the tile: every tile starts a row and ends one, so nothing is published or awaited.
    const bool lookback = RAGGED || columns.mod(TILE) != 0;
    uint32_t last_epoch = 0;
    if (t == 0) {
        if (lookback) {
            last_epoch = static_cast<uint32_t>(peek(ws.tile_runs + W * tile) >> 33);
        } else {
            s_epoch = 0;
            s_prefix = Run<A>{A{}, 0U, 0U};
        }
        if constexpr (!RAGGED) row_after = columns.mod(reverse ? tile_begin : tile_begin + TILE) == 0;
        s_closed = (no_next || row_after) ? 1U : 0U;
        s_tail = 0.0f;
    }

    uint64_t band_col = col_first;
    for (uint32_t s = 0; s < CHUNKS; ++s) {
        const uint32_t slot = s * THREADS + t;                      // the chunk's place in memory
        const uint32_t order = reverse ? kSlots - 1 - slot : slot;  // and in scan order
        const uint64_t chunk_first = thread_first + s * kBand;
        const uint64_t scan_first = reverse ? chunk_first + E - 1 : chunk_first;  // the chunk's first element in scan order
        Raw raw[kRaw];
#pragma unroll
        for (uint32_t h = 0; h < kRaw; ++h) raw[h] = s_data[kRaw * slot + h];
        T values[E];  // in scan order, as are the head bits
        unpackInScanOrder(raw, reverse, values);
        const uint32_t heads = mirrorBits(
            chunkHeads<RAGGED, E>(reverse, chunk_first, end, static_cast<uint64_t>(slot) * E, s_heads, cols, band_col, pattern), reverse);
        if (order == 0) s_starts = heads & 1U;
        A acc{};
#pragma unroll
        for (uint32_t jo = 0; jo < E; ++jo) {
            const A key = winnerKey(values[jo], reverse ? scan_first - jo : scan_first + jo, flip, place_flip);
            acc = (jo == 0 || ((heads >> jo) & 1U)) ? key : applyOp<A, KEY>(acc, key);
        }
        s_value[order] = acc;
        s_head[order] = heads != 0 ? 1U : 0U;
        band_col += band_cols;
        if (band_col >= cols) band_col -= cols;
    }

    if (lookback && t == 0) s_epoch = nextEpoch(last_epoch);
    __syncthreads();

    // 2. The slots' key runs, scanned in slot order.
    scanSlotRuns<A, KEY, THREADS, CHUNKS>(s_value, s_head, s_warp_value, s_warp_head);

    // 3-4. Warp 0 publishes the tile's key run (and the group's and supergroup's) and folds the tile's prefix: the
    // scan kernel's steps, word for word.
    const uint32_t epoch = warp == 0 ? s_epoch : 0U;  // 0: no tile publishes or waits
    if (warp == 0 && (RAGGED || epoch != 0)) {
        const Run<A> none{A{}, 0U, 0U};
        const Place place = placeOf<GROUP, SUPER>(tile, ws.num_tiles);
        const uint64_t group = place.group;
        const uint64_t super = place.super;
        const uint32_t pos = place.pos;
        const uint32_t gpos = place.gpos;
        const bool group_duty = place.group_duty;
        const bool super_duty = place.super_duty;
        const bool need_prefix = tile > 0 && s_starts == 0U;
        const Run<A> tile_run{s_warp_value[kWarps - 1], s_warp_head[kWarps - 1], 1U};
        if (lane == 0) publishRun(ws.tile_runs + W * tile, tile_run, epoch);

        const bool want_tiles = need_prefix || group_duty;
        const bool want_groups = need_prefix || super_duty;
        const bool want_previous = super > 0 && (need_prefix || super_duty);
        unsigned long long* const tile_slot = ws.tile_runs + W * (group * GROUP) + W * lane;
        unsigned long long* const group_slot = ws.group_runs + W * (super * SUPER) + W * lane;
        unsigned long long* const previous_slot = ws.super_runs + W * (super > 0 ? super - 1 : 0);
        Words<A> tile_words{};
        Words<A> group_words{};
        Words<A> previous_words{};
        if (want_tiles && lane < pos) tile_words = peekRun<A>(tile_slot);
        if (want_groups && lane < gpos) group_words = peekRun<A>(group_slot);
        if (want_previous && lane == 0) previous_words = peekRun<A>(previous_slot);

        Run<A> run = none;
        if (want_tiles && lane < pos) run = awaitRun<A>(tile_slot, epoch, tile_words);
        if (lane == pos) run = tile_run;
        warpScanRuns<A, KEY>(run, lane, s_fold_value, s_fold_head, s_fold_valid);
        const Run<A> tiles_before =
            pos > 0 ? Run<A>{s_fold_value[pos - 1], s_fold_head[pos - 1], s_fold_valid[pos - 1]} : none;
        const Run<A> group_run{s_fold_value[pos], s_fold_head[pos], 1U};
        __syncwarp();
        if (group_duty && lane == 0) publishRun(ws.group_runs + W * group, group_run, epoch);

        const bool need_groups = need_prefix && !(tiles_before.valid && tiles_before.head);
        Run<A> groups_before = none;
        Run<A> super_run = none;
        if (need_groups || super_duty) {
            run = none;
            if (lane < gpos) run = awaitRun<A>(group_slot, epoch, group_words);
            if (super_duty && lane == gpos) run = group_run;
            warpScanRuns<A, KEY>(run, lane, s_fold_value, s_fold_head, s_fold_valid);
            if (gpos > 0) groups_before = Run<A>{s_fold_value[gpos - 1], s_fold_head[gpos - 1], s_fold_valid[gpos - 1]};
            if (super_duty) super_run = Run<A>{s_fold_value[gpos], s_fold_head[gpos], 1U};
        }

        if (lane == 0) {
            const bool need_previous = need_groups && !(groups_before.valid && groups_before.head);
            Run<A> previous = none;
            if (super > 0 && (need_previous || (super_duty && super_run.head == 0U))) {
                previous = awaitRun<A>(previous_slot, epoch, previous_words);
            }
            if (super_duty) publishRun(ws.super_runs + W * super, combine<A, KEY>(previous, super_run), epoch);
            Run<A> prefix = none;
            if (need_prefix) {
                const Run<A> base = need_previous ? combine<A, KEY>(previous, groups_before) : groups_before;
                prefix = need_groups ? combine<A, KEY>(base, tiles_before) : tiles_before;
            }
            s_prefix = prefix;
        }
    }
    __syncthreads();

    // 5a. Each chunk again, with the keys before it (the tile's prefix and the slots before the chunk): its winners,
    // and its leading sum, the dy that belongs to the winner arriving at the chunk. An inclusive scan gives a winner
    // its own dy; an exclusive one gives it to the winner before (a row's first element in scan order: to nobody).
    const Run<A> prefix = s_prefix;
    if constexpr (RAGGED) end = s_end;
    band_col = col_first;
    for (uint32_t s = 0; s < CHUNKS; ++s) {
        const uint32_t slot = s * THREADS + t;
        const uint32_t order = reverse ? kSlots - 1 - slot : slot;
        const uint64_t chunk_first = thread_first + s * kBand;
        const uint64_t scan_first = reverse ? chunk_first + E - 1 : chunk_first;
        const Run<A> carry =
            combine<A, KEY>(prefix, runBefore<A, KEY, CHUNKS>(order, s_value, s_head, s_warp_value, s_warp_head));
        bool have = carry.valid != 0U;  // a key before this element in its row
        A best = carry.value;

        Raw raw[kRaw];
#pragma unroll
        for (uint32_t h = 0; h < kRaw; ++h) raw[h] = s_data[kRaw * slot + h];
        T values[E];  // in scan order, as are dy, the head bits and the winner bits
        unpackInScanOrder(raw, reverse, values);
        const uint32_t heads = mirrorBits(
            chunkHeads<RAGGED, E>(reverse, chunk_first, end, static_cast<uint64_t>(slot) * E, s_heads, cols, band_col, pattern), reverse);
        uint4 raw_grad[kGradRaw];
#pragma unroll
        for (uint32_t h = 0; h < kGradRaw; ++h) raw_grad[h] = s_grad[kGradRaw * slot + h];
        float dy[E];
        unpackInScanOrder(raw_grad, reverse, dy);

        uint32_t winners = 0;
        float lead = 0.0f;
        bool leading = true;  // no winner in the chunk yet
#pragma unroll
        for (uint32_t jo = 0; jo < E; ++jo) {
            const bool head = ((heads >> jo) & 1U) != 0U;
            const A key = winnerKey(values[jo], reverse ? scan_first - jo : scan_first + jo, flip, place_flip);
            const bool wins = head || !have || key > best;
            if (wins) best = key;
            have = true;
            if (leading && (!wins || (exclusive && !head))) lead += dy[jo];
            if (wins) {
                leading = false;
                winners |= 1U << jo;
            }
        }
        s_winners[slot] = static_cast<uint8_t>(winners);
        s_rowheads[slot] = static_cast<uint8_t>(heads);
        s_sum[kSlots - 1 - order] = lead;
        s_stop[kSlots - 1 - order] = winners != 0 ? 1U : 0U;
        band_col += band_cols;
        if (band_col >= cols) band_col -= cols;
    }
    __syncthreads();

    // 5b. The leading sums scanned from the tile's last slot in scan order to its first: a slot's run is the sum from
    // its start to the first winner at or after it (to the tile's end when there is none).
    scanSlotRuns<float, SUM, THREADS, CHUNKS>(s_sum, s_stop, s_warp_sum, s_warp_stop);
    if (t == 0 && s_epoch != 0U) {
        publishSum(gs.tile_sums + tile, s_warp_sum[kWarps - 1], s_warp_stop[kWarps - 1], s_epoch);
    }

    // 5c. Each chunk from its last element in scan order to its first: `acc` is the sum of the run being walked,
    // starting with what follows the chunk; a winner takes it and ends the run. `pending`: that run goes on past the
    // tile's end, so its winner is left to the tile where it ends (step 6) and its sum so far is the tile's tail.
    // The chunk is stored in dx's format: whole when it has no element to leave out, else element by element.
    const bool closed = s_closed != 0U;
    const uint32_t dx_bytes = formatBytes(dx_format);
    for (uint32_t s = 0; s < CHUNKS; ++s) {
        const uint32_t slot = s * THREADS + t;
        const uint32_t order = reverse ? kSlots - 1 - slot : slot;
        const uint64_t chunk_first = thread_first + s * kBand;
        const bool full = chunk_first + E <= end;
        const Run<float> after = runBefore<float, SUM, CHUNKS>(kSlots - 1 - order, s_sum, s_stop, s_warp_sum, s_warp_stop);
        float acc = after.valid != 0U ? after.value : 0.0f;
        bool pending = !(after.valid != 0U && after.head != 0U) && !closed;
        const uint32_t winners = s_winners[slot];
        const uint32_t rowheads = s_rowheads[slot];
        uint4 raw_grad[kGradRaw];
#pragma unroll
        for (uint32_t h = 0; h < kGradRaw; ++h) raw_grad[h] = s_grad[kGradRaw * slot + h];
        float dy[E];
        unpackInScanOrder(raw_grad, reverse, dy);
        float sums[E];           // in scan order
        uint32_t scan_hole = E;  // the element left to another tile, if any
#pragma unroll
        for (uint32_t back = 0; back < E; ++back) {
            const uint32_t jo = E - 1 - back;
            const bool wins = ((winners >> jo) & 1U) != 0U;
            const bool head = ((rowheads >> jo) & 1U) != 0U;
            if (!(exclusive && wins)) acc += dy[jo];
            float value = 0.0f;
            if (wins) {
                value = acc;
                if (pending) {
                    scan_hole = jo;
                    s_tail = acc;
                }
                pending = false;
                acc = (exclusive && !head) ? dy[jo] : 0.0f;
            }
            sums[jo] = value;
        }
        // Back to memory order.
        float out[E];
#pragma unroll
        for (uint32_t j = 0; j < E; ++j) out[j] = reverse ? sums[E - 1 - j] : sums[j];
        const uint32_t hole = (reverse && scan_hole != E) ? E - 1 - scan_hole : scan_hole;
        uint4 packed[kWords];
        chunkFromFloat(out, dx_format, packed);
        char* const at = static_cast<char*>(output) + chunk_first * dx_bytes;
        if (full && hole == E) {
            if (dx_bytes == 4) {
                uint4* const words = static_cast<uint4*>(__builtin_assume_aligned(at, 32));  // as loadRawChunk
                words[0] = packed[0];
                words[1] = packed[1];
            } else if (dx_bytes == 2) {
                *reinterpret_cast<uint4*>(at) = packed[0];
            } else {
                *reinterpret_cast<uint2*>(at) = make_uint2(packed[0].x, packed[0].y);
            }
        } else {
            const uint32_t word[E] = {packed[0].x, packed[0].y, packed[0].z, packed[0].w,
                                      packed[1].x, packed[1].y, packed[1].z, packed[1].w};
#pragma unroll
            for (uint32_t j = 0; j < E; ++j) {
                if (j == hole || chunk_first + j >= end) continue;
                if (dx_bytes == 4) {
                    reinterpret_cast<uint32_t*>(at)[j] = word[j];
                } else if (dx_bytes == 2) {
                    reinterpret_cast<uint16_t*>(at)[j] = static_cast<uint16_t>(word[j >> 1] >> (16U * (j & 1U)));
                } else {
                    reinterpret_cast<uint8_t*>(at)[j] = static_cast<uint8_t>(word[j >> 2] >> (8U * (j & 3U)));
                }
            }
        }
    }
    __syncthreads();

    // 6. Warp 0: publish the tile's tail; fold and publish the group's and supergroup's sums (their last tile); and,
    // if the run arriving at the tile ends in it, write that run's winner. Every word awaited is a tile's before this
    // one.
    if (warp == 0 && epoch != 0) {
        const Place place = placeOf<GROUP, SUPER>(tile, ws.num_tiles);
        const Run<float> lead{s_warp_sum[kWarps - 1], s_warp_stop[kWarps - 1], 1U};
        if (lane == 0) publishSum(gs.tile_tails + tile, s_tail, 0U, epoch);

        // The run arriving ends here at the tile's first winner, or at its end when no run goes on past it. (A
        // reverse scan's tile holding the end starts with the zeros past the end: no run arrives.)
        bool closes = tile > 0 && s_starts == 0U && prefix.valid != 0U && (lead.head != 0U || closed);
        if (reverse) closes = closes && tile_begin + TILE <= end;
        uint64_t winner = 0;
        uint64_t from_tile = tile;  // the winner's tile, in the lookback
        if (closes) {
            winner = winnerIndex(prefix.value, reverse);
            const uint64_t memory_tile = winner / TILE;
            from_tile = reverse ? ws.num_tiles - 1 - memory_tile : memory_tile;
        }
        const Place from = placeOf<GROUP, SUPER>(from_tile, ws.num_tiles);
        const bool same_group = from.group == place.group;
        const bool same_super = from.super == place.super;

        // The sums of this group's tiles before this one: all of them for the group's sum or a run from an earlier
        // group, those after the winner's tile for a run from this group.
        float before = 0.0f;
        if (lane < place.pos && (place.group_duty || (closes && (!same_group || lane > from.pos)))) {
            before = awaitSum(gs.tile_sums + place.group * GROUP + lane, epoch);
        }
        s_group_tiles[lane] = before;
        __syncwarp();
        float group_sum = 0.0f;
        if (place.group_duty && lane == 0) {
            for (uint32_t i = 0; i < place.pos; ++i) group_sum += s_group_tiles[i];
            group_sum += lead.value;
            publishSum(gs.group_sums + place.group, group_sum, 0U, epoch);
        }
        if (place.super_duty) {
            float value = 0.0f;
            if (lane < place.gpos) value = awaitSum(gs.group_sums + place.super * SUPER + lane, epoch);
            const float groups = foldLanes(value, lane, s_fold_sum, 0, place.gpos, 0.0f);
            if (lane == 0) publishSum(gs.super_sums + place.super, groups + group_sum, 0U, epoch);
        }

        if (closes) {
            float acc = 0.0f;
            if (lane == 0) acc = awaitSum(gs.tile_tails + from_tile, epoch);
            if (same_group) {
                if (lane == 0) {
                    for (uint32_t i = from.pos + 1; i < place.pos; ++i) acc += s_group_tiles[i];
                }
            } else {
                // The rest of the winner's group, whole groups, whole supergroups, groups, then this group's tiles.
                float value = 0.0f;
                if (lane > from.pos && lane < GROUP) value = awaitSum(gs.tile_sums + from.group * GROUP + lane, epoch);
                acc = foldLanes(value, lane, s_fold_sum, from.pos + 1, GROUP, acc);
                if (same_super) {
                    value = 0.0f;
                    if (lane > from.gpos && lane < place.gpos) {
                        value = awaitSum(gs.group_sums + from.super * SUPER + lane, epoch);
                    }
                    acc = foldLanes(value, lane, s_fold_sum, from.gpos + 1, place.gpos, acc);
                } else {
                    value = 0.0f;
                    if (lane > from.gpos && lane < SUPER) value = awaitSum(gs.group_sums + from.super * SUPER + lane, epoch);
                    acc = foldLanes(value, lane, s_fold_sum, from.gpos + 1, SUPER, acc);
                    if (lane == 0) {
                        for (uint64_t super = from.super + 1; super < place.super; ++super) {
                            acc += awaitSum(gs.super_sums + super, epoch);
                        }
                    }
                    value = 0.0f;
                    if (lane < place.gpos) value = awaitSum(gs.group_sums + place.super * SUPER + lane, epoch);
                    acc = foldLanes(value, lane, s_fold_sum, 0, place.gpos, acc);
                }
                if (lane == 0) {
                    for (uint32_t i = 0; i < place.pos; ++i) acc += s_group_tiles[i];
                }
            }
            if (lane == 0) storeElement(output, winner, acc + lead.value, dx_format);
        }
    }
}

#if defined(__CUDACC__)
template <typename OFF>
void launchKernel(const BackwardLaunch& args, bool reverse, cudaStream_t stream) {
    const unsigned blocks = static_cast<unsigned>(args.keys.num_tiles);
    scanMinMaxBackwardKernel<OFF><<<blocks, kThreads, 0, stream>>>(
        args.x,
        args.dy,
        args.dx,
        args.x_format,
        args.dy_format,
        args.dx_format,
        args.n,
        args.columns,
        static_cast<const OFF*>(args.offsets),
        args.rows,
        args.keys,
        args.sums,
        args.min,
        args.exclusive,
        reverse);
}
#endif

}  // namespace ThorImplementation::ScanMinMaxBackwardKernels
