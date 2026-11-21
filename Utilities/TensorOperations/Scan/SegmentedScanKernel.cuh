#pragma once

// Thor's single-pass segmented scan (plans/Thor_Scan_Kernel_Plan_2026-11-17.md): inclusive or exclusive, forward or
// reverse sum, product, min, max, argmax and argmin over dense rows (equal lengths) or ragged rows (offsets), every
// float accumulating in FP32, bitwise reproducible run to run. The host side is SegmentedScan.h; what both see is
// SegmentedScanLaunch.h.
//
// One CTA per tile of THREADS x CHUNKS x ITEMS elements: the tile of the block's own index (see below). A tile is
// CHUNKS bands of THREADS chunks of ITEMS elements; thread t owns chunk t of every band, so chunk (band s, thread t)
// is slot s x THREADS + t of the tile, and one load or store of a warp covers 32 adjacent chunks whatever CHUNKS is.
// A chunk is 32 bytes of the wider of input and output: every scan loads or stores 32 bytes per chunk, and one that
// writes a narrower float than it reads stores 16 or 8.
//   1. the thread issues the loads of its chunk of every band at once (16 B vector loads; one 8 B load for an 8-byte
//      chunk) and stashes the raw bytes in shared memory as they arrive; then, band by band, it reads its chunk back
//      from the stash, marks the chunk's row heads and folds its elements into the slot's (head, value) run;
//   2. the slots' runs are scanned in shared memory in slot order, a fixed tree: thread t scans the CHUNKS adjacent
//      slots from CHUNKS x t in registers (not the slots it folded, which are a band apart), each warp scans its
//      threads' totals, and warp 0 scans the THREADS / 32 warp totals;
//   3. the tile publishes its aggregate; the last tile of each group of GROUP tiles publishes the group's aggregate,
//      and the last group of each supergroup of SUPER groups publishes the supergroup's inclusive value;
//   4. the tile's prefix is the fixed fold  inclusive(previous supergroup) + groups before it in its supergroup +
//      tiles before it in its group. Warp 0 does the carry: one lane per tile (or group) loads its published run, and
//      a Kogge-Stone scan in shared memory folds them in a fixed tree. A level whose fold already holds a row head
//      ends the prefix there (the segmented operator returns a run with a head unchanged, so skipping the earlier
//      levels gives the same bits). Dense rows whose length divides the tile start a row in every tile: no tile
//      publishes or waits;
//   5. band by band, each thread scans its chunk again from the stash (the same operations in the same order, so the
//      same bits), the elements before its first head taking the carry and the slots before it, and writes 16 bytes
//      at a time. An exclusive scan writes each element the inclusive result of the element before it (a row's first
//      element: the op's identity), so only this step differs between the modes.
// The association is fixed whoever computes what and in which order tiles finish, so results repeat bit for bit.
// A reverse scan runs the same steps in scan order, last element first: the tiles keep their place in memory (and their
// aligned loads and stores), block b takes the tile T - 1 - b, each tile's slots and each chunk's elements are taken
// from last to first, and a run starts at a row's last element. Steps 2 to 4 see only slots in scan order, so they
// are the forward scan's, and the block index is still the tile's place in the lookback.
// Argmax and argmin run the same steps on a 64-bit key that packs each element's value and index (keyOf), and write
// the index and, when asked, the value (decoded from the key); their runs are published in two words.
//
// Why the stash and CHUNKS: on long rows a tile waits for the tiles before it, so the speed follows the bytes each SM
// holds in flight. Registers (64 Ki per SM) and threads (1536 per SM on sm_120) cap what can wait in registers at 48 KB
// of input per SM; shared memory holds twice that, and holds it with few CTAs, so the kernel need not fit a small
// register budget. THREADS, GROUP, SUPER, CHUNKS and ITEMS are template parameters so that tests can run small tiles;
// the launcher uses the constants of SegmentedScanLaunch.h.
//
// Rows: element 0 always starts one. Ragged rows start at every offset and end at offsets[rows] (or at n, if that is
// sooner): elements before offsets[0] are a row of their own, empty rows start nothing, and nothing from the end on is
// read or written (Thor's ragged contract: values past the active rows are undefined, and work follows the active
// rows). The tiles past the end only keep the workspace's epoch (a reverse scan meets them first: the tiles with rows
// may wait for their empty runs, but the last element starts a run, so no output takes them in). Input and output
// must be 16-byte aligned.

#include "Utilities/TensorOperations/Scan/SegmentedScanLaunch.h"

#include <cuda/atomic>
#include <cuda/std/bit>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <cstring>
#include <type_traits>

namespace ThorImplementation::SegmentedScanKernels {

// A value as the scan computes with it: FP32 for floats, UINT32 for UINT32.
template <typename T>
using Value = std::conditional_t<std::is_same_v<T, uint32_t>, uint32_t, float>;

// What the scan accumulates: the value, or for an arg scan a 64-bit key (keyOf below).
template <typename T, SegmentedScanOp OP>
using Acc = std::conditional_t<isArg(OP), uint64_t, Value<T>>;

template <typename T>
__host__ __device__ inline Value<T> toAcc(T value) {
    if constexpr (std::is_same_v<T, __half>) {
        return __half2float(value);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return __bfloat162float(value);
    } else {
        return static_cast<Value<T>>(value);
    }
}

template <typename O, typename A>
__host__ __device__ inline O fromAcc(A value) {
    if constexpr (std::is_same_v<O, __half>) {
        return __float2half_rn(value);
    } else if constexpr (std::is_same_v<O, __nv_bfloat16>) {
        return __float2bfloat16_rn(value);
    } else if constexpr (std::is_same_v<O, float> || std::is_same_v<O, uint32_t>) {
        return value;
    } else {
        return O(value);  // FP8: the explicit, saturating-to-finite constructor
    }
}

// An arg scan's key for the element at flat index `index` (the host takes at most 2^31 elements):
//   bits 63-32  32 bits that order as the values do (argmin inverts them, so the larger key always wins; -0 is +0, and
//               a NaN is below every number);
//   bits 31-1   2^31 - 1 - index, so of equal values the first wins (a reverse scan: the index, so the last wins, the
//               first it meets);
//   bit 0       the value is -0, so the value decodes exactly (valueOf). Indices are unique, so this bit never decides.
// Both arg scans are then one unsigned max.
template <typename T, SegmentedScanOp OP, SegmentedScanDirection DIR>
__host__ __device__ inline uint64_t keyOf(T value, uint64_t index) {
    uint32_t order;
    uint32_t negative_zero = 0U;
    if constexpr (std::is_same_v<T, uint32_t>) {
        order = OP == SegmentedScanOp::ArgMin ? ~value : value;
    } else {
        uint32_t bits = cuda::std::bit_cast<uint32_t>(toAcc(value));  // exact for every float type
        negative_zero = bits == 0x80000000U ? 1U : 0U;
        if ((bits << 1) == 0U) bits = 0U;                         // -0 as +0
        order = (bits >> 31) != 0U ? ~bits : bits | 0x80000000U;  // the sign flipped in, negatives' magnitude out
        if constexpr (OP == SegmentedScanOp::ArgMin) order = ~order;
        if ((bits & 0x7FFFFFFFU) > 0x7F800000U) order = 0U;  // NaN; no number has order 0 either way
    }
    const uint32_t place = DIR == SegmentedScanDirection::Reverse ? static_cast<uint32_t>(index) << 1
                                                                  : (0x7FFFFFFFU - static_cast<uint32_t>(index)) << 1;
    return (static_cast<uint64_t>(order) << 32) | place | negative_zero;
}

// A float as the input type, exactly (every value that reaches it came from the input type). FP8 takes the
// saturating conversion, one instruction from sm_89 (the non-saturating one is software, through double): a value
// exact in the type converts to itself and a NaN to a NaN, and only E5M2's infinities would clamp to 57344, so they
// are put back. E4M3 has no infinity.
template <typename T>
__host__ __device__ inline T fromValue(float value) {
    if constexpr (std::is_same_v<T, __half>) {
        return __float2half_rn(value);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return __float2bfloat16_rn(value);
    } else if constexpr (std::is_same_v<T, float>) {
        return value;
    } else {
        constexpr __nv_fp8_interpretation_t kind = std::is_same_v<T, __nv_fp8_e4m3> ? __NV_E4M3 : __NV_E5M2;
        __nv_fp8_storage_t bits = __nv_cvt_float_to_fp8(value, __NV_SATFINITE, kind);
        if constexpr (kind == __NV_E5M2) {
            const uint32_t f = cuda::std::bit_cast<uint32_t>(value);
            if ((f & 0x7FFFFFFFU) == 0x7F800000U) bits = static_cast<__nv_fp8_storage_t>(((f >> 24) & 0x80U) | 0x7CU);
        }
        return cuda::std::bit_cast<T>(bits);
    }
}

// The value an arg scan's key was made from (keyOf inverted): the same bits, but a NaN decodes to the canonical NaN.
template <typename T, SegmentedScanOp OP>
__host__ __device__ inline T valueOf(uint64_t key) {
    uint32_t order = static_cast<uint32_t>(key >> 32);
    if constexpr (std::is_same_v<T, uint32_t>) {
        return OP == SegmentedScanOp::ArgMin ? ~order : order;
    } else {
        const bool is_nan = order == 0U;
        if constexpr (OP == SegmentedScanOp::ArgMin) order = ~order;
        uint32_t bits = (order >> 31) != 0U ? order & 0x7FFFFFFFU : ~order;
        bits |= static_cast<uint32_t>(key & 1U) << 31;  // -0
        return fromValue<T>(cuda::std::bit_cast<float>(is_nan ? 0x7FC00000U : bits));
    }
}

// One element as the scan accumulates it.
template <typename T, SegmentedScanOp OP, SegmentedScanDirection DIR>
__host__ __device__ inline Acc<T, OP> element(T value, uint64_t index) {
    if constexpr (isArg(OP)) {
        return keyOf<T, OP, DIR>(value, index);
    } else {
        return toAcc(value);
    }
}

// What the scan writes for an accumulated value: the value in the output type, or an arg scan's index (valueOf gives
// its value).
template <typename O, SegmentedScanOp OP, SegmentedScanDirection DIR, typename A>
__host__ __device__ inline O result(A value) {
    if constexpr (isArg(OP)) {
        const uint32_t place = static_cast<uint32_t>(value) >> 1;
        return DIR == SegmentedScanDirection::Reverse ? place : 0x7FFFFFFFU - place;
    } else {
        return fromAcc<O, A>(value);
    }
}

// A float sum's or product's FP32 value as a SharedFloat output, in the format the launch chose (to nearest; FP8
// saturating, as fromAcc).
template <typename O>
__host__ __device__ inline O sharedFromFloat(float value, bool alternate) {
    O out;
    if constexpr (sizeof(O) == 2) {
        if (alternate) {
            const __nv_bfloat16 b = __float2bfloat16_rn(value);
            memcpy(&out.bits, &b, 2);
        } else {
            const __half h = __float2half_rn(value);
            memcpy(&out.bits, &h, 2);
        }
    } else {
        out.bits = __nv_cvt_float_to_fp8(value, __NV_SATFINITE, alternate ? __NV_E5M2 : __NV_E4M3);
    }
    return out;
}

// What an exclusive scan writes for a row's first element, in the type O it writes: 0 for a sum, 1 for a product;
// for a min (max) +inf (-inf), or E4M3's largest (lowest) value, +-448, as it has no infinity; UINT32's largest (0).
// These are CUB's identities, and an exclusive arg scan's value at a row's first element is the min's (argmin) or the
// max's (argmax) in the input's type.
template <typename O, SegmentedScanOp OP>
__host__ __device__ inline O identityOf() {
    static_assert(!isArg(OP), "an arg scan writes the index 0xFFFFFFFF and the min's or max's identity");
    constexpr bool kMax = OP == SegmentedScanOp::Max;
    if constexpr (OP == SegmentedScanOp::Sum || OP == SegmentedScanOp::Product) {
        return fromAcc<O>(static_cast<Value<O>>(OP == SegmentedScanOp::Sum ? 0 : 1));
    } else if constexpr (std::is_same_v<O, uint32_t>) {
        return kMax ? 0U : 0xFFFFFFFFU;
    } else if constexpr (std::is_same_v<O, __nv_fp8_e4m3>) {
        return cuda::std::bit_cast<O>(static_cast<__nv_fp8_storage_t>(kMax ? 0xFEU : 0x7EU));
    } else if constexpr (std::is_same_v<O, __nv_fp8_e5m2>) {
        return cuda::std::bit_cast<O>(static_cast<__nv_fp8_storage_t>(kMax ? 0xFCU : 0x7CU));
    } else {
        return fromAcc<O>(cuda::std::bit_cast<float>(kMax ? 0xFF800000U : 0x7F800000U));  // FP32, FP16, BF16
    }
}

// Min and max skip NaN (fminf, fmaxf), so a NaN never wins after the first element; so do the arg scans, through their
// keys. UINT32 sums and products wrap.
template <typename A, SegmentedScanOp OP>
__host__ __device__ inline A applyOp(A a, A b) {
    if constexpr (OP == SegmentedScanOp::Sum) {
        return a + b;
    } else if constexpr (OP == SegmentedScanOp::Product) {
        return a * b;
    } else if constexpr (OP == SegmentedScanOp::Max) {
        if constexpr (std::is_same_v<A, float>) {
            return fmaxf(a, b);
        } else {
            return a < b ? b : a;
        }
    } else if constexpr (OP == SegmentedScanOp::Min) {
        if constexpr (std::is_same_v<A, float>) {
            return fminf(a, b);
        } else {
            return b < a ? b : a;
        }
    } else {
        static_assert(isArg(OP) && std::is_same_v<A, uint64_t>, "an arg scan's keys");
        return a < b ? b : a;
    }
}

// A run of elements: its op value from its last head (or its start), whether it contains a head, and whether it holds
// anything at all (an empty run combines as nothing).
template <typename A>
struct Run {
    A value;
    uint32_t head;
    uint32_t valid;
};

// `a` comes before `b`.
template <typename A, SegmentedScanOp OP>
__host__ __device__ inline Run<A> combine(Run<A> a, Run<A> b) {
    if (!a.valid) return b;
    if (!b.valid || b.head == 0U) {
        if (!b.valid) return a;
        return Run<A>{applyOp<A, OP>(a.value, b.value), a.head, 1U};
    }
    return b;
}

// A published word: 32 bits of the run's value low, (epoch << 1 | head) high. A run is one word, or two for an arg
// scan's 64-bit key (word k holds bits 32k up), each with the epoch and head bit. The epoch tells this launch's words
// from earlier launches' without clearing the workspace. Every launch writes every run word, so when a launch starts
// every word holds the last launch's epoch (0 before the first): each block reads its own tile's first word, which only
// it writes, and takes the next epoch. All blocks agree without an atomic, and the host passes no epoch, so a launch
// captured in a CUDA graph is a new epoch at every replay.
__host__ __device__ inline unsigned long long packRun(uint32_t value_bits, uint32_t head, uint32_t epoch) {
    return (static_cast<unsigned long long>((epoch << 1) | head) << 32) | value_bits;
}

// A run's words as loaded from the workspace: the one word of a 32-bit run, or the two of an arg scan's. A one-word run
// is a plain word, so the value scans compile as they did before arg scans (1482).
struct WordPair {
    unsigned long long low;
    unsigned long long high;
};

template <typename A>
using Words = std::conditional_t<sizeof(A) == sizeof(uint32_t), unsigned long long, WordPair>;

template <typename A>
__host__ __device__ inline Run<A> unpackRun(Words<A> words) {
    if constexpr (sizeof(A) == sizeof(uint32_t)) {
        return Run<A>{cuda::std::bit_cast<A>(static_cast<uint32_t>(words)), static_cast<uint32_t>(words >> 32) & 1U,
                      1U};
    } else {
        const uint64_t value =
            (static_cast<uint64_t>(static_cast<uint32_t>(words.high)) << 32) | static_cast<uint32_t>(words.low);
        return Run<A>{value, static_cast<uint32_t>(words.low >> 32) & 1U, 1U};
    }
}

// Published and awaited with relaxed atomics. A word is a whole message (32 bits of the value, the head bit and the
// epoch, written by one 64-bit store, which is single-copy atomic), and nothing else is read through it, so no ordering
// against other memory is needed: coherence is enough, and a relaxed load at device scope reads L2, so a poll sees the
// store once it lands. (Release stores and acquire loads cost a MEMBAR.ALL.GPU before every publish and an L1
// invalidation after every poll: plans/Thor_Scan_Kernel_Log7_2026-11-18.md, 1468.) The two words of an arg scan's run
// need no ordering between them either: each is written once per launch, by the run's one writer, so once both hold
// this launch's epoch both are that writer's halves.
__device__ inline void publish(unsigned long long* slot, unsigned long long word) {
    cuda::atomic_ref<unsigned long long, cuda::thread_scope_device> ref(*slot);
    ref.store(word, cuda::memory_order_relaxed);
}

template <typename A>
__device__ inline void publishRun(unsigned long long* slot, Run<A> run, uint32_t epoch) {
    if constexpr (sizeof(A) == sizeof(uint32_t)) {
        publish(slot, packRun(cuda::std::bit_cast<uint32_t>(run.value), run.head, epoch));
    } else {
        publish(slot, packRun(static_cast<uint32_t>(run.value), run.head, epoch));
        publish(slot + 1, packRun(static_cast<uint32_t>(run.value >> 32), run.head, epoch));
    }
}

__device__ inline unsigned long long peek(unsigned long long* slot) {
    cuda::atomic_ref<unsigned long long, cuda::thread_scope_device> ref(*slot);
    return ref.load(cuda::memory_order_relaxed);
}

template <typename A>
__device__ inline Words<A> peekRun(unsigned long long* slot) {
    if constexpr (sizeof(A) == sizeof(uint32_t)) {
        return peek(slot);
    } else {
        return WordPair{peek(slot), peek(slot + 1)};
    }
}

// Polls a slot until it holds this launch's word, starting from a word already loaded from it.
__device__ inline unsigned long long await(unsigned long long* slot, uint32_t epoch, unsigned long long word) {
    while (static_cast<uint32_t>(word >> 33) != epoch) {
        __nanosleep(32);
        word = peek(slot);
    }
    return word;
}

// Polls a run's words until all hold this launch's, starting from words already loaded.
template <typename A>
__device__ inline Run<A> awaitRun(unsigned long long* slot, uint32_t epoch, Words<A> words) {
    if constexpr (sizeof(A) == sizeof(uint32_t)) {
        return unpackRun<A>(await(slot, epoch, words));
    } else {
        return unpackRun<A>(WordPair{await(slot, epoch, words.low), await(slot + 1, epoch, words.high)});
    }
}

// The inclusive segmented scan of one run per lane of a warp, a fixed tree in shared memory; every lane's result is
// left in the arrays (lane i: the fold of lanes 0..i). Called by a whole warp.
template <typename A, SegmentedScanOp OP>
__device__ inline void warpScanRuns(Run<A> run, uint32_t lane, A* s_value, uint32_t* s_head, uint32_t* s_valid) {
    s_value[lane] = run.value;
    s_head[lane] = run.head;
    s_valid[lane] = run.valid;
    __syncwarp();
#pragma unroll
    for (uint32_t offset = 1; offset < 32; offset <<= 1) {
        Run<A> before{A{}, 0U, 0U};
        if (lane >= offset) before = Run<A>{s_value[lane - offset], s_head[lane - offset], s_valid[lane - offset]};
        __syncwarp();
        run = combine<A, OP>(before, run);
        s_value[lane] = run.value;
        s_head[lane] = run.head;
        s_valid[lane] = run.valid;
        __syncwarp();
    }
}

// Elements per chunk for input T and output O (SegmentedScanLaunch.h).
template <typename T, typename O>
inline constexpr uint32_t kItems = chunkElements(sizeof(T), sizeof(O));

template <uint32_t THREADS, uint32_t CHUNKS, uint32_t ITEMS>
inline constexpr uint64_t kTile = static_cast<uint64_t>(THREADS) * CHUNKS * ITEMS;

// A chunk's raw input is loaded and stashed in the widest words that divide it: 16 bytes, or 8 for an 8-byte chunk.
template <uint32_t CHUNK_BYTES>
using RawWord = std::conditional_t<CHUNK_BYTES % sizeof(uint4) == 0, uint4, unsigned long long>;

// One store of 16, 8 or 4 bytes (an arg scan's values beside 16 bytes of indices; a narrowing scan's output).
template <uint32_t BYTES>
using StoreWord = std::conditional_t<BYTES == 16, uint4, std::conditional_t<BYTES == 8, uint2, uint32_t>>;

// A tile's place in the lookback: its group and supergroup, its lane in each, and whether it publishes the group's run
// (the last tile of a group) and the supergroup's inclusive run (the last tile of a supergroup) besides its own.
struct Place {
    uint64_t group;
    uint64_t super;
    uint32_t pos;   // the tile's lane in its group
    uint32_t gpos;  // the group's lane in its supergroup
    bool group_duty;
    bool super_duty;
};

template <uint32_t GROUP, uint32_t SUPER>
__host__ __device__ inline Place placeOf(uint64_t tile, uint64_t num_tiles) {
    const uint64_t group = tile / GROUP;
    const uint64_t super = group / SUPER;
    const uint64_t num_groups = (num_tiles + GROUP - 1) / GROUP;
    const uint32_t pos = static_cast<uint32_t>(tile - group * GROUP);
    const uint32_t gpos = static_cast<uint32_t>(group - super * SUPER);
    const bool group_duty = pos == GROUP - 1 || tile + 1 == num_tiles;
    const bool super_duty = group_duty && (gpos == SUPER - 1 || group + 1 == num_groups);
    return Place{group, super, pos, gpos, group_duty, super_duty};
}

// The first r in [0, rows] with offsets[r] >= target, found by all threads probing THREADS positions per round.
template <uint32_t THREADS, typename OFF>
__device__ inline uint32_t lowerBoundOffsets(const OFF* offsets, uint32_t rows, uint64_t target) {
    __shared__ uint32_t s_bounds[2];
    if (threadIdx.x == 0) {
        s_bounds[0] = 0;
        s_bounds[1] = rows + 1;
    }
    __syncthreads();
    while (true) {
        const uint32_t lo = s_bounds[0];
        const uint32_t hi = s_bounds[1];
        if (lo >= hi) break;
        const uint32_t step = (hi - lo + THREADS - 1) / THREADS;
        const uint32_t probe = lo + threadIdx.x * step;
        const bool valid = probe < hi;
        const bool below = valid && static_cast<uint64_t>(offsets[probe]) < target;
        const uint32_t probes = (hi - lo + step - 1) / step;
        const uint32_t count = static_cast<uint32_t>(__syncthreads_count(below ? 1 : 0));
        if (threadIdx.x == 0) {
            s_bounds[0] = count == 0 ? lo : lo + (count - 1) * step + 1;
            s_bounds[1] = count < probes ? lo + count * step : hi;
        }
        __syncthreads();
    }
    const uint32_t result = s_bounds[0];
    __syncthreads();
    return result;
}

// Dense rows: the head bits of a chunk of E elements whose first element is in column `col` of rows of `cols` columns
// (bit j: element j starts a run: a row's first element, or for a reverse scan its last). `pattern` has a bit at every
// multiple of cols below 32, so the chunk's heads are the pattern shifted to the chunk's first head: no work per
// element.
template <uint32_t E, SegmentedScanDirection DIR>
__device__ inline uint32_t denseHeads(uint64_t col, uint64_t cols, uint32_t pattern) {
    static_assert(E >= 1 && E <= 32, "a head bit per element");
    // The chunk's first row start, or (reverse) its first row end: the element before the first row start after it.
    const uint64_t first = DIR == SegmentedScanDirection::Reverse ? cols - col - 1 : (col == 0 ? 0 : cols - col);
    const uint32_t mask = ~0U >> (32 - E);  // the low E bits
    return first < E ? (pattern << static_cast<uint32_t>(first)) & mask : 0U;
}

// One chunk of a thread: its E elements (from the raw bytes) and its head bits (bit j: element j starts a run, a row's
// first element or for a reverse scan its last). Dense rows take them from the chunk's first column, `col`; ragged rows
// read the tile's bitmap at the chunk's place in the tile, `local`. A reverse scan's last element is a head too
// (endHead).
template <typename T, bool RAGGED, uint32_t E, SegmentedScanDirection DIR>
__device__ inline uint32_t unpackChunk(const RawWord<E * sizeof(T)>* raw,
                                       T* values,
                                       uint64_t chunk_first,
                                       uint64_t local,
                                       const uint32_t* s_heads,
                                       uint64_t cols,
                                       uint64_t col,
                                       uint32_t pattern) {
    memcpy(values, raw, E * sizeof(T));
    uint32_t heads = 0;
    if constexpr (RAGGED) {
#pragma unroll
        for (uint32_t j = 0; j < E; ++j) {
            const uint64_t k = local + j;
            heads |= ((s_heads[k >> 5] >> (k & 31U)) & 1U) << j;
        }
    } else {
        heads = denseHeads<E, DIR>(col, cols, pattern);
    }
    if constexpr (DIR == SegmentedScanDirection::Forward) {
        if (chunk_first == 0) heads |= 1U;  // the first element always starts a run
    }
    return heads;
}

// A reverse scan's head bit for the last element (end - 1), if it is in the chunk from chunk_first: it always starts a
// run, and nothing at or past the end is scanned.
template <uint32_t E>
__device__ inline uint32_t endHead(uint64_t chunk_first, uint64_t end) {
    return chunk_first < end && end - chunk_first <= E ? 1U << static_cast<uint32_t>(end - 1 - chunk_first) : 0U;
}

// MODE: inclusive or exclusive. DIR: forward or reverse. OFF: void for dense rows of columns.count elements, the offset
// type for ragged rows. arg_values: an arg scan also writes each element's winning value there (the input's type) when
// it is not null; the value scans ignore it. alternate_output: a SharedFloat output's format (BF16, E5M2); the other
// outputs ignore it.
template <typename T,
          typename O,
          SegmentedScanOp OP,
          SegmentedScanMode MODE,
          SegmentedScanDirection DIR,
          typename OFF,
          uint32_t THREADS = kThreads,
          uint32_t GROUP = kGroup,
          uint32_t SUPER = kSuper,
          uint32_t CHUNKS = kChunks,
          uint32_t ITEMS = kItems<T, O>>
__global__ void __launch_bounds__(THREADS)
    segmentedScanKernel(const T* __restrict__ input,
                        O* __restrict__ output,
                        uint64_t n,
                        Columns columns,
                        const OFF* __restrict__ offsets,
                        uint32_t rows,
                        Workspace ws,
                        T* __restrict__ arg_values,
                        bool alternate_output) {
    using A = Acc<T, OP>;
    constexpr uint32_t W = runWords(OP);  // workspace words per run
    constexpr bool RAGGED = !std::is_void_v<OFF>;
    constexpr bool REVERSE = DIR == SegmentedScanDirection::Reverse;
    const uint64_t cols = columns.count;
    constexpr uint32_t E = ITEMS;
    constexpr uint64_t TILE = kTile<THREADS, CHUNKS, ITEMS>;
    using Raw = RawWord<E * sizeof(T)>;
    constexpr uint32_t kRaw = E * sizeof(T) / sizeof(Raw);  // raw words per chunk
    constexpr uint32_t kWarps = THREADS / 32;
    constexpr uint32_t kSlots = THREADS * CHUNKS;  // chunks per tile
    static_assert(THREADS % 32 == 0 && kWarps <= 32, "whole warps; one lane of warp 0 per warp total");
    static_assert(GROUP <= 32 && SUPER <= 32, "one lane of warp 0 per tile of a group and per group of a supergroup");
    static_assert(E <= 32 && kRaw * sizeof(Raw) == E * sizeof(T), "a head bit per element; whole raw words per chunk");

    __shared__ Raw s_data[kRaw * kSlots];  // raw input: slot p at kRaw * p
    __shared__ uint32_t s_heads[RAGGED ? TILE / 32 : 1];
    __shared__ A s_value[kSlots];  // the slots' runs: each chunk's own, then (step 2) scanned
    __shared__ uint8_t s_head[kSlots];
    __shared__ A s_warp_value[kWarps];
    __shared__ uint32_t s_warp_head[kWarps];
    __shared__ uint64_t s_band_cols;     // dense rows: the columns from one band's chunk to the next, modulo cols
    __shared__ uint32_t s_head_pattern;  // dense rows: a bit at every multiple of cols below 32
    __shared__ A s_fold_value[32];
    __shared__ uint32_t s_fold_head[32];
    __shared__ uint32_t s_fold_valid[32];
    __shared__ uint32_t s_starts;  // the tile's first element starts a row
    __shared__ Run<A> s_prefix;
    __shared__ uint32_t s_epoch;
    __shared__ uint64_t s_end;  // ragged rows: where the rows end

    const uint32_t t = threadIdx.x;
    const uint32_t lane = t & 31U;
    const uint32_t warp = t >> 5;
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
    const uint64_t tile_begin = (REVERSE ? ws.num_tiles - 1 - tile : tile) * TILE;  // where it is in memory
    const uint64_t thread_first = tile_begin + static_cast<uint64_t>(t) * E;  // the thread's chunk of band 0
    constexpr uint64_t kBand = static_cast<uint64_t>(THREADS) * E;            // elements per band

    // 1. Mark the tile's row heads (ragged), then load and stash the thread's chunk of every band, then fold them.
    // `end` is where the scan ends: n, or for ragged rows the end of the last row if that is sooner.
    uint64_t end = n;
    if constexpr (RAGGED) {
        const uint64_t rows_end = offsets[rows];
        if (rows_end < end) end = rows_end;
        if (tile_begin >= end) {
            // A tile past the rows scans nothing. It still writes the words it owns, empty and with this launch's
            // epoch, so every word goes on holding the last launch's epoch whatever rows each launch has. Forward, no
            // tile with rows waits for these: a tile reads only words that tiles before it write. Reverse, they come
            // first, and what a tile with rows folds in from them reaches no output (the last element starts a run).
            if (t == 0) {
                const Place place = placeOf<GROUP, SUPER>(tile, ws.num_tiles);
                const uint32_t epoch = nextEpoch(static_cast<uint32_t>(peek(ws.tile_runs + W * tile) >> 33));
                const unsigned long long empty = packRun(0U, 0U, epoch);
                publish(ws.tile_runs + W * tile, empty);
                if (place.group_duty) publish(ws.group_runs + W * place.group, empty);
                if (place.super_duty) publish(ws.super_runs + W * place.super, empty);
                if constexpr (W == 2) {  // an arg scan's second words
                    publish(ws.tile_runs + W * tile + 1, empty);
                    if (place.group_duty) publish(ws.group_runs + W * place.group + 1, empty);
                    if (place.super_duty) publish(ws.super_runs + W * place.super + 1, empty);
                }
            }
            return;
        }
        if (t == 0) s_end = end;  // step 5 reads it back: no register holds it across the scan
        for (uint32_t i = t; i < TILE / 32; i += THREADS) s_heads[i] = 0;
        // Forward: the row starts in the tile. Reverse: the row ends, the element before each start after the tile's
        // first element (the last element, end - 1, is added as each chunk is read: endHead).
        const uint64_t first_start = REVERSE ? tile_begin + 1 : tile_begin;
        const uint32_t r0 = lowerBoundOffsets<THREADS>(offsets, rows, first_start);  // syncs after the clear
        for (uint32_t r = r0 + t; r <= rows; r += THREADS) {
            const uint64_t start = offsets[r];
            if (start >= first_start + TILE) break;
            if (start < end) {
                const uint64_t at = start - first_start;  // REVERSE: start - 1 - tile_begin
                atomicOr(&s_heads[at >> 5], 1U << (at & 31U));
            }
        }
        __syncthreads();
    }
    const uint64_t col_first = RAGGED ? 0 : columns.mod(thread_first);
    uint64_t band_cols = 0;
    uint32_t pattern = 0;
    if constexpr (!RAGGED) pattern = s_head_pattern;
    if constexpr (!RAGGED && CHUNKS > 1) band_cols = s_band_cols;

    // Every band's loads are issued before any of them is used: a load holds a warp back only where its result is
    // needed, so the thread's CHUNKS chunks are in flight together and arrive after one wait, not CHUNKS waits.
    // Only the tile holding the end can hold a partial chunk, so one test per tile keeps the element loads off every
    // other tile's path. A partial chunk is loaded with zeros past the end: they follow the last element, so the folds
    // below need no test per element, and what they fold into (with any op) reaches only outputs that are never stored
    // and the run of the tile holding the end, which no tile reads. (A reverse scan meets the zeros first, and the last
    // element after them starts a run, so nothing it computes takes them in.)
    Raw loaded[CHUNKS][kRaw];
    if (tile_begin + TILE <= end) {
#pragma unroll
        for (uint32_t s = 0; s < CHUNKS; ++s) {
            const Raw* source = reinterpret_cast<const Raw*>(input + thread_first + s * kBand);
#pragma unroll
            for (uint32_t h = 0; h < kRaw; ++h) loaded[s][h] = source[h];
        }
    } else {
#pragma unroll
        for (uint32_t s = 0; s < CHUNKS; ++s) {
            const uint64_t chunk_first = thread_first + s * kBand;
            if (chunk_first + E <= end) {
                const Raw* source = reinterpret_cast<const Raw*>(input + chunk_first);
#pragma unroll
                for (uint32_t h = 0; h < kRaw; ++h) loaded[s][h] = source[h];
            } else {
                T guarded[E];
#pragma unroll
                for (uint32_t j = 0; j < E; ++j) guarded[j] = chunk_first + j < end ? input[chunk_first + j] : T{};
                memcpy(loaded[s], guarded, sizeof(loaded[s]));
            }
        }
    }
#pragma unroll
    for (uint32_t s = 0; s < CHUNKS; ++s) {
#pragma unroll
        for (uint32_t h = 0; h < kRaw; ++h) s_data[kRaw * (s * THREADS + t) + h] = loaded[s][h];
    }

    // Dense rows that divide the tile: every tile starts a row, so no tile needs a prefix and no run is published or
    // awaited (otherwise the last tile of the last group waits about 5 us for runs nobody reads). The host decides the
    // same way whether to give the kernel a workspace.
    const bool lookback = RAGGED || columns.mod(TILE) != 0;
    // Thread 0 loads the last launch's epoch from the tile's own word here, once the loaded chunks are stashed and
    // their registers free, and uses it after the fold, which hides the load. (A ticket from one launch word instead,
    // 1475-1476, queued every block's atomic on one address: about 2 us at 4M.) With no lookback it leaves the empty
    // prefix and epoch 0, which no launch has: step 3 reads that, so no register holds `lookback` across the scan.
    uint32_t last_epoch = 0;
    if (t == 0) {
        if (lookback) {
            last_epoch = static_cast<uint32_t>(peek(ws.tile_runs + W * tile) >> 33);
        } else {
            s_epoch = 0;
            s_prefix = Run<A>{A{}, 0U, 0U};
        }
    }

    // The fold reads each chunk back from the stash, as step 5 does, so no register holds a chunk across the folds.
    // `slot` is the chunk's place in memory (its stash and its bitmap bits), `order` its place in scan order (its run
    // for step 2); element j of a chunk is its element `jo` in scan order. Forward they are the same.
    uint64_t band_col = col_first;  // the column of the thread's chunk in the current band
    for (uint32_t s = 0; s < CHUNKS; ++s) {
        const uint32_t slot = s * THREADS + t;
        const uint32_t order = REVERSE ? kSlots - 1 - slot : slot;
        const uint64_t chunk_first = thread_first + s * kBand;
        Raw raw[kRaw];
#pragma unroll
        for (uint32_t h = 0; h < kRaw; ++h) raw[h] = s_data[kRaw * slot + h];
        T values[E];
        uint32_t heads = unpackChunk<T, RAGGED, E, DIR>(
            raw, values, chunk_first, static_cast<uint64_t>(slot) * E, s_heads, cols, band_col, pattern);
        if constexpr (REVERSE) heads |= endHead<E>(chunk_first, end);
        if (order == 0) s_starts = (heads >> (REVERSE ? E - 1 : 0)) & 1U;
        A acc{};
#pragma unroll
        for (uint32_t jo = 0; jo < E; ++jo) {
            const uint32_t j = REVERSE ? E - 1 - jo : jo;
            const A x = element<T, OP, DIR>(values[j], chunk_first + j);
            acc = (jo == 0 || ((heads >> j) & 1U)) ? x : applyOp<A, OP>(acc, x);
        }
        s_value[order] = acc;
        s_head[order] = heads != 0 ? 1U : 0U;
        band_col += band_cols;
        if (band_col >= cols) band_col -= cols;
    }

    if (lookback && t == 0) s_epoch = nextEpoch(last_epoch);
    __syncthreads();

    // 2. Scan the slots' runs in slot order. Thread t takes the CHUNKS adjacent slots from CHUNKS x t and scans
    // them in registers: each is left holding its run from the thread's first slot, so the last holds the thread's
    // total. Each warp then scans its threads' totals in place, and warp 0 scans the warps' totals. That is
    // CHUNKS - 1 + 5 combines a thread; scanning every band's slots in the warp that folded them was 5 x CHUNKS.
    const uint32_t own = CHUNKS * t;             // the thread's first slot
    const uint32_t own_last = own + CHUNKS - 1;  // its last: the thread's total, then its run from the warp's start
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

    // 3-4. Warp 0: publish the tile's run (and, for the last tile of a group or supergroup, the group's run or the
    // supergroup's inclusive run), then fold the tile's prefix. Every lane loads one published run at once.
    const uint32_t epoch = warp == 0 ? s_epoch : 0U;  // 0: no tile publishes or waits (step 1)
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

        // Every word the tile may need, at all three levels, is loaded at once, so words already published cost one
        // trip to L2 together; only words not yet published are polled after (one trip each level was 1.5 to 2 us
        // under load: plans/Thor_Scan_Kernel_Log8_2026-11-18.md, 1470).
        const bool want_tiles = need_prefix || group_duty;
        const bool want_groups = need_prefix || super_duty;
        const bool want_previous = super > 0 && (need_prefix || super_duty);
        // Written so that with one word per run the addresses are the value scans' (1482).
        unsigned long long* const tile_slot = ws.tile_runs + W * (group * GROUP) + W * lane;
        unsigned long long* const group_slot = ws.group_runs + W * (super * SUPER) + W * lane;
        unsigned long long* const previous_slot = ws.super_runs + W * (super > 0 ? super - 1 : 0);
        Words<A> tile_words{};  // zero, as the value scans' words always were
        Words<A> group_words{};
        Words<A> previous_words{};
        if (want_tiles && lane < pos) tile_words = peekRun<A>(tile_slot);
        if (want_groups && lane < gpos) group_words = peekRun<A>(group_slot);
        if (want_previous && lane == 0) previous_words = peekRun<A>(previous_slot);

        // The group's tiles up to this one.
        Run<A> run = none;
        if (want_tiles && lane < pos) run = awaitRun<A>(tile_slot, epoch, tile_words);
        if (lane == pos) run = tile_run;
        warpScanRuns<A, OP>(run, lane, s_fold_value, s_fold_head, s_fold_valid);
        const Run<A> tiles_before =
            pos > 0 ? Run<A>{s_fold_value[pos - 1], s_fold_head[pos - 1], s_fold_valid[pos - 1]} : none;
        const Run<A> group_run{s_fold_value[pos], s_fold_head[pos], 1U};
        __syncwarp();
        if (group_duty && lane == 0) publishRun(ws.group_runs + W * group, group_run, epoch);

        // The supergroup's groups up to this one, when the prefix or the supergroup's inclusive run needs them.
        const bool need_groups = need_prefix && !(tiles_before.valid && tiles_before.head);
        Run<A> groups_before = none;
        Run<A> super_run = none;
        if (need_groups || super_duty) {
            run = none;
            if (lane < gpos) run = awaitRun<A>(group_slot, epoch, group_words);
            if (super_duty && lane == gpos) run = group_run;
            warpScanRuns<A, OP>(run, lane, s_fold_value, s_fold_head, s_fold_valid);
            if (gpos > 0) groups_before = Run<A>{s_fold_value[gpos - 1], s_fold_head[gpos - 1], s_fold_valid[gpos - 1]};
            if (super_duty) super_run = Run<A>{s_fold_value[gpos], s_fold_head[gpos], 1U};
        }

        if (lane == 0) {
            const bool need_previous = need_groups && !(groups_before.valid && groups_before.head);
            Run<A> previous = none;  // the previous supergroup's inclusive run
            if (super > 0 && (need_previous || (super_duty && super_run.head == 0U))) {
                previous = awaitRun<A>(previous_slot, epoch, previous_words);
            }
            if (super_duty) publishRun(ws.super_runs + W * super, combine<A, OP>(previous, super_run), epoch);
            Run<A> prefix = none;
            if (need_prefix) {
                const Run<A> base = need_previous ? combine<A, OP>(previous, groups_before) : groups_before;
                prefix = need_groups ? combine<A, OP>(base, tiles_before) : tiles_before;
            }
            s_prefix = prefix;
        }
    }
    __syncthreads();

    // 5. Band by band, scan the thread's chunk again from the stash, the elements before its first head taking the
    // tile's prefix and the slots before it, and write each 16 bytes of output (the chunk's 8 when it has fewer, FP32
    // to FP8) as soon as they are ready.
    const Run<A> prefix = s_prefix;
    if constexpr (RAGGED) end = s_end;  // the same value, read back
    constexpr uint32_t kStoreBytes = E * sizeof(O) < 16 ? E * sizeof(O) : 16;
    using Output = StoreWord<kStoreBytes>;
    constexpr uint32_t kPerWord = kStoreBytes / sizeof(O);  // outputs per store
    constexpr uint32_t kWords = E / kPerWord;
    static_assert(sizeof(Output) == kStoreBytes && kWords * kPerWord == E, "whole 16- or 8-byte stores per chunk");
    band_col = col_first;
    for (uint32_t s = 0; s < CHUNKS; ++s) {
        const uint32_t slot = s * THREADS + t;
        const uint32_t order = REVERSE ? kSlots - 1 - slot : slot;  // as in step 1
        const uint32_t owner = order / CHUNKS;  // the thread that scanned the slot in step 2
        const uint32_t owner_warp = owner >> 5;
        const uint64_t chunk_first = thread_first + s * kBand;
        const bool full = chunk_first + E <= end;
        // The slot's exclusive run within the tile: the warps before its owner's, then that warp's threads before
        // its owner (the run the last of them left in its last slot), then the owner's slots before it.
        Run<A> exclusive{A{}, 0U, 0U};
        if (owner_warp > 0) exclusive = Run<A>{s_warp_value[owner_warp - 1], s_warp_head[owner_warp - 1], 1U};
        if ((owner & 31U) > 0) {
            const uint32_t before = CHUNKS * owner - 1;
            exclusive = combine<A, OP>(exclusive, Run<A>{s_value[before], s_head[before], 1U});
        }
        if (order > CHUNKS * owner) {
            exclusive = combine<A, OP>(exclusive, Run<A>{s_value[order - 1], s_head[order - 1], 1U});
        }
        const Run<A> carry = combine<A, OP>(prefix, exclusive);
        bool open = carry.valid != 0U;  // no head in the chunk yet: the carry applies

        Raw raw[kRaw];
#pragma unroll
        for (uint32_t h = 0; h < kRaw; ++h) raw[h] = s_data[kRaw * slot + h];
        T values[E];
        uint32_t heads = unpackChunk<T, RAGGED, E, DIR>(
            raw, values, chunk_first, static_cast<uint64_t>(slot) * E, s_heads, cols, band_col, pattern);
        if constexpr (REVERSE) heads |= endHead<E>(chunk_first, end);
        A acc{};
        [[maybe_unused]] A last = carry.value;  // exclusive: the inclusive result of the element before
        // In scan order: a reverse scan takes the chunk's 16-byte words, and each word's elements, from the last.
#pragma unroll
        for (uint32_t wo = 0; wo < kWords; ++wo) {
            const uint32_t w = REVERSE ? kWords - 1 - wo : wo;
            O word[kPerWord];
            [[maybe_unused]] A keys[kPerWord];  // an arg scan's winning keys, decoded only when values are written
#pragma unroll
            for (uint32_t io = 0; io < kPerWord; ++io) {
                const uint32_t i = REVERSE ? kPerWord - 1 - io : io;
                const uint32_t j = w * kPerWord + i;
                const bool head = ((heads >> j) & 1U) != 0U;
                const A x = element<T, OP, DIR>(values[j], chunk_first + j);
                if (head) open = false;
                acc = (wo * kPerWord + io == 0 || head) ? x : applyOp<A, OP>(acc, x);
                if constexpr (MODE == SegmentedScanMode::Inclusive) {
                    if constexpr (isArg(OP)) {
                        keys[i] = open ? applyOp<A, OP>(carry.value, acc) : acc;
                        word[i] = result<O, OP, DIR>(keys[i]);
                    } else if constexpr (kSharedFloat<O>) {
                        word[i] = sharedFromFloat<O>(open ? applyOp<A, OP>(carry.value, acc) : acc, alternate_output);
                    } else {
                        word[i] = result<O, OP, DIR>(open ? applyOp<A, OP>(carry.value, acc) : acc);
                    }
                } else {
                    // A head takes the identity (an arg scan: the index 0xFFFFFFFF); the rest the result before.
                    if constexpr (isArg(OP)) {
                        keys[i] = last;
                        word[i] = head ? 0xFFFFFFFFU : result<O, OP, DIR>(last);
                    } else if constexpr (kSharedFloat<O>) {
                        constexpr float kIdentity = OP == SegmentedScanOp::Sum ? 0.0f : 1.0f;
                        word[i] = sharedFromFloat<O>(head ? kIdentity : last, alternate_output);
                    } else {
                        word[i] = head ? identityOf<O, OP>() : result<O, OP, DIR>(last);
                    }
                    last = open ? applyOp<A, OP>(carry.value, acc) : acc;
                }
            }
            if (full) {
                Output bits;
                memcpy(&bits, word, sizeof(bits));
                reinterpret_cast<Output*>(output + chunk_first)[w] = bits;
            } else {
#pragma unroll
                for (uint32_t i = 0; i < kPerWord; ++i) {
                    if (chunk_first + w * kPerWord + i < end) output[chunk_first + w * kPerWord + i] = word[i];
                }
            }
            if constexpr (isArg(OP)) {
                if (arg_values != nullptr) {  // the same for every thread; an index-only scan decodes nothing
                    T won[kPerWord];
#pragma unroll
                    for (uint32_t i = 0; i < kPerWord; ++i) won[i] = valueOf<T, OP>(keys[i]);
                    if constexpr (MODE == SegmentedScanMode::Exclusive) {
                        constexpr SegmentedScanOp kExtreme =
                            OP == SegmentedScanOp::ArgMax ? SegmentedScanOp::Max : SegmentedScanOp::Min;
#pragma unroll
                        for (uint32_t i = 0; i < kPerWord; ++i) {
                            if (((heads >> (w * kPerWord + i)) & 1U) != 0U) won[i] = identityOf<T, kExtreme>();
                        }
                    }
                    using Stored = StoreWord<kPerWord * sizeof(T)>;
                    if (full) {
                        Stored bits;
                        memcpy(&bits, won, sizeof(bits));
                        reinterpret_cast<Stored*>(arg_values + chunk_first)[w] = bits;
                    } else {
#pragma unroll
                        for (uint32_t i = 0; i < kPerWord; ++i) {
                            const uint64_t at = chunk_first + w * kPerWord + i;
                            if (at < end) arg_values[at] = won[i];
                        }
                    }
                }
            }
        }
        band_col += band_cols;
        if (band_col >= cols) band_col -= cols;
    }
}

#if defined(__CUDACC__)
template <typename T, typename O, SegmentedScanOp OP, SegmentedScanMode MODE, SegmentedScanDirection DIR, typename OFF>
void launch(const Launch& args, cudaStream_t stream) {
    const unsigned blocks = static_cast<unsigned>(args.workspace.num_tiles);
    segmentedScanKernel<T, O, OP, MODE, DIR, OFF><<<blocks, kThreads, 0, stream>>>(
        static_cast<const T*>(args.input),
        static_cast<O*>(args.output),
        args.n,
        args.columns,
        static_cast<const OFF*>(args.offsets),
        args.rows,
        args.workspace,
        static_cast<T*>(args.arg_values),
        args.alternate_output);
}

// The kernels of one (input, output, op) in one mode and direction: dense rows and ragged rows with UINT32 or UINT64
// offsets.
#define THOR_SEGMENTED_SCAN_INSTANTIATE_VARIANT(T, O, OP, MODE, DIR)                                                 \
    template void launch<T, O, SegmentedScanOp::OP, SegmentedScanMode::MODE, SegmentedScanDirection::DIR, void>(     \
        const Launch&, cudaStream_t);                                                                                \
    template void launch<T, O, SegmentedScanOp::OP, SegmentedScanMode::MODE, SegmentedScanDirection::DIR, uint32_t>( \
        const Launch&, cudaStream_t);                                                                                \
    template void launch<T, O, SegmentedScanOp::OP, SegmentedScanMode::MODE, SegmentedScanDirection::DIR, uint64_t>( \
        const Launch&, cudaStream_t)

// The kernels of one (input, output, op): both modes, both directions.
#define THOR_SEGMENTED_SCAN_INSTANTIATE(T, O, OP)                          \
    THOR_SEGMENTED_SCAN_INSTANTIATE_VARIANT(T, O, OP, Inclusive, Forward); \
    THOR_SEGMENTED_SCAN_INSTANTIATE_VARIANT(T, O, OP, Exclusive, Forward); \
    THOR_SEGMENTED_SCAN_INSTANTIATE_VARIANT(T, O, OP, Inclusive, Reverse); \
    THOR_SEGMENTED_SCAN_INSTANTIATE_VARIANT(T, O, OP, Exclusive, Reverse)
#endif

}  // namespace ThorImplementation::SegmentedScanKernels
