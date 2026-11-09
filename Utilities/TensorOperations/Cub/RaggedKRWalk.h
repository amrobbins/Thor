#pragma once

// 1265 (1268, 1270): the ragged KR main kernel's per-tile logic, as host / device functions over accessors, so the host
// test (test/Utilities/TensorOperations/RaggedKRWalkTest.cpp) runs the same code the kernel runs
// (RaggedKRKernels.cuh; 1265-1283: CubRaggedKRFp8.cu; plans/Thor_Ragged_Reduction_Plan_2026-11-06.md, phase 2).
//
// 1270: one warp per CTA, row-major tiles. A CTA's merge-path range (RaggedMergePath.h, weighted) is walked in tiles;
// everything inside a CTA is 32-bit and relative to the CTA's first coordinate (the launcher keeps a CTA below 2^31
// events). Per tile:
//   1. the ends ring holds up to `Ring` of the CTA's segment ends (relative rows), loaded in bulk and topped up by a
//      prefetch during every tile;
//   2. the tile end (raggedTileEnd): at most the rows of one aligned span of 32 x Rows bytes, and never past the last
//      loaded end, so a tile's ends always fit the ring and the ring's size does not bound the tile's rows;
//   3. every end of the tile whose segment has rows in it sets the bit of its last row in a bitmap (raggedEndKind);
//   4. each lane folds its Rows rows (one aligned chunk, 1271: in pieces of 16, a tree fold where no end falls),
//      writing the value of every segment ending in its rows to a slot ranked by its bit as it goes (raggedLaneEmit;
//      1272: one pass), a warp-wide segmented scan carries the open run from lane to lane and from the tile before,
//      and each lane completes its first written value with the run arriving at it (raggedLaneFirstValue);
//   5. one lane per end then reads its segment's value (the ranked slot, the tile's carry-in, or the identity for an
//      empty segment), finalizes and stores it, in segment order.
// A segment's value is the fold of its rows in order, the carry of the tiles before (and, across CTAs, the finisher)
// first: a fixed order, so results repeat bit for bit on one grid.
#include "Utilities/TensorOperations/Cub/RaggedMergePath.h"

#include <cstdint>

// Device passes only: nvcc's host pass hands host / device bodies to the host compiler, which rejects the pragma.
#if defined(__CUDA_ARCH__)
#define THOR_RAGGED_UNROLL _Pragma("unroll")
#else
#define THOR_RAGGED_UNROLL
#endif

namespace ThorImplementation::CubReductionInternal::RaggedMergePath {

// One element of the segmented scan: `reset` when a segment ended in it, `value` the run after its last end (or all
// of it when nothing ended). a then b: b's value alone when b reset, else a's run continued by b's.
template <typename T>
struct Carry {
    bool reset = false;
    T value{};
};

THOR_RAGGED_EXEC_CHECK_DISABLE
template <typename T, typename Combine>
THOR_RAGGED_HD Carry<T> raggedCarryCombine(const Carry<T>& a, const Carry<T>& b, const Combine& combine) {
    return Carry<T>{a.reset || b.reset, b.reset ? b.value : combine(a.value, b.value)};
}

// A coordinate relative to the CTA's first one.
struct Coordinate32 {
    uint32_t segment = 0;
    uint32_t row = 0;
};

// How many ends to load after the `held` loaded ones: as many as the ring's free slots take (a ring of `capacity`
// slots, a power of two, holds the window [current segment, current segment + capacity)), at most `ends_left`.
THOR_RAGGED_HD uint32_t raggedEndsToLoad(uint32_t held, uint32_t capacity, uint32_t ends_left) {
    const uint32_t free_slots = capacity - held;
    return free_slots < ends_left ? free_slots : ends_left;
}

// The ring slot of relative segment `segment` in a ring of mask + 1 slots.
THOR_RAGGED_HD uint32_t raggedRingSlot(uint32_t segment, uint32_t mask) { return segment & mask; }

// The tile from (x0, y0) with `held` ends loaded (ends x0 .. x0 + held - 1, the last at row `last_loaded_end`;
// `all_loaded` when they are all the CTA's remaining ends), rows up to `row_limit` (the tile's aligned span, clamped
// to the CTA's rows), and `ends_within_limit` of the loaded ends at or before row_limit:
//   - when an end that is not loaded could come before row_limit (the last loaded one is before it), the tile stops at
//     the last loaded end: (x0 + held, last_loaded_end), a path coordinate (that end consumed, rows up to it);
//   - otherwise at row_limit, with every loaded end at or before it consumed (ends at a row come before the row).
// Needs held > 0 unless all_loaded. Either way the tile advances, and its ends are loaded ones.
THOR_RAGGED_HD Coordinate32 raggedTileEnd(uint32_t x0, uint32_t held, bool all_loaded, uint32_t last_loaded_end,
                                          uint32_t row_limit, uint32_t ends_within_limit) {
    if (!all_loaded && last_loaded_end < row_limit) return Coordinate32{x0 + held, last_loaded_end};
    return Coordinate32{x0 + ends_within_limit, row_limit};
}

// What an end of a tile starting at row y0 stores: its segment has no rows (Empty: the identity), its rows ended
// before the tile (CarryIn: the open run carried into the tile; only the tile's first end can be one), or its last
// row is in the tile (Ranked: the value the lanes wrote for its bit).
enum class RaggedEndKind : uint8_t { Empty, CarryIn, Ranked };

THOR_RAGGED_HD RaggedEndKind raggedEndKind(bool has_rows, uint32_t end_row, uint32_t y0) {
    if (!has_rows) return RaggedEndKind::Empty;
    return end_row == y0 ? RaggedEndKind::CarryIn : RaggedEndKind::Ranked;
}

// 1275: an end's kind from its row and the end before it: the CTA's first end has rows when its segment does (from the
// search's bounds), any other when it ends after the end before it.
THOR_RAGGED_HD RaggedEndKind raggedEndKindAfter(bool cta_first_end, bool first_has_rows, uint32_t end_row,
                                                uint32_t previous_end, uint32_t y0) {
    return raggedEndKind(cta_first_end ? first_has_rows : end_row > previous_end, end_row, y0);
}

// 1275: the value slot of a Ranked end. The tile's bits are set one per Ranked end, in end order, and the lanes write
// the values by the rank of their bit, so a Ranked end's slot is the number of Ranked ends before it in the tile:
// `earlier` from the rounds of ends before this one, then the lanes below `lane` in this round's ballot of Ranked
// ends. 1270-1274 found it from the bit instead: two shuffles from the bit's lane, then the bits below it there.
THOR_RAGGED_HD uint32_t raggedRankedSlot(uint32_t earlier, uint32_t ranked_ballot, uint32_t lane) {
    const uint32_t below = ranked_ballot & ((1U << lane) - 1U);  // lane < 32
#if defined(__CUDA_ARCH__)
    return earlier + static_cast<uint32_t>(__popc(below));
#else
    return earlier + static_cast<uint32_t>(__builtin_popcount(below));
#endif
}

// 1275: round 0's facts, kept in one register from the bits pass to the store pass: the slot (below 2^16) and the kind.
THOR_RAGGED_HD uint32_t raggedPackEndFacts(RaggedEndKind kind, uint32_t slot) {
    return slot | (static_cast<uint32_t>(kind) << 16);
}
THOR_RAGGED_HD RaggedEndKind raggedPackedKind(uint32_t packed) { return static_cast<RaggedEndKind>(packed >> 16); }
THOR_RAGGED_HD uint32_t raggedPackedSlot(uint32_t packed) { return packed & 0xffffU; }

// One lane's Rows rows: `value(k)` (the identity for rows outside the tile), bit k of `bits` set when a segment ends
// after row k. Returns the scan element: reset when a segment ended here, value the run after the last end. Written
// branch-free (selects), so lanes with ends do not diverge from lanes without.
THOR_RAGGED_EXEC_CHECK_DISABLE
template <uint32_t Rows, typename T, typename Combine, typename Value>
THOR_RAGGED_HD Carry<T> raggedLaneRun(uint32_t bits, T identity, const Combine& combine, const Value& value) {
    T acc = identity;
THOR_RAGGED_UNROLL
    for (uint32_t k = 0; k < Rows; ++k) {
        const T next = combine(acc, value(k));
        acc = ((bits >> k) & 1U) ? identity : next;
    }
    return Carry<T>{bits != 0, acc};
}

// 1271: the fold of Rows values with no end among them, as a balanced tree (fewer dependent steps than the chain; a
// fixed order, so results still repeat bit for bit).
THOR_RAGGED_EXEC_CHECK_DISABLE
template <uint32_t Rows, typename T, typename Combine, typename Value>
THOR_RAGGED_HD T raggedTreeFold(const Combine& combine, const Value& value) {
    static_assert(Rows == 1 || Rows == 2 || Rows == 4 || Rows == 8 || Rows == 16);
    if constexpr (Rows == 1) {
        return value(0U);
    } else {
        constexpr uint32_t Half = Rows / 2;
        const T low = raggedTreeFold<Half, T>(combine, value);
        const T high = raggedTreeFold<Half, T>(combine, [&](uint32_t k) { return value(k + Half); });
        return combine(low, high);
    }
}

// The value of every segment ending in the lane's rows, in order: `emit(k, value)` for each set bit k, the first
// continuing `carry_in`. Returns the run after the last end (all of carry_in and the rows when no bit is set).
// 1272: the kernel calls it once per piece in its first pass, carry_in the lane's own run so far (the identity at the
// lane's first row), and completes the lane's first emitted value with the scan's carry-in afterwards
// (raggedLaneFirstValue): one fold of the rows, not a run and then an emit.
THOR_RAGGED_EXEC_CHECK_DISABLE
template <uint32_t Rows, typename T, typename Combine, typename Value, typename Emit>
THOR_RAGGED_HD T raggedLaneEmit(uint32_t bits, T carry_in, T identity, const Combine& combine, const Value& value,
                                const Emit& emit) {
    T acc = carry_in;
THOR_RAGGED_UNROLL
    for (uint32_t k = 0; k < Rows; ++k) {
        acc = combine(acc, value(k));
        if ((bits >> k) & 1U) {
            emit(k, acc);
            acc = identity;
        }
    }
    return acc;
}

// 1272: a lane's first emitted value covers its rows from the lane's first one; the segment's value puts the open run
// arriving at the lane (the warp scan's exclusive value, the tile's carry first) before it.
THOR_RAGGED_EXEC_CHECK_DISABLE
template <typename T, typename Combine>
THOR_RAGGED_HD T raggedLaneFirstValue(T arriving, T emitted, const Combine& combine) {
    return combine(arriving, emitted);
}

// 1272: where 16-byte chunk `chunk` of a tile's span sits in shared memory (chunk c holds lane c / P's piece c % P,
// P = Rows / 16): piece-major, so each cp.async instruction copies Lanes consecutive chunks (coalesced) and each
// piece's reads are Lanes consecutive 16-byte slots (no bank conflicts). The identity for P = 1.
template <uint32_t Rows, uint32_t Lanes>
THOR_RAGGED_HD uint32_t raggedPieceSlot(uint32_t lane, uint32_t piece) {
    return 16U * (piece * Lanes + lane);
}

template <uint32_t Rows, uint32_t Lanes>
THOR_RAGGED_HD uint32_t raggedChunkSlot(uint32_t chunk) {
    constexpr uint32_t Pieces = Rows / 16;
    return raggedPieceSlot<Rows, Lanes>(chunk / Pieces, chunk % Pieces);
}

// 1273: the KR boundary record, 16 bytes per CTA: {head segment or NO_SEGMENT, head value, tail value}
// (1280: as three arrays, raggedRecordLayout below, with D head and tail values), and no tail segment ids. CTA c's tail, when it has one, is always the
// segment CTA c + 1 starts in, and a CTA without a tail records the identity. So the finisher's scan over the tails, in
// CTA order, need only restart at each head's CTA: a CTA with ends but no head starts at its first segment's first
// row, so the CTAs before it back to the last head contribute only identities. Each head's output is the run arriving
// at its CTA, then the head value.
THOR_RAGGED_HD bool raggedRecordHasHead(uint64_t key) { return key != NO_SEGMENT; }

// The finisher's scan element of a record: reset at a head's CTA, the tail value (the identity without a tail).
template <typename T>
THOR_RAGGED_HD Carry<T> raggedRecordElement(uint64_t key, T tail_value) {
    return Carry<T>{raggedRecordHasHead(key), tail_value};
}

// 1280 (phase 3, FP8 at D = 2..15, rows under 16 bytes): the components kernel (1284: raggedKRComponentsKernel). A lane
// is one component of one worker: lane = worker * D + component, `raggedComponentWorkers(D)` workers (32 / D; the
// lanes from workers * D on are idle). A worker owns RAGGED_KR_COMPONENT_ROWS consecutive rows of the tile, so the
// tile's ends, bitmap, ranks and kinds are the D = 1 kernel's, per row; only the values are D wide (each lane folds its
// component of its worker's rows, the scan carries each component across workers, a slot holds D values). One kernel
// for every D, with no per-D accumulators: D is a runtime operand.
inline constexpr uint32_t RAGGED_KR_COMPONENT_ROWS = 16;          // FP8 rows per worker per tile (1284: 16 bytes)
inline constexpr uint32_t RAGGED_KR_COMPONENT_VALUE_SLOTS = 256;  // FP32 value slots per tile (slot * D + component)
inline constexpr uint32_t RAGGED_KR_MAX_COMPONENTS = 15;          // KR: rows under 16 bytes (FP8; any dtype's bound)

// 1284 (phase 3b, FP16 / BF16 / FP32): a lane (D = 1) or a worker (D >= 2) owns 16 bytes of rows per tile in every
// dtype: 16 / element bytes rows (16 FP8, 8 FP16 / BF16, 4 FP32), so a tile stays at most 512 bytes and the staging,
// the ends and the bitmap are the FP8 kernels' per row. D is at most 15 / element bytes (rows under 16 bytes).
THOR_RAGGED_HD uint32_t raggedKRRowsPerLane(uint32_t element_bytes) { return 16U / element_bytes; }
THOR_RAGGED_HD uint32_t raggedKRMaxComponents(uint32_t element_bytes) { return 15U / element_bytes; }

THOR_RAGGED_HD uint32_t raggedComponentWorkers(uint32_t components) { return 32U / components; }

// A tile's rows: one row run per worker (at most 256 rows and 512 bytes at D >= 2).
THOR_RAGGED_HD uint32_t raggedComponentTileRows(uint32_t components, uint32_t element_bytes = 1) {
    return raggedKRRowsPerLane(element_bytes) * raggedComponentWorkers(components);
}

// The most ends a tile may consume: its ranked ends' D values fit the value slots, and its ends the ring.
THOR_RAGGED_HD uint32_t raggedComponentEndsCap(uint32_t components, uint32_t ring) {
    const uint32_t by_slots = RAGGED_KR_COMPONENT_VALUE_SLOTS / components;
    return by_slots < ring ? by_slots : ring;
}

// 1280: the loaded ends a tile may consume, capped at `cap` (raggedTileEnd then stops at the cap-th end, or at the row
// limit with at most cap ends consumed): with more than `cap` ends held the tile treats only the first `cap` as loaded.
// The kernel passes the cap-th end as raggedTileEnd's last loaded end.
struct RaggedTileEnds {
    uint32_t held = 0;
    bool all_loaded = false;
    uint32_t ends_within = 0;
};

THOR_RAGGED_HD RaggedTileEnds raggedCapTileEnds(uint32_t held, bool all_loaded, uint32_t ends_within, uint32_t cap) {
    return RaggedTileEnds{held < cap ? held : cap, all_loaded && held <= cap, ends_within < cap ? ends_within : cap};
}

// 1280: an end's facts in 16 bits (slot < 256, kind), kept in shared memory from the bits pass for the store pass.
THOR_RAGGED_HD uint32_t raggedPackEndFacts16(RaggedEndKind kind, uint32_t slot) {
    return slot | (static_cast<uint32_t>(kind) << 8);
}
THOR_RAGGED_HD RaggedEndKind raggedPacked16Kind(uint32_t packed) { return static_cast<RaggedEndKind>(packed >> 8); }
THOR_RAGGED_HD uint32_t raggedPacked16Slot(uint32_t packed) { return packed & 0xffU; }

// 1280: the VALUE boundary records, as arrays (1273-1279: one 16-byte record per CTA): each CTA's key (its head
// segment or NO_SEGMENT, 8 bytes), then component c's head values for CTAs 0 .. grid - 1, then the tail values the same
// way. At D = 1 the same 16 bytes per CTA. The finisher runs one CTA per component, and its loads are coalesced.
struct RaggedRecordLayout {
    uint64_t keys = 0, heads = 0, tails = 0, bytes = 0;  // byte offsets of the three arrays, and the total
};

THOR_RAGGED_HD uint64_t raggedRoundUp16(uint64_t bytes) { return (bytes + 15U) & ~uint64_t{15}; }

THOR_RAGGED_HD RaggedRecordLayout raggedRecordLayout(uint32_t grid, uint32_t components) {
    RaggedRecordLayout layout;
    const uint64_t values = raggedRoundUp16(uint64_t{4} * grid * components);
    layout.keys = 0;
    layout.heads = raggedRoundUp16(uint64_t{8} * grid);
    layout.tails = layout.heads + values;
    layout.bytes = layout.tails + values;
    return layout;
}

// The value slot of component c of CTA `cta` in the heads or tails array.
THOR_RAGGED_HD uint64_t raggedRecordValueIndex(uint32_t grid, uint32_t cta, uint32_t component) {
    return uint64_t{component} * grid + cta;
}

// Set bits of `bits` below bit k.
THOR_RAGGED_HD uint32_t raggedBitsBelow(uint32_t bits, uint32_t k) {
    const uint32_t below = k >= 32 ? bits : (bits & ((1U << k) - 1U));
#if defined(__CUDA_ARCH__)
    return static_cast<uint32_t>(__popc(below));
#else
    return static_cast<uint32_t>(__builtin_popcount(below));
#endif
}

}  // namespace ThorImplementation::CubReductionInternal::RaggedMergePath
