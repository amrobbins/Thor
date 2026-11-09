#pragma once

// 1285 (phase 4, plans/Thor_Ragged_Reduction_Plan_2026-11-06.md): the ragged RK main kernel's geometry, host / device,
// so the host test (test/Utilities/TensorOperations/RaggedKRWalkTest.cpp) emulates exactly the kernel's tiles
// (RaggedRKKernels.cuh). Ragged RK takes rows of R = D x element bytes >= 16. The walk over the CTA's merge-path range
// is the KR kernels' (RaggedKRWalk.h: the ends ring and its prefetch, the tile end, the bitmap, each end's kind); what
// differs is the values:
//   - a lane owns a 16-byte chunk of a K tile of T <= 512 bytes: P = ceil(T / 16) chunks per row, W = 32 / P workers,
//     lane = worker x P + chunk (the lanes from W x P on are idle);
//   - rows of up to the tile cap are one K tile (T = R); wider rows are cut into ktiles = ceil(R / cap) tiles of
//     T = 16 x ceil(ceil(R / ktiles) / 16) bytes (the last one shorter), so a K-tiled row has W <= 16;
//   - a worker folds raggedRKRowsPerWorker consecutive rows per tile (RAGGED_RK_ROWS = 2; 1288: 4 on narrow rows, R <=
//     32 bytes in one K tile, W >= 16), so a tile is that x W rows; its bytes are staged in shared memory as the
//     16-byte chunks covering them, in one run when T = R (the rows are contiguous) and one run per row when K-tiled (a
//     row's run at row x (P + 1) x 16 bytes);
//   - the persistent grid is ranges x ktiles CTAs: CTA c takes K tile c mod ktiles of path range c / ktiles; the
//     boundary records hold D head and D tail values per range (raggedRecordLayout(ranges, D)).
#include "Utilities/TensorOperations/Cub/RaggedKRWalk.h"

#include <cstdint>

namespace ThorImplementation::CubReductionInternal::RaggedMergePath {

// 1290: per-CTA timestamps of the RK main kernel's start-up and tail, written when RaggedRKParams::probe is set
// (raggedRKSetProbe; the GPU test CubRaggedKR.DISABLED_RkStartupTailProbe). 1291: off (the kernels are 1289's); define
// THOR_RAGGED_RK_PROBE 1 to measure again.
#ifndef THOR_RAGGED_RK_PROBE
#define THOR_RAGGED_RK_PROBE 0
#endif
// The probe's record per CTA: entry globaltimer, then clock64 at entry, after the search, at the first tile, after the
// last tile, at exit; the exit globaltimer; the SM and the CTA's tiles (smid << 32 | tiles).
inline constexpr uint32_t RAGGED_RK_PROBE_WORDS = 8;

inline constexpr uint32_t RAGGED_RK_CHUNK_BYTES = 16;     // a lane's bytes of one row per tile
inline constexpr uint32_t RAGGED_RK_ROWS = 2;             // rows per worker per tile
// 1288: rows per worker on narrow rows (one K tile, W >= 16 workers, so R <= 32 bytes): the walk, the bits pass, the
// store pass's setup and the scan across W workers (4-5 rounds of 16 values) once per 4 rows instead of 2. 1287 took 4
// rows on every contiguous row: D 17 gained 1.3-1.6x, D 16 2-4%, but D 64-256 at mean lengths 16-256 lost 3-5% and the
// one kernel's rows-2 paths (K-tiled) lost 3% too; these kernels are their own instantiation (RaggedRKKernels.cuh).
inline constexpr uint32_t RAGGED_RK_ROWS_NARROW = 4;
inline constexpr uint32_t RAGGED_RK_NARROW_MIN_WORKERS = 16;
inline constexpr uint32_t RAGGED_RK_MAX_TILE_BYTES = 512;  // K tile cap: 32 lanes x 16 bytes
inline constexpr uint32_t RAGGED_RK_MIN_TILE_BYTES = 32;   // so a K-tiled row has P >= 2 chunks and W <= 16 workers
inline constexpr uint32_t RAGGED_RK_MIN_ROW_BYTES = 16;    // below: ragged KR
// The staged chunks of one tile: at most 65 in one run (T = R: 15 + 2 x W x R <= 1039 bytes) and 2 x W x (P + 1) <= 96
// when K-tiled (W <= 16); plus 16 bytes, so a lane's unaligned 16-byte read (5 words) never leaves the buffer.
inline constexpr uint32_t RAGGED_RK_STAGED_CHUNKS = 96;
inline constexpr uint32_t RAGGED_RK_SPAN_BYTES = RAGGED_RK_CHUNK_BYTES * (RAGGED_RK_STAGED_CHUNKS + 1);
// 1288: narrow tiles (4 rows per worker, W x R <= 512): at most 129 chunks in one run (15 + 4 x W x R <= 2063 bytes),
// the last unaligned read (20 bytes from its offset) ending by 2067; a buffer of 130 chunks (2080 bytes). Two buffers,
// the ring, the kinds and the bits: about 4.9 KB per CTA, 16 CTAs per SM within 100 KB.
inline constexpr uint32_t RAGGED_RK_NARROW_STAGED_CHUNKS = 129;
inline constexpr uint32_t RAGGED_RK_NARROW_SPAN_BYTES = RAGGED_RK_CHUNK_BYTES * (RAGGED_RK_NARROW_STAGED_CHUNKS + 1);

// The span buffer's bytes and its most staged chunks for tiles of `rows` per worker.
THOR_RAGGED_HD constexpr uint32_t raggedRKSpanBytesFor(uint32_t rows) {
    return rows == RAGGED_RK_ROWS_NARROW ? RAGGED_RK_NARROW_SPAN_BYTES : RAGGED_RK_SPAN_BYTES;
}
THOR_RAGGED_HD constexpr uint32_t raggedRKStagedChunksFor(uint32_t rows) {
    return rows == RAGGED_RK_ROWS_NARROW ? RAGGED_RK_NARROW_STAGED_CHUNKS : RAGGED_RK_STAGED_CHUNKS;
}
// The largest row: 1329, 2 GiB (the kernels' 32-bit row bytes and K tile offsets; rows over 512 bytes are K tiled).
inline constexpr uint64_t RAGGED_RK_MAX_ROW_BYTES = uint64_t{1} << 31;

struct RaggedRKGeometry {
    uint32_t ktiles = 1;      // K tiles per row
    uint32_t tile_bytes = 0;  // T: a K tile's bytes (the last tile: R - (ktiles - 1) x T)
    uint32_t chunks = 1;      // P: 16-byte chunks per K tile row
    uint32_t workers = 1;     // W = 32 / P
    bool contiguous = true;   // ktiles == 1: a tile's rows are one run of bytes
};

THOR_RAGGED_HD uint32_t raggedRKCeilDiv(uint64_t a, uint64_t b) { return static_cast<uint32_t>((a + b - 1) / b); }

// The geometry of rows of `row_bytes` (>= 16) under a K tile cap (a power of two, 32 to 512).
THOR_RAGGED_HD RaggedRKGeometry raggedRKGeometry(uint64_t row_bytes, uint32_t tile_cap) {
    RaggedRKGeometry g;
    if (row_bytes <= tile_cap) {
        g.ktiles = 1;
        g.tile_bytes = static_cast<uint32_t>(row_bytes);
    } else {
        g.ktiles = raggedRKCeilDiv(row_bytes, tile_cap);
        const uint32_t share = raggedRKCeilDiv(row_bytes, g.ktiles);
        g.tile_bytes = RAGGED_RK_CHUNK_BYTES * raggedRKCeilDiv(share, RAGGED_RK_CHUNK_BYTES);
    }
    g.chunks = raggedRKCeilDiv(g.tile_bytes, RAGGED_RK_CHUNK_BYTES);
    g.workers = 32U / g.chunks;
    g.contiguous = g.ktiles == 1;
    return g;
}

// K tile k's bytes.
THOR_RAGGED_HD uint32_t raggedRKTileBytesOf(const RaggedRKGeometry& g, uint64_t row_bytes, uint32_t k) {
    const uint64_t k0 = uint64_t{k} * g.tile_bytes;
    return static_cast<uint32_t>(row_bytes - k0 < g.tile_bytes ? row_bytes - k0 : g.tile_bytes);
}

// Rows per worker per tile: 4 on narrow rows (one K tile, W >= 16), else 2 (1288).
THOR_RAGGED_HD uint32_t raggedRKRowsPerWorker(const RaggedRKGeometry& g) {
    return g.contiguous && g.workers >= RAGGED_RK_NARROW_MIN_WORKERS ? RAGGED_RK_ROWS_NARROW : RAGGED_RK_ROWS;
}

// A tile's rows: raggedRKRowsPerWorker x W.
THOR_RAGGED_HD uint32_t raggedRKTileRows(const RaggedRKGeometry& g) { return raggedRKRowsPerWorker(g) * g.workers; }

// K-tiled: one row's staged run (its bytes plus the misalignment, in chunks), at row x raggedRKSlotBytes.
THOR_RAGGED_HD uint32_t raggedRKSlotBytes(const RaggedRKGeometry& g) { return RAGGED_RK_CHUNK_BYTES * (g.chunks + 1); }

// The staged byte of row j's (of the tile) chunk `chunk`, from the tile's first byte's misalignment `tile_shift`
// (contiguous: the run's; K-tiled: row 0's run, each later row's own misalignment from it).
THOR_RAGGED_HD uint32_t raggedRKStagedOffset(const RaggedRKGeometry& g, uint64_t row_bytes, uint32_t tile_shift,
                                             uint32_t j, uint32_t chunk) {
    if (g.contiguous) return tile_shift + j * static_cast<uint32_t>(row_bytes) + RAGGED_RK_CHUNK_BYTES * chunk;
    const uint32_t shift = (tile_shift + j * static_cast<uint32_t>(row_bytes & 15U)) & 15U;
    return j * raggedRKSlotBytes(g) + shift + RAGGED_RK_CHUNK_BYTES * chunk;
}

// The auto K tile cap (RaggedRKShape::tile_bytes 0): the records hold ranges x D x 8 bytes (FP32 head and tail), about
// ctas x T / element bytes x 8; the largest cap of 512 / 256 / 128 / 64 / 32 keeping that within 1/16 of the input's
// bytes, else 32. 1314: `record_bytes` is a column's head and tail bytes per range (VALUE 8; ARG 16, two 64-bit words).
THOR_RAGGED_HD uint32_t raggedRKAutoTileCap(uint64_t input_bytes, uint32_t ctas, uint32_t element_bytes,
                                            uint32_t record_bytes = 8) {
    for (uint32_t cap = RAGGED_RK_MAX_TILE_BYTES; cap > RAGGED_RK_MIN_TILE_BYTES; cap >>= 1) {
        if (uint64_t{ctas} * (cap / element_bytes) * record_bytes * 16 <= input_bytes) return cap;
    }
    return RAGGED_RK_MIN_TILE_BYTES;
}

// Path ranges for a CTA budget: ctas / ktiles, at least 1, at most `max_ranges` (the finisher's limit).
THOR_RAGGED_HD uint32_t raggedRKRanges(uint32_t ctas, uint32_t ktiles, uint32_t max_ranges) {
    const uint32_t r = ctas / ktiles;
    return r == 0 ? 1U : (r > max_ranges ? max_ranges : r);
}

}  // namespace ThorImplementation::CubReductionInternal::RaggedMergePath
