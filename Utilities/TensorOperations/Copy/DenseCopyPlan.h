#pragma once

// 1388: the plan of one copy of a strided span into a dense tensor (StridedCopy.h). CUDA-free, so a stamp can hold one
// and the planning can be tested on the host.
//
// The planner reads the span's canonical layout: extent-one axes dropped, and each axis merged into the next inner one
// when its stride is that axis's extent times its stride. Rows are the innermost axis when its stride is 1 (p words),
// else single words (p = 1); the other canonical axes are the outer axes. Words are the element widened as far as the
// alignment allows: the widest of 16, 8, 4, 2 bytes (a multiple of the element) that divides the row bytes, every
// outer stride in bytes, and both addresses.
// - Memcpy: a dense span (one canonical axis, stride 1), or a single element.
// - Transpose: rows narrower than a 32-byte sector (1389f: or of two to four 16-byte words in a copy of 4 MiB or
//   more), and an outer axis x whose source stride is p (its rows are adjacent in the source) other than the
//   innermost outer axis v (whose rows are adjacent in the destination); every other outer axis is a batch axis. A
//   tile of tile_x x rows by tile_v v rows goes through shared memory, read in
//   source order, written in destination order, each thread holding 4 to 16 loads in flight. 1389: the tile is
//   rectangular. Its side s is the square root of the vectors over 512 tiles, between the side of a 1,024-word
//   square tile and the side whose tile row is 128 bytes (128 rows of 1 byte, 64 of 2, 43 of 3, else 32); an x or v
//   shorter than s is the tile's whole side and the other side grows to keep s^2 vectors, within 32 KB of shared
//   memory. 1389d: the smallest side was 32 vectors whatever their width, so a copy of under 524,288 vectors of p
//   words had tiles of 1,024 p words, and the payload rotations of 0.1 to 1 MiB with 5 or 7 words a row took 3.9
//   to 6.6 us against 1.8 to 2.7 for one-word rows of that size (2- and 4-byte words). An x by v plane under 256
//   vectors is a row copy instead: it is a few sectors, local to the row copy's reads. 1389a: so are more than
//   2,048 tiles of under 640 words each (a small x by v plane under many batch elements). 1389f: rows of 32, 48
//   or 64 bytes in 16-byte words are transposed too when the copy is 4 MiB or more. As row copies they were at
//   0.57 to 0.96 of the copy curve from 98 MiB (1389e audit: 32-byte rows 0.57 to 0.94, 48-byte 0.65, 64-byte
//   0.85 to 0.96) and at 0.84 to 0.94 at 27 to 44 MiB when x and v are the whole copy, where the same layouts at
//   a narrower dtype (16- and 24-byte rows, transposed) were at 0.90 to 1.0. Under 4 MiB the row copy was level
//   to 12% faster than those transposes. Rows of 96 bytes and more were at 0.94 or better as row copies and stay
//   row copies; so do rows of 32 to 64 bytes in narrower words (unaligned, or not a multiple of 16 bytes), which
//   no cell measured as a transpose. 1389i: when x is 8 v or more (long source rows) and there is more than one tile
//   along x and along v, the tiles of a plane go along v (tile_v_first): the x rows of a tile are read once, their
//   tiles one after another, where along x the plane's v bands each read every x row again. In the 1389h probe
//   (cells of 16 MiB and more) every such plan was as fast or faster along v: 0.75 to 0.99 of its time along x,
//   FP32 96386 x 143 x 4 elements 372.6 to 292.4 us, FP16 134935 x 92 41.5 to 31.5, FP8 81622 x 47 x 12 80.3 to
//   59.9. Where x is about v the orders were level (0.97 to 1.03); where x is under v along v was up to 1.24x
//   slower (FP16 34 x 623019 x 5), so those keep the tiles along x. 1389j: single bytes (p = 1, 1-byte words) go
//   through a packed tile, four bytes at a time (StridedCopy.cu, denseTransposeBytesKernel): in the 1389i audit the
//   1-byte transposes were at 0.45 to 0.88 of the copy curve from 16 MiB (r 0.56 to 0.88, n 0.45 to 0.85), the same
//   rotations at 2-byte words at 0.97 to 1.04, with one load, shared store, shared load and store and an index decode
//   per byte. The plans and tiles are as before; only the tile's layout in shared memory differs. 1389k: the packed
//   byte tiles go along v from x of half v (TRANSPOSE_BYTES_ALONG_V_DIVISOR), where both orders differ. In the 1389j
//   probe (FP8 rotations, 128 x 128 tiles; times over the matrix kernel's, along v / along x) along v took 0.84 to
//   0.97 of the time along x from x of 0.9 v to 3.6 v (15048 x 16095 0.86 / 0.96, 7976 x 7072 0.92 / 0.95, 27842 x
//   7800 0.83 / 0.99) and was level at 5.8 v (1387 x 240 under 317 batch elements); at x of 0.11 v the orders were
//   level (1808 x 16368, 0.92 / 0.91) and at 0.0005 v along v was 1.10x slower (251 x 499032, 1.10 / 1.00), so
//   those keep the tiles along x. 1389l: a packed byte tile whose sides give every thread a 4 x 4 block (ceil(tile_x
//   / 4) ceil(tile_v / 4) of THREADS or more) keeps them: the short side's growth to the square's area only padded
//   its rows (a power of two of words) and cut the tiles resident on an SM. In the 1389k probe the ungrown tile took
//   0.74 of the grown one's time on 125 x 480 under 528 batch elements (125 x 131, 32 KB, to 125 x 128, 16 KB), 0.90
//   on 51 x 10080 under 187 (51 x 321 to 51 x 128), 0.94 on 234227 x 106 (154 x 106 to 128 x 106), 0.97 on 172 x
//   52 under 4800 and 0.99 on 41 x 2356 under 2567. A tile under one block a thread grows as before (8 x 365 was
//   1.27x slower as 8 x 128 in the 1389j probe). 1389m: from TRANSPOSE_PACKED_MIN_BLOCKS (320, 1.25 a thread). In
//   the 1389l audit the two ungrown tiles of 288 blocks were slower than grown (33 x 128 under 2160 batch elements
//   21.47 -> 24.02 us, 93 x 48 under 153 9.13 -> 10.33) and every one of 320 blocks or more faster or level (38 x 128
//   of 38 x 317762 22.22 -> 18.88, 51 x 128 170.4 -> 155.3, 125 x 128 56.6 -> 41.9, 128 x 99 29.15 -> 25.94).
//   1389t: a packed byte tile grown along x past four sides (512 bytes) stops there if the grown tile leaves fewer
//   than TRANSPOSE_PACKED_MIN_GROWN_TILES (2048) tiles. In the 1389s probe (FP8, time over the tile's) the grown
//   tile ran 1.33x slower than 512 x 9 on 1192132 x 9 (1820 x 9, 656 tiles, 23.6 us; 128 x 9 1.26x) and 1.52x on
//   89424 x 9 under 9 batch elements (1547 x 9, 522 tiles, 20.9 us; 128 x 9 1.50x), likely from too few CTAs (about
//   three an SM, up to 16 KB each). With 2048 or more it stays: 594 x 13 under 3315 batch elements (24 MiB) was
//   0.994 as 512 x 13 and 0.91 as 128 x 13.
//   1389r: vectors of single bytes that are not packed (1-byte words, p of 3, 5, 7, ...) in a copy of
//   TRANSPOSE_TALL_BYTES_MIN_BYTES or more take a tall tile: tile rows of TRANSPOSE_TALL_BYTES_ROW_BYTES (x of that
//   over the vector, rounded up) and as many v rows as the shared memory holds, if that holds the square tile's
//   vectors or more. That kernel moves one byte a thread per load and per store, so it is bound by its loops, and a
//   square tile of 32 x 32 on an x of 37 runs its second x tile (5 vectors) at the full tile's cost. In the 1389q
//   probe on FP8 37 x 1149980 x 5 (37 x 1149980 vectors of 5 bytes, 212 MB) 32 x 32 took 507.9 us (0.54 of the copy
//   curve), 7 x 853 402.3 (1.26x), 26 x 227 450.9, 37 x 8 456.4; the same layout at 2-byte words (34 x 623019 x 5)
//   was level at 0.97 of the curve on every tile. Under 16 MiB (FP8 22586 x 21 x 7, 80 x 15015 x 3, 0.48 to 0.50 of
//   the curve) the probe did not measure.
// - MatrixTranspose (1389e): a plain rotation, [x, v] of a dense [v, x] matrix of single words of 2 or 4 bytes
//   (1389k: not 1 byte; below) (no batch axis), both sides at least one tile of ThorKernels' matrix transpose kernel
//   (1394: StridedCopy.cu's denseMatrixTransposeKernel, from the deleted gpuMatrixTransposeKernels.cu: 32 x 32 packed
//   4-byte words, so 128, 64 or 32 words a side, a compile-time shape with no division per element), within that
//   kernel's limits (under 2^31 words, at most 65,535 row tiles). At 4-byte words: any size. At 2-byte words
//   (through 1389j also 1-byte): 24 MiB of words or more. Those
//   are where that kernel measured as fast as the Transpose family or faster in the 1389d audit. 4-byte words: level
//   to 1.26x faster at 16 MiB moved and up, about 1.1x at 1 to 16 MiB and 1.6x under 1 MiB (medians). 1- and 2-byte
//   words: level to 1.87x faster from 28 MiB of words (one cell 3% slower); under 24 MiB of words the Transpose
//   family was level to several times faster. 1389i: not when x is 8 v or more in 16 MiB of words or more; those are
//   Transposes with the tiles along v. In the 1389h probe the planner's tile along v took 0.38 to 0.97 of the
//   matrix kernel's time on each of them: FP32 41077 x 1555 428.5 to 342.8 us, FP16 63087 x 2025 575.8 to 349.0,
//   FP8 96855 x 2638 738.1 to 420.2, FP8 278450 x 198 x 4 (4-byte words) 776.9 to 297.1, FP32 77588 x 80 33.1 to
//   32.2. At x of 3 to 5 v it was 0.95 to 1.02 (FP32 13127 x 4136 0.95, FP8 27842 x 7800 1.02) and under 16 MiB it
//   was not measured; those stay with the matrix kernel. 1389k: no rotation of 1-byte words; they are Transposes
//   through the packed byte tile. In the 1389j probe that tile (128 x 128, the planner's order) took 0.83 to 1.00 of
//   the matrix kernel's time on each of the five FP8 rotations it had: 27842 x 7800 355.9 to 295.4 us, 209 x 72 x 29
//   x 555 (15048 x 16095) 387.5 to 333.3, 7976 x 7072 80.4 to 74.0, 1808 x 16368 42.5 to 38.6, 251 x 499032 level
//   (178.5).
//   1389s: a plain rotation of 2- or 4-byte words whose x is 7/8 v or more (TRANSPOSE_WIDE_ROTATION_X_EIGHTHS), from
//   TRANSPOSE_WIDE_ROTATION_MIN_BYTES (48 MiB), is a Transpose on the wide tile (4 sides along x, a quarter side
//   along v: 256 x 16 2-byte, 128 x 8 4-byte words) along v, whether it went to the matrix kernel or (x of 8 v or
//   more, 1389i) to the square tile along v. In the 1389q and 1389r probes, time over the wide tile's along v: FP16
//   19117 x 5680 (x of 3.4 v) 1.077 / 1.080 against the matrix kernel, FP32 13127 x 4136 1.062 / 1.058, FP16 10726
//   x 11492 (x of 0.93 v) 1.057 / 1.061, FP32 7579 x 8119 1.024 / 1.030, FP16 5623 x 5016 (56 MB) 1.022 / 1.032,
//   FP32 3963 x 3558 1.005 / 1.007; the square tile along v on x of 8 v or more 0.998 to 1.019 (FP32 41077 x 1555
//   1.019 / 1.012); rotations of 4-byte words of two or four elements 0.996 to 1.010 (FP8 278450 x 198 x 4, 3921 x
//   4186 x 4, 1795746 x 19 x 2, FP16 947755 x 18 x 2). Under x of 7/8 v the matrix kernel stays (1016 x 7280 1.000
//   / 1.005, 180 x 173526 0.95); under 48 MiB the probe measured only x of 8 v or more (0.998 to 1.017); single
//   bytes keep the packed byte tile (512 x 32 along v 0.97 to 0.99 on FP8 92642 x 708).
// - PackedGather (1389b): single words (p = 1) of 1, 2 or 4 bytes over two or more outer axes, fewer than 2^31 of
//   them, into a 16-byte aligned destination: what is left of a general permutation. Each thread gathers the 16 / W
//   words of one 16-byte destination word and stores them together, the source address decoded once and then stepped
//   along the two innermost outer axes. As row-copy words (through 1389a) each word had its own decode over every
//   outer axis and its own store: 0.23 to 0.44 of the copy curve at 1-byte words, 0.4 to 0.7 at 2-byte.
// - RowCopy: everything else. The destination in order, each thread rowCopyItems independent words a loop step, the
//   source address from the outer coordinates (FastDivU32 divisions below 2^31 words, else 64-bit). 1389o: four
//   bytes a thread and step at words of 2 and 4 bytes (two words, one word: a step of 1 KiB a CTA); through 1389n
//   it was 16 bytes (8 and 4 words). The 1389n probe ran every row copy of 16 MiB and more on 1, 2, 4, 8 and 16
//   words a thread, one CTA per step (below: the planner's time, the mean of its two runs, over the count's).
//   4-byte words, one word: single words at stride 33 (one word in four sectors) 1.05, FP32 4864180 words 121.2 ->
//   115.1 us, where the 1387 gather of one word a thread takes 115.0 and was 4 to 7% ahead of every such copy; rows
//   with gaps 0.99 to 1.05 on 16 (median 1.03; FP32 19452 x 53 x 12 64.5 -> 62.7); single words at strides 2 to 4
//   0.98 to 1.01; 8 and 16 words up to 2% and 4% slower than 4 on the rows. 2-byte words, two words: rows 1.00 to
//   1.05 on 16 (FP16 32730 x 63 x 12 65.1 -> 62.7), single words at strides 2 to 33 1.00 to 1.02; one word (512
//   bytes a CTA) 0.81 to 1.00 on the rows. Single bytes keep 16 (8: 0.98 to 1.02; 4: 0.94 to 1.03; 2 and 1: 0.80
//   to 1.03 and 0.49 to 0.83 on the rows) and words of 8 and 16 bytes keep 4 (one word: 1.01 to 1.03 on five rows
//   of 8-byte words, 0.97 to 1.04 at 16-byte), until they are measured again. 1389p: the short step for 2- and
//   4-byte words only up to 768 KiB of destination (ROW_COPY_SHORT_MAX_BYTES), from 16 MiB (ROW_COPY_SHORT_MIN_BYTES)
//   and on single words more than a sector apart; else the 16 bytes a thread of 1389n. In the 1389o audit (against
//   1389n) the short step was level to 1.58x faster up to 677 KB of destination (FP16 36 x 28 rows of 33 words 2.23
//   -> 1.43 us; at 945 KB 0.97), 0.99 to 1.04 from 16 MiB (FP16 149770 x 16 x 5 33.3 -> 32.3 us) and 1.00 to 1.05 on
//   single words at stride 33 at every size (FP32 4864180 words 122.8 -> 118.5 us), but 0.81 to 1.05 on rows in
//   between (median 0.97; FP32 1593 x 13 rows of 33 3.82 -> 4.70 us, FP16 62 x 170 rows of 500 13.29 -> 14.16,
//   FP16 13 x 17823 rows of 33 20.2 -> 21.0) and 0.97 to 1.04 on single words 32 bytes apart or less: likely where
//   a grid of under one resident wave of 4 KiB steps became several waves of 1 KiB (not measured).

#include <cstdint>
#include <string>
#include <vector>

namespace ThorImplementation {

enum class DenseCopyFamily : uint8_t {
    Empty = 0,
    Memcpy = 1,
    RowCopy = 2,
    Transpose = 3,
    PackedGather = 4,
    MatrixTranspose = 5
};

struct DenseCopyPlan {
    DenseCopyFamily family = DenseCopyFamily::Empty;
    uint32_t element_bytes = 0;
    uint32_t word_bytes = 0;  // 1, 2, 4, 8 or 16: a multiple of element_bytes
    uint64_t words = 0;       // the destination's words

    // RowCopy and PackedGather (p = 1): destination word j of the row-major [outer_dims..., p] at coordinates
    // (c..., e) is source word sum over k of c_k outer_strides[k], plus e. Strides in words.
    uint64_t p = 1;
    std::vector<uint64_t> outer_dims, outer_strides;

    // Transpose: x rows of p words adjacent in the source (source stride p), x_dest_stride apart in the destination;
    // v rows adjacent in the destination (destination stride p), v_source_stride apart in the source; the batch axes
    // (outer to inner) with their source and destination strides. All in words. MatrixTranspose (1389e): x and v
    // of the dense [v, x] source matrix (p = 1, v_source_stride = x, x_dest_stride = v, no batch axes, no tile).
    uint64_t x = 1, v = 1, x_dest_stride = 0, v_source_stride = 0;
    std::vector<uint64_t> batch_dims, batch_source, batch_dest;
    uint32_t tile_x = 0, tile_v = 0;
    // Transpose: the order of the tiles in a plane. false: along x within a band of tile_v v rows, so a band's
    // source rows are read through before the next band. true: along v within tile_x x rows, so a tile's destination
    // rows are complete before the next x rows. 1389i: the planner's when x is TRANSPOSE_ALONG_V_RATIO v or more and
    // both orders differ (more than one tile along x and along v); thor_dense_copy_audit --probe sets either (1389g).
    // 1389k: packed bytes (transposePacksBytes) from x of v / TRANSPOSE_BYTES_ALONG_V_DIVISOR.
    bool tile_v_first = false;

    uint32_t grid = 0;  // CTAs (RowCopy, PackedGather, Transpose): one per unit of work (1389h); else 0

    // The source layout the plan was made for, in elements: a launch on another span checks it has this layout.
    std::vector<uint64_t> source_dims, source_strides;

    bool operator==(const DenseCopyPlan&) const = default;
};

/** "empty", "memcpy_w<bytes>", "row_copy_o<outer axes>_p<p>_w<bytes>", "packed_gather_o<outer axes>_w<bytes>",
 *  "transpose_p<p>_w<bytes>_t<tile_x>x<tile_v>_b<batch axes>" (with "_vfirst" when tile_v_first) or
 *  "matrix_transpose_w<bytes>". */
[[nodiscard]] std::string denseCopyPlanName(const DenseCopyPlan& plan);

/** The patch that last changed the planner or the kernels, from the library: the audit prints it, so a run against
 *  a library that was not rebuilt shows in its header. */
[[nodiscard]] const char* denseCopyRevision();

namespace DenseCopyInternal {

inline constexpr uint32_t THREADS = 256;
inline constexpr uint32_t MAX_RANK = 32;
inline constexpr uint64_t SECTOR_BYTES = 32;
inline constexpr uint64_t MAX_32BIT_INDEX = uint64_t{1} << 31;  // FastDivU32::MAX_DIVIDEND
inline constexpr uint64_t TRANSPOSE_TARGET_TILES = 512;            // the tile side's size target
inline constexpr uint64_t TRANSPOSE_MAX_SHARED_BYTES = 32 * 1024;  // one tile
inline constexpr uint64_t TRANSPOSE_MIN_PLANE_VECTORS = 256;       // an x by v plane under this: the row copy
inline constexpr uint64_t TRANSPOSE_MIN_TILE_WORDS = 1024;         // the smallest square tile: 4 words a thread
inline constexpr uint64_t TRANSPOSE_SMALL_TILE_WORDS = 640;        // under 2.5 words a thread, and
inline constexpr uint64_t TRANSPOSE_MAX_SMALL_TILES = 2048;        // more of them than this: the row copy
inline constexpr uint64_t TRANSPOSE_MAX_WIDE_ROW_BYTES = 2 * SECTOR_BYTES;  // rows of one to two sectors (16-byte
inline constexpr uint64_t TRANSPOSE_MIN_WIDE_BYTES = uint64_t{4} << 20;     // words): transposed from this size
inline constexpr uint64_t PACK_BYTES = 16;                         // a packed gather's destination word
inline constexpr uint64_t MATRIX_TRANSPOSE_SIDE_BYTES = 128;       // the matrix kernel's tile side: 32 4-byte words
inline constexpr uint64_t MATRIX_TRANSPOSE_MAX_ROW_TILES = 65535;  // its grid's y extent
inline constexpr uint64_t MATRIX_TRANSPOSE_MIN_NARROW_BYTES = uint64_t{24} << 20;  // of 2-byte words
inline constexpr uint64_t TRANSPOSE_ALONG_V_RATIO = 8;  // 1389i: x of this many v or more: the tiles go along v, and
inline constexpr uint64_t MATRIX_TRANSPOSE_MAX_LONG_ROW_BYTES = uint64_t{16} << 20;  // a rotation this large is tiles
inline constexpr uint64_t TRANSPOSE_BYTES_ALONG_V_DIVISOR = 2;  // 1389k: packed bytes: x of v / this or more, along v
inline constexpr uint64_t TRANSPOSE_PACKED_MIN_BLOCKS = 320;  // 1389m: 4 x 4 blocks: a packed byte tile keeps its sides
inline constexpr uint64_t TRANSPOSE_PACKED_MIN_GROWN_TILES = 2048;  // 1389t: under this, x grows to 4 sides at most
inline constexpr uint64_t TRANSPOSE_TALL_BYTES_MIN_BYTES = uint64_t{16} << 20;  // 1389r: unpacked single bytes from
inline constexpr uint64_t TRANSPOSE_TALL_BYTES_ROW_BYTES = 32;                    // this take tile rows of this
inline constexpr uint64_t TRANSPOSE_WIDE_ROTATION_MIN_BYTES = uint64_t{48} << 20;  // 1389s: rotations from this,
inline constexpr uint64_t TRANSPOSE_WIDE_ROTATION_X_EIGHTHS = 7;                    // x of this / 8 v: the wide tile

inline constexpr uint64_t ROW_COPY_SHORT_MAX_BYTES = 768 * 1024;          // 1389p: row copies of 2- or 4-byte words
inline constexpr uint64_t ROW_COPY_SHORT_MIN_BYTES = uint64_t{16} << 20;  // to this or from this take short steps

/** Independent words per thread and loop step of the row copy, 16 bytes a thread: 16 single bytes, 8 2-byte words, 4
 *  4-byte words; 4 words of 8 or 16 bytes. 1389p: the long step (as through 1389n; 1389o took the short one). */
[[nodiscard]] constexpr uint32_t rowCopyItems(uint32_t word_bytes) { return word_bytes >= 4 ? 4 : 16 / word_bytes; }

/** 1389p: the short step of 2- and 4-byte words, 4 bytes a thread (a step of 1 KiB a CTA); other words as
 *  rowCopyItems. */
[[nodiscard]] constexpr uint32_t rowCopyShortItems(uint32_t word_bytes) {
    return word_bytes == 2 || word_bytes == 4 ? 4 / word_bytes : rowCopyItems(word_bytes);
}

/** 1389p: whether a row copy takes the short step: words of 2 or 4 bytes, and a destination of up to
 *  ROW_COPY_SHORT_MAX_BYTES or from ROW_COPY_SHORT_MIN_BYTES, or single words further apart than a sector. */
[[nodiscard]] inline bool rowCopyTakesShortSteps(const DenseCopyPlan& plan) {
    if (plan.word_bytes != 2 && plan.word_bytes != 4) return false;
    const uint64_t bytes = plan.words * plan.word_bytes;
    if (bytes <= ROW_COPY_SHORT_MAX_BYTES || bytes >= ROW_COPY_SHORT_MIN_BYTES) return true;
    return plan.p == 1 && !plan.outer_strides.empty() && plan.outer_strides.back() * plan.word_bytes > SECTOR_BYTES;
}

/** Loads a thread of the transpose kernel holds in flight: 16 bytes of 1-byte words, 8 of 2-byte, else 4 words
 *  (1389o: its own function; through 1389n it was the row copy's count). */
[[nodiscard]] constexpr uint32_t transposeLoadItems(uint32_t word_bytes) {
    return word_bytes >= 4 ? 4 : 16 / word_bytes;
}

/** A row copy's words per thread and loop step (1389p): rowCopyShortItems if rowCopyTakesShortSteps, else
 *  rowCopyItems. 1395: the only counts; through 1394 a plan's row_items could set another, for the audit's probe. */
[[nodiscard]] inline uint32_t rowCopyStepItems(const DenseCopyPlan& plan) {
    return rowCopyTakesShortSteps(plan) ? rowCopyShortItems(plan.word_bytes) : rowCopyItems(plan.word_bytes);
}

/** Shared-memory row pitch of the transpose tile, in words: a tile row of tile_x vectors rounded up to an odd number
 *  of banks (4-byte units, or words when wider), so the column reads of the store phase fall in distinct banks. 1389:
 *  the side is no longer a power of two, so a fixed pad of one bank could leave the pitch even. */
[[nodiscard]] constexpr uint64_t transposePitchWords(uint64_t tile_x, uint64_t p, uint32_t word_bytes) {
    const uint64_t unit = word_bytes >= 4 ? word_bytes : 4;  // bytes per bank
    uint64_t banks = (tile_x * p * word_bytes + unit - 1) / unit;
    if (banks % 2 == 0) ++banks;
    return banks * unit / word_bytes;
}

/** Words between the rows of a packed byte tile (1389j): the smallest power of two that holds tile_v single bytes in
 *  4-byte words, so a row's words can be swizzled by an XOR within it. */
[[nodiscard]] constexpr uint64_t transposeByteRowWords(uint64_t tile_v) {
    uint64_t words = 1;
    while (words * 4 < tile_v) words *= 2;
    return words;
}

/** Whether a transpose of vectors of p words of word_bytes bytes takes the packed byte tile (1389j): single bytes,
 *  p = 1. */
[[nodiscard]] constexpr bool transposePacksBytes(uint64_t p, uint32_t word_bytes) { return p == 1 && word_bytes == 1; }

/** Shared bytes of a transpose tile: tile_v rows of transposePitchWords words; 1389j: a packed byte tile is
 *  tile_x rows (rounded up to 4) of transposeByteRowWords 4-byte words, each word four v rows of one x row. */
[[nodiscard]] constexpr uint64_t transposeSharedBytes(uint64_t tile_x, uint64_t tile_v, uint64_t p,
                                                      uint32_t word_bytes) {
    if (transposePacksBytes(p, word_bytes)) return (tile_x + 3) / 4 * 4 * transposeByteRowWords(tile_v) * 4;
    return tile_v * transposePitchWords(tile_x, p, word_bytes) * word_bytes;
}

/**
 * Every field but grid, for a span of `dims` / `strides` (elements) of `element_bytes` (1, 2, 4 or 8) elements at
 * `source_address`, copied to a dense destination at `destination_address`. Throws std::invalid_argument on a rank
 * outside [1, 32], mismatched ranks, another element size or an element count past 2^64.
 */
[[nodiscard]] DenseCopyPlan planLayout(const std::vector<uint64_t>& dims, const std::vector<uint64_t>& strides,
                                       uint32_t element_bytes, uintptr_t source_address,
                                       uintptr_t destination_address);

/** The plan's units of work: transpose tiles, row-copy loop steps (THREADS x rowCopyStepItems words) or packed-gather
 *  loop steps (THREADS packs of PACK_BYTES); 0 otherwise. */
[[nodiscard]] uint64_t workUnits(const DenseCopyPlan& plan);

/**
 * The grid planCopyToDense gives `plan`: one CTA per unit of work (0 for a plan with none). Also for a plan whose
 * tile was changed by hand (thor_dense_copy_audit --probe and the tests; nothing else changes a plan).
 *
 * 1389h. Through 1389g the grid was the CTAs resident at once (170 SMs x 3 to 6 on the RTX 5090: 510 to 1,020), each
 * looping over every grid-th unit. The 1389g probe ran each plan of 16 MiB and more on larger grids, up to this one
 * (FP32, FP16 and FP8; 190 cells with more units than resident CTAs): 99 were more than 2% faster on one CTA per unit
 * and 2 more than 2% slower (2% and 3%). The median time over the plan's was 0.99 to 1.00 under 64 MiB moved, 0.97
 * to 0.98 at 64 to 256 MiB and 0.95 to 0.97 from 256 MiB. Row copies of 153 to 244 MiB went from 0.88-0.95 of the
 * copy curve to 0.99-1.03 (352.8 to 319.2 us at FP32 on 29541 x 63 x 33 rows with slab gaps, where
 * cudaMemcpy2DAsync takes 317.1); FP16 256 x 1058 x 15 x 14 (a transpose under a batch axis) from 263.4 to 148.4
 * us. A quarter of the resident CTAs was up to 1.9x slower. Copies of fewer units than resident CTAs (under about
 * 1 to 16 MiB, by family and word) launch as before.
 */
[[nodiscard]] uint32_t gridOf(const DenseCopyPlan& plan);

/** Dynamic shared memory of a transpose launch (0 for the other families). */
[[nodiscard]] uint64_t sharedBytes(const DenseCopyPlan& plan);

}  // namespace DenseCopyInternal
}  // namespace ThorImplementation
