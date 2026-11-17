#pragma once

// R / KR reductions: host-only work-shape vocabulary (FP8, and the other dtypes by element size below). No CUDA.
//
// A KR site reduces contiguous rows: [rows, row_bytes] -> [rows] (R is KR with one row). Three kernel families read the
// input as 4-byte words on the tensor's word grid (any input address works) and mask bytes outside their work:
//   - Pack  (rows_per_thread 2..16): one thread owns rows_per_thread consecutive short rows (row_bytes <= 32) and walks
//     their bytes with a row counter, storing each row's result as it completes. A `vector` Pack shape (row_bytes
//     in FP8_KR_VECTOR_ROW_BYTES, rows_per_thread = krVectorRows(row_bytes), a 16-byte multiple per thread) loads
//     its rows as aligned 16-byte vectors with compile-time row boundaries and stores FP32 results as float4; an input
//     that is not 16-byte aligned runs the same shape through the generic Pack kernel.
//   - Group (group_lanes 1..32): a group of G lanes owns one row; lane g reads the words g, g + G, ... of the row.
//   - Split (group_lanes == KR_THREADS, chunk_bytes < row_bytes): one CTA owns one chunk of chunk_bytes of a row
//     and writes a partial to the pass's scratch; a finishing kernel (one CTA per row) combines the row's partials in
//     chunk order and stores the result. No atomics (a per-row ticket would save only the finishing
//     launch, measured at 1.3-1.4 us).
//   - Block (group_lanes == KR_THREADS, chunk_bytes == row_bytes, rows of at least 1 KiB): one CTA owns one
//     whole row and stores its result: the Split partial kernel's work mapping without the scratch and the finishing
//     launch, for rows too long for Group's G <= 32 lanes to fill the GPU (in a sweep, Group at under 4 CTAs per
//     SM ran 1.32-1.39 TB/s where the Split partial kernel alone ran 6-8% faster; Block won 81-100% of cells
//     with rows of 2-64 KiB and at least one row per SM).
// VALUE partials are unfinalized FP32 (MIN/MAX in RK's folded MIN domain); ARG partials are 64-bit wide candidates
// (Fp8ArgKey.h). The scratch needs no initialization.

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ThorImplementation::ReductionInternal::KR {

inline constexpr uint32_t KR_THREADS = 256;
inline constexpr uint64_t KR_MAX_CHUNKS = 4096;
inline constexpr uint64_t KR_MIN_CHUNK_BYTES = 4096;
inline constexpr uint64_t KR_MAX_PACK_ROW_BYTES = 32;
inline constexpr uint64_t KR_MAX_PACK_SPAN_BYTES = 128;
inline constexpr uint64_t KR_MIN_BLOCK_ROW_BYTES = 1024;  // one word per thread

enum class KRFamily : uint8_t { Pack = 0, Group = 1, Split = 2, Block = 3 };

// Row lengths with a compiled vector Pack kernel (the short rows a KR audit found off line rate).
inline constexpr uint32_t FP8_KR_VECTOR_ROW_BYTES[] = {2, 3, 5, 6};
[[nodiscard]] constexpr bool fp8KRVectorRowBytes(uint64_t row_bytes) {
    for (uint32_t r : FP8_KR_VECTOR_ROW_BYTES)
        if (r == row_bytes) return true;
    return false;
}
// FP16 / BF16 rows with a compiled vector Pack kernel: 1 to 8 elements. Ported without a measured gap (a sweep's
// generic Pack reached the copy rate on short 2-byte rows on GDDR), for GPUs with more bandwidth per SM, where the
// generic kernel's 4-byte loads at a 8-16 byte lane stride may limit it.
inline constexpr uint32_t F16_KR_VECTOR_ROW_BYTES[] = {2, 4, 6, 8, 10, 12, 14, 16};
[[nodiscard]] constexpr bool f16KRVectorRowBytes(uint64_t row_bytes) {
    for (uint32_t r : F16_KR_VECTOR_ROW_BYTES)
        if (r == row_bytes) return true;
    return false;
}
// Rows per thread of the vector Pack kernel: the fewest whole rows filling whole 16-byte vectors (16 / gcd(R, 16)), and
// at least 2, since a shape with one row per thread is a Group shape (16-byte 2-byte rows: two vectors).
[[nodiscard]] constexpr uint32_t krVectorRows(uint64_t row_bytes) {
    uint64_t a = row_bytes, b = 16;
    while (b != 0) {
        const uint64_t t = a % b;
        a = b;
        b = t;
    }
    return static_cast<uint32_t>(16 / a < 2 ? 2 : 16 / a);
}
static_assert(krVectorRows(2) == 8 && krVectorRows(3) == 16 && krVectorRows(5) == 16 &&
              krVectorRows(6) == 8 && krVectorRows(16) == 2);

struct KRSegmentShape {
    uint64_t rows = 0;
    uint64_t row_bytes = 0;
    uint64_t chunk_bytes = 0;      // Split: bytes per CTA chunk (< row_bytes); otherwise >= row_bytes
    uint32_t group_lanes = 0;      // Group: 1..32; Split and Block: KR_THREADS; Pack: 1; 0: no shape
    uint32_t rows_per_thread = 1;  // Pack: > 1
    bool vector = false;           // vector Pack (see above)

    [[nodiscard]] uint64_t chunks() const {
        return chunk_bytes >= row_bytes ? 1 : (row_bytes + chunk_bytes - 1) / chunk_bytes;
    }
    [[nodiscard]] bool staged() const { return chunks() > 1; }
    [[nodiscard]] uint64_t segments() const { return rows * chunks(); }
    [[nodiscard]] KRFamily family() const {
        if (rows_per_thread > 1) return KRFamily::Pack;
        if (staged()) return KRFamily::Split;
        return group_lanes == KR_THREADS ? KRFamily::Block : KRFamily::Group;
    }
    friend bool operator==(const KRSegmentShape&, const KRSegmentShape&) = default;
};

[[nodiscard]] bool fp8KRSegmentShapeLegal(const KRSegmentShape& shape);

/** Split scratch: the partials, 4 bytes (VALUE) or 8 bytes (ARG) per chunk, rounded up to 16 bytes. Pack and Group shapes
 *  have none. */
[[nodiscard]] uint64_t krScratchBytes(const KRSegmentShape& shape, bool arg);

/** "p8" (Pack, 8 rows per thread), "v8" (vector Pack), "g8" (Group), "s16384" (Split, 16 KiB chunks), "b" (Block). */
[[nodiscard]] std::string krShapeName(const KRSegmentShape& shape);

/**
 * The shapes worth timing at [rows, row_bytes], best guess first (the tuning depth keeps a prefix):
 *   - for rows of at most 16 bytes: the vector Pack shape where one is compiled, then Pack at about 16 and 8 bytes per
 *     thread (32- and 64-byte spans lost everywhere);
 *   - Group at G = 1..32, starting from about 32 bytes per lane, then the neighbouring widths;
 *   - Split when rows alone cannot fill the GPU (rows < 16 per SM) and rows have at least 8 KiB: CTA chunks sized for
 *     about 8 CTAs per SM, then twice and half that size, and so on (2..KR_MAX_CHUNKS chunks per row);
 *   - Split also for many rows of at least 16 KiB: chunks of 8, 4, 16 and 32 KiB, interleaved with Group;
 *   - Block (one CTA per row) first for rows of 2-64 KiB with at least one row per SM.
 * Short rows list Pack first; few long rows list Split first and drop Group shapes with less than one warp per SM.
 * Every returned shape is legal and distinct.
 */
[[nodiscard]] std::vector<KRSegmentShape> fp8KRCandidateShapes(uint64_t rows, uint64_t row_bytes,
                                                               uint32_t multiprocessors);

// ---- The candidate pool, its ranker features and the ranked shortlist (benchmarks/fp8_kr_ranker).
//
// The pool is what the training sweep times and what production ranks: the heuristic list (list_position 1..n, in
// its order), then the legal shapes around it (list_position 0): vector Pack, Pack at 2, 3, 4, 6, 8, 12 and 16 rows
// per thread, Group at every G, Block, and Split at every power-of-two chunk while rows < 64 per SM or rows have at
// least 16 KiB. A shape that is not in the heuristic list and gives one thread more than KR_POOL_MAX_THREAD_BYTES
// of input is left out (it cannot compete, and timing it costs seconds).
inline constexpr uint64_t KR_POOL_MAX_THREAD_BYTES = 64 * 1024;

struct KRPoolEntry {
    KRSegmentShape shape;
    uint32_t list_position = 0;  // 1-based position in fp8KRCandidateShapes; 0: not in it
};

[[nodiscard]] std::vector<KRPoolEntry> fp8KRCandidatePool(uint64_t rows, uint64_t row_bytes,
                                                          uint32_t multiprocessors);

/** The semantics a ranker sees. kind: 0 VALUE, 1 ARG index, 2 ARG value and index (3 / 4 the same at a carried-index
 *  FP32 site). combine: 0 SUM, 1 PRODUCT, 2 MIN, 3 MAX (ARG: 2 argmin, 3 argmax). transform: 0 identity, 1 |x|, 2
 *  x^2. */
struct KRRankerSite {
    uint8_t kind = 0;
    uint8_t combine = 0;
    uint8_t transform = 0;
    bool e5m2 = false;
};

inline constexpr size_t KR_RANKER_FEATURES = 24;

/** The ranker's features of one pool entry (names: benchmarks/fp8_kr_ranker/dataset.py FEATURE_NAMES). The training
 *  sweep writes exactly these floats beside each candidate's time, so the model sees what production computes. */
[[nodiscard]] std::array<float, KR_RANKER_FEATURES> krRankerFeatures(const KRPoolEntry& entry,
                                                                           const KRRankerSite& site,
                                                                           uint32_t multiprocessors);

/** True once a trained Fp8KRRankerModel.h is installed (the placeholder has no trees). */
[[nodiscard]] bool fp8KRRankerTrained();

/** Shortlist lengths per tuning depth (Quick, Regular, Ultra) for the ranked pool. Without a trained model,
 *  production times the heuristic list's prefix at the RK depths instead.
 *  Model c82f0c6b (12256 cells, sm_count 170): train.py's rules chose {1, 2, 4}. Out-of-fold regret of the ranked
 *  top N: N=2 p99 1.012 (VALUE) / 1.018 (ARG), max 1.16 / 1.13; N=4 p99 1.002 / 1.001, max 1.025 / 1.021; N=8 the
 *  oracle shape in every out-of-fold and held-out cell (the heuristic list needed 12 for the same).
 *  {6, 12, 12}, the RK candidate counts at Quick and Regular, with margin for GPUs and sites the training sweep did not
 *  see; pools hold at most 15 shapes, so Ultra stays at 12 and differs from Regular by its rounds and trim
 *  (reductionTuningPlan). */
inline constexpr std::array<size_t, 3> FP8_KR_SHORTLIST_DEPTHS{6, 12, 12};

/** The first `count` shapes to autotune: the pool ordered by the trained ranker's margin (ties in pool order), or,
 *  without a trained model, the heuristic list's prefix. */
[[nodiscard]] std::vector<KRSegmentShape> fp8KRRankedShapes(uint64_t rows, uint64_t row_bytes,
                                                            uint32_t multiprocessors, const KRRankerSite& site,
                                                            size_t count);

// ---- Element size. The shapes are in bytes for every input dtype; the kernels read the input's 4-byte words.
// An FP32 shape (element_bytes 4) is an FP8 shape whose rows are whole elements, on a 4-byte aligned input, with no
// vector Pack (that kernel is FP8-only, ported only if the generic Pack falls short on short FP32 rows). FP32 rows are
// 4-byte multiples, which no vector row length is, so the FP8 list and pool at row_bytes = 4 R are already FP32-legal;
// the filtered functions below check it (and with element_bytes 1 return the FP8 ones unchanged). FP16 / BF16
// (element_bytes 2) the same way, on a 2-byte aligned input. Element size 2 has its own vector Pack lengths
// (F16_KR_VECTOR_ROW_BYTES): its list is FP8's without FP8's vector shapes, with the 2-byte vector shape second where
// one is compiled and the rows fill at least one thread, and its pool adds the vector shape regardless of row count
// (FP8's pool rule).

/** element_bytes 1 (FP8): fp8KRSegmentShapeLegal. 2 (FP16 / BF16): rows of whole elements, fp8KRSegmentShapeLegal
 *  except that a vector Pack shape is legal at F16_KR_VECTOR_ROW_BYTES (rows_per_thread krVectorRows). 4 (FP32):
 *  rows of whole elements, fp8KRSegmentShapeLegal, no vector Pack. */
[[nodiscard]] bool retainedKRShapeLegalForElement(const KRSegmentShape& shape, uint32_t element_bytes);
/** Legal for element size 1, 2 or 4 (the shape-level helpers that serve every dtype: scratch bytes, names, ranker
 *  features). */
[[nodiscard]] bool retainedKRShapeLegalForAnyElement(const KRSegmentShape& shape);

/** fp8KRCandidateShapes / fp8KRCandidatePool restricted to retainedKRShapeLegalForElement: the element's list
 *  (positions 1..n), then the pool's other legal shapes. row_bytes must be a multiple of element_bytes. FP32 (element
 *  4) lists Block second for rows of at least FP32_KR_BLOCK_SECOND_ROW_BYTES with at least one row per SM (an FP32
 *  sweep); FP8 (element 1) is fp8KRCandidateShapes / fp8KRCandidatePool unchanged. */
inline constexpr uint64_t FP32_KR_BLOCK_SECOND_ROW_BYTES = 64 * 1024;
/** An FP32 ARG shape: legal for element size 4, and a Split chunk of at most 2^31 elements (its packed partial holds a
 *  chunk-local index of 31 bits). */
inline constexpr uint64_t FP32_KR_ARG_MAX_CHUNK_BYTES = uint64_t{4} << 31;
[[nodiscard]] bool fp32KRArgShapeLegal(const KRSegmentShape& shape);
/** An FP16 / BF16 ARG shape: legal for element size 2, no vector Pack (2-byte vector ARG is not ported), and a row of
 *  at most 2^47 elements (the candidate's row-local index, F16ArgKey.h). */
[[nodiscard]] bool f16KRArgShapeLegal(const KRSegmentShape& shape);

[[nodiscard]] std::vector<KRSegmentShape> retainedKRCandidateShapes(uint64_t rows, uint64_t row_bytes,
                                                                    uint32_t multiprocessors,
                                                                    uint32_t element_bytes);
[[nodiscard]] std::vector<KRPoolEntry> retainedKRCandidatePool(uint64_t rows, uint64_t row_bytes,
                                                               uint32_t multiprocessors, uint32_t element_bytes);

/** Which kernels a Split launch runs. PartialsOnly (benchmarks only) launches the partial kernel and skips the
 *  finishing kernel, so a sweep can time the finishing kernel's cost as the difference; the output is not written. */
enum class KRSplitStages : uint8_t { Both = 0, PartialsOnly = 1 };

}  // namespace ThorImplementation::ReductionInternal::KR
