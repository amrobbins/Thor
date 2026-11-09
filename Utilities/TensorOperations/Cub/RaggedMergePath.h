#pragma once

// 1264: the merge-path partition of a ragged (offset-segmented) reduction, shared by the ragged KR and RK families
// (plans/Thor_Ragged_Reduction_Plan_2026-11-06.md, design: partitioner and finishing kernel). Pure index arithmetic,
// usable on the host for tests and emulation (test/Utilities/TensorOperations/RaggedMergePathTest.cpp).
//
// The problem: values [rows, D...] and offsets [segments + 1]; segment s holds rows [begin(s), end(s)). Only the first
// `rows` = min(offsets[segments], capacity) rows are active; every offset is clamped to that, so no read ever goes past
// the capacity, even with malformed runtime offsets (their contents stay the producer's contract).
//
// The path merges the `segments` segment ends with the `rows` rows, P = segments + rows events in all. At a path
// coordinate (s ends consumed, r rows consumed) the next event is the end of segment s when end(s) <= r (ties go to the
// end, so empty segments close before the next row), else row r. Segment s therefore occupies the contiguous events
// [s + begin(s), s + end(s)]: its rows, then its end. A diagonal d is the coordinate after d events; mergePathSearch
// finds it with one binary search over the ends (Merrill & Garland, SC16).
//
// CTA c of G takes the diagonals [d_c, d_c+1), d_c = (P / G) * c + min(c, P % G): every CTA gets floor(P / G) or
// ceil(P / G) events, whatever the lengths. Its range is contiguous, so it has at most one open segment at each edge
// (raggedCtaRole):
//   complete  segments whose rows and end are all in the CTA: written by the main kernel, never by the finisher;
//   head      the first segment, when its rows began in an earlier CTA and its end is here: a boundary partial;
//   tail      the last segment, when it has rows here but its end is in a later CTA: a boundary partial (a CTA inside
//             one segment has only a tail).
// The finisher (raggedFinishRun) combines, for each CTA b with a head, the tails of the CTAs before it that belong to
// the same segment, in CTA order, then b's head: a fixed order with no atomics, so results repeat run to run.
#include <cstdint>

#if defined(__CUDACC__)
#define THOR_RAGGED_HD __host__ __device__ __forceinline__
// 1265: the templates below take functors; the kernels pass device lambdas, which nvcc's host / device call check
// would flag in the (never used) host instantiation. CCCL suppresses the same check the same way.
#define THOR_RAGGED_EXEC_CHECK_DISABLE _Pragma("nv_exec_check_disable")
#else
#define THOR_RAGGED_HD inline
#define THOR_RAGGED_EXEC_CHECK_DISABLE
#endif

namespace ThorImplementation::CubReductionInternal::RaggedMergePath {

inline constexpr uint64_t NO_SEGMENT = ~uint64_t{0};

// Offsets of a runtime width: one kernel serves UINT32 and UINT64 offsets (plan decision 3).
struct Offsets {
    const void* data = nullptr;
    uint32_t bytes = 4;  // 4 or 8
};

THOR_RAGGED_HD uint64_t loadOffset(const Offsets& offsets, uint64_t index) {
    return offsets.bytes == 8 ? static_cast<const uint64_t*>(offsets.data)[index]
                              : static_cast<uint64_t>(static_cast<const uint32_t*>(offsets.data)[index]);
}

THOR_RAGGED_HD uint64_t clampRows(uint64_t value, uint64_t rows) { return value < rows ? value : rows; }

// The active rows: offsets[segments], clamped to the capacity.
THOR_RAGGED_HD uint64_t activeRows(const Offsets& offsets, uint64_t segments, uint64_t capacity_rows) {
    return clampRows(loadOffset(offsets, segments), capacity_rows);
}

struct Coordinate {
    uint64_t segment = 0;  // segment ends consumed
    uint64_t row = 0;      // rows consumed
};

// A partition problem: `rows` is the active (clamped) row count. begin(s) and end(s) are clamped to it.
struct Problem {
    Offsets offsets;
    uint64_t segments = 0;
    uint64_t rows = 0;

    [[nodiscard]] THOR_RAGGED_HD uint64_t pathLength() const { return segments + rows; }
    [[nodiscard]] THOR_RAGGED_HD uint64_t begin(uint64_t segment) const {
        return clampRows(loadOffset(offsets, segment), rows);
    }
    [[nodiscard]] THOR_RAGGED_HD uint64_t end(uint64_t segment) const {
        return clampRows(loadOffset(offsets, segment + 1), rows);
    }
};

THOR_RAGGED_HD Problem makeProblem(const Offsets& offsets, uint64_t segments, uint64_t capacity_rows) {
    return Problem{offsets, segments, activeRows(offsets, segments, capacity_rows)};
}

// The coordinate after `diagonal` events, searched within the window [low, high] of the path (low and high are
// coordinates on the path with low.segment + low.row <= diagonal <= high.segment + high.row). The whole path is the
// window [{0, 0}, {segments, rows}]; a CTA searches its own window for per-thread splits. `end_of(s)` is the clamped
// end of segment s (a functor, so staged shared-memory offsets work as well as global ones).
THOR_RAGGED_EXEC_CHECK_DISABLE
template <typename EndOf>
THOR_RAGGED_HD Coordinate mergePathSearch(uint64_t diagonal, const Coordinate& low, const Coordinate& high,
                                          const EndOf& end_of) {
    // Ends consumed x lies in [max(low.segment, diagonal - high.row), min(high.segment, diagonal - low.row)].
    uint64_t x_min = low.segment;
    if (diagonal > high.row && diagonal - high.row > x_min) x_min = diagonal - high.row;
    uint64_t x_max = diagonal - low.row;
    if (high.segment < x_max) x_max = high.segment;
    while (x_min < x_max) {
        const uint64_t pivot = x_min + ((x_max - x_min) >> 1);
        // End `pivot` precedes row diagonal - pivot - 1: it is among the first `diagonal` events.
        if (end_of(pivot) <= diagonal - pivot - 1) {
            x_min = pivot + 1;
        } else {
            x_max = pivot;
        }
    }
    return Coordinate{x_min, diagonal - x_min};
}

struct ProblemEnd {
    const Problem* problem;
    THOR_RAGGED_HD uint64_t operator()(uint64_t segment) const { return problem->end(segment); }
};

THOR_RAGGED_HD Coordinate searchPath(const Problem& problem, uint64_t diagonal) {
    return mergePathSearch(diagonal, Coordinate{0, 0}, Coordinate{problem.segments, problem.rows},
                           ProblemEnd{&problem});
}

// The first diagonal of CTA `cta` of `grid` (cta == grid gives the path length): floor or ceil of P / G events each.
THOR_RAGGED_HD uint64_t ctaDiagonal(uint64_t path_length, uint64_t grid, uint64_t cta) {
    const uint64_t share = path_length / grid, extra = path_length % grid;
    return share * cta + (cta < extra ? cta : extra);
}

// 1268: a weighted path. A row costs `row` and a segment end `end` (the bytes each moves: a row D x input bytes, an end
// its offset plus its D outputs), and CTA c takes the events that START at a cost in [c_c, c_c+1), the cost split
// evenly with ctaDiagonal over pathCost. With unit weights the cost of an event's start is its diagonal, so this is
// exactly the unweighted split. An event belongs to the CTA its start cost falls in, so a CTA whose share is at least
// the costlier event always holds an event: raggedEffectiveWeights falls back to unit weights when some share would be
// smaller, which keeps every empty CTA at the end (the finisher's walk back never crosses one). The path, the roles and
// the finisher are unchanged: only where the CTA boundaries fall moves.
struct PathWeights {
    uint64_t row = 1;
    uint64_t end = 1;
};

THOR_RAGGED_HD uint64_t pathCost(const Problem& problem, const PathWeights& weights) {
    return weights.end * problem.segments + weights.row * problem.rows;
}

// The weights a launch uses: `requested` when every CTA's share of the cost (at least floor(cost / grid)) covers the
// costlier event, unit weights otherwise. Every CTA computes the same answer from the same offsets.
THOR_RAGGED_HD PathWeights raggedEffectiveWeights(const Problem& problem, const PathWeights& requested,
                                                  uint64_t grid) {
    if (requested.row == 0 || requested.end == 0) return PathWeights{};
    const uint64_t costlier = requested.row > requested.end ? requested.row : requested.end;
    if (pathCost(problem, requested) / grid < costlier) return PathWeights{};
    return requested;
}

// The coordinate holding exactly the events whose start cost is below `cost`. End s starts at cost
// end * s + row * end(s) (s ends and end(s) rows before it, ties to the end as the unweighted path), so the ends
// consumed x is the count of s with that below `cost`, monotone in s; the rows consumed are every row before end(x-1)
// (they start before end x - 1 does) and every later row r with end * x + row * r < cost, never past end(x).
// 1272: the same from the offsets a search already holds: `floor` = end(x - 1) (0 for x = 0), `limit` = end(x) (the
// rows for x = segments).
THOR_RAGGED_HD uint64_t raggedRowsAtCostBounded(const PathWeights& w, uint64_t cost, uint64_t x, uint64_t floor,
                                                uint64_t limit) {
    uint64_t rows = 0;
    if (cost > w.end * x) rows = (cost - w.end * x + w.row - 1) / w.row;
    if (rows < floor) rows = floor;
    return rows < limit ? rows : limit;
}

THOR_RAGGED_HD uint64_t raggedRowsAtCost(const Problem& problem, const PathWeights& w, uint64_t cost, uint64_t x) {
    return raggedRowsAtCostBounded(w, cost, x, x > 0 ? problem.end(x - 1) : 0,
                                   x < problem.segments ? problem.end(x) : problem.rows);
}

// 1274: the half-warp search's first round over [0, segments), which needs neither the costs nor the row count, so its
// reads go out with the row count's: slot s's probe. With at most 15 segments the first round is also the last (one
// candidate per slot: x = s); otherwise the rounds' probe (s + 1) * step, step = (segments + 16) / 17, clamped to the
// last segment.
THOR_RAGGED_HD uint64_t raggedSearchRound0Probe(uint64_t segments, uint32_t slot) {
    if (segments <= 15) return slot;
    const uint64_t probe = ((segments + 16) / 17) * (slot + 1);
    return probe < segments ? probe : segments - 1;
}

// 1272: a coordinate with its segment's clamped begin and end (begin(segment) for segment <= segments, end(segment)
// for segment < segments; the rows otherwise), so the role and the first segment's rows need no further reads.
struct SearchedCoordinate {
    Coordinate at;
    uint64_t begin = 0;
    uint64_t end = 0;
};

// The range of x the search needs: ends at or beyond ceil(cost / end) start at or after `cost`; ends before
// ceil((cost - row * rows) / end) start below it whatever their rows.
struct CostSearchRange {
    uint64_t lo = 0, hi = 0;
};

THOR_RAGGED_HD CostSearchRange raggedCostSearchRange(const Problem& problem, const PathWeights& w, uint64_t cost) {
    CostSearchRange r;
    r.hi = (cost + w.end - 1) / w.end;
    if (r.hi > problem.segments) r.hi = problem.segments;
    const uint64_t all_rows = w.row * problem.rows;
    if (cost > all_rows) {
        r.lo = (cost - all_rows + w.end - 1) / w.end;
        if (r.lo > r.hi) r.lo = r.hi;
    }
    return r;
}

THOR_RAGGED_HD uint64_t raggedEndsBelowCost(const Problem& problem, const PathWeights& w, uint64_t cost, uint64_t lo,
                                             uint64_t hi) {
    while (lo < hi) {
        const uint64_t pivot = lo + ((hi - lo) >> 1);
        if (w.end * pivot + w.row * problem.end(pivot) < cost) {
            lo = pivot + 1;
        } else {
            hi = pivot;
        }
    }
    return lo;
}

THOR_RAGGED_HD Coordinate searchPathCost(const Problem& problem, const PathWeights& w, uint64_t cost) {
    const CostSearchRange r = raggedCostSearchRange(problem, w, cost);
    const uint64_t x = raggedEndsBelowCost(problem, w, cost, r.lo, r.hi);
    return Coordinate{x, raggedRowsAtCost(problem, w, cost, x)};
}

// CTA `cta`'s first coordinate under weights `w` (cta == grid gives the path's end).
THOR_RAGGED_HD Coordinate raggedCtaCoordinate(const Problem& problem, const PathWeights& w, uint64_t grid,
                                              uint64_t cta) {
    return searchPathCost(problem, w, ctaDiagonal(pathCost(problem, w), grid, cta));
}

struct CtaRole {
    Coordinate begin, end;               // the CTA's events are the diagonals [begin, end)
    uint64_t head = NO_SEGMENT;          // segment of the head partial (rows began earlier, end here), or NO_SEGMENT
    uint64_t tail = NO_SEGMENT;          // segment of the tail partial (rows here, end later), or NO_SEGMENT
    uint64_t first_complete = 0;         // complete segments are [first_complete, end.segment)
    uint64_t head_rows_begin = 0;        // the head's rows here are [begin.row, head_rows_end)
    uint64_t head_rows_end = 0;
    uint64_t tail_rows_begin = 0;        // the tail's rows here are [tail_rows_begin, end.row)

    [[nodiscard]] THOR_RAGGED_HD bool hasHead() const { return head != NO_SEGMENT; }
    [[nodiscard]] THOR_RAGGED_HD bool hasTail() const { return tail != NO_SEGMENT; }
};

// 1265: the role from the CTA's coordinates (kernels find them with a warp-wide search; raggedCtaRole with
// searchPath).
// 1272: from the begins and ends the search returned: first = the CTA's first coordinate's segment (begin and end),
// last = its last coordinate's segment (begin). Reads nothing.
THOR_RAGGED_HD CtaRole raggedCtaRoleFromBounds(uint64_t segments, const SearchedCoordinate& first,
                                               const SearchedCoordinate& last) {
    CtaRole role;
    role.begin = first.at;
    role.end = last.at;
    role.first_complete = role.begin.segment;
    const uint64_t s0 = role.begin.segment;
    // Head: the first segment's rows began before this CTA, and its end is here.
    if (s0 < role.end.segment && first.begin < role.begin.row) {
        role.head = s0;
        role.head_rows_begin = role.begin.row;
        role.head_rows_end = first.end;
        role.first_complete = s0 + 1;
    }
    // Tail: the last segment (its end not here) has rows here.
    const uint64_t l = role.end.segment;
    if (l < segments) {
        const uint64_t first_row = last.begin > role.begin.row ? last.begin : role.begin.row;
        if (role.end.row > first_row) {
            role.tail = l;
            role.tail_rows_begin = first_row;
        }
    }
    return role;
}

// The begin and end a search returns for coordinate `at` (raggedCtaRoleFromBounds), read from the offsets.
THOR_RAGGED_HD SearchedCoordinate raggedSearched(const Problem& problem, const Coordinate& at) {
    return SearchedCoordinate{at, at.segment <= problem.segments ? problem.begin(at.segment) : problem.rows,
                              at.segment < problem.segments ? problem.end(at.segment) : problem.rows};
}

THOR_RAGGED_HD CtaRole raggedCtaRoleFromCoordinates(const Problem& problem, const Coordinate& begin,
                                                    const Coordinate& end) {
    return raggedCtaRoleFromBounds(problem.segments, raggedSearched(problem, begin), raggedSearched(problem, end));
}

THOR_RAGGED_HD CtaRole raggedCtaRole(const Problem& problem, uint64_t grid, uint64_t cta) {
    const uint64_t path = problem.pathLength();
    return raggedCtaRoleFromCoordinates(problem, searchPath(problem, ctaDiagonal(path, grid, cta)),
                                        searchPath(problem, ctaDiagonal(path, grid, cta + 1)));
}

// The finisher's run for a CTA with a head: the CTAs [first, cta) whose tails belong to the head's segment, combined
// in order before the head. `tail_of(c)` is CTA c's recorded tail segment (NO_SEGMENT when none).
struct FinishRun {
    uint64_t segment = NO_SEGMENT;
    uint64_t first_cta = 0;  // tails of [first_cta, last_cta), then the head of last_cta
    uint64_t last_cta = 0;
};

THOR_RAGGED_EXEC_CHECK_DISABLE
template <typename TailOf>
THOR_RAGGED_HD FinishRun raggedFinishRun(uint64_t cta, uint64_t head_segment, const TailOf& tail_of) {
    uint64_t first = cta;
    while (first > 0 && tail_of(first - 1) == head_segment) --first;
    return FinishRun{head_segment, first, cta};
}

// The boundary-partial workspace: two records per CTA (head, then tail), each a segment id (8 bytes, NO_SEGMENT when
// absent) and `components` partials of `partial_bytes` (FP32 values: 4; FP8 / FP16 ARG words: 8; FP32 ARG key + tail:
// 16), each record padded to 16 bytes.
struct BoundaryLayout {
    uint64_t record_bytes = 0;
    uint64_t grid = 0;
    [[nodiscard]] THOR_RAGGED_HD uint64_t bytes() const { return 2 * grid * record_bytes; }
    [[nodiscard]] THOR_RAGGED_HD uint64_t headRecord(uint64_t cta) const { return (2 * cta) * record_bytes; }
    [[nodiscard]] THOR_RAGGED_HD uint64_t tailRecord(uint64_t cta) const { return (2 * cta + 1) * record_bytes; }
    // Partials start 16 bytes into a record (the segment id, padded), so 16-byte partials stay aligned.
    [[nodiscard]] THOR_RAGGED_HD uint64_t partialOffset(uint64_t record, uint64_t partial_bytes,
                                                        uint64_t component) const {
        return record + 16 + component * partial_bytes;
    }
};

THOR_RAGGED_HD BoundaryLayout boundaryLayout(uint64_t grid, uint64_t components, uint64_t partial_bytes) {
    const uint64_t raw = 16 + components * partial_bytes;
    return BoundaryLayout{(raw + 15) & ~uint64_t{15}, grid};
}

}  // namespace ThorImplementation::CubReductionInternal::RaggedMergePath
