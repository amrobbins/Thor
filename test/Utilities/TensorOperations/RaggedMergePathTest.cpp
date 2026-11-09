// 1264: the merge-path partition of ragged reductions (Utilities/TensorOperations/Cub/RaggedMergePath.h), host only.
// The search matches a linear merge at every diagonal (whole path and CTA windows, UINT32 and UINT64 offsets); CTA ranges
// split the path evenly; and an emulation of the two-pass reduction (main pass: complete segments plus head / tail
// partials; finisher: tails then the head, in CTA order) rebuilds every segment's exact row range, written exactly
// once, for every audit distribution, forced grids from 1 to more CTAs than events, clamped offsets past the capacity,
// no segments and no rows. Seeds are printed in the failure messages.
#include "Utilities/TensorOperations/Cub/RaggedMergePath.h"
#include "Utilities/TensorOperations/Cub/DenseReductionSelectionCache.h"
#include "benchmarks/RaggedReductionAuditCells.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using namespace ThorImplementation::CubReductionInternal::RaggedMergePath;
namespace A = ThorImplementation::RaggedReductionAudit;

namespace {

// Offsets stored at a runtime width, as the kernels read them.
struct StoredOffsets {
    std::vector<uint8_t> bytes;
    uint32_t width = 4;
    StoredOffsets(const std::vector<uint64_t>& values, uint32_t w) : bytes(values.size() * w), width(w) {
        for (size_t i = 0; i < values.size(); ++i) {
            if (w == 8) {
                std::memcpy(bytes.data() + 8 * i, &values[i], 8);
            } else {
                const uint32_t v = static_cast<uint32_t>(values[i]);
                std::memcpy(bytes.data() + 4 * i, &v, 4);
            }
        }
    }
    [[nodiscard]] Offsets view() const { return Offsets{bytes.data(), width}; }
};

// The coordinates of every diagonal 0..P by walking the merge one event at a time.
std::vector<Coordinate> linearPath(const Problem& p) {
    std::vector<Coordinate> path{Coordinate{0, 0}};
    Coordinate c;
    while (c.segment < p.segments || c.row < p.rows) {
        if (c.segment < p.segments && p.end(c.segment) <= c.row) ++c.segment;
        else ++c.row;
        path.push_back(c);
    }
    return path;
}

struct RowRange {
    uint64_t lo = 0, hi = 0;
    bool any = false;
};

// Concatenation is the strictest combine: it checks coverage and order at once.
bool append(RowRange& into, const RowRange& next) {
    if (!next.any) return true;
    if (!into.any) {
        into = next;
        return true;
    }
    if (into.hi != next.lo) return false;
    into.hi = next.hi;
    return true;
}

struct EmulationResult {
    std::string failure;  // empty when every check passed
};

// The two-pass reduction on row ranges: the main pass writes complete segments and records head / tail partials; the
// finisher combines each head's run. Every segment must come out as exactly [begin(s), end(s)), written once.
EmulationResult emulate(const Problem& p, uint64_t grid) {
    const auto fail = [&](const std::string& what) { return EmulationResult{what + " (grid " + std::to_string(grid) + ")"}; };
    std::vector<RowRange> out(p.segments);
    std::vector<int> written(p.segments, 0), row_seen(p.rows, 0), end_seen(p.segments, 0);
    std::vector<uint64_t> tail_segment(grid, NO_SEGMENT), head_segment(grid, NO_SEGMENT);
    std::vector<RowRange> tail_range(grid), head_range(grid);
    std::vector<int> tail_used(grid, 0);
    uint64_t min_events = UINT64_MAX, max_events = 0;
    for (uint64_t c = 0; c < grid; ++c) {
        const CtaRole role = raggedCtaRole(p, grid, c);
        const uint64_t events = (role.end.segment + role.end.row) - (role.begin.segment + role.begin.row);
        min_events = std::min(min_events, events);
        max_events = std::max(max_events, events);
        if (role.end.segment < role.begin.segment || role.end.row < role.begin.row) return fail("range runs backwards");
        for (uint64_t r = role.begin.row; r < role.end.row; ++r) ++row_seen[r];
        for (uint64_t s = role.begin.segment; s < role.end.segment; ++s) ++end_seen[s];
        for (uint64_t s = role.first_complete; s < role.end.segment; ++s) {
            if (p.begin(s) < role.begin.row || p.end(s) > role.end.row) return fail("complete segment leaves its CTA");
            out[s] = RowRange{p.begin(s), p.end(s), p.end(s) > p.begin(s)};
            ++written[s];
        }
        if (role.hasHead()) {
            if (role.head != role.begin.segment || role.head_rows_end > role.end.row) return fail("bad head");
            head_segment[c] = role.head;
            head_range[c] = RowRange{role.head_rows_begin, role.head_rows_end, role.head_rows_end > role.head_rows_begin};
        }
        if (role.hasTail()) {
            if (role.tail != role.end.segment) return fail("bad tail");
            tail_segment[c] = role.tail;
            tail_range[c] = RowRange{role.tail_rows_begin, role.end.row, true};
        }
    }
    if (p.pathLength() > 0 && max_events - min_events > 1) return fail("CTA event counts differ by more than one");
    // Empty CTAs only at the end (the finisher's walk back relies on it).
    bool seen_empty = false;
    for (uint64_t c = 0; c < grid; ++c) {
        const bool empty = ctaDiagonal(p.pathLength(), grid, c) == ctaDiagonal(p.pathLength(), grid, c + 1);
        if (seen_empty && !empty) return fail("an empty CTA before a non-empty one");
        seen_empty = seen_empty || empty;
    }
    // The finisher.
    const auto tail_of = [&](uint64_t c) { return tail_segment[c]; };
    for (uint64_t b = 0; b < grid; ++b) {
        if (head_segment[b] == NO_SEGMENT) continue;
        const FinishRun run = raggedFinishRun(b, head_segment[b], tail_of);
        if (run.first_cta == b) return fail("a head with no tails before it");
        RowRange total;
        for (uint64_t c = run.first_cta; c < run.last_cta; ++c) {
            ++tail_used[c];
            if (!append(total, tail_range[c])) return fail("tails out of order");
        }
        if (!append(total, head_range[b])) return fail("head out of order");
        out[run.segment] = total;
        ++written[run.segment];
    }
    for (uint64_t r = 0; r < p.rows; ++r)
        if (row_seen[r] != 1) return fail("row " + std::to_string(r) + " consumed " + std::to_string(row_seen[r]) + "x");
    for (uint64_t s = 0; s < p.segments; ++s) {
        if (end_seen[s] != 1) return fail("end of segment " + std::to_string(s) + " consumed " + std::to_string(end_seen[s]) + "x");
        if (written[s] != 1) return fail("segment " + std::to_string(s) + " written " + std::to_string(written[s]) + "x");
        const bool empty = p.begin(s) == p.end(s);
        if (empty ? out[s].any : (!out[s].any || out[s].lo != p.begin(s) || out[s].hi != p.end(s)))
            return fail("segment " + std::to_string(s) + " has the wrong rows");
    }
    for (uint64_t c = 0; c < grid; ++c)
        if (tail_segment[c] != NO_SEGMENT && tail_used[c] != 1) return fail("tail of CTA " + std::to_string(c) + " not finished once");
    return {};
}

std::vector<uint64_t> grids(uint64_t path, uint64_t seed) {
    std::vector<uint64_t> g{1, 2, 3, 7, 170, 340, path + 3};
    for (int i = 0; i < 4; ++i) g.push_back(1 + A::mix(seed + i) % (path + 2));
    return g;
}

}  // namespace

TEST(RaggedMergePath, SearchMatchesALinearMergeAtEveryDiagonal) {
    for (uint64_t trial = 0; trial < 300; ++trial) {
        const uint64_t seed = A::mix(1264 + trial);
        const uint64_t rows = seed % 120, segments = 1 + A::mix(seed) % 40;
        const A::Distribution d = A::DISTRIBUTIONS[trial % 5];
        const std::vector<uint64_t> values = A::makeOffsets(rows, segments, d, seed);
        for (uint32_t width : {4U, 8U}) {
            const StoredOffsets stored(values, width);
            const Problem p = makeProblem(stored.view(), segments, rows);
            ASSERT_EQ(p.rows, rows);
            const std::vector<Coordinate> path = linearPath(p);
            ASSERT_EQ(path.size(), p.pathLength() + 1);
            for (uint64_t diag = 0; diag <= p.pathLength(); ++diag) {
                const Coordinate c = searchPath(p, diag);
                ASSERT_EQ(c.segment, path[diag].segment) << "seed " << seed << " diagonal " << diag;
                ASSERT_EQ(c.row, path[diag].row) << "seed " << seed << " diagonal " << diag;
            }
            // A window between two path coordinates finds the same coordinates.
            const uint64_t a = A::mix(seed ^ 1) % (p.pathLength() + 1), b = A::mix(seed ^ 2) % (p.pathLength() + 1);
            const uint64_t lo = std::min(a, b), hi = std::max(a, b);
            for (uint64_t diag = lo; diag <= hi; ++diag) {
                const Coordinate c = mergePathSearch(diag, path[lo], path[hi], ProblemEnd{&p});
                ASSERT_EQ(c.segment, path[diag].segment) << "seed " << seed << " window " << lo << ".." << hi;
                ASSERT_EQ(c.row, path[diag].row) << "seed " << seed << " window " << lo << ".." << hi;
            }
        }
    }
}

// 1268: the weighted search holds exactly the events whose start cost (the cost of every event before them, a row
// `row`, an end `end`) is below the cost asked, at every cost; unit weights give the unweighted path.
TEST(RaggedMergePath, WeightedSearchHoldsTheEventsStartingBelowTheCost) {
    for (uint64_t trial = 0; trial < 200; ++trial) {
        const uint64_t seed = A::mix(1268 + trial);
        const uint64_t rows = seed % 120, segments = 1 + A::mix(seed) % 40;
        const std::vector<uint64_t> values = A::makeOffsets(rows, segments, A::DISTRIBUTIONS[trial % 5], seed);
        const StoredOffsets stored(values, trial % 2 ? 8U : 4U);
        const Problem p = makeProblem(stored.view(), segments, rows);
        const std::vector<Coordinate> path = linearPath(p);
        const PathWeights w = trial % 7 == 0 ? PathWeights{} : PathWeights{1 + A::mix(seed + 3) % 4, 1 + A::mix(seed + 4) % 13};
        std::vector<uint64_t> start_cost(path.size());  // start_cost[k]: the cost before event k (k = P: the total)
        for (size_t k = 1; k < path.size(); ++k)
            start_cost[k] = start_cost[k - 1] + (path[k].segment > path[k - 1].segment ? w.end : w.row);
        ASSERT_EQ(start_cost.back(), pathCost(p, w));
        size_t k = 0;  // events with start cost below c: path[k]
        for (uint64_t c = 0; c <= pathCost(p, w); ++c) {
            while (k < path.size() - 1 && start_cost[k] < c) ++k;
            const Coordinate got = searchPathCost(p, w, c);
            ASSERT_EQ(got.segment, path[k].segment) << "seed " << seed << " cost " << c;
            ASSERT_EQ(got.row, path[k].row) << "seed " << seed << " cost " << c;
        }
    }
}

TEST(RaggedMergePath, CtaDiagonalsSplitThePathEvenly) {
    for (uint64_t path : {0ULL, 1ULL, 5ULL, 1000ULL, 1ULL << 40}) {
        for (uint64_t grid : {1ULL, 3ULL, 170ULL, 1361ULL}) {
            EXPECT_EQ(ctaDiagonal(path, grid, 0), 0U);
            EXPECT_EQ(ctaDiagonal(path, grid, grid), path);
            for (uint64_t c = 0; c < std::min<uint64_t>(grid, 50); ++c) {
                const uint64_t events = ctaDiagonal(path, grid, c + 1) - ctaDiagonal(path, grid, c);
                EXPECT_TRUE(events == path / grid || events == path / grid + 1) << path << ' ' << grid << ' ' << c;
            }
        }
    }
}

TEST(RaggedMergePath, EmulatedTwoPassReductionRebuildsEverySegmentOnce) {
    for (uint64_t trial = 0; trial < 150; ++trial) {
        const uint64_t seed = A::mix(0x1264 + trial);
        const uint64_t rows = 1 + seed % 3000;
        const uint64_t segments = 1 + A::mix(seed) % (trial % 3 == 0 ? rows : 200);
        for (A::Distribution d : A::DISTRIBUTIONS) {
            const std::vector<uint64_t> values = A::makeOffsets(rows, segments, d, seed);
            const StoredOffsets stored(values, trial % 2 ? 8 : 4);
            const Problem p = makeProblem(stored.view(), segments, rows);
            for (uint64_t grid : grids(p.pathLength(), seed)) {
                const EmulationResult r = emulate(p, grid);
                ASSERT_TRUE(r.failure.empty()) << A::distributionName(d) << " seed " << seed << ": " << r.failure;
            }
        }
    }
}

TEST(RaggedMergePath, EveryRowItsOwnSegmentAndAllEmpty) {
    for (uint64_t rows : {1ULL, 2ULL, 97ULL}) {
        std::vector<uint64_t> ones(rows + 1);
        for (uint64_t i = 0; i <= rows; ++i) ones[i] = i;
        const StoredOffsets stored(ones, 4);
        const Problem p = makeProblem(stored.view(), rows, rows);
        for (uint64_t grid : grids(p.pathLength(), rows)) ASSERT_TRUE(emulate(p, grid).failure.empty()) << grid;
    }
    // No rows at all: every segment is empty and complete; no partials.
    const StoredOffsets zeros(std::vector<uint64_t>(65, 0), 8);
    const Problem p = makeProblem(zeros.view(), 64, 1000);
    EXPECT_EQ(p.rows, 0U);
    for (uint64_t grid : {1ULL, 5ULL, 64ULL, 200ULL}) {
        ASSERT_TRUE(emulate(p, grid).failure.empty()) << grid;
        for (uint64_t c = 0; c < grid; ++c) {
            const CtaRole role = raggedCtaRole(p, grid, c);
            EXPECT_FALSE(role.hasHead());
            EXPECT_FALSE(role.hasTail());
        }
    }
    // No segments.
    const StoredOffsets none(std::vector<uint64_t>{0}, 4);
    const Problem q = makeProblem(none.view(), 0, 1000);
    EXPECT_EQ(q.pathLength(), 0U);
    EXPECT_TRUE(emulate(q, 7).failure.empty());
}

TEST(RaggedMergePath, OffsetsPastTheCapacityAreClamped) {
    // Offsets run to 2x the capacity: the active rows stop at the capacity, later segments end there, and no row at
    // or past the capacity is consumed.
    for (uint64_t trial = 0; trial < 40; ++trial) {
        const uint64_t seed = A::mix(0xC1A3 + trial);
        const uint64_t capacity = 1 + seed % 500, segments = 1 + A::mix(seed) % 60;
        std::vector<uint64_t> values = A::makeOffsets(2 * capacity, segments, A::DISTRIBUTIONS[trial % 5], seed);
        const StoredOffsets stored(values, 8);
        const Problem p = makeProblem(stored.view(), segments, capacity);
        EXPECT_EQ(p.rows, capacity);
        for (uint64_t s = 0; s < segments; ++s) {
            EXPECT_LE(p.end(s), capacity);
            EXPECT_EQ(p.begin(s), std::min(values[s], capacity));
        }
        for (uint64_t grid : grids(p.pathLength(), seed)) ASSERT_TRUE(emulate(p, grid).failure.empty()) << seed;
    }
    // And a short active prefix inside a large capacity is not clamped.
    const StoredOffsets short_prefix(std::vector<uint64_t>{0, 3, 3, 10}, 4);
    EXPECT_EQ(makeProblem(short_prefix.view(), 3, 1 << 20).rows, 10U);
}

TEST(RaggedMergePath, BoundaryLayoutKeepsRecordsAndPartialsAligned) {
    for (uint64_t components : {1ULL, 3ULL, 16ULL, 255ULL}) {
        for (uint64_t partial_bytes : {4ULL, 8ULL, 16ULL}) {
            const BoundaryLayout layout = boundaryLayout(170, components, partial_bytes);
            EXPECT_EQ(layout.record_bytes % 16, 0U);
            EXPECT_GE(layout.record_bytes, 16 + components * partial_bytes);
            EXPECT_EQ(layout.bytes(), 2 * 170 * layout.record_bytes);
            EXPECT_EQ(layout.tailRecord(3) - layout.headRecord(3), layout.record_bytes);
            EXPECT_EQ(layout.headRecord(4), layout.tailRecord(3) + layout.record_bytes);
            EXPECT_EQ(layout.partialOffset(layout.tailRecord(5), partial_bytes, components - 1) + partial_bytes,
                      layout.tailRecord(5) + 16 + components * partial_bytes);
            EXPECT_EQ(layout.partialOffset(layout.headRecord(2), partial_bytes, 0) % 16, 0U);
        }
    }
}

TEST(RaggedMergePath, RaggedSelectionKindsAreReserved) {
    using ThorImplementation::CubReductionInternal::DenseReductionSelectionKind;
    using ThorImplementation::CubReductionInternal::denseReductionSelectionKindName;
    EXPECT_EQ(std::string(denseReductionSelectionKindName(DenseReductionSelectionKind::RaggedKR)), "ragged_kr");
    EXPECT_EQ(std::string(denseReductionSelectionKindName(DenseReductionSelectionKind::RaggedRK)), "ragged_rk");
    EXPECT_EQ(static_cast<size_t>(DenseReductionSelectionKind::RaggedRK) + 1,
              ThorImplementation::CubReductionInternal::DenseReductionSelectionKindCount);
}
