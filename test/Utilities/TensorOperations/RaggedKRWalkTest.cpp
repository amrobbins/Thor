// 1265 (1268, 1270): a host emulation of the ragged KR kernels (CubRaggedKRFp8.cu and CubRaggedFinish.cu), built on the
// same host / device functions they call (RaggedMergePath.h, RaggedKRWalk.h): the weighted CTA split found by half-warp
// rounds of 16 probes, one warp per CTA, the ends ring with its bulk loads and prefetch into freed slots, tiles over
// aligned spans capped at the last loaded end (each checked to lie on the path), the boundary bitmap, each lane's run
// and emitted values by rank, the warp's segmented scan (sequential here, same operator), one lane per end, the
// boundary records, and the finisher's scan. 1272: the search's last round with its candidates' bounds and the role
// from them, the one-pass fold (values written by rank as each lane folds, the first completed after the scan) and the
// piece-major chunk slots. 1273: one fold path per warp and piece, and the compact 16-byte records (no tail segment
// ids; the scan restarts at heads), checked against the walk back. 1274: the search from [0, segments) with its first
// round's offsets read raw, and the finisher's per-thread fold of four CTAs with its single scan. 1275: each end's kind
// found once, its value slot as the count of Ranked ends before it, in rounds of the warp's lanes, and the head as the
// CTA's first end; each slot checked against its bit's rank. Integer SUM and MIN
// must match direct per-segment results exactly, every segment written exactly once, for every audit distribution,
// small and full warps, rings from 8 to 256 ends, input misalignments, grids from 1 to more CTAs than events, four
// weight pairs, clamped offsets, zero rows, every row its own segment and long runs of empty segments. Host only.
// 1280: the components kernel (D = 2..15) and the finisher's CTA per component, emulated the same way
// (emulateComponentsKernels): workers of D lanes, tiles capped at raggedComponentEndsCap ends, values in slot x D, the
// scan across workers per component, workers ends per store round, every read inside the staged bytes; and the record
// arrays' layout. 1281: every unpredicated read of an active lane (all 16 rows) inside the span buffer. 1301: the ragged
// FP8 ARG keys (RaggedArg.h) against Fp8ArgKey.h's order, and the 16-row fold against a direct minimum. 1311: the FP16 /
// BF16 keys against F16ArgKey.h's order, and the 8-row fold. 1313: the FP32 keys against the ARG order, the 4-row fold,
// the pair records' layout and the local words' stores.
#include "Utilities/TensorOperations/Cub/F16ArgKey.h"
#include "Utilities/TensorOperations/Cub/RaggedArg.h"
#include "Utilities/TensorOperations/Cub/RaggedKRWalk.h"
#include "Utilities/TensorOperations/Cub/RaggedRKWalk.h"
#include "benchmarks/RaggedReductionAuditCells.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

using namespace ThorImplementation::CubReductionInternal::RaggedMergePath;
namespace A = ThorImplementation::RaggedReductionAudit;

namespace {

struct SumOp {
    int64_t identity() const { return 0; }
    int64_t operator()(int64_t a, int64_t b) const { return a + b; }
};
struct MinOp {
    int64_t identity() const { return std::numeric_limits<int64_t>::max(); }
    int64_t operator()(int64_t a, int64_t b) const { return std::min(a, b); }
};

int64_t rowValue(uint64_t row) { return static_cast<int64_t>(A::mix(row) % 1000) - 500; }

// raggedWarpSearchCost (probes = 32) and raggedHalfWarpSearchCost (probes = 16), lane by lane: rounds of parallel
// probes and ballots.
Coordinate warpSearchEmulated(const Problem& problem, const PathWeights& w, uint64_t cost, uint32_t probes = 32) {
    const CostSearchRange range = raggedCostSearchRange(problem, w, cost);
    uint64_t lo = range.lo, hi = range.hi;
    const auto consumed = [&](uint64_t x) { return w.end * x + w.row * problem.end(x) < cost; };
    while (hi - lo > probes) {
        const uint64_t step = (hi - lo + probes) / (probes + 1);
        std::vector<uint64_t> probe(probes);
        uint32_t count = 0;
        bool prefix = true;
        for (uint32_t lane = 0; lane < probes; ++lane) {
            probe[lane] = std::min(lo + step * (lane + 1), hi - 1);
            const bool c = consumed(probe[lane]);
            if (c && !prefix) ADD_FAILURE() << "consumed probes are not a prefix";
            if (c) ++count;
            else prefix = false;
        }
        const uint64_t first_false = probe[count < probes ? count : probes - 1];
        const uint64_t last_true = probe[count > 0 ? count - 1 : 0];
        if (count > 0) lo = last_true + 1;
        if (count < probes) hi = first_false;
    }
    uint64_t below = 0;
    for (uint32_t lane = 0; lane < probes; ++lane)
        if (lo + lane < hi && consumed(lo + lane)) ++below;
    return Coordinate{lo + below, raggedRowsAtCost(problem, w, cost, lo + below)};
}

// 1272: raggedHalfWarpSearchCost, one half: rounds of 16 probes down to 15 candidates, then one round reading every
// candidate's begin and end; the coordinate's rows and its segment's bounds come from the found candidate's lane.
// 1274: from [0, segments), the first round's offsets read raw before the row count (raggedSearchRound0Probe) and
// clamped after; with at most 15 segments that round is the last.
SearchedCoordinate halfWarpSearchEmulated(const Problem& problem, const PathWeights& w, uint64_t cost) {
    const uint64_t segments = problem.segments;
    uint64_t raw_begin[16] = {}, raw_end[16] = {};
    for (uint32_t slot = 0; slot < 16; ++slot) {
        const uint64_t x = raggedSearchRound0Probe(segments, slot);
        if (x <= segments) raw_begin[slot] = loadOffset(problem.offsets, x);
        if (x < segments) raw_end[slot] = loadOffset(problem.offsets, x + 1);
    }
    uint64_t lo = 0, hi = segments;
    const auto below = [&](uint64_t x, uint64_t end_x) { return w.end * x + w.row * end_x < cost; };
    const auto narrow = [&](const uint64_t (&probe)[16], const bool (&consumed)[16]) {
        uint32_t count = 0;
        bool prefix = true;
        for (uint32_t slot = 0; slot < 16; ++slot) {
            if (consumed[slot] && !prefix) ADD_FAILURE() << "consumed probes are not a prefix";
            if (consumed[slot]) ++count;
            else prefix = false;
        }
        const uint64_t first_false = probe[count < 16 ? count : 15];
        const uint64_t last_true = probe[count > 0 ? count - 1 : 0];
        if (count > 0) lo = last_true + 1;
        if (count < 16) hi = first_false;
    };
    if (segments > 15) {
        uint64_t probe[16];
        bool consumed[16];
        for (uint32_t slot = 0; slot < 16; ++slot) {
            probe[slot] = raggedSearchRound0Probe(segments, slot);
            consumed[slot] = below(probe[slot], clampRows(raw_end[slot], problem.rows));
        }
        narrow(probe, consumed);
    }
    while (hi - lo > 15) {
        const uint64_t step = (hi - lo + 16) / 17;
        uint64_t probe[16];
        bool consumed[16];
        for (uint32_t slot = 0; slot < 16; ++slot) {
            probe[slot] = std::min(lo + step * (slot + 1), hi - 1);
            consumed[slot] = below(probe[slot], problem.end(probe[slot]));
        }
        narrow(probe, consumed);
    }
    uint64_t lane_begin[16], lane_end[16];
    uint32_t count = 0;
    for (uint32_t slot = 0; slot < 16; ++slot) {
        const uint64_t x = lo + slot;
        if (segments <= 15) {
            if (raggedSearchRound0Probe(segments, slot) != x) ADD_FAILURE() << "the first round is not the last's slots";
            lane_begin[slot] = x <= segments ? clampRows(raw_begin[slot], problem.rows) : problem.rows;
            lane_end[slot] = x < segments ? clampRows(raw_end[slot], problem.rows) : problem.rows;
        } else {
            lane_begin[slot] = x <= segments ? problem.begin(x) : problem.rows;
            lane_end[slot] = x < segments ? problem.end(x) : problem.rows;
        }
        if (x < hi && below(x, lane_end[slot])) ++count;
    }
    if (count > 15) ADD_FAILURE() << "the last round's answer is past its lanes";
    const uint64_t segment = lo + count;
    const uint64_t begin = lane_begin[count], end = lane_end[count];
    return SearchedCoordinate{Coordinate{segment, raggedRowsAtCostBounded(w, cost, segment, segment > 0 ? begin : 0, end)},
                              begin, end};
}

struct Record {
    uint64_t segment = NO_SEGMENT;
    int64_t value = 0;
};

struct Geometry {
    uint32_t lanes = 32;           // lanes of the CTA's warp
    uint32_t ring = 256;           // ends ring slots (a power of two, a multiple of lanes)
    uint64_t byte_head = 0;        // the input's misalignment in elements (rows are staged in aligned chunks of Rows)
};

// The two kernels, emulated (1270: one warp per CTA, row-major tiles over aligned spans, the boundary bitmap, ranked
// values, one lane per end; the weighted CTA split; bulk and prefetched ends). Returns the failure, or an empty string.
template <typename Op, uint32_t Rows>
std::string emulateKernels(const Problem& problem, uint64_t grid, const Geometry& g, const PathWeights& requested,
                           std::vector<int64_t>& out, std::vector<int>& written) {
    const Op op{};
    const int64_t identity = op.identity();
    const uint32_t tile_rows = g.lanes * Rows, mask = g.ring - 1;
    if ((g.ring & mask) != 0 || g.ring % g.lanes != 0) return "bad geometry";
    out.assign(problem.segments, identity);
    written.assign(problem.segments, 0);
    std::vector<Record> heads(grid), tails(grid);  // the roles' partials, for the cross-check below
    std::vector<uint64_t> keys(grid);                // 1273: the compact records the kernel writes
    std::vector<int64_t> head_values(grid), tail_values(grid);
    const PathWeights w = raggedEffectiveWeights(problem, requested, grid);
    const uint64_t cost = pathCost(problem, w);
    bool seen_empty = false;
    for (uint64_t cta = 0; cta < grid; ++cta) {
        const SearchedCoordinate first = halfWarpSearchEmulated(problem, w, ctaDiagonal(cost, grid, cta));
        const SearchedCoordinate last = halfWarpSearchEmulated(problem, w, ctaDiagonal(cost, grid, cta + 1));
        const Coordinate cta_begin = first.at, cta_end = last.at;
        if (cta_end.segment < cta_begin.segment || cta_end.row < cta_begin.row) return "CTA coordinates go backwards";
        const bool empty = cta_end.segment == cta_begin.segment && cta_end.row == cta_begin.row;
        if (!empty && seen_empty) return "an empty CTA before a CTA with events";
        seen_empty = seen_empty || empty;
        // 1272: the role from the searched bounds, as the kernel; it must match the one read from the offsets.
        const CtaRole role = raggedCtaRoleFromBounds(problem.segments, first, last);
        const CtaRole read = raggedCtaRoleFromCoordinates(problem, cta_begin, cta_end);
        if (role.head != read.head || role.tail != read.tail || role.head_rows_end != read.head_rows_end ||
            role.tail_rows_begin != read.tail_rows_begin || role.first_complete != read.first_complete)
            return "the role from the searched bounds differs";
        const uint64_t row_base = cta_begin.row, seg_base = cta_begin.segment;
        const uint32_t seg_count = static_cast<uint32_t>(cta_end.segment - seg_base);
        const uint32_t row_count = static_cast<uint32_t>(cta_end.row - row_base);
        const uint32_t end_diagonal = seg_count + row_count;
        const uint64_t byte_base = g.byte_head + row_base;
        const uint64_t first_begin = seg_count > 0 ? first.begin : 0;
        const uint64_t first_end = seg_count > 0 ? first.end : 0;
        const bool first_has_rows = first_begin < first_end;
        const uint64_t first_count = first_end - first_begin;
        std::vector<uint32_t> ring(g.ring, 0xDEADBEEFU);  // garbage until loaded
        const auto at = [&](uint32_t s) -> uint32_t { return ring[raggedRingSlot(s, mask)]; };
        const auto end_rel = [&](uint32_t s) { return static_cast<uint32_t>(problem.end(seg_base + s) - row_base); };
        const auto load = [&](uint32_t first, uint32_t count) {
            for (uint32_t i = 0; i < count; ++i) ring[raggedRingSlot(first + i, mask)] = end_rel(first + i);
        };
        uint32_t x0 = 0, y0 = 0, held = raggedEndsToLoad(0, g.ring, seg_count), last_end = 0;
        load(0, held);
        int64_t carry = identity;
        heads[cta].segment = NO_SEGMENT;
        while (x0 + y0 < end_diagonal) {
            if (x0 + held < seg_count && held <= g.ring / 2) {
                const uint32_t count = raggedEndsToLoad(held, g.ring, seg_count - (x0 + held));
                load(x0 + held, count);
                held += count;
            }
            const bool all_loaded = x0 + held == seg_count;
            const uint32_t shift = static_cast<uint32_t>((byte_base + y0) & (Rows - 1));
            const uint32_t row_limit = std::min(y0 + tile_rows - shift, row_count);
            uint32_t ends_within = 0;
            for (uint32_t i = 0; i < held; ++i) ends_within += at(x0 + i) <= row_limit ? 1 : 0;
            const uint32_t last_loaded = held > 0 ? at(x0 + held - 1) : 0;
            if (held == 0 && !all_loaded) return "a tile with no ends loaded";
            const Coordinate32 tile_end = raggedTileEnd(x0, held, all_loaded, last_loaded, row_limit, ends_within);
            const uint32_t n = tile_end.segment - x0, y1 = tile_end.row;
            if (n > held) return "the tile consumes an end that is not loaded";
            if (y1 < y0 || y1 - y0 > tile_rows - shift) return "the tile's rows leave its span";
            if (n == 0 && y1 == y0) return "the tile makes no progress";
            const Coordinate on_path = searchPath(problem, seg_base + row_base + tile_end.segment + y1);
            if (on_path.segment != seg_base + tile_end.segment || on_path.row != row_base + y1)
                return "the tile end is not on the path";
            // The prefetch: the ends after the loaded ones, into the slots this tile frees.
            const uint32_t prefetch_first = x0 + held;
            const uint32_t prefetch_count = raggedEndsToLoad(held - n, g.ring, seg_count - prefetch_first);
            if (prefetch_count > g.ring) return "the prefetch overflows the lanes' registers";
            std::vector<uint32_t> prefetched(prefetch_count);
            for (uint32_t i = 0; i < prefetch_count; ++i) prefetched[i] = end_rel(prefetch_first + i);
            // The bits.
            std::vector<uint32_t> bits((tile_rows + 31) / 32, 0);
            // 1275: each end's kind once, from its row and the end before it (the kernel's lane below).
            std::vector<RaggedEndKind> kinds(n);
            const auto end_row = [&](uint32_t i) { return at(x0 + i); };
            const auto previous_end = [&](uint32_t i) { return i > 0 ? at(x0 + i - 1) : last_end; };
            for (uint32_t i = 0; i < n; ++i) {
                const uint32_t e = end_row(i);
                kinds[i] = raggedEndKindAfter(x0 + i == 0, first_has_rows, e, previous_end(i), y0);
                if (kinds[i] == RaggedEndKind::Ranked) {
                    const uint32_t j = shift + (e - 1 - y0);
                    if (j < shift || j >= shift + (y1 - y0)) return "a bit outside the tile's rows";
                    if (bits[j >> 5] & (1U << (j & 31U))) return "two segments end at one row";
                    bits[j >> 5] |= 1U << (j & 31U);
                }
            }
            // The lanes.
            const uint32_t lane_mask = Rows >= 32 ? 0xffffffffU : ((1U << (Rows & 31U)) - 1U);
            // 1272: as the kernel, one pass: the ranks first, then each lane's rows in pieces of 16 (a tree fold where
            // no end falls), each ended segment's value written to its rank as the lane folds; the warp scan, then
            // each lane's first value completed with the run arriving at it. 1284: one piece of Rows (16 FP8, 8 FP16 /
            // BF16, 4 FP32 rows per lane), the kernel's one chunk per lane; 32 keeps its two pieces of 16.
            constexpr uint32_t Piece = Rows < 16 ? Rows : 16, Pieces = Rows / Piece;
            constexpr uint32_t PieceMask = (1U << Piece) - 1U;
            std::vector<uint32_t> lane_bits(g.lanes), rank_base(g.lanes);
            std::vector<std::vector<int64_t>> values(g.lanes, std::vector<int64_t>(Rows, identity));
            std::vector<Carry<int64_t>> runs(g.lanes);
            std::vector<int64_t> slot(g.ring, 0x5EED);
            uint32_t ranks = 0;
            uint32_t any_piece_bits = 0;  // 1273: bit p when some lane has an end in its piece p (the warp's path)
            for (uint32_t lane = 0; lane < g.lanes; ++lane) {
                lane_bits[lane] = (bits[(Rows * lane) >> 5] >> ((Rows * lane) & 31U)) & lane_mask;
                rank_base[lane] = ranks;
                ranks += raggedBitsBelow(lane_bits[lane], 32);
                for (uint32_t piece = 0; piece < Pieces; ++piece)
                    if ((lane_bits[lane] >> (Piece * piece)) & PieceMask) any_piece_bits |= 1U << piece;
            }
            if (ranks > g.ring) return "more ranked values than slots";
            for (uint32_t lane = 0; lane < g.lanes; ++lane) {
                for (uint32_t k = 0; k < Rows; ++k) {
                    const uint32_t local = Rows * lane + k;
                    if (local >= shift && local < shift + (y1 - y0))
                        values[lane][k] = rowValue(row_base + y0 + (local - shift));
                }
                int64_t acc = identity;
                for (uint32_t piece = 0; piece < Pieces; ++piece) {
                    const auto value = [&](uint32_t k) { return values[lane][Piece * piece + k]; };
                    const uint32_t piece_bits = (lane_bits[lane] >> (Piece * piece)) & PieceMask;
                    if (((any_piece_bits >> piece) & 1U) == 0) {
                        const int64_t tree = raggedTreeFold<Piece, int64_t>(op, value);
                        acc = piece == 0 ? tree : op(acc, tree);
                    } else {
                        acc = raggedLaneEmit<Piece>(piece_bits, acc, identity, op, value, [&](uint32_t k, int64_t v) {
                            slot[rank_base[lane] + raggedBitsBelow(lane_bits[lane], Piece * piece + k)] = v;
                        });
                    }
                }
                runs[lane] = Carry<int64_t>{lane_bits[lane] != 0, acc};
            }
            const bool reset_at_start = n > 0 && at(x0) == y0;
            Carry<int64_t> running{false, reset_at_start ? identity : carry};
            for (uint32_t lane = 0; lane < g.lanes; ++lane) {
                if (lane_bits[lane] != 0)
                    slot[rank_base[lane]] = raggedLaneFirstValue(running.value, slot[rank_base[lane]], op);
                running = raggedCarryCombine(running, runs[lane], op);
            }
            // One lane per end, in rounds of the warp's lanes. 1275: a Ranked end's slot is the count of Ranked ends
            // before it (each round's ballot after the rounds before); checked against the slot of its bit, which
            // 1270-1274 read.
            std::vector<uint32_t> slots(n, 0);
            uint32_t ranked_earlier = 0;
            for (uint32_t round = 0; round * g.lanes < n; ++round) {
                uint32_t ballot = 0;
                for (uint32_t lane = 0; lane < g.lanes && round * g.lanes + lane < n; ++lane)
                    if (kinds[round * g.lanes + lane] == RaggedEndKind::Ranked) ballot |= 1U << lane;
                for (uint32_t lane = 0; lane < g.lanes && round * g.lanes + lane < n; ++lane)
                    slots[round * g.lanes + lane] = raggedRankedSlot(ranked_earlier, ballot, lane);
                // The kernel keeps round 0's in one register (raggedPackEndFacts).
                if (round == 0)
                    for (uint32_t lane = 0; lane < g.lanes && lane < n; ++lane) {
                        const uint32_t packed = raggedPackEndFacts(kinds[lane], slots[lane]);
                        if (raggedPackedKind(packed) != kinds[lane] || raggedPackedSlot(packed) != slots[lane])
                            return "round 0's packed facts do not round-trip";
                    }
                ranked_earlier += raggedBitsBelow(ballot, 32);
            }
            for (uint32_t i = 0; i < n; ++i) {
                const uint32_t e = end_row(i), previous = previous_end(i);
                const RaggedEndKind kind = kinds[i];
                int64_t v = identity;
                if (kind == RaggedEndKind::Ranked) {
                    const uint32_t j = shift + (e - 1 - y0);
                    if (slots[i] != rank_base[j / Rows] + raggedBitsBelow(lane_bits[j / Rows], j % Rows))
                        return "an end's slot differs from its bit's rank";
                    v = slot[slots[i]];
                } else if (kind == RaggedEndKind::CarryIn) {
                    v = carry;
                }
                const uint64_t s = seg_base + x0 + i;
                const uint64_t count = x0 + i == 0 ? first_count : e - previous;
                if (count != problem.end(s) - problem.begin(s)) return "a MEAN count differs";
                if ((x0 + i == 0 && role.head != NO_SEGMENT) != (s == role.head))
                    return "the head is not the CTA's first end";
                if (x0 + i == 0 && role.head != NO_SEGMENT) {  // 1275: as the kernel tests it
                    heads[cta].value = v;
                } else {
                    out[s] = v;
                    ++written[s];
                }
            }
            if (n > 0) last_end = at(x0 + n - 1);
            carry = running.value;
            for (uint32_t i = 0; i < prefetch_count; ++i) {
                const uint32_t slot_index = raggedRingSlot(prefetch_first + i, mask);
                for (uint32_t s = x0 + n; s < x0 + held; ++s)
                    if (raggedRingSlot(s, mask) == slot_index) return "the prefetch overwrites a held end";
                ring[slot_index] = prefetched[i];
            }
            held = held - n + prefetch_count;
            x0 = tile_end.segment;
            y0 = y1;
        }
        heads[cta].segment = role.head;
        tails[cta] = Record{role.tail, carry};
        keys[cta] = role.head;
        head_values[cta] = role.head != NO_SEGMENT ? heads[cta].value : identity;
        tail_values[cta] = role.tail != NO_SEGMENT ? carry : identity;
    }
    // The finisher (1273, 1274): the compact records' scan in CTA order, as the kernel groups it (each of 1024 threads
    // folds four consecutive CTAs, an exclusive scan over the threads, then each CTA's arriving run), each head taking
    // the run arriving at its CTA. Cross-checked against the roles' rule: the tails before the head of the head's
    // segment, walking back.
    // The kernel holds 4096 CTAs; larger emulated grids (more CTAs than events) take more threads of the same shape.
    const uint64_t per_thread = 4, finish_threads = std::max<uint64_t>(1024, (grid + per_thread - 1) / per_thread);
    const auto element = [&](uint64_t c) {
        return c < grid ? raggedRecordElement(keys[c], tail_values[c]) : Carry<int64_t>{false, identity};
    };
    std::vector<int64_t> arriving_at(finish_threads * per_thread, identity);
    Carry<int64_t> scan{false, identity};  // the exclusive scan over the threads' folds
    for (uint64_t t = 0; t < finish_threads; ++t) {
        Carry<int64_t> mine{false, identity}, arriving = scan;
        for (uint64_t i = 0; i < per_thread; ++i) mine = raggedCarryCombine(mine, element(per_thread * t + i), op);
        for (uint64_t i = 0; i < per_thread; ++i) {
            arriving_at[per_thread * t + i] = arriving.value;
            arriving = raggedCarryCombine(arriving, element(per_thread * t + i), op);
        }
        scan = raggedCarryCombine(scan, mine, op);
    }
    for (uint64_t c = 0; c < grid; ++c) {
        if (!raggedRecordHasHead(keys[c])) continue;
        const uint64_t s = keys[c];
        if (s != heads[c].segment) return "the record's head differs from the role's";
        int64_t walked = identity;
        for (uint64_t b = c; b > 0 && tails[b - 1].segment == s; --b) walked = op(tails[b - 1].value, walked);
        if (walked != arriving_at[c]) return "the scan's run differs from the walk back over the tails";
        out[s] = op(arriving_at[c], head_values[c]);
        ++written[s];
    }
    return {};
}

template <typename Op, uint32_t Rows = 16>
void expectExact(const Problem& problem, uint64_t grid, const Geometry& g, const std::string& what,
                 const PathWeights& weights = PathWeights{}) {
    std::vector<int64_t> out;
    std::vector<int> written;
    const std::string failure = emulateKernels<Op, Rows>(problem, grid, g, weights, out, written);
    ASSERT_TRUE(failure.empty()) << what << ": " << failure;
    const Op op{};
    for (uint64_t s = 0; s < problem.segments; ++s) {
        int64_t expected = op.identity();
        for (uint64_t r = problem.begin(s); r < problem.end(s); ++r) expected = op(expected, rowValue(r));
        ASSERT_EQ(written[s], 1) << what << " segment " << s;
        ASSERT_EQ(out[s], expected) << what << " segment " << s;
    }
}

// 1280: component c of row r (the components kernel's input).
int64_t componentValue(uint64_t row, uint32_t c) { return rowValue(row * 31 + c); }

// 1280: the components kernel (raggedKRComponentsKernel) and the finisher's one CTA per component, emulated: the
// D = 1 kernel's walk with tiles of Rows x workers rows (1284: Rows = 16 / element bytes; the input's elements of
// `element_bytes` bytes, `byte_head` a multiple of them) and at most raggedComponentEndsCap ends, a lane per component of
// each worker, the ranks counted once per worker, values in slot * D + component, the scan across workers per
// component, `workers` ends per store round, and the records as raggedRecordLayout's arrays. `out` is segments x D.
template <typename Op, uint32_t ElementBytes = 1>
std::string emulateComponentsKernels(const Problem& problem, uint64_t grid, uint32_t ring_slots, uint32_t D,
                                     uint64_t byte_head, const PathWeights& requested, std::vector<int64_t>& out,
                                     std::vector<int>& written) {
    const Op op{};
    const int64_t identity = op.identity();
    if (D < 2 || D > raggedKRMaxComponents(ElementBytes)) return "bad D";
    if (byte_head % ElementBytes != 0 || byte_head > 15) return "bad head";
    const uint32_t workers = raggedComponentWorkers(D), tile_rows = raggedComponentTileRows(D, ElementBytes);
    const uint32_t cap = raggedComponentEndsCap(D, ring_slots), mask = ring_slots - 1;
    constexpr uint32_t Rows = 16 / ElementBytes, SpanBytes = 16 * 33;
    constexpr uint32_t WorkerMask = (1U << Rows) - 1U;
    static_assert(Rows * ElementBytes == 16 && Rows <= 16);
    const uint32_t row_bytes = D * ElementBytes;
    if (workers * D > 32 || tile_rows * row_bytes > 512 || tile_rows > 256 || tile_rows != Rows * workers)
        return "the tile does not fit the kernel's buffers";
    out.assign(problem.segments * D, identity);
    written.assign(problem.segments * D, 0);
    std::vector<Record> tails(grid);
    std::vector<uint64_t> keys(grid);
    std::vector<std::vector<int64_t>> head_values(D, std::vector<int64_t>(grid, identity)),
        tail_values(D, std::vector<int64_t>(grid, identity)), tails_by_component(D, std::vector<int64_t>(grid, identity));
    const PathWeights w = raggedEffectiveWeights(problem, requested, grid);
    const uint64_t cost = pathCost(problem, w);
    for (uint64_t cta = 0; cta < grid; ++cta) {
        const SearchedCoordinate first = halfWarpSearchEmulated(problem, w, ctaDiagonal(cost, grid, cta));
        const SearchedCoordinate last = halfWarpSearchEmulated(problem, w, ctaDiagonal(cost, grid, cta + 1));
        const Coordinate cta_begin = first.at, cta_end = last.at;
        const CtaRole role = raggedCtaRoleFromBounds(problem.segments, first, last);
        const uint64_t row_base = cta_begin.row, seg_base = cta_begin.segment;
        const uint32_t seg_count = static_cast<uint32_t>(cta_end.segment - seg_base);
        const uint32_t row_count = static_cast<uint32_t>(cta_end.row - row_base);
        const uint32_t end_diagonal = seg_count + row_count;
        const uint64_t byte_base = byte_head + row_base * row_bytes;
        const uint64_t first_begin = seg_count > 0 ? first.begin : 0;
        const uint64_t first_end = seg_count > 0 ? first.end : 0;
        const bool first_has_rows = first_begin < first_end;
        const uint64_t first_count = first_end - first_begin;
        std::vector<uint32_t> ring(ring_slots, 0xDEADBEEFU);
        const auto at = [&](uint32_t s) -> uint32_t { return ring[raggedRingSlot(s, mask)]; };
        const auto end_rel = [&](uint32_t s) { return static_cast<uint32_t>(problem.end(seg_base + s) - row_base); };
        const auto load = [&](uint32_t first_index, uint32_t count) {
            for (uint32_t i = 0; i < count; ++i) ring[raggedRingSlot(first_index + i, mask)] = end_rel(first_index + i);
        };
        // The bytes a fetch at row y stages (shift plus its rows' bytes), as the kernel's fetch_rows.
        const auto fetched_bytes = [&](uint32_t y) {
            const uint32_t shift = static_cast<uint32_t>((byte_base + uint64_t{y} * row_bytes) & 15U);
            return shift + (std::min(y + tile_rows, row_count) - y) * row_bytes;
        };
        uint32_t x0 = 0, y0 = 0, held = raggedEndsToLoad(0, ring_slots, seg_count), last_end = 0;
        load(0, held);
        uint32_t staged = row_count > 0 ? fetched_bytes(0) : 0;
        std::vector<int64_t> carry(D, identity), head(D, identity);
        while (x0 + y0 < end_diagonal) {
            if (x0 + held < seg_count && held <= ring_slots / 2) {
                const uint32_t count = raggedEndsToLoad(held, ring_slots, seg_count - (x0 + held));
                load(x0 + held, count);
                held += count;
            }
            const bool all_loaded = x0 + held == seg_count;
            const uint32_t row_limit = std::min(y0 + tile_rows, row_count);
            uint32_t ends_within = 0;
            for (uint32_t i = 0; i < held; ++i) ends_within += at(x0 + i) <= row_limit ? 1 : 0;
            const RaggedTileEnds capped = raggedCapTileEnds(held, all_loaded, ends_within, cap);
            if (capped.held == 0 && !capped.all_loaded) return "a tile with no ends loaded";
            const uint32_t last_loaded = capped.held > 0 ? at(x0 + capped.held - 1) : 0;
            const Coordinate32 tile_end =
                raggedTileEnd(x0, capped.held, capped.all_loaded, last_loaded, row_limit, capped.ends_within);
            const uint32_t n = tile_end.segment - x0, y1 = tile_end.row;
            if (n > cap) return "the tile consumes more ends than its cap";
            if (n > held) return "the tile consumes an end that is not loaded";
            if (y1 < y0 || y1 - y0 > tile_rows) return "the tile's rows leave its span";
            if (n == 0 && y1 == y0) return "the tile makes no progress";
            const Coordinate on_path = searchPath(problem, seg_base + row_base + tile_end.segment + y1);
            if (on_path.segment != seg_base + tile_end.segment || on_path.row != row_base + y1)
                return "the tile end is not on the path";
            const uint32_t prefetch_first = x0 + held;
            const uint32_t prefetch_count = raggedEndsToLoad(held - n, ring_slots, seg_count - prefetch_first);
            std::vector<uint32_t> prefetched(prefetch_count);
            for (uint32_t i = 0; i < prefetch_count; ++i) prefetched[i] = end_rel(prefetch_first + i);
            const uint32_t next_staged = y1 < row_count ? fetched_bytes(y1) : 0;
            if (next_staged > SpanBytes) return "a fetch overflows the span buffer";

            // 4. The bits and every end's facts, in rounds of 32 ends.
            std::vector<uint32_t> bits(16 * Rows / 32, 0);  // the kernel's BitWords (at most 16 workers)
            std::vector<uint32_t> facts(n);
            const auto previous_end = [&](uint32_t i) { return i > 0 ? at(x0 + i - 1) : last_end; };
            uint32_t ranked_earlier = 0;
            for (uint32_t round = 0; round * 32 < n; ++round) {
                uint32_t ballot = 0;
                std::vector<RaggedEndKind> kinds(32, RaggedEndKind::Empty);
                for (uint32_t lane = 0; lane < 32 && round * 32 + lane < n; ++lane) {
                    const uint32_t i = round * 32 + lane, e = at(x0 + i);
                    kinds[lane] = raggedEndKindAfter(x0 + i == 0, first_has_rows, e, previous_end(i), y0);
                    if (kinds[lane] == RaggedEndKind::Ranked) {
                        ballot |= 1U << lane;
                        const uint32_t j = e - 1 - y0;
                        if (j >= y1 - y0) return "a bit outside the tile's rows";
                        if (bits[j >> 5] & (1U << (j & 31U))) return "two segments end at one row";
                        bits[j >> 5] |= 1U << (j & 31U);
                    }
                }
                for (uint32_t lane = 0; lane < 32 && round * 32 + lane < n; ++lane) {
                    const uint32_t slot = raggedRankedSlot(ranked_earlier, ballot, lane);
                    if (slot > 0xffU) return "a slot does not fit the 16-bit facts";
                    facts[round * 32 + lane] = raggedPackEndFacts16(kinds[lane], slot);
                    if (raggedPacked16Kind(facts[round * 32 + lane]) != kinds[lane] ||
                        raggedPacked16Slot(facts[round * 32 + lane]) != slot)
                        return "the 16-bit facts do not round-trip";
                }
                ranked_earlier += raggedBitsBelow(ballot, 32);
            }

            // 5. Each worker's rows, a lane per component.
            const uint32_t rows_in_tile = y1 - y0;
            const uint32_t shift = static_cast<uint32_t>((byte_base + uint64_t{y0} * row_bytes) & 15U);
            if (shift % ElementBytes != 0) return "a misaligned element";
            std::vector<uint32_t> worker_bits(workers), rank_base(workers);
            uint32_t ranks = 0;
            bool any_bits = false;
            for (uint32_t wk = 0; wk < workers; ++wk) {
                const uint32_t first_row = Rows * wk;
                worker_bits[wk] = (bits[first_row >> 5] >> (first_row & 31U)) & WorkerMask;
                rank_base[wk] = ranks;
                ranks += raggedBitsBelow(worker_bits[wk], 32);
                any_bits = any_bits || worker_bits[wk] != 0;
            }
            std::vector<int64_t> slot(RAGGED_KR_COMPONENT_VALUE_SLOTS, 0x5EED);
            std::vector<std::vector<Carry<int64_t>>> runs(D, std::vector<Carry<int64_t>>(workers));
            for (uint32_t wk = 0; wk < workers; ++wk) {
                const uint32_t first_row = Rows * wk;
                const uint32_t rows_valid =
                    first_row >= rows_in_tile ? 0 : std::min(Rows, rows_in_tile - first_row);
                for (uint32_t c = 0; c < D; ++c) {
                    int64_t v[Rows];
                    for (uint32_t k = 0; k < Rows; ++k) {
                        // 1281: the kernel reads all Rows rows unpredicated (masked after the decode): inside the buffer.
                        const uint32_t byte = shift + (first_row + k) * row_bytes + c * ElementBytes;
                        if (byte + ElementBytes > SpanBytes) return "an unpredicated read leaves the span buffer";
                        v[k] = identity;
                        if (k < rows_valid) {
                            if (byte + ElementBytes > staged) return "a lane reads a byte the fetch did not stage";
                            v[k] = componentValue(row_base + y0 + first_row + k, c);
                        }
                    }
                    const auto value = [&](uint32_t k) { return v[k]; };
                    int64_t acc;
                    if (!any_bits) {
                        acc = raggedTreeFold<Rows, int64_t>(op, value);
                    } else {
                        bool overflow = false;
                        acc = raggedLaneEmit<Rows>(worker_bits[wk], identity, identity, op, value,
                                                   [&](uint32_t k, int64_t sv) {
                                                       const uint32_t index =
                                                           (rank_base[wk] + raggedBitsBelow(worker_bits[wk], k)) * D + c;
                                                       if (index >= slot.size()) overflow = true;
                                                       else slot[index] = sv;
                                                   });
                        if (overflow) return "a value slot past the tile's slots";
                    }
                    runs[c][wk] = Carry<int64_t>{worker_bits[wk] != 0, acc};
                }
            }
            const bool reset_at_start = n > 0 && at(x0) == y0;
            std::vector<int64_t> next_carry(D);
            for (uint32_t c = 0; c < D; ++c) {
                Carry<int64_t> running{false, reset_at_start ? identity : carry[c]};
                for (uint32_t wk = 0; wk < workers; ++wk) {
                    if (worker_bits[wk] != 0) {
                        const uint32_t index = rank_base[wk] * D + c;
                        slot[index] = raggedLaneFirstValue(running.value, slot[index], op);
                    }
                    running = raggedCarryCombine(running, runs[c][wk], op);
                }
                next_carry[c] = running.value;
            }

            // 6. `workers` ends per round, one lane per component.
            for (uint32_t round_first = 0; round_first < n; round_first += workers) {
                for (uint32_t wk = 0; wk < workers && round_first + wk < n; ++wk) {
                    const uint32_t i = round_first + wk, e = at(x0 + i), previous = previous_end(i);
                    const RaggedEndKind kind = raggedPacked16Kind(facts[i]);
                    if (kind == RaggedEndKind::Ranked) {
                        const uint32_t j = e - 1 - y0;
                        if (raggedPacked16Slot(facts[i]) != rank_base[j / Rows] + raggedBitsBelow(worker_bits[j / Rows], j % Rows))
                            return "an end's slot differs from its bit's rank";
                    }
                    const uint64_t s = seg_base + x0 + i;
                    const uint64_t count = x0 + i == 0 ? first_count : e - previous;
                    if (count != problem.end(s) - problem.begin(s)) return "a MEAN count differs";
                    for (uint32_t c = 0; c < D; ++c) {
                        const int64_t v = kind == RaggedEndKind::Ranked ? slot[raggedPacked16Slot(facts[i]) * D + c]
                                                                        : (kind == RaggedEndKind::CarryIn ? carry[c] : identity);
                        if (x0 + i == 0 && role.head != NO_SEGMENT) {
                            head[c] = v;
                        } else {
                            out[s * D + c] = v;
                            ++written[s * D + c];
                        }
                    }
                }
            }
            if (n > 0) last_end = at(x0 + n - 1);
            carry = next_carry;
            for (uint32_t i = 0; i < prefetch_count; ++i) {
                const uint32_t slot_index = raggedRingSlot(prefetch_first + i, mask);
                for (uint32_t s2 = x0 + n; s2 < x0 + held; ++s2)
                    if (raggedRingSlot(s2, mask) == slot_index) return "the prefetch overwrites a held end";
                ring[slot_index] = prefetched[i];
            }
            held = held - n + prefetch_count;
            x0 = tile_end.segment;
            y0 = y1;
            staged = next_staged;
        }
        keys[cta] = role.head;
        tails[cta] = Record{role.tail, 0};
        for (uint32_t c = 0; c < D; ++c) {
            head_values[c][cta] = role.head != NO_SEGMENT ? head[c] : identity;
            tail_values[c][cta] = role.tail != NO_SEGMENT ? carry[c] : identity;
            tails_by_component[c][cta] = carry[c];
        }
    }
    // The finisher: one CTA per component, each the D = 1 scan over its component's records.
    const uint64_t per_thread = 4, finish_threads = std::max<uint64_t>(1024, (grid + per_thread - 1) / per_thread);
    for (uint32_t c = 0; c < D; ++c) {
        const auto element = [&](uint64_t cta) {
            return cta < grid ? raggedRecordElement(keys[cta], tail_values[c][cta]) : Carry<int64_t>{false, identity};
        };
        std::vector<int64_t> arriving_at(finish_threads * per_thread, identity);
        Carry<int64_t> scan{false, identity};
        for (uint64_t t = 0; t < finish_threads; ++t) {
            Carry<int64_t> mine{false, identity}, arriving = scan;
            for (uint64_t i = 0; i < per_thread; ++i) mine = raggedCarryCombine(mine, element(per_thread * t + i), op);
            for (uint64_t i = 0; i < per_thread; ++i) {
                arriving_at[per_thread * t + i] = arriving.value;
                arriving = raggedCarryCombine(arriving, element(per_thread * t + i), op);
            }
            scan = raggedCarryCombine(scan, mine, op);
        }
        for (uint64_t cta = 0; cta < grid; ++cta) {
            if (!raggedRecordHasHead(keys[cta])) continue;
            const uint64_t s = keys[cta];
            int64_t walked = identity;
            for (uint64_t b = cta; b > 0 && tails[b - 1].segment == s; --b) walked = op(tails_by_component[c][b - 1], walked);
            if (walked != arriving_at[cta]) return "the scan's run differs from the walk back over the tails";
            out[s * D + c] = op(arriving_at[cta], head_values[c][cta]);
            ++written[s * D + c];
        }
    }
    return {};
}

template <typename Op, uint32_t ElementBytes = 1>
void expectComponentsExact(const Problem& problem, uint64_t grid, uint32_t ring, uint32_t D, uint64_t byte_head,
                           const std::string& what, const PathWeights& weights = PathWeights{}) {
    std::vector<int64_t> out;
    std::vector<int> written;
    const std::string failure =
        emulateComponentsKernels<Op, ElementBytes>(problem, grid, ring, D, byte_head, weights, out, written);
    ASSERT_TRUE(failure.empty()) << what << ": " << failure;
    const Op op{};
    for (uint64_t s = 0; s < problem.segments; ++s) {
        for (uint32_t c = 0; c < D; ++c) {
            int64_t expected = op.identity();
            for (uint64_t r = problem.begin(s); r < problem.end(s); ++r) expected = op(expected, componentValue(r, c));
            ASSERT_EQ(written[s * D + c], 1) << what << " segment " << s << " component " << c;
            ASSERT_EQ(out[s * D + c], expected) << what << " segment " << s << " component " << c;
        }
    }
}

// 1285: the ragged RK kernel (RaggedRKKernels.cuh) and its finisher, emulated with element bytes 1: for every path range
// and K tile, the KR walk per row with tiles of raggedRKRowsPerWorker x W rows (1288: 4 on narrow rows, else 2); the staging (contiguous or one run per row,
// every read of a lane checked against the staged bytes and the buffer, and the staged byte against the input's);
// the store pass of the ends without rows in the tile; each worker's fold of its rows cut at its ends, the scan across
// workers per column, each worker's first value; the records (raggedRecordLayout's arrays) and the finisher's order
// per column. Value of row r, column c: componentValue(r, c). `out` is segments x D.
struct RKSetup {
    uint32_t ranges = 1;
    uint32_t ring = 128;
    uint64_t D = 16;
    uint32_t head = 0;      // the input's byte misalignment (0 to 15)
    uint32_t tile_cap = 512;
};

template <typename Op>
std::string emulateRKKernels(const Problem& problem, const RKSetup& s, const PathWeights& requested,
                             std::vector<int64_t>& out, std::vector<int>& written) {
    const Op op{};
    const int64_t identity = op.identity();
    const uint64_t D = s.D, R = D;  // FP8: a row's bytes are D
    const uint32_t mask = s.ring - 1;
    constexpr uint32_t Chunk = RAGGED_RK_CHUNK_BYTES;
    if (R < RAGGED_RK_MIN_ROW_BYTES) return "rows under 16 bytes";
    const RaggedRKGeometry g = raggedRKGeometry(R, s.tile_cap);
    const uint32_t Rows = raggedRKRowsPerWorker(g);  // the kernel instantiation the launcher picks
    if (Rows != RAGGED_RK_ROWS && Rows != RAGGED_RK_ROWS_NARROW) return "rows per worker without a kernel";
    const uint32_t span_bytes = raggedRKSpanBytesFor(Rows);
    const uint32_t P = g.chunks, W = g.workers, tile_rows = raggedRKTileRows(g);
    if (P == 0 || P > 32 || W * P > 32 || W == 0) return "bad lanes";
    if (!g.contiguous && W > 16) return "a K-tiled row with more than 16 workers";
    if (g.tile_bytes > s.tile_cap && !g.contiguous) return "a K tile over the cap";
    if (uint64_t{g.tile_bytes} * (g.ktiles - 1) >= R) return "an empty K tile";
    out.assign(problem.segments * D, identity);
    written.assign(problem.segments * D, 0);
    std::vector<uint64_t> keys(s.ranges, NO_SEGMENT);
    std::vector<std::vector<int64_t>> heads(D, std::vector<int64_t>(s.ranges, identity)),
        tails(D, std::vector<int64_t>(s.ranges, identity));
    std::vector<std::vector<int>> head_written(D, std::vector<int>(s.ranges, 0));
    const PathWeights w = raggedEffectiveWeights(problem, requested, s.ranges);
    const uint64_t cost = pathCost(problem, w);
    for (uint32_t range = 0; range < s.ranges; ++range) {
        const SearchedCoordinate first = halfWarpSearchEmulated(problem, w, ctaDiagonal(cost, s.ranges, range));
        const SearchedCoordinate last = halfWarpSearchEmulated(problem, w, ctaDiagonal(cost, s.ranges, range + 1));
        const Coordinate cta_begin = first.at, cta_end = last.at;
        const CtaRole role = raggedCtaRoleFromBounds(problem.segments, first, last);
        const bool has_head = role.head != NO_SEGMENT;
        keys[range] = role.head;
        const uint64_t row_base = cta_begin.row, seg_base = cta_begin.segment;
        const uint32_t seg_count = static_cast<uint32_t>(cta_end.segment - seg_base);
        const uint32_t row_count = static_cast<uint32_t>(cta_end.row - row_base);
        const uint32_t end_diagonal = seg_count + row_count;
        const uint64_t first_begin = seg_count > 0 ? first.begin : 0;
        const uint64_t first_end = seg_count > 0 ? first.end : 0;
        const bool first_has_rows = first_begin < first_end;
        const uint64_t first_count = first_end - first_begin;
        for (uint32_t ktile = 0; ktile < g.ktiles; ++ktile) {
            const uint32_t k0 = ktile * g.tile_bytes, tile_k = raggedRKTileBytesOf(g, R, ktile);
            if (tile_k == 0 || tile_k > g.tile_bytes) return "a bad K tile width";
            const uint64_t byte_base = s.head + row_base * R + k0;
            std::vector<uint32_t> ring(s.ring, 0xDEADBEEFU);
            const auto at = [&](uint32_t x) -> uint32_t { return ring[raggedRingSlot(x, mask)]; };
            const auto end_rel = [&](uint32_t x) { return static_cast<uint32_t>(problem.end(seg_base + x) - row_base); };
            const auto load = [&](uint32_t first_index, uint32_t count) {
                for (uint32_t i = 0; i < count; ++i) ring[raggedRingSlot(first_index + i, mask)] = end_rel(first_index + i);
            };
            // The lane (worker, chunk)'s columns: [k0 + 16 chunk, + its bytes), elements of one byte.
            const auto chunk_values = [&](uint32_t chunk) -> uint32_t {
                return Chunk * chunk < tile_k ? std::min(Chunk, tile_k - Chunk * chunk) : 0;
            };
            uint32_t x0 = 0, y0 = 0, held = raggedEndsToLoad(0, s.ring, seg_count), last_end = 0;
            load(0, held);
            std::vector<int64_t> carry(tile_k, identity);  // per column of the K tile
            const auto store_end = [&](uint32_t i, uint32_t column, int64_t value) -> std::string {
                const uint64_t seg = seg_base + x0 + i;
                const uint64_t c = k0 + column;
                if (x0 + i == 0 && has_head) {
                    if (seg != role.head) return "the head is not the CTA's first end";
                    heads[c][range] = value;
                    ++head_written[c][range];
                    return {};
                }
                const uint32_t e = at(x0 + i), previous = i == 0 ? last_end : at(x0 + i - 1);
                const uint64_t count = x0 + i == 0 ? first_count : e - previous;
                if (count != problem.end(seg) - problem.begin(seg)) return "a MEAN count differs";
                out[seg * D + c] = value;
                ++written[seg * D + c];
                return {};
            };
            while (x0 + y0 < end_diagonal) {
                if (x0 + held < seg_count && held <= s.ring / 2) {
                    const uint32_t count = raggedEndsToLoad(held, s.ring, seg_count - (x0 + held));
                    load(x0 + held, count);
                    held += count;
                }
                const bool all_loaded = x0 + held == seg_count;
                const uint32_t row_limit = std::min(y0 + tile_rows, row_count);
                uint32_t ends_within = 0;
                for (uint32_t i = 0; i < held; ++i) ends_within += at(x0 + i) <= row_limit ? 1 : 0;
                if (held == 0 && !all_loaded) return "a tile with no ends loaded";
                const uint32_t last_loaded = held > 0 ? at(x0 + held - 1) : 0;
                const Coordinate32 tile_end = raggedTileEnd(x0, held, all_loaded, last_loaded, row_limit, ends_within);
                const uint32_t n = tile_end.segment - x0, y1 = tile_end.row;
                if (n > held) return "the tile consumes an end that is not loaded";
                if (n > s.ring || n > 255) return "more ends than the kinds hold";
                if (y1 < y0 || y1 - y0 > tile_rows) return "the tile's rows leave its span";
                if (n == 0 && y1 == y0) return "the tile makes no progress";
                const uint32_t prefetch_first = x0 + held;
                const uint32_t prefetch_count = raggedEndsToLoad(held - n, s.ring, seg_count - prefetch_first);
                std::vector<uint32_t> prefetched(prefetch_count);
                for (uint32_t i = 0; i < prefetch_count; ++i) prefetched[i] = end_rel(prefetch_first + i);

                // The staging, as fetch_rows: contiguous, or one run per row at row x slot bytes.
                const uint32_t rows_in_tile = y1 - y0;
                const uint32_t tile_shift = static_cast<uint32_t>((byte_base + uint64_t{y0} * R) & 15U);
                std::vector<uint32_t> staged_end(tile_rows, 0);  // per tile row: its staged bytes' end in the buffer
                std::vector<uint64_t> staged_origin(tile_rows, 0);
                if (g.contiguous) {
                    const uint64_t start = byte_base + uint64_t{y0} * R;
                    const uint32_t bytes = static_cast<uint32_t>(start & 15U) + rows_in_tile * static_cast<uint32_t>(R);
                    if ((bytes + 15) / 16 > raggedRKStagedChunksFor(Rows)) return "a contiguous tile over the lanes' copies";
                    for (uint32_t j = 0; j < tile_rows; ++j) {
                        staged_end[j] = bytes;
                        staged_origin[j] = start & ~uint64_t{15};
                    }
                } else {
                    if (rows_in_tile * (P + 1) > 3 * 32) return "a K-tiled tile over the lanes' copy tasks";
                    for (uint32_t j = 0; j < rows_in_tile; ++j) {
                        const uint64_t start = byte_base + uint64_t{y0 + j} * R;
                        staged_end[j] = j * raggedRKSlotBytes(g) + static_cast<uint32_t>(start & 15U) + tile_k;
                        staged_origin[j] = (start & ~uint64_t{15}) - j * raggedRKSlotBytes(g);  // buffer offset 0's byte
                    }
                }
                for (uint32_t wk = 0; wk < W; ++wk)
                    for (uint32_t i = 0; i < Rows; ++i)
                        for (uint32_t chunk = 0; chunk < P; ++chunk) {
                            const uint32_t j = Rows * wk + i;
                            const uint32_t o = raggedRKStagedOffset(g, R, tile_shift, j, chunk);
                            if (o + 20 > span_bytes) return "a lane's read leaves the span buffer";
                            const uint32_t valid = chunk_values(chunk);
                            if (j < rows_in_tile && valid > 0) {
                                if (o + valid > staged_end[j]) return "a lane reads a byte the fetch did not stage";
                                const uint64_t address = s.head + (row_base + y0 + j) * R + k0 + Chunk * chunk;
                                if (staged_origin[j] + o != address) return "a lane reads the wrong staged byte";
                            }
                        }

                // The bits, kinds and each Ranked end's row.
                std::vector<uint8_t> kind(n), row_end(tile_rows, 0xFF);
                std::vector<uint32_t> bits(tile_rows, 0);
                for (uint32_t i = 0; i < n; ++i) {
                    const uint32_t e = at(x0 + i), previous = i > 0 ? at(x0 + i - 1) : last_end;
                    const RaggedEndKind k = raggedEndKindAfter(x0 + i == 0, first_has_rows, e, previous, y0);
                    kind[i] = static_cast<uint8_t>(k);
                    if (k == RaggedEndKind::Ranked) {
                        const uint32_t j = e - 1 - y0;
                        if (j >= rows_in_tile) return "a bit outside the tile's rows";
                        if (bits[j]) return "two segments end at one row";
                        bits[j] = 1;
                        row_end[j] = static_cast<uint8_t>(i);
                    }
                }
                // The store pass: ends without rows in the tile.
                for (uint32_t i = 0; i < n; ++i) {
                    if (static_cast<RaggedEndKind>(kind[i]) == RaggedEndKind::Ranked) continue;
                    for (uint32_t c = 0; c < tile_k; ++c) {
                        const std::string f =
                            store_end(i, c, static_cast<RaggedEndKind>(kind[i]) == RaggedEndKind::CarryIn ? carry[c] : identity);
                        if (!f.empty()) return f;
                    }
                }
                // Each worker's fold, cut at its ends; then the scan across workers per column.
                const bool reset_at_start = n > 0 && at(x0) == y0;
                std::vector<int64_t> next_carry(tile_k);
                for (uint32_t c = 0; c < tile_k; ++c) {
                    std::vector<int64_t> run(W, identity), first_value(W, identity);
                    std::vector<int> first_row(W, -1);
                    std::vector<bool> reset(W, false);
                    for (uint32_t wk = 0; wk < W; ++wk) {
                        int64_t acc = identity;
                        for (uint32_t i = 0; i < Rows; ++i) {
                            const uint32_t j = Rows * wk + i;
                            const int64_t v = j < rows_in_tile ? componentValue(row_base + y0 + j, k0 + c) : identity;
                            acc = op(acc, v);
                            if (j < rows_in_tile && bits[j]) {
                                if (first_row[wk] < 0) {
                                    first_row[wk] = static_cast<int>(i);
                                    first_value[wk] = acc;
                                } else {
                                    const std::string f = store_end(row_end[j], c, acc);
                                    if (!f.empty()) return f;
                                }
                                acc = identity;
                                reset[wk] = true;
                            }
                        }
                        run[wk] = acc;
                    }
                    const int64_t seed = reset_at_start ? identity : carry[c];
                    Carry<int64_t> arriving{false, seed};
                    for (uint32_t wk = 0; wk < W; ++wk) {
                        if (first_row[wk] >= 0) {
                            const std::string f =
                                store_end(row_end[Rows * wk + first_row[wk]], c, op(arriving.value, first_value[wk]));
                            if (!f.empty()) return f;
                        }
                        arriving = raggedCarryCombine(arriving, Carry<int64_t>{reset[wk], run[wk]}, op);
                    }
                    next_carry[c] = arriving.value;
                }
                carry = next_carry;
                if (n > 0) last_end = at(x0 + n - 1);
                for (uint32_t i = 0; i < prefetch_count; ++i) {
                    const uint32_t slot_index = raggedRingSlot(prefetch_first + i, mask);
                    for (uint32_t s2 = x0 + n; s2 < x0 + held; ++s2)
                        if (raggedRingSlot(s2, mask) == slot_index) return "the prefetch overwrites a held end";
                    ring[slot_index] = prefetched[i];
                }
                held = held - n + prefetch_count;
                x0 = tile_end.segment;
                y0 = y1;
            }
            for (uint32_t c = 0; c < tile_k; ++c) tails[k0 + c][range] = role.tail != NO_SEGMENT ? carry[c] : identity;
        }
    }
    // The finisher, per column: the scan over the ranges' records in range order, restarting at each head.
    for (uint64_t c = 0; c < D; ++c) {
        int64_t run = identity;
        for (uint32_t r = 0; r < s.ranges; ++r) {
            const bool head = raggedRecordHasHead(keys[r]);
            if (head) {
                if (head_written[c][r] != 1) return "a head value not written once";
                out[keys[r] * D + c] = op(run, heads[c][r]);
                ++written[keys[r] * D + c];
                run = tails[c][r];
            } else {
                run = op(run, tails[c][r]);
            }
        }
    }
    return {};
}

template <typename Op>
void expectRKExact(const Problem& problem, const RKSetup& s, const std::string& what,
                   const PathWeights& weights = PathWeights{}) {
    std::vector<int64_t> out;
    std::vector<int> written;
    const std::string failure = emulateRKKernels<Op>(problem, s, weights, out, written);
    ASSERT_TRUE(failure.empty()) << what << ": " << failure;
    const Op op{};
    for (uint64_t seg = 0; seg < problem.segments; ++seg) {
        for (uint64_t c = 0; c < s.D; ++c) {
            int64_t expected = op.identity();
            for (uint64_t r = problem.begin(seg); r < problem.end(seg); ++r)
                expected = op(expected, componentValue(r, static_cast<uint32_t>(c)));
            ASSERT_EQ(written[seg * s.D + c], 1) << what << " segment " << seg << " column " << c;
            ASSERT_EQ(out[seg * s.D + c], expected) << what << " segment " << seg << " column " << c;
        }
    }
}

struct Stored {
    std::vector<uint8_t> bytes;
    explicit Stored(const std::vector<uint64_t>& v, uint32_t width) : bytes(v.size() * width) {
        for (size_t i = 0; i < v.size(); ++i) {
            if (width == 8) std::memcpy(bytes.data() + 8 * i, &v[i], 8);
            else {
                const uint32_t x = static_cast<uint32_t>(v[i]);
                std::memcpy(bytes.data() + 4 * i, &x, 4);
            }
        }
    }
};

}  // namespace

TEST(RaggedKRWalk, WarpSearchesMatchSearchPathAndTheWeightedSearch) {
    for (uint64_t trial = 0; trial < 60; ++trial) {
        const uint64_t seed = A::mix(0x5EA4 + trial);
        const uint64_t rows = seed % 20000, segments = 1 + A::mix(seed) % (trial % 2 ? 5000 : 40);
        const std::vector<uint64_t> values = A::makeOffsets(rows, segments, A::DISTRIBUTIONS[trial % 5], seed);
        const Stored stored(values, 4);
        const Problem p = makeProblem(Offsets{stored.bytes.data(), 4}, segments, rows);
        const PathWeights weighted{1 + A::mix(seed + 1) % 3, 1 + A::mix(seed + 2) % 12};
        for (uint64_t k = 0; k < 64; ++k) {
            const uint64_t diag = k == 0 ? 0 : (k == 1 ? p.pathLength() : A::mix(seed + k) % (p.pathLength() + 1));
            const Coordinate b = searchPath(p, diag);
            for (uint32_t probes : {32U, 16U}) {
                const Coordinate a = warpSearchEmulated(p, PathWeights{}, diag, probes);
                ASSERT_EQ(a.segment, b.segment) << "seed " << seed << " diagonal " << diag << " probes " << probes;
                ASSERT_EQ(a.row, b.row) << "seed " << seed << " diagonal " << diag << " probes " << probes;
            }
            const SearchedCoordinate h = halfWarpSearchEmulated(p, PathWeights{}, diag);  // 1272
            const SearchedCoordinate r = raggedSearched(p, b);
            ASSERT_EQ(h.at.segment, b.segment) << "seed " << seed << " diagonal " << diag << " half warp";
            ASSERT_EQ(h.at.row, b.row) << "seed " << seed << " diagonal " << diag << " half warp";
            ASSERT_EQ(h.begin, r.begin) << "seed " << seed << " diagonal " << diag << " half warp";
            ASSERT_EQ(h.end, r.end) << "seed " << seed << " diagonal " << diag << " half warp";
            const uint64_t total = pathCost(p, weighted);
            const uint64_t c = k == 0 ? 0 : (k == 1 ? total : A::mix(seed + 99 * k) % (total + 1));
            const Coordinate y = searchPathCost(p, weighted, c);
            for (uint32_t probes : {32U, 16U}) {
                const Coordinate x = warpSearchEmulated(p, weighted, c, probes);
                ASSERT_EQ(x.segment, y.segment) << "seed " << seed << " cost " << c << " probes " << probes;
                ASSERT_EQ(x.row, y.row) << "seed " << seed << " cost " << c << " probes " << probes;
            }
            const SearchedCoordinate hw = halfWarpSearchEmulated(p, weighted, c);
            const SearchedCoordinate rw = raggedSearched(p, y);
            ASSERT_EQ(hw.at.segment, y.segment) << "seed " << seed << " cost " << c << " half warp";
            ASSERT_EQ(hw.at.row, y.row) << "seed " << seed << " cost " << c << " half warp";
            ASSERT_EQ(hw.begin, rw.begin) << "seed " << seed << " cost " << c << " half warp";
            ASSERT_EQ(hw.end, rw.end) << "seed " << seed << " cost " << c << " half warp";
        }
    }
}

static int64_t none_sum() { return 10 + 11 + 12 + 13 + 14 + 15 + 16 + 17; }

TEST(RaggedKRWalk, LaneRunAndEmitFoldEachSegmentInOrder) {
    // Bits 2 and 5: rows 0-2 end a segment (with the carry), 3-5 another, 6-7 run on.
    const auto value = [](uint32_t k) { return static_cast<int64_t>(10 + k); };
    const Carry<int64_t> run = raggedLaneRun<8>(0b100100U, int64_t{0}, SumOp{}, value);
    EXPECT_TRUE(run.reset);
    EXPECT_EQ(run.value, 16 + 17);
    std::vector<std::pair<uint32_t, int64_t>> emitted;
    const int64_t open = raggedLaneEmit<8>(0b100100U, int64_t{100}, int64_t{0}, SumOp{}, value,
                                           [&](uint32_t k, int64_t v) { emitted.emplace_back(k, v); });
    ASSERT_EQ(emitted.size(), 2U);
    EXPECT_EQ(emitted[0], std::make_pair(2U, int64_t{100 + 10 + 11 + 12}));
    EXPECT_EQ(emitted[1], std::make_pair(5U, int64_t{13 + 14 + 15}));
    EXPECT_EQ(open, run.value);  // 1272: the run after the last end
    EXPECT_EQ((raggedLaneEmit<8>(0U, int64_t{100}, int64_t{0}, SumOp{}, value, [](uint32_t, int64_t) {})),
              100 + none_sum());
    EXPECT_EQ(raggedLaneFirstValue(int64_t{7}, int64_t{30}, SumOp{}), 37);
    EXPECT_EQ(raggedLaneFirstValue(int64_t{7}, int64_t{30}, MinOp{}), 7);
    const Carry<int64_t> none = raggedLaneRun<8>(0U, int64_t{0}, SumOp{}, value);
    EXPECT_FALSE(none.reset);
    EXPECT_EQ(none.value, none_sum());
    EXPECT_EQ((raggedTreeFold<8, int64_t>(SumOp{}, value)), none.value);  // 1271
    EXPECT_EQ((raggedTreeFold<8, int64_t>(MinOp{}, value)), 10);
    EXPECT_EQ(raggedBitsBelow(0b1011U, 3), 2U);
    EXPECT_EQ(raggedBitsBelow(0xffffffffU, 32), 32U);
    // The tile end: at the span's last row, or at the last loaded end when an unloaded end could come first.
    EXPECT_EQ(raggedTileEnd(5, 3, false, 40, 30, 2).segment, 7U);
    EXPECT_EQ(raggedTileEnd(5, 3, false, 20, 30, 3).row, 20U);
    EXPECT_EQ(raggedTileEnd(5, 3, false, 20, 30, 3).segment, 8U);
    EXPECT_EQ(raggedTileEnd(5, 3, true, 20, 30, 3).row, 30U);
}

// 1272: the span's 16-byte chunks fill shared memory piece-major: a bijection onto the span's slots, chunk c holding
// lane c / P's piece c % P, so copy q (chunks 32q .. 32q + 31) reads consecutive bytes and each piece's 32 reads are
// consecutive 16-byte slots.
TEST(RaggedKRWalk, ChunkSlotsArePieceMajor) {
    const auto check = [](auto rows_constant) {
        constexpr uint32_t Rows = decltype(rows_constant)::value, Lanes = 32, Pieces = Rows / 16;
        std::vector<int> used(Lanes * Pieces, 0);
        for (uint32_t chunk = 0; chunk < Lanes * Pieces; ++chunk) {
            const uint32_t slot = raggedChunkSlot<Rows, Lanes>(chunk);
            ASSERT_EQ(slot % 16, 0U);
            ASSERT_LT(slot / 16, Lanes * Pieces);
            ++used[slot / 16];
            ASSERT_EQ(slot, (raggedPieceSlot<Rows, Lanes>(chunk / Pieces, chunk % Pieces)));
        }
        for (int u : used) ASSERT_EQ(u, 1);
        for (uint32_t piece = 0; piece < Pieces; ++piece)
            for (uint32_t lane = 1; lane < Lanes; ++lane)
                ASSERT_EQ((raggedPieceSlot<Rows, Lanes>(lane, piece)), (raggedPieceSlot<Rows, Lanes>(lane - 1, piece)) + 16);
    };
    check(std::integral_constant<uint32_t, 16>{});
    check(std::integral_constant<uint32_t, 32>{});
    EXPECT_EQ((raggedChunkSlot<16, 32>(5)), 80U);  // one piece: the identity
    EXPECT_EQ((raggedChunkSlot<32, 32>(3)), 16U * (32 + 1));  // lane 1's second piece
}

// 1275: the store pass's slot and kind rules, directly (the emulation checks them against the bits on every tile).
TEST(RaggedKRWalk, EndFactsAndRankedSlots) {
    // Ranked ends at lanes 1, 2 and 5 of a round, after 7 in the rounds before.
    const uint32_t ballot = (1U << 1) | (1U << 2) | (1U << 5);
    EXPECT_EQ(raggedRankedSlot(7, ballot, 0), 7U);
    EXPECT_EQ(raggedRankedSlot(7, ballot, 1), 7U);
    EXPECT_EQ(raggedRankedSlot(7, ballot, 2), 8U);
    EXPECT_EQ(raggedRankedSlot(7, ballot, 5), 9U);
    EXPECT_EQ(raggedRankedSlot(0, 0xffffffffU, 31), 31U);
    // An end's kind from the end before it; the CTA's first end from its segment's bounds.
    EXPECT_EQ(raggedEndKindAfter(false, false, 10, 10, 4), RaggedEndKind::Empty);
    EXPECT_EQ(raggedEndKindAfter(false, false, 10, 9, 4), RaggedEndKind::Ranked);
    EXPECT_EQ(raggedEndKindAfter(false, false, 4, 3, 4), RaggedEndKind::CarryIn);
    EXPECT_EQ(raggedEndKindAfter(true, true, 0, 0, 0), RaggedEndKind::CarryIn);
    EXPECT_EQ(raggedEndKindAfter(true, false, 6, 0, 0), RaggedEndKind::Empty);
    EXPECT_EQ(raggedEndKindAfter(true, true, 6, 0, 0), RaggedEndKind::Ranked);
    for (const RaggedEndKind kind : {RaggedEndKind::Empty, RaggedEndKind::CarryIn, RaggedEndKind::Ranked})
        for (const uint32_t slot : {0U, 1U, 127U, 0xffffU}) {
            const uint32_t packed = raggedPackEndFacts(kind, slot);
            EXPECT_EQ(raggedPackedKind(packed), kind);
            EXPECT_EQ(raggedPackedSlot(packed), slot);
        }
}

TEST(RaggedKRWalk, EmulatedKernelsMatchDirectSumAndMin) {
    const Geometry geometries[] = {Geometry{4, 8, 0}, Geometry{4, 16, 5}, Geometry{32, 32, 13}, Geometry{32, 128, 3}};
    for (uint64_t trial = 0; trial < 16; ++trial) {
        const uint64_t seed = A::mix(0x1270 + trial);
        const uint64_t rows = seed % 3000;
        const uint64_t segments = 1 + A::mix(seed) % (trial % 3 == 0 ? std::max<uint64_t>(rows, 1) : 300);
        // 1268: unit weights, the FP8 D = 1 weights (UINT32 / UINT64 offsets, FP32 output) and an odd pair.
        const PathWeights weights[] = {PathWeights{}, PathWeights{1, 8}, PathWeights{1, 12}, PathWeights{3, 5}};
        const PathWeights w = weights[trial % 4];
        for (A::Distribution d : A::DISTRIBUTIONS) {
            const std::vector<uint64_t> values = A::makeOffsets(rows, segments, d, seed);
            const Stored stored(values, trial % 2 ? 8 : 4);
            const Problem p = makeProblem(Offsets{stored.bytes.data(), trial % 2 ? 8U : 4U}, segments, rows);
            for (const Geometry& g : geometries) {
                const std::vector<uint64_t> grids = g.lanes == 32 ? std::vector<uint64_t>{1, 7, 64}
                                                                  : std::vector<uint64_t>{1, 2, 7, 64, p.pathLength() + 5};
                for (uint64_t grid : grids) {
                    const std::string what = std::string(A::distributionName(d)) + " seed " + std::to_string(seed) +
                                             " lanes " + std::to_string(g.lanes) + " ring " + std::to_string(g.ring) +
                                             " head " + std::to_string(g.byte_head) + " grid " + std::to_string(grid) +
                                             " weights " + std::to_string(w.row) + "/" + std::to_string(w.end);
                    expectExact<SumOp, 16>(p, grid, g, what + " sum r16", w);
                    expectExact<MinOp, 32>(p, grid, g, what + " min r32", w);
                    // 1284: FP16 / BF16 (8 rows per lane) and FP32 (4), the head in elements.
                    expectExact<MinOp, 8>(p, grid, g, what + " min r8", w);
                    expectExact<SumOp, 4>(p, grid, g, what + " sum r4", w);
                }
            }
        }
    }
}

TEST(RaggedKRWalk, WeightedSplitsBalanceCostAndFallBackForTinyProblems) {
    // One giant segment then a long run of empty ones: unweighted, the CTAs on the empty run hold as many events as
    // the others but move end-weight times the bytes; weighted, every CTA's cost is within one event of its share.
    std::vector<uint64_t> values(4001, 0);
    for (size_t i = 1; i < values.size(); ++i) values[i] = 40000;
    const Stored stored(values, 4);
    const Problem p = makeProblem(Offsets{stored.bytes.data(), 4}, 4000, 40000);
    const PathWeights w{1, 8};
    const uint64_t grid = 50, cost = pathCost(p, w);
    ASSERT_EQ(raggedEffectiveWeights(p, w, grid).end, 8U);
    for (uint64_t cta = 0; cta < grid; ++cta) {
        const Coordinate a = raggedCtaCoordinate(p, w, grid, cta), b = raggedCtaCoordinate(p, w, grid, cta + 1);
        const uint64_t cta_cost = w.end * (b.segment - a.segment) + w.row * (b.row - a.row);
        EXPECT_LE(cta_cost, cost / grid + 2 * w.end) << "CTA " << cta;
        EXPECT_GE(cta_cost + 2 * w.end, cost / grid) << "CTA " << cta;
    }
    expectExact<SumOp>(p, grid, Geometry{32, 64, 7}, "giant then empty", w);
    // A share below the costlier event: unit weights, so no CTA in the middle is empty.
    const Stored few(std::vector<uint64_t>{0, 1, 1, 1, 2}, 4);
    const Problem tiny = makeProblem(Offsets{few.bytes.data(), 4}, 4, 2);
    EXPECT_EQ(raggedEffectiveWeights(tiny, w, 5).end, 1U);  // cost 34 / 5 < 8
    EXPECT_EQ(raggedEffectiveWeights(tiny, w, 1).end, 8U);
    for (uint64_t grid_tiny : {1ULL, 2ULL, 3ULL, 7ULL, 40ULL}) expectExact<SumOp>(tiny, grid_tiny, Geometry{4, 8, 1}, "tiny", w);
}

TEST(RaggedKRWalk, ClampedOffsetsAndNoRows) {
    // Offsets running past the capacity are clamped: the reduction covers only active rows.
    std::vector<uint64_t> values = A::makeOffsets(4000, 37, A::Distribution::Pareto, 99);
    const Stored stored(values, 8);
    const Problem clamped = makeProblem(Offsets{stored.bytes.data(), 8}, 37, 1500);
    for (uint64_t grid : {1ULL, 3ULL, 50ULL}) {
        expectExact<SumOp>(clamped, grid, Geometry{4, 8, 2}, "clamped");
        expectExact<SumOp, 32>(clamped, grid, Geometry{32, 128, 9}, "clamped weighted", PathWeights{1, 12});
    }
    // No rows: every segment is empty and stored once.
    const Stored zeros(std::vector<uint64_t>(41, 0), 4);
    const Problem empty = makeProblem(Offsets{zeros.bytes.data(), 4}, 40, 100);
    for (uint64_t grid : {1ULL, 6ULL, 100ULL}) {
        expectExact<SumOp>(empty, grid, Geometry{4, 8, 0}, "no rows");
        expectExact<MinOp, 32>(empty, grid, Geometry{32, 32, 0}, "no rows", PathWeights{1, 8});
    }
    // Every row its own segment (ends as dense as rows), and long runs of empty segments between rows.
    std::vector<uint64_t> each(1201), bursts(2001, 0);
    for (size_t i = 0; i < each.size(); ++i) each[i] = i;
    for (size_t i = 0; i < bursts.size(); ++i) bursts[i] = (i / 300) * 7;
    const Stored s_each(each, 4), s_bursts(bursts, 4);
    const Problem p_each = makeProblem(Offsets{s_each.bytes.data(), 4}, 1200, 1200);
    const Problem p_bursts = makeProblem(Offsets{s_bursts.bytes.data(), 4}, 2000, bursts.back());
    for (uint64_t grid : {1ULL, 5ULL, 33ULL}) {
        expectExact<SumOp>(p_each, grid, Geometry{32, 32, 11}, "each row", PathWeights{1, 8});
        expectExact<MinOp, 32>(p_each, grid, Geometry{4, 8, 3}, "each row");
        expectExact<SumOp>(p_bursts, grid, Geometry{32, 32, 6}, "bursts", PathWeights{1, 8});
        expectExact<MinOp, 32>(p_bursts, grid, Geometry{4, 16, 1}, "bursts");
        expectExact<SumOp, 8>(p_each, grid, Geometry{32, 128, 7}, "each row r8", PathWeights{2, 8});
        expectExact<MinOp, 4>(p_each, grid, Geometry{32, 128, 3}, "each row r4", PathWeights{4, 12});
        expectExact<SumOp, 4>(p_bursts, grid, Geometry{32, 32, 2}, "bursts r4", PathWeights{4, 8});
        expectExact<MinOp, 8>(p_bursts, grid, Geometry{32, 128, 5}, "bursts r8");
    }
}

// 1280: the components kernel's geometry, its tile cap, its 16-bit facts and the record arrays.
TEST(RaggedKRWalk, ComponentGeometryCapAndRecordLayout) {
    for (uint32_t D = 2; D <= RAGGED_KR_MAX_COMPONENTS; ++D) {
        const uint32_t workers = raggedComponentWorkers(D);
        EXPECT_LE(workers * D, 32U) << D;
        EXPECT_GT((workers + 1) * D, 32U) << D;
        EXPECT_LE(raggedComponentTileRows(D) * D, 512U) << D;  // the span buffer: 512 bytes and the misalignment
        const uint32_t cap = raggedComponentEndsCap(D, 128);
        EXPECT_LE(cap * D, RAGGED_KR_COMPONENT_VALUE_SLOTS) << D;
        EXPECT_LE(cap, 128U) << D;
        EXPECT_GE(cap, 1U) << D;
    }
    // 1284: every dtype: 16 bytes of rows per lane or worker, rows under 16 bytes, tiles of at most 512 bytes.
    EXPECT_EQ(raggedKRRowsPerLane(1), 16U);
    EXPECT_EQ(raggedKRRowsPerLane(2), 8U);
    EXPECT_EQ(raggedKRRowsPerLane(4), 4U);
    EXPECT_EQ(raggedKRMaxComponents(1), 15U);
    EXPECT_EQ(raggedKRMaxComponents(2), 7U);
    EXPECT_EQ(raggedKRMaxComponents(4), 3U);
    for (uint32_t bytes : {1U, 2U, 4U}) {
        for (uint32_t D = 2; D <= raggedKRMaxComponents(bytes); ++D) {
            EXPECT_EQ(raggedComponentTileRows(D, bytes), raggedKRRowsPerLane(bytes) * raggedComponentWorkers(D));
            EXPECT_LE(raggedComponentTileRows(D, bytes) * D * bytes, 512U) << D << " x " << bytes;
            EXPECT_LE(raggedComponentTileRows(D, bytes), 256U) << D << " x " << bytes;
        }
        EXPECT_LE(raggedKRMaxComponents(bytes) * bytes, 15U);
    }
    EXPECT_EQ(raggedComponentEndsCap(2, 128), 128U);
    EXPECT_EQ(raggedComponentEndsCap(15, 128), 17U);
    EXPECT_EQ(raggedComponentEndsCap(3, 64), 64U);
    // The cap: more ends held than the cap treat only the first `cap` as loaded.
    const RaggedTileEnds below = raggedCapTileEnds(10, true, 7, 17);
    EXPECT_EQ(below.held, 10U);
    EXPECT_TRUE(below.all_loaded);
    EXPECT_EQ(below.ends_within, 7U);
    const RaggedTileEnds above = raggedCapTileEnds(40, true, 30, 17);
    EXPECT_EQ(above.held, 17U);
    EXPECT_FALSE(above.all_loaded);
    EXPECT_EQ(above.ends_within, 17U);
    for (const RaggedEndKind kind : {RaggedEndKind::Empty, RaggedEndKind::CarryIn, RaggedEndKind::Ranked})
        for (const uint32_t slot : {0U, 1U, 127U, 255U}) {
            const uint32_t packed = raggedPackEndFacts16(kind, slot);
            EXPECT_LT(packed, 1U << 16);
            EXPECT_EQ(raggedPacked16Kind(packed), kind);
            EXPECT_EQ(raggedPacked16Slot(packed), slot);
        }
    // The records: keys, heads, tails, each 16-byte aligned and disjoint; 16 bytes per CTA at D = 1 (1273's size).
    for (uint32_t grid : {1U, 3U, 170U, 4080U, 4096U}) {
        EXPECT_EQ(raggedRecordLayout(grid, 1).bytes, raggedRoundUp16(8ULL * grid) + 2 * raggedRoundUp16(4ULL * grid));
        if (grid % 4 == 0) {
            EXPECT_EQ(raggedRecordLayout(grid, 1).bytes, 16ULL * grid);
        }
        for (uint32_t D : {1U, 2U, 7U, 15U}) {
            const RaggedRecordLayout l = raggedRecordLayout(grid, D);
            EXPECT_EQ(l.keys, 0U);
            EXPECT_GE(l.heads, 8ULL * grid);
            EXPECT_GE(l.tails, l.heads + 4ULL * grid * D);
            EXPECT_GE(l.bytes, l.tails + 4ULL * grid * D);
            EXPECT_EQ(l.heads % 16, 0U);
            EXPECT_EQ(l.tails % 16, 0U);
            EXPECT_EQ(raggedRecordValueIndex(grid, grid - 1, D - 1), uint64_t{D} * grid - 1);
        }
    }
}

// 1280: the components kernel and the per-component finisher against direct per-segment, per-component SUM and MIN.
TEST(RaggedKRWalk, EmulatedComponentsKernelsMatchDirectSumAndMin) {
    const uint32_t ds[] = {2, 3, 4, 5, 7, 8, 9, 11, 13, 15};
    for (uint64_t trial = 0; trial < 10; ++trial) {
        const uint64_t seed = A::mix(0x1280 + trial);
        const uint64_t rows = seed % 4000;
        const uint64_t segments = 1 + A::mix(seed) % (trial % 3 == 0 ? std::max<uint64_t>(rows, 1) : 400);
        const uint32_t D = ds[trial];
        // Unit weights and the FP8 weights at D (UINT32 / UINT64 offsets, FP32 output, extra 4).
        const PathWeights weights[] = {PathWeights{}, PathWeights{D, 4 + 4 * D + 4}, PathWeights{D, 8 + 4 * D + 4}};
        for (A::Distribution d : A::DISTRIBUTIONS) {
            const std::vector<uint64_t> values = A::makeOffsets(rows, segments, d, seed);
            const Stored stored(values, trial % 2 ? 8 : 4);
            const Problem p = makeProblem(Offsets{stored.bytes.data(), trial % 2 ? 8U : 4U}, segments, rows);
            for (uint64_t grid : std::vector<uint64_t>{1, 7, 64, p.pathLength() + 5}) {
                for (uint32_t ring : {32U, 128U}) {
                    const PathWeights w = weights[(grid + ring) % 3];
                    const std::string what = std::string(A::distributionName(d)) + " seed " + std::to_string(seed) +
                                             " D " + std::to_string(D) + " ring " + std::to_string(ring) + " grid " +
                                             std::to_string(grid) + " weights " + std::to_string(w.row) + "/" +
                                             std::to_string(w.end);
                    expectComponentsExact<SumOp>(p, grid, ring, D, (seed + grid) % 16, what + " sum", w);
                    expectComponentsExact<MinOp>(p, grid, ring, D, (seed + ring) % 16, what + " min", w);
                }
            }
        }
    }
    // Ends as dense as rows (every tile at its cap at D 15) and long runs of empty segments; clamped offsets; no rows.
    std::vector<uint64_t> each(1201), bursts(2001, 0);
    for (size_t i = 0; i < each.size(); ++i) each[i] = i;
    for (size_t i = 0; i < bursts.size(); ++i) bursts[i] = (i / 300) * 7;
    const Stored s_each(each, 4), s_bursts(bursts, 4), s_zeros(std::vector<uint64_t>(41, 0), 8);
    const Problem p_each = makeProblem(Offsets{s_each.bytes.data(), 4}, 1200, 1200);
    const Problem p_bursts = makeProblem(Offsets{s_bursts.bytes.data(), 4}, 2000, bursts.back());
    const Problem p_empty = makeProblem(Offsets{s_zeros.bytes.data(), 8}, 40, 100);
    const std::vector<uint64_t> wide = A::makeOffsets(4000, 37, A::Distribution::Pareto, 99);
    const Stored s_wide(wide, 8);
    const Problem p_clamped = makeProblem(Offsets{s_wide.bytes.data(), 8}, 37, 1500);
    for (uint32_t D : {2U, 6U, 15U}) {
        for (uint64_t grid : {1ULL, 5ULL, 33ULL}) {
            const std::string what = "D " + std::to_string(D) + " grid " + std::to_string(grid);
            expectComponentsExact<SumOp>(p_each, grid, 128, D, 3, "each row " + what, PathWeights{D, 4 + 4 * D + 4});
            expectComponentsExact<MinOp>(p_each, grid, 32, D, 0, "each row " + what);
            expectComponentsExact<SumOp>(p_bursts, grid, 128, D, 9, "bursts " + what, PathWeights{D, 4 + 4 * D + 4});
            expectComponentsExact<MinOp>(p_bursts, grid, 32, D, 15, "bursts " + what);
            expectComponentsExact<SumOp>(p_empty, grid, 128, D, 1, "no rows " + what);
            expectComponentsExact<MinOp>(p_clamped, grid, 128, D, 5, "clamped " + what, PathWeights{D, 8 + 4 * D + 4});
        }
    }
}

// 1284: the components kernel over 16-bit (8 rows per worker, D 2..7) and 32-bit (4 rows per worker, D 2..3) elements,
// the input's misalignment a multiple of the element.
TEST(RaggedKRWalk, EmulatedWideComponentsKernelsMatchDirectSumAndMin) {
    for (uint64_t trial = 0; trial < 8; ++trial) {
        const uint64_t seed = A::mix(0x1284 + trial);
        const uint64_t rows = seed % 4000;
        const uint64_t segments = 1 + A::mix(seed) % (trial % 3 == 0 ? std::max<uint64_t>(rows, 1) : 400);
        const uint32_t d16 = 2 + static_cast<uint32_t>(trial % 6), d32 = 2 + static_cast<uint32_t>(trial % 2);
        for (A::Distribution d : A::DISTRIBUTIONS) {
            const std::vector<uint64_t> values = A::makeOffsets(rows, segments, d, seed);
            const Stored stored(values, trial % 2 ? 8 : 4);
            const Problem p = makeProblem(Offsets{stored.bytes.data(), trial % 2 ? 8U : 4U}, segments, rows);
            for (uint64_t grid : std::vector<uint64_t>{1, 7, 64, p.pathLength() + 5}) {
                for (uint32_t ring : {32U, 128U}) {
                    const std::string what = std::string(A::distributionName(d)) + " seed " + std::to_string(seed) +
                                             " ring " + std::to_string(ring) + " grid " + std::to_string(grid);
                    const PathWeights w16 = (grid + ring) % 2 ? PathWeights{2 * d16, 4 + 2 * d16 + 4} : PathWeights{};
                    const PathWeights w32 = (grid + ring) % 2 ? PathWeights{4 * d32, 8 + 4 * d32 + 4} : PathWeights{};
                    expectComponentsExact<SumOp, 2>(p, grid, ring, d16, 2 * ((seed + grid) % 8),
                                                    what + " D " + std::to_string(d16) + " 16-bit sum", w16);
                    expectComponentsExact<MinOp, 2>(p, grid, ring, d16, 2 * ((seed + ring) % 8),
                                                    what + " D " + std::to_string(d16) + " 16-bit min", w16);
                    expectComponentsExact<SumOp, 4>(p, grid, ring, d32, 4 * ((seed + grid) % 4),
                                                    what + " D " + std::to_string(d32) + " 32-bit sum", w32);
                    expectComponentsExact<MinOp, 4>(p, grid, ring, d32, 4 * ((seed + ring) % 4),
                                                    what + " D " + std::to_string(d32) + " 32-bit min", w32);
                }
            }
        }
    }
    // Ends as dense as rows and long runs of empty segments, at every D of each width.
    std::vector<uint64_t> each(1201), bursts(2001, 0);
    for (size_t i = 0; i < each.size(); ++i) each[i] = i;
    for (size_t i = 0; i < bursts.size(); ++i) bursts[i] = (i / 300) * 7;
    const Stored s_each(each, 4), s_bursts(bursts, 4);
    const Problem p_each = makeProblem(Offsets{s_each.bytes.data(), 4}, 1200, 1200);
    const Problem p_bursts = makeProblem(Offsets{s_bursts.bytes.data(), 4}, 2000, bursts.back());
    for (uint64_t grid : {1ULL, 5ULL, 33ULL}) {
        for (uint32_t D = 2; D <= 7; ++D) {
            const std::string what = "16-bit D " + std::to_string(D) + " grid " + std::to_string(grid);
            expectComponentsExact<SumOp, 2>(p_each, grid, 128, D, 14, "each row " + what, PathWeights{2 * D, 8 + 2 * D});
            expectComponentsExact<MinOp, 2>(p_bursts, grid, 32, D, 6, "bursts " + what);
        }
        for (uint32_t D = 2; D <= 3; ++D) {
            const std::string what = "32-bit D " + std::to_string(D) + " grid " + std::to_string(grid);
            expectComponentsExact<MinOp, 4>(p_each, grid, 32, D, 12, "each row " + what, PathWeights{4 * D, 8 + 4 * D});
            expectComponentsExact<SumOp, 4>(p_bursts, grid, 128, D, 4, "bursts " + what);
        }
    }
}

// 1285: the RK geometry: K tiles cover the row, no K tile is empty, a K-tiled row has at most 16 workers, every staged
// tile fits the lanes' copies and the buffer; the auto tile cap and the ranges. 1288: 4 rows per worker exactly on
// narrow rows (one K tile, W >= 16), within the narrow buffer; 2 elsewhere, within 1285's.
TEST(RaggedKRWalk, RKGeometryCoversRowsAndFitsTheBuffers) {
    for (uint32_t cap : {32U, 64U, 128U, 256U, 512U}) {
        for (uint64_t R = 16; R <= 5000; R += (R < 600 ? 1 : 37)) {
            const RaggedRKGeometry g = raggedRKGeometry(R, cap);
            ASSERT_GE(g.chunks, 1U);
            ASSERT_LE(g.chunks, 32U);
            ASSERT_EQ(g.workers, 32U / g.chunks);
            ASSERT_EQ(g.contiguous, g.ktiles == 1);
            uint64_t covered = 0;
            for (uint32_t k = 0; k < g.ktiles; ++k) {
                const uint32_t t = raggedRKTileBytesOf(g, R, k);
                ASSERT_GE(t, 1U) << R << " cap " << cap << " tile " << k;
                ASSERT_LE(t, g.tile_bytes);
                covered += t;
            }
            ASSERT_EQ(covered, R) << R << " cap " << cap;
            if (g.contiguous) {
                ASSERT_LE(R, cap);
                ASSERT_EQ(raggedRKRowsPerWorker(g), g.workers >= 16 ? 4U : 2U) << R;  // 1288: narrow rows
                ASSERT_LE(15 + raggedRKTileRows(g) * R, 16U * raggedRKStagedChunksFor(raggedRKRowsPerWorker(g))) << R;
            } else {
                ASSERT_LE(g.tile_bytes, cap);
                ASSERT_EQ(g.tile_bytes % 16, 0U);
                ASSERT_LE(g.workers, 16U) << R << " cap " << cap;
                ASSERT_EQ(raggedRKRowsPerWorker(g), 2U) << R;
                ASSERT_LE(raggedRKTileRows(g) * (g.chunks + 1), 3U * 32U) << R << " cap " << cap;
            }
            ASSERT_LE(raggedRKTileRows(g) * (g.contiguous ? 0U : g.chunks + 1), RAGGED_RK_STAGED_CHUNKS);
            for (uint32_t shift = 0; shift < 16; ++shift)
                for (uint32_t j = 0; j < raggedRKTileRows(g); ++j)
                    ASSERT_LE(raggedRKStagedOffset(g, R, shift, j, g.chunks - 1) + 20,
                              raggedRKSpanBytesFor(raggedRKRowsPerWorker(g)))
                        << R << " cap " << cap << " shift " << shift << " row " << j;
        }
    }
    EXPECT_EQ(raggedRKGeometry(16, 512).workers, 32U);
    EXPECT_EQ(raggedRKRowsPerWorker(raggedRKGeometry(16, 512)), 4U);
    EXPECT_EQ(raggedRKRowsPerWorker(raggedRKGeometry(17, 512)), 4U);
    EXPECT_EQ(raggedRKRowsPerWorker(raggedRKGeometry(32, 32)), 4U);
    EXPECT_EQ(raggedRKRowsPerWorker(raggedRKGeometry(33, 512)), 2U);
    EXPECT_EQ(raggedRKRowsPerWorker(raggedRKGeometry(40, 32)), 2U);  // K-tiled
    EXPECT_EQ(raggedRKGeometry(512, 512).workers, 1U);
    EXPECT_EQ(raggedRKGeometry(513, 512).ktiles, 2U);
    EXPECT_EQ(raggedRKGeometry(513, 512).tile_bytes, 272U);
    EXPECT_EQ(raggedRKGeometry(5000, 128).ktiles, 40U);
    EXPECT_EQ(raggedRKAutoTileCap(16ULL << 20, 2720, 1), 32U);
    EXPECT_EQ(raggedRKAutoTileCap(64ULL << 20, 2720, 1), 128U);
    EXPECT_EQ(raggedRKAutoTileCap(256ULL << 20, 2720, 1), 512U);
    EXPECT_EQ(raggedRKAutoTileCap(1ULL << 20, 2720, 1), 32U);
    EXPECT_EQ(raggedRKRanges(2720, 40, 4096), 68U);
    EXPECT_EQ(raggedRKRanges(2720, 4000, 4096), 1U);
    EXPECT_EQ(raggedRKRanges(20000, 1, 4096), 4096U);
}

// 1285: the RK kernels against direct per-segment, per-column SUM and MIN: every distribution, D from 16 to 1100 (one
// contiguous K tile at every W, and K-tiled rows), tile caps 32 to 512, misaligned bases, rings of 32 and 128, grids
// of 1 to more ranges than events; dense ends, bursts of empty segments, clamped offsets and no rows.
TEST(RaggedKRWalk, EmulatedRKKernelsMatchDirectSumAndMin) {
    const uint64_t ds[] = {16, 17, 24, 31, 33, 48, 64, 100, 129, 255, 256, 300, 512, 513, 700, 1100};
    const uint32_t caps[] = {512, 32, 128, 64, 256};
    for (uint64_t trial = 0; trial < 16; ++trial) {
        const uint64_t seed = A::mix(0x1285 + trial);
        const uint64_t D = ds[trial];
        const uint64_t rows = seed % (D > 256 ? 700 : 2500);
        const uint64_t segments = 1 + A::mix(seed) % (trial % 3 == 0 ? std::max<uint64_t>(rows, 1) : 300);
        const PathWeights weights[] = {PathWeights{}, PathWeights{D, 4 + 4 * D + 4}, PathWeights{D, 8 + 4 * D + 4}};
        for (A::Distribution d : A::DISTRIBUTIONS) {
            const std::vector<uint64_t> values = A::makeOffsets(rows, segments, d, seed);
            const Stored stored(values, trial % 2 ? 8 : 4);
            const Problem p = makeProblem(Offsets{stored.bytes.data(), trial % 2 ? 8U : 4U}, segments, rows);
            for (uint32_t ranges : {1U, 7U, 40U, static_cast<uint32_t>(std::min<uint64_t>(p.pathLength() + 5, 900))}) {
                const uint32_t ring = (ranges + trial) % 2 ? 32U : 128U;
                const PathWeights w = weights[(ranges + trial) % 3];
                RKSetup setup;
                setup.ranges = ranges;
                setup.ring = ring;
                setup.D = D;
                setup.head = static_cast<uint32_t>((seed + ranges) % 16);
                setup.tile_cap = caps[(trial + ranges) % 5];
                const std::string what = std::string(A::distributionName(d)) + " seed " + std::to_string(seed) +
                                         " D " + std::to_string(D) + " cap " + std::to_string(setup.tile_cap) +
                                         " ranges " + std::to_string(ranges) + " ring " + std::to_string(ring) +
                                         " head " + std::to_string(setup.head);
                expectRKExact<SumOp>(p, setup, what + " sum", w);
                setup.head = (setup.head + 7) % 16;
                expectRKExact<MinOp>(p, setup, what + " min", w);
            }
        }
    }
    std::vector<uint64_t> each(601), bursts(1501, 0);
    for (size_t i = 0; i < each.size(); ++i) each[i] = i;
    for (size_t i = 0; i < bursts.size(); ++i) bursts[i] = (i / 300) * 7;
    const Stored s_each(each, 4), s_bursts(bursts, 4), s_zeros(std::vector<uint64_t>(41, 0), 8);
    const Problem p_each = makeProblem(Offsets{s_each.bytes.data(), 4}, 600, 600);
    const Problem p_bursts = makeProblem(Offsets{s_bursts.bytes.data(), 4}, 1500, bursts.back());
    const Problem p_empty = makeProblem(Offsets{s_zeros.bytes.data(), 8}, 40, 100);
    const std::vector<uint64_t> wide = A::makeOffsets(1500, 37, A::Distribution::Pareto, 99);
    const Stored s_wide(wide, 8);
    const Problem p_clamped = makeProblem(Offsets{s_wide.bytes.data(), 8}, 37, 900);
    for (uint64_t D : {16ULL, 40ULL, 600ULL}) {
        for (uint32_t ranges : {1U, 5U, 33U}) {
            RKSetup setup;
            setup.ranges = ranges;
            setup.D = D;
            setup.head = 3;
            setup.tile_cap = D == 600 ? 128 : 512;
            const std::string what = "D " + std::to_string(D) + " ranges " + std::to_string(ranges);
            expectRKExact<SumOp>(p_each, setup, "each row " + what, PathWeights{D, 4 + 4 * D + 4});
            setup.ring = 32;
            expectRKExact<MinOp>(p_bursts, setup, "bursts " + what);
            expectRKExact<SumOp>(p_empty, setup, "no rows " + what);
            setup.head = 0;
            expectRKExact<MinOp>(p_clamped, setup, "clamped " + what, PathWeights{D, 8 + 4 * D + 4});
        }
    }
}

// 1301: raggedFp8ArgKeys orders every pair of bytes as fp8ArgKeys4 does (both operations, both formats), puts -0 on +0's
// key and NaN at 0, never produces the 0xFF marker, and keys each byte of a word independently; raggedFp8ArgFold16 is
// the smallest (key, row) of its 16 rows in the tile frame, the lowest row winning ties, with marked rows losing.
TEST(RaggedKRWalk, ArgFp8LeanKeysAndSixteenRowFold) {
    using namespace ThorImplementation::CubReductionInternal;
    for (int maximum = 0; maximum < 2; ++maximum) {
        for (int e5m2 = 0; e5m2 < 2; ++e5m2) {
            const Fp8ArgFold fold = fp8ArgFold(maximum != 0, e5m2 != 0);
            for (uint32_t a = 0; a < 256; ++a) {
                const uint32_t ka = RaggedMergePath::raggedFp8ArgKeys(a, fold) & 0xFFU;
                const uint32_t ra = fp8ArgKeys4(a, fold).keys & 0xFFU;
                ASSERT_NE(ka, 0xFFU) << a;
                ASSERT_EQ(ka == 0, ra == 0) << "NaN " << a;
                for (uint32_t b = 0; b < 256; ++b) {
                    const uint32_t kb = RaggedMergePath::raggedFp8ArgKeys(b, fold) & 0xFFU;
                    const uint32_t rb = fp8ArgKeys4(b, fold).keys & 0xFFU;
                    ASSERT_EQ(ka < kb, ra < rb) << a << " " << b << " max " << maximum << " e5m2 " << e5m2;
                    ASSERT_EQ(ka == kb, ra == rb) << a << " " << b << " max " << maximum << " e5m2 " << e5m2;
                }
            }
            EXPECT_EQ(RaggedMergePath::raggedFp8ArgKeys(0x80U, fold) & 0xFFU,
                      RaggedMergePath::raggedFp8ArgKeys(0x00U, fold) & 0xFFU);
        }
    }
    uint64_t h = 1301;
    for (int trial = 0; trial < 20000; ++trial) {
        h = ThorImplementation::RaggedReductionAudit::mix(h);
        const Fp8ArgFold fold = fp8ArgFold((h >> 40) & 1U, (h >> 41) & 1U);
        uint32_t keys[4];
        for (uint32_t w = 0; w < 4; ++w) {
            const uint32_t packed = static_cast<uint32_t>(ThorImplementation::RaggedReductionAudit::mix(h + w));
            keys[w] = RaggedMergePath::raggedFp8ArgKeys(packed, fold);
            for (uint32_t b = 0; b < 4; ++b) {
                ASSERT_EQ((keys[w] >> (8 * b)) & 0xFFU,
                          RaggedMergePath::raggedFp8ArgKeys((packed >> (8 * b)) & 0xFFU, fold) & 0xFFU);
            }
        }
        // Mark a random prefix and suffix of the rows (a tile's edge lanes), sometimes all of them.
        const uint32_t lo = static_cast<uint32_t>(h % 17), hi = lo + static_cast<uint32_t>((h >> 8) % (17 - lo));
        for (uint32_t k = 0; k < 16; ++k)
            if (k < lo || k >= hi) keys[k >> 2] |= 0xFFU << (8 * (k & 3U));
        const uint32_t first_field = 1 + static_cast<uint32_t>((h >> 16) % 497);
        uint32_t expected = 0xFFFFFFFFU;
        for (uint32_t k = 0; k < 16; ++k) {
            const uint32_t c = (((keys[k >> 2] >> (8 * (k & 3U))) & 0xFFU) << 24) | ((first_field + k) << 1);
            expected = c < expected ? c : expected;
        }
        ASSERT_EQ(RaggedMergePath::raggedFp8ArgFold16(keys, first_field), expected) << "trial " << trial;
    }
}

// 1311: raggedF16ArgKeys orders every 16-bit pattern as f16ArgKeys2 does (both operations, FP16 and BF16): sorted by the
// reference key, the lean keys rise exactly where the reference keys rise. NaN at 0, -0 on +0, never 0xFFFF, each half
// keyed independently; raggedF16ArgFold8 is the smallest (key, row) of its 8 rows, marked rows losing.
TEST(RaggedKRWalk, ArgF16LeanKeysAndEightRowFold) {
    using namespace ThorImplementation::CubReductionInternal;
    for (int maximum = 0; maximum < 2; ++maximum) {
        for (int bf16 = 0; bf16 < 2; ++bf16) {
            const F16ArgFold reference_fold = f16ArgFold(maximum != 0, bf16 != 0);
            const Fp8ArgFold fold{reference_fold.xor_mask, reference_fold.nan_bias};
            std::vector<std::pair<uint32_t, uint32_t>> keys;  // (reference, lean)
            for (uint32_t x = 0; x < 65536; ++x) {
                const uint32_t reference = f16ArgKeys2(x, reference_fold).keys & 0xFFFFU;
                const uint32_t lean = RaggedMergePath::raggedF16ArgKeys(x, fold) & 0xFFFFU;
                ASSERT_NE(lean, 0xFFFFU) << x;
                ASSERT_EQ(lean == 0, reference == 0) << "NaN " << x;
                ASSERT_EQ(RaggedMergePath::raggedF16ArgKeys(x << 16, fold) >> 16, lean) << x;  // the high half alike
                keys.emplace_back(reference, lean);
            }
            std::sort(keys.begin(), keys.end());
            for (size_t i = 1; i < keys.size(); ++i) {
                ASSERT_EQ(keys[i].first > keys[i - 1].first, keys[i].second > keys[i - 1].second)
                    << "max " << maximum << " bf16 " << bf16 << " reference " << keys[i].first;
                ASSERT_GE(keys[i].second, keys[i - 1].second);
            }
            EXPECT_EQ(RaggedMergePath::raggedF16ArgKeys(0x8000U, fold) & 0xFFFFU,
                      RaggedMergePath::raggedF16ArgKeys(0x0000U, fold) & 0xFFFFU);
        }
    }
    uint64_t h = 1311;
    for (int trial = 0; trial < 20000; ++trial) {
        h = ThorImplementation::RaggedReductionAudit::mix(h);
        const F16ArgFold reference_fold = f16ArgFold((h >> 40) & 1U, (h >> 41) & 1U);
        const Fp8ArgFold fold{reference_fold.xor_mask, reference_fold.nan_bias};
        uint32_t keys[4];
        for (uint32_t w = 0; w < 4; ++w) {
            const uint32_t packed = static_cast<uint32_t>(ThorImplementation::RaggedReductionAudit::mix(h + w));
            keys[w] = RaggedMergePath::raggedF16ArgKeys(packed, fold);
            ASSERT_EQ(keys[w] & 0xFFFFU, RaggedMergePath::raggedF16ArgKeys(packed & 0xFFFFU, fold) & 0xFFFFU);
            ASSERT_EQ(keys[w] >> 16, RaggedMergePath::raggedF16ArgKeys(packed >> 16, fold) & 0xFFFFU);
        }
        const uint32_t lo = static_cast<uint32_t>(h % 9), hi = lo + static_cast<uint32_t>((h >> 8) % (9 - lo));
        for (uint32_t k = 0; k < 8; ++k)
            if (k < lo || k >= hi) keys[k >> 1] |= 0xFFFFU << (16 * (k & 1U));
        const uint32_t first_field = 1 + static_cast<uint32_t>((h >> 16) % 249);
        uint32_t expected = 0xFFFFFFFFU;
        for (uint32_t k = 0; k < 8; ++k) {
            const uint32_t c = (((keys[k >> 1] >> (16 * (k & 1U))) & 0xFFFFU) << 16) | ((first_field + k) << 1);
            expected = c < expected ? c : expected;
        }
        ASSERT_EQ(RaggedMergePath::raggedF16ArgFold8(keys, first_field), expected) << "trial " << trial;
    }
}

// 1313: raggedFp32ArgKey orders FP32 patterns as Thor's ARG contract does (both operations): NaN first (key 0, every
// NaN equal), then by value, -0 equal to +0; never 0xFFFFFFFF. Checked on pairs drawn from specials, the neighbours of
// powers of two, random patterns and random patterns sharing high bits; the order is the one fp32ArgKey gives
// (Fp32ArgKey.cuh), whose keys differ only by one on negatives. raggedFp32ArgFold4 is the smallest (key, row) of its 4
// rows, marked rows losing; the pair records' layout.
TEST(RaggedKRWalk, ArgFp32LeanKeysFourRowFoldAndPairRecords) {
    using namespace ThorImplementation::CubReductionInternal;
    const auto value_of = [](uint32_t bits) {
        float f;
        std::memcpy(&f, &bits, 4);
        return f;
    };
    // -1: a before b; 0: tie; 1: b before a (argmin; argmax on the negated values).
    const auto order = [&](uint32_t a, uint32_t b, bool maximum) {
        const float x = value_of(a), y = value_of(b);
        if (std::isnan(x) || std::isnan(y)) return std::isnan(x) && std::isnan(y) ? 0 : (std::isnan(x) ? -1 : 1);
        const float u = maximum ? -x : x, v = maximum ? -y : y;
        return u < v ? -1 : (u > v ? 1 : 0);
    };
    std::vector<uint32_t> patterns = {0x00000000U, 0x80000000U, 0x00000001U, 0x80000001U, 0x007FFFFFU, 0x807FFFFFU,
                                      0x00800000U, 0x80800000U, 0x3F800000U, 0xBF800000U, 0x7F7FFFFFU, 0xFF7FFFFFU,
                                      0x7F800000U, 0xFF800000U, 0x7F800001U, 0xFF800001U, 0x7FC00000U, 0xFFC00000U,
                                      0x7FFFFFFFU, 0xFFFFFFFFU};
    for (uint32_t e = 0; e < 256; ++e) {
        for (uint32_t sign : {0U, 0x80000000U}) {
            patterns.push_back(sign | (e << 23));
            patterns.push_back(sign | (e << 23) | 1U);
            patterns.push_back(sign | (e << 23) | 0x7FFFFFU);
        }
    }
    const size_t structured = patterns.size();  // every pair of these is compared, the random ones' pairs sampled
    uint64_t h = 1313;
    for (int i = 0; i < 4000; ++i) {
        h = ThorImplementation::RaggedReductionAudit::mix(h);
        patterns.push_back(static_cast<uint32_t>(h));
        patterns.push_back((static_cast<uint32_t>(h >> 32) & 0xFFFF0000U) | (static_cast<uint32_t>(h) & 0xFFU));
    }
    for (int maximum = 0; maximum < 2; ++maximum) {
        const uint32_t xor_mask = maximum ? 0x80000000U : 0U;
        for (uint32_t a : patterns) {
            const uint32_t ka = RaggedMergePath::raggedFp32ArgKey(a, xor_mask);
            ASSERT_NE(ka, 0xFFFFFFFFU) << a;
            ASSERT_EQ(ka == 0, std::isnan(value_of(a))) << "NaN " << a;
        }
        for (size_t i = 0; i < patterns.size(); ++i) {
            // Every structured pair; each pattern against a seventh of the others.
            const bool all = i < structured;
            for (size_t j = all ? 0 : i % 7; j < patterns.size(); j += (all && j < structured) ? 1 : 7) {
                const uint32_t a = patterns[i], b = patterns[j];
                const uint32_t ka = RaggedMergePath::raggedFp32ArgKey(a, xor_mask);
                const uint32_t kb = RaggedMergePath::raggedFp32ArgKey(b, xor_mask);
                const int expected = order(a, b, maximum != 0);
                ASSERT_EQ(ka < kb ? -1 : (ka > kb ? 1 : 0), expected)
                    << "max " << maximum << " a " << std::hex << a << " b " << b;
            }
        }
    }
    for (int trial = 0; trial < 20000; ++trial) {
        h = ThorImplementation::RaggedReductionAudit::mix(h);
        uint32_t keys[4];
        for (uint32_t k = 0; k < 4; ++k) {
            const uint64_t x = ThorImplementation::RaggedReductionAudit::mix(h + k);
            // Ties are common: a few keys, or any key.
            keys[k] = (h >> 50) & 1U ? static_cast<uint32_t>(x % 3) * 0x40000000U : static_cast<uint32_t>(x);
            if (keys[k] == 0xFFFFFFFFU) keys[k] = 0;
        }
        const uint32_t lo = static_cast<uint32_t>(h % 5), hi = lo + static_cast<uint32_t>((h >> 8) % (5 - lo));
        for (uint32_t k = 0; k < 4; ++k)
            if (k < lo || k >= hi) keys[k] = 0xFFFFFFFFU;
        const uint32_t first_field = 1 + static_cast<uint32_t>((h >> 16) % 1000000);
        uint64_t expected = ~uint64_t{0};
        for (uint32_t k = 0; k < 4; ++k) {
            const uint64_t c = (uint64_t{keys[k]} << 32) | ((first_field + k) << 1);
            expected = c < expected ? c : expected;
        }
        ASSERT_EQ(RaggedMergePath::raggedFp32ArgFold4(keys, first_field), expected) << "trial " << trial;
    }
    // The pair records: 16-byte aligned arrays, CTA c of component k at k * stride + c, stride the grid rounded to 4.
    for (uint32_t grid : {1U, 3U, 4U, 5U, 170U, 4095U, 4096U}) {
        for (uint32_t D : {1U, 2U, 3U, 15U}) {
            const RaggedMergePath::RaggedArgPairRecordLayout layout = RaggedMergePath::raggedArgPairRecordLayout(grid, D);
            const uint64_t stride = (grid + 3U) / 4U * 4U;
            ASSERT_EQ(layout.stride, stride);
            ASSERT_EQ(layout.keys, 0U);
            ASSERT_EQ(layout.head_keys, 8U * stride);
            ASSERT_EQ(layout.head_rows, layout.head_keys + 4U * stride * D);
            ASSERT_EQ(layout.tail_keys, layout.head_rows + 8U * stride * D);
            ASSERT_EQ(layout.tail_rows, layout.tail_keys + 4U * stride * D);
            ASSERT_EQ(layout.bytes, layout.tail_rows + 8U * stride * D);
            for (uint64_t at : {layout.head_keys, layout.head_rows, layout.tail_keys, layout.tail_rows, layout.bytes})
                ASSERT_EQ(at % 16U, 0U) << grid << " " << D;
        }
    }
    // The local words' stores: the CTA's first row added, the identity the sentinel / the pair identity.
    const uint64_t word = (uint64_t{0x80001234U} << 32) | (uint64_t{77} << 1);
    EXPECT_EQ(RaggedMergePath::raggedArgLocalOutputIndex(word, 1000, 3, 2, 8), (1000U + 77U) * 3U + 2U);
    EXPECT_EQ(RaggedMergePath::raggedArgLocalOutputIndex(~uint64_t{0}, 1000, 3, 2, 4), 0xFFFFFFFFU);
    EXPECT_EQ(RaggedMergePath::raggedArgLocalWordGlobalRow(word, uint64_t{1} << 40), (uint64_t{1} << 40) + 77U);
    EXPECT_EQ(RaggedMergePath::raggedArgLocalWordKey(word), 0x80001234U);
    EXPECT_EQ(RaggedMergePath::raggedArgLocalWordGlobalRow(~uint64_t{0}, 5), ~uint64_t{0});
    EXPECT_EQ(RaggedMergePath::raggedArgLocalWordKey(~uint64_t{0}), 0xFFFFFFFFU);
    EXPECT_EQ(RaggedMergePath::raggedArgPairOutputIndex(~uint64_t{0}, 3, 1, 8), ~uint64_t{0});
    EXPECT_EQ(RaggedMergePath::raggedArgPairOutputIndex(9, 3, 1, 4), 28U);
}
