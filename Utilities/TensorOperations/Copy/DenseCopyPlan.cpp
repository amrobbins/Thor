// 1388: the host planning of a strided span's dense copy (DenseCopyPlan.h has the rules; 1389: rectangular
// transpose tiles, any number of batch axes; 1389b: the packed gather; 1389d: the smallest tile in words; 1389e:
// plain rotations on the matrix transpose kernel; 1389f: rows of two to four 16-byte words transposed from 4 MiB).
// No CUDA.

#include "Utilities/TensorOperations/Copy/DenseCopyPlan.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ThorImplementation {
namespace DenseCopyInternal {
namespace {

[[nodiscard]] uint64_t ceilDiv(uint64_t a, uint64_t b) { return a / b + (a % b != 0); }

[[nodiscard]] uint64_t checkedProduct(uint64_t a, uint64_t b) {
    if (b != 0 && a > std::numeric_limits<uint64_t>::max() / b)
        throw std::invalid_argument("Dense copy: the element count overflows 64 bits.");
    return a * b;
}

struct Axis {
    uint64_t dim, stride;
};

// Extent-one axes dropped; an axis merged into the next inner one when its stride is that one's extent times stride.
// Outer to inner, as the destination's row-major order.
[[nodiscard]] std::vector<Axis> canonical(const std::vector<uint64_t>& dims, const std::vector<uint64_t>& strides) {
    std::vector<Axis> inner_first;
    for (size_t k = dims.size(); k-- > 0;) {
        if (dims[k] == 1) continue;
        if (!inner_first.empty() && inner_first.back().stride <= std::numeric_limits<uint64_t>::max() /
                                                                     std::max<uint64_t>(inner_first.back().dim, 1) &&
            strides[k] == inner_first.back().dim * inner_first.back().stride) {
            inner_first.back().dim = checkedProduct(inner_first.back().dim, dims[k]);
            continue;
        }
        inner_first.push_back({dims[k], strides[k]});
    }
    return {inner_first.rbegin(), inner_first.rend()};
}

struct Tile {
    uint64_t x, v;
};

// The transpose tile for x by v vectors of p words (vector_bytes each) in `vectors` vectors in all (DenseCopyPlan.h).
[[nodiscard]] Tile transposeTile(uint64_t x, uint64_t v, uint64_t vectors, uint64_t p, uint32_t word_bytes) {
    const uint64_t vector_bytes = p * word_bytes;
    const uint64_t full = std::max<uint64_t>(32, ceilDiv(128, vector_bytes));  // a tile row of 128 bytes or more
    // 1389d: the smallest side is the one whose square tile is TRANSPOSE_MIN_TILE_WORDS words (32 vectors of one
    // word, 19 of 3, 13 of 7), not 32 vectors whatever their width.
    uint64_t side = 1;
    while (side * side * p < TRANSPOSE_MIN_TILE_WORDS) ++side;
    while (side < full && (side + 1) * (side + 1) * TRANSPOSE_TARGET_TILES <= vectors) ++side;
    const uint64_t area = side * side;
    Tile tile{std::min(x, side), std::min(v, side)};
    // 1389l: a packed byte tile with TRANSPOSE_PACKED_MIN_BLOCKS 4 x 4 blocks or more (1389m: 320, through 1389l one a
    // thread, 256) keeps its sides (DenseCopyPlan.h).
    const bool packed_blocks = transposePacksBytes(p, word_bytes) &&
                               ceilDiv(tile.x, 4) * ceilDiv(tile.v, 4) >= TRANSPOSE_PACKED_MIN_BLOCKS;
    if (packed_blocks) {
    } else if (tile.x < side) {
        tile.v = std::min(v, std::max(tile.v, area / tile.x));
    } else if (tile.v < side) {
        tile.x = std::min(x, std::max(tile.x, area / tile.v));
        // 1389t: a packed byte tile grown along x past four full sides stops there if it leaves fewer than
        // TRANSPOSE_PACKED_MIN_GROWN_TILES tiles (DenseCopyPlan.h).
        const uint64_t tiles = ceilDiv(x, tile.x) * ceilDiv(v, tile.v) * (vectors / (x * v));
        if (transposePacksBytes(p, word_bytes) && tile.x > 4 * full && tiles < TRANSPOSE_PACKED_MIN_GROWN_TILES)
            tile.x = 4 * full;
    }
    const auto shared = [&](const Tile& t) { return transposeSharedBytes(t.x, t.v, p, word_bytes); };
    const auto fit = [&](Tile& t) {
        while (shared(t) > TRANSPOSE_MAX_SHARED_BYTES) {
            if (t.v >= t.x) t.v -= std::max<uint64_t>(1, t.v / 8);
            else t.x -= std::max<uint64_t>(1, t.x / 8);
        }
    };
    // 1389r: unpacked single bytes from TRANSPOSE_TALL_BYTES_MIN_BYTES: the tall tile, if it holds the square's
    // vectors or more (DenseCopyPlan.h).
    if (word_bytes == 1 && !transposePacksBytes(p, word_bytes) &&
        vectors * vector_bytes >= TRANSPOSE_TALL_BYTES_MIN_BYTES) {
        Tile tall{std::min(x, ceilDiv(TRANSPOSE_TALL_BYTES_ROW_BYTES, vector_bytes)), v};
        fit(tall);
        if (tall.x * tall.v >= area) return tall;
    }
    fit(tile);
    return tile;
}

// 1389s: the wide tile of a rotation of single words of word_bytes: four sides along x, a quarter side (at least 8)
// along v, the side a tile row of 128 bytes (64 2-byte, 32 4-byte words).
[[nodiscard]] Tile wideRotationTile(uint64_t x, uint64_t v, uint32_t word_bytes) {
    const uint64_t side = std::max<uint64_t>(32, ceilDiv(128, word_bytes));
    return {std::min(x, 4 * side), std::min(v, std::max<uint64_t>(8, side / 4))};
}

// 1389e: whether the [x, v] rotation of a dense [v, x] matrix of `word`-byte words goes to the matrix transpose
// kernel (DenseCopyPlan.h has the measurements behind the sizes).
[[nodiscard]] bool matrixTransposeFits(uint64_t x, uint64_t v, uint32_t word) {
    if (word > 4) return false;
    if (transposePacksBytes(1, word)) return false;  // 1389k: single bytes take the packed byte tile (DenseCopyPlan.h)
    const uint64_t side = MATRIX_TRANSPOSE_SIDE_BYTES / word;  // words: 128, 64 or 32
    if (std::min(x, v) < side) return false;                   // a skinny side: the rectangular tile
    if (x >= MAX_32BIT_INDEX / v) return false;                // its int indices: x v under 2^31
    if (ceilDiv(v, side) > MATRIX_TRANSPOSE_MAX_ROW_TILES) return false;
    // 1389i: long source rows (x of TRANSPOSE_ALONG_V_RATIO v or more) from MATRIX_TRANSPOSE_MAX_LONG_ROW_BYTES: the
    // transpose kernel's tiles along v.
    if (x >= TRANSPOSE_ALONG_V_RATIO * v && x * v * word >= MATRIX_TRANSPOSE_MAX_LONG_ROW_BYTES) return false;
    return word == 4 || x * v * word >= MATRIX_TRANSPOSE_MIN_NARROW_BYTES;
}

}  // namespace

DenseCopyPlan planLayout(const std::vector<uint64_t>& dims, const std::vector<uint64_t>& strides,
                         uint32_t element_bytes, uintptr_t source_address, uintptr_t destination_address) {
    if (dims.empty() || dims.size() != strides.size() || dims.size() > MAX_RANK)
        throw std::invalid_argument("Dense copy: dimensions and strides need the same rank, 1 to 32.");
    if (element_bytes != 1 && element_bytes != 2 && element_bytes != 4 && element_bytes != 8)
        throw std::invalid_argument("Dense copy does not support the element size " + std::to_string(element_bytes) +
                                    " bytes.");
    DenseCopyPlan plan;
    plan.element_bytes = element_bytes;
    plan.word_bytes = element_bytes;
    plan.source_dims = dims;
    plan.source_strides = strides;
    uint64_t total = 1;
    for (uint64_t d : dims) total = checkedProduct(total, d);
    plan.words = total;
    if (total == 0) return plan;  // Empty

    const std::vector<Axis> axes = canonical(dims, strides);
    if (axes.empty() || (axes.size() == 1 && axes[0].stride == 1)) {
        plan.family = DenseCopyFamily::Memcpy;
        return plan;
    }

    // Rows: the innermost axis when contiguous, else single elements.
    const bool contiguous_rows = axes.back().stride == 1;
    const uint64_t p = contiguous_rows ? axes.back().dim : 1;
    const std::vector<Axis> outer(axes.begin(), axes.end() - (contiguous_rows ? 1 : 0));

    // The widest word that divides the rows, the outer strides and both addresses.
    uint32_t word = element_bytes;
    for (uint32_t candidate : {16u, 8u, 4u, 2u}) {
        if (candidate <= element_bytes || candidate % element_bytes != 0) continue;
        bool fits = (p * element_bytes) % candidate == 0 && source_address % candidate == 0 &&
                    destination_address % candidate == 0;
        for (const Axis& axis : outer) fits = fits && (axis.stride * element_bytes) % candidate == 0;
        if (fits) {
            word = candidate;
            break;
        }
    }
    const uint64_t per_word = word / element_bytes;
    plan.word_bytes = word;
    plan.words = total / per_word;
    plan.p = p / per_word;
    std::vector<uint64_t> outer_dims, outer_strides, dest_strides(outer.size());
    for (const Axis& axis : outer) {
        outer_dims.push_back(axis.dim);
        outer_strides.push_back(axis.stride / per_word);
    }
    uint64_t dest_stride = plan.p;
    for (size_t k = outer.size(); k-- > 0;) {
        dest_strides[k] = dest_stride;
        dest_stride *= outer_dims[k];
    }

    // Transpose: rows under a sector and an x axis (not the innermost outer axis v) whose rows are adjacent in the
    // source; the other outer axes are batch axes; 32-bit tile indices; an x by v plane of at least 256 vectors (a
    // smaller one is local to the row copy's reads anyway). 1389f: rows of one to two sectors of 16-byte words too
    // (32, 48 or 64 bytes), in a copy of TRANSPOSE_MIN_WIDE_BYTES or more (DenseCopyPlan.h has the measurements).
    const uint64_t row_bytes = p * element_bytes;
    const bool narrow_rows = row_bytes < SECTOR_BYTES;
    const bool wide_rows = !narrow_rows && row_bytes <= TRANSPOSE_MAX_WIDE_ROW_BYTES && word == 16 &&
                           total >= TRANSPOSE_MIN_WIDE_BYTES / element_bytes;
    if ((narrow_rows || wide_rows) && outer.size() >= 2) {
        const size_t v_axis = outer.size() - 1;
        size_t x_axis = outer.size();
        for (size_t k = 0; k < v_axis; ++k)
            if (outer_strides[k] == plan.p) x_axis = k;  // the innermost such axis
        if (x_axis != outer.size() && outer_dims[x_axis] < MAX_32BIT_INDEX && outer_dims[v_axis] < MAX_32BIT_INDEX) {
            // 1389s: a plain rotation of 2- or 4-byte words with x of 7/8 v or more, from
            // TRANSPOSE_WIDE_ROTATION_MIN_BYTES: the wide tile along v (DenseCopyPlan.h).
            const bool plain_rotation =
                outer.size() == 2 && plan.p == 1 && outer_strides[v_axis] == outer_dims[x_axis];
            const bool wide_rotation =
                plain_rotation && (word == 2 || word == 4) &&
                8 * outer_dims[x_axis] >= TRANSPOSE_WIDE_ROTATION_X_EIGHTHS * outer_dims[v_axis] &&
                total * element_bytes >= TRANSPOSE_WIDE_ROTATION_MIN_BYTES;
            // 1389e: a plain rotation of a dense matrix of single words, at the sizes the matrix kernel is as fast
            // or faster. A source with gaps or a broadcast axis (v's stride is not x) stays a Transpose.
            if (plain_rotation && !wide_rotation && matrixTransposeFits(outer_dims[x_axis], outer_dims[v_axis], word)) {
                plan.family = DenseCopyFamily::MatrixTranspose;
                plan.x = outer_dims[x_axis];
                plan.x_dest_stride = dest_strides[x_axis];
                plan.v = outer_dims[v_axis];
                plan.v_source_stride = outer_strides[v_axis];
                return plan;
            }
            const Tile tile = wide_rotation ? wideRotationTile(outer_dims[x_axis], outer_dims[v_axis], word)
                                            : transposeTile(outer_dims[x_axis], outer_dims[v_axis],
                                                            total / per_word / plan.p, plan.p, word);
            uint64_t tiles = ceilDiv(outer_dims[x_axis], tile.x) * ceilDiv(outer_dims[v_axis], tile.v);
            for (size_t k = 0; k < v_axis; ++k)
                if (k != x_axis) tiles = checkedProduct(tiles, outer_dims[k]);
            // 1389a: thousands of tiles of a few hundred words (a small x by v plane under many batch elements) keep
            // 256 threads on loop overhead; the row copy measured 1.3x to 1.75x faster on them.
            const bool many_small_tiles = tile.x * tile.v * plan.p < TRANSPOSE_SMALL_TILE_WORDS &&
                                          tiles > TRANSPOSE_MAX_SMALL_TILES;
            // 1389d: the plane, not the tile (through 1389c a tile under 256 vectors was the whole plane).
            const bool plane = outer_dims[x_axis] * outer_dims[v_axis] >= TRANSPOSE_MIN_PLANE_VECTORS;
            if (tiles < MAX_32BIT_INDEX && plane && !many_small_tiles) {
                plan.family = DenseCopyFamily::Transpose;
                plan.tile_x = static_cast<uint32_t>(tile.x);
                plan.tile_v = static_cast<uint32_t>(tile.v);
                // 1389i: long source rows take the tiles along v, where the two orders differ (DenseCopyPlan.h).
                // 1389k: packed bytes from x of half v.
                const bool long_rows = transposePacksBytes(plan.p, word)
                                           ? TRANSPOSE_BYTES_ALONG_V_DIVISOR * outer_dims[x_axis] >= outer_dims[v_axis]
                                           : outer_dims[x_axis] >= TRANSPOSE_ALONG_V_RATIO * outer_dims[v_axis];
                plan.tile_v_first = (long_rows || wide_rotation) && ceilDiv(outer_dims[x_axis], tile.x) > 1 &&
                                    ceilDiv(outer_dims[v_axis], tile.v) > 1;
                plan.x = outer_dims[x_axis];
                plan.x_dest_stride = dest_strides[x_axis];
                plan.v = outer_dims[v_axis];
                plan.v_source_stride = outer_strides[v_axis];
                for (size_t k = 0; k < v_axis; ++k) {
                    if (k == x_axis) continue;
                    plan.batch_dims.push_back(outer_dims[k]);
                    plan.batch_source.push_back(outer_strides[k]);
                    plan.batch_dest.push_back(dest_strides[k]);
                }
                return plan;
            }
        }
    }

    // 1389b: single words of 4 bytes or fewer over two or more outer axes, packed 16 bytes to a store.
    const bool packed = plan.p == 1 && word <= 4 && outer.size() >= 2 && plan.words < MAX_32BIT_INDEX &&
                        destination_address % PACK_BYTES == 0;
    plan.family = packed ? DenseCopyFamily::PackedGather : DenseCopyFamily::RowCopy;
    plan.outer_dims = std::move(outer_dims);
    plan.outer_strides = std::move(outer_strides);
    return plan;
}

uint64_t workUnits(const DenseCopyPlan& plan) {
    if (plan.family == DenseCopyFamily::RowCopy)
        return ceilDiv(plan.words, uint64_t{THREADS} * rowCopyStepItems(plan));
    if (plan.family == DenseCopyFamily::PackedGather)  // one step even when there is only a tail
        return ceilDiv(std::max<uint64_t>(plan.words / (PACK_BYTES / plan.word_bytes), 1), THREADS);
    if (plan.family == DenseCopyFamily::Transpose) {
        uint64_t tiles = ceilDiv(plan.x, plan.tile_x) * ceilDiv(plan.v, plan.tile_v);
        for (uint64_t d : plan.batch_dims) tiles *= d;
        return tiles;
    }
    return 0;
}

uint32_t gridOf(const DenseCopyPlan& plan) {
    return static_cast<uint32_t>(std::min<uint64_t>(workUnits(plan), 2147483647ULL));
}

uint64_t sharedBytes(const DenseCopyPlan& plan) {
    if (plan.family != DenseCopyFamily::Transpose) return 0;
    return transposeSharedBytes(plan.tile_x, plan.tile_v, plan.p, plan.word_bytes);
}

}  // namespace DenseCopyInternal

const char* denseCopyRevision() { return "1395"; }  // 1395: the planner's row-copy counts only

std::string denseCopyPlanName(const DenseCopyPlan& plan) {
    const std::string w = "_w" + std::to_string(plan.word_bytes);
    switch (plan.family) {
        case DenseCopyFamily::Empty: return "empty";
        case DenseCopyFamily::Memcpy: return "memcpy" + w;
        case DenseCopyFamily::RowCopy:
            return "row_copy_o" + std::to_string(plan.outer_dims.size()) + "_p" + std::to_string(plan.p) + w;
        case DenseCopyFamily::PackedGather: return "packed_gather_o" + std::to_string(plan.outer_dims.size()) + w;
        case DenseCopyFamily::MatrixTranspose: return "matrix_transpose" + w;
        case DenseCopyFamily::Transpose:
            return "transpose_p" + std::to_string(plan.p) + w + "_t" + std::to_string(plan.tile_x) + "x" +
                   std::to_string(plan.tile_v) + "_b" + std::to_string(plan.batch_dims.size()) +
                   (plan.tile_v_first ? "_vfirst" : "");
    }
    return "unknown";
}

}  // namespace ThorImplementation
