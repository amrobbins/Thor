#include "Utilities/TensorOperations/Reduction/KR/KRSegments.h"

#include "Utilities/TensorOperations/Reduction/Rankers/Fp8KRRankerModel.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ThorImplementation::ReductionInternal::KR {
namespace {

uint64_t ceilDiv(uint64_t a, uint64_t b) { return a / b + (a % b != 0); }

uint64_t pow2Ceil(uint64_t x) {
    uint64_t p = 1;
    while (p < x) p <<= 1;
    return p;
}

bool validGroupLanes(uint32_t g) { return g >= 1 && g <= 32 && (g & (g - 1)) == 0; }

void appendUnique(std::vector<KRSegmentShape>& shapes, const KRSegmentShape& shape) {
    if (!fp8KRSegmentShapeLegal(shape)) return;
    if (std::find(shapes.begin(), shapes.end(), shape) == shapes.end()) shapes.push_back(shape);
}

uint64_t align16(uint64_t bytes) { return (bytes + 15) & ~uint64_t{15}; }

// Threads and CTAs of a shape's main kernel, as the launch geometry launches it.
uint64_t shapeThreads(const KRSegmentShape& shape) {
    switch (shape.family()) {
        case KRFamily::Pack: return ceilDiv(shape.rows, shape.rows_per_thread);
        case KRFamily::Group: return shape.rows * shape.group_lanes;
        case KRFamily::Split: return shape.segments() * KR_THREADS;
        case KRFamily::Block: return shape.rows * KR_THREADS;
    }
    return 1;
}

uint64_t shapeBlocks(const KRSegmentShape& shape) {
    switch (shape.family()) {
        case KRFamily::Pack: return ceilDiv(ceilDiv(shape.rows, shape.rows_per_thread), KR_THREADS);
        case KRFamily::Group: return ceilDiv(shape.rows, KR_THREADS / shape.group_lanes);
        case KRFamily::Split: return shape.segments();
        case KRFamily::Block: return shape.rows;
    }
    return 1;
}

double log2Of(double x) { return std::log2(std::max(x, 1e-30)); }

}  // namespace

bool fp8KRSegmentShapeLegal(const KRSegmentShape& shape) {
    if (shape.rows == 0 || shape.row_bytes == 0 || shape.chunk_bytes == 0 || shape.rows_per_thread == 0) return false;
    // Byte offsets (row * row_bytes + head) stay far from 64-bit overflow.
    constexpr uint64_t max_bytes = uint64_t{1} << 62;
    if (shape.row_bytes > max_bytes / shape.rows) return false;
    switch (shape.family()) {
        case KRFamily::Pack:
            if (shape.vector &&
                (!fp8KRVectorRowBytes(shape.row_bytes) || shape.rows_per_thread != krVectorRows(shape.row_bytes)))
                return false;
            return shape.group_lanes == 1 && shape.rows_per_thread <= 16 &&
                   shape.row_bytes <= KR_MAX_PACK_ROW_BYTES &&
                   shape.rows_per_thread * shape.row_bytes <= KR_MAX_PACK_SPAN_BYTES;
        case KRFamily::Group:
            return !shape.vector && validGroupLanes(shape.group_lanes);
        case KRFamily::Block:
            return !shape.vector && shape.row_bytes >= KR_MIN_BLOCK_ROW_BYTES &&
                   shape.rows <= static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
        case KRFamily::Split:
            if (shape.vector) return false;
            return shape.group_lanes == KR_THREADS && shape.chunk_bytes % 16 == 0 &&
                   shape.chunk_bytes >= KR_MIN_CHUNK_BYTES && shape.chunks() <= KR_MAX_CHUNKS &&
                   shape.segments() <= static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
    }
    return false;
}

uint64_t krScratchBytes(const KRSegmentShape& shape, bool arg) {
    if (!retainedKRShapeLegalForAnyElement(shape)) throw std::invalid_argument("Illegal retained FP8 KR segment shape");
    if (shape.family() != KRFamily::Split) return 0;
    return align16(shape.segments() * (arg ? sizeof(uint64_t) : sizeof(float)));
}

std::string krShapeName(const KRSegmentShape& shape) {
    switch (shape.family()) {
        case KRFamily::Pack: return (shape.vector ? "v" : "p") + std::to_string(shape.rows_per_thread);
        case KRFamily::Group: return "g" + std::to_string(shape.group_lanes);
        case KRFamily::Split: return "s" + std::to_string(shape.chunk_bytes);
        case KRFamily::Block: return "b";
    }
    return "unknown";
}

std::vector<KRSegmentShape> fp8KRCandidateShapes(uint64_t rows, uint64_t row_bytes, uint32_t multiprocessors) {
    if (rows == 0 || row_bytes == 0) return {};
    const uint64_t sms = std::max<uint32_t>(multiprocessors, 1);
    std::vector<KRSegmentShape> pack, group, split;

    // Pack: short rows; the vector shape where compiled, then about 16 and 8 bytes per thread.
    if (fp8KRVectorRowBytes(row_bytes) && rows >= krVectorRows(row_bytes)) {
        appendUnique(pack, {rows, row_bytes, row_bytes, 1, krVectorRows(row_bytes), true});
    }
    if (row_bytes <= 16) {
        for (uint64_t span : {16u, 8u}) {
            const uint64_t per_thread = std::max<uint64_t>(2, span / row_bytes);
            appendUnique(pack, {rows, row_bytes, row_bytes, 1, static_cast<uint32_t>(per_thread), false});
        }
    }

    // Group: a lane with no word of its row only idles, so G stops at the row's word count.
    const uint64_t max_g = std::min<uint64_t>(32, pow2Ceil(ceilDiv(row_bytes + 3, 4)));
    const uint64_t g0 = std::min<uint64_t>(max_g, std::clamp<uint64_t>(pow2Ceil(ceilDiv(row_bytes, 32)), 1, 32));
    std::vector<uint64_t> widths{g0};
    for (uint64_t step = 2; step <= 32; step *= 2) {
        if (g0 * step <= max_g) widths.push_back(g0 * step);
        if (g0 / step >= 1 && g0 % step == 0) widths.push_back(g0 / step);
    }
    for (uint64_t g : widths) appendUnique(group, {rows, row_bytes, row_bytes, static_cast<uint32_t>(g), 1, false});

    const auto add_split = [&](uint64_t chunk) {
        if (chunk < row_bytes) appendUnique(split, {rows, row_bytes, chunk, KR_THREADS, 1, false});
    };
    const bool few_rows = rows < sms * 16;
    // Rows of 16 KiB or more also split when rows alone fill the GPU (in a sweep, ARG Group on such rows, with few CTAs
    // per SM, lost 5-15% to Split at 3k-7k rows, and Split was not timed past 64 rows per SM).
    const bool long_rows = !few_rows && row_bytes >= 4 * KR_MIN_CHUNK_BYTES;
    if (few_rows && row_bytes >= 2 * KR_MIN_CHUNK_BYTES) {
        // CTA chunks for about 8 CTAs per SM, twice and half that size, and so on. (Chunk counts sized to whole waves
        // of the partial kernel won 27 of 1048 cells and lost 2% on average.)
        const uint64_t chunks0 = std::clamp<uint64_t>(ceilDiv(sms * 8, rows), 2, KR_MAX_CHUNKS);
        const uint64_t c0 = std::max<uint64_t>(KR_MIN_CHUNK_BYTES, pow2Ceil(ceilDiv(row_bytes, chunks0)));
        add_split(c0);
        for (uint64_t step = 2; step <= 8; step *= 2) {
            add_split(c0 * step);
            if (c0 / step >= KR_MIN_CHUNK_BYTES) add_split(c0 / step);
        }
    } else if (long_rows) {
        // Many long rows: chunks of 8, 4, 16 and 32 KiB (the winners there), many waves, so no wave sizing.
        for (uint64_t chunk : {8192u, 4096u, 16384u, 32768u}) add_split(chunk);
    }
    // Block first for rows of 2-64 KiB and at least one row per SM. In a sweep, Block there beat every other shape in
    // 81-100% of cells by 2-24% (2.4x where Group had a few hundred rows and so only ~74 CTAs); below 2 KiB Group won,
    // from 64 KiB Split won, and with fewer rows than SMs Block left SMs idle.
    const bool block_first = row_bytes >= 2 * KR_MIN_BLOCK_ROW_BYTES &&
                             row_bytes < 64 * KR_MIN_BLOCK_ROW_BYTES && rows >= sms;
    if (!split.empty()) {
        // Group shapes with less than one warp per SM cannot compete with a split row (and cost tuning time).
        std::erase_if(group, [&](const KRSegmentShape& shape) { return rows * shape.group_lanes < sms * 32; });
    }

    std::vector<KRSegmentShape> result;
    if (block_first) appendUnique(result, {rows, row_bytes, row_bytes, KR_THREADS, 1, false});
    if (!split.empty() && few_rows) {
        result.insert(result.end(), split.begin(), split.end());
        result.insert(result.end(), group.begin(), group.end());
    } else if (!split.empty()) {
        // Many long rows: the Group guess, two Split shapes, the next Group width, then the rest alternating, so a
        // shallow tuning depth times both families.
        size_t g = 0, k = 0;
        const auto take_group = [&] { if (g < group.size()) result.push_back(group[g++]); };
        const auto take_split = [&] { if (k < split.size()) result.push_back(split[k++]); };
        take_group();
        take_split();
        take_split();
        while (g < group.size() || k < split.size()) {
            take_group();
            take_split();
        }
    } else if (!pack.empty()) {
        // Best pack guess, best group guess, then the rest of each.
        result.push_back(pack.front());
        if (!group.empty()) result.push_back(group.front());
        result.insert(result.end(), pack.begin() + 1, pack.end());
        if (group.size() > 1) result.insert(result.end(), group.begin() + 1, group.end());
    } else {
        result.insert(result.end(), group.begin(), group.end());
    }
    return result;
}

std::vector<KRPoolEntry> fp8KRCandidatePool(uint64_t rows, uint64_t row_bytes, uint32_t multiprocessors) {
    std::vector<KRPoolEntry> pool;
    const std::vector<KRSegmentShape> listed = fp8KRCandidateShapes(rows, row_bytes, multiprocessors);
    for (size_t i = 0; i < listed.size(); ++i) pool.push_back({listed[i], static_cast<uint32_t>(i + 1)});
    if (rows == 0 || row_bytes == 0) return pool;
    const uint64_t sms = std::max<uint32_t>(multiprocessors, 1);
    const uint64_t bytes = rows * row_bytes;
    const auto add = [&](const KRSegmentShape& shape) {
        if (!fp8KRSegmentShapeLegal(shape)) return;
        for (const KRPoolEntry& entry : pool)
            if (entry.shape == shape) return;
        if (ceilDiv(bytes, shapeThreads(shape)) > KR_POOL_MAX_THREAD_BYTES) return;
        pool.push_back({shape, 0});
    };
    if (fp8KRVectorRowBytes(row_bytes)) add({rows, row_bytes, row_bytes, 1, krVectorRows(row_bytes), true});
    for (uint32_t p : {2u, 3u, 4u, 6u, 8u, 12u, 16u}) add({rows, row_bytes, row_bytes, 1, p, false});
    for (uint32_t g = 1; g <= 32; g *= 2) add({rows, row_bytes, row_bytes, g, 1, false});
    add({rows, row_bytes, row_bytes, KR_THREADS, 1, false});
    if (rows < 64 * sms || row_bytes >= 4 * KR_MIN_CHUNK_BYTES) {
        for (uint64_t chunk = KR_MIN_CHUNK_BYTES; chunk < row_bytes; chunk *= 2)
            add({rows, row_bytes, chunk, KR_THREADS, 1, false});
    }
    return pool;
}

std::array<float, KR_RANKER_FEATURES> krRankerFeatures(const KRPoolEntry& entry,
                                                             const KRRankerSite& site,
                                                             uint32_t multiprocessors) {
    const KRSegmentShape& shape = entry.shape;
    if (!retainedKRShapeLegalForAnyElement(shape)) throw std::invalid_argument("Illegal retained FP8 KR segment shape");
    const double sms = std::max<uint32_t>(multiprocessors, 1);
    const double rows = static_cast<double>(shape.rows);
    const double row_bytes = static_cast<double>(shape.row_bytes);
    const double bytes = rows * row_bytes;
    const KRFamily family = shape.family();
    const double family_code = family == KRFamily::Pack ? (shape.vector ? 1.0 : 0.0)
                               : family == KRFamily::Group ? 2.0
                               : family == KRFamily::Block ? 3.0
                                                              : 4.0;
    const double lanes = family == KRFamily::Pack ? 1.0 : static_cast<double>(shape.group_lanes);
    const double chunk = family == KRFamily::Split ? static_cast<double>(shape.chunk_bytes) : row_bytes;
    const double threads = static_cast<double>(shapeThreads(shape));
    const double blocks = static_cast<double>(shapeBlocks(shape));
    const double lane_bytes = family == KRFamily::Pack ? shape.rows_per_thread * row_bytes : chunk / lanes;
    const double output_bytes = rows * (site.kind == 2 || site.kind == 4 ? 8.0 : 4.0);  // kind 4 too
    const std::array<double, KR_RANKER_FEATURES> f{
        log2Of(rows),
        log2Of(row_bytes),
        log2Of(bytes),
        family_code,
        static_cast<double>(shape.rows_per_thread),
        log2Of(lanes),
        log2Of(chunk),
        log2Of(static_cast<double>(shape.chunks())),
        log2Of(blocks),
        log2Of(blocks / sms),
        log2Of(threads),
        log2Of(bytes / threads),
        log2Of(bytes / blocks),
        log2Of(lane_bytes / 4.0),
        static_cast<double>(shape.row_bytes % 4),
        static_cast<double>(shape.row_bytes % 16),
        log2Of(rows / sms),
        static_cast<double>(site.kind),
        static_cast<double>(site.combine),
        static_cast<double>(site.transform),
        site.e5m2 ? 1.0 : 0.0,
        log2Of(output_bytes),
        static_cast<double>(entry.list_position),
        log2Of(sms)};
    std::array<float, KR_RANKER_FEATURES> out{};
    for (size_t i = 0; i < out.size(); ++i) out[i] = static_cast<float>(f[i]);
    return out;
}

bool fp8KRRankerTrained() { return Fp8::Fp8KRRankerModel::TRAINED; }

std::vector<KRSegmentShape> fp8KRRankedShapes(uint64_t rows, uint64_t row_bytes, uint32_t multiprocessors,
                                              const KRRankerSite& site, size_t count) {
    std::vector<KRSegmentShape> result;
    if (!fp8KRRankerTrained()) {
        result = fp8KRCandidateShapes(rows, row_bytes, multiprocessors);
        if (result.size() > count) result.resize(count);
        return result;
    }
    static_assert(Fp8::Fp8KRRankerModel::FEATURE_COUNT == KR_RANKER_FEATURES);
    const std::vector<KRPoolEntry> pool = fp8KRCandidatePool(rows, row_bytes, multiprocessors);
    std::vector<std::pair<float, size_t>> scored;
    scored.reserve(pool.size());
    for (size_t i = 0; i < pool.size(); ++i)
        scored.emplace_back(Fp8::Fp8KRRankerModel::margin(krRankerFeatures(pool[i], site, multiprocessors)), i);
    std::stable_sort(scored.begin(), scored.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });
    for (size_t i = 0; i < scored.size() && result.size() < count; ++i) result.push_back(pool[scored[i].second].shape);
    return result;
}

bool retainedKRShapeLegalForElement(const KRSegmentShape& shape, uint32_t element_bytes) {
    if (element_bytes != 1 && element_bytes != 2 && element_bytes != 4) return false;
    if (element_bytes == 1) return fp8KRSegmentShapeLegal(shape);
    if (shape.row_bytes % element_bytes != 0) return false;
    if (!shape.vector) return fp8KRSegmentShapeLegal(shape);
    if (element_bytes != 2) return false;
    // The 2-byte vector Pack lengths; otherwise the generic Pack shape's rules (the kernel it falls back to).
    KRSegmentShape generic = shape;
    generic.vector = false;
    return f16KRVectorRowBytes(shape.row_bytes) && shape.rows_per_thread == krVectorRows(shape.row_bytes) &&
           fp8KRSegmentShapeLegal(generic);
}

bool retainedKRShapeLegalForAnyElement(const KRSegmentShape& shape) {
    return retainedKRShapeLegalForElement(shape, 1) || retainedKRShapeLegalForElement(shape, 2) ||
           retainedKRShapeLegalForElement(shape, 4);
}

bool fp32KRArgShapeLegal(const KRSegmentShape& shape) {
    return retainedKRShapeLegalForElement(shape, 4) &&
           (shape.family() != KRFamily::Split || shape.chunk_bytes <= FP32_KR_ARG_MAX_CHUNK_BYTES);
}

bool f16KRArgShapeLegal(const KRSegmentShape& shape) {
    return retainedKRShapeLegalForElement(shape, 2) && !shape.vector && shape.row_bytes / 2 - 1 < (uint64_t{1} << 47);
}

namespace {
void requireWholeElements(uint64_t row_bytes, uint32_t element_bytes) {
    if ((element_bytes != 1 && element_bytes != 2 && element_bytes != 4) || row_bytes % element_bytes != 0)
        throw std::invalid_argument("Retained KR rows must be whole elements of 1, 2 or 4 bytes");
}
}  // namespace

std::vector<KRSegmentShape> retainedKRCandidateShapes(uint64_t rows, uint64_t row_bytes, uint32_t multiprocessors,
                                                      uint32_t element_bytes) {
    requireWholeElements(row_bytes, element_bytes);
    std::vector<KRSegmentShape> shapes = fp8KRCandidateShapes(rows, row_bytes, multiprocessors);
    std::erase_if(shapes, [&](const KRSegmentShape& shape) {
        return !retainedKRShapeLegalForElement(shape, element_bytes) || (element_bytes == 2 && shape.vector);
    });
    if (element_bytes == 2 && f16KRVectorRowBytes(row_bytes) && rows >= krVectorRows(row_bytes)) {
        // The 2-byte vector shape second, after FP8's first guess (its rows are at most 16 bytes, so the list is not
        // empty). Listed first, it lost to generic Pack at 10- and 14-byte rows (median 0.94x and 0.86x) and raised the
        // list's worst first pick from 1.22x to 1.52x; second, the first pick is unchanged.
        shapes.insert(shapes.begin() + 1, {rows, row_bytes, row_bytes, 1, krVectorRows(row_bytes), true});
    }
    if (element_bytes == 4 && !shapes.empty()) {
        // FP32 rows of at least 64 KiB with at least one row per SM list Block second. In a sweep, 512 rows of 64 KiB
        // ran Block at 1.57 TB/s against the list's best 1.47 (Split), 512 rows of 256 KiB 1.65 against 1.64; FP8's
        // list leads with Block only below 64 KiB and lists none above.
        const KRSegmentShape block{rows, row_bytes, row_bytes, KR_THREADS, 1, false};
        if (rows >= std::max<uint32_t>(multiprocessors, 1) && row_bytes >= FP32_KR_BLOCK_SECOND_ROW_BYTES &&
            retainedKRShapeLegalForElement(block, 4) && std::find(shapes.begin(), shapes.end(), block) == shapes.end())
            shapes.insert(shapes.begin() + 1, block);
    }
    return shapes;
}

std::vector<KRPoolEntry> retainedKRCandidatePool(uint64_t rows, uint64_t row_bytes, uint32_t multiprocessors,
                                                 uint32_t element_bytes) {
    // The element's list (positions 1..n), then FP8's other pool shapes legal for the element, in FP8's pool order.
    const std::vector<KRSegmentShape> listed =
        retainedKRCandidateShapes(rows, row_bytes, multiprocessors, element_bytes);
    std::vector<KRPoolEntry> pool;
    for (size_t i = 0; i < listed.size(); ++i) pool.push_back({listed[i], static_cast<uint32_t>(i + 1)});
    if (element_bytes == 2 && f16KRVectorRowBytes(row_bytes)) {
        // The 2-byte vector shape at any row count (FP8's pool rule), first among the extra shapes.
        const KRSegmentShape vector{rows, row_bytes, row_bytes, 1, krVectorRows(row_bytes), true};
        if (std::find(listed.begin(), listed.end(), vector) == listed.end()) pool.push_back({vector, 0});
    }
    for (const KRPoolEntry& entry : fp8KRCandidatePool(rows, row_bytes, multiprocessors)) {
        if (!retainedKRShapeLegalForElement(entry.shape, element_bytes) || (element_bytes == 2 && entry.shape.vector))
            continue;
        if (std::find(listed.begin(), listed.end(), entry.shape) != listed.end()) continue;
        pool.push_back({entry.shape, 0});
    }
    return pool;
}

}  // namespace ThorImplementation::ReductionInternal::KR
