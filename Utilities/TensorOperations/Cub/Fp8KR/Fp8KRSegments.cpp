#include "Utilities/TensorOperations/Cub/Fp8KR/Fp8KRSegments.h"

#include "Utilities/TensorOperations/Cub/Fp8KR/Fp8KRRankerModel.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ThorImplementation::CubReductionInternal::Fp8KR {
namespace {

uint64_t ceilDiv(uint64_t a, uint64_t b) { return a / b + (a % b != 0); }

uint64_t pow2Ceil(uint64_t x) {
    uint64_t p = 1;
    while (p < x) p <<= 1;
    return p;
}

bool validGroupLanes(uint32_t g) { return g >= 1 && g <= 32 && (g & (g - 1)) == 0; }

void appendUnique(std::vector<Fp8KRSegmentShape>& shapes, const Fp8KRSegmentShape& shape) {
    if (!fp8KRSegmentShapeLegal(shape)) return;
    if (std::find(shapes.begin(), shapes.end(), shape) == shapes.end()) shapes.push_back(shape);
}

uint64_t align16(uint64_t bytes) { return (bytes + 15) & ~uint64_t{15}; }

// Threads and CTAs of a shape's main kernel, as the launch geometry launches it.
uint64_t shapeThreads(const Fp8KRSegmentShape& shape) {
    switch (shape.family()) {
        case Fp8KRFamily::Pack: return ceilDiv(shape.rows, shape.rows_per_thread);
        case Fp8KRFamily::Group: return shape.rows * shape.group_lanes;
        case Fp8KRFamily::Split: return shape.segments() * FP8_KR_THREADS;
        case Fp8KRFamily::Block: return shape.rows * FP8_KR_THREADS;
    }
    return 1;
}

uint64_t shapeBlocks(const Fp8KRSegmentShape& shape) {
    switch (shape.family()) {
        case Fp8KRFamily::Pack: return ceilDiv(ceilDiv(shape.rows, shape.rows_per_thread), FP8_KR_THREADS);
        case Fp8KRFamily::Group: return ceilDiv(shape.rows, FP8_KR_THREADS / shape.group_lanes);
        case Fp8KRFamily::Split: return shape.segments();
        case Fp8KRFamily::Block: return shape.rows;
    }
    return 1;
}

double log2Of(double x) { return std::log2(std::max(x, 1e-30)); }

}  // namespace

bool fp8KRSegmentShapeLegal(const Fp8KRSegmentShape& shape) {
    if (shape.rows == 0 || shape.row_bytes == 0 || shape.chunk_bytes == 0 || shape.rows_per_thread == 0) return false;
    // Byte offsets (row * row_bytes + head) stay far from 64-bit overflow.
    constexpr uint64_t max_bytes = uint64_t{1} << 62;
    if (shape.row_bytes > max_bytes / shape.rows) return false;
    switch (shape.family()) {
        case Fp8KRFamily::Pack:
            if (shape.vector &&
                (!fp8KRVectorRowBytes(shape.row_bytes) || shape.rows_per_thread != fp8KRVectorRows(shape.row_bytes)))
                return false;
            return shape.group_lanes == 1 && shape.rows_per_thread <= 16 &&
                   shape.row_bytes <= FP8_KR_MAX_PACK_ROW_BYTES &&
                   shape.rows_per_thread * shape.row_bytes <= FP8_KR_MAX_PACK_SPAN_BYTES;
        case Fp8KRFamily::Group:
            return !shape.vector && validGroupLanes(shape.group_lanes);
        case Fp8KRFamily::Block:
            return !shape.vector && shape.row_bytes >= FP8_KR_MIN_BLOCK_ROW_BYTES &&
                   shape.rows <= static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
        case Fp8KRFamily::Split:
            if (shape.vector) return false;
            return shape.group_lanes == FP8_KR_THREADS && shape.chunk_bytes % 16 == 0 &&
                   shape.chunk_bytes >= FP8_KR_MIN_CHUNK_BYTES && shape.chunks() <= FP8_KR_MAX_CHUNKS &&
                   shape.segments() <= static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
    }
    return false;
}

uint64_t fp8KRScratchBytes(const Fp8KRSegmentShape& shape, bool arg) {
    if (!fp8KRSegmentShapeLegal(shape)) throw std::invalid_argument("Illegal retained FP8 KR segment shape");
    if (shape.family() != Fp8KRFamily::Split) return 0;
    return align16(shape.segments() * (arg ? sizeof(uint64_t) : sizeof(float)));
}

std::string fp8KRShapeName(const Fp8KRSegmentShape& shape) {
    switch (shape.family()) {
        case Fp8KRFamily::Pack: return (shape.vector ? "v" : "p") + std::to_string(shape.rows_per_thread);
        case Fp8KRFamily::Group: return "g" + std::to_string(shape.group_lanes);
        case Fp8KRFamily::Split: return "s" + std::to_string(shape.chunk_bytes);
        case Fp8KRFamily::Block: return "b";
    }
    return "unknown";
}

std::vector<Fp8KRSegmentShape> fp8KRCandidateShapes(uint64_t rows, uint64_t row_bytes, uint32_t multiprocessors) {
    if (rows == 0 || row_bytes == 0) return {};
    const uint64_t sms = std::max<uint32_t>(multiprocessors, 1);
    std::vector<Fp8KRSegmentShape> pack, group, split;

    // Pack: short rows; the vector shape where compiled, then about 16 and 8 bytes per thread.
    if (fp8KRVectorRowBytes(row_bytes) && rows >= fp8KRVectorRows(row_bytes)) {
        appendUnique(pack, {rows, row_bytes, row_bytes, 1, fp8KRVectorRows(row_bytes), true});
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
        if (chunk < row_bytes) appendUnique(split, {rows, row_bytes, chunk, FP8_KR_THREADS, 1, false});
    };
    const bool few_rows = rows < sms * 16;
    // 1154: rows of 16 KiB or more also split when rows alone fill the GPU (the 1153 sweep: ARG Group on such rows,
    // with few CTAs per SM, lost 5-15% to Split at 3k-7k rows, and Split was not timed past 64 rows per SM).
    const bool long_rows = !few_rows && row_bytes >= 4 * FP8_KR_MIN_CHUNK_BYTES;
    if (few_rows && row_bytes >= 2 * FP8_KR_MIN_CHUNK_BYTES) {
        // CTA chunks for about 8 CTAs per SM, twice and half that size, and so on. (1154 led with chunk counts sized to
        // whole waves of the partial kernel; the 1154 sweep had them win 27 of 1048 cells and lose 2% on average.)
        const uint64_t chunks0 = std::clamp<uint64_t>(ceilDiv(sms * 8, rows), 2, FP8_KR_MAX_CHUNKS);
        const uint64_t c0 = std::max<uint64_t>(FP8_KR_MIN_CHUNK_BYTES, pow2Ceil(ceilDiv(row_bytes, chunks0)));
        add_split(c0);
        for (uint64_t step = 2; step <= 8; step *= 2) {
            add_split(c0 * step);
            if (c0 / step >= FP8_KR_MIN_CHUNK_BYTES) add_split(c0 / step);
        }
    } else if (long_rows) {
        // Many long rows: chunks of 8, 4, 16 and 32 KiB (the 1153 winners there), many waves, so no wave sizing.
        for (uint64_t chunk : {8192u, 4096u, 16384u, 32768u}) add_split(chunk);
    }
    // 1157: Block first for rows of 2-64 KiB and at least one row per SM. The 1156 sweep: there Block beat every other
    // shape in 81-100% of cells by 2-24% (2.4x where Group had a few hundred rows and so only ~74 CTAs); below 2 KiB
    // Group won, from 64 KiB Split won, and with fewer rows than SMs Block left SMs idle.
    const bool block_first = row_bytes >= 2 * FP8_KR_MIN_BLOCK_ROW_BYTES &&
                             row_bytes < 64 * FP8_KR_MIN_BLOCK_ROW_BYTES && rows >= sms;
    if (!split.empty()) {
        // Group shapes with less than one warp per SM cannot compete with a split row (and cost tuning time).
        std::erase_if(group, [&](const Fp8KRSegmentShape& shape) { return rows * shape.group_lanes < sms * 32; });
    }

    std::vector<Fp8KRSegmentShape> result;
    if (block_first) appendUnique(result, {rows, row_bytes, row_bytes, FP8_KR_THREADS, 1, false});
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

std::vector<Fp8KRPoolEntry> fp8KRCandidatePool(uint64_t rows, uint64_t row_bytes, uint32_t multiprocessors) {
    std::vector<Fp8KRPoolEntry> pool;
    const std::vector<Fp8KRSegmentShape> listed = fp8KRCandidateShapes(rows, row_bytes, multiprocessors);
    for (size_t i = 0; i < listed.size(); ++i) pool.push_back({listed[i], static_cast<uint32_t>(i + 1)});
    if (rows == 0 || row_bytes == 0) return pool;
    const uint64_t sms = std::max<uint32_t>(multiprocessors, 1);
    const uint64_t bytes = rows * row_bytes;
    const auto add = [&](const Fp8KRSegmentShape& shape) {
        if (!fp8KRSegmentShapeLegal(shape)) return;
        for (const Fp8KRPoolEntry& entry : pool)
            if (entry.shape == shape) return;
        if (ceilDiv(bytes, shapeThreads(shape)) > FP8_KR_POOL_MAX_THREAD_BYTES) return;
        pool.push_back({shape, 0});
    };
    if (fp8KRVectorRowBytes(row_bytes)) add({rows, row_bytes, row_bytes, 1, fp8KRVectorRows(row_bytes), true});
    for (uint32_t p : {2u, 3u, 4u, 6u, 8u, 12u, 16u}) add({rows, row_bytes, row_bytes, 1, p, false});
    for (uint32_t g = 1; g <= 32; g *= 2) add({rows, row_bytes, row_bytes, g, 1, false});
    add({rows, row_bytes, row_bytes, FP8_KR_THREADS, 1, false});
    if (rows < 64 * sms || row_bytes >= 4 * FP8_KR_MIN_CHUNK_BYTES) {
        for (uint64_t chunk = FP8_KR_MIN_CHUNK_BYTES; chunk < row_bytes; chunk *= 2)
            add({rows, row_bytes, chunk, FP8_KR_THREADS, 1, false});
    }
    return pool;
}

std::array<float, FP8_KR_RANKER_FEATURES> fp8KRRankerFeatures(const Fp8KRPoolEntry& entry,
                                                             const Fp8KRRankerSite& site,
                                                             uint32_t multiprocessors) {
    const Fp8KRSegmentShape& shape = entry.shape;
    if (!fp8KRSegmentShapeLegal(shape)) throw std::invalid_argument("Illegal retained FP8 KR segment shape");
    const double sms = std::max<uint32_t>(multiprocessors, 1);
    const double rows = static_cast<double>(shape.rows);
    const double row_bytes = static_cast<double>(shape.row_bytes);
    const double bytes = rows * row_bytes;
    const Fp8KRFamily family = shape.family();
    const double family_code = family == Fp8KRFamily::Pack ? (shape.vector ? 1.0 : 0.0)
                               : family == Fp8KRFamily::Group ? 2.0
                               : family == Fp8KRFamily::Block ? 3.0
                                                              : 4.0;
    const double lanes = family == Fp8KRFamily::Pack ? 1.0 : static_cast<double>(shape.group_lanes);
    const double chunk = family == Fp8KRFamily::Split ? static_cast<double>(shape.chunk_bytes) : row_bytes;
    const double threads = static_cast<double>(shapeThreads(shape));
    const double blocks = static_cast<double>(shapeBlocks(shape));
    const double lane_bytes = family == Fp8KRFamily::Pack ? shape.rows_per_thread * row_bytes : chunk / lanes;
    const double output_bytes = rows * (site.kind == 2 || site.kind == 4 ? 8.0 : 4.0);  // 1215: kind 4 too
    const std::array<double, FP8_KR_RANKER_FEATURES> f{
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
    std::array<float, FP8_KR_RANKER_FEATURES> out{};
    for (size_t i = 0; i < out.size(); ++i) out[i] = static_cast<float>(f[i]);
    return out;
}

bool fp8KRRankerTrained() { return Fp8KRRankerModel::TRAINED; }

std::vector<Fp8KRSegmentShape> fp8KRRankedShapes(uint64_t rows, uint64_t row_bytes, uint32_t multiprocessors,
                                                 const Fp8KRRankerSite& site, size_t count) {
    std::vector<Fp8KRSegmentShape> result;
    if (!fp8KRRankerTrained()) {
        result = fp8KRCandidateShapes(rows, row_bytes, multiprocessors);
        if (result.size() > count) result.resize(count);
        return result;
    }
    static_assert(Fp8KRRankerModel::FEATURE_COUNT == FP8_KR_RANKER_FEATURES);
    const std::vector<Fp8KRPoolEntry> pool = fp8KRCandidatePool(rows, row_bytes, multiprocessors);
    std::vector<std::pair<float, size_t>> scored;
    scored.reserve(pool.size());
    for (size_t i = 0; i < pool.size(); ++i)
        scored.emplace_back(Fp8KRRankerModel::margin(fp8KRRankerFeatures(pool[i], site, multiprocessors)), i);
    std::stable_sort(scored.begin(), scored.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });
    for (size_t i = 0; i < scored.size() && result.size() < count; ++i) result.push_back(pool[scored[i].second].shape);
    return result;
}

bool retainedKRShapeLegalForElement(const Fp8KRSegmentShape& shape, uint32_t element_bytes) {
    if (element_bytes != 1 && element_bytes != 4) return false;
    if (!fp8KRSegmentShapeLegal(shape) || shape.row_bytes % element_bytes != 0) return false;
    return element_bytes == 1 || !shape.vector;
}

bool fp32KRArgShapeLegal(const Fp8KRSegmentShape& shape) {
    return retainedKRShapeLegalForElement(shape, 4) &&
           (shape.family() != Fp8KRFamily::Split || shape.chunk_bytes <= FP32_KR_ARG_MAX_CHUNK_BYTES);
}

namespace {
void requireWholeElements(uint64_t row_bytes, uint32_t element_bytes) {
    if ((element_bytes != 1 && element_bytes != 4) || row_bytes % element_bytes != 0)
        throw std::invalid_argument("Retained KR rows must be whole elements of 1 or 4 bytes");
}
}  // namespace

std::vector<Fp8KRSegmentShape> retainedKRCandidateShapes(uint64_t rows, uint64_t row_bytes, uint32_t multiprocessors,
                                                         uint32_t element_bytes) {
    requireWholeElements(row_bytes, element_bytes);
    std::vector<Fp8KRSegmentShape> shapes = fp8KRCandidateShapes(rows, row_bytes, multiprocessors);
    std::erase_if(shapes, [&](const Fp8KRSegmentShape& shape) {
        return !retainedKRShapeLegalForElement(shape, element_bytes);
    });
    if (element_bytes == 4 && !shapes.empty()) {
        // 1180: FP32 rows of at least 64 KiB with at least one row per SM list Block second. The 1179 sweep: 512 rows
        // of 64 KiB ran Block at 1.57 TB/s against the list's best 1.47 (Split), 512 rows of 256 KiB 1.65 against
        // 1.64; FP8's list leads with Block only below 64 KiB and lists none above.
        const Fp8KRSegmentShape block{rows, row_bytes, row_bytes, FP8_KR_THREADS, 1, false};
        if (rows >= std::max<uint32_t>(multiprocessors, 1) && row_bytes >= FP32_KR_BLOCK_SECOND_ROW_BYTES &&
            retainedKRShapeLegalForElement(block, 4) && std::find(shapes.begin(), shapes.end(), block) == shapes.end())
            shapes.insert(shapes.begin() + 1, block);
    }
    return shapes;
}

std::vector<Fp8KRPoolEntry> retainedKRCandidatePool(uint64_t rows, uint64_t row_bytes, uint32_t multiprocessors,
                                                    uint32_t element_bytes) {
    // The element's list (positions 1..n), then FP8's other pool shapes legal for the element, in FP8's pool order.
    const std::vector<Fp8KRSegmentShape> listed =
        retainedKRCandidateShapes(rows, row_bytes, multiprocessors, element_bytes);
    std::vector<Fp8KRPoolEntry> pool;
    for (size_t i = 0; i < listed.size(); ++i) pool.push_back({listed[i], static_cast<uint32_t>(i + 1)});
    for (const Fp8KRPoolEntry& entry : fp8KRCandidatePool(rows, row_bytes, multiprocessors)) {
        if (!retainedKRShapeLegalForElement(entry.shape, element_bytes)) continue;
        if (std::find(listed.begin(), listed.end(), entry.shape) != listed.end()) continue;
        pool.push_back({entry.shape, 0});
    }
    return pool;
}

}  // namespace ThorImplementation::CubReductionInternal::Fp8KR
