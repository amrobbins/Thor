// thor_dense_copy_audit. What Thor's dense copy of a strided span (copyToDenseAsync / launchCopyToDense,
// Utilities/TensorOperations/Copy/StridedCopy.h) achieves per layout class, against the copy curve and against
// ref_1387, the copy copyToDenseAsync made before the copy planner.
//
//   d  dense: [n] (one memcpy, the sanity case).
//   r  a rotation: [b, a] of a dense [a, b] (an F view's reduced output; a 2D transpose).
//   g  a payload rotation: [b, a, p] of a dense [a, b, p] (G; p from 2 to 33 elements).
//   h  an attention head permute: [B, S, H, D] of a dense [B, H, S, D] (D from 8 to 128).
//   k  a pitched packing: [O, R, I] rows of I contiguous elements with row gaps, slab gaps or both (H).
//   s  a rank-1 gather: [n] at stride 2, 3, 4, 8 or 33.
//   n  a general permutation of a dense rank-4 or rank-5 tensor (every dim >= 2).
//
// Each cell is timed cold (ColdTiming.h: ring views of the input and output spanning at least 2 GiB and 1 GiB,
// launches per sample about --interval-us, --samples samples interleaved across a batch of cells) with:
//   - the plan made once for the first views, launched on every view (launchCopyToDense);
//   - ref_1387 on the same views: one memcpy for a dense span, cudaMemcpy2DAsync for one pitch of dense rows, else the
//     gather kernel of one element per thread with a 64-bit division and modulo per axis (reproduced below);
//   - the copy curve at the cell's moved bytes: the source bytes the memory system reads (the distinct 32-byte sectors
//     for k and s cells, every byte otherwise) plus the destination's bytes.
// The first run of the plan is checked against ref_1387's gather on the same input (a device byte compare).
//
// --probe times, for each cell of --probe-min-mib useful bytes or more, the plan and variants of it on the cell's
// views, one CSV row per variant (its own header; not for dense_copy_audit.py), each checked against ref_1387's bytes:
//   - grid_d2, grid_d4, grid_d16, grid_d64: the plan's kernel on a half, a quarter, a 16th and a 64th of its grid (one
//     CTA per unit of work), so that each CTA runs 2 to 64 units;
//   - for a Transpose or MatrixTranspose plan, the transpose kernel on other tiles, each taken along x (_xfirst) and
//     along v (_vfirst, DenseCopyPlan::tile_v_first): `planned` (the plan's tile), `square` (128-byte tile rows, at
//     least 32 vectors a side), `tall32` and `tall128` (tile rows of 32 and 128 bytes and as many v rows as 32 KB of
//     shared memory holds) and `wide` (four times square's x, a quarter of its v);
//   - plan_repeat: the plan again, last. The first variant timed on a cell's views (the plan) was 2% to 6% slower on
//     some cells than every later variant of the same kernel and grid class; this pair measures that, since the
//     standard run also times the plan first and ref_1387 after it.
//
// Each timed item (the plan, ref_1387, a probe variant) runs once untimed at the start of each of its samples, so every
// item starts after a launch of itself. Without it, the first item of a cell in each pass started after the previous
// cell's work, which it paid for (most likely that cell's dirty output in the L2), and the items after it started after
// the item before them: the plan timed first over the same plan timed last was p50 1.001 to 1.003, p90 1.010 to 1.013,
// up to 1.029 (FP32 4864180 words at stride 33, 19.5 MB out: 118.4 against 115.2 us).
//
// Output: '#' header lines (settings, device, copy curve) and one CSV row per cell; benchmarks/dense_copy_audit.py
// reports it.
#include "benchmarks/ColdTiming.h"
#include "benchmarks/Fp8OracleBatchQueue.h"
#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/TensorOperations/Copy/StridedCopy.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace T = ThorImplementation;
namespace B = ThorImplementation::ReductionBenchmarking;
using B::ColdTiming::CopyCurve;
using B::ColdTiming::GiB;
using B::ColdTiming::InputReuseBytes;
using B::ColdTiming::MaxViews;
using B::ColdTiming::MiB;
using B::ColdTiming::OutputReuseBytes;
using B::ColdTiming::Ring;
using B::ColdTiming::TimingEvent;

namespace {

constexpr uint64_t KiB = 1ULL << 10;
constexpr uint32_t Rank1387 = 32;

void check(cudaError_t status, const char* what) { B::ColdTiming::checkCuda(status, what); }

__global__ void fillHashKernel(uint32_t* words, uint64_t count, uint32_t seed) {
    const uint64_t step = uint64_t(gridDim.x) * blockDim.x;
    for (uint64_t i = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x; i < count; i += step) {
        uint32_t x = static_cast<uint32_t>(i) * 2654435761u ^ static_cast<uint32_t>(i >> 32) ^ seed;
        x ^= x >> 15;
        x *= 2246822519u;
        x ^= x >> 13;
        words[i] = x;
    }
}

__global__ void countMismatchKernel(const uint8_t* a, const uint8_t* b, uint64_t bytes, unsigned long long* count) {
    unsigned long long local = 0;
    for (uint64_t i = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x; i < bytes; i += uint64_t(gridDim.x) * blockDim.x)
        local += a[i] != b[i];
    if (local) atomicAdd(count, local);
}

// ref_1387's gather: copyToDenseAsync's before the copy planner (Utilities/TensorOperations/Copy/StridedCopy.cu).
struct SpanLayout1387 {
    uint32_t rank = 0;
    uint64_t dimensions[Rank1387]{};
    uint64_t strides[Rank1387]{};
};

template <typename Element>
__global__ void gather1387Kernel(const Element* source, Element* destination, uint64_t numElements,
                                 SpanLayout1387 layout) {
    const uint64_t linearIndex = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linearIndex >= numElements) return;
    uint64_t remaining = linearIndex;
    uint64_t sourceIndex = 0;
    for (uint32_t reverseDimension = 0; reverseDimension < layout.rank; ++reverseDimension) {
        const uint32_t dimension = layout.rank - 1 - reverseDimension;
        const uint64_t coordinate = remaining % layout.dimensions[dimension];
        remaining /= layout.dimensions[dimension];
        sourceIndex += coordinate * layout.strides[dimension];
    }
    destination[linearIndex] = source[sourceIndex];
}

// The rows of ref_1387's cudaMemcpy2DAsync path (tryCopyPitched): a dense suffix and a prefix of one pitch.
struct Pitched1387 {
    uint64_t rows = 0, row_bytes = 0, pitch_bytes = 0;
};

std::optional<Pitched1387> pitched1387(const std::vector<uint64_t>& dims, const std::vector<uint64_t>& strides,
                                       uint64_t e) {
    size_t suffix = dims.size();
    uint64_t expected = 1;
    for (size_t i = dims.size(); i > 0; --i) {
        if (strides[i - 1] != expected) break;
        suffix = i - 1;
        expected *= dims[i - 1];
    }
    if (suffix == 0 || suffix == dims.size()) return std::nullopt;
    for (size_t i = 0; i + 1 < suffix; ++i)
        if (strides[i] != dims[i + 1] * strides[i + 1]) return std::nullopt;
    Pitched1387 p;
    p.row_bytes = e;
    for (size_t i = suffix; i < dims.size(); ++i) p.row_bytes *= dims[i];
    p.rows = 1;
    for (size_t i = 0; i < suffix; ++i) p.rows *= dims[i];
    p.pitch_bytes = strides[suffix - 1] * e;
    if (p.pitch_bytes < p.row_bytes) return std::nullopt;
    return p;
}

struct DTypeSpec {
    const char* name;
    T::DataType dtype;
    uint64_t bytes;
};
constexpr DTypeSpec DTYPES[] = {
    {"fp32", T::DataType::FP32, 4},         {"fp16", T::DataType::FP16, 2},         {"bf16", T::DataType::BF16, 2},
    {"fp8_e4m3", T::DataType::FP8_E4M3, 1}, {"fp8_e5m2", T::DataType::FP8_E5M2, 1},
};

const DTypeSpec& dtypeSpec(std::string_view name) {
    for (const DTypeSpec& d : DTYPES)
        if (name == d.name) return d;
    throw std::invalid_argument("--dtype takes fp32, fp16, bf16, fp8_e4m3 or fp8_e5m2");
}

std::vector<std::string> splitList(std::string_view text) {
    std::vector<std::string> out;
    std::stringstream stream{std::string(text)};
    for (std::string item; std::getline(stream, item, ',');)
        if (!item.empty()) out.push_back(item);
    return out;
}

constexpr std::string_view CASES = "drghksn";

struct Settings {
    const DTypeSpec* dtype = &DTYPES[0];
    std::string cases{CASES};
    uint64_t cells = 40;  // per case
    uint64_t seed = 1388;
    uint64_t min_bytes = 64 * KiB;
    uint64_t max_mib = 256;
    uint64_t max_span_bytes = 1 * GiB;
    uint64_t max_cells = UINT64_MAX;
    int samples = 5;
    double interval_us = 1000.0;
    uint64_t max_launches = 256;
    uint64_t arena_bytes = 8 * GiB;
    uint64_t output_ring_bytes = 2 * GiB;
    size_t queue_cells = 16;
    bool list_only = false;
    bool probe = false;
    uint64_t probe_min_bytes = 16 * MiB;
};

Settings parse(int argc, char** argv) {
    Settings s;
    for (int i = 1; i < argc; ++i) {
        const std::string_view a(argv[i]);
        const auto value = [&](std::string_view prefix) -> std::optional<std::string> {
            if (!a.starts_with(prefix)) return std::nullopt;
            return std::string(a.substr(prefix.size()));
        };
        if (auto v = value("--dtype=")) {
            s.dtype = &dtypeSpec(*v);
        } else if (auto v = value("--cases=")) {
            s.cases.clear();
            for (const std::string& c : splitList(*v)) {
                if (c.size() != 1 || CASES.find(c[0]) == std::string_view::npos)
                    throw std::invalid_argument("--cases takes d,r,g,h,k,s,n");
                s.cases.push_back(c[0]);
            }
        } else if (auto v = value("--cells=")) {
            s.cells = std::stoull(*v);
        } else if (auto v = value("--seed=")) {
            s.seed = std::stoull(*v);
        } else if (auto v = value("--min-kib=")) {
            s.min_bytes = std::stoull(*v) * KiB;
        } else if (auto v = value("--max-mib=")) {
            s.max_mib = std::stoull(*v);
        } else if (auto v = value("--max-span-mib=")) {
            s.max_span_bytes = std::stoull(*v) * MiB;
        } else if (auto v = value("--max-cells=")) {
            s.max_cells = std::stoull(*v);
        } else if (auto v = value("--samples=")) {
            s.samples = std::max(3, std::stoi(*v));
        } else if (auto v = value("--interval-us=")) {
            s.interval_us = std::stod(*v);
        } else if (auto v = value("--max-launches=")) {
            s.max_launches = std::max<uint64_t>(1, std::stoull(*v));
        } else if (auto v = value("--arena-gib=")) {
            s.arena_bytes = static_cast<uint64_t>(std::stod(*v) * double(GiB));
        } else if (auto v = value("--output-ring-gib=")) {
            s.output_ring_bytes = static_cast<uint64_t>(std::stod(*v) * double(GiB));
        } else if (auto v = value("--queue-cells=")) {
            s.queue_cells = std::max<size_t>(1, std::stoull(*v));
        } else if (a == "--list-only") {
            s.list_only = true;
        } else if (a == "--probe") {
            s.probe = true;
        } else if (auto v = value("--probe-min-mib=")) {
            s.probe_min_bytes = std::stoull(*v) * MiB;
        } else if (a == "--help") {
            std::cout << "thor_dense_copy_audit [--dtype=fp32] [--cases=d,r,g,h,k,s,n] [--cells=40] [--seed=1388]\n"
                         "    [--min-kib=64] [--max-mib=256] [--max-span-mib=1024] [--max-cells=N] [--samples=5]\n"
                         "    [--interval-us=1000] [--max-launches=256] [--arena-gib=8] [--output-ring-gib=2]\n"
                         "    [--queue-cells=16] [--list-only] [--probe] [--probe-min-mib=16]\n"
                         "--cells is per case; sizes are the destination's bytes, log-uniform in [min, max].\n"
                         "--probe: variants of each plan (grids, transpose tiles and tile orders) for the cells of\n"
                         "--probe-min-mib or more, one row per variant.\n"
                         "CSV on stdout, progress on stderr.\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("Unknown argument " + std::string(a));
        }
    }
    if (s.cases.empty()) throw std::invalid_argument("--cases must not be empty");
    if (s.min_bytes < 4 * KiB || s.min_bytes > s.max_mib * MiB)
        throw std::invalid_argument("need 4 KiB <= --min-kib <= --max-mib");
    if (s.arena_bytes < 4 * s.max_span_bytes || s.output_ring_bytes < 4 * s.max_mib * MiB)
        throw std::invalid_argument("the rings must hold four spans of --max-span-mib and four outputs of --max-mib");
    return s;
}

uint64_t mix(uint64_t x) {  // splitmix64 finalizer
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}
double unit(uint64_t h) { return static_cast<double>(h >> 11) * (1.0 / 9007199254740992.0); }
uint64_t logUniform(uint64_t h, uint64_t low, uint64_t high) {
    if (high <= low) return low;
    const double l = std::log(double(low)), u = std::log(double(high));
    return std::clamp<uint64_t>(static_cast<uint64_t>(std::llround(std::exp(l + (u - l) * unit(h)))), low, high);
}
template <typename V>
const typename V::value_type& pick(const V& values, uint64_t h) {
    return values[h % values.size()];
}

std::vector<uint64_t> denseStrides(const std::vector<uint64_t>& dims) {
    std::vector<uint64_t> strides(dims.size(), 1);
    for (size_t k = dims.size(); k-- > 1;) strides[k - 1] = strides[k] * dims[k];
    return strides;
}

uint64_t product(const std::vector<uint64_t>& v) {
    return std::accumulate(v.begin(), v.end(), uint64_t{1}, std::multiplies<uint64_t>());
}

uint64_t reachOf(const std::vector<uint64_t>& dims, const std::vector<uint64_t>& strides) {
    uint64_t r = 1;
    for (size_t k = 0; k < dims.size(); ++k) r += (dims[k] - 1) * strides[k];
    return r;
}

struct CellSpec {
    uint64_t id = 0;
    char kind = 'd';
    std::string sub;  // k: row_gap / slab_gap / both
    std::vector<uint64_t> dims, strides;
    uint64_t reach = 0;               // source elements the layout reaches
    uint64_t rot_a = 0, rot_b = 0;    // r: the dense [a, b] being transposed
};

// `count` dims of at least 2 whose product is near `target`.
std::vector<uint64_t> splitDims(uint64_t target, size_t count, uint64_t h) {
    std::vector<double> w(count);
    double total = 0;
    for (size_t i = 0; i < count; ++i) total += (w[i] = 0.3 + unit(mix(h + 31 * i)));
    std::vector<uint64_t> dims(count);
    const double log_target = std::log(double(std::max<uint64_t>(target, 2)));
    for (size_t i = 0; i < count; ++i)
        dims[i] = std::max<uint64_t>(2, static_cast<uint64_t>(std::llround(std::exp(log_target * w[i] / total))));
    return dims;
}

CellSpec makeCell(char kind, uint64_t index, uint64_t id, const Settings& settings) {
    const uint64_t e = settings.dtype->bytes;
    const uint64_t h = mix(settings.seed * 1000003 + uint64_t(kind) * 7919 + index);
    const uint64_t bytes = logUniform(mix(h + 1), settings.min_bytes, settings.max_mib * MiB);
    const uint64_t n = std::max<uint64_t>(64, bytes / e);
    CellSpec c;
    c.id = id;
    c.kind = kind;
    switch (kind) {
        case 'd':
            c.dims = {n};
            c.strides = {1};
            break;
        case 'r': {
            const uint64_t a = logUniform(mix(h + 2), 8, std::max<uint64_t>(8, n / 8));
            const uint64_t b = std::max<uint64_t>(8, n / a);
            c.rot_a = a;
            c.rot_b = b;
            c.dims = {b, a};
            c.strides = {1, b};
            break;
        }
        case 'g': {
            const std::vector<uint64_t> payloads = {2, 3, 4, 5, 7, 8, 12, 16, 24, 33};
            const uint64_t p = pick(payloads, mix(h + 3));
            const uint64_t m = std::max<uint64_t>(64, n / p);
            const uint64_t a = logUniform(mix(h + 2), 8, std::max<uint64_t>(8, m / 8));
            const uint64_t b = std::max<uint64_t>(8, m / a);
            c.dims = {b, a, p};
            c.strides = {p, b * p, 1};
            break;
        }
        case 'h': {
            const std::vector<uint64_t> head_dims = {8, 16, 32, 64, 128};
            const std::vector<uint64_t> heads = {4, 8, 12, 16, 32};
            const uint64_t d = pick(head_dims, mix(h + 3)), hh = pick(heads, mix(h + 4));
            const uint64_t m = std::max<uint64_t>(8, n / (d * hh));
            const uint64_t s = logUniform(mix(h + 2), 8, std::max<uint64_t>(8, m));
            const uint64_t batch = std::max<uint64_t>(1, m / s);
            c.dims = {batch, s, hh, d};
            c.strides = {hh * s * d, d, s * d, 1};
            break;
        }
        case 'k': {
            const std::vector<uint64_t> inners = {3, 4, 5, 6, 7, 8, 12, 16, 31, 33, 62, 64, 128, 500};
            const uint64_t inner = pick(inners, mix(h + 3));
            const uint64_t rows = std::max<uint64_t>(4, n / inner);
            const uint64_t r = logUniform(mix(h + 2), 2, std::max<uint64_t>(2, rows / 2));
            const uint64_t o = std::max<uint64_t>(1, rows / r);
            const int sub = static_cast<int>(mix(h + 5) % 3);
            const uint64_t row_gap = sub == 1 ? 0 : 1 + mix(h + 6) % (2 * inner);
            const uint64_t row_pitch = inner + row_gap;
            const uint64_t slab_gap = sub == 0 ? 0 : 1 + mix(h + 7) % (r * row_pitch);
            c.sub = sub == 0 ? "row_gap" : sub == 1 ? "slab_gap" : "both";
            c.dims = {o, r, inner};
            c.strides = {r * row_pitch + slab_gap, row_pitch, 1};
            break;
        }
        case 's': {
            const std::vector<uint64_t> strides = {2, 3, 4, 8, 33};
            const uint64_t s = pick(strides, mix(h + 3));
            c.sub = "stride_" + std::to_string(s);
            c.dims = {n};
            c.strides = {s};
            break;
        }
        case 'n': {
            const size_t rank = 4 + mix(h + 3) % 2;
            const std::vector<uint64_t> physical = splitDims(n, rank, mix(h + 4));
            const std::vector<uint64_t> physical_strides = denseStrides(physical);
            std::vector<size_t> order(rank);
            std::iota(order.begin(), order.end(), 0);
            for (size_t i = rank; i-- > 1;) std::swap(order[i], order[mix(h + 10 + i) % (i + 1)]);
            if (order.back() == rank - 1) std::swap(order[rank - 1], order[mix(h + 9) % (rank - 1)]);
            for (size_t k = 0; k < rank; ++k) {
                c.dims.push_back(physical[order[k]]);
                c.strides.push_back(physical_strides[order[k]]);
            }
            break;
        }
        default: throw std::invalid_argument("unknown case");
    }
    c.reach = reachOf(c.dims, c.strides);
    // Keep the span inside --max-span-mib: halve the outermost dim until it fits.
    while (c.reach * e > settings.max_span_bytes && c.dims[0] > 1) {
        c.dims[0] = std::max<uint64_t>(1, c.dims[0] / 2);
        if (c.kind == 'r') c.rot_b = c.dims[0];
        c.reach = reachOf(c.dims, c.strides);
    }
    if (c.kind == 'r') c.strides = {1, c.rot_b};
    c.reach = reachOf(c.dims, c.strides);
    return c;
}

std::vector<CellSpec> cells(const Settings& settings) {
    std::vector<CellSpec> out;
    uint64_t id = 0;
    for (char kind : settings.cases)
        for (uint64_t i = 0; i < settings.cells; ++i) out.push_back(makeCell(kind, i, id++, settings));
    return out;
}

// The source bytes the memory system reads: the distinct 32-byte sectors of k and s cells (their bases 256-byte
// aligned), every byte of the permutations.
double touchedSourceBytes(const CellSpec& c, uint64_t e) {
    const uint64_t useful = product(c.dims) * e;
    if (c.kind == 's') {
        const uint64_t pitch = c.strides[0] * e;
        if (pitch >= 32) return 32.0 * double(c.dims[0]);
        return 32.0 * double(B::ColdTiming::ceilDiv((c.dims[0] - 1) * pitch + e, 32));
    }
    if (c.kind != 'k') return double(useful);
    uint64_t sectors = 0, last = UINT64_MAX;
    const uint64_t row_bytes = c.dims[2] * e;
    for (uint64_t o = 0; o < c.dims[0]; ++o)
        for (uint64_t r = 0; r < c.dims[1]; ++r) {
            const uint64_t start = (o * c.strides[0] + r * c.strides[1]) * e;
            const uint64_t first = start / 32, end = (start + row_bytes - 1) / 32;
            sectors += end - first + 1 - (first == last ? 1 : 0);
            last = end;
        }
    return 32.0 * double(sectors);
}

std::string joined(const std::vector<uint64_t>& v) {
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) s += (i ? "x" : "") + std::to_string(v[i]);
    return s;
}

// One timed thing: run_once advances its own views.
struct Timed {
    std::function<void()> run_once;
    double probe_us = 0;
    uint64_t launches = 1;
    std::vector<std::unique_ptr<TimingEvent>> starts, stops;

    void probe(Stream& stream, const Settings& settings) {
        run_once();
        TimingEvent start, stop;
        check(cudaEventRecord(start.handle, stream.getStream()), "probe start");
        run_once();
        check(cudaEventRecord(stop.handle, stream.getStream()), "probe stop");
        stream.synchronize();
        float ms = 0;
        check(cudaEventElapsedTime(&ms, start.handle, stop.handle), "probe elapsed");
        probe_us = 1000.0 * ms;
        launches = B::ColdTiming::launchesPerSample(probe_us, settings.interval_us, settings.max_launches);
        for (int p = 0; p < settings.samples; ++p) {
            starts.push_back(std::make_unique<TimingEvent>());
            stops.push_back(std::make_unique<TimingEvent>());
        }
    }
    void submit(int pass, Stream& stream) {
        run_once();  // untimed, so the L2 holds this item's own last output when the timing starts
        check(cudaEventRecord(starts[pass]->handle, stream.getStream()), "start event");
        for (uint64_t l = 0; l < launches; ++l) run_once();
        check(cudaEventRecord(stops[pass]->handle, stream.getStream()), "stop event");
    }
    [[nodiscard]] std::vector<double> samples() const {
        std::vector<double> us;
        for (size_t p = 0; p < starts.size(); ++p) {
            float ms = 0;
            check(cudaEventElapsedTime(&ms, starts[p]->handle, stops[p]->handle), "sample elapsed");
            us.push_back(1000.0 * ms / static_cast<double>(launches));
        }
        return us;
    }
};

struct Cell {
    CellSpec spec;
    T::DenseCopyPlan plan;
    std::vector<T::StridedSpan> inputs, outputs;
    size_t next_input = 0, next_output = 0, next_ref_input = 0;
    double useful_bytes = 0, touched_bytes = 0, moved_bytes = 0, copy_us = 0;
    bool verified = false;
    std::string ref_1387 = "none";
    Timed main, ref;
    bool has_ref = false;
};

// --probe: a plan and what the variant column calls it.
struct Variant {
    std::string name;
    T::DenseCopyPlan plan;
};

// The plan, then its kernel on smaller grids, then (a Transpose or MatrixTranspose plan) the transpose kernel on other
// tiles in both tile orders, then the plan again. Tiles keep within the planner's shared memory; a tile already
// listed is not repeated.
std::vector<Variant> probeVariants(const T::DenseCopyPlan& planned) {
    namespace I = T::DenseCopyInternal;
    using B::ColdTiming::ceilDiv;
    std::vector<Variant> out{{"plan", planned}};
    if (planned.grid > 0) {
        const uint64_t units = I::workUnits(planned);
        std::vector<uint64_t> grids{planned.grid};
        const auto add = [&](const std::string& name, uint64_t grid) {
            grid = std::clamp<uint64_t>(grid, 1, units);
            if (std::find(grids.begin(), grids.end(), grid) != grids.end()) return;
            grids.push_back(grid);
            T::DenseCopyPlan plan = planned;
            plan.grid = static_cast<uint32_t>(grid);
            out.push_back({name, plan});
        };
        add("grid_d2", planned.grid / 2);
        add("grid_d4", planned.grid / 4);
        add("grid_d16", planned.grid / 16);
        add("grid_d64", planned.grid / 64);
    }
    if (planned.family == T::DenseCopyFamily::Transpose || planned.family == T::DenseCopyFamily::MatrixTranspose) {
        T::DenseCopyPlan base = planned;
        base.family = T::DenseCopyFamily::Transpose;  // a matrix transpose's plan has the transpose's x, v and strides
        const uint64_t vector_bytes = base.p * base.word_bytes;
        const auto add = [&](const std::string& name, uint64_t tile_x, uint64_t tile_v) {
            tile_x = std::clamp<uint64_t>(tile_x, 1, base.x);
            tile_v = std::clamp<uint64_t>(tile_v, 1, base.v);
            // The shared bytes of the plan's tile layout (the packed byte tile for single bytes)
            while (tile_v > 0 && I::transposeSharedBytes(tile_x, tile_v, base.p, base.word_bytes) >
                                     I::TRANSPOSE_MAX_SHARED_BYTES)
                tile_v -= std::max<uint64_t>(1, tile_v / 8);
            if (tile_v == 0) return;  // one tile row is past the shared memory
            for (bool v_first : {false, true}) {
                if (v_first && (ceilDiv(base.x, tile_x) == 1 || ceilDiv(base.v, tile_v) == 1)) continue;  // one order
                T::DenseCopyPlan plan = base;
                plan.tile_x = static_cast<uint32_t>(tile_x);
                plan.tile_v = static_cast<uint32_t>(tile_v);
                plan.tile_v_first = v_first;
                if (I::workUnits(plan) >= I::MAX_32BIT_INDEX) continue;
                const bool listed = std::any_of(out.begin(), out.end(), [&plan](const Variant& other) {
                    return other.plan.family == plan.family && other.plan.tile_x == plan.tile_x &&
                           other.plan.tile_v == plan.tile_v && other.plan.tile_v_first == plan.tile_v_first;
                });
                if (listed) continue;
                plan.grid = I::gridOf(plan);
                out.push_back({name + (v_first ? "_vfirst" : "_xfirst"), plan});
            }
        };
        const uint64_t side = std::max<uint64_t>(32, ceilDiv(128, vector_bytes));
        if (planned.family == T::DenseCopyFamily::Transpose) add("planned", planned.tile_x, planned.tile_v);
        add("square", side, side);
        add("tall32", ceilDiv(32, vector_bytes), base.v);
        add("tall128", ceilDiv(128, vector_bytes), base.v);
        add("wide", 4 * side, std::max<uint64_t>(8, side / 4));
    }
    out.push_back({"plan_repeat", planned});
    return out;
}

// One variant of a cell's plan, timed on the cell's views.
struct Probe {
    std::shared_ptr<Cell> cell;
    Variant variant;
    bool verified = false;
    Timed timed;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        const Settings settings = parse(argc, argv);
        const std::vector<CellSpec> specs = cells(settings);
        const uint64_t total = std::min<uint64_t>(specs.size(), settings.max_cells);
        const uint64_t e = settings.dtype->bytes;
        if (settings.list_only) {
            std::cout << "cell_id,case,sub,dtype,dims,strides,reach_elements\n";
            for (uint64_t n = 0; n < total; ++n) {
                const CellSpec& c = specs[n];
                std::cout << c.id << ',' << c.kind << ',' << c.sub << ',' << settings.dtype->name << ','
                          << joined(c.dims) << ',' << joined(c.strides) << ',' << c.reach << '\n';
            }
            return 0;
        }

        const T::TensorPlacement placement(T::TensorPlacement::MemDevices::GPU, 0);
        ScopedGpu scoped_gpu(0);
        Stream stream(placement);
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties, 0), "device properties");
        const T::DataType dtype = settings.dtype->dtype;

        Ring inputs(placement, settings.arena_bytes);
        Ring outputs(placement, settings.output_ring_bytes);
        fillHashKernel<<<4 * properties.multiProcessorCount, 256, 0, stream.getStream()>>>(
            static_cast<uint32_t*>(inputs.storage().getMemPtr<void>()), settings.arena_bytes / 4,
            static_cast<uint32_t>(settings.seed));
        check(cudaGetLastError(), "input ring fill");
        check(cudaMemsetAsync(outputs.storage().getMemPtr<void>(), 0, settings.output_ring_bytes, stream.getStream()),
              "output ring initialization");
        T::Tensor check_buffer(placement, T::TensorDescriptor(T::DataType::UINT8, {settings.max_mib * MiB + 256}));
        T::Tensor mismatches(placement, T::TensorDescriptor(T::DataType::UINT64, {1}));
        stream.synchronize();

        uint64_t min_moved = UINT64_MAX, max_moved = 4 * MiB;
        for (uint64_t n = 0; n < total; ++n) {
            const double moved = touchedSourceBytes(specs[n], e) + double(product(specs[n].dims) * e);
            min_moved = std::min<uint64_t>(min_moved, static_cast<uint64_t>(moved));
            max_moved = std::max<uint64_t>(max_moved, static_cast<uint64_t>(moved));
        }
        const uint64_t copy_bytes = 256 * MiB;
        const double bandwidth = B::ColdTiming::copyBandwidth(inputs, outputs, stream, copy_bytes);
        const CopyCurve curve = B::ColdTiming::copyCurve(inputs, outputs, stream, max_moved, settings.interval_us,
                                                         std::max<uint64_t>(4 * KiB, std::min(min_moved, max_moved)));

        std::cout << "# dense_copy_audit=1 version=" << T::denseCopyRevision()
                  << " method=ring_batched_probe pass_warmup=1 dtype=" << settings.dtype->name
                  << " cells=" << total << " cells_per_case=" << settings.cells << " seed=" << settings.seed
                  << " min_kib=" << settings.min_bytes / KiB << " max_mib=" << settings.max_mib
                  << " max_span_mib=" << settings.max_span_bytes / MiB << " samples=" << settings.samples
                  << " interval_us=" << settings.interval_us << " max_launches=" << settings.max_launches
                  << " arena_bytes=" << settings.arena_bytes << " output_ring_bytes=" << settings.output_ring_bytes
                  << " input_reuse_bytes=" << InputReuseBytes << " output_reuse_bytes=" << OutputReuseBytes
                  << " device=\"" << properties.name << "\" sm_count=" << properties.multiProcessorCount
                  << " cc=" << properties.major << '.' << properties.minor << " l2_bytes=" << properties.l2CacheSize
                  << " copy_bandwidth_tbps=" << std::fixed << std::setprecision(4) << bandwidth / 1e6 << '\n';
        std::cout << "# copy_curve";
        for (const auto& [moved, us] : curve.points) std::cout << ' ' << static_cast<uint64_t>(moved) << '=' << us;
        std::cout << '\n';
        if (settings.probe) {
            std::cout << "# probe=1 probe_min_mib=" << settings.probe_min_bytes / MiB << '\n'
                      << "cell_id,case,sub,dtype,dims,strides,variant,plan,grid,useful_bytes,moved_bytes,verified,"
                         "probe_us,launches,median_us,copy_us,tbps,of_copy\n";
        } else {
            std::cout << "cell_id,case,sub,dtype,dims,strides,plan,useful_bytes,touched_bytes,moved_bytes,verified,"
                         "probe_us,launches";
            for (int i = 0; i < settings.samples; ++i) std::cout << ",sample" << i << "_us";
            std::cout << ",median_us,copy_us,ref_1387,ref_1387_us,tbps,of_copy,vs_1387\n";
        }

        uint64_t collected = 0, failed = 0, unverified = 0;
        const auto wall_start = std::chrono::steady_clock::now();
        B::Fp8OracleBatchQueue queue(settings.queue_cells, settings.samples, [&] { stream.synchronize(); });

        const auto print = [&](const Cell& cell) {
            const CellSpec& c = cell.spec;
            const std::vector<double> us = cell.main.samples();
            const double med = B::ColdTiming::median(us);
            const double ref = cell.has_ref ? B::ColdTiming::median(cell.ref.samples()) : 0.0;
            std::cout << std::defaultfloat << c.id << ',' << c.kind << ',' << c.sub << ',' << settings.dtype->name
                      << ',' << joined(c.dims) << ',' << joined(c.strides) << ',' << T::denseCopyPlanName(cell.plan)
                      << ',' << uint64_t(cell.useful_bytes) << ',' << uint64_t(cell.touched_bytes) << ','
                      << uint64_t(cell.moved_bytes) << ',' << int(cell.verified) << ',' << std::setprecision(4)
                      << cell.main.probe_us << ',' << cell.main.launches;
            for (double u : us) std::cout << ',' << u;
            std::cout << ',' << med << ',' << cell.copy_us << ',' << cell.ref_1387 << ',' << ref << ','
                      << cell.moved_bytes / med / 1e6 << ',' << cell.copy_us / med << ','
                      << (ref > 0 ? ref / med : 0.0) << '\n';
            ++collected;
            if (collected % 25 == 0 || collected + failed == total) {
                const double wall =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count();
                std::cerr << "[dense-copy-audit] cells=" << collected + failed << '/' << total << " failed=" << failed
                          << " unverified=" << unverified << " wall_s=" << std::fixed << std::setprecision(1) << wall
                          << std::defaultfloat << std::endl;
            }
        };

        // --probe: one row per variant; `collected` counts the rows.
        const auto printProbe = [&](const Probe& probe) {
            const Cell& cell = *probe.cell;
            const CellSpec& c = cell.spec;
            const double med = B::ColdTiming::median(probe.timed.samples());
            std::cout << std::defaultfloat << c.id << ',' << c.kind << ',' << c.sub << ',' << settings.dtype->name
                      << ',' << joined(c.dims) << ',' << joined(c.strides) << ',' << probe.variant.name << ','
                      << T::denseCopyPlanName(probe.variant.plan) << ',' << probe.variant.plan.grid << ','
                      << uint64_t(cell.useful_bytes) << ',' << uint64_t(cell.moved_bytes) << ','
                      << int(probe.verified) << ',' << std::setprecision(4) << probe.timed.probe_us << ','
                      << probe.timed.launches << ',' << med << ',' << cell.copy_us << ','
                      << cell.moved_bytes / med / 1e6 << ',' << cell.copy_us / med << '\n';
            ++collected;
            if (collected % 50 == 0)
                std::cerr << "[dense-copy-audit] probe rows=" << collected << " unverified=" << unverified
                          << std::endl;
        };

        // The bytes of the device compare, 0 when they match.
        const auto mismatched = [&](const void* a, const void* b, uint64_t bytes) {
            check(cudaMemsetAsync(mismatches.getMemPtr<void>(), 0, 8, stream.getStream()), "mismatch reset");
            countMismatchKernel<<<4 * properties.multiProcessorCount, 256, 0, stream.getStream()>>>(
                static_cast<const uint8_t*>(a), static_cast<const uint8_t*>(b), bytes,
                static_cast<unsigned long long*>(mismatches.getMemPtr<void>()));
            check(cudaGetLastError(), "mismatch kernel");
            unsigned long long count = 0;
            check(cudaMemcpyAsync(&count, mismatches.getMemPtr<void>(), 8, cudaMemcpyDeviceToHost, stream.getStream()),
                  "mismatch copy");
            stream.synchronize();
            return count;
        };

        for (uint64_t n = 0; n < total; ++n) {
            auto cell = std::make_shared<Cell>();
            cell->spec = specs[n];
            const CellSpec& c = cell->spec;
            try {
                const uint64_t elements = product(c.dims);
                cell->useful_bytes = double(elements * e);
                cell->touched_bytes = touchedSourceBytes(c, e);
                cell->moved_bytes = cell->touched_bytes + cell->useful_bytes;
                cell->copy_us = curve.us(cell->moved_bytes);
                const uint64_t span_bytes = c.reach * e;

                // Input views, each a fresh region of the ring; output views likewise.
                const uint64_t wanted = std::clamp<uint64_t>(
                    static_cast<uint64_t>(std::ceil(double(InputReuseBytes) / std::max(cell->touched_bytes, 1.0))), 1,
                    std::min<uint64_t>(MaxViews, std::max<uint64_t>(1, inputs.bytes() / 2 / Ring::span(span_bytes))));
                for (uint64_t v = 0; v < wanted; ++v) {
                    const uint64_t offset = inputs.take(span_bytes);
                    cell->inputs.push_back(inputs.view(dtype, {c.reach}, offset).withLayout(c.dims, c.strides));
                }
                for (uint64_t v = 0, count = B::ColdTiming::viewCount(elements * e, OutputReuseBytes, outputs.bytes());
                     v < count; ++v)
                    cell->outputs.push_back(outputs.view(dtype, c.dims, outputs.take(elements * e)));

                cell->plan = T::planCopyToDense(cell->inputs[0], cell->outputs[0]);

                // ref_1387 on the same views.
                const std::optional<Pitched1387> pitched = pitched1387(c.dims, c.strides, e);
                const bool dense = c.strides == denseStrides(c.dims);
                cell->ref_1387 = dense ? "memcpy" : pitched ? "memcpy_2d" : "gather";
                SpanLayout1387 layout;
                layout.rank = static_cast<uint32_t>(c.dims.size());
                for (uint32_t k = 0; k < layout.rank; ++k) {
                    layout.dimensions[k] = c.dims[k];
                    layout.strides[k] = c.strides[k];
                }
                const uint64_t gather_blocks = B::ColdTiming::ceilDiv(elements, 256);
                const auto run1387 = [e, elements, layout, gather_blocks, dense, pitched](const T::StridedSpan& in,
                                                                                         const T::StridedSpan& out,
                                                                                         Stream& s) {
                    const void* src = in.getMemPtr<void>();
                    void* dst = out.getMemPtr<void>();
                    if (dense) {
                        check(cudaMemcpyAsync(dst, src, elements * e, cudaMemcpyDeviceToDevice, s.getStream()),
                              "1387 memcpy");
                    } else if (pitched) {
                        check(cudaMemcpy2DAsync(dst, pitched->row_bytes, src, pitched->pitch_bytes, pitched->row_bytes,
                                                pitched->rows, cudaMemcpyDeviceToDevice, s.getStream()),
                              "1387 memcpy 2d");
                    } else if (e == 1) {
                        gather1387Kernel<uint8_t><<<static_cast<uint32_t>(gather_blocks), 256, 0, s.getStream()>>>(
                            static_cast<const uint8_t*>(src), static_cast<uint8_t*>(dst), elements, layout);
                    } else if (e == 2) {
                        gather1387Kernel<uint16_t><<<static_cast<uint32_t>(gather_blocks), 256, 0, s.getStream()>>>(
                            static_cast<const uint16_t*>(src), static_cast<uint16_t*>(dst), elements, layout);
                    } else {
                        gather1387Kernel<uint32_t><<<static_cast<uint32_t>(gather_blocks), 256, 0, s.getStream()>>>(
                            static_cast<const uint32_t*>(src), static_cast<uint32_t*>(dst), elements, layout);
                    }
                    check(cudaGetLastError(), "1387 path");
                };

                // Verified run: the plan against ref_1387 on the first input view.
                T::launchCopyToDense(cell->plan, cell->inputs[0], cell->outputs[0], stream);
                const T::StridedSpan check_view =
                    T::StridedSpan(check_buffer).bytesAs(0, dtype, c.dims);
                run1387(cell->inputs[0], check_view, stream);
                const unsigned long long bad =
                    mismatched(cell->outputs[0].getMemPtr<void>(), check_view.getMemPtr<void>(), elements * e);
                cell->verified = bad == 0;
                if (!cell->verified) {
                    ++unverified;
                    std::cout << "# verify_failed," << c.id << ',' << c.kind << ",plan="
                              << T::denseCopyPlanName(cell->plan) << ",mismatched_bytes=" << bad << '\n';
                }

                // --probe: the plan's variants instead of the cell's row, each checked against ref_1387's bytes
                // (check_view) before it is timed. The variants share the cell's views.
                if (settings.probe) {
                    if (cell->useful_bytes < double(settings.probe_min_bytes)) continue;
                    for (Variant& variant : probeVariants(cell->plan)) {
                        auto probe = std::make_shared<Probe>();
                        probe->cell = cell;
                        probe->variant = std::move(variant);
                        T::launchCopyToDense(probe->variant.plan, cell->inputs[0], cell->outputs[0], stream);
                        probe->verified = mismatched(cell->outputs[0].getMemPtr<void>(),
                                                     check_view.getMemPtr<void>(), elements * e) == 0;
                        if (!probe->verified) {
                            ++unverified;
                            std::cout << "# verify_failed," << c.id << ',' << c.kind << ",variant="
                                      << probe->variant.name << ",plan=" << T::denseCopyPlanName(probe->variant.plan)
                                      << '\n';
                        }
                        Probe* raw_probe = probe.get();
                        probe->timed.run_once = [raw_probe, &stream] {
                            Cell& owner = *raw_probe->cell;
                            const T::StridedSpan& input = owner.inputs[owner.next_input];
                            owner.next_input = (owner.next_input + 1) % owner.inputs.size();
                            const T::StridedSpan& output = owner.outputs[owner.next_output];
                            owner.next_output = (owner.next_output + 1) % owner.outputs.size();
                            T::launchCopyToDense(raw_probe->variant.plan, input, output, stream);
                        };
                        probe->timed.probe(stream, settings);
                        queue.add({[] {},
                                   [probe, &stream](int pass) { probe->timed.submit(pass, stream); },
                                   [probe, &printProbe] { printProbe(*probe); }});
                    }
                    continue;
                }

                Cell* raw = cell.get();
                cell->main.run_once = [raw, &stream] {
                    const T::StridedSpan& input = raw->inputs[raw->next_input];
                    raw->next_input = (raw->next_input + 1) % raw->inputs.size();
                    const T::StridedSpan& output = raw->outputs[raw->next_output];
                    raw->next_output = (raw->next_output + 1) % raw->outputs.size();
                    T::launchCopyToDense(raw->plan, input, output, stream);
                };
                cell->main.probe(stream, settings);
                cell->ref.run_once = [raw, &stream, run1387] {
                    const T::StridedSpan& input = raw->inputs[raw->next_ref_input];
                    raw->next_ref_input = (raw->next_ref_input + 1) % raw->inputs.size();
                    const T::StridedSpan& output = raw->outputs[raw->next_output];
                    raw->next_output = (raw->next_output + 1) % raw->outputs.size();
                    run1387(input, output, stream);
                };
                cell->ref.probe(stream, settings);
                cell->has_ref = true;
            } catch (const std::exception& ex) {
                ++failed;
                std::string what = ex.what();
                std::replace(what.begin(), what.end(), '\n', ' ');
                std::cout << "# failed," << c.id << ',' << c.kind << ",dims=" << joined(c.dims)
                          << ",strides=" << joined(c.strides) << ',' << what << '\n';
                continue;
            }
            queue.add({[] {},
                       [cell, &stream](int pass) {
                           cell->main.submit(pass, stream);
                           if (cell->has_ref) cell->ref.submit(pass, stream);
                       },
                       [cell, &print] { print(*cell); }});
        }
        queue.flush();
        const double bandwidth_end = B::ColdTiming::copyBandwidth(inputs, outputs, stream, copy_bytes);
        const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count();
        std::cout << "# end cells=" << collected << " failed=" << failed << " unverified=" << unverified
                  << " batches=" << queue.batches() << " copy_bandwidth_tbps_end=" << std::fixed
                  << std::setprecision(4) << bandwidth_end / 1e6 << " input_ring_wraps=" << inputs.wraps()
                  << " output_ring_wraps=" << outputs.wraps() << " wall_s=" << std::setprecision(1) << wall << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "thor_dense_copy_audit: " << e.what() << '\n';
        return 1;
    }
}
