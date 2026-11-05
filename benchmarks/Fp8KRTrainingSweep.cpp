// FP8 R / KR training sweep (1151-1158): the KR counterpart of `thor_fp8_retained_cold_l2_sweep --training`.
//
// The questions, as the RK training sweep asked them for RK:
//   - does any retained KR shape reach line rate at this site (a kernel gap when none does), and
//   - when one does, does fp8KRCandidateShapes list it, how early, and does the production stamp pick it?
//
// Cells. A cell is one site [K, R] (K = 1 is a pure R reduction) with one sampled semantics:
//   - the grid: the audit's row lengths, plus pure R, at each --targets-mib working set;
//   - --random log-uniform geometries: R in [2, 2^30] and input bytes in [--min-mib, --max-mib], K = bytes / R
//     (R >= bytes gives a pure R site of that size), so odd rows and odd sizes are covered;
//   - --semantics cells per geometry alternate VALUE (sum, l1_norm, sum_squares, product, min, max; FP32 output) and
//     ARG (argmin, argmax; UINT32 index, with or without the FP32 value), with a sampled dtype (E4M3 / E5M2);
//   - 1179 --dtypes=fp32 (alone or with the FP8 dtypes): FP32 VALUE cells on the same kernels
//     (launchFp32KRValueSegments, not yet routed in production) and the FP32 pool (retainedKRCandidatePool at element size 4). An FP32 cell takes
//     its geometry's row length in elements (R = the grid's row length, row bytes 4 R; a pure R site keeps its bytes)
//     and a quarter of its rows, so it reads about the same bytes as an FP8 cell; the grid's short rows (R = 2..8)
//     are the generic Pack measurement on short FP32 rows that decides whether vector Pack is ported. 1181: FP32 ARG
//     cells too (launchFp32KRArgSegments, the FP32 ARG pool: fp32KRArgShapeLegal). Shape names and columns other than
//     R are in bytes; input_bytes = 4 K R; the ranker features are FP8's (FP32's ranker comes in phase 4);
//   - 1217 --carried-index=uint32|uint64 (with --dtypes=fp32 and ARG operations only): the ARG cells are carried-index
//     (intermediate) sites, kinds carried_arg_index / carried_arg_value_index: FP32 values plus a cold carried index
//     of that width (each launch reads its own view of both), timed through launchFp32KRCarriedArgSegments at row
//     stride 1 (an original run); the pool is the FP32 ARG pool, the ranker site kind 3 / 4. A carried cell takes an
//     FP32 cell's geometry with rows (or a pure R site's elements) scaled by 4 / (4 + index bytes), so its working set
//     (values and index) is the FP32 cell's; input_bytes and moved_tbps count the index. No production stamp (the
//     planner does not route carried sites to the retained kernels yet).
//   - 1217: every row carries frozen_margin, the installed ranker's margin for the candidate (FP32's model for FP32
//     cells, FP8's otherwise), so the installed model's regret on new cells is measured without retraining.
//   - a seeded shuffle, so any --max-cells prefix samples the whole grid; cell ids keep the grid order;
//   - heldout marks --heldout-percent of row lengths R (every size and semantics of a held-out R together).
//
// Candidates. 1158: production's pool, fp8KRCandidatePool, with each candidate's ranker features (fp8KRRankerFeatures,
// columns f0..f23) so benchmarks/fp8_kr_ranker trains on exactly what production ranks. The pool is every shape
// fp8KRCandidateShapes lists (in_list=1, list_position = its rank in the list), then, unless --list-only, the other
// legal shapes nearby (in_list=0): Pack at 2, 3, 4, 6, 8, 12, 16 rows per thread and the vector shape, Group at every
// G, Block, and Split at every power-of-two chunk while rows < 64 per SM or rows have at least 16 KiB; an extra shape
// that gives one thread more than FP8_KR_POOL_MAX_THREAD_BYTES of input is not in it. pool_position is the 1-based
// position in the pool. An extra shape whose probe launch takes over 20x the line-rate reference is not timed further
// (abandoned=1; the row carries the probe time). Unless --no-production, the production stamp (CubReduction /
// CubArgReduction at the default tuning depth) is made at each site and its pick recorded (production, is_production).
// 1155 --split-stages: every Split shape is also timed with its partial kernel alone (variant=partials, the output
// unwritten; not a candidate), so the finishing kernel's cost is the difference from the full row.
//
// Timing (1152: the RK sweep's method, which reproduces to about 3 digits across runs; the 1150-1151 flush-per-sample
// timing did not: the same shape timed twice in one 1150 audit run disagreed by up to 28%).
//   - Cold inputs without a flush kernel: every launch reads its own view of an --arena-gib ring (256-byte aligned,
//     as a fresh allocation), cycling through views spanning at least 2 GiB, so no input is re-read within 20x L2.
//   - Cold outputs too: every launch writes its own view of an --output-ring-gib ring, cycling through views spanning
//     at least 1 GiB, so each launch pays for the write-back of its output as a real workload does. (A shared output
//     stays in L2 between launches when it is smaller than L2: at short rows, where the output is 0.5-2x the input,
//     that hid most of the write traffic and showed moved TB/s above the copy rate.) Split partials share one scratch.
//   - Asynchronous batches as the RK sweep (Fp8OracleBatchQueue): --queue-cells cells are prepared (production stamp,
//     one probe launch of every candidate, which also loads every kernel), then all their passes are submitted at
//     once and the stream is synchronized once. Pass p visits every cell, each cell's candidates starting at a rotated
//     position, so a candidate's samples are spread over the batch and the GPU never idles between samples.
//   - A sample times `launches_per_sample` back-to-back launches (each on its own views) and divides by the count,
//     about --interval-us per sample (CUDA events here resolve about 2 us). 1153: the count is per candidate, from its
//     own probe time (1152 used the cell's line-rate estimate for every candidate, so a shape 10x off line rate took
//     10 ms per sample: 80% of the GPU time went to shapes at least 5x slower than the cell's best).
//   - Inputs are the RK sweep's constant byte 0x30 (E4M3 0.5, E5M2 0.125).
// moved_tbps = (input + output bytes) / median, read_tbps = input bytes / median; reference_us = max(--floor-us,
// moved bytes / copy bandwidth), the copy measured the same way (ring input, ring output) at start and end. 1153:
// copy_us is a device-to-device copy moving the same bytes (moved / 2 copied), interpolated from a copy curve timed
// the same way at startup: what a memcpy achieves at this size, launch and ramp included (the header lists the
// curve). Very short kernels can be host-submission bound (a launch costs the host a few us), as in production.
//
// stdout: a comment header, one CSV row per (cell, candidate) and a closing comment; stderr: progress. Typical use:
//   ./thor_fp8_kr_training_sweep --max-cells=400 2> kr-training.log | gzip -1 > kr-training.csv.gz
//   python3 ../benchmarks/fp8_kr_training.py kr-training.csv.gz [--compare=kr-training-2.csv.gz]
#include "benchmarks/ColdTiming.h"
#include "benchmarks/Fp8OracleBatchQueue.h"
#include "Utilities/TensorOperations/Cub/CubReduction.h"
#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"
#include "Utilities/TensorOperations/Cub/CubReductionStageSemantics.h"
#include "Utilities/TensorOperations/Cub/Fp32KR/Fp32KRRankerModel.h"
#include "Utilities/TensorOperations/Cub/Fp8KR/Fp8KRRankerModel.h"
#include "Utilities/TensorOperations/Cub/Fp8KR/Fp8KRSegments.h"
#include "Utilities/Common/ScopedGpu.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace T = ThorImplementation;
namespace C = ThorImplementation::CubReductionInternal;
namespace KR = ThorImplementation::CubReductionInternal::Fp8KR;
namespace B = ThorImplementation::CubReductionBenchmarking;

namespace {

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

constexpr uint64_t KiB = 1ULL << 10;
constexpr uint64_t MiB = 1ULL << 20;
constexpr uint64_t GiB = 1ULL << 30;
// 1164: the timing machinery (rings, views, probe launch counts, copy reference) is benchmarks/ColdTiming.h, shared
// with the RK training sweep.
constexpr uint64_t INPUT_REUSE_BYTES = B::ColdTiming::InputReuseBytes;
constexpr uint64_t OUTPUT_REUSE_BYTES = B::ColdTiming::OutputReuseBytes;
using B::ColdTiming::align256;
using B::ColdTiming::ceilDiv;
using B::ColdTiming::copyBandwidth;
using B::ColdTiming::copyCurve;
using B::ColdTiming::CopyCurve;
using B::ColdTiming::median;
using B::ColdTiming::Ring;
using B::ColdTiming::TimingEvent;
using B::ColdTiming::viewCount;

enum class Kind : uint8_t { Value, ArgIndex, ArgValueIndex, CarriedArgIndex, CarriedArgValueIndex };
const char* kindName(Kind kind) {
    switch (kind) {
        case Kind::Value: return "value";
        case Kind::ArgIndex: return "arg_index";
        case Kind::ArgValueIndex: return "arg_value_index";
        case Kind::CarriedArgIndex: return "carried_arg_index";  // 1217
        case Kind::CarriedArgValueIndex: return "carried_arg_value_index";
    }
    return "unknown";
}
bool isCarried(Kind kind) { return kind == Kind::CarriedArgIndex || kind == Kind::CarriedArgValueIndex; }
bool producesValue(Kind kind) { return kind == Kind::ArgValueIndex || kind == Kind::CarriedArgValueIndex; }

struct ValueOp {
    const char* name;
    T::CubReductionOp op;
};
// One VALUE operation per compiled KR semantics: SUM x {identity, abs, square}, PRODUCT, MIN, MAX.
constexpr ValueOp VALUE_OPS[] = {{"sum", T::CubReductionOp::Sum},
                                 {"l1_norm", T::CubReductionOp::L1Norm},
                                 {"sum_squares", T::CubReductionOp::SumSquares},
                                 {"product", T::CubReductionOp::Product},
                                 {"min", T::CubReductionOp::Min},
                                 {"max", T::CubReductionOp::Max}};
constexpr const char* ARG_OPS[] = {"argmin", "argmax"};

struct Settings {
    uint64_t max_cells = std::numeric_limits<uint64_t>::max();
    uint64_t seed = 1151;
    uint64_t random = 3000;
    uint32_t semantics_per_geometry = 2;  // even: VALUE, odd: ARG
    uint32_t heldout_percent = 10;        // of row lengths R
    std::vector<uint64_t> targets_mib{8, 32, 128, 512};
    uint64_t min_bytes = 4 * MiB;  // 1158: --min-kib (or --min-mib)
    uint64_t max_mib = 512;
    int samples = 5;
    std::vector<T::DataType> dtypes{T::DataType::FP8_E4M3, T::DataType::FP8_E5M2};
    std::vector<std::string> ops;  // empty: all
    bool list_only = false;
    bool production = true;
    bool split_stages = false;  // 1155: also time each Split shape's partial kernel alone
    double floor_us = 4.0;
    uint64_t arena_bytes = 8 * GiB;
    uint64_t output_ring_bytes = 3 * GiB;
    double interval_us = 1000.0;
    uint64_t max_launches = 256;
    size_t queue_cells = 128;
    std::optional<T::DataType> carried_index;  // 1217 --carried-index=uint32|uint64
};

std::vector<std::string> splitList(std::string_view text) {
    std::vector<std::string> items;
    size_t begin = 0;
    while (begin <= text.size()) {
        const size_t end = std::min(text.find(',', begin), text.size());
        if (end > begin) items.emplace_back(text.substr(begin, end - begin));
        begin = end + 1;
    }
    return items;
}

bool knownOp(const std::string& name) {
    for (const ValueOp& op : VALUE_OPS)
        if (name == op.name) return true;
    for (const char* op : ARG_OPS)
        if (name == op) return true;
    return false;
}

Settings parse(int argc, char** argv) {
    Settings s;
    for (int i = 1; i < argc; ++i) {
        const std::string_view a(argv[i]);
        const auto value = [&](std::string_view prefix) -> std::optional<std::string> {
            if (a.substr(0, prefix.size()) == prefix) return std::string(a.substr(prefix.size()));
            return std::nullopt;
        };
        if (auto v = value("--max-cells=")) {
            s.max_cells = std::stoull(*v);
        } else if (auto v = value("--seed=")) {
            s.seed = std::stoull(*v);
        } else if (auto v = value("--random=")) {
            s.random = std::stoull(*v);
        } else if (auto v = value("--semantics=")) {
            s.semantics_per_geometry = static_cast<uint32_t>(std::stoul(*v));
            if (s.semantics_per_geometry == 0) throw std::invalid_argument("--semantics must be >= 1");
        } else if (auto v = value("--heldout-percent=")) {
            s.heldout_percent = static_cast<uint32_t>(std::stoul(*v));
            if (s.heldout_percent > 100) throw std::invalid_argument("--heldout-percent must be <= 100");
        } else if (auto v = value("--targets-mib=")) {
            s.targets_mib.clear();
            for (const std::string& t : splitList(*v)) s.targets_mib.push_back(std::stoull(t));
        } else if (auto v = value("--min-mib=")) {
            s.min_bytes = std::stoull(*v) * MiB;
        } else if (auto v = value("--min-kib=")) {
            s.min_bytes = std::stoull(*v) * KiB;
        } else if (auto v = value("--max-mib=")) {
            s.max_mib = std::stoull(*v);
        } else if (auto v = value("--samples=")) {
            s.samples = std::max(3, std::stoi(*v));
        } else if (auto v = value("--dtypes=")) {
            s.dtypes.clear();
            for (const std::string& d : splitList(*v)) {
                if (d == "fp8_e4m3") s.dtypes.push_back(T::DataType::FP8_E4M3);
                else if (d == "fp8_e5m2") s.dtypes.push_back(T::DataType::FP8_E5M2);
                else if (d == "fp32") s.dtypes.push_back(T::DataType::FP32);  // 1179
                else throw std::invalid_argument("--dtypes takes fp8_e4m3,fp8_e5m2,fp32");
            }
        } else if (auto v = value("--ops=")) {
            s.ops = splitList(*v);
            for (const std::string& op : s.ops)
                if (!knownOp(op))
                    throw std::invalid_argument("--ops takes sum,l1_norm,sum_squares,product,min,max,argmin,argmax");
        } else if (a == "--list-only") {
            s.list_only = true;
        } else if (a == "--no-production") {
            s.production = false;
        } else if (a == "--split-stages") {
            s.split_stages = true;
        } else if (auto v = value("--floor-us=")) {
            s.floor_us = std::stod(*v);
        } else if (auto v = value("--arena-gib=")) {
            s.arena_bytes = std::stoull(*v) * GiB;
        } else if (auto v = value("--output-ring-gib=")) {
            s.output_ring_bytes = std::stoull(*v) * GiB;
        } else if (auto v = value("--interval-us=")) {
            s.interval_us = std::stod(*v);
        } else if (auto v = value("--max-launches=")) {
            s.max_launches = std::max<uint64_t>(1, std::stoull(*v));
        } else if (auto v = value("--queue-cells=")) {
            s.queue_cells = std::max<size_t>(1, std::stoull(*v));
        } else if (a == "--carried-index=uint32") {
            s.carried_index = T::DataType::UINT32;
        } else if (a == "--carried-index=uint64") {
            s.carried_index = T::DataType::UINT64;
        } else if (a == "--help") {
            std::cout << "thor_fp8_kr_training_sweep [--max-cells=N] [--seed=1151] [--random=3000] [--semantics=2]\n"
                         "    [--heldout-percent=10] [--targets-mib=8,32,128,512] [--min-kib=4096 | --min-mib=4]\n"
                         "    [--max-mib=512]\n"
                         "    [--samples=5] [--dtypes=fp8_e4m3,fp8_e5m2[,fp32]] [--ops=sum,...,argmax] [--list-only]\n"
                         "    [--no-production] [--split-stages] [--floor-us=4] [--arena-gib=8]\n"
                         "    [--output-ring-gib=3] [--interval-us=1000] [--max-launches=256] [--queue-cells=128]\n"
                         "    [--carried-index=uint32|uint64]  (1217; with --dtypes=fp32 --ops=argmin,argmax)\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("Unknown argument " + std::string(a));
        }
    }
    if (s.dtypes.empty()) throw std::invalid_argument("--dtypes is empty");
    if (s.carried_index) {
        const bool fp32_only = s.dtypes.size() == 1 && s.dtypes.front() == T::DataType::FP32;
        const bool arg_only = !s.ops.empty() && std::all_of(s.ops.begin(), s.ops.end(), [](const std::string& op) {
            return op == "argmin" || op == "argmax";
        });
        if (!fp32_only || !arg_only)
            throw std::invalid_argument("--carried-index runs with --dtypes=fp32 and --ops=argmin,argmax only");
    }
    if (s.min_bytes < 4 * KiB || s.min_bytes > s.max_mib * MiB)
        throw std::invalid_argument("need 4 KiB <= --min-kib (or --min-mib) <= --max-mib");
    if (s.arena_bytes < 4 * s.max_mib * MiB)
        throw std::invalid_argument("--arena-gib must hold at least four inputs of --max-mib");
    return s;
}

const char* dtypeName(T::DataType dtype) {
    if (dtype == T::DataType::FP32) return "fp32";
    return dtype == T::DataType::FP8_E5M2 ? "fp8_e5m2" : "fp8_e4m3";
}
uint32_t elementBytes(T::DataType dtype) { return dtype == T::DataType::FP32 ? 4 : 1; }
uint64_t indexBytes(T::DataType dtype) { return dtype == T::DataType::UINT64 ? 8 : 4; }

uint64_t mix(uint64_t x) {  // splitmix64 finalizer: deterministic across platforms
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}
double unit(uint64_t h) { return static_cast<double>(h >> 11) * (1.0 / 9007199254740992.0); }

struct Geometry {
    uint64_t rows, r, target_mib;  // target_mib 0: a random geometry; r in elements of the cell's dtype (1179)
    bool random;
};

struct CellSpec {
    uint64_t id;
    Geometry geometry;
    bool heldout;
    Kind kind;
    T::DataType dtype;
    std::string operation;
};

bool heldOutRow(uint64_t r, uint64_t seed, uint32_t percent) {
    return mix(mix(seed ^ 0x48454C44ULL) ^ mix(r * 0x100000001B3ULL)) % 100 < percent;
}

std::vector<Geometry> geometries(const Settings& s) {
    const uint64_t max_bytes = s.max_mib * MiB, min_bytes = s.min_bytes;
    std::vector<Geometry> result;
    const std::vector<uint64_t> rows{2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 31, 32, 48, 64, 96, 100, 128, 255, 256, 512,
                                     1000, 1024, 2048, 4096, 8192, 16384, 65536, 262144, 1ULL << 20, 1ULL << 22,
                                     1ULL << 24, 1ULL << 26};
    for (uint64_t t : s.targets_mib) {
        const uint64_t bytes = t * MiB;
        if (bytes == 0 || bytes > max_bytes) continue;
        result.push_back({1, bytes, t, false});  // pure R
        for (uint64_t r : rows)
            if (2 * r <= bytes) result.push_back({bytes / r, r, t, false});
    }
    const double lo = std::log2(static_cast<double>(min_bytes)), hi = std::log2(static_cast<double>(max_bytes));
    for (uint64_t i = 0; i < s.random; ++i) {
        const uint64_t h = mix(s.seed * 0x2545F4914F6CDD1DULL + i);
        uint64_t r = static_cast<uint64_t>(std::llround(std::exp2(1.0 + 29.0 * unit(mix(h ^ 1)))));
        const uint64_t bytes = std::clamp<uint64_t>(
            static_cast<uint64_t>(std::llround(std::exp2(lo + (hi - lo) * unit(mix(h ^ 2))))), min_bytes, max_bytes);
        r = std::max<uint64_t>(r, 2);
        if (2 * r > bytes) result.push_back({1, bytes, 0, true});
        else result.push_back({bytes / r, r, 0, true});
    }
    std::sort(result.begin(), result.end(), [](const Geometry& a, const Geometry& b) {
        return std::tie(a.r, a.rows) < std::tie(b.r, b.rows);
    });
    result.erase(std::unique(result.begin(), result.end(), [](const Geometry& a, const Geometry& b) {
        return a.r == b.r && a.rows == b.rows;
    }), result.end());
    return result;
}

std::vector<CellSpec> cells(const Settings& s) {
    std::vector<std::string> value_ops, arg_ops;
    const auto wanted = [&](const char* name) {
        return s.ops.empty() || std::find(s.ops.begin(), s.ops.end(), name) != s.ops.end();
    };
    for (const ValueOp& op : VALUE_OPS)
        if (wanted(op.name)) value_ops.push_back(op.name);
    for (const char* op : ARG_OPS)
        if (wanted(op)) arg_ops.push_back(op);
    std::vector<CellSpec> result;
    uint64_t id = 0;
    for (const Geometry& g : geometries(s)) {
        const bool heldout = heldOutRow(g.r, s.seed, s.heldout_percent);
        for (uint32_t j = 0; j < s.semantics_per_geometry; ++j) {
            const uint64_t h = mix(mix(s.seed) ^ mix(g.rows * 0x9E3779B1ULL ^ g.r * 0x85EBCA77ULL) ^ j);
            CellSpec c{id, g, heldout, Kind::Value, s.dtypes[mix(h ^ 11) % s.dtypes.size()], ""};
            const bool fp32 = c.dtype == T::DataType::FP32;
            if (fp32) {
                // 1179: R elements of 4 bytes and a quarter of the rows (a pure R site keeps its bytes).
                if (g.rows == 1) {
                    c.geometry.r = std::max<uint64_t>(1, g.r / 4);
                } else {
                    c.geometry.rows = std::max<uint64_t>(1, g.rows / 4);
                }
            }
            const bool want_arg = (j % 2) == 1;
            if ((want_arg && !arg_ops.empty()) || value_ops.empty()) {
                if (arg_ops.empty()) continue;
                c.kind = (mix(h ^ 13) & 1) ? Kind::ArgValueIndex : Kind::ArgIndex;
                c.operation = arg_ops[mix(h ^ 17) % arg_ops.size()];
                if (s.carried_index && fp32) {
                    // 1217: a carried site; the values and index together fill the FP32 cell's working set.
                    c.kind = c.kind == Kind::ArgValueIndex ? Kind::CarriedArgValueIndex : Kind::CarriedArgIndex;
                    const uint64_t element = 4 + indexBytes(*s.carried_index);
                    if (c.geometry.rows == 1) c.geometry.r = std::max<uint64_t>(1, c.geometry.r * 4 / element);
                    else c.geometry.rows = std::max<uint64_t>(1, c.geometry.rows * 4 / element);
                }
            } else {
                c.operation = value_ops[mix(h ^ 19) % value_ops.size()];
            }
            result.push_back(std::move(c));
            ++id;
        }
    }
    std::stable_sort(result.begin(), result.end(), [&](const CellSpec& a, const CellSpec& b) {
        return mix(s.seed ^ mix(a.id)) < mix(s.seed ^ mix(b.id));
    });
    return result;
}

// Launched threads and CTAs of a shape's main kernel (as fp8KRLaunchGeometry launches it).
uint64_t shapeThreads(const KR::Fp8KRSegmentShape& shape) {
    switch (shape.family()) {
        case KR::Fp8KRFamily::Pack: return ceilDiv(shape.rows, shape.rows_per_thread);
        case KR::Fp8KRFamily::Group: return shape.rows * shape.group_lanes;
        case KR::Fp8KRFamily::Split: return shape.segments() * KR::FP8_KR_THREADS;
        case KR::Fp8KRFamily::Block: return shape.rows * KR::FP8_KR_THREADS;
    }
    return 0;
}
uint64_t shapeBlocks(const KR::Fp8KRSegmentShape& shape) {
    if (shape.family() == KR::Fp8KRFamily::Split) return shape.segments();
    if (shape.family() == KR::Fp8KRFamily::Block) return shape.rows;
    return ceilDiv(shapeThreads(shape), KR::FP8_KR_THREADS);
}
const char* familyName(const KR::Fp8KRSegmentShape& shape) {
    switch (shape.family()) {
        case KR::Fp8KRFamily::Pack: return shape.vector ? "vector" : "pack";
        case KR::Fp8KRFamily::Group: return "group";
        case KR::Fp8KRFamily::Split: return "split";
        case KR::Fp8KRFamily::Block: return "block";
    }
    return "unknown";
}

struct Candidate {
    KR::Fp8KRSegmentShape shape;
    size_t list_position = 0;    // 0: not in the production list
    bool partials_only = false;  // 1155 --split-stages: the Split partial kernel alone (variant=partials)
};

std::vector<Candidate> candidates(uint64_t rows, uint64_t row_bytes, uint32_t element_bytes, bool arg, uint32_t sms,
                                  const Settings& s) {
    // 1158: production's pool (fp8KRCandidatePool: the heuristic list, then the legal shapes around it), so the
    // ranker trains on exactly the candidates it ranks; --list-only keeps the heuristic list.
    std::vector<Candidate> result;
    for (const KR::Fp8KRPoolEntry& entry : KR::retainedKRCandidatePool(rows, row_bytes, sms, element_bytes)) {
        if (s.list_only && entry.list_position == 0) continue;
        if (arg && element_bytes == 4 && !KR::fp32KRArgShapeLegal(entry.shape)) continue;
        result.push_back({entry.shape, entry.list_position});
    }
    if (s.split_stages) {
        const size_t count = result.size();
        for (size_t i = 0; i < count; ++i)
            if (result[i].shape.family() == KR::Fp8KRFamily::Split) result.push_back({result[i].shape, 0, true});
    }
    return result;
}

uint64_t outputFootprint(Kind kind, uint64_t rows) {
    return producesValue(kind) ? align256(4 * rows) + 4 * rows : 4 * rows;
}

// The Split scratch a shape needs at a cell of this kind (1217: carried shapes write wide partials).
uint64_t scratchBytes(const KR::Fp8KRSegmentShape& shape, Kind kind) {
    if (isCarried(kind)) return C::fp32KRCarriedArgScratchBytes(shape);
    return KR::fp8KRScratchBytes(shape, kind != Kind::Value);
}

struct Timed {
    Candidate candidate;
    bool abandoned = false;
    double probe_us = 0;
    uint64_t launches = 1;  // per sample, from the probe time
    std::vector<std::unique_ptr<TimingEvent>> starts, stops;  // one per pass
};

struct Cell {
    CellSpec spec;
    std::vector<T::Tensor> inputs;
    std::vector<T::Tensor> carried;  // 1217: one carried index view per input view (carried cells)
    std::vector<std::optional<T::Tensor>> values, indices;
    std::optional<C::CubReductionStageSemantics> semantics;
    bool maximum = false;
    size_t next_input = 0, next_output = 0;
    std::vector<Timed> candidates;
    std::vector<size_t> timed;  // indices of candidates timed in the passes
    std::string production;
    double stamp_ms = 0;
    double reference_us = 0;
    KR::Fp8KRRankerSite ranker_site{};  // 1158: what production's ranker sees at this site
    double copy_us = 0;          // a copy moving the same bytes (interpolated from the copy curve)
    uint64_t moved_bytes = 0;    // input + output bytes of one launch
    uint64_t first_input = 0;    // ring offset of the first input view (the production stamp's input)
};

}  // namespace

int main(int argc, char** argv) {
    try {
        const Settings settings = parse(argc, argv);
        const T::TensorPlacement placement(T::TensorPlacement::MemDevices::GPU, 0);
        ScopedGpu gpu(0);
        Stream stream(placement);

        int l2_bytes = 0, sm = 0;
        check(cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, 0), "get L2 size");
        check(cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0), "get SM count");
        if (sm <= 0) throw std::runtime_error("No CUDA SMs");
        const uint32_t sms = static_cast<uint32_t>(sm);

        const std::vector<CellSpec> specs = cells(settings);
        const uint64_t total = std::min<uint64_t>(specs.size(), settings.max_cells);
        // Size the output ring and the shared Split scratch once for the cells that will run.
        uint64_t max_output = 0, max_scratch = 16;
        for (uint64_t n = 0; n < total; ++n) {
            const CellSpec& spec = specs[n];
            max_output = std::max(max_output, outputFootprint(spec.kind, spec.geometry.rows));
            for (const Candidate& c : candidates(spec.geometry.rows, spec.geometry.r * elementBytes(spec.dtype),
                                                 elementBytes(spec.dtype), spec.kind != Kind::Value, sms, settings))
                max_scratch = std::max(max_scratch, scratchBytes(c.shape, spec.kind));
        }
        const uint64_t output_ring_bytes = std::max(settings.output_ring_bytes, Ring::span(max_output) + 256);
        Ring inputs(placement, settings.arena_bytes);
        Ring outputs(placement, output_ring_bytes);
        T::Tensor scratch(placement, T::TensorDescriptor(T::DataType::UINT8, {max_scratch}));
        check(cudaMemsetAsync(inputs.storage().getMemPtr<void>(), 0x30, settings.arena_bytes, stream.getStream()),
              "input ring initialization");
        check(cudaMemsetAsync(outputs.storage().getMemPtr<void>(), 0, output_ring_bytes, stream.getStream()),
              "output ring initialization");
        stream.synchronize();
        const uint64_t copy_bytes = 256 * MiB;
        const double bandwidth = copyBandwidth(inputs, outputs, stream, copy_bytes);
        uint64_t max_moved = 4 * MiB;
        const uint64_t carried_element_bytes = settings.carried_index ? indexBytes(*settings.carried_index) : 0;
        for (uint64_t n = 0; n < total; ++n) {
            const uint64_t element =
                elementBytes(specs[n].dtype) + (isCarried(specs[n].kind) ? carried_element_bytes : 0);
            max_moved = std::max(max_moved, specs[n].geometry.rows * (specs[n].geometry.r * element +
                                                                      (producesValue(specs[n].kind) ? 8 : 4)));
        }
        const CopyCurve curve = copyCurve(inputs, outputs, stream, max_moved, settings.interval_us);

        std::cout << "# fp8_kr_training=1 version=1217 method=ring_batched seed=" << settings.seed
                  << " random_geometries=" << settings.random
                  << " semantics_per_geometry=" << settings.semantics_per_geometry
                  << " heldout_percent=" << settings.heldout_percent << " cells=" << total
                  << " cells_available=" << specs.size() << " samples=" << settings.samples
                  << " queue_cells=" << settings.queue_cells << " interval_us=" << settings.interval_us
                  << " max_launches=" << settings.max_launches << " arena_bytes=" << settings.arena_bytes
                  << " output_ring_bytes=" << output_ring_bytes << " input_reuse_bytes=" << INPUT_REUSE_BYTES
                  << " output_reuse_bytes=" << OUTPUT_REUSE_BYTES << " input_byte=0x30 explore="
                  << (settings.list_only ? 0 : 1) << " production=" << (settings.production ? 1 : 0)
                  << " pool_max_thread_bytes=" << KR::FP8_KR_POOL_MAX_THREAD_BYTES << " sm_count=" << sm
                  << " fp8_kr_ranker_training=1 feature_count=" << KR::FP8_KR_RANKER_FEATURES
                  << " ranker_trained=" << (KR::fp8KRRankerTrained() ? 1 : 0)
                  << " split_stages=" << (settings.split_stages ? 1 : 0)
                  << " l2_bytes=" << l2_bytes << " copy_bandwidth_tbps=" << std::fixed << std::setprecision(4)
                  << bandwidth / 1e6 << " floor_us=" << settings.floor_us
                  << " value_output=fp32 arg_index=uint32 arg_value=fp32 min_kib=" << settings.min_bytes / KiB
                  << " max_mib=" << settings.max_mib << " carried_index="
                  << (!settings.carried_index ? "none"
                      : *settings.carried_index == T::DataType::UINT64 ? "uint64" : "uint32")
                  << " frozen_margin=1 targets_mib=";
        for (size_t i = 0; i < settings.targets_mib.size(); ++i) std::cout << (i ? "," : "") << settings.targets_mib[i];
        std::cout << " copy_curve_moved_mib_us=";
        for (size_t i = 0; i < curve.points.size(); ++i)
            std::cout << (i ? "," : "") << std::setprecision(0) << curve.points[i].first / double(MiB) << ':'
                      << std::setprecision(3) << curve.points[i].second;
        std::cout << "\nkind,cell_id,heldout,random_geometry,K,R,input_bytes,target_mib,dtype,operation,shape,family,"
                     "variant,rows_per_thread,group_lanes,chunk_bytes,chunks,vector,blocks,threads,in_list,"
                     "list_position,pool_position,"
                     "production,is_production,stamp_ms,scratch_bytes,launches_per_sample,input_views,output_views,"
                     "abandoned,probe_us,median_us,min_us,reference_us,copy_us,moved_tbps,read_tbps";
        for (int i = 0; i < settings.samples; ++i) std::cout << ",sample" << i << "_us";
        for (size_t i = 0; i < KR::FP8_KR_RANKER_FEATURES; ++i) std::cout << ",f" << i;
        std::cout << ",frozen_margin\n";

        const int passes = settings.samples;
        B::Fp8OracleBatchQueue queue(settings.queue_cells, passes, [&] { stream.synchronize(); });
        const auto wall_start = std::chrono::steady_clock::now();
        uint64_t prepared = 0, collected = 0, timed_candidates = 0, abandoned_candidates = 0, unsupported = 0;
        void* const scratch_ptr = scratch.getMemPtr<void>();

        const auto launch = [&](Cell& cell, size_t index) {
            const KR::Fp8KRSegmentShape& shape = cell.candidates[index].candidate.shape;
            const KR::Fp8KRSplitStages stages = cell.candidates[index].candidate.partials_only
                ? KR::Fp8KRSplitStages::PartialsOnly : KR::Fp8KRSplitStages::Both;
            const bool is_value = cell.spec.kind == Kind::Value;
            const uint64_t scratch_bytes = scratchBytes(shape, cell.spec.kind);
            void* partials = scratch_bytes != 0 ? scratch_ptr : nullptr;
            const size_t view = cell.next_input++ % cell.inputs.size();
            const T::Tensor& in = cell.inputs[view];
            const size_t out = cell.next_output++ % cell.values.size();
            if (isCarried(cell.spec.kind)) {
                C::launchFp32KRCarriedArgSegments(in, cell.carried[view],
                                                  cell.values[out] ? &*cell.values[out] : nullptr,
                                                  &*cell.indices[out], shape, partials, scratch_bytes, cell.maximum, 1,
                                                  stream, stages);
            } else if (is_value && cell.spec.dtype == T::DataType::FP32) {
                C::launchFp32KRValueSegments(*cell.semantics, in, *cell.values[out], shape, partials, scratch_bytes,
                                             1.0f, stream, stages);
            } else if (is_value) {
                C::launchFp8KRValueSegments(*cell.semantics, in, *cell.values[out], shape, partials, scratch_bytes,
                                            1.0f, stream, stages);
            } else if (cell.spec.dtype == T::DataType::FP32) {
                C::launchFp32KRArgSegments(in, cell.values[out] ? &*cell.values[out] : nullptr, &*cell.indices[out],
                                           shape, partials, scratch_bytes, cell.maximum, 1, stream, stages);
            } else {
                C::launchFp8KRArgSegments(in, cell.values[out] ? &*cell.values[out] : nullptr, &*cell.indices[out],
                                          shape, partials, scratch_bytes, cell.maximum, 1, stream, stages);
            }
        };

        const auto print = [&](const Cell& cell) {
            const CellSpec& spec = cell.spec;
            const uint64_t rows = spec.geometry.rows, r = spec.geometry.r;
            const uint64_t bytes =
                rows * r * (elementBytes(spec.dtype) + (isCarried(spec.kind) ? carried_element_bytes : 0));
            const double moved = static_cast<double>(cell.moved_bytes);
            for (size_t index = 0; index < cell.candidates.size(); ++index) {
                const Timed& t = cell.candidates[index];
                const KR::Fp8KRSegmentShape& shape = t.candidate.shape;
                std::vector<double> samples;
                if (!t.abandoned) {
                    for (int p = 0; p < passes; ++p) {
                        float ms = 0;
                        check(cudaEventElapsedTime(&ms, t.starts[p]->handle, t.stops[p]->handle), "event elapsed");
                        samples.push_back(1000.0 * ms / static_cast<double>(t.launches));
                    }
                }
                const std::vector<double> times = samples.empty() ? std::vector<double>{t.probe_us} : samples;
                const double med = median(times);
                const double fastest = *std::min_element(times.begin(), times.end());
                const std::string name = KR::fp8KRShapeName(shape);
                std::cout << kindName(spec.kind) << ',' << spec.id << ',' << (spec.heldout ? 1 : 0) << ','
                          << (spec.geometry.random ? 1 : 0) << ',' << rows << ',' << r << ',' << bytes << ','
                          << spec.geometry.target_mib << ',' << dtypeName(spec.dtype) << ',' << spec.operation << ','
                          << name << ',' << familyName(shape) << ','
                          << (t.candidate.partials_only ? "partials" : "full") << ',' << shape.rows_per_thread << ','
                          << shape.group_lanes << ','
                          << (shape.family() == KR::Fp8KRFamily::Split ? shape.chunk_bytes : 0) << ','
                          << shape.chunks() << ',' << (shape.vector ? 1 : 0) << ',' << shapeBlocks(shape) << ','
                          << shapeThreads(shape) << ',' << (t.candidate.list_position != 0 ? 1 : 0) << ','
                          << t.candidate.list_position << ',' << (t.candidate.partials_only ? 0 : index + 1) << ','
                          << cell.production << ','
                          << (!cell.production.empty() && cell.production == name ? 1 : 0) << ','
                          << std::setprecision(1) << cell.stamp_ms << ','
                          << scratchBytes(shape, spec.kind) << ',' << t.launches << ','
                          << cell.inputs.size() << ',' << cell.values.size() << ',' << (t.abandoned ? 1 : 0) << ','
                          << std::setprecision(3) << t.probe_us << ',' << med << ',' << fastest << ','
                          << cell.reference_us << ',' << cell.copy_us << ',' << std::setprecision(4)
                          << moved / med / 1e6 << ',' << double(bytes) / med / 1e6 << std::setprecision(3);
                for (double s : samples) std::cout << ',' << s;
                for (size_t s = samples.size(); s < static_cast<size_t>(passes); ++s) std::cout << ',';
                // 1158: the ranker features production computes for this pool entry (9 significant digits: exact
                // float32 round trip).
                const auto features = KR::fp8KRRankerFeatures({shape, static_cast<uint32_t>(t.candidate.list_position)},
                                                              cell.ranker_site, sms);
                std::cout << std::defaultfloat << std::setprecision(9);
                for (float f : features) std::cout << ',' << f;
                // 1217: the installed ranker's margin (FP32's model for FP32 cells).
                const float frozen = spec.dtype == T::DataType::FP32 ? C::Fp32KR::Fp32KRRankerModel::margin(features)
                                                                     : KR::Fp8KRRankerModel::margin(features);
                std::cout << ',' << frozen << std::fixed << std::setprecision(3) << '\n';
                if (t.abandoned) ++abandoned_candidates;
                else ++timed_candidates;
            }
            ++collected;
            if (collected % 25 == 0 || collected == total) {
                const double wall =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count();
                std::cerr << "[fp8-kr-training] cells=" << collected << '/' << total
                          << " candidates=" << timed_candidates << " abandoned=" << abandoned_candidates
                          << " unsupported=" << unsupported << " wall_s=" << std::fixed << std::setprecision(1) << wall
                          << " eta_s=" << wall / double(collected) * double(total - collected) << std::endl;
            }
        };

        for (uint64_t n = 0; n < total; ++n) {
            auto cell = std::make_shared<Cell>();
            cell->spec = specs[n];
            const CellSpec& spec = cell->spec;
            const uint64_t rows = spec.geometry.rows, r = spec.geometry.r, bytes = rows * r * elementBytes(spec.dtype);
            const bool is_value = spec.kind == Kind::Value;
            const bool carried = isCarried(spec.kind);
            const uint64_t carried_bytes = carried ? rows * r * carried_element_bytes : 0;
            try {
                // Views: inputs spanning >= 2 GiB, outputs spanning >= 1 GiB (fewer when one alone is that big).
                const uint64_t footprint = outputFootprint(spec.kind, rows);
                const uint64_t index_offset = producesValue(spec.kind) ? align256(4 * rows) : 0;
                for (uint64_t v = 0, count = viewCount(bytes + carried_bytes, INPUT_REUSE_BYTES, inputs.bytes());
                     v < count; ++v) {
                    const uint64_t offset = inputs.take(bytes);
                    if (v == 0) cell->first_input = offset;
                    cell->inputs.push_back(inputs.view(spec.dtype, {rows, r}, offset));
                    // 1217: the view's carried index (any values; the ring's bytes), taken right after the values.
                    if (carried)
                        cell->carried.push_back(
                            inputs.view(*settings.carried_index, {rows, r}, inputs.take(carried_bytes)));
                }
                for (uint64_t v = 0, count = viewCount(footprint, OUTPUT_REUSE_BYTES, outputs.bytes()); v < count;
                     ++v) {
                    const uint64_t base = outputs.take(footprint);
                    cell->values.push_back(is_value || producesValue(spec.kind)
                        ? std::optional<T::Tensor>(outputs.view(T::DataType::FP32, {rows}, base)) : std::nullopt);
                    cell->indices.push_back(!is_value
                        ? std::optional<T::Tensor>(outputs.view(T::DataType::UINT32, {rows}, base + index_offset))
                        : std::nullopt);
                }
                if (is_value) {
                    for (const ValueOp& op : VALUE_OPS)
                        if (spec.operation == op.name)
                            cell->semantics = C::makeValueReductionStageSemantics(op.op, C::DensePassRole::Complete, r);
                    if (!cell->semantics || !C::fp8KRValueSupports(*cell->semantics))
                        throw std::logic_error("KR training VALUE operation has no retained KR semantics");
                }
                cell->maximum = spec.operation == "argmax";
                if (is_value) {
                    cell->ranker_site = C::fp8KRValueRankerSite(*cell->semantics, spec.dtype);
                } else {
                    cell->ranker_site.kind =
                        static_cast<uint8_t>((producesValue(spec.kind) ? 2 : 1) + (carried ? 2 : 0));
                    cell->ranker_site.combine = cell->maximum ? 3 : 2;
                    cell->ranker_site.e5m2 = spec.dtype == T::DataType::FP8_E5M2;
                }
                cell->moved_bytes = bytes + carried_bytes + (producesValue(spec.kind) ? 8 : 4) * rows;
                const double moved = static_cast<double>(cell->moved_bytes);
                cell->reference_us = std::max(settings.floor_us, moved / bandwidth);
                cell->copy_us = curve.us(moved);

                // Production pick: the stamp a user gets at this site (its own autotune; the stamp is then dropped).
                if (settings.production && !carried) {
                    const bool is_r = rows == 1;
                    const std::vector<uint64_t> dims = is_r ? std::vector<uint64_t>{r} : std::vector<uint64_t>{rows, r};
                    const T::Tensor site = inputs.view(spec.dtype, dims, cell->first_input);
                    const uint32_t axis = is_r ? 0 : 1;
                    const auto stamp_start = std::chrono::steady_clock::now();
                    std::string passes_name;
                    try {
                        if (is_value) {
                            T::CubReductionOp op = T::CubReductionOp::Sum;
                            for (const ValueOp& v : VALUE_OPS)
                                if (spec.operation == v.name) op = v.op;
                            auto stamped = T::CubReduction(op, axis, T::DataType::FP32).stamp(site, stream);
                            if (stamped->usesDenseExecutablePlan())
                                for (const auto& pass : stamped->getDensePhysicalPassDiagnostics())
                                    passes_name += (passes_name.empty() ? "" : "+") + pass.family;
                        } else {
                            T::CubArgReductionOutputOptions options;
                            options.produce_value = producesValue(spec.kind);
                            if (options.produce_value) options.value_output_dtype = T::DataType::FP32;
                            options.produce_index = true;
                            options.index_output_dtype = T::DataType::UINT32;
                            auto stamped = T::CubArgReduction(cell->maximum ? T::CubArgReductionOp::ArgMax
                                                                            : T::CubArgReductionOp::ArgMin,
                                                              axis, options).stamp(site, stream);
                            for (const std::string& pass : stamped->getPlannedPassNames())
                                passes_name += (passes_name.empty() ? "" : "+") + pass;
                        }
                    } catch (const std::exception& e) {
                        passes_name = std::string("error:") + e.what();  // reported, not fatal: the shapes still time
                    }
                    stream.synchronize();
                    cell->stamp_ms =
                        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - stamp_start)
                            .count();
                    // "retained_fp8_kr:v8" / "fp8_arg_kr:v8" -> "v8"; anything else verbatim.
                    const size_t colon = passes_name.rfind(':');
                    cell->production = passes_name.find('+') == std::string::npos && colon != std::string::npos
                        ? passes_name.substr(colon + 1) : passes_name;
                    std::replace(cell->production.begin(), cell->production.end(), ',', ';');
                }

                // Probe: an untimed launch of every candidate (loads every kernel before anything is timed), then a
                // timed one, to abandon extra shapes far off line rate.
                for (const Candidate& c : candidates(rows, r * elementBytes(spec.dtype), elementBytes(spec.dtype),
                                                     !is_value, sms, settings))
                    cell->candidates.push_back({c, false, 0, 1, {}, {}});
                if (cell->candidates.empty()) throw std::logic_error("no retained KR shape");
                std::vector<std::unique_ptr<TimingEvent>> probe;
                for (size_t i = 0; i < cell->candidates.size(); ++i) {
                    probe.push_back(std::make_unique<TimingEvent>());
                    probe.push_back(std::make_unique<TimingEvent>());
                    launch(*cell, i);
                    check(cudaEventRecord(probe[2 * i]->handle, stream.getStream()), "probe start");
                    launch(*cell, i);
                    check(cudaEventRecord(probe[2 * i + 1]->handle, stream.getStream()), "probe stop");
                }
                stream.synchronize();
                for (size_t i = 0; i < cell->candidates.size(); ++i) {
                    Timed& t = cell->candidates[i];
                    float ms = 0;
                    check(cudaEventElapsedTime(&ms, probe[2 * i]->handle, probe[2 * i + 1]->handle), "probe elapsed");
                    t.probe_us = 1000.0 * ms;
                    t.abandoned = t.candidate.list_position == 0 &&
                                  t.probe_us > std::max(1000.0, 20.0 * cell->reference_us);
                    if (t.abandoned) continue;
                    t.launches = B::ColdTiming::launchesPerSample(t.probe_us, settings.interval_us,
                                                                  settings.max_launches);
                    cell->timed.push_back(i);
                    for (int p = 0; p < passes; ++p) {
                        t.starts.push_back(std::make_unique<TimingEvent>());
                        t.stops.push_back(std::make_unique<TimingEvent>());
                    }
                }
            } catch (const std::exception& e) {
                ++unsupported;
                ++collected;
                std::cout << "# unsupported," << kindName(spec.kind) << ',' << spec.id << ',' << spec.operation << ','
                          << dtypeName(spec.dtype) << ",K=" << rows << ",R=" << r << ',' << e.what() << '\n';
                continue;
            }
            ++prepared;
            queue.add({[] {},
                       [cell, &launch, &stream](int pass) {
                           const size_t count = cell->timed.size();
                           for (size_t visit = 0; visit < count; ++visit) {
                               const size_t index = cell->timed[(visit + static_cast<size_t>(pass)) % count];
                               Timed& t = cell->candidates[index];
                               check(cudaEventRecord(t.starts[pass]->handle, stream.getStream()), "start event");
                               for (uint64_t l = 0; l < t.launches; ++l) launch(*cell, index);
                               check(cudaEventRecord(t.stops[pass]->handle, stream.getStream()), "stop event");
                           }
                       },
                       [cell, &print] { print(*cell); }});
        }
        queue.flush();
        const double bandwidth_end = copyBandwidth(inputs, outputs, stream, copy_bytes);
        const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count();
        const auto timing = queue.timing();
        std::cout << "# end cells=" << collected << " prepared=" << prepared << " unsupported=" << unsupported
                  << " candidates=" << timed_candidates << " abandoned=" << abandoned_candidates
                  << " batches=" << queue.batches() << " copy_bandwidth_tbps_end=" << std::setprecision(4)
                  << bandwidth_end / 1e6 << " input_ring_wraps=" << inputs.wraps()
                  << " output_ring_wraps=" << outputs.wraps() << " wall_s=" << std::setprecision(1) << wall
                  << " submit_s=" << timing.submit_ms / 1e3 << " wait_s=" << timing.wait_ms / 1e3 << '\n';
        std::cout.flush();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "thor_fp8_kr_training_sweep: " << e.what() << '\n';
        return 1;
    }
}
