// 1335: thor_view_reduction_audit. What a user gets from a production CubReduction stamp of a non-dense view, the
// three view classes that ran CUB device reductions through 1364 (plans/Thor_View_Reduction_Plan_2026-11-08.md, phase
// 0; the baseline view-base-*.csv); 1373 deleted those CUB calls:
//
//   A  VIEW-DIRECT-1: a compact physical permutation (rank 2-4, every dim >= 2, a non-identity axis permutation of a
//      dense tensor), every axis reduced (CUB TransformReduce over the span through 1364; 1365: the retained dense
//      planner on the dense span of its bytes).
//   B  VIEW-DIRECT-2A: a compact permutation whose reduced axes are the physical suffix and whose retained axes keep
//      their physical order (logical [O..., permuted R...] over physical [O..., R...]; CUB's fixed-size segmented
//      reduce through 1364; 1365: the retained dense planner).
//   C  a rank-1 affine view [n] with stride s (elements), full reduction (CUB Reduce over an affine iterator through
//      1365; 1366: StridedR, Utilities/TensorOperations/Cub/StridedReduction.h); s = 0 is a broadcast view of one
//      element.
//
// 1375, the phase 5 measurement (nothing is rerouted): two more compact permutations whose retained axes keep their
// physical order, so each is a dense tensor reduced over its physical axes.
//   D  permutation-aware tiled with natural output: physical [O..., R..., I...] with I non-empty (O may be absent), the
//      reduced axes at other logical positions. Production ran the Thor tiled view kernels through 1375; 1376 runs the
//      dense planner on the bytes. The reference is the dense [outer, R, inner] over the R axis ([R, inner] over axis
//      0 without O).
//   E  a split physical reduction: the reduced axes form two runs in physical order (R O R, O R O R, R O R O or
//      O R O R O). Production threw through 1375 (no tiled candidate); the cell then timed the dense stamp of the
//      physical tensor over its physical reduced axes (retained ComposedDense), with reference `view_unsupported`.
//      1376 runs the view on the dense planner, timed against that dense stamp like D.
// In D and E the logical order is never the physical one: the retained axes keep their order and the reduced axes are
// placed at random logical positions.
//
// 1386, the measurement of the three view classes that still ran the Thor tiled kernels (nothing was rerouted; 1387:
// the retained dense planner with a layout pass inside the reducer; 1390: the reducer refuses them and the cell times
// the two parts a caller runs, CubReduction::splitRefusedView: F / G the reduction of the bytes in physical order then
// the dense copy of its output read rotated, passes "split_rotated:<plan>+copy:<copy plan>"; H the dense copy of the
// view then the reduction of the copy, "split_packed:copy:<copy plan>+<plan>", or (1391, phase 7 A) the reduction of
// its rows read at their pitch then the copy of the output's columns, "split_sliced:<plan>+copy:<copy plan>"):
//   F  a compact permutation with a rotated output: physical [O..., R..., I...], O and I non-empty, logical retained
//      order [I..., O...] (permutation-aware tiled, PermutedInnerOuter: the direct kernel with a shared-memory
//      transpose epilogue, or the wide shared-transpose kernel).
//   G  VIEW-DIRECT-2B: physical [A..., R..., B..., P...], all non-empty, logical retained order [B..., A..., P...] (the
//      32x33 payload-transpose kernel). With P empty it would be F.
//   H  VIEW-PITCHED-TILED: logical = physical [O..., R..., I...] with I contiguous, rows R apart by a row pitch and
//      slabs O apart by a slab pitch, with gaps: between rows (a slice of the last axis), between slabs (a slice of
//      the middle axis), or both (repeated-label einsum diagonals). O may be absent (then the rows have gaps).
//      1392 (phase 7 C): with enough outputs the reducer reads H in place (passes "pitched_complete:w<W>_t<tile>"),
//      and every H cell also times the split when its main reads in place (split_us) and the in-place read at each
//      forced W (pitched_w<W>_us, CubReduction::stampPitched), each verified like the main; one whose probe is over
//      50x the main's and 1 ms is left empty.
// The reduced axes of F and G are placed at random logical positions; in H they are logically contiguous. The reference
// is the production stamp of a dense tensor: F the physical [outer, R, inner] over axis 1 (its output in physical
// order, so the reference skips the rotation); G the physical [A, R, B x P] over axis 1 (likewise); H a packed
// [outer, R, inner] over the R axis ([R, inner] over axis 0 without O), the bytes the view would be without its gaps.
//
// Each cell is timed cold against two references from the same run:
//   - the copy curve at the cell's touched bytes (ColdTiming.h, as every reduction audit);
//   - A / B / D / E: the production stamp of the dense tensor of the same bytes (A: [N] over axis 0; B: [outer, R] over
//     axis 1; D and E above), the retained kernels phase 1 routes A / B to; C (s >= 2): a gather reference kernel
//     (below), the bound of the access pattern.
//
// Touched bytes: A / B / D / E / F / G the input plus the FP32 outputs. H: the distinct 32-byte sectors its rows cover
// (counted exactly) plus the FP32 outputs. C: the bytes the memory system moves for the gather:
// n x 32 (one sector per element) when the pitch p = s x element bytes is at least 32, the span ((n - 1) x p + element
// bytes) below that, one element at s = 0; plus the FP32 output. useful_bytes is n x element bytes plus the output.
//
// Method (ColdTiming.h, as thor_dense_production_audit): the input ring holds 0.5 in the cell dtype in every element
// (a fill kernel), so every output has a closed form (count = its reduction size: SUM / L1 count / 2, MEAN / MIN / MAX
// 0.5, SumSquares count / 4, L2 sqrt(count / 4), PRODUCT 2^-count); each cell's first run is verified against it
// (exact below 2^24 elements per output, relative 1e-3 above; PRODUCT within 2^-126). The constant input checks counts
// and finalization, not addressing: the unit tests check addressing. Each cell is stamped once on its first view, then
// run with runOnTensorsForBenchmark (1335: view stamps take view replacements) on rotating input views spanning at
// least 2 GiB of touched bytes and output views spanning at least 1 GiB. C's wide pitches (p >= 64) rotate sub-pitch
// shifts of 32 bytes inside a region as well, so a large span holds many distinct sets of touched sectors. One verified
// run, one untimed run, one timed probe, then --samples samples of about --interval-us of back-to-back runs (at most
// --max-launches), interleaved across the cells of a batch; the reference the same way.
//
// Gather reference (C): one load per element with the cell's element width and stride, each lane of a warp reading
// consecutive logical elements, 8 independent loads in flight per thread, XOR folded into one word per CTA; a grid of
// 4 CTAs of 256 threads per SM, each CTA a contiguous range. It moves what any gather of the view moves and computes
// nothing, so a reduction at its time is at the access pattern's bound.
//
// Output: '#' header lines (settings, device, copy curve) and one CSV row per cell. benchmarks/view_reduction_audit.py
// reports it.
#include "benchmarks/ColdTiming.h"
#include "benchmarks/Fp8OracleBatchQueue.h"
#include "Utilities/TensorOperations/Cub/CubReduction.h"
#include "Utilities/TensorOperations/Cub/CubReductionTuningDepth.h"
#include "Utilities/TensorOperations/Copy/StridedCopy.h"
#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Exceptions.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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
namespace B = ThorImplementation::CubReductionBenchmarking;
using B::ColdTiming::align256;
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
constexpr int GatherThreads = 256;
constexpr int GatherUnroll = 8;
constexpr int GatherCtasPerSm = 4;

void check(cudaError_t status, const char* what) { B::ColdTiming::checkCuda(status, what); }

__global__ void fillWordsKernel(uint32_t* words, uint64_t count, uint32_t pattern) {
    for (uint64_t i = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x; i < count; i += uint64_t(gridDim.x) * blockDim.x)
        words[i] = pattern;
}

template <typename WordT>
__global__ __launch_bounds__(GatherThreads) void gatherReferenceKernel(const WordT* __restrict__ base, uint64_t n,
                                                                      uint64_t stride, uint32_t* __restrict__ out) {
    const uint64_t begin = n * blockIdx.x / gridDim.x, end = n * (blockIdx.x + 1) / gridDim.x;
    uint32_t acc[GatherUnroll] = {};
    uint64_t i = begin + threadIdx.x;
    for (; i + (GatherUnroll - 1) * GatherThreads < end; i += GatherUnroll * GatherThreads) {
#pragma unroll
        for (int u = 0; u < GatherUnroll; ++u) acc[u] ^= uint32_t(base[(i + u * GatherThreads) * stride]);
    }
    for (; i < end; i += GatherThreads) acc[0] ^= uint32_t(base[i * stride]);
    uint32_t x = 0;
#pragma unroll
    for (int u = 0; u < GatherUnroll; ++u) x ^= acc[u];
    for (int offset = 16; offset > 0; offset /= 2) x ^= __shfl_xor_sync(0xFFFFFFFFu, x, offset);
    __shared__ uint32_t warps[GatherThreads / 32];
    if (threadIdx.x % 32 == 0) warps[threadIdx.x / 32] = x;
    __syncthreads();
    if (threadIdx.x == 0) {
        uint32_t total = 0;
        for (int w = 0; w < GatherThreads / 32; ++w) total ^= warps[w];
        out[blockIdx.x] = total;
    }
}

struct OpSpec {
    const char* name;
    T::CubReductionOp op;
};
constexpr OpSpec OPS[] = {
    {"sum", T::CubReductionOp::Sum},           {"mean", T::CubReductionOp::Mean},
    {"product", T::CubReductionOp::Product},   {"min", T::CubReductionOp::Min},
    {"max", T::CubReductionOp::Max},           {"l1_norm", T::CubReductionOp::L1Norm},
    {"l2_norm", T::CubReductionOp::L2Norm},    {"sum_squares", T::CubReductionOp::SumSquares},
};

const OpSpec& opSpec(std::string_view name) {
    for (const OpSpec& op : OPS)
        if (name == op.name) return op;
    throw std::invalid_argument("--ops takes sum,mean,product,min,max,l1_norm,l2_norm,sum_squares");
}

struct DTypeSpec {
    const char* name;
    T::DataType dtype;
    uint64_t bytes;
    uint32_t half_pattern;  // 0.5 in every element of a 32-bit word
};
constexpr DTypeSpec DTYPES[] = {
    {"fp32", T::DataType::FP32, 4, 0x3F000000u},     {"fp16", T::DataType::FP16, 2, 0x38003800u},
    {"bf16", T::DataType::BF16, 2, 0x3F003F00u},     {"fp8_e4m3", T::DataType::FP8_E4M3, 1, 0x30303030u},
    {"fp8_e5m2", T::DataType::FP8_E5M2, 1, 0x38383838u},
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

struct Settings {
    const DTypeSpec* dtype = &DTYPES[0];
    std::vector<char> cases{'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'};  // 1386: f, g, h
    std::vector<std::string> ops{"sum", "mean", "product", "min", "max", "l1_norm", "l2_norm", "sum_squares"};
    std::vector<uint64_t> strides{0, 2, 3, 4, 8, 16, 33, 64, 4096};
    uint64_t cells_ab = 60;      // per case (A, B, D, E, F, G, H)
    uint64_t cells_c = 8;        // per stride
    uint64_t seed = 1335;
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
    T::CubReductionTuningDepth depth = T::CubReductionTuningDepth::Regular;
    bool list_only = false;
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
                if (c.size() != 1 || std::string_view("abcdefgh").find(c[0]) == std::string_view::npos)
                    throw std::invalid_argument("--cases takes a,b,c,d,e,f,g,h");
                s.cases.push_back(c[0]);
            }
        } else if (auto v = value("--ops=")) {
            s.ops = splitList(*v);
            for (const std::string& op : s.ops) (void)opSpec(op);
        } else if (auto v = value("--strides=")) {
            s.strides.clear();
            for (const std::string& x : splitList(*v)) {
                s.strides.push_back(std::stoull(x));
                if (s.strides.back() == 1) throw std::invalid_argument("--strides: stride 1 is a dense tensor");
            }
        } else if (auto v = value("--cells=")) {
            s.cells_ab = std::stoull(*v);
        } else if (auto v = value("--cells-per-stride=")) {
            s.cells_c = std::stoull(*v);
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
        } else if (auto v = value("--tuning-depth=")) {
            if (*v == "quick") s.depth = T::CubReductionTuningDepth::Quick;
            else if (*v == "regular") s.depth = T::CubReductionTuningDepth::Regular;
            else if (*v == "ultra") s.depth = T::CubReductionTuningDepth::Ultra;
            else throw std::invalid_argument("--tuning-depth takes quick, regular or ultra");
        } else if (a == "--list-only") {
            s.list_only = true;
        } else if (a == "--help") {
            std::cout << "thor_view_reduction_audit [--dtype=fp32] [--cases=a,b,c,d,e,f,g,h] [--ops=sum,...] "
                         "[--strides=0,2,3,4,8,16,33,64,4096]\n"
                         "    [--cells=60] [--cells-per-stride=8] [--seed=1335] [--min-kib=64] [--max-mib=256]\n"
                         "    [--max-span-mib=1024] [--max-cells=N] [--samples=5] [--interval-us=1000]\n"
                         "    [--max-launches=256] [--arena-gib=8] [--output-ring-gib=2] [--queue-cells=16]\n"
                         "    [--tuning-depth=regular] [--list-only]\n"
                         "--cells is per case (A, B, D-H); --cells-per-stride per C stride; ops are drawn per "
                         "cell.\n"
                         "CSV on stdout, progress on stderr.\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("Unknown argument " + std::string(a));
        }
    }
    if (s.cases.empty() || s.ops.empty()) throw std::invalid_argument("--cases and --ops must not be empty");
    if (s.min_bytes < 4 * KiB || s.min_bytes > s.max_mib * MiB)
        throw std::invalid_argument("need 4 KiB <= --min-kib <= --max-mib");
    if (s.arena_bytes < 4 * std::max(s.max_mib * MiB, s.max_span_bytes))
        throw std::invalid_argument("--arena-gib must hold four inputs of --max-mib and four spans of --max-span-mib");
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

// `count` dims of at least 2 whose product is near `target` (log-share weights from h).
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

std::vector<size_t> nonIdentityPermutation(size_t count, uint64_t h) {
    std::vector<size_t> p(count);
    std::iota(p.begin(), p.end(), 0);
    for (size_t i = count; i-- > 1;) std::swap(p[i], p[mix(h + i) % (i + 1)]);
    bool identity = true;
    for (size_t i = 0; i < count; ++i) identity = identity && p[i] == i;
    if (identity && count >= 2) std::swap(p[0], p[1]);
    return p;
}

std::vector<uint64_t> denseStrides(const std::vector<uint64_t>& dims) {
    std::vector<uint64_t> strides(dims.size(), 1);
    for (size_t i = dims.size(); i-- > 1;) strides[i - 1] = strides[i] * dims[i];
    return strides;
}

uint64_t product(const std::vector<uint64_t>& v) {
    uint64_t p = 1;
    for (uint64_t x : v) p *= x;
    return p;
}

std::string joined(const std::vector<uint64_t>& v) {
    std::string out;
    for (uint64_t x : v) out += (out.empty() ? "" : "x") + std::to_string(x);
    return out;
}

struct CellSpec {
    uint64_t id = 0;
    char kind = 'a';
    const OpSpec* op = nullptr;
    std::vector<uint64_t> dims, strides;  // the view
    std::vector<uint32_t> axes;
    uint64_t span_elements = 0;         // elements of storage the view addresses from its first
    uint64_t stride = 0;                // C
    uint64_t outputs = 1, count = 0;    // outputs and the reduction size of each
    std::vector<uint64_t> dense_dims;   // A / B / D-G: the dense tensor of the same bytes (H: packed, no gaps)
    std::vector<uint32_t> dense_axes;   // and its reduced axes
    uint64_t pitched_outer = 1, pitched_inner = 1;  // 1386 (H): the flattened extents
    uint64_t row_pitch = 0, slab_pitch = 0;         // and pitches, in elements
};

// 1375 (D, E): the view of the dense physical tensor `physical` whose `reduced` physical axes are reduced. The retained
// axes keep their physical order; the reduced axes, shuffled, go to random logical positions; the logical order is
// never the physical one.
void layOutPhysicalView(CellSpec& c, const std::vector<uint64_t>& physical, const std::vector<bool>& reduced,
                        uint64_t h) {
    const std::vector<uint64_t> physical_strides = denseStrides(physical);
    std::vector<size_t> logical, reduced_axes;  // logical: the physical axis of each logical axis
    for (size_t p = 0; p < physical.size(); ++p) (reduced[p] ? reduced_axes : logical).push_back(p);
    for (size_t i = reduced_axes.size(); i-- > 1;) std::swap(reduced_axes[i], reduced_axes[mix(h + i) % (i + 1)]);
    for (size_t k = 0; k < reduced_axes.size(); ++k)
        logical.insert(logical.begin() + static_cast<std::ptrdiff_t>(mix(h + 101 + k) % (logical.size() + 1)),
                       reduced_axes[k]);
    bool identity = true;
    for (size_t i = 0; i < logical.size(); ++i) identity = identity && logical[i] == i;
    if (identity) {  // swap the first reduced axis with a neighbour: the retained axes keep their order
        size_t i = 0;
        while (!reduced[logical[i]]) ++i;
        std::swap(logical[i], logical[i + 1 < logical.size() ? i + 1 : i - 1]);
    }
    for (size_t i = 0; i < logical.size(); ++i) {
        c.dims.push_back(physical[logical[i]]);
        c.strides.push_back(physical_strides[logical[i]]);
        if (reduced[logical[i]]) c.axes.push_back(static_cast<uint32_t>(i));
    }
    c.span_elements = product(physical);
    c.outputs = c.count = 1;
    for (size_t p = 0; p < physical.size(); ++p) (reduced[p] ? c.count : c.outputs) *= physical[p];
}

// 1386 (F, G): the view of the dense physical tensor `physical` whose `reduced` physical axes are reduced, with the
// retained axes in the logical order `logical` (physical axes); the reduced axes, shuffled, go to random logical
// positions.
void layOutRetainedOrder(CellSpec& c, const std::vector<uint64_t>& physical, const std::vector<bool>& reduced,
                         std::vector<size_t> logical, uint64_t h) {
    const std::vector<uint64_t> physical_strides = denseStrides(physical);
    std::vector<size_t> reduced_axes;
    for (size_t p = 0; p < physical.size(); ++p)
        if (reduced[p]) reduced_axes.push_back(p);
    for (size_t i = reduced_axes.size(); i-- > 1;) std::swap(reduced_axes[i], reduced_axes[mix(h + i) % (i + 1)]);
    for (size_t k = 0; k < reduced_axes.size(); ++k)
        logical.insert(logical.begin() + static_cast<std::ptrdiff_t>(mix(h + 101 + k) % (logical.size() + 1)),
                       reduced_axes[k]);
    for (size_t i = 0; i < logical.size(); ++i) {
        c.dims.push_back(physical[logical[i]]);
        c.strides.push_back(physical_strides[logical[i]]);
        if (reduced[logical[i]]) c.axes.push_back(static_cast<uint32_t>(i));
    }
    c.span_elements = product(physical);
    c.outputs = c.count = 1;
    for (size_t p = 0; p < physical.size(); ++p) (reduced[p] ? c.count : c.outputs) *= physical[p];
}

// Physical axes [first, first + count) in order.
std::vector<size_t> axisRange(size_t first, size_t count) {
    std::vector<size_t> out(count);
    std::iota(out.begin(), out.end(), first);
    return out;
}

// 1375: 1 axis, or 2 when the run's extent allows (h decides).
std::vector<uint64_t> runDims(uint64_t target, uint64_t h) {
    return splitDims(target, (target >= 16 && h % 2) ? 2 : 1, mix(h + 1));
}

std::vector<CellSpec> cells(const Settings& s) {
    std::vector<CellSpec> out;
    const uint64_t b = s.dtype->bytes, max_bytes = s.max_mib * MiB;
    uint64_t id = 0;
    const auto drawOp = [&](uint64_t h) { return &opSpec(s.ops[h % s.ops.size()]); };
    for (char kind : s.cases) {
        if (kind == 'a') {
            for (uint64_t n = 0; n < s.cells_ab; ++n) {
                const uint64_t h = mix(s.seed * 1000003 + 1 * 100000 + n);
                CellSpec c;
                c.id = id++;
                c.kind = 'a';
                c.op = drawOp(mix(h + 1));
                const uint64_t elements = logUniform(mix(h + 2), s.min_bytes, max_bytes) / b;
                const std::vector<uint64_t> physical = splitDims(elements, 2 + mix(h + 3) % 3, mix(h + 4));
                const std::vector<uint64_t> physical_strides = denseStrides(physical);
                const std::vector<size_t> perm = nonIdentityPermutation(physical.size(), mix(h + 5));
                for (size_t i = 0; i < physical.size(); ++i) {
                    c.dims.push_back(physical[perm[i]]);
                    c.strides.push_back(physical_strides[perm[i]]);
                    c.axes.push_back(static_cast<uint32_t>(i));
                }
                c.span_elements = c.count = product(physical);
                c.dense_dims = {c.count};
                c.dense_axes = {0};
                out.push_back(c);
            }
        } else if (kind == 'b') {
            for (uint64_t n = 0; n < s.cells_ab; ++n) {
                const uint64_t h = mix(s.seed * 1000003 + 2 * 100000 + n);
                CellSpec c;
                c.id = id++;
                c.kind = 'b';
                c.op = drawOp(mix(h + 1));
                const uint64_t elements = logUniform(mix(h + 2), s.min_bytes, max_bytes) / b;
                const uint64_t r_target = logUniform(mix(h + 3), 4, std::max<uint64_t>(4, elements / 2));
                const uint64_t outer_target = std::max<uint64_t>(2, elements / r_target);
                const size_t reduced = (r_target >= 16 && mix(h + 4) % 2) ? 3 : 2;
                const size_t retained = (outer_target >= 4 && mix(h + 5) % 2) ? 2 : 1;
                std::vector<uint64_t> physical = splitDims(outer_target, retained, mix(h + 6));
                const std::vector<uint64_t> r_dims = splitDims(r_target, reduced, mix(h + 7));
                physical.insert(physical.end(), r_dims.begin(), r_dims.end());
                const std::vector<uint64_t> physical_strides = denseStrides(physical);
                const std::vector<size_t> perm = nonIdentityPermutation(reduced, mix(h + 8));
                for (size_t i = 0; i < retained; ++i) {
                    c.dims.push_back(physical[i]);
                    c.strides.push_back(physical_strides[i]);
                }
                for (size_t i = 0; i < reduced; ++i) {
                    c.dims.push_back(physical[retained + perm[i]]);
                    c.strides.push_back(physical_strides[retained + perm[i]]);
                    c.axes.push_back(static_cast<uint32_t>(retained + i));
                }
                c.outputs = product(std::vector<uint64_t>(physical.begin(), physical.begin() + retained));
                c.count = product(r_dims);
                c.span_elements = c.outputs * c.count;
                c.dense_dims = {c.outputs, c.count};
                c.dense_axes = {1};
                out.push_back(c);
            }
        } else if (kind == 'd') {
            for (uint64_t n = 0; n < s.cells_ab; ++n) {
                const uint64_t h = mix(s.seed * 1000003 + 4 * 100000 + n);
                CellSpec c;
                c.id = id++;
                c.kind = 'd';
                c.op = drawOp(mix(h + 1));
                const uint64_t elements = logUniform(mix(h + 2), s.min_bytes, max_bytes) / b;
                const uint64_t r_target = logUniform(mix(h + 3), 4, std::max<uint64_t>(4, elements / 4));
                const uint64_t rest = std::max<uint64_t>(4, elements / r_target);  // outer x inner
                const bool has_outer = rest >= 8 && mix(h + 4) % 3 != 0;
                const uint64_t inner_target = has_outer ? logUniform(mix(h + 5), 2, rest / 2) : rest;
                const uint64_t outer_target = has_outer ? std::max<uint64_t>(2, rest / inner_target) : 1;
                std::vector<uint64_t> physical;
                if (has_outer) physical = runDims(outer_target, mix(h + 6));
                const size_t outer_axes = physical.size();
                const std::vector<uint64_t> r_dims = runDims(r_target, mix(h + 7));
                const std::vector<uint64_t> i_dims = runDims(inner_target, mix(h + 8));
                physical.insert(physical.end(), r_dims.begin(), r_dims.end());
                physical.insert(physical.end(), i_dims.begin(), i_dims.end());
                std::vector<bool> reduced(physical.size(), false);
                for (size_t k = 0; k < r_dims.size(); ++k) reduced[outer_axes + k] = true;
                layOutPhysicalView(c, physical, reduced, mix(h + 9));
                const uint64_t outer = product(std::vector<uint64_t>(physical.begin(), physical.begin() + outer_axes));
                const uint64_t inner = product(i_dims);
                if (outer > 1) {
                    c.dense_dims = {outer, c.count, inner};
                    c.dense_axes = {1};
                } else {
                    c.dense_dims = {c.count, inner};
                    c.dense_axes = {0};
                }
                out.push_back(c);
            }
        } else if (kind == 'e') {
            static constexpr const char* PATTERNS[] = {"ROR", "OROR", "RORO", "ORORO"};
            for (uint64_t n = 0; n < s.cells_ab; ++n) {
                const uint64_t h = mix(s.seed * 1000003 + 5 * 100000 + n);
                CellSpec c;
                c.id = id++;
                c.kind = 'e';
                c.op = drawOp(mix(h + 1));
                const uint64_t elements = logUniform(mix(h + 2), s.min_bytes, max_bytes) / b;
                const std::string_view pattern = PATTERNS[mix(h + 3) % std::size(PATTERNS)];
                const size_t r_runs = static_cast<size_t>(std::count(pattern.begin(), pattern.end(), 'R'));
                const size_t o_runs = pattern.size() - r_runs;
                const uint64_t r_target = logUniform(mix(h + 4), 4, std::max<uint64_t>(4, elements / 2));
                const uint64_t o_target = std::max<uint64_t>(2, elements / r_target);
                const std::vector<uint64_t> r_extents = splitDims(r_target, r_runs, mix(h + 5));
                const std::vector<uint64_t> o_extents = splitDims(o_target, o_runs, mix(h + 6));
                std::vector<uint64_t> physical;
                std::vector<bool> reduced;
                size_t next_r = 0, next_o = 0;
                for (size_t k = 0; k < pattern.size(); ++k) {
                    const bool is_r = pattern[k] == 'R';
                    const std::vector<uint64_t> run =
                        runDims(is_r ? r_extents[next_r++] : o_extents[next_o++], mix(h + 10 + k));
                    physical.insert(physical.end(), run.begin(), run.end());
                    reduced.insert(reduced.end(), run.size(), is_r);
                }
                layOutPhysicalView(c, physical, reduced, mix(h + 20));
                c.dense_dims = physical;
                for (size_t p = 0; p < physical.size(); ++p)
                    if (reduced[p]) c.dense_axes.push_back(static_cast<uint32_t>(p));
                out.push_back(c);
            }
        } else if (kind == 'f') {
            for (uint64_t n = 0; n < s.cells_ab; ++n) {
                const uint64_t h = mix(s.seed * 1000003 + 6 * 100000 + n);
                CellSpec c;
                c.id = id++;
                c.kind = 'f';
                c.op = drawOp(mix(h + 1));
                const uint64_t elements = logUniform(mix(h + 2), s.min_bytes, max_bytes) / b;
                const uint64_t r_target = logUniform(mix(h + 3), 4, std::max<uint64_t>(4, elements / 4));
                const uint64_t rest = std::max<uint64_t>(4, elements / r_target);  // outer x inner
                const uint64_t outer_target = logUniform(mix(h + 4), 2, std::max<uint64_t>(2, rest / 2));
                const uint64_t inner_target = std::max<uint64_t>(2, rest / outer_target);
                const std::vector<uint64_t> o_dims = runDims(outer_target, mix(h + 6));
                const std::vector<uint64_t> r_dims = runDims(r_target, mix(h + 7));
                const std::vector<uint64_t> i_dims = runDims(inner_target, mix(h + 8));
                std::vector<uint64_t> physical = o_dims;
                physical.insert(physical.end(), r_dims.begin(), r_dims.end());
                physical.insert(physical.end(), i_dims.begin(), i_dims.end());
                std::vector<bool> reduced(physical.size(), false);
                for (size_t k = 0; k < r_dims.size(); ++k) reduced[o_dims.size() + k] = true;
                std::vector<size_t> retained = axisRange(o_dims.size() + r_dims.size(), i_dims.size());  // [I, O]
                const std::vector<size_t> o_axes = axisRange(0, o_dims.size());
                retained.insert(retained.end(), o_axes.begin(), o_axes.end());
                layOutRetainedOrder(c, physical, reduced, retained, mix(h + 9));
                c.dense_dims = {product(o_dims), c.count, product(i_dims)};
                c.dense_axes = {1};
                out.push_back(c);
            }
        } else if (kind == 'g') {
            for (uint64_t n = 0; n < s.cells_ab; ++n) {
                const uint64_t h = mix(s.seed * 1000003 + 7 * 100000 + n);
                CellSpec c;
                c.id = id++;
                c.kind = 'g';
                c.op = drawOp(mix(h + 1));
                const uint64_t elements = logUniform(mix(h + 2), s.min_bytes, max_bytes) / b;
                const uint64_t r_target = logUniform(mix(h + 3), 4, std::max<uint64_t>(4, elements / 8));
                const uint64_t rest = std::max<uint64_t>(8, elements / r_target);  // A x B x P
                const uint64_t a_target = logUniform(mix(h + 4), 2, std::max<uint64_t>(2, rest / 4));
                const uint64_t rest_bp = std::max<uint64_t>(4, rest / a_target);
                const uint64_t p_target = logUniform(mix(h + 5), 2, std::max<uint64_t>(2, rest_bp / 2));
                const uint64_t b_target = std::max<uint64_t>(2, rest_bp / p_target);
                const std::vector<uint64_t> a_dims = runDims(a_target, mix(h + 6));
                const std::vector<uint64_t> r_dims = runDims(r_target, mix(h + 7));
                const std::vector<uint64_t> b_dims = runDims(b_target, mix(h + 8));
                const std::vector<uint64_t> p_dims = runDims(p_target, mix(h + 9));
                std::vector<uint64_t> physical = a_dims;
                for (const std::vector<uint64_t>* run : {&r_dims, &b_dims, &p_dims})
                    physical.insert(physical.end(), run->begin(), run->end());
                std::vector<bool> reduced(physical.size(), false);
                for (size_t k = 0; k < r_dims.size(); ++k) reduced[a_dims.size() + k] = true;
                const size_t b_first = a_dims.size() + r_dims.size(), p_first = b_first + b_dims.size();
                std::vector<size_t> retained = axisRange(b_first, b_dims.size());  // [B, A, P]
                const std::vector<size_t> a_axes = axisRange(0, a_dims.size());
                const std::vector<size_t> p_axes = axisRange(p_first, p_dims.size());
                retained.insert(retained.end(), a_axes.begin(), a_axes.end());
                retained.insert(retained.end(), p_axes.begin(), p_axes.end());
                layOutRetainedOrder(c, physical, reduced, retained, mix(h + 10));
                c.dense_dims = {product(a_dims), c.count, product(b_dims) * product(p_dims)};
                c.dense_axes = {1};
                out.push_back(c);
            }
        } else if (kind == 'h') {
            for (uint64_t n = 0; n < s.cells_ab; ++n) {
                const uint64_t h = mix(s.seed * 1000003 + 8 * 100000 + n);
                CellSpec c;
                c.id = id++;
                c.kind = 'h';
                c.op = drawOp(mix(h + 1));
                const uint64_t elements = logUniform(mix(h + 2), s.min_bytes, max_bytes) / b;  // useful elements
                const uint64_t r_target = logUniform(mix(h + 3), 4, std::max<uint64_t>(4, elements / 4));
                const uint64_t rest = std::max<uint64_t>(4, elements / r_target);  // outer x inner
                const bool has_outer = rest >= 8 && mix(h + 4) % 3 != 0;
                const uint64_t inner_target = has_outer ? logUniform(mix(h + 5), 2, rest / 2) : rest;
                const uint64_t outer_target = has_outer ? std::max<uint64_t>(2, rest / inner_target) : 1;
                std::vector<uint64_t> o_dims;
                if (has_outer) o_dims = runDims(outer_target, mix(h + 6));
                const std::vector<uint64_t> r_dims = runDims(r_target, mix(h + 7));
                const std::vector<uint64_t> i_dims = runDims(inner_target, mix(h + 8));
                const uint64_t outer = product(o_dims), rows = product(r_dims), inner = product(i_dims);
                // 0: gaps between rows; 1: between slabs; 2: both. Without O the rows have gaps.
                const uint64_t gaps = has_outer ? mix(h + 9) % 3 : 0;
                uint64_t row_gap = gaps != 1 ? logUniform(mix(h + 10), 1, std::max<uint64_t>(1, inner)) : 0;
                uint64_t slab_gap = gaps != 0 ? logUniform(mix(h + 11), 1, std::max<uint64_t>(1, rows * inner)) : 0;
                uint64_t row_pitch = 0, slab_pitch = 0, span = 0;
                for (;;) {  // keep the span within --max-span-mib
                    row_pitch = inner + row_gap;
                    slab_pitch = rows * row_pitch + slab_gap;
                    span = (outer - 1) * slab_pitch + (rows - 1) * row_pitch + inner;
                    if (span * b <= s.max_span_bytes || (row_gap <= 1 && slab_gap <= 1)) break;
                    row_gap = row_gap ? std::max<uint64_t>(1, row_gap / 2) : 0;
                    slab_gap = slab_gap ? std::max<uint64_t>(1, slab_gap / 2) : 0;
                }
                const std::vector<uint64_t> i_strides = denseStrides(i_dims);
                std::vector<uint64_t> r_strides = denseStrides(r_dims), o_strides = denseStrides(o_dims);
                for (uint64_t& x : r_strides) x *= row_pitch;
                for (uint64_t& x : o_strides) x *= slab_pitch;
                c.dims = o_dims;
                c.strides = o_strides;
                for (size_t k = 0; k < r_dims.size(); ++k) {
                    c.axes.push_back(static_cast<uint32_t>(c.dims.size()));
                    c.dims.push_back(r_dims[k]);
                    c.strides.push_back(r_strides[k]);
                }
                c.dims.insert(c.dims.end(), i_dims.begin(), i_dims.end());
                c.strides.insert(c.strides.end(), i_strides.begin(), i_strides.end());
                c.span_elements = span;
                c.outputs = outer * inner;
                c.count = rows;
                c.pitched_outer = outer;
                c.pitched_inner = inner;
                c.row_pitch = row_pitch;
                c.slab_pitch = slab_pitch;
                if (outer > 1) {
                    c.dense_dims = {outer, rows, inner};
                    c.dense_axes = {1};
                } else {
                    c.dense_dims = {rows, inner};
                    c.dense_axes = {0};
                }
                out.push_back(c);
            }
        } else {
            for (uint64_t stride : s.strides) {
                for (uint64_t n = 0; n < s.cells_c; ++n) {
                    const uint64_t h = mix(s.seed * 1000003 + 3 * 100000 + stride * 1000 + n);
                    CellSpec c;
                    c.id = id++;
                    c.kind = 'c';
                    c.op = drawOp(mix(h + 1));
                    c.stride = stride;
                    const uint64_t target = logUniform(mix(h + 2), s.min_bytes, max_bytes);
                    uint64_t count;
                    if (stride == 0) {
                        count = std::max<uint64_t>(2, target / b);  // logical bytes
                    } else {
                        const uint64_t pitch = stride * b;
                        count = std::max<uint64_t>(2, target / std::min<uint64_t>(pitch, 32));
                        count = std::clamp<uint64_t>(count, 2, std::max<uint64_t>(2, s.max_span_bytes / pitch));
                    }
                    c.dims = {count};
                    c.strides = {stride};
                    c.axes = {0};
                    c.count = count;
                    c.span_elements = stride == 0 ? 1 : (count - 1) * stride + 1;
                    out.push_back(c);
                }
            }
        }
    }
    return out;
}

// 1386 (H): the distinct 32-byte sectors the rows cover (the views start 256-byte aligned). Rows, and slabs, are in
// address order and do not overlap, so a row adds the sectors past the previous row's last one, and a slab shares at
// most its first sector with the slab before. What a row (slab) adds depends only on its index mod 32: 32 rows (slabs)
// on, every address has moved by a multiple of 32 bytes. So each class is counted once and weighted by its size.
uint64_t classSize(uint64_t n, uint64_t c) { return c < n ? (n - 1 - c) / 32 + 1 : 0; }  // k in [0, n), k = c mod 32

double pitchedTouchedBytes(const CellSpec& c, uint64_t b) {
    const uint64_t row = c.row_pitch * b, slab = c.slab_pitch * b, len = c.pitched_inner * b;
    const uint64_t rows = c.count, slabs = c.pitched_outer;
    const auto slabSectors = [&](uint64_t x) {  // one slab whose first row starts x bytes into a sector
        const auto first = [&](uint64_t r) { return (x + r * row) / 32; };
        const auto last = [&](uint64_t r) { return (x + r * row + len - 1) / 32; };
        double total = double(last(0) - first(0) + 1);
        for (uint64_t r = 1; r <= std::min<uint64_t>(rows - 1, 32); ++r) {
            const uint64_t from = std::max(first(r), last(r - 1) + 1);
            if (last(r) >= from) total += double(last(r) - from + 1) * double(classSize(rows, r));
        }
        return total;
    };
    double sectors = 0;
    for (uint64_t o = 0; o < std::min<uint64_t>(slabs, 32); ++o)
        sectors += slabSectors((o * slab) % 32) * double(classSize(slabs, o));
    for (uint64_t o = 1; o <= std::min<uint64_t>(slabs - 1, 32); ++o) {
        const bool shared = (o * slab) / 32 == ((o - 1) * slab + (rows - 1) * row + len - 1) / 32;
        if (shared) sectors -= double(classSize(slabs, o));
    }
    return 32.0 * sectors;
}

double touchedInputBytes(const CellSpec& c, uint64_t b) {
    if (c.kind == 'h') return pitchedTouchedBytes(c, b);
    if (c.kind != 'c') return double(c.span_elements * b);
    if (c.stride == 0) return double(b);
    const uint64_t pitch = c.stride * b;
    return pitch >= 32 ? double(c.count) * 32.0 : double(c.span_elements * b);
}

// The closed form of a reduction of `count` elements of 0.5, and whether `got` matches it.
bool verifyValue(T::CubReductionOp op, uint64_t count, float got, double* rel_err) {
    const double n = double(count);
    double expected = 0;
    switch (op) {
        case T::CubReductionOp::Sum:
        case T::CubReductionOp::L1Norm: expected = 0.5 * n; break;
        case T::CubReductionOp::Mean:
        case T::CubReductionOp::Min:
        case T::CubReductionOp::Max: expected = 0.5; break;
        case T::CubReductionOp::SumSquares: expected = 0.25 * n; break;
        case T::CubReductionOp::L2Norm: expected = std::sqrt(0.25 * n); break;
        case T::CubReductionOp::Product: expected = count <= 149 ? std::ldexp(1.0, -int(count)) : 0.0; break;
    }
    const double err = std::abs(double(got) - expected);
    *rel_err = expected != 0 ? err / std::abs(expected) : err;
    if (op == T::CubReductionOp::Product) return err <= std::ldexp(1.0, -126) || *rel_err <= 1e-6;
    if (op == T::CubReductionOp::L2Norm) return *rel_err <= (count < (1u << 24) ? 1e-6 : 1e-3);
    return count < (1u << 24) ? err == 0 : *rel_err <= 1e-3;
}

const char* pathName(T::CubReductionPath path) {
    switch (path) {
        case T::CubReductionPath::DeviceTransformReduce: return "device_transform_reduce";
        case T::CubReductionPath::ContiguousFixedSegment: return "contiguous_fixed_segment";
        case T::CubReductionPath::TiledFixedSegment: return "tiled_fixed_segment";
        case T::CubReductionPath::OffsetSegmented: return "offset_segmented";
        case T::CubReductionPath::ComposedDense: return "composed_dense";
    }
    return "unknown";
}

std::string passNames(const T::StampedCubReduction& stamp) {
    std::string passes;
    if (stamp.usesDenseExecutablePlan()) {
        for (const auto& pass : stamp.getDensePhysicalPassDiagnostics())
            passes += (passes.empty() ? "" : "+") + pass.family;
    } else if (stamp.getStridedRLaunch().has_value()) {
        passes = stamp.getModernRKStrategyChain();  // 1366: "strided_r:closed_form" or ":split_g<G>" (1371)
    } else if (stamp.getPitchedCompleteLaunch().has_value()) {
        passes = stamp.getModernRKStrategyChain();  // 1392: "pitched_complete:w<W>_t<tile>"
    } else {
        passes = std::string("physical:") + pathName(stamp.getPath());
    }
    std::replace(passes.begin(), passes.end(), ',', ';');
    return passes;
}

// One timed thing (the production stamp, or its reference): run_once advances its own views.
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

// How a view runs: `stamp` on the view, or (1390, F / G / H) the two parts of `split`. A rotated view (1391: or a
// sliced H view): `stamp` reduces the bytes into split_buffer, which the copy (from rotation_source) writes into the
// output rotated (or its columns). A packed view: the copy writes the view's rows into split_buffer, which `stamp`
// reduces.
struct ViewRunner {
    std::shared_ptr<T::StampedCubReduction> stamp;
    std::optional<T::CubReductionViewSplit> split;
    T::Tensor split_buffer;
    T::StridedSpan rotation_source;
    T::DenseCopyPlan split_copy;

    // The split's stamp on `view` (and its buffer); returns the output dimensions.
    std::vector<uint64_t> stampSplit(T::CubReductionOp op, const T::StridedSpan& view,
                                     const T::TensorPlacement& placement, Stream& stream) {
        const T::CubReductionViewSplit& sp = *split;
        if (sp.copiesOutput()) {  // 1390: F / G; 1391: sliced H
            stamp = T::CubReduction(op, sp.dense_axes, T::DataType::FP32).stamp(sp.reducedInput(view), stream);
            split_buffer = stamp->getOutputTensor();
            rotation_source = sp.copySource(split_buffer);
            return sp.copy_dimensions;
        }
        split_buffer = T::Tensor(placement, T::TensorDescriptor(view.getDataType(), sp.dense_dimensions));  // packed H
        split_copy = T::planCopyToDense(view, split_buffer);
        stamp = T::CubReduction(op, sp.dense_axes, T::DataType::FP32).stamp(T::StridedSpan(split_buffer), stream);
        return stamp->getOutputTensor().getDimensions();
    }
    // Once the output views exist: the copy of a split that copies the output.
    void planOutputCopy(const T::StridedSpan& output) {
        if (split.has_value() && split->copiesOutput()) split_copy = T::planCopyToDense(rotation_source, output);
    }
    [[nodiscard]] std::string passes() const {
        std::string names = passNames(*stamp);
        if (!split.has_value()) return names;
        const std::string copy = "copy:" + T::denseCopyPlanName(split_copy);
        switch (split->kind) {
            case T::CubReductionViewSplit::Kind::RotatedOutput: return "split_rotated:" + names + "+" + copy;
            case T::CubReductionViewSplit::Kind::SlicedOutput: return "split_sliced:" + names + "+" + copy;
            case T::CubReductionViewSplit::Kind::PackedInput: return "split_packed:" + copy + "+" + names;
        }
        return names;
    }
    // One run on a view and an output.
    void run(const T::StridedSpan& input, const T::StridedSpan& output, Stream& stream) const {
        if (!split.has_value()) {
            stamp->runOnTensorsForBenchmark(input, output, stream);
        } else if (split->copiesOutput()) {
            stamp->runOnTensorsForBenchmark(split->reducedInput(input), T::StridedSpan(split_buffer), stream);
            T::launchCopyToDense(split_copy, rotation_source, output, stream);
        } else {
            T::launchCopyToDense(split_copy, input, T::StridedSpan(split_buffer), stream);
            stamp->runOnTensorsForBenchmark(T::StridedSpan(split_buffer), output, stream);
        }
    }
};

// The split of `views` as Expression takes it (CubReduction::splitRefusedView of the span; `reducer_reads_pitched`
// false: the split an H view the reducer reads in place has otherwise): an H view read at its row pitch (sliced) only
// when every view in the ring has the last row's gap inside the ring.
std::optional<T::CubReductionViewSplit> ringSplit(const std::vector<T::StridedSpan>& views,
                                                  const std::vector<uint32_t>& axes,
                                                  bool reducer_reads_pitched) {
    std::optional<T::CubReductionViewSplit> split =
        T::CubReduction::splitRefusedView(views[0], axes, T::DataType::FP32, reducer_reads_pitched);
    if (split.has_value() && split->kind == T::CubReductionViewSplit::Kind::SlicedOutput) {
        try {
            for (const T::StridedSpan& v : views) static_cast<void>(split->reducedInput(v));
        } catch (const std::invalid_argument&) {
            split = T::CubReduction::splitRefusedView(views[0].getDimensions(), views[0].getStridesElements(), axes);
        }
    }
    return split;
}

// 1392 (phase 7 C): another way to run an H cell, timed beside the main: the split (when the main reads the view in
// place) or the in-place read at a forced W. Its outputs are the cell's output views read with its output dimensions.
struct Alternative {
    std::string name;  // the CSV column is <name>_us
    ViewRunner runner;
    std::vector<T::StridedSpan> outputs;
    Timed timed;
    size_t next_input = 0, next_output = 0;
    bool verified = false;
};

struct Cell {
    CellSpec spec;
    ViewRunner run;
    std::shared_ptr<T::StampedCubReduction> dense_stamp;
    std::vector<std::unique_ptr<Alternative>> alternatives;  // 1392: H only
    std::vector<T::StridedSpan> inputs, dense_inputs, outputs;  // views of the input and output rings
    std::vector<uint64_t> input_offsets;  // byte offsets of the views' first elements in the input ring
    size_t next_input = 0, next_output = 0, next_reference = 0;
    double stamp_ms = 0, copy_us = 0, touched_bytes = 0, useful_bytes = 0, max_rel_err = 0;
    bool verified = false;
    std::string passes, dense_passes;
    const char* reference_name = "none";
    Timed main, reference;
    bool has_reference = false;
    bool view_unsupported = false;  // 1375 (E): the view stamp throws; `stamp` is then the dense stamp
};

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Settings settings = parse(argc, argv);
        const std::vector<CellSpec> specs = cells(settings);
        const uint64_t total = std::min<uint64_t>(specs.size(), settings.max_cells);
        const uint64_t b = settings.dtype->bytes;
        if (settings.list_only) {
            std::cout << "cell_id,case,operation,dtype,dims,strides,axes,count,outputs,span_elements\n";
            for (uint64_t n = 0; n < total; ++n) {
                const CellSpec& c = specs[n];
                std::vector<uint64_t> axes(c.axes.begin(), c.axes.end());
                std::cout << c.id << ',' << c.kind << ',' << c.op->name << ',' << settings.dtype->name << ','
                          << joined(c.dims) << ',' << joined(c.strides) << ',' << joined(axes) << ',' << c.count
                          << ',' << c.outputs << ',' << c.span_elements << '\n';
            }
            return 0;
        }

        const T::TensorPlacement placement(T::TensorPlacement::MemDevices::GPU, 0);
        ScopedGpu scoped_gpu(0);
        Stream stream(placement);
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties, 0), "device properties");
        const T::ScopedCubReductionTuningDepth depth_scope(settings.depth);

        Ring inputs(placement, settings.arena_bytes);
        Ring outputs(placement, settings.output_ring_bytes);
        fillWordsKernel<<<4 * properties.multiProcessorCount, 256, 0, stream.getStream()>>>(
            static_cast<uint32_t*>(inputs.storage().getMemPtr<void>()), settings.arena_bytes / 4,
            settings.dtype->half_pattern);  // 1363: the ring is UINT8, so read it untyped (getMemPtr<uint32_t> throws)
        check(cudaGetLastError(), "input ring fill");
        check(cudaMemsetAsync(outputs.storage().getMemPtr<void>(), 0, settings.output_ring_bytes, stream.getStream()),
              "output ring initialization");
        const uint64_t gather_ctas = uint64_t(GatherCtasPerSm) * properties.multiProcessorCount;
        T::Tensor gather_out(placement, T::TensorDescriptor(T::DataType::UINT32, {gather_ctas}));
        stream.synchronize();

        uint64_t min_moved = UINT64_MAX, max_moved = 4 * MiB;
        for (uint64_t n = 0; n < total; ++n) {
            const double moved = touchedInputBytes(specs[n], b) + 4.0 * double(specs[n].outputs);
            min_moved = std::min<uint64_t>(min_moved, static_cast<uint64_t>(moved));
            max_moved = std::max<uint64_t>(max_moved, static_cast<uint64_t>(moved));
        }
        const uint64_t copy_bytes = 256 * MiB;
        const double bandwidth = B::ColdTiming::copyBandwidth(inputs, outputs, stream, copy_bytes);
        const CopyCurve curve = B::ColdTiming::copyCurve(inputs, outputs, stream, max_moved, settings.interval_us,
                                                         std::max<uint64_t>(4 * KiB, std::min(min_moved, max_moved)));

        std::cout << "# view_reduction_audit=1 version=1335 method=ring_batched_probe dtype=" << settings.dtype->name
                  << " cells=" << total << " cells_per_case=" << settings.cells_ab
                  << " cells_per_stride=" << settings.cells_c << " seed=" << settings.seed
                  << " min_kib=" << settings.min_bytes / KiB << " max_mib=" << settings.max_mib
                  << " max_span_mib=" << settings.max_span_bytes / MiB << " samples=" << settings.samples
                  << " interval_us=" << settings.interval_us << " max_launches=" << settings.max_launches
                  << " arena_bytes=" << settings.arena_bytes << " output_ring_bytes=" << settings.output_ring_bytes
                  << " input_reuse_bytes=" << InputReuseBytes << " output_reuse_bytes=" << OutputReuseBytes
                  << " input=0.5 value_output=fp32 tuning_depth=" << T::cubReductionTuningDepthRank(settings.depth)
                  << " gather_ctas_per_sm=" << GatherCtasPerSm << " gather_threads=" << GatherThreads
                  << " gather_unroll=" << GatherUnroll << " device=\"" << properties.name
                  << "\" sm_count=" << properties.multiProcessorCount << " cc=" << properties.major << '.'
                  << properties.minor << " l2_bytes=" << properties.l2CacheSize << " copy_bandwidth_tbps=" << std::fixed
                  << std::setprecision(4) << bandwidth / 1e6 << '\n';
        std::cout << "# copy_curve";
        for (const auto& [moved, us] : curve.points) std::cout << ' ' << static_cast<uint64_t>(moved) << '=' << us;
        std::cout << '\n';
        std::cout << "cell_id,case,operation,dtype,dims,strides,axes,count,outputs,stride,pitch_bytes,span_bytes,"
                     "useful_bytes,touched_bytes,stamp_ms,passes,reference,reference_passes,verified,max_rel_err,"
                     "probe_us,launches";
        for (int i = 0; i < settings.samples; ++i) std::cout << ",sample" << i << "_us";
        // 1392 (phase 7 C): H cells' alternatives (empty elsewhere): the split when the main reads the view in place,
        // the in-place read at each W, and their passes.
        std::cout << ",median_us,reference_us,copy_us,tbps,of_copy,of_reference,split_us,pitched_w1_us,pitched_w2_us,"
                     "pitched_w4_us,pitched_w8_us,alternative_passes\n";

        uint64_t collected = 0, failed = 0, unverified = 0;
        const auto wall_start = std::chrono::steady_clock::now();
        B::Fp8OracleBatchQueue queue(settings.queue_cells, settings.samples, [&] { stream.synchronize(); });

        const auto print = [&](const Cell& cell) {
            const CellSpec& c = cell.spec;
            const std::vector<double> us = cell.main.samples();
            const double med = median(us);
            const double ref = cell.has_reference ? median(cell.reference.samples()) : 0.0;
            std::vector<uint64_t> axes(c.axes.begin(), c.axes.end());
            std::cout << std::defaultfloat << c.id << ',' << c.kind << ',' << c.op->name << ','
                      << settings.dtype->name << ',' << joined(c.dims) << ',' << joined(c.strides) << ','
                      << joined(axes) << ',' << c.count << ',' << c.outputs << ',' << c.stride << ',' << c.stride * b
                      << ',' << c.span_elements * b << ',' << uint64_t(cell.useful_bytes) << ','
                      << uint64_t(cell.touched_bytes) << ',' << std::setprecision(4) << cell.stamp_ms << ','
                      << cell.passes << ',' << cell.reference_name << ',' << cell.dense_passes << ','
                      << int(cell.verified) << ',' << cell.max_rel_err << ',' << cell.main.probe_us << ','
                      << cell.main.launches;
            for (double u : us) std::cout << ',' << u;
            std::cout << ',' << med << ',' << ref << ',' << cell.copy_us << ',' << cell.touched_bytes / med / 1e6
                      << ',' << cell.copy_us / med << ',' << (ref > 0 ? ref / med : 0.0);
            std::string alternative_passes;
            for (const char* name : {"split", "pitched_w1", "pitched_w2", "pitched_w4", "pitched_w8"}) {
                std::cout << ',';
                for (const std::unique_ptr<Alternative>& alt : cell.alternatives) {
                    if (alt->name != name) continue;
                    std::cout << median(alt->timed.samples());
                    if (alt->name == "split") alternative_passes = alt->runner.passes();
                }
            }
            for (const std::unique_ptr<Alternative>& alt : cell.alternatives) {
                if (alt->name == "pitched_w1") alternative_passes += (alternative_passes.empty() ? "" : "|") +
                                                                    passNames(*alt->runner.stamp);
            }
            std::replace(alternative_passes.begin(), alternative_passes.end(), ',', ';');
            std::cout << ',' << alternative_passes << '\n';
            ++collected;
            if (collected % 25 == 0 || collected + failed == total) {
                const double wall =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count();
                std::cerr << "[view-reduction-audit] cells=" << collected + failed << '/' << total
                          << " failed=" << failed << " unverified=" << unverified << " wall_s=" << std::fixed
                          << std::setprecision(1) << wall << std::defaultfloat << std::endl;
            }
        };

        for (uint64_t n = 0; n < total; ++n) {
            auto cell = std::make_shared<Cell>();
            cell->spec = specs[n];
            const CellSpec& c = cell->spec;
            try {
                cell->useful_bytes = double(c.count * c.outputs * b) + 4.0 * double(c.outputs);
                if (c.kind == 'c') cell->useful_bytes = double(c.count * b) + 4.0;
                cell->touched_bytes = touchedInputBytes(c, b) + 4.0 * double(c.outputs);
                cell->copy_us = curve.us(cell->touched_bytes);
                const T::DataType dtype = settings.dtype->dtype;
                const uint64_t span_bytes = c.span_elements * b;

                // Input views: each a fresh region of the ring; C's wide pitches also shift by 32 bytes in a region.
                const uint64_t pitch = c.stride * b;
                const uint64_t shifts = (c.kind == 'c' && pitch >= 64) ? std::min<uint64_t>(pitch / 32, MaxViews) : 1;
                const uint64_t region = span_bytes + (shifts > 1 ? pitch : 0);
                const double touched_input = touchedInputBytes(c, b);
                const uint64_t wanted = std::clamp<uint64_t>(
                    static_cast<uint64_t>(std::ceil(double(InputReuseBytes) / std::max(touched_input, 1.0))), 1,
                    MaxViews);
                const uint64_t regions = std::clamp<uint64_t>(
                    B::ColdTiming::ceilDiv(wanted, shifts), 1,
                    std::max<uint64_t>(1, inputs.bytes() / 2 / Ring::span(region)));
                for (uint64_t r = 0; r < regions && cell->inputs.size() < wanted; ++r) {
                    const uint64_t base = inputs.take(region);
                    for (uint64_t k = 0; k < shifts && cell->inputs.size() < wanted; ++k) {
                        const uint64_t offset = base + 32 * k;
                        cell->input_offsets.push_back(offset);
                        const T::StridedSpan flat = inputs.view(dtype, {c.span_elements}, offset);
                        cell->inputs.push_back(flat.withLayout(c.dims, c.strides));
                        if (c.kind != 'c') cell->dense_inputs.push_back(inputs.view(dtype, c.dense_dims, offset));
                    }
                }

                // The production stamp, bound to the first view. 1375 (E): a view no production reducer owns throws
                // NotImplementedException; the cell then times the dense stamp of its physical tensor instead.
                cell->reference_name = c.kind == 'c' ? (c.stride == 0 ? "none" : "gather") : "dense_stamp";
                const auto stamp_start = std::chrono::steady_clock::now();
                std::vector<uint64_t> out_dims;
                // 1391 (phase 7 A): the split of the span, as Expression takes it (ringSplit). 1392: none for an H
                // view the reducer reads in place.
                cell->run.split = ringSplit(cell->inputs, c.axes, true);
                if (cell->run.split.has_value()) {
                    out_dims = cell->run.stampSplit(c.op->op, cell->inputs[0], placement, stream);
                } else {
                    try {
                        cell->run.stamp =
                            T::CubReduction(c.op->op, c.axes, T::DataType::FP32).stamp(cell->inputs[0], stream);
                    } catch (const NotImplementedException&) {
                        if (c.kind != 'e') throw;
                        cell->view_unsupported = true;
                        cell->reference_name = "view_unsupported";
                        cell->run.stamp = T::CubReduction(c.op->op, c.dense_axes, T::DataType::FP32)
                                              .stamp(cell->dense_inputs[0], stream);
                    }
                    out_dims = cell->run.stamp->getOutputTensor().getDimensions();
                }
                for (uint64_t v = 0, count = B::ColdTiming::viewCount(4 * c.outputs, OutputReuseBytes,
                                                                      outputs.bytes());
                     v < count; ++v)
                    cell->outputs.push_back(outputs.view(T::DataType::FP32, out_dims, outputs.take(4 * c.outputs)));
                cell->run.planOutputCopy(cell->outputs[0]);
                stream.synchronize();
                cell->stamp_ms =
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - stamp_start).count();
                cell->passes = cell->run.passes();

                // One run of the cell on a view and an output: the stamp, or the two parts (1390).
                Cell* raw = cell.get();
                const auto run_on = [raw, &stream](const T::StridedSpan& input, const T::StridedSpan& output) {
                    raw->run.run(input, output, stream);
                };
                const std::vector<T::StridedSpan>& main_inputs =
                    cell->view_unsupported ? cell->dense_inputs : cell->inputs;

                // Verified run on the first views (`output` after a run: whether every value is the closed form).
                const auto verify = [&c, &stream](const T::StridedSpan& output, double* max_rel_err) {
                    std::vector<float> host(c.outputs);
                    check(cudaMemcpyAsync(host.data(), output.getMemPtr<float>(), 4 * c.outputs,
                                          cudaMemcpyDeviceToHost, stream.getStream()),
                          "verify copy");
                    stream.synchronize();
                    bool verified = true;
                    for (float got : host) {
                        double rel = 0;
                        verified = verifyValue(c.op->op, c.count, got, &rel) && verified;
                        *max_rel_err = std::max(*max_rel_err, rel);
                    }
                    return verified ? std::optional<float>() : std::optional<float>(host.front());
                };
                run_on(main_inputs[0], cell->outputs[0]);
                const std::optional<float> wrong = verify(cell->outputs[0], &cell->max_rel_err);
                cell->verified = !wrong.has_value();
                if (!cell->verified) {
                    ++unverified;
                    std::cout << "# verify_failed," << c.id << ',' << c.kind << ',' << c.op->name << ",count="
                              << c.count << ",got=" << *wrong << ",max_rel_err=" << cell->max_rel_err << '\n';
                }

                cell->main.run_once = [raw, run_on] {
                    const std::vector<T::StridedSpan>& views = raw->view_unsupported ? raw->dense_inputs : raw->inputs;
                    const T::StridedSpan& input = views[raw->next_input];
                    raw->next_input = (raw->next_input + 1) % views.size();
                    const T::StridedSpan& output = raw->outputs[raw->next_output];
                    raw->next_output = (raw->next_output + 1) % raw->outputs.size();
                    run_on(input, output);
                };
                cell->main.probe(stream, settings);

                // The reference: the dense stamp of the same bytes (A / B / D-G; H packed) or the gather kernel (C,
                // s >= 2).
                // An unsupported E view has none: its main is that dense stamp.
                if (cell->view_unsupported) {
                    cell->has_reference = false;
                } else if (c.kind != 'c') {
                    cell->dense_stamp = T::CubReduction(c.op->op, c.dense_axes, T::DataType::FP32)
                                            .stamp(cell->dense_inputs[0], stream);
                    cell->dense_passes = passNames(*cell->dense_stamp);
                    cell->reference.run_once = [raw, &stream] {
                        const T::StridedSpan& input = raw->dense_inputs[raw->next_reference];
                        raw->next_reference = (raw->next_reference + 1) % raw->dense_inputs.size();
                        const T::StridedSpan& output = raw->outputs[raw->next_output];
                        raw->next_output = (raw->next_output + 1) % raw->outputs.size();
                        raw->dense_stamp->runOnTensorsForBenchmark(input, output, stream);
                    };
                    cell->has_reference = true;
                } else if (c.stride != 0) {
                    const uint32_t grid = GatherCtasPerSm * properties.multiProcessorCount;
                    uint8_t* ring = inputs.storage().getMemPtr<uint8_t>();
                    uint32_t* out = gather_out.getMemPtr<uint32_t>();
                    cell->dense_passes = "gather_reference";
                    cell->reference.run_once = [raw, &stream, ring, out, grid, b] {
                        const uint64_t offset = raw->input_offsets[raw->next_reference];
                        raw->next_reference = (raw->next_reference + 1) % raw->input_offsets.size();
                        const uint64_t n = raw->spec.count, s = raw->spec.stride;
                        if (b == 1)
                            gatherReferenceKernel<uint8_t><<<grid, GatherThreads, 0, stream.getStream()>>>(
                                ring + offset, n, s, out);
                        else if (b == 2)
                            gatherReferenceKernel<uint16_t><<<grid, GatherThreads, 0, stream.getStream()>>>(
                                reinterpret_cast<const uint16_t*>(ring + offset), n, s, out);
                        else
                            gatherReferenceKernel<uint32_t><<<grid, GatherThreads, 0, stream.getStream()>>>(
                                reinterpret_cast<const uint32_t*>(ring + offset), n, s, out);
                        check(cudaGetLastError(), "gather reference");
                    };
                    cell->has_reference = true;
                }
                if (cell->has_reference) cell->reference.probe(stream, settings);

                // 1392 (phase 7 C): an H cell also times the split when its main reads the view in place, and the
                // in-place read at each W (CubReduction::stampPitched) whether or not the fill rule picks it.
                if (c.kind == 'h') {
                    if (!cell->run.split.has_value()) {
                        auto alt = std::make_unique<Alternative>();
                        alt->name = "split";
                        alt->runner.split = ringSplit(cell->inputs, c.axes, false);
                        if (alt->runner.split.has_value()) {
                            const std::vector<uint64_t> dims =
                                alt->runner.stampSplit(c.op->op, cell->inputs[0], placement, stream);
                            for (const T::StridedSpan& o : cell->outputs)
                                alt->outputs.push_back(o.withLayout(dims, T::StridedSpan::denseStrides(dims)));
                            alt->runner.planOutputCopy(alt->outputs[0]);
                            cell->alternatives.push_back(std::move(alt));
                        }
                    }
                    for (const uint32_t w : {1U, 2U, 4U, 8U}) {
                        auto alt = std::make_unique<Alternative>();
                        alt->name = "pitched_w" + std::to_string(w);
                        try {
                            alt->runner.stamp =
                                T::CubReduction(c.op->op, c.axes, T::DataType::FP32).stampPitched(cell->inputs[0], w,
                                                                                                    stream);
                        } catch (const NotImplementedException&) {
                            break;  // PacketComplete cannot read the view
                        }
                        const std::vector<uint64_t> dims = alt->runner.stamp->getOutputTensor().getDimensions();
                        for (const T::StridedSpan& o : cell->outputs)
                            alt->outputs.push_back(o.withLayout(dims, T::StridedSpan::denseStrides(dims)));
                        cell->alternatives.push_back(std::move(alt));
                    }
                    for (const std::unique_ptr<Alternative>& alt : cell->alternatives) {
                        Alternative* a = alt.get();
                        a->runner.run(cell->inputs[0], a->outputs[0], stream);
                        double rel = 0;
                        const std::optional<float> alt_wrong = verify(a->outputs[0], &rel);
                        a->verified = !alt_wrong.has_value();
                        if (!a->verified) {
                            ++unverified;
                            std::cout << "# verify_failed," << c.id << ',' << c.kind << ',' << c.op->name << ','
                                      << a->name << ",count=" << c.count << ",got=" << *alt_wrong
                                      << ",max_rel_err=" << rel << '\n';
                        }
                        a->timed.run_once = [raw, a, &stream] {
                            const T::StridedSpan& input = raw->inputs[a->next_input];
                            a->next_input = (a->next_input + 1) % raw->inputs.size();
                            const T::StridedSpan& output = a->outputs[a->next_output];
                            a->next_output = (a->next_output + 1) % a->outputs.size();
                            a->runner.run(input, output, stream);
                        };
                        a->timed.probe(stream, settings);
                    }
                    // An alternative over 50x the main and 1 ms (a few long rows per warp) is left untimed (empty).
                    std::erase_if(cell->alternatives, [&](const std::unique_ptr<Alternative>& alt) {
                        return alt->timed.probe_us > std::max(1000.0, 50.0 * cell->main.probe_us);
                    });
                }
            } catch (const std::exception& e) {
                ++failed;
                std::string what = e.what();
                std::replace(what.begin(), what.end(), '\n', ' ');
                std::cout << "# failed," << c.id << ',' << c.kind << ',' << c.op->name << ",dims=" << joined(c.dims)
                          << ",strides=" << joined(c.strides) << ',' << what << '\n';
                continue;
            }
            queue.add({[] {},
                       [cell, &stream](int pass) {
                           cell->main.submit(pass, stream);
                           if (cell->has_reference) cell->reference.submit(pass, stream);
                           for (const std::unique_ptr<Alternative>& alt : cell->alternatives)
                               alt->timed.submit(pass, stream);
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
        std::cerr << "thor_view_reduction_audit: " << e.what() << '\n';
        return 1;
    }
}
