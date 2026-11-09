// 1263: thor_ragged_reduction_audit. What a user gets from a production CubSegmentedReduction /
// CubSegmentedArgReduction stamp (values [rows, D...], row offsets [segments + 1]) at any distribution of segment
// lengths, timed cold against the copy curve of the same GPU, in TB/s. Phase 0 of
// plans/Thor_Ragged_Reduction_Plan_2026-11-06.md: the baseline of the current kernels, and the before / after check of
// every later phase.
//
// Cells (benchmarks/RaggedReductionAuditCells.h): --shapes seeded shapes, each run under every distribution in
// --distributions, so the report can compare distributions at a fixed (rows, segments, D, dtype, op). A shape draws an
// op from --ops, D from --d-values, an offset dtype from --offsets, a capacity ratio from --capacity-ratios (values hold
// ceil(rows * ratio) rows, of which the offsets make the first `rows` active), a moved-bytes target log-uniform from
// --min-kib to --max-mib, and a mean segment length log-uniform from 1 to --max-mean-length. D = 1 is rank-1 values.
// VALUE output is FP32; ARG writes a UINT32 index (UINT64 when the packed indices need it).
//
// Method (benchmarks/ColdTiming.h, as thor_dense_production_audit): the input ring holds 0.5 in the input dtype in every
// element. Each cell is stamped once on its first views (stampRuntimeOffsets, the Expression / network path, or with
// --stamp=validated the stamp that reads the offsets), then run with runOnTensorsForBenchmark on rotating input views
// spanning at least 2 GiB, rotating offsets copies spanning at least --offsets-reuse-mib (each cell owns its copies)
// and rotating output views spanning at least 1 GiB. One verified run (each segment against its closed form, see the
// header), one untimed run, one timed probe, then --samples samples of about --interval-us of back-to-back runs (at
// most --max-launches), interleaved across the cells of a batch.
//
// Moved bytes are the active input, the offsets and the outputs; the reference is the copy curve (a device copy moving
// the same bytes, measured the same way at startup).
//
// Output: '#' header lines (settings, device, copy curve) and one CSV row per cell. benchmarks/ragged_reduction_audit.py
// reports it.
//
// 1265: --backend=ragged_kr times the retained ragged KR VALUE kernel (Cub/RaggedReduction.h, not yet routed) instead
// of the production stamp: FP8 input, D = 1, SUM / MEAN / MIN / MAX only (other cells are skipped and counted). Every
// eligible cell runs once per --ragged-kr-shapes candidate on its persistent grid (one '# ragged_kr_candidate' line
// each); the report keeps each cell's best candidate for the gate and lists the candidates. The ragged_kr and
// production runs of one setting draw the same shapes, so --compare in the report is the before / after.
//
// 1275: --ragged-kr-time=main (or finish) times the main kernel (or the finisher) alone, for the phase 2 gate's speed
// and spread clauses (the main kernel against the copy curve; the finisher is a fixed tax per stamped grid). The
// outputs of those timed runs are incomplete; verification and the untimed run launch both kernels.
//
// 1280: FP8 cells with D up to 15 (rows under 16 bytes) are eligible too (the components kernel at D >= 2). Each
// candidate has a grid per kernel (D = 1 and D >= 2, their residencies differ; both on its '# ragged_kr_candidate'
// line), and each cell's workspace holds its D's records.
//
// 1284 (phase 3b): every input dtype is eligible, at D up to raggedKRMaxComponents (rows under 16 bytes: FP16 / BF16
// D <= 7, FP32 D <= 3), through the one launcher (launchRaggedKRValue).
//
// 1285 (phase 4): FP8 cells (1295: every dtype) with rows of 16 bytes and more run ragged RK (launchRaggedRKValue) once per
// --ragged-rk-shapes candidate (each candidate's CTA budget on a '# ragged_rk_candidate' line; passes
// "ragged_rk:<shape>+ragged_finish"); --ragged-kr-time applies to both families.
//
// 1298 (phase 5): ARG cells that ragged KR ARG takes (raggedKRArgSupported: FP8 at D = 1; 1300: D 1 to 15; 1311:
// FP16 / BF16 at D 1 to 7; 1313: FP32 at D 1 to 3) run launchRaggedKRArg once per --ragged-kr-shapes candidate on the
// ARG kernel's grid (arg_grid, and 1300 arg_components_grid for D >= 2, on the '# ragged_kr_candidate' line; passes
// "ragged_kr:<shape>+ragged_finish_arg"), verified against the closed form like the production ARG cells.
//
// 1314 (phase 5): ARG cells with rows of 16 bytes and more that ragged RK ARG takes (raggedRKArgSupported: FP8 at D >=
// 16; 1320: FP16 / BF16 at D >= 8; 1321: FP32 at D >= 4) run launchRaggedRKArg once per --ragged-rk-shapes candidate
// on the RK ARG kernels' CTA budget (arg_ctas on the '# ragged_rk_candidate' line; passes
// "ragged_rk:<shape>+ragged_finish_arg").
//
// 1323 (phase 6): --backend=ragged_select runs the cells --backend=ragged_kr runs, once each, on the shape the ragged
// selector picks at stamp time (selectCachedRagged, Cub/RaggedStampAutotuner.h: every candidate timed on synthetic
// uniform offsets at the cell's N, B, D and input alignment; stamp_ms is the selection's time, near 0 when another
// distribution of the shape already filled the cache). passes is the winner's name; a '# ragged_select' line per cell
// gives the contest (each candidate's time on the synthetic offsets) and whether it hit the cache. Joined with a
// --backend=ragged_kr run of the same settings (the same shapes, each candidate on the real offsets), it measures the
// pick's regret per distribution.
//
// 1325 (phase 7) added --ragged-routing=on|off; 1328 (phase 8) removed it with the switch: --backend=production now
// stamps the retained ragged kernels wherever a ragged family runs the cell (passes names them).
#include "benchmarks/ColdTiming.h"
#include "benchmarks/Fp8OracleBatchQueue.h"
#include "benchmarks/RaggedReductionAuditCells.h"
#include "Utilities/TensorOperations/Cub/CubReduction.h"
#include "Utilities/TensorOperations/Cub/CubReductionTuningDepth.h"
#include "Utilities/TensorOperations/Cub/RaggedReduction.h"
#include "Utilities/TensorOperations/Cub/RaggedStampAutotuner.h"
#include "Utilities/Common/ScopedGpu.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace T = ThorImplementation;
namespace B = ThorImplementation::CubReductionBenchmarking;
namespace A = ThorImplementation::RaggedReductionAudit;
using B::ColdTiming::align256;
using B::ColdTiming::CopyCurve;
using B::ColdTiming::GiB;
using B::ColdTiming::InputReuseBytes;
using B::ColdTiming::MiB;
using B::ColdTiming::OutputReuseBytes;
using B::ColdTiming::Ring;
using B::ColdTiming::TimingEvent;

namespace {

constexpr uint64_t KiB = 1ULL << 10;
constexpr uint64_t MaxOffsetCopies = 64;

void check(cudaError_t status, const char* what) { B::ColdTiming::checkCuda(status, what); }

struct OpSpec {
    const char* name;
    bool arg;
    A::Op op;
    T::CubReductionOp value_op;
    T::CubArgReductionOp arg_op;
};
constexpr OpSpec OPS[] = {
    {"sum", false, A::Op::Sum, T::CubReductionOp::Sum, T::CubArgReductionOp::ArgMax},
    {"mean", false, A::Op::Mean, T::CubReductionOp::Mean, T::CubArgReductionOp::ArgMax},
    {"min", false, A::Op::Min, T::CubReductionOp::Min, T::CubArgReductionOp::ArgMax},
    {"max", false, A::Op::Max, T::CubReductionOp::Max, T::CubArgReductionOp::ArgMax},
    {"argmin", true, A::Op::ArgMin, T::CubReductionOp::Min, T::CubArgReductionOp::ArgMin},
    {"argmax", true, A::Op::ArgMax, T::CubReductionOp::Max, T::CubArgReductionOp::ArgMax},
};

const OpSpec& opSpec(std::string_view name) {
    for (const OpSpec& op : OPS)
        if (name == op.name) return op;
    throw std::invalid_argument("--ops takes sum, mean, min, max, argmin and argmax");
}

struct DTypeSpec {
    const char* name;
    T::DataType dtype;
    A::StorageFormat format;
};
constexpr DTypeSpec DTYPES[] = {
    {"fp32", T::DataType::FP32, A::StorageFormat::FP32},
    {"fp16", T::DataType::FP16, A::StorageFormat::FP16},
    {"bf16", T::DataType::BF16, A::StorageFormat::BF16},
    {"fp8_e4m3", T::DataType::FP8_E4M3, A::StorageFormat::FP8_E4M3},
    {"fp8_e5m2", T::DataType::FP8_E5M2, A::StorageFormat::FP8_E5M2},
};

const DTypeSpec& dtypeSpec(std::string_view name) {
    for (const DTypeSpec& d : DTYPES)
        if (name == d.name) return d;
    throw std::invalid_argument("--dtype takes fp32, fp16, bf16, fp8_e4m3 or fp8_e5m2");
}

struct OffsetSpec {
    const char* name;
    T::DataType dtype;
    uint64_t bytes;
};
constexpr OffsetSpec OFFSETS[] = {{"uint32", T::DataType::UINT32, 4}, {"uint64", T::DataType::UINT64, 8}};

const OffsetSpec& offsetSpec(std::string_view name) {
    for (const OffsetSpec& o : OFFSETS)
        if (name == o.name) return o;
    throw std::invalid_argument("--offsets takes uint32 and uint64");
}

std::vector<std::string> splitList(std::string_view text) {
    std::vector<std::string> out;
    std::stringstream stream{std::string(text)};
    for (std::string item; std::getline(stream, item, ',');)
        if (!item.empty()) out.push_back(item);
    return out;
}

template <typename T>
std::string joinList(const std::vector<T>& items, char separator = ',') {
    std::ostringstream out;
    for (size_t i = 0; i < items.size(); ++i) out << (i ? std::string(1, separator) : "") << items[i];
    return out.str();
}

struct Settings {
    const DTypeSpec* dtype = &DTYPES[0];
    std::vector<std::string> ops{"sum", "mean", "min", "max", "argmin", "argmax"};
    std::vector<A::Distribution> distributions =
        std::vector<A::Distribution>(std::begin(A::DISTRIBUTIONS), std::end(A::DISTRIBUTIONS));
    std::vector<uint64_t> d_values{1, 2, 3, 4, 7, 8, 16, 17, 64, 255, 256, 1024, 4096, 5000};
    std::vector<std::string> offsets{"uint32"};
    std::vector<double> capacity_ratios{1.0, 1.25};
    uint64_t shapes = 120;
    uint64_t seed = 1263;
    uint64_t min_bytes = 1024 * KiB;
    uint64_t max_mib = 256;
    uint64_t max_mean_length = uint64_t{1} << 22;
    uint64_t max_cells = UINT64_MAX;
    std::vector<uint64_t> shape_ids;  // 1269: only these shapes (empty: all); the draws of the others do not change
    int samples = 5;
    double interval_us = 1000.0;
    uint64_t max_launches = 256;
    uint64_t arena_bytes = 8 * GiB;
    uint64_t output_ring_bytes = 3 * GiB;
    uint64_t offsets_reuse_bytes = 256 * MiB;
    size_t queue_cells = 8;
    T::CubReductionTuningDepth depth = T::CubReductionTuningDepth::Regular;
    bool validated_stamp = false;
    bool ragged_kr = false;  // --backend=ragged_kr
    bool ragged_select = false;  // 1323: --backend=ragged_select
    // 1272: the end's extra weight swept; unweighted dropped (it lost to weighted throughout 1268-1271). 1273: i16 won
    // 264 of 300 cells, its fit put the extra near 4; e12 for comparison, 8 CTAs per SM (1360) to see whether smaller
    // cells want a smaller grid, and i32 after its copy-order fix. 1274: c8 won every cell below 4 MiB and i32 three
    // cells in all, so i32 makes way for 4 CTAs per SM (680): the grid for a cell's size is the stamp's choice. 1275:
    // 680 CTAs won no cell in 1274 and is dropped.
    std::vector<T::CubReductionInternal::RaggedKRShape> ragged_kr_shapes{
        {16, 0, true, 4}, {16, 0, true, 12}, {16, 8, true, 4}};
    // 1285: ragged RK candidates (FP8 rows >= 16 bytes): the auto K tile cap (from the input's size), and the caps 512
    // and 64, so the first sweep shows what the tile width costs on each side of the auto choice.
    std::vector<T::CubReductionInternal::RaggedRKShape> ragged_rk_shapes{{0, 0, true, 4}, {512, 0, true, 4},
                                                                          {64, 0, true, 4}};
    // 1275: --ragged-kr-time: which kernels the timed runs launch (full: both; main or finish: that kernel alone, the
    // outputs then incomplete). Verification always runs both.
    T::CubReductionInternal::RaggedLaunchParts ragged_parts = T::CubReductionInternal::RaggedLaunchParts::MainAndFinish;
    bool verify = true;
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
        } else if (auto v = value("--ops=")) {
            s.ops = splitList(*v);
            for (const std::string& op : s.ops) (void)opSpec(op);
        } else if (auto v = value("--distributions=")) {
            s.distributions.clear();
            for (const std::string& d : splitList(*v)) s.distributions.push_back(A::distributionFromName(d));
        } else if (auto v = value("--d-values=")) {
            s.d_values.clear();
            for (const std::string& d : splitList(*v)) s.d_values.push_back(std::max<uint64_t>(1, std::stoull(d)));
        } else if (auto v = value("--offsets=")) {
            s.offsets = splitList(*v);
            for (const std::string& o : s.offsets) (void)offsetSpec(o);
        } else if (auto v = value("--capacity-ratios=")) {
            s.capacity_ratios.clear();
            for (const std::string& r : splitList(*v)) s.capacity_ratios.push_back(std::stod(r));
            for (double r : s.capacity_ratios)
                if (!(r >= 1.0 && r <= 4.0)) throw std::invalid_argument("--capacity-ratios take values in [1, 4]");
        } else if (auto v = value("--shapes=")) {
            s.shapes = std::stoull(*v);
        } else if (auto v = value("--seed=")) {
            s.seed = std::stoull(*v);
        } else if (auto v = value("--min-kib=")) {
            s.min_bytes = std::stoull(*v) * KiB;
        } else if (auto v = value("--max-mib=")) {
            s.max_mib = std::stoull(*v);
        } else if (auto v = value("--max-mean-length=")) {
            s.max_mean_length = std::max<uint64_t>(1, std::stoull(*v));
        } else if (auto v = value("--max-cells=")) {
            s.max_cells = std::stoull(*v);
        } else if (auto v = value("--shape-ids=")) {
            for (const std::string& id : splitList(*v)) s.shape_ids.push_back(std::stoull(id));
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
        } else if (auto v = value("--offsets-reuse-mib=")) {
            s.offsets_reuse_bytes = std::stoull(*v) * MiB;
        } else if (auto v = value("--queue-cells=")) {
            s.queue_cells = std::max<size_t>(1, std::stoull(*v));
        } else if (auto v = value("--tuning-depth=")) {
            if (*v == "quick") s.depth = T::CubReductionTuningDepth::Quick;
            else if (*v == "regular") s.depth = T::CubReductionTuningDepth::Regular;
            else if (*v == "ultra") s.depth = T::CubReductionTuningDepth::Ultra;
            else throw std::invalid_argument("--tuning-depth takes quick, regular or ultra");
        } else if (auto v = value("--stamp=")) {
            if (*v == "runtime") s.validated_stamp = false;
            else if (*v == "validated") s.validated_stamp = true;
            else throw std::invalid_argument("--stamp takes runtime or validated");
        } else if (auto v = value("--backend=")) {
            s.ragged_kr = *v == "ragged_kr";
            s.ragged_select = *v == "ragged_select";  // 1323
            if (*v != "production" && !s.ragged_kr && !s.ragged_select)
                throw std::invalid_argument("--backend takes production, ragged_kr or ragged_select");
        } else if (auto v = value("--ragged-kr-time=")) {
            using T::CubReductionInternal::RaggedLaunchParts;
            if (*v == "full") s.ragged_parts = RaggedLaunchParts::MainAndFinish;
            else if (*v == "main") s.ragged_parts = RaggedLaunchParts::MainOnly;
            else if (*v == "finish") s.ragged_parts = RaggedLaunchParts::FinishOnly;
            else throw std::invalid_argument("--ragged-kr-time takes full, main or finish");
        } else if (auto v = value("--ragged-kr-shapes=")) {
            s.ragged_kr_shapes.clear();
            for (const std::string& name : splitList(*v)) {
                T::CubReductionInternal::RaggedKRShape shape;
                char tail = 0;
                unsigned items = 0, ctas = 0, extra = 0;
                std::string base = name;
                shape.weighted = !(base.size() > 2 && base.ends_with("_u"));  // 1268: unweighted split
                if (!shape.weighted) base.resize(base.size() - 2);
                // 1272: "_eN" sets the end's extra weight; without it, the default for the rows.
                if (shape.weighted && std::sscanf(base.c_str(), "i%u_c%u_e%u%c", &items, &ctas, &extra, &tail) == 3) {
                    shape.end_extra = static_cast<int32_t>(std::min(extra, 1000U));
                } else if (std::sscanf(base.c_str(), "i%u_c%u%c", &items, &ctas, &tail) != 2) {
                    throw std::invalid_argument("--ragged-kr-shapes takes names like i32_c0_e8,i16_c0,i32_c0_u");
                }
                shape.items_per_thread = items;
                shape.ctas_per_sm = ctas;
                if (!T::CubReductionInternal::raggedKRShapeLegal(shape))
                    throw std::invalid_argument("illegal ragged KR shape " + name);
                s.ragged_kr_shapes.push_back(shape);
            }
        } else if (auto v = value("--ragged-rk-shapes=")) {  // 1285
            s.ragged_rk_shapes.clear();
            for (const std::string& name : splitList(*v)) {
                T::CubReductionInternal::RaggedRKShape shape;
                std::string base = name;
                shape.weighted = !(base.size() > 2 && base.ends_with("_u"));
                if (!shape.weighted) base.resize(base.size() - 2);
                char tail = 0;
                unsigned tile = 0, ctas = 0, extra = 0;
                const bool auto_tile = base.size() > 2 && base[0] == 't' && base[1] == 'a';
                const std::string rest = auto_tile ? "t0" + base.substr(2) : base;
                if (shape.weighted && std::sscanf(rest.c_str(), "t%u_c%u_e%u%c", &tile, &ctas, &extra, &tail) == 3) {
                    shape.end_extra = static_cast<int32_t>(std::min(extra, 1000U));
                } else if (std::sscanf(rest.c_str(), "t%u_c%u%c", &tile, &ctas, &tail) != 2) {
                    throw std::invalid_argument("--ragged-rk-shapes takes names like ta_c0_e4,t512_c0_e4,t64_c8_u");
                }
                shape.tile_bytes = tile;
                shape.ctas_per_sm = ctas;
                if (!T::CubReductionInternal::raggedRKShapeLegal(shape))
                    throw std::invalid_argument("illegal ragged RK shape " + name);
                s.ragged_rk_shapes.push_back(shape);
            }
        } else if (a == "--no-verify") {
            s.verify = false;
        } else if (a == "--list-only") {
            s.list_only = true;
        } else if (a == "--help") {
            std::cout
                << "thor_ragged_reduction_audit [--dtype=fp32] [--ops=sum,mean,min,max,argmin,argmax]\n"
                   "    [--distributions=uniform,random,pareto,half_empty,one_giant]\n"
                   "    [--d-values=1,2,3,4,7,8,16,17,64,255,256,1024,4096,5000] [--offsets=uint32[,uint64]]\n"
                   "    [--capacity-ratios=1,1.25] [--shapes=120] [--seed=1263] [--min-kib=1024] [--max-mib=256]\n"
                   "    [--max-mean-length=4194304] [--max-cells=N] [--shape-ids=a,b] [--samples=5] [--interval-us=1000]\n"
                   "    [--max-launches=256] [--arena-gib=8] [--output-ring-gib=3] [--offsets-reuse-mib=256]\n"
                   "    [--queue-cells=8] [--tuning-depth=regular] [--stamp=runtime|validated] [--no-verify]\n"
                   "    [--backend=production|ragged_kr|ragged_select]\n"
                   "    [--ragged-kr-shapes=i16_c0_e4,i16_c0_e12,i16_c8_e4]\n"
                   "    [--ragged-kr-time=full|main|finish] [--ragged-rk-shapes=ta_c0_e4,t512_c0_e4,t64_c0_e4]\n"
                   "    [--list-only]\n"
                   "Every shape runs under every distribution. CSV on stdout, progress on stderr.\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("Unknown argument " + std::string(a));
        }
    }
    if (s.ops.empty() || s.distributions.empty() || s.d_values.empty() || s.offsets.empty() ||
        s.capacity_ratios.empty())
        throw std::invalid_argument("--ops, --distributions, --d-values, --offsets and --capacity-ratios need values");
    if (s.ragged_kr && s.ragged_kr_shapes.empty()) throw std::invalid_argument("--ragged-kr-shapes needs a shape");
    if (s.ragged_kr && s.ragged_rk_shapes.empty()) throw std::invalid_argument("--ragged-rk-shapes needs a shape");
    if (s.min_bytes < 4 * KiB || s.min_bytes > s.max_mib * MiB)
        throw std::invalid_argument("need 4 KiB <= --min-kib <= --max-mib");
    const double max_ratio = *std::max_element(s.capacity_ratios.begin(), s.capacity_ratios.end());
    if (double(s.arena_bytes) < 4.0 * max_ratio * double(s.max_mib * MiB))
        throw std::invalid_argument("--arena-gib must hold at least four inputs of --max-mib at the largest capacity");
    return s;
}

struct CellSpec {
    uint64_t id = 0, shape_id = 0;
    A::Distribution distribution = A::Distribution::Uniform;
    const OpSpec* op = nullptr;
    const OffsetSpec* offsets = nullptr;
    uint64_t d = 1, rows = 1, segments = 1, capacity_rows = 1;
    double capacity_ratio = 1.0;
    T::DataType index_dtype = T::DataType::UINT32;
    int candidate = -1;  // --backend=ragged_kr: the index into --ragged-kr-shapes (rk: --ragged-rk-shapes)
    bool rk = false;     // 1285: a ragged RK cell (rows of 16 bytes and more)
    [[nodiscard]] uint64_t outputBytesPerElement() const {
        return op->arg ? (index_dtype == T::DataType::UINT64 ? 8 : 4) : 4;
    }
    [[nodiscard]] std::vector<uint64_t> valueDims(uint64_t leading) const {
        return d == 1 ? std::vector<uint64_t>{leading} : std::vector<uint64_t>{leading, d};
    }
    [[nodiscard]] uint64_t offsetsSeed(uint64_t seed) const { return A::mix(seed ^ A::mix(shape_id + 0x5EED)); }
};

// --backend=ragged_kr runs VALUE with rows under 16 bytes (1284: every dtype, D x element bytes <= 15; 1280: FP8 at D 1
// to 15; 1265-1279: FP8 at D = 1), SUM, MEAN, MIN, MAX; 1298: and ARG where ragged KR ARG takes it (FP8; 1300 D <= 15).
bool raggedKREligible(const Settings& s, const CellSpec& c) {
    const uint32_t bytes = static_cast<uint32_t>(A::elementBytes(s.dtype->format));
    if (c.op->arg) return T::CubReductionInternal::raggedKRArgSupported(s.dtype->dtype, c.d);  // 1298
    return c.d >= 1 && c.d <= T::CubReductionInternal::RaggedMergePath::raggedKRMaxComponents(bytes);
}

// 1285: --backend=ragged_kr runs ragged RK on FP8 VALUE cells with rows of 16 bytes and more (1295: every dtype; 1314:
// and ARG cells ragged RK ARG takes).
bool raggedRKEligible(const Settings& s, const CellSpec& c) {
    if (c.op->arg && !T::CubReductionInternal::raggedRKArgSupported(s.dtype->dtype, c.d)) return false;
    return c.d * A::elementBytes(s.dtype->format) >= T::CubReductionInternal::RaggedMergePath::RAGGED_RK_MIN_ROW_BYTES;
}

std::vector<CellSpec> cells(const Settings& s, uint64_t* skipped = nullptr) {
    std::vector<CellSpec> out;
    const uint64_t vb = A::elementBytes(s.dtype->format);
    const bool ragged = s.ragged_kr || s.ragged_select;  // 1323: ragged_select runs the ragged_kr cells
    for (uint64_t shape = 0; shape < s.shapes; ++shape) {
        const A::ShapeDraw draw =
            A::drawShape(s.seed, shape, s.ops.size(), s.d_values.size(), s.offsets.size(), s.capacity_ratios.size(),
                         s.min_bytes, s.max_mib * MiB, s.max_mean_length);
        if (!s.shape_ids.empty() && std::find(s.shape_ids.begin(), s.shape_ids.end(), shape) == s.shape_ids.end())
            continue;  // drawn above, so the other shapes are the same as in a full run
        CellSpec c;
        c.shape_id = shape;
        c.op = &opSpec(s.ops[draw.op]);
        c.offsets = &offsetSpec(s.offsets[draw.offset_dtype]);
        c.d = s.d_values[draw.d];
        c.capacity_ratio = s.capacity_ratios[draw.capacity];
        // The ARG index width depends on the capacity, which depends on the rows: a 4-byte guess, then check.
        const A::Shape shape_rows =
            A::shapeFor(draw.target_moved_bytes, draw.mean_length, c.d, vb, c.offsets->bytes, 4);
        c.rows = shape_rows.rows;
        c.segments = shape_rows.segments;
        c.capacity_rows = static_cast<uint64_t>(std::ceil(double(c.rows) * c.capacity_ratio));
        if (c.op->arg && c.capacity_rows * c.d >= uint64_t{0xFFFFFFFF}) c.index_dtype = T::DataType::UINT64;
        if (c.offsets->bytes == 4 && c.capacity_rows >= uint64_t{0xFFFFFFFF}) c.offsets = &OFFSETS[1];
        c.rk = ragged && !raggedKREligible(s, c) && raggedRKEligible(s, c);
        if (ragged && !raggedKREligible(s, c) && !c.rk) {
            if (skipped) *skipped += s.distributions.size();
            continue;
        }
        for (A::Distribution dist : s.distributions) {
            c.distribution = dist;
            const int candidates = !s.ragged_kr ? 1  // production and ragged_select: one cell
                                   : c.rk       ? static_cast<int>(s.ragged_rk_shapes.size())
                                                : static_cast<int>(s.ragged_kr_shapes.size());
            for (int k = 0; k < candidates; ++k) {
                c.candidate = s.ragged_kr ? k : -1;
                c.id = out.size();
                out.push_back(c);
            }
        }
    }
    return out;
}

// Fills `ring` with `element` repeated: one H2D copy of a 4 KiB pattern, then device-side doubling.
void fillRing(Ring& ring, const std::vector<uint8_t>& element, Stream& stream) {
    std::vector<uint8_t> pattern(4096);
    for (size_t i = 0; i < pattern.size(); ++i) pattern[i] = element[i % element.size()];
    uint8_t* base = ring.storage().getMemPtr<uint8_t>();
    const uint64_t total = ring.bytes();
    uint64_t filled = std::min<uint64_t>(pattern.size(), total);
    check(cudaMemcpyAsync(base, pattern.data(), filled, cudaMemcpyHostToDevice, stream.getStream()), "ring pattern");
    while (filled < total) {
        const uint64_t chunk = std::min(filled, total - filled);
        check(cudaMemcpyAsync(base + filled, base, chunk, cudaMemcpyDeviceToDevice, stream.getStream()), "ring fill");
        filled += chunk;
    }
    stream.synchronize();
}

struct Cell {
    CellSpec spec;
    std::shared_ptr<T::StampedCubSegmentedReduction> value_stamp;
    std::shared_ptr<T::StampedCubSegmentedArgReduction> arg_stamp;
    // --backend=ragged_kr: the candidate's shape and grid, and the boundary workspace.
    bool ragged = false;
    T::CubReductionInternal::RaggedKRShape ragged_shape;
    uint32_t ragged_grid = 0;
    bool ragged_rk = false;  // 1285: ragged RK on a CTA budget
    T::CubReductionInternal::RaggedRKShape ragged_rk_shape;
    uint32_t ragged_rk_ctas = 0;
    T::CubReductionInternal::RaggedLaunchParts ragged_parts = T::CubReductionInternal::RaggedLaunchParts::MainAndFinish;
    std::optional<T::Tensor> ragged_workspace;
    std::vector<T::Tensor> inputs, offsets, outputs;
    std::optional<T::Tensor> offsets_store;  // owned by the cell: other cells' ring takes never overwrite it
    size_t next_input = 0, next_offsets = 0, next_output = 0;
    A::OffsetStats stats;
    uint64_t input_bytes = 0, offset_bytes = 0, output_bytes = 0, moved_bytes = 0, launches = 1;
    double stamp_ms = 0, probe_us = 0, copy_us = 0;
    std::string passes;
    size_t pass_count = 0;
    int verified = -1;  // 1 passed, 0 failed, -1 not run
    std::vector<std::unique_ptr<TimingEvent>> starts, stops;
};

// 1275: `full` runs both ragged kernels whatever --ragged-kr-time says (the untimed run, so a finish-only cell's
// records are the main kernel's before it is timed).
void runOnce(Cell& cell, Stream& stream, bool full = false) {
    const T::Tensor& input = cell.inputs[cell.next_input];
    const T::Tensor& offsets = cell.offsets[cell.next_offsets];
    const T::Tensor& output = cell.outputs[cell.next_output];
    cell.next_input = (cell.next_input + 1) % cell.inputs.size();
    cell.next_offsets = (cell.next_offsets + 1) % cell.offsets.size();
    cell.next_output = (cell.next_output + 1) % cell.outputs.size();
    if (cell.ragged_rk && cell.spec.op->arg) {  // 1314
        T::Tensor out = output;
        T::CubReductionInternal::launchRaggedRKArg(cell.spec.op->arg_op, cell.ragged_rk_shape, cell.ragged_rk_ctas,
                                                   input, offsets, out, *cell.ragged_workspace, stream,
                                                   full ? T::CubReductionInternal::RaggedLaunchParts::MainAndFinish
                                                        : cell.ragged_parts);
    } else if (cell.ragged && cell.spec.op->arg) {  // 1298
        T::Tensor out = output;
        T::CubReductionInternal::launchRaggedKRArg(cell.spec.op->arg_op, cell.ragged_shape, cell.ragged_grid, input,
                                                   offsets, out, *cell.ragged_workspace, stream,
                                                   full ? T::CubReductionInternal::RaggedLaunchParts::MainAndFinish
                                                        : cell.ragged_parts);
    } else if (cell.ragged_rk) {
        T::Tensor out = output;
        T::CubReductionInternal::launchRaggedRKValue(cell.spec.op->value_op, cell.ragged_rk_shape, cell.ragged_rk_ctas,
                                                     input, offsets, out, *cell.ragged_workspace, stream,
                                                     full ? T::CubReductionInternal::RaggedLaunchParts::MainAndFinish
                                                          : cell.ragged_parts);
    } else if (cell.ragged) {
        T::Tensor out = output;
        T::CubReductionInternal::launchRaggedKRValue(cell.spec.op->value_op, cell.ragged_shape, cell.ragged_grid,
                                                     input, offsets, out, *cell.ragged_workspace, stream,
                                                     full ? T::CubReductionInternal::RaggedLaunchParts::MainAndFinish
                                                          : cell.ragged_parts);
    } else if (cell.value_stamp) {
        cell.value_stamp->runOnTensorsForBenchmark(input, offsets, output, stream);
    } else {
        cell.arg_stamp->runOnTensorsForBenchmark(input, offsets, output, stream);
    }
}

// Runs the stamp on its bound views and checks every segment against its closed form (0.5 in every element).
A::Verification verify(Cell& cell, const std::vector<uint64_t>& host_offsets, Stream& stream) {
    const CellSpec& c = cell.spec;
    if (cell.ragged_rk && c.op->arg) {  // 1314
        T::Tensor out = cell.outputs[0];
        T::CubReductionInternal::launchRaggedRKArg(c.op->arg_op, cell.ragged_rk_shape, cell.ragged_rk_ctas,
                                                   cell.inputs[0], cell.offsets[0], out, *cell.ragged_workspace, stream);
    } else if (cell.ragged && c.op->arg) {  // 1298
        T::Tensor out = cell.outputs[0];
        T::CubReductionInternal::launchRaggedKRArg(c.op->arg_op, cell.ragged_shape, cell.ragged_grid, cell.inputs[0],
                                                   cell.offsets[0], out, *cell.ragged_workspace, stream);
    } else if (cell.ragged_rk) {
        T::Tensor out = cell.outputs[0];
        T::CubReductionInternal::launchRaggedRKValue(c.op->value_op, cell.ragged_rk_shape, cell.ragged_rk_ctas,
                                                     cell.inputs[0], cell.offsets[0], out, *cell.ragged_workspace,
                                                     stream);
    } else if (cell.ragged) {
        T::Tensor out = cell.outputs[0];
        T::CubReductionInternal::launchRaggedKRValue(c.op->value_op, cell.ragged_shape, cell.ragged_grid,
                                                     cell.inputs[0], cell.offsets[0], out, *cell.ragged_workspace,
                                                     stream);
    } else if (cell.value_stamp) {
        cell.value_stamp->run();
    } else {
        cell.arg_stamp->run();
    }
    std::vector<uint8_t> raw(cell.output_bytes);
    check(cudaMemcpyAsync(raw.data(), cell.outputs[0].getMemPtr<void>(), raw.size(), cudaMemcpyDeviceToHost,
                          stream.getStream()),
          "verification copy");
    stream.synchronize();
    if (!c.op->arg) {
        std::vector<float> got(c.segments * c.d);
        std::memcpy(got.data(), raw.data(), got.size() * sizeof(float));
        return A::verifyValueOutput(c.op->op, host_offsets, c.d, 0.5f, got, 1e-3);
    }
    std::vector<uint64_t> got(c.segments * c.d);
    const bool wide = c.index_dtype == T::DataType::UINT64;
    for (size_t i = 0; i < got.size(); ++i) {
        if (wide) {
            std::memcpy(&got[i], raw.data() + 8 * i, 8);
        } else {
            uint32_t v;
            std::memcpy(&v, raw.data() + 4 * i, 4);
            got[i] = v;
        }
    }
    return A::verifyArgOutput(host_offsets, c.d, got, wide ? ~uint64_t{0} : uint64_t{0xFFFFFFFF});
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Settings settings = parse(argc, argv);
        uint64_t skipped = 0;
        const std::vector<CellSpec> specs = cells(settings, &skipped);
        const uint64_t total = std::min<uint64_t>(specs.size(), settings.max_cells);
        const uint64_t vb = A::elementBytes(settings.dtype->format);
        if (settings.list_only) {
            std::cout << "cell_id,shape_id,distribution,operation,dtype,offset_dtype,D,rows,capacity_rows,segments,"
                         "max_len,empty_segments,moved_bytes,candidate\n";
            for (uint64_t n = 0; n < total; ++n) {
                const CellSpec& c = specs[n];
                const A::OffsetStats stats = A::offsetStats(
                    A::makeOffsets(c.rows, c.segments, c.distribution, c.offsetsSeed(settings.seed)));
                std::cout << c.id << ',' << c.shape_id << ',' << A::distributionName(c.distribution) << ','
                          << c.op->name << ',' << settings.dtype->name << ',' << c.offsets->name << ',' << c.d << ','
                          << c.rows << ',' << c.capacity_rows << ',' << c.segments << ',' << stats.max_length << ','
                          << stats.empty_segments << ','
                          << A::movedBytes(c.rows, c.segments, c.d, vb, c.offsets->bytes, c.outputBytesPerElement())
                          << ','
                          << (c.candidate < 0 ? std::string("production")
                              : c.rk          ? T::CubReductionInternal::raggedRKShapeName(
                                                    settings.ragged_rk_shapes[c.candidate])
                                              : T::CubReductionInternal::raggedKRShapeName(
                                                    settings.ragged_kr_shapes[c.candidate]))
                          << '\n';
            }
            return 0;
        }

        const T::TensorPlacement placement(T::TensorPlacement::MemDevices::GPU, 0);
        ScopedGpu scoped_gpu(0);
        Stream stream(placement);
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties, 0), "device properties");
        const T::ScopedCubReductionTuningDepth depth_scope(settings.depth);

        uint64_t max_output = 0, max_moved = 4 * MiB, min_moved = UINT64_MAX;
        for (uint64_t n = 0; n < total; ++n) {
            const CellSpec& c = specs[n];
            max_output = std::max(max_output, c.segments * c.d * c.outputBytesPerElement());
            const uint64_t moved =
                A::movedBytes(c.rows, c.segments, c.d, vb, c.offsets->bytes, c.outputBytesPerElement());
            max_moved = std::max(max_moved, moved);
            min_moved = std::min(min_moved, moved);
        }
        if (min_moved == UINT64_MAX) min_moved = max_moved;
        const uint64_t output_ring_bytes = std::max(settings.output_ring_bytes, 2 * Ring::span(max_output) + 256);
        Ring inputs(placement, settings.arena_bytes);
        Ring outputs(placement, output_ring_bytes);
        const std::vector<uint8_t> half = A::elementBytesOfOneHalf(settings.dtype->format);
        fillRing(inputs, half, stream);
        check(cudaMemsetAsync(outputs.storage().getMemPtr<void>(), 0, output_ring_bytes, stream.getStream()),
              "output ring initialization");
        stream.synchronize();
        const uint64_t copy_bytes = 256 * MiB;
        const double bandwidth = B::ColdTiming::copyBandwidth(inputs, outputs, stream, copy_bytes);
        const CopyCurve curve =
            B::ColdTiming::copyCurve(inputs, outputs, stream, max_moved, settings.interval_us, min_moved);

        std::vector<std::string> distribution_names;
        for (A::Distribution d : settings.distributions) distribution_names.push_back(A::distributionName(d));
        std::cout << "# ragged_reduction_audit=1 version=1265 method=ring_batched_probe dtype=" << settings.dtype->name
                  << " cells=" << total << " shapes=" << settings.shapes << " seed=" << settings.seed
                  << " distributions=" << joinList(distribution_names, ';') << " ops=" << joinList(settings.ops, ';')
                  << " d_values=" << joinList(settings.d_values, ';') << " offsets=" << joinList(settings.offsets, ';')
                  << " capacity_ratios=" << joinList(settings.capacity_ratios, ';')
                  << " min_kib=" << settings.min_bytes / KiB << " max_mib=" << settings.max_mib
                  << " max_mean_length=" << settings.max_mean_length << " samples=" << settings.samples
                  << " interval_us=" << settings.interval_us << " max_launches=" << settings.max_launches
                  << " arena_bytes=" << settings.arena_bytes << " output_ring_bytes=" << output_ring_bytes
                  << " input_reuse_bytes=" << InputReuseBytes << " output_reuse_bytes=" << OutputReuseBytes
                  << " offsets_reuse_bytes=" << settings.offsets_reuse_bytes << " input_value=0.5 value_output=fp32"
                  << " stamp=" << (settings.validated_stamp ? "validated" : "runtime")
                  << " backend="
                  << (settings.ragged_kr ? "ragged_kr" : settings.ragged_select ? "ragged_select" : "production")
                  << " skipped=" << skipped
                  << " ragged_kr_time="
                  << (settings.ragged_parts == T::CubReductionInternal::RaggedLaunchParts::MainOnly     ? "main"
                      : settings.ragged_parts == T::CubReductionInternal::RaggedLaunchParts::FinishOnly ? "finish"
                                                                                                        : "full")
                  << " verify=" << (settings.verify ? 1 : 0)
                  << " tuning_depth=" << T::cubReductionTuningDepthRank(settings.depth) << " device=\""
                  << properties.name << "\" sm_count=" << properties.multiProcessorCount << " cc=" << properties.major
                  << '.' << properties.minor << " l2_bytes=" << properties.l2CacheSize
                  << " copy_bandwidth_tbps=" << std::fixed << std::setprecision(4) << bandwidth / 1e6 << '\n';
        std::cout << "# copy_curve";
        for (const auto& [moved, us] : curve.points) std::cout << ' ' << static_cast<uint64_t>(moved) << '=' << us;
        std::cout << '\n';
        std::vector<uint32_t> ragged_grids, ragged_component_grids;  // 1280: D = 1, and D >= 2
        // 1298: the KR ARG kernels' grids (1300: D = 1, and D >= 2), when the dtype has them.
        std::vector<uint32_t> ragged_arg_grids, ragged_arg_component_grids;
        const bool ragged_arg = T::CubReductionInternal::raggedKRArgSupported(settings.dtype->dtype, 1);
        if (settings.ragged_kr) {
            for (const auto& shape : settings.ragged_kr_shapes) {
                ragged_grids.push_back(T::CubReductionInternal::raggedKRGrid(shape, settings.dtype->dtype, 0));
                ragged_component_grids.push_back(
                    T::CubReductionInternal::raggedKRGrid(shape, settings.dtype->dtype, 0, 2));
                ragged_arg_grids.push_back(
                    ragged_arg ? T::CubReductionInternal::raggedKRArgGrid(shape, settings.dtype->dtype, 0) : 0);
                ragged_arg_component_grids.push_back(
                    ragged_arg ? T::CubReductionInternal::raggedKRArgGrid(shape, settings.dtype->dtype, 0, 2) : 0);
                std::cout << "# ragged_kr_candidate " << T::CubReductionInternal::raggedKRShapeName(shape)
                          << " grid=" << ragged_grids.back()
                          << " workspace_bytes=" << T::CubReductionInternal::raggedValueWorkspaceBytes(ragged_grids.back())
                          << " components_grid=" << ragged_component_grids.back();
                if (ragged_arg)
                    std::cout << " arg_grid=" << ragged_arg_grids.back()
                              << " arg_components_grid=" << ragged_arg_component_grids.back();
                std::cout << '\n';
            }
        }
        std::vector<uint32_t> ragged_rk_ctas;  // 1285: each RK candidate's CTA budget (1295: every dtype)
        std::vector<uint32_t> ragged_rk_arg_ctas;  // 1314: the RK ARG kernels' budget, when the dtype has them
        const bool ragged_rk_arg = T::CubReductionInternal::raggedRKArgSupported(
            settings.dtype->dtype, T::CubReductionInternal::RaggedMergePath::RAGGED_RK_MIN_ROW_BYTES);
        if (settings.ragged_kr) {
            for (const auto& shape : settings.ragged_rk_shapes) {
                ragged_rk_ctas.push_back(T::CubReductionInternal::raggedRKCtas(shape, settings.dtype->dtype, 0));
                ragged_rk_arg_ctas.push_back(
                    ragged_rk_arg ? T::CubReductionInternal::raggedRKArgCtas(shape, settings.dtype->dtype, 0) : 0);
                std::cout << "# ragged_rk_candidate " << T::CubReductionInternal::raggedRKShapeName(shape)
                          << " ctas=" << ragged_rk_ctas.back();
                if (ragged_rk_arg) std::cout << " arg_ctas=" << ragged_rk_arg_ctas.back();
                std::cout << '\n';
            }
        }
        std::cout << "cell_id,shape_id,distribution,kind,operation,dtype,offset_dtype,D,rows,capacity_rows,segments,"
                     "mean_len,max_len,empty_segments,input_bytes,offset_bytes,output_bytes,moved_bytes,stamp_ms,"
                     "passes,pass_count,probe_us,launches_per_sample";
        for (int i = 0; i < settings.samples; ++i) std::cout << ",sample" << i << "_us";
        std::cout << ",median_us,copy_us,tbps,of_copy,verified\n";

        const int passes = settings.samples;
        uint64_t collected = 0, failed = 0, verify_failed = 0;
        const auto wall_start = std::chrono::steady_clock::now();
        B::Fp8OracleBatchQueue queue(settings.queue_cells, passes, [&] { stream.synchronize(); });

        const auto print = [&](const Cell& cell) {
            std::vector<double> us;
            for (int p = 0; p < passes; ++p) {
                float ms = 0;
                check(cudaEventElapsedTime(&ms, cell.starts[p]->handle, cell.stops[p]->handle), "sample elapsed");
                us.push_back(1000.0 * ms / static_cast<double>(cell.launches));
            }
            std::vector<double> sorted = us;
            std::sort(sorted.begin(), sorted.end());
            const double median = sorted[sorted.size() / 2];
            const CellSpec& c = cell.spec;
            const char* kind = c.op->arg ? (c.index_dtype == T::DataType::UINT64 ? "arg_index64" : "arg_index")
                                         : "value";
            std::cout << c.id << ',' << c.shape_id << ',' << A::distributionName(c.distribution) << ',' << kind << ','
                      << c.op->name << ',' << settings.dtype->name << ',' << c.offsets->name << ',' << c.d << ','
                      << c.rows << ',' << c.capacity_rows << ',' << c.segments << ',' << std::setprecision(3)
                      << double(c.rows) / double(c.segments) << ',' << cell.stats.max_length << ','
                      << cell.stats.empty_segments << ',' << cell.input_bytes << ',' << cell.offset_bytes << ','
                      << cell.output_bytes << ',' << cell.moved_bytes << ',' << cell.stamp_ms << ',' << cell.passes
                      << ',' << cell.pass_count << ',' << cell.probe_us << ',' << cell.launches;
            for (double u : us) std::cout << ',' << u;
            std::cout << ',' << median << ',' << cell.copy_us << ',' << std::setprecision(4)
                      << double(cell.moved_bytes) / median / 1e6 << ',' << cell.copy_us / median << ','
                      << cell.verified << '\n';
            ++collected;
            if (collected % 25 == 0 || collected + failed == total) {
                const double wall =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count();
                std::cerr << "[ragged-reduction-audit] cells=" << collected + failed << '/' << total
                          << " failed=" << failed << " verify_failed=" << verify_failed << " wall_s="
                          << std::setprecision(1) << wall << " eta_s="
                          << wall / double(collected + failed) * double(total - collected - failed) << std::endl;
            }
        };

        for (uint64_t n = 0; n < total; ++n) {
            auto cell = std::make_shared<Cell>();
            cell->spec = specs[n];
            const CellSpec& c = cell->spec;
            try {
                const std::vector<uint64_t> host_offsets =
                    A::makeOffsets(c.rows, c.segments, c.distribution, c.offsetsSeed(settings.seed));
                cell->stats = A::offsetStats(host_offsets);

                // Input views: [capacity, D...] of 0.5, of which the offsets make the first `rows` rows active.
                const uint64_t capacity_bytes = c.capacity_rows * c.d * vb;
                cell->input_bytes = c.rows * c.d * vb;
                for (uint64_t v = 0, count = B::ColdTiming::viewCount(capacity_bytes, InputReuseBytes, inputs.bytes());
                     v < count; ++v)
                    cell->inputs.push_back(
                        inputs.view(settings.dtype->dtype, c.valueDims(c.capacity_rows), inputs.take(capacity_bytes)));

                // Offsets copies, owned by the cell: one H2D upload, then device copies.
                cell->offset_bytes = (c.segments + 1) * c.offsets->bytes;
                const uint64_t offsets_span = Ring::span(cell->offset_bytes);
                const uint64_t copies = std::clamp<uint64_t>(
                    B::ColdTiming::ceilDiv(settings.offsets_reuse_bytes, offsets_span), 1, MaxOffsetCopies);
                cell->offsets_store.emplace(placement, T::TensorDescriptor(T::DataType::UINT8, {copies * offsets_span}));
                std::vector<uint8_t> packed(cell->offset_bytes);
                for (size_t i = 0; i < host_offsets.size(); ++i) {
                    if (c.offsets->bytes == 8) {
                        std::memcpy(packed.data() + 8 * i, &host_offsets[i], 8);
                    } else {
                        const uint32_t v = static_cast<uint32_t>(host_offsets[i]);
                        std::memcpy(packed.data() + 4 * i, &v, 4);
                    }
                }
                uint8_t* store = cell->offsets_store->getMemPtr<uint8_t>();
                check(cudaMemcpyAsync(store, packed.data(), packed.size(), cudaMemcpyHostToDevice, stream.getStream()),
                      "offsets upload");
                for (uint64_t k = 1; k < copies; ++k)
                    check(cudaMemcpyAsync(store + k * offsets_span, store, packed.size(), cudaMemcpyDeviceToDevice,
                                          stream.getStream()),
                          "offsets copy");
                for (uint64_t k = 0; k < copies; ++k)
                    cell->offsets.push_back(
                        cell->offsets_store->aliasByteStorageAs(c.offsets->dtype, {c.segments + 1}, k * offsets_span));
                stream.synchronize();  // the upload source is `packed`

                // Output views.
                const T::DataType out_dtype = c.op->arg ? c.index_dtype : T::DataType::FP32;
                cell->output_bytes = c.segments * c.d * c.outputBytesPerElement();
                for (uint64_t v = 0, count = B::ColdTiming::viewCount(cell->output_bytes, OutputReuseBytes,
                                                                      outputs.bytes());
                     v < count; ++v)
                    cell->outputs.push_back(
                        outputs.view(out_dtype, c.valueDims(c.segments), outputs.take(cell->output_bytes)));
                cell->moved_bytes = cell->input_bytes + cell->offset_bytes + cell->output_bytes;
                cell->copy_us = curve.us(double(cell->moved_bytes));

                // The production stamp, bound to the first views (--backend=ragged_kr: the candidate's workspace).
                const auto stamp_start = std::chrono::steady_clock::now();
                if (settings.ragged_select) {  // 1323: the selector's pick for this cell's site
                    namespace R = T::CubReductionInternal;
                    R::RaggedSite site;
                    site.arg = c.op->arg;
                    if (c.op->arg) site.arg_op = c.op->arg_op;
                    else site.value_op = c.op->value_op;
                    site.input_dtype = settings.dtype->dtype;
                    site.offsets_dtype = c.offsets->dtype;
                    site.output_dtype = out_dtype;
                    site.components = c.d;
                    site.capacity_rows = c.capacity_rows;
                    site.segments = c.segments;
                    site.input_head =
                        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(cell->inputs[0].getMemPtr<void>()) & 15U);
                    R::DenseReductionSelectionCache& cache = R::denseReductionSelectionCache();
                    const uint64_t attempts = cache.tuningAttempts(R::DenseReductionSelectionKind::RaggedKR) +
                                              cache.tuningAttempts(R::DenseReductionSelectionKind::RaggedRK);
                    const R::RaggedSelection selection = R::selectCachedRagged(site, stream);
                    const bool tuned = cache.tuningAttempts(R::DenseReductionSelectionKind::RaggedKR) +
                                           cache.tuningAttempts(R::DenseReductionSelectionKind::RaggedRK) !=
                                       attempts;
                    const R::RaggedSelectedLaunch launch = R::raggedSelectedLaunch(selection, site, 0);
                    cell->ragged = true;
                    cell->ragged_parts = settings.ragged_parts;
                    if (selection.family == R::RaggedFamily::RK) {
                        cell->ragged_rk = true;
                        cell->ragged_rk_shape = selection.rk_shape;
                        cell->ragged_rk_ctas = launch.grid_or_ctas;
                    } else {
                        cell->ragged_shape = selection.kr_shape;
                        cell->ragged_grid = launch.grid_or_ctas;
                    }
                    cell->ragged_workspace.emplace(placement,
                                                   T::TensorDescriptor(T::DataType::UINT8, {launch.workspace_bytes}));
                    cell->passes = selection.winner_name + (c.op->arg ? "+ragged_finish_arg" : "+ragged_finish");
                    cell->pass_count = 2;
                    std::cout << "# ragged_select," << c.id << ',' << c.shape_id << ','
                              << A::distributionName(c.distribution) << ',' << selection.winner_name
                              << ",tuned=" << (tuned ? 1 : 0) << ",head=" << site.input_head << ",contest=";
                    for (size_t i = 0; i < selection.contest_names.size(); ++i)
                        std::cout << (i ? ";" : "") << selection.contest_names[i] << ':' << selection.contest_us[i];
                    std::cout << '\n';
                } else if (c.candidate >= 0 && c.rk && c.op->arg) {  // 1314
                    cell->ragged = true;
                    cell->ragged_rk = true;
                    cell->ragged_rk_shape = settings.ragged_rk_shapes[c.candidate];
                    cell->ragged_parts = settings.ragged_parts;
                    cell->ragged_rk_ctas = ragged_rk_arg_ctas[c.candidate];
                    const T::CubReductionInternal::RaggedRKLaunch launch = T::CubReductionInternal::raggedRKArgLaunch(
                        cell->ragged_rk_shape, cell->ragged_rk_ctas, settings.dtype->dtype, c.d, c.capacity_rows);
                    cell->ragged_workspace.emplace(placement,
                                                   T::TensorDescriptor(T::DataType::UINT8, {launch.workspace_bytes}));
                    cell->passes =
                        T::CubReductionInternal::raggedRKShapeName(cell->ragged_rk_shape) + "+ragged_finish_arg";
                    cell->pass_count = 2;
                } else if (c.candidate >= 0 && c.rk) {  // 1285
                    cell->ragged = true;
                    cell->ragged_rk = true;
                    cell->ragged_rk_shape = settings.ragged_rk_shapes[c.candidate];
                    cell->ragged_parts = settings.ragged_parts;
                    cell->ragged_rk_ctas = ragged_rk_ctas[c.candidate];
                    const T::CubReductionInternal::RaggedRKLaunch launch = T::CubReductionInternal::raggedRKLaunch(
                        cell->ragged_rk_shape, cell->ragged_rk_ctas, settings.dtype->dtype, c.d, c.capacity_rows);
                    cell->ragged_workspace.emplace(placement,
                                                   T::TensorDescriptor(T::DataType::UINT8, {launch.workspace_bytes}));
                    cell->passes = T::CubReductionInternal::raggedRKShapeName(cell->ragged_rk_shape) + "+ragged_finish";
                    cell->pass_count = 2;
                } else if (c.candidate >= 0 && c.op->arg) {  // 1298
                    cell->ragged = true;
                    cell->ragged_shape = settings.ragged_kr_shapes[c.candidate];
                    cell->ragged_parts = settings.ragged_parts;
                    cell->ragged_grid = (c.d == 1 ? ragged_arg_grids : ragged_arg_component_grids)[c.candidate];
                    cell->ragged_workspace.emplace(
                        placement, T::TensorDescriptor(T::DataType::UINT8,
                                                       {T::CubReductionInternal::raggedArgWorkspaceBytes(
                                                           settings.dtype->dtype, cell->ragged_grid,
                                                           static_cast<uint32_t>(c.d))}));
                    cell->passes =
                        T::CubReductionInternal::raggedKRShapeName(cell->ragged_shape) + "+ragged_finish_arg";
                    cell->pass_count = 2;
                } else if (c.candidate >= 0) {
                    cell->ragged = true;
                    cell->ragged_shape = settings.ragged_kr_shapes[c.candidate];
                    cell->ragged_parts = settings.ragged_parts;
                    cell->ragged_grid = (c.d == 1 ? ragged_grids : ragged_component_grids)[c.candidate];
                    cell->ragged_workspace.emplace(
                        placement, T::TensorDescriptor(T::DataType::UINT8,
                                                       {T::CubReductionInternal::raggedValueWorkspaceBytes(
                                                           cell->ragged_grid, static_cast<uint32_t>(c.d))}));
                    cell->passes = T::CubReductionInternal::raggedKRShapeName(cell->ragged_shape) + "+ragged_finish";
                    cell->pass_count = 2;
                } else if (!c.op->arg) {
                    const T::CubSegmentedReduction reduction(c.op->value_op, T::DataType::FP32);
                    cell->value_stamp =
                        settings.validated_stamp
                            ? reduction.stamp(cell->inputs[0], cell->outputs[0], cell->offsets[0], stream)
                            : reduction.stampRuntimeOffsets(cell->inputs[0], cell->outputs[0], cell->offsets[0], stream);
                    for (const std::string& pass : cell->value_stamp->getPassNames()) {
                        cell->passes += (cell->passes.empty() ? "" : "+") + pass;
                        ++cell->pass_count;
                    }
                } else {
                    const T::CubSegmentedArgReduction reduction(c.op->arg_op, c.index_dtype);
                    cell->arg_stamp =
                        settings.validated_stamp
                            ? reduction.stamp(cell->inputs[0], cell->outputs[0], cell->offsets[0], stream)
                            : reduction.stampRuntimeOffsets(cell->inputs[0], cell->outputs[0], cell->offsets[0], stream);
                    for (const std::string& pass : cell->arg_stamp->getPassNames()) {
                        cell->passes += (cell->passes.empty() ? "" : "+") + pass;
                        ++cell->pass_count;
                    }
                }
                stream.synchronize();
                cell->stamp_ms =
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - stamp_start).count();
                std::replace(cell->passes.begin(), cell->passes.end(), ',', ';');

                if (settings.verify) {
                    const A::Verification v = verify(*cell, host_offsets, stream);
                    cell->verified = v.ok() ? 1 : 0;
                    if (!v.ok()) {
                        ++verify_failed;
                        std::cout << "# verify_failed," << c.id << ',' << c.shape_id << ','
                                  << A::distributionName(c.distribution) << ',' << c.op->name << ",D=" << c.d
                                  << ",rows=" << c.rows << ",segments=" << c.segments << ",mismatches="
                                  << v.mismatches << '/' << v.checked << ',' << v.first << '\n';
                    }
                }

                // Untimed run (both ragged kernels), then the timed probe.
                runOnce(*cell, stream, true);
                TimingEvent start, stop;
                check(cudaEventRecord(start.handle, stream.getStream()), "probe start");
                runOnce(*cell, stream);
                check(cudaEventRecord(stop.handle, stream.getStream()), "probe stop");
                stream.synchronize();
                float ms = 0;
                check(cudaEventElapsedTime(&ms, start.handle, stop.handle), "probe elapsed");
                cell->probe_us = 1000.0 * ms;
                cell->launches =
                    B::ColdTiming::launchesPerSample(cell->probe_us, settings.interval_us, settings.max_launches);
                for (int p = 0; p < passes; ++p) {
                    cell->starts.push_back(std::make_unique<TimingEvent>());
                    cell->stops.push_back(std::make_unique<TimingEvent>());
                }
            } catch (const std::exception& e) {
                ++failed;
                std::string what = e.what();
                std::replace(what.begin(), what.end(), '\n', ' ');
                std::cout << "# failed," << c.id << ',' << c.shape_id << ',' << A::distributionName(c.distribution)
                          << ',' << c.op->name << ",D=" << c.d << ",rows=" << c.rows << ",segments=" << c.segments
                          << ',' << what << '\n';
                continue;
            }
            queue.add({[] {},
                       [cell, &stream](int pass) {
                           check(cudaEventRecord(cell->starts[pass]->handle, stream.getStream()), "start event");
                           for (uint64_t l = 0; l < cell->launches; ++l) runOnce(*cell, stream);
                           check(cudaEventRecord(cell->stops[pass]->handle, stream.getStream()), "stop event");
                       },
                       [cell, &print] { print(*cell); }});
        }
        queue.flush();
        const double bandwidth_end = B::ColdTiming::copyBandwidth(inputs, outputs, stream, copy_bytes);
        const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count();
        std::cout << "# end cells=" << collected << " failed=" << failed << " verify_failed=" << verify_failed
                  << " skipped=" << skipped
                  << " batches=" << queue.batches() << " copy_bandwidth_tbps_end=" << std::setprecision(4)
                  << bandwidth_end / 1e6 << " input_ring_wraps=" << inputs.wraps()
                  << " output_ring_wraps=" << outputs.wraps() << " wall_s=" << std::setprecision(1) << wall << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "thor_ragged_reduction_audit: " << e.what() << '\n';
        return 1;
    }
}
