// 1174: thor_dense_production_audit. What a user gets from a production CubReduction / CubArgReduction stamp at a
// dense R, KR or RK site of any input dtype, timed cold, against the copy curve of the same GPU, in TB/s.
//
// It is the first step of converting a dtype to the FP8 pattern (Utilities/TensorOperations/Cub/
// FP8_REDUCTION_PATTERN.md section 5): the baseline. It is also the before/after check of every later step of that
// dtype, and of FP8 when the FP32 path it continues into changes.
//
// Method (benchmarks/ColdTiming.h, as the FP8 sweeps): each cell is stamped once on its first input view, the stamp a
// user gets (tuning at --tuning-depth). The stamp is then run with StampedCub*Reduction::runOnTensorsForBenchmark on
// rotating input views spanning at least 2 GiB and output views spanning at least 1 GiB, so every launch reads a
// cold input and pays its output's write-back; the stamp's own intermediates and scratch are reused, as in a real run.
// One untimed run, one timed probe, then --samples samples of about --interval-us of back-to-back runs (at most
// --max-launches), interleaved across the cells of a batch. Moved bytes are the input plus the outputs; the reference
// is the copy curve (time of a device copy moving the same bytes, measured the same way at startup).
//
// Cells: --cells random geometries per family (log-uniform input size from --min-kib to --max-mib; KR: row length R
// log-uniform; RK: K log-uniform from 2 to 4096, then R), each with one operation drawn from --ops. ARG cells write a
// UINT32 index, and with --arg-outputs=value_index also an FP32 value. VALUE output is FP32.
//
// Output: '#' header lines (settings, device, copy curve) and one CSV row per cell. benchmarks/dense_production_audit.py
// reports it.
#include "benchmarks/ColdTiming.h"
#include "benchmarks/Fp8OracleBatchQueue.h"
#include "Utilities/TensorOperations/Cub/CubReduction.h"
#include "Utilities/TensorOperations/Cub/CubReductionTuningDepth.h"
#include "Utilities/Common/ScopedGpu.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
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

void check(cudaError_t status, const char* what) { B::ColdTiming::checkCuda(status, what); }

struct OpSpec {
    const char* name;
    bool arg;
    T::CubReductionOp value_op;
    T::CubArgReductionOp arg_op;
};
constexpr OpSpec OPS[] = {
    {"sum", false, T::CubReductionOp::Sum, T::CubArgReductionOp::ArgMax},
    {"mean", false, T::CubReductionOp::Mean, T::CubArgReductionOp::ArgMax},
    {"product", false, T::CubReductionOp::Product, T::CubArgReductionOp::ArgMax},
    {"min", false, T::CubReductionOp::Min, T::CubArgReductionOp::ArgMax},
    {"max", false, T::CubReductionOp::Max, T::CubArgReductionOp::ArgMax},
    {"l1_norm", false, T::CubReductionOp::L1Norm, T::CubArgReductionOp::ArgMax},
    {"l2_norm", false, T::CubReductionOp::L2Norm, T::CubArgReductionOp::ArgMax},
    {"sum_squares", false, T::CubReductionOp::SumSquares, T::CubArgReductionOp::ArgMax},
    {"argmin", true, T::CubReductionOp::Min, T::CubArgReductionOp::ArgMin},
    {"argmax", true, T::CubReductionOp::Max, T::CubArgReductionOp::ArgMax},
};

const OpSpec& opSpec(std::string_view name) {
    for (const OpSpec& op : OPS)
        if (name == op.name) return op;
    throw std::invalid_argument("--ops takes sum,mean,product,min,max,l1_norm,l2_norm,sum_squares,argmin,argmax");
}

struct DTypeSpec {
    const char* name;
    T::DataType dtype;
    uint64_t bytes;
};
constexpr DTypeSpec DTYPES[] = {
    {"fp32", T::DataType::FP32, 4},          {"fp16", T::DataType::FP16, 2},
    {"bf16", T::DataType::BF16, 2},          {"fp8_e4m3", T::DataType::FP8_E4M3, 1},
    {"fp8_e5m2", T::DataType::FP8_E5M2, 1},
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
    std::vector<std::string> families{"R", "KR", "RK"};
    std::vector<std::string> ops{"sum", "max", "argmax"};
    bool arg_value_index = false;
    uint64_t cells_per_family = 200;
    uint64_t seed = 1174;
    uint64_t min_bytes = 64 * KiB;
    uint64_t max_mib = 1024;
    uint64_t max_cells = UINT64_MAX;
    int samples = 5;
    double interval_us = 1000.0;
    uint64_t max_launches = 256;
    uint64_t arena_bytes = 8 * GiB;
    uint64_t output_ring_bytes = 3 * GiB;
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
        } else if (auto v = value("--families=")) {
            s.families = splitList(*v);
            for (const std::string& f : s.families)
                if (f != "R" && f != "KR" && f != "RK") throw std::invalid_argument("--families takes R,KR,RK");
        } else if (auto v = value("--ops=")) {
            s.ops = splitList(*v);
            for (const std::string& op : s.ops) (void)opSpec(op);
        } else if (auto v = value("--arg-outputs=")) {
            if (*v == "index") s.arg_value_index = false;
            else if (*v == "value_index") s.arg_value_index = true;
            else throw std::invalid_argument("--arg-outputs takes index or value_index");
        } else if (auto v = value("--cells=")) {
            s.cells_per_family = std::stoull(*v);
        } else if (auto v = value("--seed=")) {
            s.seed = std::stoull(*v);
        } else if (auto v = value("--min-kib=")) {
            s.min_bytes = std::stoull(*v) * KiB;
        } else if (auto v = value("--max-mib=")) {
            s.max_mib = std::stoull(*v);
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
            std::cout << "thor_dense_production_audit [--dtype=fp32] [--families=R,KR,RK] [--ops=sum,max,argmax]\n"
                         "    [--arg-outputs=index|value_index] [--cells=200] [--seed=1174] [--min-kib=64]\n"
                         "    [--max-mib=1024] [--max-cells=N] [--samples=5] [--interval-us=1000] [--max-launches=256]\n"
                         "    [--arena-gib=8] [--output-ring-gib=3] [--queue-cells=16] [--tuning-depth=regular]\n"
                         "    [--list-only]\n"
                         "--cells is per family; ops are drawn per cell. CSV on stdout, progress on stderr.\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("Unknown argument " + std::string(a));
        }
    }
    if (s.families.empty() || s.ops.empty()) throw std::invalid_argument("--families and --ops must not be empty");
    if (s.min_bytes < 4 * KiB || s.min_bytes > s.max_mib * MiB)
        throw std::invalid_argument("need 4 KiB <= --min-kib <= --max-mib");
    if (s.arena_bytes < 4 * s.max_mib * MiB)
        throw std::invalid_argument("--arena-gib must hold at least four inputs of --max-mib");
    return s;
}

uint64_t mix(uint64_t x) {  // splitmix64 finalizer: deterministic across platforms
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

struct CellSpec {
    uint64_t id = 0;
    std::string family;  // R, KR, RK
    const OpSpec* op = nullptr;
    uint64_t outer = 1, r = 1, k = 1;  // R: [r]; KR: [outer, r] over r; RK: [outer, r, k] over r
    [[nodiscard]] uint64_t elements() const { return outer * r * k; }
    [[nodiscard]] std::vector<uint64_t> dims() const {
        if (family == "R") return {r};
        if (family == "KR") return {outer, r};
        return {outer, r, k};
    }
    [[nodiscard]] uint32_t axis() const { return family == "R" ? 0 : 1; }
    [[nodiscard]] uint64_t outputs() const { return outer * k; }
};

std::vector<CellSpec> cells(const Settings& s) {
    std::vector<CellSpec> out;
    const uint64_t eb = s.dtype->bytes;
    uint64_t id = 0;
    for (size_t f = 0; f < s.families.size(); ++f) {
        for (uint64_t n = 0; n < s.cells_per_family; ++n, ++id) {
            const uint64_t h = mix(s.seed ^ mix(id * 0x100 + f));
            CellSpec c;
            c.id = id;
            c.family = s.families[f];
            c.op = &opSpec(s.ops[mix(h ^ 1) % s.ops.size()]);
            const uint64_t elements = std::max<uint64_t>(2, logUniform(mix(h ^ 2), s.min_bytes, s.max_mib * MiB) / eb);
            if (c.family == "R") {
                c.r = elements;
            } else if (c.family == "KR") {
                c.r = logUniform(mix(h ^ 3), 2, std::min<uint64_t>(elements, uint64_t{1} << 24));
                c.outer = std::max<uint64_t>(1, elements / c.r);
            } else {
                c.k = logUniform(mix(h ^ 4), 2, std::max<uint64_t>(2, std::min<uint64_t>(4096, elements / 2)));
                c.r = logUniform(mix(h ^ 5), 2, std::max<uint64_t>(2, elements / c.k));
                c.outer = std::max<uint64_t>(1, elements / (c.r * c.k));
            }
            out.push_back(c);
        }
    }
    return out;
}

const char* pathName(T::CubReductionPath path) {
    switch (path) {
        case T::CubReductionPath::DeviceTransformReduce: return "direct:device_transform_reduce";
        case T::CubReductionPath::ContiguousFixedSegment: return "direct:contiguous_fixed_segment";
        case T::CubReductionPath::TiledFixedSegment: return "direct:tiled_fixed_segment";
        case T::CubReductionPath::OffsetSegmented: return "direct:offset_segmented";
        case T::CubReductionPath::ComposedDense: return "composed_dense";
    }
    return "unknown";
}

struct Cell {
    CellSpec spec;
    std::shared_ptr<T::StampedCubReduction> value_stamp;
    std::shared_ptr<T::StampedCubArgReduction> arg_stamp;
    std::vector<T::Tensor> inputs;
    std::vector<std::optional<T::Tensor>> values, indices;
    size_t next_input = 0, next_output = 0;
    uint64_t input_bytes = 0, moved_bytes = 0, launches = 1;
    double stamp_ms = 0, probe_us = 0, copy_us = 0;
    std::string passes;
    size_t pass_count = 0;
    std::vector<std::unique_ptr<TimingEvent>> starts, stops;
};

void runOnce(Cell& cell, Stream& stream) {
    const T::Tensor& input = cell.inputs[cell.next_input];
    cell.next_input = (cell.next_input + 1) % cell.inputs.size();
    const size_t o = cell.next_output;
    cell.next_output = (cell.next_output + 1) % cell.values.size();
    if (cell.value_stamp) {
        cell.value_stamp->runOnTensorsForBenchmark(input, *cell.values[o], stream);
    } else {
        cell.arg_stamp->runOnTensorsForBenchmark(input, cell.values[o], cell.indices[o], stream);
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Settings settings = parse(argc, argv);
        const std::vector<CellSpec> specs = cells(settings);
        const uint64_t total = std::min<uint64_t>(specs.size(), settings.max_cells);
        if (settings.list_only) {
            std::cout << "cell_id,family,operation,dtype,outer,R,K\n";
            for (uint64_t n = 0; n < total; ++n) {
                const CellSpec& c = specs[n];
                std::cout << c.id << ',' << c.family << ',' << c.op->name << ',' << settings.dtype->name << ','
                          << c.outer << ',' << c.r << ',' << c.k << '\n';
            }
            return 0;
        }

        const T::TensorPlacement placement(T::TensorPlacement::MemDevices::GPU, 0);
        ScopedGpu scoped_gpu(0);
        Stream stream(placement);
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties, 0), "device properties");
        const T::ScopedCubReductionTuningDepth depth_scope(settings.depth);

        // Outputs: value FP32 [outputs], then (256-byte aligned) index UINT32 [outputs].
        const auto footprint = [&](const CellSpec& c) {
            const uint64_t value = (!c.op->arg || settings.arg_value_index) ? align256(4 * c.outputs()) : 0;
            const uint64_t index = c.op->arg ? 4 * c.outputs() : 0;
            return value + index;
        };
        uint64_t max_output = 0, max_moved = 4 * MiB;
        for (uint64_t n = 0; n < total; ++n) {
            max_output = std::max(max_output, footprint(specs[n]));
            max_moved = std::max(max_moved, specs[n].elements() * settings.dtype->bytes + footprint(specs[n]));
        }
        const uint64_t output_ring_bytes = std::max(settings.output_ring_bytes, 2 * Ring::span(max_output) + 256);
        Ring inputs(placement, settings.arena_bytes);
        Ring outputs(placement, output_ring_bytes);
        check(cudaMemsetAsync(inputs.storage().getMemPtr<void>(), 0x30, settings.arena_bytes, stream.getStream()),
              "input ring initialization");
        check(cudaMemsetAsync(outputs.storage().getMemPtr<void>(), 0, output_ring_bytes, stream.getStream()),
              "output ring initialization");
        stream.synchronize();
        const uint64_t copy_bytes = 256 * MiB;
        const double bandwidth = B::ColdTiming::copyBandwidth(inputs, outputs, stream, copy_bytes);
        // 1176: the curve starts at the smallest cell, so small cells are compared with a copy of their own size
        // (1175 extrapolated below 2 MiB, putting their reference far under the launch floor).
        uint64_t min_moved = max_moved;
        for (uint64_t n = 0; n < total; ++n)
            min_moved = std::min(min_moved, specs[n].elements() * settings.dtype->bytes + footprint(specs[n]));
        const CopyCurve curve =
            B::ColdTiming::copyCurve(inputs, outputs, stream, max_moved, settings.interval_us, min_moved);

        std::cout << "# dense_production_audit=1 version=1176 method=ring_batched_probe dtype=" << settings.dtype->name
                  << " cells=" << total << " cells_per_family=" << settings.cells_per_family
                  << " seed=" << settings.seed << " min_kib=" << settings.min_bytes / KiB
                  << " max_mib=" << settings.max_mib << " samples=" << settings.samples
                  << " interval_us=" << settings.interval_us << " max_launches=" << settings.max_launches
                  << " arena_bytes=" << settings.arena_bytes << " output_ring_bytes=" << output_ring_bytes
                  << " input_reuse_bytes=" << InputReuseBytes << " output_reuse_bytes=" << OutputReuseBytes
                  << " input_byte=0x30 value_output=fp32 arg_index=uint32 arg_outputs="
                  << (settings.arg_value_index ? "value_index" : "index")
                  << " tuning_depth=" << T::cubReductionTuningDepthRank(settings.depth)
                  << " device=\"" << properties.name << "\" sm_count=" << properties.multiProcessorCount
                  << " cc=" << properties.major << '.' << properties.minor << " l2_bytes=" << properties.l2CacheSize
                  << " copy_bandwidth_tbps=" << std::fixed << std::setprecision(4) << bandwidth / 1e6 << '\n';
        std::cout << "# copy_curve";
        for (const auto& [moved, us] : curve.points) std::cout << ' ' << static_cast<uint64_t>(moved) << '=' << us;
        std::cout << '\n';
        std::cout << "cell_id,family,kind,operation,dtype,outer,R,K,input_bytes,moved_bytes,stamp_ms,passes,pass_count,"
                     "probe_us,launches_per_sample";
        for (int i = 0; i < settings.samples; ++i) std::cout << ",sample" << i << "_us";
        std::cout << ",median_us,copy_us,tbps,of_copy\n";

        const int passes = settings.samples;
        uint64_t collected = 0, failed = 0;
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
            const char* kind = !c.op->arg ? "value" : (settings.arg_value_index ? "arg_value_index" : "arg_index");
            std::cout << c.id << ',' << c.family << ',' << kind << ',' << c.op->name << ',' << settings.dtype->name
                      << ',' << c.outer << ',' << c.r << ',' << c.k << ',' << cell.input_bytes << ','
                      << cell.moved_bytes << ',' << std::setprecision(3) << cell.stamp_ms << ',' << cell.passes << ','
                      << cell.pass_count << ',' << cell.probe_us << ',' << cell.launches;
            for (double u : us) std::cout << ',' << u;
            std::cout << ',' << median << ',' << cell.copy_us << ',' << std::setprecision(4)
                      << double(cell.moved_bytes) / median / 1e6 << ',' << cell.copy_us / median << '\n';
            ++collected;
            if (collected % 25 == 0 || collected + failed == total) {
                const double wall =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count();
                std::cerr << "[dense-production-audit] cells=" << collected + failed << '/' << total
                          << " failed=" << failed << " wall_s=" << std::setprecision(1) << wall << " eta_s="
                          << wall / double(collected + failed) * double(total - collected - failed) << std::endl;
            }
        };

        for (uint64_t n = 0; n < total; ++n) {
            auto cell = std::make_shared<Cell>();
            cell->spec = specs[n];
            const CellSpec& c = cell->spec;
            try {
                cell->input_bytes = c.elements() * settings.dtype->bytes;
                for (uint64_t v = 0, count = B::ColdTiming::viewCount(cell->input_bytes, InputReuseBytes,
                                                                      inputs.bytes());
                     v < count; ++v)
                    cell->inputs.push_back(inputs.view(settings.dtype->dtype, c.dims(), inputs.take(cell->input_bytes)));
                const uint64_t out_bytes = footprint(c);
                const bool has_value = !c.op->arg || settings.arg_value_index;
                const uint64_t index_offset = has_value ? align256(4 * c.outputs()) : 0;
                std::vector<uint64_t> out_dims = c.dims();
                out_dims.erase(out_dims.begin() + c.axis());
                if (out_dims.empty()) out_dims.push_back(1);
                for (uint64_t v = 0, count = B::ColdTiming::viewCount(out_bytes, OutputReuseBytes, outputs.bytes());
                     v < count; ++v) {
                    const uint64_t base = outputs.take(out_bytes);
                    cell->values.push_back(has_value ? std::optional<T::Tensor>(
                                                           outputs.view(T::DataType::FP32, out_dims, base))
                                                     : std::nullopt);
                    cell->indices.push_back(c.op->arg ? std::optional<T::Tensor>(outputs.view(
                                                            T::DataType::UINT32, out_dims, base + index_offset))
                                                      : std::nullopt);
                }
                cell->moved_bytes = cell->input_bytes + 4 * c.outputs() * ((c.op->arg && has_value) ? 2 : 1);
                cell->copy_us = curve.us(double(cell->moved_bytes));

                // The production stamp (its own selection and tuning), bound to the first views.
                const auto stamp_start = std::chrono::steady_clock::now();
                if (!c.op->arg) {
                    cell->value_stamp =
                        T::CubReduction(c.op->value_op, c.axis(), T::DataType::FP32).stamp(cell->inputs[0], stream);
                    if (cell->value_stamp->usesDenseExecutablePlan()) {
                        for (const auto& pass : cell->value_stamp->getDensePhysicalPassDiagnostics()) {
                            cell->passes += (cell->passes.empty() ? "" : "+") + pass.family;
                            ++cell->pass_count;
                        }
                    } else {
                        cell->passes = pathName(cell->value_stamp->getPath());
                        cell->pass_count = 1;
                    }
                } else {
                    T::CubArgReductionOutputOptions options;
                    options.produce_value = has_value;
                    if (has_value) options.value_output_dtype = T::DataType::FP32;
                    options.produce_index = true;
                    options.index_output_dtype = T::DataType::UINT32;
                    cell->arg_stamp = T::CubArgReduction(c.op->arg_op, c.axis(), options).stamp(cell->inputs[0], stream);
                    for (const std::string& pass : cell->arg_stamp->getPlannedPassNames()) {
                        cell->passes += (cell->passes.empty() ? "" : "+") + pass;
                        ++cell->pass_count;
                    }
                    if (cell->passes.empty()) {
                        cell->passes = pathName(cell->arg_stamp->getPath());
                        cell->pass_count = 1;
                    }
                }
                stream.synchronize();
                cell->stamp_ms =
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - stamp_start).count();
                std::replace(cell->passes.begin(), cell->passes.end(), ',', ';');

                // Untimed run (loads the kernels), then the timed probe.
                runOnce(*cell, stream);
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
                std::cout << "# failed," << c.id << ',' << c.family << ',' << c.op->name << ",outer=" << c.outer
                          << ",R=" << c.r << ",K=" << c.k << ',' << what << '\n';
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
        std::cout << "# end cells=" << collected << " failed=" << failed << " batches=" << queue.batches()
                  << " copy_bandwidth_tbps_end=" << std::setprecision(4) << bandwidth_end / 1e6
                  << " input_ring_wraps=" << inputs.wraps() << " output_ring_wraps=" << outputs.wraps()
                  << " wall_s=" << std::setprecision(1) << wall << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "thor_dense_production_audit: " << e.what() << '\n';
        return 1;
    }
}
