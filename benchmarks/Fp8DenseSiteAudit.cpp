// FP8 R / KR production audit (1145-1148).
//
// 1152: superseded for timing by thor_fp8_kr_training_sweep, which times the RK sweep's way (ring inputs and outputs,
// asynchronous batches) and records the production pick. This audit's flush-per-sample timing with a shared output
// disagreed with itself by up to 28% (the same shape timed as the production stamp and as a candidate in one run).
//
// 1148: CUDA events on the 1147 GPU resolve only about 2 us (every time below 20 us was a multiple of 2.048 us). A sample
// now flushes L2, then times `batch` back-to-back launches, each on its own never-read input slot of the arena (so every
// launch is still L2-cold), and divides by the batch. The batch targets about 200 us per sample at line rate (up to 32;
// 1150: 40 us left 64 MiB cells on a 5% grid).
//
// Times what a user gets from CubReduction / CubArgReduction stamps at FP8 R sites ([N], axis 0) and KR sites
// ([K, R], axis 1), against line rate. Every timed launch starts with a cold L2: before each sample a separate
// production SUM reduction reads a flush buffer of several L2 sizes. A read-only flush leaves no dirty lines, so the
// timed launch pays no write-back for the flush. The reference is max(launch floor, (input + output bytes) / B), where
// B is the device-to-device copy bandwidth (read + write bytes per second) measured once at startup (1146: the output
// is counted too; at R = 2 a VALUE result is twice its input). stamp_ms is the stamp's wall time, which includes the
// retained R / KR autotuning on a selection-cache miss. No experimental kernels.
#include "Utilities/TensorOperations/Cub/CubReduction.h"
#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"
#include "Utilities/TensorOperations/Cub/CubReductionStageSemantics.h"
#include "Utilities/TensorOperations/Cub/Fp8KR/Fp8KRSegments.h"
#include "Utilities/Common/ScopedGpu.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
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

namespace {

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

constexpr uint64_t KiB = 1ULL << 10;
constexpr uint64_t MiB = 1ULL << 20;

struct Settings {
    std::vector<std::string> ops{"sum", "max", "argmax"};
    std::vector<T::DataType> dtypes{T::DataType::FP8_E4M3};
    std::vector<std::string> families{"R", "KR"};
    uint64_t max_bytes = 1024 * MiB;
    int samples = 11;
    double floor_us = 4.0;
    uint64_t flush_l2_multiple = 4;
    bool candidates = false;  // 1147: also time every retained R / KR candidate shape
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

Settings parse(int argc, char** argv) {
    Settings s;
    for (int i = 1; i < argc; ++i) {
        const std::string_view a(argv[i]);
        const auto value = [&](std::string_view prefix) -> std::optional<std::string_view> {
            if (a.substr(0, prefix.size()) == prefix) return a.substr(prefix.size());
            return std::nullopt;
        };
        if (auto v = value("--ops=")) {
            s.ops = splitList(*v);
            for (const std::string& op : s.ops)
                if (op != "sum" && op != "max" && op != "argmax" && op != "argmin")
                    throw std::invalid_argument("--ops takes sum,max,argmax,argmin");
        } else if (auto v = value("--dtypes=")) {
            s.dtypes.clear();
            for (const std::string& d : splitList(*v)) {
                if (d == "fp8_e4m3") s.dtypes.push_back(T::DataType::FP8_E4M3);
                else if (d == "fp8_e5m2") s.dtypes.push_back(T::DataType::FP8_E5M2);
                else throw std::invalid_argument("--dtypes takes fp8_e4m3,fp8_e5m2");
            }
        } else if (auto v = value("--families=")) {
            s.families = splitList(*v);
            for (const std::string& f : s.families)
                if (f != "R" && f != "KR") throw std::invalid_argument("--families takes R,KR");
        } else if (auto v = value("--max-mib=")) {
            s.max_bytes = std::stoull(std::string(*v)) * MiB;
        } else if (auto v = value("--samples=")) {
            s.samples = std::max(3, std::stoi(std::string(*v)));
        } else if (auto v = value("--floor-us=")) {
            s.floor_us = std::stod(std::string(*v));
        } else if (a == "--candidates") {
            s.candidates = true;
        } else if (a == "--help") {
            std::cout << "thor_fp8_dense_site_audit [--ops=sum,max,argmax] [--dtypes=fp8_e4m3,fp8_e5m2]\n"
                         "    [--families=R,KR] [--max-mib=1024] [--samples=11] [--floor-us=4] [--candidates]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("Unknown argument " + std::string(a));
        }
    }
    if (s.max_bytes < 4 * MiB) throw std::invalid_argument("--max-mib must be at least 4");
    return s;
}

const char* dtypeName(T::DataType dtype) { return dtype == T::DataType::FP8_E5M2 ? "fp8_e5m2" : "fp8_e4m3"; }

const char* pathName(T::CubReductionPath path) {
    switch (path) {
        case T::CubReductionPath::DeviceTransformReduce: return "cub_device_reduce";
        case T::CubReductionPath::ContiguousFixedSegment: return "contiguous_fixed_segment";
        case T::CubReductionPath::TiledFixedSegment: return "tiled_fixed_segment";
        case T::CubReductionPath::OffsetSegmented: return "offset_segmented";
        case T::CubReductionPath::ComposedDense: return "composed_dense";
    }
    return "unknown";
}

struct TimingEvent {
    cudaEvent_t handle = nullptr;
    TimingEvent() { check(cudaEventCreate(&handle), "create timing event"); }
    ~TimingEvent() { if (handle) (void)cudaEventDestroy(handle); }
    TimingEvent(const TimingEvent&) = delete;
    TimingEvent& operator=(const TimingEvent&) = delete;
};

struct Geometry {
    std::string family;
    uint64_t k = 1;  // rows (outputs)
    uint64_t r = 1;  // row length
};

// R: whole-tensor reductions from 4 KiB to max, powers of two plus odd sizes near them.
// KR: [K, R] at several total sizes, row lengths from 2 to the whole size, including odd rows.
std::vector<Geometry> geometries(const Settings& s) {
    std::vector<Geometry> out;
    const bool want_r = std::find(s.families.begin(), s.families.end(), "R") != s.families.end();
    const bool want_kr = std::find(s.families.begin(), s.families.end(), "KR") != s.families.end();
    if (want_r) {
        for (uint64_t n = 4 * KiB; n <= s.max_bytes; n *= 2) {
            out.push_back({"R", 1, n});
            if (n * 3 / 2 + 7 <= s.max_bytes) out.push_back({"R", 1, n * 3 / 2 + 7});
        }
    }
    if (want_kr) {
        const std::vector<uint64_t> rows{2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 31, 32, 48, 64, 96, 100, 128, 255, 256, 512,
                                         1000, 1024, 2048, 4096, 8192, 16384, 65536, 262144, 1ULL << 20, 1ULL << 22,
                                         1ULL << 24, 1ULL << 26};
        for (uint64_t target : {256 * KiB, 4 * MiB, 64 * MiB, 512 * MiB}) {
            if (target > s.max_bytes) continue;
            for (uint64_t r : rows) {
                if (r * 2 > target) continue;  // K >= 2; K = 1 is the R family
                out.push_back({"KR", target / r, r});
            }
        }
    }
    return out;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

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
        const uint64_t flush_bytes = std::max<uint64_t>(64 * MiB, settings.flush_l2_multiple * uint64_t(l2_bytes));

        // One input allocation (every cell aliases its start, as a freshly allocated tensor would be aligned), one
        // copy destination for the bandwidth reference, and the flush buffer with its stamped read-only reduction.
        T::Tensor input_bytes(placement, T::TensorDescriptor(T::DataType::UINT8, {settings.max_bytes}));
        T::Tensor copy_destination(placement, T::TensorDescriptor(T::DataType::UINT8, {settings.max_bytes}));
        T::Tensor flush(placement, T::TensorDescriptor(T::DataType::FP8_E4M3, {flush_bytes}));
        flush.fillRandom(-1.0, 1.0, stream);
        auto flusher = T::CubReduction(T::CubReductionOp::Sum, uint32_t{0}, T::DataType::FP32).stamp(flush, stream);
        stream.synchronize();

        TimingEvent start, stop;
        // launch(slot) runs one reduction on input slot `slot`; a sample times slots 0..batch-1 after one flush.
        const auto time_us = [&](const std::function<void(size_t)>& launch, size_t batch) {
            std::vector<double> samples;
            launch(0);  // warm the instruction cache and any lazy state
            for (int i = 0; i < settings.samples; ++i) {
                flusher->runOn(stream);
                check(cudaEventRecord(start.handle, stream.getStream()), "start event");
                for (size_t slot = 0; slot < batch; ++slot) launch(slot);
                check(cudaEventRecord(stop.handle, stream.getStream()), "stop event");
                check(cudaEventSynchronize(stop.handle), "sync stop event");
                float ms = 0;
                check(cudaEventElapsedTime(&ms, start.handle, stop.handle), "event elapsed");
                samples.push_back(1000.0 * ms / static_cast<double>(batch));
            }
            return samples;
        };

        // Copy bandwidth (bytes read + written per us) from a cold copy of half the input allocation.
        const uint64_t copy_bytes = settings.max_bytes / 2;
        const double copy_us = median(time_us([&](size_t) {
            check(cudaMemcpyAsync(copy_destination.getMemPtr<void>(), input_bytes.getMemPtr<void>(), copy_bytes,
                                  cudaMemcpyDeviceToDevice, stream.getStream()),
                  "reference copy");
        }, 1));
        const double bandwidth = 2.0 * double(copy_bytes) / copy_us;  // bytes per us

        std::cout << "# fp8_dense_site_audit=1150 sms=" << sm << " l2_bytes=" << l2_bytes
                  << " flush_bytes=" << flush_bytes << " samples=" << settings.samples
                  << " copy_bandwidth_tbps=" << std::fixed << std::setprecision(3) << bandwidth / 1e6
                  << " floor_us=" << settings.floor_us << " batched_cold_launches=1 value_output=fp32 arg_outputs=index_uint32\n";
        std::cout << "family,kind,operation,dtype,K,R,input_bytes,path,passes,median_us,min_us,reference_us,ratio,"
                     "gbps,stamp_ms,rank\n";

        const std::vector<Geometry> grid = geometries(settings);
        uint64_t done = 0;
        for (T::DataType dtype : settings.dtypes) {
            input_bytes.aliasByteStorageAs(dtype, {settings.max_bytes}, 0).fillRandom(-100.0, 100.0, stream);
            stream.synchronize();
            for (const Geometry& g : grid) {
                const uint64_t bytes = g.k * g.r;
                const bool is_r = g.family == "R";
                const std::vector<uint64_t> dims = is_r ? std::vector<uint64_t>{g.r} : std::vector<uint64_t>{g.k, g.r};
                const uint32_t axis = is_r ? 0 : 1;
                // Input slots: 256-byte aligned, disjoint views of the arena; the batch uses the first `batch` of them.
                const uint64_t slot_stride = (bytes + 511) & ~uint64_t{255};
                const size_t batch = std::min<size_t>(
                    std::max<uint64_t>(1, settings.max_bytes / slot_stride),
                    static_cast<size_t>(std::clamp(std::ceil(200.0 / (double(bytes) / 1.5e6 + 3.0)), 1.0, 32.0)));
                std::vector<T::Tensor> inputs;
                for (size_t slot = 0; slot < batch; ++slot)
                    inputs.push_back(input_bytes.aliasByteStorageAs(dtype, dims, slot * slot_stride));
                for (const std::string& op : settings.ops) {
                    try {
                        std::string path, passes;
                        const auto stamp_start = std::chrono::steady_clock::now();
                        auto first_stamp_end = stamp_start;  // stamp_ms: the first slot's stamp (later ones hit the cache)
                        std::vector<std::shared_ptr<T::StampedCubReduction>> values;
                        std::vector<std::shared_ptr<T::StampedCubArgReduction>> args;
                        std::shared_ptr<T::StampedCubReduction> value;
                        std::shared_ptr<T::StampedCubArgReduction> arg;
                        std::function<void(size_t)> launch_slot;
                        if (op == "sum" || op == "max") {
                            const T::CubReduction reduction(op == "sum" ? T::CubReductionOp::Sum : T::CubReductionOp::Max,
                                                            axis, T::DataType::FP32);
                            for (const T::Tensor& slot_input : inputs) {
                                values.push_back(reduction.stamp(slot_input, stream));
                                if (values.size() == 1) first_stamp_end = std::chrono::steady_clock::now();
                            }
                            value = values.front();
                            path = pathName(value->getPath());
                            if (value->usesDenseExecutablePlan()) {
                                for (const auto& pass : value->getDensePhysicalPassDiagnostics())
                                    passes += (passes.empty() ? "" : "+") + pass.family;
                            }
                            launch_slot = [&](size_t slot) { values[slot]->runOn(stream); };
                        } else {
                            T::CubArgReductionOutputOptions outputs;
                            outputs.produce_value = false;
                            outputs.produce_index = true;
                            outputs.index_output_dtype = T::DataType::UINT32;
                            const T::CubArgReduction reduction(op == "argmax" ? T::CubArgReductionOp::ArgMax
                                                                              : T::CubArgReductionOp::ArgMin,
                                                               axis, outputs);
                            for (const T::Tensor& slot_input : inputs) {
                                args.push_back(reduction.stamp(slot_input, stream));
                                if (args.size() == 1) first_stamp_end = std::chrono::steady_clock::now();
                            }
                            arg = args.front();
                            path = pathName(arg->getPath());
                            for (const std::string& pass : arg->getPlannedPassNames())
                                passes += (passes.empty() ? "" : "+") + pass;
                            launch_slot = [&](size_t slot) { args[slot]->runOn(stream); };
                        }
                        if (passes.empty()) passes = path;
                        stream.synchronize();
                        const double stamp_ms =
                            std::chrono::duration<double, std::milli>(first_stamp_end - stamp_start).count();
                        const std::vector<double> samples = time_us(launch_slot, batch);
                        const double med = median(samples);
                        const uint64_t output_bytes = 4 * g.k;  // FP32 values or UINT32 indices
                        const double reference =
                            std::max(settings.floor_us, double(bytes + output_bytes) / bandwidth);
                        const char* kind = op == "sum" || op == "max" ? "VALUE" : "ARG";
                        std::cout << g.family << ',' << kind << ',' << op << ',' << dtypeName(dtype) << ',' << g.k
                                  << ',' << g.r << ',' << bytes << ',' << path << ',' << passes << ',' << std::setprecision(2) << med << ','
                                  << *std::min_element(samples.begin(), samples.end()) << ',' << reference << ','
                                  << std::setprecision(3) << med / reference << ',' << std::setprecision(1)
                                  << double(bytes) / med / 1e3 << ',' << std::setprecision(1) << stamp_ms << ",0\n";
                        if (settings.candidates) {
                            // Every retained shape in the selector's order. rank is the position in the order the
                            // production autotuner times (VALUE: the direct CUB pass is second, so shapes after the
                            // first move down one); a depth of N candidates times ranks 1..N.
                            namespace C = T::CubReductionInternal;
                            const auto shapes = C::Fp8KR::fp8KRCandidateShapes(g.k, g.r, static_cast<uint32_t>(sm));
                            const bool is_value = op == "sum" || op == "max";
                            const auto semantics = C::makeValueReductionStageSemantics(
                                op == "sum" ? T::CubReductionOp::Sum : T::CubReductionOp::Max, C::DensePassRole::Complete,
                                g.r);
                            T::Tensor out(placement, T::TensorDescriptor(is_value ? T::DataType::FP32
                                                                                   : T::DataType::UINT32, {g.k}));
                            for (size_t index = 0; index < shapes.size(); ++index) {
                                const auto& shape = shapes[index];
                                const uint64_t scratch_bytes = C::Fp8KR::fp8KRScratchBytes(shape, !is_value);
                                T::Tensor scratch;
                                if (scratch_bytes != 0) {
                                    scratch = T::Tensor(placement, T::TensorDescriptor(T::DataType::UINT8, {scratch_bytes}));
                                }
                                void* scratch_ptr = scratch_bytes != 0 ? scratch.getMemPtr<void>() : nullptr;
                                const auto candidate_samples = time_us([&](size_t slot) {
                                    if (is_value) {
                                        C::launchFp8KRValueSegments(semantics, inputs[slot], out, shape, scratch_ptr,
                                                                    scratch_bytes, 1.0f, stream);
                                    } else {
                                        C::launchFp8KRArgSegments(inputs[slot], nullptr, &out, shape, scratch_ptr,
                                                                  scratch_bytes, op == "argmax", 1, stream);
                                    }
                                }, batch);
                                const double candidate_median = median(candidate_samples);
                                const size_t rank = index + 1 + (is_value && index >= 1 ? 1 : 0);
                                std::cout << g.family << ',' << kind << ',' << op << ',' << dtypeName(dtype) << ','
                                          << g.k << ',' << g.r << ',' << bytes << ",candidate,"
                                          << C::Fp8KR::fp8KRShapeName(shape) << ',' << std::setprecision(2)
                                          << candidate_median << ','
                                          << *std::min_element(candidate_samples.begin(), candidate_samples.end())
                                          << ',' << reference << ',' << std::setprecision(3)
                                          << candidate_median / reference << ',' << std::setprecision(1)
                                          << double(bytes) / candidate_median / 1e3 << ",0," << rank << '\n';
                            }
                        }
                    } catch (const std::exception& e) {
                        // Report and continue: an unsupported geometry is itself an audit finding.
                        std::cout << "# unsupported," << g.family << ',' << op << ',' << dtypeName(dtype)
                                  << ",K=" << g.k << ",R=" << g.r << ',' << e.what() << '\n';
                    }
                }
                if (++done % 32 == 0) std::cerr << "[fp8-dense-site-audit] " << done << " geometries\n";
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "thor_fp8_dense_site_audit: " << e.what() << '\n';
        return 1;
    }
}
