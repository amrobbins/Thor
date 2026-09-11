#include "Utilities/TensorOperations/Misc/Concatenate.h"
#include "Utilities/TensorOperations/Misc/ConcatenateSpanGrouping.h"
#include "Utilities/TensorOperations/Misc/Split.h"
#include "Utilities/TensorOperations/Ragged/RaggedConcatenate.h"
#include "Utilities/TensorOperations/Ragged/RaggedSequenceConcatenate.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Common/Event.h"
#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/Expression/CudaHelpers.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

using namespace ThorImplementation;

namespace {

constexpr uint64_t MIB = 1024ULL * 1024ULL;
constexpr double BYTES_PER_GB = 1.0e9;
constexpr uint32_t THREADS_PER_CTA = 256;

struct Options {
    int gpu = 0;
    int warmup = 5;
    int iterations = 20;
    double l2_evict_multiple = 4.0;
    uint64_t min_evict_bytes = 64ULL * MIB;
    uint64_t max_copy_bytes = 128ULL * MIB;

    std::vector<std::string> kernels{
        "concatenate", "split", "ragged-concatenate", "ragged-split",
        "sequence-forward", "sequence-backward"};
    std::vector<uint64_t> outer_slices{1, 8, 64, 512, 4096};
    std::vector<uint64_t> active_rows{32, 128, 512, 2048, 8192};
    std::vector<uint32_t> num_arrays{2, 4, 8, 16};
    std::vector<uint64_t> span_bytes{4, 16, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1024, 2048, 4096};
    std::vector<std::string> span_layouts{"uniform", "skewed"};
    std::vector<uint64_t> ragged_outer_slices{1, 4};

    std::vector<uint64_t> sequence_batch_rows{32, 128, 512, 2048, 8192};
    std::vector<uint64_t> sequence_value_bytes{4, 16, 32, 64, 128};
    std::vector<uint64_t> sequence_values_per_span{1, 4, 16};
    std::vector<double> sequence_nonempty_fractions{1.0, 0.5};
    std::vector<uint32_t> sequence_offset_bytes{4};

    // 0 means production auto grouping; explicit values force the existing
    // specialization so grouping can be tuned without changing the kernel.
    std::vector<uint32_t> spans_per_cta{0};
};

uint64_t parseUnsigned(std::string_view text, std::string_view flag) {
    size_t consumed = 0;
    const unsigned long long value = std::stoull(std::string(text), &consumed);
    if (consumed != text.size()) throw std::invalid_argument("Invalid value for " + std::string(flag));
    return static_cast<uint64_t>(value);
}

double parseDouble(std::string_view text, std::string_view flag) {
    size_t consumed = 0;
    const double value = std::stod(std::string(text), &consumed);
    if (consumed != text.size() || !std::isfinite(value)) {
        throw std::invalid_argument("Invalid value for " + std::string(flag));
    }
    return value;
}

template <typename ParseFn>
auto parseCsv(std::string_view text, std::string_view flag, ParseFn &&parseOne) {
    using Value = decltype(parseOne(std::string_view{}));
    std::vector<Value> result;
    std::stringstream ss{std::string(text)};
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) continue;
        result.push_back(parseOne(token));
    }
    if (result.empty()) throw std::invalid_argument(std::string(flag) + " must contain at least one value");
    return result;
}

std::vector<uint64_t> parseUnsignedCsv(std::string_view text, std::string_view flag) {
    return parseCsv(text, flag, [&](std::string_view token) {
        const uint64_t value = parseUnsigned(token, flag);
        if (value == 0) throw std::invalid_argument(std::string(flag) + " values must be non-zero");
        return value;
    });
}

std::vector<uint32_t> parseUint32Csv(std::string_view text, std::string_view flag) {
    return parseCsv(text, flag, [&](std::string_view token) {
        const uint64_t value = parseUnsigned(token, flag);
        if (value == 0 || value > std::numeric_limits<uint32_t>::max()) {
            throw std::invalid_argument(std::string(flag) + " values must fit uint32 and be non-zero");
        }
        return static_cast<uint32_t>(value);
    });
}

std::vector<double> parseDoubleCsv(std::string_view text, std::string_view flag) {
    return parseCsv(text, flag, [&](std::string_view token) { return parseDouble(token, flag); });
}

std::vector<std::string> parseStringCsv(std::string_view text, std::string_view flag) {
    return parseCsv(text, flag, [](std::string_view token) { return std::string(token); });
}

std::vector<uint32_t> parseSpansPerCtaCsv(std::string_view text, std::string_view flag) {
    return parseCsv(text, flag, [&](std::string_view token) -> uint32_t {
        if (token == "auto") return 0;
        const uint64_t value = parseUnsigned(token, flag);
        if (value == 0 || value > 256 || (value & (value - 1)) != 0) {
            throw std::invalid_argument(std::string(flag) + " values must be auto or a power of two in [1,256]");
        }
        return static_cast<uint32_t>(value);
    });
}

void requireAllowed(const std::vector<std::string> &values,
                    std::initializer_list<std::string_view> allowed,
                    std::string_view flag) {
    for (const std::string &value : values) {
        bool ok = false;
        for (std::string_view candidate : allowed) ok = ok || value == candidate;
        if (!ok) throw std::invalid_argument("Unsupported " + std::string(flag) + " value: " + value);
    }
}

bool contains(const std::vector<std::string> &values, std::string_view value) {
    return std::find(values.begin(), values.end(), value) != values.end();
}

Options parseOptions(int argc, char **argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        auto requireValue = [&](std::string_view flag) -> std::string_view {
            if (i + 1 >= argc) throw std::invalid_argument("Missing value for " + std::string(flag));
            return argv[++i];
        };
        if (arg == "--gpu") options.gpu = static_cast<int>(parseUnsigned(requireValue(arg), arg));
        else if (arg == "--warmup") options.warmup = static_cast<int>(parseUnsigned(requireValue(arg), arg));
        else if (arg == "--iterations") options.iterations = static_cast<int>(parseUnsigned(requireValue(arg), arg));
        else if (arg == "--l2-evict-multiple") options.l2_evict_multiple = parseDouble(requireValue(arg), arg);
        else if (arg == "--min-evict-mib") options.min_evict_bytes = parseUnsigned(requireValue(arg), arg) * MIB;
        else if (arg == "--max-copy-mib") options.max_copy_bytes = parseUnsigned(requireValue(arg), arg) * MIB;
        else if (arg == "--kernels") options.kernels = parseStringCsv(requireValue(arg), arg);
        else if (arg == "--outer-slices") options.outer_slices = parseUnsignedCsv(requireValue(arg), arg);
        else if (arg == "--active-rows") options.active_rows = parseUnsignedCsv(requireValue(arg), arg);
        else if (arg == "--num-arrays") options.num_arrays = parseUint32Csv(requireValue(arg), arg);
        else if (arg == "--span-bytes") options.span_bytes = parseUnsignedCsv(requireValue(arg), arg);
        else if (arg == "--span-layouts") options.span_layouts = parseStringCsv(requireValue(arg), arg);
        else if (arg == "--ragged-outer-slices") options.ragged_outer_slices = parseUnsignedCsv(requireValue(arg), arg);
        else if (arg == "--sequence-batch-rows") options.sequence_batch_rows = parseUnsignedCsv(requireValue(arg), arg);
        else if (arg == "--sequence-value-bytes") options.sequence_value_bytes = parseUnsignedCsv(requireValue(arg), arg);
        else if (arg == "--sequence-values-per-span") options.sequence_values_per_span = parseUnsignedCsv(requireValue(arg), arg);
        else if (arg == "--sequence-nonempty-fractions") options.sequence_nonempty_fractions = parseDoubleCsv(requireValue(arg), arg);
        else if (arg == "--sequence-offset-bytes") options.sequence_offset_bytes = parseUint32Csv(requireValue(arg), arg);
        else if (arg == "--spans-per-cta") options.spans_per_cta = parseSpansPerCtaCsv(requireValue(arg), arg);
        else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: thor_concatenate_benchmark [options]\n"
                << "  --gpu N\n"
                << "  --warmup N\n"
                << "  --iterations N\n"
                << "  --l2-evict-multiple X              untimed DRAM-read eviction / L2 (default 4.0)\n"
                << "  --min-evict-mib N                  minimum eviction buffer (default 64)\n"
                << "  --max-copy-mib N                   skip logical copies larger than N MiB (default 128)\n"
                << "  --kernels concatenate,split,ragged-concatenate,ragged-split,sequence-forward,sequence-backward\n"
                << "  --outer-slices a,b,c                dense slices before concat axis\n"
                << "  --active-rows a,b,c                 packed ragged active rows\n"
                << "  --num-arrays 2,4,8,16\n"
                << "  --span-bytes a,b,c                  average bytes per dense/ragged span\n"
                << "  --span-layouts uniform,skewed       skewed preserves total bytes but gives one large span\n"
                << "  --ragged-outer-slices 1,4\n"
                << "  --sequence-batch-rows a,b,c\n"
                << "  --sequence-value-bytes a,b,c        bytes per packed sequence value\n"
                << "  --sequence-values-per-span 1,4,16   contiguous values copied by each non-empty span\n"
                << "  --sequence-nonempty-fractions 1,0.5\n"
                << "  --sequence-offset-bytes 4,8\n"
                << "  --spans-per-cta auto,1,2,...,256    force existing CTA specialization (default auto)\n\n"
                << "Each case is warmed first. Before every timed sample an untimed device-read pass\n"
                << "over >L2 bytes runs on the same stream, so source reads begin DRAM-backed while\n"
                << "allocation/JIT/warmup and eviction itself stay outside target timing.\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("Unknown argument: " + std::string(arg));
        }
    }

    if (options.gpu < 0) throw std::invalid_argument("--gpu must be non-negative");
    if (options.warmup < 0) throw std::invalid_argument("--warmup must be non-negative");
    if (options.iterations <= 0) throw std::invalid_argument("--iterations must be positive");
    if (options.l2_evict_multiple < 2.0) throw std::invalid_argument("--l2-evict-multiple must be >= 2.0");
    requireAllowed(options.kernels,
                   {"concatenate", "split", "ragged-concatenate", "ragged-split", "sequence-forward", "sequence-backward"},
                   "--kernels");
    requireAllowed(options.span_layouts, {"uniform", "skewed"}, "--span-layouts");
    for (uint32_t n : options.num_arrays) {
        if (n < 2 || n > 256) throw std::invalid_argument("--num-arrays values must be in [2,256]");
    }
    for (double fraction : options.sequence_nonempty_fractions) {
        if (!(fraction > 0.0 && fraction <= 1.0)) {
            throw std::invalid_argument("--sequence-nonempty-fractions values must be in (0,1]");
        }
    }
    for (uint32_t bytes : options.sequence_offset_bytes) {
        if (bytes != 4 && bytes != 8) throw std::invalid_argument("--sequence-offset-bytes values must be 4 or 8");
    }
    return options;
}

uint64_t checkedMultiply(uint64_t a, uint64_t b, const char *what) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) throw std::overflow_error(what);
    return a * b;
}

uint64_t checkedAdd(uint64_t a, uint64_t b, const char *what) {
    if (b > std::numeric_limits<uint64_t>::max() - a) throw std::overflow_error(what);
    return a + b;
}

uint64_t roundUp(uint64_t value, uint64_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

uint64_t splitMix64(uint64_t value) {
    uint64_t z = value + 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27U)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31U);
}

uint32_t concatenateAutoSpansPerCta(uint64_t expectedSpanBytes, uint64_t spans) {
    return ThorConcatenateSpanGrouping::selectSpansPerCta(spans, expectedSpanBytes);
}

// RaggedSequenceConcatenate was measured separately and its existing selector
// was already near-optimal.  Keep that production policy distinct from the
// dense/ragged concatenate/split selector tuned by this benchmark.
uint32_t sequenceAutoSpansPerCta(uint64_t expectedSpanBytes, uint64_t spans) {
    uint32_t spansByPayload = 1;
    if (expectedSpanBytes <= 32) spansByPayload = 256;
    else if (expectedSpanBytes <= 64) spansByPayload = 128;
    else if (expectedSpanBytes <= 128) spansByPayload = 64;
    else if (expectedSpanBytes <= 256) spansByPayload = 32;
    else if (expectedSpanBytes <= 512) spansByPayload = 16;
    else if (expectedSpanBytes <= 1024) spansByPayload = 8;
    else if (expectedSpanBytes <= 2048) spansByPayload = 4;
    else if (expectedSpanBytes <= 4096) spansByPayload = 2;

    uint32_t spansByParallelism = 1;
    if (spans >= 16384) spansByParallelism = 256;
    else if (spans >= 8192) spansByParallelism = 128;
    else if (spans >= 4096) spansByParallelism = 64;
    else if (spans >= 2048) spansByParallelism = 32;
    else if (spans >= 1024) spansByParallelism = 16;
    else if (spans >= 512) spansByParallelism = 8;
    else if (spans >= 256) spansByParallelism = 4;
    else if (spans >= 128) spansByParallelism = 2;
    return std::min(spansByPayload, spansByParallelism);
}

class DeviceAllocation {
   public:
    DeviceAllocation() = default;
    explicit DeviceAllocation(uint64_t bytes) : bytes_(bytes) {
        // Thor tensors guarantee 128 bytes of physical padding. The copy kernels
        // intentionally permit a final vector load to extend beyond the logical
        // span, so raw benchmark allocations must preserve the same contract.
        constexpr uint64_t kThorPaddingBytes = 128;
        const uint64_t logicalAllocationBytes = std::max<uint64_t>(bytes, 1);
        if (logicalAllocationBytes > std::numeric_limits<uint64_t>::max() - kThorPaddingBytes) {
            throw std::overflow_error("Benchmark allocation plus Thor padding overflows uint64");
        }
        const uint64_t allocationBytes = logicalAllocationBytes + kThorPaddingBytes;
        if (allocationBytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            throw std::overflow_error("Benchmark allocation exceeds size_t");
        }
        CUDA_CHECK(cudaMalloc(&pointer_, static_cast<size_t>(allocationBytes)));
    }
    DeviceAllocation(const DeviceAllocation &) = delete;
    DeviceAllocation &operator=(const DeviceAllocation &) = delete;
    DeviceAllocation(DeviceAllocation &&other) noexcept : pointer_(other.pointer_), bytes_(other.bytes_) {
        other.pointer_ = nullptr;
        other.bytes_ = 0;
    }
    DeviceAllocation &operator=(DeviceAllocation &&other) noexcept {
        if (this == &other) return *this;
        if (pointer_ != nullptr) (void)cudaFree(pointer_);
        pointer_ = other.pointer_;
        bytes_ = other.bytes_;
        other.pointer_ = nullptr;
        other.bytes_ = 0;
        return *this;
    }
    ~DeviceAllocation() {
        if (pointer_ != nullptr) (void)cudaFree(pointer_);
    }
    void *get() const { return pointer_; }
    uint64_t bytes() const { return bytes_; }

   private:
    void *pointer_ = nullptr;
    uint64_t bytes_ = 0;
};

template <typename T>
DeviceAllocation uploadVector(const std::vector<T> &values, Stream &stream) {
    DeviceAllocation result(checkedMultiply(values.size(), sizeof(T), "Benchmark upload byte count overflow"));
    if (!values.empty()) {
        CUDA_CHECK(cudaMemcpyAsync(result.get(), values.data(), values.size() * sizeof(T),
                                   cudaMemcpyHostToDevice, stream.getStream()));
    }
    return result;
}

DeviceAllocation makePointerTable(const std::vector<void *> &pointers, Stream &stream) {
    return uploadVector(pointers, stream);
}

void initializeAllocation(DeviceAllocation &allocation, Stream &stream, uint8_t pattern) {
    if (allocation.bytes() != 0) {
        CUDA_CHECK(cudaMemsetAsync(allocation.get(), pattern, static_cast<size_t>(allocation.bytes()), stream.getStream()));
    }
}

__global__ void readForL2EvictionKernel(const uint4 *__restrict__ bytes,
                                        uint64_t uint4Count,
                                        unsigned long long *__restrict__ sink) {
    uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    unsigned long long accumulator = 0;
    for (; index < uint4Count; index += stride) {
        const uint4 value = bytes[index];
        accumulator += static_cast<unsigned long long>(value.x) + value.y + value.z + value.w;
    }
    if ((threadIdx.x & 31U) == 0U) atomicAdd(sink, accumulator);
}

class DramReadEvictor {
   public:
    DramReadEvictor(const Options &options, const cudaDeviceProp &properties, Stream &stream)
        : bytes_(std::max<uint64_t>(options.min_evict_bytes,
                                    static_cast<uint64_t>(std::ceil(
                                        options.l2_evict_multiple * static_cast<double>(properties.l2CacheSize))))),
          storage_(roundUp(bytes_, sizeof(uint4))),
          sink_(sizeof(uint64_t)),
          blocks_(std::max(1, properties.multiProcessorCount * 8)) {
        initializeAllocation(storage_, stream, 0x5a);
        initializeAllocation(sink_, stream, 0);
        stream.synchronize();
        run(stream);
        stream.synchronize();
    }

    void run(Stream &stream) {
        constexpr uint32_t threads = 256;
        const uint64_t count = storage_.bytes() / sizeof(uint4);
        readForL2EvictionKernel<<<blocks_, threads, 0, stream.getStream()>>>(
            reinterpret_cast<const uint4 *>(storage_.get()), count,
            reinterpret_cast<unsigned long long *>(sink_.get()));
        CUDA_CHECK(cudaGetLastError());
    }

    uint64_t bytes() const { return storage_.bytes(); }

   private:
    uint64_t bytes_ = 0;
    DeviceAllocation storage_;
    DeviceAllocation sink_;
    int blocks_ = 0;
};

struct TimingStats {
    double p10_us = 0.0;
    double median_us = 0.0;
    double p90_us = 0.0;
};

double percentileSorted(const std::vector<float> &values, double percentile) {
    if (values.empty()) return 0.0;
    const double position = percentile * static_cast<double>(values.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(position));
    const size_t hi = static_cast<size_t>(std::ceil(position));
    const double alpha = position - static_cast<double>(lo);
    return (1.0 - alpha) * static_cast<double>(values[lo]) + alpha * static_cast<double>(values[hi]);
}

template <typename Fn>
TimingStats benchmarkDramReads(Stream &stream,
                               DramReadEvictor &evictor,
                               int warmup,
                               int iterations,
                               Fn &&fn) {
    for (int i = 0; i < warmup; ++i) fn();
    stream.synchronize();

    Event start(stream.getGpuNum(), /*enableTiming=*/true);
    Event stop(stream.getGpuNum(), /*enableTiming=*/true);
    std::vector<float> milliseconds;
    milliseconds.reserve(iterations);
    for (int i = 0; i < iterations; ++i) {
        evictor.run(stream);
        start.record(stream);
        fn();
        stop.record(stream);
        milliseconds.push_back(stop.synchronizeAndReportElapsedTimeInMilliseconds(start));
    }
    std::sort(milliseconds.begin(), milliseconds.end());
    return TimingStats{
        percentileSorted(milliseconds, 0.10) * 1000.0,
        percentileSorted(milliseconds, 0.50) * 1000.0,
        percentileSorted(milliseconds, 0.90) * 1000.0};
}

struct Result {
    std::string kernel;
    std::string variant;
    uint64_t workRows = 0;
    uint64_t outerSlicesPerValue = 1;
    uint32_t numArrays = 0;
    uint64_t averageSpanBytes = 0;
    uint64_t valueBytes = 0;
    uint64_t valuesPerSpan = 0;
    double nonemptyFraction = 1.0;
    uint32_t offsetBytes = 0;
    uint64_t spanCount = 0;
    uint64_t expectedSpanBytes = 0;
    std::string groupingMode;
    uint32_t autoSpansPerCta = 0;
    uint32_t spansPerCta = 0;
    uint32_t lanesPerSpan = 0;
    uint64_t copyBytes = 0;
    TimingStats timing;
};

double gbps(uint64_t bytes, double microseconds) {
    if (microseconds <= 0.0) return 0.0;
    return (static_cast<double>(bytes) / BYTES_PER_GB) / (microseconds * 1.0e-6);
}

void printHeader() {
    std::cout
        << "kernel,variant,work_rows,outer_slices_per_value,num_arrays,average_span_bytes,value_bytes,values_per_span,"
           "nonempty_fraction,offset_bytes,span_count,expected_span_bytes,grouping_mode,auto_spans_per_cta,"
           "spans_per_cta,lanes_per_span,copy_bytes,p10_us,median_us,p90_us,copy_GBps,read_write_GBps\n";
}

void printResult(const Result &r) {
    std::cout << r.kernel << ',' << r.variant << ',' << r.workRows << ',' << r.outerSlicesPerValue << ','
              << r.numArrays << ',' << r.averageSpanBytes << ',' << r.valueBytes << ',' << r.valuesPerSpan << ','
              << std::fixed << std::setprecision(3) << r.nonemptyFraction << ',' << r.offsetBytes << ','
              << r.spanCount << ',' << r.expectedSpanBytes << ',' << r.groupingMode << ',' << r.autoSpansPerCta << ','
              << r.spansPerCta << ',' << r.lanesPerSpan << ',' << r.copyBytes << ','
              << std::setprecision(3) << r.timing.p10_us << ',' << r.timing.median_us << ',' << r.timing.p90_us << ','
              << std::setprecision(6) << gbps(r.copyBytes, r.timing.median_us) << ','
              << gbps(checkedMultiply(r.copyBytes, 2, "Traffic byte count overflow"), r.timing.median_us) << '\n';
}

std::vector<uint64_t> makeSpanWidths(uint32_t numArrays, uint64_t averageSpanBytes, std::string_view layout) {
    std::vector<uint64_t> widths(numArrays, averageSpanBytes);
    if (layout == "uniform") return widths;

    const uint64_t total = checkedMultiply(numArrays, averageSpanBytes, "Span layout byte count overflow");
    const uint64_t small = std::max<uint64_t>(1, averageSpanBytes / 8);
    const uint64_t smallTotal = checkedMultiply(numArrays - 1ULL, small, "Skewed span layout overflow");
    if (smallTotal >= total) return widths;
    widths.assign(numArrays, small);
    widths[0] = total - smallTotal;
    return widths;
}

uint64_t sumWidths(const std::vector<uint64_t> &widths) {
    uint64_t total = 0;
    for (uint64_t width : widths) total = checkedAdd(total, width, "Span width sum overflow");
    return total;
}

struct DenseCaseStorage {
    std::vector<DeviceAllocation> split;
    DeviceAllocation packed;
    DeviceAllocation pointerTable;
    DeviceAllocation geometry;
};

DenseCaseStorage makeDenseCase(int gpu,
                               uint64_t outerSlices,
                               const std::vector<uint64_t> &widths,
                               Stream &stream) {
    (void)gpu;
    DenseCaseStorage storage;
    std::vector<void *> pointers;
    std::vector<uint64_t> axisElements;
    axisElements.reserve(widths.size());
    for (uint64_t width : widths) {
        axisElements.push_back(width);
        storage.split.emplace_back(checkedMultiply(outerSlices, width, "Dense split allocation overflow"));
        initializeAllocation(storage.split.back(), stream, 0x31);
        pointers.push_back(storage.split.back().get());
    }
    const uint64_t packedSliceBytes = sumWidths(widths);
    storage.packed = DeviceAllocation(checkedMultiply(outerSlices, packedSliceBytes, "Dense packed allocation overflow"));
    initializeAllocation(storage.packed, stream, 0xa7);
    storage.pointerTable = makePointerTable(pointers, stream);
    const std::vector<ConcatenateSpanGeometry> geometryHost = buildConcatenateSpanGeometry(1, 1, axisElements);
    storage.geometry = uploadVector(geometryHost, stream);
    stream.synchronize();
    return storage;
}

void runDenseBenchmarks(const Options &options,
                        Stream &stream,
                        DramReadEvictor &evictor,
                        uint64_t &skipped) {
    const bool doConcat = contains(options.kernels, "concatenate");
    const bool doSplit = contains(options.kernels, "split");
    if (!doConcat && !doSplit) return;

    for (uint64_t outerSlices : options.outer_slices) {
        for (uint32_t numArrays : options.num_arrays) {
            for (uint64_t averageSpanBytes : options.span_bytes) {
                for (const std::string &layout : options.span_layouts) {
                    const std::vector<uint64_t> widths = makeSpanWidths(numArrays, averageSpanBytes, layout);
                    const uint64_t packedSliceBytes = sumWidths(widths);
                    const uint64_t copyBytes = checkedMultiply(outerSlices, packedSliceBytes, "Dense copy bytes overflow");
                    if (copyBytes > options.max_copy_bytes) {
                        ++skipped;
                        continue;
                    }
                    const uint64_t spans = checkedMultiply(outerSlices, numArrays, "Dense span count overflow");
                    const uint64_t expectedSpanBytes = packedSliceBytes / numArrays + (packedSliceBytes % numArrays != 0);
                    const uint32_t autoGrouping = concatenateAutoSpansPerCta(expectedSpanBytes, spans);
                    DenseCaseStorage storage = makeDenseCase(options.gpu, outerSlices, widths, stream);

                    for (uint32_t requested : options.spans_per_cta) {
                        const uint32_t actual = requested == 0 ? autoGrouping : requested;
                        if (doConcat) {
                            const TimingStats timing = benchmarkDramReads(
                                stream, evictor, options.warmup, options.iterations, [&] {
                                    if (requested == 0) {
                                        launchConcatenate(storage.packed.get(), reinterpret_cast<void **>(storage.pointerTable.get()),
                                                          outerSlices, numArrays, packedSliceBytes,
                                                          static_cast<const ConcatenateSpanGeometry *>(storage.geometry.get()), stream);
                                    } else {
                                        launchConcatenateWithSpansPerCtaForBenchmark(
                                            storage.packed.get(), reinterpret_cast<void **>(storage.pointerTable.get()),
                                            outerSlices, numArrays, packedSliceBytes,
                                            static_cast<const ConcatenateSpanGeometry *>(storage.geometry.get()), requested, stream);
                                    }
                                });
                            printResult(Result{"concatenate", layout, outerSlices, 1, numArrays, averageSpanBytes, 0, 0, 1.0, 0,
                                               spans, expectedSpanBytes, requested == 0 ? "auto" : "forced", autoGrouping,
                                               actual, THREADS_PER_CTA / actual, copyBytes, timing});
                        }
                        if (doSplit) {
                            const TimingStats timing = benchmarkDramReads(
                                stream, evictor, options.warmup, options.iterations, [&] {
                                    if (requested == 0) {
                                        launchSplit(reinterpret_cast<void **>(storage.pointerTable.get()), storage.packed.get(),
                                                    outerSlices, numArrays, packedSliceBytes,
                                                    static_cast<const ConcatenateSpanGeometry *>(storage.geometry.get()), stream);
                                    } else {
                                        launchSplitWithSpansPerCtaForBenchmark(
                                            reinterpret_cast<void **>(storage.pointerTable.get()), storage.packed.get(),
                                            outerSlices, numArrays, packedSliceBytes,
                                            static_cast<const ConcatenateSpanGeometry *>(storage.geometry.get()), requested, stream);
                                    }
                                });
                            printResult(Result{"split", layout, outerSlices, 1, numArrays, averageSpanBytes, 0, 0, 1.0, 0,
                                               spans, expectedSpanBytes, requested == 0 ? "auto" : "forced", autoGrouping,
                                               actual, THREADS_PER_CTA / actual, copyBytes, timing});
                        }
                    }
                }
            }
        }
    }
}

struct RaggedCaseStorage {
    std::vector<DeviceAllocation> split;
    DeviceAllocation packed;
    DeviceAllocation pointerTable;
    DeviceAllocation geometry;
};

RaggedCaseStorage makeRaggedCase(uint64_t activeRows,
                                 uint64_t outerSlicesPerValue,
                                 const std::vector<uint64_t> &widths,
                                 Stream &stream) {
    RaggedCaseStorage storage;
    std::vector<void *> pointers;
    std::vector<uint64_t> axisOffsets;
    axisOffsets.reserve(widths.size() + 1);
    axisOffsets.push_back(0);
    uint64_t cumulative = 0;
    for (uint64_t width : widths) {
        cumulative = checkedAdd(cumulative, width, "Ragged axis offset overflow");
        axisOffsets.push_back(cumulative);
        const uint64_t valueBytes = checkedMultiply(width, outerSlicesPerValue, "Ragged input value width overflow");
        storage.split.emplace_back(checkedMultiply(activeRows, valueBytes, "Ragged split allocation overflow"));
        initializeAllocation(storage.split.back(), stream, 0x43);
        pointers.push_back(storage.split.back().get());
    }
    const uint64_t outputValueBytes = checkedMultiply(cumulative, outerSlicesPerValue, "Ragged output value width overflow");
    storage.packed = DeviceAllocation(checkedMultiply(activeRows, outputValueBytes, "Ragged packed allocation overflow"));
    initializeAllocation(storage.packed, stream, 0xb2);
    storage.pointerTable = makePointerTable(pointers, stream);
    const std::vector<RaggedConcatenateSpanGeometry> geometryHost =
        buildRaggedConcatenateSpanGeometry(1, outerSlicesPerValue, 1, axisOffsets);
    storage.geometry = uploadVector(geometryHost, stream);
    stream.synchronize();
    return storage;
}

void runRaggedBenchmarks(const Options &options,
                         Stream &stream,
                         DramReadEvictor &evictor,
                         uint64_t &skipped) {
    const bool doConcat = contains(options.kernels, "ragged-concatenate");
    const bool doSplit = contains(options.kernels, "ragged-split");
    if (!doConcat && !doSplit) return;

    for (uint64_t activeRows : options.active_rows) {
        for (uint64_t outerSlicesPerValue : options.ragged_outer_slices) {
            for (uint32_t numArrays : options.num_arrays) {
                for (uint64_t averageSpanBytes : options.span_bytes) {
                    for (const std::string &layout : options.span_layouts) {
                        const std::vector<uint64_t> widths = makeSpanWidths(numArrays, averageSpanBytes, layout);
                        const uint64_t outputSliceBytes = sumWidths(widths);
                        const uint64_t outputValueBytes = checkedMultiply(outputSliceBytes, outerSlicesPerValue,
                                                                          "Ragged output value bytes overflow");
                        const uint64_t copyBytes = checkedMultiply(activeRows, outputValueBytes, "Ragged copy bytes overflow");
                        if (copyBytes > options.max_copy_bytes) {
                            ++skipped;
                            continue;
                        }
                        const uint64_t spans = checkedMultiply(
                            checkedMultiply(activeRows, outerSlicesPerValue, "Ragged span count overflow"),
                            numArrays, "Ragged span count overflow");
                        const uint64_t expectedSpanBytes = outputSliceBytes / numArrays + (outputSliceBytes % numArrays != 0);
                        const uint32_t autoGrouping = concatenateAutoSpansPerCta(expectedSpanBytes, spans);
                        RaggedCaseStorage storage = makeRaggedCase(activeRows, outerSlicesPerValue, widths, stream);

                        for (uint32_t requested : options.spans_per_cta) {
                            const uint32_t actual = requested == 0 ? autoGrouping : requested;
                            if (doConcat) {
                                const TimingStats timing = benchmarkDramReads(
                                    stream, evictor, options.warmup, options.iterations, [&] {
                                        if (requested == 0) {
                                            launchRaggedConcatenate(
                                                storage.packed.get(), reinterpret_cast<void **>(storage.pointerTable.get()),
                                                activeRows, outputValueBytes, outerSlicesPerValue, numArrays,
                                                static_cast<const RaggedConcatenateSpanGeometry *>(storage.geometry.get()),
                                                activeRows, stream);
                                        } else {
                                            launchRaggedConcatenateWithSpansPerCtaForBenchmark(
                                                storage.packed.get(), reinterpret_cast<void **>(storage.pointerTable.get()),
                                                activeRows, outputValueBytes, outerSlicesPerValue, numArrays,
                                                static_cast<const RaggedConcatenateSpanGeometry *>(storage.geometry.get()),
                                                activeRows, requested, stream);
                                        }
                                    });
                                printResult(Result{"ragged-concatenate", layout, activeRows, outerSlicesPerValue, numArrays,
                                                   averageSpanBytes, 0, 0, 1.0, 0, spans, expectedSpanBytes,
                                                   requested == 0 ? "auto" : "forced", autoGrouping, actual,
                                                   THREADS_PER_CTA / actual, copyBytes, timing});
                            }
                            if (doSplit) {
                                const TimingStats timing = benchmarkDramReads(
                                    stream, evictor, options.warmup, options.iterations, [&] {
                                        if (requested == 0) {
                                            launchRaggedSplit(
                                                reinterpret_cast<void **>(storage.pointerTable.get()), storage.packed.get(),
                                                activeRows, outputValueBytes, outerSlicesPerValue, numArrays,
                                                static_cast<const RaggedConcatenateSpanGeometry *>(storage.geometry.get()),
                                                activeRows, stream);
                                        } else {
                                            launchRaggedSplitWithSpansPerCtaForBenchmark(
                                                reinterpret_cast<void **>(storage.pointerTable.get()), storage.packed.get(),
                                                activeRows, outputValueBytes, outerSlicesPerValue, numArrays,
                                                static_cast<const RaggedConcatenateSpanGeometry *>(storage.geometry.get()),
                                                activeRows, requested, stream);
                                        }
                                    });
                                printResult(Result{"ragged-split", layout, activeRows, outerSlicesPerValue, numArrays,
                                                   averageSpanBytes, 0, 0, 1.0, 0, spans, expectedSpanBytes,
                                                   requested == 0 ? "auto" : "forced", autoGrouping, actual,
                                                   THREADS_PER_CTA / actual, copyBytes, timing});
                            }
                        }
                    }
                }
            }
        }
    }
}

template <typename SpanT>
struct SequencePlan {
    std::vector<SpanT> spans;
    std::vector<uint64_t> tokensPerInput;
    uint64_t activeOutputValues = 0;
};

template <typename SpanT>
SequencePlan<SpanT> makeSequencePlan(uint64_t batchRows,
                                     uint32_t numInputs,
                                     uint64_t valuesPerSpan,
                                     double nonemptyFraction) {
    SequencePlan<SpanT> result;
    result.tokensPerInput.assign(numInputs, 0);
    result.spans.reserve(static_cast<size_t>(std::min<uint64_t>(
        checkedMultiply(batchRows, numInputs, "Sequence span reserve overflow"),
        static_cast<uint64_t>(std::numeric_limits<size_t>::max()))));
    const uint64_t threshold = static_cast<uint64_t>(std::llround(nonemptyFraction * 1000000.0));
    uint64_t outputBegin = 0;
    for (uint64_t row = 0; row < batchRows; ++row) {
        for (uint32_t input = 0; input < numInputs; ++input) {
            const bool nonempty = nonemptyFraction >= 1.0 ||
                (splitMix64(row * 0x9e3779b97f4a7c15ULL + input * 0xbf58476d1ce4e5b9ULL) % 1000000ULL) < threshold;
            if (!nonempty) continue;
            const uint64_t sourceBegin = result.tokensPerInput[input];
            if constexpr (std::is_same_v<SpanT, RaggedSequenceCopySpan32>) {
                if (sourceBegin > std::numeric_limits<uint32_t>::max() ||
                    outputBegin > std::numeric_limits<uint32_t>::max() ||
                    valuesPerSpan > std::numeric_limits<uint32_t>::max()) {
                    throw std::overflow_error("Sequence UINT32 copy plan overflow");
                }
                result.spans.push_back(RaggedSequenceCopySpan32{
                    input, static_cast<uint32_t>(sourceBegin), static_cast<uint32_t>(outputBegin),
                    static_cast<uint32_t>(valuesPerSpan)});
            } else {
                result.spans.push_back(RaggedSequenceCopySpan64{input, 0, sourceBegin, outputBegin, valuesPerSpan});
            }
            result.tokensPerInput[input] = checkedAdd(sourceBegin, valuesPerSpan, "Sequence input token count overflow");
            outputBegin = checkedAdd(outputBegin, valuesPerSpan, "Sequence output token count overflow");
        }
    }
    result.activeOutputValues = outputBegin;
    return result;
}

struct SequenceCaseStorage {
    std::vector<DeviceAllocation> inputs;
    DeviceAllocation output;
    DeviceAllocation inputPointerTable;
    DeviceAllocation copySpans;
};

template <typename SpanT>
SequenceCaseStorage makeSequenceCase(const SequencePlan<SpanT> &plan,
                                     uint64_t valueBytes,
                                     Stream &stream) {
    SequenceCaseStorage storage;
    std::vector<void *> pointers;
    pointers.reserve(plan.tokensPerInput.size());
    for (uint64_t tokens : plan.tokensPerInput) {
        storage.inputs.emplace_back(checkedMultiply(tokens, valueBytes, "Sequence input allocation overflow"));
        initializeAllocation(storage.inputs.back(), stream, 0x53);
        pointers.push_back(storage.inputs.back().get());
    }
    storage.output = DeviceAllocation(checkedMultiply(plan.activeOutputValues, valueBytes,
                                                       "Sequence output allocation overflow"));
    initializeAllocation(storage.output, stream, 0xc4);
    storage.inputPointerTable = makePointerTable(pointers, stream);
    storage.copySpans = uploadVector(plan.spans, stream);
    stream.synchronize();
    return storage;
}

template <typename SpanT>
void runOneSequenceGeometry(const Options &options,
                            Stream &stream,
                            DramReadEvictor &evictor,
                            uint64_t batchRows,
                            uint32_t numInputs,
                            uint64_t valueBytes,
                            uint64_t valuesPerSpan,
                            double nonemptyFraction,
                            uint32_t offsetBytes,
                            uint64_t &skipped) {
    const bool doForward = contains(options.kernels, "sequence-forward");
    const bool doBackward = contains(options.kernels, "sequence-backward");
    if (!doForward && !doBackward) return;

    SequencePlan<SpanT> plan = makeSequencePlan<SpanT>(batchRows, numInputs, valuesPerSpan, nonemptyFraction);
    if (plan.spans.empty()) return;
    const uint64_t copyBytes = checkedMultiply(plan.activeOutputValues, valueBytes, "Sequence copy bytes overflow");
    if (copyBytes > options.max_copy_bytes) {
        ++skipped;
        return;
    }
    const uint64_t expectedSpanBytes = checkedMultiply(valuesPerSpan, valueBytes, "Sequence expected span bytes overflow");
    const uint32_t autoGrouping = sequenceAutoSpansPerCta(expectedSpanBytes, plan.spans.size());
    SequenceCaseStorage storage = makeSequenceCase(plan, valueBytes, stream);

    for (uint32_t requested : options.spans_per_cta) {
        const uint32_t actual = requested == 0 ? autoGrouping : requested;
        if (doForward) {
            const TimingStats timing = benchmarkDramReads(
                stream, evictor, options.warmup, options.iterations, [&] {
                    if (requested == 0) {
                        launchRaggedSequenceConcatenate(
                            storage.output.get(), reinterpret_cast<void **>(storage.inputPointerTable.get()),
                            storage.copySpans.get(), plan.spans.size(), 1, valueBytes, offsetBytes,
                            plan.activeOutputValues, stream);
                    } else {
                        launchRaggedSequenceConcatenateWithSpansPerCtaForBenchmark(
                            storage.output.get(), reinterpret_cast<void **>(storage.inputPointerTable.get()),
                            storage.copySpans.get(), plan.spans.size(), 1, valueBytes, offsetBytes,
                            plan.activeOutputValues, requested, stream);
                    }
                });
            printResult(Result{"sequence-forward", "uniform-span-values", batchRows, 1, numInputs,
                               expectedSpanBytes, valueBytes, valuesPerSpan, nonemptyFraction, offsetBytes,
                               plan.spans.size(), expectedSpanBytes, requested == 0 ? "auto" : "forced",
                               autoGrouping, actual, THREADS_PER_CTA / actual, copyBytes, timing});
        }
        if (doBackward) {
            const TimingStats timing = benchmarkDramReads(
                stream, evictor, options.warmup, options.iterations, [&] {
                    if (requested == 0) {
                        launchRaggedSequenceConcatenateBackward(
                            reinterpret_cast<void **>(storage.inputPointerTable.get()), storage.output.get(),
                            storage.copySpans.get(), plan.spans.size(), 1, valueBytes, offsetBytes,
                            plan.activeOutputValues, stream);
                    } else {
                        launchRaggedSequenceConcatenateBackwardWithSpansPerCtaForBenchmark(
                            reinterpret_cast<void **>(storage.inputPointerTable.get()), storage.output.get(),
                            storage.copySpans.get(), plan.spans.size(), 1, valueBytes, offsetBytes,
                            plan.activeOutputValues, requested, stream);
                    }
                });
            printResult(Result{"sequence-backward", "all-gradients", batchRows, 1, numInputs,
                               expectedSpanBytes, valueBytes, valuesPerSpan, nonemptyFraction, offsetBytes,
                               plan.spans.size(), expectedSpanBytes, requested == 0 ? "auto" : "forced",
                               autoGrouping, actual, THREADS_PER_CTA / actual, copyBytes, timing});
        }
    }
}

void runSequenceBenchmarks(const Options &options,
                           Stream &stream,
                           DramReadEvictor &evictor,
                           uint64_t &skipped) {
    if (!contains(options.kernels, "sequence-forward") && !contains(options.kernels, "sequence-backward")) return;
    for (uint64_t batchRows : options.sequence_batch_rows) {
        for (uint32_t numInputs : options.num_arrays) {
            for (uint64_t valueBytes : options.sequence_value_bytes) {
                for (uint64_t valuesPerSpan : options.sequence_values_per_span) {
                    for (double fraction : options.sequence_nonempty_fractions) {
                        for (uint32_t offsetBytes : options.sequence_offset_bytes) {
                            if (offsetBytes == 4) {
                                runOneSequenceGeometry<RaggedSequenceCopySpan32>(
                                    options, stream, evictor, batchRows, numInputs, valueBytes,
                                    valuesPerSpan, fraction, offsetBytes, skipped);
                            } else {
                                runOneSequenceGeometry<RaggedSequenceCopySpan64>(
                                    options, stream, evictor, batchRows, numInputs, valueBytes,
                                    valuesPerSpan, fraction, offsetBytes, skipped);
                            }
                        }
                    }
                }
            }
        }
    }
}

}  // namespace

int main(int argc, char **argv) {
    try {
        const Options options = parseOptions(argc, argv);
        ScopedGpu scopedGpu(options.gpu);
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, options.gpu));
        if (properties.l2CacheSize <= 0) {
            throw std::runtime_error("CUDA reported zero L2 bytes; cannot construct DRAM-backed benchmark");
        }

        Stream stream(options.gpu);
        DramReadEvictor evictor(options, properties, stream);
        size_t freeBytes = 0;
        size_t totalBytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
        if (evictor.bytes() > static_cast<uint64_t>(freeBytes) / 2) {
            throw std::runtime_error("L2 eviction buffer would consume more than half of free GPU memory");
        }

        std::cerr << "Thor concatenate/split benchmark\n"
                  << "GPU: " << options.gpu << " (" << properties.name << ")\n"
                  << "L2: " << static_cast<double>(properties.l2CacheSize) / MIB << " MiB\n"
                  << "Untimed device-read eviction: " << static_cast<double>(evictor.bytes()) / MIB << " MiB ("
                  << static_cast<double>(evictor.bytes()) / properties.l2CacheSize << "x L2)\n"
                  << "Max logical copy per case: " << static_cast<double>(options.max_copy_bytes) / MIB << " MiB\n";

        printHeader();
        uint64_t skipped = 0;
        runDenseBenchmarks(options, stream, evictor, skipped);
        runRaggedBenchmarks(options, stream, evictor, skipped);
        runSequenceBenchmarks(options, stream, evictor, skipped);
        stream.synchronize();
        if (skipped != 0) std::cerr << "Skipped " << skipped << " geometries above --max-copy-mib.\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "thor_concatenate_benchmark: " << error.what() << '\n';
        return 1;
    }
}
