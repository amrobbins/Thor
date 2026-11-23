#include "Utilities/TensorOperations/Misc/Concatenate.h"
#include "Utilities/TensorOperations/Misc/Split.h"
#include "Utilities/TensorOperations/Ragged/RaggedSequenceConcatenate.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Common/Event.h"
#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/Expression/CudaHelpers.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
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
#include <utility>
#include <vector>

using namespace ThorImplementation;

namespace {

constexpr uint64_t MIB = 1024ULL * 1024ULL;
constexpr double BYTES_PER_GB = 1.0e9;

struct Options {
    int gpu = 0;
    int warmup = 5;
    int iterations = 20;
    // A timed sample is this many microseconds of back-to-back launches (at least one), so the event timer's
    // resolution is divided across them.
    double target_sample_us = 200.0;
    double l2_evict_multiple = 4.0;
    uint64_t min_evict_bytes = 64ULL * MIB;
    uint64_t max_copy_bytes = 128ULL * MIB;

    std::vector<std::string> kernels{
        "concatenate", "split", "convert-concatenate", "convert-split", "sequence-forward", "sequence-backward"};
    std::vector<uint64_t> outer_slices{1, 8, 64, 512, 4096, 32768};
    std::vector<uint32_t> num_arrays{2, 4, 8, 16};
    std::vector<uint64_t> span_bytes{4, 16, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513,
                                     1024, 2048, 4096, 16384, 65536};
    std::vector<std::string> span_layouts{"uniform", "skewed"};

    // The dense cases whose arrays are not all of the packed tensor's dtype (convert-concatenate, convert-split).
    std::vector<uint64_t> convert_outer_slices{1024, 16384};
    std::vector<uint32_t> convert_num_arrays{2, 8};
    std::vector<uint64_t> convert_span_elements{16, 128, 1024};
    std::vector<std::string> convert_dtypes{"bf16>bf16", "fp32+bf16>bf16", "fp32>bf16", "bf16>fp32", "fp8_e4m3+bf16>bf16",
                                            "fp32>fp8_e5m2", "fp8_e5m2>fp32", "uint8+fp32>fp32"};

    std::vector<uint64_t> sequence_batch_rows{32, 512, 8192};
    std::vector<uint32_t> sequence_num_inputs{2, 4, 8};
    std::vector<uint64_t> sequence_value_bytes{4, 33, 128, 2048};
    std::vector<uint64_t> sequence_values_per_span{1, 16, 256};
    std::vector<std::string> sequence_length_layouts{"uniform", "random", "skewed"};
    std::vector<double> sequence_nonempty_fractions{1.0, 0.5};
    std::vector<std::string> sequence_span_formats{"packed"};
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
        else if (arg == "--target-sample-us") options.target_sample_us = parseDouble(requireValue(arg), arg);
        else if (arg == "--l2-evict-multiple") options.l2_evict_multiple = parseDouble(requireValue(arg), arg);
        else if (arg == "--min-evict-mib") options.min_evict_bytes = parseUnsigned(requireValue(arg), arg) * MIB;
        else if (arg == "--max-copy-mib") options.max_copy_bytes = parseUnsigned(requireValue(arg), arg) * MIB;
        else if (arg == "--kernels") options.kernels = parseStringCsv(requireValue(arg), arg);
        else if (arg == "--outer-slices") options.outer_slices = parseUnsignedCsv(requireValue(arg), arg);
        else if (arg == "--num-arrays") options.num_arrays = parseUint32Csv(requireValue(arg), arg);
        else if (arg == "--span-bytes") options.span_bytes = parseUnsignedCsv(requireValue(arg), arg);
        else if (arg == "--span-layouts") options.span_layouts = parseStringCsv(requireValue(arg), arg);
        else if (arg == "--convert-outer-slices") options.convert_outer_slices = parseUnsignedCsv(requireValue(arg), arg);
        else if (arg == "--convert-num-arrays") options.convert_num_arrays = parseUint32Csv(requireValue(arg), arg);
        else if (arg == "--convert-span-elements") options.convert_span_elements = parseUnsignedCsv(requireValue(arg), arg);
        else if (arg == "--convert-dtypes") options.convert_dtypes = parseStringCsv(requireValue(arg), arg);
        else if (arg == "--sequence-batch-rows") options.sequence_batch_rows = parseUnsignedCsv(requireValue(arg), arg);
        else if (arg == "--sequence-num-inputs") options.sequence_num_inputs = parseUint32Csv(requireValue(arg), arg);
        else if (arg == "--sequence-length-layouts") options.sequence_length_layouts = parseStringCsv(requireValue(arg), arg);
        else if (arg == "--sequence-value-bytes") options.sequence_value_bytes = parseUnsignedCsv(requireValue(arg), arg);
        else if (arg == "--sequence-values-per-span") options.sequence_values_per_span = parseUnsignedCsv(requireValue(arg), arg);
        else if (arg == "--sequence-nonempty-fractions") options.sequence_nonempty_fractions = parseDoubleCsv(requireValue(arg), arg);
        else if (arg == "--sequence-span-formats") options.sequence_span_formats = parseStringCsv(requireValue(arg), arg);
        else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: thor_concatenate_benchmark [options]\n"
                << "  --gpu N\n"
                << "  --warmup N\n"
                << "  --iterations N\n"
                << "  --target-sample-us X               back-to-back launches per sample, about X us (default 200)\n"
                << "  --l2-evict-multiple X              untimed DRAM-read eviction / L2 (default 4.0)\n"
                << "  --min-evict-mib N                  minimum eviction buffer (default 64)\n"
                << "  --max-copy-mib N                   skip payloads larger than N MiB (default 128)\n"
                << "  --kernels concatenate,split,convert-concatenate,convert-split,sequence-forward,sequence-backward\n"
                << "      (the ragged trailing-axis concatenate is the dense one over rows x outer slices)\n"
                << "  --outer-slices a,b,c                dense slices before the concatenation axis\n"
                << "  --num-arrays 2,4,8,16\n"
                << "  --span-bytes a,b,c                  average bytes per dense/ragged span\n"
                << "  --span-layouts uniform,skewed       skewed keeps the total but gives one large span\n"
                << "  --convert-outer-slices a,b,c        the convert-* cases: slices before the concatenation axis\n"
                << "  --convert-num-arrays 2,8\n"
                << "  --convert-span-elements a,b,c       elements of every array's span\n"
                << "  --convert-dtypes a+b>p,...          the arrays' dtypes, taken in rotation, and the packed tensor's:\n"
                << "      fp32+bf16>bf16 is FP32 and BF16 arrays by turns beside a BF16 packed tensor (concatenated into\n"
                << "      it, or split out of it). Names: fp8_e4m3 fp8_e5m2 bf16 fp16 fp32 fp64 int8 uint8 int16 uint16\n"
                << "      int32 uint32 int64 uint64 boolean\n"
                << "  --sequence-batch-rows a,b,c\n"
                << "  --sequence-num-inputs 2,4,8\n"
                << "  --sequence-value-bytes a,b,c        bytes per packed sequence value (a token's features)\n"
                << "  --sequence-values-per-span 1,16,256 mean values (tokens) of a row of an input; the plan cuts a row\n"
                << "                                      longer than twice the batch's mean into spans of that length\n"
                << "  --sequence-length-layouts uniform,random,skewed\n"
                << "      uniform: every row the mean; random: uniform in [1, 2 mean - 1] (mean >= 2);\n"
                << "      skewed: one row of each input holds the tokens the others lack, the rest mean / 4 (mean >= 4)\n"
                << "  --sequence-nonempty-fractions 1,0.5\n"
                << "  --sequence-span-formats packed,wide  copy-plan spans of 8 or 32 bytes (default packed)\n\n"
                << "A sample is launchesPerSample back-to-back launches between two events, each over its own set of\n"
                << "inputs from a pool of at least 4x L2 and its own outputs; an untimed device-read pass of >L2 bytes\n"
                << "runs before every sample, so every launch reads DRAM-cold inputs. Each case is also measured as a\n"
                << "cudaMemcpyAsync of its payload (copy) and as an empty kernel (launch_floor), the same way. A sequence case is also\n"
                << "measured as the host-to-device copy of its span table from pinned memory (plan_upload), which the\n"
                << "layer issues on the same stream before every launch; plan_bytes is that table's size.\n"
                << "A convert-* case is measured as: kernel, the typed concatenate or split, which converts as it moves;\n"
                << "same_dtype, the byte concatenate or split with every array in the packed dtype; copy, a\n"
                << "cudaMemcpyAsync of half the bytes the operation reads and writes (the same memory traffic); and\n"
                << "launch_floor. Their read_write_GBps counts the arrays' bytes and the packed tensor's, once each.\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("Unknown argument: " + std::string(arg));
        }
    }

    if (options.gpu < 0) throw std::invalid_argument("--gpu must be non-negative");
    if (options.warmup < 0) throw std::invalid_argument("--warmup must be non-negative");
    if (options.iterations <= 0) throw std::invalid_argument("--iterations must be positive");
    if (!(options.target_sample_us > 0.0)) throw std::invalid_argument("--target-sample-us must be positive");
    if (options.l2_evict_multiple < 2.0) throw std::invalid_argument("--l2-evict-multiple must be >= 2.0");
    requireAllowed(options.kernels,
                   {"concatenate", "split", "convert-concatenate", "convert-split", "sequence-forward", "sequence-backward"},
                   "--kernels");
    requireAllowed(options.span_layouts, {"uniform", "skewed"}, "--span-layouts");
    for (uint32_t n : options.num_arrays) {
        if (n < 2 || n > 256) throw std::invalid_argument("--num-arrays values must be in [2,256]");
    }
    for (uint32_t n : options.convert_num_arrays) {
        if (n < 2 || n > 256) throw std::invalid_argument("--convert-num-arrays values must be in [2,256]");
    }
    for (uint32_t n : options.sequence_num_inputs) {
        if (n < 2 || n > 256) throw std::invalid_argument("--sequence-num-inputs values must be in [2,256]");
    }
    requireAllowed(options.sequence_length_layouts, {"uniform", "random", "skewed"}, "--sequence-length-layouts");
    for (uint64_t values : options.sequence_values_per_span) {
        if (values == 0) throw std::invalid_argument("--sequence-values-per-span values must be positive");
    }
    for (double fraction : options.sequence_nonempty_fractions) {
        if (!(fraction > 0.0 && fraction <= 1.0)) {
            throw std::invalid_argument("--sequence-nonempty-fractions values must be in (0,1]");
        }
    }
    requireAllowed(options.sequence_span_formats, {"packed", "wide"}, "--sequence-span-formats");
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

uint64_t ceilDiv(uint64_t numerator, uint64_t denominator) {
    return numerator / denominator + static_cast<uint64_t>(numerator % denominator != 0);
}

uint64_t roundUp(uint64_t value, uint64_t alignment) {
    return ceilDiv(value, alignment) * alignment;
}

uint64_t splitMix64(uint64_t value) {
    uint64_t z = value + 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27U)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31U);
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

// Pinned host bytes, as the sequence layer stages its span table.
class PinnedHostAllocation {
   public:
    explicit PinnedHostAllocation(uint64_t bytes) {
        if (bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            throw std::overflow_error("Benchmark pinned allocation exceeds size_t");
        }
        CUDA_CHECK(cudaHostAlloc(&pointer_, static_cast<size_t>(std::max<uint64_t>(bytes, 1)), cudaHostAllocPortable));
    }
    PinnedHostAllocation(const PinnedHostAllocation &) = delete;
    PinnedHostAllocation &operator=(const PinnedHostAllocation &) = delete;
    ~PinnedHostAllocation() {
        if (pointer_ != nullptr) (void)cudaFreeHost(pointer_);
    }
    void *get() const { return pointer_; }

   private:
    void *pointer_ = nullptr;
};

void initializeAllocation(DeviceAllocation &allocation, Stream &stream, uint8_t pattern) {
    if (allocation.bytes() != 0) {
        CUDA_CHECK(cudaMemsetAsync(allocation.get(), pattern, static_cast<size_t>(allocation.bytes()), stream.getStream()));
    }
}

// sets sets of the same arrays (the split arrays of one launch, or its packed tensor alone), carved from one allocation
// at 256-byte offsets so every array has a tensor's alignment; a host pointer table per set gives the launchers the
// set's arrays. A launch of a sample takes its own set, so its reads are not of lines the launch
// before it left in L2.
class ArrayPool {
   public:
    ArrayPool(uint32_t sets, const std::vector<uint64_t> &arrayBytes, uint8_t pattern, Stream &stream)
        : arrays_(static_cast<uint32_t>(arrayBytes.size())) {
        constexpr uint64_t kArrayAlignmentBytes = 256;
        uint64_t offset = 0;
        for (uint64_t bytes : arrayBytes) {
            arrayOffsets_.push_back(offset);
            offset = checkedAdd(offset, roundUp(bytes, kArrayAlignmentBytes), "Array pool set size overflow");
        }
        setStrideBytes_ = std::max<uint64_t>(offset, kArrayAlignmentBytes);
        storage_ = DeviceAllocation(checkedMultiply(sets, setStrideBytes_, "Array pool size overflow"));
        initializeAllocation(storage_, stream, pattern);
        hostTables_.resize(sets);
        for (uint32_t set = 0; set < sets; ++set) {
            for (uint32_t index = 0; index < arrays_; ++index) hostTables_[set].push_back(array(set, index));
        }
    }

    uint8_t *array(uint32_t set, uint32_t index) const {
        return static_cast<uint8_t *>(storage_.get()) + static_cast<uint64_t>(set) * setStrideBytes_ + arrayOffsets_[index];
    }
    // The set's arrays follow each other from here; the set is at least the payload.
    void *setBase(uint32_t set) const { return array(set, 0); }
    const std::vector<void *> &hostPointerTable(uint32_t set) const { return hostTables_[set]; }

   private:
    uint32_t arrays_ = 0;
    std::vector<uint64_t> arrayOffsets_;
    uint64_t setStrideBytes_ = 0;
    DeviceAllocation storage_;
    std::vector<std::vector<void *>> hostTables_;
};

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

__global__ void launchFloorKernel() {}

struct TimingStats {
    double p10_us = 0.0;
    double median_us = 0.0;
    double p90_us = 0.0;
    // CPU time to enqueue one launch (median over samples). When it reaches median_us, the GPU waits on the CPU and
    // the GPU time is the CPU's.
    double host_median_us = 0.0;
};

double percentileSorted(const std::vector<float> &values, double percentile) {
    if (values.empty()) return 0.0;
    const double position = percentile * static_cast<double>(values.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(position));
    const size_t hi = static_cast<size_t>(std::ceil(position));
    const double alpha = position - static_cast<double>(lo);
    return (1.0 - alpha) * static_cast<double>(values[lo]) + alpha * static_cast<double>(values[hi]);
}

// One timed sample: the eviction pass (untimed), then launchesPerSample back-to-back launches between two events; the
// per-launch time is the sample divided by the launch count. The CPU time of the launch loop, also per launch, is
// recorded beside it. warmupSamples untimed samples run first.
template <typename Fn>
TimingStats benchmarkLaunchSamples(Stream &stream,
                                   DramReadEvictor &evictor,
                                   int warmupSamples,
                                   int samples,
                                   uint32_t launchesPerSample,
                                   Fn &&launch) {
    for (int i = 0; i < warmupSamples; ++i) {
        evictor.run(stream);
        for (uint32_t k = 0; k < launchesPerSample; ++k) launch(k);
    }
    stream.synchronize();

    Event start(stream.getGpuNum(), /*enableTiming=*/true);
    Event stop(stream.getGpuNum(), /*enableTiming=*/true);
    std::vector<float> microsecondsPerLaunch;
    std::vector<float> hostMicrosecondsPerLaunch;
    microsecondsPerLaunch.reserve(samples);
    hostMicrosecondsPerLaunch.reserve(samples);
    for (int i = 0; i < samples; ++i) {
        evictor.run(stream);
        start.record(stream);
        const auto hostStart = std::chrono::steady_clock::now();
        for (uint32_t k = 0; k < launchesPerSample; ++k) launch(k);
        const auto hostStop = std::chrono::steady_clock::now();
        stop.record(stream);
        hostMicrosecondsPerLaunch.push_back(
            std::chrono::duration<float, std::micro>(hostStop - hostStart).count() / static_cast<float>(launchesPerSample));
        microsecondsPerLaunch.push_back(
            stop.synchronizeAndReportElapsedTimeInMilliseconds(start) * 1000.0f / static_cast<float>(launchesPerSample));
    }

    std::sort(microsecondsPerLaunch.begin(), microsecondsPerLaunch.end());
    std::sort(hostMicrosecondsPerLaunch.begin(), hostMicrosecondsPerLaunch.end());
    TimingStats stats;
    stats.p10_us = percentileSorted(microsecondsPerLaunch, 0.10);
    stats.median_us = percentileSorted(microsecondsPerLaunch, 0.50);
    stats.p90_us = percentileSorted(microsecondsPerLaunch, 0.90);
    stats.host_median_us = percentileSorted(hostMicrosecondsPerLaunch, 0.50);
    return stats;
}

// Launches per sample and sets per pool for a payload: about --target-sample-us of launches (an estimate of one launch
// is 2 us of launch and both directions at 1.4 TB/s), at most 256; a set per launch up to the sets that make a pool
// 4x L2, past which a launch's set was last touched 4x L2 of traffic ago.
struct SampleShape {
    uint32_t launchesPerSample = 1;
    uint32_t sets = 1;
};

SampleShape sampleShapeFor(const Options &options, const cudaDeviceProp &properties, uint64_t payloadBytes) {
    constexpr uint32_t kMaxLaunchesPerSample = 256;
    const double estimatedUs = std::max(2.0, 2.0 * static_cast<double>(payloadBytes) / 1.4e6);
    SampleShape shape;
    shape.launchesPerSample = static_cast<uint32_t>(
        std::clamp(std::ceil(options.target_sample_us / estimatedUs), 1.0, static_cast<double>(kMaxLaunchesPerSample)));
    const uint64_t coldSets = std::max<uint64_t>(1, ceilDiv(4 * static_cast<uint64_t>(properties.l2CacheSize), payloadBytes));
    shape.sets = static_cast<uint32_t>(std::min<uint64_t>(shape.launchesPerSample, coldSets));
    return shape;
}

struct Result {
    std::string kernel;
    std::string variant;
    uint64_t workRows = 0;
    uint64_t outerSlicesPerValue = 1;
    uint32_t numArrays = 0;
    uint64_t averageSpanBytes = 0;
    std::string layout;
    uint64_t valueBytes = 0;
    uint64_t valuesPerSpan = 0;
    double nonemptyFraction = 1.0;
    // The bytes of a copy span (sequence cases; 0 for dense).
    uint32_t spanBytes = 0;
    uint64_t spanCount = 0;
    // The bytes of the span table the launch reads (sequence cases; 0 for dense).
    uint64_t planBytes = 0;
    // The bytes the kernel reads and the bytes it writes (a convert-* case: the packed tensor's bytes).
    uint64_t payloadBytes = 0;
    // The bytes read plus the bytes written where they are not twice the payload (convert-* cases); 0 otherwise.
    uint64_t readWriteBytes = 0;
    // The arrays' dtypes and the packed tensor's (convert-* cases; empty otherwise).
    std::string dtypes;
    uint32_t launchesPerSample = 1;
    TimingStats timing;
};

double gbps(uint64_t bytes, double microseconds) {
    if (microseconds <= 0.0) return 0.0;
    return (static_cast<double>(bytes) / BYTES_PER_GB) / (microseconds * 1.0e-6);
}

void printHeader() {
    std::cout
        << "kernel,variant,work_rows,outer_slices_per_value,num_arrays,average_span_bytes,layout,value_bytes,"
           "values_per_span,nonempty_fraction,span_bytes,span_count,plan_bytes,payload_bytes,launches_per_sample,"
           "p10_us,median_us,p90_us,host_us,payload_GBps,read_write_GBps,dtypes\n";
}

void printResult(const Result &r) {
    // plan_upload moves the span table one way; launch_floor moves nothing.
    const bool planUpload = r.variant == "plan_upload";
    const bool floor = r.variant == "launch_floor";
    const uint64_t movedBytes = floor ? 0 : planUpload ? r.planBytes : r.payloadBytes;
    const uint64_t readWriteBytes = floor ? 0 : planUpload ? movedBytes : r.readWriteBytes != 0 ? r.readWriteBytes : 2 * movedBytes;
    std::cout << r.kernel << ',' << r.variant << ',' << r.workRows << ',' << r.outerSlicesPerValue << ','
              << r.numArrays << ',' << r.averageSpanBytes << ',' << r.layout << ',' << r.valueBytes << ','
              << r.valuesPerSpan << ',' << std::fixed << std::setprecision(3) << r.nonemptyFraction << ','
              << r.spanBytes << ',' << r.spanCount << ',' << r.planBytes << ',' << r.payloadBytes << ','
              << r.launchesPerSample << ','
              << std::setprecision(3) << r.timing.p10_us << ',' << r.timing.median_us << ',' << r.timing.p90_us << ','
              << r.timing.host_median_us << ','
              << std::setprecision(2) << gbps(movedBytes, r.timing.median_us) << ','
              << gbps(readWriteBytes, r.timing.median_us) << ',' << r.dtypes << '\n';
}

// A span table in pinned host memory and a device buffer to copy it to.
struct PlanUpload {
    const void *host = nullptr;
    void *device = nullptr;
    uint64_t bytes = 0;
};

// One direction of a case, measured three ways, each as samples of back-to-back launches: the kernel, launch k over
// set k of the source pool into set k of the destination pool; copy, a cudaMemcpyAsync of the payload between the same
// two sets, contiguous; launch_floor, an empty kernel. launch(set) launches the kernel over that set. With a plan, a
// fourth: plan_upload, the host-to-device copy of the span table.
template <typename Launch>
void runDirection(const Options &options,
                  Stream &stream,
                  DramReadEvictor &evictor,
                  const SampleShape &shape,
                  const ArrayPool &sourcePool,
                  const ArrayPool &destinationPool,
                  Result result,
                  Launch &&launch,
                  const PlanUpload *plan = nullptr) {
    std::vector<const char *> variants{"kernel", "copy", "launch_floor"};
    if (plan != nullptr) variants.push_back("plan_upload");
    for (const char *variant : variants) {
        const std::string_view name(variant);
        const TimingStats timing = benchmarkLaunchSamples(
            stream, evictor, options.warmup == 0 ? 0 : 1, options.iterations, shape.launchesPerSample, [&](uint32_t k) {
                const uint32_t set = k % shape.sets;
                if (name == "kernel") {
                    launch(set);
                } else if (name == "copy") {
                    CUDA_CHECK(cudaMemcpyAsync(destinationPool.setBase(set), sourcePool.setBase(set), result.payloadBytes,
                                               cudaMemcpyDeviceToDevice, stream.getStream()));
                } else if (name == "launch_floor") {
                    launchFloorKernel<<<1, 32, 0, stream.getStream()>>>();
                    CUDA_CHECK(cudaGetLastError());
                } else {
                    CUDA_CHECK(cudaMemcpyAsync(plan->device, plan->host, plan->bytes, cudaMemcpyHostToDevice,
                                               stream.getStream()));
                }
            });
        result.variant = variant;
        result.launchesPerSample = shape.launchesPerSample;
        result.timing = timing;
        printResult(result);
    }
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

void runDenseBenchmarks(const Options &options,
                        const cudaDeviceProp &properties,
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
                    const uint64_t payloadBytes = checkedMultiply(outerSlices, packedSliceBytes, "Dense payload bytes overflow");
                    if (payloadBytes > options.max_copy_bytes) {
                        ++skipped;
                        continue;
                    }
                    const SampleShape shape = sampleShapeFor(options, properties, payloadBytes);
                    std::vector<uint64_t> arrayBytes;
                    for (uint64_t width : widths) {
                        arrayBytes.push_back(checkedMultiply(outerSlices, width, "Dense split array bytes overflow"));
                    }
                    const ArrayPool splitPool(shape.sets, arrayBytes, 0x31, stream);
                    const ArrayPool packedPool(shape.sets, {payloadBytes}, 0xa7, stream);
                    // Bytes as UINT8 elements: one type throughout, so the byte kernels.
                    const std::vector<ThorImplementation::DataType> byteTypes(widths.size(), ThorImplementation::DataType::UINT8);
                    stream.synchronize();

                    Result result;
                    result.workRows = outerSlices;
                    result.numArrays = numArrays;
                    result.averageSpanBytes = averageSpanBytes;
                    result.layout = layout;
                    result.spanCount = checkedMultiply(outerSlices, numArrays, "Dense span count overflow");
                    result.payloadBytes = payloadBytes;
                    if (doConcat) {
                        result.kernel = "concatenate";
                        runDirection(options, stream, evictor, shape, splitPool, packedPool, result, [&](uint32_t set) {
                            launchConcatenate(packedPool.setBase(set), ThorImplementation::DataType::UINT8, splitPool.hostPointerTable(set), byteTypes,
                                              outerSlices, widths, stream);
                        });
                    }
                    if (doSplit) {
                        result.kernel = "split";
                        runDirection(options, stream, evictor, shape, packedPool, splitPool, result, [&](uint32_t set) {
                            launchSplit(splitPool.hostPointerTable(set), byteTypes, packedPool.setBase(set), ThorImplementation::DataType::UINT8,
                                        outerSlices, widths, stream);
                        });
                    }
                }
            }
        }
    }
}

// A dtype of a convert-* case.
struct ConvertDtype {
    std::string_view name;
    DataType dataType;
    uint64_t bytes;
};

constexpr ConvertDtype kConvertDtypes[] = {
    {"fp8_e4m3", DataType::FP8_E4M3, 1}, {"fp8_e5m2", DataType::FP8_E5M2, 1}, {"bf16", DataType::BF16, 2},
    {"fp16", DataType::FP16, 2},         {"fp32", DataType::FP32, 4},         {"fp64", DataType::FP64, 8},
    {"int8", DataType::INT8, 1},         {"uint8", DataType::UINT8, 1},       {"int16", DataType::INT16, 2},
    {"uint16", DataType::UINT16, 2},     {"int32", DataType::INT32, 4},       {"uint32", DataType::UINT32, 4},
    {"int64", DataType::INT64, 8},       {"uint64", DataType::UINT64, 8},     {"boolean", DataType::BOOLEAN, 1},
};

ConvertDtype parseConvertDtype(std::string_view name) {
    for (const ConvertDtype &dtype : kConvertDtypes) {
        if (dtype.name == name) return dtype;
    }
    throw std::invalid_argument("Unknown dtype in --convert-dtypes: " + std::string(name));
}

// "a+b>p": the arrays' dtypes, taken in rotation, and the packed tensor's.
struct ConvertMix {
    std::vector<ConvertDtype> arrays;
    ConvertDtype packed{};
};

ConvertMix parseConvertMix(std::string_view spec) {
    const size_t arrow = spec.find('>');
    if (arrow == std::string_view::npos || arrow == 0 || arrow + 1 == spec.size()) {
        throw std::invalid_argument("--convert-dtypes values are arrays>packed, as in fp32+bf16>bf16: " + std::string(spec));
    }
    ConvertMix mix;
    mix.packed = parseConvertDtype(spec.substr(arrow + 1));
    std::string_view arrays = spec.substr(0, arrow);
    while (true) {
        const size_t plus = arrays.find('+');
        mix.arrays.push_back(parseConvertDtype(arrays.substr(0, plus)));
        if (plus == std::string_view::npos) break;
        arrays.remove_prefix(plus + 1);
    }
    return mix;
}

// The dense concatenate and split when the arrays are not all of the packed tensor's dtype. `kernel` is the typed
// launch, which converts every element as it moves it. Beside it: `same_dtype`, the byte concatenate or split over
// arrays that are all in the packed dtype (the same packed bytes); `copy`, a cudaMemcpyAsync of half the bytes the
// operation reads and writes, so of the same memory traffic; and `launch_floor`. Spans are uniform, of
// --convert-span-elements elements.
void runConvertBenchmarks(const Options &options,
                          const cudaDeviceProp &properties,
                          Stream &stream,
                          DramReadEvictor &evictor,
                          uint64_t &skipped) {
    const bool doConcat = contains(options.kernels, "convert-concatenate");
    const bool doSplit = contains(options.kernels, "convert-split");
    if (!doConcat && !doSplit) return;

    for (const std::string &spec : options.convert_dtypes) {
        const ConvertMix mix = parseConvertMix(spec);
        for (uint64_t outerSlices : options.convert_outer_slices) {
            for (uint32_t numArrays : options.convert_num_arrays) {
                for (uint64_t spanElements : options.convert_span_elements) {
                    const uint64_t arrayElements = checkedMultiply(outerSlices, spanElements, "Convert array element count overflow");
                    const uint64_t packedBytes =
                        checkedMultiply(checkedMultiply(arrayElements, numArrays, "Convert element count overflow"), mix.packed.bytes,
                                        "Convert packed bytes overflow");
                    std::vector<DataType> arrayTypes;
                    const std::vector<uint64_t> spans(numArrays, spanElements);
                    std::vector<uint64_t> arrayBytes;
                    std::vector<uint64_t> packedTypeBytes;
                    uint64_t arraysBytes = 0;
                    for (uint32_t array = 0; array < numArrays; ++array) {
                        const ConvertDtype &dtype = mix.arrays[array % mix.arrays.size()];
                        arrayTypes.push_back(dtype.dataType);
                        arrayBytes.push_back(checkedMultiply(arrayElements, dtype.bytes, "Convert array bytes overflow"));
                        packedTypeBytes.push_back(arrayElements * mix.packed.bytes);
                        arraysBytes = checkedAdd(arraysBytes, arrayBytes.back(), "Convert arrays bytes overflow");
                    }
                    if (std::max(arraysBytes, packedBytes) > options.max_copy_bytes) {
                        ++skipped;
                        continue;
                    }
                    // Half the operation's traffic, on a 16-byte boundary.
                    const uint64_t copyBytes = roundUp((arraysBytes + packedBytes) / 2, 16);
                    const SampleShape shape = sampleShapeFor(options, properties, copyBytes);
                    // The arrays in their own dtypes; the packed tensor; the arrays in the packed dtype (those of
                    // `same_dtype`); the two sides of `copy`.
                    const ArrayPool arrayPool(shape.sets, arrayBytes, 0x31, stream);
                    const ArrayPool packedPool(shape.sets, {packedBytes}, 0xa7, stream);
                    const ArrayPool packedTypePool(shape.sets, packedTypeBytes, 0x31, stream);
                    const ArrayPool copySourcePool(shape.sets, {copyBytes}, 0x31, stream);
                    const ArrayPool copyDestinationPool(shape.sets, {copyBytes}, 0xa7, stream);
                    const std::vector<ThorImplementation::DataType> packedTypes(spans.size(), mix.packed.dataType);
                    stream.synchronize();

                    Result result;
                    result.workRows = outerSlices;
                    result.numArrays = numArrays;
                    result.averageSpanBytes = spanElements * mix.packed.bytes;
                    result.layout = "uniform";
                    result.valueBytes = mix.packed.bytes;
                    result.valuesPerSpan = spanElements;
                    result.spanCount = checkedMultiply(outerSlices, numArrays, "Convert span count overflow");
                    result.payloadBytes = packedBytes;
                    result.dtypes = spec;
                    result.launchesPerSample = shape.launchesPerSample;

                    for (const bool concatenate : {true, false}) {
                        if (!(concatenate ? doConcat : doSplit)) continue;
                        result.kernel = concatenate ? "convert-concatenate" : "convert-split";
                        for (const char *variant : {"kernel", "same_dtype", "copy", "launch_floor"}) {
                            const std::string_view name(variant);
                            result.timing = benchmarkLaunchSamples(
                                stream, evictor, options.warmup == 0 ? 0 : 1, options.iterations, shape.launchesPerSample, [&](uint32_t k) {
                                    const uint32_t set = k % shape.sets;
                                    if (name == "kernel" && concatenate) {
                                        launchConcatenate(packedPool.setBase(set), mix.packed.dataType, arrayPool.hostPointerTable(set), arrayTypes,
                                                          outerSlices, spans, stream);
                                    } else if (name == "kernel") {
                                        launchSplit(arrayPool.hostPointerTable(set), arrayTypes, packedPool.setBase(set), mix.packed.dataType,
                                                    outerSlices, spans, stream);
                                    } else if (name == "same_dtype" && concatenate) {
                                        launchConcatenate(packedPool.setBase(set), mix.packed.dataType, packedTypePool.hostPointerTable(set), packedTypes,
                                                          outerSlices, spans, stream);
                                    } else if (name == "same_dtype") {
                                        launchSplit(packedTypePool.hostPointerTable(set), packedTypes, packedPool.setBase(set), mix.packed.dataType,
                                                    outerSlices, spans, stream);
                                    } else if (name == "copy") {
                                        CUDA_CHECK(cudaMemcpyAsync(copyDestinationPool.setBase(set), copySourcePool.setBase(set), copyBytes,
                                                                   cudaMemcpyDeviceToDevice, stream.getStream()));
                                    } else {
                                        launchFloorKernel<<<1, 32, 0, stream.getStream()>>>();
                                        CUDA_CHECK(cudaGetLastError());
                                    }
                                });
                            result.variant = variant;
                            // The bytes read and written: the arrays' and the packed tensor's, each once.
                            result.readWriteBytes = name == "kernel" ? arraysBytes + packedBytes : name == "copy" ? 2 * copyBytes : 2 * packedBytes;
                            printResult(result);
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

// The values (tokens) of row `row` of input `input`, for a mean of meanValues a row; see --sequence-length-layouts.
uint64_t sequenceRowValues(std::string_view layout, uint64_t row, uint32_t input, uint64_t batchRows, uint64_t meanValues) {
    if (layout == "uniform") return meanValues;
    if (layout == "random") {
        const uint64_t range = checkedAdd(meanValues, meanValues - 1, "Sequence random length range overflow");
        return 1 + splitMix64(row * 0xd1b54a32d192ed03ULL + input * 0x8cb92ba72f3d8dd7ULL + 0x632be59bd9b4e019ULL) % range;
    }
    const uint64_t shortValues = std::max<uint64_t>(1, meanValues / 4);
    const uint64_t longRow = (static_cast<uint64_t>(input) * 7919ULL + batchRows / 2) % batchRows;
    if (row != longRow) return shortValues;
    return checkedMultiply(meanValues, batchRows, "Sequence skewed length overflow") - shortValues * (batchRows - 1);
}

// Whether a length layout is distinct from uniform at this mean (random needs 2, skewed 4 and two rows).
bool sequenceLayoutApplies(std::string_view layout, uint64_t batchRows, uint64_t meanValues) {
    if (layout == "random") return meanValues >= 2;
    if (layout == "skewed") return meanValues >= 4 && batchRows >= 2;
    return true;
}

// The span table StampedNetwork builds: row by row, input by input, the run of every non-empty (row, input), cut at the
// plan's span limit.
template <typename SpanT>
SequencePlan<SpanT> makeSequencePlan(uint64_t batchRows,
                                     uint32_t numInputs,
                                     uint64_t meanValues,
                                     std::string_view layout,
                                     double nonemptyFraction) {
    SequencePlan<SpanT> result;
    result.tokensPerInput.assign(numInputs, 0);
    const uint64_t threshold = static_cast<uint64_t>(std::llround(nonemptyFraction * 1000000.0));
    const auto runValues = [&](uint64_t row, uint32_t input) -> uint64_t {
        const bool nonempty = nonemptyFraction >= 1.0 ||
            (splitMix64(row * 0x9e3779b97f4a7c15ULL + input * 0xbf58476d1ce4e5b9ULL) % 1000000ULL) < threshold;
        return nonempty ? sequenceRowValues(layout, row, input, batchRows, meanValues) : 0;
    };
    uint64_t runs = 0;
    uint64_t activeValues = 0;
    for (uint64_t row = 0; row < batchRows; ++row) {
        for (uint32_t input = 0; input < numInputs; ++input) {
            const uint64_t values = runValues(row, input);
            runs += values != 0;
            activeValues = checkedAdd(activeValues, values, "Sequence output token count overflow");
        }
    }
    const uint64_t valueLimit = raggedSequenceCopySpanValueLimit(activeValues, runs);
    result.spans.resize(static_cast<size_t>(raggedSequenceCopySpanCapacity(runs)));
    uint64_t spanCount = 0;
    uint64_t outputBegin = 0;
    for (uint64_t row = 0; row < batchRows; ++row) {
        for (uint32_t input = 0; input < numInputs; ++input) {
            const uint64_t values = runValues(row, input);
            if (values == 0) continue;
            spanCount = appendRaggedSequenceCopyRun(result.spans.data(), spanCount, result.spans.size(), input,
                                                    result.tokensPerInput[input], outputBegin, values, valueLimit,
                                                    raggedSequenceCopySpanInputBits(numInputs));
            result.tokensPerInput[input] += values;
            outputBegin += values;
        }
    }
    result.spans.resize(static_cast<size_t>(spanCount));
    result.activeOutputValues = outputBegin;
    return result;
}

template <typename SpanT>
void runOneSequenceGeometry(const Options &options,
                            const cudaDeviceProp &properties,
                            Stream &stream,
                            DramReadEvictor &evictor,
                            uint64_t batchRows,
                            uint32_t numInputs,
                            uint64_t valueBytes,
                            uint64_t valuesPerSpan,
                            const std::string &layout,
                            double nonemptyFraction,
                            RaggedSequenceCopySpanFormat spanFormat,
                            uint64_t &skipped) {
    const bool doForward = contains(options.kernels, "sequence-forward");
    const bool doBackward = contains(options.kernels, "sequence-backward");
    if (!doForward && !doBackward) return;

    // The mean decides the payload before the plan is built (random and skewed keep it on average or exactly).
    const uint64_t meanPayloadBytes = checkedMultiply(
        checkedMultiply(checkedMultiply(batchRows, numInputs, "Sequence payload bytes overflow"), valuesPerSpan,
                        "Sequence payload bytes overflow"),
        valueBytes, "Sequence payload bytes overflow");
    if (static_cast<double>(meanPayloadBytes) * nonemptyFraction > 1.1 * static_cast<double>(options.max_copy_bytes)) {
        ++skipped;
        return;
    }
    const SequencePlan<SpanT> plan = makeSequencePlan<SpanT>(batchRows, numInputs, valuesPerSpan, layout, nonemptyFraction);
    if (plan.spans.empty()) return;
    const uint64_t payloadBytes = checkedMultiply(plan.activeOutputValues, valueBytes, "Sequence payload bytes overflow");
    if (payloadBytes > options.max_copy_bytes) {
        ++skipped;
        return;
    }
    const SampleShape shape = sampleShapeFor(options, properties, payloadBytes);
    std::vector<uint64_t> arrayBytes;
    for (uint64_t tokens : plan.tokensPerInput) {
        arrayBytes.push_back(checkedMultiply(tokens, valueBytes, "Sequence input array bytes overflow"));
    }
    const ArrayPool inputPool(shape.sets, arrayBytes, 0x53, stream);
    const ArrayPool outputPool(shape.sets, {payloadBytes}, 0xc4, stream);
    const DeviceAllocation copySpans = uploadVector(plan.spans, stream);
    const uint64_t planBytes = checkedMultiply(plan.spans.size(), sizeof(SpanT), "Sequence plan bytes overflow");
    PinnedHostAllocation planHost(planBytes);
    std::memcpy(planHost.get(), plan.spans.data(), static_cast<size_t>(planBytes));
    const DeviceAllocation planDevice(planBytes);
    const PlanUpload planUpload{planHost.get(), planDevice.get(), planBytes};
    stream.synchronize();

    Result result;
    result.workRows = batchRows;
    result.numArrays = numInputs;
    result.averageSpanBytes = payloadBytes / plan.spans.size();
    result.layout = layout;
    result.valueBytes = valueBytes;
    result.valuesPerSpan = valuesPerSpan;
    result.nonemptyFraction = nonemptyFraction;
    result.spanBytes = static_cast<uint32_t>(sizeof(SpanT));
    result.spanCount = plan.spans.size();
    result.planBytes = planBytes;
    result.payloadBytes = payloadBytes;
    if (doForward) {
        result.kernel = "sequence-forward";
        runDirection(options, stream, evictor, shape, inputPool, outputPool, result, [&](uint32_t set) {
            launchRaggedSequenceConcatenate(outputPool.setBase(set), inputPool.hostPointerTable(set), copySpans.get(),
                                            plan.spans.size(), 1, valueBytes, spanFormat, plan.activeOutputValues, stream);
        }, &planUpload);
    }
    if (doBackward) {
        result.kernel = "sequence-backward";
        runDirection(options, stream, evictor, shape, outputPool, inputPool, result, [&](uint32_t set) {
            launchRaggedSequenceConcatenateBackward(inputPool.hostPointerTable(set), outputPool.setBase(set), copySpans.get(),
                                                    plan.spans.size(), 1, valueBytes, spanFormat, plan.activeOutputValues,
                                                    stream);
        }, &planUpload);
    }
}

void runSequenceBenchmarks(const Options &options,
                           const cudaDeviceProp &properties,
                           Stream &stream,
                           DramReadEvictor &evictor,
                           uint64_t &skipped) {
    if (!contains(options.kernels, "sequence-forward") && !contains(options.kernels, "sequence-backward")) return;
    for (uint64_t batchRows : options.sequence_batch_rows) {
        for (uint32_t numInputs : options.sequence_num_inputs) {
            for (uint64_t valueBytes : options.sequence_value_bytes) {
                for (uint64_t valuesPerSpan : options.sequence_values_per_span) {
                    for (const std::string &layout : options.sequence_length_layouts) {
                        if (!sequenceLayoutApplies(layout, batchRows, valuesPerSpan)) continue;
                        for (double fraction : options.sequence_nonempty_fractions) {
                            for (const std::string &spanFormat : options.sequence_span_formats) {
                                if (spanFormat == "packed") {
                                    runOneSequenceGeometry<RaggedSequenceCopySpanPacked>(
                                        options, properties, stream, evictor, batchRows, numInputs, valueBytes,
                                        valuesPerSpan, layout, fraction, RaggedSequenceCopySpanFormat::PACKED, skipped);
                                } else {
                                    runOneSequenceGeometry<RaggedSequenceCopySpan64>(
                                        options, properties, stream, evictor, batchRows, numInputs, valueBytes,
                                        valuesPerSpan, layout, fraction, RaggedSequenceCopySpanFormat::WIDE, skipped);
                                }
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
        for (const std::string &spec : options.convert_dtypes) (void)parseConvertMix(spec);
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
                  << "Max payload per case: " << static_cast<double>(options.max_copy_bytes) / MIB << " MiB\n";

        printHeader();
        uint64_t skipped = 0;
        runDenseBenchmarks(options, properties, stream, evictor, skipped);
        runConvertBenchmarks(options, properties, stream, evictor, skipped);
        runSequenceBenchmarks(options, properties, stream, evictor, skipped);
        stream.synchronize();
        if (skipped != 0) std::cerr << "Skipped " << skipped << " geometries above --max-copy-mib.\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "thor_concatenate_benchmark: " << error.what() << '\n';
        return 1;
    }
}
