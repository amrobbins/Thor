#include "Utilities/TensorOperations/Ragged/RaggedAccuracy.h"
#include "Utilities/TensorOperations/Ragged/RaggedGather.h"
#include "Utilities/TensorOperations/Ragged/RaggedWeightedReduction.h"

#include "DeepLearning/Implementation/Tensor/Tensor.h"
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
#include <utility>
#include <vector>

using namespace ThorImplementation;

namespace {

constexpr uint64_t MIB = 1024ULL * 1024ULL;
constexpr double BYTES_PER_GB = 1.0e9;
constexpr uint64_t REDUCTION_ITEMS_PER_PARTIAL = 256ULL * 8ULL;

struct Options {
    int gpu = 0;
    int warmup = 5;
    int iterations = 20;
    double l2_evict_multiple = 4.0;
    uint64_t min_evict_bytes = 64ULL * MIB;
    uint64_t max_payload_bytes = 256ULL * MIB;
    bool verify = true;

    std::vector<std::string> kernels{
        "gather-forward", "gather-backward", "weighted-reduction", "binary-accuracy", "categorical-accuracy"};

    std::vector<uint64_t> batch_rows{32, 512, 8192};
    std::vector<uint64_t> source_values_per_row{32};
    std::vector<uint64_t> gather_values_per_row{1, 8};
    std::vector<uint64_t> gather_value_bytes{4, 16, 31, 32, 33, 64, 128, 256, 512};
    std::vector<std::string> row_layouts{"uniform", "skewed"};
    std::vector<uint32_t> offset_bytes{4, 8};
    std::vector<uint32_t> index_bytes{4};
    std::vector<std::string> gather_patterns{"unique", "clustered", "repeated"};
    std::vector<double> collision_rates{0.25, 0.875};
    std::vector<uint32_t> gather_copy_bytes{0};
    std::vector<uint32_t> gather_lanes{0};
    std::vector<std::string> gather_backward_dtypes{"fp32"};

    std::vector<uint64_t> active_values{127, 128, 129, 255, 256, 257, 511, 512, 513, 2047, 2048, 2049, 4095, 4096, 4097, 16384};
    std::vector<uint64_t> reduction_elements{1, 4, 16};
    std::vector<std::string> reduction_dtypes{"fp32"};
    std::vector<uint32_t> partial_blocks{0};

    std::vector<std::string> prediction_dtypes{"fp16", "fp32"};
    std::vector<std::string> binary_label_dtypes{"u8"};

    std::vector<uint64_t> categorical_active_values{128, 2048, 8192};
    std::vector<uint64_t> class_counts{
        2, 4, 8, 16, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 512, 1024};
    std::vector<std::string> categorical_label_formats{"index"};
    std::vector<uint32_t> categorical_lanes{0};
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
    if (consumed != text.size() || !std::isfinite(value))
        throw std::invalid_argument("Invalid value for " + std::string(flag));
    return value;
}

template <typename ParseFn>
auto parseCsv(std::string_view text, std::string_view flag, ParseFn&& parseOne) {
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
        if (value == 0 || value > std::numeric_limits<uint32_t>::max())
            throw std::invalid_argument(std::string(flag) + " values must fit uint32 and be non-zero");
        return static_cast<uint32_t>(value);
    });
}

std::vector<double> parseDoubleCsv(std::string_view text, std::string_view flag) {
    return parseCsv(text, flag, [&](std::string_view token) { return parseDouble(token, flag); });
}

std::vector<std::string> parseStringCsv(std::string_view text, std::string_view flag) {
    return parseCsv(text, flag, [](std::string_view token) { return std::string(token); });
}

std::vector<uint32_t> parseAutoPowerOfTwoCsv(std::string_view text,
                                             std::string_view flag,
                                             uint32_t maxValue,
                                             uint32_t minValue = 1) {
    return parseCsv(text, flag, [&](std::string_view token) -> uint32_t {
        if (token == "auto") return 0;
        const uint64_t value = parseUnsigned(token, flag);
        if (value < minValue || value > maxValue || (value & (value - 1)) != 0)
            throw std::invalid_argument(std::string(flag) + " values must be auto or an allowed power of two");
        return static_cast<uint32_t>(value);
    });
}

std::vector<uint32_t> parseAutoUint32Csv(std::string_view text,
                                           std::string_view flag,
                                           uint32_t maxValue) {
    return parseCsv(text, flag, [&](std::string_view token) -> uint32_t {
        if (token == "auto") return 0;
        const uint64_t value = parseUnsigned(token, flag);
        if (value == 0 || value > maxValue)
            throw std::invalid_argument(std::string(flag) + " values must be auto or in [1," +
                                        std::to_string(maxValue) + "]");
        return static_cast<uint32_t>(value);
    });
}

std::vector<uint32_t> parseCopyWidthCsv(std::string_view text, std::string_view flag) {
    return parseCsv(text, flag, [&](std::string_view token) -> uint32_t {
        if (token == "auto") return 0;
        const uint64_t value = parseUnsigned(token, flag);
        if (!(value == 1 || value == 2 || value == 4 || value == 8 || value == 16))
            throw std::invalid_argument(std::string(flag) + " values must be auto,1,2,4,8,16");
        return static_cast<uint32_t>(value);
    });
}

void requireAllowed(const std::vector<std::string>& values,
                    std::initializer_list<std::string_view> allowed,
                    std::string_view flag) {
    for (const std::string& value : values) {
        bool ok = false;
        for (std::string_view candidate : allowed) ok = ok || value == candidate;
        if (!ok) throw std::invalid_argument("Unsupported " + std::string(flag) + " value: " + value);
    }
}

bool contains(const std::vector<std::string>& values, std::string_view value) {
    return std::find(values.begin(), values.end(), value) != values.end();
}

Options parseOptions(int argc, char** argv) {
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
        else if (arg == "--max-payload-mib") options.max_payload_bytes = parseUnsigned(requireValue(arg), arg) * MIB;
        else if (arg == "--kernels") options.kernels = parseStringCsv(requireValue(arg), arg);
        else if (arg == "--batch-rows") options.batch_rows = parseUnsignedCsv(requireValue(arg), arg);
        else if (arg == "--source-values-per-row") options.source_values_per_row = parseUnsignedCsv(requireValue(arg), arg);
        else if (arg == "--gather-values-per-row") options.gather_values_per_row = parseUnsignedCsv(requireValue(arg), arg);
        else if (arg == "--gather-value-bytes") options.gather_value_bytes = parseUnsignedCsv(requireValue(arg), arg);
        else if (arg == "--row-layouts") options.row_layouts = parseStringCsv(requireValue(arg), arg);
        else if (arg == "--offset-bytes") options.offset_bytes = parseUint32Csv(requireValue(arg), arg);
        else if (arg == "--index-bytes") options.index_bytes = parseUint32Csv(requireValue(arg), arg);
        else if (arg == "--gather-patterns") options.gather_patterns = parseStringCsv(requireValue(arg), arg);
        else if (arg == "--collision-rates") options.collision_rates = parseDoubleCsv(requireValue(arg), arg);
        else if (arg == "--gather-copy-bytes") options.gather_copy_bytes = parseCopyWidthCsv(requireValue(arg), arg);
        else if (arg == "--gather-lanes") options.gather_lanes = parseAutoPowerOfTwoCsv(requireValue(arg), arg, 32);
        else if (arg == "--gather-backward-dtypes") options.gather_backward_dtypes = parseStringCsv(requireValue(arg), arg);
        else if (arg == "--active-values") options.active_values = parseUnsignedCsv(requireValue(arg), arg);
        else if (arg == "--reduction-elements") options.reduction_elements = parseUnsignedCsv(requireValue(arg), arg);
        else if (arg == "--reduction-dtypes") options.reduction_dtypes = parseStringCsv(requireValue(arg), arg);
        else if (arg == "--partial-blocks") options.partial_blocks = parseAutoUint32Csv(requireValue(arg), arg, 4096);
        else if (arg == "--prediction-dtypes") options.prediction_dtypes = parseStringCsv(requireValue(arg), arg);
        else if (arg == "--binary-label-dtypes") options.binary_label_dtypes = parseStringCsv(requireValue(arg), arg);
        else if (arg == "--categorical-active-values") options.categorical_active_values = parseUnsignedCsv(requireValue(arg), arg);
        else if (arg == "--class-counts") options.class_counts = parseUnsignedCsv(requireValue(arg), arg);
        else if (arg == "--categorical-label-formats") options.categorical_label_formats = parseStringCsv(requireValue(arg), arg);
        else if (arg == "--categorical-lanes") options.categorical_lanes = parseAutoPowerOfTwoCsv(requireValue(arg), arg, 32, 2);
        else if (arg == "--no-verify") options.verify = false;
        else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: thor_ragged_primitive_benchmark [options]\n"
                << "  --gpu N\n"
                << "  --warmup N\n"
                << "  --iterations N\n"
                << "  --l2-evict-multiple X               untimed device-read eviction / L2 (default 4.0)\n"
                << "  --min-evict-mib N                   minimum eviction buffer (default 64)\n"
                << "  --max-payload-mib N                 skip cases whose primary allocations exceed N MiB\n"
                << "  --kernels gather-forward,gather-backward,weighted-reduction,binary-accuracy,categorical-accuracy\n"
                << "  --batch-rows a,b,c\n"
                << "  --source-values-per-row a,b,c\n"
                << "  --gather-values-per-row a,b,c\n"
                << "  --gather-value-bytes a,b,c           forward uses UINT8 so odd byte widths are measurable\n"
                << "  --row-layouts uniform,skewed,empty\n"
                << "  --offset-bytes 4,8\n"
                << "  --index-bytes 4,8\n"
                << "  --gather-patterns unique,clustered,repeated\n"
                << "  --collision-rates 0.5,0.875          used only for repeated gather patterns\n"
                << "  --gather-copy-bytes auto,1,2,4,8,16\n"
                << "  --gather-lanes auto,1,2,4,8,16,32\n"
                << "  --gather-backward-dtypes fp16,bf16,fp32\n"
                << "  --active-values a,b,c                weighted reduction + binary accuracy active prefix\n"
                << "  --reduction-elements a,b,c           trailing scalar count per ragged value\n"
                << "  --reduction-dtypes fp16,bf16,fp32\n"
                << "  --partial-blocks auto,1,2,3,...,4096 force first-pass CTA count\n"
                << "  --prediction-dtypes fp16,fp32\n"
                << "  --binary-label-dtypes u8,fp32\n"
                << "  --categorical-active-values a,b,c\n"
                << "  --class-counts a,b,c\n"
                << "  --categorical-label-formats index,per-class\n"
                << "  --categorical-lanes auto,2,4,8,16,32\n"
                << "  --no-verify                           skip representative CPU/reference checks\n\n"
                << "Each case is warmed first. Before every timed sample an untimed device-read pass\n"
                << "over >L2 data runs on the same stream; CUDA events time only the full production\n"
                << "operation after eviction. Benchmark-only forced modes call the same kernels.\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("Unknown argument: " + std::string(arg));
        }
    }

    if (options.gpu < 0) throw std::invalid_argument("--gpu must be non-negative");
    if (options.warmup < 0) throw std::invalid_argument("--warmup must be non-negative");
    if (options.iterations <= 0) throw std::invalid_argument("--iterations must be positive");
    if (options.l2_evict_multiple < 2.0)
        throw std::invalid_argument("--l2-evict-multiple must be >= 2.0");
    requireAllowed(options.kernels,
                   {"gather-forward", "gather-backward", "weighted-reduction", "binary-accuracy", "categorical-accuracy"},
                   "--kernels");
    requireAllowed(options.row_layouts, {"uniform", "skewed", "empty"}, "--row-layouts");
    requireAllowed(options.gather_patterns, {"unique", "clustered", "repeated"}, "--gather-patterns");
    requireAllowed(options.gather_backward_dtypes, {"fp16", "bf16", "fp32"}, "--gather-backward-dtypes");
    requireAllowed(options.reduction_dtypes, {"fp16", "bf16", "fp32"}, "--reduction-dtypes");
    requireAllowed(options.prediction_dtypes, {"fp16", "fp32"}, "--prediction-dtypes");
    requireAllowed(options.binary_label_dtypes, {"u8", "fp32"}, "--binary-label-dtypes");
    requireAllowed(options.categorical_label_formats, {"index", "per-class"}, "--categorical-label-formats");
    for (uint32_t bytes : options.offset_bytes)
        if (!(bytes == 4 || bytes == 8)) throw std::invalid_argument("--offset-bytes values must be 4 or 8");
    for (uint32_t bytes : options.index_bytes)
        if (!(bytes == 4 || bytes == 8)) throw std::invalid_argument("--index-bytes values must be 4 or 8");
    for (double rate : options.collision_rates)
        if (!(rate >= 0.0 && rate < 1.0)) throw std::invalid_argument("--collision-rates values must be in [0,1)");
    for (uint64_t classes : options.class_counts)
        if (classes < 2) throw std::invalid_argument("--class-counts values must be >= 2");
    return options;
}

uint64_t checkedAdd(uint64_t lhs, uint64_t rhs, const char* what) {
    if (lhs > std::numeric_limits<uint64_t>::max() - rhs) throw std::overflow_error(what);
    return lhs + rhs;
}

uint64_t checkedMultiply(uint64_t lhs, uint64_t rhs, const char* what) {
    if (rhs != 0 && lhs > std::numeric_limits<uint64_t>::max() / rhs) throw std::overflow_error(what);
    return lhs * rhs;
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

DataType floatingDtype(std::string_view name) {
    if (name == "fp16") return DataType::FP16;
    if (name == "bf16") return DataType::BF16;
    if (name == "fp32") return DataType::FP32;
    throw std::invalid_argument("Unsupported floating dtype: " + std::string(name));
}

DataType binaryLabelDtype(std::string_view name) {
    if (name == "u8") return DataType::UINT8;
    if (name == "fp32") return DataType::FP32;
    throw std::invalid_argument("Unsupported binary label dtype: " + std::string(name));
}

uint64_t dtypeBytes(DataType dtype) {
    switch (dtype) {
        case DataType::UINT8: return 1;
        case DataType::UINT16: return 2;
        case DataType::UINT32: return 4;
        case DataType::UINT64: return 8;
        case DataType::FP16: return 2;
        case DataType::BF16: return 2;
        case DataType::FP32: return 4;
        default: throw std::invalid_argument("Benchmark dtype byte width is not defined for this dtype");
    }
}

template <typename T>
DataType dtypeFor();
template <> DataType dtypeFor<uint8_t>() { return DataType::UINT8; }
template <> DataType dtypeFor<uint32_t>() { return DataType::UINT32; }
template <> DataType dtypeFor<uint64_t>() { return DataType::UINT64; }
template <> DataType dtypeFor<float>() { return DataType::FP32; }

template <typename T>
Tensor makeGpuTensor(int gpu, const std::vector<uint64_t>& dimensions, const std::vector<T>& values, Stream& stream) {
    const TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    const TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, gpu);
    Tensor host(cpuPlacement, TensorDescriptor(dtypeFor<T>(), dimensions));
    if (host.getTotalNumElements() != values.size())
        throw std::runtime_error("Benchmark tensor value count mismatch");
    std::copy(values.begin(), values.end(), host.getMemPtr<T>());
    Tensor device(gpuPlacement, TensorDescriptor(dtypeFor<T>(), dimensions));
    device.copyFromAsync(host, stream);
    stream.synchronize();
    return device;
}

template <typename T>
std::vector<T> copyGpuTensor(const Tensor& device, Stream& stream) {
    const TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    Tensor host(cpuPlacement, TensorDescriptor(dtypeFor<T>(), device.getDimensions()));
    host.copyFromAsync(device, stream);
    stream.synchronize();
    const T* values = host.getMemPtr<T>();
    return std::vector<T>(values, values + host.getTotalNumElements());
}

class DeviceAllocation {
   public:
    DeviceAllocation() = default;
    explicit DeviceAllocation(uint64_t bytes) : bytes_(bytes) {
        constexpr uint64_t kThorPaddingBytes = 128;
        const uint64_t logicalBytes = std::max<uint64_t>(bytes, 1);
        if (logicalBytes > std::numeric_limits<uint64_t>::max() - kThorPaddingBytes)
            throw std::overflow_error("Benchmark allocation plus Thor padding overflows uint64");
        const uint64_t allocationBytes = logicalBytes + kThorPaddingBytes;
        if (allocationBytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
            throw std::overflow_error("Benchmark allocation exceeds size_t");
        CUDA_CHECK(cudaMalloc(&pointer_, static_cast<size_t>(allocationBytes)));
    }
    DeviceAllocation(const DeviceAllocation&) = delete;
    DeviceAllocation& operator=(const DeviceAllocation&) = delete;
    DeviceAllocation(DeviceAllocation&& other) noexcept : pointer_(other.pointer_), bytes_(other.bytes_) {
        other.pointer_ = nullptr;
        other.bytes_ = 0;
    }
    DeviceAllocation& operator=(DeviceAllocation&& other) noexcept {
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
    void* get() const { return pointer_; }
    uint64_t bytes() const { return bytes_; }

   private:
    void* pointer_ = nullptr;
    uint64_t bytes_ = 0;
};

__global__ void readForL2EvictionKernel(const uint4* __restrict__ bytes,
                                        uint64_t uint4Count,
                                        unsigned long long* __restrict__ sink) {
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
    DramReadEvictor(const Options& options, const cudaDeviceProp& properties, Stream& stream)
        : bytes_(std::max<uint64_t>(options.min_evict_bytes,
                                    static_cast<uint64_t>(std::ceil(
                                        options.l2_evict_multiple * static_cast<double>(properties.l2CacheSize))))),
          storage_(roundUp(bytes_, sizeof(uint4))),
          sink_(sizeof(uint64_t)),
          blocks_(std::max(1, properties.multiProcessorCount * 8)) {
        CUDA_CHECK(cudaMemsetAsync(storage_.get(), 0x5a, static_cast<size_t>(storage_.bytes()), stream.getStream()));
        CUDA_CHECK(cudaMemsetAsync(sink_.get(), 0, static_cast<size_t>(sink_.bytes()), stream.getStream()));
        stream.synchronize();
        run(stream);
        stream.synchronize();
    }

    void run(Stream& stream) {
        constexpr uint32_t threads = 256;
        readForL2EvictionKernel<<<blocks_, threads, 0, stream.getStream()>>>(
            reinterpret_cast<const uint4*>(storage_.get()), storage_.bytes() / sizeof(uint4),
            reinterpret_cast<unsigned long long*>(sink_.get()));
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

double percentileSorted(const std::vector<float>& values, double percentile) {
    if (values.empty()) return 0.0;
    const double position = percentile * static_cast<double>(values.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(position));
    const size_t hi = static_cast<size_t>(std::ceil(position));
    const double alpha = position - static_cast<double>(lo);
    return (1.0 - alpha) * static_cast<double>(values[lo]) + alpha * static_cast<double>(values[hi]);
}

template <typename Fn>
TimingStats benchmarkDramReads(Stream& stream, DramReadEvictor& evictor, int warmup, int iterations, Fn&& fn) {
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
    std::string rowLayout;
    std::string gatherPattern;
    double collisionRate = 0.0;
    uint64_t batchRows = 0;
    uint64_t nominalSourceValuesPerRow = 0;
    uint64_t gatherValuesPerRow = 0;
    uint64_t sourceRowMin = 0;
    uint64_t sourceRowMax = 0;
    uint64_t indexRowMin = 0;
    uint64_t indexRowMax = 0;
    uint64_t sourceActiveValues = 0;
    uint64_t gatheredActiveValues = 0;
    uint64_t activeValues = 0;
    uint64_t activeScalars = 0;
    uint64_t valueBytes = 0;
    uint64_t elementsPerValue = 0;
    uint64_t numClasses = 0;
    uint32_t offsetBytes = 0;
    uint32_t indexBytes = 0;
    std::string predictionDtype;
    std::string labelDtype;
    std::string labelFormat;
    std::string selectionMode;
    uint32_t blocks = 0;
    uint32_t autoCopyWidth = 0;
    uint32_t copyWidth = 0;
    uint32_t autoLanesPerToken = 0;
    uint32_t lanesPerToken = 0;
    uint32_t autoTokensPerBlock = 0;
    uint32_t tokensPerBlock = 0;
    uint64_t copyItemsPerValue = 0;
    uint32_t autoPartialBlocks = 0;
    uint32_t partialBlocks = 0;
    uint32_t passes = 1;
    uint64_t usefulBytes = 0;
    uint64_t estimatedTrafficBytes = 0;
    TimingStats timing;
};

double gbps(uint64_t bytes, double microseconds) {
    if (microseconds <= 0.0) return 0.0;
    return (static_cast<double>(bytes) / BYTES_PER_GB) / (microseconds * 1.0e-6);
}

void printHeader() {
    std::cout
        << "kernel,variant,row_layout,gather_pattern,collision_rate,batch_rows,nominal_source_values_per_row,"
           "gather_values_per_row,source_row_min,source_row_max,index_row_min,index_row_max,source_active_values,"
           "gathered_active_values,active_values,active_scalars,value_bytes,elements_per_value,num_classes,"
           "offset_bytes,index_bytes,prediction_dtype,label_dtype,label_format,selection_mode,blocks,"
           "auto_copy_width,copy_width,auto_lanes_per_token,lanes_per_token,auto_tokens_per_block,tokens_per_block,"
           "copy_items_per_value,auto_partial_blocks,partial_blocks,passes,useful_bytes,estimated_traffic_bytes,"
           "p10_us,median_us,p90_us,useful_GBps,estimated_traffic_GBps\n";
}

void printResult(const Result& r) {
    std::cout << r.kernel << ',' << r.variant << ',' << r.rowLayout << ',' << r.gatherPattern << ','
              << std::fixed << std::setprecision(4) << r.collisionRate << ',' << r.batchRows << ','
              << r.nominalSourceValuesPerRow << ',' << r.gatherValuesPerRow << ',' << r.sourceRowMin << ','
              << r.sourceRowMax << ',' << r.indexRowMin << ',' << r.indexRowMax << ',' << r.sourceActiveValues << ','
              << r.gatheredActiveValues << ',' << r.activeValues << ',' << r.activeScalars << ',' << r.valueBytes << ','
              << r.elementsPerValue << ',' << r.numClasses << ',' << r.offsetBytes << ',' << r.indexBytes << ','
              << r.predictionDtype << ',' << r.labelDtype << ',' << r.labelFormat << ',' << r.selectionMode << ','
              << r.blocks << ',' << r.autoCopyWidth << ',' << r.copyWidth << ',' << r.autoLanesPerToken << ','
              << r.lanesPerToken << ',' << r.autoTokensPerBlock << ',' << r.tokensPerBlock << ','
              << r.copyItemsPerValue << ',' << r.autoPartialBlocks << ',' << r.partialBlocks << ',' << r.passes << ','
              << r.usefulBytes << ','
              << r.estimatedTrafficBytes << ',' << std::setprecision(3) << r.timing.p10_us << ','
              << r.timing.median_us << ',' << r.timing.p90_us << ',' << std::setprecision(6)
              << gbps(r.usefulBytes, r.timing.median_us) << ','
              << gbps(r.estimatedTrafficBytes, r.timing.median_us) << '\n';
}

struct GatherGeometry {
    std::vector<uint64_t> sourceOffsets;
    std::vector<uint64_t> indexOffsets;
    std::vector<uint64_t> indices;
    uint64_t sourceActiveValues = 0;
    uint64_t gatheredActiveValues = 0;
    uint64_t sourceRowMin = std::numeric_limits<uint64_t>::max();
    uint64_t sourceRowMax = 0;
    uint64_t indexRowMin = std::numeric_limits<uint64_t>::max();
    uint64_t indexRowMax = 0;
    double actualCollisionRate = 0.0;
};

std::vector<uint64_t> makeSourceRowLengths(uint64_t batchRows,
                                           uint64_t nominalSourceValuesPerRow,
                                           uint64_t gatherValuesPerRow,
                                           std::string_view layout) {
    if (nominalSourceValuesPerRow < gatherValuesPerRow)
        throw std::invalid_argument("source-values-per-row must be >= gather-values-per-row");
    std::vector<uint64_t> lengths(batchRows, nominalSourceValuesPerRow);
    if (layout == "uniform") return lengths;
    if (layout == "empty") {
        for (uint64_t row = 0; row < batchRows; ++row)
            if ((row & 7ULL) == 0ULL) lengths[row] = 0;
        return lengths;
    }

    const uint64_t total = checkedMultiply(batchRows, nominalSourceValuesPerRow, "Gather source value count overflow");
    lengths.assign(batchRows, gatherValuesPerRow);
    const uint64_t baseTotal = checkedMultiply(batchRows, gatherValuesPerRow, "Gather source base count overflow");
    uint64_t remaining = total - baseTotal;
    std::vector<uint64_t> heavyRows;
    for (uint64_t row = 0; row < batchRows; ++row)
        if ((row & 7ULL) == 0ULL) heavyRows.push_back(row);
    if (heavyRows.empty()) heavyRows.push_back(0);
    for (size_t i = 0; i < heavyRows.size(); ++i) {
        const uint64_t share = remaining / (heavyRows.size() - i);
        lengths[heavyRows[i]] = checkedAdd(lengths[heavyRows[i]], share, "Gather skewed row overflow");
        remaining -= share;
    }
    return lengths;
}

uint64_t coprimeStep(uint64_t length, uint64_t seed) {
    if (length <= 1) return 1;
    uint64_t step = 1 + splitMix64(seed) % (length - 1);
    while (std::gcd(step, length) != 1) {
        ++step;
        if (step >= length) step = 1;
    }
    return step;
}

GatherGeometry makeGatherGeometry(uint64_t batchRows,
                                  uint64_t sourceValuesPerRow,
                                  uint64_t gatherValuesPerRow,
                                  std::string_view layout,
                                  std::string_view pattern,
                                  double requestedCollisionRate) {
    GatherGeometry geometry;
    geometry.sourceOffsets.reserve(batchRows + 1);
    geometry.indexOffsets.reserve(batchRows + 1);
    geometry.sourceOffsets.push_back(0);
    geometry.indexOffsets.push_back(0);

    const std::vector<uint64_t> rowLengths =
        makeSourceRowLengths(batchRows, sourceValuesPerRow, gatherValuesPerRow, layout);
    uint64_t totalDistinct = 0;
    for (uint64_t row = 0; row < batchRows; ++row) {
        const uint64_t sourceLength = rowLengths[row];
        geometry.sourceRowMin = std::min(geometry.sourceRowMin, sourceLength);
        geometry.sourceRowMax = std::max(geometry.sourceRowMax, sourceLength);
        geometry.sourceActiveValues = checkedAdd(geometry.sourceActiveValues, sourceLength, "Gather source offsets overflow");
        geometry.sourceOffsets.push_back(geometry.sourceActiveValues);
        const uint64_t q = sourceLength == 0 ? 0 : gatherValuesPerRow;
        geometry.indexRowMin = std::min(geometry.indexRowMin, q);
        geometry.indexRowMax = std::max(geometry.indexRowMax, q);
        if (q > sourceLength && pattern != "repeated")
            throw std::invalid_argument("unique/clustered gather requires gather-values-per-row <= every non-empty source row");

        uint64_t distinct = q;
        if (pattern == "repeated" && q != 0) {
            distinct = std::max<uint64_t>(1, static_cast<uint64_t>(std::ceil(q * (1.0 - requestedCollisionRate))));
            distinct = std::min(distinct, sourceLength);
        }
        totalDistinct += distinct;
        if (q != 0) {
            const uint64_t start = splitMix64(0x1234ULL + row) % sourceLength;
            if (pattern == "clustered") {
                for (uint64_t j = 0; j < q; ++j) geometry.indices.push_back((start + j) % sourceLength);
            } else {
                const uint64_t step = coprimeStep(sourceLength, 0x9abcULL + row);
                std::vector<uint64_t> targets;
                targets.reserve(distinct);
                for (uint64_t j = 0; j < distinct; ++j) targets.push_back((start + j * step) % sourceLength);
                for (uint64_t j = 0; j < q; ++j) geometry.indices.push_back(targets[j % distinct]);
            }
        }
        geometry.gatheredActiveValues = checkedAdd(geometry.gatheredActiveValues, q, "Gather index offsets overflow");
        geometry.indexOffsets.push_back(geometry.gatheredActiveValues);
    }
    if (geometry.gatheredActiveValues != 0)
        geometry.actualCollisionRate = 1.0 - static_cast<double>(totalDistinct) /
                                                static_cast<double>(geometry.gatheredActiveValues);
    return geometry;
}

template <typename T>
std::vector<T> checkedCastVector(const std::vector<uint64_t>& values, const char* what) {
    std::vector<T> result;
    result.reserve(values.size());
    for (uint64_t value : values) {
        if (value > static_cast<uint64_t>(std::numeric_limits<T>::max())) throw std::overflow_error(what);
        result.push_back(static_cast<T>(value));
    }
    return result;
}

Tensor makeOffsetTensor(int gpu, const std::vector<uint64_t>& values, uint32_t bytes, Stream& stream) {
    if (bytes == 4) return makeGpuTensor<uint32_t>(gpu, {values.size()}, checkedCastVector<uint32_t>(values, "Offset exceeds uint32"), stream);
    return makeGpuTensor<uint64_t>(gpu, {values.size()}, values, stream);
}

Tensor makeIndexTensor(int gpu,
                       const std::vector<uint64_t>& values,
                       uint64_t capacity,
                       uint32_t bytes,
                       Stream& stream) {
    if (bytes == 4) {
        std::vector<uint32_t> host(capacity, std::numeric_limits<uint32_t>::max());
        const auto active = checkedCastVector<uint32_t>(values, "Gather index exceeds uint32");
        std::copy(active.begin(), active.end(), host.begin());
        return makeGpuTensor<uint32_t>(gpu, {capacity}, host, stream);
    }
    std::vector<uint64_t> host(capacity, std::numeric_limits<uint64_t>::max());
    std::copy(values.begin(), values.end(), host.begin());
    return makeGpuTensor<uint64_t>(gpu, {capacity}, host, stream);
}

uint64_t partialCapacityRequirement(uint32_t partialBlocks) {
    if (partialBlocks <= 1) return 1;
    return checkedAdd(checkedMultiply(partialBlocks - 1ULL, REDUCTION_ITEMS_PER_PARTIAL,
                                      "Partial capacity requirement overflow"),
                      1, "Partial capacity requirement overflow");
}

uint64_t categoricalCapacityForPartials(uint64_t activeValues,
                                        uint64_t numClasses,
                                        uint32_t desiredPartials) {
    if (desiredPartials == 0) return activeValues;
    uint64_t capacity = activeValues;
    while (raggedAccuracyStatisticsWorkspaceDescriptor(capacity, numClasses).getDimensions().front() < desiredPartials) {
        if (capacity > std::numeric_limits<uint64_t>::max() / 2)
            throw std::overflow_error("Categorical benchmark capacity search overflow");
        capacity *= 2;
    }
    return capacity;
}

bool overPayloadLimit(uint64_t bytes, const Options& options, uint64_t& skipped) {
    if (bytes <= options.max_payload_bytes) return false;
    ++skipped;
    return true;
}

void verifyGather(int gpu, Stream& stream) {
    constexpr uint64_t batch = 4;
    constexpr uint64_t sourceCapacity = 10;
    constexpr uint64_t outputCapacity = 8;
    constexpr uint64_t valueBytes = 5;
    // Row 1 is deliberately empty so the representative check also covers
    // empty-row partition semantics.
    const std::vector<uint64_t> sourceOffsets64{0, 4, 4, 6, 9};
    const std::vector<uint64_t> indexOffsets64{0, 3, 3, 5, 7};
    const std::vector<uint64_t> indices64{3, 1, 3, 1, 0, 2, 2};
    Tensor sourceOffsets = makeOffsetTensor(gpu, sourceOffsets64, 4, stream);
    Tensor indexOffsets = makeOffsetTensor(gpu, indexOffsets64, 8, stream);
    Tensor indices = makeIndexTensor(gpu, indices64, outputCapacity, 4, stream);

    std::vector<uint8_t> sourceHost(sourceCapacity * valueBytes, 0xee);
    for (uint64_t token = 0; token < 9; ++token)
        for (uint64_t b = 0; b < valueBytes; ++b)
            sourceHost[token * valueBytes + b] = static_cast<uint8_t>((17 * token + b) & 0xffU);
    Tensor source = makeGpuTensor<uint8_t>(gpu, {sourceCapacity, valueBytes}, sourceHost, stream);
    Tensor output = makeGpuTensor<uint8_t>(gpu, {outputCapacity, valueBytes},
                                           std::vector<uint8_t>(outputCapacity * valueBytes, 0xcd), stream);
    launchRaggedGather(source, sourceOffsets, indices, indexOffsets, output, batch, stream);
    stream.synchronize();
    const std::vector<uint8_t> forward = copyGpuTensor<uint8_t>(output, stream);
    const std::vector<uint64_t> expectedTokens{3, 1, 3, 5, 4, 8, 8};
    for (uint64_t out = 0; out < expectedTokens.size(); ++out)
        for (uint64_t b = 0; b < valueBytes; ++b)
            if (forward[out * valueBytes + b] != sourceHost[expectedTokens[out] * valueBytes + b])
                throw std::runtime_error("RaggedGather forward reference check failed");
    for (uint64_t b = 0; b < valueBytes; ++b)
        if (forward[7 * valueBytes + b] != 0xcd) throw std::runtime_error("RaggedGather forward touched inactive capacity");

    output = makeGpuTensor<uint8_t>(gpu, {outputCapacity, valueBytes},
                                    std::vector<uint8_t>(outputCapacity * valueBytes, 0xcd), stream);
    launchRaggedGatherForBenchmark(source, sourceOffsets, indices, indexOffsets, output, batch, 1, 1, stream);
    stream.synchronize();
    const std::vector<uint8_t> forcedForward = copyGpuTensor<uint8_t>(output, stream);
    if (forcedForward != forward) throw std::runtime_error("RaggedGather forced forward semantics differ from auto");

    constexpr uint64_t elements = 3;
    std::vector<float> upstreamHost(outputCapacity * elements, 99.0f);
    for (uint64_t token = 0; token < 7; ++token)
        for (uint64_t e = 0; e < elements; ++e)
            upstreamHost[token * elements + e] = static_cast<float>(1 + token + e);
    Tensor upstream = makeGpuTensor<float>(gpu, {outputCapacity, elements}, upstreamHost, stream);
    Tensor gradient = makeGpuTensor<float>(gpu, {sourceCapacity, elements},
                                           std::vector<float>(sourceCapacity * elements, -77.0f), stream);
    launchRaggedGatherBackward(sourceOffsets, indices, indexOffsets, upstream, gradient, batch, stream);
    stream.synchronize();
    const std::vector<float> backward = copyGpuTensor<float>(gradient, stream);
    for (uint64_t token = 0; token < 9; ++token) {
        for (uint64_t e = 0; e < elements; ++e) {
            float expected = 0.0f;
            for (uint64_t out = 0; out < expectedTokens.size(); ++out)
                if (expectedTokens[out] == token) expected += upstreamHost[out * elements + e];
            if (backward[token * elements + e] != expected)
                throw std::runtime_error("RaggedGather backward reference check failed");
        }
    }
    for (uint64_t e = 0; e < elements; ++e)
        if (backward[9 * elements + e] != -77.0f) throw std::runtime_error("RaggedGather backward touched inactive capacity");

    gradient = makeGpuTensor<float>(gpu, {sourceCapacity, elements},
                                    std::vector<float>(sourceCapacity * elements, -77.0f), stream);
    launchRaggedGatherBackwardForBenchmark(sourceOffsets, indices, indexOffsets, upstream, gradient, batch, 1, stream);
    stream.synchronize();
    const std::vector<float> forcedBackward = copyGpuTensor<float>(gradient, stream);
    if (forcedBackward != backward) throw std::runtime_error("RaggedGather forced backward semantics differ from auto");
}

float copyScalar(const Tensor& tensor, Stream& stream) {
    return copyGpuTensor<float>(tensor, stream).front();
}

void verifyReductions(int gpu, Stream& stream) {
    const TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, gpu);

    constexpr uint64_t activeValues = 100;
    constexpr uint64_t elements = 3;
    constexpr uint64_t capacity = 4096;
    const uint64_t activeScalars = activeValues * elements;
    std::vector<float> valueHost(capacity * elements, std::numeric_limits<float>::quiet_NaN());
    std::vector<float> weightHost(capacity * elements, std::numeric_limits<float>::quiet_NaN());
    std::fill(valueHost.begin(), valueHost.begin() + activeScalars, 2.0f);
    std::fill(weightHost.begin(), weightHost.begin() + activeScalars, 0.5f);
    Tensor values = makeGpuTensor<float>(gpu, {capacity, elements}, valueHost, stream);
    Tensor weights = makeGpuTensor<float>(gpu, {capacity, elements}, weightHost, stream);
    Tensor partials(gpuPlacement, raggedWeightedMeanStatisticsWorkspaceDescriptor(capacity, elements));
    Tensor numerator(gpuPlacement, TensorDescriptor(DataType::FP32, {1}));
    Tensor denominator(gpuPlacement, TensorDescriptor(DataType::FP32, {1}));
    raggedWeightedMeanStatisticsWithPartialCountForBenchmark(
        values, weights, partials, numerator, denominator, activeValues, capacity, elements, 2, stream);
    stream.synchronize();
    const float expectedDenominator = static_cast<float>(activeValues * elements) * 0.5f;
    const float expectedNumerator = static_cast<float>(activeValues * elements);
    if (std::fabs(copyScalar(numerator, stream) - expectedNumerator) > 1.0e-4f ||
        std::fabs(copyScalar(denominator, stream) - expectedDenominator) > 1.0e-4f)
        throw std::runtime_error("RaggedWeightedReduction forced-partial reference check failed");

    std::vector<float> predictionHost(capacity, 0.75f);
    std::vector<uint8_t> binaryLabelHost(capacity, 0);
    std::fill(binaryLabelHost.begin(), binaryLabelHost.begin() + activeValues, uint8_t{1});
    Tensor predictions = makeGpuTensor<float>(gpu, {capacity, 1}, predictionHost, stream);
    Tensor binaryLabels = makeGpuTensor<uint8_t>(gpu, {capacity, 1}, binaryLabelHost, stream);
    Tensor accuracyPartials(gpuPlacement, raggedAccuracyStatisticsWorkspaceDescriptor(capacity, 1));
    Tensor correct(gpuPlacement, TensorDescriptor(DataType::FP32, {1}));
    Tensor tokens(gpuPlacement, TensorDescriptor(DataType::FP32, {1}));
    raggedBinaryAccuracyStatisticsForBenchmark(
        predictions, binaryLabels, accuracyPartials, correct, tokens, activeValues, capacity, 2, stream);
    stream.synchronize();
    if (copyScalar(correct, stream) != static_cast<float>(activeValues) ||
        copyScalar(tokens, stream) != static_cast<float>(activeValues))
        throw std::runtime_error("RaggedBinaryAccuracy forced-partial reference check failed");

    constexpr uint64_t classes = 33;
    Tensor classPredictions(gpuPlacement, TensorDescriptor(DataType::FP32, {capacity, classes}));
    Tensor classLabels(gpuPlacement, TensorDescriptor(DataType::UINT32, {capacity, 1}));
    classPredictions.fill(0.0, stream);  // first-occurrence tie means class 0
    classLabels.fill(0.0, stream);
    Tensor classPartials(gpuPlacement, raggedAccuracyStatisticsWorkspaceDescriptor(capacity, classes));
    raggedCategoricalAccuracyStatisticsForBenchmark(classPredictions,
                                                    classLabels,
                                                    classPartials,
                                                    correct,
                                                    tokens,
                                                    activeValues,
                                                    capacity,
                                                    classes,
                                                    RaggedCategoricalLabelFormat::CLASS_INDEX,
                                                    2,
                                                    16,
                                                    stream);
    stream.synchronize();
    if (copyScalar(correct, stream) != static_cast<float>(activeValues) ||
        copyScalar(tokens, stream) != static_cast<float>(activeValues))
        throw std::runtime_error("RaggedCategoricalAccuracy forced geometry reference check failed");
}

void runCorrectnessChecks(const Options& options, Stream& stream) {
    if (!options.verify) return;
    verifyGather(options.gpu, stream);
    verifyReductions(options.gpu, stream);
    std::cerr << "Representative correctness checks: PASS\n";
}

void runGatherForwardCase(const Options& options,
                          Stream& stream,
                          DramReadEvictor& evictor,
                          uint64_t batchRows,
                          uint64_t sourceValuesPerRow,
                          uint64_t gatherValuesPerRow,
                          uint64_t valueBytes,
                          std::string_view rowLayout,
                          std::string_view pattern,
                          double collisionRate,
                          uint32_t offsetBytes,
                          uint32_t indexBytes,
                          uint32_t forcedCopyWidth,
                          uint32_t forcedLanes,
                          uint64_t& skipped) {
    const GatherGeometry geometry = makeGatherGeometry(
        batchRows, sourceValuesPerRow, gatherValuesPerRow, rowLayout, pattern, collisionRate);
    constexpr uint64_t inactiveCapacityValues = 7;
    const uint64_t sourceCapacity = checkedAdd(geometry.sourceActiveValues, inactiveCapacityValues, "Gather source capacity overflow");
    const uint64_t outputCapacity = checkedAdd(geometry.gatheredActiveValues, inactiveCapacityValues, "Gather output capacity overflow");
    const uint64_t primaryBytes = checkedAdd(
        checkedMultiply(sourceCapacity, valueBytes, "Gather source bytes overflow"),
        checkedMultiply(outputCapacity, valueBytes, "Gather output bytes overflow"),
        "Gather primary bytes overflow");
    if (overPayloadLimit(primaryBytes, options, skipped)) return;

    const TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, options.gpu);
    Tensor source(gpuPlacement, TensorDescriptor(DataType::UINT8, {sourceCapacity, valueBytes}));
    Tensor output(gpuPlacement, TensorDescriptor(DataType::UINT8, {outputCapacity, valueBytes}));
    source.fill(17.0, stream);
    output.fill(205.0, stream);
    Tensor sourceOffsets = makeOffsetTensor(options.gpu, geometry.sourceOffsets, offsetBytes, stream);
    Tensor indexOffsets = makeOffsetTensor(options.gpu, geometry.indexOffsets, offsetBytes, stream);
    Tensor indices = makeIndexTensor(options.gpu, geometry.indices, outputCapacity, indexBytes, stream);
    stream.synchronize();

    const auto autoInfo = raggedGatherForwardLaunchInfoForBenchmark(source, output, batchRows, 0, 0);
    RaggedGatherForwardBenchmarkLaunchInfo selectedInfo{};
    try {
        selectedInfo = raggedGatherForwardLaunchInfoForBenchmark(
            source, output, batchRows, forcedCopyWidth, forcedLanes);
    } catch (const std::invalid_argument&) {
        ++skipped;
        return;
    }
    auto operation = [&] {
        if (forcedCopyWidth == 0 && forcedLanes == 0)
            launchRaggedGather(source, sourceOffsets, indices, indexOffsets, output, batchRows, stream);
        else
            launchRaggedGatherForBenchmark(source, sourceOffsets, indices, indexOffsets, output, batchRows,
                                           forcedCopyWidth, forcedLanes, stream);
    };

    Result result;
    result.kernel = "gather-forward";
    result.variant = pattern == "repeated" ? "repeated" : std::string(pattern);
    result.rowLayout = std::string(rowLayout);
    result.gatherPattern = std::string(pattern);
    result.collisionRate = geometry.actualCollisionRate;
    result.batchRows = batchRows;
    result.nominalSourceValuesPerRow = sourceValuesPerRow;
    result.gatherValuesPerRow = gatherValuesPerRow;
    result.sourceRowMin = geometry.sourceRowMin;
    result.sourceRowMax = geometry.sourceRowMax;
    result.indexRowMin = geometry.indexRowMin;
    result.indexRowMax = geometry.indexRowMax;
    result.sourceActiveValues = geometry.sourceActiveValues;
    result.gatheredActiveValues = geometry.gatheredActiveValues;
    result.valueBytes = valueBytes;
    result.offsetBytes = offsetBytes;
    result.indexBytes = indexBytes;
    result.predictionDtype = "u8";
    result.selectionMode = forcedCopyWidth == 0 && forcedLanes == 0 ? "auto" : "forced";
    result.blocks = selectedInfo.blocks;
    result.autoCopyWidth = autoInfo.copy_width_bytes;
    result.copyWidth = selectedInfo.copy_width_bytes;
    result.autoLanesPerToken = autoInfo.lanes_per_token;
    result.lanesPerToken = selectedInfo.lanes_per_token;
    result.autoTokensPerBlock = autoInfo.tokens_per_block;
    result.tokensPerBlock = selectedInfo.tokens_per_block;
    result.copyItemsPerValue = selectedInfo.copy_items_per_value;
    result.usefulBytes = checkedMultiply(geometry.gatheredActiveValues, valueBytes, "Gather useful bytes overflow");
    result.estimatedTrafficBytes = checkedMultiply(result.usefulBytes, 2, "Gather traffic bytes overflow");
    result.timing = benchmarkDramReads(stream, evictor, options.warmup, options.iterations, operation);
    printResult(result);
}

void runGatherBackwardCase(const Options& options,
                           Stream& stream,
                           DramReadEvictor& evictor,
                           uint64_t batchRows,
                           uint64_t sourceValuesPerRow,
                           uint64_t gatherValuesPerRow,
                           uint64_t valueBytes,
                           std::string_view rowLayout,
                           std::string_view pattern,
                           double collisionRate,
                           uint32_t offsetBytes,
                           uint32_t indexBytes,
                           std::string_view dtypeName,
                           uint32_t forcedLanes,
                           uint64_t& skipped) {
    const DataType dtype = floatingDtype(dtypeName);
    const uint64_t elementBytes = dtypeBytes(dtype);
    if (valueBytes % elementBytes != 0) {
        ++skipped;
        return;
    }
    const uint64_t elements = valueBytes / elementBytes;
    if (elements == 0) {
        ++skipped;
        return;
    }
    const GatherGeometry geometry = makeGatherGeometry(
        batchRows, sourceValuesPerRow, gatherValuesPerRow, rowLayout, pattern, collisionRate);
    constexpr uint64_t inactiveCapacityValues = 7;
    const uint64_t sourceCapacity = checkedAdd(geometry.sourceActiveValues, inactiveCapacityValues, "Gather source capacity overflow");
    const uint64_t outputCapacity = checkedAdd(geometry.gatheredActiveValues, inactiveCapacityValues, "Gather output capacity overflow");
    const uint64_t primaryBytes = checkedAdd(
        checkedMultiply(sourceCapacity, valueBytes, "Gather gradient bytes overflow"),
        checkedMultiply(outputCapacity, valueBytes, "Gather upstream bytes overflow"),
        "Gather backward primary bytes overflow");
    if (overPayloadLimit(primaryBytes, options, skipped)) return;

    const TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, options.gpu);
    Tensor sourceGradient(gpuPlacement, TensorDescriptor(dtype, {sourceCapacity, elements}));
    Tensor outputGradient(gpuPlacement, TensorDescriptor(dtype, {outputCapacity, elements}));
    sourceGradient.fill(7.0, stream);
    outputGradient.fill(1.0, stream);
    Tensor sourceOffsets = makeOffsetTensor(options.gpu, geometry.sourceOffsets, offsetBytes, stream);
    Tensor indexOffsets = makeOffsetTensor(options.gpu, geometry.indexOffsets, offsetBytes, stream);
    Tensor indices = makeIndexTensor(options.gpu, geometry.indices, outputCapacity, indexBytes, stream);
    stream.synchronize();

    const auto autoInfo = raggedGatherBackwardLaunchInfoForBenchmark(sourceGradient, batchRows, 0);
    const auto selectedInfo = raggedGatherBackwardLaunchInfoForBenchmark(sourceGradient, batchRows, forcedLanes);
    auto operation = [&] {
        if (forcedLanes == 0)
            launchRaggedGatherBackward(sourceOffsets, indices, indexOffsets, outputGradient, sourceGradient, batchRows, stream);
        else
            launchRaggedGatherBackwardForBenchmark(sourceOffsets, indices, indexOffsets, outputGradient, sourceGradient,
                                                   batchRows, forcedLanes, stream);
    };

    Result result;
    result.kernel = "gather-backward";
    result.variant = geometry.actualCollisionRate == 0.0 ? std::string(pattern) : "scatter-collision";
    result.rowLayout = std::string(rowLayout);
    result.gatherPattern = std::string(pattern);
    result.collisionRate = geometry.actualCollisionRate;
    result.batchRows = batchRows;
    result.nominalSourceValuesPerRow = sourceValuesPerRow;
    result.gatherValuesPerRow = gatherValuesPerRow;
    result.sourceRowMin = geometry.sourceRowMin;
    result.sourceRowMax = geometry.sourceRowMax;
    result.indexRowMin = geometry.indexRowMin;
    result.indexRowMax = geometry.indexRowMax;
    result.sourceActiveValues = geometry.sourceActiveValues;
    result.gatheredActiveValues = geometry.gatheredActiveValues;
    result.valueBytes = valueBytes;
    result.elementsPerValue = elements;
    result.offsetBytes = offsetBytes;
    result.indexBytes = indexBytes;
    result.predictionDtype = std::string(dtypeName);
    result.selectionMode = forcedLanes == 0 ? "auto" : "forced";
    result.blocks = selectedInfo.blocks;
    result.autoLanesPerToken = autoInfo.lanes_per_token;
    result.lanesPerToken = selectedInfo.lanes_per_token;
    result.autoTokensPerBlock = autoInfo.tokens_per_block;
    result.tokensPerBlock = selectedInfo.tokens_per_block;
    result.copyItemsPerValue = selectedInfo.elements_per_value;
    result.usefulBytes = checkedMultiply(geometry.gatheredActiveValues, valueBytes, "Gather backward useful bytes overflow");
    const uint64_t zeroWrites = checkedMultiply(geometry.sourceActiveValues, valueBytes, "Gather zero bytes overflow");
    const uint64_t scatterFactor = gatherValuesPerRow == 1 ? 1 : 2;
    const uint64_t scatterTraffic = checkedMultiply(result.usefulBytes, scatterFactor, "Gather scatter traffic overflow");
    result.estimatedTrafficBytes = checkedAdd(checkedAdd(result.usefulBytes, zeroWrites, "Gather traffic overflow"),
                                              scatterTraffic, "Gather traffic overflow");
    result.timing = benchmarkDramReads(stream, evictor, options.warmup, options.iterations, operation);
    printResult(result);
}

void runGatherBenchmarks(const Options& options, Stream& stream, DramReadEvictor& evictor, uint64_t& skipped) {
    const bool forward = contains(options.kernels, "gather-forward");
    const bool backward = contains(options.kernels, "gather-backward");
    if (!forward && !backward) return;
    for (uint64_t batchRows : options.batch_rows) {
        for (uint64_t sourceValuesPerRow : options.source_values_per_row) {
            for (uint64_t gatherValuesPerRow : options.gather_values_per_row) {
                if (sourceValuesPerRow < gatherValuesPerRow) {
                    ++skipped;
                    continue;
                }
                for (uint64_t valueBytes : options.gather_value_bytes) {
                    for (const std::string& rowLayout : options.row_layouts) {
                        for (uint32_t offsetBytes : options.offset_bytes) {
                            for (uint32_t indexBytes : options.index_bytes) {
                                for (const std::string& pattern : options.gather_patterns) {
                                    std::vector<double> rates{0.0};
                                    if (pattern == "repeated" && gatherValuesPerRow > 1)
                                        rates = options.collision_rates;
                                    for (double rate : rates) {
                                        if (forward) {
                                            for (uint32_t copyWidth : options.gather_copy_bytes)
                                                for (uint32_t lanes : options.gather_lanes)
                                                    runGatherForwardCase(options, stream, evictor, batchRows,
                                                                         sourceValuesPerRow, gatherValuesPerRow,
                                                                         valueBytes, rowLayout, pattern, rate,
                                                                         offsetBytes, indexBytes, copyWidth, lanes, skipped);
                                        }
                                        if (backward) {
                                            for (const std::string& dtype : options.gather_backward_dtypes)
                                                for (uint32_t lanes : options.gather_lanes)
                                                    runGatherBackwardCase(options, stream, evictor, batchRows,
                                                                          sourceValuesPerRow, gatherValuesPerRow,
                                                                          valueBytes, rowLayout, pattern, rate,
                                                                          offsetBytes, indexBytes, dtype, lanes, skipped);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void runWeightedCase(const Options& options,
                     Stream& stream,
                     DramReadEvictor& evictor,
                     uint64_t activeValues,
                     uint64_t elementsPerValue,
                     std::string_view dtypeName,
                     uint32_t forcedPartials,
                     uint64_t& skipped) {
    const DataType dtype = floatingDtype(dtypeName);
    const uint64_t activeScalars = checkedMultiply(activeValues, elementsPerValue, "Weighted active scalar overflow");
    const uint32_t autoPartials = raggedWeightedMeanPartialBlockCountForBenchmark(activeScalars);
    const uint32_t selectedPartials = forcedPartials == 0 ? autoPartials : forcedPartials;
    const uint64_t requiredScalars = std::max(activeScalars, partialCapacityRequirement(selectedPartials));
    const uint64_t maxTotalValues = std::max(activeValues, ceilDiv(requiredScalars, elementsPerValue));
    const uint64_t capacityScalars = checkedMultiply(maxTotalValues, elementsPerValue, "Weighted capacity scalar overflow");
    const uint64_t inputBytes = checkedMultiply(capacityScalars, 2 * dtypeBytes(dtype), "Weighted allocation bytes overflow");
    if (overPayloadLimit(inputBytes, options, skipped)) return;

    const TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, options.gpu);
    Tensor values(gpuPlacement, TensorDescriptor(dtype, {maxTotalValues, elementsPerValue}));
    Tensor weights(gpuPlacement, TensorDescriptor(dtype, {maxTotalValues, elementsPerValue}));
    values.fill(2.0, stream);
    weights.fill(0.5, stream);
    Tensor partials(gpuPlacement, raggedWeightedMeanStatisticsWorkspaceDescriptor(maxTotalValues, elementsPerValue));
    Tensor numerator(gpuPlacement, TensorDescriptor(DataType::FP32, {1}));
    Tensor denominator(gpuPlacement, TensorDescriptor(DataType::FP32, {1}));
    stream.synchronize();

    if (selectedPartials > partials.getDimensions().front()) {
        ++skipped;
        return;
    }
    auto operation = [&] {
        if (forcedPartials == 0)
            raggedWeightedMeanStatistics(values, weights, partials, numerator, denominator,
                                         activeValues, maxTotalValues, elementsPerValue, stream);
        else
            raggedWeightedMeanStatisticsWithPartialCountForBenchmark(values, weights, partials, numerator, denominator,
                                                                     activeValues, maxTotalValues, elementsPerValue,
                                                                     forcedPartials, stream);
    };

    Result result;
    result.kernel = "weighted-reduction";
    result.variant = selectedPartials == 1 ? "one-pass" : "multi-pass";
    result.rowLayout = "packed-prefix";
    result.activeValues = activeValues;
    result.activeScalars = activeScalars;
    result.valueBytes = checkedMultiply(elementsPerValue, dtypeBytes(dtype), "Weighted value bytes overflow");
    result.elementsPerValue = elementsPerValue;
    result.predictionDtype = std::string(dtypeName);
    result.labelDtype = std::string(dtypeName);
    result.selectionMode = forcedPartials == 0 ? "auto" : "forced";
    result.blocks = selectedPartials;
    result.autoPartialBlocks = autoPartials;
    result.partialBlocks = selectedPartials;
    result.passes = selectedPartials == 1 ? 1 : 2;
    result.usefulBytes = checkedMultiply(activeScalars, 2 * dtypeBytes(dtype), "Weighted useful bytes overflow");
    const uint64_t partialTraffic = selectedPartials > 1 ? checkedMultiply(selectedPartials, 16, "Weighted partial traffic overflow") : 0;
    result.estimatedTrafficBytes = checkedAdd(checkedAdd(result.usefulBytes, partialTraffic, "Weighted traffic overflow"), 8,
                                              "Weighted traffic overflow");
    result.timing = benchmarkDramReads(stream, evictor, options.warmup, options.iterations, operation);
    printResult(result);
}

void runWeightedBenchmarks(const Options& options, Stream& stream, DramReadEvictor& evictor, uint64_t& skipped) {
    if (!contains(options.kernels, "weighted-reduction")) return;
    for (uint64_t activeValues : options.active_values)
        for (uint64_t elements : options.reduction_elements)
            for (const std::string& dtype : options.reduction_dtypes)
                for (uint32_t partials : options.partial_blocks)
                    runWeightedCase(options, stream, evictor, activeValues, elements, dtype, partials, skipped);
}

void runBinaryCase(const Options& options,
                   Stream& stream,
                   DramReadEvictor& evictor,
                   uint64_t activeValues,
                   std::string_view predictionDtypeName,
                   std::string_view labelDtypeName,
                   uint32_t forcedPartials,
                   uint64_t& skipped) {
    const DataType predictionDtype = floatingDtype(predictionDtypeName);
    const DataType labelDtype = binaryLabelDtype(labelDtypeName);
    const uint32_t autoPartials = raggedBinaryAccuracyPartialBlockCountForBenchmark(activeValues);
    const uint32_t selectedPartials = forcedPartials == 0 ? autoPartials : forcedPartials;
    const uint64_t maxTotalValues = std::max(activeValues, partialCapacityRequirement(selectedPartials));
    const uint64_t primaryBytes = checkedMultiply(maxTotalValues,
                                                  dtypeBytes(predictionDtype) + dtypeBytes(labelDtype),
                                                  "Binary accuracy allocation bytes overflow");
    if (overPayloadLimit(primaryBytes, options, skipped)) return;

    const TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, options.gpu);
    Tensor predictions(gpuPlacement, TensorDescriptor(predictionDtype, {maxTotalValues, 1}));
    Tensor labels(gpuPlacement, TensorDescriptor(labelDtype, {maxTotalValues, 1}));
    predictions.fill(0.75, stream);
    labels.fill(1.0, stream);
    Tensor partials(gpuPlacement, raggedAccuracyStatisticsWorkspaceDescriptor(maxTotalValues, 1));
    Tensor correct(gpuPlacement, TensorDescriptor(DataType::FP32, {1}));
    Tensor tokens(gpuPlacement, TensorDescriptor(DataType::FP32, {1}));
    stream.synchronize();
    if (selectedPartials > partials.getDimensions().front()) {
        ++skipped;
        return;
    }

    auto operation = [&] {
        if (forcedPartials == 0)
            raggedBinaryAccuracyStatistics(predictions, labels, partials, correct, tokens,
                                           activeValues, maxTotalValues, stream);
        else
            raggedBinaryAccuracyStatisticsForBenchmark(predictions, labels, partials, correct, tokens,
                                                        activeValues, maxTotalValues, forcedPartials, stream);
    };

    Result result;
    result.kernel = "binary-accuracy";
    result.variant = selectedPartials == 1 ? "one-pass" : "multi-pass";
    result.rowLayout = "packed-prefix";
    result.activeValues = activeValues;
    result.activeScalars = activeValues;
    result.valueBytes = dtypeBytes(predictionDtype);
    result.elementsPerValue = 1;
    result.predictionDtype = std::string(predictionDtypeName);
    result.labelDtype = std::string(labelDtypeName);
    result.labelFormat = "scalar";
    result.selectionMode = forcedPartials == 0 ? "auto" : "forced";
    result.blocks = selectedPartials;
    result.autoPartialBlocks = autoPartials;
    result.partialBlocks = selectedPartials;
    result.passes = selectedPartials == 1 ? 1 : 2;
    result.usefulBytes = checkedMultiply(activeValues,
                                         dtypeBytes(predictionDtype) + dtypeBytes(labelDtype),
                                         "Binary accuracy useful bytes overflow");
    const uint64_t partialTraffic = selectedPartials > 1 ? checkedMultiply(selectedPartials, 16, "Binary partial traffic overflow") : 0;
    result.estimatedTrafficBytes = checkedAdd(checkedAdd(result.usefulBytes, partialTraffic, "Binary traffic overflow"), 8,
                                              "Binary traffic overflow");
    result.timing = benchmarkDramReads(stream, evictor, options.warmup, options.iterations, operation);
    printResult(result);
}

void runBinaryBenchmarks(const Options& options, Stream& stream, DramReadEvictor& evictor, uint64_t& skipped) {
    if (!contains(options.kernels, "binary-accuracy")) return;
    for (uint64_t activeValues : options.active_values)
        for (const std::string& predictionDtype : options.prediction_dtypes)
            for (const std::string& labelDtype : options.binary_label_dtypes)
                for (uint32_t partials : options.partial_blocks)
                    runBinaryCase(options, stream, evictor, activeValues, predictionDtype, labelDtype, partials, skipped);
}

void runCategoricalCase(const Options& options,
                        Stream& stream,
                        DramReadEvictor& evictor,
                        uint64_t activeValues,
                        uint64_t numClasses,
                        std::string_view predictionDtypeName,
                        std::string_view labelFormatName,
                        uint32_t forcedPartials,
                        uint32_t forcedLanes,
                        uint64_t& skipped) {
    const DataType predictionDtype = floatingDtype(predictionDtypeName);
    const uint32_t autoLanes = raggedCategoricalAccuracyLanesPerTokenForBenchmark(numClasses);
    const uint32_t selectedLanes = forcedLanes == 0 ? autoLanes : forcedLanes;
    const uint32_t autoPartialsForSelectedLanes =
        raggedCategoricalAccuracyPartialBlockCountForBenchmark(activeValues, numClasses, selectedLanes);
    const uint32_t productionAutoPartials =
        raggedCategoricalAccuracyPartialBlockCountForBenchmark(activeValues, numClasses, 0);
    const uint32_t selectedPartials = forcedPartials == 0 ? autoPartialsForSelectedLanes : forcedPartials;
    const uint64_t maxTotalValues = categoricalCapacityForPartials(activeValues, numClasses, selectedPartials);
    const bool perClass = labelFormatName == "per-class";
    const uint64_t predictionScalars = checkedMultiply(maxTotalValues, numClasses, "Categorical prediction scalar overflow");
    const uint64_t labelScalars = perClass ? predictionScalars : maxTotalValues;
    const uint64_t primaryBytes = checkedAdd(
        checkedMultiply(predictionScalars, dtypeBytes(predictionDtype), "Categorical prediction bytes overflow"),
        checkedMultiply(labelScalars, perClass ? 4ULL : 4ULL, "Categorical label bytes overflow"),
        "Categorical allocation bytes overflow");
    if (overPayloadLimit(primaryBytes, options, skipped)) return;

    const TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, options.gpu);
    Tensor predictions(gpuPlacement, TensorDescriptor(predictionDtype, {maxTotalValues, numClasses}));
    predictions.fill(0.0, stream);  // class 0 wins ties
    Tensor labels(gpuPlacement,
                  TensorDescriptor(perClass ? DataType::FP32 : DataType::UINT32,
                                   perClass ? std::vector<uint64_t>{maxTotalValues, numClasses}
                                            : std::vector<uint64_t>{maxTotalValues, 1}));
    labels.fill(0.0, stream);
    Tensor partials(gpuPlacement, raggedAccuracyStatisticsWorkspaceDescriptor(maxTotalValues, numClasses));
    Tensor correct(gpuPlacement, TensorDescriptor(DataType::FP32, {1}));
    Tensor tokens(gpuPlacement, TensorDescriptor(DataType::FP32, {1}));
    stream.synchronize();
    if (selectedPartials > partials.getDimensions().front()) {
        ++skipped;
        return;
    }
    const RaggedCategoricalLabelFormat labelFormat =
        perClass ? RaggedCategoricalLabelFormat::PER_CLASS : RaggedCategoricalLabelFormat::CLASS_INDEX;
    auto operation = [&] {
        if (forcedPartials == 0 && forcedLanes == 0)
            raggedCategoricalAccuracyStatistics(predictions, labels, partials, correct, tokens,
                                                activeValues, maxTotalValues, numClasses, labelFormat, stream);
        else
            raggedCategoricalAccuracyStatisticsForBenchmark(predictions, labels, partials, correct, tokens,
                                                             activeValues, maxTotalValues, numClasses, labelFormat,
                                                             forcedPartials, forcedLanes, stream);
    };

    Result result;
    result.kernel = "categorical-accuracy";
    result.variant = selectedPartials == 1 ? "one-pass" : "multi-pass";
    result.rowLayout = "packed-prefix";
    result.activeValues = activeValues;
    result.activeScalars = checkedMultiply(activeValues, numClasses, "Categorical active scalar overflow");
    result.numClasses = numClasses;
    result.valueBytes = checkedMultiply(numClasses, dtypeBytes(predictionDtype), "Categorical value bytes overflow");
    result.elementsPerValue = numClasses;
    result.predictionDtype = std::string(predictionDtypeName);
    result.labelDtype = perClass ? "fp32" : "u32";
    result.labelFormat = std::string(labelFormatName);
    result.selectionMode = forcedPartials == 0 && forcedLanes == 0 ? "auto" : "forced";
    result.blocks = selectedPartials;
    result.autoLanesPerToken = autoLanes;
    result.lanesPerToken = selectedLanes;
    result.autoTokensPerBlock = 256U / autoLanes;
    result.tokensPerBlock = 256U / selectedLanes;
    result.autoPartialBlocks = productionAutoPartials;
    result.partialBlocks = selectedPartials;
    result.passes = selectedPartials == 1 ? 1 : 2;
    const uint64_t activePredictionBytes = checkedMultiply(result.activeScalars, dtypeBytes(predictionDtype),
                                                           "Categorical useful prediction bytes overflow");
    const uint64_t activeLabelBytes = perClass
                                          ? checkedMultiply(result.activeScalars, 4, "Categorical per-class label bytes overflow")
                                          : checkedMultiply(activeValues, 4, "Categorical index label bytes overflow");
    result.usefulBytes = checkedAdd(activePredictionBytes, activeLabelBytes, "Categorical useful bytes overflow");
    const uint64_t partialTraffic = selectedPartials > 1 ? checkedMultiply(selectedPartials, 16, "Categorical partial traffic overflow") : 0;
    result.estimatedTrafficBytes = checkedAdd(checkedAdd(result.usefulBytes, partialTraffic, "Categorical traffic overflow"), 8,
                                              "Categorical traffic overflow");
    result.timing = benchmarkDramReads(stream, evictor, options.warmup, options.iterations, operation);
    printResult(result);
}

void runCategoricalBenchmarks(const Options& options, Stream& stream, DramReadEvictor& evictor, uint64_t& skipped) {
    if (!contains(options.kernels, "categorical-accuracy")) return;
    for (uint64_t activeValues : options.categorical_active_values)
        for (uint64_t classes : options.class_counts)
            for (const std::string& predictionDtype : options.prediction_dtypes)
                for (const std::string& labelFormat : options.categorical_label_formats)
                    for (uint32_t partials : options.partial_blocks)
                        for (uint32_t lanes : options.categorical_lanes)
                            runCategoricalCase(options, stream, evictor, activeValues, classes, predictionDtype,
                                               labelFormat, partials, lanes, skipped);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parseOptions(argc, argv);
        ScopedGpu scopedGpu(options.gpu);
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, options.gpu));
        if (properties.l2CacheSize <= 0)
            throw std::runtime_error("CUDA reported zero L2 bytes; cannot construct DRAM-backed benchmark");

        Stream stream(options.gpu);
        runCorrectnessChecks(options, stream);
        DramReadEvictor evictor(options, properties, stream);
        size_t freeBytes = 0;
        size_t totalBytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
        if (evictor.bytes() > static_cast<uint64_t>(freeBytes) / 2)
            throw std::runtime_error("L2 eviction buffer would consume more than half of free GPU memory");

        std::cerr << "Thor ragged primitive benchmark\n"
                  << "GPU: " << options.gpu << " (" << properties.name << ")\n"
                  << "L2: " << static_cast<double>(properties.l2CacheSize) / MIB << " MiB\n"
                  << "Untimed device-read eviction: " << static_cast<double>(evictor.bytes()) / MIB << " MiB ("
                  << static_cast<double>(evictor.bytes()) / properties.l2CacheSize << "x L2)\n"
                  << "Max primary payload per case: " << static_cast<double>(options.max_payload_bytes) / MIB << " MiB\n";

        printHeader();
        uint64_t skipped = 0;
        runGatherBenchmarks(options, stream, evictor, skipped);
        runWeightedBenchmarks(options, stream, evictor, skipped);
        runBinaryBenchmarks(options, stream, evictor, skipped);
        runCategoricalBenchmarks(options, stream, evictor, skipped);
        stream.synchronize();
        if (skipped != 0) std::cerr << "Skipped " << skipped << " incompatible/oversized geometries.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "thor_ragged_primitive_benchmark: " << error.what() << '\n';
        return 1;
    }
}
