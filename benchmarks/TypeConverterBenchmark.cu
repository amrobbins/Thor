#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "Utilities/Common/Event.h"
#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/Expression/CudaHelpers.h"
#include "Utilities/TensorOperations/DataTypeConversions/TypeConverter.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using ThorImplementation::DataType;
using ThorImplementation::TensorDescriptor;
using ThorImplementation::TypeConverter;

namespace {

constexpr std::size_t KiB = 1024;
constexpr std::size_t MiB = 1024 * KiB;
constexpr std::size_t GiB = 1024 * MiB;
constexpr std::size_t SLOT_ALIGNMENT_BYTES = 256;
constexpr int THREADS_PER_BLOCK = 256;

struct DataTypeInfo {
    DataType dataType;
    std::string_view name;
    std::size_t elementBytes;
};

constexpr std::array<DataTypeInfo, 15> STORAGE_TYPES = {{
    {DataType::FP8_E4M3, "fp8_e4m3", 1},
    {DataType::FP8_E5M2, "fp8_e5m2", 1},
    {DataType::FP16, "fp16", 2},
    {DataType::BF16, "bf16", 2},
    {DataType::FP32, "fp32", 4},
    {DataType::FP64, "fp64", 8},
    {DataType::INT8, "int8", 1},
    {DataType::INT16, "int16", 2},
    {DataType::INT32, "int32", 4},
    {DataType::INT64, "int64", 8},
    {DataType::UINT8, "uint8", 1},
    {DataType::UINT16, "uint16", 2},
    {DataType::UINT32, "uint32", 4},
    {DataType::UINT64, "uint64", 8},
    {DataType::BOOLEAN, "bool", 1},
}};

constexpr std::array<DataType, 6> ML_FLOAT_TYPES = {
    DataType::FP8_E4M3, DataType::FP8_E5M2, DataType::FP16, DataType::BF16, DataType::FP32, DataType::FP64};

struct TypePair {
    DataType source;
    DataType destination;
};

enum class BenchmarkMode { OUT_OF_PLACE, IN_PLACE };

struct Options {
    int gpu = 0;
    std::string suite = "common";
    std::vector<TypePair> explicitPairs;
    std::vector<std::uint64_t> elementCounts = {
        2ULL * 1024,
        8ULL * 1024,
        32ULL * 1024,
        64ULL * 1024,
        128ULL * 1024,
        256ULL * 1024,
        512ULL * 1024,
        2ULL * 1024 * 1024,
        8ULL * 1024 * 1024,
        32ULL * 1024 * 1024,
        128ULL * 1024 * 1024,
        256ULL * 1024 * 1024,
    };
    std::vector<BenchmarkMode> modes = {BenchmarkMode::OUT_OF_PLACE};
    int samples = 7;
    std::size_t targetTrafficBytes = 512 * MiB;
    std::size_t maxWorkingSetBytes = 4 * GiB;
    std::size_t safetyReserveBytes = 512 * MiB;
    std::uint64_t maxLaunchesPerSample = 4096;
    double l2EvictionFactor = 4.0;
    bool verify = true;
    std::optional<std::string> csvPath;
};

struct DeviceBuffer {
    DeviceBuffer() = default;

    explicit DeviceBuffer(std::size_t bytes) { allocate(bytes); }

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    DeviceBuffer(DeviceBuffer &&other) noexcept : pointer(other.pointer), bytes(other.bytes) {
        other.pointer = nullptr;
        other.bytes = 0;
    }

    DeviceBuffer &operator=(DeviceBuffer &&) = delete;

    ~DeviceBuffer() { release(); }

    void allocate(std::size_t requestedBytes) {
        release();
        if (requestedBytes == 0)
            return;
        CUDA_CHECK(cudaMalloc(&pointer, requestedBytes));
        bytes = requestedBytes;
    }

    void release() noexcept {
        if (pointer != nullptr) {
            // Destructors must not throw. Any earlier synchronization point will already
            // have surfaced asynchronous failures from the benchmarked work.
            (void)cudaFree(pointer);
            pointer = nullptr;
            bytes = 0;
        }
    }

    void *get() const { return pointer; }

    void *pointer = nullptr;
    std::size_t bytes = 0;
};

struct BenchmarkPlan {
    std::size_t sourceBytesPerLaunch;
    std::size_t destinationBytesPerLaunch;
    std::size_t sourceStrideBytes;
    std::size_t destinationStrideBytes;
    std::size_t inPlaceStrideBytes;
    std::uint64_t launchesPerSample;
    std::size_t workingSetBytes;
};

struct BenchmarkResult {
    TypePair pair;
    BenchmarkMode mode;
    std::uint64_t elements;
    BenchmarkPlan plan;
    double medianUsPerLaunch;
    double p10UsPerLaunch;
    double p90UsPerLaunch;
    double medianNsPerElement;
    double effectiveGbPerSecond;
    double inputGbPerSecond;
    double outputGbPerSecond;
};

std::size_t checkedAdd(std::size_t a, std::size_t b, std::string_view context) {
    if (b > std::numeric_limits<std::size_t>::max() - a)
        throw std::overflow_error(std::string(context) + " overflows size_t");
    return a + b;
}

std::size_t checkedMultiply(std::size_t a, std::size_t b, std::string_view context) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
        throw std::overflow_error(std::string(context) + " overflows size_t");
    return a * b;
}

std::size_t alignUp(std::size_t value, std::size_t alignment) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0)
        throw std::invalid_argument("alignment must be a nonzero power of two");
    if (value > std::numeric_limits<std::size_t>::max() - (alignment - 1))
        throw std::overflow_error("aligned size overflows size_t");
    return (value + alignment - 1) & ~(alignment - 1);
}

const DataTypeInfo &typeInfo(DataType dataType) {
    for (const DataTypeInfo &info : STORAGE_TYPES) {
        if (info.dataType == dataType)
            return info;
    }
    throw std::invalid_argument("unsupported storage data type");
}

DataType parseDataType(std::string_view name) {
    for (const DataTypeInfo &info : STORAGE_TYPES) {
        if (info.name == name)
            return info.dataType;
    }
    throw std::invalid_argument("unknown data type '" + std::string(name) + "'");
}

std::string_view modeName(BenchmarkMode mode) { return mode == BenchmarkMode::OUT_OF_PLACE ? "out_of_place" : "in_place"; }

std::uint64_t parseUnsigned(std::string_view text, std::string_view optionName) {
    if (text.empty())
        throw std::invalid_argument(std::string(optionName) + " requires a value");

    std::uint64_t multiplier = 1;
    const char suffix = text.back();
    if (suffix == 'k' || suffix == 'K') {
        multiplier = 1024;
        text.remove_suffix(1);
    } else if (suffix == 'm' || suffix == 'M') {
        multiplier = 1024 * 1024;
        text.remove_suffix(1);
    } else if (suffix == 'g' || suffix == 'G') {
        multiplier = 1024ULL * 1024ULL * 1024ULL;
        text.remove_suffix(1);
    }

    if (text.empty())
        throw std::invalid_argument(std::string(optionName) + " has an invalid numeric value");

    std::size_t consumed = 0;
    const unsigned long long value = std::stoull(std::string(text), &consumed, 10);
    if (consumed != text.size())
        throw std::invalid_argument(std::string(optionName) + " has an invalid numeric value");
    if (value > std::numeric_limits<std::uint64_t>::max() / multiplier)
        throw std::overflow_error(std::string(optionName) + " is too large");
    return static_cast<std::uint64_t>(value) * multiplier;
}

std::vector<std::uint64_t> parseElementCounts(std::string_view csv) {
    std::vector<std::uint64_t> values;
    while (!csv.empty()) {
        const std::size_t comma = csv.find(',');
        const std::string_view token = csv.substr(0, comma);
        const std::uint64_t value = parseUnsigned(token, "--sizes");
        if (value == 0)
            throw std::invalid_argument("--sizes entries must be greater than zero");
        values.push_back(value);
        if (comma == std::string_view::npos)
            break;
        csv.remove_prefix(comma + 1);
    }
    if (values.empty())
        throw std::invalid_argument("--sizes requires at least one element count");
    return values;
}

TypePair parsePair(std::string_view text) {
    const std::size_t colon = text.find(':');
    if (colon == std::string_view::npos || colon == 0 || colon + 1 == text.size())
        throw std::invalid_argument("--pair must have the form source:destination");
    const TypePair pair{parseDataType(text.substr(0, colon)), parseDataType(text.substr(colon + 1))};
    if (pair.source == pair.destination)
        throw std::invalid_argument("TypeConverter does not launch a conversion for identical source and destination types");
    return pair;
}

std::string optionValue(std::string_view argument, std::string_view option) {
    const std::string prefix = std::string(option) + "=";
    if (!argument.starts_with(prefix))
        throw std::invalid_argument("internal option parser error");
    return std::string(argument.substr(prefix.size()));
}

void printHelp(const char *program) {
    std::cout << "Usage: " << program << " [options]\n\n"
              << "Benchmarks Thor TypeConverter GPU conversions over a logarithmic tensor-size sweep.\n"
              << "Each timed sample rotates through disjoint, 256-byte-aligned tensor slots. Before\n"
              << "timing, a GPU kernel reads and writes an eviction buffer sized from the device's\n"
              << "reported L2 capacity, so repeated samples do not measure cache-resident inputs.\n\n"
              << "Options:\n"
              << "  --gpu=N                       GPU device index (default: 0)\n"
              << "  --suite=common|ml|all         Pair set when --pair is absent (default: common)\n"
              << "  --pair=SOURCE:DEST            Benchmark an explicit pair; may be repeated\n"
              << "  --sizes=N[,N...]              Element counts; K/M/G suffixes are binary\n"
              << "  --mode=out-of-place|in-place|both\n"
              << "  --samples=N                   Timed samples per case (default: 7)\n"
              << "  --target-traffic-mib=N        Desired logical read+write bytes per sample\n"
              << "  --max-working-set-mib=N       Maximum conversion buffers per case (default: 4096)\n"
              << "  --safety-reserve-mib=N        Leave this much currently free GPU memory unused\n"
              << "  --max-launches=N              Maximum launches inside one timed sample\n"
              << "  --l2-eviction-factor=N        Eviction bytes / reported L2 bytes (default: 4)\n"
              << "  --csv=PATH                    Write CSV to PATH instead of stdout\n"
              << "  --no-verify                   Skip sampled output validation\n"
              << "  --list-types                  Print accepted type names\n"
              << "  --help                        Show this message\n\n"
              << "Default sizes: 2K,8K,32K,128K,512K,2M,8M,32M,128M,256M elements.\n";
}

void printTypes() {
    for (const DataTypeInfo &info : STORAGE_TYPES)
        std::cout << info.name << '\n';
}

Options parseOptions(int argc, char **argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view argument(argv[i]);
        if (argument == "--help") {
            printHelp(argv[0]);
            std::exit(0);
        }
        if (argument == "--list-types") {
            printTypes();
            std::exit(0);
        }
        if (argument == "--no-verify") {
            options.verify = false;
        } else if (argument.starts_with("--gpu=")) {
            options.gpu = static_cast<int>(parseUnsigned(optionValue(argument, "--gpu"), "--gpu"));
        } else if (argument.starts_with("--suite=")) {
            options.suite = optionValue(argument, "--suite");
            if (options.suite != "common" && options.suite != "ml" && options.suite != "all")
                throw std::invalid_argument("--suite must be common, ml, or all");
        } else if (argument.starts_with("--pair=")) {
            options.explicitPairs.push_back(parsePair(optionValue(argument, "--pair")));
        } else if (argument.starts_with("--sizes=")) {
            options.elementCounts = parseElementCounts(optionValue(argument, "--sizes"));
        } else if (argument.starts_with("--mode=")) {
            const std::string mode = optionValue(argument, "--mode");
            if (mode == "out-of-place")
                options.modes = {BenchmarkMode::OUT_OF_PLACE};
            else if (mode == "in-place")
                options.modes = {BenchmarkMode::IN_PLACE};
            else if (mode == "both")
                options.modes = {BenchmarkMode::OUT_OF_PLACE, BenchmarkMode::IN_PLACE};
            else
                throw std::invalid_argument("--mode must be out-of-place, in-place, or both");
        } else if (argument.starts_with("--samples=")) {
            options.samples = static_cast<int>(parseUnsigned(optionValue(argument, "--samples"), "--samples"));
        } else if (argument.starts_with("--target-traffic-mib=")) {
            options.targetTrafficBytes = checkedMultiply(
                static_cast<std::size_t>(parseUnsigned(optionValue(argument, "--target-traffic-mib"), "--target-traffic-mib")),
                MiB,
                "target traffic");
        } else if (argument.starts_with("--max-working-set-mib=")) {
            options.maxWorkingSetBytes = checkedMultiply(
                static_cast<std::size_t>(parseUnsigned(optionValue(argument, "--max-working-set-mib"), "--max-working-set-mib")),
                MiB,
                "maximum working set");
        } else if (argument.starts_with("--safety-reserve-mib=")) {
            options.safetyReserveBytes = checkedMultiply(
                static_cast<std::size_t>(parseUnsigned(optionValue(argument, "--safety-reserve-mib"), "--safety-reserve-mib")),
                MiB,
                "safety reserve");
        } else if (argument.starts_with("--max-launches=")) {
            options.maxLaunchesPerSample = parseUnsigned(optionValue(argument, "--max-launches"), "--max-launches");
        } else if (argument.starts_with("--l2-eviction-factor=")) {
            const std::string value = optionValue(argument, "--l2-eviction-factor");
            options.l2EvictionFactor = std::stod(value);
        } else if (argument.starts_with("--csv=")) {
            options.csvPath = optionValue(argument, "--csv");
        } else {
            throw std::invalid_argument("unknown option '" + std::string(argument) + "'");
        }
    }

    if (options.samples < 1)
        throw std::invalid_argument("--samples must be at least one");
    if (options.targetTrafficBytes == 0)
        throw std::invalid_argument("--target-traffic-mib must be greater than zero");
    if (options.maxWorkingSetBytes == 0)
        throw std::invalid_argument("--max-working-set-mib must be greater than zero");
    if (options.maxLaunchesPerSample == 0)
        throw std::invalid_argument("--max-launches must be greater than zero");
    if (!std::isfinite(options.l2EvictionFactor) || options.l2EvictionFactor < 1.0)
        throw std::invalid_argument("--l2-eviction-factor must be finite and at least 1.0");

    std::sort(options.elementCounts.begin(), options.elementCounts.end());
    options.elementCounts.erase(std::unique(options.elementCounts.begin(), options.elementCounts.end()), options.elementCounts.end());
    return options;
}

std::vector<TypePair> commonPairs() {
    return {
        {DataType::FP32, DataType::FP16},     {DataType::FP16, DataType::FP32},     {DataType::FP32, DataType::BF16},
        {DataType::BF16, DataType::FP32},     {DataType::FP16, DataType::BF16},     {DataType::BF16, DataType::FP16},
        {DataType::FP32, DataType::FP8_E4M3}, {DataType::FP8_E4M3, DataType::FP32}, {DataType::BF16, DataType::FP8_E4M3},
        {DataType::FP8_E4M3, DataType::BF16}, {DataType::FP16, DataType::FP8_E4M3}, {DataType::FP8_E4M3, DataType::FP16},
        {DataType::FP32, DataType::FP64},     {DataType::FP64, DataType::FP32},     {DataType::INT32, DataType::FP32},
        {DataType::FP32, DataType::INT32},    {DataType::UINT8, DataType::FP32},    {DataType::FP32, DataType::UINT8},
        {DataType::BOOLEAN, DataType::FP32},  {DataType::FP32, DataType::BOOLEAN},
    };
}

std::vector<TypePair> buildPairs(const Options &options) {
    if (!options.explicitPairs.empty())
        return options.explicitPairs;

    if (options.suite == "common")
        return commonPairs();

    std::vector<TypePair> pairs;
    if (options.suite == "ml") {
        for (DataType source : ML_FLOAT_TYPES) {
            for (DataType destination : ML_FLOAT_TYPES) {
                if (source != destination)
                    pairs.push_back({source, destination});
            }
        }
        return pairs;
    }

    for (const DataTypeInfo &source : STORAGE_TYPES) {
        for (const DataTypeInfo &destination : STORAGE_TYPES) {
            if (source.dataType != destination.dataType)
                pairs.push_back({source.dataType, destination.dataType});
        }
    }
    return pairs;
}

__host__ __device__ __forceinline__ std::uint64_t mixBenchmarkIndex(std::uint64_t value) {
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

__host__ __device__ __forceinline__ std::uint32_t benchmarkNumericValue(std::uint64_t index) {
    const std::uint32_t selector = static_cast<std::uint32_t>(mixBenchmarkIndex(index) & 7ULL);
    return selector == 0 ? 0U : (1U << (selector - 1));
}

template <typename T>
__device__ __forceinline__ T benchmarkValue(std::uint64_t index) {
    return static_cast<T>(benchmarkNumericValue(index));
}

template <>
__device__ __forceinline__ half benchmarkValue<half>(std::uint64_t index) {
    return __float2half_rn(static_cast<float>(benchmarkNumericValue(index)));
}

template <>
__device__ __forceinline__ __nv_bfloat16 benchmarkValue<__nv_bfloat16>(std::uint64_t index) {
    return __float2bfloat16_rn(static_cast<float>(benchmarkNumericValue(index)));
}

template <>
__device__ __forceinline__ __nv_fp8_e4m3 benchmarkValue<__nv_fp8_e4m3>(std::uint64_t index) {
    return __nv_fp8_e4m3(static_cast<float>(benchmarkNumericValue(index)));
}

template <>
__device__ __forceinline__ __nv_fp8_e5m2 benchmarkValue<__nv_fp8_e5m2>(std::uint64_t index) {
    return __nv_fp8_e5m2(static_cast<float>(benchmarkNumericValue(index)));
}

template <typename T>
__global__ void fillBenchmarkInputKernel(T *destination, std::uint64_t elements) {
    const std::uint64_t first = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t stride = static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
    for (std::uint64_t i = first; i < elements; i += stride)
        destination[i] = benchmarkValue<T>(i);
}

__global__ void scrubL2Kernel(std::uint32_t *buffer, std::uint64_t words, std::uint32_t increment) {
    const std::uint64_t first = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t stride = static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
    for (std::uint64_t i = first; i < words; i += stride) {
        const std::uint32_t value = buffer[i];
        buffer[i] = value + increment;
    }
}

std::uint32_t gridSizeFor(std::uint64_t elements) {
    const std::uint64_t requested = (elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    return static_cast<std::uint32_t>(std::clamp<std::uint64_t>(requested, 1, 65535));
}

template <typename T>
void launchFill(void *destination, std::uint64_t elements, cudaStream_t stream) {
    fillBenchmarkInputKernel<T><<<gridSizeFor(elements), THREADS_PER_BLOCK, 0, stream>>>(static_cast<T *>(destination), elements);
}

void fillBenchmarkInput(DataType dataType, void *destination, std::size_t bytes, cudaStream_t stream) {
    const std::uint64_t elements = bytes / typeInfo(dataType).elementBytes;
    switch (dataType) {
        case DataType::FP8_E4M3:
            launchFill<__nv_fp8_e4m3>(destination, elements, stream);
            break;
        case DataType::FP8_E5M2:
            launchFill<__nv_fp8_e5m2>(destination, elements, stream);
            break;
        case DataType::FP16:
            launchFill<half>(destination, elements, stream);
            break;
        case DataType::BF16:
            launchFill<__nv_bfloat16>(destination, elements, stream);
            break;
        case DataType::FP32:
            launchFill<float>(destination, elements, stream);
            break;
        case DataType::FP64:
            launchFill<double>(destination, elements, stream);
            break;
        case DataType::INT8:
            launchFill<std::int8_t>(destination, elements, stream);
            break;
        case DataType::INT16:
            launchFill<std::int16_t>(destination, elements, stream);
            break;
        case DataType::INT32:
            launchFill<std::int32_t>(destination, elements, stream);
            break;
        case DataType::INT64:
            launchFill<std::int64_t>(destination, elements, stream);
            break;
        case DataType::UINT8:
            launchFill<std::uint8_t>(destination, elements, stream);
            break;
        case DataType::UINT16:
            launchFill<std::uint16_t>(destination, elements, stream);
            break;
        case DataType::UINT32:
            launchFill<std::uint32_t>(destination, elements, stream);
            break;
        case DataType::UINT64:
            launchFill<std::uint64_t>(destination, elements, stream);
            break;
        case DataType::BOOLEAN:
            launchFill<bool>(destination, elements, stream);
            break;
        case DataType::TF32:
            throw std::invalid_argument("TF32 is a compute token, not a TypeConverter storage type");
    }
    CUDA_CHECK(cudaGetLastError());
}

void scrubL2(DeviceBuffer &evictionBuffer, std::uint32_t increment, cudaStream_t stream) {
    const std::uint64_t words = evictionBuffer.bytes / sizeof(std::uint32_t);
    scrubL2Kernel<<<gridSizeFor(words), THREADS_PER_BLOCK, 0, stream>>>(
        static_cast<std::uint32_t *>(evictionBuffer.get()), words, increment);
    CUDA_CHECK(cudaGetLastError());
}

std::size_t computeEvictionBytes(std::size_t l2Bytes, double factor) {
    const long double scaled = static_cast<long double>(l2Bytes) * factor;
    if (scaled > static_cast<long double>(std::numeric_limits<std::size_t>::max()))
        throw std::overflow_error("L2 eviction buffer size overflows size_t");
    return alignUp(std::max<std::size_t>(64 * MiB, static_cast<std::size_t>(std::ceil(scaled))), SLOT_ALIGNMENT_BYTES);
}

BenchmarkPlan makePlan(
    TypePair pair, BenchmarkMode mode, std::uint64_t elements, const Options &options, std::size_t conversionMemoryBudgetBytes) {
    const std::size_t sourceBytes = checkedMultiply(static_cast<std::size_t>(elements), typeInfo(pair.source).elementBytes, "source bytes");
    const std::size_t destinationBytes =
        checkedMultiply(static_cast<std::size_t>(elements), typeInfo(pair.destination).elementBytes, "destination bytes");
    const std::size_t sourceStride = alignUp(sourceBytes, SLOT_ALIGNMENT_BYTES);
    const std::size_t destinationStride = alignUp(destinationBytes, SLOT_ALIGNMENT_BYTES);
    const std::size_t inPlaceStride = alignUp(std::max(sourceBytes, destinationBytes), SLOT_ALIGNMENT_BYTES);

    if (elements > static_cast<std::uint64_t>(std::numeric_limits<long>::max()))
        throw std::invalid_argument("element count exceeds TypeConverter's long numElements range");

    const std::size_t logicalBytesPerLaunch = checkedAdd(sourceBytes, destinationBytes, "logical bytes per launch");
    const std::uint64_t desiredForTraffic = static_cast<std::uint64_t>(options.targetTrafficBytes / logicalBytesPerLaunch) +
                                            static_cast<std::uint64_t>(options.targetTrafficBytes % logicalBytesPerLaunch != 0);
    std::uint64_t desiredLaunches = std::max<std::uint64_t>(1, desiredForTraffic);
    desiredLaunches = std::min(desiredLaunches, options.maxLaunchesPerSample);

    const std::size_t physicalBytesPerLaunch =
        mode == BenchmarkMode::OUT_OF_PLACE ? checkedAdd(sourceStride, destinationStride, "physical bytes per launch") : inPlaceStride;
    const std::uint64_t launchesByMemory = conversionMemoryBudgetBytes / physicalBytesPerLaunch;
    const std::uint64_t launches = std::min(desiredLaunches, launchesByMemory);
    if (launches == 0)
        throw std::runtime_error("one benchmark tensor slot exceeds the available conversion-buffer memory budget");

    return {
        sourceBytes,
        destinationBytes,
        sourceStride,
        destinationStride,
        inPlaceStride,
        launches,
        checkedMultiply(static_cast<std::size_t>(launches), physicalBytesPerLaunch, "working set bytes"),
    };
}

void launchConversion(const TypePair pair,
                      BenchmarkMode mode,
                      std::uint64_t elements,
                      const BenchmarkPlan &plan,
                      std::uint64_t slot,
                      DeviceBuffer &source,
                      DeviceBuffer &destination,
                      DeviceBuffer &inPlace,
                      Stream stream,
                      int gpu) {
    if (mode == BenchmarkMode::OUT_OF_PLACE) {
        auto *sourcePointer = static_cast<std::byte *>(source.get()) + slot * plan.sourceStrideBytes;
        auto *destinationPointer = static_cast<std::byte *>(destination.get()) + slot * plan.destinationStrideBytes;
        TypeConverter::convertType(
            sourcePointer, destinationPointer, pair.source, pair.destination, static_cast<long>(elements), stream, gpu);
    } else {
        auto *pointer = static_cast<std::byte *>(inPlace.get()) + slot * plan.inPlaceStrideBytes;
        TypeConverter::convertType(pointer, pointer, pair.source, pair.destination, static_cast<long>(elements), stream, gpu);
    }
}

void initializeCaseInput(TypePair pair,
                         BenchmarkMode mode,
                         const BenchmarkPlan &plan,
                         DeviceBuffer &source,
                         DeviceBuffer &destination,
                         DeviceBuffer &inPlace,
                         cudaStream_t stream) {
    if (mode == BenchmarkMode::OUT_OF_PLACE) {
        fillBenchmarkInput(pair.source, source.get(), source.bytes, stream);
        CUDA_CHECK(cudaMemsetAsync(destination.get(), 0xA5, destination.bytes, stream));
    } else {
        fillBenchmarkInput(pair.source, inPlace.get(), inPlace.bytes, stream);
    }
}

void verifyFirstSlot(TypePair pair,
                     BenchmarkMode mode,
                     std::uint64_t elements,
                     const BenchmarkPlan &plan,
                     DeviceBuffer &destination,
                     DeviceBuffer &inPlace,
                     cudaStream_t stream) {
    const std::uint64_t checkedElements = std::min<std::uint64_t>(elements, 16);
    const std::size_t destinationBytes =
        checkedMultiply(static_cast<std::size_t>(checkedElements), typeInfo(pair.destination).elementBytes, "verification bytes");
    std::vector<std::byte> host(destinationBytes);
    const void *devicePointer = mode == BenchmarkMode::OUT_OF_PLACE ? destination.get() : inPlace.get();
    CUDA_CHECK(cudaMemcpyAsync(host.data(), devicePointer, destinationBytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (std::uint64_t i = 0; i < checkedElements; ++i) {
        const double actual = std::stod(TensorDescriptor::getValueAsString(host.data(), i, pair.destination));
        const double sourceValue = pair.source == DataType::BOOLEAN ? static_cast<double>(benchmarkNumericValue(i) != 0)
                                                                    : static_cast<double>(benchmarkNumericValue(i));
        const double expected = pair.destination == DataType::BOOLEAN ? static_cast<double>(sourceValue != 0.0) : sourceValue;
        if (actual != expected) {
            std::ostringstream message;
            message << "verification failed for " << typeInfo(pair.source).name << " -> " << typeInfo(pair.destination).name
                    << " at element " << i << ": expected " << expected << ", got " << actual;
            throw std::runtime_error(message.str());
        }
    }
}

double percentile(const std::vector<double> &sorted, double fraction) {
    if (sorted.empty())
        throw std::invalid_argument("cannot compute percentile of an empty sample set");
    if (sorted.size() == 1)
        return sorted.front();
    const double position = fraction * static_cast<double>(sorted.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(std::floor(position));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
    const double weight = position - static_cast<double>(lower);
    return sorted[lower] * (1.0 - weight) + sorted[upper] * weight;
}

BenchmarkResult benchmarkCase(TypePair pair,
                              BenchmarkMode mode,
                              std::uint64_t elements,
                              const Options &options,
                              std::size_t conversionMemoryBudgetBytes,
                              DeviceBuffer &evictionBuffer,
                              Stream stream) {
    const BenchmarkPlan plan = makePlan(pair, mode, elements, options, conversionMemoryBudgetBytes);

    DeviceBuffer source;
    DeviceBuffer destination;
    DeviceBuffer inPlace;
    if (mode == BenchmarkMode::OUT_OF_PLACE) {
        source.allocate(checkedMultiply(static_cast<std::size_t>(plan.launchesPerSample), plan.sourceStrideBytes, "source allocation"));
        destination.allocate(
            checkedMultiply(static_cast<std::size_t>(plan.launchesPerSample), plan.destinationStrideBytes, "destination allocation"));
    } else {
        inPlace.allocate(checkedMultiply(static_cast<std::size_t>(plan.launchesPerSample), plan.inPlaceStrideBytes, "in-place allocation"));
    }

    initializeCaseInput(pair, mode, plan, source, destination, inPlace, stream.getStream());

    // Warm the launch path and allow GPU clocks to ramp using one complete untimed sample.
    for (std::uint64_t slot = 0; slot < plan.launchesPerSample; ++slot)
        launchConversion(pair, mode, elements, plan, slot, source, destination, inPlace, stream, options.gpu);
    CUDA_CHECK(cudaStreamSynchronize(stream.getStream()));

    std::vector<double> usPerLaunch;
    usPerLaunch.reserve(options.samples);
    Event start(options.gpu, /*enableTiming=*/true);
    Event stop(options.gpu, /*enableTiming=*/true);
    for (int sample = 0; sample < options.samples; ++sample) {
        if (mode == BenchmarkMode::IN_PLACE)
            initializeCaseInput(pair, mode, plan, source, destination, inPlace, stream.getStream());

        scrubL2(evictionBuffer, static_cast<std::uint32_t>(sample + 1), stream.getStream());

        start.record(stream);
        for (std::uint64_t slot = 0; slot < plan.launchesPerSample; ++slot)
            launchConversion(pair, mode, elements, plan, slot, source, destination, inPlace, stream, options.gpu);
        stop.record(stream);
        const float totalMilliseconds = stop.synchronizeAndReportElapsedTimeInMilliseconds(start);
        usPerLaunch.push_back(static_cast<double>(totalMilliseconds) * 1000.0 / static_cast<double>(plan.launchesPerSample));
    }

    if (options.verify)
        verifyFirstSlot(pair, mode, elements, plan, destination, inPlace, stream.getStream());

    std::sort(usPerLaunch.begin(), usPerLaunch.end());
    const double medianUs = percentile(usPerLaunch, 0.5);
    const double p10Us = percentile(usPerLaunch, 0.1);
    const double p90Us = percentile(usPerLaunch, 0.9);
    const double logicalBytes = static_cast<double>(plan.sourceBytesPerLaunch + plan.destinationBytesPerLaunch);

    return {
        pair,
        mode,
        elements,
        plan,
        medianUs,
        p10Us,
        p90Us,
        medianUs * 1000.0 / static_cast<double>(elements),
        logicalBytes / (medianUs * 1000.0),
        static_cast<double>(plan.sourceBytesPerLaunch) / (medianUs * 1000.0),
        static_cast<double>(plan.destinationBytesPerLaunch) / (medianUs * 1000.0),
    };
}

void writeCsvHeader(std::ostream &output) {
    output << "source_type,destination_type,mode,elements,source_bytes,destination_bytes,logical_bytes_per_launch,"
              "launches_per_sample,samples,median_us_per_launch,p10_us_per_launch,p90_us_per_launch,median_ns_per_element,"
              "effective_gbps,input_gbps,output_gbps,working_set_bytes,l2_bytes,eviction_bytes\n";
}

void writeCsvRow(
    std::ostream &output, const BenchmarkResult &result, const Options &options, std::size_t l2Bytes, std::size_t evictionBytes) {
    output << typeInfo(result.pair.source).name << ',' << typeInfo(result.pair.destination).name << ',' << modeName(result.mode) << ','
           << result.elements << ',' << result.plan.sourceBytesPerLaunch << ',' << result.plan.destinationBytesPerLaunch << ','
           << (result.plan.sourceBytesPerLaunch + result.plan.destinationBytesPerLaunch) << ',' << result.plan.launchesPerSample << ','
           << options.samples << ',' << std::fixed << std::setprecision(6) << result.medianUsPerLaunch << ',' << result.p10UsPerLaunch
           << ',' << result.p90UsPerLaunch << ',' << result.medianNsPerElement << ',' << result.effectiveGbPerSecond << ','
           << result.inputGbPerSecond << ',' << result.outputGbPerSecond << ',' << result.plan.workingSetBytes << ',' << l2Bytes << ','
           << evictionBytes << '\n';
    output.flush();
}

std::string formatBytes(std::size_t bytes) {
    std::ostringstream output;
    if (bytes >= GiB)
        output << std::fixed << std::setprecision(2) << static_cast<double>(bytes) / GiB << " GiB";
    else if (bytes >= MiB)
        output << std::fixed << std::setprecision(2) << static_cast<double>(bytes) / MiB << " MiB";
    else if (bytes >= KiB)
        output << std::fixed << std::setprecision(2) << static_cast<double>(bytes) / KiB << " KiB";
    else
        output << bytes << " B";
    return output.str();
}

}  // namespace

int main(int argc, char **argv) {
    try {
        const Options options = parseOptions(argc, argv);

        int deviceCount = 0;
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        if (options.gpu < 0 || options.gpu >= deviceCount)
            throw std::invalid_argument("--gpu is outside the available CUDA device range");
        ScopedGpu scopedGpu(options.gpu);

        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, options.gpu));

        int l2BytesAttribute = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&l2BytesAttribute, cudaDevAttrL2CacheSize, options.gpu));
        if (l2BytesAttribute <= 0)
            throw std::runtime_error("CUDA reported a non-positive L2 cache size");
        const std::size_t l2Bytes = static_cast<std::size_t>(l2BytesAttribute);
        const std::size_t evictionBytes = computeEvictionBytes(l2Bytes, options.l2EvictionFactor);

        std::size_t freeBytes = 0;
        std::size_t totalBytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
        const std::size_t nonConversionReserve = checkedAdd(options.safetyReserveBytes, evictionBytes, "GPU memory reserve");
        if (freeBytes <= nonConversionReserve)
            throw std::runtime_error("not enough currently free GPU memory for the requested safety reserve and L2 eviction buffer");
        const std::size_t conversionMemoryBudgetBytes = std::min(options.maxWorkingSetBytes, freeBytes - nonConversionReserve);

        const std::vector<TypePair> pairs = buildPairs(options);
        Stream stream(options.gpu);
        DeviceBuffer evictionBuffer(evictionBytes);
        CUDA_CHECK(cudaMemsetAsync(evictionBuffer.get(), 0x3C, evictionBuffer.bytes, stream.getStream()));
        scrubL2(evictionBuffer, 1, stream.getStream());
        CUDA_CHECK(cudaStreamSynchronize(stream.getStream()));

        std::unique_ptr<std::ofstream> csvFile;
        std::ostream *csv = &std::cout;
        if (options.csvPath.has_value()) {
            csvFile = std::make_unique<std::ofstream>(*options.csvPath, std::ios::out | std::ios::trunc);
            if (!*csvFile)
                throw std::runtime_error("could not open CSV output path '" + *options.csvPath + "'");
            csv = csvFile.get();
        }

        std::cerr << "GPU: " << properties.name << " (device " << options.gpu << ")\n"
                  << "Reported L2: " << formatBytes(l2Bytes) << "\n"
                  << "L2 eviction buffer: " << formatBytes(evictionBytes) << " (" << options.l2EvictionFactor
                  << "x reported L2, minimum 64 MiB)\n"
                  << "GPU memory: " << formatBytes(freeBytes) << " free / " << formatBytes(totalBytes) << " total\n"
                  << "Conversion-buffer budget: " << formatBytes(conversionMemoryBudgetBytes) << "\n"
                  << "Pairs: " << pairs.size() << ", sizes: " << options.elementCounts.size() << ", modes: " << options.modes.size()
                  << ", samples/case: " << options.samples << "\n";

        writeCsvHeader(*csv);
        std::size_t completed = 0;
        std::size_t skipped = 0;
        const std::size_t totalCases = pairs.size() * options.elementCounts.size() * options.modes.size();
        for (const TypePair pair : pairs) {
            for (const BenchmarkMode mode : options.modes) {
                for (const std::uint64_t elements : options.elementCounts) {
                    try {
                        const BenchmarkResult result =
                            benchmarkCase(pair, mode, elements, options, conversionMemoryBudgetBytes, evictionBuffer, stream);
                        writeCsvRow(*csv, result, options, l2Bytes, evictionBytes);
                        ++completed;
                        std::cerr << '[' << completed + skipped << '/' << totalCases << "] " << typeInfo(pair.source).name << " -> "
                                  << typeInfo(pair.destination).name << ' ' << modeName(mode) << " elements=" << elements
                                  << " median=" << std::fixed << std::setprecision(3) << result.medianUsPerLaunch << " us, "
                                  << result.effectiveGbPerSecond << " GB/s\n";
                    } catch (const std::runtime_error &error) {
                        const std::string message = error.what();
                        if (message.find("exceeds the available conversion-buffer memory budget") == std::string::npos)
                            throw;
                        ++skipped;
                        std::cerr << '[' << completed + skipped << '/' << totalCases << "] skipped " << typeInfo(pair.source).name << " -> "
                                  << typeInfo(pair.destination).name << ' ' << modeName(mode) << " elements=" << elements << ": " << message
                                  << '\n';
                    }
                }
            }
        }

        std::cerr << "Completed " << completed << " cases";
        if (skipped != 0)
            std::cerr << ", skipped " << skipped << " cases that could not fit one padded tensor slot";
        std::cerr << ".\n";
        if (options.csvPath.has_value())
            std::cerr << "CSV: " << *options.csvPath << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "TypeConverter benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
