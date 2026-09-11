#include "DeepLearning/Implementation/Data/Residency/DeviceResidentDirectMaterializationKernel.h"
#include "DeepLearning/Implementation/Data/Residency/DeviceResidentNamedGatherKernel.h"
#include "DeepLearning/Implementation/Data/Residency/DeviceResidentRaggedMaterializationKernel.h"
#include "DeepLearning/Implementation/Data/Residency/DeviceResidentRowGrouping.h"
#include "DeepLearning/Implementation/Data/Residency/DeviceResidentWindowMaterializationKernel.h"
#include "DeepLearning/Implementation/ThorError.h"
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

struct Options {
    int gpu = 0;
    int warmup = 5;
    int iterations = 20;
    double l2_evict_multiple = 4.0;
    uint64_t min_evict_bytes = 64ULL * MIB;
    uint64_t source_rows_floor = 4096;

    std::vector<std::string> kernels{"direct", "named", "ragged", "window"};
    std::vector<uint64_t> batch_rows{
        32, 64, 127, 128, 255, 256, 511, 512, 1023, 1024,
        2047, 2048, 4095, 4096, 8191, 8192, 16383, 16384};
    std::vector<uint64_t> row_bytes{
        4, 16, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257,
        511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049, 4095, 4096, 4097};
    std::vector<std::string> direct_layouts{"aligned32", "packed"};
    std::vector<std::string> ragged_reference_layouts{"aligned8", "unaligned"};
    std::vector<uint64_t> ragged_value_bytes{1, 4};
    std::vector<uint64_t> window_step_bytes{1, 4, 16, 32};
    std::vector<double> window_valid_fractions{1.0, 0.5};
    std::vector<std::string> window_modes{"payload", "mask"};
    // 0 means use the production selector. Any explicit value forces the
    // corresponding kernel specialization so launch geometry can be tuned
    // independently of the current heuristic.
    std::vector<uint32_t> rows_per_cta{0};
};

template <typename T>
DataType dtypeFor();
template <>
DataType dtypeFor<uint8_t>() { return DataType::UINT8; }
template <>
DataType dtypeFor<uint32_t>() { return DataType::UINT32; }
template <>
DataType dtypeFor<uint64_t>() { return DataType::UINT64; }

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

std::vector<double> parseDoubleCsv(std::string_view text, std::string_view flag) {
    return parseCsv(text, flag, [&](std::string_view token) {
        return parseDouble(token, flag);
    });
}

std::vector<std::string> parseStringCsv(std::string_view text, std::string_view flag) {
    return parseCsv(text, flag, [](std::string_view token) { return std::string(token); });
}

std::vector<uint32_t> parseRowsPerCtaCsv(std::string_view text, std::string_view flag) {
    return parseCsv(text, flag, [&](std::string_view token) -> uint32_t {
        if (token == "auto") return 0;
        const uint64_t value = parseUnsigned(token, flag);
        if (value == 0 || value > 256 || (value & (value - 1)) != 0) {
            throw std::invalid_argument(
                std::string(flag) + " values must be auto or a power of two in [1,256]");
        }
        return static_cast<uint32_t>(value);
    });
}

bool contains(const std::vector<std::string> &values, std::string_view value) {
    return std::find(values.begin(), values.end(), value) != values.end();
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

Options parseOptions(int argc, char **argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        auto requireValue = [&](std::string_view flag) -> std::string_view {
            if (i + 1 >= argc) throw std::invalid_argument("Missing value for " + std::string(flag));
            return argv[++i];
        };
        if (arg == "--gpu") {
            options.gpu = static_cast<int>(parseUnsigned(requireValue(arg), arg));
        } else if (arg == "--warmup") {
            options.warmup = static_cast<int>(parseUnsigned(requireValue(arg), arg));
        } else if (arg == "--iterations") {
            options.iterations = static_cast<int>(parseUnsigned(requireValue(arg), arg));
        } else if (arg == "--l2-evict-multiple") {
            options.l2_evict_multiple = parseDouble(requireValue(arg), arg);
        } else if (arg == "--min-evict-mib") {
            options.min_evict_bytes = parseUnsigned(requireValue(arg), arg) * MIB;
        } else if (arg == "--source-rows-floor") {
            options.source_rows_floor = parseUnsigned(requireValue(arg), arg);
        } else if (arg == "--kernels") {
            options.kernels = parseStringCsv(requireValue(arg), arg);
        } else if (arg == "--batch-rows") {
            options.batch_rows = parseUnsignedCsv(requireValue(arg), arg);
        } else if (arg == "--row-bytes") {
            options.row_bytes = parseUnsignedCsv(requireValue(arg), arg);
        } else if (arg == "--direct-layouts") {
            options.direct_layouts = parseStringCsv(requireValue(arg), arg);
        } else if (arg == "--ragged-reference-layouts") {
            options.ragged_reference_layouts = parseStringCsv(requireValue(arg), arg);
        } else if (arg == "--ragged-value-bytes") {
            options.ragged_value_bytes = parseUnsignedCsv(requireValue(arg), arg);
        } else if (arg == "--window-step-bytes") {
            options.window_step_bytes = parseUnsignedCsv(requireValue(arg), arg);
        } else if (arg == "--window-valid-fractions") {
            options.window_valid_fractions = parseDoubleCsv(requireValue(arg), arg);
        } else if (arg == "--window-modes") {
            options.window_modes = parseStringCsv(requireValue(arg), arg);
        } else if (arg == "--rows-per-cta") {
            options.rows_per_cta = parseRowsPerCtaCsv(requireValue(arg), arg);
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: thor_residency_materialization_benchmark [options]\n"
                << "  --gpu N\n"
                << "  --warmup N                         warm launches per case (default 5)\n"
                << "  --iterations N                     timed launches per case (default 20)\n"
                << "  --l2-evict-multiple X              device-read eviction bytes / L2 (default 4.0)\n"
                << "  --min-evict-mib N                  minimum eviction buffer size (default 64)\n"
                << "  --source-rows-floor N              minimum resident source rows (default 4096)\n"
                << "  --kernels direct,named,ragged,window\n"
                << "  --batch-rows a,b,c\n"
                << "  --row-bytes a,b,c                  payload bytes swept by direct/named/ragged and used as target window row bytes\n"
                << "  --direct-layouts aligned32,packed\n"
                << "  --ragged-reference-layouts aligned8,unaligned\n"
                << "  --ragged-value-bytes 1,4\n"
                << "  --window-step-bytes 1,4,16,32\n"
                << "  --window-valid-fractions 1.0,0.5\n"
                << "  --window-modes payload,mask\n"
                << "  --rows-per-cta auto,1,2,4,...,256  force/tune CTA row grouping (default auto)\n\n"
                << "The benchmark warms each case, then performs an untimed device read over a buffer\n"
                << "larger than L2 before every timed launch. The timed target kernel therefore sees\n"
                << "DRAM-backed source reads without including cache eviction or initialization time.\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("Unknown argument: " + std::string(arg));
        }
    }

    if (options.gpu < 0) throw std::invalid_argument("--gpu must be non-negative");
    if (options.warmup < 0) throw std::invalid_argument("--warmup must be non-negative");
    if (options.iterations <= 0) throw std::invalid_argument("--iterations must be positive");
    if (options.l2_evict_multiple < 2.0) {
        throw std::invalid_argument("--l2-evict-multiple must be >= 2.0 to reliably displace target data from L2");
    }
    requireAllowed(options.kernels, {"direct", "named", "ragged", "window"}, "--kernels");
    requireAllowed(options.direct_layouts, {"aligned32", "packed"}, "--direct-layouts");
    requireAllowed(options.ragged_reference_layouts, {"aligned8", "unaligned"}, "--ragged-reference-layouts");
    requireAllowed(options.window_modes, {"payload", "mask"}, "--window-modes");
    for (double fraction : options.window_valid_fractions) {
        if (!(fraction >= 0.0 && fraction <= 1.0)) {
            throw std::invalid_argument("--window-valid-fractions values must be in [0,1]");
        }
    }
    for (uint64_t valueBytes : options.ragged_value_bytes) {
        if (!(valueBytes == 1 || valueBytes == 2 || valueBytes == 4 || valueBytes == 8 || valueBytes == 16 || valueBytes == 32)) {
            throw std::invalid_argument("--ragged-value-bytes values must be one of 1,2,4,8,16,32");
        }
    }
    return options;
}

uint64_t ceilDiv(uint64_t numerator, uint64_t denominator) {
    return numerator / denominator + static_cast<uint64_t>(numerator % denominator != 0);
}

uint64_t roundUp(uint64_t value, uint64_t alignment) {
    return ceilDiv(value, alignment) * alignment;
}

uint32_t selectedRowsPerCta(uint64_t logicalRows, uint64_t rowBytes) {
    // Call the production selector directly so benchmark-reported auto geometry
    // cannot drift away from the kernels being measured.
    return DeviceResidentRowGrouping::selectRowsPerCta(logicalRows, rowBytes);
}

uint64_t splitMix64(uint64_t value) {
    uint64_t z = value + 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27U)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31U);
}

std::vector<uint64_t> makeCollisionFreeRowIndices(uint64_t batchRows, uint64_t sourceRows) {
    if (batchRows > sourceRows) {
        throw std::invalid_argument(
            "Collision-free benchmark row selection requires sourceRows >= batchRows");
    }
    std::vector<uint64_t> indices(batchRows);
    if (batchRows == 0) return indices;
    if (sourceRows == 1) {
        indices[0] = 0;
        return indices;
    }

    // Walk an affine permutation modulo sourceRows. Choosing a stride coprime
    // to sourceRows guarantees that no selected source row repeats until the
    // entire resident source has been traversed. The hashed start/stride keeps
    // successive logical batch rows distributed across the resident allocation
    // instead of turning this into a sequential-memory benchmark.
    const uint64_t start = splitMix64(sourceRows ^ 0x4d595df4d0f33173ULL) % sourceRows;
    uint64_t stride = splitMix64(sourceRows ^ 0x94d049bb133111ebULL) % sourceRows;
    if (stride == 0) stride = 1;
    while (std::gcd(stride, sourceRows) != 1) {
        ++stride;
        if (stride == sourceRows) stride = 1;
    }

    uint64_t current = start;
    for (uint64_t row = 0; row < batchRows; ++row) {
        indices[row] = current;
        // Overflow-free modular addition.
        current = current >= sourceRows - stride
                      ? current - (sourceRows - stride)
                      : current + stride;
    }
    return indices;
}

template <typename T>
Tensor makeGpuTensor(int gpu,
                     const std::vector<uint64_t> &dimensions,
                     const std::vector<T> &values,
                     Stream &stream) {
    const TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    const TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, gpu);
    Tensor host(cpuPlacement, TensorDescriptor(dtypeFor<T>(), dimensions));
    if (host.getTotalNumElements() != values.size()) {
        throw std::runtime_error("Residency benchmark host tensor value count mismatch");
    }
    if (!values.empty()) {
        std::memcpy(host.getMemPtr(), values.data(), values.size() * sizeof(T));
    }
    Tensor device(gpuPlacement, host.getDescriptor());
    device.copyFromAsync(host, stream);
    stream.synchronize();
    return device;
}

Tensor makeGpuBytes(int gpu, uint64_t bytes) {
    if (bytes == 0) throw std::invalid_argument("Cannot allocate a zero-byte benchmark tensor");
    return Tensor(TensorPlacement(TensorPlacement::MemDevices::GPU, gpu),
                  TensorDescriptor(DataType::UINT8, {bytes}));
}

Tensor makeGpuByteTensor(int gpu, const std::vector<uint64_t> &dimensions) {
    if (dimensions.empty()) throw std::invalid_argument("Benchmark tensor dimensions cannot be empty");
    return Tensor(TensorPlacement(TensorPlacement::MemDevices::GPU, gpu),
                  TensorDescriptor(DataType::UINT8, dimensions));
}

Tensor makeGpuPlans(int gpu,
                    const std::vector<DeviceResidentWindowRowPlan32> &plans,
                    Stream &stream) {
    const uint64_t bytes = plans.size() * sizeof(DeviceResidentWindowRowPlan32);
    const TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    const TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, gpu);
    Tensor host(cpuPlacement, TensorDescriptor(DataType::UINT8, {bytes}));
    if (bytes != 0) std::memcpy(host.getMemPtr(), plans.data(), static_cast<size_t>(bytes));
    Tensor device(gpuPlacement, host.getDescriptor());
    device.copyFromAsync(host, stream);
    stream.synchronize();
    return device;
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
    // One atomic per warp is enough to keep the reads observable. The eviction
    // launch is outside the timed interval, so this reduction overhead is irrelevant.
    if ((threadIdx.x & 31U) == 0U) atomicAdd(sink, accumulator);
}

class DramReadEvictor {
   public:
    DramReadEvictor(const Options &options,
                    const cudaDeviceProp &properties,
                    Stream &stream)
        : gpu_(options.gpu),
          bytes_(std::max<uint64_t>(options.min_evict_bytes,
                                    static_cast<uint64_t>(std::ceil(
                                        options.l2_evict_multiple * static_cast<double>(properties.l2CacheSize))))),
          storage_(makeGpuBytes(options.gpu, roundUp(bytes_, sizeof(uint4)))),
          sink_(TensorPlacement(TensorPlacement::MemDevices::GPU, options.gpu),
                TensorDescriptor(DataType::UINT64, {1})),
          blocks_(std::max(1, properties.multiProcessorCount * 8)) {
        storage_.memsetAsync(stream, static_cast<int8_t>(0x5a));
        sink_.memsetAsync(stream, 0);
        stream.synchronize();
        // Warm the eviction kernel itself before benchmark timing starts.
        run(stream);
        stream.synchronize();
    }

    void run(Stream &stream) {
        constexpr uint32_t threads = 256;
        const uint64_t count = storage_.getArraySizeInBytes() / sizeof(uint4);
        readForL2EvictionKernel<<<blocks_, threads, 0, stream.getStream()>>>(
            reinterpret_cast<const uint4 *>(storage_.getMemPtr()),
            count,
            reinterpret_cast<unsigned long long *>(sink_.getMemPtr<uint64_t>()));
        CUDA_CHECK(cudaGetLastError());
    }

    uint64_t bytes() const { return storage_.getArraySizeInBytes(); }

   private:
    int gpu_ = 0;
    uint64_t bytes_ = 0;
    Tensor storage_;
    Tensor sink_;
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
    return (1.0 - alpha) * static_cast<double>(values[lo]) +
           alpha * static_cast<double>(values[hi]);
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
        // The eviction pass is deliberately not timed. Because it is a device
        // read over >L2 bytes on the same stream, target source lines from the
        // previous sample are displaced before the next target launch. This
        // preserves a warmed launch/code path while measuring DRAM-backed reads.
        evictor.run(stream);
        start.record(stream);
        fn();
        stop.record(stream);
        milliseconds.push_back(stop.synchronizeAndReportElapsedTimeInMilliseconds(start));
    }

    std::sort(milliseconds.begin(), milliseconds.end());
    TimingStats stats;
    stats.p10_us = percentileSorted(milliseconds, 0.10) * 1000.0;
    stats.median_us = percentileSorted(milliseconds, 0.50) * 1000.0;
    stats.p90_us = percentileSorted(milliseconds, 0.90) * 1000.0;
    return stats;
}

struct Result {
    std::string kernel;
    std::string variant;
    uint64_t batchRows = 0;
    uint64_t rowBytes = 0;
    uint64_t sourceStepBytes = 0;
    uint64_t windowLength = 0;
    double validFraction = 1.0;
    std::string groupingMode = "auto";
    uint32_t autoRowsPerCta = 0;
    uint32_t rowsPerCta = 0;
    uint32_t lanesPerRow = 0;
    uint64_t sourceReadBytes = 0;
    uint64_t destinationWriteBytes = 0;
    TimingStats timing;
};

double usefulGBps(uint64_t bytes, double microseconds) {
    if (microseconds <= 0.0) return 0.0;
    return (static_cast<double>(bytes) / BYTES_PER_GB) / (microseconds * 1.0e-6);
}

void printCsvHeader() {
    std::cout
        << "kernel,variant,batch_rows,row_bytes,source_step_bytes,window_length,valid_fraction,"
        << "grouping_mode,auto_rows_per_cta,rows_per_cta,lanes_per_row,source_read_bytes,destination_write_bytes,p10_us,median_us,p90_us,"
        << "source_payload_GBps,output_payload_GBps\n";
}

void printResult(const Result &result) {
    std::cout << result.kernel << ','
              << result.variant << ','
              << result.batchRows << ','
              << result.rowBytes << ','
              << result.sourceStepBytes << ','
              << result.windowLength << ','
              << std::fixed << std::setprecision(3) << result.validFraction << ','
              << result.groupingMode << ','
              << result.autoRowsPerCta << ','
              << result.rowsPerCta << ','
              << result.lanesPerRow << ','
              << result.sourceReadBytes << ','
              << result.destinationWriteBytes << ','
              << std::setprecision(3) << result.timing.p10_us << ','
              << result.timing.median_us << ','
              << result.timing.p90_us << ','
              << std::setprecision(2) << usefulGBps(result.sourceReadBytes, result.timing.median_us) << ','
              << usefulGBps(result.destinationWriteBytes, result.timing.median_us) << '\n';
}


void runDirectBenchmarks(const Options &options,
                         Stream &stream,
                         DramReadEvictor &evictor) {
    for (const std::string &layout : options.direct_layouts) {
        for (uint64_t rowBytes : options.row_bytes) {
            if (rowBytes == 0) continue;
            const uint64_t fieldOffsetBytes = layout == "aligned32" ? 0 : 3;
            const uint64_t recordSizeBytes = layout == "aligned32"
                                                 ? roundUp(fieldOffsetBytes + rowBytes, 32)
                                                 : fieldOffsetBytes + rowBytes + 5;
            for (uint64_t batchRows : options.batch_rows) {
                const uint64_t sourceRows = std::max<uint64_t>(options.source_rows_floor, batchRows * 4);
                if (sourceRows > std::numeric_limits<uint64_t>::max() / recordSizeBytes) {
                    throw std::overflow_error("Direct benchmark source byte count overflow");
                }

                Tensor records = makeGpuBytes(options.gpu, sourceRows * recordSizeBytes);
                Tensor destination = makeGpuByteTensor(options.gpu, {batchRows, rowBytes});
                Tensor indices = makeGpuTensor<uint64_t>(
                    options.gpu, {batchRows}, makeCollisionFreeRowIndices(batchRows, sourceRows), stream);
                records.memsetAsync(stream, static_cast<int8_t>(0x3c));
                destination.memsetAsync(stream, static_cast<int8_t>(0));
                stream.synchronize();

                const uint32_t automaticRowsPerCta =
                    selectedRowsPerCta(batchRows, rowBytes);
                for (uint32_t requestedRowsPerCta : options.rows_per_cta) {
                    const uint32_t actualRowsPerCta =
                        requestedRowsPerCta == 0 ? automaticRowsPerCta : requestedRowsPerCta;
                    const TimingStats timing = benchmarkDramReads(
                        stream, evictor, options.warmup, options.iterations, [&] {
                            if (requestedRowsPerCta == 0) {
                                launchDeviceResidentDirectMaterializationKernel(
                                    records, sourceRows, recordSizeBytes, fieldOffsetBytes,
                                    rowBytes, batchRows, destination, indices, stream);
                            } else {
                                launchDeviceResidentDirectMaterializationKernelWithRowsPerCtaForBenchmark(
                                    records, sourceRows, recordSizeBytes, fieldOffsetBytes,
                                    rowBytes, batchRows, destination, indices,
                                    requestedRowsPerCta, stream);
                            }
                        });

                    Result result;
                    result.kernel = "direct";
                    result.variant = layout;
                    result.batchRows = batchRows;
                    result.rowBytes = rowBytes;
                    result.groupingMode = requestedRowsPerCta == 0 ? "auto" : "forced";
                    result.autoRowsPerCta = automaticRowsPerCta;
                    result.rowsPerCta = actualRowsPerCta;
                    result.lanesPerRow = 256U / actualRowsPerCta;
                    result.sourceReadBytes = batchRows * rowBytes;
                    result.destinationWriteBytes = batchRows * rowBytes;
                    result.timing = timing;
                    printResult(result);
                }
            }
        }
    }
}

void runNamedGatherBenchmarks(const Options &options,
                              Stream &stream,
                              DramReadEvictor &evictor) {
    for (uint64_t rowBytes : options.row_bytes) {
        for (uint64_t batchRows : options.batch_rows) {
            const uint64_t sourceRows = std::max<uint64_t>(options.source_rows_floor, batchRows * 4);
            if (sourceRows > std::numeric_limits<uint64_t>::max() / rowBytes) {
                throw std::overflow_error("Named-gather benchmark source byte count overflow");
            }

            Tensor source(TensorPlacement(TensorPlacement::MemDevices::GPU, options.gpu),
                          TensorDescriptor(DataType::UINT8, {sourceRows, rowBytes}));
            Tensor destination(TensorPlacement(TensorPlacement::MemDevices::GPU, options.gpu),
                               TensorDescriptor(DataType::UINT8, {batchRows, rowBytes}));
            Tensor indices = makeGpuTensor<uint64_t>(
                options.gpu, {batchRows}, makeCollisionFreeRowIndices(batchRows, sourceRows), stream);
            source.memsetAsync(stream, static_cast<int8_t>(0x67));
            destination.memsetAsync(stream, 0);
            stream.synchronize();

            const uint32_t automaticRowsPerCta =
                selectedRowsPerCta(batchRows, rowBytes);
            for (uint32_t requestedRowsPerCta : options.rows_per_cta) {
                const uint32_t actualRowsPerCta =
                    requestedRowsPerCta == 0 ? automaticRowsPerCta : requestedRowsPerCta;
                const TimingStats timing = benchmarkDramReads(
                    stream, evictor, options.warmup, options.iterations, [&] {
                        if (requestedRowsPerCta == 0) {
                            launchDeviceResidentNamedGatherKernel(
                                source, destination, indices, batchRows, stream);
                        } else {
                            launchDeviceResidentNamedGatherKernelWithRowsPerCtaForBenchmark(
                                source, destination, indices, batchRows,
                                requestedRowsPerCta, stream);
                        }
                    });

                Result result;
                result.kernel = "named";
                result.variant = "gather";
                result.batchRows = batchRows;
                result.rowBytes = rowBytes;
                result.groupingMode = requestedRowsPerCta == 0 ? "auto" : "forced";
                result.autoRowsPerCta = automaticRowsPerCta;
                result.rowsPerCta = actualRowsPerCta;
                result.lanesPerRow = 256U / actualRowsPerCta;
                result.sourceReadBytes = batchRows * rowBytes;
                result.destinationWriteBytes = batchRows * rowBytes;
                result.timing = timing;
                printResult(result);
            }
        }
    }
}

void writeUint64LittleEndian(std::vector<uint8_t> &bytes, uint64_t offset, uint64_t value) {
    for (uint32_t byte = 0; byte < sizeof(uint64_t); ++byte) {
        bytes.at(offset + byte) = static_cast<uint8_t>((value >> (8U * byte)) & 0xffU);
    }
}

void runRaggedBenchmarks(const Options &options,
                         Stream &stream,
                         DramReadEvictor &evictor) {
    constexpr uint64_t recordSizeBytes = 32;

    for (const std::string &referenceLayout : options.ragged_reference_layouts) {
        const uint64_t referenceOffsetBytes = referenceLayout == "aligned8" ? 8 : 3;
        for (uint64_t valueBytes : options.ragged_value_bytes) {
            for (uint64_t requestedRowBytes : options.row_bytes) {
                // Preserve the requested byte boundary exactly. The valueBytes=1
                // sweep covers every odd threshold; wider element sizes run only
                // cases that are physically representable without changing rowBytes.
                if (requestedRowBytes % valueBytes != 0) continue;
                const uint64_t valuesPerRow = requestedRowBytes / valueBytes;
                const uint64_t rowBytes = requestedRowBytes;
                for (uint64_t batchRows : options.batch_rows) {
                    const uint64_t sourceRows = std::max<uint64_t>(options.source_rows_floor, batchRows * 4);
                    if (sourceRows > std::numeric_limits<uint64_t>::max() / valuesPerRow) {
                        throw std::overflow_error("Ragged benchmark stored value count overflow");
                    }
                    const uint64_t storedValueCount = sourceRows * valuesPerRow;
                    if (storedValueCount > std::numeric_limits<uint64_t>::max() / valueBytes) {
                        throw std::overflow_error("Ragged benchmark packed byte count overflow");
                    }

                    std::vector<uint8_t> recordBytes(sourceRows * recordSizeBytes, 0);
                    for (uint64_t sourceRow = 0; sourceRow < sourceRows; ++sourceRow) {
                        const uint64_t reference = sourceRow * recordSizeBytes + referenceOffsetBytes;
                        writeUint64LittleEndian(recordBytes, reference, sourceRow * valuesPerRow);
                        writeUint64LittleEndian(recordBytes, reference + sizeof(uint64_t), valuesPerRow);
                    }
                    Tensor records = makeGpuTensor<uint8_t>(
                        options.gpu, {recordBytes.size()}, recordBytes, stream);
                    Tensor packedValues = makeGpuBytes(options.gpu, storedValueCount * valueBytes);
                    packedValues.memsetAsync(stream, static_cast<int8_t>(0x2d));

                    const std::vector<uint64_t> rowIndicesHost =
                        makeCollisionFreeRowIndices(batchRows, sourceRows);
                    Tensor indices = makeGpuTensor<uint64_t>(
                        options.gpu, {batchRows}, rowIndicesHost, stream);

                    std::vector<uint32_t> offsets(batchRows + 1, 0);
                    for (uint64_t row = 0; row <= batchRows; ++row) {
                        const uint64_t offset = row * valuesPerRow;
                        if (offset > std::numeric_limits<uint32_t>::max()) {
                            throw std::runtime_error(
                                "Ragged benchmark case exceeds UINT32 destination offsets; reduce --batch-rows or --row-bytes");
                        }
                        offsets[row] = static_cast<uint32_t>(offset);
                    }
                    Tensor destinationOffsets = makeGpuTensor<uint32_t>(
                        options.gpu, {batchRows + 1}, offsets, stream);
                    Tensor destinationValues = makeGpuBytes(options.gpu, batchRows * rowBytes);
                    destinationValues.memsetAsync(stream, 0);
                    stream.synchronize();

                    const uint32_t automaticRowsPerCta =
                    selectedRowsPerCta(batchRows, rowBytes);
                    for (uint32_t requestedRowsPerCta : options.rows_per_cta) {
                        const uint32_t actualRowsPerCta =
                            requestedRowsPerCta == 0 ? automaticRowsPerCta : requestedRowsPerCta;
                        const TimingStats timing = benchmarkDramReads(
                            stream, evictor, options.warmup, options.iterations, [&] {
                                if (requestedRowsPerCta == 0) {
                                    launchDeviceResidentRaggedMaterializationKernel(
                                        records, packedValues, sourceRows, recordSizeBytes,
                                        referenceOffsetBytes, storedValueCount, valueBytes,
                                        batchRows, destinationValues, destinationOffsets,
                                        indices, stream);
                                } else {
                                    launchDeviceResidentRaggedMaterializationKernelWithRowsPerCtaForBenchmark(
                                        records, packedValues, sourceRows, recordSizeBytes,
                                        referenceOffsetBytes, storedValueCount, valueBytes,
                                        batchRows, destinationValues, destinationOffsets,
                                        indices, requestedRowsPerCta, stream);
                                }
                            });

                        Result result;
                        result.kernel = "ragged";
                        result.variant = referenceLayout + "/v" + std::to_string(valueBytes);
                        result.batchRows = batchRows;
                        result.rowBytes = rowBytes;
                        result.groupingMode = requestedRowsPerCta == 0 ? "auto" : "forced";
                        result.autoRowsPerCta = automaticRowsPerCta;
                        result.rowsPerCta = actualRowsPerCta;
                        result.lanesPerRow = 256U / actualRowsPerCta;
                        // Useful source payload excludes record references, offsets,
                        // and row-index metadata so this remains directly comparable
                        // to the direct/named payload throughput column.
                        result.sourceReadBytes = batchRows * rowBytes;
                        result.destinationWriteBytes = batchRows * rowBytes;
                        result.timing = timing;
                        printResult(result);
                    }
                }
            }
        }
    }
}

void runWindowBenchmarks(const Options &options,
                         Stream &stream,
                         DramReadEvictor &evictor) {
    for (const std::string &mode : options.window_modes) {
        const bool materializeMask = mode == "mask";
        const std::vector<uint64_t> stepBytesSweep =
            materializeMask ? std::vector<uint64_t>{1} : options.window_step_bytes;
        for (uint64_t sourceStepBytes : stepBytesSweep) {
            for (uint64_t targetRowBytes : options.row_bytes) {
                if (targetRowBytes % sourceStepBytes != 0) continue;
                const uint64_t windowLength = targetRowBytes / sourceStepBytes;
                if (windowLength == 0 || windowLength > std::numeric_limits<uint32_t>::max()) continue;

                for (double validFraction : options.window_valid_fractions) {
                    uint64_t validSteps = static_cast<uint64_t>(std::llround(
                        validFraction * static_cast<double>(windowLength)));
                    validSteps = std::min<uint64_t>(validSteps, windowLength);
                    const uint64_t validStepBegin = (windowLength - validSteps) / 2;

                    for (uint64_t batchRows : options.batch_rows) {
                        const uint64_t sourceRows = std::max<uint64_t>(options.source_rows_floor, batchRows * 4);
                        const uint64_t sourceSpanBytes = std::max<uint64_t>(sourceStepBytes, validSteps * sourceStepBytes);
                        if (sourceRows > std::numeric_limits<uint64_t>::max() / sourceSpanBytes) {
                            throw std::overflow_error("Window benchmark source byte count overflow");
                        }
                        Tensor source = makeGpuBytes(options.gpu, sourceRows * sourceSpanBytes);
                        source.memsetAsync(stream, static_cast<int8_t>(0x49));

                        const std::vector<uint64_t> selectedSourceRows =
                            makeCollisionFreeRowIndices(batchRows, sourceRows);
                        std::vector<DeviceResidentWindowRowPlan32> plans(batchRows);
                        for (uint64_t row = 0; row < batchRows; ++row) {
                            plans[row].sourceOffsetBytes = selectedSourceRows[row] * sourceSpanBytes;
                            plans[row].validStepBegin = static_cast<uint32_t>(validStepBegin);
                            plans[row].validStepCount = static_cast<uint32_t>(validSteps);
                        }
                        Tensor planTensor = makeGpuPlans(options.gpu, plans, stream);

                        const uint64_t destinationRowBytes = materializeMask ? windowLength : targetRowBytes;
                        Tensor destination = makeGpuByteTensor(
                            options.gpu, {batchRows, destinationRowBytes});
                        destination.memsetAsync(stream, 0);
                        stream.synchronize();

                        DeviceResidentWindowMaterializationSpec spec;
                        spec.dataType = DataType::UINT8;
                        spec.windowLength = windowLength;
                        spec.sourceStepBytes = sourceStepBytes;
                        spec.padValue = 0.0;
                        spec.materializeMask = materializeMask;

                        const uint64_t selectorRowBytes = materializeMask ? windowLength : targetRowBytes;
                        const uint32_t automaticRowsPerCta =
                            selectedRowsPerCta(batchRows, selectorRowBytes);
                        for (uint32_t requestedRowsPerCta : options.rows_per_cta) {
                            const uint32_t actualRowsPerCta =
                                requestedRowsPerCta == 0 ? automaticRowsPerCta : requestedRowsPerCta;
                            const TimingStats timing = benchmarkDramReads(
                                stream, evictor, options.warmup, options.iterations, [&] {
                                    if (requestedRowsPerCta == 0) {
                                        launchDeviceResidentWindowMaterializationKernel(
                                            source, planTensor, batchRows, spec, destination, stream);
                                    } else {
                                        launchDeviceResidentWindowMaterializationKernelWithRowsPerCtaForBenchmark(
                                            source, planTensor, batchRows, spec, destination,
                                            requestedRowsPerCta, stream);
                                    }
                                });

                            Result result;
                            result.kernel = "window";
                            result.variant = mode;
                            result.batchRows = batchRows;
                            result.rowBytes = destinationRowBytes;
                            result.sourceStepBytes = sourceStepBytes;
                            result.windowLength = windowLength;
                            result.validFraction = static_cast<double>(validSteps) / static_cast<double>(windowLength);
                            result.groupingMode = requestedRowsPerCta == 0 ? "auto" : "forced";
                            result.autoRowsPerCta = automaticRowsPerCta;
                            result.rowsPerCta = actualRowsPerCta;
                            result.lanesPerRow = 256U / actualRowsPerCta;
                            result.sourceReadBytes = materializeMask ? 0 : batchRows * validSteps * sourceStepBytes;
                            result.destinationWriteBytes = batchRows * destinationRowBytes;
                            result.timing = timing;
                            printResult(result);
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
            throw std::runtime_error("CUDA reported zero L2 bytes; cannot construct the DRAM-read benchmark");
        }

        size_t freeBytes = 0;
        size_t totalBytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
        Stream stream(options.gpu);
        DramReadEvictor evictor(options, properties, stream);

        if (evictor.bytes() > static_cast<uint64_t>(freeBytes) / 2) {
            throw std::runtime_error(
                "L2 eviction buffer alone would consume more than half of currently free GPU memory; reduce --l2-evict-multiple or --min-evict-mib");
        }

        std::cerr << "Thor residency/materialization benchmark\n";
#ifdef THOR_DEBUG
        std::cerr << "WARNING: Thor was built with THOR_DEBUG. Use Release or RelWithDebInfo for performance decisions.\n";
#endif
        std::cerr << "GPU: " << properties.name << " (device " << options.gpu << ")\n"
                  << "GPU memory: " << std::fixed << std::setprecision(1)
                  << static_cast<double>(freeBytes) / static_cast<double>(MIB) << " MiB free / "
                  << static_cast<double>(totalBytes) / static_cast<double>(MIB) << " MiB total\n"
                  << "SMs: " << properties.multiProcessorCount << "\n"
                  << "L2: " << static_cast<double>(properties.l2CacheSize) / static_cast<double>(MIB) << " MiB\n"
                  << "Untimed device-read eviction: "
                  << static_cast<double>(evictor.bytes()) / static_cast<double>(MIB) << " MiB ("
                  << static_cast<double>(evictor.bytes()) / static_cast<double>(properties.l2CacheSize)
                  << "x L2) before every timed launch\n"
                  << "Warmup launches/case: " << options.warmup
                  << ", timed launches/case: " << options.iterations << "\n"
                  << "CSV results follow on stdout. source_payload_GBps is useful source payload only, not a hardware DRAM-byte counter.\n\n";

        printCsvHeader();
        if (contains(options.kernels, "direct")) {
            runDirectBenchmarks(options, stream, evictor);
        }
        if (contains(options.kernels, "named")) {
            runNamedGatherBenchmarks(options, stream, evictor);
        }
        if (contains(options.kernels, "ragged")) {
            runRaggedBenchmarks(options, stream, evictor);
        }
        if (contains(options.kernels, "window")) {
            runWindowBenchmarks(options, stream, evictor);
        }
        stream.synchronize();
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "thor_residency_materialization_benchmark: " << error.what() << '\n';
        return 1;
    }
}
