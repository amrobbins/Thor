#include "DeepLearning/Implementation/Data/Residency/DeviceResidentDirectMaterializationKernel.h"
#include "DeepLearning/Implementation/Data/Residency/DeviceResidentNamedGatherKernel.h"
#include "DeepLearning/Implementation/Data/Residency/DeviceResidentRaggedMaterializationKernel.h"
#include "DeepLearning/Implementation/Data/Residency/DeviceResidentWindowMaterializationKernel.h"
#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
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
#include <functional>
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
        } else if (arg == "--target-sample-us") {
            options.target_sample_us = parseDouble(requireValue(arg), arg);
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
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: thor_residency_materialization_benchmark [options]\n"
                << "  --gpu N\n"
                << "  --warmup N                         0 skips the untimed warmup sample of each case (default 5)\n"
                << "  --iterations N                     timed samples per case (default 20)\n"
                << "  --target-sample-us X               back-to-back launches per sample, about X us (default 200)\n"
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
                << "  --window-modes payload,mask\n\n"
                << "The benchmark warms each case, then performs an untimed device read over a buffer\n"
                << "larger than L2 before every timed sample. The timed target kernel therefore sees\n"
                << "DRAM-backed source reads without including cache eviction or initialization time.\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("Unknown argument: " + std::string(arg));
        }
    }

    if (options.gpu < 0) throw std::invalid_argument("--gpu must be non-negative");
    if (options.warmup < 0) throw std::invalid_argument("--warmup must be non-negative");
    if (options.iterations <= 0) throw std::invalid_argument("--iterations must be positive");
    if (!(options.target_sample_us > 0.0)) throw std::invalid_argument("--target-sample-us must be positive");
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

uint64_t splitMix64(uint64_t value) {
    uint64_t z = value + 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27U)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31U);
}

// rows distinct source rows in a scattered order: an affine permutation modulo sourceRows whose start and stride come
// from seed, so every index set of a sample visits different rows.
std::vector<uint64_t> scatteredRowIndices(uint64_t rows, uint64_t sourceRows, uint64_t seed) {
    if (rows > sourceRows) throw std::invalid_argument("A scattered index set needs sourceRows >= rows");
    std::vector<uint64_t> indices(rows);
    if (rows == 0 || sourceRows == 1) return indices;
    const uint64_t start = splitMix64(seed ^ 0x4d595df4d0f33173ULL) % sourceRows;
    uint64_t stride = splitMix64(seed ^ 0x94d049bb133111ebULL) % sourceRows;
    if (stride == 0) stride = 1;
    while (std::gcd(stride, sourceRows) != 1) stride = stride + 1 == sourceRows ? 1 : stride + 1;
    uint64_t current = start;
    for (uint64_t row = 0; row < rows; ++row) {
        indices[row] = current;
        current = current >= sourceRows - stride ? current - (sourceRows - stride) : current + stride;
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
    return (1.0 - alpha) * static_cast<double>(values[lo]) +
           alpha * static_cast<double>(values[hi]);
}

struct Result {
    std::string kernel;
    std::string variant;
    uint64_t batchRows = 0;
    uint64_t rowBytes = 0;
    uint64_t sourceStepBytes = 0;
    uint64_t windowLength = 0;
    double validFraction = 1.0;
    uint64_t sourceReadBytes = 0;
    uint64_t destinationWriteBytes = 0;
    // The distinct 32-byte DRAM sectors the source rows touch, in bytes (for ragged, the values' sectors; its record
    // reads are not counted). Rows that do not start or end on a sector boundary make DRAM move more than the payload.
    uint64_t sourceSectorBytes = 0;
    // Launches timed together per sample; the timing is per launch.
    uint32_t launchesPerSample = 1;
    TimingStats timing;
};

double usefulGBps(uint64_t bytes, double microseconds) {
    if (microseconds <= 0.0) return 0.0;
    return (static_cast<double>(bytes) / BYTES_PER_GB) / (microseconds * 1.0e-6);
}

void printCsvHeader() {
    std::cout
        << "kernel,variant,batch_rows,row_bytes,source_step_bytes,window_length,valid_fraction,"
        << "source_read_bytes,destination_write_bytes,"
        << "source_sector_bytes,launches_per_sample,p10_us,median_us,p90_us,host_us,source_payload_GBps,output_payload_GBps,dram_GBps\n";
}

void printResult(const Result &result) {
    std::cout << result.kernel << ','
              << result.variant << ','
              << result.batchRows << ','
              << result.rowBytes << ','
              << result.sourceStepBytes << ','
              << result.windowLength << ','
              << std::fixed << std::setprecision(3) << result.validFraction << ','
              << result.sourceReadBytes << ','
              << result.destinationWriteBytes << ','
              << result.sourceSectorBytes << ','
              << result.launchesPerSample << ','
              << std::setprecision(3) << result.timing.p10_us << ','
              << result.timing.median_us << ','
              << result.timing.p90_us << ','
              << result.timing.host_median_us << ','
              << std::setprecision(2) << usefulGBps(result.sourceReadBytes, result.timing.median_us) << ','
              << usefulGBps(result.destinationWriteBytes, result.timing.median_us) << ','
              // DRAM bytes: the 32-byte sectors the source reads touch (or the source bytes where not counted) and
              // the destination bytes.
              << usefulGBps((result.sourceSectorBytes != 0 ? result.sourceSectorBytes : result.sourceReadBytes) +
                                result.destinationWriteBytes,
                            result.timing.median_us)
              << '\n';
}

__global__ void launchFloorKernel() {}

// The rows a case selects from: rowBytes at offsetBytes of every strideBytes of storage. A named tensor's rows are back
// to back (the stride is the row, offset 0); a field of a compact record has the record size and the field's offset; a
// window reads the bytes of its valid steps. No storage: the kernel reads no source (a window mask).
struct RowSource {
    const Tensor *storage = nullptr;
    uint64_t rows = 0;
    uint64_t strideBytes = 0;
    uint64_t offsetBytes = 0;
    uint64_t rowBytes = 0;

    [[nodiscard]] const uint8_t *row(uint64_t index) const {
        return static_cast<const uint8_t *>(storage->getMemPtr()) + index * strideBytes + offsetBytes;
    }
};

// The distinct 32-byte DRAM sectors the rows of one index set touch, in bytes.
uint64_t sourceSectorBytes(const RowSource &source, std::vector<uint64_t> rowIndices) {
    if (source.storage == nullptr || source.rowBytes == 0) return 0;
    std::sort(rowIndices.begin(), rowIndices.end());
    const uint64_t base = reinterpret_cast<uintptr_t>(source.row(0)) % 32;
    uint64_t sectors = 0;
    uint64_t firstUncounted = 0;
    for (const uint64_t index : rowIndices) {
        const uint64_t first = std::max(firstUncounted, (base + index * source.strideBytes) / 32);
        const uint64_t last = (base + index * source.strideBytes + source.rowBytes - 1) / 32;
        if (last < first) continue;
        sectors += last - first + 1;
        firstUncounted = last + 1;
    }
    return sectors * 32;
}

// What tells a kernel which source row each batch row takes: wordsPerRow uint64 words per batch row, written from that
// row's source row index. The gathers take the index itself; a window takes its row plan.
struct RowSelection {
    uint32_t wordsPerRow = 1;
    std::function<void(uint64_t *words, uint64_t sourceRow)> write = [](uint64_t *words, uint64_t sourceRow) {
        *words = sourceRow;
    };
};

// One way of measuring a case:
//   Kernel       the kernel, over scattered source rows or, with consecutiveRows, over consecutive ones: the access
//                pattern of a copy, so the gap to Copy is the kernel's own cost and the gap to the scattered variant is
//                the scattered rows'. With selectionInL2 the sample's selection is read into L2 first, as when it was
//                just uploaded (the batch session copies it host to device right before the kernel); otherwise it is
//                in DRAM. The rows are in DRAM either way;
//   Copy         cudaMemcpyAsync of the output payload, contiguous, from the first of those consecutive rows;
//   Fill         cudaMemsetAsync of the output payload: what writing it costs with nothing to read;
//   LaunchFloor  an empty kernel.
struct CaseVariant {
    enum class Op { Kernel, Copy, Fill, LaunchFloor };
    std::string name;
    Op op = Op::Kernel;
    bool consecutiveRows = false;
    bool selectionInL2 = false;
};

std::vector<CaseVariant> gatherVariants() {
    return {{"gather"},
            {"gather_l2_indices", CaseVariant::Op::Kernel, false, true},
            {"gather_sequential", CaseVariant::Op::Kernel, true},
            {"copy", CaseVariant::Op::Copy},
            {"launch_floor", CaseVariant::Op::LaunchFloor}};
}

// One timed sample: the eviction pass and prepare (untimed), then launchesPerSample back-to-back launches between two
// events; the per-launch time is the sample divided by the launch count. The CPU time of the launch loop, also per
// launch, is recorded beside it. warmupSamples untimed samples run first.
template <typename Prepare, typename Fn>
TimingStats benchmarkLaunchSamples(Stream &stream,
                                   DramReadEvictor &evictor,
                                   int warmupSamples,
                                   int samples,
                                   uint32_t launchesPerSample,
                                   Prepare &&prepare,
                                   Fn &&launch) {
    for (int i = 0; i < warmupSamples; ++i) {
        evictor.run(stream);
        prepare();
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
        prepare();
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

// One case, measured as each of its variants, the same way. A sample is launchesPerSample back-to-back launches (about
// --target-sample-us), each with its own selection over a source of at least 4x L2 and its own destination from a pool
// of up to 2x L2, so its reads are DRAM-cold and its writes do not land on lines the previous launch left in L2.
// rowBytes is a destination row; launch(destination, selection) launches the kernel.
template <typename Launch>
void runRowCase(const Options &options,
                const cudaDeviceProp &properties,
                Stream &stream,
                DramReadEvictor &evictor,
                const RowSource &source,
                uint64_t batchRows,
                uint64_t rowBytes,
                const RowSelection &selection,
                const std::vector<CaseVariant> &variants,
                Result result,
                Launch &&launch) {
    constexpr uint32_t kMaxLaunchesPerSample = 256;
    const uint64_t l2Bytes = static_cast<uint64_t>(properties.l2CacheSize);
    const TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, options.gpu);
    const uint64_t payloadBytes = batchRows * rowBytes;
    // An estimate of one launch (2 us of launch, both directions at 1.4 TB/s) sizes the sample.
    const double estimatedUs = std::max(2.0, 2.0 * static_cast<double>(payloadBytes) / 1.4e6);
    const uint32_t launchesPerSample = static_cast<uint32_t>(
        std::clamp(std::ceil(options.target_sample_us / estimatedUs), 1.0, static_cast<double>(kMaxLaunchesPerSample)));
    const uint32_t destinations = static_cast<uint32_t>(
        std::min<uint64_t>(launchesPerSample, std::max<uint64_t>(1, ceilDiv(2 * l2Bytes, payloadBytes))));

    std::vector<Tensor> destination;
    for (uint32_t d = 0; d < destinations; ++d) {
        destination.emplace_back(gpuPlacement, TensorDescriptor(DataType::UINT8, {batchRows, rowBytes}));
        destination.back().memsetAsync(stream, 0);
    }

    // One selection per launch of a sample, over scattered rows and over consecutive rows, in one tensor each.
    const uint64_t wordsPerLaunch = batchRows * selection.wordsPerRow;
    std::vector<uint64_t> scattered(static_cast<uint64_t>(launchesPerSample) * wordsPerLaunch);
    std::vector<uint64_t> consecutive(scattered.size());
    std::vector<uint64_t> consecutiveStart(launchesPerSample, 0);
    uint64_t scatteredSectorBytes = 0;
    uint64_t consecutiveSectorBytes = 0;
    uint64_t copySectorBytes = 0;
    for (uint32_t k = 0; k < launchesPerSample; ++k) {
        std::vector<uint64_t> scatteredRows(batchRows, 0);
        std::vector<uint64_t> consecutiveRows(batchRows, 0);
        if (source.storage != nullptr) {
            scatteredRows = scatteredRowIndices(batchRows, source.rows, source.rows * 977 + k);
            consecutiveStart[k] = splitMix64(source.rows * 131 + k) % (source.rows - batchRows + 1);
            std::iota(consecutiveRows.begin(), consecutiveRows.end(), consecutiveStart[k]);
            scatteredSectorBytes += sourceSectorBytes(source, scatteredRows);
            consecutiveSectorBytes += sourceSectorBytes(source, consecutiveRows);
            copySectorBytes +=
                roundUp(reinterpret_cast<uintptr_t>(source.row(consecutiveStart[k])) % 32 + payloadBytes, 32);
        }
        for (uint64_t row = 0; row < batchRows; ++row) {
            const uint64_t word = static_cast<uint64_t>(k) * wordsPerLaunch + row * selection.wordsPerRow;
            selection.write(scattered.data() + word, scatteredRows[row]);
            selection.write(consecutive.data() + word, consecutiveRows[row]);
        }
    }
    Tensor scatteredSelection = makeGpuTensor<uint64_t>(options.gpu, {scattered.size()}, scattered, stream);
    Tensor consecutiveSelection = makeGpuTensor<uint64_t>(options.gpu, {consecutive.size()}, consecutive, stream);
    Tensor sink(gpuPlacement, TensorDescriptor(DataType::UINT64, {1}));
    stream.synchronize();

    for (const CaseVariant &variant : variants) {
        const Tensor &variantSelection = variant.consecutiveRows ? consecutiveSelection : scatteredSelection;
        auto prepare = [&] {
            if (!variant.selectionInL2) return;
            // Read every selection of the sample (at most 256 launches of them) into L2 after the eviction.
            readForL2EvictionKernel<<<std::max(1, properties.multiProcessorCount * 8), 256, 0, stream.getStream()>>>(
                reinterpret_cast<const uint4 *>(variantSelection.getMemPtr()),
                variantSelection.getArraySizeInBytes() / sizeof(uint4),
                reinterpret_cast<unsigned long long *>(sink.getMemPtr<uint64_t>()));
            CUDA_CHECK(cudaGetLastError());
        };
        const TimingStats timing = benchmarkLaunchSamples(
            stream, evictor, options.warmup == 0 ? 0 : 1, options.iterations, launchesPerSample, prepare, [&](uint32_t k) {
                Tensor &out = destination[k % destinations];
                switch (variant.op) {
                    case CaseVariant::Op::Kernel:
                        launch(out, TensorSpan(variantSelection, static_cast<uint64_t>(k) * wordsPerLaunch, wordsPerLaunch));
                        break;
                    case CaseVariant::Op::Copy:
                        CUDA_CHECK(cudaMemcpyAsync(out.getMemPtr(), source.row(consecutiveStart[k]), payloadBytes,
                                                   cudaMemcpyDeviceToDevice, stream.getStream()));
                        break;
                    case CaseVariant::Op::Fill:
                        CUDA_CHECK(cudaMemsetAsync(out.getMemPtr(), 1, payloadBytes, stream.getStream()));
                        break;
                    case CaseVariant::Op::LaunchFloor:
                        launchFloorKernel<<<1, 32, 0, stream.getStream()>>>();
                        CUDA_CHECK(cudaGetLastError());
                        break;
                }
            });

        result.variant = variant.name;
        result.batchRows = batchRows;
        result.rowBytes = rowBytes;
        result.sourceReadBytes = 0;
        result.sourceSectorBytes = 0;
        if (variant.op == CaseVariant::Op::Kernel) {
            result.sourceReadBytes = batchRows * source.rowBytes;
            result.sourceSectorBytes =
                (variant.consecutiveRows ? consecutiveSectorBytes : scatteredSectorBytes) / launchesPerSample;
        } else if (variant.op == CaseVariant::Op::Copy) {
            result.sourceReadBytes = payloadBytes;
            result.sourceSectorBytes = copySectorBytes / launchesPerSample;
        }
        result.destinationWriteBytes = variant.op == CaseVariant::Op::LaunchFloor ? 0 : payloadBytes;
        result.launchesPerSample = launchesPerSample;
        result.timing = timing;
        printResult(result);
    }
}

// Rows of a source of at least 4x L2 (and 4x the batch), so a launch's rows are not left in L2 by the one before.
uint64_t coldSourceRows(const Options &options, const cudaDeviceProp &properties, uint64_t batchRows, uint64_t strideBytes) {
    const uint64_t rows = std::max(
        {options.source_rows_floor, batchRows * 4, ceilDiv(4 * static_cast<uint64_t>(properties.l2CacheSize), strideBytes)});
    if (rows > std::numeric_limits<uint64_t>::max() / strideBytes) {
        throw std::overflow_error("Residency benchmark source byte count overflow");
    }
    return rows;
}

void runNamedGatherBenchmarks(const Options &options,
                              const cudaDeviceProp &properties,
                              Stream &stream,
                              DramReadEvictor &evictor) {
    const TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, options.gpu);
    for (uint64_t rowBytes : options.row_bytes) {
        for (uint64_t batchRows : options.batch_rows) {
            const uint64_t sourceRows = coldSourceRows(options, properties, batchRows, rowBytes);
            Tensor source(gpuPlacement, TensorDescriptor(DataType::UINT8, {sourceRows, rowBytes}));
            source.memsetAsync(stream, static_cast<int8_t>(0x67));

            Result result;
            result.kernel = "named";
            runRowCase(options, properties, stream, evictor, RowSource{&source, sourceRows, rowBytes, 0, rowBytes}, batchRows,
                       rowBytes, RowSelection{}, gatherVariants(), result, [&](Tensor &out, const TensorSpan &indices) {
                           launchDeviceResidentNamedGatherKernel(source, out, indices, batchRows, stream);
                       });
        }
    }
}

// A field of compact records. aligned32: the field starts its record and the record is padded to 32 bytes, so every
// field starts on a 32-byte boundary. packed: 3 bytes before the field and 5 after, so no field is aligned with its
// destination row beyond what the row length happens to give.
void runDirectBenchmarks(const Options &options,
                         const cudaDeviceProp &properties,
                         Stream &stream,
                         DramReadEvictor &evictor) {
    for (const std::string &layout : options.direct_layouts) {
        for (uint64_t rowBytes : options.row_bytes) {
            if (rowBytes == 0) continue;
            const uint64_t fieldOffsetBytes = layout == "aligned32" ? 0 : 3;
            const uint64_t recordSizeBytes =
                layout == "aligned32" ? roundUp(fieldOffsetBytes + rowBytes, 32) : fieldOffsetBytes + rowBytes + 5;
            for (uint64_t batchRows : options.batch_rows) {
                const uint64_t sourceRows = coldSourceRows(options, properties, batchRows, recordSizeBytes);
                Tensor records = makeGpuBytes(options.gpu, sourceRows * recordSizeBytes);
                records.memsetAsync(stream, static_cast<int8_t>(0x3c));

                Result result;
                result.kernel = "direct_" + layout;
                runRowCase(options, properties, stream, evictor,
                           RowSource{&records, sourceRows, recordSizeBytes, fieldOffsetBytes, rowBytes}, batchRows, rowBytes,
                           RowSelection{}, gatherVariants(), result, [&](Tensor &out, const TensorSpan &indices) {
                               launchDeviceResidentDirectMaterializationKernel(
                                   records, sourceRows, recordSizeBytes, fieldOffsetBytes, rowBytes, batchRows, out, indices,
                                   stream);
                           });
            }
        }
    }
}

// A ragged field: record i holds, at the reference offset, the index of its first value in the packed value store and
// its value count; a batch row copies its record's values to the place the batch's destination offsets give it.
// Every row here is the same length, so the offsets are regular and rowBytes is that length in bytes. aligned8: the
// reference is on an 8-byte boundary of its record; unaligned: it is 3 bytes into it. The source rows are the values;
// the records, 32 bytes per source row, are read on the way to them and sized into the cold source with them. The
// destination offsets are one tensor per case: in L2 after a sample's first launch, as when the batch session has
// just uploaded them. copy moves a batch of consecutive rows' values.
void runRaggedBenchmarks(const Options &options,
                         const cudaDeviceProp &properties,
                         Stream &stream,
                         DramReadEvictor &evictor) {
    constexpr uint64_t recordSizeBytes = 32;
    const std::vector<CaseVariant> variants{{"ragged"},
                                            {"ragged_l2_indices", CaseVariant::Op::Kernel, false, true},
                                            {"ragged_sequential", CaseVariant::Op::Kernel, true},
                                            {"copy", CaseVariant::Op::Copy},
                                            {"launch_floor", CaseVariant::Op::LaunchFloor}};
    const uint64_t largestBatchRows = *std::max_element(options.batch_rows.begin(), options.batch_rows.end());

    for (const std::string &referenceLayout : options.ragged_reference_layouts) {
        const uint64_t referenceOffsetBytes = referenceLayout == "aligned8" ? 8 : 3;
        for (uint64_t valueBytes : options.ragged_value_bytes) {
            for (uint64_t rowBytes : options.row_bytes) {
                // A row is whole values: the 1-byte values run every row length, wider ones those they divide.
                if (rowBytes == 0 || rowBytes % valueBytes != 0) continue;
                const uint64_t valuesPerRow = rowBytes / valueBytes;

                // One store for every batch size of this row length: records and values together at least 4x L2.
                const uint64_t sourceRows = coldSourceRows(options, properties, largestBatchRows, recordSizeBytes + rowBytes);
                const uint64_t storedValueCount = sourceRows * valuesPerRow;
                std::vector<uint8_t> recordBytes(sourceRows * recordSizeBytes, 0);
                for (uint64_t sourceRow = 0; sourceRow < sourceRows; ++sourceRow) {
                    const uint64_t reference[2] = {sourceRow * valuesPerRow, valuesPerRow};
                    std::memcpy(recordBytes.data() + sourceRow * recordSizeBytes + referenceOffsetBytes, reference, sizeof(reference));
                }
                Tensor records = makeGpuTensor<uint8_t>(options.gpu, {recordBytes.size()}, recordBytes, stream);
                Tensor packedValues = makeGpuBytes(options.gpu, storedValueCount * valueBytes);
                packedValues.memsetAsync(stream, static_cast<int8_t>(0x2d));

                for (uint64_t batchRows : options.batch_rows) {
                    if (batchRows * valuesPerRow > std::numeric_limits<uint32_t>::max()) {
                        throw std::runtime_error(
                            "Ragged benchmark case exceeds UINT32 destination offsets; reduce --batch-rows or --row-bytes");
                    }
                    std::vector<uint32_t> offsets(batchRows + 1, 0);
                    for (uint64_t row = 0; row <= batchRows; ++row) offsets[row] = static_cast<uint32_t>(row * valuesPerRow);
                    Tensor destinationOffsets = makeGpuTensor<uint32_t>(options.gpu, {batchRows + 1}, offsets, stream);

                    Result result;
                    result.kernel = "ragged_" + referenceLayout + "_v" + std::to_string(valueBytes);
                    runRowCase(options, properties, stream, evictor, RowSource{&packedValues, sourceRows, rowBytes, 0, rowBytes},
                               batchRows, rowBytes, RowSelection{}, variants, result, [&](Tensor &out, const TensorSpan &indices) {
                                   launchDeviceResidentRaggedMaterializationKernel(
                                       records, packedValues, sourceRows, recordSizeBytes, referenceOffsetBytes, storedValueCount,
                                       valueBytes, batchRows, out, destinationOffsets, indices, stream);
                               });
                }
            }
        }
    }
}

// Windows of a resident sequence source. A row is windowLength steps of sourceStepBytes: the middle valid_fraction of
// its steps comes from the source and the steps around it are padding. payload materializes the row; mask materializes
// one byte per step, 1 where the step is valid, and reads no source. Every selected window is its own span of the
// source, so scattered windows share no bytes and consecutive ones are back to back. copy moves a whole output payload
// and fill writes one: at valid fractions below 1 a payload launch reads less than copy and more than fill.
void runWindowBenchmarks(const Options &options,
                         const cudaDeviceProp &properties,
                         Stream &stream,
                         DramReadEvictor &evictor) {
    static_assert(sizeof(DeviceResidentWindowRowPlan32) == 2 * sizeof(uint64_t));
    for (const std::string &mode : options.window_modes) {
        const bool materializeMask = mode == "mask";
        const std::vector<uint64_t> stepBytesSweep =
            materializeMask ? std::vector<uint64_t>{1} : options.window_step_bytes;
        const std::vector<CaseVariant> variants =
            materializeMask ? std::vector<CaseVariant>{{"mask"},
                                                   {"mask_l2_plans", CaseVariant::Op::Kernel, false, true},
                                                   {"fill", CaseVariant::Op::Fill},
                                                   {"launch_floor", CaseVariant::Op::LaunchFloor}}
                            : std::vector<CaseVariant>{{"window"},
                                                   {"window_l2_plans", CaseVariant::Op::Kernel, false, true},
                                                   {"window_sequential", CaseVariant::Op::Kernel, true},
                                                   {"copy", CaseVariant::Op::Copy},
                                                   {"fill", CaseVariant::Op::Fill},
                                                   {"launch_floor", CaseVariant::Op::LaunchFloor}};
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
                    const uint64_t validBytes = validSteps * sourceStepBytes;
                    const uint64_t sourceSpanBytes = std::max<uint64_t>(sourceStepBytes, validBytes);
                    const uint64_t destinationRowBytes = materializeMask ? windowLength : targetRowBytes;

                    for (uint64_t batchRows : options.batch_rows) {
                        Tensor source;
                        RowSource rowSource;
                        if (!materializeMask) {
                            const uint64_t sourceRows = coldSourceRows(options, properties, batchRows, sourceSpanBytes);
                            // The copy reference reads a whole output payload from the first of any batchRows spans:
                            // room for it past the last span.
                            source = makeGpuBytes(
                                options.gpu, sourceRows * sourceSpanBytes + batchRows * (destinationRowBytes - sourceSpanBytes));
                            source.memsetAsync(stream, static_cast<int8_t>(0x49));
                            rowSource = RowSource{&source, sourceRows, sourceSpanBytes, 0, validBytes};
                        }

                        RowSelection plans;
                        plans.wordsPerRow = 2;
                        plans.write = [=](uint64_t *words, uint64_t sourceRow) {
                            const DeviceResidentWindowRowPlan32 plan{.sourceOffsetBytes = sourceRow * sourceSpanBytes,
                                                                     .validStepBegin = static_cast<uint32_t>(validStepBegin),
                                                                     .validStepCount = static_cast<uint32_t>(validSteps)};
                            std::memcpy(words, &plan, sizeof(plan));
                        };

                        DeviceResidentWindowMaterializationSpec spec;
                        spec.dataType = DataType::UINT8;
                        spec.windowLength = windowLength;
                        spec.sourceStepBytes = sourceStepBytes;
                        spec.padValue = 0.0;
                        spec.materializeMask = materializeMask;
                        // Every plan's source offset is a multiple of the span.
                        const uint64_t spanBits = sourceSpanBytes | 16;
                        spec.sourceOffsetAlignmentBytes = spanBits & (~spanBits + 1);

                        Result result;
                        result.kernel = "window_" + mode;
                        result.sourceStepBytes = sourceStepBytes;
                        result.windowLength = windowLength;
                        result.validFraction = static_cast<double>(validSteps) / static_cast<double>(windowLength);
                        runRowCase(options, properties, stream, evictor, rowSource, batchRows, destinationRowBytes, plans,
                                   variants, result, [&](Tensor &out, const TensorSpan &rowPlans) {
                                       launchDeviceResidentWindowMaterializationKernel(
                                           source, rowPlans, batchRows, spec, out, stream);
                                   });
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
                  << "x L2) before every timed sample\n"
                  << "Timed samples/case: " << options.iterations << "\n"
                  << "CSV results follow on stdout. source_payload_GBps is useful source payload only, not a hardware DRAM-byte counter.\n\n";

        printCsvHeader();
        if (contains(options.kernels, "direct")) {
            runDirectBenchmarks(options, properties, stream, evictor);
        }
        if (contains(options.kernels, "named")) {
            runNamedGatherBenchmarks(options, properties, stream, evictor);
        }
        if (contains(options.kernels, "ragged")) {
            runRaggedBenchmarks(options, properties, stream, evictor);
        }
        if (contains(options.kernels, "window")) {
            runWindowBenchmarks(options, properties, stream, evictor);
        }
        stream.synchronize();
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "thor_residency_materialization_benchmark: " << error.what() << '\n';
        return 1;
    }
}
