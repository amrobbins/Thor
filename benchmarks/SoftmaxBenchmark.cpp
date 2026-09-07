#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/CudnnHelper.h"
#include "Utilities/Common/Event.h"
#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/Expression/CudaHelpers.h"
#include "Utilities/TensorOperations/DeepLearning/ExpressionDenseSoftmax.h"

#include <cuda_runtime.h>
#include <cudnn.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <utility>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace ThorImplementation;

namespace {

constexpr uint64_t MIB = 1024ULL * 1024ULL;
constexpr double BYTES_PER_GB = 1.0e9;

struct Options {
    int gpu = 0;
    int warmup = 3;
    int iterations = 12;
    double l2_multiple = 2.0;
    uint64_t min_tensor_bytes = 64ULL * MIB;
    uint32_t buffer_slots = 2;
    std::vector<uint64_t> channels{128, 1024, 4096};
};

struct LegacyCudnnSoftmax {
    cudnnTensorDescriptor_t descriptor = nullptr;

    LegacyCudnnSoftmax(uint64_t outer, uint64_t channels, DataType dtype) {
        if (outer > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
            channels > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument("Legacy cuDNN Softmax benchmark dimensions exceed cuDNN int range.");
        }
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&descriptor));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(descriptor,
                                               CUDNN_TENSOR_NCHW,
                                               CudnnHelper::getCudnnDataType(dtype),
                                               static_cast<int>(outer),
                                               static_cast<int>(channels),
                                               1,
                                               1));
    }

    LegacyCudnnSoftmax(const LegacyCudnnSoftmax&) = delete;
    LegacyCudnnSoftmax& operator=(const LegacyCudnnSoftmax&) = delete;

    ~LegacyCudnnSoftmax() {
        if (descriptor != nullptr) {
            (void)cudnnDestroyTensorDescriptor(descriptor);
        }
    }

    void run(const Tensor& x, Tensor& y, Stream& stream) const {
        static constexpr float alpha = 1.0f;
        static constexpr float beta = 0.0f;
        CUDNN_CHECK(cudnnSoftmaxForward(stream.getCudnnHandle(),
                                        CUDNN_SOFTMAX_ACCURATE,
                                        CUDNN_SOFTMAX_MODE_CHANNEL,
                                        &alpha,
                                        descriptor,
                                        x.getMemPtr<void>(),
                                        &beta,
                                        descriptor,
                                        y.getMemPtr<void>()));
    }
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

std::vector<uint64_t> parseChannels(std::string_view text) {
    std::vector<uint64_t> result;
    std::stringstream ss{std::string(text)};
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) continue;
        const uint64_t value = parseUnsigned(token, "--channels");
        if (value == 0) throw std::invalid_argument("--channels values must be non-zero");
        result.push_back(value);
    }
    if (result.empty()) throw std::invalid_argument("--channels must contain at least one value");
    return result;
}

Options parseOptions(int argc, char** argv) {
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
        } else if (arg == "--l2-multiple") {
            options.l2_multiple = parseDouble(requireValue(arg), arg);
        } else if (arg == "--min-tensor-mib") {
            options.min_tensor_bytes = parseUnsigned(requireValue(arg), arg) * MIB;
        } else if (arg == "--buffer-slots") {
            options.buffer_slots = static_cast<uint32_t>(parseUnsigned(requireValue(arg), arg));
        } else if (arg == "--channels") {
            options.channels = parseChannels(requireValue(arg));
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: thor_softmax_benchmark [options]\n"
                << "  --gpu N                 GPU number (default 0)\n"
                << "  --warmup N              warmup iterations per implementation (default 3)\n"
                << "  --iterations N          timed iterations per implementation (default 12)\n"
                << "  --l2-multiple X         minimum bytes in each input tensor / L2 bytes (default 2.0)\n"
                << "  --min-tensor-mib N      absolute minimum input tensor size (default 64)\n"
                << "  --buffer-slots N        rotate N input/output pairs (default 2)\n"
                << "  --channels a,b,c        final-axis widths (default 128,1024,4096)\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("Unknown argument: " + std::string(arg));
        }
    }
    if (options.gpu < 0) throw std::invalid_argument("--gpu must be non-negative");
    if (options.iterations <= 0) throw std::invalid_argument("--iterations must be positive");
    if (options.warmup < 0) throw std::invalid_argument("--warmup must be non-negative");
    if (options.l2_multiple < 1.25) {
        throw std::invalid_argument("--l2-multiple must be >= 1.25 so each tensor exceeds L2 capacity");
    }
    if (options.buffer_slots == 0) throw std::invalid_argument("--buffer-slots must be positive");
    return options;
}

const char* dtypeName(DataType dtype) {
    switch (dtype) {
        case DataType::FP32:
            return "fp32";
        case DataType::FP16:
            return "fp16";
        case DataType::BF16:
            return "bf16";
        default:
            return "unknown";
    }
}

uint64_t bytesPerElement(DataType dtype) {
    return static_cast<uint64_t>(TensorDescriptor::getElementSizeInBytes(dtype));
}

uint64_t ceilDiv(uint64_t numerator, uint64_t denominator) {
    return numerator / denominator + static_cast<uint64_t>(numerator % denominator != 0);
}

Tensor makeGpuTensor(int gpu, DataType dtype, uint64_t outer, uint64_t channels) {
    return Tensor(TensorPlacement(TensorPlacement::MemDevices::GPU, gpu), TensorDescriptor(dtype, {outer, channels}));
}

void coldScrub(void* scrub, uint64_t scrub_bytes, Stream& stream, uint8_t pattern) {
    CUDA_CHECK(cudaMemsetAsync(scrub, pattern, static_cast<size_t>(scrub_bytes), stream));
}

double median(std::vector<float> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    if ((values.size() & 1U) != 0U) return values[middle];
    return 0.5 * (static_cast<double>(values[middle - 1]) + static_cast<double>(values[middle]));
}

template <typename Fn>
double benchmarkColdGpu(Stream& stream,
                        void* scrub,
                        uint64_t scrub_bytes,
                        int warmup,
                        int iterations,
                        uint32_t slots,
                        Fn&& fn) {
    for (int i = 0; i < warmup; ++i) {
        coldScrub(scrub, scrub_bytes, stream, static_cast<uint8_t>(0x31 + (i & 0x3f)));
        fn(static_cast<uint32_t>(i) % slots);
        stream.synchronize();
    }

    Event start(stream.getGpuNum(), /*enableTiming=*/true);
    Event stop(stream.getGpuNum(), /*enableTiming=*/true);
    std::vector<float> milliseconds;
    milliseconds.reserve(iterations);
    for (int i = 0; i < iterations; ++i) {
        // The scrub is intentionally outside the timed interval. It is ordered
        // immediately before the start event on the same stream so the prior
        // benchmark tensor cannot remain the dominant L2 resident working set.
        coldScrub(scrub, scrub_bytes, stream, static_cast<uint8_t>(0x71 + (i & 0x3f)));
        start.record(stream);
        fn(static_cast<uint32_t>(i) % slots);
        stop.record(stream);
        milliseconds.push_back(stop.synchronizeAndReportElapsedTimeInMilliseconds(start));
    }
    return median(std::move(milliseconds));
}

struct Result {
    DataType dtype;
    uint64_t channels = 0;
    uint64_t outer = 0;
    uint64_t elements = 0;
    uint64_t tensor_bytes = 0;
    double legacy_ms = 0.0;
    double expression_ms = 0.0;
};

Result runCase(const Options& options,
               Stream& stream,
               void* scrub,
               uint64_t scrub_bytes,
               uint64_t target_tensor_bytes,
               DataType dtype,
               uint64_t channels) {
    const uint64_t element_bytes = bytesPerElement(dtype);
    const uint64_t target_elements = ceilDiv(target_tensor_bytes, element_bytes);
    const uint64_t outer = std::max<uint64_t>(1, ceilDiv(target_elements, channels));
    const uint64_t elements = outer * channels;
    const uint64_t tensor_bytes = elements * element_bytes;

    std::vector<Tensor> xs;
    std::vector<Tensor> ys_legacy;
    std::vector<Tensor> ys_expression;
    xs.reserve(options.buffer_slots);
    ys_legacy.reserve(options.buffer_slots);
    ys_expression.reserve(options.buffer_slots);
    for (uint32_t i = 0; i < options.buffer_slots; ++i) {
        xs.push_back(makeGpuTensor(options.gpu, dtype, outer, channels));
        ys_legacy.push_back(makeGpuTensor(options.gpu, dtype, outer, channels));
        ys_expression.push_back(makeGpuTensor(options.gpu, dtype, outer, channels));
        CUDA_CHECK(cudaMemsetAsync(xs.back().getMemPtr<void>(), 0, tensor_bytes, stream));
        CUDA_CHECK(cudaMemsetAsync(ys_legacy.back().getMemPtr<void>(), 0, tensor_bytes, stream));
        CUDA_CHECK(cudaMemsetAsync(ys_expression.back().getMemPtr<void>(), 0, tensor_bytes, stream));
    }
    stream.synchronize();

    LegacyCudnnSoftmax legacy(outer, channels, dtype);

    ExpressionDenseSoftmaxDescriptor expression_descriptor;
    expression_descriptor.outerSize = outer;
    expression_descriptor.channelCount = channels;
    expression_descriptor.inputDataType = dtype;
    expression_descriptor.outputDataType = std::nullopt;
    expression_descriptor.computeDataType = DataType::FP32;
    expression_descriptor.kind = ExpressionDenseSoftmaxKind::Softmax;
    expression_descriptor.debugName = "softmax_benchmark";
    ExpressionDenseSoftmaxPlan expression_plan =
        ExpressionDenseSoftmax::instance().prepareForward(expression_descriptor, options.gpu);

    const double legacy_ms = benchmarkColdGpu(stream,
                                               scrub,
                                               scrub_bytes,
                                               options.warmup,
                                               options.iterations,
                                               options.buffer_slots,
                                               [&](uint32_t slot) { legacy.run(xs[slot], ys_legacy[slot], stream); });

    const double expression_ms = benchmarkColdGpu(
        stream,
        scrub,
        scrub_bytes,
        options.warmup,
        options.iterations,
        options.buffer_slots,
        [&](uint32_t slot) {
            ExpressionDenseSoftmax::instance().forward(
                expression_plan, {.x = xs[slot], .y = ys_expression[slot]}, stream);
        });

    return {.dtype = dtype,
            .channels = channels,
            .outer = outer,
            .elements = elements,
            .tensor_bytes = tensor_bytes,
            .legacy_ms = legacy_ms,
            .expression_ms = expression_ms};
}

void printResult(const Result& result) {
    const double tensor_mib = static_cast<double>(result.tensor_bytes) / static_cast<double>(MIB);
    const double ratio = result.expression_ms / result.legacy_ms;
    const double legacy_min_gbps =
        (2.0 * static_cast<double>(result.tensor_bytes) / BYTES_PER_GB) / (result.legacy_ms / 1000.0);

    // Lower bound for the staged Expression implementation assuming each FP32
    // full-size intermediate is written once and subsequently read twice, and
    // row-reduction values have perfect cache reuse. Actual DRAM traffic may be
    // higher; use Nsight Compute if exact hardware DRAM byte counters are needed.
    const double fp32_full_bytes = static_cast<double>(result.elements) * 4.0;
    const double row_bytes = static_cast<double>(result.outer) * 4.0;
    const double expression_min_bytes =
        3.0 * static_cast<double>(result.tensor_bytes) + 3.0 * fp32_full_bytes + 4.0 * row_bytes;
    const double expression_min_gbps =
        (expression_min_bytes / BYTES_PER_GB) / (result.expression_ms / 1000.0);

    std::cout << std::left << std::setw(6) << dtypeName(result.dtype) << std::right << std::setw(8) << result.channels
              << std::setw(11) << result.outer << std::setw(12) << std::fixed << std::setprecision(1) << tensor_mib
              << std::setw(13) << std::setprecision(3) << result.legacy_ms << std::setw(13) << result.expression_ms
              << std::setw(11) << std::setprecision(2) << ratio << std::setw(15) << std::setprecision(1) << legacy_min_gbps
              << std::setw(15) << expression_min_gbps << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parseOptions(argc, argv);
        ScopedGpu scoped_gpu(options.gpu);

        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, options.gpu));
        const uint64_t l2_bytes = static_cast<uint64_t>(properties.l2CacheSize);
        if (l2_bytes == 0) throw std::runtime_error("CUDA reported zero L2 cache bytes; cannot enforce cold-memory benchmark.");

        const uint64_t target_tensor_bytes = std::max<uint64_t>(
            options.min_tensor_bytes, static_cast<uint64_t>(std::ceil(options.l2_multiple * static_cast<double>(l2_bytes))));
        const uint64_t scrub_bytes = std::max<uint64_t>(2 * l2_bytes, target_tensor_bytes);

        size_t free_bytes = 0;
        size_t total_bytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        // FP16/BF16 are the worst case for Expression scratch because the full
        // FP32 exp intermediate is twice the input bytes. Keep a conservative
        // memory headroom estimate before beginning the benchmark.
        const uint64_t estimated_peak_bytes =
            scrub_bytes + static_cast<uint64_t>(options.buffer_slots) * 3ULL * target_tensor_bytes + 3ULL * target_tensor_bytes;
        if (estimated_peak_bytes > static_cast<uint64_t>(free_bytes) * 8ULL / 10ULL) {
            throw std::runtime_error("Softmax benchmark cold-working-set configuration would consume more than 80% of free GPU memory. "
                                     "Reduce --l2-multiple or --buffer-slots.");
        }

        void* scrub = nullptr;
        CUDA_CHECK(cudaMalloc(&scrub, static_cast<size_t>(scrub_bytes)));
        struct ScrubGuard {
            void* ptr;
            ~ScrubGuard() {
                if (ptr != nullptr) (void)cudaFree(ptr);
            }
        } scrub_guard{scrub};

        Stream stream(options.gpu);

        std::cout << "Thor dense Softmax benchmark (forward only)\n";
#ifdef THOR_DEBUG
        std::cout << "WARNING: Thor was built with THOR_DEBUG. Use a Release or RelWithDebInfo build for performance decisions.\n";
#endif
        std::cout << "GPU: " << properties.name << " (device " << options.gpu << ")\n"
                  << "GPU memory: " << static_cast<double>(free_bytes) / static_cast<double>(MIB) << " MiB free / "
                  << static_cast<double>(total_bytes) / static_cast<double>(MIB) << " MiB total\n"
                  << "L2: " << std::fixed << std::setprecision(1)
                  << static_cast<double>(l2_bytes) / static_cast<double>(MIB) << " MiB\n"
                  << "Input tensor target: " << static_cast<double>(target_tensor_bytes) / static_cast<double>(MIB)
                  << " MiB (" << static_cast<double>(target_tensor_bytes) / static_cast<double>(l2_bytes) << "x L2)\n"
                  << "Cache scrub: " << static_cast<double>(scrub_bytes) / static_cast<double>(MIB)
                  << " MiB, ordered immediately before every timed invocation and excluded from event timing\n"
                  << "Buffer slots: " << options.buffer_slots << ", warmup: " << options.warmup
                  << ", timed iterations: " << options.iterations << "\n\n"
                  << "Each individual input tensor exceeds L2, and prior benchmark data is scrubbed before each timed run.\n"
                  << "Thus the timing is intentionally streaming/DRAM-oriented rather than a cache-hot microbenchmark.\n"
                  << "Bandwidth columns are lower-bound algorithmic traffic estimates, not hardware counters.\n\n";

        std::cout << std::left << std::setw(6) << "dtype" << std::right << std::setw(8) << "C" << std::setw(11) << "rows"
                  << std::setw(12) << "tensorMiB" << std::setw(13) << "cudnn_ms" << std::setw(13) << "expr_ms"
                  << std::setw(11) << "expr/x" << std::setw(15) << "cudnn_GB/s*" << std::setw(15) << "expr_GB/s*" << '\n';
        std::cout << std::string(104, '-') << '\n';

        const std::vector<DataType> dtypes{DataType::FP32, DataType::FP16, DataType::BF16};
        for (DataType dtype : dtypes) {
            for (uint64_t channels : options.channels) {
                const Result result =
                    runCase(options, stream, scrub, scrub_bytes, target_tensor_bytes, dtype, channels);
                printResult(result);
            }
        }

        std::cout << "\n* GB/s uses a conservative lower bound on global-memory traffic. "
                     "The cache-cold construction is what prevents the latency comparison from becoming L2-resident.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "thor_softmax_benchmark: " << e.what() << '\n';
        return 1;
    }
}
