#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/FusedEquation.h"
#include "Utilities/Expression/StampedEquation.h"
#include "Utilities/TensorOperations/Cub/CubReduction.h"
#include "Utilities/TensorOperations/Einsum/Einsum.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace ThorImplementation;

namespace {

constexpr uint64_t MIB = 1024ULL * 1024ULL;
constexpr uint64_t DEFAULT_MIN_INPUT_BYTES = 512ULL * MIB;
constexpr uint64_t DEFAULT_L2_WORKING_SET_MULTIPLE = 8;
constexpr int WARMUP_ITERATIONS = 3;
constexpr int TIMING_SAMPLES = 5;
constexpr int TIMED_ITERATIONS_PER_SAMPLE = 4;

struct BenchmarkCase {
    const char* name;
    uint64_t reduction_j;
    uint64_t contiguous_k;
};

struct Options {
    int device = 0;
    std::optional<uint64_t> target_input_bytes;
    std::optional<std::string> case_filter;
};

struct TimingResult {
    double median_ms = 0.0;
    double best_ms = 0.0;
    double worst_ms = 0.0;
};

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " + cudaGetErrorString(status));
    }
}

[[nodiscard]] uint64_t checkedMultiply(uint64_t lhs, uint64_t rhs, const char* role) {
    if (rhs != 0 && lhs > std::numeric_limits<uint64_t>::max() / rhs) {
        throw std::overflow_error(std::string(role) + " overflowed uint64_t.");
    }
    return lhs * rhs;
}

[[nodiscard]] uint64_t checkedAdd(uint64_t lhs, uint64_t rhs, const char* role) {
    if (lhs > std::numeric_limits<uint64_t>::max() - rhs) {
        throw std::overflow_error(std::string(role) + " overflowed uint64_t.");
    }
    return lhs + rhs;
}

[[nodiscard]] uint64_t ceilDiv(uint64_t numerator, uint64_t denominator) {
    if (denominator == 0) {
        throw std::invalid_argument("ceilDiv denominator must be non-zero.");
    }
    return numerator / denominator + static_cast<uint64_t>(numerator % denominator != 0);
}

[[nodiscard]] const char* dataTypeName(DataType dtype) {
    switch (dtype) {
        case DataType::FP16:
            return "fp16";
        case DataType::BF16:
            return "bf16";
        case DataType::FP32:
            return "fp32";
        default:
            return "unsupported";
    }
}

[[nodiscard]] const char* reductionPathName(CubReductionPath path) {
    switch (path) {
        case CubReductionPath::DeviceTransformReduce:
            return "device_transform";
        case CubReductionPath::ContiguousFixedSegment:
            return "contiguous_segment";
        case CubReductionPath::TiledFixedSegment:
            return "tiled_segment";
        case CubReductionPath::StridedFixedSegment:
            return "strided_segment";
        case CubReductionPath::OffsetSegmented:
            return "offset_segmented";
    }
    return "unknown";
}

[[nodiscard]] std::string join(const std::vector<std::string>& values, std::string_view delimiter) {
    std::ostringstream out;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            out << delimiter;
        }
        out << values[i];
    }
    return out.str();
}

[[nodiscard]] std::string strideString(const std::vector<uint64_t>& strides) {
    std::ostringstream out;
    out << '[';
    for (size_t i = 0; i < strides.size(); ++i) {
        if (i != 0) {
            out << 'x';
        }
        out << strides[i];
    }
    out << ']';
    return out.str();
}

[[nodiscard]] Options parseOptions(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "--help") {
            std::cout
                << "thor_einsum_benchmark options:\n"
                << "  --device=N          CUDA device (default 0)\n"
                << "  --target-mib=N      Approximate input size per case. Default=max(512 MiB, 8x L2).\n"
                << "  --case=SUBSTRING    Run only benchmark case names containing SUBSTRING.\n";
            std::exit(EXIT_SUCCESS);
        }
        if (arg.starts_with("--device=")) {
            options.device = std::stoi(std::string(arg.substr(9)));
        } else if (arg.starts_with("--target-mib=")) {
            const uint64_t mib = std::stoull(std::string(arg.substr(13)));
            options.target_input_bytes = checkedMultiply(mib, MIB, "target input bytes");
        } else if (arg.starts_with("--case=")) {
            options.case_filter = std::string(arg.substr(7));
        } else {
            throw std::invalid_argument("Unknown argument: " + std::string(arg));
        }
    }
    return options;
}

constexpr size_t STRATEGY_COUNT = 2;

[[nodiscard]] std::array<TimingResult, STRATEGY_COUNT> benchmarkLaunchesInterleaved(
    const std::array<std::function<void()>, STRATEGY_COUNT>& launches,
    Stream& stream) {
    // Warm every strategy before taking samples, rotating order so the last warmup is not always the same implementation.
    for (int warmup = 0; warmup < WARMUP_ITERATIONS; ++warmup) {
        for (size_t offset = 0; offset < STRATEGY_COUNT; ++offset) {
            const size_t strategy = (static_cast<size_t>(warmup) + offset) % STRATEGY_COUNT;
            launches[strategy]();
        }
    }
    stream.synchronize();

    std::array<cudaEvent_t, STRATEGY_COUNT> starts{};
    std::array<cudaEvent_t, STRATEGY_COUNT> stops{};
    std::array<std::vector<double>, STRATEGY_COUNT> samples;
    for (size_t strategy = 0; strategy < STRATEGY_COUNT; ++strategy) {
        checkCuda(cudaEventCreate(&starts[strategy]), "cudaEventCreate(start)");
        checkCuda(cudaEventCreate(&stops[strategy]), "cudaEventCreate(stop)");
        samples[strategy].reserve(TIMING_SAMPLES);
    }

    // Each sample times the strategies separately, but rotates which one runs first. This keeps the benchmark sensitive
    // to host-side launch gaps (which matter when a short kernel drains the queue) without permanently advantaging one
    // implementation through boost/thermal/order effects.
    for (int sample = 0; sample < TIMING_SAMPLES; ++sample) {
        for (size_t offset = 0; offset < STRATEGY_COUNT; ++offset) {
            const size_t strategy = (static_cast<size_t>(sample) + offset) % STRATEGY_COUNT;
            checkCuda(cudaEventRecord(starts[strategy], stream.getStream()), "cudaEventRecord(start)");
            for (int iteration = 0; iteration < TIMED_ITERATIONS_PER_SAMPLE; ++iteration) {
                launches[strategy]();
            }
            checkCuda(cudaEventRecord(stops[strategy], stream.getStream()), "cudaEventRecord(stop)");
            checkCuda(cudaEventSynchronize(stops[strategy]), "cudaEventSynchronize(stop)");

            float elapsed_ms = 0.0f;
            checkCuda(cudaEventElapsedTime(&elapsed_ms, starts[strategy], stops[strategy]), "cudaEventElapsedTime");
            samples[strategy].push_back(static_cast<double>(elapsed_ms) / TIMED_ITERATIONS_PER_SAMPLE);
        }
    }

    std::array<TimingResult, STRATEGY_COUNT> results{};
    for (size_t strategy = 0; strategy < STRATEGY_COUNT; ++strategy) {
        checkCuda(cudaEventDestroy(starts[strategy]), "cudaEventDestroy(start)");
        checkCuda(cudaEventDestroy(stops[strategy]), "cudaEventDestroy(stop)");
        std::sort(samples[strategy].begin(), samples[strategy].end());
        results[strategy] = TimingResult{
            .median_ms = samples[strategy][samples[strategy].size() / 2],
            .best_ms = samples[strategy].front(),
            .worst_ms = samples[strategy].back(),
        };
    }
    return results;
}

[[nodiscard]] FusedEquation compileReduceFirstTranspose(DataType dtype, int device) {
    const Expression x = Expression::input("x", dtype, dtype);
    // [I,J,K] --sum(J)--> [I,K] --transpose/materialize--> [K,I].
    const Expression reduced = x.reduce_sum({1}, {1}, DataType::FP32).withOutputDType(dtype);
    const Expression y = reduced.transpose();
    return FusedEquation::compile(Expression::outputs({{"y", y}}).physicalOutputs(), device);
}

void printStrategyDescription(uint64_t i,
                              uint64_t j,
                              uint64_t k,
                              const Tensor& input,
                              const StampedEinsum& production,
                              const StampedExecutionPlan& materializing_reference) {
    const std::vector<uint64_t>& source_strides = input.getStridesElements();
    const std::vector<uint64_t> production_view_strides = {source_strides[2], source_strides[0], source_strides[1]};

    std::cout << "# processing production_cub: [" << i << ',' << j << ',' << k << "] strides="
              << strideString(source_strides) << " --zero-copy permute--> [" << k << ',' << i << ',' << j
              << "] strides=" << strideString(production_view_strides) << " --reduce axis J--> [" << k << ',' << i
              << "] reduction_path="
              << (production.getStandaloneReductionPath().has_value()
                      ? reductionPathName(production.getStandaloneReductionPath().value())
                      : "none")
              << " stages=" << join(production.getExpressionStageKindNames(), "+")
              << " retained_write=shared_transpose\n";

    const std::vector<CubReductionPath> reference_paths = materializing_reference.reductionPaths();
    std::cout << "# processing materializing_reference: [" << i << ',' << j << ',' << k << "] strides="
              << strideString(source_strides) << " --reduce dense axis J--> [" << i << ',' << k
              << "] --transpose/materialize--> [" << k << ',' << i << "] reduction_path="
              << (reference_paths.empty() ? "none" : reductionPathName(reference_paths.front()))
              << " stages=" << join(materializing_reference.stageKindNames(), "+") << '\n';
}

void runCase(const BenchmarkCase& benchmark_case,
             uint64_t target_input_bytes,
             int device,
             Stream& stream,
             const TensorPlacement& placement) {
    constexpr DataType dtype = DataType::FP32;
    const uint64_t element_bytes = TensorDescriptor::getElementSizeInBytes(dtype);
    const uint64_t target_elements = ceilDiv(target_input_bytes, element_bytes);
    const uint64_t jk = checkedMultiply(benchmark_case.reduction_j, benchmark_case.contiguous_k, "J*K");
    const uint64_t i = std::max<uint64_t>(1, ceilDiv(target_elements, jk));
    const uint64_t j = benchmark_case.reduction_j;
    const uint64_t k = benchmark_case.contiguous_k;

    Tensor input(placement, TensorDescriptor(dtype, {i, j, k}));
    Tensor output(placement, TensorDescriptor(dtype, {k, i}));

    // Data initialization is not timed. Zeros are sufficient because this benchmark measures traversal and storage
    // behavior, and both implementations perform identical arithmetic regardless of the values.
    checkCuda(cudaMemsetAsync(input.getMemPtr<void>(), 0, input.getArraySizeInBytes(), stream.getStream()),
              "cudaMemsetAsync(input)");
    stream.synchronize();

    std::shared_ptr<StampedEinsum> production = Einsum("ijk->ki").stamp({input}, output, stream);
    const std::optional<CubReductionPath> production_path = production->getStandaloneReductionPath();
    if (!production_path.has_value() || production_path.value() != CubReductionPath::TiledFixedSegment) {
        throw std::runtime_error(
            "production einsum benchmark expected permutation-aware TiledFixedSegment reduction.");
    }

    FusedEquation reference_equation = compileReduceFirstTranspose(dtype, device);
    StampedExecutionPlan materializing_reference =
        reference_equation.stamp({{"x", input}}, stream, {}, {{"y", output}});
    if (materializing_reference.output("y") != output) {
        throw std::runtime_error("materializing reference failed to use the caller-provided dense output tensor.");
    }

    const uint64_t output_bytes = output.getArraySizeInBytes();
    const uint64_t intermediate_bytes = output_bytes;  // [I,K] has the same element count as [K,I].
    const uint64_t production_minimum_bytes = checkedAdd(input.getArraySizeInBytes(), output_bytes, "production bytes");
    uint64_t reference_minimum_bytes = checkedAdd(input.getArraySizeInBytes(), intermediate_bytes, "reference bytes");
    reference_minimum_bytes = checkedAdd(reference_minimum_bytes, intermediate_bytes, "reference bytes");
    reference_minimum_bytes = checkedAdd(reference_minimum_bytes, output_bytes, "reference bytes");

    printStrategyDescription(i, j, k, input, *production, materializing_reference);

    const std::array<TimingResult, STRATEGY_COUNT> timings = benchmarkLaunchesInterleaved(
        {std::function<void()>([&]() { production->runOn(stream); }),
         std::function<void()>([&]() { materializing_reference.runOn(stream); })},
        stream);
    const TimingResult& production_timing = timings[0];
    const TimingResult& reference_timing = timings[1];

    const double production_gbps =
        static_cast<double>(production_minimum_bytes) / (production_timing.median_ms * 1.0e6);
    const double reference_gbps =
        static_cast<double>(reference_minimum_bytes) / (reference_timing.median_ms * 1.0e6);
    const double production_vs_reference_speedup =
        reference_timing.median_ms / production_timing.median_ms;

    const std::vector<CubReductionPath> reference_paths = materializing_reference.reductionPaths();

    std::cout << benchmark_case.name << ',' << dataTypeName(dtype) << ',' << i << ',' << j << ',' << k << ','
              << input.getArraySizeInBytes() << ',' << output_bytes << ',' << intermediate_bytes << ','
              << reductionPathName(production_path.value()) << ','
              << (reference_paths.empty() ? "none" : reductionPathName(reference_paths.front())) << ','
              << std::fixed << std::setprecision(4) << production_timing.median_ms << ',' << production_timing.best_ms << ','
              << production_timing.worst_ms << ',' << reference_timing.median_ms << ',' << reference_timing.best_ms << ','
              << reference_timing.worst_ms << ',' << std::setprecision(2) << production_gbps << ',' << reference_gbps << ','
              << std::setprecision(3) << production_vs_reference_speedup << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parseOptions(argc, argv);

        int device_count = 0;
        checkCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
        if (device_count == 0) {
            std::cerr << "EinsumBenchmark requires a CUDA GPU.\n";
            return EXIT_FAILURE;
        }
        if (options.device < 0 || options.device >= device_count) {
            throw std::invalid_argument("Requested CUDA device is out of range.");
        }

        ScopedGpu scoped_gpu(options.device);
        cudaDeviceProp properties{};
        checkCuda(cudaGetDeviceProperties(&properties, options.device), "cudaGetDeviceProperties");

        int l2_cache_bytes = 0;
        checkCuda(cudaDeviceGetAttribute(&l2_cache_bytes, cudaDevAttrL2CacheSize, options.device),
                  "cudaDeviceGetAttribute(cudaDevAttrL2CacheSize)");
        if (l2_cache_bytes <= 0) {
            throw std::runtime_error("CUDA reported a non-positive L2 cache size.");
        }

        const uint64_t default_target =
            std::max<uint64_t>(DEFAULT_MIN_INPUT_BYTES,
                               checkedMultiply(static_cast<uint64_t>(l2_cache_bytes),
                                               DEFAULT_L2_WORKING_SET_MULTIPLE,
                                               "L2 working set"));
        const uint64_t target_input_bytes = options.target_input_bytes.value_or(default_target);

        size_t free_bytes = 0;
        size_t total_bytes = 0;
        checkCuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
        if (target_input_bytes > free_bytes / 3) {
            throw std::runtime_error(
                "Insufficient free GPU memory for the requested benchmark working set. The worst initial case needs "
                "roughly input + output + reduce-first intermediate storage; lower --target-mib if necessary.");
        }

        std::cout << "# Thor einsum permutation/reduction strategy benchmark\n";
        std::cout << "# device=" << properties.name << " device_index=" << options.device << " dtype=fp32 l2_bytes=" << l2_cache_bytes
                  << " target_input_bytes=" << target_input_bytes << " target_over_l2=" << std::fixed
                  << std::setprecision(2) << static_cast<double>(target_input_bytes) / l2_cache_bytes << '\n';
        std::cout << "# equation=ijk->ki. production_cub is the production einsum lowering: zero-copy [K,I,J] view "
                     "followed by central permutation-aware tiled CUB reduction whose finalized retained values are "
                     "transposed through shared memory into coalesced dense [K,I] stores. materializing_reference reduces "
                     "dense [I,J,K] over J first and then physically transposes [I,K] to [K,I]. The reference exists only "
                     "to quantify the cost avoided by the no-materialization production path.\n";
        std::cout << "# production_minimum_bytes=input+output. materializing_reference_minimum_bytes=input+"
                     "write_intermediate+read_intermediate+output. Reported GB/s uses those minimum logical bytes; wall "
                     "time is the primary comparison. Stamping is excluded from timing.\n";
        std::cout << "# timing_samples=" << TIMING_SAMPLES
                  << " timed_iterations_per_sample=" << TIMED_ITERATIONS_PER_SAMPLE << " warmup_iterations="
                  << WARMUP_ITERATIONS << " reported_time=median strategy_order=rotated_per_sample\n";
        std::cout << "case,dtype,I,J,K,input_bytes,output_bytes,materializing_reference_intermediate_bytes,"
                     "production_reduction_path,materializing_reference_reduction_path,production_median_ms,"
                     "production_best_ms,production_worst_ms,materializing_reference_median_ms,"
                     "materializing_reference_best_ms,materializing_reference_worst_ms,production_minimum_GBps,"
                     "materializing_reference_minimum_GBps,production_vs_materializing_reference_speedup\n";

        // J controls how much source traffic is amortized over each final output element. K is the physically contiguous
        // retained width in [I,J,K], and therefore controls the production tiled reduction geometry.
        // The selected grid intentionally spans output-heavy (J=2) through reduction-heavy (J=4096) regimes.
        const std::vector<BenchmarkCase> cases = {
            {"j2_k32", 2, 32},
            {"j2_k256", 2, 256},
            {"j2_k4096", 2, 4096},
            {"j8_k64", 8, 64},
            {"j8_k512", 8, 512},
            {"j8_k4096", 8, 4096},
            {"j64_k64", 64, 64},
            {"j64_k512", 64, 512},
            {"j256_k64", 256, 64},
            {"j256_k512", 256, 512},
            {"j4096_k64", 4096, 64},
            {"j4096_k512", 4096, 512},
        };

        TensorPlacement placement(TensorPlacement::MemDevices::GPU, options.device);
        Stream stream(options.device);
        for (const BenchmarkCase& benchmark_case : cases) {
            if (options.case_filter.has_value()
                && std::string_view(benchmark_case.name).find(std::string_view(options.case_filter.value())) == std::string_view::npos) {
                continue;
            }
            runCase(benchmark_case, target_input_bytes, options.device, stream, placement);
        }

        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "EinsumBenchmark failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
