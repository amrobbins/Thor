#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/FusedEquation.h"
#include "Utilities/Expression/StampedEquation.h"
#include "Utilities/TensorOperations/Copy/StridedCopy.h"
#include "Utilities/TensorOperations/Cub/CubReduction.h"

#include "cuda_runtime.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace ThorImplementation;

namespace {

constexpr int WARMUP_ITERATIONS = 3;
constexpr int TIMING_SAMPLES = 7;
constexpr int ITERATIONS_PER_SAMPLE = 8;
constexpr uint64_t REDUCTION_WIDTH = 256;
constexpr uint64_t GEMM_M = 2048;
constexpr uint64_t GEMM_N = 2048;
constexpr size_t DEFAULT_FIT_POINT_COUNT = 5;

struct TimingResult {
    double median_ms = 0.0;
    double best_ms = 0.0;
};

struct PrimitiveSample {
    std::string primitive;
    std::string shape;
    uint64_t work_units = 0;
    uint64_t logical_bytes = 0;
    TimingResult timing;
};

struct LinearFit {
    double intercept_ms = 0.0;
    double slope_ms_per_unit = 0.0;
    double r_squared = 0.0;
};

struct PrimitiveSummary {
    std::string primitive;
    std::string work_unit;
    LinearFit fit;
    double large_median_ms_per_unit = 0.0;
    double minimum_observed_ms = 0.0;
    size_t fit_point_count = 0;
};

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " + cudaGetErrorString(status));
    }
}

uint64_t checkedMultiply(uint64_t lhs, uint64_t rhs, const char* description) {
    if (rhs != 0 && lhs > std::numeric_limits<uint64_t>::max() / rhs) {
        throw std::overflow_error(std::string(description) + " overflowed uint64_t.");
    }
    return lhs * rhs;
}

Tensor makeZeroTensor(const std::vector<uint64_t>& dimensions, Stream& stream) {
    const TensorPlacement placement(TensorPlacement::MemDevices::GPU, stream.getGpuNum());
    Tensor tensor(placement, TensorDescriptor(DataType::FP32, dimensions));
    checkCuda(cudaMemsetAsync(tensor.getMemPtr<void>(), 0, tensor.getArraySizeInBytes(), stream.getStream()),
              "cudaMemsetAsync(calibration tensor)");
    return tensor;
}

TimingResult timeLaunch(const std::function<void()>& launch, Stream& stream) {
    for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
        launch();
    }
    stream.synchronize();

    cudaEvent_t start{};
    cudaEvent_t stop{};
    checkCuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    checkCuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");

    std::vector<double> samples;
    samples.reserve(TIMING_SAMPLES);
    for (int sample = 0; sample < TIMING_SAMPLES; ++sample) {
        checkCuda(cudaEventRecord(start, stream.getStream()), "cudaEventRecord(start)");
        for (int iteration = 0; iteration < ITERATIONS_PER_SAMPLE; ++iteration) {
            launch();
        }
        checkCuda(cudaEventRecord(stop, stream.getStream()), "cudaEventRecord(stop)");
        checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
        float elapsed_ms = 0.0f;
        checkCuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime");
        samples.push_back(static_cast<double>(elapsed_ms) / ITERATIONS_PER_SAMPLE);
    }

    checkCuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
    checkCuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");
    std::sort(samples.begin(), samples.end());
    return TimingResult{samples[samples.size() / 2], samples.front()};
}

std::shared_ptr<StampedExecutionPlan> stampSingleMatmul(const Tensor& lhs, const Tensor& rhs, Stream& stream) {
    const Expression lhs_expr = Expression::input("lhs", DataType::FP32, DataType::FP32);
    const Expression rhs_expr = Expression::input("rhs", DataType::FP32, DataType::FP32);
    const Expression output_expr =
        Expression::matmul(lhs_expr, rhs_expr, false, false, DataType::FP32, DataType::FP32);
    FusedEquation equation =
        FusedEquation::compile(Expression::outputs({{"output", output_expr}}).physicalOutputs(), stream.getGpuNum());
    StampedExecutionPlan stamped = equation.stamp({{"lhs", lhs}, {"rhs", rhs}}, stream);
    return std::make_shared<StampedExecutionPlan>(std::move(stamped));
}

std::shared_ptr<StampedExecutionPlan> stampPairProduct(const Tensor& lhs, const Tensor& rhs, Stream& stream) {
    const Expression lhs_expr = Expression::input("lhs", DataType::FP32, DataType::FP32);
    const Expression rhs_expr = Expression::input("rhs", DataType::FP32, DataType::FP32);
    const Expression output_expr = lhs_expr * rhs_expr;
    FusedEquation equation =
        FusedEquation::compile(Expression::outputs({{"output", output_expr}}).physicalOutputs(), stream.getGpuNum());
    StampedExecutionPlan stamped = equation.stamp({{"lhs", lhs}, {"rhs", rhs}}, stream);
    return std::make_shared<StampedExecutionPlan>(std::move(stamped));
}

PrimitiveSample runGemmCase(uint64_t k, Stream& stream) {
    Tensor lhs = makeZeroTensor({GEMM_M, k}, stream);
    Tensor rhs = makeZeroTensor({k, GEMM_N}, stream);
    stream.synchronize();

    const std::shared_ptr<StampedExecutionPlan> plan = stampSingleMatmul(lhs, rhs, stream);
    const std::vector<StampedMatmulStageDiagnostic> diagnostics = plan->matmulStageDiagnostics();
    if (diagnostics.size() != 1) {
        throw std::runtime_error("Primitive calibration GEMM expected exactly one Matmul stage.");
    }

    PrimitiveSample sample;
    sample.primitive = "gemm";
    sample.shape = std::to_string(GEMM_M) + "x" + std::to_string(GEMM_N) + "x" + std::to_string(k);
    sample.work_units = checkedMultiply(checkedMultiply(GEMM_M, GEMM_N, "GEMM m*n"), k, "GEMM FMA count");
    sample.logical_bytes = checkedMultiply(GEMM_M, k, "GEMM lhs elements") * sizeof(float) +
                           checkedMultiply(k, GEMM_N, "GEMM rhs elements") * sizeof(float) +
                           checkedMultiply(GEMM_M, GEMM_N, "GEMM output elements") * sizeof(float);
    sample.timing = timeLaunch([&] { plan->runOn(stream); }, stream);
    return sample;
}

PrimitiveSample runFusedPairProductCase(uint64_t elements, Stream& stream) {
    Tensor lhs = makeZeroTensor({elements}, stream);
    Tensor rhs = makeZeroTensor({elements}, stream);
    stream.synchronize();

    const std::shared_ptr<StampedExecutionPlan> plan = stampPairProduct(lhs, rhs, stream);
    const std::vector<std::string> stages = plan->stageKindNames();
    if (stages.size() != 1 || stages.front() != "FusedKernel") {
        throw std::runtime_error("Primitive calibration pair product expected exactly one FusedKernel stage.");
    }

    PrimitiveSample sample;
    sample.primitive = "fused_pair_product";
    sample.shape = "elements=" + std::to_string(elements);
    sample.work_units = elements;
    sample.logical_bytes = checkedMultiply(elements, 3 * sizeof(float), "fused pair-product logical bytes");
    sample.timing = timeLaunch([&] { plan->runOn(stream); }, stream);
    return sample;
}

PrimitiveSample runReductionCase(uint64_t elements, Stream& stream) {
    if (elements % REDUCTION_WIDTH != 0) {
        throw std::invalid_argument("Primitive calibration reduction element count must be divisible by reduction width.");
    }
    const uint64_t rows = elements / REDUCTION_WIDTH;
    Tensor input = makeZeroTensor({rows, REDUCTION_WIDTH}, stream);
    stream.synchronize();

    const std::shared_ptr<StampedCubReduction> reduction =
        CubReduction(CubReductionOp::Sum, 1, DataType::FP32).stamp(input, stream);

    PrimitiveSample sample;
    sample.primitive = "cub_reduction";
    sample.shape = std::to_string(rows) + "x" + std::to_string(REDUCTION_WIDTH) + "->" + std::to_string(rows);
    sample.work_units = elements;
    sample.logical_bytes = input.getArraySizeInBytes() + reduction->getOutputTensor().getArraySizeInBytes();
    sample.timing = timeLaunch([&] { reduction->runOn(stream); }, stream);
    return sample;
}

PrimitiveSample runMaterializationCase(uint64_t elements, Stream& stream) {
    Tensor source = makeZeroTensor({elements}, stream);
    Tensor destination = makeZeroTensor({elements}, stream);
    stream.synchronize();

    PrimitiveSample sample;
    sample.primitive = "materialization";
    sample.shape = "elements=" + std::to_string(elements);
    sample.work_units = elements;
    sample.logical_bytes = checkedMultiply(elements, 2 * sizeof(float), "materialization logical bytes");
    sample.timing = timeLaunch([&] { materializeTensorViewAsync(source, destination, stream); }, stream);
    return sample;
}

LinearFit fitLine(const std::vector<PrimitiveSample>& samples, size_t fit_point_count) {
    if (samples.size() < 2) {
        throw std::invalid_argument("Primitive calibration linear fit requires at least two samples.");
    }
    fit_point_count = std::min(fit_point_count, samples.size());
    const size_t begin = samples.size() - fit_point_count;

    long double mean_x = 0.0L;
    long double mean_y = 0.0L;
    for (size_t i = begin; i < samples.size(); ++i) {
        mean_x += static_cast<long double>(samples[i].work_units);
        mean_y += static_cast<long double>(samples[i].timing.median_ms);
    }
    mean_x /= static_cast<long double>(fit_point_count);
    mean_y /= static_cast<long double>(fit_point_count);

    long double covariance = 0.0L;
    long double variance_x = 0.0L;
    long double variance_y = 0.0L;
    for (size_t i = begin; i < samples.size(); ++i) {
        const long double dx = static_cast<long double>(samples[i].work_units) - mean_x;
        const long double dy = static_cast<long double>(samples[i].timing.median_ms) - mean_y;
        covariance += dx * dy;
        variance_x += dx * dx;
        variance_y += dy * dy;
    }
    if (variance_x == 0.0L) {
        throw std::runtime_error("Primitive calibration linear fit has zero work-unit variance.");
    }

    const long double slope = covariance / variance_x;
    const long double intercept = mean_y - slope * mean_x;
    const long double r_squared = variance_y > 0.0L ? (covariance * covariance) / (variance_x * variance_y) : 1.0L;
    return LinearFit{static_cast<double>(intercept), static_cast<double>(slope), static_cast<double>(r_squared)};
}

double medianLargeUnitCost(const std::vector<PrimitiveSample>& samples, size_t point_count) {
    point_count = std::min(point_count, samples.size());
    std::vector<double> costs;
    costs.reserve(point_count);
    for (size_t i = samples.size() - point_count; i < samples.size(); ++i) {
        costs.push_back(samples[i].timing.median_ms / static_cast<double>(samples[i].work_units));
    }
    std::sort(costs.begin(), costs.end());
    return costs[costs.size() / 2];
}

PrimitiveSummary summarize(const std::string& primitive,
                           const std::string& work_unit,
                           const std::vector<PrimitiveSample>& samples,
                           size_t fit_point_count) {
    PrimitiveSummary result;
    result.primitive = primitive;
    result.work_unit = work_unit;
    result.fit_point_count = std::min(fit_point_count, samples.size());
    result.fit = fitLine(samples, result.fit_point_count);
    result.large_median_ms_per_unit = medianLargeUnitCost(samples, result.fit_point_count);
    result.minimum_observed_ms = std::min_element(samples.begin(), samples.end(), [](const auto& lhs, const auto& rhs) {
                                     return lhs.timing.median_ms < rhs.timing.median_ms;
                                 })->timing.median_ms;
    return result;
}

void printSamples(const std::vector<PrimitiveSample>& samples) {
    std::cout << "\n# primitive_samples\n"
              << "primitive,shape,work_units,logical_bytes,median_ms,best_ms,work_gunits_per_s,logical_gb_per_s\n";
    for (const PrimitiveSample& sample : samples) {
        const double work_gunits_per_s = static_cast<double>(sample.work_units) / (sample.timing.median_ms * 1.0e6);
        const double logical_gb_per_s = static_cast<double>(sample.logical_bytes) / (sample.timing.median_ms * 1.0e6);
        std::cout << sample.primitive << ',' << sample.shape << ',' << sample.work_units << ',' << sample.logical_bytes << ','
                  << std::fixed << std::setprecision(6) << sample.timing.median_ms << ',' << sample.timing.best_ms << ','
                  << std::setprecision(3) << work_gunits_per_s << ',' << logical_gb_per_s << '\n';
    }
}

void printSummary(const std::vector<PrimitiveSummary>& summaries) {
    const auto gemm_it = std::find_if(summaries.begin(), summaries.end(), [](const PrimitiveSummary& summary) {
        return summary.primitive == "gemm";
    });
    if (gemm_it == summaries.end() || gemm_it->fit.slope_ms_per_unit <= 0.0) {
        throw std::runtime_error("Primitive calibration could not establish a positive GEMM work slope.");
    }
    const double gemm_slope = gemm_it->fit.slope_ms_per_unit;

    std::cout << "\n# primitive_fit_summary\n"
              << "primitive,work_unit,fit_points,fit_intercept_us,fit_slope_ps_per_unit,fit_r_squared,"
                 "large_median_ps_per_unit,relative_fit_cost_vs_gemm_fma,minimum_observed_us,"
                 "fit_intercept_equiv_gemm_fma\n";
    for (const PrimitiveSummary& summary : summaries) {
        const double relative = summary.fit.slope_ms_per_unit > 0.0 ? summary.fit.slope_ms_per_unit / gemm_slope : 0.0;
        const double equiv_fma = summary.fit.intercept_ms > 0.0 ? summary.fit.intercept_ms / gemm_slope : 0.0;
        std::cout << summary.primitive << ',' << summary.work_unit << ',' << summary.fit_point_count << ','
                  << std::fixed << std::setprecision(6) << summary.fit.intercept_ms * 1000.0 << ','
                  << summary.fit.slope_ms_per_unit * 1.0e9 << ',' << summary.fit.r_squared << ','
                  << summary.large_median_ms_per_unit * 1.0e9 << ',' << relative << ','
                  << summary.minimum_observed_ms * 1000.0 << ',' << std::setprecision(1) << equiv_fma << '\n';
    }

    const auto findSummary = [&](std::string_view name) -> const PrimitiveSummary& {
        const auto it = std::find_if(summaries.begin(), summaries.end(), [&](const PrimitiveSummary& summary) {
            return summary.primitive == name;
        });
        if (it == summaries.end()) {
            throw std::runtime_error("Missing primitive calibration summary for " + std::string(name) + ".");
        }
        return *it;
    };

    const PrimitiveSummary& fused = findSummary("fused_pair_product");
    const PrimitiveSummary& reduction = findSummary("cub_reduction");
    const PrimitiveSummary& materialization = findSummary("materialization");
    const auto relativeSlope = [&](const PrimitiveSummary& summary) {
        return std::max(0.0, summary.fit.slope_ms_per_unit / gemm_slope);
    };
    const double fused_weight = relativeSlope(fused);
    const double reduction_weight = relativeSlope(reduction);
    const double materialization_weight = relativeSlope(materialization);
    const double result_write_weight = materialization_weight * 0.5;

    std::cout << "\n# planner_weight_guidance\n"
              << "# These ratios are backend-class calibration evidence, not an automatic policy update.\n"
              << "# GEMM is normalized to one FMA. Materialization is one FP32 read+write; result_write uses half of that\n"
              << "# slope as a first-order write-only traffic estimate. Fixed primitive costs should be represented by\n"
              << "# primitive/group counts rather than by shape-specific cuBLASLt efficiency.\n"
              << "component,relative_cost_vs_gemm_fma,nearest_integer\n"
              << "gemm_fma,1.000000,1\n"
              << "fused_element," << std::fixed << std::setprecision(6) << fused_weight << ','
              << static_cast<uint64_t>(std::llround(fused_weight)) << '\n'
              << "reduction_input_element," << reduction_weight << ','
              << static_cast<uint64_t>(std::llround(reduction_weight)) << '\n'
              << "materialization_element," << materialization_weight << ','
              << static_cast<uint64_t>(std::llround(materialization_weight)) << '\n'
              << "result_write_element," << result_write_weight << ','
              << static_cast<uint64_t>(std::llround(result_write_weight)) << '\n';
}

std::vector<uint64_t> elementSweep(bool quick) {
    if (quick) {
        return {65536ULL, 1048576ULL, 16777216ULL};
    }
    return {4096ULL, 16384ULL, 65536ULL, 262144ULL, 1048576ULL, 4194304ULL, 16777216ULL, 67108864ULL};
}

std::vector<uint64_t> gemmKSweep(bool quick) {
    if (quick) {
        return {128ULL, 512ULL, 2048ULL};
    }
    return {16ULL, 32ULL, 64ULL, 128ULL, 256ULL, 512ULL, 1024ULL, 2048ULL};
}

}  // namespace

int main(int argc, char** argv) {
    try {
        int device = 0;
        bool quick = false;
        size_t fit_point_count = DEFAULT_FIT_POINT_COUNT;
        for (int i = 1; i < argc; ++i) {
            const std::string_view arg(argv[i]);
            if (arg == "--help") {
                std::cout << "thor_einsum_primitive_cost_benchmark options:\n"
                          << "  --device=N        CUDA device (default 0)\n"
                          << "  --quick           Three-point smoke/calibration sweep\n"
                          << "  --fit-points=N    Number of largest points used for linear slope/intercept fits (default 5)\n";
                return EXIT_SUCCESS;
            }
            if (arg.starts_with("--device=")) {
                device = std::stoi(std::string(arg.substr(9)));
            } else if (arg == "--quick") {
                quick = true;
            } else if (arg.starts_with("--fit-points=")) {
                fit_point_count = std::stoull(std::string(arg.substr(13)));
                if (fit_point_count < 2) {
                    throw std::invalid_argument("--fit-points must be at least 2.");
                }
            } else {
                throw std::invalid_argument("Unknown argument: " + std::string(arg));
            }
        }

        checkCuda(cudaSetDevice(device), "cudaSetDevice");
        Stream stream(device);

        std::cout << "# Thor einsum primitive-class cost calibration\n"
                  << "# device=" << device << " gemm_m=" << GEMM_M << " gemm_n=" << GEMM_N
                  << " reduction_width=" << REDUCTION_WIDTH << " fit_points=" << fit_point_count
                  << " quick=" << (quick ? 1 : 0) << '\n'
                  << "# intent=calibrate broad GEMM-vs-CUB-reduction-vs-fused-vs-materialization costs; "
                     "do_not_fit cuBLASLt shape quirks\n";

        std::vector<PrimitiveSample> all_samples;
        std::vector<PrimitiveSample> gemm_samples;
        std::vector<PrimitiveSample> fused_samples;
        std::vector<PrimitiveSample> reduction_samples;
        std::vector<PrimitiveSample> materialization_samples;

        for (uint64_t k : gemmKSweep(quick)) {
            PrimitiveSample sample = runGemmCase(k, stream);
            gemm_samples.push_back(sample);
            all_samples.push_back(std::move(sample));
        }
        for (uint64_t elements : elementSweep(quick)) {
            PrimitiveSample fused = runFusedPairProductCase(elements, stream);
            fused_samples.push_back(fused);
            all_samples.push_back(std::move(fused));

            PrimitiveSample reduction = runReductionCase(elements, stream);
            reduction_samples.push_back(reduction);
            all_samples.push_back(std::move(reduction));

            PrimitiveSample materialization = runMaterializationCase(elements, stream);
            materialization_samples.push_back(materialization);
            all_samples.push_back(std::move(materialization));
        }

        printSamples(all_samples);
        const size_t effective_fit_points = quick ? std::min<size_t>(3, fit_point_count) : fit_point_count;
        std::vector<PrimitiveSummary> summaries;
        summaries.push_back(summarize("gemm", "fma", gemm_samples, effective_fit_points));
        summaries.push_back(summarize("fused_pair_product", "output_element", fused_samples, effective_fit_points));
        summaries.push_back(summarize("cub_reduction", "input_element", reduction_samples, effective_fit_points));
        summaries.push_back(summarize("materialization", "copied_element", materialization_samples, effective_fit_points));
        printSummary(summaries);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Benchmark failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
