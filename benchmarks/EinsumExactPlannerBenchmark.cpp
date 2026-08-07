#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/FusedEquation.h"
#include "Utilities/Expression/StampedEquation.h"
#include "Utilities/TensorOperations/Einsum/Einsum.h"

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
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace ThorImplementation;

namespace {

constexpr int WARMUP_ITERATIONS = 3;
constexpr int TIMING_SAMPLES = 7;
constexpr int ITERATIONS_PER_SAMPLE = 8;
constexpr double DEFAULT_MAX_GENERIC_MIB = 256.0;
constexpr double VALIDATION_ABSOLUTE_TOLERANCE = 2.0e-3;
constexpr double VALIDATION_RELATIVE_TOLERANCE = 1.0e-2;

struct BenchmarkFamily {
    std::string name;
    std::string equation;
    size_t operand_count = 0;
};

struct DimensionProfile {
    std::string name;
    uint64_t outer_dimension = 0;
    uint64_t inner_dimension = 0;
};

struct BenchmarkCase {
    std::string name;
    std::string family;
    std::string size;
    std::string equation;
    std::vector<uint64_t> chain_dimensions;
};

struct TimingResult {
    double median_ms = 0.0;
    double best_ms = 0.0;
};

struct DualStreamTimingResult {
    double median_pair_ms = 0.0;
    double best_pair_ms = 0.0;
    double median_per_plan_ms = 0.0;
    double best_per_plan_ms = 0.0;
};

struct ValidationResult {
    double max_abs_error = 0.0;
    double max_scaled_error = 0.0;
};

struct MatmulResourceSummary {
    size_t stage_count = 0;
    size_t measured_stage_count = 0;
    double picker_runtime_sum_ms = 0.0;
    double full_sm_time_proxy_ms = 0.0;
    double max_waves = 0.0;
    double max_sm_pressure_proxy = 0.0;
};

struct CalibrationResult {
    BenchmarkCase benchmark_case;
    std::string selected_tree;
    EinsumExactContractionCost planner_cost;
    long double generic_broadcast_mib = 0.0L;
    bool generic_measured = false;
    std::vector<StampedMatmulStageDiagnostic> exact_matmuls;
    std::vector<StampedMatmulStageDiagnostic> bad_matmuls;
    MatmulResourceSummary exact_resources;
    MatmulResourceSummary bad_resources;
    TimingResult exact_timing;
    TimingResult generic_timing;
    TimingResult bad_timing;
    std::optional<DualStreamTimingResult> exact_dual;
    std::optional<DualStreamTimingResult> bad_dual;
};

struct GemmShapeCase {
    std::string kind;
    uint64_t outer = 0;
    uint64_t inner = 0;
    uint64_t m = 0;
    uint64_t n = 0;
    uint64_t k = 0;
};

struct GemmShapeResult {
    GemmShapeCase shape;
    StampedMatmulStageDiagnostic diagnostic;
    TimingResult isolated;
    std::optional<DualStreamTimingResult> dual;
};

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " + cudaGetErrorString(status));
    }
}

Tensor makeInput(const std::vector<uint64_t>& dimensions, uint32_t seed, Stream& stream) {
    const TensorPlacement cpu_placement(TensorPlacement::MemDevices::CPU);
    const TensorPlacement gpu_placement(TensorPlacement::MemDevices::GPU, stream.getGpuNum());
    Tensor cpu(cpu_placement, TensorDescriptor(DataType::FP32, dimensions));
    float* values = cpu.getMemPtr<float>();
    const double fan_in = static_cast<double>(dimensions.front());
    const float value_scale = static_cast<float>(1.0 / std::sqrt(24.0 * fan_in));
    for (uint64_t i = 0; i < cpu.getTotalNumElements(); ++i) {
        const uint32_t mixed = static_cast<uint32_t>(i * 1664525ULL + seed * 1013904223ULL);
        values[i] = static_cast<float>(static_cast<int32_t>(mixed % 17) - 8) * value_scale;
    }
    Tensor gpu(gpu_placement, cpu.getDescriptor());
    gpu.copyFromAsync(cpu, stream);
    return gpu;
}

std::vector<float> copyToHost(const Tensor& gpu, Stream& stream) {
    const TensorPlacement cpu_placement(TensorPlacement::MemDevices::CPU);
    Tensor cpu(cpu_placement, gpu.getDescriptor());
    cpu.copyFromAsync(gpu, stream);
    stream.synchronize();
    const float* data = cpu.getMemPtr<float>();
    return std::vector<float>(data, data + cpu.getTotalNumElements());
}

ValidationResult validateClose(const Tensor& lhs,
                               const Tensor& rhs,
                               Stream& stream,
                               const char* lhs_name,
                               const char* rhs_name) {
    const std::vector<float> lhs_values = copyToHost(lhs, stream);
    const std::vector<float> rhs_values = copyToHost(rhs, stream);
    if (lhs_values.size() != rhs_values.size()) {
        throw std::runtime_error(std::string(lhs_name) + " and " + rhs_name + " output sizes differ.");
    }

    ValidationResult result;
    for (size_t i = 0; i < lhs_values.size(); ++i) {
        const double lhs_value = static_cast<double>(lhs_values[i]);
        const double rhs_value = static_cast<double>(rhs_values[i]);
        const double abs_error = std::abs(lhs_value - rhs_value);
        const double scale = std::max(std::abs(lhs_value), std::abs(rhs_value));
        const double allowed_error = VALIDATION_ABSOLUTE_TOLERANCE + VALIDATION_RELATIVE_TOLERANCE * scale;
        result.max_abs_error = std::max(result.max_abs_error, abs_error);
        result.max_scaled_error = std::max(result.max_scaled_error, abs_error / allowed_error);
    }
    if (result.max_scaled_error > 1.0) {
        throw std::runtime_error(std::string(lhs_name) + " and " + rhs_name +
                                 " differ; max_abs_error=" + std::to_string(result.max_abs_error) +
                                 " max_scaled_error=" + std::to_string(result.max_scaled_error));
    }
    return result;
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

DualStreamTimingResult timeConcurrentPair(const std::function<void()>& launch_a,
                                          Stream& stream_a,
                                          const std::function<void()>& launch_b,
                                          Stream& stream_b) {
    for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
        launch_a();
        launch_b();
    }
    stream_a.synchronize();
    stream_b.synchronize();

    cudaEvent_t start{};
    cudaEvent_t peer_done{};
    cudaEvent_t stop{};
    checkCuda(cudaEventCreate(&start), "cudaEventCreate(concurrent_start)");
    checkCuda(cudaEventCreateWithFlags(&peer_done, cudaEventDisableTiming), "cudaEventCreate(peer_done)");
    checkCuda(cudaEventCreate(&stop), "cudaEventCreate(concurrent_stop)");

    std::vector<double> pair_samples;
    pair_samples.reserve(TIMING_SAMPLES);
    for (int sample = 0; sample < TIMING_SAMPLES; ++sample) {
        checkCuda(cudaEventRecord(start, stream_a.getStream()), "cudaEventRecord(concurrent_start)");
        checkCuda(cudaStreamWaitEvent(stream_b.getStream(), start, 0), "cudaStreamWaitEvent(peer_start)");
        for (int iteration = 0; iteration < ITERATIONS_PER_SAMPLE; ++iteration) {
            launch_a();
            launch_b();
        }
        checkCuda(cudaEventRecord(peer_done, stream_b.getStream()), "cudaEventRecord(peer_done)");
        checkCuda(cudaStreamWaitEvent(stream_a.getStream(), peer_done, 0), "cudaStreamWaitEvent(peer_done)");
        checkCuda(cudaEventRecord(stop, stream_a.getStream()), "cudaEventRecord(concurrent_stop)");
        checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize(concurrent_stop)");
        float elapsed_ms = 0.0f;
        checkCuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime(concurrent)");
        pair_samples.push_back(static_cast<double>(elapsed_ms) / ITERATIONS_PER_SAMPLE);
    }

    checkCuda(cudaEventDestroy(start), "cudaEventDestroy(concurrent_start)");
    checkCuda(cudaEventDestroy(peer_done), "cudaEventDestroy(peer_done)");
    checkCuda(cudaEventDestroy(stop), "cudaEventDestroy(concurrent_stop)");
    std::sort(pair_samples.begin(), pair_samples.end());
    const double median_pair = pair_samples[pair_samples.size() / 2];
    const double best_pair = pair_samples.front();
    return DualStreamTimingResult{median_pair, best_pair, median_pair / 2.0, best_pair / 2.0};
}

std::shared_ptr<StampedExecutionPlan> stampBadLeftToRightChain(const std::vector<Tensor>& inputs, Stream& stream) {
    if (inputs.size() < 3) {
        throw std::invalid_argument("Bad-order reference requires at least three chain operands.");
    }
    std::vector<Expression> expressions;
    expressions.reserve(inputs.size());
    std::unordered_map<std::string, Tensor> bindings;
    for (size_t i = 0; i < inputs.size(); ++i) {
        const std::string name = "input_" + std::to_string(i);
        expressions.push_back(Expression::input(name));
        bindings.emplace(name, inputs[i]);
    }

    Expression result = expressions.front();
    for (size_t i = 1; i < expressions.size(); ++i) {
        result = Expression::matmul(result, expressions[i], false, false, DataType::FP32, DataType::FP32);
    }
    FusedEquation equation = FusedEquation::compile(Expression::outputs({{"output", result}}).physicalOutputs(), stream.getGpuNum());
    StampedExecutionPlan stamped = equation.stamp(bindings, stream);
    return std::make_shared<StampedExecutionPlan>(std::move(stamped));
}

std::shared_ptr<StampedExecutionPlan> stampSingleMatmul(const Tensor& lhs, const Tensor& rhs, Stream& stream) {
    const Expression lhs_expr = Expression::input("lhs", DataType::FP32, DataType::FP32);
    const Expression rhs_expr = Expression::input("rhs", DataType::FP32, DataType::FP32);
    const Expression output_expr = Expression::matmul(lhs_expr, rhs_expr, false, false, DataType::FP32, DataType::FP32);
    FusedEquation equation =
        FusedEquation::compile(Expression::outputs({{"output", output_expr}}).physicalOutputs(), stream.getGpuNum());
    StampedExecutionPlan stamped = equation.stamp({{"lhs", lhs}, {"rhs", rhs}}, stream);
    return std::make_shared<StampedExecutionPlan>(std::move(stamped));
}

std::vector<uint64_t> makeAlternatingDimensions(size_t operand_count, const DimensionProfile& profile) {
    std::vector<uint64_t> dimensions;
    dimensions.reserve(operand_count + 1);
    for (size_t i = 0; i <= operand_count; ++i) {
        dimensions.push_back((i % 2 == 0) ? profile.outer_dimension : profile.inner_dimension);
    }
    return dimensions;
}

std::string dimensionsText(const std::vector<uint64_t>& dimensions) {
    std::ostringstream out;
    for (size_t i = 0; i < dimensions.size(); ++i) {
        if (i != 0) out << 'x';
        out << dimensions[i];
    }
    return out.str();
}

std::string selectedTreeText(const EinsumExactContractionPlan& exact) {
    std::ostringstream out;
    for (size_t i = 0; i < exact.steps.size(); ++i) {
        if (i != 0) out << ';';
        const EinsumExactContractionStep& step = exact.steps[i];
        out << step.lhs_source_mask << '+' << step.rhs_source_mask << "->" << step.result_source_mask;
    }
    return out.str();
}

uint64_t checkedMultiply(uint64_t lhs, uint64_t rhs, const char* description) {
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
        throw std::overflow_error(std::string(description) + " overflows uint64_t.");
    }
    return lhs * rhs;
}

uint64_t genericBroadcastElements(const BenchmarkCase& benchmark_case) {
    uint64_t elements = 1;
    for (uint64_t dimension : benchmark_case.chain_dimensions) {
        elements = checkedMultiply(elements, dimension, "generic broadcast element estimate");
    }
    return elements;
}

long double elementsToMiB(uint64_t elements) {
    return static_cast<long double>(elements) * sizeof(float) / (1024.0L * 1024.0L);
}

double smPressureProxy(float waves_count) {
    return std::clamp(static_cast<double>(waves_count), 0.0, 1.0);
}

double pickerTflops(const StampedMatmulKernelDiagnostic& kernel) {
    if (!kernel.has_measured_kernel || kernel.picker_runtime_ms <= 0.0) {
        return 0.0;
    }
    return static_cast<double>(kernel.flop_count) / (kernel.picker_runtime_ms * 1.0e9);
}

MatmulResourceSummary summarizeMatmuls(const std::vector<StampedMatmulStageDiagnostic>& diagnostics) {
    MatmulResourceSummary summary;
    summary.stage_count = diagnostics.size();
    for (const StampedMatmulStageDiagnostic& diagnostic : diagnostics) {
        const StampedMatmulKernelDiagnostic& kernel = diagnostic.kernel;
        if (!kernel.has_measured_kernel) {
            continue;
        }
        ++summary.measured_stage_count;
        summary.picker_runtime_sum_ms += kernel.picker_runtime_ms;
        const double pressure = smPressureProxy(kernel.waves_count);
        summary.full_sm_time_proxy_ms += kernel.picker_runtime_ms * pressure;
        summary.max_waves = std::max(summary.max_waves, static_cast<double>(kernel.waves_count));
        summary.max_sm_pressure_proxy = std::max(summary.max_sm_pressure_proxy, pressure);
    }
    return summary;
}

void printStageKinds(const char* prefix, const std::vector<std::string>& stages) {
    std::cout << prefix;
    for (const std::string& stage : stages) std::cout << stage << ' ';
    std::cout << '\n';
}

void printMatmulDiagnostics(const char* prefix, const std::vector<StampedMatmulStageDiagnostic>& diagnostics) {
    for (const StampedMatmulStageDiagnostic& diagnostic : diagnostics) {
        const StampedMatmulKernelDiagnostic& kernel = diagnostic.kernel;
        std::cout << "# " << prefix
                  << " stage=" << diagnostic.stage_index
                  << " lane=" << diagnostic.lane_index
                  << " deps=" << diagnostic.dependency_count
                  << " m=" << kernel.m
                  << " n=" << kernel.n
                  << " k=" << kernel.k
                  << " batch=" << kernel.batch_count
                  << " flops=" << kernel.flop_count
                  << " workspace_bytes=" << kernel.workspace_bytes;
        if (kernel.has_measured_kernel) {
            std::cout << std::fixed << std::setprecision(6)
                      << " waves=" << kernel.waves_count
                      << " sm_pressure_proxy=" << smPressureProxy(kernel.waves_count)
                      << " picker_ms=" << kernel.picker_runtime_ms
                      << " picker_tflops=" << pickerTflops(kernel)
                      << " algo_id=" << kernel.algorithm_id;
        } else {
            std::cout << " waves=NA sm_pressure_proxy=NA picker_ms=NA picker_tflops=NA algo_id=NA";
        }
        std::cout << '\n';
    }
}

CalibrationResult runCase(const BenchmarkCase& benchmark_case,
                          Stream& stream,
                          double max_generic_mib,
                          bool verbose_plan,
                          bool measure_dual_stream) {
    if (benchmark_case.chain_dimensions.size() < 4) {
        throw std::invalid_argument("Exact-planner benchmark chain requires at least three operands.");
    }
    std::vector<Tensor> inputs;
    inputs.reserve(benchmark_case.chain_dimensions.size() - 1);
    for (size_t i = 0; i + 1 < benchmark_case.chain_dimensions.size(); ++i) {
        inputs.push_back(makeInput({benchmark_case.chain_dimensions[i], benchmark_case.chain_dimensions[i + 1]},
                                   static_cast<uint32_t>(i + 1),
                                   stream));
    }
    stream.synchronize();

    Einsum operation(benchmark_case.equation);
    const std::shared_ptr<StampedEinsum> exact = operation.stamp(inputs, stream);
    const std::shared_ptr<StampedExecutionPlan> bad = stampBadLeftToRightChain(inputs, stream);
    if (exact->getExecutionPath() != EinsumExecutionPath::EXACT_CONTRACTION || !exact->getPlan().exact_contraction) {
        throw std::runtime_error("Benchmark case did not select exact contraction execution.");
    }

    const uint64_t generic_elements = genericBroadcastElements(benchmark_case);
    const long double generic_mib = elementsToMiB(generic_elements);
    const bool measure_generic = max_generic_mib > 0.0 && generic_mib <= static_cast<long double>(max_generic_mib);
    std::shared_ptr<StampedEinsum> generic;
    if (measure_generic) {
        generic = operation.stampGenericReference(inputs, stream);
    }

    exact->run();
    bad->run();
    if (generic) generic->run();
    stream.synchronize();

    const ValidationResult bad_validation =
        validateClose(exact->getOutputTensor(), bad->output("output"), stream, "exact", "bad_left_to_right");
    std::optional<ValidationResult> generic_validation;
    if (generic) {
        generic_validation =
            validateClose(exact->getOutputTensor(), generic->getOutputTensor(), stream, "exact", "generic");
    }

    const EinsumExactContractionPlan& exact_plan = *exact->getPlan().exact_contraction;
    CalibrationResult result;
    result.benchmark_case = benchmark_case;
    result.selected_tree = selectedTreeText(exact_plan);
    result.planner_cost = exact_plan.cost;
    result.generic_broadcast_mib = generic_mib;
    result.generic_measured = static_cast<bool>(generic);
    result.exact_matmuls = exact->getExpressionMatmulStageDiagnostics();
    result.bad_matmuls = bad->matmulStageDiagnostics();
    result.exact_resources = summarizeMatmuls(result.exact_matmuls);
    result.bad_resources = summarizeMatmuls(result.bad_matmuls);

    std::cout << "\n# case=" << benchmark_case.name
              << " family=" << benchmark_case.family
              << " size=" << benchmark_case.size
              << " equation=" << benchmark_case.equation
              << " dimensions=" << dimensionsText(benchmark_case.chain_dimensions) << '\n';
    std::cout << "# selected_tree=" << result.selected_tree
              << " planner_estimated=" << exact_plan.cost.estimated_execution_units
              << " fma=" << exact_plan.cost.matmul_fma_count
              << " fused=" << exact_plan.cost.fused_elementwise_count
              << " reduction=" << exact_plan.cost.reduction_input_elements
              << " materialization=" << exact_plan.cost.materialization_elements
              << " writes=" << exact_plan.cost.result_write_elements
              << " gemm_groups=" << exact_plan.cost.matmul_group_count
              << " fused_ops=" << exact_plan.cost.fused_kernel_count
              << " reduction_ops=" << exact_plan.cost.reduction_op_count
              << " materialization_ops=" << exact_plan.cost.materialization_op_count
              << " peak_temp=" << exact_plan.cost.peak_temporary_elements
              << " peak_intermediate=" << exact_plan.cost.peak_intermediate_elements << '\n';
    const char* generic_status = generic ? "measured" : (max_generic_mib <= 0.0 ? "disabled" : "skipped_over_cap");
    std::cout << std::fixed << std::setprecision(3)
              << "# generic_broadcast_estimate_mib=" << static_cast<double>(generic_mib)
              << " generic_status=" << generic_status
              << " max_generic_mib=" << max_generic_mib << '\n';
    std::cout << std::setprecision(6)
              << "# validation_bad_max_abs=" << bad_validation.max_abs_error
              << " validation_bad_scaled=" << bad_validation.max_scaled_error;
    if (generic_validation) {
        std::cout << " validation_generic_max_abs=" << generic_validation->max_abs_error
                  << " validation_generic_scaled=" << generic_validation->max_scaled_error;
    }
    std::cout << '\n';

    if (verbose_plan) {
        std::cout << EinsumPlanner::describeExactContraction(exact->getPlan()) << '\n';
    }
    printStageKinds("# exact_stages=", exact->getExpressionStageKindNames());
    if (generic) printStageKinds("# generic_stages=", generic->getExpressionStageKindNames());
    printStageKinds("# bad_left_to_right_stages=", bad->stageKindNames());
    printMatmulDiagnostics("exact_matmul", result.exact_matmuls);
    printMatmulDiagnostics("bad_matmul", result.bad_matmuls);
    std::cout << "# resource_proxy_note=sm_pressure_proxy=min(1,waves); full_sm_time_proxy_ms=sum(picker_ms*sm_pressure_proxy)"
              << '\n';
    std::cout << "# exact_resource_summary picker_ms_sum=" << result.exact_resources.picker_runtime_sum_ms
              << " full_sm_time_proxy_ms=" << result.exact_resources.full_sm_time_proxy_ms
              << " max_waves=" << result.exact_resources.max_waves
              << " max_sm_pressure_proxy=" << result.exact_resources.max_sm_pressure_proxy << '\n';
    std::cout << "# bad_resource_summary picker_ms_sum=" << result.bad_resources.picker_runtime_sum_ms
              << " full_sm_time_proxy_ms=" << result.bad_resources.full_sm_time_proxy_ms
              << " max_waves=" << result.bad_resources.max_waves
              << " max_sm_pressure_proxy=" << result.bad_resources.max_sm_pressure_proxy << '\n';

    result.exact_timing = timeLaunch([&] { exact->runOn(stream); }, stream);
    if (generic) result.generic_timing = timeLaunch([&] { generic->runOn(stream); }, stream);
    result.bad_timing = timeLaunch([&] { bad->runOn(stream); }, stream);

    if (measure_dual_stream) {
        Stream peer_stream(stream.getGpuNum());
        const std::shared_ptr<StampedEinsum> exact_peer = operation.stamp(inputs, peer_stream);
        const std::shared_ptr<StampedExecutionPlan> bad_peer = stampBadLeftToRightChain(inputs, peer_stream);
        result.exact_dual = timeConcurrentPair([&] { exact->runOn(stream); },
                                               stream,
                                               [&] { exact_peer->runOn(peer_stream); },
                                               peer_stream);
        result.bad_dual = timeConcurrentPair([&] { bad->runOn(stream); },
                                             stream,
                                             [&] { bad_peer->runOn(peer_stream); },
                                             peer_stream);
    }

    std::cout << "strategy,median_ms,best_ms,relative_to_exact,dual_per_plan_ms,dual_throughput_scale\n"
              << "selected_exact," << result.exact_timing.median_ms << ',' << result.exact_timing.best_ms << ",1,";
    if (result.exact_dual) {
        std::cout << result.exact_dual->median_per_plan_ms << ','
                  << result.exact_timing.median_ms / result.exact_dual->median_per_plan_ms << '\n';
    } else {
        std::cout << "NA,NA\n";
    }
    if (generic) {
        std::cout << "whole_equation_generic," << result.generic_timing.median_ms << ',' << result.generic_timing.best_ms << ','
                  << result.generic_timing.median_ms / result.exact_timing.median_ms << ",NA,NA\n";
    } else {
        std::cout << "whole_equation_generic,NA,NA,NA,NA,NA\n";
    }
    std::cout << "bad_left_to_right," << result.bad_timing.median_ms << ',' << result.bad_timing.best_ms << ','
              << result.bad_timing.median_ms / result.exact_timing.median_ms << ',';
    if (result.bad_dual) {
        std::cout << result.bad_dual->median_per_plan_ms << ','
                  << result.bad_timing.median_ms / result.bad_dual->median_per_plan_ms << '\n';
    } else {
        std::cout << "NA,NA\n";
    }
    return result;
}

void printCalibrationSummary(const std::vector<CalibrationResult>& results) {
    std::cout << "\n# calibration_summary\n"
              << "family,size,dimensions,selected_tree,planner_estimated,fma,fused,reduction,materialization,writes,"
                 "gemm_groups,fused_ops,reduction_ops,materialization_ops,"
                 "peak_temp,peak_intermediate,generic_broadcast_mib,exact_picker_ms_sum,exact_full_sm_time_proxy_ms,"
                 "exact_max_waves,bad_picker_ms_sum,bad_full_sm_time_proxy_ms,bad_max_waves,exact_median_ms,"
                 "exact_dual_per_plan_ms,exact_dual_throughput_scale,generic_median_ms,generic_relative,bad_median_ms,"
                 "bad_relative,bad_dual_per_plan_ms,bad_dual_throughput_scale\n";
    for (const CalibrationResult& result : results) {
        const EinsumExactContractionCost& cost = result.planner_cost;
        std::cout << result.benchmark_case.family << ','
                  << result.benchmark_case.size << ','
                  << dimensionsText(result.benchmark_case.chain_dimensions) << ','
                  << result.selected_tree << ','
                  << cost.estimated_execution_units << ','
                  << cost.matmul_fma_count << ','
                  << cost.fused_elementwise_count << ','
                  << cost.reduction_input_elements << ','
                  << cost.materialization_elements << ','
                  << cost.result_write_elements << ','
                  << cost.matmul_group_count << ','
                  << cost.fused_kernel_count << ','
                  << cost.reduction_op_count << ','
                  << cost.materialization_op_count << ','
                  << cost.peak_temporary_elements << ','
                  << cost.peak_intermediate_elements << ','
                  << static_cast<double>(result.generic_broadcast_mib) << ','
                  << result.exact_resources.picker_runtime_sum_ms << ','
                  << result.exact_resources.full_sm_time_proxy_ms << ','
                  << result.exact_resources.max_waves << ','
                  << result.bad_resources.picker_runtime_sum_ms << ','
                  << result.bad_resources.full_sm_time_proxy_ms << ','
                  << result.bad_resources.max_waves << ','
                  << result.exact_timing.median_ms << ',';
        if (result.exact_dual) {
            std::cout << result.exact_dual->median_per_plan_ms << ','
                      << result.exact_timing.median_ms / result.exact_dual->median_per_plan_ms << ',';
        } else {
            std::cout << "NA,NA,";
        }
        if (result.generic_measured) {
            std::cout << result.generic_timing.median_ms << ','
                      << result.generic_timing.median_ms / result.exact_timing.median_ms << ',';
        } else {
            std::cout << "NA,NA,";
        }
        std::cout << result.bad_timing.median_ms << ','
                  << result.bad_timing.median_ms / result.exact_timing.median_ms << ',';
        if (result.bad_dual) {
            std::cout << result.bad_dual->median_per_plan_ms << ','
                      << result.bad_timing.median_ms / result.bad_dual->median_per_plan_ms << '\n';
        } else {
            std::cout << "NA,NA\n";
        }
    }
}

std::vector<GemmShapeCase> makeGemmShapeCases() {
    const std::vector<uint64_t> outers = {256, 512, 1024, 2048, 4096};
    const std::vector<uint64_t> inners = {16, 32, 48, 64, 80, 96, 128, 192, 256};
    std::vector<GemmShapeCase> cases;
    cases.reserve(outers.size() * inners.size() * 4);
    for (uint64_t outer : outers) {
        for (uint64_t inner : inners) {
            cases.push_back({"bottleneck_contract", outer, inner, inner, inner, outer});
            cases.push_back({"skinny_expand", outer, inner, outer, inner, inner});
            cases.push_back({"wide_expand", outer, inner, outer, outer, inner});
            cases.push_back({"wide_reduce", outer, inner, outer, inner, outer});
        }
    }
    return cases;
}

GemmShapeResult runGemmShapeCase(const GemmShapeCase& shape, Stream& stream, bool measure_dual_stream) {
    Tensor lhs = makeInput({shape.m, shape.k}, 101, stream);
    Tensor rhs = makeInput({shape.k, shape.n}, 211, stream);
    stream.synchronize();
    const std::shared_ptr<StampedExecutionPlan> plan = stampSingleMatmul(lhs, rhs, stream);
    const std::vector<StampedMatmulStageDiagnostic> diagnostics = plan->matmulStageDiagnostics();
    if (diagnostics.size() != 1) {
        throw std::runtime_error("Focused GEMM shape benchmark expected exactly one Matmul stage.");
    }

    plan->runOn(stream);
    stream.synchronize();

    GemmShapeResult result;
    result.shape = shape;
    result.diagnostic = diagnostics.front();
    result.isolated = timeLaunch([&] { plan->runOn(stream); }, stream);

    if (measure_dual_stream) {
        Stream peer_stream(stream.getGpuNum());
        const std::shared_ptr<StampedExecutionPlan> peer_plan = stampSingleMatmul(lhs, rhs, peer_stream);
        result.dual = timeConcurrentPair([&] { plan->runOn(stream); },
                                         stream,
                                         [&] { peer_plan->runOn(peer_stream); },
                                         peer_stream);
    }

    const StampedMatmulKernelDiagnostic& kernel = result.diagnostic.kernel;
    std::cout << std::fixed << std::setprecision(6)
              << "shape_kind=" << shape.kind
              << " outer=" << shape.outer
              << " inner=" << shape.inner
              << " m=" << shape.m
              << " n=" << shape.n
              << " k=" << shape.k
              << " waves=" << (kernel.has_measured_kernel ? std::to_string(kernel.waves_count) : "NA")
              << " sm_pressure_proxy=" << (kernel.has_measured_kernel ? std::to_string(smPressureProxy(kernel.waves_count)) : "NA")
              << " picker_ms=" << (kernel.has_measured_kernel ? std::to_string(kernel.picker_runtime_ms) : "NA")
              << " picker_tflops=" << (kernel.has_measured_kernel ? std::to_string(pickerTflops(kernel)) : "NA")
              << " isolated_ms=" << result.isolated.median_ms
              << " dual_per_plan_ms=";
    if (result.dual) {
        std::cout << result.dual->median_per_plan_ms
                  << " dual_throughput_scale=" << result.isolated.median_ms / result.dual->median_per_plan_ms;
    } else {
        std::cout << "NA dual_throughput_scale=NA";
    }
    std::cout << " workspace_bytes=" << kernel.workspace_bytes
              << " algo_id=" << (kernel.has_measured_kernel ? std::to_string(kernel.algorithm_id) : "NA") << '\n';
    return result;
}

void printGemmShapeSummary(const std::vector<GemmShapeResult>& results) {
    std::cout << "\n# gemm_shape_summary\n"
              << "kind,outer,inner,m,n,k,flops,waves,sm_pressure_proxy,picker_ms,picker_tflops,workspace_bytes,algo_id,"
                 "isolated_median_ms,dual_per_plan_ms,dual_throughput_scale\n";
    for (const GemmShapeResult& result : results) {
        const StampedMatmulKernelDiagnostic& kernel = result.diagnostic.kernel;
        std::cout << result.shape.kind << ','
                  << result.shape.outer << ','
                  << result.shape.inner << ','
                  << result.shape.m << ','
                  << result.shape.n << ','
                  << result.shape.k << ','
                  << kernel.flop_count << ',';
        if (kernel.has_measured_kernel) {
            std::cout << kernel.waves_count << ','
                      << smPressureProxy(kernel.waves_count) << ','
                      << kernel.picker_runtime_ms << ','
                      << pickerTflops(kernel) << ','
                      << kernel.workspace_bytes << ','
                      << kernel.algorithm_id << ',';
        } else {
            std::cout << "NA,NA,NA,NA," << kernel.workspace_bytes << ",NA,";
        }
        std::cout << result.isolated.median_ms << ',';
        if (result.dual) {
            std::cout << result.dual->median_per_plan_ms << ','
                      << result.isolated.median_ms / result.dual->median_per_plan_ms << '\n';
        } else {
            std::cout << "NA,NA\n";
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        int device = 0;
        std::string case_filter;
        std::string size_filter;
        std::string shape_kind_filter;
        std::optional<uint64_t> outer_filter;
        std::optional<uint64_t> inner_filter;
        double max_generic_mib = DEFAULT_MAX_GENERIC_MIB;
        bool verbose_plan = false;
        bool gemm_shape_sweep = false;
        bool measure_dual_stream = true;
        for (int i = 1; i < argc; ++i) {
            const std::string_view arg(argv[i]);
            if (arg == "--help") {
                std::cout << "thor_einsum_exact_planner_benchmark options:\n"
                          << "  --device=N             CUDA device (default 0)\n"
                          << "  --case=SUBSTRING       Run only matching chain families/cases\n"
                          << "  --size=SUBSTRING       Run only matching size profiles\n"
                          << "  --max-generic-mib=N    Skip generic reference above N MiB estimated broadcast tensor (default 256)\n"
                          << "  --verbose-plan         Print full physical exact-plan diagnostics per case\n"
                          << "  --no-dual-stream       Disable two-copy concurrent throughput measurements\n"
                          << "  --gemm-shape-sweep     Run focused cuBLASLt GEMM shape calibration instead of chain cases\n"
                          << "  --shape-kind=SUBSTRING Filter focused GEMM shape kind\n"
                          << "  --outer=N              Filter focused GEMM sweep outer dimension\n"
                          << "  --inner=N              Filter focused GEMM sweep inner dimension\n";
                return EXIT_SUCCESS;
            }
            if (arg.starts_with("--device=")) {
                device = std::stoi(std::string(arg.substr(9)));
            } else if (arg.starts_with("--case=")) {
                case_filter = std::string(arg.substr(7));
            } else if (arg.starts_with("--size=")) {
                size_filter = std::string(arg.substr(7));
            } else if (arg.starts_with("--max-generic-mib=")) {
                max_generic_mib = std::stod(std::string(arg.substr(18)));
                if (max_generic_mib < 0.0) {
                    throw std::invalid_argument("--max-generic-mib must be non-negative.");
                }
            } else if (arg == "--verbose-plan") {
                verbose_plan = true;
            } else if (arg == "--no-dual-stream") {
                measure_dual_stream = false;
            } else if (arg == "--gemm-shape-sweep") {
                gemm_shape_sweep = true;
            } else if (arg.starts_with("--shape-kind=")) {
                shape_kind_filter = std::string(arg.substr(13));
            } else if (arg.starts_with("--outer=")) {
                outer_filter = std::stoull(std::string(arg.substr(8)));
            } else if (arg.starts_with("--inner=")) {
                inner_filter = std::stoull(std::string(arg.substr(8)));
            } else {
                throw std::invalid_argument("Unknown argument: " + std::string(arg));
            }
        }

        checkCuda(cudaSetDevice(device), "cudaSetDevice");
        Stream stream(device);

        if (gemm_shape_sweep) {
            std::vector<GemmShapeResult> results;
            for (const GemmShapeCase& shape : makeGemmShapeCases()) {
                if (!shape_kind_filter.empty() &&
                    std::string_view(shape.kind).find(shape_kind_filter) == std::string_view::npos) {
                    continue;
                }
                if (outer_filter && shape.outer != *outer_filter) {
                    continue;
                }
                if (inner_filter && shape.inner != *inner_filter) {
                    continue;
                }
                results.push_back(runGemmShapeCase(shape, stream, measure_dual_stream));
            }
            if (results.empty()) {
                throw std::invalid_argument("No focused GEMM shape case matched the requested filters.");
            }
            printGemmShapeSummary(results);
            return EXIT_SUCCESS;
        }

        const std::vector<BenchmarkFamily> families = {
            {"three_operand_right_to_left", "ab,bc,cd->ad", 3},
            {"four_operand_alternating", "ab,bc,cd,de->ae", 4},
            {"five_operand_alternating", "ab,bc,cd,de,ef->af", 5},
        };
        const std::vector<DimensionProfile> profiles = {
            {"tiny", 100, 2},
            {"small", 256, 16},
            {"medium", 1024, 64},
            {"large", 2048, 128},
            {"xlarge", 4096, 256},
        };

        std::vector<CalibrationResult> results;
        for (const BenchmarkFamily& family : families) {
            for (const DimensionProfile& profile : profiles) {
                BenchmarkCase benchmark_case;
                benchmark_case.name = family.name + "_" + profile.name;
                benchmark_case.family = family.name;
                benchmark_case.size = profile.name;
                benchmark_case.equation = family.equation;
                benchmark_case.chain_dimensions = makeAlternatingDimensions(family.operand_count, profile);

                if (!case_filter.empty() &&
                    std::string_view(benchmark_case.name).find(case_filter) == std::string_view::npos &&
                    std::string_view(benchmark_case.family).find(case_filter) == std::string_view::npos) {
                    continue;
                }
                if (!size_filter.empty() && std::string_view(benchmark_case.size).find(size_filter) == std::string_view::npos) {
                    continue;
                }
                results.push_back(runCase(benchmark_case, stream, max_generic_mib, verbose_plan, measure_dual_stream));
            }
        }
        if (results.empty()) {
            throw std::invalid_argument("No benchmark case matched the requested filters.");
        }
        printCalibrationSummary(results);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
