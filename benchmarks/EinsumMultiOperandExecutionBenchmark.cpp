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

constexpr double DEFAULT_MAX_GENERIC_MIB = 256.0;
constexpr double VALIDATION_ABSOLUTE_TOLERANCE = 2.0e-3;
constexpr double VALIDATION_RELATIVE_TOLERANCE = 1.0e-2;
constexpr uint64_t LARGE_CHAIN_MIN_MATMUL_FMAS = 64ULL * 1024ULL * 1024ULL;

struct Profile {
    std::string name;
    uint64_t outer = 0;
    uint64_t inner = 0;
    uint64_t batch = 0;
    uint64_t local_reduction = 0;
};

struct BenchmarkCase {
    std::string name;
    std::string family;
    std::string equation;
    std::vector<std::vector<uint64_t>> input_dimensions;
    bool supports_bad_left_to_right = false;
};

struct Options {
    int device = 0;
    std::string profile = "large";
    std::string case_filter;
    int warmup_iterations = 3;
    int timing_samples = 5;
    int iterations_per_sample = 4;
    double max_generic_mib = DEFAULT_MAX_GENERIC_MIB;
    bool verbose_plan = false;
};

struct TimingResult {
    double median_ms = 0.0;
    double best_ms = 0.0;
    double worst_ms = 0.0;
};

struct ValidationResult {
    double max_abs_error = 0.0;
    double max_scaled_error = 0.0;
};

struct CaseResult {
    BenchmarkCase benchmark_case;
    std::string planning_mode;
    std::string execution_path;
    EinsumExactContractionCost cost;
    long double generic_broadcast_mib = 0.0L;
    bool generic_measured = false;
    TimingResult selected_timing;
    std::optional<TimingResult> bad_timing;
    std::optional<TimingResult> generic_timing;
    size_t stage_count = 0;
    size_t matmul_stage_count = 0;
    size_t reduction_stage_count = 0;
    size_t fused_stage_count = 0;
    size_t helper_lane_matmul_count = 0;
    uint32_t max_matmul_lane = 0;
    uint64_t selected_min_matmul_fmas = 0;
    uint64_t bad_min_matmul_fmas = 0;
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
    const uint64_t fan_in = dimensions.empty() ? 1 : std::max<uint64_t>(1, dimensions.back());
    const float scale = static_cast<float>(0.125 / std::sqrt(static_cast<double>(fan_in)));
    for (uint64_t i = 0; i < cpu.getTotalNumElements(); ++i) {
        const uint32_t mixed = static_cast<uint32_t>(i * 1664525ULL + seed * 1013904223ULL);
        values[i] = static_cast<float>(static_cast<int32_t>(mixed % 17) - 8) * scale;
    }
    Tensor gpu(gpu_placement, cpu.getDescriptor());
    gpu.copyFromAsync(cpu, stream);
    return gpu;
}

std::vector<Tensor> makeInputs(const std::vector<std::vector<uint64_t>>& dimensions, Stream& stream) {
    std::vector<Tensor> inputs;
    inputs.reserve(dimensions.size());
    for (size_t i = 0; i < dimensions.size(); ++i) {
        inputs.push_back(makeInput(dimensions[i], static_cast<uint32_t>(i + 1), stream));
    }
    return inputs;
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

TimingResult timeLaunch(const std::function<void()>& launch, Stream& stream, const Options& options) {
    for (int i = 0; i < options.warmup_iterations; ++i) {
        launch();
    }
    stream.synchronize();

    cudaEvent_t start{};
    cudaEvent_t stop{};
    checkCuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    checkCuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");

    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(options.timing_samples));
    for (int sample = 0; sample < options.timing_samples; ++sample) {
        checkCuda(cudaEventRecord(start, stream.getStream()), "cudaEventRecord(start)");
        for (int iteration = 0; iteration < options.iterations_per_sample; ++iteration) {
            launch();
        }
        checkCuda(cudaEventRecord(stop, stream.getStream()), "cudaEventRecord(stop)");
        checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
        float elapsed_ms = 0.0f;
        checkCuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime");
        samples.push_back(static_cast<double>(elapsed_ms) / options.iterations_per_sample);
    }

    checkCuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
    checkCuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");
    std::sort(samples.begin(), samples.end());
    return TimingResult{
        .median_ms = samples[samples.size() / 2],
        .best_ms = samples.front(),
        .worst_ms = samples.back(),
    };
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
        expressions.push_back(Expression::input(name, DataType::FP32, DataType::FP32));
        bindings.emplace(name, inputs[i]);
    }

    Expression result = expressions.front();
    for (size_t i = 1; i < expressions.size(); ++i) {
        result = Expression::matmul(result, expressions[i], false, false, DataType::FP32, DataType::FP32);
    }
    FusedEquation equation =
        FusedEquation::compile(Expression::outputs({{"output", result}}).physicalOutputs(), stream.getGpuNum());
    StampedExecutionPlan stamped = equation.stamp(bindings, stream);
    return std::make_shared<StampedExecutionPlan>(std::move(stamped));
}

std::vector<std::vector<uint64_t>> makeAlternatingChainDimensions(size_t operand_count,
                                                                  const Profile& profile,
                                                                  bool batched) {
    std::vector<uint64_t> chain_dimensions;
    chain_dimensions.reserve(operand_count + 1);
    for (size_t i = 0; i <= operand_count; ++i) {
        chain_dimensions.push_back((i % 2 == 0) ? profile.outer : profile.inner);
    }

    std::vector<std::vector<uint64_t>> inputs;
    inputs.reserve(operand_count);
    for (size_t i = 0; i < operand_count; ++i) {
        if (batched) {
            inputs.push_back({profile.batch, chain_dimensions[i], chain_dimensions[i + 1]});
        } else {
            inputs.push_back({chain_dimensions[i], chain_dimensions[i + 1]});
        }
    }
    return inputs;
}

Profile profileNamed(std::string_view name) {
    if (name == "tiny") {
        return Profile{"tiny", 32, 8, 2, 2};
    }
    if (name == "small") {
        return Profile{"small", 256, 32, 4, 4};
    }
    if (name == "medium") {
        return Profile{"medium", 1024, 64, 4, 8};
    }
    if (name == "large") {
        // The chain dimensions are intentionally large enough that the preferred
        // alternating-chain plans do not contain the low-utilization 64x64x1024
        // GEMMs that distorted the earlier primitive calibration. The branching
        // fixture uses inner-sized square matrices so its helper-stream timing is
        // also outside that low-utilization regime; only the local-reduction
        // dimension remains intentionally modest.
        return Profile{"large", 2048, 512, 4, 8};
    }
    throw std::invalid_argument("Unknown --profile value '" + std::string(name) +
                                "'. Expected tiny, small, medium, or large.");
}

std::vector<BenchmarkCase> makeCases(const Profile& profile) {
    std::vector<BenchmarkCase> cases;
    cases.push_back({"exact3_matrix_chain",
                     "exact_matrix_chain",
                     "ab,bc,cd->ad",
                     makeAlternatingChainDimensions(3, profile, false),
                     true});
    cases.push_back({"exact4_matrix_chain",
                     "exact_matrix_chain",
                     "ab,bc,cd,de->ae",
                     makeAlternatingChainDimensions(4, profile, false),
                     true});
    cases.push_back({"exact5_matrix_chain",
                     "exact_matrix_chain",
                     "ab,bc,cd,de,ef->af",
                     makeAlternatingChainDimensions(5, profile, false),
                     true});
    cases.push_back({"bridge6_matrix_chain",
                     "bridge_matrix_chain",
                     "ab,bc,cd,de,ef,fg->ag",
                     makeAlternatingChainDimensions(6, profile, false),
                     true});
    cases.push_back({"beam7_matrix_chain",
                     "beam_matrix_chain",
                     "ab,bc,cd,de,ef,fg,gh->ah",
                     makeAlternatingChainDimensions(7, profile, false),
                     true});
    cases.push_back({"beam10_matrix_chain",
                     "beam_matrix_chain",
                     "ab,bc,cd,de,ef,fg,gh,hi,ij,jk->ak",
                     makeAlternatingChainDimensions(10, profile, false),
                     true});
    cases.push_back({"beam7_batched_chain",
                     "beam_batched_chain",
                     "zab,zbc,zcd,zde,zef,zfg,zgh->zah",
                     makeAlternatingChainDimensions(7, profile, true),
                     true});

    cases.push_back({
        "beam7_branching",
        "beam_branching",
        // Two matrix paths from b to e form a genuine branching/cyclic tensor
        // network.  With the large profile every operand is 512x512, so helper-
        // stream concurrency is measured with substantial GEMMs instead of the
        // 32x32x512 GEMM produced by the earlier connector-tensor fixture.
        "ab,bc,cd,de,ef,fg,bg->ae",
        {
            {profile.inner, profile.inner},
            {profile.inner, profile.inner},
            {profile.inner, profile.inner},
            {profile.inner, profile.inner},
            {profile.inner, profile.inner},
            {profile.inner, profile.inner},
            {profile.inner, profile.inner},
        },
        false,
    });

    cases.push_back({
        "beam7_reduction_gemm",
        "beam_reduction_gemm",
        "bxij,bjk,kl,lm,mn,no,op->bip",
        {
            {profile.batch, profile.local_reduction, profile.outer, profile.inner},
            {profile.batch, profile.inner, profile.inner},
            {profile.inner, profile.inner},
            {profile.inner, profile.inner},
            {profile.inner, profile.inner},
            {profile.inner, profile.inner},
            {profile.inner, profile.outer},
        },
        false,
    });
    return cases;
}

std::vector<std::vector<uint64_t>> makeTinyValidationDimensions(const BenchmarkCase& benchmark_case) {
    std::vector<std::vector<uint64_t>> dimensions;
    dimensions.reserve(benchmark_case.input_dimensions.size());
    for (const auto& input : benchmark_case.input_dimensions) {
        dimensions.emplace_back(input.size(), 2);
    }
    return dimensions;
}

long double genericBroadcastMiB(const EinsumPlan& plan) {
    long double elements = 1.0L;
    for (uint64_t dimension : plan.iteration_dimensions) {
        elements *= static_cast<long double>(dimension);
    }
    return elements * sizeof(float) / (1024.0L * 1024.0L);
}

const EinsumExactContractionCost& selectedCost(const EinsumPlan& plan) {
    if (plan.beam_contraction.has_value()) {
        return plan.beam_contraction->cost;
    }
    if (plan.exact_contraction.has_value()) {
        return plan.exact_contraction->cost;
    }
    throw std::runtime_error("Multi-operand execution benchmark expected a contraction-tree plan.");
}

std::string planningModeName(const EinsumPlan& plan) {
    if (plan.beam_contraction.has_value()) {
        return "beam";
    }
    if (!plan.exact_contraction.has_value()) {
        return "none";
    }
    return plan.exact_contraction->planning_mode == EinsumContractionPlanningMode::SIX_OPERAND_BRIDGE
               ? "six_operand_bridge"
               : "exact";
}

const char* executionPathName(EinsumExecutionPath path) {
    switch (path) {
        case EinsumExecutionPath::GENERIC: return "generic";
        case EinsumExecutionPath::GEMM: return "gemm";
        case EinsumExecutionPath::BATCHED_GEMM: return "batched_gemm";
        case EinsumExecutionPath::PAIR_PRODUCT: return "pair_product";
        case EinsumExecutionPath::EXACT_CONTRACTION: return "exact_contraction";
        case EinsumExecutionPath::BEAM_CONTRACTION: return "beam_contraction";
    }
    return "unknown";
}

void validateOptimizedAgainstTinyGeneric(const BenchmarkCase& benchmark_case, Stream& stream) {
    const auto tiny_dimensions = makeTinyValidationDimensions(benchmark_case);
    std::vector<Tensor> tiny_inputs = makeInputs(tiny_dimensions, stream);
    stream.synchronize();

    Einsum operation(benchmark_case.equation);
    const std::shared_ptr<StampedEinsum> optimized = operation.stamp(tiny_inputs, stream);
    const std::shared_ptr<StampedEinsum> generic = operation.stampGenericReference(tiny_inputs, stream);
    optimized->run();
    generic->run();
    stream.synchronize();
    validateClose(optimized->getOutputTensor(), generic->getOutputTensor(), stream, "optimized_tiny", "generic_tiny");
}

CaseResult runCase(const BenchmarkCase& benchmark_case, Stream& stream, const Options& options) {
    validateOptimizedAgainstTinyGeneric(benchmark_case, stream);

    std::vector<Tensor> inputs = makeInputs(benchmark_case.input_dimensions, stream);
    stream.synchronize();

    Einsum operation(benchmark_case.equation);
    const std::shared_ptr<StampedEinsum> selected = operation.stamp(inputs, stream);
    const EinsumPlan& plan = selected->getPlan();
    if (inputs.size() <= EinsumPlanner::MAX_BRIDGED_ACTIVE_OPERANDS) {
        if (selected->getExecutionPath() != EinsumExecutionPath::EXACT_CONTRACTION) {
            throw std::runtime_error(benchmark_case.name + " did not lower through exact/bridge contraction execution.");
        }
    } else if (selected->getExecutionPath() != EinsumExecutionPath::BEAM_CONTRACTION) {
        throw std::runtime_error(benchmark_case.name + " did not lower through beam contraction execution.");
    }

    std::shared_ptr<StampedExecutionPlan> bad;
    if (benchmark_case.supports_bad_left_to_right) {
        bad = stampBadLeftToRightChain(inputs, stream);
    }

    const long double generic_mib = genericBroadcastMiB(plan);
    std::shared_ptr<StampedEinsum> generic;
    if (options.max_generic_mib > 0.0 && generic_mib <= static_cast<long double>(options.max_generic_mib)) {
        generic = operation.stampGenericReference(inputs, stream);
    }

    selected->run();
    if (bad) bad->run();
    if (generic) generic->run();
    stream.synchronize();

    if (bad) {
        validateClose(selected->getOutputTensor(), bad->output("output"), stream, "selected", "bad_left_to_right");
    }
    if (generic) {
        validateClose(selected->getOutputTensor(), generic->getOutputTensor(), stream, "selected", "generic");
    }

    CaseResult result;
    result.benchmark_case = benchmark_case;
    result.planning_mode = planningModeName(plan);
    result.execution_path = executionPathName(selected->getExecutionPath());
    result.cost = selectedCost(plan);
    result.generic_broadcast_mib = generic_mib;
    result.generic_measured = static_cast<bool>(generic);

    const std::vector<std::string> stage_kinds = selected->getExpressionStageKindNames();
    result.stage_count = stage_kinds.size();
    result.matmul_stage_count = static_cast<size_t>(std::count(stage_kinds.begin(), stage_kinds.end(), "Matmul"));
    result.reduction_stage_count = static_cast<size_t>(std::count(stage_kinds.begin(), stage_kinds.end(), "Reduction"));
    result.fused_stage_count = static_cast<size_t>(std::count(stage_kinds.begin(), stage_kinds.end(), "FusedKernel"));
    for (const StampedMatmulStageDiagnostic& diagnostic : selected->getExpressionMatmulStageDiagnostics()) {
        result.max_matmul_lane = std::max(result.max_matmul_lane, diagnostic.lane_index);
        const uint64_t fmas = diagnostic.kernel.flop_count / 2;
        if (result.selected_min_matmul_fmas == 0 || fmas < result.selected_min_matmul_fmas) {
            result.selected_min_matmul_fmas = fmas;
        }
        if (diagnostic.lane_index > 0) {
            ++result.helper_lane_matmul_count;
        }
    }
    if (bad) {
        for (const StampedMatmulStageDiagnostic& diagnostic : bad->matmulStageDiagnostics()) {
            const uint64_t fmas = diagnostic.kernel.flop_count / 2;
            if (result.bad_min_matmul_fmas == 0 || fmas < result.bad_min_matmul_fmas) {
                result.bad_min_matmul_fmas = fmas;
            }
        }
    }

    if (options.profile == "large" &&
        (benchmark_case.supports_bad_left_to_right || benchmark_case.family == "beam_branching")) {
        if (result.selected_min_matmul_fmas < LARGE_CHAIN_MIN_MATMUL_FMAS) {
            throw std::runtime_error(benchmark_case.name +
                                     " large profile produced a selected matmul below the performance floor: " +
                                     std::to_string(result.selected_min_matmul_fmas) + " FMAs.");
        }
        if (benchmark_case.supports_bad_left_to_right &&
            result.bad_min_matmul_fmas < LARGE_CHAIN_MIN_MATMUL_FMAS) {
            throw std::runtime_error(benchmark_case.name +
                                     " large profile produced a bad-order matmul below the performance floor: " +
                                     std::to_string(result.bad_min_matmul_fmas) + " FMAs.");
        }
    }

    if (benchmark_case.family == "beam_branching" && result.helper_lane_matmul_count == 0) {
        throw std::runtime_error("Branching benchmark did not place any matmul stage on an Expression helper lane.");
    }
    if (benchmark_case.family == "beam_reduction_gemm" && result.reduction_stage_count == 0) {
        throw std::runtime_error("Reduction+GEMM benchmark did not contain a centralized reduction stage.");
    }

    if (options.verbose_plan) {
        std::cout << "\n# " << benchmark_case.name << " equation=" << benchmark_case.equation << '\n';
        if (plan.beam_contraction.has_value()) {
            std::cout << EinsumPlanner::describeBeamContraction(plan) << '\n';
        } else {
            std::cout << EinsumPlanner::describeExactContraction(plan) << '\n';
        }
        std::cout << "# stages=";
        for (const std::string& stage : stage_kinds) {
            std::cout << stage << ' ';
        }
        std::cout << '\n';
    }

    result.selected_timing = timeLaunch([&] { selected->runOn(stream); }, stream, options);
    if (bad) {
        result.bad_timing = timeLaunch([&] { bad->runOn(stream); }, stream, options);
    }
    if (generic) {
        result.generic_timing = timeLaunch([&] { generic->runOn(stream); }, stream, options);
    }
    return result;
}

std::string csvNumberOrNa(const std::optional<TimingResult>& timing, double TimingResult::*member) {
    if (!timing.has_value()) {
        return "NA";
    }
    std::ostringstream out;
    out << std::fixed << std::setprecision(6) << (timing.value().*member);
    return out.str();
}

std::string csvRatioOrNa(double numerator, const std::optional<TimingResult>& denominator) {
    if (!denominator.has_value() || numerator <= 0.0) {
        return "NA";
    }
    std::ostringstream out;
    out << std::fixed << std::setprecision(4) << denominator->median_ms / numerator;
    return out.str();
}

void printSummary(const std::vector<CaseResult>& results) {
    std::cout << "family,case,operands,planning_mode,execution_path,selected_median_ms,selected_best_ms,selected_worst_ms,"
                 "bad_left_to_right_median_ms,selected_speedup_vs_bad,generic_status,generic_broadcast_mib,"
                 "generic_median_ms,selected_speedup_vs_generic,estimated_execution_units,peak_intermediate_elements,"
                 "matmul_groups,fused_ops,reduction_ops,materialization_ops,stage_count,matmul_stages,reduction_stages,"
                 "fused_stages,helper_lane_matmuls,max_matmul_lane,selected_min_matmul_fmas,bad_min_matmul_fmas\n";
    for (const CaseResult& result : results) {
        const double generic_mib = static_cast<double>(result.generic_broadcast_mib);
        std::cout << result.benchmark_case.family << ','
                  << result.benchmark_case.name << ','
                  << result.benchmark_case.input_dimensions.size() << ','
                  << result.planning_mode << ','
                  << result.execution_path << ','
                  << std::fixed << std::setprecision(6)
                  << result.selected_timing.median_ms << ','
                  << result.selected_timing.best_ms << ','
                  << result.selected_timing.worst_ms << ','
                  << csvNumberOrNa(result.bad_timing, &TimingResult::median_ms) << ','
                  << csvRatioOrNa(result.selected_timing.median_ms, result.bad_timing) << ','
                  << (result.generic_measured ? "measured" : "skipped_over_cap") << ','
                  << std::setprecision(3) << generic_mib << ','
                  << csvNumberOrNa(result.generic_timing, &TimingResult::median_ms) << ','
                  << csvRatioOrNa(result.selected_timing.median_ms, result.generic_timing) << ','
                  << result.cost.estimated_execution_units << ','
                  << result.cost.peak_intermediate_elements << ','
                  << result.cost.matmul_group_count << ','
                  << result.cost.fused_kernel_count << ','
                  << result.cost.reduction_op_count << ','
                  << result.cost.materialization_op_count << ','
                  << result.stage_count << ','
                  << result.matmul_stage_count << ','
                  << result.reduction_stage_count << ','
                  << result.fused_stage_count << ','
                  << result.helper_lane_matmul_count << ','
                  << result.max_matmul_lane << ','
                  << result.selected_min_matmul_fmas << ','
                  << result.bad_min_matmul_fmas << '\n';
    }
}

Options parseOptions(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "--help") {
            std::cout << "thor_einsum_multi_operand_execution_benchmark options:\n"
                      << "  --device=N             CUDA device (default 0)\n"
                      << "  --profile=NAME         tiny, small, medium, or large (default large)\n"
                      << "  --case=SUBSTRING       Run matching case/family names only\n"
                      << "  --samples=N            Timing samples (default 5)\n"
                      << "  --iterations=N         Launches per sample (default 4)\n"
                      << "  --warmup=N             Warmup launches (default 3)\n"
                      << "  --max-generic-mib=N    Time whole-equation generic only below this broadcast cap (default 256)\n"
                      << "  --verbose-plan         Print selected physical contraction plans and stage kinds\n";
            std::exit(EXIT_SUCCESS);
        }
        if (arg.starts_with("--device=")) {
            options.device = std::stoi(std::string(arg.substr(9)));
        } else if (arg.starts_with("--profile=")) {
            options.profile = std::string(arg.substr(10));
        } else if (arg.starts_with("--case=")) {
            options.case_filter = std::string(arg.substr(7));
        } else if (arg.starts_with("--samples=")) {
            options.timing_samples = std::stoi(std::string(arg.substr(10)));
        } else if (arg.starts_with("--iterations=")) {
            options.iterations_per_sample = std::stoi(std::string(arg.substr(13)));
        } else if (arg.starts_with("--warmup=")) {
            options.warmup_iterations = std::stoi(std::string(arg.substr(9)));
        } else if (arg.starts_with("--max-generic-mib=")) {
            options.max_generic_mib = std::stod(std::string(arg.substr(18)));
        } else if (arg == "--verbose-plan") {
            options.verbose_plan = true;
        } else {
            throw std::invalid_argument("Unknown argument: " + std::string(arg));
        }
    }
    if (options.timing_samples <= 0 || options.iterations_per_sample <= 0 || options.warmup_iterations < 0) {
        throw std::invalid_argument("Timing sample/iteration counts must be positive and warmup must be non-negative.");
    }
    if (options.max_generic_mib < 0.0) {
        throw std::invalid_argument("--max-generic-mib must be non-negative.");
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parseOptions(argc, argv);
        const Profile profile = profileNamed(options.profile);
        checkCuda(cudaSetDevice(options.device), "cudaSetDevice");
        Stream stream(options.device);

        std::cout << "# Thor einsum multi-operand execution benchmark\n"
                  << "# profile=" << profile.name
                  << " outer=" << profile.outer
                  << " inner=" << profile.inner
                  << " batch=" << profile.batch
                  << " local_reduction=" << profile.local_reduction
                  << " planner_beam_width=" << EinsumPlanner::DEFAULT_BEAM_WIDTH
                  << " samples=" << options.timing_samples
                  << " iterations_per_sample=" << options.iterations_per_sample
                  << " warmup=" << options.warmup_iterations
                  << " max_generic_mib=" << options.max_generic_mib << '\n'
                  << "# every case first performs a tiny optimized-vs-generic differential correctness check\n"
                  << "# bad_left_to_right is timed only for pure matrix/batched matrix chains\n";

        std::vector<CaseResult> results;
        for (const BenchmarkCase& benchmark_case : makeCases(profile)) {
            if (!options.case_filter.empty() &&
                std::string_view(benchmark_case.name).find(options.case_filter) == std::string_view::npos &&
                std::string_view(benchmark_case.family).find(options.case_filter) == std::string_view::npos) {
                continue;
            }
            results.push_back(runCase(benchmark_case, stream, options));
        }
        if (results.empty()) {
            throw std::invalid_argument("No benchmark case matched --case filter.");
        }
        printSummary(results);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "thor_einsum_multi_operand_execution_benchmark: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
