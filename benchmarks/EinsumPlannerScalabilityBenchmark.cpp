#include "Utilities/TensorOperations/Einsum/EinsumPlanner.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace ThorImplementation;

namespace {

constexpr uint32_t DEFAULT_TIMING_SAMPLES = 3;
constexpr std::string_view MATRIX_CHAIN_FAMILY = "matrix_chain";
constexpr std::string_view BATCHED_CHAIN_FAMILY = "batched_chain";

struct BenchmarkCase {
    std::string family;
    size_t operand_count = 0;
    std::string equation;
    std::vector<std::vector<uint64_t>> input_dimensions;
};

struct TimingSummary {
    double median_ms = 0.0;
    double best_ms = 0.0;
    double worst_ms = 0.0;
};

struct BenchmarkResult {
    BenchmarkCase benchmark_case;
    uint32_t timing_samples = 0;
    TimingSummary timing;
    EinsumBeamContractionPlan beam;
    std::string plan_description;
};

std::vector<char> regularLabels() {
    std::vector<char> labels;
    labels.reserve(52);
    for (char label = 'A'; label <= 'Z'; ++label) {
        labels.push_back(label);
    }
    for (char label = 'a'; label <= 'z'; ++label) {
        labels.push_back(label);
    }
    return labels;
}

uint64_t chainDimension(size_t boundary_index) {
    // Keep dimensions small enough that 40-operand planning cannot overflow the
    // cost model, while varying them enough to exercise meaningful order choices.
    return 2 + ((boundary_index * 11 + 3) % 19);
}

BenchmarkCase makeMatrixChainCase(size_t operand_count, bool batched) {
    if (operand_count <= EinsumPlanner::MAX_BRIDGED_ACTIVE_OPERANDS) {
        throw std::invalid_argument(
            "Planner scalability cases must contain at least seven operands so beam planning is exercised.");
    }
    if (operand_count > EinsumPlanner::MAX_SOURCE_OPERANDS) {
        throw std::invalid_argument("Requested operand count exceeds EinsumPlanner::MAX_SOURCE_OPERANDS.");
    }

    std::vector<char> labels = regularLabels();
    char batch_label = '\0';
    if (batched) {
        batch_label = labels.back();
        labels.pop_back();
    }
    if (operand_count + 1 > labels.size()) {
        throw std::invalid_argument(
            "Requested chain needs more distinct einsum labels than the ASCII-letter equation syntax provides.");
    }

    BenchmarkCase benchmark_case;
    benchmark_case.family = std::string(batched ? BATCHED_CHAIN_FAMILY : MATRIX_CHAIN_FAMILY);
    benchmark_case.operand_count = operand_count;
    benchmark_case.input_dimensions.reserve(operand_count);

    for (size_t operand = 0; operand < operand_count; ++operand) {
        if (operand != 0) {
            benchmark_case.equation += ',';
        }
        if (batched) {
            benchmark_case.equation += batch_label;
            benchmark_case.input_dimensions.push_back(
                {4, chainDimension(operand), chainDimension(operand + 1)});
        } else {
            benchmark_case.input_dimensions.push_back(
                {chainDimension(operand), chainDimension(operand + 1)});
        }
        benchmark_case.equation += labels[operand];
        benchmark_case.equation += labels[operand + 1];
    }

    benchmark_case.equation += "->";
    if (batched) {
        benchmark_case.equation += batch_label;
    }
    benchmark_case.equation += labels.front();
    benchmark_case.equation += labels[operand_count];
    return benchmark_case;
}

TimingSummary summarizeTimings(std::vector<double> samples_ms) {
    if (samples_ms.empty()) {
        throw std::logic_error("Planner scalability benchmark recorded no timing samples.");
    }
    std::sort(samples_ms.begin(), samples_ms.end());
    return TimingSummary{
        samples_ms[samples_ms.size() / 2],
        samples_ms.front(),
        samples_ms.back(),
    };
}

BenchmarkResult runCase(const BenchmarkCase& benchmark_case,
                        uint32_t timing_samples,
                        uint32_t beam_width) {
    std::vector<double> samples_ms;
    samples_ms.reserve(timing_samples);

    std::string reference_description;
    EinsumBeamContractionPlan reference_beam;
    for (uint32_t sample = 0; sample < timing_samples; ++sample) {
        const auto start = std::chrono::steady_clock::now();
        const EinsumPlan plan = EinsumPlanner::parseAndPlanWithBeamWidthForDiagnostics(
            benchmark_case.equation, benchmark_case.input_dimensions, beam_width);
        const auto stop = std::chrono::steady_clock::now();

        if (!plan.beam_contraction.has_value()) {
            throw std::logic_error(
                "Planner scalability benchmark expected beam planning but no beam contraction was selected.");
        }
        const std::string description = EinsumPlanner::describeBeamContraction(plan);
        if (sample == 0) {
            reference_description = description;
            reference_beam = *plan.beam_contraction;
        } else if (description != reference_description) {
            throw std::logic_error(
                "Beam planner diagnostics changed across identical benchmark samples; "
                "planner output is not deterministic.");
        }

        samples_ms.push_back(
            std::chrono::duration<double, std::milli>(stop - start).count());
    }

    const uint64_t unique_states = reference_beam.generated_state_count -
                                   reference_beam.deduplicated_state_count;
    if (unique_states != reference_beam.truncated_state_count +
                             reference_beam.retained_state_count) {
        throw std::logic_error(
            "Beam planner state accounting is inconsistent: unique states must equal truncated plus retained states.");
    }
    if (reference_beam.exact_tail_count > reference_beam.beam_width) {
        throw std::logic_error("Beam planner evaluated more exact tails than its retained beam width permits.");
    }

    return BenchmarkResult{
        benchmark_case,
        timing_samples,
        summarizeTimings(std::move(samples_ms)),
        std::move(reference_beam),
        std::move(reference_description),
    };
}

template <typename UnsignedInteger>
std::vector<UnsignedInteger> parseUnsignedCsv(std::string_view text,
                                              std::string_view option_name) {
    std::vector<UnsignedInteger> values;
    size_t begin = 0;
    while (begin < text.size()) {
        const size_t comma = text.find(',', begin);
        const std::string token(text.substr(
            begin, comma == std::string_view::npos ? text.size() - begin : comma - begin));
        if (token.empty()) {
            throw std::invalid_argument(std::string(option_name) + " contains an empty entry.");
        }
        const unsigned long long parsed = std::stoull(token);
        if (parsed > std::numeric_limits<UnsignedInteger>::max()) {
            throw std::invalid_argument(std::string(option_name) + " entry exceeds its integer range.");
        }
        values.push_back(static_cast<UnsignedInteger>(parsed));
        if (comma == std::string_view::npos) {
            break;
        }
        begin = comma + 1;
    }
    if (values.empty()) {
        throw std::invalid_argument(std::string(option_name) + " must contain at least one value.");
    }
    return values;
}

std::vector<size_t> parseOperandCounts(std::string_view text) {
    return parseUnsignedCsv<size_t>(text, "--operands");
}

std::vector<uint32_t> parseBeamWidths(std::string_view text) {
    std::vector<uint32_t> widths = parseUnsignedCsv<uint32_t>(text, "--beam-widths");
    for (uint32_t width : widths) {
        if (width == 0) {
            throw std::invalid_argument("--beam-widths entries must be greater than zero.");
        }
    }
    return widths;
}

void printResult(const BenchmarkResult& result) {
    const EinsumBeamContractionPlan& beam = result.beam;
    const uint64_t unique_states = beam.generated_state_count - beam.deduplicated_state_count;
    const double truncated_percent = unique_states == 0
                                         ? 0.0
                                         : 100.0 * static_cast<double>(beam.truncated_state_count) /
                                               static_cast<double>(unique_states);

    std::cout << result.benchmark_case.family << ','
              << result.benchmark_case.operand_count << ','
              << result.timing_samples << ','
              << beam.beam_width << ','
              << std::fixed << std::setprecision(3)
              << result.timing.median_ms << ','
              << result.timing.best_ms << ','
              << result.timing.worst_ms << ','
              << beam.beam_levels << ','
              << beam.expanded_state_count << ','
              << beam.generated_state_count << ','
              << beam.deduplicated_state_count << ','
              << unique_states << ','
              << beam.truncated_state_count << ','
              << beam.retained_state_count << ','
              << std::setprecision(2) << truncated_percent << ','
              << beam.deferred_disconnected_pair_count << ','
              << beam.exact_tail_count << ','
              << beam.cost.estimated_execution_units << ','
              << beam.cost.peak_intermediate_elements << ','
              << beam.cost.matmul_group_count << ','
              << beam.cost.fused_kernel_count << ','
              << beam.cost.reduction_op_count << ','
              << beam.cost.materialization_op_count << '\n';
}

void printHeader() {
    std::cout
        << "family,operands,samples,beam_width,median_ms,best_ms,worst_ms,beam_levels,expanded_states,"
           "generated_states,deduplicated_states,unique_states,truncated_states,retained_states,"
           "truncated_unique_percent,deferred_disconnected_pairs,exact_tails,estimated_execution_units,"
           "peak_intermediate_elements,matmul_groups,fused_ops,reduction_ops,materialization_ops\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string family = std::string(MATRIX_CHAIN_FAMILY);
        std::vector<size_t> operand_counts{7, 10, 20, 40};
        uint32_t timing_samples = DEFAULT_TIMING_SAMPLES;
        std::vector<uint32_t> beam_widths{EinsumPlanner::DEFAULT_BEAM_WIDTH};
        bool verbose_plan = false;

        for (int argument = 1; argument < argc; ++argument) {
            const std::string_view arg(argv[argument]);
            if (arg == "--help") {
                std::cout
                    << "thor_einsum_planner_scalability_benchmark options:\n"
                    << "  --family=matrix_chain|batched_chain|all\n"
                    << "      Planner topology to measure (default matrix_chain).\n"
                    << "  --operands=7,10,20,40\n"
                    << "      Comma-separated beam-planned operand counts (default 7,10,20,40).\n"
                    << "  --samples=N\n"
                    << "      CPU wall-clock planning samples per case (default 3).\n"
                    << "  --beam-widths=16,32,64,128\n"
                    << "      Comma-separated diagnostic beam widths (default production width 32).\n"
                    << "  --verbose-plan\n"
                    << "      Print the deterministic selected beam tree after each CSV result.\n";
                return EXIT_SUCCESS;
            }
            if (arg.starts_with("--family=")) {
                family = std::string(arg.substr(9));
                if (family != MATRIX_CHAIN_FAMILY && family != BATCHED_CHAIN_FAMILY &&
                    family != "all") {
                    throw std::invalid_argument(
                        "--family must be matrix_chain, batched_chain, or all.");
                }
            } else if (arg.starts_with("--operands=")) {
                operand_counts = parseOperandCounts(arg.substr(11));
            } else if (arg.starts_with("--beam-widths=")) {
                beam_widths = parseBeamWidths(arg.substr(14));
            } else if (arg.starts_with("--samples=")) {
                const unsigned long parsed = std::stoul(std::string(arg.substr(10)));
                if (parsed == 0 || parsed > std::numeric_limits<uint32_t>::max()) {
                    throw std::invalid_argument("--samples must be in [1, UINT32_MAX].");
                }
                timing_samples = static_cast<uint32_t>(parsed);
            } else if (arg == "--verbose-plan") {
                verbose_plan = true;
            } else {
                throw std::invalid_argument("Unknown argument: " + std::string(arg));
            }
        }

        std::cout << "# Thor einsum planner scalability benchmark\n"
                  << "# planner-only CPU wall-clock time; no tensor allocation or GPU execution\n"
                  << "# production_beam_width=" << EinsumPlanner::DEFAULT_BEAM_WIDTH
                  << " exact_tail_active_operands=" << EinsumPlanner::MAX_EXACT_ACTIVE_OPERANDS
                  << '\n';
        printHeader();

        for (size_t operand_count : operand_counts) {
            const auto run_family = [&](bool batched) {
                const BenchmarkCase benchmark_case = makeMatrixChainCase(operand_count, batched);
                for (uint32_t beam_width : beam_widths) {
                    const BenchmarkResult result =
                        runCase(benchmark_case, timing_samples, beam_width);
                    printResult(result);
                    if (verbose_plan) {
                        std::cout << result.plan_description << '\n';
                    }
                }
            };

            if (family == MATRIX_CHAIN_FAMILY || family == "all") {
                run_family(false);
            }
            if (family == BATCHED_CHAIN_FAMILY || family == "all") {
                run_family(true);
            }
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "thor_einsum_planner_scalability_benchmark: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
