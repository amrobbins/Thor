#include "Utilities/TensorOperations/Cub/DenseReductionPlanning.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ThorImplementation::CubReductionInternal {
namespace {

[[nodiscard]] uint64_t checkedMultiply(uint64_t lhs, uint64_t rhs, const char* message) {
    if (rhs != 0 && lhs > std::numeric_limits<uint64_t>::max() / rhs) {
        throw std::invalid_argument(message);
    }
    return lhs * rhs;
}

[[nodiscard]] uint64_t dataTypeBytes(DataType dtype) {
    switch (dtype) {
        case DataType::BOOLEAN:
        case DataType::INT8:
        case DataType::UINT8:
        case DataType::FP8_E4M3:
        case DataType::FP8_E5M2:
            return 1;
        case DataType::FP16:
        case DataType::BF16:
        case DataType::INT16:
        case DataType::UINT16:
            return 2;
        case DataType::FP32:
        case DataType::INT32:
        case DataType::UINT32:
            return 4;
        case DataType::FP64:
        case DataType::INT64:
        case DataType::UINT64:
            return 8;
        case DataType::TF32:
            throw std::invalid_argument("TF32 is not a dense reduction storage dtype.");
    }
    throw std::invalid_argument("Unsupported dense reduction storage dtype.");
}

[[nodiscard]] bool isSupportedCarriedArgIndexType(DataType dtype) {
    return dtype == DataType::UINT32 || dtype == DataType::UINT64;
}

void validateOutputSpec(DenseReductionAggregateKind aggregate_kind,
                        const DenseReductionOutputSpec& final_output,
                        DataType carried_arg_index_dtype) {
    if (aggregate_kind == DenseReductionAggregateKind::Value) {
        if (!final_output.produce_value || final_output.produce_index) {
            throw std::invalid_argument("VALUE dense reduction must request exactly one value output.");
        }
        static_cast<void>(dataTypeBytes(final_output.value_dtype));
        return;
    }

    if (!final_output.produce_value && !final_output.produce_index) {
        throw std::invalid_argument("ARG dense reduction must request a value output, an index output, or both.");
    }
    if (!isSupportedCarriedArgIndexType(carried_arg_index_dtype)) {
        throw std::invalid_argument("ARG dense reduction carried index dtype must be UINT32 or UINT64.");
    }
    if (final_output.produce_value) {
        static_cast<void>(dataTypeBytes(final_output.value_dtype));
    }
    if (final_output.produce_index) {
        if (!isSupportedCarriedArgIndexType(final_output.index_dtype)) {
            throw std::invalid_argument("ARG dense reduction final index dtype must be UINT32 or UINT64.");
        }
        static_cast<void>(dataTypeBytes(final_output.index_dtype));
    }
}

void appendRetainedRun(std::vector<DenseReductionRun>& runs, uint64_t extent) {
    if (extent == 1) {
        return;
    }
    if (!runs.empty() && runs.back().kind == DenseReductionRunKind::Retained) {
        runs.back().extent = checkedMultiply(
            runs.back().extent, extent, "Dense reduction retained-run extent overflows uint64_t.");
        return;
    }
    runs.push_back(DenseReductionRun{.kind = DenseReductionRunKind::Retained,
                                     .extent = extent,
                                     .original_arg_domain_stride = 0});
}

void appendOriginalReductionRun(std::vector<DenseReductionRun>& runs, uint64_t extent) {
    if (!runs.empty() && runs.back().kind == DenseReductionRunKind::OriginalReduction) {
        runs.back().extent = checkedMultiply(
            runs.back().extent, extent, "Dense reduction original-run extent overflows uint64_t.");
        return;
    }
    runs.push_back(DenseReductionRun{.kind = DenseReductionRunKind::OriginalReduction,
                                     .extent = extent,
                                     .original_arg_domain_stride = 0});
}

void assignOriginalArgDomainStrides(std::vector<DenseReductionRun>& runs) {
    uint64_t stride = 1;
    for (size_t run_index = runs.size(); run_index-- > 0;) {
        DenseReductionRun& run = runs[run_index];
        if (run.kind != DenseReductionRunKind::OriginalReduction) {
            run.original_arg_domain_stride = 0;
            continue;
        }
        run.original_arg_domain_stride = stride;
        stride = checkedMultiply(stride, run.extent, "Dense reduction original ARG domain overflows uint64_t.");
    }
}

void normalizeAfterPass(std::vector<DenseReductionRun>& runs) {
    std::vector<DenseReductionRun> normalized;
    normalized.reserve(runs.size());
    for (const DenseReductionRun& run : runs) {
        if (run.extent == 1 && isDenseReductionRunReduced(run.kind)) {
            continue;
        }
        if (run.extent == 0) {
            throw std::invalid_argument("Dense reduction run extent must be non-zero.");
        }
        if (run.kind == DenseReductionRunKind::Retained && !normalized.empty()
            && normalized.back().kind == DenseReductionRunKind::Retained) {
            normalized.back().extent = checkedMultiply(
                normalized.back().extent, run.extent, "Dense reduction normalized retained-run extent overflows uint64_t.");
            continue;
        }
        normalized.push_back(run);
    }
    runs = std::move(normalized);
}

[[nodiscard]] const DenseReductionRun& requireReducedSite(const DenseReductionProblem& problem,
                                                          DenseReductionSite site) {
    if (site.run_index >= problem.runs.size()) {
        throw std::invalid_argument("Dense reduction site run index is out of range.");
    }
    const DenseReductionRun& run = problem.runs[site.run_index];
    if (!isDenseReductionRunReduced(run.kind)) {
        throw std::invalid_argument("Dense reduction site must nominate a reduced run.");
    }
    if (run.extent == 0) {
        throw std::invalid_argument("Dense reduction site extent must be non-zero.");
    }
    return run;
}

[[nodiscard]] uint64_t currentElementCount(const DenseReductionProblem& problem) {
    uint64_t elements = 1;
    for (const DenseReductionRun& run : problem.runs) {
        if (run.extent == 0) {
            throw std::invalid_argument("Dense reduction run extent must be non-zero.");
        }
        elements = checkedMultiply(elements, run.extent, "Dense reduction problem element count overflows uint64_t.");
    }
    return elements;
}

}  // namespace

bool isDenseReductionRunReduced(DenseReductionRunKind kind) {
    return kind == DenseReductionRunKind::OriginalReduction || kind == DenseReductionRunKind::IntermediateReduction;
}

bool hasDenseReductionSites(const DenseReductionProblem& problem) {
    return std::any_of(problem.runs.begin(), problem.runs.end(), [](const DenseReductionRun& run) {
        return isDenseReductionRunReduced(run.kind);
    });
}

DenseReductionProblem makeInitialDenseReductionProblem(const std::vector<uint64_t>& input_dimensions,
                                                        const std::vector<uint32_t>& reduction_axes,
                                                        DenseReductionAggregateKind aggregate_kind,
                                                        const DenseReductionOutputSpec& final_output,
                                                        DataType carried_arg_index_dtype) {
    if (input_dimensions.empty()) {
        throw std::invalid_argument("Dense reduction input rank must be non-zero.");
    }
    if (input_dimensions.size() > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("Dense reduction input rank exceeds uint32_t axis representation.");
    }
    if (reduction_axes.empty()) {
        throw std::invalid_argument("Dense reduction axes must be non-empty.");
    }
    validateOutputSpec(aggregate_kind, final_output, carried_arg_index_dtype);

    for (uint64_t extent : input_dimensions) {
        if (extent == 0) {
            throw std::invalid_argument("Dense reduction dimensions must be non-zero.");
        }
    }
    uint32_t previous_axis = 0;
    bool first_axis = true;
    for (uint32_t axis : reduction_axes) {
        if (axis >= input_dimensions.size()) {
            throw std::invalid_argument("Dense reduction axis is out of range.");
        }
        if (!first_axis && axis <= previous_axis) {
            throw std::invalid_argument("Dense reduction axes must be unique and strictly increasing.");
        }
        previous_axis = axis;
        first_axis = false;
    }

    std::vector<bool> reduced(input_dimensions.size(), false);
    bool has_non_singleton_reduction = false;
    for (uint32_t axis : reduction_axes) {
        reduced[axis] = true;
        has_non_singleton_reduction = has_non_singleton_reduction || input_dimensions[axis] > 1;
    }

    DenseReductionProblem problem;
    problem.aggregate_kind = aggregate_kind;
    problem.input_state = DenseReductionInputState::OriginalInput;
    problem.carried_arg_index_dtype = carried_arg_index_dtype;
    problem.final_output = final_output;
    problem.finalizer_pending = true;
    problem.runs.reserve(input_dimensions.size());

    if (has_non_singleton_reduction) {
        for (uint32_t axis = 0; axis < input_dimensions.size(); ++axis) {
            const uint64_t extent = input_dimensions[axis];
            if (extent == 1) {
                continue;
            }
            if (reduced[axis]) {
                appendOriginalReductionRun(problem.runs, extent);
            } else {
                appendRetainedRun(problem.runs, extent);
            }
        }
    } else {
        // Reduction by singleton axes still requires one pass for the input transform/finalizer. Place one synthetic
        // R=1 at the first reduced logical axis so its R/KR/RK topology relative to real retained runs remains correct.
        const uint32_t synthetic_axis = reduction_axes.front();
        for (uint32_t axis = 0; axis < input_dimensions.size(); ++axis) {
            if (axis == synthetic_axis) {
                appendOriginalReductionRun(problem.runs, 1);
            }
            if (!reduced[axis]) {
                appendRetainedRun(problem.runs, input_dimensions[axis]);
            }
        }
    }

    assignOriginalArgDomainStrides(problem.runs);
    return problem;
}

std::vector<DenseReductionSite> enumerateDenseReductionSites(const DenseReductionProblem& problem) {
    std::vector<DenseReductionSite> sites;
    for (size_t run_index = 0; run_index < problem.runs.size(); ++run_index) {
        if (!isDenseReductionRunReduced(problem.runs[run_index].kind)) {
            continue;
        }
        if (run_index > std::numeric_limits<uint32_t>::max()) {
            throw std::invalid_argument("Dense reduction run count exceeds uint32_t site representation.");
        }
        sites.push_back(DenseReductionSite{.run_index = static_cast<uint32_t>(run_index)});
    }
    return sites;
}

DenseReducerFamily classifyDenseReducerFamily(const DenseReductionProblem& problem, DenseReductionSite site) {
    static_cast<void>(requireReducedSite(problem, site));

    if (problem.runs.size() == 1) {
        return DenseReducerFamily::R;
    }
    if (site.run_index + 1U == problem.runs.size()) {
        return DenseReducerFamily::KR;
    }
    return DenseReducerFamily::RK;
}

DenseReductionProblem advanceDenseReductionProblem(const DenseReductionProblem& problem,
                                                    DenseReductionSite site,
                                                    uint64_t next_reduction_extent) {
    const DenseReductionRun& selected = requireReducedSite(problem, site);
    if (next_reduction_extent == 0 || next_reduction_extent > selected.extent) {
        throw std::invalid_argument("Dense reduction next extent must be in [1,current_extent].");
    }

    const bool semantic_singleton_pass = selected.extent == 1 && next_reduction_extent == 1
                                         && problem.input_state == DenseReductionInputState::OriginalInput;
    if (next_reduction_extent == selected.extent && !semantic_singleton_pass) {
        throw std::invalid_argument("Dense reduction pass must strictly reduce its nominated extent.");
    }

    DenseReductionProblem next = problem;
    if (next_reduction_extent == 1) {
        next.runs.erase(next.runs.begin() + site.run_index);
    } else {
        DenseReductionRun& next_run = next.runs[site.run_index];
        next_run.extent = next_reduction_extent;
        next_run.kind = DenseReductionRunKind::IntermediateReduction;
        next_run.original_arg_domain_stride = 0;
    }
    normalizeAfterPass(next.runs);

    next.input_state = DenseReductionInputState::IntermediateAggregate;
    next.finalizer_pending = hasDenseReductionSites(next);
    return next;
}

DensePassRole deriveDensePassRole(const DenseReductionProblem& current_problem,
                                  const DenseReductionProblem& next_problem) {
    const bool consumes_original_input = current_problem.input_state == DenseReductionInputState::OriginalInput;
    const bool finishes_reduction = !hasDenseReductionSites(next_problem);

    if (consumes_original_input && finishes_reduction) {
        return DensePassRole::Complete;
    }
    if (consumes_original_input) {
        return DensePassRole::First;
    }
    if (finishes_reduction) {
        return DensePassRole::Final;
    }
    return DensePassRole::Intermediate;
}

std::optional<uint64_t> denseArgOriginalIndexStride(const DenseReductionRun& run) {
    if (run.kind != DenseReductionRunKind::OriginalReduction) {
        return std::nullopt;
    }
    if (run.original_arg_domain_stride == 0) {
        throw std::invalid_argument("Original dense ARG reduction run must have a non-zero original domain stride.");
    }
    return run.original_arg_domain_stride;
}

uint64_t denseReductionProblemBytes(const DenseReductionProblem& problem) {
    const uint64_t elements = currentElementCount(problem);
    if (hasDenseReductionSites(problem)) {
        if (problem.input_state != DenseReductionInputState::IntermediateAggregate) {
            throw std::invalid_argument("next_bytes is defined for reducer-produced aggregate problems, not raw input.");
        }
        uint64_t bytes_per_element = dataTypeBytes(DataType::FP32);
        if (problem.aggregate_kind == DenseReductionAggregateKind::Arg) {
            if (!isSupportedCarriedArgIndexType(problem.carried_arg_index_dtype)) {
                throw std::invalid_argument("ARG dense reduction carried index dtype must be UINT32 or UINT64.");
            }
            bytes_per_element += dataTypeBytes(problem.carried_arg_index_dtype);
        }
        return checkedMultiply(elements, bytes_per_element, "Dense reduction intermediate byte count overflows uint64_t.");
    }

    validateOutputSpec(problem.aggregate_kind, problem.final_output, problem.carried_arg_index_dtype);
    uint64_t bytes_per_output = 0;
    if (problem.final_output.produce_value) {
        bytes_per_output += dataTypeBytes(problem.final_output.value_dtype);
    }
    if (problem.final_output.produce_index) {
        bytes_per_output += dataTypeBytes(problem.final_output.index_dtype);
    }
    return checkedMultiply(elements, bytes_per_output, "Dense reduction final-output byte count overflows uint64_t.");
}

}  // namespace ThorImplementation::CubReductionInternal
