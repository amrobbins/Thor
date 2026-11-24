// ARG reduction: ArgReducer and StampedArgReducer (validation, the planned dense ARG passes and their run).

#include "Utilities/TensorOperations/Reduction/ArgReducer.h"

#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Common/TensorRequirements.h"
#include "Utilities/TensorOperations/Reduction/Arg/Fp32ArgPhysicalPlan.h"
#include "Utilities/TensorOperations/Reduction/Arg/Fp8ArgPhysicalPlan.h"
#include "Utilities/TensorOperations/Reduction/FrontEnd.h"
#include "Utilities/TensorOperations/Reduction/Kernels/KernelLaunchers.h"
#include "Utilities/TensorOperations/Reduction/Planner/DensePhysicalPassExecution.h"
#include "Utilities/TensorOperations/Reduction/Planner/DenseProductionPhysicalSelectionContext.h"
#include "Utilities/TensorOperations/Reduction/Planner/DenseReductionFamilies.h"
#include "Utilities/TensorOperations/Reduction/Planner/DenseReductionPlanner.h"

#include <cuda_runtime_api.h>

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace ThorImplementation {

namespace {

using namespace TensorRequirements;
using namespace ReductionInternal;
using namespace ReductionFrontEnd;

void requireExpectedArgOutput(const StridedSpan& input,
                              const Tensor& output,
                              DataType output_dtype,
                              const ReductionGeometry& geometry,
                              const char* role) {
    requireDenseContiguousReductionOutput(output, role);
    if (input.getPlacement() != output.getPlacement()) {
        throw std::invalid_argument(std::string("input and ") + role + " must live on the same GPU placement.");
    }
    if (output.getDataType() != output_dtype) {
        throw std::invalid_argument(std::string("Arg reduction preallocated ") + role
                                    + " dtype does not match the configured dtype.");
    }
    if (stripSingletonDimensions(output.getDimensions())
        != stripSingletonDimensions(geometry.squeezed_output_dimensions)) {
        throw std::invalid_argument(std::string("Arg reduction preallocated ") + role
                                    + " dimensions must preserve the reduction output's non-singleton layout.");
    }
    if (output.getTotalNumElements() != geometry.output_elements) {
        throw std::invalid_argument(std::string("Arg reduction preallocated ") + role
                                    + " element count does not match the reduction geometry.");
    }
    if (tensorStorageOverlaps(input, output)) {
        throw std::invalid_argument(std::string("Arg reduction input and ") + role + " storage must not overlap.");
    }
}

void requireArgOutputsDoNotOverlap(const std::optional<Tensor>& value_output,
                                   const std::optional<Tensor>& index_output) {
    if (value_output.has_value() && index_output.has_value()
        && tensorStorageOverlaps(value_output.value(), index_output.value())) {
        throw std::invalid_argument("Arg reduction value and index output storage must not overlap.");
    }
}

// Dense ARG is planned by planDenseReduction(), like VALUE. Every supported input dtype (FP8, FP16, BF16, FP32) runs
// the planned pass list; there is no other dense ARG executor.
[[nodiscard]] bool isFp8ArgInput(DataType input_dtype) {
    return input_dtype == DataType::FP8_E4M3 || input_dtype == DataType::FP8_E5M2;
}

// FP8, FP32 (its FP32 ARG passes) and FP16 / BF16 input (its 2-byte ARG passes, then the FP32 carried ARG passes).
[[nodiscard]] bool usesPlannedDenseArg(DataType input_dtype) {
    return isFp8ArgInput(input_dtype) || input_dtype == DataType::FP32 || input_dtype == DataType::FP16 ||
           input_dtype == DataType::BF16;
}

[[nodiscard]] DataType denseArgCarriedIndexDType(uint64_t original_reduction_size) {
    return original_reduction_size <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) ? DataType::UINT32
                                                                                                 : DataType::UINT64;
}

// One planned ARG pass. geometry is the pass's own direct geometry over the current (normalized) run dimensions;
// domain_stride is the nominated run's original ARG-domain stride (0 for a reducer-created intermediate run, whose
// elements already carry their original indices).
struct PlannedDenseArgPass {
    ReductionGeometry geometry;
    uint64_t domain_stride = 0;
    std::vector<uint64_t> intermediate_dimensions;
    bool final_pass = false;
    // An FP8 ARG physical pass (no scratch), or null.
    std::shared_ptr<const Fp8ArgPhysicalPlan> fp8_arg;
    // An FP32 ARG pass, original or carried; its scratch is plan.scratch_bytes.
    std::shared_ptr<const Fp32ArgPhysicalPlan> fp32_arg;
    [[nodiscard]] uint64_t retainedScratchBytes() const {
        return fp8_arg ? retainedFp8ArgScratchBytes(*fp8_arg) : fp32_arg ? fp32_arg->scratch_bytes : 0;
    }
};

struct PlannedDenseArg {
    std::vector<PlannedDenseArgPass> passes;
    DataType carried_index_dtype = DataType::UINT32;
};

[[nodiscard]] PlannedDenseArg planProductionDenseArg(ArgReductionOp op,
                                                     const std::vector<uint64_t>& input_dimensions,
                                                     const std::vector<uint32_t>& axes,
                                                     DataType input_dtype,
                                                     std::optional<DataType> value_output_dtype,
                                                     std::optional<DataType> index_output_dtype,
                                                     uint64_t original_reduction_size,
                                                     ReductionTuningDepth tuning_depth,
                                                     const Stream& stream,
                                                     uint64_t input_alignment_bytes = 16) {
    DenseReductionOutputSpec output_spec;
    output_spec.produce_value = value_output_dtype.has_value();
    output_spec.value_dtype = value_output_dtype.value_or(DataType::FP32);
    output_spec.produce_index = index_output_dtype.has_value();
    output_spec.index_dtype = index_output_dtype.value_or(DataType::UINT32);

    PlannedDenseArg planned;
    planned.carried_index_dtype = denseArgCarriedIndexDType(original_reduction_size);
    const DenseReductionProblem problem = makeInitialDenseReductionProblem(
        input_dimensions, axes, DenseReductionAggregateKind::Arg, output_spec, planned.carried_index_dtype);
    // The context names the VALUE operation whose FP8 selection ARG mirrors (MIN for argmin, MAX for argmax) and the
    // FP8 VALUE workspace budget used by that selection.
    constexpr uint64_t kRetainedFp8WorkspaceBudget = uint64_t{256} * 1024 * 1024;
    const DenseProductionPhysicalSelectionContext production_context(
        input_dtype,
        op == ArgReductionOp::ArgMin ? DenseRKValueOperation::Minimum : DenseRKValueOperation::Maximum,
        stream,
        original_reduction_size,
        1.0f,
        op == ArgReductionOp::ArgMin ? ReductionOp::Min : ReductionOp::Max,
        kRetainedFp8WorkspaceBudget,
        tuning_depth,
        input_alignment_bytes);
    const DenseReductionPlan plan = planDenseReduction(problem, production_context.context());

    for (const DenseReductionPlanStep& step : plan.steps) {
        const DenseReductionCandidate& candidate = step.selected_candidate;
        const auto fp8_arg =
            std::dynamic_pointer_cast<const Fp8ArgPhysicalPlan>(candidate.physical_plan);
        const auto fp32_arg =
            std::dynamic_pointer_cast<const Fp32ArgPhysicalPlan>(candidate.physical_plan);
        if (!fp8_arg && !fp32_arg) {
            throw std::logic_error("Dense ARG plan selected a physical pass that has no ARG executor.");
        }
        const DensePhysicalPassLayout layout = describeDensePhysicalPass(*candidate.physical_plan);
        PlannedDenseArgPass pass;
        pass.fp8_arg = fp8_arg;
        pass.fp32_arg = fp32_arg;
        pass.geometry = layout.geometry;
        if (!isDirectDenseReductionPath(pass.geometry.path)) {
            throw std::logic_error("Dense ARG plan pass did not resolve to a direct reducer family.");
        }
        const DenseReductionRun& site_run = step.input_problem.runs.at(candidate.site.run_index);
        pass.domain_stride = denseArgOriginalIndexStride(site_run).value_or(0);
        pass.final_pass = !hasDenseReductionSites(candidate.next_problem);
        pass.intermediate_dimensions = layout.intermediate_dimensions;
        if (!pass.final_pass) {
            const uint64_t elements = TensorDescriptor(DataType::FP32, pass.intermediate_dimensions).getTotalNumElements();
            const uint64_t bytes = TensorDescriptor::getArraySizeInBytes(elements, DataType::FP32) +
                                   TensorDescriptor::getArraySizeInBytes(elements, planned.carried_index_dtype);
            if (bytes != candidate.next_bytes) {
                throw std::logic_error("Dense ARG pass intermediate differs from the selected next problem.");
            }
        }
        planned.passes.push_back(std::move(pass));
    }
    if (planned.passes.empty() || !planned.passes.back().final_pass) {
        throw std::logic_error("Dense ARG planner produced no final pass.");
    }
    return planned;
}

}  // namespace

uint64_t ArgReducer::composeOriginalArgIndex(uint64_t previous_index,
                                             uint64_t local_run_index,
                                             uint64_t domain_stride) {
    if (domain_stride != 0 && local_run_index > std::numeric_limits<uint64_t>::max() / domain_stride) {
        throw std::overflow_error("Arg reduction original-index contribution overflows uint64_t.");
    }
    const uint64_t contribution = local_run_index * domain_stride;
    if (previous_index > std::numeric_limits<uint64_t>::max() - contribution) {
        throw std::overflow_error("Arg reduction composed original index overflows uint64_t.");
    }
    return previous_index + contribution;
}

ArgReducer::ArgReducer(ArgReductionOp op,
                       uint32_t axis,
                       ArgReductionOutputOptions outputs,
                       std::optional<ReductionTuningDepth> tuning_depth)
    : ArgReducer(op, std::vector<uint32_t>{axis}, std::move(outputs), tuning_depth) {}

ArgReducer::ArgReducer(ArgReductionOp op,
                       std::vector<uint32_t> axes,
                       ArgReductionOutputOptions outputs,
                       std::optional<ReductionTuningDepth> tuning_depth)
    : op(op), axes(std::move(axes)), outputs(std::move(outputs)), tuning_depth(resolveReductionTuningDepth(tuning_depth)) {
    requireSupportedArgOperation(op);
    if (this->axes.empty()) {
        throw std::invalid_argument("Arg reduction requires at least one reduction axis.");
    }
    for (size_t i = 1; i < this->axes.size(); ++i) {
        if (this->axes[i] <= this->axes[i - 1]) {
            throw std::invalid_argument("Arg reduction axes must be unique and strictly increasing.");
        }
    }
    if (!this->outputs.produce_value && !this->outputs.produce_index) {
        throw std::invalid_argument("Arg reduction must produce a value, an index, or both.");
    }
    if (!this->outputs.produce_value && this->outputs.value_output_dtype.has_value()) {
        throw std::invalid_argument("Arg reduction cannot configure a value dtype when value output is disabled.");
    }
    if (this->outputs.value_output_dtype.has_value()) {
        requireSupportedFloatingStorageDType(this->outputs.value_output_dtype.value(), "value output");
    }
    if (this->outputs.produce_index) {
        requireSupportedArgIndexDType(this->outputs.index_output_dtype);
    }
}

DataType ArgReducer::resolveValueOutputDataType(DataType input_dtype) const {
    requireSupportedFloatingStorageDType(input_dtype, "input");
    const DataType resolved = outputs.value_output_dtype.value_or(input_dtype);
    requireSupportedFloatingStorageDType(resolved, "value output");
    return resolved;
}

float ArgReducer::getFp32EmptyReductionValue(ArgReductionOp op) {
    requireSupportedArgOperation(op);
    return op == ArgReductionOp::ArgMin ? std::numeric_limits<float>::infinity()
                                           : -std::numeric_limits<float>::infinity();
}

size_t ArgReducer::queryWorkspaceSizeInBytes(const TensorDescriptor& input_descriptor,
                                                          const Stream& stream) const {
    requireSupportedFloatingStorageDType(input_descriptor.getDataType(), "input");
    ReductionGeometry geometry = analyzeDenseGeometry(input_descriptor.getDimensions(), axes);
    if (outputs.produce_index && outputs.index_output_dtype == DataType::UINT32
        && geometry.reduction_size > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::invalid_argument("Arg reduction domain does not fit in a UINT32 local index.");
    }

    const std::optional<DataType> value_output_dtype =
        outputs.produce_value ? std::optional<DataType>(resolveValueOutputDataType(input_descriptor.getDataType()))
                              : std::nullopt;
    const std::optional<DataType> index_output_dtype =
        outputs.produce_index ? std::optional<DataType>(outputs.index_output_dtype) : std::nullopt;
    ScopedGpu scoped_gpu(stream.getGpuNum());

    if (!usesPlannedDenseArg(input_descriptor.getDataType())) {
        throw std::logic_error("Dense ARG has no planned executor for this input dtype.");
    }
    // Mirrors stampPlannedDenseValidated(): the same deterministic plan, intermediates and per-pass scratch.
    const PlannedDenseArg planned = planProductionDenseArg(op,
                                                           input_descriptor.getDimensions(),
                                                           axes,
                                                           input_descriptor.getDataType(),
                                                           value_output_dtype,
                                                           index_output_dtype,
                                                           geometry.reduction_size,
                                                           tuning_depth,
                                                           stream);
    size_t workspace_size_bytes = 0;
    auto add_bytes = [&](uint64_t bytes) {
        if (bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max() - workspace_size_bytes)) {
            throw std::overflow_error("Arg reduction workspace size overflows size_t.");
        }
        workspace_size_bytes += static_cast<size_t>(bytes);
    };
    for (const PlannedDenseArgPass& pass : planned.passes) {
        add_bytes(pass.retainedScratchBytes());
        if (!pass.final_pass) {
            const uint64_t elements =
                TensorDescriptor(DataType::FP32, pass.intermediate_dimensions).getTotalNumElements();
            add_bytes(TensorDescriptor::getArraySizeInBytes(elements, DataType::FP32));
            add_bytes(TensorDescriptor::getArraySizeInBytes(elements, planned.carried_index_dtype));
        }
    }
    return workspace_size_bytes;
}

std::shared_ptr<StampedArgReducer> ArgReducer::stamp(const StridedSpan& input,
                                                     const Stream& stream) const {
    return stamp(input, std::nullopt, std::nullopt, stream);
}

std::shared_ptr<StampedArgReducer> ArgReducer::stamp(
    const StridedSpan& input,
    const std::optional<Tensor>& preallocated_value_output,
    const std::optional<Tensor>& preallocated_index_output,
    const Stream& stream) const {
    // ARG reads dense input (it may start partway into its storage, a dense span).
    requireGpuTensorView(input, "input");
    if (!input.isDenseContiguous()) {
        throw std::invalid_argument("input must be dense contiguous.");
    }
    requireCompatibleStream(input, stream);
    requireSupportedFloatingStorageDType(input.getDataType(), "input");

    ReductionGeometry geometry = analyzeDenseGeometry(input.getDimensions(), axes);
    if (outputs.produce_index && outputs.index_output_dtype == DataType::UINT32
        && geometry.reduction_size > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::invalid_argument("Arg reduction domain does not fit in a UINT32 local index.");
    }

    std::optional<Tensor> value_output = std::nullopt;
    if (outputs.produce_value) {
        const DataType value_dtype = resolveValueOutputDataType(input.getDataType());
        if (preallocated_value_output.has_value()) {
            requireExpectedArgOutput(input, preallocated_value_output.value(), value_dtype, geometry, "value output");
            value_output = preallocated_value_output;
        } else {
            value_output = Tensor(input.getPlacement(), TensorDescriptor(value_dtype, geometry.output_dimensions));
        }
    } else if (preallocated_value_output.has_value()) {
        throw std::invalid_argument("Arg reduction received a value output while value production is disabled.");
    }

    std::optional<Tensor> index_output = std::nullopt;
    if (outputs.produce_index) {
        if (preallocated_index_output.has_value()) {
            requireExpectedArgOutput(
                input, preallocated_index_output.value(), outputs.index_output_dtype, geometry, "index output");
            index_output = preallocated_index_output;
        } else {
            index_output = Tensor(
                input.getPlacement(), TensorDescriptor(outputs.index_output_dtype, geometry.output_dimensions));
        }
    } else if (preallocated_index_output.has_value()) {
        throw std::invalid_argument("Arg reduction received an index output while index production is disabled.");
    }

    requireArgOutputsDoNotOverlap(value_output, index_output);
    if (!usesPlannedDenseArg(input.getDataType())) {
        throw std::logic_error("Dense ARG has no planned executor for this input dtype.");
    }
    // Every dense ARG executes the planned pass list.
    return stampPlannedDenseValidated(input, std::move(value_output), std::move(index_output), geometry, stream);
}

std::shared_ptr<StampedArgReducer> ArgReducer::stampPlannedDenseValidated(
    const StridedSpan& input,
    std::optional<Tensor> value_output,
    std::optional<Tensor> index_output,
    const ReductionGeometry& geometry,
    const Stream& stream) const {
    ScopedGpu scoped_gpu(stream.getGpuNum());
    const PlannedDenseArg planned = planProductionDenseArg(
        op,
        input.getDimensions(),
        axes,
        input.getDataType(),
        value_output.has_value() ? std::optional<DataType>(value_output->getDataType()) : std::nullopt,
        index_output.has_value() ? std::optional<DataType>(index_output->getDataType()) : std::nullopt,
        geometry.reduction_size,
        tuning_depth,
        stream,
        input.alignmentBytes(16));

    std::vector<std::shared_ptr<StampedArgReducer>> stages;
    stages.reserve(planned.passes.size());
    StridedSpan current_value_input = input;
    std::optional<Tensor> current_index_input = std::nullopt;
    size_t workspace_size_bytes = 0;
    auto add_bytes = [&](uint64_t bytes) {
        if (bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max() - workspace_size_bytes)) {
            throw std::overflow_error("Arg reduction workspace size overflows size_t.");
        }
        workspace_size_bytes += static_cast<size_t>(bytes);
    };

    for (const PlannedDenseArgPass& pass : planned.passes) {
        // Tensors are dense; a pass only needs the element count it reads. Its own run dimensions live in geometry.
        if (current_value_input.getTotalNumElements() != pass.geometry.input_elements) {
            throw std::logic_error("Planned dense ARG pass input does not match the previous pass output.");
        }
        std::optional<Tensor> stage_value_output;
        std::optional<Tensor> stage_index_output;
        if (pass.final_pass) {
            stage_value_output = value_output;
            stage_index_output = index_output;
        } else {
            stage_value_output =
                Tensor(input.getPlacement(), TensorDescriptor(DataType::FP32, pass.intermediate_dimensions));
            stage_index_output = Tensor(
                input.getPlacement(), TensorDescriptor(planned.carried_index_dtype, pass.intermediate_dimensions));
        }
        const size_t temp_storage_bytes = static_cast<size_t>(pass.retainedScratchBytes());
        Tensor temp_storage;
        if (temp_storage_bytes != 0) {
            temp_storage = Tensor(
                input.getPlacement(), TensorDescriptor(DataType::UINT8, {static_cast<uint64_t>(temp_storage_bytes)}));
        }
        stages.push_back(std::shared_ptr<StampedArgReducer>(new StampedArgReducer(op,
                                                                                  pass.geometry,
                                                                                  current_value_input,
                                                                                  current_index_input,
                                                                                  stage_value_output,
                                                                                  stage_index_output,
                                                                                  temp_storage_bytes,
                                                                                  temp_storage,
                                                                                  pass.domain_stride,
                                                                                  planned.carried_index_dtype,
                                                                                  stream)));
        stages.back()->fp8_arg_plan = pass.fp8_arg;
        stages.back()->fp32_arg_plan = pass.fp32_arg;
        add_bytes(temp_storage_bytes);
        if (!pass.final_pass) {
            add_bytes(stage_value_output->getArraySizeInBytes());
            add_bytes(stage_index_output->getArraySizeInBytes());
            current_value_input = stage_value_output.value();
            current_index_input = stage_index_output.value();
        }
    }

    // The public geometry (and so getPath()) stays the analyzed dense geometry; execution is the planned pass list.
    return std::shared_ptr<StampedArgReducer>(new StampedArgReducer(op,
                                                                             geometry,
                                                                             input,
                                                                             std::move(value_output),
                                                                             std::move(index_output),
                                                                             workspace_size_bytes,
                                                                             std::move(stages),
                                                                             stream));
}

StampedArgReducer::StampedArgReducer(ArgReductionOp op,
                                     ReductionGeometry geometry,
                                     const StridedSpan& value_input,
                                     std::optional<Tensor> carried_index_input,
                                     std::optional<Tensor> value_output,
                                     std::optional<Tensor> index_output,
                                     size_t temp_storage_bytes,
                                     const Tensor& temp_storage,
                                     uint64_t domain_stride,
                                     DataType carried_index_dtype,
                                     const Stream& stream)
    : op(op),
      geometry(std::move(geometry)),
      input(value_input),
      value_output(std::move(value_output)),
      index_output(std::move(index_output)),
      temp_storage_bytes(temp_storage_bytes),
      temp_storage(temp_storage),
      carried_index_input(std::move(carried_index_input)),
      composed_carried_index_dtype(carried_index_dtype),
      composed_domain_stride(domain_stride),
      stream(stream) {
    if (temp_storage_bytes != 0) {
        requireTempStorage(this->temp_storage, value_input.getPlacement(), temp_storage_bytes);
    } else if (this->temp_storage.isInitialized()) {
        throw std::invalid_argument("Zero-scratch ARG stage must not own a temporary-storage Tensor.");
    }
    THOR_THROW_IF_FALSE(isDirectDenseReductionPath(this->geometry.path));
    requireSupportedArgIndexDType(carried_index_dtype);
    if (this->carried_index_input.has_value()) {
        THOR_THROW_IF_FALSE(this->input.getDataType() == DataType::FP32);
        THOR_THROW_IF_FALSE(this->carried_index_input->getDataType() == carried_index_dtype);
        THOR_THROW_IF_FALSE(this->carried_index_input->getDimensions() == this->input.getDimensions());
    }
    if (!this->value_output.has_value() && !this->index_output.has_value()) {
        throw std::invalid_argument("Stamped composed arg stage requires at least one output.");
    }
    bindLaunchSpans();
}

void StampedArgReducer::bindLaunchSpans() {
    if (value_output.has_value()) value_output_span = StridedSpan(*value_output);
    if (index_output.has_value()) index_output_span = StridedSpan(*index_output);
    if (carried_index_input.has_value()) carried_index_span = StridedSpan(*carried_index_input);
}

StampedArgReducer::StampedArgReducer(
    ArgReductionOp op,
    ReductionGeometry geometry,
    const StridedSpan& input,
    std::optional<Tensor> value_output,
    std::optional<Tensor> index_output,
    size_t workspace_size_bytes,
    std::vector<std::shared_ptr<StampedArgReducer>> composed_stages,
    const Stream& stream)
    : op(op),
      geometry(std::move(geometry)),
      input(input),
      value_output(std::move(value_output)),
      index_output(std::move(index_output)),
      temp_storage_bytes(workspace_size_bytes),
      composed_stages(std::move(composed_stages)),
      stream(stream) {
    // Planned dense ARG: the public geometry may be any dense path, DisjointRuns included.
    THOR_THROW_IF_FALSE(this->geometry.path == ReductionLayout::DisjointRuns ||
                        isDirectDenseReductionPath(this->geometry.path));
    THOR_THROW_IF_FALSE(!this->composed_stages.empty());
    for (const auto& stage : this->composed_stages) {
        THOR_THROW_IF_FALSE(stage != nullptr);
        THOR_THROW_IF_FALSE(stage->geometry.path != ReductionLayout::DisjointRuns);
        THOR_THROW_IF_FALSE(stage->composed_carried_index_dtype.has_value());
    }
    if (!this->value_output.has_value() && !this->index_output.has_value()) {
        throw std::invalid_argument("Stamped composed arg reduction requires at least one public output.");
    }
    bindLaunchSpans();
}

std::vector<std::vector<uint32_t>> StampedArgReducer::getComposedStageAxes() const {
    if (geometry.path != ReductionLayout::DisjointRuns) {
        return {};
    }
    std::vector<std::vector<uint32_t>> stage_axes;
    stage_axes.reserve(composed_stages.size());
    for (const auto& stage : composed_stages) {
        THOR_THROW_IF_FALSE(stage != nullptr);
        stage_axes.push_back(stage->geometry.axes);
    }
    return stage_axes;
}

void StampedArgReducer::run() { runOn(stream); }

void StampedArgReducer::runOn(Stream& run_stream) const {
    requireCompatibleStream(input, run_stream);
    ScopedGpu scoped_gpu(run_stream.getGpuNum());

    if (!composed_stages.empty()) {
        for (const auto& stage : composed_stages) {
            THOR_THROW_IF_FALSE(stage != nullptr);
            stage->runOn(run_stream);
        }
        return;
    }

    launchBoundStage(input, run_stream);
}

void StampedArgReducer::launchBoundStage(const StridedSpan& stage_input, Stream& run_stream) const {
    launchBoundStage(stage_input, value_output_span.has_value() ? &value_output_span.value() : nullptr,
                     index_output_span.has_value() ? &index_output_span.value() : nullptr, run_stream);
}

void StampedArgReducer::launchBoundStage(const StridedSpan& stage_input, const StridedSpan* value_output_ptr,
                                         const StridedSpan* index_output_ptr, Stream& run_stream) const {
    if (fp32_arg_plan) {
        // An FP32 ARG pass; a carried pass reads this stage's carried index, and staged / Split partials live in this
        // stage's scratch.
        THOR_THROW_IF_FALSE(carried_index_input.has_value() == fp32_arg_plan->carried);
        THOR_THROW_IF_FALSE(temp_storage_bytes >= fp32_arg_plan->scratch_bytes);
        void* scratch = fp32_arg_plan->scratch_bytes == 0
                            ? nullptr
                            : const_cast<void*>(static_cast<const void*>(temp_storage.getMemPtr<void>()));
        ReductionInternal::launchRetainedFp32ArgPlan(*fp32_arg_plan, stage_input,
                                                     carried_index_span.has_value() ? &*carried_index_span : nullptr,
                                                     value_output_ptr, index_output_ptr, geometry, scratch,
                                                     op == ArgReductionOp::ArgMax, composed_domain_stride,
                                                     run_stream);
        return;
    }
    if (fp8_arg_plan) {
        // An FP8 ARG pass (packed staged twins carry their partials in this stage's scratch).
        THOR_THROW_IF_FALSE(!carried_index_input.has_value());
        uint32_t* packed = nullptr;
        if (fp8_arg_plan->packed_partials) {
            THOR_THROW_IF_FALSE(temp_storage_bytes >= ReductionInternal::retainedFp8ArgScratchBytes(*fp8_arg_plan));
            packed = static_cast<uint32_t*>(const_cast<void*>(static_cast<const void*>(temp_storage.getMemPtr<void>())));
        }
        ReductionInternal::launchRetainedFp8ArgPlan(*fp8_arg_plan, stage_input, value_output_ptr, index_output_ptr,
                                                    geometry, packed, op == ArgReductionOp::ArgMax,
                                                    composed_domain_stride, run_stream);
        return;
    }
    // Every stage is an FP8 or FP32 ARG pass.
    throw std::logic_error("Stamped dense ARG stage has no retained ARG plan.");
}

std::vector<std::string> StampedArgReducer::getPlannedPassNames() const {
    std::vector<std::string> names;
    if (geometry.path != ReductionLayout::DisjointRuns && composed_stages.empty()) return names;
    for (const auto& stage : composed_stages) {
        if (stage->fp32_arg_plan) {
            names.push_back(ReductionInternal::retainedFp32ArgPassName(*stage->fp32_arg_plan));
            continue;
        }
        if (stage->fp8_arg_plan && stage->fp8_arg_plan->kr.group_lanes != 0) {
            names.push_back("fp8_arg_kr:" + ReductionInternal::KR::krShapeName(stage->fp8_arg_plan->kr));
            continue;
        }
        if (stage->fp8_arg_plan && stage->fp8_arg_plan->family == ReductionInternal::RK::PhysicalFamily::RuntimeK &&
            stage->fp8_arg_plan->tile_components != 0) {
            names.push_back("fp8_arg_runtime_k:packet_complete_w" + std::to_string(stage->fp8_arg_plan->warps_per_output) +
                            "_tile" + std::to_string(stage->fp8_arg_plan->tile_components));
            if (stage->fp8_arg_plan->packed_partials) names.emplace_back("fp8_arg_packed_continuation");
            continue;
        }
        if (stage->fp8_arg_plan && stage->fp8_arg_plan->family == ReductionInternal::RK::PhysicalFamily::RuntimeK) {
            names.push_back("fp8_arg_runtime_k:p" + std::to_string(stage->fp8_arg_plan->packet_bytes) +
                            (stage->fp8_arg_plan->shards == 1 ? std::string("_complete")
                                                              : "_rows" + std::to_string(stage->fp8_arg_plan->rows_per_shard)));
            if (stage->fp8_arg_plan->packed_partials) names.emplace_back("fp8_arg_packed_continuation");
            continue;
        }
        if (stage->fp8_arg_plan && stage->fp8_arg_plan->family == ReductionInternal::RK::PhysicalFamily::AlignedK) {
            names.push_back("fp8_arg_aligned_k:p" + std::to_string(stage->fp8_arg_plan->packet_bytes) + "_cta" +
                            std::to_string(stage->fp8_arg_plan->cta_threads) + "_shards" +
                            std::to_string(stage->fp8_arg_plan->shards));
            if (stage->fp8_arg_plan->packed_partials) names.emplace_back("fp8_arg_packed_continuation");
            continue;
        }
        if (stage->fp8_arg_plan && stage->fp8_arg_plan->family == ReductionInternal::RK::PhysicalFamily::DeepRSmallK) {
            names.push_back("fp8_arg_deep_r_small_k:p16_rows" + std::to_string(stage->fp8_arg_plan->rows_per_shard));
            if (stage->fp8_arg_plan->packed_partials) names.emplace_back("fp8_arg_packed_continuation");
            continue;
        }
        if (stage->fp8_arg_plan) {
            const ReductionInternal::OneWindowUnit& unit = stage->fp8_arg_plan->unit;
            names.push_back("fp8_arg_one_window:o" + std::to_string(unit.outputs_per_warp) + "_w" +
                            std::to_string(stage->fp8_arg_plan->warps_per_output) + "_p" +
                            std::to_string(unit.packet_bytes) + "_c" + std::to_string(unit.max_components_per_lane) +
                            (unit.load_ahead ? "_la" : ""));
            if (stage->fp8_arg_plan->packed_partials) names.emplace_back("fp8_arg_packed_continuation");
            continue;
        }
        throw std::logic_error("Planned dense ARG stage has no retained ARG plan.");
    }
    return names;
}

}  // namespace ThorImplementation
