// The stamps' benchmark hooks: running a stamp on replacement tensors of the same layout.

#include "Utilities/TensorOperations/Reduction/ArgReducer.h"

#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Common/TensorRequirements.h"
#include "Utilities/TensorOperations/Reduction/FrontEnd.h"
#include "Utilities/TensorOperations/Reduction/Kernels/StridedReduction.h"
#include "Utilities/TensorOperations/Reduction/Planner/DensePhysicalPassExecution.h"
#include "Utilities/TensorOperations/Reduction/Planner/DenseProductionPhysicalSelectionContext.h"
#include "Utilities/TensorOperations/Reduction/Ragged/RaggedStampAutotuner.h"
#include "Utilities/TensorOperations/Reduction/Reducer.h"
#include "Utilities/TensorOperations/Reduction/SegmentedReducer.h"

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

namespace ThorImplementation {

namespace {

using namespace TensorRequirements;
using namespace ReductionInternal;
using namespace ReductionFrontEnd;

template <typename Replacement, typename Bound>
void requireBenchmarkReplacement(const Replacement& replacement, const Bound& bound, const char* what) {
    if (!replacement.isInitialized() || replacement.getDataType() != bound.getDataType() ||
        replacement.getTotalNumElements() != bound.getTotalNumElements() ||
        replacement.getPlacement() != bound.getPlacement() || !replacement.isDenseContiguous()) {
        throw std::invalid_argument(std::string("runOnTensorsForBenchmark: the replacement ") + what +
                                    " must be a dense tensor of the stamp's dtype, element count and device.");
    }
}

// A stamp bound to a non-dense view takes a replacement view of the same dtype, device, dims and strides (its geometry
// addresses the view's strides from the replacement's address); a dense stamp takes a dense replacement.
void requireBenchmarkInputReplacement(const StridedSpan& replacement, const StridedSpan& bound) {
    if (bound.isDenseContiguous()) {
        requireBenchmarkReplacement(replacement, bound, "input");
        return;
    }
    if (!replacement.isInitialized() || replacement.getDataType() != bound.getDataType() ||
        replacement.getPlacement() != bound.getPlacement() || replacement.getDimensions() != bound.getDimensions() ||
        replacement.getStridesElements() != bound.getStridesElements()) {
        throw std::invalid_argument(
            "runOnTensorsForBenchmark: the replacement input of a view stamp must be a view of the stamp's dtype, "
            "device, dimensions and strides.");
    }
}

// The replacement with the bound tensor's dimensions: a copy of the handle (same storage), reshaped. Launches read the
// geometry; the replacement only changes addresses.
StridedSpan reshapedForBenchmark(const StridedSpan& replacement, const std::vector<uint64_t>& dimensions) {
    StridedSpan reshaped = replacement;
    reshaped.reshape(dimensions);
    return reshaped;
}

}  // namespace

void StampedArgReducer::runOnInputForBenchmark(const StridedSpan& replacement_input, Stream& run_stream) const {
    if (composed_stages.empty() || replacement_input.getDataType() != input.getDataType() ||
        replacement_input.getTotalNumElements() != input.getTotalNumElements()) {
        throw std::invalid_argument("runOnInputForBenchmark requires a planned ARG stamp and a matching input.");
    }
    requireCompatibleStream(replacement_input, run_stream);
    ScopedGpu scoped_gpu(run_stream.getGpuNum());
    composed_stages.front()->launchBoundStage(replacement_input, run_stream);
    for (size_t stage = 1; stage < composed_stages.size(); ++stage) {
        composed_stages[stage]->launchBoundStage(composed_stages[stage]->input, run_stream);
    }
}

void StampedSegmentedReducer::runOnTensorsForBenchmark(const StridedSpan& replacement_input,
                                                       const StridedSpan& replacement_segment_offsets,
                                                       const StridedSpan& replacement_output,
                                                       Stream& run_stream) const {
    requireBenchmarkReplacement(replacement_input, input, "input");
    requireBenchmarkReplacement(replacement_segment_offsets, segment_offsets, "segment offsets");
    requireBenchmarkReplacement(replacement_output, output, "output");
    requireCompatibleStream(replacement_input, run_stream);
    ScopedGpu scoped_gpu(run_stream.getGpuNum());
    const StridedSpan bound_input = reshapedForBenchmark(replacement_input, input.getDimensions());
    const StridedSpan bound_offsets = reshapedForBenchmark(replacement_segment_offsets, segment_offsets.getDimensions());
    const StridedSpan bound_output = reshapedForBenchmark(replacement_output, output.getDimensions());
    ReductionInternal::runRaggedRouted(*ragged, bound_input, bound_offsets, bound_output, workspace_span, run_stream);
}

void StampedSegmentedArgReducer::runOnTensorsForBenchmark(const StridedSpan& replacement_input,
                                                          const StridedSpan& replacement_segment_offsets,
                                                          const StridedSpan& replacement_index_output,
                                                          Stream& run_stream) const {
    requireBenchmarkReplacement(replacement_input, input, "input");
    requireBenchmarkReplacement(replacement_segment_offsets, segment_offsets, "segment offsets");
    requireBenchmarkReplacement(replacement_index_output, index_output, "index output");
    requireCompatibleStream(replacement_input, run_stream);
    ScopedGpu scoped_gpu(run_stream.getGpuNum());
    const StridedSpan bound_input = reshapedForBenchmark(replacement_input, input.getDimensions());
    const StridedSpan bound_offsets = reshapedForBenchmark(replacement_segment_offsets, segment_offsets.getDimensions());
    const StridedSpan bound_index_output = reshapedForBenchmark(replacement_index_output, index_output.getDimensions());
    ReductionInternal::runRaggedRouted(*ragged, bound_input, bound_offsets, bound_index_output, workspace_span,
                                       run_stream);
}

void StampedArgReducer::runOnTensorsForBenchmark(const StridedSpan& replacement_input,
                                                 const std::optional<StridedSpan>& replacement_value_output,
                                                 const std::optional<StridedSpan>& replacement_index_output,
                                                 Stream& run_stream) const {
    requireBenchmarkReplacement(replacement_input, input, "input");
    if (replacement_value_output.has_value() != value_output.has_value() ||
        replacement_index_output.has_value() != index_output.has_value()) {
        throw std::invalid_argument("runOnTensorsForBenchmark: replacement outputs must match the stamp's outputs.");
    }
    if (replacement_value_output) requireBenchmarkReplacement(*replacement_value_output, *value_output, "value output");
    if (replacement_index_output) requireBenchmarkReplacement(*replacement_index_output, *index_output, "index output");
    requireCompatibleStream(replacement_input, run_stream);
    ScopedGpu scoped_gpu(run_stream.getGpuNum());

    // Aliases with the bound outputs' dimensions: launches read the geometry, the aliases only change addresses.
    std::optional<StridedSpan> value = replacement_value_output
        ? std::optional<StridedSpan>(reshapedForBenchmark(*replacement_value_output, value_output->getDimensions()))
        : std::nullopt;
    std::optional<StridedSpan> index = replacement_index_output
        ? std::optional<StridedSpan>(reshapedForBenchmark(*replacement_index_output, index_output->getDimensions()))
        : std::nullopt;
    const StridedSpan first_input = reshapedForBenchmark(replacement_input, input.getDimensions());

    if (composed_stages.empty()) {
        launchBoundStage(first_input, value ? &*value : nullptr, index ? &*index : nullptr, run_stream);
        return;
    }
    for (size_t stage = 0; stage < composed_stages.size(); ++stage) {
        const StampedArgReducer& executor = *composed_stages[stage];
        const StridedSpan& stage_input = stage == 0 ? first_input : executor.input;
        if (stage + 1 < composed_stages.size()) {
            executor.launchBoundStage(stage_input, run_stream);
            continue;
        }
        // The last stage writes the public outputs.
        if (executor.value_output.has_value() != value.has_value() ||
            executor.index_output.has_value() != index.has_value()) {
            throw std::logic_error("runOnTensorsForBenchmark: the last ARG stage's outputs differ from the stamp's.");
        }
        executor.launchBoundStage(stage_input, value ? &*value : nullptr, index ? &*index : nullptr, run_stream);
    }
}

void StampedReducer::runOnTensorsForBenchmark(const StridedSpan& replacement_input, const StridedSpan& replacement_output,
                                              Stream& run_stream) const {
    requireBenchmarkInputReplacement(replacement_input, input);  // a strided stamp takes a strided replacement
    requireBenchmarkReplacement(replacement_output, output, "output");
    requireCompatibleStream(replacement_input, run_stream);
    ScopedGpu scoped_gpu(run_stream.getGpuNum());

    if (strided_r_launch.has_value()) {  // the replacement is a view of the stamp's dims and strides
        ReductionInternal::launchStridedR(op, *strided_r_launch, replacement_input,
                                          reshapedForBenchmark(replacement_output, output.getDimensions()),
                                          temp_storage_span, output_scale, run_stream);
        return;
    }
    if (pitched_complete_launch.has_value()) {  // as is the pitched view's
        launchPitchedComplete(*pitched_complete_launch,
                              ReductionInternal::makeValueReductionStageSemantics(
                                  op, physicalStageRole(true, true), semantic_total_reduction_size),
                              pitchedRowsSpan(replacement_input, *geometry.pitched_layout),
                              reshapedForBenchmark(replacement_output, output.getDimensions()),
                              geometry,
                              output_scale,
                              run_stream);
        return;
    }
    if (dense_executable_plan.passes.empty()) {  // every stamp is a dense plan, StridedR or pitched
        throw std::logic_error("A value reduction stamp has no executor.");
    }
    const size_t last = dense_executable_plan.passes.size() - 1;
    for (size_t index = 0; index <= last; ++index) {
        const ReductionInternal::DenseExecutablePass& pass = dense_executable_plan.passes[index];
        ReductionInternal::ReductionStageSemantics stage_semantics =
            ReductionInternal::makeValueReductionStageSemantics(op, pass.role, semantic_total_reduction_size);
        if (!apply_input_transform) {
            stage_semantics.input_transform = ReductionInternal::ReductionStageInputTransform::Identity;
        }
        if (!apply_output_finalize) {
            stage_semantics.finalize = ReductionInternal::ReductionStageFinalize::Identity;
            stage_semantics.finalize_divisor = 1;
        }
        const float stage_output_scale = pass.role == ReductionInternal::DensePassRole::Complete ||
                                                 pass.role == ReductionInternal::DensePassRole::Final
                                             ? output_scale
                                             : 1.0f;
        // The first pass reads the replacement input, the last writes the replacement output (aliased to the pass's own
        // dimensions); every other tensor is the stamp's. A view stamp's first pass reads the replacement view as
        // dense, as the stamp read its own view.
        const StridedSpan pass_input =
            index != 0                  ? pass.input
            : input.isDenseContiguous() ? reshapedForBenchmark(replacement_input, pass.input.getDimensions())
                                        : replacement_input.asDense(pass.input.getDimensions());
        const StridedSpan pass_output = index == last
                                            ? reshapedForBenchmark(replacement_output, pass.output.getDimensions())
                                            : pass.output_span;
        ReductionInternal::launchDensePhysicalPass(*pass.physical_plan, stage_semantics, pass.temp_storage_span,
                                                   pass.temp_storage_bytes, pass_input, pass_output, pass.geometry,
                                                   stage_output_scale, run_stream);
    }
}

}  // namespace ThorImplementation
