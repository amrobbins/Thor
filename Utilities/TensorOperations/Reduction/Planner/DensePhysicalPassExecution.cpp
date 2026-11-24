// Dense planner passes: describe, size, launch and materialize the physical passes of a dense VALUE plan.

#include "Utilities/TensorOperations/Reduction/Planner/DensePhysicalPassExecution.h"

#include "Utilities/Common/TensorRequirements.h"
#include "Utilities/TensorOperations/Reduction/Arg/Fp32ArgPhysicalPlan.h"
#include "Utilities/TensorOperations/Reduction/Arg/Fp8ArgPhysicalPlan.h"
#include "Utilities/TensorOperations/Reduction/FrontEnd.h"
#include "Utilities/TensorOperations/Reduction/KR/KRPhysicalPlan.h"
#include "Utilities/TensorOperations/Reduction/Planner/DenseProductionPhysicalSelectionContext.h"
#include "Utilities/TensorOperations/Reduction/Planner/DenseReductionPlanner.h"
#include "Utilities/TensorOperations/Reduction/RK/RKPhysicalKernels.h"
#include "Utilities/TensorOperations/Reduction/RK/RKPhysicalPlan.h"
#include "Utilities/TensorOperations/Reduction/Reducer.h"

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <utility>

namespace ThorImplementation {

namespace {

using namespace TensorRequirements;
using namespace ReductionInternal;
using namespace ReductionFrontEnd;

[[nodiscard]] DenseExecutablePlan materializeDenseValueExecutablePlanImpl(ReductionOp op,
                                                                       const StridedSpan& input,
                                                                       const Tensor& output,
                                                                       const DenseReductionPlan& plan,
                                                                       uint64_t semantic_total_reduction_size,
                                                                       float output_scale,
                                                                       const Stream& stream) {
    DenseExecutablePlan executable;
    executable.passes.reserve(plan.steps.size());
    // The first pass reads the input in place (a dense span, read under the pass's dimensions); later passes read the
    // previous pass's output.
    StridedSpan current_input = input;

    for (const DenseReductionPlanStep& step : plan.steps) {
        const DenseReductionCandidate& candidate = step.selected_candidate;
        if (!candidate.physical_plan) throw std::logic_error("Dense VALUE plan step is missing a physical plan.");
        const DensePhysicalPassLayout layout = describeDensePhysicalPass(*candidate.physical_plan);
        current_input.reshape(layout.input_dimensions);
        ReductionGeometry stage_geometry = layout.geometry;
        const DensePassRole role = deriveDensePassRole(step.input_problem, candidate.next_problem);
        const bool final_pass = !hasDenseReductionSites(candidate.next_problem);

        Tensor stage_output =
            final_pass
                ? output
                : Tensor(input.getPlacement(),
                         TensorDescriptor(DataType::FP32,
                                          layout.intermediate_dimensions));
        if (final_pass) {
            // The caller's logical output may be squeezed. Only reshape the
            // stamp-local Tensor handle to the selected physical pass's rank.
            stage_output.reshape(stage_geometry.output_dimensions);
        }
        if (stage_output.getArraySizeInBytes() != candidate.next_bytes)
            throw std::logic_error("Dense VALUE physical output differs from the selected next problem");
        const size_t intermediate_bytes = final_pass ? 0 : stage_output.getArraySizeInBytes();

        const ReductionStageSemantics stage_semantics =
            makeValueReductionStageSemantics(op, role, semantic_total_reduction_size);
        const size_t temp_storage_bytes = queryDensePhysicalPassScratchBytes(*candidate.physical_plan,
                                                                              stage_semantics,
                                                                              current_input.getDataType(),
                                                                              current_input.getMemPtr<void>(),
                                                                              current_input.getTotalNumElements(),
                                                                              stage_output.getDataType(),
                                                                              stage_output.getMemPtr<void>(),
                                                                              stage_geometry,
                                                                              final_pass ? output_scale : 1.0f,
                                                                              stream);
        Tensor temp_storage;
        if (temp_storage_bytes != 0) {
            // Allocate only the reported per-pass scratch. Direct CUB queries
            // retain their existing minimum-one-byte contract for now.
            temp_storage = Tensor(input.getPlacement(),
                                  TensorDescriptor(DataType::UINT8, {temp_storage_bytes}));
        }

        DenseExecutablePass pass;
        pass.physical_plan = candidate.physical_plan;
        pass.role = role;
        pass.input = current_input;
        pass.output = stage_output;
        pass.geometry = std::move(stage_geometry);
        pass.intermediate_bytes = intermediate_bytes;
        pass.temp_storage_bytes = temp_storage_bytes;
        pass.temp_storage = std::move(temp_storage);
        pass.output_span = pass.output;
        pass.temp_storage_span = pass.temp_storage;
        executable.workspace_size_bytes = checkedWorkspaceAdd(executable.workspace_size_bytes,
                                                               pass.workspaceBytes(),
                                                               "Dense executable VALUE workspace size overflows size_t.");
        executable.passes.push_back(std::move(pass));
        current_input = stage_output;
    }

    if (executable.passes.empty()) throw std::logic_error("Dense VALUE planner produced no executable passes.");
    return executable;
}

}  // namespace

namespace ReductionInternal {
namespace {

// Capability gate shared by stamp-time scratch query and a direct physical
// launch. It does not inspect selectors, allocate resources or query CUDA.
void requireSupportedRetainedFp8PhysicalPass(
    const RKPhysicalPlan& fp8,
    const ReductionStageSemantics& semantics,
    DataType input_dtype, uint64_t input_elements, DataType output_dtype,
    const ReductionGeometry& geometry) {
    // All public VALUE combines, first-stage transforms and finalizers are implemented by the typed kernels. Complete
    // kernels convert to the requested floating storage dtype; staged kernels emit FP32 only.
    // Keep the physical boundary strict (no fallback).
    if (!isSupportedFloatingStorageDType(output_dtype))
        throw std::logic_error("Retained FP8 RK physical pass requires supported floating output storage");
    if (fp8.kernel.staged() &&
        (output_dtype != DataType::FP32 ||
         semantics.finalize != ReductionStageFinalize::Identity))
        throw std::logic_error("Retained FP8 RK staged pass must emit unfinalized FP32 partials");
    if (input_dtype != fp8.input_dtype || output_dtype != fp8.output_dtype ||
        input_elements != geometry.input_elements ||
        geometry.outer_size != fp8.outer_size ||
        geometry.reduction_size != fp8.reduction_size ||
        geometry.inner_size != fp8.inner_size ||
        fp8.reduction_axis != 1 ||
        fp8.output_dimensions.size() != 3 ||
        fp8.output_dimensions[0] != fp8.outer_size ||
        fp8.output_dimensions[2] != fp8.inner_size)
        throw std::invalid_argument(
            "Retained FP8 RK physical query input/output geometry or dtype mismatch");
}

}  // namespace

DensePhysicalPassLayout describeDensePhysicalPass(const DenseReductionPhysicalPlan& physical_plan) {
    DensePhysicalPassLayout layout;
    if (const auto* kr = dynamic_cast<const KRPhysicalPlan*>(&physical_plan)) {
        // One Complete pass over the site's current runs; staged chunk partials are the pass's scratch.
        layout.input_dimensions = kr->input_dimensions;
        layout.geometry = Reducer::analyzeGeometry(layout.input_dimensions,
                                                   std::vector<uint32_t>{kr->reduction_axis});
    } else if (const auto* arg = dynamic_cast<const Fp8ArgPhysicalPlan*>(&physical_plan)) {
        // OneWindow ARG twins are Complete ([O,K]); DeepRSmallK ARG twins and multi-shard RuntimeK/AlignedK twins write
        // [O,shards,K] partials. An R / KR pass reduces its last current run.
        layout.input_dimensions = arg->input_dimensions;
        const uint32_t axis = arg->kr.group_lanes != 0 ? static_cast<uint32_t>(arg->input_dimensions.size() - 1) : 1U;
        layout.geometry = Reducer::analyzeGeometry(layout.input_dimensions, std::vector<uint32_t>{axis});
        if (arg->shards > 1 && !arg->packed_partials)
            layout.intermediate_dimensions = {arg->outer_size, arg->shards, arg->inner_size};
    } else if (const auto* fp32_arg = dynamic_cast<const Fp32ArgPhysicalPlan*>(&physical_plan)) {
        // A Complete FP32 ARG pass (RK over {outer, r, inner} at axis 1; R / KR over the current runs, the nominated
        // run last).
        layout.input_dimensions = fp32_arg->input_dimensions;
        layout.geometry = Reducer::analyzeGeometry(layout.input_dimensions,
                                                   std::vector<uint32_t>{fp32_arg->reduction_axis});
    } else if (const auto* fp8 = dynamic_cast<const RKPhysicalPlan*>(&physical_plan)) {
        // The selected first kernel has its own real [O,N,K] output. A staged
        // output is an ordinary FP32 intermediate, never hidden scratch.
        layout.input_dimensions = fp8->input_dimensions;
        layout.geometry = Reducer::analyzeGeometry(
            layout.input_dimensions, std::vector<uint32_t>{fp8->reduction_axis});
        layout.intermediate_dimensions = fp8->output_dimensions;
    } else {
        throw std::logic_error("Dense VALUE physical plan has no geometry adapter.");
    }
    if (layout.intermediate_dimensions.empty()) layout.intermediate_dimensions = layout.geometry.output_dimensions;
    return layout;
}

size_t queryDensePhysicalPassScratchBytes(const DenseReductionPhysicalPlan& physical_plan,
                                         const ReductionStageSemantics& semantics,
                                         DataType input_dtype,
                                         const void* input,
                                         uint64_t input_elements,
                                         DataType output_dtype,
                                         void* output,
                                         const ReductionGeometry& geometry,
                                         float output_scale,
                                         const Stream& stream) {
    if (const auto* kr = dynamic_cast<const KRPhysicalPlan*>(&physical_plan)) {
        // An FP32 pass's shape is in bytes, 4 per element. 2 per FP16 / BF16 element.
        const uint64_t element_bytes = kr->input_dtype == DataType::FP32 ? 4
                                       : kr->input_dtype == DataType::FP16 || kr->input_dtype == DataType::BF16 ? 2
                                                                                                                : 1;
        if (input_dtype != kr->input_dtype || input_elements * element_bytes != kr->shape.rows * kr->shape.row_bytes ||
            geometry.output_elements != kr->shape.rows || !isSupportedFloatingStorageDType(output_dtype))
            throw std::invalid_argument("Retained FP8 KR physical query input/output geometry or dtype mismatch");
        static_cast<void>(input);
        static_cast<void>(output);
        static_cast<void>(output_scale);
        static_cast<void>(stream);
        return static_cast<size_t>(KR::krScratchBytes(kr->shape, false));
    }
    if (const auto* fp8 = dynamic_cast<const RKPhysicalPlan*>(&physical_plan)) {
        requireSupportedRetainedFp8PhysicalPass(*fp8, semantics, input_dtype,
                                               input_elements, output_dtype,
                                               geometry);
        // first_stage_scratch_bytes is the staged output Tensor, already
        // accounted for by the ordinary materializer. No additional scratch.
        return 0;
    }
    static_cast<void>(input);
    static_cast<void>(output);
    static_cast<void>(output_scale);
    static_cast<void>(stream);
    throw std::logic_error("Dense VALUE physical pass has no registered physical scratch query.");
}

void launchDensePhysicalPass(const DenseReductionPhysicalPlan& physical_plan,
                             const ReductionStageSemantics& semantics,
                             const StridedSpan& temp_storage,
                             size_t temp_storage_bytes,
                             const StridedSpan& input,
                             const StridedSpan& output,
                             const ReductionGeometry& geometry,
                             float output_scale,
                             Stream& stream) {
    if (const auto* kr = dynamic_cast<const KRPhysicalPlan*>(&physical_plan)) {
        // One R / KR kernel, plus the finishing kernel of a staged shape (partials in this scratch).
        if (input.getDataType() != kr->input_dtype || geometry.output_elements != kr->shape.rows)
            throw std::invalid_argument("Retained FP8 KR physical launch binding mismatch");
        void* scratch = temp_storage_bytes == 0
                            ? nullptr
                            : const_cast<void*>(static_cast<const void*>(temp_storage.getMemPtr<void>()));
        if (kr->input_dtype == DataType::FP32) {  // the FP32 R / KR pass
            launchFp32KRValueSegments(semantics, input, output, kr->shape, scratch, temp_storage_bytes, output_scale,
                                      stream);
        } else if (kr->input_dtype == DataType::FP16 || kr->input_dtype == DataType::BF16) {  // the 2-byte pass
            launchF16KRValueSegments(semantics, input, output, kr->shape, scratch, temp_storage_bytes, output_scale,
                                     stream);
        } else {
            launchFp8KRValueSegments(semantics, input, output, kr->shape, scratch, temp_storage_bytes, output_scale,
                                     stream);
        }
        return;
    }
    if (const auto* fp8 = dynamic_cast<const RKPhysicalPlan*>(&physical_plan)) {
        // This dispatch launches precisely ONE physical CUDA kernel.
        // The caller's flat pass list owns any following FP32 reductions.
        requireSupportedRetainedFp8PhysicalPass(*fp8, semantics,
            input.getDataType(), input.getTotalNumElements(), output.getDataType(),
            geometry);
        // Avoid getDimensions() here: Tensor returns a vector by value and
        // runOn must never allocate to revalidate a stamped shape.
        if (temp_storage_bytes != 0 ||
            output.getTotalNumElements() !=
                geometry.output_elements * fp8->output_dimensions[1])
            throw std::invalid_argument("Retained FP8 RK physical launch binding mismatch");
        RK::launchPhysicalFirstStage(fp8->kernel, semantics, input, output,
                                     geometry, output_scale, stream);
        return;
    }
    throw std::logic_error("Dense executable VALUE pass has no registered physical executor.");
}

DenseExecutablePlan materializeDenseValueExecutablePlan(
    ReductionOp op, const StridedSpan& input, const Tensor& output,
    const DenseReductionPlan& plan, uint64_t semantic_total_reduction_size,
    float output_scale, const Stream& stream) {
    return materializeDenseValueExecutablePlanImpl(op, input, output, plan,
                                                   semantic_total_reduction_size,
                                                   output_scale, stream);
}

}  // namespace ReductionInternal

}  // namespace ThorImplementation
