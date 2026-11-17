// Value reduction: Reducer and StampedReducer (stamps, workspace, the dense VALUE plan and its run).

#include "Utilities/TensorOperations/Reduction/Reducer.h"

#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Exceptions.h"
#include "Utilities/TensorOperations/Cub/CubDevicePrimitiveSupport.h"
#include "Utilities/TensorOperations/Reduction/FrontEnd.h"
#include "Utilities/TensorOperations/Reduction/KR/KRPhysicalPlan.h"
#include "Utilities/TensorOperations/Reduction/Kernels/StridedReduction.h"
#include "Utilities/TensorOperations/Reduction/Planner/DensePhysicalPassExecution.h"
#include "Utilities/TensorOperations/Reduction/Planner/DenseProductionPhysicalSelectionContext.h"
#include "Utilities/TensorOperations/Reduction/Planner/DenseReductionPlanner.h"
#include "Utilities/TensorOperations/Reduction/RK/RKPhysicalPlan.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace ThorImplementation {

namespace {

using namespace CubDevicePrimitiveSupport;
using namespace ReductionInternal;
using namespace ReductionFrontEnd;

// The diagnostic family prefix of an R / KR VALUE pass by its input dtype.
[[nodiscard]] std::string retainedKRFamilyPrefix(DataType input_dtype) {
    if (input_dtype == DataType::FP32) return "retained_fp32_kr:";
    if (input_dtype == DataType::FP16 || input_dtype == DataType::BF16) return "retained_f16_kr:";
    return "retained_fp8_kr:";
}

[[nodiscard]] DenseReductionPlan planProductionDenseValue(ReductionOp op,
                                                           const std::vector<uint64_t>& input_dimensions,
                                                           const std::vector<uint32_t>& axes,
                                                           DataType original_input_dtype,
                                                           DataType output_dtype,
                                                           uint64_t original_reduction_size,
                                                           float output_scale,
                                                           ReductionTuningDepth tuning_depth,
                                                           const Stream& stream,
                                                           std::string* strategy_chain = nullptr,
                                                           uint64_t input_alignment_bytes = 16) {
    DenseReductionOutputSpec output_spec;
    output_spec.produce_value = true;
    output_spec.value_dtype = output_dtype;
    output_spec.produce_index = false;
    const DenseReductionProblem problem = makeInitialDenseReductionProblem(
        input_dimensions, axes, DenseReductionAggregateKind::Value, output_spec);

    // The context survives every physical selection made by this plan (the selectors read its stream).
    constexpr uint64_t kRetainedFp8RKWorkspaceBudget = uint64_t{256} * 1024 * 1024;
    const bool is_fp8 = original_input_dtype == DataType::FP8_E4M3 ||
                        original_input_dtype == DataType::FP8_E5M2;
    const DenseProductionPhysicalSelectionContext production_context(
        original_input_dtype, denseRKValueOperation(op), stream,
        original_reduction_size, output_scale, op,
        is_fp8 ? kRetainedFp8RKWorkspaceBudget : std::numeric_limits<uint64_t>::max(), tuning_depth,
        input_alignment_bytes);

    DenseReductionPlan plan = planDenseReduction(problem, production_context.context());
    if (strategy_chain != nullptr) {
        strategy_chain->clear();
        for (const DenseReductionPlanStep& step : plan.steps) {
            if (!strategy_chain->empty()) strategy_chain->append(">");
            if (const auto fp8 = std::dynamic_pointer_cast<const RKPhysicalPlan>(
                    step.selected_candidate.physical_plan); fp8) {
                strategy_chain->append(fp8->input_dtype == DataType::FP32   ? "retained_fp32_rk:"
                                       : fp8->kernel.element_bytes == 2 ? "retained_f16_rk:"
                                                                        : "retained_fp8_rk:");
                switch (fp8->kernel.family) {
                    case RK::PhysicalFamily::OneWindow: strategy_chain->append("one_window"); break;
                    case RK::PhysicalFamily::DeepRSmallK: strategy_chain->append("deep_r_small_k"); break;
                    case RK::PhysicalFamily::AlignedK: strategy_chain->append("aligned_k"); break;
                    case RK::PhysicalFamily::RuntimeK: strategy_chain->append("runtime_k"); break;
                }
            } else if (const auto kr = std::dynamic_pointer_cast<const KRPhysicalPlan>(
                    step.selected_candidate.physical_plan); kr) {
                strategy_chain->append(retainedKRFamilyPrefix(kr->input_dtype) + KR::krShapeName(kr->shape));
            } else {
                throw std::logic_error("Dense VALUE plan selected a physical pass without a retained executor");
            }
        }
    }
    return plan;
}

[[nodiscard]] bool isOrdinaryDenseValueGeometry(const ReductionGeometry& geometry) {
    const bool ordinary_path = geometry.path == ReductionLayout::SingleOutput
                               || geometry.path == ReductionLayout::TrailingBlock
                               || geometry.path == ReductionLayout::MiddleBlock
                               || geometry.path == ReductionLayout::DisjointRuns;
    return ordinary_path
           && !geometry.permuted_layout.has_value()
           && !geometry.pitched_layout.has_value()
           && !geometry.payload_transpose_layout.has_value()
           && !geometry.tiled_output_permuted
           && !geometry.tiled_output_shared_transpose;
}

void requireExpectedOutput(const StridedSpan& input,
                           const Tensor& output,
                           DataType output_dtype,
                           const ReductionGeometry& geometry) {
    requireDenseContiguousReductionOutput(output, "output");
    if (input.getPlacement() != output.getPlacement()) {
        throw std::invalid_argument("input and output must live on the same GPU placement.");
    }
    if (output.getDataType() != output_dtype) {
        throw std::invalid_argument("Tensor reduction preallocated output dtype does not match the configured output dtype.");
    }
    const std::vector<uint64_t>& output_dimensions = output.getDimensions();
    if (stripSingletonDimensions(output_dimensions) != stripSingletonDimensions(geometry.squeezed_output_dimensions)) {
        throw std::invalid_argument(
            "Tensor reduction preallocated output dimensions must preserve the reduction output's non-singleton layout.");
    }
    if (output.getTotalNumElements() != geometry.output_elements) {
        throw std::invalid_argument("Tensor reduction preallocated output element count does not match the reduction geometry.");
    }

    if (tensorStorageOverlaps(input, output)) {
        throw std::invalid_argument("Tensor reduction input and output storage must not overlap.");
    }
}

[[nodiscard]] size_t queryDenseValueExecutableWorkspace(ReductionOp op,
                                                         const TensorDescriptor& input_descriptor,
                                                         DataType output_dtype,
                                                         const DenseReductionPlan& plan,
                                                         uint64_t semantic_total_reduction_size,
                                                         float output_scale,
                                                         const Stream& stream) {
    TensorDescriptor current_descriptor = input_descriptor;
    size_t workspace_bytes = 0;

    for (const DenseReductionPlanStep& step : plan.steps) {
        const DenseReductionCandidate& candidate = step.selected_candidate;
        if (!candidate.physical_plan) throw std::logic_error("Dense VALUE plan step is missing a physical plan.");
        const DensePhysicalPassLayout layout = describeDensePhysicalPass(*candidate.physical_plan);
        current_descriptor = TensorDescriptor(current_descriptor.getDataType(), layout.input_dimensions);
        const ReductionGeometry& stage_geometry = layout.geometry;
        const bool final_pass = !hasDenseReductionSites(candidate.next_problem);
        const DataType stage_output_dtype = final_pass ? output_dtype : DataType::FP32;
        const std::vector<uint64_t> stage_output_dimensions =
            final_pass ? stage_geometry.output_dimensions
                       : layout.intermediate_dimensions;
        const TensorDescriptor stage_output_descriptor(stage_output_dtype, stage_output_dimensions);

        if (!final_pass) {
            workspace_bytes = checkedWorkspaceAdd(workspace_bytes,
                                                   stage_output_descriptor.getArraySizeInBytes(),
                                                   "Dense VALUE intermediate workspace size overflows size_t.");
        }

        const ReductionStageSemantics stage_semantics =
            makeValueReductionStageSemantics(op,
                                             deriveDensePassRole(step.input_problem, candidate.next_problem),
                                             semantic_total_reduction_size);
        const size_t scratch_bytes = queryDensePhysicalPassScratchBytes(*candidate.physical_plan,
                                                                         stage_semantics,
                                                                         current_descriptor.getDataType(),
                                                                         nullptr,
                                                                         current_descriptor.getTotalNumElements(),
                                                                         stage_output_dtype,
                                                                         nullptr,
                                                                         stage_geometry,
                                                                         final_pass ? output_scale : 1.0f,
                                                                         stream);
        workspace_bytes = checkedWorkspaceAdd(workspace_bytes,
                                               scratch_bytes,
                                               "Dense VALUE physical scratch workspace size overflows size_t.");
        current_descriptor = stage_output_descriptor;
    }
    return workspace_bytes;
}

}  // namespace

ReductionInternal::ReductionStageSemantics ReductionInternal::makeValueReductionStageSemantics(
    ReductionOp op, ReductionStageRole role, uint64_t total_reduction_size) {
    ReductionStageSemantics semantics;

    switch (op) {
        case ReductionOp::Sum:
        case ReductionOp::Mean:
        case ReductionOp::L1Norm:
        case ReductionOp::L2Norm:
        case ReductionOp::SumSquares:
            semantics.combine = ReductionStageCombine::Sum;
            break;
        case ReductionOp::Product:
            semantics.combine = ReductionStageCombine::Product;
            break;
        case ReductionOp::Min:
            semantics.combine = ReductionStageCombine::Minimum;
            break;
        case ReductionOp::Max:
            semantics.combine = ReductionStageCombine::Maximum;
            break;
        default:
            throw std::invalid_argument("Unsupported tensor reduction operation.");
    }

    const bool applies_input_transform =
        role == ReductionStageRole::Complete || role == ReductionStageRole::First;
    if (applies_input_transform) {
        if (op == ReductionOp::L1Norm) {
            semantics.input_transform = ReductionStageInputTransform::AbsoluteValue;
        } else if (op == ReductionOp::L2Norm || op == ReductionOp::SumSquares) {
            semantics.input_transform = ReductionStageInputTransform::Square;
        }
    }

    const bool applies_finalizer =
        role == ReductionStageRole::Complete || role == ReductionStageRole::Final;
    if (applies_finalizer) {
        if (op == ReductionOp::Mean) {
            semantics.finalize = ReductionStageFinalize::Divide;
            semantics.finalize_divisor = total_reduction_size;
        } else if (op == ReductionOp::L2Norm) {
            semantics.finalize = ReductionStageFinalize::SquareRoot;
        }
    }

    return semantics;
}

Reducer::Reducer(ReductionOp op,
                 uint32_t axis,
                 std::optional<DataType> output_dtype,
                 float output_scale,
                 std::optional<ReductionTuningDepth> tuning_depth)
    : Reducer(op, std::vector<uint32_t>{axis}, output_dtype, output_scale, tuning_depth) {}

Reducer::Reducer(ReductionOp op,
                 std::vector<uint32_t> axes,
                 std::optional<DataType> output_dtype,
                 float output_scale,
                 std::optional<ReductionTuningDepth> tuning_depth)
    : op(op), axes(std::move(axes)), output_dtype(output_dtype), output_scale(output_scale),
      tuning_depth(resolveReductionTuningDepth(tuning_depth)) {
    requireSupportedOperation(op);
    if (this->axes.empty()) {
        throw std::invalid_argument("Tensor reduction requires at least one reduction axis.");
    }
    for (size_t i = 1; i < this->axes.size(); ++i) {
        if (this->axes[i] <= this->axes[i - 1]) {
            throw std::invalid_argument("Tensor reduction axes must be unique and strictly increasing.");
        }
    }
    if (output_dtype.has_value()) {
        requireSupportedFloatingStorageDType(output_dtype.value(), "output");
    }
    if (!std::isfinite(output_scale)) {
        throw std::invalid_argument("Tensor reduction output scale must be finite.");
    }
}

DataType Reducer::resolveOutputDataType(DataType input_dtype) const {
    requireSupportedFloatingStorageDType(input_dtype, "input");
    const DataType resolved = output_dtype.value_or(input_dtype);
    requireSupportedFloatingStorageDType(resolved, "output");
    return resolved;
}

size_t Reducer::queryWorkspaceSizeInBytes(const TensorDescriptor& input_descriptor,
                                          const Stream& stream) const {
    requireSupportedFloatingStorageDType(input_descriptor.getDataType(), "input");
    const ReductionGeometry geometry = analyzeValueGeometry(op, input_descriptor.getDimensions(), axes);
    const DataType resolved_output_dtype = resolveOutputDataType(input_descriptor.getDataType());
    ScopedGpu scoped_gpu(stream.getGpuNum());

    if (isOrdinaryDenseValueGeometry(geometry)) {
        const DenseReductionPlan plan = planProductionDenseValue(op,
                                                                 input_descriptor.getDimensions(),
                                                                 geometry.axes,
                                                                 input_descriptor.getDataType(),
                                                                 resolved_output_dtype,
                                                                 geometry.reduction_size,
                                                                 output_scale,
                                                                 tuning_depth,
                                                                 stream);
        return queryDenseValueExecutableWorkspace(op,
                                                  input_descriptor,
                                                  resolved_output_dtype,
                                                  plan,
                                                  geometry.reduction_size,
                                                  output_scale,
                                                  stream);
    }

    // A dense descriptor's geometry is always an ordinary dense one (only a view's carries layout candidates).
    throw std::logic_error("A dense value reduction descriptor did not classify as an ordinary dense geometry.");
}

float Reducer::getFp32EmptyReductionValue(ReductionOp op) {
    requireSupportedOperation(op);
    switch (op) {
        case ReductionOp::Sum:
        case ReductionOp::Mean:
        case ReductionOp::L1Norm:
        case ReductionOp::L2Norm:
        case ReductionOp::SumSquares:
            return 0.0f;
        case ReductionOp::Min:
            return std::numeric_limits<float>::infinity();
        case ReductionOp::Max:
            return -std::numeric_limits<float>::infinity();
        case ReductionOp::Product:
            return 1.0f;
    }
    throw std::invalid_argument("Unsupported tensor reduction operation.");
}

std::shared_ptr<StampedReducer> Reducer::stamp(const StridedSpan& input, const Stream& stream) const {
    requireGpuTensorView(input, "input");
    requireCompatibleStream(input, stream);
    const ReductionGeometry geometry =
        analyzeValueSpanGeometry(op, input, axes, resolveOutputDataType(input.getDataType()));
    const DataType resolved_output_dtype = resolveOutputDataType(input.getDataType());
    Tensor output(input.getPlacement(), TensorDescriptor(resolved_output_dtype, geometry.output_dimensions));
    return stampValidated(input, output, geometry, stream);
}

std::shared_ptr<StampedReducer> Reducer::stampValidated(const StridedSpan& input,
                                                        const Tensor& output,
                                                        const ReductionGeometry& geometry,
                                                        const Stream& stream) const {
    requireSupportedFloatingStorageDType(input.getDataType(), "input");
    requireSupportedFloatingStorageDType(output.getDataType(), "output");
    requireExpectedOutput(input, output, output.getDataType(), geometry);

    // A pitched view stamp() accepted (analyzeValueSpanGeometry) is read in place.
    if (geometry.pitched_layout.has_value()) {
        const std::optional<PitchedCompleteLaunch> launch =
            pitchedCompleteLaunch(input, axes, output.getDataType());
        if (!launch.has_value()) {
            throw std::logic_error("A pitched view reached the reducer without a PacketComplete launch.");
        }
        return stampPitchedComplete(input, output, geometry, *launch, stream);
    }

    ScopedGpu scoped_gpu(stream.getGpuNum());

    // A view whose reduced axes are all or a physical suffix runs as the dense tensor of its bytes; so does every
    // compact permutation whose retained axes keep their physical order (natural-output permutation-aware views, split
    // physical reductions). The stamp keeps the view as its input and the view's geometry (getPath() stays its
    // classification); the passes read the dense span. Such a view takes this branch whatever layout candidates the
    // classifier also noted (a candidate is only an alternative layout reading).
    const std::optional<ViewPhysicalDenseProblem> view_physical = viewPhysicalDenseProblem(input, geometry);
    if (view_physical.has_value() || (input.isDenseContiguous() && isOrdinaryDenseValueGeometry(geometry))) {
        const StridedSpan dense_input = view_physical.has_value() ? input.asDense(view_physical->dimensions) : input;
        std::string strategy_chain;
        const DenseReductionPlan plan = planProductionDenseValue(op,
                                                                 dense_input.getDimensions(),
                                                                 view_physical.has_value() ? view_physical->axes
                                                                                           : geometry.axes,
                                                                 input.getDataType(),
                                                                 output.getDataType(),
                                                                 geometry.reduction_size,
                                                                 output_scale,
                                                                 tuning_depth,
                                                                 stream,
                                                                 &strategy_chain,
                                                                 dense_input.alignmentBytes(16));
        if (view_physical.has_value()) strategy_chain = "view_physical:" + strategy_chain;
        DenseExecutablePlan executable_plan = ReductionInternal::materializeDenseValueExecutablePlan(op,
                                                                                  dense_input,
                                                                                  output,
                                                                                  plan,
                                                                                  geometry.reduction_size,
                                                                                  output_scale,
                                                                                  stream);
        auto stamped = std::shared_ptr<StampedReducer>(new StampedReducer(op,
                                                                            geometry,
                                                                            input,
                                                                            output,
                                                                            std::move(executable_plan),
                                                                            std::move(strategy_chain),
                                                                            true,
                                                                            true,
                                                                            geometry.reduction_size,
                                                                            output_scale,
                                                                            stream));
        return stamped;
    }

    if (geometry.path == ReductionLayout::DisjointRuns) {
        throw std::logic_error(
            "Ordinary dense DisjointRuns VALUE geometry must execute through DenseExecutablePlan.");
    }

    // Case C, a rank-1 strided view, runs StridedR (StridedReduction.h): the closed form at stride 0, else Split (no
    // contest; the grid follows from n). The stamp keeps the view and its geometry; the strategy chain is the launch's
    // name.
    if (geometry.device_transform_uses_affine_stride) {
        const ReductionInternal::StridedRLaunch launch =
            ReductionInternal::stridedRLaunchFor(input, output.getDataType(), static_cast<int>(stream.getGpuNum()));
        Tensor workspace;
        if (launch.workspace_bytes != 0) {
            workspace = Tensor(input.getPlacement(), TensorDescriptor(DataType::UINT8, {launch.workspace_bytes}));
            // Split's ticket starts at 0 and every launch leaves it there.
            ReductionInternal::stridedRInitWorkspace(StridedSpan(workspace), stream);
        }
        auto stamped = std::shared_ptr<StampedReducer>(new StampedReducer(op,
                                                                          geometry,
                                                                          input,
                                                                          output,
                                                                          launch.workspace_bytes,
                                                                          workspace,
                                                                          true,
                                                                          true,
                                                                          geometry.reduction_size,
                                                                          output_scale,
                                                                          stream));
        stamped->strided_r_launch = launch;
        stamped->modern_rk_strategy_chain = ReductionInternal::stridedRLaunchName(launch);
        return stamped;
    }

    // analyzeGeometry refuses every view the branches above do not run.
    throw std::logic_error("A value reduction geometry reached the reducer with no executor.");
}

std::shared_ptr<StampedReducer> Reducer::stamp(const StridedSpan& input,
                                               const Tensor& preallocated_output,
                                               const Stream& stream) const {
    requireGpuTensorView(input, "input");
    requireCompatibleStream(input, stream);
    const ReductionGeometry geometry =
        analyzeValueSpanGeometry(op, input, axes, resolveOutputDataType(input.getDataType()));
    const DataType resolved_output_dtype = resolveOutputDataType(input.getDataType());
    requireExpectedOutput(input, preallocated_output, resolved_output_dtype, geometry);
    return stampValidated(input, preallocated_output, geometry, stream);
}

std::shared_ptr<StampedReducer> Reducer::stampPitched(const StridedSpan& input,
                                                      uint32_t warps_per_output,
                                                      const Stream& stream) const {
    requireGpuTensorView(input, "input");
    requireCompatibleStream(input, stream);
    requireSupportedOperation(op);
    const std::optional<PitchedCompleteLaunch> launch =
        pitchedCompleteLaunch(input, axes, resolveOutputDataType(input.getDataType()), warps_per_output);
    if (!launch.has_value()) {
        throw NotImplementedException("RuntimeK PacketComplete cannot read this view at its row and slab pitches.");
    }
    const ReductionGeometry geometry = classifyGeometry(input.getDimensions(), input.getStridesElements(), axes);
    Tensor output(input.getPlacement(),
                  TensorDescriptor(resolveOutputDataType(input.getDataType()), geometry.output_dimensions));
    return stampPitchedComplete(input, output, geometry, *launch, stream);
}

// A pitched view read in place, one PacketComplete launch at its pitches. The stamp keeps the view and its geometry
// (getPath() stays its classification); the strategy chain is the launch's name.
std::shared_ptr<StampedReducer> Reducer::stampPitchedComplete(
    const StridedSpan& input,
    const Tensor& output,
    const ReductionGeometry& geometry,
    const PitchedCompleteLaunch& launch,
    const Stream& stream) const {
    requireSupportedFloatingStorageDType(input.getDataType(), "input");
    requireSupportedFloatingStorageDType(output.getDataType(), "output");
    requireExpectedOutput(input, output, output.getDataType(), geometry);
    if (!geometry.pitched_layout.has_value()) {
        throw std::logic_error("A pitched PacketComplete stamp requires a pitched view geometry.");
    }
    ScopedGpu scoped_gpu(stream.getGpuNum());
    (void)pitchedRowsSpan(input, *geometry.pitched_layout);  // throws if the rows leave the source
    auto stamped = std::shared_ptr<StampedReducer>(new StampedReducer(op,
                                                                      geometry,
                                                                      input,
                                                                      output,
                                                                      0,
                                                                      Tensor(),
                                                                      true,
                                                                      true,
                                                                      geometry.pitched_layout
                                                                                    ->reduction_size,
                                                                      output_scale,
                                                                      stream));
    stamped->pitched_complete_launch = launch;
    stamped->modern_rk_strategy_chain = "pitched_complete:w" + std::to_string(launch.warps_per_output) + "_t" +
                                        std::to_string(launch.tile_components);
    return stamped;
}

StampedReducer::StampedReducer(ReductionOp op,
                               ReductionGeometry geometry,
                               const StridedSpan& input,
                               const Tensor& output,
                               size_t temp_storage_bytes,
                               const Tensor& temp_storage,
                               bool apply_input_transform,
                               bool apply_output_finalize,
                               uint64_t semantic_total_reduction_size,
                               float output_scale,
                               const Stream& stream)
    : op(op),
      geometry(std::move(geometry)),
      input(input),
      output(output),
      temp_storage_bytes(temp_storage_bytes),
      temp_storage(temp_storage),
      apply_input_transform(apply_input_transform),
      apply_output_finalize(apply_output_finalize),
      semantic_total_reduction_size(semantic_total_reduction_size),
      output_scale(output_scale),
      stream(stream) {
    if (temp_storage_bytes != 0) {
        requireTempStorage(this->temp_storage, input.getPlacement(), temp_storage_bytes);
    } else if (this->temp_storage.isInitialized()) {
        throw std::invalid_argument("Zero-scratch reduction must not own a temporary-storage Tensor.");
    }
}

StampedReducer::StampedReducer(
    ReductionOp op,
    ReductionGeometry geometry,
    const StridedSpan& input,
    const Tensor& output,
    ReductionInternal::DenseExecutablePlan dense_executable_plan,
    std::string modern_rk_strategy_chain,
    bool apply_input_transform,
    bool apply_output_finalize,
    uint64_t semantic_total_reduction_size,
    float output_scale,
    const Stream& stream)
    : op(op),
      geometry(std::move(geometry)),
      input(input),
      output(output),
      temp_storage_bytes(dense_executable_plan.workspace_size_bytes),
      dense_executable_plan(std::move(dense_executable_plan)),
      modern_rk_strategy_chain(std::move(modern_rk_strategy_chain)),
      apply_input_transform(apply_input_transform),
      apply_output_finalize(apply_output_finalize),
      semantic_total_reduction_size(semantic_total_reduction_size),
      output_scale(output_scale),
      stream(stream) {
    // A view that runs as the dense tensor of its bytes (viewPhysicalDenseProblem) keeps its classification geometry,
    // which can still note a layout candidate (a natural-output permutation-aware view, case D).
    THOR_THROW_IF_FALSE(isOrdinaryDenseValueGeometry(this->geometry)
                        || viewPhysicalDenseProblem(this->input, this->geometry).has_value());
    THOR_THROW_IF_FALSE(!this->dense_executable_plan.passes.empty());

    size_t summed_workspace_bytes = 0;
    for (const ReductionInternal::DenseExecutablePass& pass : this->dense_executable_plan.passes) {
        THOR_THROW_IF_FALSE(pass.physical_plan != nullptr);
        if (summed_workspace_bytes > std::numeric_limits<size_t>::max() - pass.workspaceBytes()) {
            throw std::overflow_error("Dense executable plan workspace accounting overflows size_t.");
        }
        summed_workspace_bytes += pass.workspaceBytes();
    }
    THOR_THROW_IF_FALSE(summed_workspace_bytes == this->dense_executable_plan.workspace_size_bytes);
    THOR_THROW_IF_FALSE(this->dense_executable_plan.passes.back().role == ReductionInternal::DensePassRole::Complete
                        || this->dense_executable_plan.passes.back().role == ReductionInternal::DensePassRole::Final);
}

std::vector<DensePhysicalPassDiagnostic> StampedReducer::getDensePhysicalPassDiagnostics() const {
    std::vector<DensePhysicalPassDiagnostic> result;
    result.reserve(dense_executable_plan.passes.size());
    for (const auto& pass : dense_executable_plan.passes) {
        DensePhysicalPassDiagnostic info;
        info.role = pass.role;
        info.input_dtype = pass.input.getDataType();
        info.output_dtype = pass.output.getDataType();
        info.input_dimensions = pass.input.getDimensions();
        info.output_dimensions = pass.output.getDimensions();
        info.intermediate_bytes = pass.intermediate_bytes;
        info.scratch_bytes = pass.temp_storage_bytes;
        if (const auto* fp8 = dynamic_cast<const ReductionInternal::RKPhysicalPlan*>(
                pass.physical_plan.get())) {
            const auto& kernel = fp8->kernel;
            switch (kernel.family) {
                case ReductionInternal::RK::PhysicalFamily::OneWindow:
                    info.family = "retained_fp8_rk:one_window";
                    info.packet_bytes = kernel.c;
                    break;
                case ReductionInternal::RK::PhysicalFamily::DeepRSmallK:
                    info.family = "retained_fp8_rk:deep_r_small_k";
                    break;
                case ReductionInternal::RK::PhysicalFamily::AlignedK:
                    info.family = "retained_fp8_rk:aligned_k";
                    info.packet_bytes = kernel.a;
                    info.block_threads = static_cast<uint32_t>(kernel.b);
                    break;
                case ReductionInternal::RK::PhysicalFamily::RuntimeK:
                    info.family = "retained_fp8_rk:runtime_k";
                    break;
            }
            // An FP32 RK pass is "retained_fp32_rk:...". A 2-byte pass "retained_f16_rk:...".
            if (fp8->input_dtype == DataType::FP32) info.family.replace(0, 15, "retained_fp32_rk");
            if (kernel.element_bytes == 2) info.family.replace(0, 15, "retained_f16_rk");
            if (kernel.first_stage_ctas != 0) info.stage_blocks = kernel.first_stage_ctas;
            // The actual next tensor geometry is authoritative for staged N.
            info.shards_per_output = kernel.staged() ? fp8->output_dimensions.at(1) : 1;
        } else if (const auto* kr = dynamic_cast<const ReductionInternal::KRPhysicalPlan*>(
                       pass.physical_plan.get())) {
            info.family = retainedKRFamilyPrefix(kr->input_dtype) +
                          ReductionInternal::KR::krShapeName(kr->shape);
            info.block_threads = ReductionInternal::KR::KR_THREADS;
            info.shards_per_output = kr->shape.chunks();
        } else {
            throw std::logic_error("Dense VALUE pass without a retained physical plan");
        }
        result.push_back(std::move(info));
    }
    return result;
}

bool StampedReducer::usesModernRKFamilyPlan() const {
    if (dense_executable_plan.passes.empty()) {
        return false;
    }
    return std::all_of(dense_executable_plan.passes.begin(),
                       dense_executable_plan.passes.end(),
                       [](const ReductionInternal::DenseExecutablePass& pass) {
                           // Every RK pass is an RK physical plan.
                           return std::dynamic_pointer_cast<const ReductionInternal::RKPhysicalPlan>(
                                      pass.physical_plan) != nullptr;
                       });
}

std::vector<std::vector<uint32_t>> StampedReducer::getComposedStageAxes() const {
    if (geometry.path != ReductionLayout::DisjointRuns) {
        return {};
    }
    THOR_THROW_IF_FALSE(!dense_executable_plan.passes.empty());
    std::vector<std::vector<uint32_t>> stage_axes;
    stage_axes.reserve(dense_executable_plan.passes.size());
    for (const ReductionInternal::DenseExecutablePass& pass : dense_executable_plan.passes) {
        stage_axes.push_back(pass.geometry.axes);
    }
    return stage_axes;
}

void StampedReducer::run() { runOn(stream, output_scale); }

void StampedReducer::run(float runtime_output_scale) { runOn(stream, runtime_output_scale); }

void StampedReducer::runOn(Stream& run_stream) const { runOn(run_stream, output_scale); }

void StampedReducer::runOn(Stream& run_stream, float runtime_output_scale) const {
    THOR_THROW_IF_FALSE(std::isfinite(runtime_output_scale));
    requireCompatibleStream(input, run_stream);
    ScopedGpu scoped_gpu(run_stream.getGpuNum());

    const ReductionInternal::ReductionStageSemantics semantics =
        ReductionInternal::makeValueReductionStageSemantics(
            op,
            physicalStageRole(apply_input_transform, apply_output_finalize),
            semantic_total_reduction_size);

    if (pitched_complete_launch.has_value()) {
        launchPitchedComplete(*pitched_complete_launch,
                              semantics,
                              pitchedRowsSpan(input, *geometry.pitched_layout),
                              output_span,
                              geometry,
                              runtime_output_scale,
                              run_stream);
        return;
    }

    if (!dense_executable_plan.passes.empty()) {
        for (const ReductionInternal::DenseExecutablePass& pass : dense_executable_plan.passes) {
            ReductionInternal::ReductionStageSemantics stage_semantics =
                ReductionInternal::makeValueReductionStageSemantics(
                    op, pass.role, semantic_total_reduction_size);
            if (!apply_input_transform) {
                stage_semantics.input_transform = ReductionInternal::ReductionStageInputTransform::Identity;
            }
            if (!apply_output_finalize) {
                stage_semantics.finalize = ReductionInternal::ReductionStageFinalize::Identity;
                stage_semantics.finalize_divisor = 1;
            }

            const float stage_output_scale =
                pass.role == ReductionInternal::DensePassRole::Complete
                        || pass.role == ReductionInternal::DensePassRole::Final
                    ? runtime_output_scale
                    : 1.0f;

            // Every pass carries only its own actually required scratch.
            // Zero-scratch passes have an uninitialized Tensor handle.
            ReductionInternal::launchDensePhysicalPass(*pass.physical_plan,
                                                       stage_semantics,
                                                       pass.temp_storage_span,
                                                       pass.temp_storage_bytes,
                                                       pass.input,
                                                       pass.output_span,
                                                       pass.geometry,
                                                       stage_output_scale,
                                                       run_stream);
        }
        return;
    }

    if (strided_r_launch.has_value()) {
        ReductionInternal::launchStridedR(op, *strided_r_launch, input, output_span, temp_storage_span,
                                          runtime_output_scale, run_stream);
        return;
    }

    // Every stamp is a dense plan, StridedR or pitched.
    throw std::logic_error("A value reduction stamp has no executor.");
}

}  // namespace ThorImplementation
