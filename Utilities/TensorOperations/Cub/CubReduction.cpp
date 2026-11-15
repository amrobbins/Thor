#include "Utilities/TensorOperations/Cub/CubReduction.h"

#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Exceptions.h"
#include "Utilities/TensorOperations/Cub/CubDataTypePolicy.h"
#include "Utilities/TensorOperations/Cub/CubDevicePrimitiveSupport.h"
#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"
#include "Utilities/TensorOperations/Cub/DenseReductionFamilies.h"
#include "Utilities/TensorOperations/Cub/DenseReductionPlanner.h"
#include "Utilities/TensorOperations/Cub/DenseReductionSelectionCache.h"
#include "Utilities/TensorOperations/Cub/DenseProductionPhysicalSelectionContext.h"
#include "Utilities/TensorOperations/Cub/DensePhysicalPassExecution.h"
#include "Utilities/TensorOperations/Cub/DenseRetainedFp8ArgPhysicalPlan.h"
#include "Utilities/TensorOperations/Cub/DenseRetainedFp8RKPhysicalPlan.h"
#include "Utilities/TensorOperations/Cub/DenseRetainedFp8KRPhysicalPlan.h"
#include "Utilities/TensorOperations/Cub/DenseRetainedFp32ArgPhysicalPlan.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKLaunchLimits.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKPhysicalKernels.h"
#include "Utilities/TensorOperations/Cub/RaggedStampAutotuner.h"
#include "Utilities/TensorOperations/Cub/StridedReduction.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace ThorImplementation {

CubReductionInternal::CubReductionStageSemantics CubReductionInternal::makeValueReductionStageSemantics(
    CubReductionOp op, CubReductionStageRole role, uint64_t total_reduction_size) {
    CubReductionStageSemantics semantics;

    switch (op) {
        case CubReductionOp::Sum:
        case CubReductionOp::Mean:
        case CubReductionOp::L1Norm:
        case CubReductionOp::L2Norm:
        case CubReductionOp::SumSquares:
            semantics.combine = CubReductionStageCombine::Sum;
            break;
        case CubReductionOp::Product:
            semantics.combine = CubReductionStageCombine::Product;
            break;
        case CubReductionOp::Min:
            semantics.combine = CubReductionStageCombine::Minimum;
            break;
        case CubReductionOp::Max:
            semantics.combine = CubReductionStageCombine::Maximum;
            break;
        default:
            throw std::invalid_argument("Unsupported CUB tensor reduction operation.");
    }

    const bool applies_input_transform =
        role == CubReductionStageRole::Complete || role == CubReductionStageRole::First;
    if (applies_input_transform) {
        if (op == CubReductionOp::L1Norm) {
            semantics.input_transform = CubReductionStageInputTransform::AbsoluteValue;
        } else if (op == CubReductionOp::L2Norm || op == CubReductionOp::SumSquares) {
            semantics.input_transform = CubReductionStageInputTransform::Square;
        }
    }

    const bool applies_finalizer =
        role == CubReductionStageRole::Complete || role == CubReductionStageRole::Final;
    if (applies_finalizer) {
        if (op == CubReductionOp::Mean) {
            semantics.finalize = CubReductionStageFinalize::Divide;
            semantics.finalize_divisor = total_reduction_size;
        } else if (op == CubReductionOp::L2Norm) {
            semantics.finalize = CubReductionStageFinalize::SquareRoot;
        }
    }

    return semantics;
}

namespace {

using namespace CubDevicePrimitiveSupport;
using namespace CubReductionInternal;

// 1220 / 1254: the diagnostic family prefix of a retained R / KR VALUE pass by its input dtype.
[[nodiscard]] std::string retainedKRFamilyPrefix(DataType input_dtype) {
    if (input_dtype == DataType::FP32) return "retained_fp32_kr:";
    if (input_dtype == DataType::FP16 || input_dtype == DataType::BF16) return "retained_f16_kr:";
    return "retained_fp8_kr:";
}

[[nodiscard]] DenseReductionPlan planProductionDenseValue(CubReductionOp op,
                                                           const std::vector<uint64_t>& input_dimensions,
                                                           const std::vector<uint32_t>& axes,
                                                           DataType original_input_dtype,
                                                           DataType output_dtype,
                                                           uint64_t original_reduction_size,
                                                           float output_scale,
                                                           CubReductionTuningDepth tuning_depth,
                                                           const Stream& stream,
                                                           std::string* strategy_chain = nullptr,
                                                           uint64_t input_alignment_bytes = 16) {
    DenseReductionOutputSpec output_spec;
    output_spec.produce_value = true;
    output_spec.value_dtype = output_dtype;
    output_spec.produce_index = false;
    const DenseReductionProblem problem = makeInitialDenseReductionProblem(
        input_dimensions, axes, DenseReductionAggregateKind::Value, output_spec);

    // The context survives every physical selection made by this plan (the retained selectors read its stream).
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
            if (const auto fp8 = std::dynamic_pointer_cast<const DenseRetainedFp8RKPhysicalPlan>(
                    step.selected_candidate.physical_plan); fp8) {
                strategy_chain->append(fp8->input_dtype == DataType::FP32   ? "retained_fp32_rk:"  // 1220
                                       : fp8->kernel.element_bytes == 2 ? "retained_f16_rk:"   // 1254
                                                                        : "retained_fp8_rk:");
                switch (fp8->kernel.family) {
                    case Fp8RK::PhysicalFamily::OneWindow: strategy_chain->append("one_window"); break;
                    case Fp8RK::PhysicalFamily::DeepRSmallK: strategy_chain->append("deep_r_small_k"); break;
                    case Fp8RK::PhysicalFamily::AlignedK: strategy_chain->append("aligned_k"); break;
                    case Fp8RK::PhysicalFamily::RuntimeK: strategy_chain->append("runtime_k"); break;
                }
            } else if (const auto kr = std::dynamic_pointer_cast<const DenseRetainedFp8KRPhysicalPlan>(
                    step.selected_candidate.physical_plan); kr) {
                strategy_chain->append(retainedKRFamilyPrefix(kr->input_dtype) + Fp8KR::fp8KRShapeName(kr->shape));
            } else {
                throw std::logic_error("Dense VALUE plan selected a physical pass without a retained executor");
            }
        }
    }
    return plan;
}

[[nodiscard]] bool isOrdinaryDenseValueGeometry(const CubReductionGeometry& geometry) {
    const bool ordinary_path = geometry.path == CubReductionPath::DeviceTransformReduce
                               || geometry.path == CubReductionPath::ContiguousFixedSegment
                               || geometry.path == CubReductionPath::TiledFixedSegment
                               || geometry.path == CubReductionPath::ComposedDense;
    return ordinary_path
           && !geometry.permutation_aware_tiled_geometry.has_value()
           && !geometry.pitched_tiled_geometry.has_value()
           && !geometry.payload_transpose_tiled_geometry.has_value()
           && !geometry.tiled_output_permuted
           && !geometry.tiled_output_shared_transpose;
}

// 1365 (plans/Thor_View_Reduction_Plan_2026-11-08.md, phase 1; 1376, phase 5): the dense problem a view reduction is
// in storage order. A compact physical permutation whose retained axes keep their physical order is the bytes of a
// dense tensor reduced over its physical axes, and its outputs are already in the logical output's order. That covers
// - VIEW-DIRECT-1 (A): every axis reduced, a dense [input_elements] over axis 0;
// - VIEW-DIRECT-2A (B): the reduced axes the physical suffix, a dense [output_elements, reduction_size] over axis 1;
// - 1376, natural-output permutation-aware tiled views (D): one reduced physical block with an inner block, a dense
//   [outer, R, inner] over the R axis (the Thor tiled kernels were 1.04-15,000x slower in every 1375 audit cell);
// - 1376, split physical reductions (E): the reduced axes several physical runs, which no reducer owned (they threw).
// The physical non-singleton axes are collapsed into alternating retained / reduced runs. The problem runs the
// retained dense planner on StridedSpan::asDense of the input: the dense tensor's kernels, selection cache entries and
// bits. A permuted output order (VIEW-DIRECT-2B, the rotated tiled output) returns nothing here: 1390 refuses it, and
// the caller runs it in two parts (refusedViewSplit).
struct ViewPhysicalDenseProblem {
    std::vector<uint64_t> dimensions;
    std::vector<uint32_t> axes;
};

// The runs of a compact physical permutation (geometry.physical_layout_is_dense_permutation), if its retained axes keep
// their physical order and it reduces at least one non-singleton axis; else nothing.
[[nodiscard]] std::optional<ViewPhysicalDenseProblem> densePermutationRuns(const std::vector<uint64_t>& dimensions,
                                                                          const CubReductionGeometry& geometry) {
    if (!geometry.physical_layout_is_dense_permutation) return std::nullopt;
    std::vector<bool> reduced(dimensions.size(), false);
    for (uint32_t axis : geometry.axes) reduced[axis] = true;
    std::vector<uint32_t> logical_retained;
    for (uint32_t axis = 0; axis < dimensions.size(); ++axis) {
        if (!reduced[axis] && dimensions[axis] > 1) logical_retained.push_back(axis);
    }
    ViewPhysicalDenseProblem runs;
    std::vector<uint32_t> physical_retained;
    bool previous_reduced = false;
    for (uint32_t axis : geometry.physical_non_singleton_axis_order) {
        if (!reduced[axis]) physical_retained.push_back(axis);
        if (!runs.dimensions.empty() && reduced[axis] == previous_reduced) {
            runs.dimensions.back() *= dimensions[axis];
        } else {
            if (reduced[axis]) runs.axes.push_back(static_cast<uint32_t>(runs.dimensions.size()));
            runs.dimensions.push_back(dimensions[axis]);
        }
        previous_reduced = reduced[axis];
    }
    if (physical_retained != logical_retained || runs.axes.empty()) return std::nullopt;
    return runs;
}

[[nodiscard]] std::optional<ViewPhysicalDenseProblem> viewPhysicalDenseProblem(const std::vector<uint64_t>& dimensions,
                                                                              bool dense_contiguous,
                                                                              const CubReductionGeometry& geometry) {
    if (dense_contiguous || !geometry.physical_layout_is_dense_permutation ||
        geometry.device_transform_uses_affine_stride) {
        return std::nullopt;
    }
    std::optional<ViewPhysicalDenseProblem> runs = densePermutationRuns(dimensions, geometry);
    // A view of one element (every axis extent one, any strides) has no non-singleton axis: a dense [1].
    if (!runs.has_value() && geometry.output_elements == 1) {
        return ViewPhysicalDenseProblem{{geometry.input_elements}, {0}};
    }
    return runs;
}

[[nodiscard]] std::optional<ViewPhysicalDenseProblem> viewPhysicalDenseProblem(const StridedSpan& input,
                                                                              const CubReductionGeometry& geometry) {
    return viewPhysicalDenseProblem(input.getDimensions(), input.isDenseContiguous(), geometry);
}

// 1390 (plans/Thor_Tiled_View_Replacement_Plan_2026-11-12.md): the views the reducer refuses and how a caller runs
// them in two parts (CubReductionViewSplit, CubReduction.h). The Thor tiled kernels that owned them ran 5-22,000x
// slower than the dense stamp of the same bytes in the 1386 audit; 1387 ran them on the dense planner with a layout
// pass inside the reducer, and from 1390 the caller does the layout part with its dense copy (copyToDenseAsync).
// - rotated (F: a compact permutation [O, R, I] with output [I, O]; G: VIEW-DIRECT-2B [A, R, B, P] with output
//   [B, A, P]): the bytes are the dense [a, r, b x p] over axis 1, which writes [a, b, p]; the output is that read as
//   [b, a, p].
// - packed (H: VIEW-PITCHED-TILED, rows of contiguous elements with gaps between them): the caller reduces the
//   view's dense copy (its own dimensions and dtype) over the same axes. 1391 (phase 7 A): when the view's storage is
//   known, slicedRowSplit may read the rows at their pitch instead (CubReductionViewSplit::Kind::SlicedOutput).
// A compact permutation whose retained axes keep their physical order is not refused (viewPhysicalDenseProblem).
[[nodiscard]] std::optional<CubReductionViewSplit> refusedViewSplit(const std::vector<uint64_t>& dimensions,
                                                                    bool dense_contiguous,
                                                                    const CubReductionGeometry& geometry) {
    if (dense_contiguous || viewPhysicalDenseProblem(dimensions, dense_contiguous, geometry).has_value()) {
        return std::nullopt;
    }
    CubReductionViewSplit split;
    uint64_t reduction = 1;
    if (geometry.permutation_aware_tiled_geometry.has_value() &&
        geometry.permutation_aware_tiled_geometry->retained_output_order ==
            CubReductionTiledRetainedOutputOrder::PermutedInnerOuter) {
        const CubReductionPermutationAwareTiledGeometry& rotated = *geometry.permutation_aware_tiled_geometry;
        split.a = rotated.outer_size;
        split.b = rotated.inner_size;
        reduction = rotated.reduction_size;
    } else if (geometry.payload_transpose_tiled_geometry.has_value()) {
        const CubReductionPayloadTransposeTiledGeometry& payload = *geometry.payload_transpose_tiled_geometry;
        split.a = payload.a_size;
        split.b = payload.b_size;
        split.p = payload.payload_size;
        reduction = payload.reduction_size;
    } else if (geometry.pitched_tiled_geometry.has_value()) {
        split.kind = CubReductionViewSplit::Kind::PackedInput;
        split.dense_dimensions = dimensions;
        split.dense_axes = geometry.axes;
        return split;
    } else {
        return std::nullopt;
    }
    if (!geometry.physical_layout_is_dense_permutation ||
        split.a * reduction * split.b * split.p != geometry.input_elements) {
        throw std::logic_error("A rotated view reduction's geometry is not a compact [A, R, B, P] permutation.");
    }
    split.kind = CubReductionViewSplit::Kind::RotatedOutput;
    split.dense_dimensions = {split.a, reduction, split.b * split.p};
    split.dense_axes = {1};
    split.buffer_dimensions = {split.a, split.b * split.p};
    split.copy_dimensions = {split.b, split.a, split.p};
    split.copy_strides = {split.p, split.b * split.p, 1};
    return split;
}

// 1391 (plans/Thor_Tiled_View_Replacement_Plan_2026-11-12.md, phase 7 A): a pitched view [O, R, I] (rows of I
// contiguous elements at row pitch Sr, slabs at pitch So) whose slabs have no gap between them (So = R x Sr, or O = 1)
// is the dense [O, R, Sr] from its first element with the columns [I, Sr) dropped: its dense reduction over R into
// [O, Sr], then the copy of the [O, I] columns. That reads the gap elements, Sr - I past the last row too, so it needs
// that many readable elements of the view's source (`readable_elements`). Chosen when it moves fewer bytes than the
// packed copy: O R Sr b + O Sr c + 2 O I c against 3 O R I b + O I c, b and c the input and output element bytes.
// The view's logical axes are in physical order (the pitched classification flattens them in order), so its keepdims
// output is dense [O, I].
[[nodiscard]] std::optional<CubReductionViewSplit> slicedRowSplit(const CubReductionGeometry& geometry,
                                                                  uint64_t readable_elements,
                                                                  uint64_t input_bytes,
                                                                  uint64_t output_bytes) {
    if (!geometry.pitched_tiled_geometry.has_value()) return std::nullopt;
    const CubReductionPitchedTiledGeometry& pitched = *geometry.pitched_tiled_geometry;
    const uint64_t O = pitched.outer_size, R = pitched.reduction_size, I = pitched.inner_size;
    const uint64_t Sr = pitched.reduction_stride;
    if (R < 2 || Sr <= I || (O > 1 && pitched.outer_stride != R * Sr)) return std::nullopt;
    const long double rows = static_cast<long double>(O) * R;
    if (rows * Sr > static_cast<long double>(readable_elements)) return std::nullopt;
    const long double sliced = rows * Sr * input_bytes + static_cast<long double>(O) * (Sr + 2 * I) * output_bytes;
    const long double packed = 3.0L * rows * I * input_bytes + static_cast<long double>(O) * I * output_bytes;
    if (!(sliced < packed)) return std::nullopt;
    CubReductionViewSplit split;
    split.kind = CubReductionViewSplit::Kind::SlicedOutput;
    split.dense_dimensions = {O, R, Sr};
    split.dense_axes = {1};
    split.buffer_dimensions = {O, Sr};
    split.copy_dimensions = {O, I};
    split.copy_strides = {Sr, 1};
    split.rows_outer = O;
    split.rows = R;
    split.row_width = I;
    split.row_pitch = Sr;
    return split;
}

[[nodiscard]] bool isSupportedFloatingStorageDType(DataType dtype) {
    switch (dtype) {
#if THOR_CUB_ENABLE_FP8_TYPES
        case DataType::FP8_E4M3:
        case DataType::FP8_E5M2:
#endif
        case DataType::FP16:
        case DataType::BF16:
        case DataType::FP32:
            return true;
        default:  // 1258: no FP64 reductions (user decision 2026-11-05)
            return false;
    }
}

void requireSupportedFloatingStorageDType(DataType dtype, const char* role) {
    if (!isSupportedFloatingStorageDType(dtype)) {
        throw std::invalid_argument(std::string("CUB tensor reduction does not support ") + role + " dtype " +
                                    dtypeName(dtype) +
                                    ". Supported storage dtypes are FP8 (per THOR_CUB_ENABLE_FP8_TYPES), FP16, BF16 "
                                    "and FP32; TF32 is not a storage dtype.");
    }
}

[[nodiscard]] bool isSupportedArgIndexDType(DataType dtype) {
    return dtype == DataType::UINT32 || dtype == DataType::UINT64;
}

void requireSupportedArgIndexDType(DataType dtype) {
    if (!isSupportedArgIndexDType(dtype)) {
        throw std::invalid_argument("CUB arg reduction index output dtype must be UINT32 or UINT64.");
    }
}

// Tensor or StridedSpan (both read their placement the same way).
template <typename TensorLike>
void requireCompatibleStream(const TensorLike& input, const Stream& stream) {
    if (!stream.isInitialized()) {
        throw std::invalid_argument("CUB tensor reduction requires an initialized stream.");
    }
    if (input.getPlacement().getDeviceNum() != stream.getGpuNum()) {
        throw std::invalid_argument("CUB tensor reduction stream must belong to the input tensor's GPU.");
    }
}

template <typename TensorLike>
void requireGpuTensorView(const TensorLike& tensor, const char* name) {
    if (!tensor.isInitialized()) {
        throw std::invalid_argument(std::string(name) + " must be initialized.");
    }
    if (tensor.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU) {
        throw std::invalid_argument(std::string(name) + " must be a GPU tensor.");
    }
}


[[nodiscard]] std::vector<uint64_t> stripSingletonDimensions(const std::vector<uint64_t>& dimensions) {
    std::vector<uint64_t> stripped;
    stripped.reserve(dimensions.size());
    for (uint64_t dimension : dimensions) {
        if (dimension != 1) {
            stripped.push_back(dimension);
        }
    }
    if (stripped.empty()) {
        stripped.push_back(1);
    }
    return stripped;
}

[[nodiscard]] bool isDirectDenseReductionPath(CubReductionPath path) {
    return path == CubReductionPath::DeviceTransformReduce || path == CubReductionPath::ContiguousFixedSegment
           || path == CubReductionPath::TiledFixedSegment;
}

[[nodiscard]] bool isOrdainedDenseReductionPath(CubReductionPath path) {
    return isDirectDenseReductionPath(path) || path == CubReductionPath::ComposedDense;
}

void requireOrdainedDenseReductionPath(const CubReductionGeometry& geometry) {
    if (isOrdainedDenseReductionPath(geometry.path)) {
        return;
    }

    throw std::logic_error(
        "DENSE-GATE-FINAL invariant violated: ordinary dense tensor reduction was not assigned to an ordained "
        "dense execution family.");
}

[[nodiscard]] CubReductionInternal::CubReductionStageRole physicalStageRole(bool apply_input_transform,
                                                                             bool apply_output_finalize) {
    using CubReductionInternal::CubReductionStageRole;
    if (apply_input_transform && apply_output_finalize) {
        return CubReductionStageRole::Complete;
    }
    if (apply_input_transform) {
        return CubReductionStageRole::First;
    }
    if (apply_output_finalize) {
        return CubReductionStageRole::Final;
    }
    return CubReductionStageRole::Intermediate;
}

[[nodiscard]] CubReductionDenseRunGeometry analyzeDenseRuns(const std::vector<uint64_t>& input_dimensions,
                                                               const std::vector<uint64_t>& dense_strides,
                                                               const std::vector<uint32_t>& axes) {
    CubReductionDenseRunGeometry dense_runs;
    dense_runs.runs.reserve(input_dimensions.size());

    size_t reduced_cursor = 0;
    for (uint32_t dimension = 0; dimension < input_dimensions.size(); ++dimension) {
        const bool reduced = reduced_cursor < axes.size() && axes[reduced_cursor] == dimension;
        if (reduced) {
            ++reduced_cursor;
        }

        // Singleton dimensions carry no runtime coordinate and do not break a physically linear run.
        if (input_dimensions[dimension] == 1) {
            continue;
        }

        const CubReductionDenseRunKind kind =
            reduced ? CubReductionDenseRunKind::Reduced : CubReductionDenseRunKind::Retained;
        if (!dense_runs.runs.empty() && dense_runs.runs.back().kind == kind) {
            CubReductionDenseRun& run = dense_runs.runs.back();
            if (run.extent > std::numeric_limits<uint64_t>::max() / input_dimensions[dimension]) {
                throw std::invalid_argument("CUB tensor reduction dense run extent overflows uint64_t.");
            }
            run.extent *= input_dimensions[dimension];
            run.physical_stride = dense_strides[dimension];
            continue;
        }

        dense_runs.runs.push_back(CubReductionDenseRun{
            .kind = kind,
            .extent = input_dimensions[dimension],
            .physical_stride = dense_strides[dimension],
            .domain_stride = 1,
        });
    }

    uint64_t reduced_domain_stride = 1;
    uint64_t retained_domain_stride = 1;
    for (size_t run_index = dense_runs.runs.size(); run_index-- > 0;) {
        CubReductionDenseRun& run = dense_runs.runs[run_index];
        if (run.kind == CubReductionDenseRunKind::Reduced) {
            run.domain_stride = reduced_domain_stride;
            if (reduced_domain_stride > std::numeric_limits<uint64_t>::max() / run.extent) {
                throw std::invalid_argument("CUB tensor reduction dense reduced-run domain overflows uint64_t.");
            }
            reduced_domain_stride *= run.extent;
            ++dense_runs.reduced_run_count;
        } else {
            run.domain_stride = retained_domain_stride;
            if (retained_domain_stride > std::numeric_limits<uint64_t>::max() / run.extent) {
                throw std::invalid_argument("CUB tensor reduction dense retained-run domain overflows uint64_t.");
            }
            retained_domain_stride *= run.extent;
            ++dense_runs.retained_run_count;
        }
    }

    return dense_runs;
}

[[nodiscard]] bool analyzeDensePhysicalPermutation(const std::vector<uint64_t>& input_dimensions,
                                                   const std::vector<uint64_t>& input_strides,
                                                   CubReductionGeometry& geometry) {
    std::vector<uint32_t> physical_order;
    physical_order.reserve(input_dimensions.size());
    for (uint32_t axis = 0; axis < input_dimensions.size(); ++axis) {
        if (input_dimensions[axis] > 1) {
            if (input_strides[axis] == 0) {
                return false;
            }
            physical_order.push_back(axis);
        }
    }

    std::sort(physical_order.begin(), physical_order.end(), [&](uint32_t lhs, uint32_t rhs) {
        if (input_strides[lhs] != input_strides[rhs]) {
            return input_strides[lhs] > input_strides[rhs];
        }
        return lhs < rhs;
    });

    uint64_t expected_stride = 1;
    for (size_t physical = physical_order.size(); physical-- > 0;) {
        const uint32_t axis = physical_order[physical];
        if (input_strides[axis] != expected_stride) {
            return false;
        }
        if (expected_stride > std::numeric_limits<uint64_t>::max() / input_dimensions[axis]) {
            return false;
        }
        expected_stride *= input_dimensions[axis];
    }

    if (expected_stride != geometry.input_elements) {
        return false;
    }

    geometry.physical_layout_is_dense_permutation = true;
    geometry.physical_non_singleton_axis_order = std::move(physical_order);
    return true;
}

[[nodiscard]] bool analyzePermutationAwareContiguousCandidate(const std::vector<uint64_t>& input_dimensions,
                                                               const std::vector<uint32_t>& axes,
                                                               CubReductionGeometry& geometry) {
    geometry.permutation_aware_contiguous_segments = false;
    if (!geometry.physical_layout_is_dense_permutation || geometry.output_elements <= 1) {
        return false;
    }

    std::vector<bool> reduced(input_dimensions.size(), false);
    for (uint32_t axis : axes) {
        reduced[axis] = true;
    }

    const std::vector<uint32_t>& physical_order = geometry.physical_non_singleton_axis_order;
    size_t first_reduced = physical_order.size();
    for (size_t physical = 0; physical < physical_order.size(); ++physical) {
        if (reduced[physical_order[physical]]) {
            first_reduced = physical;
            break;
        }
    }
    if (first_reduced == physical_order.size()) {
        return false;
    }

    // VIEW-DIRECT-2A is deliberately the physically trailing case only. Once the first non-singleton reduced axis is
    // reached in physical order, every remaining non-singleton physical axis must also be reduced. That makes each
    // retained output one ordinary contiguous segment of exactly geometry.reduction_size values.
    for (size_t physical = first_reduced; physical < physical_order.size(); ++physical) {
        if (!reduced[physical_order[physical]]) {
            return false;
        }
    }

    std::vector<uint32_t> physical_retained_axes;
    physical_retained_axes.reserve(first_reduced);
    for (size_t physical = 0; physical < first_reduced; ++physical) {
        physical_retained_axes.push_back(physical_order[physical]);
    }

    std::vector<uint32_t> logical_non_singleton_retained_axes;
    logical_non_singleton_retained_axes.reserve(input_dimensions.size() - axes.size());
    for (uint32_t axis = 0; axis < input_dimensions.size(); ++axis) {
        if (!reduced[axis] && input_dimensions[axis] > 1) {
            logical_non_singleton_retained_axes.push_back(axis);
        }
    }

    // The dense [outer, R] reading emits segments in physical prefix order. VIEW-DIRECT-2A intentionally activates only
    // when that order is already Thor's dense logical output order. If these differ, input traversal is still usable
    // but an output permutation is required; that is the separate VIEW-DIRECT-2B problem.
    if (physical_retained_axes != logical_non_singleton_retained_axes) {
        return false;
    }

    uint64_t physical_reduction_size = 1;
    for (size_t physical = first_reduced; physical < physical_order.size(); ++physical) {
        const uint32_t axis = physical_order[physical];
        if (physical_reduction_size > std::numeric_limits<uint64_t>::max() / input_dimensions[axis]) {
            throw std::invalid_argument("CUB permutation-aware contiguous reduction size overflows uint64_t.");
        }
        physical_reduction_size *= input_dimensions[axis];
    }

    uint64_t physical_output_elements = 1;
    for (uint32_t axis : physical_retained_axes) {
        if (physical_output_elements > std::numeric_limits<uint64_t>::max() / input_dimensions[axis]) {
            throw std::invalid_argument("CUB permutation-aware contiguous output size overflows uint64_t.");
        }
        physical_output_elements *= input_dimensions[axis];
    }

    if (physical_reduction_size != geometry.reduction_size || physical_output_elements != geometry.output_elements) {
        throw std::logic_error("CUB permutation-aware contiguous geometry is internally inconsistent.");
    }

    geometry.permutation_aware_contiguous_segments = true;
    return true;
}

void analyzePermutationAwareTiledCandidate(const std::vector<uint64_t>& input_dimensions,
                                           const std::vector<uint32_t>& axes,
                                           CubReductionGeometry& geometry) {
    if (!geometry.physical_layout_is_dense_permutation || geometry.output_elements <= 1) {
        return;
    }

    std::vector<bool> reduced(input_dimensions.size(), false);
    for (uint32_t axis : axes) {
        reduced[axis] = true;
    }

    const std::vector<uint32_t>& physical_order = geometry.physical_non_singleton_axis_order;
    size_t first_reduced = physical_order.size();
    size_t last_reduced = 0;
    bool found_non_singleton_reduction = false;
    for (size_t physical = 0; physical < physical_order.size(); ++physical) {
        if (!reduced[physical_order[physical]]) {
            continue;
        }
        if (!found_non_singleton_reduction) {
            first_reduced = physical;
        }
        last_reduced = physical;
        found_non_singleton_reduction = true;
    }
    if (!found_non_singleton_reduction) {
        return;
    }
    for (size_t physical = first_reduced; physical <= last_reduced; ++physical) {
        if (!reduced[physical_order[physical]]) {
            return;
        }
    }

    CubReductionPermutationAwareTiledGeometry candidate;
    for (size_t physical = 0; physical < first_reduced; ++physical) {
        candidate.physical_outer_axes.push_back(physical_order[physical]);
    }
    for (size_t physical = first_reduced; physical <= last_reduced; ++physical) {
        candidate.physical_reduction_axes.push_back(physical_order[physical]);
    }
    for (size_t physical = last_reduced + 1; physical < physical_order.size(); ++physical) {
        candidate.physical_inner_axes.push_back(physical_order[physical]);
    }

    auto productOfAxes = [&](const std::vector<uint32_t>& physical_axes) {
        uint64_t product = 1;
        for (uint32_t axis : physical_axes) {
            if (product > std::numeric_limits<uint64_t>::max() / input_dimensions[axis]) {
                throw std::invalid_argument("CUB permutation-aware tiled geometry size overflows uint64_t.");
            }
            product *= input_dimensions[axis];
        }
        return product;
    };

    candidate.outer_size = productOfAxes(candidate.physical_outer_axes);
    candidate.reduction_size = geometry.reduction_size;
    candidate.inner_size = productOfAxes(candidate.physical_inner_axes);

    // This production candidate intentionally targets the tuned tiled family.  A physically trailing reduction has
    // inner_size == 1 and belongs to the contiguous-segment family rather than this path.
    if (candidate.inner_size <= 1) {
        return;
    }

    for (uint32_t axis = 0; axis < input_dimensions.size(); ++axis) {
        if (!reduced[axis] && input_dimensions[axis] > 1) {
            candidate.logical_non_singleton_retained_axes.push_back(axis);
        }
    }

    std::vector<uint32_t> natural_retained_axes = candidate.physical_outer_axes;
    natural_retained_axes.insert(
        natural_retained_axes.end(), candidate.physical_inner_axes.begin(), candidate.physical_inner_axes.end());

    std::vector<uint32_t> permuted_retained_axes = candidate.physical_inner_axes;
    permuted_retained_axes.insert(
        permuted_retained_axes.end(), candidate.physical_outer_axes.begin(), candidate.physical_outer_axes.end());

    if (candidate.logical_non_singleton_retained_axes == natural_retained_axes) {
        candidate.retained_output_order = CubReductionTiledRetainedOutputOrder::NaturalOuterInner;
    } else if (candidate.logical_non_singleton_retained_axes == permuted_retained_axes) {
        candidate.retained_output_order = CubReductionTiledRetainedOutputOrder::PermutedInnerOuter;
    } else {
        return;
    }

    geometry.permutation_aware_tiled_geometry = std::move(candidate);
}

void analyzePayloadTransposeTiledCandidate(const std::vector<uint64_t>& input_dimensions,
                                            const std::vector<uint32_t>& axes,
                                            CubReductionGeometry& geometry) {
    if (!geometry.physical_layout_is_dense_permutation || geometry.output_elements <= 1) {
        return;
    }

    std::vector<bool> reduced(input_dimensions.size(), false);
    for (uint32_t axis : axes) {
        reduced[axis] = true;
    }

    const std::vector<uint32_t>& physical_order = geometry.physical_non_singleton_axis_order;
    size_t first_reduced = physical_order.size();
    size_t last_reduced = 0;
    bool found_non_singleton_reduction = false;
    for (size_t physical = 0; physical < physical_order.size(); ++physical) {
        if (!reduced[physical_order[physical]]) {
            continue;
        }
        if (!found_non_singleton_reduction) {
            first_reduced = physical;
        }
        last_reduced = physical;
        found_non_singleton_reduction = true;
    }
    if (!found_non_singleton_reduction) {
        return;
    }
    for (size_t physical = first_reduced; physical <= last_reduced; ++physical) {
        if (!reduced[physical_order[physical]]) {
            return;
        }
    }

    // VIEW-DIRECT-2B is deliberately the middle-reduction case. A and B must both be genuine retained physical
    // groups; physically trailing output reorder remains outside this strategy.
    if (first_reduced == 0 || last_reduced + 1 >= physical_order.size()) {
        return;
    }

    CubReductionPayloadTransposeTiledGeometry candidate;
    candidate.physical_a_axes.assign(physical_order.begin(), physical_order.begin() + first_reduced);
    candidate.physical_reduction_axes.assign(
        physical_order.begin() + first_reduced, physical_order.begin() + last_reduced + 1);

    std::vector<uint32_t> physical_inner_axes(
        physical_order.begin() + last_reduced + 1, physical_order.end());
    std::vector<uint32_t> logical_retained_axes;
    logical_retained_axes.reserve(input_dimensions.size() - axes.size());
    for (uint32_t axis = 0; axis < input_dimensions.size(); ++axis) {
        if (!reduced[axis] && input_dimensions[axis] > 1) {
            logical_retained_axes.push_back(axis);
        }
    }

    // Split the physical suffix as [B..., payload...] and accept only the exact logical retained ordering
    // [B..., A..., payload...]. This is the payload-preserving A/B rotation from the VIEW-DIRECT-2B design, not a
    // general permutation recognizer. B must contain at least one non-singleton axis; payload may be scalar (empty).
    bool matched = false;
    for (size_t b_axis_count = 1; b_axis_count <= physical_inner_axes.size(); ++b_axis_count) {
        std::vector<uint32_t> requested_order;
        requested_order.reserve(logical_retained_axes.size());
        requested_order.insert(
            requested_order.end(), physical_inner_axes.begin(), physical_inner_axes.begin() + b_axis_count);
        requested_order.insert(
            requested_order.end(), candidate.physical_a_axes.begin(), candidate.physical_a_axes.end());
        requested_order.insert(
            requested_order.end(), physical_inner_axes.begin() + b_axis_count, physical_inner_axes.end());
        if (requested_order != logical_retained_axes) {
            continue;
        }
        candidate.physical_b_axes.assign(
            physical_inner_axes.begin(), physical_inner_axes.begin() + b_axis_count);
        candidate.physical_payload_axes.assign(
            physical_inner_axes.begin() + b_axis_count, physical_inner_axes.end());
        matched = true;
        break;
    }
    if (!matched) {
        return;
    }

    auto productOfAxes = [&](const std::vector<uint32_t>& physical_axes, const char* label) {
        uint64_t product = 1;
        for (uint32_t axis : physical_axes) {
            if (product > std::numeric_limits<uint64_t>::max() / input_dimensions[axis]) {
                throw std::invalid_argument(std::string("CUB VIEW-DIRECT-2B ") + label + " size overflows uint64_t.");
            }
            product *= input_dimensions[axis];
        }
        return product;
    };

    candidate.a_size = productOfAxes(candidate.physical_a_axes, "A");
    candidate.reduction_size = productOfAxes(candidate.physical_reduction_axes, "reduction");
    candidate.b_size = productOfAxes(candidate.physical_b_axes, "B");
    candidate.payload_size = productOfAxes(candidate.physical_payload_axes, "payload");

    if (candidate.a_size <= 1 || candidate.b_size <= 1 || candidate.reduction_size != geometry.reduction_size) {
        return;
    }
    if (candidate.a_size > std::numeric_limits<uint64_t>::max() / candidate.b_size
        || candidate.a_size * candidate.b_size > std::numeric_limits<uint64_t>::max() / candidate.payload_size
        || candidate.a_size * candidate.b_size * candidate.payload_size != geometry.output_elements) {
        throw std::logic_error("CUB VIEW-DIRECT-2B retained-output geometry is internally inconsistent.");
    }

    geometry.payload_transpose_tiled_geometry = std::move(candidate);
}

void activatePayloadTransposeTiledCandidate(CubReductionGeometry& geometry) {
    if (!geometry.payload_transpose_tiled_geometry.has_value()) {
        throw std::logic_error("CUB VIEW-DIRECT-2B activation requires detected payload-transpose geometry.");
    }
    const CubReductionPayloadTransposeTiledGeometry& candidate =
        geometry.payload_transpose_tiled_geometry.value();
    if (candidate.reduction_size != geometry.reduction_size || candidate.a_size <= 1 || candidate.b_size <= 1) {
        throw std::logic_error("CUB VIEW-DIRECT-2B geometry is internally inconsistent.");
    }
    geometry.path = CubReductionPath::TiledFixedSegment;
}

void activatePermutationAwareTiledCandidate(CubReductionGeometry& geometry) {
    if (!geometry.permutation_aware_tiled_geometry.has_value()) {
        throw std::logic_error("CUB permutation-aware tiled activation requires detected physical geometry.");
    }

    const CubReductionPermutationAwareTiledGeometry& candidate = geometry.permutation_aware_tiled_geometry.value();
    if (candidate.inner_size <= 1 || candidate.reduction_size != geometry.reduction_size) {
        throw std::logic_error("CUB permutation-aware tiled geometry is internally inconsistent.");
    }
    if (candidate.outer_size != 0
        && candidate.inner_size > std::numeric_limits<uint64_t>::max() / candidate.outer_size) {
        throw std::logic_error("CUB permutation-aware tiled output size overflows uint64_t.");
    }
    if (candidate.outer_size * candidate.inner_size != geometry.output_elements) {
        throw std::logic_error("CUB permutation-aware tiled retained-output size is internally inconsistent.");
    }

    // The tuned TiledFixedSegment kernels only need the visible pointer and the dense physical
    // [outer,reduction,inner] extents. They do not decode logical coordinates, so a zero-copy permutation view can
    // use exactly the same source traversal as its underlying dense storage.
    geometry.outer_size = candidate.outer_size;
    geometry.reduction_size = candidate.reduction_size;
    geometry.inner_size = candidate.inner_size;
    geometry.tiled_output_shared_transpose = false;

    switch (candidate.retained_output_order) {
        case CubReductionTiledRetainedOutputOrder::NaturalOuterInner:
            geometry.tiled_output_outer_stride = candidate.inner_size;
            geometry.tiled_output_inner_stride = 1;
            geometry.tiled_output_permuted = false;
            break;
        case CubReductionTiledRetainedOutputOrder::PermutedInnerOuter:
            geometry.tiled_output_outer_stride = 1;
            geometry.tiled_output_inner_stride = candidate.outer_size;
            geometry.tiled_output_permuted = true;
            geometry.tiled_output_shared_transpose = true;
            break;
    }

    geometry.path = CubReductionPath::TiledFixedSegment;
}

struct FlattenedAffineGroup {
    uint64_t extent = 1;
    uint64_t stride = 1;
    bool has_non_singleton_axis = false;
};

[[nodiscard]] std::optional<FlattenedAffineGroup> flattenAffineAxisRange(
    const std::vector<uint64_t>& input_dimensions,
    const std::vector<uint64_t>& input_strides,
    uint32_t first_axis,
    uint32_t last_axis_exclusive) {
    FlattenedAffineGroup group;
    if (first_axis >= last_axis_exclusive) {
        return group;
    }

    uint64_t expected_outer_stride = 0;
    for (uint32_t axis = last_axis_exclusive; axis-- > first_axis;) {
        const uint64_t dimension = input_dimensions[axis];
        const uint64_t stride = input_strides[axis];
        if (group.extent > std::numeric_limits<uint64_t>::max() / dimension) {
            return std::nullopt;
        }
        group.extent *= dimension;

        if (dimension == 1) {
            continue;
        }
        if (stride == 0) {
            return std::nullopt;
        }

        if (!group.has_non_singleton_axis) {
            group.stride = stride;
            if (dimension > std::numeric_limits<uint64_t>::max() / stride) {
                return std::nullopt;
            }
            expected_outer_stride = dimension * stride;
            group.has_non_singleton_axis = true;
            continue;
        }

        if (stride != expected_outer_stride) {
            return std::nullopt;
        }
        if (dimension > std::numeric_limits<uint64_t>::max() / stride) {
            return std::nullopt;
        }
        expected_outer_stride = dimension * stride;
    }

    return group;
}

void analyzePitchedTiledCandidate(const std::vector<uint64_t>& input_dimensions,
                                  const std::vector<uint64_t>& input_strides,
                                  const std::vector<uint32_t>& axes,
                                  CubReductionGeometry& geometry) {
    if (input_dimensions.size() <= 1 || axes.empty() || !geometry.reduced_axes_are_contiguous) {
        return;
    }

    const uint32_t first_reduced_axis = axes.front();
    const uint32_t last_reduced_axis = axes.back();

    const std::optional<FlattenedAffineGroup> outer =
        flattenAffineAxisRange(input_dimensions, input_strides, 0, first_reduced_axis);
    const std::optional<FlattenedAffineGroup> reduction =
        flattenAffineAxisRange(input_dimensions, input_strides, first_reduced_axis, last_reduced_axis + 1);
    const std::optional<FlattenedAffineGroup> inner = flattenAffineAxisRange(
        input_dimensions, input_strides, last_reduced_axis + 1, static_cast<uint32_t>(input_dimensions.size()));
    if (!outer.has_value() || !reduction.has_value() || !inner.has_value()) {
        return;
    }

    if (outer->extent != geometry.outer_size || reduction->extent != geometry.reduction_size
        || inner->extent != geometry.inner_size) {
        throw std::logic_error("CUB pitched tiled flattened extents are internally inconsistent.");
    }

    // The dedicated kernel assigns adjacent retained components to adjacent lanes. Keep that access exactly
    // contiguous so every active warp issues coalesced reads. Singleton-only inner groups have extent one and need
    // no physical stride constraint.
    if (inner->extent > 1 && (!inner->has_non_singleton_axis || inner->stride != 1)) {
        return;
    }

    const uint64_t reduction_stride =
        reduction->has_non_singleton_axis ? reduction->stride : std::max<uint64_t>(inner->extent, 1);
    if (reduction->extent > 1 && reduction_stride < inner->extent) {
        return;
    }

    uint64_t reduction_span = inner->extent;
    if (reduction->extent > 1) {
        const uint64_t last_reduction = reduction->extent - 1;
        if (last_reduction > (std::numeric_limits<uint64_t>::max() - inner->extent) / reduction_stride) {
            return;
        }
        reduction_span = last_reduction * reduction_stride + inner->extent;
    }

    uint64_t outer_stride = 0;
    if (outer->extent > 1) {
        if (!outer->has_non_singleton_axis) {
            return;
        }
        outer_stride = outer->stride;
        // Deliberately exclude overlapping/broadcast aliases. Adjacent flattened outer entries must own disjoint
        // pitched reduction slabs. Gaps are allowed and are exactly what repeated-label diagonal views introduce.
        if (outer_stride < reduction_span) {
            return;
        }
    }

    // Prove the complete reachable source offset also fits the uint64_t arithmetic used by the kernel. Tensor views
    // already validate their backing allocation; this check is specifically about keeping the affine address math exact.
    uint64_t max_source_offset = inner->extent - 1;
    const auto appendOffset = [&](uint64_t count_minus_one, uint64_t stride) -> bool {
        if (count_minus_one == 0) {
            return true;
        }
        if (stride > (std::numeric_limits<uint64_t>::max() - max_source_offset) / count_minus_one) {
            return false;
        }
        max_source_offset += count_minus_one * stride;
        return true;
    };
    if (!appendOffset(reduction->extent - 1, reduction_stride)
        || !appendOffset(outer->extent - 1, outer_stride)) {
        return;
    }

    CubReductionPitchedTiledGeometry candidate;
    candidate.outer_size = outer->extent;
    candidate.reduction_size = reduction->extent;
    candidate.inner_size = inner->extent;
    candidate.outer_stride = outer_stride;
    candidate.reduction_stride = reduction_stride;
    geometry.pitched_tiled_geometry = candidate;
}

void activatePitchedTiledCandidate(CubReductionGeometry& geometry) {
    if (!geometry.pitched_tiled_geometry.has_value()) {
        throw std::logic_error("CUB pitched tiled activation requires detected affine geometry.");
    }

    const CubReductionPitchedTiledGeometry& candidate = geometry.pitched_tiled_geometry.value();
    if (candidate.outer_size != geometry.outer_size || candidate.reduction_size != geometry.reduction_size
        || candidate.inner_size != geometry.inner_size) {
        throw std::logic_error("CUB pitched tiled geometry is internally inconsistent.");
    }
    if (candidate.outer_size != 0
        && candidate.inner_size > std::numeric_limits<uint64_t>::max() / candidate.outer_size) {
        throw std::logic_error("CUB pitched tiled output size overflows uint64_t.");
    }
    if (candidate.outer_size * candidate.inner_size != geometry.output_elements) {
        throw std::logic_error("CUB pitched tiled retained-output size is internally inconsistent.");
    }

    geometry.tiled_output_outer_stride = candidate.inner_size;
    geometry.tiled_output_inner_stride = 1;
    geometry.tiled_output_permuted = false;
    geometry.tiled_output_shared_transpose = false;
    geometry.path = CubReductionPath::TiledFixedSegment;
}

[[nodiscard]] uint64_t segmentedElementsPerValue(const Tensor& input) {
    const std::vector<uint64_t>& dimensions = input.getDimensions();
    if (dimensions.empty()) {
        throw std::invalid_argument("CUB segmented reduction requires values with rank >= 1.");
    }
    uint64_t elements_per_value = 1;
    for (size_t axis = 1; axis < dimensions.size(); ++axis) {
        if (dimensions[axis] != 0
            && elements_per_value > std::numeric_limits<uint64_t>::max() / dimensions[axis]) {
            throw std::invalid_argument("CUB segmented-reduction trailing value size overflows uint64_t.");
        }
        elements_per_value *= dimensions[axis];
    }
    if (elements_per_value == 0) {
        throw std::invalid_argument("CUB segmented reduction requires a non-zero trailing value size.");
    }
    return elements_per_value;
}

struct TensorReachableAddressSpan {
    uintptr_t begin = 0;
    uintptr_t end = 0;
};

template <typename TensorLike>
[[nodiscard]] TensorReachableAddressSpan tensorReachableAddressSpan(const TensorLike& tensor) {
    const std::vector<uint64_t> dimensions = tensor.getDimensions();
    const std::vector<uint64_t> strides = tensor.getStridesElements();
    if (dimensions.size() != strides.size() || dimensions.empty()) {
        throw std::logic_error("CUB tensor reduction received invalid tensor view metadata while checking storage overlap.");
    }

    uint64_t max_element_offset = 0;
    for (size_t dimension = 0; dimension < dimensions.size(); ++dimension) {
        const uint64_t extent_minus_one = dimensions[dimension] - 1;
        if (extent_minus_one != 0
            && strides[dimension] > (std::numeric_limits<uint64_t>::max() - max_element_offset) / extent_minus_one) {
            throw std::invalid_argument("CUB tensor reduction tensor view address span overflows uint64_t.");
        }
        max_element_offset += extent_minus_one * strides[dimension];
    }

    if (max_element_offset == std::numeric_limits<uint64_t>::max()) {
        throw std::invalid_argument("CUB tensor reduction tensor view address span overflows uint64_t.");
    }
    const uint64_t span_elements = max_element_offset + 1;
    const uint64_t element_bytes = static_cast<uint64_t>(TensorDescriptor::getElementSizeInBytes(tensor.getDataType()));
    if (span_elements > std::numeric_limits<uint64_t>::max() / element_bytes) {
        throw std::invalid_argument("CUB tensor reduction tensor view byte span overflows uint64_t.");
    }
    const uint64_t span_bytes = span_elements * element_bytes;

    const uintptr_t begin = reinterpret_cast<uintptr_t>(tensor.template getMemPtr<void>());
    if (span_bytes > static_cast<uint64_t>(std::numeric_limits<uintptr_t>::max() - begin)) {
        throw std::invalid_argument("CUB tensor reduction tensor view address span exceeds the process address space.");
    }
    return TensorReachableAddressSpan{begin, begin + static_cast<uintptr_t>(span_bytes)};
}

template <typename Lhs, typename Rhs>
[[nodiscard]] bool tensorStorageOverlaps(const Lhs& lhs, const Rhs& rhs) {
    // Tensor::getArraySizeInBytes() is the logical payload size. For alias views that may be much larger than the
    // physical address span (broadcast/zero-stride) or smaller than it (gapped/strided). Safety checks therefore use
    // the conservative interval containing every address the view can actually reach. Strides are non-negative in
    // Tensor, so the interval begins at getMemPtr() and ends at the maximum strided coordinate plus one element.
    const TensorReachableAddressSpan lhs_span = tensorReachableAddressSpan(lhs);
    const TensorReachableAddressSpan rhs_span = tensorReachableAddressSpan(rhs);
    return lhs_span.begin < rhs_span.end && rhs_span.begin < lhs_span.end;
}

void requireDenseContiguousReductionOutput(const Tensor& output, const char* role) {
    if (!output.isInitialized()) {
        throw std::invalid_argument(std::string(role) + " must be initialized.");
    }
    if (output.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU) {
        throw std::invalid_argument(std::string(role) + " must be a GPU tensor.");
    }
    // Reduction outputs only require a dense contiguous address range beginning at getMemPtr().
    if (!output.isDenseContiguous()) {
        throw std::invalid_argument(std::string(role) + " must be dense contiguous.");
    }
}

void requireExpectedOutput(const StridedSpan& input,
                           const Tensor& output,
                           DataType output_dtype,
                           const CubReductionGeometry& geometry) {
    requireDenseContiguousReductionOutput(output, "output");
    if (input.getPlacement() != output.getPlacement()) {
        throw std::invalid_argument("input and output must live on the same GPU placement.");
    }
    if (output.getDataType() != output_dtype) {
        throw std::invalid_argument("CUB tensor reduction preallocated output dtype does not match the configured output dtype.");
    }
    const std::vector<uint64_t>& output_dimensions = output.getDimensions();
    if (stripSingletonDimensions(output_dimensions) != stripSingletonDimensions(geometry.squeezed_output_dimensions)) {
        throw std::invalid_argument(
            "CUB tensor reduction preallocated output dimensions must preserve the reduction output's non-singleton layout.");
    }
    if (output.getTotalNumElements() != geometry.output_elements) {
        throw std::invalid_argument("CUB tensor reduction preallocated output element count does not match the reduction geometry.");
    }

    if (tensorStorageOverlaps(input, output)) {
        throw std::invalid_argument("CUB tensor reduction input and output storage must not overlap.");
    }
}

void requireExpectedArgOutput(const StridedSpan& input,
                              const Tensor& output,
                              DataType output_dtype,
                              const CubReductionGeometry& geometry,
                              const char* role) {
    requireDenseContiguousReductionOutput(output, role);
    if (input.getPlacement() != output.getPlacement()) {
        throw std::invalid_argument(std::string("input and ") + role + " must live on the same GPU placement.");
    }
    if (output.getDataType() != output_dtype) {
        throw std::invalid_argument(std::string("CUB arg reduction preallocated ") + role
                                    + " dtype does not match the configured dtype.");
    }
    if (stripSingletonDimensions(output.getDimensions())
        != stripSingletonDimensions(geometry.squeezed_output_dimensions)) {
        throw std::invalid_argument(std::string("CUB arg reduction preallocated ") + role
                                    + " dimensions must preserve the reduction output's non-singleton layout.");
    }
    if (output.getTotalNumElements() != geometry.output_elements) {
        throw std::invalid_argument(std::string("CUB arg reduction preallocated ") + role
                                    + " element count does not match the reduction geometry.");
    }
    if (tensorStorageOverlaps(input, output)) {
        throw std::invalid_argument(std::string("CUB arg reduction input and ") + role + " storage must not overlap.");
    }
}

void requireArgOutputsDoNotOverlap(const std::optional<Tensor>& value_output,
                                   const std::optional<Tensor>& index_output) {
    if (value_output.has_value() && index_output.has_value()
        && tensorStorageOverlaps(value_output.value(), index_output.value())) {
        throw std::invalid_argument("CUB arg reduction value and index output storage must not overlap.");
    }
}

void requireSupportedSegmentedOperation(CubReductionOp op) {
    switch (op) {
        case CubReductionOp::Sum:
        case CubReductionOp::Mean:
        case CubReductionOp::Min:
        case CubReductionOp::Max:
            return;
        case CubReductionOp::Product:
        case CubReductionOp::L1Norm:
        case CubReductionOp::L2Norm:
        case CubReductionOp::SumSquares:
            break;
    }
    throw std::invalid_argument("Offset-segmented CUB reduction supports sum, mean, min, and max.");
}

void requireSegmentOffsets(const Tensor& input, const Tensor& segment_offsets, const Stream& stream) {
    requireDenseContiguousGpuTensor(segment_offsets, "segment offsets");
    requireSameGpuPlacement(input, segment_offsets, "input", "segment offsets");
    requireCompatibleStream(input, stream);
    if (!CubSegmentedReduction::isOffsetDataTypeSupported(segment_offsets.getDataType())) {
        throw std::invalid_argument("CUB segmented-reduction offsets must use UINT32 or an enabled UINT64 dtype.");
    }
    const std::vector<uint64_t>& dimensions = segment_offsets.getDimensions();
    if (dimensions.size() != 1 || dimensions[0] < 2) {
        throw std::invalid_argument(
            "CUB segmented-reduction offsets must be rank 1 with shape [num_segments + 1] and at least one segment.");
    }
    if (tensorStorageOverlaps(input, segment_offsets)) {
        throw std::invalid_argument("CUB segmented-reduction input and offsets storage must not overlap.");
    }
}

void validateSegmentOffsetContentsAtStamp(const Tensor& input,
                                          const Tensor& segment_offsets,
                                          const Stream& stream) {
    Tensor cpu_offsets(TensorPlacement(TensorPlacement::MemDevices::CPU), segment_offsets.getDescriptor());
    cpu_offsets.copyFromAsync(segment_offsets, stream);
    stream.synchronize();

    const uint64_t offset_count = segment_offsets.getTotalNumElements();
    auto validate = [&](auto* offsets) {
        if (static_cast<uint64_t>(offsets[0]) != 0) {
            throw std::invalid_argument("CUB segmented-reduction offsets must begin at zero.");
        }
        uint64_t previous = 0;
        for (uint64_t i = 1; i < offset_count; ++i) {
            const uint64_t current = static_cast<uint64_t>(offsets[i]);
            if (current < previous) {
                throw std::invalid_argument("CUB segmented-reduction offsets must be nondecreasing.");
            }
            if (current > input.getDimensions()[0]) {
                throw std::invalid_argument("CUB segmented-reduction offset exceeds the values tensor row capacity.");
            }
            previous = current;
        }
    };

    switch (segment_offsets.getDataType()) {
        case DataType::UINT32:
            validate(cpu_offsets.getMemPtr<uint32_t>());
            return;
        case DataType::UINT64:
            validate(cpu_offsets.getMemPtr<uint64_t>());
            return;
        default:
            throw std::invalid_argument("CUB segmented-reduction offsets must use UINT32 or UINT64.");
    }
}

void requireExpectedSegmentedOutput(const Tensor& input,
                                    const Tensor& output,
                                    const Tensor& segment_offsets,
                                    DataType output_dtype,
                                    uint64_t num_segments) {
    requireDenseContiguousGpuTensor(output, "output");
    requireSameGpuPlacement(input, output, "input", "output");
    if (output.getDataType() != output_dtype) {
        throw std::invalid_argument(
            "CUB segmented-reduction preallocated output dtype does not match the configured output dtype.");
    }

    if (input.getDimensions().size() == 1) {
        if (output.getTotalNumElements() != num_segments
            || stripSingletonDimensions(output.getDimensions())
                   != stripSingletonDimensions(std::vector<uint64_t>{num_segments})) {
            throw std::invalid_argument(
                "CUB segmented-reduction scalar output must preserve the [num_segments] non-singleton layout.");
        }
    } else {
        std::vector<uint64_t> expected_dimensions = input.getDimensions();
        expected_dimensions[0] = num_segments;
        if (output.getDimensions() != expected_dimensions) {
            throw std::invalid_argument("CUB segmented-reduction vector output must have shape [num_segments,D...].");
        }
    }

    if (tensorStorageOverlaps(input, output) || tensorStorageOverlaps(segment_offsets, output)) {
        throw std::invalid_argument(
            "CUB segmented-reduction output storage must not overlap input or offsets storage.");
    }
}

void requireExpectedSegmentedArgOutput(const Tensor& input,
                                       const Tensor& output,
                                       const Tensor& segment_offsets,
                                       DataType output_dtype,
                                       uint64_t num_segments) {
    requireDenseContiguousGpuTensor(output, "index output");
    requireSameGpuPlacement(input, output, "input", "index output");
    if (output.getDataType() != output_dtype) {
        throw std::invalid_argument("CUB segmented arg-reduction output dtype does not match the configured index dtype.");
    }
    std::vector<uint64_t> expected_dimensions = input.getDimensions();
    expected_dimensions[0] = num_segments;
    if (output.getDimensions() != expected_dimensions) {
        throw std::invalid_argument(
            "CUB segmented arg-reduction output must have shape [num_segments,D...].");
    }
    if (tensorStorageOverlaps(input, output) || tensorStorageOverlaps(segment_offsets, output)) {
        throw std::invalid_argument("CUB segmented arg-reduction output storage must not overlap input or offsets storage.");
    }
}

void requireSupportedArgOperation(CubArgReductionOp op) {
    switch (op) {
        case CubArgReductionOp::ArgMin:
        case CubArgReductionOp::ArgMax:
            return;
    }
    throw std::invalid_argument("Unsupported CUB arg reduction operation.");
}

void requireSupportedOperation(CubReductionOp op) {
    switch (op) {
        case CubReductionOp::Sum:
        case CubReductionOp::Min:
        case CubReductionOp::Max:
        case CubReductionOp::Product:
        case CubReductionOp::Mean:
        case CubReductionOp::L1Norm:
        case CubReductionOp::L2Norm:
        case CubReductionOp::SumSquares:
            return;
    }
    throw std::invalid_argument("Unsupported CUB tensor reduction operation.");
}

[[nodiscard]] size_t checkedWorkspaceAdd(size_t lhs, size_t rhs, const char* message) {
    if (lhs > std::numeric_limits<size_t>::max() - rhs) throw std::overflow_error(message);
    return lhs + rhs;
}

[[nodiscard]] size_t queryDenseValueExecutableWorkspace(CubReductionOp op,
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
        const CubReductionGeometry& stage_geometry = layout.geometry;
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

        const CubReductionStageSemantics stage_semantics =
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

[[nodiscard]] DenseExecutablePlan materializeDenseValueExecutablePlanImpl(CubReductionOp op,
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
        CubReductionGeometry stage_geometry = layout.geometry;
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

        const CubReductionStageSemantics stage_semantics =
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

// ARG-P1: dense ARG is planned by planDenseReduction(), like VALUE. Since 1261 every supported input dtype (FP8,
// FP16, BF16, FP32) runs the planned retained pass list; there is no other dense ARG executor.
[[nodiscard]] bool isFp8ArgInput(DataType input_dtype) {
    return input_dtype == DataType::FP8_E4M3 || input_dtype == DataType::FP8_E5M2;
}

// 1221 / 1227: and FP32 input (its retained FP32 ARG passes run in this executor). 1254 / 1257: and FP16 / BF16 input
// (its 2-byte ARG passes, then the retained FP32 carried ARG passes).
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
    CubReductionGeometry geometry;
    uint64_t domain_stride = 0;
    std::vector<uint64_t> intermediate_dimensions;
    bool final_pass = false;
    // ARG-A2: a retained FP8 ARG physical pass (no scratch); null for a direct ARG pass.
    std::shared_ptr<const DenseRetainedFp8ArgPhysicalPlan> fp8_arg;
    // 1221: a retained FP32 ARG pass, original or carried; its scratch is plan.scratch_bytes.
    std::shared_ptr<const DenseRetainedFp32ArgPhysicalPlan> fp32_arg;
    [[nodiscard]] uint64_t retainedScratchBytes() const {
        return fp8_arg ? retainedFp8ArgScratchBytes(*fp8_arg) : fp32_arg ? fp32_arg->scratch_bytes : 0;
    }
};

struct PlannedDenseArg {
    std::vector<PlannedDenseArgPass> passes;
    DataType carried_index_dtype = DataType::UINT32;
};

[[nodiscard]] PlannedDenseArg planProductionDenseArg(CubArgReductionOp op,
                                                     const std::vector<uint64_t>& input_dimensions,
                                                     const std::vector<uint32_t>& axes,
                                                     DataType input_dtype,
                                                     std::optional<DataType> value_output_dtype,
                                                     std::optional<DataType> index_output_dtype,
                                                     uint64_t original_reduction_size,
                                                     CubReductionTuningDepth tuning_depth,
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
    // ARG-A2: the context names the VALUE operation whose retained FP8 selection ARG mirrors (MIN for argmin, MAX for
    // argmax) and the FP8 VALUE workspace budget used by that selection.
    constexpr uint64_t kRetainedFp8WorkspaceBudget = uint64_t{256} * 1024 * 1024;
    const DenseProductionPhysicalSelectionContext production_context(
        input_dtype,
        op == CubArgReductionOp::ArgMin ? DenseRKValueOperation::Minimum : DenseRKValueOperation::Maximum,
        stream,
        original_reduction_size,
        1.0f,
        op == CubArgReductionOp::ArgMin ? CubReductionOp::Min : CubReductionOp::Max,
        kRetainedFp8WorkspaceBudget,
        tuning_depth,
        input_alignment_bytes);
    const DenseReductionPlan plan = planDenseReduction(problem, production_context.context());

    for (const DenseReductionPlanStep& step : plan.steps) {
        const DenseReductionCandidate& candidate = step.selected_candidate;
        const auto fp8_arg =
            std::dynamic_pointer_cast<const DenseRetainedFp8ArgPhysicalPlan>(candidate.physical_plan);
        const auto fp32_arg =
            std::dynamic_pointer_cast<const DenseRetainedFp32ArgPhysicalPlan>(candidate.physical_plan);
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

namespace CubReductionInternal {
namespace {
// Capability gate shared by stamp-time scratch query and a direct physical
// launch. It does not inspect selectors, allocate resources or query CUDA.
void requireSupportedRetainedFp8PhysicalPass(
    const DenseRetainedFp8RKPhysicalPlan& fp8,
    const CubReductionStageSemantics& semantics,
    DataType input_dtype, uint64_t input_elements, DataType output_dtype,
    const CubReductionGeometry& geometry) {
    // All public VALUE combines, first-stage transforms and finalizers are
    // implemented by the retained typed kernels. Complete kernels convert to
    // the requested floating storage dtype; staged kernels emit FP32 only.
    // Keep the physical boundary strict without resurrecting the old
    // former SUM/FP32-only cutover gate or any legacy fallback.
    if (!isSupportedFloatingStorageDType(output_dtype))
        throw std::logic_error("Retained FP8 RK physical pass requires supported floating output storage");
    if (fp8.kernel.staged() &&
        (output_dtype != DataType::FP32 ||
         semantics.finalize != CubReductionStageFinalize::Identity))
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
    if (const auto* kr = dynamic_cast<const DenseRetainedFp8KRPhysicalPlan*>(&physical_plan)) {
        // 1146: one Complete pass over the site's current runs; staged chunk partials are the pass's scratch.
        layout.input_dimensions = kr->input_dimensions;
        layout.geometry = CubReduction::analyzeGeometry(layout.input_dimensions,
                                                        std::vector<uint32_t>{kr->reduction_axis});
    } else if (const auto* arg = dynamic_cast<const DenseRetainedFp8ArgPhysicalPlan*>(&physical_plan)) {
        // OneWindow ARG twins are Complete ([O,K]); DeepRSmallK ARG twins and multi-shard RuntimeK/AlignedK twins write
        // [O,shards,K] partials. 1146: an R / KR pass reduces its last current run.
        layout.input_dimensions = arg->input_dimensions;
        const uint32_t axis = arg->kr.group_lanes != 0 ? static_cast<uint32_t>(arg->input_dimensions.size() - 1) : 1U;
        layout.geometry = CubReduction::analyzeGeometry(layout.input_dimensions, std::vector<uint32_t>{axis});
        if (arg->shards > 1 && !arg->packed_partials)
            layout.intermediate_dimensions = {arg->outer_size, arg->shards, arg->inner_size};
    } else if (const auto* fp32_arg = dynamic_cast<const DenseRetainedFp32ArgPhysicalPlan*>(&physical_plan)) {
        // 1221: a Complete FP32 ARG pass (RK over {outer, r, inner} at axis 1; R / KR over the current runs, the
        // nominated run last).
        layout.input_dimensions = fp32_arg->input_dimensions;
        layout.geometry = CubReduction::analyzeGeometry(layout.input_dimensions,
                                                        std::vector<uint32_t>{fp32_arg->reduction_axis});
    } else if (const auto* fp8 = dynamic_cast<const DenseRetainedFp8RKPhysicalPlan*>(&physical_plan)) {
        // The selected first kernel has its own real [O,N,K] output. A staged
        // output is an ordinary FP32 intermediate, never hidden scratch.
        layout.input_dimensions = fp8->input_dimensions;
        layout.geometry = CubReduction::analyzeGeometry(
            layout.input_dimensions, std::vector<uint32_t>{fp8->reduction_axis});
        layout.intermediate_dimensions = fp8->output_dimensions;
    } else {
        throw std::logic_error("Dense VALUE physical plan has no geometry adapter.");
    }
    if (layout.intermediate_dimensions.empty()) layout.intermediate_dimensions = layout.geometry.output_dimensions;
    return layout;
}

size_t queryDensePhysicalPassScratchBytes(const DenseReductionPhysicalPlan& physical_plan,
                                         const CubReductionStageSemantics& semantics,
                                         DataType input_dtype,
                                         const void* input,
                                         uint64_t input_elements,
                                         DataType output_dtype,
                                         void* output,
                                         const CubReductionGeometry& geometry,
                                         float output_scale,
                                         const Stream& stream) {
    if (const auto* kr = dynamic_cast<const DenseRetainedFp8KRPhysicalPlan*>(&physical_plan)) {
        // 1220: an FP32 pass's shape is in bytes, 4 per element. 1254: 2 per FP16 / BF16 element.
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
        return static_cast<size_t>(Fp8KR::fp8KRScratchBytes(kr->shape, false));
    }
    if (const auto* fp8 = dynamic_cast<const DenseRetainedFp8RKPhysicalPlan*>(&physical_plan)) {
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
                             const CubReductionStageSemantics& semantics,
                             const StridedSpan& temp_storage,
                             size_t temp_storage_bytes,
                             const StridedSpan& input,
                             const StridedSpan& output,
                             const CubReductionGeometry& geometry,
                             float output_scale,
                             Stream& stream) {
    if (const auto* kr = dynamic_cast<const DenseRetainedFp8KRPhysicalPlan*>(&physical_plan)) {
        // 1146: one retained R / KR kernel, plus the finishing kernel of a staged shape (partials in this scratch).
        if (input.getDataType() != kr->input_dtype || geometry.output_elements != kr->shape.rows)
            throw std::invalid_argument("Retained FP8 KR physical launch binding mismatch");
        void* scratch = temp_storage_bytes == 0
                            ? nullptr
                            : const_cast<void*>(static_cast<const void*>(temp_storage.getMemPtr<void>()));
        if (kr->input_dtype == DataType::FP32) {  // 1220: the retained FP32 R / KR pass
            launchFp32KRValueSegments(semantics, input, output, kr->shape, scratch, temp_storage_bytes, output_scale,
                                      stream);
        } else if (kr->input_dtype == DataType::FP16 || kr->input_dtype == DataType::BF16) {  // 1254: the 2-byte pass
            launchF16KRValueSegments(semantics, input, output, kr->shape, scratch, temp_storage_bytes, output_scale,
                                     stream);
        } else {
            launchFp8KRValueSegments(semantics, input, output, kr->shape, scratch, temp_storage_bytes, output_scale,
                                     stream);
        }
        return;
    }
    if (const auto* fp8 = dynamic_cast<const DenseRetainedFp8RKPhysicalPlan*>(&physical_plan)) {
        // This dispatch launches precisely ONE retained physical CUDA kernel.
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
        Fp8RK::launchPhysicalFirstStage(fp8->kernel, semantics, input, output,
                                        geometry, output_scale, stream);
        return;
    }
    throw std::logic_error("Dense executable VALUE pass has no registered physical executor.");
}

DenseExecutablePlan materializeDenseValueExecutablePlan(
    CubReductionOp op, const StridedSpan& input, const Tensor& output,
    const DenseReductionPlan& plan, uint64_t semantic_total_reduction_size,
    float output_scale, const Stream& stream) {
    return materializeDenseValueExecutablePlanImpl(op, input, output, plan,
                                                   semantic_total_reduction_size,
                                                   output_scale, stream);
}

}  // namespace CubReductionInternal

CubReduction::CubReduction(CubReductionOp op,
                           uint32_t axis,
                           std::optional<DataType> output_dtype,
                           float output_scale,
                           std::optional<CubReductionTuningDepth> tuning_depth)
    : CubReduction(op, std::vector<uint32_t>{axis}, output_dtype, output_scale, tuning_depth) {}

CubReduction::CubReduction(CubReductionOp op,
                           std::vector<uint32_t> axes,
                           std::optional<DataType> output_dtype,
                           float output_scale,
                           std::optional<CubReductionTuningDepth> tuning_depth)
    : op(op), axes(std::move(axes)), output_dtype(output_dtype), output_scale(output_scale),
      tuning_depth(resolveCubReductionTuningDepth(tuning_depth)) {
    requireSupportedOperation(op);
    if (this->axes.empty()) {
        throw std::invalid_argument("CUB tensor reduction requires at least one reduction axis.");
    }
    for (size_t i = 1; i < this->axes.size(); ++i) {
        if (this->axes[i] <= this->axes[i - 1]) {
            throw std::invalid_argument("CUB tensor reduction axes must be unique and strictly increasing.");
        }
    }
    if (output_dtype.has_value()) {
        requireSupportedFloatingStorageDType(output_dtype.value(), "output");
    }
    if (!std::isfinite(output_scale)) {
        throw std::invalid_argument("CUB tensor reduction output scale must be finite.");
    }
}

DataType CubReduction::resolveOutputDataType(DataType input_dtype) const {
    requireSupportedFloatingStorageDType(input_dtype, "input");
    const DataType resolved = output_dtype.value_or(input_dtype);
    requireSupportedFloatingStorageDType(resolved, "output");
    return resolved;
}

size_t CubReduction::queryWorkspaceSizeInBytes(const TensorDescriptor& input_descriptor,
                                               const Stream& stream) const {
    requireSupportedFloatingStorageDType(input_descriptor.getDataType(), "input");
    const CubReductionGeometry geometry = analyzeValueGeometry(op, input_descriptor.getDimensions(), axes);
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

    // 1393: a dense descriptor's geometry is always an ordinary dense one (views carry the tiled metadata).
    throw std::logic_error("A dense value reduction descriptor did not classify as an ordinary dense geometry.");
}

float CubReduction::getFp32EmptyReductionValue(CubReductionOp op) {
    requireSupportedOperation(op);
    switch (op) {
        case CubReductionOp::Sum:
        case CubReductionOp::Mean:
        case CubReductionOp::L1Norm:
        case CubReductionOp::L2Norm:
        case CubReductionOp::SumSquares:
            return 0.0f;
        case CubReductionOp::Min:
            return std::numeric_limits<float>::infinity();
        case CubReductionOp::Max:
            return -std::numeric_limits<float>::infinity();
        case CubReductionOp::Product:
            return 1.0f;
    }
    throw std::invalid_argument("Unsupported CUB tensor reduction operation.");
}

CubReductionGeometry CubReduction::analyzeGeometry(const std::vector<uint64_t>& input_dimensions, uint32_t axis) {
    return analyzeGeometry(input_dimensions, std::vector<uint32_t>{axis});
}

CubReductionGeometry CubReduction::analyzeGeometry(const std::vector<uint64_t>& input_dimensions,
                                                    const std::vector<uint32_t>& axes) {
    std::vector<uint64_t> dense_strides(input_dimensions.size(), 1);
    uint64_t running_stride = 1;
    for (size_t dimension = input_dimensions.size(); dimension-- > 0;) {
        dense_strides[dimension] = running_stride;
        if (input_dimensions[dimension] != 0
            && running_stride > std::numeric_limits<uint64_t>::max() / input_dimensions[dimension]) {
            throw std::invalid_argument("CUB tensor reduction input element count overflows uint64_t.");
        }
        running_stride *= input_dimensions[dimension];
    }
    return analyzeGeometry(input_dimensions, dense_strides, axes);
}

CubReductionGeometry CubReduction::analyzeGeometry(const std::vector<uint64_t>& input_dimensions,
                                                    const std::vector<uint64_t>& input_strides,
                                                    const std::vector<uint32_t>& axes) {
    CubReductionGeometry geometry = classifyGeometry(input_dimensions, input_strides, axes);
    // 1390: a rotated (F / G) or pitched (H) view is the caller's to run in two parts (splitRefusedView).
    if (refusedViewSplit(input_dimensions, input_strides == StridedSpan::denseStrides(input_dimensions), geometry)
            .has_value()) {
        throw NotImplementedException(
            "CUB tensor reduction refuses a view whose retained axes are permuted or whose rows have gaps: run it in "
            "two parts with the dense copy (CubReduction::splitRefusedView).");
    }
    return geometry;
}

std::optional<CubReductionViewSplit> CubReduction::splitRefusedView(const std::vector<uint64_t>& input_dimensions,
                                                                    const std::vector<uint64_t>& input_strides,
                                                                    const std::vector<uint32_t>& axes) {
    try {
        const CubReductionGeometry geometry = classifyGeometry(input_dimensions, input_strides, axes);
        return refusedViewSplit(
            input_dimensions, input_strides == StridedSpan::denseStrides(input_dimensions), geometry);
    } catch (const NotImplementedException&) {
        return std::nullopt;  // no reducer owns the layout, in one part or two
    }
}

std::optional<CubReductionViewSplit> CubReduction::splitRefusedView(const StridedSpan& input,
                                                                    const std::vector<uint32_t>& axes,
                                                                    DataType output_dtype) {
    return splitRefusedView(input, axes, output_dtype, true);
}

std::optional<CubReductionViewSplit> CubReduction::splitRefusedView(const StridedSpan& input,
                                                                    const std::vector<uint32_t>& axes,
                                                                    DataType output_dtype,
                                                                    bool reducer_reads_pitched) {
    std::optional<CubReductionViewSplit> split =
        splitRefusedView(input.getDimensions(), input.getStridesElements(), axes);
    if (!split.has_value() || split->kind != CubReductionViewSplit::Kind::PackedInput) {
        return split;
    }
    if (reducer_reads_pitched && pitchedCompleteLaunch(input, axes, output_dtype).has_value()) {
        return std::nullopt;  // 1392: the reducer reads the rows at their pitches
    }
    const CubReductionGeometry geometry = classifyGeometry(input.getDimensions(), input.getStridesElements(), axes);
    const auto elementBytes = [](DataType dtype) {
        return std::max<uint64_t>(1, static_cast<uint64_t>(TensorDescriptor::getElementSizeInBytes(dtype)));
    };
    if (std::optional<CubReductionViewSplit> sliced = slicedRowSplit(
            geometry, input.sourceElementsFromFirst(), elementBytes(input.getDataType()), elementBytes(output_dtype));
        sliced.has_value()) {
        return sliced;
    }
    return split;
}

namespace {

// 1392 (phase 7 C): the PacketComplete tile cap of an input dtype (one 16-byte packet per lane, 31 of the 32 lanes'
// components), or 0 when PacketComplete does not read it.
[[nodiscard]] uint32_t pitchedCompleteMaxTile(DataType dtype) {
    switch (dtype) {
        case DataType::FP32:
            return 124;
        case DataType::FP16:
        case DataType::BF16:
            return 248;
#if THOR_CUB_ENABLE_FP8_TYPES
        case DataType::FP8_E4M3:
        case DataType::FP8_E5M2:
            return 496;
#endif
        default:
            return 0;
    }
}

// 1392: the view's rows as the 3-D span [O, R, I] at strides [So, Sr, 1] from its first element (the view's own
// elements; the pitched classification orders its logical axes physically, so element [0, ..., 0] is the first).
[[nodiscard]] StridedSpan pitchedRowsSpan(const StridedSpan& view, const CubReductionPitchedTiledGeometry& pitched) {
    return view.withSourceLayout({pitched.outer_size, pitched.reduction_size, pitched.inner_size},
                                 {pitched.outer_stride, pitched.reduction_stride, 1});
}

// 1392: one in-place PacketComplete launch of a pitched view's rows (`rows`, from pitchedRowsSpan) into the dense
// keepdims output.
void launchPitchedComplete(const CubReductionPitchedCompleteLaunch& launch,
                           const CubReductionInternal::CubReductionStageSemantics& semantics,
                           const StridedSpan& rows,
                           const StridedSpan& output,
                           const CubReductionGeometry& geometry,
                           float output_scale,
                           Stream& stream) {
    const CubReductionPitchedTiledGeometry& pitched = *geometry.pitched_tiled_geometry;
    CubReductionGeometry launch_geometry = geometry;
    launch_geometry.outer_size = pitched.outer_size;
    launch_geometry.reduction_size = pitched.reduction_size;
    launch_geometry.inner_size = pitched.inner_size;
    launch_geometry.input_elements = pitched.outer_size * pitched.reduction_size * pitched.inner_size;
    launch_geometry.output_dimensions = output.getDimensions();
    switch (rows.getDataType()) {
        case DataType::FP32:
            CubReductionInternal::launchFp32PacketComplete(rows, output, launch_geometry, launch.warps_per_output,
                                                           launch.tile_components, stream, semantics, output_scale);
            return;
        case DataType::FP16:
        case DataType::BF16:
            CubReductionInternal::launchF16PacketComplete(rows, output, launch_geometry, launch.warps_per_output,
                                                          launch.tile_components, stream, semantics, output_scale);
            return;
        default:
            CubReductionInternal::launchFp8GeneralRParallelKTileRCooperativeCalibrationComplete(
                rows, output, launch_geometry, launch.warps_per_output, launch.tile_components, stream, semantics,
                output_scale);
            return;
    }
}

}  // namespace

std::optional<CubReductionPitchedCompleteLaunch> CubReduction::pitchedCompleteLaunch(
    const StridedSpan& input,
    const std::vector<uint32_t>& axes,
    DataType output_dtype,
    std::optional<uint32_t> warps_per_output) {
    if (warps_per_output.has_value() && *warps_per_output != 1 && *warps_per_output != 2 && *warps_per_output != 4 &&
        *warps_per_output != 8) {
        throw std::invalid_argument("A pitched view's PacketComplete takes 1, 2, 4 or 8 warps per output.");
    }
    if (!input.isInitialized()) return std::nullopt;
    const std::optional<CubReductionViewSplit> split =
        splitRefusedView(input.getDimensions(), input.getStridesElements(), axes);
    if (!split.has_value() || split->kind != CubReductionViewSplit::Kind::PackedInput) return std::nullopt;
    const CubReductionGeometry geometry = classifyGeometry(input.getDimensions(), input.getStridesElements(), axes);
    if (!geometry.pitched_tiled_geometry.has_value()) return std::nullopt;
    const CubReductionPitchedTiledGeometry& pitched = *geometry.pitched_tiled_geometry;
    const uint64_t O = pitched.outer_size, R = pitched.reduction_size, I = pitched.inner_size;
    const uint32_t max_tile = pitchedCompleteMaxTile(input.getDataType());
    const uint64_t element_bytes = std::max<uint64_t>(1, TensorDescriptor::getElementSizeInBytes(input.getDataType()));
    if (max_tile == 0 || O == 0 || R == 0 || I == 0 || O > Fp8RK::FP8_RK_MAX_UINT32 || I > Fp8RK::FP8_RK_MAX_UINT32 ||
        R > Fp8RK::FP8_RK_PACKET_COMPLETE_MAX_ROWS ||
        reinterpret_cast<uintptr_t>(input.getMemPtr<void>()) % element_bytes != 0) {
        return std::nullopt;
    }
    // Balanced K tiles: as few as the cap allows, of equal width but the last.
    const uint64_t tiles = (I + max_tile - 1) / max_tile;
    const uint32_t tile = static_cast<uint32_t>((I + tiles - 1) / tiles);
    if (warps_per_output.has_value()) return CubReductionPitchedCompleteLaunch{*warps_per_output, tile};

    // A warp holds one row's tile in flight (the next row's packets, P5-H load-ahead), so the read rate is the bytes in
    // flight: narrow tiles starve it (1392's audit: no tile under 256 bytes was faster in place at any W).
    constexpr uint64_t MinTileBytes = 256;
    const uint64_t tile_bytes = tile * element_bytes;
    if (tile_bytes < MinTileBytes) return std::nullopt;
    const uint64_t sm_count = std::max<uint32_t>(
        1, CubReductionInternal::denseReductionDeviceIdentity(input.getPlacement().getDeviceNum()).sm_count);
    const uint64_t resident_warps = 32 * static_cast<uint64_t>(sm_count);  // four 8-warp CTAs per SM
    const uint64_t work = O * tiles;  // [outer, K tile] items, W warps each

    // 1392a: W is the most warps per item whose launch still fits one wave of resident warps (a second, part-filled
    // wave left a tail: 1392's cell 432 at W 4, 1.11 waves, 64 us against 58 at W 2), and at least four rows per warp
    // (cell 460, R = 7: W 1 9.9 us, W 4 13.7 at FP8).
    uint32_t warps = 1;
    while (warps < 8 && work * warps * 2 <= resident_warps) warps *= 2;
    while (warps > 1 && static_cast<uint64_t>(warps) * 4 > R) warps /= 2;

    // 1392a: in place when the launch keeps enough bytes in flight to beat the split it replaces. A sliced split reads
    // about the view's bytes once (near the dense rate), so the in-place read must run near the copy rate: 4 KiB per SM
    // (0.70 MB on 170 SMs; 1392: 462 at 0.64 MB lost 17% to sliced, 460 at 0.81 MB tied it). A packed split moves the
    // bytes three times: 2 KiB per SM (478 at 0.76 MB was 1.6-2.5x faster than packed; 461 at 0.32-0.36 MB tied it).
    // The sliced test is the layout's and the dtypes' only (slicedRowSplit with the source's reach left out), so the
    // decision follows dims, strides, dtypes and device, as Expression caches it; a view whose source ends at its last
    // row packs, and is held to the sliced bar.
    const uint64_t in_flight = std::min(work * warps, resident_warps) * tile_bytes;
    const bool sliced = slicedRowSplit(geometry,
                                       std::numeric_limits<uint64_t>::max(),
                                       element_bytes,
                                       TensorDescriptor::getElementSizeInBytes(output_dtype))
                            .has_value();
    if (in_flight < (sliced ? 4096 : 2048) * static_cast<uint64_t>(sm_count)) return std::nullopt;
    return CubReductionPitchedCompleteLaunch{warps, tile};
}

bool CubReduction::readsSpanInPlace(const StridedSpan& input,
                                    const std::vector<uint32_t>& axes,
                                    DataType output_dtype) {
    try {
        (void)analyzeGeometry(input.getDimensions(), input.getStridesElements(), axes);
        return true;
    } catch (const NotImplementedException&) {
        return pitchedCompleteLaunch(input, axes, output_dtype).has_value();
    }
}

CubReductionGeometry CubReduction::analyzeValueSpanGeometry(CubReductionOp op,
                                                             const StridedSpan& input,
                                                             const std::vector<uint32_t>& axes,
                                                             DataType output_dtype) {
    requireSupportedOperation(op);
    if (pitchedCompleteLaunch(input, axes, output_dtype).has_value()) {
        return classifyGeometry(input.getDimensions(), input.getStridesElements(), axes);
    }
    return analyzeGeometry(input.getDimensions(), input.getStridesElements(), axes);
}

CubReductionGeometry CubReduction::classifyGeometry(const std::vector<uint64_t>& input_dimensions,
                                                    const std::vector<uint64_t>& input_strides,
                                                    const std::vector<uint32_t>& axes) {
    if (input_dimensions.empty()) {
        throw std::invalid_argument("CUB tensor reduction requires at least one input dimension.");
    }
    if (input_dimensions.size() != input_strides.size()) {
        throw std::invalid_argument("CUB tensor reduction input stride rank must match the input rank.");
    }
    if (input_dimensions.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::invalid_argument("CUB tensor reduction rank exceeds the uint32 axis representation limit.");
    }
    if (axes.empty()) {
        throw std::invalid_argument("CUB tensor reduction requires at least one reduction axis.");
    }
    if (axes.size() > input_dimensions.size()) {
        throw std::invalid_argument("CUB tensor reduction has more reduction axes than input dimensions.");
    }
    for (size_t i = 0; i < axes.size(); ++i) {
        if (axes[i] >= input_dimensions.size()) {
            throw std::invalid_argument("CUB tensor reduction axis is outside the input rank.");
        }
        if (i > 0 && axes[i] <= axes[i - 1]) {
            throw std::invalid_argument("CUB tensor reduction axes must be unique and strictly increasing.");
        }
    }

    auto checkedMultiply = [](uint64_t a, uint64_t b, const char* quantity) {
        if (b != 0 && a > std::numeric_limits<uint64_t>::max() / b) {
            throw std::invalid_argument(std::string("CUB tensor reduction ") + quantity + " overflows uint64_t.");
        }
        return a * b;
    };

    for (uint64_t dimension : input_dimensions) {
        if (dimension == 0) {
            throw std::invalid_argument("CUB tensor reduction does not support zero-sized tensor dimensions.");
        }
    }

    CubReductionGeometry geometry;
    geometry.axes = axes;
    geometry.rank = static_cast<uint32_t>(input_dimensions.size());
    geometry.input_elements = 1;
    geometry.reduction_size = 1;
    geometry.output_elements = 1;
    geometry.output_dimensions = input_dimensions;
    geometry.squeezed_output_dimensions.reserve(input_dimensions.size() - axes.size());

    std::vector<uint64_t> dense_strides(input_dimensions.size(), 1);
    uint64_t running_stride = 1;
    for (size_t dimension = input_dimensions.size(); dimension-- > 0;) {
        dense_strides[dimension] = running_stride;
        running_stride = checkedMultiply(running_stride, input_dimensions[dimension], "input element count");
    }
    geometry.input_elements = running_stride;
    const bool input_is_dense_contiguous = input_strides == dense_strides;
    if (input_is_dense_contiguous) {
        geometry.dense_run_geometry = analyzeDenseRuns(input_dimensions, dense_strides, axes);
    }

    size_t reduced_cursor = 0;
    for (uint32_t dimension = 0; dimension < input_dimensions.size(); ++dimension) {
        const bool reduced = reduced_cursor < axes.size() && axes[reduced_cursor] == dimension;
        if (reduced) {
            geometry.output_dimensions[dimension] = 1;
            geometry.reduction_size = checkedMultiply(
                geometry.reduction_size, input_dimensions[dimension], "reduction element count");
            ++reduced_cursor;
        } else {
            geometry.output_elements = checkedMultiply(
                geometry.output_elements, input_dimensions[dimension], "output element count");
            geometry.squeezed_output_dimensions.push_back(input_dimensions[dimension]);
        }
    }
    if (geometry.squeezed_output_dimensions.empty()) {
        geometry.squeezed_output_dimensions.push_back(1);
    }

    if (geometry.input_elements > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        throw std::invalid_argument("CUB tensor reduction input element count exceeds CUB's int64 item-count limit.");
    }
    if (geometry.output_elements > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        throw std::invalid_argument("CUB tensor reduction output element count exceeds CUB's int64 segment-count limit.");
    }

    geometry.reduced_axes_are_contiguous = true;
    for (size_t i = 1; i < axes.size(); ++i) {
        if (axes[i] != axes[i - 1] + 1) {
            geometry.reduced_axes_are_contiguous = false;
            break;
        }
    }

    if (geometry.reduced_axes_are_contiguous) {
        geometry.outer_size = 1;
        geometry.inner_size = 1;
        for (uint32_t dimension = 0; dimension < axes.front(); ++dimension) {
            geometry.outer_size = checkedMultiply(geometry.outer_size, input_dimensions[dimension], "outer size");
        }
        for (uint32_t dimension = axes.back() + 1; dimension < input_dimensions.size(); ++dimension) {
            geometry.inner_size = checkedMultiply(geometry.inner_size, input_dimensions[dimension], "inner size");
        }
        geometry.tiled_output_outer_stride = geometry.inner_size;
        geometry.tiled_output_inner_stride = 1;
    }

    static_cast<void>(analyzeDensePhysicalPermutation(input_dimensions, input_strides, geometry));
    if (!input_is_dense_contiguous) {
        static_cast<void>(analyzePermutationAwareContiguousCandidate(input_dimensions, axes, geometry));
        analyzePermutationAwareTiledCandidate(input_dimensions, axes, geometry);
        // VIEW-DIRECT-2B fills only the compact [A,reduction,B,payload] -> [B,A,payload] retained-order gap left by
        // the existing permutation-aware tiled family. It is intentionally considered only after those ordained
        // candidates reject the geometry.
        if (!geometry.permutation_aware_contiguous_segments
            && !geometry.permutation_aware_tiled_geometry.has_value()) {
            analyzePayloadTransposeTiledCandidate(input_dimensions, axes, geometry);
        }
        // Preserve ownership of every already-ordained compact-permutation path. VIEW-PITCHED-TILED is considered
        // only when no direct compact-permutation strategy applies. Keeping these candidates mutually exclusive also
        // makes launch dispatch unambiguous.
        if (!geometry.permutation_aware_contiguous_segments
            && !geometry.permutation_aware_tiled_geometry.has_value()
            && !geometry.payload_transpose_tiled_geometry.has_value()) {
            analyzePitchedTiledCandidate(input_dimensions, input_strides, axes, geometry);
        }
    }

    const bool reduces_to_single_output = geometry.output_elements == 1;
    const bool has_physically_contiguous_fixed_segments =
        geometry.reduced_axes_are_contiguous && geometry.inner_size == 1;

    if (input_is_dense_contiguous) {
        if (reduces_to_single_output) {
            geometry.path = CubReductionPath::DeviceTransformReduce;
        } else if (has_physically_contiguous_fixed_segments) {
            geometry.path = CubReductionPath::ContiguousFixedSegment;
        } else if (geometry.reduced_axes_are_contiguous) {
            geometry.path = CubReductionPath::TiledFixedSegment;
        } else {
            // ComposedDense is the structural classification for ordinary dense disjoint reduced runs. VALUE and
            // ARG both execute it through planDenseReduction(), which selects R / KR / RK passes and replans after
            // every pass (1261: the interval-composition planner and ARG executor are deleted).
            geometry.path = CubReductionPath::ComposedDense;
        }
        requireOrdainedDenseReductionPath(geometry);
    } else if (reduces_to_single_output && geometry.physical_layout_is_dense_permutation) {
        // VIEW-DIRECT-1: when every logical axis is reduced, a compact physical permutation is just one contiguous
        // physical value domain. Value reductions are insensitive to the logical visitation order, and a proven dense
        // physical permutation contains each logical value exactly once in input_elements contiguous elements from
        // input.getMemPtr(). 1365: CubReduction::stampValidated runs it as that dense [input_elements] on the retained
        // dense planner (viewPhysicalDenseProblem); the path is its classification.
        geometry.path = CubReductionPath::DeviceTransformReduce;
    } else if (geometry.permutation_aware_contiguous_segments) {
        // VIEW-DIRECT-2A: the non-singleton reduced axes are the complete physical suffix and the physical retained
        // prefix is already in Thor's logical dense output order. Each retained output is therefore one ordinary
        // contiguous segment in storage. 1365: CubReduction::stampValidated runs it as the dense [output_elements,
        // reduction_size] over axis 1 on the retained dense planner (viewPhysicalDenseProblem); the path is its
        // classification. Cases that need retained-output permutation deliberately remain for VIEW-DIRECT-2B.
        geometry.path = CubReductionPath::ContiguousFixedSegment;
    } else if (geometry.permutation_aware_tiled_geometry.has_value()) {
        // A logical permutation view over physically dense storage: the candidate supplies the physical
        // [outer,reduction,inner] traversal and whether the final dense retained output is natural [outer,inner] or
        // the [inner,outer] rotation. 1376: a natural output runs the retained dense planner on the bytes
        // (viewPhysicalDenseProblem); 1390: analyzeGeometry refuses the rotation, which the caller runs in two parts
        // (refusedViewSplit). The path is the classification.
        activatePermutationAwareTiledCandidate(geometry);
    } else if (geometry.payload_transpose_tiled_geometry.has_value()) {
        // VIEW-DIRECT-2B: the compact source is physically [A,reduction,B,payload] while Thor's dense retained output
        // is [B,A,payload]. 1390: analyzeGeometry refuses it; the caller reduces the bytes as [A,reduction,BxP] and its
        // dense copy writes [B,A,payload] (refusedViewSplit; the 32x33 payload-transpose kernel ran 5-2,447x slower in
        // 1386).
        activatePayloadTransposeTiledCandidate(geometry);
    } else if (reduces_to_single_output && input_dimensions.size() == 1) {
        // Rank-1 views are always affine, including diagonals (stride > 1) and broadcast aliases (stride == 0), and need
        // no general logical-coordinate metadata or indexing. 1366: CubReduction::stampValidated runs them on StridedR
        // (StridedReduction.h: the stride-0 closed form, else Split); the path is the classification.
        geometry.path = CubReductionPath::DeviceTransformReduce;
        geometry.device_transform_uses_affine_stride = true;
        geometry.affine_input_stride = input_strides[0];
    } else if (geometry.pitched_tiled_geometry.has_value()) {
        // VIEW-PITCHED-TILED: a non-dense logical [outer,reduction,inner] view with a contiguous inner payload can be
        // addressed with two affine pitches. 1390: analyzeGeometry refuses it; the caller reduces its dense copy
        // (refusedViewSplit; the pitched kernel ran up to 22,000x slower in 1386), until pitch-aware dense passes
        // (plans/Thor_Tiled_View_Replacement_Plan_2026-11-12.md, phase 7).
        activatePitchedTiledCandidate(geometry);
    } else if (densePermutationRuns(input_dimensions, geometry).has_value()) {
        // 1376 (phase 5): a split physical reduction, a compact permutation whose reduced axes are several physical
        // runs and whose retained axes keep their physical order. It is a dense tensor reduced over disjoint runs,
        // ComposedDense's classification; CubReduction::stampValidated runs it on the retained dense planner
        // (viewPhysicalDenseProblem).
        geometry.path = CubReductionPath::ComposedDense;
    } else {
        // DELETE: Thor no longer has an arbitrary per-scalar logical-index reduction backend. Every real workload must
        // be owned by an explicit physical-layout-aware reducer; exotic views fail rather than silently executing a
        // catastrophically slow catch-all implementation.
        throw NotImplementedException(
            "Unsupported CUB tensor-reduction geometry: no ordained production reducer owns this layout. "
            "Materialize/reorder the view or add an explicit high-efficiency reduction strategy for this geometry.");
    }

    // 1374: no path inherits CUB's `int` fixed-segment size; no reduction calls DeviceSegmentedReduce (1373), and the
    // retained, tiled and StridedR kernels carry 64-bit geometry.
    return geometry;
}

CubReductionGeometry CubReduction::analyzeValueGeometry(CubReductionOp op,
                                                          const std::vector<uint64_t>& input_dimensions,
                                                          const std::vector<uint32_t>& axes) {
    requireSupportedOperation(op);
    return analyzeGeometry(input_dimensions, axes);
}

CubReductionGeometry CubReduction::analyzeValueGeometry(CubReductionOp op,
                                                          const std::vector<uint64_t>& input_dimensions,
                                                          const std::vector<uint64_t>& input_strides,
                                                          const std::vector<uint32_t>& axes) {
    requireSupportedOperation(op);
    return analyzeGeometry(input_dimensions, input_strides, axes);
}

std::shared_ptr<StampedCubReduction> CubReduction::stamp(const StridedSpan& input, const Stream& stream) const {
    requireGpuTensorView(input, "input");
    requireCompatibleStream(input, stream);
    const CubReductionGeometry geometry =
        analyzeValueSpanGeometry(op, input, axes, resolveOutputDataType(input.getDataType()));
    const DataType resolved_output_dtype = resolveOutputDataType(input.getDataType());
    Tensor output(input.getPlacement(), TensorDescriptor(resolved_output_dtype, geometry.output_dimensions));
    return stampValidated(input, output, geometry, stream);
}

std::shared_ptr<StampedCubReduction> CubReduction::stampValidated(const StridedSpan& input,
                                                                  const Tensor& output,
                                                                  const CubReductionGeometry& geometry,
                                                                  const Stream& stream) const {
    requireSupportedFloatingStorageDType(input.getDataType(), "input");
    requireSupportedFloatingStorageDType(output.getDataType(), "output");
    requireExpectedOutput(input, output, output.getDataType(), geometry);

    // 1392 (phase 7 C): a pitched view stamp() accepted (analyzeValueSpanGeometry) is read in place.
    if (geometry.pitched_tiled_geometry.has_value()) {
        const std::optional<CubReductionPitchedCompleteLaunch> launch =
            pitchedCompleteLaunch(input, axes, output.getDataType());
        if (!launch.has_value()) {
            throw std::logic_error("A pitched view reached the reducer without a PacketComplete launch.");
        }
        return stampPitchedComplete(input, output, geometry, *launch, stream);
    }

    ScopedGpu scoped_gpu(stream.getGpuNum());

    // 1365: a VIEW-DIRECT-1 / -2A view runs as the dense tensor of its bytes; 1376: so does every compact permutation
    // whose retained axes keep their physical order (natural-output tiled views, split physical reductions). The stamp
    // keeps the view as its input and the view's geometry (getPath() stays its classification); the passes read the
    // dense span. 1373: such a view takes this branch whatever tiled candidates the classifier also noted (a
    // candidate is only an alternative layout reading).
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
        DenseExecutablePlan executable_plan = CubReductionInternal::materializeDenseValueExecutablePlan(op,
                                                                                  dense_input,
                                                                                  output,
                                                                                  plan,
                                                                                  geometry.reduction_size,
                                                                                  output_scale,
                                                                                  stream);
        auto stamped = std::shared_ptr<StampedCubReduction>(new StampedCubReduction(op,
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

    if (geometry.path == CubReductionPath::ComposedDense) {
        throw std::logic_error(
            "Ordinary dense ComposedDense VALUE geometry must execute through DenseExecutablePlan.");
    }

    // 1366: case C, a rank-1 strided view (the view plan, phase 2), runs StridedR (StridedReduction.h): the closed form at
    // stride 0, else Split (1371: no contest; the grid follows from n). The stamp keeps the view and its geometry; the
    // strategy chain is the launch's name.
    if (geometry.device_transform_uses_affine_stride) {
        const CubReductionInternal::StridedRLaunch launch =
            CubReductionInternal::stridedRLaunchFor(input, output.getDataType(), static_cast<int>(stream.getGpuNum()));
        Tensor workspace;
        if (launch.workspace_bytes != 0) {
            workspace = Tensor(input.getPlacement(), TensorDescriptor(DataType::UINT8, {launch.workspace_bytes}));
            // 1370: Split's ticket starts at 0 and every launch leaves it there.
            CubReductionInternal::stridedRInitWorkspace(StridedSpan(workspace), stream);
        }
        auto stamped = std::shared_ptr<StampedCubReduction>(new StampedCubReduction(op,
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
        stamped->modern_rk_strategy_chain = CubReductionInternal::stridedRLaunchName(launch);
        return stamped;
    }

    // 1373: what was left was the Thor tiled view family (permutation-aware tiled, VIEW-DIRECT-2B, VIEW-PITCHED-TILED).
    // 1390: analyzeGeometry refuses every view the branches above do not run; 1393 deleted the tiled kernels.
    throw std::logic_error("A value reduction geometry reached the reducer with no executor (the tiled view family "
                           "was deleted in 1393).");
}

std::shared_ptr<StampedCubReduction> CubReduction::stamp(const StridedSpan& input,
                                                         const Tensor& preallocated_output,
                                                         const Stream& stream) const {
    requireGpuTensorView(input, "input");
    requireCompatibleStream(input, stream);
    const CubReductionGeometry geometry =
        analyzeValueSpanGeometry(op, input, axes, resolveOutputDataType(input.getDataType()));
    const DataType resolved_output_dtype = resolveOutputDataType(input.getDataType());
    requireExpectedOutput(input, preallocated_output, resolved_output_dtype, geometry);
    return stampValidated(input, preallocated_output, geometry, stream);
}

std::shared_ptr<StampedCubReduction> CubReduction::stampPitched(const StridedSpan& input,
                                                                uint32_t warps_per_output,
                                                                const Stream& stream) const {
    requireGpuTensorView(input, "input");
    requireCompatibleStream(input, stream);
    requireSupportedOperation(op);
    const std::optional<CubReductionPitchedCompleteLaunch> launch =
        pitchedCompleteLaunch(input, axes, resolveOutputDataType(input.getDataType()), warps_per_output);
    if (!launch.has_value()) {
        throw NotImplementedException("RuntimeK PacketComplete cannot read this view at its row and slab pitches.");
    }
    const CubReductionGeometry geometry = classifyGeometry(input.getDimensions(), input.getStridesElements(), axes);
    Tensor output(input.getPlacement(),
                  TensorDescriptor(resolveOutputDataType(input.getDataType()), geometry.output_dimensions));
    return stampPitchedComplete(input, output, geometry, *launch, stream);
}

// 1392 (phase 7 C): a pitched view read in place, one PacketComplete launch at its pitches. The stamp keeps the view
// and its geometry (getPath() stays its classification); the strategy chain is the launch's name.
std::shared_ptr<StampedCubReduction> CubReduction::stampPitchedComplete(
    const StridedSpan& input,
    const Tensor& output,
    const CubReductionGeometry& geometry,
    const CubReductionPitchedCompleteLaunch& launch,
    const Stream& stream) const {
    requireSupportedFloatingStorageDType(input.getDataType(), "input");
    requireSupportedFloatingStorageDType(output.getDataType(), "output");
    requireExpectedOutput(input, output, output.getDataType(), geometry);
    if (!geometry.pitched_tiled_geometry.has_value()) {
        throw std::logic_error("A pitched PacketComplete stamp requires a pitched view geometry.");
    }
    ScopedGpu scoped_gpu(stream.getGpuNum());
    (void)pitchedRowsSpan(input, *geometry.pitched_tiled_geometry);  // throws if the rows leave the source
    auto stamped = std::shared_ptr<StampedCubReduction>(new StampedCubReduction(op,
                                                                                geometry,
                                                                                input,
                                                                                output,
                                                                                0,
                                                                                Tensor(),
                                                                                true,
                                                                                true,
                                                                                geometry.pitched_tiled_geometry
                                                                                    ->reduction_size,
                                                                                output_scale,
                                                                                stream));
    stamped->pitched_complete_launch = launch;
    stamped->modern_rk_strategy_chain = "pitched_complete:w" + std::to_string(launch.warps_per_output) + "_t" +
                                        std::to_string(launch.tile_components);
    return stamped;
}

CubSegmentedReduction::CubSegmentedReduction(CubReductionOp op,
                                                       std::optional<DataType> output_dtype)
    : op(op), output_dtype(output_dtype) {
    requireSupportedSegmentedOperation(op);
    if (output_dtype.has_value()) {
        requireSupportedFloatingStorageDType(output_dtype.value(), "segmented output");
    }
}

bool CubSegmentedReduction::isInputDataTypeSupported(DataType dtype) {
    return isSupportedFloatingStorageDType(dtype);
}

bool CubSegmentedReduction::isOffsetDataTypeSupported(DataType dtype) {
    if (dtype == DataType::UINT32) {
        return true;
    }
#if THOR_CUB_ENABLE_64BIT_SEGMENT_OFFSETS
    if (dtype == DataType::UINT64) {
        return true;
    }
#endif
    return false;
}

DataType CubSegmentedReduction::resolveOutputDataType(DataType input_dtype) const {
    requireSupportedFloatingStorageDType(input_dtype, "segmented input");
    const DataType resolved = output_dtype.value_or(input_dtype);
    requireSupportedFloatingStorageDType(resolved, "segmented output");
    return resolved;
}

std::shared_ptr<StampedCubSegmentedReduction> CubSegmentedReduction::stamp(
    const Tensor& input, const Tensor& segment_offsets, const Stream& stream) const {
    requireDenseContiguousGpuTensor(input, "input");
    static_cast<void>(segmentedElementsPerValue(input));
    requireSupportedFloatingStorageDType(input.getDataType(), "segmented input");
    requireSegmentOffsets(input, segment_offsets, stream);
    validateSegmentOffsetContentsAtStamp(input, segment_offsets, stream);
    const uint64_t num_segments = segment_offsets.getDimensions()[0] - 1;
    std::vector<uint64_t> output_dimensions = input.getDimensions();
    output_dimensions[0] = num_segments;
    Tensor output(input.getPlacement(),
                  TensorDescriptor(resolveOutputDataType(input.getDataType()), output_dimensions));
    return stampValidated(input, output, segment_offsets, num_segments, stream);
}

std::shared_ptr<StampedCubSegmentedReduction> CubSegmentedReduction::stamp(
    const Tensor& input,
    const Tensor& preallocated_output,
    const Tensor& segment_offsets,
    const Stream& stream) const {
    requireDenseContiguousGpuTensor(input, "input");
    static_cast<void>(segmentedElementsPerValue(input));
    requireSupportedFloatingStorageDType(input.getDataType(), "segmented input");
    requireSegmentOffsets(input, segment_offsets, stream);
    validateSegmentOffsetContentsAtStamp(input, segment_offsets, stream);
    const uint64_t num_segments = segment_offsets.getDimensions()[0] - 1;
    requireExpectedSegmentedOutput(input,
                                   preallocated_output,
                                   segment_offsets,
                                   resolveOutputDataType(input.getDataType()),
                                   num_segments);
    return stampValidated(input, preallocated_output, segment_offsets, num_segments, stream);
}

std::shared_ptr<StampedCubSegmentedReduction> CubSegmentedReduction::stampRuntimeOffsets(
    const Tensor& input,
    const Tensor& preallocated_output,
    const Tensor& segment_offsets,
    const Stream& stream) const {
    requireDenseContiguousGpuTensor(input, "input");
    static_cast<void>(segmentedElementsPerValue(input));
    requireSupportedFloatingStorageDType(input.getDataType(), "segmented input");
    requireSegmentOffsets(input, segment_offsets, stream);
    const uint64_t num_segments = segment_offsets.getDimensions()[0] - 1;
    requireExpectedSegmentedOutput(input,
                                   preallocated_output,
                                   segment_offsets,
                                   resolveOutputDataType(input.getDataType()),
                                   num_segments);
    return stampValidated(input, preallocated_output, segment_offsets, num_segments, stream);
}

std::shared_ptr<StampedCubSegmentedReduction> CubSegmentedReduction::stampValidated(
    const Tensor& input,
    const Tensor& output,
    const Tensor& segment_offsets,
    uint64_t num_segments,
    const Stream& stream) const {
    const uint64_t num_items = input.getTotalNumElements();
    if (input.getDimensions().size() == 1
        && (num_items > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())
            || num_segments > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))) {
        throw std::invalid_argument("Scalar CUB segmented-reduction item or segment count exceeds int64 limits.");
    }

    ScopedGpu scoped_gpu(stream.getGpuNum());
    // 1325 (phase 7): the stamp runs the selector's ragged winner; its boundary workspace is the stamp's temporary
    // storage. 1330 (phase 9): every site (the legacy offset-segmented kernels are deleted); raggedRouteSegmented
    // throws, with the reason, for a site no ragged family runs (rows over 2 GiB).
    std::shared_ptr<const CubReductionInternal::RaggedRoutedStamp> routed = CubReductionInternal::raggedRouteSegmented(
        false, op, CubArgReductionOp::ArgMin, input, output, segment_offsets, stream);
    const size_t bytes = static_cast<size_t>(routed->launch.workspace_bytes);
    Tensor workspace(input.getPlacement(), TensorDescriptor(DataType::UINT8, {static_cast<uint64_t>(bytes)}));
    std::shared_ptr<StampedCubSegmentedReduction> stamped(new StampedCubSegmentedReduction(
        op, input, output, segment_offsets, num_items, num_segments, bytes, workspace, stream));
    stamped->ragged = std::move(routed);
    return stamped;
}

StampedCubSegmentedReduction::StampedCubSegmentedReduction(CubReductionOp op,
                                                           const Tensor& input,
                                                           const Tensor& output,
                                                           const Tensor& segment_offsets,
                                                           uint64_t num_items,
                                                           uint64_t num_segments,
                                                           size_t temp_storage_bytes,
                                                           const Tensor& temp_storage,
                                                           const Stream& stream)
    : op(op),
      input(input),
      output(output),
      segment_offsets(segment_offsets),
      num_items(num_items),
      num_segments(num_segments),
      temp_storage_bytes(temp_storage_bytes),
      temp_storage(temp_storage),
      stream(stream) {
    requireTempStorage(this->temp_storage, input.getPlacement(), temp_storage_bytes);
}

void StampedCubSegmentedReduction::run() { runOn(stream); }

void StampedCubSegmentedReduction::runOn(Stream& run_stream) const {
    requireCompatibleStream(input, run_stream);
    ScopedGpu scoped_gpu(run_stream.getGpuNum());
    CubReductionInternal::runRaggedRouted(*ragged, input_span, offsets_span, output_span, workspace_span, run_stream);
}

CubSegmentedArgReduction::CubSegmentedArgReduction(CubArgReductionOp op, DataType index_output_dtype)
    : op(op), index_output_dtype(index_output_dtype) {
    requireSupportedArgOperation(op);
    requireSupportedArgIndexDType(index_output_dtype);
}

bool CubSegmentedArgReduction::isInputDataTypeSupported(DataType dtype) {
    return isSupportedFloatingStorageDType(dtype);
}

bool CubSegmentedArgReduction::isOffsetDataTypeSupported(DataType dtype) {
    if (dtype == DataType::UINT32) {
        return true;
    }
#if THOR_CUB_ENABLE_64BIT_SEGMENT_OFFSETS
    if (dtype == DataType::UINT64) {
        return true;
    }
#endif
    return false;
}

std::shared_ptr<StampedCubSegmentedArgReduction> CubSegmentedArgReduction::stamp(
    const Tensor& input, const Tensor& segment_offsets, const Stream& stream) const {
    requireDenseContiguousGpuTensor(input, "input");
    static_cast<void>(segmentedElementsPerValue(input));
    requireSupportedFloatingStorageDType(input.getDataType(), "segmented arg input");
    requireSegmentOffsets(input, segment_offsets, stream);
    validateSegmentOffsetContentsAtStamp(input, segment_offsets, stream);
    const uint64_t num_segments = segment_offsets.getDimensions()[0] - 1;
    std::vector<uint64_t> output_dimensions = input.getDimensions();
    output_dimensions[0] = num_segments;
    Tensor output(input.getPlacement(), TensorDescriptor(index_output_dtype, output_dimensions));
    return stampValidated(input, output, segment_offsets, num_segments, stream);
}

std::shared_ptr<StampedCubSegmentedArgReduction> CubSegmentedArgReduction::stamp(
    const Tensor& input,
    const Tensor& preallocated_index_output,
    const Tensor& segment_offsets,
    const Stream& stream) const {
    requireDenseContiguousGpuTensor(input, "input");
    static_cast<void>(segmentedElementsPerValue(input));
    requireSupportedFloatingStorageDType(input.getDataType(), "segmented arg input");
    requireSegmentOffsets(input, segment_offsets, stream);
    validateSegmentOffsetContentsAtStamp(input, segment_offsets, stream);
    const uint64_t num_segments = segment_offsets.getDimensions()[0] - 1;
    requireExpectedSegmentedArgOutput(input, preallocated_index_output, segment_offsets, index_output_dtype, num_segments);
    return stampValidated(input, preallocated_index_output, segment_offsets, num_segments, stream);
}

std::shared_ptr<StampedCubSegmentedArgReduction> CubSegmentedArgReduction::stampRuntimeOffsets(
    const Tensor& input,
    const Tensor& preallocated_index_output,
    const Tensor& segment_offsets,
    const Stream& stream) const {
    requireDenseContiguousGpuTensor(input, "input");
    static_cast<void>(segmentedElementsPerValue(input));
    requireSupportedFloatingStorageDType(input.getDataType(), "segmented arg input");
    requireSegmentOffsets(input, segment_offsets, stream);
    const uint64_t num_segments = segment_offsets.getDimensions()[0] - 1;
    requireExpectedSegmentedArgOutput(input, preallocated_index_output, segment_offsets, index_output_dtype, num_segments);
    return stampValidated(input, preallocated_index_output, segment_offsets, num_segments, stream);
}

std::shared_ptr<StampedCubSegmentedArgReduction> CubSegmentedArgReduction::stampValidated(
    const Tensor& input,
    const Tensor& index_output,
    const Tensor& segment_offsets,
    uint64_t num_segments,
    const Stream& stream) const {
    const uint64_t num_items = input.getTotalNumElements();
    if (index_output_dtype == DataType::UINT32 && num_items > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::invalid_argument("CUB segmented arg-reduction packed index domain does not fit in UINT32.");
    }
    if (num_segments > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        throw std::invalid_argument("CUB segmented arg-reduction segment count exceeds int64 limits.");
    }

    ScopedGpu scoped_gpu(stream.getGpuNum());
    // 1325 (phase 7): as CubSegmentedReduction::stampValidated, with the ragged ARG kernels (1330: every site).
    std::shared_ptr<const CubReductionInternal::RaggedRoutedStamp> routed = CubReductionInternal::raggedRouteSegmented(
        true, CubReductionOp::Min, op, input, index_output, segment_offsets, stream);
    const size_t bytes = static_cast<size_t>(routed->launch.workspace_bytes);
    Tensor workspace(input.getPlacement(), TensorDescriptor(DataType::UINT8, {static_cast<uint64_t>(bytes)}));
    std::shared_ptr<StampedCubSegmentedArgReduction> stamped(new StampedCubSegmentedArgReduction(
        op, input, index_output, segment_offsets, num_segments, bytes, workspace, stream));
    stamped->ragged = std::move(routed);
    return stamped;
}

StampedCubSegmentedArgReduction::StampedCubSegmentedArgReduction(CubArgReductionOp op,
                                                                 const Tensor& input,
                                                                 const Tensor& index_output,
                                                                 const Tensor& segment_offsets,
                                                                 uint64_t num_segments,
                                                                 size_t temp_storage_bytes,
                                                                 const Tensor& temp_storage,
                                                                 const Stream& stream)
    : op(op),
      input(input),
      index_output(index_output),
      segment_offsets(segment_offsets),
      num_segments(num_segments),
      temp_storage_bytes(temp_storage_bytes),
      temp_storage(temp_storage),
      stream(stream) {
    requireTempStorage(this->temp_storage, input.getPlacement(), temp_storage_bytes);
}

void StampedCubSegmentedArgReduction::run() { runOn(stream); }

void StampedCubSegmentedArgReduction::runOn(Stream& run_stream) const {
    requireCompatibleStream(input, run_stream);
    ScopedGpu scoped_gpu(run_stream.getGpuNum());
    CubReductionInternal::runRaggedRouted(*ragged, input_span, offsets_span, index_output_span, workspace_span,
                                          run_stream);
}

CubReductionGeometry CubArgReduction::analyzeDenseGeometry(
    const std::vector<uint64_t>& input_dimensions,
    const std::vector<uint32_t>& axes) {
    CubReductionGeometry geometry = CubReduction::analyzeGeometry(input_dimensions, axes);
    // DENSE-GATE-FINAL: this public helper accepts only canonical dense geometry, so returning the legacy arbitrary
    // view family would be an internal architecture violation rather than a recoverable execution choice.
    requireOrdainedDenseReductionPath(geometry);
    return geometry;
}

uint64_t CubArgReduction::composeOriginalArgIndex(uint64_t previous_index,
                                                  uint64_t local_run_index,
                                                  uint64_t domain_stride) {
    if (domain_stride != 0 && local_run_index > std::numeric_limits<uint64_t>::max() / domain_stride) {
        throw std::overflow_error("CUB arg reduction original-index contribution overflows uint64_t.");
    }
    const uint64_t contribution = local_run_index * domain_stride;
    if (previous_index > std::numeric_limits<uint64_t>::max() - contribution) {
        throw std::overflow_error("CUB arg reduction composed original index overflows uint64_t.");
    }
    return previous_index + contribution;
}

CubArgReduction::CubArgReduction(CubArgReductionOp op,
                                 uint32_t axis,
                                 CubArgReductionOutputOptions outputs,
                                 std::optional<CubReductionTuningDepth> tuning_depth)
    : CubArgReduction(op, std::vector<uint32_t>{axis}, std::move(outputs), tuning_depth) {}

CubArgReduction::CubArgReduction(CubArgReductionOp op,
                                 std::vector<uint32_t> axes,
                                 CubArgReductionOutputOptions outputs,
                                 std::optional<CubReductionTuningDepth> tuning_depth)
    : op(op), axes(std::move(axes)), outputs(std::move(outputs)), tuning_depth(resolveCubReductionTuningDepth(tuning_depth)) {
    requireSupportedArgOperation(op);
    if (this->axes.empty()) {
        throw std::invalid_argument("CUB arg reduction requires at least one reduction axis.");
    }
    for (size_t i = 1; i < this->axes.size(); ++i) {
        if (this->axes[i] <= this->axes[i - 1]) {
            throw std::invalid_argument("CUB arg reduction axes must be unique and strictly increasing.");
        }
    }
    if (!this->outputs.produce_value && !this->outputs.produce_index) {
        throw std::invalid_argument("CUB arg reduction must produce a value, an index, or both.");
    }
    if (!this->outputs.produce_value && this->outputs.value_output_dtype.has_value()) {
        throw std::invalid_argument("CUB arg reduction cannot configure a value dtype when value output is disabled.");
    }
    if (this->outputs.value_output_dtype.has_value()) {
        requireSupportedFloatingStorageDType(this->outputs.value_output_dtype.value(), "value output");
    }
    if (this->outputs.produce_index) {
        requireSupportedArgIndexDType(this->outputs.index_output_dtype);
    }
}

DataType CubArgReduction::resolveValueOutputDataType(DataType input_dtype) const {
    requireSupportedFloatingStorageDType(input_dtype, "input");
    const DataType resolved = outputs.value_output_dtype.value_or(input_dtype);
    requireSupportedFloatingStorageDType(resolved, "value output");
    return resolved;
}

float CubArgReduction::getFp32EmptyReductionValue(CubArgReductionOp op) {
    requireSupportedArgOperation(op);
    return op == CubArgReductionOp::ArgMin ? std::numeric_limits<float>::infinity()
                                           : -std::numeric_limits<float>::infinity();
}

size_t CubArgReduction::queryWorkspaceSizeInBytes(const TensorDescriptor& input_descriptor,
                                                          const Stream& stream) const {
    requireSupportedFloatingStorageDType(input_descriptor.getDataType(), "input");
    CubReductionGeometry geometry = analyzeDenseGeometry(input_descriptor.getDimensions(), axes);
    if (outputs.produce_index && outputs.index_output_dtype == DataType::UINT32
        && geometry.reduction_size > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::invalid_argument("CUB arg reduction domain does not fit in a UINT32 local index.");
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
            throw std::overflow_error("CUB arg reduction workspace size overflows size_t.");
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

std::shared_ptr<StampedCubArgReduction> CubArgReduction::stamp(const StridedSpan& input,
                                                               const Stream& stream) const {
    return stamp(input, std::nullopt, std::nullopt, stream);
}

std::shared_ptr<StampedCubArgReduction> CubArgReduction::stamp(
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

    CubReductionGeometry geometry = analyzeDenseGeometry(input.getDimensions(), axes);
    if (outputs.produce_index && outputs.index_output_dtype == DataType::UINT32
        && geometry.reduction_size > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::invalid_argument("CUB arg reduction domain does not fit in a UINT32 local index.");
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
        throw std::invalid_argument("CUB arg reduction received a value output while value production is disabled.");
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
        throw std::invalid_argument("CUB arg reduction received an index output while index production is disabled.");
    }

    requireArgOutputsDoNotOverlap(value_output, index_output);
    if (!usesPlannedDenseArg(input.getDataType())) {
        throw std::logic_error("Dense ARG has no planned executor for this input dtype.");
    }
    // 1261: every dense ARG executes the planned retained pass list; the interval-composition executor is deleted.
    return stampPlannedDenseValidated(input, std::move(value_output), std::move(index_output), geometry, stream);
}

std::shared_ptr<StampedCubArgReduction> CubArgReduction::stampPlannedDenseValidated(
    const StridedSpan& input,
    std::optional<Tensor> value_output,
    std::optional<Tensor> index_output,
    const CubReductionGeometry& geometry,
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

    std::vector<std::shared_ptr<StampedCubArgReduction>> stages;
    stages.reserve(planned.passes.size());
    StridedSpan current_value_input = input;
    std::optional<Tensor> current_index_input = std::nullopt;
    size_t workspace_size_bytes = 0;
    auto add_bytes = [&](uint64_t bytes) {
        if (bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max() - workspace_size_bytes)) {
            throw std::overflow_error("CUB arg reduction workspace size overflows size_t.");
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
        stages.push_back(std::shared_ptr<StampedCubArgReduction>(new StampedCubArgReduction(op,
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
    return std::shared_ptr<StampedCubArgReduction>(new StampedCubArgReduction(op,
                                                                             geometry,
                                                                             input,
                                                                             std::move(value_output),
                                                                             std::move(index_output),
                                                                             workspace_size_bytes,
                                                                             std::move(stages),
                                                                             stream));
}

StampedCubArgReduction::StampedCubArgReduction(CubArgReductionOp op,
                                               CubReductionGeometry geometry,
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
        throw std::invalid_argument("Stamped composed CUB arg stage requires at least one output.");
    }
    bindLaunchSpans();
}

void StampedCubArgReduction::bindLaunchSpans() {
    if (value_output.has_value()) value_output_span = StridedSpan(*value_output);
    if (index_output.has_value()) index_output_span = StridedSpan(*index_output);
    if (carried_index_input.has_value()) carried_index_span = StridedSpan(*carried_index_input);
}

StampedCubArgReduction::StampedCubArgReduction(
    CubArgReductionOp op,
    CubReductionGeometry geometry,
    const StridedSpan& input,
    std::optional<Tensor> value_output,
    std::optional<Tensor> index_output,
    size_t workspace_size_bytes,
    std::vector<std::shared_ptr<StampedCubArgReduction>> composed_stages,
    const Stream& stream)
    : op(op),
      geometry(std::move(geometry)),
      input(input),
      value_output(std::move(value_output)),
      index_output(std::move(index_output)),
      temp_storage_bytes(workspace_size_bytes),
      composed_stages(std::move(composed_stages)),
      stream(stream) {
    // Planned dense ARG (ARG-P1): the public geometry may be any dense path, ComposedDense included.
    THOR_THROW_IF_FALSE(this->geometry.path == CubReductionPath::ComposedDense ||
                        isDirectDenseReductionPath(this->geometry.path));
    THOR_THROW_IF_FALSE(!this->composed_stages.empty());
    for (const auto& stage : this->composed_stages) {
        THOR_THROW_IF_FALSE(stage != nullptr);
        THOR_THROW_IF_FALSE(stage->geometry.path != CubReductionPath::ComposedDense);
        THOR_THROW_IF_FALSE(stage->composed_carried_index_dtype.has_value());
    }
    if (!this->value_output.has_value() && !this->index_output.has_value()) {
        throw std::invalid_argument("Stamped composed CUB arg reduction requires at least one public output.");
    }
    bindLaunchSpans();
}

std::vector<std::vector<uint32_t>> StampedCubArgReduction::getComposedStageAxes() const {
    if (geometry.path != CubReductionPath::ComposedDense) {
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

void StampedCubArgReduction::run() { runOn(stream); }

void StampedCubArgReduction::runOn(Stream& run_stream) const {
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

void StampedCubArgReduction::launchBoundStage(const StridedSpan& stage_input, Stream& run_stream) const {
    launchBoundStage(stage_input, value_output_span.has_value() ? &value_output_span.value() : nullptr,
                     index_output_span.has_value() ? &index_output_span.value() : nullptr, run_stream);
}

void StampedCubArgReduction::launchBoundStage(const StridedSpan& stage_input, const StridedSpan* value_output_ptr,
                                              const StridedSpan* index_output_ptr, Stream& run_stream) const {
    if (fp32_arg_plan) {
        // 1221: a retained FP32 ARG pass; a carried pass reads this stage's carried index, and staged / Split partials
        // live in this stage's scratch.
        THOR_THROW_IF_FALSE(carried_index_input.has_value() == fp32_arg_plan->carried);
        THOR_THROW_IF_FALSE(temp_storage_bytes >= fp32_arg_plan->scratch_bytes);
        void* scratch = fp32_arg_plan->scratch_bytes == 0
                            ? nullptr
                            : const_cast<void*>(static_cast<const void*>(temp_storage.getMemPtr<void>()));
        CubReductionInternal::launchRetainedFp32ArgPlan(*fp32_arg_plan, stage_input,
                                                        carried_index_span.has_value() ? &*carried_index_span : nullptr,
                                                        value_output_ptr, index_output_ptr, geometry, scratch,
                                                        op == CubArgReductionOp::ArgMax, composed_domain_stride,
                                                        run_stream);
        return;
    }
    if (fp8_arg_plan) {
        // Retained FP8 ARG pass (packed staged twins carry their partials in this stage's scratch).
        THOR_THROW_IF_FALSE(!carried_index_input.has_value());
        uint32_t* packed = nullptr;
        if (fp8_arg_plan->packed_partials) {
            THOR_THROW_IF_FALSE(temp_storage_bytes >= CubReductionInternal::retainedFp8ArgScratchBytes(*fp8_arg_plan));
            packed = static_cast<uint32_t*>(const_cast<void*>(static_cast<const void*>(temp_storage.getMemPtr<void>())));
        }
        CubReductionInternal::launchRetainedFp8ArgPlan(*fp8_arg_plan, stage_input, value_output_ptr, index_output_ptr,
                                                       geometry, packed, op == CubArgReductionOp::ArgMax,
                                                       composed_domain_stride, run_stream);
        return;
    }
    // 1261: the CUB direct and interval-composition ARG stages are deleted; every stage is a retained pass.
    throw std::logic_error("Stamped dense ARG stage has no retained ARG plan.");
}

void StampedCubArgReduction::runOnInputForBenchmark(const StridedSpan& replacement_input, Stream& run_stream) const {
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

namespace {

template <typename Replacement, typename Bound>
void requireBenchmarkReplacement(const Replacement& replacement, const Bound& bound, const char* what) {
    if (!replacement.isInitialized() || replacement.getDataType() != bound.getDataType() ||
        replacement.getTotalNumElements() != bound.getTotalNumElements() ||
        replacement.getPlacement() != bound.getPlacement() || !replacement.isDenseContiguous()) {
        throw std::invalid_argument(std::string("runOnTensorsForBenchmark: the replacement ") + what +
                                    " must be a dense tensor of the stamp's dtype, element count and device.");
    }
}

// 1335: a stamp bound to a non-dense view takes a replacement view of the same dtype, device, dims and strides (its
// geometry addresses the view's strides from the replacement's address); a dense stamp takes a dense replacement.
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

void StampedCubSegmentedReduction::runOnTensorsForBenchmark(const StridedSpan& replacement_input,
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
    CubReductionInternal::runRaggedRouted(*ragged, bound_input, bound_offsets, bound_output, workspace_span, run_stream);
}

std::vector<std::string> StampedCubSegmentedReduction::getPassNames() const {
    return CubReductionInternal::raggedRoutedPassNames(*ragged);  // 1325
}

void StampedCubSegmentedArgReduction::runOnTensorsForBenchmark(const StridedSpan& replacement_input,
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
    CubReductionInternal::runRaggedRouted(*ragged, bound_input, bound_offsets, bound_index_output, workspace_span,
                                          run_stream);
}

std::vector<std::string> StampedCubSegmentedArgReduction::getPassNames() const {
    return CubReductionInternal::raggedRoutedPassNames(*ragged);  // 1325
}

void StampedCubArgReduction::runOnTensorsForBenchmark(const StridedSpan& replacement_input,
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
        const StampedCubArgReduction& executor = *composed_stages[stage];
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

std::vector<std::string> StampedCubArgReduction::getPlannedPassNames() const {
    std::vector<std::string> names;
    if (geometry.path != CubReductionPath::ComposedDense && composed_stages.empty()) return names;
    for (const auto& stage : composed_stages) {
        if (stage->fp32_arg_plan) {  // 1221
            names.push_back(CubReductionInternal::retainedFp32ArgPassName(*stage->fp32_arg_plan));
            continue;
        }
        if (stage->fp8_arg_plan && stage->fp8_arg_plan->kr.group_lanes != 0) {
            names.push_back("fp8_arg_kr:" + CubReductionInternal::Fp8KR::fp8KRShapeName(stage->fp8_arg_plan->kr));
            continue;
        }
        if (stage->fp8_arg_plan && stage->fp8_arg_plan->family == CubReductionInternal::Fp8RK::PhysicalFamily::RuntimeK &&
            stage->fp8_arg_plan->tile_components != 0) {
            names.push_back("fp8_arg_runtime_k:packet_complete_w" + std::to_string(stage->fp8_arg_plan->warps_per_output) +
                            "_tile" + std::to_string(stage->fp8_arg_plan->tile_components));
            if (stage->fp8_arg_plan->packed_partials) names.emplace_back("fp8_arg_packed_continuation");
            continue;
        }
        if (stage->fp8_arg_plan && stage->fp8_arg_plan->family == CubReductionInternal::Fp8RK::PhysicalFamily::RuntimeK) {
            names.push_back("fp8_arg_runtime_k:p" + std::to_string(stage->fp8_arg_plan->packet_bytes) +
                            (stage->fp8_arg_plan->shards == 1 ? std::string("_complete")
                                                              : "_rows" + std::to_string(stage->fp8_arg_plan->rows_per_shard)));
            if (stage->fp8_arg_plan->packed_partials) names.emplace_back("fp8_arg_packed_continuation");
            continue;
        }
        if (stage->fp8_arg_plan && stage->fp8_arg_plan->family == CubReductionInternal::Fp8RK::PhysicalFamily::AlignedK) {
            names.push_back("fp8_arg_aligned_k:p" + std::to_string(stage->fp8_arg_plan->packet_bytes) + "_cta" +
                            std::to_string(stage->fp8_arg_plan->cta_threads) + "_shards" +
                            std::to_string(stage->fp8_arg_plan->shards));
            if (stage->fp8_arg_plan->packed_partials) names.emplace_back("fp8_arg_packed_continuation");
            continue;
        }
        if (stage->fp8_arg_plan && stage->fp8_arg_plan->family == CubReductionInternal::Fp8RK::PhysicalFamily::DeepRSmallK) {
            names.push_back("fp8_arg_deep_r_small_k:p16_rows" + std::to_string(stage->fp8_arg_plan->rows_per_shard));
            if (stage->fp8_arg_plan->packed_partials) names.emplace_back("fp8_arg_packed_continuation");
            continue;
        }
        if (stage->fp8_arg_plan) {
            const CubReductionInternal::Fp8OneWindowRuntimeKUnit& unit = stage->fp8_arg_plan->unit;
            names.push_back("fp8_arg_one_window:o" + std::to_string(unit.outputs_per_warp) + "_w" +
                            std::to_string(stage->fp8_arg_plan->warps_per_output) + "_p" +
                            std::to_string(unit.packet_bytes) + "_c" + std::to_string(unit.max_components_per_lane) +
                            (unit.load_ahead ? "_la" : ""));
            if (stage->fp8_arg_plan->packed_partials) names.emplace_back("fp8_arg_packed_continuation");
            continue;
        }
        throw std::logic_error("Planned dense ARG stage has no retained ARG plan.");  // 1261
    }
    return names;
}

StampedCubReduction::StampedCubReduction(CubReductionOp op,
                                         CubReductionGeometry geometry,
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

StampedCubReduction::StampedCubReduction(
    CubReductionOp op,
    CubReductionGeometry geometry,
    const StridedSpan& input,
    const Tensor& output,
    CubReductionInternal::DenseExecutablePlan dense_executable_plan,
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
    // 1377: a view that runs as the dense tensor of its bytes (viewPhysicalDenseProblem) keeps its classification
    // geometry, which can still note a tiled candidate (a natural-output permutation-aware tiled view, case D).
    THOR_THROW_IF_FALSE(isOrdinaryDenseValueGeometry(this->geometry)
                        || viewPhysicalDenseProblem(this->input, this->geometry).has_value());
    THOR_THROW_IF_FALSE(!this->dense_executable_plan.passes.empty());

    size_t summed_workspace_bytes = 0;
    for (const CubReductionInternal::DenseExecutablePass& pass : this->dense_executable_plan.passes) {
        THOR_THROW_IF_FALSE(pass.physical_plan != nullptr);
        if (summed_workspace_bytes > std::numeric_limits<size_t>::max() - pass.workspaceBytes()) {
            throw std::overflow_error("Dense executable plan workspace accounting overflows size_t.");
        }
        summed_workspace_bytes += pass.workspaceBytes();
    }
    THOR_THROW_IF_FALSE(summed_workspace_bytes == this->dense_executable_plan.workspace_size_bytes);
    THOR_THROW_IF_FALSE(this->dense_executable_plan.passes.back().role == CubReductionInternal::DensePassRole::Complete
                        || this->dense_executable_plan.passes.back().role == CubReductionInternal::DensePassRole::Final);
}

std::vector<DensePhysicalPassDiagnostic> StampedCubReduction::getDensePhysicalPassDiagnostics() const {
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
        if (const auto* fp8 = dynamic_cast<const CubReductionInternal::DenseRetainedFp8RKPhysicalPlan*>(
                pass.physical_plan.get())) {
            const auto& kernel = fp8->kernel;
            switch (kernel.family) {
                case CubReductionInternal::Fp8RK::PhysicalFamily::OneWindow:
                    info.family = "retained_fp8_rk:one_window";
                    info.packet_bytes = kernel.c;
                    break;
                case CubReductionInternal::Fp8RK::PhysicalFamily::DeepRSmallK:
                    info.family = "retained_fp8_rk:deep_r_small_k";
                    break;
                case CubReductionInternal::Fp8RK::PhysicalFamily::AlignedK:
                    info.family = "retained_fp8_rk:aligned_k";
                    info.packet_bytes = kernel.a;
                    info.block_threads = static_cast<uint32_t>(kernel.b);
                    break;
                case CubReductionInternal::Fp8RK::PhysicalFamily::RuntimeK:
                    info.family = "retained_fp8_rk:runtime_k";
                    break;
            }
            // 1220: a retained FP32 RK pass is "retained_fp32_rk:...". 1254: a 2-byte pass "retained_f16_rk:...".
            if (fp8->input_dtype == DataType::FP32) info.family.replace(0, 15, "retained_fp32_rk");
            if (kernel.element_bytes == 2) info.family.replace(0, 15, "retained_f16_rk");
            if (kernel.first_stage_ctas != 0) info.stage_blocks = kernel.first_stage_ctas;
            // The actual next tensor geometry is authoritative for staged N.
            info.shards_per_output = kernel.staged() ? fp8->output_dimensions.at(1) : 1;
        } else if (const auto* kr = dynamic_cast<const CubReductionInternal::DenseRetainedFp8KRPhysicalPlan*>(
                       pass.physical_plan.get())) {
            info.family = retainedKRFamilyPrefix(kr->input_dtype) +
                          CubReductionInternal::Fp8KR::fp8KRShapeName(kr->shape);
            info.block_threads = CubReductionInternal::Fp8KR::FP8_KR_THREADS;
            info.shards_per_output = kr->shape.chunks();
        } else {
            throw std::logic_error("Dense VALUE pass without a retained physical plan");
        }
        result.push_back(std::move(info));
    }
    return result;
}

bool StampedCubReduction::usesModernRKFamilyPlan() const {
    if (dense_executable_plan.passes.empty()) {
        return false;
    }
    return std::all_of(dense_executable_plan.passes.begin(),
                       dense_executable_plan.passes.end(),
                       [](const CubReductionInternal::DenseExecutablePass& pass) {
                           // 1258: every RK pass is a retained RK physical plan (the legacy RK families are gone).
                           return std::dynamic_pointer_cast<const CubReductionInternal::DenseRetainedFp8RKPhysicalPlan>(
                                      pass.physical_plan) != nullptr;
                       });
}

std::vector<std::vector<uint32_t>> StampedCubReduction::getComposedStageAxes() const {
    if (geometry.path != CubReductionPath::ComposedDense) {
        return {};
    }
    THOR_THROW_IF_FALSE(!dense_executable_plan.passes.empty());
    std::vector<std::vector<uint32_t>> stage_axes;
    stage_axes.reserve(dense_executable_plan.passes.size());
    for (const CubReductionInternal::DenseExecutablePass& pass : dense_executable_plan.passes) {
        stage_axes.push_back(pass.geometry.axes);
    }
    return stage_axes;
}

void StampedCubReduction::run() { runOn(stream, output_scale); }

void StampedCubReduction::run(float runtime_output_scale) { runOn(stream, runtime_output_scale); }

void StampedCubReduction::runOn(Stream& run_stream) const { runOn(run_stream, output_scale); }

void StampedCubReduction::runOnTensorsForBenchmark(const StridedSpan& replacement_input, const StridedSpan& replacement_output,
                                                   Stream& run_stream) const {
    requireBenchmarkInputReplacement(replacement_input, input);  // 1335: a strided stamp takes a strided replacement
    requireBenchmarkReplacement(replacement_output, output, "output");
    requireCompatibleStream(replacement_input, run_stream);
    ScopedGpu scoped_gpu(run_stream.getGpuNum());

    if (strided_r_launch.has_value()) {  // 1366: the replacement is a view of the stamp's dims and strides
        CubReductionInternal::launchStridedR(op, *strided_r_launch, replacement_input,
                                             reshapedForBenchmark(replacement_output, output.getDimensions()),
                                             temp_storage_span, output_scale, run_stream);
        return;
    }
    if (pitched_complete_launch.has_value()) {  // 1392: as is the pitched view's
        launchPitchedComplete(*pitched_complete_launch,
                              CubReductionInternal::makeValueReductionStageSemantics(
                                  op, physicalStageRole(true, true), semantic_total_reduction_size),
                              pitchedRowsSpan(replacement_input, *geometry.pitched_tiled_geometry),
                              reshapedForBenchmark(replacement_output, output.getDimensions()),
                              geometry,
                              output_scale,
                              run_stream);
        return;
    }
    if (dense_executable_plan.passes.empty()) {  // 1393: every stamp is a dense plan, StridedR or pitched
        throw std::logic_error("A value reduction stamp has no executor (the tiled view family was deleted in 1393).");
    }
    const size_t last = dense_executable_plan.passes.size() - 1;
    for (size_t index = 0; index <= last; ++index) {
        const CubReductionInternal::DenseExecutablePass& pass = dense_executable_plan.passes[index];
        CubReductionInternal::CubReductionStageSemantics stage_semantics =
            CubReductionInternal::makeValueReductionStageSemantics(op, pass.role, semantic_total_reduction_size);
        if (!apply_input_transform) {
            stage_semantics.input_transform = CubReductionInternal::CubReductionStageInputTransform::Identity;
        }
        if (!apply_output_finalize) {
            stage_semantics.finalize = CubReductionInternal::CubReductionStageFinalize::Identity;
            stage_semantics.finalize_divisor = 1;
        }
        const float stage_output_scale = pass.role == CubReductionInternal::DensePassRole::Complete ||
                                                 pass.role == CubReductionInternal::DensePassRole::Final
                                             ? output_scale
                                             : 1.0f;
        // The first pass reads the replacement input, the last writes the replacement output (aliased to the pass's
        // own dimensions); every other tensor is the stamp's. 1365: a view stamp's first pass reads the replacement
        // view as dense, as the stamp read its own view.
        const StridedSpan pass_input =
            index != 0                  ? pass.input
            : input.isDenseContiguous() ? reshapedForBenchmark(replacement_input, pass.input.getDimensions())
                                        : replacement_input.asDense(pass.input.getDimensions());
        const StridedSpan pass_output = index == last
                                            ? reshapedForBenchmark(replacement_output, pass.output.getDimensions())
                                            : pass.output_span;
        CubReductionInternal::launchDensePhysicalPass(*pass.physical_plan, stage_semantics, pass.temp_storage_span,
                                                      pass.temp_storage_bytes, pass_input, pass_output, pass.geometry,
                                                      stage_output_scale, run_stream);
    }
}

void StampedCubReduction::runOn(Stream& run_stream, float runtime_output_scale) const {
    THOR_THROW_IF_FALSE(std::isfinite(runtime_output_scale));
    requireCompatibleStream(input, run_stream);
    ScopedGpu scoped_gpu(run_stream.getGpuNum());

    const CubReductionInternal::CubReductionStageSemantics semantics =
        CubReductionInternal::makeValueReductionStageSemantics(
            op,
            physicalStageRole(apply_input_transform, apply_output_finalize),
            semantic_total_reduction_size);

    if (pitched_complete_launch.has_value()) {  // 1392
        launchPitchedComplete(*pitched_complete_launch,
                              semantics,
                              pitchedRowsSpan(input, *geometry.pitched_tiled_geometry),
                              output_span,
                              geometry,
                              runtime_output_scale,
                              run_stream);
        return;
    }

    if (!dense_executable_plan.passes.empty()) {
        for (const CubReductionInternal::DenseExecutablePass& pass : dense_executable_plan.passes) {
            CubReductionInternal::CubReductionStageSemantics stage_semantics =
                CubReductionInternal::makeValueReductionStageSemantics(
                    op, pass.role, semantic_total_reduction_size);
            if (!apply_input_transform) {
                stage_semantics.input_transform = CubReductionInternal::CubReductionStageInputTransform::Identity;
            }
            if (!apply_output_finalize) {
                stage_semantics.finalize = CubReductionInternal::CubReductionStageFinalize::Identity;
                stage_semantics.finalize_divisor = 1;
            }

            const float stage_output_scale =
                pass.role == CubReductionInternal::DensePassRole::Complete
                        || pass.role == CubReductionInternal::DensePassRole::Final
                    ? runtime_output_scale
                    : 1.0f;

            // Every pass carries only its own actually required scratch.
            // Zero-scratch passes have an uninitialized Tensor handle.
            CubReductionInternal::launchDensePhysicalPass(*pass.physical_plan,
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
        CubReductionInternal::launchStridedR(op, *strided_r_launch, input, output_span, temp_storage_span,
                                             runtime_output_scale, run_stream);
        return;
    }

    // 1393: every stamp is a dense plan, StridedR or pitched (the tiled view family's generic launch is deleted).
    throw std::logic_error("A value reduction stamp has no executor (the tiled view family was deleted in 1393).");
}

}  // namespace ThorImplementation
