#include "Utilities/TensorOperations/Cub/CubReduction.h"

#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Exceptions.h"
#include "Utilities/TensorOperations/Cub/CubDataTypePolicy.h"
#include "Utilities/TensorOperations/Cub/CubDevicePrimitiveSupport.h"
#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"
#include "Utilities/TensorOperations/Cub/DenseReductionFamilies.h"
#include "Utilities/TensorOperations/Cub/DenseReductionPlanner.h"

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

struct ProductionRKOccupancyContext {
    uint32_t max_threads_per_sm = 0;
    uint32_t max_blocks_per_sm = 0;
    uint32_t warp_size = 32;
};

[[nodiscard]] DenseRKValueOperation denseRKValueOperation(CubReductionOp op) {
    switch (op) {
        case CubReductionOp::Sum: return DenseRKValueOperation::Sum;
        case CubReductionOp::Mean: return DenseRKValueOperation::Mean;
        case CubReductionOp::L1Norm: return DenseRKValueOperation::L1;
        case CubReductionOp::L2Norm: return DenseRKValueOperation::L2;
        case CubReductionOp::SumSquares: return DenseRKValueOperation::SumSquares;
        case CubReductionOp::Product: return DenseRKValueOperation::Product;
        case CubReductionOp::Min: return DenseRKValueOperation::Minimum;
        case CubReductionOp::Max: return DenseRKValueOperation::Maximum;
    }
    throw std::logic_error("Invalid DenseRK value operation.");
}

[[nodiscard]] const char* denseRKStrategyName(DenseRKStrategy strategy) {
    switch (strategy) {
        case DenseRKStrategy::KParallel: return "k_parallel";
        case DenseRKStrategy::RCooperative: return "r_cooperative";
    }
    return "unknown";
}

[[nodiscard]] const char* denseRKAccessName(DenseRKAccess access) {
    switch (access) {
        case DenseRKAccess::Aligned: return "aligned";
        case DenseRKAccess::Rotated: return "rotated";
        case DenseRKAccess::FlatRows: return "flat_rows";
    }
    return "unknown";
}

[[nodiscard]] DenseRKOccupancyInfo queryProductionRKOccupancy(const DenseRKFamilyPhysicalPlan& plan,
                                                               void* user_context) {
    CubKernelOccupancyInfo exact;
    switch (plan.implementation) {
        case CubReductionInternal::DenseRKProductionImplementation::KParallelPass: {
            const KParallelTiledStagePlan physical{
                plan.progress == CubReductionInternal::DenseRKProgress::Complete ? TiledRKStageTopology::Complete
                                                           : TiledRKStageTopology::Staged,
                plan.packet_bytes,
                plan.block_threads,
                plan.shards_per_output};
            exact = queryKParallelTiledStageOccupancy(plan.input_dtype, physical);
            break;
        }
        case CubReductionInternal::DenseRKProductionImplementation::RCooperativeAlignedStaged:
            exact = queryCooperativeShardedTiledFirstStageOccupancy(plan.input_dtype, plan.rows_per_shard);
            break;
        case CubReductionInternal::DenseRKProductionImplementation::RCooperativeRotatedStaged:
            exact = queryAwkwardAlignmentRotatedShardedFirstStageOccupancy(plan.input_dtype, plan.rows_per_shard);
            break;
        case CubReductionInternal::DenseRKProductionImplementation::RCooperativeFlatRowsStaged:
            exact = queryNarrowLowPrecisionFlatRCooperativeFirstStageOccupancy(plan.input_dtype, plan.rows_per_shard);
            break;
        case CubReductionInternal::DenseRKProductionImplementation::RCooperativeAlignedComplete: {
            const auto* fallback = static_cast<const ProductionRKOccupancyContext*>(user_context);
            if (fallback == nullptr || plan.block_threads == 0 || fallback->warp_size == 0) {
                throw std::logic_error("Packet-adaptive production occupancy is missing device geometry.");
            }
            const uint32_t thread_limited = fallback->max_threads_per_sm / plan.block_threads;
            const uint32_t blocks = std::min(fallback->max_blocks_per_sm, thread_limited);
            const uint32_t warps_per_block = plan.block_threads / fallback->warp_size;
            if (blocks == 0 || warps_per_block == 0) {
                throw std::logic_error("Packet-adaptive production occupancy found no resident blocks.");
            }
            return DenseRKOccupancyInfo{blocks, blocks * warps_per_block};
        }
        default:
            throw std::logic_error("Production DenseRK occupancy received a non-family implementation.");
    }
    if (exact.max_active_blocks_per_sm <= 0 || exact.max_active_warps_per_sm <= 0) {
        throw std::logic_error("Production DenseRK exact occupancy returned zero residency.");
    }
    return DenseRKOccupancyInfo{static_cast<uint32_t>(exact.max_active_blocks_per_sm),
                                static_cast<uint32_t>(exact.max_active_warps_per_sm)};
}

[[nodiscard]] DenseReductionPlan planProductionDenseValue(CubReductionOp op,
                                                           const std::vector<uint64_t>& input_dimensions,
                                                           const std::vector<uint32_t>& axes,
                                                           DataType original_input_dtype,
                                                           DataType output_dtype,
                                                           const Stream& stream,
                                                           std::string* strategy_chain = nullptr) {
    DenseReductionOutputSpec output_spec;
    output_spec.produce_value = true;
    output_spec.value_dtype = output_dtype;
    output_spec.produce_index = false;
    const DenseReductionProblem problem = makeInitialDenseReductionProblem(
        input_dimensions, axes, DenseReductionAggregateKind::Value, output_spec);

    cudaDeviceProp properties{};
    const cudaError_t prop_status = cudaGetDeviceProperties(&properties, static_cast<int>(stream.getGpuNum()));
    if (prop_status != cudaSuccess || properties.multiProcessorCount <= 0 || properties.warpSize <= 0
        || properties.maxThreadsPerMultiProcessor <= 0 || properties.maxBlocksPerMultiProcessor <= 0) {
        throw std::runtime_error(std::string("Failed to query CUDA device geometry for dense VALUE planning: ")
                                 + cudaGetErrorString(prop_status));
    }
    ProductionRKOccupancyContext occupancy_context{
        static_cast<uint32_t>(properties.maxThreadsPerMultiProcessor),
        static_cast<uint32_t>(properties.maxBlocksPerMultiProcessor),
        static_cast<uint32_t>(properties.warpSize)};
    DenseReductionPlannerContext context;
    context.rk.original_input_dtype = original_input_dtype;
    context.rk.value_operation = denseRKValueOperation(op);
    context.rk.multiprocessors = static_cast<uint32_t>(properties.multiProcessorCount);
    context.rk.warp_size = static_cast<uint32_t>(properties.warpSize);
    context.rk.max_threads_per_sm = static_cast<uint32_t>(properties.maxThreadsPerMultiProcessor);
    context.rk.max_blocks_per_sm = static_cast<uint32_t>(properties.maxBlocksPerMultiProcessor);
    context.rk.occupancy_query = queryProductionRKOccupancy;
    context.rk.occupancy_query_context = &occupancy_context;
    // Production needs total coverage while packet-incompatible FP8 and a small shallow/awkward low-precision envelope
    // remain outside the calibrated modern inventory. DENSE-FP8-VALUE-1 admits naturally packetable FP8 through
    // 16/8/4/2-byte KParallel specializations; odd-K shapes that cannot form even the 2-byte packet remain on the
    // generic TiledFixedSegment adapter, not any deleted historical direct/full-row/row-split kernel. The strict RK
    // census keeps this flag false.
    context.rk.allow_generic_rk_fallback = true;

    DenseReductionPlan plan = planDenseReduction(problem, context);
    if (strategy_chain != nullptr) {
        strategy_chain->clear();
        for (const DenseReductionPlanStep& step : plan.steps) {
            if (!strategy_chain->empty()) strategy_chain->append(">");
            if (const auto rk = std::dynamic_pointer_cast<const DenseRKFamilyPhysicalPlan>(
                    step.selected_candidate.physical_plan); rk) {
                strategy_chain->append(denseRKStrategyName(rk->strategy));
                strategy_chain->append(":");
                strategy_chain->append(denseRKAccessName(rk->access));
                strategy_chain->append(rk->progress == DenseRKProgress::Complete ? ":complete" : ":staged");
            } else {
                switch (step.selected_candidate.family) {
                    case DenseReducerFamily::R: strategy_chain->append("r:direct"); break;
                    case DenseReducerFamily::KR: strategy_chain->append("kr:direct"); break;
                    case DenseReducerFamily::RK: strategy_chain->append("rk:generic"); break;
                }
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

[[nodiscard]] bool isSupportedFloatingStorageDType(DataType dtype) {
    switch (dtype) {
#if THOR_CUB_ENABLE_FP8_TYPES
        case DataType::FP8_E4M3:
        case DataType::FP8_E5M2:
#endif
        case DataType::FP16:
        case DataType::BF16:
        case DataType::FP32:
#if THOR_CUB_ENABLE_64BIT_TYPES
        case DataType::FP64:
#endif
            return true;
        default:
            return false;
    }
}

void requireSupportedFloatingStorageDType(DataType dtype, const char* role) {
    if (!isSupportedFloatingStorageDType(dtype)) {
        throw std::invalid_argument(std::string("CUB tensor reduction does not support ") + role + " dtype " + dtypeName(dtype) +
                                    ". Supported storage dtypes follow Thor's THOR_CUB_ENABLE_FP8_TYPES and "
                                    "THOR_CUB_ENABLE_64BIT_TYPES build policy; TF32 is not a storage dtype.");
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

void requireCompatibleStream(const Tensor& input, const Stream& stream) {
    if (!stream.isInitialized()) {
        throw std::invalid_argument("CUB tensor reduction requires an initialized stream.");
    }
    if (input.getPlacement().getDeviceNum() != stream.getGpuNum()) {
        throw std::invalid_argument("CUB tensor reduction stream must belong to the input tensor's GPU.");
    }
}

void requireGpuTensorView(const Tensor& tensor, const char* name) {
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

struct DenseReducedRunSpan {
    uint32_t first_axis = 0;
    uint32_t last_axis = 0;
    uint32_t reduced_run_ordinal = 0;
    std::optional<uint32_t> dense_run_index = std::nullopt;
    uint64_t reduction_extent = 1;
    uint64_t domain_stride = 1;
};

struct DenseDirectStageGeometry {
    CubReductionPath path = CubReductionPath::TiledFixedSegment;
    uint64_t input_elements = 1;
    uint64_t output_elements = 1;
    uint64_t reduction_size = 1;
    uint64_t outer_size = 1;
    uint64_t inner_size = 1;
};

struct DenseArgCompositionCostContext {
    DataType input_dtype = DataType::FP32;
    DataType carried_index_dtype = DataType::UINT32;
    bool produce_value = true;
    DataType value_output_dtype = DataType::FP32;
    bool produce_index = true;
    DataType index_output_dtype = DataType::UINT32;
    uint64_t l2_cache_bytes = 0;
};

enum class DenseArgPlannerStageClass : uint8_t {
    DeviceTransform,
    Contiguous,
    TiledNarrow,
    TiledSimpleAlignedFullTiles,
    TiledAwkwardHeadBulkTail,
};

enum class DensePlannerStageClass : uint8_t {
    DeviceTransform,
    ContiguousCubFixedSegment,
    VectorizedFullRow,
    AsyncNarrowFullRow,
    AsyncFullRow,
    VectorizedBlockSharded,
    AlignmentSafeShapedBlockSharded,
    DirectComponentTiled,
};

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

[[nodiscard]] bool usesCubFixedSegmentSize(CubReductionPath path) {
    // DeviceSegmentedReduce's fixed-size overload takes an `int segment_size`. Thor-owned TiledFixedSegment kernels
    // use uint64_t geometry throughout and therefore do not inherit that CUB API limit.
    return path == CubReductionPath::ContiguousFixedSegment;
}

[[nodiscard]] DenseDirectStageGeometry analyzeDirectDenseStageGeometry(
    const std::vector<uint64_t>& input_dimensions,
    const std::vector<uint32_t>& reduction_axes) {
    THOR_THROW_IF_FALSE(!reduction_axes.empty());
    DenseDirectStageGeometry stage;

    size_t reduced_cursor = 0;
    for (uint32_t axis = 0; axis < input_dimensions.size(); ++axis) {
        const bool reduced = reduced_cursor < reduction_axes.size() && reduction_axes[reduced_cursor] == axis;
        if (reduced) {
            THOR_THROW_IF_FALSE(stage.reduction_size
                                <= std::numeric_limits<uint64_t>::max() / input_dimensions[axis]);
            stage.reduction_size *= input_dimensions[axis];
            ++reduced_cursor;
        } else {
            THOR_THROW_IF_FALSE(stage.output_elements
                                <= std::numeric_limits<uint64_t>::max() / input_dimensions[axis]);
            stage.output_elements *= input_dimensions[axis];
        }
        if (axis > reduction_axes.back()) {
            THOR_THROW_IF_FALSE(stage.inner_size
                                <= std::numeric_limits<uint64_t>::max() / input_dimensions[axis]);
            stage.inner_size *= input_dimensions[axis];
        }
    }
    THOR_THROW_IF_FALSE(reduced_cursor == reduction_axes.size());
    THOR_THROW_IF_FALSE(stage.output_elements
                        <= std::numeric_limits<uint64_t>::max() / stage.reduction_size);
    stage.input_elements = stage.output_elements * stage.reduction_size;
    THOR_THROW_IF_FALSE(stage.reduction_size <= std::numeric_limits<uint64_t>::max() / stage.inner_size);
    const uint64_t reduction_inner = stage.reduction_size * stage.inner_size;
    THOR_THROW_IF_FALSE(reduction_inner != 0);
    THOR_THROW_IF_FALSE(stage.input_elements % reduction_inner == 0);
    stage.outer_size = stage.input_elements / reduction_inner;

    if (stage.output_elements == 1) {
        stage.path = CubReductionPath::DeviceTransformReduce;
    } else if (stage.inner_size == 1) {
        stage.path = CubReductionPath::ContiguousFixedSegment;
    } else {
        stage.path = CubReductionPath::TiledFixedSegment;
    }
    return stage;
}

[[nodiscard]] std::vector<DenseReducedRunSpan> denseReducedRunSpans(
    const CubReductionGeometry& geometry,
    const std::vector<uint64_t>& input_dimensions) {
    THOR_THROW_IF_FALSE(geometry.dense_run_geometry.has_value());

    std::vector<bool> is_reduced(input_dimensions.size(), false);
    for (uint32_t axis : geometry.axes) {
        is_reduced[axis] = true;
    }

    std::vector<DenseReducedRunSpan> spans;
    bool previous_non_singleton_was_reduced = false;
    for (uint32_t axis = 0; axis < input_dimensions.size(); ++axis) {
        if (input_dimensions[axis] == 1) {
            continue;
        }
        if (is_reduced[axis]) {
            if (!previous_non_singleton_was_reduced) {
                spans.push_back(DenseReducedRunSpan{.first_axis = axis, .last_axis = axis});
            } else {
                spans.back().last_axis = axis;
            }
        }
        previous_non_singleton_was_reduced = is_reduced[axis];
    }

    size_t reduced_run_ordinal = 0;
    for (size_t dense_run_index = 0; dense_run_index < geometry.dense_run_geometry->runs.size(); ++dense_run_index) {
        const CubReductionDenseRun& run = geometry.dense_run_geometry->runs[dense_run_index];
        if (run.kind != CubReductionDenseRunKind::Reduced) {
            continue;
        }
        THOR_THROW_IF_FALSE(reduced_run_ordinal < spans.size());
        DenseReducedRunSpan& span = spans[reduced_run_ordinal];
        THOR_THROW_IF_FALSE(reduced_run_ordinal <= std::numeric_limits<uint32_t>::max());
        THOR_THROW_IF_FALSE(dense_run_index <= std::numeric_limits<uint32_t>::max());
        span.reduced_run_ordinal = static_cast<uint32_t>(reduced_run_ordinal);
        span.dense_run_index = static_cast<uint32_t>(dense_run_index);
        span.reduction_extent = run.extent;
        span.domain_stride = run.domain_stride;
        ++reduced_run_ordinal;
    }

    // Every reduced axis being singleton is the only case without a physical reduced run. One extent-1 stage is kept
    // so value composition can still perform conversion/finalization and future arg composition has an explicit root.
    if (spans.empty()) {
        THOR_THROW_IF_FALSE(!geometry.axes.empty());
        spans.push_back(DenseReducedRunSpan{.first_axis = geometry.axes.front(),
                                            .last_axis = geometry.axes.front(),
                                            .reduced_run_ordinal = 0,
                                            .dense_run_index = std::nullopt,
                                            .reduction_extent = 1,
                                            .domain_stride = 1});
    } else {
        THOR_THROW_IF_FALSE(reduced_run_ordinal == spans.size());
        THOR_THROW_IF_FALSE(reduced_run_ordinal == geometry.dense_run_geometry->reduced_run_count);
    }
    return spans;
}

[[nodiscard]] std::vector<uint32_t> denseReducedRunAxes(const DenseReducedRunSpan& span) {
    std::vector<uint32_t> axes;
    axes.reserve(static_cast<size_t>(span.last_axis - span.first_axis) + 1U);
    for (uint32_t axis = span.first_axis; axis <= span.last_axis; ++axis) {
        axes.push_back(axis);
    }
    return axes;
}

[[nodiscard]] std::vector<uint64_t> denseReductionStateDimensions(
    const std::vector<uint64_t>& input_dimensions,
    const std::vector<DenseReducedRunSpan>& spans,
    size_t state_left,
    size_t state_right) {
    std::vector<uint64_t> dimensions = input_dimensions;
    for (size_t run_index = 0; run_index < spans.size(); ++run_index) {
        if (run_index >= state_left && run_index <= state_right) {
            continue;
        }
        for (uint32_t axis = spans[run_index].first_axis; axis <= spans[run_index].last_axis; ++axis) {
            dimensions[axis] = 1;
        }
    }
    return dimensions;
}

[[nodiscard]] std::optional<CubReductionDenseCompositionStage> makeDenseCompositionStagePlan(
    const std::vector<uint64_t>& current_dimensions,
    const DenseReducedRunSpan& span,
    CubReductionDenseCompositionStageRole role) {
    CubReductionDenseCompositionStage stage;
    stage.reduced_run_ordinal = span.reduced_run_ordinal;
    stage.dense_run_index = span.dense_run_index;
    stage.reduction_extent = span.reduction_extent;
    stage.domain_stride = span.domain_stride;
    stage.role = role;
    stage.reduction_axes = denseReducedRunAxes(span);
    stage.input_dimensions = current_dimensions;

    const DenseDirectStageGeometry direct_geometry =
        analyzeDirectDenseStageGeometry(current_dimensions, stage.reduction_axes);
    stage.expected_path = direct_geometry.path;
    stage.input_elements = direct_geometry.input_elements;
    stage.output_elements = direct_geometry.output_elements;
    stage.stage_reduction_size = direct_geometry.reduction_size;
    stage.outer_size = direct_geometry.outer_size;
    stage.inner_size = direct_geometry.inner_size;
    THOR_THROW_IF_FALSE(isDirectDenseReductionPath(stage.expected_path));
    THOR_THROW_IF_FALSE(stage.stage_reduction_size == stage.reduction_extent);

    if (usesCubFixedSegmentSize(stage.expected_path)
        && direct_geometry.reduction_size > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        return std::nullopt;
    }

    stage.output_dimensions = current_dimensions;
    for (uint32_t axis : stage.reduction_axes) {
        stage.output_dimensions[axis] = 1;
    }
    return stage;
}

[[nodiscard]] DenseDirectStageGeometry directStageGeometryFromPlan(
    const CubReductionDenseCompositionStage& stage) {
    return DenseDirectStageGeometry{.path = stage.expected_path,
                                    .input_elements = stage.input_elements,
                                    .output_elements = stage.output_elements,
                                    .reduction_size = stage.stage_reduction_size,
                                    .outer_size = stage.outer_size,
                                    .inner_size = stage.inner_size};
}

[[nodiscard]] uint64_t ceilDivideDensePlanner(uint64_t numerator, uint64_t denominator) {
    THOR_THROW_IF_FALSE(denominator != 0);
    return numerator / denominator + static_cast<uint64_t>(numerator % denominator != 0);
}

[[nodiscard]] uint64_t densePlannerGroupedFullRowWarps(uint64_t inner_size, uint64_t input_element_bytes) {
    using namespace CubReductionTiledPolicy;
    const uint64_t component_warps = ceilDivideDensePlanner(inner_size, FULL_ROW_COMPONENTS_PER_WARP);
    THOR_THROW_IF_FALSE(inner_size <= std::numeric_limits<uint64_t>::max() / input_element_bytes);
    const uint64_t row_bytes = inner_size * input_element_bytes;
    const uint64_t stage_warps = ceilDivideDensePlanner(row_bytes, ASYNC_STAGE_BYTES_PER_WARP);
    const uint64_t required_warps = std::max(component_warps, stage_warps);

    uint64_t warps = 1;
    while (warps < FULL_ROW_MAX_WARPS_PER_OUTPUT && warps < required_warps) {
        warps *= 2;
    }
    return warps >= required_warps ? warps : 0;
}

[[nodiscard]] uint64_t densePlannerDirectComponentActiveWarps(const DenseDirectStageGeometry& stage) {
    using namespace CubReductionTiledPolicy;
    const uint64_t component_tiles = ceilDivideDensePlanner(stage.inner_size, WARP_THREADS);
    THOR_THROW_IF_FALSE(stage.outer_size <= std::numeric_limits<uint64_t>::max() / component_tiles);
    const uint64_t output_tiles = stage.outer_size * component_tiles;
    const uint64_t useful_warps_from_rows = stage.reduction_size;
    const uint64_t desired_warps_per_tile =
        ceilDivideDensePlanner(TARGET_ACTIVE_WARPS, std::max<uint64_t>(output_tiles, 1));

    uint64_t warps_per_tile = 1;
    while (warps_per_tile < WARPS_PER_BLOCK && warps_per_tile < desired_warps_per_tile
           && warps_per_tile < useful_warps_from_rows) {
        warps_per_tile *= 2;
    }
    while (warps_per_tile > 1 && warps_per_tile > useful_warps_from_rows) {
        warps_per_tile /= 2;
    }
    if (output_tiles > std::numeric_limits<uint64_t>::max() / warps_per_tile) {
        return TARGET_ACTIVE_WARPS;
    }
    return std::min<uint64_t>(TARGET_ACTIVE_WARPS, output_tiles * warps_per_tile);
}

[[nodiscard]] DensePlannerStageClass classifyDensePlannerStage(const DenseDirectStageGeometry& stage,
                                                                DataType input_dtype) {
    using namespace CubReductionTiledPolicy;
    if (stage.path == CubReductionPath::DeviceTransformReduce) {
        return DensePlannerStageClass::DeviceTransform;
    }
    if (stage.path == CubReductionPath::ContiguousFixedSegment) {
        return DensePlannerStageClass::ContiguousCubFixedSegment;
    }

    THOR_THROW_IF_FALSE(stage.path == CubReductionPath::TiledFixedSegment);
    const uint64_t input_element_bytes =
        static_cast<uint64_t>(TensorDescriptor::getElementSizeInBytes(input_dtype));

    if (stage.inner_size == 32 || stage.inner_size == 64 || stage.inner_size == 128
        || stage.inner_size == 256 || stage.inner_size == FULL_ROW_COMPONENTS_PER_WARP
        || stage.inner_size == 1024 || stage.inner_size == 2048
        || stage.inner_size == FULL_ROW_GROUP_MAX_INNER_SIZE) {
        return DensePlannerStageClass::VectorizedFullRow;
    }
    if (stage.inner_size <= 32) {
        return DensePlannerStageClass::AsyncNarrowFullRow;
    }
    if (stage.inner_size < FULL_ROW_COMPONENTS_PER_WARP) {
        const uint64_t async_stage_capacity = ASYNC_STAGE_BYTES_PER_WARP / input_element_bytes;
        return stage.inner_size <= async_stage_capacity ? DensePlannerStageClass::AsyncFullRow
                                                        : DensePlannerStageClass::DirectComponentTiled;
    }
    if (stage.inner_size < FULL_ROW_GROUP_MAX_INNER_SIZE) {
        return densePlannerGroupedFullRowWarps(stage.inner_size, input_element_bytes) != 0
                   ? DensePlannerStageClass::AsyncFullRow
                   : DensePlannerStageClass::DirectComponentTiled;
    }
    if (stage.inner_size > FULL_ROW_GROUP_MAX_INNER_SIZE) {
        return stage.inner_size % FULL_ROW_COMPONENTS_PER_BLOCK == 0
                   ? DensePlannerStageClass::VectorizedBlockSharded
                   : DensePlannerStageClass::AlignmentSafeShapedBlockSharded;
    }
    return DensePlannerStageClass::DirectComponentTiled;
}

[[nodiscard]] long double densePlannerExecutionPenalty(const DenseDirectStageGeometry& stage,
                                                        DataType input_dtype,
                                                        DensePlannerStageClass stage_class) {
    const uint64_t input_element_bytes =
        static_cast<uint64_t>(TensorDescriptor::getElementSizeInBytes(input_dtype));
    const bool low_precision_input = input_element_bytes <= 2;

    // These are deliberately broad execution-regime weights, not per-shape timing constants. Release calibration
    // shows three stable effects that the first traffic-only planner missed:
    //   * CUB fixed-size segmented reductions have a pronounced short-segment trough, especially for FP16/BF16;
    //   * the alignment-safe shaped-block fallback carries more instruction/control work per logical byte;
    //   * very short async-narrow reductions are not pure streaming kernels.
    // Keeping the buckets coarse lets the interval DP account for those families without learning individual census
    // shapes or embedding measured microseconds from one GPU.
    switch (stage_class) {
        case DensePlannerStageClass::ContiguousCubFixedSegment:
            if (stage.reduction_size >= 17 && stage.reduction_size <= 31) {
                return low_precision_input ? 3.25L : 1.50L;
            }
            if (stage.reduction_size >= 32 && stage.reduction_size <= 127) {
                return low_precision_input ? 1.75L : 1.00L;
            }
            return 1.00L;
        case DensePlannerStageClass::AlignmentSafeShapedBlockSharded:
            return stage.reduction_size >= 256 ? 2.45L : 1.40L;
        case DensePlannerStageClass::AsyncNarrowFullRow:
            if (stage.reduction_size <= 31) {
                return low_precision_input ? 1.75L : 1.50L;
            }
            return 1.00L;
        default:
            return 1.00L;
    }
}

/**
 * Estimate the useful warp concurrency exposed by the direct reducer that will execute this stage.
 *
 * This deliberately mirrors only the broad ownership policy of the production kernels. It does not attempt to predict
 * instruction-level throughput: the dense planner should choose between fundamentally different traffic/parallelism
 * shapes, not encode one GPU's measured microseconds. The 1024-warp target is the same architecture-level target used
 * by the direct Tiled fallback when it decides how many warps should cooperate on one output tile.
 */
[[nodiscard]] uint64_t estimateDenseDirectStageActiveWarps(const DenseDirectStageGeometry& stage,
                                                            DataType input_dtype) {
    using namespace CubReductionTiledPolicy;
    if (stage.path == CubReductionPath::DeviceTransformReduce) {
        return std::min<uint64_t>(
            TARGET_ACTIVE_WARPS,
            std::max<uint64_t>(1, ceilDivideDensePlanner(stage.input_elements, WARP_THREADS)));
    }

    if (stage.path == CubReductionPath::ContiguousFixedSegment) {
        const uint64_t warps_per_segment =
            std::min<uint64_t>(WARPS_PER_BLOCK, ceilDivideDensePlanner(stage.reduction_size, WARP_THREADS));
        if (stage.output_elements > std::numeric_limits<uint64_t>::max() / warps_per_segment) {
            return TARGET_ACTIVE_WARPS;
        }
        return std::min<uint64_t>(
            TARGET_ACTIVE_WARPS, std::max<uint64_t>(1, stage.output_elements * warps_per_segment));
    }

    THOR_THROW_IF_FALSE(stage.path == CubReductionPath::TiledFixedSegment);

    const uint64_t input_element_bytes =
        static_cast<uint64_t>(TensorDescriptor::getElementSizeInBytes(input_dtype));

    uint64_t warps_per_output = 1;
    if (stage.inner_size == 1024) {
        warps_per_output = 2;
    } else if (stage.inner_size == 2048) {
        warps_per_output = 4;
    } else if (stage.inner_size == FULL_ROW_GROUP_MAX_INNER_SIZE) {
        warps_per_output = FULL_ROW_MAX_WARPS_PER_OUTPUT;
    } else if (stage.inner_size > FULL_ROW_COMPONENTS_PER_WARP
               && stage.inner_size < FULL_ROW_GROUP_MAX_INNER_SIZE) {
        warps_per_output = densePlannerGroupedFullRowWarps(stage.inner_size, input_element_bytes);
        if (warps_per_output == 0) {
            return densePlannerDirectComponentActiveWarps(stage);
        }
    } else if (stage.inner_size > FULL_ROW_GROUP_MAX_INNER_SIZE) {
        const uint64_t shards = ceilDivideDensePlanner(stage.inner_size, FULL_ROW_COMPONENTS_PER_BLOCK);
        if (stage.outer_size > std::numeric_limits<uint64_t>::max() / shards
            || stage.outer_size * shards
                   > std::numeric_limits<uint64_t>::max() / WARPS_PER_BLOCK) {
            return TARGET_ACTIVE_WARPS;
        }
        return std::min<uint64_t>(TARGET_ACTIVE_WARPS, stage.outer_size * shards * WARPS_PER_BLOCK);
    } else if (stage.inner_size > 256 && stage.inner_size < FULL_ROW_COMPONENTS_PER_WARP) {
        const uint64_t async_stage_capacity = ASYNC_STAGE_BYTES_PER_WARP / input_element_bytes;
        if (stage.inner_size > async_stage_capacity) {
            return densePlannerDirectComponentActiveWarps(stage);
        }
    }

    if (stage.outer_size > std::numeric_limits<uint64_t>::max() / warps_per_output) {
        return TARGET_ACTIVE_WARPS;
    }
    return std::min<uint64_t>(
        TARGET_ACTIVE_WARPS, std::max<uint64_t>(1, stage.outer_size * warps_per_output));
}

/**
 * Broad dense-stage execution model used only to choose a stamped pass order.
 *
 * The model charges logical input/output traffic, then accounts for useful-warp concurrency and the broad execution
 * regime of the direct reducer that would execute the stage. A non-root intermediate that fits entirely in L2 receives
 * an 8x input-read discount. These are intentionally coarse architecture-level weights: the planner distinguishes CUB
 * short fixed segments, shaped-block Tiled work, and short async-narrow work, but never keys on a benchmark case or a
 * measured microsecond value.
 */
[[nodiscard]] long double estimateDenseValueStageCost(const DenseDirectStageGeometry& stage,
                                                       DataType input_dtype,
                                                       DataType output_dtype,
                                                       bool intermediate_is_hot,
                                                       uint64_t l2_cache_bytes) {
    constexpr long double L2_RESIDENT_INPUT_DISCOUNT = 8.0L;
    const uint64_t input_element_bytes =
        static_cast<uint64_t>(TensorDescriptor::getElementSizeInBytes(input_dtype));
    const uint64_t output_element_bytes =
        static_cast<uint64_t>(TensorDescriptor::getElementSizeInBytes(output_dtype));
    const long double input_bytes =
        static_cast<long double>(stage.input_elements) * static_cast<long double>(input_element_bytes);
    const long double output_bytes =
        static_cast<long double>(stage.output_elements) * static_cast<long double>(output_element_bytes);

    long double charged_input_bytes = input_bytes;
    if (intermediate_is_hot && l2_cache_bytes != 0
        && input_bytes <= static_cast<long double>(l2_cache_bytes)) {
        charged_input_bytes /= L2_RESIDENT_INPUT_DISCOUNT;
    }

    const DensePlannerStageClass stage_class = classifyDensePlannerStage(stage, input_dtype);
    const uint64_t active_warps = estimateDenseDirectStageActiveWarps(stage, input_dtype);
    const long double active_warp_ratio =
        static_cast<long double>(CubReductionTiledPolicy::TARGET_ACTIVE_WARPS)
        / static_cast<long double>(std::max<uint64_t>(active_warps, 1));
    const long double parallelism_penalty =
        std::max<long double>(1.0L, std::pow(active_warp_ratio, 1.15L));
    const long double execution_penalty = densePlannerExecutionPenalty(stage, input_dtype, stage_class);
    long double estimated_cost =
        (charged_input_bytes + output_bytes) * parallelism_penalty * execution_penalty;

    // Async pipelines have a real producer/consumer setup floor that dominates tiny low-concurrency stages even when
    // their logical byte count is minuscule. Express that floor in the same abstract byte-cost units as the traffic
    // model and fade it out as useful concurrency approaches the normal saturation target.
    if ((stage_class == DensePlannerStageClass::AsyncNarrowFullRow
         || stage_class == DensePlannerStageClass::AsyncFullRow)
        && active_warps < CubReductionTiledPolicy::TARGET_ACTIVE_WARPS) {
        constexpr long double ASYNC_LOW_CONCURRENCY_EQUIVALENT_BYTES = 8'000'000.0L;
        const long double active_fraction =
            static_cast<long double>(active_warps)
            / static_cast<long double>(CubReductionTiledPolicy::TARGET_ACTIVE_WARPS);
        estimated_cost = std::max(estimated_cost,
                                  ASYNC_LOW_CONCURRENCY_EQUIVALENT_BYTES * (1.0L - active_fraction));
    }
    return estimated_cost;
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

[[nodiscard]] bool cooperativeShardedTiledNaturalGeometry(const CubReductionGeometry& geometry) {
    return geometry.path == CubReductionPath::TiledFixedSegment
           && !geometry.payload_transpose_tiled_geometry.has_value()
           && !geometry.pitched_tiled_geometry.has_value()
           && !geometry.tiled_output_permuted
           && !geometry.tiled_output_shared_transpose
           && geometry.tiled_output_outer_stride == geometry.inner_size
           && geometry.tiled_output_inner_stride == 1;
}

/**
 * Production gate for the benchmarked alignment-rotated replacement of awkward retained-width reducers.
 *
 * The wide 2-D census proves that packet-misaligned K >4096 needs R sharding from R=512 onward, including large outer
 * counts. For that domain a simple shard-depth progression keeps at least two R shards: rows256 below R=1024, rows512
 * below R=2048, and rows1024 thereafter. The earlier wide/sub-4096 censuses retain the deep low-output rules below the
 * wide boundary. The sub-512 census extends the same architecture to packet-misaligned 257..511 widths, and the
 * sub-256 census extends it again to packet-misaligned 65..255 widths once the reduction is at least 32768 deep, and
 * the final sub-64 census extends that same geometry rule to packet-misaligned 33..63 widths.
 *
 * Below 512 retained components, shard depth follows launch geometry rather than a K-by-K lookup. For 257..511 the
 * previous rows1024/rows256/rows512 rule remains unchanged. For 33..255, low-precision input uses rows256 while a single
 * logical component tile is less than half occupied, rows512 once that tile is at least half occupied, and rows1024
 * once a second logical tile is required; FP32 uses the corresponding 1/2/3-tile progression. If that preferred shard
 * depth cannot expose the benchmarked SM-wave target, the selector steps down to a more parallel specialization. The
 * resulting rule retained about 99.8% of the measured per-case best bandwidth in the sub-256 census while avoiding
 * width-specific tuning. The same ownership, packet, and shared-memory geometry is used for Product/Min/Max; only the
 * associative FP32 combine and its identity differ.
 */
[[nodiscard]] long double estimateCooperativeShardedRemainderCost(uint64_t outer_size,
                                                                  uint64_t shards_per_output,
                                                                  uint64_t inner_size,
                                                                  DataType output_dtype,
                                                                  uint64_t l2_cache_bytes) {
    using namespace CubReductionTiledPolicy;
    DenseDirectStageGeometry remainder_geometry;
    remainder_geometry.path = CubReductionPath::TiledFixedSegment;
    remainder_geometry.input_elements = outer_size * shards_per_output * inner_size;
    remainder_geometry.output_elements = outer_size * inner_size;
    remainder_geometry.reduction_size = shards_per_output;
    remainder_geometry.outer_size = outer_size;
    remainder_geometry.inner_size = inner_size;

    // The ordinary second stage consumes FP32 partials. When it falls inside the packet-adaptive cooperative gate,
    // model the reducer that will actually execute rather than the older full-row ownership approximation used by
    // the general dense-composition planner. This is essential for narrow retained widths: a single output tile can
    // own up to 32 cooperating warps even though the legacy model would count only one useful warp.
    if (shards_per_output >= PACKET_COOPERATIVE_MIN_REDUCTION_SIZE
        && inner_size >= PACKET_COOPERATIVE_MIN_INNER_SIZE
        && inner_size <= PACKET_COOPERATIVE_MAX_INNER_SIZE) {
        constexpr uint64_t fp32_bytes = sizeof(float);
        constexpr uint64_t components_16 = WARP_THREADS * (16 / fp32_bytes);
        constexpr uint64_t components_8 = WARP_THREADS * (8 / fp32_bytes);
        const uint64_t row_bytes = inner_size * fp32_bytes;

        uint64_t packet_bytes = 16;
        if (inner_size <= components_16 / 2 && row_bytes % 8 == 0) {
            packet_bytes = 8;
            if (inner_size <= components_8 / 2 && row_bytes % 4 == 0) {
                packet_bytes = 4;
            }
        }
        const uint64_t items_per_lane = packet_bytes / fp32_bytes;
        const uint64_t components_per_tile = WARP_THREADS * items_per_lane;
        const uint64_t component_tiles = ceilDivideDensePlanner(inner_size, components_per_tile);
        if (outer_size <= std::numeric_limits<uint64_t>::max() / component_tiles) {
            const uint64_t output_tiles = std::max<uint64_t>(1, outer_size * component_tiles);
            const uint64_t adaptive_target_warps = TARGET_ACTIVE_WARPS * DENSE_RK_TARGET_SM_WAVES;
            const uint64_t desired_warps = ceilDivideDensePlanner(adaptive_target_warps, output_tiles);

            uint64_t warps_per_tile = 1;
            while (warps_per_tile < 32 && warps_per_tile < desired_warps
                   && warps_per_tile < shards_per_output) {
                warps_per_tile <<= 1;
            }
            while (warps_per_tile < 32 && shards_per_output > warps_per_tile
                   && (row_bytes * warps_per_tile) % packet_bytes != 0) {
                warps_per_tile <<= 1;
            }

            if (warps_per_tile >= PACKET_COOPERATIVE_MIN_WARPS_PER_TILE) {
                const uint64_t active_warps = std::min<uint64_t>(
                    TARGET_ACTIVE_WARPS,
                    output_tiles > std::numeric_limits<uint64_t>::max() / warps_per_tile
                        ? TARGET_ACTIVE_WARPS
                        : output_tiles * warps_per_tile);
                const long double input_bytes =
                    static_cast<long double>(remainder_geometry.input_elements) * fp32_bytes;
                const long double output_bytes =
                    static_cast<long double>(remainder_geometry.output_elements)
                    * static_cast<long double>(TensorDescriptor::getElementSizeInBytes(output_dtype));
                const long double charged_input_bytes =
                    l2_cache_bytes != 0 && input_bytes <= static_cast<long double>(l2_cache_bytes)
                        ? input_bytes / 8.0L
                        : input_bytes;
                const long double active_warp_ratio =
                    static_cast<long double>(TARGET_ACTIVE_WARPS)
                    / static_cast<long double>(std::max<uint64_t>(active_warps, 1));
                const long double parallelism_penalty =
                    std::max<long double>(1.0L, std::pow(active_warp_ratio, 1.15L));
                return (charged_input_bytes + output_bytes) * parallelism_penalty;
            }
        }
    }

    return estimateDenseValueStageCost(
        remainder_geometry, DataType::FP32, output_dtype, true, l2_cache_bytes);
}

/**
 * Select the benchmark-proven y8/16-byte cooperative shard size.
 *
 * The original production gate covered the very-low-output/deep <=4096 retained-width regime. The wide 2-D census
 * subsequently showed the same ownership requirement for packet-aligned K >4096 once R reaches 1024: the old direct
 * block-sharded family serially walks too much R per CTA while this staged reducer remains near the memory-bandwidth
 * regime by sharding both R and K. Keep R=512 wide aligned work on the direct path because that control remains healthy.
 *
 * Within the admitted regime, candidate shard sizes are costed as complete two-stage plans. SUM may use all four
 * measured depths; Product/Min/Max stop at rows512 because the all-operation census showed rows1024 can reintroduce a
 * long dependency chain at R>=2048. The first-stage term charges input + FP32-partial traffic and distance from three
 * SM waves of independent CTAs; the remainder term models the ordinary FP32 [outer,shards,inner] reducer. No measured
 * microseconds or benchmark-case names are embedded here.
 */

[[nodiscard]] uint64_t denseArgPlannerElementBytes(DataType dtype) {
    return static_cast<uint64_t>(TensorDescriptor::getElementSizeInBytes(dtype));
}

[[nodiscard]] uint64_t denseArgPlannerItemsPerLane(DataType value_dtype) {
    const uint64_t element_bytes = denseArgPlannerElementBytes(value_dtype);
    if (element_bytes == 0 || 16 % element_bytes != 0) {
        return 1;
    }
    return 16 / element_bytes;
}

[[nodiscard]] uint64_t denseArgPlannerAlignmentPeriod(uint64_t inner_size, uint64_t element_bytes) {
    uint64_t period = 1;
    while (period < 16 && ((inner_size % 16) * (element_bytes % 16) * period) % 16 != 0) {
        period <<= 1;
    }
    return period;
}

[[nodiscard]] uint64_t denseArgPlannerTargetPartitions(uint64_t reduction_size) {
    uint64_t partitions = 1;
    while (partitions < 32 && partitions * 16 < reduction_size) {
        partitions <<= 1;
    }
    return partitions;
}

[[nodiscard]] uint64_t denseArgPlannerWarpsPerTile(const DenseDirectStageGeometry& stage,
                                                    DataType value_dtype,
                                                    bool has_carried_index,
                                                    DataType carried_index_dtype) {
    constexpr uint64_t WARP_THREADS = 32;
    constexpr uint64_t WARPS_PER_BLOCK = 8;
    constexpr uint64_t TARGET_ACTIVE_WARPS = CubReductionTiledPolicy::TARGET_ACTIVE_WARPS;

    const uint64_t items_per_lane = denseArgPlannerItemsPerLane(value_dtype);
    const uint64_t complete_packets = stage.inner_size / items_per_lane;
    uint64_t desired_group_threads = 8;
    while (desired_group_threads < WARP_THREADS && (desired_group_threads << 1) <= complete_packets) {
        desired_group_threads <<= 1;
    }

    uint64_t alignment_period = denseArgPlannerAlignmentPeriod(stage.inner_size, denseArgPlannerElementBytes(value_dtype));
    if (has_carried_index) {
        alignment_period = std::max(
            alignment_period,
            denseArgPlannerAlignmentPeriod(stage.inner_size, denseArgPlannerElementBytes(carried_index_dtype)));
    }
    const uint64_t row_partitions = std::max(denseArgPlannerTargetPartitions(stage.reduction_size), alignment_period);
    const uint64_t max_group_threads = std::max<uint64_t>(1, WARP_THREADS * WARPS_PER_BLOCK / row_partitions);
    const uint64_t group_threads = std::min(desired_group_threads, max_group_threads);
    const uint64_t tile_components = std::max<uint64_t>(1, group_threads * items_per_lane);
    const uint64_t component_tiles = ceilDivideDensePlanner(stage.inner_size, tile_components);

    uint64_t output_tiles = TARGET_ACTIVE_WARPS;
    if (stage.outer_size <= std::numeric_limits<uint64_t>::max() / component_tiles) {
        output_tiles = std::max<uint64_t>(1, stage.outer_size * component_tiles);
    }
    const uint64_t desired_warps = ceilDivideDensePlanner(TARGET_ACTIVE_WARPS, output_tiles);
    uint64_t warps = 1;
    while (warps < WARPS_PER_BLOCK && warps < desired_warps && warps < stage.reduction_size) {
        warps <<= 1;
    }

    // Match the production alignment requirement for revisited rows. A UINT64 carried stream is intentionally treated
    // conservatively: it is outside the current normal direct hot path and should never make an awkward stage look cheap.
    while (warps <= WARPS_PER_BLOCK) {
        const bool one_row_per_warp = stage.reduction_size <= warps;
        const bool values_aligned = one_row_per_warp
                                    || ((stage.inner_size % 16) * (denseArgPlannerElementBytes(value_dtype) % 16) * warps) % 16 == 0;
        const bool indices_aligned = !has_carried_index || one_row_per_warp
                                     || ((stage.inner_size % 16)
                                         * (denseArgPlannerElementBytes(carried_index_dtype) % 16) * warps)
                                            % 16
                                            == 0;
        if (values_aligned && indices_aligned) {
            return warps;
        }
        warps <<= 1;
    }
    return WARPS_PER_BLOCK;
}

[[nodiscard]] DenseArgPlannerStageClass classifyDenseArgPlannerStage(const DenseDirectStageGeometry& stage,
                                                                     DataType value_dtype,
                                                                     bool has_carried_index,
                                                                     DataType carried_index_dtype) {
    if (stage.path == CubReductionPath::DeviceTransformReduce) {
        return DenseArgPlannerStageClass::DeviceTransform;
    }
    if (stage.path == CubReductionPath::ContiguousFixedSegment) {
        return DenseArgPlannerStageClass::Contiguous;
    }
    THOR_THROW_IF_FALSE(stage.path == CubReductionPath::TiledFixedSegment);

    if (stage.inner_size <= 32) {
        return DenseArgPlannerStageClass::TiledNarrow;
    }
    if (carried_index_dtype != DataType::UINT32) {
        return DenseArgPlannerStageClass::TiledAwkwardHeadBulkTail;
    }

    const uint64_t value_bytes = denseArgPlannerElementBytes(value_dtype);
    if (value_bytes < 2) {
        return DenseArgPlannerStageClass::TiledAwkwardHeadBulkTail;
    }
    const uint64_t items_per_lane = denseArgPlannerItemsPerLane(value_dtype);
    const uint64_t complete_packets = stage.inner_size / items_per_lane;
    uint64_t desired_group_threads = 8;
    while (desired_group_threads < 32 && (desired_group_threads << 1) <= complete_packets) {
        desired_group_threads <<= 1;
    }
    uint64_t alignment_period = denseArgPlannerAlignmentPeriod(stage.inner_size, value_bytes);
    if (has_carried_index) {
        alignment_period = std::max<uint64_t>(alignment_period, denseArgPlannerAlignmentPeriod(stage.inner_size, 4));
    }
    const uint64_t row_partitions = std::max(denseArgPlannerTargetPartitions(stage.reduction_size), alignment_period);
    const uint64_t max_group_threads = std::max<uint64_t>(1, 256 / row_partitions);
    const uint64_t group_threads = std::min(desired_group_threads, max_group_threads);
    const uint64_t tile_components = std::max<uint64_t>(1, group_threads * items_per_lane);
    const bool complete_tiles = stage.inner_size % tile_components == 0;
    const bool value_rows_aligned = ((stage.inner_size % 16) * (value_bytes % 16)) % 16 == 0;
    const bool carried_rows_aligned = !has_carried_index || ((stage.inner_size % 4) * 4) % 16 == 0;
    return complete_tiles && value_rows_aligned && carried_rows_aligned
               ? DenseArgPlannerStageClass::TiledSimpleAlignedFullTiles
               : DenseArgPlannerStageClass::TiledAwkwardHeadBulkTail;
}

[[nodiscard]] uint64_t estimateDenseArgStageActiveWarps(const DenseDirectStageGeometry& stage,
                                                         DataType value_dtype,
                                                         bool has_carried_index,
                                                         DataType carried_index_dtype) {
    constexpr uint64_t TARGET_ACTIVE_WARPS = CubReductionTiledPolicy::TARGET_ACTIVE_WARPS;
    if (stage.path == CubReductionPath::DeviceTransformReduce) {
        return std::min<uint64_t>(
            TARGET_ACTIVE_WARPS,
            std::max<uint64_t>(1, ceilDivideDensePlanner(stage.input_elements, uint64_t{32})));
    }
    if (stage.path == CubReductionPath::ContiguousFixedSegment) {
        const uint64_t items_per_lane = denseArgPlannerItemsPerLane(value_dtype);
        const uint64_t packets = stage.reduction_size / items_per_lane;
        uint64_t group_threads = 1;
        while (group_threads < 32 && group_threads < packets) {
            group_threads <<= 1;
        }
        long double logical_threads = static_cast<long double>(stage.output_elements)
                                      * static_cast<long double>(std::max<uint64_t>(group_threads, 1));
        const uint64_t warps = logical_threads >= static_cast<long double>(TARGET_ACTIVE_WARPS * 32)
                                   ? TARGET_ACTIVE_WARPS
                                   : std::max<uint64_t>(1, static_cast<uint64_t>(std::ceil(logical_threads / 32.0L)));
        return std::min<uint64_t>(TARGET_ACTIVE_WARPS, warps);
    }

    THOR_THROW_IF_FALSE(stage.path == CubReductionPath::TiledFixedSegment);
    if (stage.inner_size <= 32) {
        if (stage.outer_size > std::numeric_limits<uint64_t>::max() / 8) {
            return TARGET_ACTIVE_WARPS;
        }
        return std::min<uint64_t>(TARGET_ACTIVE_WARPS, std::max<uint64_t>(1, stage.outer_size * 8));
    }

    const uint64_t items_per_lane = denseArgPlannerItemsPerLane(value_dtype);
    const uint64_t complete_packets = stage.inner_size / items_per_lane;
    uint64_t group_threads = 8;
    while (group_threads < 32 && (group_threads << 1) <= complete_packets) {
        group_threads <<= 1;
    }
    uint64_t alignment_period = denseArgPlannerAlignmentPeriod(stage.inner_size, denseArgPlannerElementBytes(value_dtype));
    if (has_carried_index) {
        alignment_period = std::max(
            alignment_period,
            denseArgPlannerAlignmentPeriod(stage.inner_size, denseArgPlannerElementBytes(carried_index_dtype)));
    }
    const uint64_t row_partitions = std::max(denseArgPlannerTargetPartitions(stage.reduction_size), alignment_period);
    const uint64_t max_group_threads = std::max<uint64_t>(1, 256 / row_partitions);
    group_threads = std::min(group_threads, max_group_threads);
    const uint64_t tile_components = std::max<uint64_t>(1, group_threads * items_per_lane);
    const uint64_t component_tiles = ceilDivideDensePlanner(stage.inner_size, tile_components);
    if (stage.outer_size > std::numeric_limits<uint64_t>::max() / component_tiles) {
        return TARGET_ACTIVE_WARPS;
    }
    const uint64_t output_tiles = std::max<uint64_t>(1, stage.outer_size * component_tiles);
    const uint64_t warps_per_tile = denseArgPlannerWarpsPerTile(stage, value_dtype, has_carried_index, carried_index_dtype);
    if (output_tiles > std::numeric_limits<uint64_t>::max() / warps_per_tile) {
        return TARGET_ACTIVE_WARPS;
    }
    return std::min<uint64_t>(TARGET_ACTIVE_WARPS, output_tiles * warps_per_tile);
}

[[nodiscard]] long double denseArgPlannerExecutionPenalty(const DenseDirectStageGeometry& stage,
                                                           DenseArgPlannerStageClass stage_class,
                                                           DataType value_dtype,
                                                           DataType carried_index_dtype,
                                                           uint64_t active_warps) {
    using namespace CubReductionTiledPolicy;
    const bool low_precision = denseArgPlannerElementBytes(value_dtype) <= 2;

    // ARG-PLAN-2B calibration: the forced-order census exposed two throughput troughs that a traffic-only model cannot
    // represent. Very short contiguous rows and short narrow tiled rows can be dramatically slower than their logical
    // byte count suggests once there is enough work to reach their steady execution regime. Conversely, multiplying
    // that throughput penalty into a tiny under-filled later pass grossly overcharges it because the generic active-warp
    // term already models the lack of concurrency. Apply these regime penalties only once at least one quarter of the
    // normal saturation target is exposed. The buckets are intentionally broad and keyed only to direct-kernel geometry
    // so the model remains easy to recalibrate when the pinned direct ARG kernels change.
    constexpr uint64_t THROUGHPUT_REGIME_WARPS = TARGET_ACTIVE_WARPS / 4;
    const bool throughput_regime = active_warps >= THROUGHPUT_REGIME_WARPS;

    long double penalty = 1.0L;
    switch (stage_class) {
        case DenseArgPlannerStageClass::DeviceTransform:
        case DenseArgPlannerStageClass::TiledSimpleAlignedFullTiles:
            penalty = 1.0L;
            break;
        case DenseArgPlannerStageClass::Contiguous:
            if (!throughput_regime) {
                penalty = 1.0L;
            } else if (stage.reduction_size <= 16) {
                penalty = low_precision ? 30.0L : 20.0L;
            } else if (stage.reduction_size <= 24) {
                penalty = low_precision ? 8.0L : 4.0L;
            } else if (stage.reduction_size <= 31) {
                penalty = low_precision ? 4.0L : 2.0L;
            } else if (stage.reduction_size <= 63) {
                penalty = low_precision ? 2.0L : 1.0L;
            } else {
                penalty = 1.0L;
            }
            break;
        case DenseArgPlannerStageClass::TiledNarrow:
            if (!throughput_regime) {
                penalty = 1.0L;
            } else if (stage.reduction_size <= 24) {
                penalty = low_precision ? 4.0L : 3.0L;
            } else if (stage.reduction_size <= 31) {
                penalty = low_precision ? 1.50L : 1.25L;
            } else {
                penalty = low_precision ? 1.25L : 1.10L;
            }
            break;
        case DenseArgPlannerStageClass::TiledAwkwardHeadBulkTail:
            // Current census data puts awkward FP16/BF16 tiled ARG in roughly the 0.4-0.6 TB/s regime while the clean
            // aligned path is materially healthier. Keep this one coarse family weight so a later direct-kernel pass
            // can recalibrate one constant instead of rewriting topology logic.
            penalty = low_precision ? 2.75L : 1.85L;
            break;
    }
    if (carried_index_dtype == DataType::UINT64) {
        penalty *= 1.50L;
    }
    return penalty;
}

[[nodiscard]] long double estimateDenseArgStageCost(const DenseDirectStageGeometry& stage,
                                                     DataType value_input_dtype,
                                                     bool has_carried_index,
                                                     DataType carried_index_dtype,
                                                     uint64_t output_value_bytes_per_element,
                                                     uint64_t output_index_bytes_per_element,
                                                     bool intermediate_is_hot,
                                                     uint64_t l2_cache_bytes) {
    constexpr long double L2_RESIDENT_INPUT_DISCOUNT = 8.0L;
    const uint64_t value_input_bytes = denseArgPlannerElementBytes(value_input_dtype);
    const uint64_t carried_index_bytes = has_carried_index ? denseArgPlannerElementBytes(carried_index_dtype) : 0;
    const long double input_bytes = static_cast<long double>(stage.input_elements)
                                    * static_cast<long double>(value_input_bytes + carried_index_bytes);
    const long double output_bytes = static_cast<long double>(stage.output_elements)
                                     * static_cast<long double>(output_value_bytes_per_element
                                                                + output_index_bytes_per_element);

    long double charged_input_bytes = input_bytes;
    if (intermediate_is_hot && l2_cache_bytes != 0 && input_bytes <= static_cast<long double>(l2_cache_bytes)) {
        charged_input_bytes /= L2_RESIDENT_INPUT_DISCOUNT;
    }

    const DenseArgPlannerStageClass stage_class =
        classifyDenseArgPlannerStage(stage, value_input_dtype, has_carried_index, carried_index_dtype);
    const uint64_t active_warps =
        estimateDenseArgStageActiveWarps(stage, value_input_dtype, has_carried_index, carried_index_dtype);
    const long double active_warp_ratio = static_cast<long double>(CubReductionTiledPolicy::TARGET_ACTIVE_WARPS)
                                          / static_cast<long double>(std::max<uint64_t>(active_warps, 1));
    const long double parallelism_penalty = std::max<long double>(1.0L, std::pow(active_warp_ratio, 1.10L));
    const long double execution_penalty =
        denseArgPlannerExecutionPenalty(stage, stage_class, value_input_dtype, carried_index_dtype, active_warps);
    return (charged_input_bytes + output_bytes) * parallelism_penalty * execution_penalty;
}

[[nodiscard]] DenseArgCompositionCostContext denseArgCompositionCostContext(
    DataType input_dtype,
    DataType carried_index_dtype,
    const CubArgReductionOutputOptions& outputs,
    DataType value_output_dtype,
    const Stream& stream) {
    int l2_cache_bytes = 0;
    const cudaError_t status = cudaDeviceGetAttribute(
        &l2_cache_bytes, cudaDevAttrL2CacheSize, static_cast<int>(stream.getGpuNum()));
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("Failed to query GPU L2 size for dense ARG planning: ")
                                 + cudaGetErrorString(status));
    }
    return DenseArgCompositionCostContext{.input_dtype = input_dtype,
                                          .carried_index_dtype = carried_index_dtype,
                                          .produce_value = outputs.produce_value,
                                          .value_output_dtype = value_output_dtype,
                                          .produce_index = outputs.produce_index,
                                          .index_output_dtype = outputs.index_output_dtype,
                                          .l2_cache_bytes = l2_cache_bytes > 0 ? static_cast<uint64_t>(l2_cache_bytes) : 0};
}

/**
 * Transitional interval-DP topology planner retained for structural ComposedDense validation and ARG execution.
 *
 * VALUE no longer executes this left/right composition: ordinary dense VALUE is planned globally through
 * planDenseReduction() and materialized as a flat DenseExecutablePlan. ARG still uses this interval planner until
 * DENSE-ARG-CUTOVER, so keep it isolated from VALUE execution rather than evolving it into a second dense planner.
 */
template <typename StageCostFn>
[[nodiscard]] std::optional<CubReductionDenseCompositionPlan> makeDenseIntervalCompositionPlan(
    const CubReductionGeometry& geometry,
    const std::vector<uint64_t>& input_dimensions,
    StageCostFn&& stage_cost) {
    if (!geometry.dense_run_geometry.has_value()) {
        return std::nullopt;
    }
    THOR_THROW_IF_FALSE(input_dimensions.size() == geometry.rank);

    const std::vector<DenseReducedRunSpan> reduced_run_spans = denseReducedRunSpans(geometry, input_dimensions);
    const size_t reduced_run_count = reduced_run_spans.size();
    THOR_THROW_IF_FALSE(reduced_run_count != 0);

    if (reduced_run_count == 1) {
        const std::optional<CubReductionDenseCompositionStage> stage = makeDenseCompositionStagePlan(
            input_dimensions, reduced_run_spans.front(), CubReductionDenseCompositionStageRole::Complete);
        if (!stage.has_value() || !std::isfinite(stage_cost(stage.value()))) {
            return std::nullopt;
        }
        CubReductionDenseCompositionPlan plan;
        plan.stages.push_back(stage.value());
        THOR_THROW_IF_FALSE(plan.stages.back().output_dimensions == geometry.output_dimensions);
        return plan;
    }

    const long double infinity = std::numeric_limits<long double>::infinity();
    std::vector<std::vector<long double>> subproblem_cost(
        reduced_run_count, std::vector<long double>(reduced_run_count, infinity));
    std::vector<std::vector<char>> subproblem_choice(
        reduced_run_count, std::vector<char>(reduced_run_count, '\0'));

    // Base states are final passes over hot partials.
    for (size_t run_index = 0; run_index < reduced_run_count; ++run_index) {
        const std::vector<uint64_t> state_dimensions =
            denseReductionStateDimensions(input_dimensions, reduced_run_spans, run_index, run_index);
        const std::optional<CubReductionDenseCompositionStage> stage = makeDenseCompositionStagePlan(
            state_dimensions, reduced_run_spans[run_index], CubReductionDenseCompositionStageRole::Final);
        if (!stage.has_value()) {
            continue;
        }
        subproblem_cost[run_index][run_index] = stage_cost(stage.value());
    }

    // Proper non-root interval states consume and produce hot partials.
    for (size_t remaining_runs = 2; remaining_runs < reduced_run_count; ++remaining_runs) {
        for (size_t state_left = 0; state_left + remaining_runs <= reduced_run_count; ++state_left) {
            const size_t state_right = state_left + remaining_runs - 1;
            const std::vector<uint64_t> state_dimensions =
                denseReductionStateDimensions(input_dimensions, reduced_run_spans, state_left, state_right);

            auto candidateCost = [&](size_t selected_run, size_t child_left, size_t child_right) {
                if (!std::isfinite(subproblem_cost[child_left][child_right])) {
                    return infinity;
                }
                const std::optional<CubReductionDenseCompositionStage> stage = makeDenseCompositionStagePlan(
                    state_dimensions,
                    reduced_run_spans[selected_run],
                    CubReductionDenseCompositionStageRole::Intermediate);
                if (!stage.has_value()) {
                    return infinity;
                }
                const long double own_cost = stage_cost(stage.value());
                return std::isfinite(own_cost) ? own_cost + subproblem_cost[child_left][child_right] : infinity;
            };

            const long double left_cost = candidateCost(state_left, state_left + 1, state_right);
            const long double right_cost = candidateCost(state_right, state_left, state_right - 1);
            if (left_cost <= right_cost) {
                subproblem_cost[state_left][state_right] = left_cost;
                subproblem_choice[state_left][state_right] = 'L';
            } else {
                subproblem_cost[state_left][state_right] = right_cost;
                subproblem_choice[state_left][state_right] = 'R';
            }
        }
    }

    // The root is the only cold stage.
    const size_t root_left = 0;
    const size_t root_right = reduced_run_count - 1;
    auto rootCandidateCost = [&](size_t selected_run, size_t child_left, size_t child_right) {
        if (!std::isfinite(subproblem_cost[child_left][child_right])) {
            return infinity;
        }
        const std::optional<CubReductionDenseCompositionStage> stage = makeDenseCompositionStagePlan(
            input_dimensions, reduced_run_spans[selected_run], CubReductionDenseCompositionStageRole::First);
        if (!stage.has_value()) {
            return infinity;
        }
        const long double own_cost = stage_cost(stage.value());
        return std::isfinite(own_cost) ? own_cost + subproblem_cost[child_left][child_right] : infinity;
    };

    const long double root_left_cost = rootCandidateCost(root_left, root_left + 1, root_right);
    const long double root_right_cost = rootCandidateCost(root_right, root_left, root_right - 1);
    if (!std::isfinite(root_left_cost) && !std::isfinite(root_right_cost)) {
        return std::nullopt;
    }
    if (root_left_cost <= root_right_cost) {
        subproblem_cost[root_left][root_right] = root_left_cost;
        subproblem_choice[root_left][root_right] = 'L';
    } else {
        subproblem_cost[root_left][root_right] = root_right_cost;
        subproblem_choice[root_left][root_right] = 'R';
    }

    std::vector<size_t> stage_run_order;
    stage_run_order.reserve(reduced_run_count);
    size_t state_left = root_left;
    size_t state_right = root_right;
    while (state_left < state_right) {
        const char choice = subproblem_choice[state_left][state_right];
        if (choice == 'L') {
            stage_run_order.push_back(state_left++);
        } else if (choice == 'R') {
            stage_run_order.push_back(state_right--);
        } else {
            return std::nullopt;
        }
    }
    stage_run_order.push_back(state_left);

    CubReductionDenseCompositionPlan plan;
    plan.stages.reserve(stage_run_order.size());
    std::vector<uint64_t> current_dimensions = input_dimensions;
    for (size_t stage_index = 0; stage_index < stage_run_order.size(); ++stage_index) {
        const CubReductionDenseCompositionStageRole role =
            stage_index == 0 ? CubReductionDenseCompositionStageRole::First
            : stage_index + 1 == stage_run_order.size() ? CubReductionDenseCompositionStageRole::Final
                                                       : CubReductionDenseCompositionStageRole::Intermediate;
        const std::optional<CubReductionDenseCompositionStage> stage = makeDenseCompositionStagePlan(
            current_dimensions, reduced_run_spans[stage_run_order[stage_index]], role);
        if (!stage.has_value()) {
            return std::nullopt;
        }
        current_dimensions = stage->output_dimensions;
        plan.stages.push_back(stage.value());
    }

    THOR_THROW_IF_FALSE(!plan.stages.empty());
    THOR_THROW_IF_FALSE(current_dimensions == geometry.output_dimensions);
    return plan;
}


[[nodiscard]] std::optional<CubReductionDenseCompositionPlan> makeDenseIntervalCompositionPlanForRunOrder(
    const CubReductionGeometry& geometry,
    const std::vector<uint64_t>& input_dimensions,
    const std::vector<uint32_t>& run_order) {
    if (!geometry.dense_run_geometry.has_value()) {
        return std::nullopt;
    }
    const std::vector<DenseReducedRunSpan> reduced_run_spans = denseReducedRunSpans(geometry, input_dimensions);
    if (run_order.size() != reduced_run_spans.size() || run_order.empty()) {
        return std::nullopt;
    }

    size_t left = 0;
    size_t right = reduced_run_spans.size() - 1;
    std::vector<uint64_t> current_dimensions = input_dimensions;
    CubReductionDenseCompositionPlan plan;
    plan.stages.reserve(run_order.size());
    for (size_t stage_index = 0; stage_index < run_order.size(); ++stage_index) {
        const size_t selected = static_cast<size_t>(run_order[stage_index]);
        if (selected >= reduced_run_spans.size()) {
            return std::nullopt;
        }
        if (left == right) {
            if (selected != left || stage_index + 1 != run_order.size()) {
                return std::nullopt;
            }
        } else if (selected == left) {
            ++left;
        } else if (selected == right) {
            --right;
        } else {
            return std::nullopt;
        }

        const CubReductionDenseCompositionStageRole role =
            run_order.size() == 1 ? CubReductionDenseCompositionStageRole::Complete
            : stage_index == 0 ? CubReductionDenseCompositionStageRole::First
            : stage_index + 1 == run_order.size() ? CubReductionDenseCompositionStageRole::Final
                                                  : CubReductionDenseCompositionStageRole::Intermediate;
        const std::optional<CubReductionDenseCompositionStage> stage =
            makeDenseCompositionStagePlan(current_dimensions, reduced_run_spans[selected], role);
        if (!stage.has_value()) {
            return std::nullopt;
        }
        current_dimensions = stage->output_dimensions;
        plan.stages.push_back(stage.value());
    }
    if (current_dimensions != geometry.output_dimensions) {
        return std::nullopt;
    }
    return plan;
}

[[nodiscard]] std::optional<CubReductionDenseCompositionPlan> makeStructuralDenseCompositionPlan(
    const CubReductionGeometry& geometry,
    const std::vector<uint64_t>& input_dimensions) {
    return makeDenseIntervalCompositionPlan(
        geometry, input_dimensions, [](const CubReductionDenseCompositionStage&) { return 1.0L; });
}

[[nodiscard]] std::optional<CubArgReductionDenseCompositionPlan> makeDenseArgCompositionPlan(
    const CubReductionGeometry& geometry,
    const std::vector<uint64_t>& input_dimensions,
    std::optional<DenseArgCompositionCostContext> cost_context = std::nullopt) {
    if (!geometry.dense_run_geometry.has_value()) {
        return std::nullopt;
    }

    const DataType carried_index_dtype =
        geometry.reduction_size <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())
            ? DataType::UINT32
            : DataType::UINT64;

    std::optional<CubReductionDenseCompositionPlan> topology;
    if (!cost_context.has_value()) {
        topology = makeStructuralDenseCompositionPlan(geometry, input_dimensions);
    } else {
        DenseArgCompositionCostContext planner_context = cost_context.value();
        planner_context.carried_index_dtype = carried_index_dtype;
        const uint64_t intermediate_value_bytes = sizeof(float);
        const uint64_t intermediate_index_bytes = denseArgPlannerElementBytes(carried_index_dtype);
        const uint64_t final_value_bytes = planner_context.produce_value
                                               ? denseArgPlannerElementBytes(planner_context.value_output_dtype)
                                               : 0;
        const uint64_t final_index_bytes = planner_context.produce_index
                                               ? denseArgPlannerElementBytes(planner_context.index_output_dtype)
                                               : 0;
        topology = makeDenseIntervalCompositionPlan(
            geometry,
            input_dimensions,
            [&](const CubReductionDenseCompositionStage& stage) {
                const DenseDirectStageGeometry direct_geometry = directStageGeometryFromPlan(stage);
                switch (stage.role) {
                    case CubReductionDenseCompositionStageRole::Complete:
                        return estimateDenseArgStageCost(direct_geometry,
                                                         planner_context.input_dtype,
                                                         false,
                                                         carried_index_dtype,
                                                         final_value_bytes,
                                                         final_index_bytes,
                                                         false,
                                                         planner_context.l2_cache_bytes);
                    case CubReductionDenseCompositionStageRole::First:
                        return estimateDenseArgStageCost(direct_geometry,
                                                         planner_context.input_dtype,
                                                         false,
                                                         carried_index_dtype,
                                                         intermediate_value_bytes,
                                                         intermediate_index_bytes,
                                                         false,
                                                         planner_context.l2_cache_bytes);
                    case CubReductionDenseCompositionStageRole::Intermediate:
                        return estimateDenseArgStageCost(direct_geometry,
                                                         DataType::FP32,
                                                         true,
                                                         carried_index_dtype,
                                                         intermediate_value_bytes,
                                                         intermediate_index_bytes,
                                                         true,
                                                         planner_context.l2_cache_bytes);
                    case CubReductionDenseCompositionStageRole::Final:
                        return estimateDenseArgStageCost(direct_geometry,
                                                         DataType::FP32,
                                                         true,
                                                         carried_index_dtype,
                                                         final_value_bytes,
                                                         final_index_bytes,
                                                         true,
                                                         planner_context.l2_cache_bytes);
                }
                throw std::logic_error("Unknown dense ARG composition stage role.");
            });
    }

    if (!topology.has_value()) {
        return std::nullopt;
    }
    CubArgReductionDenseCompositionPlan plan;
    plan.topology = std::move(topology.value());
    plan.carried_index_dtype = carried_index_dtype;
    return plan;
}

void requireExecutableFixedSegmentSize(const CubReductionGeometry& geometry) {
    if (usesCubFixedSegmentSize(geometry.path)
        && geometry.reduction_size > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("CUB fixed-size segmented reduction segment size exceeds its int limit.");
    }
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

    // DeviceSegmentedReduce emits segments in physical prefix order. VIEW-DIRECT-2A intentionally activates only
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

[[nodiscard]] TensorReachableAddressSpan tensorReachableAddressSpan(const Tensor& tensor) {
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

    const uintptr_t begin = reinterpret_cast<uintptr_t>(tensor.getMemPtr<void>());
    if (span_bytes > static_cast<uint64_t>(std::numeric_limits<uintptr_t>::max() - begin)) {
        throw std::invalid_argument("CUB tensor reduction tensor view address span exceeds the process address space.");
    }
    return TensorReachableAddressSpan{begin, begin + static_cast<uintptr_t>(span_bytes)};
}

[[nodiscard]] bool tensorStorageOverlaps(const Tensor& lhs, const Tensor& rhs) {
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
    // A dense alias into a larger allocation satisfies that contract even though Tensor records
    // aliasView() strides as custom metadata.
    if (!output.isDenseContiguous()) {
        throw std::invalid_argument(std::string(role) + " must be dense contiguous.");
    }
}

void requireExpectedOutput(const Tensor& input,
                           const Tensor& output,
                           DataType output_dtype,
                           const CubReductionGeometry& geometry) {
    requireDenseContiguousReductionOutput(output, "output");
    requireSameGpuPlacement(input, output, "input", "output");
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

void requireExpectedArgOutput(const Tensor& input,
                              const Tensor& output,
                              DataType output_dtype,
                              const CubReductionGeometry& geometry,
                              const char* role) {
    requireDenseContiguousReductionOutput(output, role);
    requireSameGpuPlacement(input, output, "input", role);
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

size_t queryReductionBytes(const CubReductionInternal::CubReductionStageSemantics& semantics,
                           DataType input_dtype,
                           const void* input,
                           uint64_t input_elements,
                           DataType output_dtype,
                           void* output,
                           const CubReductionGeometry& geometry,
                           float output_scale,
                           const Stream& stream) {
    using CubReductionInternal::CubReductionStageCombine;
    using CubReductionInternal::CubReductionStageFinalize;
    using CubReductionInternal::CubReductionStageInputTransform;

    if (semantics.combine == CubReductionStageCombine::Sum) {
        if (semantics.input_transform == CubReductionStageInputTransform::Identity) {
            if (semantics.finalize == CubReductionStageFinalize::Identity) {
                return CubReductionInternal::querySumReductionBytes(
                    input_dtype, input, input_elements, output_dtype, output, geometry, output_scale, stream);
            }
            if (semantics.finalize == CubReductionStageFinalize::Divide) {
                return CubReductionInternal::querySumDivideReductionBytes(input_dtype,
                                                                           input,
                                                                           input_elements,
                                                                           output_dtype,
                                                                           output,
                                                                           geometry,
                                                                           semantics.finalize_divisor,
                                                                           output_scale,
                                                                           stream);
            }
            if (semantics.finalize == CubReductionStageFinalize::SquareRoot) {
                return CubReductionInternal::querySumSqrtReductionBytes(
                    input_dtype, input, input_elements, output_dtype, output, geometry, output_scale, stream);
            }
        } else if (semantics.input_transform == CubReductionStageInputTransform::AbsoluteValue
                   && semantics.finalize == CubReductionStageFinalize::Identity) {
            return CubReductionInternal::queryL1NormReductionBytes(
                input_dtype, input, input_elements, output_dtype, output, geometry, output_scale, stream);
        } else if (semantics.input_transform == CubReductionStageInputTransform::Square) {
            if (semantics.finalize == CubReductionStageFinalize::Identity) {
                return CubReductionInternal::querySumSquaresReductionBytes(
                    input_dtype, input, input_elements, output_dtype, output, geometry, output_scale, stream);
            }
            if (semantics.finalize == CubReductionStageFinalize::SquareRoot) {
                return CubReductionInternal::queryL2NormReductionBytes(
                    input_dtype, input, input_elements, output_dtype, output, geometry, output_scale, stream);
            }
        }
    } else if (semantics.input_transform == CubReductionStageInputTransform::Identity
               && semantics.finalize == CubReductionStageFinalize::Identity) {
        switch (semantics.combine) {
            case CubReductionStageCombine::Product:
                return CubReductionInternal::queryProductReductionBytes(
                    input_dtype, input, input_elements, output_dtype, output, geometry, output_scale, stream);
            case CubReductionStageCombine::Minimum:
                return CubReductionInternal::queryMinReductionBytes(
                    input_dtype, input, input_elements, output_dtype, output, geometry, output_scale, stream);
            case CubReductionStageCombine::Maximum:
                return CubReductionInternal::queryMaxReductionBytes(
                    input_dtype, input, input_elements, output_dtype, output, geometry, output_scale, stream);
            case CubReductionStageCombine::Sum:
                break;
        }
    }

    throw std::logic_error("Unsupported CUB value-reduction stage semantic combination.");
}

void launchReduction(const CubReductionInternal::CubReductionStageSemantics& semantics,
                     const Tensor& temp_storage,
                     size_t temp_storage_bytes,
                     const Tensor& input,
                     Tensor& output,
                     const CubReductionGeometry& geometry,
                     float output_scale,
                     Stream& stream) {
    using CubReductionInternal::CubReductionStageCombine;
    using CubReductionInternal::CubReductionStageFinalize;
    using CubReductionInternal::CubReductionStageInputTransform;

    if (semantics.combine == CubReductionStageCombine::Sum) {
        if (semantics.input_transform == CubReductionStageInputTransform::Identity) {
            if (semantics.finalize == CubReductionStageFinalize::Identity) {
                CubReductionInternal::launchSumReduction(
                    temp_storage, temp_storage_bytes, input, output, geometry, output_scale, stream);
                return;
            }
            if (semantics.finalize == CubReductionStageFinalize::Divide) {
                CubReductionInternal::launchSumDivideReduction(temp_storage,
                                                                temp_storage_bytes,
                                                                input,
                                                                output,
                                                                geometry,
                                                                semantics.finalize_divisor,
                                                                output_scale,
                                                                stream);
                return;
            }
            if (semantics.finalize == CubReductionStageFinalize::SquareRoot) {
                CubReductionInternal::launchSumSqrtReduction(
                    temp_storage, temp_storage_bytes, input, output, geometry, output_scale, stream);
                return;
            }
        } else if (semantics.input_transform == CubReductionStageInputTransform::AbsoluteValue
                   && semantics.finalize == CubReductionStageFinalize::Identity) {
            CubReductionInternal::launchL1NormReduction(
                temp_storage, temp_storage_bytes, input, output, geometry, output_scale, stream);
            return;
        } else if (semantics.input_transform == CubReductionStageInputTransform::Square) {
            if (semantics.finalize == CubReductionStageFinalize::Identity) {
                CubReductionInternal::launchSumSquaresReduction(
                    temp_storage, temp_storage_bytes, input, output, geometry, output_scale, stream);
                return;
            }
            if (semantics.finalize == CubReductionStageFinalize::SquareRoot) {
                CubReductionInternal::launchL2NormReduction(
                    temp_storage, temp_storage_bytes, input, output, geometry, output_scale, stream);
                return;
            }
        }
    } else if (semantics.input_transform == CubReductionStageInputTransform::Identity
               && semantics.finalize == CubReductionStageFinalize::Identity) {
        switch (semantics.combine) {
            case CubReductionStageCombine::Product:
                CubReductionInternal::launchProductReduction(
                    temp_storage, temp_storage_bytes, input, output, geometry, output_scale, stream);
                return;
            case CubReductionStageCombine::Minimum:
                CubReductionInternal::launchMinReduction(
                    temp_storage, temp_storage_bytes, input, output, geometry, output_scale, stream);
                return;
            case CubReductionStageCombine::Maximum:
                CubReductionInternal::launchMaxReduction(
                    temp_storage, temp_storage_bytes, input, output, geometry, output_scale, stream);
                return;
            case CubReductionStageCombine::Sum:
                break;
        }
    }

    throw std::logic_error("Unsupported CUB value-reduction stage semantic combination.");
}

size_t queryArgReductionBytes(CubArgReductionOp op,
                              const Tensor& input,
                              Tensor* value_output,
                              Tensor* index_output,
                              const CubReductionGeometry& geometry,
                              const Stream& stream) {
    switch (op) {
        case CubArgReductionOp::ArgMin:
            return CubReductionInternal::queryArgMinReductionBytes(
                input, value_output, index_output, geometry, stream);
        case CubArgReductionOp::ArgMax:
            return CubReductionInternal::queryArgMaxReductionBytes(
                input, value_output, index_output, geometry, stream);
    }
    throw std::invalid_argument("Unsupported CUB arg reduction operation.");
}

size_t queryArgReductionBytes(CubArgReductionOp op,
                              DataType input_dtype,
                              std::optional<DataType> value_output_dtype,
                              std::optional<DataType> index_output_dtype,
                              const CubReductionGeometry& geometry,
                              const Stream& stream) {
    switch (op) {
        case CubArgReductionOp::ArgMin:
            return CubReductionInternal::queryArgMinReductionBytes(
                input_dtype, value_output_dtype, index_output_dtype, geometry, stream);
        case CubArgReductionOp::ArgMax:
            return CubReductionInternal::queryArgMaxReductionBytes(
                input_dtype, value_output_dtype, index_output_dtype, geometry, stream);
    }
    throw std::invalid_argument("Unsupported CUB arg reduction operation.");
}


[[nodiscard]] CubReductionGeometry densePhysicalPassGeometry(const DenseReductionPhysicalPlan& physical_plan) {
    if (const auto* direct = dynamic_cast<const DenseDirectReductionPhysicalPlan*>(&physical_plan); direct != nullptr) {
        return CubReduction::analyzeGeometry(direct->input_dimensions, std::vector<uint32_t>{direct->reduction_axis});
    }
    if (const auto* rk = dynamic_cast<const DenseRKFamilyPhysicalPlan*>(&physical_plan); rk != nullptr) {
        return CubReduction::analyzeGeometry(rk->input_dimensions, std::vector<uint32_t>{rk->reduction_axis});
    }
    throw std::logic_error("Dense VALUE physical plan has no geometry adapter.");
}

[[nodiscard]] std::vector<uint64_t> densePhysicalPassInputDimensions(const DenseReductionPhysicalPlan& physical_plan) {
    if (const auto* direct = dynamic_cast<const DenseDirectReductionPhysicalPlan*>(&physical_plan); direct != nullptr) {
        return direct->input_dimensions;
    }
    if (const auto* rk = dynamic_cast<const DenseRKFamilyPhysicalPlan*>(&physical_plan); rk != nullptr) {
        return rk->input_dimensions;
    }
    throw std::logic_error("Dense VALUE physical plan has no input-dimension adapter.");
}

[[nodiscard]] size_t checkedWorkspaceAdd(size_t lhs, size_t rhs, const char* message) {
    if (lhs > std::numeric_limits<size_t>::max() - rhs) throw std::overflow_error(message);
    return lhs + rhs;
}

/**
 * Physical storage shape emitted by one non-terminal VALUE pass.
 *
 * DenseReductionProblem intentionally canonicalizes/merges adjacent logical runs after every pass. That canonical
 * shape is a planner representation, not necessarily the shape required by the physical kernel that emits the
 * intermediate. In particular, staged RK kernels always materialize [outer, shards, inner] FP32 partials and a later
 * pass is free to reshape that same dense storage to its own canonical input dimensions.
 */
[[nodiscard]] std::vector<uint64_t> densePhysicalIntermediateDimensions(
    const DenseReductionPhysicalPlan& physical_plan,
    const CubReductionGeometry& geometry) {
    if (const auto* rk = dynamic_cast<const DenseRKFamilyPhysicalPlan*>(&physical_plan); rk != nullptr) {
        if (rk->progress == DenseRKProgress::Staged) {
            return {rk->outer_size, rk->shards_per_output, rk->inner_size};
        }
    }
    return geometry.output_dimensions;
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
        const std::vector<uint64_t> input_dimensions = densePhysicalPassInputDimensions(*candidate.physical_plan);
        current_descriptor = TensorDescriptor(current_descriptor.getDataType(), input_dimensions);
        const CubReductionGeometry stage_geometry = densePhysicalPassGeometry(*candidate.physical_plan);
        const bool final_pass = !hasDenseReductionSites(candidate.next_problem);
        const DataType stage_output_dtype = final_pass ? output_dtype : DataType::FP32;
        const std::vector<uint64_t> stage_output_dimensions =
            final_pass ? stage_geometry.output_dimensions
                       : densePhysicalIntermediateDimensions(*candidate.physical_plan, stage_geometry);
        const TensorDescriptor stage_output_descriptor(stage_output_dtype, stage_output_dimensions);

        if (!final_pass) {
            workspace_bytes = checkedWorkspaceAdd(workspace_bytes,
                                                   stage_output_descriptor.getArraySizeInBytes(),
                                                   "Dense VALUE intermediate workspace size overflows size_t.");
        }

        if (std::dynamic_pointer_cast<const DenseDirectReductionPhysicalPlan>(candidate.physical_plan)) {
            const CubReductionStageSemantics stage_semantics =
                makeValueReductionStageSemantics(op,
                                                 deriveDensePassRole(step.input_problem, candidate.next_problem),
                                                 semantic_total_reduction_size);
            const size_t scratch_bytes = queryReductionBytes(stage_semantics,
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
                                                   "Dense VALUE direct scratch workspace size overflows size_t.");
        }
        current_descriptor = stage_output_descriptor;
    }
    return workspace_bytes;
}

[[nodiscard]] DenseExecutablePlan materializeDenseValueExecutablePlan(CubReductionOp op,
                                                                       const Tensor& input,
                                                                       const Tensor& output,
                                                                       const DenseReductionPlan& plan,
                                                                       uint64_t semantic_total_reduction_size,
                                                                       float output_scale,
                                                                       const Stream& stream) {
    DenseExecutablePlan executable;
    executable.passes.reserve(plan.steps.size());
    Tensor current_input = input;

    for (const DenseReductionPlanStep& step : plan.steps) {
        const DenseReductionCandidate& candidate = step.selected_candidate;
        if (!candidate.physical_plan) throw std::logic_error("Dense VALUE plan step is missing a physical plan.");
        const std::vector<uint64_t> input_dimensions = densePhysicalPassInputDimensions(*candidate.physical_plan);
        current_input.reshape(input_dimensions);
        CubReductionGeometry stage_geometry = densePhysicalPassGeometry(*candidate.physical_plan);
        const DensePassRole role = deriveDensePassRole(step.input_problem, candidate.next_problem);
        const bool final_pass = !hasDenseReductionSites(candidate.next_problem);

        Tensor stage_output =
            final_pass
                ? output
                : Tensor(input.getPlacement(),
                         TensorDescriptor(DataType::FP32,
                                          densePhysicalIntermediateDimensions(*candidate.physical_plan, stage_geometry)));
        const size_t intermediate_bytes = final_pass ? 0 : stage_output.getArraySizeInBytes();

        size_t temp_storage_bytes = 0;
        Tensor temp_storage;
        if (std::dynamic_pointer_cast<const DenseDirectReductionPhysicalPlan>(candidate.physical_plan)) {
            const CubReductionStageSemantics stage_semantics =
                makeValueReductionStageSemantics(op, role, semantic_total_reduction_size);
            temp_storage_bytes = queryReductionBytes(stage_semantics,
                                                      current_input.getDataType(),
                                                      current_input.getMemPtr<void>(),
                                                      current_input.getTotalNumElements(),
                                                      stage_output.getDataType(),
                                                      stage_output.getMemPtr<void>(),
                                                      stage_geometry,
                                                      final_pass ? output_scale : 1.0f,
                                                      stream);
            temp_storage = Tensor(input.getPlacement(),
                                  TensorDescriptor(DataType::UINT8, {std::max<size_t>(1, temp_storage_bytes)}));
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
        executable.workspace_size_bytes = checkedWorkspaceAdd(executable.workspace_size_bytes,
                                                               pass.workspaceBytes(),
                                                               "Dense executable VALUE workspace size overflows size_t.");
        executable.passes.push_back(std::move(pass));
        current_input = stage_output;
    }

    if (executable.passes.empty()) throw std::logic_error("Dense VALUE planner produced no executable passes.");
    return executable;
}

void launchArgReduction(CubArgReductionOp op,
                        const Tensor& temp_storage,
                        size_t temp_storage_bytes,
                        const Tensor& input,
                        Tensor* value_output,
                        Tensor* index_output,
                        const CubReductionGeometry& geometry,
                        Stream& stream) {
    switch (op) {
        case CubArgReductionOp::ArgMin:
            CubReductionInternal::launchArgMinReduction(
                temp_storage, temp_storage_bytes, input, value_output, index_output, geometry, stream);
            return;
        case CubArgReductionOp::ArgMax:
            CubReductionInternal::launchArgMaxReduction(
                temp_storage, temp_storage_bytes, input, value_output, index_output, geometry, stream);
            return;
    }
    throw std::invalid_argument("Unsupported CUB arg reduction operation.");
}

size_t queryComposedArgReductionStageBytes(CubArgReductionOp op,
                                           const Tensor& value_input,
                                           const Tensor* carried_index_input,
                                           Tensor* value_output,
                                           Tensor* index_output,
                                           const CubReductionGeometry& geometry,
                                           uint64_t domain_stride,
                                           DataType carried_index_dtype,
                                           const Stream& stream) {
    switch (op) {
        case CubArgReductionOp::ArgMin:
            return CubReductionInternal::queryComposedArgMinReductionStageBytes(value_input,
                                                                                carried_index_input,
                                                                                value_output,
                                                                                index_output,
                                                                                geometry,
                                                                                domain_stride,
                                                                                carried_index_dtype,
                                                                                stream);
        case CubArgReductionOp::ArgMax:
            return CubReductionInternal::queryComposedArgMaxReductionStageBytes(value_input,
                                                                                carried_index_input,
                                                                                value_output,
                                                                                index_output,
                                                                                geometry,
                                                                                domain_stride,
                                                                                carried_index_dtype,
                                                                                stream);
    }
    throw std::invalid_argument("Unsupported CUB arg reduction operation.");
}

size_t queryComposedArgReductionStageBytes(CubArgReductionOp op,
                                           DataType value_input_dtype,
                                           bool has_carried_index_input,
                                           std::optional<DataType> value_output_dtype,
                                           std::optional<DataType> index_output_dtype,
                                           const CubReductionGeometry& geometry,
                                           uint64_t domain_stride,
                                           DataType carried_index_dtype,
                                           const Stream& stream) {
    switch (op) {
        case CubArgReductionOp::ArgMin:
            return CubReductionInternal::queryComposedArgMinReductionStageBytes(value_input_dtype,
                                                                                has_carried_index_input,
                                                                                value_output_dtype,
                                                                                index_output_dtype,
                                                                                geometry,
                                                                                domain_stride,
                                                                                carried_index_dtype,
                                                                                stream);
        case CubArgReductionOp::ArgMax:
            return CubReductionInternal::queryComposedArgMaxReductionStageBytes(value_input_dtype,
                                                                                has_carried_index_input,
                                                                                value_output_dtype,
                                                                                index_output_dtype,
                                                                                geometry,
                                                                                domain_stride,
                                                                                carried_index_dtype,
                                                                                stream);
    }
    throw std::invalid_argument("Unsupported CUB arg reduction operation.");
}

void launchComposedArgReductionStage(CubArgReductionOp op,
                                     const Tensor& temp_storage,
                                     size_t temp_storage_bytes,
                                     const Tensor& value_input,
                                     const Tensor* carried_index_input,
                                     Tensor* value_output,
                                     Tensor* index_output,
                                     const CubReductionGeometry& geometry,
                                     uint64_t domain_stride,
                                     DataType carried_index_dtype,
                                     Stream& stream) {
    switch (op) {
        case CubArgReductionOp::ArgMin:
            CubReductionInternal::launchComposedArgMinReductionStage(temp_storage,
                                                                     temp_storage_bytes,
                                                                     value_input,
                                                                     carried_index_input,
                                                                     value_output,
                                                                     index_output,
                                                                     geometry,
                                                                     domain_stride,
                                                                     carried_index_dtype,
                                                                     stream);
            return;
        case CubArgReductionOp::ArgMax:
            CubReductionInternal::launchComposedArgMaxReductionStage(temp_storage,
                                                                     temp_storage_bytes,
                                                                     value_input,
                                                                     carried_index_input,
                                                                     value_output,
                                                                     index_output,
                                                                     geometry,
                                                                     domain_stride,
                                                                     carried_index_dtype,
                                                                     stream);
            return;
    }
    throw std::invalid_argument("Unsupported CUB arg reduction operation.");
}

}  // namespace

CubReduction::CubReduction(CubReductionOp op,
                           uint32_t axis,
                           std::optional<DataType> output_dtype,
                           float output_scale)
    : CubReduction(op, std::vector<uint32_t>{axis}, output_dtype, output_scale) {}

CubReduction::CubReduction(CubReductionOp op,
                           std::vector<uint32_t> axes,
                           std::optional<DataType> output_dtype,
                           float output_scale)
    : op(op), axes(std::move(axes)), output_dtype(output_dtype), output_scale(output_scale) {
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

size_t CubReduction::queryPhysicalStageWorkspace(CubReductionOp op,
                                                 const TensorDescriptor& input_descriptor,
                                                 DataType output_dtype,
                                                 const CubReductionGeometry& geometry,
                                                 bool apply_input_transform,
                                                 bool apply_output_finalize,
                                                 uint64_t semantic_total_reduction_size,
                                                 float output_scale,
                                                 const Stream& stream) {
    const CubReductionInternal::CubReductionStageSemantics semantics =
        CubReductionInternal::makeValueReductionStageSemantics(
            op,
            physicalStageRole(apply_input_transform, apply_output_finalize),
            semantic_total_reduction_size);

    return queryReductionBytes(semantics,
                               input_descriptor.getDataType(),
                               nullptr,
                               input_descriptor.getTotalNumElements(),
                               output_dtype,
                               nullptr,
                               geometry,
                               output_scale,
                               stream);
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
                                                                 stream);
        return queryDenseValueExecutableWorkspace(op,
                                                  input_descriptor,
                                                  resolved_output_dtype,
                                                  plan,
                                                  geometry.reduction_size,
                                                  output_scale,
                                                  stream);
    }

    return queryPhysicalStageWorkspace(op,
                                       input_descriptor,
                                       resolved_output_dtype,
                                       geometry,
                                       true,
                                       true,
                                       geometry.reduction_size,
                                       output_scale,
                                       stream);
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
            // DENSE-GATE-FINAL: ComposedDense remains the structural classification for ordinary dense disjoint
            // reduced runs. The structural interval planner proves that the mask is reducible through direct dense
            // stages, but VALUE execution no longer consumes that interval plan: planDenseReduction() globally
            // selects R/KR/RK passes and replans after every pass. ARG still uses the transitional interval executor
            // until DENSE-ARG-CUTOVER.
            const std::optional<CubReductionDenseCompositionPlan> structural_plan =
                makeStructuralDenseCompositionPlan(geometry, input_dimensions);
            if (!structural_plan.has_value()) {
                throw std::logic_error(
                    "DENSE-GATE-FINAL invariant violated: dense disjoint reduction has no all-direct composition "
                    "plan.");
            }
            geometry.path = CubReductionPath::ComposedDense;
        }
        requireOrdainedDenseReductionPath(geometry);
    } else if (reduces_to_single_output && geometry.physical_layout_is_dense_permutation) {
        // VIEW-DIRECT-1: when every logical axis is reduced, a compact physical permutation is just one contiguous
        // physical value domain. Value reductions are insensitive to the logical visitation order, so feed that span
        // directly to CUB DeviceReduce rather than reconstructing one logical coordinate per scalar through the
        // arbitrary-view mapper. input.getMemPtr() already points at the view's storage offset, and a proven dense
        // physical permutation contains each logical value exactly once in input_elements contiguous elements.
        geometry.path = CubReductionPath::DeviceTransformReduce;
    } else if (geometry.permutation_aware_contiguous_segments) {
        // VIEW-DIRECT-2A: the non-singleton reduced axes are the complete physical suffix and the physical retained
        // prefix is already in Thor's logical dense output order. Each retained output is therefore one ordinary
        // contiguous segment in storage, so use CUB DeviceSegmentedReduce directly with no logical input mapper.
        // Cases that need retained-output permutation deliberately remain for VIEW-DIRECT-2B.
        geometry.path = CubReductionPath::ContiguousFixedSegment;
    } else if (geometry.permutation_aware_tiled_geometry.has_value()) {
        // A logical permutation view over physically dense storage can reuse the tuned dense tiled family directly.
        // The candidate supplies the physical [outer,reduction,inner] traversal and whether the final dense retained
        // output is natural [outer,inner] or the [inner,outer] rotation. No input or output intermediate is required.
        activatePermutationAwareTiledCandidate(geometry);
    } else if (geometry.payload_transpose_tiled_geometry.has_value()) {
        // VIEW-DIRECT-2B: the compact source is physically [A,reduction,B,payload] while Thor's dense retained output
        // is [B,A,payload]. A dedicated 32x33 shared-memory strategy preserves coalesced payload reads and writes the
        // logical output directly, with no mixed-radix mapper or global transpose intermediate.
        activatePayloadTransposeTiledCandidate(geometry);
    } else if (reduces_to_single_output && input_dimensions.size() == 1) {
        // Rank-1 views are always affine, including diagonals (stride > 1) and
        // broadcast aliases (stride == 0).  Keep the single-output CUB fast
        // path and avoid the general logical-coordinate metadata/indexing.
        geometry.path = CubReductionPath::DeviceTransformReduce;
        geometry.device_transform_uses_affine_stride = true;
        geometry.affine_input_stride = input_strides[0];
    } else if (geometry.pitched_tiled_geometry.has_value()) {
        // VIEW-PITCHED-TILED: a non-dense logical [outer,reduction,inner] view with a contiguous inner payload can be
        // addressed with two affine pitches. The dedicated pitched kernel keeps adjacent inner components coalesced
        // and performs no per-scalar logical-coordinate reconstruction. It is intentionally separate from the sealed
        // dense/permutation tiled kernels.
        activatePitchedTiledCandidate(geometry);
    } else {
        // DELETE: Thor no longer has an arbitrary per-scalar logical-index reduction backend. Every real workload must
        // be owned by an explicit physical-layout-aware reducer; exotic views fail rather than silently executing a
        // catastrophically slow catch-all implementation.
        throw NotImplementedException(
            "Unsupported CUB tensor-reduction geometry: no ordained production reducer owns this layout. "
            "Materialize/reorder the view or add an explicit high-efficiency reduction strategy for this geometry.");
    }

    // The fixed-segment int limit belongs only to paths that actually execute the CUB fixed-size segmented
    // primitive. Thor-owned tiled/composed stages carry uint64_t geometry.
    if (usesCubFixedSegmentSize(geometry.path)
        && geometry.reduction_size > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("CUB fixed-size segmented reduction segment size exceeds its int limit.");
    }

    return geometry;
}

CubReductionGeometry CubReduction::analyzeValueGeometry(CubReductionOp op,
                                                          const std::vector<uint64_t>& input_dimensions,
                                                          const std::vector<uint32_t>& axes) {
    requireSupportedOperation(op);
    CubReductionGeometry geometry = analyzeGeometry(input_dimensions, axes);
    requireExecutableFixedSegmentSize(geometry);
    return geometry;
}

CubReductionGeometry CubReduction::analyzeValueGeometry(CubReductionOp op,
                                                          const std::vector<uint64_t>& input_dimensions,
                                                          const std::vector<uint64_t>& input_strides,
                                                          const std::vector<uint32_t>& axes) {
    requireSupportedOperation(op);
    CubReductionGeometry geometry = analyzeGeometry(input_dimensions, input_strides, axes);
    requireExecutableFixedSegmentSize(geometry);
    return geometry;
}

std::optional<CubReductionDenseCompositionPlan> CubReduction::analyzeDenseCompositionPlan(
    const std::vector<uint64_t>& input_dimensions,
    const std::vector<uint32_t>& axes) {
    const CubReductionGeometry geometry = analyzeGeometry(input_dimensions, axes);
    return makeStructuralDenseCompositionPlan(geometry, input_dimensions);
}

std::shared_ptr<StampedCubReduction> CubReduction::stamp(const Tensor& input, const Stream& stream) const {
    requireGpuTensorView(input, "input");
    requireCompatibleStream(input, stream);
    const CubReductionGeometry geometry =
        analyzeValueGeometry(op, input.getDimensions(), input.getStridesElements(), axes);
    const DataType resolved_output_dtype = resolveOutputDataType(input.getDataType());
    Tensor output(input.getPlacement(), TensorDescriptor(resolved_output_dtype, geometry.output_dimensions));
    return stampValidated(input, output, geometry, stream);
}

std::shared_ptr<StampedCubReduction> CubReduction::stampPhysicalStage(
    CubReductionOp op,
    const Tensor& input,
    const Tensor& output,
    CubReductionGeometry geometry,
    bool apply_input_transform,
    bool apply_output_finalize,
    uint64_t semantic_total_reduction_size,
    float output_scale,
    const Stream& stream) {
    requireExpectedOutput(input, output, output.getDataType(), geometry);

    const CubReductionInternal::CubReductionStageSemantics semantics =
        CubReductionInternal::makeValueReductionStageSemantics(
            op,
            physicalStageRole(apply_input_transform, apply_output_finalize),
            semantic_total_reduction_size);

    Tensor mutable_output = output;
    const size_t temp_storage_bytes = queryReductionBytes(semantics,
                                                          input.getDataType(),
                                                          input.getMemPtr<void>(),
                                                          input.getTotalNumElements(),
                                                          mutable_output.getDataType(),
                                                          mutable_output.getMemPtr<void>(),
                                                          geometry,
                                                          output_scale,
                                                          stream);
    Tensor temp_storage(
        input.getPlacement(), TensorDescriptor(DataType::UINT8, {static_cast<uint64_t>(temp_storage_bytes)}));
    return std::shared_ptr<StampedCubReduction>(new StampedCubReduction(op,
                                                                        std::move(geometry),
                                                                        input,
                                                                        output,
                                                                        temp_storage_bytes,
                                                                        temp_storage,
                                                                        apply_input_transform,
                                                                        apply_output_finalize,
                                                                        semantic_total_reduction_size,
                                                                        output_scale,
                                                                        stream));
}

std::shared_ptr<StampedCubReduction> CubReduction::stampValidated(const Tensor& input,
                                                                  const Tensor& output,
                                                                  const CubReductionGeometry& geometry,
                                                                  const Stream& stream) const {
    requireSupportedFloatingStorageDType(input.getDataType(), "input");
    requireSupportedFloatingStorageDType(output.getDataType(), "output");
    requireExpectedOutput(input, output, output.getDataType(), geometry);

    ScopedGpu scoped_gpu(stream.getGpuNum());

    if (isOrdinaryDenseValueGeometry(geometry) && input.isDenseContiguous()) {
        std::string strategy_chain;
        const DenseReductionPlan plan = planProductionDenseValue(op,
                                                                 input.getDimensions(),
                                                                 geometry.axes,
                                                                 input.getDataType(),
                                                                 output.getDataType(),
                                                                 stream,
                                                                 &strategy_chain);
        DenseExecutablePlan executable_plan = materializeDenseValueExecutablePlan(op,
                                                                                  input,
                                                                                  output,
                                                                                  plan,
                                                                                  geometry.reduction_size,
                                                                                  output_scale,
                                                                                  stream);
        return std::shared_ptr<StampedCubReduction>(new StampedCubReduction(op,
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
    }

    if (geometry.path == CubReductionPath::ComposedDense) {
        throw std::logic_error(
            "Ordinary dense ComposedDense VALUE geometry must execute through DenseExecutablePlan.");
    }

    return stampPhysicalStage(op,
                              input,
                              output,
                              geometry,
                              true,
                              true,
                              geometry.reduction_size,
                              output_scale,
                              stream);
}

std::shared_ptr<StampedCubReduction> CubReduction::stamp(const Tensor& input,
                                                         const Tensor& preallocated_output,
                                                         const Stream& stream) const {
    requireGpuTensorView(input, "input");
    requireCompatibleStream(input, stream);
    const CubReductionGeometry geometry =
        analyzeValueGeometry(op, input.getDimensions(), input.getStridesElements(), axes);
    const DataType resolved_output_dtype = resolveOutputDataType(input.getDataType());
    requireExpectedOutput(input, preallocated_output, resolved_output_dtype, geometry);
    return stampValidated(input, preallocated_output, geometry, stream);
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
    Tensor mutable_output = output;
    const size_t temp_storage_bytes = CubReductionInternal::queryOffsetSegmentedReductionBytes(
        op, input, mutable_output, segment_offsets, num_items, num_segments, stream);
    Tensor temp_storage(
        input.getPlacement(), TensorDescriptor(DataType::UINT8, {static_cast<uint64_t>(temp_storage_bytes)}));
    return std::shared_ptr<StampedCubSegmentedReduction>(new StampedCubSegmentedReduction(op,
                                                                                         input,
                                                                                         output,
                                                                                         segment_offsets,
                                                                                         num_items,
                                                                                         num_segments,
                                                                                         temp_storage_bytes,
                                                                                         temp_storage,
                                                                                         stream));
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
    CubReductionInternal::launchOffsetSegmentedReduction(op,
                                                         temp_storage,
                                                         temp_storage_bytes,
                                                         input,
                                                         output,
                                                         segment_offsets,
                                                         num_items,
                                                         num_segments,
                                                         run_stream);
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
    Tensor mutable_output = index_output;
    const size_t temp_storage_bytes = CubReductionInternal::queryOffsetSegmentedArgReductionBytes(
        op, input, mutable_output, segment_offsets, num_segments, stream);
    Tensor temp_storage(input.getPlacement(),
                        TensorDescriptor(DataType::UINT8, {static_cast<uint64_t>(temp_storage_bytes)}));
    return std::shared_ptr<StampedCubSegmentedArgReduction>(new StampedCubSegmentedArgReduction(op,
                                                                                                input,
                                                                                                index_output,
                                                                                                segment_offsets,
                                                                                                num_segments,
                                                                                                temp_storage_bytes,
                                                                                                temp_storage,
                                                                                                stream));
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
    CubReductionInternal::launchOffsetSegmentedArgReduction(
        op, temp_storage, temp_storage_bytes, input, index_output, segment_offsets, num_segments, run_stream);
}

CubReductionGeometry CubArgReduction::analyzeDenseGeometry(
    const std::vector<uint64_t>& input_dimensions,
    const std::vector<uint32_t>& axes) {
    CubReductionGeometry geometry = CubReduction::analyzeGeometry(input_dimensions, axes);
    // DENSE-GATE-FINAL: this public helper accepts only canonical dense geometry, so returning the legacy arbitrary
    // view family would be an internal architecture violation rather than a recoverable execution choice.
    requireOrdainedDenseReductionPath(geometry);
    requireExecutableFixedSegmentSize(geometry);
    return geometry;
}

std::optional<CubArgReductionDenseCompositionPlan> CubArgReduction::analyzeDenseCompositionPlan(
    const std::vector<uint64_t>& input_dimensions,
    const std::vector<uint32_t>& axes) {
    const CubReductionGeometry geometry = CubReduction::analyzeGeometry(input_dimensions, axes);
    return makeDenseArgCompositionPlan(geometry, input_dimensions);
}

std::optional<CubArgReductionDenseCompositionPlan> CubArgReduction::analyzeDenseCompositionPlanForExecution(
    const std::vector<uint64_t>& input_dimensions,
    const std::vector<uint32_t>& axes,
    DataType input_dtype,
    const CubArgReductionOutputOptions& outputs,
    const Stream& stream) {
    const CubReductionGeometry geometry = CubReduction::analyzeGeometry(input_dimensions, axes);
    if (!geometry.dense_run_geometry.has_value()) {
        return std::nullopt;
    }
    const DataType carried_index_dtype =
        geometry.reduction_size <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())
            ? DataType::UINT32
            : DataType::UINT64;
    const DataType value_output_dtype = outputs.value_output_dtype.value_or(input_dtype);
    const DenseArgCompositionCostContext cost_context = denseArgCompositionCostContext(
        input_dtype, carried_index_dtype, outputs, value_output_dtype, stream);
    return makeDenseArgCompositionPlan(geometry, input_dimensions, cost_context);
}


std::optional<CubArgReductionDenseCompositionPlan> CubArgReduction::analyzeDenseCompositionPlanForRunOrder(
    const std::vector<uint64_t>& input_dimensions,
    const std::vector<uint32_t>& axes,
    const std::vector<uint32_t>& run_order) {
    const CubReductionGeometry geometry = CubReduction::analyzeGeometry(input_dimensions, axes);
    const std::optional<CubReductionDenseCompositionPlan> topology =
        makeDenseIntervalCompositionPlanForRunOrder(geometry, input_dimensions, run_order);
    if (!topology.has_value()) {
        return std::nullopt;
    }
    CubArgReductionDenseCompositionPlan plan;
    plan.topology = topology.value();
    plan.carried_index_dtype =
        geometry.reduction_size <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())
            ? DataType::UINT32
            : DataType::UINT64;
    return plan;
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
                                 CubArgReductionOutputOptions outputs)
    : CubArgReduction(op, std::vector<uint32_t>{axis}, std::move(outputs)) {}

CubArgReduction::CubArgReduction(CubArgReductionOp op,
                                 std::vector<uint32_t> axes,
                                 CubArgReductionOutputOptions outputs)
    : op(op), axes(std::move(axes)), outputs(std::move(outputs)) {
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

    if (geometry.path != CubReductionPath::ComposedDense) {
        requireExecutableFixedSegmentSize(geometry);
        return queryArgReductionBytes(
            op, input_descriptor.getDataType(), value_output_dtype, index_output_dtype, geometry, stream);
    }

    const std::optional<CubArgReductionDenseCompositionPlan> plan = analyzeDenseCompositionPlanForExecution(
        input_descriptor.getDimensions(), axes, input_descriptor.getDataType(), outputs, stream);
    if (!plan.has_value()) {
        throw std::logic_error("Composed dense ARG geometry is missing its production composition plan.");
    }

    size_t workspace_size_bytes = 0;
    auto add_workspace_bytes = [&](uint64_t bytes) {
        if (bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max() - workspace_size_bytes)) {
            throw std::overflow_error("CUB arg reduction workspace size overflows size_t.");
        }
        workspace_size_bytes += static_cast<size_t>(bytes);
    };

    DataType current_value_dtype = input_descriptor.getDataType();
    for (size_t stage_index = 0; stage_index < plan->topology.stages.size(); ++stage_index) {
        const CubReductionDenseCompositionStage& stage_plan = plan->topology.stages[stage_index];
        const bool is_first_stage = stage_index == 0;
        const bool is_final_stage = stage_index + 1 == plan->topology.stages.size();
        CubReductionGeometry stage_geometry =
            CubReduction::analyzeGeometry(stage_plan.input_dimensions, stage_plan.reduction_axes);
        if (stage_geometry.path != stage_plan.expected_path || !isDirectDenseReductionPath(stage_geometry.path)) {
            throw std::logic_error("Composed dense CUB arg stage did not resolve to its planned direct reducer family.");
        }
        requireExecutableFixedSegmentSize(stage_geometry);
        if (stage_geometry.reduction_size != stage_plan.reduction_extent
            || stage_geometry.input_elements != stage_plan.input_elements
            || stage_geometry.output_elements != stage_plan.output_elements
            || stage_geometry.outer_size != stage_plan.outer_size
            || stage_geometry.inner_size != stage_plan.inner_size
            || stage_geometry.output_dimensions != stage_plan.output_dimensions) {
            throw std::logic_error("Composed dense CUB arg stage geometry changed after planning.");
        }

        const std::optional<DataType> stage_value_output_dtype =
            is_final_stage ? value_output_dtype : std::optional<DataType>(DataType::FP32);
        const std::optional<DataType> stage_index_output_dtype =
            is_final_stage ? index_output_dtype : std::optional<DataType>(plan->carried_index_dtype);
        add_workspace_bytes(queryComposedArgReductionStageBytes(op,
                                                                 current_value_dtype,
                                                                 !is_first_stage,
                                                                 stage_value_output_dtype,
                                                                 stage_index_output_dtype,
                                                                 stage_geometry,
                                                                 stage_plan.domain_stride,
                                                                 plan->carried_index_dtype,
                                                                 stream));

        if (!is_final_stage) {
            add_workspace_bytes(TensorDescriptor::getArraySizeInBytes(stage_geometry.output_elements, DataType::FP32));
            add_workspace_bytes(
                TensorDescriptor::getArraySizeInBytes(stage_geometry.output_elements, plan->carried_index_dtype));
            current_value_dtype = DataType::FP32;
        }
    }
    return workspace_size_bytes;
}

std::shared_ptr<StampedCubArgReduction> CubArgReduction::stamp(const Tensor& input,
                                                               const Stream& stream) const {
    return stamp(input, std::nullopt, std::nullopt, stream);
}

std::shared_ptr<StampedCubArgReduction> CubArgReduction::stamp(
    const Tensor& input,
    const std::optional<Tensor>& preallocated_value_output,
    const std::optional<Tensor>& preallocated_index_output,
    const Stream& stream) const {
    requireDenseContiguousGpuTensor(input, "input");
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
    if (geometry.path == CubReductionPath::ComposedDense) {
        const std::optional<CubArgReductionDenseCompositionPlan> plan =
            analyzeDenseCompositionPlanForExecution(input.getDimensions(), axes, input.getDataType(), outputs, stream);
        if (!plan.has_value()) {
            throw std::logic_error("Composed dense ARG geometry is missing its production composition plan.");
        }
        return stampComposedDenseValidated(
            input, std::move(value_output), std::move(index_output), geometry, plan.value(), stream);
    }
    return stampValidated(input, std::move(value_output), std::move(index_output), geometry, stream);
}

std::shared_ptr<StampedCubArgReduction> CubArgReduction::stampComposedDense(const Tensor& input,
                                                                            const Stream& stream) const {
    return stampComposedDense(input, std::nullopt, std::nullopt, stream);
}

std::shared_ptr<StampedCubArgReduction> CubArgReduction::stampComposedDense(
    const Tensor& input,
    const std::optional<Tensor>& preallocated_value_output,
    const std::optional<Tensor>& preallocated_index_output,
    const Stream& stream) const {
    requireDenseContiguousGpuTensor(input, "input");
    requireCompatibleStream(input, stream);
    requireSupportedFloatingStorageDType(input.getDataType(), "input");

    const CubReductionGeometry geometry = CubReduction::analyzeGeometry(input.getDimensions(), axes);
    const std::optional<CubArgReductionDenseCompositionPlan> plan =
        analyzeDenseCompositionPlan(input.getDimensions(), axes);
    if (!plan.has_value()) {
        throw std::invalid_argument("CUB arg reduction cannot compose the requested dense reduction geometry.");
    }
    if (outputs.produce_index && outputs.index_output_dtype == DataType::UINT32
        && geometry.reduction_size > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::invalid_argument("CUB arg reduction domain does not fit in a UINT32 public index.");
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
    return stampComposedDenseValidated(
        input, std::move(value_output), std::move(index_output), geometry, plan.value(), stream);
}


std::shared_ptr<StampedCubArgReduction> CubArgReduction::stampComposedDenseWithPlan(
    const Tensor& input,
    const CubArgReductionDenseCompositionPlan& plan,
    const Stream& stream) const {
    requireDenseContiguousGpuTensor(input, "input");
    requireCompatibleStream(input, stream);
    requireSupportedFloatingStorageDType(input.getDataType(), "input");

    const CubReductionGeometry geometry = CubReduction::analyzeGeometry(input.getDimensions(), axes);
    if (!geometry.dense_run_geometry.has_value() || plan.topology.stages.empty()) {
        throw std::invalid_argument("CUB arg reduction received an invalid explicit dense composition plan.");
    }
    const DataType expected_carried_index_dtype =
        geometry.reduction_size <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())
            ? DataType::UINT32
            : DataType::UINT64;
    if (plan.carried_index_dtype != expected_carried_index_dtype) {
        throw std::invalid_argument("CUB arg reduction explicit plan uses the wrong carried-index width.");
    }
    if (plan.topology.stages.front().input_dimensions != input.getDimensions()
        || plan.topology.stages.back().output_dimensions != geometry.output_dimensions) {
        throw std::invalid_argument("CUB arg reduction explicit plan does not match the requested geometry.");
    }
    if (outputs.produce_index && outputs.index_output_dtype == DataType::UINT32
        && geometry.reduction_size > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::invalid_argument("CUB arg reduction domain does not fit in a UINT32 public index.");
    }

    std::optional<Tensor> value_output = std::nullopt;
    if (outputs.produce_value) {
        value_output = Tensor(input.getPlacement(),
                              TensorDescriptor(resolveValueOutputDataType(input.getDataType()),
                                               geometry.output_dimensions));
    }
    std::optional<Tensor> index_output = std::nullopt;
    if (outputs.produce_index) {
        index_output = Tensor(input.getPlacement(),
                              TensorDescriptor(outputs.index_output_dtype, geometry.output_dimensions));
    }
    requireArgOutputsDoNotOverlap(value_output, index_output);
    return stampComposedDenseValidated(
        input, std::move(value_output), std::move(index_output), geometry, plan, stream);
}

std::shared_ptr<StampedCubArgReduction> CubArgReduction::stampComposedDenseValidated(
    const Tensor& input,
    std::optional<Tensor> value_output,
    std::optional<Tensor> index_output,
    const CubReductionGeometry& geometry,
    const CubArgReductionDenseCompositionPlan& plan,
    const Stream& stream) const {
    if (plan.topology.stages.empty()) {
        throw std::logic_error("Composed dense CUB arg reduction has no stages.");
    }
    requireSupportedArgIndexDType(plan.carried_index_dtype);

    ScopedGpu scoped_gpu(stream.getGpuNum());
    std::vector<std::shared_ptr<StampedCubArgReduction>> composed_stages;
    composed_stages.reserve(plan.topology.stages.size());

    Tensor current_value_input = input;
    std::optional<Tensor> current_index_input = std::nullopt;
    size_t workspace_size_bytes = 0;

    for (size_t stage_index = 0; stage_index < plan.topology.stages.size(); ++stage_index) {
        const CubReductionDenseCompositionStage& stage_plan = plan.topology.stages[stage_index];
        const bool is_first_stage = stage_index == 0;
        const bool is_final_stage = stage_index + 1 == plan.topology.stages.size();

        if (current_value_input.getDimensions() != stage_plan.input_dimensions) {
            throw std::logic_error("Composed dense CUB arg stage input dimensions do not match the stamped plan.");
        }
        if (is_first_stage != !current_index_input.has_value()) {
            throw std::logic_error("Composed dense CUB arg stage has an invalid carried-index input state.");
        }
        if (!is_first_stage) {
            if (current_value_input.getDataType() != DataType::FP32
                || current_index_input->getDataType() != plan.carried_index_dtype
                || current_index_input->getDimensions() != current_value_input.getDimensions()) {
                throw std::logic_error("Composed dense CUB arg stage received an invalid SoA pair intermediate.");
            }
        }

        CubReductionGeometry stage_geometry =
            CubReduction::analyzeGeometry(current_value_input.getDimensions(), stage_plan.reduction_axes);
        if (stage_geometry.path != stage_plan.expected_path || !isDirectDenseReductionPath(stage_geometry.path)) {
            throw std::logic_error("Composed dense CUB arg stage did not resolve to its planned direct reducer family.");
        }
        requireExecutableFixedSegmentSize(stage_geometry);
        if (stage_geometry.reduction_size != stage_plan.reduction_extent
            || stage_geometry.input_elements != stage_plan.input_elements
            || stage_geometry.output_elements != stage_plan.output_elements
            || stage_geometry.outer_size != stage_plan.outer_size
            || stage_geometry.inner_size != stage_plan.inner_size
            || stage_geometry.output_dimensions != stage_plan.output_dimensions) {
            throw std::logic_error("Composed dense CUB arg stage geometry changed after planning.");
        }

        std::optional<Tensor> stage_value_output;
        std::optional<Tensor> stage_index_output;
        if (is_final_stage) {
            stage_value_output = value_output;
            stage_index_output = index_output;
        } else {
            stage_value_output = Tensor(
                input.getPlacement(), TensorDescriptor(DataType::FP32, stage_plan.output_dimensions));
            stage_index_output = Tensor(
                input.getPlacement(), TensorDescriptor(plan.carried_index_dtype, stage_plan.output_dimensions));
        }

        Tensor* stage_value_output_ptr =
            stage_value_output.has_value() ? &stage_value_output.value() : nullptr;
        Tensor* stage_index_output_ptr =
            stage_index_output.has_value() ? &stage_index_output.value() : nullptr;
        const Tensor* carried_index_input_ptr =
            current_index_input.has_value() ? &current_index_input.value() : nullptr;
        const size_t temp_storage_bytes = queryComposedArgReductionStageBytes(op,
                                                                              current_value_input,
                                                                              carried_index_input_ptr,
                                                                              stage_value_output_ptr,
                                                                              stage_index_output_ptr,
                                                                              stage_geometry,
                                                                              stage_plan.domain_stride,
                                                                              plan.carried_index_dtype,
                                                                              stream);
        Tensor temp_storage(
            input.getPlacement(), TensorDescriptor(DataType::UINT8, {static_cast<uint64_t>(temp_storage_bytes)}));

        std::shared_ptr<StampedCubArgReduction> stamped_stage(
            new StampedCubArgReduction(op,
                                       std::move(stage_geometry),
                                       current_value_input,
                                       current_index_input,
                                       stage_value_output,
                                       stage_index_output,
                                       temp_storage_bytes,
                                       temp_storage,
                                       stage_plan.domain_stride,
                                       plan.carried_index_dtype,
                                       stream));
        workspace_size_bytes += temp_storage_bytes;

        if (!is_final_stage) {
            THOR_THROW_IF_FALSE(stage_value_output.has_value());
            THOR_THROW_IF_FALSE(stage_index_output.has_value());
            workspace_size_bytes += static_cast<size_t>(stage_value_output->getArraySizeInBytes());
            workspace_size_bytes += static_cast<size_t>(stage_index_output->getArraySizeInBytes());
            current_value_input = stage_value_output.value();
            current_index_input = stage_index_output.value();
        }
        composed_stages.push_back(std::move(stamped_stage));
    }

    CubReductionGeometry composed_geometry = geometry;
    composed_geometry.path = CubReductionPath::ComposedDense;
    return std::shared_ptr<StampedCubArgReduction>(new StampedCubArgReduction(op,
                                                                             std::move(composed_geometry),
                                                                             input,
                                                                             std::move(value_output),
                                                                             std::move(index_output),
                                                                             workspace_size_bytes,
                                                                             std::move(composed_stages),
                                                                             stream));
}

std::shared_ptr<StampedCubArgReduction> CubArgReduction::stampValidated(
    const Tensor& input,
    std::optional<Tensor> value_output,
    std::optional<Tensor> index_output,
    const CubReductionGeometry& geometry,
    const Stream& stream) const {
    ScopedGpu scoped_gpu(stream.getGpuNum());
    Tensor* value_output_ptr = value_output.has_value() ? &value_output.value() : nullptr;
    Tensor* index_output_ptr = index_output.has_value() ? &index_output.value() : nullptr;
    const size_t temp_storage_bytes =
        queryArgReductionBytes(op, input, value_output_ptr, index_output_ptr, geometry, stream);
    Tensor temp_storage(
        input.getPlacement(), TensorDescriptor(DataType::UINT8, {static_cast<uint64_t>(temp_storage_bytes)}));

    return std::shared_ptr<StampedCubArgReduction>(new StampedCubArgReduction(op,
                                                                             geometry,
                                                                             input,
                                                                             std::move(value_output),
                                                                             std::move(index_output),
                                                                             temp_storage_bytes,
                                                                             temp_storage,
                                                                             stream));
}

StampedCubArgReduction::StampedCubArgReduction(CubArgReductionOp op,
                                               CubReductionGeometry geometry,
                                               const Tensor& input,
                                               std::optional<Tensor> value_output,
                                               std::optional<Tensor> index_output,
                                               size_t temp_storage_bytes,
                                               const Tensor& temp_storage,
                                               const Stream& stream)
    : op(op),
      geometry(std::move(geometry)),
      input(input),
      value_output(std::move(value_output)),
      index_output(std::move(index_output)),
      temp_storage_bytes(temp_storage_bytes),
      temp_storage(temp_storage),
      stream(stream) {
    requireTempStorage(this->temp_storage, input.getPlacement(), temp_storage_bytes);
    if (!this->value_output.has_value() && !this->index_output.has_value()) {
        throw std::invalid_argument("Stamped CUB arg reduction requires at least one output.");
    }
}

StampedCubArgReduction::StampedCubArgReduction(CubArgReductionOp op,
                                               CubReductionGeometry geometry,
                                               const Tensor& value_input,
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
    requireTempStorage(this->temp_storage, value_input.getPlacement(), temp_storage_bytes);
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
}

StampedCubArgReduction::StampedCubArgReduction(
    CubArgReductionOp op,
    CubReductionGeometry geometry,
    const Tensor& input,
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
    THOR_THROW_IF_FALSE(this->geometry.path == CubReductionPath::ComposedDense);
    THOR_THROW_IF_FALSE(!this->composed_stages.empty());
    for (const auto& stage : this->composed_stages) {
        THOR_THROW_IF_FALSE(stage != nullptr);
        THOR_THROW_IF_FALSE(stage->geometry.path != CubReductionPath::ComposedDense);
        THOR_THROW_IF_FALSE(stage->composed_carried_index_dtype.has_value());
    }
    if (!this->value_output.has_value() && !this->index_output.has_value()) {
        throw std::invalid_argument("Stamped composed CUB arg reduction requires at least one public output.");
    }
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

    if (geometry.path == CubReductionPath::ComposedDense) {
        THOR_THROW_IF_FALSE(!composed_stages.empty());
        for (const auto& stage : composed_stages) {
            THOR_THROW_IF_FALSE(stage != nullptr);
            stage->runOn(run_stream);
        }
        return;
    }

    Tensor* value_output_ptr = value_output.has_value() ? &value_output.value() : nullptr;
    Tensor* index_output_ptr = index_output.has_value() ? &index_output.value() : nullptr;
    if (composed_carried_index_dtype.has_value()) {
        const Tensor* carried_index_input_ptr =
            carried_index_input.has_value() ? &carried_index_input.value() : nullptr;
        launchComposedArgReductionStage(op,
                                        temp_storage,
                                        temp_storage_bytes,
                                        input,
                                        carried_index_input_ptr,
                                        value_output_ptr,
                                        index_output_ptr,
                                        geometry,
                                        composed_domain_stride,
                                        composed_carried_index_dtype.value(),
                                        run_stream);
        return;
    }

    launchArgReduction(
        op, temp_storage, temp_storage_bytes, input, value_output_ptr, index_output_ptr, geometry, run_stream);
}

StampedCubReduction::StampedCubReduction(CubReductionOp op,
                                         CubReductionGeometry geometry,
                                         const Tensor& input,
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
    requireTempStorage(this->temp_storage, input.getPlacement(), temp_storage_bytes);
}

StampedCubReduction::StampedCubReduction(
    CubReductionOp op,
    CubReductionGeometry geometry,
    const Tensor& input,
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
      temp_storage(input.getPlacement(), TensorDescriptor(DataType::UINT8, {1})),
      dense_executable_plan(std::move(dense_executable_plan)),
      modern_rk_strategy_chain(std::move(modern_rk_strategy_chain)),
      apply_input_transform(apply_input_transform),
      apply_output_finalize(apply_output_finalize),
      semantic_total_reduction_size(semantic_total_reduction_size),
      output_scale(output_scale),
      stream(stream) {
    THOR_THROW_IF_FALSE(isOrdinaryDenseValueGeometry(this->geometry));
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

namespace {

[[nodiscard]] std::shared_ptr<const CubReductionInternal::DenseRKFamilyPhysicalPlan> firstDenseRKPlan(
    const CubReductionInternal::DenseExecutablePlan& executable_plan) {
    if (executable_plan.passes.empty()) {
        return nullptr;
    }
    return std::dynamic_pointer_cast<const CubReductionInternal::DenseRKFamilyPhysicalPlan>(
        executable_plan.passes.front().physical_plan);
}

}  // namespace

bool StampedCubReduction::usesModernRKFamilyPlan() const {
    if (dense_executable_plan.passes.empty()) {
        return false;
    }
    return std::all_of(dense_executable_plan.passes.begin(),
                       dense_executable_plan.passes.end(),
                       [](const CubReductionInternal::DenseExecutablePass& pass) {
                           const auto rk = std::dynamic_pointer_cast<const CubReductionInternal::DenseRKFamilyPhysicalPlan>(
                               pass.physical_plan);
                           return rk != nullptr && CubReductionInternal::denseRKIsFamilyImplementation(rk->implementation);
                       });
}

size_t StampedCubReduction::getModernRKFirstPacketBytes() const {
    const auto plan = firstDenseRKPlan(dense_executable_plan);
    return plan ? plan->packet_bytes : 0;
}

uint32_t StampedCubReduction::getModernRKFirstBlockThreads() const {
    const auto plan = firstDenseRKPlan(dense_executable_plan);
    return plan ? plan->block_threads : 0;
}

uint64_t StampedCubReduction::getModernRKFirstStageBlocks() const {
    const auto plan = firstDenseRKPlan(dense_executable_plan);
    return plan ? plan->first_stage_blocks : 0;
}

uint64_t StampedCubReduction::getModernRKFirstShardsPerOutput() const {
    const auto plan = firstDenseRKPlan(dense_executable_plan);
    return plan ? plan->shards_per_output : 0;
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

void StampedCubReduction::runOn(Stream& run_stream, float runtime_output_scale) const {
    THOR_THROW_IF_FALSE(std::isfinite(runtime_output_scale));
    requireCompatibleStream(input, run_stream);
    ScopedGpu scoped_gpu(run_stream.getGpuNum());

    const CubReductionInternal::CubReductionStageSemantics semantics =
        CubReductionInternal::makeValueReductionStageSemantics(
            op,
            physicalStageRole(apply_input_transform, apply_output_finalize),
            semantic_total_reduction_size);

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

            if (std::dynamic_pointer_cast<const CubReductionInternal::DenseDirectReductionPhysicalPlan>(
                    pass.physical_plan)) {
                launchReduction(stage_semantics,
                                pass.temp_storage,
                                pass.temp_storage_bytes,
                                pass.input,
                                pass.output,
                                pass.geometry,
                                stage_output_scale,
                                run_stream);
                continue;
            }

            const auto rk_plan = std::dynamic_pointer_cast<const CubReductionInternal::DenseRKFamilyPhysicalPlan>(
                pass.physical_plan);
            if (!rk_plan || !CubReductionInternal::denseRKIsFamilyImplementation(rk_plan->implementation)) {
                throw std::logic_error("Dense executable VALUE pass has no registered physical executor.");
            }

            switch (rk_plan->implementation) {
                case CubReductionInternal::DenseRKProductionImplementation::KParallelPass: {
                    const CubReductionInternal::KParallelTiledStagePlan kplan{
                        rk_plan->progress == CubReductionInternal::DenseRKProgress::Complete
                            ? CubReductionInternal::TiledRKStageTopology::Complete
                            : CubReductionInternal::TiledRKStageTopology::Staged,
                        rk_plan->packet_bytes,
                        rk_plan->block_threads,
                        rk_plan->shards_per_output};
                    CubReductionInternal::launchKParallelTiledStage(
                        stage_semantics, pass.input, pass.output, pass.geometry, kplan, stage_output_scale, run_stream);
                    break;
                }
                case CubReductionInternal::DenseRKProductionImplementation::RCooperativeAlignedComplete:
                    // The retained generic TiledFixedSegment entry point now contains only the modern packet-adaptive
                    // aligned completion plus view/generic fallbacks; all historical full-row dispatch is deleted.
                    launchReduction(stage_semantics,
                                    temp_storage,
                                    1,
                                    pass.input,
                                    pass.output,
                                    pass.geometry,
                                    stage_output_scale,
                                    run_stream);
                    break;
                case CubReductionInternal::DenseRKProductionImplementation::RCooperativeAlignedStaged:
                    CubReductionInternal::launchCooperativeShardedTiledFirstStage(stage_semantics,
                                                                                  pass.input,
                                                                                  pass.output,
                                                                                  pass.geometry,
                                                                                  rk_plan->rows_per_shard,
                                                                                  rk_plan->shards_per_output,
                                                                                  run_stream);
                    break;
                case CubReductionInternal::DenseRKProductionImplementation::RCooperativeRotatedStaged:
                    CubReductionInternal::launchAwkwardAlignmentRotatedShardedFirstStage(stage_semantics,
                                                                                           pass.input,
                                                                                           pass.output,
                                                                                           pass.geometry,
                                                                                           rk_plan->rows_per_shard,
                                                                                           rk_plan->shards_per_output,
                                                                                           run_stream);
                    break;
                case CubReductionInternal::DenseRKProductionImplementation::RCooperativeFlatRowsStaged:
                    CubReductionInternal::launchNarrowLowPrecisionFlatRCooperativeFirstStage(stage_semantics,
                                                                                              pass.input,
                                                                                              pass.output,
                                                                                              pass.geometry,
                                                                                              rk_plan->rows_per_shard,
                                                                                              rk_plan->shards_per_output,
                                                                                              run_stream);
                    break;
                default:
                    throw std::logic_error("Stamped modern DenseRK plan contains a non-VALUE family implementation.");
            }
        }
        return;
    }

    launchReduction(
        semantics, temp_storage, temp_storage_bytes, input, output, geometry, runtime_output_scale, run_stream);
}

}  // namespace ThorImplementation
