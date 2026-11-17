// Views the reducer runs in two parts (refused views, sliced rows) and the in-place pitched Complete launch.

#include "Utilities/TensorOperations/Reduction/Layout/ViewSplit.h"

#include "Utilities/Exceptions.h"
#include "Utilities/TensorOperations/Cub/CubDataTypePolicy.h"
#include "Utilities/TensorOperations/Cub/CubDevicePrimitiveSupport.h"
#include "Utilities/TensorOperations/Reduction/FrontEnd.h"
#include "Utilities/TensorOperations/Reduction/Kernels/KernelLaunchers.h"
#include "Utilities/TensorOperations/Reduction/Planner/DenseProductionPhysicalSelectionContext.h"
#include "Utilities/TensorOperations/Reduction/Planner/DenseReductionSelectionCache.h"
#include "Utilities/TensorOperations/Reduction/RK/RKLaunchLimits.h"
#include "Utilities/TensorOperations/Reduction/Reducer.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ThorImplementation {

namespace ReductionFrontEnd {

using namespace CubDevicePrimitiveSupport;
using namespace ReductionInternal;

[[nodiscard]] std::optional<ViewPhysicalDenseProblem> densePermutationRuns(const std::vector<uint64_t>& dimensions,
                                                                          const ReductionGeometry& geometry) {
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
                                                                              const ReductionGeometry& geometry) {
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
                                                                              const ReductionGeometry& geometry) {
    return viewPhysicalDenseProblem(input.getDimensions(), input.isDenseContiguous(), geometry);
}

[[nodiscard]] std::optional<ViewSplit> refusedViewSplit(const std::vector<uint64_t>& dimensions,
                                                        bool dense_contiguous,
                                                        const ReductionGeometry& geometry) {
    if (dense_contiguous || viewPhysicalDenseProblem(dimensions, dense_contiguous, geometry).has_value()) {
        return std::nullopt;
    }
    ViewSplit split;
    uint64_t reduction = 1;
    if (geometry.permuted_layout.has_value() &&
        geometry.permuted_layout->retained_output_order ==
            RetainedOutputOrder::PermutedInnerOuter) {
        const PermutedLayout& rotated = *geometry.permuted_layout;
        split.a = rotated.outer_size;
        split.b = rotated.inner_size;
        reduction = rotated.reduction_size;
    } else if (geometry.payload_transpose_layout.has_value()) {
        const PayloadTransposeLayout& payload = *geometry.payload_transpose_layout;
        split.a = payload.a_size;
        split.b = payload.b_size;
        split.p = payload.payload_size;
        reduction = payload.reduction_size;
    } else if (geometry.pitched_layout.has_value()) {
        split.kind = ViewSplit::Kind::PackedInput;
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
    split.kind = ViewSplit::Kind::RotatedOutput;
    split.dense_dimensions = {split.a, reduction, split.b * split.p};
    split.dense_axes = {1};
    split.buffer_dimensions = {split.a, split.b * split.p};
    split.copy_dimensions = {split.b, split.a, split.p};
    split.copy_strides = {split.p, split.b * split.p, 1};
    return split;
}

[[nodiscard]] StridedSpan pitchedRowsSpan(const StridedSpan& view, const PitchedLayout& pitched) {
    return view.withSourceLayout({pitched.outer_size, pitched.reduction_size, pitched.inner_size},
                                 {pitched.outer_stride, pitched.reduction_stride, 1});
}

void launchPitchedComplete(const PitchedCompleteLaunch& launch,
                           const ReductionInternal::ReductionStageSemantics& semantics,
                           const StridedSpan& rows,
                           const StridedSpan& output,
                           const ReductionGeometry& geometry,
                           float output_scale,
                           Stream& stream) {
    const PitchedLayout& pitched = *geometry.pitched_layout;
    ReductionGeometry launch_geometry = geometry;
    launch_geometry.outer_size = pitched.outer_size;
    launch_geometry.reduction_size = pitched.reduction_size;
    launch_geometry.inner_size = pitched.inner_size;
    launch_geometry.input_elements = pitched.outer_size * pitched.reduction_size * pitched.inner_size;
    launch_geometry.output_dimensions = output.getDimensions();
    switch (rows.getDataType()) {
        case DataType::FP32:
            ReductionInternal::launchFp32PacketComplete(rows, output, launch_geometry, launch.warps_per_output,
                                                        launch.tile_components, stream, semantics, output_scale);
            return;
        case DataType::FP16:
        case DataType::BF16:
            ReductionInternal::launchF16PacketComplete(rows, output, launch_geometry, launch.warps_per_output,
                                                       launch.tile_components, stream, semantics, output_scale);
            return;
        default:
            ReductionInternal::launchFp8PacketComplete(
                rows, output, launch_geometry, launch.warps_per_output, launch.tile_components, stream, semantics,
                output_scale);
            return;
    }
}

}  // namespace ReductionFrontEnd

namespace {

using namespace CubDevicePrimitiveSupport;
using namespace ReductionInternal;
using namespace ReductionFrontEnd;

// A pitched view [O, R, I] (rows of I contiguous elements at row pitch Sr, slabs at pitch So) whose slabs have no gap
// between them (So = R x Sr, or O = 1) is the dense [O, R, Sr] from its first element with the columns [I, Sr) dropped:
// its dense reduction over R into [O, Sr], then the copy of the [O, I] columns. That reads the gap elements, Sr - I
// past the last row too, so it needs that many readable elements of the view's source (`readable_elements`). Chosen
// when it moves fewer bytes than the packed copy: O R Sr b + O Sr c + 2 O I c against 3 O R I b + O I c, b and c the
// input and output element bytes.
// The view's logical axes are in physical order (the pitched classification flattens them in order), so its keepdims
// output is dense [O, I].
[[nodiscard]] std::optional<ViewSplit> slicedRowSplit(const ReductionGeometry& geometry,
                                                      uint64_t readable_elements,
                                                      uint64_t input_bytes,
                                                      uint64_t output_bytes) {
    if (!geometry.pitched_layout.has_value()) return std::nullopt;
    const PitchedLayout& pitched = *geometry.pitched_layout;
    const uint64_t O = pitched.outer_size, R = pitched.reduction_size, I = pitched.inner_size;
    const uint64_t Sr = pitched.reduction_stride;
    if (R < 2 || Sr <= I || (O > 1 && pitched.outer_stride != R * Sr)) return std::nullopt;
    const long double rows = static_cast<long double>(O) * R;
    if (rows * Sr > static_cast<long double>(readable_elements)) return std::nullopt;
    const long double sliced = rows * Sr * input_bytes + static_cast<long double>(O) * (Sr + 2 * I) * output_bytes;
    const long double packed = 3.0L * rows * I * input_bytes + static_cast<long double>(O) * I * output_bytes;
    if (!(sliced < packed)) return std::nullopt;
    ViewSplit split;
    split.kind = ViewSplit::Kind::SlicedOutput;
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

// The PacketComplete tile cap of an input dtype (one 16-byte packet per lane, 31 of the 32 lanes' components), or 0
// when PacketComplete does not read it.
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

}  // namespace

std::optional<ViewSplit> Reducer::splitRefusedView(const std::vector<uint64_t>& input_dimensions,
                                                   const std::vector<uint64_t>& input_strides,
                                                   const std::vector<uint32_t>& axes) {
    try {
        const ReductionGeometry geometry = classifyGeometry(input_dimensions, input_strides, axes);
        return refusedViewSplit(
            input_dimensions, input_strides == StridedSpan::denseStrides(input_dimensions), geometry);
    } catch (const NotImplementedException&) {
        return std::nullopt;  // no reducer owns the layout, in one part or two
    }
}

std::optional<ViewSplit> Reducer::splitRefusedView(const StridedSpan& input,
                                                   const std::vector<uint32_t>& axes,
                                                   DataType output_dtype) {
    return splitRefusedView(input, axes, output_dtype, true);
}

std::optional<ViewSplit> Reducer::splitRefusedView(const StridedSpan& input,
                                                   const std::vector<uint32_t>& axes,
                                                   DataType output_dtype,
                                                   bool reducer_reads_pitched) {
    std::optional<ViewSplit> split =
        splitRefusedView(input.getDimensions(), input.getStridesElements(), axes);
    if (!split.has_value() || split->kind != ViewSplit::Kind::PackedInput) {
        return split;
    }
    if (reducer_reads_pitched && pitchedCompleteLaunch(input, axes, output_dtype).has_value()) {
        return std::nullopt;  // the reducer reads the rows at their pitches
    }
    const ReductionGeometry geometry = classifyGeometry(input.getDimensions(), input.getStridesElements(), axes);
    const auto elementBytes = [](DataType dtype) {
        return std::max<uint64_t>(1, static_cast<uint64_t>(TensorDescriptor::getElementSizeInBytes(dtype)));
    };
    if (std::optional<ViewSplit> sliced = slicedRowSplit(
            geometry, input.sourceElementsFromFirst(), elementBytes(input.getDataType()), elementBytes(output_dtype));
        sliced.has_value()) {
        return sliced;
    }
    return split;
}

std::optional<PitchedCompleteLaunch> Reducer::pitchedCompleteLaunch(
    const StridedSpan& input,
    const std::vector<uint32_t>& axes,
    DataType output_dtype,
    std::optional<uint32_t> warps_per_output) {
    if (warps_per_output.has_value() && *warps_per_output != 1 && *warps_per_output != 2 && *warps_per_output != 4 &&
        *warps_per_output != 8) {
        throw std::invalid_argument("A pitched view's PacketComplete takes 1, 2, 4 or 8 warps per output.");
    }
    if (!input.isInitialized()) return std::nullopt;
    const std::optional<ViewSplit> split =
        splitRefusedView(input.getDimensions(), input.getStridesElements(), axes);
    if (!split.has_value() || split->kind != ViewSplit::Kind::PackedInput) return std::nullopt;
    const ReductionGeometry geometry = classifyGeometry(input.getDimensions(), input.getStridesElements(), axes);
    if (!geometry.pitched_layout.has_value()) return std::nullopt;
    const PitchedLayout& pitched = *geometry.pitched_layout;
    const uint64_t O = pitched.outer_size, R = pitched.reduction_size, I = pitched.inner_size;
    const uint32_t max_tile = pitchedCompleteMaxTile(input.getDataType());
    const uint64_t element_bytes = std::max<uint64_t>(1, TensorDescriptor::getElementSizeInBytes(input.getDataType()));
    if (max_tile == 0 || O == 0 || R == 0 || I == 0 || O > RK::RK_MAX_UINT32 || I > RK::RK_MAX_UINT32 ||
        R > RK::RK_PACKET_COMPLETE_MAX_ROWS ||
        reinterpret_cast<uintptr_t>(input.getMemPtr<void>()) % element_bytes != 0) {
        return std::nullopt;
    }
    // Balanced K tiles: as few as the cap allows, of equal width but the last.
    const uint64_t tiles = (I + max_tile - 1) / max_tile;
    const uint32_t tile = static_cast<uint32_t>((I + tiles - 1) / tiles);
    if (warps_per_output.has_value()) return PitchedCompleteLaunch{*warps_per_output, tile};

    // A warp holds one row's tile in flight (the next row's packets, loaded ahead), so the read rate is the bytes in
    // flight: narrow tiles starve it (in the audit, no tile under 256 bytes was faster in place at any W).
    constexpr uint64_t MinTileBytes = 256;
    const uint64_t tile_bytes = tile * element_bytes;
    if (tile_bytes < MinTileBytes) return std::nullopt;
    const uint64_t sm_count = std::max<uint32_t>(
        1, ReductionInternal::denseReductionDeviceIdentity(input.getPlacement().getDeviceNum()).sm_count);
    const uint64_t resident_warps = 32 * static_cast<uint64_t>(sm_count);  // four 8-warp CTAs per SM
    const uint64_t work = O * tiles;  // [outer, K tile] items, W warps each

    // W is the most warps per item whose launch still fits one wave of resident warps (a second, part-filled wave left
    // a tail: audit cell 432 at W 4, 1.11 waves, 64 us against 58 at W 2), and at least four rows per warp (cell 460,
    // R = 7: W 1 9.9 us, W 4 13.7 at FP8).
    uint32_t warps = 1;
    while (warps < 8 && work * warps * 2 <= resident_warps) warps *= 2;
    while (warps > 1 && static_cast<uint64_t>(warps) * 4 > R) warps /= 2;

    // In place when the launch keeps enough bytes in flight to beat the split it replaces. A sliced split reads about
    // the view's bytes once (near the dense rate), so the in-place read must run near the copy rate: 4 KiB per SM (0.70
    // MB on 170 SMs; 462 at 0.64 MB lost 17% to sliced, 460 at 0.81 MB tied it). A packed split moves the bytes three
    // times: 2 KiB per SM (478 at 0.76 MB was 1.6-2.5x faster than packed; 461 at 0.32-0.36 MB tied it).
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
    return PitchedCompleteLaunch{warps, tile};
}

bool Reducer::readsSpanInPlace(const StridedSpan& input,
                               const std::vector<uint32_t>& axes,
                               DataType output_dtype) {
    try {
        (void)analyzeGeometry(input.getDimensions(), input.getStridesElements(), axes);
        return true;
    } catch (const NotImplementedException&) {
        return pitchedCompleteLaunch(input, axes, output_dtype).has_value();
    }
}

}  // namespace ThorImplementation
