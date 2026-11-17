// Geometry classification of a reduction: dense runs, physical permutations, pitched and payload layouts.

#include "Utilities/TensorOperations/Reduction/Layout/ReductionGeometry.h"

#include "Utilities/Exceptions.h"
#include "Utilities/TensorOperations/Cub/CubDevicePrimitiveSupport.h"
#include "Utilities/TensorOperations/Reduction/ArgReducer.h"
#include "Utilities/TensorOperations/Reduction/FrontEnd.h"
#include "Utilities/TensorOperations/Reduction/Planner/DenseProductionPhysicalSelectionContext.h"
#include "Utilities/TensorOperations/Reduction/Reducer.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace ThorImplementation {

namespace ReductionFrontEnd {

using namespace CubDevicePrimitiveSupport;
using namespace ReductionInternal;

[[nodiscard]] bool isDirectDenseReductionPath(ReductionLayout path) {
    return path == ReductionLayout::SingleOutput || path == ReductionLayout::TrailingBlock
           || path == ReductionLayout::MiddleBlock;
}

}  // namespace ReductionFrontEnd

namespace {

using namespace CubDevicePrimitiveSupport;
using namespace ReductionInternal;
using namespace ReductionFrontEnd;

[[nodiscard]] bool isOrdainedDenseReductionPath(ReductionLayout path) {
    return isDirectDenseReductionPath(path) || path == ReductionLayout::DisjointRuns;
}

void requireOrdainedDenseReductionPath(const ReductionGeometry& geometry) {
    if (isOrdainedDenseReductionPath(geometry.path)) {
        return;
    }

    throw std::logic_error(
        "Invariant violated: an ordinary dense tensor reduction was not assigned to an ordained "
        "dense execution family.");
}

[[nodiscard]] DenseRunGeometry analyzeDenseRuns(const std::vector<uint64_t>& input_dimensions,
                                                               const std::vector<uint64_t>& dense_strides,
                                                               const std::vector<uint32_t>& axes) {
    DenseRunGeometry dense_runs;
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

        const DenseRunKind kind =
            reduced ? DenseRunKind::Reduced : DenseRunKind::Retained;
        if (!dense_runs.runs.empty() && dense_runs.runs.back().kind == kind) {
            DenseRun& run = dense_runs.runs.back();
            if (run.extent > std::numeric_limits<uint64_t>::max() / input_dimensions[dimension]) {
                throw std::invalid_argument("Tensor reduction dense run extent overflows uint64_t.");
            }
            run.extent *= input_dimensions[dimension];
            run.physical_stride = dense_strides[dimension];
            continue;
        }

        dense_runs.runs.push_back(DenseRun{
            .kind = kind,
            .extent = input_dimensions[dimension],
            .physical_stride = dense_strides[dimension],
            .domain_stride = 1,
        });
    }

    uint64_t reduced_domain_stride = 1;
    uint64_t retained_domain_stride = 1;
    for (size_t run_index = dense_runs.runs.size(); run_index-- > 0;) {
        DenseRun& run = dense_runs.runs[run_index];
        if (run.kind == DenseRunKind::Reduced) {
            run.domain_stride = reduced_domain_stride;
            if (reduced_domain_stride > std::numeric_limits<uint64_t>::max() / run.extent) {
                throw std::invalid_argument("Tensor reduction dense reduced-run domain overflows uint64_t.");
            }
            reduced_domain_stride *= run.extent;
            ++dense_runs.reduced_run_count;
        } else {
            run.domain_stride = retained_domain_stride;
            if (retained_domain_stride > std::numeric_limits<uint64_t>::max() / run.extent) {
                throw std::invalid_argument("Tensor reduction dense retained-run domain overflows uint64_t.");
            }
            retained_domain_stride *= run.extent;
            ++dense_runs.retained_run_count;
        }
    }

    return dense_runs;
}

[[nodiscard]] bool analyzeDensePhysicalPermutation(const std::vector<uint64_t>& input_dimensions,
                                                   const std::vector<uint64_t>& input_strides,
                                                   ReductionGeometry& geometry) {
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
                                                               ReductionGeometry& geometry) {
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

    // The trailing block (B) is deliberately the physically trailing case only. Once the first non-singleton reduced
    // axis is reached in physical order, every remaining non-singleton physical axis must also be reduced. That makes
    // each retained output one ordinary contiguous segment of exactly geometry.reduction_size values.
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

    // The dense [outer, R] reading emits segments in physical prefix order. The trailing block (B) intentionally
    // activates only when that order is already Thor's dense logical output order. If these differ, input traversal is
    // still usable but an output permutation is required; that is the separate payload-transpose (G) problem.
    if (physical_retained_axes != logical_non_singleton_retained_axes) {
        return false;
    }

    uint64_t physical_reduction_size = 1;
    for (size_t physical = first_reduced; physical < physical_order.size(); ++physical) {
        const uint32_t axis = physical_order[physical];
        if (physical_reduction_size > std::numeric_limits<uint64_t>::max() / input_dimensions[axis]) {
            throw std::invalid_argument("Permutation-aware contiguous reduction size overflows uint64_t.");
        }
        physical_reduction_size *= input_dimensions[axis];
    }

    uint64_t physical_output_elements = 1;
    for (uint32_t axis : physical_retained_axes) {
        if (physical_output_elements > std::numeric_limits<uint64_t>::max() / input_dimensions[axis]) {
            throw std::invalid_argument("Permutation-aware contiguous output size overflows uint64_t.");
        }
        physical_output_elements *= input_dimensions[axis];
    }

    if (physical_reduction_size != geometry.reduction_size || physical_output_elements != geometry.output_elements) {
        throw std::logic_error("Permutation-aware contiguous geometry is internally inconsistent.");
    }

    geometry.permutation_aware_contiguous_segments = true;
    return true;
}

void analyzePermutedLayoutCandidate(const std::vector<uint64_t>& input_dimensions,
                                    const std::vector<uint32_t>& axes,
                                    ReductionGeometry& geometry) {
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

    PermutedLayout candidate;
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
                throw std::invalid_argument("Permuted layout geometry size overflows uint64_t.");
            }
            product *= input_dimensions[axis];
        }
        return product;
    };

    candidate.outer_size = productOfAxes(candidate.physical_outer_axes);
    candidate.reduction_size = geometry.reduction_size;
    candidate.inner_size = productOfAxes(candidate.physical_inner_axes);

    // The candidate is a middle block (inner_size > 1). A physically trailing reduction has inner_size == 1 and is the
    // trailing block (B), not this path.
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
        candidate.retained_output_order = RetainedOutputOrder::NaturalOuterInner;
    } else if (candidate.logical_non_singleton_retained_axes == permuted_retained_axes) {
        candidate.retained_output_order = RetainedOutputOrder::PermutedInnerOuter;
    } else {
        return;
    }

    geometry.permuted_layout = std::move(candidate);
}

void analyzePayloadTransposeLayoutCandidate(const std::vector<uint64_t>& input_dimensions,
                                            const std::vector<uint32_t>& axes,
                                            ReductionGeometry& geometry) {
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

    // The payload transpose (G) is deliberately the middle-reduction case. A and B must both be genuine retained
    // physical groups; physically trailing output reorder remains outside this strategy.
    if (first_reduced == 0 || last_reduced + 1 >= physical_order.size()) {
        return;
    }

    PayloadTransposeLayout candidate;
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

    // Split the physical suffix as [B..., payload...] and accept only the exact logical retained ordering [B..., A...,
    // payload...]. This is the payload-preserving A/B rotation, not a general permutation recognizer. B must contain at
    // least one non-singleton axis; payload may be scalar (empty).
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
                throw std::invalid_argument(std::string("Payload-transpose layout ") + label
                                            + " size overflows uint64_t.");
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
        throw std::logic_error("Payload-transpose layout retained-output geometry is internally inconsistent.");
    }

    geometry.payload_transpose_layout = std::move(candidate);
}

void activatePayloadTransposeLayoutCandidate(ReductionGeometry& geometry) {
    if (!geometry.payload_transpose_layout.has_value()) {
        throw std::logic_error("Payload-transpose layout activation requires detected geometry.");
    }
    const PayloadTransposeLayout& candidate =
        geometry.payload_transpose_layout.value();
    if (candidate.reduction_size != geometry.reduction_size || candidate.a_size <= 1 || candidate.b_size <= 1) {
        throw std::logic_error("Payload-transpose layout geometry is internally inconsistent.");
    }
    geometry.path = ReductionLayout::MiddleBlock;
}

void activatePermutedLayoutCandidate(ReductionGeometry& geometry) {
    if (!geometry.permuted_layout.has_value()) {
        throw std::logic_error("Permuted layout activation requires detected physical geometry.");
    }

    const PermutedLayout& candidate = geometry.permuted_layout.value();
    if (candidate.inner_size <= 1 || candidate.reduction_size != geometry.reduction_size) {
        throw std::logic_error("Permuted layout geometry is internally inconsistent.");
    }
    if (candidate.outer_size != 0
        && candidate.inner_size > std::numeric_limits<uint64_t>::max() / candidate.outer_size) {
        throw std::logic_error("Permuted layout output size overflows uint64_t.");
    }
    if (candidate.outer_size * candidate.inner_size != geometry.output_elements) {
        throw std::logic_error("Permuted layout retained-output size is internally inconsistent.");
    }

    // The tuned MiddleBlock kernels only need the visible pointer and the dense physical
    // [outer,reduction,inner] extents. They do not decode logical coordinates, so a zero-copy permutation view can
    // use exactly the same source traversal as its underlying dense storage.
    geometry.outer_size = candidate.outer_size;
    geometry.reduction_size = candidate.reduction_size;
    geometry.inner_size = candidate.inner_size;
    geometry.tiled_output_shared_transpose = false;

    switch (candidate.retained_output_order) {
        case RetainedOutputOrder::NaturalOuterInner:
            geometry.tiled_output_outer_stride = candidate.inner_size;
            geometry.tiled_output_inner_stride = 1;
            geometry.tiled_output_permuted = false;
            break;
        case RetainedOutputOrder::PermutedInnerOuter:
            geometry.tiled_output_outer_stride = 1;
            geometry.tiled_output_inner_stride = candidate.outer_size;
            geometry.tiled_output_permuted = true;
            geometry.tiled_output_shared_transpose = true;
            break;
    }

    geometry.path = ReductionLayout::MiddleBlock;
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

void analyzePitchedLayoutCandidate(const std::vector<uint64_t>& input_dimensions,
                                   const std::vector<uint64_t>& input_strides,
                                   const std::vector<uint32_t>& axes,
                                   ReductionGeometry& geometry) {
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
        throw std::logic_error("Pitched layout flattened extents are internally inconsistent.");
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

    PitchedLayout candidate;
    candidate.outer_size = outer->extent;
    candidate.reduction_size = reduction->extent;
    candidate.inner_size = inner->extent;
    candidate.outer_stride = outer_stride;
    candidate.reduction_stride = reduction_stride;
    geometry.pitched_layout = candidate;
}

void activatePitchedLayoutCandidate(ReductionGeometry& geometry) {
    if (!geometry.pitched_layout.has_value()) {
        throw std::logic_error("Pitched layout activation requires detected affine geometry.");
    }

    const PitchedLayout& candidate = geometry.pitched_layout.value();
    if (candidate.outer_size != geometry.outer_size || candidate.reduction_size != geometry.reduction_size
        || candidate.inner_size != geometry.inner_size) {
        throw std::logic_error("Pitched layout geometry is internally inconsistent.");
    }
    if (candidate.outer_size != 0
        && candidate.inner_size > std::numeric_limits<uint64_t>::max() / candidate.outer_size) {
        throw std::logic_error("Pitched layout output size overflows uint64_t.");
    }
    if (candidate.outer_size * candidate.inner_size != geometry.output_elements) {
        throw std::logic_error("Pitched layout retained-output size is internally inconsistent.");
    }

    geometry.tiled_output_outer_stride = candidate.inner_size;
    geometry.tiled_output_inner_stride = 1;
    geometry.tiled_output_permuted = false;
    geometry.tiled_output_shared_transpose = false;
    geometry.path = ReductionLayout::MiddleBlock;
}

}  // namespace

ReductionGeometry Reducer::analyzeGeometry(const std::vector<uint64_t>& input_dimensions, uint32_t axis) {
    return analyzeGeometry(input_dimensions, std::vector<uint32_t>{axis});
}

ReductionGeometry Reducer::analyzeGeometry(const std::vector<uint64_t>& input_dimensions,
                                                    const std::vector<uint32_t>& axes) {
    std::vector<uint64_t> dense_strides(input_dimensions.size(), 1);
    uint64_t running_stride = 1;
    for (size_t dimension = input_dimensions.size(); dimension-- > 0;) {
        dense_strides[dimension] = running_stride;
        if (input_dimensions[dimension] != 0
            && running_stride > std::numeric_limits<uint64_t>::max() / input_dimensions[dimension]) {
            throw std::invalid_argument("Tensor reduction input element count overflows uint64_t.");
        }
        running_stride *= input_dimensions[dimension];
    }
    return analyzeGeometry(input_dimensions, dense_strides, axes);
}

ReductionGeometry Reducer::analyzeGeometry(const std::vector<uint64_t>& input_dimensions,
                                                    const std::vector<uint64_t>& input_strides,
                                                    const std::vector<uint32_t>& axes) {
    ReductionGeometry geometry = classifyGeometry(input_dimensions, input_strides, axes);
    // A rotated (F / G) or pitched (H) view is the caller's to run in two parts (splitRefusedView).
    if (refusedViewSplit(input_dimensions, input_strides == StridedSpan::denseStrides(input_dimensions), geometry)
            .has_value()) {
        throw NotImplementedException(
            "Tensor reduction refuses a view whose retained axes are permuted or whose rows have gaps: run it in "
            "two parts with the dense copy (Reducer::splitRefusedView).");
    }
    return geometry;
}

ReductionGeometry Reducer::analyzeValueSpanGeometry(ReductionOp op,
                                                             const StridedSpan& input,
                                                             const std::vector<uint32_t>& axes,
                                                             DataType output_dtype) {
    requireSupportedOperation(op);
    if (pitchedCompleteLaunch(input, axes, output_dtype).has_value()) {
        return classifyGeometry(input.getDimensions(), input.getStridesElements(), axes);
    }
    return analyzeGeometry(input.getDimensions(), input.getStridesElements(), axes);
}

ReductionGeometry Reducer::classifyGeometry(const std::vector<uint64_t>& input_dimensions,
                                            const std::vector<uint64_t>& input_strides,
                                            const std::vector<uint32_t>& axes) {
    if (input_dimensions.empty()) {
        throw std::invalid_argument("Tensor reduction requires at least one input dimension.");
    }
    if (input_dimensions.size() != input_strides.size()) {
        throw std::invalid_argument("Tensor reduction input stride rank must match the input rank.");
    }
    if (input_dimensions.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::invalid_argument("Tensor reduction rank exceeds the uint32 axis representation limit.");
    }
    if (axes.empty()) {
        throw std::invalid_argument("Tensor reduction requires at least one reduction axis.");
    }
    if (axes.size() > input_dimensions.size()) {
        throw std::invalid_argument("Tensor reduction has more reduction axes than input dimensions.");
    }
    for (size_t i = 0; i < axes.size(); ++i) {
        if (axes[i] >= input_dimensions.size()) {
            throw std::invalid_argument("Tensor reduction axis is outside the input rank.");
        }
        if (i > 0 && axes[i] <= axes[i - 1]) {
            throw std::invalid_argument("Tensor reduction axes must be unique and strictly increasing.");
        }
    }

    auto checkedMultiply = [](uint64_t a, uint64_t b, const char* quantity) {
        if (b != 0 && a > std::numeric_limits<uint64_t>::max() / b) {
            throw std::invalid_argument(std::string("Tensor reduction ") + quantity + " overflows uint64_t.");
        }
        return a * b;
    };

    for (uint64_t dimension : input_dimensions) {
        if (dimension == 0) {
            throw std::invalid_argument("Tensor reduction does not support zero-sized tensor dimensions.");
        }
    }

    ReductionGeometry geometry;
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
        throw std::invalid_argument("Tensor reduction input element count exceeds int64 limits.");
    }
    if (geometry.output_elements > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        throw std::invalid_argument("Tensor reduction output element count exceeds int64 limits.");
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
        analyzePermutedLayoutCandidate(input_dimensions, axes, geometry);
        // The payload transpose (G) fills only the compact [A,reduction,B,payload] -> [B,A,payload] retained-order gap
        // left by the permutation-aware candidates, so it is considered only after they reject the geometry.
        if (!geometry.permutation_aware_contiguous_segments
            && !geometry.permuted_layout.has_value()) {
            analyzePayloadTransposeLayoutCandidate(input_dimensions, axes, geometry);
        }
        // Preserve ownership of every already-ordained compact-permutation path. The pitched layout (H) is considered
        // only when no direct compact-permutation strategy applies. Keeping these candidates mutually exclusive also
        // makes launch dispatch unambiguous.
        if (!geometry.permutation_aware_contiguous_segments
            && !geometry.permuted_layout.has_value()
            && !geometry.payload_transpose_layout.has_value()) {
            analyzePitchedLayoutCandidate(input_dimensions, input_strides, axes, geometry);
        }
    }

    const bool reduces_to_single_output = geometry.output_elements == 1;
    const bool has_physically_contiguous_fixed_segments =
        geometry.reduced_axes_are_contiguous && geometry.inner_size == 1;

    if (input_is_dense_contiguous) {
        if (reduces_to_single_output) {
            geometry.path = ReductionLayout::SingleOutput;
        } else if (has_physically_contiguous_fixed_segments) {
            geometry.path = ReductionLayout::TrailingBlock;
        } else if (geometry.reduced_axes_are_contiguous) {
            geometry.path = ReductionLayout::MiddleBlock;
        } else {
            // DisjointRuns is the structural classification for ordinary dense disjoint reduced runs. VALUE and ARG
            // both execute it through planDenseReduction(), which selects R / KR / RK passes and replans after every
            // pass.
            geometry.path = ReductionLayout::DisjointRuns;
        }
        requireOrdainedDenseReductionPath(geometry);
    } else if (reduces_to_single_output && geometry.physical_layout_is_dense_permutation) {
        // (A) When every logical axis is reduced, a compact physical permutation is just one contiguous
        // physical value domain. Value reductions are insensitive to the logical visitation order, and a proven dense
        // physical permutation contains each logical value exactly once in input_elements contiguous elements from
        // input.getMemPtr(). Reducer::stampValidated runs it as that dense [input_elements] on the
        // dense planner (viewPhysicalDenseProblem); the path is its classification.
        geometry.path = ReductionLayout::SingleOutput;
    } else if (geometry.permutation_aware_contiguous_segments) {
        // (B) The non-singleton reduced axes are the complete physical suffix and the physical retained
        // prefix is already in Thor's logical dense output order. Each retained output is therefore one ordinary
        // contiguous segment in storage. Reducer::stampValidated runs it as the dense [output_elements,
        // reduction_size] over axis 1 on the dense planner (viewPhysicalDenseProblem); the path is its
        // classification. Cases that need retained-output permutation are the payload transpose (G).
        geometry.path = ReductionLayout::TrailingBlock;
    } else if (geometry.permuted_layout.has_value()) {
        // A logical permutation view over physically dense storage: the candidate supplies the physical
        // [outer,reduction,inner] traversal and whether the final dense retained output is natural [outer,inner] or the
        // [inner,outer] rotation. A natural output runs the dense planner on the bytes (viewPhysicalDenseProblem);
        // analyzeGeometry refuses the rotation, which the caller runs in two parts (refusedViewSplit). The path is the
        // classification.
        activatePermutedLayoutCandidate(geometry);
    } else if (geometry.payload_transpose_layout.has_value()) {
        // (G) The compact source is physically [A,reduction,B,payload] while Thor's dense retained output
        // is [B,A,payload]. analyzeGeometry refuses it; the caller reduces the bytes as [A,reduction,BxP] and its
        // dense copy writes [B,A,payload] (refusedViewSplit; a 32x33 payload-transpose kernel ran 5-2,447x
        // slower).
        activatePayloadTransposeLayoutCandidate(geometry);
    } else if (reduces_to_single_output && input_dimensions.size() == 1) {
        // Rank-1 views are always affine, including diagonals (stride > 1) and broadcast aliases (stride == 0), and
        // need no general logical-coordinate metadata or indexing. Reducer::stampValidated runs them on StridedR
        // (StridedReduction.h: the stride-0 closed form, else Split); the path is the classification.
        geometry.path = ReductionLayout::SingleOutput;
        geometry.device_transform_uses_affine_stride = true;
        geometry.affine_input_stride = input_strides[0];
    } else if (geometry.pitched_layout.has_value()) {
        // (H) A non-dense logical [outer,reduction,inner] view with a contiguous inner payload can be addressed with
        // two affine pitches. analyzeGeometry refuses it; the caller runs it in two parts (refusedViewSplit: its dense
        // copy, or its rows at their pitch). A tiled pitched kernel ran up to 22,000x slower.
        activatePitchedLayoutCandidate(geometry);
    } else if (densePermutationRuns(input_dimensions, geometry).has_value()) {
        // A split physical reduction, a compact permutation whose reduced axes are several physical runs and whose
        // retained axes keep their physical order. It is a dense tensor reduced over disjoint runs, DisjointRuns's
        // classification; Reducer::stampValidated runs it on the dense planner (viewPhysicalDenseProblem).
        geometry.path = ReductionLayout::DisjointRuns;
    } else {
        // Thor has no per-scalar logical-index reduction backend. Every real workload must be owned by an explicit
        // physical-layout-aware reducer; exotic views fail rather than silently executing a catastrophically slow
        // catch-all implementation.
        throw NotImplementedException(
            "Unsupported tensor-reduction geometry: no ordained production reducer owns this layout. "
            "Materialize/reorder the view or add an explicit high-efficiency reduction strategy for this geometry.");
    }

    // No path inherits CUB's `int` fixed-segment size; no reduction calls DeviceSegmentedReduce, and the dense and
    // StridedR kernels carry 64-bit geometry.
    return geometry;
}

ReductionGeometry Reducer::analyzeValueGeometry(ReductionOp op,
                                                          const std::vector<uint64_t>& input_dimensions,
                                                          const std::vector<uint32_t>& axes) {
    requireSupportedOperation(op);
    return analyzeGeometry(input_dimensions, axes);
}

ReductionGeometry Reducer::analyzeValueGeometry(ReductionOp op,
                                                          const std::vector<uint64_t>& input_dimensions,
                                                          const std::vector<uint64_t>& input_strides,
                                                          const std::vector<uint32_t>& axes) {
    requireSupportedOperation(op);
    return analyzeGeometry(input_dimensions, input_strides, axes);
}

ReductionGeometry ArgReducer::analyzeDenseGeometry(
    const std::vector<uint64_t>& input_dimensions,
    const std::vector<uint32_t>& axes) {
    ReductionGeometry geometry = Reducer::analyzeGeometry(input_dimensions, axes);
    // This public helper accepts only canonical dense geometry, so returning a view family would be an internal
    // architecture violation rather than a recoverable execution choice.
    requireOrdainedDenseReductionPath(geometry);
    return geometry;
}

}  // namespace ThorImplementation
