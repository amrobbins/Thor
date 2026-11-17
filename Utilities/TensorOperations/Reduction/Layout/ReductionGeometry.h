#pragma once

// How a reduction reads its input: the classification of a reduction's layout (ReductionGeometry) that every
// stamp is planned from.

#include <cstdint>
#include <optional>
#include <vector>

namespace ThorImplementation {

// The layout class of a reduction (ReductionGeometry::path). A view is classed by the dense layout of the bytes it
// reads (a compact permutation by its physical order, a pitched view by its rows).
enum class ReductionLayout : uint8_t {
    SingleOutput = 0,   // every element reduces into one output
    TrailingBlock = 1,  // the reduced axes are the trailing block: each output reads one contiguous segment
    MiddleBlock = 2,    // one reduced block with retained axes on both sides: [outer, reduction, inner]
    Segmented = 4,      // offset-segmented (ragged) rows: SegmentedReducer
    DisjointRuns = 5,   // the reduced axes are several runs of the dense layout
};

enum class RetainedOutputOrder : uint8_t {
    NaturalOuterInner = 0,
    PermutedInnerOuter = 1,
};

/** Role of one collapsed run in a dense row-major reduction traversal. */
enum class DenseRunKind : uint8_t {
    Reduced = 0,
    Retained = 1,
};

/**
 * One collapsed run of adjacent, traversal-equivalent dimensions in a dense row-major tensor.
 *
 * extent is the product of the non-singleton dimensions represented by the run. physical_stride is the source-element
 * stride between adjacent coordinates of that collapsed run. domain_stride is the row-major multiplier of that run in
 * the flattened reduction domain (for Reduced runs) or flattened output domain (for Retained runs). Singleton
 * dimensions are omitted because they neither advance storage nor contribute a coordinate; runs on either side of an
 * omitted singleton may therefore collapse together when they have the same role.
 */
struct DenseRun {
    DenseRunKind kind = DenseRunKind::Reduced;
    uint64_t extent = 1;
    uint64_t physical_stride = 1;
    uint64_t domain_stride = 1;
};

/**
 * Stamp-time traversal plan for an ordinary dense row-major tensor.
 *
 * Replacement dense-run reducers can consume this compact run sequence instead of rediscovering multidimensional
 * coordinates from every scalar's flattened logical index. Launch policy is intentionally not part of this structure;
 * block size, sharding, and scratch policy are selected separately.
 */
struct DenseRunGeometry {
    std::vector<DenseRun> runs;
    uint32_t reduced_run_count = 0;
    uint32_t retained_run_count = 0;
};

/**
 * Production planning metadata for a zero-copy logical permutation whose visible storage is physically dense.
 *
 * Singleton axes are intentionally omitted from the physical axis vectors and logical retained-order vector because
 * they do not affect address order.
 * The source can therefore be traversed as one dense [outer, reduction, inner] tensor even though its logical Tensor
 * view is non-dense. retained_output_order describes whether the logical dense output order matches the natural
 * [outer..., inner...] retained order or requires the flattened [inner..., outer...] retained-axis permutation.
 *
 * The dense planner reduces such a view as the dense physical [outer, reduction, inner]. A permuted [inner...,
 * outer...] output is refused (ViewSplit). The metadata is the classification.
 */
struct PermutedLayout {
    std::vector<uint32_t> physical_outer_axes;
    std::vector<uint32_t> physical_reduction_axes;
    std::vector<uint32_t> physical_inner_axes;
    std::vector<uint32_t> logical_non_singleton_retained_axes;

    uint64_t outer_size = 0;
    uint64_t reduction_size = 0;
    uint64_t inner_size = 0;

    RetainedOutputOrder retained_output_order =
        RetainedOutputOrder::NaturalOuterInner;
};

/**
 * Pitched (H) metadata for a non-overlapping affine view that is logically [outer..., reduction..., inner...] and whose
 * trailing inner block is physically contiguous.
 *
 * The outer and reduction groups may contain padding between adjacent flattened entries,
 * but each group must itself be row-major flattenable to one constant element stride. The
 * resulting source address is therefore exactly
 *
 *   outer * outer_stride + reduction * reduction_stride + inner
 *
 * with no per-scalar mixed-radix coordinate reconstruction. This is intentionally narrower
 * than a general strided-view reducer: zero-stride/broadcast aliases, overlapping views,
 * retained-axis permutations, and non-contiguous inner payloads are not candidates.
 * Such a view is refused; the caller reduces its dense copy (ViewSplit). The metadata is its classification.
 */
struct PitchedLayout {
    uint64_t outer_size = 1;
    uint64_t reduction_size = 1;
    uint64_t inner_size = 1;
    uint64_t outer_stride = 0;
    uint64_t reduction_stride = 1;
};

/**
 * Payload-transpose (G) metadata for the compact-permutation retained-order transform
 *
 *   physical source: [A..., reduction..., B..., payload...]
 *   logical output:  [B..., A..., payload...]
 *
 * after singleton axes are removed. The source is physically dense, so all sizes below are flattened once on the host.
 * Such a view is refused; the caller reduces the bytes as the dense [A, reduction, B x payload] and its dense copy
 * writes logical [B,A,payload] (ViewSplit). The metadata is the classification.
 */
struct PayloadTransposeLayout {
    std::vector<uint32_t> physical_a_axes;
    std::vector<uint32_t> physical_reduction_axes;
    std::vector<uint32_t> physical_b_axes;
    std::vector<uint32_t> physical_payload_axes;

    uint64_t a_size = 1;
    uint64_t reduction_size = 1;
    uint64_t b_size = 1;
    uint64_t payload_size = 1;
};

struct ReductionGeometry {
    std::vector<uint32_t> axes;
    uint32_t rank = 0;

    // For an ordinary dense tensor whose reduced axes form one contiguous logical block, the input can be viewed as
    // [outer_size, reduction_size, inner_size]. For a natural-output permutation-aware view (case D) these fields are
    // instead the equivalent dense physical [outer, reduction, inner] geometry recovered from the view's strides.
    // They remain zero for layouts with neither reading.
    bool reduced_axes_are_contiguous = false;
    uint64_t outer_size = 0;
    uint64_t inner_size = 0;

    uint64_t input_elements = 0;
    uint64_t reduction_size = 0;
    uint64_t output_elements = 0;
    std::vector<uint64_t> output_dimensions;
    std::vector<uint64_t> squeezed_output_dimensions;
    // Populated only for ordinary dense row-major storage. This is traversal geometry, not launch policy. Production
    // composed reducers consume this metadata during value-path planning and stamping. SUM can eliminate arbitrary
    // reduced-run sequences through direct reducer stages; other value operations and arg reductions use it as their
    // migration metadata until their composed semantics are implemented.
    std::optional<DenseRunGeometry> dense_run_geometry = std::nullopt;

    // A rank-1 non-dense view (an extracted matrix diagonal, a column, a stride-0 broadcast) is an affine sequence
    // reduced without materializing it or paying the general logical-axis index mapping. It runs StridedR
    // (StridedReduction.h).
    bool device_transform_uses_affine_stride = false;
    uint64_t affine_input_stride = 1;

    // Physical-layout analysis is relative to input.getMemPtr(), which already includes a Tensor view's storage
    // element offset.  Storage offsets are therefore intentionally not baked into this reusable geometry.
    bool physical_layout_is_dense_permutation = false;
    std::vector<uint32_t> physical_non_singleton_axis_order;

    // (B) A compact physical permutation whose non-singleton reduced axes form the complete physical trailing block
    // reduces as a dense [outputs, R] when the physical retained-axis order already matches Thor's logical dense
    // output order. Each output is then one ordinary contiguous physical segment; no logical-to-physical input mapper
    // or output permutation is required. More general retained-order permutations are the payload transpose (G).
    bool permutation_aware_contiguous_segments = false;

    std::optional<PermutedLayout> permuted_layout = std::nullopt;

    // (G) Compact dense physical [A,reduction,B,payload] storage whose logical retained order is
    // [B,A,payload]. Refused (ViewSplit).
    std::optional<PayloadTransposeLayout> payload_transpose_layout = std::nullopt;

    // (H) Classified under MiddleBlock. Output order is always the natural logical
    // [outer,inner] order. Refused (ViewSplit).
    std::optional<PitchedLayout> pitched_layout = std::nullopt;

    // MiddleBlock logically produces a dense [outer_size, inner_size] matrix after reducing the physically contiguous
    // middle domain. Natural output uses output_index = outer * inner_size + inner. The output strides describe it
    // (AlignedK writes at them); tiled_output_permuted marks the dense [inner_size, outer_size] rotation, which the
    // reducer refuses (ViewSplit).
    uint64_t tiled_output_outer_stride = 0;
    uint64_t tiled_output_inner_stride = 0;
    bool tiled_output_permuted = false;
    // Set with tiled_output_permuted: a classification (no kernel transposes the output through shared memory).
    bool tiled_output_shared_transpose = false;

    ReductionLayout path = ReductionLayout::SingleOutput;
};

}  // namespace ThorImplementation
