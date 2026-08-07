#pragma once

#include "Utilities/TensorOperations/Einsum/EinsumParser.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace ThorImplementation {

enum class DataType;

enum class EinsumPlanKind {
    UNARY,
    ELEMENTWISE,
    REDUCTION,
    GEMM,
    BATCHED_GEMM,
    GENERAL,
    PAIR_PRODUCT,
};

// A repeated label within one operand denotes a diagonal.  The source axes in
// this structure are collapsed to one logical axis before any permutation or
// broadcast described by the rest of the plan is applied.
struct EinsumDiagonalPlan {
    int32_t label = -1;
    std::vector<uint32_t> source_axes;
};

// Backend-independent lowering of one operand to the plan's canonical
// iteration order.  This is sufficient to express every resolved einsum as:
//
//   1. extract repeated-label diagonals,
//   2. permute each operand into canonical order,
//   3. insert/broadcast missing or singleton axes,
//   4. multiply all operands elementwise,
//   5. reduce the trailing reduction axes.
//
// permutation contains diagonalized source-axis indices in canonical label
// order, omitting inserted axes.  aligned_dimensions has one entry per
// iteration label and describes the operand shape before broadcasting.
struct EinsumOperandPlan {
    std::vector<int32_t> diagonalized_labels;
    std::vector<uint64_t> diagonalized_dimensions;
    std::vector<EinsumDiagonalPlan> diagonals;

    std::vector<uint32_t> permutation;
    std::vector<uint32_t> inserted_axes;
    std::vector<uint32_t> broadcast_axes;
    std::vector<uint64_t> aligned_dimensions;

    bool requiresDiagonalExtraction() const { return !diagonals.empty(); }
    bool requiresPermutation() const;
    bool requiresBroadcast() const { return !broadcast_axes.empty(); }
};

// Matrix-multiply-specific transform for an input operand.  canonical_labels
// contains only labels that participate in that operand's matrix-multiply
// view.  Batch labels may be absent physically; inserted_axes identifies those
// singleton batch axes.  permutation maps each physically present canonical
// axis to a diagonalized source axis.
//
// When requires_materialized_permutation is false, the operand's logical label
// groups are in an order that can be represented directly or through a GEMM
// transpose flag. Execution still validates the actual physical strides (for
// example after repeated-label diagonal extraction); a non-BLAS-addressable
// logical view may require operand-local materialization even when this flag is
// false. When it is true, optimized execution must materialize only this
// operand into canonical dense matrix-group order before GEMM; it must not
// imply construction of the full generic broadcast product.
struct EinsumMatrixOperandPlan {
    std::vector<int32_t> canonical_labels;
    std::vector<uint32_t> permutation;
    std::vector<uint32_t> inserted_axes;
    std::vector<uint32_t> broadcast_axes;

    bool transpose = false;
    bool requires_materialized_permutation = false;
};

struct EinsumMatrixMultiplyPlan {
    std::vector<int32_t> batch_labels;
    std::vector<int32_t> lhs_free_labels;
    // Operand-local reductions are performed independently before the matrix
    // contraction. Besides labels present on only one operand, this includes
    // a shared contraction label whose opposite operand has singleton extent:
    // sum_k A[..., k] * B[..., 0] factors into sum_k A[..., k] times B[..., 0].
    // The singleton counterpart is removed as a zero-copy logical-axis elision.
    // Neither category participates in the resulting matrix operand layout,
    // and all actual reductions remain centralized Expression/CubReduction
    // stages.
    std::vector<int32_t> lhs_reduction_labels;
    std::vector<int32_t> lhs_broadcast_elision_labels;
    std::vector<int32_t> contraction_labels;
    std::vector<int32_t> rhs_broadcast_elision_labels;
    std::vector<int32_t> rhs_reduction_labels;
    std::vector<int32_t> rhs_free_labels;
    std::vector<int32_t> canonical_output_labels;

    EinsumMatrixOperandPlan lhs;
    EinsumMatrixOperandPlan rhs;

    // For each requested output axis, gives the corresponding axis in
    // canonical_output_labels. Identity means the canonical GEMM orientation
    // already has the requested einsum output order. Execution may also
    // satisfy the complete [batch..., M..., N...] -> [batch..., N..., M...]
    // swap algebraically with rhs^T @ lhs^T, without materializing a result
    // transpose. Other requested output orders remain eligible for optimized
    // contraction: execution forms the compact canonical GEMM result first,
    // then materializes only an output-sized permutation.
    std::vector<uint32_t> output_permutation;

    uint64_t batch_count = 1;
    uint64_t m = 1;
    uint64_t n = 1;
    uint64_t k = 1;

    bool requires_output_permutation = false;

    // True only for the trivial matrix lowering where pre-reductions,
    // diagonal extraction, materialized input permutations, broadcasting, and
    // output permutation are all unnecessary. Reshaping contiguous label
    // groups and using GEMM
    // transpose flags do not make a plan non-direct. Execution does not use
    // this flag as its admission gate: non-direct plans may still lower to
    // Expression::matmul when the physical layouts are BLAS-addressable.
    bool direct = false;
};

// Two-operand product with no label that must be reduced across both operands.
// Any reduction labels are local to exactly one operand and may therefore be
// reduced independently before the surviving operands are aligned to the
// requested output and multiplied. This avoids forming the generic Cartesian
// product over operand-local reduction dimensions while deliberately avoiding
// a fake K=1 GEMM for ordinary outer/elementwise products.
//
// Planner-side description of a logical einsum operand at a pair-contraction
// boundary. labels are unique (repeated-label diagonals have already been
// collapsed logically), dimensions are the operand-local extents for those
// labels, and strides_elements describe the physical view that would feed the
// pair lowerer.  An original diagonal therefore remains a non-dense strided
// logical operand rather than pretending to be contiguous.
//
// storage_dtype is optional because the parser/planner API historically plans
// from shapes alone.  Multi-operand execution can populate it from the actual
// tensors; a materialized pair result preserves that storage dtype while
// reduction accumulation remains FP32.
//
// source_operand_indices records which original equation operands contributed
// to this logical value.  It is not needed by two-operand execution, but gives
// the exact/heuristic multi-operand planners stable provenance for subset
// states without coupling the planner to Expression objects.
struct EinsumLogicalOperandPlan {
    std::vector<int32_t> labels;
    std::vector<uint64_t> dimensions;
    std::vector<uint64_t> strides_elements;
    std::optional<DataType> storage_dtype;
    std::vector<uint32_t> source_operand_indices;
    bool dense_storage = false;
    bool diagonal_view = false;
};

struct EinsumPairProductPlan {
    std::vector<int32_t> lhs_reduction_labels;
    std::vector<int32_t> lhs_broadcast_elision_labels;
    std::vector<int32_t> rhs_broadcast_elision_labels;
    std::vector<int32_t> rhs_reduction_labels;
};

// First-order, backend-aware cost description for one physical realization of
// a binary contraction.  The planner intentionally records raw work/traffic
// components instead of hiding them behind one hardware-specific scalar:
// exact/heuristic search can evolve its weighting policy without replanning the
// algebra.
//
// *_elements is always populated.  *_bytes is populated when the logical
// operands carry a concrete tensor storage dtype.
struct EinsumPairPhysicalCost {
    uint64_t matmul_fma_count = 0;
    uint64_t fused_elementwise_count = 0;
    uint64_t reduction_input_elements = 0;

    // Primitive counts are deliberately hardware-agnostic. They let exact and
    // heuristic search prefer fewer launches when weighted work is otherwise
    // equal without encoding one GPU/cuBLASLt implementation's latency quirks.
    uint64_t fused_kernel_count = 0;
    uint64_t reduction_op_count = 0;
    uint64_t materialization_op_count = 0;

    uint64_t lhs_materialization_elements = 0;
    uint64_t rhs_materialization_elements = 0;
    uint64_t output_materialization_elements = 0;
    uint64_t result_elements = 0;
    uint64_t peak_temporary_elements = 0;

    std::optional<uint64_t> lhs_materialization_bytes;
    std::optional<uint64_t> rhs_materialization_bytes;
    std::optional<uint64_t> output_materialization_bytes;
    std::optional<uint64_t> result_bytes;
    std::optional<uint64_t> peak_temporary_bytes;

    // One logical matmul may decompose into several independent strided-batch
    // groups when irregular batch broadcasting/layout prevents one regular
    // strided batched launch.  This is useful execution-cost signal even though
    // Expression remains responsible for scheduling the groups.
    uint64_t matmul_group_count = 0;
};

// One viable physical realization of a pair contraction.  result.labels stays
// in the caller-requested logical surviving-label order, while
// result.strides_elements describes the actual physical backing layout.  Thus a
// natural [batch,M,N] GEMM or algebraically swapped [batch,N,M] GEMM may remain
// a zero-copy logical view instead of paying an eager output permutation.
//
// A candidate with output_materialized=true explicitly pays for a dense copy in
// the logical surviving-label order.  This small candidate set lets future
// exact search reason about downstream layout compatibility without exploring
// arbitrary factorial output permutations.
struct EinsumPairPhysicalCandidate {
    EinsumPlanKind kind = EinsumPlanKind::GENERAL;
    std::optional<EinsumMatrixMultiplyPlan> matrix_multiply;
    std::optional<EinsumPairProductPlan> pair_product;

    EinsumLogicalOperandPlan result;
    // Dense backing label order before result is viewed in logical
    // surviving-label order. Equal to result.labels when the candidate writes
    // the persistent result densely in logical order.
    std::vector<int32_t> physical_result_labels;
    EinsumPairPhysicalCost cost;

    bool swapped_gemm_orientation = false;
    bool lhs_materialized = false;
    bool rhs_materialized = false;
    bool output_materialized = false;
};

// Reusable planner result for contracting two logical operands down to the
// caller-specified surviving labels.  This is intentionally independent of an
// Expression DAG: planning remains speculative, and only a selected tree is
// lowered later.
//
// result remains the dense execution-compatible result used by today's
// two-operand path. physical_candidates exposes the richer set that future
// multi-operand search will retain and compare. preferred_physical_candidate
// identifies the candidate matching today's execution policy.
struct EinsumPairContractionPlan {
    EinsumPlanKind kind = EinsumPlanKind::GENERAL;
    std::vector<int32_t> surviving_labels;
    std::vector<uint64_t> pair_label_dimensions;
    std::vector<int32_t> reduction_labels;

    std::optional<EinsumMatrixMultiplyPlan> matrix_multiply;
    std::optional<EinsumPairProductPlan> pair_product;

    EinsumLogicalOperandPlan result;
    std::optional<DataType> reduction_accumulation_dtype;

    std::vector<EinsumPairPhysicalCandidate> physical_candidates;
    uint32_t preferred_physical_candidate = 0;
};



// Aggregate cost of one complete exact contraction tree.  "Exact" means the
// tree search exhaustively considers every binary partition and every retained
// physical pair realization; the scalar estimate is the deterministic
// first-order policy used to choose among otherwise valid trees.
//
// The raw components remain visible so tests/diagnostics can explain why one
// tree won and so later heuristic planning can share the same policy.
struct EinsumExactContractionCost {
    uint64_t matmul_fma_count = 0;
    uint64_t fused_elementwise_count = 0;
    uint64_t reduction_input_elements = 0;
    uint64_t materialization_elements = 0;
    uint64_t result_write_elements = 0;

    uint64_t matmul_group_count = 0;
    uint64_t fused_kernel_count = 0;
    uint64_t reduction_op_count = 0;
    uint64_t materialization_op_count = 0;

    uint64_t peak_temporary_elements = 0;
    uint64_t peak_intermediate_elements = 0;

    uint64_t estimated_execution_units = 0;
};

// One selected binary node in an exact contraction tree.  Steps are stored in
// postorder, so lhs/rhs subset results (when non-leaves) have already appeared
// earlier in the vector.  Source masks use original equation operand indices.
struct EinsumExactContractionStep {
    uint64_t lhs_source_mask = 0;
    uint64_t rhs_source_mask = 0;
    uint64_t result_source_mask = 0;

    // Retain the physical inputs actually seen by the selected pair planner so
    // diagnostics can explain why this realization won without reconstructing
    // transient DP state or creating Expression objects.
    EinsumLogicalOperandPlan lhs;
    EinsumLogicalOperandPlan rhs;
    std::vector<int32_t> surviving_labels;
    std::vector<int32_t> eliminated_labels;
    EinsumPairPhysicalCandidate physical_candidate;

    // Scalarized cost of this pair alone, plus the cumulative selected-subtree
    // cost at this postorder node. Raw pair components remain available on
    // physical_candidate.cost.
    uint64_t incremental_estimated_execution_units = 0;
    EinsumExactContractionCost cumulative_cost;
};

enum class EinsumContractionPlanningMode {
    EXACT,
    SIX_OPERAND_BRIDGE,
};

// Selected contraction result for the small-N optimized planner.  Three through
// five active operands use the exhaustive subset DP.  Six operands use the
// bridge search: force each possible original-operand pair in turn, then solve
// the resulting five-active-operand tail exactly with the same physical-state
// and cost machinery.
struct EinsumExactContractionPlan {
    std::vector<EinsumExactContractionStep> steps;
    EinsumLogicalOperandPlan result;
    EinsumExactContractionCost cost;
    EinsumContractionPlanningMode planning_mode = EinsumContractionPlanningMode::EXACT;

    // Non-zero only for SIX_OPERAND_BRIDGE.  This records the original source
    // pair forced into one bridge leaf; independent branches may appear before
    // that pair in executable postorder, so the seed cannot be inferred from
    // steps.front().
    uint64_t bridge_seed_pair_mask = 0;
};

// Selected bounded-beam contraction result for equations larger than the
// six-operand optimized bridge. Beam search contracts until only five active
// logical operands remain, then invokes the exact five-active-operand planner
// for each retained frontier state. Runtime lowers the selected postorder tree
// through the same pair execution machinery used by exact_contraction.
struct EinsumBeamContractionPlan {
    std::vector<EinsumExactContractionStep> steps;
    EinsumLogicalOperandPlan result;
    EinsumExactContractionCost cost;

    uint32_t beam_width = 0;
    uint32_t exact_tail_active_operands = 0;
    uint32_t beam_levels = 0;

    // Planner diagnostics accumulated across all beam levels. These counts are
    // deterministic for a fixed equation, dimensions, and beam width.
    uint64_t expanded_state_count = 0;
    uint64_t generated_state_count = 0;
    uint64_t deduplicated_state_count = 0;
    uint64_t retained_state_count = 0;
    uint64_t exact_tail_count = 0;
};

struct EinsumPlan {
    EinsumPlanKind kind = EinsumPlanKind::GENERAL;
    ResolvedEinsumEquation equation;

    // Generic fallback iteration order: requested output labels first, then
    // reduction labels.  Therefore reduction_axes is always a trailing range.
    std::vector<int32_t> iteration_labels;
    std::vector<uint64_t> iteration_dimensions;
    std::vector<uint32_t> reduction_axes;
    std::vector<EinsumOperandPlan> operands;
    std::vector<EinsumLogicalOperandPlan> logical_operands;

    // For a two-input equation, the existing whole-equation optimized plan is
    // now produced through the reusable pair planner below.  The legacy
    // matrix_multiply/pair_product fields remain as the execution-facing
    // compatibility surface for this behavior-preserving phase.
    std::optional<EinsumPairContractionPlan> pair_contraction;
    std::optional<EinsumMatrixMultiplyPlan> matrix_multiply;
    std::optional<EinsumPairProductPlan> pair_product;

    // For three through five inputs this is the exhaustive contraction tree.
    // Six inputs use the first-pair bridge into an exact five-active-operand
    // tail.  The legacy field name is retained because runtime lowering consumes
    // both modes through the same postorder contraction representation.
    std::optional<EinsumExactContractionPlan> exact_contraction;

    // Seven through MAX_BEAM_SOURCE_OPERANDS inputs are planned with bounded
    // beam search and executed through the selected postorder pair-contraction
    // tree. The whole-equation generic expression remains the fallback when no
    // executable contraction tree is available.
    std::optional<EinsumBeamContractionPlan> beam_contraction;
};

class EinsumPlanner {
   public:
    static constexpr uint32_t MAX_EXACT_ACTIVE_OPERANDS = 5;
    static constexpr uint32_t MAX_BRIDGED_ACTIVE_OPERANDS = 6;
    static constexpr uint32_t DEFAULT_BEAM_WIDTH = 32;
    // Contraction provenance is represented by a uint64_t source mask. Keep
    // one bit unavailable so every supported source count can construct the
    // full mask with (1 << count) - 1 without a 64-bit shift.
    static constexpr uint32_t MAX_SOURCE_OPERANDS = 63;
    static constexpr uint32_t MAX_BEAM_SOURCE_OPERANDS = MAX_SOURCE_OPERANDS;

    static EinsumPlan plan(const ResolvedEinsumEquation& equation,
                           const std::vector<std::vector<uint64_t>>& input_dimensions);

    static EinsumPlan parseAndPlan(const std::string& equation,
                                   const std::vector<std::vector<uint64_t>>& input_dimensions);

    // Deterministic, human-readable description of the selected exact tree.
    // Intended for tests, diagnostics, and benchmark output; planning remains
    // speculative and Expression-free. Returns an empty string when no exact
    // contraction was selected.
    static std::string describeExactContraction(const EinsumPlan& plan);

    // Deterministic summary of the beam tree selected for 7+ operands. Returns
    // an empty string when beam planning did not apply.
    static std::string describeBeamContraction(const EinsumPlan& plan);

    // Plan one binary contraction without requiring an original two-input
    // ResolvedEinsumEquation.  surviving_labels is ordered: it defines the
    // logical order of the persistent intermediate produced by this pair.
    // Labels from lhs/rhs that are not present in surviving_labels are reduced
    // during this pair contraction when algebraically valid.
    static EinsumPairContractionPlan planPair(const EinsumLogicalOperandPlan& lhs,
                                              const EinsumLogicalOperandPlan& rhs,
                                              const std::vector<int32_t>& surviving_labels);
};

}  // namespace ThorImplementation
