#pragma once

#include "Utilities/TensorOperations/Einsum/EinsumParser.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace ThorImplementation {

enum class EinsumPlanKind {
    UNARY,
    ELEMENTWISE,
    REDUCTION,
    GEMM,
    BATCHED_GEMM,
    GENERAL,
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
// When requires_materialized_permutation is false, the operand can be viewed
// directly in the required grouping.  transpose then indicates whether the
// two matrix groups are reversed in memory and can be handled by a GEMM
// transpose flag instead of a data movement.
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
    std::vector<int32_t> contraction_labels;
    std::vector<int32_t> rhs_free_labels;
    std::vector<int32_t> canonical_output_labels;

    EinsumMatrixOperandPlan lhs;
    EinsumMatrixOperandPlan rhs;

    // For each requested output axis, gives the corresponding axis in
    // canonical_output_labels.  Identity means the GEMM result already has the
    // requested einsum output order.
    std::vector<uint32_t> output_permutation;

    uint64_t batch_count = 1;
    uint64_t m = 1;
    uint64_t n = 1;
    uint64_t k = 1;

    bool requires_output_permutation = false;

    // True only when diagonal extraction, materialized input permutations,
    // shared-label broadcasting, and output permutation are all unnecessary.
    // Reshaping contiguous label groups and using GEMM transpose flags do not
    // make a plan non-direct.
    bool direct = false;
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

    std::optional<EinsumMatrixMultiplyPlan> matrix_multiply;
};

class EinsumPlanner {
   public:
    static EinsumPlan plan(const ResolvedEinsumEquation& equation,
                           const std::vector<std::vector<uint64_t>>& input_dimensions);

    static EinsumPlan parseAndPlan(const std::string& equation,
                                   const std::vector<std::vector<uint64_t>>& input_dimensions);
};

}  // namespace ThorImplementation
