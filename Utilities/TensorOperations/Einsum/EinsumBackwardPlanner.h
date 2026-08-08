#pragma once

#include "Utilities/TensorOperations/Einsum/EinsumParser.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ThorImplementation {

// Thor graph tensors carry a runtime batch axis that is not part of the
// user-visible einsum equation. Backward planning therefore uses one internal
// label outside the parser's non-negative user/ellipsis label namespace. The
// implementation layer will prepend this label/axis to every physical input,
// output, upstream gradient, and operand gradient so examples can never be
// contracted with one another.
struct EinsumLayerBatchContract {
    static constexpr int32_t kImplicitBatchLabel = -1;

    [[nodiscard]] static bool isImplicitBatchLabel(int32_t label) {
        return label == kImplicitBatchLabel;
    }

    [[nodiscard]] static std::vector<int32_t> prependImplicitBatchLabel(const std::vector<int32_t>& feature_labels);
    [[nodiscard]] static std::vector<uint64_t> prependBatchDimension(uint64_t batch_size,
                                                                     const std::vector<uint64_t>& feature_dimensions);
};

// Raw einsum contraction used to form the logical gradient for one operand
// occurrence before restoring that operand's local broadcast/diagonal shape.
// Input 0 is always dOutput. Inputs 1..N are the original forward operands
// listed by other_operand_indices in the same order.
struct EinsumBackwardContractionPlan {
    std::vector<int32_t> upstream_gradient_feature_labels;
    std::vector<uint32_t> other_operand_indices;
    std::vector<std::vector<int32_t>> other_operand_feature_axis_labels;

    // Unique target labels that are actually available from dOutput and/or the
    // other forward operands. Target-only labels reduced by the forward pass
    // are absent here and are restored by missing_axis_expansions below.
    std::vector<int32_t> output_feature_labels;
    std::vector<uint64_t> output_feature_dimensions;

    // Same normalized equation with Thor's implicit runtime batch label
    // prepended. These fields are intentionally ready for Patch 23's physical
    // layer without needing to reconstruct the batch contract there.
    std::vector<std::vector<int32_t>> physical_input_axis_labels;
    std::vector<int32_t> physical_output_axis_labels;
};

// If one operand had local extent 1 for a label whose equation-wide extent is
// larger, its raw gradient is produced at the equation-wide extent and must be
// summed back into the singleton. The reduction keeps the axis so subsequent
// target-label ordering is stable.
struct EinsumBackwardBroadcastReductionPlan {
    int32_t label = -1;
    uint32_t contraction_output_feature_axis = 0;
    uint64_t source_dimension = 0;
    uint64_t target_dimension = 1;
};

// A target label can be present in the backward contraction only through
// singleton other operands even though the target itself owns extent N. The
// raw contraction keeps that size-1 axis, then broadcasts it back to the target
// extent. This is distinct from a completely missing target-only label.
struct EinsumBackwardExistingAxisExpansionPlan {
    int32_t label = -1;
    uint32_t contraction_output_feature_axis = 0;
    uint32_t target_unique_feature_axis = 0;
    uint64_t source_dimension = 1;
    uint64_t target_dimension = 0;
};

// A label that appears only in the target operand and was reduced away by the
// forward equation is not present in the backward contraction inputs. The raw
// gradient therefore cannot name it as an einsum output and must insert and
// broadcast this logical axis afterward.
struct EinsumBackwardMissingAxisExpansionPlan {
    int32_t label = -1;
    uint32_t target_unique_feature_axis = 0;
    uint64_t target_dimension = 0;
};

// Repeated labels are diagonal views in forward. Backward first computes a
// gradient over the collapsed unique logical label, then scatters it back onto
// all repeated source axes, leaving off-diagonal elements zero.
struct EinsumBackwardDiagonalScatterPlan {
    int32_t label = -1;
    uint32_t source_unique_feature_axis = 0;
    std::vector<uint32_t> target_feature_axes;
    uint64_t dimension = 0;
};

struct EinsumOperandBackwardPlan {
    uint32_t operand_index = 0;

    // The target operand after repeated-label diagonal axes have been collapsed
    // logically, in first-physical-occurrence order.
    std::vector<int32_t> target_unique_feature_labels;
    std::vector<uint64_t> target_unique_feature_dimensions;

    // Expected execution order is: raw contraction; keep-axis singleton
    // reductions; existing/missing axis broadcast restoration; diagonal scatter.
    EinsumBackwardContractionPlan contraction;
    std::vector<EinsumBackwardBroadcastReductionPlan> broadcast_reductions;
    // Existing raw contraction axes whose only backward-input extent is 1,
    // while the target operand itself owns a larger forward extent.
    std::vector<EinsumBackwardExistingAxisExpansionPlan> existing_axis_expansions;
    std::vector<EinsumBackwardMissingAxisExpansionPlan> missing_axis_expansions;
    std::vector<EinsumBackwardDiagonalScatterPlan> diagonal_scatters;

    // Dense feature shape required by the upstream Thor connection after all
    // postprocessing. The physical error tensor will be [batch, ...shape].
    std::vector<uint64_t> final_feature_dimensions;
};

// CPU-only layer/backward metadata. This deliberately does not stamp or execute
// any Expression work; Patch 23/24 can consume it directly.
struct EinsumLayerBackwardPlan {
    std::string equation;
    ResolvedEinsumEquation feature_equation;

    // Normalized forward physical labels with the implicit batch axis prepended.
    std::vector<std::vector<int32_t>> physical_forward_input_axis_labels;
    std::vector<int32_t> physical_forward_output_axis_labels;

    // One gradient plan per operand occurrence. Occurrences are never deduped:
    // if the same symbolic Tensor is supplied twice, both product-rule terms
    // remain explicit for the implementation/network layer to accumulate.
    std::vector<EinsumOperandBackwardPlan> operand_gradients;
};

class EinsumBackwardPlanner {
   public:
    // input_feature_dimensions intentionally excludes Thor's runtime batch
    // dimension. The user equation therefore describes one example, exactly as
    // other DeepLearning/Api layer shapes do.
    [[nodiscard]] static EinsumLayerBackwardPlan parseAndPlan(
        const std::string& equation,
        const std::vector<std::vector<uint64_t>>& input_feature_dimensions);
};

}  // namespace ThorImplementation
