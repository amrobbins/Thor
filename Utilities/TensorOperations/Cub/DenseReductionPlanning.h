#pragma once

#include "DeepLearning/Implementation/Tensor/DataType.h"
#include "Utilities/TensorOperations/Cub/CubReductionPassRole.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace ThorImplementation::CubReductionInternal {

/** Capability family for one nominated site in an ordinary dense row-major reduction problem. */
enum class DenseReducerFamily : uint8_t {
    R = 0,
    KR = 1,
    RK = 2,
};

/** Aggregate semantics carried between physical passes. */
enum class DenseReductionAggregateKind : uint8_t {
    Value = 0,
    Arg = 1,
};

/** Whether the current physical input is the user's original values or a reducer-produced aggregate tensor. */
enum class DenseReductionInputState : uint8_t {
    OriginalInput = 0,
    IntermediateAggregate = 1,
};

/**
 * Meaning of one current dense run.
 *
 * OriginalReduction coordinates still correspond to coordinates in the user's original reduction domain and therefore
 * contribute coordinate * original_arg_domain_stride when an ARG candidate is constructed. IntermediateReduction is a
 * reducer-created shard/partial dimension: its elements already carry their winning original indices, so its local
 * coordinate must never be added to that index again.
 */
enum class DenseReductionRunKind : uint8_t {
    Retained = 0,
    OriginalReduction = 1,
    IntermediateReduction = 2,
};

struct DenseReductionRun {
    DenseReductionRunKind kind = DenseReductionRunKind::Retained;
    uint64_t extent = 1;

    // Meaningful only for OriginalReduction. Preserved as zero for retained/intermediate runs so accidental use is
    // visible in diagnostics/tests instead of silently manufacturing an ARG coordinate contribution.
    uint64_t original_arg_domain_stride = 0;

    bool operator==(const DenseReductionRun&) const = default;
};

/** Requested final storage. VALUE uses only value_dtype; ARG may request either or both outputs. */
struct DenseReductionOutputSpec {
    bool produce_value = true;
    DataType value_dtype = DataType::FP32;
    bool produce_index = false;
    DataType index_dtype = DataType::UINT32;

    bool operator==(const DenseReductionOutputSpec&) const = default;
};

/**
 * The complete state needed to plan the next ordinary-dense physical pass.
 *
 * runs always describes the current tensor rather than the original tensor. A partial pass replaces its nominated R
 * extent with the emitted partial count; a complete reduction of that site removes it, after which adjacent retained
 * runs are collapsed. No site identity survives elimination. The only history retained is aggregate semantics needed
 * to correctly consume intermediate state.
 */
struct DenseReductionProblem {
    std::vector<DenseReductionRun> runs;
    DenseReductionAggregateKind aggregate_kind = DenseReductionAggregateKind::Value;
    DenseReductionInputState input_state = DenseReductionInputState::OriginalInput;

    // ARG partials carry FP32 values plus this original-index payload. The width is selected from the complete original
    // reduction domain and remains fixed through every intermediate pass.
    DataType carried_arg_index_dtype = DataType::UINT32;

    DenseReductionOutputSpec final_output;

    // True until the Complete/Final pass applies the operation-specific finalizer and output scale. Input transforms
    // are required exactly when input_state == OriginalInput.
    bool finalizer_pending = true;

    bool operator==(const DenseReductionProblem&) const = default;
};

/** Ephemeral nomination of one reduced run in one current problem. It is deliberately not a persistent site ID. */
struct DenseReductionSite {
    uint32_t run_index = 0;

    bool operator==(const DenseReductionSite&) const = default;
};

/**
 * Type-erased family-local physical plan payload.
 *
 * The overall planner must never inspect CUDA geometry. Concrete family planners introduced after DENSE-FOUNDATION-1
 * derive their own payload from this base and retain ownership through DenseReductionCandidate.
 */
struct DenseReductionPhysicalPlan {
    virtual ~DenseReductionPhysicalPlan() = default;
};

/** Common family-independent result of proposing one physical pass for one nominated site. */
struct DenseReductionCandidate {
    DenseReducerFamily family = DenseReducerFamily::R;
    DenseReductionSite site;
    std::shared_ptr<const DenseReductionPhysicalPlan> physical_plan;
    DenseReductionProblem next_problem;
    uint64_t next_bytes = 0;
    uint64_t scratch_bytes = 0;
};

using DensePassRole = CubReductionPassRole;

[[nodiscard]] bool isDenseReductionRunReduced(DenseReductionRunKind kind);
[[nodiscard]] bool hasDenseReductionSites(const DenseReductionProblem& problem);

/**
 * Constructs the canonical initial problem for an ordinary dense row-major tensor.
 *
 * Singleton dimensions normally disappear while adjacent traversal-equivalent runs collapse. If every selected
 * reduction axis is singleton, one extent-1 OriginalReduction run is retained at the first selected logical position so
 * the required transform/finalizer pass remains representable. That transient site disappears after the pass.
 */
[[nodiscard]] DenseReductionProblem makeInitialDenseReductionProblem(
    const std::vector<uint64_t>& input_dimensions,
    const std::vector<uint32_t>& reduction_axes,
    DenseReductionAggregateKind aggregate_kind,
    const DenseReductionOutputSpec& final_output,
    DataType carried_arg_index_dtype = DataType::UINT32);

/** Returns all currently reducible runs in stable current-run order. */
[[nodiscard]] std::vector<DenseReductionSite> enumerateDenseReductionSites(const DenseReductionProblem& problem);

/** Classifies the physical topology of one nominated current R without considering any other site as a candidate. */
[[nodiscard]] DenseReducerFamily classifyDenseReducerFamily(const DenseReductionProblem& problem,
                                                             DenseReductionSite site);

/**
 * Produces the fresh problem after one pass reduces the nominated current R to next_reduction_extent partials.
 * next_reduction_extent == 1 eliminates that R. Values >1 keep an IntermediateReduction run of that size. The only
 * equal-size transition accepted is the initial semantic extent-1 reduction required when all selected axes are
 * singleton.
 */
[[nodiscard]] DenseReductionProblem advanceDenseReductionProblem(const DenseReductionProblem& problem,
                                                                 DenseReductionSite site,
                                                                 uint64_t next_reduction_extent);

/** Derives semantic pass role only from original-vs-aggregate input and whether the candidate finishes all reductions. */
[[nodiscard]] DensePassRole deriveDensePassRole(const DenseReductionProblem& current_problem,
                                                const DenseReductionProblem& next_problem);

/** Original ARG-domain stride contributed by this run's local coordinate, or nullopt for retained/partial runs. */
[[nodiscard]] std::optional<uint64_t> denseArgOriginalIndexStride(const DenseReductionRun& run);

/** Current aggregate/final-output bytes represented by a candidate's next problem. */
[[nodiscard]] uint64_t denseReductionProblemBytes(const DenseReductionProblem& problem);

}  // namespace ThorImplementation::CubReductionInternal
