#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/SqueezeAxes.h"

namespace ThorImplementation {

inline constexpr const char* DEFAULT_BACKWARD_UPSTREAM_INPUT_NAME = "__grad_output";

// A caller may retain selected forward values and bind them as
// explicit inputs of its backward graph.  The key is the node index in the
// physical forward expression; forwardValue() stops at that node instead of
// replaying its forward subtree.  This is intentionally a value-binding
// mechanism, not checkpointing: the named backward input must be supplied by
// the real forward execution.
using SavedForwardValueInputNames = std::unordered_map<uint32_t, std::string>;

// A backward expression built in forward-value-requirement mode records every
// non-root primal value that must be supplied from the real forward execution.
// BR1 only declares these dependencies; BR2 is responsible for teaching generic
// CustomLayer forward plans to retain and bind them automatically.
enum class ForwardValueRequirementKind : uint8_t {
    NodeOutput = 0,
    MatmulEpilogueAux = 1,
};

struct ForwardValueRequirement {
    uint32_t forward_node_index = UINT32_MAX;
    std::string backward_input_name;
    ForwardValueRequirementKind kind = ForwardValueRequirementKind::NodeOutput;
};

struct BackwardBuildResult {
    PhysicalOutputs outputs;
    std::vector<ForwardValueRequirement> forward_value_requirements;
};

PhysicalOutputs buildBackwardOutputs(
    const PhysicalOutputs& forward_outputs,
    const std::vector<std::string>& wrt_names = {},
    const std::optional<std::string>& upstream_input_name = std::nullopt,
    const std::optional<std::unordered_map<std::string, std::vector<uint64_t>>>& forward_input_dims = std::nullopt,
    bool accumulate_grad_outputs = false);

PhysicalOutputs buildBackwardOutputs(
    const PhysicalOutputs& forward_outputs,
    const std::vector<std::string>& wrt_names,
    const std::unordered_map<std::string, std::string>& upstream_input_names_by_output,
    const std::optional<std::unordered_map<std::string, std::vector<uint64_t>>>& forward_input_dims = std::nullopt,
    bool accumulate_grad_outputs = false);

// As above, but records the runtime dtype of each upstream gradient seed.  The map
// is keyed by forward-output name, matching upstream_input_names_by_output.  This
// lets autodiff deliberately lower widened FP32 gradients before low-precision
// matrix-gradient GEMMs without guessing from the forward node's output dtype.
PhysicalOutputs buildBackwardOutputs(
    const PhysicalOutputs& forward_outputs,
    const std::vector<std::string>& wrt_names,
    const std::unordered_map<std::string, std::string>& upstream_input_names_by_output,
    const std::unordered_map<std::string, DataType>& upstream_input_dtypes_by_output,
    const std::optional<std::unordered_map<std::string, std::vector<uint64_t>>>& forward_input_dims = std::nullopt,
    bool accumulate_grad_outputs = false,
    const SavedForwardValueInputNames& saved_forward_value_input_names = {});

// Existing buildBackwardOutputs(...) overloads intentionally retain the legacy
// recompute-if-unbound behavior until BR2 wires these dependencies through the
// generic forward plan.
//
// Build a backward graph without implicitly replaying computed primal values.
// Instead, forwardValue() turns each required non-root materialized value into a
// synthetic backward input and returns the corresponding dependency list.
// Explicit saved_forward_value_input_names may be supplied to preserve stable
// ABI names for values already retained by specialized callers. Requirement
// mode is the normal no-replay training contract: every computed
// primal dependency must be supplied by the real forward as a retained node or
// backend auxiliary state. Explicit recompute remains available only through the
// legacy/opt-in compatibility path.
BackwardBuildResult buildBackwardOutputsWithForwardValueRequirements(
    const PhysicalOutputs& forward_outputs,
    const std::vector<std::string>& wrt_names,
    const std::unordered_map<std::string, std::string>& upstream_input_names_by_output,
    const std::unordered_map<std::string, DataType>& upstream_input_dtypes_by_output,
    const std::optional<std::unordered_map<std::string, std::vector<uint64_t>>>& forward_input_dims = std::nullopt,
    bool accumulate_grad_outputs = false,
    const SavedForwardValueInputNames& saved_forward_value_input_names = {});

PhysicalOutputs buildBackwardOutputs(
    const PhysicalOutputs& forward_outputs,
    const std::vector<std::string>& wrt_names,
    const std::unordered_map<std::string, std::string>& upstream_input_names_by_output,
    const std::unordered_map<std::string, uint32_t>& upstream_node_indices_by_output,
    const std::optional<std::unordered_map<std::string, std::vector<uint64_t>>>& forward_input_dims = std::nullopt,
    bool accumulate_grad_outputs = false);

// Builds the initial backward equation template used by FusedEquation::compileBackward when
// forward input dimensions are not known yet. Shape-sensitive backward aliases may be
// placeholders in this template; FusedEquation rebuilds the real backward graph with
// runtime forward dimensions during stamping/shape-specialization.
PhysicalOutputs buildDeferredShapeBackwardOutputsTemplate(
    const PhysicalOutputs& forward_outputs,
    const std::vector<std::string>& wrt_names = {},
    const std::optional<std::string>& upstream_input_name = std::nullopt,
    bool accumulate_grad_outputs = false);

PhysicalOutputs buildDeferredShapeBackwardOutputsTemplate(
    const PhysicalOutputs& forward_outputs,
    const std::vector<std::string>& wrt_names,
    const std::unordered_map<std::string, std::string>& upstream_input_names_by_output,
    bool accumulate_grad_outputs = false);

}  // namespace ThorImplementation
