#pragma once

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/SqueezeAxes.h"

namespace ThorImplementation {

inline constexpr const char* DEFAULT_BACKWARD_UPSTREAM_INPUT_NAME = "__grad_output";

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
    bool accumulate_grad_outputs = false);

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
