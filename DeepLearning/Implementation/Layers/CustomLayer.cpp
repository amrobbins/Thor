#include <optional>
#include "DeepLearning/Implementation/Layers/CustomLayer.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <set>
#include <stdexcept>
#include <unordered_set>

#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Diagnostics/TrainingDiagnostics.h"
#include "DeepLearning/Implementation/Layers/LayerSubmitDiagnostics.h"
#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"
#include "Utilities/Expression/AutoDiff.h"
using namespace std;

namespace ThorImplementation {

namespace {
uint64_t bestEffortExecutionPlanFlopCount(const StampedExecutionPlan& plan) noexcept {
    try {
        return plan.flopCount();
    } catch (...) {
        // Layer-submit diagnostics are optional telemetry.  Never allow FLOP
        // accounting to fail an otherwise successful layer execution.
        return 0;
    }
}

std::set<std::string> toNameSet(const std::vector<std::string>& names) { return std::set<std::string>(names.begin(), names.end()); }

bool isInternalExpressionInputName(const std::string& name) { return name.rfind("__", 0) == 0; }

std::string joinNames(const std::set<std::string>& names) {
    std::string result;
    for (const auto& name : names) {
        result += name + " ";
    }
    return result;
}

std::string optimizerFusionNamePrefix(const std::string& parameterName) { return "__optimizer_fused_" + parameterName + "__"; }

std::string optimizerFusionOutputName(const std::string& parameterName, const std::string& outputName) {
    return optimizerFusionNamePrefix(parameterName) + outputName;
}

#if THOR_ENABLE_TRAINING_UPDATE_DIAGNOSTICS
bool trainingUpdateDiagnosticsEnabled() {
    const char* value = std::getenv("THOR_TRAINING_UPDATE_DIAGNOSTICS");
    return value != nullptr && value[0] != '\0' && std::string(value) != "0";
}
#else
constexpr bool trainingUpdateDiagnosticsEnabled() { return false; }
#endif

std::string joinNames(const std::vector<std::string>& names) {
    std::string result;
    for (const auto& name : names) {
        if (!result.empty()) {
            result += ",";
        }
        result += name;
    }
    if (result.empty()) {
        return "<none>";
    }
    return result;
}

std::string joinNames(const std::unordered_set<std::string>& names) {
    std::vector<std::string> sorted(names.begin(), names.end());
    std::sort(sorted.begin(), sorted.end());
    return joinNames(sorted);
}

std::string joinNames(const std::unordered_map<std::string, uint64_t>& batchSizes) {
    std::vector<std::string> items;
    items.reserve(batchSizes.size());
    for (const auto& [name, batchSize] : batchSizes) {
        items.push_back(name + "=" + std::to_string(batchSize));
    }
    std::sort(items.begin(), items.end());
    return joinNames(items);
}

PreparedDynamicExpression::TensorMap filterTensorInputsForPhysicalOutputs(
    const PreparedDynamicExpression::TensorMap& availableInputs,
    const PhysicalOutputs& outputs) {
    PreparedDynamicExpression::TensorMap filteredInputs;
    if (outputs.expr == nullptr) {
        return filteredInputs;
    }

    for (const NamedInput& input : outputs.expr->inputs) {
        if (input.kind != NamedInput::Kind::Tensor) {
            continue;
        }

        auto it = availableInputs.find(input.name);
        if (it == availableInputs.end()) {
            throw runtime_error("CustomLayer backward plan missing required tensor input: " + input.name);
        }
        filteredInputs.emplace(input.name, it->second);
    }
    return filteredInputs;
}

void bindRetainedForwardValues(const BackwardBuildResult& backwardBuild,
                               const StampedExecutionPlan& forwardPlan,
                               PreparedDynamicExpression::TensorMap& inputs) {
    for (const ForwardValueRequirement& requirement : backwardBuild.forward_value_requirements) {
        if (requirement.forward_node_index == UINT32_MAX || requirement.backward_input_name.empty()) {
            throw runtime_error("CustomLayer backward contains an incomplete saved-forward requirement.");
        }
        Tensor retained;
        switch (requirement.kind) {
            case ForwardValueRequirementKind::NodeOutput:
                retained = forwardPlan.retainedForwardValue(
                    requirement.conditional_branch_path, requirement.forward_node_index);
                break;
            case ForwardValueRequirementKind::MatmulEpilogueAux:
                retained = forwardPlan.retainedForwardEpilogueAux(
                    requirement.conditional_branch_path, requirement.forward_node_index);
                break;
            case ForwardValueRequirementKind::ConditionalPredicate:
                retained = forwardPlan.retainedConditionalPredicate(requirement.conditional_branch_path);
                break;
            default:
                throw runtime_error("CustomLayer backward contains an unknown saved-forward requirement kind.");
        }
        auto [it, inserted] = inputs.emplace(requirement.backward_input_name, retained);
        if (!inserted && !(it->second == retained)) {
            throw runtime_error("CustomLayer saved-forward input name collides with a different tensor: " +
                                requirement.backward_input_name);
        }
    }
}

PreparedDynamicExpression::TensorScalarMap filterTensorScalarInputsForPhysicalOutputs(
    const PreparedDynamicExpression::TensorScalarMap& availableInputs,
    const PhysicalOutputs& outputs) {
    PreparedDynamicExpression::TensorScalarMap filteredInputs;
    if (outputs.expr == nullptr) {
        return filteredInputs;
    }

    for (const NamedInput& input : outputs.expr->inputs) {
        if (input.kind != NamedInput::Kind::TensorRuntimeScalar) {
            continue;
        }

        auto it = availableInputs.find(input.name);
        if (it == availableInputs.end()) {
            throw runtime_error("CustomLayer backward plan missing required tensor runtime scalar input: " + input.name);
        }
        filteredInputs.emplace(input.name, it->second);
    }
    return filteredInputs;
}

void applyCustomLayerInputGradientStorageContractInPlace(PhysicalOutputs& outputs) {
    if (outputs.expr == nullptr) {
        throw runtime_error("CustomLayer backward dtype specialization requires non-null PhysicalOutputs.expr.");
    }

    if (!outputs.isConditional()) {
        // A CustomLayer expression may intentionally promote a physical feature input
        // for forward compute, for example BF16 storage -> FP32 logical value via
        // context.input(..., output_dtype=FP32).  The gradient exposed back to the
        // surrounding graph is nevertheless the gradient of the *physical input
        // tensor* and must therefore materialize in that tensor's storage dtype.
        //
        // resolveOutputsDTypesInPlace() defaults INPUT.backward_output_dtype to the
        // logical forward output dtype.  Without correcting that default here, a
        // promoted BF16 input asks AutoDiff for an FP32 dInput even though the
        // already-connected upstream error buffer is BF16.  FusedEquation then
        // correctly rejects stamping that FP32 compiled output into the BF16
        // preallocated tensor.  Keep FP32 backward_compute_dtype when requested,
        // but restore the graph-boundary storage contract for every tensor INPUT.
        for (ExprNode& node : outputs.expr->nodes) {
            if (node.op == ExprOp::INPUT && node.input_tensor_dtype.has_value()) {
                node.backward_output_dtype = node.input_tensor_dtype.value();
            }
        }
        return;
    }

    applyCustomLayerInputGradientStorageContractInPlace(outputs.conditional->predicate);
    applyCustomLayerInputGradientStorageContractInPlace(outputs.conditional->then_branch);
    applyCustomLayerInputGradientStorageContractInPlace(outputs.conditional->else_branch);
}

std::unordered_map<uint32_t, NamedInput> inputBySlot(const PhysicalExpression& expr) {
    std::unordered_map<uint32_t, NamedInput> bySlot;
    for (const NamedInput& input : expr.inputs) {
        bySlot.emplace(input.slot, input);
    }
    return bySlot;
}

uint32_t appendInputNode(PhysicalExpression& expr, const std::string& name, NamedInput::Kind kind) {
    ExprNode node{};
    switch (kind) {
        case NamedInput::Kind::Tensor:
            node.op = ExprOp::INPUT;
            break;
        case NamedInput::Kind::RuntimeScalarFp32:
            node.op = ExprOp::RUNTIME_SCALAR;
            break;
        case NamedInput::Kind::TensorRuntimeScalar:
            node.op = ExprOp::TENSOR_RUNTIME_SCALAR;
            break;
    }
    node.input_slot = expr.getOrCreateInputSlot(name, kind);
    const uint32_t nodeIndex = static_cast<uint32_t>(expr.nodes.size());
    expr.nodes.push_back(std::move(node));
    return nodeIndex;
}

uint32_t appendInputNode(PhysicalExpression& expr, const NamedInput& input) {
    return appendInputNode(expr, input.name, input.kind);
}

uint32_t appendTensorInputNode(PhysicalExpression& expr, const std::string& name) {
    return appendInputNode(expr, name, NamedInput::Kind::Tensor);
}

uint32_t cloneExpressionNodeWithInputReplacements(const PhysicalExpression& src,
                                                  uint32_t srcNodeIndex,
                                                  PhysicalExpression& dst,
                                                  const std::unordered_map<std::string, uint32_t>& inputReplacements,
                                                  const std::unordered_map<uint32_t, NamedInput>& srcInputBySlot,
                                                  uint32_t cudaKernelExpressionOffset,
                                                  std::unordered_map<uint32_t, uint32_t>& clonedNodes) {
    auto existing = clonedNodes.find(srcNodeIndex);
    if (existing != clonedNodes.end()) {
        return existing->second;
    }
    if (srcNodeIndex >= src.nodes.size()) {
        throw runtime_error("CustomLayer fused CustomLoss gradient expression has a node index out of range.");
    }

    const ExprNode& srcNode = src.nodes[srcNodeIndex];
    if (srcNode.op == ExprOp::INPUT || srcNode.op == ExprOp::RUNTIME_SCALAR || srcNode.op == ExprOp::TENSOR_RUNTIME_SCALAR) {
        auto inputIt = srcInputBySlot.find(srcNode.input_slot);
        if (inputIt == srcInputBySlot.end()) {
            throw runtime_error("CustomLayer fused CustomLoss gradient expression contains an input node with an unknown slot.");
        }

        auto replacementIt = inputReplacements.find(inputIt->second.name);
        if (replacementIt == inputReplacements.end()) {
            throw runtime_error("CustomLayer fused CustomLoss gradient expression contains unsupported input '" + inputIt->second.name +
                                "'. Only the predictions and labels inputs are supported for fused CustomLoss gradients.");
        }
        clonedNodes[srcNodeIndex] = replacementIt->second;
        return replacementIt->second;
    }

    auto cloneRef = [&](uint32_t maybeNodeIndex) -> uint32_t {
        if (maybeNodeIndex == UINT32_MAX) {
            return UINT32_MAX;
        }
        return cloneExpressionNodeWithInputReplacements(
            src, maybeNodeIndex, dst, inputReplacements, srcInputBySlot, cudaKernelExpressionOffset, clonedNodes);
    };

    ExprNode cloned = srcNode;
    cloned.lhs = cloneRef(srcNode.lhs);
    cloned.rhs = cloneRef(srcNode.rhs);
    cloned.aux = cloneRef(srcNode.aux);
    cloned.alpha_node = cloneRef(srcNode.alpha_node);
    cloned.beta_node = cloneRef(srcNode.beta_node);
    cloned.matmul_epilogue_aux = cloneRef(srcNode.matmul_epilogue_aux);
    // ROPE carries auxiliary expression dependencies outside lhs/rhs/aux. When the
    // forward subtree is cloned into a fused CustomLoss backward expression, those
    // node references must be remapped just like Attention's auxiliary inputs below.
    // Leaving the source-expression indices in place can make the cloned ROPE node
    // read an unrelated destination node as its position ids/effective length.
    cloned.rope_effective_sequence_length_node = cloneRef(srcNode.rope_effective_sequence_length_node);
    cloned.rope_position_ids_node = cloneRef(srcNode.rope_position_ids_node);
    cloned.attention_seq_len_q_node = cloneRef(srcNode.attention_seq_len_q_node);
    cloned.attention_seq_len_kv_node = cloneRef(srcNode.attention_seq_len_kv_node);
    cloned.attention_ragged_offset_q_node = cloneRef(srcNode.attention_ragged_offset_q_node);
    cloned.attention_ragged_offset_kv_node = cloneRef(srcNode.attention_ragged_offset_kv_node);
    cloned.attention_page_table_k_node = cloneRef(srcNode.attention_page_table_k_node);
    cloned.attention_page_table_v_node = cloneRef(srcNode.attention_page_table_v_node);
    cloned.attention_dropout_seed_node = cloneRef(srcNode.attention_dropout_seed_node);
    cloned.attention_dropout_offset_node = cloneRef(srcNode.attention_dropout_offset_node);
    cloned.attention_descale_q_node = cloneRef(srcNode.attention_descale_q_node);
    cloned.attention_descale_k_node = cloneRef(srcNode.attention_descale_k_node);
    cloned.attention_descale_v_node = cloneRef(srcNode.attention_descale_v_node);
    cloned.attention_descale_s_node = cloneRef(srcNode.attention_descale_s_node);
    cloned.attention_scale_s_node = cloneRef(srcNode.attention_scale_s_node);
    cloned.attention_scale_o_node = cloneRef(srcNode.attention_scale_o_node);
    cloned.attention_amax_s_node = cloneRef(srcNode.attention_amax_s_node);
    cloned.attention_amax_o_node = cloneRef(srcNode.attention_amax_o_node);
    for (uint32_t& inputNode : cloned.cuda_kernel_input_nodes) {
        inputNode = cloneRef(inputNode);
    }
    if (cloned.cuda_kernel_spec_index != UINT32_MAX) {
        cloned.cuda_kernel_spec_index += cudaKernelExpressionOffset;
    }

    const uint32_t clonedIndex = static_cast<uint32_t>(dst.nodes.size());
    dst.nodes.push_back(std::move(cloned));
    clonedNodes[srcNodeIndex] = clonedIndex;
    return clonedIndex;
}

std::string customLossFusedLabelsInputName(uint32_t outputFlatIndex) {
    return "__custom_loss_fused_labels_" + std::to_string(outputFlatIndex);
}

std::string customLossFusedBatchValidityMaskInputName(uint32_t outputFlatIndex) {
    return "__custom_loss_fused_batch_validity_mask_" + std::to_string(outputFlatIndex);
}

std::string customLossFusedSeedInputName(const std::string& outputName) {
    return "__custom_loss_fused_seed_" + outputName;
}


}  // namespace

CustomLayer::CustomLayer(DynamicExpression expr,
                         const TensorPlacement& placement,
                         const std::vector<std::shared_ptr<PhysicalParameter>>& parameters,
                         bool inferenceOnly,
                         int64_t stampedId)
    : CustomLayer(std::move(expr),
                  std::vector<std::string>{"feature_input"},
                  std::vector<std::string>{"feature_output"},
                  placement,
                  parameters,
                  inferenceOnly,
                  stampedId) {}

CustomLayer::CustomLayer(DynamicExpression expr,
                         std::vector<std::string> inputNames,
                         std::vector<std::string> outputNames,
                         const TensorPlacement& placement,
                         const std::vector<std::shared_ptr<PhysicalParameter>>& parameters,
                         bool inferenceOnly,
                         int64_t stampedId)
    : CustomLayer(std::move(expr),
                  std::move(inputNames),
                  std::move(outputNames),
                  placement,
                  parameters,
                  inferenceOnly,
                  stampedId,
                  {}) {}

CustomLayer::CustomLayer(DynamicExpression expr,
                         std::vector<std::string> inputNames,
                         std::vector<std::string> outputNames,
                         const TensorPlacement& placement,
                         const std::vector<std::shared_ptr<PhysicalParameter>>& parameters,
                         bool inferenceOnly,
                         int64_t stampedId,
                         std::vector<DeclaredOutputDescriptor> declaredOutputDescriptors,
                         bool usesBatchValidity,
                         bool requiresFullBatch,
                         std::vector<bool> inputDimensionsIncludeBatch,
                         std::optional<uint32_t> fixedBatchCapacity)
    : CustomLayer(std::move(expr),
                  std::move(inputNames),
                  std::move(outputNames),
                  placement,
                  parameters,
                  inferenceOnly,
                  stampedId,
                  std::move(declaredOutputDescriptors),
                  usesBatchValidity,
                  requiresFullBatch,
                  std::move(inputDimensionsIncludeBatch),
                  fixedBatchCapacity,
                  {}) {}

CustomLayer::CustomLayer(DynamicExpression expr,
                         std::vector<std::string> inputNames,
                         std::vector<std::string> outputNames,
                         const TensorPlacement& placement,
                         const std::vector<std::shared_ptr<PhysicalParameter>>& parameters,
                         bool inferenceOnly,
                         int64_t stampedId,
                         std::vector<DeclaredOutputDescriptor> declaredOutputDescriptors,
                         bool usesBatchValidity,
                         bool requiresFullBatch,
                         std::vector<bool> inputDimensionsIncludeBatch,
                         std::optional<uint32_t> fixedBatchCapacity,
                         std::set<std::string> trustedReservedInputNames)
    : TrainableLayer(placement, inferenceOnly, stampedId),
      layerDefinitionExpression(std::move(expr)),
      batchValidityMaskEnabled(usesBatchValidity),
      fullBatchRequired(requiresFullBatch),
      inputNames(std::move(inputNames)),
      outputNames(std::move(outputNames)),
      declaredOutputDescriptors(std::move(declaredOutputDescriptors)),
      inputDimensionsIncludeBatch(std::move(inputDimensionsIncludeBatch)),
      fixedBatchCapacity(fixedBatchCapacity) {
    if (batchValidityMaskEnabled && fullBatchRequired)
        throw runtime_error("CustomLayer cannot both use batch validity and require a full batch.");
    validatePortNames(this->inputNames, "input", trustedReservedInputNames);
    validatePortNames(this->outputNames, "output");
    const std::vector<std::string>& expectedExpressionInputs = layerDefinitionExpression.getExpectedInputNames();
    if (!expectedExpressionInputs.empty()) {
        const bool expressionUsesValidityMask =
            std::find(expectedExpressionInputs.begin(), expectedExpressionInputs.end(), std::string(Thor::BATCH_VALIDITY_MASK_NAME)) !=
            expectedExpressionInputs.end();
        if (expressionUsesValidityMask != batchValidityMaskEnabled) {
            throw runtime_error(batchValidityMaskEnabled
                                    ? "CustomLayer validity-mask support requires the expression to consume Thor::BATCH_VALIDITY_MASK_NAME."
                                    : "CustomLayer expression consumes Thor::BATCH_VALIDITY_MASK_NAME without enabling validity-mask support.");
        }
    }

    if (!this->declaredOutputDescriptors.empty() && this->declaredOutputDescriptors.size() != this->outputNames.size()) {
        throw runtime_error("CustomLayer declared output descriptor count must match the number of output ports.");
    }
    if (this->inputDimensionsIncludeBatch.empty()) {
        this->inputDimensionsIncludeBatch.assign(this->inputNames.size(), false);
    } else if (this->inputDimensionsIncludeBatch.size() != this->inputNames.size()) {
        throw runtime_error("CustomLayer batch-included input flag count must match the number of input ports.");
    }
    if (this->fixedBatchCapacity.has_value() && this->fixedBatchCapacity.value() == 0) {
        throw runtime_error("CustomLayer fixed batch capacity must be non-zero.");
    }
    for (uint32_t outputPort = 0; outputPort < this->declaredOutputDescriptors.size(); ++outputPort) {
        const auto& descriptor = this->declaredOutputDescriptors[outputPort];
        for (uint64_t dimension : descriptor.featureDimensions) {
            if (dimension == 0) {
                throw runtime_error("CustomLayer declared output dimensions must be non-zero for output port '" +
                                    this->outputNames[outputPort] + "'.");
            }
        }
    }

    for (uint32_t i = 0; i < this->inputNames.size(); ++i) {
        const auto [it, inserted] = inputNameToPort.emplace(this->inputNames[i], i);
        if (!inserted) {
            throw runtime_error("Duplicate CustomLayer input name: " + this->inputNames[i]);
        }
    }

    for (uint32_t i = 0; i < this->outputNames.size(); ++i) {
        const auto [it, inserted] = outputNameToPort.emplace(this->outputNames[i], i);
        if (!inserted) {
            throw runtime_error("Duplicate CustomLayer output name: " + this->outputNames[i]);
        }
    }

    for (const auto& param : parameters) {
        const string& paramName = param->getName();
        if (paramName.empty())
            throw runtime_error("CustomLayer parameter name cannot be empty.");

        if (paramName.length() >= 2 && paramName[0] == '_' && paramName[1] == '_') {
            throw runtime_error("CustomLayer parameter names cannot start with __ that is reserved. Parameter name " + paramName +
                                " is illegal.");
        }

        if (inputNameToPort.contains(paramName)) {
            throw runtime_error("CustomLayer parameter name collides with an input port name: " + paramName);
        }

        if (outputNameToPort.contains(paramName)) {
            throw runtime_error("CustomLayer parameter name collides with an output port name: " + paramName);
        }

        param->informExpressionBased();
        addParameter(param);  // verifies parameter name uniqueness
    }
}

void CustomLayer::validatePortNames(const std::vector<std::string>& names,
                                    const std::string& what,
                                    const std::set<std::string>& trustedReservedNames) {
    if (names.empty()) {
        throw runtime_error("CustomLayer requires at least one " + what + " port.");
    }

    for (const std::string& trustedName : trustedReservedNames) {
        if (trustedName.length() < 2 || trustedName[0] != '_' || trustedName[1] != '_') {
            throw runtime_error("CustomLayer trusted internal port names must use the reserved __ prefix: " + trustedName);
        }
    }

    std::set<std::string> seen;
    for (const std::string& name : names) {
        if (name.empty()) {
            throw runtime_error("CustomLayer " + what + " port name cannot be empty.");
        }
        if (name.length() >= 2 && name[0] == '_' && name[1] == '_' && !trustedReservedNames.contains(name)) {
            throw runtime_error("CustomLayer " + what + " port names cannot start with __ that is reserved. Name " + name + " is illegal.");
        }
        if (!seen.insert(name).second) {
            throw runtime_error("Duplicate CustomLayer " + what + " port name: " + name);
        }
    }

    for (const std::string& trustedName : trustedReservedNames) {
        if (!seen.contains(trustedName)) {
            throw runtime_error("CustomLayer trusted internal port name is not present in the declared " + what + " ports: " + trustedName);
        }
    }
}

uint32_t CustomLayer::inputFlatIndex(uint32_t applicationIndex, uint32_t inputPortIndex) const {
    THOR_THROW_IF_FALSE(inputPortIndex < inputNames.size());
    return applicationIndex * inputNames.size() + inputPortIndex;
}

uint32_t CustomLayer::outputFlatIndex(uint32_t applicationIndex, uint32_t outputPortIndex) const {
    THOR_THROW_IF_FALSE(outputPortIndex < outputNames.size());
    return applicationIndex * outputNames.size() + outputPortIndex;
}

CustomLayer::DecodedConnection CustomLayer::decodeInputConnectionType(int connectionType) const {
    if (connectionType < 0) {
        throw runtime_error("CustomLayer input connection type out of range.");
    }
    const uint32_t encoded = static_cast<uint32_t>(connectionType);
    return DecodedConnection{encoded / static_cast<uint32_t>(inputNames.size()), encoded % static_cast<uint32_t>(inputNames.size())};
}

CustomLayer::DecodedConnection CustomLayer::decodeOutputConnectionType(int connectionType) const {
    if (connectionType < 0) {
        throw runtime_error("CustomLayer output connection type out of range.");
    }
    const uint32_t encoded = static_cast<uint32_t>(connectionType);
    return DecodedConnection{encoded / static_cast<uint32_t>(outputNames.size()), encoded % static_cast<uint32_t>(outputNames.size())};
}

uint32_t CustomLayer::primaryInputFlatIndex(uint32_t applicationIndex) const {
    if (applicationIndex >= applications.size()) {
        throw runtime_error("CustomLayer application index out of range.");
    }

    for (uint32_t inputPort = 0; inputPort < inputNames.size(); ++inputPort) {
        const uint32_t flat = inputFlatIndex(applicationIndex, inputPort);
        if (flat < featureInputs.size() && featureInputs[flat].has_value()) {
            return flat;
        }
    }
    throw runtime_error("CustomLayer requires at least one connected input port before execution.");
}

Stream& CustomLayer::computeStream(uint32_t applicationIndex) {
    const uint32_t flat = primaryInputFlatIndex(applicationIndex);
    THOR_THROW_IF_FALSE(flat < streams.size());
    return streams[flat];
}

const Stream& CustomLayer::computeStream(uint32_t applicationIndex) const {
    const uint32_t flat = primaryInputFlatIndex(applicationIndex);
    THOR_THROW_IF_FALSE(flat < streams.size());
    return streams[flat];
}

Stream& CustomLayer::computeStream() { return computeStream(0); }

const Stream& CustomLayer::computeStream() const { return computeStream(0); }

void CustomLayer::ensureApplicationStorageAllocated(uint32_t applicationIndex) {
    if (applications.size() <= applicationIndex) {
        applications.resize(applicationIndex + 1);
    }
    if (applications[applicationIndex].forwardInputReadyEvents.size() < inputNames.size()) {
        applications[applicationIndex].forwardInputReadyEvents.resize(inputNames.size());
    }

    const size_t requiredInputs = static_cast<size_t>(applicationIndex + 1) * inputNames.size();
    const size_t requiredOutputs = static_cast<size_t>(applicationIndex + 1) * outputNames.size();

    if (featureInputs.size() < requiredInputs)
        featureInputs.resize(requiredInputs, std::nullopt);
    if (errorOutputs.size() < requiredInputs)
        errorOutputs.resize(requiredInputs, std::nullopt);
    if (previousLayers.size() < requiredInputs)
        previousLayers.resize(requiredInputs, std::nullopt);
    if (streams.size() < requiredInputs)
        streams.resize(requiredInputs);
    if (featureInputsConnectedForPorts.size() < requiredInputs)
        featureInputsConnectedForPorts.resize(requiredInputs, std::nullopt);
    if (errorOutputsConnectedForPorts.size() < requiredInputs)
        errorOutputsConnectedForPorts.resize(requiredInputs, std::nullopt);

    if (featureOutputs.size() < requiredOutputs)
        featureOutputs.resize(requiredOutputs, std::nullopt);
    if (errorInputs.size() < requiredOutputs)
        errorInputs.resize(requiredOutputs, std::nullopt);
    if (nextLayers.size() < requiredOutputs)
        nextLayers.resize(requiredOutputs, std::nullopt);
    if (featureOutputsConnectedForPorts.size() < requiredOutputs)
        featureOutputsConnectedForPorts.resize(requiredOutputs, std::nullopt);
    if (errorInputsConnectedForPorts.size() < requiredOutputs)
        errorInputsConnectedForPorts.resize(requiredOutputs, std::nullopt);
}

void CustomLayer::ensurePortStorageAllocated() { ensureApplicationStorageAllocated(0); }

bool CustomLayer::applicationHasAllInputPortsConnected(uint32_t applicationIndex) const {
    if (applicationIndex >= applications.size()) {
        return false;
    }

    for (uint32_t inputPort = 0; inputPort < inputNames.size(); ++inputPort) {
        const uint32_t flat = inputFlatIndex(applicationIndex, inputPort);
        if (flat >= featureInputs.size() || !featureInputs[flat].has_value()) {
            return false;
        }
    }

    return true;
}

void CustomLayer::requireApplicationInputInterfaceConnected(uint32_t applicationIndex) const {
    if (applicationHasAllInputPortsConnected(applicationIndex)) {
        return;
    }

    std::string missingPorts;
    if (applicationIndex >= applications.size()) {
        missingPorts = "<entire interface>";
    } else {
        for (uint32_t inputPort = 0; inputPort < inputNames.size(); ++inputPort) {
            const uint32_t flat = inputFlatIndex(applicationIndex, inputPort);
            if (flat >= featureInputs.size() || !featureInputs[flat].has_value()) {
                if (!missingPorts.empty()) {
                    missingPorts += ", ";
                }
                missingPorts += inputNames[inputPort];
            }
        }
    }

    throw runtime_error("CustomLayer cannot construct an output tensor for application " + std::to_string(applicationIndex) +
                        " until every input port in that interface is connected. Missing input port(s): " + missingPorts + ".");
}

void CustomLayer::clearForwardArrivalBookkeeping(uint32_t applicationIndex) {
    THOR_THROW_IF_FALSE(applicationIndex < applications.size());
    ApplicationState& app = applications[applicationIndex];
    app.allForwardInputTensorIds.clear();
    app.stillWaitingForForwardInputTensorIds.clear();
    app.forwardRanThisPass = false;
    app.forwardVariantThisPass.reset();
    app.currentValidExampleCount = 0;
    app.batchCardinalitySet = false;

    for (uint32_t inputPort = 0; inputPort < inputNames.size(); ++inputPort) {
        const uint32_t flat = inputFlatIndex(applicationIndex, inputPort);
        if (flat < featureInputs.size() && featureInputs[flat].has_value()) {
            app.allForwardInputTensorIds.insert(featureInputs[flat].value().getTensorId());
        }
    }
    app.stillWaitingForForwardInputTensorIds = app.allForwardInputTensorIds;
}

void CustomLayer::clearForwardArrivalBookkeeping() {
    for (uint32_t app = 0; app < applications.size(); ++app) {
        clearForwardArrivalBookkeeping(app);
    }
}

void CustomLayer::clearBackwardArrivalBookkeeping(uint32_t applicationIndex) {
    THOR_THROW_IF_FALSE(applicationIndex < applications.size());
    ApplicationState& app = applications[applicationIndex];
    app.allBackwardErrorInputTensorIds.clear();
    app.stillWaitingForBackwardErrorInputTensorIds.clear();
    app.backwardRanThisPass = false;

    if (app.backwardGradientPatternCompiled) {
        // Once compileImpl() has run, the expected incoming-gradient set is fixed for each
        // application. Runtime backward() should only wait for that compile-time pattern;
        // missing output ports are known to never send gradients for this application.
        app.allBackwardErrorInputTensorIds = app.expectedBackwardErrorInputTensorIds;
    } else {
        // During connection/compile setup, derive the same pattern from currently connected
        // downstream error inputs. compileImpl() snapshots it into expectedBackwardErrorInputTensorIds.
        for (uint32_t outputPort = 0; outputPort < outputNames.size(); ++outputPort) {
            const uint32_t flat = outputFlatIndex(applicationIndex, outputPort);
            if (flat < errorInputs.size() && errorInputs[flat].has_value()) {
                app.allBackwardErrorInputTensorIds.insert(errorInputs[flat].value().getTensorId());
            }
        }
    }

    app.stillWaitingForBackwardErrorInputTensorIds = app.allBackwardErrorInputTensorIds;
}

void CustomLayer::clearBackwardArrivalBookkeeping() {
    for (uint32_t app = 0; app < applications.size(); ++app) {
        clearBackwardArrivalBookkeeping(app);
    }
    numBackwardApplicationsCompletedThisPass = 0;
    effectiveBatchSizeByParameterName.clear();
}

bool CustomLayer::applicationHasAnyDownstreamBackprop(uint32_t applicationIndex) const {
    if (applicationIndex >= applications.size()) {
        return false;
    }
    for (uint32_t outputPort = 0; outputPort < outputNames.size(); ++outputPort) {
        const uint32_t flat = outputFlatIndex(applicationIndex, outputPort);
        if (flat < errorInputs.size() && errorInputs[flat].has_value()) {
            return true;
        }
    }
    return false;
}

CustomLayer::StampedExecutionVariant& CustomLayer::stampedVariant(uint32_t applicationIndex,
                                                                  DynamicExpressionVariantId variantId) {
    if (applicationIndex >= applications.size()) {
        throw runtime_error("CustomLayer execution variant requested for an invalid application index.");
    }
    auto it = applications[applicationIndex].stampedVariants.find(variantId);
    if (it == applications[applicationIndex].stampedVariants.end()) {
        throw runtime_error("CustomLayer application " + std::to_string(applicationIndex) +
                            " has no stamped execution variant " + std::to_string(variantId) + ".");
    }
    return it->second;
}

const CustomLayer::StampedExecutionVariant& CustomLayer::stampedVariant(uint32_t applicationIndex,
                                                                        DynamicExpressionVariantId variantId) const {
    if (applicationIndex >= applications.size()) {
        throw runtime_error("CustomLayer execution variant requested for an invalid application index.");
    }
    auto it = applications[applicationIndex].stampedVariants.find(variantId);
    if (it == applications[applicationIndex].stampedVariants.end()) {
        throw runtime_error("CustomLayer application " + std::to_string(applicationIndex) +
                            " has no stamped execution variant " + std::to_string(variantId) + ".");
    }
    return it->second;
}

CustomLayer::StampedExecutionVariant& CustomLayer::backwardVariantForApplication(uint32_t applicationIndex) {
    ApplicationState& app = applications.at(applicationIndex);
    if (!app.forwardVariantThisPass.has_value()) {
        throw runtime_error("CustomLayer backward requires a forward execution variant for the current pass.");
    }
    StampedExecutionVariant& variant = stampedVariant(applicationIndex, app.forwardVariantThisPass.value());
    if (!variant.supportsBackward) {
        throw runtime_error("CustomLayer execution variant " + std::to_string(app.forwardVariantThisPass.value()) +
                            " does not support backward execution.");
    }
    return variant;
}

const CustomLayer::StampedExecutionVariant& CustomLayer::backwardVariantForApplication(uint32_t applicationIndex) const {
    const ApplicationState& app = applications.at(applicationIndex);
    if (!app.forwardVariantThisPass.has_value()) {
        throw runtime_error("CustomLayer backward requires a forward execution variant for the current pass.");
    }
    const StampedExecutionVariant& variant = stampedVariant(applicationIndex, app.forwardVariantThisPass.value());
    if (!variant.supportsBackward) {
        throw runtime_error("CustomLayer execution variant " + std::to_string(app.forwardVariantThisPass.value()) +
                            " does not support backward execution.");
    }
    return variant;
}

void CustomLayer::setActiveTrainingExecutionVariant(DynamicExpressionVariantId variantId) {
    if (isCompiled() && !isStartOfForward) {
        // Forward-only validation/inference clears each application's recorded
        // variant without running backward. Permit a stage-boundary switch in
        // that drained state, but reject any training forward that still owns a
        // matching backward pass.
        const bool trainingForwardAwaitingBackward =
            std::any_of(applications.begin(), applications.end(), [](const ApplicationState& app) {
                return app.forwardVariantThisPass.has_value();
            });
        if (trainingForwardAwaitingBackward) {
            throw runtime_error("CustomLayer training execution variant may only change between drained execution passes.");
        }
    }

    if (isCompiled()) {
        for (uint32_t applicationIndex = 0; applicationIndex < applications.size(); ++applicationIndex) {
            const StampedExecutionVariant& variant = stampedVariant(applicationIndex, variantId);
            if (!variant.supportsBackward) {
                throw runtime_error("CustomLayer execution variant " + std::to_string(variantId) +
                                    " does not support training backward execution.");
            }
        }
    }

    activeTrainingVariantId = variantId;
}

void CustomLayer::recordEffectiveParameterBatchSizeForApplication(uint32_t applicationIndex, uint32_t batchSize) {
    if (applicationIndex >= applications.size()) {
        return;
    }

    const StampedExecutionVariant& variant = backwardVariantForApplication(applicationIndex);
    if (trainingUpdateDiagnosticsEnabled()) {
        std::fprintf(stderr,
                     "THOR_TRAINING_UPDATE_DIAGNOSTIC layer=%s app=%u record_effective_batch batch=%u active_parameters=%s fused_parameters=%s before=%s\n",
                     diagnosticLabel().c_str(),
                     applicationIndex,
                     batchSize,
                     joinNames(variant.activeParameterTargetNames).c_str(),
                     joinNames(variant.optimizerUpdateFusedParameterNames).c_str(),
                     joinNames(effectiveBatchSizeByParameterName).c_str());
    }
    for (const std::string& parameterName : variant.activeParameterTargetNames) {
        if (variant.optimizerUpdateFusedParameterNames.contains(parameterName)) {
            continue;
        }
        effectiveBatchSizeByParameterName[parameterName] += batchSize;
    }
    if (trainingUpdateDiagnosticsEnabled()) {
        std::fprintf(stderr,
                     "THOR_TRAINING_UPDATE_DIAGNOSTIC layer=%s app=%u record_effective_batch after=%s\n",
                     diagnosticLabel().c_str(),
                     applicationIndex,
                     joinNames(effectiveBatchSizeByParameterName).c_str());
    }
}

bool CustomLayer::applicationHasConditionalBackwardVariant(uint32_t applicationIndex) const {
    if (applicationIndex >= applications.size()) {
        return false;
    }
    const ApplicationState& app = applications[applicationIndex];
    if (app.forwardPrepared == nullptr) {
        return false;
    }

    for (DynamicExpressionVariantId variantId : app.forwardPrepared->executionVariantIds()) {
        if (!app.forwardPrepared->executionVariantSupportsBackward(variantId)) {
            continue;
        }
        if (app.forwardPrepared->equationForVariant(variantId).physicalOutputs().isConditional()) {
            return true;
        }
    }
    return false;
}

bool CustomLayer::canFuseOptimizerUpdatesForApplication(uint32_t applicationIndex) const {
    // Dense optimizer-update fusion is only correct for a single application with one
    // equation-bound feature input. Multi-application layers and expressions with multiple
    // mathematical feature inputs share materialized gradient buffers across applications/ports
    // and need the explicit overwrite-then-accumulate path below so effective batch-size
    // accounting and shared parameter accumulation stay well-defined.
    //
    // A declared input that is present only in preForwardOnlyInputs() is different: it is a
    // structural/runtime dependency, not an input to the differentiable equation. Ragged
    // specializations use such ports for row-partition offsets. Those ports must not disable
    // optimizer fusion merely because they make the physical layer interface multi-port.
    if (applicationIndex != 0 || applications.size() != 1 || applicationIndex >= applications.size()) {
        return false;
    }

    const ApplicationState& app = applications[applicationIndex];
    if (app.forwardPrepared == nullptr) {
        return false;
    }

    uint32_t equationBoundFeatureInputs = 0;
    for (const std::string& inputName : inputNames) {
        if (app.forwardPrepared->preForwardOnlyInputs().contains(inputName)) {
            continue;
        }
        if (!app.forwardPrepared->stampInputs().contains(inputName)) {
            // validatePreparedExpressionInputs() should already make this impossible, but
            // optimizer-fusion eligibility is deliberately conservative.
            return false;
        }
        ++equationBoundFeatureInputs;
    }
    if (equationBoundFeatureInputs != 1) {
        return false;
    }

    // The dense optimizer fusion surface consumes one flat gradient Expression. A graph-level
    // conditional backward is intentionally represented as a conditional PhysicalOutputs tree,
    // so keep the parameter gradient materialized and use the ordinary optimizer update path.
    // This is a correctness-preserving fallback; the conditional VJP itself remains fully fused
    // within each selected branch.
    return !applicationHasConditionalBackwardVariant(applicationIndex);
}

bool CustomLayer::applicationHasFusedCustomLossGradient(uint32_t applicationIndex) const {
    return applicationIndex < applications.size() && !applications[applicationIndex].fusedCustomLossGradientsByOutput.empty();
}

uint32_t CustomLayer::getNumFusedCustomLossGradients() const {
    return static_cast<uint32_t>(fusedCustomLossGradientByOutputFlatIndex.size());
}

#ifdef THOR_DEBUG
std::optional<CustomLayer::GenericSharedBackwardDebugDiagnostic>
CustomLayer::genericSharedBackwardDebugDiagnostic(uint32_t applicationIndex) const {
    if (applicationIndex >= applications.size()) {
        return std::nullopt;
    }
    const ApplicationState& app = applications[applicationIndex];
    if (!app.forwardVariantThisPass.has_value()) {
        return std::nullopt;
    }
    const auto variantIt = app.stampedVariants.find(app.forwardVariantThisPass.value());
    if (variantIt == app.stampedVariants.end()) {
        return std::nullopt;
    }
    const StampedExecutionVariant& variant = variantIt->second;
    if (variant.genericSharedBackwardClear == nullptr) {
        return std::nullopt;
    }

    return GenericSharedBackwardDebugDiagnostic{
        .clearStageKindNames = variant.genericSharedBackwardClear->stageKindNames(),
        .clearStageDependencyIndices = variant.genericSharedBackwardClear->stageDependencyIndices(),
        .clearExecutionCount = variant.genericSharedBackwardClearExecutionCount,
        .accumulateExecutionCount = variant.genericSharedBackwardAccumulateExecutionCount,
    };
}
#endif

bool CustomLayer::registerFusedCustomLossGradient(const Tensor& predictions,
                                                 const Tensor& labels,
                                                 DynamicExpression gradientExpression,
                                                 std::string predictionsName,
                                                 std::string labelsName,
                                                 std::string gradientName,
                                                 const Tensor& batchValidityMask,
                                                 std::string batchValidityMaskName,
                                                 Loss* ownerLoss) {
    if (isInferenceOnly()) {
        return false;
    }

    ensurePortStorageAllocated();

    std::optional<uint32_t> matchedFlatIndex;
    for (uint32_t flat = 0; flat < featureOutputs.size(); ++flat) {
        if (featureOutputs[flat].has_value() && featureOutputs[flat].value() == predictions) {
            if (matchedFlatIndex.has_value()) {
                return false;
            }
            matchedFlatIndex = flat;
        }
    }

    if (!matchedFlatIndex.has_value()) {
        return false;
    }
    if (fusedCustomLossGradientByOutputFlatIndex.contains(matchedFlatIndex.value())) {
        return false;
    }

    THOR_THROW_IF_FALSE(batchValidityMask.isInitialized());
    THOR_THROW_IF_FALSE(batchValidityMask.getDataType() == DataType::FP32);
    THOR_THROW_IF_FALSE(batchValidityMask.getPlacement() == predictions.getPlacement());
    const std::vector<uint64_t> maskDimensions = batchValidityMask.getDimensions();
    const std::vector<uint64_t> predictionDimensions = predictions.getDimensions();
    THOR_THROW_IF_FALSE(maskDimensions.size() == predictionDimensions.size());
    THOR_THROW_IF_FALSE(!predictionDimensions.empty());
    THOR_THROW_IF_FALSE(maskDimensions.front() == predictionDimensions.front());
    for (size_t axis = 1; axis < maskDimensions.size(); ++axis)
        THOR_THROW_IF_FALSE(maskDimensions[axis] == 1 || maskDimensions[axis] == predictionDimensions[axis]);

    FusedCustomLossGradient fused{predictions,
                                  labels,
                                  std::move(gradientExpression),
                                  std::move(predictionsName),
                                  std::move(labelsName),
                                  std::move(gradientName),
                                  batchValidityMask,
                                  std::move(batchValidityMaskName),
                                  customLossFusedLabelsInputName(matchedFlatIndex.value()),
                                  customLossFusedBatchValidityMaskInputName(matchedFlatIndex.value()),
                                  ownerLoss};
    fusedCustomLossGradientByOutputFlatIndex.emplace(matchedFlatIndex.value(), std::move(fused));
    return true;
}

bool CustomLayer::unregisterFusedCustomLossGradient(const Tensor& predictions) {
    if (isInferenceOnly() || isCompiled()) {
        return false;
    }

    ensurePortStorageAllocated();

    std::optional<uint32_t> matchedFlatIndex;
    for (uint32_t flat = 0; flat < featureOutputs.size(); ++flat) {
        if (featureOutputs[flat].has_value() && featureOutputs[flat].value() == predictions) {
            if (matchedFlatIndex.has_value()) {
                return false;
            }
            matchedFlatIndex = flat;
        }
    }

    if (!matchedFlatIndex.has_value()) {
        return false;
    }
    return fusedCustomLossGradientByOutputFlatIndex.erase(matchedFlatIndex.value()) != 0;
}

BackwardBuildResult CustomLayer::buildBackwardOutputsForApplication(
    uint32_t applicationIndex,
    DynamicExpressionVariantId variantId,
    const std::vector<std::string>& wrtNames,
    bool accumulateGradOutputs) {
    GradientAccumulationTargets accumulateWrtNames;
    if (accumulateGradOutputs) {
        accumulateWrtNames.insert(wrtNames.begin(), wrtNames.end());
    }
    return buildBackwardOutputsForApplication(applicationIndex, variantId, wrtNames, accumulateWrtNames);
}

BackwardBuildResult CustomLayer::buildBackwardOutputsForApplication(
    uint32_t applicationIndex,
    DynamicExpressionVariantId variantId,
    const std::vector<std::string>& wrtNames,
    const GradientAccumulationTargets& accumulateWrtNames) {
    ApplicationState& app = applications[applicationIndex];

    PreparedDynamicExpression::ShapeMap forwardInputDims;
    for (const auto& [name, tensor] : app.forwardPrepared->stampInputs()) {
        forwardInputDims[name] = tensor.getDimensions();
    }

    std::unordered_map<std::string, DataType> upstreamInputDTypesByOutput;
    upstreamInputDTypesByOutput.reserve(app.upstreamInputNamesByOutput.size() + app.fusedCustomLossGradientsByOutput.size());
    for (const auto& [outputName, upstreamInputName] : app.upstreamInputNamesByOutput) {
        (void)upstreamInputName;
        auto outputPortIt = outputNameToPort.find(outputName);
        if (outputPortIt == outputNameToPort.end()) {
            throw runtime_error("CustomLayer backward dtype discovery found unknown output name: " + outputName);
        }
        const uint32_t flat = outputFlatIndex(applicationIndex, outputPortIt->second);
        if (flat >= errorInputs.size() || !errorInputs[flat].has_value()) {
            throw runtime_error("CustomLayer backward dtype discovery expected a connected incoming gradient for output: " + outputName);
        }
        upstreamInputDTypesByOutput.emplace(outputName, errorInputs[flat].value().getDataType());
    }
    for (const auto& [outputName, fused] : app.fusedCustomLossGradientsByOutput) {
        // CustomLoss validates that its gradient tensor has the same descriptor as
        // predictions, so the prediction dtype is the exact synthetic seed dtype.
        upstreamInputDTypesByOutput.emplace(outputName, fused.predictionsTensor.getDataType());
    }

    // Every CustomLayer contribution differentiates the same retention-aware
    // physical forward graph that the real training forward stamps. There is no
    // alternate replay graph and no per-downstream-branch AutoDiff invocation.
    // An eligible fused GELU is therefore represented as a MATMUL/GEMM carrying
    // a forward epilogue-aux provider, and every computed primal dependency is
    // reported through ForwardValueRequirement.
    PhysicalOutputs forwardOutputs =
        app.forwardPrepared->equationForVariant(variantId).physicalOutputsForTrainingBackward(
            app.forwardPrepared->stampInputs(), app.forwardPrepared->tensorScalarInputsForVariant(variantId));

    // physicalOutputsForTrainingBackward() resolves and optimizes the training
    // preview inside FusedEquation. Re-apply the CustomLayer graph-boundary
    // contract: logical promotion is a forward/compute choice and must not widen
    // the storage dtype of the gradient handed back to the physical producer.
    applyCustomLayerInputGradientStorageContractInPlace(forwardOutputs);

    if (app.fusedCustomLossGradientsByOutput.empty()) {
        return buildBackwardOutputsWithForwardValueRequirements(forwardOutputs,
                                                                wrtNames,
                                                                app.upstreamInputNamesByOutput,
                                                                upstreamInputDTypesByOutput,
                                                                forwardInputDims,
                                                                accumulateWrtNames);
    }

    if (!forwardOutputs.expr) {
        throw runtime_error("CustomLayer fused CustomLoss backward requires non-null forward expression.");
    }

    // Keep AutoDiff focused on the driving layer only.  We first build the normal
    // layer-backward graph with a synthetic upstream-gradient input for each fused
    // loss output, then inline the CustomLoss gradient expression into the resulting
    // backward expression by replacing that synthetic seed input.  This makes the
    // loss gradient an adjoint seed without adding the loss-gradient graph to the
    // primal forward graph that AutoDiff differentiates through.
    std::unordered_map<std::string, std::string> upstreamInputNamesByOutput = app.upstreamInputNamesByOutput;
    std::unordered_map<std::string, std::string> fusedSeedInputNameByOutput;
    for (const auto& [outputName, fused] : app.fusedCustomLossGradientsByOutput) {
        (void)fused;
        const std::string seedName = customLossFusedSeedInputName(outputName);
        upstreamInputNamesByOutput.emplace(outputName, seedName);
        fusedSeedInputNameByOutput.emplace(outputName, seedName);
    }

    BackwardBuildResult seededBackwardBuild = buildBackwardOutputsWithForwardValueRequirements(
        forwardOutputs,
        wrtNames,
        upstreamInputNamesByOutput,
        upstreamInputDTypesByOutput,
        forwardInputDims,
        accumulateWrtNames);
    PhysicalOutputs& seededBackwardOutputs = seededBackwardBuild.outputs;
    if (!seededBackwardOutputs.expr) {
        throw runtime_error("CustomLayer fused CustomLoss backward produced a null backward expression.");
    }

    PhysicalOutputs fusedBackwardOutputs;
    fusedBackwardOutputs.expr = std::make_shared<PhysicalExpression>();
    PhysicalExpression& fusedExpr = *fusedBackwardOutputs.expr;

    std::unordered_map<std::string, uint32_t> fusedExprInputNodeByName;
    auto ensureInputNode = [&](const NamedInput& input) -> uint32_t {
        auto existing = fusedExprInputNodeByName.find(input.name);
        if (existing != fusedExprInputNodeByName.end()) {
            return existing->second;
        }
        uint32_t node = appendInputNode(fusedExpr, input);
        fusedExprInputNodeByName.emplace(input.name, node);
        return node;
    };

    auto isFusedSeedInputName = [&](const std::string& name) -> bool {
        for (const auto& [_, seedName] : fusedSeedInputNameByOutput) {
            if (seedName == name) {
                return true;
            }
        }
        return false;
    };

    // Recreate every real backward input in the final expression.  Synthetic seed
    // inputs are intentionally omitted because they are replaced by the inlined
    // CustomLoss gradient expression below.
    for (const NamedInput& input : seededBackwardOutputs.expr->inputs) {
        if (!isFusedSeedInputName(input.name)) {
            ensureInputNode(input);
        }
    }
    // The inlined loss gradient may need forward inputs that the layer-gradient
    // expression itself did not otherwise need for the requested wrt set.
    for (const NamedInput& input : forwardOutputs.expr->inputs) {
        ensureInputNode(input);
    }

    std::unordered_map<std::string, uint32_t> seedReplacementNodeByName;
    for (const auto& [outputName, fused] : app.fusedCustomLossGradientsByOutput) {
        auto seedNameIt = fusedSeedInputNameByOutput.find(outputName);
        if (seedNameIt == fusedSeedInputNameByOutput.end()) {
            throw runtime_error("CustomLayer fused CustomLoss gradient lost synthetic seed for output: " + outputName);
        }

        std::optional<uint32_t> forwardOutputNode;
        for (const NamedOutput& output : forwardOutputs.outputs) {
            if (output.name == outputName) {
                forwardOutputNode = output.node_idx;
                break;
            }
        }
        if (!forwardOutputNode.has_value()) {
            throw runtime_error("CustomLayer fused CustomLoss gradient references unknown output: " + outputName);
        }

        PreparedDynamicExpression::TensorMap gradientInputs;
        gradientInputs.emplace(fused.predictionsName, fused.predictionsTensor);
        gradientInputs.emplace(fused.labelsName, fused.labelsTensor);
        gradientInputs.emplace(fused.batchValidityMaskName, fused.batchValidityMask);

        DynamicExpressionBuild gradientBuild = fused.gradientExpression.build(gradientInputs, {}, computeStream(applicationIndex));
        if (!gradientBuild.tensor_scalar_inputs.empty()) {
            throw runtime_error("CustomLayer fused CustomLoss gradients do not support tensor-scalar runtime inputs.");
        }
        if (!gradientBuild.preallocated_outputs.empty()) {
            throw runtime_error("CustomLayer fused CustomLoss gradients must not require preallocated outputs.");
        }

        const PhysicalOutputs gradientOutputs = gradientBuild.equation->physicalOutputs();
        if (!gradientOutputs.expr) {
            throw runtime_error("CustomLayer fused CustomLoss gradient expression produced a null physical expression.");
        }

        std::optional<uint32_t> gradientOutputNode;
        for (const NamedOutput& gradientOutput : gradientOutputs.outputs) {
            if (gradientOutput.name == fused.gradientName) {
                gradientOutputNode = gradientOutput.node_idx;
                break;
            }
        }
        if (!gradientOutputNode.has_value()) {
            throw runtime_error("CustomLayer fused CustomLoss gradient expression did not produce expected output: " + fused.gradientName);
        }

        // The fused loss seed consumes the actual prediction produced by the real
        // forward.  It must never clone the prediction subtree into backward.
        uint32_t predictionNode = UINT32_MAX;
        std::string savedPredictionInputName;
        auto existingRequirementIt = std::find_if(
            seededBackwardBuild.forward_value_requirements.begin(),
            seededBackwardBuild.forward_value_requirements.end(),
            [&](const ForwardValueRequirement& requirement) {
                // A fused CustomLoss needs the actual prediction value. BR4 may
                // already have declared a MatmulEpilogueAux requirement at this
                // same logical output node for GELU backward; that tensor is the
                // preactivation and must never be substituted for the post-GELU
                // prediction.
                return requirement.forward_node_index == forwardOutputNode.value() &&
                       requirement.kind == ForwardValueRequirementKind::NodeOutput;
            });
        if (existingRequirementIt != seededBackwardBuild.forward_value_requirements.end()) {
            savedPredictionInputName = existingRequirementIt->backward_input_name;
        } else {
            const std::string baseName = "__thor_saved_forward_value_" + std::to_string(forwardOutputNode.value());
            savedPredictionInputName = baseName;
            for (uint32_t suffix = 1; fusedExprInputNodeByName.contains(savedPredictionInputName); ++suffix) {
                savedPredictionInputName = baseName + "_" + std::to_string(suffix);
            }
            seededBackwardBuild.forward_value_requirements.push_back(ForwardValueRequirement{
                .forward_node_index = forwardOutputNode.value(),
                .backward_input_name = savedPredictionInputName,
                .kind = ForwardValueRequirementKind::NodeOutput,
            });
        }

        auto inputNodeIt = fusedExprInputNodeByName.find(savedPredictionInputName);
        if (inputNodeIt != fusedExprInputNodeByName.end()) {
            predictionNode = inputNodeIt->second;
        } else {
            NamedInput savedPredictionInput{
                savedPredictionInputName,
                static_cast<uint32_t>(fusedExpr.inputs.size()),
                NamedInput::Kind::Tensor,
            };
            predictionNode = ensureInputNode(savedPredictionInput);
        }

        const ExprNode& sourcePrediction = forwardOutputs.expr->nodes.at(forwardOutputNode.value());
        const std::optional<DataType> predictionDType = sourcePrediction.output_dtype.has_value()
                                                            ? sourcePrediction.output_dtype
                                                            : sourcePrediction.input_tensor_dtype;
        fusedExpr.nodes.at(predictionNode).input_tensor_dtype = predictionDType;
        fusedExpr.nodes.at(predictionNode).output_dtype = predictionDType;

        const uint32_t labelsInputNode = appendTensorInputNode(fusedExpr, fused.fusedLabelsInputName);
        const uint32_t batchValidityMaskInputNode =
            appendTensorInputNode(fusedExpr, fused.fusedBatchValidityMaskInputName);

        const uint32_t gradientCudaKernelExpressionOffset = static_cast<uint32_t>(fusedExpr.cuda_kernel_expressions.size());
        fusedExpr.cuda_kernel_expressions.insert(fusedExpr.cuda_kernel_expressions.end(),
                                                gradientOutputs.expr->cuda_kernel_expressions.begin(),
                                                gradientOutputs.expr->cuda_kernel_expressions.end());

        std::unordered_map<std::string, uint32_t> gradientInputReplacements{
            {fused.predictionsName, predictionNode},
            {fused.labelsName, labelsInputNode},
            {fused.batchValidityMaskName, batchValidityMaskInputNode},
        };
        std::unordered_map<uint32_t, uint32_t> clonedGradientNodes;
        const uint32_t fusedSeedNode = cloneExpressionNodeWithInputReplacements(*gradientOutputs.expr,
                                                                                gradientOutputNode.value(),
                                                                                fusedExpr,
                                                                                gradientInputReplacements,
                                                                                inputBySlot(*gradientOutputs.expr),
                                                                                gradientCudaKernelExpressionOffset,
                                                                                clonedGradientNodes);
        seedReplacementNodeByName.emplace(seedNameIt->second, fusedSeedNode);
    }

    const uint32_t backwardCudaKernelExpressionOffset = static_cast<uint32_t>(fusedExpr.cuda_kernel_expressions.size());
    fusedExpr.cuda_kernel_expressions.insert(fusedExpr.cuda_kernel_expressions.end(),
                                            seededBackwardOutputs.expr->cuda_kernel_expressions.begin(),
                                            seededBackwardOutputs.expr->cuda_kernel_expressions.end());

    std::unordered_map<std::string, uint32_t> backwardInputReplacements;
    for (const NamedInput& input : seededBackwardOutputs.expr->inputs) {
        auto seedIt = seedReplacementNodeByName.find(input.name);
        if (seedIt != seedReplacementNodeByName.end()) {
            backwardInputReplacements.emplace(input.name, seedIt->second);
            continue;
        }

        auto inputNodeIt = fusedExprInputNodeByName.find(input.name);
        if (inputNodeIt == fusedExprInputNodeByName.end()) {
            throw runtime_error("CustomLayer fused CustomLoss backward missing real input clone for: " + input.name);
        }
        backwardInputReplacements.emplace(input.name, inputNodeIt->second);
    }

    std::unordered_map<std::string, std::string> inputNameByGradOutputName;
    for (const std::string& inputName : inputNames) {
        inputNameByGradOutputName.emplace(inputName + "_grad", inputName);
    }

    uint32_t numInputGradientOutputs = 0;
    for (const NamedOutput& output : seededBackwardOutputs.outputs) {
        if (inputNameByGradOutputName.contains(output.name)) {
            ++numInputGradientOutputs;
        }
    }
    const bool disambiguateInputGradientOutputs = numInputGradientOutputs > 1;

    auto makeInputSpecificTerminalGradientNode = [&](const std::string& outputName, uint32_t gradNode) -> uint32_t {
        auto inputNameIt = inputNameByGradOutputName.find(outputName);
        if (inputNameIt == inputNameByGradOutputName.end() || !disambiguateInputGradientOutputs) {
            return gradNode;
        }

        auto inputNodeIt = fusedExprInputNodeByName.find(inputNameIt->second);
        if (inputNodeIt == fusedExprInputNodeByName.end()) {
            throw runtime_error("CustomLayer fused CustomLoss backward missing input node for terminal gradient: " + inputNameIt->second);
        }

        // The expression compiler is allowed to coalesce equivalent final outputs onto one physical output tensor.
        // That is normally valid, but graph-level input-error outputs are already connected to distinct upstream
        // ports before this backward equation is stamped.  When one fused-loss backward stamp writes multiple input
        // gradients, make each terminal input gradient structurally depend on its corresponding input through an
        // input-specific zero term, so equivalent derivatives such as d((x + y) * scale)/dx and
        // d((x + y) * scale)/dy still materialize into their own preconnected tensors.
        ExprNode zeroNode{};
        zeroNode.op = ExprOp::SUB;
        zeroNode.lhs = inputNodeIt->second;
        zeroNode.rhs = inputNodeIt->second;
        const uint32_t zeroNodeIndex = static_cast<uint32_t>(fusedExpr.nodes.size());
        fusedExpr.nodes.push_back(std::move(zeroNode));

        ExprNode terminalNode{};
        terminalNode.op = ExprOp::ADD;
        terminalNode.lhs = gradNode;
        terminalNode.rhs = zeroNodeIndex;

        const auto logicalInputIt = app.forwardPrepared->stampInputs().find(inputNameIt->second);
        if (logicalInputIt != app.forwardPrepared->stampInputs().end()) {
            const DataType dtype = logicalInputIt->second.getDescriptor().getDataType();
            terminalNode.output_dtype = dtype;
            terminalNode.backward_output_dtype = dtype;
        }

        const uint32_t terminalNodeIndex = static_cast<uint32_t>(fusedExpr.nodes.size());
        fusedExpr.nodes.push_back(std::move(terminalNode));
        return terminalNodeIndex;
    };

    std::unordered_map<uint32_t, uint32_t> clonedBackwardNodes;
    fusedBackwardOutputs.outputs.reserve(seededBackwardOutputs.outputs.size());
    for (const NamedOutput& output : seededBackwardOutputs.outputs) {
        const uint32_t clonedOutputNode = cloneExpressionNodeWithInputReplacements(*seededBackwardOutputs.expr,
                                                                                   output.node_idx,
                                                                                   fusedExpr,
                                                                                   backwardInputReplacements,
                                                                                   inputBySlot(*seededBackwardOutputs.expr),
                                                                                   backwardCudaKernelExpressionOffset,
                                                                                   clonedBackwardNodes);
        const uint32_t terminalOutputNode = makeInputSpecificTerminalGradientNode(output.name, clonedOutputNode);
        fusedBackwardOutputs.outputs.push_back(NamedOutput{
            .name = output.name,
            .node_idx = terminalOutputNode,
            .materialization = output.materialization,
        });
    }

    std::sort(seededBackwardBuild.forward_value_requirements.begin(),
              seededBackwardBuild.forward_value_requirements.end(),
              [](const ForwardValueRequirement& lhs, const ForwardValueRequirement& rhs) {
                  if (lhs.conditional_branch_path != rhs.conditional_branch_path) {
                      return lhs.conditional_branch_path < rhs.conditional_branch_path;
                  }
                  if (lhs.forward_node_index != rhs.forward_node_index) {
                      return lhs.forward_node_index < rhs.forward_node_index;
                  }
                  if (lhs.kind != rhs.kind) {
                      return static_cast<uint8_t>(lhs.kind) < static_cast<uint8_t>(rhs.kind);
                  }
                  return lhs.backward_input_name < rhs.backward_input_name;
              });
    seededBackwardBuild.forward_value_requirements.erase(
        std::unique(seededBackwardBuild.forward_value_requirements.begin(),
                    seededBackwardBuild.forward_value_requirements.end(),
                    [](const ForwardValueRequirement& lhs, const ForwardValueRequirement& rhs) {
                        return lhs.conditional_branch_path == rhs.conditional_branch_path &&
                               lhs.forward_node_index == rhs.forward_node_index &&
                               lhs.kind == rhs.kind &&
                               lhs.backward_input_name == rhs.backward_input_name;
                    }),
        seededBackwardBuild.forward_value_requirements.end());

    seededBackwardBuild.outputs = std::move(fusedBackwardOutputs);
    return seededBackwardBuild;
}

std::shared_ptr<StampedExecutionPlan> CustomLayer::stampBackwardForApplication(
    uint32_t applicationIndex,
    DynamicExpressionVariantId variantId,
    const BackwardBuildResult& backwardBuild,
    bool accumulateGradOutputs,
    const PreparedDynamicExpression::TensorMap& preallocatedGradOutputs,
    Stream& runStream) {
    ApplicationState& app = applications[applicationIndex];
    if (!backwardBuild.outputs.expr || backwardBuild.outputs.outputs.empty()) {
        return nullptr;
    }

    FusedEquation backwardEquation = FusedEquation::compile(backwardBuild.outputs, placement.getDeviceNum());

    // Differentiate the prepared forward equation, not merely the layer's externally connected inputs.
    // DynamicExpression builders may add internal stamp-time tensors (for example ragged Attention
    // synthetic row partitions and RoPE position-id buffers) that are real inputs to the physical
    // expression and therefore remain dependencies of its backward graph.
    PreparedDynamicExpression::TensorMap stampInputs = app.forwardPrepared->stampInputs();
    for (const auto& [name, tensor] : app.backwardAdditionalInputsByName) {
        stampInputs[name] = tensor;
    }
    StampedExecutionVariant& variant = stampedVariant(applicationIndex, variantId);
    if (!backwardBuild.forward_value_requirements.empty()) {
        if (variant.forward == nullptr) {
            throw runtime_error("CustomLayer saved-forward backward requires the real forward execution plan.");
        }
        bindRetainedForwardValues(backwardBuild, *variant.forward, stampInputs);
    }
    if (accumulateGradOutputs) {
        // CustomLayer builds the backward graph directly so layer-specific shape-specialized
        // autodiff rules (for example RMSNorm) can see the concrete forward input dimensions.
        // AutoDiff still represents accumulation as `wrt_grad = wrt_grad + newly_computed_grad`,
        // which means the existing gradient buffer is a real tensor input as well as the
        // preallocated output. Bind those tensors explicitly.
        for (const auto& [name, tensor] : preallocatedGradOutputs) {
            stampInputs[name] = tensor;
        }
    }

    // Filter to the exact backward ABI before stamping. Standalone backward
    // equations preserve the original forward-root interface, while composed
    // optimizer/loss expressions may still eliminate inputs during expression
    // merging. Retained-forward tensors are added above and survive this filter
    // only when the differentiated graph actually consumes them.
    PreparedDynamicExpression::TensorMap filteredStampInputs =
        filterTensorInputsForPhysicalOutputs(stampInputs, backwardBuild.outputs);
    PreparedDynamicExpression::TensorScalarMap filteredTensorScalarInputs =
        filterTensorScalarInputsForPhysicalOutputs(
            app.forwardPrepared->tensorScalarInputsForVariant(variantId), backwardBuild.outputs);

    // This plan is linked immediately to the already-stamped real forward. For
    // graph-level conditionals, defer child CUDA-graph capture until those links
    // are installed so no unlinked RMSNorm/Attention backward operation is ever
    // captured or executed.
    auto backwardPlan = std::make_shared<StampedExecutionPlan>(
        backwardEquation.stampForImmediateCrossPlanBackwardLinking(filteredStampInputs,
                                                                   runStream,
                                                                   filteredTensorScalarInputs,
                                                                   preallocatedGradOutputs));
    if (variant.forward != nullptr) {
        backwardPlan->linkRmsNormBackwardStatesFrom(*variant.forward);
        backwardPlan->linkAttentionBackwardStatesFrom(*variant.forward);
        variant.forward->rebuildConditionalGraphsAfterCrossPlanLinking();
        backwardPlan->rebuildConditionalGraphsAfterCrossPlanLinking();
    }
    return backwardPlan;
}

std::shared_ptr<StampedExecutionPlan> CustomLayer::buildGenericSharedBackwardWithFusedOptimizerPlan(
    uint32_t applicationIndex,
    DynamicExpressionVariantId variantId,
    const BackwardBuildResult& sharedBackwardBuild,
    const std::vector<std::string>& fusedParameterTargets,
    const PreparedDynamicExpression::TensorMap& ordinaryPreallocatedOutputs,
    const std::unordered_map<std::string, Tensor>& optimizerUpdateInputs,
    Stream& runStream) {
    ApplicationState& app = applications[applicationIndex];
    StampedExecutionVariant& variant = stampedVariant(applicationIndex, variantId);

    const PhysicalOutputs& backwardOutputs = sharedBackwardBuild.outputs;
    if (!backwardOutputs.expr || backwardOutputs.isConditional()) {
        throw runtime_error("CustomLayer fused shared backward requires one flat backward expression.");
    }

    const std::unordered_set<std::string> fusedParameterSet(
        fusedParameterTargets.begin(), fusedParameterTargets.end());
    std::unordered_set<std::string> nonMaterializedParameterNames;
    for (const auto& parameter : parameters) {
        if (!parameter->isTrainingEnabled() || !parameter->hasOptimizer() || parameter->getOptimizer() == nullptr) {
            continue;
        }
        if (!parameter->getOptimizer()->getWeightsGradient().has_value()) {
            nonMaterializedParameterNames.insert(parameter->getName());
        }
    }

    // Import the shared physical backward graph once so every selected logical
    // output retains the source graph's shared ancestry, even when outputs are
    // routed to different fused/ordinary destinations below.
    const Outputs logicalBackwardOutputs = Outputs::fromPhysicalOutputs(backwardOutputs);

    std::unordered_map<std::string, Expression> gradientsByFusedParameter;
    std::vector<std::pair<std::string, Expression>> combinedOutputs;
    std::unordered_map<std::string, OutputMaterializationContract> originalMaterializationByOutput;
    PreparedDynamicExpression::TensorMap preallocatedOutputs;

    auto parameterNameFromGradientOutput = [](const std::string& outputName) -> std::optional<std::string> {
        constexpr const char* suffix = "_grad";
        constexpr size_t suffixLen = 5;
        if (outputName.size() < suffixLen ||
            outputName.compare(outputName.size() - suffixLen, suffixLen, suffix) != 0) {
            return std::nullopt;
        }
        return outputName.substr(0, outputName.size() - suffixLen);
    };

    for (const NamedOutput& output : backwardOutputs.outputs) {
        const std::optional<std::string> parameterName = parameterNameFromGradientOutput(output.name);
        if (parameterName.has_value() && nonMaterializedParameterNames.contains(parameterName.value())) {
            if (fusedParameterSet.contains(parameterName.value())) {
                gradientsByFusedParameter.emplace(
                    parameterName.value(), logicalBackwardOutputs.outputExpression(output.name));
            }
            // A non-materialized parameter that is inactive in this variant has
            // no optimizer action and therefore no public dParameter destination.
            continue;
        }

        auto destinationIt = ordinaryPreallocatedOutputs.find(output.name);
        if (destinationIt == ordinaryPreallocatedOutputs.end()) {
            throw runtime_error(
                "CustomLayer shared backward has no physical destination for retained output '" +
                output.name + "'.");
        }
        combinedOutputs.emplace_back(output.name, logicalBackwardOutputs.outputExpression(output.name));
        originalMaterializationByOutput.emplace(output.name, output.materialization);
        preallocatedOutputs.emplace(output.name, destinationIt->second);
    }

    std::unordered_map<std::string, Tensor> stampInputs = app.forwardPrepared->stampInputs();
    for (const auto& [name, tensor] : app.backwardAdditionalInputsByName) {
        stampInputs[name] = tensor;
    }
    if (!sharedBackwardBuild.forward_value_requirements.empty()) {
        if (variant.forward == nullptr) {
            throw runtime_error("CustomLayer fused shared backward requires the real forward execution plan.");
        }
        bindRetainedForwardValues(sharedBackwardBuild, *variant.forward, stampInputs);
    }

    variant.fusedOptimizerRuntimeScalarBindings.clear();
    variant.fusedOptimizerRuntimeScalars.clear();

    for (const std::string& parameterName : fusedParameterTargets) {
        auto gradIt = gradientsByFusedParameter.find(parameterName);
        if (gradIt == gradientsByFusedParameter.end()) {
            throw runtime_error(
                "CustomLayer could not find expression-local gradient for fused parameter '" + parameterName + "'.");
        }
        auto storageIt = optimizerUpdateInputs.find(parameterName);
        if (storageIt == optimizerUpdateInputs.end()) {
            throw runtime_error(
                "CustomLayer could not find storage for fused parameter '" + parameterName + "'.");
        }

        shared_ptr<PhysicalParameter> targetParameter;
        shared_ptr<Optimizer> optimizer;
        for (const auto& parameter : parameters) {
            if (parameter->getName() == parameterName) {
                targetParameter = parameter;
                optimizer = parameter->getOptimizer();
                break;
            }
        }
        if (targetParameter == nullptr || optimizer == nullptr || !optimizer->supportsDenseUpdateFusion()) {
            throw runtime_error(
                "CustomLayer optimizer fusion requested for unsupported parameter '" + parameterName + "'.");
        }
        if (optimizer->getWeightsGradient().has_value()) {
            throw runtime_error(
                "CustomLayer expression-fused parameter unexpectedly owns a materialized dense gradient: '" +
                parameterName + "'.");
        }

        const std::string prefix = optimizerFusionNamePrefix(parameterName);
        variant.fusedOptimizerRuntimeScalarBindings.push_back({parameterName, optimizer, prefix});
        DenseOptimizerExpression updateExpression =
            optimizer->toDenseUpdateExpression(storageIt->second, gradIt->second, prefix);

        for (const auto& [name, tensor] : updateExpression.inputs) {
            auto [_, inserted] = stampInputs.emplace(name, tensor);
            if (!inserted) {
                throw runtime_error("CustomLayer fused optimizer input name collision: " + name);
            }
        }

        // Optimizers may return several outputs that share internal update
        // ancestry. Import the complete physical update graph once before
        // selecting/transforming individual roots (including weights constraints).
        const Outputs logicalUpdateOutputs = Outputs::fromPhysicalOutputs(updateExpression.outputs);
        for (const NamedOutput& output : updateExpression.outputs.outputs) {
            const std::string uniqueOutputName = optimizerFusionOutputName(parameterName, output.name);
            auto preallocIt = updateExpression.preallocatedOutputs.find(output.name);
            if (preallocIt == updateExpression.preallocatedOutputs.end()) {
                throw runtime_error(
                    "CustomLayer fused optimizer missing preallocated output '" + output.name +
                    "' for parameter '" + parameterName + "'.");
            }
            Expression outputExpression = logicalUpdateOutputs.outputExpression(output.name);
            if (output.name == "weights" && targetParameter->hasConstraints() &&
                targetParameter->supportsDenseExpressionConstraintFusion()) {
                outputExpression = targetParameter->applyDenseExpressionConstraints(
                    outputExpression, prefix + "constraints__");
            }
            combinedOutputs.emplace_back(uniqueOutputName, outputExpression);
            preallocatedOutputs.emplace(uniqueOutputName, preallocIt->second);
        }
    }

    if (combinedOutputs.empty()) {
        throw runtime_error("CustomLayer fused shared backward produced no executable outputs.");
    }

    PhysicalOutputs physicalOutputs = Expression::outputs(combinedOutputs).physicalOutputs();
    for (NamedOutput& output : physicalOutputs.outputs) {
        auto materializationIt = originalMaterializationByOutput.find(output.name);
        if (materializationIt != originalMaterializationByOutput.end()) {
            output.materialization = materializationIt->second;
        }
    }

    PreparedDynamicExpression::TensorMap filteredStampInputs =
        filterTensorInputsForPhysicalOutputs(stampInputs, physicalOutputs);
    PreparedDynamicExpression::TensorScalarMap filteredTensorScalarInputs =
        filterTensorScalarInputsForPhysicalOutputs(
            app.forwardPrepared->tensorScalarInputsForVariant(variantId), physicalOutputs);
    FusedEquation equation = FusedEquation::compile(physicalOutputs, placement.getDeviceNum());
    auto plan = std::make_shared<StampedExecutionPlan>(
        equation.stampForImmediateCrossPlanBackwardLinking(
            filteredStampInputs, runStream, filteredTensorScalarInputs, preallocatedOutputs));
    if (variant.forward != nullptr) {
        plan->linkRmsNormBackwardStatesFrom(*variant.forward);
        plan->linkAttentionBackwardStatesFrom(*variant.forward);
        variant.forward->rebuildConditionalGraphsAfterCrossPlanLinking();
        plan->rebuildConditionalGraphsAfterCrossPlanLinking();
    }
    return plan;
}

const std::unordered_map<std::string, float>& CustomLayer::updateFusedOptimizerRuntimeScalars(
    uint32_t applicationIndex, DynamicExpressionVariantId variantId, uint32_t batchSize) {
    if (batchSize == 0) {
        throw runtime_error("CustomLayer fused optimizer update requires a non-zero batch size.");
    }
    if (applicationIndex >= applications.size()) {
        throw runtime_error("CustomLayer fused optimizer update requested for an invalid application index.");
    }

    StampedExecutionVariant& variant = stampedVariant(applicationIndex, variantId);
    for (const FusedOptimizerRuntimeScalarBinding& binding : variant.fusedOptimizerRuntimeScalarBindings) {
        if (binding.optimizer == nullptr) {
            throw runtime_error("CustomLayer fused optimizer update lost optimizer for parameter '" + binding.parameterName + "'.");
        }

        auto scalars = binding.optimizer->denseUpdateRuntimeScalars(batchSize, binding.namePrefix);
        variant.fusedOptimizerRuntimeScalars.reserve(variant.fusedOptimizerRuntimeScalars.size() + scalars.size());
        for (const auto& [name, value] : scalars) {
            auto [it, inserted] = variant.fusedOptimizerRuntimeScalars.emplace(name, value);
            if (!inserted) {
                it->second = value;
            }
        }
    }
    return variant.fusedOptimizerRuntimeScalars;
}

void CustomLayer::initialize() {
    TrainableLayer::initialize();
    clearForwardArrivalBookkeeping();
    clearBackwardArrivalBookkeeping();
}

PhysicalParameter::StorageContext CustomLayer::buildParameterStorageContext() const {
    if (applications.empty()) {
        throw runtime_error("CustomLayer requires at least one application before parameter storage can be built.");
    }

    std::vector<Tensor> connectedFeatureInputs;
    connectedFeatureInputs.reserve(featureInputs.size());
    for (const auto& featureInput : featureInputs) {
        if (featureInput.has_value()) {
            connectedFeatureInputs.push_back(featureInput.value());
        }
    }

    std::unordered_map<std::string, Tensor> namedFeatureInputs;
    for (uint32_t inputPort = 0; inputPort < inputNames.size(); ++inputPort) {
        const uint32_t flat = inputFlatIndex(0, inputPort);
        if (flat >= featureInputs.size() || !featureInputs[flat].has_value()) {
            throw runtime_error("CustomLayer missing connected feature input for port '" + inputNames[inputPort] + "'.");
        }
        namedFeatureInputs.emplace(inputNames[inputPort], featureInputs[flat].value());
    }

    return PhysicalParameter::StorageContext(std::move(namedFeatureInputs));
}

std::vector<uint64_t> CustomLayer::batchValidityMaskDimensionsForPrimaryInput(const Tensor& primaryInput) const {
    const std::vector<uint64_t> inputDimensions = primaryInput.getDimensions();
    THOR_THROW_IF_FALSE(!inputDimensions.empty());
    std::vector<uint64_t> maskDimensions(inputDimensions.size(), 1);
    maskDimensions.front() = inputDimensions.front();
    return maskDimensions;
}

PreparedDynamicExpression::TensorMap CustomLayer::buildForwardInputs(uint32_t applicationIndex) {
    // Output metadata inference can prepare parameter storage and compile optimizer
    // expressions while graph connections are still being formed, before compileImpl().
    // Network-owned layers already have their model pool installed before connection;
    // standalone implementation layers lazily create their own owner-scoped pool here.
    attachGradientUpdateStream();

    PreparedDynamicExpression::TensorMap inputs;

    for (uint32_t inputPort = 0; inputPort < inputNames.size(); ++inputPort) {
        const uint32_t flat = inputFlatIndex(applicationIndex, inputPort);
        if (flat >= featureInputs.size() || !featureInputs[flat].has_value()) {
            throw runtime_error("CustomLayer missing connected feature input for port '" + inputNames[inputPort] + "'.");
        }
        inputs[inputNames[inputPort]] = featureInputs[flat].value();
    }

    const PhysicalParameter::StorageContext parameterStorageContext = buildParameterStorageContext();
    for (const auto& param : parameters) {
        if (!param->isStorageInitialized()) {
            param->compileStorage(parameterStorageContext);
        }
        std::optional<Tensor> paramStorage = param->getStorage();
        THOR_THROW_IF_FALSE(paramStorage.has_value());
        inputs[param->getName()] = paramStorage.value();
    }

    if (batchValidityMaskEnabled) {
        const uint32_t primaryFlat = primaryInputFlatIndex(applicationIndex);
        THOR_THROW_IF_FALSE(primaryFlat < featureInputs.size());
        THOR_THROW_IF_FALSE(featureInputs[primaryFlat].has_value());
        const std::vector<uint64_t> maskDimensions =
            batchValidityMaskDimensionsForPrimaryInput(featureInputs[primaryFlat].value());
        THOR_THROW_IF_FALSE(!maskDimensions.empty());
        ApplicationState& app = applications.at(applicationIndex);
        app.batchValidityMask = Tensor(
            featureInputs[primaryFlat].value().getPlacement(), TensorDescriptor(DataType::FP32, maskDimensions));
        inputs[Thor::BATCH_VALIDITY_MASK_NAME] = app.batchValidityMask;
    }

    return inputs;
}

PreparedDynamicExpression::TensorMap CustomLayer::buildForwardOutputs(uint32_t applicationIndex) const {
    PreparedDynamicExpression::TensorMap outputs;
    for (uint32_t outputPort = 0; outputPort < outputNames.size(); ++outputPort) {
        const uint32_t flat = outputFlatIndex(applicationIndex, outputPort);
        if (flat < featureOutputs.size() && featureOutputs[flat].has_value()) {
            outputs[outputNames[outputPort]] = featureOutputs[flat].value();
        }
    }
    return outputs;
}

PreparedDynamicExpression::TensorMap CustomLayer::buildBackwardAdditionalInputs(uint32_t applicationIndex) const {
    PreparedDynamicExpression::TensorMap backwardAdditionalInputs;

    if (applicationIndex >= applications.size()) {
        return backwardAdditionalInputs;
    }

    const ApplicationState& app = applications[applicationIndex];

    if (app.backwardGradientPatternCompiled) {
        for (const auto& [outputName, upstreamGradientName] : app.upstreamInputNamesByOutput) {
            const auto outputPortIt = outputNameToPort.find(outputName);
            if (outputPortIt == outputNameToPort.end()) {
                throw runtime_error("CustomLayer compiled backward pattern contains unknown output name: " + outputName);
            }

            const uint32_t flat = outputFlatIndex(applicationIndex, outputPortIt->second);
            if (flat >= errorInputs.size() || !errorInputs[flat].has_value()) {
                throw runtime_error("CustomLayer compiled backward pattern expected an incoming gradient for output port '" + outputName +
                                    "', but that error input is no longer connected.");
            }
            backwardAdditionalInputs[upstreamGradientName] = errorInputs[flat].value();
        }
        for (const auto& [outputName, fusedLossGradient] : app.fusedCustomLossGradientsByOutput) {
            (void)outputName;
            backwardAdditionalInputs[fusedLossGradient.fusedLabelsInputName] = fusedLossGradient.labelsTensor;
            backwardAdditionalInputs[fusedLossGradient.fusedBatchValidityMaskInputName] = fusedLossGradient.batchValidityMask;
        }
        return backwardAdditionalInputs;
    }

    if (!applicationHasAnyDownstreamBackprop(applicationIndex)) {
        return backwardAdditionalInputs;
    }

    for (uint32_t outputPort = 0; outputPort < outputNames.size(); ++outputPort) {
        const uint32_t flat = outputFlatIndex(applicationIndex, outputPort);
        if (flat < errorInputs.size() && errorInputs[flat].has_value()) {
            backwardAdditionalInputs[errorInputNameForOutput(outputPort)] = errorInputs[flat].value();
        }
    }

    return backwardAdditionalInputs;
}

PreparedDynamicExpression::TensorMap CustomLayer::buildBackwardInputGradOutputs(uint32_t applicationIndex) const {
    PreparedDynamicExpression::TensorMap outputs;
    if (applicationIndex >= applications.size()) {
        return outputs;
    }

    const ApplicationState& app = applications[applicationIndex];

    for (uint32_t inputPort = 0; inputPort < inputNames.size(); ++inputPort) {
        const uint32_t flat = inputFlatIndex(applicationIndex, inputPort);
        if (flat >= errorOutputs.size() || !errorOutputs[flat].has_value()) {
            continue;
        }

        Tensor gradOutput = errorOutputs[flat].value();

        // The public graph-level error output must keep the same shape as the connected feature input so that the
        // previous layer receives the gradient shape it expects. However, DynamicExpression may have intentionally
        // rebound this input to a different logical view, e.g. FullyConnected collapses [batch, C, H, W] to
        // [batch, C * H * W] before matmul. In that case, stamp the backward expression with a metadata-only view of
        // the same gradient storage that matches the logical input seen by the expression, while leaving
        // errorOutputs[flat] itself unchanged for upstream propagation.
        if (app.forwardPrepared != nullptr) {
            const auto& logicalInputs = app.forwardPrepared->stampInputs();
            const auto logicalIt = logicalInputs.find(inputNames[inputPort]);
            if (logicalIt != logicalInputs.end()) {
                const Tensor& logicalInput = logicalIt->second;

                if (gradOutput.getPlacement() != logicalInput.getPlacement()) {
                    throw runtime_error("CustomLayer backward gradient output placement does not match logical input placement for port '" +
                                        inputNames[inputPort] + "'.");
                }
                if (gradOutput.getDescriptor().getDataType() != logicalInput.getDescriptor().getDataType()) {
                    throw runtime_error("CustomLayer backward gradient output dtype does not match logical input dtype for port '" +
                                        inputNames[inputPort] + "'.");
                }
                if (gradOutput.getDescriptor().getTotalNumElements() != logicalInput.getDescriptor().getTotalNumElements()) {
                    throw runtime_error(
                        "CustomLayer backward gradient output element count does not match logical input element count for port '" +
                        inputNames[inputPort] + "'.");
                }

                if (gradOutput.getDimensions() != logicalInput.getDimensions()) {
                    gradOutput.reshape(logicalInput.getDimensions());
                }
            }
        }

        outputs[errorOutputNameForInput(inputPort)] = gradOutput;
    }
    return outputs;
}

std::string CustomLayer::errorInputNameForOutput(uint32_t outputPortIndex) const {
    THOR_THROW_IF_FALSE(outputPortIndex < outputNames.size());
    return "__grad_" + outputNames[outputPortIndex];
}

std::string CustomLayer::errorOutputNameForInput(uint32_t inputPortIndex) const {
    THOR_THROW_IF_FALSE(inputPortIndex < inputNames.size());
    return inputNames[inputPortIndex] + "_grad";
}

void CustomLayer::validatePreparedExpressionInputs(const PreparedDynamicExpression& prepared) {
    std::set<std::string> expectedInputNames;
    for (const auto& name : inputNames) {
        expectedInputNames.insert(name);
    }
    for (const auto& param : parameters) {
        expectedInputNames.insert(param->getName());
    }

    std::set<std::string> actualInputNames;
    for (const auto& [name, _] : prepared.stampInputs()) {
        actualInputNames.insert(name);
    }
    // DynamicExpression may legitimately consume a declared CustomLayer input only
    // in its pre-forward hook (for example Attention's ragged RoPE row origins),
    // while the fused equation consumes a hook-generated internal tensor instead.
    // Such dependencies are still real layer inputs even though they must not be
    // forwarded to FusedEquation::stamp().
    for (const auto& [name, _] : prepared.preForwardOnlyInputs()) {
        actualInputNames.insert(name);
    }

    std::set<std::string> missingInputNames;
    for (const auto& expectedName : expectedInputNames) {
        if (!actualInputNames.contains(expectedName)) {
            missingInputNames.insert(expectedName);
        }
    }

    std::set<std::string> unexpectedInputNames;
    for (const auto& actualName : actualInputNames) {
        if (!expectedInputNames.contains(actualName) && !isInternalExpressionInputName(actualName)) {
            unexpectedInputNames.insert(actualName);
        }
    }

    if (!missingInputNames.empty() || !unexpectedInputNames.empty()) {
        throw runtime_error("CustomLayer expression input mismatch. Expected inputs: " + joinNames(expectedInputNames) +
                            " Missing expected inputs: " + joinNames(missingInputNames) +
                            " Unexpected non-internal inputs: " + joinNames(unexpectedInputNames) +
                            " Actual inputs used by prepared expression: " + joinNames(actualInputNames));
    }
}

void CustomLayer::validateStampedOutputNames(const StampedExecutionPlan& stamped,
                                             const std::vector<std::string>& expectedNames,
                                             const char* phase) {
    const std::set<std::string> actualNames = toNameSet(stamped.outputNames());
    const std::set<std::string> expectedNameSet = toNameSet(expectedNames);
    if (actualNames != expectedNameSet) {
        std::string expected;
        for (const auto& name : expectedNameSet)
            expected += name + " ";

        std::string actual;
        for (const auto& name : actualNames)
            actual += name + " ";

        throw runtime_error(std::string("CustomLayer ") + phase + " output mismatch. Expected outputs: " + expected +
                            " Actual outputs: " + actual);
    }
}

std::optional<Tensor> CustomLayer::inferFeatureOutputTensor(uint32_t applicationIndex, uint32_t outputPortIndex) {
    if (outputPortIndex >= outputNames.size()) {
        throw runtime_error("CustomLayer output port index out of range.");
    }
    if (applicationIndex >= applications.size()) {
        throw runtime_error("CustomLayer application index out of range.");
    }
    requireApplicationInputInterfaceConnected(applicationIndex);

    if (!declaredOutputDescriptors.empty()) {
        const DeclaredOutputDescriptor& declared = declaredOutputDescriptors[outputPortIndex];
        if (declared.dimensionsIncludeBatch) {
            return Tensor(placement, TensorDescriptor(declared.dataType, declared.featureDimensions));
        }

        std::optional<uint64_t> batchSize = fixedBatchCapacity.has_value()
                                                ? std::optional<uint64_t>(fixedBatchCapacity.value())
                                                : std::nullopt;
        for (uint32_t inputPort = 0; inputPort < inputNames.size(); ++inputPort) {
            const uint32_t flat = inputFlatIndex(applicationIndex, inputPort);
            if (flat >= featureInputs.size() || !featureInputs[flat].has_value()) {
                throw runtime_error("CustomLayer missing connected feature input for port '" + inputNames[inputPort] + "'.");
            }
            if (inputDimensionsIncludeBatch[inputPort]) {
                continue;
            }

            const std::vector<uint64_t>& inputDimensions = featureInputs[flat].value().getDescriptor().getDimensions();
            if (inputDimensions.empty()) {
                throw runtime_error("CustomLayer feature input port '" + inputNames[inputPort] +
                                    "' has no physical batch dimension.");
            }
            if (!batchSize.has_value()) {
                batchSize = inputDimensions.front();
            } else if (batchSize.value() != inputDimensions.front()) {
                throw runtime_error("CustomLayer feature inputs disagree on physical batch size for application " +
                                    std::to_string(applicationIndex) + ". Expected batch " + std::to_string(batchSize.value()) +
                                    ", while port '" + inputNames[inputPort] + "' has batch " +
                                    std::to_string(inputDimensions.front()) + ".");
            }
        }

        if (!batchSize.has_value() || batchSize.value() == 0) {
            throw runtime_error("CustomLayer requires a non-zero physical batch size to construct output tensors.");
        }

        std::vector<uint64_t> physicalDimensions;
        physicalDimensions.reserve(declared.featureDimensions.size() + 1);
        physicalDimensions.push_back(batchSize.value());
        physicalDimensions.insert(
            physicalDimensions.end(), declared.featureDimensions.begin(), declared.featureDimensions.end());
        return Tensor(placement, TensorDescriptor(declared.dataType, physicalDimensions));
    }

    PreparedDynamicExpression::TensorMap discoveredOutputs;
    PreparedDynamicExpression prepared =
        layerDefinitionExpression.prepare(buildForwardInputs(applicationIndex), discoveredOutputs, computeStream(applicationIndex));
    validatePreparedExpressionInputs(prepared);

    const std::string& outputName = outputNames[outputPortIndex];

    // Output construction during graph connection should discover only metadata.
    // Stamping a throwaway execution plan here is too early for CustomLayer expressions that
    // intentionally rebind inputs to logical views before a specialized stage, such as
    // RMSNorm flattening [outer..., hidden] to [outer, hidden] and then reshaping the
    // public output back to the original feature shape.  The real forward stamp below still
    // receives caller-owned output tensors and therefore preserves the fused CustomLayer path.
    if (prepared.tensorScalarInputs().empty() && prepared.preallocatedOutputs().empty() &&
        prepared.requestedOutputShapes().empty()) {
        const auto outputShapes = prepared.equation().getOutputShapes(prepared.stampInputs());
        const auto outputDataTypes = prepared.equation().getOutputDataTypes(prepared.stampInputs());

        const auto shapeIt = outputShapes.find(outputName);
        if (shapeIt == outputShapes.end()) {
            throw runtime_error("CustomLayer expression did not infer output shape for port '" + outputName + "'.");
        }
        const auto dtypeIt = outputDataTypes.find(outputName);
        if (dtypeIt == outputDataTypes.end()) {
            throw runtime_error("CustomLayer expression did not infer output dtype for port '" + outputName + "'.");
        }

        return Tensor(placement, TensorDescriptor(dtypeIt->second, shapeIt->second));
    }

    StampedExecutionPlan stamped = prepared.stamp();
    return stamped.output(outputName);
}

void CustomLayer::compileImpl() {
    TrainableLayer::compileImpl();

    THOR_THROW_IF_FALSE(placement.getMemDevice() == TensorPlacement::MemDevices::GPU);

    if (applications.empty()) {
        throw runtime_error("CustomLayer must have at least one connected input interface.");
    }

    clearForwardArrivalBookkeeping();
    clearBackwardArrivalBookkeeping();

    bool compiledParameterInitializers = false;
    bool compiledOptimizers = false;
    numBackwardApplications = 0;

    for (uint32_t applicationIndex = 0; applicationIndex < applications.size(); ++applicationIndex) {
        ApplicationState& app = applications[applicationIndex];

        for (uint32_t inputPort = 0; inputPort < inputNames.size(); ++inputPort) {
            const uint32_t flat = inputFlatIndex(applicationIndex, inputPort);
            if (flat >= featureInputs.size() || !featureInputs[flat].has_value()) {
                throw runtime_error(getLayerType() + " missing connected input port '" + inputNames[inputPort] +
                                    "' for application " + std::to_string(applicationIndex) + ".");
            }
        }

        for (uint32_t outputPort = 0; outputPort < outputNames.size(); ++outputPort) {
            const uint32_t flat = outputFlatIndex(applicationIndex, outputPort);
            if (flat >= featureOutputs.size()) {
                ensureApplicationStorageAllocated(applicationIndex);
            }
            if (flat >= featureOutputs.size()) {
                throw runtime_error("CustomLayer internal error while allocating output port '" + outputNames[outputPort] +
                                    "' for application " + std::to_string(applicationIndex) + ".");
            }
            if (!featureOutputs[flat].has_value()) {
                throw runtime_error("CustomLayer missing connected output port '" + outputNames[outputPort] + "' for application " +
                                    std::to_string(applicationIndex) + ".");
            }
        }

        app.forwardInputsByName = buildForwardInputs(applicationIndex);
        app.forwardOutputsByName = buildForwardOutputs(applicationIndex);

        app.forwardPrepared = std::make_shared<PreparedDynamicExpression>(
            layerDefinitionExpression.prepare(app.forwardInputsByName, app.forwardOutputsByName, computeStream(applicationIndex)));
        validatePreparedExpressionInputs(*app.forwardPrepared);

        // Expression-local optimizer-fusion policy for this application. A name
        // appears here only when every backward-capable execution variant that can
        // update it produces the parameter gradient entirely inside an ordinary
        // fused Expression region.
        // Stage-boundary gradients and all multi-application cases remain materialized.
        std::unordered_set<std::string> expressionFusedOptimizerParameterNames;

        for (DynamicExpressionVariantId variantId : app.forwardPrepared->executionVariantIds()) {
            const auto inferredOutputShapes = app.forwardPrepared->equationForVariant(variantId).getOutputShapes(
                app.forwardPrepared->stampInputs(), app.forwardPrepared->tensorScalarInputsForVariant(variantId));
            const auto inferredOutputDataTypes = app.forwardPrepared->equationForVariant(variantId).getOutputDataTypes(
                app.forwardPrepared->stampInputs(), app.forwardPrepared->tensorScalarInputsForVariant(variantId));
            for (const std::string& outputName : outputNames) {
                const auto actualOutputIt = app.forwardOutputsByName.find(outputName);
                const auto shapeIt = inferredOutputShapes.find(outputName);
                const auto dataTypeIt = inferredOutputDataTypes.find(outputName);
                if (actualOutputIt == app.forwardOutputsByName.end() || shapeIt == inferredOutputShapes.end() ||
                    dataTypeIt == inferredOutputDataTypes.end()) {
                    throw runtime_error("CustomLayer execution variant " + std::to_string(variantId) +
                                        " failed to validate the declared physical output for port '" + outputName + "'.");
                }

                const TensorDescriptor inferredDescriptor(dataTypeIt->second, shapeIt->second);
                const TensorDescriptor& declaredDescriptor = actualOutputIt->second.getDescriptor();
                if (inferredDescriptor != declaredDescriptor) {
                    throw runtime_error("CustomLayer execution variant " + std::to_string(variantId) +
                                        " output descriptor does not match the API-declared output for port '" + outputName +
                                        "' in application " + std::to_string(applicationIndex) + ". Variant inferred " +
                                        inferredDescriptor.toString() + ", but the runtime output tensor is " +
                                        declaredDescriptor.toString() +
                                        ". Python CustomLayer build(context) implementations must derive batch-dependent shapes from the "
                                        "placement-time context rather than retaining the batch-1 tensor used for API shape inference.");
                }
            }
        }

        // Decide dense-gradient materialization per parameter from the *actual
        // backward gradient expression*. For the one flat single-application
        // case where optimizer fusion is even possible, defer optimizer compilation
        // until the downstream-gradient pattern has been captured below. All other
        // cases keep the ordinary materialized-gradient contract immediately.
        const bool deferOptimizerCompilationForExpressionFusion =
            !compiledOptimizers && !isInferenceOnly() && !executesNativeSharedBackwardPlan() &&
            applicationHasAnyDownstreamBackprop(applicationIndex) &&
            canFuseOptimizerUpdatesForApplication(applicationIndex);
        if (!compiledOptimizers && !deferOptimizerCompilationForExpressionFusion) {
            for (const auto& parameter : parameters) {
                if (!parameter->isTrainable()) {
                    continue;
                }
                parameter->compileOptimizer(gradientUpdateStream, isInferenceOnly(), true);
            }
            compiledOptimizers = true;
        }

        if (!compiledParameterInitializers) {
            std::unordered_set<std::string> parameterNames;
            for (const auto& parameter : parameters) {
                parameterNames.insert(parameter->getName());
            }
            const auto parameterFanOverrides = app.forwardPrepared->getParameterFanOverrides(parameterNames);
            for (const auto& parameter : parameters) {
                auto it = parameterFanOverrides.find(parameter->getName());
                if (it != parameterFanOverrides.end()) {
                    parameter->compileInitializer(it->second.fan_in, it->second.fan_out);
                } else {
                    parameter->compileInitializer();
                }
            }
            compiledParameterInitializers = true;
        }

        app.stampedVariants.clear();
        app.evaluationVariantId = app.forwardPrepared->evaluationVariantId();
        app.forwardVariantThisPass.reset();
        const std::vector<DynamicExpressionVariantId> executionVariantIds = app.forwardPrepared->executionVariantIds();
        for (DynamicExpressionVariantId variantId : executionVariantIds) {
            StampedExecutionVariant variant;
            variant.preForwardHook = app.forwardPrepared->preForwardHookForVariant(variantId);
            variant.supportsBackward = app.forwardPrepared->executionVariantSupportsBackward(variantId);
            app.stampedVariants.emplace(variantId, std::move(variant));
        }

        // BR2 must discover every saved-forward dependency before stamping the real
        // forward.  A backward plan stores Tensor handles from that one forward plan,
        // so replacing a previously stamped plan after binding a backward would leave
        // the backward attached to tensors that are never produced.  Centralize forward
        // stamping here and require exactly one stamp per execution variant.
        auto stampForwardVariant = [&](DynamicExpressionVariantId variantId,
                                       const ConditionalForwardRetentionMap& retainedForwardNodes,
                                       const ConditionalForwardRetentionMap& retainedForwardEpilogueAuxNodes) {
            StampedExecutionVariant& variant = stampedVariant(applicationIndex, variantId);
            if (variant.forward != nullptr) {
                throw runtime_error("CustomLayer attempted to stamp one forward execution variant more than once.");
            }

            const bool retainAnything = !retainedForwardNodes.empty() || !retainedForwardEpilogueAuxNodes.empty();
            if (!retainAnything) {
                variant.forward = std::make_shared<StampedExecutionPlan>(
                    app.forwardPrepared->stampExecutionVariant(variantId, app.forwardOutputsByName));
            } else {
                variant.forward = std::make_shared<StampedExecutionPlan>(
                    app.forwardPrepared->stampExecutionVariantRetainingForwardValues(variantId,
                                                                                      retainedForwardNodes,
                                                                                      retainedForwardEpilogueAuxNodes,
                                                                                      app.forwardOutputsByName));
            }
            validateStampedOutputNames(*variant.forward,
                                       outputNames,
                                       retainAnything ? "forward execution variant with retained backward values"
                                                      : "forward execution variant");
            onForwardExecutionVariantStamped(applicationIndex, variantId, variant.forward, variant.supportsBackward);
        };
        auto stampAllForwardVariantsWithoutRetention = [&]() {
            static const ConditionalForwardRetentionMap noRetainedForwardNodes;
            static const ConditionalForwardRetentionMap noRetainedForwardEpilogueAuxNodes;
            for (DynamicExpressionVariantId variantId : executionVariantIds) {
                stampForwardVariant(variantId, noRetainedForwardNodes, noRetainedForwardEpilogueAuxNodes);
            }
        };

        const StampedExecutionVariant& activeTrainingVariant = stampedVariant(applicationIndex, activeTrainingVariantId);
        if (!activeTrainingVariant.supportsBackward && !isInferenceOnly()) {
            throw runtime_error("CustomLayer active training execution variant " +
                                std::to_string(activeTrainingVariantId) + " does not support backward execution.");
        }

        app.backwardAdditionalInputsByName.clear();
        app.backwardInputGradOutputsByName.clear();
        app.expectedBackwardErrorInputTensorIds.clear();
        app.upstreamInputNamesByOutput.clear();
        app.fusedCustomLossGradientsByOutput.clear();
        app.upstreamOutputNames.clear();
        app.backwardGradientPatternCompiled = false;

        if (isInferenceOnly() || isBackPropStub()) {
            stampAllForwardVariantsWithoutRetention();
            app.backwardGradientPatternCompiled = true;
            continue;
        }

        if (!applicationHasAnyDownstreamBackprop(applicationIndex)) {
            stampAllForwardVariantsWithoutRetention();
            pruneUpstreamErrorOutputsForApplication(applicationIndex);
            app.backwardAdditionalInputsByName.clear();
            app.backwardInputGradOutputsByName.clear();
            app.backwardGradientPatternCompiled = true;
            continue;
        }

        // Fused CustomLoss seeding currently inlines the loss-gradient expression into one flat
        // backward expression. A conditional backward is a PhysicalOutputs tree, and a conditional
        // CustomLoss gradient likewise cannot be cloned into that flat expression. Keep either case
        // on the ordinary materialized gradient path; notifying the owning loss before it compiles
        // makes it stamp its normal gradient expression into the error tensor.
        // The native shared backward consumes the ordinary materialized downstream
        // gradient tensor.  Do not inline a CustomLoss gradient into a second backward
        // construction path; the loss will stamp its normal gradient producer instead.
        const bool allowFusedCustomLossGradient =
            !executesNativeSharedBackwardPlan() && !applicationHasConditionalBackwardVariant(applicationIndex);
        auto fusedCustomLossGradientCanInline = [&](const FusedCustomLossGradient& fused) {
            PreparedDynamicExpression::TensorMap gradientInputs;
            gradientInputs.emplace(fused.predictionsName, fused.predictionsTensor);
            gradientInputs.emplace(fused.labelsName, fused.labelsTensor);
            gradientInputs.emplace(fused.batchValidityMaskName, fused.batchValidityMask);

            DynamicExpressionBuild gradientBuild =
                fused.gradientExpression.build(gradientInputs, {}, computeStream(applicationIndex));
            return !gradientBuild.equation->physicalOutputs().isConditional();
        };

        // Snapshot the per-application downstream-gradient pattern at compile time. For a
        // given application, each output gradient either exists every backward pass or never
        // exists, based on the downstream topology observed during compileImpl(). Runtime
        // backward() only waits for this fixed set and the backward stamps are specialized
        // to this fixed partial upstream map.
        for (uint32_t outputPort = 0; outputPort < outputNames.size(); ++outputPort) {
            const uint32_t flat = outputFlatIndex(applicationIndex, outputPort);
            if (flat < errorInputs.size() && errorInputs[flat].has_value()) {
                app.expectedBackwardErrorInputTensorIds.insert(errorInputs[flat].value().getTensorId());
                auto fusedLossIt = fusedCustomLossGradientByOutputFlatIndex.find(flat);
                const bool canInlineFusedLoss =
                    fusedLossIt != fusedCustomLossGradientByOutputFlatIndex.end() && allowFusedCustomLossGradient &&
                    fusedCustomLossGradientCanInline(fusedLossIt->second);
                if (canInlineFusedLoss) {
                    app.fusedCustomLossGradientsByOutput.emplace(outputNames[outputPort], fusedLossIt->second);
                } else {
                    if (fusedLossIt != fusedCustomLossGradientByOutputFlatIndex.end() && fusedLossIt->second.ownerLoss != nullptr) {
                        fusedLossIt->second.ownerLoss->notifyFusedGradientUnregisteredFromDrivingLayer(
                            fusedLossIt->second.predictionsTensor);
                    }
                    app.upstreamInputNamesByOutput[outputNames[outputPort]] = errorInputNameForOutput(outputPort);
                }
                app.upstreamOutputNames.insert(outputNames[outputPort]);
            }
        }
        app.backwardGradientPatternCompiled = true;

        app.backwardAdditionalInputsByName = buildBackwardAdditionalInputs(applicationIndex);
        if (!app.backwardAdditionalInputsByName.empty()) {
            numBackwardApplications += 1;
        }

        // Keep every connected upstream error-output target for this application, even when a target is not
        // reachable from the subset of forward outputs that received incoming gradients. In that case AutoDiff
        // emits a zero gradient for the requested wrt input without requiring a synthetic upstream zero tensor.
        // This preserves graph-level backprop bookkeeping: upstream layers connected to the full input interface
        // still receive exactly one backward() call carrying a zero tensor for inactive input ports.
        std::vector<std::string> inputTargets;
        for (uint32_t inputPort = 0; inputPort < inputNames.size(); ++inputPort) {
            const uint32_t flat = inputFlatIndex(applicationIndex, inputPort);
            if (flat < errorOutputs.size() && errorOutputs[flat].has_value()) {
                inputTargets.push_back(inputNames[inputPort]);
            }
        }
        app.backwardInputGradOutputsByName = buildBackwardInputGradOutputs(applicationIndex);

        std::vector<std::string> allTrainableParameterTargets;
        for (auto& parameter : parameters) {
            if (parameter->isTrainingEnabled()) {
                allTrainableParameterTargets.push_back(parameter->getName());
            }
        }

        auto combinedBackwardTargets = [](const std::vector<std::string>& inputs,
                                          const std::vector<std::string>& parameterTargets) {
            std::vector<std::string> targets = inputs;
            for (const std::string& parameterName : parameterTargets) {
                if (std::find(targets.begin(), targets.end(), parameterName) == targets.end()) {
                    targets.push_back(parameterName);
                }
            }
            return targets;
        };

        // Build the one generic clear VJP before deciding optimizer storage so each
        // flat dParameter can be classified directly from the exact combined VJP
        // that becomes the runtime owner; no second AutoDiff invocation is needed
        // merely to decide whether an optimizer can be grafted onto it. Graph-level
        // conditionals are included here as well. They remain ineligible for
        // optimizer fusion, but their one conditional VJP owns dInput and every
        // materialized dParameter together.
        if (!wantsNativeSharedBackwardPlan() &&
            !app.backwardAdditionalInputsByName.empty() &&
            !allTrainableParameterTargets.empty()) {
            for (DynamicExpressionVariantId buildVariantId : executionVariantIds) {
                StampedExecutionVariant& buildVariant = stampedVariant(applicationIndex, buildVariantId);
                if (!buildVariant.supportsBackward) {
                    continue;
                }

                std::vector<std::string> activeParameterTargets =
                    app.forwardPrepared->equationForVariant(buildVariantId).filterTensorInputNamesReachableFromOutputs(
                        allTrainableParameterTargets, app.upstreamOutputNames);
                if (activeParameterTargets.empty() && !allTrainableParameterTargets.empty() &&
                    !app.upstreamOutputNames.empty()) {
                    activeParameterTargets = allTrainableParameterTargets;
                }
                buildVariant.activeParameterTargetNames =
                    std::unordered_set<std::string>(activeParameterTargets.begin(), activeParameterTargets.end());

                const std::vector<std::string> clearTargets =
                    combinedBackwardTargets(inputTargets, allTrainableParameterTargets);
                buildVariant.genericSharedBackwardClearBuild = buildBackwardOutputsForApplication(
                    applicationIndex, buildVariantId, clearTargets, GradientAccumulationTargets{});
                onGenericSharedBackwardBuildCreated(
                    applicationIndex,
                    buildVariantId,
                    buildVariant.genericSharedBackwardClearBuild.value(),
                    GradientAccumulationTargets{});
            }
        }

        if (!compiledOptimizers) {
            // Restore optimizer fusion only when one application owns the complete
            // parameter gradient and that dParameter is expression-local.
            // A GEMM/conv/reduction/RMSNorm/custom-CUDA/etc. anywhere in the dParameter
            // dependency subgraph is a physical stage boundary, so that parameter keeps
            // a materialized dW and the optimizer remains downstream of backward.
            if (canFuseOptimizerUpdatesForApplication(applicationIndex) &&
                !wantsNativeSharedBackwardPlan() &&
                !applicationHasConditionalBackwardVariant(applicationIndex)) {
                for (const auto& parameter : parameters) {
                    if (parameter->isTrainingEnabled() && parameter->hasOptimizer() &&
                        parameter->getOptimizer() != nullptr &&
                        parameter->getOptimizer()->supportsDenseUpdateFusion()) {
                        expressionFusedOptimizerParameterNames.insert(parameter->getName());
                    }
                }

                for (DynamicExpressionVariantId candidateVariantId : executionVariantIds) {
                    const StampedExecutionVariant& candidateVariant = stampedVariant(applicationIndex, candidateVariantId);
                    if (!candidateVariant.supportsBackward || expressionFusedOptimizerParameterNames.empty()) {
                        continue;
                    }
                    if (!candidateVariant.genericSharedBackwardClearBuild.has_value() ||
                        !candidateVariant.genericSharedBackwardClearBuild->outputs.expr ||
                        candidateVariant.genericSharedBackwardClearBuild->outputs.isConditional()) {
                        // Optimizer storage is chosen once for the parameter, not per
                        // execution variant. If any backward-capable variant cannot be
                        // classified from the one shared clear VJP, none of the remaining
                        // candidates may safely omit their materialized gradient buffer.
                        expressionFusedOptimizerParameterNames.clear();
                        break;
                    }

                    std::unordered_map<std::string, uint32_t> gradientNodeByParameter;
                    for (const NamedOutput& output : candidateVariant.genericSharedBackwardClearBuild->outputs.outputs) {
                        constexpr const char* suffix = "_grad";
                        constexpr size_t suffixLen = 5;
                        if (output.name.size() >= suffixLen &&
                            output.name.compare(output.name.size() - suffixLen, suffixLen, suffix) == 0) {
                            gradientNodeByParameter.emplace(
                                output.name.substr(0, output.name.size() - suffixLen), output.node_idx);
                        }
                    }

                    // Dense-gradient allocation is global to the optimizer. A parameter
                    // may be optimizer-fused only if it is active in *every* backward-
                    // capable variant. If a clear contribution selects a branch/output
                    // where the parameter is inactive, Thor still needs to establish a
                    // real zero gradient and run the ordinary optimizer semantics (which
                    // can matter for momentum/weight decay even when dW == 0). Therefore
                    // an inactive variant forces that parameter back to materialized dW.
                    std::vector<std::string> candidates(
                        expressionFusedOptimizerParameterNames.begin(),
                        expressionFusedOptimizerParameterNames.end());
                    for (const std::string& parameterName : candidates) {
                        if (!candidateVariant.activeParameterTargetNames.contains(parameterName)) {
                            expressionFusedOptimizerParameterNames.erase(parameterName);
                            continue;
                        }
                        const auto gradIt = gradientNodeByParameter.find(parameterName);
                        if (gradIt == gradientNodeByParameter.end() ||
                            expressionSubgraphContainsStageBoundary(
                                *candidateVariant.genericSharedBackwardClearBuild->outputs.expr, gradIt->second)) {
                            expressionFusedOptimizerParameterNames.erase(parameterName);
                        }
                    }
                }
            }

            for (const auto& parameter : parameters) {
                if (!parameter->isTrainable()) {
                    continue;
                }
                const bool materializeDenseGradient =
                    !expressionFusedOptimizerParameterNames.contains(parameter->getName());
                parameter->compileOptimizer(gradientUpdateStream, isInferenceOnly(), materializeDenseGradient);
            }
            compiledOptimizers = true;
        }

        if (executesNativeSharedBackwardPlan() && applications.size() != 1) {
            throw runtime_error(
                "Native shared-backward execution currently requires exactly one physical application; "
                "Attention's public layer contract provides one application.");
        }
        if (executesNativeSharedBackwardPlan() && !wantsNativeSharedBackwardPlan()) {
            throw runtime_error("Native shared-backward execution requires native shared-backward stamping.");
        }

        for (DynamicExpressionVariantId variantId : executionVariantIds) {
            StampedExecutionVariant& variant = stampedVariant(applicationIndex, variantId);
            if (!variant.supportsBackward) {
                static const ConditionalForwardRetentionMap noRetainedForwardNodes;
                static const ConditionalForwardRetentionMap noRetainedForwardEpilogueAuxNodes;
                stampForwardVariant(variantId, noRetainedForwardNodes, noRetainedForwardEpilogueAuxNodes);
                continue;
            }

            std::vector<std::string> activeParameterTargets;
            if (variant.genericSharedBackwardClearBuild.has_value()) {
                for (const std::string& parameterName : allTrainableParameterTargets) {
                    if (variant.activeParameterTargetNames.contains(parameterName)) {
                        activeParameterTargets.push_back(parameterName);
                    }
                }
            } else {
                activeParameterTargets =
                    app.forwardPrepared->equationForVariant(variantId).filterTensorInputNamesReachableFromOutputs(
                        allTrainableParameterTargets, app.upstreamOutputNames);
                if (activeParameterTargets.empty() && !allTrainableParameterTargets.empty() &&
                    !app.upstreamOutputNames.empty()) {
                    // The reachability filter is an optimization for reducing the set of
                    // parameter-gradient targets. It must not be allowed to turn a live
                    // backward path into a silent no-op optimizer update.
                    activeParameterTargets = allTrainableParameterTargets;
                }
                variant.activeParameterTargetNames =
                    std::unordered_set<std::string>(activeParameterTargets.begin(), activeParameterTargets.end());
            }

            // Construct the production generic shared-VJP builds. Each build is
            // one complete VJP for one application contribution; clear versus
            // accumulate changes gradient-commit semantics, not VJP cardinality
            // within that contribution. The clear build must establish every
            // trainable parameter gradient (including an explicit zero for an
            // unreachable/inactive parameter), while the accumulate build only
            // contributes parameters reachable from this execution variant. dInput
            // always overwrites; only parameter targets accumulate.
            //
            // Native shared-backward specializations already own a one-VJP path.
            // For conditionals, FusedEquation's reachability query conservatively
            // keeps every requested parameter target because branch selection is a
            // runtime decision; conditional AutoDiff then emits zero-on-clear or
            // preserve-on-accumulate behavior for a parameter absent from the
            // selected branch.
            if (!wantsNativeSharedBackwardPlan() &&
                !app.backwardAdditionalInputsByName.empty() &&
                !allTrainableParameterTargets.empty()) {
                std::vector<std::string> accumulateParameterTargets;
                accumulateParameterTargets.reserve(activeParameterTargets.size());
                for (const std::string& parameterName : activeParameterTargets) {
                    if (!expressionFusedOptimizerParameterNames.contains(parameterName)) {
                        accumulateParameterTargets.push_back(parameterName);
                    }
                }
                const std::vector<std::string> accumulateTargets =
                    combinedBackwardTargets(inputTargets, accumulateParameterTargets);
                // With exactly one physical application, this application is always
                // the first and only contribution in a backward pass, so an
                // accumulate variant is unreachable regardless of how many materialized
                // parameter gradients it owns. Multi-application layers retain the
                // clear/accumulate pair.
                const bool accumulatePlanCanEverRun = applications.size() > 1;
                if (accumulatePlanCanEverRun && !accumulateTargets.empty()) {
                    const GradientAccumulationTargets accumulateWrtNames(
                        accumulateParameterTargets.begin(), accumulateParameterTargets.end());
                    variant.genericSharedBackwardAccumulateWrtNames = accumulateWrtNames;
                    variant.genericSharedBackwardAccumulateBuild = buildBackwardOutputsForApplication(
                        applicationIndex, variantId, accumulateTargets, accumulateWrtNames);
                    onGenericSharedBackwardBuildCreated(
                        applicationIndex,
                        variantId,
                        variant.genericSharedBackwardAccumulateBuild.value(),
                        accumulateWrtNames);
                }
            }

            if (wantsNativeSharedBackwardPlan() && !app.backwardAdditionalInputsByName.empty() &&
                app.fusedCustomLossGradientsByOutput.empty()) {
                std::vector<std::string> combinedTargets = inputTargets;
                for (const std::string& parameterName : activeParameterTargets) {
                    if (std::find(combinedTargets.begin(), combinedTargets.end(), parameterName) == combinedTargets.end()) {
                        combinedTargets.push_back(parameterName);
                    }
                }

                if (!combinedTargets.empty()) {
                    // The native shared VJP participates in the same generic
                    // retained-forward contract as every other flat backward graph.
                    // Preflight the VJP before stamping the real forward, retain the
                    // exact computed primals it declares, then bind those tensors into
                    // the one shared backward. This replaces the old Attention-only
                    // Q/K/V/O saved-input enumeration and also covers projection/post-op
                    // intermediates without replay or ad-hoc special cases.
                    BackwardBuildResult sharedBackwardBuild = buildBackwardOutputsForApplication(
                        applicationIndex, variantId, combinedTargets, false);

                    ConditionalForwardRetentionMap retainedForwardNodes;
                    ConditionalForwardRetentionMap retainedForwardEpilogueAuxNodes;
                    for (const ForwardValueRequirement& requirement : sharedBackwardBuild.forward_value_requirements) {
                        if (requirement.forward_node_index == UINT32_MAX || requirement.backward_input_name.empty()) {
                            throw runtime_error("Native shared backward produced an incomplete saved-forward requirement.");
                        }
                        switch (requirement.kind) {
                            case ForwardValueRequirementKind::NodeOutput:
                                retainedForwardNodes[requirement.conditional_branch_path].push_back(
                                    requirement.forward_node_index);
                                break;
                            case ForwardValueRequirementKind::MatmulEpilogueAux:
                                retainedForwardEpilogueAuxNodes[requirement.conditional_branch_path].push_back(
                                    requirement.forward_node_index);
                                break;
                            case ForwardValueRequirementKind::ConditionalPredicate:
                                // The forward conditional already materializes its
                                // predicate tensor; no hidden retained output is
                                // needed for the branch decision.
                                break;
                            default:
                                throw runtime_error("Native shared backward produced an unknown saved-forward requirement kind.");
                        }
                    }
                    for (auto& [path, nodes] : retainedForwardNodes) {
                        (void)path;
                        std::sort(nodes.begin(), nodes.end());
                        nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
                    }
                    for (auto& [path, nodes] : retainedForwardEpilogueAuxNodes) {
                        (void)path;
                        std::sort(nodes.begin(), nodes.end());
                        nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
                    }

                    stampForwardVariant(variantId, retainedForwardNodes, retainedForwardEpilogueAuxNodes);

                    PhysicalOutputs& sharedBackwardOutputs = sharedBackwardBuild.outputs;
                    FusedEquation sharedBackwardEquation =
                        FusedEquation::compile(sharedBackwardOutputs, placement.getDeviceNum());
                    PreparedDynamicExpression::TensorMap availableSharedBackwardInputs =
                        app.forwardPrepared->stampInputs();
                    for (const auto& [name, tensor] : app.backwardAdditionalInputsByName) {
                        availableSharedBackwardInputs[name] = tensor;
                    }
                    if (variant.forward == nullptr) {
                        throw runtime_error("Native shared backward requires the real forward execution plan.");
                    }
                    bindRetainedForwardValues(sharedBackwardBuild, *variant.forward, availableSharedBackwardInputs);

                    // Retained-forward bindings are a superset of the inputs used by
                    // any particular differentiated target set. Stamp only the ABI
                    // actually declared by this shared backward equation.
                    PreparedDynamicExpression::TensorMap sharedBackwardInputs =
                        filterTensorInputsForPhysicalOutputs(availableSharedBackwardInputs, sharedBackwardOutputs);
                    PreparedDynamicExpression::TensorScalarMap sharedBackwardTensorScalarInputs =
                        filterTensorScalarInputsForPhysicalOutputs(
                            app.forwardPrepared->tensorScalarInputsForVariant(variantId), sharedBackwardOutputs);

                    PreparedDynamicExpression::TensorMap sharedBackwardPreallocatedOutputs;
                    if (executesNativeSharedBackwardPlan()) {
                        // One native backward owns every requested gradient destination. Input
                        // gradients go directly to their graph-connected error tensors and parameter
                        // gradients go directly to the optimizer-owned materialized buffers. No
                        // second VJP, temporary gradient, or D2D save/copy is introduced.
                        PreparedDynamicExpression::TensorMap candidateOutputs = app.backwardInputGradOutputsByName;
                        for (const std::string& parameterName : activeParameterTargets) {
                            shared_ptr<PhysicalParameter> targetParameter;
                            for (const auto& parameter : parameters) {
                                if (parameter->getName() == parameterName) {
                                    targetParameter = parameter;
                                    break;
                                }
                            }
                            if (targetParameter == nullptr || !targetParameter->hasOptimizer() ||
                                targetParameter->getOptimizer() == nullptr ||
                                !targetParameter->getOptimizer()->getWeightsGradient().has_value()) {
                                throw runtime_error(
                                    "Native shared backward requires an optimizer-owned materialized gradient buffer for parameter '" +
                                    parameterName + "'.");
                            }
                            candidateOutputs.emplace(
                                parameterName + "_grad", targetParameter->getOptimizer()->getWeightsGradient().value());
                        }

                        for (const NamedOutput& output : sharedBackwardOutputs.outputs) {
                            auto outputIt = candidateOutputs.find(output.name);
                            if (outputIt == candidateOutputs.end()) {
                                throw runtime_error(
                                    "Native shared backward has no preallocated physical destination for output '" +
                                    output.name + "'.");
                            }
                            sharedBackwardPreallocatedOutputs.emplace(output.name, outputIt->second);
                        }
                    }

                    auto sharedBackwardPlan = std::make_shared<StampedExecutionPlan>(
                        sharedBackwardEquation.stampForImmediateCrossPlanBackwardLinking(
                            sharedBackwardInputs,
                            computeStream(applicationIndex),
                            sharedBackwardTensorScalarInputs,
                            sharedBackwardPreallocatedOutputs));
                    sharedBackwardPlan->linkRmsNormBackwardStatesFrom(*variant.forward);
                    sharedBackwardPlan->linkAttentionBackwardStatesFrom(*variant.forward);
                    variant.forward->rebuildConditionalGraphsAfterCrossPlanLinking();
                    sharedBackwardPlan->rebuildConditionalGraphsAfterCrossPlanLinking();
                    variant.nativeSharedBackward = sharedBackwardPlan;
                    onNativeSharedBackwardExecutionVariantStamped(applicationIndex, variantId, sharedBackwardPlan);
                }
            }

            if (wantsNativeSharedBackwardPlan() && variant.forward == nullptr) {
                static const ConditionalForwardRetentionMap noRetainedForwardNodes;
                static const ConditionalForwardRetentionMap noRetainedForwardEpilogueAuxNodes;
                stampForwardVariant(variantId, noRetainedForwardNodes, noRetainedForwardEpilogueAuxNodes);
            }

            if (executesNativeSharedBackwardPlan()) {
                if (!app.backwardAdditionalInputsByName.empty() && variant.nativeSharedBackward == nullptr) {
                    throw runtime_error("Native shared backward execution requires a stamped shared backward plan.");
                }
                // Attention's parameter gradients are already outputs of the one shared VJP.
                // Do not stamp backwardWeights{Clear,Accumulate} or a fused optimizer plan,
                // any of which would independently differentiate the forward expression again.
                continue;
            }

            if (trainingUpdateDiagnosticsEnabled()) {
                std::fprintf(stderr,
                             "THOR_TRAINING_UPDATE_DIAGNOSTIC layer=%s app=%u variant=%u compile_backward downstream=%d fused_loss=%d backward_additional_inputs=%zu upstream_outputs=%s all_trainable=%s active_trainable=%s gradient_update_stream=%d\n",
                             diagnosticLabel().c_str(),
                             applicationIndex,
                             variantId,
                             applicationHasAnyDownstreamBackprop(applicationIndex) ? 1 : 0,
                             applicationHasFusedCustomLossGradient(applicationIndex) ? 1 : 0,
                             app.backwardAdditionalInputsByName.size(),
                             joinNames(app.upstreamOutputNames).c_str(),
                             joinNames(allTrainableParameterTargets).c_str(),
                             joinNames(variant.activeParameterTargetNames).c_str(),
                             gradientUpdateStream.has_value() ? 1 : 0);
            }

            std::vector<std::string> fusedParameterTargets;
            // Optimizer fusion is restored only for active parameters whose complete
            // dParameter subgraph was proven expression-local above. The
            // optimizer is grafted onto the one shared VJP below; this list must not
            // cause a second differentiated parameter-gradient graph.
            for (const std::string& parameterName : activeParameterTargets) {
                if (expressionFusedOptimizerParameterNames.contains(parameterName)) {
                    fusedParameterTargets.push_back(parameterName);
                    variant.optimizerUpdateFusedParameterNames.insert(parameterName);
                }
            }

            std::vector<std::string> allMaterializedParameterTargets;
            std::vector<std::string> activeMaterializedParameterTargets;
            PreparedDynamicExpression::TensorMap allMaterializedParameterPreallocatedOutputs;
            PreparedDynamicExpression::TensorMap activeMaterializedParameterPreallocatedOutputs;
            std::unordered_map<std::string, Tensor> parameterStorageByName;

            for (auto& parameter : parameters) {
                if (!parameter->isTrainingEnabled()) {
                    continue;
                }

                THOR_THROW_IF_FALSE(parameter->hasOptimizer());
                const shared_ptr<Optimizer>& parameterOptimizer = parameter->getOptimizer();
                THOR_THROW_IF_FALSE(parameterOptimizer != nullptr);
                THOR_THROW_IF_FALSE(parameter->getStorage().has_value());
                parameterStorageByName[parameter->getName()] = parameter->getStorage().value();

                if (variant.optimizerUpdateFusedParameterNames.contains(parameter->getName())) {
                    continue;
                }

                const std::optional<Tensor> gradientTensor = parameterOptimizer->getWeightsGradient();
                if (!gradientTensor.has_value()) {
                    // Dense optimizer fusion intentionally compiles fusion-capable parameters without an
                    // optimizer-owned gradient buffer. A parameter can nevertheless be inactive for this
                    // execution variant/application (for example, when only one of several CustomLayer
                    // outputs participates in backprop). Such a parameter is neither fused nor materialized
                    // for this variant and needs no gradient clear/update at runtime.
                    //
                    // An *active* non-fused parameter without a dense buffer would be a real ownership bug:
                    // its gradient has nowhere to go. Keep that case as a hard failure instead of silently
                    // dropping a contribution.
                    if (variant.activeParameterTargetNames.contains(parameter->getName())) {
                        throw runtime_error(
                            "CustomLayer active non-fused parameter '" + parameter->getName() +
                            "' has no optimizer-owned materialized gradient buffer.");
                    }
                    continue;
                }

                const std::string gradName = parameter->getName() + "_grad";
                allMaterializedParameterTargets.push_back(parameter->getName());
                allMaterializedParameterPreallocatedOutputs[gradName] = gradientTensor.value();
                if (variant.activeParameterTargetNames.contains(parameter->getName())) {
                    activeMaterializedParameterTargets.push_back(parameter->getName());
                    activeMaterializedParameterPreallocatedOutputs[gradName] = gradientTensor.value();
                }
            }

            const bool useGenericSharedBackwardOwnership =
                variant.genericSharedBackwardClearBuild.has_value();

            // Generic trainable CustomLayer variants have exactly one derivative
            // owner: the combined shared VJP. The only remaining split
            // backward build is the parameterless/non-trainable dInput-only path.
            // Preflight it before stamping the real forward so retained primals bind
            // to the one forward plan that will actually execute.
            std::optional<BackwardBuildResult> backwardErrorBuild;
            if (!app.backwardAdditionalInputsByName.empty() &&
                !useGenericSharedBackwardOwnership &&
                !inputTargets.empty()) {
                backwardErrorBuild = buildBackwardOutputsForApplication(
                    applicationIndex, variantId, inputTargets, false);
            }

            ConditionalForwardRetentionMap retainedForwardNodes;
            ConditionalForwardRetentionMap retainedForwardEpilogueAuxNodes;
            auto appendRequirements = [&](const std::optional<BackwardBuildResult>& build) {
                if (!build.has_value()) {
                    return;
                }
                for (const ForwardValueRequirement& requirement : build->forward_value_requirements) {
                    if (requirement.forward_node_index == UINT32_MAX || requirement.backward_input_name.empty()) {
                        throw runtime_error("CustomLayer BR2/BR4 preflight produced an incomplete saved-forward requirement.");
                    }
                    switch (requirement.kind) {
                        case ForwardValueRequirementKind::NodeOutput:
                            retainedForwardNodes[requirement.conditional_branch_path].push_back(
                                requirement.forward_node_index);
                            break;
                        case ForwardValueRequirementKind::MatmulEpilogueAux:
                            retainedForwardEpilogueAuxNodes[requirement.conditional_branch_path].push_back(
                                requirement.forward_node_index);
                            break;
                        case ForwardValueRequirementKind::ConditionalPredicate:
                            // The predicate is already a durable output of the
                            // forward conditional's predicate child plan.
                            break;
                        default:
                            throw runtime_error("CustomLayer BR4 preflight produced an unknown saved-forward requirement kind.");
                    }
                }
            };
            appendRequirements(backwardErrorBuild);
            // the shared plans are linked to this same real forward,
            // so its retention set includes the combined VJPs. Under generic
            // shared ownership (flat or conditional) the legacy split builds
            // above are intentionally absent.
            appendRequirements(variant.genericSharedBackwardClearBuild);
            appendRequirements(variant.genericSharedBackwardAccumulateBuild);
            for (auto& [path, nodes] : retainedForwardNodes) {
                (void)path;
                std::sort(nodes.begin(), nodes.end());
                nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
            }
            for (auto& [path, nodes] : retainedForwardEpilogueAuxNodes) {
                (void)path;
                std::sort(nodes.begin(), nodes.end());
                nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
            }

            if (variant.forward == nullptr) {
                stampForwardVariant(variantId, retainedForwardNodes, retainedForwardEpilogueAuxNodes);
            } else if (!retainedForwardNodes.empty() || !retainedForwardEpilogueAuxNodes.empty()) {
                // A specialization that asks for the legacy native shared plan but does
                // not execute it would have stamped the forward before generic BR2
                // requirements were known. No current layer has that contract; reject it
                // rather than silently replacing the plan and invalidating native state.
                throw runtime_error(
                    "CustomLayer cannot add generic saved-forward retention after a native-shared forward plan was stamped.");
            }

            // stamp the combined generic VJPs against their physical
            // destinations. These plans are the sole runtime owner for generic
            // trainable variants, including conditionals. dInput is graph-connected
            // directly and dParameter writes optimizer-owned materialized buffers.
            auto exactPreallocatedOutputsForBuild = [](const BackwardBuildResult& build,
                                                       const PreparedDynamicExpression::TensorMap& candidates,
                                                       const char* planKind) {
                PreparedDynamicExpression::TensorMap exactOutputs;
                for (const NamedOutput& output : build.outputs.outputs) {
                    const auto candidateIt = candidates.find(output.name);
                    if (candidateIt == candidates.end()) {
                        throw runtime_error(
                            std::string("CustomLayer ") + planKind +
                            " shared backward has no preallocated destination for output '" + output.name + "'.");
                    }
                    exactOutputs.emplace(output.name, candidateIt->second);
                }
                return exactOutputs;
            };

            if (variant.genericSharedBackwardClearBuild.has_value()) {
                PreparedDynamicExpression::TensorMap clearCandidates = app.backwardInputGradOutputsByName;
                clearCandidates.insert(allMaterializedParameterPreallocatedOutputs.begin(),
                                       allMaterializedParameterPreallocatedOutputs.end());

                if (!fusedParameterTargets.empty()) {
                    variant.genericSharedBackwardClear = buildGenericSharedBackwardWithFusedOptimizerPlan(
                        applicationIndex,
                        variantId,
                        variant.genericSharedBackwardClearBuild.value(),
                        fusedParameterTargets,
                        clearCandidates,
                        parameterStorageByName,
                        computeStream(applicationIndex));
                } else {
                    PreparedDynamicExpression::TensorMap clearPreallocatedOutputs =
                        exactPreallocatedOutputsForBuild(
                            variant.genericSharedBackwardClearBuild.value(), clearCandidates, "clear");
                    variant.genericSharedBackwardClear = stampBackwardForApplication(
                        applicationIndex,
                        variantId,
                        variant.genericSharedBackwardClearBuild.value(),
                        false,
                        clearPreallocatedOutputs,
                        computeStream(applicationIndex));
                }

                if (variant.genericSharedBackwardClear == nullptr) {
                    throw runtime_error("CustomLayer failed to stamp the generic shared clear backward plan.");
                }
                onGenericSharedBackwardExecutionVariantStamped(
                    applicationIndex,
                    variantId,
                    variant.genericSharedBackwardClear,
                    GradientAccumulationTargets{});
            }

            if (variant.genericSharedBackwardAccumulateBuild.has_value()) {
                PreparedDynamicExpression::TensorMap accumulateCandidates = app.backwardInputGradOutputsByName;
                accumulateCandidates.insert(activeMaterializedParameterPreallocatedOutputs.begin(),
                                            activeMaterializedParameterPreallocatedOutputs.end());
                PreparedDynamicExpression::TensorMap accumulatePreallocatedOutputs =
                    exactPreallocatedOutputsForBuild(
                        variant.genericSharedBackwardAccumulateBuild.value(), accumulateCandidates, "accumulate");
                const GradientAccumulationTargets& accumulateWrtNames =
                    variant.genericSharedBackwardAccumulateWrtNames;

                variant.genericSharedBackwardAccumulate = stampBackwardForApplication(
                    applicationIndex,
                    variantId,
                    variant.genericSharedBackwardAccumulateBuild.value(),
                    true,
                    accumulatePreallocatedOutputs,
                    computeStream(applicationIndex));
                if (variant.genericSharedBackwardAccumulate == nullptr) {
                    throw runtime_error("CustomLayer failed to stamp the generic shared accumulate backward plan.");
                }
                onGenericSharedBackwardExecutionVariantStamped(
                    applicationIndex,
                    variantId,
                    variant.genericSharedBackwardAccumulate,
                    accumulateWrtNames);
            }

            if (backwardErrorBuild.has_value()) {
                variant.backwardError = stampBackwardForApplication(applicationIndex,
                                                                    variantId,
                                                                    backwardErrorBuild.value(),
                                                                    false,
                                                                    app.backwardInputGradOutputsByName,
                                                                    computeStream(applicationIndex));
            }

            if (trainingUpdateDiagnosticsEnabled()) {
                std::fprintf(stderr,
                             "THOR_TRAINING_UPDATE_DIAGNOSTIC layer=%s app=%u variant=%u compile_parameter_targets fused=%s all_materialized=%s active_materialized=%s materialized_preallocated_outputs=%zu active_preallocated_outputs=%zu\n",
                             diagnosticLabel().c_str(),
                             applicationIndex,
                             variantId,
                             joinNames(fusedParameterTargets).c_str(),
                             joinNames(allMaterializedParameterTargets).c_str(),
                             joinNames(activeMaterializedParameterTargets).c_str(),
                             allMaterializedParameterPreallocatedOutputs.size(),
                             activeMaterializedParameterPreallocatedOutputs.size());
            }

            if (useGenericSharedBackwardOwnership && variant.backwardError != nullptr) {
                throw runtime_error(
                    "CustomLayer generic shared backward cannot coexist with a split dInput backward owner.");
            }
        }
    }

    if (trainingUpdateDiagnosticsEnabled()) {
        for (uint32_t applicationIndex = 0; applicationIndex < applications.size(); ++applicationIndex) {
            const ApplicationState& app = applications[applicationIndex];
            for (const auto& [variantId, variant] : app.stampedVariants) {
                std::fprintf(stderr,
                             "THOR_TRAINING_UPDATE_DIAGNOSTIC layer=%s app=%u variant=%u compiled_stamps backward_supported=%d shared_clear=%d shared_accumulate=%d backward_error=%d expected_backward_errors=%zu num_backward_applications=%u\n",
                             diagnosticLabel().c_str(),
                             applicationIndex,
                             variantId,
                             variant.supportsBackward ? 1 : 0,
                             variant.genericSharedBackwardClear != nullptr ? 1 : 0,
                             variant.genericSharedBackwardAccumulate != nullptr ? 1 : 0,
                             variant.backwardError != nullptr ? 1 : 0,
                             app.expectedBackwardErrorInputTensorIds.size(),
                             numBackwardApplications);
            }
        }
    }

    // Now that every application has a compiled backward-gradient pattern, reset
    // the arrival bookkeeping from those fixed per-application expectations.
    clearBackwardArrivalBookkeeping();
}

std::optional<Tensor> CustomLayer::createFeatureOutputTensor() {
    if (outputNames.size() != 1) {
        throw runtime_error("CustomLayer::createFeatureOutputTensor() without a connection type is only valid for single-output layers.");
    }
    std::optional<Tensor> featureOutput = inferFeatureOutputTensor(0, 0);
    THOR_THROW_IF_FALSE(featureOutput.has_value());
    return featureOutput;
}

std::optional<Tensor> CustomLayer::createErrorOutputTensor(bool backPropagateError, uint32_t connectionNumber) {
    if (!backPropagateError || isInferenceOnly()) {
        return std::nullopt;
    }

    DecodedConnection decoded = decodeInputConnectionType(static_cast<int>(connectionNumber));
    const uint32_t flat = inputFlatIndex(decoded.applicationIndex, decoded.portIndex);
    THOR_THROW_IF_FALSE(flat < featureInputs.size());
    THOR_THROW_IF_FALSE(featureInputs[flat].has_value());
    return featureInputs[flat].value().clone();
}

std::optional<Tensor> CustomLayer::connectToPreviousLayer(
    Layer* previousLayer, std::optional<Tensor> featureInput, Stream stream, bool backPropagateError, int connectionType) {
    THOR_THROW_IF_FALSE(!compiled);

    const DecodedConnection decoded = decodeInputConnectionType(connectionType);
    ensureApplicationStorageAllocated(decoded.applicationIndex);
    const uint32_t flat = inputFlatIndex(decoded.applicationIndex, decoded.portIndex);

    THOR_THROW_IF_FALSE(featureInput.has_value());
    THOR_THROW_IF_FALSE(!previousLayers[flat].has_value());
    THOR_THROW_IF_FALSE(!featureInputs[flat].has_value());
    THOR_THROW_IF_FALSE(!errorOutputs[flat].has_value());

    previousLayers[flat] = previousLayer;
    featureInputs[flat] = featureInput;
    featureInputsConnectedForPorts[flat] = featureInput;
    streams[flat] = stream;
    errorOutputs[flat] = createErrorOutputTensor(backPropagateError, connectionType);
    errorOutputsConnectedForPorts[flat] = errorOutputs[flat];

    ensureNoDeviceCrossing(placement);
    return errorOutputs[flat];
}

void CustomLayer::connectToNextLayer(Layer* nextLayer, int driverConnectionType, int loaderConnectionType) {
    THOR_THROW_IF_FALSE(!compiled);

    const DecodedConnection decoded = decodeOutputConnectionType(driverConnectionType);
    ensureApplicationStorageAllocated(decoded.applicationIndex);
    const uint32_t flat = outputFlatIndex(decoded.applicationIndex, decoded.portIndex);

    if (!featureOutputs[flat].has_value()) {
        std::optional<Tensor> outputTensor = inferFeatureOutputTensor(decoded.applicationIndex, decoded.portIndex);
        THOR_THROW_IF_FALSE(outputTensor.has_value());
        featureOutputs[flat] = outputTensor;
        featureOutputsConnectedForPorts[flat] = outputTensor;
    }

    nextLayers[flat] = nextLayer;

    errorInputs[flat] = nextLayer->connectToPreviousLayer(this,
                                                          featureOutputs[flat],
                                                          computeStream(decoded.applicationIndex),
                                                          shouldConnectToBackPropErrorIn() && !isBackPropStub(),
                                                          loaderConnectionType);
    errorInputsConnectedForPorts[flat] = errorInputs[flat];

    ensureNoDeviceCrossing(placement);
}

void CustomLayer::pruneUpstreamErrorOutputsForApplication(uint32_t applicationIndex) {
    for (uint32_t inputPort = 0; inputPort < inputNames.size(); ++inputPort) {
        const uint32_t inputFlat = inputFlatIndex(applicationIndex, inputPort);
        if (inputFlat >= errorOutputs.size() || !errorOutputs[inputFlat].has_value()) {
            continue;
        }

        if (previousLayers[inputFlat].has_value()) {
            previousLayers[inputFlat].value()->replaceErrorInput(errorOutputs[inputFlat], std::nullopt);
        }

        errorOutputs[inputFlat].reset();
        errorOutputsConnectedForPorts[inputFlat].reset();
    }
}

void CustomLayer::replaceErrorInput(std::optional<Tensor> oldErrorInput, std::optional<Tensor> newErrorInput) {
    THOR_THROW_IF_FALSE(oldErrorInput.has_value());

    bool replacementHappened = false;
    std::set<uint32_t> affectedApplications;
    for (uint32_t flat = 0; flat < errorInputs.size(); ++flat) {
        if (!errorInputs[flat].has_value() || errorInputs[flat].value() != oldErrorInput.value()) {
            continue;
        }
        errorInputs[flat] = newErrorInput;
        errorInputsConnectedForPorts[flat] = newErrorInput;
        affectedApplications.insert(flat / outputNames.size());
        replacementHappened = true;
    }
    THOR_THROW_IF_FALSE(replacementHappened);

    for (uint32_t applicationIndex : affectedApplications) {
        if (applicationHasAnyDownstreamBackprop(applicationIndex)) {
            continue;
        }
        for (uint32_t inputPort = 0; inputPort < inputNames.size(); ++inputPort) {
            const uint32_t inputFlat = inputFlatIndex(applicationIndex, inputPort);
            if (inputFlat >= errorOutputs.size() || !errorOutputs[inputFlat].has_value()) {
                continue;
            }
            if (previousLayers[inputFlat].has_value()) {
                previousLayers[inputFlat].value()->replaceErrorInput(errorOutputs[inputFlat], std::nullopt);
            }
            errorOutputs[inputFlat].reset();
            errorOutputsConnectedForPorts[inputFlat].reset();
        }
        clearBackwardArrivalBookkeeping(applicationIndex);
    }
}


std::string CustomLayer::diagnosticLabel() {
    std::string label = getLayerType() + "#" + std::to_string(getId());
    std::string layerName = getName();
    if (!layerName.empty()) {
        label += "(" + layerName + ")";
    }
    return label;
}

void CustomLayer::synchronizeComputeStreamForForwardInputs(uint32_t applicationIndex) {
    Stream& runStream = computeStream(applicationIndex);
    const uint32_t runFlat = primaryInputFlatIndex(applicationIndex);
    for (uint32_t inputPort = 0; inputPort < inputNames.size(); ++inputPort) {
        const uint32_t flat = inputFlatIndex(applicationIndex, inputPort);
        if (flat == runFlat || flat >= streams.size() || !featureInputs[flat].has_value()) {
            continue;
        }
        ApplicationState& app = applications[applicationIndex];
        THOR_THROW_IF_FALSE(inputPort < app.forwardInputReadyEvents.size());
        runStream.waitFor(streams[flat], app.forwardInputReadyEvents[inputPort]);
    }
}

void CustomLayer::propagateApplicationRowPartitionHostState(uint32_t applicationIndex, uint32_t sourceInputPort) {
    THOR_THROW_IF_FALSE(applicationIndex < applications.size());
    THOR_THROW_IF_FALSE(sourceInputPort < inputNames.size());
    const uint32_t sourceFlat = inputFlatIndex(applicationIndex, sourceInputPort);
    THOR_THROW_IF_FALSE(sourceFlat < featureInputs.size());
    THOR_THROW_IF_FALSE(featureInputs[sourceFlat].has_value());
    const Tensor sourceCarrier = featureInputs[sourceFlat].value();

    for (uint32_t outputPort = 0; outputPort < outputNames.size(); ++outputPort) {
        const uint32_t outputFlat = outputFlatIndex(applicationIndex, outputPort);
        THOR_THROW_IF_FALSE(outputFlat < featureOutputs.size());
        if (!featureOutputs[outputFlat].has_value())
            continue;
        RowPartitionRuntime::propagateHostState(sourceCarrier, featureOutputs[outputFlat].value());
    }
}

void CustomLayer::forward(std::optional<Tensor> featureInput, bool validationPass, uint32_t batchSize) {
    THOR_THROW_IF_FALSE(running);
    THOR_THROW_IF_FALSE(featureInput.has_value());

    // If training was enabled or disabled, expression will need to be recompiled
    // because the set of gradients changed.
    bool needsRecompile = false;
    for (const auto& param : parameters) {
        if (param->needsExpressionRecompile()) {
            needsRecompile = true;
            param->informExpressionRecompiled();
        }
    }
    if (needsRecompile)
        compileImpl();

    const bool singleApplicationSingleInputFastPath =
        applications.size() == 1 && inputNames.size() == 1 && featureInputs.size() == 1 && featureInputs[0].has_value() &&
        featureInputs[0].value() == featureInput.value();

    std::vector<uint32_t> candidateApplications;
    if (singleApplicationSingleInputFastPath) {
        candidateApplications.push_back(0);
    } else {
        std::set<uint32_t> deduplicatedCandidateApplications;
        for (uint32_t flat = 0; flat < featureInputs.size(); ++flat) {
            if (featureInputs[flat].has_value() && featureInputs[flat].value() == featureInput.value()) {
                deduplicatedCandidateApplications.insert(flat / inputNames.size());
            }
        }
        candidateApplications.assign(deduplicatedCandidateApplications.begin(), deduplicatedCandidateApplications.end());
    }
    THOR_THROW_IF_FALSE(!candidateApplications.empty());

    if (isStartOfForward) {
        if (weightsAreUpToDateEventValid) {
            for (const Stream& dataStream : uniqueDataStreams) {
                dataStream.waitEvent(weightsAreUpToDateEvent);
            }
        }
        weightsAreUpToDateEventValid = false;
        isStartOfForward = false;
        isStartOfBackward = true;
        clearGradientFirstThisBackwardPass = false;
        clearForwardArrivalBookkeeping();
    }

    const unsigned long tensorId = featureInput.value().getTensorId();
    for (uint32_t applicationIndex : candidateApplications) {
        ApplicationState& app = applications[applicationIndex];
        if (app.forwardRanThisPass) {
            if (trainingUpdateDiagnosticsEnabled()) {
                std::fprintf(stderr,
                             "THOR_TRAINING_UPDATE_DIAGNOSTIC layer=%s app=%u forward_skip reason=already_ran tensor=%lu batch=%u validation=%d is_start_forward=%d is_start_backward=%d downstream_backprop=%d expected_backward_errors=%zu waiting_forward=%zu num_backward_applications=%u completed_backward_applications=%u\n",
                             diagnosticLabel().c_str(),
                             applicationIndex,
                             tensorId,
                             batchSize,
                             validationPass ? 1 : 0,
                             isStartOfForward ? 1 : 0,
                             isStartOfBackward ? 1 : 0,
                             applicationHasAnyDownstreamBackprop(applicationIndex) ? 1 : 0,
                             app.expectedBackwardErrorInputTensorIds.size(),
                             app.stillWaitingForForwardInputTensorIds.size(),
                             numBackwardApplications,
                             numBackwardApplicationsCompletedThisPass);
            }
            continue;
        }
        std::optional<uint32_t> physicalBatchCapacity = fixedBatchCapacity;
        if (!physicalBatchCapacity.has_value()) {
            bool currentInputHasImplicitBatchDimension = false;
            for (uint32_t inputPort = 0; inputPort < inputNames.size(); ++inputPort) {
                const uint32_t flat = inputFlatIndex(applicationIndex, inputPort);
                if (flat < featureInputs.size() && featureInputs[flat].has_value() &&
                    featureInputs[flat].value() == featureInput.value() && !inputDimensionsIncludeBatch[inputPort]) {
                    currentInputHasImplicitBatchDimension = true;
                    break;
                }
            }
            if (currentInputHasImplicitBatchDimension) {
                const std::vector<uint64_t> inputDimensions = featureInput.value().getDimensions();
                THOR_THROW_IF_FALSE(!inputDimensions.empty());
                THOR_THROW_IF_FALSE(inputDimensions.front() >= 1);
                THOR_THROW_IF_FALSE(inputDimensions.front() <= std::numeric_limits<uint32_t>::max());
                physicalBatchCapacity = static_cast<uint32_t>(inputDimensions.front());
            }
        }

        // Inputs whose dimensions already include the complete physical shape (for example a scalar produced by a
        // batch reduction) do not expose an implicit leading batch-capacity dimension.  In that case batchSize is
        // runtime cardinality metadata and must not be bounded by the tensor's first logical dimension.
        const uint32_t resolvedValidExampleCount =
            batchSize == 0 ? physicalBatchCapacity.value_or(1U) : batchSize;
        THOR_THROW_IF_FALSE(resolvedValidExampleCount >= 1);
        if (physicalBatchCapacity.has_value()) {
            THOR_THROW_IF_FALSE(resolvedValidExampleCount <= physicalBatchCapacity.value());
        }
        if (app.batchCardinalitySet) {
            THOR_THROW_IF_FALSE(app.currentValidExampleCount == resolvedValidExampleCount);
        } else {
            app.currentValidExampleCount = resolvedValidExampleCount;
            app.batchCardinalitySet = true;
        }

        if (!singleApplicationSingleInputFastPath && app.stillWaitingForForwardInputTensorIds.count(tensorId) == 0) {
            if (trainingUpdateDiagnosticsEnabled()) {
                std::fprintf(stderr,
                             "THOR_TRAINING_UPDATE_DIAGNOSTIC layer=%s app=%u forward_skip reason=unexpected_input tensor=%lu batch=%u validation=%d is_start_forward=%d waiting_forward=%zu all_forward=%zu num_backward_applications=%u completed_backward_applications=%u\n",
                             diagnosticLabel().c_str(),
                             applicationIndex,
                             tensorId,
                             batchSize,
                             validationPass ? 1 : 0,
                             isStartOfForward ? 1 : 0,
                             app.stillWaitingForForwardInputTensorIds.size(),
                             app.allForwardInputTensorIds.size(),
                             numBackwardApplications,
                             numBackwardApplicationsCompletedThisPass);
            }
            continue;
        }
        if (!singleApplicationSingleInputFastPath) {
            app.stillWaitingForForwardInputTensorIds.erase(tensorId);
        }

        if (!singleApplicationSingleInputFastPath && !app.stillWaitingForForwardInputTensorIds.empty()) {
            if (trainingUpdateDiagnosticsEnabled()) {
                std::fprintf(stderr,
                             "THOR_TRAINING_UPDATE_DIAGNOSTIC layer=%s app=%u forward_waiting tensor=%lu batch=%u validation=%d remaining=%zu all_forward=%zu\n",
                             diagnosticLabel().c_str(),
                             applicationIndex,
                             tensorId,
                             batchSize,
                             validationPass ? 1 : 0,
                             app.stillWaitingForForwardInputTensorIds.size(),
                             app.allForwardInputTensorIds.size());
            }
            continue;
        }

        if (trainingUpdateDiagnosticsEnabled()) {
            std::fprintf(stderr,
                         "THOR_TRAINING_UPDATE_DIAGNOSTIC layer=%s app=%u forward_run tensor=%lu batch=%u validation=%d downstream_backprop=%d expected_backward_errors=%zu num_backward_applications=%u completed_backward_applications=%u\n",
                         diagnosticLabel().c_str(),
                         applicationIndex,
                         tensorId,
                         batchSize,
                         validationPass ? 1 : 0,
                         applicationHasAnyDownstreamBackprop(applicationIndex) ? 1 : 0,
                         app.expectedBackwardErrorInputTensorIds.size(),
                         numBackwardApplications,
                         numBackwardApplicationsCompletedThisPass);
        }

        const bool emitLayerDiagnostics = layerSubmitDiagnosticsActive();
        const auto appForwardStart = emitLayerDiagnostics ? layerSubmitDiagnosticNow() : LayerSubmitDiagnosticTimePoint();
        uint64_t syncMicros = 0;
        uint64_t computeMicros = 0;
        uint64_t downstreamMicros = 0;
        uint64_t resetMicros = 0;

        app.forwardRanThisPass = true;
        if (!singleApplicationSingleInputFastPath) {
            const auto syncStart = emitLayerDiagnostics ? layerSubmitDiagnosticNow() : LayerSubmitDiagnosticTimePoint();
            synchronizeComputeStreamForForwardInputs(applicationIndex);
            if (emitLayerDiagnostics) {
                syncMicros = layerSubmitDiagnosticElapsedMicros(syncStart, layerSubmitDiagnosticNow());
            }
        }
        const auto computeStart = emitLayerDiagnostics ? layerSubmitDiagnosticNow() : LayerSubmitDiagnosticTimePoint();
        if (batchValidityMaskEnabled) {
            THOR_THROW_IF_FALSE(app.batchValidityMask.isInitialized());
            writeBatchValidityMask(app.batchValidityMask, app.currentValidExampleCount, computeStream(applicationIndex));
        }
        computeFeatureOutForPass(inputFlatIndex(applicationIndex, 0), validationPass);
        prepareApplicationOutputsForDownstream(applicationIndex);
        if (emitLayerDiagnostics) {
            computeMicros = layerSubmitDiagnosticElapsedMicros(computeStart, layerSubmitDiagnosticNow());
        }

        const auto downstreamStart = emitLayerDiagnostics ? layerSubmitDiagnosticNow() : LayerSubmitDiagnosticTimePoint();
        uint64_t downstreamCount = 0;
        for (uint32_t outputPort = 0; outputPort < outputNames.size(); ++outputPort) {
            const uint32_t flat = outputFlatIndex(applicationIndex, outputPort);
            if (!nextLayers[flat].has_value())
                continue;
            downstreamCount += 1;
            nextLayers[flat].value()->forward(featureOutputs[flat], validationPass, app.currentValidExampleCount);
        }
        if (emitLayerDiagnostics) {
            downstreamMicros = layerSubmitDiagnosticElapsedMicros(downstreamStart, layerSubmitDiagnosticNow());
        }

        // In inference-only / forward-only topologies there is no backward pass to mark the end of an execution
        // cycle. Reset this application as soon as its forward has been emitted so the next call can wait for a
        // fresh set of input arrivals and re-run the stamped expression. Training applications keep the existing
        // forward/backward cycle reset so gradients and parameter updates stay pass-scoped.
        // Validation/inference runs do not invoke backward(), even when the topology has
        // downstream backprop connections for training.  A validation pass therefore
        // must reset this application's forward-arrival state here; otherwise the next
        // train/validation batch sees app.forwardRanThisPass from the validation pass,
        // skips the forward computation, and downstream multi-input layers can receive
        // a second labels tensor for a stale feature tensor.
        const auto resetStart = emitLayerDiagnostics ? layerSubmitDiagnosticNow() : LayerSubmitDiagnosticTimePoint();
        const bool resetForwardState = validationPass || !applicationHasAnyDownstreamBackprop(applicationIndex);
        if (resetForwardState) {
            clearForwardArrivalBookkeeping(applicationIndex);
        }
        if (emitLayerDiagnostics) {
            resetMicros = layerSubmitDiagnosticElapsedMicros(resetStart, layerSubmitDiagnosticNow());
            emitLayerSubmitDiagnostic("custom_forward_application",
                                      diagnosticLabel(),
                                      getId(),
                                      layerSubmitDiagnosticElapsedMicros(appForwardStart, layerSubmitDiagnosticNow()),
                                      {{"app", applicationIndex},
                                       {"sync_us", syncMicros},
                                       {"compute_us", computeMicros},
                                       {"downstream_us", downstreamMicros},
                                       {"reset_us", resetMicros},
                                       {"downstream_count", downstreamCount},
                                       {"reset_forward_state", resetForwardState ? 1UL : 0UL}});
        }
    }
}

void CustomLayer::backward(std::optional<Tensor> errorInput, uint32_t batchSize) {
    THOR_THROW_IF_FALSE(running);

    if (!errorInput.has_value()) {
        if (trainingUpdateDiagnosticsEnabled()) {
            std::fprintf(stderr,
                         "THOR_TRAINING_UPDATE_DIAGNOSTIC layer=%s backward_skip reason=no_error_input batch=%u is_start_forward=%d is_start_backward=%d num_backward_applications=%u completed_backward_applications=%u\n",
                         diagnosticLabel().c_str(),
                         batchSize,
                         isStartOfForward ? 1 : 0,
                         isStartOfBackward ? 1 : 0,
                         numBackwardApplications,
                         numBackwardApplicationsCompletedThisPass);
        }
        return;
    }

    const bool singleApplicationSingleInputOutputFastPath =
        applications.size() == 1 && inputNames.size() == 1 && outputNames.size() == 1 && errorInputs.size() == 1 &&
        errorInputs[0].has_value() && errorInputs[0].value() == errorInput.value() &&
        applications[0].expectedBackwardErrorInputTensorIds.size() == 1;

    std::vector<uint32_t> candidateApplications;
    if (singleApplicationSingleInputOutputFastPath) {
        candidateApplications.push_back(0);
    } else {
        std::set<uint32_t> deduplicatedCandidateApplications;
        for (uint32_t flat = 0; flat < errorInputs.size(); ++flat) {
            if (errorInputs[flat].has_value() && errorInputs[flat].value() == errorInput.value()) {
                deduplicatedCandidateApplications.insert(flat / outputNames.size());
            }
        }
        candidateApplications.assign(deduplicatedCandidateApplications.begin(), deduplicatedCandidateApplications.end());
    }
    THOR_THROW_IF_FALSE(!candidateApplications.empty());

    if (trainingUpdateDiagnosticsEnabled()) {
        std::fprintf(stderr,
                     "THOR_TRAINING_UPDATE_DIAGNOSTIC layer=%s backward_entry batch=%u candidate_applications=%zu is_start_of_backward=%d num_backward_applications=%u completed_this_pass=%u\n",
                     diagnosticLabel().c_str(),
                     batchSize,
                     candidateApplications.size(),
                     isStartOfBackward ? 1 : 0,
                     numBackwardApplications,
                     numBackwardApplicationsCompletedThisPass);
    }

    if (isStartOfBackward) {
        clearBackwardArrivalBookkeeping();
        isStartOfBackward = false;
        clearGradientFirstThisBackwardPass = true;
        lastParameterGradientContributionApplicationThisPass.reset();
    }

    const unsigned long tensorId = errorInput.value().getTensorId();
    for (uint32_t applicationIndex : candidateApplications) {
        ApplicationState& app = applications[applicationIndex];
        if (app.backwardRanThisPass) {
            continue;
        }
        if (!singleApplicationSingleInputOutputFastPath && app.stillWaitingForBackwardErrorInputTensorIds.count(tensorId) == 0) {
            continue;
        }
        if (!singleApplicationSingleInputOutputFastPath) {
            app.stillWaitingForBackwardErrorInputTensorIds.erase(tensorId);
        }

        if (!singleApplicationSingleInputOutputFastPath && !app.stillWaitingForBackwardErrorInputTensorIds.empty()) {
            if (trainingUpdateDiagnosticsEnabled()) {
                std::fprintf(stderr,
                             "THOR_TRAINING_UPDATE_DIAGNOSTIC layer=%s app=%u backward_waiting remaining_errors=%zu\n",
                             diagnosticLabel().c_str(),
                             applicationIndex,
                             app.stillWaitingForBackwardErrorInputTensorIds.size());
            }
            continue;
        }

        StampedExecutionVariant& variant = backwardVariantForApplication(applicationIndex);
        const DynamicExpressionVariantId variantId = app.forwardVariantThisPass.value();
        const bool useNativeSharedBackward = executesNativeSharedBackwardPlan();
        const bool useGenericSharedBackward = variant.genericSharedBackwardClear != nullptr;
        if (useNativeSharedBackward && variant.nativeSharedBackward == nullptr) {
            throw runtime_error("Native shared backward execution requested without a stamped shared backward plan.");
        }
        if (useNativeSharedBackward && useGenericSharedBackward) {
            throw runtime_error("CustomLayer cannot execute native and generic shared backward ownership simultaneously.");
        }
        if (useGenericSharedBackward && variant.backwardError != nullptr) {
            throw runtime_error("CustomLayer generic shared backward found a duplicate split dInput runtime owner.");
        }

        if (trainingUpdateDiagnosticsEnabled()) {
            std::fprintf(stderr,
                         "THOR_TRAINING_UPDATE_DIAGNOSTIC layer=%s app=%u variant=%u backward_ready batch=%u active_parameters=%s generic_shared=%d backward_error_stamp=%d fused_parameters=%s\n",
                         diagnosticLabel().c_str(),
                         applicationIndex,
                         variantId,
                         batchSize,
                         joinNames(variant.activeParameterTargetNames).c_str(),
                         useGenericSharedBackward ? 1 : 0,
                         variant.backwardError != nullptr ? 1 : 0,
                         joinNames(variant.optimizerUpdateFusedParameterNames).c_str());
        }

        const bool emitLayerDiagnostics = layerSubmitDiagnosticsActive();
        const auto appBackwardStart = emitLayerDiagnostics ? layerSubmitDiagnosticNow() : LayerSubmitDiagnosticTimePoint();
        uint64_t errorComputeMicros = 0;
        uint64_t recordBatchMicros = 0;
        uint64_t upstreamMicros = 0;
        uint64_t upstreamCount = 0;

        app.backwardRanThisPass = true;

        std::optional<Event> errorOutHasBeenComputedEvent = std::nullopt;
        bool genericSharedWroteParameterGradient = false;
        if (useNativeSharedBackward) {
            const auto errorComputeStart = emitLayerDiagnostics ? layerSubmitDiagnosticNow() : LayerSubmitDiagnosticTimePoint();
            variant.nativeSharedBackward->run();
            computeStream(applicationIndex).putEvent(app.backwardErrorReadyEvent);
            errorOutHasBeenComputedEvent = app.backwardErrorReadyEvent;
            if (emitLayerDiagnostics) {
                errorComputeMicros = layerSubmitDiagnosticElapsedMicros(errorComputeStart, layerSubmitDiagnosticNow());
            }
        } else if (useGenericSharedBackward) {
            const bool runClearSharedPlan = clearGradientFirstThisBackwardPass;
            std::shared_ptr<StampedExecutionPlan> sharedPlan =
                runClearSharedPlan ? variant.genericSharedBackwardClear
                                   : variant.genericSharedBackwardAccumulate;

            // The first contribution must establish every materialized parameter
            // gradient, including explicit zeros for inactive parameters. Later
            // contributions may legitimately be dInput-only when this application
            // has no active parameter target.
            if (runClearSharedPlan && sharedPlan == nullptr) {
                throw runtime_error("CustomLayer first generic shared backward contribution has no clear plan.");
            }

            genericSharedWroteParameterGradient =
                runClearSharedPlan || !variant.genericSharedBackwardAccumulateWrtNames.empty();

            if (sharedPlan != nullptr) {
                Stream& sharedStream = computeStream(applicationIndex);

                // Parameter-gradient buffers are shared by every application of
                // this layer. Serialize only plans that write those buffers.
                // This is an explicit stream/event dependency; no fake arithmetic
                // edge is introduced into the VJP.
                if (genericSharedWroteParameterGradient &&
                    lastParameterGradientContributionApplicationThisPass.has_value()) {
                    const uint32_t previousApplication =
                        lastParameterGradientContributionApplicationThisPass.value();
                    THOR_THROW_IF_FALSE(previousApplication < applications.size());
                    sharedStream.waitEvent(applications[previousApplication].backwardErrorReadyEvent);
                }

                const auto errorComputeStart =
                    emitLayerDiagnostics ? layerSubmitDiagnosticNow() : LayerSubmitDiagnosticTimePoint();
                if (!runClearSharedPlan && !variant.optimizerUpdateFusedParameterNames.empty()) {
                    throw runtime_error(
                        "CustomLayer expression-fused optimizer update cannot execute on an accumulate contribution.");
                }
#ifdef THOR_DEBUG
                if (runClearSharedPlan) {
                    ++variant.genericSharedBackwardClearExecutionCount;
                } else {
                    ++variant.genericSharedBackwardAccumulateExecutionCount;
                }
#endif
                if (runClearSharedPlan && !variant.optimizerUpdateFusedParameterNames.empty()) {
                    sharedPlan->run(updateFusedOptimizerRuntimeScalars(applicationIndex, variantId, batchSize));
                } else {
                    sharedPlan->run();
                }
                // One recurring completion event is sufficient for both consumers:
                // upstream dInput readers and the next application's serialized dW
                // contribution. StampedExecutionPlan::run() has already joined any
                // internal helper lanes back to this stream before the event is recorded.
                sharedStream.putEvent(app.backwardErrorReadyEvent);
                errorOutHasBeenComputedEvent = app.backwardErrorReadyEvent;

                if (genericSharedWroteParameterGradient) {
                    lastParameterGradientContributionApplicationThisPass = applicationIndex;
                }

                if (emitLayerDiagnostics) {
                    errorComputeMicros =
                        layerSubmitDiagnosticElapsedMicros(errorComputeStart, layerSubmitDiagnosticNow());
                }
            }
        } else if (variant.backwardError != nullptr) {
            const auto errorComputeStart = emitLayerDiagnostics ? layerSubmitDiagnosticNow() : LayerSubmitDiagnosticTimePoint();
            errorOutHasBeenComputedEvent = computeErrorOut(inputFlatIndex(applicationIndex, 0));
            if (emitLayerDiagnostics) {
                errorComputeMicros = layerSubmitDiagnosticElapsedMicros(errorComputeStart, layerSubmitDiagnosticNow());
            }
        }

        if (errorOutHasBeenComputedEvent.has_value()) {
            errorOutHasBeenComputedEvents.push_back(errorOutHasBeenComputedEvent.value());

            // The backward expression runs on this application's primary compute stream, but a
            // multi-input layer can have a different upstream stream for every input port.
            // Publish the produced input gradients to those connection streams before recursively
            // invoking the upstream layers.  For native Attention this same event also marks all
            // parameter gradients ready; the optimizer stream joins it before applying updates.
            Stream& backwardComputeStream = computeStream(applicationIndex);
            for (uint32_t inputPort = 0; inputPort < inputNames.size(); ++inputPort) {
                const uint32_t inputFlat = inputFlatIndex(applicationIndex, inputPort);
                if (inputFlat >= streams.size() || !previousLayers[inputFlat].has_value() ||
                    !errorOutputs[inputFlat].has_value()) {
                    continue;
                }
                Stream& upstreamStream = streams[inputFlat];
                if (upstreamStream != backwardComputeStream) {
                    upstreamStream.waitEvent(errorOutHasBeenComputedEvent.value());
                }
            }
        }

        const auto recordBatchStart = emitLayerDiagnostics ? layerSubmitDiagnosticNow() : LayerSubmitDiagnosticTimePoint();
        recordEffectiveParameterBatchSizeForApplication(applicationIndex, batchSize);
        if (emitLayerDiagnostics) {
            recordBatchMicros = layerSubmitDiagnosticElapsedMicros(recordBatchStart, layerSubmitDiagnosticNow());
        }
        clearGradientFirstThisBackwardPass = false;

        const auto upstreamStart = emitLayerDiagnostics ? layerSubmitDiagnosticNow() : LayerSubmitDiagnosticTimePoint();
        for (uint32_t inputPort = 0; inputPort < inputNames.size(); ++inputPort) {
            const uint32_t inputFlat = inputFlatIndex(applicationIndex, inputPort);
            if (!previousLayers[inputFlat].has_value() || !errorOutputs[inputFlat].has_value()) {
                continue;
            }
            upstreamCount += 1;
            previousLayers[inputFlat].value()->backward(errorOutputs[inputFlat], batchSize);
        }
        if (emitLayerDiagnostics) {
            upstreamMicros = layerSubmitDiagnosticElapsedMicros(upstreamStart, layerSubmitDiagnosticNow());
        }

        numBackwardApplicationsCompletedThisPass += 1;
        if (emitLayerDiagnostics) {
            emitLayerSubmitDiagnostic("custom_backward_application",
                                      diagnosticLabel(),
                                      getId(),
                                      layerSubmitDiagnosticElapsedMicros(appBackwardStart, layerSubmitDiagnosticNow()),
                                      {{"app", applicationIndex},
                                       {"error_compute_us", errorComputeMicros},
                                       {"record_batch_us", recordBatchMicros},
                                       {"upstream_us", upstreamMicros},
                                       {"upstream_count", upstreamCount}});
        }
    }

    if (numBackwardApplications > 0 && numBackwardApplicationsCompletedThisPass == numBackwardApplications) {
        if (trainingUpdateDiagnosticsEnabled()) {
            std::fprintf(stderr,
                         "THOR_TRAINING_UPDATE_DIAGNOSTIC layer=%s backward_pass_complete effective_batches=%s gradient_update_stream=%d\n",
                         diagnosticLabel().c_str(),
                         joinNames(effectiveBatchSizeByParameterName).c_str(),
                         gradientUpdateStream.has_value() ? 1 : 0);
        }
        numBackwardApplicationsCompletedThisPass = 0;
        weightsAreUpToDateEventValid = false;

        std::unordered_set<Loss*> fusedCustomLossOwners;
        for (const ApplicationState& app : applications) {
            if (!app.backwardRanThisPass) {
                continue;
            }
            for (const auto& [outputName, fusedLossGradient] : app.fusedCustomLossGradientsByOutput) {
                (void)outputName;
                if (fusedLossGradient.ownerLoss != nullptr) {
                    fusedCustomLossOwners.insert(fusedLossGradient.ownerLoss);
                }
            }
        }
        auto notifyFusedCustomLossOwners = [&](const Event& consumersDone) {
            for (Loss* ownerLoss : fusedCustomLossOwners) {
                ownerLoss->notifyFusedGradientConsumptionComplete(consumersDone);
            }
        };

        if (gradientUpdateStream.has_value()) {
            const bool emitApplyDiagnostics = layerSubmitDiagnosticsActive();
            const auto waitErrorOutputsStart = emitApplyDiagnostics ? layerSubmitDiagnosticNow() : LayerSubmitDiagnosticTimePoint();
            for (const Event& eOutComputedEvent : errorOutHasBeenComputedEvents) {
                gradientUpdateStream.value().waitEvent(eOutComputedEvent);
            }
            const uint64_t waitErrorOutputsMicros =
                emitApplyDiagnostics ? layerSubmitDiagnosticElapsedMicros(waitErrorOutputsStart, layerSubmitDiagnosticNow()) : 0;

            // Native and generic shared backward may produce dX and dW together
            // on an application compute stream. The optimizer stream joins every
            // local backward-completion event before applying materialized gradients.
            // This stream has now joined every local backward-completion event, so
            // an event recorded here is the local last-use point for any labels or
            // batch-validity-mask tensors captured by a fused CustomLoss gradient.
            // Return that dependency to the owning loss before the next batch can
            // enqueue reuse of those tensors on its labels/loss streams.
            if (!fusedCustomLossOwners.empty()) {
                gradientUpdateStream.value().putEvent(fusedLossConsumersDoneEvent);
                notifyFusedCustomLossOwners(fusedLossConsumersDoneEvent);
            }


            const auto applyTotalStart = emitApplyDiagnostics ? layerSubmitDiagnosticNow() : LayerSubmitDiagnosticTimePoint();
            uint64_t fusedParameterScanMicros = 0;
            uint64_t parameterApplyMicros = 0;
            uint64_t constraintApplyMicros = 0;
            uint64_t skippedParameters = 0;
            uint64_t appliedParameters = 0;
            uint64_t constrainedParameters = 0;
            uint64_t fusedSkippedParameters = 0;
            bool anyWeightsUpdated = false;
            std::set<std::string> fusedUpdateParameterNames;
            const auto fusedParameterScanStart = emitApplyDiagnostics ? layerSubmitDiagnosticNow() : LayerSubmitDiagnosticTimePoint();
            for (const ApplicationState& app : applications) {
                if (!app.backwardRanThisPass || !app.forwardVariantThisPass.has_value()) {
                    continue;
                }
                const StampedExecutionVariant& variant =
                    app.stampedVariants.at(app.forwardVariantThisPass.value());
                if (!variant.optimizerUpdateFusedParameterNames.empty()) {
                    anyWeightsUpdated = true;
                    fusedUpdateParameterNames.insert(variant.optimizerUpdateFusedParameterNames.begin(),
                                                     variant.optimizerUpdateFusedParameterNames.end());
                }
            }
            if (emitApplyDiagnostics) {
                fusedParameterScanMicros = layerSubmitDiagnosticElapsedMicros(fusedParameterScanStart, layerSubmitDiagnosticNow());
            }
            for (const auto& parameter : parameters) {
                if (!parameter->isTrainingEnabled()) {
                    if (trainingUpdateDiagnosticsEnabled()) {
                        std::fprintf(stderr,
                                     "THOR_TRAINING_UPDATE_DIAGNOSTIC layer=%s parameter=%s apply_skip reason=training_disabled\n",
                                     diagnosticLabel().c_str(),
                                     parameter->getName().c_str());
                    }
                    skippedParameters += 1;
                    continue;
                }

                if (fusedUpdateParameterNames.contains(parameter->getName())) {
                    if (!parameter->hasConstraints() || parameter->supportsDenseExpressionConstraintFusion()) {
                        if (trainingUpdateDiagnosticsEnabled()) {
                            std::fprintf(stderr,
                                         "THOR_TRAINING_UPDATE_DIAGNOSTIC layer=%s parameter=%s apply_skip reason=fused_optimizer_update_constraints_fused\n",
                                         diagnosticLabel().c_str(),
                                         parameter->getName().c_str());
                        }
                        fusedSkippedParameters += 1;
                        continue;
                    }

                    if (trainingUpdateDiagnosticsEnabled()) {
                        std::fprintf(stderr,
                                     "THOR_TRAINING_UPDATE_DIAGNOSTIC layer=%s parameter=%s apply_post_update_constraints reason=fused_optimizer_update_unfused_constraint\n",
                                     diagnosticLabel().c_str(),
                                     parameter->getName().c_str());
                    }
                    const auto constraintStart = emitApplyDiagnostics ? layerSubmitDiagnosticNow() : LayerSubmitDiagnosticTimePoint();
                    parameter->applyConstraintsAfterExternalUpdate();
                    if (emitApplyDiagnostics) {
                        constraintApplyMicros += layerSubmitDiagnosticElapsedMicros(constraintStart, layerSubmitDiagnosticNow());
                    }
                    constrainedParameters += 1;
                    continue;
                }

                const auto effectiveBatchSizeIt = effectiveBatchSizeByParameterName.find(parameter->getName());
                if (effectiveBatchSizeIt == effectiveBatchSizeByParameterName.end() || effectiveBatchSizeIt->second == 0) {
                    if (trainingUpdateDiagnosticsEnabled()) {
                        std::fprintf(stderr,
                                     "THOR_TRAINING_UPDATE_DIAGNOSTIC layer=%s parameter=%s apply_skip reason=no_effective_batch effective_batches=%s\n",
                                     diagnosticLabel().c_str(),
                                     parameter->getName().c_str(),
                                     joinNames(effectiveBatchSizeByParameterName).c_str());
                    }
                    skippedParameters += 1;
                    continue;
                }

                if (effectiveBatchSizeIt->second > std::numeric_limits<uint32_t>::max()) {
                    throw runtime_error("CustomLayer effective parameter batch size exceeds uint32_t range for parameter " +
                                        parameter->getName() + ".");
                }

                if (trainingUpdateDiagnosticsEnabled()) {
                    std::fprintf(stderr,
                                 "THOR_TRAINING_UPDATE_DIAGNOSTIC layer=%s parameter=%s apply_gradient batch=%llu\n",
                                 diagnosticLabel().c_str(),
                                 parameter->getName().c_str(),
                                 static_cast<unsigned long long>(effectiveBatchSizeIt->second));
                }
                const auto parameterApplyStart = emitApplyDiagnostics ? layerSubmitDiagnosticNow() : LayerSubmitDiagnosticTimePoint();
                anyWeightsUpdated |= parameter->applyGradient(static_cast<uint32_t>(effectiveBatchSizeIt->second));
                if (emitApplyDiagnostics) {
                    parameterApplyMicros += layerSubmitDiagnosticElapsedMicros(parameterApplyStart, layerSubmitDiagnosticNow());
                }
                appliedParameters += 1;
            }
            if (emitApplyDiagnostics) {
                emitLayerSubmitDiagnostic("custom_apply_gradients",
                                          diagnosticLabel(),
                                          getId(),
                                          layerSubmitDiagnosticElapsedMicros(applyTotalStart, layerSubmitDiagnosticNow()),
                                          {{"wait_error_outputs_us", waitErrorOutputsMicros},
                                           {"fused_scan_us", fusedParameterScanMicros},
                                           {"parameter_apply_us", parameterApplyMicros},
                                           {"constraint_apply_us", constraintApplyMicros},
                                           {"parameters", parameters.size()},
                                           {"applied", appliedParameters},
                                           {"skipped", skippedParameters},
                                           {"fused_skipped", fusedSkippedParameters},
                                           {"constrained", constrainedParameters},
                                           {"any_updated", anyWeightsUpdated ? 1UL : 0UL}});
            }
            effectiveBatchSizeByParameterName.clear();
            if (trainingUpdateDiagnosticsEnabled()) {
                std::fprintf(stderr,
                             "THOR_TRAINING_UPDATE_DIAGNOSTIC layer=%s update_result any_weights_updated=%d\n",
                             diagnosticLabel().c_str(),
                             anyWeightsUpdated ? 1 : 0);
            }
            if (anyWeightsUpdated) {
                gradientUpdateStream.value().putEvent(weightsAreUpToDateEvent);
                weightsAreUpToDateEventValid = true;
            }
        } else {
            // A parameterless CustomLayer has no gradient-update stream to serve
            // as the join point.  In that case the only fused-gradient readers are
            // the backward-error plans themselves, so make the owning loss wait
            // on each of their completion events directly.
            if (!fusedCustomLossOwners.empty()) {
                for (const Event& eOutComputedEvent : errorOutHasBeenComputedEvents) {
                    notifyFusedCustomLossOwners(eOutComputedEvent);
                }
            }
        }
        errorOutHasBeenComputedEvents.clear();
        lastParameterGradientContributionApplicationThisPass.reset();
        isStartOfForward = true;
    }
}

void CustomLayer::computeFeatureOut(uint32_t connectionNumber) {
    computeFeatureOutForPass(connectionNumber, false);
}

void CustomLayer::computeFeatureOutForPass(uint32_t connectionNumber, bool validationPass) {
    const bool emitDiagnostics = layerSubmitDiagnosticsActive();
    const auto totalStart = emitDiagnostics ? layerSubmitDiagnosticNow() : LayerSubmitDiagnosticTimePoint();
    uint64_t preRunHookMicros = 0;
    uint64_t runMicros = 0;
    DecodedConnection decoded = decodeInputConnectionType(static_cast<int>(connectionNumber));
    if (decoded.applicationIndex >= applications.size()) {
        throw runtime_error("CustomLayer::computeFeatureOut requires a stamped forward plan.");
    }

    ApplicationState& application = applications[decoded.applicationIndex];
    DynamicExpressionVariantId variantId = activeTrainingVariantId;
    const bool useEvaluationVariant =
        (validationPass || isInferenceOnly()) && application.evaluationVariantId.has_value();
    if (useEvaluationVariant) {
        variantId = application.evaluationVariantId.value();
    }
    StampedExecutionVariant& variant = stampedVariant(decoded.applicationIndex, variantId);
    if (variant.forward == nullptr) {
        throw runtime_error("CustomLayer::computeFeatureOut requires a stamped forward plan for execution variant " +
                            std::to_string(variantId) + ".");
    }
    application.forwardVariantThisPass = variantId;
    const std::function<void(Stream&)>& preRunHook = variant.preForwardHook;
    StampedExecutionPlan& executionPlan = *variant.forward;

    if (preRunHook) {
        const auto preRunStart = emitDiagnostics ? layerSubmitDiagnosticNow() : LayerSubmitDiagnosticTimePoint();
        preRunHook(computeStream(decoded.applicationIndex));
        if (emitDiagnostics) {
            preRunHookMicros = layerSubmitDiagnosticElapsedMicros(preRunStart, layerSubmitDiagnosticNow());
        }
    }
    const auto runStart = emitDiagnostics ? layerSubmitDiagnosticNow() : LayerSubmitDiagnosticTimePoint();
    executionPlan.run();
    if (emitDiagnostics) {
        runMicros = layerSubmitDiagnosticElapsedMicros(runStart, layerSubmitDiagnosticNow());
        emitLayerSubmitDiagnostic("custom_forward_compute",
                                  diagnosticLabel(),
                                  getId(),
                                  layerSubmitDiagnosticElapsedMicros(totalStart, layerSubmitDiagnosticNow()),
                                  {{"app", decoded.applicationIndex},
                                   {"connection", connectionNumber},
                                   {"variant", variantId},
                                   {"evaluation_variant", useEvaluationVariant ? 1UL : 0UL},
                                   {"prerun_us", preRunHookMicros},
                                   {"run_us", runMicros},
                                   {"has_prerun", preRunHook ? 1UL : 0UL},
                                   {"flops", bestEffortExecutionPlanFlopCount(executionPlan)}});
    }
}

std::optional<Event> CustomLayer::computeErrorOut(uint32_t connectionNumber) {
    const bool emitDiagnostics = layerSubmitDiagnosticsActive();
    const auto totalStart = emitDiagnostics ? layerSubmitDiagnosticNow() : LayerSubmitDiagnosticTimePoint();
    uint64_t runMicros = 0;
    uint64_t eventMicros = 0;
    DecodedConnection decoded = decodeInputConnectionType(static_cast<int>(connectionNumber));
    if (decoded.applicationIndex >= applications.size()) {
        return std::nullopt;
    }
    StampedExecutionVariant& variant = backwardVariantForApplication(decoded.applicationIndex);
    if (variant.backwardError == nullptr) {
        if (emitDiagnostics) {
            emitLayerSubmitDiagnostic("custom_backward_error_skip",
                                      diagnosticLabel(),
                                      getId(),
                                      layerSubmitDiagnosticElapsedMicros(totalStart, layerSubmitDiagnosticNow()),
                                      {{"app", decoded.applicationIndex}, {"connection", connectionNumber}, {"reason_no_stamp", 1}});
        }
        return std::nullopt;
    }
    const auto runStart = emitDiagnostics ? layerSubmitDiagnosticNow() : LayerSubmitDiagnosticTimePoint();
    variant.backwardError->run();
    if (emitDiagnostics) {
        runMicros = layerSubmitDiagnosticElapsedMicros(runStart, layerSubmitDiagnosticNow());
    }
    const auto eventStart = emitDiagnostics ? layerSubmitDiagnosticNow() : LayerSubmitDiagnosticTimePoint();
    ApplicationState& app = applications[decoded.applicationIndex];
    computeStream(decoded.applicationIndex).putEvent(app.backwardErrorReadyEvent);
    if (emitDiagnostics) {
        eventMicros = layerSubmitDiagnosticElapsedMicros(eventStart, layerSubmitDiagnosticNow());
        emitLayerSubmitDiagnostic("custom_backward_error_compute",
                                  diagnosticLabel(),
                                  getId(),
                                  layerSubmitDiagnosticElapsedMicros(totalStart, layerSubmitDiagnosticNow()),
                                  {{"app", decoded.applicationIndex},
                                   {"connection", connectionNumber},
                                   {"run_us", runMicros},
                                   {"put_event_us", eventMicros},
                                   {"flops", bestEffortExecutionPlanFlopCount(*variant.backwardError)}});
    }
    return app.backwardErrorReadyEvent;
}

void CustomLayer::cleanup() {
    for (ApplicationState& app : applications) {
        for (Event& event : app.forwardInputReadyEvents) {
            event = Event();
        }
        app.backwardErrorReadyEvent = Event();
    }
    fusedLossConsumersDoneEvent = Event();
    lastParameterGradientContributionApplicationThisPass.reset();
    TrainableLayer::cleanup();
}

uint64_t CustomLayer::flopCountForward() {
    uint64_t flops = 0;
    for (const ApplicationState& app : applications) {
        auto it = app.stampedVariants.find(kPrimaryDynamicExpressionVariant);
        if (it != app.stampedVariants.end() && it->second.forward != nullptr) {
            flops += it->second.forward->flopCount();
        }
    }
    return flops;
}

uint64_t CustomLayer::flopCountBackward() {
    uint64_t flops = 0;
    bool genericClearContributionAccounted = false;

    // Report the cost of one whole layer backward pass. Generic shared ownership
    // executes exactly one clear contribution and then accumulate contributions
    // for later applications. The static estimate uses application-index order to
    // choose which primary-variant plan is the clear contribution; runtime arrival
    // order can differ, but we no longer count an accumulate plan for every app or
    // report an unreachable single-application accumulate plan.
    for (const ApplicationState& app : applications) {
        auto it = app.stampedVariants.find(kPrimaryDynamicExpressionVariant);
        if (it == app.stampedVariants.end()) {
            continue;
        }
        const StampedExecutionVariant& variant = it->second;
        if (variant.nativeSharedBackward != nullptr) {
            flops += variant.nativeSharedBackward->flopCount();
            continue;
        }
        if (variant.genericSharedBackwardClear != nullptr) {
            if (!genericClearContributionAccounted) {
                flops += variant.genericSharedBackwardClear->flopCount();
                genericClearContributionAccounted = true;
            } else if (variant.genericSharedBackwardAccumulate != nullptr) {
                flops += variant.genericSharedBackwardAccumulate->flopCount();
            } else {
                // A second generic shared contribution without an accumulate plan
                // would also be invalid at runtime. Keep the estimator conservative
                // rather than silently dropping all of that application's work.
                flops += variant.genericSharedBackwardClear->flopCount();
            }
            continue;
        }
        if (variant.backwardError != nullptr) {
            flops += variant.backwardError->flopCount();
        }
    }
    return flops;
}

uint64_t CustomLayer::batchSizeForFlopEstimate() const {
    if (fixedBatchCapacity.has_value()) {
        return fixedBatchCapacity.value();
    }
    auto batchFromTensor = [](const Tensor& tensor) -> uint64_t {
        std::vector<uint64_t> dimensions = tensor.getDescriptor().getDimensions();
        if (!dimensions.empty() && dimensions[0] > 0) {
            return dimensions[0];
        }
        return 0;
    };

    for (const ApplicationState& app : applications) {
        for (const auto& nameAndTensor : app.forwardOutputsByName) {
            const uint64_t batchSize = batchFromTensor(nameAndTensor.second);
            if (batchSize > 0) {
                return batchSize;
            }
        }
        for (const auto& nameAndTensor : app.forwardInputsByName) {
            const uint64_t batchSize = batchFromTensor(nameAndTensor.second);
            if (batchSize > 0) {
                return batchSize;
            }
        }
    }
    return 1;
}

uint64_t CustomLayer::floatingPointOperationsPerExampleForward() {
    const uint64_t batchSize = batchSizeForFlopEstimate();
    return batchSize == 0 ? 0 : flopCountForward() / batchSize;
}

uint64_t CustomLayer::floatingPointOperationsPerExampleBackward() {
    const uint64_t batchSize = batchSizeForFlopEstimate();
    return batchSize == 0 ? 0 : flopCountBackward() / batchSize;
}

bool CustomLayer::hasTrainableParameterRequiringDownstreamError() {
    if (isInferenceOnly()) {
        return false;
    }

    for (const auto& parameter : parameters) {
        if (parameter != nullptr && parameter->isTrainingEnabled()) {
            return true;
        }
    }
    return false;
}

bool CustomLayer::hasConnectedUpstreamErrorOutput() const {
    for (const auto& errorOutput : errorOutputs) {
        if (errorOutput.has_value()) {
            return true;
        }
    }
    return false;
}

bool CustomLayer::isBackPropStub() {
    // isBackPropStub() answers the connection-level question: should the next
    // layer send this layer a downstream error tensor?  A first trainable layer
    // may not send an input-gradient tensor farther upstream, but it still needs
    // the downstream error tensor to compute parameter gradients.
    if (hasTrainableParameterRequiringDownstreamError()) {
        return false;
    }

    return !hasConnectedUpstreamErrorOutput();
}

}  // namespace ThorImplementation
