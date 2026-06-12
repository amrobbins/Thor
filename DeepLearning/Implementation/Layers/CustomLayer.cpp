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
#include "Utilities/Expression/AutoDiff.h"
using namespace std;

namespace ThorImplementation {

namespace {
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

bool trainingUpdateDiagnosticsEnabled() {
    const char* value = std::getenv("THOR_TRAINING_UPDATE_DIAGNOSTICS");
    return value != nullptr && value[0] != '\0' && std::string(value) != "0";
}

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
            throw runtime_error("CustomLayer fused optimizer update missing required tensor input: " + input.name);
        }
        filteredInputs.emplace(input.name, it->second);
    }
    return filteredInputs;
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
    : TrainableLayer(placement, inferenceOnly, stampedId),
      layerDefinitionExpression(std::move(expr)),
      inputNames(std::move(inputNames)),
      outputNames(std::move(outputNames)) {
    validatePortNames(this->inputNames, "input");
    validatePortNames(this->outputNames, "output");

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

    attachGradientUpdateStream();
}

void CustomLayer::validatePortNames(const std::vector<std::string>& names, const std::string& what) {
    if (names.empty()) {
        throw runtime_error("CustomLayer requires at least one " + what + " port.");
    }

    std::set<std::string> seen;
    for (const std::string& name : names) {
        if (name.empty()) {
            throw runtime_error("CustomLayer " + what + " port name cannot be empty.");
        }
        if (name.length() >= 2 && name[0] == '_' && name[1] == '_') {
            throw runtime_error("CustomLayer " + what + " port names cannot start with __ that is reserved. Name " + name + " is illegal.");
        }
        if (!seen.insert(name).second) {
            throw runtime_error("Duplicate CustomLayer " + what + " port name: " + name);
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

void CustomLayer::recordEffectiveParameterBatchSizeForApplication(uint32_t applicationIndex, uint32_t batchSize) {
    if (applicationIndex >= applications.size()) {
        return;
    }

    const ApplicationState& app = applications[applicationIndex];
    if (trainingUpdateDiagnosticsEnabled()) {
        std::fprintf(stderr,
                     "THOR_TRAINING_UPDATE_DIAGNOSTIC layer=%s app=%u record_effective_batch batch=%u active_parameters=%s fused_parameters=%s before=%s\n",
                     getName().c_str(),
                     applicationIndex,
                     batchSize,
                     joinNames(app.activeParameterTargetNames).c_str(),
                     joinNames(app.optimizerUpdateFusedParameterNames).c_str(),
                     joinNames(effectiveBatchSizeByParameterName).c_str());
    }
    for (const std::string& parameterName : app.activeParameterTargetNames) {
        if (app.optimizerUpdateFusedParameterNames.contains(parameterName)) {
            continue;
        }
        effectiveBatchSizeByParameterName[parameterName] += batchSize;
    }
    if (trainingUpdateDiagnosticsEnabled()) {
        std::fprintf(stderr,
                     "THOR_TRAINING_UPDATE_DIAGNOSTIC layer=%s app=%u record_effective_batch after=%s\n",
                     getName().c_str(),
                     applicationIndex,
                     joinNames(effectiveBatchSizeByParameterName).c_str());
    }
}

bool CustomLayer::canFuseOptimizerUpdatesForApplication(uint32_t applicationIndex) const {
    // Dense optimizer-update fusion is only correct for the simple single-application,
    // single-input CustomLayer surface.  Multi-application and multi-input layers share
    // materialized gradient buffers across applications/ports and need the explicit
    // overwrite-then-accumulate path below so effective batch-size accounting and shared
    // parameter accumulation stay well-defined.
    return applicationIndex == 0 && applications.size() == 1 && inputNames.size() == 1;
}

bool CustomLayer::applicationHasFusedCustomLossGradient(uint32_t applicationIndex) const {
    return applicationIndex < applications.size() && !applications[applicationIndex].fusedCustomLossGradientsByOutput.empty();
}

uint32_t CustomLayer::getNumFusedCustomLossGradients() const {
    return static_cast<uint32_t>(fusedCustomLossGradientByOutputFlatIndex.size());
}

bool CustomLayer::registerFusedCustomLossGradient(const Tensor& predictions,
                                                 const Tensor& labels,
                                                 DynamicExpression gradientExpression,
                                                 std::string predictionsName,
                                                 std::string labelsName,
                                                 std::string gradientName) {
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

    FusedCustomLossGradient fused{predictions,
                                  labels,
                                  std::move(gradientExpression),
                                  std::move(predictionsName),
                                  std::move(labelsName),
                                  std::move(gradientName),
                                  customLossFusedLabelsInputName(matchedFlatIndex.value())};
    fusedCustomLossGradientByOutputFlatIndex.emplace(matchedFlatIndex.value(), std::move(fused));
    return true;
}

PhysicalOutputs CustomLayer::buildBackwardOutputsForApplication(uint32_t applicationIndex,
                                                                const std::vector<std::string>& wrtNames,
                                                                bool accumulateGradOutputs) {
    ApplicationState& app = applications[applicationIndex];

    PreparedDynamicExpression::ShapeMap forwardInputDims;
    for (const auto& [name, tensor] : app.forwardPrepared->stampInputs()) {
        forwardInputDims[name] = tensor.getDimensions();
    }

    if (app.fusedCustomLossGradientsByOutput.empty()) {
        return buildBackwardOutputs(app.forwardPrepared->equation().physicalOutputs(),
                                    wrtNames,
                                    app.upstreamInputNamesByOutput,
                                    forwardInputDims,
                                    accumulateGradOutputs);
    }

    const PhysicalOutputs& forwardOutputs = app.forwardPrepared->equation().physicalOutputs();
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

    PhysicalOutputs seededBackwardOutputs = buildBackwardOutputs(forwardOutputs,
                                                                 wrtNames,
                                                                 upstreamInputNamesByOutput,
                                                                 forwardInputDims,
                                                                 accumulateGradOutputs);
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

        const uint32_t forwardCudaKernelExpressionOffset = static_cast<uint32_t>(fusedExpr.cuda_kernel_expressions.size());
        fusedExpr.cuda_kernel_expressions.insert(fusedExpr.cuda_kernel_expressions.end(),
                                                forwardOutputs.expr->cuda_kernel_expressions.begin(),
                                                forwardOutputs.expr->cuda_kernel_expressions.end());

        std::unordered_map<std::string, uint32_t> forwardInputReplacements;
        for (const NamedInput& input : forwardOutputs.expr->inputs) {
            auto inputNodeIt = fusedExprInputNodeByName.find(input.name);
            if (inputNodeIt == fusedExprInputNodeByName.end()) {
                throw runtime_error("CustomLayer fused CustomLoss backward missing forward input clone for: " + input.name);
            }
            forwardInputReplacements.emplace(input.name, inputNodeIt->second);
        }

        std::unordered_map<uint32_t, uint32_t> clonedForwardNodes;
        const uint32_t predictionNode = cloneExpressionNodeWithInputReplacements(*forwardOutputs.expr,
                                                                                 forwardOutputNode.value(),
                                                                                 fusedExpr,
                                                                                 forwardInputReplacements,
                                                                                 inputBySlot(*forwardOutputs.expr),
                                                                                 forwardCudaKernelExpressionOffset,
                                                                                 clonedForwardNodes);

        const uint32_t labelsInputNode = appendTensorInputNode(fusedExpr, fused.fusedLabelsInputName);

        const uint32_t gradientCudaKernelExpressionOffset = static_cast<uint32_t>(fusedExpr.cuda_kernel_expressions.size());
        fusedExpr.cuda_kernel_expressions.insert(fusedExpr.cuda_kernel_expressions.end(),
                                                gradientOutputs.expr->cuda_kernel_expressions.begin(),
                                                gradientOutputs.expr->cuda_kernel_expressions.end());

        std::unordered_map<std::string, uint32_t> gradientInputReplacements{
            {fused.predictionsName, predictionNode},
            {fused.labelsName, labelsInputNode},
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
        fusedBackwardOutputs.outputs.push_back(NamedOutput{output.name, terminalOutputNode});
    }

    return fusedBackwardOutputs;
}

std::shared_ptr<StampedExecutionPlan> CustomLayer::stampBackwardForApplication(
    uint32_t applicationIndex,
    const std::vector<std::string>& wrtNames,
    bool accumulateGradOutputs,
    const PreparedDynamicExpression::TensorMap& preallocatedGradOutputs,
    Stream& runStream) {
    ApplicationState& app = applications[applicationIndex];
    if (wrtNames.empty()) {
        return nullptr;
    }

    if (!applicationHasFusedCustomLossGradient(applicationIndex)) {
        return std::make_shared<StampedExecutionPlan>(app.forwardPrepared->stampBackward(wrtNames,
                                                                                         app.upstreamInputNamesByOutput,
                                                                                         accumulateGradOutputs,
                                                                                         app.backwardAdditionalInputsByName,
                                                                                         {},
                                                                                         preallocatedGradOutputs));
    }

    PhysicalOutputs backwardOutputs = buildBackwardOutputsForApplication(applicationIndex, wrtNames, accumulateGradOutputs);
    FusedEquation backwardEquation = FusedEquation::compile(backwardOutputs, placement.getDeviceNum());

    PreparedDynamicExpression::TensorMap stampInputs = app.forwardInputsByName;
    for (const auto& [name, tensor] : app.backwardAdditionalInputsByName) {
        stampInputs[name] = tensor;
    }
    if (accumulateGradOutputs) {
        // The custom fused-loss path builds the backward graph directly instead of going through
        // PreparedDynamicExpression::stampBackward(), so the FusedEquation does not carry its
        // usual BackwardEquationConfig metadata.  AutoDiff still represents accumulation as
        // `wrt_grad = wrt_grad + newly_computed_grad`, which means the existing gradient buffer
        // is a real tensor input as well as the preallocated output.  Bind those tensors explicitly
        // for this custom path.
        for (const auto& [name, tensor] : preallocatedGradOutputs) {
            stampInputs[name] = tensor;
        }
    }

    return std::make_shared<StampedExecutionPlan>(
        backwardEquation.stamp(stampInputs, runStream, app.forwardPrepared->tensorScalarInputs(), preallocatedGradOutputs));
}

std::shared_ptr<StampedExecutionPlan> CustomLayer::buildFusedOptimizerUpdatePlan(
    uint32_t applicationIndex,
    const std::vector<std::string>& fusedParameterTargets,
    const std::unordered_map<std::string, Tensor>& optimizerUpdateInputs) {
    if (fusedParameterTargets.empty()) {
        return nullptr;
    }
    if (!gradientUpdateStream.has_value()) {
        throw runtime_error("CustomLayer fused optimizer update requires a gradient update stream.");
    }

    const ApplicationState& app = applications[applicationIndex];

    PhysicalOutputs backwardOutputs = buildBackwardOutputsForApplication(applicationIndex, fusedParameterTargets, false);

    std::unordered_map<std::string, Expression> gradientsByParameter;
    for (const NamedOutput& output : backwardOutputs.outputs) {
        constexpr const char* suffix = "_grad";
        constexpr size_t suffixLen = 5;
        if (output.name.size() < suffixLen || output.name.compare(output.name.size() - suffixLen, suffixLen, suffix) != 0) {
            continue;
        }
        const std::string parameterName = output.name.substr(0, output.name.size() - suffixLen);
        gradientsByParameter.emplace(parameterName, Expression::fromPhysicalNode(backwardOutputs.expr, output.node_idx));
    }

    std::vector<std::pair<std::string, Expression>> fusedOutputs;
    std::unordered_map<std::string, Tensor> stampInputs = app.forwardInputsByName;
    for (const auto& [name, tensor] : app.backwardAdditionalInputsByName) {
        stampInputs[name] = tensor;
    }

    std::unordered_map<std::string, Tensor> preallocatedOutputs;

    for (const std::string& parameterName : fusedParameterTargets) {
        auto gradIt = gradientsByParameter.find(parameterName);
        if (gradIt == gradientsByParameter.end()) {
            throw runtime_error("CustomLayer could not build fused optimizer update: missing gradient expression for parameter '" +
                                parameterName + "'.");
        }
        auto storageIt = optimizerUpdateInputs.find(parameterName);
        if (storageIt == optimizerUpdateInputs.end()) {
            throw runtime_error("CustomLayer could not build fused optimizer update: missing storage for parameter '" + parameterName + "'.");
        }

        shared_ptr<Optimizer> optimizer;
        for (const auto& parameter : parameters) {
            if (parameter->getName() == parameterName) {
                optimizer = parameter->getOptimizer();
                break;
            }
        }
        if (optimizer == nullptr || !optimizer->supportsDenseUpdateFusion()) {
            throw runtime_error("CustomLayer fused optimizer update requested for unsupported parameter '" + parameterName + "'.");
        }

        const std::string prefix = optimizerFusionNamePrefix(parameterName);

        // Dense optimizer fusion consumes the parameter-gradient expression directly.  Do not
        // materialize the optimizer-owned dense gradient tensor for fused parameters; that memory
        // exists only for the legacy materialized optimizer path.
        THOR_THROW_IF_FALSE(!optimizer->getWeightsGradient().has_value());

        DenseOptimizerExpression updateExpression = optimizer->toDenseUpdateExpression(storageIt->second, gradIt->second, prefix);

        for (const auto& [name, tensor] : updateExpression.inputs) {
            auto [_, inserted] = stampInputs.emplace(name, tensor);
            if (!inserted) {
                throw runtime_error("CustomLayer fused optimizer update input name collision: " + name);
            }
        }

        for (const NamedOutput& output : updateExpression.outputs.outputs) {
            const std::string uniqueOutputName = optimizerFusionOutputName(parameterName, output.name);
            auto preallocIt = updateExpression.preallocatedOutputs.find(output.name);
            if (preallocIt == updateExpression.preallocatedOutputs.end()) {
                throw runtime_error("CustomLayer fused optimizer update missing preallocated output for '" + output.name +
                                    "' on parameter '" + parameterName + "'.");
            }
            fusedOutputs.emplace_back(uniqueOutputName, Expression::fromPhysicalNode(updateExpression.outputs.expr, output.node_idx));
            preallocatedOutputs.emplace(uniqueOutputName, preallocIt->second);
        }
    }

    Outputs outputs = Expression::outputs(fusedOutputs);
    PhysicalOutputs physicalOutputs = outputs.physicalOutputs();
    PreparedDynamicExpression::TensorMap filteredStampInputs = filterTensorInputsForPhysicalOutputs(stampInputs, physicalOutputs);
    FusedEquation fusedUpdateEquation = FusedEquation::compile(physicalOutputs, placement.getDeviceNum());
    return std::make_shared<StampedExecutionPlan>(
        fusedUpdateEquation.stamp(filteredStampInputs, gradientUpdateStream.value(), {}, preallocatedOutputs));
}

std::unordered_map<std::string, float> CustomLayer::buildFusedOptimizerRuntimeScalars(uint32_t applicationIndex, uint32_t batchSize) {
    if (batchSize == 0) {
        throw runtime_error("CustomLayer fused optimizer update requires a non-zero batch size.");
    }
    if (applicationIndex >= applications.size()) {
        return {};
    }

    const ApplicationState& app = applications[applicationIndex];
    std::unordered_map<std::string, float> runtimeScalars;
    for (const std::string& parameterName : app.optimizerUpdateFusedParameterNames) {
        shared_ptr<Optimizer> optimizer;
        for (const auto& parameter : parameters) {
            if (parameter->getName() == parameterName) {
                optimizer = parameter->getOptimizer();
                break;
            }
        }
        if (optimizer == nullptr) {
            throw runtime_error("CustomLayer fused optimizer update lost optimizer for parameter '" + parameterName + "'.");
        }

        const std::string prefix = optimizerFusionNamePrefix(parameterName);
        auto scalars = optimizer->denseUpdateRuntimeScalars(batchSize, prefix);
        for (const auto& [name, value] : scalars) {
            auto [_, inserted] = runtimeScalars.emplace(name, value);
            if (!inserted) {
                throw runtime_error("CustomLayer fused optimizer runtime scalar name collision: " + name);
            }
        }
    }
    return runtimeScalars;
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

PreparedDynamicExpression::TensorMap CustomLayer::buildForwardInputs(uint32_t applicationIndex) {
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
        if (param->isTrainable()) {
            // Must compile optimizer every time to properly toggle parameter trainability.
            // For the single-application CustomLayer dense-fusion path, the optimizer consumes the
            // parameter gradient expression directly, so the optimizer-owned dense gradient tensor
            // would be dead storage.  Compile the optimizer state without that dense gradient buffer.
            bool materializeDenseGradient = true;
            if (canFuseOptimizerUpdatesForApplication(applicationIndex) && param->hasOptimizer() &&
                param->getOptimizer()->supportsDenseUpdateFusion()) {
                materializeDenseGradient = false;
            }
            param->compileOptimizer(gradientUpdateStream, isInferenceOnly(), materializeDenseGradient);
        }

        std::optional<Tensor> paramStorage = param->getStorage();
        THOR_THROW_IF_FALSE(paramStorage.has_value());
        inputs[param->getName()] = paramStorage.value();
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
    numBackwardApplications = 0;

    for (uint32_t applicationIndex = 0; applicationIndex < applications.size(); ++applicationIndex) {
        ApplicationState& app = applications[applicationIndex];

        for (uint32_t inputPort = 0; inputPort < inputNames.size(); ++inputPort) {
            const uint32_t flat = inputFlatIndex(applicationIndex, inputPort);
            if (flat >= featureInputs.size() || !featureInputs[flat].has_value()) {
                throw runtime_error("CustomLayer missing connected input port '" + inputNames[inputPort] + "' for application " +
                                    std::to_string(applicationIndex) + ".");
            }
        }

        for (uint32_t outputPort = 0; outputPort < outputNames.size(); ++outputPort) {
            const uint32_t flat = outputFlatIndex(applicationIndex, outputPort);
            if (flat >= featureOutputs.size() || !featureOutputs[flat].has_value()) {
                throw runtime_error("CustomLayer missing connected output port '" + outputNames[outputPort] + "' for application " +
                                    std::to_string(applicationIndex) + ".");
            }
        }

        app.forwardInputsByName = buildForwardInputs(applicationIndex);
        app.forwardOutputsByName = buildForwardOutputs(applicationIndex);

        app.forwardPrepared = std::make_shared<PreparedDynamicExpression>(
            layerDefinitionExpression.prepare(app.forwardInputsByName, app.forwardOutputsByName, computeStream(applicationIndex)));
        app.forwardPreRunHook = app.forwardPrepared->preForwardHook();
        validatePreparedExpressionInputs(*app.forwardPrepared);

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

        app.forwardStamped = std::make_shared<StampedExecutionPlan>(app.forwardPrepared->stamp(app.forwardOutputsByName));
        validateStampedOutputNames(*app.forwardStamped, outputNames, "forward");

        app.backwardErrorStamped = nullptr;
        app.backwardWeightsClearStamped = nullptr;
        app.backwardWeightsAccumulateStamped = nullptr;
        app.backwardWeightsFusedOptimizerUpdateStamped = nullptr;
        app.optimizerUpdateFusedParameterNames.clear();
        app.backwardAdditionalInputsByName.clear();
        app.backwardInputGradOutputsByName.clear();
        app.expectedBackwardErrorInputTensorIds.clear();
        app.upstreamInputNamesByOutput.clear();
        app.fusedCustomLossGradientsByOutput.clear();
        app.upstreamOutputNames.clear();
        app.activeParameterTargetNames.clear();
        app.backwardGradientPatternCompiled = false;

        if (isInferenceOnly() || isBackPropStub()) {
            app.backwardGradientPatternCompiled = true;
            continue;
        }

        if (!applicationHasAnyDownstreamBackprop(applicationIndex)) {
            pruneUpstreamErrorOutputsForApplication(applicationIndex);
            app.backwardErrorStamped = nullptr;
            app.backwardWeightsClearStamped = nullptr;
            app.backwardWeightsAccumulateStamped = nullptr;
            app.backwardWeightsFusedOptimizerUpdateStamped = nullptr;
            app.optimizerUpdateFusedParameterNames.clear();
            app.backwardAdditionalInputsByName.clear();
            app.backwardInputGradOutputsByName.clear();
            app.backwardGradientPatternCompiled = true;
            continue;
        }

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
                if (fusedLossIt != fusedCustomLossGradientByOutputFlatIndex.end()) {
                    app.fusedCustomLossGradientsByOutput.emplace(outputNames[outputPort], fusedLossIt->second);
                } else {
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

        if (!inputTargets.empty() && !app.backwardAdditionalInputsByName.empty()) {
            if (applicationHasFusedCustomLossGradient(applicationIndex)) {
                app.backwardErrorStamped = stampBackwardForApplication(applicationIndex,
                                                                       inputTargets,
                                                                       false,
                                                                       app.backwardInputGradOutputsByName,
                                                                       computeStream(applicationIndex));
            } else {
                app.backwardErrorStamped =
                    std::make_shared<StampedExecutionPlan>(app.forwardPrepared->stampBackward(inputTargets,
                                                                                              app.upstreamInputNamesByOutput,
                                                                                              false,
                                                                                              app.backwardAdditionalInputsByName,
                                                                                              {},
                                                                                              app.backwardInputGradOutputsByName));
            }
        }

        std::vector<std::string> allTrainableParameterTargets;
        for (auto& parameter : parameters) {
            if (parameter->isTrainingEnabled()) {
                allTrainableParameterTargets.push_back(parameter->getName());
            }
        }

        std::vector<std::string> activeParameterTargets = app.forwardPrepared->equation().filterTensorInputNamesReachableFromOutputs(
            allTrainableParameterTargets, app.upstreamOutputNames);
        if (activeParameterTargets.empty() && !allTrainableParameterTargets.empty() && !app.upstreamOutputNames.empty()) {
            // The reachability filter is an optimization for reducing the set of
            // parameter-gradient targets. It must not be allowed to turn a live
            // backward path into a silent no-op optimizer update. Some expression
            // lowerings can hide parameter reachability behind internal nodes that
            // are not visible to the generic filter; in that case, conservatively
            // stamp all trainable parameter gradients and let autodiff produce zeros
            // for any truly-unreachable parameter.
            activeParameterTargets = allTrainableParameterTargets;
        }
        app.activeParameterTargetNames = std::unordered_set<std::string>(activeParameterTargets.begin(), activeParameterTargets.end());

        if (trainingUpdateDiagnosticsEnabled()) {
            std::fprintf(stderr,
                         "THOR_TRAINING_UPDATE_DIAGNOSTIC layer=%s app=%u compile_backward downstream=%d fused_loss=%d backward_additional_inputs=%zu upstream_outputs=%s all_trainable=%s active_trainable=%s gradient_update_stream=%d\n",
                         getName().c_str(),
                         applicationIndex,
                         applicationHasAnyDownstreamBackprop(applicationIndex) ? 1 : 0,
                         applicationHasFusedCustomLossGradient(applicationIndex) ? 1 : 0,
                         app.backwardAdditionalInputsByName.size(),
                         joinNames(app.upstreamOutputNames).c_str(),
                         joinNames(allTrainableParameterTargets).c_str(),
                         joinNames(app.activeParameterTargetNames).c_str(),
                         gradientUpdateStream.has_value() ? 1 : 0);
        }

        std::vector<std::string> fusedParameterTargets;
        if (canFuseOptimizerUpdatesForApplication(applicationIndex)) {
            for (const std::string& parameterName : activeParameterTargets) {
                for (const auto& parameter : parameters) {
                    if (parameter->getName() == parameterName && parameter->isTrainingEnabled() && parameter->hasOptimizer() &&
                        parameter->getOptimizer()->supportsDenseUpdateFusion()) {
                        fusedParameterTargets.push_back(parameterName);
                        app.optimizerUpdateFusedParameterNames.insert(parameterName);
                        break;
                    }
                }
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

            if (app.optimizerUpdateFusedParameterNames.contains(parameter->getName())) {
                continue;
            }

            THOR_THROW_IF_FALSE(parameterOptimizer->getWeightsGradient().has_value());
            const std::string gradName = parameter->getName() + "_grad";
            Tensor gradientTensor = parameterOptimizer->getWeightsGradient().value();
            allMaterializedParameterTargets.push_back(parameter->getName());
            allMaterializedParameterPreallocatedOutputs[gradName] = gradientTensor;
            if (app.activeParameterTargetNames.contains(parameter->getName())) {
                activeMaterializedParameterTargets.push_back(parameter->getName());
                activeMaterializedParameterPreallocatedOutputs[gradName] = gradientTensor;
            }
        }

        if (trainingUpdateDiagnosticsEnabled()) {
            std::fprintf(stderr,
                         "THOR_TRAINING_UPDATE_DIAGNOSTIC layer=%s app=%u compile_parameter_targets fused=%s all_materialized=%s active_materialized=%s materialized_preallocated_outputs=%zu active_preallocated_outputs=%zu\n",
                         getName().c_str(),
                         applicationIndex,
                         joinNames(fusedParameterTargets).c_str(),
                         joinNames(allMaterializedParameterTargets).c_str(),
                         joinNames(activeMaterializedParameterTargets).c_str(),
                         allMaterializedParameterPreallocatedOutputs.size(),
                         activeMaterializedParameterPreallocatedOutputs.size());
        }

        if (!allTrainableParameterTargets.empty() && !app.backwardAdditionalInputsByName.empty()) {
            THOR_THROW_IF_FALSE(gradientUpdateStream.has_value());

            if (!fusedParameterTargets.empty()) {
                app.backwardWeightsFusedOptimizerUpdateStamped =
                    buildFusedOptimizerUpdatePlan(applicationIndex, fusedParameterTargets, parameterStorageByName);
            }

            if (!allMaterializedParameterTargets.empty()) {
                // Every application with downstream backprop gets a clear-first stamp that writes every materialized
                // gradient buffer. Parameters handled by backwardWeightsFusedOptimizerUpdateStamped intentionally
                // skip this dense gradient write/read round trip.
                if (applicationHasFusedCustomLossGradient(applicationIndex)) {
                    app.backwardWeightsClearStamped = stampBackwardForApplication(applicationIndex,
                                                                                  allMaterializedParameterTargets,
                                                                                  false,
                                                                                  allMaterializedParameterPreallocatedOutputs,
                                                                                  gradientUpdateStream.value());

                    if (!activeMaterializedParameterTargets.empty()) {
                        app.backwardWeightsAccumulateStamped = stampBackwardForApplication(applicationIndex,
                                                                                           activeMaterializedParameterTargets,
                                                                                           true,
                                                                                           activeMaterializedParameterPreallocatedOutputs,
                                                                                           gradientUpdateStream.value());
                    }
                } else {
                    PreparedDynamicExpression gradientPrepared =
                        layerDefinitionExpression.prepare(app.forwardInputsByName, app.forwardOutputsByName, gradientUpdateStream.value());
                    validatePreparedExpressionInputs(gradientPrepared);

                    app.backwardWeightsClearStamped =
                        std::make_shared<StampedExecutionPlan>(gradientPrepared.stampBackward(allMaterializedParameterTargets,
                                                                                              app.upstreamInputNamesByOutput,
                                                                                              false,
                                                                                              app.backwardAdditionalInputsByName,
                                                                                              {},
                                                                                              allMaterializedParameterPreallocatedOutputs));

                    if (!activeMaterializedParameterTargets.empty()) {
                        app.backwardWeightsAccumulateStamped =
                            std::make_shared<StampedExecutionPlan>(gradientPrepared.stampBackward(activeMaterializedParameterTargets,
                                                                                                  app.upstreamInputNamesByOutput,
                                                                                                  true,
                                                                                                  app.backwardAdditionalInputsByName,
                                                                                                  {},
                                                                                                  activeMaterializedParameterPreallocatedOutputs));
                    }
                }
            }
        }
    }

    if (trainingUpdateDiagnosticsEnabled()) {
        for (uint32_t applicationIndex = 0; applicationIndex < applications.size(); ++applicationIndex) {
            const ApplicationState& app = applications[applicationIndex];
            std::fprintf(stderr,
                         "THOR_TRAINING_UPDATE_DIAGNOSTIC layer=%s app=%u compiled_stamps backward_error=%d weights_clear=%d weights_accumulate=%d fused_update=%d expected_backward_errors=%zu num_backward_applications=%u\n",
                         getName().c_str(),
                         applicationIndex,
                         app.backwardErrorStamped != nullptr ? 1 : 0,
                         app.backwardWeightsClearStamped != nullptr ? 1 : 0,
                         app.backwardWeightsAccumulateStamped != nullptr ? 1 : 0,
                         app.backwardWeightsFusedOptimizerUpdateStamped != nullptr ? 1 : 0,
                         app.expectedBackwardErrorInputTensorIds.size(),
                         numBackwardApplications);
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

void CustomLayer::synchronizeComputeStreamForForwardInputs(uint32_t applicationIndex) {
    Stream& runStream = computeStream(applicationIndex);
    const uint32_t runFlat = primaryInputFlatIndex(applicationIndex);
    for (uint32_t inputPort = 0; inputPort < inputNames.size(); ++inputPort) {
        const uint32_t flat = inputFlatIndex(applicationIndex, inputPort);
        if (flat == runFlat || flat >= streams.size() || !featureInputs[flat].has_value()) {
            continue;
        }
        Event readyEvent = streams[flat].putEvent();
        runStream.waitEvent(readyEvent);
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

    std::set<uint32_t> candidateApplications;
    for (uint32_t flat = 0; flat < featureInputs.size(); ++flat) {
        if (featureInputs[flat].has_value() && featureInputs[flat].value() == featureInput.value()) {
            candidateApplications.insert(flat / inputNames.size());
        }
    }
    THOR_THROW_IF_FALSE(!candidateApplications.empty());

    if (isStartOfForward) {
        if (weightsAreUpToDateEvent.has_value()) {
            for (const Stream& dataStream : uniqueDataStreams) {
                dataStream.waitEvent(weightsAreUpToDateEvent.value());
            }
        }
        weightsAreUpToDateEvent.reset();
        isStartOfForward = false;
        isStartOfBackward = true;
        clearGradientFirstThisBackwardPass = false;
        clearForwardArrivalBookkeeping();
    }

    const unsigned long tensorId = featureInput.value().getTensorId();
    for (uint32_t applicationIndex : candidateApplications) {
        ApplicationState& app = applications[applicationIndex];
        if (app.forwardRanThisPass) {
            continue;
        }
        if (app.stillWaitingForForwardInputTensorIds.count(tensorId) == 0) {
            continue;
        }
        app.stillWaitingForForwardInputTensorIds.erase(tensorId);

        if (!app.stillWaitingForForwardInputTensorIds.empty()) {
            continue;
        }

        app.forwardRanThisPass = true;
        synchronizeComputeStreamForForwardInputs(applicationIndex);
        computeFeatureOut(inputFlatIndex(applicationIndex, 0));

        for (uint32_t outputPort = 0; outputPort < outputNames.size(); ++outputPort) {
            const uint32_t flat = outputFlatIndex(applicationIndex, outputPort);
            if (!nextLayers[flat].has_value())
                continue;
            nextLayers[flat].value()->forward(featureOutputs[flat], validationPass, batchSize);
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
        if (validationPass || !applicationHasAnyDownstreamBackprop(applicationIndex)) {
            clearForwardArrivalBookkeeping(applicationIndex);
        }
    }
}

void CustomLayer::backward(std::optional<Tensor> errorInput, uint32_t batchSize) {
    THOR_THROW_IF_FALSE(running);

    if (!errorInput.has_value())
        return;

    std::set<uint32_t> candidateApplications;
    for (uint32_t flat = 0; flat < errorInputs.size(); ++flat) {
        if (errorInputs[flat].has_value() && errorInputs[flat].value() == errorInput.value()) {
            candidateApplications.insert(flat / outputNames.size());
        }
    }
    THOR_THROW_IF_FALSE(!candidateApplications.empty());

    if (trainingUpdateDiagnosticsEnabled()) {
        std::fprintf(stderr,
                     "THOR_TRAINING_UPDATE_DIAGNOSTIC layer=%s backward_entry batch=%u candidate_applications=%zu is_start_of_backward=%d num_backward_applications=%u completed_this_pass=%u\n",
                     getName().c_str(),
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
    }

    const unsigned long tensorId = errorInput.value().getTensorId();
    for (uint32_t applicationIndex : candidateApplications) {
        ApplicationState& app = applications[applicationIndex];
        if (app.backwardRanThisPass) {
            continue;
        }
        if (app.stillWaitingForBackwardErrorInputTensorIds.count(tensorId) == 0) {
            continue;
        }
        app.stillWaitingForBackwardErrorInputTensorIds.erase(tensorId);

        if (!app.stillWaitingForBackwardErrorInputTensorIds.empty()) {
            if (trainingUpdateDiagnosticsEnabled()) {
                std::fprintf(stderr,
                             "THOR_TRAINING_UPDATE_DIAGNOSTIC layer=%s app=%u backward_waiting remaining_errors=%zu\n",
                             getName().c_str(),
                             applicationIndex,
                             app.stillWaitingForBackwardErrorInputTensorIds.size());
            }
            continue;
        }

        if (trainingUpdateDiagnosticsEnabled()) {
            std::fprintf(stderr,
                         "THOR_TRAINING_UPDATE_DIAGNOSTIC layer=%s app=%u backward_ready batch=%u active_parameters=%s backward_error_stamp=%d weights_clear_stamp=%d weights_accumulate_stamp=%d fused_update_stamp=%d\n",
                         getName().c_str(),
                         applicationIndex,
                         batchSize,
                         joinNames(app.activeParameterTargetNames).c_str(),
                         app.backwardErrorStamped != nullptr ? 1 : 0,
                         app.backwardWeightsClearStamped != nullptr ? 1 : 0,
                         app.backwardWeightsAccumulateStamped != nullptr ? 1 : 0,
                         app.backwardWeightsFusedOptimizerUpdateStamped != nullptr ? 1 : 0);
        }

        app.backwardRanThisPass = true;

        std::optional<Event> errorInputReadyEvent = std::nullopt;
        if (gradientUpdateStream.has_value()) {
            errorInputReadyEvent = computeStream(applicationIndex).putEvent();
        }

        std::optional<Event> errorOutHasBeenComputedEvent = std::nullopt;
        if (app.backwardErrorStamped != nullptr) {
            errorOutHasBeenComputedEvent = computeErrorOut(inputFlatIndex(applicationIndex, 0));
            if (errorOutHasBeenComputedEvent.has_value()) {
                errorOutHasBeenComputedEvents.push_back(errorOutHasBeenComputedEvent.value());
            }
        }

        if (gradientUpdateStream.has_value() && errorInputReadyEvent.has_value()) {
            gradientUpdateStream.value().waitEvent(errorInputReadyEvent.value());
        }
        if (gradientUpdateStream.has_value() && errorOutHasBeenComputedEvent.has_value() &&
            app.backwardWeightsFusedOptimizerUpdateStamped != nullptr) {
            // Fused optimizer stamps update parameter storage directly, so they must preserve the same
            // ordering as the legacy materialized path: upstream input-gradient computation reads old
            // weights before the optimizer overwrites them.
            gradientUpdateStream.value().waitEvent(errorOutHasBeenComputedEvent.value());
        }

        accumulateWeightsGradientForApplication(applicationIndex, clearGradientFirstThisBackwardPass, batchSize);
        recordEffectiveParameterBatchSizeForApplication(applicationIndex, batchSize);
        clearGradientFirstThisBackwardPass = false;

        for (uint32_t inputPort = 0; inputPort < inputNames.size(); ++inputPort) {
            const uint32_t inputFlat = inputFlatIndex(applicationIndex, inputPort);
            if (!previousLayers[inputFlat].has_value() || !errorOutputs[inputFlat].has_value()) {
                continue;
            }
            previousLayers[inputFlat].value()->backward(errorOutputs[inputFlat], batchSize);
        }

        numBackwardApplicationsCompletedThisPass += 1;
    }

    if (numBackwardApplications > 0 && numBackwardApplicationsCompletedThisPass == numBackwardApplications) {
        if (trainingUpdateDiagnosticsEnabled()) {
            std::fprintf(stderr,
                         "THOR_TRAINING_UPDATE_DIAGNOSTIC layer=%s backward_pass_complete effective_batches=%s gradient_update_stream=%d\n",
                         getName().c_str(),
                         joinNames(effectiveBatchSizeByParameterName).c_str(),
                         gradientUpdateStream.has_value() ? 1 : 0);
        }
        numBackwardApplicationsCompletedThisPass = 0;
        weightsAreUpToDateEvent.reset();

        if (gradientUpdateStream.has_value()) {
            for (const Event& eOutComputedEvent : errorOutHasBeenComputedEvents) {
                gradientUpdateStream.value().waitEvent(eOutComputedEvent);
            }

            bool anyWeightsUpdated = false;
            for (const ApplicationState& app : applications) {
                if (app.backwardRanThisPass && !app.optimizerUpdateFusedParameterNames.empty()) {
                    anyWeightsUpdated = true;
                    break;
                }
            }
            for (const auto& parameter : parameters) {
                if (!parameter->isTrainingEnabled()) {
                    if (trainingUpdateDiagnosticsEnabled()) {
                        std::fprintf(stderr,
                                     "THOR_TRAINING_UPDATE_DIAGNOSTIC layer=%s parameter=%s apply_skip reason=training_disabled\n",
                                     getName().c_str(),
                                     parameter->getName().c_str());
                    }
                    continue;
                }

                const auto effectiveBatchSizeIt = effectiveBatchSizeByParameterName.find(parameter->getName());
                if (effectiveBatchSizeIt == effectiveBatchSizeByParameterName.end() || effectiveBatchSizeIt->second == 0) {
                    if (trainingUpdateDiagnosticsEnabled()) {
                        std::fprintf(stderr,
                                     "THOR_TRAINING_UPDATE_DIAGNOSTIC layer=%s parameter=%s apply_skip reason=no_effective_batch effective_batches=%s\n",
                                     getName().c_str(),
                                     parameter->getName().c_str(),
                                     joinNames(effectiveBatchSizeByParameterName).c_str());
                    }
                    continue;
                }

                if (effectiveBatchSizeIt->second > std::numeric_limits<uint32_t>::max()) {
                    throw runtime_error("CustomLayer effective parameter batch size exceeds uint32_t range for parameter " +
                                        parameter->getName() + ".");
                }

                if (trainingUpdateDiagnosticsEnabled()) {
                    std::fprintf(stderr,
                                 "THOR_TRAINING_UPDATE_DIAGNOSTIC layer=%s parameter=%s apply_gradient batch=%llu\n",
                                 getName().c_str(),
                                 parameter->getName().c_str(),
                                 static_cast<unsigned long long>(effectiveBatchSizeIt->second));
                }
                anyWeightsUpdated |= parameter->applyGradient(static_cast<uint32_t>(effectiveBatchSizeIt->second));
            }
            effectiveBatchSizeByParameterName.clear();
            if (trainingUpdateDiagnosticsEnabled()) {
                std::fprintf(stderr,
                             "THOR_TRAINING_UPDATE_DIAGNOSTIC layer=%s update_result any_weights_updated=%d\n",
                             getName().c_str(),
                             anyWeightsUpdated ? 1 : 0);
            }
            if (anyWeightsUpdated) {
                weightsAreUpToDateEvent = gradientUpdateStream.value().putEvent();
            }
        }
        errorOutHasBeenComputedEvents.clear();
        isStartOfForward = true;
    }
}

void CustomLayer::computeFeatureOut(uint32_t connectionNumber) {
    DecodedConnection decoded = decodeInputConnectionType(static_cast<int>(connectionNumber));
    if (decoded.applicationIndex >= applications.size() || !applications[decoded.applicationIndex].forwardStamped) {
        throw runtime_error("CustomLayer::computeFeatureOut requires a stamped forward plan.");
    }
    if (applications[decoded.applicationIndex].forwardPreRunHook) {
        applications[decoded.applicationIndex].forwardPreRunHook(computeStream(decoded.applicationIndex));
    }
    applications[decoded.applicationIndex].forwardStamped->run();
}

std::optional<Event> CustomLayer::computeErrorOut(uint32_t connectionNumber) {
    DecodedConnection decoded = decodeInputConnectionType(static_cast<int>(connectionNumber));
    if (decoded.applicationIndex >= applications.size() || applications[decoded.applicationIndex].backwardErrorStamped == nullptr) {
        return std::nullopt;
    }
    applications[decoded.applicationIndex].backwardErrorStamped->run();
    return computeStream(decoded.applicationIndex).putEvent();
}

void CustomLayer::accumulateWeightsGradientForApplication(uint32_t applicationIndex, bool clearGradientFirst, uint32_t batchSize) {
    if (!gradientUpdateStream.has_value()) {
        return;
    }
    if (applicationIndex >= applications.size()) {
        return;
    }
    ApplicationState& app = applications[applicationIndex];

    // backwardWeightsClearStamped is an overwrite-mode backward-gradient stamp, not a zero-only
    // memset.  It computes this application's materialized parameter gradient and writes it into
    // the dense gradient buffer.  Therefore the first materialized application in a backward pass
    // must run the clear stamp instead of the accumulate stamp; running both for the same
    // application double-counts the gradient.  Later applications use the accumulate stamp to add
    // their contribution to the buffer established by the first application.
    const bool ranMaterializedOverwrite = clearGradientFirst && app.backwardWeightsClearStamped != nullptr;
    if (trainingUpdateDiagnosticsEnabled()) {
        std::fprintf(stderr,
                     "THOR_TRAINING_UPDATE_DIAGNOSTIC layer=%s app=%u accumulate batch=%u clear_first=%d run_clear=%d run_accumulate=%d run_fused_update=%d active_parameters=%s\n",
                     getName().c_str(),
                     applicationIndex,
                     batchSize,
                     clearGradientFirst ? 1 : 0,
                     ranMaterializedOverwrite ? 1 : 0,
                     (!ranMaterializedOverwrite && app.backwardWeightsAccumulateStamped != nullptr) ? 1 : 0,
                     app.backwardWeightsFusedOptimizerUpdateStamped != nullptr ? 1 : 0,
                     joinNames(app.activeParameterTargetNames).c_str());
    }
    if (ranMaterializedOverwrite) {
        app.backwardWeightsClearStamped->run();
    }

    if (app.backwardWeightsFusedOptimizerUpdateStamped != nullptr) {
        app.backwardWeightsFusedOptimizerUpdateStamped->run(buildFusedOptimizerRuntimeScalars(applicationIndex, batchSize));
    }

    if (!ranMaterializedOverwrite && app.backwardWeightsAccumulateStamped != nullptr) {
        app.backwardWeightsAccumulateStamped->run();
    }
}

void CustomLayer::accumulateWeightsGradient(uint32_t connectionNumber, bool clearGradientFirst) {
    DecodedConnection decoded = decodeInputConnectionType(static_cast<int>(connectionNumber));
    accumulateWeightsGradientForApplication(decoded.applicationIndex, clearGradientFirst, 0);
}

uint64_t CustomLayer::flopCountForward() {
    uint64_t flops = 0;
    for (const ApplicationState& app : applications) {
        if (app.forwardStamped != nullptr) {
            flops += app.forwardStamped->flopCount();
        }
    }
    return flops;
}

uint64_t CustomLayer::flopCountBackward() {
    uint64_t flops = 0;
    for (const ApplicationState& app : applications) {
        if (app.backwardErrorStamped != nullptr) {
            flops += app.backwardErrorStamped->flopCount();
        }
        if (app.backwardWeightsAccumulateStamped != nullptr) {
            flops += app.backwardWeightsAccumulateStamped->flopCount();
        }
        if (app.backwardWeightsFusedOptimizerUpdateStamped != nullptr) {
            flops += app.backwardWeightsFusedOptimizerUpdateStamped->flopCount();
        }
    }
    return flops;
}

uint64_t CustomLayer::batchSizeForFlopEstimate() const {
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
