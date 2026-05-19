#include <optional>
#include "DeepLearning/Implementation/Layers/CustomLayer.h"

#include <limits>
#include <set>
#include <stdexcept>
#include <unordered_set>

#include "DeepLearning/Implementation/ThorError.h"
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
}  // namespace

CustomLayer::CustomLayer(DynamicExpression expr,
                         const TensorPlacement& placement,
                         const std::vector<std::shared_ptr<PhysicalParameter>>& parameters,
                         bool inferenceOnly,
                         int64_t stampedId,
                         bool useFastMath)
    : CustomLayer(std::move(expr),
                  std::vector<std::string>{"feature_input"},
                  std::vector<std::string>{"feature_output"},
                  placement,
                  parameters,
                  inferenceOnly,
                  stampedId,
                  useFastMath) {}

CustomLayer::CustomLayer(DynamicExpression expr,
                         std::vector<std::string> inputNames,
                         std::vector<std::string> outputNames,
                         const TensorPlacement& placement,
                         const std::vector<std::shared_ptr<PhysicalParameter>>& parameters,
                         bool inferenceOnly,
                         int64_t stampedId,
                         bool useFastMath)
    : TrainableLayer(placement, inferenceOnly, stampedId),
      layerDefinitionExpression(std::move(expr)),
      inputNames(std::move(inputNames)),
      outputNames(std::move(outputNames)),
      useFastMath(useFastMath) {
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
    for (const std::string& parameterName : app.activeParameterTargetNames) {
        effectiveBatchSizeByParameterName[parameterName] += batchSize;
    }
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
            // Must compile optimizer every time to properly toggle parameter trainability
            param->compileOptimizer(gradientUpdateStream, isInferenceOnly());
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
        app.backwardAdditionalInputsByName.clear();
        app.backwardInputGradOutputsByName.clear();
        app.expectedBackwardErrorInputTensorIds.clear();
        app.upstreamInputNamesByOutput.clear();
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
                app.upstreamInputNamesByOutput[outputNames[outputPort]] = errorInputNameForOutput(outputPort);
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
            app.backwardErrorStamped =
                std::make_shared<StampedExecutionPlan>(app.forwardPrepared->stampBackward(inputTargets,
                                                                                          app.upstreamInputNamesByOutput,
                                                                                          false,
                                                                                          app.backwardAdditionalInputsByName,
                                                                                          {},
                                                                                          app.backwardInputGradOutputsByName));
        }

        std::vector<std::string> allTrainableParameterTargets;
        for (auto& parameter : parameters) {
            if (parameter->isTrainingEnabled()) {
                allTrainableParameterTargets.push_back(parameter->getName());
            }
        }

        std::vector<std::string> activeParameterTargets = app.forwardPrepared->equation().filterTensorInputNamesReachableFromOutputs(
            allTrainableParameterTargets, app.upstreamOutputNames);
        app.activeParameterTargetNames = std::unordered_set<std::string>(activeParameterTargets.begin(), activeParameterTargets.end());

        PreparedDynamicExpression::TensorMap allParameterPreallocatedOutputs;
        PreparedDynamicExpression::TensorMap activeParameterPreallocatedOutputs;
        for (auto& parameter : parameters) {
            if (!parameter->isTrainingEnabled()) {
                continue;
            }

            THOR_THROW_IF_FALSE(parameter->hasOptimizer());
            const shared_ptr<Optimizer>& parameterOptimizer = parameter->getOptimizer();
            THOR_THROW_IF_FALSE(parameterOptimizer != nullptr);
            THOR_THROW_IF_FALSE(parameterOptimizer->getWeightsGradient().has_value());

            const std::string gradName = parameter->getName() + "_grad";
            Tensor gradientTensor = parameterOptimizer->getWeightsGradient().value();
            allParameterPreallocatedOutputs[gradName] = gradientTensor;
            if (app.activeParameterTargetNames.contains(parameter->getName())) {
                activeParameterPreallocatedOutputs[gradName] = gradientTensor;
            }
        }

        if (!allTrainableParameterTargets.empty() && !app.backwardAdditionalInputsByName.empty()) {
            THOR_THROW_IF_FALSE(gradientUpdateStream.has_value());

            PreparedDynamicExpression gradientPrepared =
                layerDefinitionExpression.prepare(app.forwardInputsByName, app.forwardOutputsByName, gradientUpdateStream.value());
            validatePreparedExpressionInputs(gradientPrepared);

            // Every application with downstream backprop gets a clear-first stamp that writes every trainable
            // gradient buffer. Whichever application arrives first in a backward pass can therefore initialize
            // all parameter gradients without a separate memset, while the partial upstream map makes inactive
            // outputs contribute zeros rather than synthetic zero tensors.
            app.backwardWeightsClearStamped =
                std::make_shared<StampedExecutionPlan>(gradientPrepared.stampBackward(allTrainableParameterTargets,
                                                                                      app.upstreamInputNamesByOutput,
                                                                                      false,
                                                                                      app.backwardAdditionalInputsByName,
                                                                                      {},
                                                                                      allParameterPreallocatedOutputs));

            if (!activeParameterTargets.empty()) {
                app.backwardWeightsAccumulateStamped =
                    std::make_shared<StampedExecutionPlan>(gradientPrepared.stampBackward(activeParameterTargets,
                                                                                          app.upstreamInputNamesByOutput,
                                                                                          true,
                                                                                          app.backwardAdditionalInputsByName,
                                                                                          {},
                                                                                          activeParameterPreallocatedOutputs));
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
        if (!applicationHasAnyDownstreamBackprop(applicationIndex)) {
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
            continue;
        }

        app.backwardRanThisPass = true;

        std::optional<Event> errorInputReadyEvent = std::nullopt;
        if (gradientUpdateStream.has_value()) {
            errorInputReadyEvent = computeStream(applicationIndex).putEvent();
        }

        if (app.backwardErrorStamped != nullptr) {
            std::optional<Event> errorOutHasBeenComputedEvent = computeErrorOut(inputFlatIndex(applicationIndex, 0));
            if (errorOutHasBeenComputedEvent.has_value()) {
                errorOutHasBeenComputedEvents.push_back(errorOutHasBeenComputedEvent.value());
            }
        }

        if (gradientUpdateStream.has_value() && errorInputReadyEvent.has_value()) {
            gradientUpdateStream.value().waitEvent(errorInputReadyEvent.value());
        }

        accumulateWeightsGradient(inputFlatIndex(applicationIndex, 0), clearGradientFirstThisBackwardPass);
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
        numBackwardApplicationsCompletedThisPass = 0;
        weightsAreUpToDateEvent.reset();

        if (gradientUpdateStream.has_value()) {
            for (const Event& eOutComputedEvent : errorOutHasBeenComputedEvents) {
                gradientUpdateStream.value().waitEvent(eOutComputedEvent);
            }

            bool anyWeightsUpdated = false;
            for (const auto& parameter : parameters) {
                if (!parameter->isTrainingEnabled()) {
                    continue;
                }

                const auto effectiveBatchSizeIt = effectiveBatchSizeByParameterName.find(parameter->getName());
                if (effectiveBatchSizeIt == effectiveBatchSizeByParameterName.end() || effectiveBatchSizeIt->second == 0) {
                    continue;
                }

                if (effectiveBatchSizeIt->second > std::numeric_limits<uint32_t>::max()) {
                    throw runtime_error("CustomLayer effective parameter batch size exceeds uint32_t range for parameter " +
                                        parameter->getName() + ".");
                }

                anyWeightsUpdated |= parameter->applyGradient(static_cast<uint32_t>(effectiveBatchSizeIt->second));
            }
            effectiveBatchSizeByParameterName.clear();
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

void CustomLayer::accumulateWeightsGradient(uint32_t connectionNumber, bool clearGradientFirst) {
    if (!gradientUpdateStream.has_value()) {
        return;
    }

    DecodedConnection decoded = decodeInputConnectionType(static_cast<int>(connectionNumber));
    if (decoded.applicationIndex >= applications.size()) {
        return;
    }
    ApplicationState& app = applications[decoded.applicationIndex];

    if (clearGradientFirst) {
        if (app.backwardWeightsClearStamped != nullptr) {
            app.backwardWeightsClearStamped->run();
        }
        return;
    }

    if (app.backwardWeightsAccumulateStamped != nullptr) {
        app.backwardWeightsAccumulateStamped->run();
    }
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
    }
    return flops;
}

bool CustomLayer::isBackPropStub() {
    for (const auto& errorOutput : errorOutputs) {
        if (errorOutput.has_value()) {
            return false;
        }
    }
    return true;
}

}  // namespace ThorImplementation
