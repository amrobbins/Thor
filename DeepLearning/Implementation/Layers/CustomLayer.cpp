#include "DeepLearning/Implementation/Layers/CustomLayer.h"

#include <set>
#include <stdexcept>

using namespace std;

namespace ThorImplementation {

namespace {
std::set<std::string> toNameSet(const std::vector<std::string>& names) { return std::set<std::string>(names.begin(), names.end()); }
}  // namespace

CustomLayer::CustomLayer(DynamicExpression expr,
                         const TensorPlacement& placement,
                         const std::vector<std::shared_ptr<Parameter>>& parameters,
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
                         const std::vector<std::shared_ptr<Parameter>>& parameters,
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

        addParameter(param);  // verifies parameter name uniqueness
    }

    ensurePortStorageAllocated();
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

uint32_t CustomLayer::primaryInputPort() const {
    for (uint32_t i = 0; i < featureInputs.size(); ++i) {
        if (featureInputs[i].isPresent()) {
            return i;
        }
    }
    throw runtime_error("CustomLayer requires at least one connected input port before execution.");
}

Stream& CustomLayer::computeStream() {
    const uint32_t port = primaryInputPort();
    assert(port < streams.size());
    return streams[port];
}

const Stream& CustomLayer::computeStream() const {
    const uint32_t port = primaryInputPort();
    assert(port < streams.size());
    return streams[port];
}

void CustomLayer::ensurePortStorageAllocated() {
    featureInputs.resize(inputNames.size(), Optional<Tensor>::empty());
    errorOutputs.resize(inputNames.size(), Optional<Tensor>::empty());
    previousLayers.resize(inputNames.size(), Optional<Layer*>::empty());
    streams.resize(inputNames.size());

    featureOutputs.resize(outputNames.size(), Optional<Tensor>::empty());
    errorInputs.resize(outputNames.size(), Optional<Tensor>::empty());
    nextLayers.resize(outputNames.size(), Optional<Layer*>::empty());

    featureInputsConnectedForPorts.resize(inputNames.size(), Optional<Tensor>::empty());
    errorOutputsConnectedForPorts.resize(inputNames.size(), Optional<Tensor>::empty());
    featureOutputsConnectedForPorts.resize(outputNames.size(), Optional<Tensor>::empty());
    errorInputsConnectedForPorts.resize(outputNames.size(), Optional<Tensor>::empty());
}

void CustomLayer::clearForwardArrivalBookkeeping() {
    allForwardInputTensorIds.clear();
    stillWaitingForForwardInputTensorIds.clear();
    for (const auto& input : featureInputs) {
        if (input.isPresent()) {
            allForwardInputTensorIds.insert(input.get().getTensorId());
        }
    }
    stillWaitingForForwardInputTensorIds = allForwardInputTensorIds;
}

void CustomLayer::clearBackwardArrivalBookkeeping() {
    allBackwardErrorInputTensorIds.clear();
    stillWaitingForBackwardErrorInputTensorIds.clear();
    for (const auto& errorInput : errorInputs) {
        if (errorInput.isPresent()) {
            allBackwardErrorInputTensorIds.insert(errorInput.get().getTensorId());
        }
    }
    stillWaitingForBackwardErrorInputTensorIds = allBackwardErrorInputTensorIds;
}

void CustomLayer::initialize() {
    TrainableLayer::initialize();
    clearForwardArrivalBookkeeping();
    clearBackwardArrivalBookkeeping();
}

Parameter::StorageContext CustomLayer::buildParameterStorageContext() const {
    std::vector<Tensor> connectedFeatureInputs;
    connectedFeatureInputs.reserve(inputNames.size());
    std::unordered_map<std::string, Tensor> namedFeatureInputs;

    for (uint32_t i = 0; i < inputNames.size(); ++i) {
        if (featureInputs[i].isEmpty()) {
            throw runtime_error("CustomLayer missing connected feature input for port '" + inputNames[i] + "'.");
        }
        connectedFeatureInputs.push_back(featureInputs[i].get());
        namedFeatureInputs.emplace(inputNames[i], featureInputs[i].get());
    }

    return Parameter::StorageContext(
        featureInputs[primaryInputPort()].get(), std::move(connectedFeatureInputs), std::move(namedFeatureInputs));
}

PreparedDynamicExpression::TensorMap CustomLayer::buildForwardInputs() {
    PreparedDynamicExpression::TensorMap inputs;

    for (uint32_t i = 0; i < inputNames.size(); ++i) {
        if (featureInputs[i].isEmpty()) {
            throw runtime_error("CustomLayer missing connected feature input for port '" + inputNames[i] + "'.");
        }
        inputs[inputNames[i]] = featureInputs[i].get();
    }

    const Parameter::StorageContext parameterStorageContext = buildParameterStorageContext();
    for (const auto& param : parameters) {
        if (!param->isStorageInitialized()) {
            param->compileStorageAndOptimizer(parameterStorageContext, gradientUpdateStream, isInferenceOnly());
        }

        Optional<Tensor> paramStorage = param->getStorage();
        assert(paramStorage.isPresent());
        inputs[param->getName()] = paramStorage.get();
    }

    return inputs;
}

PreparedDynamicExpression::TensorMap CustomLayer::buildForwardOutputs() const {
    PreparedDynamicExpression::TensorMap outputs;
    for (uint32_t i = 0; i < outputNames.size(); ++i) {
        if (featureOutputs[i].isPresent()) {
            outputs[outputNames[i]] = featureOutputs[i].get();
        }
    }
    return outputs;
}

PreparedDynamicExpression::TensorMap CustomLayer::buildBackwardAdditionalInputs() const {
    PreparedDynamicExpression::TensorMap backwardAdditionalInputs;
    for (uint32_t outputPort = 0; outputPort < outputNames.size(); ++outputPort) {
        if (errorInputs[outputPort].isPresent()) {
            backwardAdditionalInputs[errorInputNameForOutput(outputPort)] = errorInputs[outputPort].get();
        }
    }
    return backwardAdditionalInputs;
}

PreparedDynamicExpression::TensorMap CustomLayer::buildBackwardInputGradOutputs() const {
    PreparedDynamicExpression::TensorMap outputs;
    for (uint32_t inputPort = 0; inputPort < inputNames.size(); ++inputPort) {
        if (errorOutputs[inputPort].isPresent()) {
            outputs[errorOutputNameForInput(inputPort)] = errorOutputs[inputPort].get();
        }
    }
    return outputs;
}

std::string CustomLayer::errorInputNameForOutput(uint32_t outputPortIndex) const {
    assert(outputPortIndex < outputNames.size());
    return "__grad_" + outputNames[outputPortIndex];
}

std::string CustomLayer::errorOutputNameForInput(uint32_t inputPortIndex) const {
    assert(inputPortIndex < inputNames.size());
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

    if (actualInputNames != expectedInputNames) {
        std::string expectedInputNamesString;
        for (const auto& name : expectedInputNames)
            expectedInputNamesString += name + " ";

        std::string actualInputNamesString;
        for (const auto& name : actualInputNames)
            actualInputNamesString += name + " ";

        throw runtime_error("CustomLayer expression input mismatch. Expected inputs: " + expectedInputNamesString +
                            " Actual inputs used by prepared expression: " + actualInputNamesString);
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

Optional<Tensor> CustomLayer::inferFeatureOutputTensor(uint32_t outputPortIndex) {
    if (outputPortIndex >= outputNames.size()) {
        throw runtime_error("CustomLayer output port index out of range.");
    }

    PreparedDynamicExpression::TensorMap discoveredOutputs;
    PreparedDynamicExpression prepared = layerDefinitionExpression.prepare(buildForwardInputs(), discoveredOutputs, computeStream());
    validatePreparedExpressionInputs(prepared);
    StampedExecutionPlan stamped = prepared.stamp();
    return stamped.output(outputNames[outputPortIndex]);
}

void CustomLayer::compileImpl() {
    TrainableLayer::compileImpl();

    assert(placement.getMemDevice() == TensorPlacement::MemDevices::GPU);

    ensurePortStorageAllocated();
    clearForwardArrivalBookkeeping();
    clearBackwardArrivalBookkeeping();

    for (uint32_t inputPort = 0; inputPort < inputNames.size(); ++inputPort) {
        if (featureInputs[inputPort].isEmpty()) {
            throw runtime_error("CustomLayer missing connected input port '" + inputNames[inputPort] + "'.");
        }
    }

    for (uint32_t outputPort = 0; outputPort < outputNames.size(); ++outputPort) {
        if (featureOutputs[outputPort].isEmpty()) {
            throw runtime_error("CustomLayer missing connected output port '" + outputNames[outputPort] + "'.");
        }
    }

    forwardInputsByName = buildForwardInputs();
    forwardOutputsByName = buildForwardOutputs();

    forwardPrepared = std::make_shared<PreparedDynamicExpression>(
        layerDefinitionExpression.prepare(forwardInputsByName, forwardOutputsByName, computeStream()));
    validatePreparedExpressionInputs(*forwardPrepared);

    std::unordered_set<std::string> parameterNames;
    for (const auto& parameter : parameters) {
        parameterNames.insert(parameter->getName());
    }
    const auto parameterFanOverrides = forwardPrepared->getParameterFanOverrides(parameterNames);
    for (const auto& parameter : parameters) {
        auto it = parameterFanOverrides.find(parameter->getName());
        if (it != parameterFanOverrides.end()) {
            parameter->compileInitializer(it->second.fan_in, it->second.fan_out);
        } else {
            parameter->compileInitializer();
        }
    }

    forwardStamped = std::make_shared<StampedExecutionPlan>(forwardPrepared->stamp(forwardOutputsByName));
    validateStampedOutputNames(*forwardStamped, outputNames, "forward");

    backwardErrorStamped = nullptr;
    backwardWeightsClearStamped = nullptr;
    backwardWeightsAccumulateStamped = nullptr;
    backwardAdditionalInputsByName.clear();
    backwardInputGradOutputsByName.clear();

    if (isInferenceOnly() || isBackPropStub()) {
        return;
    }

    backwardAdditionalInputsByName = buildBackwardAdditionalInputs();
    backwardInputGradOutputsByName = buildBackwardInputGradOutputs();

    std::unordered_map<std::string, std::string> upstreamInputNamesByOutput;
    for (uint32_t outputPort = 0; outputPort < outputNames.size(); ++outputPort) {
        if (errorInputs[outputPort].isPresent()) {
            upstreamInputNamesByOutput[outputNames[outputPort]] = errorInputNameForOutput(outputPort);
        }
    }

    std::vector<std::string> inputTargets;
    for (uint32_t inputPort = 0; inputPort < inputNames.size(); ++inputPort) {
        if (errorOutputs[inputPort].isPresent()) {
            inputTargets.push_back(inputNames[inputPort]);
        }
    }

    if (!inputTargets.empty() && !backwardAdditionalInputsByName.empty()) {
        backwardErrorStamped = std::make_shared<StampedExecutionPlan>(forwardPrepared->stampBackward(
            inputTargets, upstreamInputNamesByOutput, false, backwardAdditionalInputsByName, {}, backwardInputGradOutputsByName));
    }

    std::vector<std::string> parameterTargets;
    PreparedDynamicExpression::TensorMap backwardParameterPreallocatedOutputs;
    for (auto& parameter : parameters) {
        if (!parameter->isTrainable()) {
            continue;
        }

        assert(parameter->hasOptimizer());
        const shared_ptr<Optimizer>& parameterOptimizer = parameter->getOptimizer();
        assert(parameterOptimizer != nullptr);
        assert(parameterOptimizer->getWeightsGradient().isPresent());

        const std::string gradName = parameter->getName() + "_grad";
        parameterTargets.push_back(parameter->getName());
        backwardParameterPreallocatedOutputs[gradName] = parameterOptimizer->getWeightsGradient().get();
    }

    if (!parameterTargets.empty() && !backwardAdditionalInputsByName.empty()) {
        assert(gradientUpdateStream.isPresent());

        PreparedDynamicExpression gradientPrepared =
            layerDefinitionExpression.prepare(forwardInputsByName, forwardOutputsByName, gradientUpdateStream.get());
        validatePreparedExpressionInputs(gradientPrepared);

        backwardWeightsClearStamped = std::make_shared<StampedExecutionPlan>(gradientPrepared.stampBackward(
            parameterTargets, upstreamInputNamesByOutput, false, backwardAdditionalInputsByName, {}, backwardParameterPreallocatedOutputs));

        backwardWeightsAccumulateStamped = std::make_shared<StampedExecutionPlan>(gradientPrepared.stampBackward(
            parameterTargets, upstreamInputNamesByOutput, true, backwardAdditionalInputsByName, {}, backwardParameterPreallocatedOutputs));
    }
}

Optional<Tensor> CustomLayer::createFeatureOutputTensor() {
    if (outputNames.size() != 1) {
        throw runtime_error("CustomLayer::createFeatureOutputTensor() without a connection type is only valid for single-output layers.");
    }
    Optional<Tensor> featureOutput = inferFeatureOutputTensor(0);
    assert(featureOutput.isPresent());
    return featureOutput;
}

Optional<Tensor> CustomLayer::createErrorOutputTensor(bool backPropagateError, uint32_t connectionNumber) {
    if (!backPropagateError || isInferenceOnly()) {
        return Optional<Tensor>::empty();
    }

    assert(connectionNumber < featureInputs.size());
    assert(featureInputs[connectionNumber].isPresent());
    return featureInputs[connectionNumber].get().clone();
}

Optional<Tensor> CustomLayer::connectToPreviousLayer(
    Layer* previousLayer, Optional<Tensor> featureInput, Stream stream, bool backPropagateError, int connectionType) {
    assert(!compiled);
    ensurePortStorageAllocated();

    if (connectionType < 0 || static_cast<size_t>(connectionType) >= inputNames.size()) {
        throw runtime_error("CustomLayer input connection type out of range.");
    }
    const uint32_t inputPort = static_cast<uint32_t>(connectionType);

    assert(featureInput.isPresent());
    assert(previousLayers[inputPort].isEmpty());
    assert(featureInputs[inputPort].isEmpty());
    assert(errorOutputs[inputPort].isEmpty());

    previousLayers[inputPort] = previousLayer;
    featureInputs[inputPort] = featureInput;
    featureInputsConnectedForPorts[inputPort] = featureInput;
    streams[inputPort] = stream;
    errorOutputs[inputPort] = createErrorOutputTensor(backPropagateError, inputPort);
    errorOutputsConnectedForPorts[inputPort] = errorOutputs[inputPort];

    ensureNoDeviceCrossing(placement);
    return errorOutputs[inputPort];
}

void CustomLayer::connectToNextLayer(Layer* nextLayer, int driverConnectionType, int loaderConnectionType) {
    assert(!compiled);
    ensurePortStorageAllocated();

    if (driverConnectionType < 0 || static_cast<size_t>(driverConnectionType) >= outputNames.size()) {
        throw runtime_error("CustomLayer output connection type out of range.");
    }
    const uint32_t outputPort = static_cast<uint32_t>(driverConnectionType);

    if (featureOutputs[outputPort].isEmpty()) {
        Optional<Tensor> outputTensor = inferFeatureOutputTensor(outputPort);
        assert(outputTensor.isPresent());
        featureOutputs[outputPort] = outputTensor;
        featureOutputsConnectedForPorts[outputPort] = outputTensor;
    }

    nextLayers[outputPort] = nextLayer;
    errorInputs[outputPort] = nextLayer->connectToPreviousLayer(
        this, featureOutputs[outputPort], computeStream(), shouldConnectToBackPropErrorIn() && !isBackPropStub(), loaderConnectionType);
    errorInputsConnectedForPorts[outputPort] = errorInputs[outputPort];

    // If there is no downstream backward path for any output port, clear all upstream error outputs.
    bool anyDownstreamBackprop = false;
    for (const auto& errorInput : errorInputs) {
        if (errorInput.isPresent()) {
            anyDownstreamBackprop = true;
            break;
        }
    }

    if (!anyDownstreamBackprop) {
        for (uint32_t inputPort = 0; inputPort < errorOutputs.size(); ++inputPort) {
            if (errorOutputs[inputPort].isPresent() && previousLayers[inputPort].isPresent()) {
                previousLayers[inputPort].get()->replaceErrorInput(errorOutputs[inputPort], Optional<Tensor>::empty());
                errorOutputs[inputPort].clear();
                errorOutputsConnectedForPorts[inputPort].clear();
            }
        }
    }

    ensureNoDeviceCrossing(placement);
}

void CustomLayer::replaceErrorInput(Optional<Tensor> oldErrorInput, Optional<Tensor> newErrorInput) {
    assert(oldErrorInput.isPresent());

    bool replacementHappened = false;
    for (uint32_t outputPort = 0; outputPort < errorInputs.size(); ++outputPort) {
        if (errorInputs[outputPort].isEmpty() || errorInputs[outputPort].get() != oldErrorInput.get()) {
            continue;
        }
        errorInputs[outputPort] = newErrorInput;
        errorInputsConnectedForPorts[outputPort] = newErrorInput;
        replacementHappened = true;
    }
    assert(replacementHappened);

    bool anyDownstreamBackprop = false;
    for (const auto& errorInput : errorInputs) {
        if (errorInput.isPresent()) {
            anyDownstreamBackprop = true;
            break;
        }
    }

    if (!anyDownstreamBackprop) {
        for (uint32_t inputPort = 0; inputPort < errorOutputs.size(); ++inputPort) {
            if (errorOutputs[inputPort].isPresent()) {
                if (previousLayers[inputPort].isPresent()) {
                    previousLayers[inputPort].get()->replaceErrorInput(errorOutputs[inputPort], Optional<Tensor>::empty());
                }
                errorOutputs[inputPort].clear();
                errorOutputsConnectedForPorts[inputPort].clear();
            }
        }
    }

    clearBackwardArrivalBookkeeping();
}

void CustomLayer::synchronizeComputeStreamForForwardInputs() {
    Stream& runStream = computeStream();
    const uint32_t runPort = primaryInputPort();
    for (uint32_t inputPort = 0; inputPort < streams.size(); ++inputPort) {
        if (inputPort == runPort || featureInputs[inputPort].isEmpty()) {
            continue;
        }
        Event readyEvent = streams[inputPort].putEvent();
        runStream.waitEvent(readyEvent);
    }
}

void CustomLayer::forward(Optional<Tensor> featureInput, bool validationPass, uint32_t batchSize) {
    assert(running);
    assert(featureInput.isPresent());

    uint32_t inputPort = inputNames.size();
    for (uint32_t i = 0; i < featureInputs.size(); ++i) {
        if (featureInputs[i].isPresent() && featureInputs[i].get() == featureInput.get()) {
            inputPort = i;
            break;
        }
    }
    assert(inputPort != inputNames.size());

    if (isStartOfForward) {
        if (weightsAreUpToDateEvent.isPresent()) {
            for (const Stream& dataStream : uniqueDataStreams) {
                dataStream.waitEvent(weightsAreUpToDateEvent);
            }
        }
        weightsAreUpToDateEvent.clear();
        isStartOfForward = false;
        isStartOfBackward = true;
        clearForwardArrivalBookkeeping();
    }

    assert(stillWaitingForForwardInputTensorIds.count(featureInput.get().getTensorId()) == 1);
    stillWaitingForForwardInputTensorIds.erase(featureInput.get().getTensorId());

    if (!stillWaitingForForwardInputTensorIds.empty()) {
        return;
    }
    stillWaitingForForwardInputTensorIds = allForwardInputTensorIds;

    synchronizeComputeStreamForForwardInputs();
    computeFeatureOut(0);

    for (uint32_t outputPort = 0; outputPort < outputNames.size(); ++outputPort) {
        if (nextLayers[outputPort].isEmpty())
            continue;
        nextLayers[outputPort].get()->forward(featureOutputs[outputPort], validationPass, batchSize);
    }
}

void CustomLayer::backward(Optional<Tensor> errorInput, uint32_t batchSize) {
    assert(running);

    if (errorInput.isEmpty())
        return;

    uint32_t outputPort = outputNames.size();
    for (uint32_t i = 0; i < errorInputs.size(); ++i) {
        if (errorInputs[i].isPresent() && errorInputs[i].get() == errorInput.get()) {
            outputPort = i;
            break;
        }
    }
    assert(outputPort != outputNames.size());

    bool clearGradientFirst = false;
    if (isStartOfBackward) {
        clearBackwardArrivalBookkeeping();
        isStartOfBackward = false;  // important
        clearGradientFirst = true;  // first backward arrival in this pass
    }

    assert(stillWaitingForBackwardErrorInputTensorIds.count(errorInput.get().getTensorId()) == 1);
    stillWaitingForBackwardErrorInputTensorIds.erase(errorInput.get().getTensorId());

    if (!stillWaitingForBackwardErrorInputTensorIds.empty()) {
        return;
    }
    stillWaitingForBackwardErrorInputTensorIds = allBackwardErrorInputTensorIds;

    Optional<Event> errorInputReadyEvent = Optional<Event>::empty();
    if (gradientUpdateStream.isPresent()) {
        errorInputReadyEvent = computeStream().putEvent();
    }

    if (backwardErrorStamped != nullptr) {
        Optional<Event> errorOutHasBeenComputedEvent = computeErrorOut(0);
        if (errorOutHasBeenComputedEvent.isPresent()) {
            errorOutHasBeenComputedEvents.push_back(errorOutHasBeenComputedEvent);
        }
    }

    if (gradientUpdateStream.isPresent() && errorInputReadyEvent.isPresent()) {
        gradientUpdateStream.get().waitEvent(errorInputReadyEvent);
    }
    accumulateWeightsGradient(0, clearGradientFirst);

    const bool gradientComplete = true;
    if (gradientComplete) {
        weightsAreUpToDateEvent.clear();

        if (gradientUpdateStream.isPresent()) {
            for (const Event& eOutComputedEvent : errorOutHasBeenComputedEvents) {
                gradientUpdateStream.get().waitEvent(eOutComputedEvent);
            }

            bool anyWeightsUpdated = false;
            for (const auto& parameter : parameters) {
                anyWeightsUpdated |= parameter->applyGradient(batchSize);
            }
            if (anyWeightsUpdated) {
                weightsAreUpToDateEvent = gradientUpdateStream.get().putEvent();
            }
        }
        errorOutHasBeenComputedEvents.clear();
        isStartOfForward = true;
    }

    for (uint32_t inputPort = 0; inputPort < inputNames.size(); ++inputPort) {
        if (previousLayers[inputPort].isEmpty() || errorOutputs[inputPort].isEmpty()) {
            continue;
        }
        previousLayers[inputPort].get()->backward(errorOutputs[inputPort], batchSize);
    }
}

void CustomLayer::computeFeatureOut(uint32_t connectionNumber) {
    (void)connectionNumber;
    if (!forwardStamped) {
        throw runtime_error("CustomLayer::computeFeatureOut requires a stamped forward plan.");
    }
    forwardStamped->run();
}

Optional<Event> CustomLayer::computeErrorOut(uint32_t connectionNumber) {
    (void)connectionNumber;
    if (backwardErrorStamped == nullptr) {
        return Optional<Event>::empty();
    }
    backwardErrorStamped->run();
    return computeStream().putEvent();
}

void CustomLayer::accumulateWeightsGradient(uint32_t connectionNumber, bool clearGradientFirst) {
    (void)connectionNumber;
    if (!gradientUpdateStream.isPresent()) {
        return;
    }

    if (clearGradientFirst) {
        if (backwardWeightsClearStamped != nullptr) {
            backwardWeightsClearStamped->run();
        }
        return;
    }

    if (backwardWeightsAccumulateStamped != nullptr) {
        backwardWeightsAccumulateStamped->run();
    }
}

uint64_t CustomLayer::flopCountForward() { return forwardStamped == nullptr ? 0 : forwardStamped->flopCount(); }

uint64_t CustomLayer::flopCountBackward() {
    uint64_t flops = 0;
    if (backwardErrorStamped != nullptr) {
        flops += backwardErrorStamped->flopCount();
    }
    if (backwardWeightsAccumulateStamped != nullptr) {
        flops += backwardWeightsAccumulateStamped->flopCount();
    }
    return flops;
}

bool CustomLayer::isBackPropStub() {
    for (const auto& errorOutput : errorOutputs) {
        if (errorOutput.isPresent()) {
            return false;
        }
    }
    return true;
}

}  // namespace ThorImplementation
