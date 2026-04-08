#include "DeepLearning/Implementation/Layers/NeuralNetwork/CustomLayer.h"

#include <stdexcept>

using namespace std;

namespace ThorImplementation {

CustomLayer::CustomLayer(DynamicExpression expr,
                         const string& inputName,
                         const vector<shared_ptr<Parameter>>& parameters,
                         int deviceNum,
                         bool useFastMath,
                         int64_t stampedId)
    : TrainableLayer(stampedId),
      layerDefinitionExpression(std::move(expr)),
      inputName(inputName),
      deviceNum(deviceNum),
      useFastMath(useFastMath),
      featureInName(inputName),
      errorOutName(inputName + "_grad") {
    if (inputName.empty())
        throw runtime_error("Custom layer input sent empty name");

    if (inputName.length() >= 2 && inputName[0] == '_' && inputName[1] == '_')
        throw runtime_error("Custom layer input names cannot start with __ that is reserved. Input name " + inputName + " is illegal.");

    for (const auto& param : parameters) {
        const string& paramName = param->getName();
        if (paramName.empty())
            throw runtime_error("Custom layer parameter name cannot be empty.");

        if (paramName.length() >= 2 && paramName[0] == '_' && paramName[1] == '_')
            throw runtime_error("Custom layer parameter names cannot start with __ that is reserved. Parameter name " + paramName +
                                " is illegal.");

        addParam(param);  // verifies name uniqueness
    }
}

void CustomLayer::compileImpl() {
    TrainableLayer::compileImpl();

    // This layer always fuses eout grad and w grad
    wGradFusedWithEOutGrad = true;

    forwardInputsByConnection.clear();
    forwardPreparedByConnection.clear();
    forwardStampedByConnection.clear();

    backwardErrorStampedByConnection.clear();
    backwardWeightsClearStampedByConnection.clear();
    backwardWeightsAccumulateStampedByConnection.clear();
    backwardOutputsByConnection.clear();

    // Forward stamps: require both input and output tensors for the connection.
    const uint32_t numForwardConnections = static_cast<uint32_t>(featureInputs.size());
    for (uint32_t connectionNumber = 0; connectionNumber < numForwardConnections; ++connectionNumber) {
        if (featureInputs[connectionNumber].isPresent() && featureOutputs.size() > connectionNumber &&
            featureOutputs[connectionNumber].isPresent()) {
            stampForward(connectionNumber);
        }
    }

    if (isInferenceOnly() || isBackPropStub()) {
        return;
    }

    // Backward stamps: stamp for every connection that has a feature input and either
    // an incoming error or an outgoing error tensor. stampBackward() already handles
    // empty errorInput/errorOutput cases appropriately.
    const uint32_t numBackwardConnections = static_cast<uint32_t>(featureInputs.size());
    for (uint32_t connectionNumber = 0; connectionNumber < numBackwardConnections; ++connectionNumber) {
        if (!featureInputs[connectionNumber].isPresent()) {
            continue;
        }

        const bool hasErrorInput = errorInputs.size() > connectionNumber && errorInputs[connectionNumber].isPresent();
        const bool hasErrorOutput = errorOutputs.size() > connectionNumber && errorOutputs[connectionNumber].isPresent();

        if (hasErrorInput || hasErrorOutput) {
            stampBackward(connectionNumber);
        }
    }
}

std::unordered_map<std::string, Tensor> CustomLayer::buildForwardInputs(const Tensor& dataIn) {
    std::unordered_map<std::string, Tensor> inputs;
    inputs[inputName] = dataIn;

    for (const auto& param : parameters) {
        inputs[param->getName()] = param->getStorage();
    }

    return inputs;
}

// When connection number is set to UINT32_MAX, discard the stamp but return the output tensor.
Optional<Tensor> CustomLayer::stampForward(uint32_t connectionNumber) {
    Optional<Tensor> featureInput;
    Optional<Tensor> featureOutput;
    Stream stream;

    if (connectionNumber == UINT32_MAX) {
        featureInput = getFirstPresentTensor(featureInputs);
        assert(featureInput.isPresent());
        assert(!streams.empty());
        stream = streams[0];
    } else {
        assert(connectionNumber == forwardInputsByConnection.size());
        assert(connectionNumber == forwardPreparedByConnection.size());
        assert(connectionNumber == forwardStampedByConnection.size());
        assert(featureInputs[connectionNumber].isPresent());
        assert(featureOutputs[connectionNumber].isPresent());

        featureInput = featureInputs[connectionNumber];
        featureOutput = featureOutputs[connectionNumber];
        stream = streams[connectionNumber];
    }

    if (featureOutput.isPresent()) {
        forwardInputsByConnection.push_back(buildForwardInputs(featureInput.get()));

        forwardPreparedByConnection.push_back(
            std::make_shared<PreparedDynamicExpression>(layerDefinitionExpression.prepare(forwardInputsByConnection.back(), stream)));

        std::unordered_map<std::string, Tensor> forwardPreallocatedOutputs;
        forwardPreallocatedOutputs[featureOutName] = featureOutput.get();

        forwardStampedByConnection.push_back(
            std::make_shared<StampedExecutionPlan>(forwardPreparedByConnection.back()->stamp(forwardPreallocatedOutputs)));

        validatePreparedExpressionInputs(*forwardPreparedByConnection.back());

        return Optional<Tensor>::empty();
    } else {
        std::unordered_map<std::string, Tensor> forwardInputs = buildForwardInputs(featureInput.get());
        PreparedDynamicExpression forwardPrepared = layerDefinitionExpression.prepare(forwardInputs, stream);
        validatePreparedExpressionInputs(forwardPrepared);
        StampedExecutionPlan forwardStamped = forwardPrepared.stamp();
        return forwardStamped.output(featureOutName);
    }
}

Optional<Tensor> CustomLayer::stampBackward(uint32_t connectionNumber) {
    if (isInferenceOnly() || isBackPropStub()) {
        return Optional<Tensor>::empty();
    }

    Optional<Tensor> errorInput;
    PreparedDynamicExpression* preparedBackwardSource = nullptr;
    std::optional<PreparedDynamicExpression> transientPrepared;

    if (connectionNumber == UINT32_MAX) {
        Optional<Tensor> featureInput = getFirstPresentTensor(featureInputs);
        errorInput = getFirstPresentTensor(errorInputs);
        assert(featureInput.isPresent());
        assert(errorInput.isPresent());
        assert(!streams.empty());

        std::unordered_map<std::string, Tensor> forwardInputs = buildForwardInputs(featureInput.get());
        transientPrepared.emplace(layerDefinitionExpression.prepare(forwardInputs, streams[0]));
        validatePreparedExpressionInputs(*transientPrepared);
        preparedBackwardSource = &(*transientPrepared);
    } else {
        assert(featureInputs.size() > connectionNumber);
        assert(featureInputs[connectionNumber].isPresent());
        assert(errorInputs.size() > connectionNumber);
        assert(connectionNumber < streams.size());
        assert(connectionNumber < forwardPreparedByConnection.size());
        assert(forwardPreparedByConnection[connectionNumber] != nullptr);

        errorInput = errorInputs[connectionNumber];
        preparedBackwardSource = forwardPreparedByConnection[connectionNumber].get();
    }

    Optional<Tensor> errorOutput;
    if (connectionNumber != UINT32_MAX && errorOutputs.size() > connectionNumber) {
        errorOutput = errorOutputs[connectionNumber];
    }

    std::vector<std::string> errorTargets = {inputName};

    std::vector<std::string> parameterTargets;
    for (const auto& param : parameters) {
        if (param->isTrainable()) {
            parameterTargets.push_back(param->getName());
        }
    }

    std::unordered_map<std::string, std::string> upstreamInputNamesByOutput;
    upstreamInputNamesByOutput[featureOutName] = errorInName;

    PreparedDynamicExpression::TensorMap backwardAdditionalInputs;
    if (errorInput.isPresent()) {
        backwardAdditionalInputs[errorInName] = errorInput.get();
    }

    // Transient shape/materialization path for createErrorOutputTensor().
    // This path is only used to obtain the error output tensor, so keep it minimal.
    if (connectionNumber == UINT32_MAX) {
        StampedExecutionPlan backwardErrorStamped = preparedBackwardSource->stampBackward(errorTargets,
                                                                                          upstreamInputNamesByOutput,
                                                                                          /*accumulate_grad_outputs=*/false,
                                                                                          backwardAdditionalInputs);

        return backwardErrorStamped.output(errorOutName);
    }

    if (connectionNumber >= backwardErrorStampedByConnection.size()) {
        backwardErrorStampedByConnection.resize(connectionNumber + 1);
        backwardWeightsClearStampedByConnection.resize(connectionNumber + 1);
        backwardWeightsAccumulateStampedByConnection.resize(connectionNumber + 1);
        backwardOutputsByConnection.resize(connectionNumber + 1);
    }

    auto& backwardOutputs = backwardOutputsByConnection[connectionNumber];
    backwardOutputs.clear();

    // Main backward stamp always owns:
    //   - error out when errorOutput is present
    //   - clear-first parameter grads when trainable params exist
    if (errorInput.isPresent() && (errorOutput.isPresent() || !parameterTargets.empty())) {
        std::vector<std::string> mainTargets;

        if (errorOutput.isPresent()) {
            mainTargets.push_back(inputName);
        }

        for (const std::string& parameterTarget : parameterTargets) {
            mainTargets.push_back(parameterTarget);
        }

        PreparedDynamicExpression::TensorMap backwardMainPreallocatedOutputs;
        if (errorOutput.isPresent()) {
            backwardMainPreallocatedOutputs[errorOutName] = errorOutput.get();
        }

        backwardErrorStampedByConnection[connectionNumber] =
            std::make_shared<StampedExecutionPlan>(preparedBackwardSource->stampBackward(mainTargets,
                                                                                         upstreamInputNamesByOutput,
                                                                                         /*accumulate_grad_outputs=*/false,
                                                                                         backwardAdditionalInputs,
                                                                                         {},
                                                                                         backwardMainPreallocatedOutputs));

        for (const std::string& parameterTarget : parameterTargets) {
            const std::string gradName = parameterTarget + "_grad";
            backwardOutputs[gradName] = backwardErrorStampedByConnection[connectionNumber]->output(gradName);
        }
    } else {
        if (errorOutput.isPresent()) {
            errorOutput.get().memsetAsync(streams[connectionNumber], 0);
        }
        backwardErrorStampedByConnection[connectionNumber] = nullptr;
    }

    // Clear-first parameter grad computation is always fused into the main backward stamp.
    backwardWeightsClearStampedByConnection[connectionNumber] = nullptr;

    // Separate accumulate-only stamp for later additive contributions to weight grads.
    if (!parameterTargets.empty() && errorInput.isPresent()) {
        backwardWeightsAccumulateStampedByConnection[connectionNumber] =
            std::make_shared<StampedExecutionPlan>(preparedBackwardSource->stampBackward(parameterTargets,
                                                                                         upstreamInputNamesByOutput,
                                                                                         /*accumulate_grad_outputs=*/true,
                                                                                         backwardAdditionalInputs,
                                                                                         {},
                                                                                         backwardOutputs));
    } else {
        backwardWeightsAccumulateStampedByConnection[connectionNumber] = nullptr;
    }

    return Optional<Tensor>::empty();
}

// Note: A featureInput is guaranteed to be connected before createFeatureOutputTensor() is called.
Optional<Tensor> CustomLayer::createFeatureOutputTensor() {
    assert(!featureInputs.empty());
    Optional<Tensor> aFeatureInput = getFirstPresentTensor(featureInputs);
    assert(aFeatureInput.isPresent());

    Optional<Tensor> aFeatureOutput = getFirstPresentTensor(featureOutputs);
    if (aFeatureOutput.isPresent()) {
        return aFeatureOutput.get().clone();
    } else {
        Optional<Tensor> featureOutput = stampForward(UINT32_MAX);
        assert(featureOutput.isPresent());
        return featureOutput;
    }
}

Optional<Tensor> CustomLayer::createErrorOutputTensor(bool backPropagateError, uint32_t connectionNumber) {
    if (!backPropagateError || isInferenceOnly() || isBackPropStub()) {
        return Optional<Tensor>::empty();
    }

    assert(featureInputs.size() > connectionNumber);
    assert(featureInputs[connectionNumber].isPresent());
    assert(errorInputs.size() > connectionNumber);

    if (errorInputs[connectionNumber].isEmpty()) {
        return Optional<Tensor>::empty();
    }

    assert(errorInputs[connectionNumber].isPresent());

    Optional<Tensor> errorOutput = getFirstPresentTensor(errorOutputs);
    if (errorOutput.isPresent()) {
        return errorOutput.get().clone();
    } else {
        return stampBackward(UINT32_MAX);
    }
}

void CustomLayer::computeFeatureOut(uint32_t connectionNumber) {
    if (featureOutputs.empty())
        throw runtime_error("CustomLayer::infer requires an output tensor.");
    if (featureOutputs[connectionNumber].isEmpty())
        throw runtime_error("CustomLayer::infer requires a present output tensor.");

    if (featureInputs.empty())
        throw runtime_error("CustomLayer::infer requires an input tensor.");
    if (featureInputs[connectionNumber].isEmpty())
        throw runtime_error("CustomLayer::infer requires a present input tensor.");

    assert(connectionNumber < forwardStampedByConnection.size());
    assert(forwardStampedByConnection[connectionNumber] != nullptr);

    forwardStampedByConnection[connectionNumber]->run();
}

// Error in is up-to-date by the end of the data stream.
Optional<Event> CustomLayer::computeErrorOut(uint32_t connectionNumber) {
    assert(connectionNumber < errorInputs.size());

    if (errorInputs[connectionNumber].isEmpty()) {
        // No incoming gradient, potentially a StopGradientLayer was put there.
        // In that case there is no backward work to run for this connection.
        return Optional<Event>::empty();
    }

    // If for some reason we do not have the stamp for this backward connection,
    // then the contract is that it was decided that this backward connection is a no-op.
    if (connectionNumber >= backwardErrorStampedByConnection.size()) {
        return Optional<Event>::empty();
    }
    if (backwardErrorStampedByConnection[connectionNumber] == nullptr) {
        return Optional<Event>::empty();
    }

    // Run the main backward stamp, which computes error-out gradients and
    // also performs the clear-first weight-gradient computation when applicable.
    backwardErrorStampedByConnection[connectionNumber]->run();
    assert(connectionNumber < streams.size());
    return streams[connectionNumber].putEvent();
}

// Error in is up-to-date by the end of the data stream.
// Gradient update stream must wait for that.
void CustomLayer::accumulateWeightsGradient(uint32_t connectionNumber, bool clearGradientFirst) {
    // The initial non-accumulating weight-grad write is always done by the
    // main backward stamp in computeErrorOut(), so there is nothing to do
    // in the clear then accumulate case.
    if (clearGradientFirst) {
        return;
    }

    assert(connectionNumber < errorInputs.size());
    if (errorInputs[connectionNumber].isEmpty()) {
        // No incoming gradient, potentially a StopGradientLayer was put there.
        // So there is no gradient work to be done for this connection.
        return;
    }

    // Accumulation pass for later additive contributions to weight grads.
    if (connectionNumber >= backwardWeightsAccumulateStampedByConnection.size()) {
        return;
    }
    if (backwardWeightsAccumulateStampedByConnection[connectionNumber] == nullptr) {
        return;
    }

    assert(gradientUpdateStream.isPresent());
    assert(connectionNumber < streams.size());
    gradientUpdateStream.get().waitEvent(streams[connectionNumber].putEvent());
    backwardWeightsAccumulateStampedByConnection[connectionNumber]->run();
}

void CustomLayer::validatePreparedExpressionInputs(const PreparedDynamicExpression& prepared) {
    std::set<std::string> expectedInputNames;
    expectedInputNames.insert(inputName);
    for (const auto& param : parameters) {
        expectedInputNames.insert(param->getName());
    }

    std::set<std::string> actualInputNames;
    for (const auto& name : prepared.stampInputs() | views::keys) {
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

}  // namespace ThorImplementation
