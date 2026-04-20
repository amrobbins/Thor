#include "DeepLearning/Implementation/Layers/CustomLayer.h"

#include <stdexcept>

using namespace std;

namespace ThorImplementation {

CustomLayer::CustomLayer(DynamicExpression expr,
                         const TensorPlacement& placement,
                         const std::vector<std::shared_ptr<Parameter>>& parameters,
                         bool inferenceOnly,
                         int64_t stampedId,
                         bool useFastMath)
    : TrainableLayer(placement, inferenceOnly, stampedId), layerDefinitionExpression(std::move(expr)), useFastMath(useFastMath) {
    inputName = "feature_input";

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

        addParameter(param);  // verifies name uniqueness
    }
    attachGradientUpdateStream();
}

void CustomLayer::compileImpl() {
    TrainableLayer::compileImpl();

    assert(placement.getMemDevice() == TensorPlacement::MemDevices::GPU);

    // Error-output and parameter-gradient backward work are stamped separately.
    wGradFusedWithEOutGrad = false;

    forwardInputsByConnection.clear();
    forwardPreparedByConnection.clear();
    forwardStampedByConnection.clear();

    backwardErrorStampedByConnection.clear();
    backwardWeightsClearStampedByConnection.clear();
    backwardWeightsAccumulateStampedByConnection.clear();
    backwardOutputsByConnection.clear();

    // A bias-only layer may have no feature input.
    Optional<Tensor> aFeatureInput = getFirstPresentTensor(featureInputs);
    // A CustomLayer with no featureOutput would just waste computation, and doesn't make sense.
    Optional<Tensor> aFeatureOutput = getFirstPresentTensor(featureOutputs);
    assert(aFeatureOutput.isPresent());

    for (const auto& parameter : parameters) {
        if (!parameter->isTrainable())
            continue;
        assert(gradientUpdateStream.isPresent());
    }

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
        if (!param->isStorageInitialized()) {
            param->compileStorageAndOptimizer(dataIn, gradientUpdateStream, isInferenceOnly());
        }

        Optional<Tensor> paramStorage = param->getStorage();
        assert(paramStorage.isPresent());
        inputs[param->getName()] = paramStorage.get();
    }

    return inputs;
}

// When connection number is set to UINT32_MAX, discard the stamp but return the output tensor.
Optional<Tensor> CustomLayer::stampForward(uint32_t connectionNumber) {
    Optional<Tensor> featureInput;
    Optional<Tensor> featureOutput;
    Stream stream;

    PreparedDynamicExpression::TensorMap forwardOutputs;
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
        forwardOutputs[featureOutName] = featureOutput;
        stream = streams[connectionNumber];
    }

    if (featureOutput.isPresent()) {
        forwardInputsByConnection.push_back(buildForwardInputs(featureInput.get()));

        forwardPreparedByConnection.push_back(std::make_shared<PreparedDynamicExpression>(
            layerDefinitionExpression.prepare(forwardInputsByConnection.back(), forwardOutputs, stream)));

        std::unordered_set<std::string> parameterNames;
        for (const auto& parameter : parameters) {
            parameterNames.insert(parameter->getName());
        }
        const auto parameterFanOverrides = forwardPreparedByConnection.front()->getParameterFanOverrides(parameterNames);

        const auto outputDims = featureOutput.get().getDescriptor().getDimensions();
        for (const auto& parameter : parameters) {
            auto it = parameterFanOverrides.find(parameter->getName());
            if (it != parameterFanOverrides.end()) {
                parameter->compileInitializer(it->second.fan_in, it->second.fan_out);
            } else {
                parameter->compileInitializer();
            }
        }

        std::unordered_map<std::string, Tensor> forwardPreallocatedOutputs;
        forwardPreallocatedOutputs[featureOutName] = featureOutput.get();

        forwardStampedByConnection.push_back(
            std::make_shared<StampedExecutionPlan>(forwardPreparedByConnection.back()->stamp(forwardPreallocatedOutputs)));

        validatePreparedExpressionInputs(*forwardPreparedByConnection.back());

        return Optional<Tensor>::empty();
    } else {
        std::unordered_map<std::string, Tensor> forwardInputs = buildForwardInputs(featureInput.get());
        PreparedDynamicExpression forwardPrepared = layerDefinitionExpression.prepare(forwardInputs, forwardOutputs, stream);
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

    PreparedDynamicExpression::TensorMap forwardOutputs;
    Optional<Tensor> featureOutput = featureOutputs[connectionNumber];
    assert(featureOutput.isPresent());
    forwardOutputs[featureOutName] = featureOutput;

    assert(featureInputs.size() > connectionNumber);
    assert(featureInputs[connectionNumber].isPresent());
    assert(errorInputs.size() > connectionNumber);
    assert(connectionNumber < streams.size());
    assert(connectionNumber < forwardPreparedByConnection.size());
    assert(forwardPreparedByConnection[connectionNumber] != nullptr);

    errorInput = errorInputs[connectionNumber];
    preparedBackwardSource = forwardPreparedByConnection[connectionNumber].get();

    Optional<Tensor> errorOutput;
    if (errorOutputs.size() > connectionNumber) {
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

    PreparedDynamicExpression* preparedWeightsBackwardSource = preparedBackwardSource;
    std::optional<PreparedDynamicExpression> gradientPreparedBackwardSource;
    if (!parameterTargets.empty()) {
        assert(gradientUpdateStream.isPresent());
        gradientPreparedBackwardSource.emplace(
            layerDefinitionExpression.prepare(forwardInputsByConnection[connectionNumber], forwardOutputs, gradientUpdateStream.get()));
        validatePreparedExpressionInputs(*gradientPreparedBackwardSource);
        preparedWeightsBackwardSource = &gradientPreparedBackwardSource.value();
    }

    PreparedDynamicExpression::TensorMap backwardAdditionalInputs;
    if (errorInput.isPresent()) {
        backwardAdditionalInputs[errorInName] = errorInput.get();
    }

    if (connectionNumber >= backwardErrorStampedByConnection.size()) {
        backwardErrorStampedByConnection.resize(connectionNumber + 1);
        backwardWeightsClearStampedByConnection.resize(connectionNumber + 1);
        backwardWeightsAccumulateStampedByConnection.resize(connectionNumber + 1);
        backwardOutputsByConnection.resize(connectionNumber + 1);
    }

    auto& backwardOutputs = backwardOutputsByConnection[connectionNumber];
    backwardOutputs.clear();

    // 1. Separate error-output backward stamp on the regular data stream.
    if (errorInput.isPresent() && errorOutput.isPresent()) {
        PreparedDynamicExpression::TensorMap backwardErrorPreallocatedOutputs;
        backwardErrorPreallocatedOutputs[errorOutName()] = errorOutput.get();

        backwardErrorStampedByConnection[connectionNumber] =
            std::make_shared<StampedExecutionPlan>(preparedBackwardSource->stampBackward(std::vector<std::string>{inputName},
                                                                                         upstreamInputNamesByOutput,
                                                                                         /*accumulate_grad_outputs=*/false,
                                                                                         backwardAdditionalInputs,
                                                                                         {},
                                                                                         backwardErrorPreallocatedOutputs));
    } else {
        if (errorOutput.isPresent()) {
            errorOutput.get().memsetAsync(streams[connectionNumber], 0);
        }
        backwardErrorStampedByConnection[connectionNumber] = nullptr;
    }

    // 2/3. Separate parameter-gradient backward stamps on the gradient-update stream:
    //      - clear-first / overwrite existing gradient buffers
    //      - accumulate / add into existing gradient buffers
    PreparedDynamicExpression::TensorMap backwardParameterPreallocatedOutputs;
    for (auto& parameter : parameters) {
        if (!parameter->isTrainable()) {
            continue;
        }

        assert(parameter->hasOptimizer());
        shared_ptr<Optimizer> parameterOptimizer = parameter->getOptimizer();
        assert(parameterOptimizer != nullptr);
        assert(parameterOptimizer->getWeightsGradient().isPresent());

        const std::string gradName = parameter->getName() + "_grad";
        backwardParameterPreallocatedOutputs[gradName] = parameterOptimizer->getWeightsGradient().get();
        backwardOutputs[gradName] = parameterOptimizer->getWeightsGradient().get();
    }

    if (!parameterTargets.empty() && errorInput.isPresent()) {
        backwardWeightsClearStampedByConnection[connectionNumber] =
            std::make_shared<StampedExecutionPlan>(preparedWeightsBackwardSource->stampBackward(parameterTargets,
                                                                                                upstreamInputNamesByOutput,
                                                                                                /*accumulate_grad_outputs=*/false,
                                                                                                backwardAdditionalInputs,
                                                                                                {},
                                                                                                backwardParameterPreallocatedOutputs));

        backwardWeightsAccumulateStampedByConnection[connectionNumber] =
            std::make_shared<StampedExecutionPlan>(preparedWeightsBackwardSource->stampBackward(parameterTargets,
                                                                                                upstreamInputNamesByOutput,
                                                                                                /*accumulate_grad_outputs=*/true,
                                                                                                backwardAdditionalInputs,
                                                                                                {},
                                                                                                backwardOutputs));
    } else {
        backwardWeightsClearStampedByConnection[connectionNumber] = nullptr;
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
    if (!backPropagateError || isInferenceOnly()) {
        return Optional<Tensor>::empty();
    }

    assert(featureInputs.size() > connectionNumber);
    assert(featureInputs[connectionNumber].isPresent());
    return featureInputs[connectionNumber].get().clone();
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

// Error-output backward work runs on the data stream.
Optional<Event> CustomLayer::computeErrorOut(uint32_t connectionNumber, bool clearWeightsGradientFirstIfFused) {
    assert(connectionNumber < errorInputs.size());
    // Custom layer currently never fuses wGrad and eOutGrad
    // FIXME: As an optimization, when there is just 1 stage, then fuse them.
    // Then will need two fused equations, one that clears gradient one that accumulates.
    assert(!wGradFusedWithEOutGrad);

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

    // Run the error-output backward stamp on the regular data stream.
    backwardErrorStampedByConnection[connectionNumber]->run();
    assert(connectionNumber < streams.size());
    return streams[connectionNumber].putEvent();
}

// Gradient update stream synchronization is handled by TrainableLayer::backward().
void CustomLayer::accumulateWeightsGradient(uint32_t connectionNumber, bool clearGradientFirst) {
    assert(connectionNumber < errorInputs.size());
    if (errorInputs[connectionNumber].isEmpty()) {
        // No incoming gradient, potentially a StopGradientLayer was put there.
        // So there is no gradient work to be done for this connection.
        return;
    }

    assert(gradientUpdateStream.isPresent());

    if (clearGradientFirst) {
        if (connectionNumber >= backwardWeightsClearStampedByConnection.size()) {
            return;
        }
        if (backwardWeightsClearStampedByConnection[connectionNumber] == nullptr) {
            return;
        }
        backwardWeightsClearStampedByConnection[connectionNumber]->run();
        return;
    }

    if (connectionNumber >= backwardWeightsAccumulateStampedByConnection.size()) {
        return;
    }
    if (backwardWeightsAccumulateStampedByConnection[connectionNumber] == nullptr) {
        return;
    }

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

uint64_t CustomLayer::flopCountForward() {
    uint64_t flops = 0;
    for (const auto& stamped : forwardStampedByConnection) {
        if (stamped != nullptr) {
            flops += stamped->flopCount();
        }
    }
    return flops;
}

uint64_t CustomLayer::flopCountBackward() {
    uint64_t flops = 0;
    for (const auto& stamped : backwardErrorStampedByConnection) {
        if (stamped != nullptr) {
            flops += stamped->flopCount();
        }
    }
    // Every stage is reported as an accumulation stage, so initially 0 + grad.
    // This remains valid when there is multi-stage accumulation.
    for (const auto& stamped : backwardWeightsAccumulateStampedByConnection) {
        if (stamped != nullptr) {
            flops += stamped->flopCount();
        }
    }
    return flops;
}

}  // namespace ThorImplementation
