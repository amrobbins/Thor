#include "DeepLearning/Implementation/Layers/NeuralNetwork/CustomLayer.h"

#include <stdexcept>

using namespace std;

namespace ThorImplementation {

CustomLayer::CustomLayer(
    Expression expr, const string& inputName, vector<shared_ptr<Parameter>> parameters, int deviceNum, bool useFastMath, int64_t stampedId)
    : TrainableWeightsBiasesLayer(false, stampedId),
      expr(std::move(expr)),
      inputName(inputName),
      deviceNum(deviceNum),
      useFastMath(useFastMath),
      featureInName(inputName),
      errorOutName(inputName + "_grad") {
    set<string> expressionInputs = expr.getInputNames();
    if (inputName.length() >= 2 && inputName[0] == '_' && inputName[1] == '_')
        throw runtime_error("Custom layer input names cannot start with __ that is reserved. Input name " + name + " is illegal.");
    if (inputName.empty())
        throw runtime_error("Custom layer input sent empty name");
    if (!expressionInputs.contains(inputName)) {
        string expectedInputNamesString;
        for (const auto& name : expressionInputs)
            expectedInputNamesString += name + " ";
        throw runtime_error("Provided input name: " + inputName +
                            " is not part of the expression, the expression expects: " + expectedInputNamesString);
    }
    for (const auto& param : parameters) {
        string paramName = param->getName();
        if (!expressionInputs.contains(paramName)) {
            string expectedInputNamesString;
            for (const auto& name : expressionInputs)
                expectedInputNamesString += name + " ";
            throw runtime_error("Provided parameter name: " + paramName +
                                " is not part of the expression, the expression expects: " + expectedInputNamesString);
        }
        addParam(param);  // verifies name uniqueness
        parameterNames.push_back(paramName);
    }
    const uint32_t numInputsSent = parameters.size() + 1;
    if (numInputsSent != expressionInputs.size()) {
        string expectedInputNamesString;
        for (const auto& name : expressionInputs)
            expectedInputNamesString += name + " ";
        string actualInputNamesString = inputName + " ";
        for (const auto& param : parameters) {
            expectedInputNamesString += param->getName() + " ";
        }
        throw runtime_error("Wrong number of inputs and parameters for the expression, sent " + to_string(numInputsSent) + " expected " +
                            to_string(expressionInputs.size()) + ". Expected inputs: " + expectedInputNamesString +
                            " Actual inputs: " + actualInputNamesString);
    }
}

void CustomLayer::compileImpl() { TrainableWeightsBiasesLayer::compileImpl(); }

void CustomLayer::stampForward(Tensor featureInput) {
    batchSize = featureInput.getDescriptor().getDimensions()[0];

    forwardEq = std::make_shared<FusedEquation>(FusedEquation::compile(expr.expression(), deviceNum, useFastMath));
    // V1 Assumption: Exactly 1 input. V2 could be multiple or none even.
    forwardInputs = buildForwardInputs(featureInput);
    forwardStamped = make_shared<StampedExecutionPlan>(forwardEq->stamp(forwardInputs, streams[0]));
}

void CustomLayer::stampBackward(Tensor featureInput, Tensor errorInput) {
    backwardEq = nullptr;
    if (!isInferenceOnly()) {
        // Then compile back prop, when there is a needed gradient (e.g. errorOut or parameter update)
        vector<string> backwardTargets;

        if (!isBackPropStub()) {
            // Compute error out
            backwardTargets.push_back(inputName);
        }
        for (const auto& param : parameters) {
            if (param->isTrainingEnabled()) {
                backwardTargets.push_back(param->getName());
            }
        }
        if (!backwardTargets.empty()) {
            unordered_map<string, string> errorOutputNameToErrorInputName;
            errorOutputNameToErrorInputName[errorOutName] = errorInName;
            backwardEq = std::make_shared<FusedEquation>(forwardEq->compileBackward(backwardTargets, errorOutputNameToErrorInputName));
            backwardInputs = buildBackwardInputs(featureInput, errorInputs[0]);
            // Backward needs to use the optimizers gradient update stream, since it will update the gradients and error out in one shot
            // So that the gradients are computed prior to optimizer->computeWeightsUpdate,
            // where CustomLayer::computeWeightsGradient is a no-op.
            backwardStamped =
                make_shared<StampedExecutionPlan>(backwardEq->stamp(backwardInputs, optimizer.get()->getGradientUpdateStream()));
        }
    }
}

// Note: A featureInput is guaranteed to be connected before createFeatureOutputTensor() is called.
Optional<Tensor> CustomLayer::createFeatureOutputTensor() {
    // Feature input is already connected
    stampForward(featureInputs[0]);
    // Error input is not yet connected, since it needs to be shaped like the feature output that I am creating here.
    return forwardStamped->output(featureOutName);
}

Optional<Tensor> CustomLayer::createErrorOutputTensor(bool backPropagateError, uint32_t connectionNumber) {
    if (backPropagateError && !isInferenceOnly()) {
        assert(errorInputs.size() > connectionNumber);
        assert(errorInputs[connectionNumber].isPresent());
        stampBackward(featureInputs[connectionNumber], errorInputs[connectionNumber]);
        return backwardStamped->output(errorOutName);
    } else {
        return Optional<Tensor>::empty();
    }
}

unordered_map<string, Tensor> CustomLayer::buildForwardInputs(const Tensor& dataIn) {
    unordered_map<string, Tensor> inputs;
    inputs[inputName] = dataIn;
    for (const auto& paramName : parameterNames) {
        inputs[paramName] = getParamStorage(paramName);
    }
    return inputs;
}

unordered_map<string, Tensor> CustomLayer::buildBackwardInputs(const Tensor& dataIn, const Tensor& errorIn) {
    unordered_map<string, Tensor> inputs = buildForwardInputs(dataIn);
    inputs[errorInName] = errorIn;
    return inputs;
}

void CustomLayer::infer(Optional<Tensor> inputTensor,
                        Optional<Tensor> outputTensor,
                        Stream stream,
                        unsigned int connectionNumber,
                        Tensor weights,
                        Optional<Tensor> biases) {
    // V1 Assumption: Not multiple connections. V3 maybe multiple connections
    //    - but that complicates matters cause each connection can be multiple tensors.
    (void)connectionNumber;
    (void)weights;
    (void)biases;

    if (outputTensor.isEmpty())
        throw runtime_error("CustomLayer::infer requires an output tensor.");

    // V1 Assumption: Exactly 1 input. V2 could be multiple or none even.
    if (inputTensor.isEmpty())
        throw runtime_error("V1 CustomLayer::infer requires an input tensor.");

    forwardStamped->run();
}

void CustomLayer::backProp(Optional<Tensor> dataIn,
                           Optional<Tensor> errorIn,
                           Optional<Tensor> errorOut,
                           Stream dataStream,
                           unsigned int connectionNumber,
                           bool accumulateGradient) {
    (void)connectionNumber;
    (void)accumulateGradient;

    if (!isInferenceOnly()) {
        assert(optimizer.isPresent());
        if (errorIn.isEmpty())
            throw runtime_error("CustomLayer::backProp requires upstream gradient tensor.");
        assert(dataIn.isPresent());
        assert(errorOut.isPresent());

        // * grad stream is now synced with data stream
        // Computes all gradients, so weights update and error out, on the gradient update stream.
        backwardStamped->run();

        // backward() syncs gradient stream with data stream prior to calling this to ensure error in is ready at end of gradient stream
        optimizer.get()->computeWeightsUpdate(dataIn, errorIn, accumulateGradient);
        // FIXME: Above optimizer is going to use its own weightsGradient tensor into which it expects computeWeightsUpdate to populate
        //        the gradient. That assumption no longer holds now that I have N parameters.

        // * grad stream is now running to compute weights update

        // Now at the end of gradientUpdateStream errorOut and gradients are ready from the updates for this connection.

        // Upon processing the last connection, schedule the update to the weights memory.
        if (stillWaitingForErrorInputTensors.empty()) {
            optimizer.get()->updateWeights(weights, biases, batchSize);
        }

        // weights will be updated at the current end of the gradientUpdateStream
        // so Forward() must wait until gradientUpdateStream is finished.
        // This is accomplished in TrainableWeightsBiasesLayer::forward().
    }
}

}  // namespace ThorImplementation
