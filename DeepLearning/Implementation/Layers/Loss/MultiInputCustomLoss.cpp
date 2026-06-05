#include "DeepLearning/Implementation/Layers/Loss/MultiInputCustomLoss.h"
#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/ThorError.h"

#include "Utilities/Expression/FusedEquation.h"

#include <sstream>
#include <stdexcept>
#include <utility>

using namespace std;

namespace ThorImplementation {

MultiInputCustomLoss::MultiInputCustomLoss(DynamicExpression lossExpression,
                                           DynamicExpression gradientExpression,
                                           vector<string> inputNames,
                                           vector<optional<string>> gradientNames,
                                           string lossName,
                                           DataType lossDataType)
    : lossExpression(std::move(lossExpression)),
      gradientExpression(std::move(gradientExpression)),
      inputNames(std::move(inputNames)),
      gradientNames(std::move(gradientNames)),
      lossName(std::move(lossName)),
      lossDataType(lossDataType) {
    THOR_THROW_IF_FALSE(!this->inputNames.empty());
    THOR_THROW_IF_FALSE(this->inputNames.size() == this->gradientNames.size());
    THOR_THROW_IF_FALSE(!presentNames(this->gradientNames).empty());
    THOR_THROW_IF_FALSE(!this->lossName.empty());
    THOR_THROW_IF_FALSE(this->lossDataType == DataType::FP16 || this->lossDataType == DataType::FP32);

    featureInputs.resize(this->inputNames.size(), std::nullopt);
    errorOutputs.resize(this->inputNames.size(), std::nullopt);
    inputStreams.resize(this->inputNames.size());
    previousLayers.resize(this->inputNames.size(), std::nullopt);
}

set<string> MultiInputCustomLoss::presentNames(const vector<optional<string>>& names) {
    set<string> result;
    for (const optional<string>& name : names) {
        if (name.has_value())
            result.insert(name.value());
    }
    return result;
}

string MultiInputCustomLoss::joinNames(const set<string>& names) {
    if (names.empty())
        return "<none>";
    ostringstream oss;
    bool first = true;
    for (const string& name : names) {
        if (!first)
            oss << ", ";
        oss << name;
        first = false;
    }
    return oss.str();
}

DataType MultiInputCustomLoss::findOutputDType(const shared_ptr<CompiledOutputs>& compiledOutputs, const string& outputName) {
    optional<DataType> outputDType;
    for (const CompiledExecutionStage& stage : compiledOutputs->stages) {
        for (size_t outputIndex = 0; outputIndex < stage.outputs.size(); ++outputIndex) {
            const CompiledStageOutput& output = stage.outputs[outputIndex];
            if (output.name == outputName) {
                outputDType = stage.outputDType(outputIndex);
                break;
            }
        }
        if (outputDType.has_value())
            break;
    }

    if (!outputDType.has_value()) {
        for (const CompiledStageOutput& finalOutput : compiledOutputs->final_outputs) {
            if (finalOutput.name != outputName)
                continue;
            for (const CompiledExecutionStage& stage : compiledOutputs->stages) {
                for (size_t outputIndex = 0; outputIndex < stage.outputs.size(); ++outputIndex) {
                    if (stage.outputs[outputIndex].value_id == finalOutput.value_id) {
                        outputDType = stage.outputDType(outputIndex);
                        break;
                    }
                }
                if (outputDType.has_value())
                    break;
            }
        }
    }

    if (!outputDType.has_value())
        throw runtime_error("MultiInputCustomLoss expression did not infer output dtype for '" + outputName + "'.");
    return outputDType.value();
}

uint32_t MultiInputCustomLoss::requireInputIndexFromConnectionType(int connectionType) const {
    if (connectionType < 0 || static_cast<size_t>(connectionType) >= inputNames.size()) {
        throw runtime_error("MultiInputCustomLoss input connection type is out of range.");
    }
    return static_cast<uint32_t>(connectionType);
}

Stream& MultiInputCustomLoss::computeStream() {
    THOR_THROW_IF_FALSE(!inputStreams.empty());
    THOR_THROW_IF_FALSE(inputStreams.front().isInitialized());
    return inputStreams.front();
}

const Stream& MultiInputCustomLoss::computeStream() const {
    THOR_THROW_IF_FALSE(!inputStreams.empty());
    THOR_THROW_IF_FALSE(inputStreams.front().isInitialized());
    return inputStreams.front();
}

Stream MultiInputCustomLoss::getStream() { return computeStream(); }

MultiInputCustomLoss::TensorMap MultiInputCustomLoss::buildLossInputs() const {
    TensorMap inputs;
    for (size_t i = 0; i < inputNames.size(); ++i) {
        THOR_THROW_IF_FALSE(featureInputs[i].has_value());
        inputs.emplace(inputNames[i], featureInputs[i].value());
    }
    return inputs;
}

MultiInputCustomLoss::TensorMap MultiInputCustomLoss::buildLossOutputs() const {
    THOR_THROW_IF_FALSE(featureOutput.has_value());
    return TensorMap{{lossName, featureOutput.value()}};
}

MultiInputCustomLoss::TensorMap MultiInputCustomLoss::buildGradientOutputs() const {
    TensorMap outputs;
    for (size_t i = 0; i < gradientNames.size(); ++i) {
        if (!gradientNames[i].has_value())
            continue;
        THOR_THROW_IF_FALSE(errorOutputs[i].has_value());
        outputs.emplace(gradientNames[i].value(), errorOutputs[i].value());
    }
    return outputs;
}

void MultiInputCustomLoss::validateExpressionOutputNames(const DynamicExpression& expression,
                                                         const set<string>& expectedOutputNames,
                                                         const string& what) const {
    const vector<string>& expectedOutputs = expression.getExpectedOutputNames();
    if (!expectedOutputs.empty()) {
        const set<string> actual(expectedOutputs.begin(), expectedOutputs.end());
        if (actual != expectedOutputNames) {
            throw runtime_error("MultiInputCustomLoss " + what + " expression output name mismatch. Expected {" +
                                joinNames(expectedOutputNames) + "}, got {" + joinNames(actual) + "}.");
        }
    }
}

pair<vector<uint64_t>, DataType> MultiInputCustomLoss::inferExpressionOutputDescriptor(const DynamicExpression& expression,
                                                                                       const string& outputName,
                                                                                       const string& what) const {
    THOR_THROW_IF_FALSE(computeStream().isInitialized());

    DynamicExpressionBuild build = expression.build(buildLossInputs(), {}, const_cast<Stream&>(computeStream()));

    const vector<string> outputNames = build.equation->getOutputNames();
    const set<string> actualOutputNames(outputNames.begin(), outputNames.end());
    const set<string> expectedOutputNames{outputName};
    if (actualOutputNames != expectedOutputNames) {
        throw runtime_error("MultiInputCustomLoss " + what + " expression output name mismatch. Expected {" +
                            joinNames(expectedOutputNames) + "}, got {" + joinNames(actualOutputNames) + "}.");
    }

    unordered_map<string, vector<uint64_t>> outputShapes = build.equation->getOutputShapes(build.stamp_inputs, build.tensor_scalar_inputs);
    auto shapeIt = outputShapes.find(outputName);
    if (shapeIt == outputShapes.end()) {
        throw runtime_error("MultiInputCustomLoss " + what + " expression did not infer output shape for '" + outputName + "'.");
    }

    shared_ptr<CompiledOutputs> compiledOutputs = build.equation->compileForInputs(build.stamp_inputs, {}, build.tensor_scalar_inputs);
    return {shapeIt->second, findOutputDType(compiledOutputs, outputName)};
}

optional<Tensor> MultiInputCustomLoss::createFeatureOutputTensor() {
    const auto [outputShape, outputDType] = inferExpressionOutputDescriptor(lossExpression, lossName, "loss");
    THOR_THROW_IF_FALSE(outputDType == lossDataType);
    THOR_THROW_IF_FALSE(!featureInputs.empty() && featureInputs.front().has_value());
    return Tensor(featureInputs.front().value().getPlacement(), TensorDescriptor(outputDType, outputShape));
}

optional<Tensor> MultiInputCustomLoss::createErrorOutputTensor(bool backPropagateError) {
    (void)backPropagateError;
    THOR_UNREACHABLE();
    return nullopt;
}

optional<Tensor> MultiInputCustomLoss::connectToPreviousLayer(
    Layer* previousLayer, optional<Tensor> featureInput, Stream stream, bool backPropagateError, int connectionType) {
    THOR_THROW_IF_FALSE(!compiled);
    THOR_THROW_IF_FALSE(featureInput.has_value());
    const uint32_t inputIndex = requireInputIndexFromConnectionType(connectionType);
    THOR_THROW_IF_FALSE(!featureInputs[inputIndex].has_value());
    THOR_THROW_IF_FALSE(!previousLayers[inputIndex].has_value());

    for (size_t i = 0; i < featureInputs.size(); ++i) {
        if (!featureInputs[i].has_value())
            continue;
        THOR_THROW_IF_FALSE(featureInputs[i].value().getPlacement() == featureInput.value().getPlacement());
    }
    if (featureOutput.has_value())
        THOR_THROW_IF_FALSE(featureOutput.value().getPlacement() == featureInput.value().getPlacement());

    previousLayers[inputIndex] = previousLayer;
    featureInputs[inputIndex] = featureInput.value();
    inputStreams[inputIndex] = stream;
    if (gradientNames[inputIndex].has_value() && backPropagateError && !isInferenceOnly())
        errorOutputs[inputIndex] = featureInput.value().clone();
    else
        errorOutputs[inputIndex] = nullopt;

    return errorOutputs[inputIndex];
}

void MultiInputCustomLoss::connectToNextLayer(Layer* nextLayer, int driverConnectionType, int loaderConnectionType) {
    (void)driverConnectionType;
    THOR_THROW_IF_FALSE(!compiled);
    THOR_THROW_IF_FALSE(!this->nextLayer.has_value());
    this->nextLayer = nextLayer;
    if (nextLayer->hasFeatureInput())
        featureOutput = createFeatureOutputTensor();
    else
        featureOutput = nullopt;

    errorInput = nullopt;
    nextLayer->connectToPreviousLayer(this, featureOutput, computeStream(), shouldConnectToBackPropErrorIn() && !isBackPropStub(), loaderConnectionType);

    ensureNoDeviceCrossing();
}

void MultiInputCustomLoss::replaceErrorInput(optional<Tensor> oldErrorInput, optional<Tensor> newErrorInput) {
    THOR_THROW_IF_FALSE(oldErrorInput.has_value());
    bool replaced = false;
    for (size_t i = 0; i < errorOutputs.size(); ++i) {
        if (!errorOutputs[i].has_value() || errorOutputs[i].value() != oldErrorInput.value())
            continue;
        if (previousLayers[i].has_value())
            previousLayers[i].value()->replaceErrorInput(errorOutputs[i], newErrorInput);
        errorOutputs[i] = newErrorInput;
        replaced = true;
    }
    THOR_THROW_IF_FALSE(replaced);
}

void MultiInputCustomLoss::compileImpl() {
    Layer::compileImpl();

    THOR_THROW_IF_FALSE(featureOutput.has_value());
    THOR_THROW_IF_FALSE(featureOutput.value().isInitialized());
    THOR_THROW_IF_FALSE(featureOutput.value().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    THOR_THROW_IF_FALSE(featureOutput.value().getDescriptor().getDataType() == lossDataType);

    set<string> inputNameSet(inputNames.begin(), inputNames.end());
    THOR_THROW_IF_FALSE(inputNameSet.size() == inputNames.size());
    set<string> gradientNameSet = presentNames(gradientNames);
    THOR_THROW_IF_FALSE(!gradientNameSet.empty());

    const vector<string>& lossExpectedInputs = lossExpression.getExpectedInputNames();
    if (!lossExpectedInputs.empty()) {
        set<string> actual(lossExpectedInputs.begin(), lossExpectedInputs.end());
        if (actual != inputNameSet) {
            throw runtime_error("MultiInputCustomLoss loss expression input name mismatch. Expected {" + joinNames(inputNameSet) +
                                "}, got {" + joinNames(actual) + "}.");
        }
    }
    const vector<string>& gradientExpectedInputs = gradientExpression.getExpectedInputNames();
    if (!gradientExpectedInputs.empty()) {
        set<string> actual(gradientExpectedInputs.begin(), gradientExpectedInputs.end());
        if (actual != inputNameSet) {
            throw runtime_error("MultiInputCustomLoss gradient expression input name mismatch. Expected {" + joinNames(inputNameSet) +
                                "}, got {" + joinNames(actual) + "}.");
        }
    }

    validateExpressionOutputNames(lossExpression, {lossName}, "loss");
    validateExpressionOutputNames(gradientExpression, gradientNameSet, "gradient");

    for (size_t i = 0; i < featureInputs.size(); ++i) {
        THOR_THROW_IF_FALSE(featureInputs[i].has_value());
        THOR_THROW_IF_FALSE(featureInputs[i].value().isInitialized());
        THOR_THROW_IF_FALSE(featureInputs[i].value().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        THOR_THROW_IF_FALSE(featureInputs[i].value().getPlacement() == featureOutput.value().getPlacement());
        if (!isInferenceOnly() && gradientNames[i].has_value()) {
            THOR_THROW_IF_FALSE(errorOutputs[i].has_value());
            THOR_THROW_IF_FALSE(errorOutputs[i].value().isInitialized());
            THOR_THROW_IF_FALSE(errorOutputs[i].value().getPlacement() == featureInputs[i].value().getPlacement());
            THOR_THROW_IF_FALSE(errorOutputs[i].value().getDescriptor() == featureInputs[i].value().getDescriptor());
        } else if (!gradientNames[i].has_value()) {
            THOR_THROW_IF_FALSE(!errorOutputs[i].has_value());
        }
    }

    TensorMap inputs = buildLossInputs();
    TensorMap lossOutputs = buildLossOutputs();
    lossPrepared = make_shared<PreparedDynamicExpression>(lossExpression.prepare(inputs, lossOutputs, computeStream()));
    lossPreRunHook = lossPrepared->preForwardHook();
    lossStamped = make_shared<StampedExecutionPlan>(lossPrepared->stamp(lossOutputs));

    if (!isInferenceOnly()) {
        TensorMap gradientOutputs = buildGradientOutputs();
        gradientPrepared = make_shared<PreparedDynamicExpression>(gradientExpression.prepare(inputs, gradientOutputs, computeStream()));
        gradientPreRunHook = gradientPrepared->preForwardHook();
        gradientStamped = make_shared<StampedExecutionPlan>(gradientPrepared->stamp(gradientOutputs));
    } else {
        gradientPrepared.reset();
        gradientStamped.reset();
        gradientPreRunHook = nullptr;
    }
}

void MultiInputCustomLoss::cleanup() {
    lossStamped.reset();
    lossPrepared.reset();
    lossPreRunHook = nullptr;
    gradientStamped.reset();
    gradientPrepared.reset();
    gradientPreRunHook = nullptr;
    Layer::cleanup();
}

void MultiInputCustomLoss::resetForwardBookkeeping() {
    allForwardInputTensorIds.clear();
    for (const optional<Tensor>& input : featureInputs) {
        THOR_THROW_IF_FALSE(input.has_value());
        allForwardInputTensorIds.insert(input.value().getTensorId());
    }
    stillWaitingForForwardInputTensorIds = allForwardInputTensorIds;
}

void MultiInputCustomLoss::initialize() {
    Layer::initialize();
    resetForwardBookkeeping();
    currentBatchSize = 0;
}

void MultiInputCustomLoss::synchronizeComputeStreamForInputs() {
    Stream& runStream = computeStream();
    for (size_t i = 0; i < inputStreams.size(); ++i) {
        THOR_THROW_IF_FALSE(inputStreams[i].isInitialized());
        if (i == 0)
            continue;
        runStream.waitEvent(inputStreams[i].putEvent());
    }
}

void MultiInputCustomLoss::forward(optional<Tensor> featureInput, bool validationPass, uint32_t batchSize) {
    THOR_THROW_IF_FALSE(running);
    THOR_THROW_IF_FALSE(featureInput.has_value());
    if (batchSize != 0)
        currentBatchSize = batchSize;

    bool matched = false;
    for (const optional<Tensor>& input : featureInputs) {
        if (input.has_value() && input.value() == featureInput.value()) {
            matched = true;
            break;
        }
    }
    THOR_THROW_IF_FALSE(matched);

    const unsigned long tensorId = featureInput.value().getTensorId();
    THOR_THROW_IF_FALSE(stillWaitingForForwardInputTensorIds.count(tensorId) == 1);
    stillWaitingForForwardInputTensorIds.erase(tensorId);
    if (!stillWaitingForForwardInputTensorIds.empty())
        return;

    synchronizeComputeStreamForInputs();
    THOR_THROW_IF_FALSE(lossStamped != nullptr);
    if (lossPreRunHook)
        lossPreRunHook(computeStream());
    lossStamped->run();

    if (gradientStamped != nullptr) {
        if (gradientPreRunHook)
            gradientPreRunHook(computeStream());
        gradientStamped->run();
    }

    Event lossReady = computeStream().putEvent();
    for (size_t i = 1; i < inputStreams.size(); ++i)
        inputStreams[i].waitEvent(lossReady);

    resetForwardBookkeeping();

    if (nextLayer.has_value())
        nextLayer.value()->forward(featureOutput, validationPass, currentBatchSize);

    if (isInferenceOnly() || validationPass)
        return;

    backward(nullopt, currentBatchSize);
}

void MultiInputCustomLoss::backward(optional<Tensor> errorInput, uint32_t batchSize) {
    THOR_THROW_IF_FALSE(running);
    THOR_THROW_IF_FALSE(!errorInput.has_value());
    (void)batchSize;

    THOR_THROW_IF_FALSE(gradientStamped != nullptr);
    for (size_t i = 0; i < previousLayers.size(); ++i) {
        if (!previousLayers[i].has_value() || !errorOutputs[i].has_value())
            continue;
        previousLayers[i].value()->backward(errorOutputs[i], currentBatchSize);
    }
}

optional<Tensor> MultiInputCustomLoss::getErrorOutput(uint32_t inputIndex) const {
    THOR_THROW_IF_FALSE(inputIndex < errorOutputs.size());
    return errorOutputs[inputIndex];
}

void MultiInputCustomLoss::infer(optional<Tensor> inputTensor, optional<Tensor> outputTensor, Stream stream) {
    (void)inputTensor;
    (void)outputTensor;
    (void)stream;
    THOR_UNREACHABLE();
}

void MultiInputCustomLoss::backProp(optional<Tensor> dataIn, optional<Tensor> errorIn, optional<Tensor> errorOut, Stream stream) {
    (void)dataIn;
    (void)errorIn;
    (void)errorOut;
    (void)stream;
    THOR_UNREACHABLE();
}

}  // namespace ThorImplementation
