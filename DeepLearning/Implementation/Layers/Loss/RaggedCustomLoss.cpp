#include "DeepLearning/Implementation/Layers/Loss/RaggedCustomLoss.h"

#include "DeepLearning/Implementation/Layers/Loss/RegressionLossDType.h"
#include "DeepLearning/Implementation/Layers/Loss/WeightedLossExpression.h"
#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionDescriptor.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/FusedEquation.h"

#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

using namespace std;

namespace ThorImplementation {
namespace {

string joinNames(const set<string>& names) {
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

void validatePublicName(const string& name, const char* what) {
    if (name.empty())
        throw invalid_argument(string("RaggedCustomLoss ") + what + " name cannot be empty.");
    if (name.length() >= 2 && name[0] == '_' && name[1] == '_') {
        throw invalid_argument(string("RaggedCustomLoss ") + what +
                               " names cannot begin with '__'; that prefix is reserved by Thor.");
    }
}

uint64_t checkedElementsPerValue(const vector<uint64_t>& dimensions) {
    if (dimensions.empty())
        throw invalid_argument("RaggedCustomLoss packed values must have rank >= 1.");
    uint64_t elements = 1;
    for (size_t axis = 1; axis < dimensions.size(); ++axis) {
        if (dimensions[axis] == 0 || elements > numeric_limits<uint64_t>::max() / dimensions[axis])
            throw invalid_argument("RaggedCustomLoss trailing value element count overflows uint64_t.");
        elements *= dimensions[axis];
    }
    return elements;
}

}  // namespace

RaggedCustomLoss::RaggedCustomLoss(DynamicExpression lossExpression,
                                   DynamicExpression gradientExpression,
                                   uint64_t batchSize,
                                   uint64_t maxTotalValues,
                                   string predictionsName,
                                   string labelsName,
                                   string lossName,
                                   string gradientName,
                                   DataType lossDataType,
                                   optional<float> lossWeight,
                                   optional<string> exampleWeightsName,
                                   vector<string> secondaryInputNames,
                                   vector<string> secondaryGradientNames)
    : Loss(lossDataType),
      lossExpression(std::move(lossExpression)),
      gradientExpression(std::move(gradientExpression)),
      batchSize(batchSize),
      maxTotalValues(maxTotalValues),
      predictionsName(std::move(predictionsName)),
      labelsName(std::move(labelsName)),
      lossName(std::move(lossName)),
      gradientName(std::move(gradientName)),
      lossWeight(normalizeLossWeight(lossWeight)),
      exampleWeightsName(std::move(exampleWeightsName)) {
    if (secondaryInputNames.size() != secondaryGradientNames.size())
        throw invalid_argument("RaggedCustomLoss secondary input and gradient names must be configured in pairs.");
    secondaryInputs.reserve(secondaryInputNames.size());
    for (size_t i = 0; i < secondaryInputNames.size(); ++i) {
        secondaryInputs.push_back(SecondaryInputState{
            .inputName = std::move(secondaryInputNames[i]),
            .gradientName = std::move(secondaryGradientNames[i]),
        });
    }
    if (batchSize == 0 || batchSize > numeric_limits<uint32_t>::max())
        throw invalid_argument("RaggedCustomLoss logical batch size must fit in uint32_t and be non-zero.");
    if (maxTotalValues == 0)
        throw invalid_argument("RaggedCustomLoss max_total_values must be non-zero.");

    validatePublicName(this->predictionsName, "predictions input");
    validatePublicName(this->labelsName, "labels input");
    validatePublicName(this->lossName, "loss output");
    validatePublicName(this->gradientName, "gradient output");
    if (this->exampleWeightsName.has_value())
        validatePublicName(this->exampleWeightsName.value(), "example_weights input");
    set<string> gradientNames{this->gradientName};
    for (const SecondaryInputState& secondary : secondaryInputs) {
        validatePublicName(secondary.inputName, "secondary input");
        validatePublicName(secondary.gradientName, "secondary gradient output");
        if (!gradientNames.insert(secondary.gradientName).second)
            throw invalid_argument("RaggedCustomLoss secondary gradient names must be distinct from every other gradient output.");
    }
    if (this->predictionsName == this->labelsName)
        throw invalid_argument("RaggedCustomLoss predictions and labels input names must be distinct.");
    if (this->predictionsName == RAGGED_OFFSETS_INPUT_NAME || this->labelsName == RAGGED_OFFSETS_INPUT_NAME ||
        (this->exampleWeightsName.has_value() && this->exampleWeightsName.value() == RAGGED_OFFSETS_INPUT_NAME))
        throw invalid_argument("RaggedCustomLoss public input names collide with Thor's structural offsets port.");
    set<string> inputNames{this->predictionsName, this->labelsName};
    if (this->exampleWeightsName.has_value()) inputNames.insert(this->exampleWeightsName.value());
    for (const SecondaryInputState& secondary : secondaryInputs) {
        if (secondary.inputName == RAGGED_OFFSETS_INPUT_NAME || !inputNames.insert(secondary.inputName).second)
            throw invalid_argument("RaggedCustomLoss secondary input names must be distinct from all other inputs.");
    }
    if (this->exampleWeightsName.has_value() && inputNames.size() != 3 + secondaryInputs.size())
        throw invalid_argument("RaggedCustomLoss example_weights input name must be distinct from other inputs.");

    validateExpressionContract(this->lossExpression, {this->lossName}, "loss");
    set<string> gradientOutputs{this->gradientName};
    for (const SecondaryInputState& secondary : secondaryInputs) gradientOutputs.insert(secondary.gradientName);
    validateExpressionContract(this->gradientExpression, gradientOutputs, "gradient");
}

RaggedCustomLoss::RaggedCustomLoss(DynamicExpression lossExpression,
                                   DynamicExpression gradientExpression,
                                   uint64_t batchSize,
                                   uint64_t maxTotalValues,
                                   string predictionsName,
                                   string labelsName,
                                   string lossName,
                                   string gradientName,
                                   DataType lossDataType,
                                   optional<float> lossWeight,
                                   optional<string> exampleWeightsName,
                                   optional<string> secondaryInputName,
                                   optional<string> secondaryGradientName)
    : RaggedCustomLoss(std::move(lossExpression),
                       std::move(gradientExpression),
                       batchSize,
                       maxTotalValues,
                       std::move(predictionsName),
                       std::move(labelsName),
                       std::move(lossName),
                       std::move(gradientName),
                       lossDataType,
                       lossWeight,
                       std::move(exampleWeightsName),
                       secondaryInputName.has_value() ? vector<string>{std::move(secondaryInputName.value())} : vector<string>{},
                       secondaryGradientName.has_value() ? vector<string>{std::move(secondaryGradientName.value())} : vector<string>{}) {}

void RaggedCustomLoss::validateExpressionContract(const DynamicExpression& expression,
                                                   const set<string>& expectedOutputNames,
                                                   const char* what) const {
    const vector<string>& expectedInputs = expression.getExpectedInputNames();
    if (!expectedInputs.empty()) {
        const set<string> actual(expectedInputs.begin(), expectedInputs.end());
        set<string> expected{predictionsName, labelsName};
        for (const SecondaryInputState& secondary : secondaryInputs) expected.insert(secondary.inputName);
        if (exampleWeightsName.has_value()) expected.insert(exampleWeightsName.value());
        if (actual != expected) {
            throw invalid_argument(string("RaggedCustomLoss ") + what + " expression input mismatch. Expected {" +
                                   joinNames(expected) + "}, got {" + joinNames(actual) + "}.");
        }
    }

    const vector<string>& expectedOutputs = expression.getExpectedOutputNames();
    if (!expectedOutputs.empty()) {
        const set<string> actual(expectedOutputs.begin(), expectedOutputs.end());
        if (actual != expectedOutputNames) {
            throw invalid_argument(string("RaggedCustomLoss ") + what + " expression output mismatch. Expected {" +
                                   joinNames(expectedOutputNames) + "}, got {" + joinNames(actual) + "}.");
        }
    }
}

optional<Tensor> RaggedCustomLoss::createFeatureOutputTensor() {
    THOR_THROW_IF_FALSE(featureInput.has_value());
    return featureInput.value().clone(lossDataType);
}

optional<Tensor> RaggedCustomLoss::connectToPreviousLayer(Layer* previousLayer,
                                                          optional<Tensor> connectedInput,
                                                          Stream connectedStream,
                                                          bool backPropagateError,
                                                          int connectionType) {
    THOR_THROW_IF_FALSE(!compiled);
    THOR_THROW_IF_FALSE(connectedInput.has_value());

    switch (static_cast<InputConnection>(connectionType)) {
        case InputConnection::PREDICTIONS:
            return Loss::connectToPredictionsInputLayer(previousLayer, connectedInput, connectedStream, backPropagateError);
        case InputConnection::LABELS:
            if (backPropagateError)
                throw invalid_argument("RaggedCustomLoss labels are non-differentiable inputs.");
            return Loss::connectToLabelsInputLayer(previousLayer, connectedInput, connectedStream);
        case InputConnection::OFFSETS: {
            if (backPropagateError)
                throw invalid_argument("RaggedCustomLoss offsets are structural and non-differentiable.");
            if (offsetsInput.has_value())
                throw logic_error("RaggedCustomLoss offsets input is already connected.");
            const Tensor& offsets = connectedInput.value();
            const DataType dtype = offsets.getDataType();
            if (!RowPartitionDescriptor::isValidOffsetsDataType(dtype))
                throw invalid_argument("RaggedCustomLoss offsets dtype must be UINT32 or UINT64.");
            const RowPartitionDescriptor descriptor(batchSize, maxTotalValues, dtype);
            if (offsets.getDescriptor() != descriptor.getOffsetsDescriptor())
                throw invalid_argument("RaggedCustomLoss offsets must have canonical shape [batch_size + 1].");
            offsetsInput = offsets;
            offsetsStream = connectedStream;
            return nullopt;
        }
        case InputConnection::EXAMPLE_WEIGHTS: {
            if (!exampleWeightsName.has_value())
                throw invalid_argument("RaggedCustomLoss was not configured with example_weights.");
            (void)backPropagateError;  // Row weights are an auxiliary, non-differentiable input.
            if (exampleWeightsInput.has_value())
                throw logic_error("RaggedCustomLoss example_weights input is already connected.");
            const Tensor& weights = connectedInput.value();
            RegressionLossDType::validateExampleWeightDType("RaggedCustomLoss", weights.getDataType());
            if (weights.getDimensions() != vector<uint64_t>{maxTotalValues, 1})
                throw invalid_argument("RaggedCustomLoss example_weights must be packed scalar weights with shape [max_total_values, 1].");
            exampleWeightsInput = weights;
            exampleWeightsStream = connectedStream;
            return nullopt;
        }
        default:
            break;
    }

    const int secondaryBase = static_cast<int>(InputConnection::SECONDARY_INPUT_BASE);
    const int secondaryIndex = connectionType - secondaryBase;
    if (secondaryIndex < 0 || static_cast<size_t>(secondaryIndex) >= secondaryInputs.size())
        throw invalid_argument("RaggedCustomLoss input connection type is out of range.");

    SecondaryInputState& secondary = secondaryInputs[static_cast<size_t>(secondaryIndex)];
    if (secondary.input.has_value())
        throw logic_error("RaggedCustomLoss secondary input is already connected.");
    const Tensor& input = connectedInput.value();
    if (input.getDimensions().empty() || input.getDimensions().front() != maxTotalValues)
        throw invalid_argument("RaggedCustomLoss secondary input must use max_total_values as its packed leading dimension.");
    secondary.input = input;
    secondary.previousLayer = previousLayer;
    secondary.stream = connectedStream;
    if (!isInferenceOnly())
        secondary.errorOutput = input.clone();
    return secondary.errorOutput;
}

uint64_t RaggedCustomLoss::elementsPerValue() const {
    THOR_THROW_IF_FALSE(featureInput.has_value());
    return checkedElementsPerValue(featureInput.value().getDimensions());
}

RaggedCustomLoss::TensorMap RaggedCustomLoss::buildInputs() const {
    THOR_THROW_IF_FALSE(featureInput.has_value());
    THOR_THROW_IF_FALSE(labelsInput.has_value());
    THOR_THROW_IF_FALSE(offsetsInput.has_value());
    TensorMap inputs{{predictionsName, featureInput.value()},
                     {labelsName, labelsInput.value()},
                     {RAGGED_OFFSETS_INPUT_NAME, offsetsInput.value()}};
    for (const SecondaryInputState& secondary : secondaryInputs) {
        THOR_THROW_IF_FALSE(secondary.input.has_value());
        inputs.emplace(secondary.inputName, secondary.input.value());
    }
    if (exampleWeightsName.has_value()) {
        THOR_THROW_IF_FALSE(exampleWeightsInput.has_value());
        inputs.emplace(exampleWeightsName.value(), exampleWeightsInput.value());
    }
    return inputs;
}

DynamicExpression RaggedCustomLoss::withRaggedExtent(const DynamicExpression& expression,
                                                      const unordered_map<string, DataType>& outputDataTypes,
                                                      const char* what) const {
    vector<string> wrappedInputs{predictionsName, labelsName, RAGGED_OFFSETS_INPUT_NAME};
    for (const SecondaryInputState& secondary : secondaryInputs) wrappedInputs.push_back(secondary.inputName);
    if (exampleWeightsName.has_value()) wrappedInputs.push_back(exampleWeightsName.value());
    vector<string> wrappedOutputs;
    wrappedOutputs.reserve(outputDataTypes.size());
    for (const auto& [name, dtype] : outputDataTypes) {
        (void)dtype;
        wrappedOutputs.push_back(name);
    }
    const uint64_t runtimeBatchSize = batchSize;
    const uint64_t runtimeMaxTotalValues = maxTotalValues;
    const uint64_t runtimeElementsPerValue = elementsPerValue();
    const string whatString = string("RaggedCustomLoss ") + what;

    return DynamicExpression(
        wrappedInputs,
        wrappedOutputs,
        [expression,
         outputDataTypes,
         runtimeBatchSize,
         runtimeMaxTotalValues,
         runtimeElementsPerValue,
         whatString](const DynamicExpression::TensorMap& inputs,
                     const DynamicExpression::TensorMap& outputs,
                     Stream& stream) -> DynamicExpressionBuild {
            auto offsetsIt = inputs.find(RAGGED_OFFSETS_INPUT_NAME);
            if (offsetsIt == inputs.end())
                throw invalid_argument(whatString + " requires the structural offsets input.");

            DynamicExpression::TensorMap valueInputs = inputs;
            valueInputs.erase(RAGGED_OFFSETS_INPUT_NAME);
            DynamicExpressionBuild build = expression.build(valueInputs, {}, stream);
            const PhysicalOutputs& rawOutputs = build.equation->physicalOutputs();
            PhysicalOutputs raggedOutputs = detail::transformDynamicExpressionOutputsRecursively(
                rawOutputs,
                [outputDataTypes,
                 offsetsDType = offsetsIt->second.getDataType(),
                 runtimeBatchSize,
                 runtimeMaxTotalValues,
                 runtimeElementsPerValue,
                 &whatString](const string& outputName, const Expression& raw) {
                    auto dtypeIt = outputDataTypes.find(outputName);
                    if (dtypeIt == outputDataTypes.end())
                        throw runtime_error(whatString + " encountered unexpected output '" + outputName + "'.");
                    const Expression offsets = Expression::input(RAGGED_OFFSETS_INPUT_NAME, nullopt, offsetsDType);
                    return raw.withOutputDType(dtypeIt->second)
                        .withRaggedRuntimeExtent(offsets,
                                                 runtimeBatchSize,
                                                 runtimeMaxTotalValues,
                                                 runtimeElementsPerValue);
                },
                whatString);

            build.stamp_inputs.emplace(RAGGED_OFFSETS_INPUT_NAME, offsetsIt->second);
            return DynamicExpressionBuild{
                .equation = make_shared<FusedEquation>(FusedEquation::compile(raggedOutputs, stream.getGpuNum())),
                .stamp_inputs = std::move(build.stamp_inputs),
                .tensor_scalar_inputs = std::move(build.tensor_scalar_inputs),
                .preallocated_outputs = outputs,
                .requested_output_shapes = std::move(build.requested_output_shapes),
                .pre_forward_hook = std::move(build.pre_forward_hook),
                .pre_forward_only_inputs = std::move(build.pre_forward_only_inputs),
            };
        });
}

void RaggedCustomLoss::compileImpl() {
    Layer::compileImpl();

    bool missingSecondary = false;
    for (const SecondaryInputState& secondary : secondaryInputs) missingSecondary |= !secondary.input.has_value();
    if (!featureInput.has_value() || !labelsInput.has_value() || !offsetsInput.has_value() || !featureOutput.has_value() ||
        (exampleWeightsName.has_value() && !exampleWeightsInput.has_value()) || missingSecondary)
        throw logic_error("RaggedCustomLoss requires all configured inputs and a raw-loss consumer before compile.");

    const Tensor& predictions = featureInput.value();
    const Tensor& labels = labelsInput.value();
    const Tensor& offsets = offsetsInput.value();
    const vector<uint64_t> predictionDimensions = predictions.getDimensions();
    const vector<uint64_t> labelDimensions = labels.getDimensions();

    if (predictions.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU ||
        labels.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU ||
        offsets.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU ||
        (exampleWeightsInput.has_value() && exampleWeightsInput->getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU)) {
        throw invalid_argument("RaggedCustomLoss currently requires GPU-resident inputs.");
    }
    for (const SecondaryInputState& secondary : secondaryInputs)
        if (!secondary.input.has_value() || secondary.input->getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU)
            throw invalid_argument("RaggedCustomLoss currently requires GPU-resident secondary inputs.");
    if (predictionDimensions.empty() || predictionDimensions.front() != maxTotalValues)
        throw invalid_argument("RaggedCustomLoss predictions must have leading packed capacity max_total_values.");
    if (labelDimensions != predictionDimensions)
        throw invalid_argument("RaggedCustomLoss labels must have the same packed value shape as predictions.");
    RegressionLossDType::validatePredictionsDType("RaggedCustomLoss", predictions.getDataType());
    RegressionLossDType::validateLabelsDType("RaggedCustomLoss", labels.getDataType());
    for (const SecondaryInputState& secondary : secondaryInputs)
        if (!secondary.input.has_value() || secondary.input->getDimensions() != predictionDimensions)
            throw invalid_argument("RaggedCustomLoss secondary differentiable inputs must match prediction packed geometry.");
    if (exampleWeightsInput.has_value()) {
        RegressionLossDType::validateExampleWeightDType("RaggedCustomLoss", exampleWeightsInput->getDataType());
        if (exampleWeightsInput->getDimensions() != vector<uint64_t>{maxTotalValues, 1})
            throw invalid_argument("RaggedCustomLoss example_weights must have packed shape [max_total_values, 1].");
    }
    RegressionLossDType::validateLossDType("RaggedCustomLoss", lossDataType);
    if (featureOutput.value().getDimensions() != predictionDimensions || featureOutput.value().getDataType() != lossDataType)
        throw logic_error("RaggedCustomLoss raw loss tensor must preserve the packed value shape.");
    if (!isInferenceOnly()) {
        if (!errorOutput.has_value())
            throw logic_error("RaggedCustomLoss training requires prediction-gradient storage.");
        if (errorOutput.value().getDescriptor() != predictions.getDescriptor())
            throw logic_error("RaggedCustomLoss prediction gradient must match the prediction descriptor.");
        for (const SecondaryInputState& secondary : secondaryInputs) {
            if (!secondary.input.has_value() || !secondary.errorOutput.has_value() ||
                secondary.errorOutput->getDescriptor() != secondary.input->getDescriptor())
                throw logic_error("RaggedCustomLoss secondary gradient must match the secondary input descriptor.");
        }
    }

    ensureNoDeviceCrossing();
    (void)elementsPerValue();

    TensorMap inputs = buildInputs();
    const DynamicExpression weightedLoss =
        applyLossWeightToDynamicExpression(lossExpression, {{lossName, lossDataType}}, lossWeight, "RaggedCustomLoss loss");
    const DynamicExpression wrappedLoss = withRaggedExtent(weightedLoss, {{lossName, lossDataType}}, "loss");
    TensorMap lossOutputs{{lossName, featureOutput.value()}};
    lossPrepared = make_shared<PreparedDynamicExpression>(wrappedLoss.prepare(inputs, lossOutputs, stream));
    lossPreRunHook = lossPrepared->preForwardHook();
    lossStamped = make_shared<StampedExecutionPlan>(lossPrepared->stamp(lossOutputs));

    if (!isInferenceOnly()) {
        unordered_map<string, DataType> gradientDTypes{{gradientName, predictions.getDataType()}};
        TensorMap gradientOutputs{{gradientName, errorOutput.value()}};
        for (const SecondaryInputState& secondary : secondaryInputs) {
            THOR_THROW_IF_FALSE(secondary.input.has_value() && secondary.errorOutput.has_value());
            gradientDTypes.emplace(secondary.gradientName, secondary.input->getDataType());
            gradientOutputs.emplace(secondary.gradientName, secondary.errorOutput.value());
        }
        const DynamicExpression weightedGradient = applyLossWeightToDynamicExpression(
            gradientExpression, gradientDTypes, lossWeight, "RaggedCustomLoss gradient");
        const DynamicExpression wrappedGradient = withRaggedExtent(weightedGradient, gradientDTypes, "gradient");
        gradientPrepared = make_shared<PreparedDynamicExpression>(wrappedGradient.prepare(inputs, gradientOutputs, stream));
        gradientPreRunHook = gradientPrepared->preForwardHook();
        gradientStamped = make_shared<StampedExecutionPlan>(gradientPrepared->stamp(gradientOutputs));
    } else {
        gradientPrepared.reset();
        gradientStamped.reset();
        gradientPreRunHook = nullptr;
    }
}

void RaggedCustomLoss::initialize() {
    Loss::initialize();
    offsetsReceived = false;
    exampleWeightsReceived = false;
    for (SecondaryInputState& secondary : secondaryInputs) secondary.received = false;
    featureInputReceived = false;
    labelsReceived = false;
    currentValidExampleCount = 0;
    batchCardinalitySet = false;
}

void RaggedCustomLoss::cleanup() {
    lossStamped.reset();
    lossPrepared.reset();
    lossPreRunHook = nullptr;
    gradientStamped.reset();
    gradientPrepared.reset();
    gradientPreRunHook = nullptr;
    offsetsReadyEvent = Event();
    offsetsReusableEvent = Event();
    offsetsReceived = false;
    exampleWeightsReadyEvent = Event();
    exampleWeightsReusableEvent = Event();
    exampleWeightsReceived = false;
    for (SecondaryInputState& secondary : secondaryInputs) {
        secondary.readyEvent = Event();
        secondary.reusableEvent = Event();
        secondary.received = false;
    }
    Loss::cleanup();
}

void RaggedCustomLoss::replaceErrorInput(optional<Tensor> oldErrorInput, optional<Tensor> newErrorInput) {
    if (oldErrorInput.has_value()) {
        for (SecondaryInputState& secondary : secondaryInputs) {
            if (secondary.errorOutput.has_value() && secondary.errorOutput.value() == oldErrorInput.value()) {
                if (secondary.previousLayer.has_value())
                    secondary.previousLayer.value()->replaceErrorInput(secondary.errorOutput, newErrorInput);
                secondary.errorOutput = newErrorInput;
                return;
            }
        }
    }
    Loss::replaceErrorInput(oldErrorInput, newErrorInput);
}

void RaggedCustomLoss::pruneTrainingBackpropPathIfInactive() {
    Loss::pruneTrainingBackpropPathIfInactive();
    if (trainingActive || isInferenceOnly())
        return;
    for (SecondaryInputState& secondary : secondaryInputs) {
        if (!secondary.errorOutput.has_value()) continue;
        if (secondary.previousLayer.has_value())
            secondary.previousLayer.value()->replaceErrorInput(secondary.errorOutput, nullopt);
        secondary.errorOutput = nullopt;
    }
}

uint32_t RaggedCustomLoss::resolveValidExampleCount(uint32_t validExampleCount) const {
    const uint32_t logicalBatchSize = static_cast<uint32_t>(batchSize);
    const uint32_t resolved = validExampleCount == 0 ? logicalBatchSize : validExampleCount;
    if (resolved == 0 || resolved > logicalBatchSize)
        throw invalid_argument("RaggedCustomLoss valid example count exceeds the logical row batch size.");
    return resolved;
}

void RaggedCustomLoss::recordLogicalBatchCardinality(uint32_t validExampleCount) {
    const uint32_t resolved = resolveValidExampleCount(validExampleCount);
    if (batchCardinalitySet) {
        if (currentValidExampleCount != resolved)
            throw invalid_argument("RaggedCustomLoss inputs disagreed on valid logical example count.");
        return;
    }
    currentValidExampleCount = resolved;
    batchCardinalitySet = true;
}

void RaggedCustomLoss::forward(optional<Tensor> inputTensor, bool validationPass, uint32_t validExampleCount) {
    THOR_THROW_IF_FALSE(running);
    if (!inputTensor.has_value())
        throw invalid_argument("RaggedCustomLoss forward requires an arriving connected input tensor.");

    recordLogicalBatchCardinality(validExampleCount);
    const Tensor& input = inputTensor.value();
    if (featureInput.has_value() && input == featureInput.value()) {
        if (featureInputReceived)
            throw logic_error("RaggedCustomLoss predictions arrived twice in one batch.");
        featureInputReceived = true;
    } else if (labelsInput.has_value() && input == labelsInput.value()) {
        if (labelsReceived)
            throw logic_error("RaggedCustomLoss labels arrived twice in one batch.");
        labelsReceived = true;
    } else if (offsetsInput.has_value() && input == offsetsInput.value()) {
        if (offsetsReceived)
            throw logic_error("RaggedCustomLoss offsets arrived twice in one batch.");
        offsetsReceived = true;
    } else if (exampleWeightsInput.has_value() && input == exampleWeightsInput.value()) {
        if (exampleWeightsReceived)
            throw logic_error("RaggedCustomLoss example_weights arrived twice in one batch.");
        exampleWeightsReceived = true;
    } else {
        bool matchedSecondary = false;
        for (SecondaryInputState& secondary : secondaryInputs) {
            if (!secondary.input.has_value() || input != secondary.input.value()) continue;
            if (secondary.received)
                throw logic_error("RaggedCustomLoss secondary input arrived twice in one batch.");
            secondary.received = true;
            matchedSecondary = true;
            break;
        }
        if (!matchedSecondary)
            throw invalid_argument("RaggedCustomLoss received an unconnected input tensor.");
    }

    advanceDataIfReady(validationPass);
}

void RaggedCustomLoss::synchronizeComputeStreamForInputs() {
    THOR_THROW_IF_FALSE(stream.isInitialized());
    THOR_THROW_IF_FALSE(labelsStream.isInitialized());
    THOR_THROW_IF_FALSE(offsetsStream.isInitialized());
    stream.waitFor(labelsStream, labelsReadyEvent);
    stream.waitFor(offsetsStream, offsetsReadyEvent);
    if (exampleWeightsInput.has_value()) {
        THOR_THROW_IF_FALSE(exampleWeightsStream.isInitialized());
        stream.waitFor(exampleWeightsStream, exampleWeightsReadyEvent);
    }
    for (SecondaryInputState& secondary : secondaryInputs) {
        THOR_THROW_IF_FALSE(secondary.stream.isInitialized());
        stream.waitFor(secondary.stream, secondary.readyEvent);
    }
}

void RaggedCustomLoss::markAuxiliaryInputsReusableAfterCompute() {
    labelsStream.waitFor(stream, labelsReusableEvent);
    offsetsStream.waitFor(stream, offsetsReusableEvent);
    if (exampleWeightsInput.has_value())
        exampleWeightsStream.waitFor(stream, exampleWeightsReusableEvent);
    for (SecondaryInputState& secondary : secondaryInputs)
        secondary.stream.waitFor(stream, secondary.reusableEvent);
}

void RaggedCustomLoss::advanceDataIfReady(bool validationPass) {
    bool missingSecondary = false;
    for (const SecondaryInputState& secondary : secondaryInputs) missingSecondary |= !secondary.received;
    if (!featureInputReceived || !labelsReceived || !offsetsReceived ||
        (exampleWeightsInput.has_value() && !exampleWeightsReceived) || missingSecondary)
        return;

    synchronizeComputeStreamForInputs();
    infer(featureInput, featureOutput, stream);

    const bool trainingPass = !isInferenceOnly() && !validationPass && trainingActive;
    if (!trainingPass)
        markAuxiliaryInputsReusableAfterCompute();

    featureInputReceived = false;
    labelsReceived = false;
    offsetsReceived = false;
    exampleWeightsReceived = false;
    for (SecondaryInputState& secondary : secondaryInputs) secondary.received = false;
    batchCardinalitySet = false;

    if (nextLayer.has_value())
        nextLayer.value()->forward(featureOutput, validationPass, currentValidExampleCount);

    if (trainingPass)
        backward(nullopt, currentValidExampleCount);
}

void RaggedCustomLoss::infer(optional<Tensor> predictions, optional<Tensor> loss, Stream runStream) {
    (void)predictions;
    (void)loss;
    THOR_THROW_IF_FALSE(runStream == stream);
    THOR_THROW_IF_FALSE(lossStamped != nullptr);
    if (lossPreRunHook)
        lossPreRunHook(stream);
    lossStamped->run();
}

void RaggedCustomLoss::backProp(optional<Tensor> labels,
                                optional<Tensor> predictions,
                                optional<Tensor> lossGradient,
                                Stream runStream) {
    (void)labels;
    (void)predictions;
    (void)lossGradient;
    THOR_THROW_IF_FALSE(runStream == stream);
    THOR_THROW_IF_FALSE(gradientStamped != nullptr);
    if (gradientPreRunHook)
        gradientPreRunHook(stream);
    gradientStamped->run();
}

void RaggedCustomLoss::backward(optional<Tensor> incomingError, uint32_t validExampleCount) {
    THOR_THROW_IF_FALSE(running);
    if (incomingError.has_value())
        throw invalid_argument("RaggedCustomLoss is a gradient origin and does not accept an incoming error tensor.");
    if (isInferenceOnly())
        throw logic_error("RaggedCustomLoss backward cannot run for an inference-only layer.");
    if (!errorOutput.has_value() || gradientStamped == nullptr)
        throw logic_error("RaggedCustomLoss backward requires compiled prediction-gradient storage.");

    const uint32_t resolved = validExampleCount == 0 ? currentValidExampleCount : resolveValidExampleCount(validExampleCount);
    if (resolved != currentValidExampleCount)
        throw invalid_argument("RaggedCustomLoss backward valid example count differs from the forward batch.");

    backProp(labelsInput, featureInput, errorOutput, stream);
    markAuxiliaryInputsReusableAfterCompute();

    if (previousLayer.has_value())
        previousLayer.value()->backward(errorOutput, resolved);
    for (SecondaryInputState& secondary : secondaryInputs)
        if (secondary.previousLayer.has_value() && secondary.errorOutput.has_value())
            secondary.previousLayer.value()->backward(secondary.errorOutput, resolved);
}

vector<Stream> RaggedCustomLoss::getProcessingStreams() {
    vector<Stream> result;
    set<uint64_t> ids;
    for (const Stream* candidate : {&stream, &labelsStream, &offsetsStream, &exampleWeightsStream}) {
        if (!candidate->isInitialized() || !ids.insert(candidate->getId()).second)
            continue;
        result.push_back(*candidate);
    }
    for (const SecondaryInputState& secondary : secondaryInputs) {
        if (!secondary.stream.isInitialized() || !ids.insert(secondary.stream.getId()).second)
            continue;
        result.push_back(secondary.stream);
    }
    return result;
}

vector<Event> RaggedCustomLoss::getSynchronizeEvents() {
    vector<Event> events;
    set<uint64_t> ids;
    appendSynchronizeEvent(events, ids, stream);
    appendSynchronizeEvent(events, ids, labelsStream);
    appendSynchronizeEvent(events, ids, offsetsStream);
    appendSynchronizeEvent(events, ids, exampleWeightsStream);
    for (const SecondaryInputState& secondary : secondaryInputs)
        appendSynchronizeEvent(events, ids, secondary.stream);
    return events;
}

void RaggedCustomLoss::ensureNoDeviceCrossing() {
    Loss::ensureNoDeviceCrossing();
    if (!offsetsInput.has_value())
        return;
    const TensorPlacement offsetsPlacement = offsetsInput.value().getPlacement();
    if (featureInput.has_value() && featureInput.value().getPlacement() != offsetsPlacement)
        throw invalid_argument("RaggedCustomLoss predictions and offsets must share placement.");
    if (labelsInput.has_value() && labelsInput.value().getPlacement() != offsetsPlacement)
        throw invalid_argument("RaggedCustomLoss labels and offsets must share placement.");
    if (featureOutput.has_value() && featureOutput.value().getPlacement() != offsetsPlacement)
        throw invalid_argument("RaggedCustomLoss raw loss and offsets must share placement.");
    if (errorOutput.has_value() && errorOutput.value().getPlacement() != offsetsPlacement)
        throw invalid_argument("RaggedCustomLoss prediction gradient and offsets must share placement.");
    if (exampleWeightsInput.has_value() && exampleWeightsInput.value().getPlacement() != offsetsPlacement)
        throw invalid_argument("RaggedCustomLoss example_weights and offsets must share placement.");
    for (const SecondaryInputState& secondary : secondaryInputs) {
        if (secondary.input.has_value() && secondary.input.value().getPlacement() != offsetsPlacement)
            throw invalid_argument("RaggedCustomLoss secondary input and offsets must share placement.");
        if (secondary.errorOutput.has_value() && secondary.errorOutput.value().getPlacement() != offsetsPlacement)
            throw invalid_argument("RaggedCustomLoss secondary gradient and offsets must share placement.");
    }
}

}  // namespace ThorImplementation
