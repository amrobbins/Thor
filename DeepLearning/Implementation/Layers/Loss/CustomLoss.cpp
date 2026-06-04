#include "DeepLearning/Implementation/Layers/Loss/CustomLoss.h"

#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Expression/FusedEquation.h"

#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

using namespace std;

namespace ThorImplementation {
namespace {

std::string joinNames(const std::set<std::string>& names) {
    if (names.empty())
        return "<none>";

    std::ostringstream oss;
    bool first = true;
    for (const std::string& name : names) {
        if (!first)
            oss << ", ";
        oss << name;
        first = false;
    }
    return oss.str();
}

std::set<std::string> toNameSet(const std::vector<std::string>& names) { return std::set<std::string>(names.begin(), names.end()); }

void validateName(const std::string& name, const std::string& what) {
    if (name.empty())
        throw std::invalid_argument("CustomLoss " + what + " name cannot be empty.");
    if (name.length() >= 2 && name[0] == '_' && name[1] == '_')
        throw std::invalid_argument("CustomLoss " + what + " names cannot start with __ that is reserved. Name " + name + " is illegal.");
}

DataType findOutputDType(const std::shared_ptr<CompiledOutputs>& compiledOutputs, const std::string& outputName) {
    std::optional<DataType> outputDType;
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
        throw std::runtime_error("CustomLoss expression did not infer output dtype for '" + outputName + "'.");
    return outputDType.value();
}

}  // namespace

CustomLoss::CustomLoss(DynamicExpression lossExpression,
                       DynamicExpression gradientExpression,
                       std::string predictionsName,
                       std::string labelsName,
                       std::string lossName,
                       std::string gradientName,
                       DataType lossDataType)
    : Loss(lossDataType),
      lossExpression(std::move(lossExpression)),
      gradientExpression(std::move(gradientExpression)),
      predictionsName(std::move(predictionsName)),
      labelsName(std::move(labelsName)),
      lossName(std::move(lossName)),
      gradientName(std::move(gradientName)) {
    validateName(this->predictionsName, "predictions input");
    validateName(this->labelsName, "labels input");
    validateName(this->lossName, "loss output");
    validateName(this->gradientName, "gradient output");
    if (this->predictionsName == this->labelsName)
        throw std::invalid_argument("CustomLoss predictions and labels input names must be distinct.");
}

CustomLoss::TensorMap CustomLoss::buildLossInputs() const {
    THOR_THROW_IF_FALSE(featureInput.has_value());
    THOR_THROW_IF_FALSE(labelsInput.has_value());

    TensorMap inputs;
    inputs.emplace(predictionsName, featureInput.value());
    inputs.emplace(labelsName, labelsInput.value());
    return inputs;
}

CustomLoss::TensorMap CustomLoss::buildLossOutputs() const {
    THOR_THROW_IF_FALSE(featureOutput.has_value());

    TensorMap outputs;
    outputs.emplace(lossName, featureOutput.value());
    return outputs;
}

CustomLoss::TensorMap CustomLoss::buildGradientOutputs() const {
    THOR_THROW_IF_FALSE(errorOutput.has_value());

    TensorMap outputs;
    outputs.emplace(gradientName, errorOutput.value());
    return outputs;
}

void CustomLoss::validateExpressionOutputNames(const DynamicExpression& expression,
                                               const std::string& expectedOutputName,
                                               const std::string& what) const {
    const std::vector<std::string>& expectedOutputs = expression.getExpectedOutputNames();
    if (!expectedOutputs.empty()) {
        const std::set<std::string> expected{expectedOutputName};
        const std::set<std::string> actual(expectedOutputs.begin(), expectedOutputs.end());
        if (actual != expected) {
            throw std::runtime_error("CustomLoss " + what + " expression output name mismatch. Expected {" + joinNames(expected) +
                                     "}, got {" + joinNames(actual) + "}.");
        }
    }
}

std::pair<std::vector<uint64_t>, DataType> CustomLoss::inferExpressionOutputDescriptor(const DynamicExpression& expression,
                                                                                       const std::string& outputName,
                                                                                       const std::string& what) const {
    THOR_THROW_IF_FALSE(featureInput.has_value());
    THOR_THROW_IF_FALSE(labelsInput.has_value());
    THOR_THROW_IF_FALSE(stream.isInitialized());

    DynamicExpressionBuild build = expression.build(buildLossInputs(), {}, const_cast<Stream&>(stream));

    const std::set<std::string> actualOutputNames = toNameSet(build.equation->getOutputNames());
    const std::set<std::string> expectedOutputNames{outputName};
    if (actualOutputNames != expectedOutputNames) {
        throw std::runtime_error("CustomLoss " + what + " expression output name mismatch. Expected {" + joinNames(expectedOutputNames) +
                                 "}, got {" + joinNames(actualOutputNames) + "}.");
    }

    std::unordered_map<std::string, std::vector<uint64_t>> outputShapes =
        build.equation->getOutputShapes(build.stamp_inputs, build.tensor_scalar_inputs);
    auto shapeIt = outputShapes.find(outputName);
    if (shapeIt == outputShapes.end()) {
        throw std::runtime_error("CustomLoss " + what + " expression did not infer output shape for '" + outputName + "'.");
    }

    std::shared_ptr<CompiledOutputs> compiledOutputs =
        build.equation->compileForInputs(build.stamp_inputs, {}, build.tensor_scalar_inputs);

    return {shapeIt->second, findOutputDType(compiledOutputs, outputName)};
}

std::optional<Tensor> CustomLoss::createFeatureOutputTensor() {
    const auto [outputShape, outputDType] = inferExpressionOutputDescriptor(lossExpression, lossName, "loss");
    THOR_THROW_IF_FALSE(outputDType == lossDataType);
    THOR_THROW_IF_FALSE(featureInput.has_value());
    return Tensor(featureInput.value().getPlacement(), TensorDescriptor(outputDType, outputShape));
}

void CustomLoss::tryFuseGradientIntoDrivingLayer() {
    if (gradientFusedIntoDrivingLayer || isInferenceOnly()) {
        return;
    }
    if (!previousLayer.has_value() || !featureInput.has_value() || !labelsInput.has_value() || !errorOutput.has_value()) {
        return;
    }

    auto* customLayer = dynamic_cast<CustomLayer*>(previousLayer.value());
    if (customLayer == nullptr) {
        return;
    }

    gradientFusedIntoDrivingLayer = customLayer->registerFusedCustomLossGradient(featureInput.value(),
                                                                                 labelsInput.value(),
                                                                                 gradientExpression,
                                                                                 predictionsName,
                                                                                 labelsName,
                                                                                 gradientName);
}

std::optional<Tensor> CustomLoss::connectToPredictionsInputLayer(Layer* predictionsInputLayer,
                                                                 std::optional<Tensor> featureInput,
                                                                 Stream stream,
                                                                 bool backPropagateError) {
    std::optional<Tensor> error = Loss::connectToPredictionsInputLayer(predictionsInputLayer, featureInput, stream, backPropagateError);
    tryFuseGradientIntoDrivingLayer();
    return error;
}

std::optional<Tensor> CustomLoss::connectToLabelsInputLayer(Layer* labelsLayer, std::optional<Tensor> labels, Stream labelsStream) {
    std::optional<Tensor> error = Loss::connectToLabelsInputLayer(labelsLayer, labels, labelsStream);
    tryFuseGradientIntoDrivingLayer();
    return error;
}

void CustomLoss::compileImpl() {
    Loss::compileImpl();

    THOR_THROW_IF_FALSE(featureInput.has_value());
    THOR_THROW_IF_FALSE(featureOutput.has_value());
    THOR_THROW_IF_FALSE(labelsInput.has_value());
    THOR_THROW_IF_FALSE(featureInput.value().isInitialized());
    THOR_THROW_IF_FALSE(featureOutput.value().isInitialized());
    THOR_THROW_IF_FALSE(labelsInput.value().isInitialized());
    THOR_THROW_IF_FALSE(featureInput.value().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    THOR_THROW_IF_FALSE(featureOutput.value().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    THOR_THROW_IF_FALSE(labelsInput.value().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    THOR_THROW_IF_FALSE(featureInput.value().getPlacement() == featureOutput.value().getPlacement());
    THOR_THROW_IF_FALSE(featureInput.value().getPlacement() == labelsInput.value().getPlacement());
    THOR_THROW_IF_FALSE(featureOutput.value().getDescriptor().getDataType() == lossDataType);

    validateExpressionOutputNames(lossExpression, lossName, "loss");
    validateExpressionOutputNames(gradientExpression, gradientName, "gradient");

    TensorMap inputs = buildLossInputs();
    TensorMap lossOutputs = buildLossOutputs();
    lossPrepared = std::make_shared<PreparedDynamicExpression>(lossExpression.prepare(inputs, lossOutputs, stream));
    lossPreRunHook = lossPrepared->preForwardHook();
    lossStamped = std::make_shared<StampedExecutionPlan>(lossPrepared->stamp(lossOutputs));
    validateExpressionOutputNames(lossExpression, lossName, "loss");

    if (!isInferenceOnly() && !gradientFusedIntoDrivingLayer) {
        THOR_THROW_IF_FALSE(errorOutput.has_value());
        THOR_THROW_IF_FALSE(errorOutput.value().isInitialized());
        THOR_THROW_IF_FALSE(errorOutput.value().getPlacement() == featureInput.value().getPlacement());
        THOR_THROW_IF_FALSE(errorOutput.value().getDescriptor() == featureInput.value().getDescriptor());

        TensorMap gradientOutputs = buildGradientOutputs();
        gradientPrepared = std::make_shared<PreparedDynamicExpression>(gradientExpression.prepare(inputs, gradientOutputs, stream));
        gradientPreRunHook = gradientPrepared->preForwardHook();
        gradientStamped = std::make_shared<StampedExecutionPlan>(gradientPrepared->stamp(gradientOutputs));
    } else {
        gradientPrepared.reset();
        gradientStamped.reset();
        gradientPreRunHook = nullptr;
    }
}

void CustomLoss::cleanup() {
    lossStamped.reset();
    lossPrepared.reset();
    lossPreRunHook = nullptr;
    gradientStamped.reset();
    gradientPrepared.reset();
    gradientPreRunHook = nullptr;
    Loss::cleanup();
}

void CustomLoss::infer(std::optional<Tensor> predictions, std::optional<Tensor> loss, Stream runStream) {
    THOR_THROW_IF_FALSE(predictions.has_value());
    THOR_THROW_IF_FALSE(loss.has_value());
    THOR_THROW_IF_FALSE(predictions.value() == featureInput.value());
    THOR_THROW_IF_FALSE(loss.value() == featureOutput.value());
    THOR_THROW_IF_FALSE(labelsInput.has_value());
    THOR_THROW_IF_FALSE(lossStamped != nullptr);

    runStream.waitEvent(labelsStream.putEvent());
    if (lossPreRunHook)
        lossPreRunHook(this->stream);
    lossStamped->run();

    if (gradientStamped != nullptr) {
        if (gradientPreRunHook)
            gradientPreRunHook(this->stream);
        gradientStamped->run();
    }
    labelsStream.waitEvent(runStream.putEvent());
}

void CustomLoss::backProp(std::optional<Tensor> labels,
                          std::optional<Tensor> predictions,
                          std::optional<Tensor> lossGradient,
                          Stream runStream) {
    THOR_THROW_IF_FALSE(labels.has_value());
    THOR_THROW_IF_FALSE(predictions.has_value());
    THOR_THROW_IF_FALSE(lossGradient.has_value());
    THOR_THROW_IF_FALSE(labels.value() == labelsInput.value());
    THOR_THROW_IF_FALSE(predictions.value() == featureInput.value());
    THOR_THROW_IF_FALSE(lossGradient.value() == errorOutput.value());

    (void)runStream;
    // Loss layers originate backpropagation. The prediction gradient is prepared during infer(),
    // matching the other loss implementations. When fused, the driving CustomLayer ignores this
    // materialized error tensor and seeds its backward graph directly from the CustomLoss gradient expression.
    THOR_THROW_IF_FALSE(gradientStamped != nullptr || gradientFusedIntoDrivingLayer);
}

}  // namespace ThorImplementation
