#include "DeepLearning/Implementation/Layers/Loss/CustomLoss.h"
#include "DeepLearning/Implementation/Layers/Loss/WeightedLossExpression.h"

#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "DeepLearning/Implementation/Layers/Utility/TensorFanout.h"
#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Expression/FusedEquation.h"
#include "Utilities/TensorOperations/Masking/BatchValidity.h"

#include <algorithm>
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


DynamicExpression applyBatchValidityMaskToGradient(const DynamicExpression& expression,
                                                   const std::string& gradientName,
                                                   DataType gradientDataType,
                                                   bool expressionUsesBatchValidityMask) {
    std::vector<std::string> expectedInputs = expression.getExpectedInputNames();
    if (!expectedInputs.empty() && !expressionUsesBatchValidityMask)
        expectedInputs.push_back(Thor::BATCH_VALIDITY_MASK_NAME);

    return DynamicExpression(
        std::move(expectedInputs),
        expression.getExpectedOutputNames(),
        [expression, gradientName, gradientDataType, expressionUsesBatchValidityMask](const DynamicExpression::TensorMap& inputs,
                                                     const DynamicExpression::TensorMap& outputs,
                                                     Stream& stream) {
            auto maskIt = inputs.find(Thor::BATCH_VALIDITY_MASK_NAME);
            if (maskIt == inputs.end())
                throw std::invalid_argument("CustomLoss masked gradient requires a batch validity mask input.");

            DynamicExpression::TensorMap expressionInputs = inputs;
            if (!expressionUsesBatchValidityMask)
                expressionInputs.erase(Thor::BATCH_VALIDITY_MASK_NAME);
            DynamicExpressionBuild build = expression.build(expressionInputs, {}, stream);
            const PhysicalOutputs& rawOutputs = build.equation->physicalOutputs();
            PhysicalOutputs maskedOutputs = detail::transformDynamicExpressionOutputsRecursively(
                rawOutputs,
                [gradientName, gradientDataType](const std::string& outputName, const Expression& raw) {
                    if (outputName != gradientName) {
                        throw std::runtime_error(
                            "CustomLoss batch validity masking encountered unexpected gradient output '" + outputName + "'.");
                    }
                    Expression mask =
                        Expression::input(Thor::BATCH_VALIDITY_MASK_NAME, DataType::FP32, DataType::FP32);
                    return (raw * mask).withOutputDType(gradientDataType);
                },
                "CustomLoss batch validity masking");

            build.stamp_inputs.emplace(Thor::BATCH_VALIDITY_MASK_NAME, maskIt->second);
            return DynamicExpressionBuild{
                .equation = std::make_shared<FusedEquation>(FusedEquation::compile(maskedOutputs, stream.getGpuNum())),
                .stamp_inputs = std::move(build.stamp_inputs),
                .tensor_scalar_inputs = std::move(build.tensor_scalar_inputs),
                .preallocated_outputs = outputs,
                .requested_output_shapes = std::move(build.requested_output_shapes),
                .pre_forward_hook = std::move(build.pre_forward_hook),
                .pre_forward_only_inputs = std::move(build.pre_forward_only_inputs),
            };
        });
}

}  // namespace

CustomLoss::CustomLoss(DynamicExpression lossExpression,
                       DynamicExpression gradientExpression,
                       std::string predictionsName,
                       std::string labelsName,
                       std::string lossName,
                       std::string gradientName,
                       DataType lossDataType,
                       std::optional<float> lossWeight,
                       bool usesBatchValidity,
                       bool requiresFullBatch)
    : Loss(lossDataType),
      lossExpression(std::move(lossExpression)),
      gradientExpression(std::move(gradientExpression)),
      predictionsName(std::move(predictionsName)),
      labelsName(std::move(labelsName)),
      lossName(std::move(lossName)),
      gradientName(std::move(gradientName)),
      lossWeight(normalizeLossWeight(lossWeight)),
      batchValidityMaskEnabled(usesBatchValidity),
      fullBatchRequired(requiresFullBatch) {
    if (batchValidityMaskEnabled && fullBatchRequired)
        throw std::invalid_argument("CustomLoss cannot both use batch validity and require a full batch.");
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
    if (batchValidityMaskEnabled) {
        THOR_THROW_IF_FALSE(batchValidityMask.isInitialized());
        inputs.emplace(Thor::BATCH_VALIDITY_MASK_NAME, batchValidityMask);
    }
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

    const std::unordered_map<std::string, DataType> outputDTypes =
        build.equation->getOutputDataTypes(build.stamp_inputs, build.tensor_scalar_inputs);
    auto dtypeIt = outputDTypes.find(outputName);
    if (dtypeIt == outputDTypes.end()) {
        throw std::runtime_error("CustomLoss " + what + " expression did not infer output dtype for '" + outputName + "'.");
    }

    return {shapeIt->second, dtypeIt->second};
}

std::optional<Tensor> CustomLoss::createFeatureOutputTensor() {
    const auto [outputShape, outputDType] = inferExpressionOutputDescriptor(weightedLossExpression(), lossName, "loss");
    THOR_THROW_IF_FALSE(outputDType == lossDataType);
    THOR_THROW_IF_FALSE(featureInput.has_value());
    return Tensor(featureInput.value().getPlacement(), TensorDescriptor(outputDType, outputShape));
}

DynamicExpression CustomLoss::weightedLossExpression() const {
    return applyLossWeightToDynamicExpression(lossExpression, {{lossName, lossDataType}}, lossWeight, "CustomLoss loss");
}

DynamicExpression CustomLoss::weightedGradientExpression() const {
    THOR_THROW_IF_FALSE(featureInput.has_value());
    return applyLossWeightToDynamicExpression(gradientExpression,
                                              {{gradientName, featureInput.value().getDescriptor().getDataType()}},
                                              lossWeight,
                                              "CustomLoss gradient");
}

DynamicExpression CustomLoss::maskedWeightedGradientExpression() const {
    THOR_THROW_IF_FALSE(featureInput.has_value());
    return applyBatchValidityMaskToGradient(
        weightedGradientExpression(),
        gradientName,
        featureInput.value().getDescriptor().getDataType(),
        batchValidityMaskEnabled);
}

void CustomLoss::tryFuseGradientIntoDrivingLayer() {
    if (gradientFusedIntoDrivingLayer || isInferenceOnly()) {
        return;
    }
    if (!previousLayer.has_value() || !featureInput.has_value() || !labelsInput.has_value() || !errorOutput.has_value()) {
        return;
    }

    auto* customLayer = dynamic_cast<CustomLayer*>(previousLayer.value());
    if (customLayer != nullptr) {
        gradientFusedIntoDrivingLayer = customLayer->registerFusedCustomLossGradient(featureInput.value(),
                                                                                     labelsInput.value(),
                                                                                     maskedWeightedGradientExpression(),
                                                                                     predictionsName,
                                                                                     labelsName,
                                                                                     gradientName,
                                                                                     batchValidityMask,
                                                                                     Thor::BATCH_VALIDITY_MASK_NAME,
                                                                                     this);
        return;
    }

    // A reporting/debug output connected to the predictions tensor inserts a TensorFanout
    // between the driving CustomLayer and the loss.  The fanout still forwards the same
    // predictions tensor, so let it register the fused loss gradient on its single upstream
    // CustomLayer while the loss forward path below continues to materialize into featureOutput.
    auto* tensorFanout = dynamic_cast<TensorFanout*>(previousLayer.value());
    if (tensorFanout != nullptr) {
        gradientFusedIntoDrivingLayer = tensorFanout->registerFusedCustomLossGradientWithDrivingLayer(featureInput.value(),
                                                                                                      labelsInput.value(),
                                                                                                      maskedWeightedGradientExpression(),
                                                                                                      predictionsName,
                                                                                                      labelsName,
                                                                                                      gradientName,
                                                                                                      batchValidityMask,
                                                                                                      Thor::BATCH_VALIDITY_MASK_NAME,
                                                                                                      this);
    }
}

void CustomLoss::notifyFusedGradientUnregisteredFromDrivingLayer(const Tensor& predictions) {
    if (featureInput.has_value() && featureInput.value() == predictions) {
        gradientFusedIntoDrivingLayer = false;
    }
}

void CustomLoss::notifyFusedGradientConsumptionComplete(const Event& consumersDone) {
    THOR_THROW_IF_FALSE(gradientFusedIntoDrivingLayer);
    THOR_THROW_IF_FALSE(consumersDone.isInitialized());

    // The fused gradient is a deferred consumer of both tensors.  Their ordinary
    // forward-side waits only cover the CustomLoss forward expression; they do
    // not cover the later read performed from the driving CustomLayer's backward
    // expression.  Make that deferred lifetime explicit on the streams that are
    // allowed to reuse/rewrite the tensors for the next batch.
    labelsStream.waitEvent(consumersDone);
    stream.waitEvent(consumersDone);
}

std::optional<Tensor> CustomLoss::connectToPredictionsInputLayer(Layer* predictionsInputLayer,
                                                                 std::optional<Tensor> featureInput,
                                                                 Stream stream,
                                                                 bool backPropagateError) {
    std::optional<Tensor> error = Loss::connectToPredictionsInputLayer(predictionsInputLayer, featureInput, stream, backPropagateError);
    THOR_THROW_IF_FALSE(this->featureInput.has_value());
    std::vector<uint64_t> maskDimensions = this->featureInput.value().getDimensions();
    for (size_t axis = 1; axis < maskDimensions.size(); ++axis)
        maskDimensions[axis] = 1;
    batchValidityMask = Tensor(this->featureInput.value().getPlacement(), TensorDescriptor(DataType::FP32, maskDimensions));
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
    if (gradientFusedIntoDrivingLayer || batchValidityMaskEnabled) {
        THOR_THROW_IF_FALSE(batchValidityMask.isInitialized());
        THOR_THROW_IF_FALSE(batchValidityMask.getDataType() == DataType::FP32);
        THOR_THROW_IF_FALSE(batchValidityMask.getPlacement() == featureInput.value().getPlacement());
        const std::vector<uint64_t> maskDimensions = batchValidityMask.getDimensions();
        const std::vector<uint64_t> predictionDimensions = featureInput.value().getDimensions();
        THOR_THROW_IF_FALSE(maskDimensions.size() == predictionDimensions.size());
        THOR_THROW_IF_FALSE(maskDimensions.front() == predictionDimensions.front());
        for (size_t axis = 1; axis < maskDimensions.size(); ++axis)
            THOR_THROW_IF_FALSE(maskDimensions[axis] == 1);
    } else {
        batchValidityMask.dropReference();
    }

    validateExpressionOutputNames(lossExpression, lossName, "loss");
    validateExpressionOutputNames(gradientExpression, gradientName, "gradient");

    TensorMap inputs = buildLossInputs();
    // Always stamp the loss forward expression into featureOutput.  Fusing the loss gradient
    // only changes the backward seed path; it must not redirect or duplicate the materialized
    // loss tensor used by NetworkOutput/LossShaper/stat reporting.
    TensorMap lossOutputs = buildLossOutputs();
    lossPrepared = std::make_shared<PreparedDynamicExpression>(weightedLossExpression().prepare(inputs, lossOutputs, stream));
    lossPreRunHook = lossPrepared->preForwardHook();
    lossStamped = std::make_shared<StampedExecutionPlan>(lossPrepared->stamp(lossOutputs));
    validateExpressionOutputNames(lossExpression, lossName, "loss");

    if (!isInferenceOnly() && !gradientFusedIntoDrivingLayer) {
        THOR_THROW_IF_FALSE(errorOutput.has_value());
        THOR_THROW_IF_FALSE(errorOutput.value().isInitialized());
        THOR_THROW_IF_FALSE(errorOutput.value().getPlacement() == featureInput.value().getPlacement());
        THOR_THROW_IF_FALSE(errorOutput.value().getDescriptor() == featureInput.value().getDescriptor());

        TensorMap gradientOutputs = buildGradientOutputs();
        gradientPrepared =
            std::make_shared<PreparedDynamicExpression>(weightedGradientExpression().prepare(inputs, gradientOutputs, stream));
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
    batchValidityMask.dropReference();
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
    if (gradientFusedIntoDrivingLayer || batchValidityMaskEnabled)
        writeBatchValidityMask(batchValidityMask, getValidExampleCount(), runStream);
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
