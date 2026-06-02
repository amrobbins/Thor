#include "DeepLearning/Implementation/Layers/Metrics/CustomMetric.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Expression/FusedEquation.h"

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

}  // namespace

CustomMetric::CustomMetric(DynamicExpression expr,
                           std::string predictionsName,
                           std::string labelsName,
                           std::string metricName,
                           std::string displayName)
    : metricExpression(std::move(expr)),
      predictionsName(std::move(predictionsName)),
      labelsName(std::move(labelsName)),
      metricName(std::move(metricName)),
      displayName(std::move(displayName)) {
    if (this->predictionsName.empty())
        throw std::invalid_argument("CustomMetric predictions input name cannot be empty.");
    if (this->metricName.empty())
        throw std::invalid_argument("CustomMetric metric output name cannot be empty.");
    if (!this->labelsName.empty() && this->predictionsName == this->labelsName)
        throw std::invalid_argument("CustomMetric predictions and labels input names must be distinct.");
    if (this->displayName.empty())
        this->displayName = "Metric";
}

CustomMetric::TensorMap CustomMetric::buildMetricInputs() const {
    THOR_THROW_IF_FALSE(featureInput.has_value());
    THOR_THROW_IF_FALSE(!requiresLabelsInput() || labelsInput.has_value());

    TensorMap inputs;
    inputs.emplace(predictionsName, featureInput.value());
    if (requiresLabelsInput())
        inputs.emplace(labelsName, labelsInput.value());
    return inputs;
}

CustomMetric::TensorMap CustomMetric::buildMetricOutputs() const {
    THOR_THROW_IF_FALSE(featureOutput.has_value());

    TensorMap outputs;
    outputs.emplace(metricName, featureOutput.value());
    return outputs;
}

void CustomMetric::validateMetricOutputNames(const std::vector<std::string>& outputNames) const {
    const std::set<std::string> actual = toNameSet(outputNames);
    const std::set<std::string> expected{metricName};
    if (actual != expected) {
        throw std::runtime_error("CustomMetric expression output name mismatch. Expected {" + joinNames(expected) + "}, got {" +
                                 joinNames(actual) + "}.");
    }
}

std::pair<std::vector<uint64_t>, DataType> CustomMetric::inferMetricOutputDescriptor() const {
    THOR_THROW_IF_FALSE(featureInput.has_value());
    THOR_THROW_IF_FALSE(!requiresLabelsInput() || labelsInput.has_value());
    THOR_THROW_IF_FALSE(stream.isInitialized());

    DynamicExpressionBuild build = metricExpression.build(buildMetricInputs(), {}, const_cast<Stream&>(stream));
    validateMetricOutputNames(build.equation->getOutputNames());

    std::unordered_map<std::string, std::vector<uint64_t>> outputShapes =
        build.equation->getOutputShapes(build.stamp_inputs, build.tensor_scalar_inputs);
    auto shapeIt = outputShapes.find(metricName);
    if (shapeIt == outputShapes.end()) {
        throw std::runtime_error("CustomMetric expression did not infer output shape for '" + metricName + "'.");
    }

    std::shared_ptr<CompiledOutputs> compiledOutputs =
        build.equation->compileForInputs(build.stamp_inputs, {}, build.tensor_scalar_inputs);

    std::optional<DataType> outputDType;
    for (const CompiledExecutionStage& stage : compiledOutputs->stages) {
        for (size_t outputIndex = 0; outputIndex < stage.outputs.size(); ++outputIndex) {
            const CompiledStageOutput& output = stage.outputs[outputIndex];
            if (output.name == metricName) {
                outputDType = stage.outputDType(outputIndex);
                break;
            }
        }
        if (outputDType.has_value())
            break;
    }

    if (!outputDType.has_value()) {
        for (const CompiledStageOutput& finalOutput : compiledOutputs->final_outputs) {
            if (finalOutput.name != metricName)
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
        throw std::runtime_error("CustomMetric expression did not infer output dtype for '" + metricName + "'.");

    return {shapeIt->second, outputDType.value()};
}

std::optional<Tensor> CustomMetric::createFeatureOutputTensor() {
    if (isInferenceOnly())
        return std::nullopt;

    const auto [outputShape, outputDType] = inferMetricOutputDescriptor();
    THOR_THROW_IF_FALSE(featureInput.has_value());
    return Tensor(featureInput.value().getPlacement(), TensorDescriptor(outputDType, outputShape));
}

void CustomMetric::compileImpl() {
    Metric::compileImpl();

    if (isInferenceOnly())
        return;

    THOR_THROW_IF_FALSE(featureInput.has_value());
    THOR_THROW_IF_FALSE(!requiresLabelsInput() || labelsInput.has_value());
    THOR_THROW_IF_FALSE(featureOutput.has_value());
    THOR_THROW_IF_FALSE(stream.isInitialized());
    THOR_THROW_IF_FALSE(!requiresLabelsInput() || labelsStream.isInitialized());
    THOR_THROW_IF_FALSE(featureInput.value().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    if (requiresLabelsInput()) {
        THOR_THROW_IF_FALSE(labelsInput.value().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        THOR_THROW_IF_FALSE(featureInput.value().getPlacement() == labelsInput.value().getPlacement());
    }
    THOR_THROW_IF_FALSE(featureOutput.value().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    THOR_THROW_IF_FALSE(featureInput.value().getPlacement() == featureOutput.value().getPlacement());

    TensorMap inputs = buildMetricInputs();
    TensorMap outputs = buildMetricOutputs();
    metricPrepared = std::make_shared<PreparedDynamicExpression>(metricExpression.prepare(inputs, outputs, stream));
    metricPreRunHook = metricPrepared->preForwardHook();
    metricStamped = std::make_shared<StampedExecutionPlan>(metricPrepared->stamp(outputs));
    validateMetricOutputNames(metricStamped->outputNames());
}

void CustomMetric::computeMetric(Tensor labels, Tensor predictions, Tensor metric, Stream stream) {
    THOR_THROW_IF_FALSE(stream == this->stream);
    THOR_THROW_IF_FALSE(metricStamped != nullptr);
    THOR_THROW_IF_FALSE(!requiresLabelsInput() || labelsInput.has_value());
    THOR_THROW_IF_FALSE(featureInput.has_value());
    THOR_THROW_IF_FALSE(featureOutput.has_value());
    if (requiresLabelsInput())
        THOR_THROW_IF_FALSE(labels == labelsInput.value());
    THOR_THROW_IF_FALSE(predictions == featureInput.value());
    THOR_THROW_IF_FALSE(metric == featureOutput.value());

    if (metricPreRunHook)
        metricPreRunHook(this->stream);
    metricStamped->run();
}

std::string CustomMetric::toDisplayString(Tensor metric_h) {
    THOR_THROW_IF_FALSE(metric_h.getPlacement().getMemDevice() == TensorPlacement::MemDevices::CPU);
    if (metric_h.getDescriptor().getDataType() == DataType::FP32 && metric_h.getTotalNumElements() == 1) {
        return displayName + ": " + std::to_string(*metric_h.getMemPtr<float>());
    }
    return displayName;
}

}  // namespace ThorImplementation
