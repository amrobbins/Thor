#include "DeepLearning/Implementation/Layers/Metrics/CustomMetric.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Expression/FusedEquation.h"
#include "Utilities/TensorOperations/Masking/BatchValidity.h"

#include <algorithm>
#include <limits>
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

uint64_t elementCount(const std::vector<uint64_t>& dimensions) {
    uint64_t count = 1;
    for (uint64_t dimension : dimensions) {
        if (dimension != 0 && count > std::numeric_limits<uint64_t>::max() / dimension)
            throw std::overflow_error("CustomMetric output element count overflow.");
        count *= dimension;
    }
    return count;
}

std::optional<DataType> finalOutputDType(const std::shared_ptr<CompiledOutputs>& compiledOutputs,
                                        const std::string& outputName) {
    for (const CompiledExecutionStage& stage : compiledOutputs->stages) {
        for (size_t outputIndex = 0; outputIndex < stage.outputs.size(); ++outputIndex) {
            if (stage.outputs[outputIndex].name == outputName)
                return stage.outputDType(outputIndex);
        }
    }
    for (const CompiledStageOutput& finalOutput : compiledOutputs->final_outputs) {
        if (finalOutput.name != outputName)
            continue;
        for (const CompiledExecutionStage& stage : compiledOutputs->stages) {
            for (size_t outputIndex = 0; outputIndex < stage.outputs.size(); ++outputIndex) {
                if (stage.outputs[outputIndex].value_id == finalOutput.value_id)
                    return stage.outputDType(outputIndex);
            }
        }
    }
    return std::nullopt;
}

}  // namespace

CustomMetric::CustomMetric(DynamicExpression expr,
                           std::string predictionsName,
                           std::string labelsName,
                           std::string metricName,
                           std::string displayName,
                           Thor::MetricAggregation aggregation,
                           std::optional<std::string> batchValidityMaskName)
    : metricExpression(std::move(expr)),
      predictionsName(std::move(predictionsName)),
      labelsName(std::move(labelsName)),
      metricName(std::move(metricName)),
      displayName(std::move(displayName)),
      aggregation(aggregation),
      batchValidityMaskName(std::move(batchValidityMaskName)) {
    if (this->predictionsName.empty())
        throw std::invalid_argument("CustomMetric predictions input name cannot be empty.");
    if (this->metricName.empty())
        throw std::invalid_argument("CustomMetric metric output name cannot be empty.");
    if (!this->labelsName.empty() && this->predictionsName == this->labelsName)
        throw std::invalid_argument("CustomMetric predictions and labels input names must be distinct.");
    if (this->displayName.empty())
        this->displayName = "Metric";
    if (this->batchValidityMaskName.has_value()) {
        if (this->batchValidityMaskName.value().empty())
            throw std::invalid_argument("CustomMetric batch-validity mask input name cannot be empty.");
        if (this->batchValidityMaskName.value() == this->predictionsName ||
            this->batchValidityMaskName.value() == this->labelsName) {
            throw std::invalid_argument("CustomMetric batch-validity mask input name must be distinct.");
        }
    }
}

std::vector<std::string> CustomMetric::expectedMetricOutputNames() const {
    std::vector<std::string> names{metricName};
    if (isRatioMetric()) {
        names.emplace_back(Thor::METRIC_AGGREGATION_NUMERATOR_NAME);
        names.emplace_back(Thor::METRIC_AGGREGATION_DENOMINATOR_NAME);
    }
    return names;
}

std::optional<Tensor> CustomMetric::connectToFeatureInputLayer(Layer* featureInputLayer,
                                                               std::optional<Tensor> featureInput,
                                                               Stream stream,
                                                               bool backPropagateError) {
    std::optional<Tensor> error =
        Metric::connectToFeatureInputLayer(featureInputLayer, featureInput, stream, backPropagateError);
    if (batchValidityMaskName.has_value()) {
        THOR_THROW_IF_FALSE(this->featureInput.has_value());
        std::vector<uint64_t> maskDimensions = this->featureInput.value().getDimensions();
        for (size_t axis = 1; axis < maskDimensions.size(); ++axis)
            maskDimensions[axis] = 1;
        batchValidityMask = Tensor(
            this->featureInput.value().getPlacement(), TensorDescriptor(DataType::FP32, maskDimensions));
    }
    return error;
}

CustomMetric::TensorMap CustomMetric::buildMetricInputs() const {
    THOR_THROW_IF_FALSE(featureInput.has_value());
    THOR_THROW_IF_FALSE(!requiresLabelsInput() || labelsInput.has_value());

    TensorMap inputs;
    inputs.emplace(predictionsName, featureInput.value());
    if (requiresLabelsInput())
        inputs.emplace(labelsName, labelsInput.value());
    if (batchValidityMaskName.has_value()) {
        THOR_THROW_IF_FALSE(batchValidityMask.isInitialized());
        inputs.emplace(batchValidityMaskName.value(), batchValidityMask);
    }
    return inputs;
}

CustomMetric::TensorMap CustomMetric::buildMetricOutputs() const {
    THOR_THROW_IF_FALSE(featureOutput.has_value());

    TensorMap outputs;
    outputs.emplace(metricName, featureOutput.value());
    if (isRatioMetric()) {
        THOR_THROW_IF_FALSE(ratioNumerator.isInitialized());
        THOR_THROW_IF_FALSE(ratioDenominator.isInitialized());
        outputs.emplace(Thor::METRIC_AGGREGATION_NUMERATOR_NAME, ratioNumerator);
        outputs.emplace(Thor::METRIC_AGGREGATION_DENOMINATOR_NAME, ratioDenominator);
    }
    return outputs;
}

void CustomMetric::validateMetricOutputNames(const std::vector<std::string>& outputNames) const {
    const std::set<std::string> actual = toNameSet(outputNames);
    const std::vector<std::string> expectedNames = expectedMetricOutputNames();
    const std::set<std::string> expected(expectedNames.begin(), expectedNames.end());
    if (actual != expected) {
        throw std::runtime_error("CustomMetric expression output name mismatch. Expected {" + joinNames(expected) + "}, got {" +
                                 joinNames(actual) + "}.");
    }
}

std::unordered_map<std::string, CustomMetric::OutputDescriptor> CustomMetric::inferMetricOutputDescriptors() const {
    THOR_THROW_IF_FALSE(featureInput.has_value());
    THOR_THROW_IF_FALSE(!requiresLabelsInput() || labelsInput.has_value());
    THOR_THROW_IF_FALSE(stream.isInitialized());

    DynamicExpressionBuild build = metricExpression.build(buildMetricInputs(), {}, const_cast<Stream&>(stream));
    validateMetricOutputNames(build.equation->getOutputNames());

    const std::unordered_map<std::string, std::vector<uint64_t>> outputShapes =
        build.equation->getOutputShapes(build.stamp_inputs, build.tensor_scalar_inputs);
    const std::shared_ptr<CompiledOutputs> compiledOutputs =
        build.equation->compileForInputs(build.stamp_inputs, {}, build.tensor_scalar_inputs);

    std::unordered_map<std::string, OutputDescriptor> descriptors;
    for (const std::string& outputName : expectedMetricOutputNames()) {
        auto shapeIt = outputShapes.find(outputName);
        if (shapeIt == outputShapes.end())
            throw std::runtime_error("CustomMetric expression did not infer output shape for '" + outputName + "'.");
        const std::optional<DataType> outputDType = finalOutputDType(compiledOutputs, outputName);
        if (!outputDType.has_value())
            throw std::runtime_error("CustomMetric expression did not infer output dtype for '" + outputName + "'.");
        descriptors.emplace(outputName, OutputDescriptor{shapeIt->second, outputDType.value()});
    }
    return descriptors;
}

std::optional<Tensor> CustomMetric::createFeatureOutputTensor() {
    const std::unordered_map<std::string, OutputDescriptor> descriptors = inferMetricOutputDescriptors();
    THOR_THROW_IF_FALSE(featureInput.has_value());

    const OutputDescriptor& metricDescriptor = descriptors.at(metricName);
    if (isRatioMetric()) {
        if (elementCount(metricDescriptor.dimensions) != 1)
            throw std::runtime_error("CustomMetric ratio metric output '" + metricName + "' must be scalar.");
        const OutputDescriptor& numeratorDescriptor = descriptors.at(Thor::METRIC_AGGREGATION_NUMERATOR_NAME);
        const OutputDescriptor& denominatorDescriptor = descriptors.at(Thor::METRIC_AGGREGATION_DENOMINATOR_NAME);
        for (const auto& [name, descriptor] : {
                 std::pair<std::string, OutputDescriptor>{Thor::METRIC_AGGREGATION_NUMERATOR_NAME, numeratorDescriptor},
                 std::pair<std::string, OutputDescriptor>{Thor::METRIC_AGGREGATION_DENOMINATOR_NAME, denominatorDescriptor}}) {
            if (elementCount(descriptor.dimensions) != 1)
                throw std::runtime_error("CustomMetric ratio statistic output '" + name + "' must be scalar.");
            if (descriptor.dataType != DataType::FP32)
                throw std::runtime_error("CustomMetric ratio statistic output '" + name + "' must be FP32.");
        }
        ratioNumerator = Tensor(featureInput.value().getPlacement(),
                                TensorDescriptor(numeratorDescriptor.dataType, numeratorDescriptor.dimensions));
        ratioDenominator = Tensor(featureInput.value().getPlacement(),
                                  TensorDescriptor(denominatorDescriptor.dataType, denominatorDescriptor.dimensions));
    }

    return Tensor(featureInput.value().getPlacement(),
                  TensorDescriptor(metricDescriptor.dataType, metricDescriptor.dimensions));
}

void CustomMetric::compileImpl() {
    Metric::compileImpl();

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
    if (batchValidityMaskName.has_value()) {
        THOR_THROW_IF_FALSE(batchValidityMask.isInitialized());
        THOR_THROW_IF_FALSE(batchValidityMask.getDataType() == DataType::FP32);
        THOR_THROW_IF_FALSE(batchValidityMask.getPlacement() == featureInput.value().getPlacement());
        const std::vector<uint64_t> maskDimensions = batchValidityMask.getDimensions();
        const std::vector<uint64_t> inputDimensions = featureInput.value().getDimensions();
        THOR_THROW_IF_FALSE(maskDimensions.size() == inputDimensions.size());
        THOR_THROW_IF_FALSE(maskDimensions.front() == inputDimensions.front());
        for (size_t axis = 1; axis < maskDimensions.size(); ++axis)
            THOR_THROW_IF_FALSE(maskDimensions[axis] == 1);
    }
    if (isRatioMetric()) {
        THOR_THROW_IF_FALSE(ratioNumerator.isInitialized());
        THOR_THROW_IF_FALSE(ratioDenominator.isInitialized());
        THOR_THROW_IF_FALSE(ratioNumerator.getDataType() == DataType::FP32);
        THOR_THROW_IF_FALSE(ratioDenominator.getDataType() == DataType::FP32);
        THOR_THROW_IF_FALSE(ratioNumerator.getTotalNumElements() == 1);
        THOR_THROW_IF_FALSE(ratioDenominator.getTotalNumElements() == 1);
    }

    TensorMap inputs = buildMetricInputs();
    TensorMap outputs = buildMetricOutputs();
    metricPrepared = std::make_shared<PreparedDynamicExpression>(metricExpression.prepare(inputs, outputs, stream));
    metricPreRunHook = metricPrepared->preForwardHook();
    metricStamped = std::make_shared<StampedExecutionPlan>(metricPrepared->stamp(outputs));
    validateMetricOutputNames(metricStamped->outputNames());
    if (isRatioMetric())
        allocateRatioStatisticSlots(1);
}

void CustomMetric::cleanup() {
    metricStamped.reset();
    metricPrepared.reset();
    metricPreRunHook = {};
    batchValidityMask.dropReference();
    ratioNumerator.dropReference();
    ratioDenominator.dropReference();
    for (RatioStatisticSlot& slot : ratioStatisticSlots) {
        slot.numeratorHost.dropReference();
        slot.denominatorHost.dropReference();
        slot.numeratorBuffer.dropReference();
        slot.denominatorBuffer.dropReference();
    }
    ratioStatisticSlots.clear();
    ratioStatisticDownloadStream.reset();
    Metric::cleanup();
}

void CustomMetric::computeMetric(
    Tensor labels, Tensor predictions, Tensor metric, Stream stream, uint32_t validExampleCount) {
    THOR_THROW_IF_FALSE(stream == this->stream);
    THOR_THROW_IF_FALSE(metricStamped != nullptr);
    THOR_THROW_IF_FALSE(!requiresLabelsInput() || labelsInput.has_value());
    THOR_THROW_IF_FALSE(featureInput.has_value());
    THOR_THROW_IF_FALSE(featureOutput.has_value());
    if (requiresLabelsInput())
        THOR_THROW_IF_FALSE(labels == labelsInput.value());
    THOR_THROW_IF_FALSE(predictions == featureInput.value());
    THOR_THROW_IF_FALSE(metric == featureOutput.value());
    THOR_THROW_IF_FALSE(validExampleCount >= 1);
    THOR_THROW_IF_FALSE(validExampleCount <= getPhysicalBatchCapacity());
    if (batchValidityMaskName.has_value()) {
        writeBatchValidityMask(batchValidityMask, validExampleCount, this->stream);
    } else if (validExampleCount != getPhysicalBatchCapacity()) {
        throw std::logic_error("CustomMetric does not define exact partial-batch semantics.");
    }

    if (metricPreRunHook)
        metricPreRunHook(this->stream);
    metricStamped->run();
    if (isRatioMetric())
        captureRatioStatistics();
}

void CustomMetric::allocateRatioStatisticSlots(uint32_t numSlots) {
    THOR_THROW_IF_FALSE(numSlots >= 1);
    if (!isRatioMetric())
        return;
    THOR_THROW_IF_FALSE(ratioNumerator.isInitialized());
    THOR_THROW_IF_FALSE(ratioDenominator.isInitialized());
    if (!ratioStatisticDownloadStream.has_value())
        ratioStatisticDownloadStream = Stream::getNextDownloadStream(ratioNumerator.getPlacement().getDeviceNum());

    const TensorPlacement hostPlacement(TensorPlacement::MemDevices::CPU);
    while (ratioStatisticSlots.size() < numSlots) {
        RatioStatisticSlot slot;
        slot.numeratorHost = ratioNumerator.clone(hostPlacement);
        slot.denominatorHost = ratioDenominator.clone(hostPlacement);
        slot.numeratorBuffer = ratioNumerator.clone();
        slot.denominatorBuffer = ratioDenominator.clone();
        ratioStatisticSlots.push_back(std::move(slot));
    }
}

void CustomMetric::preallocateMetricStatisticSlots(uint32_t numSlots) { allocateRatioStatisticSlots(numSlots); }

void CustomMetric::requireRatioStatisticSlot(uint32_t slotIndex) const {
    THOR_THROW_IF_FALSE(isRatioMetric());
    THOR_THROW_IF_FALSE(slotIndex < ratioStatisticSlots.size());
}

void CustomMetric::setActiveMetricStatisticSlot(uint32_t slotIndex) {
    if (!isRatioMetric()) {
        (void)slotIndex;
        return;
    }
    requireRatioStatisticSlot(slotIndex);
    activeMetricStatisticSlot = slotIndex;
}

void CustomMetric::captureRatioStatistics() {
    requireRatioStatisticSlot(activeMetricStatisticSlot);
    THOR_THROW_IF_FALSE(ratioStatisticDownloadStream.has_value());
    RatioStatisticSlot& slot = ratioStatisticSlots[activeMetricStatisticSlot];

    if (slot.readyEvent.isInitialized())
        stream.waitEvent(slot.readyEvent);
    slot.numeratorBuffer.copyFromAsync(ratioNumerator, stream);
    slot.denominatorBuffer.copyFromAsync(ratioDenominator, stream);
    stream.putEvent(slot.bufferReadyEvent);

    Stream& downloadStream = ratioStatisticDownloadStream.value();
    downloadStream.waitEvent(slot.bufferReadyEvent);
    if (slot.writableEvent.isInitialized())
        downloadStream.waitEvent(slot.writableEvent);
    slot.numeratorHost.copyFromAsync(slot.numeratorBuffer, downloadStream);
    slot.denominatorHost.copyFromAsync(slot.denominatorBuffer, downloadStream);
    downloadStream.putEvent(slot.readyEvent, false, true);
    slot.writableEvent = slot.readyEvent;
}

std::optional<MetricBatchStatisticTensors> CustomMetric::getMetricBatchStatisticTensorsForSlot(uint32_t slotIndex) const {
    if (!isRatioMetric()) {
        (void)slotIndex;
        return MetricBatchStatisticTensors{aggregation, std::nullopt, std::nullopt, Event{}};
    }
    requireRatioStatisticSlot(slotIndex);
    const RatioStatisticSlot& slot = ratioStatisticSlots[slotIndex];
    return MetricBatchStatisticTensors{aggregation, slot.numeratorHost, slot.denominatorHost, slot.readyEvent};
}

void CustomMetric::extendMetricStatisticWritableEventForSlot(uint32_t slotIndex, Event event) {
    if (!isRatioMetric()) {
        (void)slotIndex;
        (void)event;
        return;
    }
    requireRatioStatisticSlot(slotIndex);
    ratioStatisticSlots[slotIndex].writableEvent = event;
}

std::vector<Event> CustomMetric::getSynchronizeEvents() {
    std::vector<Event> events = Metric::getSynchronizeEvents();
    if (ratioStatisticDownloadStream.has_value())
        events.emplace_back(ratioStatisticDownloadStream.value().putEvent(false, true));
    return events;
}

std::string CustomMetric::toDisplayString(Tensor metric_h) {
    THOR_THROW_IF_FALSE(metric_h.getPlacement().getMemDevice() == TensorPlacement::MemDevices::CPU);
    if (metric_h.getDescriptor().getDataType() == DataType::FP32 && metric_h.getTotalNumElements() == 1) {
        return displayName + ": " + std::to_string(*metric_h.getMemPtr<float>());
    }
    return displayName;
}

}  // namespace ThorImplementation
