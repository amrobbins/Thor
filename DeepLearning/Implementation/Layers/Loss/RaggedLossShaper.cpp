#include "DeepLearning/Implementation/Layers/Loss/RaggedLossShaper.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/RaggedTensorDescriptor.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionDescriptor.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/FusedEquation.h"
#include "Utilities/Expression/RaggedExpression.h"

#include <limits>
#include <set>
#include <stdexcept>
#include <utility>

using namespace std;

namespace ThorImplementation {
namespace {

constexpr const char* kValuesInputName = "__thor_ragged_loss_values";
constexpr const char* kOffsetsInputName = "__thor_ragged_loss_offsets";
constexpr const char* kPerExampleOutputName = "__thor_ragged_per_example_loss";

uint64_t checkedElementsPerValue(const vector<uint64_t>& dimensions) {
    if (dimensions.empty())
        throw invalid_argument("RaggedLossShaper packed values must have rank >= 1.");
    uint64_t elements = 1;
    for (size_t axis = 1; axis < dimensions.size(); ++axis) {
        if (dimensions[axis] == 0 || elements > numeric_limits<uint64_t>::max() / dimensions[axis])
            throw invalid_argument("RaggedLossShaper trailing value element count overflows uint64_t.");
        elements *= dimensions[axis];
    }
    return elements;
}

}  // namespace

RaggedLossShaper::RaggedLossShaper(OutputLossType outputLossType, uint64_t batchSize, uint64_t maxTotalValues)
    : outputLossType(outputLossType), batchSize(batchSize), maxTotalValues(maxTotalValues) {
    if (batchSize == 0 || batchSize > numeric_limits<uint32_t>::max())
        throw invalid_argument("RaggedLossShaper logical batch size must fit in uint32_t and be non-zero.");
    if (maxTotalValues == 0)
        throw invalid_argument("RaggedLossShaper max_total_values must be non-zero.");
    if (outputLossType == OutputLossType::PER_OUTPUT)
        throw invalid_argument("RaggedLossShaper does not support PER_OUTPUT for unequal ragged rows.");
    setConstructForInferenceOnly(true);
}

optional<Tensor> RaggedLossShaper::createFeatureOutputTensor() {
    THOR_THROW_IF_FALSE(featureInput.has_value());
    switch (outputLossType) {
        case OutputLossType::RAW:
            return featureInput;
        case OutputLossType::PER_EXAMPLE:
            return featureInput.value().clone({batchSize, 1});
        case OutputLossType::BATCH:
            return featureInput.value().clone({1, 1});
        case OutputLossType::PER_OUTPUT:
            THOR_UNREACHABLE();
    }
    THOR_UNREACHABLE();
}

optional<Tensor> RaggedLossShaper::connectToPreviousLayer(Layer* previousLayer,
                                                          optional<Tensor> connectedInput,
                                                          Stream connectedStream,
                                                          bool backPropagateError,
                                                          int connectionType) {
    THOR_THROW_IF_FALSE(!compiled);
    if (!connectedInput.has_value())
        throw invalid_argument("RaggedLossShaper requires connected input tensors.");
    switch (static_cast<InputConnection>(connectionType)) {
        case InputConnection::VALUES:
            // Reporting layers are constructed inference-only, so the base
            // connection path will return no error tensor even when the
            // upstream graph requests backpropagation.  Let that normal
            // mechanism prune the reporting branch instead of rejecting a
            // training graph during wiring.
            return Layer::connectToPreviousLayer(previousLayer, connectedInput, connectedStream, backPropagateError, 0);
        case InputConnection::OFFSETS: {
            if (offsetsInput.has_value())
                throw logic_error("RaggedLossShaper offsets input is already connected.");
            const Tensor& offsets = connectedInput.value();
            if (!RowPartitionDescriptor::isValidOffsetsDataType(offsets.getDataType()))
                throw invalid_argument("RaggedLossShaper offsets dtype must be UINT32 or UINT64.");
            const RowPartitionDescriptor partition(batchSize, maxTotalValues, offsets.getDataType());
            if (offsets.getDescriptor() != partition.getOffsetsDescriptor())
                throw invalid_argument("RaggedLossShaper offsets must have canonical shape [batch_size + 1].");
            offsetsInput = offsets;
            offsetsStream = connectedStream;
            return nullopt;
        }
        default:
            throw invalid_argument("RaggedLossShaper input connection type is out of range.");
    }
}

uint64_t RaggedLossShaper::elementsPerValue() const {
    THOR_THROW_IF_FALSE(featureInput.has_value());
    return checkedElementsPerValue(featureInput.value().getDimensions());
}

DynamicExpression RaggedLossShaper::buildPerExampleExpression() const {
    THOR_THROW_IF_FALSE(featureInput.has_value());
    THOR_THROW_IF_FALSE(offsetsInput.has_value());
    const Tensor& valuesTensor = featureInput.value();
    const Tensor& offsetsTensor = offsetsInput.value();
    const vector<uint64_t> dimensions = valuesTensor.getDimensions();
    const vector<uint64_t> trailingDimensions(dimensions.begin() + 1, dimensions.end());
    const DataType valuesDType = valuesTensor.getDataType();
    const DataType offsetsDType = offsetsTensor.getDataType();
    const uint64_t runtimeBatchSize = batchSize;
    const uint64_t runtimeMaxTotalValues = maxTotalValues;

    return DynamicExpression(
        {kValuesInputName, kOffsetsInputName},
        {kPerExampleOutputName},
        [trailingDimensions, valuesDType, offsetsDType, runtimeBatchSize, runtimeMaxTotalValues](
            const TensorMap& inputs, const TensorMap& outputs, Stream& stream) -> DynamicExpressionBuild {
            const RaggedTensorDescriptor descriptor(valuesDType,
                                                    trailingDimensions,
                                                    runtimeBatchSize,
                                                    runtimeMaxTotalValues,
                                                    offsetsDType);
            const Expression values = Expression::input(kValuesInputName, DataType::FP32, valuesDType);
            const Expression offsets = Expression::input(kOffsetsInputName, nullopt, offsetsDType);
            const RaggedExpression ragged(values, offsets, descriptor);

            Expression perExample = ragged.segment_sum();
            if (!trailingDimensions.empty()) {
                vector<uint64_t> reductionAxes(trailingDimensions.size());
                for (uint64_t axis = 1; axis <= trailingDimensions.size(); ++axis)
                    reductionAxes[axis - 1] = axis;
                perExample = perExample.reduce_sum(reductionAxes, {}, DataType::FP32);
            }
            perExample = perExample.reshape({runtimeBatchSize, 1}).withOutputDType(valuesDType);

            return DynamicExpressionBuild{
                .equation = make_shared<FusedEquation>(
                    FusedEquation::compile(Expression::outputs({{kPerExampleOutputName, perExample}}).physicalOutputs(),
                                           stream.getGpuNum())),
                .stamp_inputs = inputs,
                .tensor_scalar_inputs = {},
                .preallocated_outputs = outputs,
                .requested_output_shapes = {},
            };
        });
}

void RaggedLossShaper::compileImpl() {
    Layer::compileImpl();
    if (!featureInput.has_value() || !offsetsInput.has_value() || !featureOutput.has_value())
        throw logic_error("RaggedLossShaper requires raw loss values, offsets, and an output consumer before compile.");

    const Tensor& values = featureInput.value();
    const Tensor& offsets = offsetsInput.value();
    const vector<uint64_t> dimensions = values.getDimensions();
    if (values.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU ||
        offsets.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU)
        throw invalid_argument("RaggedLossShaper currently requires GPU-resident inputs.");
    if (dimensions.empty() || dimensions.front() != maxTotalValues)
        throw invalid_argument("RaggedLossShaper raw loss values must have leading packed capacity max_total_values.");
    if (values.getDataType() != DataType::FP16 && values.getDataType() != DataType::FP32)
        throw invalid_argument("RaggedLossShaper raw loss dtype must be FP16 or FP32.");
    (void)elementsPerValue();
    ensureNoDeviceCrossing();

    perExampleWorkspace.reset();
    perExamplePrepared.reset();
    perExampleStamped.reset();
    perExamplePreRunHook = nullptr;
    batchReduction.reset();

    if (outputLossType == OutputLossType::RAW) {
        if (featureOutput.value() != featureInput.value())
            throw logic_error("RaggedLossShaper RAW output must alias its packed input values.");
        return;
    }

    if (outputLossType == OutputLossType::PER_EXAMPLE) {
        perExampleWorkspace = featureOutput.value();
    } else {
        perExampleWorkspace = values.clone({batchSize, 1});
    }

    const DynamicExpression expression = buildPerExampleExpression();
    TensorMap inputs{{kValuesInputName, values}, {kOffsetsInputName, offsets}};
    TensorMap outputs{{kPerExampleOutputName, perExampleWorkspace.value()}};
    perExamplePrepared = make_shared<PreparedDynamicExpression>(expression.prepare(inputs, outputs, stream));
    perExamplePreRunHook = perExamplePrepared->preForwardHook();
    perExampleStamped = make_shared<StampedExecutionPlan>(perExamplePrepared->stamp(outputs));

    if (outputLossType == OutputLossType::BATCH) {
        CubReduction reduction(CubReductionOp::Sum, std::vector<uint32_t>{0}, values.getDataType(), 1.0f);
        batchReduction = reduction.stamp(perExampleWorkspace.value(), featureOutput.value(), stream);
    }
}

void RaggedLossShaper::initialize() {
    Layer::initialize();
    valuesReceived = false;
    offsetsReceived = false;
    currentValidExampleCount = 0;
    batchCardinalitySet = false;
}

void RaggedLossShaper::cleanup() {
    batchReduction.reset();
    perExampleStamped.reset();
    perExamplePrepared.reset();
    perExamplePreRunHook = nullptr;
    perExampleWorkspace.reset();
    offsetsReadyEvent = Event();
    offsetsReusableEvent = Event();
    valuesReceived = false;
    offsetsReceived = false;
    batchCardinalitySet = false;
    Layer::cleanup();
}

uint32_t RaggedLossShaper::resolveValidExampleCount(uint32_t validExampleCount) const {
    const uint32_t logicalBatchSize = static_cast<uint32_t>(batchSize);
    const uint32_t resolved = validExampleCount == 0 ? logicalBatchSize : validExampleCount;
    if (resolved == 0 || resolved > logicalBatchSize)
        throw invalid_argument("RaggedLossShaper valid example count exceeds the logical row batch size.");
    return resolved;
}

void RaggedLossShaper::recordLogicalBatchCardinality(uint32_t validExampleCount) {
    const uint32_t resolved = resolveValidExampleCount(validExampleCount);
    if (batchCardinalitySet) {
        if (currentValidExampleCount != resolved)
            throw invalid_argument("RaggedLossShaper inputs disagreed on valid logical example count.");
        return;
    }
    currentValidExampleCount = resolved;
    batchCardinalitySet = true;
}

void RaggedLossShaper::forward(optional<Tensor> inputTensor, bool validationPass, uint32_t validExampleCount) {
    THOR_THROW_IF_FALSE(running);
    if (!inputTensor.has_value())
        throw invalid_argument("RaggedLossShaper forward requires an arriving connected input tensor.");
    recordLogicalBatchCardinality(validExampleCount);

    const Tensor& input = inputTensor.value();
    if (featureInput.has_value() && input == featureInput.value()) {
        if (valuesReceived)
            throw logic_error("RaggedLossShaper raw loss values arrived twice in one batch.");
        valuesReceived = true;
    } else if (offsetsInput.has_value() && input == offsetsInput.value()) {
        if (offsetsReceived)
            throw logic_error("RaggedLossShaper offsets arrived twice in one batch.");
        offsetsReceived = true;
    } else {
        throw invalid_argument("RaggedLossShaper received an unconnected input tensor.");
    }

    advanceDataIfReady(validationPass);
}

void RaggedLossShaper::advanceDataIfReady(bool validationPass) {
    if (!valuesReceived || !offsetsReceived)
        return;

    stream.waitFor(offsetsStream, offsetsReadyEvent);
    infer(featureInput, featureOutput, stream);
    offsetsStream.waitFor(stream, offsetsReusableEvent);

    valuesReceived = false;
    offsetsReceived = false;
    batchCardinalitySet = false;

    if (nextLayer.has_value())
        nextLayer.value()->forward(featureOutput, validationPass, currentValidExampleCount);
}

void RaggedLossShaper::infer(optional<Tensor> inputTensor, optional<Tensor> outputTensor, Stream runStream) {
    THOR_THROW_IF_FALSE(inputTensor.has_value());
    THOR_THROW_IF_FALSE(outputTensor.has_value());
    THOR_THROW_IF_FALSE(runStream == stream);

    if (outputLossType == OutputLossType::RAW) {
        THOR_THROW_IF_FALSE(inputTensor.value() == outputTensor.value());
        return;
    }

    THOR_THROW_IF_FALSE(perExampleStamped != nullptr);
    if (perExamplePreRunHook)
        perExamplePreRunHook(stream);
    perExampleStamped->run();

    if (outputLossType == OutputLossType::BATCH) {
        THOR_THROW_IF_FALSE(batchReduction != nullptr);
        batchReduction->runOn(stream, 1.0f / static_cast<float>(currentValidExampleCount));
    }
}

void RaggedLossShaper::backProp(optional<Tensor>, optional<Tensor>, optional<Tensor>, Stream) { THOR_UNREACHABLE(); }

vector<Stream> RaggedLossShaper::getProcessingStreams() {
    vector<Stream> result;
    set<uint64_t> ids;
    for (const Stream& candidate : {stream, offsetsStream}) {
        if (!candidate.isInitialized() || !ids.insert(candidate.getId()).second)
            continue;
        result.push_back(candidate);
    }
    return result;
}

vector<Event> RaggedLossShaper::getSynchronizeEvents() {
    vector<Event> events;
    set<uint64_t> ids;
    appendSynchronizeEvent(events, ids, stream);
    appendSynchronizeEvent(events, ids, offsetsStream);
    return events;
}

void RaggedLossShaper::ensureNoDeviceCrossing() {
    Layer::ensureNoDeviceCrossing();
    if (!offsetsInput.has_value())
        return;
    const TensorPlacement offsetsPlacement = offsetsInput.value().getPlacement();
    if (featureInput.has_value() && featureInput.value().getPlacement() != offsetsPlacement)
        throw invalid_argument("RaggedLossShaper raw loss values and offsets must share placement.");
    if (featureOutput.has_value() && featureOutput.value().getPlacement() != offsetsPlacement)
        throw invalid_argument("RaggedLossShaper output and offsets must share placement.");
}

}  // namespace ThorImplementation
