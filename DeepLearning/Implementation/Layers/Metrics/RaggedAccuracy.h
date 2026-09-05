#pragma once

#include "DeepLearning/Implementation/Layers/Metrics/CustomMetric.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/FusedEquation.h"
#include "Utilities/TensorOperations/Ragged/RaggedAccuracy.h"
#include "Utilities/TensorOperations/Ragged/RowPartition.h"

#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ThorImplementation {
namespace RaggedAccuracyDetail {

struct RuntimeState {
    uint64_t validRowCount = 0;
};

enum class Kind { BINARY, CATEGORICAL };

inline bool isPredictionDTypeSupported(DataType dtype) {
    return dtype == DataType::FP16 || dtype == DataType::FP32;
}

inline bool isBinaryOrPerClassLabelDTypeSupported(DataType dtype) {
    return dtype == DataType::UINT8 || dtype == DataType::UINT16 || dtype == DataType::UINT32 ||
           dtype == DataType::INT8 || dtype == DataType::INT16 || dtype == DataType::INT32 ||
           dtype == DataType::FP16 || dtype == DataType::FP32;
}

inline bool isClassIndexLabelDTypeSupported(DataType dtype) {
    return dtype == DataType::UINT8 || dtype == DataType::UINT16 || dtype == DataType::UINT32 ||
           dtype == DataType::INT8 || dtype == DataType::INT16 || dtype == DataType::INT32;
}

inline DynamicExpression makeExpression(Kind kind,
                                        uint64_t batchSize,
                                        uint64_t maxTotalValues,
                                        uint64_t numClasses,
                                        RaggedCategoricalLabelFormat categoricalLabelFormat,
                                        std::shared_ptr<RuntimeState> runtimeState) {
    return DynamicExpression(
        {"predictions", "labels", "offsets"},
        {"metric", Thor::METRIC_AGGREGATION_NUMERATOR_NAME, Thor::METRIC_AGGREGATION_DENOMINATOR_NAME},
        [kind,
         batchSize,
         maxTotalValues,
         numClasses,
         categoricalLabelFormat,
         runtimeState](const DynamicExpression::TensorMap& inputs,
                       const DynamicExpression::TensorMap& outputs,
                       Stream& stream) -> DynamicExpressionBuild {
            const Tensor& predictions = inputs.at("predictions");
            const Tensor& labels = inputs.at("labels");
            const Tensor& offsets = inputs.at("offsets");
            if (!isPredictionDTypeSupported(predictions.getDataType()))
                throw std::invalid_argument("Ragged accuracy predictions must be FP16 or FP32.");
            const RowPartitionDescriptor partition(batchSize, maxTotalValues, offsets.getDataType());
            if (offsets.getDescriptor() != partition.getOffsetsDescriptor())
                throw std::invalid_argument("Ragged accuracy offsets must have canonical shape [batch_size + 1].");
            if (predictions.getDimensions().empty() || predictions.getDimensions().front() != maxTotalValues ||
                labels.getDimensions().empty() || labels.getDimensions().front() != maxTotalValues) {
                throw std::invalid_argument("Ragged accuracy packed tensors must use max_total_values as the leading dimension.");
            }

            if (kind == Kind::BINARY) {
                if (predictions.getDimensions() != std::vector<uint64_t>{maxTotalValues, 1} ||
                    labels.getDimensions() != std::vector<uint64_t>{maxTotalValues, 1}) {
                    throw std::invalid_argument(
                        "Ragged BinaryAccuracy predictions and labels must contain one scalar per packed token.");
                }
                if (!isBinaryOrPerClassLabelDTypeSupported(labels.getDataType()))
                    throw std::invalid_argument("Ragged BinaryAccuracy labels use an unsupported dtype.");
            } else {
                if (numClasses < 2 ||
                    predictions.getDimensions() != std::vector<uint64_t>{maxTotalValues, numClasses}) {
                    throw std::invalid_argument(
                        "Ragged CategoricalAccuracy predictions must have one trailing class dimension of width num_classes.");
                }
                if (categoricalLabelFormat == RaggedCategoricalLabelFormat::CLASS_INDEX) {
                    if (labels.getDimensions() != std::vector<uint64_t>{maxTotalValues, 1})
                        throw std::invalid_argument(
                            "Ragged CategoricalAccuracy class-index labels must contain one scalar per packed token.");
                    if (!isClassIndexLabelDTypeSupported(labels.getDataType()))
                        throw std::invalid_argument(
                            "Ragged CategoricalAccuracy class-index labels must use an integer dtype.");
                } else {
                    if (labels.getDimensions() != std::vector<uint64_t>{maxTotalValues, numClasses})
                        throw std::invalid_argument(
                            "Ragged CategoricalAccuracy per-class labels must match prediction class width.");
                    if (!isBinaryOrPerClassLabelDTypeSupported(labels.getDataType()))
                        throw std::invalid_argument("Ragged CategoricalAccuracy per-class labels use an unsupported dtype.");
                }
            }

            Tensor correctStatistic(predictions.getPlacement(), TensorDescriptor(DataType::FP32, {1}));
            Tensor tokenStatistic(predictions.getPlacement(), TensorDescriptor(DataType::FP32, {1}));
            const Expression correct =
                Expression::input("__thor_ragged_accuracy_correct", DataType::FP32, DataType::FP32);
            const Expression tokens =
                Expression::input("__thor_ragged_accuracy_tokens", DataType::FP32, DataType::FP32);
            const Expression metric = Expression::where(tokens == Expression(0.0), Expression(0.0), correct / tokens);
            const PhysicalOutputs expressionOutputs = Expression::outputs({
                {"metric", metric},
                {Thor::METRIC_AGGREGATION_NUMERATOR_NAME, correct},
                {Thor::METRIC_AGGREGATION_DENOMINATOR_NAME, tokens},
            }).physicalOutputs();

            DynamicExpressionBuild build{
                .equation = std::make_shared<FusedEquation>(FusedEquation::compile(expressionOutputs, stream.getGpuNum())),
                .stamp_inputs = {{"__thor_ragged_accuracy_correct", correctStatistic},
                                 {"__thor_ragged_accuracy_tokens", tokenStatistic}},
                .tensor_scalar_inputs = {},
                .preallocated_outputs = outputs,
                .requested_output_shapes = {},
                .pre_forward_hook = [kind,
                                     predictions,
                                     labels,
                                     offsets,
                                     correctStatistic,
                                     tokenStatistic,
                                     batchSize,
                                     maxTotalValues,
                                     numClasses,
                                     categoricalLabelFormat,
                                     runtimeState](Stream& runStream) mutable {
                    if (!runtimeState || runtimeState->validRowCount == 0 || runtimeState->validRowCount > batchSize)
                        throw std::logic_error("Ragged accuracy has invalid runtime valid-row count.");
                    if (kind == Kind::BINARY) {
                        raggedBinaryAccuracyStatistics(predictions,
                                                       labels,
                                                       offsets,
                                                       correctStatistic,
                                                       tokenStatistic,
                                                       runtimeState->validRowCount,
                                                       maxTotalValues,
                                                       runStream);
                    } else {
                        raggedCategoricalAccuracyStatistics(predictions,
                                                            labels,
                                                            offsets,
                                                            correctStatistic,
                                                            tokenStatistic,
                                                            runtimeState->validRowCount,
                                                            maxTotalValues,
                                                            numClasses,
                                                            categoricalLabelFormat,
                                                            runStream);
                    }
                },
            };
            build.pre_forward_only_inputs.emplace("predictions", predictions);
            build.pre_forward_only_inputs.emplace("labels", labels);
            build.pre_forward_only_inputs.emplace("offsets", offsets);
            return build;
        });
}

}  // namespace RaggedAccuracyDetail

class RaggedAccuracyMetric : public CustomMetric {
   public:
    RaggedAccuracyMetric(RaggedAccuracyDetail::Kind kind,
                         uint64_t batchSize,
                         uint64_t maxTotalValues,
                         uint64_t numClasses = 1,
                         RaggedCategoricalLabelFormat categoricalLabelFormat = RaggedCategoricalLabelFormat::PER_CLASS)
        : RaggedAccuracyMetric(kind,
                               batchSize,
                               maxTotalValues,
                               numClasses,
                               categoricalLabelFormat,
                               std::make_shared<RaggedAccuracyDetail::RuntimeState>()) {}

    bool supportsPartialBatches() const override { return true; }

    std::optional<Tensor> connectToPreviousLayer(Layer* previousLayer,
                                                 std::optional<Tensor> input,
                                                 Stream inputStream,
                                                 bool backPropagateError,
                                                 int connectionType = 0) override {
        if (connectionType == static_cast<int>(ConnectionType::FORWARD))
            return connectPredictions(previousLayer, input, inputStream, backPropagateError);
        if (connectionType == static_cast<int>(ConnectionType::LABELS))
            return connectLabels(previousLayer, input, inputStream);
        if (connectionType == static_cast<int>(ConnectionType::STRUCTURAL))
            return connectOffsets(previousLayer, input, inputStream);
        throw std::invalid_argument("Ragged accuracy received an unsupported connection type.");
    }

    std::vector<Stream> getProcessingStreams() override {
        std::vector<Stream> streams = CustomMetric::getProcessingStreams();
        if (offsetsStream.isInitialized())
            streams.push_back(offsetsStream);
        return streams;
    }

    std::vector<Event> getSynchronizeEvents() override {
        std::vector<Event> events = CustomMetric::getSynchronizeEvents();
        if (offsetsStream.isInitialized())
            events.emplace_back(offsetsStream.putEvent(false, true));
        return events;
    }

    void initialize() override {
        CustomMetric::initialize();
        offsetsReceived = false;
    }

    void cleanup() override {
        offsetsReadyEvent = Event();
        offsetsReusableEvent = Event();
        CustomMetric::cleanup();
    }

    void forward(std::optional<Tensor> inputTensor,
                 bool validationPass,
                 uint32_t validExampleCount = 0) override {
        THOR_THROW_IF_FALSE(running);
        if (!inputTensor.has_value())
            throw std::invalid_argument("Ragged accuracy forward requires a connected input tensor.");
        const uint32_t logicalBatch = static_cast<uint32_t>(batchSize);
        const uint32_t resolved = validExampleCount == 0 ? logicalBatch : validExampleCount;
        if (resolved == 0 || resolved > logicalBatch)
            throw std::invalid_argument("Ragged accuracy valid example count exceeds logical batch size.");
        if (batchCardinalitySet && currentValidExampleCount != resolved)
            throw std::invalid_argument("Ragged accuracy inputs disagreed on valid logical example count.");
        currentValidExampleCount = resolved;
        batchCardinalitySet = true;
        runtimeState->validRowCount = resolved;

        if (featureInput.has_value() && inputTensor.value() == featureInput.value()) {
            forwardFeatures(inputTensor.value(), validationPass);
        } else if (labelsInput.has_value() && inputTensor.value() == labelsInput.value()) {
            forwardLabels(inputTensor.value(), validationPass);
        } else if (offsetsInput.has_value() && inputTensor.value() == offsetsInput.value()) {
            if (offsetsReceived)
                throw std::logic_error("Ragged accuracy structural offsets were delivered twice for one batch.");
            offsetsReceived = true;
            advanceDataIfReady(validationPass);
        } else {
            throw std::invalid_argument("Ragged accuracy received an unconnected input tensor.");
        }
    }

    void computeMetric(Tensor labels,
                       Tensor predictions,
                       Tensor metric,
                       Stream runStream,
                       uint32_t validExampleCount) override {
        THOR_THROW_IF_FALSE(labelsInput.has_value() && labels == labelsInput.value());
        THOR_THROW_IF_FALSE(featureInput.has_value() && predictions == featureInput.value());
        THOR_THROW_IF_FALSE(featureOutput.has_value() && metric == featureOutput.value());
        THOR_THROW_IF_FALSE(validExampleCount >= 1 && validExampleCount <= batchSize);
        runtimeState->validRowCount = validExampleCount;
        runPreparedMetricExpression(runStream);
    }

    std::string getType() override {
        return kind == RaggedAccuracyDetail::Kind::BINARY ? "RaggedBinaryAccuracy" : "RaggedCategoricalAccuracy";
    }

   protected:
    TensorMap buildMetricInputs() const override {
        TensorMap inputs = CustomMetric::buildMetricInputs();
        if (!offsetsInput.has_value())
            throw std::logic_error("Ragged accuracy structural offsets are not connected.");
        inputs.emplace("offsets", offsetsInput.value());
        return inputs;
    }

    void compileImpl() override {
        validateInputs();
        CustomMetric::compileImpl();
    }

    void advanceDataIfReady(bool validationPass) override {
        if (!(featureInputReceived && labelsReceived && offsetsReceived))
            return;
        THOR_THROW_IF_FALSE(batchCardinalitySet);
        THOR_THROW_IF_FALSE(labelsInput.has_value());
        THOR_THROW_IF_FALSE(offsetsInput.has_value());

        waitForLabelsReady();
        stream.waitFor(offsetsStream, offsetsReadyEvent);
        computeMetric(labelsInput.value(), featureInput.value(), featureOutput.value(), stream, currentValidExampleCount);
        markLabelsReusableAfterCompute();
        offsetsStream.waitFor(stream, offsetsReusableEvent);

        featureInputReceived = false;
        labelsReceived = false;
        offsetsReceived = false;
        batchCardinalitySet = false;

        if (nextLayer.has_value())
            nextLayer.value()->forward(featureOutput, validationPass, currentValidExampleCount);
    }

   private:
    RaggedAccuracyMetric(RaggedAccuracyDetail::Kind kind,
                         uint64_t batchSize,
                         uint64_t maxTotalValues,
                         uint64_t numClasses,
                         RaggedCategoricalLabelFormat categoricalLabelFormat,
                         std::shared_ptr<RaggedAccuracyDetail::RuntimeState> runtimeState)
        : CustomMetric(RaggedAccuracyDetail::makeExpression(
                           kind, batchSize, maxTotalValues, numClasses, categoricalLabelFormat, runtimeState),
                       "predictions",
                       "labels",
                       "metric",
                       kind == RaggedAccuracyDetail::Kind::BINARY ? "Accuracy" : "CategoricalAccuracy",
                       Thor::MetricAggregation::RATIO),
          kind(kind),
          batchSize(batchSize),
          maxTotalValues(maxTotalValues),
          numClasses(numClasses),
          categoricalLabelFormat(categoricalLabelFormat),
          runtimeState(std::move(runtimeState)) {
        if (batchSize == 0 || batchSize > std::numeric_limits<uint32_t>::max())
            throw std::invalid_argument("Ragged accuracy logical batch size must fit uint32 and be non-zero.");
        if (maxTotalValues == 0)
            throw std::invalid_argument("Ragged accuracy max_total_values must be non-zero.");
        if (kind == RaggedAccuracyDetail::Kind::CATEGORICAL && numClasses < 2)
            throw std::invalid_argument("Ragged CategoricalAccuracy requires at least two classes.");
        this->runtimeState->validRowCount = batchSize;
    }

    std::optional<Tensor> connectPredictions(Layer* previousLayer,
                                             std::optional<Tensor> input,
                                             Stream inputStream,
                                             bool backPropagateError) {
        (void)backPropagateError;
        if (!input.has_value())
            throw std::invalid_argument("Ragged accuracy requires packed predictions.");
        if (featureInput.has_value())
            throw std::logic_error("Ragged accuracy predictions are already connected.");
        validatePredictions(input.value());
        if (labelsInput.has_value() && labelsInput->getPlacement() != input->getPlacement())
            throw std::invalid_argument("Ragged accuracy predictions and labels must share placement.");
        if (offsetsInput.has_value() && offsetsInput->getPlacement() != input->getPlacement())
            throw std::invalid_argument("Ragged accuracy predictions and offsets must share placement.");
        Layer::connectToPreviousLayer(previousLayer, input, inputStream, false);
        return std::nullopt;
    }

    std::optional<Tensor> connectLabels(Layer* previousLayer,
                                        std::optional<Tensor> input,
                                        Stream inputStream) {
        (void)previousLayer;
        if (!input.has_value())
            throw std::invalid_argument("Ragged accuracy requires packed labels.");
        if (labelsInput.has_value())
            throw std::logic_error("Ragged accuracy labels are already connected.");
        validateLabels(input.value());
        if (featureInput.has_value() && featureInput->getPlacement() != input->getPlacement())
            throw std::invalid_argument("Ragged accuracy predictions and labels must share placement.");
        if (offsetsInput.has_value() && offsetsInput->getPlacement() != input->getPlacement())
            throw std::invalid_argument("Ragged accuracy labels and offsets must share placement.");
        labelsInput = input;
        labelsStream = inputStream;
        return std::nullopt;
    }

    std::optional<Tensor> connectOffsets(Layer* previousLayer,
                                         std::optional<Tensor> input,
                                         Stream inputStream) {
        (void)previousLayer;
        if (!input.has_value())
            throw std::invalid_argument("Ragged accuracy requires structural offsets.");
        if (offsetsInput.has_value())
            throw std::logic_error("Ragged accuracy offsets are already connected.");
        const RowPartitionDescriptor partition(batchSize, maxTotalValues, input->getDataType());
        if (input->getDescriptor() != partition.getOffsetsDescriptor())
            throw std::invalid_argument("Ragged accuracy offsets must have canonical shape [batch_size + 1].");
        if (featureInput.has_value() && featureInput->getPlacement() != input->getPlacement())
            throw std::invalid_argument("Ragged accuracy predictions and offsets must share placement.");
        if (labelsInput.has_value() && labelsInput->getPlacement() != input->getPlacement())
            throw std::invalid_argument("Ragged accuracy labels and offsets must share placement.");
        offsetsInput = input;
        offsetsStream = inputStream;
        return std::nullopt;
    }

    void validatePredictions(const Tensor& tensor) const {
        if (!RaggedAccuracyDetail::isPredictionDTypeSupported(tensor.getDataType()))
            throw std::invalid_argument("Ragged accuracy predictions must be FP16 or FP32.");
        const std::vector<uint64_t> dims = tensor.getDimensions();
        const std::vector<uint64_t> expected = kind == RaggedAccuracyDetail::Kind::BINARY
                                                   ? std::vector<uint64_t>{maxTotalValues, 1}
                                                   : std::vector<uint64_t>{maxTotalValues, numClasses};
        if (dims != expected)
            throw std::invalid_argument("Ragged accuracy packed prediction dimensions do not match the metric contract.");
    }

    void validateLabels(const Tensor& tensor) const {
        const std::vector<uint64_t> dims = tensor.getDimensions();
        if (kind == RaggedAccuracyDetail::Kind::BINARY) {
            if (dims != std::vector<uint64_t>{maxTotalValues, 1})
                throw std::invalid_argument("Ragged BinaryAccuracy labels must contain one scalar per packed token.");
            if (!RaggedAccuracyDetail::isBinaryOrPerClassLabelDTypeSupported(tensor.getDataType()))
                throw std::invalid_argument("Ragged BinaryAccuracy labels use an unsupported dtype.");
            return;
        }
        if (categoricalLabelFormat == RaggedCategoricalLabelFormat::CLASS_INDEX) {
            if (dims != std::vector<uint64_t>{maxTotalValues, 1})
                throw std::invalid_argument("Ragged CategoricalAccuracy class-index labels must contain one scalar per packed token.");
            if (!RaggedAccuracyDetail::isClassIndexLabelDTypeSupported(tensor.getDataType()))
                throw std::invalid_argument("Ragged CategoricalAccuracy class-index labels must use an integer dtype.");
        } else {
            if (dims != std::vector<uint64_t>{maxTotalValues, numClasses})
                throw std::invalid_argument("Ragged CategoricalAccuracy per-class labels must match prediction class width.");
            if (!RaggedAccuracyDetail::isBinaryOrPerClassLabelDTypeSupported(tensor.getDataType()))
                throw std::invalid_argument("Ragged CategoricalAccuracy per-class labels use an unsupported dtype.");
        }
    }

    void validateInputs() const {
        if (!featureInput.has_value() || !labelsInput.has_value() || !offsetsInput.has_value())
            throw std::logic_error("Ragged accuracy requires predictions, labels, and structural offsets.");
        validatePredictions(featureInput.value());
        validateLabels(labelsInput.value());
        const RowPartitionDescriptor partition(batchSize, maxTotalValues, offsetsInput->getDataType());
        if (offsetsInput->getDescriptor() != partition.getOffsetsDescriptor())
            throw std::invalid_argument("Ragged accuracy offsets must have canonical shape [batch_size + 1].");
        if (featureInput->getPlacement() != labelsInput->getPlacement() ||
            featureInput->getPlacement() != offsetsInput->getPlacement())
            throw std::invalid_argument("Ragged accuracy inputs must share placement.");
    }

    RaggedAccuracyDetail::Kind kind;
    uint64_t batchSize;
    uint64_t maxTotalValues;
    uint64_t numClasses;
    RaggedCategoricalLabelFormat categoricalLabelFormat;
    std::shared_ptr<RaggedAccuracyDetail::RuntimeState> runtimeState;
    std::optional<Tensor> offsetsInput;
    Stream offsetsStream;
    Event offsetsReadyEvent;
    Event offsetsReusableEvent;
    bool offsetsReceived = false;
};

}  // namespace ThorImplementation
