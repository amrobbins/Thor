#pragma once

#include "DeepLearning/Implementation/Layers/MultiConnectionLayer.h"
#include "Utilities/TensorOperations/Einsum/Einsum.h"
#include "Utilities/TensorOperations/Einsum/EinsumBackwardPlanner.h"
#include "Utilities/TensorOperations/Einsum/EinsumParser.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/FusedEquation.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ThorImplementation {

/**
 * Physical implementation layer for a user-visible per-example einsum.
 *
 * The equation describes feature axes only.  Thor's leading physical batch
 * axis is prepended as a synthetic resolved label and is therefore always
 * preserved; examples can never be contracted with one another.  Forward
 * execution is delegated entirely to the existing stamped Einsum/Expression
 * backend.
 *
 * Forward and backward contractions are both lowered through the production
 * Einsum/Expression backend. Backward metadata comes from EinsumBackwardPlanner;
 * shape restoration uses centralized Expression reductions/broadcasting and
 * STRIDED_VIEW_BACKWARD for repeated-label diagonal scatter.
 */
class EinsumLayer : public MultiConnectionLayer {
   public:
    explicit EinsumLayer(std::string equation) : equation(std::move(equation)) {
        if (this->equation.empty()) {
            throw std::invalid_argument("EinsumLayer equation must not be empty.");
        }

        const EinsumEquation parsed = EinsumParser::parse(this->equation);
        if (parsed.inputs.empty()) {
            throw std::invalid_argument("EinsumLayer requires at least one input operand.");
        }
        if (parsed.inputs.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
            throw std::invalid_argument("EinsumLayer operand count exceeds the supported uint32_t range.");
        }
        expectedNumInputs = static_cast<uint32_t>(parsed.inputs.size());

        previousLayers.resize(expectedNumInputs);
        featureInputs.resize(expectedNumInputs);
        streams.resize(expectedNumInputs);
        errorOutputs.resize(expectedNumInputs);
    }

    ~EinsumLayer() override = default;

    [[nodiscard]] const std::string& getEquation() const { return equation; }
    [[nodiscard]] uint32_t getExpectedNumInputs() const { return expectedNumInputs; }
    [[nodiscard]] const EinsumLayerBackwardPlan& getBackwardPlan() const {
        if (!backwardPlan.has_value()) {
            throw std::logic_error("EinsumLayer backward metadata is not available until all inputs are connected.");
        }
        return backwardPlan.value();
    }
    [[nodiscard]] std::shared_ptr<StampedEinsum> getStampedForwardExecution() const { return stampedForward; }
    [[nodiscard]] std::shared_ptr<StampedEinsum> getStampedBackwardContraction(uint32_t operand_index) const {
        if (operand_index >= stampedBackward.size() || !stampedBackward[operand_index].has_value()) {
            return nullptr;
        }
        return stampedBackward[operand_index]->contraction;
    }
    [[nodiscard]] bool backwardOperandHasPostprocess(uint32_t operand_index) const {
        return operand_index < stampedBackward.size() && stampedBackward[operand_index].has_value() &&
               stampedBackward[operand_index]->postprocess != nullptr;
    }

    Stream getStream() override {
        THOR_THROW_IF_FALSE(!streams.empty());
        THOR_THROW_IF_FALSE(streams[0].isInitialized());
        return streams[0];
    }

    std::optional<Tensor> createFeatureOutputTensor() override {
        resolveForwardMetadata();
        THOR_THROW_IF_FALSE(backwardPlan.has_value());
        THOR_THROW_IF_FALSE(featureInputs[0].has_value());

        const Tensor& reference = featureInputs[0].value();
        std::vector<uint64_t> outputDimensions;
        outputDimensions.reserve(backwardPlan->feature_equation.output_dimensions.size() + 1);
        outputDimensions.push_back(batchCapacity());
        outputDimensions.insert(outputDimensions.end(),
                                backwardPlan->feature_equation.output_dimensions.begin(),
                                backwardPlan->feature_equation.output_dimensions.end());
        return Tensor(reference.getPlacement(), TensorDescriptor(reference.getDataType(), outputDimensions));
    }

    void compileImpl() override {
        MultiConnectionLayer::compileImpl();
        requireAllInputsConnected();
        THOR_THROW_IF_FALSE(featureOutputs.size() == 1);
        THOR_THROW_IF_FALSE(featureOutputs[0].has_value());
        THOR_THROW_IF_FALSE(nextLayers.size() == 1);
        THOR_THROW_IF_FALSE(!streams.empty());
        THOR_THROW_IF_FALSE(streams[0].isInitialized());

        resolveForwardMetadata();
        const ResolvedEinsumEquation physicalEquation = makePhysicalForwardEquation();
        std::vector<Tensor> inputs = concreteFeatureInputs();
        stampedForward = Einsum::stampResolvedEquation(physicalEquation, inputs, featureOutputs[0].value(), streams[0]);
        THOR_THROW_IF_FALSE(stampedForward != nullptr);
        THOR_THROW_IF_FALSE(stampedForward->getOutputTensor() == featureOutputs[0].value());
        stampBackwardExecutions();
    }

    void cleanup() override {
        stampedForward.reset();
        stampedBackward.clear();
        stillWaitingForFeatureInputTensorIds.clear();
        allFeatureInputTensorIds.clear();
        currentValidExampleCount = 0;
        batchCardinalitySet = false;
        Layer::cleanup();
    }

    void initialize() override {
        MultiConnectionLayer::initialize();
        allFeatureInputTensorIds.clear();
        for (const std::optional<Tensor>& input : featureInputs) {
            THOR_THROW_IF_FALSE(input.has_value());
            allFeatureInputTensorIds.insert(input->getTensorId());
        }
        stillWaitingForFeatureInputTensorIds = allFeatureInputTensorIds;
        currentValidExampleCount = 0;
        batchCardinalitySet = false;
    }

    void forward(std::optional<Tensor> featureInput, bool validationPass, uint32_t validExampleCount = 0) override {
        THOR_THROW_IF_FALSE(running);
        THOR_THROW_IF_FALSE(featureInput.has_value());
        THOR_THROW_IF_FALSE(stampedForward != nullptr);
        THOR_THROW_IF_FALSE(featureOutputs.size() == 1 && featureOutputs[0].has_value());

        const uint32_t capacity = checkedBatchCapacity(featureInput.value());
        THOR_THROW_IF_FALSE(capacity == batchCapacity());
        const uint32_t resolvedValidExampleCount = validExampleCount == 0 ? capacity : validExampleCount;
        THOR_THROW_IF_FALSE(resolvedValidExampleCount >= 1);
        THOR_THROW_IF_FALSE(resolvedValidExampleCount <= capacity);

        if (batchCardinalitySet) {
            THOR_THROW_IF_FALSE(currentValidExampleCount == resolvedValidExampleCount);
        } else {
            currentValidExampleCount = resolvedValidExampleCount;
            batchCardinalitySet = true;
        }

        const uint64_t tensorId = featureInput->getTensorId();
        auto waiting = stillWaitingForFeatureInputTensorIds.find(tensorId);
        THOR_THROW_IF_FALSE(waiting != stillWaitingForFeatureInputTensorIds.end());
        stillWaitingForFeatureInputTensorIds.erase(waiting);

        if (!stillWaitingForFeatureInputTensorIds.empty()) {
            return;
        }
        stillWaitingForFeatureInputTensorIds = allFeatureInputTensorIds;

        // Each producer reaches this method only after its own stream has
        // enqueued the tensor-producing work.  Join every distinct producer
        // stream onto the canonical einsum stream before launching the stamped
        // Expression DAG.
        std::set<uint64_t> joinedStreamIds;
        joinedStreamIds.insert(streams[0].getId());
        for (size_t i = 1; i < streams.size(); ++i) {
            if (!streams[i].isInitialized()) {
                continue;
            }
            if (!joinedStreamIds.insert(streams[i].getId()).second) {
                continue;
            }
            streams[0].waitEvent(streams[i].putEvent());
        }

        Stream runStream = streams[0];
        stampedForward->runOn(runStream);

        if (nextLayers[0].has_value()) {
            nextLayers[0].value()->forward(featureOutputs[0], validationPass, currentValidExampleCount);
        }

        currentValidExampleCount = 0;
        batchCardinalitySet = false;
    }

    void backward(std::optional<Tensor> errorInput, uint32_t batchSize = 0) override {
        THOR_THROW_IF_FALSE(running);
        if (!errorInput.has_value()) {
            return;
        }
        THOR_THROW_IF_FALSE(errorInputs.size() == 1);
        THOR_THROW_IF_FALSE(errorInputs[0].has_value());
        THOR_THROW_IF_FALSE(errorInput.value() == errorInputs[0].value());
        THOR_THROW_IF_FALSE(stampedBackward.size() == expectedNumInputs);

        const uint32_t capacity = checkedBatchCapacity(errorInput.value());
        THOR_THROW_IF_FALSE(capacity == batchCapacity());
        const uint32_t resolvedBatchSize = batchSize == 0 ? capacity : batchSize;
        THOR_THROW_IF_FALSE(resolvedBatchSize >= 1);
        THOR_THROW_IF_FALSE(resolvedBatchSize <= capacity);

        const Event errorReady = streams[0].putEvent();
        std::set<uint64_t> streamsWaitingOnError;
        streamsWaitingOnError.insert(streams[0].getId());

        for (uint32_t operandIndex = 0; operandIndex < expectedNumInputs; ++operandIndex) {
            if (!errorOutputs[operandIndex].has_value()) {
                continue;
            }
            THOR_THROW_IF_FALSE(stampedBackward[operandIndex].has_value());
            StampedOperandBackwardExecution& execution = stampedBackward[operandIndex].value();
            THOR_THROW_IF_FALSE(execution.contraction != nullptr);

            Stream& runStream = streams[operandIndex];
            if (streamsWaitingOnError.insert(runStream.getId()).second) {
                runStream.waitEvent(errorReady);
            }

            execution.contraction->runOn(runStream);
            if (execution.postprocess != nullptr) {
                execution.postprocess->runOn(runStream);
            }

            if (previousLayers[operandIndex].has_value()) {
                previousLayers[operandIndex].value()->backward(errorOutputs[operandIndex], resolvedBatchSize);
            }
        }
    }

    void connectToNextLayer(Layer* nextLayer, int driverConnectionType = 0, int loaderConnectionType = 0) override {
        (void)driverConnectionType;
        THOR_THROW_IF_FALSE(!compiled);
        THOR_THROW_IF_FALSE(nextLayer != nullptr);
        THOR_THROW_IF_FALSE(nextLayers.empty());
        requireAllInputsConnected();

        nextLayers.push_back(nextLayer);
        if (nextLayer->hasFeatureInput()) {
            featureOutputs.emplace_back(createFeatureOutputTensor());
        } else {
            featureOutputs.emplace_back(std::nullopt);
        }

        // isBackPropStub() is true whenever none of the input connections asked
        // for gradients. Unused operand-gradient paths are pruned exactly like
        // the other multi-input implementation layers.
        errorInputs.emplace_back(nextLayer->connectToPreviousLayer(
            this,
            featureOutputs.back(),
            streams[0],
            shouldConnectToBackPropErrorIn() && !isBackPropStub(),
            loaderConnectionType));

        if (!errorInputs.back().has_value()) {
            for (size_t i = 0; i < errorOutputs.size(); ++i) {
                if (!errorOutputs[i].has_value() || !previousLayers[i].has_value()) {
                    continue;
                }
                previousLayers[i].value()->replaceErrorInput(errorOutputs[i], std::nullopt);
                errorOutputs[i].reset();
            }
        }

        ensureNoDeviceCrossing();
    }

    std::optional<Tensor> connectToPreviousLayer(Layer* previousLayer,
                                                  std::optional<Tensor> featureInput,
                                                  Stream stream,
                                                  bool backPropagateError,
                                                  int connectionType = 0) override {
        THOR_THROW_IF_FALSE(!compiled);
        THOR_THROW_IF_FALSE(previousLayer != nullptr);
        THOR_THROW_IF_FALSE(featureInput.has_value());
        if (connectionType < 0 || static_cast<uint32_t>(connectionType) >= expectedNumInputs) {
            throw std::logic_error("EinsumLayer input connection type " + std::to_string(connectionType) +
                                   " is outside the declared operand range [0," +
                                   std::to_string(expectedNumInputs - 1) + "].");
        }

        const uint32_t inputIndex = static_cast<uint32_t>(connectionType);
        if (featureInputs[inputIndex].has_value() || previousLayers[inputIndex].has_value()) {
            throw std::logic_error("EinsumLayer operand[" + std::to_string(inputIndex) +
                                   "] was connected more than once.");
        }

        const Tensor& incoming = featureInput.value();
        validateIncomingInput(incoming, inputIndex);
        streams[inputIndex] = stream;
        previousLayers[inputIndex] = previousLayer;
        featureInputs[inputIndex] = incoming;
        if (backPropagateError && !isInferenceOnly()) {
            errorOutputs[inputIndex] = incoming.clone();
        } else {
            errorOutputs[inputIndex] = std::nullopt;
        }

        ensureNoDeviceCrossing(incoming.getPlacement());
        return errorOutputs[inputIndex];
    }

   protected:
    void infer(std::optional<Tensor> inputTensor,
               std::optional<Tensor> outputTensor,
               Stream stream,
               unsigned int connectionNumber) override {
        (void)inputTensor;
        (void)outputTensor;
        (void)stream;
        (void)connectionNumber;
        THOR_UNREACHABLE();
    }

    void backProp(std::optional<Tensor> dataIn,
                  std::optional<Tensor> errorIn,
                  std::optional<Tensor> errorOut,
                  Stream stream,
                  unsigned int connectionNumber) override {
        (void)dataIn;
        (void)errorIn;
        (void)errorOut;
        (void)stream;
        (void)connectionNumber;
        THOR_UNREACHABLE();
    }

   private:
    struct StampedOperandBackwardExecution {
        std::shared_ptr<StampedEinsum> contraction;
        std::shared_ptr<StampedExecutionPlan> postprocess;
    };

    static uint64_t checkedMultiply(uint64_t lhs, uint64_t rhs, const char* what) {
        if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
            throw std::overflow_error(std::string("EinsumLayer ") + what + " overflows uint64_t.");
        }
        return lhs * rhs;
    }

    static uint64_t mergeBroadcastDimension(uint64_t current, uint64_t incoming) {
        if (current == 0 || current == incoming) {
            return incoming;
        }
        if (current == 1) {
            return incoming;
        }
        if (incoming == 1) {
            return current;
        }
        throw std::logic_error("EinsumLayer backward contraction inputs are not broadcast-compatible.");
    }

    static bool needsPostprocess(const EinsumOperandBackwardPlan& gradient) {
        return !gradient.broadcast_reductions.empty() || !gradient.existing_axis_expansions.empty() ||
               !gradient.missing_axis_expansions.empty() || !gradient.diagonal_scatters.empty();
    }

    std::vector<Tensor> backwardContractionInputs(const EinsumOperandBackwardPlan& gradient) const {
        THOR_THROW_IF_FALSE(errorInputs.size() == 1 && errorInputs[0].has_value());
        std::vector<Tensor> inputs;
        inputs.reserve(1 + gradient.contraction.other_operand_indices.size());
        inputs.push_back(errorInputs[0].value());
        for (uint32_t operandIndex : gradient.contraction.other_operand_indices) {
            THOR_THROW_IF_FALSE(operandIndex < featureInputs.size());
            THOR_THROW_IF_FALSE(featureInputs[operandIndex].has_value());
            inputs.push_back(featureInputs[operandIndex].value());
        }
        return inputs;
    }

    ResolvedEinsumEquation makePhysicalBackwardContractionEquation(
        const EinsumOperandBackwardPlan& gradient,
        const std::vector<Tensor>& contractionInputs) const {
        THOR_THROW_IF_FALSE(backwardPlan.has_value());
        THOR_THROW_IF_FALSE(contractionInputs.size() == gradient.contraction.physical_input_axis_labels.size());

        ResolvedEinsumEquation resolved;
        resolved.inputs.reserve(contractionInputs.size());
        if (backwardPlan->feature_equation.label_dimensions.size() >=
            static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
            throw std::logic_error("EinsumLayer cannot allocate an internal backward batch label id.");
        }
        const int32_t batchLabel = static_cast<int32_t>(backwardPlan->feature_equation.label_dimensions.size());
        resolved.label_dimensions.resize(static_cast<size_t>(batchLabel) + 1, 0);

        auto physicalLabel = [batchLabel](int32_t label) -> int32_t {
            return EinsumLayerBatchContract::isImplicitBatchLabel(label) ? batchLabel : label;
        };

        for (size_t inputIndex = 0; inputIndex < contractionInputs.size(); ++inputIndex) {
            const std::vector<uint64_t>& dimensions = contractionInputs[inputIndex].getDimensions();
            const std::vector<int32_t>& plannedLabels = gradient.contraction.physical_input_axis_labels[inputIndex];
            THOR_THROW_IF_FALSE(dimensions.size() == plannedLabels.size());

            ResolvedEinsumOperand operand;
            operand.axis_labels.reserve(plannedLabels.size());
            for (size_t axis = 0; axis < plannedLabels.size(); ++axis) {
                const int32_t label = physicalLabel(plannedLabels[axis]);
                THOR_THROW_IF_FALSE(label >= 0 && static_cast<size_t>(label) < resolved.label_dimensions.size());
                operand.axis_labels.push_back(label);
                resolved.label_dimensions[label] =
                    mergeBroadcastDimension(resolved.label_dimensions[label], dimensions[axis]);
            }
            resolved.inputs.push_back(std::move(operand));
        }

        resolved.output_labels.reserve(gradient.contraction.physical_output_axis_labels.size());
        for (int32_t label : gradient.contraction.physical_output_axis_labels) {
            resolved.output_labels.push_back(physicalLabel(label));
        }
        resolved.output_dimensions = EinsumLayerBatchContract::prependBatchDimension(
            batchCapacity(), gradient.contraction.output_feature_dimensions);
        THOR_THROW_IF_FALSE(resolved.output_labels.size() == resolved.output_dimensions.size());
        for (size_t axis = 0; axis < resolved.output_labels.size(); ++axis) {
            const int32_t label = resolved.output_labels[axis];
            THOR_THROW_IF_FALSE(label >= 0 && static_cast<size_t>(label) < resolved.label_dimensions.size());
            THOR_THROW_IF_FALSE(resolved.label_dimensions[label] == resolved.output_dimensions[axis]);
        }

        std::unordered_set<int32_t> outputLabels(resolved.output_labels.begin(), resolved.output_labels.end());
        std::unordered_set<int32_t> recordedReductions;
        for (const ResolvedEinsumOperand& operand : resolved.inputs) {
            for (int32_t label : operand.axis_labels) {
                if (outputLabels.count(label) == 0 && recordedReductions.insert(label).second) {
                    resolved.reduction_labels.push_back(label);
                }
            }
        }
        resolved.explicit_output = true;
        return resolved;
    }

    std::vector<uint64_t> diagonalViewStrides(const EinsumOperandBackwardPlan& gradient) const {
        THOR_THROW_IF_FALSE(backwardPlan.has_value());
        THOR_THROW_IF_FALSE(gradient.operand_index < backwardPlan->feature_equation.inputs.size());
        const std::vector<int32_t>& targetAxisLabels =
            backwardPlan->feature_equation.inputs[gradient.operand_index].axis_labels;
        THOR_THROW_IF_FALSE(targetAxisLabels.size() == gradient.final_feature_dimensions.size());

        const std::vector<uint64_t> denseDimensions = EinsumLayerBatchContract::prependBatchDimension(
            batchCapacity(), gradient.final_feature_dimensions);
        std::vector<uint64_t> denseStrides(denseDimensions.size(), 1);
        for (int64_t axis = static_cast<int64_t>(denseDimensions.size()) - 2; axis >= 0; --axis) {
            denseStrides[axis] = checkedMultiply(denseStrides[axis + 1], denseDimensions[axis + 1], "dense gradient stride");
        }

        std::vector<uint64_t> viewStrides;
        viewStrides.reserve(gradient.target_unique_feature_labels.size() + 1);
        viewStrides.push_back(denseStrides[0]);
        for (int32_t label : gradient.target_unique_feature_labels) {
            uint64_t stride = 0;
            for (size_t featureAxis = 0; featureAxis < targetAxisLabels.size(); ++featureAxis) {
                if (targetAxisLabels[featureAxis] != label) {
                    continue;
                }
                const uint64_t addend = denseStrides[featureAxis + 1];
                if (stride > std::numeric_limits<uint64_t>::max() - addend) {
                    throw std::overflow_error("EinsumLayer diagonal view stride overflows uint64_t.");
                }
                stride += addend;
            }
            THOR_THROW_IF_FALSE(stride != 0);
            viewStrides.push_back(stride);
        }
        return viewStrides;
    }

    std::shared_ptr<StampedExecutionPlan> stampBackwardPostprocess(const EinsumOperandBackwardPlan& gradient,
                                                                   const Tensor& rawGradient,
                                                                   const Tensor& finalGradient,
                                                                   Stream& stream) const {
        constexpr const char* kRawGradientName = "raw_gradient";
        constexpr const char* kFinalGradientName = "operand_gradient";

        Expression current = Expression::input(kRawGradientName, rawGradient.getDataType(), rawGradient.getDataType());

        if (!gradient.broadcast_reductions.empty()) {
            std::vector<uint64_t> reductionAxes;
            reductionAxes.reserve(gradient.broadcast_reductions.size());
            for (const EinsumBackwardBroadcastReductionPlan& reduction : gradient.broadcast_reductions) {
                reductionAxes.push_back(static_cast<uint64_t>(reduction.contraction_output_feature_axis) + 1);
            }
            current = current.reduce_sum(reductionAxes, {}, DataType::FP32);
        }

        if (!gradient.missing_axis_expansions.empty()) {
            std::vector<uint64_t> unsqueezeAxes;
            unsqueezeAxes.reserve(gradient.missing_axis_expansions.size());
            for (const EinsumBackwardMissingAxisExpansionPlan& expansion : gradient.missing_axis_expansions) {
                unsqueezeAxes.push_back(static_cast<uint64_t>(expansion.target_unique_feature_axis) + 1);
            }
            current = current.unsqueeze(unsqueezeAxes);
        }

        if (!gradient.existing_axis_expansions.empty() || !gradient.missing_axis_expansions.empty()) {
            const std::vector<uint64_t> uniquePhysicalDimensions = EinsumLayerBatchContract::prependBatchDimension(
                batchCapacity(), gradient.target_unique_feature_dimensions);
            current = current + Expression::fill(0.0, uniquePhysicalDimensions, rawGradient.getDataType());
        }

        if (!gradient.diagonal_scatters.empty()) {
            const std::vector<uint64_t> densePhysicalDimensions = EinsumLayerBatchContract::prependBatchDimension(
                batchCapacity(), gradient.final_feature_dimensions);
            const std::vector<uint64_t> uniquePhysicalDimensions = EinsumLayerBatchContract::prependBatchDimension(
                batchCapacity(), gradient.target_unique_feature_dimensions);
            current = current.stridedViewBackward(
                densePhysicalDimensions, uniquePhysicalDimensions, diagonalViewStrides(gradient), 0);
        }
        current = current.withOutputDType(finalGradient.getDataType());

        FusedEquation equation = FusedEquation::compile(
            Expression::outputs({{kFinalGradientName, current}}).physicalOutputs(), finalGradient.getPlacement().getDeviceNum());
        std::unordered_map<std::string, Tensor> inputs{{kRawGradientName, rawGradient}};
        std::unordered_map<std::string, Tensor> outputs{{kFinalGradientName, finalGradient}};
        StampedExecutionPlan stamped = equation.stamp(inputs, stream, {}, outputs);
        THOR_THROW_IF_FALSE(stamped.output(kFinalGradientName) == finalGradient);
        return std::make_shared<StampedExecutionPlan>(std::move(stamped));
    }

    void stampBackwardExecutions() {
        stampedBackward.clear();
        stampedBackward.resize(expectedNumInputs);
        if (errorInputs.empty() || !errorInputs[0].has_value()) {
            return;
        }
        THOR_THROW_IF_FALSE(backwardPlan.has_value());
        THOR_THROW_IF_FALSE(backwardPlan->operand_gradients.size() == expectedNumInputs);

        for (uint32_t operandIndex = 0; operandIndex < expectedNumInputs; ++operandIndex) {
            if (!errorOutputs[operandIndex].has_value()) {
                continue;
            }
            const EinsumOperandBackwardPlan& gradient = backwardPlan->operand_gradients[operandIndex];
            THOR_THROW_IF_FALSE(gradient.operand_index == operandIndex);
            THOR_THROW_IF_FALSE(errorOutputs[operandIndex]->getDimensions() == featureInputs[operandIndex]->getDimensions());

            std::vector<Tensor> contractionInputs = backwardContractionInputs(gradient);
            const ResolvedEinsumEquation contractionEquation =
                makePhysicalBackwardContractionEquation(gradient, contractionInputs);

            const bool postprocessRequired = needsPostprocess(gradient);
            Tensor contractionOutput = postprocessRequired
                                           ? Tensor(errorOutputs[operandIndex]->getPlacement(),
                                                    TensorDescriptor(errorOutputs[operandIndex]->getDataType(),
                                                                     contractionEquation.output_dimensions))
                                           : errorOutputs[operandIndex].value();

            StampedOperandBackwardExecution execution;
            execution.contraction = Einsum::stampResolvedEquation(
                contractionEquation, contractionInputs, contractionOutput, streams[operandIndex]);
            THOR_THROW_IF_FALSE(execution.contraction != nullptr);
            THOR_THROW_IF_FALSE(execution.contraction->getOutputTensor() == contractionOutput);
            if (postprocessRequired) {
                execution.postprocess = stampBackwardPostprocess(
                    gradient, contractionOutput, errorOutputs[operandIndex].value(), streams[operandIndex]);
            }
            stampedBackward[operandIndex] = std::move(execution);
        }
    }

    static uint32_t checkedBatchCapacity(const Tensor& tensor) {
        const std::vector<uint64_t>& dimensions = tensor.getDescriptor().getDimensions();
        if (dimensions.empty()) {
            throw std::logic_error("EinsumLayer physical tensors must include Thor's leading batch dimension.");
        }
        if (dimensions.front() == 0 || dimensions.front() > std::numeric_limits<uint32_t>::max()) {
            throw std::logic_error("EinsumLayer batch capacity is outside the supported uint32_t range.");
        }
        return static_cast<uint32_t>(dimensions.front());
    }

    uint32_t batchCapacity() const {
        THOR_THROW_IF_FALSE(featureInputs.size() == expectedNumInputs);
        THOR_THROW_IF_FALSE(featureInputs[0].has_value());
        return checkedBatchCapacity(featureInputs[0].value());
    }

    void requireAllInputsConnected() const {
        if (featureInputs.size() != expectedNumInputs) {
            throw std::logic_error("EinsumLayer internal operand table has an unexpected size.");
        }
        for (uint32_t i = 0; i < expectedNumInputs; ++i) {
            if (!featureInputs[i].has_value() || !previousLayers[i].has_value()) {
                throw std::logic_error("EinsumLayer operand[" + std::to_string(i) + "] is not connected.");
            }
        }
    }

    void validateIncomingInput(const Tensor& incoming, uint32_t inputIndex) const {
        if (!incoming.isInitialized()) {
            throw std::logic_error("EinsumLayer operand[" + std::to_string(inputIndex) + "] is uninitialized.");
        }
        if (incoming.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU) {
            throw std::logic_error("EinsumLayer currently requires GPU feature tensors.");
        }
        if (!incoming.isDenseContiguous()) {
            throw std::logic_error("EinsumLayer currently requires dense contiguous feature tensors.");
        }
        (void)checkedBatchCapacity(incoming);

        for (uint32_t i = 0; i < expectedNumInputs; ++i) {
            if (!featureInputs[i].has_value()) {
                continue;
            }
            const Tensor& prior = featureInputs[i].value();
            if (incoming.getPlacement() != prior.getPlacement()) {
                throw std::logic_error("EinsumLayer operands must use the same GPU placement.");
            }
            if (incoming.getDataType() != prior.getDataType()) {
                throw std::logic_error("EinsumLayer operands must use the same storage data type.");
            }
            if (checkedBatchCapacity(incoming) != checkedBatchCapacity(prior)) {
                throw std::logic_error("EinsumLayer operands must have the same physical batch capacity.");
            }
        }
    }

    std::vector<std::vector<uint64_t>> inputFeatureDimensions() const {
        requireAllInputsConnected();
        std::vector<std::vector<uint64_t>> result;
        result.reserve(featureInputs.size());
        for (const std::optional<Tensor>& input : featureInputs) {
            const std::vector<uint64_t>& physicalDimensions = input->getDescriptor().getDimensions();
            result.emplace_back(physicalDimensions.begin() + 1, physicalDimensions.end());
        }
        return result;
    }

    void resolveForwardMetadata() {
        requireAllInputsConnected();
        const std::vector<std::vector<uint64_t>> featureDimensions = inputFeatureDimensions();
        backwardPlan = EinsumBackwardPlanner::parseAndPlan(equation, featureDimensions);
    }

    ResolvedEinsumEquation makePhysicalForwardEquation() const {
        THOR_THROW_IF_FALSE(backwardPlan.has_value());
        ResolvedEinsumEquation physical = backwardPlan->feature_equation;

        if (physical.label_dimensions.size() >= static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
            throw std::logic_error("EinsumLayer cannot allocate an internal batch label id.");
        }
        const int32_t batchLabel = static_cast<int32_t>(physical.label_dimensions.size());
        physical.label_dimensions.push_back(batchCapacity());

        for (ResolvedEinsumOperand& operand : physical.inputs) {
            operand.axis_labels.insert(operand.axis_labels.begin(), batchLabel);
        }
        physical.output_labels.insert(physical.output_labels.begin(), batchLabel);
        physical.output_dimensions.insert(physical.output_dimensions.begin(), batchCapacity());
        // The batch label is intentionally absent from reduction_labels.
        return physical;
    }

    std::vector<Tensor> concreteFeatureInputs() const {
        requireAllInputsConnected();
        std::vector<Tensor> result;
        result.reserve(featureInputs.size());
        for (const std::optional<Tensor>& input : featureInputs) {
            result.push_back(input.value());
        }
        return result;
    }

    std::string equation;
    uint32_t expectedNumInputs = 0;
    std::optional<EinsumLayerBackwardPlan> backwardPlan;
    std::shared_ptr<StampedEinsum> stampedForward;
    std::vector<std::optional<StampedOperandBackwardExecution>> stampedBackward;

    std::set<uint64_t> allFeatureInputTensorIds;
    std::set<uint64_t> stillWaitingForFeatureInputTensorIds;
    uint32_t currentValidExampleCount = 0;
    bool batchCardinalitySet = false;
};

}  // namespace ThorImplementation
