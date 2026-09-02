#pragma once

#include "DeepLearning/Implementation/Layers/Loss.h"
#include "Utilities/Expression/DynamicExpression.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <set>
#include <unordered_map>
#include <vector>

namespace ThorImplementation {

// Internal valuewise loss execution primitive for rank-1 ragged sequences.
//
// Predictions and labels are packed values with physical shape
//   [max_total_values, ...trailing value dimensions...]
// and offsets is the canonical structural tensor [batch_size + 1].  The
// offsets tensor, not packed capacity, controls the device-side execution
// extent: only [0, offsets[batch_size]) is read or written.  Inactive packed
// capacity is deliberately left untouched.
//
// The logical batch cardinality is carried independently from
// max_total_values.  This is important for Thor's optimizer semantics: valid
// examples are logical rows, never packed tokens.  For a partial batch, the
// canonical partition must represent invalid tail rows [valid_examples, B) as
// empty, matching Thor's ragged batch materialization contract.  RaggedLossShaper
// owns reduction/shaping semantics; this layer intentionally produces only the raw packed
// valuewise loss and packed gradients for configured differentiable value inputs. The
// secondary inputs cover distribution losses with additional learned token-varying
// parameters. Every configured secondary is differentiable and uses the same packed
// geometry as predictions.
class RaggedCustomLoss : public Loss {
   public:
    static constexpr const char* RAGGED_OFFSETS_INPUT_NAME = "__thor_ragged_offsets";

    enum class InputConnection : int {
        PREDICTIONS = 0,
        LABELS = 1,
        OFFSETS = 2,
        EXAMPLE_WEIGHTS = 3,
        SECONDARY_INPUT = 4,
        SECONDARY_INPUT_BASE = SECONDARY_INPUT,
    };

    RaggedCustomLoss(DynamicExpression lossExpression,
                     DynamicExpression gradientExpression,
                     uint64_t batchSize,
                     uint64_t maxTotalValues,
                     std::string predictionsName = "predictions",
                     std::string labelsName = "labels",
                     std::string lossName = "loss",
                     std::string gradientName = "predictions_grad",
                     DataType lossDataType = DataType::FP32,
                     std::optional<float> lossWeight = std::nullopt,
                     std::optional<std::string> exampleWeightsName = std::nullopt,
                     std::vector<std::string> secondaryInputNames = {},
                     std::vector<std::string> secondaryGradientNames = {});

    // Source-compatible one-secondary constructor retained for existing internal consumers.
    RaggedCustomLoss(DynamicExpression lossExpression,
                     DynamicExpression gradientExpression,
                     uint64_t batchSize,
                     uint64_t maxTotalValues,
                     std::string predictionsName,
                     std::string labelsName,
                     std::string lossName,
                     std::string gradientName,
                     DataType lossDataType,
                     std::optional<float> lossWeight,
                     std::optional<std::string> exampleWeightsName,
                     std::optional<std::string> secondaryInputName,
                     std::optional<std::string> secondaryGradientName);

    ~RaggedCustomLoss() override = default;

    std::optional<Tensor> createFeatureOutputTensor() override;
    std::optional<Tensor> connectToPreviousLayer(Layer* previousLayer,
                                                 std::optional<Tensor> featureInput,
                                                 Stream stream,
                                                 bool backPropagateError,
                                                 int connectionType = 0) override;

    void compileImpl() override;
    void initialize() override;
    void cleanup() override;
    void replaceErrorInput(std::optional<Tensor> oldErrorInput, std::optional<Tensor> newErrorInput) override;
    void pruneTrainingBackpropPathIfInactive() override;
    void forward(std::optional<Tensor> inputTensor, bool validationPass, uint32_t validExampleCount = 0) override;
    void backward(std::optional<Tensor> errorInput, uint32_t validExampleCount = 0) override;

    std::vector<Stream> getProcessingStreams() override;
    std::vector<Event> getSynchronizeEvents() override;
    void ensureNoDeviceCrossing() override;

    std::string getType() override { return "RaggedCustomLoss"; }
    bool supportsPartialBatches() const override { return true; }

    [[nodiscard]] uint64_t getLogicalBatchSize() const { return batchSize; }
    [[nodiscard]] uint64_t getMaxTotalValues() const { return maxTotalValues; }
    [[nodiscard]] std::optional<Tensor> getOffsetsInput() const { return offsetsInput; }
    [[nodiscard]] std::optional<Tensor> getExampleWeightsInput() const { return exampleWeightsInput; }
    [[nodiscard]] std::optional<Tensor> getSecondaryInput() const {
        return secondaryInputs.empty() ? std::nullopt : secondaryInputs.front().input;
    }
    [[nodiscard]] std::optional<Tensor> getSecondaryErrorOutput() const {
        return secondaryInputs.empty() ? std::nullopt : secondaryInputs.front().errorOutput;
    }
    [[nodiscard]] size_t getNumSecondaryInputs() const { return secondaryInputs.size(); }
    [[nodiscard]] std::optional<Tensor> getSecondaryInput(size_t index) const {
        if (index >= secondaryInputs.size()) return std::nullopt;
        return secondaryInputs[index].input;
    }
    [[nodiscard]] std::optional<Tensor> getSecondaryErrorOutput(size_t index) const {
        if (index >= secondaryInputs.size()) return std::nullopt;
        return secondaryInputs[index].errorOutput;
    }

   protected:
    void advanceDataIfReady(bool validationPass) override;

   private:
    using TensorMap = std::unordered_map<std::string, Tensor>;

    DynamicExpression withRaggedExtent(const DynamicExpression& expression,
                                       const std::unordered_map<std::string, DataType>& outputDataTypes,
                                       const char* what) const;
    void validateExpressionContract(const DynamicExpression& expression,
                                    const std::set<std::string>& expectedOutputNames,
                                    const char* what) const;
    TensorMap buildInputs() const;
    uint64_t elementsPerValue() const;
    uint32_t resolveValidExampleCount(uint32_t validExampleCount) const;
    void recordLogicalBatchCardinality(uint32_t validExampleCount);
    void synchronizeComputeStreamForInputs();
    void markAuxiliaryInputsReusableAfterCompute();

    void infer(std::optional<Tensor> predictions, std::optional<Tensor> loss, Stream stream) override;
    void backProp(std::optional<Tensor> labels,
                  std::optional<Tensor> predictions,
                  std::optional<Tensor> lossGradient,
                  Stream stream) override;

    DynamicExpression lossExpression;
    DynamicExpression gradientExpression;
    uint64_t batchSize;
    uint64_t maxTotalValues;
    std::string predictionsName;
    std::string labelsName;
    std::string lossName;
    std::string gradientName;
    std::optional<float> lossWeight;
    std::optional<std::string> exampleWeightsName;

    struct SecondaryInputState {
        std::string inputName;
        std::string gradientName;
        std::optional<Tensor> input;
        std::optional<Tensor> errorOutput;
        std::optional<Layer*> previousLayer;
        Stream stream;
        Event readyEvent;
        Event reusableEvent;
        bool received = false;
    };
    std::vector<SecondaryInputState> secondaryInputs;

    std::optional<Tensor> offsetsInput;
    Stream offsetsStream;
    Event offsetsReadyEvent;
    Event offsetsReusableEvent;
    bool offsetsReceived = false;

    std::optional<Tensor> exampleWeightsInput;
    Stream exampleWeightsStream;
    Event exampleWeightsReadyEvent;
    Event exampleWeightsReusableEvent;
    bool exampleWeightsReceived = false;

    std::shared_ptr<PreparedDynamicExpression> lossPrepared;
    std::shared_ptr<StampedExecutionPlan> lossStamped;
    std::function<void(Stream&)> lossPreRunHook;

    std::shared_ptr<PreparedDynamicExpression> gradientPrepared;
    std::shared_ptr<StampedExecutionPlan> gradientStamped;
    std::function<void(Stream&)> gradientPreRunHook;
};

}  // namespace ThorImplementation
