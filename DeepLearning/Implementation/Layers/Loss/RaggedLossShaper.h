#pragma once

#include "DeepLearning/Implementation/Layers/Layer.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/TensorOperations/Cub/CubReduction.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace ThorImplementation {

// Internal loss shaper for rank-1 ragged raw losses.
//
// Raw loss values use packed storage [max_total_values, ...trailing...], while
// offsets is the canonical [batch_size + 1] partition.  The supported shapes
// are intentionally narrower than dense LossShaper:
//
//   RAW         packed raw loss, preserving the exact input partition
//   PER_EXAMPLE dense [B, 1], summing every active token/trailing scalar in row B
//   BATCH       dense [1, 1], sum per-row losses then divide by valid examples
//
// PER_OUTPUT has no unambiguous meaning for unequal sequence lengths and is
// rejected.  Empty valid rows therefore contribute zero to the numerator but
// still participate in the BATCH denominator.  For a partial batch, invalid
// tail rows must be empty in the canonical offsets, matching RaggedCustomLoss
// and Thor's ragged batch materialization contract.
class RaggedLossShaper : public Layer {
   public:
    enum class OutputLossType { RAW, BATCH, PER_EXAMPLE, PER_OUTPUT };
    enum class InputConnection : int { VALUES = 0, OFFSETS = 1 };

    RaggedLossShaper(OutputLossType outputLossType, uint64_t batchSize, uint64_t maxTotalValues);
    ~RaggedLossShaper() override = default;

    std::optional<Tensor> createFeatureOutputTensor() override;
    std::optional<Tensor> connectToPreviousLayer(Layer* previousLayer,
                                                 std::optional<Tensor> featureInput,
                                                 Stream stream,
                                                 bool backPropagateError,
                                                 int connectionType = 0) override;

    void compileImpl() override;
    void initialize() override;
    void cleanup() override;
    void forward(std::optional<Tensor> inputTensor, bool validationPass, uint32_t validExampleCount = 0) override;

    std::vector<Stream> getProcessingStreams() override;
    std::vector<Event> getSynchronizeEvents() override;
    void ensureNoDeviceCrossing() override;

    [[nodiscard]] OutputLossType getOutputLossType() const { return outputLossType; }
    [[nodiscard]] uint64_t getLogicalBatchSize() const { return batchSize; }
    [[nodiscard]] uint64_t getMaxTotalValues() const { return maxTotalValues; }
    [[nodiscard]] std::optional<Tensor> getOffsetsInput() const { return offsetsInput; }
    [[nodiscard]] std::optional<Tensor> getRawOutputOffsets() const {
        if (outputLossType == OutputLossType::RAW)
            return offsetsInput;
        return std::nullopt;
    }

    std::string getType() override { return "RaggedLossShaper"; }
    bool supportsPartialBatches() const override { return true; }

   private:
    using TensorMap = DynamicExpression::TensorMap;

    uint32_t resolveValidExampleCount(uint32_t validExampleCount) const;
    void recordLogicalBatchCardinality(uint32_t validExampleCount);
    uint64_t elementsPerValue() const;
    DynamicExpression buildPerExampleExpression() const;
    void advanceDataIfReady(bool validationPass);

    void infer(std::optional<Tensor> inputTensor, std::optional<Tensor> outputTensor, Stream stream) override;
    void backProp(std::optional<Tensor> dataIn,
                  std::optional<Tensor> errorIn,
                  std::optional<Tensor> errorOut,
                  Stream stream) override;

    OutputLossType outputLossType;
    uint64_t batchSize;
    uint64_t maxTotalValues;

    std::optional<Tensor> offsetsInput;
    Stream offsetsStream;
    Event offsetsReadyEvent;
    Event offsetsReusableEvent;
    bool valuesReceived = false;
    bool offsetsReceived = false;
    uint32_t currentValidExampleCount = 0;
    bool batchCardinalitySet = false;

    std::optional<Tensor> perExampleWorkspace;
    std::shared_ptr<PreparedDynamicExpression> perExamplePrepared;
    std::shared_ptr<StampedExecutionPlan> perExampleStamped;
    std::function<void(Stream&)> perExamplePreRunHook;
    std::shared_ptr<StampedCubReduction> batchReduction;
};

}  // namespace ThorImplementation
