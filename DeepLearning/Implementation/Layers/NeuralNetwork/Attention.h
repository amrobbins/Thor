#pragma once

#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "DeepLearning/Implementation/Layers/TrainingDropoutControllable.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ThorImplementation {

// Attention-specific semantic wrapper around CustomLayer's generic execution
// variants. CustomLayer knows only which execution variant is active; Attention
// owns the meaning of the stochastic and deterministic variants.
class Attention final : public CustomLayer, public TrainingDropoutControllable {
   public:
    struct RaggedQueryMetadata {
        uint32_t valuesInputPort = 0;
        uint32_t offsetsInputPort = 0;
        RowPartitionDescriptor rowPartitionDescriptor;
        uint64_t inputElementsPerValue = 0;
        uint64_t outputElementsPerValue = 0;
    };

    Attention(DynamicExpression expr,
              std::vector<std::string> inputNames,
              std::vector<std::string> outputNames,
              const TensorPlacement& placement,
              const std::vector<std::shared_ptr<PhysicalParameter>>& parameters,
              bool inferenceOnly,
              int64_t stampedId,
              std::vector<DeclaredOutputDescriptor> declaredOutputDescriptors,
              std::optional<DynamicExpressionVariantId> deterministicTrainingVariantId,
              bool trainingDropoutEnabled,
              std::vector<bool> inputDimensionsIncludeBatch = {},
              std::optional<uint32_t> fixedBatchCapacity = std::nullopt,
              std::optional<RaggedQueryMetadata> raggedQueryMetadata = std::nullopt)
        : CustomLayer(std::move(expr),
                      inputNames,
                      outputNames,
                      placement,
                      parameters,
                      inferenceOnly,
                      stampedId,
                      std::move(declaredOutputDescriptors),
                      false,
                      false,
                      std::move(inputDimensionsIncludeBatch),
                      fixedBatchCapacity),
          deterministicTrainingVariantId(deterministicTrainingVariantId),
          raggedQueryMetadata(raggedQueryMetadata),
          inputPortCount(static_cast<uint32_t>(inputNames.size())),
          outputPortCount(static_cast<uint32_t>(outputNames.size())) {
        if (this->raggedQueryMetadata.has_value()) {
            const RaggedQueryMetadata& metadata = this->raggedQueryMetadata.value();
            if (metadata.valuesInputPort >= inputPortCount || metadata.offsetsInputPort >= inputPortCount ||
                metadata.valuesInputPort == metadata.offsetsInputPort ||
                metadata.rowPartitionDescriptor.getMaxTotalValues() == 0 || metadata.inputElementsPerValue == 0 ||
                metadata.outputElementsPerValue == 0 || outputPortCount == 0) {
                throw std::invalid_argument("Attention ragged-query runtime metadata is invalid.");
            }
        }
        setTrainingDropoutEnabled(trainingDropoutEnabled);
    }

    void setTrainingDropoutEnabled(bool enabled) override {
        if (deterministicTrainingVariantId.has_value()) {
            setActiveTrainingExecutionVariant(
                enabled ? kPrimaryDynamicExpressionVariant : deterministicTrainingVariantId.value());
        }
        trainingDropoutEnabled = enabled;
    }

    [[nodiscard]] bool isTrainingDropoutEnabled() const override {
        return trainingDropoutEnabled;
    }

   protected:
    void beforeForwardExpressionRun(uint32_t connectionNumber, Stream& stream) override {
        (void)stream;
        if (!raggedQueryMetadata.has_value()) return;
        const uint32_t applicationIndex = connectionNumber / inputPortCount;
        const uint32_t offsetsFlatIndex = applicationIndex * inputPortCount + raggedQueryMetadata->offsetsInputPort;
        if (offsetsFlatIndex >= featureInputs.size() || !featureInputs[offsetsFlatIndex].has_value()) {
            throw std::runtime_error("Attention ragged query row-partition input is not connected for this application.");
        }

        RowPartitionRuntime rowPartition(
            featureInputs[offsetsFlatIndex].value(), raggedQueryMetadata->rowPartitionDescriptor);
        const std::optional<uint64_t> activeRows = rowPartition.getHostActiveValueCountIfAvailable();
        if (!activeRows.has_value()) {
            throw std::runtime_error(
                "Attention requires a host-known active-value count on its ragged query row partition for tail canonicalization.");
        }
        if (activeRows.value() > raggedQueryMetadata->rowPartitionDescriptor.getMaxTotalValues()) {
            throw std::runtime_error("Attention ragged query active row count exceeds packed capacity.");
        }
        if (activeRowsByApplication.size() <= applicationIndex) {
            activeRowsByApplication.resize(applicationIndex + 1, 0);
        }
        activeRowsByApplication[applicationIndex] = activeRows.value();
    }

    void afterForwardExpressionRun(uint32_t connectionNumber, Stream& stream) override {
        if (!raggedQueryMetadata.has_value()) return;
        const uint32_t applicationIndex = connectionNumber / inputPortCount;
        if (applicationIndex >= activeRowsByApplication.size()) {
            throw std::runtime_error("Attention has no ragged active-row state for this application.");
        }
        const uint32_t outputFlatIndex = applicationIndex * outputPortCount;
        if (outputFlatIndex >= featureOutputs.size() || !featureOutputs[outputFlatIndex].has_value()) {
            throw std::runtime_error("Attention ragged feature output is not connected for this application.");
        }
        Tensor output = featureOutputs[outputFlatIndex].value();
        const uint64_t activeRows = activeRowsByApplication[applicationIndex];
        zeroInactiveTail(output, activeRows, raggedQueryMetadata->outputElementsPerValue, stream);
    }

    void afterBackwardErrorExpressionRun(uint32_t connectionNumber, Stream& stream) override {
        if (!raggedQueryMetadata.has_value()) return;
        const uint32_t applicationIndex = connectionNumber / inputPortCount;
        if (applicationIndex >= activeRowsByApplication.size()) return;
        const uint32_t valuesFlatIndex = applicationIndex * inputPortCount + raggedQueryMetadata->valuesInputPort;
        if (valuesFlatIndex >= errorOutputs.size() || !errorOutputs[valuesFlatIndex].has_value()) return;
        Tensor dQuery = errorOutputs[valuesFlatIndex].value();
        const uint64_t activeRows = activeRowsByApplication[applicationIndex];
        zeroInactiveTail(dQuery, activeRows, raggedQueryMetadata->inputElementsPerValue, stream);
    }

   private:
    void zeroInactiveTail(Tensor tensor, uint64_t activeRows, uint64_t rowWidth, Stream stream) const {
        if (!raggedQueryMetadata.has_value()) return;
        const uint64_t capacityRows = raggedQueryMetadata->rowPartitionDescriptor.getMaxTotalValues();
        if (activeRows > capacityRows || rowWidth == 0 || tensor.getTotalNumElements() != capacityRows * rowWidth) {
            throw std::runtime_error("Attention ragged tensor shape is incompatible with its runtime metadata.");
        }
        if (activeRows == capacityRows) return;
        Tensor tail = tensor.aliasView({capacityRows - activeRows, rowWidth}, {rowWidth, 1}, activeRows * rowWidth);
        tail.memsetAsync(stream, 0);
    }

    std::optional<DynamicExpressionVariantId> deterministicTrainingVariantId;
    std::optional<RaggedQueryMetadata> raggedQueryMetadata;
    uint32_t inputPortCount = 0;
    uint32_t outputPortCount = 0;
    std::vector<uint64_t> activeRowsByApplication;
    bool trainingDropoutEnabled = true;
};

}  // namespace ThorImplementation
