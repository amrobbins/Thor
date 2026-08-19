#pragma once

#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Initializers/Initializer.h"
#include "DeepLearning/Api/Layers/Learning/CustomLayer.h"
#include "DeepLearning/Api/Layers/Learning/LayerEpilogue.h"
#include "DeepLearning/Api/Layers/TrainingDropoutControllable.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"
#include "DeepLearning/Api/Tensor/Tensor.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/TensorOperations/GpuAttention/CudnnAttention.h"

#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace Thor {

// Training-first transformer attention layer.
//
// Dense query input API shape:   [query_sequence, query_features]
// Dense context input API shape: [key_value_sequence, context_features] when contextInput() is provided.
// Dense output API shape:        [query_sequence, output_features]
// Ragged inputs use canonical RaggedTensor values [max_total_values, features] plus UINT32/UINT64 row partitions;
// the logical ragged output reuses the query row partition.
//
// Internally this layer performs per-token Q/K/V projections, optional RoPE on Q/K, cuDNN-backed SDPA, head merge,
// and a per-token output projection.  It intentionally does not manage paged KV caches; that path remains frozen at
// the expression level for inference validation.
class Attention : public CustomLayer, public TrainingDropoutControllable {
   public:
    class Builder;
    friend class Builder;

    Attention(ThorImplementation::DynamicExpression expression,
              std::vector<std::string> inputNames,
              const std::vector<TensorMap>& inputInterfaces,
              const std::vector<TensorMap>& outputInterfaces,
              std::vector<std::shared_ptr<ParameterSpecification>> parameters,
              std::optional<ThorImplementation::Expression> epilogue,
              std::vector<std::pair<std::string, Tensor>> epilogueInputBindings,
              uint32_t numHeads,
              uint32_t numKeyValueHeads,
              uint32_t headDim,
              uint32_t valueDim,
              uint32_t outputFeatures,
              bool hasBias,
              bool useRope,
              bool ropeInPlace,
              ThorImplementation::RotaryPositionEmbeddingOptions ropeOptions,
              int64_t queryRopePositionOffset,
              int64_t keyRopePositionOffset,
              ThorImplementation::AttentionMaskKind maskKind,
              int64_t diagonalLeftBound,
              int64_t diagonalRightBound,
              bool useAlibiMask,
              std::optional<double> attentionScale,
              float sdpaDropoutProbability,
              int64_t dropoutSeed,
              int64_t dropoutOffset,
              float outputDropoutProbability,
              int64_t outputDropoutSeed,
              std::optional<Tensor> residualInput,
              std::optional<RaggedTensor> raggedResidualInput,
              std::optional<Tensor> contextInput,
              std::optional<Tensor> scoreBiasInput,
              std::optional<Tensor> querySequenceLengthsInput,
              std::optional<Tensor> keyValueSequenceLengthsInput,
              std::optional<Tensor> queryRopePositionOffsetsInput,
              std::optional<Tensor> keyRopePositionOffsetsInput,
              std::optional<RaggedTensor> raggedFeatureInput,
              std::optional<RaggedTensor> raggedContextInput,
              DataType weightsDataType,
              DataType computeDataType,
              DataType outputDataType)
        : CustomLayer(std::move(expression),
                      std::move(inputNames),
                      {"feature_output"},
                      inputInterfaces,
                      outputInterfaces,
                      std::move(parameters),
                      SerializationContract::LAYER_PROVIDES_OWN_ARCHITECTURE,
                      false,
                      false,
                      [&]() {
                          std::set<std::string> names;
                          if (raggedFeatureInput.has_value()) {
                              names.insert("feature_input");
                              names.insert("query_row_partition");
                              if (raggedResidualInput.has_value()) {
                                  names.insert("residual_input");
                              }
                              for (const auto& [name, tensor] : epilogueInputBindings) {
                                  (void)tensor;
                                  names.insert(name);
                              }
                          }
                          if (raggedContextInput.has_value()) {
                              names.insert("context_input");
                              names.insert("key_value_row_partition");
                          } else if (raggedFeatureInput.has_value() && !contextInput.has_value()) {
                              // Ragged self-attention shares the query partition with K/V.
                              names.insert("key_value_row_partition");
                          }
                          return names;
                      }(),
                      raggedFeatureInput.has_value() ? std::set<std::string>{"feature_output"} : std::set<std::string>{},
                      raggedFeatureInput.has_value()
                          ? std::optional<uint32_t>(static_cast<uint32_t>(raggedFeatureInput->getBatchSize()))
                          : (raggedContextInput.has_value()
                                 ? std::optional<uint32_t>(static_cast<uint32_t>(raggedContextInput->getBatchSize()))
                                 : std::nullopt)),
          numHeads(numHeads),
          numKeyValueHeads(numKeyValueHeads),
          headDim(headDim),
          valueDim(valueDim),
          outputFeatures(outputFeatures),
          hasBias(hasBias),
          useRope(useRope),
          ropeInPlace(ropeInPlace),
          ropeOptions(ropeOptions),
          queryRopePositionOffset(queryRopePositionOffset),
          keyRopePositionOffset(keyRopePositionOffset),
          maskKind(maskKind),
          diagonalLeftBound(diagonalLeftBound),
          diagonalRightBound(diagonalRightBound),
          useAlibiMask(useAlibiMask),
          attentionScale(attentionScale),
          sdpaDropoutProbability(sdpaDropoutProbability),
          dropoutSeed(dropoutSeed),
          dropoutOffset(dropoutOffset),
          outputDropoutProbability(outputDropoutProbability),
          outputDropoutSeed(outputDropoutSeed),
          residualInput(std::move(residualInput)),
          raggedResidualInput(std::move(raggedResidualInput)),
          epilogue(std::move(epilogue)),
          epilogueInputBindings(std::move(epilogueInputBindings)),
          contextInput(std::move(contextInput)),
          scoreBiasInput(std::move(scoreBiasInput)),
          querySequenceLengthsInput(std::move(querySequenceLengthsInput)),
          keyValueSequenceLengthsInput(std::move(keyValueSequenceLengthsInput)),
          queryRopePositionOffsetsInput(std::move(queryRopePositionOffsetsInput)),
          keyRopePositionOffsetsInput(std::move(keyRopePositionOffsetsInput)),
          raggedFeatureInput(std::move(raggedFeatureInput)),
          raggedContextInput(std::move(raggedContextInput)),
          weightsDataType(weightsDataType),
          computeDataType(computeDataType),
          outputDataType(outputDataType) {
        if (this->raggedFeatureInput.has_value()) {
            raggedFeatureOutput = RaggedTensor(getOutput("feature_output"), this->raggedFeatureInput->getOffsets());
        }
    }

    ~Attention() override = default;

    // Dormant compile-time experiment switch for benchmarking packed QKV projection against the maintained split-Q/K/V path.
    // Packed QKV is intentionally not updated by split-path projection/RoPE fusion work unless a future use case reactivates it.
    static constexpr bool USE_PACKED_QKV_PROJECTION = false;

    std::shared_ptr<Layer> clone() const override { return std::make_shared<Attention>(*this); }
    std::string getLayerType() const override { return "Attention"; }

    nlohmann::json serialize(thor_file::TarWriter& archiveWriter,
                             Stream stream,
                             bool saveOptimizerState,
                             ThorImplementation::StampedNetwork& stampedNetwork) const override;
    static void deserialize(std::shared_ptr<thor_file::TarReader>& archiveReader, const nlohmann::json& j, Network* network);
    nlohmann::json architectureJson() const override;

    static const char* epilogueInputName() { return "__attention_epilogue_input"; }
    static const char* epilogueOutputName() { return "__attention_epilogue_output"; }

    [[nodiscard]] static ThorImplementation::Expression epilogueInput(
        std::optional<ThorImplementation::DataType> computeDType = std::nullopt,
        std::optional<ThorImplementation::DataType> outputDType = std::nullopt) {
        return LayerEpilogue::input(epilogueInputName(), computeDType, outputDType);
    }

    [[nodiscard]] static ThorImplementation::Expression epilogueAuxInput(
        const std::string& inputName,
        std::optional<ThorImplementation::DataType> computeDType = std::nullopt,
        std::optional<ThorImplementation::DataType> outputDType = std::nullopt) {
        validateEpilogueAuxInputName(inputName);
        return LayerEpilogue::input(inputName, computeDType, outputDType);
    }

    [[nodiscard]] static ThorImplementation::ExpressionDefinition makeEpilogueDefinition(
        const ThorImplementation::Expression& expression,
        const std::vector<std::string>& auxiliaryInputNames = {}) {
        ThorImplementation::ExpressionDefinition definition =
            LayerEpilogue::makeDefinition(expression, epilogueInputName(), auxiliaryInputNames, epilogueOutputName(), "Attention");
        validateEpilogueShapePreserving(definition);
        return definition;
    }

    static void validateEpilogueExpression(const ThorImplementation::Expression& expression,
                                           const std::vector<std::string>& auxiliaryInputNames = {}) {
        ThorImplementation::ExpressionDefinition definition =
            LayerEpilogue::makeDefinition(expression, epilogueInputName(), auxiliaryInputNames, epilogueOutputName(), "Attention");
        validateEpilogueShapePreserving(definition);
    }

    static void validateEpilogueDefinition(const ThorImplementation::ExpressionDefinition& definition,
                                           const std::vector<std::string>& auxiliaryInputNames = {}) {
        LayerEpilogue::validateDefinition(definition, epilogueInputName(), auxiliaryInputNames, epilogueOutputName(), "Attention");
        validateEpilogueShapePreserving(definition);
    }

    [[nodiscard]] static ThorImplementation::Expression epilogueExpressionFromDefinition(
        const ThorImplementation::ExpressionDefinition& definition,
        const std::vector<std::string>& auxiliaryInputNames = {}) {
        validateEpilogueDefinition(definition, auxiliaryInputNames);
        return LayerEpilogue::expressionFromDefinition(
            definition, epilogueInputName(), auxiliaryInputNames, epilogueOutputName(), "Attention");
    }

    [[nodiscard]] static ThorImplementation::Expression applyEpilogue(const ThorImplementation::Expression& input,
                                                                      const ThorImplementation::Expression& epilogue) {
        return LayerEpilogue::apply(input, epilogue, epilogueInputName());
    }

    static void validateEpilogueAuxInputName(const std::string& inputName);
    static void validateEpilogueShapePreserving(const ThorImplementation::ExpressionDefinition& definition);

    uint32_t getNumHeads() const { return numHeads; }
    uint32_t getNumKeyValueHeads() const { return numKeyValueHeads; }
    uint32_t getHeadDim() const { return headDim; }
    uint32_t getValueDim() const { return valueDim; }
    uint32_t getOutputFeatures() const { return outputFeatures; }
    bool getHasBias() const { return hasBias; }
    bool getUseRope() const { return useRope; }
    // When true, private split Q/K projection outputs may be rotated in-place to reduce peak memory.
    // Defaults false because the out-of-place fused RoPE path has benchmarked faster.
    bool getRopeInPlace() const { return ropeInPlace; }
    // Shared RoPE basis/scaling configuration. position_offset remains the legacy/shared default captured from
    // ropeOptions(); the effective Q/K origins are always reported by the dedicated getters below.
    const ThorImplementation::RotaryPositionEmbeddingOptions& getRopeOptions() const { return ropeOptions; }
    int64_t getQueryRopePositionOffset() const { return queryRopePositionOffset; }
    int64_t getKeyRopePositionOffset() const { return keyRopePositionOffset; }
    ThorImplementation::AttentionMaskKind getMaskKind() const { return maskKind; }
    int64_t getDiagonalLeftBound() const { return diagonalLeftBound; }
    int64_t getDiagonalRightBound() const { return diagonalRightBound; }
    bool getUseAlibiMask() const { return useAlibiMask; }
    std::optional<double> getAttentionScale() const { return attentionScale; }
    // Canonical name: this is dropout on the SDPA probability matrix, not on
    // the Attention output projection. getDropoutProbability() remains as a
    // source-compatible alias for older callers.
    float getSdpaDropoutProbability() const { return sdpaDropoutProbability; }
    float getDropoutProbability() const { return getSdpaDropoutProbability(); }
    int64_t getSdpaDropoutSeed() const { return dropoutSeed; }
    int64_t getSdpaDropoutOffset() const { return dropoutOffset; }
    // Backward-compatible aliases.
    int64_t getDropoutSeed() const { return getSdpaDropoutSeed(); }
    int64_t getDropoutOffset() const { return getSdpaDropoutOffset(); }
    float getOutputDropoutProbability() const { return outputDropoutProbability; }
    int64_t getOutputDropoutSeed() const { return outputDropoutSeed; }
    std::optional<Tensor> getResidualInput() const { return residualInput; }
    std::optional<RaggedTensor> getRaggedResidualInput() const { return raggedResidualInput; }
    bool getUseResidual() const { return residualInput.has_value(); }
    std::optional<Tensor> getFeatureInput() const override { return getInputInterface().at("feature_input"); }
    std::optional<Tensor> getContextInput() const { return contextInput; }
    bool getUseCrossAttention() const { return contextInput.has_value(); }
    std::optional<Tensor> getScoreBiasInput() const { return scoreBiasInput; }
    bool getUseScoreBias() const { return scoreBiasInput.has_value(); }
    std::optional<Tensor> getQuerySequenceLengthsInput() const { return querySequenceLengthsInput; }
    std::optional<Tensor> getKeyValueSequenceLengthsInput() const { return keyValueSequenceLengthsInput; }
    bool getUseSequenceLengths() const { return querySequenceLengthsInput.has_value(); }
    // Optional ragged-only per-row absolute RoPE origins. When present, they replace the scalar origin for that side.
    // API shape is [1]; at runtime the normal batch dimension yields one INT32 origin per logical ragged row.
    std::optional<Tensor> getQueryRopePositionOffsetsInput() const { return queryRopePositionOffsetsInput; }
    std::optional<Tensor> getKeyRopePositionOffsetsInput() const { return keyRopePositionOffsetsInput; }
    bool getUseRagged() const { return raggedFeatureInput.has_value() || raggedContextInput.has_value(); }
    bool getQueryRagged() const { return raggedFeatureInput.has_value(); }
    bool getKeyValueRagged() const {
        return raggedContextInput.has_value() || (raggedFeatureInput.has_value() && !contextInput.has_value());
    }
    std::optional<RaggedTensor> getRaggedFeatureInput() const { return raggedFeatureInput; }
    std::optional<RaggedTensor> getRaggedContextInput() const { return raggedContextInput; }
    std::optional<RaggedTensor> getRaggedFeatureOutput() const { return raggedFeatureOutput; }
    DataType getWeightsDataType() const { return weightsDataType; }
    DataType getComputeDataType() const { return computeDataType; }
    DataType getOutputDataType() const { return outputDataType; }
    bool hasEpilogue() const { return epilogue.has_value(); }
    const std::vector<std::pair<std::string, Tensor>>& getEpilogueInputBindings() const { return epilogueInputBindings; }

   protected:
    std::shared_ptr<ThorImplementation::CustomLayer> createPhysicalLayer(
        ThorImplementation::DynamicExpression expression,
        std::vector<std::string> physicalInputNames,
        std::vector<std::string> physicalOutputNames,
        ThorImplementation::TensorPlacement placement,
        const std::vector<std::shared_ptr<ThorImplementation::PhysicalParameter>>& physicalParameters,
        bool inferenceOnly,
        int64_t stampedId,
        std::vector<ThorImplementation::CustomLayer::DeclaredOutputDescriptor> declaredOutputDescriptors,
        bool usesBatchValidity,
        bool requiresFullBatch,
        std::vector<bool> inputDimensionsIncludeBatch,
        std::optional<uint32_t> fixedBatchCapacity) const override;

   private:
    uint32_t numHeads;
    uint32_t numKeyValueHeads;
    uint32_t headDim;
    uint32_t valueDim;
    uint32_t outputFeatures;
    bool hasBias;
    bool useRope;
    bool ropeInPlace;
    ThorImplementation::RotaryPositionEmbeddingOptions ropeOptions;
    int64_t queryRopePositionOffset;
    int64_t keyRopePositionOffset;
    ThorImplementation::AttentionMaskKind maskKind;
    int64_t diagonalLeftBound;
    int64_t diagonalRightBound;
    bool useAlibiMask;
    std::optional<double> attentionScale;
    float sdpaDropoutProbability;
    int64_t dropoutSeed;
    int64_t dropoutOffset;
    float outputDropoutProbability;
    int64_t outputDropoutSeed;
    std::optional<Tensor> residualInput;
    std::optional<RaggedTensor> raggedResidualInput;
    const std::optional<ThorImplementation::Expression> epilogue;
    std::vector<std::pair<std::string, Tensor>> epilogueInputBindings;
    mutable std::optional<ThorImplementation::ExpressionDefinition> serializableEpilogue;
    std::optional<Tensor> contextInput;
    std::optional<Tensor> scoreBiasInput;
    std::optional<Tensor> querySequenceLengthsInput;
    std::optional<Tensor> keyValueSequenceLengthsInput;
    std::optional<Tensor> queryRopePositionOffsetsInput;
    std::optional<Tensor> keyRopePositionOffsetsInput;
    std::optional<RaggedTensor> raggedFeatureInput;
    std::optional<RaggedTensor> raggedContextInput;
    std::optional<RaggedTensor> raggedFeatureOutput;
    DataType weightsDataType;
    DataType computeDataType;
    DataType outputDataType;
};

class Attention::Builder {
   public:
    virtual ~Builder() = default;

    virtual Attention build();

    virtual Attention::Builder& network(Network& network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &network;
        return *this;
    }

    virtual Attention::Builder& featureInput(Tensor input) {
        THOR_THROW_IF_FALSE(!this->_featureInput.has_value());
        THOR_THROW_IF_FALSE(!this->_raggedFeatureInput.has_value());
        this->_featureInput = input;
        return *this;
    }

    virtual Attention::Builder& featureInput(RaggedTensor input) {
        THOR_THROW_IF_FALSE(!this->_featureInput.has_value());
        THOR_THROW_IF_FALSE(!this->_raggedFeatureInput.has_value());
        this->_raggedFeatureInput = input;
        this->_featureInput = input.getValues();
        return *this;
    }

    virtual Attention::Builder& contextInput(Tensor input) {
        THOR_THROW_IF_FALSE(!this->_contextInput.has_value());
        THOR_THROW_IF_FALSE(!this->_raggedContextInput.has_value());
        this->_contextInput = input;
        return *this;
    }

    virtual Attention::Builder& contextInput(RaggedTensor input) {
        THOR_THROW_IF_FALSE(!this->_contextInput.has_value());
        THOR_THROW_IF_FALSE(!this->_raggedContextInput.has_value());
        this->_raggedContextInput = input;
        this->_contextInput = input.getValues();
        return *this;
    }

    virtual Attention::Builder& scoreBiasInput(Tensor input) {
        THOR_THROW_IF_FALSE(!this->_scoreBiasInput.has_value());
        this->_scoreBiasInput = input;
        return *this;
    }

    virtual Attention::Builder& numHeads(uint32_t value) {
        THOR_THROW_IF_FALSE(!this->_numHeads.has_value());
        this->_numHeads = value;
        return *this;
    }

    virtual Attention::Builder& numKeyValueHeads(uint32_t value) {
        THOR_THROW_IF_FALSE(!this->_numKeyValueHeads.has_value());
        this->_numKeyValueHeads = value;
        return *this;
    }

    virtual Attention::Builder& headDim(uint32_t value) {
        THOR_THROW_IF_FALSE(!this->_headDim.has_value());
        this->_headDim = value;
        return *this;
    }

    virtual Attention::Builder& valueDim(uint32_t value) {
        THOR_THROW_IF_FALSE(!this->_valueDim.has_value());
        this->_valueDim = value;
        return *this;
    }

    virtual Attention::Builder& outputFeatures(uint32_t value) {
        THOR_THROW_IF_FALSE(!this->_outputFeatures.has_value());
        this->_outputFeatures = value;
        return *this;
    }

    virtual Attention::Builder& hasBias(bool value) {
        THOR_THROW_IF_FALSE(!this->_hasBias.has_value());
        this->_hasBias = value;
        return *this;
    }

    virtual Attention::Builder& causal(bool enabled = true) {
        THOR_THROW_IF_FALSE(!this->_maskKind.has_value());
        this->_maskKind = enabled ? ThorImplementation::AttentionMaskKind::CausalTopLeft : ThorImplementation::AttentionMaskKind::None;
        return *this;
    }

    virtual Attention::Builder& maskKind(ThorImplementation::AttentionMaskKind value) {
        THOR_THROW_IF_FALSE(!this->_maskKind.has_value());
        this->_maskKind = value;
        return *this;
    }

    virtual Attention::Builder& diagonalLeftBound(int64_t value) {
        THOR_THROW_IF_FALSE(!this->_diagonalLeftBound.has_value());
        this->_diagonalLeftBound = value;
        return *this;
    }

    virtual Attention::Builder& diagonalRightBound(int64_t value) {
        THOR_THROW_IF_FALSE(!this->_diagonalRightBound.has_value());
        this->_diagonalRightBound = value;
        return *this;
    }

    virtual Attention::Builder& useAlibiMask(bool value = true) {
        THOR_THROW_IF_FALSE(!this->_useAlibiMask.has_value());
        this->_useAlibiMask = value;
        return *this;
    }

    virtual Attention::Builder& attentionScale(double value) {
        THOR_THROW_IF_FALSE(!this->_attentionScale.has_value());
        this->_attentionScale = value;
        return *this;
    }

    // SDPA attention-probability dropout. Applied only during training;
    // validation and inference execute a separately compiled no-dropout SDPA plan.
    virtual Attention::Builder& sdpaDropoutProbability(float value) {
        THOR_THROW_IF_FALSE(!this->_sdpaDropoutProbability.has_value());
        this->_sdpaDropoutProbability = value;
        return *this;
    }

    // Backward-compatible C++ alias. New code should use sdpaDropoutProbability().
    virtual Attention::Builder& dropoutProbability(float value) { return sdpaDropoutProbability(value); }

    virtual Attention::Builder& sdpaDropoutSeed(int64_t value) {
        THOR_THROW_IF_FALSE(!this->_dropoutSeed.has_value());
        this->_dropoutSeed = value;
        return *this;
    }

    virtual Attention::Builder& sdpaDropoutOffset(int64_t value) {
        THOR_THROW_IF_FALSE(!this->_dropoutOffset.has_value());
        this->_dropoutOffset = value;
        return *this;
    }

    // Backward-compatible C++ aliases.
    virtual Attention::Builder& dropoutSeed(int64_t value) { return sdpaDropoutSeed(value); }
    virtual Attention::Builder& dropoutOffset(int64_t value) { return sdpaDropoutOffset(value); }

    virtual Attention::Builder& sdpaDropout(float probability, int64_t seed, int64_t offset) {
        return sdpaDropoutProbability(probability).sdpaDropoutSeed(seed).sdpaDropoutOffset(offset);
    }

    // Backward-compatible C++ convenience alias.
    virtual Attention::Builder& dropout(float probability, int64_t seed, int64_t offset) {
        return sdpaDropout(probability, seed, offset);
    }

    // Dropout on the post-projection Attention branch. When residualInput() is
    // present, Attention guarantees residual + dropout(projected_output). Thor
    // chooses GEMM+residual when this rate is inactive and a fused native
    // dropout+residual post-op when it is active.
    virtual Attention::Builder& outputDropoutProbability(float value) {
        THOR_THROW_IF_FALSE(!this->_outputDropoutProbability.has_value());
        this->_outputDropoutProbability = value;
        return *this;
    }

    // Optional deterministic seed for output dropout. If omitted while output
    // dropout is enabled, Thor chooses an independent per-layer seed at build time.
    virtual Attention::Builder& outputDropoutSeed(int64_t value) {
        THOR_THROW_IF_FALSE(!this->_outputDropoutSeed.has_value());
        this->_outputDropoutSeed = value;
        return *this;
    }

    virtual Attention::Builder& residualInput(Tensor input) {
        THOR_THROW_IF_FALSE(!this->_residualInput.has_value());
        THOR_THROW_IF_FALSE(!this->_raggedResidualInput.has_value());
        this->_residualInput = input;
        return *this;
    }

    virtual Attention::Builder& residualInput(RaggedTensor input) {
        THOR_THROW_IF_FALSE(!this->_residualInput.has_value());
        THOR_THROW_IF_FALSE(!this->_raggedResidualInput.has_value());
        this->_raggedResidualInput = input;
        this->_residualInput = input.getValues();
        return *this;
    }

    virtual Attention::Builder& querySequenceLengthsInput(Tensor input) {
        THOR_THROW_IF_FALSE(!this->_querySequenceLengthsInput.has_value());
        this->_querySequenceLengthsInput = input;
        return *this;
    }

    virtual Attention::Builder& keyValueSequenceLengthsInput(Tensor input) {
        THOR_THROW_IF_FALSE(!this->_keyValueSequenceLengthsInput.has_value());
        this->_keyValueSequenceLengthsInput = input;
        return *this;
    }

    virtual Attention::Builder& useRope(bool value = true) {
        THOR_THROW_IF_FALSE(!this->_useRope.has_value());
        this->_useRope = value;
        return *this;
    }

    virtual Attention::Builder& ropeInPlace(bool value = true) {
        THOR_THROW_IF_FALSE(!this->_ropeInPlace.has_value());
        this->_ropeInPlace = value;
        return *this;
    }

    virtual Attention::Builder& ropeOptions(ThorImplementation::RotaryPositionEmbeddingOptions value) {
        THOR_THROW_IF_FALSE(!this->_ropeOptions.has_value());
        this->_ropeOptions = value;
        this->_useRope = true;
        return *this;
    }

    // Convenience for the common self-attention case: use one positional origin for Q and K.
    virtual Attention::Builder& ropePositionOffset(int64_t value) {
        THOR_THROW_IF_FALSE(!this->_queryRopePositionOffset.has_value());
        THOR_THROW_IF_FALSE(!this->_keyRopePositionOffset.has_value());
        this->_queryRopePositionOffset = value;
        this->_keyRopePositionOffset = value;
        this->_useRope = true;
        return *this;
    }

    // Cross-attention may place Q and K on different absolute timelines while sharing the same RoPE basis/scaling.
    virtual Attention::Builder& queryRopePositionOffset(int64_t value) {
        THOR_THROW_IF_FALSE(!this->_queryRopePositionOffset.has_value());
        this->_queryRopePositionOffset = value;
        this->_useRope = true;
        return *this;
    }

    virtual Attention::Builder& keyRopePositionOffset(int64_t value) {
        THOR_THROW_IF_FALSE(!this->_keyRopePositionOffset.has_value());
        this->_keyRopePositionOffset = value;
        this->_useRope = true;
        return *this;
    }

    // Ragged-only per-row absolute positional origins. The logical API tensor shape is [1], producing one
    // INT32 origin per batch row at runtime. A supplied per-row origin replaces the scalar origin for that side.
    virtual Attention::Builder& queryRopePositionOffsetsInput(Tensor input) {
        THOR_THROW_IF_FALSE(!this->_queryRopePositionOffsetsInput.has_value());
        this->_queryRopePositionOffsetsInput = input;
        this->_useRope = true;
        return *this;
    }

    virtual Attention::Builder& keyRopePositionOffsetsInput(Tensor input) {
        THOR_THROW_IF_FALSE(!this->_keyRopePositionOffsetsInput.has_value());
        this->_keyRopePositionOffsetsInput = input;
        this->_useRope = true;
        return *this;
    }

    virtual Attention::Builder& weightsDataType(DataType value) {
        THOR_THROW_IF_FALSE(!this->_weightsDataType.has_value());
        this->_weightsDataType = value;
        return *this;
    }

    virtual Attention::Builder& computeDataType(DataType value) {
        THOR_THROW_IF_FALSE(!this->_computeDataType.has_value());
        this->_computeDataType = value;
        return *this;
    }

    virtual Attention::Builder& outputDataType(DataType value) {
        THOR_THROW_IF_FALSE(!this->_outputDataType.has_value());
        this->_outputDataType = value;
        return *this;
    }

    virtual Attention::Builder& epilogue(const ThorImplementation::Expression& expression) {
        THOR_THROW_IF_FALSE(!this->_epilogue.has_value());
        Attention::validateEpilogueExpression(expression, epilogueAuxInputNames());
        _epilogue = expression;
        return *this;
    }

    virtual Attention::Builder& epilogueInput(const std::string& inputName, Tensor tensor) {
        Attention::validateEpilogueAuxInputName(inputName);
        THOR_THROW_IF_FALSE(tensor.isInitialized());
        for (const auto& [existingName, existingTensor] : _epilogueInputBindings) {
            (void)existingTensor;
            if (existingName == inputName) {
                throw std::invalid_argument("Attention epilogue input name is duplicated: " + inputName + ".");
            }
        }
        _epilogueInputBindings.emplace_back(inputName, tensor);
        if (_epilogue.has_value()) {
            Attention::validateEpilogueExpression(_epilogue.value(), epilogueAuxInputNames());
        }
        return *this;
    }

    virtual Attention::Builder& weightsInitializer(std::shared_ptr<Initializer> initializer) {
        THOR_THROW_IF_FALSE(this->_weightsInitializer == nullptr);
        this->_weightsInitializer = std::move(initializer);
        return *this;
    }

    virtual Attention::Builder& biasInitializer(std::shared_ptr<Initializer> initializer) {
        THOR_THROW_IF_FALSE(this->_biasInitializer == nullptr);
        this->_biasInitializer = std::move(initializer);
        return *this;
    }

    virtual Attention::Builder& optimizer(std::shared_ptr<Optimizer> optimizer) {
        THOR_THROW_IF_FALSE(this->_optimizer == nullptr);
        this->_optimizer = std::move(optimizer);
        return *this;
    }

   private:
    void verifyConfig() const;

    std::optional<Network*> _network;
    std::optional<Tensor> _featureInput;
    std::optional<Tensor> _contextInput;
    std::optional<RaggedTensor> _raggedFeatureInput;
    std::optional<RaggedTensor> _raggedContextInput;
    std::optional<Tensor> _scoreBiasInput;
    std::optional<uint32_t> _numHeads;
    std::optional<uint32_t> _numKeyValueHeads;
    std::optional<uint32_t> _headDim;
    std::optional<uint32_t> _valueDim;
    std::optional<uint32_t> _outputFeatures;
    std::optional<bool> _hasBias;
    std::optional<ThorImplementation::AttentionMaskKind> _maskKind;
    std::optional<int64_t> _diagonalLeftBound;
    std::optional<int64_t> _diagonalRightBound;
    std::optional<bool> _useAlibiMask;
    std::optional<double> _attentionScale;
    std::optional<float> _sdpaDropoutProbability;
    std::optional<int64_t> _dropoutSeed;
    std::optional<int64_t> _dropoutOffset;
    std::optional<float> _outputDropoutProbability;
    std::optional<int64_t> _outputDropoutSeed;
    std::optional<Tensor> _residualInput;
    std::optional<RaggedTensor> _raggedResidualInput;
    std::optional<Tensor> _querySequenceLengthsInput;
    std::optional<Tensor> _keyValueSequenceLengthsInput;
    std::optional<bool> _useRope;
    std::optional<bool> _ropeInPlace;
    std::optional<ThorImplementation::RotaryPositionEmbeddingOptions> _ropeOptions;
    std::optional<int64_t> _queryRopePositionOffset;
    std::optional<int64_t> _keyRopePositionOffset;
    std::optional<Tensor> _queryRopePositionOffsetsInput;
    std::optional<Tensor> _keyRopePositionOffsetsInput;
    std::optional<DataType> _weightsDataType;
    std::optional<DataType> _computeDataType;
    std::optional<DataType> _outputDataType;
    std::optional<ThorImplementation::Expression> _epilogue;
    std::vector<std::pair<std::string, Tensor>> _epilogueInputBindings;
    std::shared_ptr<Initializer> _weightsInitializer;
    std::shared_ptr<Initializer> _biasInitializer;
    std::shared_ptr<Optimizer> _optimizer;

    std::vector<std::string> epilogueAuxInputNames() const {
        std::vector<std::string> names;
        names.reserve(_epilogueInputBindings.size());
        for (const auto& [name, tensor] : _epilogueInputBindings) {
            (void)tensor;
            names.push_back(name);
        }
        return names;
    }
};

}  // namespace Thor
