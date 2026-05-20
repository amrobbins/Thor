#pragma once

#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Initializers/Initializer.h"
#include "DeepLearning/Api/Layers/Learning/CustomLayer.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"
#include "DeepLearning/Api/Tensor/Tensor.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/TensorOperations/GpuAttention/CudnnAttention.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace Thor {

// Training-first transformer attention layer.
//
// Query input API shape:   [query_sequence, query_features]
// Context input API shape: [key_value_sequence, context_features] when contextInput() is provided.
// Output API shape:        [query_sequence, output_features]
//
// Internally this layer performs per-token Q/K/V projections, optional RoPE on Q/K, cuDNN-backed SDPA, head merge,
// and a per-token output projection.  It intentionally does not manage paged KV caches; that path remains frozen at
// the expression level for inference validation.
class Attention : public CustomLayer {
   public:
    class Builder;
    friend class Builder;

    Attention(ThorImplementation::DynamicExpression expression,
              std::vector<std::string> inputNames,
              const std::vector<TensorMap>& inputInterfaces,
              const std::vector<TensorMap>& outputInterfaces,
              std::vector<std::shared_ptr<ParameterSpecification>> parameters,
              uint32_t numHeads,
              uint32_t numKeyValueHeads,
              uint32_t headDim,
              uint32_t valueDim,
              uint32_t outputFeatures,
              bool hasBias,
              bool useRope,
              bool ropeInPlace,
              ThorImplementation::RotaryPositionEmbeddingOptions ropeOptions,
              ThorImplementation::AttentionMaskKind maskKind,
              int64_t diagonalLeftBound,
              int64_t diagonalRightBound,
              bool useAlibiMask,
              std::optional<double> attentionScale,
              float dropoutProbability,
              int64_t dropoutSeed,
              int64_t dropoutOffset,
              std::optional<Tensor> contextInput,
              std::optional<Tensor> scoreBiasInput,
              std::optional<Tensor> querySequenceLengthsInput,
              std::optional<Tensor> keyValueSequenceLengthsInput,
              std::optional<Tensor> queryRaggedOffsetsInput,
              std::optional<Tensor> keyValueRaggedOffsetsInput,
              Tensor::DataType weightsDataType,
              Tensor::DataType computeDataType,
              Tensor::DataType outputDataType)
        : CustomLayer(std::move(expression),
                      std::move(inputNames),
                      {"feature_output"},
                      inputInterfaces,
                      outputInterfaces,
                      std::move(parameters),
                      false),
          numHeads(numHeads),
          numKeyValueHeads(numKeyValueHeads),
          headDim(headDim),
          valueDim(valueDim),
          outputFeatures(outputFeatures),
          hasBias(hasBias),
          useRope(useRope),
          ropeInPlace(ropeInPlace),
          ropeOptions(ropeOptions),
          maskKind(maskKind),
          diagonalLeftBound(diagonalLeftBound),
          diagonalRightBound(diagonalRightBound),
          useAlibiMask(useAlibiMask),
          attentionScale(attentionScale),
          dropoutProbability(dropoutProbability),
          dropoutSeed(dropoutSeed),
          dropoutOffset(dropoutOffset),
          contextInput(std::move(contextInput)),
          scoreBiasInput(std::move(scoreBiasInput)),
          querySequenceLengthsInput(std::move(querySequenceLengthsInput)),
          keyValueSequenceLengthsInput(std::move(keyValueSequenceLengthsInput)),
          queryRaggedOffsetsInput(std::move(queryRaggedOffsetsInput)),
          keyValueRaggedOffsetsInput(std::move(keyValueRaggedOffsetsInput)),
          weightsDataType(weightsDataType),
          computeDataType(computeDataType),
          outputDataType(outputDataType) {}

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
    const ThorImplementation::RotaryPositionEmbeddingOptions& getRopeOptions() const { return ropeOptions; }
    ThorImplementation::AttentionMaskKind getMaskKind() const { return maskKind; }
    int64_t getDiagonalLeftBound() const { return diagonalLeftBound; }
    int64_t getDiagonalRightBound() const { return diagonalRightBound; }
    bool getUseAlibiMask() const { return useAlibiMask; }
    std::optional<double> getAttentionScale() const { return attentionScale; }
    float getDropoutProbability() const { return dropoutProbability; }
    int64_t getDropoutSeed() const { return dropoutSeed; }
    int64_t getDropoutOffset() const { return dropoutOffset; }
    std::optional<Tensor> getFeatureInput() const override { return getInputInterface().at("feature_input"); }
    std::optional<Tensor> getContextInput() const { return contextInput; }
    bool getUseCrossAttention() const { return contextInput.has_value(); }
    std::optional<Tensor> getScoreBiasInput() const { return scoreBiasInput; }
    bool getUseScoreBias() const { return scoreBiasInput.has_value(); }
    std::optional<Tensor> getQuerySequenceLengthsInput() const { return querySequenceLengthsInput; }
    std::optional<Tensor> getKeyValueSequenceLengthsInput() const { return keyValueSequenceLengthsInput; }
    std::optional<Tensor> getQueryRaggedOffsetsInput() const { return queryRaggedOffsetsInput; }
    std::optional<Tensor> getKeyValueRaggedOffsetsInput() const { return keyValueRaggedOffsetsInput; }
    bool getUseSequenceLengths() const { return querySequenceLengthsInput.has_value(); }
    bool getUseRaggedOffsets() const { return queryRaggedOffsetsInput.has_value(); }
    Tensor::DataType getWeightsDataType() const { return weightsDataType; }
    Tensor::DataType getComputeDataType() const { return computeDataType; }
    Tensor::DataType getOutputDataType() const { return outputDataType; }

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
    ThorImplementation::AttentionMaskKind maskKind;
    int64_t diagonalLeftBound;
    int64_t diagonalRightBound;
    bool useAlibiMask;
    std::optional<double> attentionScale;
    float dropoutProbability;
    int64_t dropoutSeed;
    int64_t dropoutOffset;
    std::optional<Tensor> contextInput;
    std::optional<Tensor> scoreBiasInput;
    std::optional<Tensor> querySequenceLengthsInput;
    std::optional<Tensor> keyValueSequenceLengthsInput;
    std::optional<Tensor> queryRaggedOffsetsInput;
    std::optional<Tensor> keyValueRaggedOffsetsInput;
    Tensor::DataType weightsDataType;
    Tensor::DataType computeDataType;
    Tensor::DataType outputDataType;
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
        this->_featureInput = input;
        return *this;
    }

    virtual Attention::Builder& contextInput(Tensor input) {
        THOR_THROW_IF_FALSE(!this->_contextInput.has_value());
        this->_contextInput = input;
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

    virtual Attention::Builder& dropoutProbability(float value) {
        THOR_THROW_IF_FALSE(!this->_dropoutProbability.has_value());
        this->_dropoutProbability = value;
        return *this;
    }

    virtual Attention::Builder& dropoutSeed(int64_t value) {
        THOR_THROW_IF_FALSE(!this->_dropoutSeed.has_value());
        this->_dropoutSeed = value;
        return *this;
    }

    virtual Attention::Builder& dropoutOffset(int64_t value) {
        THOR_THROW_IF_FALSE(!this->_dropoutOffset.has_value());
        this->_dropoutOffset = value;
        return *this;
    }

    virtual Attention::Builder& dropout(float probability, int64_t seed, int64_t offset) {
        return dropoutProbability(probability).dropoutSeed(seed).dropoutOffset(offset);
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

    virtual Attention::Builder& queryRaggedOffsetsInput(Tensor input) {
        THOR_THROW_IF_FALSE(!this->_queryRaggedOffsetsInput.has_value());
        this->_queryRaggedOffsetsInput = input;
        return *this;
    }

    virtual Attention::Builder& keyValueRaggedOffsetsInput(Tensor input) {
        THOR_THROW_IF_FALSE(!this->_keyValueRaggedOffsetsInput.has_value());
        this->_keyValueRaggedOffsetsInput = input;
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

    virtual Attention::Builder& weightsDataType(Tensor::DataType value) {
        THOR_THROW_IF_FALSE(!this->_weightsDataType.has_value());
        this->_weightsDataType = value;
        return *this;
    }

    virtual Attention::Builder& computeDataType(Tensor::DataType value) {
        THOR_THROW_IF_FALSE(!this->_computeDataType.has_value());
        this->_computeDataType = value;
        return *this;
    }

    virtual Attention::Builder& outputDataType(Tensor::DataType value) {
        THOR_THROW_IF_FALSE(!this->_outputDataType.has_value());
        this->_outputDataType = value;
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
    std::optional<float> _dropoutProbability;
    std::optional<int64_t> _dropoutSeed;
    std::optional<int64_t> _dropoutOffset;
    std::optional<Tensor> _querySequenceLengthsInput;
    std::optional<Tensor> _keyValueSequenceLengthsInput;
    std::optional<Tensor> _queryRaggedOffsetsInput;
    std::optional<Tensor> _keyValueRaggedOffsetsInput;
    std::optional<bool> _useRope;
    std::optional<bool> _ropeInPlace;
    std::optional<ThorImplementation::RotaryPositionEmbeddingOptions> _ropeOptions;
    std::optional<Tensor::DataType> _weightsDataType;
    std::optional<Tensor::DataType> _computeDataType;
    std::optional<Tensor::DataType> _outputDataType;
    std::shared_ptr<Initializer> _weightsInitializer;
    std::shared_ptr<Initializer> _biasInitializer;
    std::shared_ptr<Optimizer> _optimizer;
};

}  // namespace Thor
