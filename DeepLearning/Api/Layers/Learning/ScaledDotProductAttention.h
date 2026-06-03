#pragma once

#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Learning/CustomLayer.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/TensorOperations/GpuAttention/CudnnAttention.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace Thor {

// API-level wrapper around the first-class cuDNN-backed expression Attention primitive.
//
// This layer consumes already-projected multi-head tensors.  The API tensor dimensions omit the runtime batch dimension.
// The default API layout is BHSD-without-B, i.e. [heads, sequence, head_dim].  BSHD-without-B, i.e.
// [sequence, heads, head_dim], is also supported so higher-level transformer layers can keep token-major projection
// outputs without an extra transpose.
class ScaledDotProductAttention : public CustomLayer {
   public:
    class Builder;
    friend class Builder;

    ScaledDotProductAttention(ThorImplementation::DynamicExpression expression,
                              std::vector<std::string> inputNames,
                              std::vector<std::string> outputNames,
                              const std::vector<TensorMap>& inputInterfaces,
                              const std::vector<TensorMap>& outputInterfaces,
                              ThorImplementation::AttentionTensorLayout tensorLayout,
                              ThorImplementation::AttentionMaskKind maskKind,
                              int64_t diagonalLeftBound,
                              int64_t diagonalRightBound,
                              bool useAlibiMask,
                              std::optional<double> attentionScale,
                              float dropoutProbability,
                              int64_t dropoutSeed,
                              int64_t dropoutOffset,
                              std::optional<Tensor> querySequenceLengthsInput,
                              std::optional<Tensor> keyValueSequenceLengthsInput,
                              std::optional<Tensor> queryRaggedOffsetsInput,
                              std::optional<Tensor> keyValueRaggedOffsetsInput,
                              std::optional<Tensor> fp8DescaleQInput,
                              std::optional<Tensor> fp8DescaleKInput,
                              std::optional<Tensor> fp8DescaleVInput,
                              std::optional<Tensor> fp8DescaleSInput,
                              std::optional<Tensor> fp8ScaleSInput,
                              std::optional<Tensor> fp8ScaleOInput,
                              std::optional<Tensor> fp8AmaxSInput,
                              std::optional<Tensor> fp8AmaxOInput,
                              DataType computeDataType,
                              DataType outputDataType)
        : CustomLayer(std::move(expression), std::move(inputNames), std::move(outputNames), inputInterfaces, outputInterfaces, {}),
          tensorLayout(tensorLayout),
          maskKind(maskKind),
          diagonalLeftBound(diagonalLeftBound),
          diagonalRightBound(diagonalRightBound),
          useAlibiMask(useAlibiMask),
          attentionScale(attentionScale),
          dropoutProbability(dropoutProbability),
          dropoutSeed(dropoutSeed),
          dropoutOffset(dropoutOffset),
          querySequenceLengthsInput(std::move(querySequenceLengthsInput)),
          keyValueSequenceLengthsInput(std::move(keyValueSequenceLengthsInput)),
          queryRaggedOffsetsInput(std::move(queryRaggedOffsetsInput)),
          keyValueRaggedOffsetsInput(std::move(keyValueRaggedOffsetsInput)),
          fp8DescaleQInput(std::move(fp8DescaleQInput)),
          fp8DescaleKInput(std::move(fp8DescaleKInput)),
          fp8DescaleVInput(std::move(fp8DescaleVInput)),
          fp8DescaleSInput(std::move(fp8DescaleSInput)),
          fp8ScaleSInput(std::move(fp8ScaleSInput)),
          fp8ScaleOInput(std::move(fp8ScaleOInput)),
          fp8AmaxSInput(std::move(fp8AmaxSInput)),
          fp8AmaxOInput(std::move(fp8AmaxOInput)),
          computeDataType(computeDataType),
          outputDataType(outputDataType) {}

    ~ScaledDotProductAttention() override = default;

    std::shared_ptr<Layer> clone() const override { return std::make_shared<ScaledDotProductAttention>(*this); }

    std::string getLayerType() const override { return "ScaledDotProductAttention"; }

    nlohmann::json serialize(thor_file::TarWriter& archiveWriter,
                             Stream stream,
                             bool saveOptimizerState,
                             ThorImplementation::StampedNetwork& stampedNetwork) const override;
    static void deserialize(std::shared_ptr<thor_file::TarReader>& archiveReader, const nlohmann::json& j, Network* network);
    nlohmann::json architectureJson() const override;

    ThorImplementation::AttentionTensorLayout getTensorLayout() const { return tensorLayout; }
    ThorImplementation::AttentionMaskKind getMaskKind() const { return maskKind; }
    int64_t getDiagonalLeftBound() const { return diagonalLeftBound; }
    int64_t getDiagonalRightBound() const { return diagonalRightBound; }
    bool getUseAlibiMask() const { return useAlibiMask; }
    std::optional<double> getAttentionScale() const { return attentionScale; }
    float getDropoutProbability() const { return dropoutProbability; }
    int64_t getDropoutSeed() const { return dropoutSeed; }
    int64_t getDropoutOffset() const { return dropoutOffset; }
    bool getUseSequenceLengths() const { return querySequenceLengthsInput.has_value(); }
    bool getUseRaggedOffsets() const { return queryRaggedOffsetsInput.has_value(); }
    bool getUseBias() const { return getInputInterface().contains("bias"); }
    bool getUseFp8ForwardScaling() const { return fp8DescaleQInput.has_value(); }
    std::optional<Tensor> getBiasInput() const {
        const auto inputInterface = getInputInterface();
        const auto it = inputInterface.find("bias");
        if (it == inputInterface.end()) {
            return std::nullopt;
        }
        return it->second;
    }
    std::optional<Tensor> getQuerySequenceLengthsInput() const { return querySequenceLengthsInput; }
    std::optional<Tensor> getKeyValueSequenceLengthsInput() const { return keyValueSequenceLengthsInput; }
    std::optional<Tensor> getQueryRaggedOffsetsInput() const { return queryRaggedOffsetsInput; }
    std::optional<Tensor> getKeyValueRaggedOffsetsInput() const { return keyValueRaggedOffsetsInput; }
    std::optional<Tensor> getFp8DescaleQInput() const { return fp8DescaleQInput; }
    std::optional<Tensor> getFp8DescaleKInput() const { return fp8DescaleKInput; }
    std::optional<Tensor> getFp8DescaleVInput() const { return fp8DescaleVInput; }
    std::optional<Tensor> getFp8DescaleSInput() const { return fp8DescaleSInput; }
    std::optional<Tensor> getFp8ScaleSInput() const { return fp8ScaleSInput; }
    std::optional<Tensor> getFp8ScaleOInput() const { return fp8ScaleOInput; }
    std::optional<Tensor> getFp8AmaxSInput() const { return fp8AmaxSInput; }
    std::optional<Tensor> getFp8AmaxOInput() const { return fp8AmaxOInput; }
    DataType getComputeDataType() const { return computeDataType; }
    DataType getOutputDataType() const { return outputDataType; }

   private:
    ThorImplementation::AttentionTensorLayout tensorLayout;
    ThorImplementation::AttentionMaskKind maskKind;
    int64_t diagonalLeftBound;
    int64_t diagonalRightBound;
    bool useAlibiMask;
    std::optional<double> attentionScale;
    float dropoutProbability;
    int64_t dropoutSeed;
    int64_t dropoutOffset;
    std::optional<Tensor> querySequenceLengthsInput;
    std::optional<Tensor> keyValueSequenceLengthsInput;
    std::optional<Tensor> queryRaggedOffsetsInput;
    std::optional<Tensor> keyValueRaggedOffsetsInput;
    std::optional<Tensor> fp8DescaleQInput;
    std::optional<Tensor> fp8DescaleKInput;
    std::optional<Tensor> fp8DescaleVInput;
    std::optional<Tensor> fp8DescaleSInput;
    std::optional<Tensor> fp8ScaleSInput;
    std::optional<Tensor> fp8ScaleOInput;
    std::optional<Tensor> fp8AmaxSInput;
    std::optional<Tensor> fp8AmaxOInput;
    DataType computeDataType;
    DataType outputDataType;
};

class ScaledDotProductAttention::Builder {
   public:
    virtual ~Builder() = default;

    virtual ScaledDotProductAttention build();

    virtual ScaledDotProductAttention::Builder& network(Network& network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &network;
        return *this;
    }

    virtual ScaledDotProductAttention::Builder& selfInput(Tensor input) {
        THOR_THROW_IF_FALSE(!this->_queryInput.has_value());
        THOR_THROW_IF_FALSE(!this->_keyInput.has_value());
        THOR_THROW_IF_FALSE(!this->_valueInput.has_value());
        this->_queryInput = input;
        this->_keyInput = input;
        this->_valueInput = input;
        return *this;
    }

    virtual ScaledDotProductAttention::Builder& queryInput(Tensor input) {
        THOR_THROW_IF_FALSE(!this->_queryInput.has_value());
        this->_queryInput = input;
        return *this;
    }

    virtual ScaledDotProductAttention::Builder& keyInput(Tensor input) {
        THOR_THROW_IF_FALSE(!this->_keyInput.has_value());
        this->_keyInput = input;
        return *this;
    }

    virtual ScaledDotProductAttention::Builder& valueInput(Tensor input) {
        THOR_THROW_IF_FALSE(!this->_valueInput.has_value());
        this->_valueInput = input;
        return *this;
    }

    virtual ScaledDotProductAttention::Builder& biasInput(Tensor input) {
        THOR_THROW_IF_FALSE(!this->_biasInput.has_value());
        this->_biasInput = input;
        return *this;
    }

    // Convenience for self-attention: use the same logical sequence lengths for Q and K/V.
    virtual ScaledDotProductAttention::Builder& sequenceLengthsInput(Tensor input) {
        THOR_THROW_IF_FALSE(!this->_querySequenceLengthsInput.has_value());
        THOR_THROW_IF_FALSE(!this->_keyValueSequenceLengthsInput.has_value());
        this->_querySequenceLengthsInput = input;
        this->_keyValueSequenceLengthsInput = input;
        return *this;
    }

    virtual ScaledDotProductAttention::Builder& querySequenceLengthsInput(Tensor input) {
        THOR_THROW_IF_FALSE(!this->_querySequenceLengthsInput.has_value());
        this->_querySequenceLengthsInput = input;
        return *this;
    }

    virtual ScaledDotProductAttention::Builder& keyValueSequenceLengthsInput(Tensor input) {
        THOR_THROW_IF_FALSE(!this->_keyValueSequenceLengthsInput.has_value());
        this->_keyValueSequenceLengthsInput = input;
        return *this;
    }

    // Convenience for self-attention: use the same ragged element offsets for Q/O and K/V.
    virtual ScaledDotProductAttention::Builder& raggedOffsetsInput(Tensor input) {
        THOR_THROW_IF_FALSE(!this->_queryRaggedOffsetsInput.has_value());
        THOR_THROW_IF_FALSE(!this->_keyValueRaggedOffsetsInput.has_value());
        this->_queryRaggedOffsetsInput = input;
        this->_keyValueRaggedOffsetsInput = input;
        return *this;
    }

    virtual ScaledDotProductAttention::Builder& queryRaggedOffsetsInput(Tensor input) {
        THOR_THROW_IF_FALSE(!this->_queryRaggedOffsetsInput.has_value());
        this->_queryRaggedOffsetsInput = input;
        return *this;
    }

    virtual ScaledDotProductAttention::Builder& keyValueRaggedOffsetsInput(Tensor input) {
        THOR_THROW_IF_FALSE(!this->_keyValueRaggedOffsetsInput.has_value());
        this->_keyValueRaggedOffsetsInput = input;
        return *this;
    }


    // Experimental FP8 forward-only SDPA support.  These are explicit cuDNN FP8 scalar tensors with
    // shape [1,1,1,1] and dtype fp32: descale Q/K/V/S, scale S/O, and output amax S/O.
    virtual ScaledDotProductAttention::Builder& fp8ForwardScalingInputs(Tensor descaleQ,
                                                                         Tensor descaleK,
                                                                         Tensor descaleV,
                                                                         Tensor descaleS,
                                                                         Tensor scaleS,
                                                                         Tensor scaleO,
                                                                         Tensor amaxS,
                                                                         Tensor amaxO) {
        THOR_THROW_IF_FALSE(!this->_fp8DescaleQInput.has_value());
        THOR_THROW_IF_FALSE(!this->_fp8DescaleKInput.has_value());
        THOR_THROW_IF_FALSE(!this->_fp8DescaleVInput.has_value());
        THOR_THROW_IF_FALSE(!this->_fp8DescaleSInput.has_value());
        THOR_THROW_IF_FALSE(!this->_fp8ScaleSInput.has_value());
        THOR_THROW_IF_FALSE(!this->_fp8ScaleOInput.has_value());
        THOR_THROW_IF_FALSE(!this->_fp8AmaxSInput.has_value());
        THOR_THROW_IF_FALSE(!this->_fp8AmaxOInput.has_value());
        this->_fp8DescaleQInput = descaleQ;
        this->_fp8DescaleKInput = descaleK;
        this->_fp8DescaleVInput = descaleV;
        this->_fp8DescaleSInput = descaleS;
        this->_fp8ScaleSInput = scaleS;
        this->_fp8ScaleOInput = scaleO;
        this->_fp8AmaxSInput = amaxS;
        this->_fp8AmaxOInput = amaxO;
        return *this;
    }

    virtual ScaledDotProductAttention::Builder& experimentalFp8ForwardScalingInputs(Tensor descaleQ,
                                                                                     Tensor descaleK,
                                                                                     Tensor descaleV,
                                                                                     Tensor descaleS,
                                                                                     Tensor scaleS,
                                                                                     Tensor scaleO,
                                                                                     Tensor amaxS,
                                                                                     Tensor amaxO) {
        return fp8ForwardScalingInputs(descaleQ, descaleK, descaleV, descaleS, scaleS, scaleO, amaxS, amaxO);
    }

    virtual ScaledDotProductAttention::Builder& tensorLayout(ThorImplementation::AttentionTensorLayout value) {
        THOR_THROW_IF_FALSE(!this->_tensorLayout.has_value());
        this->_tensorLayout = value;
        return *this;
    }

    virtual ScaledDotProductAttention::Builder& bhsdLayout() { return tensorLayout(ThorImplementation::AttentionTensorLayout::BHSD); }
    virtual ScaledDotProductAttention::Builder& bshdLayout() { return tensorLayout(ThorImplementation::AttentionTensorLayout::BSHD); }

    virtual ScaledDotProductAttention::Builder& maskKind(ThorImplementation::AttentionMaskKind value) {
        THOR_THROW_IF_FALSE(!this->_maskKind.has_value());
        this->_maskKind = value;
        return *this;
    }

    virtual ScaledDotProductAttention::Builder& causal(bool enabled = true) {
        THOR_THROW_IF_FALSE(!this->_maskKind.has_value());
        this->_maskKind = enabled ? ThorImplementation::AttentionMaskKind::CausalTopLeft : ThorImplementation::AttentionMaskKind::None;
        return *this;
    }

    virtual ScaledDotProductAttention::Builder& diagonalLeftBound(int64_t value) {
        THOR_THROW_IF_FALSE(!this->_diagonalLeftBound.has_value());
        this->_diagonalLeftBound = value;
        return *this;
    }

    virtual ScaledDotProductAttention::Builder& diagonalRightBound(int64_t value) {
        THOR_THROW_IF_FALSE(!this->_diagonalRightBound.has_value());
        this->_diagonalRightBound = value;
        return *this;
    }

    virtual ScaledDotProductAttention::Builder& useAlibiMask(bool value = true) {
        THOR_THROW_IF_FALSE(!this->_useAlibiMask.has_value());
        this->_useAlibiMask = value;
        return *this;
    }

    virtual ScaledDotProductAttention::Builder& attentionScale(double value) {
        THOR_THROW_IF_FALSE(!this->_attentionScale.has_value());
        this->_attentionScale = value;
        return *this;
    }

    virtual ScaledDotProductAttention::Builder& dropoutProbability(float value) {
        THOR_THROW_IF_FALSE(!this->_dropoutProbability.has_value());
        this->_dropoutProbability = value;
        return *this;
    }

    virtual ScaledDotProductAttention::Builder& dropoutSeed(int64_t value) {
        THOR_THROW_IF_FALSE(!this->_dropoutSeed.has_value());
        this->_dropoutSeed = value;
        return *this;
    }

    virtual ScaledDotProductAttention::Builder& dropoutOffset(int64_t value) {
        THOR_THROW_IF_FALSE(!this->_dropoutOffset.has_value());
        this->_dropoutOffset = value;
        return *this;
    }

    virtual ScaledDotProductAttention::Builder& dropout(float probability, int64_t seed, int64_t offset) {
        return dropoutProbability(probability).dropoutSeed(seed).dropoutOffset(offset);
    }

    virtual ScaledDotProductAttention::Builder& computeDataType(DataType value) {
        THOR_THROW_IF_FALSE(!this->_computeDataType.has_value());
        this->_computeDataType = value;
        return *this;
    }

    virtual ScaledDotProductAttention::Builder& outputDataType(DataType value) {
        THOR_THROW_IF_FALSE(!this->_outputDataType.has_value());
        this->_outputDataType = value;
        return *this;
    }

   private:
    void verifyConfig() const;

    std::optional<Network*> _network;
    std::optional<Tensor> _queryInput;
    std::optional<Tensor> _keyInput;
    std::optional<Tensor> _valueInput;
    std::optional<Tensor> _biasInput;
    std::optional<Tensor> _querySequenceLengthsInput;
    std::optional<Tensor> _keyValueSequenceLengthsInput;
    std::optional<Tensor> _queryRaggedOffsetsInput;
    std::optional<Tensor> _keyValueRaggedOffsetsInput;
    std::optional<Tensor> _fp8DescaleQInput;
    std::optional<Tensor> _fp8DescaleKInput;
    std::optional<Tensor> _fp8DescaleVInput;
    std::optional<Tensor> _fp8DescaleSInput;
    std::optional<Tensor> _fp8ScaleSInput;
    std::optional<Tensor> _fp8ScaleOInput;
    std::optional<Tensor> _fp8AmaxSInput;
    std::optional<Tensor> _fp8AmaxOInput;
    std::optional<ThorImplementation::AttentionTensorLayout> _tensorLayout;
    std::optional<ThorImplementation::AttentionMaskKind> _maskKind;
    std::optional<int64_t> _diagonalLeftBound;
    std::optional<int64_t> _diagonalRightBound;
    std::optional<bool> _useAlibiMask;
    std::optional<double> _attentionScale;
    std::optional<float> _dropoutProbability;
    std::optional<int64_t> _dropoutSeed;
    std::optional<int64_t> _dropoutOffset;
    std::optional<DataType> _computeDataType;
    std::optional<DataType> _outputDataType;
};

}  // namespace Thor
