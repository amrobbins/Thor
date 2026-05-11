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
                              Tensor::DataType computeDataType,
                              Tensor::DataType outputDataType)
        : CustomLayer(std::move(expression), std::move(inputNames), std::move(outputNames), inputInterfaces, outputInterfaces, {}, false),
          tensorLayout(tensorLayout),
          maskKind(maskKind),
          diagonalLeftBound(diagonalLeftBound),
          diagonalRightBound(diagonalRightBound),
          useAlibiMask(useAlibiMask),
          attentionScale(attentionScale),
          computeDataType(computeDataType),
          outputDataType(outputDataType) {}

    ~ScaledDotProductAttention() override = default;

    std::shared_ptr<Layer> clone() const override { return std::make_shared<ScaledDotProductAttention>(*this); }

    std::string getLayerType() const override { return "ScaledDotProductAttention"; }

    ThorImplementation::AttentionTensorLayout getTensorLayout() const { return tensorLayout; }
    ThorImplementation::AttentionMaskKind getMaskKind() const { return maskKind; }
    int64_t getDiagonalLeftBound() const { return diagonalLeftBound; }
    int64_t getDiagonalRightBound() const { return diagonalRightBound; }
    bool getUseAlibiMask() const { return useAlibiMask; }
    std::optional<double> getAttentionScale() const { return attentionScale; }
    Tensor::DataType getComputeDataType() const { return computeDataType; }
    Tensor::DataType getOutputDataType() const { return outputDataType; }

   private:
    ThorImplementation::AttentionTensorLayout tensorLayout;
    ThorImplementation::AttentionMaskKind maskKind;
    int64_t diagonalLeftBound;
    int64_t diagonalRightBound;
    bool useAlibiMask;
    std::optional<double> attentionScale;
    Tensor::DataType computeDataType;
    Tensor::DataType outputDataType;
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

    virtual ScaledDotProductAttention::Builder& computeDataType(Tensor::DataType value) {
        THOR_THROW_IF_FALSE(!this->_computeDataType.has_value());
        this->_computeDataType = value;
        return *this;
    }

    virtual ScaledDotProductAttention::Builder& outputDataType(Tensor::DataType value) {
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
    std::optional<ThorImplementation::AttentionTensorLayout> _tensorLayout;
    std::optional<ThorImplementation::AttentionMaskKind> _maskKind;
    std::optional<int64_t> _diagonalLeftBound;
    std::optional<int64_t> _diagonalRightBound;
    std::optional<bool> _useAlibiMask;
    std::optional<double> _attentionScale;
    std::optional<Tensor::DataType> _computeDataType;
    std::optional<Tensor::DataType> _outputDataType;
};

}  // namespace Thor
