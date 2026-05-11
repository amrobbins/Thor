#include "DeepLearning/Api/Layers/Learning/ScaledDotProductAttention.h"

#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/FusedEquation.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

using DataType = ThorImplementation::TensorDescriptor::DataType;

namespace {

std::string dtypeName(DataType dtype) {
    switch (dtype) {
        case DataType::FP16: return "fp16";
        case DataType::BF16: return "bf16";
        case DataType::FP32: return "fp32";
        case DataType::FP8_E4M3: return "fp8_e4m3";
        case DataType::FP8_E5M2: return "fp8_e5m2";
        default: return "dtype(" + std::to_string(static_cast<int>(dtype)) + ")";
    }
}

bool isAttentionStorageDType(DataType dtype) { return dtype == DataType::FP16 || dtype == DataType::BF16; }
bool isAttentionComputeDType(DataType dtype) { return dtype == DataType::FP32; }

void requireRank3(const Thor::Tensor& tensor, const std::string& what) {
    if (!tensor.isInitialized()) {
        throw std::invalid_argument("ScaledDotProductAttention " + what + " tensor is not initialized.");
    }
    const std::vector<uint64_t> dims = tensor.getDimensions();
    if (dims.size() != 3) {
        throw std::invalid_argument("ScaledDotProductAttention " + what + " tensor must have rank 3 at the API level.");
    }
    for (uint64_t dim : dims) {
        if (dim == 0) {
            throw std::invalid_argument("ScaledDotProductAttention " + what + " tensor dimensions must be non-zero.");
        }
    }
}

struct LogicalAttentionDims {
    uint64_t heads;
    uint64_t sequence;
    uint64_t head_dim;
};

LogicalAttentionDims logicalDims(const std::vector<uint64_t>& dims, ThorImplementation::AttentionTensorLayout layout) {
    if (layout == ThorImplementation::AttentionTensorLayout::BHSD) {
        return {dims.at(0), dims.at(1), dims.at(2)};
    }
    if (layout == ThorImplementation::AttentionTensorLayout::BSHD) {
        return {dims.at(1), dims.at(0), dims.at(2)};
    }
    throw std::invalid_argument("ScaledDotProductAttention API layer only supports BHSD or BSHD tensor layouts.");
}

std::vector<uint64_t> outputDims(uint64_t query_heads,
                                 uint64_t query_sequence,
                                 uint64_t value_dim,
                                 ThorImplementation::AttentionTensorLayout layout) {
    if (layout == ThorImplementation::AttentionTensorLayout::BHSD) {
        return {query_heads, query_sequence, value_dim};
    }
    if (layout == ThorImplementation::AttentionTensorLayout::BSHD) {
        return {query_sequence, query_heads, value_dim};
    }
    throw std::invalid_argument("ScaledDotProductAttention API layer only supports BHSD or BSHD tensor layouts.");
}

ThorImplementation::DynamicExpression makeAttentionExpression(bool useBias,
                                                              ThorImplementation::AttentionTensorLayout tensorLayout,
                                                              ThorImplementation::AttentionMaskKind maskKind,
                                                              int64_t diagonalLeftBound,
                                                              int64_t diagonalRightBound,
                                                              bool useAlibiMask,
                                                              std::optional<double> attentionScale,
                                                              DataType qkvDType,
                                                              DataType biasDType,
                                                              DataType computeDType,
                                                              DataType outputDType) {
    using ThorImplementation::AttentionOptions;
    using ThorImplementation::DynamicExpression;
    using ThorImplementation::Expression;
    using ThorImplementation::ExpressionDefinition;

    AttentionOptions options;
    options.q_layout = tensorLayout;
    options.k_layout = tensorLayout;
    options.v_layout = tensorLayout;
    options.o_layout = tensorLayout;
    options.mask_kind = maskKind;
    options.diagonal_left_bound = diagonalLeftBound;
    options.diagonal_right_bound = diagonalRightBound;
    options.use_alibi_mask = useAlibiMask;
    options.compute_dtype = computeDType;
    options.output_dtype = outputDType;
    if (attentionScale.has_value()) {
        options.attention_scale = static_cast<float>(attentionScale.value());
    }

    Expression q = Expression::input("query", qkvDType, qkvDType);
    Expression k = Expression::input("key", qkvDType, qkvDType);
    Expression v = Expression::input("value", qkvDType, qkvDType);

    Expression out = useBias ? Expression::scaledDotProductAttention(q, k, v, Expression::input("bias", biasDType, biasDType), options)
                             : Expression::scaledDotProductAttention(q, k, v, options);
    out = out.withOutputDType(outputDType);

    ExpressionDefinition definition = ExpressionDefinition::fromOutputs(Expression::outputs({{"output", out}}));
    return DynamicExpression::fromExpressionDefinition(definition);
}

}  // namespace

namespace Thor {

void ScaledDotProductAttention::Builder::verifyConfig() const {
    if (!_network.has_value()) {
        throw std::invalid_argument("ScaledDotProductAttention::Builder requires network().");
    }
    if (!_queryInput.has_value() || !_keyInput.has_value() || !_valueInput.has_value()) {
        throw std::invalid_argument("ScaledDotProductAttention::Builder requires queryInput(), keyInput(), and valueInput(), or selfInput().");
    }

    requireRank3(_queryInput.value(), "query");
    requireRank3(_keyInput.value(), "key");
    requireRank3(_valueInput.value(), "value");

    const auto layout = _tensorLayout.value_or(ThorImplementation::AttentionTensorLayout::BHSD);
    const auto qDims = logicalDims(_queryInput->getDimensions(), layout);
    const auto kDims = logicalDims(_keyInput->getDimensions(), layout);
    const auto vDims = logicalDims(_valueInput->getDimensions(), layout);

    if (_queryInput->getDataType() != _keyInput->getDataType() || _queryInput->getDataType() != _valueInput->getDataType()) {
        throw std::invalid_argument("ScaledDotProductAttention query/key/value tensors must have the same dtype.");
    }
    if (!isAttentionStorageDType(_queryInput->getDataType())) {
        throw std::invalid_argument("ScaledDotProductAttention query/key/value dtype must be fp16 or bf16 for the current cuDNN training path. Got " +
                                    dtypeName(_queryInput->getDataType()) + ".");
    }

    if (qDims.head_dim != kDims.head_dim) {
        throw std::invalid_argument("ScaledDotProductAttention query and key head_dim must match.");
    }
    if (kDims.heads != vDims.heads) {
        throw std::invalid_argument("ScaledDotProductAttention key and value head counts must match.");
    }
    if (kDims.sequence != vDims.sequence) {
        throw std::invalid_argument("ScaledDotProductAttention key and value sequence lengths must match.");
    }
    if (qDims.heads % kDims.heads != 0) {
        throw std::invalid_argument("ScaledDotProductAttention query head count must be a multiple of key/value head count for MHA/GQA/MQA.");
    }

    const DataType computeDType = _computeDataType.value_or(DataType::FP32);
    if (!isAttentionComputeDType(computeDType)) {
        throw std::invalid_argument("ScaledDotProductAttention computeDataType must currently be fp32 for cuDNN SDPA training. Got " +
                                    dtypeName(computeDType) + ".");
    }

    const DataType outputDType = _outputDataType.value_or(_queryInput->getDataType());
    if (!isAttentionStorageDType(outputDType)) {
        throw std::invalid_argument("ScaledDotProductAttention outputDataType must be fp16 or bf16 for the current cuDNN training path. Got " +
                                    dtypeName(outputDType) + ".");
    }

    const auto maskKind = _maskKind.value_or(ThorImplementation::AttentionMaskKind::None);
    const bool useAlibi = _useAlibiMask.value_or(false);
    const int64_t rightBound = _diagonalRightBound.value_or(0);
    if (useAlibi && maskKind != ThorImplementation::AttentionMaskKind::CausalTopLeft &&
        maskKind != ThorImplementation::AttentionMaskKind::CausalBottomRight &&
        maskKind != ThorImplementation::AttentionMaskKind::SlidingWindowTopLeft &&
        maskKind != ThorImplementation::AttentionMaskKind::SlidingWindowBottomRight) {
        throw std::invalid_argument("ScaledDotProductAttention ALiBi requires a causal/sliding-window diagonal mask.");
    }
    if (useAlibi && rightBound != 0) {
        throw std::invalid_argument("ScaledDotProductAttention ALiBi requires diagonalRightBound == 0.");
    }

    if (_attentionScale.has_value() && (!std::isfinite(_attentionScale.value()) || _attentionScale.value() <= 0.0)) {
        throw std::invalid_argument("ScaledDotProductAttention attentionScale must be finite and positive.");
    }

    if (_biasInput.has_value()) {
        requireRank3(_biasInput.value(), "bias");
        const auto biasDims = _biasInput->getDimensions();
        if (biasDims[0] != qDims.heads || biasDims[1] != qDims.sequence || biasDims[2] != kDims.sequence) {
            throw std::invalid_argument("ScaledDotProductAttention additive bias dimensions must be [query_heads, query_sequence, key_sequence].");
        }
        if (_biasInput->getDataType() != computeDType) {
            throw std::invalid_argument(
                "ScaledDotProductAttention additive bias dtype must match attention computeDataType for the current cuDNN wrapper contract.");
        }
        if (maskKind == ThorImplementation::AttentionMaskKind::CausalBottomRight ||
            maskKind == ThorImplementation::AttentionMaskKind::SlidingWindowBottomRight) {
            throw std::invalid_argument(
                "ScaledDotProductAttention bottom-right/decode masks cannot currently be combined with additive bias in cuDNN SDPA.");
        }
    }
}

ScaledDotProductAttention ScaledDotProductAttention::Builder::build() {
    if (!_tensorLayout.has_value()) {
        _tensorLayout = ThorImplementation::AttentionTensorLayout::BHSD;
    }
    if (!_maskKind.has_value()) {
        _maskKind = ThorImplementation::AttentionMaskKind::None;
    }
    if (!_diagonalLeftBound.has_value()) {
        _diagonalLeftBound = 0;
    }
    if (!_diagonalRightBound.has_value()) {
        _diagonalRightBound = 0;
    }
    if (!_useAlibiMask.has_value()) {
        _useAlibiMask = false;
    }
    if (!_computeDataType.has_value()) {
        _computeDataType = DataType::FP32;
    }
    if (!_outputDataType.has_value() && _queryInput.has_value()) {
        _outputDataType = _queryInput->getDataType();
    }

    verifyConfig();

    const auto qDims = logicalDims(_queryInput->getDimensions(), _tensorLayout.value());
    const auto vDims = logicalDims(_valueInput->getDimensions(), _tensorLayout.value());
    Tensor output(_outputDataType.value(), outputDims(qDims.heads, qDims.sequence, vDims.head_dim, _tensorLayout.value()));

    std::vector<std::string> inputNames = {"query", "key", "value"};
    CustomLayer::TensorMap inputInterface = {{"query", _queryInput.value()}, {"key", _keyInput.value()}, {"value", _valueInput.value()}};
    if (_biasInput.has_value()) {
        inputNames.push_back("bias");
        inputInterface.emplace("bias", _biasInput.value());
    }

    ScaledDotProductAttention attention(makeAttentionExpression(_biasInput.has_value(),
                                                _tensorLayout.value(),
                                                _maskKind.value(),
                                                _diagonalLeftBound.value(),
                                                _diagonalRightBound.value(),
                                                _useAlibiMask.value(),
                                                _attentionScale,
                                                _queryInput->getDataType(),
                                                _biasInput.has_value() ? _biasInput->getDataType() : _computeDataType.value(),
                                                _computeDataType.value(),
                                                _outputDataType.value()),
                        inputNames,
                        {"output"},
                        {inputInterface},
                        {{{"output", output}}},
                        _tensorLayout.value(),
                        _maskKind.value(),
                        _diagonalLeftBound.value(),
                        _diagonalRightBound.value(),
                        _useAlibiMask.value(),
                        _attentionScale,
                        _computeDataType.value(),
                        _outputDataType.value());

    attention.addToNetwork(_network.value());
    return attention;
}

}  // namespace Thor
