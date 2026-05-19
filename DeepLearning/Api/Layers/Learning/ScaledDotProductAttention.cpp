#include "DeepLearning/Api/Layers/Learning/ScaledDotProductAttention.h"

#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/FusedEquation.h"

#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>

using DataType = ThorImplementation::TensorDescriptor::DataType;

namespace {

constexpr const char* kQueryInputName = "query";
constexpr const char* kKeyInputName = "key";
constexpr const char* kValueInputName = "value";
constexpr const char* kBiasInputName = "bias";
constexpr const char* kQuerySequenceLengthsInputName = "query_sequence_lengths";
constexpr const char* kKeyValueSequenceLengthsInputName = "key_value_sequence_lengths";
constexpr const char* kQueryRaggedOffsetsInputName = "query_ragged_offsets";
constexpr const char* kKeyValueRaggedOffsetsInputName = "key_value_ragged_offsets";
constexpr const char* kOutputName = "output";

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

void requireSequenceLengthsInput(const Thor::Tensor& tensor, const std::string& what) {
    if (!tensor.isInitialized()) {
        throw std::invalid_argument("ScaledDotProductAttention " + what + " tensor is not initialized.");
    }
    if (tensor.getDataType() != DataType::INT32) {
        throw std::invalid_argument("ScaledDotProductAttention " + what + " must have dtype int32.");
    }
    if (tensor.getDimensions() != std::vector<uint64_t>{1}) {
        throw std::invalid_argument("ScaledDotProductAttention " + what + " must have logical shape [1].");
    }
}

void requireRaggedOffsetsInput(const Thor::Tensor& tensor, const std::string& what) {
    if (!tensor.isInitialized()) {
        throw std::invalid_argument("ScaledDotProductAttention " + what + " tensor is not initialized.");
    }
    if (tensor.getDataType() != DataType::INT32) {
        throw std::invalid_argument("ScaledDotProductAttention " + what + " must have dtype int32.");
    }
    if (tensor.getDimensions() != std::vector<uint64_t>{2}) {
        throw std::invalid_argument("ScaledDotProductAttention " + what + " must have logical shape [2].");
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

bool isAllowedApiBiasDims(const std::vector<uint64_t>& dims, const LogicalAttentionDims& qDims, const LogicalAttentionDims& kDims) {
    // API tensors exclude the runtime batch dimension, so cuDNN's score-bias broadcast surface
    // [1|B,1|Hq,Sq,Skv] is represented here as [1|Hq,Sq,Skv].
    return dims.size() == 3 && (dims[0] == 1 || dims[0] == qDims.heads) && dims[1] == qDims.sequence && dims[2] == kDims.sequence;
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

std::vector<std::string> attentionInputNames(bool useBias, bool useSequenceLengths, bool useRaggedOffsets) {
    std::vector<std::string> names = {kQueryInputName, kKeyInputName, kValueInputName};
    if (useBias) {
        names.push_back(kBiasInputName);
    }
    if (useSequenceLengths) {
        names.push_back(kQuerySequenceLengthsInputName);
        names.push_back(kKeyValueSequenceLengthsInputName);
    }
    if (useRaggedOffsets) {
        names.push_back(kQueryRaggedOffsetsInputName);
        names.push_back(kKeyValueRaggedOffsetsInputName);
    }
    return names;
}

Thor::CustomLayer::TensorMap attentionInputInterface(const Thor::Tensor& queryInput,
                                                     const Thor::Tensor& keyInput,
                                                     const Thor::Tensor& valueInput,
                                                     const std::optional<Thor::Tensor>& biasInput,
                                                     const std::optional<Thor::Tensor>& querySequenceLengthsInput,
                                                     const std::optional<Thor::Tensor>& keyValueSequenceLengthsInput,
                                                     const std::optional<Thor::Tensor>& queryRaggedOffsetsInput,
                                                     const std::optional<Thor::Tensor>& keyValueRaggedOffsetsInput) {
    Thor::CustomLayer::TensorMap interface{{kQueryInputName, queryInput}, {kKeyInputName, keyInput}, {kValueInputName, valueInput}};
    if (biasInput.has_value()) {
        interface.emplace(kBiasInputName, biasInput.value());
    }
    if (querySequenceLengthsInput.has_value()) {
        interface.emplace(kQuerySequenceLengthsInputName, querySequenceLengthsInput.value());
        interface.emplace(kKeyValueSequenceLengthsInputName, keyValueSequenceLengthsInput.value());
    }
    if (queryRaggedOffsetsInput.has_value()) {
        interface.emplace(kQueryRaggedOffsetsInputName, queryRaggedOffsetsInput.value());
        interface.emplace(kKeyValueRaggedOffsetsInputName, keyValueRaggedOffsetsInput.value());
    }
    return interface;
}

ThorImplementation::Expression makeAttentionOutputExpression(bool useBias,
                                                              bool useSequenceLengths,
                                                              bool useRaggedOffsets,
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
    using ThorImplementation::Expression;

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

    Expression q = Expression::input(kQueryInputName, qkvDType, qkvDType);
    Expression k = Expression::input(kKeyInputName, qkvDType, qkvDType);
    Expression v = Expression::input(kValueInputName, qkvDType, qkvDType);
    std::optional<Expression> bias;
    if (useBias) {
        bias = Expression::input(kBiasInputName, biasDType, biasDType);
    }

    Expression out = [&]() {
        if (useRaggedOffsets) {
            Expression qSeq = Expression::input(kQuerySequenceLengthsInputName, DataType::INT32, DataType::INT32);
            Expression kvSeq = Expression::input(kKeyValueSequenceLengthsInputName, DataType::INT32, DataType::INT32);
            Expression qOffsets = Expression::input(kQueryRaggedOffsetsInputName, DataType::INT32, DataType::INT32);
            Expression kvOffsets = Expression::input(kKeyValueRaggedOffsetsInputName, DataType::INT32, DataType::INT32);
            if (bias.has_value()) {
                return Expression::scaledDotProductAttentionRagged(q, k, v, bias.value(), qSeq, kvSeq, qOffsets, kvOffsets, options);
            }
            return Expression::scaledDotProductAttentionRagged(q, k, v, qSeq, kvSeq, qOffsets, kvOffsets, options);
        }
        if (useSequenceLengths) {
            Expression qSeq = Expression::input(kQuerySequenceLengthsInputName, DataType::INT32, DataType::INT32);
            Expression kvSeq = Expression::input(kKeyValueSequenceLengthsInputName, DataType::INT32, DataType::INT32);
            if (bias.has_value()) {
                return Expression::scaledDotProductAttention(q, k, v, bias.value(), qSeq, kvSeq, options);
            }
            return Expression::scaledDotProductAttention(q, k, v, qSeq, kvSeq, options);
        }
        if (bias.has_value()) {
            return Expression::scaledDotProductAttention(q, k, v, bias.value(), options);
        }
        return Expression::scaledDotProductAttention(q, k, v, options);
    }();
    return out.withOutputDType(outputDType);
}

std::shared_ptr<const ThorImplementation::ExpressionDefinition> makeSerializableAttentionDefinition(bool useBias,
                                                                                                     bool useSequenceLengths,
                                                                                                     bool useRaggedOffsets,
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
    using ThorImplementation::Expression;
    using ThorImplementation::ExpressionDefinition;

    Expression out = makeAttentionOutputExpression(useBias,
                                                   useSequenceLengths,
                                                   useRaggedOffsets,
                                                   tensorLayout,
                                                   maskKind,
                                                   diagonalLeftBound,
                                                   diagonalRightBound,
                                                   useAlibiMask,
                                                   attentionScale,
                                                   qkvDType,
                                                   biasDType,
                                                   computeDType,
                                                   outputDType);
    return std::make_shared<ExpressionDefinition>(ExpressionDefinition::fromOutputs(Expression::outputs({{kOutputName, out}})));
}

ThorImplementation::DynamicExpression makeAttentionExpression(bool useBias,
                                                              bool useSequenceLengths,
                                                              bool useRaggedOffsets,
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
    using ThorImplementation::DynamicExpression;
    using ThorImplementation::DynamicExpressionBuild;
    using ThorImplementation::FusedEquation;
    using ThorImplementation::Tensor;

    auto serializedDefinition = makeSerializableAttentionDefinition(useBias,
                                                                    useSequenceLengths,
                                                                    useRaggedOffsets,
                                                                    tensorLayout,
                                                                    maskKind,
                                                                    diagonalLeftBound,
                                                                    diagonalRightBound,
                                                                    useAlibiMask,
                                                                    attentionScale,
                                                                    qkvDType,
                                                                    biasDType,
                                                                    computeDType,
                                                                    outputDType);
    std::vector<std::string> expectedInputs = attentionInputNames(useBias, useSequenceLengths, useRaggedOffsets);

    return DynamicExpression(
        expectedInputs,
        {kOutputName},
        [serializedDefinition, useSequenceLengths, useRaggedOffsets](const DynamicExpression::TensorMap& inputs,
                                                                     const DynamicExpression::TensorMap& outputs,
                                                                     Stream& stream) -> DynamicExpressionBuild {
            const Tensor& query = inputs.at(kQueryInputName);
            const auto queryDims = query.getDimensions();
            if (queryDims.size() != 4 || queryDims[0] == 0) {
                throw std::runtime_error("ScaledDotProductAttention runtime query tensor must include batch and have rank 4.");
            }
            const uint64_t batch = queryDims[0];

            DynamicExpression::TensorMap stampInputs = inputs;
            auto normalizeSequenceLengths = [&](const char* name) {
                Tensor seq = inputs.at(name);
                if (seq.getDataType() != DataType::INT32) {
                    throw std::runtime_error(std::string("ScaledDotProductAttention ") + name + " dtype must be INT32.");
                }
                const auto seqDims = seq.getDimensions();
                if (seqDims == std::vector<uint64_t>{batch, 1}) {
                    seq.reshape({batch});
                } else if (seqDims != std::vector<uint64_t>{batch}) {
                    throw std::runtime_error(std::string("ScaledDotProductAttention ") + name + " shape must be [batch] or [batch, 1].");
                }
                stampInputs[name] = std::move(seq);
            };
            auto normalizeRaggedOffsets = [&](const char* name) {
                Tensor offsets = inputs.at(name);
                if (offsets.getDataType() != DataType::INT32) {
                    throw std::runtime_error(std::string("ScaledDotProductAttention ") + name + " dtype must be INT32.");
                }
                if (offsets.getTotalNumElements() < batch + 1) {
                    throw std::runtime_error(std::string("ScaledDotProductAttention ") + name + " must contain at least batch+1 elements.");
                }
                stampInputs[name] = offsets.aliasView({batch + 1}, {1}, 0);
            };

            if (useSequenceLengths) {
                normalizeSequenceLengths(kQuerySequenceLengthsInputName);
                normalizeSequenceLengths(kKeyValueSequenceLengthsInputName);
            }
            if (useRaggedOffsets) {
                normalizeRaggedOffsets(kQueryRaggedOffsetsInputName);
                normalizeRaggedOffsets(kKeyValueRaggedOffsetsInputName);
            }

            return DynamicExpressionBuild{
                std::make_shared<FusedEquation>(FusedEquation::compile(serializedDefinition->outputs, stream.getGpuNum())),
                std::move(stampInputs),
                {},
                outputs,
                {},
            };
        },
        serializedDefinition);
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

    const bool useSequenceLengths = _querySequenceLengthsInput.has_value() || _keyValueSequenceLengthsInput.has_value();
    const bool useRaggedOffsets = _queryRaggedOffsetsInput.has_value() || _keyValueRaggedOffsetsInput.has_value();
    if (_querySequenceLengthsInput.has_value() != _keyValueSequenceLengthsInput.has_value()) {
        throw std::invalid_argument(
            "ScaledDotProductAttention requires both querySequenceLengthsInput and keyValueSequenceLengthsInput, or sequenceLengthsInput().");
    }
    if (_queryRaggedOffsetsInput.has_value() != _keyValueRaggedOffsetsInput.has_value()) {
        throw std::invalid_argument(
            "ScaledDotProductAttention requires both queryRaggedOffsetsInput and keyValueRaggedOffsetsInput, or raggedOffsetsInput().");
    }
    if (useSequenceLengths) {
        requireSequenceLengthsInput(_querySequenceLengthsInput.value(), "querySequenceLengthsInput");
        requireSequenceLengthsInput(_keyValueSequenceLengthsInput.value(), "keyValueSequenceLengthsInput");
    }
    if (useRaggedOffsets) {
        requireRaggedOffsetsInput(_queryRaggedOffsetsInput.value(), "queryRaggedOffsetsInput");
        requireRaggedOffsetsInput(_keyValueRaggedOffsetsInput.value(), "keyValueRaggedOffsetsInput");
        if (!useSequenceLengths) {
            throw std::invalid_argument("ScaledDotProductAttention raggedOffsetsInput requires sequenceLengthsInput.");
        }
        if (qDims.head_dim != vDims.head_dim) {
            throw std::invalid_argument(
                "ScaledDotProductAttention ragged offsets currently require value head_dim to match query/key head_dim because shared Q/O and K/V element offsets are used.");
        }
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
    if (useAlibi && (maskKind == ThorImplementation::AttentionMaskKind::CausalBottomRight ||
                     maskKind == ThorImplementation::AttentionMaskKind::SlidingWindowBottomRight)) {
        throw std::invalid_argument(
            "ScaledDotProductAttention ALiBi cannot currently be combined with bottom-right/decode masks in cuDNN SDPA.");
    }

    if (_attentionScale.has_value() && (!std::isfinite(_attentionScale.value()) || _attentionScale.value() <= 0.0)) {
        throw std::invalid_argument("ScaledDotProductAttention attentionScale must be finite and positive.");
    }

    if (_biasInput.has_value()) {
        requireRank3(_biasInput.value(), "bias");
        const auto biasDims = _biasInput->getDimensions();
        if (!isAllowedApiBiasDims(biasDims, qDims, kDims)) {
            throw std::invalid_argument(
                "ScaledDotProductAttention additive bias dimensions must be [1|query_heads, query_sequence, key_sequence] at the API level.");
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

    const bool useSequenceLengths = _querySequenceLengthsInput.has_value();
    const bool useRaggedOffsets = _queryRaggedOffsetsInput.has_value();
    const auto qDims = logicalDims(_queryInput->getDimensions(), _tensorLayout.value());
    const auto vDims = logicalDims(_valueInput->getDimensions(), _tensorLayout.value());
    Tensor output(_outputDataType.value(), outputDims(qDims.heads, qDims.sequence, vDims.head_dim, _tensorLayout.value()));

    std::vector<std::string> inputNames = attentionInputNames(_biasInput.has_value(), useSequenceLengths, useRaggedOffsets);
    CustomLayer::TensorMap inputInterface = attentionInputInterface(_queryInput.value(),
                                                                    _keyInput.value(),
                                                                    _valueInput.value(),
                                                                    _biasInput,
                                                                    _querySequenceLengthsInput,
                                                                    _keyValueSequenceLengthsInput,
                                                                    _queryRaggedOffsetsInput,
                                                                    _keyValueRaggedOffsetsInput);

    ScaledDotProductAttention attention(makeAttentionExpression(_biasInput.has_value(),
                                                                useSequenceLengths,
                                                                useRaggedOffsets,
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
                                                {kOutputName},
                                                {inputInterface},
                                                {{{kOutputName, output}}},
                                                _tensorLayout.value(),
                                                _maskKind.value(),
                                                _diagonalLeftBound.value(),
                                                _diagonalRightBound.value(),
                                                _useAlibiMask.value(),
                                                _attentionScale,
                                                _querySequenceLengthsInput,
                                                _keyValueSequenceLengthsInput,
                                                _queryRaggedOffsetsInput,
                                                _keyValueRaggedOffsetsInput,
                                                _computeDataType.value(),
                                                _outputDataType.value());

    attention.addToNetwork(_network.value());
    return attention;
}

}  // namespace Thor
