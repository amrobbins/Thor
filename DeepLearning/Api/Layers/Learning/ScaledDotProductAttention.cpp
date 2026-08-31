#include "DeepLearning/Api/Layers/Learning/ScaledDotProductAttention.h"
#include "Utilities/TensorOperations/Ragged/RowPartitionDTypePolicy.h"

#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/FusedEquation.h"
#include "Utilities/TensorOperations/Scalar/SetScalar.h"

#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>

using DataType = ThorImplementation::DataType;
using json = nlohmann::json;

namespace {

constexpr const char* kQueryInputName = "query";
constexpr const char* kKeyInputName = "key";
constexpr const char* kValueInputName = "value";
constexpr const char* kBiasInputName = "bias";
constexpr const char* kQuerySequenceLengthsInputName = "query_sequence_lengths";
constexpr const char* kKeyValueSequenceLengthsInputName = "key_value_sequence_lengths";
constexpr const char* kQueryRaggedOffsetsInputName = "query_ragged_offsets";
constexpr const char* kKeyValueRaggedOffsetsInputName = "key_value_ragged_offsets";
constexpr const char* kSyntheticQueryRaggedOffsetsInputName = "__sdpa_uniform_query_ragged_offsets";
constexpr const char* kSyntheticKeyValueRaggedOffsetsInputName = "__sdpa_uniform_key_value_ragged_offsets";
constexpr const char* kDropoutSeedInputName = "__sdpa_dropout_seed";
constexpr const char* kDropoutOffsetInputName = "__sdpa_dropout_offset";
constexpr const char* kFp8DescaleQInputName = "fp8_descale_q";
constexpr const char* kFp8DescaleKInputName = "fp8_descale_k";
constexpr const char* kFp8DescaleVInputName = "fp8_descale_v";
constexpr const char* kFp8DescaleSInputName = "fp8_descale_s";
constexpr const char* kFp8ScaleSInputName = "fp8_scale_s";
constexpr const char* kFp8ScaleOInputName = "fp8_scale_o";
constexpr const char* kFp8AmaxSInputName = "fp8_amax_s";
constexpr const char* kFp8AmaxOInputName = "fp8_amax_o";
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
bool isFp8DType(DataType dtype) { return dtype == DataType::FP8_E4M3 || dtype == DataType::FP8_E5M2; }
bool isAttentionComputeDType(DataType dtype) { return dtype == DataType::FP32; }

uint64_t checkedMul(uint64_t a, uint64_t b, const char* what) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
        throw std::invalid_argument(std::string("ScaledDotProductAttention ") + what + " overflows uint64_t.");
    }
    return a * b;
}

uint64_t checkedDropoutOffsetAdvance(uint64_t batch, uint64_t queryHeads, uint64_t querySequence, uint64_t keyValueSequence) {
    return checkedMul(checkedMul(batch, queryHeads, "dropout offset batch-head count"),
                      checkedMul(querySequence, keyValueSequence, "dropout offset score count"),
                      "dropout offset advance");
}

class SdpaDropoutRuntimeState {
   public:
    SdpaDropoutRuntimeState(int64_t seed, int64_t initialOffset) : seed(seed), nextOffset(initialOffset) {
        if (initialOffset < 0) {
            throw std::invalid_argument("ScaledDotProductAttention dropoutOffset must be non-negative when dropout is enabled.");
        }
    }

    void setOffsetAdvance(uint64_t advance) {
        if (advance == 0) {
            advance = 1;
        }
        if (advance > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            throw std::overflow_error("ScaledDotProductAttention dropout offset advance exceeds int64_t range.");
        }
        offsetAdvance = advance;
    }

    ThorImplementation::TensorScalarBinding seedBinding(ThorImplementation::TensorPlacement placement) {
        ensureBuffer(placement);
        return ThorImplementation::TensorScalarBinding{seedOffsetBuffer, kSeedByteOffset, DataType::INT64};
    }

    ThorImplementation::TensorScalarBinding offsetBinding(ThorImplementation::TensorPlacement placement) {
        ensureBuffer(placement);
        return ThorImplementation::TensorScalarBinding{seedOffsetBuffer, kOffsetByteOffset, DataType::INT64};
    }

    void uploadForForward(Stream& stream) {
        ThorImplementation::TensorPlacement placement(ThorImplementation::TensorPlacement::MemDevices::GPU, stream.getGpuNum());
        ensureBuffer(placement);

        ThorImplementation::launchSetInt64Pair(seedOffsetBuffer.getMemPtr<int64_t>(), seed, nextOffset, stream);

        const uint64_t remaining = static_cast<uint64_t>(std::numeric_limits<int64_t>::max() - nextOffset);
        if (offsetAdvance > remaining) {
            throw std::overflow_error("ScaledDotProductAttention automatic dropout offset advance would exceed int64_t range.");
        }
        nextOffset += static_cast<int64_t>(offsetAdvance);
    }

   private:
    static constexpr uint64_t kSeedByteOffset = 0;
    static constexpr uint64_t kOffsetByteOffset = sizeof(int64_t);

    void ensureBuffer(ThorImplementation::TensorPlacement placement) {
        if (seedOffsetBuffer.isInitialized() && seedOffsetBuffer.getPlacement() == placement) {
            return;
        }

        ThorImplementation::TensorDescriptor descriptor(DataType::INT64, {2});
        seedOffsetBuffer = ThorImplementation::Tensor(placement, descriptor);
    }

    int64_t seed;
    int64_t nextOffset;
    uint64_t offsetAdvance = 1;
    ThorImplementation::Tensor seedOffsetBuffer;
};

std::string attentionTensorLayoutToString(ThorImplementation::AttentionTensorLayout value) {
    switch (value) {
        case ThorImplementation::AttentionTensorLayout::BHSD:
            return "bhsd";
        case ThorImplementation::AttentionTensorLayout::BSHD:
            return "bshd";
        default:
            throw std::runtime_error("Unknown ScaledDotProductAttention tensor layout value.");
    }
}

ThorImplementation::AttentionTensorLayout attentionTensorLayoutFromString(const std::string& value) {
    if (value == "bhsd")
        return ThorImplementation::AttentionTensorLayout::BHSD;
    if (value == "bshd")
        return ThorImplementation::AttentionTensorLayout::BSHD;
    throw std::runtime_error("Unknown ScaledDotProductAttention tensor layout: " + value);
}

std::string attentionMaskKindToString(ThorImplementation::AttentionMaskKind value) {
    switch (value) {
        case ThorImplementation::AttentionMaskKind::None:
            return "none";
        case ThorImplementation::AttentionMaskKind::CausalTopLeft:
            return "causal_top_left";
        case ThorImplementation::AttentionMaskKind::CausalBottomRight:
            return "causal_bottom_right";
        case ThorImplementation::AttentionMaskKind::SlidingWindowTopLeft:
            return "sliding_window_top_left";
        case ThorImplementation::AttentionMaskKind::SlidingWindowBottomRight:
            return "sliding_window_bottom_right";
    }
    throw std::runtime_error("Unknown ScaledDotProductAttention mask kind value.");
}

ThorImplementation::AttentionMaskKind attentionMaskKindFromString(const std::string& value) {
    if (value == "none")
        return ThorImplementation::AttentionMaskKind::None;
    if (value == "causal_top_left")
        return ThorImplementation::AttentionMaskKind::CausalTopLeft;
    if (value == "causal_bottom_right")
        return ThorImplementation::AttentionMaskKind::CausalBottomRight;
    if (value == "sliding_window_top_left")
        return ThorImplementation::AttentionMaskKind::SlidingWindowTopLeft;
    if (value == "sliding_window_bottom_right")
        return ThorImplementation::AttentionMaskKind::SlidingWindowBottomRight;
    throw std::runtime_error("Unknown ScaledDotProductAttention mask kind: " + value);
}

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

void requireRaggedOffsetsInput(const Thor::Tensor& tensor, const std::string& what, uint64_t batchSize) {
    if (!tensor.isInitialized()) {
        throw std::invalid_argument("ScaledDotProductAttention " + what + " tensor is not initialized.");
    }
    if (!ThorImplementation::isCanonicalRowPartitionOffsetDataType(tensor.getDataType())) {
        throw std::invalid_argument("ScaledDotProductAttention " + what + " must have canonical row-partition dtype uint32 or uint64.");
    }
    if (batchSize == std::numeric_limits<uint64_t>::max() || tensor.getDimensions() != std::vector<uint64_t>{batchSize + 1}) {
        throw std::invalid_argument("ScaledDotProductAttention " + what + " must have canonical shape [batch_size + 1].");
    }
}

void requireFp8ScaleInput(const Thor::Tensor& tensor, const std::string& what) {
    if (!tensor.isInitialized()) {
        throw std::invalid_argument("ScaledDotProductAttention " + what + " tensor is not initialized.");
    }
    if (tensor.getDataType() != DataType::FP32) {
        throw std::invalid_argument("ScaledDotProductAttention " + what + " must have dtype fp32.");
    }
    if (tensor.getDimensions() != std::vector<uint64_t>{1, 1, 1, 1}) {
        throw std::invalid_argument("ScaledDotProductAttention " + what + " must have shape [1,1,1,1].");
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

LogicalAttentionDims runtimeLogicalDims(const std::vector<uint64_t>& dims, ThorImplementation::AttentionTensorLayout layout) {
    if (dims.size() == 3) {
        if (layout != ThorImplementation::AttentionTensorLayout::BSHD) {
            throw std::runtime_error("Canonical packed ragged ScaledDotProductAttention runtime tensors require BSHD/token-major layout.");
        }
        return {dims.at(1), dims.at(0), dims.at(2)};
    }
    if (dims.size() != 4) {
        throw std::runtime_error("ScaledDotProductAttention runtime tensors must use dense rank-4 or packed ragged rank-3 storage.");
    }
    if (layout == ThorImplementation::AttentionTensorLayout::BHSD) {
        return {dims.at(1), dims.at(2), dims.at(3)};
    }
    if (layout == ThorImplementation::AttentionTensorLayout::BSHD) {
        return {dims.at(2), dims.at(1), dims.at(3)};
    }
    throw std::runtime_error("ScaledDotProductAttention runtime only supports BHSD or BSHD tensor layouts.");
}

bool isAllowedApiBiasDims(const std::vector<uint64_t>& dims, const LogicalAttentionDims& qDims, const LogicalAttentionDims& kDims) {
    // API tensors exclude the runtime batch dimension, so cuDNN's score-bias broadcast surface
    // [1|B,1|Hq,1|Sq,1|Skv] is represented here as [1|Hq,1|Sq,1|Skv].
    return dims.size() == 3 && (dims[0] == 1 || dims[0] == qDims.heads) &&
           (dims[1] == qDims.sequence || dims[1] == 1) && (dims[2] == kDims.sequence || dims[2] == 1);
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

ThorImplementation::Tensor makeUniformAttentionRowPartition(ThorImplementation::TensorPlacement placement,
                                                               DataType dtype,
                                                               uint64_t batch,
                                                               uint64_t sequenceLength,
                                                               Stream& stream) {
    if (!ThorImplementation::isCanonicalRowPartitionOffsetDataType(dtype)) {
        throw std::invalid_argument("ScaledDotProductAttention uniform row partition dtype must be UINT32 or UINT64.");
    }
    if (batch == 0 || sequenceLength == 0) {
        throw std::invalid_argument("ScaledDotProductAttention uniform row partition requires non-zero batch and sequence length.");
    }
    const uint64_t total = checkedMul(batch, sequenceLength, "uniform row partition capacity");
    if (dtype == DataType::UINT32 && total > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::overflow_error("ScaledDotProductAttention uniform UINT32 row partition exceeds representable capacity.");
    }

    ThorImplementation::Tensor device(
        placement, ThorImplementation::TensorDescriptor(dtype, {batch + 1}));
    ThorImplementation::Tensor host(
        ThorImplementation::TensorPlacement(ThorImplementation::TensorPlacement::MemDevices::CPU),
        ThorImplementation::TensorDescriptor(dtype, {batch + 1}));
    if (dtype == DataType::UINT32) {
        auto* values = static_cast<uint32_t*>(host.getMemPtr());
        for (uint64_t i = 0; i <= batch; ++i) values[i] = static_cast<uint32_t>(i * sequenceLength);
    } else {
        auto* values = static_cast<uint64_t*>(host.getMemPtr());
        for (uint64_t i = 0; i <= batch; ++i) values[i] = i * sequenceLength;
    }
    device.copyFromAsync(host, stream);
    // This immutable synthetic partition is created only once while stamping. Complete the
    // tiny host-to-device copy before the temporary host tensor is released.
    stream.synchronize();
    return device;
}

std::vector<std::string> attentionInputNames(bool useBias,
                                                   bool useSequenceLengths,
                                                   bool queryRagged,
                                                   bool keyValueRagged,
                                                   bool useFp8ForwardScaling) {
    std::vector<std::string> names = {kQueryInputName, kKeyInputName, kValueInputName};
    if (useBias) {
        names.push_back(kBiasInputName);
    }
    if (useSequenceLengths) {
        names.push_back(kQuerySequenceLengthsInputName);
        names.push_back(kKeyValueSequenceLengthsInputName);
    }
    if (queryRagged) {
        names.push_back(kQueryRaggedOffsetsInputName);
    }
    if (keyValueRagged) {
        names.push_back(kKeyValueRaggedOffsetsInputName);
    }
    if (useFp8ForwardScaling) {
        names.push_back(kFp8DescaleQInputName);
        names.push_back(kFp8DescaleKInputName);
        names.push_back(kFp8DescaleVInputName);
        names.push_back(kFp8DescaleSInputName);
        names.push_back(kFp8ScaleSInputName);
        names.push_back(kFp8ScaleOInputName);
        names.push_back(kFp8AmaxSInputName);
        names.push_back(kFp8AmaxOInputName);
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
                                                     const std::optional<Thor::Tensor>& keyValueRaggedOffsetsInput,
                                                     const std::optional<Thor::Tensor>& fp8DescaleQInput,
                                                     const std::optional<Thor::Tensor>& fp8DescaleKInput,
                                                     const std::optional<Thor::Tensor>& fp8DescaleVInput,
                                                     const std::optional<Thor::Tensor>& fp8DescaleSInput,
                                                     const std::optional<Thor::Tensor>& fp8ScaleSInput,
                                                     const std::optional<Thor::Tensor>& fp8ScaleOInput,
                                                     const std::optional<Thor::Tensor>& fp8AmaxSInput,
                                                     const std::optional<Thor::Tensor>& fp8AmaxOInput) {
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
    }
    if (keyValueRaggedOffsetsInput.has_value()) {
        interface.emplace(kKeyValueRaggedOffsetsInputName, keyValueRaggedOffsetsInput.value());
    }
    if (fp8DescaleQInput.has_value()) {
        interface.emplace(kFp8DescaleQInputName, fp8DescaleQInput.value());
        interface.emplace(kFp8DescaleKInputName, fp8DescaleKInput.value());
        interface.emplace(kFp8DescaleVInputName, fp8DescaleVInput.value());
        interface.emplace(kFp8DescaleSInputName, fp8DescaleSInput.value());
        interface.emplace(kFp8ScaleSInputName, fp8ScaleSInput.value());
        interface.emplace(kFp8ScaleOInputName, fp8ScaleOInput.value());
        interface.emplace(kFp8AmaxSInputName, fp8AmaxSInput.value());
        interface.emplace(kFp8AmaxOInputName, fp8AmaxOInput.value());
    }
    return interface;
}

ThorImplementation::Expression makeAttentionOutputExpression(bool useBias,
                                                              bool useSequenceLengths,
                                                              bool queryRagged,
                                                              bool keyValueRagged,
                                                              bool useFp8ForwardScaling,
                                                              ThorImplementation::AttentionTensorLayout tensorLayout,
                                                              ThorImplementation::AttentionMaskKind maskKind,
                                                              int64_t diagonalLeftBound,
                                                              int64_t diagonalRightBound,
                                                              bool useAlibiMask,
                                                              std::optional<double> attentionScale,
                                                              float dropoutProbability,
                                                              DataType qkvDType,
                                                              DataType biasDType,
                                                              DataType queryRaggedOffsetDType,
                                                              DataType keyValueRaggedOffsetDType,
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
    options.dropout_probability = dropoutProbability;
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
    std::optional<Expression> descaleQ;
    std::optional<Expression> descaleK;
    std::optional<Expression> descaleV;
    std::optional<Expression> descaleS;
    std::optional<Expression> scaleS;
    std::optional<Expression> scaleO;
    std::optional<Expression> amaxS;
    std::optional<Expression> amaxO;
    if (useFp8ForwardScaling) {
        descaleQ = Expression::input(kFp8DescaleQInputName, DataType::FP32, DataType::FP32);
        descaleK = Expression::input(kFp8DescaleKInputName, DataType::FP32, DataType::FP32);
        descaleV = Expression::input(kFp8DescaleVInputName, DataType::FP32, DataType::FP32);
        descaleS = Expression::input(kFp8DescaleSInputName, DataType::FP32, DataType::FP32);
        scaleS = Expression::input(kFp8ScaleSInputName, DataType::FP32, DataType::FP32);
        scaleO = Expression::input(kFp8ScaleOInputName, DataType::FP32, DataType::FP32);
        amaxS = Expression::input(kFp8AmaxSInputName, DataType::FP32, DataType::FP32);
        amaxO = Expression::input(kFp8AmaxOInputName, DataType::FP32, DataType::FP32);
    }

    Expression out = [&]() {
        if (useFp8ForwardScaling) {
            if (useSequenceLengths) {
                Expression qSeq = Expression::input(kQuerySequenceLengthsInputName, DataType::INT32, DataType::INT32);
                Expression kvSeq = Expression::input(kKeyValueSequenceLengthsInputName, DataType::INT32, DataType::INT32);
                return Expression::scaledDotProductAttentionFp8Forward(q,
                                                                        k,
                                                                        v,
                                                                        qSeq,
                                                                        kvSeq,
                                                                        descaleQ.value(),
                                                                        descaleK.value(),
                                                                        descaleV.value(),
                                                                        descaleS.value(),
                                                                        scaleS.value(),
                                                                        scaleO.value(),
                                                                        amaxS.value(),
                                                                        amaxO.value(),
                                                                        options);
            }
            return Expression::scaledDotProductAttentionFp8Forward(q,
                                                                    k,
                                                                    v,
                                                                    descaleQ.value(),
                                                                    descaleK.value(),
                                                                    descaleV.value(),
                                                                    descaleS.value(),
                                                                    scaleS.value(),
                                                                    scaleO.value(),
                                                                    amaxS.value(),
                                                                    amaxO.value(),
                                                                    options);
        }
        if (queryRagged || keyValueRagged) {
            const char* queryOffsetsName =
                queryRagged ? kQueryRaggedOffsetsInputName : kSyntheticQueryRaggedOffsetsInputName;
            const char* keyValueOffsetsName =
                keyValueRagged ? kKeyValueRaggedOffsetsInputName : kSyntheticKeyValueRaggedOffsetsInputName;
            Expression qOffsets = Expression::input(queryOffsetsName, queryRaggedOffsetDType, queryRaggedOffsetDType);
            Expression kvOffsets =
                Expression::input(keyValueOffsetsName, keyValueRaggedOffsetDType, keyValueRaggedOffsetDType);
            if (dropoutProbability > 0.0f) {
                Expression dropoutSeed = Expression::tensorRuntimeScalar(kDropoutSeedInputName, DataType::INT64, DataType::INT64);
                Expression dropoutOffset = Expression::tensorRuntimeScalar(kDropoutOffsetInputName, DataType::INT64, DataType::INT64);
                if (bias.has_value()) {
                    return Expression::scaledDotProductAttentionRagged(
                        q, k, v, bias.value(), qOffsets, kvOffsets, dropoutSeed, dropoutOffset, options);
                }
                return Expression::scaledDotProductAttentionRagged(q, k, v, qOffsets, kvOffsets, dropoutSeed, dropoutOffset, options);
            }
            if (bias.has_value()) {
                return Expression::scaledDotProductAttentionRagged(q, k, v, bias.value(), qOffsets, kvOffsets, options);
            }
            return Expression::scaledDotProductAttentionRagged(q, k, v, qOffsets, kvOffsets, options);
        }
        if (useSequenceLengths) {
            Expression qSeq = Expression::input(kQuerySequenceLengthsInputName, DataType::INT32, DataType::INT32);
            Expression kvSeq = Expression::input(kKeyValueSequenceLengthsInputName, DataType::INT32, DataType::INT32);
            if (dropoutProbability > 0.0f) {
                Expression dropoutSeed = Expression::tensorRuntimeScalar(kDropoutSeedInputName, DataType::INT64, DataType::INT64);
                Expression dropoutOffset = Expression::tensorRuntimeScalar(kDropoutOffsetInputName, DataType::INT64, DataType::INT64);
                if (bias.has_value()) {
                    return Expression::scaledDotProductAttention(q, k, v, bias.value(), qSeq, kvSeq, dropoutSeed, dropoutOffset, options);
                }
                return Expression::scaledDotProductAttention(q, k, v, qSeq, kvSeq, dropoutSeed, dropoutOffset, options);
            }
            if (bias.has_value()) {
                return Expression::scaledDotProductAttention(q, k, v, bias.value(), qSeq, kvSeq, options);
            }
            return Expression::scaledDotProductAttention(q, k, v, qSeq, kvSeq, options);
        }
        if (dropoutProbability > 0.0f) {
            Expression dropoutSeed = Expression::tensorRuntimeScalar(kDropoutSeedInputName, DataType::INT64, DataType::INT64);
            Expression dropoutOffset = Expression::tensorRuntimeScalar(kDropoutOffsetInputName, DataType::INT64, DataType::INT64);
            if (bias.has_value()) {
                return Expression::scaledDotProductAttentionWithDropout(q, k, v, bias.value(), dropoutSeed, dropoutOffset, options);
            }
            return Expression::scaledDotProductAttentionWithDropout(q, k, v, dropoutSeed, dropoutOffset, options);
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
                                                                                                     bool queryRagged,
                                                                                                     bool keyValueRagged,
                                                                                                     bool useFp8ForwardScaling,
                                                                                                     ThorImplementation::AttentionTensorLayout tensorLayout,
                                                                                                     ThorImplementation::AttentionMaskKind maskKind,
                                                                                                     int64_t diagonalLeftBound,
                                                                                                     int64_t diagonalRightBound,
                                                                                                     bool useAlibiMask,
                                                                                                     std::optional<double> attentionScale,
                                                                                                     float dropoutProbability,
                                                                                                     DataType qkvDType,
                                                                                                     DataType biasDType,
                                                                                                     DataType queryRaggedOffsetDType,
                                                                                                     DataType keyValueRaggedOffsetDType,
                                                                                                     DataType computeDType,
                                                                                                     DataType outputDType) {
    using ThorImplementation::Expression;
    using ThorImplementation::ExpressionDefinition;

    Expression out = makeAttentionOutputExpression(useBias,
                                                   useSequenceLengths,
                                                   queryRagged,
                                                   keyValueRagged,
                                                   useFp8ForwardScaling,
                                                   tensorLayout,
                                                   maskKind,
                                                   diagonalLeftBound,
                                                   diagonalRightBound,
                                                   useAlibiMask,
                                                   attentionScale,
                                                   dropoutProbability,
                                                   qkvDType,
                                                   biasDType,
                                                   queryRaggedOffsetDType,
                                                   keyValueRaggedOffsetDType,
                                                   computeDType,
                                                   outputDType);
    return std::make_shared<ExpressionDefinition>(ExpressionDefinition::fromOutputs(Expression::outputs({{kOutputName, out}})));
}

ThorImplementation::DynamicExpression makeAttentionExpression(bool useBias,
                                                              bool useSequenceLengths,
                                                              bool queryRagged,
                                                              bool keyValueRagged,
                                                              bool useFp8ForwardScaling,
                                                              ThorImplementation::AttentionTensorLayout tensorLayout,
                                                              ThorImplementation::AttentionMaskKind maskKind,
                                                              int64_t diagonalLeftBound,
                                                              int64_t diagonalRightBound,
                                                              bool useAlibiMask,
                                                              std::optional<double> attentionScale,
                                                              float dropoutProbability,
                                                              int64_t dropoutSeed,
                                                              int64_t dropoutOffset,
                                                              DataType qkvDType,
                                                              DataType biasDType,
                                                              DataType queryRaggedOffsetDType,
                                                              DataType keyValueRaggedOffsetDType,
                                                              DataType computeDType,
                                                              DataType outputDType) {
    using ThorImplementation::DynamicExpression;
    using ThorImplementation::DynamicExpressionBuild;
    using ThorImplementation::FusedEquation;
    using ThorImplementation::Tensor;

    const bool useRaggedOffsets = queryRagged || keyValueRagged;
    auto serializedDefinition = makeSerializableAttentionDefinition(useBias,
                                                                    useSequenceLengths,
                                                                    queryRagged,
                                                                    keyValueRagged,
                                                                    useFp8ForwardScaling,
                                                                    tensorLayout,
                                                                    maskKind,
                                                                    diagonalLeftBound,
                                                                    diagonalRightBound,
                                                                    useAlibiMask,
                                                                    attentionScale,
                                                                    dropoutProbability,
                                                                    qkvDType,
                                                                    biasDType,
                                                                    queryRaggedOffsetDType,
                                                                    keyValueRaggedOffsetDType,
                                                                    computeDType,
                                                                    outputDType);
    std::vector<std::string> expectedInputs =
        attentionInputNames(useBias, useSequenceLengths, queryRagged, keyValueRagged, useFp8ForwardScaling);

    return DynamicExpression(
        expectedInputs,
        {kOutputName},
        [serializedDefinition,
         useSequenceLengths,
         queryRagged,
         keyValueRagged,
         useRaggedOffsets,
         tensorLayout,
         dropoutProbability,
         dropoutSeed,
         dropoutOffset,
         queryRaggedOffsetDType,
         keyValueRaggedOffsetDType](const DynamicExpression::TensorMap& inputs,
                                    const DynamicExpression::TensorMap& outputs,
                                    Stream& stream) -> DynamicExpressionBuild {
            const Tensor& query = inputs.at(kQueryInputName);
            const Tensor& key = inputs.at(kKeyInputName);
            const Tensor& value = inputs.at(kValueInputName);
            const auto queryDims = query.getDimensions();
            const auto keyDims = key.getDimensions();
            const auto valueDims = value.getDimensions();

            auto requireRuntimeDomain = [&](const Tensor& tensor, bool ragged, const char* what) {
                const auto dims = tensor.getDimensions();
                if (ragged) {
                    if (dims.size() != 3 || dims[0] == 0) {
                        throw std::runtime_error(std::string("Canonical ragged ScaledDotProductAttention runtime ") + what +
                                                 " tensor must use packed [T,H,D] storage.");
                    }
                } else if (dims.size() != 4 || dims[0] == 0) {
                    throw std::runtime_error(std::string("Dense ScaledDotProductAttention runtime ") + what +
                                             " tensor must include batch and have rank 4.");
                }
            };
            requireRuntimeDomain(query, queryRagged, "query");
            requireRuntimeDomain(key, keyValueRagged, "key");
            requireRuntimeDomain(value, keyValueRagged, "value");

            uint64_t batch = 0;
            if (!queryRagged) {
                batch = queryDims[0];
            } else {
                const Tensor& queryOffsets = inputs.at(kQueryRaggedOffsetsInputName);
                if (!ThorImplementation::isCanonicalRowPartitionOffsetDataType(queryOffsets.getDataType())) {
                    throw std::runtime_error("ScaledDotProductAttention query_ragged_offsets dtype must be UINT32 or UINT64.");
                }
                if (queryOffsets.getDimensions().size() != 1 || queryOffsets.getTotalNumElements() < 2) {
                    throw std::runtime_error("ScaledDotProductAttention query_ragged_offsets must have canonical shape [batch+1].");
                }
                batch = queryOffsets.getTotalNumElements() - 1;
            }
            if (batch == 0) {
                throw std::runtime_error("ScaledDotProductAttention runtime batch size must be non-zero.");
            }
            if (!keyValueRagged && keyDims[0] != batch) {
                throw std::runtime_error("ScaledDotProductAttention dense key/value batch size must match the query batch size.");
            }
            if (!keyValueRagged && valueDims[0] != batch) {
                throw std::runtime_error("ScaledDotProductAttention dense value batch size must match the query batch size.");
            }

            const LogicalAttentionDims queryLogicalDims = runtimeLogicalDims(queryDims, tensorLayout);
            const LogicalAttentionDims keyValueLogicalDims = runtimeLogicalDims(keyDims, tensorLayout);

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
            auto normalizeRaggedOffsets = [&](const char* name, DataType expectedDType) -> Tensor {
                Tensor offsets = inputs.at(name);
                if (!ThorImplementation::isCanonicalRowPartitionOffsetDataType(offsets.getDataType()) ||
                    offsets.getDataType() != expectedDType) {
                    throw std::runtime_error(std::string("ScaledDotProductAttention ") + name +
                                             " dtype must match its canonical UINT32/UINT64 API row-partition dtype.");
                }
                if (offsets.getDimensions() != std::vector<uint64_t>{batch + 1}) {
                    throw std::runtime_error(std::string("ScaledDotProductAttention ") + name + " must have canonical shape [batch+1].");
                }
                if (!offsets.isDenseContiguous() || offsets.getStorageElementOffset() != 0 || offsets.hasCustomStrides()) {
                    throw std::runtime_error(std::string("ScaledDotProductAttention ") + name +
                                             " must be a canonical contiguous row-partition tensor, not a view.");
                }
                return offsets;
            };

            if (useSequenceLengths) {
                normalizeSequenceLengths(kQuerySequenceLengthsInputName);
                normalizeSequenceLengths(kKeyValueSequenceLengthsInputName);
            }
            if (useRaggedOffsets) {
                Tensor queryOffsets = queryRagged
                    ? normalizeRaggedOffsets(kQueryRaggedOffsetsInputName, queryRaggedOffsetDType)
                    : makeUniformAttentionRowPartition(
                          query.getPlacement(), queryRaggedOffsetDType, batch, queryLogicalDims.sequence, stream);
                Tensor keyValueOffsets = keyValueRagged
                    ? normalizeRaggedOffsets(kKeyValueRaggedOffsetsInputName, keyValueRaggedOffsetDType)
                    : makeUniformAttentionRowPartition(
                          query.getPlacement(), keyValueRaggedOffsetDType, batch, keyValueLogicalDims.sequence, stream);
                stampInputs[queryRagged ? kQueryRaggedOffsetsInputName : kSyntheticQueryRaggedOffsetsInputName] =
                    std::move(queryOffsets);
                stampInputs[keyValueRagged ? kKeyValueRaggedOffsetsInputName : kSyntheticKeyValueRaggedOffsetsInputName] =
                    std::move(keyValueOffsets);
            }

            std::unordered_map<std::string, ThorImplementation::TensorScalarBinding> tensorScalarInputs;
            std::function<void(Stream&)> preForwardHook;
            if (dropoutProbability > 0.0f) {
                auto dropoutState = std::make_shared<SdpaDropoutRuntimeState>(dropoutSeed, dropoutOffset);
                dropoutState->setOffsetAdvance(
                    checkedDropoutOffsetAdvance(batch, queryLogicalDims.heads, queryLogicalDims.sequence, keyValueLogicalDims.sequence));
                tensorScalarInputs[kDropoutSeedInputName] = dropoutState->seedBinding(query.getPlacement());
                tensorScalarInputs[kDropoutOffsetInputName] = dropoutState->offsetBinding(query.getPlacement());
                preForwardHook = [dropoutState](Stream& runStream) { dropoutState->uploadForForward(runStream); };
            }

            return DynamicExpressionBuild{
                std::make_shared<FusedEquation>(FusedEquation::compile(serializedDefinition->outputs, stream.getGpuNum())),
                std::move(stampInputs),
                std::move(tensorScalarInputs),
                outputs,
                {},
                std::move(preForwardHook),
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
    const bool useFp8ForwardScaling = _fp8DescaleQInput.has_value() || _fp8DescaleKInput.has_value() || _fp8DescaleVInput.has_value() ||
                                      _fp8DescaleSInput.has_value() || _fp8ScaleSInput.has_value() || _fp8ScaleOInput.has_value() ||
                                      _fp8AmaxSInput.has_value() || _fp8AmaxOInput.has_value();
    const bool allFp8ScalarsProvided = _fp8DescaleQInput.has_value() && _fp8DescaleKInput.has_value() && _fp8DescaleVInput.has_value() &&
                                       _fp8DescaleSInput.has_value() && _fp8ScaleSInput.has_value() && _fp8ScaleOInput.has_value() &&
                                       _fp8AmaxSInput.has_value() && _fp8AmaxOInput.has_value();
    if (useFp8ForwardScaling != allFp8ScalarsProvided) {
        throw std::invalid_argument("ScaledDotProductAttention experimental FP8 forward requires all descale/scale/amax tensors.");
    }
    if (useFp8ForwardScaling) {
        if (!isFp8DType(_queryInput->getDataType())) {
            throw std::invalid_argument("ScaledDotProductAttention experimental FP8 forward requires fp8_e4m3 or fp8_e5m2 query/key/value tensors. Got " +
                                        dtypeName(_queryInput->getDataType()) + ".");
        }
    } else if (!isAttentionStorageDType(_queryInput->getDataType())) {
        throw std::invalid_argument("ScaledDotProductAttention query/key/value dtype must be fp16 or bf16 for training, or fp8 with fp8ForwardScalingInputs() for experimental forward-only SDPA. Got " +
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
    const bool queryRagged = _queryRaggedInput.has_value();
    const bool keyValueRagged = _keyRaggedInput.has_value();
    const bool useRaggedOffsets = queryRagged || keyValueRagged;
    if (_querySequenceLengthsInput.has_value() != _keyValueSequenceLengthsInput.has_value()) {
        throw std::invalid_argument(
            "ScaledDotProductAttention requires both querySequenceLengthsInput and keyValueSequenceLengthsInput, or sequenceLengthsInput().");
    }
    if (_valueRaggedInput.has_value() != keyValueRagged) {
        throw std::invalid_argument("ScaledDotProductAttention key and value inputs must both be dense or both be RaggedTensor inputs.");
    }
    if (_queryRaggedOffsetsInput.has_value() != queryRagged || _keyValueRaggedOffsetsInput.has_value() != keyValueRagged) {
        throw std::invalid_argument(
            "ScaledDotProductAttention internal canonical ragged metadata does not match the public Q/KV input domains.");
    }
    if (useSequenceLengths) {
        requireSequenceLengthsInput(_querySequenceLengthsInput.value(), "querySequenceLengthsInput");
        requireSequenceLengthsInput(_keyValueSequenceLengthsInput.value(), "keyValueSequenceLengthsInput");
    }
    if (queryRagged) {
        requireRaggedOffsetsInput(
            _queryRaggedOffsetsInput.value(), "queryRaggedOffsetsInput", _queryRaggedInput->getBatchSize());
    }
    if (keyValueRagged) {
        requireRaggedOffsetsInput(
            _keyValueRaggedOffsetsInput.value(), "keyValueRaggedOffsetsInput", _keyRaggedInput->getBatchSize());
    }
    if (useRaggedOffsets) {
        if (useSequenceLengths) {
            throw std::invalid_argument(
                "ScaledDotProductAttention RaggedTensor inputs already define sequence lengths; do not also provide sequenceLengthsInput.");
        }
        if (layout != ThorImplementation::AttentionTensorLayout::BSHD) {
            throw std::invalid_argument(
                "ScaledDotProductAttention mixed/ragged execution requires BSHD layout because packed row offsets count token-major [S,H,D] values.");
        }
    }

    const DataType computeDType = _computeDataType.value_or(DataType::FP32);
    if (!isAttentionComputeDType(computeDType)) {
        throw std::invalid_argument("ScaledDotProductAttention computeDataType must currently be fp32 for cuDNN SDPA training. Got " +
                                    dtypeName(computeDType) + ".");
    }

    const DataType outputDType = _outputDataType.value_or(_queryInput->getDataType());
    if (useFp8ForwardScaling) {
        if (outputDType != _queryInput->getDataType()) {
            throw std::invalid_argument("ScaledDotProductAttention experimental FP8 forward outputDataType must match the fp8 query/key/value dtype.");
        }
    } else {
        if (!isAttentionStorageDType(outputDType)) {
            throw std::invalid_argument("ScaledDotProductAttention outputDataType must be fp16 or bf16 for the current cuDNN training path. Got " +
                                        dtypeName(outputDType) + ".");
        }
        if (outputDType != _queryInput->getDataType()) {
            throw std::invalid_argument(
                "ScaledDotProductAttention query/key/value and output tensors must use the same FP16 or BF16 dtype. "
                "Thor will not insert hidden attention-stage conversions. input=" +
                dtypeName(_queryInput->getDataType()) + ", output=" + dtypeName(outputDType) + ".");
        }
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
        throw std::invalid_argument(
            "ScaledDotProductAttention ALiBi requires diagonalRightBound == 0 because cuDNN rejects ALiBi with positive right bounds.");
    }
    if (useAlibi && (maskKind == ThorImplementation::AttentionMaskKind::CausalBottomRight ||
                     maskKind == ThorImplementation::AttentionMaskKind::SlidingWindowBottomRight)) {
        throw std::invalid_argument(
            "ScaledDotProductAttention ALiBi cannot currently be combined with bottom-right/decode masks in cuDNN SDPA.");
    }

    if (_attentionScale.has_value() && (!std::isfinite(_attentionScale.value()) || _attentionScale.value() <= 0.0)) {
        throw std::invalid_argument("ScaledDotProductAttention attentionScale must be finite and positive.");
    }

    const float dropoutProbability = _dropoutProbability.value_or(0.0f);
    if (!std::isfinite(dropoutProbability) || dropoutProbability < 0.0f || dropoutProbability >= 1.0f) {
        throw std::invalid_argument("ScaledDotProductAttention dropoutProbability must be finite and in [0, 1).");
    }
    if (dropoutProbability > 0.0f && (maskKind == ThorImplementation::AttentionMaskKind::CausalBottomRight ||
                                      maskKind == ThorImplementation::AttentionMaskKind::SlidingWindowBottomRight)) {
        throw std::invalid_argument("ScaledDotProductAttention dropout cannot currently be combined with bottom-right/decode masks in cuDNN SDPA.");
    }
    if (dropoutProbability > 0.0f && _dropoutOffset.value_or(0) < 0) {
        throw std::invalid_argument("ScaledDotProductAttention dropoutOffset must be non-negative when dropout is enabled.");
    }
    if (useFp8ForwardScaling) {
        requireFp8ScaleInput(_fp8DescaleQInput.value(), "fp8DescaleQInput");
        requireFp8ScaleInput(_fp8DescaleKInput.value(), "fp8DescaleKInput");
        requireFp8ScaleInput(_fp8DescaleVInput.value(), "fp8DescaleVInput");
        requireFp8ScaleInput(_fp8DescaleSInput.value(), "fp8DescaleSInput");
        requireFp8ScaleInput(_fp8ScaleSInput.value(), "fp8ScaleSInput");
        requireFp8ScaleInput(_fp8ScaleOInput.value(), "fp8ScaleOInput");
        requireFp8ScaleInput(_fp8AmaxSInput.value(), "fp8AmaxSInput");
        requireFp8ScaleInput(_fp8AmaxOInput.value(), "fp8AmaxOInput");
        if (_biasInput.has_value()) {
            throw std::invalid_argument("ScaledDotProductAttention experimental FP8 forward does not support additive score bias on the validated cuDNN surface.");
        }
        if (dropoutProbability > 0.0f) {
            throw std::invalid_argument("ScaledDotProductAttention experimental FP8 forward does not support dropout on the validated cuDNN surface.");
        }
        if (useRaggedOffsets) {
            throw std::invalid_argument("ScaledDotProductAttention experimental FP8 forward does not support canonical RaggedTensor inputs on the validated cuDNN surface.");
        }
        if (useAlibi) {
            throw std::invalid_argument("ScaledDotProductAttention experimental FP8 forward does not support ALiBi on the validated cuDNN surface.");
        }
        if (maskKind == ThorImplementation::AttentionMaskKind::CausalBottomRight ||
            maskKind == ThorImplementation::AttentionMaskKind::SlidingWindowTopLeft ||
            maskKind == ThorImplementation::AttentionMaskKind::SlidingWindowBottomRight) {
            throw std::invalid_argument("ScaledDotProductAttention experimental FP8 forward supports only no mask or cuDNN's causal mask API.");
        }
        if (qDims.head_dim > 128 || vDims.head_dim > 128 || qDims.head_dim % 16 != 0 || vDims.head_dim % 16 != 0) {
            throw std::invalid_argument("ScaledDotProductAttention experimental FP8 forward requires query/key and value head_dim to be multiples of 16 and <= 128.");
        }
        if (qDims.sequence == 1 && kDims.sequence > 1) {
            throw std::invalid_argument("ScaledDotProductAttention experimental FP8 forward does not support decode-style Sq=1, Skv>1 shapes on the validated cuDNN surface.");
        }
    }

    if (_biasInput.has_value()) {
        requireRank3(_biasInput.value(), "bias");
        const auto biasDims = _biasInput->getDimensions();
        if (!isAllowedApiBiasDims(biasDims, qDims, kDims)) {
            throw std::invalid_argument(
                "ScaledDotProductAttention additive bias dimensions must be [1|query_heads, 1|query_sequence, 1|key_sequence] at the API level.");
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
    const bool queryRagged = _queryRaggedInput.has_value();
    const bool keyValueRagged = _keyRaggedInput.has_value() || _valueRaggedInput.has_value();
    if (_keyRaggedInput.has_value() != _valueRaggedInput.has_value()) {
        throw std::invalid_argument("ScaledDotProductAttention key and value inputs must both be dense or both be RaggedTensor inputs.");
    }
    if (queryRagged) {
        if (_queryInput.has_value()) {
            throw std::invalid_argument("ScaledDotProductAttention query cannot be provided as both Tensor and RaggedTensor.");
        }
        if (_queryRaggedInput->getBatchSize() > std::numeric_limits<uint32_t>::max()) {
            throw std::invalid_argument("ScaledDotProductAttention RaggedTensor query batch size exceeds the supported placement batch capacity.");
        }
        _queryInput = _queryRaggedInput->getValues();
        _queryRaggedOffsetsInput = _queryRaggedInput->getOffsets();
    }
    if (keyValueRagged) {
        if (_keyInput.has_value() || _valueInput.has_value()) {
            throw std::invalid_argument("ScaledDotProductAttention key/value cannot be provided as both Tensor and RaggedTensor.");
        }
        if (_keyRaggedInput->getBatchSize() != _valueRaggedInput->getBatchSize() ||
            _keyRaggedInput->getOffsets() != _valueRaggedInput->getOffsets()) {
            throw std::invalid_argument("ScaledDotProductAttention ragged key and value inputs must share one logical row partition.");
        }
        if (_keyRaggedInput->getBatchSize() > std::numeric_limits<uint32_t>::max()) {
            throw std::invalid_argument("ScaledDotProductAttention RaggedTensor key/value batch size exceeds the supported placement batch capacity.");
        }
        _keyInput = _keyRaggedInput->getValues();
        _valueInput = _valueRaggedInput->getValues();
        _keyValueRaggedOffsetsInput = _keyRaggedInput->getOffsets();
    }
    if (queryRagged && keyValueRagged && _queryRaggedInput->getBatchSize() != _keyRaggedInput->getBatchSize()) {
        throw std::invalid_argument("ScaledDotProductAttention ragged query and key/value inputs must have the same logical batch size.");
    }

    const bool anyRaggedInput = queryRagged || keyValueRagged;
    if (!_tensorLayout.has_value()) {
        _tensorLayout = anyRaggedInput ? ThorImplementation::AttentionTensorLayout::BSHD
                                       : ThorImplementation::AttentionTensorLayout::BHSD;
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
    if (!_dropoutProbability.has_value()) {
        _dropoutProbability = 0.0f;
    }
    if (!_dropoutSeed.has_value()) {
        _dropoutSeed = 0;
    }
    if (!_dropoutOffset.has_value()) {
        _dropoutOffset = 0;
    }
    if (!_computeDataType.has_value()) {
        _computeDataType = DataType::FP32;
    }
    if (!_outputDataType.has_value() && _queryInput.has_value()) {
        _outputDataType = _queryInput->getDataType();
    }

    verifyConfig();

    const bool useSequenceLengths = _querySequenceLengthsInput.has_value();
    const bool useFp8ForwardScaling = _fp8DescaleQInput.has_value();
    const auto qDims = logicalDims(_queryInput->getDimensions(), _tensorLayout.value());
    const auto vDims = logicalDims(_valueInput->getDimensions(), _tensorLayout.value());
    Tensor output(_outputDataType.value(), outputDims(qDims.heads, qDims.sequence, vDims.head_dim, _tensorLayout.value()));

    std::vector<std::string> inputNames =
        attentionInputNames(_biasInput.has_value(), useSequenceLengths, queryRagged, keyValueRagged, useFp8ForwardScaling);
    CustomLayer::TensorMap inputInterface = attentionInputInterface(_queryInput.value(),
                                                                    _keyInput.value(),
                                                                    _valueInput.value(),
                                                                    _biasInput,
                                                                    _querySequenceLengthsInput,
                                                                    _keyValueSequenceLengthsInput,
                                                                    _queryRaggedOffsetsInput,
                                                                    _keyValueRaggedOffsetsInput,
                                                                    _fp8DescaleQInput,
                                                                    _fp8DescaleKInput,
                                                                    _fp8DescaleVInput,
                                                                    _fp8DescaleSInput,
                                                                    _fp8ScaleSInput,
                                                                    _fp8ScaleOInput,
                                                                    _fp8AmaxSInput,
                                                                    _fp8AmaxOInput);

    ScaledDotProductAttention attention(makeAttentionExpression(_biasInput.has_value(),
                                                                useSequenceLengths,
                                                                queryRagged,
                                                                keyValueRagged,
                                                                useFp8ForwardScaling,
                                                                _tensorLayout.value(),
                                                                _maskKind.value(),
                                                                _diagonalLeftBound.value(),
                                                                _diagonalRightBound.value(),
                                                                _useAlibiMask.value(),
                                                                _attentionScale,
                                                                _dropoutProbability.value(),
                                                                _dropoutSeed.value(),
                                                                _dropoutOffset.value(),
                                                                _queryInput->getDataType(),
                                                                _biasInput.has_value() ? _biasInput->getDataType() : _computeDataType.value(),
                                                                _queryRaggedOffsetsInput.has_value() ? _queryRaggedOffsetsInput->getDataType()
                                                                                                    : ThorImplementation::kDefaultRowPartitionOffsetDataType,
                                                                _keyValueRaggedOffsetsInput.has_value() ? _keyValueRaggedOffsetsInput->getDataType()
                                                                                                        : ThorImplementation::kDefaultRowPartitionOffsetDataType,
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
                                                _dropoutProbability.value(),
                                                _dropoutSeed.value(),
                                                _dropoutOffset.value(),
                                                _querySequenceLengthsInput,
                                                _keyValueSequenceLengthsInput,
                                                _queryRaggedInput,
                                                _keyRaggedInput,
                                                _valueRaggedInput,
                                                _queryRaggedOffsetsInput,
                                                _keyValueRaggedOffsetsInput,
                                                _fp8DescaleQInput,
                                                _fp8DescaleKInput,
                                                _fp8DescaleVInput,
                                                _fp8DescaleSInput,
                                                _fp8ScaleSInput,
                                                _fp8ScaleOInput,
                                                _fp8AmaxSInput,
                                                _fp8AmaxOInput,
                                                _computeDataType.value(),
                                                _outputDataType.value());

    attention.addToNetwork(_network.value());
    return attention;
}

json ScaledDotProductAttention::architectureJson() const {
    json j;
    j["factory"] = Layer::Factory::Learning.value();
    j["version"] = "2.1.0";
    j["layer_type"] = "scaled_dot_product_attention";
    j["layer_name"] = std::string("layer") + std::to_string(getId());

    j["tensor_layout"] = attentionTensorLayoutToString(tensorLayout);
    j["mask_kind"] = attentionMaskKindToString(maskKind);
    j["diagonal_left_bound"] = diagonalLeftBound;
    j["diagonal_right_bound"] = diagonalRightBound;
    j["use_alibi_mask"] = useAlibiMask;
    j["attention_scale"] = attentionScale.has_value() ? json(attentionScale.value()) : json(nullptr);
    j["dropout_probability"] = dropoutProbability;
    j["dropout_seed"] = dropoutSeed;
    j["dropout_offset"] = dropoutOffset;
    j["use_bias"] = getInputInterface().contains(kBiasInputName);
    j["use_sequence_lengths"] = querySequenceLengthsInput.has_value();
    j["use_ragged_input"] = queryRaggedInput.has_value() || keyRaggedInput.has_value();
    j["query_ragged"] = queryRaggedInput.has_value();
    j["key_value_ragged"] = keyRaggedInput.has_value();
    j["use_fp8_forward_scaling"] = fp8DescaleQInput.has_value();
    j["compute_data_type"] = computeDataType;
    j["output_data_type"] = outputDataType;
    j["parameters"] = json::object();

    const CustomLayer::TensorMap inputInterface = getInputInterface();
    j["query_input"] = inputInterface.at(kQueryInputName).architectureJson();
    j["key_input"] = inputInterface.at(kKeyInputName).architectureJson();
    j["value_input"] = inputInterface.at(kValueInputName).architectureJson();
    if (inputInterface.contains(kBiasInputName)) {
        j["bias_input"] = inputInterface.at(kBiasInputName).architectureJson();
    }
    if (querySequenceLengthsInput.has_value()) {
        j["query_sequence_lengths_input"] = querySequenceLengthsInput.value().architectureJson();
        j["key_value_sequence_lengths_input"] = keyValueSequenceLengthsInput.value().architectureJson();
    }
    if (queryRaggedInput.has_value()) {
        j["query_ragged_input"] = queryRaggedInput->architectureJson();
    }
    if (keyRaggedInput.has_value()) {
        j["key_ragged_input"] = keyRaggedInput->architectureJson();
        j["value_ragged_input"] = valueRaggedInput->architectureJson();
    }
    if (fp8DescaleQInput.has_value()) {
        j["fp8_descale_q_input"] = fp8DescaleQInput.value().architectureJson();
        j["fp8_descale_k_input"] = fp8DescaleKInput.value().architectureJson();
        j["fp8_descale_v_input"] = fp8DescaleVInput.value().architectureJson();
        j["fp8_descale_s_input"] = fp8DescaleSInput.value().architectureJson();
        j["fp8_scale_s_input"] = fp8ScaleSInput.value().architectureJson();
        j["fp8_scale_o_input"] = fp8ScaleOInput.value().architectureJson();
        j["fp8_amax_s_input"] = fp8AmaxSInput.value().architectureJson();
        j["fp8_amax_o_input"] = fp8AmaxOInput.value().architectureJson();
    }
    j["output"] = getOutput(kOutputName).architectureJson();
    return j;
}

json ScaledDotProductAttention::serialize(thor_file::TarWriter& archiveWriter,
                                          Stream stream,
                                          bool saveOptimizerState,
                                          ThorImplementation::StampedNetwork& stampedNetwork) const {
    (void)archiveWriter;
    (void)stream;
    (void)saveOptimizerState;
    (void)stampedNetwork;
    return architectureJson();
}

void ScaledDotProductAttention::deserialize(std::shared_ptr<thor_file::TarReader>& archiveReader, const json& j, Network* network) {
    const std::string version = j.at("version").get<std::string>();
    if (version != "2.0.0" && version != "2.1.0") {
        throw std::runtime_error("Unsupported version in ScaledDotProductAttention::deserialize: " + version);
    }
    if (j.at("layer_type").get<std::string>() != "scaled_dot_product_attention") {
        throw std::runtime_error("Layer type mismatch in ScaledDotProductAttention::deserialize: " + j.at("layer_type").get<std::string>());
    }

    Tensor queryInput = network->getApiTensorByOriginalId(j.at("query_input").at("id").get<uint64_t>());
    Tensor keyInput = network->getApiTensorByOriginalId(j.at("key_input").at("id").get<uint64_t>());
    Tensor valueInput = network->getApiTensorByOriginalId(j.at("value_input").at("id").get<uint64_t>());
    std::optional<Tensor> biasInput = std::nullopt;
    if (j.value("use_bias", false) || j.contains("bias_input")) {
        if (!j.contains("bias_input")) {
            throw std::runtime_error("ScaledDotProductAttention deserialize missing bias_input.");
        }
        biasInput = network->getApiTensorByOriginalId(j.at("bias_input").at("id").get<uint64_t>());
    }

    if (j.contains("sequence_lengths_input") || j.contains("ragged_offsets_input") ||
        j.contains("use_separate_sequence_lengths") || j.contains("use_separate_ragged_offsets")) {
        throw std::runtime_error(
            "ScaledDotProductAttention deserialize does not support legacy single-metadata fields; use query/key-value sequence lengths or canonical RaggedTensor inputs.");
    }

    std::optional<Tensor> querySequenceLengthsInput = std::nullopt;
    std::optional<Tensor> keyValueSequenceLengthsInput = std::nullopt;
    if (j.value("use_sequence_lengths", false) || j.contains("query_sequence_lengths_input") ||
        j.contains("key_value_sequence_lengths_input")) {
        if (!j.contains("query_sequence_lengths_input") || !j.contains("key_value_sequence_lengths_input")) {
            throw std::runtime_error(
                "ScaledDotProductAttention deserialize missing query_sequence_lengths_input/key_value_sequence_lengths_input.");
        }
        querySequenceLengthsInput = network->getApiTensorByOriginalId(j.at("query_sequence_lengths_input").at("id").get<uint64_t>());
        keyValueSequenceLengthsInput =
            network->getApiTensorByOriginalId(j.at("key_value_sequence_lengths_input").at("id").get<uint64_t>());
    }

    if (j.contains("use_ragged_offsets") || j.contains("query_ragged_offsets_input") ||
        j.contains("key_value_ragged_offsets_input")) {
        throw std::runtime_error(
            "ScaledDotProductAttention archives with raw ragged-offset inputs are unsupported; use canonical RaggedTensor inputs.");
    }
    auto raggedFromMetadata = [&](const json& r, const char* field) -> RaggedTensor {
        if (!r.contains("values") || !r.contains("offsets"))
            throw std::runtime_error(std::string("ScaledDotProductAttention deserialize malformed ") + field + ".");
        Tensor values = network->getApiTensorByOriginalId(r.at("values").at("id").get<uint64_t>());
        Tensor offsets = network->getApiTensorByOriginalId(r.at("offsets").at("id").get<uint64_t>());
        return r.contains("max_values_per_row")
            ? RaggedTensor(values, offsets, r.at("max_values_per_row").get<uint64_t>())
            : RaggedTensor(values, offsets);
    };
    std::optional<RaggedTensor> queryRaggedInput = std::nullopt;
    std::optional<RaggedTensor> keyRaggedInput = std::nullopt;
    std::optional<RaggedTensor> valueRaggedInput = std::nullopt;
    std::optional<Tensor> queryRaggedOffsetsInput = std::nullopt;
    std::optional<Tensor> keyValueRaggedOffsetsInput = std::nullopt;
    const bool legacyAllRagged = version == "2.0.0" && (j.value("use_ragged_input", false) || j.contains("query_ragged_input"));
    const bool queryRagged = version == "2.0.0" ? legacyAllRagged : j.value("query_ragged", j.contains("query_ragged_input"));
    const bool keyValueRagged =
        version == "2.0.0" ? legacyAllRagged : j.value("key_value_ragged", j.contains("key_ragged_input") || j.contains("value_ragged_input"));
    if (j.contains("use_ragged_input") && j.at("use_ragged_input").get<bool>() != (queryRagged || keyValueRagged)) {
        throw std::runtime_error("ScaledDotProductAttention deserialize use_ragged_input disagrees with query/key-value ragged metadata.");
    }
    if (queryRagged) {
        if (!j.contains("query_ragged_input")) {
            throw std::runtime_error("ScaledDotProductAttention deserialize missing canonical RaggedTensor query metadata.");
        }
        queryRaggedInput = raggedFromMetadata(j.at("query_ragged_input"), "query_ragged_input");
        if (queryRaggedInput->getBatchSize() > UINT32_MAX) {
            throw std::runtime_error("ScaledDotProductAttention deserialize RaggedTensor query batch size exceeds the supported placement batch capacity.");
        }
        queryInput = queryRaggedInput->getValues();
        queryRaggedOffsetsInput = queryRaggedInput->getOffsets();
    } else if (j.contains("query_ragged_input")) {
        throw std::runtime_error("ScaledDotProductAttention deserialize query_ragged_input requires query_ragged=true.");
    }
    if (keyValueRagged) {
        if (!j.contains("key_ragged_input") || !j.contains("value_ragged_input")) {
            throw std::runtime_error("ScaledDotProductAttention deserialize missing canonical RaggedTensor key/value metadata.");
        }
        keyRaggedInput = raggedFromMetadata(j.at("key_ragged_input"), "key_ragged_input");
        valueRaggedInput = raggedFromMetadata(j.at("value_ragged_input"), "value_ragged_input");
        if (keyRaggedInput->getBatchSize() != valueRaggedInput->getBatchSize() ||
            keyRaggedInput->getOffsets() != valueRaggedInput->getOffsets()) {
            throw std::runtime_error("ScaledDotProductAttention deserialize ragged key/value inputs do not share one logical row partition.");
        }
        if (keyRaggedInput->getBatchSize() > UINT32_MAX) {
            throw std::runtime_error("ScaledDotProductAttention deserialize RaggedTensor key/value batch size exceeds the supported placement batch capacity.");
        }
        keyInput = keyRaggedInput->getValues();
        valueInput = valueRaggedInput->getValues();
        keyValueRaggedOffsetsInput = keyRaggedInput->getOffsets();
    } else if (j.contains("key_ragged_input") || j.contains("value_ragged_input")) {
        throw std::runtime_error("ScaledDotProductAttention deserialize ragged key/value metadata requires key_value_ragged=true.");
    }
    if (queryRagged && keyValueRagged && queryRaggedInput->getBatchSize() != keyRaggedInput->getBatchSize()) {
        throw std::runtime_error("ScaledDotProductAttention deserialize ragged query and key/value inputs must have the same logical batch size.");
    }

    std::optional<Tensor> fp8DescaleQInput = std::nullopt;
    std::optional<Tensor> fp8DescaleKInput = std::nullopt;
    std::optional<Tensor> fp8DescaleVInput = std::nullopt;
    std::optional<Tensor> fp8DescaleSInput = std::nullopt;
    std::optional<Tensor> fp8ScaleSInput = std::nullopt;
    std::optional<Tensor> fp8ScaleOInput = std::nullopt;
    std::optional<Tensor> fp8AmaxSInput = std::nullopt;
    std::optional<Tensor> fp8AmaxOInput = std::nullopt;
    if (j.value("use_fp8_forward_scaling", false) || j.contains("fp8_descale_q_input")) {
        const std::vector<const char*> required = {"fp8_descale_q_input",
                                                  "fp8_descale_k_input",
                                                  "fp8_descale_v_input",
                                                  "fp8_descale_s_input",
                                                  "fp8_scale_s_input",
                                                  "fp8_scale_o_input",
                                                  "fp8_amax_s_input",
                                                  "fp8_amax_o_input"};
        for (const char* name : required) {
            if (!j.contains(name)) {
                throw std::runtime_error(std::string("ScaledDotProductAttention deserialize missing ") + name + ".");
            }
        }
        fp8DescaleQInput = network->getApiTensorByOriginalId(j.at("fp8_descale_q_input").at("id").get<uint64_t>());
        fp8DescaleKInput = network->getApiTensorByOriginalId(j.at("fp8_descale_k_input").at("id").get<uint64_t>());
        fp8DescaleVInput = network->getApiTensorByOriginalId(j.at("fp8_descale_v_input").at("id").get<uint64_t>());
        fp8DescaleSInput = network->getApiTensorByOriginalId(j.at("fp8_descale_s_input").at("id").get<uint64_t>());
        fp8ScaleSInput = network->getApiTensorByOriginalId(j.at("fp8_scale_s_input").at("id").get<uint64_t>());
        fp8ScaleOInput = network->getApiTensorByOriginalId(j.at("fp8_scale_o_input").at("id").get<uint64_t>());
        fp8AmaxSInput = network->getApiTensorByOriginalId(j.at("fp8_amax_s_input").at("id").get<uint64_t>());
        fp8AmaxOInput = network->getApiTensorByOriginalId(j.at("fp8_amax_o_input").at("id").get<uint64_t>());
    }

    Tensor output = Tensor::deserialize(j.at("output"), archiveReader.get());

    const ThorImplementation::AttentionTensorLayout tensorLayout =
        attentionTensorLayoutFromString(j.value("tensor_layout", std::string("bhsd")));
    if ((queryRagged || keyValueRagged) && tensorLayout != ThorImplementation::AttentionTensorLayout::BSHD) {
        throw std::runtime_error(
            "ScaledDotProductAttention deserialize mixed/ragged inputs require BSHD token-major layout.");
    }
    const ThorImplementation::AttentionMaskKind maskKind = attentionMaskKindFromString(j.value("mask_kind", std::string("none")));
    const int64_t diagonalLeftBound = j.value("diagonal_left_bound", int64_t{0});
    const int64_t diagonalRightBound = j.value("diagonal_right_bound", int64_t{0});
    const bool useAlibiMask = j.value("use_alibi_mask", false);
    std::optional<double> attentionScale = std::nullopt;
    if (j.contains("attention_scale") && !j.at("attention_scale").is_null()) {
        attentionScale = j.at("attention_scale").get<double>();
    }
    const float dropoutProbability = j.value("dropout_probability", 0.0f);
    const int64_t dropoutSeed = j.value("dropout_seed", int64_t{0});
    const int64_t dropoutOffset = j.value("dropout_offset", int64_t{0});
    const DataType computeDataType = j.at("compute_data_type").get<DataType>();
    const DataType outputDataType = j.at("output_data_type").get<DataType>();

    requireRank3(queryInput, "query");
    requireRank3(keyInput, "key");
    requireRank3(valueInput, "value");
    const auto qDims = logicalDims(queryInput.getDimensions(), tensorLayout);
    const auto kDims = logicalDims(keyInput.getDimensions(), tensorLayout);
    const auto vDims = logicalDims(valueInput.getDimensions(), tensorLayout);
    const bool useFp8ForwardScaling = fp8DescaleQInput.has_value();
    if (queryInput.getDataType() != keyInput.getDataType() || queryInput.getDataType() != valueInput.getDataType()) {
        throw std::runtime_error("ScaledDotProductAttention deserialize query/key/value tensors must have the same dtype.");
    }
    if (useFp8ForwardScaling) {
        if (!isFp8DType(queryInput.getDataType())) {
            throw std::runtime_error("ScaledDotProductAttention deserialize FP8 forward requires fp8_e4m3 or fp8_e5m2 query/key/value dtype.");
        }
    } else if (!isAttentionStorageDType(queryInput.getDataType())) {
        throw std::runtime_error("ScaledDotProductAttention deserialize query/key/value dtype must be fp16 or bf16, or fp8 with FP8 scale inputs.");
    }
    if (qDims.head_dim != kDims.head_dim || kDims.heads != vDims.heads || kDims.sequence != vDims.sequence) {
        throw std::runtime_error("ScaledDotProductAttention deserialize found inconsistent query/key/value dimensions.");
    }
    if (kDims.heads == 0 || qDims.heads % kDims.heads != 0) {
        throw std::runtime_error("ScaledDotProductAttention deserialize query head count must be a multiple of key/value head count.");
    }
    if (!isAttentionComputeDType(computeDataType) || (!useFp8ForwardScaling && !isAttentionStorageDType(outputDataType)) ||
        (useFp8ForwardScaling && outputDataType != queryInput.getDataType())) {
        throw std::runtime_error("ScaledDotProductAttention deserialize found unsupported compute/output dtype.");
    }
    if (attentionScale.has_value() && (!std::isfinite(attentionScale.value()) || attentionScale.value() <= 0.0)) {
        throw std::runtime_error("ScaledDotProductAttention deserialize attention_scale must be finite and positive.");
    }
    if (!std::isfinite(dropoutProbability) || dropoutProbability < 0.0f || dropoutProbability >= 1.0f) {
        throw std::runtime_error("ScaledDotProductAttention deserialize dropout_probability must be finite and in [0, 1).");
    }
    if (dropoutProbability > 0.0f && dropoutOffset < 0) {
        throw std::runtime_error("ScaledDotProductAttention deserialize dropout_offset must be non-negative when dropout is enabled.");
    }
    if (dropoutProbability > 0.0f && (maskKind == ThorImplementation::AttentionMaskKind::CausalBottomRight ||
                                      maskKind == ThorImplementation::AttentionMaskKind::SlidingWindowBottomRight)) {
        throw std::runtime_error("ScaledDotProductAttention deserialize dropout cannot be combined with bottom-right/decode masks.");
    }
    if (useAlibiMask && maskKind != ThorImplementation::AttentionMaskKind::CausalTopLeft &&
        maskKind != ThorImplementation::AttentionMaskKind::CausalBottomRight &&
        maskKind != ThorImplementation::AttentionMaskKind::SlidingWindowTopLeft &&
        maskKind != ThorImplementation::AttentionMaskKind::SlidingWindowBottomRight) {
        throw std::runtime_error("ScaledDotProductAttention deserialize ALiBi requires a causal/sliding-window diagonal mask.");
    }
    if (useAlibiMask && diagonalRightBound != 0) {
        throw std::runtime_error(
            "ScaledDotProductAttention deserialize ALiBi requires diagonal_right_bound == 0 because cuDNN rejects ALiBi with positive right bounds.");
    }
    if (useAlibiMask && (maskKind == ThorImplementation::AttentionMaskKind::CausalBottomRight ||
                         maskKind == ThorImplementation::AttentionMaskKind::SlidingWindowBottomRight)) {
        throw std::runtime_error("ScaledDotProductAttention deserialize ALiBi cannot be combined with bottom-right/decode masks.");
    }
    if (biasInput.has_value()) {
        requireRank3(biasInput.value(), "bias");
        if (!isAllowedApiBiasDims(biasInput->getDimensions(), qDims, kDims) || biasInput->getDataType() != computeDataType) {
            throw std::runtime_error("ScaledDotProductAttention deserialize found invalid additive bias tensor.");
        }
        if (maskKind == ThorImplementation::AttentionMaskKind::CausalBottomRight ||
            maskKind == ThorImplementation::AttentionMaskKind::SlidingWindowBottomRight) {
            throw std::runtime_error("ScaledDotProductAttention deserialize bottom-right/decode masks cannot be combined with additive bias.");
        }
    }
    if (querySequenceLengthsInput.has_value() != keyValueSequenceLengthsInput.has_value()) {
        throw std::runtime_error("ScaledDotProductAttention deserialize requires both query/key-value sequence lengths.");
    }
    if (queryRaggedOffsetsInput.has_value() != queryRagged || keyValueRaggedOffsetsInput.has_value() != keyValueRagged) {
        throw std::runtime_error("ScaledDotProductAttention deserialize row-partition metadata does not match the serialized Q/KV domains.");
    }
    if (querySequenceLengthsInput.has_value()) {
        requireSequenceLengthsInput(querySequenceLengthsInput.value(), "querySequenceLengthsInput");
        requireSequenceLengthsInput(keyValueSequenceLengthsInput.value(), "keyValueSequenceLengthsInput");
    }
    if ((queryRagged || keyValueRagged) && querySequenceLengthsInput.has_value()) {
        throw std::runtime_error(
            "ScaledDotProductAttention deserialize found redundant sequence lengths with canonical ragged row partitions.");
    }
    if (queryRagged) {
        requireRaggedOffsetsInput(
            queryRaggedOffsetsInput.value(), "queryRaggedOffsetsInput", queryRaggedInput->getBatchSize());
    }
    if (keyValueRagged) {
        requireRaggedOffsetsInput(
            keyValueRaggedOffsetsInput.value(), "keyValueRaggedOffsetsInput", keyRaggedInput->getBatchSize());
    }
    if (useFp8ForwardScaling) {
        requireFp8ScaleInput(fp8DescaleQInput.value(), "fp8DescaleQInput");
        requireFp8ScaleInput(fp8DescaleKInput.value(), "fp8DescaleKInput");
        requireFp8ScaleInput(fp8DescaleVInput.value(), "fp8DescaleVInput");
        requireFp8ScaleInput(fp8DescaleSInput.value(), "fp8DescaleSInput");
        requireFp8ScaleInput(fp8ScaleSInput.value(), "fp8ScaleSInput");
        requireFp8ScaleInput(fp8ScaleOInput.value(), "fp8ScaleOInput");
        requireFp8ScaleInput(fp8AmaxSInput.value(), "fp8AmaxSInput");
        requireFp8ScaleInput(fp8AmaxOInput.value(), "fp8AmaxOInput");
        if (biasInput.has_value() || dropoutProbability > 0.0f || queryRagged || keyValueRagged || useAlibiMask ||
            maskKind == ThorImplementation::AttentionMaskKind::CausalBottomRight ||
            maskKind == ThorImplementation::AttentionMaskKind::SlidingWindowTopLeft ||
            maskKind == ThorImplementation::AttentionMaskKind::SlidingWindowBottomRight || qDims.head_dim > 128 || vDims.head_dim > 128 ||
            qDims.head_dim % 16 != 0 || vDims.head_dim % 16 != 0 || (qDims.sequence == 1 && kDims.sequence > 1)) {
            throw std::runtime_error("ScaledDotProductAttention deserialize found unsupported experimental FP8 forward configuration.");
        }
    }
    if (output.getDataType() != outputDataType || output.getDimensions() != outputDims(qDims.heads, qDims.sequence, vDims.head_dim, tensorLayout)) {
        throw std::runtime_error("ScaledDotProductAttention deserialize output tensor does not match configured output dtype/dimensions.");
    }

    const bool useSequenceLengths = querySequenceLengthsInput.has_value();
    std::vector<std::string> inputNames =
        attentionInputNames(biasInput.has_value(), useSequenceLengths, queryRagged, keyValueRagged, useFp8ForwardScaling);
    CustomLayer::TensorMap inputInterface = attentionInputInterface(queryInput,
                                                                    keyInput,
                                                                    valueInput,
                                                                    biasInput,
                                                                    querySequenceLengthsInput,
                                                                    keyValueSequenceLengthsInput,
                                                                    queryRaggedOffsetsInput,
                                                                    keyValueRaggedOffsetsInput,
                                                                    fp8DescaleQInput,
                                                                    fp8DescaleKInput,
                                                                    fp8DescaleVInput,
                                                                    fp8DescaleSInput,
                                                                    fp8ScaleSInput,
                                                                    fp8ScaleOInput,
                                                                    fp8AmaxSInput,
                                                                    fp8AmaxOInput);

    ScaledDotProductAttention layer(makeAttentionExpression(biasInput.has_value(),
                                                            useSequenceLengths,
                                                            queryRagged,
                                                            keyValueRagged,
                                                            useFp8ForwardScaling,
                                                            tensorLayout,
                                                            maskKind,
                                                            diagonalLeftBound,
                                                            diagonalRightBound,
                                                            useAlibiMask,
                                                            attentionScale,
                                                            dropoutProbability,
                                                            dropoutSeed,
                                                            dropoutOffset,
                                                            queryInput.getDataType(),
                                                            biasInput.has_value() ? biasInput->getDataType() : computeDataType,
                                                            queryRaggedOffsetsInput.has_value() ? queryRaggedOffsetsInput->getDataType()
                                                                                              : ThorImplementation::kDefaultRowPartitionOffsetDataType,
                                                            keyValueRaggedOffsetsInput.has_value() ? keyValueRaggedOffsetsInput->getDataType()
                                                                                                  : ThorImplementation::kDefaultRowPartitionOffsetDataType,
                                                            computeDataType,
                                                            outputDataType),
                                    inputNames,
                                    {kOutputName},
                                    {inputInterface},
                                    {{{kOutputName, output}}},
                                    tensorLayout,
                                    maskKind,
                                    diagonalLeftBound,
                                    diagonalRightBound,
                                    useAlibiMask,
                                    attentionScale,
                                    dropoutProbability,
                                    dropoutSeed,
                                    dropoutOffset,
                                    querySequenceLengthsInput,
                                    keyValueSequenceLengthsInput,
                                    queryRaggedInput,
                                    keyRaggedInput,
                                    valueRaggedInput,
                                    queryRaggedOffsetsInput,
                                    keyValueRaggedOffsetsInput,
                                    fp8DescaleQInput,
                                    fp8DescaleKInput,
                                    fp8DescaleVInput,
                                    fp8DescaleSInput,
                                    fp8ScaleSInput,
                                    fp8ScaleOInput,
                                    fp8AmaxSInput,
                                    fp8AmaxOInput,
                                    computeDataType,
                                    outputDataType);
    layer.addToNetwork(network);
}

}  // namespace Thor

namespace {
static const bool registeredScaledDotProductAttention = [] {
    Thor::TrainableLayer::register_layer("scaled_dot_product_attention", &Thor::ScaledDotProductAttention::deserialize);
    return true;
}();
}  // namespace
