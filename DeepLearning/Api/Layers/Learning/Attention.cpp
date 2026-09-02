
#include "DeepLearning/Api/Layers/Learning/Attention.h"
#include "DeepLearning/Implementation/Layers/NeuralNetwork/Attention.h"
#include "Utilities/TensorOperations/Ragged/RowPartition.h"
#include "Utilities/TensorOperations/Ragged/RowPartitionDTypePolicy.h"

#include "DeepLearning/Api/Initializers/Glorot.h"
#include "DeepLearning/Api/Parameter/ParameterSpecification.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/FusedEquation.h"
#include "Utilities/Expression/CudaKernelExpression.h"
#include "Utilities/Expression/DropOutPostOp.h"
#include "Utilities/TensorOperations/Scalar/SetScalar.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <set>
#include <stdexcept>
#include <utility>

using DataType = ThorImplementation::DataType;
using json = nlohmann::json;

namespace {

constexpr const char* kAttentionQueryInputName = "query_input";
constexpr const char* kAttentionKeyInputName = "key_input";
constexpr const char* kAttentionValueInputName = "value_input";
constexpr const char* kAttentionScoreBiasInputName = "score_bias_input";
constexpr const char* kAttentionQuerySequenceLengthsInputName = "query_sequence_lengths";
constexpr const char* kAttentionKeyValueSequenceLengthsInputName = "key_value_sequence_lengths";
constexpr const char* kAttentionQueryRowPartitionInputName = "query_row_partition";
constexpr const char* kAttentionKeyValueRowPartitionInputName = "key_value_row_partition";
constexpr const char* kAttentionSyntheticQueryRowPartitionInputName = "__attention_uniform_query_row_partition";
constexpr const char* kAttentionSyntheticKeyValueRowPartitionInputName = "__attention_uniform_key_value_row_partition";
constexpr const char* kAttentionQueryRopePositionOffsetsInputName = "query_rope_position_offsets";
constexpr const char* kAttentionKeyRopePositionOffsetsInputName = "key_rope_position_offsets";
constexpr const char* kAttentionQueryRopePositionIdsInputName = "__attention_query_rope_position_ids";
constexpr const char* kAttentionKeyRopePositionIdsInputName = "__attention_key_rope_position_ids";
constexpr const char* kAttentionDropoutSeedInputName = "__attention_dropout_seed";
constexpr const char* kAttentionDropoutOffsetInputName = "__attention_dropout_offset";
constexpr const char* kAttentionResidualInputName = "residual_input";
constexpr const char* kAttentionOutputDropoutSeedInputName = "__attention_output_dropout_seed";
constexpr const char* kAttentionOutputDropoutSequenceInputName = "__attention_output_dropout_sequence";
constexpr ThorImplementation::DynamicExpressionVariantId kAttentionEvaluationVariant = 1;

std::string dtypeName(DataType dtype) {
    switch (dtype) {
        case DataType::FP16:
            return "fp16";
        case DataType::BF16:
            return "bf16";
        case DataType::FP32:
            return "fp32";
        case DataType::FP8_E4M3:
            return "fp8_e4m3";
        case DataType::FP8_E5M2:
            return "fp8_e5m2";
        default:
            return "dtype(" + std::to_string(static_cast<int>(dtype)) + ")";
    }
}

bool isStorageDType(DataType dtype) { return dtype == DataType::FP16 || dtype == DataType::BF16; }
bool isComputeDType(DataType dtype) { return dtype == DataType::FP32; }

uint64_t checkedMul(uint64_t a, uint64_t b, const char* what) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
        throw std::invalid_argument(std::string("Attention ") + what + " overflows uint64_t.");
    }
    return a * b;
}

constexpr uint64_t kMaxExactlyRepresentableFp32Integer = uint64_t{1} << 24;

bool ropeScalingUsesSequenceLength(ThorImplementation::RotaryScalingKind scalingKind) {
    return scalingKind == ThorImplementation::RotaryScalingKind::DynamicNTK ||
           scalingKind == ThorImplementation::RotaryScalingKind::LongRope;
}

std::optional<std::string> ropeFp32SequenceLengthValidationError(
    bool useRope,
    const ThorImplementation::RotaryPositionEmbeddingOptions& options,
    int64_t queryPositionOffset,
    int64_t keyPositionOffset,
    uint64_t maximumPossibleQuerySequenceLength,
    uint64_t maximumPossibleKeySequenceLength) {
    if (!useRope || !ropeScalingUsesSequenceLength(options.scaling_kind)) {
        return std::nullopt;
    }

    if (options.original_max_position_embeddings > kMaxExactlyRepresentableFp32Integer) {
        return "Attention Dynamic-NTK/LongRoPE currently requires original_max_position_embeddings <= 16777216 "
               "because RoPE sequence-length scaling uses FP32 metadata.";
    }

    const uint64_t positiveQueryPositionOffset =
        queryPositionOffset > 0 ? static_cast<uint64_t>(queryPositionOffset) : uint64_t{0};
    const uint64_t positiveKeyPositionOffset =
        keyPositionOffset > 0 ? static_cast<uint64_t>(keyPositionOffset) : uint64_t{0};
    const bool queryExtentTooLarge =
        positiveQueryPositionOffset > kMaxExactlyRepresentableFp32Integer ||
        maximumPossibleQuerySequenceLength > kMaxExactlyRepresentableFp32Integer - positiveQueryPositionOffset;
    const bool keyExtentTooLarge =
        positiveKeyPositionOffset > kMaxExactlyRepresentableFp32Integer ||
        maximumPossibleKeySequenceLength > kMaxExactlyRepresentableFp32Integer - positiveKeyPositionOffset;
    if (queryExtentTooLarge || keyExtentTooLarge) {
        return "Attention Dynamic-NTK/LongRoPE currently requires maximum possible sequence length plus positive "
               "Q/K position offset <= 16777216 because RoPE sequence-length scaling uses FP32 metadata. "
               "maximum_possible_query_sequence_length=" +
               std::to_string(maximumPossibleQuerySequenceLength) + ", positive_query_position_offset=" +
               std::to_string(positiveQueryPositionOffset) + ", maximum_possible_key_sequence_length=" +
               std::to_string(maximumPossibleKeySequenceLength) + ", positive_key_position_offset=" +
               std::to_string(positiveKeyPositionOffset) + ".";
    }

    return std::nullopt;
}

void requireRank2FeatureInput(const Thor::Tensor& tensor, const char* inputName = "feature input") {
    if (!tensor.isInitialized()) {
        throw std::invalid_argument(std::string("Attention ") + inputName + " tensor is not initialized.");
    }
    const std::vector<uint64_t> dims = tensor.getDimensions();
    if (dims.size() != 2) {
        throw std::invalid_argument(std::string("Attention ") + inputName +
                                    " must have rank 2 [sequence, features] at the API level.");
    }
    if (dims[0] == 0 || dims[1] == 0) {
        throw std::invalid_argument(std::string("Attention ") + inputName + " dimensions must be non-zero.");
    }
}

void requireSequenceLengthsInput(const Thor::Tensor& tensor, const char* inputName) {
    if (!tensor.isInitialized()) {
        throw std::invalid_argument(std::string("Attention ") + inputName + " tensor is not initialized.");
    }
    if (tensor.getDataType() != DataType::INT32) {
        throw std::invalid_argument(std::string("Attention ") + inputName + " must have dtype int32.");
    }
    if (tensor.getDimensions() != std::vector<uint64_t>{1}) {
        throw std::invalid_argument(std::string("Attention ") + inputName + " must have logical shape [1].");
    }
}

void requireRopePositionOffsetsInput(const Thor::Tensor& tensor, const char* inputName) {
    if (!tensor.isInitialized()) {
        throw std::invalid_argument(std::string("Attention ") + inputName + " tensor is not initialized.");
    }
    if (tensor.getDataType() != DataType::INT32) {
        throw std::invalid_argument(std::string("Attention ") + inputName + " must have dtype int32.");
    }
    if (tensor.getDimensions() != std::vector<uint64_t>{1}) {
        throw std::invalid_argument(std::string("Attention ") + inputName +
                                    " must have logical shape [1] (one origin per batch row at runtime).");
    }
}

void requireRaggedFeatureInput(const Thor::RaggedTensor& ragged, const char* inputName) {
    if (!ragged.isInitialized()) {
        throw std::invalid_argument(std::string("Attention ") + inputName + " is not initialized.");
    }
    if (ragged.getTrailingDimensions().size() != 1 || ragged.getTrailingDimensions().front() == 0) {
        throw std::invalid_argument(std::string("Attention ") + inputName +
                                    " must have one trailing feature dimension: packed values [max_total_values, features].");
    }
    if (!ThorImplementation::isCanonicalRowPartitionOffsetDataType(ragged.getOffsetsDataType())) {
        throw std::invalid_argument(std::string("Attention ") + inputName + " row partition must use uint32 or uint64 offsets.");
    }
    if (ragged.getBatchSize() == 0 || ragged.getBatchSize() > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument(std::string("Attention ") + inputName + " batch size must fit uint32 and be non-zero.");
    }
}


void requireScoreBiasInput(const Thor::Tensor& tensor,
                           uint32_t numHeads,
                           uint64_t querySequenceLength,
                           uint64_t keyValueSequenceLength,
                           DataType computeDataType) {
    if (!tensor.isInitialized()) {
        throw std::invalid_argument("Attention scoreBiasInput tensor is not initialized.");
    }
    const std::vector<uint64_t> dims = tensor.getDimensions();
    if (dims.size() != 3 || (dims[0] != 1 && dims[0] != numHeads) ||
        (dims[1] != querySequenceLength && dims[1] != 1) || (dims[2] != keyValueSequenceLength && dims[2] != 1)) {
        throw std::invalid_argument(
            "Attention scoreBiasInput dimensions must be [1|num_heads, 1|query_sequence, 1|key_value_sequence] at the API level.");
    }
    if (tensor.getDataType() != computeDataType) {
        throw std::invalid_argument("Attention scoreBiasInput dtype must match attention computeDataType.");
    }
}

std::vector<std::string> publicAttentionInputNames(bool useScoreBias,
                                                   bool useSequenceLengths,
                                                   bool queryRagged,
                                                   bool keyValueRagged,
                                                   bool useQueryRopePositionOffsets,
                                                   bool useKeyRopePositionOffsets,
                                                   bool useResidual,
                                                   const std::vector<std::string>& epilogueAuxInputNames) {
    std::vector<std::string> names{kAttentionQueryInputName, kAttentionKeyInputName, kAttentionValueInputName};
    if (useScoreBias) {
        names.push_back(kAttentionScoreBiasInputName);
    }
    if (useSequenceLengths) {
        names.push_back(kAttentionQuerySequenceLengthsInputName);
        names.push_back(kAttentionKeyValueSequenceLengthsInputName);
    }
    if (queryRagged) {
        names.push_back(kAttentionQueryRowPartitionInputName);
    }
    if (keyValueRagged) {
        names.push_back(kAttentionKeyValueRowPartitionInputName);
    }
    if (useQueryRopePositionOffsets) {
        names.push_back(kAttentionQueryRopePositionOffsetsInputName);
    }
    if (useKeyRopePositionOffsets) {
        names.push_back(kAttentionKeyRopePositionOffsetsInputName);
    }
    if (useResidual) {
        names.push_back(kAttentionResidualInputName);
    }
    names.insert(names.end(), epilogueAuxInputNames.begin(), epilogueAuxInputNames.end());
    return names;
}

Thor::CustomLayer::TensorMap publicAttentionInputInterface(const Thor::Tensor& queryInput,
                                                           const Thor::Tensor& keyInput,
                                                           const Thor::Tensor& valueInput,
                                                           const std::optional<Thor::Tensor>& scoreBiasInput,
                                                           const std::optional<Thor::Tensor>& querySequenceLengthsInput,
                                                           const std::optional<Thor::Tensor>& keyValueSequenceLengthsInput,
                                                           const std::optional<Thor::Tensor>& queryRopePositionOffsetsInput,
                                                           const std::optional<Thor::Tensor>& keyRopePositionOffsetsInput,
                                                           const std::optional<Thor::RaggedTensor>& raggedQueryInput,
                                                           const std::optional<Thor::RaggedTensor>& raggedKeyInput,
                                                           const std::optional<Thor::Tensor>& residualInput,
                                                           const std::vector<std::pair<std::string, Thor::Tensor>>& epilogueInputBindings) {
    Thor::CustomLayer::TensorMap inputInterface{{kAttentionQueryInputName, queryInput},
                                                {kAttentionKeyInputName, keyInput},
                                                {kAttentionValueInputName, valueInput}};
    if (scoreBiasInput.has_value()) {
        inputInterface[kAttentionScoreBiasInputName] = scoreBiasInput.value();
    }
    if (querySequenceLengthsInput.has_value()) {
        inputInterface[kAttentionQuerySequenceLengthsInputName] = querySequenceLengthsInput.value();
        inputInterface[kAttentionKeyValueSequenceLengthsInputName] = keyValueSequenceLengthsInput.value();
    }
    if (raggedQueryInput.has_value()) {
        inputInterface[kAttentionQueryRowPartitionInputName] = raggedQueryInput->getOffsets();
    }
    if (raggedKeyInput.has_value()) {
        inputInterface[kAttentionKeyValueRowPartitionInputName] = raggedKeyInput->getOffsets();
    }
    if (queryRopePositionOffsetsInput.has_value()) {
        inputInterface[kAttentionQueryRopePositionOffsetsInputName] = queryRopePositionOffsetsInput.value();
    }
    if (keyRopePositionOffsetsInput.has_value()) {
        inputInterface[kAttentionKeyRopePositionOffsetsInputName] = keyRopePositionOffsetsInput.value();
    }
    if (residualInput.has_value()) {
        inputInterface[kAttentionResidualInputName] = residualInput.value();
    }
    for (const auto& [name, tensor] : epilogueInputBindings) {
        inputInterface[name] = tensor;
    }
    return inputInterface;
}

ThorImplementation::Tensor makeUniformAttentionRowPartition(ThorImplementation::TensorPlacement placement,
                                                               DataType dtype,
                                                               uint64_t batch,
                                                               uint64_t sequenceLength,
                                                               Stream& stream) {
    if (!ThorImplementation::isCanonicalRowPartitionOffsetDataType(dtype)) {
        throw std::invalid_argument("Attention uniform row partition dtype must be UINT32 or UINT64.");
    }
    if (batch == 0 || sequenceLength == 0) {
        throw std::invalid_argument("Attention uniform row partition requires non-zero batch and sequence length.");
    }
    const uint64_t total = checkedMul(batch, sequenceLength, "uniform attention row partition capacity");
    if (dtype == DataType::UINT32 && total > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::overflow_error("Attention uniform UINT32 row partition exceeds representable capacity.");
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

uint64_t checkedDropoutOffsetAdvance(uint64_t batch, uint32_t numHeads, uint64_t querySequenceLength, uint64_t keyValueSequenceLength) {
    return checkedMul(checkedMul(batch, numHeads, "dropout offset batch-head count"),
                      checkedMul(querySequenceLength, keyValueSequenceLength, "dropout offset score count"),
                      "dropout offset advance");
}

class AttentionDropoutRuntimeState {
   public:
    AttentionDropoutRuntimeState(int64_t seed, int64_t initialOffset) : seed(seed), nextOffset(initialOffset) {
        if (initialOffset < 0) {
            throw std::invalid_argument("Attention sdpaDropoutOffset must be non-negative when SDPA dropout is enabled.");
        }
    }

    void setOffsetAdvance(uint64_t advance) {
        if (advance == 0) {
            advance = 1;
        }
        if (advance > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            throw std::overflow_error("Attention dropout offset advance exceeds int64_t range.");
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
            throw std::overflow_error("Attention automatic dropout offset advance would exceed int64_t range.");
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

std::shared_ptr<Thor::ParameterSpecification> makeParameter(const std::string& name,
                                                            const std::vector<uint64_t>& shape,
                                                            DataType dtype,
                                                            const std::shared_ptr<Thor::Initializer>& initializer,
                                                            const std::shared_ptr<Thor::Optimizer>& optimizer) {
    Thor::ParameterSpecification::Builder builder;
    builder.name(name).shape(shape).dtype(dtype).initializer(initializer->clone()).trainable(true);
    if (optimizer != nullptr) {
        std::shared_ptr<Thor::Optimizer> optimizer_copy = optimizer;
        builder.optimizer(optimizer_copy);
    }
    return std::make_shared<Thor::ParameterSpecification>(builder.build());
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
    throw std::runtime_error("Unknown AttentionMaskKind value.");
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
    throw std::runtime_error("Unknown Attention mask kind: " + value);
}

std::string rotaryScalingKindToString(ThorImplementation::RotaryScalingKind value) {
    switch (value) {
        case ThorImplementation::RotaryScalingKind::None:
            return "none";
        case ThorImplementation::RotaryScalingKind::Linear:
            return "linear";
        case ThorImplementation::RotaryScalingKind::DynamicNTK:
            return "dynamic_ntk";
        case ThorImplementation::RotaryScalingKind::Yarn:
            return "yarn";
        case ThorImplementation::RotaryScalingKind::LongRope:
            return "longrope";
        case ThorImplementation::RotaryScalingKind::Llama3:
            return "llama3";
    }
    throw std::runtime_error("Unknown RotaryScalingKind value.");
}

ThorImplementation::RotaryScalingKind rotaryScalingKindFromString(const std::string& value) {
    if (value == "none")
        return ThorImplementation::RotaryScalingKind::None;
    if (value == "linear")
        return ThorImplementation::RotaryScalingKind::Linear;
    if (value == "dynamic_ntk")
        return ThorImplementation::RotaryScalingKind::DynamicNTK;
    if (value == "yarn")
        return ThorImplementation::RotaryScalingKind::Yarn;
    if (value == "longrope")
        return ThorImplementation::RotaryScalingKind::LongRope;
    if (value == "llama3")
        return ThorImplementation::RotaryScalingKind::Llama3;
    throw std::runtime_error("Unknown Attention RoPE scaling kind: " + value);
}

json optionalDataTypeToJson(std::optional<DataType> value) {
    if (!value.has_value())
        return nullptr;
    return value.value();
}

std::optional<DataType> optionalDataTypeFromJson(const json& j) {
    if (j.is_null())
        return std::nullopt;
    return j.get<DataType>();
}

json ropeOptionsToJson(const ThorImplementation::RotaryPositionEmbeddingOptions& opts) {
    json j;
    j["sequence_axis"] = opts.sequence_axis;
    j["head_dim_axis"] = opts.head_dim_axis;
    j["rotary_dim"] = opts.rotary_dim;
    j["base"] = opts.base;
    j["position_offset"] = opts.position_offset;
    j["interleaved"] = opts.interleaved;
    j["inverse"] = opts.inverse;
    j["scaling_kind"] = rotaryScalingKindToString(opts.scaling_kind);
    j["scaling_factor"] = opts.scaling_factor;
    j["original_max_position_embeddings"] = opts.original_max_position_embeddings;
    j["attention_factor"] = opts.attention_factor.has_value() ? json(opts.attention_factor.value()) : json(nullptr);
    j["yarn_beta_fast"] = opts.yarn_beta_fast;
    j["yarn_beta_slow"] = opts.yarn_beta_slow;
    j["llama3_low_freq_factor"] = opts.llama3_low_freq_factor;
    j["llama3_high_freq_factor"] = opts.llama3_high_freq_factor;
    j["long_rope_short_factors"] = opts.long_rope_short_factors;
    j["long_rope_long_factors"] = opts.long_rope_long_factors;
    j["output_dtype"] = optionalDataTypeToJson(opts.output_dtype);
    j["compute_dtype"] = optionalDataTypeToJson(opts.compute_dtype);
    j["allow_in_place_materialization"] = opts.allow_in_place_materialization;
    return j;
}

ThorImplementation::RotaryPositionEmbeddingOptions ropeOptionsFromJson(const json& j) {
    ThorImplementation::RotaryPositionEmbeddingOptions opts;
    opts.sequence_axis = j.value("sequence_axis", opts.sequence_axis);
    opts.head_dim_axis = j.value("head_dim_axis", opts.head_dim_axis);
    opts.rotary_dim = j.value("rotary_dim", opts.rotary_dim);
    opts.base = j.value("base", opts.base);
    opts.position_offset = j.value("position_offset", opts.position_offset);
    opts.interleaved = j.value("interleaved", opts.interleaved);
    opts.inverse = j.value("inverse", opts.inverse);
    opts.scaling_kind = rotaryScalingKindFromString(j.value("scaling_kind", std::string("none")));
    opts.scaling_factor = j.value("scaling_factor", opts.scaling_factor);
    opts.original_max_position_embeddings = j.value("original_max_position_embeddings", opts.original_max_position_embeddings);
    if (j.contains("attention_factor") && !j.at("attention_factor").is_null()) {
        opts.attention_factor = j.at("attention_factor").get<double>();
    }
    opts.yarn_beta_fast = j.value("yarn_beta_fast", opts.yarn_beta_fast);
    opts.yarn_beta_slow = j.value("yarn_beta_slow", opts.yarn_beta_slow);
    opts.llama3_low_freq_factor = j.value("llama3_low_freq_factor", opts.llama3_low_freq_factor);
    opts.llama3_high_freq_factor = j.value("llama3_high_freq_factor", opts.llama3_high_freq_factor);
    opts.long_rope_short_factors = j.value("long_rope_short_factors", std::vector<double>{});
    opts.long_rope_long_factors = j.value("long_rope_long_factors", std::vector<double>{});
    if (j.contains("output_dtype")) {
        opts.output_dtype = optionalDataTypeFromJson(j.at("output_dtype"));
    }
    if (j.contains("compute_dtype")) {
        opts.compute_dtype = optionalDataTypeFromJson(j.at("compute_dtype"));
    }
    opts.allow_in_place_materialization = j.value("allow_in_place_materialization", opts.allow_in_place_materialization);
    return opts;
}

struct AttentionEpilogueInputDataTypes {
    std::optional<DataType> computeDataType;
    std::optional<DataType> outputDataType;
};

AttentionEpilogueInputDataTypes attentionEpilogueInputDataTypes(
    const ThorImplementation::Expression& expression,
    const std::string& inputName) {
    const ThorImplementation::PhysicalExpression physicalExpression = expression.expression();
    std::optional<AttentionEpilogueInputDataTypes> resolved;

    for (const ThorImplementation::ExprNode& node : physicalExpression.nodes) {
        if (node.op != ThorImplementation::ExprOp::INPUT) {
            continue;
        }
        if (node.input_slot >= physicalExpression.inputs.size()) {
            throw std::runtime_error("Attention epilogue input node has an invalid input slot.");
        }
        if (physicalExpression.inputs[node.input_slot].name != inputName) {
            continue;
        }

        const AttentionEpilogueInputDataTypes candidate{node.compute_dtype, node.output_dtype};
        if (resolved.has_value() &&
            (resolved->computeDataType != candidate.computeDataType ||
             resolved->outputDataType != candidate.outputDataType)) {
            throw std::runtime_error("Attention epilogue input '" + inputName +
                                     "' is used with inconsistent dtype annotations.");
        }
        resolved = candidate;
    }

    if (!resolved.has_value()) {
        throw std::runtime_error("Attention epilogue expression does not contain expected input '" + inputName + "'.");
    }
    return resolved.value();
}

ThorImplementation::DynamicExpression makeAttentionExpression(uint64_t querySequenceLength,
                                                              uint64_t keyValueSequenceLength,
                                                              uint64_t queryInputFeatures,
                                                              uint64_t keyInputFeatures,
                                                              uint64_t valueInputFeatures,
                                                              uint64_t outputFeatures,
                                                              uint32_t numHeads,
                                                              uint32_t numKeyValueHeads,
                                                              uint32_t headDim,
                                                              uint32_t valueDim,
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
                                                              bool useResidual,
                                                              bool useScoreBias,
                                                              bool useSequenceLengths,
                                                              bool queryRagged,
                                                              bool keyValueRagged,
                                                              bool useQueryRopePositionOffsets,
                                                              bool useKeyRopePositionOffsets,
                                                              uint64_t raggedBatchSize,
                                                              DataType queryRowPartitionDType,
                                                              DataType keyValueRowPartitionDType,
                                                              DataType inputDType,
                                                              DataType weightsDType,
                                                              DataType computeDType,
                                                              DataType outputDType,
                                                              std::optional<ThorImplementation::Expression> epilogue,
                                                              std::vector<std::string> epilogueAuxInputNames) {
    using ThorImplementation::AttentionOptions;
    using ThorImplementation::AttentionTensorLayout;
    using ThorImplementation::DynamicExpression;
    using ThorImplementation::DynamicExpressionBuild;
    using ThorImplementation::DynamicExpressionVariant;
    using ThorImplementation::Expression;
    using ThorImplementation::FusedEquation;
    using ThorImplementation::Tensor;
    using ThorImplementation::TensorScalarBinding;

    const bool useAnyRagged = queryRagged || keyValueRagged;
    std::vector<std::string> expectedInputs{kAttentionQueryInputName, kAttentionKeyInputName, kAttentionValueInputName};
    if (useScoreBias) {
        expectedInputs.push_back(kAttentionScoreBiasInputName);
    }
    if (useSequenceLengths) {
        expectedInputs.push_back(kAttentionQuerySequenceLengthsInputName);
        expectedInputs.push_back(kAttentionKeyValueSequenceLengthsInputName);
    }
    if (queryRagged) {
        expectedInputs.push_back(kAttentionQueryRowPartitionInputName);
    }
    if (keyValueRagged) {
        expectedInputs.push_back(kAttentionKeyValueRowPartitionInputName);
    }
    if (useQueryRopePositionOffsets) {
        expectedInputs.push_back(kAttentionQueryRopePositionOffsetsInputName);
    }
    if (useKeyRopePositionOffsets) {
        expectedInputs.push_back(kAttentionKeyRopePositionOffsetsInputName);
    }
    if (useResidual) {
        expectedInputs.push_back(kAttentionResidualInputName);
    }
    expectedInputs.push_back("query_weights");
    expectedInputs.push_back("key_weights");
    expectedInputs.push_back("value_weights");
    expectedInputs.push_back("output_weights");
    if (hasBias) {
        expectedInputs.push_back("query_bias");
        expectedInputs.push_back("key_bias");
        expectedInputs.push_back("value_bias");
        expectedInputs.push_back("output_bias");
    }
    expectedInputs.insert(expectedInputs.end(), epilogueAuxInputNames.begin(), epilogueAuxInputNames.end());

    return DynamicExpression(
        expectedInputs,
        {"feature_output"},
        [querySequenceLength,
         keyValueSequenceLength,
         queryInputFeatures,
         keyInputFeatures,
         valueInputFeatures,
         outputFeatures,
         numHeads,
         numKeyValueHeads,
         headDim,
         valueDim,
         hasBias,
         useRope,
         ropeInPlace,
         ropeOptions,
         queryRopePositionOffset,
         keyRopePositionOffset,
         maskKind,
         diagonalLeftBound,
         diagonalRightBound,
         useAlibiMask,
         attentionScale,
         sdpaDropoutProbability,
         dropoutSeed,
         dropoutOffset,
         outputDropoutProbability,
         outputDropoutSeed,
         useResidual,
         useScoreBias,
         useSequenceLengths,
         useAnyRagged,
         queryRagged,
         keyValueRagged,
         useQueryRopePositionOffsets,
         useKeyRopePositionOffsets,
         raggedBatchSize,
         queryRowPartitionDType,
         keyValueRowPartitionDType,
         inputDType,
         weightsDType,
         computeDType,
         outputDType,
         epilogue,
         epilogueAuxInputNames = std::move(epilogueAuxInputNames)](const DynamicExpression::TensorMap& inputs,
                      const DynamicExpression::TensorMap& outputs,
                      Stream& stream) -> DynamicExpressionBuild {
            Tensor queryInput = inputs.at(kAttentionQueryInputName);
            Tensor keyInput = inputs.at(kAttentionKeyInputName);
            Tensor valueInput = inputs.at(kAttentionValueInputName);
            const auto queryInputDims = queryInput.getDimensions();
            const uint64_t batch = useAnyRagged ? raggedBatchSize : (queryInputDims.empty() ? 0 : queryInputDims.front());
            const std::vector<uint64_t> expectedQueryInputDimensions =
                queryRagged ? std::vector<uint64_t>{querySequenceLength, queryInputFeatures}
                            : std::vector<uint64_t>{batch, querySequenceLength, queryInputFeatures};
            if (batch == 0 || queryInputDims != expectedQueryInputDimensions) {
                throw std::runtime_error(queryRagged
                                             ? "Attention runtime ragged query values must be [max_total_values, query_features]."
                                             : "Attention runtime query input must be [batch, query_sequence, query_features].");
            }
            if (queryInput.getDataType() != inputDType) {
                throw std::runtime_error("Attention runtime query input dtype does not match the API input dtype.");
            }

            const std::vector<uint64_t> runtimeFeatureOutputDimensions =
                queryRagged ? std::vector<uint64_t>{querySequenceLength, outputFeatures}
                            : std::vector<uint64_t>{batch, querySequenceLength, outputFeatures};
            if (outputs.contains("feature_output")) {
                const Tensor& featureOutput = outputs.at("feature_output");
                if (featureOutput.getDimensions() != runtimeFeatureOutputDimensions) {
                    throw std::runtime_error(queryRagged
                                                 ? "Attention runtime ragged feature output values must remain [max_total_values, output_features]."
                                                 : "Attention runtime feature output shape must remain [batch, query_sequence, output_features].");
                }
                if (featureOutput.getDataType() != outputDType) {
                    throw std::runtime_error("Attention runtime feature output dtype must match outputDataType.");
                }
                if (featureOutput.getPlacement() != queryInput.getPlacement()) {
                    throw std::runtime_error("Attention runtime feature output placement must match the query input placement.");
                }
            }

            const auto keyInputDims = keyInput.getDimensions();
            const auto valueInputDims = valueInput.getDimensions();
            const std::vector<uint64_t> expectedKeyInputDimensions =
                keyValueRagged ? std::vector<uint64_t>{keyValueSequenceLength, keyInputFeatures}
                               : std::vector<uint64_t>{batch, keyValueSequenceLength, keyInputFeatures};
            const std::vector<uint64_t> expectedValueInputDimensions =
                keyValueRagged ? std::vector<uint64_t>{keyValueSequenceLength, valueInputFeatures}
                               : std::vector<uint64_t>{batch, keyValueSequenceLength, valueInputFeatures};
            if (keyInputDims != expectedKeyInputDimensions) {
                throw std::runtime_error(keyValueRagged
                                             ? "Attention runtime ragged key values must be [max_total_values, key_features]."
                                             : "Attention runtime key input must be [batch, key_value_sequence, key_features].");
            }
            if (valueInputDims != expectedValueInputDimensions) {
                throw std::runtime_error(keyValueRagged
                                             ? "Attention runtime ragged value values must be [max_total_values, value_features]."
                                             : "Attention runtime value input must be [batch, key_value_sequence, value_features].");
            }
            if (keyInput.getDataType() != inputDType || valueInput.getDataType() != inputDType) {
                throw std::runtime_error("Attention runtime query/key/value input dtypes must match.");
            }
            if (keyInput.getPlacement() != queryInput.getPlacement() || valueInput.getPlacement() != queryInput.getPlacement()) {
                throw std::runtime_error("Attention runtime query/key/value inputs must use the same placement.");
            }

            for (const std::string& auxInputName : epilogueAuxInputNames) {
                const Tensor& auxTensor = inputs.at(auxInputName);
                if (auxTensor.getDimensions() != runtimeFeatureOutputDimensions) {
                    throw std::runtime_error("Attention epilogue auxiliary input '" + auxInputName +
                                             "' shape must match the attention output values shape.");
                }
                if (auxTensor.getDataType() != outputDType) {
                    throw std::runtime_error("Attention epilogue auxiliary input '" + auxInputName +
                                             "' dtype must match outputDataType.");
                }
                if (auxTensor.getPlacement() != queryInput.getPlacement()) {
                    throw std::runtime_error("Attention epilogue auxiliary input '" + auxInputName +
                                             "' placement must match the attention feature input placement.");
                }
            }

            std::optional<Tensor> scoreBiasInput;
            if (useScoreBias) {
                Tensor scoreBias = inputs.at(kAttentionScoreBiasInputName);
                const auto scoreBiasDims = scoreBias.getDimensions();
                if (scoreBiasDims.size() != 4 || scoreBiasDims[0] != batch ||
                    (scoreBiasDims[1] != 1 && scoreBiasDims[1] != numHeads) ||
                    (scoreBiasDims[2] != querySequenceLength && scoreBiasDims[2] != 1) ||
                    (scoreBiasDims[3] != keyValueSequenceLength && scoreBiasDims[3] != 1)) {
                    throw std::runtime_error(
                        "Attention runtime score_bias_input must be [batch, 1|num_heads, 1|query_sequence, 1|key_value_sequence].");
                }
                if (scoreBias.getDataType() != computeDType) {
                    throw std::runtime_error("Attention runtime score_bias_input dtype must match attention compute dtype.");
                }
                scoreBiasInput = std::move(scoreBias);
            }

            const uint64_t queryWidth = checkedMul(numHeads, headDim, "query projection width");
            const uint64_t keyWidth = checkedMul(numKeyValueHeads, headDim, "key projection width");
            const uint64_t valueWidth = checkedMul(numKeyValueHeads, valueDim, "value projection width");

            auto validateWeight = [&](const char* name, uint64_t in, uint64_t out) {
                const Tensor& w = inputs.at(name);
                if (w.getDimensions() != std::vector<uint64_t>{in, out}) {
                    throw std::runtime_error(std::string("Attention ") + name + " shape mismatch.");
                }
                if (w.getDataType() != weightsDType) {
                    throw std::runtime_error(std::string("Attention ") + name + " dtype mismatch.");
                }
            };
            validateWeight("query_weights", queryInputFeatures, queryWidth);
            validateWeight("key_weights", keyInputFeatures, keyWidth);
            validateWeight("value_weights", valueInputFeatures, valueWidth);
            validateWeight("output_weights", checkedMul(numHeads, valueDim, "output projection input width"), outputFeatures);

            if (hasBias) {
                auto validateBias = [&](const char* name, uint64_t width) {
                    const Tensor& b = inputs.at(name);
                    if (b.getDimensions() != std::vector<uint64_t>{width}) {
                        throw std::runtime_error(std::string("Attention ") + name + " shape mismatch.");
                    }
                    if (b.getDataType() != weightsDType) {
                        throw std::runtime_error(std::string("Attention ") + name + " dtype mismatch.");
                    }
                };
                validateBias("query_bias", queryWidth);
                validateBias("key_bias", keyWidth);
                validateBias("value_bias", valueWidth);
                validateBias("output_bias", outputFeatures);
            }

            auto normalizeSequenceLengths = [&](const char* inputName) -> Tensor {
                Tensor seq = inputs.at(inputName);
                if (seq.getDataType() != DataType::INT32) {
                    throw std::runtime_error(std::string("Attention ") + inputName + " dtype must be INT32.");
                }
                const auto seqDims = seq.getDimensions();
                if (seqDims == std::vector<uint64_t>{batch, 1}) {
                    seq.reshape({batch});
                } else if (seqDims != std::vector<uint64_t>{batch}) {
                    throw std::runtime_error(std::string("Attention ") + inputName + " shape must be [batch] or [batch, 1].");
                }
                return seq;
            };

            std::optional<Tensor> querySequenceLengths;
            std::optional<Tensor> keyValueSequenceLengths;
            if (useSequenceLengths) {
                querySequenceLengths = normalizeSequenceLengths(kAttentionQuerySequenceLengthsInputName);
                keyValueSequenceLengths = normalizeSequenceLengths(kAttentionKeyValueSequenceLengthsInputName);
            }

            auto normalizeRaggedOffsets = [&](const char* inputName, DataType expectedDType) -> Tensor {
                Tensor offsets = inputs.at(inputName);
                if (!ThorImplementation::isCanonicalRowPartitionOffsetDataType(offsets.getDataType()) ||
                    offsets.getDataType() != expectedDType) {
                    throw std::runtime_error(std::string("Attention ") + inputName +
                                             " dtype must match its canonical UINT32/UINT64 API row-partition dtype.");
                }
                if (offsets.getDimensions() != std::vector<uint64_t>{batch + 1}) {
                    throw std::runtime_error(std::string("Attention ") + inputName + " shape must be exactly [batch+1].");
                }
                return offsets;
            };

            std::optional<Tensor> queryRaggedOffsets;
            std::optional<Tensor> keyValueRaggedOffsets;
            if (queryRagged) {
                queryRaggedOffsets = normalizeRaggedOffsets(kAttentionQueryRowPartitionInputName, queryRowPartitionDType);
            }
            if (keyValueRagged) {
                keyValueRaggedOffsets = normalizeRaggedOffsets(kAttentionKeyValueRowPartitionInputName, keyValueRowPartitionDType);
            }

            std::optional<Tensor> syntheticQueryRaggedOffsets;
            std::optional<Tensor> syntheticKeyValueRaggedOffsets;
            if (useAnyRagged && !queryRagged) {
                syntheticQueryRaggedOffsets = makeUniformAttentionRowPartition(
                    queryInput.getPlacement(), queryRowPartitionDType, batch, querySequenceLength, stream);
            }
            if (useAnyRagged && !keyValueRagged) {
                syntheticKeyValueRaggedOffsets = makeUniformAttentionRowPartition(
                    queryInput.getPlacement(), keyValueRowPartitionDType, batch, keyValueSequenceLength, stream);
            }

            auto normalizeRopePositionOffsets = [&](const char* inputName) -> Tensor {
                Tensor origins = inputs.at(inputName);
                if (origins.getDataType() != DataType::INT32) {
                    throw std::runtime_error(std::string("Attention ") + inputName + " dtype must be INT32.");
                }
                const auto dims = origins.getDimensions();
                if (dims == std::vector<uint64_t>{batch, 1}) {
                    origins.reshape({batch});
                } else if (dims != std::vector<uint64_t>{batch}) {
                    throw std::runtime_error(std::string("Attention ") + inputName + " shape must be [batch] or [batch, 1].");
                }
                return origins;
            };

            std::optional<Tensor> queryRopePositionOffsets;
            std::optional<Tensor> keyRopePositionOffsets;
            if (useQueryRopePositionOffsets) {
                queryRopePositionOffsets = normalizeRopePositionOffsets(kAttentionQueryRopePositionOffsetsInputName);
            }
            if (useKeyRopePositionOffsets) {
                keyRopePositionOffsets = normalizeRopePositionOffsets(kAttentionKeyRopePositionOffsetsInputName);
            }

            std::optional<Tensor> queryRopePositionIds;
            std::optional<Tensor> keyRopePositionIds;
            if (queryRagged && useRope) {
                queryRopePositionIds.emplace(
                    queryInput.getPlacement(), ThorImplementation::TensorDescriptor(DataType::FP32, {querySequenceLength}));
            }
            if (keyValueRagged && useRope) {
                keyRopePositionIds.emplace(
                    queryInput.getPlacement(), ThorImplementation::TensorDescriptor(DataType::FP32, {keyValueSequenceLength}));
            }

            Expression rawQ = Expression::input(kAttentionQueryInputName, inputDType, inputDType);
            Expression rawK = Expression::input(kAttentionKeyInputName, inputDType, inputDType);
            Expression rawV = Expression::input(kAttentionValueInputName, inputDType, inputDType);
            Expression qx = rawQ;
            Expression kx = rawK;
            Expression vx = rawV;
            if (queryRagged) {
                Expression offsets = Expression::input(
                    kAttentionQueryRowPartitionInputName, queryRowPartitionDType, queryRowPartitionDType);
                qx = qx.withRaggedRuntimeExtent(offsets, batch, querySequenceLength, queryInputFeatures);
            } else {
                qx = qx.reshape({batch * querySequenceLength, queryInputFeatures});
            }
            if (keyValueRagged) {
                Expression offsets = Expression::input(
                    kAttentionKeyValueRowPartitionInputName, keyValueRowPartitionDType, keyValueRowPartitionDType);
                kx = kx.withRaggedRuntimeExtent(offsets, batch, keyValueSequenceLength, keyInputFeatures);
                vx = vx.withRaggedRuntimeExtent(offsets, batch, keyValueSequenceLength, valueInputFeatures);
            } else {
                kx = kx.reshape({batch * keyValueSequenceLength, keyInputFeatures});
                vx = vx.reshape({batch * keyValueSequenceLength, valueInputFeatures});
            }
            std::optional<Expression> scoreBiasExpr;
            if (useScoreBias) {
                scoreBiasExpr = Expression::input(kAttentionScoreBiasInputName, computeDType, computeDType);
            }

            Expression qw = Expression::input("query_weights", weightsDType, weightsDType);
            Expression kw = Expression::input("key_weights", weightsDType, weightsDType);
            Expression vw = Expression::input("value_weights", weightsDType, weightsDType);

            Expression q = Expression::matmul(
                qx, qw, false, false, computeDType, outputDType,
                queryRagged ? std::optional<uint64_t>(querySequenceLength) : std::nullopt);
            Expression k = Expression::matmul(
                kx, kw, false, false, computeDType, outputDType,
                keyValueRagged ? std::optional<uint64_t>(keyValueSequenceLength) : std::nullopt);
            Expression v = Expression::matmul(
                vx, vw, false, false, computeDType, outputDType,
                keyValueRagged ? std::optional<uint64_t>(keyValueSequenceLength) : std::nullopt);
            if (hasBias) {
                q = q + Expression::input("query_bias", weightsDType, weightsDType);
                k = k + Expression::input("key_bias", weightsDType, weightsDType);
                v = v + Expression::input("value_bias", weightsDType, weightsDType);
            }

            if (queryRagged) {
                q = q.reshape({querySequenceLength, numHeads, headDim}).withOutputDType(outputDType);
            } else {
                q = q.reshape({batch, querySequenceLength, numHeads, headDim}).withOutputDType(outputDType);
            }
            if (keyValueRagged) {
                k = k.reshape({keyValueSequenceLength, numKeyValueHeads, headDim}).withOutputDType(outputDType);
                v = v.reshape({keyValueSequenceLength, numKeyValueHeads, valueDim}).withOutputDType(outputDType);
            } else {
                k = k.reshape({batch, keyValueSequenceLength, numKeyValueHeads, headDim}).withOutputDType(outputDType);
                v = v.reshape({batch, keyValueSequenceLength, numKeyValueHeads, valueDim}).withOutputDType(outputDType);
            }


            if (useRope) {
                ThorImplementation::RotaryPositionEmbeddingOptions queryOpts = ropeOptions;
                ThorImplementation::RotaryPositionEmbeddingOptions keyOpts = ropeOptions;
                queryOpts.position_offset = queryRagged ? 0 : queryRopePositionOffset;
                keyOpts.position_offset = keyValueRagged ? 0 : keyRopePositionOffset;
                queryOpts.sequence_axis = queryRagged ? 0 : 1;
                keyOpts.sequence_axis = keyValueRagged ? 0 : 1;
                queryOpts.head_dim_axis = queryRagged ? 2 : 3;
                keyOpts.head_dim_axis = keyValueRagged ? 2 : 3;
                queryOpts.allow_in_place_materialization = queryRagged ? false : ropeInPlace;
                keyOpts.allow_in_place_materialization = keyValueRagged ? false : ropeInPlace;
                if (!queryOpts.compute_dtype.has_value()) queryOpts.compute_dtype = computeDType;
                if (!keyOpts.compute_dtype.has_value()) keyOpts.compute_dtype = computeDType;
                if (!queryOpts.output_dtype.has_value()) queryOpts.output_dtype = outputDType;
                if (!keyOpts.output_dtype.has_value()) keyOpts.output_dtype = outputDType;

                const bool scalingNeedsLogicalSequenceLength =
                    ropeOptions.scaling_kind == ThorImplementation::RotaryScalingKind::DynamicNTK ||
                    ropeOptions.scaling_kind == ThorImplementation::RotaryScalingKind::LongRope;

                if (useAnyRagged) {
                    std::optional<Expression> queryPositions;
                    std::optional<Expression> keyPositions;
                    if (queryRagged) {
                        queryPositions = Expression::input(
                            kAttentionQueryRopePositionIdsInputName, DataType::FP32, DataType::FP32);
                    }
                    if (keyValueRagged) {
                        keyPositions = Expression::input(
                            kAttentionKeyRopePositionIdsInputName, DataType::FP32, DataType::FP32);
                    }

                    std::optional<Expression> absoluteExtent;
                    if (scalingNeedsLogicalSequenceLength) {
                        const double positiveQueryOffset = static_cast<double>(std::max<int64_t>(0, queryRopePositionOffset));
                        const double positiveKeyOffset = static_cast<double>(std::max<int64_t>(0, keyRopePositionOffset));
                        auto logicalRaggedPositionMax = [&](const Expression& positions,
                                                             const char* offsetsInputName,
                                                             DataType offsetsDataType) {
                            Expression offsets = Expression::input(offsetsInputName, offsetsDataType, offsetsDataType);
                            // Position-id storage has the same packed capacity as the ragged values, but only
                            // [0, offsets[B]) is logical. Reduce each row through the explicit partition first,
                            // then reduce the dense [B] result. This keeps the full-capacity reduction from
                            // depending on any incidental contents in inactive position-id storage.
                            return positions.segmentedReduceMax(offsets).reduce_max({0}, {});
                        };
                        Expression queryExtent = queryRagged
                            ? logicalRaggedPositionMax(queryPositions.value(),
                                                       kAttentionQueryRowPartitionInputName,
                                                       queryRowPartitionDType) +
                                  Expression::constantScalar(1.0)
                            : Expression::constantScalar(static_cast<double>(querySequenceLength) + positiveQueryOffset);
                        Expression keyExtent = keyValueRagged
                            ? logicalRaggedPositionMax(keyPositions.value(),
                                                       kAttentionKeyValueRowPartitionInputName,
                                                       keyValueRowPartitionDType) +
                                  Expression::constantScalar(1.0)
                            : Expression::constantScalar(static_cast<double>(keyValueSequenceLength) + positiveKeyOffset);
                        absoluteExtent = queryExtent.max(keyExtent).max(Expression::constantScalar(1.0));
                    }

                    if (queryRagged) {
                        if (absoluteExtent.has_value()) {
                            q = q.rotaryPositionEmbeddingWithPositionIdsAndEffectiveSequenceLength(
                                queryPositions.value(), absoluteExtent.value(), queryOpts);
                        } else {
                            q = q.rotaryPositionEmbeddingWithPositionIds(queryPositions.value(), queryOpts);
                        }
                    } else if (absoluteExtent.has_value()) {
                        const double positiveQueryOffset = static_cast<double>(std::max<int64_t>(0, queryRopePositionOffset));
                        q = q.rotaryPositionEmbeddingWithEffectiveSequenceLength(
                            absoluteExtent.value() - Expression::constantScalar(positiveQueryOffset), queryOpts);
                    } else {
                        q = q.rotaryPositionEmbedding(queryOpts);
                    }

                    if (keyValueRagged) {
                        if (absoluteExtent.has_value()) {
                            k = k.rotaryPositionEmbeddingWithPositionIdsAndEffectiveSequenceLength(
                                keyPositions.value(), absoluteExtent.value(), keyOpts);
                        } else {
                            k = k.rotaryPositionEmbeddingWithPositionIds(keyPositions.value(), keyOpts);
                        }
                    } else if (absoluteExtent.has_value()) {
                        const double positiveKeyOffset = static_cast<double>(std::max<int64_t>(0, keyRopePositionOffset));
                        k = k.rotaryPositionEmbeddingWithEffectiveSequenceLength(
                            absoluteExtent.value() - Expression::constantScalar(positiveKeyOffset), keyOpts);
                    } else {
                        k = k.rotaryPositionEmbedding(keyOpts);
                    }
                } else {
                    std::optional<Expression> queryEffectiveSequenceLength;
                    std::optional<Expression> keyEffectiveSequenceLength;
                    const double positiveQueryOffset = static_cast<double>(std::max<int64_t>(0, queryRopePositionOffset));
                    const double positiveKeyOffset = static_cast<double>(std::max<int64_t>(0, keyRopePositionOffset));
                    if (scalingNeedsLogicalSequenceLength && useSequenceLengths) {
                        Expression qLengths = Expression::input(
                            kAttentionQuerySequenceLengthsInputName, DataType::INT32, DataType::INT32);
                        Expression kvLengths = Expression::input(
                            kAttentionKeyValueSequenceLengthsInputName, DataType::INT32, DataType::INT32);
                        Expression qMax = qLengths.cast(DataType::FP32).reduce_max({0}, {});
                        Expression kvMax = kvLengths.cast(DataType::FP32).reduce_max({0}, {});
                        Expression absoluteExtent = (qMax + Expression::constantScalar(positiveQueryOffset))
                                                        .max(kvMax + Expression::constantScalar(positiveKeyOffset));
                        queryEffectiveSequenceLength = absoluteExtent - Expression::constantScalar(positiveQueryOffset);
                        keyEffectiveSequenceLength = absoluteExtent - Expression::constantScalar(positiveKeyOffset);
                    } else if (scalingNeedsLogicalSequenceLength) {
                        const double queryAbsoluteExtent = static_cast<double>(querySequenceLength) + positiveQueryOffset;
                        const double keyAbsoluteExtent = static_cast<double>(keyValueSequenceLength) + positiveKeyOffset;
                        if (queryAbsoluteExtent != keyAbsoluteExtent) {
                            const double absoluteExtentValue = std::max(queryAbsoluteExtent, keyAbsoluteExtent);
                            if (queryAbsoluteExtent != absoluteExtentValue) {
                                queryEffectiveSequenceLength =
                                    Expression::constantScalar(absoluteExtentValue - positiveQueryOffset);
                            }
                            if (keyAbsoluteExtent != absoluteExtentValue) {
                                keyEffectiveSequenceLength =
                                    Expression::constantScalar(absoluteExtentValue - positiveKeyOffset);
                            }
                        }
                    }
                    q = queryEffectiveSequenceLength.has_value()
                            ? q.rotaryPositionEmbeddingWithEffectiveSequenceLength(queryEffectiveSequenceLength.value(), queryOpts)
                            : q.rotaryPositionEmbedding(queryOpts);
                    k = keyEffectiveSequenceLength.has_value()
                            ? k.rotaryPositionEmbeddingWithEffectiveSequenceLength(keyEffectiveSequenceLength.value(), keyOpts)
                            : k.rotaryPositionEmbedding(keyOpts);
                }
            }

            AttentionOptions options;
            options.q_layout = AttentionTensorLayout::BSHD;
            options.k_layout = AttentionTensorLayout::BSHD;
            options.v_layout = AttentionTensorLayout::BSHD;
            options.o_layout = AttentionTensorLayout::BSHD;
            options.mask_kind = maskKind;
            options.diagonal_left_bound = diagonalLeftBound;
            options.diagonal_right_bound = diagonalRightBound;
            options.use_alibi_mask = useAlibiMask;
            options.compute_dtype = computeDType;
            options.output_dtype = outputDType;
            if (attentionScale.has_value()) {
                options.attention_scale = static_cast<float>(attentionScale.value());
            }

            auto buildSdpa = [&](bool enableDropout) -> Expression {
                AttentionOptions activeOptions = options;
                activeOptions.dropout_probability = enableDropout ? sdpaDropoutProbability : 0.0f;

                if (useAnyRagged) {
                    const char* queryOffsetsName = queryRagged
                        ? kAttentionQueryRowPartitionInputName
                        : kAttentionSyntheticQueryRowPartitionInputName;
                    const char* keyValueOffsetsName = keyValueRagged
                        ? kAttentionKeyValueRowPartitionInputName
                        : kAttentionSyntheticKeyValueRowPartitionInputName;
                    Expression qRaggedOffsetsExpr = Expression::input(
                        queryOffsetsName, queryRowPartitionDType, queryRowPartitionDType);
                    Expression kvRaggedOffsetsExpr = Expression::input(
                        keyValueOffsetsName, keyValueRowPartitionDType, keyValueRowPartitionDType);
                    if (enableDropout) {
                        Expression dropoutSeedExpr =
                            Expression::tensorRuntimeScalar(kAttentionDropoutSeedInputName, DataType::INT64, DataType::INT64);
                        Expression dropoutOffsetExpr =
                            Expression::tensorRuntimeScalar(kAttentionDropoutOffsetInputName, DataType::INT64, DataType::INT64);
                        if (scoreBiasExpr.has_value()) {
                            return Expression::scaledDotProductAttentionRagged(q,
                                                                               k,
                                                                               v,
                                                                               scoreBiasExpr.value(),
                                                                               qRaggedOffsetsExpr,
                                                                               kvRaggedOffsetsExpr,
                                                                               dropoutSeedExpr,
                                                                               dropoutOffsetExpr,
                                                                               activeOptions)
                                .withOutputDType(outputDType);
                        }
                        return Expression::scaledDotProductAttentionRagged(q,
                                                                           k,
                                                                           v,
                                                                           qRaggedOffsetsExpr,
                                                                           kvRaggedOffsetsExpr,
                                                                           dropoutSeedExpr,
                                                                           dropoutOffsetExpr,
                                                                           activeOptions)
                            .withOutputDType(outputDType);
                    }
                    if (scoreBiasExpr.has_value()) {
                        return Expression::scaledDotProductAttentionRagged(q,
                                                                           k,
                                                                           v,
                                                                           scoreBiasExpr.value(),
                                                                           qRaggedOffsetsExpr,
                                                                           kvRaggedOffsetsExpr,
                                                                           activeOptions)
                            .withOutputDType(outputDType);
                    }
                    return Expression::scaledDotProductAttentionRagged(
                               q, k, v, qRaggedOffsetsExpr, kvRaggedOffsetsExpr, activeOptions)
                        .withOutputDType(outputDType);
                }
                if (useSequenceLengths) {
                    Expression qSeqLenExpr = Expression::input(kAttentionQuerySequenceLengthsInputName, DataType::INT32, DataType::INT32);
                    Expression kvSeqLenExpr = Expression::input(kAttentionKeyValueSequenceLengthsInputName, DataType::INT32, DataType::INT32);
                    if (enableDropout) {
                        Expression dropoutSeedExpr = Expression::tensorRuntimeScalar(kAttentionDropoutSeedInputName, DataType::INT64, DataType::INT64);
                        Expression dropoutOffsetExpr = Expression::tensorRuntimeScalar(kAttentionDropoutOffsetInputName, DataType::INT64, DataType::INT64);
                        if (scoreBiasExpr.has_value()) {
                            return Expression::scaledDotProductAttention(q,
                                                                         k,
                                                                         v,
                                                                         scoreBiasExpr.value(),
                                                                         qSeqLenExpr,
                                                                         kvSeqLenExpr,
                                                                         dropoutSeedExpr,
                                                                         dropoutOffsetExpr,
                                                                         activeOptions)
                                .withOutputDType(outputDType);
                        }
                        return Expression::scaledDotProductAttention(
                                   q, k, v, qSeqLenExpr, kvSeqLenExpr, dropoutSeedExpr, dropoutOffsetExpr, activeOptions)
                            .withOutputDType(outputDType);
                    }
                    if (scoreBiasExpr.has_value()) {
                        return Expression::scaledDotProductAttention(
                                   q, k, v, scoreBiasExpr.value(), qSeqLenExpr, kvSeqLenExpr, activeOptions)
                            .withOutputDType(outputDType);
                    }
                    return Expression::scaledDotProductAttention(q, k, v, qSeqLenExpr, kvSeqLenExpr, activeOptions)
                        .withOutputDType(outputDType);
                }
                if (enableDropout) {
                    Expression dropoutSeedExpr = Expression::tensorRuntimeScalar(kAttentionDropoutSeedInputName, DataType::INT64, DataType::INT64);
                    Expression dropoutOffsetExpr = Expression::tensorRuntimeScalar(kAttentionDropoutOffsetInputName, DataType::INT64, DataType::INT64);
                    if (scoreBiasExpr.has_value()) {
                        return Expression::scaledDotProductAttentionWithDropout(
                                   q, k, v, scoreBiasExpr.value(), dropoutSeedExpr, dropoutOffsetExpr, activeOptions)
                            .withOutputDType(outputDType);
                    }
                    return Expression::scaledDotProductAttentionWithDropout(
                               q, k, v, dropoutSeedExpr, dropoutOffsetExpr, activeOptions)
                        .withOutputDType(outputDType);
                }
                if (scoreBiasExpr.has_value()) {
                    return Expression::scaledDotProductAttention(q, k, v, scoreBiasExpr.value(), activeOptions)
                        .withOutputDType(outputDType);
                }
                return Expression::scaledDotProductAttention(q, k, v, activeOptions).withOutputDType(outputDType);
            };

            const std::vector<uint64_t> foldedOutputDimensions =
                {queryRagged ? querySequenceLength : batch * querySequenceLength, outputFeatures};
            auto buildProjectedOutput = [&](Expression attn, bool enableOutputDropout) -> Expression {
                const uint64_t mergedWidth = checkedMul(numHeads, valueDim, "merged head width");
                Expression merged = attn.reshape(
                    {queryRagged ? querySequenceLength : batch * querySequenceLength, mergedWidth});
                if (queryRagged) {
                    Expression offsets = Expression::input(
                        kAttentionQueryRowPartitionInputName, queryRowPartitionDType, queryRowPartitionDType);
                    merged = merged.withRaggedRuntimeExtent(
                        offsets, batch, querySequenceLength, mergedWidth);
                }
                Expression outputWeights = Expression::input("output_weights", weightsDType, weightsDType);
                Expression out = Expression::matmul(
                    merged,
                    outputWeights,
                    false,
                    false,
                    computeDType,
                    outputDType,
                    queryRagged ? std::optional<uint64_t>(querySequenceLength) : std::nullopt);
                if (hasBias) {
                    out = out + Expression::input("output_bias", weightsDType, weightsDType);
                }

                std::optional<Expression> flattenedResidual;
                if (useResidual) {
                    flattenedResidual = Expression::input(kAttentionResidualInputName, outputDType, outputDType)
                                            .reshape(foldedOutputDimensions);
                }

                if (enableOutputDropout && outputDropoutProbability > 0.0f) {
                    ThorImplementation::CudaKernelExpression outputDropout =
                        ThorImplementation::makeDropOutPostOpKernel(outputDType,
                                                                  outputDropoutProbability,
                                                                  useResidual,
                                                                  queryRagged,
                                                                  queryRowPartitionDType,
                                                                  queryRagged ? batch : 0,
                                                                  outputFeatures,
                                                                  "Attention output");
                    std::unordered_map<std::string, Expression> dropoutInputs{
                        {"projected", out},
                        {"seed", Expression::tensorRuntimeScalar(
                                     kAttentionOutputDropoutSeedInputName, DataType::INT64, DataType::INT64)},
                        {"sequence", Expression::tensorRuntimeScalar(
                                         kAttentionOutputDropoutSequenceInputName, DataType::INT64, DataType::INT64)},
                    };
                    if (flattenedResidual.has_value()) {
                        dropoutInputs.emplace("residual", flattenedResidual.value());
                    }
                    if (queryRagged) {
                        dropoutInputs.emplace(
                            "offsets",
                            Expression::input(kAttentionQueryRowPartitionInputName, queryRowPartitionDType, queryRowPartitionDType));
                    }
                    ThorImplementation::Outputs dropoutOutputs = outputDropout.apply(dropoutInputs);
                    const auto& namedDropoutOutputs = dropoutOutputs.namedOutputs();
                    if (namedDropoutOutputs.size() != 1 || namedDropoutOutputs.front().name != "output") {
                        throw std::logic_error("Attention output-dropout kernel produced an unexpected output interface.");
                    }
                    out = Expression::fromPhysicalNode(
                        dropoutOutputs.expression(), namedDropoutOutputs.front().node_idx);
                } else if (flattenedResidual.has_value()) {
                    // With no active output dropout this add is intentionally kept adjacent to the
                    // projection so EquationCompiler can lower it into the output GEMM beta/residual path.
                    out = out + flattenedResidual.value();
                }

                if (epilogue.has_value()) {
                    // Attention exposes [B, Q, O], while the output projection is physically [B*Q, O].
                    // Apply pointwise epilogues in the projection geometry so matmul + residual can lower
                    // to the existing GEMM beta-add path with no materialized projection result or separate
                    // elementwise launch. Restore the public rank only after the epilogue, exactly as the
                    // prefix-preserving FullyConnected path does. Shape-changing epilogues are rejected.
                    Expression effectiveEpilogue = epilogue.value();
                    for (const std::string& auxInputName : epilogueAuxInputNames) {
                        const AttentionEpilogueInputDataTypes inputDataTypes =
                            attentionEpilogueInputDataTypes(effectiveEpilogue, auxInputName);
                        Expression flattenedAuxInput =
                            Expression::input(auxInputName, inputDataTypes.computeDataType, inputDataTypes.outputDataType)
                                .reshape(foldedOutputDimensions);
                        effectiveEpilogue = effectiveEpilogue.substituteInput(auxInputName, flattenedAuxInput);
                    }
                    out = Thor::Attention::applyEpilogue(out, effectiveEpilogue);
                }
                if (!queryRagged) {
                    out = out.reshape({batch, querySequenceLength, outputFeatures});
                }
                if (!epilogue.has_value()) {
                    // Preserve the historical non-epilogue expression exactly. Epilogue expressions must resolve
                    // to the declared storage dtype on their own; do not hide an incompatible epilogue result
                    // behind an implicit final output conversion.
                    out = out.withOutputDType(outputDType);
                }
                return out;
            };

            auto expressionOutputs = Expression::outputs(
                {{"feature_output", buildProjectedOutput(buildSdpa(sdpaDropoutProbability > 0.0f), outputDropoutProbability > 0.0f)}});
            std::shared_ptr<FusedEquation> evaluationEquation;
            if (sdpaDropoutProbability > 0.0f || outputDropoutProbability > 0.0f) {
                auto validationExpressionOutputs =
                    Expression::outputs({{"feature_output", buildProjectedOutput(buildSdpa(false), false)}});
                evaluationEquation = std::make_shared<FusedEquation>(
                    FusedEquation::compile(validationExpressionOutputs.physicalOutputs(), stream.getGpuNum()));
            }

            DynamicExpression::TensorMap stampInputs = inputs;
            // Per-row RoPE origins are consumed by the pre-forward hook that materializes the
            // explicit position-id tensors below. They are public DynamicExpression inputs so
            // the hook can observe the current batch values, but they are not roots of the
            // fused attention equation itself. Do not forward hook-only inputs to FusedEquation,
            // which intentionally rejects named inputs that its compiled expression does not use.
            if (useQueryRopePositionOffsets) {
                stampInputs.erase(kAttentionQueryRopePositionOffsetsInputName);
            }
            if (useKeyRopePositionOffsets) {
                stampInputs.erase(kAttentionKeyRopePositionOffsetsInputName);
            }
            if (useScoreBias) {
                stampInputs[kAttentionScoreBiasInputName] = scoreBiasInput.value();
            }
            if (useSequenceLengths) {
                stampInputs[kAttentionQuerySequenceLengthsInputName] = querySequenceLengths.value();
                stampInputs[kAttentionKeyValueSequenceLengthsInputName] = keyValueSequenceLengths.value();
            }
            if (useAnyRagged) {
                stampInputs[queryRagged ? kAttentionQueryRowPartitionInputName
                                        : kAttentionSyntheticQueryRowPartitionInputName] =
                    queryRagged ? queryRaggedOffsets.value() : syntheticQueryRaggedOffsets.value();
                stampInputs[keyValueRagged ? kAttentionKeyValueRowPartitionInputName
                                           : kAttentionSyntheticKeyValueRowPartitionInputName] =
                    keyValueRagged ? keyValueRaggedOffsets.value() : syntheticKeyValueRaggedOffsets.value();
            }

            std::function<void(Stream&)> ropePreForwardHook;
            if (useAnyRagged && useRope) {
                if (queryRagged) stampInputs[kAttentionQueryRopePositionIdsInputName] = queryRopePositionIds.value();
                if (keyValueRagged) stampInputs[kAttentionKeyRopePositionIdsInputName] = keyRopePositionIds.value();
                std::optional<Tensor> queryOffsets = queryRaggedOffsets;
                std::optional<Tensor> keyOffsets = keyValueRaggedOffsets;
                std::optional<Tensor> queryPositions = queryRopePositionIds;
                std::optional<Tensor> keyPositions = keyRopePositionIds;
                std::optional<Tensor> queryOrigins = queryRopePositionOffsets;
                std::optional<Tensor> keyOrigins = keyRopePositionOffsets;
                ropePreForwardHook = [queryRagged,
                                      keyValueRagged,
                                      queryOffsets,
                                      keyOffsets,
                                      queryOrigins,
                                      keyOrigins,
                                      queryPositions,
                                      keyPositions,
                                      queryRopePositionOffset,
                                      keyRopePositionOffset,
                                      batch,
                                      querySequenceLength,
                                      keyValueSequenceLength](Stream& runStream) mutable {
                    if (queryRagged) {
                        ThorImplementation::rowPartitionOffsetsToRopePositionIds(
                            queryOffsets.value(),
                            queryOrigins.has_value() ? &queryOrigins.value() : nullptr,
                            queryRopePositionOffset,
                            queryPositions.value(),
                            batch,
                            querySequenceLength,
                            runStream);
                    }
                    if (keyValueRagged) {
                        ThorImplementation::rowPartitionOffsetsToRopePositionIds(
                            keyOffsets.value(),
                            keyOrigins.has_value() ? &keyOrigins.value() : nullptr,
                            keyRopePositionOffset,
                            keyPositions.value(),
                            batch,
                            keyValueSequenceLength,
                            runStream);
                    }
                };
            }

            std::unordered_map<std::string, TensorScalarBinding> tensorScalarInputs;
            std::function<void(Stream&)> preForwardHook = ropePreForwardHook;
            auto appendPreForwardHook = [&](std::function<void(Stream&)> nextHook) {
                const std::function<void(Stream&)> previousHook = preForwardHook;
                preForwardHook = [previousHook, nextHook = std::move(nextHook)](Stream& runStream) mutable {
                    if (previousHook) previousHook(runStream);
                    nextHook(runStream);
                };
            };
            if (sdpaDropoutProbability > 0.0f) {
                auto dropoutState = std::make_shared<AttentionDropoutRuntimeState>(dropoutSeed, dropoutOffset);
                dropoutState->setOffsetAdvance(checkedDropoutOffsetAdvance(batch, numHeads, querySequenceLength, keyValueSequenceLength));
                tensorScalarInputs[kAttentionDropoutSeedInputName] = dropoutState->seedBinding(queryInput.getPlacement());
                tensorScalarInputs[kAttentionDropoutOffsetInputName] = dropoutState->offsetBinding(queryInput.getPlacement());
                appendPreForwardHook([dropoutState](Stream& runStream) { dropoutState->uploadForForward(runStream); });
            }
            if (outputDropoutProbability > 0.0f) {
                auto outputDropoutState = std::make_shared<ThorImplementation::DropOutRuntimeState>(
                    outputDropoutSeed, 0, "Attention output");
                outputDropoutState->setSequenceAdvance(1);
                tensorScalarInputs[kAttentionOutputDropoutSeedInputName] =
                    outputDropoutState->seedBinding(queryInput.getPlacement());
                tensorScalarInputs[kAttentionOutputDropoutSequenceInputName] =
                    outputDropoutState->sequenceBinding(queryInput.getPlacement());
                appendPreForwardHook(
                    [outputDropoutState](Stream& runStream) { outputDropoutState->uploadForForward(runStream); });
            }

            auto equation = std::make_shared<FusedEquation>(FusedEquation::compile(expressionOutputs.physicalOutputs(), stream.getGpuNum()));
            const auto inferredOutputShapes = equation->getOutputShapes(stampInputs, tensorScalarInputs);
            const auto inferredOutputDataTypes = equation->getOutputDataTypes(stampInputs, tensorScalarInputs);
            if (inferredOutputShapes.at("feature_output") != runtimeFeatureOutputDimensions) {
                throw std::runtime_error("Attention epilogue must preserve the feature output shape.");
            }
            if (inferredOutputDataTypes.at("feature_output") != outputDType) {
                throw std::runtime_error("Attention epilogue must preserve the feature output dtype.");
            }
            if (evaluationEquation) {
                const auto evaluationOutputShapes = evaluationEquation->getOutputShapes(stampInputs);
                const auto evaluationOutputDataTypes = evaluationEquation->getOutputDataTypes(stampInputs);
                if (evaluationOutputShapes.at("feature_output") != runtimeFeatureOutputDimensions ||
                    evaluationOutputDataTypes.at("feature_output") != outputDType) {
                    throw std::runtime_error("Attention evaluation expression must preserve the feature output descriptor.");
                }
            }

            DynamicExpressionBuild build{
                std::move(equation),
                stampInputs,
                std::move(tensorScalarInputs),
                outputs,
                {},
                std::move(preForwardHook),
            };
            // These tensors drive the ragged RoPE position-id generator in the
            // pre-forward hook. They are declared Attention/CustomLayer inputs, but
            // are intentionally not roots of the fused equation; the equation reads
            // the generated __attention_*_rope_position_ids tensors instead.
            if (useQueryRopePositionOffsets) {
                build.pre_forward_only_inputs.emplace(
                    kAttentionQueryRopePositionOffsetsInputName, queryRopePositionOffsets.value());
            }
            if (useKeyRopePositionOffsets) {
                build.pre_forward_only_inputs.emplace(
                    kAttentionKeyRopePositionOffsetsInputName, keyRopePositionOffsets.value());
            }
            if (evaluationEquation) {
                build.execution_variants.emplace(
                    kAttentionEvaluationVariant,
                    DynamicExpressionVariant{
                        .equation = std::move(evaluationEquation),
                        .tensor_scalar_inputs = {},
                        .pre_forward_hook = ropePreForwardHook,
                        .supports_backward = true,
                    });
                build.evaluation_variant_id = kAttentionEvaluationVariant;
            }
            return build;
        });
}

}  // namespace

namespace Thor {

std::shared_ptr<ThorImplementation::CustomLayer> Attention::createPhysicalLayer(
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
    std::optional<uint32_t> fixedBatchCapacity) const {
    THOR_THROW_IF_FALSE(!usesBatchValidity);
    THOR_THROW_IF_FALSE(!requiresFullBatch);
    const std::optional<ThorImplementation::DynamicExpressionVariantId> deterministicTrainingVariant =
        (sdpaDropoutProbability > 0.0f || outputDropoutProbability > 0.0f)
            ? std::optional<ThorImplementation::DynamicExpressionVariantId>(kAttentionEvaluationVariant)
            : std::nullopt;

    return std::make_shared<ThorImplementation::Attention>(std::move(expression),
                                                            std::move(physicalInputNames),
                                                            std::move(physicalOutputNames),
                                                            placement,
                                                            physicalParameters,
                                                            inferenceOnly,
                                                            stampedId,
                                                            std::move(declaredOutputDescriptors),
                                                            deterministicTrainingVariant,
                                                            isTrainingDropoutEnabled(),
                                                            std::move(inputDimensionsIncludeBatch),
                                                            fixedBatchCapacity);
}

void Attention::validateEpilogueShapePreserving(const ThorImplementation::ExpressionDefinition& definition) {
    using ThorImplementation::ExprOp;
    if (definition.outputs.expr == nullptr) {
        throw std::invalid_argument("Attention epilogue expression must have a backing expression graph.");
    }

    for (const ThorImplementation::ExprNode& node : definition.outputs.expr->nodes) {
        switch (node.op) {
            case ExprOp::FILL:
            case ExprOp::RESHAPE:
            case ExprOp::STRIDED_VIEW:
            case ExprOp::STRIDED_VIEW_BACKWARD:
            case ExprOp::UNSQUEEZE:
            case ExprOp::SQUEEZE:
            case ExprOp::TRANSPOSE:
            case ExprOp::TAKE_ALONG_AXIS:
            case ExprOp::MATMUL:
            case ExprOp::GEMM:
            case ExprOp::CONV2D:
            case ExprOp::CONV2D_BACKWARD_DATA:
            case ExprOp::CONV2D_BACKWARD_FILTER:
            case ExprOp::CONV3D:
            case ExprOp::CONV3D_BACKWARD_DATA:
            case ExprOp::CONV3D_BACKWARD_FILTER:
            case ExprOp::REDUCE_SUM:
            case ExprOp::REDUCE_PROD:
            case ExprOp::REDUCE_MIN:
            case ExprOp::REDUCE_MAX:
            case ExprOp::REDUCE_ARGMIN:
            case ExprOp::REDUCE_ARGMAX:
            case ExprOp::REDUCE_MIN_BACKWARD:
            case ExprOp::REDUCE_MAX_BACKWARD:
            case ExprOp::SCAN_MIN_BACKWARD:
            case ExprOp::SCAN_MAX_BACKWARD:
            case ExprOp::SEGMENTED_SCAN_MIN_BACKWARD:
            case ExprOp::SEGMENTED_SCAN_MAX_BACKWARD:
            case ExprOp::REDUCE_AVG:
            case ExprOp::REDUCE_NORM1:
            case ExprOp::REDUCE_NORM2:
            case ExprOp::REDUCE_SUM_SQUARES:
            case ExprOp::SCAN:
            case ExprOp::ATTENTION:
            case ExprOp::ATTENTION_BACKWARD_Q:
            case ExprOp::ATTENTION_BACKWARD_K:
            case ExprOp::ATTENTION_BACKWARD_V:
            case ExprOp::ATTENTION_BACKWARD_BIAS:
            case ExprOp::EMBEDDING_LOOKUP:
            case ExprOp::CUDA_KERNEL_OUTPUT:
            case ExprOp::SEGMENTED_SCAN:
            case ExprOp::SEGMENTED_REDUCE_SUM:
            case ExprOp::SEGMENTED_REDUCE_MIN:
            case ExprOp::SEGMENTED_REDUCE_MAX:
            case ExprOp::RAGGED_VALUEWISE_EXTENT:
            case ExprOp::SEGMENTED_REDUCE_MEAN:
                throw std::invalid_argument(
                    "Attention epilogue must preserve the output projection shape; shape-changing operations are not supported.");
            default:
                break;
        }
    }
}

void Attention::validateEpilogueAuxInputName(const std::string& inputName) {
    if (inputName.empty()) {
        throw std::invalid_argument("Attention epilogue auxiliary input name cannot be empty.");
    }
    if (inputName.rfind("__", 0) == 0) {
        throw std::invalid_argument("Attention epilogue auxiliary input names cannot start with __: " + inputName + ".");
    }
    static const std::set<std::string> reservedNames = {
        kAttentionQueryInputName,
        kAttentionKeyInputName,
        kAttentionValueInputName,
        kAttentionScoreBiasInputName,
        kAttentionQuerySequenceLengthsInputName,
        kAttentionKeyValueSequenceLengthsInputName,
        kAttentionQueryRowPartitionInputName,
        kAttentionKeyValueRowPartitionInputName,
        kAttentionQueryRopePositionOffsetsInputName,
        kAttentionKeyRopePositionOffsetsInputName,
        kAttentionResidualInputName,
        "feature_output",
        "query_weights",
        "key_weights",
        "value_weights",
        "output_weights",
        "query_bias",
        "key_bias",
        "value_bias",
        "output_bias",
        epilogueInputName(),
        epilogueOutputName(),
    };
    if (reservedNames.contains(inputName)) {
        throw std::invalid_argument("Attention epilogue auxiliary input name is reserved: " + inputName + ".");
    }
}

void Attention::Builder::verifyConfig() const {
    if (!_network.has_value()) {
        throw std::invalid_argument("Attention::Builder requires network().");
    }
    if (!_queryInput.has_value() || !_keyInput.has_value() || !_valueInput.has_value()) {
        throw std::invalid_argument("Attention::Builder requires queryInput(), keyInput(), and valueInput().");
    }
    if (_raggedKeyInput.has_value() != _raggedValueInput.has_value()) {
        throw std::invalid_argument("Attention keyInput and valueInput must both be dense or both be RaggedTensor inputs.");
    }
    if (!_numHeads.has_value()) {
        throw std::invalid_argument("Attention::Builder requires numHeads().");
    }
    const bool queryRagged = _raggedQueryInput.has_value();
    const bool keyValueRagged = _raggedKeyInput.has_value();
    const bool useAnyRagged = queryRagged || keyValueRagged;
    if (queryRagged) {
        requireRaggedFeatureInput(_raggedQueryInput.value(), "queryInput(RaggedTensor)");
    } else {
        requireRank2FeatureInput(_queryInput.value(), "query input");
    }
    if (_raggedKeyInput.has_value()) {
        requireRaggedFeatureInput(_raggedKeyInput.value(), "keyInput(RaggedTensor)");
        requireRaggedFeatureInput(_raggedValueInput.value(), "valueInput(RaggedTensor)");
    } else {
        requireRank2FeatureInput(_keyInput.value(), "key input");
        requireRank2FeatureInput(_valueInput.value(), "value input");
    }
    if (_keyInput->getDataType() != _queryInput->getDataType() ||
        _valueInput->getDataType() != _queryInput->getDataType()) {
        throw std::invalid_argument("Attention query, key, and value input dtypes must match for the current training path.");
    }
    if (_keyInput->getDimensions().at(0) != _valueInput->getDimensions().at(0)) {
        throw std::invalid_argument("Attention key and value inputs must have the same sequence length.");
    }
    if (keyValueRagged &&
        (_raggedKeyInput->getBatchSize() != _raggedValueInput->getBatchSize() ||
         _raggedKeyInput->getMaxTotalValues() != _raggedValueInput->getMaxTotalValues() ||
         _raggedKeyInput->getOffsets() != _raggedValueInput->getOffsets())) {
        throw std::invalid_argument(
            "Attention ragged keyInput and valueInput must use the exact same row partition and capacity.");
    }
    if (queryRagged && keyValueRagged && _raggedKeyInput.has_value() &&
        _raggedKeyInput->getBatchSize() != _raggedQueryInput->getBatchSize()) {
        throw std::invalid_argument("Attention ragged query and key/value inputs must have the same logical batch size.");
    }
    if (_querySequenceLengthsInput.has_value() != _keyValueSequenceLengthsInput.has_value()) {
        throw std::invalid_argument(
            "Attention requires both querySequenceLengthsInput and keyValueSequenceLengthsInput.");
    }
    if (_querySequenceLengthsInput.has_value()) {
        requireSequenceLengthsInput(_querySequenceLengthsInput.value(), "querySequenceLengthsInput");
        requireSequenceLengthsInput(_keyValueSequenceLengthsInput.value(), "keyValueSequenceLengthsInput");
    }
    if (useAnyRagged && _querySequenceLengthsInput.has_value()) {
        throw std::invalid_argument(
            "Attention RaggedTensor inputs already define sequence lengths through their row partitions; "
            "querySequenceLengthsInput/keyValueSequenceLengthsInput are not allowed in ragged mode.");
    }
    if (_queryRopePositionOffsetsInput.has_value()) {
        requireRopePositionOffsetsInput(_queryRopePositionOffsetsInput.value(), "queryRopePositionOffsetsInput");
    }
    if (_keyRopePositionOffsetsInput.has_value()) {
        requireRopePositionOffsetsInput(_keyRopePositionOffsetsInput.value(), "keyRopePositionOffsetsInput");
    }
    if (_queryRopePositionOffsetsInput.has_value() && !queryRagged) {
        throw std::invalid_argument(
            "Attention queryRopePositionOffsetsInput requires a RaggedTensor query input.");
    }
    if (_keyRopePositionOffsetsInput.has_value() && !keyValueRagged) {
        throw std::invalid_argument(
            "Attention keyRopePositionOffsetsInput requires a RaggedTensor key/value input.");
    }
    const uint64_t maximumPossibleQuerySequenceLength =
        queryRagged ? _raggedQueryInput->getMaxTotalValues() : _queryInput->getDimensions().at(0);
    const uint64_t maximumPossibleKeySequenceLength =
        keyValueRagged ? _raggedKeyInput->getMaxTotalValues() : _keyInput->getDimensions().at(0);
    const ThorImplementation::RotaryPositionEmbeddingOptions resolvedRopeOptions =
        _ropeOptions.value_or(ThorImplementation::RotaryPositionEmbeddingOptions{});
    const int64_t queryRopePositionOffset = _queryRopePositionOffset.value_or(resolvedRopeOptions.position_offset);
    const int64_t keyRopePositionOffset = _keyRopePositionOffset.value_or(resolvedRopeOptions.position_offset);
    if (const std::optional<std::string> error = ropeFp32SequenceLengthValidationError(
            _useRope.value_or(false),
            resolvedRopeOptions,
            _queryRopePositionOffsetsInput.has_value() ? 0 : queryRopePositionOffset,
            _keyRopePositionOffsetsInput.has_value() ? 0 : keyRopePositionOffset,
            maximumPossibleQuerySequenceLength,
            maximumPossibleKeySequenceLength);
        error.has_value()) {
        throw std::invalid_argument(error.value());
    }
    if (_numHeads.value() == 0) {
        throw std::invalid_argument("Attention numHeads must be non-zero.");
    }
    if (_numKeyValueHeads.value_or(_numHeads.value()) == 0) {
        throw std::invalid_argument("Attention numKeyValueHeads must be non-zero.");
    }
    if (_numHeads.value() % _numKeyValueHeads.value_or(_numHeads.value()) != 0) {
        throw std::invalid_argument("Attention numHeads must be a multiple of numKeyValueHeads for MHA/GQA/MQA.");
    }
    if (_headDim.value_or(1) == 0 || _valueDim.value_or(1) == 0) {
        throw std::invalid_argument("Attention headDim/valueDim must be non-zero.");
    }

    const DataType inputDType = _queryInput->getDataType();
    const DataType weightsDType = _weightsDataType.value_or(inputDType);
    const DataType outputDType = _outputDataType.value_or(inputDType);
    const DataType computeDType = _computeDataType.value_or(DataType::FP32);
    if (!isStorageDType(inputDType)) {
        throw std::invalid_argument("Attention query input dtype must be fp16 or bf16 for the current training path. Got " +
                                    dtypeName(inputDType) + ".");
    }
    if (!isStorageDType(weightsDType)) {
        throw std::invalid_argument("Attention weightsDataType must be fp16 or bf16 for the current training path. Got " +
                                    dtypeName(weightsDType) + ".");
    }
    if (!isStorageDType(outputDType)) {
        throw std::invalid_argument("Attention outputDataType must be fp16 or bf16 for the current training path. Got " +
                                    dtypeName(outputDType) + ".");
    }
    if (!isComputeDType(computeDType)) {
        throw std::invalid_argument("Attention computeDataType must currently be fp32 for cuDNN SDPA training. Got " +
                                    dtypeName(computeDType) + ".");
    }
    if (inputDType != weightsDType || inputDType != outputDType) {
        throw std::invalid_argument(
            "Attention requires query/key/value inputs, projection weights, and attention output storage to use the same "
            "FP16 or BF16 dtype for the current execution path. Thor will not insert hidden conversions between attention "
            "operands. input=" +
            dtypeName(inputDType) + ", weights=" + dtypeName(weightsDType) + ", output=" + dtypeName(outputDType) + ".");
    }

    const auto maskKind = _maskKind.value_or(ThorImplementation::AttentionMaskKind::None);
    const bool useAlibi = _useAlibiMask.value_or(false);
    const int64_t rightBound = _diagonalRightBound.value_or(0);
    if (useAlibi && maskKind != ThorImplementation::AttentionMaskKind::CausalTopLeft &&
        maskKind != ThorImplementation::AttentionMaskKind::CausalBottomRight &&
        maskKind != ThorImplementation::AttentionMaskKind::SlidingWindowTopLeft &&
        maskKind != ThorImplementation::AttentionMaskKind::SlidingWindowBottomRight) {
        throw std::invalid_argument("Attention ALiBi requires a causal/sliding-window diagonal mask.");
    }
    if (useAlibi && rightBound != 0) {
        throw std::invalid_argument(
            "Attention ALiBi requires diagonalRightBound == 0 because cuDNN rejects ALiBi with positive right bounds.");
    }
    if (useAlibi && (maskKind == ThorImplementation::AttentionMaskKind::CausalBottomRight ||
                     maskKind == ThorImplementation::AttentionMaskKind::SlidingWindowBottomRight)) {
        throw std::invalid_argument("Attention ALiBi cannot currently be combined with bottom-right/decode masks in cuDNN SDPA.");
    }
    const float sdpaDropoutProbability = _sdpaDropoutProbability.value_or(0.0f);
    if (!std::isfinite(sdpaDropoutProbability) || sdpaDropoutProbability < 0.0f || sdpaDropoutProbability >= 1.0f) {
        throw std::invalid_argument("Attention sdpaDropoutProbability must be finite and in [0, 1).");
    }
    if (sdpaDropoutProbability > 0.0f && (maskKind == ThorImplementation::AttentionMaskKind::CausalBottomRight ||
                                      maskKind == ThorImplementation::AttentionMaskKind::SlidingWindowBottomRight)) {
        throw std::invalid_argument("Attention SDPA dropout cannot currently be combined with bottom-right/decode masks in cuDNN SDPA.");
    }
    if (sdpaDropoutProbability > 0.0f && _dropoutOffset.value_or(0) < 0) {
        throw std::invalid_argument("Attention sdpaDropoutOffset must be non-negative when SDPA dropout is enabled.");
    }

    const float outputDropoutProbability = _outputDropoutProbability.value_or(0.0f);
    if (!std::isfinite(outputDropoutProbability) || outputDropoutProbability < 0.0f || outputDropoutProbability >= 1.0f) {
        throw std::invalid_argument("Attention outputDropoutProbability must be finite and in [0, 1).");
    }
    if (_epilogue.has_value() && (_residualInput.has_value() || outputDropoutProbability > 0.0f)) {
        throw std::invalid_argument(
            "Attention residualInput/outputDropoutProbability cannot currently be combined with a custom epilogue; "
            "use the first-class residual/output-dropout path or the custom epilogue, not both.");
    }
    if (_residualInput.has_value()) {
        const std::vector<uint64_t> expectedResidualDimensions = {
            _queryInput->getDimensions().at(0),
            _outputFeatures.value_or(static_cast<uint32_t>(_queryInput->getDimensions().at(1))),
        };
        if (!_residualInput->isInitialized()) {
            throw std::invalid_argument("Attention residualInput tensor is not initialized.");
        }
        if (_residualInput->getDataType() != outputDType) {
            throw std::invalid_argument("Attention residualInput dtype must match outputDataType.");
        }
        if (_residualInput->getDimensions() != expectedResidualDimensions) {
            throw std::invalid_argument("Attention residualInput shape must match the feature output shape.");
        }
        if (queryRagged != _raggedResidualInput.has_value()) {
            throw std::invalid_argument(
                "Attention residualInput must be ragged exactly when the query/output is ragged.");
        }
        if (_raggedResidualInput.has_value()) {
            requireRaggedFeatureInput(_raggedResidualInput.value(), "residualInput(RaggedTensor)");
            if (_raggedResidualInput->getBatchSize() != _raggedQueryInput->getBatchSize() ||
                _raggedResidualInput->getMaxTotalValues() != _raggedQueryInput->getMaxTotalValues() ||
                _raggedResidualInput->getOffsets() != _raggedQueryInput->getOffsets()) {
                throw std::invalid_argument(
                    "Attention ragged residualInput must use the exact query row partition and capacity.");
            }
        }
    } else if (_raggedResidualInput.has_value()) {
        throw std::invalid_argument("Attention ragged residual metadata requires residualInput.");
    }
    if (_ropeInPlace.value_or(false) && !_useRope.value_or(false)) {
        throw std::invalid_argument("Attention ropeInPlace requires useRope to be enabled.");
    }
    if (_attentionScale.has_value() && (!std::isfinite(_attentionScale.value()) || _attentionScale.value() <= 0.0)) {
        throw std::invalid_argument("Attention attentionScale must be finite and positive.");
    }
    if (useAnyRagged && _scoreBiasInput.has_value()) {
        throw std::invalid_argument(
            "Attention ragged mode does not support scoreBiasInput because the current cuDNN ragged SDPA backward path "
            "does not support additive-bias gradients.");
    }
    if (_scoreBiasInput.has_value()) {
        const std::vector<uint64_t> queryDims = _queryInput->getDimensions();
        const std::vector<uint64_t> keyDims = _keyInput->getDimensions();
        requireScoreBiasInput(_scoreBiasInput.value(), _numHeads.value(), queryDims.at(0), keyDims.at(0), computeDType);
        if (maskKind == ThorImplementation::AttentionMaskKind::CausalBottomRight ||
            maskKind == ThorImplementation::AttentionMaskKind::SlidingWindowBottomRight) {
            throw std::invalid_argument(
                "Attention bottom-right/decode masks cannot currently be combined with scoreBiasInput in cuDNN SDPA.");
        }
    }

    if (_epilogue.has_value()) {
        const std::vector<std::string> auxiliaryInputNames = epilogueAuxInputNames();
        Attention::validateEpilogueExpression(_epilogue.value(), auxiliaryInputNames);

        auto validateExpressionInputDTypes = [&](const std::string& inputName) {
            const AttentionEpilogueInputDataTypes inputDataTypes =
                attentionEpilogueInputDataTypes(_epilogue.value(), inputName);
            if (inputDataTypes.outputDataType.has_value() && inputDataTypes.outputDataType.value() != outputDType) {
                throw std::invalid_argument("Attention epilogue input '" + inputName +
                                            "' output dtype annotation must match outputDataType.");
            }
            if (inputDataTypes.computeDataType.has_value() && inputDataTypes.computeDataType.value() != computeDType) {
                throw std::invalid_argument("Attention epilogue input '" + inputName +
                                            "' compute dtype annotation must match computeDataType.");
            }
        };
        validateExpressionInputDTypes(Attention::epilogueInputName());
        for (const std::string& inputName : auxiliaryInputNames) {
            validateExpressionInputDTypes(inputName);
        }
    } else if (!_epilogueInputBindings.empty()) {
        throw std::invalid_argument("Attention epilogue_inputs were provided without an epilogue expression.");
    }
    const std::vector<uint64_t> expectedEpilogueInputDims = {
        _queryInput->getDimensions().at(0),
        _outputFeatures.value_or(static_cast<uint32_t>(_queryInput->getDimensions().at(1))),
    };
    for (const auto& [name, tensor] : _epilogueInputBindings) {
        Attention::validateEpilogueAuxInputName(name);
        if (!tensor.isInitialized()) {
            throw std::invalid_argument("Attention epilogue input '" + name + "' is not initialized.");
        }
        if (tensor.getDataType() != outputDType) {
            throw std::invalid_argument("Attention epilogue input '" + name + "' dtype must match outputDataType.");
        }
        if (tensor.getDimensions() != expectedEpilogueInputDims) {
            throw std::invalid_argument("Attention epilogue input '" + name + "' shape must match feature output shape.");
        }
    }
}

Attention Attention::Builder::build() {
    if (!_numKeyValueHeads.has_value() && _numHeads.has_value()) {
        _numKeyValueHeads = _numHeads.value();
    }
    if (!_headDim.has_value()) {
        if (!_queryInput.has_value() || !_numHeads.has_value()) {
            throw std::invalid_argument("Attention headDim default requires queryInput and numHeads.");
        }
        const uint64_t inputFeatures = _queryInput->getDimensions().at(1);
        if (inputFeatures % _numHeads.value() != 0) {
            throw std::invalid_argument("Attention default headDim requires query input features divisible by numHeads.");
        }
        _headDim = static_cast<uint32_t>(inputFeatures / _numHeads.value());
    }
    if (!_valueDim.has_value()) {
        _valueDim = _headDim.value();
    }
    if (!_outputFeatures.has_value() && _queryInput.has_value()) {
        _outputFeatures = static_cast<uint32_t>(_queryInput->getDimensions().at(1));
    }
    if (!_hasBias.has_value()) _hasBias = false;
    if (!_maskKind.has_value()) _maskKind = ThorImplementation::AttentionMaskKind::None;
    if (!_diagonalLeftBound.has_value()) _diagonalLeftBound = 0;
    if (!_diagonalRightBound.has_value()) _diagonalRightBound = 0;
    if (!_useAlibiMask.has_value()) _useAlibiMask = false;
    if (!_sdpaDropoutProbability.has_value()) _sdpaDropoutProbability = 0.0f;
    if (!_dropoutSeed.has_value()) _dropoutSeed = 0;
    if (!_dropoutOffset.has_value()) _dropoutOffset = 0;
    if (!_outputDropoutProbability.has_value()) _outputDropoutProbability = 0.0f;
    if (!_outputDropoutSeed.has_value()) {
        if (_outputDropoutProbability.value() > 0.0f) {
            std::random_device rd;
            const uint64_t high = static_cast<uint64_t>(rd()) << 32U;
            const uint64_t low = static_cast<uint64_t>(rd());
            _outputDropoutSeed =
                static_cast<int64_t>((high ^ low) & static_cast<uint64_t>(std::numeric_limits<int64_t>::max()));
        } else {
            _outputDropoutSeed = 0;
        }
    }
    if (!_useRope.has_value()) _useRope = false;
    if (!_ropeInPlace.has_value()) _ropeInPlace = false;
    if (!_ropeOptions.has_value()) _ropeOptions = ThorImplementation::RotaryPositionEmbeddingOptions{};
    if (!_queryRopePositionOffset.has_value()) _queryRopePositionOffset = _ropeOptions->position_offset;
    if (!_keyRopePositionOffset.has_value()) _keyRopePositionOffset = _ropeOptions->position_offset;
    if (!_weightsDataType.has_value() && _queryInput.has_value()) _weightsDataType = _queryInput->getDataType();
    if (!_computeDataType.has_value()) _computeDataType = DataType::FP32;
    if (!_outputDataType.has_value() && _queryInput.has_value()) _outputDataType = _queryInput->getDataType();
    if (_weightsInitializer == nullptr) _weightsInitializer = Glorot::Builder().build();
    if (_biasInitializer == nullptr) _biasInitializer = Glorot::Builder().build();

    verifyConfig();

    const auto queryDims = _queryInput->getDimensions();
    const auto keyDims = _keyInput->getDimensions();
    const auto valueDims = _valueInput->getDimensions();
    const uint64_t querySequenceLength = queryDims.at(0);
    const uint64_t keyValueSequenceLength = keyDims.at(0);
    const uint64_t queryInputFeatures = queryDims.at(1);
    const uint64_t keyInputFeatures = keyDims.at(1);
    const uint64_t valueInputFeatures = valueDims.at(1);
    const uint64_t qWidth = checkedMul(_numHeads.value(), _headDim.value(), "query projection width");
    const uint64_t kvKeyWidth = checkedMul(_numKeyValueHeads.value(), _headDim.value(), "key projection width");
    const uint64_t kvValueWidth = checkedMul(_numKeyValueHeads.value(), _valueDim.value(), "value projection width");
    const uint64_t mergedWidth = checkedMul(_numHeads.value(), _valueDim.value(), "merged head width");

    const bool useScoreBias = _scoreBiasInput.has_value();
    const bool useSequenceLengths = _querySequenceLengthsInput.has_value();
    const bool queryRagged = _raggedQueryInput.has_value();
    const bool keyValueRagged = _raggedKeyInput.has_value();
    const bool useQueryRopePositionOffsets = _queryRopePositionOffsetsInput.has_value();
    const bool useKeyRopePositionOffsets = _keyRopePositionOffsetsInput.has_value();
    const std::vector<std::string> epilogueAuxNames = epilogueAuxInputNames();

    std::vector<std::shared_ptr<ParameterSpecification>> parameters;
    parameters.push_back(
        makeParameter("query_weights", {queryInputFeatures, qWidth}, _weightsDataType.value(), _weightsInitializer, _optimizer));
    parameters.push_back(
        makeParameter("key_weights", {keyInputFeatures, kvKeyWidth}, _weightsDataType.value(), _weightsInitializer, _optimizer));
    parameters.push_back(
        makeParameter("value_weights", {valueInputFeatures, kvValueWidth}, _weightsDataType.value(), _weightsInitializer, _optimizer));
    parameters.push_back(
        makeParameter("output_weights", {mergedWidth, _outputFeatures.value()}, _weightsDataType.value(), _weightsInitializer, _optimizer));
    if (_hasBias.value()) {
        parameters.push_back(makeParameter("query_bias", {qWidth}, _weightsDataType.value(), _biasInitializer, _optimizer));
        parameters.push_back(makeParameter("key_bias", {kvKeyWidth}, _weightsDataType.value(), _biasInitializer, _optimizer));
        parameters.push_back(makeParameter("value_bias", {kvValueWidth}, _weightsDataType.value(), _biasInitializer, _optimizer));
        parameters.push_back(
            makeParameter("output_bias", {_outputFeatures.value()}, _weightsDataType.value(), _biasInitializer, _optimizer));
    }

    Tensor output(_outputDataType.value(), {querySequenceLength, _outputFeatures.value()});
    Attention layer(makeAttentionExpression(querySequenceLength,
                                            keyValueSequenceLength,
                                            queryInputFeatures,
                                            keyInputFeatures,
                                            valueInputFeatures,
                                            _outputFeatures.value(),
                                            _numHeads.value(),
                                            _numKeyValueHeads.value(),
                                            _headDim.value(),
                                            _valueDim.value(),
                                            _hasBias.value(),
                                            _useRope.value(),
                                            _ropeInPlace.value(),
                                            _ropeOptions.value(),
                                            _queryRopePositionOffset.value(),
                                            _keyRopePositionOffset.value(),
                                            _maskKind.value(),
                                            _diagonalLeftBound.value(),
                                            _diagonalRightBound.value(),
                                            _useAlibiMask.value(),
                                            _attentionScale,
                                            _sdpaDropoutProbability.value(),
                                            _dropoutSeed.value(),
                                            _dropoutOffset.value(),
                                            _outputDropoutProbability.value(),
                                            _outputDropoutSeed.value(),
                                            _residualInput.has_value(),
                                            useScoreBias,
                                            useSequenceLengths,
                                            queryRagged,
                                            keyValueRagged,
                                            useQueryRopePositionOffsets,
                                            useKeyRopePositionOffsets,
                                            queryRagged ? _raggedQueryInput->getBatchSize()
                                                        : (keyValueRagged ? _raggedKeyInput->getBatchSize() : 0),
                                            queryRagged ? _raggedQueryInput->getOffsetsDataType()
                                                        : ThorImplementation::kDefaultRowPartitionOffsetDataType,
                                            keyValueRagged ? _raggedKeyInput->getOffsetsDataType()
                                                           : ThorImplementation::kDefaultRowPartitionOffsetDataType,
                                            _queryInput->getDataType(),
                                            _weightsDataType.value(),
                                            _computeDataType.value(),
                                            _outputDataType.value(),
                                            _epilogue,
                                            epilogueAuxNames),
                    publicAttentionInputNames(useScoreBias,
                                              useSequenceLengths,
                                              queryRagged,
                                              keyValueRagged,
                                              useQueryRopePositionOffsets,
                                              useKeyRopePositionOffsets,
                                              _residualInput.has_value(),
                                              epilogueAuxNames),
                    {publicAttentionInputInterface(_queryInput.value(),
                                                   _keyInput.value(),
                                                   _valueInput.value(),
                                                   _scoreBiasInput,
                                                   _querySequenceLengthsInput,
                                                   _keyValueSequenceLengthsInput,
                                                   _queryRopePositionOffsetsInput,
                                                   _keyRopePositionOffsetsInput,
                                                   _raggedQueryInput,
                                                   _raggedKeyInput,
                                                   _residualInput,
                                                   _epilogueInputBindings)},
                    {{{"feature_output", output}}},
                    std::move(parameters),
                    _epilogue,
                    _epilogueInputBindings,
                    _numHeads.value(),
                    _numKeyValueHeads.value(),
                    _headDim.value(),
                    _valueDim.value(),
                    _outputFeatures.value(),
                    _hasBias.value(),
                    _useRope.value(),
                    _ropeInPlace.value(),
                    _ropeOptions.value(),
                    _queryRopePositionOffset.value(),
                    _keyRopePositionOffset.value(),
                    _maskKind.value(),
                    _diagonalLeftBound.value(),
                    _diagonalRightBound.value(),
                    _useAlibiMask.value(),
                    _attentionScale,
                    _sdpaDropoutProbability.value(),
                    _dropoutSeed.value(),
                    _dropoutOffset.value(),
                    _outputDropoutProbability.value(),
                    _outputDropoutSeed.value(),
                    _residualInput,
                    _raggedResidualInput,
                    _keyInput.value(),
                    _valueInput.value(),
                    _scoreBiasInput,
                    _querySequenceLengthsInput,
                    _keyValueSequenceLengthsInput,
                    _queryRopePositionOffsetsInput,
                    _keyRopePositionOffsetsInput,
                    _raggedQueryInput,
                    _raggedKeyInput,
                    _raggedValueInput,
                    _weightsDataType.value(),
                    _computeDataType.value(),
                    _outputDataType.value());

    layer.addToNetwork(_network.value());
    return layer;
}

json Attention::architectureJson() const {
    json j;
    j["factory"] = Layer::Factory::Learning.value();
    j["version"] = "2.0.0";
    j["layer_type"] = "attention";
    j["layer_name"] = std::string("layer") + std::to_string(getId());

    j["num_heads"] = numHeads;
    j["num_key_value_heads"] = numKeyValueHeads;
    j["head_dim"] = headDim;
    j["value_dim"] = valueDim;
    j["output_features"] = outputFeatures;
    j["has_bias"] = hasBias;
    j["use_rope"] = useRope;
    j["rope_in_place"] = ropeInPlace;
    ThorImplementation::RotaryPositionEmbeddingOptions serializedRopeOptions = ropeOptions;
    if (queryRopePositionOffset == keyRopePositionOffset) {
        // Keep the shared Q/K origin mirrored in rope_options as well as the explicit schema-2 fields.
        serializedRopeOptions.position_offset = queryRopePositionOffset;
    }
    j["rope_options"] = ropeOptionsToJson(serializedRopeOptions);
    j["rope_query_position_offset"] = queryRopePositionOffset;
    j["rope_key_position_offset"] = keyRopePositionOffset;
    j["mask_kind"] = attentionMaskKindToString(maskKind);
    j["diagonal_left_bound"] = diagonalLeftBound;
    j["diagonal_right_bound"] = diagonalRightBound;
    j["use_alibi_mask"] = useAlibiMask;
    j["attention_scale"] = attentionScale.has_value() ? json(attentionScale.value()) : json(nullptr);
    j["sdpa_dropout_probability"] = sdpaDropoutProbability;
    j["dropout_probability"] = sdpaDropoutProbability;
    j["sdpa_dropout_seed"] = dropoutSeed;
    j["sdpa_dropout_offset"] = dropoutOffset;
    j["dropout_seed"] = dropoutSeed;
    j["dropout_offset"] = dropoutOffset;
    j["output_dropout_probability"] = outputDropoutProbability;
    j["output_dropout_seed"] = outputDropoutSeed;
    j["use_residual"] = residualInput.has_value();
    j["use_score_bias"] = scoreBiasInput.has_value();
    j["use_sequence_lengths"] = querySequenceLengthsInput.has_value();
    j["use_query_rope_position_offsets"] = queryRopePositionOffsetsInput.has_value();
    j["use_key_rope_position_offsets"] = keyRopePositionOffsetsInput.has_value();
    const bool queryRagged = raggedQueryInput.has_value();
    const bool keyValueRagged = raggedKeyInput.has_value();
    j["use_ragged"] = queryRagged || keyValueRagged;
    j["query_ragged"] = queryRagged;
    j["key_value_ragged"] = keyValueRagged;
    j["weights_data_type"] = weightsDataType;
    j["compute_data_type"] = computeDataType;
    j["output_data_type"] = outputDataType;

    if (epilogue.has_value()) {
        if (!serializableEpilogue.has_value()) {
            std::vector<std::string> auxiliaryInputNames;
            auxiliaryInputNames.reserve(epilogueInputBindings.size());
            for (const auto& [name, tensor] : epilogueInputBindings) {
                (void)tensor;
                auxiliaryInputNames.push_back(name);
            }
            serializableEpilogue = makeEpilogueDefinition(epilogue.value(), auxiliaryInputNames);
        }
        j["epilogue"] = serializableEpilogue.value().architectureJson();
    } else {
        j["epilogue"] = nullptr;
    }

    json epilogueInputs = json::array();
    for (const auto& [name, tensor] : epilogueInputBindings) {
        epilogueInputs.push_back(json{{"name", name}, {"tensor", tensor.architectureJson()}});
    }
    j["epilogue_inputs"] = std::move(epilogueInputs);

    const std::optional<Tensor> output = getFeatureOutput();
    if (!output.has_value()) {
        throw std::runtime_error("Attention serialization requires one feature output.");
    }
    j["query_input"] = getQueryInput().architectureJson();
    j["key_input"] = keyInput.architectureJson();
    j["value_input"] = valueInput.architectureJson();
    if (residualInput.has_value()) j["residual_input"] = residualInput->architectureJson();
    if (scoreBiasInput.has_value()) j["score_bias_input"] = scoreBiasInput.value().architectureJson();
    if (querySequenceLengthsInput.has_value()) {
        j["query_sequence_lengths_input"] = querySequenceLengthsInput.value().architectureJson();
        j["key_value_sequence_lengths_input"] = keyValueSequenceLengthsInput.value().architectureJson();
    }
    if (queryRopePositionOffsetsInput.has_value()) {
        j["query_rope_position_offsets_input"] = queryRopePositionOffsetsInput.value().architectureJson();
    }
    if (keyRopePositionOffsetsInput.has_value()) {
        j["key_rope_position_offsets_input"] = keyRopePositionOffsetsInput.value().architectureJson();
    }
    if (raggedQueryInput.has_value()) {
        j["ragged_query_input"] = raggedQueryInput->architectureJson();
        if (!raggedFeatureOutput.has_value()) {
            throw std::runtime_error("Attention ragged-query serialization requires a logical RaggedTensor output.");
        }
        j["ragged_feature_output"] = raggedFeatureOutput->architectureJson();
    }
    if (raggedKeyInput.has_value()) {
        j["ragged_key_input"] = raggedKeyInput->architectureJson();
        j["ragged_value_input"] = raggedValueInput->architectureJson();
    }
    if (raggedResidualInput.has_value()) j["ragged_residual_input"] = raggedResidualInput->architectureJson();
    j["feature_output"] = output.value().architectureJson();
    j["parameters"] = getParametersArchitectureJson()["parameters"];
    return j;
}

json Attention::serialize(thor_file::TarWriter& archiveWriter,
                          Stream stream,
                          bool saveOptimizerState,
                          ThorImplementation::StampedNetwork& stampedNetwork) const {
    json j = architectureJson();
    Parameterizable::serializeParameters(j["parameters"], archiveWriter, stream, saveOptimizerState, stampedNetwork, "layer" + std::to_string(getId()));
    return j;
}

void Attention::deserialize(std::shared_ptr<thor_file::TarReader>& archiveReader, const json& j, Network* network) {
    const std::string serializedVersion = j.at("version").get<std::string>();
    if (serializedVersion != "2.0.0") {
        throw std::runtime_error("Unsupported version in Attention::deserialize: " + serializedVersion);
    }
    if (j.at("layer_type").get<std::string>() != "attention") {
        throw std::runtime_error("Layer type mismatch in Attention::deserialize: " + j.at("layer_type").get<std::string>());
    }
    // Schema 2 owns raggedness through canonical RaggedTensor metadata. Reject
    // the removed raw-offset controls rather than silently accepting an archive
    // whose meaning depends on the pre-canonical interface.
    static constexpr const char* removedRawRaggedFields[] = {
        "ragged_offsets_input",
        "use_separate_ragged_offsets",
        "use_ragged_offsets",
        "query_ragged_offsets_input",
        "key_value_ragged_offsets_input",
    };
    for (const char* field : removedRawRaggedFields) {
        if (j.contains(field)) {
            throw std::runtime_error(std::string("Attention schema 2.0.0 does not support removed raw ragged metadata field: ") +
                                     field + ".");
        }
    }
    if (!j.contains("query_input") || !j.contains("key_input") || !j.contains("value_input")) {
        throw std::runtime_error("Attention schema 2.0.0 requires query_input, key_input, and value_input.");
    }

    Tensor queryInput = network->getApiTensorByOriginalId(j.at("query_input").at("id").get<uint64_t>());
    Tensor keyInput = network->getApiTensorByOriginalId(j.at("key_input").at("id").get<uint64_t>());
    Tensor valueInput = network->getApiTensorByOriginalId(j.at("value_input").at("id").get<uint64_t>());

    std::optional<Tensor> residualInput = std::nullopt;
    if (j.value("use_residual", false) || j.contains("residual_input")) {
        if (!j.contains("residual_input")) throw std::runtime_error("Attention deserialize missing residual_input.");
        residualInput = network->getApiTensorByOriginalId(j.at("residual_input").at("id").get<uint64_t>());
    }
    std::optional<Tensor> scoreBiasInput = std::nullopt;
    if (j.value("use_score_bias", false) || j.contains("score_bias_input")) {
        if (!j.contains("score_bias_input")) throw std::runtime_error("Attention deserialize missing score_bias_input.");
        scoreBiasInput = network->getApiTensorByOriginalId(j.at("score_bias_input").at("id").get<uint64_t>());
    }

    std::vector<std::pair<std::string, Tensor>> epilogueInputBindings;
    std::vector<std::string> epilogueAuxInputNames;
    if (j.contains("epilogue_inputs")) {
        if (!j.at("epilogue_inputs").is_array()) throw std::runtime_error("Attention epilogue_inputs must be an array.");
        std::set<std::string> seenNames;
        for (const json& entry : j.at("epilogue_inputs")) {
            if (!entry.is_object() || !entry.contains("name") || !entry.contains("tensor")) {
                throw std::runtime_error("Attention epilogue_inputs entries must contain name and tensor fields.");
            }
            const std::string name = entry.at("name").get<std::string>();
            validateEpilogueAuxInputName(name);
            if (!seenNames.insert(name).second) {
                throw std::runtime_error("Attention serialized epilogue input name is duplicated: " + name + ".");
            }
            Tensor tensor = network->getApiTensorByOriginalId(entry.at("tensor").at("id").get<uint64_t>());
            epilogueInputBindings.emplace_back(name, tensor);
            epilogueAuxInputNames.push_back(name);
        }
    }
    std::optional<ThorImplementation::Expression> epilogue = std::nullopt;
    if (j.contains("epilogue") && !j.at("epilogue").is_null()) {
        ThorImplementation::ExpressionDefinition definition =
            ThorImplementation::ExpressionDefinition::deserialize(j.at("epilogue"));
        epilogue = epilogueExpressionFromDefinition(definition, epilogueAuxInputNames);
    } else if (!epilogueInputBindings.empty()) {
        throw std::runtime_error("Attention serialized epilogue_inputs require a non-null epilogue expression.");
    }

    std::optional<Tensor> querySequenceLengthsInput = std::nullopt;
    std::optional<Tensor> keyValueSequenceLengthsInput = std::nullopt;
    if (j.value("use_sequence_lengths", false) || j.contains("query_sequence_lengths_input") ||
        j.contains("key_value_sequence_lengths_input")) {
        if (!j.contains("query_sequence_lengths_input") || !j.contains("key_value_sequence_lengths_input")) {
            throw std::runtime_error(
                "Attention deserialize missing query_sequence_lengths_input/key_value_sequence_lengths_input.");
        }
        querySequenceLengthsInput =
            network->getApiTensorByOriginalId(j.at("query_sequence_lengths_input").at("id").get<uint64_t>());
        keyValueSequenceLengthsInput =
            network->getApiTensorByOriginalId(j.at("key_value_sequence_lengths_input").at("id").get<uint64_t>());
    }

    std::optional<Tensor> queryRopePositionOffsetsInput = std::nullopt;
    std::optional<Tensor> keyRopePositionOffsetsInput = std::nullopt;
    if (j.value("use_query_rope_position_offsets", false) || j.contains("query_rope_position_offsets_input")) {
        if (!j.contains("query_rope_position_offsets_input")) {
            throw std::runtime_error("Attention deserialize missing query_rope_position_offsets_input.");
        }
        queryRopePositionOffsetsInput =
            network->getApiTensorByOriginalId(j.at("query_rope_position_offsets_input").at("id").get<uint64_t>());
    }
    if (j.value("use_key_rope_position_offsets", false) || j.contains("key_rope_position_offsets_input")) {
        if (!j.contains("key_rope_position_offsets_input")) {
            throw std::runtime_error("Attention deserialize missing key_rope_position_offsets_input.");
        }
        keyRopePositionOffsetsInput =
            network->getApiTensorByOriginalId(j.at("key_rope_position_offsets_input").at("id").get<uint64_t>());
    }

    auto raggedFromNetworkMetadata = [&](const json& raggedJson, const char* fieldName) -> RaggedTensor {
        if (!raggedJson.is_object() || !raggedJson.contains("values") || !raggedJson.contains("offsets") ||
            !raggedJson.contains("batch_size") || !raggedJson.contains("max_total_values")) {
            throw std::runtime_error(std::string("Attention serialized ") + fieldName + " metadata is incomplete.");
        }
        Tensor values = network->getApiTensorByOriginalId(raggedJson.at("values").at("id").get<uint64_t>());
        Tensor offsets = network->getApiTensorByOriginalId(raggedJson.at("offsets").at("id").get<uint64_t>());
        RaggedTensor ragged(values, offsets);
        if (ragged.getBatchSize() != raggedJson.at("batch_size").get<uint64_t>() ||
            ragged.getMaxTotalValues() != raggedJson.at("max_total_values").get<uint64_t>()) {
            throw std::runtime_error(std::string("Attention serialized ") + fieldName +
                                     " batch/capacity metadata does not match its values/offsets tensors.");
        }
        return ragged;
    };

    const bool queryRagged = j.value("query_ragged", false);
    const bool keyValueRagged = j.value("key_value_ragged", false);
    std::optional<RaggedTensor> raggedQueryInput = std::nullopt;
    std::optional<RaggedTensor> raggedKeyInput = std::nullopt;
    std::optional<RaggedTensor> raggedValueInput = std::nullopt;
    if (queryRagged) {
        if (!j.contains("ragged_query_input") || !j.contains("ragged_feature_output")) {
            throw std::runtime_error(
                "Attention deserialize ragged query mode requires ragged_query_input and ragged_feature_output.");
        }
        raggedQueryInput = raggedFromNetworkMetadata(j.at("ragged_query_input"), "ragged_query_input");
        if (raggedQueryInput->getValues() != queryInput) {
            throw std::runtime_error("Attention serialized ragged_query_input values do not match query_input.");
        }
    } else if (j.contains("ragged_query_input")) {
        throw std::runtime_error("Attention serialized ragged_query_input requires query_ragged=true.");
    }
    if (keyValueRagged) {
        if (!j.contains("ragged_key_input") || !j.contains("ragged_value_input")) {
            throw std::runtime_error(
                "Attention deserialize ragged key/value mode requires ragged_key_input and ragged_value_input.");
        }
        raggedKeyInput = raggedFromNetworkMetadata(j.at("ragged_key_input"), "ragged_key_input");
        raggedValueInput = raggedFromNetworkMetadata(j.at("ragged_value_input"), "ragged_value_input");
        if (raggedKeyInput->getValues() != keyInput || raggedValueInput->getValues() != valueInput) {
            throw std::runtime_error("Attention serialized ragged key/value values do not match key_input/value_input.");
        }
        if (raggedKeyInput->getBatchSize() != raggedValueInput->getBatchSize() ||
            raggedKeyInput->getMaxTotalValues() != raggedValueInput->getMaxTotalValues() ||
            raggedKeyInput->getOffsets() != raggedValueInput->getOffsets()) {
            throw std::runtime_error(
                "Attention serialized ragged key and value inputs must use the exact same row partition and capacity.");
        }
    } else if (j.contains("ragged_key_input") || j.contains("ragged_value_input")) {
        throw std::runtime_error("Attention serialized ragged key/value metadata requires key_value_ragged=true.");
    }
    if (queryRagged && keyValueRagged && raggedQueryInput->getBatchSize() != raggedKeyInput->getBatchSize()) {
        throw std::runtime_error("Attention serialized ragged query and key/value inputs must have the same logical batch size.");
    }

    std::optional<RaggedTensor> raggedResidualInput = std::nullopt;
    if (j.contains("ragged_residual_input")) {
        raggedResidualInput = raggedFromNetworkMetadata(j.at("ragged_residual_input"), "ragged_residual_input");
        if (!residualInput.has_value() || raggedResidualInput->getValues() != residualInput.value()) {
            throw std::runtime_error("Attention serialized ragged_residual_input values do not match residual_input.");
        }
    }

    Tensor featureOutput = Tensor::deserialize(j.at("feature_output"), archiveReader.get());

    requireRank2FeatureInput(queryInput, "query_input");
    requireRank2FeatureInput(keyInput, "key_input");
    requireRank2FeatureInput(valueInput, "value_input");
    if (keyInput.getDimensions().at(0) != valueInput.getDimensions().at(0)) {
        throw std::runtime_error("Attention serialized key_input and value_input must have the same sequence length.");
    }
    if (queryInput.getDataType() != keyInput.getDataType() || queryInput.getDataType() != valueInput.getDataType()) {
        throw std::runtime_error("Attention serialized query/key/value inputs must use the same dtype.");
    }

    const uint32_t numHeads = j.at("num_heads").get<uint32_t>();
    const uint32_t numKeyValueHeads = j.at("num_key_value_heads").get<uint32_t>();
    const uint32_t headDim = j.at("head_dim").get<uint32_t>();
    const uint32_t valueDim = j.at("value_dim").get<uint32_t>();
    const uint32_t outputFeatures = j.at("output_features").get<uint32_t>();
    const bool hasBias = j.at("has_bias").get<bool>();
    const bool useRope = j.value("use_rope", false);
    const bool ropeInPlace = j.value("rope_in_place", false);
    ThorImplementation::RotaryPositionEmbeddingOptions ropeOptions =
        j.contains("rope_options") ? ropeOptionsFromJson(j.at("rope_options"))
                                   : ThorImplementation::RotaryPositionEmbeddingOptions{};
    const int64_t queryRopePositionOffset = j.value("rope_query_position_offset", ropeOptions.position_offset);
    const int64_t keyRopePositionOffset = j.value("rope_key_position_offset", ropeOptions.position_offset);
    const ThorImplementation::AttentionMaskKind maskKind =
        attentionMaskKindFromString(j.value("mask_kind", std::string("none")));
    const int64_t diagonalLeftBound = j.value("diagonal_left_bound", int64_t{0});
    const int64_t diagonalRightBound = j.value("diagonal_right_bound", int64_t{0});
    const bool useAlibiMask = j.value("use_alibi_mask", false);
    std::optional<double> attentionScale = std::nullopt;
    if (j.contains("attention_scale") && !j.at("attention_scale").is_null()) {
        attentionScale = j.at("attention_scale").get<double>();
    }
    const float sdpaDropoutProbability = j.value("sdpa_dropout_probability", 0.0f);
    const int64_t dropoutSeed = j.value("sdpa_dropout_seed", int64_t{0});
    const int64_t dropoutOffset = j.value("sdpa_dropout_offset", int64_t{0});
    const float outputDropoutProbability = j.value("output_dropout_probability", 0.0f);
    const int64_t outputDropoutSeed = j.value("output_dropout_seed", int64_t{0});
    const DataType weightsDataType = j.at("weights_data_type").get<DataType>();
    const DataType computeDataType = j.at("compute_data_type").get<DataType>();
    const DataType outputDataType = j.at("output_data_type").get<DataType>();

    if (numHeads == 0 || numKeyValueHeads == 0 || numHeads % numKeyValueHeads != 0 || headDim == 0 || valueDim == 0 ||
        outputFeatures == 0) {
        throw std::runtime_error("Attention serialized head/output configuration is invalid.");
    }
    if (!isStorageDType(queryInput.getDataType()) || queryInput.getDataType() != weightsDataType ||
        queryInput.getDataType() != outputDataType || !isComputeDType(computeDataType)) {
        throw std::runtime_error(
            "Attention serialized query/key/value, weights, and output storage dtypes must match FP16/BF16 and compute dtype must be FP32.");
    }
    if (!std::isfinite(sdpaDropoutProbability) || sdpaDropoutProbability < 0.0f || sdpaDropoutProbability >= 1.0f ||
        !std::isfinite(outputDropoutProbability) || outputDropoutProbability < 0.0f || outputDropoutProbability >= 1.0f) {
        throw std::runtime_error("Attention serialized dropout probabilities must be finite and in [0, 1).");
    }
    if (epilogue.has_value() && (residualInput.has_value() || outputDropoutProbability > 0.0f)) {
        throw std::runtime_error("Attention serialized custom epilogue cannot be combined with residual/output dropout.");
    }

    const uint64_t querySequenceLength = queryInput.getDimensions().at(0);
    const uint64_t keyValueSequenceLength = keyInput.getDimensions().at(0);
    const uint64_t queryInputFeatures = queryInput.getDimensions().at(1);
    const uint64_t keyInputFeatures = keyInput.getDimensions().at(1);
    const uint64_t valueInputFeatures = valueInput.getDimensions().at(1);
    if (featureOutput.getDimensions() != std::vector<uint64_t>{querySequenceLength, outputFeatures} ||
        featureOutput.getDataType() != outputDataType) {
        throw std::runtime_error("Attention serialized feature_output does not match query length/output configuration.");
    }
    if (residualInput.has_value()) {
        if (residualInput->getDimensions() != featureOutput.getDimensions() || residualInput->getDataType() != outputDataType) {
            throw std::runtime_error("Attention serialized residual_input must match feature_output shape and dtype.");
        }
        if (queryRagged != raggedResidualInput.has_value()) {
            throw std::runtime_error("Attention serialized residual_input must be ragged exactly when the query is ragged.");
        }
        if (raggedResidualInput.has_value() &&
            (raggedResidualInput->getBatchSize() != raggedQueryInput->getBatchSize() ||
             raggedResidualInput->getMaxTotalValues() != raggedQueryInput->getMaxTotalValues() ||
             raggedResidualInput->getOffsets() != raggedQueryInput->getOffsets())) {
            throw std::runtime_error("Attention serialized ragged residual must use the query row partition.");
        }
    }
    for (const auto& [name, tensor] : epilogueInputBindings) {
        if (tensor.getDimensions() != featureOutput.getDimensions() || tensor.getDataType() != outputDataType) {
            throw std::runtime_error("Attention serialized epilogue input '" + name + "' must match feature_output.");
        }
    }

    const bool useAnyRagged = queryRagged || keyValueRagged;
    if (useAnyRagged && querySequenceLengthsInput.has_value()) {
        throw std::runtime_error("Attention serialized ragged inputs cannot also use explicit sequence lengths.");
    }
    if (useAnyRagged && scoreBiasInput.has_value()) {
        throw std::runtime_error(
            "Attention serialized ragged/mixed mode does not support score_bias_input because cuDNN ragged SDPA backward does not support additive-bias gradients.");
    }
    if (scoreBiasInput.has_value()) {
        requireScoreBiasInput(scoreBiasInput.value(), numHeads, querySequenceLength, keyValueSequenceLength, computeDataType);
    }
    if (querySequenceLengthsInput.has_value()) {
        requireSequenceLengthsInput(querySequenceLengthsInput.value(), "querySequenceLengthsInput");
        requireSequenceLengthsInput(keyValueSequenceLengthsInput.value(), "keyValueSequenceLengthsInput");
    }
    if (queryRopePositionOffsetsInput.has_value()) requireRopePositionOffsetsInput(queryRopePositionOffsetsInput.value(), "queryRopePositionOffsetsInput");
    if (keyRopePositionOffsetsInput.has_value()) requireRopePositionOffsetsInput(keyRopePositionOffsetsInput.value(), "keyRopePositionOffsetsInput");
    if ((queryRopePositionOffsetsInput.has_value() || keyRopePositionOffsetsInput.has_value()) && !useRope) {
        throw std::runtime_error("Attention serialized per-row RoPE position offsets require use_rope=true.");
    }
    if (queryRopePositionOffsetsInput.has_value() && !queryRagged) {
        throw std::runtime_error("Attention serialized per-row query RoPE position offsets require a ragged query input.");
    }
    if (keyRopePositionOffsetsInput.has_value() && !keyValueRagged) {
        throw std::runtime_error("Attention serialized per-row key RoPE position offsets require ragged key/value inputs.");
    }
    const uint64_t maxQ = queryRagged ? raggedQueryInput->getMaxTotalValues() : querySequenceLength;
    const uint64_t maxK = keyValueRagged ? raggedKeyInput->getMaxTotalValues() : keyValueSequenceLength;
    if (const std::optional<std::string> error = ropeFp32SequenceLengthValidationError(
            useRope,
            ropeOptions,
            queryRopePositionOffsetsInput.has_value() ? 0 : queryRopePositionOffset,
            keyRopePositionOffsetsInput.has_value() ? 0 : keyRopePositionOffset,
            maxQ,
            maxK);
        error.has_value()) {
        throw std::runtime_error(error.value());
    }
    if (queryRagged) {
        const json& raggedOutputJson = j.at("ragged_feature_output");
        if (raggedOutputJson.at("values").at("id").get<uint64_t>() != featureOutput.getOriginalId() ||
            raggedOutputJson.at("offsets").at("id").get<uint64_t>() != raggedQueryInput->getOffsets().getOriginalId()) {
            throw std::runtime_error("Attention serialized ragged_feature_output must use feature_output values and the query row partition.");
        }
    }

    std::vector<std::shared_ptr<ParameterSpecification>> parameters;
    if (!j.contains("parameters") || !j.at("parameters").is_object()) {
        throw std::runtime_error("Attention parameters must be an object keyed by parameter name.");
    }
    for (auto it = j.at("parameters").begin(); it != j.at("parameters").end(); ++it) {
        ParameterSpecification parameter = ParameterSpecification::deserialize(it.value(), archiveReader);
        parameters.push_back(std::make_shared<ParameterSpecification>(std::move(parameter)));
    }
    std::vector<std::string> requiredParameterNames{"query_weights", "key_weights", "value_weights", "output_weights"};
    if (hasBias) {
        requiredParameterNames.insert(requiredParameterNames.end(), {"query_bias", "key_bias", "value_bias", "output_bias"});
    }
    for (const std::string& requiredName : requiredParameterNames) {
        bool found = false;
        for (const auto& parameter : parameters) {
            if (parameter != nullptr && parameter->getName() == requiredName) {
                found = true;
                break;
            }
        }
        if (!found) throw std::runtime_error("Attention deserialize did not find required parameter '" + requiredName + "'.");
    }

    const bool useScoreBias = scoreBiasInput.has_value();
    const bool useSequenceLengths = querySequenceLengthsInput.has_value();
    const bool useQueryRopePositionOffsets = queryRopePositionOffsetsInput.has_value();
    const bool useKeyRopePositionOffsets = keyRopePositionOffsetsInput.has_value();

    Attention layer(makeAttentionExpression(querySequenceLength,
                                            keyValueSequenceLength,
                                            queryInputFeatures,
                                            keyInputFeatures,
                                            valueInputFeatures,
                                            outputFeatures,
                                            numHeads,
                                            numKeyValueHeads,
                                            headDim,
                                            valueDim,
                                            hasBias,
                                            useRope,
                                            ropeInPlace,
                                            ropeOptions,
                                            queryRopePositionOffset,
                                            keyRopePositionOffset,
                                            maskKind,
                                            diagonalLeftBound,
                                            diagonalRightBound,
                                            useAlibiMask,
                                            attentionScale,
                                            sdpaDropoutProbability,
                                            dropoutSeed,
                                            dropoutOffset,
                                            outputDropoutProbability,
                                            outputDropoutSeed,
                                            residualInput.has_value(),
                                            useScoreBias,
                                            useSequenceLengths,
                                            queryRagged,
                                            keyValueRagged,
                                            useQueryRopePositionOffsets,
                                            useKeyRopePositionOffsets,
                                            queryRagged ? raggedQueryInput->getBatchSize()
                                                        : (keyValueRagged ? raggedKeyInput->getBatchSize() : 0),
                                            queryRagged ? raggedQueryInput->getOffsetsDataType()
                                                        : ThorImplementation::kDefaultRowPartitionOffsetDataType,
                                            keyValueRagged ? raggedKeyInput->getOffsetsDataType()
                                                           : ThorImplementation::kDefaultRowPartitionOffsetDataType,
                                            queryInput.getDataType(),
                                            weightsDataType,
                                            computeDataType,
                                            outputDataType,
                                            epilogue,
                                            epilogueAuxInputNames),
                    publicAttentionInputNames(useScoreBias,
                                              useSequenceLengths,
                                              queryRagged,
                                              keyValueRagged,
                                              useQueryRopePositionOffsets,
                                              useKeyRopePositionOffsets,
                                              residualInput.has_value(),
                                              epilogueAuxInputNames),
                    {publicAttentionInputInterface(queryInput,
                                                   keyInput,
                                                   valueInput,
                                                   scoreBiasInput,
                                                   querySequenceLengthsInput,
                                                   keyValueSequenceLengthsInput,
                                                   queryRopePositionOffsetsInput,
                                                   keyRopePositionOffsetsInput,
                                                   raggedQueryInput,
                                                   raggedKeyInput,
                                                   residualInput,
                                                   epilogueInputBindings)},
                    {{{"feature_output", featureOutput}}},
                    std::move(parameters),
                    epilogue,
                    epilogueInputBindings,
                    numHeads,
                    numKeyValueHeads,
                    headDim,
                    valueDim,
                    outputFeatures,
                    hasBias,
                    useRope,
                    ropeInPlace,
                    std::move(ropeOptions),
                    queryRopePositionOffset,
                    keyRopePositionOffset,
                    maskKind,
                    diagonalLeftBound,
                    diagonalRightBound,
                    useAlibiMask,
                    attentionScale,
                    sdpaDropoutProbability,
                    dropoutSeed,
                    dropoutOffset,
                    outputDropoutProbability,
                    outputDropoutSeed,
                    residualInput,
                    raggedResidualInput,
                    keyInput,
                    valueInput,
                    scoreBiasInput,
                    querySequenceLengthsInput,
                    keyValueSequenceLengthsInput,
                    queryRopePositionOffsetsInput,
                    keyRopePositionOffsetsInput,
                    raggedQueryInput,
                    raggedKeyInput,
                    raggedValueInput,
                    weightsDataType,
                    computeDataType,
                    outputDataType);
    layer.addToNetwork(network);
}

}  // namespace Thor


namespace {
static const bool registeredAttention = [] {
    Thor::TrainableLayer::register_layer("attention", &Thor::Attention::deserialize);
    return true;
}();
}  // namespace
