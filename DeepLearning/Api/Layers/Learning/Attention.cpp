
#include "DeepLearning/Api/Layers/Learning/Attention.h"
#include "DeepLearning/Implementation/Layers/NeuralNetwork/Attention.h"
#include "Utilities/TensorOperations/Ragged/RowPartitionDTypePolicy.h"

#include "DeepLearning/Api/Initializers/Glorot.h"
#include "DeepLearning/Api/Parameter/ParameterSpecification.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/FusedEquation.h"
#include "Utilities/TensorOperations/Scalar/SetScalar.h"

#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <utility>

using DataType = ThorImplementation::DataType;
using json = nlohmann::json;

namespace {

constexpr const char* kAttentionFeatureInputName = "feature_input";
constexpr const char* kAttentionContextInputName = "context_input";
constexpr const char* kAttentionScoreBiasInputName = "score_bias_input";
constexpr const char* kAttentionQuerySequenceLengthsInputName = "query_sequence_lengths";
constexpr const char* kAttentionKeyValueSequenceLengthsInputName = "key_value_sequence_lengths";
constexpr const char* kAttentionQueryRowPartitionInputName = "query_row_partition";
constexpr const char* kAttentionKeyValueRowPartitionInputName = "key_value_row_partition";
constexpr const char* kAttentionDropoutSeedInputName = "__attention_dropout_seed";
constexpr const char* kAttentionDropoutOffsetInputName = "__attention_dropout_offset";
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
    bool useRope, const ThorImplementation::RotaryPositionEmbeddingOptions& options, uint64_t maximumPossibleSequenceLength) {
    if (!useRope || !ropeScalingUsesSequenceLength(options.scaling_kind)) {
        return std::nullopt;
    }

    if (options.original_max_position_embeddings > kMaxExactlyRepresentableFp32Integer) {
        return "Attention Dynamic-NTK/LongRoPE currently requires original_max_position_embeddings <= 16777216 "
               "because RoPE sequence-length scaling uses FP32 metadata.";
    }

    const uint64_t positivePositionOffset =
        options.position_offset > 0 ? static_cast<uint64_t>(options.position_offset) : uint64_t{0};
    if (positivePositionOffset > kMaxExactlyRepresentableFp32Integer ||
        maximumPossibleSequenceLength > kMaxExactlyRepresentableFp32Integer - positivePositionOffset) {
        return "Attention Dynamic-NTK/LongRoPE currently requires maximum possible sequence length plus positive "
               "position_offset <= 16777216 because RoPE sequence-length scaling uses FP32 metadata. "
               "maximum_possible_sequence_length=" +
               std::to_string(maximumPossibleSequenceLength) + ", positive_position_offset=" +
               std::to_string(positivePositionOffset) + ".";
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

std::vector<std::string> publicAttentionInputNames(bool useContextInput,
                                                   bool useScoreBias,
                                                   bool useSequenceLengths,
                                                   bool useRagged,
                                                   const std::vector<std::string>& epilogueAuxInputNames) {
    std::vector<std::string> names{kAttentionFeatureInputName};
    if (useContextInput) {
        names.push_back(kAttentionContextInputName);
    }
    if (useScoreBias) {
        names.push_back(kAttentionScoreBiasInputName);
    }
    if (useSequenceLengths) {
        names.push_back(kAttentionQuerySequenceLengthsInputName);
        names.push_back(kAttentionKeyValueSequenceLengthsInputName);
    }
    if (useRagged) {
        names.push_back(kAttentionQueryRowPartitionInputName);
        names.push_back(kAttentionKeyValueRowPartitionInputName);
    }
    names.insert(names.end(), epilogueAuxInputNames.begin(), epilogueAuxInputNames.end());
    return names;
}

Thor::CustomLayer::TensorMap publicAttentionInputInterface(const Thor::Tensor& featureInput,
                                                           const std::optional<Thor::Tensor>& contextInput,
                                                           const std::optional<Thor::Tensor>& scoreBiasInput,
                                                           const std::optional<Thor::Tensor>& querySequenceLengthsInput,
                                                           const std::optional<Thor::Tensor>& keyValueSequenceLengthsInput,
                                                           const std::optional<Thor::RaggedTensor>& raggedFeatureInput,
                                                           const std::optional<Thor::RaggedTensor>& raggedContextInput,
                                                           const std::vector<std::pair<std::string, Thor::Tensor>>& epilogueInputBindings) {
    Thor::CustomLayer::TensorMap inputInterface{{kAttentionFeatureInputName, featureInput}};
    if (contextInput.has_value()) {
        inputInterface[kAttentionContextInputName] = contextInput.value();
    }
    if (scoreBiasInput.has_value()) {
        inputInterface[kAttentionScoreBiasInputName] = scoreBiasInput.value();
    }
    if (querySequenceLengthsInput.has_value()) {
        inputInterface[kAttentionQuerySequenceLengthsInputName] = querySequenceLengthsInput.value();
        inputInterface[kAttentionKeyValueSequenceLengthsInputName] = keyValueSequenceLengthsInput.value();
    }
    if (raggedFeatureInput.has_value()) {
        inputInterface[kAttentionQueryRowPartitionInputName] = raggedFeatureInput->getOffsets();
        inputInterface[kAttentionKeyValueRowPartitionInputName] =
            raggedContextInput.has_value() ? raggedContextInput->getOffsets() : raggedFeatureInput->getOffsets();
    }
    for (const auto& [name, tensor] : epilogueInputBindings) {
        inputInterface[name] = tensor;
    }
    return inputInterface;
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
            throw std::invalid_argument("Attention dropoutOffset must be non-negative when dropout is enabled.");
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

constexpr bool kUsePackedQkvProjection = Thor::Attention::USE_PACKED_QKV_PROJECTION;

bool usePackedQkvProjectionForLayer(bool useRope, bool useContextInput = false) {
    // PackedQkvProjection is not being supported anymore as it was shown to be slower.
    // It's being left here as an orphaned reference if there is some future opportunity to gain performance using a packed QKV.
    if constexpr (!kUsePackedQkvProjection) {
        return false;
    } else {
        // RoPE is still a generic expression op and must not consume non-dense Q/K views sliced out of packed QKV.
        // Keep RoPE layers on the legacy split projection path until a layout-aware RoPE materialization path lands.
        return !useRope && !useContextInput;
    }
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
                                                              uint64_t contextInputFeatures,
                                                              uint64_t outputFeatures,
                                                              uint32_t numHeads,
                                                              uint32_t numKeyValueHeads,
                                                              uint32_t headDim,
                                                              uint32_t valueDim,
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
                                                              bool useContextInput,
                                                              bool useScoreBias,
                                                              bool useSequenceLengths,
                                                              bool useRagged,
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

    const bool usePackedQkvProjection = !useRagged && usePackedQkvProjectionForLayer(useRope, useContextInput);
    std::vector<std::string> expectedInputs;
    if (usePackedQkvProjection) {
        expectedInputs = {kAttentionFeatureInputName};
        if (useContextInput) {
            expectedInputs.push_back(kAttentionContextInputName);
        }
        if (useScoreBias) {
            expectedInputs.push_back(kAttentionScoreBiasInputName);
        }
        if (useSequenceLengths) {
            expectedInputs.push_back(kAttentionQuerySequenceLengthsInputName);
            expectedInputs.push_back(kAttentionKeyValueSequenceLengthsInputName);
        }
        if (useRagged) {
            expectedInputs.push_back(kAttentionQueryRowPartitionInputName);
            expectedInputs.push_back(kAttentionKeyValueRowPartitionInputName);
        }
        expectedInputs.push_back("qkv_weights");
        expectedInputs.push_back("output_weights");
        if (hasBias) {
            expectedInputs.push_back("qkv_bias");
            expectedInputs.push_back("output_bias");
        }
    } else {
        expectedInputs = {kAttentionFeatureInputName};
        if (useContextInput) {
            expectedInputs.push_back(kAttentionContextInputName);
        }
        if (useScoreBias) {
            expectedInputs.push_back(kAttentionScoreBiasInputName);
        }
        if (useSequenceLengths) {
            expectedInputs.push_back(kAttentionQuerySequenceLengthsInputName);
            expectedInputs.push_back(kAttentionKeyValueSequenceLengthsInputName);
        }
        if (useRagged) {
            expectedInputs.push_back(kAttentionQueryRowPartitionInputName);
            expectedInputs.push_back(kAttentionKeyValueRowPartitionInputName);
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
    }
    expectedInputs.insert(expectedInputs.end(), epilogueAuxInputNames.begin(), epilogueAuxInputNames.end());

    return DynamicExpression(
        expectedInputs,
        {"feature_output"},
        [querySequenceLength,
         keyValueSequenceLength,
         queryInputFeatures,
         contextInputFeatures,
         outputFeatures,
         usePackedQkvProjection,
         numHeads,
         numKeyValueHeads,
         headDim,
         valueDim,
         hasBias,
         useRope,
         ropeInPlace,
         ropeOptions,
         maskKind,
         diagonalLeftBound,
         diagonalRightBound,
         useAlibiMask,
         attentionScale,
         dropoutProbability,
         dropoutSeed,
         dropoutOffset,
         useContextInput,
         useScoreBias,
         useSequenceLengths,
         useRagged,
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
            (void)stream;

            Tensor featureInput = inputs.at(kAttentionFeatureInputName);
            const auto queryInputDims = featureInput.getDimensions();
            const uint64_t batch = useRagged ? raggedBatchSize : (queryInputDims.empty() ? 0 : queryInputDims.front());
            const std::vector<uint64_t> expectedQueryInputDimensions =
                useRagged ? std::vector<uint64_t>{querySequenceLength, queryInputFeatures}
                          : std::vector<uint64_t>{batch, querySequenceLength, queryInputFeatures};
            if (batch == 0 || queryInputDims != expectedQueryInputDimensions) {
                throw std::runtime_error(useRagged
                                             ? "Attention runtime ragged feature values must be [max_total_values, query_features]."
                                             : "Attention runtime feature input must be [batch, query_sequence, query_features].");
            }
            if (featureInput.getDataType() != inputDType) {
                throw std::runtime_error("Attention runtime feature input dtype does not match the API input dtype.");
            }

            const std::vector<uint64_t> runtimeFeatureOutputDimensions =
                useRagged ? std::vector<uint64_t>{querySequenceLength, outputFeatures}
                          : std::vector<uint64_t>{batch, querySequenceLength, outputFeatures};
            if (outputs.contains("feature_output")) {
                const Tensor& featureOutput = outputs.at("feature_output");
                if (featureOutput.getDimensions() != runtimeFeatureOutputDimensions) {
                    throw std::runtime_error(useRagged
                                                 ? "Attention runtime ragged feature output values must remain [max_total_values, output_features]."
                                                 : "Attention runtime feature output shape must remain [batch, query_sequence, output_features].");
                }
                if (featureOutput.getDataType() != outputDType) {
                    throw std::runtime_error("Attention runtime feature output dtype must match outputDataType.");
                }
                if (featureOutput.getPlacement() != featureInput.getPlacement()) {
                    throw std::runtime_error("Attention runtime feature output placement must match the feature input placement.");
                }
            }

            Tensor contextInput = useContextInput ? inputs.at(kAttentionContextInputName) : featureInput;
            const auto contextInputDims = contextInput.getDimensions();
            const std::vector<uint64_t> expectedContextInputDimensions =
                useRagged ? std::vector<uint64_t>{keyValueSequenceLength, contextInputFeatures}
                          : std::vector<uint64_t>{batch, keyValueSequenceLength, contextInputFeatures};
            if (contextInputDims != expectedContextInputDimensions) {
                throw std::runtime_error(useRagged
                                             ? "Attention runtime ragged context values must be [max_total_values, context_features]."
                                             : "Attention runtime context input must be [batch, key_value_sequence, context_features].");
            }
            if (contextInput.getDataType() != inputDType) {
                throw std::runtime_error("Attention runtime context input dtype must match feature input dtype.");
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
                if (auxTensor.getPlacement() != featureInput.getPlacement()) {
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
            const uint64_t qkvWidth = queryWidth + keyWidth + valueWidth;

            auto validateWeight = [&](const char* name, uint64_t in, uint64_t out) {
                const Tensor& w = inputs.at(name);
                if (w.getDimensions() != std::vector<uint64_t>{in, out}) {
                    throw std::runtime_error(std::string("Attention ") + name + " shape mismatch.");
                }
                if (w.getDataType() != weightsDType) {
                    throw std::runtime_error(std::string("Attention ") + name + " dtype mismatch.");
                }
            };
            if (usePackedQkvProjection) {
                validateWeight("qkv_weights", queryInputFeatures, qkvWidth);
            } else {
                validateWeight("query_weights", queryInputFeatures, queryWidth);
                validateWeight("key_weights", contextInputFeatures, keyWidth);
                validateWeight("value_weights", contextInputFeatures, valueWidth);
            }
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
                if (usePackedQkvProjection) {
                    validateBias("qkv_bias", qkvWidth);
                } else {
                    validateBias("query_bias", queryWidth);
                    validateBias("key_bias", keyWidth);
                    validateBias("value_bias", valueWidth);
                }
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
            if (useRagged) {
                queryRaggedOffsets = normalizeRaggedOffsets(kAttentionQueryRowPartitionInputName, queryRowPartitionDType);
                keyValueRaggedOffsets =
                    normalizeRaggedOffsets(kAttentionKeyValueRowPartitionInputName, keyValueRowPartitionDType);
            }

            Expression x = Expression::input(kAttentionFeatureInputName, inputDType, inputDType);
            if (!useRagged) {
                x = x.reshape({batch * querySequenceLength, queryInputFeatures});
            }
            Expression kvx = useContextInput ? Expression::input(kAttentionContextInputName, inputDType, inputDType) : x;
            if (useContextInput && !useRagged) {
                kvx = kvx.reshape({batch * keyValueSequenceLength, contextInputFeatures});
            }
            std::optional<Expression> scoreBiasExpr;
            if (useScoreBias) {
                scoreBiasExpr = Expression::input(kAttentionScoreBiasInputName, computeDType, computeDType);
            }

            struct ProjectedQkv {
                Expression q;
                Expression k;
                Expression v;
            };

            auto buildSplitProjection = [&]() -> ProjectedQkv {
                Expression qw = Expression::input("query_weights", weightsDType, weightsDType);
                Expression kw = Expression::input("key_weights", weightsDType, weightsDType);
                Expression vw = Expression::input("value_weights", weightsDType, weightsDType);

                Expression q = Expression::matmul(x, qw, false, false, computeDType, outputDType);
                Expression k = Expression::matmul(kvx, kw, false, false, computeDType, outputDType);
                Expression v = Expression::matmul(kvx, vw, false, false, computeDType, outputDType);
                if (hasBias) {
                    q = q + Expression::input("query_bias", weightsDType, weightsDType);
                    k = k + Expression::input("key_bias", weightsDType, weightsDType);
                    v = v + Expression::input("value_bias", weightsDType, weightsDType);
                }

                if (useRagged) {
                    // Canonical ragged Attention remains physically packed THD: the row partition owns the logical batch/sequence
                    // structure, while Q/K/V values contain only packed token-major storage.
                    q = q.reshape({querySequenceLength, numHeads, headDim}).withOutputDType(outputDType);
                    k = k.reshape({keyValueSequenceLength, numKeyValueHeads, headDim}).withOutputDType(outputDType);
                    v = v.reshape({keyValueSequenceLength, numKeyValueHeads, valueDim}).withOutputDType(outputDType);
                } else {
                    // Dense high-level attention remains logical [B,S,H,D]. AttentionTensorLayout::BSHD tells the cuDNN
                    // adapter how to reinterpret that token-major storage for cuDNN's [B,H,S,D] descriptor contract.
                    q = q.reshape({batch, querySequenceLength, numHeads, headDim}).withOutputDType(outputDType);
                    k = k.reshape({batch, keyValueSequenceLength, numKeyValueHeads, headDim}).withOutputDType(outputDType);
                    v = v.reshape({batch, keyValueSequenceLength, numKeyValueHeads, valueDim}).withOutputDType(outputDType);
                }
                return ProjectedQkv{std::move(q), std::move(k), std::move(v)};
            };

            auto buildPackedProjection = [&]() -> ProjectedQkv {
                if (useRagged) {
                    throw std::runtime_error("Packed QKV projection is not used by canonical ragged Attention.");
                }
                Expression qkvWeights = Expression::input("qkv_weights", weightsDType, weightsDType);
                Expression qkv = Expression::matmul(x, qkvWeights, false, false, computeDType, outputDType);
                if (hasBias) {
                    qkv = qkv + Expression::input("qkv_bias", weightsDType, weightsDType);
                }

                // Packed QKV produces one token-major [B*S, Q+K+V] buffer.  Q/K/V are zero-copy strided views into
                // that buffer, with row stride equal to the full packed width.  This is the final no-split form needed
                // for packed-QKV forward and for packed-QKV training once view backward accumulates dQ/dK/dV to dQKV.
                const uint64_t batchStride = querySequenceLength * qkvWidth;
                Expression q = qkv.stridedView({batch, querySequenceLength, numHeads, headDim}, {batchStride, qkvWidth, headDim, 1}, 0)
                                   .withOutputDType(outputDType);
                Expression k =
                    qkv.stridedView({batch, keyValueSequenceLength, numKeyValueHeads, headDim}, {batchStride, qkvWidth, headDim, 1}, queryWidth)
                        .withOutputDType(outputDType);
                Expression v =
                    qkv.stridedView(
                           {batch, keyValueSequenceLength, numKeyValueHeads, valueDim}, {batchStride, qkvWidth, valueDim, 1}, queryWidth + keyWidth)
                        .withOutputDType(outputDType);
                return ProjectedQkv{std::move(q), std::move(k), std::move(v)};
            };

            ProjectedQkv projected = [&]() -> ProjectedQkv {
                if constexpr (kUsePackedQkvProjection) {
                    if (usePackedQkvProjection) {
                        return buildPackedProjection();
                    }
                }
                return buildSplitProjection();
            }();

            Expression q = std::move(projected.q);
            Expression k = std::move(projected.k);
            Expression v = std::move(projected.v);

            if (useRope) {
                ThorImplementation::RotaryPositionEmbeddingOptions opts = ropeOptions;
                opts.sequence_axis = useRagged ? 0 : 1;
                opts.head_dim_axis = useRagged ? 2 : 3;
                opts.allow_in_place_materialization = ropeInPlace;
                if (!opts.compute_dtype.has_value()) {
                    opts.compute_dtype = computeDType;
                }
                if (!opts.output_dtype.has_value()) {
                    opts.output_dtype = outputDType;
                }

                const bool scalingNeedsLogicalSequenceLength =
                    opts.scaling_kind == ThorImplementation::RotaryScalingKind::DynamicNTK ||
                    opts.scaling_kind == ThorImplementation::RotaryScalingKind::LongRope;
                std::optional<Expression> effectiveSequenceLength;
                if (scalingNeedsLogicalSequenceLength && useRagged) {
                    // Packed ragged storage uses maxTotalValues as its physical sequence-axis capacity. Dynamic-NTK
                    // and LongRoPE instead require the longest logical sequence in the batch. Derive max(diff(offsets))
                    // entirely on-device; the row partition remains structural, non-differentiable metadata.
                    Expression offsets = Expression::input(
                        kAttentionQueryRowPartitionInputName, queryRowPartitionDType, queryRowPartitionDType);
                    Expression starts = offsets.stridedView({batch}, {1}, 0);
                    Expression ends = offsets.stridedView({batch}, {1}, 1);
                    // Subtract while offsets are still exact UINT32/UINT64 metadata. Only the resulting row lengths are
                    // converted to FP32, which is the numeric domain used by the RoPE scaling formulas themselves.
                    effectiveSequenceLength = (ends - starts).cast(DataType::FP32).reduce_max({0}, {});
                } else if (scalingNeedsLogicalSequenceLength && useSequenceLengths) {
                    // Dense variable-length attention must likewise use active logical lengths rather than padded S.
                    // Q/K share one RoPE frequency basis, so cross-attention uses the larger active maximum.
                    Expression qLengths = Expression::input(
                        kAttentionQuerySequenceLengthsInputName, DataType::INT32, DataType::INT32);
                    Expression kvLengths = Expression::input(
                        kAttentionKeyValueSequenceLengthsInputName, DataType::INT32, DataType::INT32);
                    effectiveSequenceLength = qLengths.cast(DataType::FP32)
                                                  .reduce_max({0}, {})
                                                  .max(kvLengths.cast(DataType::FP32).reduce_max({0}, {}));
                }

                if (effectiveSequenceLength.has_value()) {
                    q = q.rotaryPositionEmbeddingWithEffectiveSequenceLength(effectiveSequenceLength.value(), opts);
                    k = k.rotaryPositionEmbeddingWithEffectiveSequenceLength(effectiveSequenceLength.value(), opts);
                } else {
                    q = q.rotaryPositionEmbedding(opts);
                    k = k.rotaryPositionEmbedding(opts);
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
                activeOptions.dropout_probability = enableDropout ? dropoutProbability : 0.0f;

                if (useRagged) {
                    Expression qRaggedOffsetsExpr = Expression::input(
                        kAttentionQueryRowPartitionInputName, queryRowPartitionDType, queryRowPartitionDType);
                    Expression kvRaggedOffsetsExpr = Expression::input(
                        kAttentionKeyValueRowPartitionInputName, keyValueRowPartitionDType, keyValueRowPartitionDType);
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
                {useRagged ? querySequenceLength : batch * querySequenceLength, outputFeatures};
            auto buildProjectedOutput = [&](Expression attn) -> Expression {
                Expression merged = attn.reshape(
                    {useRagged ? querySequenceLength : batch * querySequenceLength,
                     checkedMul(numHeads, valueDim, "merged head width")});
                Expression outputWeights = Expression::input("output_weights", weightsDType, weightsDType);
                Expression out = Expression::matmul(merged, outputWeights, false, false, computeDType, outputDType);
                if (hasBias) {
                    out = out + Expression::input("output_bias", weightsDType, weightsDType);
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
                if (!useRagged) {
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

            auto expressionOutputs = Expression::outputs({{"feature_output", buildProjectedOutput(buildSdpa(dropoutProbability > 0.0f))}});
            std::shared_ptr<FusedEquation> evaluationEquation;
            if (dropoutProbability > 0.0f) {
                auto validationExpressionOutputs =
                    Expression::outputs({{"feature_output", buildProjectedOutput(buildSdpa(false))}});
                evaluationEquation = std::make_shared<FusedEquation>(
                    FusedEquation::compile(validationExpressionOutputs.physicalOutputs(), stream.getGpuNum()));
            }

            DynamicExpression::TensorMap stampInputs = inputs;
            if (useScoreBias) {
                stampInputs[kAttentionScoreBiasInputName] = scoreBiasInput.value();
            }
            if (useSequenceLengths) {
                stampInputs[kAttentionQuerySequenceLengthsInputName] = querySequenceLengths.value();
                stampInputs[kAttentionKeyValueSequenceLengthsInputName] = keyValueSequenceLengths.value();
            }
            if (useRagged) {
                stampInputs[kAttentionQueryRowPartitionInputName] = queryRaggedOffsets.value();
                stampInputs[kAttentionKeyValueRowPartitionInputName] = keyValueRaggedOffsets.value();
            }
            std::unordered_map<std::string, TensorScalarBinding> tensorScalarInputs;
            std::function<void(Stream&)> preForwardHook;
            if (dropoutProbability > 0.0f) {
                auto dropoutState = std::make_shared<AttentionDropoutRuntimeState>(dropoutSeed, dropoutOffset);
                dropoutState->setOffsetAdvance(checkedDropoutOffsetAdvance(batch, numHeads, querySequenceLength, keyValueSequenceLength));
                tensorScalarInputs[kAttentionDropoutSeedInputName] = dropoutState->seedBinding(featureInput.getPlacement());
                tensorScalarInputs[kAttentionDropoutOffsetInputName] = dropoutState->offsetBinding(featureInput.getPlacement());
                preForwardHook = [dropoutState](Stream& runStream) { dropoutState->uploadForForward(runStream); };
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
            if (evaluationEquation) {
                build.execution_variants.emplace(
                    kAttentionEvaluationVariant,
                    DynamicExpressionVariant{
                        .equation = std::move(evaluationEquation),
                        .tensor_scalar_inputs = {},
                        .pre_forward_hook = {},
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
        dropoutProbability > 0.0f
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
        kAttentionFeatureInputName,
        kAttentionContextInputName,
        kAttentionScoreBiasInputName,
        kAttentionQuerySequenceLengthsInputName,
        kAttentionKeyValueSequenceLengthsInputName,
        kAttentionQueryRowPartitionInputName,
        kAttentionKeyValueRowPartitionInputName,
        "feature_output",
        "qkv_weights",
        "query_weights",
        "key_weights",
        "value_weights",
        "output_weights",
        "qkv_bias",
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
    if (!_featureInput.has_value()) {
        throw std::invalid_argument("Attention::Builder requires featureInput().");
    }
    if (!_numHeads.has_value()) {
        throw std::invalid_argument("Attention::Builder requires numHeads().");
    }
    const bool useRagged = _raggedFeatureInput.has_value();
    if (useRagged) {
        requireRaggedFeatureInput(_raggedFeatureInput.value(), "featureInput(RaggedTensor)");
        if (_contextInput.has_value() != _raggedContextInput.has_value()) {
            throw std::invalid_argument(
                "Attention ragged feature input requires a RaggedTensor context input when cross-attention is used.");
        }
        if (_raggedContextInput.has_value()) {
            requireRaggedFeatureInput(_raggedContextInput.value(), "contextInput(RaggedTensor)");
            if (_raggedContextInput->getBatchSize() != _raggedFeatureInput->getBatchSize()) {
                throw std::invalid_argument("Attention ragged query and key/value inputs must have the same logical batch size.");
            }
        }
    } else {
        requireRank2FeatureInput(_featureInput.value(), "feature input");
        if (_raggedContextInput.has_value()) {
            throw std::invalid_argument("Attention dense feature input cannot be combined with a RaggedTensor context input.");
        }
    }
    if (_contextInput.has_value()) {
        if (!useRagged) {
            requireRank2FeatureInput(_contextInput.value(), "context input");
        }
        if (_contextInput->getDataType() != _featureInput->getDataType()) {
            throw std::invalid_argument("Attention context input dtype must match feature input dtype for the current training path.");
        }
    }
    if (_querySequenceLengthsInput.has_value() != _keyValueSequenceLengthsInput.has_value()) {
        throw std::invalid_argument(
            "Attention requires both querySequenceLengthsInput and keyValueSequenceLengthsInput.");
    }
    if (_querySequenceLengthsInput.has_value()) {
        requireSequenceLengthsInput(_querySequenceLengthsInput.value(), "querySequenceLengthsInput");
        requireSequenceLengthsInput(_keyValueSequenceLengthsInput.value(), "keyValueSequenceLengthsInput");
    }
    if (useRagged && _querySequenceLengthsInput.has_value()) {
        throw std::invalid_argument(
            "Attention RaggedTensor inputs already define sequence lengths through their row partitions; "
            "querySequenceLengthsInput/keyValueSequenceLengthsInput are not allowed in ragged mode.");
    }
    if (useRagged && _useRope.value_or(false) && _raggedContextInput.has_value() &&
        _raggedFeatureInput->getOffsets() != _raggedContextInput->getOffsets()) {
        throw std::invalid_argument(
            "Attention ragged cross-attention with RoPE requires query and key/value to share the same row partition. "
            "Independent ragged partitions need explicit positional semantics, which Attention does not infer.");
    }
    const uint64_t maximumPossibleRopeSequenceLength =
        useRagged
            ? _raggedFeatureInput->getMaxTotalValues()
            : std::max(_featureInput->getDimensions().at(0),
                       _contextInput.has_value() ? _contextInput->getDimensions().at(0) : _featureInput->getDimensions().at(0));
    if (const std::optional<std::string> error = ropeFp32SequenceLengthValidationError(
            _useRope.value_or(false), _ropeOptions.value_or(ThorImplementation::RotaryPositionEmbeddingOptions{}),
            maximumPossibleRopeSequenceLength);
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

    const DataType inputDType = _featureInput->getDataType();
    const DataType weightsDType = _weightsDataType.value_or(inputDType);
    const DataType outputDType = _outputDataType.value_or(inputDType);
    const DataType computeDType = _computeDataType.value_or(DataType::FP32);
    if (!isStorageDType(inputDType)) {
        throw std::invalid_argument("Attention feature input dtype must be fp16 or bf16 for the current training path. Got " +
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
            "Attention requires feature/context inputs, projection weights, and attention output storage to use the same "
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
    const float dropoutProbability = _dropoutProbability.value_or(0.0f);
    if (!std::isfinite(dropoutProbability) || dropoutProbability < 0.0f || dropoutProbability >= 1.0f) {
        throw std::invalid_argument("Attention dropoutProbability must be finite and in [0, 1).");
    }
    if (dropoutProbability > 0.0f && (maskKind == ThorImplementation::AttentionMaskKind::CausalBottomRight ||
                                      maskKind == ThorImplementation::AttentionMaskKind::SlidingWindowBottomRight)) {
        throw std::invalid_argument("Attention dropout cannot currently be combined with bottom-right/decode masks in cuDNN SDPA.");
    }
    if (dropoutProbability > 0.0f && _dropoutOffset.value_or(0) < 0) {
        throw std::invalid_argument("Attention dropoutOffset must be non-negative when dropout is enabled.");
    }
    if (_ropeInPlace.value_or(false) && !_useRope.value_or(false)) {
        throw std::invalid_argument("Attention ropeInPlace requires useRope to be enabled.");
    }
    if (_attentionScale.has_value() && (!std::isfinite(_attentionScale.value()) || _attentionScale.value() <= 0.0)) {
        throw std::invalid_argument("Attention attentionScale must be finite and positive.");
    }
    if (useRagged && _scoreBiasInput.has_value()) {
        throw std::invalid_argument(
            "Attention ragged mode does not support scoreBiasInput because the current cuDNN ragged SDPA backward path "
            "does not support additive-bias gradients.");
    }
    if (_scoreBiasInput.has_value()) {
        const std::vector<uint64_t> queryDims = _featureInput->getDimensions();
        const std::vector<uint64_t> contextDims = _contextInput.has_value() ? _contextInput->getDimensions() : queryDims;
        requireScoreBiasInput(_scoreBiasInput.value(), _numHeads.value(), queryDims.at(0), contextDims.at(0), computeDType);
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
        _featureInput->getDimensions().at(0),
        _outputFeatures.value_or(static_cast<uint32_t>(_featureInput->getDimensions().at(1))),
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
        if (!_featureInput.has_value() || !_numHeads.has_value()) {
            throw std::invalid_argument("Attention headDim default requires featureInput and numHeads.");
        }
        const uint64_t inputFeatures = _featureInput->getDimensions().at(1);
        if (inputFeatures % _numHeads.value() != 0) {
            throw std::invalid_argument("Attention default headDim requires input features divisible by numHeads.");
        }
        _headDim = static_cast<uint32_t>(inputFeatures / _numHeads.value());
    }
    if (!_valueDim.has_value()) {
        _valueDim = _headDim.value();
    }
    if (!_outputFeatures.has_value() && _featureInput.has_value()) {
        _outputFeatures = static_cast<uint32_t>(_featureInput->getDimensions().at(1));
    }
    if (!_hasBias.has_value()) {
        _hasBias = false;
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
    if (!_useRope.has_value()) {
        _useRope = false;
    }
    if (!_ropeInPlace.has_value()) {
        _ropeInPlace = false;
    }
    if (!_ropeOptions.has_value()) {
        _ropeOptions = ThorImplementation::RotaryPositionEmbeddingOptions{};
    }
    if (!_weightsDataType.has_value() && _featureInput.has_value()) {
        _weightsDataType = _featureInput->getDataType();
    }
    if (!_computeDataType.has_value()) {
        _computeDataType = DataType::FP32;
    }
    if (!_outputDataType.has_value() && _featureInput.has_value()) {
        _outputDataType = _featureInput->getDataType();
    }
    if (_weightsInitializer == nullptr) {
        _weightsInitializer = Glorot::Builder().build();
    }
    if (_biasInitializer == nullptr) {
        _biasInitializer = Glorot::Builder().build();
    }

    verifyConfig();

    const auto inputDims = _featureInput->getDimensions();
    const auto contextDims = _contextInput.has_value() ? _contextInput->getDimensions() : inputDims;
    const uint64_t querySequenceLength = inputDims.at(0);
    const uint64_t keyValueSequenceLength = contextDims.at(0);
    const uint64_t queryInputFeatures = inputDims.at(1);
    const uint64_t contextInputFeatures = contextDims.at(1);
    const uint64_t qWidth = checkedMul(_numHeads.value(), _headDim.value(), "query projection width");
    const uint64_t kvKeyWidth = checkedMul(_numKeyValueHeads.value(), _headDim.value(), "key projection width");
    const uint64_t kvValueWidth = checkedMul(_numKeyValueHeads.value(), _valueDim.value(), "value projection width");
    const uint64_t mergedWidth = checkedMul(_numHeads.value(), _valueDim.value(), "merged head width");

    const uint64_t qkvWidth = qWidth + kvKeyWidth + kvValueWidth;
    const bool useScoreBias = _scoreBiasInput.has_value();
    const bool useSequenceLengths = _querySequenceLengthsInput.has_value();
    const bool useRagged = _raggedFeatureInput.has_value();
    const bool usePackedQkvProjection = !useRagged && usePackedQkvProjectionForLayer(_useRope.value(), _contextInput.has_value());
    const std::vector<std::string> epilogueAuxNames = epilogueAuxInputNames();

    std::vector<std::shared_ptr<ParameterSpecification>> parameters;
    if (usePackedQkvProjection) {
        parameters.push_back(
            makeParameter("qkv_weights", {queryInputFeatures, qkvWidth}, _weightsDataType.value(), _weightsInitializer, _optimizer));
    } else {
        parameters.push_back(
            makeParameter("query_weights", {queryInputFeatures, qWidth}, _weightsDataType.value(), _weightsInitializer, _optimizer));
        parameters.push_back(
            makeParameter("key_weights", {contextInputFeatures, kvKeyWidth}, _weightsDataType.value(), _weightsInitializer, _optimizer));
        parameters.push_back(
            makeParameter("value_weights", {contextInputFeatures, kvValueWidth}, _weightsDataType.value(), _weightsInitializer, _optimizer));
    }
    parameters.push_back(
        makeParameter("output_weights", {mergedWidth, _outputFeatures.value()}, _weightsDataType.value(), _weightsInitializer, _optimizer));
    if (_hasBias.value()) {
        if (usePackedQkvProjection) {
            parameters.push_back(makeParameter("qkv_bias", {qkvWidth}, _weightsDataType.value(), _biasInitializer, _optimizer));
        } else {
            parameters.push_back(makeParameter("query_bias", {qWidth}, _weightsDataType.value(), _biasInitializer, _optimizer));
            parameters.push_back(makeParameter("key_bias", {kvKeyWidth}, _weightsDataType.value(), _biasInitializer, _optimizer));
            parameters.push_back(makeParameter("value_bias", {kvValueWidth}, _weightsDataType.value(), _biasInitializer, _optimizer));
        }
        parameters.push_back(
            makeParameter("output_bias", {_outputFeatures.value()}, _weightsDataType.value(), _biasInitializer, _optimizer));
    }

    Tensor output(_outputDataType.value(), {querySequenceLength, _outputFeatures.value()});
    Attention layer(makeAttentionExpression(querySequenceLength,
                                            keyValueSequenceLength,
                                            queryInputFeatures,
                                            contextInputFeatures,
                                            _outputFeatures.value(),
                                            _numHeads.value(),
                                            _numKeyValueHeads.value(),
                                            _headDim.value(),
                                            _valueDim.value(),
                                            _hasBias.value(),
                                            _useRope.value(),
                                            _ropeInPlace.value(),
                                            _ropeOptions.value(),
                                            _maskKind.value(),
                                            _diagonalLeftBound.value(),
                                            _diagonalRightBound.value(),
                                            _useAlibiMask.value(),
                                            _attentionScale,
                                            _dropoutProbability.value(),
                                            _dropoutSeed.value(),
                                            _dropoutOffset.value(),
                                            _contextInput.has_value(),
                                            useScoreBias,
                                            useSequenceLengths,
                                            useRagged,
                                            useRagged ? _raggedFeatureInput->getBatchSize() : 0,
                                            useRagged ? _raggedFeatureInput->getOffsetsDataType()
                                                      : ThorImplementation::kDefaultRowPartitionOffsetDataType,
                                            useRagged
                                                ? (_raggedContextInput.has_value() ? _raggedContextInput->getOffsetsDataType()
                                                                                  : _raggedFeatureInput->getOffsetsDataType())
                                                : ThorImplementation::kDefaultRowPartitionOffsetDataType,
                                            _featureInput->getDataType(),
                                            _weightsDataType.value(),
                                            _computeDataType.value(),
                                            _outputDataType.value(),
                                            _epilogue,
                                            epilogueAuxNames),
                    publicAttentionInputNames(
                        _contextInput.has_value(), useScoreBias, useSequenceLengths, useRagged, epilogueAuxNames),
                    {publicAttentionInputInterface(_featureInput.value(),
                                                   _contextInput,
                                                   _scoreBiasInput,
                                                   _querySequenceLengthsInput,
                                                   _keyValueSequenceLengthsInput,
                                                   _raggedFeatureInput,
                                                   _raggedContextInput,
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
                    _maskKind.value(),
                    _diagonalLeftBound.value(),
                    _diagonalRightBound.value(),
                    _useAlibiMask.value(),
                    _attentionScale,
                    _dropoutProbability.value(),
                    _dropoutSeed.value(),
                    _dropoutOffset.value(),
                    _contextInput,
                    _scoreBiasInput,
                    _querySequenceLengthsInput,
                    _keyValueSequenceLengthsInput,
                    _raggedFeatureInput,
                    _raggedContextInput,
                    _weightsDataType.value(),
                    _computeDataType.value(),
                    _outputDataType.value());

    layer.addToNetwork(_network.value());
    return layer;
}


json Attention::architectureJson() const {
    json j;
    j["factory"] = Layer::Factory::Learning.value();
    j["version"] = "1.0.0";
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
    j["rope_options"] = ropeOptionsToJson(ropeOptions);
    j["mask_kind"] = attentionMaskKindToString(maskKind);
    j["diagonal_left_bound"] = diagonalLeftBound;
    j["diagonal_right_bound"] = diagonalRightBound;
    j["use_alibi_mask"] = useAlibiMask;
    j["attention_scale"] = attentionScale.has_value() ? json(attentionScale.value()) : json(nullptr);
    j["dropout_probability"] = dropoutProbability;
    j["dropout_seed"] = dropoutSeed;
    j["dropout_offset"] = dropoutOffset;
    j["use_cross_attention"] = contextInput.has_value();
    j["use_score_bias"] = scoreBiasInput.has_value();
    j["use_sequence_lengths"] = querySequenceLengthsInput.has_value();
    j["use_ragged"] = raggedFeatureInput.has_value();
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

    const std::optional<Tensor> input = getFeatureInput();
    const std::optional<Tensor> output = getFeatureOutput();
    if (!input.has_value() || !output.has_value()) {
        throw std::runtime_error("Attention serialization requires one feature input and one feature output.");
    }
    j["feature_input"] = input.value().architectureJson();
    if (contextInput.has_value()) {
        j["context_input"] = contextInput.value().architectureJson();
    }
    if (scoreBiasInput.has_value()) {
        j["score_bias_input"] = scoreBiasInput.value().architectureJson();
    }
    if (querySequenceLengthsInput.has_value()) {
        j["query_sequence_lengths_input"] = querySequenceLengthsInput.value().architectureJson();
        j["key_value_sequence_lengths_input"] = keyValueSequenceLengthsInput.value().architectureJson();
    }
    if (raggedFeatureInput.has_value()) {
        j["ragged_feature_input"] = raggedFeatureInput->architectureJson();
        if (raggedContextInput.has_value()) {
            j["ragged_context_input"] = raggedContextInput->architectureJson();
        }
        if (!raggedFeatureOutput.has_value()) {
            throw std::runtime_error("Attention ragged serialization requires a logical RaggedTensor output.");
        }
        j["ragged_feature_output"] = raggedFeatureOutput->architectureJson();
    }
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
    if (j.at("version").get<std::string>() != "1.0.0") {
        throw std::runtime_error("Unsupported version in Attention::deserialize: " + j.at("version").get<std::string>());
    }
    if (j.at("layer_type").get<std::string>() != "attention") {
        throw std::runtime_error("Layer type mismatch in Attention::deserialize: " + j.at("layer_type").get<std::string>());
    }

    const uint64_t inputOriginalId = j.at("feature_input").at("id").get<uint64_t>();
    Tensor featureInput = network->getApiTensorByOriginalId(inputOriginalId);
    std::optional<Tensor> contextInput = std::nullopt;
    if (j.value("use_cross_attention", false) || j.contains("context_input")) {
        if (!j.contains("context_input")) {
            throw std::runtime_error("Attention deserialize missing context_input.");
        }
        contextInput = network->getApiTensorByOriginalId(j.at("context_input").at("id").get<uint64_t>());
    }
    std::optional<Tensor> scoreBiasInput = std::nullopt;
    if (j.value("use_score_bias", false) || j.contains("score_bias_input")) {
        if (!j.contains("score_bias_input")) {
            throw std::runtime_error("Attention deserialize missing score_bias_input.");
        }
        scoreBiasInput = network->getApiTensorByOriginalId(j.at("score_bias_input").at("id").get<uint64_t>());
    }

    std::vector<std::pair<std::string, Tensor>> epilogueInputBindings;
    std::vector<std::string> epilogueAuxInputNames;
    if (j.contains("epilogue_inputs")) {
        if (!j.at("epilogue_inputs").is_array()) {
            throw std::runtime_error("Attention epilogue_inputs must be an array.");
        }
        std::set<std::string> seenEpilogueInputNames;
        for (const json& epilogueInputJson : j.at("epilogue_inputs")) {
            if (!epilogueInputJson.is_object() || !epilogueInputJson.contains("name") || !epilogueInputJson.contains("tensor")) {
                throw std::runtime_error("Attention epilogue_inputs entries must contain name and tensor fields.");
            }
            const std::string inputName = epilogueInputJson.at("name").get<std::string>();
            validateEpilogueAuxInputName(inputName);
            if (!seenEpilogueInputNames.insert(inputName).second) {
                throw std::runtime_error("Attention serialized epilogue input name is duplicated: " + inputName + ".");
            }
            const json& tensorJson = epilogueInputJson.at("tensor");
            if (!tensorJson.is_object() || !tensorJson.contains("id")) {
                throw std::runtime_error("Attention serialized epilogue input tensor metadata is invalid.");
            }
            const uint64_t originalTensorId = tensorJson.at("id").get<uint64_t>();
            epilogueInputBindings.emplace_back(inputName, network->getApiTensorByOriginalId(originalTensorId));
            epilogueAuxInputNames.push_back(inputName);
        }
    }

    std::optional<ThorImplementation::Expression> epilogue = std::nullopt;
    if (j.contains("epilogue") && !j.at("epilogue").is_null()) {
        if (!j.at("epilogue").is_object()) {
            throw std::runtime_error("Attention epilogue metadata must be an object or null.");
        }
        ThorImplementation::ExpressionDefinition epilogueDefinition =
            ThorImplementation::ExpressionDefinition::deserialize(j.at("epilogue"));
        epilogue = epilogueExpressionFromDefinition(epilogueDefinition, epilogueAuxInputNames);
    } else if (!epilogueInputBindings.empty()) {
        throw std::runtime_error("Attention serialized epilogue_inputs require a non-null epilogue expression.");
    }
    Tensor featureOutput = Tensor::deserialize(j.at("feature_output"), archiveReader.get());

    if (j.contains("sequence_lengths_input") || j.contains("ragged_offsets_input") ||
        j.contains("use_separate_sequence_lengths") || j.contains("use_separate_ragged_offsets") ||
        j.contains("use_ragged_offsets") || j.contains("query_ragged_offsets_input") ||
        j.contains("key_value_ragged_offsets_input")) {
        throw std::runtime_error(
            "Attention deserialize does not support the removed raw ragged-offset representation. "
            "Ragged Attention archives must use canonical RaggedTensor metadata.");
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

    std::optional<RaggedTensor> raggedFeatureInput = std::nullopt;
    std::optional<RaggedTensor> raggedContextInput = std::nullopt;
    const bool serializedRagged = j.value("use_ragged", false) || j.contains("ragged_feature_input") ||
                                  j.contains("ragged_context_input") || j.contains("ragged_feature_output");
    if (serializedRagged) {
        if (!j.contains("ragged_feature_input") || !j.contains("ragged_feature_output")) {
            throw std::runtime_error("Attention deserialize ragged mode requires ragged_feature_input and ragged_feature_output.");
        }
        raggedFeatureInput = raggedFromNetworkMetadata(j.at("ragged_feature_input"), "ragged_feature_input");
        if (raggedFeatureInput->getValues() != featureInput) {
            throw std::runtime_error("Attention serialized ragged_feature_input values do not match feature_input.");
        }
        if (contextInput.has_value()) {
            if (!j.contains("ragged_context_input")) {
                throw std::runtime_error("Attention deserialize ragged cross-attention requires ragged_context_input.");
            }
            raggedContextInput = raggedFromNetworkMetadata(j.at("ragged_context_input"), "ragged_context_input");
            if (raggedContextInput->getValues() != contextInput.value()) {
                throw std::runtime_error("Attention serialized ragged_context_input values do not match context_input.");
            }
        } else if (j.contains("ragged_context_input")) {
            throw std::runtime_error("Attention serialized ragged_context_input requires context_input.");
        }
        if (querySequenceLengthsInput.has_value()) {
            throw std::runtime_error("Attention serialized RaggedTensor inputs cannot also carry sequence-length metadata.");
        }
    }

    const std::vector<uint64_t> inputDims = featureInput.getDimensions();
    if (inputDims.size() != 2) {
        throw std::runtime_error("Attention deserialize expected rank-2 feature_input.");
    }
    const std::vector<uint64_t> contextDims = contextInput.has_value() ? contextInput->getDimensions() : inputDims;
    if (contextDims.size() != 2) {
        throw std::runtime_error("Attention deserialize expected rank-2 context_input.");
    }
    const uint64_t querySequenceLength = inputDims.at(0);
    const uint64_t keyValueSequenceLength = contextDims.at(0);
    const uint64_t queryInputFeatures = inputDims.at(1);
    const uint64_t contextInputFeatures = contextDims.at(1);

    const uint32_t numHeads = j.at("num_heads").get<uint32_t>();
    const uint32_t numKeyValueHeads = j.at("num_key_value_heads").get<uint32_t>();
    const uint32_t headDim = j.at("head_dim").get<uint32_t>();
    const uint32_t valueDim = j.at("value_dim").get<uint32_t>();
    const uint32_t outputFeatures = j.at("output_features").get<uint32_t>();
    const bool hasBias = j.at("has_bias").get<bool>();
    const bool useRope = j.at("use_rope").get<bool>();
    const bool ropeInPlace = j.value("rope_in_place", false);
    ThorImplementation::RotaryPositionEmbeddingOptions ropeOptions = ropeOptionsFromJson(j.at("rope_options"));
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
    const DataType weightsDataType = j.at("weights_data_type").get<DataType>();
    const DataType computeDataType = j.at("compute_data_type").get<DataType>();
    const DataType outputDataType = j.at("output_data_type").get<DataType>();
    const uint64_t maximumPossibleRopeSequenceLength =
        raggedFeatureInput.has_value() ? raggedFeatureInput->getMaxTotalValues()
                                       : std::max(querySequenceLength, keyValueSequenceLength);
    if (const std::optional<std::string> error =
            ropeFp32SequenceLengthValidationError(useRope, ropeOptions, maximumPossibleRopeSequenceLength);
        error.has_value()) {
        throw std::runtime_error(error.value());
    }
    if (epilogue.has_value()) {
        auto validateSerializedExpressionInputDTypes = [&](const std::string& inputName) {
            const AttentionEpilogueInputDataTypes inputDataTypes =
                attentionEpilogueInputDataTypes(epilogue.value(), inputName);
            if (inputDataTypes.outputDataType.has_value() && inputDataTypes.outputDataType.value() != outputDataType) {
                throw std::runtime_error("Attention serialized epilogue input '" + inputName +
                                         "' output dtype annotation does not match output_data_type.");
            }
            if (inputDataTypes.computeDataType.has_value() && inputDataTypes.computeDataType.value() != computeDataType) {
                throw std::runtime_error("Attention serialized epilogue input '" + inputName +
                                         "' compute dtype annotation does not match compute_data_type.");
            }
        };
        validateSerializedExpressionInputDTypes(Attention::epilogueInputName());
        for (const std::string& inputName : epilogueAuxInputNames) {
            validateSerializedExpressionInputDTypes(inputName);
        }
    }
    if (featureOutput.getDimensions() != std::vector<uint64_t>{querySequenceLength, outputFeatures}) {
        throw std::runtime_error("Attention serialized feature_output shape does not match query sequence and output features.");
    }
    if (featureOutput.getDataType() != outputDataType) {
        throw std::runtime_error("Attention serialized feature_output dtype does not match output_data_type.");
    }
    for (const auto& [name, tensor] : epilogueInputBindings) {
        if (tensor.getDimensions() != featureOutput.getDimensions()) {
            throw std::runtime_error("Attention serialized epilogue input '" + name +
                                     "' shape does not match the feature output shape.");
        }
        if (tensor.getDataType() != outputDataType) {
            throw std::runtime_error("Attention serialized epilogue input '" + name +
                                     "' dtype does not match output_data_type.");
        }
    }
    if (raggedFeatureInput.has_value() && scoreBiasInput.has_value()) {
        throw std::runtime_error(
            "Attention serialized ragged mode does not support score_bias_input because the current cuDNN ragged SDPA "
            "backward path does not support additive-bias gradients.");
    }
    if (scoreBiasInput.has_value()) {
        requireScoreBiasInput(scoreBiasInput.value(), numHeads, querySequenceLength, keyValueSequenceLength, computeDataType);
        if (maskKind == ThorImplementation::AttentionMaskKind::CausalBottomRight ||
            maskKind == ThorImplementation::AttentionMaskKind::SlidingWindowBottomRight) {
            throw std::runtime_error(
                "Attention deserialize bottom-right/decode masks cannot currently be combined with score_bias_input.");
        }
    }
    if (querySequenceLengthsInput.has_value()) {
        requireSequenceLengthsInput(querySequenceLengthsInput.value(), "querySequenceLengthsInput");
        requireSequenceLengthsInput(keyValueSequenceLengthsInput.value(), "keyValueSequenceLengthsInput");
    }
    if (raggedFeatureInput.has_value()) {
        requireRaggedFeatureInput(raggedFeatureInput.value(), "ragged_feature_input");
        if (raggedContextInput.has_value()) {
            requireRaggedFeatureInput(raggedContextInput.value(), "ragged_context_input");
            if (raggedContextInput->getBatchSize() != raggedFeatureInput->getBatchSize()) {
                throw std::runtime_error("Attention serialized ragged query and key/value inputs must have the same batch size.");
            }
            if (useRope && raggedContextInput->getOffsets() != raggedFeatureInput->getOffsets()) {
                throw std::runtime_error(
                    "Attention serialized ragged cross-attention with RoPE requires query and key/value to share one row partition.");
            }
        }
        const json& raggedOutputJson = j.at("ragged_feature_output");
        if (raggedOutputJson.at("values").at("id").get<uint64_t>() != featureOutput.getOriginalId() ||
            raggedOutputJson.at("offsets").at("id").get<uint64_t>() != raggedFeatureInput->getOffsets().getOriginalId()) {
            throw std::runtime_error("Attention serialized ragged_feature_output must use feature_output values and the query row partition.");
        }
    }

    std::vector<std::shared_ptr<ParameterSpecification>> parameters;
    if (j.contains("parameters")) {
        const json& parametersJson = j.at("parameters");
        if (!parametersJson.is_object()) {
            throw std::runtime_error("Attention parameters must be an object keyed by parameter name.");
        }
        for (auto it = parametersJson.begin(); it != parametersJson.end(); ++it) {
            ParameterSpecification parameter = ParameterSpecification::deserialize(it.value(), archiveReader);
            parameters.push_back(std::make_shared<ParameterSpecification>(std::move(parameter)));
        }
    }

    std::vector<std::string> requiredParameterNames;
    if (!raggedFeatureInput.has_value() && usePackedQkvProjectionForLayer(useRope, contextInput.has_value())) {
        requiredParameterNames = {"qkv_weights", "output_weights"};
        if (hasBias) {
            requiredParameterNames.push_back("qkv_bias");
            requiredParameterNames.push_back("output_bias");
        }
    } else {
        requiredParameterNames = {"query_weights", "key_weights", "value_weights", "output_weights"};
        if (hasBias) {
            requiredParameterNames.push_back("query_bias");
            requiredParameterNames.push_back("key_bias");
            requiredParameterNames.push_back("value_bias");
            requiredParameterNames.push_back("output_bias");
        }
    }
    for (const std::string& requiredName : requiredParameterNames) {
        bool found = false;
        for (const auto& parameter : parameters) {
            if (parameter != nullptr && parameter->getName() == requiredName) {
                found = true;
                break;
            }
        }
        if (!found) {
            throw std::runtime_error("Attention deserialize did not find required parameter '" + requiredName + "'.");
        }
    }

    const bool useScoreBias = scoreBiasInput.has_value();
    const bool useSequenceLengths = querySequenceLengthsInput.has_value();
    const bool useRagged = raggedFeatureInput.has_value();

    Attention layer(makeAttentionExpression(querySequenceLength,
                                            keyValueSequenceLength,
                                            queryInputFeatures,
                                            contextInputFeatures,
                                            outputFeatures,
                                            numHeads,
                                            numKeyValueHeads,
                                            headDim,
                                            valueDim,
                                            hasBias,
                                            useRope,
                                            ropeInPlace,
                                            ropeOptions,
                                            maskKind,
                                            diagonalLeftBound,
                                            diagonalRightBound,
                                            useAlibiMask,
                                            attentionScale,
                                            dropoutProbability,
                                            dropoutSeed,
                                            dropoutOffset,
                                            contextInput.has_value(),
                                            useScoreBias,
                                            useSequenceLengths,
                                            useRagged,
                                            useRagged ? raggedFeatureInput->getBatchSize() : 0,
                                            useRagged ? raggedFeatureInput->getOffsetsDataType()
                                                      : ThorImplementation::kDefaultRowPartitionOffsetDataType,
                                            useRagged
                                                ? (raggedContextInput.has_value() ? raggedContextInput->getOffsetsDataType()
                                                                                  : raggedFeatureInput->getOffsetsDataType())
                                                : ThorImplementation::kDefaultRowPartitionOffsetDataType,
                                            featureInput.getDataType(),
                                            weightsDataType,
                                            computeDataType,
                                            outputDataType,
                                            epilogue,
                                            epilogueAuxInputNames),
                    publicAttentionInputNames(
                        contextInput.has_value(), useScoreBias, useSequenceLengths, useRagged, epilogueAuxInputNames),
                    {publicAttentionInputInterface(featureInput,
                                                   contextInput,
                                                   scoreBiasInput,
                                                   querySequenceLengthsInput,
                                                   keyValueSequenceLengthsInput,
                                                   raggedFeatureInput,
                                                   raggedContextInput,
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
                    maskKind,
                    diagonalLeftBound,
                    diagonalRightBound,
                    useAlibiMask,
                    attentionScale,
                    dropoutProbability,
                    dropoutSeed,
                    dropoutOffset,
                    contextInput,
                    scoreBiasInput,
                    querySequenceLengthsInput,
                    keyValueSequenceLengthsInput,
                    raggedFeatureInput,
                    raggedContextInput,
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
