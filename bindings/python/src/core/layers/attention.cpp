#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cmath>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include "DeepLearning/Api/Initializers/Initializer.h"
#include "DeepLearning/Api/Layers/Learning/Attention.h"
#include "DeepLearning/Api/Layers/Learning/CustomLayer.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"
#include "DeepLearning/Api/Tensor/Tensor.h"
#include "Utilities/Expression/DynamicExpression.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace Thor;

using DataType = ThorImplementation::TensorDescriptor::DataType;
using AttentionMaskKind = ThorImplementation::AttentionMaskKind;
using RotaryScalingKind = ThorImplementation::RotaryScalingKind;

namespace {

AttentionMaskKind parseAttentionMaskKind(const std::string& value) {
    if (value == "none")
        return AttentionMaskKind::None;
    if (value == "causal_top_left")
        return AttentionMaskKind::CausalTopLeft;
    if (value == "causal_bottom_right")
        return AttentionMaskKind::CausalBottomRight;
    if (value == "sliding_window_top_left")
        return AttentionMaskKind::SlidingWindowTopLeft;
    if (value == "sliding_window_bottom_right")
        return AttentionMaskKind::SlidingWindowBottomRight;
    throw nb::value_error(
        "Attention mask_kind must be one of: none, causal_top_left, causal_bottom_right, "
        "sliding_window_top_left, sliding_window_bottom_right.");
}

std::string attentionMaskKindName(AttentionMaskKind value) {
    switch (value) {
        case AttentionMaskKind::None:
            return "none";
        case AttentionMaskKind::CausalTopLeft:
            return "causal_top_left";
        case AttentionMaskKind::CausalBottomRight:
            return "causal_bottom_right";
        case AttentionMaskKind::SlidingWindowTopLeft:
            return "sliding_window_top_left";
        case AttentionMaskKind::SlidingWindowBottomRight:
            return "sliding_window_bottom_right";
    }
    return "unknown";
}

RotaryScalingKind parseRotaryScalingKind(const std::string& value) {
    if (value == "none")
        return RotaryScalingKind::None;
    if (value == "linear")
        return RotaryScalingKind::Linear;
    if (value == "dynamic_ntk" || value == "dynamic")
        return RotaryScalingKind::DynamicNTK;
    if (value == "yarn")
        return RotaryScalingKind::Yarn;
    if (value == "longrope" || value == "long_rope")
        return RotaryScalingKind::LongRope;
    if (value == "llama3")
        return RotaryScalingKind::Llama3;
    throw nb::value_error("Attention rope_scaling_kind must be one of: none, linear, dynamic_ntk, yarn, longrope, llama3.");
}

std::string rotaryScalingKindName(RotaryScalingKind value) {
    switch (value) {
        case RotaryScalingKind::None:
            return "none";
        case RotaryScalingKind::Linear:
            return "linear";
        case RotaryScalingKind::DynamicNTK:
            return "dynamic_ntk";
        case RotaryScalingKind::Yarn:
            return "yarn";
        case RotaryScalingKind::LongRope:
            return "longrope";
        case RotaryScalingKind::Llama3:
            return "llama3";
    }
    return "unknown";
}

bool attentionUsesPackedQkvProjection(const Attention& self) {
    if constexpr (!Attention::USE_PACKED_QKV_PROJECTION) {
        return false;
    } else {
        return !self.getUseRope() && !self.getUseCrossAttention();
    }
}

}  // namespace

void bind_attention(nb::module_& layers) {
    auto attention = nb::class_<Attention, CustomLayer>(layers, "Attention");
    attention.attr("__module__") = "thor.layers";

    attention.def(
        "__init__",
        [](Attention* self,
           Network& network,
           Tensor feature_input,
           uint32_t num_heads,
           std::optional<uint32_t> num_key_value_heads,
           std::optional<uint32_t> head_dim,
           std::optional<uint32_t> value_dim,
           std::optional<uint32_t> output_features,
           bool has_bias,
           std::string mask_kind,
           int64_t diagonal_left_bound,
           int64_t diagonal_right_bound,
           bool use_alibi_mask,
           std::optional<double> attention_scale,
           bool use_rope,
           uint64_t rope_rotary_dim,
           double rope_base,
           int64_t rope_position_offset,
           bool rope_interleaved,
           std::string rope_scaling_kind,
           double rope_scaling_factor,
           uint64_t rope_original_max_position_embeddings,
           std::optional<double> rope_attention_factor,
           double rope_yarn_beta_fast,
           double rope_yarn_beta_slow,
           double rope_llama3_low_freq_factor,
           double rope_llama3_high_freq_factor,
           std::vector<double> rope_long_rope_short_factors,
           std::vector<double> rope_long_rope_long_factors,
           std::optional<DataType> weights_data_type,
           DataType compute_data_type,
           std::optional<DataType> output_data_type,
           std::shared_ptr<Initializer> weights_initializer,
           std::shared_ptr<Initializer> bias_initializer,
           std::shared_ptr<Optimizer> optimizer,
           bool rope_in_place,
           float dropout_probability,
           int64_t dropout_seed,
           int64_t dropout_offset,
           std::optional<Tensor> sequence_lengths,
           std::optional<Tensor> query_sequence_lengths,
           std::optional<Tensor> key_value_sequence_lengths,
           std::optional<Tensor> ragged_offsets,
           std::optional<Tensor> query_ragged_offsets,
           std::optional<Tensor> key_value_ragged_offsets,
           std::optional<Tensor> context_input) {
            if (num_heads == 0) {
                throw nb::value_error("Attention instance: num_heads must be > 0.");
            }
            if (num_key_value_heads.has_value() && num_key_value_heads.value() == 0) {
                throw nb::value_error("Attention instance: num_key_value_heads must be > 0.");
            }
            if (head_dim.has_value() && head_dim.value() == 0) {
                throw nb::value_error("Attention instance: head_dim must be > 0.");
            }
            if (value_dim.has_value() && value_dim.value() == 0) {
                throw nb::value_error("Attention instance: value_dim must be > 0.");
            }
            if (output_features.has_value() && output_features.value() == 0) {
                throw nb::value_error("Attention instance: output_features must be > 0.");
            }
            if (!std::isfinite(dropout_probability) || dropout_probability < 0.0f || dropout_probability >= 1.0f) {
                throw nb::value_error("Attention instance: dropout_probability must be finite and in [0, 1).");
            }
            if (sequence_lengths.has_value() && (query_sequence_lengths.has_value() || key_value_sequence_lengths.has_value())) {
                throw nb::value_error(
                    "Attention instance: use either sequence_lengths or query_sequence_lengths/key_value_sequence_lengths, not both.");
            }
            if (query_sequence_lengths.has_value() != key_value_sequence_lengths.has_value()) {
                throw nb::value_error(
                    "Attention instance: query_sequence_lengths and key_value_sequence_lengths must be provided together.");
            }
            if (ragged_offsets.has_value() && (query_ragged_offsets.has_value() || key_value_ragged_offsets.has_value())) {
                throw nb::value_error(
                    "Attention instance: use either ragged_offsets or query_ragged_offsets/key_value_ragged_offsets, not both.");
            }
            if (query_ragged_offsets.has_value() != key_value_ragged_offsets.has_value()) {
                throw nb::value_error(
                    "Attention instance: query_ragged_offsets and key_value_ragged_offsets must be provided together.");
            }

            Attention::Builder builder;
            builder.network(network)
                .featureInput(feature_input)
                .numHeads(num_heads)
                .hasBias(has_bias)
                .maskKind(parseAttentionMaskKind(mask_kind));

            if (context_input.has_value()) {
                builder.contextInput(context_input.value());
            }

            if (sequence_lengths.has_value()) {
                builder.sequenceLengthsInput(sequence_lengths.value());
            }
            if (query_sequence_lengths.has_value()) {
                builder.querySequenceLengthsInput(query_sequence_lengths.value());
                builder.keyValueSequenceLengthsInput(key_value_sequence_lengths.value());
            }
            if (ragged_offsets.has_value()) {
                builder.raggedOffsetsInput(ragged_offsets.value());
            }
            if (query_ragged_offsets.has_value()) {
                builder.queryRaggedOffsetsInput(query_ragged_offsets.value());
                builder.keyValueRaggedOffsetsInput(key_value_ragged_offsets.value());
            }

            if (num_key_value_heads.has_value()) {
                builder.numKeyValueHeads(num_key_value_heads.value());
            }
            if (head_dim.has_value()) {
                builder.headDim(head_dim.value());
            }
            if (value_dim.has_value()) {
                builder.valueDim(value_dim.value());
            }
            if (output_features.has_value()) {
                builder.outputFeatures(output_features.value());
            }
            if (diagonal_left_bound != 0) {
                builder.diagonalLeftBound(diagonal_left_bound);
            }
            if (diagonal_right_bound != 0) {
                builder.diagonalRightBound(diagonal_right_bound);
            }
            if (use_alibi_mask) {
                builder.useAlibiMask(true);
            }
            if (attention_scale.has_value()) {
                builder.attentionScale(attention_scale.value());
            }
            if (dropout_probability != 0.0f) {
                builder.dropoutProbability(dropout_probability).dropoutSeed(dropout_seed).dropoutOffset(dropout_offset);
            }
            if (use_rope) {
                ThorImplementation::RotaryPositionEmbeddingOptions rope_options;
                rope_options.rotary_dim = rope_rotary_dim;
                rope_options.base = rope_base;
                rope_options.position_offset = rope_position_offset;
                rope_options.interleaved = rope_interleaved;
                rope_options.scaling_kind = parseRotaryScalingKind(rope_scaling_kind);
                rope_options.scaling_factor = rope_scaling_factor;
                rope_options.original_max_position_embeddings = rope_original_max_position_embeddings;
                rope_options.attention_factor = rope_attention_factor;
                rope_options.yarn_beta_fast = rope_yarn_beta_fast;
                rope_options.yarn_beta_slow = rope_yarn_beta_slow;
                rope_options.llama3_low_freq_factor = rope_llama3_low_freq_factor;
                rope_options.llama3_high_freq_factor = rope_llama3_high_freq_factor;
                rope_options.long_rope_short_factors = std::move(rope_long_rope_short_factors);
                rope_options.long_rope_long_factors = std::move(rope_long_rope_long_factors);
                builder.ropeOptions(std::move(rope_options));
            }
            if (rope_in_place) {
                builder.ropeInPlace(true);
            }
            if (weights_data_type.has_value()) {
                builder.weightsDataType(weights_data_type.value());
            }
            builder.computeDataType(compute_data_type);
            if (output_data_type.has_value()) {
                builder.outputDataType(output_data_type.value());
            }
            if (weights_initializer != nullptr) {
                builder.weightsInitializer(std::move(weights_initializer));
            }
            if (bias_initializer != nullptr) {
                builder.biasInitializer(std::move(bias_initializer));
            }
            if (optimizer != nullptr) {
                builder.optimizer(std::move(optimizer));
            }

            new (self) Attention(std::move(builder.build()));
        },
        "network"_a,
        "feature_input"_a,
        "num_heads"_a,
        "num_key_value_heads"_a.none() = nb::none(),
        "head_dim"_a.none() = nb::none(),
        "value_dim"_a.none() = nb::none(),
        "output_features"_a.none() = nb::none(),
        "has_bias"_a = false,
        "mask_kind"_a = "none",
        "diagonal_left_bound"_a = 0,
        "diagonal_right_bound"_a = 0,
        "use_alibi_mask"_a = false,
        "attention_scale"_a.none() = nb::none(),
        "use_rope"_a = false,
        "rope_rotary_dim"_a = 0,
        "rope_base"_a = 10000.0,
        "rope_position_offset"_a = 0,
        "rope_interleaved"_a = false,
        "rope_scaling_kind"_a = "none",
        "rope_scaling_factor"_a = 1.0,
        "rope_original_max_position_embeddings"_a = 0,
        "rope_attention_factor"_a.none() = nb::none(),
        "rope_yarn_beta_fast"_a = 32.0,
        "rope_yarn_beta_slow"_a = 1.0,
        "rope_llama3_low_freq_factor"_a = 1.0,
        "rope_llama3_high_freq_factor"_a = 4.0,
        "rope_long_rope_short_factors"_a = std::vector<double>{},
        "rope_long_rope_long_factors"_a = std::vector<double>{},
        "weights_data_type"_a.none() = nb::none(),
        "compute_data_type"_a = DataType::FP32,
        "output_data_type"_a.none() = nb::none(),
        "weights_initializer"_a.none() = nb::none(),
        "bias_initializer"_a.none() = nb::none(),
        "optimizer"_a.none() = nb::none(),
        "rope_in_place"_a = false,
        "dropout_probability"_a = 0.0f,
        "dropout_seed"_a = int64_t{0},
        "dropout_offset"_a = int64_t{0},
        "sequence_lengths"_a.none() = nb::none(),
        "query_sequence_lengths"_a.none() = nb::none(),
        "key_value_sequence_lengths"_a.none() = nb::none(),
        "ragged_offsets"_a.none() = nb::none(),
        "query_ragged_offsets"_a.none() = nb::none(),
        "key_value_ragged_offsets"_a.none() = nb::none(),
        "context_input"_a.none() = nb::none(),
        R"nbdoc(
Public transformer attention layer.

The logical feature input shape is ``[sequence, input_features]``.  Placement adds the batch dimension, so the physical
hot path consumes ``[batch, sequence, input_features]``.
)nbdoc");

    attention.def("get_feature_output", [](Attention& self) -> Tensor { return self.getOutput("feature_output"); });
    attention.def("get_num_heads", &Attention::getNumHeads);
    attention.def("get_num_key_value_heads", &Attention::getNumKeyValueHeads);
    attention.def("get_head_dim", &Attention::getHeadDim);
    attention.def("get_value_dim", &Attention::getValueDim);
    attention.def("get_output_features", &Attention::getOutputFeatures);
    attention.def("get_has_bias", &Attention::getHasBias);
    attention.def("get_use_rope", &Attention::getUseRope);
    attention.def("get_rope_in_place", &Attention::getRopeInPlace);
    attention.def("get_rope_scaling_kind", [](Attention& self) { return rotaryScalingKindName(self.getRopeOptions().scaling_kind); });
    attention.def("get_rope_scaling_factor", [](Attention& self) { return self.getRopeOptions().scaling_factor; });
    attention.def("get_rope_original_max_position_embeddings",
                  [](Attention& self) { return self.getRopeOptions().original_max_position_embeddings; });
    attention.def("get_mask_kind", [](Attention& self) { return attentionMaskKindName(self.getMaskKind()); });
    attention.def("get_diagonal_left_bound", &Attention::getDiagonalLeftBound);
    attention.def("get_diagonal_right_bound", &Attention::getDiagonalRightBound);
    attention.def("get_use_alibi_mask", &Attention::getUseAlibiMask);
    attention.def("get_attention_scale", &Attention::getAttentionScale);
    attention.def("get_dropout_probability", &Attention::getDropoutProbability);
    attention.def("get_dropout_seed", &Attention::getDropoutSeed);
    attention.def("get_dropout_offset", &Attention::getDropoutOffset);
    attention.def("get_use_cross_attention", &Attention::getUseCrossAttention);
    attention.def("get_context_input", [](Attention& self) { return self.getContextInput(); });
    attention.def("get_use_sequence_lengths", &Attention::getUseSequenceLengths);
    attention.def("get_use_ragged_offsets", &Attention::getUseRaggedOffsets);
    attention.def("get_sequence_lengths_input", [](Attention& self) { return self.getSequenceLengthsInput(); });
    attention.def("get_query_sequence_lengths_input", [](Attention& self) { return self.getQuerySequenceLengthsInput(); });
    attention.def("get_key_value_sequence_lengths_input", [](Attention& self) { return self.getKeyValueSequenceLengthsInput(); });
    attention.def("get_ragged_offsets_input", [](Attention& self) { return self.getRaggedOffsetsInput(); });
    attention.def("get_query_ragged_offsets_input", [](Attention& self) { return self.getQueryRaggedOffsetsInput(); });
    attention.def("get_key_value_ragged_offsets_input", [](Attention& self) { return self.getKeyValueRaggedOffsetsInput(); });
    attention.def("get_weights_data_type", &Attention::getWeightsDataType);
    attention.def("get_compute_data_type", &Attention::getComputeDataType);
    attention.def("get_output_data_type", &Attention::getOutputDataType);

    attention.def("_debug_uses_packed_qkv_projection", &attentionUsesPackedQkvProjection);
    attention.def("_debug_qkv_projection_mode",
                  [](Attention& self) { return attentionUsesPackedQkvProjection(self) ? "packed" : "split"; });
    attention.def("_debug_expression", [](Attention& self) { return self.getExpression(); });
}
