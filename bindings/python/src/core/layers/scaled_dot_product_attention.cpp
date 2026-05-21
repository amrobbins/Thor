#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>

#include "DeepLearning/Api/Layers/Learning/CustomLayer.h"
#include "DeepLearning/Api/Layers/Learning/ScaledDotProductAttention.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"
#include "Utilities/TensorOperations/GpuAttention/CudnnAttention.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace Thor;

using DataType = ThorImplementation::TensorDescriptor::DataType;
using AttentionMaskKind = ThorImplementation::AttentionMaskKind;
using AttentionTensorLayout = ThorImplementation::AttentionTensorLayout;

namespace {

AttentionTensorLayout parseAttentionTensorLayout(const std::string& value) {
    if (value == "bhsd")
        return AttentionTensorLayout::BHSD;
    if (value == "bshd")
        return AttentionTensorLayout::BSHD;
    throw nb::value_error("ScaledDotProductAttention tensor_layout must be one of: bhsd, bshd.");
}

std::string attentionTensorLayoutName(AttentionTensorLayout value) {
    switch (value) {
        case AttentionTensorLayout::BHSD:
            return "bhsd";
        case AttentionTensorLayout::BSHD:
            return "bshd";
        default:
            return "unknown";
    }
}

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
        "ScaledDotProductAttention mask_kind must be one of: none, causal_top_left, causal_bottom_right, "
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

void requireNoMixedConvenienceAndExplicit(const std::optional<Tensor>& shared,
                                          const std::optional<Tensor>& q,
                                          const std::optional<Tensor>& kv,
                                          const char* what) {
    if (shared.has_value() && (q.has_value() || kv.has_value())) {
        std::string error = std::string("ScaledDotProductAttention instance: use either ") + what + " or explicit query/key_value " + what +
                            " inputs, not both.";
        throw nb::value_error(error.c_str());
    }
}

}  // namespace

void bind_scaled_dot_product_attention(nb::module_& layers) {
    auto sdpa = nb::class_<ScaledDotProductAttention, CustomLayer>(layers, "ScaledDotProductAttention");
    sdpa.attr("__module__") = "thor.layers";

    sdpa.def(
        "__init__",
        [](ScaledDotProductAttention* self,
           Network& network,
           Tensor query_input,
           std::optional<Tensor> key_input,
           std::optional<Tensor> value_input,
           std::optional<Tensor> bias_input,
           std::string tensor_layout,
           std::string mask_kind,
           int64_t diagonal_left_bound,
           int64_t diagonal_right_bound,
           bool use_alibi_mask,
           std::optional<double> attention_scale,
           DataType compute_data_type,
           std::optional<DataType> output_data_type,
           std::optional<Tensor> sequence_lengths,
           std::optional<Tensor> query_sequence_lengths,
           std::optional<Tensor> key_value_sequence_lengths,
           std::optional<Tensor> ragged_offsets,
           std::optional<Tensor> query_ragged_offsets,
           std::optional<Tensor> key_value_ragged_offsets,
           float dropout_probability,
           int64_t dropout_seed,
           int64_t dropout_offset,
           std::optional<Tensor> fp8_descale_q,
           std::optional<Tensor> fp8_descale_k,
           std::optional<Tensor> fp8_descale_v,
           std::optional<Tensor> fp8_descale_s,
           std::optional<Tensor> fp8_scale_s,
           std::optional<Tensor> fp8_scale_o,
           std::optional<Tensor> fp8_amax_s,
           std::optional<Tensor> fp8_amax_o) {
            requireNoMixedConvenienceAndExplicit(sequence_lengths, query_sequence_lengths, key_value_sequence_lengths, "sequence_lengths");
            requireNoMixedConvenienceAndExplicit(ragged_offsets, query_ragged_offsets, key_value_ragged_offsets, "ragged_offsets");
            if (attention_scale.has_value() && (!std::isfinite(attention_scale.value()) || attention_scale.value() <= 0.0)) {
                throw nb::value_error("ScaledDotProductAttention instance: attention_scale must be finite and > 0.");
            }
            if (!std::isfinite(dropout_probability) || dropout_probability < 0.0f || dropout_probability >= 1.0f) {
                throw nb::value_error("ScaledDotProductAttention instance: dropout_probability must be finite and in [0, 1).");
            }
            if (dropout_probability > 0.0f && dropout_offset < 0) {
                throw nb::value_error("ScaledDotProductAttention instance: dropout_offset must be non-negative when dropout is enabled.");
            }

            ScaledDotProductAttention::Builder builder;
            builder.network(network).queryInput(query_input);
            if (key_input.has_value() || value_input.has_value()) {
                if (!key_input.has_value() || !value_input.has_value()) {
                    throw nb::value_error("ScaledDotProductAttention instance: key_input and value_input must be provided together.");
                }
                builder.keyInput(key_input.value()).valueInput(value_input.value());
            } else {
                builder.keyInput(query_input).valueInput(query_input);
            }
            if (bias_input.has_value()) {
                builder.biasInput(bias_input.value());
            }
            builder.tensorLayout(parseAttentionTensorLayout(tensor_layout)).maskKind(parseAttentionMaskKind(mask_kind));
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
                builder.dropout(dropout_probability, dropout_seed, dropout_offset);
            }
            builder.computeDataType(compute_data_type);
            if (output_data_type.has_value()) {
                builder.outputDataType(output_data_type.value());
            }
            if (sequence_lengths.has_value()) {
                builder.sequenceLengthsInput(sequence_lengths.value());
            } else {
                if (query_sequence_lengths.has_value()) {
                    builder.querySequenceLengthsInput(query_sequence_lengths.value());
                }
                if (key_value_sequence_lengths.has_value()) {
                    builder.keyValueSequenceLengthsInput(key_value_sequence_lengths.value());
                }
            }
            if (ragged_offsets.has_value()) {
                builder.raggedOffsetsInput(ragged_offsets.value());
            } else {
                if (query_ragged_offsets.has_value()) {
                    builder.queryRaggedOffsetsInput(query_ragged_offsets.value());
                }
                if (key_value_ragged_offsets.has_value()) {
                    builder.keyValueRaggedOffsetsInput(key_value_ragged_offsets.value());
                }
            }
            const bool any_fp8_scale = fp8_descale_q.has_value() || fp8_descale_k.has_value() || fp8_descale_v.has_value() ||
                                       fp8_descale_s.has_value() || fp8_scale_s.has_value() || fp8_scale_o.has_value() ||
                                       fp8_amax_s.has_value() || fp8_amax_o.has_value();
            const bool all_fp8_scale = fp8_descale_q.has_value() && fp8_descale_k.has_value() && fp8_descale_v.has_value() &&
                                       fp8_descale_s.has_value() && fp8_scale_s.has_value() && fp8_scale_o.has_value() &&
                                       fp8_amax_s.has_value() && fp8_amax_o.has_value();
            if (any_fp8_scale != all_fp8_scale) {
                throw nb::value_error("ScaledDotProductAttention instance: FP8 forward requires all descale/scale/amax tensors.");
            }
            if (all_fp8_scale) {
                builder.fp8ForwardScalingInputs(fp8_descale_q.value(),
                                                fp8_descale_k.value(),
                                                fp8_descale_v.value(),
                                                fp8_descale_s.value(),
                                                fp8_scale_s.value(),
                                                fp8_scale_o.value(),
                                                fp8_amax_s.value(),
                                                fp8_amax_o.value());
            }

            new (self) ScaledDotProductAttention(std::move(builder.build()));
        },
        "network"_a,
        "query_input"_a,
        "key_input"_a.none() = nb::none(),
        "value_input"_a.none() = nb::none(),
        "bias_input"_a.none() = nb::none(),
        "tensor_layout"_a = "bhsd",
        "mask_kind"_a = "none",
        "diagonal_left_bound"_a = 0,
        "diagonal_right_bound"_a = 0,
        "use_alibi_mask"_a = false,
        "attention_scale"_a.none() = nb::none(),
        "compute_data_type"_a = DataType::FP32,
        "output_data_type"_a.none() = nb::none(),
        "sequence_lengths"_a.none() = nb::none(),
        "query_sequence_lengths"_a.none() = nb::none(),
        "key_value_sequence_lengths"_a.none() = nb::none(),
        "ragged_offsets"_a.none() = nb::none(),
        "query_ragged_offsets"_a.none() = nb::none(),
        "key_value_ragged_offsets"_a.none() = nb::none(),
        "dropout_probability"_a = 0.0f,
        "dropout_seed"_a = int64_t{0},
        "dropout_offset"_a = int64_t{0},
        "fp8_descale_q"_a.none() = nb::none(),
        "fp8_descale_k"_a.none() = nb::none(),
        "fp8_descale_v"_a.none() = nb::none(),
        "fp8_descale_s"_a.none() = nb::none(),
        "fp8_scale_s"_a.none() = nb::none(),
        "fp8_scale_o"_a.none() = nb::none(),
        "fp8_amax_s"_a.none() = nb::none(),
        "fp8_amax_o"_a.none() = nb::none());

    sdpa.def(
        "get_feature_output",
        [](ScaledDotProductAttention& self) -> Tensor { return self.getOutput("output"); },
        R"nbdoc(Return the output tensor produced by this layer.)nbdoc");
    sdpa.def("get_tensor_layout", [](ScaledDotProductAttention& self) { return attentionTensorLayoutName(self.getTensorLayout()); });
    sdpa.def("get_mask_kind", [](ScaledDotProductAttention& self) { return attentionMaskKindName(self.getMaskKind()); });
    sdpa.def("get_diagonal_left_bound", &ScaledDotProductAttention::getDiagonalLeftBound);
    sdpa.def("get_diagonal_right_bound", &ScaledDotProductAttention::getDiagonalRightBound);
    sdpa.def("get_use_alibi_mask", &ScaledDotProductAttention::getUseAlibiMask);
    sdpa.def("get_attention_scale", &ScaledDotProductAttention::getAttentionScale);
    sdpa.def("get_dropout_probability", &ScaledDotProductAttention::getDropoutProbability);
    sdpa.def("get_dropout_seed", &ScaledDotProductAttention::getDropoutSeed);
    sdpa.def("get_dropout_offset", &ScaledDotProductAttention::getDropoutOffset);
    sdpa.def("get_use_sequence_lengths", &ScaledDotProductAttention::getUseSequenceLengths);
    sdpa.def("get_use_ragged_offsets", &ScaledDotProductAttention::getUseRaggedOffsets);
    sdpa.def("get_use_bias", &ScaledDotProductAttention::getUseBias);
    sdpa.def("get_bias_input", &ScaledDotProductAttention::getBiasInput);
    sdpa.def("get_query_sequence_lengths_input", &ScaledDotProductAttention::getQuerySequenceLengthsInput);
    sdpa.def("get_key_value_sequence_lengths_input", &ScaledDotProductAttention::getKeyValueSequenceLengthsInput);
    sdpa.def("get_query_ragged_offsets_input", &ScaledDotProductAttention::getQueryRaggedOffsetsInput);
    sdpa.def("get_key_value_ragged_offsets_input", &ScaledDotProductAttention::getKeyValueRaggedOffsetsInput);
    sdpa.def("get_use_fp8_forward_scaling", &ScaledDotProductAttention::getUseFp8ForwardScaling);
    sdpa.def("get_fp8_descale_q_input", &ScaledDotProductAttention::getFp8DescaleQInput);
    sdpa.def("get_fp8_descale_k_input", &ScaledDotProductAttention::getFp8DescaleKInput);
    sdpa.def("get_fp8_descale_v_input", &ScaledDotProductAttention::getFp8DescaleVInput);
    sdpa.def("get_fp8_descale_s_input", &ScaledDotProductAttention::getFp8DescaleSInput);
    sdpa.def("get_fp8_scale_s_input", &ScaledDotProductAttention::getFp8ScaleSInput);
    sdpa.def("get_fp8_scale_o_input", &ScaledDotProductAttention::getFp8ScaleOInput);
    sdpa.def("get_fp8_amax_s_input", &ScaledDotProductAttention::getFp8AmaxSInput);
    sdpa.def("get_fp8_amax_o_input", &ScaledDotProductAttention::getFp8AmaxOInput);
    sdpa.def("get_compute_data_type", &ScaledDotProductAttention::getComputeDataType);
    sdpa.def("get_output_data_type", &ScaledDotProductAttention::getOutputDataType);

    sdpa.attr("__doc__") = R"nbdoc(
cuDNN-backed scaled dot-product attention layer for already-projected Q/K/V tensors.

API tensor shapes omit batch.  ``tensor_layout='bhsd'`` means
``[heads, sequence, head_dim]`` and ``tensor_layout='bshd'`` means
``[sequence, heads, head_dim]``.  Placement adds the batch dimension and the
cuDNN stage consumes semantic ``[B, H, S, D]`` tensors.

FP16/BF16 production support:

* Q/K/V/O must all use the same FP16 or BF16 dtype.  ``compute_data_type`` should
  be FP32 and ``output_data_type`` should normally match Q/K/V.
* Forward and backward are supported for self-attention, cross-attention, MHA,
  GQA, and MQA.  Query heads must be an integer multiple of key/value heads.
* Masks: ``none``, ``causal_top_left``, ``causal_bottom_right``,
  ``sliding_window_top_left``, and ``sliding_window_bottom_right``.
* ALiBi requires a causal/sliding diagonal mask and ``diagonal_right_bound == 0``.
  Positive right bounds with ALiBi are rejected because cuDNN rejects that graph.
* ``bias_input`` is score-space additive bias with API shape
  ``[1|Hq, 1|Sq, 1|Skv]`` and dtype equal to ``compute_data_type``.  Forward
  supports sequence broadcast.  Backward materializes sequence-broadcast bias to
  dense score space before cuDNN backward and reduces dBias back to the public
  bias shape.  Ragged + additive-bias backward is rejected.
* ``sequence_lengths`` and ``ragged_offsets`` are self-attention conveniences.
  Cross-attention can use ``query_sequence_lengths``/``key_value_sequence_lengths``
  and ``query_ragged_offsets``/``key_value_ragged_offsets``.  Sequence lengths are
  int32 logical ``[1]`` tensors.  Ragged offsets are int32 logical ``[2]`` tensors
  that NetworkInput batches into cuDNN ``[B + 1]`` offset vectors.
* ``dropout_probability``/``dropout_seed``/``dropout_offset`` expose cuDNN Philox
  attention dropout.  Thor advances the runtime dropout offset by
  ``B * Hq * Sq * Skv``.

Experimental FP8 forward-only support:

* Enable by passing all eight scalar FP32 tensors:
  ``fp8_descale_q``, ``fp8_descale_k``, ``fp8_descale_v``, ``fp8_descale_s``,
  ``fp8_scale_s``, ``fp8_scale_o``, ``fp8_amax_s``, and ``fp8_amax_o``.
* Q/K/V/O must all be the same FP8 format, either E4M3 or E5M2.  QK and V head
  dimensions must be multiples of 16 and no larger than 128 on the validated
  production surface.
* FP8 backward is not supported.  FP8 additive bias, dropout, ALiBi, ragged,
  paged KV, bottom-right/decode masks, sliding-window masks, and decode-style
  ``Sq=1, Skv>1`` are rejected on the validated public surface.
* FP8 padding masks / sequence lengths are supported for forward; ragged offsets
  remain disabled for FP8.

Important combination rules:

* Bottom-right/decode masks currently require additive bias, ALiBi, and dropout
  to be disabled in the production cuDNN primary SDPA path.
* Paged KV cache is not exposed by this layer.  It is available only through the
  low-level physical expression API as an inference-only FP16/BF16 path with
  padding-mask sequence lengths, no bias, and no dropout.
* Experimental cuDNN support-surface probe environment variables can bypass some
  guards for measurement, but those combinations are not user-facing support
  guarantees.
)nbdoc";
}
