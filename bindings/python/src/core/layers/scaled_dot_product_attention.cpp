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
           std::optional<Tensor> key_value_ragged_offsets) {
            requireNoMixedConvenienceAndExplicit(sequence_lengths, query_sequence_lengths, key_value_sequence_lengths, "sequence_lengths");
            requireNoMixedConvenienceAndExplicit(ragged_offsets, query_ragged_offsets, key_value_ragged_offsets, "ragged_offsets");
            if (attention_scale.has_value() && (!std::isfinite(attention_scale.value()) || attention_scale.value() <= 0.0)) {
                throw nb::value_error("ScaledDotProductAttention instance: attention_scale must be finite and > 0.");
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
        "key_value_ragged_offsets"_a.none() = nb::none());

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
    sdpa.def("get_use_sequence_lengths", &ScaledDotProductAttention::getUseSequenceLengths);
    sdpa.def("get_use_ragged_offsets", &ScaledDotProductAttention::getUseRaggedOffsets);
    sdpa.def("get_query_sequence_lengths_input", &ScaledDotProductAttention::getQuerySequenceLengthsInput);
    sdpa.def("get_key_value_sequence_lengths_input", &ScaledDotProductAttention::getKeyValueSequenceLengthsInput);
    sdpa.def("get_query_ragged_offsets_input", &ScaledDotProductAttention::getQueryRaggedOffsetsInput);
    sdpa.def("get_key_value_ragged_offsets_input", &ScaledDotProductAttention::getKeyValueRaggedOffsetsInput);
    sdpa.def("get_compute_data_type", &ScaledDotProductAttention::getComputeDataType);
    sdpa.def("get_output_data_type", &ScaledDotProductAttention::getOutputDataType);

    sdpa.attr("__doc__") = R"nbdoc(
        cuDNN-backed scaled dot-product attention layer for already-projected Q/K/V tensors.

        API tensor shapes omit batch.  tensor_layout='bhsd' means [heads, sequence, head_dim];
        tensor_layout='bshd' means [sequence, heads, head_dim].  Additive bias, when provided,
        is dense score-space [query_heads, query_sequence, key_sequence] and fp32 by default.

        sequence_lengths and ragged_offsets are self-attention conveniences.  Cross-attention can
        use query_sequence_lengths/key_value_sequence_lengths and query_ragged_offsets/key_value_ragged_offsets.
        Ragged offsets use the same compact API convention as thor.layers.Attention: logical shape [2]
        is batched by NetworkInput into enough storage for the [batch + 1] cuDNN offset vector.
        )nbdoc";
}
