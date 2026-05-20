import json

import pytest
import thor


def _net(name="test_net_scaled_dot_product_attention"):
    return thor.Network(name)


def _input_tensor(n: thor.Network, name: str, dims, dtype):
    ni = thor.layers.NetworkInput(n, name, dims, dtype)
    return ni.get_feature_output()


def _only_layer_architecture(n: thor.Network, layer_type: str):
    layers = [layer for layer in json.loads(n.get_architecture_json())["layers"] if layer["layer_type"] == layer_type]
    assert len(layers) == 1
    return layers[0]


def test_scaled_dot_product_attention_exposes_ragged_bias_public_surface():
    n = _net("test_sdpa_ragged_bias_public_surface")
    q = _input_tensor(n, "q", [6, 2, 32], thor.DataType.fp16)
    k = _input_tensor(n, "k", [6, 2, 32], thor.DataType.fp16)
    v = _input_tensor(n, "v", [6, 2, 32], thor.DataType.fp16)
    bias = _input_tensor(n, "bias", [1, 6, 6], thor.DataType.fp32)
    seq = _input_tensor(n, "sequence_lengths", [1], thor.DataType.int32)
    offsets = _input_tensor(n, "ragged_offsets", [2], thor.DataType.int32)

    attention = thor.layers.ScaledDotProductAttention(
        n,
        q,
        key_input=k,
        value_input=v,
        bias_input=bias,
        tensor_layout="bshd",
        sequence_lengths=seq,
        ragged_offsets=offsets,
        output_data_type=thor.DataType.fp16,
    )

    assert attention.get_use_sequence_lengths()
    assert attention.get_use_ragged_offsets()
    assert attention.get_tensor_layout() == "bshd"
    assert attention.get_input_names() == [
        "query",
        "key",
        "value",
        "bias",
        "query_sequence_lengths",
        "key_value_sequence_lengths",
        "query_ragged_offsets",
        "key_value_ragged_offsets",
    ]
    assert attention.get_feature_output().get_dimensions() == [6, 2, 32]


def test_scaled_dot_product_attention_accepts_sequence_broadcast_bias_shape():
    n = _net("test_sdpa_sequence_broadcast_bias_shape")
    q = _input_tensor(n, "q", [4, 5, 16], thor.DataType.fp16)
    k = _input_tensor(n, "k", [2, 7, 16], thor.DataType.fp16)
    v = _input_tensor(n, "v", [2, 7, 16], thor.DataType.fp16)
    bias = _input_tensor(n, "bias", [1, 1, 7], thor.DataType.fp32)

    attention = thor.layers.ScaledDotProductAttention(
        n,
        q,
        key_input=k,
        value_input=v,
        bias_input=bias,
        output_data_type=thor.DataType.fp16,
    )

    assert attention.get_use_bias()
    assert attention.get_bias_input().get_dimensions() == [1, 1, 7]
    assert attention.get_feature_output().get_dimensions() == [4, 5, 16]


def test_scaled_dot_product_attention_allows_cross_attention_separate_metadata():
    n = _net("test_sdpa_cross_attention_separate_metadata")
    q = _input_tensor(n, "q", [4, 4, 16], thor.DataType.bf16)
    k = _input_tensor(n, "k", [5, 2, 16], thor.DataType.bf16)
    v = _input_tensor(n, "v", [5, 2, 16], thor.DataType.bf16)
    q_seq = _input_tensor(n, "q_seq", [1], thor.DataType.int32)
    kv_seq = _input_tensor(n, "kv_seq", [1], thor.DataType.int32)
    q_offsets = _input_tensor(n, "q_offsets", [2], thor.DataType.int32)
    kv_offsets = _input_tensor(n, "kv_offsets", [2], thor.DataType.int32)

    attention = thor.layers.ScaledDotProductAttention(
        n,
        q,
        key_input=k,
        value_input=v,
        query_sequence_lengths=q_seq,
        key_value_sequence_lengths=kv_seq,
        query_ragged_offsets=q_offsets,
        key_value_ragged_offsets=kv_offsets,
        tensor_layout="bshd",
    )

    assert attention.get_use_sequence_lengths()
    assert attention.get_use_ragged_offsets()
    assert attention.get_feature_output().get_dimensions() == [4, 4, 16]


def test_scaled_dot_product_attention_exposes_philox_dropout_and_serializes_public_surface():
    n = _net("test_sdpa_dropout_public_surface")
    q = _input_tensor(n, "q", [4, 8, 32], thor.DataType.fp16)
    seq = _input_tensor(n, "seq", [1], thor.DataType.int32)

    attention = thor.layers.ScaledDotProductAttention(
        n,
        q,
        sequence_lengths=seq,
        dropout_probability=0.125,
        dropout_seed=1234,
        dropout_offset=5678,
    )

    assert attention.get_dropout_probability() == pytest.approx(0.125)
    assert attention.get_dropout_seed() == 1234
    assert attention.get_dropout_offset() == 5678

    arch = _only_layer_architecture(n, "scaled_dot_product_attention")
    assert arch["tensor_layout"] == "bhsd"
    assert arch["mask_kind"] == "none"
    assert arch["attention_scale"] is None
    assert arch["dropout_probability"] == pytest.approx(0.125)
    assert arch["dropout_seed"] == 1234
    assert arch["dropout_offset"] == 5678
    assert arch["use_bias"] is False
    assert arch["use_sequence_lengths"] is True
    assert arch["use_ragged_offsets"] is False
    assert arch["query_sequence_lengths_input"]["id"] == seq.get_id()
    assert arch["key_value_sequence_lengths_input"]["id"] == seq.get_id()
    assert arch["output"]["dimensions"] == [4, 8, 32]


def test_scaled_dot_product_attention_rejects_invalid_variable_length_metadata():
    n = _net("test_sdpa_rejects_invalid_variable_length_metadata")
    q = _input_tensor(n, "q", [2, 8, 32], thor.DataType.fp16)
    seq = _input_tensor(n, "seq", [1], thor.DataType.int32)
    offsets = _input_tensor(n, "offsets", [2], thor.DataType.int32)
    bad_seq = _input_tensor(n, "bad_seq", [2], thor.DataType.int32)
    bad_offsets = _input_tensor(n, "bad_offsets", [1], thor.DataType.int32)

    with pytest.raises((RuntimeError, ValueError), match="raggedOffsetsInput requires sequenceLengthsInput"):
        thor.layers.ScaledDotProductAttention(n, q, ragged_offsets=offsets)

    with pytest.raises((RuntimeError, ValueError), match="SequenceLengthsInput"):
        thor.layers.ScaledDotProductAttention(n, q, sequence_lengths=bad_seq)

    with pytest.raises((RuntimeError, ValueError), match="RaggedOffsetsInput"):
        thor.layers.ScaledDotProductAttention(n, q, sequence_lengths=seq, ragged_offsets=bad_offsets)

    with pytest.raises((RuntimeError, ValueError), match="either sequence_lengths"):
        thor.layers.ScaledDotProductAttention(
            n, q, sequence_lengths=seq, query_sequence_lengths=seq, key_value_sequence_lengths=seq)


def test_scaled_dot_product_attention_rejects_invalid_dropout_configuration():
    n = _net("test_sdpa_rejects_invalid_dropout_configuration")
    q = _input_tensor(n, "q", [2, 8, 32], thor.DataType.fp16)

    with pytest.raises((RuntimeError, ValueError), match="dropout_probability"):
        thor.layers.ScaledDotProductAttention(n, q, dropout_probability=-0.01)

    with pytest.raises((RuntimeError, ValueError), match="dropout_probability"):
        thor.layers.ScaledDotProductAttention(n, q, dropout_probability=1.0)

    with pytest.raises((RuntimeError, ValueError), match="dropout_offset"):
        thor.layers.ScaledDotProductAttention(n, q, dropout_probability=0.1, dropout_offset=-1)

    with pytest.raises((RuntimeError, ValueError), match="dropout"):
        thor.layers.ScaledDotProductAttention(n, q, mask_kind="causal_bottom_right", dropout_probability=0.1)


def _fp8_scale_inputs(n: thor.Network):
    names = [
        "descale_q",
        "descale_k",
        "descale_v",
        "descale_s",
        "scale_s",
        "scale_o",
        "amax_s",
        "amax_o",
    ]
    return {name: _input_tensor(n, name, [1, 1, 1, 1], thor.DataType.fp32) for name in names}


def test_scaled_dot_product_attention_exposes_experimental_fp8_forward_surface():
    n = _net("test_sdpa_experimental_fp8_forward_surface")
    q = _input_tensor(n, "q", [4, 4, 64], thor.DataType.fp8_e4m3)
    scales = _fp8_scale_inputs(n)
    seq = _input_tensor(n, "seq", [1], thor.DataType.int32)

    attention = thor.layers.ScaledDotProductAttention(
        n,
        q,
        tensor_layout="bshd",
        mask_kind="causal_top_left",
        sequence_lengths=seq,
        output_data_type=thor.DataType.fp8_e4m3,
        fp8_descale_q=scales["descale_q"],
        fp8_descale_k=scales["descale_k"],
        fp8_descale_v=scales["descale_v"],
        fp8_descale_s=scales["descale_s"],
        fp8_scale_s=scales["scale_s"],
        fp8_scale_o=scales["scale_o"],
        fp8_amax_s=scales["amax_s"],
        fp8_amax_o=scales["amax_o"],
    )

    assert attention.get_use_fp8_forward_scaling()
    assert attention.get_feature_output().get_data_type() == thor.DataType.fp8_e4m3
    assert attention.get_feature_output().get_dimensions() == [4, 4, 64]
    assert attention.get_input_names() == [
        "query",
        "key",
        "value",
        "query_sequence_lengths",
        "key_value_sequence_lengths",
        "fp8_descale_q",
        "fp8_descale_k",
        "fp8_descale_v",
        "fp8_descale_s",
        "fp8_scale_s",
        "fp8_scale_o",
        "fp8_amax_s",
        "fp8_amax_o",
    ]

    arch = _only_layer_architecture(n, "scaled_dot_product_attention")
    assert arch["use_fp8_forward_scaling"] is True
    assert arch["output_data_type"] == "fp8_e4m3"
    assert arch["fp8_descale_q_input"]["id"] == scales["descale_q"].get_id()
    assert arch["fp8_amax_o_input"]["id"] == scales["amax_o"].get_id()


def test_scaled_dot_product_attention_rejects_unsupported_experimental_fp8_forward_surface():
    n = _net("test_sdpa_rejects_unsupported_experimental_fp8_forward_surface")
    q = _input_tensor(n, "q", [4, 4, 64], thor.DataType.fp8_e4m3)
    scales = _fp8_scale_inputs(n)
    bias = _input_tensor(n, "bias", [1, 4, 4], thor.DataType.fp32)

    with pytest.raises((RuntimeError, ValueError), match="FP8 forward requires all"):
        thor.layers.ScaledDotProductAttention(n, q, fp8_descale_q=scales["descale_q"])

    with pytest.raises((RuntimeError, ValueError), match="additive score bias"):
        thor.layers.ScaledDotProductAttention(
            n,
            q,
            bias_input=bias,
            output_data_type=thor.DataType.fp8_e4m3,
            fp8_descale_q=scales["descale_q"],
            fp8_descale_k=scales["descale_k"],
            fp8_descale_v=scales["descale_v"],
            fp8_descale_s=scales["descale_s"],
            fp8_scale_s=scales["scale_s"],
            fp8_scale_o=scales["scale_o"],
            fp8_amax_s=scales["amax_s"],
            fp8_amax_o=scales["amax_o"],
        )

    with pytest.raises((RuntimeError, ValueError), match="dropout"):
        thor.layers.ScaledDotProductAttention(
            n,
            q,
            dropout_probability=0.1,
            output_data_type=thor.DataType.fp8_e4m3,
            fp8_descale_q=scales["descale_q"],
            fp8_descale_k=scales["descale_k"],
            fp8_descale_v=scales["descale_v"],
            fp8_descale_s=scales["descale_s"],
            fp8_scale_s=scales["scale_s"],
            fp8_scale_o=scales["scale_o"],
            fp8_amax_s=scales["amax_s"],
            fp8_amax_o=scales["amax_o"],
        )

    q_d256 = _input_tensor(n, "q_d256", [4, 4, 256], thor.DataType.fp8_e4m3)
    with pytest.raises((RuntimeError, ValueError), match="<= 128"):
        thor.layers.ScaledDotProductAttention(
            n,
            q_d256,
            output_data_type=thor.DataType.fp8_e4m3,
            fp8_descale_q=scales["descale_q"],
            fp8_descale_k=scales["descale_k"],
            fp8_descale_v=scales["descale_v"],
            fp8_descale_s=scales["descale_s"],
            fp8_scale_s=scales["scale_s"],
            fp8_scale_o=scales["scale_o"],
            fp8_amax_s=scales["amax_s"],
            fp8_amax_o=scales["amax_o"],
        )
