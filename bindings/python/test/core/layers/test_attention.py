import json

import pytest
import thor


def _net(name="test_net_attention"):
    return thor.Network(name)


def _input_tensor(n: thor.Network, name: str, dims, dtype):
    ni = thor.layers.NetworkInput(n, name, dims, dtype)
    return ni.get_feature_output()


def _only_layer_architecture(n: thor.Network, layer_type: str):
    layers = [layer for layer in json.loads(n.get_architecture_json())["layers"] if layer["layer_type"] == layer_type]
    assert len(layers) == 1
    return layers[0]


def test_attention_exposes_public_query_key_value_sequence_lengths_and_ragged_offsets():
    n = _net("test_net_attention_public_variable_lengths")
    x = _input_tensor(n, "tokens", [8, 64], thor.DataType.fp16)
    q_lengths = _input_tensor(n, "query_sequence_lengths", [1], thor.DataType.int32)
    kv_lengths = _input_tensor(n, "key_value_sequence_lengths", [1], thor.DataType.int32)
    q_offsets = _input_tensor(n, "query_ragged_offsets", [2], thor.DataType.int32)
    kv_offsets = _input_tensor(n, "key_value_ragged_offsets", [2], thor.DataType.int32)

    attention = thor.layers.Attention(
        n,
        x,
        4,
        query_sequence_lengths=q_lengths,
        key_value_sequence_lengths=kv_lengths,
        query_ragged_offsets=q_offsets,
        key_value_ragged_offsets=kv_offsets,
        head_dim=16,
    )

    assert attention.get_use_sequence_lengths()
    assert attention.get_use_ragged_offsets()
    assert attention.get_query_sequence_lengths_input().get_dimensions() == [1]
    assert attention.get_key_value_sequence_lengths_input().get_dimensions() == [1]
    assert attention.get_query_ragged_offsets_input().get_dimensions() == [2]
    assert attention.get_key_value_ragged_offsets_input().get_dimensions() == [2]
    assert not hasattr(attention, "get_sequence_lengths_input")
    assert not hasattr(attention, "get_ragged_offsets_input")

    arch = _only_layer_architecture(n, "attention")
    assert arch["use_sequence_lengths"] is True
    assert arch["use_ragged_offsets"] is True
    assert "use_separate_sequence_lengths" not in arch
    assert "use_separate_ragged_offsets" not in arch
    assert "sequence_lengths_input" not in arch
    assert "ragged_offsets_input" not in arch
    assert arch["query_sequence_lengths_input"]["dimensions"] == [1]
    assert arch["key_value_sequence_lengths_input"]["dimensions"] == [1]
    assert arch["query_ragged_offsets_input"]["dimensions"] == [2]
    assert arch["key_value_ragged_offsets_input"]["dimensions"] == [2]


def test_attention_rejects_invalid_public_variable_length_inputs():
    n = _net("test_net_attention_rejects_invalid_public_variable_lengths")
    x = _input_tensor(n, "tokens", [8, 64], thor.DataType.fp16)
    q_lengths = _input_tensor(n, "query_sequence_lengths", [1], thor.DataType.int32)
    kv_lengths = _input_tensor(n, "key_value_sequence_lengths", [1], thor.DataType.int32)
    bad_q_lengths_dtype = _input_tensor(n, "bad_query_sequence_lengths_dtype", [1], thor.DataType.fp16)
    bad_q_lengths_shape = _input_tensor(n, "bad_query_sequence_lengths_shape", [2], thor.DataType.int32)
    q_offsets = _input_tensor(n, "query_ragged_offsets", [2], thor.DataType.int32)
    kv_offsets = _input_tensor(n, "key_value_ragged_offsets", [2], thor.DataType.int32)
    bad_q_offsets_dtype = _input_tensor(n, "bad_query_ragged_offsets_dtype", [2], thor.DataType.fp16)
    bad_q_offsets_shape = _input_tensor(n, "bad_query_ragged_offsets_shape", [1], thor.DataType.int32)

    with pytest.raises((RuntimeError, ValueError), match="querySequenceLengthsInput"):
        thor.layers.Attention(
            n,
            x,
            4,
            query_sequence_lengths=bad_q_lengths_dtype,
            key_value_sequence_lengths=kv_lengths,
        )

    with pytest.raises((RuntimeError, ValueError), match="querySequenceLengthsInput"):
        thor.layers.Attention(
            n,
            x,
            4,
            query_sequence_lengths=bad_q_lengths_shape,
            key_value_sequence_lengths=kv_lengths,
        )

    with pytest.raises((RuntimeError, ValueError), match="requires querySequenceLengthsInput"):
        thor.layers.Attention(
            n,
            x,
            4,
            query_ragged_offsets=q_offsets,
            key_value_ragged_offsets=kv_offsets,
        )

    with pytest.raises((RuntimeError, ValueError), match="queryRaggedOffsetsInput"):
        thor.layers.Attention(
            n,
            x,
            4,
            query_sequence_lengths=q_lengths,
            key_value_sequence_lengths=kv_lengths,
            query_ragged_offsets=bad_q_offsets_dtype,
            key_value_ragged_offsets=kv_offsets,
        )

    with pytest.raises((RuntimeError, ValueError), match="queryRaggedOffsetsInput"):
        thor.layers.Attention(
            n,
            x,
            4,
            query_sequence_lengths=q_lengths,
            key_value_sequence_lengths=kv_lengths,
            query_ragged_offsets=bad_q_offsets_shape,
            key_value_ragged_offsets=kv_offsets,
        )


def test_attention_legacy_single_metadata_python_kwargs_are_removed():
    n = _net("test_net_attention_legacy_single_metadata_kwargs_removed")
    x = _input_tensor(n, "tokens", [8, 64], thor.DataType.fp16)
    sequence_lengths = _input_tensor(n, "sequence_lengths", [1], thor.DataType.int32)
    ragged_offsets = _input_tensor(n, "ragged_offsets", [2], thor.DataType.int32)

    with pytest.raises(TypeError, match="sequence_lengths"):
        thor.layers.Attention(n, x, 4, sequence_lengths=sequence_lengths)

    with pytest.raises(TypeError, match="ragged_offsets"):
        thor.layers.Attention(n, x, 4, ragged_offsets=ragged_offsets)


def test_attention_allows_ragged_offsets_with_dropout_and_rope():
    n = _net("test_net_attention_ragged_offsets_with_dropout_and_rope")
    x = _input_tensor(n, "tokens", [8, 64], thor.DataType.fp16)
    q_lengths = _input_tensor(n, "query_sequence_lengths", [1], thor.DataType.int32)
    kv_lengths = _input_tensor(n, "key_value_sequence_lengths", [1], thor.DataType.int32)
    q_offsets = _input_tensor(n, "query_ragged_offsets", [2], thor.DataType.int32)
    kv_offsets = _input_tensor(n, "key_value_ragged_offsets", [2], thor.DataType.int32)

    attention = thor.layers.Attention(
        n,
        x,
        4,
        query_sequence_lengths=q_lengths,
        key_value_sequence_lengths=kv_lengths,
        query_ragged_offsets=q_offsets,
        key_value_ragged_offsets=kv_offsets,
        head_dim=16,
        use_rope=True,
        dropout_probability=0.1,
        dropout_seed=7,
        dropout_offset=11,
    )

    assert attention.get_use_ragged_offsets()
    assert attention.get_use_rope()
    assert attention.get_dropout_probability() == pytest.approx(0.1)
    assert attention.get_dropout_seed() == 7
    assert attention.get_dropout_offset() == 11

    arch = _only_layer_architecture(n, "attention")
    assert arch["use_ragged_offsets"] is True
    assert arch["use_rope"] is True
    assert arch["dropout_probability"] == pytest.approx(0.1)
    assert arch["dropout_seed"] == 7
    assert arch["dropout_offset"] == 11


def _assert_parameter_shape(arch, name: str, shape):
    assert name in arch["parameters"]
    assert arch["parameters"][name]["name"] == name
    assert arch["parameters"][name]["shape"] == shape


def test_attention_exposes_context_input_and_splits_query_context_parameter_shapes():
    n = _net("test_net_attention_context_input_split_parameter_shapes")
    decoder = _input_tensor(n, "decoder_tokens", [5, 32], thor.DataType.fp16)
    encoder = _input_tensor(n, "encoder_tokens", [7, 48], thor.DataType.fp16)

    attention = thor.layers.Attention(
        n,
        decoder,
        4,
        context_input=encoder,
        num_key_value_heads=2,
        head_dim=8,
        value_dim=6,
        output_features=40,
        has_bias=True,
    )

    assert attention.get_use_cross_attention()
    assert attention.get_context_input().get_dimensions() == [7, 48]
    assert attention.get_feature_output().get_dimensions() == [5, 40]

    arch = _only_layer_architecture(n, "attention")
    assert arch["use_cross_attention"] is True
    assert arch["feature_input"]["dimensions"] == [5, 32]
    assert arch["context_input"]["dimensions"] == [7, 48]
    assert arch["feature_output"]["dimensions"] == [5, 40]

    _assert_parameter_shape(arch, "query_weights", [32, 32])
    _assert_parameter_shape(arch, "key_weights", [48, 16])
    _assert_parameter_shape(arch, "value_weights", [48, 12])
    _assert_parameter_shape(arch, "output_weights", [24, 40])
    _assert_parameter_shape(arch, "query_bias", [32])
    _assert_parameter_shape(arch, "key_bias", [16])
    _assert_parameter_shape(arch, "value_bias", [12])
    _assert_parameter_shape(arch, "output_bias", [40])


def test_attention_self_attention_architecture_remains_context_free():
    n = _net("test_net_attention_self_attention_context_free")
    x = _input_tensor(n, "tokens", [8, 64], thor.DataType.fp16)

    attention = thor.layers.Attention(n, x, 4, head_dim=16)

    assert not attention.get_use_cross_attention()
    assert attention.get_context_input() is None

    arch = _only_layer_architecture(n, "attention")
    assert arch["use_cross_attention"] is False
    assert "context_input" not in arch
    _assert_parameter_shape(arch, "query_weights", [64, 64])
    _assert_parameter_shape(arch, "key_weights", [64, 64])
    _assert_parameter_shape(arch, "value_weights", [64, 64])


def test_attention_context_input_rejects_invalid_current_scope_inputs():
    n = _net("test_net_attention_context_input_validation")
    decoder = _input_tensor(n, "decoder_tokens", [5, 32], thor.DataType.fp16)
    encoder_bf16 = _input_tensor(n, "encoder_tokens_bf16", [7, 48], thor.DataType.bf16)

    with pytest.raises((RuntimeError, ValueError), match="context input dtype"):
        thor.layers.Attention(n, decoder, 4, context_input=encoder_bf16, head_dim=8)


def test_attention_cross_attention_accepts_query_key_value_sequence_lengths():
    n = _net("test_net_attention_cross_attention_sequence_lengths")
    decoder = _input_tensor(n, "decoder_tokens", [5, 32], thor.DataType.fp16)
    encoder = _input_tensor(n, "encoder_tokens", [7, 48], thor.DataType.fp16)
    q_lengths = _input_tensor(n, "query_sequence_lengths", [1], thor.DataType.int32)
    kv_lengths = _input_tensor(n, "key_value_sequence_lengths", [1], thor.DataType.int32)

    attention = thor.layers.Attention(
        n,
        decoder,
        4,
        context_input=encoder,
        query_sequence_lengths=q_lengths,
        key_value_sequence_lengths=kv_lengths,
        num_key_value_heads=2,
        head_dim=8,
        value_dim=6,
        output_features=40,
    )

    assert attention.get_use_cross_attention()
    assert attention.get_use_sequence_lengths()
    assert attention.get_query_sequence_lengths_input().get_dimensions() == [1]
    assert attention.get_key_value_sequence_lengths_input().get_dimensions() == [1]
    assert attention.get_feature_output().get_dimensions() == [5, 40]

    arch = _only_layer_architecture(n, "attention")
    assert arch["use_cross_attention"] is True
    assert arch["use_sequence_lengths"] is True
    assert "use_separate_sequence_lengths" not in arch
    assert "sequence_lengths_input" not in arch
    assert arch["query_sequence_lengths_input"]["dimensions"] == [1]
    assert arch["key_value_sequence_lengths_input"]["dimensions"] == [1]
    assert arch["feature_input"]["dimensions"] == [5, 32]
    assert arch["context_input"]["dimensions"] == [7, 48]
    assert arch["feature_output"]["dimensions"] == [5, 40]

    _assert_parameter_shape(arch, "query_weights", [32, 32])
    _assert_parameter_shape(arch, "key_weights", [48, 16])
    _assert_parameter_shape(arch, "value_weights", [48, 12])
    _assert_parameter_shape(arch, "output_weights", [24, 40])


def test_attention_rejects_incomplete_query_key_value_sequence_lengths():
    n = _net("test_net_attention_rejects_incomplete_sequence_lengths")
    decoder = _input_tensor(n, "decoder_tokens", [5, 32], thor.DataType.fp16)
    encoder = _input_tensor(n, "encoder_tokens", [7, 48], thor.DataType.fp16)
    q_lengths = _input_tensor(n, "query_sequence_lengths", [1], thor.DataType.int32)
    kv_lengths = _input_tensor(n, "key_value_sequence_lengths", [1], thor.DataType.int32)
    bad_q_lengths = _input_tensor(n, "bad_query_sequence_lengths", [2], thor.DataType.int32)

    with pytest.raises((RuntimeError, ValueError), match="query_sequence_lengths and key_value_sequence_lengths"):
        thor.layers.Attention(n, decoder, 4, query_sequence_lengths=q_lengths, head_dim=8)

    with pytest.raises((RuntimeError, ValueError), match="query_sequence_lengths and key_value_sequence_lengths"):
        thor.layers.Attention(n, decoder, 4, key_value_sequence_lengths=kv_lengths, head_dim=8)

    with pytest.raises((RuntimeError, ValueError), match="querySequenceLengthsInput"):
        thor.layers.Attention(
            n,
            decoder,
            4,
            context_input=encoder,
            query_sequence_lengths=bad_q_lengths,
            key_value_sequence_lengths=kv_lengths,
            head_dim=8,
        )


def test_attention_cross_attention_accepts_query_key_value_ragged_metadata():
    n = _net("test_net_attention_cross_attention_ragged_metadata")
    decoder = _input_tensor(n, "decoder_tokens", [5, 32], thor.DataType.fp16)
    encoder = _input_tensor(n, "encoder_tokens", [7, 48], thor.DataType.fp16)
    q_lengths = _input_tensor(n, "query_sequence_lengths", [1], thor.DataType.int32)
    kv_lengths = _input_tensor(n, "key_value_sequence_lengths", [1], thor.DataType.int32)
    q_offsets = _input_tensor(n, "query_ragged_offsets", [2], thor.DataType.int32)
    kv_offsets = _input_tensor(n, "key_value_ragged_offsets", [2], thor.DataType.int32)

    attention = thor.layers.Attention(
        n,
        decoder,
        4,
        context_input=encoder,
        query_sequence_lengths=q_lengths,
        key_value_sequence_lengths=kv_lengths,
        query_ragged_offsets=q_offsets,
        key_value_ragged_offsets=kv_offsets,
        num_key_value_heads=2,
        head_dim=8,
        value_dim=8,
        output_features=40,
        dropout_probability=0.1,
        dropout_seed=17,
        dropout_offset=23,
    )

    assert attention.get_use_cross_attention()
    assert attention.get_use_sequence_lengths()
    assert attention.get_use_ragged_offsets()
    assert attention.get_query_ragged_offsets_input().get_dimensions() == [2]
    assert attention.get_key_value_ragged_offsets_input().get_dimensions() == [2]
    assert attention.get_feature_output().get_dimensions() == [5, 40]

    arch = _only_layer_architecture(n, "attention")
    assert arch["use_cross_attention"] is True
    assert arch["use_sequence_lengths"] is True
    assert arch["use_ragged_offsets"] is True
    assert "use_separate_sequence_lengths" not in arch
    assert "use_separate_ragged_offsets" not in arch
    assert "ragged_offsets_input" not in arch
    assert arch["query_ragged_offsets_input"]["dimensions"] == [2]
    assert arch["key_value_ragged_offsets_input"]["dimensions"] == [2]
    assert arch["query_sequence_lengths_input"]["dimensions"] == [1]
    assert arch["key_value_sequence_lengths_input"]["dimensions"] == [1]
    assert arch["feature_input"]["dimensions"] == [5, 32]
    assert arch["context_input"]["dimensions"] == [7, 48]
    assert arch["feature_output"]["dimensions"] == [5, 40]
    assert arch["dropout_probability"] == pytest.approx(0.1)
    assert arch["dropout_seed"] == 17
    assert arch["dropout_offset"] == 23

    _assert_parameter_shape(arch, "query_weights", [32, 32])
    _assert_parameter_shape(arch, "key_weights", [48, 16])
    _assert_parameter_shape(arch, "value_weights", [48, 16])
    _assert_parameter_shape(arch, "output_weights", [32, 40])


def test_attention_rejects_incomplete_query_key_value_ragged_metadata():
    n = _net("test_net_attention_rejects_incomplete_ragged_metadata")
    decoder = _input_tensor(n, "decoder_tokens", [5, 32], thor.DataType.fp16)
    q_lengths = _input_tensor(n, "query_sequence_lengths", [1], thor.DataType.int32)
    kv_lengths = _input_tensor(n, "key_value_sequence_lengths", [1], thor.DataType.int32)
    q_offsets = _input_tensor(n, "query_ragged_offsets", [2], thor.DataType.int32)
    kv_offsets = _input_tensor(n, "key_value_ragged_offsets", [2], thor.DataType.int32)
    bad_q_offsets = _input_tensor(n, "bad_query_ragged_offsets", [1], thor.DataType.int32)

    with pytest.raises((RuntimeError, ValueError), match="query_ragged_offsets and key_value_ragged_offsets"):
        thor.layers.Attention(
            n,
            decoder,
            4,
            query_sequence_lengths=q_lengths,
            key_value_sequence_lengths=kv_lengths,
            query_ragged_offsets=q_offsets,
            head_dim=8,
        )

    with pytest.raises((RuntimeError, ValueError), match="requires querySequenceLengthsInput"):
        thor.layers.Attention(
            n,
            decoder,
            4,
            query_ragged_offsets=q_offsets,
            key_value_ragged_offsets=kv_offsets,
            head_dim=8,
        )

    with pytest.raises((RuntimeError, ValueError), match="queryRaggedOffsetsInput"):
        thor.layers.Attention(
            n,
            decoder,
            4,
            query_sequence_lengths=q_lengths,
            key_value_sequence_lengths=kv_lengths,
            query_ragged_offsets=bad_q_offsets,
            key_value_ragged_offsets=kv_offsets,
            head_dim=8,
        )


def test_attention_exposes_public_score_bias_input_and_preserves_projection_bias_api():
    n = _net("test_net_attention_score_bias_input")
    x = _input_tensor(n, "tokens", [5, 32], thor.DataType.fp16)
    score_bias = _input_tensor(n, "score_bias", [4, 5, 5], thor.DataType.fp32)

    attention = thor.layers.Attention(
        n,
        x,
        4,
        head_dim=8,
        has_bias=True,
        score_bias_input=score_bias,
    )

    assert attention.get_use_score_bias()
    assert attention.get_score_bias_input().get_dimensions() == [4, 5, 5]
    assert attention.get_has_bias()
    assert attention.get_feature_output().get_dimensions() == [5, 32]

    arch = _only_layer_architecture(n, "attention")
    assert arch["use_score_bias"] is True
    assert arch["score_bias_input"]["dimensions"] == [4, 5, 5]
    _assert_parameter_shape(arch, "query_bias", [32])
    _assert_parameter_shape(arch, "key_bias", [32])
    _assert_parameter_shape(arch, "value_bias", [32])
    _assert_parameter_shape(arch, "output_bias", [32])


def test_attention_score_bias_accepts_head_broadcast_and_cross_attention_key_value_length():
    n = _net("test_net_attention_score_bias_cross_attention")
    decoder = _input_tensor(n, "decoder_tokens", [3, 32], thor.DataType.fp16)
    encoder = _input_tensor(n, "encoder_tokens", [7, 48], thor.DataType.fp16)
    score_bias = _input_tensor(n, "score_bias", [1, 3, 7], thor.DataType.fp32)

    attention = thor.layers.Attention(
        n,
        decoder,
        4,
        context_input=encoder,
        score_bias_input=score_bias,
        num_key_value_heads=2,
        head_dim=8,
        value_dim=6,
        output_features=40,
    )

    assert attention.get_use_cross_attention()
    assert attention.get_use_score_bias()
    assert attention.get_score_bias_input().get_dimensions() == [1, 3, 7]
    assert attention.get_feature_output().get_dimensions() == [3, 40]

    arch = _only_layer_architecture(n, "attention")
    assert arch["use_cross_attention"] is True
    assert arch["use_score_bias"] is True
    assert arch["score_bias_input"]["dimensions"] == [1, 3, 7]
    assert arch["feature_input"]["dimensions"] == [3, 32]
    assert arch["context_input"]["dimensions"] == [7, 48]
    _assert_parameter_shape(arch, "query_weights", [32, 32])
    _assert_parameter_shape(arch, "key_weights", [48, 16])
    _assert_parameter_shape(arch, "value_weights", [48, 12])


def test_attention_score_bias_accepts_sequence_broadcast_shape():
    n = _net("test_net_attention_score_bias_sequence_broadcast")
    decoder = _input_tensor(n, "decoder_tokens", [3, 32], thor.DataType.fp16)
    encoder = _input_tensor(n, "encoder_tokens", [7, 48], thor.DataType.fp16)
    score_bias = _input_tensor(n, "score_bias", [4, 1, 7], thor.DataType.fp32)

    attention = thor.layers.Attention(
        n,
        decoder,
        4,
        context_input=encoder,
        score_bias_input=score_bias,
        num_key_value_heads=2,
        head_dim=8,
        value_dim=6,
        output_features=40,
    )

    assert attention.get_use_score_bias()
    assert attention.get_score_bias_input().get_dimensions() == [4, 1, 7]
    arch = _only_layer_architecture(n, "attention")
    assert arch["use_score_bias"] is True
    assert arch["score_bias_input"]["dimensions"] == [4, 1, 7]


def test_attention_score_bias_rejects_invalid_shape_dtype_and_decode_masks():
    n = _net("test_net_attention_score_bias_validation")
    x = _input_tensor(n, "tokens", [5, 32], thor.DataType.fp16)
    good_score_bias = _input_tensor(n, "good_score_bias", [1, 5, 5], thor.DataType.fp32)
    bad_head_score_bias = _input_tensor(n, "bad_head_score_bias", [2, 5, 5], thor.DataType.fp32)
    bad_sequence_score_bias = _input_tensor(n, "bad_sequence_score_bias", [1, 5, 6], thor.DataType.fp32)
    bad_dtype_score_bias = _input_tensor(n, "bad_dtype_score_bias", [1, 5, 5], thor.DataType.fp16)

    with pytest.raises((RuntimeError, ValueError), match="scoreBiasInput dimensions"):
        thor.layers.Attention(n, x, 4, head_dim=8, score_bias_input=bad_head_score_bias)

    with pytest.raises((RuntimeError, ValueError), match="scoreBiasInput dimensions"):
        thor.layers.Attention(n, x, 4, head_dim=8, score_bias_input=bad_sequence_score_bias)

    with pytest.raises((RuntimeError, ValueError), match="scoreBiasInput dtype"):
        thor.layers.Attention(n, x, 4, head_dim=8, score_bias_input=bad_dtype_score_bias)

    with pytest.raises((RuntimeError, ValueError), match="scoreBiasInput"):
        thor.layers.Attention(
            n,
            x,
            4,
            head_dim=8,
            score_bias_input=good_score_bias,
            mask_kind="causal_bottom_right",
        )
