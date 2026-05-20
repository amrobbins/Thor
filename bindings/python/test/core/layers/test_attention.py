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


def test_attention_exposes_public_sequence_lengths_and_ragged_offsets():
    n = _net("test_net_attention_public_variable_lengths")
    x = _input_tensor(n, "tokens", [8, 64], thor.DataType.fp16)
    sequence_lengths = _input_tensor(n, "sequence_lengths", [1], thor.DataType.int32)
    ragged_offsets = _input_tensor(n, "ragged_offsets", [2], thor.DataType.int32)

    attention = thor.layers.Attention(
        n,
        x,
        4,
        sequence_lengths=sequence_lengths,
        ragged_offsets=ragged_offsets,
        head_dim=16,
    )

    assert attention.get_use_sequence_lengths()
    assert attention.get_use_ragged_offsets()
    assert attention.get_sequence_lengths_input().get_dimensions() == [1]
    assert attention.get_ragged_offsets_input().get_dimensions() == [2]

    arch = _only_layer_architecture(n, "attention")
    assert arch["use_sequence_lengths"] is True
    assert arch["use_ragged_offsets"] is True
    assert arch["sequence_lengths_input"]["dimensions"] == [1]
    assert arch["ragged_offsets_input"]["dimensions"] == [2]


def test_attention_rejects_invalid_public_variable_length_inputs():
    n = _net("test_net_attention_rejects_invalid_public_variable_lengths")
    x = _input_tensor(n, "tokens", [8, 64], thor.DataType.fp16)
    sequence_lengths = _input_tensor(n, "sequence_lengths", [1], thor.DataType.int32)
    bad_sequence_lengths_dtype = _input_tensor(n, "bad_sequence_lengths_dtype", [1], thor.DataType.fp16)
    bad_sequence_lengths_shape = _input_tensor(n, "bad_sequence_lengths_shape", [2], thor.DataType.int32)
    ragged_offsets = _input_tensor(n, "ragged_offsets", [2], thor.DataType.int32)
    bad_ragged_offsets_dtype = _input_tensor(n, "bad_ragged_offsets_dtype", [2], thor.DataType.fp16)
    bad_ragged_offsets_shape = _input_tensor(n, "bad_ragged_offsets_shape", [1], thor.DataType.int32)

    with pytest.raises((RuntimeError, ValueError), match="sequenceLengthsInput"):
        thor.layers.Attention(n, x, 4, sequence_lengths=bad_sequence_lengths_dtype)

    with pytest.raises((RuntimeError, ValueError), match="sequenceLengthsInput"):
        thor.layers.Attention(n, x, 4, sequence_lengths=bad_sequence_lengths_shape)

    with pytest.raises((RuntimeError, ValueError), match="raggedOffsetsInput requires sequenceLengthsInput"):
        thor.layers.Attention(n, x, 4, ragged_offsets=ragged_offsets)

    with pytest.raises((RuntimeError, ValueError), match="raggedOffsetsInput"):
        thor.layers.Attention(n, x, 4, sequence_lengths=sequence_lengths, ragged_offsets=bad_ragged_offsets_dtype)

    with pytest.raises((RuntimeError, ValueError), match="raggedOffsetsInput"):
        thor.layers.Attention(n, x, 4, sequence_lengths=sequence_lengths, ragged_offsets=bad_ragged_offsets_shape)

def test_attention_allows_ragged_offsets_with_dropout_and_rope():
    n = _net("test_net_attention_ragged_offsets_with_dropout_and_rope")
    x = _input_tensor(n, "tokens", [8, 64], thor.DataType.fp16)
    sequence_lengths = _input_tensor(n, "sequence_lengths", [1], thor.DataType.int32)
    ragged_offsets = _input_tensor(n, "ragged_offsets", [2], thor.DataType.int32)

    attention = thor.layers.Attention(
        n,
        x,
        4,
        sequence_lengths=sequence_lengths,
        ragged_offsets=ragged_offsets,
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
    sequence_lengths = _input_tensor(n, "sequence_lengths", [1], thor.DataType.int32)

    with pytest.raises((RuntimeError, ValueError), match="context input dtype"):
        thor.layers.Attention(n, decoder, 4, context_input=encoder_bf16, head_dim=8)

    with pytest.raises((RuntimeError, ValueError), match="requires querySequenceLengthsInput"):
        thor.layers.Attention(n, decoder, 4, context_input=decoder, sequence_lengths=sequence_lengths, head_dim=8)


def test_attention_cross_attention_accepts_separate_query_key_value_sequence_lengths():
    n = _net("test_net_attention_cross_attention_separate_sequence_lengths")
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
    assert attention.get_sequence_lengths_input() is None
    assert attention.get_query_sequence_lengths_input().get_dimensions() == [1]
    assert attention.get_key_value_sequence_lengths_input().get_dimensions() == [1]
    assert attention.get_feature_output().get_dimensions() == [5, 40]

    arch = _only_layer_architecture(n, "attention")
    assert arch["use_cross_attention"] is True
    assert arch["use_sequence_lengths"] is True
    assert arch["use_separate_sequence_lengths"] is True
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


def test_attention_rejects_mixed_or_incomplete_separate_sequence_lengths():
    n = _net("test_net_attention_rejects_mixed_separate_sequence_lengths")
    decoder = _input_tensor(n, "decoder_tokens", [5, 32], thor.DataType.fp16)
    encoder = _input_tensor(n, "encoder_tokens", [7, 48], thor.DataType.fp16)
    sequence_lengths = _input_tensor(n, "sequence_lengths", [1], thor.DataType.int32)
    q_lengths = _input_tensor(n, "query_sequence_lengths", [1], thor.DataType.int32)
    kv_lengths = _input_tensor(n, "key_value_sequence_lengths", [1], thor.DataType.int32)
    bad_q_lengths = _input_tensor(n, "bad_query_sequence_lengths", [2], thor.DataType.int32)

    with pytest.raises((RuntimeError, ValueError), match="sequence_lengths or query_sequence_lengths"):
        thor.layers.Attention(
            n,
            decoder,
            4,
            sequence_lengths=sequence_lengths,
            query_sequence_lengths=q_lengths,
            key_value_sequence_lengths=kv_lengths,
            head_dim=8,
        )

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


def test_attention_cross_attention_accepts_separate_query_key_value_ragged_metadata():
    n = _net("test_net_attention_cross_attention_separate_ragged_metadata")
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
    assert attention.get_ragged_offsets_input() is None
    assert attention.get_query_ragged_offsets_input().get_dimensions() == [2]
    assert attention.get_key_value_ragged_offsets_input().get_dimensions() == [2]
    assert attention.get_feature_output().get_dimensions() == [5, 40]

    arch = _only_layer_architecture(n, "attention")
    assert arch["use_cross_attention"] is True
    assert arch["use_sequence_lengths"] is True
    assert arch["use_separate_sequence_lengths"] is True
    assert arch["use_ragged_offsets"] is True
    assert arch["use_separate_ragged_offsets"] is True
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


def test_attention_rejects_mixed_or_incomplete_separate_ragged_metadata():
    n = _net("test_net_attention_rejects_mixed_separate_ragged_metadata")
    decoder = _input_tensor(n, "decoder_tokens", [5, 32], thor.DataType.fp16)
    encoder = _input_tensor(n, "encoder_tokens", [7, 48], thor.DataType.fp16)
    sequence_lengths = _input_tensor(n, "sequence_lengths", [1], thor.DataType.int32)
    q_lengths = _input_tensor(n, "query_sequence_lengths", [1], thor.DataType.int32)
    kv_lengths = _input_tensor(n, "key_value_sequence_lengths", [1], thor.DataType.int32)
    ragged_offsets = _input_tensor(n, "ragged_offsets", [2], thor.DataType.int32)
    q_offsets = _input_tensor(n, "query_ragged_offsets", [2], thor.DataType.int32)
    kv_offsets = _input_tensor(n, "key_value_ragged_offsets", [2], thor.DataType.int32)
    bad_q_offsets = _input_tensor(n, "bad_query_ragged_offsets", [1], thor.DataType.int32)

    with pytest.raises((RuntimeError, ValueError), match="ragged_offsets or query_ragged_offsets"):
        thor.layers.Attention(
            n,
            decoder,
            4,
            sequence_lengths=sequence_lengths,
            ragged_offsets=ragged_offsets,
            query_ragged_offsets=q_offsets,
            key_value_ragged_offsets=kv_offsets,
            head_dim=8,
        )

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

    with pytest.raises((RuntimeError, ValueError), match="requires queryRaggedOffsetsInput"):
        thor.layers.Attention(
            n,
            decoder,
            4,
            context_input=encoder,
            query_sequence_lengths=q_lengths,
            key_value_sequence_lengths=kv_lengths,
            ragged_offsets=ragged_offsets,
            head_dim=8,
        )
