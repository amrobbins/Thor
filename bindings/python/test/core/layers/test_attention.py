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



def test_attention_rejects_invalid_public_sequence_length_inputs():
    n = _net("test_net_attention_rejects_invalid_public_sequence_lengths")
    x = _input_tensor(n, "tokens", [8, 64], thor.DataType.fp16)
    kv_lengths = _input_tensor(n, "key_value_sequence_lengths", [1], thor.DataType.int32)
    bad_q_lengths_dtype = _input_tensor(n, "bad_query_sequence_lengths_dtype", [1], thor.DataType.fp16)
    bad_q_lengths_shape = _input_tensor(n, "bad_query_sequence_lengths_shape", [2], thor.DataType.int32)

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

def test_attention_legacy_single_metadata_python_kwargs_are_removed():
    n = _net("test_net_attention_legacy_single_metadata_kwargs_removed")
    x = _input_tensor(n, "tokens", [8, 64], thor.DataType.fp16)
    sequence_lengths = _input_tensor(n, "sequence_lengths", [1], thor.DataType.int32)
    ragged_offsets = _input_tensor(n, "ragged_offsets", [2], thor.DataType.uint32)

    with pytest.raises(TypeError, match="sequence_lengths"):
        thor.layers.Attention(n, x, 4, sequence_lengths=sequence_lengths)

    with pytest.raises(TypeError, match="ragged_offsets"):
        thor.layers.Attention(n, x, 4, ragged_offsets=ragged_offsets)

    with pytest.raises(TypeError, match="query_ragged_offsets"):
        thor.layers.Attention(n, x, 4, query_ragged_offsets=ragged_offsets)

    with pytest.raises(TypeError, match="key_value_ragged_offsets"):
        thor.layers.Attention(n, x, 4, key_value_ragged_offsets=ragged_offsets)



def test_attention_accepts_and_returns_canonical_ragged_tensor():
    n = _net("test_attention_canonical_ragged_tensor")
    tokens = thor.layers.RaggedNetworkInput(
        n,
        "tokens",
        thor.DataType.fp16,
        [32],
        max_total_values=11,
        batch_size=3,
        offsets_data_type=thor.DataType.uint64,
    )

    attention = thor.layers.Attention(n, tokens, 4, head_dim=8, output_features=24)

    assert isinstance(attention.get_feature_output(), thor.RaggedTensor)
    assert attention.get_feature_output().values.get_dimensions() == [11, 24]
    assert attention.get_feature_output().offsets == tokens.offsets
    assert attention.get_context_input() is None

    arch = _only_layer_architecture(n, "attention")
    assert arch["use_ragged"] is True
    assert "ragged_feature_input" in arch
    assert "ragged_feature_output" in arch
    assert "query_ragged_offsets_input" not in arch
    assert "key_value_ragged_offsets_input" not in arch


def test_attention_ragged_cross_attention_rope_allows_independent_partitions():
    n = _net("test_attention_ragged_cross_attention_rope_partition_policy")
    query = thor.layers.RaggedNetworkInput(
        n,
        "query",
        thor.DataType.fp16,
        [32],
        max_total_values=9,
        batch_size=2,
        offsets_data_type=thor.DataType.uint32,
    )
    context = thor.layers.RaggedNetworkInput(
        n,
        "context",
        thor.DataType.fp16,
        [32],
        max_total_values=12,
        batch_size=2,
        offsets_data_type=thor.DataType.uint64,
    )

    attention = thor.layers.Attention(
        n,
        query,
        4,
        head_dim=8,
        context_input=context,
        use_rope=True,
        rope_rotary_dim=8,
    )

    assert attention.get_feature_output().offsets == query.offsets
    assert attention.get_context_input() == context


def test_attention_ragged_cross_attention_accepts_per_row_rope_origins():
    n = _net("test_attention_ragged_cross_attention_per_row_rope_origins")
    query = thor.layers.RaggedNetworkInput(
        n, "query", thor.DataType.fp16, [32], max_total_values=9, batch_size=2
    )
    context = thor.layers.RaggedNetworkInput(
        n, "context", thor.DataType.fp16, [32], max_total_values=12, batch_size=2
    )
    query_origins = _input_tensor(n, "query_origins", [1], thor.DataType.int32)
    key_origins = _input_tensor(n, "key_origins", [1], thor.DataType.int32)

    attention = thor.layers.Attention(
        n,
        query,
        4,
        head_dim=8,
        context_input=context,
        use_rope=True,
        rope_rotary_dim=8,
        rope_query_position_offsets=query_origins,
        rope_key_position_offsets=key_origins,
    )

    assert attention.get_rope_query_position_offsets_input() == query_origins
    assert attention.get_rope_key_position_offsets_input() == key_origins
    arch = _only_layer_architecture(n, "attention")
    assert arch["use_query_rope_position_offsets"] is True
    assert arch["use_key_rope_position_offsets"] is True
    assert arch["query_rope_position_offsets_input"]["dimensions"] == [1]
    assert arch["key_rope_position_offsets_input"]["dimensions"] == [1]


def test_attention_per_row_rope_origins_require_rope_and_ragged_inputs():
    n = _net("test_attention_per_row_rope_origins_require_rope_and_ragged")
    dense = _input_tensor(n, "dense", [4, 32], thor.DataType.fp16)
    origins = _input_tensor(n, "origins", [1], thor.DataType.int32)

    with pytest.raises((RuntimeError, ValueError), match="require use_rope=True"):
        thor.layers.Attention(n, dense, 4, head_dim=8, rope_query_position_offsets=origins)

    with pytest.raises((RuntimeError, ValueError), match="Ragged|ragged|per-row"):
        thor.layers.Attention(
            n,
            dense,
            4,
            head_dim=8,
            use_rope=True,
            rope_rotary_dim=8,
            rope_query_position_offsets=origins,
        )


def test_attention_cross_attention_exposes_independent_rope_query_key_offsets():
    n = _net("test_attention_cross_attention_independent_rope_offsets")
    query = _input_tensor(n, "query", [3, 32], thor.DataType.fp16)
    context = _input_tensor(n, "context", [5, 32], thor.DataType.fp16)

    attention = thor.layers.Attention(
        n,
        query,
        4,
        context_input=context,
        head_dim=8,
        use_rope=True,
        rope_rotary_dim=8,
        rope_position_offset=7,
        rope_query_position_offset=100,
        rope_key_position_offset=0,
    )

    assert attention.get_rope_query_position_offset() == 100
    assert attention.get_rope_key_position_offset() == 0
    arch = _only_layer_architecture(n, "attention")
    assert arch["rope_options"]["position_offset"] == 7
    assert arch["rope_query_position_offset"] == 100
    assert arch["rope_key_position_offset"] == 0


def test_attention_shared_rope_position_offset_remains_query_key_default():
    n = _net("test_attention_shared_rope_offset_default")
    x = _input_tensor(n, "tokens", [4, 32], thor.DataType.fp16)

    attention = thor.layers.Attention(
        n,
        x,
        4,
        head_dim=8,
        use_rope=True,
        rope_rotary_dim=8,
        rope_position_offset=13,
    )

    assert attention.get_rope_query_position_offset() == 13
    assert attention.get_rope_key_position_offset() == 13


def test_attention_independent_rope_offsets_require_rope_enabled():
    n = _net("test_attention_independent_rope_offsets_require_rope")
    x = _input_tensor(n, "tokens", [4, 32], thor.DataType.fp16)

    with pytest.raises((RuntimeError, ValueError), match="require use_rope=True"):
        thor.layers.Attention(n, x, 4, head_dim=8, rope_query_position_offset=4)


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


def _residual_epilogue(dtype=thor.DataType.bf16):
    attention_output = thor.layers.Attention.epilogue_input(
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    residual = thor.layers.Attention.epilogue_aux_input(
        "residual",
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    return attention_output + residual


def test_attention_python_binding_builds_self_attention_residual_epilogue():
    n = _net("test_attention_python_self_residual_epilogue")
    x = _input_tensor(n, "tokens", [5, 8], thor.DataType.bf16)
    residual = _input_tensor(n, "residual", [5, 8], thor.DataType.bf16)

    attention = thor.layers.Attention(
        n,
        x,
        2,
        head_dim=4,
        weights_data_type=thor.DataType.bf16,
        compute_data_type=thor.DataType.fp32,
        output_data_type=thor.DataType.bf16,
        epilogue=_residual_epilogue(),
        epilogue_inputs={"residual": residual},
    )

    assert attention.get_has_epilogue()
    assert attention.get_epilogue_input_names() == ["residual"]
    assert attention.get_feature_output().get_dimensions() == [5, 8]
    assert attention.get_feature_output().get_data_type() == thor.DataType.bf16

    arch = _only_layer_architecture(n, "attention")
    assert arch["epilogue"] is not None
    assert set(arch["epilogue"]["expected_input_names"]) == {
        "__attention_epilogue_input",
        "residual",
    }
    assert len(arch["epilogue_inputs"]) == 1
    assert arch["epilogue_inputs"][0]["name"] == "residual"
    assert arch["epilogue_inputs"][0]["tensor"]["dimensions"] == [5, 8]


def test_attention_python_binding_builds_cross_attention_residual_epilogue():
    n = _net("test_attention_python_cross_residual_epilogue")
    query = _input_tensor(n, "query", [5, 8], thor.DataType.bf16)
    context = _input_tensor(n, "context", [7, 12], thor.DataType.bf16)
    residual = _input_tensor(n, "residual", [5, 8], thor.DataType.bf16)

    attention = thor.layers.Attention(
        n,
        query,
        2,
        context_input=context,
        num_key_value_heads=1,
        head_dim=4,
        value_dim=4,
        output_features=8,
        weights_data_type=thor.DataType.bf16,
        compute_data_type=thor.DataType.fp32,
        output_data_type=thor.DataType.bf16,
        epilogue=_residual_epilogue(),
        epilogue_inputs={"residual": residual},
    )

    assert attention.get_use_cross_attention()
    assert attention.get_has_epilogue()
    assert attention.get_epilogue_input_names() == ["residual"]
    assert attention.get_feature_output().get_dimensions() == [5, 8]


def test_attention_python_binding_rejects_invalid_residual_epilogue_bindings():
    n = _net("test_attention_python_rejects_invalid_residual_epilogue_bindings")
    x = _input_tensor(n, "tokens", [5, 8], thor.DataType.bf16)
    residual = _input_tensor(n, "residual", [5, 8], thor.DataType.bf16)
    bad_shape = _input_tensor(n, "bad_shape", [4, 8], thor.DataType.bf16)
    bad_dtype = _input_tensor(n, "bad_dtype", [5, 8], thor.DataType.fp32)

    with pytest.raises((RuntimeError, ValueError), match="input mismatch"):
        thor.layers.Attention(
            n,
            x,
            2,
            head_dim=4,
            output_data_type=thor.DataType.bf16,
            epilogue=_residual_epilogue(),
        )

    with pytest.raises((RuntimeError, ValueError), match="shape must match"):
        thor.layers.Attention(
            n,
            x,
            2,
            head_dim=4,
            output_data_type=thor.DataType.bf16,
            epilogue=_residual_epilogue(),
            epilogue_inputs={"residual": bad_shape},
        )

    with pytest.raises((RuntimeError, ValueError), match="dtype must match"):
        thor.layers.Attention(
            n,
            x,
            2,
            head_dim=4,
            output_data_type=thor.DataType.bf16,
            epilogue=_residual_epilogue(),
            epilogue_inputs={"residual": bad_dtype},
        )

    wrong_storage_annotation = (
        thor.layers.Attention.epilogue_input(
            output_dtype=thor.DataType.bf16,
            compute_dtype=thor.DataType.fp32,
        )
        + thor.layers.Attention.epilogue_aux_input(
            "residual",
            output_dtype=thor.DataType.fp16,
            compute_dtype=thor.DataType.fp32,
        )
    )
    with pytest.raises((RuntimeError, ValueError), match="output dtype annotation"):
        thor.layers.Attention(
            n,
            x,
            2,
            head_dim=4,
            output_data_type=thor.DataType.bf16,
            epilogue=wrong_storage_annotation,
            epilogue_inputs={"residual": residual},
        )

    primary_only = thor.layers.Attention.epilogue_input(
        output_dtype=thor.DataType.bf16,
        compute_dtype=thor.DataType.fp32,
    )
    with pytest.raises((RuntimeError, ValueError), match="input mismatch"):
        thor.layers.Attention(
            n,
            x,
            2,
            head_dim=4,
            output_data_type=thor.DataType.bf16,
            epilogue=primary_only,
            epilogue_inputs={"residual": residual},
        )

    residual_only = thor.layers.Attention.epilogue_aux_input(
        "residual",
        output_dtype=thor.DataType.bf16,
        compute_dtype=thor.DataType.fp32,
    )
    with pytest.raises((RuntimeError, ValueError), match="must include tensor input"):
        thor.layers.Attention(
            n,
            x,
            2,
            head_dim=4,
            output_data_type=thor.DataType.bf16,
            epilogue=residual_only,
            epilogue_inputs={"residual": residual},
        )

    with pytest.raises((RuntimeError, ValueError), match="reserved"):
        thor.layers.Attention.epilogue_aux_input("feature_input")

    shape_changing = thor.layers.Attention.epilogue_input(
        output_dtype=thor.DataType.bf16,
        compute_dtype=thor.DataType.fp32,
    ).reshape([1])
    with pytest.raises((RuntimeError, ValueError), match="preserve the output projection shape"):
        thor.layers.Attention(
            n,
            x,
            2,
            head_dim=4,
            output_data_type=thor.DataType.bf16,
            epilogue=shape_changing,
        )


def test_attention_python_binding_without_epilogue_is_unchanged():
    n = _net("test_attention_python_without_epilogue_is_unchanged")
    x = _input_tensor(n, "tokens", [5, 8], thor.DataType.bf16)
    attention = thor.layers.Attention(n, x, 2, head_dim=4)

    assert not attention.get_has_epilogue()
    assert attention.get_epilogue_input_names() == []
    arch = _only_layer_architecture(n, "attention")
    assert arch["epilogue"] is None
    assert arch["epilogue_inputs"] == []


def test_attention_exposes_transient_training_dropout_control():
    n = _net("test_net_attention_training_dropout_control")
    x = _input_tensor(n, "tokens", [4, 32], thor.DataType.fp16)
    attention = thor.layers.Attention(
        n,
        x,
        2,
        head_dim=16,
        dropout_probability=0.25,
        dropout_seed=1234,
        dropout_offset=5678,
    )

    assert attention.is_training_dropout_enabled() is True
    assert n.get_num_training_dropout_controllable_layers() == 1

    attention.set_training_dropout_enabled(False)
    assert attention.is_training_dropout_enabled() is False
    assert n.is_training_dropout_enabled() is False

    architecture = _only_layer_architecture(n, "attention")
    assert architecture["dropout_probability"] == pytest.approx(0.25)
    assert "training_dropout_enabled" not in architecture
