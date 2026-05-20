import pytest
import thor


def _net(name="test_net_scaled_dot_product_attention"):
    return thor.Network(name)


def _input_tensor(n: thor.Network, name: str, dims, dtype):
    ni = thor.layers.NetworkInput(n, name, dims, dtype)
    return ni.get_feature_output()


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
