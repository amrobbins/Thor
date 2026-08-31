import json

import numpy as np
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


def _cpu_tensor(values: np.ndarray, dtype: thor.DataType) -> thor.physical.PhysicalTensor:
    values = np.asarray(values, dtype=thor.physical.numpy_dtypes.from_thor(dtype), order="C")
    placement = thor.physical.Placement(thor.physical.DeviceType.cpu, 0)
    descriptor = thor.physical.PhysicalTensor.Descriptor(dtype, list(values.shape))
    tensor = thor.physical.PhysicalTensor(placement, descriptor)
    tensor.numpy()[...] = values
    return tensor


def test_scaled_dot_product_attention_exposes_ragged_bias_public_surface():
    n = _net("test_sdpa_ragged_bias_public_surface")
    q = thor.layers.RaggedNetworkInput(
        n, "qkv", thor.DataType.fp16, [2, 32], max_total_values=6, batch_size=1
    )
    bias = _input_tensor(n, "bias", [1, 6, 6], thor.DataType.fp32)

    attention = thor.layers.ScaledDotProductAttention(
        n,
        q,
        bias_input=bias,
        output_data_type=thor.DataType.fp16,
    )

    assert not attention.get_use_sequence_lengths()
    assert attention.get_use_ragged_input()
    assert attention.get_tensor_layout() == "bshd"
    assert attention.get_input_names() == [
        "query",
        "key",
        "value",
        "bias",
        "query_ragged_offsets",
        "key_value_ragged_offsets",
    ]
    output = attention.get_feature_output()
    assert isinstance(output, thor.RaggedTensor)
    assert output.values.get_dimensions() == [6, 2, 32]
    assert output.offsets == q.offsets

def test_scaled_dot_product_attention_canonical_ragged_offsets_scale_with_batch_size():
    n = _net("test_sdpa_canonical_ragged_offsets_batch_gt_one")
    q = thor.layers.RaggedNetworkInput(
        n,
        "qkv",
        thor.DataType.fp16,
        [2, 16],
        max_total_values=9,
        batch_size=3,
        offsets_data_type=thor.DataType.uint64,
    )

    attention = thor.layers.ScaledDotProductAttention(n, q)

    assert attention.get_use_ragged_input()
    assert attention.get_query_ragged_input().offsets.get_dimensions() == [4]
    assert attention.get_feature_output().offsets == q.offsets




@pytest.mark.cuda
def test_scaled_dot_product_attention_canonical_ragged_surface_executes_packed_runtime():
    batch_size = 2
    n = _net("test_sdpa_canonical_ragged_executes")
    q = thor.layers.RaggedNetworkInput(
        n,
        "qkv",
        thor.DataType.fp16,
        [2, 16],
        max_total_values=5,
        batch_size=batch_size,
        offsets_data_type=thor.DataType.uint32,
    )
    attention = thor.layers.ScaledDotProductAttention(n, q)
    thor.layers.RaggedNetworkOutput(n, "output", attention.get_feature_output())

    placed = n.place(
        batch_size,
        inference_only=True,
        forced_devices=[0],
        forced_num_stamps_per_gpu=1,
    )
    values_np = np.linspace(-0.5, 0.5, 5 * 2 * 16, dtype=np.float16).reshape(5, 2, 16)
    offsets_np = np.array([0, 2, 5], dtype=np.uint32)
    result = placed.infer(
        {
            "qkv": thor.physical.PhysicalRaggedTensor(
                _cpu_tensor(values_np, thor.DataType.fp16),
                _cpu_tensor(offsets_np, thor.DataType.uint32),
            )
        }
    )["output"]

    assert isinstance(result, thor.physical.PhysicalRaggedTensor)
    assert np.array_equal(result.offsets.numpy(), offsets_np)
    output_values = np.asarray(result.values.numpy())
    assert output_values.shape == (5, 2, 16)
    assert np.all(np.isfinite(output_values))


def test_scaled_dot_product_attention_raw_ragged_metadata_kwargs_are_removed():
    n = _net("test_sdpa_raw_ragged_metadata_kwargs_removed")
    q = _input_tensor(n, "q", [2, 8, 16], thor.DataType.fp16)
    offsets = _input_tensor(n, "offsets", [2], thor.DataType.uint32)

    for kwargs in (
        {"ragged_offsets": offsets},
        {"query_ragged_offsets": offsets},
        {"key_value_ragged_offsets": offsets},
    ):
        with pytest.raises(TypeError):
            thor.layers.ScaledDotProductAttention(n, q, **kwargs)


def test_scaled_dot_product_attention_rejects_alibi_causal_top_left_positive_right_bound():
    n = _net("test_sdpa_rejects_alibi_causal_top_left_positive_right_bound")
    q = _input_tensor(n, "q", [4, 6, 16], thor.DataType.fp16)

    with pytest.raises(ValueError, match="ALiBi.*diagonalRightBound == 0"):
        thor.layers.ScaledDotProductAttention(
            n,
            q,
            mask_kind="causal_top_left",
            diagonal_right_bound=1,
            use_alibi_mask=True,
            output_data_type=thor.DataType.fp16,
        )


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
    q = thor.layers.RaggedNetworkInput(
        n, "q", thor.DataType.bf16, [4, 16], max_total_values=4, batch_size=1, offsets_data_type=thor.DataType.uint32
    )
    kv = thor.layers.RaggedNetworkInput(
        n, "kv", thor.DataType.bf16, [2, 16], max_total_values=5, batch_size=1, offsets_data_type=thor.DataType.uint64
    )

    attention = thor.layers.ScaledDotProductAttention(n, q, key_input=kv, value_input=kv)

    assert not attention.get_use_sequence_lengths()
    assert attention.get_use_ragged_input()
    output = attention.get_feature_output()
    assert isinstance(output, thor.RaggedTensor)
    assert output.values.get_dimensions() == [4, 4, 16]
    assert output.offsets == q.offsets

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
    assert arch["use_ragged_input"] is False
    assert arch["query_sequence_lengths_input"]["id"] == seq.get_id()
    assert arch["key_value_sequence_lengths_input"]["id"] == seq.get_id()
    assert arch["output"]["dimensions"] == [4, 8, 32]


def test_scaled_dot_product_attention_rejects_invalid_variable_length_metadata():
    n = _net("test_sdpa_rejects_invalid_variable_length_metadata")
    q = thor.layers.RaggedNetworkInput(
        n, "q", thor.DataType.fp16, [8, 32], max_total_values=2, batch_size=1
    )
    seq = _input_tensor(n, "seq", [1], thor.DataType.int32)
    bad_seq = _input_tensor(n, "bad_seq", [2], thor.DataType.int32)

    attention = thor.layers.ScaledDotProductAttention(n, q)
    assert attention.get_use_ragged_input()
    assert not attention.get_use_sequence_lengths()

    dense_q = _input_tensor(n, "dense_q", [2, 8, 32], thor.DataType.fp16)
    with pytest.raises((RuntimeError, ValueError), match="SequenceLengthsInput"):
        thor.layers.ScaledDotProductAttention(n, dense_q, sequence_lengths=bad_seq)

    with pytest.raises((RuntimeError, ValueError), match="not also provide sequenceLengthsInput"):
        thor.layers.ScaledDotProductAttention(n, q, sequence_lengths=seq)

    with pytest.raises((RuntimeError, ValueError), match="BSHD"):
        thor.layers.ScaledDotProductAttention(n, q, tensor_layout="bhsd")

    with pytest.raises((RuntimeError, ValueError), match="either sequence_lengths"):
        thor.layers.ScaledDotProductAttention(
            n, dense_q, sequence_lengths=seq, query_sequence_lengths=seq, key_value_sequence_lengths=seq
        )

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
    return {
        name: _input_tensor(n, name, [1, 1, 1, 1], thor.DataType.fp32) for name in names
    }


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


def test_scaled_dot_product_attention_dense_query_ragged_kv_public_surface():
    n = _net("test_sdpa_dense_query_ragged_kv_public_surface")
    q = _input_tensor(n, "q", [3, 2, 16], thor.DataType.fp16)
    kv = thor.layers.RaggedNetworkInput(
        n,
        "kv",
        thor.DataType.fp16,
        [2, 16],
        max_total_values=5,
        batch_size=2,
        offsets_data_type=thor.DataType.uint64,
    )

    attention = thor.layers.ScaledDotProductAttention(n, q, key_input=kv, value_input=kv)

    assert attention.get_use_ragged_input()
    assert not attention.get_query_is_ragged()
    assert attention.get_key_value_is_ragged()
    assert attention.get_tensor_layout() == "bshd"
    assert attention.get_input_names() == ["query", "key", "value", "key_value_ragged_offsets"]
    output = attention.get_feature_output()
    assert isinstance(output, thor.Tensor)
    assert output.get_dimensions() == [3, 2, 16]

    arch = _only_layer_architecture(n, "scaled_dot_product_attention")
    assert arch["version"] == "2.1.0"
    assert arch["use_ragged_input"] is True
    assert arch["query_ragged"] is False
    assert arch["key_value_ragged"] is True
    assert "query_ragged_input" not in arch
    assert "key_ragged_input" in arch
    assert "value_ragged_input" in arch


def test_scaled_dot_product_attention_ragged_query_dense_kv_public_surface():
    n = _net("test_sdpa_ragged_query_dense_kv_public_surface")
    q = thor.layers.RaggedNetworkInput(
        n,
        "q",
        thor.DataType.bf16,
        [2, 16],
        max_total_values=5,
        batch_size=2,
        offsets_data_type=thor.DataType.uint32,
    )
    kv = _input_tensor(n, "kv", [4, 2, 16], thor.DataType.bf16)

    attention = thor.layers.ScaledDotProductAttention(n, q, key_input=kv, value_input=kv)

    assert attention.get_use_ragged_input()
    assert attention.get_query_is_ragged()
    assert not attention.get_key_value_is_ragged()
    assert attention.get_tensor_layout() == "bshd"
    assert attention.get_input_names() == ["query", "key", "value", "query_ragged_offsets"]
    output = attention.get_feature_output()
    assert isinstance(output, thor.RaggedTensor)
    assert output.values.get_dimensions() == [5, 2, 16]
    assert output.offsets == q.offsets

    arch = _only_layer_architecture(n, "scaled_dot_product_attention")
    assert arch["query_ragged"] is True
    assert arch["key_value_ragged"] is False
    assert "query_ragged_input" in arch
    assert "key_ragged_input" not in arch
    assert "value_ragged_input" not in arch


def test_scaled_dot_product_attention_mixed_mode_rejects_mismatched_key_value_domains():
    n = _net("test_sdpa_mixed_mode_rejects_mismatched_key_value_domains")
    q = _input_tensor(n, "q", [3, 2, 16], thor.DataType.fp16)
    dense_kv = _input_tensor(n, "dense_kv", [4, 2, 16], thor.DataType.fp16)
    ragged_kv = thor.layers.RaggedNetworkInput(
        n,
        "ragged_kv",
        thor.DataType.fp16,
        [2, 16],
        max_total_values=5,
        batch_size=2,
    )

    with pytest.raises(TypeError, match="key_input and value_input must both be dense or both be ragged"):
        thor.layers.ScaledDotProductAttention(n, q, key_input=ragged_kv, value_input=dense_kv)
    with pytest.raises(TypeError, match="key_input and value_input must both be dense or both be ragged"):
        thor.layers.ScaledDotProductAttention(n, q, key_input=dense_kv, value_input=ragged_kv)


@pytest.mark.cuda
def test_scaled_dot_product_attention_dense_query_ragged_kv_executes_mixed_runtime():
    batch_size = 2
    n = _net("test_sdpa_dense_query_ragged_kv_executes_mixed_runtime")
    q = thor.layers.NetworkInput(n, "q", [3, 2, 16], thor.DataType.fp16)
    kv = thor.layers.RaggedNetworkInput(
        n,
        "kv",
        thor.DataType.fp16,
        [2, 16],
        max_total_values=5,
        batch_size=batch_size,
        offsets_data_type=thor.DataType.uint64,
    )
    attention = thor.layers.ScaledDotProductAttention(
        n, q.get_feature_output(), key_input=kv, value_input=kv
    )
    thor.layers.NetworkOutput(n, "output", attention.get_feature_output(), thor.DataType.fp16)

    placed = n.place(batch_size, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)
    q_np = np.linspace(-0.4, 0.4, batch_size * 3 * 2 * 16, dtype=np.float16).reshape(batch_size, 3, 2, 16)
    kv_np = np.linspace(-0.5, 0.5, 5 * 2 * 16, dtype=np.float16).reshape(5, 2, 16)
    offsets_np = np.array([0, 2, 5], dtype=np.uint64)

    result = placed.infer(
        {
            "q": _cpu_tensor(q_np, thor.DataType.fp16),
            "kv": thor.physical.PhysicalRaggedTensor(
                _cpu_tensor(kv_np, thor.DataType.fp16),
                _cpu_tensor(offsets_np, thor.DataType.uint64),
            ),
        }
    )["output"]

    output = np.asarray(result.numpy())
    assert output.shape == (batch_size, 3, 2, 16)
    assert np.all(np.isfinite(output))


@pytest.mark.cuda
def test_scaled_dot_product_attention_ragged_query_dense_kv_executes_mixed_runtime():
    batch_size = 2
    n = _net("test_sdpa_ragged_query_dense_kv_executes_mixed_runtime")
    q = thor.layers.RaggedNetworkInput(
        n,
        "q",
        thor.DataType.fp16,
        [2, 16],
        max_total_values=5,
        batch_size=batch_size,
        offsets_data_type=thor.DataType.uint32,
    )
    kv = thor.layers.NetworkInput(n, "kv", [4, 2, 16], thor.DataType.fp16)
    attention = thor.layers.ScaledDotProductAttention(
        n, q, key_input=kv.get_feature_output(), value_input=kv.get_feature_output()
    )
    thor.layers.RaggedNetworkOutput(n, "output", attention.get_feature_output())

    placed = n.place(batch_size, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)
    q_np = np.linspace(-0.5, 0.5, 5 * 2 * 16, dtype=np.float16).reshape(5, 2, 16)
    offsets_np = np.array([0, 1, 5], dtype=np.uint32)
    kv_np = np.linspace(-0.4, 0.4, batch_size * 4 * 2 * 16, dtype=np.float16).reshape(batch_size, 4, 2, 16)

    result = placed.infer(
        {
            "q": thor.physical.PhysicalRaggedTensor(
                _cpu_tensor(q_np, thor.DataType.fp16),
                _cpu_tensor(offsets_np, thor.DataType.uint32),
            ),
            "kv": _cpu_tensor(kv_np, thor.DataType.fp16),
        }
    )["output"]

    assert isinstance(result, thor.physical.PhysicalRaggedTensor)
    assert np.array_equal(result.offsets.numpy(), offsets_np)
    output_values = np.asarray(result.values.numpy())
    assert output_values.shape == (5, 2, 16)
    assert np.all(np.isfinite(output_values))
