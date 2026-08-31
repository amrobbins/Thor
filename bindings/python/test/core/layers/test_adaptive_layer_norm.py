import json

import numpy as np
import pytest
import thor
from thor.physical import numpy_dtypes


def _net():
    return thor.Network("test_net_adaptive_layer_norm")


def _input_tensor(n: thor.Network, name, dims, dtype=thor.DataType.fp16):
    ni = thor.layers.NetworkInput(n, name, dims, dtype)
    return ni.get_feature_output()


def _only_layer_architecture(n: thor.Network, layer_type: str):
    layers = [layer for layer in json.loads(n.get_architecture_json())["layers"] if layer["layer_type"] == layer_type]
    assert len(layers) == 1
    return layers[0]


def _cpu_tensor(values: np.ndarray, dtype: thor.DataType) -> thor.physical.PhysicalTensor:
    values = np.asarray(values, dtype=numpy_dtypes.from_thor(dtype), order="C")
    placement = thor.physical.Placement(thor.physical.DeviceType.cpu, 0)
    descriptor = thor.physical.PhysicalTensor.Descriptor(dtype, list(values.shape))
    tensor = thor.physical.PhysicalTensor(placement, descriptor)
    tensor.numpy()[...] = values
    return tensor


def _physical_ragged(
    values: np.ndarray,
    offsets: np.ndarray,
    offsets_dtype: thor.DataType,
    *,
    max_values_per_row: int,
):
    return thor.physical.PhysicalRaggedTensor(
        _cpu_tensor(values, thor.DataType.fp16),
        _cpu_tensor(offsets, offsets_dtype),
        max_values_per_row=max_values_per_row,
    )


def _adaptive_layer_norm_reference(x: np.ndarray, scale: np.ndarray, bias: np.ndarray, normalized_shape, epsilon: float) -> np.ndarray:
    rank = len(normalized_shape)
    axes = tuple(range(x.ndim - rank, x.ndim))
    x32 = x.astype(np.float32)
    scale32 = scale.astype(np.float32)
    bias32 = bias.astype(np.float32)
    leading_rank = x.ndim - rank - 1
    broadcast_shape = (x.shape[0],) + (1,) * leading_rank + tuple(normalized_shape)
    scale32 = scale32.reshape(broadcast_shape)
    bias32 = bias32.reshape(broadcast_shape)
    mean = np.mean(x32, axis=axes, keepdims=True)
    variance = np.mean((x32 - mean) * (x32 - mean), axis=axes, keepdims=True)
    normalized = (x32 - mean) / np.sqrt(variance + np.float32(epsilon))
    return normalized * scale32 + bias32


def _adaptive_layer_norm_reference_for_dtype(
    values: np.ndarray,
    scale: np.ndarray,
    bias: np.ndarray,
    normalized_shape,
    epsilon: float,
    dtype: thor.DataType,
) -> np.ndarray:
    # Match the values actually provided to Thor.  The feature input is stored
    # in the requested dtype, while adaptive scale/bias are fp32 input tensors.
    feature_dtype = numpy_dtypes.from_thor(dtype)
    quantized_values = np.asarray(values, dtype=feature_dtype)
    quantized_scale = np.asarray(scale, dtype=np.float32)
    quantized_bias = np.asarray(bias, dtype=np.float32)
    return _adaptive_layer_norm_reference(quantized_values, quantized_scale, quantized_bias, normalized_shape, epsilon).astype(
        feature_dtype
    )


def _run_adaptive_layer_norm_network(
    values: np.ndarray,
    scale: np.ndarray,
    bias: np.ndarray,
    feature_dims,
    dtype: thor.DataType,
    *,
    normalized_shape=None,
    epsilon=1e-5,
) -> np.ndarray:
    dtype_name = str(dtype).split(".")[-1]
    n = thor.Network(f"test_net_adaptive_layer_norm_numerical_{dtype_name}_{len(feature_dims)}d")
    x = _input_tensor(n, "x", feature_dims, dtype)
    scale_bias_dims = normalized_shape if normalized_shape is not None else [feature_dims[-1]]
    scale_input = _input_tensor(n, "scale", scale_bias_dims, thor.DataType.fp32)
    bias_input = _input_tensor(n, "bias", scale_bias_dims, thor.DataType.fp32)
    kwargs = {"epsilon": epsilon}
    if normalized_shape is not None:
        kwargs["normalized_shape"] = normalized_shape
    aln = thor.layers.AdaptiveLayerNorm(n, x, scale_input, bias_input, **kwargs)
    thor.layers.NetworkOutput(n, "output", aln.get_feature_output(), dtype)

    placed = n.place(
        values.shape[0],
        inference_only=True,
        forced_devices=[0],
        forced_num_stamps_per_gpu=1,
    )
    outputs = placed.infer(
        {
            "x": _cpu_tensor(values, dtype),
            "scale": _cpu_tensor(scale, thor.DataType.fp32),
            "bias": _cpu_tensor(bias, thor.DataType.fp32),
        }
    )
    assert set(outputs.keys()) == {"output"}
    return np.array(outputs["output"].numpy(), copy=True)


def test_adaptive_layer_norm_constructs_default_last_dim_and_output_preserves_shape_dtype():
    n = _net()
    x = _input_tensor(n, "x", [8, 16], thor.DataType.fp16)
    scale = _input_tensor(n, "scale", [16], thor.DataType.fp32)
    bias = _input_tensor(n, "bias", [16], thor.DataType.fp32)

    aln = thor.layers.AdaptiveLayerNorm(n, x, scale, bias)

    assert isinstance(aln, thor.layers.AdaptiveLayerNorm)
    assert aln.get_normalized_shape() == [16]
    assert aln.get_epsilon() == pytest.approx(1e-5)
    assert aln.get_scale_bias_data_type() == thor.DataType.fp32

    y = aln.get_feature_output()
    assert isinstance(y, thor.Tensor)
    assert y.get_dimensions() == x.get_dimensions()
    assert y.get_data_type() == x.get_data_type()


def test_adaptive_layer_norm_constructs_explicit_trailing_shape_and_serializes():
    n = _net()
    x = _input_tensor(n, "x", [4, 8, 16], thor.DataType.bf16)
    scale = _input_tensor(n, "scale", [8, 16], thor.DataType.fp32)
    bias = _input_tensor(n, "bias", [8, 16], thor.DataType.fp32)

    aln = thor.layers.AdaptiveLayerNorm(n, x, scale, bias, normalized_shape=[8, 16], epsilon=1e-4)
    assert aln.get_normalized_shape() == [8, 16]
    assert aln.get_epsilon() == pytest.approx(1e-4)

    arch = _only_layer_architecture(n, "adaptive_layer_norm")
    assert arch["normalized_shape"] == [8, 16]
    assert arch["epsilon"] == pytest.approx(1e-4)
    assert arch["scale_bias_data_type"] == "fp32"
    assert [inp["port"] for inp in arch["inputs"]] == ["feature_input", "scale_input", "bias_input"]


def test_ragged_adaptive_layer_norm_builds_row_conditioned_partition_preserving_output():
    n = thor.Network("test_net_ragged_adaptive_layer_norm_contract")
    tokens = thor.layers.RaggedNetworkInput(
        n,
        "tokens",
        thor.DataType.fp16,
        [32],
        max_total_values=9,
        batch_size=3,
        max_values_per_row=5,
        offsets_data_type=thor.DataType.uint64,
    )
    scale = thor.layers.NetworkInput(n, "scale", [32], thor.DataType.fp32)
    bias = thor.layers.NetworkInput(n, "bias", [32], thor.DataType.fp32)

    aln = thor.layers.AdaptiveLayerNorm(n, tokens, scale.get_feature_output(), bias.get_feature_output())
    output = aln.get_feature_output()
    assert isinstance(output, thor.RaggedTensor)
    assert aln.get_use_ragged()
    assert output.offsets == tokens.offsets
    assert output.values.get_dimensions() == [9, 32]

    arch = _only_layer_architecture(n, "adaptive_layer_norm")
    assert arch["use_ragged"] is True
    assert arch["ragged_feature_input"]["offsets"]["id"] == arch["ragged_feature_output"]["offsets"]["id"]


def test_ragged_adaptive_layer_norm_save_load_preserves_row_conditioning_contract(tmp_path):
    name = "test_ragged_adaptive_layer_norm_round_trip"
    n = thor.Network(name)
    tokens = thor.layers.RaggedNetworkInput(
        n,
        "tokens",
        thor.DataType.fp16,
        [32],
        max_total_values=8,
        batch_size=3,
        max_values_per_row=5,
        offsets_data_type=thor.DataType.uint32,
    )
    scale = thor.layers.NetworkInput(n, "scale", [32], thor.DataType.fp32)
    bias = thor.layers.NetworkInput(n, "bias", [32], thor.DataType.fp32)
    aln = thor.layers.AdaptiveLayerNorm(n, tokens, scale.get_feature_output(), bias.get_feature_output())
    thor.layers.RaggedNetworkOutput(n, "output", aln.get_feature_output())

    save_dir = tmp_path / "ragged_adaptive_layer_norm_model"
    n.save(str(save_dir), overwrite=False)

    loaded = thor.Network(name)
    loaded.load(str(save_dir))
    arch = _only_layer_architecture(loaded, "adaptive_layer_norm")
    assert arch["use_ragged"] is True
    assert arch["ragged_feature_input"]["offsets"]["id"] == arch["ragged_feature_output"]["offsets"]["id"]
    assert arch["ragged_feature_input"]["max_values_per_row"] == 5

    placed = loaded.place(3, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)
    offsets = np.asarray([0, 1, 4, 6], dtype=np.uint32)
    values = np.full((8, 32), np.nan, dtype=np.float16)
    active_values = (np.arange(6 * 32, dtype=np.float32).reshape(6, 32) % 17 - 8.0) / 4.0
    values[:6] = active_values.astype(np.float16)
    row_scale = np.stack(
        [
            np.linspace(0.75, 1.25, 32, dtype=np.float32),
            np.linspace(1.0, 1.5, 32, dtype=np.float32),
            np.linspace(-0.5, 0.5, 32, dtype=np.float32),
        ]
    )
    row_bias = np.stack(
        [
            np.linspace(-0.1, 0.1, 32, dtype=np.float32),
            np.linspace(0.2, -0.2, 32, dtype=np.float32),
            np.linspace(0.05, 0.25, 32, dtype=np.float32),
        ]
    )
    result = placed.infer(
        {
            "tokens": _physical_ragged(values, offsets, thor.DataType.uint32, max_values_per_row=5),
            "scale": _cpu_tensor(row_scale, thor.DataType.fp32),
            "bias": _cpu_tensor(row_bias, thor.DataType.fp32),
        }
    )["output"]

    expected = np.empty((6, 32), dtype=np.float32)
    for row, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        row_values = values[int(start) : int(end)].astype(np.float32)
        mean = row_values.mean(axis=1, keepdims=True)
        variance = ((row_values - mean) ** 2).mean(axis=1, keepdims=True)
        expected[int(start) : int(end)] = (row_values - mean) / np.sqrt(variance + np.float32(1.0e-5))
        expected[int(start) : int(end)] = expected[int(start) : int(end)] * row_scale[row] + row_bias[row]

    np.testing.assert_array_equal(np.array(result.offsets.numpy(), copy=True), offsets)
    np.testing.assert_allclose(
        np.array(result.values.numpy(), copy=True)[:6].astype(np.float32),
        expected.astype(np.float16).astype(np.float32),
        atol=3.0e-3,
        rtol=3.0e-3,
    )


def test_adaptive_layer_norm_rejects_bad_normalized_shape():
    n = _net()
    x = _input_tensor(n, "x", [4, 8, 16], thor.DataType.fp16)
    scale = _input_tensor(n, "scale", [16], thor.DataType.fp32)
    bias = _input_tensor(n, "bias", [16], thor.DataType.fp32)

    with pytest.raises((RuntimeError, ValueError), match="normalizedShape"):
        thor.layers.AdaptiveLayerNorm(n, x, scale, bias, normalized_shape=[16, 8])

    with pytest.raises((RuntimeError, ValueError), match="normalizedShape"):
        thor.layers.AdaptiveLayerNorm(n, x, scale, bias, normalized_shape=[])


def test_adaptive_layer_norm_rejects_bad_epsilon():
    n = _net()
    x = _input_tensor(n, "x", [8, 16], thor.DataType.fp16)
    scale = _input_tensor(n, "scale", [16], thor.DataType.fp32)
    bias = _input_tensor(n, "bias", [16], thor.DataType.fp32)

    with pytest.raises(ValueError, match="epsilon must be > 0"):
        thor.layers.AdaptiveLayerNorm(n, x, scale, bias, epsilon=0.0)

    with pytest.raises(ValueError, match="epsilon must be > 0"):
        thor.layers.AdaptiveLayerNorm(n, x, scale, bias, epsilon=-1e-5)



def test_adaptive_layer_norm_rejects_fp32_normalized_feature_count_that_cudnn_primary_engines_do_not_support():
    n = _net()
    x = _input_tensor(n, "x", [3, 16], thor.DataType.fp32)
    scale = _input_tensor(n, "scale", [16], thor.DataType.fp32)
    bias = _input_tensor(n, "bias", [16], thor.DataType.fp32)

    with pytest.raises((RuntimeError, ValueError), match="multiple of 32"):
        thor.layers.AdaptiveLayerNorm(n, x, scale, bias)


def test_adaptive_layer_norm_rejects_unsupported_dtypes_and_shapes():
    n = _net()
    x = _input_tensor(n, "x", [8, 16], thor.DataType.fp16)
    scale = _input_tensor(n, "scale", [16], thor.DataType.fp32)
    bias = _input_tensor(n, "bias", [16], thor.DataType.fp32)

    bad_scale = _input_tensor(n, "bad_scale", [16], thor.DataType.fp16)
    with pytest.raises((RuntimeError, ValueError), match="fp32"):
        thor.layers.AdaptiveLayerNorm(n, x, bad_scale, bias)

    bad_bias = _input_tensor(n, "bad_bias", [8, 16], thor.DataType.fp32)
    with pytest.raises((RuntimeError, ValueError), match="dimensions"):
        thor.layers.AdaptiveLayerNorm(n, x, scale, bad_bias)

    n2 = thor.Network("test_net_adaptive_layer_norm_int")
    int_x = _input_tensor(n2, "x", [8, 16], thor.DataType.int32)
    int_scale = _input_tensor(n2, "scale", [16], thor.DataType.fp32)
    int_bias = _input_tensor(n2, "bias", [16], thor.DataType.fp32)
    with pytest.raises((RuntimeError, ValueError), match="dtype"):
        thor.layers.AdaptiveLayerNorm(n2, int_x, int_scale, int_bias)

    ragged_net = thor.Network("test_net_ragged_adaptive_layer_norm_bad_geometry")
    ragged = thor.layers.RaggedNetworkInput(
        ragged_net,
        "tokens",
        thor.DataType.fp16,
        [2, 16],
        max_total_values=8,
        batch_size=2,
    )
    ragged_scale = thor.layers.NetworkInput(ragged_net, "scale", [16], thor.DataType.fp32)
    ragged_bias = thor.layers.NetworkInput(ragged_net, "bias", [16], thor.DataType.fp32)
    with pytest.raises((RuntimeError, ValueError), match="exactly one"):
        thor.layers.AdaptiveLayerNorm(
            ragged_net,
            ragged,
            ragged_scale.get_feature_output(),
            ragged_bias.get_feature_output(),
            normalized_shape=[16],
        )


def test_adaptive_layer_norm_rejects_wrong_types_and_arity():
    n = _net()
    x = _input_tensor(n, "x", [8, 16], thor.DataType.fp16)
    scale = _input_tensor(n, "scale", [16], thor.DataType.fp32)
    bias = _input_tensor(n, "bias", [16], thor.DataType.fp32)

    with pytest.raises(TypeError):
        thor.layers.AdaptiveLayerNorm()

    with pytest.raises(TypeError):
        thor.layers.AdaptiveLayerNorm(n)

    with pytest.raises(TypeError):
        thor.layers.AdaptiveLayerNorm("not a network", x, scale, bias)

    with pytest.raises(TypeError):
        thor.layers.AdaptiveLayerNorm(n, "not a tensor", scale, bias)

    with pytest.raises(TypeError):
        thor.layers.AdaptiveLayerNorm(n, x, scale, bias, epsilon="1e-5")


@pytest.mark.cuda
@pytest.mark.parametrize("dtype,atol,rtol", [
    (thor.DataType.fp16, 2.0e-3, 2.0e-3),
    (thor.DataType.fp32, 2.5e-5, 2.5e-5),
])
def test_adaptive_layer_norm_forward_matches_numpy_default_last_dim(dtype, atol, rtol):
    # cuDNN's primary AdaptiveLayerNorm engines vectorize the normalized
    # dimension; use hidden=32 to satisfy the fp32 primary-engine
    # load-vector contract while keeping a small nontrivial leading dimension.
    values = (
        np.arange(2 * 4 * 32, dtype=np.float32).reshape(2, 4, 32) % 23 - 11.0
    ) / 5.0
    scale = (
        np.arange(2 * 32, dtype=np.float32).reshape(2, 32) % 7 - 3.0
    ) / 4.0
    bias = (
        np.arange(2 * 32, dtype=np.float32).reshape(2, 32) % 5 - 2.0
    ) / 3.0

    actual = _run_adaptive_layer_norm_network(values, scale, bias, [4, 32], dtype)
    expected = _adaptive_layer_norm_reference_for_dtype(values, scale, bias, [32], 1e-5, dtype)

    np.testing.assert_allclose(actual.astype(np.float32), expected.astype(np.float32), atol=atol, rtol=rtol)


@pytest.mark.cuda
def test_adaptive_layer_norm_forward_matches_numpy_explicit_trailing_shape():
    dtype = thor.DataType.fp16
    epsilon = 1e-4
    values = (np.arange(2 * 2 * 3 * 4, dtype=np.float32).reshape(2, 2, 3, 4) - 11.5) / 3.0
    scale = (np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4) % 7 - 3.0) / 4.0
    bias = (np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4) % 5 - 2.0) / 3.0

    actual = _run_adaptive_layer_norm_network(
        values,
        scale,
        bias,
        [2, 3, 4],
        dtype,
        normalized_shape=[3, 4],
        epsilon=epsilon,
    )
    expected = _adaptive_layer_norm_reference_for_dtype(values, scale, bias, [3, 4], epsilon, dtype)

    np.testing.assert_allclose(actual.astype(np.float32), expected.astype(np.float32), atol=2.0e-3, rtol=2.0e-3)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "offsets_dtype,np_offsets_dtype",
    [
        (thor.DataType.uint32, np.uint32),
        (thor.DataType.uint64, np.uint64),
    ],
)
def test_ragged_adaptive_layer_norm_runtime_broadcasts_per_row_and_ignores_poison_tail(offsets_dtype, np_offsets_dtype):
    batch_size = 3
    max_total_values = 8
    channels = 32
    n = thor.Network(f"test_net_ragged_adaptive_layer_norm_runtime_{np_offsets_dtype.__name__}")
    tokens = thor.layers.RaggedNetworkInput(
        n,
        "tokens",
        thor.DataType.fp16,
        [channels],
        max_total_values=max_total_values,
        batch_size=batch_size,
        max_values_per_row=5,
        offsets_data_type=offsets_dtype,
    )
    scale_input = thor.layers.NetworkInput(n, "scale", [channels], thor.DataType.fp32)
    bias_input = thor.layers.NetworkInput(n, "bias", [channels], thor.DataType.fp32)
    aln = thor.layers.AdaptiveLayerNorm(n, tokens, scale_input.get_feature_output(), bias_input.get_feature_output())
    thor.layers.RaggedNetworkOutput(n, "output", aln.get_feature_output())

    placed = n.place(batch_size, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)

    scale = np.stack(
        [
            np.linspace(0.5, 1.5, channels, dtype=np.float32),
            np.linspace(1.0, 2.0, channels, dtype=np.float32),
            np.linspace(-0.75, 0.75, channels, dtype=np.float32),
        ]
    )
    bias = np.stack(
        [
            np.linspace(-0.2, 0.2, channels, dtype=np.float32),
            np.linspace(0.3, -0.3, channels, dtype=np.float32),
            np.linspace(0.1, 0.5, channels, dtype=np.float32),
        ]
    )

    def run(offset_values, active_values):
        values = np.full((max_total_values, channels), np.nan, dtype=np.float16)
        values[: len(active_values)] = np.asarray(active_values, dtype=np.float16)
        result = placed.infer(
            {
                "tokens": _physical_ragged(
                    values,
                    np.asarray(offset_values, dtype=np_offsets_dtype),
                    offsets_dtype,
                    max_values_per_row=5,
                ),
                "scale": _cpu_tensor(scale, thor.DataType.fp32),
                "bias": _cpu_tensor(bias, thor.DataType.fp32),
            }
        )["output"]
        return result, np.array(result.values.numpy(), copy=True)

    short_offsets = [0, 2, 2, 5]
    short_values = (np.arange(5 * channels, dtype=np.float32).reshape(5, channels) % 19 - 9.0) / 4.0
    short_result, short_output = run(short_offsets, short_values)
    expected_rows = []
    for row, (start, end) in enumerate(zip(short_offsets[:-1], short_offsets[1:])):
        if end == start:
            continue
        row_values = np.asarray(short_values[start:end], dtype=np.float16).astype(np.float32)
        mean = row_values.mean(axis=1, keepdims=True)
        variance = ((row_values - mean) ** 2).mean(axis=1, keepdims=True)
        normalized = (row_values - mean) / np.sqrt(variance + np.float32(1.0e-5))
        expected_rows.append(normalized * scale[row] + bias[row])
    expected = np.concatenate(expected_rows, axis=0).astype(np.float16)
    np.testing.assert_allclose(short_output[:5].astype(np.float32), expected.astype(np.float32), atol=3.0e-3, rtol=3.0e-3)
    np.testing.assert_array_equal(np.array(short_result.offsets.numpy(), copy=True), np.asarray(short_offsets, dtype=np_offsets_dtype))

    # Reuse the same placed executable across a larger active extent and then an
    # all-empty partition. Undefined inactive values remain NaN poison throughout.
    long_offsets = [0, 3, 4, 7]
    long_values = (np.arange(7 * channels, dtype=np.float32).reshape(7, channels) % 23 - 11.0) / 5.0
    long_result, _ = run(long_offsets, long_values)
    assert int(np.array(long_result.offsets.numpy(), copy=False)[-1]) == 7

    empty_result, _ = run([0, 0, 0, 0], np.empty((0, channels), dtype=np.float32))
    assert int(np.array(empty_result.offsets.numpy(), copy=False)[-1]) == 0

    short_again, short_again_output = run(short_offsets, short_values)
    assert int(np.array(short_again.offsets.numpy(), copy=False)[-1]) == 5
    np.testing.assert_allclose(short_again_output[:5], short_output[:5], atol=0, rtol=0)
