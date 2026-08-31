import json

import numpy as np
import pytest
import thor


def _net():
    return thor.Network("test_net_layer_norm")


def _input_tensor(n: thor.Network, dims, dtype=thor.DataType.fp16):
    ni = thor.layers.NetworkInput(n, "input", dims, dtype)
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


def _layer_norm_reference(x: np.ndarray, normalized_shape, epsilon: float) -> np.ndarray:
    rank = len(normalized_shape)
    axes = tuple(range(x.ndim - rank, x.ndim))
    x32 = x.astype(np.float32)
    mean = np.mean(x32, axis=axes, keepdims=True)
    variance = np.mean((x32 - mean) * (x32 - mean), axis=axes, keepdims=True)
    return (x32 - mean) / np.sqrt(variance + np.float32(epsilon))


def _layer_norm_reference_for_dtype(values: np.ndarray, normalized_shape, epsilon: float, dtype: thor.DataType) -> np.ndarray:
    # Match the values actually provided to Thor: _cpu_tensor stores the input
    # in the requested dtype before device execution.  For fp16/bf16 this can
    # slightly change the normalized values, especially for repeated thirds.
    quantized_values = np.asarray(values, dtype=thor.physical.numpy_dtypes.from_thor(dtype))
    return _layer_norm_reference(quantized_values, normalized_shape, epsilon).astype(thor.physical.numpy_dtypes.from_thor(dtype))


def _run_layer_norm_network(values: np.ndarray, feature_dims, dtype: thor.DataType, *, normalized_shape=None, epsilon=1e-5) -> np.ndarray:
    dtype_name = str(dtype).split(".")[-1]
    n = thor.Network(f"test_net_layer_norm_numerical_{dtype_name}_{len(feature_dims)}d")
    x = _input_tensor(n, feature_dims, dtype)
    kwargs = {"epsilon": epsilon}
    if normalized_shape is not None:
        kwargs["normalized_shape"] = normalized_shape
    ln = thor.layers.LayerNorm(n, x, **kwargs)
    thor.layers.NetworkOutput(n, "output", ln.get_feature_output(), dtype)

    placed = n.place(
        values.shape[0],
        inference_only=True,
        forced_devices=[0],
        forced_num_stamps_per_gpu=1,
    )
    outputs = placed.infer({"input": _cpu_tensor(values, dtype)})
    assert set(outputs.keys()) == {"output"}
    return np.array(outputs["output"].numpy(), copy=True)


def test_layer_norm_constructs_default_last_dim_and_output_preserves_shape_dtype():
    n = _net()
    x = _input_tensor(n, [8, 16], thor.DataType.fp16)

    ln = thor.layers.LayerNorm(n, x)

    assert isinstance(ln, thor.layers.LayerNorm)
    assert ln.get_normalized_shape() == [16]
    assert ln.get_epsilon() == pytest.approx(1e-5)
    assert ln.get_parameter_data_type() == thor.DataType.fp32

    y = ln.get_feature_output()
    assert isinstance(y, thor.Tensor)
    assert y.get_dimensions() == x.get_dimensions()
    assert y.get_data_type() == x.get_data_type()


def test_layer_norm_constructs_explicit_trailing_shape_and_serializes():
    n = _net()
    x = _input_tensor(n, [4, 8, 16], thor.DataType.bf16)

    ln = thor.layers.LayerNorm(n, x, normalized_shape=[8, 16], epsilon=1e-4)
    assert ln.get_normalized_shape() == [8, 16]
    assert ln.get_epsilon() == pytest.approx(1e-4)

    arch = _only_layer_architecture(n, "layer_norm")
    assert arch["normalized_shape"] == [8, 16]
    assert arch["epsilon"] == pytest.approx(1e-4)
    assert "weights" in arch["parameters"]
    assert "biases" in arch["parameters"]


def test_layer_norm_rejects_bad_normalized_shape():
    n = _net()
    x = _input_tensor(n, [4, 8, 16], thor.DataType.fp16)

    with pytest.raises((RuntimeError, ValueError), match="normalizedShape"):
        thor.layers.LayerNorm(n, x, normalized_shape=[16, 8])

    with pytest.raises((RuntimeError, ValueError), match="normalizedShape"):
        thor.layers.LayerNorm(n, x, normalized_shape=[])


def test_layer_norm_rejects_bad_epsilon():
    n = _net()
    x = _input_tensor(n, [8, 16], thor.DataType.fp16)

    with pytest.raises(ValueError, match="epsilon must be > 0"):
        thor.layers.LayerNorm(n, x, epsilon=0.0)

    with pytest.raises(ValueError, match="epsilon must be > 0"):
        thor.layers.LayerNorm(n, x, epsilon=-1e-5)


def test_layer_norm_rejects_unsupported_dtypes():
    n = _net()
    x = _input_tensor(n, [8, 16], thor.DataType.fp16)

    with pytest.raises((RuntimeError, ValueError), match="fp32"):
        thor.layers.LayerNorm(n, x, parameter_data_type=thor.DataType.fp16)

    n2 = thor.Network("test_net_layer_norm_int")
    int_x = _input_tensor(n2, [8, 16], thor.DataType.int32)
    with pytest.raises((RuntimeError, ValueError), match="dtype"):
        thor.layers.LayerNorm(n2, int_x)


def test_layer_norm_rejects_wrong_types_and_arity():
    n = _net()
    x = _input_tensor(n, [8, 16], thor.DataType.fp16)

    with pytest.raises(TypeError):
        thor.layers.LayerNorm()

    with pytest.raises(TypeError):
        thor.layers.LayerNorm(n)

    with pytest.raises(TypeError):
        thor.layers.LayerNorm("not a network", x)

    with pytest.raises(TypeError):
        thor.layers.LayerNorm(n, "not a tensor")

    with pytest.raises(TypeError):
        thor.layers.LayerNorm(n, x, epsilon="1e-5")


@pytest.mark.cuda
@pytest.mark.parametrize("dtype,atol,rtol", [
    (thor.DataType.fp16, 1.5e-3, 1.5e-3),
    (thor.DataType.fp32, 2.5e-5, 2.5e-5),
])
def test_layer_norm_forward_matches_numpy_default_last_dim(dtype, atol, rtol):
    values = np.array(
        [
            [[-2.0, -1.0, 0.0, 1.0], [1.5, 2.0, 3.0, 4.0], [-3.0, 0.5, 2.5, 5.0]],
            [[0.25, -0.75, 1.25, 2.25], [4.0, 1.0, -2.0, -5.0], [3.5, 3.0, 2.5, 2.0]],
        ],
        dtype=np.float32,
    )

    actual = _run_layer_norm_network(values, [3, 4], dtype)
    expected = _layer_norm_reference_for_dtype(values, [4], 1e-5, dtype)

    np.testing.assert_allclose(actual.astype(np.float32), expected.astype(np.float32), atol=atol, rtol=rtol)


@pytest.mark.cuda
def test_layer_norm_forward_matches_numpy_explicit_trailing_shape():
    dtype = thor.DataType.fp16
    epsilon = 1e-4
    values = (np.arange(2 * 2 * 3 * 4, dtype=np.float32).reshape(2, 2, 3, 4) - 11.5) / 3.0

    actual = _run_layer_norm_network(values, [2, 3, 4], dtype, normalized_shape=[3, 4], epsilon=epsilon)
    expected = _layer_norm_reference_for_dtype(values, [3, 4], epsilon, dtype)

    np.testing.assert_allclose(actual.astype(np.float32), expected.astype(np.float32), atol=1.5e-3, rtol=1.5e-3)


def _physical_ragged(values: np.ndarray, offsets: np.ndarray, dtype: thor.DataType = thor.DataType.fp32):
    return thor.physical.PhysicalRaggedTensor(
        _cpu_tensor(values, dtype),
        _cpu_tensor(offsets, thor.DataType.uint32),
    )


def test_layer_norm_accepts_ragged_tensor_and_preserves_partition():
    n = thor.Network("test_ragged_layer_norm_build")
    x = thor.layers.RaggedNetworkInput(
        n,
        "tokens",
        thor.DataType.fp32,
        [4],
        max_total_values=16,
        max_values_per_row=9,
        batch_size=3,
        offsets_data_type=thor.DataType.uint32,
    )

    ln = thor.layers.LayerNorm(n, x)
    y = ln.get_feature_output()

    assert ln.get_use_ragged() is True
    assert isinstance(y, thor.RaggedTensor)
    assert y.values.get_dimensions() == [16, 4]
    assert y.offsets == x.offsets
    assert y.max_values_per_row == 9
    assert ln.get_normalized_shape() == [4]

    arch = _only_layer_architecture(n, "layer_norm")
    assert arch["use_ragged"] is True
    assert arch["ragged_inputs"][0]["offsets"]["id"] == arch["ragged_outputs"][0]["offsets"]["id"]


def test_layer_norm_rejects_ragged_geometry_outside_single_channel_contract():
    n = thor.Network("test_ragged_layer_norm_bad_shape")
    x = thor.layers.RaggedNetworkInput(
        n,
        "tokens",
        thor.DataType.fp32,
        [4],
        max_total_values=16,
        batch_size=3,
        offsets_data_type=thor.DataType.uint32,
    )
    with pytest.raises((RuntimeError, ValueError), match="packed ragged row dimension"):
        thor.layers.LayerNorm(n, x, normalized_shape=[16, 4])

    matrix = thor.layers.RaggedNetworkInput(
        n,
        "matrix_tokens",
        thor.DataType.fp32,
        [2, 2],
        max_total_values=16,
        batch_size=3,
        offsets_data_type=thor.DataType.uint32,
    )
    with pytest.raises((RuntimeError, ValueError), match="exactly one non-zero trailing channel dimension"):
        thor.layers.LayerNorm(n, matrix, normalized_shape=[2])


@pytest.mark.cuda
def test_ragged_layer_norm_matches_active_prefix_and_ignores_poisoned_capacity():
    epsilon = 1e-4
    n = thor.Network("test_ragged_layer_norm_forward")
    x = thor.layers.RaggedNetworkInput(
        n,
        "tokens",
        thor.DataType.fp32,
        [4],
        max_total_values=16,
        batch_size=3,
        offsets_data_type=thor.DataType.uint32,
    )
    ln = thor.layers.LayerNorm(n, x, epsilon=epsilon)
    thor.layers.RaggedNetworkOutput(n, "output", ln.get_feature_output())

    active_rows = 11
    values = np.full((16, 4), np.float32(np.nan), dtype=np.float32)
    values[:active_rows] = (np.arange(active_rows * 4, dtype=np.float32).reshape(active_rows, 4) - 10.0) / 5.0
    offsets = np.array([0, 3, 3, active_rows], dtype=np.uint32)

    placed = n.place(3, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)
    result = placed.infer({"tokens": _physical_ragged(values, offsets)})["output"]
    actual = np.array(result.values.numpy(), copy=True)
    expected = _layer_norm_reference(values[:active_rows], [4], epsilon)

    assert np.array_equal(result.offsets.numpy(), offsets)
    np.testing.assert_allclose(actual[:active_rows], expected, rtol=2.5e-5, atol=2.5e-5)


def test_ragged_layer_norm_save_load_preserves_partition_metadata(tmp_path):
    name = "test_ragged_layer_norm_round_trip"
    n = thor.Network(name)
    x = thor.layers.RaggedNetworkInput(
        n,
        "tokens",
        thor.DataType.fp32,
        [4],
        max_total_values=16,
        max_values_per_row=9,
        batch_size=3,
        offsets_data_type=thor.DataType.uint32,
    )
    ln = thor.layers.LayerNorm(n, x, epsilon=1e-4)
    thor.layers.RaggedNetworkOutput(n, "output", ln.get_feature_output())

    save_dir = tmp_path / "ragged_layer_norm_model"
    n.save(str(save_dir), overwrite=False)

    loaded = thor.Network(name)
    loaded.load(str(save_dir))
    arch = _only_layer_architecture(loaded, "layer_norm")
    assert arch["use_ragged"] is True
    assert arch["ragged_inputs"][0]["offsets"]["id"] == arch["ragged_outputs"][0]["offsets"]["id"]
    assert arch["ragged_inputs"][0]["max_values_per_row"] == 9
