import json

import numpy as np
import pytest
import thor


def _net():
    return thor.Network("test_net_rms_norm")


def _input_tensor(n: thor.Network, dims, dtype=thor.DataType.fp16):
    ni = thor.layers.NetworkInput(n, "input", dims, dtype)
    return ni.get_feature_output()


def _only_layer_architecture(n: thor.Network, layer_type: str):
    layers = [layer for layer in json.loads(n.get_architecture_json())["layers"] if layer["layer_type"] == layer_type]
    assert len(layers) == 1
    return layers[0]


def _swish_epilogue():
    return thor.activations.Swish().to_expression(thor.layers.RMSNorm.epilogue_input())


def _cpu_tensor(values: np.ndarray, dtype: thor.DataType) -> thor.physical.PhysicalTensor:
    values = np.asarray(values, dtype=thor.physical.numpy_dtypes.from_thor(dtype), order="C")
    placement = thor.physical.Placement(thor.physical.DeviceType.cpu, 0)
    descriptor = thor.physical.PhysicalTensor.Descriptor(dtype, list(values.shape))
    tensor = thor.physical.PhysicalTensor(placement, descriptor)
    tensor.numpy()[...] = values
    return tensor


def _rms_norm_reference(x: np.ndarray, normalized_shape, epsilon: float) -> np.ndarray:
    rank = len(normalized_shape)
    axes = tuple(range(x.ndim - rank, x.ndim))
    x32 = x.astype(np.float32)
    square_mean = np.mean(x32 * x32, axis=axes, keepdims=True)
    return x32 / np.sqrt(square_mean + np.float32(epsilon))


def _rms_norm_reference_for_dtype(values: np.ndarray, normalized_shape, epsilon: float, dtype: thor.DataType) -> np.ndarray:
    # Match the values actually provided to Thor: _cpu_tensor stores the input
    # in the requested dtype before device execution.  For fp16/bf16 this can
    # slightly change the normalized values before cuDNN's fp32 accumulation.
    quantized_values = np.asarray(values, dtype=thor.physical.numpy_dtypes.from_thor(dtype))
    return _rms_norm_reference(quantized_values, normalized_shape, epsilon).astype(thor.physical.numpy_dtypes.from_thor(dtype))


def _run_rms_norm_network(values: np.ndarray, feature_dims, dtype: thor.DataType, *, normalized_shape=None, epsilon=1e-5) -> np.ndarray:
    dtype_name = str(dtype).split(".")[-1]
    n = thor.Network(f"test_net_rms_norm_numerical_{dtype_name}_{len(feature_dims)}d")
    x = _input_tensor(n, feature_dims, dtype)
    kwargs = {"epsilon": epsilon}
    if normalized_shape is not None:
        kwargs["normalized_shape"] = normalized_shape
    rn = thor.layers.RMSNorm(n, x, **kwargs)
    thor.layers.NetworkOutput(n, "output", rn.get_feature_output(), dtype)

    placed = n.place(
        values.shape[0],
        inference_only=True,
        forced_devices=[0],
        forced_num_stamps_per_gpu=1,
    )
    outputs = placed.infer({"input": _cpu_tensor(values, dtype)})
    assert set(outputs.keys()) == {"output"}
    return np.array(outputs["output"].numpy(), copy=True)


def test_rms_norm_constructs_default_last_dim_and_output_preserves_shape_dtype():
    n = _net()
    x = _input_tensor(n, [8, 16], thor.DataType.fp16)

    rn = thor.layers.RMSNorm(n, x)

    assert isinstance(rn, thor.layers.RMSNorm)
    assert rn.get_normalized_shape() == [16]
    assert rn.get_epsilon() == pytest.approx(1e-5)
    assert rn.get_parameter_data_type() == thor.DataType.fp32

    y = rn.get_feature_output()
    assert isinstance(y, thor.Tensor)
    assert y.get_dimensions() == x.get_dimensions()
    assert y.get_data_type() == x.get_data_type()


def test_rms_norm_constructs_explicit_trailing_shape_and_serializes_weights_only():
    n = _net()
    x = _input_tensor(n, [4, 8, 16], thor.DataType.bf16)

    rn = thor.layers.RMSNorm(n, x, normalized_shape=[8, 16], epsilon=1e-4)
    assert rn.get_normalized_shape() == [8, 16]
    assert rn.get_epsilon() == pytest.approx(1e-4)

    arch = _only_layer_architecture(n, "rms_norm")
    assert arch["normalized_shape"] == [8, 16]
    assert arch["epsilon"] == pytest.approx(1e-4)
    assert "weights" in arch["parameters"]
    assert "biases" not in arch["parameters"]
    assert arch["epilogue"] is None
    assert "fused_activation" not in arch


def test_rms_norm_rejects_bad_normalized_shape():
    n = _net()
    x = _input_tensor(n, [4, 8, 16], thor.DataType.fp16)

    with pytest.raises((RuntimeError, ValueError), match="normalizedShape"):
        thor.layers.RMSNorm(n, x, normalized_shape=[16, 8])

    with pytest.raises((RuntimeError, ValueError), match="normalizedShape"):
        thor.layers.RMSNorm(n, x, normalized_shape=[])


def test_rms_norm_rejects_bad_epsilon():
    n = _net()
    x = _input_tensor(n, [8, 16], thor.DataType.fp16)

    with pytest.raises(ValueError, match="epsilon must be > 0"):
        thor.layers.RMSNorm(n, x, epsilon=0.0)

    with pytest.raises(ValueError, match="epsilon must be > 0"):
        thor.layers.RMSNorm(n, x, epsilon=-1e-5)


def test_rms_norm_rejects_unsupported_dtypes():
    n = _net()
    x = _input_tensor(n, [8, 16], thor.DataType.fp16)

    with pytest.raises((RuntimeError, ValueError), match="fp32"):
        thor.layers.RMSNorm(n, x, parameter_data_type=thor.DataType.fp16)

    n2 = thor.Network("test_net_rms_norm_int")
    int_x = _input_tensor(n2, [8, 16], thor.DataType.int32)
    with pytest.raises((RuntimeError, ValueError), match="dtype"):
        thor.layers.RMSNorm(n2, int_x)


def test_rms_norm_rejects_wrong_types_and_arity():
    n = _net()
    x = _input_tensor(n, [8, 16], thor.DataType.fp16)

    with pytest.raises(TypeError):
        thor.layers.RMSNorm()

    with pytest.raises(TypeError):
        thor.layers.RMSNorm(n)

    with pytest.raises(TypeError):
        thor.layers.RMSNorm("not a network", x)

    with pytest.raises(TypeError):
        thor.layers.RMSNorm(n, "not a tensor")

    with pytest.raises(TypeError):
        thor.layers.RMSNorm(n, x, epsilon="1e-5")


@pytest.mark.cuda
@pytest.mark.parametrize("dtype,atol,rtol", [
    (thor.DataType.fp16, 1.5e-3, 1.5e-3),
    (thor.DataType.fp32, 2.5e-5, 2.5e-5),
])
def test_rms_norm_forward_matches_numpy_default_last_dim(dtype, atol, rtol):
    values = np.array(
        [
            [[-2.0, -1.0, 0.0, 1.0], [1.5, 2.0, 3.0, 4.0], [-3.0, 0.5, 2.5, 5.0]],
            [[0.25, -0.75, 1.25, 2.25], [4.0, 1.0, -2.0, -5.0], [3.5, 3.0, 2.5, 2.0]],
        ],
        dtype=np.float32,
    )

    actual = _run_rms_norm_network(values, [3, 4], dtype)
    expected = _rms_norm_reference_for_dtype(values, [4], 1e-5, dtype)

    np.testing.assert_allclose(actual.astype(np.float32), expected.astype(np.float32), atol=atol, rtol=rtol)


@pytest.mark.cuda
def test_rms_norm_forward_matches_numpy_explicit_trailing_shape():
    dtype = thor.DataType.fp16
    epsilon = 1e-4
    values = (np.arange(2 * 2 * 3 * 4, dtype=np.float32).reshape(2, 2, 3, 4) - 11.5) / 3.0

    actual = _run_rms_norm_network(values, [2, 3, 4], dtype, normalized_shape=[3, 4], epsilon=epsilon)
    expected = _rms_norm_reference_for_dtype(values, [3, 4], epsilon, dtype)

    np.testing.assert_allclose(actual.astype(np.float32), expected.astype(np.float32), atol=1.5e-3, rtol=1.5e-3)


def test_rms_norm_accepts_swish_epilogue_and_serializes_expression():
    n = _net()
    x = _input_tensor(n, [8, 16], thor.DataType.bf16)

    rn = thor.layers.RMSNorm(n, x, epilogue=_swish_epilogue())

    assert rn.get_parameter_data_type() == thor.DataType.fp32
    arch = _only_layer_architecture(n, "rms_norm")
    assert arch["epilogue"] is not None
    assert "fused_activation" not in arch


def test_rms_norm_accepts_bf16_weights_for_swish_epilogue_fusion_candidate():
    n = _net()
    x = _input_tensor(n, [8, 16], thor.DataType.bf16)

    rn = thor.layers.RMSNorm(n, x, parameter_data_type=thor.DataType.bf16, epilogue=_swish_epilogue())
    assert rn.get_parameter_data_type() == thor.DataType.bf16

    n2 = thor.Network("test_net_rms_norm_bf16_without_swish")
    x2 = _input_tensor(n2, [8, 16], thor.DataType.bf16)
    with pytest.raises((RuntimeError, ValueError), match="Swish epilogue"):
        thor.layers.RMSNorm(n2, x2, parameter_data_type=thor.DataType.bf16)

    n3 = thor.Network("test_net_rms_norm_bf16_weights_bad_input")
    x3 = _input_tensor(n3, [8, 16], thor.DataType.fp16)
    with pytest.raises((RuntimeError, ValueError), match="bf16 feature inputs"):
        thor.layers.RMSNorm(n3, x3, parameter_data_type=thor.DataType.bf16, epilogue=_swish_epilogue())


@pytest.mark.cuda
def test_rms_norm_bf16_swish_fusion_candidate_places_via_custom_layer():
    n = thor.Network("test_net_rms_norm_bf16_swish_places")
    x = _input_tensor(n, [3, 4], thor.DataType.bf16)
    rn = thor.layers.RMSNorm(n, x, parameter_data_type=thor.DataType.bf16, epilogue=_swish_epilogue())
    thor.layers.NetworkOutput(n, "output", rn.get_feature_output(), thor.DataType.bf16)

    n.place(
        2,
        inference_only=True,
        forced_devices=[0],
        forced_num_stamps_per_gpu=1,
    )


def test_rms_norm_rejects_bad_epilogue_type():
    n = _net()
    x = _input_tensor(n, [8, 16], thor.DataType.fp16)

    with pytest.raises(TypeError, match="epilogue"):
        thor.layers.RMSNorm(n, x, epilogue="swish")


def _physical_ragged(values: np.ndarray, offsets: np.ndarray, dtype: thor.DataType = thor.DataType.fp32):
    return thor.physical.PhysicalRaggedTensor(
        _cpu_tensor(values, dtype),
        _cpu_tensor(offsets, thor.DataType.uint32),
    )


def test_rms_norm_accepts_ragged_tensor_and_preserves_partition():
    n = thor.Network("test_ragged_rms_norm_build")
    x = thor.layers.RaggedNetworkInput(
        n,
        "tokens",
        thor.DataType.fp32,
        [4],
        max_total_values=66,
        batch_size=2,
        offsets_data_type=thor.DataType.uint32,
    )

    rn = thor.layers.RMSNorm(n, x)
    y = rn.get_feature_output()

    assert rn.get_use_ragged() is True
    assert isinstance(y, thor.RaggedTensor)
    assert y.values.get_dimensions() == [66, 4]
    assert y.offsets == x.offsets
    assert rn.get_normalized_shape() == [4]

    arch = _only_layer_architecture(n, "rms_norm")
    assert arch["use_ragged"] is True
    assert arch["ragged_inputs"][0]["offsets"]["id"] == arch["ragged_outputs"][0]["offsets"]["id"]


def test_rms_norm_rejects_ragged_normalized_shape_that_includes_packed_row_dimension():
    n = thor.Network("test_ragged_rms_norm_bad_shape")
    x = thor.layers.RaggedNetworkInput(
        n,
        "tokens",
        thor.DataType.fp32,
        [4],
        max_total_values=66,
        batch_size=2,
        offsets_data_type=thor.DataType.uint32,
    )

    with pytest.raises((RuntimeError, ValueError), match="packed ragged row dimension"):
        thor.layers.RMSNorm(n, x, normalized_shape=[66, 4])


@pytest.mark.cuda
@pytest.mark.parametrize("active_rows", [7, 9, 31, 33, 66])
def test_ragged_rms_norm_matches_dense_prefix_across_capacity_buckets(active_rows):
    epsilon = 1e-5
    n = thor.Network(f"test_ragged_rms_norm_bucket_{active_rows}")
    x = thor.layers.RaggedNetworkInput(
        n,
        "tokens",
        thor.DataType.fp32,
        [4],
        max_total_values=66,
        batch_size=2,
        offsets_data_type=thor.DataType.uint32,
    )
    rn = thor.layers.RMSNorm(n, x, epsilon=epsilon)
    thor.layers.RaggedNetworkOutput(n, "output", rn.get_feature_output())

    values = np.full((66, 4), np.float32(9999.0), dtype=np.float32)
    rows = np.arange(active_rows, dtype=np.float32)[:, None]
    cols = np.arange(4, dtype=np.float32)[None, :]
    values[:active_rows] = ((rows % 7.0) - 3.0) * np.float32(0.3) + (cols + 1.0) * np.float32(0.125)
    offsets = np.array([0, active_rows // 2, active_rows], dtype=np.uint32)

    placed = n.place(2, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)
    result = placed.infer({"tokens": _physical_ragged(values, offsets)})["output"]
    actual = np.array(result.values.numpy(), copy=True)
    expected = _rms_norm_reference(values[:active_rows], [4], epsilon)

    assert np.array_equal(result.offsets.numpy(), offsets)
    np.testing.assert_allclose(actual[:active_rows], expected, rtol=2.5e-5, atol=2.5e-5)
    np.testing.assert_array_equal(actual[active_rows:], np.zeros_like(actual[active_rows:]))


@pytest.mark.cuda
def test_ragged_rms_norm_save_load_preserves_execution(tmp_path):
    epsilon = 1e-4
    name = "test_ragged_rms_norm_save_load"
    n = thor.Network(name)
    x = thor.layers.RaggedNetworkInput(
        n,
        "tokens",
        thor.DataType.fp32,
        [4],
        max_total_values=16,
        batch_size=3,
        offsets_data_type=thor.DataType.uint32,
    )
    rn = thor.layers.RMSNorm(n, x, epsilon=epsilon)
    thor.layers.RaggedNetworkOutput(n, "output", rn.get_feature_output())

    save_dir = tmp_path / "model"
    n.save(str(save_dir), overwrite=False)
    loaded = thor.Network(name)
    loaded.load(str(save_dir))

    values = np.full((16, 4), np.float32(-7777.0), dtype=np.float32)
    active_rows = 11
    values[:active_rows] = (np.arange(active_rows * 4, dtype=np.float32).reshape(active_rows, 4) - 10.0) / 5.0
    offsets = np.array([0, 3, 3, active_rows], dtype=np.uint32)
    physical = _physical_ragged(values, offsets)

    placed = loaded.place(3, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)
    result = placed.infer({"tokens": physical})["output"]
    actual = np.array(result.values.numpy(), copy=True)
    expected = _rms_norm_reference(values[:active_rows], [4], epsilon)

    assert np.array_equal(result.offsets.numpy(), offsets)
    np.testing.assert_allclose(actual[:active_rows], expected, rtol=2.5e-5, atol=2.5e-5)
    np.testing.assert_array_equal(actual[active_rows:], np.zeros_like(actual[active_rows:]))
