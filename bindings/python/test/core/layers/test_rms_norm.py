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
