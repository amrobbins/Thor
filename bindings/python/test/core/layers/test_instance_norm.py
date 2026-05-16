import json

import numpy as np
import pytest
import thor
from thor.physical import numpy_dtypes


def _net():
    return thor.Network("test_net_instance_norm")


def _input_tensor(n: thor.Network, dims, dtype=thor.DataType.fp16):
    ni = thor.layers.NetworkInput(n, "input", dims, dtype)
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


def _instance_norm_reference(x: np.ndarray, epsilon: float) -> np.ndarray:
    x32 = x.astype(np.float32)
    axes = tuple(range(2, x32.ndim))
    mean = np.mean(x32, axis=axes, keepdims=True)
    variance = np.mean((x32 - mean) * (x32 - mean), axis=axes, keepdims=True)
    return (x32 - mean) / np.sqrt(variance + np.float32(epsilon))


def _instance_norm_reference_for_dtype(values: np.ndarray, epsilon: float, dtype: thor.DataType) -> np.ndarray:
    # Match the values actually provided to Thor: _cpu_tensor stores the input
    # in the requested dtype before device execution.  For fp16 this can
    # slightly change the normalized values before cuDNN's fp32 accumulation.
    quantized_values = np.asarray(values, dtype=numpy_dtypes.from_thor(dtype))
    return _instance_norm_reference(quantized_values, epsilon).astype(numpy_dtypes.from_thor(dtype))


def _run_instance_norm_network(values: np.ndarray, feature_dims, dtype: thor.DataType, *, epsilon=1e-5) -> np.ndarray:
    dtype_name = str(dtype).split(".")[-1]
    n = thor.Network(f"test_net_instance_norm_numerical_{dtype_name}_{len(feature_dims)}d")
    x = _input_tensor(n, feature_dims, dtype)
    inn = thor.layers.InstanceNorm(n, x, epsilon=epsilon)
    thor.layers.NetworkOutput(n, "output", inn.get_feature_output(), dtype)

    placed = n.place(
        values.shape[0],
        inference_only=True,
        forced_devices=[0],
        forced_num_stamps_per_gpu=1,
    )
    outputs = placed.infer({"input": _cpu_tensor(values, dtype)})
    assert set(outputs.keys()) == {"output"}
    return np.array(outputs["output"].numpy(), copy=True)


def test_instance_norm_constructs_and_output_preserves_shape_dtype():
    n = _net()
    x = _input_tensor(n, [8, 16, 16], thor.DataType.fp16)

    inn = thor.layers.InstanceNorm(n, x)

    assert isinstance(inn, thor.layers.InstanceNorm)
    assert inn.get_channel_count() == 8
    assert inn.get_epsilon() == pytest.approx(1e-5)
    assert inn.get_parameter_data_type() == thor.DataType.fp32

    y = inn.get_feature_output()
    assert isinstance(y, thor.Tensor)
    assert y.get_dimensions() == x.get_dimensions()
    assert y.get_data_type() == x.get_data_type()


def test_instance_norm_accepts_1d_spatial_input_and_serializes():
    n = _net()
    x = _input_tensor(n, [8, 32], thor.DataType.bf16)

    inn = thor.layers.InstanceNorm(n, x, epsilon=1e-4)
    assert inn.get_channel_count() == 8
    assert inn.get_epsilon() == pytest.approx(1e-4)

    arch = _only_layer_architecture(n, "instance_norm")
    assert arch["channel_count"] == 8
    assert arch["epsilon"] == pytest.approx(1e-4)
    assert "weights" in arch["parameters"]
    assert "biases" in arch["parameters"]


def test_instance_norm_rejects_bad_shape():
    n = _net()
    rank_one = _input_tensor(n, [8], thor.DataType.fp16)

    with pytest.raises((RuntimeError, ValueError), match="InstanceNorm"):
        thor.layers.InstanceNorm(n, rank_one)


def test_instance_norm_rejects_reduced_precision_channel_count_that_cudnn_primary_engines_do_not_support():
    n = _net()
    fp16_bad_channels = _input_tensor(n, [4, 32], thor.DataType.fp16)

    with pytest.raises((RuntimeError, ValueError), match="multiple of 8"):
        thor.layers.InstanceNorm(n, fp16_bad_channels)

    n2 = thor.Network("test_net_instance_norm_bf16_bad_channels")
    bf16_bad_channels = _input_tensor(n2, [4, 32], thor.DataType.bf16)
    with pytest.raises((RuntimeError, ValueError), match="multiple of 8"):
        thor.layers.InstanceNorm(n2, bf16_bad_channels)


def test_instance_norm_rejects_bad_epsilon():
    n = _net()
    x = _input_tensor(n, [8, 16, 16], thor.DataType.fp16)

    with pytest.raises(ValueError, match="epsilon must be > 0"):
        thor.layers.InstanceNorm(n, x, epsilon=0.0)

    with pytest.raises(ValueError, match="epsilon must be > 0"):
        thor.layers.InstanceNorm(n, x, epsilon=-1e-5)


def test_instance_norm_rejects_unsupported_dtypes():
    n = _net()
    x = _input_tensor(n, [8, 16, 16], thor.DataType.fp16)

    with pytest.raises((RuntimeError, ValueError), match="fp32"):
        thor.layers.InstanceNorm(n, x, parameter_data_type=thor.DataType.fp16)

    n2 = thor.Network("test_net_instance_norm_int")
    int_x = _input_tensor(n2, [8, 16, 16], thor.DataType.int32)
    with pytest.raises((RuntimeError, ValueError), match="dtype"):
        thor.layers.InstanceNorm(n2, int_x)


def test_instance_norm_rejects_wrong_types_and_arity():
    n = _net()
    x = _input_tensor(n, [8, 16, 16], thor.DataType.fp16)

    with pytest.raises(TypeError):
        thor.layers.InstanceNorm()

    with pytest.raises(TypeError):
        thor.layers.InstanceNorm(n)

    with pytest.raises(TypeError):
        thor.layers.InstanceNorm("not a network", x)

    with pytest.raises(TypeError):
        thor.layers.InstanceNorm(n, "not a tensor")

    with pytest.raises(TypeError):
        thor.layers.InstanceNorm(n, x, epsilon="1e-5")


@pytest.mark.cuda
@pytest.mark.parametrize("dtype,atol,rtol", [
    (thor.DataType.fp16, 2.0e-3, 2.0e-3),
    (thor.DataType.fp32, 2.5e-5, 2.5e-5),
])
def test_instance_norm_forward_matches_numpy_2d_spatial(dtype, atol, rtol):
    # cuDNN's primary InstanceNorm engines vectorize across channels and
    # require channel_count to satisfy their load alignment constraints.  Use
    # eight channels so the fp16 path exercises the real cuDNN graph instead of
    # depending on unsupported tiny channel-count cases.
    values = (
        np.arange(2 * 8 * 2 * 4, dtype=np.float32).reshape(2, 8, 2, 4) % 17 - 8.0
    ) / 3.0

    actual = _run_instance_norm_network(values, [8, 2, 4], dtype)
    expected = _instance_norm_reference_for_dtype(values, 1e-5, dtype)

    np.testing.assert_allclose(actual.astype(np.float32), expected.astype(np.float32), atol=atol, rtol=rtol)


@pytest.mark.cuda
def test_instance_norm_forward_matches_numpy_1d_spatial():
    dtype = thor.DataType.fp16
    epsilon = 1e-4
    values = (
        np.arange(3 * 8 * 8, dtype=np.float32).reshape(3, 8, 8) % 19 - 9.0
    ) / 4.0

    actual = _run_instance_norm_network(values, [8, 8], dtype, epsilon=epsilon)
    expected = _instance_norm_reference_for_dtype(values, epsilon, dtype)

    np.testing.assert_allclose(actual.astype(np.float32), expected.astype(np.float32), atol=2.0e-3, rtol=2.0e-3)

