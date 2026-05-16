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
