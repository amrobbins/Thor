import json

import pytest
import thor


def _net():
    return thor.Network("test_net_adaptive_layer_norm")


def _input_tensor(n: thor.Network, name, dims, dtype=thor.DataType.fp16):
    ni = thor.layers.NetworkInput(n, name, dims, dtype)
    return ni.get_feature_output()


def _only_layer_architecture(n: thor.Network, layer_type: str):
    layers = [layer for layer in json.loads(n.get_architecture_json())["layers"] if layer["layer_type"] == layer_type]
    assert len(layers) == 1
    return layers[0]


def test_adaptive_layer_norm_constructs_default_last_dim_and_output_preserves_shape_dtype():
    n = _net()
    x = _input_tensor(n, "x", [8, 16], thor.DataType.fp16)
    scale = _input_tensor(n, "scale", [8, 16], thor.DataType.fp32)
    bias = _input_tensor(n, "bias", [8, 16], thor.DataType.fp32)

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
    scale = _input_tensor(n, "scale", [4, 8, 16], thor.DataType.fp32)
    bias = _input_tensor(n, "bias", [4, 8, 16], thor.DataType.fp32)

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
    scale = _input_tensor(n, "scale", [4, 8, 16], thor.DataType.fp32)
    bias = _input_tensor(n, "bias", [4, 8, 16], thor.DataType.fp32)

    with pytest.raises((RuntimeError, ValueError), match="normalizedShape"):
        thor.layers.AdaptiveLayerNorm(n, x, scale, bias, normalized_shape=[16, 8])

    with pytest.raises((RuntimeError, ValueError), match="normalizedShape"):
        thor.layers.AdaptiveLayerNorm(n, x, scale, bias, normalized_shape=[])


def test_adaptive_layer_norm_rejects_bad_epsilon():
    n = _net()
    x = _input_tensor(n, "x", [8, 16], thor.DataType.fp16)
    scale = _input_tensor(n, "scale", [8, 16], thor.DataType.fp32)
    bias = _input_tensor(n, "bias", [8, 16], thor.DataType.fp32)

    with pytest.raises(ValueError, match="epsilon must be > 0"):
        thor.layers.AdaptiveLayerNorm(n, x, scale, bias, epsilon=0.0)

    with pytest.raises(ValueError, match="epsilon must be > 0"):
        thor.layers.AdaptiveLayerNorm(n, x, scale, bias, epsilon=-1e-5)


def test_adaptive_layer_norm_rejects_unsupported_dtypes_and_shapes():
    n = _net()
    x = _input_tensor(n, "x", [8, 16], thor.DataType.fp16)
    scale = _input_tensor(n, "scale", [8, 16], thor.DataType.fp32)
    bias = _input_tensor(n, "bias", [8, 16], thor.DataType.fp32)

    bad_scale = _input_tensor(n, "bad_scale", [8, 16], thor.DataType.fp16)
    with pytest.raises((RuntimeError, ValueError), match="fp32"):
        thor.layers.AdaptiveLayerNorm(n, x, bad_scale, bias)

    bad_bias = _input_tensor(n, "bad_bias", [16], thor.DataType.fp32)
    with pytest.raises((RuntimeError, ValueError), match="dimensions"):
        thor.layers.AdaptiveLayerNorm(n, x, scale, bad_bias)

    n2 = thor.Network("test_net_adaptive_layer_norm_int")
    int_x = _input_tensor(n2, "x", [8, 16], thor.DataType.int32)
    int_scale = _input_tensor(n2, "scale", [8, 16], thor.DataType.fp32)
    int_bias = _input_tensor(n2, "bias", [8, 16], thor.DataType.fp32)
    with pytest.raises((RuntimeError, ValueError), match="dtype"):
        thor.layers.AdaptiveLayerNorm(n2, int_x, int_scale, int_bias)


def test_adaptive_layer_norm_rejects_wrong_types_and_arity():
    n = _net()
    x = _input_tensor(n, "x", [8, 16], thor.DataType.fp16)
    scale = _input_tensor(n, "scale", [8, 16], thor.DataType.fp32)
    bias = _input_tensor(n, "bias", [8, 16], thor.DataType.fp32)

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
