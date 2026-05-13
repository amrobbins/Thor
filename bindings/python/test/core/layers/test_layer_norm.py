import json

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
