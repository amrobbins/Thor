import json

import pytest
import thor


def _net():
    return thor.Network("test_net_instance_norm")


def _input_tensor(n: thor.Network, dims, dtype=thor.DataType.fp16):
    ni = thor.layers.NetworkInput(n, "input", dims, dtype)
    return ni.get_feature_output()


def _only_layer_architecture(n: thor.Network, layer_type: str):
    layers = [layer for layer in json.loads(n.get_architecture_json())["layers"] if layer["layer_type"] == layer_type]
    assert len(layers) == 1
    return layers[0]


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
    x = _input_tensor(n, [4, 32], thor.DataType.bf16)

    inn = thor.layers.InstanceNorm(n, x, epsilon=1e-4)
    assert inn.get_channel_count() == 4
    assert inn.get_epsilon() == pytest.approx(1e-4)

    arch = _only_layer_architecture(n, "instance_norm")
    assert arch["channel_count"] == 4
    assert arch["epsilon"] == pytest.approx(1e-4)
    assert "weights" in arch["parameters"]
    assert "biases" in arch["parameters"]


def test_instance_norm_rejects_bad_shape():
    n = _net()
    rank_one = _input_tensor(n, [8], thor.DataType.fp16)

    with pytest.raises((RuntimeError, ValueError), match="InstanceNorm"):
        thor.layers.InstanceNorm(n, rank_one)


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
