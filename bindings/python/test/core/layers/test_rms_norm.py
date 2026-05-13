import json

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


def test_rms_norm_accepts_fused_swish_activation_and_serializes():
    n = _net()
    x = _input_tensor(n, [8, 16], thor.DataType.bf16)

    rn = thor.layers.RMSNorm(n, x, fused_activation="swish")

    assert rn.get_fused_activation() == "swish"
    assert rn.get_parameter_data_type() == thor.DataType.bf16
    arch = _only_layer_architecture(n, "rms_norm")
    assert arch["fused_activation"] == "swish"


def test_rms_norm_accepts_silu_alias_and_rejects_unknown_fused_activation():
    n = _net()
    x = _input_tensor(n, [8, 16], thor.DataType.bf16)

    rn = thor.layers.RMSNorm(n, x, fused_activation="silu")
    assert rn.get_fused_activation() == "swish"

    n2 = thor.Network("test_net_rms_norm_bad_fused_activation")
    x2 = _input_tensor(n2, [8, 16], thor.DataType.fp16)
    with pytest.raises(ValueError, match="fused activation"):
        thor.layers.RMSNorm(n2, x2, fused_activation="relu")

    n3 = thor.Network("test_net_rms_norm_fused_fp16")
    x3 = _input_tensor(n3, [8, 16], thor.DataType.fp16)
    with pytest.raises((RuntimeError, ValueError), match="bf16"):
        thor.layers.RMSNorm(n3, x3, fused_activation="swish")
