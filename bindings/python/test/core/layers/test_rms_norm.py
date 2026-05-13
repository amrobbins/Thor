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


def _swish_epilogue():
    return thor.activations.Swish().to_expression(thor.layers.RMSNorm.epilogue_input())


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


def test_rms_norm_rejects_bad_epilogue_type():
    n = _net()
    x = _input_tensor(n, [8, 16], thor.DataType.fp16)

    with pytest.raises(TypeError, match="epilogue"):
        thor.layers.RMSNorm(n, x, epilogue="swish")
