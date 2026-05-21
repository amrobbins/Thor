import json

import pytest
import thor


def _network_input(n: thor.Network, shape, dtype=thor.DataType.fp32):
    return thor.layers.NetworkInput(n, "input", shape, dtype).get_feature_output()


def _only_layer_architecture(n: thor.Network, layer_type: str):
    layers = [layer for layer in json.loads(n.get_architecture_json())["layers"] if layer["layer_type"] == layer_type]
    assert len(layers) == 1
    return layers[0]


def test_transpose_constructs_and_swaps_trailing_feature_dimensions():
    n = thor.Network("test_transpose_constructs_and_swaps_trailing_feature_dimensions")
    x = _network_input(n, [3, 4, 5], thor.DataType.fp16)

    layer = thor.layers.Transpose(n, x)

    assert isinstance(layer, thor.layers.Transpose)
    y = layer.get_feature_output()
    assert y.get_data_type() == thor.DataType.fp16
    assert y.get_dimensions() == [3, 5, 4]


def test_transpose_rejects_rank1_feature_input():
    n = thor.Network("test_transpose_rejects_rank1_feature_input")
    x = _network_input(n, [8], thor.DataType.fp32)

    with pytest.raises(ValueError, match="rank >= 2"):
        thor.layers.Transpose(n, x)


def test_transpose_constructor_argument_validation():
    n = thor.Network("test_transpose_constructor_argument_validation")
    x = _network_input(n, [3, 4], thor.DataType.fp32)

    with pytest.raises(TypeError):
        thor.layers.Transpose()
    with pytest.raises(TypeError):
        thor.layers.Transpose(n)
    with pytest.raises((TypeError, RuntimeError, ValueError)):
        thor.layers.Transpose(n, x, 123)
    with pytest.raises(TypeError):
        thor.layers.Transpose(n, x, thor.DataType.fp32, None, 123)
    with pytest.raises(TypeError):
        thor.layers.Transpose("not a network", x)
    with pytest.raises(TypeError):
        thor.layers.Transpose(n, "not a tensor")


def test_transpose_accepts_optional_output_dtype_and_serializes_it():
    n = thor.Network("test_transpose_accepts_optional_output_dtype_and_serializes_it")
    x = _network_input(n, [3, 4, 5], thor.DataType.fp16)

    layer = thor.layers.Transpose(n, x, output_dtype=thor.DataType.fp32)

    assert layer.get_output_data_type() == thor.DataType.fp32
    y = layer.get_feature_output()
    assert y.get_data_type() == thor.DataType.fp32
    assert y.get_dimensions() == [3, 5, 4]

    arch = _only_layer_architecture(n, "transpose")
    assert arch["output_data_type"] == "fp32"
    assert arch["epilogue"] is None


def test_transpose_accepts_epilogue_and_serializes_expression():
    n = thor.Network("test_transpose_accepts_epilogue_and_serializes_expression")
    x = _network_input(n, [3, 4, 5], thor.DataType.fp16)

    epilogue_input = thor.layers.Transpose.epilogue_input(
        output_dtype=thor.DataType.fp32,
        compute_dtype=thor.DataType.fp32,
    )
    epilogue = epilogue_input * 2.0 + 1.0

    layer = thor.layers.Transpose(n, x, output_dtype=thor.DataType.fp32, epilogue=epilogue)

    assert layer.get_feature_output().get_data_type() == thor.DataType.fp32
    assert layer.get_feature_output().get_dimensions() == [3, 5, 4]

    arch = _only_layer_architecture(n, "transpose")
    assert arch["output_data_type"] == "fp32"
    assert arch["epilogue"] is not None
    assert arch["epilogue"]["expected_input_names"] == ["__transpose_epilogue_input"]
    assert arch["epilogue"]["expected_output_names"] == ["__transpose_epilogue_output"]


def test_transpose_rejects_wrong_epilogue_type():
    n = thor.Network("test_transpose_rejects_wrong_epilogue_type")
    x = _network_input(n, [3, 4], thor.DataType.fp32)

    with pytest.raises(TypeError, match="epilogue must be"):
        thor.layers.Transpose(n, x, epilogue=123)
