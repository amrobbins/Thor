import pytest
import thor


def _network_input(n: thor.Network, shape, dtype=thor.DataType.fp32):
    return thor.layers.NetworkInput(n, "input", shape, dtype).get_feature_output()


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
    with pytest.raises(TypeError):
        thor.layers.Transpose(n, x, 123)
    with pytest.raises(TypeError):
        thor.layers.Transpose("not a network", x)
    with pytest.raises(TypeError):
        thor.layers.Transpose(n, "not a tensor")
