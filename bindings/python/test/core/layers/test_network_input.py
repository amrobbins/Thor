import pytest
import thor


def _net():
    return thor.Network("test_net_network_input")


def test_network_input_constructs_and_returns_feature_output():
    n = _net()
    ni = thor.layers.NetworkInput(n, "input", [16], thor.DataType.fp16)

    assert ni is not None
    assert isinstance(ni, thor.layers.NetworkInput)

    out = ni.get_feature_output()
    assert out is not None
    assert isinstance(out, thor.Tensor)


def test_network_input_rejects_empty_name():
    n = _net()
    with pytest.raises(ValueError, match=r"name must have non-zero length"):
        thor.layers.NetworkInput(n, "", [16], thor.DataType.fp16)


def test_network_input_rejects_empty_dimensions():
    n = _net()
    with pytest.raises(ValueError, match=r"dimensions must be non-zero"):
        thor.layers.NetworkInput(n, "input", [], thor.DataType.fp16)


def test_network_input_accepts_multi_dimensional_shape():
    n = _net()
    ni = thor.layers.NetworkInput(n, "img", [3, 224, 224], thor.DataType.fp16)
    assert isinstance(ni, thor.layers.NetworkInput)
    out = ni.get_feature_output()
    assert isinstance(out, thor.Tensor)


def test_network_input_rejects_wrong_types_and_arity():
    n = _net()

    with pytest.raises(TypeError):
        thor.layers.NetworkInput()  # missing args

    with pytest.raises(TypeError):
        thor.layers.NetworkInput(n, "input", [16])  # missing data_type

    with pytest.raises(TypeError):
        thor.layers.NetworkInput(n, "input", [16], thor.DataType.fp16, 123)  # extra arg

    with pytest.raises(TypeError):
        thor.layers.NetworkInput("not a network", "input", [16], thor.DataType.fp16)

    with pytest.raises(TypeError):
        thor.layers.NetworkInput(n, 123, [16], thor.DataType.fp16)  # name must be str

    with pytest.raises(TypeError):
        thor.layers.NetworkInput(n, "input", "not a list", thor.DataType.fp16)

    with pytest.raises(TypeError):
        thor.layers.NetworkInput(n, "input", [16], "fp16")  # data_type must be enum

    with pytest.raises(TypeError):
        thor.layers.NetworkInput(n, "input", [16.5], thor.DataType.fp16)
