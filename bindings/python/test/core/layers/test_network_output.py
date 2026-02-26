import pytest
import thor


def _net():
    return thor.Network("test_net_network_output")


def _tensor_1d(size: int = 1, dtype=thor.DataType.fp32):
    # API tensor: dims + dtype
    return thor.Tensor([size], dtype)


def test_network_output_constructs_and_returns_feature_output():
    n = _net()

    # Use a NetworkInput to produce a connected tensor, which is the most realistic input_tensor.
    ni = thor.layers.NetworkInput(n, "input", [16], thor.DataType.fp16)
    x = ni.get_feature_output()

    no = thor.layers.NetworkOutput(n, "output", x, thor.DataType.fp16)

    assert no is not None
    assert isinstance(no, thor.layers.NetworkOutput)

    out = no.get_feature_output()
    assert out is not None
    assert isinstance(out, thor.Tensor)


def test_network_output_rejects_empty_name():
    n = _net()
    x = _tensor_1d(1, thor.DataType.fp16)

    with pytest.raises(ValueError, match=r"name must have non-zero length"):
        thor.layers.NetworkOutput(n, "", x, thor.DataType.fp16)


def test_network_output_rejects_wrong_types_and_arity():
    n = _net()
    x = _tensor_1d(1, thor.DataType.fp16)

    with pytest.raises(TypeError):
        thor.layers.NetworkOutput()  # missing args

    with pytest.raises(TypeError):
        thor.layers.NetworkOutput(n, "out", x)  # missing data_type

    with pytest.raises(TypeError):
        thor.layers.NetworkOutput(n, "out", x, thor.DataType.fp16, 123)  # extra arg

    with pytest.raises(TypeError):
        thor.layers.NetworkOutput("not a network", "out", x, thor.DataType.fp16)

    with pytest.raises(TypeError):
        thor.layers.NetworkOutput(n, 123, x, thor.DataType.fp16)  # name must be str

    with pytest.raises(TypeError):
        thor.layers.NetworkOutput(n, "out", "not a tensor", thor.DataType.fp16)

    with pytest.raises(TypeError):
        thor.layers.NetworkOutput(n, "out", x, "fp16")  # data_type must be enum
