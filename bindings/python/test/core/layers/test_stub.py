import pytest
import thor


def _net():
    return thor.Network("test_net_stub")


def test_stub_constructs_with_network_input_tensor():
    n = _net()
    ni = thor.layers.NetworkInput(n, "input", [16], thor.DataType.fp16)
    x = ni.get_feature_output()

    layer = thor.layers.Stub(n, x)
    assert layer is not None
    assert isinstance(layer, thor.layers.Stub)


def test_stub_rejects_wrong_types_and_arity():
    n = _net()
    ni = thor.layers.NetworkInput(n, "input", [16], thor.DataType.fp16)
    x = ni.get_feature_output()

    with pytest.raises(TypeError):
        thor.layers.Stub()  # missing args

    with pytest.raises(TypeError):
        thor.layers.Stub(n)  # missing input_tensor

    with pytest.raises(TypeError):
        thor.layers.Stub(n, x, 123)  # extra arg

    with pytest.raises(TypeError):
        thor.layers.Stub("not a network", x)

    with pytest.raises(TypeError):
        thor.layers.Stub(n, "not a tensor")
