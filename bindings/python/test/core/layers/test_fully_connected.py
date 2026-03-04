import pytest

import thor


def _net():
    return thor.Network("test_net_fully_connected")


def _input_tensor(n: thor.Network, in_features: int, dtype=thor.DataType.fp16):
    ni = thor.layers.NetworkInput(n, "input", [in_features], dtype)
    return ni.get_feature_output()


def test_fully_connected_constructs_defaults_and_output_shape_dtype():
    n = _net()
    x = _input_tensor(n, 32, thor.DataType.fp16)

    fc = thor.layers.FullyConnected(n, x, 64, True)
    assert fc is not None
    assert isinstance(fc, thor.layers.FullyConnected)

    y = fc.get_feature_output()
    assert y is not None
    assert isinstance(y, thor.Tensor)

    # API expectation: output is 1D [num_output_features]
    assert y.get_dimensions() == [64]
    # Typically FC preserves dtype of feature_input at API level; if your builder forces fp32, change this.
    assert y.get_data_type() == x.get_data_type()


def test_fully_connected_constructs_no_activation_when_none():
    n = _net()
    x = _input_tensor(n, 16, thor.DataType.fp16)

    fc = thor.layers.FullyConnected(
        n,
        x,
        8,
        True,
        activation=None,
    )
    assert isinstance(fc, thor.layers.FullyConnected)
    y = fc.get_feature_output()
    assert y.get_dimensions() == [8]


def test_fully_connected_constructs_with_activation_and_initializers():
    n = _net()
    x = _input_tensor(n, 16, thor.DataType.fp16)

    act = thor.activations.Elu(1.0) if hasattr(thor.activations, "Elu") else None
    winit = thor.initializers.Glorot(thor.initializers.Glorot.Mode.UNIFORM)
    binit = thor.initializers.Glorot(thor.initializers.Glorot.Mode.NORMAL)

    fc = thor.layers.FullyConnected(
        n,
        x,
        10,
        True,
        activation=act,
        weights_initializer=winit,
        biases_initializer=binit,
    )
    assert isinstance(fc, thor.layers.FullyConnected)
    y = fc.get_feature_output()
    assert y.get_dimensions() == [10]
    assert y.get_data_type() == x.get_data_type()


def test_fully_connected_rejects_num_output_features_zero():
    n = _net()
    x = _input_tensor(n, 16, thor.DataType.fp16)

    # Requires the binding-side check (recommended). If you didn't add it,
    # this might assert/crash, so keep this test only after adding the check.
    with pytest.raises(ValueError, match=r"num_output_features must be > 0"):
        thor.layers.FullyConnected(n, x, 0, True)


def test_fully_connected_rejects_wrong_types_and_arity():
    n = _net()
    x = _input_tensor(n, 16, thor.DataType.fp16)

    with pytest.raises(TypeError):
        thor.layers.FullyConnected()  # missing args

    with pytest.raises(TypeError):
        thor.layers.FullyConnected(n, x)  # missing num_output_features

    with pytest.raises(TypeError):
        thor.layers.FullyConnected(n, x, 8, True, 123)  # activation wrong type

    with pytest.raises(TypeError):
        thor.layers.FullyConnected("not a network", x, 8, True)

    with pytest.raises(TypeError):
        thor.layers.FullyConnected(n, "not a tensor", 8, True)

    with pytest.raises(TypeError):
        thor.layers.FullyConnected(n, x, 8, "True")
