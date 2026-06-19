import json

import pytest

import thor


def _net():
    return thor.Network("test_net_fully_connected")


def _input_tensor(n: thor.Network, in_features: int, dtype=thor.DataType.fp16):
    ni = thor.layers.NetworkInput(n, "input", [in_features], dtype)
    return ni.get_feature_output()


def _only_layer_architecture(n: thor.Network, layer_type: str):
    layers = [layer for layer in json.loads(n.get_architecture_json())["layers"] if layer["layer_type"] == layer_type]
    assert len(layers) == 1
    return layers[0]


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

    fc_arch = _only_layer_architecture(n, "fully_connected")
    assert fc_arch["activation"]["layer_type"] == "gelu"


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

    fc_arch = _only_layer_architecture(n, "fully_connected")
    assert fc_arch["activation"] is None




def test_fully_connected_can_preserve_prefix_dimensions_for_tokenwise_projection():
    n = thor.Network("test_net_fully_connected_tokenwise")
    x_in = thor.layers.NetworkInput(n, "tokens", [5, 16], thor.DataType.fp16)

    fc = thor.layers.FullyConnected(
        n,
        x_in.get_feature_output(),
        8,
        True,
        activation=None,
        preserve_prefix_dimensions=True,
    )

    y = fc.get_feature_output()
    assert y.get_dimensions() == [5, 8]
    assert y.get_data_type() == thor.DataType.fp16

    arch = _only_layer_architecture(n, "fully_connected")
    assert arch["preserve_input_prefix_dimensions"] is True
    assert arch["outputs"][0]["dimensions"] == [5, 8]


def test_fully_connected_default_flattens_prefix_dimensions():
    n = thor.Network("test_net_fully_connected_flatten_prefix")
    x_in = thor.layers.NetworkInput(n, "tokens", [5, 16], thor.DataType.fp16)

    fc = thor.layers.FullyConnected(
        n,
        x_in.get_feature_output(),
        8,
        True,
        activation=None,
    )

    assert fc.get_feature_output().get_dimensions() == [8]
    arch = _only_layer_architecture(n, "fully_connected")
    assert arch["preserve_input_prefix_dimensions"] is False

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


def test_fully_connected_accepts_epilogue_expression_and_serializes_it():
    n = thor.Network("test_net_fully_connected_epilogue")
    x = _input_tensor(n, 16, thor.DataType.fp16)

    epilogue_input = thor.layers.FullyConnected.epilogue_input(
        output_dtype=thor.DataType.fp32,
        compute_dtype=thor.DataType.fp32,
    )
    epilogue = epilogue_input * 2.0 + 1.0

    fc = thor.layers.FullyConnected(
        n,
        x,
        8,
        True,
        activation=None,
        epilogue=epilogue,
    )

    assert fc.get_feature_output().get_dimensions() == [8]
    arch = _only_layer_architecture(n, "fully_connected")
    assert arch["activation"] is None
    assert arch["epilogue"] is not None
    assert arch["epilogue"]["expected_input_names"] == ["__fully_connected_epilogue_input"]
    assert arch["epilogue"]["expected_output_names"] == ["__fully_connected_epilogue_output"]


def test_fully_connected_rejects_wrong_epilogue_type():
    n = _net()
    x = _input_tensor(n, 16, thor.DataType.fp16)

    with pytest.raises(TypeError, match="epilogue must be"):
        thor.layers.FullyConnected(n, x, 8, True, epilogue=123)


def test_fully_connected_serializes_weight_constraints():
    n = _net()
    x = _input_tensor(n, 16, thor.DataType.fp32)

    fc = thor.layers.FullyConnected(
        n,
        x,
        8,
        True,
        activation=None,
        weights_constraints=thor.NonNegativeParameterConstraint(),
    )

    assert isinstance(fc, thor.layers.FullyConnected)
    arch = _only_layer_architecture(n, "fully_connected")
    weight_constraints = arch["parameters"]["weights"].get("constraints", [])
    assert len(weight_constraints) == 1
    assert weight_constraints[0]["constraint_type"] == "non_negative"


def test_fully_connected_rejects_invalid_weight_constraint():
    n = _net()
    x = _input_tensor(n, 16, thor.DataType.fp32)

    with pytest.raises(TypeError, match="weights_constraints"):
        thor.layers.FullyConnected(
            n,
            x,
            8,
            True,
            activation=None,
            weights_constraints=123,
        )
