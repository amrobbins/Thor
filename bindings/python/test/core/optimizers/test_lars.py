import json

import pytest
import thor


def _net():
    return thor.Network("test_net_lars")


def _only_fully_connected_layer(network: thor.Network):
    layers = [layer for layer in json.loads(network.get_architecture_json())["layers"] if layer["layer_type"] == "fully_connected"]
    assert len(layers) == 1
    return layers[0]


def test_lars_constructs_defaults():
    n = _net()
    opt = thor.optimizers.Lars(network=n)
    assert opt is not None
    assert isinstance(opt, thor.optimizers.Lars)

    arch = json.loads(n.get_architecture_json())["default_optimizer"]
    assert arch["optimizer_type"] == "lars"
    assert arch["alpha"] == pytest.approx(0.01)
    assert arch["momentum"] == pytest.approx(0.9)
    assert arch["weight_decay"] == pytest.approx(0.0)
    assert arch["trust_coefficient"] == pytest.approx(0.001)
    assert arch["epsilon"] == pytest.approx(1e-8)
    assert arch["use_nesterov"] is False


def test_lars_constructs_defaults_without_network():
    opt = thor.optimizers.Lars()
    assert opt is not None
    assert isinstance(opt, thor.optimizers.Lars)


def test_lars_constructs_custom_params():
    n = _net()
    opt = thor.optimizers.Lars(
        network=n,
        alpha=0.2,
        momentum=0.8,
        weight_decay=0.01,
        trust_coefficient=0.002,
        epsilon=1e-6,
        nesterov_momentum=True,
    )
    assert isinstance(opt, thor.optimizers.Lars)

    arch = json.loads(n.get_architecture_json())["default_optimizer"]
    assert arch["optimizer_type"] == "lars"
    assert arch["alpha"] == pytest.approx(0.2)
    assert arch["momentum"] == pytest.approx(0.8)
    assert arch["weight_decay"] == pytest.approx(0.01)
    assert arch["trust_coefficient"] == pytest.approx(0.002)
    assert arch["epsilon"] == pytest.approx(1e-6)
    assert arch["use_nesterov"] is True


def test_lars_rejects_invalid_values():
    n = _net()

    with pytest.raises(ValueError, match=r"alpha must be > 0"):
        thor.optimizers.Lars(network=n, alpha=0.0)
    with pytest.raises(ValueError, match=r"alpha must be > 0"):
        thor.optimizers.Lars(alpha=-1.0)

    with pytest.raises(ValueError, match=r"momentum must be >= 0"):
        thor.optimizers.Lars(momentum=-0.1)

    with pytest.raises(ValueError, match=r"weight_decay must be >= 0"):
        thor.optimizers.Lars(weight_decay=-0.01)

    with pytest.raises(ValueError, match=r"trust_coefficient must be > 0"):
        thor.optimizers.Lars(trust_coefficient=0.0)

    with pytest.raises(ValueError, match=r"epsilon must be > 0"):
        thor.optimizers.Lars(epsilon=0.0)


def test_lars_rejects_wrong_types_and_kwargs():
    n = _net()

    with pytest.raises(TypeError):
        thor.optimizers.Lars("not a network")

    with pytest.raises(TypeError):
        thor.optimizers.Lars(network=n, alpha="0.01")

    with pytest.raises(TypeError):
        thor.optimizers.Lars(0.01, 0.9, 0.0, 0.001, 1e-8, False, 123, network=n)

    with pytest.raises(TypeError):
        thor.optimizers.Lars(bogus=123, network=n)


def test_lars_is_optimizer_subclass_if_exposed():
    Optimizer = getattr(thor.optimizers, "Optimizer", None)
    if Optimizer is None:
        pytest.skip("thor.optimizers.Optimizer not exposed in Python")
    assert isinstance(thor.optimizers.Lars(), Optimizer)


def test_lars_multiple_optimizers_on_same_network_throws():
    n = _net()
    opt = thor.optimizers.Lars(network=n)
    assert opt is not None
    assert isinstance(opt, thor.optimizers.Lars)

    with pytest.raises(RuntimeError, match=r".*Multiple default optimizers.*"):
        thor.optimizers.Lars(network=n)


def test_lars_parameter_override_serializes_in_layer_architecture():
    n = thor.Network("test_net_lars_parameter_override")
    input_layer = thor.layers.NetworkInput(n, "input", [4], thor.DataType.fp32)
    weights_optimizer = thor.optimizers.Lars(alpha=0.02, momentum=0.8, weight_decay=0.01, trust_coefficient=0.002, epsilon=1e-6)
    biases_optimizer = thor.optimizers.Lars(alpha=0.03, momentum=0.7, weight_decay=0.02, trust_coefficient=0.003, epsilon=1e-5)

    thor.layers.FullyConnected(
        n,
        input_layer.get_feature_output(),
        3,
        True,
        activation=None,
        weights_optimizer=weights_optimizer,
        biases_optimizer=biases_optimizer,
    )

    fc_layer = _only_fully_connected_layer(n)
    weights_json = fc_layer["parameters"]["weights"]["optimizer_override"]
    biases_json = fc_layer["parameters"]["biases"]["optimizer_override"]

    assert weights_json["optimizer_type"] == "lars"
    assert weights_json["alpha"] == pytest.approx(0.02)
    assert weights_json["momentum"] == pytest.approx(0.8)
    assert weights_json["weight_decay"] == pytest.approx(0.01)
    assert weights_json["trust_coefficient"] == pytest.approx(0.002)
    assert weights_json["epsilon"] == pytest.approx(1e-6)

    assert biases_json["optimizer_type"] == "lars"
    assert biases_json["alpha"] == pytest.approx(0.03)
    assert biases_json["momentum"] == pytest.approx(0.7)
    assert biases_json["weight_decay"] == pytest.approx(0.02)
    assert biases_json["trust_coefficient"] == pytest.approx(0.003)
    assert biases_json["epsilon"] == pytest.approx(1e-5)
