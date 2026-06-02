import json

import pytest
import thor


def _net():
    return thor.Network("test_net_asgd")


def _only_fully_connected_layer(network: thor.Network):
    layers = [layer for layer in json.loads(network.get_architecture_json())["layers"] if layer["layer_type"] == "fully_connected"]
    assert len(layers) == 1
    return layers[0]


def test_asgd_constructs_defaults():
    n = _net()
    opt = thor.optimizers.ASGD(network=n)
    assert opt is not None
    assert isinstance(opt, thor.optimizers.ASGD)


def test_asgd_constructs_defaults_without_network():
    opt = thor.optimizers.ASGD()
    assert opt is not None
    assert isinstance(opt, thor.optimizers.ASGD)


def test_asgd_constructs_custom_params():
    n = _net()
    opt = thor.optimizers.ASGD(network=n, alpha=0.1, lambd=0.01, power=0.5, t0=10.0, weight_decay=0.03)
    assert isinstance(opt, thor.optimizers.ASGD)


def test_asgd_rejects_non_positive_alpha():
    n = _net()
    with pytest.raises(ValueError, match=r"alpha must be > 0"):
        thor.optimizers.ASGD(network=n, alpha=0.0)
    with pytest.raises(ValueError, match=r"alpha must be > 0"):
        thor.optimizers.ASGD(alpha=-1.0)


def test_asgd_rejects_invalid_lambd():
    n = _net()
    with pytest.raises(ValueError, match=r"lambd must be >= 0"):
        thor.optimizers.ASGD(network=n, lambd=-0.1)


def test_asgd_rejects_invalid_power():
    n = _net()
    with pytest.raises(ValueError, match=r"power must be >= 0"):
        thor.optimizers.ASGD(network=n, power=-0.1)


def test_asgd_rejects_invalid_t0():
    n = _net()
    with pytest.raises(ValueError, match=r"t0 must be >= 1"):
        thor.optimizers.ASGD(network=n, t0=0.0)


def test_asgd_rejects_invalid_weight_decay():
    n = _net()
    with pytest.raises(ValueError, match=r"weight_decay must be >= 0"):
        thor.optimizers.ASGD(network=n, weight_decay=-0.1)


def test_asgd_rejects_wrong_types():
    n = _net()

    with pytest.raises(TypeError):
        thor.optimizers.ASGD("not a network")

    with pytest.raises(TypeError):
        thor.optimizers.ASGD(network=n, alpha="0.1")

    with pytest.raises(TypeError):
        thor.optimizers.ASGD(network=n, lambd="0.01")

    with pytest.raises(TypeError):
        thor.optimizers.ASGD(network=n, power="0.75")

    with pytest.raises(TypeError):
        thor.optimizers.ASGD(network=n, t0="1000")

    with pytest.raises(TypeError):
        thor.optimizers.ASGD(network=n, weight_decay="0.0")


def test_asgd_rejects_wrong_arity_and_kwargs():
    n = _net()

    with pytest.raises(TypeError):
        thor.optimizers.ASGD(0.01, 1e-4, 0.75, 1e6, 0.0, 123, network=n)  # extra positional

    with pytest.raises(TypeError):
        thor.optimizers.ASGD(network=n, bogus=123)  # wrong kw

    with pytest.raises(TypeError):
        thor.optimizers.ASGD(network=n, alpha=0.01, extra=123)  # extra kw


def test_asgd_is_optimizer_subclass_if_exposed():
    Optimizer = getattr(thor.optimizers, "Optimizer", None)
    if Optimizer is None:
        pytest.skip("thor.optimizers.Optimizer not exposed in Python")
    assert isinstance(thor.optimizers.ASGD(), Optimizer)


def test_asgd_multiple_optimizers_on_same_network_throws():
    n = _net()
    opt = thor.optimizers.ASGD(network=n)
    assert opt is not None
    assert isinstance(opt, thor.optimizers.ASGD)

    with pytest.raises(RuntimeError, match=r".*Multiple default optimizers.*"):
        thor.optimizers.ASGD(network=n)


def test_asgd_parameter_override_serializes_in_layer_architecture():
    n = thor.Network("test_net_asgd_parameter_override")
    input_layer = thor.layers.NetworkInput(n, "input", [4], thor.DataType.fp32)
    weights_optimizer = thor.optimizers.ASGD(alpha=0.1, lambd=0.01, power=0.5, t0=10.0, weight_decay=0.03)
    biases_optimizer = thor.optimizers.ASGD(alpha=0.05, lambd=0.02, power=0.25, t0=20.0, weight_decay=0.04)

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

    assert weights_json["optimizer_type"] == "asgd"
    assert weights_json["alpha"] == pytest.approx(0.1)
    assert weights_json["lambd"] == pytest.approx(0.01)
    assert weights_json["power"] == pytest.approx(0.5)
    assert weights_json["t0"] == pytest.approx(10.0)
    assert weights_json["weight_decay"] == pytest.approx(0.03)

    assert biases_json["optimizer_type"] == "asgd"
    assert biases_json["alpha"] == pytest.approx(0.05)
    assert biases_json["lambd"] == pytest.approx(0.02)
    assert biases_json["power"] == pytest.approx(0.25)
    assert biases_json["t0"] == pytest.approx(20.0)
    assert biases_json["weight_decay"] == pytest.approx(0.04)


def test_asgd_network_save_load_preserves_parameter_override(tmp_path):
    n = thor.Network("test_net_asgd_save_load")
    input_layer = thor.layers.NetworkInput(n, "input", [4], thor.DataType.fp32)
    fc = thor.layers.FullyConnected(
        n,
        input_layer.get_feature_output(),
        3,
        True,
        activation=None,
        weights_optimizer=thor.optimizers.ASGD(alpha=0.1, lambd=0.01, power=0.5, t0=10.0, weight_decay=0.03),
    )
    thor.layers.NetworkOutput(n, "output", fc.get_feature_output(), thor.DataType.fp32)

    save_dir = tmp_path / "asgd_network"
    n.save(str(save_dir), overwrite=False)

    loaded = thor.Network("test_net_asgd_save_load")
    loaded.load(str(save_dir))

    fc_layer = _only_fully_connected_layer(loaded)
    weights_json = fc_layer["parameters"]["weights"]["optimizer_override"]
    assert weights_json["optimizer_type"] == "asgd"
    assert weights_json["alpha"] == pytest.approx(0.1)
    assert weights_json["lambd"] == pytest.approx(0.01)
    assert weights_json["power"] == pytest.approx(0.5)
    assert weights_json["t0"] == pytest.approx(10.0)
    assert weights_json["weight_decay"] == pytest.approx(0.03)
