import json

import pytest
import thor


def _net():
    return thor.Network("test_net_adagrad")


def _only_fully_connected_layer(network: thor.Network):
    layers = [layer for layer in json.loads(network.get_architecture_json())["layers"] if layer["layer_type"] == "fully_connected"]
    assert len(layers) == 1
    return layers[0]


def test_adagrad_constructs_defaults():
    n = _net()
    opt = thor.optimizers.Adagrad(network=n)
    assert opt is not None
    assert isinstance(opt, thor.optimizers.Adagrad)


def test_adagrad_constructs_defaults_without_network():
    opt = thor.optimizers.Adagrad()
    assert opt is not None
    assert isinstance(opt, thor.optimizers.Adagrad)


def test_adagrad_constructs_custom_params():
    n = _net()
    opt = thor.optimizers.Adagrad(network=n, alpha=0.02, epsilon=1e-5)
    assert isinstance(opt, thor.optimizers.Adagrad)


def test_adagrad_rejects_non_positive_alpha():
    n = _net()
    with pytest.raises(ValueError, match=r"alpha must be > 0"):
        thor.optimizers.Adagrad(network=n, alpha=0.0)
    with pytest.raises(ValueError, match=r"alpha must be > 0"):
        thor.optimizers.Adagrad(alpha=-1.0)


def test_adagrad_rejects_non_positive_epsilon():
    n = _net()
    with pytest.raises(ValueError, match=r"epsilon must be > 0"):
        thor.optimizers.Adagrad(network=n, epsilon=0.0)
    with pytest.raises(ValueError, match=r"epsilon must be > 0"):
        thor.optimizers.Adagrad(epsilon=-1e-7)


def test_adagrad_rejects_wrong_types():
    n = _net()

    with pytest.raises(TypeError):
        thor.optimizers.Adagrad("not a network")

    with pytest.raises(TypeError):
        thor.optimizers.Adagrad(network=n, alpha="0.01")

    with pytest.raises(TypeError):
        thor.optimizers.Adagrad(network=n, epsilon="1e-7")


def test_adagrad_rejects_wrong_arity_and_kwargs():
    n = _net()

    with pytest.raises(TypeError):
        thor.optimizers.Adagrad(0.01, 1e-7, 123, network=n)  # extra positional

    with pytest.raises(TypeError):
        thor.optimizers.Adagrad(bogus=123, network=n)  # wrong kw

    with pytest.raises(TypeError):
        thor.optimizers.Adagrad(alpha=0.01, extra=123, network=n)  # extra kw


def test_adagrad_is_optimizer_subclass_if_exposed():
    Optimizer = getattr(thor.optimizers, "Optimizer", None)
    if Optimizer is None:
        pytest.skip("thor.optimizers.Optimizer not exposed in Python")
    assert isinstance(thor.optimizers.Adagrad(), Optimizer)


def test_adagrad_multiple_optimizers_on_same_network_throws():
    n = _net()
    opt = thor.optimizers.Adagrad(network=n)
    assert opt is not None
    assert isinstance(opt, thor.optimizers.Adagrad)

    with pytest.raises(RuntimeError, match=r".*Multiple default optimizers.*"):
        thor.optimizers.Adagrad(network=n)


def test_adagrad_parameter_override_serializes_in_layer_architecture():
    n = thor.Network("test_net_adagrad_parameter_override")
    input_layer = thor.layers.NetworkInput(n, "input", [4], thor.DataType.fp32)
    weights_optimizer = thor.optimizers.Adagrad(alpha=0.002, epsilon=1e-5)
    biases_optimizer = thor.optimizers.Adagrad(alpha=0.003, epsilon=1e-6)

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

    assert weights_json["optimizer_type"] == "adagrad"
    assert weights_json["alpha"] == pytest.approx(0.002)
    assert weights_json["epsilon"] == pytest.approx(1e-5)

    assert biases_json["optimizer_type"] == "adagrad"
    assert biases_json["alpha"] == pytest.approx(0.003)
    assert biases_json["epsilon"] == pytest.approx(1e-6)


def test_adagrad_network_save_load_preserves_parameter_override(tmp_path):
    n = thor.Network("test_net_adagrad_save_load")
    input_layer = thor.layers.NetworkInput(n, "input", [4], thor.DataType.fp32)
    fc = thor.layers.FullyConnected(
        n,
        input_layer.get_feature_output(),
        3,
        True,
        activation=None,
        weights_optimizer=thor.optimizers.Adagrad(alpha=0.002, epsilon=1e-5),
    )
    thor.layers.NetworkOutput(n, "output", fc.get_feature_output(), thor.DataType.fp32)

    save_dir = tmp_path / "adagrad_network"
    n.save(str(save_dir), overwrite=False)

    loaded = thor.Network("test_net_adagrad_save_load")
    loaded.load(str(save_dir))

    fc_layer = _only_fully_connected_layer(loaded)
    weights_json = fc_layer["parameters"]["weights"]["optimizer_override"]
    assert weights_json["optimizer_type"] == "adagrad"
    assert weights_json["alpha"] == pytest.approx(0.002)
    assert weights_json["epsilon"] == pytest.approx(1e-5)
