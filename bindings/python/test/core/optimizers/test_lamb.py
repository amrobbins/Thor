import json

import pytest
import thor


def _net():
    return thor.Network("test_net_lamb")


def _only_fully_connected_layer(network: thor.Network):
    layers = [layer for layer in json.loads(network.get_architecture_json())["layers"] if layer["layer_type"] == "fully_connected"]
    assert len(layers) == 1
    return layers[0]


def test_lamb_constructs_defaults():
    n = _net()
    opt = thor.optimizers.Lamb(network=n)
    assert opt is not None
    assert isinstance(opt, thor.optimizers.Lamb)

    arch = json.loads(n.get_architecture_json())["default_optimizer"]
    assert arch["optimizer_type"] == "lamb"
    assert arch["alpha"] == pytest.approx(0.001)
    assert arch["beta1"] == pytest.approx(0.9)
    assert arch["beta2"] == pytest.approx(0.999)
    assert arch["epsilon"] == pytest.approx(1e-6)
    assert arch["weight_decay"] == pytest.approx(0.01)
    assert arch["trust_ratio_epsilon"] == pytest.approx(1e-6)


def test_lamb_constructs_defaults_without_network():
    opt = thor.optimizers.Lamb()
    assert opt is not None
    assert isinstance(opt, thor.optimizers.Lamb)


def test_lamb_constructs_custom_params():
    n = _net()
    opt = thor.optimizers.Lamb(
        network=n,
        alpha=0.02,
        beta1=0.8,
        beta2=0.97,
        epsilon=1e-5,
        weight_decay=0.04,
        trust_ratio_epsilon=1e-4,
    )
    assert isinstance(opt, thor.optimizers.Lamb)

    arch = json.loads(n.get_architecture_json())["default_optimizer"]
    assert arch["optimizer_type"] == "lamb"
    assert arch["alpha"] == pytest.approx(0.02)
    assert arch["beta1"] == pytest.approx(0.8)
    assert arch["beta2"] == pytest.approx(0.97)
    assert arch["epsilon"] == pytest.approx(1e-5)
    assert arch["weight_decay"] == pytest.approx(0.04)
    assert arch["trust_ratio_epsilon"] == pytest.approx(1e-4)


def test_lamb_rejects_invalid_values():
    n = _net()

    with pytest.raises(ValueError, match=r"alpha must be > 0"):
        thor.optimizers.Lamb(network=n, alpha=0.0)
    with pytest.raises(ValueError, match=r"alpha must be > 0"):
        thor.optimizers.Lamb(alpha=-1.0)

    with pytest.raises(ValueError, match=r"0 <= beta1 < 1"):
        thor.optimizers.Lamb(beta1=-0.1)
    with pytest.raises(ValueError, match=r"0 <= beta1 < 1"):
        thor.optimizers.Lamb(beta1=1.0)

    with pytest.raises(ValueError, match=r"0 <= beta2 < 1"):
        thor.optimizers.Lamb(beta2=-0.1)
    with pytest.raises(ValueError, match=r"0 <= beta2 < 1"):
        thor.optimizers.Lamb(beta2=1.0)

    with pytest.raises(ValueError, match=r"epsilon must be > 0"):
        thor.optimizers.Lamb(epsilon=0.0)

    with pytest.raises(ValueError, match=r"weight_decay must be >= 0"):
        thor.optimizers.Lamb(weight_decay=-0.01)

    with pytest.raises(ValueError, match=r"trust_ratio_epsilon must be > 0"):
        thor.optimizers.Lamb(trust_ratio_epsilon=0.0)


def test_lamb_rejects_wrong_types_and_kwargs():
    n = _net()

    with pytest.raises(TypeError):
        thor.optimizers.Lamb("not a network")

    with pytest.raises(TypeError):
        thor.optimizers.Lamb(network=n, alpha="0.01")

    with pytest.raises(TypeError):
        thor.optimizers.Lamb(0.01, 0.9, 0.999, 1e-6, 0.01, 1e-6, 123, network=n)

    with pytest.raises(TypeError):
        thor.optimizers.Lamb(bogus=123, network=n)


def test_lamb_is_optimizer_subclass_if_exposed():
    Optimizer = getattr(thor.optimizers, "Optimizer", None)
    if Optimizer is None:
        pytest.skip("thor.optimizers.Optimizer not exposed in Python")
    assert isinstance(thor.optimizers.Lamb(), Optimizer)


def test_lamb_multiple_optimizers_on_same_network_throws():
    n = _net()
    opt = thor.optimizers.Lamb(network=n)
    assert opt is not None
    assert isinstance(opt, thor.optimizers.Lamb)

    with pytest.raises(RuntimeError, match=r".*Multiple default optimizers.*"):
        thor.optimizers.Lamb(network=n)


def test_lamb_parameter_override_serializes_in_layer_architecture():
    n = thor.Network("test_net_lamb_parameter_override")
    input_layer = thor.layers.NetworkInput(n, "input", [4], thor.DataType.fp32)
    weights_optimizer = thor.optimizers.Lamb(alpha=0.002, beta1=0.8, beta2=0.97, epsilon=1e-5, weight_decay=0.04, trust_ratio_epsilon=1e-4)
    biases_optimizer = thor.optimizers.Lamb(alpha=0.003, beta1=0.7, beta2=0.98, epsilon=1e-6, weight_decay=0.02, trust_ratio_epsilon=1e-5)

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

    assert weights_json["optimizer_type"] == "lamb"
    assert weights_json["alpha"] == pytest.approx(0.002)
    assert weights_json["beta1"] == pytest.approx(0.8)
    assert weights_json["beta2"] == pytest.approx(0.97)
    assert weights_json["epsilon"] == pytest.approx(1e-5)
    assert weights_json["weight_decay"] == pytest.approx(0.04)
    assert weights_json["trust_ratio_epsilon"] == pytest.approx(1e-4)

    assert biases_json["optimizer_type"] == "lamb"
    assert biases_json["alpha"] == pytest.approx(0.003)
    assert biases_json["beta1"] == pytest.approx(0.7)
    assert biases_json["beta2"] == pytest.approx(0.98)
    assert biases_json["epsilon"] == pytest.approx(1e-6)
    assert biases_json["weight_decay"] == pytest.approx(0.02)
    assert biases_json["trust_ratio_epsilon"] == pytest.approx(1e-5)


def test_lamb_network_save_load_preserves_parameter_override(tmp_path):
    n = thor.Network("test_net_lamb_save_load")
    input_layer = thor.layers.NetworkInput(n, "input", [4], thor.DataType.fp32)
    fc = thor.layers.FullyConnected(
        n,
        input_layer.get_feature_output(),
        3,
        True,
        activation=None,
        weights_optimizer=thor.optimizers.Lamb(alpha=0.002, beta1=0.8, beta2=0.97, epsilon=1e-5, weight_decay=0.04, trust_ratio_epsilon=1e-4),
    )
    thor.layers.NetworkOutput(n, "output", fc.get_feature_output(), thor.DataType.fp32)

    save_dir = tmp_path / "lamb_network"
    n.save(str(save_dir), overwrite=False)

    loaded = thor.Network("test_net_lamb_save_load")
    loaded.load(str(save_dir))

    fc_layer = _only_fully_connected_layer(loaded)
    weights_json = fc_layer["parameters"]["weights"]["optimizer_override"]
    assert weights_json["optimizer_type"] == "lamb"
    assert weights_json["alpha"] == pytest.approx(0.002)
    assert weights_json["beta1"] == pytest.approx(0.8)
    assert weights_json["beta2"] == pytest.approx(0.97)
    assert weights_json["epsilon"] == pytest.approx(1e-5)
    assert weights_json["weight_decay"] == pytest.approx(0.04)
    assert weights_json["trust_ratio_epsilon"] == pytest.approx(1e-4)
