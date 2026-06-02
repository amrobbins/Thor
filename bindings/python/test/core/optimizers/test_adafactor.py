import json

import pytest
import thor


def _net():
    return thor.Network("test_net_adafactor")


def _only_fully_connected_layer(network: thor.Network):
    layers = [layer for layer in json.loads(network.get_architecture_json())["layers"] if layer["layer_type"] == "fully_connected"]
    assert len(layers) == 1
    return layers[0]


def test_adafactor_constructs_defaults():
    n = _net()
    opt = thor.optimizers.Adafactor(network=n)
    assert opt is not None
    assert isinstance(opt, thor.optimizers.Adafactor)


def test_adafactor_constructs_defaults_without_network():
    opt = thor.optimizers.Adafactor()
    assert opt is not None
    assert isinstance(opt, thor.optimizers.Adafactor)


def test_adafactor_constructs_custom_params():
    n = _net()
    opt = thor.optimizers.Adafactor(
        network=n,
        alpha=0.004,
        beta2=0.98,
        epsilon=1e-6,
        weight_decay=0.01,
        factor_second_moment=False,
    )
    assert isinstance(opt, thor.optimizers.Adafactor)


def test_adafactor_rejects_non_positive_alpha():
    n = _net()
    with pytest.raises(ValueError, match=r"alpha must be > 0"):
        thor.optimizers.Adafactor(network=n, alpha=0.0)
    with pytest.raises(ValueError, match=r"alpha must be > 0"):
        thor.optimizers.Adafactor(alpha=-1.0)


def test_adafactor_rejects_invalid_beta2():
    n = _net()
    with pytest.raises(ValueError, match=r"0 <= beta2 < 1"):
        thor.optimizers.Adafactor(network=n, beta2=-0.1)
    with pytest.raises(ValueError, match=r"0 <= beta2 < 1"):
        thor.optimizers.Adafactor(beta2=1.0)


def test_adafactor_rejects_non_positive_epsilon():
    n = _net()
    with pytest.raises(ValueError, match=r"epsilon must be > 0"):
        thor.optimizers.Adafactor(network=n, epsilon=0.0)
    with pytest.raises(ValueError, match=r"epsilon must be > 0"):
        thor.optimizers.Adafactor(epsilon=-1e-7)


def test_adafactor_rejects_negative_weight_decay():
    n = _net()
    with pytest.raises(ValueError, match=r"weight_decay must be >= 0"):
        thor.optimizers.Adafactor(network=n, weight_decay=-1e-4)
    with pytest.raises(ValueError, match=r"weight_decay must be >= 0"):
        thor.optimizers.Adafactor(weight_decay=-1e-4)


def test_adafactor_rejects_wrong_types():
    n = _net()

    with pytest.raises(TypeError):
        thor.optimizers.Adafactor("not a network")

    with pytest.raises(TypeError):
        thor.optimizers.Adafactor(network=n, alpha="1.0")

    with pytest.raises(TypeError):
        thor.optimizers.Adafactor(network=n, beta2="0.999")

    with pytest.raises(TypeError):
        thor.optimizers.Adafactor(network=n, epsilon="1e-30")

    with pytest.raises(TypeError):
        thor.optimizers.Adafactor(network=n, weight_decay="0.0")

    with pytest.raises(TypeError):
        thor.optimizers.Adafactor(network=n, factor_second_moment="true")


def test_adafactor_rejects_wrong_arity_and_kwargs():
    n = _net()

    with pytest.raises(TypeError):
        thor.optimizers.Adafactor(0.001, 0.999, 1e-30, 0.0, True, 123, network=n)  # extra positional

    with pytest.raises(TypeError):
        thor.optimizers.Adafactor(bogus=123, network=n)  # wrong kw

    with pytest.raises(TypeError):
        thor.optimizers.Adafactor(alpha=0.001, extra=123, network=n)  # extra kw


def test_adafactor_is_optimizer_subclass_if_exposed():
    Optimizer = getattr(thor.optimizers, "Optimizer", None)
    if Optimizer is None:
        pytest.skip("thor.optimizers.Optimizer not exposed in Python")
    assert isinstance(thor.optimizers.Adafactor(), Optimizer)


def test_adafactor_multiple_optimizers_on_same_network_throws():
    n = _net()
    opt = thor.optimizers.Adafactor(network=n)
    assert opt is not None
    assert isinstance(opt, thor.optimizers.Adafactor)

    with pytest.raises(RuntimeError, match=r".*Multiple default optimizers.*"):
        thor.optimizers.Adafactor(network=n)


def test_adafactor_parameter_override_serializes_in_layer_architecture():
    n = thor.Network("test_net_adafactor_parameter_override")
    input_layer = thor.layers.NetworkInput(n, "input", [4], thor.DataType.fp32)
    weights_optimizer = thor.optimizers.Adafactor(alpha=0.004, beta2=0.98, epsilon=1e-6, weight_decay=0.01)
    biases_optimizer = thor.optimizers.Adafactor(
        alpha=0.002,
        beta2=0.95,
        epsilon=1e-5,
        weight_decay=0.02,
        factor_second_moment=False,
    )

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

    assert weights_json["optimizer_type"] == "adafactor"
    assert weights_json["alpha"] == pytest.approx(0.004)
    assert weights_json["beta2"] == pytest.approx(0.98)
    assert weights_json["epsilon"] == pytest.approx(1e-6)
    assert weights_json["weight_decay"] == pytest.approx(0.01)
    assert weights_json["factor_second_moment"] is True

    assert biases_json["optimizer_type"] == "adafactor"
    assert biases_json["alpha"] == pytest.approx(0.002)
    assert biases_json["beta2"] == pytest.approx(0.95)
    assert biases_json["epsilon"] == pytest.approx(1e-5)
    assert biases_json["weight_decay"] == pytest.approx(0.02)
    assert biases_json["factor_second_moment"] is False


def test_adafactor_network_save_load_preserves_parameter_override(tmp_path):
    n = thor.Network("test_net_adafactor_save_load")
    input_layer = thor.layers.NetworkInput(n, "input", [4], thor.DataType.fp32)
    fc = thor.layers.FullyConnected(
        n,
        input_layer.get_feature_output(),
        3,
        True,
        activation=None,
        weights_optimizer=thor.optimizers.Adafactor(
            alpha=0.004,
            beta2=0.98,
            epsilon=1e-6,
            weight_decay=0.01,
            factor_second_moment=False,
        ),
    )
    thor.layers.NetworkOutput(n, "output", fc.get_feature_output(), thor.DataType.fp32)

    save_dir = tmp_path / "adafactor_network"
    n.save(str(save_dir), overwrite=False)

    loaded = thor.Network("test_net_adafactor_save_load")
    loaded.load(str(save_dir))

    fc_layer = _only_fully_connected_layer(loaded)
    weights_json = fc_layer["parameters"]["weights"]["optimizer_override"]
    assert weights_json["optimizer_type"] == "adafactor"
    assert weights_json["alpha"] == pytest.approx(0.004)
    assert weights_json["beta2"] == pytest.approx(0.98)
    assert weights_json["epsilon"] == pytest.approx(1e-6)
    assert weights_json["weight_decay"] == pytest.approx(0.01)
    assert weights_json["factor_second_moment"] is False
