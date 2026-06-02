import json

import pytest
import thor


def _net():
    return thor.Network("test_net_radam")


def _only_fully_connected_layer(network: thor.Network):
    layers = [layer for layer in json.loads(network.get_architecture_json())["layers"] if layer["layer_type"] == "fully_connected"]
    assert len(layers) == 1
    return layers[0]


def test_radam_constructs_defaults():
    n = _net()
    opt = thor.optimizers.RAdam(network=n)
    assert opt is not None
    assert isinstance(opt, thor.optimizers.RAdam)


def test_radam_constructs_defaults_without_network():
    opt = thor.optimizers.RAdam()
    assert opt is not None
    assert isinstance(opt, thor.optimizers.RAdam)


def test_radam_constructs_custom_params():
    n = _net()
    opt = thor.optimizers.RAdam(network=n, alpha=0.02, beta1=0.85, beta2=0.95, epsilon=1e-5)
    assert isinstance(opt, thor.optimizers.RAdam)


def test_radam_rejects_non_positive_alpha():
    n = _net()
    with pytest.raises(ValueError, match=r"alpha must be > 0"):
        thor.optimizers.RAdam(network=n, alpha=0.0)
    with pytest.raises(ValueError, match=r"alpha must be > 0"):
        thor.optimizers.RAdam(alpha=-1.0)


def test_radam_rejects_non_positive_epsilon():
    n = _net()
    with pytest.raises(ValueError, match=r"epsilon must be > 0"):
        thor.optimizers.RAdam(network=n, epsilon=0.0)
    with pytest.raises(ValueError, match=r"epsilon must be > 0"):
        thor.optimizers.RAdam(epsilon=-1e-7)


def test_radam_rejects_invalid_beta1():
    n = _net()
    with pytest.raises(ValueError, match=r"0 <= beta1 < 1"):
        thor.optimizers.RAdam(network=n, beta1=-0.01)
    with pytest.raises(ValueError, match=r"0 <= beta1 < 1"):
        thor.optimizers.RAdam(beta1=1.0)
    with pytest.raises(ValueError, match=r"0 <= beta1 < 1"):
        thor.optimizers.RAdam(network=n, beta1=1.01)


def test_radam_rejects_invalid_beta2():
    n = _net()
    with pytest.raises(ValueError, match=r"0 <= beta2 < 1"):
        thor.optimizers.RAdam(beta2=-0.01)
    with pytest.raises(ValueError, match=r"0 <= beta2 < 1"):
        thor.optimizers.RAdam(network=n, beta2=1.0)
    with pytest.raises(ValueError, match=r"0 <= beta2 < 1"):
        thor.optimizers.RAdam(network=n, beta2=1.01)


def test_radam_rejects_wrong_types():
    n = _net()

    with pytest.raises(TypeError):
        thor.optimizers.RAdam("not a network")

    with pytest.raises(TypeError):
        thor.optimizers.RAdam(network=n, alpha="0.01")

    with pytest.raises(TypeError):
        thor.optimizers.RAdam(network=n, beta1="0.9")

    with pytest.raises(TypeError):
        thor.optimizers.RAdam(network=n, beta2="0.999")

    with pytest.raises(TypeError):
        thor.optimizers.RAdam(network=n, epsilon="1e-7")


def test_radam_rejects_wrong_arity_and_kwargs():
    n = _net()

    with pytest.raises(TypeError):
        thor.optimizers.RAdam(0.002, 0.9, 0.999, 1e-7, 123, network=n)  # extra positional

    with pytest.raises(TypeError):
        thor.optimizers.RAdam(bogus=123, network=n)  # wrong kw

    with pytest.raises(TypeError):
        thor.optimizers.RAdam(alpha=0.002, extra=123, network=n)  # extra kw


def test_radam_is_optimizer_subclass_if_exposed():
    Optimizer = getattr(thor.optimizers, "Optimizer", None)
    if Optimizer is None:
        pytest.skip("thor.optimizers.Optimizer not exposed in Python")
    assert isinstance(thor.optimizers.RAdam(), Optimizer)


def test_radam_multiple_optimizers_on_same_network_throws():
    n = _net()
    opt = thor.optimizers.RAdam(network=n)
    assert opt is not None
    assert isinstance(opt, thor.optimizers.RAdam)

    with pytest.raises(RuntimeError, match=r".*Multiple default optimizers.*"):
        thor.optimizers.RAdam(network=n)


def test_radam_parameter_override_serializes_in_layer_architecture():
    n = thor.Network("test_net_radam_parameter_override")
    input_layer = thor.layers.NetworkInput(n, "input", [4], thor.DataType.fp32)
    weights_optimizer = thor.optimizers.RAdam(alpha=0.002, beta1=0.85, beta2=0.95, epsilon=1e-5)
    biases_optimizer = thor.optimizers.RAdam(alpha=0.003, beta1=0.8, beta2=0.9, epsilon=1e-6)

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

    assert weights_json["optimizer_type"] == "radam"
    assert weights_json["alpha"] == pytest.approx(0.002)
    assert weights_json["beta1"] == pytest.approx(0.85)
    assert weights_json["beta2"] == pytest.approx(0.95)
    assert weights_json["epsilon"] == pytest.approx(1e-5)

    assert biases_json["optimizer_type"] == "radam"
    assert biases_json["alpha"] == pytest.approx(0.003)
    assert biases_json["beta1"] == pytest.approx(0.8)
    assert biases_json["beta2"] == pytest.approx(0.9)
    assert biases_json["epsilon"] == pytest.approx(1e-6)


def test_radam_network_save_load_preserves_parameter_override(tmp_path):
    n = thor.Network("test_net_radam_save_load")
    input_layer = thor.layers.NetworkInput(n, "input", [4], thor.DataType.fp32)
    fc = thor.layers.FullyConnected(
        n,
        input_layer.get_feature_output(),
        3,
        True,
        activation=None,
        weights_optimizer=thor.optimizers.RAdam(alpha=0.002, beta1=0.85, beta2=0.95, epsilon=1e-5),
    )
    thor.layers.NetworkOutput(n, "output", fc.get_feature_output(), thor.DataType.fp32)

    save_dir = tmp_path / "radam_network"
    n.save(str(save_dir), overwrite=False)

    loaded = thor.Network("test_net_radam_save_load")
    loaded.load(str(save_dir))

    fc_layer = _only_fully_connected_layer(loaded)
    weights_json = fc_layer["parameters"]["weights"]["optimizer_override"]
    assert weights_json["optimizer_type"] == "radam"
    assert weights_json["alpha"] == pytest.approx(0.002)
    assert weights_json["beta1"] == pytest.approx(0.85)
    assert weights_json["beta2"] == pytest.approx(0.95)
    assert weights_json["epsilon"] == pytest.approx(1e-5)
