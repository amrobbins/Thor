import json

import pytest
import thor


def _net():
    return thor.Network("test_net_muon")


def _only_fully_connected_layer(network: thor.Network):
    layers = [layer for layer in json.loads(network.get_architecture_json())["layers"] if layer["layer_type"] == "fully_connected"]
    assert len(layers) == 1
    return layers[0]


def test_muon_constructs_defaults_with_adamw_fallback():
    n = _net()
    opt = thor.optimizers.Muon(network=n)
    assert opt is not None
    assert isinstance(opt, thor.optimizers.Muon)

    arch = json.loads(n.get_architecture_json())["default_optimizer"]
    assert arch["optimizer_type"] == "muon"
    assert arch["alpha"] == pytest.approx(0.02)
    assert arch["beta"] == pytest.approx(0.95)
    assert arch["epsilon"] == pytest.approx(1.0e-8)
    assert arch["weight_decay"] == pytest.approx(0.0)
    assert arch["nesterov"] is True
    assert arch["num_iterations"] == 5
    assert arch["fallback_optimizer"]["optimizer_type"] == "adamw"


def test_muon_constructs_custom_params_and_custom_fallback():
    fallback = thor.optimizers.Sgd(initial_learning_rate=0.04)
    opt = thor.optimizers.Muon(
        alpha=0.03,
        beta=0.8,
        epsilon=1e-6,
        weight_decay=0.02,
        nesterov=False,
        num_iterations=3,
        coefficient_a=3.0,
        coefficient_b=-4.0,
        coefficient_c=2.0,
        transpose_tall_matrices=False,
        fallback_optimizer=fallback,
    )
    assert isinstance(opt, thor.optimizers.Muon)

    n = thor.Network("test_net_muon_custom_fallback")
    input_layer = thor.layers.NetworkInput(n, "input", [4], thor.DataType.fp32)
    thor.layers.FullyConnected(
        n,
        input_layer.get_feature_output(),
        3,
        True,
        activation=None,
        weights_optimizer=opt,
    )

    fc_layer = _only_fully_connected_layer(n)
    weights_json = fc_layer["parameters"]["weights"]["optimizer_override"]
    assert weights_json["optimizer_type"] == "muon"
    assert weights_json["alpha"] == pytest.approx(0.03)
    assert weights_json["beta"] == pytest.approx(0.8)
    assert weights_json["epsilon"] == pytest.approx(1e-6)
    assert weights_json["weight_decay"] == pytest.approx(0.02)
    assert weights_json["nesterov"] is False
    assert weights_json["num_iterations"] == 3
    assert weights_json["coefficient_a"] == pytest.approx(3.0)
    assert weights_json["coefficient_b"] == pytest.approx(-4.0)
    assert weights_json["coefficient_c"] == pytest.approx(2.0)
    assert weights_json["transpose_tall_matrices"] is False
    assert weights_json["fallback_optimizer"]["optimizer_type"] == "sgd"
    assert weights_json["fallback_optimizer"]["initial_learning_rate"] == pytest.approx(0.04)


def test_muon_rejects_invalid_params():
    with pytest.raises(ValueError, match=r"alpha must be > 0"):
        thor.optimizers.Muon(alpha=0.0)
    with pytest.raises(ValueError, match=r"0 <= beta < 1"):
        thor.optimizers.Muon(beta=1.0)
    with pytest.raises(ValueError, match=r"epsilon must be > 0"):
        thor.optimizers.Muon(epsilon=0.0)
    with pytest.raises(ValueError, match=r"weight_decay must be >= 0"):
        thor.optimizers.Muon(weight_decay=-1.0)
    with pytest.raises(ValueError, match=r"num_iterations must be > 0"):
        thor.optimizers.Muon(num_iterations=0)


def test_muon_rejects_wrong_types_and_kwargs():
    with pytest.raises(TypeError):
        thor.optimizers.Muon(alpha="0.02")
    with pytest.raises(TypeError):
        thor.optimizers.Muon(fallback_optimizer="adamw")
    with pytest.raises(TypeError):
        thor.optimizers.Muon(bogus=123)


def test_muon_is_optimizer_subclass_if_exposed():
    Optimizer = getattr(thor.optimizers, "Optimizer", None)
    if Optimizer is None:
        pytest.skip("thor.optimizers.Optimizer not exposed in Python")
    assert isinstance(thor.optimizers.Muon(), Optimizer)


def test_muon_multiple_optimizers_on_same_network_throws():
    n = _net()
    opt = thor.optimizers.Muon(network=n)
    assert opt is not None
    with pytest.raises(RuntimeError, match=r".*Multiple default optimizers.*"):
        thor.optimizers.Muon(network=n)


def test_muon_network_save_load_preserves_parameter_override(tmp_path):
    n = thor.Network("test_net_muon_save_load")
    input_layer = thor.layers.NetworkInput(n, "input", [4], thor.DataType.fp32)
    fc = thor.layers.FullyConnected(
        n,
        input_layer.get_feature_output(),
        3,
        True,
        activation=None,
        weights_optimizer=thor.optimizers.Muon(alpha=0.03, beta=0.8, fallback_optimizer=thor.optimizers.AdamW(alpha=0.002)),
    )
    thor.layers.NetworkOutput(n, "output", fc.get_feature_output(), thor.DataType.fp32)

    save_dir = tmp_path / "muon_network"
    n.save(str(save_dir), overwrite=False)

    loaded = thor.Network("test_net_muon_save_load")
    loaded.load(str(save_dir))

    fc_layer = _only_fully_connected_layer(loaded)
    weights_json = fc_layer["parameters"]["weights"]["optimizer_override"]
    assert weights_json["optimizer_type"] == "muon"
    assert weights_json["alpha"] == pytest.approx(0.03)
    assert weights_json["beta"] == pytest.approx(0.8)
    assert weights_json["fallback_optimizer"]["optimizer_type"] == "adamw"
    assert weights_json["fallback_optimizer"]["alpha"] == pytest.approx(0.002)
