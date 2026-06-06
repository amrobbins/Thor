import json

import pytest
import thor


def _make_single_fc_network(name: str):
    n = thor.Network(name)
    x_in = thor.layers.NetworkInput(n, "input", [3], thor.DataType.fp32)
    fc = thor.layers.FullyConnected(n, x_in.get_feature_output(), 2, True, activation=None)
    thor.layers.NetworkOutput(n, "output", fc.get_feature_output(), thor.DataType.fp32)
    return n, fc


def _fully_connected_arch(n: thor.Network):
    layers = [layer for layer in json.loads(n.get_architecture_json())["layers"] if layer["layer_type"] == "fully_connected"]
    assert len(layers) == 1
    return layers[0]


def _parameter_training_enabled(fc_arch, name: str) -> bool:
    return fc_arch["parameters"][name]["training_enabled"]


def test_trainable_layer_freeze_and_unfreeze_training_updates_parameter_state():
    n, fc = _make_single_fc_network("test_net_layer_freeze_training")

    assert fc.is_training_frozen() is False
    assert _parameter_training_enabled(_fully_connected_arch(n), "weights") is True
    assert _parameter_training_enabled(_fully_connected_arch(n), "biases") is True

    fc.freeze_training()

    assert fc.is_training_frozen() is True
    fc_arch = _fully_connected_arch(n)
    assert _parameter_training_enabled(fc_arch, "weights") is False
    assert _parameter_training_enabled(fc_arch, "biases") is False

    fc.unfreeze_training()

    assert fc.is_training_frozen() is False
    fc_arch = _fully_connected_arch(n)
    assert _parameter_training_enabled(fc_arch, "weights") is True
    assert _parameter_training_enabled(fc_arch, "biases") is True


def test_network_freeze_and_unfreeze_training_updates_all_trainable_layers():
    n = thor.Network("test_net_network_freeze_training")
    x_in = thor.layers.NetworkInput(n, "input", [3], thor.DataType.fp32)
    fc1 = thor.layers.FullyConnected(n, x_in.get_feature_output(), 4, True, activation=None)
    fc2 = thor.layers.FullyConnected(n, fc1.get_feature_output(), 2, True, activation=None)
    thor.layers.NetworkOutput(n, "output", fc2.get_feature_output(), thor.DataType.fp32)

    n.freeze_training()
    arch = json.loads(n.get_architecture_json())
    fc_layers = [layer for layer in arch["layers"] if layer["layer_type"] == "fully_connected"]
    assert len(fc_layers) == 2
    for fc_arch in fc_layers:
        assert _parameter_training_enabled(fc_arch, "weights") is False
        assert _parameter_training_enabled(fc_arch, "biases") is False

    n.unfreeze_training()
    arch = json.loads(n.get_architecture_json())
    fc_layers = [layer for layer in arch["layers"] if layer["layer_type"] == "fully_connected"]
    for fc_arch in fc_layers:
        assert _parameter_training_enabled(fc_arch, "weights") is True
        assert _parameter_training_enabled(fc_arch, "biases") is True


@pytest.mark.cuda
def test_frozen_trainable_layer_does_not_require_optimizer_for_training_placement():
    n, fc = _make_single_fc_network("test_net_frozen_layer_no_optimizer_training_place")
    fc.freeze_training()

    placed = n.place(2, inference_only=False, forced_devices=[0], forced_num_stamps_per_gpu=1)
    assert placed.get_num_trainable_layers() == 1
    bound_parameters = fc.get_bound_parameters(placed)
    assert [parameter.name for parameter in bound_parameters] == ["weights", "biases"]
    assert all(parameter.is_training_enabled() is False for parameter in bound_parameters)


@pytest.mark.cuda
def test_unfrozen_trainable_layer_still_requires_optimizer_for_training_placement():
    n, _ = _make_single_fc_network("test_net_unfrozen_layer_requires_optimizer")

    with pytest.raises(RuntimeError, match="does not have an optimizer assigned"):
        n.place(2, inference_only=False, forced_devices=[0], forced_num_stamps_per_gpu=1)


def test_freeze_training_state_save_load_round_trip(tmp_path):
    name = "test_net_freeze_training_round_trip"
    n, fc = _make_single_fc_network(name)
    fc.freeze_training()

    save_dir = tmp_path / "frozen_training_model"
    n.save(str(save_dir), overwrite=False)

    loaded = thor.Network(name)
    loaded.load(str(save_dir))
    fc_arch = _fully_connected_arch(loaded)
    assert _parameter_training_enabled(fc_arch, "weights") is False
    assert _parameter_training_enabled(fc_arch, "biases") is False
