import json
import shutil
from tempfile import TemporaryDirectory

import pytest

import thor


def test_network_constructor_and_name():
    n = thor.Network("mine")
    assert n.get_network_name() == "mine"


def test_network_status_code_is_nested_enum():
    # Make sure the nested enum exists and values are accessible
    assert hasattr(thor.Network, "StatusCode")
    assert thor.Network.StatusCode.success is not None


def test_network_status_code_to_string_success():
    n = thor.Network("mine")
    s = n.status_code_to_string(thor.Network.StatusCode.success)
    assert isinstance(s, str)
    assert len(s) > 0


def test_network_place_returns_status_code_and_updates_num_stamps():
    n = thor.Network("pytest_net")
    # Minimal graph: NetworkInput -> NetworkOutput
    ni = thor.layers.NetworkInput(n, "in", [1], thor.DataType.fp16)
    no = thor.layers.NetworkOutput(n, "out", ni.get_feature_output(), thor.DataType.fp16)
    fo = ni.get_feature_output()

    actual_architecture = json.loads(n.get_architecture_json())

    ni_fi_id = actual_architecture["layers"][0]["feature_input"]["id"]
    ni_fo_id = actual_architecture["layers"][0]["feature_output"]["id"]
    no_fo_id = actual_architecture["layers"][1]["feature_output"]["id"]

    assert isinstance(ni_fi_id, int)
    assert isinstance(ni_fo_id, int)
    assert isinstance(no_fo_id, int)

    expected_architecture = {
        "layers":
            [
                {
                    "data_type": "fp16",
                    "dimensions": [1],
                    "factory": "layer",
                    "feature_input": {
                        "data_type": "fp16",
                        "dimensions": [1],
                        "id": ni_fi_id,
                        "version": fo.version(),
                    },
                    "feature_output": {
                        "data_type": "fp16",
                        "dimensions": [1],
                        "id": ni_fo_id,
                        "version": fo.version(),
                    },
                    "layer_type": "network_input",
                    "name": "in",
                    "version": ni.version(),
                },
                {
                    "data_type": "fp16",
                    "factory": "layer",
                    "feature_input": {
                        "data_type": "fp16",
                        "dimensions": [1],
                        "id": ni_fo_id,
                        "version": fo.version(),
                    },
                    "feature_output": {
                        "data_type": "fp16",
                        "dimensions": [1],
                        "id": no_fo_id,
                        "version": fo.version(),
                    },
                    "layer_type": "network_output",
                    "name": "out",
                    "version": no.version(),
                },
            ]
    }

    assert actual_architecture == expected_architecture

    placed_network = n.place(
        batch_size=1,
        inference_only=True,
    )

    # Ensure it's the enum type you expect
    assert isinstance(placed_network, thor.PlacedNetwork)

    # If placement succeeds, num_stamps should be non-zero
    assert placed_network is not None
    assert placed_network.get_num_stamps() >= 1


def test_network_architecture_json_valid_for_partial_network():
    n = thor.Network("pytest_net")
    # Partial graph: NetworkInput -> <Nothing>
    ni = thor.layers.NetworkInput(n, "in", [1], thor.DataType.fp16)
    fo = ni.get_feature_output()

    actual_architecture = json.loads(n.get_architecture_json())

    ni_fi_id = actual_architecture["layers"][0]["feature_input"]["id"]
    ni_fo_id = actual_architecture["layers"][0]["feature_output"]["id"]

    expected_architecture = {
        "layers":
            [
                {
                    "data_type": "fp16",
                    "dimensions": [1],
                    "factory": "layer",
                    "feature_input": {
                        "data_type": "fp16",
                        "dimensions": [1],
                        "id": ni_fi_id,
                        "version": fo.version(),
                    },
                    "feature_output": {
                        "data_type": "fp16",
                        "dimensions": [1],
                        "id": ni_fo_id,
                        "version": fo.version(),
                    },
                    "layer_type": "network_input",
                    "name": "in",
                    "version": ni.version(),
                },
            ]
    }

    assert actual_architecture == expected_architecture


def test_network_save_load():
    with TemporaryDirectory(prefix="thor_pytest_") as tmp_path:
        from thor.layers import FullyConnected
        from thor.activations import Relu
        from thor.optimizers import Adam
        n = thor.Network("pytest_net")
        ni = thor.layers.NetworkInput(n, "in", [1], thor.DataType.fp16)
        fc = thor.layers.FullyConnected(n, ni.get_feature_output(), 1, True, thor.activations.Relu(), optimizer=Adam())
        thor.layers.NetworkOutput(n, "out", fc.get_feature_output(), thor.DataType.fp16)

        n.save(tmp_path, overwrite=True)

        n2 = thor.Network("pytest_net")
        n2.load(tmp_path)

        net_structure = json.loads(n2.get_architecture_json())

        expected_layers = {
            'network_input',
            'fully_connected',
            'relu',
            'network_output',
        }
        for layer in net_structure['layers']:
            assert layer['layer_type'] in expected_layers
            expected_layers.remove(layer['layer_type'])
