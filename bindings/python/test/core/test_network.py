import pytest
import thor


def _build_minimal_network(name: str = "pytest_net") -> thor.Network:
    n = thor.Network(name)
    # Minimal graph: NetworkInput -> NetworkOutput
    ni = thor.layers.NetworkInput(n, "in", [1], thor.DataType.fp16)
    thor.layers.NetworkOutput(n, "out", ni.get_feature_output(), thor.DataType.fp16)
    return n


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
    n = _build_minimal_network()

    status = n.place(
        batch_size=1,
        inference_only=True,
    )

    # Ensure it's the enum type you expect
    assert isinstance(status, thor.Network.StatusCode)

    # If placement succeeds, num_stamps should be non-zero
    assert status == thor.Network.StatusCode.success
    assert n.get_num_stamps() >= 1
