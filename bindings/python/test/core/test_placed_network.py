import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import thor


def _build_minimal_network() -> thor.Network:
    n = thor.Network("pytest_net")
    # Minimal graph: NetworkInput -> NetworkOutput
    ni = thor.layers.NetworkInput(n, "in", [1], thor.DataType.fp16)
    thor.layers.NetworkOutput(n, "out", ni.get_feature_output(), thor.DataType.fp16)
    return n


@pytest.mark.cuda
def test_network_place_returns_placed_network():
    net = _build_minimal_network()

    placed = net.place(32)

    assert isinstance(placed, thor.PlacedNetwork)
    assert placed.get_network_name() == "pytest_net"
    assert placed.get_num_stamps() >= 1
    assert placed.get_num_trainable_layers() == net.get_num_trainable_layers()


@pytest.mark.cuda
def test_placed_network_save():
    with TemporaryDirectory(prefix="thor_pytest_") as tmp_path:
        net = _build_minimal_network()

        placed = net.place(7)

        out_dir = Path(tmp_path)
        placed.save(str(out_dir), overwrite=False, save_optimizer_state=False)

        assert out_dir.exists()
        assert out_dir.is_dir()
        assert out_dir.is_dir()
        archivePath = out_dir / "pytest_net.thor.tar"
        assert archivePath.exists()
        assert archivePath.is_file()


@pytest.mark.cuda
def test_placed_network_save_overwrite():
    with TemporaryDirectory(prefix="thor_pytest_") as tmp_path:
        net = _build_minimal_network()

        placed = net.place(50)

        out_dir = Path(tmp_path) / "placed_network"
        placed.save(str(out_dir), overwrite=False, save_optimizer_state=False)
        placed.save(str(out_dir), overwrite=True, save_optimizer_state=False)

        assert out_dir.exists()
        assert out_dir.is_dir()
        archivePath = out_dir / "pytest_net.thor.tar"
        assert archivePath.exists()
        assert archivePath.is_file()


@pytest.mark.cuda
def test_placed_network_basic_api(tmp_path):
    net = _build_minimal_network()

    placed = net.place(1)

    assert isinstance(placed, thor.PlacedNetwork)
    assert placed.get_network_name() == "pytest_net"

    num_stamps = placed.get_num_stamps()
    assert isinstance(num_stamps, int)
    assert num_stamps >= 1

    num_trainable_layers = placed.get_num_trainable_layers()
    assert isinstance(num_trainable_layers, int)
    assert num_trainable_layers == net.get_num_trainable_layers()

    save_dir = tmp_path / "saved"
    placed.save(str(save_dir), False, False)
    assert save_dir.exists()

    n = thor.Network("pytest_net")
    n.load(str(save_dir))
    placed2 = n.place(8)

    assert isinstance(placed2, thor.PlacedNetwork)
    assert placed2.get_network_name() == "pytest_net"

    num_stamps = placed2.get_num_stamps()
    assert isinstance(num_stamps, int)
    assert num_stamps >= 1

    num_trainable_layers = placed2.get_num_trainable_layers()
    assert isinstance(num_trainable_layers, int)
    assert num_trainable_layers == net.get_num_trainable_layers()
