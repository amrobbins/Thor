import json

import numpy as np
import pytest

import thor


def _place(network: thor.Network, batch_size: int) -> thor.runtime.PlacedNetwork:
    return network.place(
        batch_size,
        inference_only=True,
        forced_devices=[0],
        forced_num_stamps_per_gpu=1,
    )


def _cpu_tensor_from_numpy(values: np.ndarray, dtype: thor.DataType) -> thor.physical.PhysicalTensor:
    values = np.asarray(values, dtype=thor.physical.numpy_dtypes.from_thor(dtype), order="C")
    placement = thor.physical.Placement(thor.physical.DeviceType.cpu, 0)
    descriptor = thor.physical.PhysicalTensor.Descriptor(dtype, list(values.shape))
    tensor = thor.physical.PhysicalTensor(placement, descriptor)
    tensor.numpy()[...] = values
    return tensor


def _tail_network(name: str) -> tuple[thor.Network, thor.layers.Slice]:
    network = thor.Network(name)
    sequence = thor.layers.NetworkInput(network, "sequence", [6, 2], thor.DataType.fp32).get_feature_output()
    tail = thor.layers.Slice(network, sequence, axis=0, start=-3, length=3)
    thor.layers.NetworkOutput(network, "tail", tail.get_feature_output(), thor.DataType.fp32)
    return network, tail


def test_slice_uses_logical_axes_and_serializes_declaratively():
    network = thor.Network("slice-logical-axis")
    sequence = thor.layers.NetworkInput(network, "sequence", [6, 2], thor.DataType.fp32).get_feature_output()

    time_tail = thor.layers.Slice(network, sequence, axis=0, start=-3, length=3)
    final_channel = thor.layers.Slice(network, sequence, axis=1, start=-1, length=1)

    assert time_tail.get_feature_output().get_dimensions() == [3, 2]
    assert final_channel.get_feature_output().get_dimensions() == [6, 1]
    assert time_tail.axis == 0
    assert time_tail.start == -3
    assert time_tail.length == 3

    architecture = json.loads(network.get_architecture_json())
    slice_layers = [layer for layer in architecture["layers"] if layer["layer_type"] == "slice"]
    assert len(slice_layers) == 2
    assert any(
        layer["axis"] == 0 and layer["start"] == -3 and layer["length"] == 3
        for layer in slice_layers
    )
    assert all("expression" not in layer for layer in slice_layers)


@pytest.mark.cuda
def test_slice_forward_is_batch_polymorphic_across_placements():
    network, _ = _tail_network("slice-forward-batch-polymorphic")

    placed_four = _place(network, batch_size=4)
    values4 = np.arange(4 * 6 * 2, dtype=np.float32).reshape(4, 6, 2)
    outputs4 = placed_four.infer({"sequence": _cpu_tensor_from_numpy(values4, thor.DataType.fp32)})
    np.testing.assert_array_equal(np.array(outputs4["tail"].numpy(), copy=True), values4[:, -3:, :])

    # PlacedNetwork descriptors are currently static, so a smaller final batch uses
    # a second placement of the same logical model rather than changing an existing stamp.
    placed_two = _place(network, batch_size=2)
    values2 = np.arange(2 * 6 * 2, dtype=np.float32).reshape(2, 6, 2) + np.float32(1000.0)
    outputs2 = placed_two.infer({"sequence": _cpu_tensor_from_numpy(values2, thor.DataType.fp32)})
    np.testing.assert_array_equal(np.array(outputs2["tail"].numpy(), copy=True), values2[:, -3:, :])


@pytest.mark.cuda
def test_slice_save_load_places_at_different_batch_size(tmp_path):
    network_name = "slice-save-load-different-batch"
    network, _ = _tail_network(network_name)

    # Exercise the architecture once at batch 1, then reload and place at batch 4.
    placed_one = _place(network, batch_size=1)
    values1 = np.arange(6 * 2, dtype=np.float32).reshape(1, 6, 2)
    outputs1 = placed_one.infer({"sequence": _cpu_tensor_from_numpy(values1, thor.DataType.fp32)})
    np.testing.assert_array_equal(np.array(outputs1["tail"].numpy(), copy=True), values1[:, -3:, :])

    save_dir = tmp_path / "slice_model"
    network.save(str(save_dir), overwrite=False)

    loaded = thor.Network(network_name)
    loaded.load(str(save_dir))
    placed_four = _place(loaded, batch_size=4)
    values4 = np.arange(4 * 6 * 2, dtype=np.float32).reshape(4, 6, 2)
    outputs4 = placed_four.infer({"sequence": _cpu_tensor_from_numpy(values4, thor.DataType.fp32)})
    np.testing.assert_array_equal(np.array(outputs4["tail"].numpy(), copy=True), values4[:, -3:, :])


@pytest.mark.cuda
def test_slice_dense_output_feeds_concatenate():
    network = thor.Network("slice-to-concatenate")
    sequence = thor.layers.NetworkInput(network, "sequence", [6, 2], thor.DataType.fp32).get_feature_output()
    extra = thor.layers.NetworkInput(network, "extra", [3, 1], thor.DataType.fp32).get_feature_output()
    tail = thor.layers.Slice(network, sequence, axis=0, start=-3, length=3).get_feature_output()
    joined = thor.layers.Concatenate(network, [tail, extra], concatenation_axis=1).get_feature_output()
    thor.layers.NetworkOutput(network, "joined", joined, thor.DataType.fp32)

    placed = _place(network, batch_size=4)
    sequence_values = np.arange(4 * 6 * 2, dtype=np.float32).reshape(4, 6, 2)
    extra_values = np.arange(4 * 3, dtype=np.float32).reshape(4, 3, 1) + np.float32(1000.0)
    outputs = placed.infer({
        "sequence": _cpu_tensor_from_numpy(sequence_values, thor.DataType.fp32),
        "extra": _cpu_tensor_from_numpy(extra_values, thor.DataType.fp32),
    })
    expected = np.concatenate([sequence_values[:, -3:, :], extra_values], axis=2)
    np.testing.assert_array_equal(np.array(outputs["joined"].numpy(), copy=True), expected)


@pytest.mark.cuda
def test_slice_dense_output_feeds_fully_connected():
    network = thor.Network("slice-to-fully-connected")
    sequence = thor.layers.NetworkInput(network, "sequence", [6, 2], thor.DataType.fp32).get_feature_output()
    tail = thor.layers.Slice(network, sequence, axis=0, start=-3, length=3).get_feature_output()
    projected = thor.layers.FullyConnected(
        network,
        tail,
        num_output_features=4,
        activation=None,
        preserve_prefix_dimensions=True,
        weights_data_type=thor.DataType.fp32,
        compute_data_type=thor.DataType.fp32,
        output_data_type=thor.DataType.fp32,
    ).get_feature_output()
    thor.layers.NetworkOutput(network, "projected", projected, thor.DataType.fp32)

    assert projected.get_dimensions() == [3, 4]
    placed = _place(network, batch_size=4)
    values = np.arange(4 * 6 * 2, dtype=np.float32).reshape(4, 6, 2)
    outputs = placed.infer({"sequence": _cpu_tensor_from_numpy(values, thor.DataType.fp32)})
    assert list(outputs["projected"].get_dimensions()) == [4, 3, 4]
    assert np.isfinite(np.array(outputs["projected"].numpy(), copy=True)).all()
