import json

import numpy as np
import pytest

import thor


def _cpu_tensor(values: np.ndarray, dtype: thor.DataType) -> thor.physical.PhysicalTensor:
    values = np.asarray(values, dtype=thor.physical.numpy_dtypes.from_thor(dtype), order="C")
    placement = thor.physical.Placement(thor.physical.DeviceType.cpu, 0)
    descriptor = thor.physical.PhysicalTensor.Descriptor(dtype, list(values.shape))
    tensor = thor.physical.PhysicalTensor(placement, descriptor)
    tensor.numpy()[...] = values
    return tensor


def _physical_ragged(values: np.ndarray, offsets: np.ndarray) -> thor.physical.PhysicalRaggedTensor:
    return thor.physical.PhysicalRaggedTensor(
        _cpu_tensor(values, thor.DataType.fp32),
        _cpu_tensor(offsets, thor.DataType.uint32),
    )


def _channel_reduction_network(name: str):
    network = thor.Network(name)
    history = thor.layers.RaggedNetworkInput(
        network,
        "history",
        thor.DataType.fp32,
        [3],
        max_total_values=9,
        batch_size=3,
        offsets_data_type=thor.DataType.uint32,
    )
    channel0_layer = thor.layers.Slice(network, history, axis=0, start=0, length=1)
    channel0 = channel0_layer.get_feature_output()
    reductions = {}
    for reduction_name, reduction_type in (
        ("sum", thor.layers.SegmentedReduction.Type.sum),
        ("mean", thor.layers.SegmentedReduction.Type.mean),
        ("min", thor.layers.SegmentedReduction.Type.min),
        ("max", thor.layers.SegmentedReduction.Type.max),
    ):
        layer = thor.layers.SegmentedReduction(network, channel0, reduction_type)
        reductions[reduction_name] = layer
        thor.layers.NetworkOutput(network, reduction_name, layer.get_feature_output(), thor.DataType.fp32)
    return network, history, channel0_layer, channel0, reductions


def test_ragged_slice_preserves_partition_and_segmented_reduction_returns_dense_per_example_shape():
    network, history, channel0_layer, channel0, reductions = _channel_reduction_network("ragged-slice-segment-shapes")

    assert channel0_layer.get_use_ragged()
    assert isinstance(channel0, thor.RaggedTensor)
    assert channel0.values.get_dimensions() == [9, 1]
    assert channel0.offsets == history.offsets
    for layer in reductions.values():
        output = layer.get_feature_output()
        assert isinstance(output, thor.Tensor)
        assert output.get_dimensions() == [1]

    architecture = json.loads(network.get_architecture_json())
    slice_json = next(layer for layer in architecture["layers"] if layer["layer_type"] == "slice")
    assert slice_json["use_ragged"] is True
    assert slice_json["ragged_feature_input"]["offsets"]["id"] == slice_json["ragged_feature_output"]["offsets"]["id"]
    assert {layer["reduction_type"] for layer in architecture["layers"] if layer["layer_type"] == "segmented_reduction"} == {
        "sum",
        "mean",
        "min",
        "max",
    }


@pytest.mark.cuda
def test_channel_zero_segmented_reductions_handle_unequal_lengths_empty_row_and_poisoned_capacity():
    network, _, _, _, _ = _channel_reduction_network("ragged-channel-zero-segmented-reductions")
    placed = network.place(3, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)

    # offsets [0,3,3,7] gives lengths 3,0,4. Rows 7 and 8 are inactive and
    # contain values chosen to corrupt both min and max if capacity leaks in.
    values = np.array(
        [
            [1.0, 10.0, 100.0],
            [3.0, 20.0, 200.0],
            [2.0, 30.0, 300.0],
            [5.0, 40.0, 400.0],
            [4.0, 50.0, 500.0],
            [8.0, 60.0, 600.0],
            [6.0, 70.0, 700.0],
            [-1000.0, 8000.0, 80000.0],
            [1000.0, -9000.0, -90000.0],
        ],
        dtype=np.float32,
    )
    offsets = np.array([0, 3, 3, 7], dtype=np.uint32)
    outputs = placed.infer({"history": _physical_ragged(values, offsets)})

    sums = np.array(outputs["sum"].numpy(), copy=True)
    means = np.array(outputs["mean"].numpy(), copy=True)
    mins = np.array(outputs["min"].numpy(), copy=True)
    maxs = np.array(outputs["max"].numpy(), copy=True)
    assert sums.shape == (3, 1)
    assert means.shape == (3, 1)
    np.testing.assert_allclose(sums, np.array([[6.0], [0.0], [23.0]], dtype=np.float32), rtol=0, atol=1e-6)
    np.testing.assert_allclose(means, np.array([[2.0], [0.0], [5.75]], dtype=np.float32), rtol=0, atol=1e-6)
    # Empty-row min/max identities are intentionally not part of this API assertion;
    # validate the populated rows, including protection from poisoned packed capacity.
    np.testing.assert_allclose(mins[[0, 2]], np.array([[1.0], [4.0]], dtype=np.float32), rtol=0, atol=1e-6)
    np.testing.assert_allclose(maxs[[0, 2]], np.array([[3.0], [8.0]], dtype=np.float32), rtol=0, atol=1e-6)


@pytest.mark.cuda
def test_ragged_slice_and_segment_mean_save_load_preserve_execution(tmp_path):
    name = "ragged-slice-segment-mean-save-load"
    network = thor.Network(name)
    history = thor.layers.RaggedNetworkInput(
        network,
        "history",
        thor.DataType.fp32,
        [3],
        max_total_values=9,
        batch_size=3,
        offsets_data_type=thor.DataType.uint32,
    )
    channel0 = thor.layers.Slice(network, history, axis=0, start=0, length=1).get_feature_output()
    mean = thor.layers.SegmentedReduction(network, channel0, thor.layers.SegmentedReduction.Type.mean).get_feature_output()
    thor.layers.NetworkOutput(network, "mean", mean, thor.DataType.fp32)

    save_dir = tmp_path / "model"
    network.save(str(save_dir), overwrite=False)
    loaded = thor.Network(name)
    loaded.load(str(save_dir))
    placed = loaded.place(3, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)

    values = np.arange(27, dtype=np.float32).reshape(9, 3)
    values[7:] = np.float32(9999.0)
    offsets = np.array([0, 3, 3, 7], dtype=np.uint32)
    output = np.array(placed.infer({"history": _physical_ragged(values, offsets)})["mean"].numpy(), copy=True)
    expected = np.array([[3.0], [0.0], [13.5]], dtype=np.float32)
    np.testing.assert_allclose(output, expected, rtol=0, atol=1e-6)
