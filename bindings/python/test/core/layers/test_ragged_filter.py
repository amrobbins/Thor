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


def _physical_ragged(
    values: np.ndarray,
    values_dtype: thor.DataType,
    offsets: np.ndarray,
    offsets_dtype: thor.DataType,
    *,
    max_values_per_row: int,
):
    return thor.physical.PhysicalRaggedTensor(
        _cpu_tensor(values, values_dtype),
        _cpu_tensor(offsets, offsets_dtype),
        max_values_per_row=max_values_per_row,
    )


def _network(name: str, offsets_dtype: thor.DataType = thor.DataType.uint32, *, with_slice: bool = False):
    network = thor.Network(name)
    feature = thor.layers.RaggedNetworkInput(
        network,
        "feature",
        thor.DataType.fp32,
        [2],
        max_total_values=10,
        batch_size=4,
        offsets_data_type=offsets_dtype,
        max_values_per_row=4,
    )
    mask = thor.layers.RaggedNetworkInput(
        network,
        "mask",
        thor.DataType.bool,
        [],
        partition=feature,
    )
    filter_layer = thor.layers.RaggedFilter(network, feature, mask)
    output = filter_layer.get_feature_output()
    if with_slice:
        output = thor.layers.RaggedSequenceSlice(network, output, start=1, length=2).get_feature_output()
    thor.layers.RaggedNetworkOutput(network, "filtered", output)
    return network, feature, mask, filter_layer, output


def test_ragged_filter_builds_fresh_partition_and_requires_shared_boolean_scalar_mask():
    network, feature, mask, filter_layer, output = _network(
        "ragged-filter-structure", thor.DataType.uint64
    )

    assert isinstance(output, thor.RaggedTensor)
    assert filter_layer.get_feature_input() == feature
    assert filter_layer.get_mask_input() == mask
    assert output.values.get_dimensions() == [10, 2]
    assert output.offsets.get_dimensions() == [5]
    assert output.offsets.get_data_type() == thor.DataType.uint64
    assert output.max_values_per_row == 4
    assert output.offsets != feature.offsets

    architecture = json.loads(network.get_architecture_json())
    layer = next(layer for layer in architecture["layers"] if layer["layer_type"] == "ragged_filter")
    assert layer["ragged_mask"]["offsets"]["id"] == layer["ragged_input"]["offsets"]["id"]
    assert layer["ragged_output"]["offsets"]["id"] != layer["ragged_input"]["offsets"]["id"]
    assert layer["ragged_output"]["values"]["dimensions"] == [10, 2]

    with pytest.raises(ValueError, match="exact same offsets"):
        other_mask = thor.layers.RaggedNetworkInput(
            network, "other_mask", thor.DataType.bool, [], max_total_values=10, batch_size=4, max_values_per_row=4
        )
        thor.layers.RaggedFilter(network, feature, other_mask)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "offsets_dtype,np_offsets_dtype",
    [
        (thor.DataType.uint32, np.uint32),
        (thor.DataType.uint64, np.uint64),
    ],
)
def test_ragged_filter_runtime_is_stable_ignores_poison_and_reuses_extents(offsets_dtype, np_offsets_dtype):
    network, *_ = _network("ragged-filter-runtime", offsets_dtype)
    placed = network.place(4, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)

    def run(active, offsets, active_mask):
        values = np.full((10, 2), np.nan, dtype=np.float32)
        values[: len(active)] = active
        mask_values = np.ones((10,), dtype=np.bool_)  # poison inactive tail with true
        mask_values[: len(active_mask)] = np.asarray(active_mask, dtype=np.bool_)
        physical_offsets = np.asarray(offsets, dtype=np_offsets_dtype)
        return placed.infer(
            {
                "feature": _physical_ragged(
                    values,
                    thor.DataType.fp32,
                    physical_offsets,
                    offsets_dtype,
                    max_values_per_row=4,
                ),
                "mask": _cpu_tensor(mask_values, thor.DataType.bool),
            }
        )["filtered"]

    active = (1000.0 + np.arange(16, dtype=np.float32)).reshape(8, 2)
    output = run(active, [0, 4, 5, 8, 8], [1, 0, 1, 1, 0, 1, 0, 1])
    np.testing.assert_array_equal(
        np.array(output.offsets.numpy(), copy=True), np.array([0, 3, 3, 5, 5], dtype=np_offsets_dtype)
    )
    np.testing.assert_array_equal(np.array(output.values.numpy(), copy=True)[:5], active[[0, 2, 3, 5, 7]])

    active2 = (2000.0 + np.arange(20, dtype=np.float32)).reshape(10, 2)
    mask2 = [0, 1, 1, 0, 1, 1, 0, 1, 0, 1]
    output2 = run(active2, [0, 2, 6, 7, 10], mask2)
    np.testing.assert_array_equal(
        np.array(output2.offsets.numpy(), copy=True), np.array([0, 1, 4, 4, 6], dtype=np_offsets_dtype)
    )
    np.testing.assert_array_equal(np.array(output2.values.numpy(), copy=True)[:6], active2[[1, 2, 4, 5, 7, 9]])

    active3 = (3000.0 + np.arange(8, dtype=np.float32)).reshape(4, 2)
    empty = run(active3, [0, 1, 1, 3, 4], [0, 0, 0, 0])
    np.testing.assert_array_equal(np.array(empty.offsets.numpy(), copy=True), np.zeros((5,), dtype=np_offsets_dtype))


@pytest.mark.cuda
def test_ragged_filter_save_load_accepts_new_runtime_mask_and_partition(tmp_path):
    name = "ragged-filter-save-load"
    network, *_ = _network(name)
    save_dir = tmp_path / "model"
    network.save(str(save_dir), overwrite=False)

    loaded = thor.Network(name)
    loaded.load(str(save_dir))
    placed = loaded.place(4, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)

    active = (4000.0 + np.arange(18, dtype=np.float32)).reshape(9, 2)
    values = np.full((10, 2), np.nan, dtype=np.float32)
    values[:9] = active
    mask_values = np.ones((10,), dtype=np.bool_)
    mask_values[:9] = np.array([1, 0, 1, 0, 1, 1, 0, 0, 1], dtype=np.bool_)
    offsets = np.array([0, 3, 3, 7, 9], dtype=np.uint32)
    output = placed.infer(
        {
            "feature": _physical_ragged(
                values, thor.DataType.fp32, offsets, thor.DataType.uint32, max_values_per_row=4
            ),
            "mask": _cpu_tensor(mask_values, thor.DataType.bool),
        }
    )["filtered"]

    np.testing.assert_array_equal(np.array(output.offsets.numpy(), copy=True), np.array([0, 2, 2, 4, 5], dtype=np.uint32))
    np.testing.assert_array_equal(np.array(output.values.numpy(), copy=True)[:5], active[[0, 2, 4, 5, 8]])


@pytest.mark.cuda
def test_ragged_filter_output_composes_with_another_partition_changing_operation():
    network, *_ = _network("ragged-filter-then-slice", with_slice=True)
    placed = network.place(4, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)

    active = (5000.0 + np.arange(20, dtype=np.float32)).reshape(10, 2)
    values = np.array(active, copy=True)
    mask_values = np.array([1, 1, 1, 0, 1, 1, 1, 1, 0, 1], dtype=np.bool_)
    offsets = np.array([0, 4, 6, 7, 10], dtype=np.uint32)
    output = placed.infer(
        {
            "feature": _physical_ragged(
                values, thor.DataType.fp32, offsets, thor.DataType.uint32, max_values_per_row=4
            ),
            "mask": _cpu_tensor(mask_values, thor.DataType.bool),
        }
    )["filtered"]

    # Filter rows become [0,1,2], [4,5], [6], [7,9]. Slice(start=1,length=2)
    # therefore yields [1,2], [5], [], [9].
    np.testing.assert_array_equal(np.array(output.offsets.numpy(), copy=True), np.array([0, 2, 3, 3, 4], dtype=np.uint32))
    np.testing.assert_array_equal(np.array(output.values.numpy(), copy=True)[:4], active[[1, 2, 5, 9]])
