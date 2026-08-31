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
    offsets: np.ndarray,
    offsets_dtype: thor.DataType,
    *,
    max_values_per_row: int,
):
    return thor.physical.PhysicalRaggedTensor(
        _cpu_tensor(values, thor.DataType.fp32),
        _cpu_tensor(offsets, offsets_dtype),
        max_values_per_row=max_values_per_row,
    )


def _network(name: str, offsets_dtype: thor.DataType = thor.DataType.uint32):
    network = thor.Network(name)
    source = thor.layers.RaggedNetworkInput(
        network,
        "source",
        thor.DataType.fp32,
        [2],
        max_total_values=10,
        batch_size=4,
        offsets_data_type=offsets_dtype,
        max_values_per_row=4,
    )
    slice_layer = thor.layers.RaggedSequenceSlice(network, source, start=1, length=2)
    output = slice_layer.get_feature_output()
    thor.layers.RaggedNetworkOutput(network, "sliced", output)
    return network, source, slice_layer, output


def test_ragged_sequence_slice_builds_new_partition_and_compact_capacity():
    network, source, slice_layer, output = _network("ragged-sequence-slice-structure", thor.DataType.uint64)

    assert isinstance(output, thor.RaggedTensor)
    assert slice_layer.start == 1
    assert slice_layer.length == 2
    assert output.values.get_dimensions() == [8, 2]
    assert output.offsets.get_dimensions() == [5]
    assert output.offsets.get_data_type() == thor.DataType.uint64
    assert output.offsets != source.offsets

    architecture = json.loads(network.get_architecture_json())
    layer = next(layer for layer in architecture["layers"] if layer["layer_type"] == "ragged_sequence_slice")
    assert layer["start"] == 1
    assert layer["length"] == 2
    assert layer["ragged_output"]["values"]["dimensions"] == [8, 2]
    assert layer["ragged_output"]["max_values_per_row"] == 2
    assert layer["ragged_output"]["offsets"]["id"] != layer["ragged_input"]["offsets"]["id"]


def test_ragged_sequence_slice_rejects_zero_length():
    network = thor.Network("ragged-sequence-slice-zero-length")
    source = thor.layers.RaggedNetworkInput(
        network,
        "source",
        thor.DataType.fp32,
        [2],
        max_total_values=5,
        batch_size=2,
        max_values_per_row=3,
    )
    with pytest.raises(ValueError, match="length must be greater than zero"):
        thor.layers.RaggedSequenceSlice(network, source, start=0, length=0)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "offsets_dtype,np_offsets_dtype",
    [
        (thor.DataType.uint32, np.uint32),
        (thor.DataType.uint64, np.uint64),
    ],
)
def test_ragged_sequence_slice_runtime_clips_rows_ignores_poison_and_reuses_extents(
    offsets_dtype, np_offsets_dtype
):
    network, *_ = _network("ragged-sequence-slice-runtime", offsets_dtype)
    placed = network.place(4, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)

    def run(active, offsets):
        values = np.full((10, 2), np.nan, dtype=np.float32)
        values[: len(active)] = active
        return placed.infer(
            {
                "source": _physical_ragged(
                    values,
                    np.asarray(offsets, dtype=np_offsets_dtype),
                    offsets_dtype,
                    max_values_per_row=4,
                )
            }
        )["sliced"]

    active = (1000.0 + np.arange(16, dtype=np.float32)).reshape(8, 2)
    sliced = run(active, [0, 4, 5, 8, 8])
    np.testing.assert_array_equal(
        np.array(sliced.offsets.numpy(), copy=True), np.array([0, 2, 2, 4, 4], dtype=np_offsets_dtype)
    )
    np.testing.assert_array_equal(np.array(sliced.values.numpy(), copy=True)[:4], active[[1, 2, 6, 7]])

    active2 = (2000.0 + np.arange(20, dtype=np.float32)).reshape(10, 2)
    sliced2 = run(active2, [0, 2, 6, 7, 10])
    np.testing.assert_array_equal(
        np.array(sliced2.offsets.numpy(), copy=True), np.array([0, 1, 3, 3, 5], dtype=np_offsets_dtype)
    )
    np.testing.assert_array_equal(np.array(sliced2.values.numpy(), copy=True)[:5], active2[[1, 3, 4, 8, 9]])

    active3 = (3000.0 + np.arange(4, dtype=np.float32)).reshape(2, 2)
    empty = run(active3, [0, 0, 1, 1, 2])
    np.testing.assert_array_equal(
        np.array(empty.offsets.numpy(), copy=True), np.zeros((5,), dtype=np_offsets_dtype)
    )


@pytest.mark.cuda
def test_ragged_sequence_slice_save_load_accepts_new_runtime_partition(tmp_path):
    name = "ragged-sequence-slice-save-load"
    network, *_ = _network(name)
    save_dir = tmp_path / "model"
    network.save(str(save_dir), overwrite=False)

    loaded = thor.Network(name)
    loaded.load(str(save_dir))
    placed = loaded.place(4, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)

    active = (4000.0 + np.arange(18, dtype=np.float32)).reshape(9, 2)
    values = np.full((10, 2), np.nan, dtype=np.float32)
    values[:9] = active
    output = placed.infer(
        {
            "source": _physical_ragged(
                values,
                np.array([0, 3, 3, 7, 9], dtype=np.uint32),
                thor.DataType.uint32,
                max_values_per_row=4,
            )
        }
    )["sliced"]

    np.testing.assert_array_equal(np.array(output.offsets.numpy(), copy=True), np.array([0, 2, 2, 4, 5], dtype=np.uint32))
    np.testing.assert_array_equal(np.array(output.values.numpy(), copy=True)[:5], active[[1, 2, 4, 5, 8]])


@pytest.mark.cuda
def test_partition_changing_chain_publishes_host_metadata_for_active_prefix_consumers():
    network = thor.Network("ragged-partition-changing-host-metadata")
    left = thor.layers.RaggedNetworkInput(
        network,
        "left",
        thor.DataType.fp32,
        [2],
        max_total_values=5,
        batch_size=3,
        max_values_per_row=3,
    )
    right = thor.layers.RaggedNetworkInput(
        network,
        "right",
        thor.DataType.fp32,
        [2],
        max_total_values=6,
        batch_size=3,
        max_values_per_row=3,
    )
    joined = thor.layers.RaggedSequenceConcatenate(network, [left, right]).get_feature_output()
    sliced = thor.layers.RaggedSequenceSlice(network, joined, start=1, length=2).get_feature_output()
    checked = thor.layers.FiniteCheck(
        network,
        sliced,
        tensor_label="partition_changed",
        check_backward=False,
    ).get_feature_output()
    thor.layers.RaggedNetworkOutput(network, "output", checked)
    placed = network.place(3, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)

    left_active = (5000.0 + np.arange(8, dtype=np.float32)).reshape(4, 2)
    right_active = (6000.0 + np.arange(8, dtype=np.float32)).reshape(4, 2)
    left_values = np.full((5, 2), np.nan, dtype=np.float32)
    right_values = np.full((6, 2), np.nan, dtype=np.float32)
    left_values[:4] = left_active
    right_values[:4] = right_active
    output = placed.infer(
        {
            "left": _physical_ragged(
                left_values,
                np.array([0, 2, 2, 4], dtype=np.uint32),
                thor.DataType.uint32,
                max_values_per_row=3,
            ),
            "right": _physical_ragged(
                right_values,
                np.array([0, 1, 3, 4], dtype=np.uint32),
                thor.DataType.uint32,
                max_values_per_row=3,
            ),
        }
    )["output"]

    np.testing.assert_array_equal(np.array(output.offsets.numpy(), copy=True), np.array([0, 2, 3, 5], dtype=np.uint32))
    expected = np.stack([left_active[1], right_active[0], right_active[2], left_active[3], right_active[3]])
    np.testing.assert_array_equal(np.array(output.values.numpy(), copy=True)[:5], expected)
