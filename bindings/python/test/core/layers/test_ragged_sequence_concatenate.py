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
    left = thor.layers.RaggedNetworkInput(
        network,
        "left",
        thor.DataType.fp32,
        [2],
        max_total_values=6,
        batch_size=3,
        offsets_data_type=offsets_dtype,
        max_values_per_row=3,
    )
    right = thor.layers.RaggedNetworkInput(
        network,
        "right",
        thor.DataType.fp32,
        [2],
        max_total_values=7,
        batch_size=3,
        offsets_data_type=offsets_dtype,
        max_values_per_row=4,
    )
    concatenate = thor.layers.RaggedSequenceConcatenate(network, [left, right])
    output = concatenate.get_feature_output()
    thor.layers.RaggedNetworkOutput(network, "joined", output)
    return network, left, right, concatenate, output


def test_ragged_sequence_concatenate_builds_new_partition_with_summed_capacity():
    network, left, right, _, output = _network("ragged-sequence-concatenate-structure", thor.DataType.uint64)

    assert isinstance(output, thor.RaggedTensor)
    assert output.values.get_dimensions() == [13, 2]
    assert output.offsets.get_dimensions() == [4]
    assert output.offsets.get_data_type() == thor.DataType.uint64
    assert output.offsets != left.offsets
    assert output.offsets != right.offsets

    architecture = json.loads(network.get_architecture_json())
    layer = next(layer for layer in architecture["layers"] if layer["layer_type"] == "ragged_sequence_concatenate")
    assert len(layer["ragged_inputs"]) == 2
    assert layer["ragged_output"]["values"]["dimensions"] == [13, 2]
    assert layer["ragged_output"]["max_values_per_row"] == 7
    assert layer["ragged_output"]["offsets"]["id"] not in {
        layer["ragged_inputs"][0]["offsets"]["id"],
        layer["ragged_inputs"][1]["offsets"]["id"],
    }


@pytest.mark.cuda
@pytest.mark.parametrize(
    "offsets_dtype,np_offsets_dtype",
    [
        (thor.DataType.uint32, np.uint32),
        (thor.DataType.uint64, np.uint64),
    ],
)
def test_ragged_sequence_concatenate_runtime_produces_offsets_ignores_poison_and_reuses_extents(
    offsets_dtype, np_offsets_dtype
):
    network, *_ = _network("ragged-sequence-concatenate-runtime", offsets_dtype)
    placed = network.place(3, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)

    def run(left_active, left_offsets, right_active, right_offsets):
        left_values = np.full((6, 2), np.nan, dtype=np.float32)
        right_values = np.full((7, 2), np.nan, dtype=np.float32)
        left_values[: len(left_active)] = left_active
        right_values[: len(right_active)] = right_active
        return placed.infer(
            {
                "left": _physical_ragged(
                    left_values,
                    np.asarray(left_offsets, dtype=np_offsets_dtype),
                    offsets_dtype,
                    max_values_per_row=3,
                ),
                "right": _physical_ragged(
                    right_values,
                    np.asarray(right_offsets, dtype=np_offsets_dtype),
                    offsets_dtype,
                    max_values_per_row=4,
                ),
            }
        )["joined"]

    left = (1000.0 + np.arange(10, dtype=np.float32)).reshape(5, 2)
    right = (2000.0 + np.arange(8, dtype=np.float32)).reshape(4, 2)
    joined = run(left, [0, 2, 2, 5], right, [0, 1, 4, 4])
    np.testing.assert_array_equal(np.array(joined.offsets.numpy(), copy=True), np.array([0, 3, 6, 9], dtype=np_offsets_dtype))
    expected = np.concatenate([left[0:2], right[0:1], right[1:4], left[2:5]], axis=0)
    np.testing.assert_array_equal(np.array(joined.values.numpy(), copy=True)[:9], expected)

    # Reuse the executable with both larger and differently distributed rows.
    left2 = (3000.0 + np.arange(12, dtype=np.float32)).reshape(6, 2)
    right2 = (4000.0 + np.arange(12, dtype=np.float32)).reshape(6, 2)
    joined2 = run(left2, [0, 1, 3, 6], right2, [0, 2, 5, 6])
    np.testing.assert_array_equal(np.array(joined2.offsets.numpy(), copy=True), np.array([0, 3, 8, 12], dtype=np_offsets_dtype))
    expected2 = np.concatenate([left2[0:1], right2[0:2], left2[1:3], right2[2:5], left2[3:6], right2[5:6]], axis=0)
    np.testing.assert_array_equal(np.array(joined2.values.numpy(), copy=True)[:12], expected2)

    empty = run(np.empty((0, 2), dtype=np.float32), [0, 0, 0, 0], np.empty((0, 2), dtype=np.float32), [0, 0, 0, 0])
    np.testing.assert_array_equal(np.array(empty.offsets.numpy(), copy=True), np.zeros((4,), dtype=np_offsets_dtype))


@pytest.mark.cuda
def test_ragged_sequence_concatenate_save_load_accepts_new_runtime_partitions(tmp_path):
    name = "ragged-sequence-concatenate-save-load"
    network, *_ = _network(name)
    save_dir = tmp_path / "model"
    network.save(str(save_dir), overwrite=False)

    loaded = thor.Network(name)
    loaded.load(str(save_dir))
    placed = loaded.place(3, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)

    left_values = np.full((6, 2), np.nan, dtype=np.float32)
    right_values = np.full((7, 2), np.nan, dtype=np.float32)
    left_active = (5000.0 + np.arange(8, dtype=np.float32)).reshape(4, 2)
    right_active = (6000.0 + np.arange(10, dtype=np.float32)).reshape(5, 2)
    left_values[:4] = left_active
    right_values[:5] = right_active
    output = placed.infer(
        {
            "left": _physical_ragged(
                left_values,
                np.array([0, 0, 2, 4], dtype=np.uint32),
                thor.DataType.uint32,
                max_values_per_row=3,
            ),
            "right": _physical_ragged(
                right_values,
                np.array([0, 2, 2, 5], dtype=np.uint32),
                thor.DataType.uint32,
                max_values_per_row=4,
            ),
        }
    )["joined"]

    np.testing.assert_array_equal(np.array(output.offsets.numpy(), copy=True), np.array([0, 2, 4, 9], dtype=np.uint32))
    expected = np.concatenate([right_active[0:2], left_active[0:2], left_active[2:4], right_active[2:5]], axis=0)
    np.testing.assert_array_equal(np.array(output.values.numpy(), copy=True)[:9], expected)


@pytest.mark.cuda
def test_ragged_sequence_concatenate_shared_partition_uses_one_structural_input_at_runtime():
    network = thor.Network("ragged-sequence-concatenate-shared-partition")
    source = thor.layers.RaggedNetworkInput(
        network,
        "source",
        thor.DataType.fp32,
        [4],
        max_total_values=6,
        batch_size=2,
        max_values_per_row=3,
    )
    first = thor.layers.Slice(network, source, axis=0, start=0, length=2)
    second = thor.layers.Slice(network, source, axis=0, start=2, length=2)
    joined_layer = thor.layers.RaggedSequenceConcatenate(
        network, [first.get_feature_output(), second.get_feature_output()]
    )
    thor.layers.RaggedNetworkOutput(network, "joined", joined_layer.get_feature_output())
    placed = network.place(2, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)

    active = (7000.0 + np.arange(16, dtype=np.float32)).reshape(4, 4)
    values = np.full((6, 4), np.nan, dtype=np.float32)
    values[:4] = active
    output = placed.infer(
        {
            "source": _physical_ragged(
                values,
                np.array([0, 1, 4], dtype=np.uint32),
                thor.DataType.uint32,
                max_values_per_row=3,
            )
        }
    )["joined"]

    np.testing.assert_array_equal(np.array(output.offsets.numpy(), copy=True), np.array([0, 2, 8], dtype=np.uint32))
    expected = np.concatenate(
        [active[0:1, 0:2], active[0:1, 2:4], active[1:4, 0:2], active[1:4, 2:4]],
        axis=0,
    )
    np.testing.assert_array_equal(np.array(output.values.numpy(), copy=True)[:8], expected)
