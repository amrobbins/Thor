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


def _physical_ragged(values, values_dtype, offsets, offsets_dtype, *, max_values_per_row):
    return thor.physical.PhysicalRaggedTensor(
        _cpu_tensor(values, values_dtype),
        _cpu_tensor(offsets, offsets_dtype),
        max_values_per_row=max_values_per_row,
    )


def _network(name, source_offsets_dtype=thor.DataType.uint32, indices_offsets_dtype=thor.DataType.uint64):
    network = thor.Network(name)
    source = thor.layers.RaggedNetworkInput(
        network,
        "source",
        thor.DataType.fp32,
        [2],
        max_total_values=10,
        batch_size=4,
        offsets_data_type=source_offsets_dtype,
        max_values_per_row=4,
    )
    indices = thor.layers.RaggedNetworkInput(
        network,
        "indices",
        thor.DataType.uint32,
        [],
        max_total_values=9,
        batch_size=4,
        offsets_data_type=indices_offsets_dtype,
        max_values_per_row=3,
    )
    gather = thor.layers.RaggedGather(network, source, indices)
    output = gather.get_feature_output()
    thor.layers.RaggedNetworkOutput(network, "gathered", output)
    return network, source, indices, gather, output


def test_ragged_gather_uses_indices_partition_and_source_value_geometry():
    network, source, indices, gather, output = _network("ragged-gather-structure")

    assert isinstance(output, thor.RaggedTensor)
    assert gather.get_source_input() == source
    assert gather.get_indices_input() == indices
    assert output.values.get_dimensions() == [9, 2]
    assert output.offsets == indices.offsets
    assert output.offsets != source.offsets
    assert output.max_values_per_row == 3

    architecture = json.loads(network.get_architecture_json())
    layer = next(layer for layer in architecture["layers"] if layer["layer_type"] == "ragged_gather")
    assert layer["ragged_output"]["offsets"]["id"] == layer["ragged_indices"]["offsets"]["id"]
    assert layer["ragged_output"]["values"]["dimensions"] == [9, 2]

    with pytest.raises(ValueError, match="UINT32 or UINT64"):
        bad_indices = thor.layers.RaggedNetworkInput(
            network,
            "bad_indices",
            thor.DataType.fp32,
            [],
            max_total_values=9,
            batch_size=4,
            max_values_per_row=3,
        )
        thor.layers.RaggedGather(network, source, bad_indices)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "source_offsets_dtype,np_source_offsets_dtype,indices_offsets_dtype,np_indices_offsets_dtype",
    [
        (thor.DataType.uint32, np.uint32, thor.DataType.uint32, np.uint32),
        (thor.DataType.uint32, np.uint32, thor.DataType.uint64, np.uint64),
        (thor.DataType.uint64, np.uint64, thor.DataType.uint32, np.uint32),
        (thor.DataType.uint64, np.uint64, thor.DataType.uint64, np.uint64),
    ],
)
def test_ragged_gather_runtime_row_local_duplicates_poison_and_reuse(
    source_offsets_dtype,
    np_source_offsets_dtype,
    indices_offsets_dtype,
    np_indices_offsets_dtype,
):
    network, *_ = _network("ragged-gather-runtime", source_offsets_dtype, indices_offsets_dtype)
    placed = network.place(4, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)

    def run(active_source, source_offsets, active_indices, indices_offsets):
        source_values = np.full((10, 2), np.nan, dtype=np.float32)
        source_values[: len(active_source)] = active_source
        indices_values = np.full((9,), np.uint32(0xFFFFFFFF), dtype=np.uint32)
        indices_values[: len(active_indices)] = np.asarray(active_indices, dtype=np.uint32)
        return placed.infer(
            {
                "source": _physical_ragged(
                    source_values,
                    thor.DataType.fp32,
                    np.asarray(source_offsets, dtype=np_source_offsets_dtype),
                    source_offsets_dtype,
                    max_values_per_row=4,
                ),
                "indices": _physical_ragged(
                    indices_values,
                    thor.DataType.uint32,
                    np.asarray(indices_offsets, dtype=np_indices_offsets_dtype),
                    indices_offsets_dtype,
                    max_values_per_row=3,
                ),
            }
        )["gathered"]

    source = (1000.0 + np.arange(18, dtype=np.float32)).reshape(9, 2)
    gathered = run(source, [0, 3, 5, 5, 9], [2, 0, 2, 1, 0, 3, 1, 3], [0, 3, 5, 5, 8])
    np.testing.assert_array_equal(
        np.array(gathered.offsets.numpy(), copy=True),
        np.array([0, 3, 5, 5, 8], dtype=np_indices_offsets_dtype),
    )
    np.testing.assert_array_equal(np.array(gathered.values.numpy(), copy=True)[:8], source[[2, 0, 2, 4, 3, 8, 6, 8]])

    source2 = (2000.0 + np.arange(20, dtype=np.float32)).reshape(10, 2)
    gathered2 = run(source2, [0, 2, 6, 7, 10], [1, 0, 3, 1, 0, 2, 0], [0, 1, 4, 5, 7])
    np.testing.assert_array_equal(
        np.array(gathered2.offsets.numpy(), copy=True),
        np.array([0, 1, 4, 5, 7], dtype=np_indices_offsets_dtype),
    )
    np.testing.assert_array_equal(np.array(gathered2.values.numpy(), copy=True)[:7], source2[[1, 2, 5, 3, 6, 9, 7]])


@pytest.mark.cuda
def test_ragged_gather_save_load_preserves_q_alias_and_accepts_new_partitions(tmp_path):
    name = "ragged-gather-save-load"
    network, *_ = _network(name)
    save_dir = tmp_path / "model"
    network.save(str(save_dir), overwrite=False)

    loaded = thor.Network(name)
    loaded.load(str(save_dir))
    architecture = json.loads(loaded.get_architecture_json())
    layer = next(layer for layer in architecture["layers"] if layer["layer_type"] == "ragged_gather")
    assert layer["ragged_output"]["offsets"]["id"] == layer["ragged_indices"]["offsets"]["id"]

    placed = loaded.place(4, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)
    source = (3000.0 + np.arange(20, dtype=np.float32)).reshape(10, 2)
    indices_values = np.full((9,), np.uint32(0xFFFFFFFF), dtype=np.uint32)
    indices_values[:6] = np.array([1, 0, 2, 0, 1, 0], dtype=np.uint32)
    result = placed.infer(
        {
            "source": _physical_ragged(
                source,
                thor.DataType.fp32,
                np.array([0, 2, 5, 8, 10], dtype=np.uint32),
                thor.DataType.uint32,
                max_values_per_row=4,
            ),
            "indices": _physical_ragged(
                indices_values,
                thor.DataType.uint32,
                np.array([0, 2, 3, 5, 6], dtype=np.uint64),
                thor.DataType.uint64,
                max_values_per_row=3,
            ),
        }
    )["gathered"]
    np.testing.assert_array_equal(np.array(result.offsets.numpy(), copy=True), np.array([0, 2, 3, 5, 6], dtype=np.uint64))
    np.testing.assert_array_equal(np.array(result.values.numpy(), copy=True)[:6], source[[1, 0, 4, 5, 6, 8]])


@pytest.mark.cuda
def test_ragged_gather_accepts_shared_partition_indices_without_duplicate_offsets_input():
    network = thor.Network("ragged-gather-shared-partition")
    source = thor.layers.RaggedNetworkInput(
        network,
        "source",
        thor.DataType.fp32,
        [1],
        max_total_values=6,
        batch_size=3,
        max_values_per_row=3,
    )
    indices = thor.layers.RaggedNetworkInput(network, "indices", thor.DataType.uint32, [], partition=source)
    output = thor.layers.RaggedGather(network, source, indices).get_feature_output()
    thor.layers.RaggedNetworkOutput(network, "gathered", output)
    placed = network.place(3, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)

    values = np.arange(6, dtype=np.float32).reshape(6, 1) + 10
    offsets = np.array([0, 2, 5, 5], dtype=np.uint32)
    packed_indices = np.array([1, 0, 2, 0, 1, 99], dtype=np.uint32)
    result = placed.infer(
        {
            "source": _physical_ragged(
                values,
                thor.DataType.fp32,
                offsets,
                thor.DataType.uint32,
                max_values_per_row=3,
            ),
            "indices": _cpu_tensor(packed_indices, thor.DataType.uint32),
        }
    )["gathered"]
    np.testing.assert_array_equal(np.array(result.offsets.numpy(), copy=True), offsets)
    np.testing.assert_array_equal(np.array(result.values.numpy(), copy=True)[:5, 0], values[[1, 0, 4, 2, 3], 0])
