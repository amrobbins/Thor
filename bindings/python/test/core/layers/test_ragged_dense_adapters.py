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


def _physical_ragged(values, offsets, offsets_dtype, *, max_values_per_row):
    return thor.physical.PhysicalRaggedTensor(
        _cpu_tensor(np.asarray(values, dtype=np.float32), thor.DataType.fp32),
        _cpu_tensor(np.asarray(offsets, dtype=thor.physical.numpy_dtypes.from_thor(offsets_dtype)), offsets_dtype),
        max_values_per_row=max_values_per_row,
    )


def _round_trip_network(name: str, offsets_dtype=thor.DataType.uint32):
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
    padded_layer = thor.layers.RaggedToPaddedDense(network, source, padding_value=-123.0)
    padded = padded_layer.get_feature_output()
    restored_layer = thor.layers.PaddedDenseToRagged(network, padded, source)
    restored = restored_layer.get_feature_output()
    thor.layers.RaggedNetworkOutput(network, "restored", restored)
    return network, source, padded_layer, padded, restored_layer, restored


def test_ragged_dense_adapters_build_lossless_partition_reusing_round_trip():
    network, source, padded_layer, padded, restored_layer, restored = _round_trip_network(
        "ragged-dense-adapter-structure", thor.DataType.uint64
    )

    assert padded.get_dimensions() == [4, 2]
    assert padded_layer.padding_value == -123.0
    assert isinstance(restored, thor.RaggedTensor)
    assert restored.values.get_dimensions() == [10, 2]
    assert restored.offsets == source.offsets
    assert restored.offsets.get_data_type() == thor.DataType.uint64
    assert restored_layer.get_partition_input().offsets == source.offsets

    architecture = json.loads(network.get_architecture_json())
    to_dense = next(layer for layer in architecture["layers"] if layer["layer_type"] == "ragged_to_padded_dense")
    to_ragged = next(layer for layer in architecture["layers"] if layer["layer_type"] == "padded_dense_to_ragged")
    assert to_dense["feature_output"]["dimensions"] == [4, 2]
    assert to_ragged["ragged_feature_output"]["offsets"]["id"] == to_ragged["partition_input"]["offsets"]["id"]


def test_ragged_dense_adapters_require_finite_partition_width():
    network = thor.Network("ragged-dense-adapter-validation")
    unbounded = thor.layers.RaggedNetworkInput(
        network,
        "unbounded",
        thor.DataType.fp32,
        [2],
        max_total_values=8,
        batch_size=3,
    )
    with pytest.raises(ValueError, match="max_values_per_row"):
        thor.layers.RaggedToPaddedDense(network, unbounded)

    bounded = thor.layers.RaggedNetworkInput(
        network,
        "bounded",
        thor.DataType.fp32,
        [2],
        max_total_values=8,
        batch_size=3,
        max_values_per_row=4,
    )
    too_narrow = thor.layers.NetworkInput(network, "too_narrow", [3, 2], thor.DataType.fp32).get_feature_output()
    with pytest.raises(ValueError, match="padded width"):
        thor.layers.PaddedDenseToRagged(network, too_narrow, bounded)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "offsets_dtype,np_offsets_dtype",
    [
        (thor.DataType.uint32, np.uint32),
        (thor.DataType.uint64, np.uint64),
    ],
)
def test_ragged_to_padded_dense_pads_rows_ignores_poison_and_reuses_extents(offsets_dtype, np_offsets_dtype):
    network = thor.Network("ragged-to-padded-dense-runtime")
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
    padded = thor.layers.RaggedToPaddedDense(network, source, padding_value=-9.0).get_feature_output()
    thor.layers.NetworkOutput(network, "padded", padded, thor.DataType.fp32)
    placed = network.place(4, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)

    def run(active, offsets):
        values = np.full((10, 2), np.nan, dtype=np.float32)
        values[: len(active)] = active
        result = placed.infer(
            {
                "source": _physical_ragged(
                    values,
                    np.asarray(offsets, dtype=np_offsets_dtype),
                    offsets_dtype,
                    max_values_per_row=4,
                )
            }
        )["padded"]
        return np.array(result.numpy(), copy=True)

    active = (1000.0 + np.arange(16, dtype=np.float32)).reshape(8, 2)
    dense = run(active, [0, 4, 5, 8, 8])
    expected = np.full((4, 4, 2), -9.0, dtype=np.float32)
    expected[0, :4] = active[:4]
    expected[1, :1] = active[4:5]
    expected[2, :3] = active[5:8]
    np.testing.assert_array_equal(dense, expected)

    active2 = (2000.0 + np.arange(20, dtype=np.float32)).reshape(10, 2)
    dense2 = run(active2, [0, 2, 6, 7, 10])
    expected2 = np.full((4, 4, 2), -9.0, dtype=np.float32)
    expected2[0, :2] = active2[:2]
    expected2[1, :4] = active2[2:6]
    expected2[2, :1] = active2[6:7]
    expected2[3, :3] = active2[7:10]
    np.testing.assert_array_equal(dense2, expected2)

    empty = run(np.empty((0, 2), dtype=np.float32), [0, 0, 0, 0, 0])
    np.testing.assert_array_equal(empty, np.full((4, 4, 2), -9.0, dtype=np.float32))


@pytest.mark.cuda
def test_ragged_dense_adapter_round_trip_and_save_load_accept_new_partition(tmp_path):
    name = "ragged-dense-adapter-round-trip-save-load"
    network, *_ = _round_trip_network(name, thor.DataType.uint64)
    save_dir = tmp_path / "model"
    network.save(str(save_dir), overwrite=False)

    loaded = thor.Network(name)
    loaded.load(str(save_dir))
    placed = loaded.place(4, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)

    active = (3000.0 + np.arange(18, dtype=np.float32)).reshape(9, 2)
    values = np.full((10, 2), np.nan, dtype=np.float32)
    values[:9] = active
    output = placed.infer(
        {
            "source": _physical_ragged(
                values,
                np.array([0, 3, 3, 7, 9], dtype=np.uint64),
                thor.DataType.uint64,
                max_values_per_row=4,
            )
        }
    )["restored"]

    np.testing.assert_array_equal(np.array(output.offsets.numpy(), copy=True), np.array([0, 3, 3, 7, 9], dtype=np.uint64))
    np.testing.assert_array_equal(np.array(output.values.numpy(), copy=True)[:9], active)
