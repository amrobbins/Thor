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


def _physical_ragged(values: np.ndarray, offsets: np.ndarray, offsets_dtype: thor.DataType):
    return thor.physical.PhysicalRaggedTensor(
        _cpu_tensor(values, thor.DataType.fp32),
        _cpu_tensor(offsets, offsets_dtype),
    )


def _network(name: str, offsets_dtype: thor.DataType = thor.DataType.uint32):
    network = thor.Network(name)
    tokens = thor.layers.RaggedNetworkInput(
        network,
        "tokens",
        thor.DataType.fp32,
        [2, 3, 4],
        max_total_values=8,
        batch_size=3,
        offsets_data_type=offsets_dtype,
    )
    reshape = thor.layers.Reshape(network, tokens, [4, 6])
    flatten = thor.layers.Flatten(network, tokens, 2)
    transpose = thor.layers.Transpose(network, tokens)
    thor.layers.RaggedNetworkOutput(network, "reshape", reshape.get_feature_output())
    thor.layers.RaggedNetworkOutput(network, "flatten", flatten.get_feature_output())
    thor.layers.RaggedNetworkOutput(network, "transpose", transpose.get_feature_output())
    return network, tokens, reshape, flatten, transpose


def test_ragged_shape_ops_preserve_partition_and_transform_only_trailing_shape():
    network, tokens, reshape, flatten, transpose = _network("ragged-shape-ops-structure", thor.DataType.uint64)

    reshaped = reshape.get_feature_output()
    flattened = flatten.get_feature_output()
    transposed = transpose.get_feature_output()
    assert isinstance(reshaped, thor.RaggedTensor)
    assert isinstance(flattened, thor.RaggedTensor)
    assert isinstance(transposed, thor.RaggedTensor)
    assert reshaped.offsets == tokens.offsets
    assert flattened.offsets == tokens.offsets
    assert transposed.offsets == tokens.offsets
    assert reshaped.values.get_dimensions() == [8, 4, 6]
    assert flattened.values.get_dimensions() == [8, 2, 12]
    assert transposed.values.get_dimensions() == [8, 2, 4, 3]

    architecture = json.loads(network.get_architecture_json())
    by_type = {layer["layer_type"]: layer for layer in architecture["layers"] if layer["layer_type"] in {"reshape", "flatten", "transpose"}}
    assert set(by_type) == {"reshape", "flatten", "transpose"}
    assert all(layer.get("use_ragged") is True for layer in by_type.values())

    reject = thor.Network("ragged-shape-ops-reject")
    vector = thor.layers.RaggedNetworkInput(
        reject,
        "vector",
        thor.DataType.fp32,
        [8],
        max_total_values=4,
        batch_size=2,
    )
    with pytest.raises(Exception, match="elements per packed value|number of elements"):
        thor.layers.Reshape(reject, vector, [7])
    with pytest.raises(Exception, match="num_output_dimensions"):
        thor.layers.Flatten(reject, vector, 1)
    with pytest.raises(Exception, match="at least two trailing"):
        thor.layers.Transpose(reject, vector)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "offsets_dtype,np_offsets_dtype",
    [
        (thor.DataType.uint32, np.uint32),
        (thor.DataType.uint64, np.uint64),
    ],
)
def test_ragged_shape_ops_runtime_ignore_poison_and_reuse_changing_extents(offsets_dtype, np_offsets_dtype):
    network, *_ = _network("ragged-shape-ops-runtime", offsets_dtype)
    placed = network.place(3, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)

    def run(active_values: np.ndarray, offsets: np.ndarray):
        packed = np.full((8, 2, 3, 4), np.nan, dtype=np.float32)
        packed[: active_values.shape[0]] = active_values
        return placed.infer({"tokens": _physical_ragged(packed, offsets, offsets_dtype)})

    short = np.arange(3 * 24, dtype=np.float32).reshape(3, 2, 3, 4)
    short_offsets = np.array([0, 1, 1, 3], dtype=np_offsets_dtype)
    short_outputs = run(short, short_offsets)
    np.testing.assert_array_equal(
        np.array(short_outputs["reshape"].values.numpy(), copy=True)[:3],
        short.reshape(3, 4, 6),
    )
    np.testing.assert_array_equal(
        np.array(short_outputs["flatten"].values.numpy(), copy=True)[:3],
        short.reshape(3, 2, 12),
    )
    np.testing.assert_array_equal(
        np.array(short_outputs["transpose"].values.numpy(), copy=True)[:3],
        short.transpose(0, 1, 3, 2),
    )
    for name in ("reshape", "flatten", "transpose"):
        np.testing.assert_array_equal(np.array(short_outputs[name].offsets.numpy(), copy=True), short_offsets)

    # Reuse the same executable with a larger logical prefix. If transpose cached
    # the old runtime extent, rows 3..5 would remain stale/undefined here.
    long = (1000.0 + np.arange(6 * 24, dtype=np.float32)).reshape(6, 2, 3, 4)
    long_offsets = np.array([0, 2, 4, 6], dtype=np_offsets_dtype)
    long_outputs = run(long, long_offsets)
    np.testing.assert_array_equal(
        np.array(long_outputs["transpose"].values.numpy(), copy=True)[:6],
        long.transpose(0, 1, 3, 2),
    )

    # Then shrink again, including an empty middle row, to catch retained stale
    # active-extent state in the opposite direction.
    short_again = (2000.0 + np.arange(2 * 24, dtype=np.float32)).reshape(2, 2, 3, 4)
    short_again_offsets = np.array([0, 1, 1, 2], dtype=np_offsets_dtype)
    short_again_outputs = run(short_again, short_again_offsets)
    np.testing.assert_array_equal(
        np.array(short_again_outputs["transpose"].values.numpy(), copy=True)[:2],
        short_again.transpose(0, 1, 3, 2),
    )

    # An all-empty partition is valid even though the complete packed capacity is
    # NaN poison. The logical operation must not touch that inactive storage.
    empty = run(np.empty((0, 2, 3, 4), dtype=np.float32), np.zeros((4,), dtype=np_offsets_dtype))
    for name in ("reshape", "flatten", "transpose"):
        assert int(np.array(empty[name].offsets.numpy(), copy=False)[-1]) == 0


@pytest.mark.cuda
def test_ragged_shape_ops_save_load_accepts_different_runtime_partition(tmp_path):
    name = "ragged-shape-ops-save-load"
    network, *_ = _network(name)
    save_dir = tmp_path / "model"
    network.save(str(save_dir), overwrite=False)

    loaded = thor.Network(name)
    loaded.load(str(save_dir))
    placed = loaded.place(3, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)

    active = (3000.0 + np.arange(5 * 24, dtype=np.float32)).reshape(5, 2, 3, 4)
    values = np.full((8, 2, 3, 4), np.nan, dtype=np.float32)
    values[:5] = active
    offsets = np.array([0, 2, 2, 5], dtype=np.uint32)
    outputs = placed.infer({"tokens": _physical_ragged(values, offsets, thor.DataType.uint32)})

    np.testing.assert_array_equal(
        np.array(outputs["reshape"].values.numpy(), copy=True)[:5],
        active.reshape(5, 4, 6),
    )
    np.testing.assert_array_equal(
        np.array(outputs["flatten"].values.numpy(), copy=True)[:5],
        active.reshape(5, 2, 12),
    )
    np.testing.assert_array_equal(
        np.array(outputs["transpose"].values.numpy(), copy=True)[:5],
        active.transpose(0, 1, 3, 2),
    )
    for output in outputs.values():
        np.testing.assert_array_equal(np.array(output.offsets.numpy(), copy=True), offsets)
