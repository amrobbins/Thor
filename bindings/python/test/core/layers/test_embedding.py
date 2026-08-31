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
        _cpu_tensor(values, thor.DataType.uint32),
        _cpu_tensor(offsets, offsets_dtype),
        max_values_per_row=5,
    )


def _network(name: str, offsets_dtype: thor.DataType = thor.DataType.uint32):
    network = thor.Network(name)
    tokens = thor.layers.RaggedNetworkInput(
        network,
        "tokens",
        thor.DataType.uint32,
        [],
        max_total_values=7,
        batch_size=3,
        offsets_data_type=offsets_dtype,
        max_values_per_row=5,
    )
    embedding = thor.layers.Embedding(
        network,
        tokens,
        vocabulary_size=16,
        embedding_dim=4,
        weights_data_type=thor.DataType.fp32,
    )
    thor.layers.RaggedNetworkOutput(network, "output", embedding.get_feature_output())
    return network, tokens, embedding


def test_ragged_embedding_preserves_partition_and_architecture_metadata():
    network, tokens, embedding = _network("ragged-embedding-structure", thor.DataType.uint64)
    output = embedding.get_feature_output()
    assert isinstance(output, thor.RaggedTensor)
    assert embedding.use_ragged is True
    assert output.offsets == tokens.offsets
    assert output.values.get_dimensions() == [7, 4]
    assert output.batch_size == 3
    assert output.max_total_values == 7
    assert output.max_values_per_row == 5

    architecture = json.loads(network.get_architecture_json())
    embedding_arch = next(layer for layer in architecture["layers"] if layer["layer_type"] == "embedding")
    assert embedding_arch["version"] == "1.1.0"
    assert embedding_arch["use_ragged"] is True
    assert embedding_arch["ragged_inputs"][0]["offsets"]["id"] == embedding_arch["ragged_outputs"][0]["offsets"]["id"]


@pytest.mark.cuda
@pytest.mark.parametrize(
    "offsets_dtype,np_offsets_dtype",
    [
        (thor.DataType.uint32, np.uint32),
        (thor.DataType.uint64, np.uint64),
    ],
)
def test_ragged_embedding_runtime_reuses_changing_active_extent(offsets_dtype, np_offsets_dtype):
    network, *_ = _network("ragged-embedding-runtime", offsets_dtype)
    placed = network.place(3, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)

    def run(active_indices, offsets, tail):
        packed = np.asarray(list(active_indices) + list(tail), dtype=np.uint32)
        assert packed.shape == (7,)
        outputs = placed.infer({
            "tokens": _physical_ragged(packed, np.asarray(offsets, dtype=np_offsets_dtype), offsets_dtype),
        })
        return outputs["output"]

    short = run([1, 2, 1], [0, 1, 1, 3], [np.iinfo(np.uint32).max] * 4)
    short_values = np.array(short.values.numpy(), copy=True)[:3]
    assert np.all(np.isfinite(short_values))
    np.testing.assert_allclose(short_values[0], short_values[2], rtol=0, atol=0)
    np.testing.assert_array_equal(np.array(short.offsets.numpy(), copy=True), np.array([0, 1, 1, 3], dtype=np_offsets_dtype))

    # Grow the logical prefix on the same placed executable. Repeated ids give
    # a weight-independent correctness check for rows written after the old extent.
    long = run([3, 4, 5, 3, 4, 5], [0, 2, 4, 6], [np.iinfo(np.uint32).max])
    long_values = np.array(long.values.numpy(), copy=True)[:6]
    assert np.all(np.isfinite(long_values))
    np.testing.assert_allclose(long_values[0], long_values[3], rtol=0, atol=0)
    np.testing.assert_allclose(long_values[1], long_values[4], rtol=0, atol=0)
    np.testing.assert_allclose(long_values[2], long_values[5], rtol=0, atol=0)

    # Then shrink and finally go all-empty. The inactive capacity contains
    # deliberately out-of-range poison indices throughout.
    short_again = run([7], [0, 0, 1, 1], [np.iinfo(np.uint32).max] * 6)
    assert np.all(np.isfinite(np.array(short_again.values.numpy(), copy=True)[:1]))
    empty = run([], [0, 0, 0, 0], [np.iinfo(np.uint32).max] * 7)
    assert int(np.array(empty.offsets.numpy(), copy=False)[-1]) == 0


@pytest.mark.cuda
def test_ragged_embedding_save_load_accepts_different_runtime_partition(tmp_path):
    name = "ragged-embedding-save-load"
    network, *_ = _network(name, thor.DataType.uint64)
    save_dir = tmp_path / "model"
    network.save(str(save_dir), overwrite=False)

    loaded = thor.Network(name)
    loaded.load(str(save_dir))
    placed = loaded.place(3, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)

    packed = np.array([2, 6, 2, 6, 9, np.iinfo(np.uint32).max, np.iinfo(np.uint32).max], dtype=np.uint32)
    offsets = np.array([0, 2, 2, 5], dtype=np.uint64)
    output = placed.infer({"tokens": _physical_ragged(packed, offsets, thor.DataType.uint64)})["output"]
    values = np.array(output.values.numpy(), copy=True)[:5]
    assert np.all(np.isfinite(values))
    np.testing.assert_allclose(values[0], values[2], rtol=0, atol=0)
    np.testing.assert_allclose(values[1], values[3], rtol=0, atol=0)
    np.testing.assert_array_equal(np.array(output.offsets.numpy(), copy=True), offsets)
