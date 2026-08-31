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
        [2],
        max_total_values=8,
        batch_size=3,
        offsets_data_type=offsets_dtype,
    )
    rows = thor.layers.NetworkInput(network, "rows", [3], thor.DataType.fp32)
    softmax = thor.layers.SegmentedSoftmax(network, tokens)
    log_softmax = thor.layers.SegmentedLogSoftmax(network, tokens)
    broadcast = thor.layers.SegmentedBroadcast(network, rows.get_feature_output(), tokens)
    thor.layers.RaggedNetworkOutput(network, "softmax", softmax.get_feature_output())
    thor.layers.RaggedNetworkOutput(network, "log_softmax", log_softmax.get_feature_output())
    thor.layers.RaggedNetworkOutput(network, "broadcast", broadcast.get_feature_output())
    return network, tokens, rows, softmax, log_softmax, broadcast


def test_segmented_primitives_preserve_partition_and_reject_fp64():
    network, tokens, _, softmax, log_softmax, broadcast = _network("segmented-primitives-shape")
    assert softmax.get_feature_output().offsets == tokens.offsets
    assert log_softmax.get_feature_output().offsets == tokens.offsets
    assert broadcast.get_feature_output().offsets == tokens.offsets
    assert broadcast.get_feature_output().values.get_dimensions() == [8, 3]

    architecture = json.loads(network.get_architecture_json())
    layer_types = {layer["layer_type"] for layer in architecture["layers"]}
    assert "segmented_softmax" in layer_types
    assert "segmented_log_softmax" in layer_types
    assert "segmented_broadcast" in layer_types

    fp64 = thor.Network("segmented-primitives-fp64")
    fp64_tokens = thor.layers.RaggedNetworkInput(
        fp64,
        "tokens",
        thor.DataType.fp64,
        [2],
        max_total_values=4,
        batch_size=2,
    )
    fp64_rows = thor.layers.NetworkInput(fp64, "rows", [2], thor.DataType.fp64)
    with pytest.raises(Exception, match="FP16, BF16, and FP32"):
        thor.layers.SegmentedSoftmax(fp64, fp64_tokens)
    with pytest.raises(Exception, match="FP16, BF16, and FP32"):
        thor.layers.SegmentedLogSoftmax(fp64, fp64_tokens)
    with pytest.raises(Exception, match="FP16, BF16, and FP32"):
        thor.layers.SegmentedBroadcast(fp64, fp64_rows.get_feature_output(), fp64_tokens)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "offsets_dtype,np_offsets_dtype",
    [
        (thor.DataType.uint32, np.uint32),
        (thor.DataType.uint64, np.uint64),
    ],
)
def test_segmented_primitives_runtime_ignores_poison_and_supports_empty_rows(offsets_dtype, np_offsets_dtype):
    network, *_ = _network("segmented-primitives-runtime", offsets_dtype)
    placed = network.place(3, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)

    values = np.full((8, 2), np.nan, dtype=np.float32)
    values[:5] = np.array([[0, 1], [2, 3], [1, 0], [1, 1], [1, 2]], dtype=np.float32)
    offsets = np.array([0, 2, 2, 5], dtype=np_offsets_dtype)
    rows = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.float32)
    outputs = placed.infer({
        "tokens": _physical_ragged(values, offsets, offsets_dtype),
        "rows": _cpu_tensor(rows, thor.DataType.fp32),
    })

    soft = np.array(outputs["softmax"].values.numpy(), copy=True)[:5]
    log_soft = np.array(outputs["log_softmax"].values.numpy(), copy=True)[:5]
    broad = np.array(outputs["broadcast"].values.numpy(), copy=True)[:5]
    expected_row0 = np.exp(np.array([[0, 1], [2, 3]], dtype=np.float32))
    expected_row0 /= expected_row0.sum(axis=0, keepdims=True)
    expected_row2 = np.exp(np.array([[1, 0], [1, 1], [1, 2]], dtype=np.float32))
    expected_row2 /= expected_row2.sum(axis=0, keepdims=True)
    expected_soft = np.concatenate([expected_row0, expected_row2], axis=0)
    np.testing.assert_allclose(soft, expected_soft, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(np.exp(log_soft), expected_soft, rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(
        broad,
        np.array(
            [[10, 20, 30], [10, 20, 30], [70, 80, 90], [70, 80, 90], [70, 80, 90]],
            dtype=np.float32,
        ),
        rtol=0,
        atol=0,
    )

    # Reuse the placed executable with an all-empty partition; every packed value
    # is NaN poison and remains outside the logical domain.
    empty_values = np.full((8, 2), np.nan, dtype=np.float32)
    empty_offsets = np.zeros((4,), dtype=np_offsets_dtype)
    empty = placed.infer({
        "tokens": _physical_ragged(empty_values, empty_offsets, offsets_dtype),
        "rows": _cpu_tensor(rows, thor.DataType.fp32),
    })
    assert int(np.array(empty["softmax"].offsets.numpy(), copy=False)[-1]) == 0
    assert int(np.array(empty["log_softmax"].offsets.numpy(), copy=False)[-1]) == 0
    assert int(np.array(empty["broadcast"].offsets.numpy(), copy=False)[-1]) == 0


@pytest.mark.cuda
def test_segmented_primitives_save_load_accepts_different_runtime_partition(tmp_path):
    name = "segmented-primitives-save-load"
    network, *_ = _network(name)
    save_dir = tmp_path / "model"
    network.save(str(save_dir), overwrite=False)

    loaded = thor.Network(name)
    loaded.load(str(save_dir))
    placed = loaded.place(3, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)

    values = np.full((8, 2), np.nan, dtype=np.float32)
    values[:6] = np.arange(12, dtype=np.float32).reshape(6, 2)
    offsets = np.array([0, 1, 4, 6], dtype=np.uint32)
    rows = np.array([[2, 3, 4], [5, 7, 9], [11, 13, 15]], dtype=np.float32)
    outputs = placed.infer({
        "tokens": _physical_ragged(values, offsets, thor.DataType.uint32),
        "rows": _cpu_tensor(rows, thor.DataType.fp32),
    })
    broadcast = np.array(outputs["broadcast"].values.numpy(), copy=True)[:6]
    np.testing.assert_array_equal(
        broadcast,
        np.array(
            [[2, 3, 4], [5, 7, 9], [5, 7, 9], [5, 7, 9], [11, 13, 15], [11, 13, 15]],
            dtype=np.float32,
        ),
    )
