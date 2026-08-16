import numpy as np
import pytest

import thor


BATCH_SIZE = 3
CAPACITY = 8
FEATURES = 4
OFFSETS = np.asarray([0, 3, 3, 5], dtype=np.uint32)


def _cpu_tensor(values: np.ndarray, dtype: thor.DataType) -> thor.physical.PhysicalTensor:
    values = np.asarray(values, dtype=thor.physical.numpy_dtypes.from_thor(dtype), order="C")
    placement = thor.physical.Placement(thor.physical.DeviceType.cpu, 0)
    descriptor = thor.physical.PhysicalTensor.Descriptor(dtype, list(values.shape))
    tensor = thor.physical.PhysicalTensor(placement, descriptor)
    tensor.numpy()[...] = values
    return tensor


def _physical_ragged(values: np.ndarray) -> thor.physical.PhysicalRaggedTensor:
    return thor.physical.PhysicalRaggedTensor(
        _cpu_tensor(values, thor.DataType.fp32),
        _cpu_tensor(OFFSETS, thor.DataType.uint32),
    )


def _build_network(name: str) -> thor.Network:
    network = thor.Network(name)
    history = thor.layers.RaggedNetworkInput(
        network,
        "history",
        thor.DataType.fp32,
        [FEATURES],
        max_total_values=CAPACITY,
        batch_size=BATCH_SIZE,
    )
    relu = thor.activations.Relu().add_to_network(network, history)
    swish = thor.activations.Swish().add_to_network(network, history)
    summed = thor.layers.Add(network, relu, swish).get_feature_output()
    # Exercise the generic ragged CustomLayer path used by the product forecaster's
    # robust historical GLM statistics. Keeping log1p/negate/expm1/negate in one
    # Expression lets EquationCompiler fuse the complete pointwise transform.
    def build_activity(context: thor.layers.CustomLayerBuildContext):
        x = context.input("feature_input")
        logged = thor.physical.Expression.log1p(x)
        return {"feature_output": -thor.physical.Expression.expm1(-logged)}

    activity = thor.layers.CustomLayer(network=network, inputs=relu, build=build_activity)["feature_output"]
    lengths = thor.layers.RaggedRowLengths(network, history).get_feature_output()
    thor.layers.RaggedNetworkOutput(network, "summed", summed)
    thor.layers.RaggedNetworkOutput(network, "activity", activity)
    thor.layers.NetworkOutput(network, "lengths", lengths, thor.DataType.int32)
    return network


def test_ragged_add_rejects_distinct_row_partition_objects():
    network = thor.Network("ragged_add_partition_guard")
    left = thor.layers.RaggedNetworkInput(
        network, "left", thor.DataType.fp32, [FEATURES], CAPACITY, BATCH_SIZE)
    right = thor.layers.RaggedNetworkInput(
        network, "right", thor.DataType.fp32, [FEATURES], CAPACITY, BATCH_SIZE)
    with pytest.raises(Exception):
        thor.layers.Add(network, left, right)


@pytest.mark.cuda
def test_ragged_add_and_row_lengths_execute_over_authoritative_partition_and_round_trip(tmp_path):
    network = _build_network("ragged_add_row_lengths")
    placed = network.place(BATCH_SIZE, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)

    active = np.asarray(
        [[-1.0, 0.5, 2.0, -0.25],
         [0.25, -0.5, 1.5, 3.0],
         [-2.0, -1.0, 0.25, 0.75],
         [1.0, 2.0, -3.0, -4.0],
         [0.1, 0.2, 0.3, 0.4]],
        dtype=np.float32,
    )
    values = np.full((CAPACITY, FEATURES), 12345.0, dtype=np.float32)
    values[: active.shape[0]] = active

    outputs = placed.infer({"history": _physical_ragged(values)})
    np.testing.assert_array_equal(outputs["lengths"].numpy(), np.asarray([[3], [0], [2]], dtype=np.int32))
    summed = outputs["summed"]
    assert isinstance(summed, thor.physical.PhysicalRaggedTensor)
    np.testing.assert_array_equal(summed.offsets.numpy(), OFFSETS)

    expected = np.maximum(active, 0.0) + active / (1.0 + np.exp(-active))
    np.testing.assert_allclose(summed.values.numpy()[: active.shape[0]], expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(summed.values.numpy()[active.shape[0]:], 0.0, rtol=0.0, atol=0.0)
    activity = outputs["activity"]
    expected_activity = 1.0 - np.exp(-np.log1p(np.maximum(active, 0.0)))
    np.testing.assert_allclose(
        activity.values.numpy()[: active.shape[0]], expected_activity, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(activity.values.numpy()[active.shape[0]:], 0.0, rtol=0.0, atol=0.0)

    save_dir = tmp_path / "model"
    placed.save(str(save_dir), overwrite=False, save_optimizer_state=False)
    loaded = thor.Network("ragged_add_row_lengths")
    loaded.load(str(save_dir))
    loaded_placed = loaded.place(BATCH_SIZE, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)
    loaded_outputs = loaded_placed.infer({"history": _physical_ragged(values)})
    np.testing.assert_array_equal(loaded_outputs["lengths"].numpy(), outputs["lengths"].numpy())
    np.testing.assert_array_equal(loaded_outputs["summed"].offsets.numpy(), OFFSETS)
    np.testing.assert_allclose(loaded_outputs["summed"].values.numpy(), summed.values.numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(loaded_outputs["activity"].offsets.numpy(), OFFSETS)
    np.testing.assert_allclose(
        loaded_outputs["activity"].values.numpy(), activity.values.numpy(), rtol=1e-5, atol=1e-5)
