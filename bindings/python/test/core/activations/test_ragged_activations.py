import math

import numpy as np
import pytest
import thor


def _cpu_tensor(array: np.ndarray, dtype):
    array = np.asarray(array, dtype=thor.physical.numpy_dtypes.from_thor(dtype), order="C")
    placement = thor.physical.Placement(thor.physical.DeviceType.cpu, 0)
    descriptor = thor.physical.PhysicalTensor.Descriptor(dtype, list(array.shape))
    tensor = thor.physical.PhysicalTensor(placement, descriptor)
    tensor.numpy()[...] = array
    return tensor


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("activation_factory", "reference"),
    [
        (thor.activations.Relu, lambda x: np.maximum(x, 0.0)),
        (thor.activations.Swish, lambda x: x / (1.0 + np.exp(-x))),
        (thor.activations.Tanh, np.tanh),
    ],
)
def test_shape_preserving_activation_accepts_ragged_tensor_and_preserves_partition(activation_factory, reference):
    batch_size = 2
    net = thor.Network("pytest_ragged_activation")
    tokens = thor.layers.RaggedNetworkInput(
        net,
        "tokens",
        thor.DataType.fp32,
        [3],
        max_total_values=8,
        batch_size=batch_size,
        offsets_data_type=thor.DataType.uint32,
    )
    output = activation_factory().add_to_network(net, tokens)
    assert isinstance(output, thor.RaggedTensor)
    thor.layers.RaggedNetworkOutput(net, "output", output)

    placed = net.place(batch_size, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)
    values_np = np.array(
        [
            [-2.0, -1.0, 0.0],
            [0.5, 1.0, 2.0],
            [-0.5, 3.0, -4.0],
            [1.5, -1.5, 0.25],
            [99.0, 99.0, 99.0],
            [99.0, 99.0, 99.0],
            [99.0, 99.0, 99.0],
            [99.0, 99.0, 99.0],
        ],
        dtype=np.float32,
    )
    offsets_np = np.array([0, 1, 4], dtype=np.uint32)
    physical = thor.physical.PhysicalRaggedTensor(
        _cpu_tensor(values_np, thor.DataType.fp32),
        _cpu_tensor(offsets_np, thor.DataType.uint32),
    )

    result = placed.infer({"tokens": physical})["output"]
    assert isinstance(result, thor.physical.PhysicalRaggedTensor)
    assert np.array_equal(result.offsets.numpy(), offsets_np)
    expected = reference(values_np[:4])
    assert np.allclose(result.values.numpy()[:4], expected, rtol=2e-5, atol=2e-5)


def _ordinary_softmax_reference(values):
    values = np.asarray(values, dtype=np.float32)
    shifted = values - values.max(axis=-1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=-1, keepdims=True)


def _segmented_softmax_reference(values, offsets):
    values = np.asarray(values, dtype=np.float32)
    offsets = np.asarray(offsets, dtype=np.int64)
    result = np.empty_like(values)
    for row in range(len(offsets) - 1):
        start = int(offsets[row])
        end = int(offsets[row + 1])
        if start == end:
            continue
        row_values = values[start:end]
        shifted = row_values - row_values.max(axis=0, keepdims=True)
        exp_values = np.exp(shifted)
        result[start:end] = exp_values / exp_values.sum(axis=0, keepdims=True)
    return result


@pytest.mark.cuda
@pytest.mark.parametrize(
    "offsets_dtype,np_offsets_dtype",
    [
        (thor.DataType.uint32, np.uint32),
        (thor.DataType.uint64, np.uint64),
    ],
)
def test_ragged_softmax_is_ordinary_final_axis_and_distinct_from_segmented_with_reuse(
    offsets_dtype, np_offsets_dtype
):
    batch_size = 3
    capacity = 8
    net = thor.Network("pytest_ragged_ordinary_vs_segmented_softmax")
    tokens = thor.layers.RaggedNetworkInput(
        net,
        "tokens",
        thor.DataType.fp32,
        [3],
        max_total_values=capacity,
        batch_size=batch_size,
        offsets_data_type=offsets_dtype,
    )
    ordinary = thor.activations.Softmax().add_to_network(net, tokens)
    segmented = thor.layers.SegmentedSoftmax(net, tokens).get_feature_output()
    assert isinstance(ordinary, thor.RaggedTensor)
    assert ordinary.offsets == tokens.offsets
    assert segmented.offsets == tokens.offsets
    thor.layers.RaggedNetworkOutput(net, "ordinary", ordinary)
    thor.layers.RaggedNetworkOutput(net, "segmented", segmented)

    placed = net.place(batch_size, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)

    cases = [
        (
            np.array([0, 2, 2, 4], dtype=np_offsets_dtype),
            np.array([[0.0, 1.0, 2.0], [2.0, 0.0, -1.0], [1.0, 1.0, 1.0], [2.0, 3.0, 4.0]], dtype=np.float32),
        ),
        (
            np.array([0, 3, 5, 7], dtype=np_offsets_dtype),
            (np.arange(21, dtype=np.float32).reshape(7, 3) - 8.0) / 3.0,
        ),
        (np.zeros((batch_size + 1,), dtype=np_offsets_dtype), np.empty((0, 3), dtype=np.float32)),
        (
            np.array([0, 1, 1, 3], dtype=np_offsets_dtype),
            np.array([[3.0, -1.0, 0.5], [-2.0, 1.0, 4.0], [0.25, 0.5, 0.75]], dtype=np.float32),
        ),
    ]

    saw_deliberate_difference = False
    for offsets_np, active_values in cases:
        values_np = np.full((capacity, 3), np.nan, dtype=np.float32)
        values_np[: len(active_values)] = active_values
        physical = thor.physical.PhysicalRaggedTensor(
            _cpu_tensor(values_np, thor.DataType.fp32),
            _cpu_tensor(offsets_np, offsets_dtype),
        )
        outputs = placed.infer({"tokens": physical})
        ordinary_result = outputs["ordinary"]
        segmented_result = outputs["segmented"]
        assert np.array_equal(ordinary_result.offsets.numpy(), offsets_np)
        assert np.array_equal(segmented_result.offsets.numpy(), offsets_np)

        active_count = int(offsets_np[-1])
        if active_count == 0:
            continue
        expected_ordinary = _ordinary_softmax_reference(active_values)
        expected_segmented = _segmented_softmax_reference(active_values, offsets_np)
        actual_ordinary = np.array(ordinary_result.values.numpy(), copy=True)[:active_count]
        actual_segmented = np.array(segmented_result.values.numpy(), copy=True)[:active_count]
        np.testing.assert_allclose(actual_ordinary, expected_ordinary, rtol=3e-5, atol=3e-5)
        np.testing.assert_allclose(actual_segmented, expected_segmented, rtol=3e-5, atol=3e-5)
        saw_deliberate_difference |= not np.allclose(expected_ordinary, expected_segmented, rtol=1e-5, atol=1e-5)

    assert saw_deliberate_difference, "ordinary final-axis Softmax must not alias SegmentedSoftmax"


@pytest.mark.cuda
def test_ragged_softmax_places_training_backward_through_partition_preserving_output():
    batch_size = 3
    net = thor.Network("pytest_ragged_softmax_training_backward")
    tokens = thor.layers.RaggedNetworkInput(
        net,
        "tokens",
        thor.DataType.fp32,
        [3],
        max_total_values=8,
        batch_size=batch_size,
        offsets_data_type=thor.DataType.uint32,
    )
    ordinary = thor.activations.Softmax().add_to_network(net, tokens)
    pooled = thor.layers.SegmentedReduction(
        net, ordinary, thor.layers.SegmentedReduction.Type.mean
    ).get_feature_output()
    labels = thor.layers.NetworkInput(net, "labels", [3], thor.DataType.fp32).get_feature_output()
    loss = thor.losses.MSE(net, pooled, labels, thor.DataType.fp32, thor.losses.LossShape.batch)
    thor.layers.NetworkOutput(net, "loss", loss.get_loss(), thor.DataType.fp32)

    placed = net.place(
        batch_size,
        inference_only=False,
        forced_devices=[0],
        forced_num_stamps_per_gpu=1,
    )
    assert placed is not None


@pytest.mark.cuda
def test_ragged_softmax_save_load_preserves_partition_and_accepts_changed_runtime_partition(tmp_path):
    batch_size = 3
    capacity = 8
    network_name = "pytest_ragged_softmax_save_load"
    net = thor.Network(network_name)
    tokens = thor.layers.RaggedNetworkInput(
        net,
        "tokens",
        thor.DataType.fp32,
        [3],
        max_total_values=capacity,
        batch_size=batch_size,
        offsets_data_type=thor.DataType.uint64,
    )
    ordinary = thor.activations.Softmax().add_to_network(net, tokens)
    thor.layers.RaggedNetworkOutput(net, "output", ordinary)

    save_dir = tmp_path / "ragged_softmax_model"
    net.save(str(save_dir), overwrite=False)

    loaded = thor.Network(network_name)
    loaded.load(str(save_dir))
    placed = loaded.place(batch_size, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)

    offsets_np = np.array([0, 1, 4, 6], dtype=np.uint64)
    active_values = np.array(
        [
            [1.0, 0.0, -1.0],
            [2.0, 4.0, 1.0],
            [-2.0, 0.5, 3.0],
            [0.25, 0.5, 0.75],
            [5.0, 1.0, -3.0],
            [-0.5, -0.25, 0.0],
        ],
        dtype=np.float32,
    )
    values_np = np.full((capacity, 3), np.nan, dtype=np.float32)
    values_np[: len(active_values)] = active_values
    result = placed.infer(
        {
            "tokens": thor.physical.PhysicalRaggedTensor(
                _cpu_tensor(values_np, thor.DataType.fp32),
                _cpu_tensor(offsets_np, thor.DataType.uint64),
            )
        }
    )["output"]
    assert np.array_equal(result.offsets.numpy(), offsets_np)
    np.testing.assert_allclose(
        np.array(result.values.numpy(), copy=True)[: len(active_values)],
        _ordinary_softmax_reference(active_values),
        rtol=3e-5,
        atol=3e-5,
    )


def _gelu_exact(x):
    erf = np.vectorize(math.erf, otypes=[np.float64])
    x64 = np.asarray(x, dtype=np.float64)
    return (0.5 * x64 * (1.0 + erf(x64 / np.sqrt(2.0)))).astype(np.float32)


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("activation_factory", "gate_reference"),
    [
        (thor.activations.Glu, lambda x: 1.0 / (1.0 + np.exp(-x))),
        (thor.activations.Reglu, lambda x: np.maximum(x, 0.0)),
        (thor.activations.Geglu, _gelu_exact),
        (thor.activations.Swiglu, lambda x: x / (1.0 + np.exp(-x))),
        (thor.activations.BilinearGlu, lambda x: x),
    ],
)
def test_gated_activation_accepts_ragged_tensor_halves_width_and_preserves_partition(
    activation_factory, gate_reference
):
    batch_size = 2
    net = thor.Network("pytest_ragged_glu")
    tokens = thor.layers.RaggedNetworkInput(
        net,
        "tokens",
        thor.DataType.fp32,
        [6],
        max_total_values=8,
        batch_size=batch_size,
        offsets_data_type=thor.DataType.uint32,
    )
    output = activation_factory().add_to_network(net, tokens)
    assert isinstance(output, thor.RaggedTensor)
    assert output.values.get_dimensions() == [8, 3]
    thor.layers.RaggedNetworkOutput(net, "output", output)

    placed = net.place(batch_size, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)
    values_np = np.array(
        [
            [1.0, -2.0, 0.5, -1.0, 2.0, 0.25],
            [0.5, 3.0, -4.0, 1.5, -0.5, 2.0],
            [-1.0, 0.25, 2.0, -2.0, 0.75, -1.5],
            [4.0, -0.5, 1.0, 0.0, 1.0, -3.0],
            [99.0, 99.0, 99.0, 99.0, 99.0, 99.0],
            [99.0, 99.0, 99.0, 99.0, 99.0, 99.0],
            [99.0, 99.0, 99.0, 99.0, 99.0, 99.0],
            [99.0, 99.0, 99.0, 99.0, 99.0, 99.0],
        ],
        dtype=np.float32,
    )
    offsets_np = np.array([0, 1, 4], dtype=np.uint32)
    physical = thor.physical.PhysicalRaggedTensor(
        _cpu_tensor(values_np, thor.DataType.fp32),
        _cpu_tensor(offsets_np, thor.DataType.uint32),
    )

    result = placed.infer({"tokens": physical})["output"]
    assert isinstance(result, thor.physical.PhysicalRaggedTensor)
    assert np.array_equal(result.offsets.numpy(), offsets_np)

    value = values_np[:4, :3]
    gate = values_np[:4, 3:]
    expected = value * gate_reference(gate)
    assert np.allclose(result.values.numpy()[:4], expected, rtol=5e-5, atol=5e-5)


@pytest.mark.cuda
def test_ragged_swiglu_save_load_preserves_shape_partition_and_execution(tmp_path):
    batch_size = 2
    network_name = "pytest_ragged_swiglu_save_load"
    net = thor.Network(network_name)
    tokens = thor.layers.RaggedNetworkInput(
        net,
        "tokens",
        thor.DataType.fp32,
        [4],
        max_total_values=6,
        batch_size=batch_size,
        offsets_data_type=thor.DataType.uint32,
    )
    output = thor.activations.Swiglu().add_to_network(net, tokens)
    assert output.values.get_dimensions() == [6, 2]
    thor.layers.RaggedNetworkOutput(net, "output", output)

    save_dir = tmp_path / "ragged_swiglu_model"
    net.save(str(save_dir), overwrite=False)

    loaded = thor.Network(network_name)
    loaded.load(str(save_dir))
    placed = loaded.place(batch_size, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)

    values_np = np.array(
        [
            [2.0, -1.0, 0.5, -2.0],
            [1.0, 3.0, -0.25, 1.5],
            [-4.0, 0.5, 2.0, -1.0],
            [99.0, 99.0, 99.0, 99.0],
            [99.0, 99.0, 99.0, 99.0],
            [99.0, 99.0, 99.0, 99.0],
        ],
        dtype=np.float32,
    )
    offsets_np = np.array([0, 1, 3], dtype=np.uint32)
    physical = thor.physical.PhysicalRaggedTensor(
        _cpu_tensor(values_np, thor.DataType.fp32),
        _cpu_tensor(offsets_np, thor.DataType.uint32),
    )

    result = placed.infer({"tokens": physical})["output"]
    assert np.array_equal(result.offsets.numpy(), offsets_np)
    value = values_np[:3, :2]
    gate = values_np[:3, 2:]
    expected = value * (gate / (1.0 + np.exp(-gate)))
    assert np.allclose(result.values.numpy()[:3], expected, rtol=5e-5, atol=5e-5)


@pytest.mark.cuda
def test_ragged_swish_save_load_preserves_partition_and_execution(tmp_path):
    batch_size = 2
    network_name = "pytest_ragged_swish_save_load"
    net = thor.Network(network_name)
    tokens = thor.layers.RaggedNetworkInput(
        net,
        "tokens",
        thor.DataType.fp32,
        [2],
        max_total_values=6,
        batch_size=batch_size,
        offsets_data_type=thor.DataType.uint32,
    )
    output = thor.activations.Swish().add_to_network(net, tokens)
    thor.layers.RaggedNetworkOutput(net, "output", output)

    save_dir = tmp_path / "ragged_swish_model"
    net.save(str(save_dir), overwrite=False)

    loaded = thor.Network(network_name)
    loaded.load(str(save_dir))
    placed = loaded.place(batch_size, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)

    values_np = np.array(
        [[-1.0, 2.0], [0.5, -0.25], [3.0, -4.0], [99.0, 99.0], [99.0, 99.0], [99.0, 99.0]],
        dtype=np.float32,
    )
    offsets_np = np.array([0, 1, 3], dtype=np.uint32)
    physical = thor.physical.PhysicalRaggedTensor(
        _cpu_tensor(values_np, thor.DataType.fp32),
        _cpu_tensor(offsets_np, thor.DataType.uint32),
    )

    result = placed.infer({"tokens": physical})["output"]
    assert np.array_equal(result.offsets.numpy(), offsets_np)
    expected = values_np[:3] / (1.0 + np.exp(-values_np[:3]))
    assert np.allclose(result.values.numpy()[:3], expected, rtol=2e-5, atol=2e-5)
