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


def test_ragged_softmax_remains_explicitly_deferred():
    net = thor.Network("pytest_ragged_activation_rejections")
    tokens = thor.layers.RaggedNetworkInput(
        net,
        "tokens",
        thor.DataType.fp32,
        [4],
        max_total_values=8,
        batch_size=2,
        offsets_data_type=thor.DataType.uint32,
    )
    with pytest.raises((RuntimeError, ValueError), match="does not yet support standalone RaggedTensor"):
        thor.activations.Softmax().add_to_network(net, tokens)


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
