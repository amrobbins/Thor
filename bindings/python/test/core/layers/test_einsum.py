import json

import numpy as np
import pytest
import thor


def _network_input(network: thor.Network, name: str, shape, dtype=thor.DataType.fp32) -> thor.Tensor:
    return thor.layers.NetworkInput(network, name, list(shape), dtype).get_feature_output()


def _cpu_tensor(values: np.ndarray, dtype: thor.DataType) -> thor.physical.PhysicalTensor:
    values = np.asarray(values, dtype=thor.physical.numpy_dtypes.from_thor(dtype), order="C")
    placement = thor.physical.Placement(thor.physical.DeviceType.cpu, 0)
    descriptor = thor.physical.PhysicalTensor.Descriptor(dtype, list(values.shape))
    tensor = thor.physical.PhysicalTensor(placement, descriptor)
    tensor.numpy()[...] = values
    return tensor


def _einsum_layers(network: thor.Network):
    return [
        layer
        for layer in json.loads(network.get_architecture_json())["layers"]
        if layer["layer_type"] == "einsum"
    ]


def test_einsum_layer_binding_constructs_symbolic_layer_and_infers_shape():
    network = thor.Network("python_einsum_layer_binding")
    lhs = _network_input(network, "lhs", [3, 4], thor.DataType.fp32)
    rhs = _network_input(network, "rhs", [4, 5], thor.DataType.fp32)

    layer = thor.layers.Einsum(network, "ik,kj->ij", [lhs, rhs])

    assert isinstance(layer, thor.layers.Einsum)
    assert isinstance(layer, thor.layers.MultiConnectionLayer)
    assert not isinstance(layer, thor.layers.TrainableLayer)
    assert layer.get_equation() == "ik,kj->ij"
    output = layer.get_feature_output()
    assert output.get_dimensions() == [3, 5]
    assert output.get_data_type() == thor.DataType.fp32


def test_top_level_einsum_infers_unique_network_and_returns_tensor():
    network = thor.Network("python_top_level_einsum")
    lhs = _network_input(network, "lhs", [2, 3])
    rhs = _network_input(network, "rhs", [3, 4])

    output = thor.einsum("ik,kj->ij", lhs, rhs)

    assert isinstance(output, thor.Tensor)
    assert output.get_dimensions() == [2, 4]
    layers = _einsum_layers(network)
    assert len(layers) == 1
    assert layers[0]["equation"] == "ik,kj->ij"


def test_top_level_einsum_supports_ellipsis_and_multi_operand_equations():
    network = thor.Network("python_einsum_ellipsis_multi_operand")
    x = _network_input(network, "x", [2, 3, 4])
    w = _network_input(network, "w", [4, 5])
    y = thor.einsum("i...j,jk->i...k", x, w)
    assert y.get_dimensions() == [2, 3, 5]

    a = _network_input(network, "a", [2, 3])
    b = _network_input(network, "b", [3, 4])
    c = _network_input(network, "c", [4, 6])
    z = thor.einsum("ab,bc,cd->ad", a, b, c)
    assert z.get_dimensions() == [2, 6]


def test_top_level_einsum_supports_duplicate_symbolic_operand_occurrences():
    network = thor.Network("python_einsum_duplicate_operand")
    x = _network_input(network, "x", [2, 3])

    output = thor.einsum("ij,ij->ij", x, x)

    assert output.get_dimensions() == [2, 3]
    layer = _einsum_layers(network)[0]
    assert len(layer["inputs"]) == 2
    assert layer["inputs"][0]["id"] == layer["inputs"][1]["id"]


def test_top_level_einsum_accepts_explicit_network_for_unowned_symbolic_tensors():
    network = thor.Network("python_einsum_explicit_network")
    lhs = thor.Tensor([2, 3], thor.DataType.fp32)
    rhs = thor.Tensor([3, 4], thor.DataType.fp32)

    output = thor.einsum("ik,kj->ij", lhs, rhs, network=network)

    assert output.get_dimensions() == [2, 4]
    assert len(_einsum_layers(network)) == 1


def test_top_level_einsum_requires_explicit_network_when_tensor_ownership_is_ambiguous():
    shared = thor.Tensor([3], thor.DataType.fp32)
    first = thor.Network("python_einsum_ambiguous_first")
    second = thor.Network("python_einsum_ambiguous_second")
    thor.layers.NetworkOutput(first, "first_output", shared, thor.DataType.fp32)
    thor.layers.NetworkOutput(second, "second_output", shared, thor.DataType.fp32)

    with pytest.raises(ValueError, match=r"more than one live Network"):
        thor.einsum("i->i", shared)

    output = thor.einsum("i->i", shared, network=first)
    assert output.get_dimensions() == [3]
    assert len(_einsum_layers(first)) == 1
    assert len(_einsum_layers(second)) == 0


def test_top_level_einsum_rejects_unowned_tensors_without_network():
    lhs = thor.Tensor([2, 3], thor.DataType.fp32)
    rhs = thor.Tensor([3, 4], thor.DataType.fp32)

    with pytest.raises(ValueError, match=r"could not infer a Network containing every operand"):
        thor.einsum("ik,kj->ij", lhs, rhs)


def test_top_level_einsum_argument_validation():
    network = thor.Network("python_einsum_argument_validation")
    x = _network_input(network, "x", [2, 3])

    with pytest.raises(ValueError, match="at least one operand"):
        thor.einsum("i->i")
    with pytest.raises(TypeError, match="equation must be str"):
        thor.einsum(123, x)
    with pytest.raises(TypeError, match=r"operand\[1\] must be thor.Tensor"):
        thor.einsum("ij,ij->ij", x, object())
    with pytest.raises(TypeError, match="network must be thor.Network or None"):
        thor.einsum("ij->ij", x, network="not a network")


def test_einsum_layer_binding_rejects_non_sequence_and_bad_operand_type():
    network = thor.Network("python_einsum_layer_validation")
    x = _network_input(network, "x", [2, 3])

    with pytest.raises(TypeError, match=r"feature_inputs.*sequence of thor.Tensor"):
        thor.layers.Einsum(network, "ij->ij", x)
    with pytest.raises(TypeError, match=r"feature_inputs'\[1\].*thor.Tensor"):
        thor.layers.Einsum(network, "ij,ij->ij", [x, object()])


@pytest.mark.cuda
def test_top_level_einsum_executes_with_implicit_batch_dimension():
    network = thor.Network("python_einsum_batched_execution")
    lhs = _network_input(network, "lhs", [2, 3], thor.DataType.fp32)
    rhs = _network_input(network, "rhs", [3, 4], thor.DataType.fp32)
    output = thor.einsum("ik,kj->ij", lhs, rhs)
    thor.layers.NetworkOutput(network, "output", output, thor.DataType.fp32)

    batch_size = 3
    placed = network.place(
        batch_size,
        inference_only=True,
        forced_devices=[0],
        forced_num_stamps_per_gpu=1,
    )

    lhs_values = np.arange(batch_size * 2 * 3, dtype=np.float32).reshape(batch_size, 2, 3) / 7.0
    rhs_values = (np.arange(batch_size * 3 * 4, dtype=np.float32).reshape(batch_size, 3, 4) - 9.0) / 11.0
    expected = np.einsum("bik,bkj->bij", lhs_values, rhs_values)

    outputs = placed.infer(
        {
            "lhs": _cpu_tensor(lhs_values, thor.DataType.fp32),
            "rhs": _cpu_tensor(rhs_values, thor.DataType.fp32),
        }
    )

    assert set(outputs) == {"output"}
    assert np.allclose(outputs["output"].numpy(), expected, rtol=2.0e-4, atol=2.0e-4)
