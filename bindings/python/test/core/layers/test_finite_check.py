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


def _finite_check_architecture(network: thor.Network):
    layers = [layer for layer in json.loads(network.get_architecture_json())["layers"] if layer["layer_type"] == "finite_check"]
    assert len(layers) == 1
    return layers[0]


def test_finite_check_constructs_and_serializes_policy():
    network = thor.Network("finite_check_constructs")
    source = thor.layers.NetworkInput(network, "input", [4], thor.DataType.fp32)
    check = thor.layers.FiniteCheck(
        network,
        source.get_feature_output(),
        tensor_label="after_encoder",
        check_forward=True,
        check_backward=False,
        fail_on_non_finite=False,
        max_reported_indices=5,
    )

    assert isinstance(check, thor.layers.FiniteCheck)
    assert check.get_feature_output() != source.get_feature_output()
    assert check.get_feature_output().get_dimensions() == [4]
    assert check.get_tensor_label() == "after_encoder"
    assert check.get_check_forward() is True
    assert check.get_check_backward() is False
    assert check.get_fail_on_non_finite() is False
    assert check.get_max_reported_indices() == 5

    architecture = _finite_check_architecture(network)
    assert architecture["tensor_label"] == "after_encoder"
    assert architecture["check_forward"] is True
    assert architecture["check_backward"] is False
    assert architecture["fail_on_non_finite"] is False
    assert architecture["max_reported_indices"] == 5


def test_finite_check_rejects_no_enabled_direction():
    network = thor.Network("finite_check_bad_policy")
    source = thor.layers.NetworkInput(network, "input", [4], thor.DataType.fp32)
    with pytest.raises(ValueError, match="must check forward, backward, or both"):
        thor.layers.FiniteCheck(
            network,
            source.get_feature_output(),
            check_forward=False,
            check_backward=False,
        )


def test_finite_check_rejects_excessive_sample_count():
    network = thor.Network("finite_check_bad_sample_count")
    source = thor.layers.NetworkInput(network, "input", [4], thor.DataType.fp32)
    with pytest.raises(ValueError, match="supported maximum of 32"):
        thor.layers.FiniteCheck(
            network,
            source.get_feature_output(),
            max_reported_indices=33,
        )


@pytest.mark.cuda
def test_finite_check_forward_passes_finite_values_unchanged():
    network = thor.Network("finite_check_finite_forward")
    source = thor.layers.NetworkInput(network, "input", [4], thor.DataType.fp32)
    check = thor.layers.FiniteCheck(network, source.get_feature_output(), tensor_label="finite_activation")
    thor.layers.NetworkOutput(network, "output", check.get_feature_output(), thor.DataType.fp32)

    values = np.array([[1.0, -2.0, 3.5, 4.25], [-5.0, 6.0, -7.25, 8.5]], dtype=np.float32)
    placed = network.place(values.shape[0], inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)
    outputs = placed.infer({"input": _cpu_tensor(values, thor.DataType.fp32)})
    np.testing.assert_array_equal(np.array(outputs["output"].numpy(), copy=True), values)


@pytest.mark.cuda
def test_finite_check_forward_reports_dtype_counts_and_indices():
    network = thor.Network("finite_check_non_finite_forward")
    source = thor.layers.NetworkInput(network, "input", [4], thor.DataType.fp32)
    check = thor.layers.FiniteCheck(
        network,
        source.get_feature_output(),
        tensor_label="after_projection",
        max_reported_indices=4,
    )
    thor.layers.NetworkOutput(network, "output", check.get_feature_output(), thor.DataType.fp32)

    values = np.array([[1.0, np.nan, np.inf, -np.inf], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
    placed = network.place(values.shape[0], inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)

    with pytest.raises(RuntimeError) as error:
        placed.infer({"input": _cpu_tensor(values, thor.DataType.fp32)})

    message = str(error.value)
    assert "FiniteCheck detected non-finite values" in message
    assert 'label="after_projection"' in message
    assert "direction=forward" in message
    assert "tensor_role=activation" in message
    assert "dtype=fp32" in message
    assert "shape=[2, 4]" in message
    assert "non_finite=3" in message
    assert "nan=1" in message
    assert "positive_infinity=1" in message
    assert "negative_infinity=1" in message
    assert "flat_index=" in message
    assert "index=[" in message
