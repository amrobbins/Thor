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
        enabled=False,
        check_forward=True,
        check_backward=False,
        fail_on_non_finite=False,
        max_reported_indices=5,
    )

    assert isinstance(check, thor.layers.FiniteCheck)
    assert check.get_feature_output() != source.get_feature_output()
    assert check.get_feature_output().get_dimensions() == [4]
    assert check.get_tensor_label() == "after_encoder"
    assert check.get_enabled() is False
    assert check.get_check_forward() is True
    assert check.get_check_backward() is False
    assert check.get_fail_on_non_finite() is False
    assert check.get_max_reported_indices() == 5

    architecture = _finite_check_architecture(network)
    assert architecture["tensor_label"] == "after_encoder"
    assert architecture["enabled"] is False
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


@pytest.mark.cuda
def test_disabled_finite_check_passes_non_finite_values_unchanged(capfd):
    network = thor.Network("finite_check_disabled")
    source = thor.layers.NetworkInput(network, "input", [4], thor.DataType.fp32)
    check = thor.layers.FiniteCheck(
        network,
        source.get_feature_output(),
        tensor_label="disabled_check",
        enabled=False,
    )
    thor.layers.NetworkOutput(network, "output", check.get_feature_output(), thor.DataType.fp32)

    values = np.array([[np.nan, np.inf, -np.inf, 4.0]], dtype=np.float32)
    placed = network.place(1, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)
    assert "FiniteCheck layer is enabled" not in capfd.readouterr().err
    outputs = placed.infer({"input": _cpu_tensor(values, thor.DataType.fp32)})
    np.testing.assert_array_equal(np.array(outputs["output"].numpy(), copy=True), values)


@pytest.mark.cuda
def test_enabled_finite_check_warns_when_stamped(capfd):
    network = thor.Network("finite_check_warning")
    source = thor.layers.NetworkInput(network, "input", [4], thor.DataType.fp32)
    check = thor.layers.FiniteCheck(
        network,
        source.get_feature_output(),
        tensor_label="warning_check",
        enabled=True,
    )
    thor.layers.NetworkOutput(network, "output", check.get_feature_output(), thor.DataType.fp32)

    network.place(1, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)
    warning = capfd.readouterr().err
    assert "Thor warning: FiniteCheck layer is enabled" in warning
    assert 'label="warning_check"' in warning
    assert "intended for diagnostic runs" in warning
    assert "will hurt performance" in warning


def _physical_ragged(values: np.ndarray, offsets: np.ndarray, offsets_dtype: thor.DataType):
    values_tensor = _cpu_tensor(values, thor.DataType.fp32)
    offsets_tensor = _cpu_tensor(offsets, offsets_dtype)
    return thor.physical.PhysicalRaggedTensor(values_tensor, offsets_tensor)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "offsets_dtype,np_offsets_dtype",
    [
        (thor.DataType.uint32, np.uint32),
        (thor.DataType.uint64, np.uint64),
    ],
)
def test_ragged_finite_check_ignores_poisoned_inactive_capacity_and_reuses_partition(offsets_dtype, np_offsets_dtype):
    network = thor.Network("ragged_finite_check_active_prefix")
    source = thor.layers.RaggedNetworkInput(
        network,
        "input",
        thor.DataType.fp32,
        [2],
        max_total_values=6,
        batch_size=3,
        offsets_data_type=offsets_dtype,
    )
    check = thor.layers.FiniteCheck(network, source, tensor_label="ragged_history")
    assert check.get_use_ragged()
    output = check.get_feature_output()
    assert isinstance(output, thor.RaggedTensor)
    assert output.get_offsets() == source.get_offsets()
    thor.layers.RaggedNetworkOutput(network, "output", output)

    placed = network.place(3, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)

    def infer(values, offsets):
        return placed.infer(
            {
                "input": _physical_ragged(
                    np.ascontiguousarray(values, dtype=np.float32),
                    np.ascontiguousarray(offsets, dtype=np_offsets_dtype),
                    offsets_dtype,
                )
            }
        )["output"]

    poisoned = np.asarray(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [np.nan, np.inf],
            [-np.inf, np.nan],
            [np.inf, -np.inf],
        ],
        dtype=np.float32,
    )
    short = infer(poisoned, [0, 1, 1, 3])
    assert isinstance(short, thor.physical.PhysicalRaggedTensor)
    np.testing.assert_array_equal(np.array(short.offsets.numpy(), copy=True), np.asarray([0, 1, 1, 3], dtype=np_offsets_dtype))

    # Reuse the same placed executable with a longer active prefix; a non-finite
    # value that moves into the authoritative prefix must now be reported.
    with pytest.raises(RuntimeError, match=r"checked_elements=8"):
        infer(poisoned, [0, 2, 2, 4])

    finite = poisoned.copy()
    finite[3] = [7.0, 8.0]
    longer = infer(finite, [0, 2, 2, 4])
    np.testing.assert_array_equal(np.array(longer.offsets.numpy(), copy=True), np.asarray([0, 2, 2, 4], dtype=np_offsets_dtype))

    empty = infer(poisoned, [0, 0, 0, 0])
    np.testing.assert_array_equal(np.array(empty.offsets.numpy(), copy=True), np.zeros((4,), dtype=np_offsets_dtype))


@pytest.mark.cuda
def test_ragged_finite_check_save_load_accepts_different_runtime_partition(tmp_path):
    name = "ragged_finite_check_save_load"
    network = thor.Network(name)
    source = thor.layers.RaggedNetworkInput(
        network,
        "input",
        thor.DataType.fp32,
        [2],
        max_total_values=6,
        batch_size=3,
        offsets_data_type=thor.DataType.uint64,
    )
    check = thor.layers.FiniteCheck(network, source, tensor_label="saved_ragged")
    thor.layers.RaggedNetworkOutput(network, "output", check.get_feature_output())
    save_dir = tmp_path / "model"
    network.save(str(save_dir), overwrite=False)

    loaded = thor.Network(name)
    loaded.load(str(save_dir))
    placed = loaded.place(3, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)
    values = np.asarray(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [np.nan, np.inf], [np.nan, -np.inf]],
        dtype=np.float32,
    )
    offsets = np.asarray([0, 2, 2, 4], dtype=np.uint64)
    result = placed.infer({"input": _physical_ragged(values, offsets, thor.DataType.uint64)})["output"]
    np.testing.assert_array_equal(np.array(result.offsets.numpy(), copy=True), offsets)
