import json

import numpy as np
import pytest
import thor


def _tensor_1d(size: int, dtype=thor.DataType.fp32):
    return thor.Tensor([size], dtype)


def _cpu_tensor(values: np.ndarray, dtype: thor.DataType) -> thor.physical.PhysicalTensor:
    values = np.asarray(values, dtype=thor.physical.numpy_dtypes.from_thor(dtype), order="C")
    placement = thor.physical.Placement(thor.physical.DeviceType.cpu, 0)
    descriptor = thor.physical.PhysicalTensor.Descriptor(dtype, list(values.shape))
    tensor = thor.physical.PhysicalTensor(placement, descriptor)
    tensor.numpy()[...] = values
    return tensor


def _laplace_nll_reference(location, scale, labels):
    location = np.asarray(location, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    return (np.log(2.0 * scale) + np.abs(labels - location) / scale).astype(np.float32)


def _reduce_loss(raw, shape):
    if shape == thor.losses.LossShape.raw:
        return raw
    if shape == thor.losses.LossShape.per_example:
        return np.sum(raw, axis=1, keepdims=True)
    if shape == thor.losses.LossShape.per_output:
        return np.mean(raw, axis=0, keepdims=True)
    if shape == thor.losses.LossShape.batch:
        return np.array([[np.sum(raw) / raw.shape[0]]], dtype=np.float32)
    raise AssertionError(shape)


def test_laplace_nll_loss_constructs_defaults():
    n = thor.Network("test_laplace_defaults")
    location = _tensor_1d(4)
    scale = _tensor_1d(4)
    labels = _tensor_1d(4)

    loss = thor.losses.distribution.LaplaceNLLLoss(n, location, scale, labels)

    assert loss.log_scale is True
    assert loss.eps == pytest.approx(1.0e-8)
    assert loss.location == location
    assert loss.scale == scale


def test_laplace_nll_loss_accepts_direct_scale_weights_and_raw_reporting():
    n = thor.Network("test_laplace_options")
    location = _tensor_1d(3)
    scale = _tensor_1d(3)
    labels = _tensor_1d(3)
    weights = _tensor_1d(1)

    loss = thor.losses.distribution.LaplaceNLLLoss(
        n,
        location,
        scale,
        labels,
        log_scale=False,
        reported_loss_shape=thor.losses.LossShape.raw,
        example_weights=weights,
    )

    assert loss.log_scale is False
    assert loss.get_example_weights() == weights
    assert loss.get_loss().get_dimensions() == [3]


def test_laplace_nll_loss_rejects_shape_and_dtype_mismatches():
    n = thor.Network("test_laplace_validation")
    location = _tensor_1d(3)
    scale = _tensor_1d(4)
    labels = _tensor_1d(3)

    with pytest.raises(ValueError, match=r"scale dimensions [\s\S]* must match location dimensions"):
        thor.losses.distribution.LaplaceNLLLoss(n, location, scale, labels)

    scale = _tensor_1d(3)
    bad_labels = _tensor_1d(3, thor.DataType.uint16)
    with pytest.raises(ValueError, match=r"labels must use fp16 or fp32 dtype"):
        thor.losses.distribution.LaplaceNLLLoss(n, location, scale, bad_labels)


@pytest.mark.cuda
@pytest.mark.parametrize("log_scale", [False, True])
@pytest.mark.parametrize(
    "reported_loss_shape",
    [
        thor.losses.LossShape.raw,
        thor.losses.LossShape.per_example,
        thor.losses.LossShape.per_output,
        thor.losses.LossShape.batch,
    ],
)
def test_laplace_nll_loss_forward_matches_reference(log_scale, reported_loss_shape):
    location = np.array([[0.0, -1.0, 2.5, 8.0], [1.5, 0.25, -3.0, 4.0]], dtype=np.float32)
    scale = np.array([[0.2, 0.75, 1.5, 3.0], [0.5, 2.0, 0.35, 1.25]], dtype=np.float32)
    labels = np.array([[0.5, -2.0, 2.0, 12.0], [0.0, 1.25, -1.0, 4.5]], dtype=np.float32)

    n = thor.Network(f"test_laplace_forward_{log_scale}_{reported_loss_shape}")
    dtype = thor.DataType.fp32
    location_input = thor.layers.NetworkInput(n, "location", [4], dtype)
    scale_input = thor.layers.NetworkInput(n, "scale", [4], dtype)
    labels_input = thor.layers.NetworkInput(n, "labels", [4], dtype)
    loss = thor.losses.distribution.LaplaceNLLLoss(
        n,
        location_input.get_feature_output(),
        scale_input.get_feature_output(),
        labels_input.get_feature_output(),
        log_scale=log_scale,
        loss_data_type=dtype,
        reported_loss_shape=reported_loss_shape,
    )
    thor.layers.NetworkOutput(n, "loss", loss.get_loss(), dtype)

    placed = n.place(2, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)
    outputs = placed.infer(
        {
            "location": _cpu_tensor(location, dtype),
            "scale": _cpu_tensor(np.log(scale) if log_scale else scale, dtype),
            "labels": _cpu_tensor(labels, dtype),
        }
    )

    expected = _reduce_loss(_laplace_nll_reference(location, scale, labels), reported_loss_shape)
    np.testing.assert_allclose(np.array(outputs["loss"].numpy(), copy=True), expected, rtol=2e-5, atol=2e-5)


@pytest.mark.cuda
def test_laplace_nll_loss_elementwise_weights_scale_raw_loss():
    location = np.array([[0.0, 2.0, -1.0], [1.5, 4.0, 10.0]], dtype=np.float32)
    scale = np.array([[0.5, 1.0, 2.0], [0.75, 0.3, 4.0]], dtype=np.float32)
    labels = np.array([[1.0, 3.0, -1.5], [0.0, 6.0, 4.0]], dtype=np.float32)
    weights = np.array([[1.0, 0.0, 0.5], [0.25, 1.5, 0.0]], dtype=np.float32)

    n = thor.Network("test_laplace_weighted")
    dtype = thor.DataType.fp32
    location_input = thor.layers.NetworkInput(n, "location", [3], dtype)
    scale_input = thor.layers.NetworkInput(n, "scale", [3], dtype)
    labels_input = thor.layers.NetworkInput(n, "labels", [3], dtype)
    weights_input = thor.layers.NetworkInput(n, "weights", [3], dtype)
    loss = thor.losses.distribution.LaplaceNLLLoss(
        n,
        location_input.get_feature_output(),
        scale_input.get_feature_output(),
        labels_input.get_feature_output(),
        log_scale=False,
        reported_loss_shape=thor.losses.LossShape.raw,
        example_weights=weights_input.get_feature_output(),
    )
    thor.layers.NetworkOutput(n, "loss", loss.get_loss(), dtype)

    placed = n.place(2, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)
    outputs = placed.infer(
        {
            "location": _cpu_tensor(location, dtype),
            "scale": _cpu_tensor(scale, dtype),
            "labels": _cpu_tensor(labels, dtype),
            "weights": _cpu_tensor(weights, dtype),
        }
    )
    expected = _laplace_nll_reference(location, scale, labels) * weights
    np.testing.assert_allclose(np.array(outputs["loss"].numpy(), copy=True), expected, rtol=2e-5, atol=2e-5)


def test_laplace_nll_loss_save_load_round_trip_serializes_support_layers(tmp_path):
    n = thor.Network("test_laplace_round_trip")
    dtype = thor.DataType.fp32
    location_input = thor.layers.NetworkInput(n, "location", [4], dtype)
    scale_input = thor.layers.NetworkInput(n, "scale", [4], dtype)
    labels_input = thor.layers.NetworkInput(n, "labels", [4], dtype)
    weights_input = thor.layers.NetworkInput(n, "weights", [1], dtype)
    loss = thor.losses.distribution.LaplaceNLLLoss(
        n,
        location_input.get_feature_output(),
        scale_input.get_feature_output(),
        labels_input.get_feature_output(),
        log_scale=True,
        eps=1.0e-5,
        loss_data_type=dtype,
        reported_loss_shape=thor.losses.LossShape.per_example,
        loss_weight=2.0,
        example_weights=weights_input.get_feature_output(),
    )
    thor.layers.NetworkOutput(n, "loss", loss.get_loss(), dtype)

    arch = json.loads(n.get_architecture_json())
    assert sum(1 for layer in arch["layers"] if layer["layer_type"] == "multi_input_custom_loss") == 1
    assert sum(1 for layer in arch["layers"] if layer["layer_type"] == "loss_shaper") == 1

    save_dir = tmp_path / "laplace_nll_model"
    n.save(str(save_dir), overwrite=False)
    loaded = thor.Network("test_laplace_round_trip")
    loaded.load(str(save_dir))
    loaded_arch = json.loads(loaded.get_architecture_json())
    assert sum(1 for layer in loaded_arch["layers"] if layer["layer_type"] == "multi_input_custom_loss") == 1
    assert sum(1 for layer in loaded_arch["layers"] if layer["layer_type"] == "loss_shaper") == 1


def test_laplace_nll_loss_rejects_non_positive_eps_as_value_error():
    n = thor.Network("laplace_bad_eps")
    location = _tensor_1d(1)
    scale = _tensor_1d(1)
    labels = _tensor_1d(1)

    with pytest.raises(ValueError, match=r"eps must be greater than zero"):
        thor.losses.distribution.LaplaceNLLLoss(n, location, scale, labels, eps=0.0)


def _r10j_laplace_ragged_inputs(network, prefix="r10j_laplace"):
    location = thor.layers.RaggedNetworkInput(
        network,
        f"{prefix}_location",
        thor.DataType.fp32,
        [2],
        batch_size=3,
        max_total_values=8,
        max_values_per_row=4,
    )
    scale = thor.layers.RaggedNetworkInput(
        network,
        f"{prefix}_scale",
        thor.DataType.fp32,
        [2],
        partition=location,
    )
    labels = thor.layers.RaggedNetworkInput(
        network,
        f"{prefix}_labels",
        thor.DataType.fp32,
        [2],
        partition=location,
    )
    return location, scale, labels


def test_laplace_nll_loss_r10j_constructs_ragged_secondary_parameter_and_raw_loss():
    n = thor.Network("r10j_laplace_api")
    location, scale, labels = _r10j_laplace_ragged_inputs(n)
    weights = thor.layers.NetworkInput(n, "weights", [1], thor.DataType.fp16).get_feature_output()
    loss = thor.losses.distribution.LaplaceNLLLoss(
        n,
        location,
        scale,
        labels,
        log_scale=False,
        example_weights=weights,
        reported_loss_shape=thor.losses.LossShape.raw,
    )
    assert loss.is_ragged
    assert loss.location == location
    assert loss.scale == scale
    assert loss.get_labels() == labels
    assert isinstance(loss.get_loss(), thor.RaggedTensor)
    assert loss.get_loss().offsets == location.offsets
    assert loss.example_weights == weights


def test_laplace_nll_loss_r10j_rejects_per_output_and_mismatched_partition():
    n = thor.Network("r10j_laplace_reject")
    location, scale, labels = _r10j_laplace_ragged_inputs(n, "base")
    with pytest.raises(ValueError, match=r"per_output.*undefined"):
        thor.losses.distribution.LaplaceNLLLoss(
            n, location, scale, labels, reported_loss_shape=thor.losses.LossShape.per_output
        )
    different = thor.layers.RaggedNetworkInput(
        n,
        "different_scale",
        thor.DataType.fp32,
        [2],
        batch_size=3,
        max_total_values=8,
        max_values_per_row=4,
    )
    with pytest.raises(ValueError, match=r"exact same row partition"):
        thor.losses.distribution.LaplaceNLLLoss(n, location, different, labels)
