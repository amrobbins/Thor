import json
import math

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


def _student_t_nll_reference(location, log_scale, labels, degrees_of_freedom):
    location = np.asarray(location, dtype=np.float64)
    log_scale = np.asarray(log_scale, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    nu = np.asarray(degrees_of_freedom, dtype=np.float64)
    z = (location - labels) * np.exp(-log_scale)
    lgamma_nu_half = np.vectorize(math.lgamma)(0.5 * nu)
    lgamma_nu_plus_one_half = np.vectorize(math.lgamma)(0.5 * (nu + 1.0))
    return (
        log_scale
        + lgamma_nu_half
        - lgamma_nu_plus_one_half
        + 0.5 * np.log(nu * np.pi)
        + 0.5 * (nu + 1.0) * np.log1p((z * z) / nu)
    ).astype(np.float32)


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


def test_student_t_nll_loss_constructs_fixed_df_defaults():
    n = thor.Network("test_student_t_defaults")
    location = _tensor_1d(4)
    log_scale = _tensor_1d(4)
    labels = _tensor_1d(4)

    loss = thor.losses.distribution.StudentTNLLLoss(n, location, log_scale, labels)

    assert loss.location == location
    assert loss.log_scale == log_scale
    assert loss.degrees_of_freedom == pytest.approx(3.0)
    assert loss.learned_log_degrees_of_freedom is None
    assert loss.minimum_degrees_of_freedom == pytest.approx(0.0)


def test_student_t_nll_loss_accepts_learned_log_df_weights_and_raw_reporting():
    n = thor.Network("test_student_t_learned_options")
    location = _tensor_1d(3)
    log_scale = _tensor_1d(3)
    labels = _tensor_1d(3)
    log_df = _tensor_1d(3)
    weights = _tensor_1d(1)

    loss = thor.losses.distribution.StudentTNLLLoss(
        n,
        location,
        log_scale,
        labels,
        minimum_degrees_of_freedom=2.0,
        learned_log_degrees_of_freedom=log_df,
        reported_loss_shape=thor.losses.LossShape.raw,
        example_weights=weights,
    )

    assert loss.degrees_of_freedom is None
    assert loss.learned_log_degrees_of_freedom == log_df
    assert loss.minimum_degrees_of_freedom == pytest.approx(2.0)
    assert loss.get_example_weights() == weights
    assert loss.get_loss().get_dimensions() == [3]


def test_student_t_nll_loss_rejects_invalid_parameterization_shape_and_dtype():
    n = thor.Network("test_student_t_validation")
    location = _tensor_1d(3)
    log_scale = _tensor_1d(4)
    labels = _tensor_1d(3)

    with pytest.raises(ValueError, match=r"log_scale dimensions [\s\S]* must match location dimensions"):
        thor.losses.distribution.StudentTNLLLoss(n, location, log_scale, labels)

    log_scale = _tensor_1d(3)
    bad_log_df = _tensor_1d(4)
    with pytest.raises(ValueError, match=r"learned_log_degrees_of_freedom dimensions [\s\S]* must match location dimensions"):
        thor.losses.distribution.StudentTNLLLoss(
            n, location, log_scale, labels, learned_log_degrees_of_freedom=bad_log_df
        )

    with pytest.raises(ValueError, match=r"either fixed degrees_of_freedom or learned_log_degrees_of_freedom"):
        thor.losses.distribution.StudentTNLLLoss(
            n, location, log_scale, labels, 4.0, learned_log_degrees_of_freedom=_tensor_1d(3)
        )

    with pytest.raises(ValueError, match=r"degrees_of_freedom must be greater than zero"):
        thor.losses.distribution.StudentTNLLLoss(n, location, log_scale, labels, 0.0)

    with pytest.raises(ValueError, match=r"minimum_degrees_of_freedom must be finite and non-negative"):
        thor.losses.distribution.StudentTNLLLoss(
            n, location, log_scale, labels, minimum_degrees_of_freedom=-1.0
        )

    with pytest.raises(ValueError, match=r"fixed degrees_of_freedom must be greater than minimum_degrees_of_freedom"):
        thor.losses.distribution.StudentTNLLLoss(
            n, location, log_scale, labels, 2.0, minimum_degrees_of_freedom=2.0
        )

    bad_labels = _tensor_1d(3, thor.DataType.uint16)
    with pytest.raises(ValueError, match=r"labels must use fp16 or fp32 dtype"):
        thor.losses.distribution.StudentTNLLLoss(n, location, log_scale, bad_labels)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "reported_loss_shape",
    [
        thor.losses.LossShape.raw,
        thor.losses.LossShape.per_example,
        thor.losses.LossShape.per_output,
        thor.losses.LossShape.batch,
    ],
)
def test_student_t_nll_loss_fixed_df_forward_matches_reference(reported_loss_shape):
    location = np.array([[0.0, -1.0, 2.5, 8.0], [1.5, 0.25, -3.0, 4.0]], dtype=np.float32)
    scale = np.array([[0.2, 0.75, 1.5, 3.0], [0.5, 2.0, 0.35, 1.25]], dtype=np.float32)
    log_scale = np.log(scale).astype(np.float32)
    labels = np.array([[0.5, -2.0, 2.0, 12.0], [0.0, 1.25, -1.0, 4.5]], dtype=np.float32)
    nu = 4.5

    n = thor.Network(f"test_student_t_fixed_forward_{reported_loss_shape}")
    dtype = thor.DataType.fp32
    location_input = thor.layers.NetworkInput(n, "location", [4], dtype)
    log_scale_input = thor.layers.NetworkInput(n, "log_scale", [4], dtype)
    labels_input = thor.layers.NetworkInput(n, "labels", [4], dtype)
    loss = thor.losses.distribution.StudentTNLLLoss(
        n,
        location_input.get_feature_output(),
        log_scale_input.get_feature_output(),
        labels_input.get_feature_output(),
        degrees_of_freedom=nu,
        loss_data_type=dtype,
        reported_loss_shape=reported_loss_shape,
    )
    thor.layers.NetworkOutput(n, "loss", loss.get_loss(), dtype)

    placed = n.place(2, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)
    outputs = placed.infer(
        {
            "location": _cpu_tensor(location, dtype),
            "log_scale": _cpu_tensor(log_scale, dtype),
            "labels": _cpu_tensor(labels, dtype),
        }
    )

    expected = _reduce_loss(_student_t_nll_reference(location, log_scale, labels, nu), reported_loss_shape)
    np.testing.assert_allclose(np.array(outputs["loss"].numpy(), copy=True), expected, rtol=3e-5, atol=3e-5)


@pytest.mark.cuda
def test_student_t_nll_loss_learned_df_forward_and_weights_match_reference():
    location = np.array([[0.0, 2.0, -1.0], [1.5, 4.0, 10.0]], dtype=np.float32)
    scale = np.array([[0.5, 1.0, 2.0], [0.75, 0.3, 4.0]], dtype=np.float32)
    log_scale = np.log(scale).astype(np.float32)
    minimum_nu = 2.0
    nu = np.array([[2.5, 3.0, 4.0], [8.0, 12.0, 3.0]], dtype=np.float32)
    log_nu = np.log(nu - minimum_nu).astype(np.float32)
    labels = np.array([[1.0, 3.0, -1.5], [0.0, 6.0, 4.0]], dtype=np.float32)
    weights = np.array([[1.0, 0.0, 0.5], [0.25, 1.5, 0.0]], dtype=np.float32)

    n = thor.Network("test_student_t_learned_weighted")
    dtype = thor.DataType.fp32
    location_input = thor.layers.NetworkInput(n, "location", [3], dtype)
    log_scale_input = thor.layers.NetworkInput(n, "log_scale", [3], dtype)
    labels_input = thor.layers.NetworkInput(n, "labels", [3], dtype)
    log_nu_input = thor.layers.NetworkInput(n, "log_nu", [3], dtype)
    weights_input = thor.layers.NetworkInput(n, "weights", [3], dtype)
    loss = thor.losses.distribution.StudentTNLLLoss(
        n,
        location_input.get_feature_output(),
        log_scale_input.get_feature_output(),
        labels_input.get_feature_output(),
        minimum_degrees_of_freedom=minimum_nu,
        learned_log_degrees_of_freedom=log_nu_input.get_feature_output(),
        reported_loss_shape=thor.losses.LossShape.raw,
        example_weights=weights_input.get_feature_output(),
    )
    thor.layers.NetworkOutput(n, "loss", loss.get_loss(), dtype)

    placed = n.place(2, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)
    outputs = placed.infer(
        {
            "location": _cpu_tensor(location, dtype),
            "log_scale": _cpu_tensor(log_scale, dtype),
            "labels": _cpu_tensor(labels, dtype),
            "log_nu": _cpu_tensor(log_nu, dtype),
            "weights": _cpu_tensor(weights, dtype),
        }
    )
    expected = _student_t_nll_reference(location, log_scale, labels, nu) * weights
    np.testing.assert_allclose(np.array(outputs["loss"].numpy(), copy=True), expected, rtol=4e-5, atol=4e-5)


def test_student_t_nll_loss_save_load_round_trip_serializes_support_layers(tmp_path):
    n = thor.Network("test_student_t_round_trip")
    dtype = thor.DataType.fp32
    location_input = thor.layers.NetworkInput(n, "location", [4], dtype)
    log_scale_input = thor.layers.NetworkInput(n, "log_scale", [4], dtype)
    labels_input = thor.layers.NetworkInput(n, "labels", [4], dtype)
    log_nu_input = thor.layers.NetworkInput(n, "log_nu", [4], dtype)
    weights_input = thor.layers.NetworkInput(n, "weights", [1], dtype)
    loss = thor.losses.distribution.StudentTNLLLoss(
        n,
        location_input.get_feature_output(),
        log_scale_input.get_feature_output(),
        labels_input.get_feature_output(),
        minimum_degrees_of_freedom=2.0,
        learned_log_degrees_of_freedom=log_nu_input.get_feature_output(),
        loss_data_type=dtype,
        reported_loss_shape=thor.losses.LossShape.per_example,
        loss_weight=2.0,
        example_weights=weights_input.get_feature_output(),
    )
    thor.layers.NetworkOutput(n, "loss", loss.get_loss(), dtype)

    arch = json.loads(n.get_architecture_json())
    assert sum(1 for layer in arch["layers"] if layer["layer_type"] == "multi_input_custom_loss") == 1
    assert sum(1 for layer in arch["layers"] if layer["layer_type"] == "loss_shaper") == 1

    save_dir = tmp_path / "student_t_nll_model"
    n.save(str(save_dir), overwrite=False)
    loaded = thor.Network("test_student_t_round_trip")
    loaded.load(str(save_dir))
    loaded_arch = json.loads(loaded.get_architecture_json())
    assert sum(1 for layer in loaded_arch["layers"] if layer["layer_type"] == "multi_input_custom_loss") == 1
    assert sum(1 for layer in loaded_arch["layers"] if layer["layer_type"] == "loss_shaper") == 1
