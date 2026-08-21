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


def _nb_nll_reference(mean, dispersion, labels):
    mean = np.asarray(mean, dtype=np.float64)
    dispersion = np.asarray(dispersion, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    concentration = 1.0 / dispersion
    lgamma = np.vectorize(math.lgamma)
    result = (
        lgamma(concentration)
        + lgamma(labels + 1.0)
        - lgamma(labels + concentration)
        + (concentration + labels) * np.log1p(dispersion * mean)
        - labels * np.log(dispersion)
        - labels * np.log(mean)
    )
    return result.astype(np.float32)


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


def test_negative_binomial_nll_loss_constructs_defaults():
    n = thor.Network("test_negative_binomial_defaults")
    mean = _tensor_1d(4)
    dispersion = _tensor_1d(4)
    labels = _tensor_1d(4, thor.DataType.uint16)

    loss = thor.losses.distribution.NegativeBinomialNLLLoss(n, mean, dispersion, labels)

    assert loss.log_mean is True
    assert loss.log_dispersion is True
    assert loss.eps == pytest.approx(1.0e-8)
    assert loss.mean == mean
    assert loss.dispersion == dispersion


def test_negative_binomial_nll_loss_accepts_direct_parameters_weights_and_raw_reporting():
    n = thor.Network("test_negative_binomial_options")
    mean = _tensor_1d(3)
    dispersion = _tensor_1d(3)
    labels = _tensor_1d(3)
    weights = _tensor_1d(1)

    loss = thor.losses.distribution.NegativeBinomialNLLLoss(
        n,
        mean,
        dispersion,
        labels,
        log_mean=False,
        log_dispersion=False,
        reported_loss_shape=thor.losses.LossShape.raw,
        example_weights=weights,
    )

    assert loss.log_mean is False
    assert loss.log_dispersion is False
    assert loss.get_example_weights() == weights
    assert loss.get_loss().get_dimensions() == [3]


def test_negative_binomial_nll_loss_rejects_shape_and_dtype_mismatches():
    n = thor.Network("test_negative_binomial_validation")
    mean = _tensor_1d(3)
    dispersion = _tensor_1d(4)
    labels = _tensor_1d(3)

    with pytest.raises(ValueError, match=r"must match mean dimensions"):
        thor.losses.distribution.NegativeBinomialNLLLoss(n, mean, dispersion, labels)

    dispersion = _tensor_1d(3)
    bad_labels = _tensor_1d(3, thor.DataType.int32)
    with pytest.raises(ValueError, match=r"labels must use boolean, unsigned integer, fp16, or fp32"):
        thor.losses.distribution.NegativeBinomialNLLLoss(n, mean, dispersion, bad_labels)


@pytest.mark.cuda
@pytest.mark.parametrize("log_parameters", [False, True])
@pytest.mark.parametrize(
    "reported_loss_shape",
    [
        thor.losses.LossShape.raw,
        thor.losses.LossShape.per_example,
        thor.losses.LossShape.per_output,
        thor.losses.LossShape.batch,
    ],
)
def test_negative_binomial_nll_loss_forward_matches_reference(log_parameters, reported_loss_shape):
    mean = np.array([[0.3, 1.2, 5.0, 12.0], [2.5, 0.75, 7.0, 20.0]], dtype=np.float32)
    dispersion = np.array([[0.15, 0.4, 0.8, 0.2], [0.6, 1.1, 0.35, 0.1]], dtype=np.float32)
    labels = np.array([[0.0, 1.0, 7.0, 9.0], [4.0, 0.0, 12.0, 25.0]], dtype=np.float32)

    n = thor.Network(f"test_negative_binomial_forward_{log_parameters}_{reported_loss_shape}")
    dtype = thor.DataType.fp32
    mean_input = thor.layers.NetworkInput(n, "mean", [4], dtype)
    dispersion_input = thor.layers.NetworkInput(n, "dispersion", [4], dtype)
    labels_input = thor.layers.NetworkInput(n, "labels", [4], dtype)
    loss = thor.losses.distribution.NegativeBinomialNLLLoss(
        n,
        mean_input.get_feature_output(),
        dispersion_input.get_feature_output(),
        labels_input.get_feature_output(),
        log_mean=log_parameters,
        log_dispersion=log_parameters,
        loss_data_type=dtype,
        reported_loss_shape=reported_loss_shape,
    )
    thor.layers.NetworkOutput(n, "loss", loss.get_loss(), dtype)

    placed = n.place(2, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)
    outputs = placed.infer(
        {
            "mean": _cpu_tensor(np.log(mean) if log_parameters else mean, dtype),
            "dispersion": _cpu_tensor(np.log(dispersion) if log_parameters else dispersion, dtype),
            "labels": _cpu_tensor(labels, dtype),
        }
    )

    expected = _reduce_loss(_nb_nll_reference(mean, dispersion, labels), reported_loss_shape)
    np.testing.assert_allclose(np.array(outputs["loss"].numpy(), copy=True), expected, rtol=2e-5, atol=2e-5)


@pytest.mark.cuda
def test_negative_binomial_nll_loss_elementwise_weights_scale_raw_loss():
    mean = np.array([[0.5, 2.0, 8.0], [1.5, 4.0, 10.0]], dtype=np.float32)
    dispersion = np.array([[0.2, 0.4, 0.7], [0.5, 0.3, 0.9]], dtype=np.float32)
    labels = np.array([[0.0, 3.0, 12.0], [1.0, 6.0, 4.0]], dtype=np.float32)
    weights = np.array([[1.0, 0.0, 0.5], [0.25, 1.5, 0.0]], dtype=np.float32)

    n = thor.Network("test_negative_binomial_weighted")
    dtype = thor.DataType.fp32
    mean_input = thor.layers.NetworkInput(n, "mean", [3], dtype)
    dispersion_input = thor.layers.NetworkInput(n, "dispersion", [3], dtype)
    labels_input = thor.layers.NetworkInput(n, "labels", [3], dtype)
    weights_input = thor.layers.NetworkInput(n, "weights", [3], dtype)
    loss = thor.losses.distribution.NegativeBinomialNLLLoss(
        n,
        mean_input.get_feature_output(),
        dispersion_input.get_feature_output(),
        labels_input.get_feature_output(),
        log_mean=False,
        log_dispersion=False,
        reported_loss_shape=thor.losses.LossShape.raw,
        example_weights=weights_input.get_feature_output(),
    )
    thor.layers.NetworkOutput(n, "loss", loss.get_loss(), dtype)

    placed = n.place(2, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)
    outputs = placed.infer(
        {
            "mean": _cpu_tensor(mean, dtype),
            "dispersion": _cpu_tensor(dispersion, dtype),
            "labels": _cpu_tensor(labels, dtype),
            "weights": _cpu_tensor(weights, dtype),
        }
    )
    expected = _nb_nll_reference(mean, dispersion, labels) * weights
    np.testing.assert_allclose(np.array(outputs["loss"].numpy(), copy=True), expected, rtol=2e-5, atol=2e-5)
