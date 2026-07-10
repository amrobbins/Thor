import numpy as np
import pytest
import thor


def _net():
    return thor.Network("test_net_asymmetric_power_loss")


def _tensor_1d(size: int, dtype=thor.DataType.fp32):
    return thor.Tensor([size], dtype)


def _cpu_tensor(values: np.ndarray, dtype: thor.DataType) -> thor.physical.PhysicalTensor:
    values = np.asarray(values, dtype=thor.physical.numpy_dtypes.from_thor(dtype), order="C")
    placement = thor.physical.Placement(thor.physical.DeviceType.cpu, 0)
    descriptor = thor.physical.PhysicalTensor.Descriptor(dtype, list(values.shape))
    tensor = thor.physical.PhysicalTensor(placement, descriptor)
    tensor.numpy()[...] = values
    return tensor


def _reference(predictions: np.ndarray, labels: np.ndarray, level: float, exponent: float) -> np.ndarray:
    error = labels.astype(np.float32) - predictions.astype(np.float32)
    weight = np.where(error > 0.0, 2.0 * level, 2.0 * (1.0 - level))
    return (weight * np.power(np.abs(error), exponent)).astype(np.float32)


def _reduce_loss(raw: np.ndarray, reported_loss_shape: thor.losses.LossShape) -> np.ndarray:
    if reported_loss_shape == thor.losses.LossShape.raw:
        return raw
    if reported_loss_shape == thor.losses.LossShape.elementwise:
        return np.sum(raw, axis=1, keepdims=True)

    batch_size = raw.shape[0]
    if reported_loss_shape == thor.losses.LossShape.classwise:
        return (np.sum(raw, axis=0, keepdims=True) / batch_size).astype(np.float32)
    if reported_loss_shape == thor.losses.LossShape.batch:
        return np.array([[np.sum(raw) / batch_size]], dtype=np.float32)
    raise AssertionError(f"Unhandled loss shape: {reported_loss_shape}")


def _run_network(
    predictions: np.ndarray,
    labels: np.ndarray,
    level: float,
    exponent: float,
    reported_loss_shape: thor.losses.LossShape,
    example_weights: np.ndarray | None = None,
) -> np.ndarray:
    n = thor.Network("test_net_asymmetric_power_loss_numerical")
    dtype = thor.DataType.fp32
    feature_dims = list(predictions.shape[1:])
    predictions_input = thor.layers.NetworkInput(n, "predictions", feature_dims, dtype)
    labels_input = thor.layers.NetworkInput(n, "labels", feature_dims, dtype)
    example_weights_tensor = None
    if example_weights is not None:
        example_weights_input = thor.layers.NetworkInput(n, "example_weights", list(example_weights.shape[1:]), dtype)
        example_weights_tensor = example_weights_input.get_feature_output()

    loss = thor.losses.AsymmetricPowerLoss(
        n,
        predictions_input.get_feature_output(),
        labels_input.get_feature_output(),
        level,
        exponent,
        dtype,
        reported_loss_shape,
        example_weights=example_weights_tensor,
    )
    thor.layers.NetworkOutput(n, "loss", loss.get_loss(), dtype)

    placed = n.place(
        predictions.shape[0],
        inference_only=True,
        forced_devices=[0],
        forced_num_stamps_per_gpu=1,
    )
    inputs = {"predictions": _cpu_tensor(predictions, dtype), "labels": _cpu_tensor(labels, dtype)}
    if example_weights is not None:
        inputs["example_weights"] = _cpu_tensor(example_weights, dtype)
    outputs = placed.infer(inputs)
    return np.array(outputs["loss"].numpy(), copy=True)


def test_asymmetric_power_loss_constructs_defaults():
    n = _net()
    loss = thor.losses.AsymmetricPowerLoss(n, _tensor_1d(1), _tensor_1d(1))

    assert isinstance(loss, thor.losses.AsymmetricPowerLoss)
    assert loss.level == pytest.approx(0.5)
    assert loss.exponent == pytest.approx(1.5)


def test_asymmetric_power_loss_constructs_forecast_horizon_and_example_weights():
    n = _net()
    preds = _tensor_1d(100)
    labels = _tensor_1d(100)
    weights = _tensor_1d(1)

    loss = thor.losses.AsymmetricPowerLoss(
        n,
        preds,
        labels,
        0.9,
        1.5,
        thor.DataType.fp32,
        thor.losses.LossShape.raw,
        loss_weight=2.6667,
        example_weights=weights,
    )
    assert loss.level == pytest.approx(0.9)
    assert loss.exponent == pytest.approx(1.5)
    assert loss.loss_weight == pytest.approx(2.6667)
    assert loss.example_weights == weights


@pytest.mark.parametrize("level", [0.0, -0.1, 1.0, 1.1, float("nan"), float("inf")])
def test_asymmetric_power_loss_rejects_invalid_level(level):
    n = _net()
    with pytest.raises(ValueError, match=r"level must be finite, greater than zero, and less than one"):
        thor.losses.AsymmetricPowerLoss(n, _tensor_1d(1), _tensor_1d(1), level)


@pytest.mark.parametrize("exponent", [0.0, 0.999, -1.0, float("nan"), float("inf")])
def test_asymmetric_power_loss_rejects_invalid_exponent(exponent):
    n = _net()
    with pytest.raises(ValueError, match=r"exponent must be finite and greater than or equal to 1.0"):
        thor.losses.AsymmetricPowerLoss(n, _tensor_1d(1), _tensor_1d(1), 0.5, exponent)


def test_asymmetric_power_loss_rejects_invalid_inputs():
    n = _net()
    with pytest.raises(ValueError, match=r"labels dimensions [\s\S]* must match predictions dimensions"):
        thor.losses.AsymmetricPowerLoss(n, _tensor_1d(2), _tensor_1d(3))
    with pytest.raises(ValueError, match=r"predictions must be a 1 dimensional tensor"):
        thor.losses.AsymmetricPowerLoss(
            n,
            thor.Tensor([1, 1], thor.DataType.fp32),
            thor.Tensor([1, 1], thor.DataType.fp32),
        )
    with pytest.raises(ValueError, match=r"loss_data_type must be fp16 or fp32"):
        thor.losses.AsymmetricPowerLoss(n, _tensor_1d(1), _tensor_1d(1), 0.5, 1.5, thor.DataType.int32)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "reported_loss_shape",
    [
        thor.losses.LossShape.raw,
        thor.losses.LossShape.elementwise,
        thor.losses.LossShape.classwise,
        thor.losses.LossShape.batch,
    ],
)
def test_asymmetric_power_loss_numerical_forward_matches_reference(reported_loss_shape):
    predictions = np.array([[0.0, 0.25, 1.5, -2.0], [-1.0, 0.75, 2.25, -0.5]], dtype=np.float32)
    labels = np.array([[0.0, -0.25, 0.0, -0.5], [0.5, 0.25, 1.0, -1.5]], dtype=np.float32)
    level = 0.9
    exponent = 1.5

    expected = _reduce_loss(_reference(predictions, labels, level, exponent), reported_loss_shape)
    actual = _run_network(predictions, labels, level, exponent, reported_loss_shape)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.cuda
def test_center_matches_mean_power_and_example_weights_apply():
    predictions = np.array([[0.0, 0.25, 1.5], [-1.0, 0.75, 2.25]], dtype=np.float32)
    labels = np.array([[0.5, -0.25, 0.0], [0.5, 0.25, 1.0]], dtype=np.float32)
    example_weights = np.array([[0.25], [1.5]], dtype=np.float32)
    exponent = 1.5

    expected = np.power(np.abs(labels - predictions), exponent).astype(np.float32) * example_weights
    actual = _run_network(
        predictions,
        labels,
        0.5,
        exponent,
        thor.losses.LossShape.raw,
        example_weights,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.cuda
def test_exponent_two_matches_expectile_reference():
    predictions = np.array([[0.0, 0.25, 1.5], [-1.0, 0.75, 2.25]], dtype=np.float32)
    labels = np.array([[0.5, -0.25, 0.0], [0.5, 0.25, 1.0]], dtype=np.float32)
    level = 0.1

    expected = _reference(predictions, labels, level, 2.0)
    actual = _run_network(predictions, labels, level, 2.0, thor.losses.LossShape.raw)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
