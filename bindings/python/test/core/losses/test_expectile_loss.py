import numpy as np
import pytest
import thor


def _net():
    return thor.Network("test_net_expectile_loss")


def _tensor_1d(size: int, dtype=thor.DataType.fp32):
    return thor.Tensor([size], dtype)


def _cpu_tensor(values: np.ndarray, dtype: thor.DataType) -> thor.physical.PhysicalTensor:
    values = np.asarray(values, dtype=thor.physical.numpy_dtypes.from_thor(dtype), order="C")
    placement = thor.physical.Placement(thor.physical.DeviceType.cpu, 0)
    descriptor = thor.physical.PhysicalTensor.Descriptor(dtype, list(values.shape))
    tensor = thor.physical.PhysicalTensor(placement, descriptor)
    tensor.numpy()[...] = values
    return tensor


def _expectile_reference(predictions: np.ndarray, labels: np.ndarray, expectile: float) -> np.ndarray:
    error = labels.astype(np.float32) - predictions.astype(np.float32)
    weight = np.where(error > 0.0, 2.0 * expectile, 2.0 * (1.0 - expectile))
    return (weight * error * error).astype(np.float32)


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


def _run_expectile_loss_network(
    predictions: np.ndarray,
    labels: np.ndarray,
    expectile: float,
    reported_loss_shape: thor.losses.LossShape,
    example_weights: np.ndarray | None = None,
) -> np.ndarray:
    n = thor.Network("test_net_expectile_loss_numerical")
    dtype = thor.DataType.fp32
    feature_dims = list(predictions.shape[1:])
    predictions_input = thor.layers.NetworkInput(n, "predictions", feature_dims, dtype)
    labels_input = thor.layers.NetworkInput(n, "labels", feature_dims, dtype)
    example_weights_tensor = None
    if example_weights is not None:
        example_weights_input = thor.layers.NetworkInput(n, "example_weights", list(example_weights.shape[1:]), dtype)
        example_weights_tensor = example_weights_input.get_feature_output()

    loss = thor.losses.ExpectileLoss(
        n,
        predictions_input.get_feature_output(),
        labels_input.get_feature_output(),
        expectile,
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


def test_expectile_loss_constructs_defaults():
    n = _net()
    preds = _tensor_1d(1)
    labels = _tensor_1d(1)

    loss = thor.losses.ExpectileLoss(n, preds, labels)
    assert isinstance(loss, thor.losses.ExpectileLoss)
    assert loss.expectile == pytest.approx(0.5)


def test_expectile_loss_constructs_forecast_horizon_and_example_weights():
    n = _net()
    preds = _tensor_1d(100)
    labels = _tensor_1d(100)
    weights = _tensor_1d(1)

    loss = thor.losses.ExpectileLoss(
        n,
        preds,
        labels,
        0.9,
        thor.DataType.fp32,
        thor.losses.LossShape.raw,
        loss_weight=2.6667,
        example_weights=weights,
    )
    assert loss.expectile == pytest.approx(0.9)
    assert loss.loss_weight == pytest.approx(2.6667)
    assert loss.example_weights == weights


@pytest.mark.parametrize("expectile", [0.0, -0.1, 1.0, 1.1])
def test_expectile_loss_rejects_invalid_expectile(expectile):
    n = _net()
    with pytest.raises(ValueError, match=r"expectile must be greater than zero and less than one"):
        thor.losses.ExpectileLoss(n, _tensor_1d(1), _tensor_1d(1), expectile)


def test_expectile_loss_rejects_invalid_inputs():
    n = _net()
    with pytest.raises(ValueError, match=r"labels dimensions [\s\S]* must match predictions dimensions"):
        thor.losses.ExpectileLoss(n, _tensor_1d(2), _tensor_1d(3))
    with pytest.raises(ValueError, match=r"predictions must be a 1 dimensional tensor"):
        thor.losses.ExpectileLoss(
            n,
            thor.Tensor([1, 1], thor.DataType.fp32),
            thor.Tensor([1, 1], thor.DataType.fp32),
        )
    with pytest.raises(ValueError, match=r"loss_data_type must be fp16 or fp32"):
        thor.losses.ExpectileLoss(n, _tensor_1d(1), _tensor_1d(1), 0.5, thor.DataType.int32)


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
def test_expectile_loss_numerical_forward_matches_reference(reported_loss_shape):
    predictions = np.array([[0.0, 0.25, 1.5, -2.0], [-1.0, 0.75, 2.25, -0.5]], dtype=np.float32)
    labels = np.array([[0.0, -0.25, 0.0, -0.5], [0.5, 0.25, 1.0, -1.5]], dtype=np.float32)
    expectile = 0.9

    expected = _reduce_loss(_expectile_reference(predictions, labels, expectile), reported_loss_shape)
    actual = _run_expectile_loss_network(predictions, labels, expectile, reported_loss_shape)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.cuda
def test_center_expectile_exactly_matches_mse_and_example_weights_apply():
    predictions = np.array([[0.0, 0.25, 1.5], [-1.0, 0.75, 2.25]], dtype=np.float32)
    labels = np.array([[0.5, -0.25, 0.0], [0.5, 0.25, 1.0]], dtype=np.float32)
    example_weights = np.array([[0.25], [1.5]], dtype=np.float32)

    raw_expected = np.square(labels - predictions).astype(np.float32) * example_weights
    actual = _run_expectile_loss_network(
        predictions,
        labels,
        0.5,
        thor.losses.LossShape.raw,
        example_weights,
    )
    np.testing.assert_allclose(actual, raw_expected, rtol=1e-5, atol=1e-6)
