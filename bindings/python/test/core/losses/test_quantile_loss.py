import numpy as np
import pytest
import thor


def _net():
    return thor.Network("test_net_quantile_loss")


def _tensor_1d(size: int, dtype=thor.DataType.fp32):
    return thor.Tensor([size], dtype)


def _cpu_tensor(values: np.ndarray, dtype: thor.DataType) -> thor.physical.PhysicalTensor:
    values = np.asarray(values, dtype=thor.physical.numpy_dtypes.from_thor(dtype), order="C")
    placement = thor.physical.Placement(thor.physical.DeviceType.cpu, 0)
    descriptor = thor.physical.PhysicalTensor.Descriptor(dtype, list(values.shape))
    tensor = thor.physical.PhysicalTensor(placement, descriptor)
    tensor.numpy()[...] = values
    return tensor


def _quantile_reference(predictions: np.ndarray, labels: np.ndarray, quantile: float) -> np.ndarray:
    error = labels.astype(np.float32) - predictions.astype(np.float32)
    return np.where(error > 0.0, quantile * error, (quantile - 1.0) * error).astype(np.float32)


def _reduce_loss(raw: np.ndarray, reported_loss_shape: thor.losses.LossShape) -> np.ndarray:
    if reported_loss_shape == thor.losses.LossShape.raw:
        return raw
    if reported_loss_shape == thor.losses.LossShape.elementwise:
        return np.sum(raw, axis=1, keepdims=True)

    # LossShaper averages over the batch dimension whenever that dimension is
    # reduced. It sums over the feature/class dimension.
    batch_size = raw.shape[0]
    if reported_loss_shape == thor.losses.LossShape.classwise:
        return (np.sum(raw, axis=0, keepdims=True) / batch_size).astype(np.float32)
    if reported_loss_shape == thor.losses.LossShape.batch:
        return np.array([[np.sum(raw) / batch_size]], dtype=np.float32)
    raise AssertionError(f"Unhandled loss shape: {reported_loss_shape}")


def _run_quantile_loss_network(
    predictions: np.ndarray,
    labels: np.ndarray,
    quantile: float,
    reported_loss_shape: thor.losses.LossShape,
    example_weights: np.ndarray | None = None,
) -> np.ndarray:
    shape_name = str(reported_loss_shape).split(".")[-1]
    weight_suffix = "_weighted" if example_weights is not None else ""
    n = thor.Network(f"test_net_quantile_loss_numerical_{shape_name}{weight_suffix}")
    dtype = thor.DataType.fp32
    feature_dims = list(predictions.shape[1:])
    predictions_input = thor.layers.NetworkInput(n, "predictions", feature_dims, dtype)
    labels_input = thor.layers.NetworkInput(n, "labels", feature_dims, dtype)
    example_weights_tensor = None
    if example_weights is not None:
        example_weights_dims = list(example_weights.shape[1:])
        example_weights_input = thor.layers.NetworkInput(n, "example_weights", example_weights_dims, dtype)
        example_weights_tensor = example_weights_input.get_feature_output()
    loss = thor.losses.QuantileLoss(
        n,
        predictions_input.get_feature_output(),
        labels_input.get_feature_output(),
        quantile,
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
    assert set(outputs.keys()) == {"loss"}
    return np.array(outputs["loss"].numpy(), copy=True)


def test_quantile_loss_constructs_defaults():
    n = _net()
    preds = _tensor_1d(1)
    labels = _tensor_1d(1)

    loss = thor.losses.QuantileLoss(n, preds, labels)
    assert isinstance(loss, thor.losses.QuantileLoss)
    assert loss.quantile == pytest.approx(0.5)


def test_quantile_loss_constructs_vector_width_100_for_forecast_horizon():
    n = _net()
    preds = _tensor_1d(100)
    labels = _tensor_1d(100)

    loss = thor.losses.QuantileLoss(
        n,
        preds,
        labels,
        0.9,
        thor.DataType.fp32,
        thor.losses.LossShape.raw,
        loss_weight=2.6667,
    )
    assert isinstance(loss, thor.losses.QuantileLoss)
    assert loss.quantile == pytest.approx(0.9)
    assert loss.loss_weight == pytest.approx(2.6667)


def test_pinball_loss_alias_is_available():
    assert thor.losses.PinballLoss is thor.losses.QuantileLoss


@pytest.mark.parametrize("shape", ["batch", "classwise", "elementwise", "raw"])
def test_quantile_loss_reported_loss_shape_variants_construct(shape):
    n = _net()
    preds = _tensor_1d(3)
    labels = _tensor_1d(3)

    loss = thor.losses.QuantileLoss(n, preds, labels, 0.8, None, getattr(thor.losses.LossShape, shape))
    assert isinstance(loss, thor.losses.QuantileLoss)


@pytest.mark.parametrize("quantile", [0.0, -0.1, 1.0, 1.1])
def test_quantile_loss_rejects_invalid_quantile(quantile):
    n = _net()
    preds = _tensor_1d(1)
    labels = _tensor_1d(1)

    with pytest.raises(ValueError, match=r"quantile must be greater than zero and less than one"):
        thor.losses.QuantileLoss(n, preds, labels, quantile)


def test_quantile_loss_rejects_mismatched_labels():
    n = _net()
    preds = _tensor_1d(2)
    labels = _tensor_1d(3)

    with pytest.raises(ValueError, match=r"labels dimensions [\s\S]* must match predictions dimensions"):
        thor.losses.QuantileLoss(n, preds, labels)


def test_quantile_loss_rejects_predictions_not_1d():
    n = _net()
    preds = thor.Tensor([1, 1], thor.DataType.fp32)
    labels = thor.Tensor([1, 1], thor.DataType.fp32)

    with pytest.raises(ValueError, match=r"predictions must be a 1 dimensional tensor"):
        thor.losses.QuantileLoss(n, preds, labels)


def test_quantile_loss_rejects_invalid_loss_data_type():
    n = _net()
    preds = _tensor_1d(1)
    labels = _tensor_1d(1)

    with pytest.raises(ValueError, match=r"loss_data_type must be fp16 or fp32"):
        thor.losses.QuantileLoss(n, preds, labels, 0.5, thor.DataType.int32)


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
def test_quantile_loss_numerical_forward_matches_reference(reported_loss_shape):
    quantile = 0.9
    predictions = np.array(
        [
            [0.0, 0.25, 1.5, -2.0, 3.0],
            [-1.0, 0.75, 2.25, -0.5, 0.125],
        ],
        dtype=np.float32,
    )
    labels = np.array(
        [
            [0.0, -0.25, 0.0, -0.5, 0.5],
            [0.5, 0.25, 1.0, -1.5, -0.125],
        ],
        dtype=np.float32,
    )

    raw_expected = _quantile_reference(predictions, labels, quantile)
    expected = _reduce_loss(raw_expected, reported_loss_shape)
    actual = _run_quantile_loss_network(predictions, labels, quantile, reported_loss_shape)

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_quantile_loss_constructs_with_example_weights():
    n = _net()
    preds = _tensor_1d(5)
    labels = _tensor_1d(5)
    scalar_weights = _tensor_1d(1)
    elementwise_weights = _tensor_1d(5, thor.DataType.fp16)

    loss = thor.losses.QuantileLoss(n, preds, labels, 0.9, example_weights=scalar_weights)
    assert isinstance(loss, thor.losses.QuantileLoss)
    assert loss.example_weights == scalar_weights
    assert loss.get_example_weights() == scalar_weights

    loss = thor.losses.QuantileLoss(n, preds, labels, 0.9, example_weights=elementwise_weights)
    assert isinstance(loss, thor.losses.QuantileLoss)
    assert loss.example_weights == elementwise_weights


def test_quantile_loss_rejects_bad_example_weights():
    n = _net()
    preds = _tensor_1d(5)
    labels = _tensor_1d(5)

    with pytest.raises(ValueError, match=r"example_weights must be distinct"):
        thor.losses.QuantileLoss(n, preds, labels, example_weights=labels)

    with pytest.raises(ValueError, match=r"example_weights must be fp16 or fp32"):
        thor.losses.QuantileLoss(n, preds, labels, example_weights=_tensor_1d(1, thor.DataType.uint32))

    with pytest.raises(ValueError, match=r"example_weights dimensions must be \[1\]"):
        thor.losses.QuantileLoss(n, preds, labels, example_weights=_tensor_1d(3, thor.DataType.fp32))


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
def test_quantile_loss_with_example_weights_numerical_forward_matches_reference(reported_loss_shape):
    quantile = 0.9
    predictions = np.array(
        [
            [0.0, 0.25, 1.5, -2.0, 3.0],
            [-1.0, 0.75, 2.25, -0.5, 0.125],
        ],
        dtype=np.float32,
    )
    labels = np.array(
        [
            [0.0, -0.25, 0.0, -0.5, 0.5],
            [0.5, 0.25, 1.0, -1.5, -0.125],
        ],
        dtype=np.float32,
    )
    example_weights = np.array([[0.25], [1.5]], dtype=np.float32)

    raw_expected = _quantile_reference(predictions, labels, quantile) * example_weights
    expected = _reduce_loss(raw_expected, reported_loss_shape)
    actual = _run_quantile_loss_network(predictions, labels, quantile, reported_loss_shape, example_weights)

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
