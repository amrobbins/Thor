import numpy as np
import pytest
import thor


def _net():
    return thor.Network("test_net_binary_focal_loss")


def _tensor_1d(size: int, dtype=thor.DataType.fp32):
    return thor.Tensor([size], dtype)


def _cpu_tensor(values: np.ndarray, dtype: thor.DataType) -> thor.physical.PhysicalTensor:
    values = np.asarray(values, dtype=thor.physical.numpy_dtypes.from_thor(dtype), order="C")
    placement = thor.physical.Placement(thor.physical.DeviceType.cpu, 0)
    descriptor = thor.physical.PhysicalTensor.Descriptor(dtype, list(values.shape))
    tensor = thor.physical.PhysicalTensor(placement, descriptor)
    tensor.numpy()[...] = values
    return tensor


def _binary_focal_reference(logits: np.ndarray, labels: np.ndarray, gamma: float, alpha: float) -> np.ndarray:
    logits = logits.astype(np.float32)
    labels = labels.astype(np.float32)
    bce = np.maximum(logits, 0.0) - logits * labels + np.log1p(np.exp(-np.abs(logits)))
    pt = np.exp(-bce)
    alpha_t = labels * alpha + (1.0 - labels) * (1.0 - alpha)
    return (alpha_t * np.power(1.0 - pt, gamma) * bce).astype(np.float32)


def _reduce_loss(raw: np.ndarray, reported_loss_shape: thor.losses.LossShape) -> np.ndarray:
    if reported_loss_shape == thor.losses.LossShape.raw:
        return raw
    if reported_loss_shape == thor.losses.LossShape.elementwise:
        return np.sum(raw, axis=1, keepdims=True)
    if reported_loss_shape == thor.losses.LossShape.batch:
        return np.array([[np.sum(raw) / raw.shape[0]]], dtype=np.float32)
    raise AssertionError(f"Unhandled loss shape: {reported_loss_shape}")


def _run_binary_focal_loss_network(
    predictions: np.ndarray,
    labels: np.ndarray,
    gamma: float,
    alpha: float,
    reported_loss_shape: thor.losses.LossShape,
) -> np.ndarray:
    shape_name = str(reported_loss_shape).split(".")[-1]
    n = thor.Network(f"test_net_binary_focal_loss_numerical_{shape_name}")
    dtype = thor.DataType.fp32
    predictions_input = thor.layers.NetworkInput(n, "predictions", [1], dtype)
    labels_input = thor.layers.NetworkInput(n, "labels", [1], dtype)
    loss = thor.losses.classification.BinaryFocalLoss(
        n,
        predictions_input.get_feature_output(),
        labels_input.get_feature_output(),
        gamma,
        alpha,
        dtype,
        reported_loss_shape,
    )
    thor.layers.NetworkOutput(n, "loss", loss.get_loss(), dtype)

    placed = n.place(
        predictions.shape[0],
        inference_only=True,
        forced_devices=[0],
        forced_num_stamps_per_gpu=1,
    )
    outputs = placed.infer({"predictions": _cpu_tensor(predictions, dtype), "labels": _cpu_tensor(labels, dtype)})
    assert set(outputs.keys()) == {"loss"}
    return np.array(outputs["loss"].numpy(), copy=True)


def test_binary_focal_loss_constructs_defaults():
    n = _net()
    preds = _tensor_1d(1)
    labels = _tensor_1d(1)

    loss = thor.losses.classification.BinaryFocalLoss(n, preds, labels)
    assert isinstance(loss, thor.losses.classification.BinaryFocalLoss)
    assert loss.gamma == pytest.approx(2.0)
    assert loss.alpha == pytest.approx(0.25)


def test_binary_focal_loss_constructs_custom_values():
    n = _net()
    preds = _tensor_1d(1, thor.DataType.fp16)
    labels = _tensor_1d(1, thor.DataType.uint8)

    loss = thor.losses.classification.BinaryFocalLoss(
        n,
        preds,
        labels,
        1.5,
        0.75,
        thor.DataType.fp32,
        thor.losses.LossShape.elementwise,
    )
    assert isinstance(loss, thor.losses.classification.BinaryFocalLoss)
    assert loss.gamma == pytest.approx(1.5)
    assert loss.alpha == pytest.approx(0.75)


@pytest.mark.parametrize("shape", ["batch", "elementwise", "raw"])
def test_binary_focal_loss_reported_loss_shape_variants_construct(shape):
    n = _net()
    preds = _tensor_1d(1)
    labels = _tensor_1d(1)

    loss = thor.losses.classification.BinaryFocalLoss(n, preds, labels, 2.0, 0.25, None, getattr(thor.losses.LossShape, shape))
    assert isinstance(loss, thor.losses.classification.BinaryFocalLoss)


def test_binary_focal_loss_rejects_classwise_shape():
    n = _net()
    preds = _tensor_1d(1)
    labels = _tensor_1d(1)

    with pytest.raises(ValueError, match=r"Invalid value .* reported_loss_shape"):
        thor.losses.classification.BinaryFocalLoss(n, preds, labels, 2.0, 0.25, None, thor.losses.LossShape.classwise)


def test_binary_focal_loss_rejects_invalid_gamma_alpha_and_dtype():
    n = _net()
    preds = _tensor_1d(1)
    labels = _tensor_1d(1)

    with pytest.raises(ValueError, match=r"gamma must be non-negative"):
        thor.losses.classification.BinaryFocalLoss(n, preds, labels, -1.0)
    with pytest.raises(ValueError, match=r"alpha must be in the range"):
        thor.losses.classification.BinaryFocalLoss(n, preds, labels, 2.0, -0.1)
    with pytest.raises(ValueError, match=r"alpha must be in the range"):
        thor.losses.classification.BinaryFocalLoss(n, preds, labels, 2.0, 1.1)
    with pytest.raises(ValueError, match=r"loss_data_type must be fp16 or fp32"):
        thor.losses.classification.BinaryFocalLoss(n, preds, labels, 2.0, 0.25, thor.DataType.int32)


def test_binary_focal_loss_rejects_bad_shapes_and_dtypes():
    n = _net()
    labels = _tensor_1d(1)

    with pytest.raises(ValueError, match=r"predictions must be a 1 dimensional logits tensor of size one"):
        thor.losses.classification.BinaryFocalLoss(n, _tensor_1d(2), labels)
    with pytest.raises(ValueError, match=r"labels must be a 1 dimensional tensor of size one"):
        thor.losses.classification.BinaryFocalLoss(n, _tensor_1d(1), _tensor_1d(2))
    with pytest.raises(ValueError, match=r"predictions must use fp16 or fp32 dtype"):
        thor.losses.classification.BinaryFocalLoss(n, _tensor_1d(1, thor.DataType.uint8), labels)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "reported_loss_shape",
    [
        thor.losses.LossShape.raw,
        thor.losses.LossShape.elementwise,
        thor.losses.LossShape.batch,
    ],
)
def test_binary_focal_loss_numerical_forward_matches_reference(reported_loss_shape):
    gamma = 2.0
    alpha = 0.25
    predictions = np.array([[-2.0], [-0.25], [0.0], [1.5]], dtype=np.float32)
    labels = np.array([[0.0], [1.0], [0.0], [1.0]], dtype=np.float32)

    raw_expected = _binary_focal_reference(predictions, labels, gamma, alpha)
    expected = _reduce_loss(raw_expected, reported_loss_shape)
    actual = _run_binary_focal_loss_network(predictions, labels, gamma, alpha, reported_loss_shape)

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
