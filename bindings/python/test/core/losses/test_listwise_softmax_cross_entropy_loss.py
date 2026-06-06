import numpy as np
import pytest
import thor


def _net():
    return thor.Network("test_net_listwise_softmax_cross_entropy_loss")


def _tensor_1d(size: int, dtype=thor.DataType.fp32):
    return thor.Tensor([size], dtype)


def _cpu_tensor(values: np.ndarray, dtype: thor.DataType) -> thor.physical.PhysicalTensor:
    values = np.asarray(values, dtype=thor.physical.numpy_dtypes.from_thor(dtype), order="C")
    placement = thor.physical.Placement(thor.physical.DeviceType.cpu, 0)
    descriptor = thor.physical.PhysicalTensor.Descriptor(dtype, list(values.shape))
    tensor = thor.physical.PhysicalTensor(placement, descriptor)
    tensor.numpy()[...] = values
    return tensor


def _log_softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    values = values.astype(np.float64)
    shifted = values - np.max(values, axis=axis, keepdims=True)
    log_denominator = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    return (shifted - log_denominator).astype(np.float32)


def _listwise_softmax_cross_entropy_reference(scores: np.ndarray, labels: np.ndarray, temperature: float) -> np.ndarray:
    scaled_scores = scores.astype(np.float32) / np.float32(temperature)
    return -np.sum(labels.astype(np.float32) * _log_softmax(scaled_scores, axis=1), axis=1).astype(np.float32)


def _reduce_loss(raw: np.ndarray, reported_loss_shape: thor.losses.LossShape) -> np.ndarray:
    raw = raw.astype(np.float32)
    if reported_loss_shape == thor.losses.LossShape.raw:
        return raw.reshape(raw.shape[0], 1)
    if reported_loss_shape == thor.losses.LossShape.elementwise:
        return raw.reshape(raw.shape[0], 1)

    batch_size = raw.shape[0]
    if reported_loss_shape == thor.losses.LossShape.classwise:
        return np.array([[np.sum(raw) / batch_size]], dtype=np.float32)
    if reported_loss_shape == thor.losses.LossShape.batch:
        return np.array([[np.sum(raw) / batch_size]], dtype=np.float32)
    raise AssertionError(f"Unhandled loss shape: {reported_loss_shape}")


def _run_listwise_softmax_cross_entropy_loss_network(
    predictions: np.ndarray,
    labels: np.ndarray,
    temperature: float,
    reported_loss_shape: thor.losses.LossShape,
) -> np.ndarray:
    shape_name = str(reported_loss_shape).split(".")[-1]
    n = thor.Network(f"test_net_listwise_softmax_cross_entropy_loss_numerical_{shape_name}")
    dtype = thor.DataType.fp32
    feature_dims = list(predictions.shape[1:])
    predictions_input = thor.layers.NetworkInput(n, "predictions", feature_dims, dtype)
    labels_input = thor.layers.NetworkInput(n, "labels", feature_dims, dtype)
    loss = thor.losses.ListwiseSoftmaxCrossEntropyLoss(
        n,
        predictions_input.get_feature_output(),
        labels_input.get_feature_output(),
        temperature,
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


def test_listwise_softmax_cross_entropy_loss_constructs_defaults():
    n = _net()
    preds = _tensor_1d(4)
    labels = _tensor_1d(4)

    loss = thor.losses.ListwiseSoftmaxCrossEntropyLoss(n, preds, labels)
    assert isinstance(loss, thor.losses.ListwiseSoftmaxCrossEntropyLoss)
    assert loss.temperature == pytest.approx(1.0)


def test_listwise_softmax_cross_entropy_loss_constructs_with_temperature_loss_dtype_and_shape():
    n = _net()
    preds = _tensor_1d(4, thor.DataType.fp16)
    labels = _tensor_1d(4, thor.DataType.fp16)

    loss = thor.losses.ListwiseSoftmaxCrossEntropyLoss(
        n,
        preds,
        labels,
        0.25,
        thor.DataType.fp32,
        thor.losses.LossShape.elementwise,
    )
    assert isinstance(loss, thor.losses.ListwiseSoftmaxCrossEntropyLoss)
    assert loss.temperature == pytest.approx(0.25)


@pytest.mark.parametrize("shape", ["batch", "classwise", "elementwise", "raw"])
def test_listwise_softmax_cross_entropy_loss_reported_loss_shape_variants_construct(shape):
    n = _net()
    preds = _tensor_1d(3)
    labels = _tensor_1d(3)

    loss = thor.losses.ListwiseSoftmaxCrossEntropyLoss(n, preds, labels, 1.0, None, getattr(thor.losses.LossShape, shape))
    assert isinstance(loss, thor.losses.ListwiseSoftmaxCrossEntropyLoss)


def test_listwise_softmax_cross_entropy_loss_rejects_non_positive_temperature():
    n = _net()
    preds = _tensor_1d(4)
    labels = _tensor_1d(4)

    with pytest.raises(ValueError, match=r"temperature must be greater than zero"):
        thor.losses.ListwiseSoftmaxCrossEntropyLoss(n, preds, labels, 0.0)


def test_listwise_softmax_cross_entropy_loss_rejects_mismatched_labels():
    n = _net()
    preds = _tensor_1d(4)
    labels = _tensor_1d(3)

    with pytest.raises(ValueError, match=r"labels dimensions [\s\S]* must match predictions dimensions"):
        thor.losses.ListwiseSoftmaxCrossEntropyLoss(n, preds, labels)


def test_listwise_softmax_cross_entropy_loss_rejects_predictions_not_1d():
    n = _net()
    preds = thor.Tensor([2, 2], thor.DataType.fp32)
    labels = thor.Tensor([2, 2], thor.DataType.fp32)

    with pytest.raises(ValueError, match=r"predictions must be a 1 dimensional fixed-size list score tensor"):
        thor.losses.ListwiseSoftmaxCrossEntropyLoss(n, preds, labels)


def test_listwise_softmax_cross_entropy_loss_rejects_single_document_predictions():
    n = _net()
    preds = _tensor_1d(1)
    labels = _tensor_1d(1)

    with pytest.raises(ValueError, match=r"more than one document"):
        thor.losses.ListwiseSoftmaxCrossEntropyLoss(n, preds, labels)


def test_listwise_softmax_cross_entropy_loss_rejects_invalid_dtypes():
    n = _net()
    preds = _tensor_1d(4)
    labels = _tensor_1d(4)

    with pytest.raises(ValueError, match=r"loss_data_type must be fp16 or fp32"):
        thor.losses.ListwiseSoftmaxCrossEntropyLoss(n, preds, labels, 1.0, thor.DataType.int32)
    with pytest.raises(ValueError, match=r"predictions must use fp16 or fp32 dtype"):
        thor.losses.ListwiseSoftmaxCrossEntropyLoss(n, _tensor_1d(4, thor.DataType.uint8), labels)
    with pytest.raises(ValueError, match=r"labels must use fp16 or fp32 dtype"):
        thor.losses.ListwiseSoftmaxCrossEntropyLoss(n, preds, _tensor_1d(4, thor.DataType.uint8))


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
def test_listwise_softmax_cross_entropy_loss_numerical_forward_matches_reference(reported_loss_shape):
    temperature = 0.7
    predictions = np.array(
        [
            [0.25, 1.5, -0.5, 0.75],
            [1.25, -0.25, 0.5, -1.0],
            [-0.75, 0.125, 1.75, 0.375],
        ],
        dtype=np.float32,
    )
    labels = np.array(
        [
            [0.70, 0.10, 0.15, 0.05],
            [0.05, 0.60, 0.25, 0.10],
            [0.40, 0.00, 0.35, 0.15],
        ],
        dtype=np.float32,
    )

    raw_expected = _listwise_softmax_cross_entropy_reference(predictions, labels, temperature)
    expected = _reduce_loss(raw_expected, reported_loss_shape)
    actual = _run_listwise_softmax_cross_entropy_loss_network(predictions, labels, temperature, reported_loss_shape)

    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
