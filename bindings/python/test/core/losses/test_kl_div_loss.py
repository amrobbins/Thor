import numpy as np
import pytest
import thor


def _net():
    return thor.Network("test_net_kl_div_loss")


def _tensor_1d(size: int, dtype=thor.DataType.fp32):
    return thor.Tensor([size], dtype)


def _cpu_tensor(values: np.ndarray, dtype: thor.DataType) -> thor.physical.PhysicalTensor:
    values = np.asarray(values, dtype=thor.physical.numpy_dtypes.from_thor(dtype), order="C")
    placement = thor.physical.Placement(thor.physical.DeviceType.cpu, 0)
    descriptor = thor.physical.PhysicalTensor.Descriptor(dtype, list(values.shape))
    tensor = thor.physical.PhysicalTensor(placement, descriptor)
    tensor.numpy()[...] = values
    return tensor


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float32)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))


def _kl_div_reference(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    labels = labels.astype(np.float32)
    log_probs = _log_softmax(logits)
    result = np.zeros_like(labels, dtype=np.float32)
    positive = labels > 0.0
    result[positive] = labels[positive] * (np.log(labels[positive]) - log_probs[positive])
    return result


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


def _run_kl_div_loss_network(
    predictions: np.ndarray,
    labels: np.ndarray,
    reported_loss_shape: thor.losses.LossShape,
) -> np.ndarray:
    shape_name = str(reported_loss_shape).split(".")[-1]
    n = thor.Network(f"test_net_kl_div_loss_numerical_{shape_name}")
    dtype = thor.DataType.fp32
    feature_dims = list(predictions.shape[1:])
    predictions_input = thor.layers.NetworkInput(n, "predictions", feature_dims, dtype)
    labels_input = thor.layers.NetworkInput(n, "labels", feature_dims, dtype)
    loss = thor.losses.KLDivLoss(
        n,
        predictions_input.get_feature_output(),
        labels_input.get_feature_output(),
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


def test_kl_div_loss_constructs_defaults():
    n = _net()
    preds = _tensor_1d(5)
    labels = _tensor_1d(5)

    loss = thor.losses.KLDivLoss(n, preds, labels)
    assert isinstance(loss, thor.losses.KLDivLoss)


def test_kl_div_loss_constructs_with_loss_dtype_and_shape():
    n = _net()
    preds = _tensor_1d(4, thor.DataType.fp16)
    labels = _tensor_1d(4, thor.DataType.fp16)

    loss = thor.losses.KLDivLoss(
        n,
        preds,
        labels,
        thor.DataType.fp32,
        thor.losses.LossShape.elementwise,
    )
    assert isinstance(loss, thor.losses.KLDivLoss)


@pytest.mark.parametrize("shape", ["batch", "classwise", "elementwise", "raw"])
def test_kl_div_loss_reported_loss_shape_variants_construct(shape):
    n = _net()
    preds = _tensor_1d(3)
    labels = _tensor_1d(3)

    loss = thor.losses.KLDivLoss(n, preds, labels, None, getattr(thor.losses.LossShape, shape))
    assert isinstance(loss, thor.losses.KLDivLoss)


def test_kl_div_loss_rejects_mismatched_labels():
    n = _net()
    preds = _tensor_1d(2)
    labels = _tensor_1d(3)

    with pytest.raises(ValueError, match=r"labels dimensions [\s\S]* must match predictions dimensions"):
        thor.losses.KLDivLoss(n, preds, labels)


def test_kl_div_loss_rejects_predictions_not_1d():
    n = _net()
    preds = thor.Tensor([1, 1], thor.DataType.fp32)
    labels = thor.Tensor([1, 1], thor.DataType.fp32)

    with pytest.raises(ValueError, match=r"predictions must be a 1 dimensional logits tensor"):
        thor.losses.KLDivLoss(n, preds, labels)


def test_kl_div_loss_rejects_integer_labels():
    n = _net()
    preds = _tensor_1d(3)
    labels = _tensor_1d(3, thor.DataType.uint16)

    with pytest.raises(ValueError, match=r"labels must use fp16 or fp32 dtype"):
        thor.losses.KLDivLoss(n, preds, labels)


def test_kl_div_loss_rejects_invalid_loss_data_type():
    n = _net()
    preds = _tensor_1d(1)
    labels = _tensor_1d(1)

    with pytest.raises(ValueError, match=r"loss_data_type must be fp16 or fp32"):
        thor.losses.KLDivLoss(n, preds, labels, thor.DataType.int32)


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
def test_kl_div_loss_numerical_forward_matches_reference(reported_loss_shape):
    predictions = np.array(
        [
            [1.0, -0.25, 0.5, 2.0],
            [-1.5, 0.75, 0.25, -0.5],
        ],
        dtype=np.float32,
    )
    labels = np.array(
        [
            [0.70, 0.10, 0.15, 0.05],
            [0.00, 0.60, 0.25, 0.15],
        ],
        dtype=np.float32,
    )

    raw_expected = _kl_div_reference(predictions, labels)
    expected = _reduce_loss(raw_expected, reported_loss_shape)
    actual = _run_kl_div_loss_network(predictions, labels, reported_loss_shape)

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
