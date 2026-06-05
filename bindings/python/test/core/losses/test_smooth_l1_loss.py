import numpy as np
import pytest
import thor


def _net():
    return thor.Network("test_net_smooth_l1_loss")


def _tensor_1d(size: int, dtype=thor.DataType.fp32):
    return thor.Tensor([size], dtype)


def _cpu_tensor(values: np.ndarray, dtype: thor.DataType) -> thor.physical.PhysicalTensor:
    values = np.asarray(values, dtype=thor.physical.numpy_dtypes.from_thor(dtype), order="C")
    placement = thor.physical.Placement(thor.physical.DeviceType.cpu, 0)
    descriptor = thor.physical.PhysicalTensor.Descriptor(dtype, list(values.shape))
    tensor = thor.physical.PhysicalTensor(placement, descriptor)
    tensor.numpy()[...] = values
    return tensor


def _smooth_l1_reference(predictions: np.ndarray, labels: np.ndarray, beta: float) -> np.ndarray:
    diff = predictions.astype(np.float32) - labels.astype(np.float32)
    abs_diff = np.abs(diff)
    return np.where(abs_diff < beta, 0.5 * diff * diff / beta, abs_diff - 0.5 * beta).astype(np.float32)


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


def _run_smooth_l1_loss_network(
    predictions: np.ndarray,
    labels: np.ndarray,
    beta: float,
    reported_loss_shape: thor.losses.LossShape,
) -> np.ndarray:
    shape_name = str(reported_loss_shape).split(".")[-1]
    n = thor.Network(f"test_net_smooth_l1_loss_numerical_{shape_name}")
    dtype = thor.DataType.fp32
    feature_dims = list(predictions.shape[1:])
    predictions_input = thor.layers.NetworkInput(n, "predictions", feature_dims, dtype)
    labels_input = thor.layers.NetworkInput(n, "labels", feature_dims, dtype)
    loss = thor.losses.SmoothL1Loss(
        n,
        predictions_input.get_feature_output(),
        labels_input.get_feature_output(),
        beta,
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


def test_smooth_l1_loss_constructs_defaults():
    n = _net()
    preds = _tensor_1d(1)
    labels = _tensor_1d(1)

    loss = thor.losses.SmoothL1Loss(n, preds, labels)
    assert isinstance(loss, thor.losses.SmoothL1Loss)
    assert loss.beta == pytest.approx(1.0)


def test_smooth_l1_loss_constructs_with_beta_loss_dtype_and_shape():
    n = _net()
    preds = _tensor_1d(4, thor.DataType.fp16)
    labels = _tensor_1d(4, thor.DataType.fp16)

    loss = thor.losses.SmoothL1Loss(
        n,
        preds,
        labels,
        0.5,
        thor.DataType.fp32,
        thor.losses.LossShape.elementwise,
    )
    assert isinstance(loss, thor.losses.SmoothL1Loss)
    assert loss.beta == pytest.approx(0.5)


@pytest.mark.parametrize("shape", ["batch", "classwise", "elementwise", "raw"])
def test_smooth_l1_loss_reported_loss_shape_variants_construct(shape):
    n = _net()
    preds = _tensor_1d(3)
    labels = _tensor_1d(3)

    loss = thor.losses.SmoothL1Loss(n, preds, labels, 1.0, None, getattr(thor.losses.LossShape, shape))
    assert isinstance(loss, thor.losses.SmoothL1Loss)


def test_smooth_l1_loss_rejects_non_positive_beta():
    n = _net()
    preds = _tensor_1d(1)
    labels = _tensor_1d(1)

    with pytest.raises(ValueError, match=r"beta must be greater than zero"):
        thor.losses.SmoothL1Loss(n, preds, labels, 0.0)


def test_smooth_l1_loss_rejects_mismatched_labels():
    n = _net()
    preds = _tensor_1d(2)
    labels = _tensor_1d(3)

    with pytest.raises(ValueError, match=r"labels dimensions [\s\S]* must match predictions dimensions"):
        thor.losses.SmoothL1Loss(n, preds, labels)


def test_smooth_l1_loss_rejects_predictions_not_1d():
    n = _net()
    preds = thor.Tensor([1, 1], thor.DataType.fp32)
    labels = thor.Tensor([1, 1], thor.DataType.fp32)

    with pytest.raises(ValueError, match=r"predictions must be a 1 dimensional tensor"):
        thor.losses.SmoothL1Loss(n, preds, labels)


def test_smooth_l1_loss_rejects_invalid_loss_data_type():
    n = _net()
    preds = _tensor_1d(1)
    labels = _tensor_1d(1)

    with pytest.raises(ValueError, match=r"loss_data_type must be fp16 or fp32"):
        thor.losses.SmoothL1Loss(n, preds, labels, 1.0, thor.DataType.int32)


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
def test_smooth_l1_loss_numerical_forward_matches_reference(reported_loss_shape):
    beta = 0.75
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

    raw_expected = _smooth_l1_reference(predictions, labels, beta)
    expected = _reduce_loss(raw_expected, reported_loss_shape)
    actual = _run_smooth_l1_loss_network(predictions, labels, beta, reported_loss_shape)

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
