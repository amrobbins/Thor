import numpy as np
import pytest
import thor


def _net():
    return thor.Network("test_net_dice_loss")


def _tensor(dims, dtype=thor.DataType.fp32):
    return thor.Tensor(list(dims), dtype)


def _tensor_1d(size: int, dtype=thor.DataType.fp32):
    return _tensor([size], dtype)


def _cpu_tensor(values: np.ndarray, dtype: thor.DataType) -> thor.physical.PhysicalTensor:
    values = np.asarray(values, dtype=thor.physical.numpy_dtypes.from_thor(dtype), order="C")
    placement = thor.physical.Placement(thor.physical.DeviceType.cpu, 0)
    descriptor = thor.physical.PhysicalTensor.Descriptor(dtype, list(values.shape))
    tensor = thor.physical.PhysicalTensor(placement, descriptor)
    tensor.numpy()[...] = values
    return tensor


def _segmentation_sum(values: np.ndarray) -> np.ndarray:
    if values.ndim == 2:
        return np.sum(values, axis=1, keepdims=True)
    return np.sum(values, axis=tuple(range(2, values.ndim)))


def _dice_reference(predictions: np.ndarray, labels: np.ndarray, smooth: float) -> np.ndarray:
    predictions = predictions.astype(np.float32)
    labels = labels.astype(np.float32)
    intersection = _segmentation_sum(predictions * labels)
    denominator = np.maximum(_segmentation_sum(predictions) + _segmentation_sum(labels) + smooth, 1.0e-7)
    return (1.0 - ((2.0 * intersection + smooth) / denominator)).astype(np.float32)


def _reduce_loss(raw: np.ndarray, reported_loss_shape: thor.losses.LossShape) -> np.ndarray:
    if reported_loss_shape == thor.losses.LossShape.raw:
        return raw
    if reported_loss_shape == thor.losses.LossShape.elementwise:
        return np.sum(raw, axis=1, keepdims=True).astype(np.float32)
    if reported_loss_shape == thor.losses.LossShape.classwise:
        return (np.sum(raw, axis=0, keepdims=True) / raw.shape[0]).astype(np.float32)
    if reported_loss_shape == thor.losses.LossShape.batch:
        return np.array([[np.sum(raw) / raw.shape[0]]], dtype=np.float32)
    raise AssertionError(f"Unhandled loss shape: {reported_loss_shape}")


def _run_dice_loss_network(
    predictions: np.ndarray,
    labels: np.ndarray,
    smooth: float,
    reported_loss_shape: thor.losses.LossShape,
) -> np.ndarray:
    shape_name = str(reported_loss_shape).split(".")[-1]
    n = thor.Network(f"test_net_dice_loss_numerical_{shape_name}")
    dtype = thor.DataType.fp32
    feature_dims = list(predictions.shape[1:])
    predictions_input = thor.layers.NetworkInput(n, "predictions", feature_dims, dtype)
    labels_input = thor.layers.NetworkInput(n, "labels", feature_dims, dtype)
    loss = thor.losses.segmentation.DiceLoss(
        n,
        predictions_input.get_feature_output(),
        labels_input.get_feature_output(),
        smooth,
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


def test_dice_loss_constructs_defaults():
    n = _net()
    preds = _tensor_1d(4)
    labels = _tensor_1d(4)

    loss = thor.losses.segmentation.DiceLoss(n, preds, labels)
    assert isinstance(loss, thor.losses.segmentation.DiceLoss)
    assert loss.smooth == pytest.approx(1.0)


def test_dice_loss_constructs_multidimensional_probabilities():
    n = _net()
    preds = _tensor([2, 3, 4], thor.DataType.fp16)
    labels = _tensor([2, 3, 4], thor.DataType.uint8)

    loss = thor.losses.segmentation.DiceLoss(n, preds, labels, 0.25, thor.DataType.fp32, thor.losses.LossShape.raw)
    assert isinstance(loss, thor.losses.segmentation.DiceLoss)
    assert loss.smooth == pytest.approx(0.25)


@pytest.mark.parametrize("shape", ["batch", "classwise", "elementwise", "raw"])
def test_dice_loss_reported_loss_shape_variants_construct(shape):
    n = _net()
    preds = _tensor([2, 3, 4])
    labels = _tensor([2, 3, 4])

    loss = thor.losses.segmentation.DiceLoss(n, preds, labels, 1.0, None, getattr(thor.losses.LossShape, shape))
    assert isinstance(loss, thor.losses.segmentation.DiceLoss)


def test_dice_loss_rejects_mismatched_labels():
    n = _net()
    preds = _tensor_1d(2)
    labels = _tensor_1d(3)

    with pytest.raises(ValueError, match=r"labels dimensions [\s\S]* must match predictions dimensions"):
        thor.losses.segmentation.DiceLoss(n, preds, labels)


def test_dice_loss_rejects_bad_params():
    n = _net()

    with pytest.raises(ValueError, match=r"smooth must be non-negative"):
        thor.losses.segmentation.DiceLoss(n, _tensor_1d(3), _tensor_1d(3), -0.1)
    with pytest.raises(ValueError, match=r"loss_data_type must be fp16 or fp32"):
        thor.losses.segmentation.DiceLoss(n, _tensor_1d(3), _tensor_1d(3), 1.0, thor.DataType.int32)


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
def test_dice_loss_numerical_forward_matches_reference(reported_loss_shape):
    smooth = 0.5
    predictions = np.array(
        [
            [
                [[0.9, 0.1, 0.8], [0.2, 0.6, 0.4]],
                [[0.3, 0.7, 0.2], [0.8, 0.1, 0.5]],
            ],
            [
                [[0.2, 0.7, 0.3], [0.9, 0.4, 0.1]],
                [[0.6, 0.2, 0.9], [0.1, 0.5, 0.7]],
            ],
        ],
        dtype=np.float32,
    )
    labels = np.array(
        [
            [
                [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
                [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]],
            ],
            [
                [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
                [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
            ],
        ],
        dtype=np.float32,
    )

    raw_expected = _dice_reference(predictions, labels, smooth)
    expected = _reduce_loss(raw_expected, reported_loss_shape)
    actual = _run_dice_loss_network(predictions, labels, smooth, reported_loss_shape)

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
