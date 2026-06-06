import numpy as np
import pytest
import thor


_LOSS_CLASSES = [
    ("iou", thor.losses.IoULoss),
    ("giou", thor.losses.GIoULoss),
    ("diou", thor.losses.DIoULoss),
    ("ciou", thor.losses.CIoULoss),
]


def _net(name: str = "test_net_box_iou_loss"):
    return thor.Network(name)


def _tensor(dims, dtype=thor.DataType.fp32):
    return thor.Tensor(list(dims), dtype)


def _cpu_tensor(values: np.ndarray, dtype: thor.DataType) -> thor.physical.PhysicalTensor:
    values = np.asarray(values, dtype=thor.physical.numpy_dtypes.from_thor(dtype), order="C")
    placement = thor.physical.Placement(thor.physical.DeviceType.cpu, 0)
    descriptor = thor.physical.PhysicalTensor.Descriptor(dtype, list(values.shape))
    tensor = thor.physical.PhysicalTensor(placement, descriptor)
    tensor.numpy()[...] = values
    return tensor


def _positive_length(hi: np.ndarray, lo: np.ndarray) -> np.ndarray:
    return np.maximum(hi - lo, np.float32(0.0))


def _box_iou_reference(predictions: np.ndarray, labels: np.ndarray, kind: str, eps: float) -> np.ndarray:
    p = predictions.astype(np.float32)
    t = labels.astype(np.float32)
    eps32 = np.float32(eps)

    p_x1, p_y1, p_x2, p_y2 = [p[..., i] for i in range(4)]
    t_x1, t_y1, t_x2, t_y2 = [t[..., i] for i in range(4)]

    p_w = _positive_length(p_x2, p_x1)
    p_h = _positive_length(p_y2, p_y1)
    t_w = _positive_length(t_x2, t_x1)
    t_h = _positive_length(t_y2, t_y1)
    p_area = p_w * p_h
    t_area = t_w * t_h

    inter_w = _positive_length(np.minimum(p_x2, t_x2), np.maximum(p_x1, t_x1))
    inter_h = _positive_length(np.minimum(p_y2, t_y2), np.maximum(p_y1, t_y1))
    inter_area = inter_w * inter_h
    union_area = np.maximum(p_area + t_area - inter_area, eps32)
    iou = inter_area / (union_area + eps32)
    loss = np.float32(1.0) - iou

    enclosing_w = _positive_length(np.maximum(p_x2, t_x2), np.minimum(p_x1, t_x1))
    enclosing_h = _positive_length(np.maximum(p_y2, t_y2), np.minimum(p_y1, t_y1))
    enclosing_area = np.maximum(enclosing_w * enclosing_h, eps32)
    enclosing_diag_sq = np.maximum(enclosing_w * enclosing_w + enclosing_h * enclosing_h + eps32, eps32)

    if kind == "giou":
        loss = loss + (enclosing_area - union_area) / (enclosing_area + eps32)
    elif kind in {"diou", "ciou"}:
        p_cx2 = p_x1 + p_x2
        p_cy2 = p_y1 + p_y2
        t_cx2 = t_x1 + t_x2
        t_cy2 = t_y1 + t_y2
        center_distance_sq = ((p_cx2 - t_cx2) ** 2 + (p_cy2 - t_cy2) ** 2) * np.float32(0.25)
        loss = loss + center_distance_sq / enclosing_diag_sq
        if kind == "ciou":
            v = ((np.arctan(t_w / (t_h + eps32)) - np.arctan(p_w / (p_h + eps32))) ** 2) * np.float32(
                4.0 / (np.pi * np.pi)
            )
            alpha = v / ((np.float32(1.0) - iou) + v + eps32)
            loss = loss + alpha * v

    return loss.astype(np.float32)


def _reduce_loss(raw: np.ndarray, reported_loss_shape: thor.losses.LossShape) -> np.ndarray:
    if raw.ndim == 1:
        raw = raw.reshape(raw.shape[0], 1)
    if reported_loss_shape == thor.losses.LossShape.raw:
        return raw.astype(np.float32)
    if reported_loss_shape == thor.losses.LossShape.elementwise:
        return np.sum(raw, axis=1, keepdims=True).astype(np.float32)

    batch_size = raw.shape[0]
    if reported_loss_shape == thor.losses.LossShape.classwise:
        return (np.sum(raw, axis=0, keepdims=True) / batch_size).astype(np.float32)
    if reported_loss_shape == thor.losses.LossShape.batch:
        return np.array([[np.sum(raw) / batch_size]], dtype=np.float32)
    raise AssertionError(f"Unhandled loss shape: {reported_loss_shape}")


def _run_box_iou_loss_network(loss_cls, predictions: np.ndarray, labels: np.ndarray, eps: float, reported_loss_shape):
    shape_name = str(reported_loss_shape).split(".")[-1]
    n = thor.Network(f"test_net_{loss_cls.__name__.lower()}_numerical_{shape_name}")
    dtype = thor.DataType.fp32
    feature_dims = list(predictions.shape[1:])
    predictions_input = thor.layers.NetworkInput(n, "predictions", feature_dims, dtype)
    labels_input = thor.layers.NetworkInput(n, "labels", feature_dims, dtype)
    loss = loss_cls(
        n,
        predictions_input.get_feature_output(),
        labels_input.get_feature_output(),
        "xyxy",
        eps,
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
    outputs = placed.infer(
        {
            "predictions": _cpu_tensor(predictions, dtype),
            "labels": _cpu_tensor(labels, dtype),
        }
    )
    assert set(outputs.keys()) == {"loss"}
    return np.array(outputs["loss"].numpy(), copy=True)


@pytest.mark.parametrize("_, loss_cls", _LOSS_CLASSES)
def test_box_iou_loss_constructs_defaults(_, loss_cls):
    n = _net()
    predictions = _tensor([4])
    labels = _tensor([4])

    loss = loss_cls(n, predictions, labels)
    assert isinstance(loss, loss_cls)
    assert loss.box_format == "xyxy"
    assert loss.eps == pytest.approx(1.0e-7)


@pytest.mark.parametrize("_, loss_cls", _LOSS_CLASSES)
def test_box_iou_loss_constructs_with_box_format_eps_loss_dtype_and_shape(_, loss_cls):
    n = _net()
    predictions = _tensor([3, 4], thor.DataType.fp16)
    labels = _tensor([3, 4], thor.DataType.fp16)

    loss = loss_cls(
        n,
        predictions,
        labels,
        "xyxy",
        1.0e-5,
        thor.DataType.fp32,
        thor.losses.LossShape.elementwise,
    )
    assert isinstance(loss, loss_cls)
    assert loss.box_format == "xyxy"
    assert loss.eps == pytest.approx(1.0e-5)


@pytest.mark.parametrize("shape", ["batch", "classwise", "elementwise", "raw"])
@pytest.mark.parametrize("_, loss_cls", _LOSS_CLASSES)
def test_box_iou_loss_reported_loss_shape_variants_construct(shape, _, loss_cls):
    n = _net()
    predictions = _tensor([2, 4])
    labels = _tensor([2, 4])

    loss = loss_cls(n, predictions, labels, "xyxy", 1.0e-7, None, getattr(thor.losses.LossShape, shape))
    assert isinstance(loss, loss_cls)


@pytest.mark.parametrize("_, loss_cls", _LOSS_CLASSES)
def test_box_iou_loss_rejects_invalid_shapes_dtype_format_eps_and_duplicate_tensors(_, loss_cls):
    n = _net()
    predictions = _tensor([2, 4])
    labels = _tensor([2, 4])

    with pytest.raises(ValueError, match=r"labels dimensions [\s\S]* must match predictions dimensions"):
        loss_cls(n, predictions, _tensor([3, 4]))
    with pytest.raises(ValueError, match=r"predictions must have dimensions \[4\] or \[boxes, 4\]"):
        loss_cls(n, _tensor([2, 2]), labels)
    with pytest.raises(ValueError, match=r"box_format must be 'xyxy'"):
        loss_cls(n, predictions, labels, "cxcywh")
    with pytest.raises(ValueError, match=r"eps must be greater than zero"):
        loss_cls(n, predictions, labels, "xyxy", 0.0)
    with pytest.raises(ValueError, match=r"loss_data_type must be fp16 or fp32"):
        loss_cls(n, predictions, labels, "xyxy", 1.0e-7, thor.DataType.int32)
    with pytest.raises(ValueError, match=r"predictions must use fp16 or fp32 dtype"):
        loss_cls(n, _tensor([2, 4], thor.DataType.uint8), labels)
    with pytest.raises(ValueError, match=r"predictions and labels must be distinct tensors"):
        loss_cls(n, predictions, predictions)


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
@pytest.mark.parametrize("kind, loss_cls", _LOSS_CLASSES)
def test_box_iou_loss_numerical_forward_matches_reference(kind, loss_cls, reported_loss_shape):
    eps = 1.0e-7
    predictions = np.array(
        [
            [[0.10, 0.20, 1.40, 1.80], [2.00, 0.50, 3.50, 1.90]],
            [[0.30, 2.10, 1.70, 3.40], [1.20, 1.30, 2.80, 2.90]],
        ],
        dtype=np.float32,
    )
    labels = np.array(
        [
            [[0.00, 0.10, 1.20, 1.60], [1.80, 0.30, 3.00, 2.20]],
            [[0.50, 1.90, 1.90, 3.10], [1.00, 1.10, 3.10, 3.20]],
        ],
        dtype=np.float32,
    )

    raw_expected = _box_iou_reference(predictions, labels, kind, eps)
    expected = _reduce_loss(raw_expected, reported_loss_shape)
    actual = _run_box_iou_loss_network(loss_cls, predictions, labels, eps, reported_loss_shape)

    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.cuda
@pytest.mark.parametrize("kind, loss_cls", _LOSS_CLASSES)
def test_single_box_rank_forward_matches_reference(kind, loss_cls):
    eps = 1.0e-7
    predictions = np.array(
        [[0.10, 0.20, 1.40, 1.80], [2.00, 0.50, 3.50, 1.90]],
        dtype=np.float32,
    )
    labels = np.array(
        [[0.00, 0.10, 1.20, 1.60], [1.80, 0.30, 3.00, 2.20]],
        dtype=np.float32,
    )

    raw_expected = _box_iou_reference(predictions, labels, kind, eps).reshape(predictions.shape[0], 1)
    actual = _run_box_iou_loss_network(loss_cls, predictions, labels, eps, thor.losses.LossShape.raw)

    np.testing.assert_allclose(actual, raw_expected, rtol=2e-5, atol=2e-6)
