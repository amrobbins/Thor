import numpy as np
import pytest
import thor


def _net():
    return thor.Network("test_net_triplet_loss")


def _tensor_1d(size: int, dtype=thor.DataType.fp32):
    return thor.Tensor([size], dtype)


def _cpu_tensor(values: np.ndarray, dtype: thor.DataType) -> thor.physical.PhysicalTensor:
    values = np.asarray(values, dtype=thor.physical.numpy_dtypes.from_thor(dtype), order="C")
    placement = thor.physical.Placement(thor.physical.DeviceType.cpu, 0)
    descriptor = thor.physical.PhysicalTensor.Descriptor(dtype, list(values.shape))
    tensor = thor.physical.PhysicalTensor(placement, descriptor)
    tensor.numpy()[...] = values
    return tensor


def _triplet_reference(anchor: np.ndarray, positive: np.ndarray, negative: np.ndarray, margin: float, eps: float) -> np.ndarray:
    anchor = anchor.astype(np.float32)
    positive = positive.astype(np.float32)
    negative = negative.astype(np.float32)
    d_ap = np.sqrt(np.sum((anchor - positive) ** 2, axis=1, keepdims=True) + np.float32(eps))
    d_an = np.sqrt(np.sum((anchor - negative) ** 2, axis=1, keepdims=True) + np.float32(eps))
    return np.maximum(d_ap - d_an + np.float32(margin), np.float32(0.0)).astype(np.float32)


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


def _run_triplet_loss_network(
    anchor: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    margin: float,
    eps: float,
    reported_loss_shape: thor.losses.LossShape,
) -> np.ndarray:
    shape_name = str(reported_loss_shape).split(".")[-1]
    n = thor.Network(f"test_net_triplet_loss_numerical_{shape_name}")
    dtype = thor.DataType.fp32
    feature_dims = list(anchor.shape[1:])
    anchor_input = thor.layers.NetworkInput(n, "anchor", feature_dims, dtype)
    positive_input = thor.layers.NetworkInput(n, "positive", feature_dims, dtype)
    negative_input = thor.layers.NetworkInput(n, "negative", feature_dims, dtype)
    loss = thor.losses.metric_learning.TripletLoss(
        n,
        anchor_input.get_feature_output(),
        positive_input.get_feature_output(),
        negative_input.get_feature_output(),
        margin,
        eps,
        dtype,
        reported_loss_shape,
    )
    thor.layers.NetworkOutput(n, "loss", loss.get_loss(), dtype)

    placed = n.place(
        anchor.shape[0],
        inference_only=True,
        forced_devices=[0],
        forced_num_stamps_per_gpu=1,
    )
    outputs = placed.infer(
        {
            "anchor": _cpu_tensor(anchor, dtype),
            "positive": _cpu_tensor(positive, dtype),
            "negative": _cpu_tensor(negative, dtype),
        }
    )
    assert set(outputs.keys()) == {"loss"}
    return np.array(outputs["loss"].numpy(), copy=True)


def test_triplet_loss_constructs_defaults():
    n = _net()
    anchor = _tensor_1d(4)
    positive = _tensor_1d(4)
    negative = _tensor_1d(4)

    loss = thor.losses.metric_learning.TripletLoss(n, anchor, positive, negative)
    assert isinstance(loss, thor.losses.metric_learning.TripletLoss)
    assert loss.margin == pytest.approx(1.0)
    assert loss.eps == pytest.approx(1.0e-6)


def test_triplet_loss_constructs_with_margin_eps_loss_dtype_and_shape():
    n = _net()
    anchor = _tensor_1d(4, thor.DataType.fp16)
    positive = _tensor_1d(4, thor.DataType.fp16)
    negative = _tensor_1d(4, thor.DataType.fp16)

    loss = thor.losses.metric_learning.TripletLoss(
        n,
        anchor,
        positive,
        negative,
        0.25,
        1.0e-4,
        thor.DataType.fp32,
        thor.losses.LossShape.elementwise,
    )
    assert isinstance(loss, thor.losses.metric_learning.TripletLoss)
    assert loss.margin == pytest.approx(0.25)
    assert loss.eps == pytest.approx(1.0e-4)


@pytest.mark.parametrize("shape", ["batch", "classwise", "elementwise", "raw"])
def test_triplet_loss_reported_loss_shape_variants_construct(shape):
    n = _net()
    anchor = _tensor_1d(3)
    positive = _tensor_1d(3)
    negative = _tensor_1d(3)

    loss = thor.losses.metric_learning.TripletLoss(n, anchor, positive, negative, 1.0, 1.0e-6, None, getattr(thor.losses.LossShape, shape))
    assert isinstance(loss, thor.losses.metric_learning.TripletLoss)


def test_triplet_loss_rejects_non_positive_margin_or_eps():
    n = _net()
    anchor = _tensor_1d(4)
    positive = _tensor_1d(4)
    negative = _tensor_1d(4)

    with pytest.raises(ValueError, match=r"margin must be greater than zero"):
        thor.losses.metric_learning.TripletLoss(n, anchor, positive, negative, 0.0)
    with pytest.raises(ValueError, match=r"eps must be greater than zero"):
        thor.losses.metric_learning.TripletLoss(n, anchor, positive, negative, 1.0, 0.0)


def test_triplet_loss_rejects_mismatched_shapes_and_rank():
    n = _net()
    anchor = _tensor_1d(4)
    positive = _tensor_1d(3)
    negative = _tensor_1d(4)

    with pytest.raises(ValueError, match=r"positive dimensions [\s\S]* must match anchor dimensions"):
        thor.losses.metric_learning.TripletLoss(n, anchor, positive, negative)

    with pytest.raises(ValueError, match=r"anchor must be a 1 dimensional embedding tensor"):
        thor.losses.metric_learning.TripletLoss(n, thor.Tensor([2, 2], thor.DataType.fp32), thor.Tensor([2, 2], thor.DataType.fp32), thor.Tensor([2, 2], thor.DataType.fp32))


def test_triplet_loss_rejects_invalid_dtypes_and_duplicate_tensors():
    n = _net()
    anchor = _tensor_1d(4)
    positive = _tensor_1d(4)
    negative = _tensor_1d(4)

    with pytest.raises(ValueError, match=r"loss_data_type must be fp16 or fp32"):
        thor.losses.metric_learning.TripletLoss(n, anchor, positive, negative, 1.0, 1.0e-6, thor.DataType.int32)
    with pytest.raises(ValueError, match=r"anchor must use fp16 or fp32 dtype"):
        thor.losses.metric_learning.TripletLoss(n, _tensor_1d(4, thor.DataType.uint8), positive, negative)
    with pytest.raises(ValueError, match=r"must use the same fp16 or fp32 dtype"):
        thor.losses.metric_learning.TripletLoss(n, anchor, _tensor_1d(4, thor.DataType.fp16), negative)
    with pytest.raises(ValueError, match=r"must be distinct tensors"):
        thor.losses.metric_learning.TripletLoss(n, anchor, anchor, negative)


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
def test_triplet_loss_numerical_forward_matches_reference(reported_loss_shape):
    margin = 0.75
    eps = 1.0e-6
    anchor = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, -0.5, 0.25],
        ],
        dtype=np.float32,
    )
    positive = np.array(
        [
            [0.2, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.5, -0.25, 0.1],
        ],
        dtype=np.float32,
    )
    negative = np.array(
        [
            [0.8, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.6, -0.4, 0.2],
        ],
        dtype=np.float32,
    )

    raw_expected = _triplet_reference(anchor, positive, negative, margin, eps)
    expected = _reduce_loss(raw_expected, reported_loss_shape)
    actual = _run_triplet_loss_network(anchor, positive, negative, margin, eps, reported_loss_shape)

    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
