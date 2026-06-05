import numpy as np
import pytest
import thor


def _net():
    return thor.Network("test_net_cosine_embedding_loss")


def _tensor_1d(size: int, dtype=thor.DataType.fp32):
    return thor.Tensor([size], dtype)


def _cpu_tensor(values: np.ndarray, dtype: thor.DataType) -> thor.physical.PhysicalTensor:
    values = np.asarray(values, dtype=thor.physical.numpy_dtypes.from_thor(dtype), order="C")
    placement = thor.physical.Placement(thor.physical.DeviceType.cpu, 0)
    descriptor = thor.physical.PhysicalTensor.Descriptor(dtype, list(values.shape))
    tensor = thor.physical.PhysicalTensor(placement, descriptor)
    tensor.numpy()[...] = values
    return tensor


def _cosine_embedding_reference(input1: np.ndarray, input2: np.ndarray, target: np.ndarray, margin: float, eps: float) -> np.ndarray:
    input1 = input1.astype(np.float32)
    input2 = input2.astype(np.float32)
    target = target.astype(np.float32)
    dot = np.sum(input1 * input2, axis=1, keepdims=True)
    input1_sq = np.sum(input1 * input1, axis=1, keepdims=True) + np.float32(eps)
    input2_sq = np.sum(input2 * input2, axis=1, keepdims=True) + np.float32(eps)
    cosine = dot / np.sqrt(input1_sq * input2_sq)
    positive_loss = np.float32(1.0) - cosine
    negative_loss = np.maximum(cosine - np.float32(margin), np.float32(0.0))
    return np.where(target > 0.0, positive_loss, negative_loss).astype(np.float32)


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


def _run_cosine_embedding_loss_network(
    input1: np.ndarray,
    input2: np.ndarray,
    target: np.ndarray,
    margin: float,
    eps: float,
    reported_loss_shape: thor.losses.LossShape,
) -> np.ndarray:
    shape_name = str(reported_loss_shape).split(".")[-1]
    n = thor.Network(f"test_net_cosine_embedding_loss_numerical_{shape_name}")
    dtype = thor.DataType.fp32
    feature_dims = list(input1.shape[1:])
    input1_layer = thor.layers.NetworkInput(n, "input1", feature_dims, dtype)
    input2_layer = thor.layers.NetworkInput(n, "input2", feature_dims, dtype)
    target_layer = thor.layers.NetworkInput(n, "target", [1], dtype)
    loss = thor.losses.CosineEmbeddingLoss(
        n,
        input1_layer.get_feature_output(),
        input2_layer.get_feature_output(),
        target_layer.get_feature_output(),
        margin,
        eps,
        dtype,
        reported_loss_shape,
    )
    thor.layers.NetworkOutput(n, "loss", loss.get_loss(), dtype)

    placed = n.place(
        input1.shape[0],
        inference_only=True,
        forced_devices=[0],
        forced_num_stamps_per_gpu=1,
    )
    outputs = placed.infer(
        {
            "input1": _cpu_tensor(input1, dtype),
            "input2": _cpu_tensor(input2, dtype),
            "target": _cpu_tensor(target, dtype),
        }
    )
    assert set(outputs.keys()) == {"loss"}
    return np.array(outputs["loss"].numpy(), copy=True)


def test_cosine_embedding_loss_constructs_defaults():
    n = _net()
    input1 = _tensor_1d(4)
    input2 = _tensor_1d(4)
    target = _tensor_1d(1)

    loss = thor.losses.CosineEmbeddingLoss(n, input1, input2, target)
    assert isinstance(loss, thor.losses.CosineEmbeddingLoss)
    assert loss.margin == pytest.approx(0.0)
    assert loss.eps == pytest.approx(1.0e-8)


def test_cosine_embedding_loss_constructs_with_margin_eps_loss_dtype_and_shape():
    n = _net()
    input1 = _tensor_1d(4, thor.DataType.fp16)
    input2 = _tensor_1d(4, thor.DataType.fp16)
    target = _tensor_1d(1, thor.DataType.int32)

    loss = thor.losses.CosineEmbeddingLoss(
        n,
        input1,
        input2,
        target,
        0.25,
        1.0e-6,
        thor.DataType.fp32,
        thor.losses.LossShape.elementwise,
    )
    assert isinstance(loss, thor.losses.CosineEmbeddingLoss)
    assert loss.margin == pytest.approx(0.25)
    assert loss.eps == pytest.approx(1.0e-6)


@pytest.mark.parametrize("shape", ["batch", "classwise", "elementwise", "raw"])
def test_cosine_embedding_loss_reported_loss_shape_variants_construct(shape):
    n = _net()
    input1 = _tensor_1d(3)
    input2 = _tensor_1d(3)
    target = _tensor_1d(1)

    loss = thor.losses.CosineEmbeddingLoss(n, input1, input2, target, 0.0, 1.0e-8, None, getattr(thor.losses.LossShape, shape))
    assert isinstance(loss, thor.losses.CosineEmbeddingLoss)


def test_cosine_embedding_loss_rejects_invalid_margin_or_eps():
    n = _net()
    input1 = _tensor_1d(4)
    input2 = _tensor_1d(4)
    target = _tensor_1d(1)

    with pytest.raises(ValueError, match=r"margin must be between -1 and 1"):
        thor.losses.CosineEmbeddingLoss(n, input1, input2, target, -1.25)
    with pytest.raises(ValueError, match=r"margin must be between -1 and 1"):
        thor.losses.CosineEmbeddingLoss(n, input1, input2, target, 1.25)
    with pytest.raises(ValueError, match=r"eps must be greater than zero"):
        thor.losses.CosineEmbeddingLoss(n, input1, input2, target, 0.0, 0.0)


def test_cosine_embedding_loss_rejects_mismatched_shapes_and_rank():
    n = _net()
    input1 = _tensor_1d(4)
    input2 = _tensor_1d(3)
    target = _tensor_1d(1)

    with pytest.raises(ValueError, match=r"input2 dimensions [\s\S]* must match input1 dimensions"):
        thor.losses.CosineEmbeddingLoss(n, input1, input2, target)

    with pytest.raises(ValueError, match=r"input1 must be a 1 dimensional embedding tensor"):
        thor.losses.CosineEmbeddingLoss(n, thor.Tensor([2, 2], thor.DataType.fp32), thor.Tensor([2, 2], thor.DataType.fp32), target)

    with pytest.raises(ValueError, match=r"target must be a 1 dimensional tensor with exactly one label per example"):
        thor.losses.CosineEmbeddingLoss(n, input1, _tensor_1d(4), _tensor_1d(2))


def test_cosine_embedding_loss_rejects_invalid_dtypes_and_duplicate_tensors():
    n = _net()
    input1 = _tensor_1d(4)
    input2 = _tensor_1d(4)
    target = _tensor_1d(1)

    with pytest.raises(ValueError, match=r"loss_data_type must be fp16 or fp32"):
        thor.losses.CosineEmbeddingLoss(n, input1, input2, target, 0.0, 1.0e-8, thor.DataType.int32)
    with pytest.raises(ValueError, match=r"input1 must use fp16 or fp32 dtype"):
        thor.losses.CosineEmbeddingLoss(n, _tensor_1d(4, thor.DataType.uint8), input2, target)
    with pytest.raises(ValueError, match=r"same fp16 or fp32 dtype"):
        thor.losses.CosineEmbeddingLoss(n, input1, _tensor_1d(4, thor.DataType.fp16), target)
    with pytest.raises(ValueError, match=r"target must use int8, int16, int32, int64, fp16, or fp32 dtype"):
        thor.losses.CosineEmbeddingLoss(n, input1, input2, _tensor_1d(1, thor.DataType.uint8))
    with pytest.raises(ValueError, match=r"must be distinct tensors"):
        thor.losses.CosineEmbeddingLoss(n, input1, input1, target)


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
def test_cosine_embedding_loss_numerical_forward_matches_reference(reported_loss_shape):
    margin = 0.2
    eps = 1.0e-6
    input1 = np.array(
        [
            [1.0, 2.0, -0.5],
            [0.5, -1.0, 1.5],
            [-1.0, 0.25, 0.75],
            [2.0, 0.5, -1.5],
        ],
        dtype=np.float32,
    )
    input2 = np.array(
        [
            [0.75, 1.5, -0.25],
            [1.0, 0.25, -1.0],
            [0.25, -0.5, 1.25],
            [-0.75, 1.0, 0.5],
        ],
        dtype=np.float32,
    )
    target = np.array([[1.0], [-1.0], [1.0], [-1.0]], dtype=np.float32)

    raw_expected = _cosine_embedding_reference(input1, input2, target, margin, eps)
    expected = _reduce_loss(raw_expected, reported_loss_shape)
    actual = _run_cosine_embedding_loss_network(input1, input2, target, margin, eps, reported_loss_shape)

    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
