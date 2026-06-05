import numpy as np
import pytest
import thor


def _net():
    return thor.Network("test_net_margin_ranking_loss")


def _tensor_1d(size: int, dtype=thor.DataType.fp32):
    return thor.Tensor([size], dtype)


def _cpu_tensor(values: np.ndarray, dtype: thor.DataType) -> thor.physical.PhysicalTensor:
    values = np.asarray(values, dtype=thor.physical.numpy_dtypes.from_thor(dtype), order="C")
    placement = thor.physical.Placement(thor.physical.DeviceType.cpu, 0)
    descriptor = thor.physical.PhysicalTensor.Descriptor(dtype, list(values.shape))
    tensor = thor.physical.PhysicalTensor(placement, descriptor)
    tensor.numpy()[...] = values
    return tensor


def _margin_ranking_reference(input1: np.ndarray, input2: np.ndarray, target: np.ndarray, margin: float) -> np.ndarray:
    input1 = input1.astype(np.float32)
    input2 = input2.astype(np.float32)
    target = target.astype(np.float32)
    return np.maximum(np.float32(margin) - target * (input1 - input2), np.float32(0.0)).astype(np.float32)


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


def _run_margin_ranking_loss_network(
    input1: np.ndarray,
    input2: np.ndarray,
    target: np.ndarray,
    margin: float,
    reported_loss_shape: thor.losses.LossShape,
) -> np.ndarray:
    shape_name = str(reported_loss_shape).split(".")[-1]
    n = thor.Network(f"test_net_margin_ranking_loss_numerical_{shape_name}")
    dtype = thor.DataType.fp32
    feature_dims = list(input1.shape[1:])
    input1_layer = thor.layers.NetworkInput(n, "input1", feature_dims, dtype)
    input2_layer = thor.layers.NetworkInput(n, "input2", feature_dims, dtype)
    target_layer = thor.layers.NetworkInput(n, "target", feature_dims, dtype)
    loss = thor.losses.MarginRankingLoss(
        n,
        input1_layer.get_feature_output(),
        input2_layer.get_feature_output(),
        target_layer.get_feature_output(),
        margin,
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


def test_margin_ranking_loss_constructs_defaults():
    n = _net()
    input1 = _tensor_1d(4)
    input2 = _tensor_1d(4)
    target = _tensor_1d(4)

    loss = thor.losses.MarginRankingLoss(n, input1, input2, target)
    assert isinstance(loss, thor.losses.MarginRankingLoss)
    assert loss.margin == pytest.approx(0.0)


def test_margin_ranking_loss_constructs_with_margin_loss_dtype_and_shape():
    n = _net()
    input1 = _tensor_1d(4, thor.DataType.fp16)
    input2 = _tensor_1d(4, thor.DataType.fp16)
    target = _tensor_1d(4, thor.DataType.int32)

    loss = thor.losses.MarginRankingLoss(
        n,
        input1,
        input2,
        target,
        0.25,
        thor.DataType.fp32,
        thor.losses.LossShape.elementwise,
    )
    assert isinstance(loss, thor.losses.MarginRankingLoss)
    assert loss.margin == pytest.approx(0.25)


@pytest.mark.parametrize("shape", ["batch", "classwise", "elementwise", "raw"])
def test_margin_ranking_loss_reported_loss_shape_variants_construct(shape):
    n = _net()
    input1 = _tensor_1d(3)
    input2 = _tensor_1d(3)
    target = _tensor_1d(3)

    loss = thor.losses.MarginRankingLoss(n, input1, input2, target, 0.0, None, getattr(thor.losses.LossShape, shape))
    assert isinstance(loss, thor.losses.MarginRankingLoss)


def test_margin_ranking_loss_rejects_negative_margin():
    n = _net()
    input1 = _tensor_1d(4)
    input2 = _tensor_1d(4)
    target = _tensor_1d(4)

    with pytest.raises(ValueError, match=r"margin must be non-negative"):
        thor.losses.MarginRankingLoss(n, input1, input2, target, -0.25)


def test_margin_ranking_loss_rejects_mismatched_shapes():
    n = _net()
    input1 = _tensor_1d(4)
    input2 = _tensor_1d(3)
    target = _tensor_1d(4)

    with pytest.raises(ValueError, match=r"input2 dimensions [\s\S]* must match input1 dimensions"):
        thor.losses.MarginRankingLoss(n, input1, input2, target)

    with pytest.raises(ValueError, match=r"target dimensions [\s\S]* must match input1 dimensions"):
        thor.losses.MarginRankingLoss(n, input1, _tensor_1d(4), _tensor_1d(3))


def test_margin_ranking_loss_rejects_invalid_dtypes_and_duplicate_tensors():
    n = _net()
    input1 = _tensor_1d(4)
    input2 = _tensor_1d(4)
    target = _tensor_1d(4)

    with pytest.raises(ValueError, match=r"loss_data_type must be fp16 or fp32"):
        thor.losses.MarginRankingLoss(n, input1, input2, target, 0.0, thor.DataType.int32)
    with pytest.raises(ValueError, match=r"input1 must use fp16 or fp32 dtype"):
        thor.losses.MarginRankingLoss(n, _tensor_1d(4, thor.DataType.uint8), input2, target)
    with pytest.raises(ValueError, match=r"same fp16 or fp32 dtype"):
        thor.losses.MarginRankingLoss(n, input1, _tensor_1d(4, thor.DataType.fp16), target)
    with pytest.raises(ValueError, match=r"target must use int8, int16, int32, int64, fp16, or fp32 dtype"):
        thor.losses.MarginRankingLoss(n, input1, input2, _tensor_1d(4, thor.DataType.uint8))
    with pytest.raises(ValueError, match=r"must be distinct tensors"):
        thor.losses.MarginRankingLoss(n, input1, input1, target)


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
def test_margin_ranking_loss_numerical_forward_matches_reference(reported_loss_shape):
    margin = 0.4
    input1 = np.array(
        [
            [1.2, -0.1, 0.3],
            [0.5, 1.1, -1.3],
            [-0.7, 0.8, 2.0],
            [1.5, -1.0, 0.2],
        ],
        dtype=np.float32,
    )
    input2 = np.array(
        [
            [0.4, 0.2, 0.8],
            [0.7, 0.1, -0.5],
            [-0.9, 1.4, 1.0],
            [0.2, -0.4, -0.1],
        ],
        dtype=np.float32,
    )
    target = np.array(
        [
            [1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
        dtype=np.float32,
    )

    raw_expected = _margin_ranking_reference(input1, input2, target, margin)
    expected = _reduce_loss(raw_expected, reported_loss_shape)
    actual = _run_margin_ranking_loss_network(input1, input2, target, margin, reported_loss_shape)

    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
