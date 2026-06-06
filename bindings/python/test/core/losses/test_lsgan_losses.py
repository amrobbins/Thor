import numpy as np
import pytest
import thor


def _net(name="test_net_lsgan_loss"):
    return thor.Network(name)


def _tensor_1d(size: int, dtype=thor.DataType.fp32):
    return thor.Tensor([size], dtype)


def _cpu_tensor(values: np.ndarray, dtype: thor.DataType) -> thor.physical.PhysicalTensor:
    values = np.asarray(values, dtype=thor.physical.numpy_dtypes.from_thor(dtype), order="C")
    placement = thor.physical.Placement(thor.physical.DeviceType.cpu, 0)
    descriptor = thor.physical.PhysicalTensor.Descriptor(dtype, list(values.shape))
    tensor = thor.physical.PhysicalTensor(placement, descriptor)
    tensor.numpy()[...] = values
    return tensor


def _discriminator_reference(
    real_scores: np.ndarray,
    fake_scores: np.ndarray,
    real_target: float = 1.0,
    fake_target: float = 0.0,
) -> np.ndarray:
    real_scores = real_scores.astype(np.float32)
    fake_scores = fake_scores.astype(np.float32)
    real_diff = real_scores - np.float32(real_target)
    fake_diff = fake_scores - np.float32(fake_target)
    return (np.float32(0.5) * (real_diff * real_diff + fake_diff * fake_diff)).astype(np.float32)


def _generator_reference(fake_scores: np.ndarray, target: float = 1.0) -> np.ndarray:
    fake_scores = fake_scores.astype(np.float32)
    diff = fake_scores - np.float32(target)
    return (np.float32(0.5) * diff * diff).astype(np.float32)


def _reduce_loss(raw: np.ndarray, reported_loss_shape: thor.losses.LossShape) -> np.ndarray:
    if reported_loss_shape == thor.losses.LossShape.raw:
        return raw
    if reported_loss_shape == thor.losses.LossShape.elementwise:
        return np.sum(raw, axis=1, keepdims=True).astype(np.float32)

    batch_size = raw.shape[0]
    if reported_loss_shape == thor.losses.LossShape.classwise:
        return (np.sum(raw, axis=0, keepdims=True) / batch_size).astype(np.float32)
    if reported_loss_shape == thor.losses.LossShape.batch:
        return np.array([[np.sum(raw) / batch_size]], dtype=np.float32)
    raise AssertionError(f"Unhandled loss shape: {reported_loss_shape}")


def _run_discriminator_loss_network(
    real_scores: np.ndarray,
    fake_scores: np.ndarray,
    reported_loss_shape: thor.losses.LossShape,
    real_target: float = 1.0,
    fake_target: float = 0.0,
) -> np.ndarray:
    shape_name = str(reported_loss_shape).split(".")[-1]
    n = thor.Network(f"test_net_lsgan_discriminator_loss_numerical_{shape_name}")
    dtype = thor.DataType.fp32
    feature_dims = list(real_scores.shape[1:])
    real_input = thor.layers.NetworkInput(n, "real_scores", feature_dims, dtype)
    fake_input = thor.layers.NetworkInput(n, "fake_scores", feature_dims, dtype)
    loss = thor.losses.LSGANDiscriminatorLoss(
        n,
        real_input.get_feature_output(),
        fake_input.get_feature_output(),
        dtype,
        reported_loss_shape,
        real_target,
        fake_target,
    )
    thor.layers.NetworkOutput(n, "loss", loss.get_loss(), dtype)

    placed = n.place(
        real_scores.shape[0],
        inference_only=True,
        forced_devices=[0],
        forced_num_stamps_per_gpu=1,
    )
    outputs = placed.infer(
        {
            "real_scores": _cpu_tensor(real_scores, dtype),
            "fake_scores": _cpu_tensor(fake_scores, dtype),
        }
    )
    assert set(outputs.keys()) == {"loss"}
    return np.array(outputs["loss"].numpy(), copy=True)


def _run_generator_loss_network(
    fake_scores: np.ndarray,
    reported_loss_shape: thor.losses.LossShape,
    target: float = 1.0,
) -> np.ndarray:
    shape_name = str(reported_loss_shape).split(".")[-1]
    n = thor.Network(f"test_net_lsgan_generator_loss_numerical_{shape_name}")
    dtype = thor.DataType.fp32
    feature_dims = list(fake_scores.shape[1:])
    fake_input = thor.layers.NetworkInput(n, "fake_scores", feature_dims, dtype)
    loss = thor.losses.LSGANGeneratorLoss(
        n,
        fake_input.get_feature_output(),
        dtype,
        reported_loss_shape,
        target,
    )
    thor.layers.NetworkOutput(n, "loss", loss.get_loss(), dtype)

    placed = n.place(
        fake_scores.shape[0],
        inference_only=True,
        forced_devices=[0],
        forced_num_stamps_per_gpu=1,
    )
    outputs = placed.infer({"fake_scores": _cpu_tensor(fake_scores, dtype)})
    assert set(outputs.keys()) == {"loss"}
    return np.array(outputs["loss"].numpy(), copy=True)


def test_lsgan_discriminator_loss_constructs_defaults():
    n = _net()
    real_scores = _tensor_1d(4)
    fake_scores = _tensor_1d(4)

    loss = thor.losses.LSGANDiscriminatorLoss(n, real_scores, fake_scores)
    assert isinstance(loss, thor.losses.LSGANDiscriminatorLoss)
    assert loss.get_real_scores() == real_scores
    assert loss.get_fake_scores() == fake_scores
    assert loss.real_target == pytest.approx(1.0)
    assert loss.fake_target == pytest.approx(0.0)


def test_lsgan_generator_loss_constructs_defaults():
    n = _net()
    fake_scores = _tensor_1d(4)

    loss = thor.losses.LSGANGeneratorLoss(n, fake_scores)
    assert isinstance(loss, thor.losses.LSGANGeneratorLoss)
    assert loss.get_fake_scores() == fake_scores
    assert loss.target == pytest.approx(1.0)


def test_lsgan_losses_construct_with_loss_dtype_shape_and_targets():
    n = _net()
    real_scores = _tensor_1d(4, thor.DataType.fp16)
    fake_scores = _tensor_1d(4, thor.DataType.fp16)

    d_loss = thor.losses.LSGANDiscriminatorLoss(
        n,
        real_scores,
        fake_scores,
        thor.DataType.fp32,
        thor.losses.LossShape.elementwise,
        0.9,
        -0.1,
    )
    g_loss = thor.losses.LSGANGeneratorLoss(
        n,
        fake_scores,
        thor.DataType.fp32,
        thor.losses.LossShape.raw,
        0.8,
    )
    assert isinstance(d_loss, thor.losses.LSGANDiscriminatorLoss)
    assert isinstance(g_loss, thor.losses.LSGANGeneratorLoss)
    assert d_loss.real_target == pytest.approx(0.9)
    assert d_loss.fake_target == pytest.approx(-0.1)
    assert g_loss.target == pytest.approx(0.8)


@pytest.mark.parametrize("shape", ["batch", "classwise", "elementwise", "raw"])
def test_lsgan_loss_reported_loss_shape_variants_construct(shape):
    n = _net()
    real_scores = _tensor_1d(3)
    fake_scores = _tensor_1d(3)

    d_loss = thor.losses.LSGANDiscriminatorLoss(
        n,
        real_scores,
        fake_scores,
        None,
        getattr(thor.losses.LossShape, shape),
    )
    g_loss = thor.losses.LSGANGeneratorLoss(
        n,
        fake_scores,
        None,
        getattr(thor.losses.LossShape, shape),
    )
    assert isinstance(d_loss, thor.losses.LSGANDiscriminatorLoss)
    assert isinstance(g_loss, thor.losses.LSGANGeneratorLoss)


def test_lsgan_discriminator_loss_rejects_mismatched_shapes_dtypes_and_duplicate_tensors():
    n = _net()
    real_scores = _tensor_1d(4)
    fake_scores = _tensor_1d(4)

    with pytest.raises(ValueError, match=r"fake_scores dimensions [\s\S]* must match real_scores dimensions"):
        thor.losses.LSGANDiscriminatorLoss(n, real_scores, _tensor_1d(3))
    with pytest.raises(ValueError, match=r"loss_data_type must be fp16 or fp32"):
        thor.losses.LSGANDiscriminatorLoss(n, real_scores, fake_scores, thor.DataType.int32)
    with pytest.raises(ValueError, match=r"real_scores must use fp16 or fp32 dtype"):
        thor.losses.LSGANDiscriminatorLoss(n, _tensor_1d(4, thor.DataType.uint8), fake_scores)
    with pytest.raises(ValueError, match=r"same fp16 or fp32 dtype"):
        thor.losses.LSGANDiscriminatorLoss(n, real_scores, _tensor_1d(4, thor.DataType.fp16))
    with pytest.raises(ValueError, match=r"real_scores and fake_scores must be distinct tensors"):
        thor.losses.LSGANDiscriminatorLoss(n, real_scores, real_scores)


def test_lsgan_generator_loss_rejects_invalid_shape_and_dtype():
    n = _net()

    with pytest.raises(ValueError, match=r"fake_scores must be a non-empty 1D score tensor"):
        thor.losses.LSGANGeneratorLoss(n, thor.Tensor([2, 2], thor.DataType.fp32))
    with pytest.raises(ValueError, match=r"loss_data_type must be fp16 or fp32"):
        thor.losses.LSGANGeneratorLoss(n, _tensor_1d(4), thor.DataType.int32)
    with pytest.raises(ValueError, match=r"fake_scores must use fp16 or fp32 dtype"):
        thor.losses.LSGANGeneratorLoss(n, _tensor_1d(4, thor.DataType.uint8))


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
def test_lsgan_discriminator_loss_numerical_forward_matches_reference(reported_loss_shape):
    real_scores = np.array(
        [
            [1.5, 0.2, -0.5],
            [0.9, 1.0, 2.0],
            [-1.2, 0.4, 1.1],
            [2.5, -0.1, 0.8],
        ],
        dtype=np.float32,
    )
    fake_scores = np.array(
        [
            [-1.5, -0.2, 0.5],
            [-0.9, -1.0, -2.0],
            [1.2, -0.4, -1.1],
            [-2.5, 0.1, -0.8],
        ],
        dtype=np.float32,
    )
    real_target = 0.9
    fake_target = -0.2

    raw_expected = _discriminator_reference(real_scores, fake_scores, real_target, fake_target)
    expected = _reduce_loss(raw_expected, reported_loss_shape)
    actual = _run_discriminator_loss_network(real_scores, fake_scores, reported_loss_shape, real_target, fake_target)

    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)


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
def test_lsgan_generator_loss_numerical_forward_matches_reference(reported_loss_shape):
    fake_scores = np.array(
        [
            [-1.5, -0.2, 0.5],
            [-0.9, -1.0, -2.0],
            [1.2, -0.4, -1.1],
            [-2.5, 0.1, -0.8],
        ],
        dtype=np.float32,
    )
    target = 0.85

    raw_expected = _generator_reference(fake_scores, target)
    expected = _reduce_loss(raw_expected, reported_loss_shape)
    actual = _run_generator_loss_network(fake_scores, reported_loss_shape, target)

    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
