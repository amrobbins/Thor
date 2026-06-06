import numpy as np
import pytest
import thor


def _net(name="test_net_hinge_gan_loss"):
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


def _discriminator_reference(real_scores: np.ndarray, fake_scores: np.ndarray) -> np.ndarray:
    real_scores = real_scores.astype(np.float32)
    fake_scores = fake_scores.astype(np.float32)
    return (
        np.maximum(np.float32(1.0) - real_scores, np.float32(0.0))
        + np.maximum(np.float32(1.0) + fake_scores, np.float32(0.0))
    ).astype(np.float32)


def _generator_reference(fake_scores: np.ndarray) -> np.ndarray:
    return (-fake_scores.astype(np.float32)).astype(np.float32)


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
) -> np.ndarray:
    shape_name = str(reported_loss_shape).split(".")[-1]
    n = thor.Network(f"test_net_hinge_gan_discriminator_loss_numerical_{shape_name}")
    dtype = thor.DataType.fp32
    feature_dims = list(real_scores.shape[1:])
    real_input = thor.layers.NetworkInput(n, "real_scores", feature_dims, dtype)
    fake_input = thor.layers.NetworkInput(n, "fake_scores", feature_dims, dtype)
    loss = thor.losses.gan.HingeGANDiscriminatorLoss(
        n,
        real_input.get_feature_output(),
        fake_input.get_feature_output(),
        dtype,
        reported_loss_shape,
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
) -> np.ndarray:
    shape_name = str(reported_loss_shape).split(".")[-1]
    n = thor.Network(f"test_net_hinge_gan_generator_loss_numerical_{shape_name}")
    dtype = thor.DataType.fp32
    feature_dims = list(fake_scores.shape[1:])
    fake_input = thor.layers.NetworkInput(n, "fake_scores", feature_dims, dtype)
    loss = thor.losses.gan.HingeGANGeneratorLoss(
        n,
        fake_input.get_feature_output(),
        dtype,
        reported_loss_shape,
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


def test_hinge_gan_discriminator_loss_constructs_defaults():
    n = _net()
    real_scores = _tensor_1d(4)
    fake_scores = _tensor_1d(4)

    loss = thor.losses.gan.HingeGANDiscriminatorLoss(n, real_scores, fake_scores)
    assert isinstance(loss, thor.losses.gan.HingeGANDiscriminatorLoss)
    assert loss.get_real_scores() == real_scores
    assert loss.get_fake_scores() == fake_scores


def test_hinge_gan_generator_loss_constructs_defaults():
    n = _net()
    fake_scores = _tensor_1d(4)

    loss = thor.losses.gan.HingeGANGeneratorLoss(n, fake_scores)
    assert isinstance(loss, thor.losses.gan.HingeGANGeneratorLoss)
    assert loss.get_fake_scores() == fake_scores


def test_hinge_gan_losses_construct_with_loss_dtype_and_shape():
    n = _net()
    real_scores = _tensor_1d(4, thor.DataType.fp16)
    fake_scores = _tensor_1d(4, thor.DataType.fp16)

    d_loss = thor.losses.gan.HingeGANDiscriminatorLoss(
        n,
        real_scores,
        fake_scores,
        thor.DataType.fp32,
        thor.losses.LossShape.elementwise,
    )
    g_loss = thor.losses.gan.HingeGANGeneratorLoss(
        n,
        fake_scores,
        thor.DataType.fp32,
        thor.losses.LossShape.raw,
    )
    assert isinstance(d_loss, thor.losses.gan.HingeGANDiscriminatorLoss)
    assert isinstance(g_loss, thor.losses.gan.HingeGANGeneratorLoss)


@pytest.mark.parametrize("shape", ["batch", "classwise", "elementwise", "raw"])
def test_hinge_gan_loss_reported_loss_shape_variants_construct(shape):
    n = _net()
    real_scores = _tensor_1d(3)
    fake_scores = _tensor_1d(3)

    d_loss = thor.losses.gan.HingeGANDiscriminatorLoss(
        n,
        real_scores,
        fake_scores,
        None,
        getattr(thor.losses.LossShape, shape),
    )
    g_loss = thor.losses.gan.HingeGANGeneratorLoss(
        n,
        fake_scores,
        None,
        getattr(thor.losses.LossShape, shape),
    )
    assert isinstance(d_loss, thor.losses.gan.HingeGANDiscriminatorLoss)
    assert isinstance(g_loss, thor.losses.gan.HingeGANGeneratorLoss)


def test_hinge_gan_discriminator_loss_rejects_mismatched_shapes_dtypes_and_duplicate_tensors():
    n = _net()
    real_scores = _tensor_1d(4)
    fake_scores = _tensor_1d(4)

    with pytest.raises(ValueError, match=r"fake_scores dimensions [\s\S]* must match real_scores dimensions"):
        thor.losses.gan.HingeGANDiscriminatorLoss(n, real_scores, _tensor_1d(3))
    with pytest.raises(ValueError, match=r"loss_data_type must be fp16 or fp32"):
        thor.losses.gan.HingeGANDiscriminatorLoss(n, real_scores, fake_scores, thor.DataType.int32)
    with pytest.raises(ValueError, match=r"real_scores must use fp16 or fp32 dtype"):
        thor.losses.gan.HingeGANDiscriminatorLoss(n, _tensor_1d(4, thor.DataType.uint8), fake_scores)
    with pytest.raises(ValueError, match=r"same fp16 or fp32 dtype"):
        thor.losses.gan.HingeGANDiscriminatorLoss(n, real_scores, _tensor_1d(4, thor.DataType.fp16))
    with pytest.raises(ValueError, match=r"real_scores and fake_scores must be distinct tensors"):
        thor.losses.gan.HingeGANDiscriminatorLoss(n, real_scores, real_scores)


def test_hinge_gan_generator_loss_rejects_invalid_shape_and_dtype():
    n = _net()

    with pytest.raises(ValueError, match=r"fake_scores must be a non-empty 1D score tensor"):
        thor.losses.gan.HingeGANGeneratorLoss(n, thor.Tensor([2, 2], thor.DataType.fp32))
    with pytest.raises(ValueError, match=r"loss_data_type must be fp16 or fp32"):
        thor.losses.gan.HingeGANGeneratorLoss(n, _tensor_1d(4), thor.DataType.int32)
    with pytest.raises(ValueError, match=r"fake_scores must use fp16 or fp32 dtype"):
        thor.losses.gan.HingeGANGeneratorLoss(n, _tensor_1d(4, thor.DataType.uint8))


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
def test_hinge_gan_discriminator_loss_numerical_forward_matches_reference(reported_loss_shape):
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

    raw_expected = _discriminator_reference(real_scores, fake_scores)
    expected = _reduce_loss(raw_expected, reported_loss_shape)
    actual = _run_discriminator_loss_network(real_scores, fake_scores, reported_loss_shape)

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
def test_hinge_gan_generator_loss_numerical_forward_matches_reference(reported_loss_shape):
    fake_scores = np.array(
        [
            [-1.5, -0.2, 0.5],
            [-0.9, -1.0, -2.0],
            [1.2, -0.4, -1.1],
            [-2.5, 0.1, -0.8],
        ],
        dtype=np.float32,
    )

    raw_expected = _generator_reference(fake_scores)
    expected = _reduce_loss(raw_expected, reported_loss_shape)
    actual = _run_generator_loss_network(fake_scores, reported_loss_shape)

    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
