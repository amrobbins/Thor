import numpy as np
import pytest
import thor


def _net(name="test_net_wasserstein_gan_loss"):
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


def _critic_reference(real_scores: np.ndarray, fake_scores: np.ndarray) -> np.ndarray:
    return (fake_scores.astype(np.float32) - real_scores.astype(np.float32)).astype(np.float32)


def _generator_reference(fake_scores: np.ndarray) -> np.ndarray:
    return (-fake_scores.astype(np.float32)).astype(np.float32)


def _gp_reference(
    real_scores: np.ndarray,
    fake_scores: np.ndarray,
    sample_gradients: np.ndarray,
    gradient_penalty_weight: float,
    target_gradient_norm: float,
) -> np.ndarray:
    critic = _critic_reference(real_scores, fake_scores)
    flat = sample_gradients.astype(np.float32).reshape(sample_gradients.shape[0], -1)
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    penalty = gradient_penalty_weight * np.square(norms - target_gradient_norm)
    return (critic + penalty.astype(np.float32)).astype(np.float32)


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


def _run_critic_loss_network(
    real_scores: np.ndarray,
    fake_scores: np.ndarray,
    reported_loss_shape: thor.losses.LossShape,
) -> np.ndarray:
    shape_name = str(reported_loss_shape).split(".")[-1]
    n = thor.Network(f"test_net_wasserstein_gan_critic_loss_numerical_{shape_name}")
    dtype = thor.DataType.fp32
    feature_dims = list(real_scores.shape[1:])
    real_input = thor.layers.NetworkInput(n, "real_scores", feature_dims, dtype)
    fake_input = thor.layers.NetworkInput(n, "fake_scores", feature_dims, dtype)
    loss = thor.losses.gan.WassersteinGANCriticLoss(
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
    n = thor.Network(f"test_net_wasserstein_gan_generator_loss_numerical_{shape_name}")
    dtype = thor.DataType.fp32
    feature_dims = list(fake_scores.shape[1:])
    fake_input = thor.layers.NetworkInput(n, "fake_scores", feature_dims, dtype)
    loss = thor.losses.gan.WassersteinGANGeneratorLoss(
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


def _run_gp_loss_network(
    real_scores: np.ndarray,
    fake_scores: np.ndarray,
    sample_gradients: np.ndarray,
    reported_loss_shape: thor.losses.LossShape,
    gradient_penalty_weight: float,
    target_gradient_norm: float,
) -> np.ndarray:
    shape_name = str(reported_loss_shape).split(".")[-1]
    n = thor.Network(f"test_net_wasserstein_gan_gp_loss_numerical_{shape_name}")
    dtype = thor.DataType.fp32
    real_input = thor.layers.NetworkInput(n, "real_scores", list(real_scores.shape[1:]), dtype)
    fake_input = thor.layers.NetworkInput(n, "fake_scores", list(fake_scores.shape[1:]), dtype)
    gradient_input = thor.layers.NetworkInput(n, "sample_gradients", list(sample_gradients.shape[1:]), dtype)
    loss = thor.losses.gan.WassersteinGANCriticGradientPenaltyLoss(
        n,
        real_input.get_feature_output(),
        fake_input.get_feature_output(),
        gradient_input.get_feature_output(),
        gradient_penalty_weight,
        target_gradient_norm,
        1.0e-12,
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
            "sample_gradients": _cpu_tensor(sample_gradients, dtype),
        }
    )
    assert set(outputs.keys()) == {"loss"}
    return np.array(outputs["loss"].numpy(), copy=True)


def test_wasserstein_gan_losses_construct_defaults():
    n = _net()
    real_scores = _tensor_1d(4)
    fake_scores = _tensor_1d(4)

    critic_loss = thor.losses.gan.WassersteinGANCriticLoss(n, real_scores, fake_scores)
    generator_loss = thor.losses.gan.WassersteinGANGeneratorLoss(n, fake_scores)

    assert isinstance(critic_loss, thor.losses.gan.WassersteinGANCriticLoss)
    assert isinstance(generator_loss, thor.losses.gan.WassersteinGANGeneratorLoss)
    assert critic_loss.get_real_scores() == real_scores
    assert critic_loss.get_fake_scores() == fake_scores
    assert generator_loss.get_fake_scores() == fake_scores


def test_wasserstein_gan_gradient_penalty_loss_constructs_defaults():
    n = _net()
    real_scores = _tensor_1d(1)
    fake_scores = _tensor_1d(1)
    sample_gradients = thor.Tensor([3, 4], thor.DataType.fp32)

    loss = thor.losses.gan.WassersteinGANCriticGradientPenaltyLoss(n, real_scores, fake_scores, sample_gradients)

    assert isinstance(loss, thor.losses.gan.WassersteinGANCriticGradientPenaltyLoss)
    assert loss.get_real_scores() == real_scores
    assert loss.get_fake_scores() == fake_scores
    assert loss.get_sample_gradients() == sample_gradients
    assert loss.gradient_penalty_weight == pytest.approx(10.0)
    assert loss.target_gradient_norm == pytest.approx(1.0)
    assert loss.eps == pytest.approx(1.0e-12)


@pytest.mark.parametrize("shape", ["batch", "classwise", "elementwise", "raw"])
def test_wasserstein_gan_loss_reported_loss_shape_variants_construct(shape):
    n = _net()
    real_scores = _tensor_1d(3)
    fake_scores = _tensor_1d(3)

    critic_loss = thor.losses.gan.WassersteinGANCriticLoss(
        n,
        real_scores,
        fake_scores,
        None,
        getattr(thor.losses.LossShape, shape),
    )
    generator_loss = thor.losses.gan.WassersteinGANGeneratorLoss(
        n,
        fake_scores,
        None,
        getattr(thor.losses.LossShape, shape),
    )
    assert isinstance(critic_loss, thor.losses.gan.WassersteinGANCriticLoss)
    assert isinstance(generator_loss, thor.losses.gan.WassersteinGANGeneratorLoss)


@pytest.mark.parametrize("shape", ["batch", "classwise", "elementwise", "raw"])
def test_wasserstein_gan_gradient_penalty_loss_reported_loss_shape_variants_construct(shape):
    n = _net()
    real_scores = _tensor_1d(1)
    fake_scores = _tensor_1d(1)
    sample_gradients = thor.Tensor([2, 2], thor.DataType.fp32)

    loss = thor.losses.gan.WassersteinGANCriticGradientPenaltyLoss(
        n,
        real_scores,
        fake_scores,
        sample_gradients,
        7.5,
        1.25,
        1.0e-8,
        None,
        getattr(thor.losses.LossShape, shape),
    )
    assert isinstance(loss, thor.losses.gan.WassersteinGANCriticGradientPenaltyLoss)


def test_wasserstein_gan_critic_loss_rejects_mismatched_shapes_dtypes_and_duplicate_tensors():
    n = _net()
    real_scores = _tensor_1d(4)
    fake_scores = _tensor_1d(4)

    with pytest.raises(ValueError, match=r"fake_scores dimensions [\s\S]* must match real_scores dimensions"):
        thor.losses.gan.WassersteinGANCriticLoss(n, real_scores, _tensor_1d(3))
    with pytest.raises(ValueError, match=r"loss_data_type must be fp16 or fp32"):
        thor.losses.gan.WassersteinGANCriticLoss(n, real_scores, fake_scores, thor.DataType.int32)
    with pytest.raises(ValueError, match=r"real_scores must use fp16 or fp32 dtype"):
        thor.losses.gan.WassersteinGANCriticLoss(n, _tensor_1d(4, thor.DataType.uint8), fake_scores)
    with pytest.raises(ValueError, match=r"same fp16 or fp32 dtype"):
        thor.losses.gan.WassersteinGANCriticLoss(n, real_scores, _tensor_1d(4, thor.DataType.fp16))
    with pytest.raises(ValueError, match=r"real_scores and fake_scores must be distinct tensors"):
        thor.losses.gan.WassersteinGANCriticLoss(n, real_scores, real_scores)


def test_wasserstein_gan_generator_loss_rejects_invalid_shape_and_dtype():
    n = _net()

    with pytest.raises(ValueError, match=r"fake_scores must be a non-empty 1D score tensor"):
        thor.losses.gan.WassersteinGANGeneratorLoss(n, thor.Tensor([2, 2], thor.DataType.fp32))
    with pytest.raises(ValueError, match=r"loss_data_type must be fp16 or fp32"):
        thor.losses.gan.WassersteinGANGeneratorLoss(n, _tensor_1d(4), thor.DataType.int32)
    with pytest.raises(ValueError, match=r"fake_scores must use fp16 or fp32 dtype"):
        thor.losses.gan.WassersteinGANGeneratorLoss(n, _tensor_1d(4, thor.DataType.uint8))


def test_wasserstein_gan_gradient_penalty_loss_rejects_invalid_arguments():
    n = _net()
    real_scores = _tensor_1d(1)
    fake_scores = _tensor_1d(1)
    sample_gradients = thor.Tensor([3], thor.DataType.fp32)

    with pytest.raises(ValueError, match=r"real_scores must be a scalar 1D score tensor"):
        thor.losses.gan.WassersteinGANCriticGradientPenaltyLoss(n, _tensor_1d(2), _tensor_1d(2), sample_gradients)
    with pytest.raises(ValueError, match=r"fake_scores dimensions"):
        thor.losses.gan.WassersteinGANCriticGradientPenaltyLoss(n, real_scores, _tensor_1d(2), sample_gradients)
    with pytest.raises(ValueError, match=r"sample_gradients must use fp16 or fp32 dtype"):
        thor.losses.gan.WassersteinGANCriticGradientPenaltyLoss(n, real_scores, fake_scores, thor.Tensor([3], thor.DataType.uint8))
    with pytest.raises(ValueError, match=r"gradient_penalty_weight must be non-negative"):
        thor.losses.gan.WassersteinGANCriticGradientPenaltyLoss(n, real_scores, fake_scores, sample_gradients, -1.0)
    with pytest.raises(ValueError, match=r"target_gradient_norm must be greater than zero"):
        thor.losses.gan.WassersteinGANCriticGradientPenaltyLoss(n, real_scores, fake_scores, sample_gradients, 10.0, 0.0)
    with pytest.raises(ValueError, match=r"eps must be greater than zero"):
        thor.losses.gan.WassersteinGANCriticGradientPenaltyLoss(n, real_scores, fake_scores, sample_gradients, 10.0, 1.0, 0.0)
    with pytest.raises(ValueError, match=r"must be distinct tensors"):
        thor.losses.gan.WassersteinGANCriticGradientPenaltyLoss(n, real_scores, fake_scores, real_scores)


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
def test_wasserstein_gan_critic_loss_numerical_forward_matches_reference(reported_loss_shape):
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

    raw_expected = _critic_reference(real_scores, fake_scores)
    expected = _reduce_loss(raw_expected, reported_loss_shape)
    actual = _run_critic_loss_network(real_scores, fake_scores, reported_loss_shape)

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
def test_wasserstein_gan_generator_loss_numerical_forward_matches_reference(reported_loss_shape):
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
def test_wasserstein_gan_gradient_penalty_loss_numerical_forward_matches_reference(reported_loss_shape):
    real_scores = np.array([[0.5], [1.0], [-0.25], [1.75]], dtype=np.float32)
    fake_scores = np.array([[-1.5], [0.25], [1.0], [-0.5]], dtype=np.float32)
    sample_gradients = np.array(
        [
            [[0.5, 0.0], [0.0, 0.5]],
            [[1.0, 0.0], [0.0, 0.0]],
            [[0.5, 0.5], [0.5, 0.5]],
            [[2.0, 0.0], [0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    gradient_penalty_weight = 7.5
    target_gradient_norm = 1.25

    raw_expected = _gp_reference(
        real_scores,
        fake_scores,
        sample_gradients,
        gradient_penalty_weight,
        target_gradient_norm,
    )
    expected = _reduce_loss(raw_expected, reported_loss_shape)
    actual = _run_gp_loss_network(
        real_scores,
        fake_scores,
        sample_gradients,
        reported_loss_shape,
        gradient_penalty_weight,
        target_gradient_norm,
    )

    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
