import json

import numpy as np
import pytest
import thor


def _net(name: str):
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


def _gamma_nll_reference(predictions: np.ndarray, labels: np.ndarray, eps: float) -> np.ndarray:
    mean = np.maximum(predictions.astype(np.float32), eps)
    target = labels.astype(np.float32)
    return (np.log(mean) + target / mean).astype(np.float32)


def _tweedie_reference(predictions: np.ndarray, labels: np.ndarray, power: float, eps: float) -> np.ndarray:
    mean = np.maximum(predictions.astype(np.float32), eps)
    target = np.maximum(labels.astype(np.float32), 0.0)
    safe_target = np.maximum(target, eps)

    if np.isclose(power, 0.0, atol=1.0e-6):
        loss = (target - mean) ** 2
    elif np.isclose(power, 1.0, atol=1.0e-6):
        loss = 2.0 * (target * np.log(safe_target / mean) - target + mean)
    elif np.isclose(power, 2.0, atol=1.0e-6):
        loss = 2.0 * (np.log(mean / safe_target) + target / mean - 1.0)
    else:
        p = np.float32(power)
        loss = 2.0 * (
            safe_target ** (2.0 - p) / ((1.0 - p) * (2.0 - p))
            - target * mean ** (1.0 - p) / (1.0 - p)
            + mean ** (2.0 - p) / (2.0 - p)
        )
    return loss.astype(np.float32)


def _run_gamma_nll_loss_network(
    predictions: np.ndarray,
    labels: np.ndarray,
    eps: float,
    reported_loss_shape: thor.losses.LossShape,
) -> np.ndarray:
    shape_name = str(reported_loss_shape).split(".")[-1]
    n = thor.Network(f"test_net_gamma_nll_loss_numerical_{shape_name}")
    dtype = thor.DataType.fp32
    feature_dims = list(predictions.shape[1:])
    predictions_input = thor.layers.NetworkInput(n, "predictions", feature_dims, dtype)
    labels_input = thor.layers.NetworkInput(n, "labels", feature_dims, dtype)
    loss = thor.losses.GammaNLLLoss(
        n,
        predictions_input.get_feature_output(),
        labels_input.get_feature_output(),
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
    outputs = placed.infer({"predictions": _cpu_tensor(predictions, dtype), "labels": _cpu_tensor(labels, dtype)})
    assert set(outputs.keys()) == {"loss"}
    return np.array(outputs["loss"].numpy(), copy=True)


def _run_tweedie_loss_network(
    predictions: np.ndarray,
    labels: np.ndarray,
    power: float,
    eps: float,
    reported_loss_shape: thor.losses.LossShape,
) -> np.ndarray:
    shape_name = str(reported_loss_shape).split(".")[-1]
    n = thor.Network(f"test_net_tweedie_loss_numerical_{shape_name}_{power}")
    dtype = thor.DataType.fp32
    feature_dims = list(predictions.shape[1:])
    predictions_input = thor.layers.NetworkInput(n, "predictions", feature_dims, dtype)
    labels_input = thor.layers.NetworkInput(n, "labels", feature_dims, dtype)
    loss = thor.losses.TweedieLoss(
        n,
        predictions_input.get_feature_output(),
        labels_input.get_feature_output(),
        power,
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
    outputs = placed.infer({"predictions": _cpu_tensor(predictions, dtype), "labels": _cpu_tensor(labels, dtype)})
    assert set(outputs.keys()) == {"loss"}
    return np.array(outputs["loss"].numpy(), copy=True)


def test_gamma_nll_loss_constructs_defaults():
    n = _net("test_net_gamma_nll_loss")
    preds = _tensor_1d(5)
    labels = _tensor_1d(5)

    loss = thor.losses.GammaNLLLoss(n, preds, labels)
    assert isinstance(loss, thor.losses.GammaNLLLoss)
    assert loss.eps == pytest.approx(1.0e-6)


def test_gamma_nll_loss_constructs_with_options_loss_dtype_and_shape():
    n = _net("test_net_gamma_nll_loss_options")
    preds = _tensor_1d(4, thor.DataType.fp16)
    labels = _tensor_1d(4, thor.DataType.fp16)

    loss = thor.losses.GammaNLLLoss(
        n,
        preds,
        labels,
        1.0e-5,
        thor.DataType.fp32,
        thor.losses.LossShape.elementwise,
    )
    assert isinstance(loss, thor.losses.GammaNLLLoss)
    assert loss.eps == pytest.approx(1.0e-5)


def test_tweedie_loss_constructs_defaults():
    n = _net("test_net_tweedie_loss")
    preds = _tensor_1d(5)
    labels = _tensor_1d(5)

    loss = thor.losses.TweedieLoss(n, preds, labels)
    assert isinstance(loss, thor.losses.TweedieLoss)
    assert loss.power == pytest.approx(1.5)
    assert loss.eps == pytest.approx(1.0e-6)


def test_tweedie_loss_constructs_with_options_loss_dtype_and_shape():
    n = _net("test_net_tweedie_loss_options")
    preds = _tensor_1d(4, thor.DataType.fp16)
    labels = _tensor_1d(4, thor.DataType.fp16)

    loss = thor.losses.TweedieLoss(
        n,
        preds,
        labels,
        2.0,
        1.0e-5,
        thor.DataType.fp32,
        thor.losses.LossShape.elementwise,
    )
    assert isinstance(loss, thor.losses.TweedieLoss)
    assert loss.power == pytest.approx(2.0)
    assert loss.eps == pytest.approx(1.0e-5)


@pytest.mark.parametrize("loss_class", [thor.losses.GammaNLLLoss, thor.losses.TweedieLoss])
@pytest.mark.parametrize("shape", ["batch", "classwise", "elementwise", "raw"])
def test_gamma_tweedie_loss_reported_loss_shape_variants_construct(loss_class, shape):
    n = _net(f"test_net_{loss_class.__name__}_{shape}")
    preds = _tensor_1d(3)
    labels = _tensor_1d(3)
    loss_shape = getattr(thor.losses.LossShape, shape)

    if loss_class is thor.losses.TweedieLoss:
        loss = loss_class(n, preds, labels, 1.5, 1.0e-6, None, loss_shape)
    else:
        loss = loss_class(n, preds, labels, 1.0e-6, None, loss_shape)
    assert isinstance(loss, loss_class)


@pytest.mark.parametrize("loss_class", [thor.losses.GammaNLLLoss, thor.losses.TweedieLoss])
def test_gamma_tweedie_loss_rejects_mismatched_labels(loss_class):
    n = _net(f"test_net_{loss_class.__name__}_mismatch")
    preds = _tensor_1d(2)
    labels = _tensor_1d(3)

    with pytest.raises(ValueError, match=r"labels dimensions [\s\S]* must match predictions dimensions"):
        loss_class(n, preds, labels)


@pytest.mark.parametrize("loss_class", [thor.losses.GammaNLLLoss, thor.losses.TweedieLoss])
def test_gamma_tweedie_loss_rejects_predictions_not_1d(loss_class):
    n = _net(f"test_net_{loss_class.__name__}_not_1d")
    preds = thor.Tensor([1, 1], thor.DataType.fp32)
    labels = thor.Tensor([1, 1], thor.DataType.fp32)

    with pytest.raises(ValueError, match=r"predictions must be a 1 dimensional mean tensor"):
        loss_class(n, preds, labels)


@pytest.mark.parametrize("loss_class", [thor.losses.GammaNLLLoss, thor.losses.TweedieLoss])
def test_gamma_tweedie_loss_rejects_integer_predictions(loss_class):
    n = _net(f"test_net_{loss_class.__name__}_int_pred")
    preds = _tensor_1d(3, thor.DataType.uint16)
    labels = _tensor_1d(3)

    with pytest.raises(ValueError, match=r"predictions must use fp16 or fp32 dtype"):
        loss_class(n, preds, labels)


@pytest.mark.parametrize("loss_class", [thor.losses.GammaNLLLoss, thor.losses.TweedieLoss])
def test_gamma_tweedie_loss_rejects_invalid_loss_data_type(loss_class):
    n = _net(f"test_net_{loss_class.__name__}_invalid_loss_dtype")
    preds = _tensor_1d(1)
    labels = _tensor_1d(1)

    with pytest.raises(ValueError, match=r"loss_data_type must be fp16 or fp32"):
        if loss_class is thor.losses.TweedieLoss:
            loss_class(n, preds, labels, 1.5, 1.0e-6, thor.DataType.int32)
        else:
            loss_class(n, preds, labels, 1.0e-6, thor.DataType.int32)


@pytest.mark.parametrize("loss_class", [thor.losses.GammaNLLLoss, thor.losses.TweedieLoss])
def test_gamma_tweedie_loss_rejects_non_positive_eps(loss_class):
    n = _net(f"test_net_{loss_class.__name__}_bad_eps")
    preds = _tensor_1d(1)
    labels = _tensor_1d(1)

    with pytest.raises(ValueError, match=r"eps must be greater than zero"):
        if loss_class is thor.losses.TweedieLoss:
            loss_class(n, preds, labels, 1.5, 0.0)
        else:
            loss_class(n, preds, labels, 0.0)


def test_tweedie_loss_rejects_non_finite_power():
    n = _net("test_net_tweedie_loss_non_finite_power")
    preds = _tensor_1d(1)
    labels = _tensor_1d(1)

    with pytest.raises(ValueError, match=r"power must be finite"):
        thor.losses.TweedieLoss(n, preds, labels, float("nan"))


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
def test_gamma_nll_loss_numerical_forward_matches_reference(reported_loss_shape):
    eps = 1.0e-5
    predictions = np.array(
        [
            [0.5, 1.25, 3.5, 0.75],
            [2.0, 0.25, 1.5, 4.0],
        ],
        dtype=np.float32,
    )
    labels = np.array(
        [
            [0.25, 1.0, 2.0, 4.0],
            [3.0, 0.5, 5.0, 1.0],
        ],
        dtype=np.float32,
    )

    raw_expected = _gamma_nll_reference(predictions, labels, eps)
    expected = _reduce_loss(raw_expected, reported_loss_shape)
    actual = _run_gamma_nll_loss_network(predictions, labels, eps, reported_loss_shape)

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


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
@pytest.mark.parametrize("power", [0.0, 1.0, 1.5, 2.0])
def test_tweedie_loss_numerical_forward_matches_reference(reported_loss_shape, power):
    eps = 1.0e-5
    predictions = np.array(
        [
            [0.5, 1.25, 3.5, 0.75],
            [2.0, 0.25, 1.5, 4.0],
        ],
        dtype=np.float32,
    )
    labels = np.array(
        [
            [0.25, 1.0, 2.0, 4.0],
            [3.0, 0.5, 5.0, 1.0],
        ],
        dtype=np.float32,
    )

    raw_expected = _tweedie_reference(predictions, labels, power, eps)
    expected = _reduce_loss(raw_expected, reported_loss_shape)
    actual = _run_tweedie_loss_network(predictions, labels, power, eps, reported_loss_shape)

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_gamma_nll_loss_save_load_round_trip_serializes_support_layers(tmp_path):
    n = thor.Network("test_net_gamma_nll_loss_round_trip")
    dtype = thor.DataType.fp32
    predictions_input = thor.layers.NetworkInput(n, "predictions", [4], dtype)
    labels_input = thor.layers.NetworkInput(n, "labels", [4], dtype)
    loss = thor.losses.GammaNLLLoss(
        n,
        predictions_input.get_feature_output(),
        labels_input.get_feature_output(),
        1.0e-5,
        dtype,
        thor.losses.LossShape.elementwise,
    )
    thor.layers.NetworkOutput(n, "loss", loss.get_loss(), dtype)

    arch = json.loads(n.get_architecture_json())
    assert sum(1 for layer in arch["layers"] if layer["layer_type"] == "custom_loss") == 1
    assert sum(1 for layer in arch["layers"] if layer["layer_type"] == "loss_shaper") == 1

    save_dir = tmp_path / "gamma_nll_model"
    n.save(str(save_dir), overwrite=False)
    loaded = thor.Network("test_net_gamma_nll_loss_round_trip")
    loaded.load(str(save_dir))
    loaded_arch = json.loads(loaded.get_architecture_json())
    assert sum(1 for layer in loaded_arch["layers"] if layer["layer_type"] == "custom_loss") == 1
    assert sum(1 for layer in loaded_arch["layers"] if layer["layer_type"] == "loss_shaper") == 1


def test_tweedie_loss_save_load_round_trip_serializes_support_layers(tmp_path):
    n = thor.Network("test_net_tweedie_loss_round_trip")
    dtype = thor.DataType.fp32
    predictions_input = thor.layers.NetworkInput(n, "predictions", [4], dtype)
    labels_input = thor.layers.NetworkInput(n, "labels", [4], dtype)
    loss = thor.losses.TweedieLoss(
        n,
        predictions_input.get_feature_output(),
        labels_input.get_feature_output(),
        1.5,
        1.0e-5,
        dtype,
        thor.losses.LossShape.elementwise,
    )
    thor.layers.NetworkOutput(n, "loss", loss.get_loss(), dtype)

    arch = json.loads(n.get_architecture_json())
    assert sum(1 for layer in arch["layers"] if layer["layer_type"] == "custom_loss") == 1
    assert sum(1 for layer in arch["layers"] if layer["layer_type"] == "loss_shaper") == 1

    save_dir = tmp_path / "tweedie_model"
    n.save(str(save_dir), overwrite=False)
    loaded = thor.Network("test_net_tweedie_loss_round_trip")
    loaded.load(str(save_dir))
    loaded_arch = json.loads(loaded.get_architecture_json())
    assert sum(1 for layer in loaded_arch["layers"] if layer["layer_type"] == "custom_loss") == 1
    assert sum(1 for layer in loaded_arch["layers"] if layer["layer_type"] == "loss_shaper") == 1
