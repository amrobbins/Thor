import json

import numpy as np
import pytest
import thor


def _net():
    return thor.Network("test_net_poisson_nll_loss")


def _tensor_1d(size: int, dtype=thor.DataType.fp32):
    return thor.Tensor([size], dtype)


def _cpu_tensor(values: np.ndarray, dtype: thor.DataType) -> thor.physical.PhysicalTensor:
    values = np.asarray(values, dtype=thor.physical.numpy_dtypes.from_thor(dtype), order="C")
    placement = thor.physical.Placement(thor.physical.DeviceType.cpu, 0)
    descriptor = thor.physical.PhysicalTensor.Descriptor(dtype, list(values.shape))
    tensor = thor.physical.PhysicalTensor(placement, descriptor)
    tensor.numpy()[...] = values
    return tensor


def _poisson_nll_reference(predictions: np.ndarray, labels: np.ndarray, log_input: bool, full: bool, eps: float) -> np.ndarray:
    predictions = predictions.astype(np.float32)
    labels = labels.astype(np.float32)
    if log_input:
        loss = np.exp(predictions) - labels * predictions
    else:
        loss = predictions - labels * np.log(predictions + eps)
    if full:
        stirling = labels * np.log(np.maximum(labels, 1.0)) - labels + 0.5 * np.log(2.0 * np.pi * np.maximum(labels, 1.0))
        loss = loss + np.where(labels > 1.0, stirling, 0.0)
    return loss.astype(np.float32)


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


def _run_poisson_nll_loss_network(
    predictions: np.ndarray,
    labels: np.ndarray,
    log_input: bool,
    full: bool,
    eps: float,
    reported_loss_shape: thor.losses.LossShape,
) -> np.ndarray:
    shape_name = str(reported_loss_shape).split(".")[-1]
    n = thor.Network(f"test_net_poisson_nll_loss_numerical_{shape_name}_{log_input}_{full}")
    dtype = thor.DataType.fp32
    feature_dims = list(predictions.shape[1:])
    predictions_input = thor.layers.NetworkInput(n, "predictions", feature_dims, dtype)
    labels_input = thor.layers.NetworkInput(n, "labels", feature_dims, dtype)
    loss = thor.losses.distribution.PoissonNLLLoss(
        n,
        predictions_input.get_feature_output(),
        labels_input.get_feature_output(),
        log_input,
        full,
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


def test_poisson_nll_loss_constructs_defaults():
    n = _net()
    preds = _tensor_1d(5)
    labels = _tensor_1d(5)

    loss = thor.losses.distribution.PoissonNLLLoss(n, preds, labels)
    assert isinstance(loss, thor.losses.distribution.PoissonNLLLoss)
    assert loss.log_input is True
    assert loss.full is False
    assert loss.eps == pytest.approx(1.0e-8)


def test_poisson_nll_loss_constructs_with_options_loss_dtype_and_shape():
    n = _net()
    preds = _tensor_1d(4, thor.DataType.fp16)
    labels = _tensor_1d(4, thor.DataType.uint16)

    loss = thor.losses.distribution.PoissonNLLLoss(
        n,
        preds,
        labels,
        False,
        True,
        1.0e-5,
        thor.DataType.fp32,
        thor.losses.LossShape.elementwise,
    )
    assert isinstance(loss, thor.losses.distribution.PoissonNLLLoss)
    assert loss.log_input is False
    assert loss.full is True
    assert loss.eps == pytest.approx(1.0e-5)


@pytest.mark.parametrize("shape", ["batch", "classwise", "elementwise", "raw"])
def test_poisson_nll_loss_reported_loss_shape_variants_construct(shape):
    n = _net()
    preds = _tensor_1d(3)
    labels = _tensor_1d(3)

    loss = thor.losses.distribution.PoissonNLLLoss(n, preds, labels, True, False, 1.0e-8, None, getattr(thor.losses.LossShape, shape))
    assert isinstance(loss, thor.losses.distribution.PoissonNLLLoss)


def test_poisson_nll_loss_rejects_mismatched_labels():
    n = _net()
    preds = _tensor_1d(2)
    labels = _tensor_1d(3)

    with pytest.raises(ValueError, match=r"labels dimensions [\s\S]* must match predictions dimensions"):
        thor.losses.distribution.PoissonNLLLoss(n, preds, labels)


def test_poisson_nll_loss_rejects_predictions_not_1d():
    n = _net()
    preds = thor.Tensor([1, 1], thor.DataType.fp32)
    labels = thor.Tensor([1, 1], thor.DataType.fp32)

    with pytest.raises(ValueError, match=r"predictions must be a 1 dimensional tensor"):
        thor.losses.distribution.PoissonNLLLoss(n, preds, labels)


def test_poisson_nll_loss_rejects_integer_predictions():
    n = _net()
    preds = _tensor_1d(3, thor.DataType.uint16)
    labels = _tensor_1d(3)

    with pytest.raises(ValueError, match=r"predictions must use fp16 or fp32 dtype"):
        thor.losses.distribution.PoissonNLLLoss(n, preds, labels)


def test_poisson_nll_loss_rejects_invalid_loss_data_type():
    n = _net()
    preds = _tensor_1d(1)
    labels = _tensor_1d(1)

    with pytest.raises(ValueError, match=r"loss_data_type must be fp16 or fp32"):
        thor.losses.distribution.PoissonNLLLoss(n, preds, labels, True, False, 1.0e-8, thor.DataType.int32)


def test_poisson_nll_loss_rejects_non_positive_eps():
    n = _net()
    preds = _tensor_1d(1)
    labels = _tensor_1d(1)

    with pytest.raises(ValueError, match=r"eps must be greater than zero"):
        thor.losses.distribution.PoissonNLLLoss(n, preds, labels, False, False, 0.0)


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
@pytest.mark.parametrize("log_input,full", [(True, False), (True, True), (False, False), (False, True)])
def test_poisson_nll_loss_numerical_forward_matches_reference(reported_loss_shape, log_input, full):
    eps = 1.0e-5
    if log_input:
        predictions = np.array(
            [
                [0.0, 0.25, 1.25, -0.5],
                [0.75, -0.25, 0.5, 1.5],
            ],
            dtype=np.float32,
        )
    else:
        predictions = np.array(
            [
                [0.5, 1.25, 3.5, 0.75],
                [2.0, 0.25, 1.5, 4.0],
            ],
            dtype=np.float32,
        )
    labels = np.array(
        [
            [0.0, 1.0, 2.0, 4.0],
            [3.0, 0.0, 5.0, 1.0],
        ],
        dtype=np.float32,
    )

    raw_expected = _poisson_nll_reference(predictions, labels, log_input, full, eps)
    expected = _reduce_loss(raw_expected, reported_loss_shape)
    actual = _run_poisson_nll_loss_network(predictions, labels, log_input, full, eps, reported_loss_shape)

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)



def test_poisson_nll_loss_save_load_round_trip_serializes_support_layers(tmp_path):
    n = thor.Network("test_net_poisson_nll_loss_round_trip")
    dtype = thor.DataType.fp32
    predictions_input = thor.layers.NetworkInput(n, "predictions", [4], dtype)
    labels_input = thor.layers.NetworkInput(n, "labels", [4], dtype)
    loss = thor.losses.distribution.PoissonNLLLoss(
        n,
        predictions_input.get_feature_output(),
        labels_input.get_feature_output(),
        False,
        True,
        1.0e-5,
        dtype,
        thor.losses.LossShape.elementwise,
    )
    thor.layers.NetworkOutput(n, "loss", loss.get_loss(), dtype)

    arch = json.loads(n.get_architecture_json())
    assert sum(1 for layer in arch["layers"] if layer["layer_type"] == "custom_loss") == 1
    assert sum(1 for layer in arch["layers"] if layer["layer_type"] == "loss_shaper") == 1

    save_dir = tmp_path / "poisson_nll_model"
    n.save(str(save_dir), overwrite=False)
    loaded = thor.Network("test_net_poisson_nll_loss_round_trip")
    loaded.load(str(save_dir))
    loaded_arch = json.loads(loaded.get_architecture_json())
    assert sum(1 for layer in loaded_arch["layers"] if layer["layer_type"] == "custom_loss") == 1
    assert sum(1 for layer in loaded_arch["layers"] if layer["layer_type"] == "loss_shaper") == 1
