import numpy as np
import pytest
import thor


def _cpu_tensor(array: np.ndarray, dtype: thor.DataType):
    array = np.asarray(array, dtype=thor.physical.numpy_dtypes.from_thor(dtype), order="C")
    placement = thor.physical.Placement(thor.physical.DeviceType.cpu, 0)
    descriptor = thor.physical.PhysicalTensor.Descriptor(dtype, list(array.shape))
    tensor = thor.physical.PhysicalTensor(placement, descriptor)
    tensor.numpy()[...] = array
    return tensor


def _log_softmax(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    shifted = values - np.max(values, axis=-1, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))


def _softmax(values: np.ndarray) -> np.ndarray:
    return np.exp(_log_softmax(values))


def _raw_reference(kind: str, logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    log_probabilities = _log_softmax(logits)
    if kind in {"soft_target", "categorical_ce"}:
        return -(labels * log_probabilities)
    gamma = 1.5
    alpha = 0.35
    probabilities = np.exp(log_probabilities)
    return -(alpha * labels * np.power(1.0 - probabilities, gamma) * log_probabilities)


def _build_loss(kind: str, network, predictions, labels):
    if kind == "soft_target":
        return thor.losses.SoftTargetCrossEntropy(
            network, predictions, labels, reported_loss_shape=thor.losses.LossShape.batch
        )
    if kind == "categorical_focal":
        return thor.losses.classification.CategoricalFocalLoss(
            network,
            predictions,
            labels,
            gamma=1.5,
            alpha=0.35,
            reported_loss_shape=thor.losses.LossShape.batch,
        )
    if kind == "categorical_ce":
        return thor.losses.CategoricalCrossEntropy(
            network, predictions, labels, reported_loss_shape=thor.losses.LossShape.batch
        )
    raise AssertionError(kind)


@pytest.mark.cuda
@pytest.mark.parametrize("kind", ["soft_target", "categorical_focal", "categorical_ce"])
@pytest.mark.parametrize(
    "offsets_dtype,np_offsets_dtype",
    [(thor.DataType.uint32, np.uint32), (thor.DataType.uint64, np.uint64)],
)
def test_r11c_ragged_categorical_losses_match_tokenwise_class_axis_and_ignore_poison(
    kind, offsets_dtype, np_offsets_dtype
):
    batch_size = 3
    capacity = 8
    max_values_per_row = 5
    network = thor.Network(f"pytest_r11c_{kind}_{np_offsets_dtype.__name__}")
    predictions = thor.layers.RaggedNetworkInput(
        network,
        "predictions",
        thor.DataType.fp32,
        [3],
        batch_size=batch_size,
        max_total_values=capacity,
        max_values_per_row=max_values_per_row,
        offsets_data_type=offsets_dtype,
    )
    labels = thor.layers.RaggedNetworkInput(
        network,
        "labels",
        thor.DataType.fp32,
        [3],
        partition=predictions,
    )
    loss = _build_loss(kind, network, predictions, labels)
    thor.layers.NetworkOutput(network, "loss", loss.get_loss(), thor.DataType.fp32)
    placed = network.place(
        batch_size, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1
    )

    active_logits = np.array(
        [
            [1.0, -0.5, 0.25],
            [-1.0, 0.5, 1.5],
            [0.0, 0.0, 0.0],
            [2.0, 0.25, -1.0],
            [-0.25, 1.25, 0.5],
        ],
        dtype=np.float32,
    )
    active_labels = np.array(
        [
            [0.70, 0.20, 0.10],
            [0.00, 1.00, 0.00],
            [0.25, 0.25, 0.50],
            [1.00, 0.00, 0.00],
            [0.10, 0.60, 0.30],
        ],
        dtype=np.float32,
    )
    offsets = np.array([0, 2, 2, 5], dtype=np_offsets_dtype)
    logits_storage = np.full((capacity, 3), np.nan, dtype=np.float32)
    labels_storage = np.full((capacity, 3), np.nan, dtype=np.float32)
    logits_storage[:5] = active_logits
    labels_storage[:5] = active_labels
    physical_predictions = thor.physical.PhysicalRaggedTensor(
        _cpu_tensor(logits_storage, thor.DataType.fp32),
        _cpu_tensor(offsets, offsets_dtype),
        max_values_per_row=max_values_per_row,
    )

    result = placed.infer(
        {
            "predictions": physical_predictions,
            # labels shares the predictions partition, so only its values are external.
            "labels": _cpu_tensor(labels_storage, thor.DataType.fp32),
        }
    )["loss"]

    expected = np.sum(_raw_reference(kind, active_logits, active_labels), dtype=np.float64) / batch_size
    actual = float(np.asarray(result.numpy(), dtype=np.float32).reshape(-1)[0])
    assert actual == pytest.approx(expected, rel=4e-5, abs=4e-5)
