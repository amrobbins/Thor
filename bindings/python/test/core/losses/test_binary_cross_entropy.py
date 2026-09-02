# test/test_losses_binary_cross_entropy.py
import pytest
import thor


def _net():
    return thor.Network("test_net_binary_cross_entropy")


def _tensor_1d(size: int, dtype=thor.DataType.fp32):
    return thor.Tensor([size], dtype)


def test_binary_cross_entropy_constructs_defaults():
    n = _net()
    preds = _tensor_1d(1, thor.DataType.fp32)
    labels = _tensor_1d(1, thor.DataType.fp32)

    loss = thor.losses.BinaryCrossEntropy(n, preds, labels)
    assert loss is not None
    assert isinstance(loss, thor.losses.BinaryCrossEntropy)


def test_binary_cross_entropy_constructs_vector_width_100():
    n = _net()
    preds = _tensor_1d(100, thor.DataType.fp32)
    labels = _tensor_1d(100, thor.DataType.fp32)

    loss = thor.losses.BinaryCrossEntropy(n, preds, labels)
    assert loss is not None
    assert isinstance(loss, thor.losses.BinaryCrossEntropy)


def test_binary_cross_entropy_constructs_fp16():
    n = _net()
    preds = _tensor_1d(1, thor.DataType.fp16)
    labels = _tensor_1d(1, thor.DataType.fp16)

    loss = thor.losses.BinaryCrossEntropy(
        n,
        preds,
        labels,
        thor.DataType.fp16,
    )
    assert isinstance(loss, thor.losses.BinaryCrossEntropy)


def test_binary_cross_entropy_constructs_reports_per_example():
    n = _net()
    preds = _tensor_1d(1, thor.DataType.fp32)
    labels = _tensor_1d(1, thor.DataType.fp32)

    loss = thor.losses.BinaryCrossEntropy(
        n,
        preds,
        labels,
        thor.DataType.fp32,
        thor.losses.LossShape.per_example,
    )
    assert isinstance(loss, thor.losses.BinaryCrossEntropy)


@pytest.mark.parametrize(
    "reported_loss_shape, expected_dimensions",
    [
        (thor.losses.LossShape.batch, [1]),
        (thor.losses.LossShape.per_example, [1]),
        (thor.losses.LossShape.per_output, [2, 3, 4]),
        (thor.losses.LossShape.raw, [2, 3, 4]),
    ],
)
def test_binary_cross_entropy_constructs_multidimensional_outputs(reported_loss_shape, expected_dimensions):
    n = _net()
    preds = thor.Tensor([2, 3, 4], thor.DataType.fp32)
    labels = thor.Tensor([2, 3, 4], thor.DataType.fp32)

    loss = thor.losses.BinaryCrossEntropy(n, preds, labels, reported_loss_shape=reported_loss_shape)
    assert loss.get_loss().get_dimensions() == expected_dimensions


def test_binary_cross_entropy_rejects_mismatched_label_dimensions():
    n = _net()
    preds = _tensor_1d(1, thor.DataType.fp32)
    labels = _tensor_1d(2, thor.DataType.fp32)

    with pytest.raises(ValueError, match=r"predictions and labels dimensions must match"):
        thor.losses.BinaryCrossEntropy(n, preds, labels)


def test_binary_cross_entropy_rejects_invalid_loss_data_type():
    n = _net()
    preds = _tensor_1d(1, thor.DataType.fp32)
    labels = _tensor_1d(1, thor.DataType.fp32)

    with pytest.raises(ValueError, match=r"loss_data_type must be fp16 or fp32"):
        thor.losses.BinaryCrossEntropy(
            n,
            preds,
            labels,
            thor.DataType.int32,
        )


def test_binary_cross_entropy_rejects_wrong_types():
    n = _net()
    preds = _tensor_1d(1, thor.DataType.fp32)
    labels = _tensor_1d(1, thor.DataType.fp32)

    with pytest.raises(TypeError):
        thor.losses.BinaryCrossEntropy("not a network", preds, labels)

    with pytest.raises(TypeError):
        thor.losses.BinaryCrossEntropy(n, "not a tensor", labels)

    with pytest.raises(TypeError):
        thor.losses.BinaryCrossEntropy(n, preds, "not a tensor")


def test_binary_cross_entropy_rejects_wrong_arity():
    n = _net()
    preds = _tensor_1d(1, thor.DataType.fp32)
    labels = _tensor_1d(1, thor.DataType.fp32)

    with pytest.raises(TypeError):
        thor.losses.BinaryCrossEntropy(n, preds)  # missing labels

    with pytest.raises(TypeError):
        thor.losses.BinaryCrossEntropy(n, preds, labels, thor.DataType.fp32, thor.losses.LossShape.batch, 123, 456)  # extra arg


def _r10h_ragged_bce_pair(network, prediction_dtype=thor.DataType.fp32, label_dtype=thor.DataType.uint8):
    predictions = thor.layers.RaggedNetworkInput(
        network,
        "r10h_bce_predictions",
        prediction_dtype,
        [2],
        batch_size=3,
        max_total_values=8,
        max_values_per_row=4,
    )
    labels = thor.layers.RaggedNetworkInput(
        network,
        "r10h_bce_labels",
        label_dtype,
        [2],
        partition=predictions,
    )
    return predictions, labels


def test_binary_cross_entropy_r10h_constructs_ragged_raw_and_preserves_partition():
    n = _net()
    predictions, labels = _r10h_ragged_bce_pair(n)
    loss = thor.losses.BinaryCrossEntropy(
        n,
        predictions,
        labels,
        reported_loss_shape=thor.losses.LossShape.raw,
    )
    assert loss.is_ragged
    assert loss.get_predictions() == predictions
    assert loss.get_labels() == labels
    assert isinstance(loss.get_loss(), thor.RaggedTensor)
    assert loss.get_loss().offsets == predictions.offsets
    assert loss.get_raw_loss().values.get_data_type() == thor.DataType.fp32


def test_binary_cross_entropy_r10h_rejects_per_output_and_different_partition():
    n = _net()
    predictions, labels = _r10h_ragged_bce_pair(n)
    with pytest.raises(ValueError, match=r"per_output.*undefined"):
        thor.losses.BinaryCrossEntropy(n, predictions, labels, reported_loss_shape=thor.losses.LossShape.per_output)

    different_labels = thor.layers.RaggedNetworkInput(
        n,
        "r10h_bce_different_labels",
        thor.DataType.uint8,
        [2],
        batch_size=3,
        max_total_values=8,
        max_values_per_row=4,
    )
    with pytest.raises(ValueError, match=r"exact same row partition"):
        thor.losses.BinaryCrossEntropy(n, predictions, different_labels)
