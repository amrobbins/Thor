# test/test_losses_mape.py
import pytest
import thor


def _net():
    return thor.Network("test_net_mape")


def _tensor_1d(size: int, dtype=thor.DataType.fp32):
    return thor.Tensor([size], dtype)


def test_mape_constructs_defaults():
    n = _net()
    preds = _tensor_1d(1, thor.DataType.fp32)
    labels = _tensor_1d(1, thor.DataType.fp32)

    loss = thor.losses.MAPE(n, preds, labels)
    assert loss is not None
    assert isinstance(loss, thor.losses.MAPE)


def test_mape_constructs_vector_width_100():
    n = _net()
    preds = _tensor_1d(100, thor.DataType.fp32)
    labels = _tensor_1d(100, thor.DataType.fp32)

    loss = thor.losses.MAPE(n, preds, labels)
    assert loss is not None
    assert isinstance(loss, thor.losses.MAPE)


def test_mape_constructs_with_loss_data_type():
    n = _net()
    preds = _tensor_1d(1, thor.DataType.fp16)
    labels = _tensor_1d(1, thor.DataType.fp16)

    loss = thor.losses.MAPE(
        n,
        preds,
        labels,
        thor.DataType.fp32,  # explicitly override builder.lossDataType(...)
        thor.losses.LossShape.batch,
    )
    assert isinstance(loss, thor.losses.MAPE)


def test_mape_constructs_reports_per_example():
    n = _net()
    preds = _tensor_1d(1, thor.DataType.fp32)
    labels = _tensor_1d(1, thor.DataType.fp32)

    loss = thor.losses.MAPE(
        n,
        preds,
        labels,
        None,
        thor.losses.LossShape.per_example,
    )
    assert isinstance(loss, thor.losses.MAPE)


@pytest.mark.parametrize(
    "reported_loss_shape, expected_dimensions",
    [
        (thor.losses.LossShape.batch, [1]),
        (thor.losses.LossShape.per_example, [1]),
        (thor.losses.LossShape.per_output, [2, 3, 4]),
        (thor.losses.LossShape.raw, [2, 3, 4]),
    ],
)
def test_mape_constructs_multidimensional_predictions_and_labels(reported_loss_shape, expected_dimensions):
    n = _net()
    predictions = thor.Tensor([2, 3, 4], thor.DataType.fp32)
    labels = thor.Tensor([2, 3, 4], thor.DataType.fp32)

    loss = thor.losses.MAPE(n, predictions, labels, reported_loss_shape=reported_loss_shape)

    assert loss.get_loss().get_dimensions() == expected_dimensions


def test_mape_rejects_mismatched_label_dimensions():
    n = _net()
    preds = _tensor_1d(1, thor.DataType.fp32)

    labels = _tensor_1d(2, thor.DataType.fp32)
    with pytest.raises(ValueError, match=r"predictions and labels dimensions must match"):
        thor.losses.MAPE(n, preds, labels)

    labels = thor.Tensor([1, 1], thor.DataType.fp32)
    with pytest.raises(ValueError, match=r"predictions and labels dimensions must match"):
        thor.losses.MAPE(n, preds, labels)


def test_mape_rejects_wrong_types():
    n = _net()
    preds = _tensor_1d(1, thor.DataType.fp32)
    labels = _tensor_1d(1, thor.DataType.fp32)

    with pytest.raises(TypeError):
        thor.losses.MAPE("not a network", preds, labels)

    with pytest.raises(TypeError):
        thor.losses.MAPE(n, "not a tensor", labels)

    with pytest.raises(TypeError):
        thor.losses.MAPE(n, preds, "not a tensor")


def test_mape_rejects_wrong_arity():
    n = _net()
    preds = _tensor_1d(1, thor.DataType.fp32)
    labels = _tensor_1d(1, thor.DataType.fp32)

    with pytest.raises(TypeError):
        thor.losses.MAPE(n, preds)  # missing labels

    with pytest.raises(TypeError):
        thor.losses.MAPE(n, preds, labels, None, thor.losses.LossShape.batch, 123, 456)  # extra arg


def _ragged_pair_mape(network):
    predictions = thor.layers.RaggedNetworkInput(
        network, "ragged_predictions", thor.DataType.fp32, [2], batch_size=3, max_total_values=8, max_values_per_row=4
    )
    labels = thor.layers.RaggedNetworkInput(network, "ragged_labels", thor.DataType.uint8, [2], partition=predictions)
    return predictions, labels


def test_mape_constructs_ragged_and_rejects_per_output():
    n = _net()
    predictions, labels = _ragged_pair_mape(n)
    loss = thor.losses.MAPE(n, predictions, labels, thor.DataType.fp32, thor.losses.LossShape.raw)
    assert loss.is_ragged
    assert loss.get_predictions() == predictions
    assert loss.get_labels() == labels
    assert isinstance(loss.get_loss(), thor.RaggedTensor)
    assert loss.get_loss().offsets == predictions.offsets

    n2 = _net()
    predictions2, labels2 = _ragged_pair_mape(n2)
    with pytest.raises(ValueError, match=r"per_output.*undefined"):
        thor.losses.MAPE(n2, predictions2, labels2, reported_loss_shape=thor.losses.LossShape.per_output)
