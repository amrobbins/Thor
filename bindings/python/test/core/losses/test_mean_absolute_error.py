# test/test_losses_mae.py
import pytest
import thor


def _net():
    return thor.Network("test_net_mae")


def _tensor_1d(size: int, dtype=thor.DataType.fp32):
    return thor.Tensor([size], dtype)


def test_mae_constructs_defaults():
    n = _net()
    preds = _tensor_1d(1, thor.DataType.fp32)
    labels = _tensor_1d(1, thor.DataType.fp32)

    loss = thor.losses.MAE(n, preds, labels)
    assert loss is not None
    assert isinstance(loss, thor.losses.MAE)


def test_mae_constructs_vector_width_100():
    n = _net()
    preds = _tensor_1d(100, thor.DataType.fp32)
    labels = _tensor_1d(100, thor.DataType.fp32)

    loss = thor.losses.MAE(n, preds, labels)
    assert loss is not None
    assert isinstance(loss, thor.losses.MAE)


def test_mae_constructs_with_loss_data_type():
    n = _net()
    preds = _tensor_1d(1, thor.DataType.fp16)
    labels = _tensor_1d(1, thor.DataType.fp16)

    loss = thor.losses.MAE(
        n,
        preds,
        labels,
        thor.DataType.fp32,  # explicitly override builder.lossDataType(...)
        thor.losses.LossShape.batch,
    )
    assert isinstance(loss, thor.losses.MAE)


def test_mae_constructs_reports_per_example():
    n = _net()
    preds = _tensor_1d(1, thor.DataType.fp32)
    labels = _tensor_1d(1, thor.DataType.fp32)

    loss = thor.losses.MAE(
        n,
        preds,
        labels,
        None,
        thor.losses.LossShape.per_example,
    )
    assert isinstance(loss, thor.losses.MAE)


@pytest.mark.parametrize(
    "reported_loss_shape, expected_dimensions",
    [
        (thor.losses.LossShape.batch, [1]),
        (thor.losses.LossShape.per_example, [1]),
        (thor.losses.LossShape.per_output, [2, 3, 4]),
        (thor.losses.LossShape.raw, [2, 3, 4]),
    ],
)
def test_mae_constructs_all_reported_loss_shapes(reported_loss_shape, expected_dimensions):
    n = _net()
    preds = thor.Tensor([2, 3, 4], thor.DataType.fp32)
    labels = thor.Tensor([2, 3, 4], thor.DataType.fp32)

    loss = thor.losses.MAE(n, preds, labels, reported_loss_shape=reported_loss_shape)
    assert loss.get_loss().get_dimensions() == expected_dimensions


def test_mae_rejects_mismatched_label_dimensions():
    n = _net()
    preds = _tensor_1d(1, thor.DataType.fp32)

    labels = _tensor_1d(2, thor.DataType.fp32)
    with pytest.raises(ValueError, match=r"predictions and labels dimensions must match"):
        thor.losses.MAE(n, preds, labels)

    labels = thor.Tensor([1, 1], thor.DataType.fp32)
    with pytest.raises(ValueError, match=r"predictions and labels dimensions must match"):
        thor.losses.MAE(n, preds, labels)


def test_mae_rejects_wrong_types():
    n = _net()
    preds = _tensor_1d(1, thor.DataType.fp32)
    labels = _tensor_1d(1, thor.DataType.fp32)

    with pytest.raises(TypeError):
        thor.losses.MAE("not a network", preds, labels)

    with pytest.raises(TypeError):
        thor.losses.MAE(n, "not a tensor", labels)

    with pytest.raises(TypeError):
        thor.losses.MAE(n, preds, "not a tensor")


def test_mae_rejects_wrong_arity():
    n = _net()
    preds = _tensor_1d(1, thor.DataType.fp32)
    labels = _tensor_1d(1, thor.DataType.fp32)

    with pytest.raises(TypeError):
        thor.losses.MAE(n, preds)  # missing labels

    with pytest.raises(TypeError):
        thor.losses.MAE(n, preds, labels, None, thor.losses.LossShape.batch, 123, 456)  # extra arg


def test_mae_constructs_with_scalar_example_weights():
    n = _net()
    preds = _tensor_1d(4, thor.DataType.fp32)
    labels = _tensor_1d(4, thor.DataType.fp32)
    weights = _tensor_1d(1, thor.DataType.fp32)

    loss = thor.losses.MAE(n, preds, labels, example_weights=weights)
    assert isinstance(loss, thor.losses.MAE)
    assert loss.example_weights == weights
    assert loss.get_example_weights() == weights


def test_mae_constructs_with_elementwise_example_weights():
    n = _net()
    preds = _tensor_1d(4, thor.DataType.fp32)
    labels = _tensor_1d(4, thor.DataType.fp32)
    weights = _tensor_1d(4, thor.DataType.fp16)

    loss = thor.losses.MAE(n, preds, labels, example_weights=weights)
    assert isinstance(loss, thor.losses.MAE)
    assert loss.example_weights == weights


def test_mae_rejects_bad_example_weights():
    n = _net()
    preds = _tensor_1d(4, thor.DataType.fp32)
    labels = _tensor_1d(4, thor.DataType.fp32)

    with pytest.raises(ValueError, match=r"example_weights must be distinct"):
        thor.losses.MAE(n, preds, labels, example_weights=labels)

    with pytest.raises(ValueError, match=r"example_weights must use fp8_e4m3"):
        thor.losses.MAE(n, preds, labels, example_weights=_tensor_1d(1, thor.DataType.uint32))

    with pytest.raises(ValueError, match=r"example_weights dimensions must be \[1\]"):
        thor.losses.MAE(n, preds, labels, example_weights=_tensor_1d(3, thor.DataType.fp32))
