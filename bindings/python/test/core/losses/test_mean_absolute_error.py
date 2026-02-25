# test/test_losses_mean_absolute_error.py
import pytest
import thor


def _net():
    return thor.Network("test_net_mean_absolute_error")


def _tensor_1d(size: int, dtype=thor.DataType.fp32):
    return thor.Tensor([size], dtype)


def test_mean_absolute_error_constructs_defaults():
    n = _net()
    preds = _tensor_1d(1, thor.DataType.fp32)
    labels = _tensor_1d(1, thor.DataType.fp32)

    loss = thor.losses.MeanAbsoluteError(n, preds, labels)
    assert loss is not None
    assert isinstance(loss, thor.losses.MeanAbsoluteError)


def test_mean_absolute_error_constructs_with_loss_data_type():
    n = _net()
    preds = _tensor_1d(1, thor.DataType.fp16)
    labels = _tensor_1d(1, thor.DataType.fp16)

    loss = thor.losses.MeanAbsoluteError(
        n,
        preds,
        labels,
        thor.DataType.fp32,  # explicitly override builder.lossDataType(...)
        False,
    )
    assert isinstance(loss, thor.losses.MeanAbsoluteError)


def test_mean_absolute_error_constructs_reports_elementwise():
    n = _net()
    preds = _tensor_1d(1, thor.DataType.fp32)
    labels = _tensor_1d(1, thor.DataType.fp32)

    loss = thor.losses.MeanAbsoluteError(
        n,
        preds,
        labels,
        None,
        True,
    )
    assert isinstance(loss, thor.losses.MeanAbsoluteError)


def test_mean_absolute_error_rejects_labels_not_1d_size_1():
    n = _net()
    preds = _tensor_1d(1, thor.DataType.fp32)

    labels = _tensor_1d(2, thor.DataType.fp32)
    with pytest.raises(ValueError, match=r"labels must be a 1 dimensional tensor of size 1"):
        thor.losses.MeanAbsoluteError(n, preds, labels)

    labels = thor.Tensor([1, 1], thor.DataType.fp32)
    with pytest.raises(ValueError, match=r"labels must be a 1 dimensional tensor of size 1"):
        thor.losses.MeanAbsoluteError(n, preds, labels)


def test_mean_absolute_error_rejects_wrong_types():
    n = _net()
    preds = _tensor_1d(1, thor.DataType.fp32)
    labels = _tensor_1d(1, thor.DataType.fp32)

    with pytest.raises(TypeError):
        thor.losses.MeanAbsoluteError("not a network", preds, labels)

    with pytest.raises(TypeError):
        thor.losses.MeanAbsoluteError(n, "not a tensor", labels)

    with pytest.raises(TypeError):
        thor.losses.MeanAbsoluteError(n, preds, "not a tensor")


def test_mean_absolute_error_rejects_wrong_arity():
    n = _net()
    preds = _tensor_1d(1, thor.DataType.fp32)
    labels = _tensor_1d(1, thor.DataType.fp32)

    with pytest.raises(TypeError):
        thor.losses.MeanAbsoluteError(n, preds)  # missing labels

    with pytest.raises(TypeError):
        thor.losses.MeanAbsoluteError(n, preds, labels, None, False, 123)  # extra arg
