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


def test_mae_constructs_with_loss_data_type():
    n = _net()
    preds = _tensor_1d(1, thor.DataType.fp16)
    labels = _tensor_1d(1, thor.DataType.fp16)

    loss = thor.losses.MAE(
        n,
        preds,
        labels,
        thor.DataType.fp32,  # explicitly override builder.lossDataType(...)
        False,
    )
    assert isinstance(loss, thor.losses.MAE)


def test_mae_constructs_reports_elementwise():
    n = _net()
    preds = _tensor_1d(1, thor.DataType.fp32)
    labels = _tensor_1d(1, thor.DataType.fp32)

    loss = thor.losses.MAE(
        n,
        preds,
        labels,
        None,
        True,
    )
    assert isinstance(loss, thor.losses.MAE)


def test_mae_rejects_labels_not_1d_size_1():
    n = _net()
    preds = _tensor_1d(1, thor.DataType.fp32)

    labels = _tensor_1d(2, thor.DataType.fp32)
    with pytest.raises(ValueError, match=r"labels must be a 1 dimensional tensor of size 1"):
        thor.losses.MAE(n, preds, labels)

    labels = thor.Tensor([1, 1], thor.DataType.fp32)
    with pytest.raises(ValueError, match=r"labels must be a 1 dimensional tensor of size 1"):
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
        thor.losses.MAE(n, preds, labels, None, False, 123)  # extra arg
