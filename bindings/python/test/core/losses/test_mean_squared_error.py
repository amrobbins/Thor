# test/test_losses_mse.py
import pytest
import thor


def _net():
    return thor.Network("test_net_mse")


def _tensor_1d(size: int, dtype=thor.DataType.fp32):
    return thor.Tensor([size], dtype)


def test_mse_constructs_defaults():
    n = _net()
    preds = _tensor_1d(1, thor.DataType.fp32)
    labels = _tensor_1d(1, thor.DataType.fp32)

    loss = thor.losses.MSE(n, preds, labels)
    assert loss is not None
    assert isinstance(loss, thor.losses.MSE)


def test_mse_constructs_vector_width_100():
    n = _net()
    preds = _tensor_1d(100, thor.DataType.fp32)
    labels = _tensor_1d(100, thor.DataType.fp32)

    loss = thor.losses.MSE(n, preds, labels)
    assert loss is not None
    assert isinstance(loss, thor.losses.MSE)




def test_mse_exposes_default_and_custom_loss_weight():
    n = _net()
    preds = _tensor_1d(1, thor.DataType.fp32)
    labels = _tensor_1d(1, thor.DataType.fp32)

    default_loss = thor.losses.MSE(n, preds, labels)
    assert default_loss.loss_weight is None

    explicit_one_loss = thor.losses.MSE(n, preds, labels, loss_weight=1.0)
    assert explicit_one_loss.loss_weight is None

    weighted_loss = thor.losses.MSE(n, preds, labels, loss_weight=2.5)
    assert weighted_loss.loss_weight == pytest.approx(2.5)


def test_mse_constructs_with_loss_data_type():
    n = _net()
    preds = _tensor_1d(1, thor.DataType.fp16)
    labels = _tensor_1d(1, thor.DataType.fp16)

    loss = thor.losses.MSE(
        n,
        preds,
        labels,
        thor.DataType.fp32,  # explicitly override builder.lossDataType(...)
        False,
    )
    assert isinstance(loss, thor.losses.MSE)


def test_mse_constructs_reports_elementwise():
    n = _net()
    preds = _tensor_1d(1, thor.DataType.fp32)
    labels = _tensor_1d(1, thor.DataType.fp32)

    loss = thor.losses.MSE(
        n,
        preds,
        labels,
        None,
        True,
    )
    assert isinstance(loss, thor.losses.MSE)


def test_mse_rejects_mismatched_label_dimensions():
    n = _net()
    preds = _tensor_1d(1, thor.DataType.fp32)

    labels = _tensor_1d(2, thor.DataType.fp32)
    with pytest.raises(ValueError, match=r"predictions and labels dimensions must match"):
        thor.losses.MSE(n, preds, labels)

    labels = thor.Tensor([1, 1], thor.DataType.fp32)
    with pytest.raises(ValueError, match=r"predictions and labels dimensions must match"):
        thor.losses.MSE(n, preds, labels)


def test_mse_rejects_wrong_types():
    n = _net()
    preds = _tensor_1d(1, thor.DataType.fp32)
    labels = _tensor_1d(1, thor.DataType.fp32)

    with pytest.raises(TypeError):
        thor.losses.MSE("not a network", preds, labels)

    with pytest.raises(TypeError):
        thor.losses.MSE(n, "not a tensor", labels)

    with pytest.raises(TypeError):
        thor.losses.MSE(n, preds, "not a tensor")


def test_mse_rejects_wrong_arity():
    n = _net()
    preds = _tensor_1d(1, thor.DataType.fp32)
    labels = _tensor_1d(1, thor.DataType.fp32)

    with pytest.raises(TypeError):
        thor.losses.MSE(n, preds)  # missing labels

    with pytest.raises(TypeError):
        thor.losses.MSE(n, preds, labels, None, False, 123, 456)  # extra arg


def test_mse_constructs_with_scalar_example_weights():
    n = _net()
    preds = _tensor_1d(4, thor.DataType.fp32)
    labels = _tensor_1d(4, thor.DataType.fp32)
    weights = _tensor_1d(1, thor.DataType.fp32)

    loss = thor.losses.MSE(n, preds, labels, example_weights=weights)
    assert isinstance(loss, thor.losses.MSE)
    assert loss.example_weights == weights
    assert loss.get_example_weights() == weights


def test_mse_constructs_with_elementwise_example_weights():
    n = _net()
    preds = _tensor_1d(4, thor.DataType.fp32)
    labels = _tensor_1d(4, thor.DataType.fp32)
    weights = _tensor_1d(4, thor.DataType.fp16)

    loss = thor.losses.MSE(n, preds, labels, example_weights=weights)
    assert isinstance(loss, thor.losses.MSE)
    assert loss.example_weights == weights


def test_mse_rejects_bad_example_weights():
    n = _net()
    preds = _tensor_1d(4, thor.DataType.fp32)
    labels = _tensor_1d(4, thor.DataType.fp32)

    with pytest.raises(ValueError, match=r"example_weights must be distinct"):
        thor.losses.MSE(n, preds, labels, example_weights=labels)

    with pytest.raises(ValueError, match=r"example_weights must use fp8_e4m3"):
        thor.losses.MSE(n, preds, labels, example_weights=_tensor_1d(1, thor.DataType.uint32))

    with pytest.raises(ValueError, match=r"example_weights dimensions must be \[1\]"):
        thor.losses.MSE(n, preds, labels, example_weights=_tensor_1d(3, thor.DataType.fp32))
