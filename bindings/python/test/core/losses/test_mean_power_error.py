import math

import pytest
import thor


def _net():
    return thor.Network("test_net_mean_power_error")


def _tensor_1d(size: int, dtype=thor.DataType.fp32):
    return thor.Tensor([size], dtype)


def test_mean_power_error_constructs_default_exponent():
    n = _net()
    preds = _tensor_1d(1, thor.DataType.fp32)
    labels = _tensor_1d(1, thor.DataType.fp32)

    loss = thor.losses.MeanPowerError(n, preds, labels)
    assert loss is not None
    assert isinstance(loss, thor.losses.MeanPowerError)
    assert loss.exponent == pytest.approx(1.5)


def test_mean_power_error_constructs_mae_and_mse_exponents():
    n = _net()
    preds = _tensor_1d(4, thor.DataType.fp32)
    labels = _tensor_1d(4, thor.DataType.fp32)

    mae_like = thor.losses.MeanPowerError(n, preds, labels, exponent=1.0)
    mse_like = thor.losses.MeanPowerError(n, preds, labels, exponent=2.0)

    assert mae_like.exponent == pytest.approx(1.0)
    assert mse_like.exponent == pytest.approx(2.0)


def test_mean_power_error_constructs_with_loss_data_type():
    n = _net()
    preds = _tensor_1d(1, thor.DataType.fp16)
    labels = _tensor_1d(1, thor.DataType.fp16)

    loss = thor.losses.MeanPowerError(
        n,
        preds,
        labels,
        1.25,
        thor.DataType.fp32,
        False,
    )
    assert isinstance(loss, thor.losses.MeanPowerError)
    assert loss.exponent == pytest.approx(1.25)


def test_mean_power_error_constructs_reports_elementwise():
    n = _net()
    preds = _tensor_1d(1, thor.DataType.fp32)
    labels = _tensor_1d(1, thor.DataType.fp32)

    loss = thor.losses.MeanPowerError(
        n,
        preds,
        labels,
        1.5,
        None,
        True,
    )
    assert isinstance(loss, thor.losses.MeanPowerError)


def test_mean_power_error_rejects_bad_exponents():
    n = _net()
    preds = _tensor_1d(1, thor.DataType.fp32)
    labels = _tensor_1d(1, thor.DataType.fp32)

    for exponent in (0.999, 0.0, -1.0, math.inf, math.nan):
        with pytest.raises(ValueError, match=r"exponent must be finite and greater than or equal to 1.0"):
            thor.losses.MeanPowerError(n, preds, labels, exponent=exponent)


def test_mean_power_error_rejects_mismatched_label_dimensions():
    n = _net()
    preds = _tensor_1d(1, thor.DataType.fp32)

    labels = _tensor_1d(2, thor.DataType.fp32)
    with pytest.raises(ValueError, match=r"predictions and labels dimensions must match"):
        thor.losses.MeanPowerError(n, preds, labels)

    labels = thor.Tensor([1, 1], thor.DataType.fp32)
    with pytest.raises(ValueError, match=r"predictions and labels dimensions must match"):
        thor.losses.MeanPowerError(n, preds, labels)


def test_mean_power_error_exposes_default_and_custom_loss_weight():
    n = _net()
    preds = _tensor_1d(1, thor.DataType.fp32)
    labels = _tensor_1d(1, thor.DataType.fp32)

    default_loss = thor.losses.MeanPowerError(n, preds, labels)
    assert default_loss.loss_weight is None

    explicit_one_loss = thor.losses.MeanPowerError(n, preds, labels, loss_weight=1.0)
    assert explicit_one_loss.loss_weight is None

    weighted_loss = thor.losses.MeanPowerError(n, preds, labels, loss_weight=2.5)
    assert weighted_loss.loss_weight == pytest.approx(2.5)


def test_mean_power_error_constructs_with_scalar_example_weights():
    n = _net()
    preds = _tensor_1d(4, thor.DataType.fp32)
    labels = _tensor_1d(4, thor.DataType.fp32)
    weights = _tensor_1d(1, thor.DataType.fp32)

    loss = thor.losses.MeanPowerError(n, preds, labels, example_weights=weights)
    assert isinstance(loss, thor.losses.MeanPowerError)
    assert loss.example_weights == weights
    assert loss.get_example_weights() == weights


def test_mean_power_error_constructs_with_elementwise_example_weights():
    n = _net()
    preds = _tensor_1d(4, thor.DataType.fp32)
    labels = _tensor_1d(4, thor.DataType.fp32)
    weights = _tensor_1d(4, thor.DataType.fp16)

    loss = thor.losses.MeanPowerError(n, preds, labels, example_weights=weights)
    assert isinstance(loss, thor.losses.MeanPowerError)
    assert loss.example_weights == weights


def test_mean_power_error_rejects_bad_example_weights():
    n = _net()
    preds = _tensor_1d(4, thor.DataType.fp32)
    labels = _tensor_1d(4, thor.DataType.fp32)

    with pytest.raises(ValueError, match=r"example_weights must be distinct"):
        thor.losses.MeanPowerError(n, preds, labels, example_weights=labels)

    with pytest.raises(ValueError, match=r"example_weights must be fp16 or fp32"):
        thor.losses.MeanPowerError(n, preds, labels, example_weights=_tensor_1d(1, thor.DataType.uint32))

    with pytest.raises(ValueError, match=r"example_weights dimensions must be \[1\]"):
        thor.losses.MeanPowerError(n, preds, labels, example_weights=_tensor_1d(3, thor.DataType.fp32))
