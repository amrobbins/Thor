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
        thor.losses.LossShape.batch,
    )
    assert isinstance(loss, thor.losses.MeanPowerError)
    assert loss.exponent == pytest.approx(1.25)


def test_mean_power_error_constructs_reports_per_example():
    n = _net()
    preds = _tensor_1d(1, thor.DataType.fp32)
    labels = _tensor_1d(1, thor.DataType.fp32)

    loss = thor.losses.MeanPowerError(
        n,
        preds,
        labels,
        1.5,
        None,
        thor.losses.LossShape.per_example,
    )
    assert isinstance(loss, thor.losses.MeanPowerError)


@pytest.mark.parametrize(
    "reported_loss_shape, expected_dimensions",
    [
        (thor.losses.LossShape.batch, [1]),
        (thor.losses.LossShape.per_example, [1]),
        (thor.losses.LossShape.per_output, [2, 3, 4]),
        (thor.losses.LossShape.raw, [2, 3, 4]),
    ],
)
def test_mean_power_error_constructs_all_reported_loss_shapes(reported_loss_shape, expected_dimensions):
    n = _net()
    preds = thor.Tensor([2, 3, 4], thor.DataType.fp32)
    labels = thor.Tensor([2, 3, 4], thor.DataType.fp32)

    loss = thor.losses.MeanPowerError(n, preds, labels, reported_loss_shape=reported_loss_shape)
    assert loss.get_loss().get_dimensions() == expected_dimensions


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

    with pytest.raises(ValueError, match=r"example_weights must use fp8_e4m3"):
        thor.losses.MeanPowerError(n, preds, labels, example_weights=_tensor_1d(1, thor.DataType.uint32))

    with pytest.raises(ValueError, match=r"example_weights dimensions must be \[1\]"):
        thor.losses.MeanPowerError(n, preds, labels, example_weights=_tensor_1d(3, thor.DataType.fp32))


def _ragged_pair(network, prediction_dtype=thor.DataType.fp32, label_dtype=None):
    if label_dtype is None:
        label_dtype = prediction_dtype
    predictions = thor.layers.RaggedNetworkInput(
        network,
        "ragged_predictions",
        prediction_dtype,
        [2],
        batch_size=3,
        max_total_values=8,
        max_values_per_row=4,
    )
    labels = thor.layers.RaggedNetworkInput(
        network,
        "ragged_labels",
        label_dtype,
        [2],
        partition=predictions,
    )
    return predictions, labels


def test_mean_power_error_constructs_ragged_raw_and_preserves_exponent():
    n = _net()
    predictions, labels = _ragged_pair(n, thor.DataType.bf16, thor.DataType.int32)
    loss = thor.losses.MeanPowerError(
        n,
        predictions,
        labels,
        exponent=1.25,
        reported_loss_shape=thor.losses.LossShape.raw,
    )
    assert loss.is_ragged
    assert loss.exponent == pytest.approx(1.25)
    assert loss.get_predictions() == predictions
    assert loss.get_labels() == labels
    assert isinstance(loss.get_loss(), thor.RaggedTensor)
    assert loss.get_loss().offsets == predictions.offsets
    assert loss.get_raw_loss().values.get_data_type() == thor.DataType.fp32


def test_mean_power_error_ragged_rejects_per_output_and_different_partition():
    n = _net()
    predictions, labels = _ragged_pair(n)
    with pytest.raises(ValueError, match=r"per_output.*undefined"):
        thor.losses.MeanPowerError(
            n,
            predictions,
            labels,
            reported_loss_shape=thor.losses.LossShape.per_output,
        )

    different_labels = thor.layers.RaggedNetworkInput(
        n,
        "different_labels",
        thor.DataType.fp32,
        [2],
        batch_size=3,
        max_total_values=8,
        max_values_per_row=4,
    )
    with pytest.raises(ValueError, match=r"exact same row partition"):
        thor.losses.MeanPowerError(n, predictions, different_labels)


def test_mean_power_error_ragged_accepts_dense_per_row_example_weights():
    n = _net()
    predictions, labels = _ragged_pair(n)
    weights_input = thor.layers.NetworkInput(n, "weights", [1], thor.DataType.bf16)
    weights = weights_input.get_feature_output()
    loss = thor.losses.MeanPowerError(n, predictions, labels, exponent=1.75, example_weights=weights)
    assert loss.example_weights == weights
    assert loss.get_example_weights() == weights
    assert loss.exponent == pytest.approx(1.75)


@pytest.mark.parametrize(
    "dtype, expected_loss_dtype",
    [
        (thor.DataType.fp8_e4m3, thor.DataType.fp32),
        (thor.DataType.fp8_e5m2, thor.DataType.fp32),
        (thor.DataType.fp16, thor.DataType.fp16),
        (thor.DataType.bf16, thor.DataType.fp32),
        (thor.DataType.fp32, thor.DataType.fp32),
    ],
)
def test_mean_power_error_ragged_matches_dense_prediction_dtype_contract(dtype, expected_loss_dtype):
    n = _net()
    predictions, labels = _ragged_pair(n, dtype, thor.DataType.int32)
    loss = thor.losses.MeanPowerError(
        n,
        predictions,
        labels,
        exponent=1.5,
        reported_loss_shape=thor.losses.LossShape.raw,
    )
    assert loss.get_raw_loss().values.get_data_type() == expected_loss_dtype
