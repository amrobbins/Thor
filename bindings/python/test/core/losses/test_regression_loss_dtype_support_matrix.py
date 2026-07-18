import pytest
import thor


def _tensor(dtype: thor.DataType, width: int = 4) -> thor.Tensor:
    return thor.Tensor([width], dtype)


def _loss_builders():
    return [
        (
            "MSE",
            lambda n, p, y, loss_dtype=None, weights=None: thor.losses.MSE(
                n, p, y, loss_dtype, False, example_weights=weights
            ),
        ),
        (
            "MAE",
            lambda n, p, y, loss_dtype=None, weights=None: thor.losses.MAE(
                n, p, y, loss_dtype, False, example_weights=weights
            ),
        ),
        (
            "MeanPowerError",
            lambda n, p, y, loss_dtype=None, weights=None: thor.losses.MeanPowerError(
                n, p, y, 1.5, loss_dtype, False, example_weights=weights
            ),
        ),
        (
            "QuantileLoss",
            lambda n, p, y, loss_dtype=None, weights=None: thor.losses.QuantileLoss(
                n, p, y, 0.1, loss_dtype, thor.losses.LossShape.raw, example_weights=weights
            ),
        ),
        (
            "ExpectileLoss",
            lambda n, p, y, loss_dtype=None, weights=None: thor.losses.ExpectileLoss(
                n, p, y, 0.9, loss_dtype, thor.losses.LossShape.raw, example_weights=weights
            ),
        ),
        (
            "AsymmetricPowerLoss",
            lambda n, p, y, loss_dtype=None, weights=None: thor.losses.AsymmetricPowerLoss(
                n, p, y, 0.9, 1.5, loss_dtype, thor.losses.LossShape.raw, example_weights=weights
            ),
        ),
    ]


_FLOATING_VALUE_DTYPES = [
    thor.DataType.fp8_e4m3,
    thor.DataType.fp8_e5m2,
    thor.DataType.fp16,
    thor.DataType.bf16,
    thor.DataType.fp32,
]

_LABEL_DTYPES = [
    getattr(thor.DataType, "bool"),
    thor.DataType.int8,
    thor.DataType.int16,
    thor.DataType.int32,
    thor.DataType.int64,
    thor.DataType.uint8,
    thor.DataType.uint16,
    thor.DataType.uint32,
    thor.DataType.uint64,
    *_FLOATING_VALUE_DTYPES,
]


@pytest.mark.parametrize("loss_name,build_loss", _loss_builders())
@pytest.mark.parametrize("predictions_dtype", _FLOATING_VALUE_DTYPES)
def test_regression_losses_accept_all_differentiable_thor_prediction_dtypes(
    loss_name, build_loss, predictions_dtype
):
    n = thor.Network(f"test_{loss_name}_{predictions_dtype}_predictions")
    loss = build_loss(
        n,
        _tensor(predictions_dtype),
        _tensor(thor.DataType.fp32),
        thor.DataType.fp32,
    )
    assert isinstance(loss, thor.losses.Loss)


@pytest.mark.parametrize("loss_name,build_loss", _loss_builders())
@pytest.mark.parametrize("labels_dtype", _LABEL_DTYPES)
def test_regression_losses_accept_expression_convertible_label_dtypes(loss_name, build_loss, labels_dtype):
    n = thor.Network(f"test_{loss_name}_{labels_dtype}_labels")
    loss = build_loss(
        n,
        _tensor(thor.DataType.bf16),
        _tensor(labels_dtype),
        thor.DataType.fp32,
    )
    assert isinstance(loss, thor.losses.Loss)


@pytest.mark.parametrize("loss_name,build_loss", _loss_builders())
@pytest.mark.parametrize("weights_dtype", _FLOATING_VALUE_DTYPES)
def test_regression_losses_accept_all_differentiable_thor_example_weight_dtypes(
    loss_name, build_loss, weights_dtype
):
    n = thor.Network(f"test_{loss_name}_{weights_dtype}_weights")
    loss = build_loss(
        n,
        _tensor(thor.DataType.bf16),
        _tensor(thor.DataType.fp32),
        thor.DataType.fp32,
        _tensor(weights_dtype, 1),
    )
    assert isinstance(loss, thor.losses.Loss)


@pytest.mark.parametrize("loss_name,build_loss", _loss_builders())
@pytest.mark.parametrize("predictions_dtype", [thor.DataType.fp8_e4m3, thor.DataType.fp8_e5m2, thor.DataType.bf16])
def test_regression_losses_default_fp8_and_bf16_loss_storage_to_fp32(
    loss_name, build_loss, predictions_dtype
):
    n = thor.Network(f"test_{loss_name}_{predictions_dtype}_default_loss_dtype")
    loss = build_loss(n, _tensor(predictions_dtype), _tensor(thor.DataType.fp32))
    assert loss.get_loss().get_data_type() == thor.DataType.fp32


@pytest.mark.parametrize("loss_name,build_loss", _loss_builders())
def test_regression_losses_reject_unsupported_value_and_loss_dtypes(loss_name, build_loss):
    n = thor.Network(f"test_{loss_name}_reject_dtypes")

    with pytest.raises(ValueError, match=r"predictions must use fp8_e4m3"):
        build_loss(n, _tensor(thor.DataType.int32), _tensor(thor.DataType.fp32), thor.DataType.fp32)

    with pytest.raises(ValueError, match=r"predictions must use fp8_e4m3"):
        build_loss(n, _tensor(thor.DataType.fp64), _tensor(thor.DataType.fp32), thor.DataType.fp32)

    with pytest.raises(ValueError, match=r"labels must use a Thor integer dtype"):
        build_loss(n, _tensor(thor.DataType.fp32), _tensor(thor.DataType.fp64), thor.DataType.fp32)

    with pytest.raises(ValueError, match=r"loss_data_type must be fp16 or fp32"):
        build_loss(n, _tensor(thor.DataType.fp32), _tensor(thor.DataType.fp32), thor.DataType.bf16)

    with pytest.raises(ValueError, match=r"example_weights must use fp8_e4m3"):
        build_loss(
            n,
            _tensor(thor.DataType.fp32),
            _tensor(thor.DataType.fp32),
            thor.DataType.fp32,
            _tensor(thor.DataType.int32, 1),
        )
