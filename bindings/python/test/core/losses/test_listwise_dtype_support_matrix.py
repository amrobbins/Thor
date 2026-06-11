import pytest
import thor


def _tensor_1d(size: int, dtype: thor.DataType):
    return thor.Tensor([size], dtype)


def _listwise_loss_builders():
    return [
        (
            "ListwiseSoftmaxCrossEntropyLoss",
            lambda n, predictions, labels, loss_dtype, mask=None: thor.losses.ranking.ListwiseSoftmaxCrossEntropyLoss(
                n,
                predictions,
                labels,
                0.75,
                loss_dtype,
                thor.losses.LossShape.raw,
                mask=mask,
            ),
        ),
        (
            "ListNetLoss",
            lambda n, predictions, labels, loss_dtype, mask=None: thor.losses.ranking.ListNetLoss(
                n,
                predictions,
                labels,
                0.8,
                0.6,
                loss_dtype,
                thor.losses.LossShape.raw,
                mask=mask,
            ),
        ),
    ]


@pytest.mark.parametrize("loss_name, build_loss", _listwise_loss_builders())
@pytest.mark.parametrize("predictions_dtype", [thor.DataType.fp16, thor.DataType.fp32])
@pytest.mark.parametrize("labels_dtype", [thor.DataType.fp16, thor.DataType.fp32])
@pytest.mark.parametrize("loss_dtype", [thor.DataType.fp16, thor.DataType.fp32])
def test_listwise_losses_accept_fp16_and_fp32_value_and_loss_tensors(
    loss_name, build_loss, predictions_dtype, labels_dtype, loss_dtype
):
    n = thor.Network(f"test_net_{loss_name}_value_dtype_matrix")
    predictions = _tensor_1d(4, predictions_dtype)
    labels = _tensor_1d(4, labels_dtype)

    loss = build_loss(n, predictions, labels, loss_dtype)

    assert isinstance(loss, thor.losses.Loss)


@pytest.mark.parametrize("loss_name, build_loss", _listwise_loss_builders())
@pytest.mark.parametrize(
    "mask_dtype",
    [getattr(thor.DataType, "bool"), thor.DataType.uint8, thor.DataType.fp16, thor.DataType.fp32],
)
def test_listwise_losses_accept_bool_uint8_fp16_and_fp32_masks(loss_name, build_loss, mask_dtype):
    n = thor.Network(f"test_net_{loss_name}_mask_dtype_matrix")
    predictions = _tensor_1d(4, thor.DataType.fp32)
    labels = _tensor_1d(4, thor.DataType.fp32)
    mask = _tensor_1d(4, mask_dtype)

    loss = build_loss(n, predictions, labels, thor.DataType.fp32, mask=mask)

    assert isinstance(loss, thor.losses.Loss)


@pytest.mark.parametrize("loss_name, build_loss", _listwise_loss_builders())
@pytest.mark.parametrize(
    "unsupported_dtype",
    [getattr(thor.DataType, "bool"), thor.DataType.uint8, thor.DataType.uint16, thor.DataType.int32, thor.DataType.fp64, thor.DataType.bf16],
)
def test_listwise_losses_reject_unsupported_prediction_label_and_loss_dtypes(loss_name, build_loss, unsupported_dtype):
    n = thor.Network(f"test_net_{loss_name}_reject_value_dtype_matrix")
    predictions = _tensor_1d(4, thor.DataType.fp32)
    labels = _tensor_1d(4, thor.DataType.fp32)

    with pytest.raises(ValueError, match=r"predictions must use fp16 or fp32 dtype"):
        build_loss(n, _tensor_1d(4, unsupported_dtype), labels, thor.DataType.fp32)

    with pytest.raises(ValueError, match=r"labels must use fp16 or fp32 dtype"):
        build_loss(n, predictions, _tensor_1d(4, unsupported_dtype), thor.DataType.fp32)

    with pytest.raises(ValueError, match=r"loss_data_type must be fp16 or fp32"):
        build_loss(n, predictions, labels, unsupported_dtype)


@pytest.mark.parametrize("loss_name, build_loss", _listwise_loss_builders())
@pytest.mark.parametrize(
    "unsupported_mask_dtype",
    [thor.DataType.uint16, thor.DataType.int32, thor.DataType.fp64, thor.DataType.bf16],
)
def test_listwise_losses_reject_unsupported_mask_dtypes(loss_name, build_loss, unsupported_mask_dtype):
    n = thor.Network(f"test_net_{loss_name}_reject_mask_dtype_matrix")
    predictions = _tensor_1d(4, thor.DataType.fp32)
    labels = _tensor_1d(4, thor.DataType.fp32)
    mask = _tensor_1d(4, unsupported_mask_dtype)

    with pytest.raises(ValueError, match=r"mask must use bool, uint8, fp16, or fp32 dtype"):
        build_loss(n, predictions, labels, thor.DataType.fp32, mask=mask)
