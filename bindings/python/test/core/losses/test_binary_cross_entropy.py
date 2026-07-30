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
