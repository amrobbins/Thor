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


def test_binary_cross_entropy_constructs_reports_elementwise():
    n = _net()
    preds = _tensor_1d(1, thor.DataType.fp32)
    labels = _tensor_1d(1, thor.DataType.fp32)

    loss = thor.losses.BinaryCrossEntropy(
        n,
        preds,
        labels,
        thor.DataType.fp32,
        True,
    )
    assert isinstance(loss, thor.losses.BinaryCrossEntropy)


def test_binary_cross_entropy_rejects_predictions_not_1d_size_1():
    n = _net()
    labels = _tensor_1d(1, thor.DataType.fp32)

    preds = _tensor_1d(2, thor.DataType.fp32)
    with pytest.raises(ValueError, match=r"predictions must be a 1 dimensional tensor of size one"):
        thor.losses.BinaryCrossEntropy(n, preds, labels)

    preds = thor.Tensor([1, 1], thor.DataType.fp32)
    with pytest.raises(ValueError, match=r"predictions must be a 1 dimensional tensor of size one"):
        thor.losses.BinaryCrossEntropy(n, preds, labels)


def test_binary_cross_entropy_rejects_labels_not_1d_size_1():
    n = _net()
    preds = _tensor_1d(1, thor.DataType.fp32)

    labels = _tensor_1d(2, thor.DataType.fp32)
    with pytest.raises(ValueError, match=r"labels must be a 1 dimensional tensor of size one"):
        thor.losses.BinaryCrossEntropy(n, preds, labels)

    labels = thor.Tensor([1, 1], thor.DataType.fp32)
    with pytest.raises(ValueError, match=r"labels must be a 1 dimensional tensor of size one"):
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
        thor.losses.BinaryCrossEntropy(n, preds, labels, thor.DataType.fp32, False, 123)  # extra arg
