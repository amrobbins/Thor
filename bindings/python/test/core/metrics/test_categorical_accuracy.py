# test/test_metrics_categorical_accuracy.py
import pytest
import thor
from thor.metrics import CategoricalAccuracy


def _net():
    return thor.Network("test_net_categorical_accuracy")


def _tensor_1d(size: int, dtype=thor.DataType.fp32):
    # API tensor: just dims + dtype
    return thor.Tensor([size], dtype)


def test_categorical_accuracy_one_hot_constructs():
    n = _net()
    # one_hot: both must be 1D and same length
    preds = _tensor_1d(5)
    labels = _tensor_1d(5)

    m = thor.metrics.CategoricalAccuracy(
        n,
        preds,
        labels,
        thor.losses.LabelType.one_hot,   # adjust enum names if yours differ
        None,
    )
    assert m is not None
    assert isinstance(m, CategoricalAccuracy)


def test_categorical_accuracy_one_hot_with_num_classes_constructs():
    n = _net()
    preds = _tensor_1d(7)
    labels = _tensor_1d(7)

    m = thor.metrics.CategoricalAccuracy(
        n,
        preds,
        labels,
        thor.losses.LabelType.one_hot,
        7,
    )
    assert isinstance(m, thor.metrics.CategoricalAccuracy)


def test_categorical_accuracy_one_hot_rejects_predictions_not_1d():
    n = _net()
    preds = thor.Tensor([2, 3], thor.DataType.fp32)  # 2D => should error
    labels = _tensor_1d(6)

    with pytest.raises(ValueError, match=r"one_hot predictions must have 1 dimension"):
        thor.metrics.CategoricalAccuracy(
            n,
            preds,
            labels,
            thor.losses.LabelType.one_hot,
            None,
        )


def test_categorical_accuracy_one_hot_rejects_labels_not_1d():
    n = _net()
    preds = _tensor_1d(6)
    labels = thor.Tensor([2, 3], thor.DataType.fp32)  # 2D => should error

    with pytest.raises(ValueError, match=r"one_hot labels must have 1 dimension"):
        thor.metrics.CategoricalAccuracy(
            n,
            preds,
            labels,
            thor.losses.LabelType.one_hot,
            None,
        )


def test_categorical_accuracy_one_hot_rejects_mismatched_sizes():
    n = _net()
    preds = _tensor_1d(5)
    labels = _tensor_1d(6)

    with pytest.raises(ValueError, match=r"mismatch between predictions size 5 and labels tensor size 6"):
        thor.metrics.CategoricalAccuracy(
            n,
            preds,
            labels,
            thor.losses.LabelType.one_hot,
            None,
        )


def test_categorical_accuracy_one_hot_rejects_num_classes_mismatch():
    n = _net()
    preds = _tensor_1d(5)
    labels = _tensor_1d(5)

    with pytest.raises(ValueError, match=r"mismatch between num_classes 6 and predictions tensor size 5"):
        thor.metrics.CategoricalAccuracy(
            n,
            preds,
            labels,
            thor.losses.LabelType.one_hot,
            6,
        )


def test_categorical_accuracy_one_hot_rejects_num_classes_non_positive():
    n = _net()
    preds = _tensor_1d(5)
    labels = _tensor_1d(5)

    # Your code treats <= 0 as error when num_classes is provided in ONE_HOT path too.
    with pytest.raises(ValueError, match=r"mismatch between num_classes 0 and predictions tensor size 5"):
        thor.metrics.CategoricalAccuracy(
            n,
            preds,
            labels,
            thor.losses.LabelType.one_hot,
            0,
        )


def test_categorical_accuracy_index_constructs():
    n = _net()
    preds = _tensor_1d(10, thor.DataType.uint8)  # 10 classes
    labels = _tensor_1d(1, thor.DataType.uint8)  # index label: must be 1D size 1

    m = thor.metrics.CategoricalAccuracy(
        n,
        preds,
        labels,
        thor.losses.LabelType.index,
        10,
    )
    assert isinstance(m, thor.metrics.CategoricalAccuracy)


def test_categorical_accuracy_index_requires_num_classes():
    n = _net()
    preds = _tensor_1d(10)
    labels = _tensor_1d(1)

    with pytest.raises(ValueError, match=r"label_type set to LabelType\.index but num_classes is None"):
        thor.metrics.CategoricalAccuracy(
            n,
            preds,
            labels,
            thor.losses.LabelType.index,
            None,
        )


def test_categorical_accuracy_index_rejects_num_classes_non_positive():
    n = _net()
    preds = _tensor_1d(10)
    labels = _tensor_1d(1)

    with pytest.raises(ValueError, match=r"num_classes must be a positive integer"):
        thor.metrics.CategoricalAccuracy(
            n,
            preds,
            labels,
            thor.losses.LabelType.index,
            0,
        )


def test_categorical_accuracy_index_rejects_labels_not_size_1():
    n = _net()
    preds = _tensor_1d(10)

    labels = _tensor_1d(2)  # wrong: must be [1]
    with pytest.raises(ValueError, match=r"labels must be a 1 dimensional tensor of size 1"):
        thor.metrics.CategoricalAccuracy(
            n,
            preds,
            labels,
            thor.losses.LabelType.index,
            10,
        )

    labels = thor.Tensor([1, 1], thor.DataType.fp32)  # wrong: not 1D
    with pytest.raises(ValueError, match=r"labels must be a 1 dimensional tensor of size 1"):
        thor.metrics.CategoricalAccuracy(
            n,
            preds,
            labels,
            thor.losses.LabelType.index,
            10,
        )


def test_categorical_accuracy_rejects_wrong_types():
    n = _net()
    preds = _tensor_1d(5)
    labels = _tensor_1d(5)

    with pytest.raises(TypeError):
        thor.metrics.CategoricalAccuracy("not a network", preds, labels, thor.losses.LabelType.one_hot, None)

    with pytest.raises(TypeError):
        thor.metrics.CategoricalAccuracy(n, "not a tensor", labels, thor.losses.LabelType.one_hot, None)

    with pytest.raises(TypeError):
        thor.metrics.CategoricalAccuracy(n, preds, "not a tensor", thor.losses.LabelType.one_hot, None)


def test_categorical_accuracy_rejects_wrong_arity():
    n = _net()
    preds = _tensor_1d(5)
    labels = _tensor_1d(5)

    with pytest.raises(TypeError):
        thor.metrics.CategoricalAccuracy(n, preds, labels)  # missing label_type

    with pytest.raises(TypeError):
        thor.metrics.CategoricalAccuracy(n, preds, labels, thor.losses.LabelType.one_hot, None, 123)  # extra
