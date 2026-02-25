# test/test_losses_categorical_cross_entropy.py
import pytest
import thor


def _net():
    return thor.Network("test_net_categorical_cross_entropy")


def _tensor_1d(size: int, dtype=thor.DataType.fp32):
    return thor.Tensor([size], dtype)


def test_categorical_cross_entropy_one_hot_constructs_defaults():
    n = _net()
    preds = _tensor_1d(5, thor.DataType.fp32)
    labels = _tensor_1d(5, thor.DataType.fp32)

    # defaults: loss_data_type=fp32, reported_loss_shape=batch
    loss = thor.losses.CategoricalCrossEntropy(
        n,
        preds,
        labels,
        thor.losses.LabelType.one_hot,
        None,
    )
    assert loss is not None
    assert isinstance(loss, thor.losses.CategoricalCrossEntropy)


def test_categorical_cross_entropy_one_hot_constructs_with_num_classes():
    n = _net()
    preds = _tensor_1d(7)
    labels = _tensor_1d(7)

    loss = thor.losses.CategoricalCrossEntropy(
        n,
        preds,
        labels,
        thor.losses.LabelType.one_hot,
        7,
    )
    assert isinstance(loss, thor.losses.CategoricalCrossEntropy)


def test_categorical_cross_entropy_one_hot_rejects_predictions_not_1d():
    n = _net()
    preds = thor.Tensor([2, 3], thor.DataType.fp32)
    labels = _tensor_1d(6)

    with pytest.raises(ValueError, match=r"one_hot predictions must have 1 dimension"):
        thor.losses.CategoricalCrossEntropy(n, preds, labels, thor.losses.LabelType.one_hot, None)


def test_categorical_cross_entropy_one_hot_rejects_labels_not_1d():
    n = _net()
    preds = _tensor_1d(6)
    labels = thor.Tensor([2, 3], thor.DataType.fp32)

    with pytest.raises(ValueError, match=r"one_hot labels must have 1 dimension"):
        thor.losses.CategoricalCrossEntropy(n, preds, labels, thor.losses.LabelType.one_hot, None)


def test_categorical_cross_entropy_one_hot_rejects_mismatched_sizes():
    n = _net()
    preds = _tensor_1d(5)
    labels = _tensor_1d(6)

    with pytest.raises(ValueError, match=r"mismatch between predictions size 5 and labels tensor size 6"):
        thor.losses.CategoricalCrossEntropy(n, preds, labels, thor.losses.LabelType.one_hot, None)


def test_categorical_cross_entropy_one_hot_rejects_num_classes_mismatch():
    n = _net()
    preds = _tensor_1d(5)
    labels = _tensor_1d(5)

    with pytest.raises(ValueError, match=r"mismatch between num_classes 6 and predictions tensor size 5"):
        thor.losses.CategoricalCrossEntropy(n, preds, labels, thor.losses.LabelType.one_hot, 6)


def test_categorical_cross_entropy_one_hot_rejects_num_classes_non_positive():
    n = _net()
    preds = _tensor_1d(5)
    labels = _tensor_1d(5)

    # Your ONE_HOT check errors when num_classes is provided and <= 0
    with pytest.raises(ValueError, match=r"mismatch between num_classes 0 and predictions tensor size 5"):
        thor.losses.CategoricalCrossEntropy(n, preds, labels, thor.losses.LabelType.one_hot, 0)


def test_categorical_cross_entropy_index_constructs_uint16_labels():
    n = _net()
    preds = _tensor_1d(10, thor.DataType.fp32)
    labels = _tensor_1d(1, thor.DataType.uint16)  # IMPORTANT: integer dtype for index labels

    loss = thor.losses.CategoricalCrossEntropy(
        n,
        preds,
        labels,
        thor.losses.LabelType.index,
        10,
    )
    assert isinstance(loss, thor.losses.CategoricalCrossEntropy)


def test_categorical_cross_entropy_index_requires_num_classes():
    n = _net()
    preds = _tensor_1d(10)
    labels = _tensor_1d(1, thor.DataType.uint16)

    with pytest.raises(ValueError, match=r"label_type set to LabelType\.index but num_classes is None"):
        thor.losses.CategoricalCrossEntropy(n, preds, labels, thor.losses.LabelType.index, None)


def test_categorical_cross_entropy_index_rejects_num_classes_non_positive():
    n = _net()
    preds = _tensor_1d(10)
    labels = _tensor_1d(1, thor.DataType.uint16)

    with pytest.raises(ValueError, match=r"num_classes must be a positive integer"):
        thor.losses.CategoricalCrossEntropy(n, preds, labels, thor.losses.LabelType.index, 0)


def test_categorical_cross_entropy_index_rejects_labels_not_size_1():
    n = _net()
    preds = _tensor_1d(10)

    labels = _tensor_1d(2, thor.DataType.uint16)
    with pytest.raises(ValueError, match=r"labels must be a 1 dimensional tensor of size 1"):
        thor.losses.CategoricalCrossEntropy(n, preds, labels, thor.losses.LabelType.index, 10)

    labels = thor.Tensor([1, 1], thor.DataType.uint16)
    with pytest.raises(ValueError, match=r"labels must be a 1 dimensional tensor of size 1"):
        thor.losses.CategoricalCrossEntropy(n, preds, labels, thor.losses.LabelType.index, 10)


@pytest.mark.parametrize(
    "shape",
    [
        "batch",
        "classwise",
        "elementwise",
        "raw",
    ],
)
def test_categorical_cross_entropy_reported_loss_shape_variants_construct(shape):
    n = _net()
    preds = _tensor_1d(5, thor.DataType.fp32)
    labels = _tensor_1d(5, thor.DataType.fp32)

    loss_shape = getattr(thor.losses.LossShape, shape)

    loss = thor.losses.CategoricalCrossEntropy(
        n,
        preds,
        labels,
        thor.losses.LabelType.one_hot,
        None,
        thor.DataType.fp32,
        loss_shape,
    )
    assert isinstance(loss, thor.losses.CategoricalCrossEntropy)


def test_categorical_cross_entropy_rejects_invalid_reported_loss_shape():
    n = _net()
    preds = _tensor_1d(5)
    labels = _tensor_1d(5)

    # Force an invalid enum value by casting an int into the enum type.
    # Depending on your binding, this may raise TypeError earlier; if so, that's fine too.
    try:
        bogus = thor.losses.LossShape(123456)
    except Exception:
        bogus = 123456  # fallback; should TypeError at call site

    with pytest.raises((ValueError, TypeError), match=r"(Invalid value|reported_loss_shape)"):
        thor.losses.CategoricalCrossEntropy(
            n,
            preds,
            labels,
            thor.losses.LabelType.one_hot,
            None,
            thor.DataType.fp32,
            bogus,
        )


def test_categorical_cross_entropy_rejects_wrong_types():
    n = _net()
    preds = _tensor_1d(5)
    labels = _tensor_1d(5)

    with pytest.raises(TypeError):
        thor.losses.CategoricalCrossEntropy("not a network", preds, labels, thor.losses.LabelType.one_hot, None)

    with pytest.raises(TypeError):
        thor.losses.CategoricalCrossEntropy(n, "not a tensor", labels, thor.losses.LabelType.one_hot, None)

    with pytest.raises(TypeError):
        thor.losses.CategoricalCrossEntropy(n, preds, "not a tensor", thor.losses.LabelType.one_hot, None)


def test_categorical_cross_entropy_rejects_wrong_arity():
    n = _net()
    preds = _tensor_1d(5)
    labels = _tensor_1d(5)

    with pytest.raises(TypeError):
        thor.losses.CategoricalCrossEntropy(n, preds, labels)  # missing required args

    with pytest.raises(TypeError):
        thor.losses.CategoricalCrossEntropy(
            n, preds, labels, thor.losses.LabelType.one_hot, None, thor.DataType.fp32, thor.losses.LossShape.batch, 123)
