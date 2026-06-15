# test/test_losses_categorical_cross_entropy.py
import pytest
import thor


def _net():
    return thor.Network("test_net_categorical_cross_entropy")


def _tensor_1d(size: int, dtype=thor.DataType.fp32):
    return thor.Tensor([size], dtype)


def test_categorical_cross_entropy_dense_constructs_defaults():
    n = _net()
    preds = _tensor_1d(5, thor.DataType.fp32)
    labels = _tensor_1d(5, thor.DataType.fp32)

    loss = thor.losses.CategoricalCrossEntropy(n, preds, labels)
    assert loss is not None
    assert isinstance(loss, thor.losses.CategoricalCrossEntropy)
    assert not isinstance(loss, thor.losses.SparseCategoricalCrossEntropy)


def test_categorical_cross_entropy_dense_accepts_sequence_prefix_dimensions():
    n = _net()
    preds = thor.Tensor([4, 5], thor.DataType.fp32)
    labels = thor.Tensor([4, 5], thor.DataType.fp32)

    loss = thor.losses.CategoricalCrossEntropy(n, preds, labels)
    assert isinstance(loss, thor.losses.CategoricalCrossEntropy)


def test_categorical_cross_entropy_dense_accepts_multi_prefix_dimensions():
    n = _net()
    preds = thor.Tensor([3, 7, 11], thor.DataType.fp32)
    labels = thor.Tensor([3, 7, 11], thor.DataType.fp32)

    loss = thor.losses.CategoricalCrossEntropy(n, preds, labels)
    assert isinstance(loss, thor.losses.CategoricalCrossEntropy)


def test_categorical_cross_entropy_dense_rejects_label_dimensions_that_do_not_match_predictions():
    n = _net()
    preds = _tensor_1d(6)
    labels = thor.Tensor([2, 3], thor.DataType.fp32)

    with pytest.raises(ValueError, match=r"dense labels dimensions"):
        thor.losses.CategoricalCrossEntropy(n, preds, labels)


def test_categorical_cross_entropy_dense_rejects_mismatched_sizes():
    n = _net()
    preds = _tensor_1d(5)
    labels = _tensor_1d(6)

    with pytest.raises(ValueError, match=r"dense labels dimensions"):
        thor.losses.CategoricalCrossEntropy(n, preds, labels)


def test_sparse_categorical_cross_entropy_constructs_uint16_labels():
    n = _net()
    preds = _tensor_1d(10, thor.DataType.fp32)
    labels = _tensor_1d(1, thor.DataType.uint16)

    loss = thor.losses.SparseCategoricalCrossEntropy(n, preds, labels, 10)
    assert isinstance(loss, thor.losses.SparseCategoricalCrossEntropy)
    assert isinstance(loss, thor.losses.CategoricalCrossEntropy)


def test_sparse_categorical_cross_entropy_accepts_per_token_labels():
    n = _net()
    preds = thor.Tensor([7, 10], thor.DataType.fp32)
    labels = thor.Tensor([7], thor.DataType.uint32)

    loss = thor.losses.SparseCategoricalCrossEntropy(n, preds, labels, 10)
    assert isinstance(loss, thor.losses.SparseCategoricalCrossEntropy)


def test_sparse_categorical_cross_entropy_accepts_per_token_labels_with_trailing_singleton():
    n = _net()
    preds = thor.Tensor([7, 10], thor.DataType.fp32)
    labels = thor.Tensor([7, 1], thor.DataType.uint32)

    loss = thor.losses.SparseCategoricalCrossEntropy(n, preds, labels, 10)
    assert isinstance(loss, thor.losses.SparseCategoricalCrossEntropy)


def test_sparse_categorical_cross_entropy_accepts_multi_prefix_lm_labels():
    n = _net()
    preds = thor.Tensor([3, 7, 257], thor.DataType.fp32)
    labels = thor.Tensor([3, 7], thor.DataType.uint32)

    loss = thor.losses.SparseCategoricalCrossEntropy(n, preds, labels, 257)
    assert isinstance(loss, thor.losses.SparseCategoricalCrossEntropy)


def test_sparse_categorical_cross_entropy_accepts_multi_prefix_lm_labels_with_trailing_singleton():
    n = _net()
    preds = thor.Tensor([3, 7, 257], thor.DataType.fp32)
    labels = thor.Tensor([3, 7, 1], thor.DataType.uint32)

    loss = thor.losses.SparseCategoricalCrossEntropy(n, preds, labels, 257)
    assert isinstance(loss, thor.losses.SparseCategoricalCrossEntropy)


def test_sparse_categorical_cross_entropy_rejects_incomplete_multi_prefix_labels():
    n = _net()
    preds = thor.Tensor([3, 7, 257], thor.DataType.fp32)
    labels = thor.Tensor([3], thor.DataType.uint32)

    with pytest.raises(ValueError, match=r"sparse labels dimensions"):
        thor.losses.SparseCategoricalCrossEntropy(n, preds, labels, 257)


def test_sparse_categorical_cross_entropy_rejects_num_classes_non_positive():
    n = _net()
    preds = _tensor_1d(10)
    labels = _tensor_1d(1, thor.DataType.uint16)

    with pytest.raises(ValueError, match=r"num_classes must be greater than one"):
        thor.losses.SparseCategoricalCrossEntropy(n, preds, labels, 0)


def test_sparse_categorical_cross_entropy_rejects_num_classes_mismatch():
    n = _net()
    preds = _tensor_1d(10)
    labels = _tensor_1d(1, thor.DataType.uint16)

    with pytest.raises(ValueError, match=r"mismatch between num_classes 11 and predictions final class dimension 10"):
        thor.losses.SparseCategoricalCrossEntropy(n, preds, labels, 11)


def test_sparse_categorical_cross_entropy_rejects_labels_not_size_1():
    n = _net()
    preds = _tensor_1d(10)

    labels = _tensor_1d(2, thor.DataType.uint16)
    with pytest.raises(ValueError, match=r"sparse labels dimensions"):
        thor.losses.SparseCategoricalCrossEntropy(n, preds, labels, 10)

    labels = thor.Tensor([1, 1], thor.DataType.uint16)
    with pytest.raises(ValueError, match=r"sparse labels dimensions"):
        thor.losses.SparseCategoricalCrossEntropy(n, preds, labels, 10)


def test_sparse_categorical_cross_entropy_rejects_float_labels():
    n = _net()
    preds = _tensor_1d(10)
    labels = _tensor_1d(1, thor.DataType.fp32)

    with pytest.raises(ValueError, match=r"labels must use uint8, uint16, or uint32 dtype"):
        thor.losses.SparseCategoricalCrossEntropy(n, preds, labels, 10)


@pytest.mark.parametrize(
    "loss_cls,labels,args",
    [
        (thor.losses.CategoricalCrossEntropy, _tensor_1d(5, thor.DataType.fp32), ()),
        (thor.losses.SparseCategoricalCrossEntropy, _tensor_1d(1, thor.DataType.uint16), (5,)),
    ],
)
def test_categorical_cross_entropy_reported_loss_shape_variants_construct(loss_cls, labels, args):
    for shape in ["batch", "classwise", "elementwise", "raw"]:
        n = _net()
        preds = _tensor_1d(5, thor.DataType.fp32)
        loss_shape = getattr(thor.losses.LossShape, shape)

        loss = loss_cls(n, preds, labels, *args, thor.DataType.fp32, loss_shape)
        assert isinstance(loss, thor.losses.CategoricalCrossEntropy)


def test_categorical_cross_entropy_rejects_invalid_reported_loss_shape():
    n = _net()
    preds = _tensor_1d(5)
    labels = _tensor_1d(5)

    try:
        bogus = thor.losses.LossShape(123456)
    except Exception:
        bogus = 123456

    with pytest.raises((ValueError, TypeError), match=r"(Invalid value|reported_loss_shape)"):
        thor.losses.CategoricalCrossEntropy(n, preds, labels, thor.DataType.fp32, bogus)


def test_sparse_categorical_cross_entropy_rejects_invalid_reported_loss_shape():
    n = _net()
    preds = _tensor_1d(5)
    labels = _tensor_1d(1, thor.DataType.uint16)

    try:
        bogus = thor.losses.LossShape(123456)
    except Exception:
        bogus = 123456

    with pytest.raises((ValueError, TypeError), match=r"(Invalid value|reported_loss_shape)"):
        thor.losses.SparseCategoricalCrossEntropy(n, preds, labels, 5, thor.DataType.fp32, bogus)


def test_categorical_cross_entropy_rejects_wrong_types():
    n = _net()
    preds = _tensor_1d(5)
    labels = _tensor_1d(5)

    with pytest.raises(TypeError):
        thor.losses.CategoricalCrossEntropy("not a network", preds, labels)

    with pytest.raises(TypeError):
        thor.losses.CategoricalCrossEntropy(n, "not a tensor", labels)

    with pytest.raises(TypeError):
        thor.losses.CategoricalCrossEntropy(n, preds, "not a tensor")


def test_sparse_categorical_cross_entropy_rejects_wrong_types():
    n = _net()
    preds = _tensor_1d(5)
    labels = _tensor_1d(1, thor.DataType.uint16)

    with pytest.raises(TypeError):
        thor.losses.SparseCategoricalCrossEntropy("not a network", preds, labels, 5)

    with pytest.raises(TypeError):
        thor.losses.SparseCategoricalCrossEntropy(n, "not a tensor", labels, 5)

    with pytest.raises(TypeError):
        thor.losses.SparseCategoricalCrossEntropy(n, preds, "not a tensor", 5)


def test_categorical_cross_entropy_rejects_wrong_arity():
    n = _net()
    preds = _tensor_1d(5)
    labels = _tensor_1d(5)

    with pytest.raises(TypeError):
        thor.losses.CategoricalCrossEntropy(n, preds, labels, thor.DataType.fp32, thor.losses.LossShape.batch, 123)

    sparse_labels = _tensor_1d(1, thor.DataType.uint16)
    with pytest.raises(TypeError):
        thor.losses.SparseCategoricalCrossEntropy(n, preds, sparse_labels)  # missing num_classes

    with pytest.raises(TypeError):
        thor.losses.SparseCategoricalCrossEntropy(n, preds, sparse_labels, 5, thor.DataType.fp32, thor.losses.LossShape.batch, 123)
