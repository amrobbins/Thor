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
    for shape in ["batch", "per_output", "per_example", "raw"]:
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


def test_sparse_categorical_cross_entropy_raw_loss_is_prediction_prefix_shape():
    n = _net()
    preds = thor.Tensor([3, 7, 257], thor.DataType.fp32)
    labels = thor.Tensor([3, 7], thor.DataType.uint32)

    loss = thor.losses.SparseCategoricalCrossEntropy(
        n,
        preds,
        labels,
        257,
        thor.DataType.fp32,
        thor.losses.LossShape.raw,
    )

    assert loss.get_loss().get_dimensions() == [3, 7]


def test_sparse_categorical_cross_entropy_accepts_ignore_index_and_mask():
    n = _net()
    preds = thor.Tensor([3, 7, 257], thor.DataType.fp32)
    labels = thor.Tensor([3, 7], thor.DataType.uint32)
    mask = thor.Tensor([3, 7], thor.DataType.uint8)

    loss = thor.losses.SparseCategoricalCrossEntropy(
        n,
        preds,
        labels,
        257,
        ignore_index=0,
        mask=mask,
    )

    assert isinstance(loss, thor.losses.SparseCategoricalCrossEntropy)


@pytest.mark.parametrize("mask_dtype", [thor.DataType.uint8, thor.DataType.bool, thor.DataType.fp16, thor.DataType.fp32])
def test_sparse_categorical_cross_entropy_accepts_mask_dtypes(mask_dtype):
    n = _net()
    preds = thor.Tensor([5, 11], thor.DataType.fp32)
    labels = thor.Tensor([5], thor.DataType.uint32)
    mask = thor.Tensor([5], mask_dtype)

    loss = thor.losses.SparseCategoricalCrossEntropy(n, preds, labels, 11, mask=mask)
    assert isinstance(loss, thor.losses.SparseCategoricalCrossEntropy)


def test_sparse_categorical_cross_entropy_rejects_bad_mask_shape():
    n = _net()
    preds = thor.Tensor([3, 7, 257], thor.DataType.fp32)
    labels = thor.Tensor([3, 7], thor.DataType.uint32)
    mask = thor.Tensor([3], thor.DataType.uint8)

    with pytest.raises(ValueError, match=r"mask dimensions"):
        thor.losses.SparseCategoricalCrossEntropy(n, preds, labels, 257, mask=mask)


def test_sparse_categorical_cross_entropy_rejects_bad_mask_dtype():
    n = _net()
    preds = thor.Tensor([3, 7, 257], thor.DataType.fp32)
    labels = thor.Tensor([3, 7], thor.DataType.uint32)
    mask = thor.Tensor([3, 7], thor.DataType.uint32)

    with pytest.raises(ValueError, match=r"mask must use"):
        thor.losses.SparseCategoricalCrossEntropy(n, preds, labels, 257, mask=mask)


def test_sparse_categorical_cross_entropy_rejects_negative_ignore_index():
    n = _net()
    preds = thor.Tensor([7, 10], thor.DataType.fp32)
    labels = thor.Tensor([7], thor.DataType.uint32)

    with pytest.raises(ValueError, match=r"ignore_index"):
        thor.losses.SparseCategoricalCrossEntropy(n, preds, labels, 10, ignore_index=-1)


def _r11c_categorical_ce_ragged_pair(network, *, offsets_dtype=thor.DataType.uint32, prefix="r11c_ce"):
    predictions = thor.layers.RaggedNetworkInput(
        network,
        f"{prefix}_predictions",
        thor.DataType.fp32,
        [3],
        batch_size=3,
        max_total_values=8,
        max_values_per_row=4,
        offsets_data_type=offsets_dtype,
    )
    labels = thor.layers.RaggedNetworkInput(
        network,
        f"{prefix}_labels",
        thor.DataType.fp32,
        [3],
        partition=predictions,
    )
    return predictions, labels


@pytest.mark.parametrize("offsets_dtype", [thor.DataType.uint32, thor.DataType.uint64])
def test_categorical_cross_entropy_r11c_constructs_ragged_dense_targets_and_preserves_partition(offsets_dtype):
    n = thor.Network(f"test_categorical_ce_r11c_{offsets_dtype}")
    predictions, labels = _r11c_categorical_ce_ragged_pair(n, offsets_dtype=offsets_dtype)
    loss = thor.losses.CategoricalCrossEntropy(
        n,
        predictions,
        labels,
        reported_loss_shape=thor.losses.LossShape.raw,
    )
    assert loss.is_ragged
    assert loss.get_predictions() == predictions
    assert loss.get_labels() == labels
    assert isinstance(loss.get_raw_loss(), thor.RaggedTensor)
    assert isinstance(loss.get_loss(), thor.RaggedTensor)
    assert loss.get_loss().offsets == predictions.offsets
    assert loss.get_loss().values.get_dimensions() == [8, 3]


def test_categorical_cross_entropy_r11c_rejects_per_output_mixed_inputs_and_different_partition():
    n = thor.Network("test_categorical_ce_r11c_reject")
    predictions, labels = _r11c_categorical_ce_ragged_pair(n)
    with pytest.raises(ValueError, match=r"per_output.*undefined"):
        thor.losses.CategoricalCrossEntropy(
            n, predictions, labels, reported_loss_shape=thor.losses.LossShape.per_output
        )

    different = thor.layers.RaggedNetworkInput(
        n,
        "r11c_ce_different",
        thor.DataType.fp32,
        [3],
        batch_size=3,
        max_total_values=8,
        max_values_per_row=4,
    )
    with pytest.raises(ValueError, match=r"exact same row partition"):
        thor.losses.CategoricalCrossEntropy(n, predictions, different)

    with pytest.raises(TypeError, match=r"both be thor.Tensor or both be thor.RaggedTensor"):
        thor.losses.CategoricalCrossEntropy(n, predictions, _tensor_1d(3))


def _r11c1_sparse_ragged_inputs(
    network,
    *,
    offsets_dtype=thor.DataType.uint32,
    label_dtype=thor.DataType.uint32,
    mask_dtype=thor.DataType.uint8,
    prefix="r11c1_sparse",
):
    predictions = thor.layers.RaggedNetworkInput(
        network,
        f"{prefix}_predictions",
        thor.DataType.fp32,
        [3],
        batch_size=3,
        max_total_values=8,
        max_values_per_row=5,
        offsets_data_type=offsets_dtype,
    )
    labels = thor.layers.RaggedNetworkInput(
        network,
        f"{prefix}_labels",
        label_dtype,
        [],
        partition=predictions,
    )
    mask = thor.layers.RaggedNetworkInput(
        network,
        f"{prefix}_mask",
        mask_dtype,
        [],
        partition=predictions,
    )
    return predictions, labels, mask


@pytest.mark.parametrize("label_dtype", [thor.DataType.uint8, thor.DataType.uint16, thor.DataType.uint32])
def test_sparse_categorical_cross_entropy_r11c1_accepts_all_sparse_label_dtypes(label_dtype):
    n = thor.Network(f"test_sparse_categorical_ce_r11c1_label_{label_dtype}")
    predictions, labels, mask = _r11c1_sparse_ragged_inputs(n, label_dtype=label_dtype)
    loss = thor.losses.SparseCategoricalCrossEntropy(n, predictions, labels, 3, mask=mask)
    assert isinstance(loss, thor.losses.SparseCategoricalCrossEntropy)


@pytest.mark.parametrize("mask_dtype", [thor.DataType.bool, thor.DataType.uint8, thor.DataType.fp16, thor.DataType.fp32])
def test_sparse_categorical_cross_entropy_r11c1_accepts_all_mask_dtypes(mask_dtype):
    n = thor.Network(f"test_sparse_categorical_ce_r11c1_mask_{mask_dtype}")
    predictions, labels, mask = _r11c1_sparse_ragged_inputs(n, mask_dtype=mask_dtype)
    loss = thor.losses.SparseCategoricalCrossEntropy(n, predictions, labels, 3, mask=mask)
    assert isinstance(loss, thor.losses.SparseCategoricalCrossEntropy)


@pytest.mark.parametrize("offsets_dtype", [thor.DataType.uint32, thor.DataType.uint64])
def test_sparse_categorical_cross_entropy_r11c1_constructs_ragged_scalar_targets_and_preserves_partition(offsets_dtype):
    n = thor.Network(f"test_sparse_categorical_ce_r11c1_{offsets_dtype}")
    predictions, labels, mask = _r11c1_sparse_ragged_inputs(n, offsets_dtype=offsets_dtype)
    loss = thor.losses.SparseCategoricalCrossEntropy(
        n,
        predictions,
        labels,
        3,
        reported_loss_shape=thor.losses.LossShape.raw,
        ignore_index=2,
        mask=mask,
    )

    assert loss.is_ragged
    assert loss.get_predictions() == predictions
    assert loss.get_labels() == labels
    assert isinstance(loss.get_raw_loss(), thor.RaggedTensor)
    assert isinstance(loss.get_loss(), thor.RaggedTensor)
    assert loss.get_loss().offsets == predictions.offsets
    assert loss.get_loss().values.get_dimensions() == [8]


def test_sparse_categorical_cross_entropy_r11c1_accepts_trailing_singleton_labels_and_mask():
    n = thor.Network("test_sparse_categorical_ce_r11c1_singleton")
    predictions = thor.layers.RaggedNetworkInput(
        n,
        "predictions",
        thor.DataType.fp32,
        [3],
        batch_size=3,
        max_total_values=8,
        max_values_per_row=5,
    )
    labels = thor.layers.RaggedNetworkInput(n, "labels", thor.DataType.uint16, [1], partition=predictions)
    mask = thor.layers.RaggedNetworkInput(n, "mask", thor.DataType.fp32, [1], partition=predictions)
    loss = thor.losses.SparseCategoricalCrossEntropy(n, predictions, labels, 3, mask=mask)
    assert isinstance(loss, thor.losses.SparseCategoricalCrossEntropy)


@pytest.mark.parametrize("shape", ["batch", "per_example", "raw", "none"])
def test_sparse_categorical_cross_entropy_r11c1_reporting_shapes_construct(shape):
    n = thor.Network(f"test_sparse_categorical_ce_r11c1_{shape}")
    predictions, labels, _ = _r11c1_sparse_ragged_inputs(n, prefix=shape)
    loss = thor.losses.SparseCategoricalCrossEntropy(
        n,
        predictions,
        labels,
        3,
        reported_loss_shape=getattr(thor.losses.LossShape, shape),
    )
    if shape == "raw":
        assert isinstance(loss.get_loss(), thor.RaggedTensor)
        assert loss.get_loss().values.get_dimensions() == [8]
    elif shape == "none":
        with pytest.raises(RuntimeError, match=r"LossShape::NONE"):
            loss.get_loss()
    else:
        assert isinstance(loss.get_loss(), thor.Tensor)
        assert loss.get_loss().get_dimensions() == [1]


def test_sparse_categorical_cross_entropy_r11c1_rejects_per_output_mixed_inputs_and_partition_mismatch():
    n = thor.Network("test_sparse_categorical_ce_r11c1_reject")
    predictions, labels, _ = _r11c1_sparse_ragged_inputs(n)
    with pytest.raises(ValueError, match=r"per_output.*undefined"):
        thor.losses.SparseCategoricalCrossEntropy(
            n,
            predictions,
            labels,
            3,
            reported_loss_shape=thor.losses.LossShape.per_output,
        )

    with pytest.raises(TypeError, match=r"both be thor.Tensor or both be thor.RaggedTensor"):
        thor.losses.SparseCategoricalCrossEntropy(n, predictions, _tensor_1d(1, thor.DataType.uint32), 3)

    different_labels = thor.layers.RaggedNetworkInput(
        n,
        "different_labels",
        thor.DataType.uint32,
        [],
        batch_size=3,
        max_total_values=8,
        max_values_per_row=5,
    )
    with pytest.raises(ValueError, match=r"exact same row partition"):
        thor.losses.SparseCategoricalCrossEntropy(n, predictions, different_labels, 3)

    dense_mask = thor.Tensor([8], thor.DataType.uint8)
    with pytest.raises(TypeError, match=r"ragged predictions require mask to be thor.RaggedTensor"):
        thor.losses.SparseCategoricalCrossEntropy(n, predictions, labels, 3, mask=dense_mask)


def test_sparse_categorical_cross_entropy_r11c1_rejects_bad_scalar_dtype_mask_geometry_and_class_count():
    n = thor.Network("test_sparse_categorical_ce_r11c1_validation")
    predictions, labels, _ = _r11c1_sparse_ragged_inputs(n)

    wrong_label_dtype = thor.layers.RaggedNetworkInput(
        n, "wrong_label_dtype", thor.DataType.fp32, [], partition=predictions
    )
    with pytest.raises(ValueError, match=r"labels must use uint8, uint16, or uint32"):
        thor.losses.SparseCategoricalCrossEntropy(n, predictions, wrong_label_dtype, 3)

    wrong_label_shape = thor.layers.RaggedNetworkInput(
        n, "wrong_label_shape", thor.DataType.uint32, [2], partition=predictions
    )
    with pytest.raises(ValueError, match=r"scalar per active token"):
        thor.losses.SparseCategoricalCrossEntropy(n, predictions, wrong_label_shape, 3)

    wrong_mask_shape = thor.layers.RaggedNetworkInput(
        n, "wrong_mask_shape", thor.DataType.uint8, [2], partition=predictions
    )
    with pytest.raises(ValueError, match=r"ragged mask must be scalar per active token"):
        thor.losses.SparseCategoricalCrossEntropy(n, predictions, labels, 3, mask=wrong_mask_shape)

    wrong_mask_dtype = thor.layers.RaggedNetworkInput(
        n, "wrong_mask_dtype", thor.DataType.uint32, [], partition=predictions
    )
    with pytest.raises(ValueError, match=r"mask must use bool, uint8, fp16, or fp32"):
        thor.losses.SparseCategoricalCrossEntropy(n, predictions, labels, 3, mask=wrong_mask_dtype)

    with pytest.raises(ValueError, match=r"mismatch between num_classes 4"):
        thor.losses.SparseCategoricalCrossEntropy(n, predictions, labels, 4)
