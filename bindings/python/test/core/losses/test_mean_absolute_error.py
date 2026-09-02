# test/test_losses_mae.py
import pytest
import thor


def _net():
    return thor.Network("test_net_mae")


def _tensor_1d(size: int, dtype=thor.DataType.fp32):
    return thor.Tensor([size], dtype)


def test_mae_constructs_defaults():
    n = _net()
    preds = _tensor_1d(1, thor.DataType.fp32)
    labels = _tensor_1d(1, thor.DataType.fp32)

    loss = thor.losses.MAE(n, preds, labels)
    assert loss is not None
    assert isinstance(loss, thor.losses.MAE)


def test_mae_constructs_vector_width_100():
    n = _net()
    preds = _tensor_1d(100, thor.DataType.fp32)
    labels = _tensor_1d(100, thor.DataType.fp32)

    loss = thor.losses.MAE(n, preds, labels)
    assert loss is not None
    assert isinstance(loss, thor.losses.MAE)


def test_mae_constructs_with_loss_data_type():
    n = _net()
    preds = _tensor_1d(1, thor.DataType.fp16)
    labels = _tensor_1d(1, thor.DataType.fp16)

    loss = thor.losses.MAE(
        n,
        preds,
        labels,
        thor.DataType.fp32,  # explicitly override builder.lossDataType(...)
        thor.losses.LossShape.batch,
    )
    assert isinstance(loss, thor.losses.MAE)


def test_mae_constructs_reports_per_example():
    n = _net()
    preds = _tensor_1d(1, thor.DataType.fp32)
    labels = _tensor_1d(1, thor.DataType.fp32)

    loss = thor.losses.MAE(
        n,
        preds,
        labels,
        None,
        thor.losses.LossShape.per_example,
    )
    assert isinstance(loss, thor.losses.MAE)


@pytest.mark.parametrize(
    "reported_loss_shape, expected_dimensions",
    [
        (thor.losses.LossShape.batch, [1]),
        (thor.losses.LossShape.per_example, [1]),
        (thor.losses.LossShape.per_output, [2, 3, 4]),
        (thor.losses.LossShape.raw, [2, 3, 4]),
    ],
)
def test_mae_constructs_all_reported_loss_shapes(reported_loss_shape, expected_dimensions):
    n = _net()
    preds = thor.Tensor([2, 3, 4], thor.DataType.fp32)
    labels = thor.Tensor([2, 3, 4], thor.DataType.fp32)

    loss = thor.losses.MAE(n, preds, labels, reported_loss_shape=reported_loss_shape)
    assert loss.get_loss().get_dimensions() == expected_dimensions


def test_mae_rejects_mismatched_label_dimensions():
    n = _net()
    preds = _tensor_1d(1, thor.DataType.fp32)

    labels = _tensor_1d(2, thor.DataType.fp32)
    with pytest.raises(ValueError, match=r"predictions and labels dimensions must match"):
        thor.losses.MAE(n, preds, labels)

    labels = thor.Tensor([1, 1], thor.DataType.fp32)
    with pytest.raises(ValueError, match=r"predictions and labels dimensions must match"):
        thor.losses.MAE(n, preds, labels)


def test_mae_rejects_wrong_types():
    n = _net()
    preds = _tensor_1d(1, thor.DataType.fp32)
    labels = _tensor_1d(1, thor.DataType.fp32)

    with pytest.raises(TypeError):
        thor.losses.MAE("not a network", preds, labels)

    with pytest.raises(TypeError):
        thor.losses.MAE(n, "not a tensor", labels)

    with pytest.raises(TypeError):
        thor.losses.MAE(n, preds, "not a tensor")


def test_mae_rejects_wrong_arity():
    n = _net()
    preds = _tensor_1d(1, thor.DataType.fp32)
    labels = _tensor_1d(1, thor.DataType.fp32)

    with pytest.raises(TypeError):
        thor.losses.MAE(n, preds)  # missing labels

    with pytest.raises(TypeError):
        thor.losses.MAE(n, preds, labels, None, thor.losses.LossShape.batch, 123, 456)  # extra arg


def test_mae_constructs_with_scalar_example_weights():
    n = _net()
    preds = _tensor_1d(4, thor.DataType.fp32)
    labels = _tensor_1d(4, thor.DataType.fp32)
    weights = _tensor_1d(1, thor.DataType.fp32)

    loss = thor.losses.MAE(n, preds, labels, example_weights=weights)
    assert isinstance(loss, thor.losses.MAE)
    assert loss.example_weights == weights
    assert loss.get_example_weights() == weights


def test_mae_constructs_with_elementwise_example_weights():
    n = _net()
    preds = _tensor_1d(4, thor.DataType.fp32)
    labels = _tensor_1d(4, thor.DataType.fp32)
    weights = _tensor_1d(4, thor.DataType.fp16)

    loss = thor.losses.MAE(n, preds, labels, example_weights=weights)
    assert isinstance(loss, thor.losses.MAE)
    assert loss.example_weights == weights


def test_mae_rejects_bad_example_weights():
    n = _net()
    preds = _tensor_1d(4, thor.DataType.fp32)
    labels = _tensor_1d(4, thor.DataType.fp32)

    with pytest.raises(ValueError, match=r"example_weights must be distinct"):
        thor.losses.MAE(n, preds, labels, example_weights=labels)

    with pytest.raises(ValueError, match=r"example_weights must use fp8_e4m3"):
        thor.losses.MAE(n, preds, labels, example_weights=_tensor_1d(1, thor.DataType.uint32))

    with pytest.raises(ValueError, match=r"example_weights dimensions must be \[1\]"):
        thor.losses.MAE(n, preds, labels, example_weights=_tensor_1d(3, thor.DataType.fp32))


def _ragged_pair(network, dtype=thor.DataType.fp32):
    predictions = thor.layers.RaggedNetworkInput(
        network,
        "ragged_predictions",
        dtype,
        [2],
        batch_size=3,
        max_total_values=8,
        max_values_per_row=4,
    )
    labels = thor.layers.RaggedNetworkInput(
        network,
        "ragged_labels",
        dtype,
        [2],
        partition=predictions,
    )
    return predictions, labels


def test_mae_constructs_ragged_raw_and_preserves_partition():
    n = _net()
    predictions, labels = _ragged_pair(n)
    loss = thor.losses.MAE(n, predictions, labels, reported_loss_shape=thor.losses.LossShape.raw)

    assert loss.is_ragged
    assert loss.get_predictions() == predictions
    assert loss.get_labels() == labels
    raw = loss.get_loss()
    assert isinstance(raw, thor.RaggedTensor)
    assert raw.offsets == predictions.offsets
    assert raw.trailing_dimensions == [2]
    assert loss.get_raw_loss() == raw


def test_mae_constructs_ragged_dense_reporting_shapes():
    for shape in (thor.losses.LossShape.per_example, thor.losses.LossShape.batch):
        n = _net()
        predictions, labels = _ragged_pair(n, thor.DataType.fp16)
        loss = thor.losses.MAE(n, predictions, labels, thor.DataType.fp32, shape)
        reported = loss.get_loss()
        assert isinstance(reported, thor.Tensor)
        assert reported.get_dimensions() == [1]
        assert loss.get_raw_loss().offsets == predictions.offsets


def test_mae_ragged_rejects_per_output_and_different_partition():
    n = _net()
    predictions, labels = _ragged_pair(n)
    with pytest.raises(ValueError, match=r"per_output.*undefined"):
        thor.losses.MAE(n, predictions, labels, reported_loss_shape=thor.losses.LossShape.per_output)

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
        thor.losses.MAE(n, predictions, different_labels)


@pytest.mark.parametrize(
    "weight_dtype",
    [
        thor.DataType.fp8_e4m3,
        thor.DataType.fp8_e5m2,
        thor.DataType.fp16,
        thor.DataType.bf16,
        thor.DataType.fp32,
    ],
)
def test_mae_ragged_accepts_dense_per_row_example_weights_with_dense_dtype_parity(weight_dtype):
    n = _net()
    predictions, labels = _ragged_pair(n)
    weights_input = thor.layers.NetworkInput(n, "weights", [1], weight_dtype)
    weights = weights_input.get_feature_output()
    loss = thor.losses.MAE(n, predictions, labels, example_weights=weights)
    assert loss.example_weights == weights
    assert loss.get_example_weights() == weights


def test_mae_ragged_rejects_non_per_row_or_non_floating_example_weights():
    n = _net()
    predictions, labels = _ragged_pair(n)
    with pytest.raises(ValueError, match=r"ragged example_weights dimensions must be \[1\]"):
        thor.losses.MAE(n, predictions, labels, example_weights=thor.Tensor([2], thor.DataType.fp32))
    with pytest.raises(ValueError, match=r"example_weights must use fp8_e4m3"):
        thor.losses.MAE(n, predictions, labels, example_weights=thor.Tensor([1], thor.DataType.uint32))


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
def test_mae_ragged_matches_dense_prediction_dtype_and_default_loss_storage(dtype, expected_loss_dtype):
    n = _net()
    predictions, labels = _ragged_pair(n, dtype)
    loss = thor.losses.MAE(n, predictions, labels, reported_loss_shape=thor.losses.LossShape.raw)
    assert loss.get_raw_loss().values.get_data_type() == expected_loss_dtype


@pytest.mark.parametrize(
    "label_dtype",
    [
        thor.DataType.bool,
        thor.DataType.int8,
        thor.DataType.int16,
        thor.DataType.int32,
        thor.DataType.int64,
        thor.DataType.uint8,
        thor.DataType.uint16,
        thor.DataType.uint32,
        thor.DataType.uint64,
        thor.DataType.fp8_e4m3,
        thor.DataType.fp8_e5m2,
        thor.DataType.fp16,
        thor.DataType.bf16,
        thor.DataType.fp32,
    ],
)
def test_mae_ragged_matches_dense_label_dtype_contract(label_dtype):
    n = _net()
    predictions = thor.layers.RaggedNetworkInput(
        n,
        "ragged_predictions",
        thor.DataType.bf16,
        [2],
        batch_size=3,
        max_total_values=8,
        max_values_per_row=4,
    )
    labels = thor.layers.RaggedNetworkInput(
        n,
        "ragged_labels",
        label_dtype,
        [2],
        partition=predictions,
    )
    thor.losses.MAE(n, predictions, labels, thor.DataType.fp32, thor.losses.LossShape.raw)
