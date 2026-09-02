# test/test_losses_mse.py
import pytest
import thor


def _net():
    return thor.Network("test_net_mse")


def _tensor_1d(size: int, dtype=thor.DataType.fp32):
    return thor.Tensor([size], dtype)


def test_mse_constructs_defaults():
    n = _net()
    preds = _tensor_1d(1, thor.DataType.fp32)
    labels = _tensor_1d(1, thor.DataType.fp32)

    loss = thor.losses.MSE(n, preds, labels)
    assert loss is not None
    assert isinstance(loss, thor.losses.MSE)


def test_mse_constructs_vector_width_100():
    n = _net()
    preds = _tensor_1d(100, thor.DataType.fp32)
    labels = _tensor_1d(100, thor.DataType.fp32)

    loss = thor.losses.MSE(n, preds, labels)
    assert loss is not None
    assert isinstance(loss, thor.losses.MSE)


def test_mse_exposes_default_and_custom_loss_weight():
    n = _net()
    preds = _tensor_1d(1, thor.DataType.fp32)
    labels = _tensor_1d(1, thor.DataType.fp32)

    default_loss = thor.losses.MSE(n, preds, labels)
    assert default_loss.loss_weight is None

    explicit_one_loss = thor.losses.MSE(n, preds, labels, loss_weight=1.0)
    assert explicit_one_loss.loss_weight is None

    weighted_loss = thor.losses.MSE(n, preds, labels, loss_weight=2.5)
    assert weighted_loss.loss_weight == pytest.approx(2.5)


def test_mse_constructs_with_loss_data_type():
    n = _net()
    preds = _tensor_1d(1, thor.DataType.fp16)
    labels = _tensor_1d(1, thor.DataType.fp16)

    loss = thor.losses.MSE(
        n,
        preds,
        labels,
        thor.DataType.fp32,  # explicitly override builder.lossDataType(...)
        thor.losses.LossShape.batch,
    )
    assert isinstance(loss, thor.losses.MSE)


def test_mse_constructs_reports_per_example():
    n = _net()
    preds = _tensor_1d(1, thor.DataType.fp32)
    labels = _tensor_1d(1, thor.DataType.fp32)

    loss = thor.losses.MSE(
        n,
        preds,
        labels,
        None,
        thor.losses.LossShape.per_example,
    )
    assert isinstance(loss, thor.losses.MSE)


@pytest.mark.parametrize(
    "reported_loss_shape, expected_dimensions",
    [
        (thor.losses.LossShape.batch, [1]),
        (thor.losses.LossShape.per_example, [1]),
        (thor.losses.LossShape.per_output, [2, 3, 4]),
        (thor.losses.LossShape.raw, [2, 3, 4]),
    ],
)
def test_mse_constructs_all_reported_loss_shapes(reported_loss_shape, expected_dimensions):
    n = _net()
    preds = thor.Tensor([2, 3, 4], thor.DataType.fp32)
    labels = thor.Tensor([2, 3, 4], thor.DataType.fp32)

    loss = thor.losses.MSE(n, preds, labels, reported_loss_shape=reported_loss_shape)
    assert loss.get_loss().get_dimensions() == expected_dimensions


def test_mse_rejects_mismatched_label_dimensions():
    n = _net()
    preds = _tensor_1d(1, thor.DataType.fp32)

    labels = _tensor_1d(2, thor.DataType.fp32)
    with pytest.raises(ValueError, match=r"predictions and labels dimensions must match"):
        thor.losses.MSE(n, preds, labels)

    labels = thor.Tensor([1, 1], thor.DataType.fp32)
    with pytest.raises(ValueError, match=r"predictions and labels dimensions must match"):
        thor.losses.MSE(n, preds, labels)


def test_mse_rejects_wrong_types():
    n = _net()
    preds = _tensor_1d(1, thor.DataType.fp32)
    labels = _tensor_1d(1, thor.DataType.fp32)

    with pytest.raises(TypeError):
        thor.losses.MSE("not a network", preds, labels)

    with pytest.raises(TypeError):
        thor.losses.MSE(n, "not a tensor", labels)

    with pytest.raises(TypeError):
        thor.losses.MSE(n, preds, "not a tensor")


def test_mse_rejects_wrong_arity():
    n = _net()
    preds = _tensor_1d(1, thor.DataType.fp32)
    labels = _tensor_1d(1, thor.DataType.fp32)

    with pytest.raises(TypeError):
        thor.losses.MSE(n, preds)  # missing labels

    with pytest.raises(TypeError):
        thor.losses.MSE(n, preds, labels, None, thor.losses.LossShape.batch, 123, 456)  # extra arg


def test_mse_constructs_with_scalar_example_weights():
    n = _net()
    preds = _tensor_1d(4, thor.DataType.fp32)
    labels = _tensor_1d(4, thor.DataType.fp32)
    weights = _tensor_1d(1, thor.DataType.fp32)

    loss = thor.losses.MSE(n, preds, labels, example_weights=weights)
    assert isinstance(loss, thor.losses.MSE)
    assert loss.example_weights == weights
    assert loss.get_example_weights() == weights


def test_mse_constructs_with_elementwise_example_weights():
    n = _net()
    preds = _tensor_1d(4, thor.DataType.fp32)
    labels = _tensor_1d(4, thor.DataType.fp32)
    weights = _tensor_1d(4, thor.DataType.fp16)

    loss = thor.losses.MSE(n, preds, labels, example_weights=weights)
    assert isinstance(loss, thor.losses.MSE)
    assert loss.example_weights == weights


def test_mse_rejects_bad_example_weights():
    n = _net()
    preds = _tensor_1d(4, thor.DataType.fp32)
    labels = _tensor_1d(4, thor.DataType.fp32)

    with pytest.raises(ValueError, match=r"example_weights must be distinct"):
        thor.losses.MSE(n, preds, labels, example_weights=labels)

    with pytest.raises(ValueError, match=r"example_weights must use fp8_e4m3"):
        thor.losses.MSE(n, preds, labels, example_weights=_tensor_1d(1, thor.DataType.uint32))

    with pytest.raises(ValueError, match=r"example_weights dimensions must be \[1\]"):
        thor.losses.MSE(n, preds, labels, example_weights=_tensor_1d(3, thor.DataType.fp32))


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


def test_mse_constructs_ragged_raw_and_dense_reporting():
    n = _net()
    predictions, labels = _ragged_pair(n, thor.DataType.bf16, thor.DataType.int32)
    raw_loss = thor.losses.MSE(n, predictions, labels, reported_loss_shape=thor.losses.LossShape.raw)
    assert raw_loss.is_ragged
    assert raw_loss.get_predictions() == predictions
    assert raw_loss.get_labels() == labels
    assert isinstance(raw_loss.get_loss(), thor.RaggedTensor)
    assert raw_loss.get_loss().offsets == predictions.offsets
    assert raw_loss.get_raw_loss().values.get_data_type() == thor.DataType.fp32

    n2 = _net()
    predictions2, labels2 = _ragged_pair(n2)
    per_example = thor.losses.MSE(n2, predictions2, labels2, reported_loss_shape=thor.losses.LossShape.per_example)
    assert isinstance(per_example.get_loss(), thor.Tensor)
    assert per_example.get_loss().get_dimensions() == [1]


def test_mse_ragged_rejects_per_output_and_different_partition():
    n = _net()
    predictions, labels = _ragged_pair(n)
    with pytest.raises(ValueError, match=r"per_output.*undefined"):
        thor.losses.MSE(n, predictions, labels, reported_loss_shape=thor.losses.LossShape.per_output)

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
        thor.losses.MSE(n, predictions, different_labels)


def test_mse_ragged_accepts_dense_per_row_example_weights():
    n = _net()
    predictions, labels = _ragged_pair(n)
    weights_input = thor.layers.NetworkInput(n, "weights", [1], thor.DataType.fp8_e4m3)
    weights = weights_input.get_feature_output()
    loss = thor.losses.MSE(n, predictions, labels, example_weights=weights)
    assert loss.example_weights == weights
    assert loss.get_example_weights() == weights


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
def test_mse_ragged_matches_dense_prediction_dtype_contract(dtype, expected_loss_dtype):
    n = _net()
    predictions, labels = _ragged_pair(n, dtype, thor.DataType.int32)
    loss = thor.losses.MSE(n, predictions, labels, reported_loss_shape=thor.losses.LossShape.raw)
    assert loss.get_raw_loss().values.get_data_type() == expected_loss_dtype
