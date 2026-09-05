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
    assert m.aggregation is thor.MetricAggregation.MEAN_BY_EXAMPLE


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


def _r10o_cpu_tensor(values, dtype):
    import numpy as np

    values = np.asarray(values, dtype=thor.physical.numpy_dtypes.from_thor(dtype), order="C")
    placement = thor.physical.Placement(thor.physical.DeviceType.cpu, 0)
    descriptor = thor.physical.PhysicalTensor.Descriptor(dtype, list(values.shape))
    tensor = thor.physical.PhysicalTensor(placement, descriptor)
    tensor.numpy()[...] = values
    return tensor


def _r10o_categorical_inputs(network, offsets_dtype=thor.DataType.uint32, one_hot=False):
    predictions = thor.layers.RaggedNetworkInput(
        network,
        "predictions",
        thor.DataType.fp32,
        [3],
        batch_size=4,
        max_total_values=8,
        max_values_per_row=4,
        offsets_data_type=offsets_dtype,
    )
    label_width = [3] if one_hot else [1]
    label_dtype = thor.DataType.fp32 if one_hot else thor.DataType.uint32
    labels = thor.layers.RaggedNetworkInput(network, "labels", label_dtype, label_width, partition=predictions)
    return predictions, labels


@pytest.mark.parametrize("offsets_dtype", [thor.DataType.uint32, thor.DataType.uint64])
@pytest.mark.parametrize("label_type,one_hot", [
    (thor.losses.LabelType.index, False),
    (thor.losses.LabelType.one_hot, True),
])
def test_r10o_categorical_accuracy_constructs_from_same_partition_ragged_tokens(offsets_dtype, label_type, one_hot):
    net = thor.Network(f"r10o_cat_api_{offsets_dtype}_{one_hot}")
    predictions, labels = _r10o_categorical_inputs(net, offsets_dtype, one_hot)

    metric = thor.metrics.CategoricalAccuracy(net, predictions, labels, label_type, 3)

    assert metric.ragged_predictions == predictions
    assert metric.ragged_labels == labels
    assert metric.aggregation is thor.MetricAggregation.RATIO
    assert metric.get_metric().get_data_type() == thor.DataType.fp32


def test_r10o_categorical_accuracy_rejects_different_ragged_partitions():
    net = thor.Network("r10o_cat_partition_reject")
    predictions = thor.layers.RaggedNetworkInput(
        net, "predictions", thor.DataType.fp32, [3], batch_size=3, max_total_values=7
    )
    labels = thor.layers.RaggedNetworkInput(
        net, "labels", thor.DataType.uint32, [1], batch_size=3, max_total_values=7
    )
    with pytest.raises(ValueError, match=r"exact same row partition"):
        thor.metrics.CategoricalAccuracy(net, predictions, labels, thor.losses.LabelType.index, 3)


@pytest.mark.cuda
@pytest.mark.parametrize("offsets_dtype,np_offsets_dtype", [
    (thor.DataType.uint32, "uint32"),
    (thor.DataType.uint64, "uint64"),
])
def test_r10o_categorical_index_accuracy_keeps_sequence_and_class_axes_separate(offsets_dtype, np_offsets_dtype):
    import numpy as np

    net = thor.Network(f"r10o_cat_index_execute_{offsets_dtype}")
    predictions, labels = _r10o_categorical_inputs(net, offsets_dtype, False)
    metric = thor.metrics.CategoricalAccuracy(net, predictions, labels, thor.losses.LabelType.index, 3)
    thor.layers.NetworkOutput(net, "accuracy", metric.get_metric(), thor.DataType.fp32)
    placed = net.place(4, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)

    active_predictions = np.asarray([
        [9.0, 1.0, 0.0],   # class 0, correct
        [0.0, 2.0, 8.0],   # class 2, wrong for label 1
        [0.0, 7.0, 1.0],   # class 1, correct
        [6.0, 5.0, 4.0],   # class 0, correct
        [1.0, 3.0, 2.0],   # class 1, wrong for label 2
    ], dtype=np.float32)
    active_labels = np.asarray([[0], [1], [1], [0], [2]], dtype=np.uint32)
    packed_predictions = np.full((8, 3), np.nan, dtype=np.float32)
    packed_labels = np.full((8, 1), np.uint32(99), dtype=np.uint32)
    packed_predictions[:5] = active_predictions
    packed_labels[:5] = active_labels
    offsets_np = np.asarray([0, 2, 2, 5, 5], dtype=np.dtype(np_offsets_dtype))
    physical_predictions = thor.physical.PhysicalRaggedTensor(
        _r10o_cpu_tensor(packed_predictions, thor.DataType.fp32),
        _r10o_cpu_tensor(offsets_np, offsets_dtype),
        max_values_per_row=4,
    )
    outputs = placed.infer({
        "predictions": physical_predictions,
        "labels": _r10o_cpu_tensor(packed_labels, thor.DataType.uint32),
    })
    assert float(outputs["accuracy"].numpy().reshape(-1)[0]) == pytest.approx(3.0 / 5.0, rel=1e-6, abs=1e-6)


@pytest.mark.cuda
def test_r10o_categorical_one_hot_accuracy_argmaxes_each_active_token():
    import numpy as np

    net = thor.Network("r10o_cat_one_hot_execute")
    predictions, labels = _r10o_categorical_inputs(net, thor.DataType.uint32, True)
    metric = thor.metrics.CategoricalAccuracy(net, predictions, labels, thor.losses.LabelType.one_hot, 3)
    thor.layers.NetworkOutput(net, "accuracy", metric.get_metric(), thor.DataType.fp32)
    placed = net.place(4, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)

    active_predictions = np.asarray([
        [0.9, 0.1, 0.0],
        [0.1, 0.2, 0.7],
        [0.2, 0.6, 0.2],
        [0.8, 0.1, 0.1],
    ], dtype=np.float32)
    active_labels = np.asarray([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.9, 0.1],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    packed_predictions = np.full((8, 3), np.nan, dtype=np.float32)
    packed_labels = np.full((8, 3), np.nan, dtype=np.float32)
    packed_predictions[:4] = active_predictions
    packed_labels[:4] = active_labels
    offsets_np = np.asarray([0, 1, 1, 4, 4], dtype=np.uint32)
    physical_predictions = thor.physical.PhysicalRaggedTensor(
        _r10o_cpu_tensor(packed_predictions, thor.DataType.fp32),
        _r10o_cpu_tensor(offsets_np, thor.DataType.uint32),
        max_values_per_row=4,
    )
    outputs = placed.infer({
        "predictions": physical_predictions,
        "labels": _r10o_cpu_tensor(packed_labels, thor.DataType.fp32),
    })
    # Tokens 0 and 2 are correct; tokens 1 and 3 are wrong.
    assert float(outputs["accuracy"].numpy().reshape(-1)[0]) == pytest.approx(0.5, rel=1e-6, abs=1e-6)
