# test/test_metrics_binary_accuracy.py
import pytest

import thor


def _make_binary_vectors(n: int = 1):
    net = thor.Network("test_net_binary_accuracy")

    preds = thor.Tensor([n], thor.DataType.fp32)  # shape [n]
    labs = thor.Tensor([n], thor.DataType.fp32)  # or bool / int, depending on your API

    return net, preds, labs


def test_binary_accuracy_constructs():
    net, preds, labs = _make_binary_vectors()

    m = thor.metrics.BinaryAccuracy(net, preds, labs)

    assert m is not None
    assert isinstance(m, thor.metrics.BinaryAccuracy)
    assert m.aggregation is thor.MetricAggregation.MEAN_BY_EXAMPLE


def test_binary_accuracy_rejects_wrong_arity():
    net, preds, labs = _make_binary_vectors()

    with pytest.raises(TypeError):
        thor.metrics.BinaryAccuracy(net, preds)  # missing labels

    with pytest.raises(TypeError):
        thor.metrics.BinaryAccuracy(net, preds, labs, 123)  # extra arg


def test_binary_accuracy_rejects_wrong_types():
    net, preds, labs = _make_binary_vectors()

    with pytest.raises(TypeError):
        thor.metrics.BinaryAccuracy("not a network", preds, labs)

    with pytest.raises(TypeError):
        thor.metrics.BinaryAccuracy(net, "not a tensor", labs)

    with pytest.raises(TypeError):
        thor.metrics.BinaryAccuracy(net, preds, "not a tensor")


def _r10o_cpu_tensor(values, dtype):
    import numpy as np

    values = np.asarray(values, dtype=thor.physical.numpy_dtypes.from_thor(dtype), order="C")
    placement = thor.physical.Placement(thor.physical.DeviceType.cpu, 0)
    descriptor = thor.physical.PhysicalTensor.Descriptor(dtype, list(values.shape))
    tensor = thor.physical.PhysicalTensor(placement, descriptor)
    tensor.numpy()[...] = values
    return tensor


@pytest.mark.parametrize("offsets_dtype", [thor.DataType.uint32, thor.DataType.uint64])
def test_r10o_binary_accuracy_constructs_from_same_partition_ragged_tokens(offsets_dtype):
    net = thor.Network(f"r10o_binary_api_{offsets_dtype}")
    predictions = thor.layers.RaggedNetworkInput(
        net,
        "predictions",
        thor.DataType.fp16,
        [1],
        batch_size=4,
        max_total_values=9,
        max_values_per_row=4,
        offsets_data_type=offsets_dtype,
    )
    labels = thor.layers.RaggedNetworkInput(
        net,
        "labels",
        thor.DataType.uint8,
        [1],
        partition=predictions,
    )

    metric = thor.metrics.BinaryAccuracy(net, predictions, labels)

    assert metric.ragged_predictions == predictions
    assert metric.ragged_labels == labels
    assert metric.aggregation is thor.MetricAggregation.RATIO
    assert metric.get_metric().get_data_type() == thor.DataType.fp32


def test_r10o_binary_accuracy_rejects_different_ragged_partitions():
    net = thor.Network("r10o_binary_partition_reject")
    predictions = thor.layers.RaggedNetworkInput(
        net, "predictions", thor.DataType.fp32, [1], batch_size=3, max_total_values=7
    )
    labels = thor.layers.RaggedNetworkInput(
        net, "labels", thor.DataType.uint8, [1], batch_size=3, max_total_values=7
    )

    with pytest.raises(ValueError, match=r"exact same row partition"):
        thor.metrics.BinaryAccuracy(net, predictions, labels)


@pytest.mark.cuda
@pytest.mark.parametrize("offsets_dtype,np_offsets_dtype", [
    (thor.DataType.uint32, "uint32"),
    (thor.DataType.uint64, "uint64"),
])
def test_r10o_binary_accuracy_uses_active_tokens_only(offsets_dtype, np_offsets_dtype):
    import numpy as np

    batch_size = 4
    capacity = 9
    net = thor.Network(f"r10o_binary_execute_{offsets_dtype}")
    predictions = thor.layers.RaggedNetworkInput(
        net,
        "predictions",
        thor.DataType.fp32,
        [1],
        batch_size=batch_size,
        max_total_values=capacity,
        max_values_per_row=4,
        offsets_data_type=offsets_dtype,
    )
    labels = thor.layers.RaggedNetworkInput(net, "labels", thor.DataType.uint8, [1], partition=predictions)
    metric = thor.metrics.BinaryAccuracy(net, predictions, labels)
    thor.layers.NetworkOutput(net, "accuracy", metric.get_metric(), thor.DataType.fp32)
    placed = net.place(batch_size, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)

    active_predictions = np.asarray([[0.9], [0.1], [0.8], [0.2], [0.7]], dtype=np.float32)
    active_labels = np.asarray([[1], [1], [1], [0], [0]], dtype=np.uint8)
    packed_predictions = np.full((capacity, 1), np.float32(np.nan), dtype=np.float32)
    packed_labels = np.full((capacity, 1), np.uint8(255), dtype=np.uint8)
    packed_predictions[: len(active_predictions)] = active_predictions
    packed_labels[: len(active_labels)] = active_labels
    offsets_np = np.asarray([0, 2, 2, 5, 5], dtype=np.dtype(np_offsets_dtype))
    physical_predictions = thor.physical.PhysicalRaggedTensor(
        _r10o_cpu_tensor(packed_predictions, thor.DataType.fp32),
        _r10o_cpu_tensor(offsets_np, offsets_dtype),
        max_values_per_row=4,
    )

    outputs = placed.infer({
        "predictions": physical_predictions,
        "labels": _r10o_cpu_tensor(packed_labels, thor.DataType.uint8),
    })
    # Correct tokens: 0, 2, 3 => 3 / 5.
    assert float(outputs["accuracy"].numpy().reshape(-1)[0]) == pytest.approx(3.0 / 5.0, rel=1e-6, abs=1e-6)


@pytest.mark.cuda
def test_r10o_binary_accuracy_all_empty_reports_zero():
    import numpy as np

    net = thor.Network("r10o_binary_all_empty")
    predictions = thor.layers.RaggedNetworkInput(
        net, "predictions", thor.DataType.fp32, [1], batch_size=3, max_total_values=7
    )
    labels = thor.layers.RaggedNetworkInput(net, "labels", thor.DataType.uint8, [1], partition=predictions)
    metric = thor.metrics.BinaryAccuracy(net, predictions, labels)
    thor.layers.NetworkOutput(net, "accuracy", metric.get_metric(), thor.DataType.fp32)
    placed = net.place(3, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)

    physical_predictions = thor.physical.PhysicalRaggedTensor(
        _r10o_cpu_tensor(np.full((7, 1), np.nan, dtype=np.float32), thor.DataType.fp32),
        _r10o_cpu_tensor(np.zeros(4, dtype=np.uint32), thor.DataType.uint32),
    )
    outputs = placed.infer({
        "predictions": physical_predictions,
        "labels": _r10o_cpu_tensor(np.full((7, 1), 255, dtype=np.uint8), thor.DataType.uint8),
    })
    assert float(outputs["accuracy"].numpy().reshape(-1)[0]) == pytest.approx(0.0)
