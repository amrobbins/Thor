import numpy as np
import pytest

import thor


SUPPORTED_REDUCTION_DTYPES = [
    thor.DataType.fp8_e4m3,
    thor.DataType.fp8_e5m2,
    thor.DataType.fp16,
    thor.DataType.bf16,
    thor.DataType.fp32,
]


def _network_and_values(n: int = 4):
    net = thor.Network("test_net_reduction_metrics")
    values = thor.Tensor([n], thor.DataType.fp32)
    return net, values


def _cpu_tensor(values: np.ndarray, dtype: thor.DataType) -> thor.physical.PhysicalTensor:
    values = np.asarray(values, dtype=thor.physical.numpy_dtypes.from_thor(dtype), order="C")
    placement = thor.physical.Placement(thor.physical.DeviceType.cpu, 0)
    descriptor = thor.physical.PhysicalTensor.Descriptor(dtype, list(values.shape))
    tensor = thor.physical.PhysicalTensor(placement, descriptor)
    tensor.numpy()[...] = values
    return tensor


@pytest.mark.parametrize("metric_type", [thor.metrics.Mean, thor.metrics.Sum, thor.metrics.Min, thor.metrics.Max])
def test_unary_reduction_metric_constructs(metric_type):
    net, values = _network_and_values()

    metric = metric_type(net, values)

    assert metric is not None
    assert isinstance(metric, metric_type)
    assert metric.values == values
    expected = {
        thor.metrics.Mean: thor.MetricAggregation.MEAN_BY_EXAMPLE,
        thor.metrics.Sum: thor.MetricAggregation.SUM,
        thor.metrics.Min: thor.MetricAggregation.MIN,
        thor.metrics.Max: thor.MetricAggregation.MAX,
    }[metric_type]
    assert metric.aggregation is expected


@pytest.mark.parametrize("dtype", SUPPORTED_REDUCTION_DTYPES)
@pytest.mark.parametrize("metric_type", [thor.metrics.Mean, thor.metrics.Sum, thor.metrics.Min, thor.metrics.Max])
def test_unary_reduction_metric_supports_thor_floating_storage_dtypes(metric_type, dtype):
    net = thor.Network(f"test_{metric_type.__name__}_{dtype}")
    values = thor.Tensor([4], dtype)

    metric = metric_type(net, values)

    assert metric.values.get_data_type() == dtype
    assert metric.get_metric().get_data_type() == thor.DataType.fp32

@pytest.mark.parametrize("dtype", SUPPORTED_REDUCTION_DTYPES)
def test_weighted_mean_supports_thor_floating_storage_dtypes(dtype):
    net = thor.Network(f"test_weighted_mean_{dtype}")
    values = thor.Tensor([4], dtype)
    weights = thor.Tensor([4], dtype)

    metric = thor.metrics.WeightedMean(net, values, weights)

    assert metric.values.get_data_type() == dtype
    assert metric.weights.get_data_type() == dtype
    assert metric.get_metric().get_data_type() == thor.DataType.fp32


@pytest.mark.parametrize("metric_type", [thor.metrics.Mean, thor.metrics.Sum, thor.metrics.Min, thor.metrics.Max])
def test_unary_reduction_metric_rejects_wrong_arity(metric_type):
    net, values = _network_and_values()

    with pytest.raises(TypeError):
        metric_type(net)

    with pytest.raises(TypeError):
        metric_type(net, values, values)


def test_weighted_mean_constructs():
    net, values = _network_and_values()
    weights = thor.Tensor([4], thor.DataType.fp32)

    metric = thor.metrics.WeightedMean(net, values, weights)

    assert metric is not None
    assert isinstance(metric, thor.metrics.WeightedMean)
    assert metric.values == values
    assert metric.weights == weights
    assert metric.aggregation is thor.MetricAggregation.RATIO


def test_weighted_mean_rejects_wrong_arity():
    net, values = _network_and_values()
    weights = thor.Tensor([4], thor.DataType.fp32)

    with pytest.raises(TypeError):
        thor.metrics.WeightedMean(net, values)

    with pytest.raises(TypeError):
        thor.metrics.WeightedMean(net, values, weights, weights)


def _r10l_ragged_values(network, name, dtype=thor.DataType.fp32, offsets_dtype=thor.DataType.uint32):
    return thor.layers.RaggedNetworkInput(
        network,
        name,
        dtype,
        [2],
        batch_size=4,
        max_total_values=9,
        max_values_per_row=4,
        offsets_data_type=offsets_dtype,
    )


@pytest.mark.parametrize("metric_type", [thor.metrics.Sum, thor.metrics.Mean])
@pytest.mark.parametrize("offsets_dtype", [thor.DataType.uint32, thor.DataType.uint64])
def test_r10l_sum_and_mean_construct_from_ragged_values(metric_type, offsets_dtype):
    net = thor.Network(f"r10l_{metric_type.__name__}_{offsets_dtype}")
    values = _r10l_ragged_values(net, "values", thor.DataType.fp16, offsets_dtype)

    metric = metric_type(net, values)

    assert metric.ragged_values == values
    assert metric.values == values.values
    assert metric.get_metric().get_data_type() == thor.DataType.fp32
    expected_aggregation = (
        thor.MetricAggregation.RATIO if metric_type is thor.metrics.Mean else thor.MetricAggregation.SUM
    )
    assert metric.aggregation is expected_aggregation


def test_r10l_min_and_max_do_not_accidentally_accept_ragged_values():
    net = thor.Network("r10l_extrema_not_yet")
    values = _r10l_ragged_values(net, "values")
    with pytest.raises(TypeError):
        thor.metrics.Min(net, values)
    with pytest.raises(TypeError):
        thor.metrics.Max(net, values)


@pytest.mark.cuda
@pytest.mark.parametrize("offsets_dtype,np_offsets_dtype", [
    (thor.DataType.uint32, np.uint32),
    (thor.DataType.uint64, np.uint64),
])
def test_r10l_ragged_sum_and_mean_ignore_inactive_capacity_and_handle_empty_rows(offsets_dtype, np_offsets_dtype):
    batch_size = 4
    capacity = 9
    net = thor.Network(f"r10l_execute_{offsets_dtype}")
    values = thor.layers.RaggedNetworkInput(
        net,
        "values",
        thor.DataType.fp16,
        [2],
        batch_size=batch_size,
        max_total_values=capacity,
        max_values_per_row=4,
        offsets_data_type=offsets_dtype,
    )
    sum_metric = thor.metrics.Sum(net, values)
    mean_metric = thor.metrics.Mean(net, values)
    thor.layers.NetworkOutput(net, "sum", sum_metric.get_metric(), thor.DataType.fp32)
    thor.layers.NetworkOutput(net, "mean", mean_metric.get_metric(), thor.DataType.fp32)

    placed = net.place(batch_size, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)

    active = np.asarray(
        [[1.0, -2.0], [3.0, 4.0], [-5.0, 6.0], [7.0, 8.0], [9.0, -10.0]], dtype=np.float16
    )
    packed = np.full((capacity, 2), np.float16(60000.0), dtype=np.float16)
    packed[: len(active)] = active
    offsets_np = np.asarray([0, 2, 2, 5, 5], dtype=np_offsets_dtype)
    physical = thor.physical.PhysicalRaggedTensor(
        _cpu_tensor(packed, thor.DataType.fp16),
        _cpu_tensor(offsets_np, offsets_dtype),
        max_values_per_row=4,
    )

    outputs = placed.infer({"values": physical})
    expected_sum = np.sum(active.astype(np.float32), dtype=np.float32)
    expected_mean = expected_sum / np.float32(active.size)
    assert float(outputs["sum"].numpy().reshape(-1)[0]) == pytest.approx(float(expected_sum), rel=1e-5, abs=1e-5)
    assert float(outputs["mean"].numpy().reshape(-1)[0]) == pytest.approx(float(expected_mean), rel=1e-5, abs=1e-5)


@pytest.mark.cuda
def test_r10l_ragged_sum_and_mean_all_empty_batch_report_zero():
    batch_size = 3
    capacity = 7
    net = thor.Network("r10l_all_empty")
    values = thor.layers.RaggedNetworkInput(
        net, "values", thor.DataType.fp32, [3], batch_size=batch_size, max_total_values=capacity
    )
    sum_metric = thor.metrics.Sum(net, values)
    mean_metric = thor.metrics.Mean(net, values)
    thor.layers.NetworkOutput(net, "sum", sum_metric.get_metric(), thor.DataType.fp32)
    thor.layers.NetworkOutput(net, "mean", mean_metric.get_metric(), thor.DataType.fp32)
    placed = net.place(batch_size, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)

    packed = np.full((capacity, 3), np.float32(np.nan), dtype=np.float32)
    offsets = np.zeros(batch_size + 1, dtype=np.uint32)
    physical = thor.physical.PhysicalRaggedTensor(
        _cpu_tensor(packed, thor.DataType.fp32), _cpu_tensor(offsets, thor.DataType.uint32)
    )
    outputs = placed.infer({"values": physical})
    assert float(outputs["sum"].numpy().reshape(-1)[0]) == pytest.approx(0.0)
    assert float(outputs["mean"].numpy().reshape(-1)[0]) == pytest.approx(0.0)
