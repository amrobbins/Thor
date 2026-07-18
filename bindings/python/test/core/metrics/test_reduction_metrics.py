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


@pytest.mark.parametrize("metric_type", [thor.metrics.Mean, thor.metrics.Sum, thor.metrics.Min, thor.metrics.Max])
def test_unary_reduction_metric_constructs(metric_type):
    net, values = _network_and_values()

    metric = metric_type(net, values)

    assert metric is not None
    assert isinstance(metric, metric_type)
    assert metric.values == values


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


def test_weighted_mean_rejects_wrong_arity():
    net, values = _network_and_values()
    weights = thor.Tensor([4], thor.DataType.fp32)

    with pytest.raises(TypeError):
        thor.metrics.WeightedMean(net, values)

    with pytest.raises(TypeError):
        thor.metrics.WeightedMean(net, values, weights, weights)
