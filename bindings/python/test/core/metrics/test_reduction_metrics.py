import pytest

import thor


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
