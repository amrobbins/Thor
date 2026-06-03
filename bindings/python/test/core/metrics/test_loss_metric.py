import pytest

import thor


def _network_predictions_labels(n: int = 4):
    net = thor.Network("test_net_loss_metric")
    predictions = thor.Tensor([n], thor.DataType.fp32)
    labels = thor.Tensor([n], thor.DataType.fp32)
    return net, predictions, labels


def test_loss_metric_constructs_default_mse():
    net, predictions, labels = _network_predictions_labels()

    metric = thor.metrics.LossMetric(net, predictions, labels)

    assert metric is not None
    assert isinstance(metric, thor.metrics.LossMetric)
    assert metric.predictions == predictions
    assert metric.labels == labels
    assert metric.formula == thor.metrics.LossFormula.mean_squared_error


@pytest.mark.parametrize(
    "formula",
    [
        thor.metrics.LossFormula.mean_squared_error,
        thor.metrics.LossFormula.mean_absolute_error,
        thor.metrics.LossFormula.mean_absolute_percentage_error,
    ],
)
def test_loss_metric_constructs_supported_formulas(formula):
    net, predictions, labels = _network_predictions_labels()

    metric = thor.metrics.LossMetric(net, predictions, labels, formula=formula)

    assert metric is not None
    assert isinstance(metric, thor.metrics.LossMetric)
    assert metric.formula == formula


def test_loss_metric_rejects_wrong_arity():
    net, predictions, labels = _network_predictions_labels()

    with pytest.raises(TypeError):
        thor.metrics.LossMetric(net, predictions)

    with pytest.raises(TypeError):
        thor.metrics.LossMetric(net, predictions, labels, labels)
