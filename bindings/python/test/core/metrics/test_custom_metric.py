import pytest

import thor
from thor.physical import DynamicExpression, DynamicExpressionBuild
from thor.physical import Expression as ex


def _make_mse_expression() -> DynamicExpression:
    def builder(inputs, outputs, stream):
        predictions = ex.input("predictions")
        labels = ex.input("labels")
        diff = predictions - labels
        metric_outputs = ex.outputs({
            "metric": ex.reduce_mean(diff * diff, axis=[0, 1], squeeze=[0], compute_dtype=thor.DataType.fp32),
        })
        equation = ex.compile(metric_outputs, device_num=stream.get_gpu_num())
        return DynamicExpressionBuild(
            equation=equation,
            stamp_inputs=inputs,
            preallocated_outputs=outputs,
        )

    return DynamicExpression(builder)


def _make_metric_vectors(n: int = 3):
    net = thor.Network("test_net_custom_metric")
    preds = thor.Tensor([n], thor.DataType.fp32)
    labs = thor.Tensor([n], thor.DataType.fp32)
    return net, preds, labs


@pytest.mark.cuda
def test_custom_metric_constructs_expression_backed_metric():
    net, preds, labs = _make_metric_vectors()

    m = thor.metrics.CustomMetric(
        network=net,
        expression=_make_mse_expression(),
        predictions=preds,
        labels=labs,
        display_name="MSE",
    )

    assert m is not None
    assert isinstance(m, thor.metrics.CustomMetric)
    assert m.predictions_name == "predictions"
    assert m.labels_name == "labels"
    assert m.metric_name == "metric"
    assert m.display_name == "MSE"


@pytest.mark.cuda
def test_custom_metric_rejects_wrong_arity():
    net, preds, labs = _make_metric_vectors()
    expr = _make_mse_expression()

    with pytest.raises(TypeError):
        thor.metrics.CustomMetric(net, expr, preds)  # missing labels

    with pytest.raises(TypeError):
        thor.metrics.CustomMetric(net, expr, preds, labs, "predictions", "labels", "metric", "Metric", False, 123)


@pytest.mark.cuda
def test_custom_metric_rejects_wrong_types():
    net, preds, labs = _make_metric_vectors()
    expr = _make_mse_expression()

    with pytest.raises(TypeError):
        thor.metrics.CustomMetric("not a network", expr, preds, labs)

    with pytest.raises(TypeError):
        thor.metrics.CustomMetric(net, "not an expression", preds, labs)

    with pytest.raises(TypeError):
        thor.metrics.CustomMetric(net, expr, "not a tensor", labs)

    with pytest.raises(TypeError):
        thor.metrics.CustomMetric(net, expr, preds, "not a tensor")
