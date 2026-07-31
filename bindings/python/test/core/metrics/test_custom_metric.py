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




def _make_ratio_expression(*, include_numerator: bool = True, include_denominator: bool = True) -> DynamicExpression:
    def builder(inputs, outputs, stream):
        predictions = ex.input("predictions")
        labels = ex.input("labels")
        numerator = ex.reduce_sum(predictions * labels, axis=[0, 1], squeeze=[0], compute_dtype=thor.DataType.fp32)
        denominator = ex.reduce_sum(labels, axis=[0, 1], squeeze=[0], compute_dtype=thor.DataType.fp32)
        named_outputs = {"metric": numerator / denominator}
        if include_numerator:
            named_outputs[thor.METRIC_AGGREGATION_NUMERATOR_NAME] = numerator
        if include_denominator:
            named_outputs[thor.METRIC_AGGREGATION_DENOMINATOR_NAME] = denominator
        metric_outputs = ex.outputs(named_outputs)
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
        aggregation=thor.MetricAggregation.MEAN_BY_EXAMPLE,
        display_name="MSE",
    )

    assert m is not None
    assert isinstance(m, thor.metrics.CustomMetric)
    assert m.predictions_name == "predictions"
    assert m.labels_name == "labels"
    assert m.metric_name == "metric"
    assert m.display_name == "MSE"
    assert m.aggregation is thor.MetricAggregation.MEAN_BY_EXAMPLE
    assert m.uses_batch_validity is False
    assert not hasattr(m, "uses_batch_validity_mask")
    assert not hasattr(m, "supports_partial_batches")


@pytest.mark.cuda
def test_custom_metric_rejects_wrong_arity():
    net, preds, labs = _make_metric_vectors()
    expr = _make_mse_expression()

    with pytest.raises(TypeError):
        thor.metrics.CustomMetric(net, expr, preds, labs)  # missing aggregation

    with pytest.raises(TypeError):
        thor.metrics.CustomMetric(
            net, expr, preds, labs, thor.MetricAggregation.MEAN_BY_EXAMPLE,
            "predictions", "labels", "metric", "Metric", False, 123
        )


@pytest.mark.cuda
def test_custom_metric_rejects_wrong_types():
    net, preds, labs = _make_metric_vectors()
    expr = _make_mse_expression()

    with pytest.raises(TypeError):
        thor.metrics.CustomMetric("not a network", expr, preds, labs, thor.MetricAggregation.MEAN_BY_EXAMPLE)

    with pytest.raises(TypeError):
        thor.metrics.CustomMetric(net, "not an expression", preds, labs, thor.MetricAggregation.MEAN_BY_EXAMPLE)

    with pytest.raises(TypeError):
        thor.metrics.CustomMetric(net, expr, "not a tensor", labs, thor.MetricAggregation.MEAN_BY_EXAMPLE)

    with pytest.raises(TypeError):
        thor.metrics.CustomMetric(net, expr, preds, "not a tensor", thor.MetricAggregation.MEAN_BY_EXAMPLE)


@pytest.mark.cuda
def test_custom_metric_rejects_deprecated_partial_batch_keyword():
    net, preds, labs = _make_metric_vectors()
    with pytest.raises(TypeError):
        thor.metrics.CustomMetric(
            net,
            _make_mse_expression(),
            preds,
            labs,
            thor.MetricAggregation.MEAN_BY_EXAMPLE,
            supports_partial_batches=False,
        )


def test_metric_aggregation_enum_is_public():
    assert thor.MetricAggregation.MEAN_BY_EXAMPLE.name == "MEAN_BY_EXAMPLE"
    assert thor.MetricAggregation.SUM.name == "SUM"
    assert thor.MetricAggregation.MIN.name == "MIN"
    assert thor.MetricAggregation.MAX.name == "MAX"
    assert thor.MetricAggregation.RATIO.name == "RATIO"


@pytest.mark.cuda
def test_custom_ratio_metric_requires_and_accepts_sufficient_statistics():
    net, preds, labs = _make_metric_vectors()
    metric = thor.metrics.CustomMetric(
        network=net,
        expression=_make_ratio_expression(),
        predictions=preds,
        labels=labs,
        aggregation=thor.MetricAggregation.RATIO,
    )
    assert metric.metric_name == "metric"
    assert metric.aggregation is thor.MetricAggregation.RATIO

    net_missing, preds_missing, labs_missing = _make_metric_vectors()
    with pytest.raises(RuntimeError):
        thor.metrics.CustomMetric(
            network=net_missing,
            expression=_make_ratio_expression(include_denominator=False),
            predictions=preds_missing,
            labels=labs_missing,
            aggregation=thor.MetricAggregation.RATIO,
        )


def test_ratio_statistic_names_are_public_constants():
    assert thor.METRIC_AGGREGATION_NUMERATOR_NAME == "__thor_metric_aggregation_numerator"
    assert thor.METRIC_AGGREGATION_DENOMINATOR_NAME == "__thor_metric_aggregation_denominator"
