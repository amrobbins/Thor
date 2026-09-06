from typing import overload

import thor
import thor._thor.metrics
from thor._thor.metrics import LossFormula as LossFormula
import thor.losses
import thor.physical


class Metric:
    def get_metric(self) -> thor.Tensor: ...

    @property
    def aggregation(self) -> thor.MetricAggregation: ...

    def get_feature_output(self) -> thor.Tensor | None: ...

class BinaryAccuracy(Metric):
    """
    Binary Accuracy metric.

    Parameters
    ----------
    network : thor.Network
    predictions : thor.Tensor or thor.RaggedTensor
    labels : thor.Tensor or thor.RaggedTensor

    For ragged inputs, predictions and labels must share the exact logical row
    partition and contain one scalar per active token. Ragged predictions are FP16 or
    FP32. The metric aggregates correct-token and active-token sufficient statistics,
    so partial/unequal batches combine exactly across an epoch.
    """

    @overload
    def __init__(self, network: thor.Network, predictions: thor.Tensor, labels: thor.Tensor) -> None:
        """Construct a Binary Accuracy metric."""

    @overload
    def __init__(self, network: thor.Network, predictions: thor.RaggedTensor, labels: thor.RaggedTensor) -> None:
        """Construct a Binary Accuracy metric over same-partition ragged tokens."""

    @property
    def ragged_predictions(self) -> thor.RaggedTensor | None: ...

    @property
    def ragged_labels(self) -> thor.RaggedTensor | None: ...

class CategoricalAccuracy(Metric):
    """
    Categorical Accuracy metric.

    Parameters
    ----------
    network : thor.Network
    predictions : thor.Tensor or thor.RaggedTensor
    labels : thor.Tensor or thor.RaggedTensor

    For ragged inputs, predictions and labels must share the exact logical row
    partition. Predictions have one trailing class dimension and are FP16 or FP32;
    labels follow the selected per-class or integer class-index contract. The metric
    aggregates correct-token and active-token sufficient statistics exactly across
    partial and unequal batches.
    """

    @overload
    def __init__(self, network: thor.Network, predictions: thor.Tensor, labels: thor.Tensor, label_type: thor.losses.LabelType, num_classes: int | None = None) -> None:
        """Construct a Categorical Accuracy metric."""

    @overload
    def __init__(self, network: thor.Network, predictions: thor.RaggedTensor, labels: thor.RaggedTensor, label_type: thor.losses.LabelType, num_classes: int | None = None) -> None:
        """
        Construct a Categorical Accuracy metric over same-partition ragged tokens.
        """

    @property
    def ragged_predictions(self) -> thor.RaggedTensor | None: ...

    @property
    def ragged_labels(self) -> thor.RaggedTensor | None: ...

class CustomMetric(Metric):
    """
    Expression-backed custom metric.

    Parameters
    ----------
    network : thor.Network
    expression : thor.physical.DynamicExpression
    predictions : thor.Tensor
    labels : thor.Tensor
    aggregation : thor.MetricAggregation
        Declares how this metric's scalar batch result combines across an epoch. A ``RATIO`` expression must also emit
        FP32 scalar outputs named ``thor.METRIC_AGGREGATION_NUMERATOR_NAME`` and
        ``thor.METRIC_AGGREGATION_DENOMINATOR_NAME``. These are internal sufficient statistics and do not become public
        network outputs.
    predictions_name : str, default "predictions"
    labels_name : str, default "labels"
    metric_name : str, default "metric"
    display_name : str, default "Metric"
    uses_batch_validity : bool, default False
        Declares that the expression consumes runtime batch validity. Thor currently supplies it through the reserved
        ``__thor_batch_validity_mask`` FP32 prefix-mask input so invalid tail rows can be excluded from batch-coupled computation.

    Notes
    -----
    ``CustomMetric`` remains a dense-input API. Use the first-class ragged reduction
    or accuracy metric classes for rank-1 ragged values.
    """

    def __init__(self, network: thor.Network, expression: thor.physical.DynamicExpression, predictions: thor.Tensor, labels: thor.Tensor, aggregation: thor.MetricAggregation, predictions_name: str = 'predictions', labels_name: str = 'labels', metric_name: str = 'metric', display_name: str = 'Metric', uses_batch_validity: bool = False) -> None:
        """Construct an expression-backed CustomMetric."""

    @property
    def predictions_name(self) -> str: ...

    @property
    def labels_name(self) -> str: ...

    @property
    def metric_name(self) -> str: ...

    @property
    def display_name(self) -> str: ...

    @property
    def uses_batch_validity(self) -> bool: ...

class Mean(Metric):
    @overload
    def __init__(self, network: thor.Network, values: thor.Tensor) -> None:
        """Construct a Mean metric over dense or ragged values."""

    @overload
    def __init__(self, network: thor.Network, values: thor.RaggedTensor) -> None: ...

    @property
    def values(self) -> thor.Tensor: ...

    @property
    def ragged_values(self) -> thor.RaggedTensor | None: ...

class Sum(Metric):
    @overload
    def __init__(self, network: thor.Network, values: thor.Tensor) -> None:
        """Construct a Sum metric over dense or ragged values."""

    @overload
    def __init__(self, network: thor.Network, values: thor.RaggedTensor) -> None: ...

    @property
    def values(self) -> thor.Tensor: ...

    @property
    def ragged_values(self) -> thor.RaggedTensor | None: ...

class Min(Metric):
    @overload
    def __init__(self, network: thor.Network, values: thor.Tensor) -> None:
        """Construct a Min metric over dense or ragged values."""

    @overload
    def __init__(self, network: thor.Network, values: thor.RaggedTensor) -> None: ...

    @property
    def values(self) -> thor.Tensor: ...

    @property
    def ragged_values(self) -> thor.RaggedTensor | None: ...

class Max(Metric):
    @overload
    def __init__(self, network: thor.Network, values: thor.Tensor) -> None:
        """Construct a Max metric over dense or ragged values."""

    @overload
    def __init__(self, network: thor.Network, values: thor.RaggedTensor) -> None: ...

    @property
    def values(self) -> thor.Tensor: ...

    @property
    def ragged_values(self) -> thor.RaggedTensor | None: ...

class WeightedMean(Metric):
    @overload
    def __init__(self, network: thor.Network, values: thor.Tensor, weights: thor.Tensor) -> None:
        """Construct a WeightedMean metric over dense values and weights tensors."""

    @overload
    def __init__(self, network: thor.Network, values: thor.RaggedTensor, weights: thor.RaggedTensor) -> None:
        """
        Construct a WeightedMean metric over same-partition ragged values and weights.
        """

    @property
    def values(self) -> thor.Tensor: ...

    @property
    def weights(self) -> thor.Tensor: ...

    @property
    def ragged_values(self) -> thor.RaggedTensor | None: ...

    @property
    def ragged_weights(self) -> thor.RaggedTensor | None: ...

mean_squared_error: thor._thor.metrics.LossFormula = thor._thor.metrics.LossFormula.mean_squared_error

mean_absolute_error: thor._thor.metrics.LossFormula = thor._thor.metrics.LossFormula.mean_absolute_error

mean_absolute_percentage_error: thor._thor.metrics.LossFormula = thor._thor.metrics.LossFormula.mean_absolute_percentage_error

class LossMetric(Metric):
    """
    Forward-only dense loss-formula metric.

    ``LossMetric`` remains a dense-input API. Rank-1 ragged metric support is
    provided by the first-class reduction and accuracy metric classes.
    """

    def __init__(self, network: thor.Network, predictions: thor.Tensor, labels: thor.Tensor, formula: thor._thor.metrics.LossFormula = thor._thor.metrics.LossFormula.mean_squared_error, epsilon: float | None = None, max_magnitude: float | None = None, display_name: str | None = None) -> None:
        """Track a loss formula as a forward-only metric."""

    @property
    def formula(self) -> thor._thor.metrics.LossFormula: ...

    @property
    def predictions(self) -> thor.Tensor: ...

    @property
    def labels(self) -> thor.Tensor: ...

    @property
    def display_name(self) -> str: ...
