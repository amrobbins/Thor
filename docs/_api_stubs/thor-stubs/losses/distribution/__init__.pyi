"""Distributional negative log-likelihood losses."""

import thor
import thor.losses


class GammaNLLLoss(thor.losses.Loss):
    """
    Gamma negative log-likelihood loss for positive mean predictions.

    Predictions are per-element means and labels are targets. Predictions are
    clamped to at least eps for numerical stability. Target-independent constants
    are omitted. The raw loss is:

        log(max(predictions, eps)) + labels / max(predictions, eps)
    """

    def __init__(self, network: thor.Network, predictions: thor.Tensor, labels: thor.Tensor, eps: float = 9.999999974752427e-07, loss_data_type: thor.DataType | None = None, reported_loss_shape: thor.losses.LossShape | None = thor.losses.LossShape.batch, *, loss_weight: float | None = None) -> None:
        """Construct a Gamma negative log-likelihood loss."""

    @property
    def eps(self) -> float: ...

class GaussianNLLLoss(thor.losses.Loss):
    """
    Gaussian negative log-likelihood loss.

    Predictions are means, labels are targets, and variance is the per-element
    variance tensor. Variance is clamped to at least eps for numerical stability.
    The raw loss is:

        0.5 * (log(max(variance, eps)) + (predictions - labels)^2 / max(variance, eps))

    If full is True, the constant 0.5 * log(2*pi) is included.
    """

    def __init__(self, network: thor.Network, predictions: thor.Tensor, labels: thor.Tensor, variance: thor.Tensor, full: bool = False, eps: float = 9.999999974752427e-07, loss_data_type: thor.DataType | None = None, reported_loss_shape: thor.losses.LossShape | None = thor.losses.LossShape.batch, *, loss_weight: float | None = None) -> None:
        """Construct a Gaussian negative log-likelihood loss."""

    @property
    def variance(self) -> thor.Tensor: ...

    @property
    def full(self) -> bool: ...

    @property
    def eps(self) -> float: ...

class PoissonNLLLoss(thor.losses.Loss):
    """
    Poisson negative log-likelihood loss.

    When log_input is True, predictions are log-rates and the raw loss is:

        exp(predictions) - labels * predictions

    When log_input is False, predictions are rates and the raw loss is:

        predictions - labels * log(predictions + eps)

    If full is True, the Stirling approximation term for labels > 1 is included.
    """

    def __init__(self, network: thor.Network, predictions: thor.Tensor, labels: thor.Tensor, log_input: bool = True, full: bool = False, eps: float = 9.99999993922529e-09, loss_data_type: thor.DataType | None = None, reported_loss_shape: thor.losses.LossShape | None = thor.losses.LossShape.batch, *, loss_weight: float | None = None) -> None:
        """Construct a Poisson negative log-likelihood loss."""

    @property
    def log_input(self) -> bool: ...

    @property
    def full(self) -> bool: ...

    @property
    def eps(self) -> float: ...

class TweedieLoss(thor.losses.Loss):
    """
    Tweedie unit deviance loss for positive mean predictions.

    Predictions are per-element means and labels are targets. Predictions are
    clamped to at least eps for numerical stability. power selects the Tweedie
    variance power. Special cases are handled directly for power 0, 1, and 2.
    """

    def __init__(self, network: thor.Network, predictions: thor.Tensor, labels: thor.Tensor, power: float = 1.5, eps: float = 9.999999974752427e-07, loss_data_type: thor.DataType | None = None, reported_loss_shape: thor.losses.LossShape | None = thor.losses.LossShape.batch, *, loss_weight: float | None = None, example_weights: thor.Tensor | None = None) -> None:
        """Construct a Tweedie deviance loss."""

    @property
    def power(self) -> float: ...

    @property
    def eps(self) -> float: ...

__all__: list = ['PoissonNLLLoss', 'GaussianNLLLoss', 'GammaNLLLoss', 'TweedieLoss']
