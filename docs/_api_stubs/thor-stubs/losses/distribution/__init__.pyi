"""Distributional likelihood and deviance losses."""

import thor
import thor.losses


class GammaNLLLoss(thor.losses.Loss):
    """
    Gamma negative log-likelihood loss in mean/dispersion parameterization.

    Without dispersion, this preserves Thor's legacy unit-dispersion (shape=1)
    Gamma/exponential loss:

        log(mean) + labels / mean

    When dispersion is supplied, Thor uses Var(Y) = dispersion * mean^2, with
    concentration = 1 / dispersion and scale = mean * dispersion, and evaluates
    the full per-element Gamma NLL. log_mean=True and log_dispersion=True allow
    unconstrained network heads to supply log-parameters directly.

    example_weights may be a [1] per-example weight tensor or match predictions for
    elementwise weighting. Weights scale the raw loss and all learned-parameter
    gradients before loss-shape reduction.
    """

    def __init__(self, network: thor.Network, predictions: thor.Tensor, labels: thor.Tensor, eps: float = 9.999999974752427e-07, loss_data_type: thor.DataType | None = None, reported_loss_shape: thor.losses.LossShape | None = thor.losses.LossShape.batch, *, loss_weight: float | None = None, dispersion: thor.Tensor | None = None, log_mean: bool | None = False, log_dispersion: bool = False, example_weights: thor.Tensor | None = None) -> None:
        """Construct a Gamma negative log-likelihood loss."""

    @property
    def dispersion(self) -> thor.Tensor | None: ...

    @property
    def log_mean(self) -> bool: ...

    @property
    def log_dispersion(self) -> bool: ...

    @property
    def eps(self) -> float: ...

class GaussianNLLLoss(thor.losses.Loss):
    """
    Gaussian negative log-likelihood loss.

    predictions are means and labels are targets. By default variance contains a
    positive per-element variance and is clamped to at least eps. With
    log_variance=True, variance contains log(variance), so an unconstrained network
    head can be trained directly and the raw loss is evaluated as:

        0.5 * (log_variance + (predictions - labels)^2 * exp(-log_variance))

    If full is True, the constant 0.5 * log(2*pi) is included.

    example_weights may be a [1] per-example weight tensor or match predictions for
    elementwise weighting. Weights multiply the raw loss and both learned-parameter
    gradients before loss-shape reduction.
    """

    def __init__(self, network: thor.Network, predictions: thor.Tensor, labels: thor.Tensor, variance: thor.Tensor, full: bool = False, eps: float = 9.999999974752427e-07, loss_data_type: thor.DataType | None = None, reported_loss_shape: thor.losses.LossShape | None = thor.losses.LossShape.batch, *, loss_weight: float | None = None, log_variance: bool | None = False, example_weights: thor.Tensor | None = None) -> None:
        """Construct a Gaussian negative log-likelihood loss."""

    @property
    def variance(self) -> thor.Tensor: ...

    @property
    def log_variance(self) -> bool: ...

    @property
    def full(self) -> bool: ...

    @property
    def eps(self) -> float: ...

class LaplaceNLLLoss(thor.losses.Loss):
    """
    Laplace negative log-likelihood using location and scale parameters.

    For location m and scale b > 0, the per-element negative log-likelihood is:

        log(2 * b) + abs(target - m) / b

    By default scale contains log(b), allowing an unconstrained network head. Set
    log_scale=False to supply positive scale directly; direct scale is floored by
    eps for numerical stability.

    example_weights may be [1] for per-example weighting or may match location for
    elementwise weighting. Weights scale the raw NLL and both learned-parameter
    gradients before loss-shape reduction.
    """

    def __init__(self, network: thor.Network, location: thor.Tensor, scale: thor.Tensor, labels: thor.Tensor, log_scale: bool = True, eps: float = 9.99999993922529e-09, loss_data_type: thor.DataType | None = None, reported_loss_shape: thor.losses.LossShape | None = thor.losses.LossShape.batch, *, loss_weight: float | None = None, example_weights: thor.Tensor | None = None) -> None:
        """Construct a Laplace negative log-likelihood loss."""

    @property
    def location(self) -> thor.Tensor: ...

    @property
    def scale(self) -> thor.Tensor: ...

    @property
    def log_scale(self) -> bool: ...

    @property
    def eps(self) -> float: ...

class NegativeBinomialNLLLoss(thor.losses.Loss):
    """
    Negative Binomial negative log-likelihood using the NB2 mean/dispersion parameterization.

    The distribution is parameterized by mean mu and dispersion alpha:

        Var(Y) = mu + alpha * mu^2

    Equivalently, the Negative Binomial concentration is r = 1 / alpha. By default
    mean and dispersion tensors contain log(mu) and log(alpha), allowing both model
    heads to be unconstrained. Set log_mean=False and/or log_dispersion=False when
    supplying positive parameters directly; direct parameters are floored by eps.

    labels must contain non-negative counts. Floating labels are accepted for
    training pipelines that represent counts in fp16/fp32.

    example_weights may be [1] for per-example weighting or may match mean for
    elementwise weighting. Weights scale the raw NLL and both learned-parameter
    gradients before loss-shape reduction.
    """

    def __init__(self, network: thor.Network, mean: thor.Tensor, dispersion: thor.Tensor, labels: thor.Tensor, log_mean: bool = True, log_dispersion: bool = True, eps: float = 9.99999993922529e-09, loss_data_type: thor.DataType | None = None, reported_loss_shape: thor.losses.LossShape | None = thor.losses.LossShape.batch, *, loss_weight: float | None = None, example_weights: thor.Tensor | None = None) -> None:
        """Construct a Negative Binomial negative log-likelihood loss."""

    @property
    def mean(self) -> thor.Tensor: ...

    @property
    def dispersion(self) -> thor.Tensor: ...

    @property
    def log_mean(self) -> bool: ...

    @property
    def log_dispersion(self) -> bool: ...

    @property
    def eps(self) -> float: ...

class StudentTNLLLoss(thor.losses.Loss):
    """
    Student-t negative log-likelihood using location, log-scale, and fixed or learned degrees of freedom.

    For location m, scale s > 0, degrees of freedom nu > 0, and standardized
    residual z = (target - m) / s, the per-element negative log-likelihood is:

        log(s) + lgamma(nu / 2) - lgamma((nu + 1) / 2)
        + 0.5 * log(nu * pi)
        + 0.5 * (nu + 1) * log1p(z^2 / nu)

    log_scale always contains log(s), allowing an unconstrained scale head. Supply
    `degrees_of_freedom` for fixed nu. Alternatively supply
    `learned_log_degrees_of_freedom`, whose tensor receives an analytical gradient.
    With the default `minimum_degrees_of_freedom=0.0`, it contains log(nu). When a
    positive minimum m is supplied, learned nu is parameterized as
    `nu = m + exp(learned_log_degrees_of_freedom)`, so the tensor contains the log
    of the degrees-of-freedom excess above the floor. If neither fixed nor learned
    degrees of freedom is supplied, fixed nu defaults to 3.0. Fixed nu must be
    greater than the configured minimum.

    example_weights may be [1] for per-example weighting or may match location for
    elementwise weighting. Weights scale the raw NLL and all learned-parameter
    gradients before loss-shape reduction.
    """

    def __init__(self, network: thor.Network, location: thor.Tensor, log_scale: thor.Tensor, labels: thor.Tensor, degrees_of_freedom: float | None = None, loss_data_type: thor.DataType | None = None, reported_loss_shape: thor.losses.LossShape | None = thor.losses.LossShape.batch, *, minimum_degrees_of_freedom: float = 0.0, learned_log_degrees_of_freedom: thor.Tensor | None = None, loss_weight: float | None = None, example_weights: thor.Tensor | None = None) -> None:
        """Construct a Student-t negative log-likelihood loss."""

    @property
    def location(self) -> thor.Tensor: ...

    @property
    def log_scale(self) -> thor.Tensor: ...

    @property
    def degrees_of_freedom(self) -> float | None: ...

    @property
    def learned_log_degrees_of_freedom(self) -> thor.Tensor | None: ...

    @property
    def minimum_degrees_of_freedom(self) -> float: ...

class PoissonNLLLoss(thor.losses.Loss):
    """
    Poisson negative log-likelihood loss.

    When log_input is True, predictions are log-rates and the raw loss is:

        exp(predictions) - labels * predictions

    When log_input is False, predictions are rates and the raw loss is:

        predictions - labels * log(predictions + eps)

    If full is True, the Stirling approximation term for labels > 1 is included.

    example_weights may be a [1] per-example weight tensor or match predictions
    for elementwise weighting. Weights multiply both the raw loss and the
    prediction gradient before loss-shape reduction.
    """

    def __init__(self, network: thor.Network, predictions: thor.Tensor, labels: thor.Tensor, log_input: bool = True, full: bool = False, eps: float = 9.99999993922529e-09, loss_data_type: thor.DataType | None = None, reported_loss_shape: thor.losses.LossShape | None = thor.losses.LossShape.batch, *, loss_weight: float | None = None, example_weights: thor.Tensor | None = None) -> None:
        """Construct a Poisson negative log-likelihood loss."""

    @property
    def log_input(self) -> bool: ...

    @property
    def full(self) -> bool: ...

    @property
    def eps(self) -> float: ...

class TweedieLoss(thor.losses.Loss):
    """
    Tweedie unit-deviance loss for positive mean predictions.

    This is a Tweedie unit deviance objective, not a full normalized Tweedie
    negative log-likelihood. Predictions are per-element means and labels are
    targets. Predictions are clamped to at least eps for numerical stability.
    power selects the Tweedie variance power; powers 0, 1, and 2 use direct special
    cases.
    """

    def __init__(self, network: thor.Network, predictions: thor.Tensor, labels: thor.Tensor, power: float = 1.5, eps: float = 9.999999974752427e-07, loss_data_type: thor.DataType | None = None, reported_loss_shape: thor.losses.LossShape | None = thor.losses.LossShape.batch, *, loss_weight: float | None = None, example_weights: thor.Tensor | None = None) -> None:
        """Construct a Tweedie unit-deviance loss."""

    @property
    def power(self) -> float: ...

    @property
    def eps(self) -> float: ...

__all__: list = ['PoissonNLLLoss', 'GaussianNLLLoss', 'NegativeBinomialNLLLoss', 'LaplaceNLLLoss', 'StudentTNLLLoss', 'GammaNLLLoss', 'TweedieLoss']
