import thor


class Optimizer:
    pass

class Sgd(Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer.

    This optimizer updates a layer's trainable parameters (weights and, if present, biases)
    using classic SGD with optional momentum, optional Nesterov momentum, and optional learning
    rate decay.

    Parameters
    ----------
    initial_learning_rate : float, default 0.01
        Base learning rate used for the update step.
    decay : float, default 0.0
        Per-epoch learning rate decay factor. When decay is non-zero, the effective learning rate
        is reduced each epoch, e.g. ``lr <- lr * (1 - decay)`` each epoch.
    momentum : float, default 0.0
        Momentum coefficient in ``[0, 1]``. When non-zero, the optimizer maintains a
        velocity buffer for each parameter and performs momentum updates.
    nesterov_momentum : bool, default False
        If True, use Nesterov momentum (lookahead / projected parameters) for training-time
        forward passes.
    network : thor.Network, default None
        When network is passed in, then this optimizer will be set as the default optimizer in
        the network and attached to all layers that do not have a layer specific optimizer
        already attached, at network stamping time. You would not pass network here when you
        want this optimizer to be specific to one or more layers, but not applied to the others
        by default.


    Notes
    -----
    **Momentum.**
    With momentum enabled, SGD maintains a velocity buffer ``u`` per parameter:

    - ``u <- momentum * update - learning_rate * gradient``

    and applies the parameter update using the velocity.

    **Nesterov momentum.**
    When Nesterov is enabled, training-time forward passes use a *projected* (lookahead)
    parameter:

    - ``p = w + mu * u``

    where ``w`` is the current parameter and ``u`` is its velocity buffer. Backprop computes
    gradients at ``p``. Inference-time forward passes use the real (non-projected) parameters.

    **Bias parameters.**
    If the underlying layer has biases, SGD maintains separate velocity/gradient buffers
    for biases as well.

    Examples
    --------
    Basic usage::

        from thor.optimizers import Sgd

        opt = Sgd(initial_learning_rate=0.1)

    With momentum::

        opt = Sgd(initial_learning_rate=0.1, momentum=0.9)

    With Nesterov momentum and decay::

        opt = Sgd(initial_learning_rate=0.1, decay=0.1, momentum=0.9, nesterov_momentum=True)

    See Also
    --------
    Adam : Adaptive Moment Estimation optimizer.
    RMSprop : RMSprop optimizer.
    """

    def __init__(self, initial_learning_rate: float = 0.009999999776482582, decay: float = 0.0, momentum: float = 0.0, nesterov_momentum: bool = False, network: thor.Network | None = None) -> None:
        """Construct an SGD optimizer."""

class ASGD(Optimizer):
    """
    ASGD optimizer (Averaged Stochastic Gradient Descent).

    ASGD applies a decayed SGD update and maintains a separate running average of
    the dense parameter tensor. The averaged tensor is optimizer state and is saved
    when optimizer state saving is enabled.

    Parameters
    ----------
    alpha : float, default 0.01
        Base learning rate.
    lambd : float, default 1e-4
        ASGD decay coefficient used in the decayed step size and multiplicative
        weight shrinkage.
    power : float, default 0.75
        Exponent for the decayed step size schedule.
    t0 : float, default 1e6
        First update step at which the averaged parameter tensor starts tracking
        the running average.
    weight_decay : float, default 0.0
        Additional coupled weight decay added to the gradient update vector.
    network : thor.Network, default None
        When network is passed in, this optimizer is set as the network default optimizer.

    Notes
    -----
    Thor's dense ASGD update is:

    - ``eta <- alpha / (1 + lambd * alpha * t) ** power``
    - ``w <- (1 - lambd * eta) * w - eta * (g + weight_decay * w)``
    - before ``t0``: ``averaged_weights`` is unchanged
    - from ``t0`` onward: ``averaged_weights`` tracks the running average of ``w``

    where ``g`` is Thor's batch/loss-scale normalized gradient and ``t`` is the
    optimizer update count. Sparse-row updates are intentionally not supported
    because the averaged parameter tensor is full-weight state and untouched rows
    would otherwise become stale.

    Examples
    --------
    Basic usage::

        from thor.optimizers import ASGD

        opt = ASGD(alpha=0.01, lambd=1e-4, power=0.75, t0=1000)

    See Also
    --------
    Sgd : Stochastic gradient descent with optional momentum.
    """

    def __init__(self, alpha: float = 0.009999999776482582, lambd: float = 9.999999747378752e-05, power: float = 0.75, t0: float = 1000000.0, weight_decay: float = 0.0, network: thor.Network | None = None) -> None:
        """Construct an ASGD optimizer."""

class Adam(Optimizer):
    """
    ADAM optimizer (Adaptive Moment Estimation).

    Adam is an adaptive learning-rate optimizer that combines ideas from momentum and
    RMSprop by maintaining exponentially decayed moving averages of the gradient
    (first moment) and of the squared gradient (second moment). These moment estimates
    are bias-corrected and used to scale the parameter update for each parameter individually.

    Parameters
    ----------
    alpha : float, default 0.001
        Base learning rate (step size).
    beta1 : float, default 0.9
        Exponential decay rate for the first-moment estimate (mean of gradients).
        Typical values are close to 0.9.
    beta2 : float, default 0.999
        Exponential decay rate for the second-moment estimate (mean of squared gradients).
        Typical values are close to 0.999.
    epsilon : float, default fp32: 1e-7, fp16: 1e-4
        Small constant added to the denominator for numerical stability.
    amsgrad : bool, default False
        If True, use AMSGrad by tracking the maximum second-moment estimate and using
        that maximum in Adam's denominator.
    network : thor.Network, default None
        When network is passed in, then this optimizer will be set as the default optimizer in
        the network and attached to all layers that do not have a layer specific optimizer
        already attached, at network stamping time. You would not pass network here when you
        want this optimizer to be specific to one or more layers, but not applied to the others
        by default.

    Notes
    -----
    Adam maintains, for each parameter, a first-moment buffer ``m`` and a second-moment
    buffer ``v``. With ``amsgrad=True``, it also maintains ``vhat``, the elementwise
    maximum of the second-moment estimate:

    - ``m <- beta1 * m + (1 - beta1) * g``
    - ``v <- beta2 * v + (1 - beta2) * (g * g)``
    - ``vhat <- max(vhat, v)`` when ``amsgrad=True``

    It then uses bias-corrected moments:

    - ``m_hat = m / (1 - beta1^t)``
    - ``v_hat = v / (1 - beta2^t)``

    and applies the update:

    - ``w <- w - alpha * m_hat / (sqrt(v_hat) + epsilon)``

    For ``amsgrad=True``, the denominator uses the tracked maximum second moment.

    where ``t`` is the update step count.


    Examples
    --------
    Basic usage::

        from thor.optimizers import Adam

        opt = Adam(network)

    Custom hyperparameters::

        opt = Adam(network, alpha=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-7)

    AMSGrad::

        opt = Adam(alpha=1e-3, amsgrad=True)

    See Also
    --------
    Sgd : Stochastic Gradient Descent optimizer (optionally with momentum / Nesterov).
    RMSprop : RMSprop optimizer.
    """

    def __init__(self, alpha: float = 0.0010000000474974513, beta1: float = 0.8999999761581421, beta2: float = 0.9990000128746033, epsilon: float = 1.0000000116860974e-07, amsgrad: bool = False, network: thor.Network | None = None) -> None:
        """Construct an ADAM optimizer."""

class AdamW(Optimizer):
    """
    AdamW optimizer.

    AdamW is Adam with decoupled weight decay. It maintains first- and second-moment
    buffers like Adam, but applies weight decay directly to the parameter rather than
    adding an L2 penalty into the gradient.

    Parameters
    ----------
    alpha : float, default 0.001
        Base learning rate.
    beta1 : float, default 0.9
        Exponential decay rate for the first-moment estimate.
    beta2 : float, default 0.999
        Exponential decay rate for the second-moment estimate.
    epsilon : float, default 1e-7
        Small constant added to the denominator for numerical stability.
    weight_decay : float, default 0.01
        Decoupled weight decay coefficient. Set to 0.0 for Adam-equivalent behavior.
    network : thor.Network, default None
        When network is passed in, this optimizer is set as the network default optimizer.

    Notes
    -----
    The dense update is:

    - ``m <- beta1 * m + (1 - beta1) * g``
    - ``v <- beta2 * v + (1 - beta2) * (g * g)``
    - ``w <- w - alpha * weight_decay * w - alpha_t * m / (sqrt(v) + epsilon)``

    where ``alpha_t`` is the Adam bias-corrected learning rate.

    For sparse-row embedding updates, the same expression is applied to the touched rows.

    Examples
    --------
    Basic usage::

        from thor.optimizers import AdamW

        opt = AdamW(alpha=1e-3, weight_decay=0.01)

    See Also
    --------
    Adam : Adaptive Moment Estimation optimizer without decoupled weight decay.
    Sgd : Stochastic Gradient Descent optimizer.
    """

    def __init__(self, alpha: float = 0.0010000000474974513, beta1: float = 0.8999999761581421, beta2: float = 0.9990000128746033, epsilon: float = 1.0000000116860974e-07, weight_decay: float = 0.009999999776482582, network: thor.Network | None = None) -> None:
        """Construct an AdamW optimizer."""

class Adamax(Optimizer):
    """
    Adamax optimizer.

    Adamax is the infinity-norm variant of Adam. It maintains a first-moment buffer
    ``m`` and an exponentially decayed infinity-norm buffer ``u`` for each parameter.

    Parameters
    ----------
    alpha : float, default 0.002
        Base learning rate.
    beta1 : float, default 0.9
        Exponential decay rate for the first-moment estimate. Must be in ``[0, 1)``.
    beta2 : float, default 0.999
        Exponential decay rate for the infinity-norm estimate. Must be in ``[0, 1)``.
    epsilon : float, default 1e-7
        Small constant added to the denominator for numerical stability.
    network : thor.Network, default None
        When network is passed in, this optimizer is set as the network default optimizer.

    Notes
    -----
    The dense update is:

    - ``m <- beta1 * m + (1 - beta1) * g``
    - ``u <- max(beta2 * u, abs(g))``
    - ``w <- w - (alpha / (1 - beta1**t)) * m / (u + epsilon)``

    where ``g`` is Thor's batch/loss-scale normalized gradient.

    For sparse-row embedding updates, the same expression is applied to the touched rows.

    Examples
    --------
    Basic usage::

        from thor.optimizers import Adamax

        opt = Adamax(alpha=0.002, beta1=0.9, beta2=0.999, epsilon=1e-7)

    See Also
    --------
    Adam : Adaptive Moment Estimation optimizer.
    AdamW : Adam with decoupled weight decay.
    RMSprop : RMSprop optimizer.
    Sgd : Stochastic Gradient Descent optimizer.
    """

    def __init__(self, alpha: float = 0.0020000000949949026, beta1: float = 0.8999999761581421, beta2: float = 0.9990000128746033, epsilon: float = 1.0000000116860974e-07, network: thor.Network | None = None) -> None:
        """Construct an Adamax optimizer."""

class NAdam(Optimizer):
    """
    NAdam optimizer.

    NAdam combines Adam's adaptive moments with a Nesterov-style first-moment lookahead.
    It maintains first-moment ``m`` and second-moment ``v`` buffers for each parameter.

    Parameters
    ----------
    alpha : float, default 0.002
        Base learning rate.
    beta1 : float, default 0.9
        Exponential decay rate for the first-moment estimate. Must be in ``[0, 1)``.
    beta2 : float, default 0.999
        Exponential decay rate for the second-moment estimate. Must be in ``[0, 1)``.
    epsilon : float, default 1e-7
        Small constant added to the denominator for numerical stability.
    network : thor.Network, default None
        When network is passed in, this optimizer is set as the network default optimizer.

    Notes
    -----
    The dense update is:

    - ``m <- beta1 * m + (1 - beta1) * g``
    - ``v <- beta2 * v + (1 - beta2) * g * g``
    - ``w <- w - (m_scale * m + gradient_scale * g) / (sqrt(v) + epsilon)``

    where ``m_scale`` and ``gradient_scale`` include the learning rate and NAdam bias-correction terms,
    and ``g`` is Thor's batch/loss-scale normalized gradient.

    For sparse-row embedding updates, the same expression is applied to the touched rows.

    Examples
    --------
    Basic usage::

        from thor.optimizers import NAdam

        opt = NAdam(alpha=0.002, beta1=0.9, beta2=0.999, epsilon=1e-7)

    See Also
    --------
    Adam : Adaptive Moment Estimation optimizer.
    Adamax : Infinity-norm variant of Adam.
    AdamW : Adam with decoupled weight decay.
    RMSprop : RMSprop optimizer.
    """

    def __init__(self, alpha: float = 0.0020000000949949026, beta1: float = 0.8999999761581421, beta2: float = 0.9990000128746033, epsilon: float = 1.0000000116860974e-07, network: thor.Network | None = None) -> None:
        """Construct a NAdam optimizer."""

class RAdam(Optimizer):
    """
    RAdam optimizer.

    RAdam is Adam with a variance rectification term for the adaptive denominator. During early
    steps where the estimated variance is unreliable, it falls back to the unrectified
    bias-corrected first-moment step.

    Parameters
    ----------
    alpha : float, default 0.001
        Base learning rate.
    beta1 : float, default 0.9
        Exponential decay rate for the first-moment estimate. Must be in ``[0, 1)``.
    beta2 : float, default 0.999
        Exponential decay rate for the second-moment estimate. Must be in ``[0, 1)``.
    epsilon : float, default 1e-7
        Small constant added to the denominator for numerical stability.
    network : thor.Network, default None
        When network is passed in, this optimizer is set as the network default optimizer.

    Notes
    -----
    The dense update is:

    - ``m <- beta1 * m + (1 - beta1) * g``
    - ``v <- beta2 * v + (1 - beta2) * g * g``
    - compute ``rho_t`` from ``beta2`` and the current step
    - if ``rho_t >= 5``, use ``rectified_alpha_t * m / (sqrt(v) + epsilon)``
    - otherwise use ``unrectified_alpha_t * m``

    where the runtime step sizes include the learning rate and bias-correction terms,
    and ``g`` is Thor's batch/loss-scale normalized gradient.

    For sparse-row embedding updates, the same expression is applied to the touched rows.

    Examples
    --------
    Basic usage::

        from thor.optimizers import RAdam

        opt = RAdam(alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-7)

    See Also
    --------
    Adam : Adaptive Moment Estimation optimizer.
    NAdam : Adam with Nesterov-style first-moment lookahead.
    AdamW : Adam with decoupled weight decay.
    """

    def __init__(self, alpha: float = 0.0010000000474974513, beta1: float = 0.8999999761581421, beta2: float = 0.9990000128746033, epsilon: float = 1.0000000116860974e-07, network: thor.Network | None = None) -> None:
        """Construct a RAdam optimizer."""

class Adagrad(Optimizer):
    """
    Adagrad optimizer.

    Adagrad adapts the learning rate for each parameter using a running sum of
    squared gradients. It is often useful for sparse features because frequently
    updated parameters receive smaller effective steps over time.

    Parameters
    ----------
    alpha : float, default 0.01
        Base learning rate.
    epsilon : float, default 1e-7
        Small constant added to the denominator for numerical stability.
    network : thor.Network, default None
        When network is passed in, this optimizer is set as the network default optimizer.

    Notes
    -----
    The dense update is:

    - ``accumulator <- accumulator + g * g``
    - ``w <- w - alpha * g / (sqrt(accumulator) + epsilon)``

    where ``g`` is Thor's batch/loss-scale normalized gradient.

    For sparse-row embedding updates, the same expression is applied to the touched rows.

    Examples
    --------
    Basic usage::

        from thor.optimizers import Adagrad

        opt = Adagrad(alpha=0.01, epsilon=1e-7)

    See Also
    --------
    Adam : Adaptive Moment Estimation optimizer.
    AdamW : Adam with decoupled weight decay.
    Sgd : Stochastic Gradient Descent optimizer.
    """

    def __init__(self, alpha: float = 0.009999999776482582, epsilon: float = 1.0000000116860974e-07, network: thor.Network | None = None) -> None:
        """Construct an Adagrad optimizer."""

class Adadelta(Optimizer):
    """
    Adadelta optimizer.

    Adadelta adapts each parameter's step size using exponentially decayed moving
    averages of squared gradients and squared updates. Unlike Adagrad, the update
    history keeps the effective learning rate from shrinking monotonically forever.

    Parameters
    ----------
    alpha : float, default 1.0
        Global learning-rate multiplier applied to the Adadelta update.
    rho : float, default 0.95
        Exponential decay rate for the running averages. Must be in ``[0, 1)``.
    epsilon : float, default 1e-7
        Small constant added inside the root-mean-square terms for numerical stability.
    network : thor.Network, default None
        When network is passed in, this optimizer is set as the network default optimizer.

    Notes
    -----
    The dense update is:

    - ``gradient_square_average <- rho * gradient_square_average + (1 - rho) * g * g``
    - ``update <- sqrt(update_square_average + epsilon) / sqrt(gradient_square_average + epsilon) * g``
    - ``update_square_average <- rho * update_square_average + (1 - rho) * update * update``
    - ``w <- w - alpha * update``

    where ``g`` is Thor's batch/loss-scale normalized gradient.

    For sparse-row embedding updates, the same expression is applied to the touched rows.

    Examples
    --------
    Basic usage::

        from thor.optimizers import Adadelta

        opt = Adadelta(alpha=1.0, rho=0.95, epsilon=1e-7)

    See Also
    --------
    Adagrad : Accumulated-gradient adaptive optimizer.
    RMSprop : Moving-average adaptive optimizer.
    Adam : Adaptive Moment Estimation optimizer.
    """

    def __init__(self, alpha: float = 1.0, rho: float = 0.949999988079071, epsilon: float = 1.0000000116860974e-07, network: thor.Network | None = None) -> None:
        """Construct an Adadelta optimizer."""

class Adafactor(Optimizer):
    """
    Adafactor optimizer.

    Adafactor uses an exponential moving average of squared gradients to normalize
    updates. For rank-2 and higher dense tensors, Thor uses Adafactor's memory-saving
    factored second-moment estimate over the final two dimensions. Rank-1 tensors and
    sparse-row embedding updates use the unfactored second-moment fallback.

    Parameters
    ----------
    alpha : float, default 0.001
        Learning rate.
    beta2 : float, default 0.999
        Exponential decay rate for the second-moment estimate. Must be in ``[0, 1)``.
    epsilon : float, default 1e-30
        Small constant added for numerical stability.
    weight_decay : float, default 0.0
        Decoupled weight-decay coefficient.
    factor_second_moment : bool, default True
        Use factored second-moment state for rank-2 and higher dense tensors.
    network : thor.Network, default None
        When network is passed in, this optimizer is set as the network default optimizer.
    """

    def __init__(self, alpha: float = 0.0010000000474974513, beta2: float = 0.9990000128746033, epsilon: float = 1.0000000031710769e-30, weight_decay: float = 0.0, factor_second_moment: bool = True, network: thor.Network | None = None) -> None:
        """Construct an Adafactor optimizer."""

class RMSprop(Optimizer):
    """
    RMSprop optimizer.

    RMSprop adapts each parameter's step size using an exponentially decayed moving
    average of squared gradients. Compared with Adagrad, the exponential decay keeps
    the effective learning rate from shrinking monotonically forever.

    Parameters
    ----------
    alpha : float, default 0.001
        Base learning rate.
    rho : float, default 0.9
        Exponential decay rate for the running average of squared gradients. Must be
        in ``[0, 1)``.
    epsilon : float, default 1e-7
        Small constant added to the denominator for numerical stability.
    network : thor.Network, default None
        When network is passed in, this optimizer is set as the network default optimizer.

    Notes
    -----
    The dense update is:

    - ``square_average <- rho * square_average + (1 - rho) * g * g``
    - ``w <- w - alpha * g / (sqrt(square_average) + epsilon)``

    where ``g`` is Thor's batch/loss-scale normalized gradient.

    For sparse-row embedding updates, the same expression is applied to the touched rows.

    Examples
    --------
    Basic usage::

        from thor.optimizers import RMSprop

        opt = RMSprop(alpha=0.001, rho=0.9, epsilon=1e-7)

    See Also
    --------
    Adagrad : Accumulated-gradient adaptive optimizer.
    Adam : Adaptive Moment Estimation optimizer.
    AdamW : Adam with decoupled weight decay.
    Sgd : Stochastic Gradient Descent optimizer.
    """

    def __init__(self, alpha: float = 0.0010000000474974513, rho: float = 0.8999999761581421, epsilon: float = 1.0000000116860974e-07, network: thor.Network | None = None) -> None:
        """Construct an RMSprop optimizer."""

class Lamb(Optimizer):
    """
    LAMB optimizer.

    LAMB combines Adam-style first/second moment adaptation with a layer-wise trust
    ratio. It is commonly used for large-batch transformer training.

    Parameters
    ----------
    alpha : float, default 0.001
        Base learning rate.
    beta1 : float, default 0.9
        Exponential decay rate for the first-moment estimate.
    beta2 : float, default 0.999
        Exponential decay rate for the second-moment estimate.
    epsilon : float, default 1e-6
        Small constant added to the Adam denominator for numerical stability.
    weight_decay : float, default 0.01
        Weight decay coefficient included in the layer-wise update vector.
    trust_ratio_epsilon : float, default 1e-6
        Small constant added to the trust-ratio denominator.
    network : thor.Network, default None
        When network is passed in, this optimizer is set as the network default optimizer.

    Notes
    -----
    The dense update is:

    - ``m <- beta1 * m + (1 - beta1) * g``
    - ``v <- beta2 * v + (1 - beta2) * (g * g)``
    - ``u <- m_hat / (sqrt(v_hat) + epsilon) + weight_decay * w``
    - ``trust_ratio <- ||w||_2 / (||u||_2 + trust_ratio_epsilon)``
    - ``w <- w - alpha * trust_ratio * u``

    where ``g`` is Thor's batch/loss-scale normalized gradient. For 1-D tensors,
    Thor uses a trust ratio of 1.0, matching the common practice of excluding bias
    and normalization parameters from LAMB's layer-wise scaling.

    Sparse-row embedding updates are intentionally not supported because true LAMB
    needs layer-wide norms, while sparse-row optimizer fusion only sees touched rows.

    Examples
    --------
    Basic usage::

        from thor.optimizers import Lamb

        opt = Lamb(alpha=1e-3, weight_decay=0.01)

    See Also
    --------
    AdamW : Adam with decoupled weight decay.
    Muon : Momentum optimizer with Newton-Schulz orthogonalized matrix updates.
    """

    def __init__(self, alpha: float = 0.0010000000474974513, beta1: float = 0.8999999761581421, beta2: float = 0.9990000128746033, epsilon: float = 9.999999974752427e-07, weight_decay: float = 0.009999999776482582, trust_ratio_epsilon: float = 9.999999974752427e-07, network: thor.Network | None = None) -> None:
        """Construct a LAMB optimizer."""

class Lars(Optimizer):
    """
    LARS optimizer.

    LARS applies SGD with momentum and a layer-wise adaptive trust ratio. It is often
    used for large-batch convolutional training.

    Parameters
    ----------
    alpha : float, default 0.01
        Base learning rate.
    momentum : float, default 0.9
        Momentum coefficient.
    weight_decay : float, default 0.0
        Coupled weight decay coefficient included in the LARS update vector.
    trust_coefficient : float, default 0.001
        Coefficient used to scale the layer-wise trust ratio.
    epsilon : float, default 1e-8
        Small constant added to the trust-ratio denominator.
    nesterov_momentum : bool, default False
        Whether to use Nesterov momentum.
    network : thor.Network, default None
        When network is passed in, this optimizer is set as the network default optimizer.

    Notes
    -----
    The dense update is:

    - ``u <- g + weight_decay * w``
    - ``trust_ratio <- trust_coefficient * ||w||_2 / (||g||_2 + weight_decay * ||w||_2 + epsilon)``
    - ``v <- momentum * v + alpha * trust_ratio * u``
    - ``w <- w - v``

    where ``g`` is Thor's batch/loss-scale normalized gradient. For 1-D tensors,
    Thor uses a trust ratio of 1.0, matching the common practice of excluding bias
    and normalization parameters from layer-wise scaling.

    Sparse-row embedding updates are intentionally not supported because true LARS
    needs layer-wide norms, while sparse-row optimizer fusion only sees touched rows.

    Examples
    --------
    Basic usage::

        from thor.optimizers import Lars

        opt = Lars(alpha=0.1, momentum=0.9, weight_decay=1e-4)

    See Also
    --------
    Sgd : Stochastic gradient descent with optional momentum.
    Lamb : Adam-style optimizer with a layer-wise trust ratio.
    """

    def __init__(self, alpha: float = 0.009999999776482582, momentum: float = 0.8999999761581421, weight_decay: float = 0.0, trust_coefficient: float = 0.0010000000474974513, epsilon: float = 9.99999993922529e-09, nesterov_momentum: bool = False, network: thor.Network | None = None) -> None:
        """Construct a LARS optimizer."""

class Muon(Optimizer):
    """
    Muon optimizer.

    Muon applies momentum to dense rank-2 matrix parameters, orthogonalizes the
    resulting update with Newton-Schulz iterations, and applies a decoupled weight
    decay term. Non-matrix parameters and sparse-row updates are routed to a fallback
    optimizer. The builder default fallback is AdamW.

    Parameters
    ----------
    alpha : float, default 0.02
        Matrix-path learning rate.
    beta : float, default 0.95
        Momentum coefficient.
    epsilon : float, default 1e-8
        Newton-Schulz normalization epsilon.
    weight_decay : float, default 0.0
        Decoupled matrix-path weight decay.
    nesterov : bool, default True
        Whether to use a Nesterov-style momentum source before orthogonalization.
    num_iterations : int, default 5
        Number of Newton-Schulz iterations.
    fallback_optimizer : thor.optimizers.Optimizer, default None
        Optimizer used for non-matrix parameters and sparse-row updates. When omitted,
        AdamW is used.
    network : thor.Network, default None
        When network is passed in, this optimizer is set as the network default optimizer.
    """

    def __init__(self, alpha: float = 0.019999999552965164, beta: float = 0.949999988079071, epsilon: float = 9.99999993922529e-09, weight_decay: float = 0.0, nesterov: bool = True, num_iterations: int = 5, coefficient_a: float = 3.444499969482422, coefficient_b: float = -4.775000095367432, coefficient_c: float = 2.0315001010894775, transpose_tall_matrices: bool = True, fallback_optimizer: Optimizer | None = None, network: thor.Network | None = None) -> None:
        """Construct a Muon optimizer."""
