from typing import overload

import thor
import thor.physical


class Activation:
    def to_expression(self, input: thor.physical.Expression) -> thor.physical.Expression:
        """
        Return an expression equivalent to applying this activation to the supplied expression.
        """

    @staticmethod
    def epilogue_input(output_dtype: object | None = None, compute_dtype: object | None = None) -> thor.physical.Expression:
        """
        Return the primary tensor input expression expected by an activation epilogue.
        """

    @staticmethod
    def epilogue_aux_input(name: str, output_dtype: object | None = None, compute_dtype: object | None = None) -> thor.physical.Expression:
        """
        Return a named auxiliary tensor input expression for an activation epilogue.
        """

    @overload
    def add_to_network(self, network: thor.Network, feature_input: thor.Tensor, epilogue: object | None = None, epilogue_inputs: object | None = None) -> thor.Tensor:
        """
        Attach this activation as a standalone expression-backed layer and return its feature output tensor.
        """

    @overload
    def add_to_network(self, network: thor.Network, feature_input: thor.RaggedTensor, epilogue: object | None = None, epilogue_inputs: object | None = None) -> thor.RaggedTensor:
        """
        Attach this activation to packed ragged values and preserve the input row partition.
        """

class Glu(Activation):
    """
    Gated Linear Unit activation.

    This activation splits the final feature dimension into two equal halves,
    then returns the first half multiplied by a transformed gate half.
    It is intended as a standalone shape-changing activation layer.
    """

    def __init__(self) -> None:
        """Initialize a Glu activation (construction happens in __new__)."""

class Reglu(Activation):
    """
    Rectified gated linear unit activation.

    This activation splits the final feature dimension into two equal halves,
    then returns the first half multiplied by a transformed gate half.
    It is intended as a standalone shape-changing activation layer.
    """

    def __init__(self) -> None:
        """Initialize a Reglu activation (construction happens in __new__)."""

class Geglu(Activation):
    """
    GELU gated linear unit activation.

    This activation splits the final feature dimension into two equal halves,
    then returns the first half multiplied by a transformed gate half.
    It is intended as a standalone shape-changing activation layer.
    """

    def __init__(self) -> None:
        """Initialize a Geglu activation (construction happens in __new__)."""

class Swiglu(Activation):
    """
    Swish gated linear unit activation.

    This activation splits the final feature dimension into two equal halves,
    then returns the first half multiplied by a transformed gate half.
    It is intended as a standalone shape-changing activation layer.
    """

    def __init__(self) -> None:
        """Initialize a Swiglu activation (construction happens in __new__)."""

class BilinearGlu(Activation):
    """
    Bilinear gated linear unit activation.

    This activation splits the final feature dimension into two equal halves,
    then returns the first half multiplied by a transformed gate half.
    It is intended as a standalone shape-changing activation layer.
    """

    def __init__(self) -> None:
        """Initialize a BilinearGlu activation (construction happens in __new__)."""

class Elu(Activation):
    """
    Exponential Linear Unit (ELU) activation.

    ELU is defined elementwise as

        f(x) = x                    if x > 0
               alpha * (exp(x) - 1) if x <= 0

    where ``alpha`` is a positive constant (typically ``alpha = 1``).
    Compared to ReLU, ELU has negative outputs for negative inputs.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        """Construct an ELU activation."""

class Exponential(Activation):
    """
    Exponential activation.

    Applied elementwise, this activation is defined as

        f(x) = exp(x)

    It maps all real inputs to positive outputs and grows rapidly for
    large positive values. This can be useful in certain architectures,
    but it may also lead to exploding activations if not combined with
    appropriate normalization or regularization.
    """

    def __init__(self) -> None:
        """
        Initialize an Exponential activation (construction happens in __new__).
        """

class Gelu(Activation):
    """
    Gaussian Error Linear Unit (GELU) activation.

    Applied elementwise, the exact GELU is defined as

        f(x) = x * Φ(x)

    where Φ(x) is the CDF of a standard normal distribution. A common
    tanh-based approximation is

        f(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³))).
    """

    def __init__(self) -> None:
        """Initialize a GELU activation (construction happens in __new__)."""

class HardSigmoid(Activation):
    """
    Hard sigmoid activation.

    Hard-sigmoid is a piecewise-linear approximation to the standard sigmoid,
    defined elementwise as a clipped line:

        f(x) ≈ clip(a * x + b, 0, 1)

    where ``a`` and ``b`` are chosen so that the function transitions smoothly
    from 0 to 1 over a finite interval. Compared to the standard sigmoid,
    hard-sigmoid is cheaper to evaluate and has constant slopes in the central
    region.
    """

    def __init__(self) -> None:
        """Initialize a HardSigmoid activation (construction happens in __new__)."""

class HardSwish(Activation):
    """
    Hard swish activation.

    Applied elementwise, hard-swish is defined as

        f(x) = x * relu6(x + 3) / 6

    It is a piecewise-linear approximation to swish.
    """

    def __init__(self) -> None:
        """Initialize a HardSwish activation (construction happens in __new__)."""

class HardTanh(Activation):
    """
    Hard tanh activation.

    Applied elementwise, hard-tanh clamps x to [min_value, max_value].
    """

    def __init__(self, min_value: float = -1.0, max_value: float = 1.0) -> None:
        """Initialize a HardTanh activation (construction happens in __new__)."""

class Mish(Activation):
    """
    Mish activation.

    Applied elementwise, Mish is defined as

        f(x) = x * tanh(softplus(x))

    It is a smooth, non-monotonic activation.
    """

    def __init__(self) -> None:
        """Initialize a Mish activation (construction happens in __new__)."""

class Relu(Activation):
    """
    Rectified Linear Unit (ReLU) activation.

    Applied elementwise, ReLU is defined as

        f(x) = max(0, x)

    ReLU preserves positive inputs and sets negative inputs to zero. It is
    computationally cheap, helps mitigate vanishing gradients compared to
    saturating activations (e.g., sigmoid/tanh), and is widely used in deep
    networks. A potential drawback is "dead" neurons when inputs remain
    negative for long periods, causing zero gradients in that region.
    """

    def __init__(self) -> None:
        """Initialize a ReLU activation (construction happens in __new__)."""

class Relu6(Activation):
    """
    ReLU6 activation.

    Applied elementwise, ReLU6 is defined as

        f(x) = min(max(x, 0), 6)

    It is commonly used in mobile-efficient networks.
    """

    def __init__(self) -> None:
        """Initialize a Relu6 activation (construction happens in __new__)."""

class Selu(Activation):
    """
    Scaled Exponential Linear Unit (SELU) activation.

    SELU is defined elementwise as

        f(x) = λ * x                if x > 0
               λ * α * (exp(x) - 1) if x <= 0

    where λ (lambda) and α (alpha) are fixed positive constants
    (α ≈ 1.67326 and λ ≈ 1.05070). With appropriate weight
    initialization and architecture constraints, SELU can encourage
    self-normalizing behavior, keeping activations close to zero mean
    and unit variance throughout deep networks.
    """

    def __init__(self) -> None:
        """Initialize a SELU activation (construction happens in __new__)."""

class Sigmoid(Activation):
    """
    Sigmoid activation.

    Applied elementwise, this activation is defined as

        f(x) = 1 / (1 + exp(-x))

    It maps real-valued inputs into the interval (0, 1) and is commonly used
    when outputs are interpreted as probabilities or gates (e.g., in recurrent
    networks). Note that sigmoid can suffer from saturation for large |x|,
    which may slow down learning if not combined with appropriate initialization
    or normalization.
    """

    def __init__(self) -> None:
        """Initialize a Sigmoid activation (construction happens in __new__)."""

class SoftPlus(Activation):
    """
    SoftPlus activation.

    SoftPlus is a smooth approximation to ReLU, defined elementwise as

        f(x) = log(1 + exp(x))

    It maps real inputs to positive outputs and grows roughly linearly for
    large positive x, while remaining strictly positive and differentiable
    everywhere. Compared to ReLU, SoftPlus avoids a hard kink at zero.
    """

    def __init__(self) -> None:
        """Initialize a SoftPlus activation (construction happens in __new__)."""

class SoftSign(Activation):
    """
    SoftSign activation.

    SoftSign is a smooth, bounded activation defined elementwise as

        f(x) = x / (1 + |x|)

    It squashes large positive and negative values toward +1 and -1,
    respectively, while remaining smooth and differentiable everywhere.
    Compared to tanh, SoftSign has polynomial rather than exponential
    tails, which can lead to slightly different gradient behavior for
    large |x|.
    """

    def __init__(self) -> None:
        """Initialize a SoftSign activation (construction happens in __new__)."""

class Softmax(Activation):
    """
    Softmax activation.

    Softmax is typically applied along the last (feature) dimension of a tensor
    to convert raw scores (logits) into a probability distribution. For an input
    vector x, softmax is defined as

        softmax(x_i) = exp(x_i) / Σ_j exp(x_j)

    for each component i. The outputs are all positive and sum to 1, making
    softmax a natural choice for multi-class classification outputs. A numerically
    stable form (subtracting max(x) before exponentiation) is used to avoid overflow.
    """

    def __init__(self) -> None:
        """Initialize a Softmax activation (construction happens in __new__)."""

class Swish(Activation):
    """
    Swish (SiLU) activation.

    Swish is a smooth, non-monotonic activation defined elementwise as

        f(x) = x * sigmoid(x)
             = x / (1 + exp(-x))

    It behaves roughly like a smoothed, self-gated ReLU: small negative inputs
    are softly suppressed, while large positive inputs pass through almost
    linearly. Swish (also known as SiLU) has been shown to work well in a
    variety of deep architectures, particularly in modern convolutional and
    transformer-based models.
    """

    def __init__(self) -> None:
        """
        Initialize a Swish (SiLU) activation (construction happens in __new__).
        """

class Tanh(Activation):
    """
    Hyperbolic tangent (tanh) activation.

    Applied elementwise, tanh is defined as

        f(x) = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    It squashes real-valued inputs into the range (-1, 1) with approximately
    linear behavior around zero and saturation for large |x|. Tanh is often
    used in recurrent networks and can be viewed as a zero-centered alternative
    to the logistic sigmoid.
    """

    def __init__(self) -> None:
        """Initialize a Tanh activation (construction happens in __new__)."""

class Threshold(Activation):
    """
    Threshold activation.

    Applied elementwise, threshold returns x when x > threshold, otherwise value.
    """

    def __init__(self, threshold: float = 0.0, value: float = 0.0) -> None:
        """Initialize a Threshold activation (construction happens in __new__)."""
