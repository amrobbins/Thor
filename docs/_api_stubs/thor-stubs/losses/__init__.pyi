import enum
from typing import TypeAlias

import thor
import thor._thor.losses.classification as classification
import thor._thor.losses.detection as detection
import thor._thor.losses.distribution as distribution
import thor._thor.losses.gan as gan
import thor._thor.losses.metric_learning as metric_learning
import thor._thor.losses.ranking as ranking
import thor._thor.losses.segmentation as segmentation
import thor.physical


class Loss:
    def get_predictions(self) -> thor.Tensor: ...

    def get_labels(self) -> thor.Tensor: ...

    def get_loss(self) -> thor.Tensor: ...

    def get_example_weights(self) -> thor.Tensor | None: ...

    @property
    def loss_weight(self) -> float | None: ...

    @property
    def example_weights(self) -> thor.Tensor | None: ...

    class LabelType(enum.Enum):
        index = 5

        one_hot = 6

    class LossShape(enum.Enum):
        """
        Controls only the reported loss tensor shape. ``none`` disables the user-facing
        loss report while retaining the raw training objective. ``per_example`` sums
        all non-batch loss values independently for each example. ``per_output`` averages
        over the batch while preserving every non-batch loss dimension.
        """

        none = 0

        batch = 1

        per_example = 2

        per_output = 3

        raw = 4

class LabelType(enum.Enum):
    index = 5

    one_hot = 6

class LossShape(enum.Enum):
    """
    Controls only the reported loss tensor shape. ``none`` disables the user-facing
    loss report while retaining the raw training objective. ``per_example`` sums
    all non-batch loss values independently for each example. ``per_output`` averages
    over the batch while preserving every non-batch loss dimension.
    """

    none = 0

    batch = 1

    per_example = 2

    per_output = 3

    raw = 4

class CustomLoss(Loss):
    """
    Expression-backed custom loss.

    A CUDA-backed loss does not need a separate CudaKernelLoss type. Build one
    CudaKernelExpression for the raw loss, another for dLoss/dPredictions, convert
    both with ``as_dynamic_expression()``, and pass them here. The same
    Network-level CUDA-kernel source inspection and save/load key policy applies.

    Parameters
    ----------
    network : thor.Network
    loss_expression : thor.physical.DynamicExpression
        Expression that maps predictions and labels to a raw per-example loss tensor.
    gradient_expression : thor.physical.DynamicExpression
        Expression that maps predictions and labels to dLoss/dPredictions. Its output descriptor must match predictions.
    predictions : thor.Tensor
    labels : thor.Tensor
    loss_data_type : thor.DataType, default thor.DataType.FP32
    reported_loss_shape : thor.losses.Loss.LossShape, default LossShape.batch
        Use ``LossShape.none`` to keep the raw training objective without exposing a report tensor.
    predictions_name : str, default "predictions"
    labels_name : str, default "labels"
    loss_name : str, default "loss"
    gradient_name : str, default "predictions_grad"
    uses_batch_validity : bool, default False
        Declares that both expressions consume runtime batch validity. Thor currently supplies it through
        ``thor.BATCH_VALIDITY_MASK_NAME`` as an FP32 prefix mask.
    requires_full_batch : bool, default False
        Rejects partial-tail submissions for a batch-coupled loss that does not implement masked semantics.
    """

    def __init__(self, network: thor.Network, loss_expression: thor.physical.DynamicExpression, gradient_expression: thor.physical.DynamicExpression, predictions: thor.Tensor, labels: thor.Tensor, loss_data_type: thor.DataType = thor.DataType.fp32, reported_loss_shape: LossShape = LossShape.batch, predictions_name: str = 'predictions', labels_name: str = 'labels', loss_name: str = 'loss', gradient_name: str = 'predictions_grad', *, loss_weight: float | None = None, uses_batch_validity: bool | None = False, requires_full_batch: bool = False) -> None:
        """Construct an expression-backed CustomLoss."""

    @property
    def predictions_name(self) -> str: ...

    @property
    def labels_name(self) -> str: ...

    @property
    def loss_name(self) -> str: ...

    @property
    def gradient_name(self) -> str: ...

    @property
    def uses_batch_validity(self) -> bool: ...

    @property
    def requires_full_batch(self) -> bool: ...

class BinaryCrossEntropy(Loss):
    """
    Binary cross-entropy loss.

    Parameters
    ----------
    network : thor.Network
    predictions : thor.Tensor
    labels : thor.Tensor
    loss_data_type : thor.DataType, default thor.DataType.fp32
    reported_loss_shape : thor.losses.LossShape, default thor.losses.LossShape.batch
        Controls the reported loss tensor:

        * ``none`` does not expose a reportable loss tensor; the raw loss remains the training objective.
        * ``batch`` averages over the batch after summing all non-batch values.
        * ``per_example`` sums all non-batch values independently for each example.
        * ``per_output`` averages over the batch and preserves every non-batch dimension.
        * ``raw`` reports the unreduced pointwise loss.

    If you want to inspect mutually exclusive binary categories, it may be more convenient
    to use SparseCategoricalCrossEntropy with two classes.
    """

    def __init__(self, network: thor.Network, predictions: thor.Tensor, labels: thor.Tensor, loss_data_type: thor.DataType = thor.DataType.fp32, reported_loss_shape: LossShape = LossShape.batch, *, loss_weight: float | None = None) -> None:
        """Construct a Binary Cross Entropy loss."""

class CategoricalCrossEntropy(Loss):
    r"""
    Dense categorical cross-entropy loss.

    Parameters
    ----------
    network : thor.Network
    predictions : thor.Tensor
        Logits tensor whose final dimension is the class dimension.
    labels : thor.Tensor
        Dense class target tensor with the same dimensions as predictions. One-hot labels and soft labels are both supported.
    loss_data_type : thor.DataType, default thor.DataType.FP32
    reported_loss_shape : thor.losses.LossShape, default batch
        This setting does not affect training; it only controls the reported loss tensor shape.

    Notes
    -----
    A softmax is applied internally to convert logits into probabilities:

        p_c = exp(z_c) / \sum_{j=1}^{C} exp(z_j)

    The per-example dense categorical cross-entropy is then:

        L = -\sum_{c=1}^{C} y_c \log(p_c)

    Use SparseCategoricalCrossEntropy when labels are integer class ids.
    """

    def __init__(self, network: thor.Network, predictions: thor.Tensor, labels: thor.Tensor, loss_data_type: thor.DataType = thor.DataType.fp32, reported_loss_shape: LossShape = LossShape.batch, *, loss_weight: float | None = None) -> None:
        """Construct a dense/soft-label categorical cross-entropy loss."""

class SparseCategoricalCrossEntropy(CategoricalCrossEntropy):
    """
    Sparse categorical cross-entropy loss.

    Parameters
    ----------
    network : thor.Network
    predictions : thor.Tensor
        Logits tensor whose final dimension is the class dimension.
    labels : thor.Tensor
        Sparse integer class ids. Dimensions must match the prediction prefix dimensions, or that prefix with a trailing singleton.
    num_classes : int
        Number of classes in predictions.
    loss_data_type : thor.DataType, default thor.DataType.FP32
    reported_loss_shape : thor.losses.LossShape, default batch
        This setting does not affect training; it only controls the reported loss tensor shape.
    ignore_index : int, optional keyword-only
        Label id that contributes zero loss and zero logits gradient.
    mask : thor.Tensor, optional keyword-only
        Prefix-shaped boolean/uint8/fp16/fp32 mask. Entries > 0.5 are valid; masked entries contribute zero loss and zero gradient.

    Notes
    -----
    Sparse categorical cross-entropy is logits-native: it computes logsumexp(logits) - logits[class_id]
    without materializing a separate softmax tensor or a per-class raw loss tensor. The raw loss shape is
    the predictions prefix shape, e.g. predictions [B, S, V] produce raw loss [B, S].

    The logits gradient is dense and equivalent to softmax(logits) - one_hot(class_id).
    """

    def __init__(self, network: thor.Network, predictions: thor.Tensor, labels: thor.Tensor, num_classes: int, loss_data_type: thor.DataType = thor.DataType.fp32, reported_loss_shape: LossShape = LossShape.batch, *, loss_weight: float | None = None, ignore_index: int | None = None, mask: thor.Tensor | None = None) -> None:
        """Construct a sparse categorical cross-entropy loss."""

class MAE(Loss):
    """
    MAE loss.

    Parameters
    ----------
    network : thor.Network
    predictions : thor.Tensor
    labels : thor.Tensor
    loss_data_type : thor.DataType | None, default fp16 for fp16 predictions, otherwise fp32
    reported_loss_shape : thor.losses.LossShape, default thor.losses.LossShape.batch
        Controls the reported loss tensor:

        * ``none`` does not expose a reportable loss tensor; the raw loss remains the training objective.
        * ``batch`` averages over the batch after summing all non-batch values.
        * ``per_example`` sums all non-batch values independently for each example.
        * ``per_output`` averages over the batch and preserves every non-batch dimension.
        * ``raw`` reports the unreduced pointwise loss.
    """

    def __init__(self, network: thor.Network, predictions: thor.Tensor, labels: thor.Tensor, loss_data_type: thor.DataType | None = None, reported_loss_shape: LossShape | None = LossShape.batch, *, loss_weight: float | None = None, example_weights: thor.Tensor | None = None) -> None:
        """Construct a MAE loss."""

class MAPE(Loss):
    """
    MAPE loss.

    Parameters
    ----------
    network : thor.Network
    predictions : thor.Tensor
    labels : thor.Tensor
    loss_data_type : thor.DataType | None, default same data type as predictions
    reported_loss_shape : thor.losses.LossShape, default thor.losses.LossShape.batch
        Controls the reported loss tensor:

        * ``none`` does not expose a reportable loss tensor; the raw loss remains the training objective.
        * ``batch`` averages over the batch after summing all non-batch values.
        * ``per_example`` sums all non-batch values independently for each example.
        * ``per_output`` averages over the batch and preserves every non-batch dimension.
        * ``raw`` reports the unreduced pointwise loss.
    """

    def __init__(self, network: thor.Network, predictions: thor.Tensor, labels: thor.Tensor, loss_data_type: thor.DataType | None = None, reported_loss_shape: LossShape | None = LossShape.batch, *, loss_weight: float | None = None) -> None:
        """Construct a MAPE loss."""

class MSE(Loss):
    """
    MSE loss.

    Parameters
    ----------
    network : thor.Network
    predictions : thor.Tensor
    labels : thor.Tensor
    loss_data_type : thor.DataType | None, default fp16 for fp16 predictions, otherwise fp32
    reported_loss_shape : thor.losses.LossShape, default thor.losses.LossShape.batch
        Controls the reported loss tensor:

        * ``none`` does not expose a reportable loss tensor; the raw loss remains the training objective.
        * ``batch`` averages over the batch after summing all non-batch values.
        * ``per_example`` sums all non-batch values independently for each example.
        * ``per_output`` averages over the batch and preserves every non-batch dimension.
        * ``raw`` reports the unreduced pointwise loss.
    """

    def __init__(self, network: thor.Network, predictions: thor.Tensor, labels: thor.Tensor, loss_data_type: thor.DataType | None = None, reported_loss_shape: LossShape | None = LossShape.batch, *, loss_weight: float | None = None, example_weights: thor.Tensor | None = None) -> None:
        """Construct a MSE loss."""

class MeanPowerError(Loss):
    """
    MeanPowerError loss.

    MeanPowerError computes the mean absolute residual raised to a configurable
    power:

        loss = mean(abs(prediction - label) ** exponent)

    The exponent must be finite and greater than or equal to 1.0. The most useful
    range for ordinary regression losses is usually 1.0 <= exponent <= 2.0:

        MeanPowerError(exponent=1.0) is MeanAbsoluteError / MAE.
        MeanPowerError(exponent=2.0) is MeanSquaredError / MSE.
        1.0 < exponent < 2.0 gives behavior between MAE and MSE.

    Values greater than 2.0 are allowed for cases that intentionally give very large
    errors more leverage than MSE, but they are more outlier-sensitive.

    Parameters
    ----------
    network : thor.Network
    predictions : thor.Tensor
    labels : thor.Tensor
    exponent : float, default 1.5
        Power applied to abs(prediction - label). Must be >= 1.0.
    loss_data_type : thor.DataType | None, default fp16 for fp16 predictions, otherwise fp32
    reported_loss_shape : thor.losses.LossShape, default thor.losses.LossShape.batch
        Controls the reported loss tensor:

        * ``none`` does not expose a reportable loss tensor; the raw loss remains the training objective.
        * ``batch`` averages over the batch after summing all non-batch values.
        * ``per_example`` sums all non-batch values independently for each example.
        * ``per_output`` averages over the batch and preserves every non-batch dimension.
        * ``raw`` reports the unreduced pointwise loss.
    """

    def __init__(self, network: thor.Network, predictions: thor.Tensor, labels: thor.Tensor, exponent: float = 1.5, loss_data_type: thor.DataType | None = None, reported_loss_shape: LossShape | None = LossShape.batch, *, loss_weight: float | None = None, example_weights: thor.Tensor | None = None) -> None:
        """Construct a MeanPowerError loss."""

    @property
    def exponent(self) -> float: ...

class SmoothL1Loss(Loss):
    """
    Smooth L1 loss.

    SmoothL1Loss uses the PyTorch-style beta parameterization:

        0.5 * (prediction - label)^2 / beta    if |prediction - label| < beta
        |prediction - label| - 0.5 * beta      otherwise

    HuberLoss(delta=beta) is beta times SmoothL1Loss(beta=beta).
    """

    def __init__(self, network: thor.Network, predictions: thor.Tensor, labels: thor.Tensor, beta: float = 1.0, loss_data_type: thor.DataType | None = None, reported_loss_shape: LossShape | None = LossShape.batch, *, loss_weight: float | None = None) -> None:
        """Construct a Smooth L1 loss."""

    @property
    def beta(self) -> float: ...

class HuberLoss(Loss):
    """
    Huber loss.

    HuberLoss uses the standard delta parameterization:

        0.5 * (prediction - label)^2                    if |prediction - label| <= delta
        delta * (|prediction - label| - 0.5 * delta)    otherwise

    HuberLoss(delta=beta) is beta times SmoothL1Loss(beta=beta).
    """

    def __init__(self, network: thor.Network, predictions: thor.Tensor, labels: thor.Tensor, delta: float = 1.0, loss_data_type: thor.DataType | None = None, reported_loss_shape: LossShape | None = LossShape.batch, *, loss_weight: float | None = None) -> None:
        """Construct a Huber loss."""

    @property
    def delta(self) -> float: ...

class SoftTargetCrossEntropy(Loss):
    """
    Soft-target categorical cross entropy from logits.

    The predictions tensor contains unnormalized logits and the labels tensor contains
    a dense target distribution with the same class dimension. The raw loss is:

        -target * log_softmax(logits)

    The gradient assumes targets are normalized distributions.
    """

    def __init__(self, network: thor.Network, predictions: thor.Tensor, labels: thor.Tensor, loss_data_type: thor.DataType | None = None, reported_loss_shape: LossShape | None = LossShape.batch, *, loss_weight: float | None = None) -> None:
        """Construct a soft-target cross entropy loss."""

class KLDivLoss(Loss):
    """
    KL divergence from target distribution to model distribution.

    The predictions tensor contains unnormalized logits and the labels tensor contains
    a dense target distribution with the same class dimension. The raw loss is:

        target * (log(target) - log_softmax(logits))

    Zero target entries contribute zero to the loss. The gradient assumes targets are
    normalized distributions.
    """

    def __init__(self, network: thor.Network, predictions: thor.Tensor, labels: thor.Tensor, loss_data_type: thor.DataType | None = None, reported_loss_shape: LossShape | None = LossShape.batch, *, loss_weight: float | None = None) -> None:
        """Construct a KL divergence loss."""

class QuantileLoss(Loss):
    """
    Quantile / pinball loss.

    For quantile q and error y_true - y_pred:

        q * error          if error > 0
        (q - 1) * error    otherwise

    The subgradient at zero error is defined as 0.
    """

    def __init__(self, network: thor.Network, predictions: thor.Tensor, labels: thor.Tensor, quantile: float = 0.5, loss_data_type: thor.DataType | None = None, reported_loss_shape: LossShape | None = LossShape.batch, *, loss_weight: float | None = None, example_weights: thor.Tensor | None = None) -> None:
        """Construct a Quantile / pinball loss."""

    @property
    def quantile(self) -> float: ...

PinballLoss: TypeAlias = QuantileLoss

class ExpectileLoss(Loss):
    """
    Asymmetric squared-error expectile loss.

    For expectile tau and error y_true - y_pred, Thor uses:

        2 * tau       * error**2    if error > 0
        2 * (1 - tau) * error**2    otherwise

    The factor of two makes expectile=0.5 exactly equal to MSE, including its gradient.
    Expectiles below 0.5 emphasize over-prediction errors and estimate lower conditional
    expectiles; expectiles above 0.5 emphasize under-prediction errors and estimate upper
    conditional expectiles.
    """

    def __init__(self, network: thor.Network, predictions: thor.Tensor, labels: thor.Tensor, expectile: float = 0.5, loss_data_type: thor.DataType | None = None, reported_loss_shape: LossShape | None = LossShape.batch, *, loss_weight: float | None = None, example_weights: thor.Tensor | None = None) -> None:
        """Construct an asymmetric squared-error expectile loss."""

    @property
    def expectile(self) -> float: ...

class AsymmetricPowerLoss(Loss):
    """
    Asymmetric absolute-power regression loss.

    For level tau, exponent p, and error y_true - y_pred, Thor uses:

        2 * tau       * abs(error)**p    if error > 0
        2 * (1 - tau) * abs(error)**p    otherwise

    The normalization gives these exact relationships:

        AsymmetricPowerLoss(level=0.5, exponent=p) == MeanPowerError(exponent=p)
        AsymmetricPowerLoss(level=tau, exponent=2) == ExpectileLoss(expectile=tau)

    At exponent=1 this is twice the conventional pinball loss. That constant does not
    change the fitted optimum, and preserves exact equality with MeanPowerError at the
    central level. Exponents between 1 and 2 provide asymmetric bounds with robustness
    between quantile loss and expectile loss.
    """

    def __init__(self, network: thor.Network, predictions: thor.Tensor, labels: thor.Tensor, level: float = 0.5, exponent: float = 1.5, loss_data_type: thor.DataType | None = None, reported_loss_shape: LossShape | None = LossShape.batch, *, loss_weight: float | None = None, example_weights: thor.Tensor | None = None) -> None:
        """Construct an asymmetric absolute-power regression loss."""

    @property
    def level(self) -> float: ...

    @property
    def exponent(self) -> float: ...

class CtcLoss(Loss):
    def __init__(self, network: thor.Network, logits: thor.Tensor, labels: thor.RaggedTensor, input_lengths: thor.Tensor, reported_loss_shape: LossShape = LossShape.batch, *, loss_weight: float | None = None, out_of_bounds_gradients: str | None = 'zero') -> None:
        """
        Canonical cuDNN-backed CTC loss.

        ``labels`` must be a rank-1 ``thor.RaggedTensor`` with INT32 packed values and
        canonical UINT32/UINT64 row-partition offsets. Label lengths are derived from
        those offsets on device; there is no padded-label or separately supplied
        label-length API.
        """

    def get_labels(self) -> thor.RaggedTensor: ...

    def get_ragged_labels(self) -> thor.RaggedTensor: ...

    def get_input_lengths(self) -> thor.Tensor: ...

    def get_out_of_bounds_gradients(self) -> str: ...

__all__: list = ['AsymmetricPowerLoss', 'BinaryCrossEntropy', 'CategoricalCrossEntropy', 'CtcLoss', 'CustomLoss', 'ExpectileLoss', 'HuberLoss', 'KLDivLoss', 'LabelType', 'Loss', 'LossShape', 'MAE', 'MAPE', 'MSE', 'MeanPowerError', 'PinballLoss', 'QuantileLoss', 'SmoothL1Loss', 'SoftTargetCrossEntropy', 'SparseCategoricalCrossEntropy', 'classification', 'detection', 'distribution', 'gan', 'metric_learning', 'ranking', 'segmentation']
