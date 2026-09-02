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
    predictions : thor.Tensor or thor.RaggedTensor
    labels : thor.Tensor or thor.RaggedTensor
    loss_data_type : thor.DataType, default thor.DataType.fp32
    reported_loss_shape : thor.losses.LossShape, default thor.losses.LossShape.batch
        Controls the reported loss tensor:

        * ``none`` does not expose a reportable loss tensor; the raw loss remains the training objective.
        * ``batch`` averages over the batch after summing all non-batch values.
        * ``per_example`` sums all non-batch values independently for each example.
        * ``per_output`` averages over the batch and preserves every non-batch dimension for dense inputs; it is undefined for ragged inputs.
        * ``raw`` reports the unreduced pointwise loss.

    If you want to inspect mutually exclusive binary categories, it may be more convenient
    to use SparseCategoricalCrossEntropy with two classes.
    """

    def __init__(self, network: thor.Network, predictions: object, labels: object, loss_data_type: thor.DataType = thor.DataType.fp32, reported_loss_shape: LossShape = LossShape.batch, *, loss_weight: float | None = None) -> None:
        """Construct a dense or rank-1 ragged Binary Cross Entropy loss."""

    def get_predictions(self) -> object: ...

    def get_labels(self) -> object: ...

    def get_raw_loss(self) -> object: ...

    def get_loss(self) -> object: ...

    @property
    def is_ragged(self) -> bool: ...

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

    ``predictions`` and ``labels`` may both be dense ``thor.Tensor`` objects or
    rank-1 ``thor.RaggedTensor`` objects. Ragged inputs must share the exact same
    row-partition tensor. Ragged loss reporting supports ``none``, ``raw``,
    ``per_example``, and ``batch``; ``per_output`` is intentionally undefined.

    For ragged inputs, ``raw`` returns a ``thor.RaggedTensor`` with the same row
    partition. ``per_example`` returns one dense scalar per logical row and
    ``batch`` averages those row sums over valid logical examples rather than over
    active tokens. For ragged inputs, ``example_weights`` must have dimensions
    ``[1]`` and supplies one scalar weight per logical row; the weight is broadcast
    to that row's active tokens and scales both loss and prediction gradient.
    """

    def __init__(self, network: thor.Network, predictions: object, labels: object, loss_data_type: thor.DataType | None = None, reported_loss_shape: LossShape | None = LossShape.batch, *, loss_weight: float | None = None, example_weights: thor.Tensor | None = None) -> None:
        """Construct a dense or rank-1 ragged MAE loss."""

    def get_predictions(self) -> object: ...

    def get_labels(self) -> object: ...

    def get_raw_loss(self) -> object: ...

    def get_loss(self) -> object: ...

    @property
    def is_ragged(self) -> bool: ...

class MAPE(Loss):
    """
    MAPE loss.

    ``predictions`` and ``labels`` may both be dense ``thor.Tensor`` objects or rank-1
    ``thor.RaggedTensor`` objects. Ragged inputs must share the exact same row
    partition. Ragged reporting supports ``none``, ``raw``, ``per_example``, and
    ``batch``; ``per_output`` is intentionally undefined. ``batch`` averages
    per-row active-token sums over valid logical examples rather than active tokens.
    Ragged MAPE preserves the dense MAPE stability contract: epsilon=1e-4 and a maximum loss/gradient magnitude of 1000.
    """

    def __init__(self, network: thor.Network, predictions: object, labels: object, loss_data_type: thor.DataType | None = None, reported_loss_shape: LossShape | None = LossShape.batch, *, loss_weight: float | None = None) -> None:
        """Construct a dense or rank-1 ragged MAPE loss."""

    def get_predictions(self) -> object: ...

    def get_labels(self) -> object: ...

    def get_raw_loss(self) -> object: ...

    def get_loss(self) -> object: ...

    @property
    def is_ragged(self) -> bool: ...

class MSE(Loss):
    """
    MSE loss.

    ``predictions`` and ``labels`` may both be dense ``thor.Tensor`` objects or
    rank-1 ``thor.RaggedTensor`` objects. Ragged inputs must share the exact same
    row-partition tensor. Ragged loss reporting supports ``none``, ``raw``,
    ``per_example``, and ``batch``; ``per_output`` is intentionally undefined.

    For ragged inputs, ``raw`` returns a ``thor.RaggedTensor`` with the same row
    partition. ``per_example`` returns one dense scalar per logical row and
    ``batch`` averages those row sums over valid logical examples rather than over
    active tokens. For ragged inputs, ``example_weights`` must have dimensions
    ``[1]`` and supplies one scalar weight per logical row; the weight is broadcast
    to that row's active tokens and scales both loss and prediction gradient.
    """

    def __init__(self, network: thor.Network, predictions: object, labels: object, loss_data_type: thor.DataType | None = None, reported_loss_shape: LossShape | None = LossShape.batch, *, loss_weight: float | None = None, example_weights: thor.Tensor | None = None) -> None:
        """Construct a dense or rank-1 ragged MSE loss."""

    def get_predictions(self) -> object: ...

    def get_labels(self) -> object: ...

    def get_raw_loss(self) -> object: ...

    def get_loss(self) -> object: ...

    @property
    def is_ragged(self) -> bool: ...

class MeanPowerError(Loss):
    """
    MeanPowerError loss.

    ``predictions`` and ``labels`` may both be dense ``thor.Tensor`` objects or
    rank-1 ``thor.RaggedTensor`` objects. Ragged inputs must share the exact same
    row-partition tensor. Ragged loss reporting supports ``none``, ``raw``,
    ``per_example``, and ``batch``; ``per_output`` is intentionally undefined.

    The exponent must be finite and greater than or equal to 1.0. For ragged
    inputs, ``raw`` preserves the partition, ``per_example`` returns one dense
    scalar per logical row, and ``batch`` averages those row sums over valid
    logical examples rather than active tokens. Dense ``[1]`` example weights are
    broadcast to active tokens in each row and scale both loss and prediction
    gradient.
    """

    def __init__(self, network: thor.Network, predictions: object, labels: object, exponent: float = 1.5, loss_data_type: thor.DataType | None = None, reported_loss_shape: LossShape | None = LossShape.batch, *, loss_weight: float | None = None, example_weights: thor.Tensor | None = None) -> None:
        """Construct a dense or rank-1 ragged MeanPowerError loss."""

    def get_predictions(self) -> object: ...

    def get_labels(self) -> object: ...

    def get_raw_loss(self) -> object: ...

    def get_loss(self) -> object: ...

    @property
    def is_ragged(self) -> bool: ...

    @property
    def exponent(self) -> float: ...

class SmoothL1Loss(Loss):
    """
    SmoothL1Loss loss.

    ``predictions`` and ``labels`` may both be dense ``thor.Tensor`` objects or rank-1
    ``thor.RaggedTensor`` objects. Ragged inputs must share the exact same row
    partition. Ragged reporting supports ``none``, ``raw``, ``per_example``, and
    ``batch``; ``per_output`` is intentionally undefined. ``batch`` averages
    per-row active-token sums over valid logical examples rather than active tokens.
    SmoothL1Loss uses the PyTorch-style beta parameterization.
    """

    def __init__(self, network: thor.Network, predictions: object, labels: object, beta: float = 1.0, loss_data_type: thor.DataType | None = None, reported_loss_shape: LossShape | None = LossShape.batch, *, loss_weight: float | None = None) -> None:
        """Construct a dense or rank-1 ragged SmoothL1Loss loss."""

    def get_predictions(self) -> object: ...

    def get_labels(self) -> object: ...

    def get_raw_loss(self) -> object: ...

    def get_loss(self) -> object: ...

    @property
    def is_ragged(self) -> bool: ...

    @property
    def beta(self) -> float: ...

class HuberLoss(Loss):
    """
    HuberLoss loss.

    ``predictions`` and ``labels`` may both be dense ``thor.Tensor`` objects or rank-1
    ``thor.RaggedTensor`` objects. Ragged inputs must share the exact same row
    partition. Ragged reporting supports ``none``, ``raw``, ``per_example``, and
    ``batch``; ``per_output`` is intentionally undefined. ``batch`` averages
    per-row active-token sums over valid logical examples rather than active tokens.
    HuberLoss uses the standard delta parameterization.
    """

    def __init__(self, network: thor.Network, predictions: object, labels: object, delta: float = 1.0, loss_data_type: thor.DataType | None = None, reported_loss_shape: LossShape | None = LossShape.batch, *, loss_weight: float | None = None) -> None:
        """Construct a dense or rank-1 ragged HuberLoss loss."""

    def get_predictions(self) -> object: ...

    def get_labels(self) -> object: ...

    def get_raw_loss(self) -> object: ...

    def get_loss(self) -> object: ...

    @property
    def is_ragged(self) -> bool: ...

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


    Predictions and labels may both be dense tensors or rank-1 ragged tensors. Ragged
    inputs must share the exact row partition. Ragged reporting supports none, raw,
    per-example, and batch; per-output is undefined. Dense [1] example weights are
    broadcast over each logical row's active tokens.
    """

    def __init__(self, network: thor.Network, predictions: object, labels: object, quantile: float = 0.5, loss_data_type: thor.DataType | None = None, reported_loss_shape: LossShape | None = LossShape.batch, *, loss_weight: float | None = None, example_weights: thor.Tensor | None = None) -> None:
        """Construct a dense or rank-1 ragged Quantile / pinball loss."""

    def get_predictions(self) -> object: ...

    def get_labels(self) -> object: ...

    def get_raw_loss(self) -> object: ...

    def get_loss(self) -> object: ...

    @property
    def is_ragged(self) -> bool: ...

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


    Predictions and labels may both be dense tensors or rank-1 ragged tensors. Ragged
    inputs must share the exact row partition. Ragged reporting supports none, raw,
    per-example, and batch; per-output is undefined. Dense [1] example weights are
    broadcast over each logical row's active tokens.
    """

    def __init__(self, network: thor.Network, predictions: object, labels: object, expectile: float = 0.5, loss_data_type: thor.DataType | None = None, reported_loss_shape: LossShape | None = LossShape.batch, *, loss_weight: float | None = None, example_weights: thor.Tensor | None = None) -> None:
        """
        Construct a dense or rank-1 ragged asymmetric squared-error expectile loss.
        """

    def get_predictions(self) -> object: ...

    def get_labels(self) -> object: ...

    def get_raw_loss(self) -> object: ...

    def get_loss(self) -> object: ...

    @property
    def is_ragged(self) -> bool: ...

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


    Predictions and labels may both be dense tensors or rank-1 ragged tensors. Ragged
    inputs must share the exact row partition. Ragged reporting supports none, raw,
    per-example, and batch; per-output is undefined. Dense [1] example weights are
    broadcast over each logical row's active tokens.
    """

    def __init__(self, network: thor.Network, predictions: object, labels: object, level: float = 0.5, exponent: float = 1.5, loss_data_type: thor.DataType | None = None, reported_loss_shape: LossShape | None = LossShape.batch, *, loss_weight: float | None = None, example_weights: thor.Tensor | None = None) -> None:
        """
        Construct a dense or rank-1 ragged asymmetric absolute-power regression loss.
        """

    def get_predictions(self) -> object: ...

    def get_labels(self) -> object: ...

    def get_raw_loss(self) -> object: ...

    def get_loss(self) -> object: ...

    @property
    def is_ragged(self) -> bool: ...

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
