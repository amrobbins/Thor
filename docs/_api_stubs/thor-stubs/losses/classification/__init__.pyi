"""Classification losses beyond Thor's core flat loss namespace."""

import thor
import thor.losses


class BinaryFocalLoss(thor.losses.Loss):
    """
    Binary focal loss from logits.

    The predictions tensor contains a nonempty per-example tensor of independent unnormalized binary
    logits, and the labels tensor contains matching binary targets. A shape of [1] is the standard
    binary-classification case; wider or higher-rank tensors support multi-output, multilabel, and dense
    prediction objectives. The raw loss is applied pointwise:

        alpha_t * (1 - p_t) ** gamma * BCEWithLogits(logit, target)

    where alpha_t is alpha for positive targets and 1 - alpha for negative targets.

    Predictions and labels may both be dense tensors or rank-1 ragged tensors. Ragged
    inputs must share the exact row partition; raw loss preserves that partition and
    per-example/batch reporting uses active tokens only.
    """

    def __init__(self, network: thor.Network, predictions: object, labels: object, gamma: float = 2.0, alpha: float = 0.25, loss_data_type: thor.DataType | None = None, reported_loss_shape: thor.losses.LossShape | None = thor.losses.LossShape.batch, *, loss_weight: float | None = None) -> None:
        """Construct a dense or rank-1 ragged binary focal loss."""

    def get_predictions(self) -> object: ...

    def get_labels(self) -> object: ...

    def get_raw_loss(self) -> object: ...

    def get_loss(self) -> object: ...

    @property
    def is_ragged(self) -> bool: ...

    @property
    def gamma(self) -> float: ...

    @property
    def alpha(self) -> float: ...

class CategoricalFocalLoss(thor.losses.Loss):
    """
    Categorical focal loss from logits and dense target distributions.

    The predictions tensor contains unnormalized logits and the labels tensor contains a dense
    one-hot or soft target distribution with the same class dimension. The raw loss is:

        -alpha * target * (1 - softmax(logits)) ** gamma * log_softmax(logits)

    For sparse class-index targets, use a sparse focal wrapper later rather than this dense-target loss.
    """

    def __init__(self, network: thor.Network, predictions: thor.Tensor, labels: thor.Tensor, gamma: float = 2.0, alpha: float = 1.0, loss_data_type: thor.DataType | None = None, reported_loss_shape: thor.losses.LossShape | None = thor.losses.LossShape.batch, *, loss_weight: float | None = None) -> None:
        """Construct a categorical focal loss from logits and dense targets."""

    @property
    def gamma(self) -> float: ...

    @property
    def alpha(self) -> float: ...

__all__: list = ['BinaryFocalLoss', 'CategoricalFocalLoss']
