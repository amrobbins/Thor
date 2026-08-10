"""Segmentation losses."""

import thor
import thor.losses


class DiceLoss(thor.losses.Loss):
    """
    Soft Dice loss over dense probability tensors.

    Predictions are expected to already be probabilities, not logits. For a one-dimensional feature tensor [N], Dice is computed globally over N for each sample. For [C, ...spatial], Dice is computed per class/channel C by reducing only the spatial axes:

        1 - (2 * sum_spatial(prediction * label) + smooth) / (sum_spatial(prediction) + sum_spatial(label) + smooth)

    Use a sigmoid/softmax activation before this loss if the model output is logits.
    """

    def __init__(self, network: thor.Network, predictions: thor.Tensor, labels: thor.Tensor, smooth: float = 1.0, loss_data_type: thor.DataType | None = None, reported_loss_shape: thor.losses.LossShape | None = thor.losses.LossShape.batch, *, loss_weight: float | None = None) -> None:
        """Construct a Dice loss."""

    @property
    def smooth(self) -> float: ...

class FocalTverskyLoss(thor.losses.Loss):
    """
    Focal Tversky loss over dense probability tensors.

    Predictions are expected to already be probabilities, not logits. For a one-dimensional feature tensor [N], Focal Tversky is computed globally over N for each sample. For [C, ...spatial], Focal Tversky is computed per class/channel C by reducing only the spatial axes:

        (1 - TverskyIndex) ** gamma

    where TverskyIndex = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth).
    """

    def __init__(self, network: thor.Network, predictions: thor.Tensor, labels: thor.Tensor, alpha: float = 0.30000001192092896, beta: float = 0.699999988079071, gamma: float = 0.75, smooth: float = 1.0, loss_data_type: thor.DataType | None = None, reported_loss_shape: thor.losses.LossShape | None = thor.losses.LossShape.batch, *, loss_weight: float | None = None) -> None:
        """Construct a focal Tversky loss."""

    @property
    def alpha(self) -> float: ...

    @property
    def beta(self) -> float: ...

    @property
    def gamma(self) -> float: ...

    @property
    def smooth(self) -> float: ...

class TverskyLoss(thor.losses.Loss):
    """
    Tversky loss over dense probability tensors.

    Predictions are expected to already be probabilities, not logits. For a one-dimensional feature tensor [N], Tversky is computed globally over N for each sample. For [C, ...spatial], Tversky is computed per class/channel C by reducing only the spatial axes:

        1 - (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

    alpha weights false positives and beta weights false negatives.
    """

    def __init__(self, network: thor.Network, predictions: thor.Tensor, labels: thor.Tensor, alpha: float = 0.5, beta: float = 0.5, smooth: float = 1.0, loss_data_type: thor.DataType | None = None, reported_loss_shape: thor.losses.LossShape | None = thor.losses.LossShape.batch, *, loss_weight: float | None = None) -> None:
        """Construct a Tversky loss."""

    @property
    def alpha(self) -> float: ...

    @property
    def beta(self) -> float: ...

    @property
    def smooth(self) -> float: ...

__all__: list = ['DiceLoss', 'TverskyLoss', 'FocalTverskyLoss']
