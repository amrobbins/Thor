"""Metric learning and contrastive representation-learning losses."""

import thor
import thor.losses


class ContrastiveLoss(thor.losses.Loss):
    """
    Distance-based contrastive loss.

    The predictions tensor contains pair distances and the labels tensor contains binary pair labels.
    Labels greater than 0.5 are treated as positive/similar pairs. The raw loss is:

        distance ** 2                         if label > 0.5
        max(margin - distance, 0) ** 2        otherwise

    The predictions tensor is expected to contain non-negative distances; Thor does not clamp it.
    """

    def __init__(self, network: thor.Network, predictions: thor.Tensor, labels: thor.Tensor, margin: float = 1.0, loss_data_type: thor.DataType | None = None, reported_loss_shape: thor.losses.LossShape | None = thor.losses.LossShape.batch, *, loss_weight: float | None = None) -> None:
        """Construct a distance-based contrastive loss."""

    @property
    def margin(self) -> float: ...

class CosineEmbeddingLoss(thor.losses.Loss):
    """
    Cosine embedding loss over two embedding tensors and one target label per example.

    For target > 0, the raw loss is:

        1 - cosine(input1, input2)

    For target <= 0, the raw loss is:

        max(cosine(input1, input2) - margin, 0)

    Gradients are produced for input1 and input2. The target tensor is an auxiliary,
    non-differentiable input.
    """

    def __init__(self, network: thor.Network, input1: thor.Tensor, input2: thor.Tensor, target: thor.Tensor, margin: float = 0.0, eps: float = 9.99999993922529e-09, loss_data_type: thor.DataType | None = None, reported_loss_shape: thor.losses.LossShape | None = thor.losses.LossShape.batch, *, loss_weight: float | None = None) -> None:
        """
        Construct a CosineEmbeddingLoss over two embedding tensors and a target label tensor.
        """

    @property
    def margin(self) -> float: ...

    @property
    def eps(self) -> float: ...

    def get_input1(self) -> thor.Tensor: ...

    def get_input2(self) -> thor.Tensor: ...

    def get_target(self) -> thor.Tensor: ...

class InfoNCELoss(thor.losses.Loss):
    """
    InfoNCE loss from similarity logits and dense target distributions.

    The predictions tensor contains unnormalized similarity logits over the candidate set and
    the labels tensor contains a dense one-hot, multi-hot, or soft target distribution with
    the same candidate dimension. The raw loss is:

        -target * log_softmax(logits / temperature)

    For the standard one-positive in-batch contrastive case, pass one-hot labels whose positive
    entry selects the matching candidate for each batch item.
    """

    def __init__(self, network: thor.Network, predictions: thor.Tensor, labels: thor.Tensor, temperature: float = 1.0, loss_data_type: thor.DataType | None = None, reported_loss_shape: thor.losses.LossShape | None = thor.losses.LossShape.batch, *, loss_weight: float | None = None) -> None:
        """Construct an InfoNCE loss from similarity logits and dense targets."""

    @property
    def temperature(self) -> float: ...

class TripletLoss(thor.losses.Loss):
    """
    Triplet margin loss over anchor, positive, and negative embedding tensors.

    The raw loss per example is:

        max(||anchor - positive||_2 - ||anchor - negative||_2 + margin, 0)

    All three inputs are differentiable. This loss expects already-formed triplets; triplet
    mining is intentionally separate from the loss.
    """

    def __init__(self, network: thor.Network, anchor: thor.Tensor, positive: thor.Tensor, negative: thor.Tensor, margin: float = 1.0, eps: float = 9.999999974752427e-07, loss_data_type: thor.DataType | None = None, reported_loss_shape: thor.losses.LossShape | None = thor.losses.LossShape.batch, *, loss_weight: float | None = None) -> None:
        """
        Construct a Triplet margin loss over anchor, positive, and negative embeddings.
        """

    @property
    def margin(self) -> float: ...

    @property
    def eps(self) -> float: ...

    def get_anchor(self) -> thor.Tensor: ...

    def get_positive(self) -> thor.Tensor: ...

    def get_negative(self) -> thor.Tensor: ...

__all__: list = ['ContrastiveLoss', 'InfoNCELoss', 'TripletLoss', 'CosineEmbeddingLoss']
