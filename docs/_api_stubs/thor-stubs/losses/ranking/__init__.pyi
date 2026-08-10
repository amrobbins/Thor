"""Ranking losses."""

import thor
import thor.losses


class ListNetLoss(thor.losses.Loss):
    """
    ListNet loss over fixed-size query/document lists.

    The predictions tensor contains unnormalized model scores for the documents in one fixed-size
    list, and the labels tensor contains relevance labels with the same list dimension. The target
    ranking distribution is computed with softmax(labels / label_temperature). The prediction
    ranking distribution is computed from predictions / score_temperature, and the raw loss is one
    scalar per list:

        -sum(target * log_softmax(predictions / score_temperature))

    The current implementation supports fixed-size lists. Use the optional mask tensor for padded
    fixed-size lists. Mask values > 0.5 mark valid documents; values <= 0.5 mark padded documents.
    Masked documents contribute zero loss and zero prediction gradient, and fully masked rows
    produce zero raw loss and zero prediction gradient. True ragged query/document groups require
    later segment/ragged scaffolding.
    """

    def __init__(self, network: thor.Network, predictions: thor.Tensor, labels: thor.Tensor, score_temperature: float = 1.0, label_temperature: float = 1.0, loss_data_type: thor.DataType | None = None, reported_loss_shape: thor.losses.LossShape | None = thor.losses.LossShape.batch, mask: thor.Tensor | None = None, *, loss_weight: float | None = None) -> None:
        """Construct a ListNet loss over fixed-size query/document lists."""

    @property
    def score_temperature(self) -> float: ...

    @property
    def label_temperature(self) -> float: ...

class ListwiseSoftmaxCrossEntropyLoss(thor.losses.Loss):
    """
    Listwise softmax cross entropy over fixed-size query/document lists.

    The predictions tensor contains unnormalized model scores for the documents in one fixed-size
    list, and the labels tensor contains a target distribution or nonnegative target weights with
    the same list dimension. An optional mask tensor may mark padded documents with values <= 0.5
    and valid documents with values > 0.5. Masked documents contribute zero loss and zero prediction
    gradient; fully masked rows produce zero raw loss and zero prediction gradient. The prediction
    ranking distribution is computed from predictions / temperature across the valid documents,
    and the raw loss is one scalar per list:

        -sum(labels * log_softmax(predictions / temperature))

    When labels sum to one, this is standard listwise softmax cross entropy. When labels are
    nonnegative weights, the gradient uses the per-list target mass.
    """

    def __init__(self, network: thor.Network, predictions: thor.Tensor, labels: thor.Tensor, temperature: float = 1.0, loss_data_type: thor.DataType | None = None, reported_loss_shape: thor.losses.LossShape | None = thor.losses.LossShape.batch, mask: thor.Tensor | None = None, *, loss_weight: float | None = None) -> None:
        """
        Construct a listwise softmax cross entropy loss over fixed-size query/document lists.
        """

    @property
    def temperature(self) -> float: ...

class MarginRankingLoss(thor.losses.Loss):
    """
    Margin ranking loss over two score tensors and one target tensor.

    The raw elementwise loss is:

        max(margin - target * (input1 - input2), 0)

    For target = 1, this encourages input1 to rank above input2 by at least margin.
    For target = -1, this encourages input2 to rank above input1 by at least margin.

    Gradients are produced for input1 and input2. The target tensor is an auxiliary,
    non-differentiable input.
    """

    def __init__(self, network: thor.Network, input1: thor.Tensor, input2: thor.Tensor, target: thor.Tensor, margin: float = 0.0, loss_data_type: thor.DataType | None = None, reported_loss_shape: thor.losses.LossShape | None = thor.losses.LossShape.batch, *, loss_weight: float | None = None) -> None:
        """
        Construct a MarginRankingLoss over two score tensors and a target tensor.
        """

    @property
    def margin(self) -> float: ...

    def get_input1(self) -> thor.Tensor: ...

    def get_input2(self) -> thor.Tensor: ...

    def get_target(self) -> thor.Tensor: ...

__all__: list = ['MarginRankingLoss', 'ListNetLoss', 'ListwiseSoftmaxCrossEntropyLoss']
