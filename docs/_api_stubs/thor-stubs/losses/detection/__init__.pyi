"""Object detection losses."""

import thor
import thor.losses


class CIoULoss(thor.losses.Loss):
    def __init__(self, network: thor.Network, predictions: thor.Tensor, labels: thor.Tensor, box_format: str = 'xyxy', eps: float = 1.0000000116860974e-07, loss_data_type: thor.DataType | None = None, reported_loss_shape: thor.losses.LossShape | None = thor.losses.LossShape.batch, *, loss_weight: float | None = None) -> None:
        """Construct a Complete IoU box regression loss for xyxy boxes."""

    @property
    def eps(self) -> float: ...

    @property
    def box_format(self) -> str: ...

class DIoULoss(thor.losses.Loss):
    def __init__(self, network: thor.Network, predictions: thor.Tensor, labels: thor.Tensor, box_format: str = 'xyxy', eps: float = 1.0000000116860974e-07, loss_data_type: thor.DataType | None = None, reported_loss_shape: thor.losses.LossShape | None = thor.losses.LossShape.batch, *, loss_weight: float | None = None) -> None:
        """Construct a Distance IoU box regression loss for xyxy boxes."""

    @property
    def eps(self) -> float: ...

    @property
    def box_format(self) -> str: ...

class GIoULoss(thor.losses.Loss):
    def __init__(self, network: thor.Network, predictions: thor.Tensor, labels: thor.Tensor, box_format: str = 'xyxy', eps: float = 1.0000000116860974e-07, loss_data_type: thor.DataType | None = None, reported_loss_shape: thor.losses.LossShape | None = thor.losses.LossShape.batch, *, loss_weight: float | None = None) -> None:
        """Construct a Generalized IoU box regression loss for xyxy boxes."""

    @property
    def eps(self) -> float: ...

    @property
    def box_format(self) -> str: ...

class IoULoss(thor.losses.Loss):
    def __init__(self, network: thor.Network, predictions: thor.Tensor, labels: thor.Tensor, box_format: str = 'xyxy', eps: float = 1.0000000116860974e-07, loss_data_type: thor.DataType | None = None, reported_loss_shape: thor.losses.LossShape | None = thor.losses.LossShape.batch, *, loss_weight: float | None = None) -> None:
        """Construct an IoU box regression loss for xyxy boxes."""

    @property
    def eps(self) -> float: ...

    @property
    def box_format(self) -> str: ...

__all__: list = ['IoULoss', 'GIoULoss', 'DIoULoss', 'CIoULoss']
