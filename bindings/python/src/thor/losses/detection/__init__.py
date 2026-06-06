"""Object detection losses."""

from __future__ import annotations

from ..._thor.losses.detection import CIoULoss, DIoULoss, GIoULoss, IoULoss

__all__ = [
    "IoULoss",
    "GIoULoss",
    "DIoULoss",
    "CIoULoss",
]
