"""Segmentation losses."""

from __future__ import annotations

from ..._thor.losses.segmentation import DiceLoss, FocalTverskyLoss, TverskyLoss

__all__ = [
    "DiceLoss",
    "TverskyLoss",
    "FocalTverskyLoss",
]
