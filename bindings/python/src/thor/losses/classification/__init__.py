"""Classification losses beyond Thor's core flat loss namespace."""

from __future__ import annotations

from ..._thor.losses.classification import BinaryFocalLoss, CategoricalFocalLoss

__all__ = [
    "BinaryFocalLoss",
    "CategoricalFocalLoss",
]
