"""Ranking losses."""

from __future__ import annotations

from ..._thor.losses.ranking import ListNetLoss, ListwiseSoftmaxCrossEntropyLoss, MarginRankingLoss

__all__ = [
    "MarginRankingLoss",
    "ListNetLoss",
    "ListwiseSoftmaxCrossEntropyLoss",
]
