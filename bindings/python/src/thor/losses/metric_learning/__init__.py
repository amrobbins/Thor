"""Metric learning and contrastive representation-learning losses."""

from __future__ import annotations

from ..._thor.losses.metric_learning import ContrastiveLoss, CosineEmbeddingLoss, InfoNCELoss, TripletLoss

__all__ = [
    "ContrastiveLoss",
    "InfoNCELoss",
    "TripletLoss",
    "CosineEmbeddingLoss",
]
