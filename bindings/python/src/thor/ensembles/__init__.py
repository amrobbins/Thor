"""Ensemble model artifact manifest support."""

from __future__ import annotations

from ._manifest import EnsembleAggregation
from ._manifest import EnsembleMemberSpec
from ._manifest import EnsembleModel

__all__ = [
    "EnsembleAggregation",
    "EnsembleMemberSpec",
    "EnsembleModel",
]


def __dir__() -> list[str]:
    return sorted(__all__)
