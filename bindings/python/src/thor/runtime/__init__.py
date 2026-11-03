"""Runtime execution namespace."""

from __future__ import annotations

from .. import _thor as _thor

PlacedNetwork = _thor.PlacedNetwork
ReductionTuningDepth = _thor.ReductionTuningDepth
StatusCode = _thor.StatusCode

__all__ = [
    "PlacedNetwork",
    "ReductionTuningDepth",
    "StatusCode",
]


def _set_public_name(obj: object, name: str) -> None:
    for attr, value in (("__module__", __name__), ("__qualname__", name), ("__name__", name)):
        try:
            setattr(obj, attr, value)
        except (AttributeError, TypeError):
            pass


for _obj, _name in (
    (PlacedNetwork, "PlacedNetwork"),
    (ReductionTuningDepth, "ReductionTuningDepth"),
    (StatusCode, "StatusCode"),
):
    _set_public_name(_obj, _name)


def __dir__() -> list[str]:
    return sorted(__all__)


del _obj, _name
