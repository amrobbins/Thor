"""Parameter specification and reference namespace."""

from __future__ import annotations

from .. import _thor as _thor

BoundParameter = _thor.BoundParameter
ParameterReference = _thor.ParameterReference
ParameterSpecification = _thor.ParameterSpecification

__all__ = [
    "BoundParameter",
    "ParameterReference",
    "ParameterSpecification",
]


def _set_public_name(obj: object, name: str) -> None:
    for attr, value in (("__module__", __name__), ("__qualname__", name), ("__name__", name)):
        try:
            setattr(obj, attr, value)
        except (AttributeError, TypeError):
            pass


for _obj, _name in (
    (BoundParameter, "BoundParameter"),
    (ParameterReference, "ParameterReference"),
    (ParameterSpecification, "ParameterSpecification"),
):
    _set_public_name(_obj, _name)

_set_public_name(ParameterSpecification.StorageContext, "ParameterSpecification.StorageContext")


def __dir__() -> list[str]:
    return sorted(__all__)


del _obj, _name
