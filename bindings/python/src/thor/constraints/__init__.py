"""Parameter constraint namespace."""

from __future__ import annotations

from .. import _thor as _thor

ParameterConstraint = _thor.ParameterConstraint
NonNegative = _thor.NonNegativeParameterConstraint
NonPositive = _thor.NonPositiveParameterConstraint
Min = _thor.MinParameterConstraint
Max = _thor.MaxParameterConstraint
MinMax = _thor.MinMaxParameterConstraint

__all__ = [
    "ParameterConstraint",
    "NonNegative",
    "NonPositive",
    "Min",
    "Max",
    "MinMax",
]


def _set_public_name(obj: object, name: str) -> None:
    for attr, value in (("__module__", __name__), ("__qualname__", name), ("__name__", name)):
        try:
            setattr(obj, attr, value)
        except (AttributeError, TypeError):
            pass


for _obj, _name in (
    (ParameterConstraint, "ParameterConstraint"),
    (NonNegative, "NonNegative"),
    (NonPositive, "NonPositive"),
    (Min, "Min"),
    (Max, "Max"),
    (MinMax, "MinMax"),
):
    _set_public_name(_obj, _name)


def __dir__() -> list[str]:
    return sorted(__all__)


del _obj, _name
