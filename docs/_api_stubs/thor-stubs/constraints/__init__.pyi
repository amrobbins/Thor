"""Parameter constraint namespace."""



class ParameterConstraint:
    @property
    def constraint_type(self) -> str: ...

    def get_architecture_json(self) -> str: ...

class NonNegative(ParameterConstraint):
    """
    Post-update parameter constraint that clips parameter values to be non-negative.

    This is a general Thor parameter constraint, not a layer-specific hack. It can be
    attached to any trainable ParameterSpecification or to layer builders that expose
    parameter-specific constraint arguments.
    """

    def __init__(self) -> None: ...

class NonPositive(ParameterConstraint):
    """
    Post-update parameter constraint that clips parameter values to be non-positive.
    """

    def __init__(self) -> None: ...

class Min(ParameterConstraint):
    """
    Post-update parameter constraint that clips parameter values to be at least min_value.
    """

    def __init__(self, min_value: float) -> None: ...

    @property
    def min_value(self) -> float: ...

class Max(ParameterConstraint):
    """
    Post-update parameter constraint that clips parameter values to be at most max_value.
    """

    def __init__(self, max_value: float) -> None: ...

    @property
    def max_value(self) -> float: ...

class MinMax(ParameterConstraint):
    """
    Post-update parameter constraint that clips parameter values into [min_value, max_value].
    """

    def __init__(self, min_value: float, max_value: float) -> None: ...

    @property
    def min_value(self) -> float: ...

    @property
    def max_value(self) -> float: ...

__all__: list = ['ParameterConstraint', 'NonNegative', 'NonPositive', 'Min', 'Max', 'MinMax']

def __dir__() -> list[str]: ...
