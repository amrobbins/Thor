import enum


class Initializer:
    pass

class Glorot(Initializer):
    """
    A Glorot (Xavier) initializer.

    Draws each weight from a uniform (or normal) distribution. See: Glorot.Mode.

    References:
    X. Glorot and Y. Bengio, “Understanding the difficulty of training deep feedforward neural networks,”
    AISTATS 2010. https://proceedings.mlr.press/v9/glorot10a.html
    """

    def __init__(self, mode: Glorot.Mode = Glorot.Mode.UNIFORM) -> None:
        """Initialize a Glorot initializer (construction happens in __new__)."""

    class Mode(enum.Enum):
        NORMAL = 8

        UNIFORM = 7

class UniformRandom(Initializer):
    """
    A uniform random initializer.

    Draws each weight from a uniform distribution:

        U[min_value, max_value]

    Where min_value <= max_value. When min_value == max_value, the constant is written to each weight.
    """

    def __init__(self, min_value: float, max_value: float) -> None:
        """Initialize a UniformRandom initializer."""
