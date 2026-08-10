"""Parameter specification and reference namespace."""

from collections.abc import Sequence
from typing import TypeAlias, overload

import thor
import thor.constraints
import thor.initializers
import thor.optimizers
import thor.physical


class BoundParameter:
    @property
    def name(self) -> str: ...

    @property
    def trainable(self) -> bool: ...

    def is_trainable(self) -> bool: ...

    def is_training_enabled(self) -> bool: ...

    def set_training_enabled(self, enabled: bool) -> None: ...

    def has_optimizer(self) -> bool: ...

class ParameterReference:
    def __init__(self, parameterizable_id: int, parameter_name: str) -> None: ...

    @property
    def parameterizable_id(self) -> int: ...

    @property
    def parameter_name(self) -> str: ...

    def is_initialized(self) -> bool: ...

    def get_architecture_json(self) -> str: ...

    def __eq__(self, arg: ParameterReference, /) -> bool: ...

class ParameterSpecification:
    @overload
    def __init__(self, name: str, shape: Sequence[int], dtype: thor.DataType = thor.DataType.fp32, initializer: thor.initializers.Initializer | None = None, trainable: bool = True, optimizer: thor.optimizers.Optimizer | None = None, training_initially_enabled: bool | None = None, constraints: object | None = None) -> None:
        """
        Create an API parameter with storage attributes determined at parameter definition time.

        Provide:
        - ``shape``: the parameter shape
        - ``dtype``: the parameter dtype, default ``fp32``

        This form is for statically-shaped parameters. For compile-time-dynamic parameters, use
        ``create_storage_from_context=...`` instead.
        """

    @overload
    def __init__(self, name: str, create_storage_from_context: object, initializer: thor.initializers.Initializer | None = None, trainable: bool = True, optimizer_override: thor.optimizers.Optimizer | None = None, training_initially_enabled: bool | None = None, constraints: object | None = None) -> None:
        """
        Create an API parameter whose implementation storage is allocated at physical layer compile time.

        Provide ``create_storage_from_context``. For single-input layers, the default feature input name is
        ``"feature_input"``, and ``ParameterSpecification.StorageContext.get_feature_input()`` returns that tensor when
        exactly one input is present.
        """

    StorageContext: TypeAlias = ParameterSpecification.StorageContext

    @staticmethod
    def allocate_storage(input_tensor: thor.physical.PhysicalTensor, shape: Sequence[int], dtype: thor.DataType) -> thor.physical.PhysicalTensor:
        """
        Allocate implementation storage on the same placement as ``input_tensor`` with the requested shape and dtype.
        """

    @property
    def name(self) -> str: ...

    @property
    def trainable(self) -> bool: ...

    def is_trainable(self) -> bool: ...

    def is_training_initially_enabled(self) -> bool: ...

    def has_optimizer(self) -> bool: ...

    def get_architecture_json(self) -> str: ...

    def has_constraints(self) -> bool: ...

    def get_constraints(self) -> list[thor.constraints.ParameterConstraint]: ...

__all__: list = ['BoundParameter', 'ParameterReference', 'ParameterSpecification']

def __dir__() -> list[str]: ...
