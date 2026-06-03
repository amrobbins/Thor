from __future__ import annotations

import typing as _typing

from .._thor.layers import *  # noqa: F401,F403
from .._thor.physical import CudaKernelExpression as _CudaKernelExpression


class CudaKernelLayer(CustomLayer):
    """Convenience wrapper for using a CudaKernelExpression as a CustomLayer.

    This class intentionally does not create a separate serialized layer kind.
    It lowers directly to CustomLayer with ``kernel.as_dynamic_expression()``, so
    the existing CUDA-kernel source inspection and save/load key policy continues
    to apply without a second security path.
    """

    def __init__(
        self,
        network: _typing.Any,
        inputs: _typing.Any,
        kernel: _CudaKernelExpression,
        output_names: str | list[str] | tuple[str, ...] | None = None,
        parameters: _typing.Any = None,
        optimizer: _typing.Any = None,
    ) -> None:
        if not isinstance(kernel, _CudaKernelExpression):
            raise TypeError("CudaKernelLayer kernel must be a thor.physical.CudaKernelExpression.")

        if output_names is None:
            output_names = kernel.output_names()

        super().__init__(
            network=network,
            inputs=inputs,
            output_names=output_names,
            build=kernel.as_dynamic_expression(),
            parameters=parameters,
            optimizer=optimizer,
        )
