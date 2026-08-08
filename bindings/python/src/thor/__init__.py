"""Public Thor Python API."""

from __future__ import annotations

from . import _bootstrap as _bootstrap

_bootstrap.configure()

from ._thor import (
    BATCH_VALIDITY_MASK_NAME,
    METRIC_AGGREGATION_DENOMINATOR_NAME,
    METRIC_AGGREGATION_NUMERATOR_NAME,
    DataType,
    MetricAggregation,
    Network,
    RaggedTensor,
    Tensor,
    _infer_network_for_tensors,
)


class _NetworkLoadDescriptor:
    """Dual-use Network.load descriptor.

    Nanobind does not allow a static method and an instance method to share the
    same overload name.  Keep the public API ergonomic by dispatching
    ``thor.Network.load(path, network_name=...)`` to the native static loader and
    ``network.load(path)`` to the native in-place loader.
    """

    def __get__(self, instance, owner=None):
        if instance is None:
            return Network._load_from_path
        return instance._load_in_place


Network.load = _NetworkLoadDescriptor()

from .ensembles import EnsembleModel
from ._thor import __git_version__, __version__

from . import activations as activations
from . import constraints as constraints
from . import data as data
from . import ensembles as ensembles
from . import initializers as initializers
from . import layers as layers
from . import losses as losses
from . import metrics as metrics
from . import optimizers as optimizers
from . import parameters as parameters
from . import physical as physical
from . import random as random
from . import runtime as runtime
from . import training as training


def einsum(
    equation: str,
    *operands: Tensor,
    network: Network | None = None,
) -> Tensor:
    """Create a symbolic einsum operation and return its output tensor.

    The equation describes feature dimensions only; Thor's batch dimension is
    implicit and is preserved. The operation owns no trainable parameters and
    remains differentiable with respect to every operand on a live gradient path.

    When ``network`` is omitted, Thor infers the unique live Python-created
    network that contains every operand. Pass ``network=...`` explicitly when
    working with ambiguous loaded/cloned networks or tensors that are not yet
    associated with a network.
    """
    if not isinstance(equation, str):
        raise TypeError(f"thor.einsum() equation must be str, got {type(equation).__name__}")
    if not operands:
        raise ValueError("thor.einsum() requires at least one operand.")
    if any(not isinstance(operand, Tensor) for operand in operands):
        bad_index = next(i for i, operand in enumerate(operands) if not isinstance(operand, Tensor))
        raise TypeError(
            f"thor.einsum() operand[{bad_index}] must be thor.Tensor, "
            f"got {type(operands[bad_index]).__name__}"
        )
    if network is not None and not isinstance(network, Network):
        raise TypeError(f"thor.einsum() network must be thor.Network or None, got {type(network).__name__}")

    resolved_network = network if network is not None else _infer_network_for_tensors(list(operands))
    layer = layers.Einsum(resolved_network, equation, list(operands))
    return layer.get_feature_output()


__all__ = [
    "BATCH_VALIDITY_MASK_NAME",
    "METRIC_AGGREGATION_DENOMINATOR_NAME",
    "METRIC_AGGREGATION_NUMERATOR_NAME",
    "DataType",
    "MetricAggregation",
    "EnsembleModel",
    "Network",
    "RaggedTensor",
    "Tensor",
    "__git_version__",
    "__version__",
    "activations",
    "constraints",
    "data",
    "ensembles",
    "einsum",
    "initializers",
    "layers",
    "losses",
    "metrics",
    "optimizers",
    "parameters",
    "physical",
    "random",
    "runtime",
    "training",
]


def __dir__() -> list[str]:
    return sorted(__all__)


# Hide the native implementation module from the package namespace after all
# public wrapper modules have bound the native symbols they need.
try:
    del _thor
except NameError:
    pass
