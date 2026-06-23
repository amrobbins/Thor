"""Public Thor Python API."""

from __future__ import annotations

from . import _bootstrap as _bootstrap

_bootstrap.configure()

from ._thor import DataType, Network, Tensor
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

__all__ = [
    "DataType",
    "EnsembleModel",
    "Network",
    "Tensor",
    "__git_version__",
    "__version__",
    "activations",
    "constraints",
    "data",
    "ensembles",
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
