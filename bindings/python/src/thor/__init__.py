"""
Thor Python package.

This wraps the native nanobind extension `thor._thor` and re-exports its public API.
"""

from __future__ import annotations

from ._thor import *
from ._thor import __version__, __git_version__

# from . import _thor as _native
#
# # Expose native submodules at the top-level (keeps the old UX: thor.layers, thor.activations, etc.)
# activations = _native.activations
# initializers = _native.initializers
# layers = _native.layers
# losses = _native.losses
# metrics = _native.metrics
# optimizers = _native.optimizers
#
# del _native

# Make `import thor.layers` work too (not just attribute access).
# import sys as _sys
# _sys.modules[__name__ + ".activations"] = activations
# _sys.modules[__name__ + ".initializers"] = initializers
# _sys.modules[__name__ + ".layers"] = layers
# _sys.modules[__name__ + ".losses"] = losses
# _sys.modules[__name__ + ".metrics"] = metrics
# _sys.modules[__name__ + ".optimizers"] = optimizers
# del _sys
