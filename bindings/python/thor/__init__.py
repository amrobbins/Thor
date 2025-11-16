"""
Python wrapper package for the Thor C++/CUDA deep learning framework.
"""

from . import _thor as _core

# Public-facing version info
__version__ = _core.__version__
__git_version__ = _core.__git_version__

version = _core.version
git_version = _core.git_version

__all__ = [
    "version",
    "git_version",
]