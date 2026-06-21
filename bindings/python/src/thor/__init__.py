"""Public Thor Python API."""

from __future__ import annotations

import ctypes as _ctypes
import os as _os
from pathlib import Path as _Path


def _find_site_packages_root() -> _Path:
    return _Path(__file__).resolve().parents[1]


def _set_env_from_first_existing_include(env_name: str, candidates: list[_Path], sentinel: _Path) -> None:
    if env_name in _os.environ:
        return

    for candidate in candidates:
        if (candidate / sentinel).exists():
            _os.environ.setdefault(env_name, str(candidate))
            return


def _configure_cuda_include_dirs() -> None:
    site_packages = _find_site_packages_root()

    # CUDA 13 toolkit wheels use the unified nvidia/cu13/include layout.
    # Older/split wheels may still expose component-specific include roots.
    cuda_include_candidates = [
        site_packages / "nvidia" / "cu13" / "include",
        site_packages / "nvidia" / "cuda_runtime" / "include",
    ]
    _set_env_from_first_existing_include(
        "THOR_CUDA_INCLUDE_DIR",
        cuda_include_candidates,
        _Path("vector_types.h"),
    )

    # CUB is shipped as part of CCCL. Current CUDA 13 toolkit wheels expose
    # CUB under nvidia/cu13/include/cccl. Keep the unified include root and
    # older split-wheel layout as fallbacks.
    cub_include_candidates = [
        site_packages / "nvidia" / "cu13" / "include" / "cccl",
        site_packages / "nvidia" / "cu13" / "include",
        site_packages / "nvidia" / "cuda_cccl" / "include",
    ]
    _set_env_from_first_existing_include(
        "THOR_CUDA_CCCL_INCLUDE_DIR",
        cub_include_candidates,
        _Path("cub") / "cub.cuh",
    )


def _preload_cuda_user_space_libs() -> None:
    site_packages = _find_site_packages_root()

    lib_dirs = [
        site_packages / "nvidia" / "cu13" / "lib",
        site_packages / "nvidia" / "cudnn" / "lib",
    ]

    libs = [
        "libcudart.so.13",
        "libnvrtc.so.13",
        "libnvJitLink.so.13",
        "libcublas.so.13",
        "libcublasLt.so.13",
        "libcusolver.so.12",
        "libcusparse.so.12",
        "libcudnn.so.9",
    ]
    lib_globs = [
        "libnvrtc-builtins.so.13.*",
    ]

    rtld_global = getattr(_os, "RTLD_GLOBAL", 0)

    def preload(path: _Path) -> None:
        try:
            _ctypes.CDLL(str(path), mode=rtld_global)
        except OSError:
            pass

    for lib_dir in lib_dirs:
        if not lib_dir.is_dir():
            continue
        for lib in libs:
            p = lib_dir / lib
            if p.exists():
                preload(p)
        for pattern in lib_globs:
            for p in sorted(lib_dir.glob(pattern)):
                preload(p)


def _configure_cudnn_frontend_include_dir() -> None:
    if "THOR_CUDNN_FRONTEND_INCLUDE_DIR" in _os.environ:
        return

    site_packages = _find_site_packages_root()
    candidate = site_packages / "include"

    if (candidate / "cudnn_frontend.h").exists():
        _os.environ.setdefault("THOR_CUDNN_FRONTEND_INCLUDE_DIR", str(candidate))


_configure_cuda_include_dirs()
_configure_cudnn_frontend_include_dir()
_preload_cuda_user_space_libs()

from ._thor import DataType, Network, Tensor
from ._thor import __git_version__, __version__

from . import activations as activations
from . import constraints as constraints
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
    "Network",
    "Tensor",
    "__git_version__",
    "__version__",
    "activations",
    "constraints",
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
