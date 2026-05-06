"""
Thor Python package.

This wraps the native nanobind extension `thor._thor` and re-exports its public API.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path


def _find_site_packages_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _configure_cuda_include_dir() -> None:
    if "THOR_CUDA_INCLUDE_DIR" in os.environ:
        return

    here = Path(__file__).resolve()
    site_packages = _find_site_packages_root()

    candidates = [
        site_packages / "nvidia" / "cu13" / "include",
        site_packages / "nvidia" / "cuda_runtime" / "include",
    ]

    for candidate in candidates:
        if (candidate / "vector_types.h").exists():
            os.environ.setdefault("THOR_CUDA_INCLUDE_DIR", str(candidate))
            return


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
        "libnvrtc-builtins.so.13.2",
        "libcudnn.so.9",
    ]

    rtld_global = getattr(os, "RTLD_GLOBAL", 0)

    for lib_dir in lib_dirs:
        if not lib_dir.is_dir():
            continue
        for lib in libs:
            p = lib_dir / lib
            if p.exists():
                try:
                    ctypes.CDLL(str(p), mode=rtld_global)
                except OSError:
                    pass


_configure_cuda_include_dir()
_preload_cuda_user_space_libs()

from ._thor import *
from ._thor import __version__, __git_version__
