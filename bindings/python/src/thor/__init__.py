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


def _set_env_from_first_existing_include(env_name: str, candidates: list[Path], sentinel: Path) -> None:
    if env_name in os.environ:
        return

    for candidate in candidates:
        if (candidate / sentinel).exists():
            os.environ.setdefault(env_name, str(candidate))
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
        Path("vector_types.h"),
    )

    # CUB is shipped as part of CCCL. CUDA 13 installs it into the unified
    # nvidia/cu13/include tree; older layouts may use nvidia/cuda_cccl/include.
    cub_include_candidates = [
        site_packages / "nvidia" / "cu13" / "include",
        site_packages / "nvidia" / "cuda_cccl" / "include",
    ]
    _set_env_from_first_existing_include(
        "THOR_CUDA_CCCL_INCLUDE_DIR",
        cub_include_candidates,
        Path("cub") / "cub.cuh",
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

    rtld_global = getattr(os, "RTLD_GLOBAL", 0)

    def preload(path: Path) -> None:
        try:
            ctypes.CDLL(str(path), mode=rtld_global)
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
    if "THOR_CUDNN_FRONTEND_INCLUDE_DIR" in os.environ:
        return

    site_packages = _find_site_packages_root()
    candidate = site_packages / "include"

    if (candidate / "cudnn_frontend.h").exists():
        os.environ.setdefault("THOR_CUDNN_FRONTEND_INCLUDE_DIR", str(candidate))


_configure_cuda_include_dirs()
_configure_cudnn_frontend_include_dir()
_preload_cuda_user_space_libs()

from ._thor import *
from ._thor import __version__, __git_version__

# Keep ``thor.losses`` as a Python package so organized domain namespaces such
# as ``thor.losses.ranking`` are available after a plain ``import thor``.
from . import losses as losses
