"""Runtime cache location for NVRTC bundled CUDA headers."""

from __future__ import annotations

import os as _os
from pathlib import Path as _Path
import tempfile as _tempfile

_ENV_NAME = "THOR_NVRTC_BUNDLED_HEADERS_DIR"


def _default_cache_root() -> _Path:
    xdg_cache_home = _os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        return _Path(xdg_cache_home)

    try:
        return _Path.home() / ".cache"
    except RuntimeError:
        return _Path(_tempfile.gettempdir())


def nvrtc_bundled_headers_dir(cuda_version: str) -> _Path:
    """Return the directory NVRTC should use for bundled CUDA headers.

    CUDA 13.3+ ships CUDA Runtime and CCCL headers inside
    ``libnvrtc-builtins``.  Thor passes this directory to NVRTC via
    ``--use-bundled-headers=<path>``; NVRTC installs the matching headers there
    on demand and reuses them for later JIT compilations.
    """

    override = _os.environ.get(_ENV_NAME)
    if override:
        return _Path(override)

    return _default_cache_root() / "thor" / "nvrtc_bundled_headers" / f"cuda-{cuda_version}"


__all__ = ["nvrtc_bundled_headers_dir"]
