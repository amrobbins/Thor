"""Kernel-wheel PEP 517 shim reusing Thor's shared wheel backend."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any


_PYTHON_BINDINGS_ROOT = Path(__file__).resolve().parents[3]
_SHARED_BACKEND_ROOT = _PYTHON_BINDINGS_ROOT / "build_backend"
if str(_SHARED_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_SHARED_BACKEND_ROOT))

from thor_build_backend import (  # noqa: E402
    build_wheel as _build_wheel,
    get_requires_for_build_wheel as _get_requires_for_build_wheel,
    prepare_metadata_for_build_wheel as _prepare_metadata_for_build_wheel,
)


def get_requires_for_build_wheel(config_settings: dict[str, Any] | None = None) -> list[str]:
    return _get_requires_for_build_wheel(config_settings)


def prepare_metadata_for_build_wheel(
    metadata_directory: str,
    config_settings: dict[str, Any] | None = None,
) -> str:
    return _prepare_metadata_for_build_wheel(metadata_directory, config_settings)


def build_wheel(
    wheel_directory: str,
    config_settings: dict[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    return _build_wheel(wheel_directory, config_settings, metadata_directory)


def get_requires_for_build_sdist(config_settings: dict[str, Any] | None = None) -> list[str]:
    return []


def build_sdist(sdist_directory: str, config_settings: dict[str, Any] | None = None) -> str:
    raise RuntimeError(
        "Thor CUDA kernel distributions are binary-only release artifacts; build them with --wheel."
    )
