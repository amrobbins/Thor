"""PEP 517 wrapper that freezes Thor's CUDA stack before scikit-build-core."""

from __future__ import annotations

import os
from pathlib import Path
import sysconfig
from typing import Any

from scikit_build_core import build as _skbuild

from thor_build import compiler_environment, cuda_stack


def _with_cuda_requirements(requirements: list[str] | tuple[str, ...] | None) -> list[str]:
    merged = list(requirements or [])
    for requirement in cuda_stack.cuda_build_requirements():
        if requirement not in merged:
            merged.append(requirement)
    return merged



def _prepare_cuda_stack_for_wheel_build() -> None:
    compiler_environment.configure_host_compiler_environment()
    cuda_stack.configure_build_environment()


def get_requires_for_build_wheel(config_settings: dict[str, Any] | None = None) -> list[str]:
    return _with_cuda_requirements(_skbuild.get_requires_for_build_wheel(config_settings))


def get_requires_for_build_editable(config_settings: dict[str, Any] | None = None) -> list[str]:
    base = getattr(_skbuild, "get_requires_for_build_editable", lambda _settings=None: [])(config_settings)
    return _with_cuda_requirements(base)


def get_requires_for_build_sdist(config_settings: dict[str, Any] | None = None) -> list[str]:
    base = getattr(_skbuild, "get_requires_for_build_sdist", lambda _settings=None: [])(config_settings)
    return _with_cuda_requirements(base)


def prepare_metadata_for_build_wheel(metadata_directory: str, config_settings: dict[str, Any] | None = None) -> str:
    _prepare_cuda_stack_for_wheel_build()
    return _skbuild.prepare_metadata_for_build_wheel(metadata_directory, config_settings)


def _wheel_platform_tag() -> str:
    # Match the raw platform spelling used by scikit-build-core before an
    # auditwheel repair step (for example linux_x86_64).
    return sysconfig.get_platform().replace("-", "_").replace(".", "_")


def build_wheel(
    wheel_directory: str,
    config_settings: dict[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    _prepare_cuda_stack_for_wheel_build()

    # The public thor-cuda CMake build emits the architecture-specific kernel
    # wheels as sidecar release artifacts from the same native object graph.
    # PEP 517 only returns one filename, so this wrapper supplies CMake with the
    # frontend's wheel output directory while scikit-build-core still owns the
    # public thor-cuda wheel itself.
    previous_output_dir = os.environ.get("THOR_WHEEL_OUTPUT_DIR")
    previous_platform_tag = os.environ.get("THOR_WHEEL_PLATFORM_TAG")
    os.environ["THOR_WHEEL_OUTPUT_DIR"] = str(Path(wheel_directory).resolve())
    os.environ["THOR_WHEEL_PLATFORM_TAG"] = _wheel_platform_tag()
    try:
        return _skbuild.build_wheel(wheel_directory, config_settings, metadata_directory)
    finally:
        if previous_output_dir is None:
            os.environ.pop("THOR_WHEEL_OUTPUT_DIR", None)
        else:
            os.environ["THOR_WHEEL_OUTPUT_DIR"] = previous_output_dir
        if previous_platform_tag is None:
            os.environ.pop("THOR_WHEEL_PLATFORM_TAG", None)
        else:
            os.environ["THOR_WHEEL_PLATFORM_TAG"] = previous_platform_tag


def build_editable(
    wheel_directory: str,
    config_settings: dict[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    _prepare_cuda_stack_for_wheel_build()
    return _skbuild.build_editable(wheel_directory, config_settings, metadata_directory)


def build_sdist(sdist_directory: str, config_settings: dict[str, Any] | None = None) -> str:
    return _skbuild.build_sdist(sdist_directory, config_settings)
