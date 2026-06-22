"""PEP 517 wrapper that freezes Thor's CUDA stack before scikit-build-core."""

from __future__ import annotations

from typing import Any

from scikit_build_core import build as _skbuild

from thor_build import cuda_stack


def _with_cuda_requirements(requirements: list[str] | tuple[str, ...] | None) -> list[str]:
    merged = list(requirements or [])
    for requirement in cuda_stack.cuda_build_requirements():
        if requirement not in merged:
            merged.append(requirement)
    return merged


def _prepare_cuda_stack_for_wheel_build() -> None:
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


def build_wheel(
    wheel_directory: str,
    config_settings: dict[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    _prepare_cuda_stack_for_wheel_build()
    return _skbuild.build_wheel(wheel_directory, config_settings, metadata_directory)


def build_editable(
    wheel_directory: str,
    config_settings: dict[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    _prepare_cuda_stack_for_wheel_build()
    return _skbuild.build_editable(wheel_directory, config_settings, metadata_directory)


def build_sdist(sdist_directory: str, config_settings: dict[str, Any] | None = None) -> str:
    return _skbuild.build_sdist(sdist_directory, config_settings)
