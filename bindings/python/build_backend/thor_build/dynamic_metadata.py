"""scikit-build-core dynamic metadata provider for Thor dependencies."""

from __future__ import annotations

from typing import Any, Mapping

from . import cuda_stack, version_metadata


_KERNEL_WHEEL_DISTRIBUTIONS = (
    "thor-cuda-kernels-sm89",
    "thor-cuda-kernels-sm120",
)


def kernel_wheel_dependencies(version: str | None = None) -> list[str]:
    """Return exact-version dependencies for every supported Thor kernel wheel."""

    resolved_version = version if version is not None else version_metadata.thor_version()
    return [f"{distribution}=={resolved_version}" for distribution in _KERNEL_WHEEL_DISTRIBUTIONS]


def dynamic_metadata(
    field: str,
    settings: Mapping[str, Any] | None = None,
    project: Mapping[str, Any] | None = None,
) -> list[str]:
    if field != "dependencies":
        raise RuntimeError(f"Thor dynamic metadata provider only supports dependencies, got {field!r}")
    if settings:
        raise RuntimeError(f"Thor dynamic dependencies provider does not accept settings: {sorted(settings)}")
    return cuda_stack.runtime_dependencies() + kernel_wheel_dependencies()


def get_requires_for_dynamic_metadata(_settings: Mapping[str, Any] | None = None) -> list[str]:
    # scikit-build-core may call this before metadata generation.  Returning the
    # broad build-time CUDA requirements keeps the provider usable if the custom
    # Thor PEP 517 wrapper is ever refactored.
    return cuda_stack.cuda_build_requirements()
