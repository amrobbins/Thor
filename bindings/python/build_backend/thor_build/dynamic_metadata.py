"""scikit-build-core dynamic metadata provider for Thor dependencies."""

from __future__ import annotations

from typing import Any, Mapping

from . import cuda_stack


def dynamic_metadata(
    field: str,
    settings: Mapping[str, Any] | None = None,
    project: Mapping[str, Any] | None = None,
) -> list[str]:
    if field != "dependencies":
        raise RuntimeError(f"Thor dynamic metadata provider only supports dependencies, got {field!r}")
    if settings:
        raise RuntimeError(f"Thor dynamic dependencies provider does not accept settings: {sorted(settings)}")
    return cuda_stack.runtime_dependencies()


def get_requires_for_dynamic_metadata(_settings: Mapping[str, Any] | None = None) -> list[str]:
    # scikit-build-core may call this before metadata generation.  Returning the
    # broad build-time CUDA requirements keeps the provider usable if the custom
    # Thor PEP 517 wrapper is ever refactored.
    return cuda_stack.cuda_build_requirements()
