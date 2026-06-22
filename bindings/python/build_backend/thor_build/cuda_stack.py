"""Build-time CUDA stack resolver for Thor wheels.

The source tree selects a CUDA family such as ``13.3``.  During wheel builds,
PEP 517 installs broad CUDA-family build requirements, this module records the
exact installed NVIDIA component versions, and the wheel metadata pins those
exact versions.  Runtime import code must consume the generated resolved
manifest and must not re-resolve newer package versions for an existing wheel.
"""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
from importlib import metadata
import importlib.util
import os
from pathlib import Path
import shlex
import sys
from typing import Iterable, Sequence


_REPO_PYTHON_ROOT = Path(__file__).resolve().parents[2]
_THOR_SRC_ROOT = _REPO_PYTHON_ROOT / "src" / "thor"
_SOURCE_STACK_PATH = _THOR_SRC_ROOT / "_cuda_stack.py"
_RESOLVED_STACK_PATH = _THOR_SRC_ROOT / "_cuda_stack_resolved.py"


def _load_source_stack_module():
    spec = importlib.util.spec_from_file_location("_thor_source_cuda_stack", _SOURCE_STACK_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Thor CUDA stack source config from {_SOURCE_STACK_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_SOURCE_STACK = _load_source_stack_module()


@dataclass(frozen=True)
class ResolvedDistribution:
    name: str
    version: str


def cuda_build_requirements() -> list[str]:
    return list(_SOURCE_STACK.cuda_build_requirements())


def base_dependencies() -> list[str]:
    return list(_SOURCE_STACK.BASE_DEPENDENCIES)


def _distribution_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError as error:
        requirements = ", ".join(cuda_build_requirements())
        raise RuntimeError(
            f"Thor CUDA stack resolution failed: required build distribution {name!r} is not installed. "
            f"Build the wheel in an isolated PEP 517 environment so these requirements are installed: {requirements}"
        ) from error


def resolve_cuda_stack() -> tuple[ResolvedDistribution, ...]:
    return tuple(
        ResolvedDistribution(name, _distribution_version(name))
        for name in _SOURCE_STACK.resolved_distribution_names()
    )


def exact_cuda_dependency_pins(distributions: Iterable[ResolvedDistribution] | None = None) -> list[str]:
    if distributions is None:
        distributions = resolve_cuda_stack()
    return [f"{dist.name}=={dist.version}" for dist in distributions]


def runtime_dependencies() -> list[str]:
    distributions = resolve_cuda_stack()
    write_resolved_cuda_stack(distributions)
    return base_dependencies() + exact_cuda_dependency_pins(distributions)


def _render_tuple(items: Iterable[str], indent: str = "        ") -> str:
    return "\n".join(f'{indent}"{item}",' for item in items)


def _render_patterns(patterns: tuple[str, ...]) -> str:
    return "".join(f'"{pattern}", ' for pattern in patterns)


def write_resolved_cuda_stack(distributions: Iterable[ResolvedDistribution] | None = None) -> Path:
    if distributions is None:
        distributions = resolve_cuda_stack()
    distributions = tuple(distributions)

    include_lines = "\n".join(
        f'        IncludeSpec("{spec.env_name}", "{spec.distribution}", Path({str(spec.sentinel)!r})),'
        for spec in _SOURCE_STACK.INCLUDE_SPECS
    )
    library_lines = "\n".join(
        f'        LibrarySpec("{spec.distribution}", ({_render_patterns(spec.patterns)})),'
        for spec in _SOURCE_STACK.LIBRARY_SPECS
    )
    distribution_lines = "\n".join(
        f'        CudaDistribution("{dist.name}", "{dist.version}"),' for dist in distributions
    )

    text = f'''"""Resolved CUDA stack frozen into this Thor wheel.

This file is generated during wheel builds from ``thor._cuda_stack`` after the
build environment has installed the latest compatible NVIDIA CUDA dependencies
for the selected CUDA family.  Import-time code must use this exact manifest and
must not resolve newer package versions for an already-built wheel.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


CUDA_VERSION = "{_SOURCE_STACK.CUDA_VERSION}"
CUDA_STACK_NAME = "CUDA {_SOURCE_STACK.CUDA_VERSION}"


@dataclass(frozen=True)
class CudaDistribution:
    name: str
    version: str


@dataclass(frozen=True)
class IncludeSpec:
    env_name: str
    distribution: str
    sentinel: Path


@dataclass(frozen=True)
class LibrarySpec:
    distribution: str
    patterns: tuple[str, ...]


@dataclass(frozen=True)
class CudaStack:
    cuda_version: str
    name: str
    distributions: tuple[CudaDistribution, ...]
    includes: tuple[IncludeSpec, ...]
    libraries: tuple[LibrarySpec, ...]


CUDA_STACK = CudaStack(
    cuda_version=CUDA_VERSION,
    name=CUDA_STACK_NAME,
    distributions=(
{distribution_lines}
    ),
    includes=(
{include_lines}
    ),
    libraries=(
{library_lines}
    ),
)


def dependency_specifiers() -> tuple[str, ...]:
    """Return exact PEP 508 dependency pins for this built wheel."""

    return tuple(f"{{dist.name}}=={{dist.version}}" for dist in CUDA_STACK.distributions)
'''
    _RESOLVED_STACK_PATH.write_text(text)
    return _RESOLVED_STACK_PATH


def _dedupe_existing(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        if not path.exists():
            continue
        try:
            key = str(path.resolve())
        except OSError:
            key = str(path)
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result


def _distribution(name: str) -> metadata.Distribution:
    try:
        return metadata.distribution(name)
    except metadata.PackageNotFoundError as error:
        raise RuntimeError(f"Thor CUDA stack resolution failed: distribution {name!r} is not installed") from error


def _distribution_files_matching(distribution_name: str, patterns: Sequence[str]) -> list[Path]:
    dist = _distribution(distribution_name)
    if dist.files is None:
        raise RuntimeError(f"Thor CUDA stack resolution failed: distribution {distribution_name!r} has no file metadata")

    matches: list[Path] = []
    for file in dist.files:
        relative_path = str(file).replace("\\", "/")
        if any(fnmatch(relative_path, pattern) for pattern in patterns):
            matches.append(Path(dist.locate_file(file)))
    return _dedupe_existing(matches)


def _resolve_one_distribution_file(distribution_name: str, patterns: Sequence[str], description: str) -> Path:
    matches = _distribution_files_matching(distribution_name, patterns)
    if len(matches) != 1:
        rendered_patterns = ", ".join(patterns)
        if not matches:
            raise RuntimeError(
                f"Thor CUDA stack resolution failed: {distribution_name!r} is missing required "
                f"{description} matching {rendered_patterns!r}"
            )
        rendered_matches = ", ".join(str(path) for path in matches)
        raise RuntimeError(
            f"Thor CUDA stack resolution failed: {distribution_name!r} has ambiguous {description} "
            f"for patterns {rendered_patterns!r}: {rendered_matches}"
        )
    return matches[0]


def _distribution_root_containing(distribution_name: str, sentinel: Path) -> Path:
    sentinel_posix = sentinel.as_posix()
    sentinel_depth = len(sentinel.parts)
    sentinel_path = _resolve_one_distribution_file(
        distribution_name,
        (f"**/{sentinel_posix}", sentinel_posix),
        f"include sentinel {sentinel_posix}",
    )

    root = sentinel_path
    for _ in range(sentinel_depth):
        root = root.parent
    return root


def _library_path(distribution_name: str, patterns: Sequence[str]) -> Path:
    return _resolve_one_distribution_file(distribution_name, patterns, "shared library")


def resolved_include_roots() -> dict[str, Path]:
    return {
        spec.env_name: _distribution_root_containing(spec.distribution, spec.sentinel)
        for spec in _SOURCE_STACK.INCLUDE_SPECS
    }


def resolved_library_paths() -> list[Path]:
    return [_library_path(spec.distribution, spec.patterns) for spec in _SOURCE_STACK.LIBRARY_SPECS]


def _set_env_exact(env_name: str, value: Path | str) -> None:
    value_str = str(value)
    existing = os.environ.get(env_name)
    if existing is not None and existing != value_str:
        raise RuntimeError(
            f"Thor CUDA stack build environment conflict: {env_name} is set to {existing!r}, "
            f"but the resolved Thor CUDA stack requires {value_str!r}"
        )
    os.environ[env_name] = value_str


def _append_unique_path_env(env_name: str, paths: Iterable[Path]) -> None:
    existing = [part for part in os.environ.get(env_name, "").split(os.pathsep) if part]
    seen = set(existing)
    merged = existing[:]
    for path in paths:
        path_str = str(path)
        if path_str in seen:
            continue
        seen.add(path_str)
        merged.append(path_str)
    if merged:
        os.environ[env_name] = os.pathsep.join(merged)


def _append_cmake_args(definitions: dict[str, str]) -> None:
    existing = os.environ.get("CMAKE_ARGS", "")
    appended = " ".join(f"-D{name}={shlex.quote(value)}" for name, value in definitions.items())
    os.environ["CMAKE_ARGS"] = f"{existing} {appended}".strip()


def configure_build_environment() -> None:
    """Freeze the CUDA stack and point the wheel CMake build at it.

    The manylinux image still provides nvcc.  The wheel build should use the
    Python NVIDIA component wheels for headers and link/runtime libraries so the
    generated manifest, wheel metadata, and native binary agree on the CUDA user
    space stack.
    """

    distributions = resolve_cuda_stack()
    write_resolved_cuda_stack(distributions)

    include_roots = resolved_include_roots()
    library_paths = resolved_library_paths()
    library_dirs = tuple(dict.fromkeys(path.parent for path in library_paths))

    for env_name, include_root in include_roots.items():
        _set_env_exact(env_name, include_root)

    cudnn_library = _library_path("nvidia-cudnn-cu13", ("**/libcudnn.so.9",))
    cuda_include = include_roots["THOR_CUDA_INCLUDE_DIR"]
    cudatoolkit_root = cuda_include.parent

    _append_unique_path_env("LD_LIBRARY_PATH", library_dirs)
    _append_unique_path_env("LIBRARY_PATH", library_dirs)
    _append_unique_path_env("CMAKE_LIBRARY_PATH", library_dirs)

    _append_cmake_args(
        {
            "CUDAToolkit_ROOT": str(cudatoolkit_root),
            "THOR_CUDA_CCCL_INCLUDE_DIR": str(include_roots["THOR_CUDA_CCCL_INCLUDE_DIR"]),
            "THOR_CUDNN_LIBRARY": str(cudnn_library),
            "THOR_CUDNN_FRONTEND_INCLUDE_DIR": str(include_roots["THOR_CUDNN_FRONTEND_INCLUDE_DIR"]),
        }
    )
