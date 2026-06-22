"""Import-time CUDA dependency bootstrap for the Thor Python package."""

from __future__ import annotations

import ctypes as _ctypes
from fnmatch import fnmatch as _fnmatch
from importlib import metadata as _metadata
import os as _os
from pathlib import Path as _Path
import sys as _sys
import re as _re
from typing import Iterable as _Iterable
from typing import Sequence as _Sequence

from ._cuda_stack import CUDA_VERSION as _SOURCE_CUDA_VERSION
from ._cuda_stack_resolved import CUDA_STACK as _CUDA_STACK
from ._cuda_stack_resolved import LibrarySpec as _LibrarySpec


class CudaBootstrapError(ImportError):
    """Raised when Thor's selected CUDA user-space dependency stack is unavailable."""


_CUDA_DISTRIBUTION_SPECIFIERS = {dist.name: f"=={dist.version}" for dist in _CUDA_STACK.distributions}
_CUDA_DISTRIBUTIONS = _CUDA_STACK.distributions
_INCLUDE_SPECS = _CUDA_STACK.includes
_LIBRARY_SPECS: tuple[_LibrarySpec, ...] = _CUDA_STACK.libraries

_configured = False

_SOURCE_TREE_BOOTSTRAP_ENV = "THOR_CUDA_BOOTSTRAP_SOURCE_TREE"


_VERSION_PART_RE = _re.compile(r"^(\d+)")
_SPECIFIER_RE = _re.compile(r"^(==|!=|<=|>=|<|>)\s*(.+)$")


def _expected_stack_summary() -> str:
    return ", ".join(
        f"{dist.name}=={dist.version}" for dist in _CUDA_STACK.distributions
    )


def _raise_bootstrap_error(message: str) -> None:
    raise CudaBootstrapError(
        f"Thor CUDA bootstrap failed for {_CUDA_STACK.name}: {message}\n"
        f"Expected Python CUDA dependency stack: {_expected_stack_summary()}"
    )


def _version_key(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for raw_part in version.split("."):
        match = _VERSION_PART_RE.match(raw_part)
        if match is None:
            break
        parts.append(int(match.group(1)))
    return tuple(parts)


def _compare_versions(left: str, right: str) -> int:
    left_parts = list(_version_key(left))
    right_parts = list(_version_key(right))
    width = max(len(left_parts), len(right_parts))
    left_parts.extend([0] * (width - len(left_parts)))
    right_parts.extend([0] * (width - len(right_parts)))
    if left_parts < right_parts:
        return -1
    if left_parts > right_parts:
        return 1
    return 0


def _version_satisfies(version: str, specifier: str) -> bool:
    for clause in (part.strip() for part in specifier.split(",")):
        if not clause:
            continue
        match = _SPECIFIER_RE.match(clause)
        if match is None:
            _raise_bootstrap_error(f"internal invalid version specifier clause {clause!r}")
        operator, expected = match.groups()
        if operator == "==":
            if version != expected:
                return False
            continue
        if operator == "!=":
            if version == expected:
                return False
            continue
        comparison = _compare_versions(version, expected)
        if operator == ">=" and comparison < 0:
            return False
        if operator == ">" and comparison <= 0:
            return False
        if operator == "<=" and comparison > 0:
            return False
        if operator == "<" and comparison >= 0:
            return False
    return True


def _source_tree_bootstrap_enabled() -> bool:
    """Return true for CMake/source-tree test imports.

    Source-tree CMake test targets run against the just-built extension before
    the final wheel has been installed with its frozen NVIDIA Python wheel
    dependency metadata.  Those tests use the build/link environment selected
    by CMake rather than the wheel-runtime CUDA preload path, so they opt out
    explicitly via this environment variable.  Normal installed-wheel imports
    do not set it and still require the exact frozen CUDA stack.
    """

    return _os.environ.get(_SOURCE_TREE_BOOTSTRAP_ENV) == "1"


def _running_under_nanobind_stubgen() -> bool:
    """Return true for nanobind's build-time stub generator process.

    Stub generation imports the build-tree Python package before the final wheel
    has been installed with its runtime dependency metadata.  In that context
    the native extension is resolved from the build tree and linked build
    environment, while the built wheel still records and enforces the exact
    CUDA Python dependency stack for normal imports.
    """

    if not _sys.argv:
        return False

    argv0 = _Path(_sys.argv[0])
    if argv0.name != "stubgen.py":
        return False

    return any(part.lower() == "nanobind" for part in argv0.parts)


def _validate_resolved_manifest_matches_source_selection() -> None:
    if _SOURCE_CUDA_VERSION != _CUDA_STACK.cuda_version:
        _raise_bootstrap_error(
            f"resolved CUDA manifest is stale: source selects CUDA {_SOURCE_CUDA_VERSION!r}, "
            f"but the resolved wheel manifest was generated for CUDA {_CUDA_STACK.cuda_version!r}; "
            "rebuild the Thor wheel so _cuda_stack_resolved.py is regenerated"
        )


def _metadata_distribution(name: str) -> _metadata.Distribution:
    try:
        dist = _metadata.distribution(name)
    except _metadata.PackageNotFoundError:
        _raise_bootstrap_error(f"missing required distribution {name!r}")

    expected_specifier = _CUDA_DISTRIBUTION_SPECIFIERS.get(name)
    if expected_specifier is not None and not _version_satisfies(dist.version, expected_specifier):
        _raise_bootstrap_error(
            f"distribution {name!r} has version {dist.version!r}, expected {expected_specifier!r}"
        )

    return dist


def _dedupe_existing(paths: _Iterable[_Path]) -> list[_Path]:
    seen: set[str] = set()
    result: list[_Path] = []
    for path in paths:
        if not path.exists():
            continue
        try:
            canonical_path = path.resolve(strict=True)
        except OSError:
            canonical_path = path.absolute()
        key = str(canonical_path)
        if key in seen:
            continue
        seen.add(key)
        result.append(canonical_path)
    return result


def _distribution_files_matching(distribution_name: str, patterns: _Sequence[str]) -> list[_Path]:
    dist = _metadata_distribution(distribution_name)
    if dist.files is None:
        _raise_bootstrap_error(f"distribution {distribution_name!r} does not expose installed file metadata")

    matches: list[_Path] = []
    for file in dist.files:
        relative_path = str(file).replace("\\", "/")
        if not any(_fnmatch(relative_path, pattern) for pattern in patterns):
            continue
        matches.append(_Path(dist.locate_file(file)).absolute())

    return _dedupe_existing(matches)


def _resolve_one_distribution_file(distribution_name: str, patterns: _Sequence[str], description: str) -> _Path:
    matches = _distribution_files_matching(distribution_name, patterns)
    if len(matches) != 1:
        rendered_patterns = ", ".join(patterns)
        if not matches:
            _raise_bootstrap_error(
                f"{distribution_name!r} is missing required {description} matching {rendered_patterns!r}"
            )
        rendered_matches = ", ".join(str(path) for path in matches)
        _raise_bootstrap_error(
            f"{distribution_name!r} has ambiguous {description} for patterns {rendered_patterns!r}: {rendered_matches}"
        )
    return matches[0]


def _distribution_root_containing(distribution_name: str, sentinel: _Path) -> _Path:
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


def _set_strict_env(env_name: str, value: _Path) -> None:
    value_str = str(value)
    existing = _os.environ.get(env_name)
    if existing is not None and existing != value_str:
        _raise_bootstrap_error(
            f"{env_name} is set to {existing!r}, but Thor requires {value_str!r}; "
            f"unset {env_name} or use the Thor {_CUDA_STACK.name} wheel dependencies"
        )
    _os.environ[env_name] = value_str


def _configure_include_dirs() -> None:
    for spec in _INCLUDE_SPECS:
        include_root = _distribution_root_containing(spec.distribution, spec.sentinel)
        _set_strict_env(spec.env_name, include_root)


def _preload_cuda_user_space_libs() -> None:
    rtld_flags = getattr(_os, "RTLD_GLOBAL", 0) | getattr(_os, "RTLD_NOW", 0)
    preloaded: set[str] = set()

    def preload(path: _Path) -> None:
        try:
            key = str(path.resolve())
        except OSError:
            key = str(path)
        if key in preloaded:
            return
        preloaded.add(key)
        try:
            _ctypes.CDLL(str(path), mode=rtld_flags)
        except OSError as error:
            _raise_bootstrap_error(f"failed to load required CUDA library {str(path)!r}: {error}")

    for spec in _LIBRARY_SPECS:
        path = _resolve_one_distribution_file(spec.distribution, spec.patterns, "shared library")
        preload(path)


def configure() -> None:
    global _configured
    if _configured:
        return

    if _running_under_nanobind_stubgen() or _source_tree_bootstrap_enabled():
        _configured = True
        return

    _validate_resolved_manifest_matches_source_selection()

    # Validate every declared distribution before mutating the process loader
    # state so version or installation problems fail with a single Thor-specific
    # diagnostic rather than a later low-level dynamic linker error.
    for dist in _CUDA_DISTRIBUTIONS:
        _metadata_distribution(dist.name)

    _configure_include_dirs()
    _preload_cuda_user_space_libs()
    _configured = True
