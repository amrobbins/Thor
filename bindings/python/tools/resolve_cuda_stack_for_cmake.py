#!/usr/bin/env python3
"""Resolve Thor's Python CUDA wheel stack for CMake local builds.

This script intentionally mirrors the runtime bootstrap contract without
importing ``thor`` itself.  CMake uses it to make source-tree builds fail fast
unless the selected Python environment contains the exact CUDA user-space stack
frozen into ``thor._cuda_stack_resolved``.
"""

from __future__ import annotations

from fnmatch import fnmatch
from importlib import metadata
import importlib.util
import json
from pathlib import Path
import sys
from typing import Iterable, Sequence


class ResolveError(RuntimeError):
    pass


def _load_resolved_stack(repo_root: Path):
    resolved_path = repo_root / "bindings" / "python" / "src" / "thor" / "_cuda_stack_resolved.py"
    if not resolved_path.exists():
        raise ResolveError(f"missing resolved CUDA manifest: {resolved_path}")

    spec = importlib.util.spec_from_file_location("_thor_cuda_stack_resolved_for_cmake", resolved_path)
    if spec is None or spec.loader is None:
        raise ResolveError(f"failed to load resolved CUDA manifest: {resolved_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.CUDA_STACK


def _distribution(name: str, expected_version: str) -> metadata.Distribution:
    try:
        dist = metadata.distribution(name)
    except metadata.PackageNotFoundError as error:
        raise ResolveError(f"missing required Python CUDA distribution {name!r}") from error

    if dist.version != expected_version:
        raise ResolveError(
            f"Python CUDA distribution {name!r} has version {dist.version!r}, "
            f"expected exact wheel-runtime version {expected_version!r}"
        )

    if dist.files is None:
        raise ResolveError(f"Python CUDA distribution {name!r} does not expose installed file metadata")

    return dist


def _dedupe_existing(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
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


def _distribution_files_matching(dist: metadata.Distribution, patterns: Sequence[str]) -> list[Path]:
    matches: list[Path] = []
    for file in dist.files or ():
        relative_path = str(file).replace("\\", "/")
        if any(fnmatch(relative_path, pattern) for pattern in patterns):
            matches.append(Path(dist.locate_file(file)).absolute())
    return _dedupe_existing(matches)


def _resolve_one_file(
    distributions: dict[str, metadata.Distribution],
    distribution_name: str,
    patterns: Sequence[str],
    description: str,
) -> Path:
    matches = _distribution_files_matching(distributions[distribution_name], patterns)
    if len(matches) == 1:
        return matches[0]

    rendered_patterns = ", ".join(patterns)
    if not matches:
        raise ResolveError(
            f"{distribution_name!r} is missing required {description} matching {rendered_patterns!r}"
        )

    rendered_matches = ", ".join(str(path) for path in matches)
    raise ResolveError(
        f"{distribution_name!r} has ambiguous {description} for patterns {rendered_patterns!r}: {rendered_matches}"
    )


def _include_root_containing(
    distributions: dict[str, metadata.Distribution],
    distribution_name: str,
    sentinel: Path,
) -> Path:
    sentinel_posix = sentinel.as_posix()
    sentinel_path = _resolve_one_file(
        distributions,
        distribution_name,
        (f"**/{sentinel_posix}", sentinel_posix),
        f"include sentinel {sentinel_posix}",
    )

    root = sentinel_path
    for _ in sentinel.parts:
        root = root.parent
    return root


# These are compile-time include requirements.  They deliberately go beyond
# thor._cuda_stack.INCLUDE_SPECS, which are import-time environment variables
# for NVRTC and cuDNN Frontend only.
_INCLUDE_VARIABLE_SPECS: tuple[tuple[str, str, Path], ...] = (
    ("THOR_CUDA_INCLUDE_DIR", "nvidia-cuda-runtime", Path("cuda.h")),
    ("THOR_CUDA_CCCL_INCLUDE_DIR", "nvidia-cuda-cccl", Path("cub") / "cub.cuh"),
    ("THOR_CUDNN_INCLUDE_DIR", "nvidia-cudnn-cu13", Path("cudnn.h")),
    ("THOR_CUDNN_FRONTEND_INCLUDE_DIR", "nvidia-cudnn-frontend", Path("cudnn_frontend.h")),
    ("THOR_CUBLAS_INCLUDE_DIR", "nvidia-cublas", Path("cublas_v2.h")),
    ("THOR_CUSOLVER_INCLUDE_DIR", "nvidia-cusolver", Path("cusolverDn.h")),
    ("THOR_CUSPARSE_INCLUDE_DIR", "nvidia-cusparse", Path("cusparse.h")),
    ("THOR_NVRTC_INCLUDE_DIR", "nvidia-cuda-nvrtc", Path("nvrtc.h")),
    ("THOR_NVJITLINK_INCLUDE_DIR", "nvidia-nvjitlink", Path("nvJitLink.h")),
)

_LIBRARY_VARIABLE_SPECS: tuple[tuple[str, str, tuple[str, ...], bool], ...] = (
    ("THOR_CUDART_LIBRARY", "nvidia-cuda-runtime", ("**/libcudart.so.13",), True),
    ("THOR_NVJITLINK_LIBRARY", "nvidia-nvjitlink", ("**/libnvJitLink.so.13",), True),
    ("THOR_NVRTC_BUILTINS_LIBRARY", "nvidia-cuda-nvrtc", ("**/libnvrtc-builtins.so.13.*",), False),
    ("THOR_NVRTC_LIBRARY", "nvidia-cuda-nvrtc", ("**/libnvrtc.so.13",), True),
    ("THOR_CUBLAS_LIBRARY", "nvidia-cublas", ("**/libcublas.so.13",), True),
    ("THOR_CUBLASLT_LIBRARY", "nvidia-cublas", ("**/libcublasLt.so.13",), True),
    ("THOR_CUSPARSE_LIBRARY", "nvidia-cusparse", ("**/libcusparse.so.12",), True),
    ("THOR_CUSOLVER_LIBRARY", "nvidia-cusolver", ("**/libcusolver.so.12",), True),
    ("THOR_CUDNN_LIBRARY", "nvidia-cudnn-cu13", ("**/libcudnn.so.9",), True),
)


def resolve(repo_root: Path) -> dict[str, object]:
    stack = _load_resolved_stack(repo_root)
    expected_versions = {dist.name: dist.version for dist in stack.distributions}
    distributions = {
        name: _distribution(name, expected_version)
        for name, expected_version in expected_versions.items()
    }

    variables: dict[str, str] = {}
    include_dirs: list[str] = []
    for variable, distribution_name, sentinel in _INCLUDE_VARIABLE_SPECS:
        include_dir = _include_root_containing(distributions, distribution_name, sentinel)
        variables[variable] = str(include_dir)
        include_dirs.append(str(include_dir))

    link_libraries: list[str] = []
    for variable, distribution_name, patterns, link in _LIBRARY_VARIABLE_SPECS:
        library_path = _resolve_one_file(distributions, distribution_name, patterns, "shared library")
        variables[variable] = str(library_path)
        if link:
            link_libraries.append(str(library_path))

    # Keep CMake command lines tidy when multiple component wheels share a
    # unified include tree.
    include_dirs = list(dict.fromkeys(include_dirs))

    return {
        "cuda_version": stack.cuda_version,
        "stack_name": stack.name,
        "variables": variables,
        "include_dirs": include_dirs,
        "link_libraries": link_libraries,
    }


def main(argv: Sequence[str]) -> int:
    if len(argv) != 2:
        print("usage: resolve_cuda_stack_for_cmake.py <repo-root>", file=sys.stderr)
        return 2

    try:
        repo_root = Path(argv[1]).resolve(strict=True)
        print(json.dumps(resolve(repo_root), sort_keys=True))
        return 0
    except Exception as error:
        print(f"Thor Python CUDA stack resolution failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
