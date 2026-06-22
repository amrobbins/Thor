"""Source-level CUDA stack selection for Thor wheel builds.

Only the CUDA toolkit family is selected here.  A wheel build resolves the
latest compatible NVIDIA Python wheels for this family, freezes their exact
versions into ``thor._cuda_stack_resolved``, and emits those exact pins in the
wheel metadata.  Import-time bootstrap must use the resolved manifest only; it
must never re-resolve newer NVIDIA packages for an already-built wheel.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# Change this one value when moving the Thor wheel to a different CUDA stack.
# For example, move from "13.3" to "13.4" and build in a clean isolated build
# environment; the build backend will resolve and freeze the latest compatible
# component wheels for that family.
CUDA_VERSION = "13.3"

# Non-CUDA runtime dependencies that should remain in the final wheel metadata.
BASE_DEPENDENCIES = (
    "numpy>=2.3.4",
    "ml_dtypes>=0.5.4",
)

# CUDA Toolkit extras that pull in the component wheels Thor directly uses.
CUDA_TOOLKIT_EXTRAS = (
    "cccl",
    "cublas",
    "cudart",
    "cusolver",
    "cusparse",
    "nvjitlink",
    "nvrtc",
)

# Component distribution names frozen into the built wheel.  The cuda-toolkit
# metapackage is a build-time resolver only and is intentionally not required at
# import time.
CUDA_TOOLKIT_COMPONENT_DISTRIBUTIONS = (
    "nvidia-cuda-runtime",
    "nvidia-cuda-nvrtc",
    "nvidia-nvjitlink",
    "nvidia-cublas",
    "nvidia-cusparse",
    "nvidia-cusolver",
    "nvidia-cuda-cccl",
)

# cuDNN is versioned independently of cuda-toolkit.  Keep these broad enough to
# take current cuDNN/cuDNN Frontend bugfixes at wheel-build time, but the built
# wheel will still freeze exact versions.
CUDNN_BUILD_REQUIREMENTS_BY_CUDA_MAJOR = {
    "13": (
        "nvidia-cudnn-cu13>=9.23,<10",
        "nvidia-cudnn-frontend>=1.25,<2",
    ),
}
CUDNN_DISTRIBUTIONS_BY_CUDA_MAJOR = {
    "13": (
        "nvidia-cudnn-cu13",
        "nvidia-cudnn-frontend",
    ),
}


@dataclass(frozen=True)
class IncludeSpec:
    env_name: str
    distribution: str
    sentinel: Path


@dataclass(frozen=True)
class LibrarySpec:
    distribution: str
    patterns: tuple[str, ...]


# Runtime bootstrap skips the CUDA Toolkit/CCCL include specs below because
# Thor vendors a complete CUDA Toolkit header tree into the package for NVRTC.
# They remain in the manifest so local CMake can validate that the selected
# Python environment has the expected NVIDIA component wheels installed.
INCLUDE_SPECS = (
    IncludeSpec("THOR_CUDA_INCLUDE_DIR", "nvidia-cuda-runtime", Path("vector_types.h")),
    IncludeSpec("THOR_CUDA_CCCL_INCLUDE_DIR", "nvidia-cuda-cccl", Path("cub") / "cub.cuh"),
    IncludeSpec("THOR_CUDNN_FRONTEND_INCLUDE_DIR", "nvidia-cudnn-frontend", Path("cudnn_frontend.h")),
)

# Libraries are preloaded by absolute path, in dependency order, from the
# resolved distributions.  Patterns are matched only inside the owning wheel's
# installed-file metadata.  Each spec must resolve to exactly one existing file.
LIBRARY_SPECS = (
    LibrarySpec("nvidia-cuda-runtime", ("**/libcudart.so.13",)),
    LibrarySpec("nvidia-nvjitlink", ("**/libnvJitLink.so.13",)),
    LibrarySpec("nvidia-cuda-nvrtc", ("**/libnvrtc-builtins.so.13.*",)),
    LibrarySpec("nvidia-cuda-nvrtc", ("**/libnvrtc.so.13",)),
    LibrarySpec("nvidia-cublas", ("**/libcublas.so.13",)),
    LibrarySpec("nvidia-cublas", ("**/libcublasLt.so.13",)),
    LibrarySpec("nvidia-cusparse", ("**/libcusparse.so.12",)),
    LibrarySpec("nvidia-cusolver", ("**/libcusolver.so.12",)),
    LibrarySpec("nvidia-cudnn-cu13", ("**/libcudnn.so.9",)),
)


def cuda_major() -> str:
    return CUDA_VERSION.split(".", 1)[0]


def cuda_toolkit_requirement() -> str:
    extras = ",".join(CUDA_TOOLKIT_EXTRAS)
    major, minor = CUDA_VERSION.split(".", 1)
    next_minor = int(minor) + 1
    return f"cuda-toolkit[{extras}]>={CUDA_VERSION},<{major}.{next_minor}"


def cudnn_build_requirements() -> tuple[str, ...]:
    major = cuda_major()
    try:
        return CUDNN_BUILD_REQUIREMENTS_BY_CUDA_MAJOR[major]
    except KeyError as error:  # pragma: no cover - defensive build guard
        known = ", ".join(sorted(CUDNN_BUILD_REQUIREMENTS_BY_CUDA_MAJOR))
        raise RuntimeError(f"No Thor cuDNN build requirements for CUDA major {major!r}; known: {known}") from error


def cuda_build_requirements() -> tuple[str, ...]:
    """Broad CUDA requirements installed into the isolated wheel build env."""

    return (cuda_toolkit_requirement(),) + cudnn_build_requirements()


def resolved_distribution_names() -> tuple[str, ...]:
    major = cuda_major()
    try:
        cudnn_distributions = CUDNN_DISTRIBUTIONS_BY_CUDA_MAJOR[major]
    except KeyError as error:  # pragma: no cover - defensive build guard
        known = ", ".join(sorted(CUDNN_DISTRIBUTIONS_BY_CUDA_MAJOR))
        raise RuntimeError(f"No Thor cuDNN distributions for CUDA major {major!r}; known: {known}") from error
    return CUDA_TOOLKIT_COMPONENT_DISTRIBUTIONS + cudnn_distributions
