"""Resolved CUDA stack frozen into this Thor wheel.

This file is generated during wheel builds from ``thor._cuda_stack`` after the
build environment has installed the latest compatible NVIDIA CUDA dependencies
for the selected CUDA family.  Import-time code must use this exact manifest and
must not resolve newer package versions for an already-built wheel.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


CUDA_VERSION = "13.3"
CUDA_STACK_NAME = "CUDA 13.3"


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
        CudaDistribution("nvidia-cuda-runtime", "13.3.29"),
        CudaDistribution("nvidia-cuda-nvrtc", "13.3.33"),
        CudaDistribution("nvidia-nvjitlink", "13.3.33"),
        CudaDistribution("nvidia-cublas", "13.6.0.2"),
        CudaDistribution("nvidia-cusparse", "12.8.2.51"),
        CudaDistribution("nvidia-cusolver", "12.2.6.9"),
        CudaDistribution("nvidia-cuda-cccl", "13.3.3.4.1"),
        CudaDistribution("nvidia-cudnn-cu13", "9.24.0.43"),
        CudaDistribution("nvidia-cudnn-frontend", "1.25.0"),
    ),
    includes=(
        IncludeSpec("THOR_CUDA_INCLUDE_DIR", "nvidia-cuda-runtime", Path('vector_types.h')),
        IncludeSpec("THOR_CUDA_CCCL_INCLUDE_DIR", "nvidia-cuda-cccl", Path('cub/cub.cuh')),
        IncludeSpec("THOR_CUDNN_FRONTEND_INCLUDE_DIR", "nvidia-cudnn-frontend", Path('cudnn_frontend.h')),
    ),
    libraries=(
        LibrarySpec("nvidia-cuda-runtime", ("**/libcudart.so.13", )),
        LibrarySpec("nvidia-nvjitlink", ("**/libnvJitLink.so.13", )),
        LibrarySpec("nvidia-cuda-nvrtc", ("**/libnvrtc-builtins.so.13.*", )),
        LibrarySpec("nvidia-cuda-nvrtc", ("**/libnvrtc.so.13", )),
        LibrarySpec("nvidia-cublas", ("**/libcublas.so.13", )),
        LibrarySpec("nvidia-cublas", ("**/libcublasLt.so.13", )),
        LibrarySpec("nvidia-cusparse", ("**/libcusparse.so.12", )),
        LibrarySpec("nvidia-cusolver", ("**/libcusolver.so.12", )),
        LibrarySpec("nvidia-cudnn-cu13", ("**/libcudnn.so.9", )),
    ),
)


def dependency_specifiers() -> tuple[str, ...]:
    """Return exact PEP 508 dependency pins for this built wheel."""

    return tuple(f"{dist.name}=={dist.version}" for dist in CUDA_STACK.distributions)
