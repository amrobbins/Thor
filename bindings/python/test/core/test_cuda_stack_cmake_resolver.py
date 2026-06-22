from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


_RESOLVER_PATH = Path(__file__).resolve().parents[2] / "tools" / "resolve_cuda_stack_for_cmake.py"
_spec = importlib.util.spec_from_file_location("thor_cuda_stack_cmake_resolver", _RESOLVER_PATH)
assert _spec is not None and _spec.loader is not None
resolver = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = resolver
_spec.loader.exec_module(resolver)


class _FakeDistribution:

    def __init__(self, root: Path, version: str, *files: str):
        self._root = root
        self.version = version
        self.files = [Path(file) for file in files]

    def locate_file(self, file: Path) -> Path:
        return self._root / file


def _touch(root: Path, *files: str) -> None:
    for file in files:
        path = root / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("// sentinel\n")


def _fake_stack() -> SimpleNamespace:
    versions = {
        "nvidia-cuda-runtime": "13.3.29",
        "nvidia-cuda-nvrtc": "13.3.33",
        "nvidia-nvjitlink": "13.3.33",
        "nvidia-cublas": "13.5.1.27",
        "nvidia-cusparse": "12.8.1.7",
        "nvidia-cusolver": "12.2.2.18",
        "nvidia-cuda-cccl": "13.3.3.3.1",
        "nvidia-cudnn-cu13": "9.23.2.1",
        "nvidia-cudnn-frontend": "1.25.0",
    }
    return SimpleNamespace(
        cuda_version="13.3",
        name="CUDA 13.3",
        distributions=tuple(SimpleNamespace(name=name, version=version) for name, version in versions.items()),
    )


def _fake_distributions(tmp_path: Path) -> dict[str, _FakeDistribution]:
    files_by_dist = {
        "nvidia-cuda-runtime": ["nvidia/cu13/include/cuda.h", "nvidia/cu13/lib/libcudart.so.13"],
        "nvidia-cuda-nvrtc": [
            "nvidia/cu13/include/nvrtc.h",
            "nvidia/cu13/lib/libnvrtc-builtins.so.13.3.33",
            "nvidia/cu13/lib/libnvrtc.so.13",
        ],
        "nvidia-nvjitlink": ["nvidia/cu13/include/nvJitLink.h", "nvidia/cu13/lib/libnvJitLink.so.13"],
        "nvidia-cublas": [
            "nvidia/cu13/include/cublas_v2.h",
            "nvidia/cu13/lib/libcublas.so.13",
            "nvidia/cu13/lib/libcublasLt.so.13",
        ],
        "nvidia-cusparse": ["nvidia/cu13/include/cusparse.h", "nvidia/cu13/lib/libcusparse.so.12"],
        "nvidia-cusolver": ["nvidia/cu13/include/cusolverDn.h", "nvidia/cu13/lib/libcusolver.so.12"],
        "nvidia-cuda-cccl": ["nvidia/cu13/include/cccl/cub/cub.cuh"],
        "nvidia-cudnn-cu13": ["nvidia/cudnn/include/cudnn.h", "nvidia/cudnn/lib/libcudnn.so.9"],
        "nvidia-cudnn-frontend": ["include/cudnn_frontend.h"],
    }
    versions = {dist.name: dist.version for dist in _fake_stack().distributions}
    distributions: dict[str, _FakeDistribution] = {}
    for name, files in files_by_dist.items():
        root = tmp_path / name
        _touch(root, *files)
        distributions[name] = _FakeDistribution(root, versions[name], *files)
    return distributions


def test_cmake_resolver_requires_and_returns_exact_python_cuda_stack(monkeypatch, tmp_path):
    distributions = _fake_distributions(tmp_path)

    monkeypatch.setattr(resolver, "_load_resolved_stack", lambda _repo_root: _fake_stack())
    monkeypatch.setattr(resolver.metadata, "distribution", lambda name: distributions[name])

    resolved = resolver.resolve(tmp_path)

    assert resolved["cuda_version"] == "13.3"
    assert resolved["stack_name"] == "CUDA 13.3"
    assert resolved["variables"]["THOR_CUDA_INCLUDE_DIR"] == str(
        (tmp_path / "nvidia-cuda-runtime" / "nvidia/cu13/include").resolve()
    )
    assert resolved["variables"]["THOR_CUDA_CCCL_INCLUDE_DIR"] == str(
        (tmp_path / "nvidia-cuda-cccl" / "nvidia/cu13/include/cccl").resolve()
    )
    assert resolved["variables"]["THOR_CUDNN_LIBRARY"].endswith("libcudnn.so.9")
    assert resolved["variables"]["THOR_NVRTC_BUILTINS_LIBRARY"].endswith("libnvrtc-builtins.so.13.3.33")
    assert resolved["variables"]["THOR_NVRTC_BUILTINS_LIBRARY"] not in resolved["link_libraries"]
    assert resolved["variables"]["THOR_CUDART_LIBRARY"] in resolved["link_libraries"]


def test_cmake_resolver_rejects_version_drift_from_wheel_manifest(monkeypatch, tmp_path):
    distributions = _fake_distributions(tmp_path)
    distributions["nvidia-cuda-runtime"].version = "13.3.30"

    monkeypatch.setattr(resolver, "_load_resolved_stack", lambda _repo_root: _fake_stack())
    monkeypatch.setattr(resolver.metadata, "distribution", lambda name: distributions[name])

    with pytest.raises(resolver.ResolveError, match="expected exact wheel-runtime version '13.3.29'"):
        resolver.resolve(tmp_path)
