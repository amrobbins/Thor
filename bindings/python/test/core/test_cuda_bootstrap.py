from __future__ import annotations

import os
from pathlib import Path
import sys
import tomllib

import pytest

import thor._bootstrap as bootstrap
import thor._nvrtc_headers as nvrtc_headers
import thor._cuda_stack as source_cuda_stack
import thor._cuda_stack_resolved as resolved_cuda_stack

_BUILD_BACKEND_ROOT = Path(__file__).resolve().parents[2] / "build_backend"
if str(_BUILD_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BUILD_BACKEND_ROOT))

from thor_build import cuda_stack as build_cuda_stack  # noqa: E402
from thor_build import dynamic_metadata as build_dynamic_metadata  # noqa: E402


class _FakeDistribution:

    def __init__(self, root: Path, files: list[Path], version: str):
        self._root = root
        self.files = files
        self.version = version

    def locate_file(self, file: Path) -> Path:
        return self._root / file


def _dist(root: Path, version: str, *files: str) -> _FakeDistribution:
    return _FakeDistribution(root, [Path(file) for file in files], version)


def test_pyproject_uses_dynamic_dependencies_from_thor_cuda_stack_provider():
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text())

    assert "dependencies" in pyproject["project"]["dynamic"]
    assert "dependencies" not in pyproject["project"]
    assert pyproject["build-system"]["build-backend"] == "thor_build_backend"
    assert pyproject["build-system"]["backend-path"] == ["build_backend"]
    assert pyproject["tool"]["scikit-build"]["metadata"]["dependencies"]["provider"] == "thor_build.dynamic_metadata"


def test_build_requirements_are_derived_from_single_cuda_version():
    requirements = build_cuda_stack.cuda_build_requirements()

    assert requirements[0] == (
        "cuda-toolkit[cccl,cublas,cudart,cusolver,cusparse,nvjitlink,nvrtc]>=13.3,<13.4"
    )
    assert "nvidia-cudnn-cu13>=9.23,<10" in requirements
    assert "nvidia-cudnn-frontend>=1.25,<2" in requirements


def test_dynamic_dependencies_freeze_exact_installed_cuda_stack(monkeypatch, tmp_path):
    versions = {
        "nvidia-cuda-runtime": "13.3.30",
        "nvidia-cuda-nvrtc": "13.3.34",
        "nvidia-nvjitlink": "13.3.34",
        "nvidia-cublas": "13.5.2.1",
        "nvidia-cusparse": "12.8.2.1",
        "nvidia-cusolver": "12.2.3.1",
        "nvidia-cuda-cccl": "13.3.4.1",
        "nvidia-cudnn-cu13": "9.23.3.1",
        "nvidia-cudnn-frontend": "1.25.1",
    }

    monkeypatch.setattr(build_cuda_stack.metadata, "version", lambda name: versions[name])
    monkeypatch.setattr(build_cuda_stack, "_RESOLVED_STACK_PATH", tmp_path / "_cuda_stack_resolved.py")

    dependencies = build_dynamic_metadata.dynamic_metadata("dependencies")

    assert "numpy>=2.3.4" in dependencies
    assert "ml_dtypes>=0.5.4" in dependencies
    for name, version in versions.items():
        assert f"{name}=={version}" in dependencies

    generated = (tmp_path / "_cuda_stack_resolved.py").read_text()
    assert 'CUDA_VERSION = "13.3"' in generated
    assert 'CudaDistribution("nvidia-cuda-runtime", "13.3.30")' in generated
    assert 'CudaDistribution("nvidia-cudnn-frontend", "1.25.1")' in generated


def test_resolved_runtime_dependencies_are_exact_pins():
    pins = resolved_cuda_stack.dependency_specifiers()
    resolved_distributions = resolved_cuda_stack.CUDA_STACK.distributions

    assert resolved_cuda_stack.CUDA_STACK.cuda_version == source_cuda_stack.CUDA_VERSION
    assert tuple(dist.name for dist in resolved_distributions) == source_cuda_stack.resolved_distribution_names()
    assert pins == tuple(f"{dist.name}=={dist.version}" for dist in resolved_distributions)
    assert all(dist.version for dist in resolved_distributions)
    assert all(">=" not in pin and "<" not in pin for pin in pins)


def test_runtime_bootstrap_requires_exact_resolved_versions():
    assert bootstrap._version_satisfies("13.3.29", "==13.3.29")
    assert not bootstrap._version_satisfies("13.3.30", "==13.3.29")
    assert not bootstrap._version_satisfies("13.3.29.0", "==13.3.29")


def test_runtime_bootstrap_rejects_stale_resolved_manifest(monkeypatch):
    monkeypatch.setattr(bootstrap, "_SOURCE_CUDA_VERSION", "13.4")

    with pytest.raises(bootstrap.CudaBootstrapError, match="resolved CUDA manifest is stale"):
        bootstrap._validate_resolved_manifest_matches_source_selection()


def test_runtime_bootstrap_detects_source_tree_test_bypass(monkeypatch):
    monkeypatch.setenv("THOR_CUDA_BOOTSTRAP_SOURCE_TREE", "1")

    assert bootstrap._source_tree_bootstrap_enabled()


def test_runtime_bootstrap_source_tree_bypass_requires_exact_one(monkeypatch):
    monkeypatch.setenv("THOR_CUDA_BOOTSTRAP_SOURCE_TREE", "true")

    assert not bootstrap._source_tree_bootstrap_enabled()

def test_runtime_bootstrap_detects_nanobind_stubgen(monkeypatch):
    monkeypatch.setattr(
        bootstrap._sys,
        "argv",
        ["/opt/python/cp312-cp312/lib/python3.12/site-packages/nanobind/stubgen.py"],
    )

    assert bootstrap._running_under_nanobind_stubgen()


def test_runtime_bootstrap_does_not_treat_other_stubgen_as_nanobind(monkeypatch):
    monkeypatch.setattr(bootstrap._sys, "argv", ["/tmp/other_tool/stubgen.py"])

    assert not bootstrap._running_under_nanobind_stubgen()


def test_configure_skips_cuda_distribution_validation_for_nanobind_stubgen(monkeypatch):
    calls: list[str] = []

    monkeypatch.setattr(bootstrap, "_configured", False)
    monkeypatch.setattr(
        bootstrap._sys,
        "argv",
        ["/opt/python/cp312-cp312/lib/python3.12/site-packages/nanobind/stubgen.py"],
    )
    monkeypatch.setattr(
        bootstrap,
        "_validate_resolved_manifest_matches_source_selection",
        lambda: calls.append("manifest"),
    )
    monkeypatch.setattr(bootstrap, "_configure_include_dirs", lambda: calls.append("includes"))
    monkeypatch.setattr(bootstrap, "_preload_cuda_user_space_libs", lambda: calls.append("libs"))

    def fail_metadata(_name: str):
        raise AssertionError("stub generation must not require installed CUDA wheel metadata")

    monkeypatch.setattr(bootstrap, "_metadata_distribution", fail_metadata)

    bootstrap.configure()

    assert bootstrap._configured
    assert calls == []


def test_configure_skips_cuda_distribution_validation_for_source_tree_test(monkeypatch):
    calls: list[str] = []

    monkeypatch.setattr(bootstrap, "_configured", False)
    monkeypatch.setenv("THOR_CUDA_BOOTSTRAP_SOURCE_TREE", "1")
    monkeypatch.setattr(
        bootstrap,
        "_validate_resolved_manifest_matches_source_selection",
        lambda: calls.append("manifest"),
    )
    monkeypatch.setattr(bootstrap, "_configure_include_dirs", lambda: calls.append("includes"))
    monkeypatch.setattr(bootstrap, "_preload_cuda_user_space_libs", lambda: calls.append("libs"))

    def fail_metadata(_name: str):
        raise AssertionError("source-tree tests must not require installed CUDA wheel metadata")

    monkeypatch.setattr(bootstrap, "_metadata_distribution", fail_metadata)

    bootstrap.configure()

    assert bootstrap._configured
    assert calls == []

def test_cuda_include_env_uses_nvrtc_bundled_headers_and_exact_declared_distribution_files(monkeypatch, tmp_path):
    cudnn_frontend_include = tmp_path / "frontend_root" / "include"
    cudnn_frontend_include.mkdir(parents=True)
    (cudnn_frontend_include / "cudnn_frontend.h").write_text("// sentinel\n")

    distributions = {
        "nvidia-cudnn-frontend": _dist(
            tmp_path / "frontend_root",
            "1.25.0",
            "include/cudnn_frontend.h",
        ),
    }

    bundled_headers = tmp_path / "nvrtc_headers" / "cuda-13.3"
    monkeypatch.delenv("THOR_CUDA_INCLUDE_DIR", raising=False)
    monkeypatch.delenv("THOR_CUDA_CCCL_INCLUDE_DIR", raising=False)
    monkeypatch.delenv("THOR_NVRTC_BUNDLED_HEADERS_DIR", raising=False)
    monkeypatch.delenv("THOR_CUDNN_FRONTEND_INCLUDE_DIR", raising=False)
    monkeypatch.setattr(nvrtc_headers, "nvrtc_bundled_headers_dir", lambda cuda_version: bundled_headers)
    monkeypatch.setattr(bootstrap, "_nvrtc_bundled_headers_dir", lambda cuda_version: bundled_headers)
    monkeypatch.setattr(bootstrap, "_metadata_distribution", lambda name: distributions[name])

    bootstrap._configure_include_dirs()

    assert "THOR_CUDA_INCLUDE_DIR" not in os.environ
    assert "THOR_CUDA_CCCL_INCLUDE_DIR" not in os.environ
    assert Path(os.environ["THOR_NVRTC_BUNDLED_HEADERS_DIR"]) == bundled_headers
    assert Path(os.environ["THOR_CUDNN_FRONTEND_INCLUDE_DIR"]) == cudnn_frontend_include

def test_nvrtc_bundled_headers_default_cache_uses_cuda_version(monkeypatch, tmp_path):
    monkeypatch.delenv("THOR_NVRTC_BUNDLED_HEADERS_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    assert nvrtc_headers.nvrtc_bundled_headers_dir("13.3") == tmp_path / "cache" / "thor" / "nvrtc_bundled_headers" / "cuda-13.3"

def test_nvrtc_bundled_headers_env_override(monkeypatch, tmp_path):
    override = tmp_path / "override"
    monkeypatch.setenv("THOR_NVRTC_BUNDLED_HEADERS_DIR", str(override))

    assert nvrtc_headers.nvrtc_bundled_headers_dir("13.3") == override

def test_cuda_include_env_rejects_conflicting_user_override(monkeypatch, tmp_path):
    monkeypatch.setenv("THOR_NVRTC_BUNDLED_HEADERS_DIR", str(tmp_path / "wrong_headers"))

    with pytest.raises(bootstrap.CudaBootstrapError, match="THOR_NVRTC_BUNDLED_HEADERS_DIR"):
        bootstrap._set_strict_env("THOR_NVRTC_BUNDLED_HEADERS_DIR", tmp_path / "right_headers")

def test_metadata_distribution_rejects_missing_required_distribution(monkeypatch):

    def missing(_name):
        raise bootstrap._metadata.PackageNotFoundError

    monkeypatch.setattr(bootstrap._metadata, "distribution", missing)

    with pytest.raises(bootstrap.CudaBootstrapError, match="missing required distribution 'nvidia-cuda-runtime'"):
        bootstrap._metadata_distribution("nvidia-cuda-runtime")


def test_metadata_distribution_rejects_version_mismatch(monkeypatch, tmp_path):
    monkeypatch.setattr(
        bootstrap._metadata,
        "distribution",
        lambda _name: _dist(tmp_path, "13.3.30", "nvidia/cu13/lib/libcudart.so.13"),
    )

    with pytest.raises(bootstrap.CudaBootstrapError, match="expected '==13.3.29'"):
        bootstrap._metadata_distribution("nvidia-cuda-runtime")


def test_preload_uses_only_exact_distribution_library_paths(monkeypatch, tmp_path):
    real_root = tmp_path / "real_cublas_root"
    symlink_root = tmp_path / "cublas_root"
    lib_dir = real_root / "nvidia" / "cu13" / "lib"
    lib_dir.mkdir(parents=True)
    cublas = lib_dir / "libcublas.so.13"
    cublas.write_text("")
    symlink_root.symlink_to(real_root, target_is_directory=True)

    distributions = {
        "nvidia-cublas": _dist(
            symlink_root,
            "13.5.1.27",
            "nvidia/cu13/lib/libcublas.so.13",
        ),
    }
    loaded: list[Path] = []

    monkeypatch.setattr(bootstrap, "_metadata_distribution", lambda name: distributions[name])
    monkeypatch.setattr(bootstrap._ctypes, "CDLL", lambda path, mode=0: loaded.append(Path(path)))

    monkeypatch.setattr(
        bootstrap,
        "_LIBRARY_SPECS",
        (bootstrap._LibrarySpec("nvidia-cublas", ("**/libcublas.so.13",)),),
    )

    bootstrap._preload_cuda_user_space_libs()

    assert loaded == [cublas.resolve()]


def test_resolve_one_distribution_file_rejects_missing_library(monkeypatch, tmp_path):
    distributions = {
        "nvidia-cublas": _dist(
            tmp_path / "cublas_root",
            "13.5.1.27",
            "nvidia/cu13/lib/libcublas.so.13",
        ),
    }
    monkeypatch.setattr(bootstrap, "_metadata_distribution", lambda name: distributions[name])

    with pytest.raises(bootstrap.CudaBootstrapError, match="missing required shared library"):
        bootstrap._resolve_one_distribution_file("nvidia-cublas", ("**/libcublasLt.so.13",), "shared library")
