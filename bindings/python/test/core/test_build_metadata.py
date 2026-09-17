from __future__ import annotations

from pathlib import Path
import sys
import tomllib


_BUILD_BACKEND_ROOT = Path(__file__).resolve().parents[2] / "build_backend"
if str(_BUILD_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BUILD_BACKEND_ROOT))

from thor_build import dynamic_metadata, version_metadata


def test_kernel_wheel_dependencies_pin_every_supported_backend_to_thor_version():
    assert dynamic_metadata.kernel_wheel_dependencies("1.2.3") == [
        "thor-cuda-kernels-sm89==1.2.3",
        "thor-cuda-kernels-sm120==1.2.3",
    ]


def test_thor_cuda_dependencies_append_exact_version_kernel_wheels(monkeypatch):
    monkeypatch.setattr(
        dynamic_metadata.cuda_stack,
        "runtime_dependencies",
        lambda: ["numpy>=2.3.4", "nvidia-cudnn-cu13==9.23.2.1"],
    )
    monkeypatch.setattr(version_metadata, "thor_version", lambda: "4.5.6")

    assert dynamic_metadata.dynamic_metadata("dependencies") == [
        "numpy>=2.3.4",
        "nvidia-cudnn-cu13==9.23.2.1",
        "thor-cuda-kernels-sm89==4.5.6",
        "thor-cuda-kernels-sm120==4.5.6",
    ]


def test_dependency_and_version_metadata_share_canonical_thor_version():
    version = version_metadata.thor_version()

    assert version_metadata.dynamic_metadata("version") == version
    assert dynamic_metadata.kernel_wheel_dependencies() == [
        f"thor-cuda-kernels-sm89=={version}",
        f"thor-cuda-kernels-sm120=={version}",
    ]


def test_thor_cuda_wheel_build_targets_complete_cmake_release_bundle():
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    assert pyproject["tool"]["scikit-build"]["build"]["targets"] == ["thor_release_wheels"]
    assert "wheel>=0.47" in pyproject["build-system"]["requires"]


def test_cmake_kernel_sidecar_packer_is_present():
    repo_root = Path(__file__).resolve().parents[4]
    pack_script = repo_root / "cmake" / "PackThorKernelWheel.cmake"

    assert pack_script.is_file()
    contents = pack_script.read_text(encoding="utf-8")
    assert "python" not in pack_script.name.lower()
    assert "-m wheel pack" in contents
    assert "Root-Is-Purelib: false" in contents
    assert "Description-Content-Type: text/markdown" in contents
    assert "is not intended to be used directly" in contents


def test_kernel_wheel_projects_publish_long_description_metadata():
    kernel_wheels_root = Path(__file__).resolve().parents[2] / "kernel_wheels"

    for sm in ("sm89", "sm120"):
        pyproject = tomllib.loads((kernel_wheels_root / sm / "pyproject.toml").read_text(encoding="utf-8"))
        readme = pyproject["project"]["readme"]

        assert readme["content-type"] == "text/markdown"
        assert "installed automatically by `thor-cuda`" in readme["text"]
        assert "not intended to be used directly" in readme["text"]


def test_release_bundle_pipelines_link_package_with_next_sm_cuda_compile():
    repo_root = Path(__file__).resolve().parents[4]
    cmake_contents = (repo_root / "CMakeLists.txt").read_text(encoding="utf-8")

    # sm120 compilation must no longer wait for the completed sm89 wheel; that
    # would serialize sm89 link/package instead of overlapping it with sm120
    # CUDA compilation.
    assert "add_dependencies(ThorCudaSm120Objects thor_kernel_wheel_sm89)" not in cmake_contents

    # The hard ordering edge is file-level: the completion stamp waits for the
    # shared host objects and every sm89 CUDA object, then releases sm120 source
    # compilation.  sm89 link/package is an independent branch from that point.
    assert "function(thor_pipeline_cuda_compile_after previous_sm previous_cuda_target next_sm next_cuda_target)" in cmake_contents
    assert "$<TARGET_OBJECTS:ThorHostObjects>" in cmake_contents
    assert "$<TARGET_OBJECTS:${previous_cuda_target}>" in cmake_contents
    assert "add_dependencies(${next_cuda_target} thor_cuda_sm${previous_sm}_compile_complete)" in cmake_contents
    assert "thor_pipeline_cuda_compile_after(89 ThorCudaSm89Objects 120 ThorCudaSm120Objects)" in cmake_contents

    # Compilation of the next SM overlaps the previous backend tail, but the
    # next backend target itself waits for the previous wheel stage.  Because
    # ThorCudaSm120Objects remains independent of this dependency, Ninja can
    # compile sm120 while sm89 device-links/links/packages, then must wait
    # before starting the sm120 device-link/link.
    assert "function(thor_pipeline_backend_after_wheel previous_sm previous_wheel_target next_sm next_backend_target)" in cmake_contents
    assert "add_dependencies(${next_backend_target} ${previous_wheel_target})" in cmake_contents
    assert "thor_pipeline_backend_after_wheel(89 thor_kernel_wheel_sm89 120 Thor)" in cmake_contents
    assert "add_dependencies(ThorCudaSm120Objects thor_kernel_wheel_sm89)" not in cmake_contents


def test_nsight_profile_launcher_is_packaged_as_a_console_script():
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    assert pyproject["project"]["scripts"]["thor-nsys-profile"] == "thor_nsys_profile:main"
    assert "src/thor_nsys_profile" in pyproject["tool"]["scikit-build"]["wheel"]["packages"]
