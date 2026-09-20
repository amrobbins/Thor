from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import thor_nsys_profile as _nsight


def test_build_nsys_command_uses_cuda_profiler_range_and_sync_repeated_reports(tmp_path: Path):
    command = _nsight._build_nsys_command(tmp_path, ["python", "train.py"])

    assert command[:2] == ["nsys", "profile"]
    assert "--trace=cuda,nvtx,osrt" in command
    assert "--sample=none" in command
    assert "--cpuctxsw=none" in command
    assert "--backtrace=none" in command
    assert "--show-source-info=false" in command
    assert "--resolve-symbols=false" in command
    assert "--capture-range=cudaProfilerApi" in command
    assert "--capture-range-end=repeat:sync" in command
    assert f"--output={tmp_path / 'capture-%p'}" in command
    callback_argument = next(argument for argument in command if argument.startswith("--after-report-ready="))
    callback = Path(callback_argument.split("=", 1)[1])
    assert callback.is_file()
    assert os.access(callback, os.X_OK)
    callback_text = callback.read_text(encoding="utf-8")
    assert callback_text.startswith(f"#!{sys.executable}\n")
    package_root = Path(_nsight.__file__).resolve().parent.parent
    assert f"sys.path.insert(0, {str(package_root)!r})" in callback_text
    assert command[-2:] == ["python", "train.py"]



def test_build_nsys_command_only_uses_lean_flags_supported_by_installed_nsys(tmp_path: Path):
    help_text = "--cpuctxsw --backtrace --resolve-symbols"
    command = _nsight._build_nsys_command(
        tmp_path,
        ["python", "train.py"],
        nsys_profile_help=help_text,
    )

    assert "--cpuctxsw=none" in command
    assert "--backtrace=none" in command
    assert "--resolve-symbols=false" in command
    assert "--show-source-info=false" not in command

def test_report_ready_callback_moves_multiple_reports_in_request_order(tmp_path: Path):
    callback = _nsight._callback_script(tmp_path)
    destinations = [
        tmp_path / "requested" / "glm.nsys-rep",
        tmp_path / "requested" / "transformer.nsys-rep",
    ]
    for index, destination in enumerate(destinations):
        (tmp_path / f"profile_output_path.{index:020d}").write_text(
            str(destination) + "\n", encoding="utf-8"
        )

    for index, destination in enumerate(destinations):
        source = tmp_path / f"capture-{index}.nsys-rep"
        source.write_bytes(f"profile-{index}".encode())
        env = os.environ.copy()
        env["NSYS_REPORT_PATH"] = str(source)
        completed = subprocess.run([str(callback)], env=env, check=False)
        assert completed.returncode == 0
        assert not source.exists()
        assert destination.read_bytes() == f"profile-{index}".encode()
        assert not (tmp_path / f"profile_output_path.{index:020d}").exists()
        assert (tmp_path / f"report_output_path.{index:020d}").read_text(encoding="utf-8").strip() == str(
            destination
        )


def test_report_ready_callback_uses_repeat_report_ordinal_when_callbacks_finish_out_of_order(tmp_path: Path):
    callback = _nsight._callback_script(tmp_path)
    destinations = [
        tmp_path / "requested" / "glm.nsys-rep",
        tmp_path / "requested" / "transformer.nsys-rep",
    ]
    for index, destination in enumerate(destinations):
        (tmp_path / f"profile_output_path.{index:020d}").write_text(
            str(destination) + "\n", encoding="utf-8"
        )

    first = tmp_path / "capture-1234.1.nsys-rep"
    second = tmp_path / "capture-1234.2.nsys-rep"
    first.write_bytes(b"glm")
    second.write_bytes(b"transformer")

    # Simulate the second asynchronous finalizer invoking its callback first.
    env = os.environ.copy()
    env["NSYS_REPORT_PATH"] = str(second)
    completed = subprocess.run([str(callback)], env=env, check=False)
    assert completed.returncode == 0
    assert destinations[1].read_bytes() == b"transformer"
    assert not destinations[0].exists()

    env["NSYS_REPORT_PATH"] = str(first)
    completed = subprocess.run([str(callback)], env=env, check=False)
    assert completed.returncode == 0
    assert destinations[0].read_bytes() == b"glm"


def test_package_has_module_entrypoint_for_source_tree_launches():
    module_entrypoint = Path(_nsight.__file__).with_name("__main__.py")
    assert module_entrypoint.is_file()
    assert "raise SystemExit(main())" in module_entrypoint.read_text(encoding="utf-8")


def test_process_exit_recovery_moves_finalized_report_when_callback_did_not_run(tmp_path: Path):
    destination = tmp_path / "requested" / "glm.nsys-rep"
    (tmp_path / "profile_output_path.00000000000000000000").write_text(
        str(destination) + "\n", encoding="utf-8"
    )
    source = tmp_path / "capture-4321.1.nsys-rep"
    source.write_bytes(b"glm")

    failures = _nsight._recover_finalized_reports(tmp_path)

    assert failures == 0
    assert not source.exists()
    assert destination.read_bytes() == b"glm"
    assert not (tmp_path / "profile_output_path.00000000000000000000").exists()
    assert (tmp_path / "report_output_path.00000000000000000000").read_text(encoding="utf-8").strip() == str(
        destination
    )


def test_report_ready_marker_is_published_only_after_destination_exists(tmp_path: Path):
    destination = tmp_path / "requested" / "transformer.nsys-rep"
    request = tmp_path / "profile_output_path.00000000000000000000"
    request.write_text(str(destination) + "\n", encoding="utf-8")
    source = tmp_path / "capture-4321.1.nsys-rep"
    source.write_bytes(b"transformer")

    assert _nsight._relocate_ready_report(tmp_path, source) == 0

    marker = tmp_path / "report_output_path.00000000000000000000"
    assert destination.is_file()
    assert marker.is_file()
    assert marker.read_text(encoding="utf-8").strip() == str(destination)


def test_nsys_environment_disables_debuginfod_and_sets_control_directory(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DEBUGINFOD_URLS", "https://debuginfod.example.invalid")
    monkeypatch.setenv("UNRELATED_THOR_TEST_VARIABLE", "preserved")

    env = _nsight._nsys_environment(tmp_path)

    assert "DEBUGINFOD_URLS" not in env
    assert env["THOR_NSYS_CONTROL_DIR"] == str(tmp_path)
    assert env["UNRELATED_THOR_TEST_VARIABLE"] == "preserved"
