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
    assert "--capture-range=cudaProfilerApi" in command
    assert "--capture-range-end=repeat:sync" in command
    assert f"--output={tmp_path / 'capture-%p'}" in command
    callback_argument = next(argument for argument in command if argument.startswith("--after-report-ready="))
    callback = Path(callback_argument.split("=", 1)[1])
    assert callback.is_file()
    assert os.access(callback, os.X_OK)
    assert callback.read_text(encoding="utf-8").startswith(f"#!{sys.executable}\n")
    assert command[-2:] == ["python", "train.py"]


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
