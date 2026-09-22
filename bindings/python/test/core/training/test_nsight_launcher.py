from __future__ import annotations

from pathlib import Path

import thor_nsys_profile as _nsight


def test_build_nsys_command_uses_cuda_profiler_range_and_sync_repeated_reports_without_callback(tmp_path: Path):
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
    assert not any(argument.startswith("--after-report-ready=") for argument in command)
    assert not (tmp_path / "report_ready.py").exists()
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
    assert not any(argument.startswith("--after-report-ready=") for argument in command)


def test_package_has_module_entrypoint_for_source_tree_launches():
    module_entrypoint = Path(_nsight.__file__).with_name("__main__.py")
    assert module_entrypoint.is_file()
    assert "raise SystemExit(main())" in module_entrypoint.read_text(encoding="utf-8")


def test_nsys_environment_disables_debuginfod_and_sets_control_directory(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DEBUGINFOD_URLS", "https://debuginfod.example.invalid")
    monkeypatch.setenv("UNRELATED_THOR_TEST_VARIABLE", "preserved")

    env = _nsight._nsys_environment(tmp_path)

    assert "DEBUGINFOD_URLS" not in env
    assert env["THOR_NSYS_CONTROL_DIR"] == str(tmp_path)
    assert env["UNRELATED_THOR_TEST_VARIABLE"] == "preserved"
