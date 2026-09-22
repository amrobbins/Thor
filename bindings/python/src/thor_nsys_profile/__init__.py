"""Nsight Systems launcher for Trainer-controlled epoch capture windows."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

_CONTROL_DIR_ENV = "THOR_NSYS_CONTROL_DIR"


def _nsys_environment(control_dir: Path) -> dict[str, str]:
    """Return the environment for Nsight without remote debug-symbol lookup."""
    env = os.environ.copy()
    # Nsight Systems/libdw can use DEBUGINFOD_URLS to contact remote symbol
    # servers while finalizing a report. These GPU timeline captures do not
    # need host debug symbols, and network symbol lookup can stall training for
    # minutes at cudaProfilerStop(), so explicitly prevent the profiler process
    # and its target from inheriting that opt-in.
    env.pop("DEBUGINFOD_URLS", None)
    env[_CONTROL_DIR_ENV] = str(control_dir)
    return env


def _build_nsys_command(
    control_dir: Path,
    command: list[str],
    *,
    nsys_profile_help: str | None = None,
) -> list[str]:
    # These captures are intended to answer GPU timeline/synchronization
    # questions. Disable CPU sampling, context-switch collection, CPU
    # backtraces, source-line lookup, and symbol resolution so report
    # generation cannot spend minutes downloading host debug symbols that are
    # irrelevant to the CUDA/NVTX/OS-runtime timeline.
    #
    # repeat:sync materializes every Trainer-selected cudaProfilerStart/Stop
    # range independently and blocks the application thread in
    # cudaProfilerStop() until that report is finalized. The native Trainer
    # therefore owns synchronous relocation after cudaProfilerStop() returns;
    # no Nsight report-ready callback is required.
    profiler_options = [
        "--trace=cuda,nvtx,osrt",
        "--sample=none",
    ]
    optional_lean_options = [
        ("--cpuctxsw", "--cpuctxsw=none"),
        ("--backtrace", "--backtrace=none"),
        ("--show-source-info", "--show-source-info=false"),
        ("--resolve-symbols", "--resolve-symbols=false"),
    ]
    for option_name, argument in optional_lean_options:
        if nsys_profile_help is None or option_name in nsys_profile_help:
            profiler_options.append(argument)

    return [
        "nsys",
        "profile",
        *profiler_options,
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=repeat:sync",
        f"--output={control_dir / 'capture-%p'}",
        *command,
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="thor-nsys-profile",
        description=(
            "Launch a Thor training command under Nsight Systems. The Trainer's "
            "nsight_profile parameter chooses the phase-relative capture window and final .nsys-rep path."
        ),
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="command to launch (for example: python train.py)",
    )
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a training command is required")
    if shutil.which("nsys") is None:
        parser.error("Nsight Systems CLI `nsys` was not found on PATH")

    control_dir = Path(tempfile.mkdtemp(prefix="thor-nsys-"))
    env = _nsys_environment(control_dir)
    profile_help = ""
    try:
        help_result = subprocess.run(
            ["nsys", "profile", "--help"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
        )
        profile_help = (help_result.stdout or "") + "\n" + (help_result.stderr or "")
    except (OSError, subprocess.SubprocessError):
        # The required profiling switches are already part of Thor's supported
        # Nsight contract. If help probing fails, simply omit the optional
        # symbol/CPU-detail suppression switches rather than risking an
        # unsupported-option failure on an older Nsight installation.
        profile_help = ""
    nsys_command = _build_nsys_command(control_dir, command, nsys_profile_help=profile_help)
    completed = subprocess.run(nsys_command, env=env, check=False)

    # The native Trainer normally relocates each finalized report immediately
    # after synchronous cudaProfilerStop(). If a process exits/crashes before
    # that relocation completes, preserve any raw report for manual recovery
    # rather than guessing which configured destination it belonged to.
    remaining_reports = sorted(control_dir.glob("capture-*.nsys-rep"))
    if remaining_reports:
        print(
            f"Thor: preserving unrelocated Nsight profiler files in {control_dir}",
            file=sys.stderr,
        )
    else:
        shutil.rmtree(control_dir, ignore_errors=True)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
