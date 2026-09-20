"""Nsight Systems launcher for Trainer-controlled epoch capture windows."""

from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

_CONTROL_DIR_ENV = "THOR_NSYS_CONTROL_DIR"
_PROFILE_OUTPUT_REQUEST_PREFIX = "profile_output_path."
_REPORT_OUTPUT_PREFIX = "report_output_path."


def _nsys_environment(control_dir: Path) -> dict[str, str]:
    """Return the environment for Nsight without remote debug-symbol lookup."""
    env = os.environ.copy()
    # Nsight Systems/libdw can use DEBUGINFOD_URLS to contact remote symbol
    # servers while finalizing a report.  These GPU timeline captures do not
    # need host debug symbols, and network symbol lookup can stall training for
    # minutes at cudaProfilerStop(), so explicitly prevent the profiler process
    # and its target from inheriting that opt-in.
    env.pop("DEBUGINFOD_URLS", None)
    env[_CONTROL_DIR_ENV] = str(control_dir)
    return env


def _report_ordinal(report: Path) -> int | None:
    """Return Nsight's 1-based repeat-report ordinal when it is present."""
    parts = report.name.split(".")
    if len(parts) >= 3 and parts[-1] == "nsys-rep" and parts[-2].isdigit():
        ordinal = int(parts[-2])
        if ordinal >= 1:
            return ordinal
    return None


def _report_sort_key(report: Path) -> tuple[int, str]:
    ordinal = _report_ordinal(report)
    return (ordinal if ordinal is not None else sys.maxsize, report.name)


def _relocate_ready_report(control_dir: Path, report: Path) -> int:
    """Move one finalized Nsight report to the Trainer-requested destination.

    The file lock makes this safe both for Nsight's report-ready callback and
    for the launcher's process-exit recovery pass.  A completion marker is
    published atomically only after the final report is present at its requested
    path.
    """
    control_dir = Path(control_dir)
    report = Path(report)
    lock_path = control_dir / "report_ready.lock"
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        request_files = sorted(control_dir.glob(_PROFILE_OUTPUT_REQUEST_PREFIX + "*"))
        if not request_files:
            print(
                f"Thor: Nsight report is ready at {report}, but no Trainer output request was published.",
                file=sys.stderr,
            )
            return 2

        request_file: Path | None = None
        ordinal = _report_ordinal(report)
        if ordinal is not None:
            ordinal_request = control_dir / (
                _PROFILE_OUTPUT_REQUEST_PREFIX + f"{ordinal - 1:020d}"
            )
            if ordinal_request.exists():
                request_file = ordinal_request
        if request_file is None:
            request_file = request_files[0]

        destination_text = request_file.read_text(encoding="utf-8").strip()
        if not destination_text:
            print("Thor: Nsight Trainer output request is empty.", file=sys.stderr)
            return 2
        destination = Path(destination_text)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            print(
                f"Thor: refusing to overwrite existing Nsight report {destination}.",
                file=sys.stderr,
            )
            return 2
        if not report.is_file():
            print(f"Thor: finalized Nsight report is missing: {report}.", file=sys.stderr)
            return 2

        shutil.move(str(report), str(destination))
        if not destination.is_file():
            print(
                f"Thor: Nsight report relocation did not produce {destination}.",
                file=sys.stderr,
            )
            return 2

        request_file.unlink()
        suffix = request_file.name.removeprefix(_PROFILE_OUTPUT_REQUEST_PREFIX)
        marker = control_dir / (_REPORT_OUTPUT_PREFIX + suffix)
        marker_tmp = control_dir / f".{marker.name}.tmp-{os.getpid()}"
        marker_tmp.write_text(str(destination) + "\n", encoding="utf-8")
        os.replace(marker_tmp, marker)
        print(f"Thor: Nsight Systems report ready: {destination}", file=sys.stderr)
        return 0


def _callback_script(control_dir: Path) -> Path:
    script = control_dir / "report_ready.py"
    # The launcher can run directly from a source checkout even when the
    # thor-nsys-profile console script has not been installed.  Nsight invokes
    # this callback as a fresh Python process, so make the package root explicit
    # instead of depending on the launcher's transient sys.path.
    package_root = Path(__file__).resolve().parent.parent
    content = f'''#!{sys.executable}
from __future__ import annotations
import os
from pathlib import Path
import sys
sys.path.insert(0, {str(package_root)!r})
from thor_nsys_profile import _relocate_ready_report

control_dir = Path({str(control_dir)!r})
report = os.environ.get("NSYS_REPORT_PATH")
if not report:
    print("Thor: Nsight report callback did not receive NSYS_REPORT_PATH.", file=sys.stderr)
    raise SystemExit(2)
raise SystemExit(_relocate_ready_report(control_dir, Path(report)))
'''
    script.write_text(content, encoding="utf-8")
    script.chmod(0o700)
    return script


def _recover_finalized_reports(control_dir: Path) -> int:
    """Best-effort fallback if an Nsight report-ready callback did not run."""
    failures = 0
    for report in sorted(control_dir.glob("capture-*.nsys-rep"), key=_report_sort_key):
        if _relocate_ready_report(control_dir, report) != 0:
            failures += 1
    return failures


def _build_nsys_command(
    control_dir: Path,
    command: list[str],
    *,
    nsys_profile_help: str | None = None,
) -> list[str]:
    callback = _callback_script(control_dir)
    # These captures are intended to answer GPU timeline/synchronization
    # questions. Disable CPU sampling, context-switch collection, CPU
    # backtraces, source-line lookup, and symbol resolution so report
    # generation cannot spend minutes downloading host debug symbols that are
    # irrelevant to the CUDA/NVTX/OS-runtime timeline.
    #
    # repeat:sync materializes every Trainer-selected cudaProfilerStart/Stop
    # range independently and blocks the application thread at cudaProfilerStop()
    # until Nsight has finished generating that report. The Trainer then waits
    # for the report-ready callback to finish relocating the report before the
    # training phase resumes.
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
        f"--after-report-ready={callback}",
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

    # If Nsight generated a report but its report-ready callback failed for any
    # reason, do not strand the artifact in /tmp.  At target-process exit all
    # generated reports are finalized, so a second relocation attempt is safe.
    _recover_finalized_reports(control_dir)

    pending_requests = list(control_dir.glob(_PROFILE_OUTPUT_REQUEST_PREFIX + "*"))
    if not pending_requests:
        shutil.rmtree(control_dir, ignore_errors=True)
    else:
        # Keep the directory only when a requested capture was never produced or
        # could not be relocated.  It contains the exact destination request and
        # any remaining raw report for manual recovery.
        print(
            f"Thor: Nsight report relocation did not complete; preserving profiler files in {control_dir}",
            file=sys.stderr,
        )
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
