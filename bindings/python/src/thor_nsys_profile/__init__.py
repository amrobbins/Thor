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
_PROFILE_OUTPUT_REQUEST_PREFIX = "profile_output_path."


def _callback_script(control_dir: Path) -> Path:
    script = control_dir / "report_ready.py"
    content = f'''#!{sys.executable}
from __future__ import annotations
import fcntl
import os
from pathlib import Path
import shutil
import sys

control_dir = Path({str(control_dir)!r})
request_prefix = {_PROFILE_OUTPUT_REQUEST_PREFIX!r}
report = os.environ.get("NSYS_REPORT_PATH")
if not report:
    print("Thor: Nsight report callback did not receive NSYS_REPORT_PATH.", file=sys.stderr)
    raise SystemExit(2)
lock_path = control_dir / "report_ready.lock"
with lock_path.open("a+") as lock_file:
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
    request_files = sorted(control_dir.glob(request_prefix + "*"))
    if not request_files:
        print(f"Thor: Nsight report is ready at {{report}}, but no Trainer output request was published.", file=sys.stderr)
        raise SystemExit(2)

    # With --capture-range-end=repeat:async Nsight names sequential reports
    # <base>.1.nsys-rep, <base>.2.nsys-rep, ... . Match that ordinal to
    # Thor's zero-based request sequence so asynchronously finalized reports
    # cannot be assigned to the wrong destination. Fall back to the oldest
    # pending request for profiler versions/names that do not expose an ordinal
    # or when an earlier capture attempt consumed a sequence without producing
    # a report.
    source = Path(report)
    request_file = None
    report_parts = source.name.split(".")
    if len(report_parts) >= 3 and report_parts[-1] == "nsys-rep" and report_parts[-2].isdigit():
        report_ordinal = int(report_parts[-2])
        if report_ordinal >= 1:
            ordinal_request = control_dir / (request_prefix + f"{{report_ordinal - 1:020d}}")
            if ordinal_request.exists():
                request_file = ordinal_request
    if request_file is None:
        request_file = request_files[0]
    destination_text = request_file.read_text(encoding="utf-8").strip()
    if not destination_text:
        print("Thor: Nsight Trainer output request is empty.", file=sys.stderr)
        raise SystemExit(2)
    destination = Path(destination_text)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"Thor: refusing to overwrite existing Nsight report {{destination}}.", file=sys.stderr)
        raise SystemExit(2)
    shutil.move(str(source), str(destination))
    request_file.unlink()
    marker = control_dir / ("report_output_path." + request_file.name.removeprefix(request_prefix))
    marker.write_text(str(destination) + "\\n", encoding="utf-8")
    print(f"Thor: Nsight Systems report ready: {{destination}}", file=sys.stderr)
'''
    script.write_text(content, encoding="utf-8")
    script.chmod(0o700)
    return script


def _build_nsys_command(control_dir: Path, command: list[str]) -> list[str]:
    callback = _callback_script(control_dir)
    # repeat:sync materializes every Trainer-selected cudaProfilerStart/Stop
    # range independently and blocks the application thread at cudaProfilerStop()
    # until Nsight has finished generating that report. The Trainer then waits
    # for the report-ready callback to finish relocating the report before the
    # training phase resumes.
    return [
        "nsys",
        "profile",
        "--trace=cuda,nvtx,osrt",
        "--sample=none",
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
    env = os.environ.copy()
    env[_CONTROL_DIR_ENV] = str(control_dir)
    nsys_command = _build_nsys_command(control_dir, command)
    completed = subprocess.run(nsys_command, env=env, check=False)

    pending_requests = list(control_dir.glob(_PROFILE_OUTPUT_REQUEST_PREFIX + "*"))
    if not pending_requests:
        shutil.rmtree(control_dir, ignore_errors=True)
    else:
        # Do not delete reports if an after-report-ready callback failed. Keeping
        # this unique directory gives the user a recovery path for every pending
        # capture in the multi-phase specification.
        print(
            f"Thor: Nsight report relocation did not complete; preserving profiler files in {control_dir}",
            file=sys.stderr,
        )
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
