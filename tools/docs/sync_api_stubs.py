#!/usr/bin/env python3
"""Sync nanobind-generated Thor stubs into the documentation metadata snapshot."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import tempfile


_DESTINATION_RELATIVE = Path("docs/_api_stubs/thor-stubs")
_REQUIRED_ROOT_STUBS = (Path("__init__.pyi"), Path("_thor.pyi"))


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy an already-generated Thor .pyi tree into the documentation "
            "stub snapshot without importing Thor."
        )
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Path to the generated thor package directory containing __init__.pyi and _thor.pyi.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit nonzero when the checked-in documentation snapshot differs from source.",
    )
    return parser.parse_args()


def _validate_source(source: Path) -> Path:
    source = source.resolve()
    if not source.is_dir():
        raise ValueError(f"stub source is not a directory: {source}")

    missing = [str(relative) for relative in _REQUIRED_ROOT_STUBS if not (source / relative).is_file()]
    if missing:
        raise ValueError(
            f"stub source {source} is missing required generated file(s): {', '.join(missing)}"
        )
    return source


def _stub_files(root: Path) -> list[Path]:
    return sorted(path.relative_to(root) for path in root.rglob("*.pyi") if path.is_file())


def _copy_stub_tree(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for relative in _stub_files(source):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)


def _trees_match(source: Path, destination: Path) -> bool:
    source_files = _stub_files(source)
    if not destination.is_dir() or source_files != _stub_files(destination):
        return False
    return all((source / relative).read_bytes() == (destination / relative).read_bytes() for relative in source_files)


def main() -> int:
    args = _parse_args()
    try:
        source = _validate_source(args.source)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    destination = _repository_root() / _DESTINATION_RELATIVE

    if args.check:
        if _trees_match(source, destination):
            print(f"Thor documentation API stub snapshot is current: {destination}")
            return 0
        print(
            "error: Thor documentation API stub snapshot is stale; "
            f"run {Path(__file__).as_posix()} {source}",
            file=sys.stderr,
        )
        return 1

    with tempfile.TemporaryDirectory(prefix="thor-doc-stubs-") as temporary_directory:
        staged = Path(temporary_directory) / "thor-stubs"
        _copy_stub_tree(source, staged)
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(staged, destination)

    print(f"Synced {_len_stub_files(destination)} Thor stub files to {destination}")
    return 0


def _len_stub_files(root: Path) -> int:
    return len(_stub_files(root))


if __name__ == "__main__":
    raise SystemExit(main())
