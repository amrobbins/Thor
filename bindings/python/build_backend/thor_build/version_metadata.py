"""Shared dynamic version provider for Thor wheel distributions."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Mapping


_REPO_ROOT = Path(__file__).resolve().parents[4]
_THOR_VERSION_HEADER = _REPO_ROOT / "ThorVersion.h.in"
_THOR_VERSION_RE = re.compile(r'^#define\s+THOR_VERSION\s+"v?(?P<value>[^"]+)"$', re.MULTILINE)


def thor_version() -> str:
    """Return the canonical Thor distribution version from ``ThorVersion.h.in``."""

    match = _THOR_VERSION_RE.search(_THOR_VERSION_HEADER.read_text(encoding="utf-8"))
    if match is None:
        raise RuntimeError(f"Unable to read THOR_VERSION from {_THOR_VERSION_HEADER}")
    return match.group("value")


def dynamic_metadata(
    field: str,
    settings: Mapping[str, Any] | None = None,
    project: Mapping[str, Any] | None = None,
) -> str:
    if field != "version":
        raise RuntimeError(f"Thor version metadata provider only supports version, got {field!r}")
    if settings:
        raise RuntimeError(f"Thor version metadata provider does not accept settings: {sorted(settings)}")

    return thor_version()
