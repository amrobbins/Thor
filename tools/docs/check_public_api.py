#!/usr/bin/env python3
"""Check Thor's public Python API documentation without importing Thor."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
import re
import sys


_MKDOCSTRINGS_DIRECTIVE = re.compile(r"^\s*:::\s+([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*$")
_INCOMPLETE_MARKER = re.compile(r"\b(?:FIXME|TODO)\b", re.IGNORECASE)


@dataclass(frozen=True)
class NamespaceInventory:
    module: str
    public_names: frozenset[str]
    resolved: bool
    source: str


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inventory Thor's public Python API and validate API-reference coverage "
            "without importing thor or loading _thor.so."
        )
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help=(
            "Fail when any resolved public symbol lacks an mkdocstrings directive or "
            "when a public namespace cannot be resolved without native stub metadata."
        ),
    )
    parser.add_argument(
        "--show-all-missing",
        action="store_true",
        help="Print every currently undocumented resolved public symbol.",
    )
    return parser.parse_args()


def _module_name(package_root: Path, init_file: Path) -> str:
    relative = init_file.parent.relative_to(package_root.parent)
    return ".".join(relative.parts)


def _public_package_modules(package_root: Path) -> dict[str, Path]:
    modules: dict[str, Path] = {}
    for init_file in sorted(package_root.rglob("__init__.py")):
        relative_parts = init_file.parent.relative_to(package_root).parts
        if any(part.startswith("_") for part in relative_parts):
            continue
        modules[_module_name(package_root, init_file)] = init_file
    return modules


def _literal_string_collection(node: ast.AST) -> set[str] | None:
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError):
        return None
    if not isinstance(value, (list, tuple, set, frozenset)):
        return None
    if not all(isinstance(item, str) for item in value):
        return None
    return set(value)


def _source_public_names(init_file: Path) -> set[str] | None:
    tree = ast.parse(init_file.read_text(encoding="utf-8"), filename=str(init_file))
    for statement in tree.body:
        value: ast.AST | None = None
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__" for target in statement.targets
        ):
            value = statement.value
        elif (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == "__all__"
        ):
            value = statement.value
        if value is not None:
            return _literal_string_collection(value)
    return None


def _stub_file(stub_root: Path, module: str) -> Path:
    parts = module.split(".")
    assert parts[0] == "thor"
    relative = Path(*parts[1:])
    return stub_root / relative / "__init__.pyi" if relative.parts else stub_root / "__init__.pyi"


def _stub_public_names(stub_file: Path) -> set[str] | None:
    if not stub_file.is_file():
        return None
    tree = ast.parse(stub_file.read_text(encoding="utf-8"), filename=str(stub_file))

    for statement in tree.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__" for target in statement.targets
        ):
            literal = _literal_string_collection(statement.value)
            if literal is not None:
                return literal

    names: set[str] = set()
    for statement in tree.body:
        if isinstance(statement, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if not statement.name.startswith("_"):
                names.add(statement.name)
        elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            if not statement.target.id.startswith("_"):
                names.add(statement.target.id)
        elif isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    names.add(target.id)
    return names or None


def _build_inventory(root: Path) -> tuple[dict[str, NamespaceInventory], set[str]]:
    package_root = root / "bindings/python/src/thor"
    stub_root = root / "docs/_api_stubs/thor-stubs"
    modules = _public_package_modules(package_root)
    module_names = set(modules)
    inventory: dict[str, NamespaceInventory] = {}

    for module, init_file in sorted(modules.items()):
        names = _source_public_names(init_file)
        if names is not None:
            inventory[module] = NamespaceInventory(module, frozenset(names), True, "python __all__")
            continue

        stub_names = _stub_public_names(_stub_file(stub_root, module))
        if stub_names is not None:
            inventory[module] = NamespaceInventory(module, frozenset(stub_names), True, "generated .pyi")
            continue

        inventory[module] = NamespaceInventory(module, frozenset(), False, "native metadata unavailable")

    return inventory, module_names


def _public_symbol_paths(
    inventory: dict[str, NamespaceInventory], module_names: set[str]
) -> set[str]:
    symbols: set[str] = set()
    for module, namespace in inventory.items():
        if not namespace.resolved:
            continue
        for name in namespace.public_names:
            candidate = f"{module}.{name}"
            if candidate in module_names:
                continue
            symbols.add(candidate)
    return symbols


def _reference_directives(reference_root: Path) -> tuple[set[str], list[str]]:
    directives: set[str] = set()
    malformed: list[str] = []
    for markdown in sorted(reference_root.rglob("*.md")):
        for line_number, line in enumerate(markdown.read_text(encoding="utf-8").splitlines(), start=1):
            if ":::" not in line:
                continue
            match = _MKDOCSTRINGS_DIRECTIVE.fullmatch(line)
            if match:
                directives.add(match.group(1))
            elif line.lstrip().startswith(":::"):
                malformed.append(f"{markdown.relative_to(reference_root.parent)}:{line_number}: {line.strip()}")
    return directives, malformed


def _incomplete_markers(reference_root: Path) -> list[str]:
    findings: list[str] = []
    for markdown in sorted(reference_root.rglob("*.md")):
        for line_number, line in enumerate(markdown.read_text(encoding="utf-8").splitlines(), start=1):
            if _INCOMPLETE_MARKER.search(line):
                findings.append(f"{markdown.relative_to(reference_root.parent)}:{line_number}: {line.strip()}")
    return findings


def _directive_status(
    directive: str,
    inventory: dict[str, NamespaceInventory],
    module_names: set[str],
) -> str:
    if directive in module_names:
        return "public"

    parts = directive.split(".")
    for split_index in range(len(parts) - 1, 0, -1):
        module = ".".join(parts[:split_index])
        namespace = inventory.get(module)
        if namespace is None:
            continue
        if not namespace.resolved:
            return "unverified"
        first_object = parts[split_index]
        return "public" if first_object in namespace.public_names else "stale"
    return "stale"


def _print_items(label: str, items: list[str], *, limit: int | None = None) -> None:
    print(label)
    visible = items if limit is None else items[:limit]
    for item in visible:
        print(f"  - {item}")
    if limit is not None and len(items) > limit:
        print(f"  ... and {len(items) - limit} more")


def main() -> int:
    args = _parse_args()
    root = _repository_root()
    reference_root = root / "docs/reference"

    inventory, module_names = _build_inventory(root)
    public_symbols = _public_symbol_paths(inventory, module_names)
    directives, malformed = _reference_directives(reference_root)
    markers = _incomplete_markers(reference_root)

    stale = sorted(
        directive
        for directive in directives
        if _directive_status(directive, inventory, module_names) == "stale"
    )
    unverified_directives = sorted(
        directive
        for directive in directives
        if _directive_status(directive, inventory, module_names) == "unverified"
    )
    documented_public = {
        directive
        for directive in directives
        if _directive_status(directive, inventory, module_names) == "public"
    }
    missing = sorted(public_symbols - documented_public)
    unresolved_namespaces = sorted(
        module for module, namespace in inventory.items() if not namespace.resolved
    )

    print("Thor public API documentation inventory")
    print(f"  public package namespaces: {len(inventory)}")
    print(f"  statically resolved public symbols: {len(public_symbols)}")
    print(f"  mkdocstrings directives: {len(directives)}")
    print(f"  resolved symbols without directives: {len(missing)}")
    print(f"  namespaces awaiting native stub metadata: {len(unresolved_namespaces)}")

    if missing:
        limit = None if args.show_all_missing else 25
        _print_items("Undocumented resolved public symbols:", missing, limit=limit)
    if unresolved_namespaces:
        _print_items("Namespaces awaiting generated native stub metadata:", unresolved_namespaces)
    if unverified_directives:
        _print_items("Directives that cannot yet be verified without native metadata:", unverified_directives)

    errors = False
    if stale:
        _print_items("error: API directives that do not name a public symbol:", stale)
        errors = True
    if malformed:
        _print_items("error: malformed mkdocstrings directives:", malformed)
        errors = True
    if markers:
        _print_items("error: incomplete markers in API reference Markdown:", markers)
        errors = True

    if args.require_complete:
        if missing:
            print("error: resolved public API documentation is incomplete", file=sys.stderr)
            errors = True
        if unresolved_namespaces:
            print(
                "error: public API completeness cannot be proven until generated native stubs are available",
                file=sys.stderr,
            )
            errors = True
        if unverified_directives:
            print("error: some API directives cannot be verified", file=sys.stderr)
            errors = True

    if errors:
        return 1

    if missing or unresolved_namespaces:
        print(
            "Coverage is currently informational; stale directives and incomplete reference markers are gated. "
            "Use --require-complete once the API reference and native stub snapshot are complete."
        )
    else:
        print("Public API documentation coverage is complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
