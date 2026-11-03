#!/usr/bin/env python3
"""Inventory compiled FP8 CUDA kernels in *production* ThorCubCudaObjects.

The selected-inventory CMake target builds isolated source copies; it cannot
answer whether additional FP8 kernels still exist in production CUDA objects.
This independent read-only census scans the actual built .cu.o files. It does
not decide whether an extra FP8 kernel is legitimate (R/KR, ARG and view paths
may own independent kernels). Review the 'other-fp8-kernel' rows individually.

Requires the CUDA toolkit cuobjdump, binutils readelf/c++filt, and a completed
ThorCubCudaObjects build. Produces a CSV and a summary; never changes sources
or build objects. Results are per-object/per-device-ELF, not linked libThor size.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

if __package__:
    from .fp8_selected_inventory_size import FAMILIES, is_named_kernel_function, parse_func_symbols
else:
    from fp8_selected_inventory_size import FAMILIES, is_named_kernel_function, parse_func_symbols

FP8_TYPE = re.compile(r'(?:__nv_fp8_(?:e4m3|e5m2)|\bFP8_(?:E4M3|E5M2)\b)', re.IGNORECASE)
# Matches first-party kernels, not CUDA device helpers that happen to process FP8.
KERNEL_FUNCTION = re.compile(r'\b\w*[Kk]ernel\s*(?:<|\()')


def classify_function(name: str, mangled: str = '') -> str:
    """Classify CUDA FUNCs by demangled identity, or exact mangled identifier.

    NVCC device ELFs can contain C++ spellings a host c++filt does not decode.
    Itanium mangling preserves each source identifier, with its byte length;
    use that identity rather than pretending unrecognized functions are absent.
    """
    family = next((family for kernel, family in FAMILIES.items()
                   if is_named_kernel_function(name, kernel)), None)
    if family is not None:
        # Rotated RK kernels also serve FP16/BF16/FP32 production.  Do not
        # count those non-FP8 instances as retained FP8 specializations just
        # because they share RuntimeK's kernel function name.
        if (family == 'RuntimeK' and
            'awkwardAlignmentRotatedShardedFirstStageKernel' in name and
            not (FP8_TYPE.search(name) or FP8_TYPE.search(mangled))):
            return 'other-device-function'
        return family
    for kernel, family in FAMILIES.items():
        if f'{len(kernel)}{kernel}' in mangled:
            if (family == 'RuntimeK' and
                kernel == 'awkwardAlignmentRotatedShardedFirstStageKernel' and
                not FP8_TYPE.search(mangled)):
                return 'other-device-function'
            return family
    if 'ThorImplementation::' in name and KERNEL_FUNCTION.search(name) and (
        FP8_TYPE.search(name) or re.search(r'\b\w*[Ff]p8\w*[Kk]ernel\s*(?:<|\()', name)
    ):
        return 'other-fp8-kernel'
    if mangled.startswith('_Z') and 'ThorImplementation' in mangled and (
        re.search(r'[Ff]p8[A-Za-z0-9_]*[Kk]ernel', mangled) or
        (FP8_TYPE.search(mangled) and re.search(r'[Kk]ernel', mangled))
    ):
        return 'other-fp8-kernel'
    return 'other-device-function'


def run(cmd: list[str], *, cwd: Path | None = None, input_text: str | None = None) -> str:
    result = subprocess.run(cmd, cwd=cwd, input=input_text, capture_output=True,
                            text=True, errors='replace', check=False)
    if result.returncode:
        raise RuntimeError(f'Command failed ({result.returncode}): {cmd}\n'
                           f'{result.stdout[-1500:]}\n{result.stderr[-2500:]}')
    return result.stdout


def demangle_func_symbols(symbols: list[tuple[str, int]], source: str) -> list[str]:
    """Demangle every FUNC symbol without relying on c++filt's stdin line framing.

    Some device ELFs contain long/complex symbols whose text-mode c++filt
    output isn't line-for-line with the input stream. Pass names as individual
    argv entries in bounded batches instead. Never silently truncate a report
    with zip(symbols, demangled) if cardinality differs.
    """
    names: list[str] = []
    batch: list[str] = []
    arg_bytes = 0

    def flush() -> None:
        nonlocal arg_bytes
        if not batch:
            return
        output = run(['c++filt', '-n', '--', *batch])
        demangled = output.splitlines()
        if len(demangled) != len(batch) or any(not name for name in demangled):
            # Isolate an exceptional spelling rather than assigning subsequent
            # demangled names to the wrong FUNCs and corrupting the census.
            demangled = []
            for mangled in batch:
                single = run(['c++filt', '-n', '--', mangled]).rstrip('\r\n')
                if not single:
                    raise RuntimeError(f'c++filt produced no name for {mangled!r} in {source}')
                # Preserve embedded newlines visibly while keeping the CSV row
                # and symbol count unambiguous.
                demangled.append(single.replace('\r', r'\r').replace('\n', r'\n'))
        names.extend(demangled)
        batch.clear()
        arg_bytes = 0

    for mangled, _ in symbols:
        size = len(mangled.encode('utf-8')) + 1
        if batch and (len(batch) >= 128 or arg_bytes + size > 48 * 1024):
            flush()
        batch.append(mangled)
        arg_bytes += size
    flush()
    if len(names) != len(symbols):
        raise RuntimeError(f'Expected {len(symbols)} demangled FUNC symbols, '
                           f'got {len(names)} in {source}')
    return names


def selected_manifest_counts(build: Path) -> Counter[str] | None:
    """Read the *generated* isolated selector inventory, if present.

    This is a count comparator, not proof that same-named production kernels
    correspond one-for-one to retained candidate identities. The complete
    source-level manifest remains the authority for selector reachability.
    """
    manifest = build / 'fp8-selected-inventory' / 'expected.json'
    if not manifest.exists():
        return None
    entries = json.loads(manifest.read_text())
    if not isinstance(entries, list) or any(
        not isinstance(entry, dict) or entry.get('mode') not in set(FAMILIES.values())
        for entry in entries
    ):
        raise RuntimeError(f'Invalid selected-inventory manifest: {manifest}')
    return Counter(entry['mode'] for entry in entries)


def family_count_discrepancies(production: Counter[str], selected: Counter[str]) -> dict[str, int]:
    """Compiled family-name count minus selected manifest count, NOT extra identities."""
    return {family: production[family] - selected[family]
            for family in dict.fromkeys(FAMILIES.values())}



def _architecture(cubin_name: str) -> str:
    """Keep CUDA architectures independent; sm_89 and sm_120 are not duplicates."""
    match = re.search(r'sm[_-](\d+)', cubin_name)
    if match is None:
        raise RuntimeError(f'Cannot identify CUDA architecture in {cubin_name!r}')
    return f'sm_{match.group(1)}'


# The isolated CUDA TUs are copies of these production TUs.  The compiler
# assigns different hashed anonymous-namespace names in separate builds, so
# the raw Itanium spellings are not comparable byte for byte.  Compare the
# *full demangled kernel signature* within its original source TU instead;
# this preserves every template argument while removing the generated salt.
SELECTED_SOURCE_TUS = {
    'narrow': 'CubReductionNarrowFlatRCooperative.cu.o',
    'aligned': 'CubReductionKParallelShardedTiled.cu.o',
    'rotated': 'CubReductionAwkwardRotatedSharded.cu.o',
}
PRODUCTION_SOURCE_TUS = {name: short for short, name in SELECTED_SOURCE_TUS.items()}


def _kernel_source_group(row: dict, *, selected: bool) -> str:
    obj = Path(row['object']).name
    if selected:
        if obj not in SELECTED_SOURCE_TUS:
            raise RuntimeError(f'Unknown isolated selector object {obj!r}')
        return obj
    # An unrelated production TU should never be counted as a copy of a
    # selected kernel, even if its anonymous-namespace function is homonymous.
    return PRODUCTION_SOURCE_TUS.get(obj, f'other:{obj}')


def compare_selected_kernel_identities(production_rows: list[dict],
                                       selected_rows: list[dict]) -> tuple[list[dict], Counter]:
    """Compare exact typed kernel *signatures*, scoped to their source TUs.

    NVCC salts its anonymous namespaces per compilation unit.  Raw `_Z...`
    names from production and isolated copies therefore differ even for the
    same kernel.  A fully demangled signature includes every template value,
    operation, input dtype and parameter; combined with source-TU identity
    and architecture it gives an equivalent comparison without false misses.
    Preserve both raw spellings in the CSV for auditing.
    """
    families = set(FAMILIES.values())
    counts = [Counter(), Counter()]
    spellings = [{}, {}]
    source_objects = [defaultdict(set), defaultdict(set)]
    raw_symbols = [defaultdict(set), defaultdict(set)]
    for index, source in enumerate((production_rows, selected_rows)):
        for row in source:
            family = row['category'] if index == 0 else row['mode']
            if family not in families:
                continue
            mangled = row.get('mangled_name', '')
            if not mangled or not mangled.startswith('_Z') or any(c.isspace() for c in mangled):
                raise RuntimeError(
                    'Exact production/selected comparison requires raw mangled names; '
                    'rerun the selected-inventory analyzer and production census with '
                    'the corrected readelf parser (no CUDA rebuild needed)')
            name = row.get('name', '')
            if not name or name.startswith('_Z') or name == mangled:
                raise RuntimeError(f'Kernel was not demangled: {mangled!r}')
            group = _kernel_source_group(row, selected=(index == 1))
            key = (_architecture(row['cubin']), family, group, name)
            counts[index][key] += 1
            spellings[index][key] = name
            source_objects[index][key].add(row['object'])
            raw_symbols[index][key].add(mangled)
    prod, selected = counts
    output = []
    states = Counter()
    for key in sorted(set(prod) | set(selected)):
        arch, family, source_group, name = key
        pc, sc = prod[key], selected[key]
        matched = min(pc, sc)
        surplus = max(pc - sc, 0)
        missing = max(sc - pc, 0)
        if matched:
            states[(family, 'matched_instances')] += matched
        if surplus:
            states[(family, 'production_only_instances' if not sc else 'repeated_instances')] += surplus
        if missing:
            states[(family, 'missing_selected_instances')] += missing
        state = ('production-only' if not sc else 'selected-missing' if not pc
                 else 'repeated-in-production' if pc > sc
                 else 'underrepresented-in-production' if pc < sc else 'matched')
        prod_spellings = sorted(raw_symbols[0][key])
        selected_spellings = sorted(raw_symbols[1][key])
        output.append(dict(architecture=arch, family=family, source_group=source_group,
                           status=state, production_instances=pc, selected_instances=sc,
                           surplus_instances=surplus, missing_instances=missing,
                           production_objects='; '.join(sorted(source_objects[0][key])),
                           selected_objects='; '.join(sorted(source_objects[1][key])),
                           mangled_name=(prod_spellings or selected_spellings)[0],
                           production_mangled_names='; '.join(prod_spellings),
                           selected_mangled_names='; '.join(selected_spellings),
                           name=name))
    return output, states


def load_selected_symbol_rows(build: Path) -> list[dict] | None:
    report = build / 'fp8-selected-inventory' / 'report' / 'device-functions.csv'
    if not report.exists():
        return None
    with report.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    required = {'object', 'cubin', 'mode', 'name', 'mangled_name'}
    if not rows or not required <= rows[0].keys():
        raise RuntimeError(f'Isolated selector symbol report is outdated: {report}. '
                           'Rebuild thor_fp8_selected_inventory_size to regenerate it.')
    if any(row['mode'] in set(FAMILIES.values()) and
           (not row['mangled_name'].startswith('_Z') or
            any(c.isspace() for c in row['mangled_name'])) for row in rows):
        raise RuntimeError(f'Isolated selector symbol report contains readelf prefixes: '
                           f'{report}. Rerun fp8_selected_inventory_size.py analyze '
                           '(CUDA recompilation is not needed).')
    return rows


def find_objects(build: Path) -> list[Path]:
    root = build / 'CMakeFiles' / 'ThorCubCudaObjects.dir'
    if not root.is_dir():
        raise ValueError(f'Production CUDA object directory not found: {root}')
    objects = sorted(root.rglob('*.cu.o'))
    if not objects:
        raise ValueError(f'No production CUDA objects found under {root}; build Thor first')
    return objects


def analyze(build: Path, report: Path) -> None:
    for program in ('cuobjdump', 'readelf', 'c++filt'):
        if shutil.which(program) is None:
            raise RuntimeError(f'Missing required tool: {program}')
    objects = find_objects(build)
    report.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | int]] = []
    object_sizes: list[dict[str, str | int]] = []
    for obj in objects:
        relative = obj.relative_to(build)
        with tempfile.TemporaryDirectory(prefix='thor-fp8-prod-census-') as temp:
            extraction_dir = Path(temp)
            run(['cuobjdump', '-xelf', 'all', str(obj.resolve())], cwd=extraction_dir)
            cubins = sorted(extraction_dir.rglob('*.cubin'))
            if not cubins:
                raise RuntimeError(f'Production CUDA object has no extractable device ELF: {obj}')
            object_sizes.append({'object': str(relative), 'object_bytes': obj.stat().st_size,
                                 'embedded_elf_bytes': sum(x.stat().st_size for x in cubins)})
            for cubin in cubins:
                symbols = parse_func_symbols(run(['readelf', '-W', '-s', str(cubin)]))
                names = demangle_func_symbols(symbols, f'{relative} / {cubin.name}')
                for (mangled, size), name in zip(symbols, names):
                    category = classify_function(name, mangled)
                    rows.append({'object': str(relative), 'cubin': cubin.name,
                                 'category': category, 'sass_bytes': size,
                                 'name': name, 'mangled_name': mangled})
    with (report / 'device-functions.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, ['object', 'cubin', 'category', 'sass_bytes', 'name', 'mangled_name'])
        writer.writeheader(); writer.writerows(rows)
    selected_rows = load_selected_symbol_rows(build)
    identity_rows = None
    identity_states = None
    if selected_rows is not None:
        identity_rows, identity_states = compare_selected_kernel_identities(rows, selected_rows)
        with (report / 'selector-identity-comparison.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=[
                'architecture', 'family', 'source_group', 'status',
                'production_instances', 'selected_instances',
                'surplus_instances', 'missing_instances',
                'production_objects', 'selected_objects', 'mangled_name',
                'production_mangled_names', 'selected_mangled_names', 'name'])
            writer.writeheader()
            writer.writerows(identity_rows)
    with (report / 'object-sizes.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, ['object', 'object_bytes', 'embedded_elf_bytes'])
        writer.writeheader(); writer.writerows(object_sizes)
    totals = Counter(row['category'] for row in rows)
    bytes_by_category = defaultdict(int)
    for row in rows:
        bytes_by_category[row['category']] += int(row['sass_bytes'])
    lines = [f'Production object root: {build.resolve()}',
             f'Objects: {len(objects)}; device ELFs: {len({(row["object"], row["cubin"]) for row in rows})}',
             'Counts are per compiled object/device ELF, not unique runtime kernels or linked library bytes.', '']
    for label in ('OneWindow', 'DeepRSmallK', 'AlignedK', 'RuntimeK', 'ARG',
                  'other-fp8-kernel', 'other-device-function'):
        lines.append(f'{label:22} {totals[label]:6} FUNC symbols, {bytes_by_category[label]:>13,} SASS text bytes')
    extra = [row for row in rows if row['category'] == 'other-fp8-kernel']
    retained = sum(totals[family] for family in set(FAMILIES.values()))
    lines += ['', f'Other-named FP8 kernels (outside the known family names): {len(extra)}',
              'Family-name classification does NOT establish selector reachability.',
              'These other-named kernels are NOT necessarily obsolete: R/KR and views may be legitimate.']
    selected_counts = selected_manifest_counts(build)
    if selected_counts is None:
        lines += ['', 'No generated selector manifest found at fp8-selected-inventory/expected.json.',
                  'Run thor_fp8_selected_inventory_size to compare family-name counts.']
    else:
        differences = family_count_discrepancies(totals, selected_counts)
        lines += ['', 'Production family-name FUNC counts versus isolated selector manifest:',
                  '  family                   production  selected  difference']
        for family, difference in differences.items():
            lines.append(f'  {family:22} {totals[family]:10}  {selected_counts[family]:8}  {difference:+11}')
        if any(differences.values()):
            lines += [
                'WARNING: compiled family-name symbol counts differ from selector inventory.',
                'This can reflect additional instantiations, repeated symbols, or differing build/SM flags.',
                'It does NOT prove which individual physical kernels are unused.',
                'Compare device-functions.csv with the selector manifest before deciding deletions.']
    if identity_states is None:
        lines += ['', 'No exact isolated selector symbol report available.',
                  'Run thor_fp8_selected_inventory_size, then rerun this census for identity-level comparison.']
    else:
        lines += ['', 'Exact selected-versus-production kernel identity comparison (per architecture):',
                  '  family                 matched  production-only  repeated  missing']
        for family in dict.fromkeys(FAMILIES.values()):
            lines.append(
                f'  {family:22} '
                f'{identity_states[family, "matched_instances"]:7}  '
                f'{identity_states[family, "production_only_instances"]:15}  '
                f'{identity_states[family, "repeated_instances"]:8}  '
                f'{identity_states[family, "missing_selected_instances"]:7}')
        lines += [
            'Production-only means absent from the compiled isolated selector inventory;',
            'it does not itself prove the kernel is unreachable from every other domain.',
            'Repeated means a selected typed kernel signature has more production object/ELF instances.',
            'Matching uses full demangled signatures and source TUs because NVCC salts anonymous namespaces.',
            'Both raw linker symbols are preserved in selector-identity-comparison.csv.',
            'Review selector-identity-comparison.csv for individual identities and source objects.']
    lines += ['See device-functions.csv for exact symbol names and source objects.',
              'The isolated selected-inventory census is independent: compare compatible build/SM/compiler.',
              'No claim is made about the final linked libThor.so contents.']
    if retained == 0:
        # A zero-kernel production report is not a successful deletion audit.
        # Persist the complete source spellings for attribution/recovery.
        suspect = [row for row in rows if re.search(
            'fp8', str(row['name']) + str(row['mangled_name']), re.IGNORECASE)]
        lines += ['', 'WARNING: no retained production FP8 FUNC symbols identified.',
                  f'Unattributed functions with FP8 in their names: {len(suspect)}',
                  'Review name and mangled_name in device-functions.csv.']
    (report / 'summary.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    if retained == 0:
        raise RuntimeError('Zero retained FP8 FUNCs in production census: '
                           'the attribution is unverified; inspect ' +
                           str(report / 'device-functions.csv'))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True, help='Existing CMake build directory')
    parser.add_argument('--out', type=Path, required=True, help='Report output directory')
    args = parser.parse_args()
    analyze(args.build, args.out)


if __name__ == '__main__':
    main()
