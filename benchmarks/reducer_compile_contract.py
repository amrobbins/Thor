#!/usr/bin/env python3
"""The reducer's compile contract, checked from a build's compile_commands.json. Read-only.

The reducer's and the scan kernel's CUDA translation units (every .cu directly in Utilities/TensorOperations/Scan/,
and every .cu anywhere under Utilities/TensorOperations/Reduction/) compile as
self-contained objects: no relocatable device code and no device LTO (CMakeLists.txt,
THOR_SELF_CONTAINED_CUDA_SOURCES). Compiled like the rest of Thor they regress, FP8
E5M2 especially. This tool checks, for every such file:

  - it has a compile command, and that command builds an object of the self-contained target (ThorSelfContainedCuda*Objects);
  - the command asks for no relocatable device code (-rdc=true, --relocatable-device-code=true, -dc, --device-c) and no
    link-time optimization (-dlto, --dlink-time-opt, -flto, -gen-opt-lto, an lto_NN target code);
  - with --compare=OLD.json (an earlier build's compile_commands.json): its options equal the old command's for the
    same file, or for the file it was moved from (RENAMED_SOURCES) or split out of (SPLIT_SOURCES), once the file,
    object and dependency paths are set aside; files with no predecessor (NEW_SOURCES) are not compared.

Usage (from the build directory; copy compile_commands.json aside before reconfiguring to keep the old one):
  python3 ../benchmarks/reducer_compile_contract.py --build . --source ..
  python3 ../benchmarks/reducer_compile_contract.py --build . --source .. --compare compile_commands-old.json
  python3 ../benchmarks/reducer_compile_contract.py --self-test
Prints one line per file and exits 1 if any check fails.
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from pathlib import Path

CUB = 'Utilities/TensorOperations/Cub'  # the old home of the reducer's sources (RENAMED_SOURCES)
SCAN = 'Utilities/TensorOperations/Scan'
REDUCTION = 'Utilities/TensorOperations/Reduction'
# Reducer CUDA sources moved from an old path: new path -> old path, both from the source root.
RENAMED_SOURCES = {
    f'{REDUCTION}/Kernels/StridedR.cu': f'{CUB}/CubStridedR.cu',
    f'{REDUCTION}/Kernels/RKRuntimeK.cu': f'{CUB}/CubReductionAwkwardRotatedSharded.cu',
    f'{REDUCTION}/Kernels/RKAlignedK.cu': f'{CUB}/CubReductionKParallelShardedTiled.cu',
    f'{REDUCTION}/Kernels/RKDeepRSmallK.cu': f'{CUB}/CubReductionNarrowFlatRCooperative.cu',
    **{f'{REDUCTION}/Ragged/{n}.cu': f'{CUB}/Cub{n}.cu' for n in (  # without the Cub prefix
        'RaggedFinish', 'RaggedFinishArg', 'RaggedFinishArgFp32', 'RaggedFinishArgPairWide', 'RaggedFinishArgWide',
        'RaggedKR', 'RaggedKRArg', 'RaggedKRArgF16', 'RaggedKRArgFp32', 'RaggedKRArgFp8', 'RaggedKRF16',
        'RaggedKRFp32', 'RaggedKRFp8', 'RaggedRK', 'RaggedRKArg', 'RaggedRKArgF16', 'RaggedRKArgFp32',
        'RaggedRKArgFp8', 'RaggedRKF16', 'RaggedRKF16Narrow', 'RaggedRKFp32', 'RaggedRKFp32Narrow', 'RaggedRKFp8',
        'RaggedRKFp8Narrow')},
}
# Files split out of a file that remains (new path -> the file they came from).
SPLIT_SOURCES = {
    f'{REDUCTION}/Kernels/KR.cu': f'{CUB}/CubReductionNarrowFlatRCooperative.cu',
    f'{REDUCTION}/Kernels/RKOneWindow.cu': f'{CUB}/CubReductionNarrowFlatRCooperative.cu',
}
# Files with no predecessor (from the source root): --compare has no old command for them.
NEW_SOURCES = {f'{SCAN}/ScanMinMaxBackward.cu', f'{SCAN}/SegmentedScanArg.cu', f'{SCAN}/SegmentedScanF16.cu',
               f'{SCAN}/SegmentedScanFp32.cu', f'{SCAN}/SegmentedScanFp8.cu'}
# CMake asks nvcc for device LTO in the target code, --generate-code=arch=compute_120,code=[compute_120,lto_120],
# not with -dlto: an lto_NN code is LTO.
FORBIDDEN = re.compile(r'^(-rdc=true|--relocatable-device-code=true|-dc|--device-c|-dlto|--dlink-time-opt|-flto.*|'
                       r'-gen-opt-lto|.*[=,]-flto.*|.*-dlto.*|.*\blto_\d+.*)$')
TARGET = re.compile(r'ThorSelfContainedCuda\w*Objects')


def reducer_sources(source: Path) -> list[str]:
    files = sorted((source / SCAN).glob('*.cu')) + sorted((source / REDUCTION).rglob('*.cu'))
    return [str(p.relative_to(source)) for p in files]


def arguments(entry: dict) -> list[str]:
    return list(entry['arguments']) if 'arguments' in entry else shlex.split(entry['command'])


def options(entry: dict) -> list[str]:
    """The command's options without the compiler, the source, the object and dependency-file paths."""
    args = arguments(entry)[1:]
    out, skip = [], False
    source = Path(entry['file']).name
    for i, a in enumerate(args):
        if skip:
            skip = False
            continue
        if a in ('-o', '-MF', '-MT', '-MQ'):
            skip = True
            continue
        if a.endswith(source) or a.endswith(source + '.o') or a.startswith(('-MF', '-MT', '-MQ')):
            continue
        out.append(a)
    return out


def index(commands: list[dict], source: Path) -> dict[str, dict]:
    out = {}
    for entry in commands:
        f = Path(entry['file'])
        if not f.is_absolute():
            f = Path(entry.get('directory', '.')) / f
        try:
            out[str(f.resolve().relative_to(source.resolve()))] = entry
        except ValueError:
            continue
    return out


def check(build_commands: list[dict], source: Path, files: list[str], old_commands: list[dict] | None = None,
          old_source: Path | None = None) -> tuple[list[str], int]:
    mine = index(build_commands, source)
    theirs = index(old_commands, old_source or source) if old_commands is not None else None
    lines, failures = [], 0
    for rel in files:
        entry = mine.get(rel)
        problems = []
        if entry is None:
            problems.append('no compile command')
        else:
            args = arguments(entry)
            if not any(TARGET.search(a) for a in args + [entry.get('output', '')]):
                problems.append('not built by the self-contained target (ThorSelfContainedCuda*Objects)')
            bad = [a for a in args if FORBIDDEN.match(a)]
            if bad:
                problems.append('relocatable device code or LTO: ' + ' '.join(bad))
            if theirs is not None and rel not in NEW_SOURCES:
                old = RENAMED_SOURCES.get(rel) or SPLIT_SOURCES.get(rel, rel)
                if old not in theirs:
                    problems.append(f'no command for {old} in the old build')
                elif options(theirs[old]) != options(entry):
                    a, b = options(theirs[old]), options(entry)
                    only_old = ' '.join(x for x in a if x not in b)
                    only_new = ' '.join(x for x in b if x not in a)
                    problems.append(f'options differ from the old build: only old {only_old} | only new {only_new}')
        failures += bool(problems)
        note = 'ok' if not problems else '; '.join(problems)
        moved = f' (was {RENAMED_SOURCES[rel]})' if rel in RENAMED_SOURCES else ''
        moved = f' (split from {SPLIT_SOURCES[rel]})' if rel in SPLIT_SOURCES else moved
        moved = ' (new: not compared)' if rel in NEW_SOURCES and theirs is not None else moved
        lines.append(f'  {rel}{moved}: {note}')
    lines.insert(0, f'# reducer compile contract: {len(files)} CUDA sources, {failures} failing')
    return lines, failures


def self_test():
    src = Path('/src')
    cub = '/src/' + CUB
    red = '/src/' + REDUCTION

    def cmd(path, target='ThorSelfContainedCudaObjects', extra=''):
        name = Path(path).name
        return {'directory': '/b', 'file': path,
                'command': f'/usr/local/cuda/bin/nvcc -DX=1 -I/src -O3 --use_fast_math {extra} -x cu -c {path} '
                           f'-o CMakeFiles/{target}.dir/{name}.o -MD -MT CMakeFiles/{target}.dir/{name}.o '
                           f'-MF CMakeFiles/{target}.dir/{name}.o.d'}
    old = [cmd(f'{cub}/A.cu'), cmd(f'{cub}/CubStridedR.cu')]
    new = [cmd(f'{cub}/A.cu'), cmd(f'{red}/Kernels/StridedR.cu')]
    files = [f'{CUB}/A.cu', f'{REDUCTION}/Kernels/StridedR.cu']
    lines, failures = check(new, src, files, old)
    assert failures == 0, lines
    assert '(was Utilities/TensorOperations/Cub/CubStridedR.cu): ok' in '\n'.join(lines), lines
    bad = [cmd(f'{cub}/A.cu', 'Thor', '-rdc=true'), cmd(f'{red}/Kernels/StridedR.cu', extra='-dlto')]
    lines, failures = check(bad, src, files, old)
    text = '\n'.join(lines)
    assert failures == 2 and 'not built by the self-contained target' in text and '-rdc=true' in text, text
    assert 'relocatable device code or LTO: -dlto' in text and 'only new -dlto' in text, text
    lines, failures = check([cmd(f'{cub}/A.cu')], src, files)
    assert failures == 1 and 'no compile command' in '\n'.join(lines), lines
    lto = '"--generate-code=arch=compute_120,code=[compute_120,lto_120]" -rdc=true'
    lines, failures = check([cmd(f'{cub}/A.cu', extra=lto)], src, files[:1])
    assert failures == 1 and 'lto_120' in '\n'.join(lines) and '-rdc=true' in '\n'.join(lines), lines
    lines, failures = check([cmd(f'{cub}/A.cu', extra='-G')], src, files[:1], old)
    assert failures == 1 and 'only new -G' in '\n'.join(lines), lines
    # A file split out of one that remains compares with that file.
    narrow = f'{cub}/CubReductionNarrowFlatRCooperative.cu'
    split_files = [f'{CUB}/CubReductionNarrowFlatRCooperative.cu', f'{REDUCTION}/Kernels/KR.cu']
    lines, failures = check([cmd(narrow), cmd(f'{red}/Kernels/KR.cu')], src, split_files, [cmd(narrow)])
    assert failures == 0 and '(split from Utilities/TensorOperations/Cub/CubReductionNarrowFlatRCooperative.cu): ok' \
        in '\n'.join(lines), lines
    lines, failures = check([cmd(narrow), cmd(f'{red}/Kernels/KR.cu', extra='-G')], src,
                            [f'{REDUCTION}/Kernels/KR.cu'], [cmd(narrow)])
    assert failures == 1 and 'only new -G' in '\n'.join(lines), lines
    print('self-test: the contract checks and the old-build comparison OK')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--build', type=Path)
    parser.add_argument('--source', type=Path, help='the Thor source root (.. from the build directory)')
    parser.add_argument('--compare', type=Path, help="an earlier build's compile_commands.json")
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.build is None or args.source is None:
        parser.error('--build and --source are required (or --self-test)')
    commands = json.loads((args.build / 'compile_commands.json').read_text())
    old = json.loads(args.compare.read_text()) if args.compare else None
    lines, failures = check(commands, args.source, reducer_sources(args.source), old)
    print('\n'.join(lines))
    if failures:
        sys.exit(1)


if __name__ == '__main__':
    main()
