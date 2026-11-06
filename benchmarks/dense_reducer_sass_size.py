#!/usr/bin/env python3
"""SASS size of the dense reducer kernel families, by dtype, from the production objects (1232). Read-only.

1118's fp8_selected_inventory_size.py measured the selected FP8 RK first stages in an isolated archive. That
isolation was needed while the production translation units still compiled unselected FP8 experiments. Today every
retained kernel in them is reachable, so this tool measures the production objects themselves:

  CubReductionNarrowFlatRCooperative.cu.o   OneWindow, DeepRSmallK, RuntimeK PacketComplete, ARG twins, R / KR
  CubReductionKParallelShardedTiled.cu.o    AlignedK (FP8, FP32, FP16/BF16), FP32 ARG continuations
  CubReductionAwkwardRotatedSharded.cu.o    RuntimeK rotated (FP8, FP32, FP16/BF16), rotated ARG

1259 deleted the legacy KParallel / RCooperative kernels (and CubReductionCooperativeShardedTiled.cu), so every
kernel in these objects is retained; an instantiation the table does not know is an error.

For every device function in their cubins (readelf FUNC symbols, sizes in bytes of SASS text), it records the kernel,
its input dtype (the first template argument of the shared templates), and:

  owner    FP32 retained | FP8 retained | FP16/BF16 retained (1233) |
           FP8+FP32 shared (input-independent, e.g. the KR Split finish) | device helper
  family   RK OneWindow / DeepRSmallK / AlignedK / RuntimeK PacketComplete / RuntimeK Rotated / ARG continuation,
           KR Pack / Vector / Group / Block / Split / Split finish
  kind     VALUE | ARG | carried ARG

Every __global__ kernel declared in the three source files must be classified, or the tool fails (classify() is the
one table to extend). Device helpers (noinline functions) are counted per object, not attributed to a kernel.

Usage (from the build directory):
  python3 ../benchmarks/dense_reducer_sass_size.py --build . --source .. --out dense-sass
  python3 ../benchmarks/dense_reducer_sass_size.py --self-test
Writes <out>/device-functions.csv and <out>/summary.txt and prints the summary. Requires cuobjdump, readelf and
c++filt on PATH (CUDA toolkit and binutils).
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

SOURCES = {
    'CubReductionNarrowFlatRCooperative.cu.o': 'CubReductionNarrowFlatRCooperative.cu',
    'CubReductionKParallelShardedTiled.cu.o': 'CubReductionKParallelShardedTiled.cu',
    'CubReductionAwkwardRotatedSharded.cu.o': 'CubReductionAwkwardRotatedSharded.cu',
}
CUDA_GLOBAL_DECLARATION = re.compile(
    r'__global__\s+(?:__launch_bounds__\((?:[^()]|\([^()]*\))*\)\s+)?void\s+(\w+)\s*\(')

DTYPES = {'float': 'FP32', '__nv_fp8_e4m3': 'FP8', '__nv_fp8_e5m2': 'FP8', '__half': 'FP16', '__nv_bfloat16': 'BF16'}

# Kernels templated on the input type: the dtype comes from the first template argument.
SHARED_VALUE = {
    'fp8OneWindowRuntimeKCompleteKernel': 'RK OneWindow',
    'narrowFp8PackedFlatRCooperativeFirstStageKernel': 'RK DeepRSmallK',
    'narrowFp8PackedFlatRCooperativeTinyKFirstStageKernel': 'RK DeepRSmallK',
    'fp8GeneralRParallelKTileP16CompleteKernel': 'RK RuntimeK PacketComplete',
    'fp8KRValueGroupKernel': 'KR Group',
    'fp8KRValuePackKernel': 'KR Pack',
    'fp8KRValueVectorKernel': 'KR Vector',
    'fp8KRValueSplitKernel': 'KR Split',
    'fp8KRValueBlockKernel': 'KR Block',
    'fp32AlignedKKernel': 'RK AlignedK',  # 1243: FP32 and FP16 / BF16 input
}
FP8_ALIGNED_VALUE = ('kParallelFp8PacketKernel', 'kParallelFp8StagedHoistedCombineKernel',
                     'kParallelFp8CompleteUnifiedKernel')
# Kernels of one dtype only: (owner, family, kind).
FIXED = {
    'fp8KRValueFinishKernel': ('FP8+FP32 shared', 'KR Split finish', 'VALUE'),
    'fp8OneWindowArgCompleteKernel': ('FP8 retained', 'RK OneWindow', 'ARG'),
    'fp8DeepRArgFirstStageKernel': ('FP8 retained', 'RK DeepRSmallK', 'ARG'),
    'fp8DeepRArgTinyKFirstStageKernel': ('FP8 retained', 'RK DeepRSmallK', 'ARG'),
    'fp8RotatedArgKernel': ('FP8 retained', 'RK RuntimeK Rotated', 'ARG'),
    'fp8PacketCompleteArgKernel': ('FP8 retained', 'RK RuntimeK PacketComplete', 'ARG'),
    'fp8AlignedKArgKernel': ('FP8 retained', 'RK AlignedK', 'ARG'),
    'fp8ArgPackedContinuationKernel': ('FP8 retained', 'RK ARG continuation', 'ARG'),
    'fp8ArgPackedContinuationSplitKernel': ('FP8 retained', 'RK ARG continuation', 'ARG'),
    'fp8KRArgVectorKernel': ('FP8 retained', 'KR Vector', 'ARG'),
    'fp32AlignedKCarriedArgKernel': ('FP32 retained', 'RK AlignedK', 'carried ARG'),
    'fp32RotatedCarriedArgKernel': ('FP32 retained', 'RK RuntimeK Rotated', 'carried ARG'),
    'fp32OneWindowCarriedArgKernel': ('FP32 retained', 'RK OneWindow', 'carried ARG'),
    'fp32DeepRCarriedArgKernel': ('FP32 retained', 'RK DeepRSmallK', 'carried ARG'),
    'fp32DeepRCarriedArgTinyKKernel': ('FP32 retained', 'RK DeepRSmallK', 'carried ARG'),
    'fp32ArgPackedContinuationKernel': ('FP32 retained', 'RK ARG continuation', 'ARG'),
    'fp32ArgPackedContinuationSplitKernel': ('FP32 retained', 'RK ARG continuation', 'ARG'),
    'fp32ArgWideContinuationKernel': ('FP32 retained', 'RK ARG continuation', 'carried ARG'),
    'fp32ArgWideContinuationSplitKernel': ('FP32 retained', 'RK ARG continuation', 'carried ARG'),
    'fp32KRArgGroupKernel': ('FP32 retained', 'KR Group', 'ARG'),
    'fp32KRArgBlockKernel': ('FP32 retained', 'KR Block', 'ARG'),
    'fp32KRArgPackKernel': ('FP32 retained', 'KR Pack', 'ARG'),
    'fp32KRArgSplitKernel': ('FP32 retained', 'KR Split', 'ARG'),
    'fp32KRArgFinishKernel': ('FP32 retained', 'KR Split finish', 'ARG'),
    'fp32KRCarriedArgGroupKernel': ('FP32 retained', 'KR Group', 'carried ARG'),
    'fp32KRCarriedArgBlockKernel': ('FP32 retained', 'KR Block', 'carried ARG'),
    'fp32KRCarriedArgPackKernel': ('FP32 retained', 'KR Pack', 'carried ARG'),
    'fp32KRCarriedArgSplitKernel': ('FP32 retained', 'KR Split', 'carried ARG'),
    'fp32KRCarriedArgFinishKernel': ('FP32 retained', 'KR Split finish', 'carried ARG'),
}
# 1248: ARG kernels templated on the input type (the dtype comes from the first template argument), as SHARED_VALUE.
SHARED_DTYPE_ARG = {
    'fp32AlignedKArgKernel': 'RK AlignedK',
    'fp32RotatedArgKernel': 'RK RuntimeK Rotated',  # 1249
    'fp32PacketCompleteArgKernel': 'RK RuntimeK PacketComplete',  # 1249
    'fp32OneWindowArgKernel': 'RK OneWindow',  # 1250
    'fp32DeepRArgKernel': 'RK DeepRSmallK',  # 1250
    'fp32DeepRArgTinyKKernel': 'RK DeepRSmallK',  # 1250
}
# 1237: the R / KR ARG kernels shared by FP8 and FP16 / BF16 through an element-format trait (the first template
# argument: Fp8KRArgFormat or F16KRArgFormat).
SHARED_ARG = {
    'fp8KRArgBlockKernel': 'KR Block',
    'fp8KRArgGroupKernel': 'KR Group',
    'fp8KRArgPackKernel': 'KR Pack',
    'fp8KRArgSplitKernel': 'KR Split',
    'fp8KRArgFinishKernel': 'KR Split finish',
}
ARG_FORMATS = {'Fp8KRArgFormat': ('FP8 retained', 'FP8'), 'F16KRArgFormat': ('FP16/BF16 retained', 'FP16/BF16')}
ROTATED = 'awkwardAlignmentRotatedShardedFirstStageKernel'  # <InputT, Op, Transform, Rows, PacketBytes, Warps, Exact>
KNOWN = set(SHARED_VALUE) | set(SHARED_DTYPE_ARG) | set(SHARED_ARG) | set(FP8_ALIGNED_VALUE) | set(FIXED) | {ROTATED}


def template_arguments(name: str, kernel: str) -> list[str] | None:
    """Top-level template arguments of `kernel<...>` in a demangled name, or None when it has none."""
    match = re.search(r'(?:::|\b)' + re.escape(kernel) + r'\s*<', name)
    if not match:
        return None
    depth, start, args = 0, match.end(), []
    for i in range(match.end() - 1, len(name)):
        ch = name[i]
        if ch in '<(':
            depth += 1
        elif ch in '>)':
            depth -= 1
            if depth == 0:
                args.append(name[start:i].strip())
                return args
        elif ch == ',' and depth == 1:
            args.append(name[start:i].strip())
            start = i + 1
    return None


def kernel_of(name: str) -> str | None:
    """The known kernel this demangled function is, or None (a device helper or an unknown function)."""
    flat = name.replace('(anonymous namespace)', 'anonymous')
    # A function nested in a kernel (an out-of-line lambda, `kernel<...>(...)::{lambda...}`) is a helper.
    if re.search(r'\)\s*(?:const\s*)?::', flat):
        return None
    for kernel in KNOWN:
        if re.search(r'(?:::|\b)' + re.escape(kernel) + r'\s*[<(]', flat):
            return kernel
    return None


def classify(name: str) -> tuple[str, str, str, str, str] | None:
    """(kernel, owner, family, kind, dtype) for a kernel's demangled name; None for a device helper."""
    kernel = kernel_of(name)
    if kernel is None:
        return None
    args = template_arguments(name, kernel) or []
    dtype = DTYPES.get(args[0], args[0]) if args else ''
    if kernel in FIXED:
        owner, family, kind = FIXED[kernel]
        return kernel, owner, family, kind, {'FP32 retained': 'FP32', 'FP8 retained': 'FP8'}.get(owner, 'FP8+FP32')
    if kernel in SHARED_ARG:
        trait = args[0].split('::')[-1] if args else ''
        if trait not in ARG_FORMATS:
            raise ValueError(f'{kernel} instantiated for an unexpected format {trait!r}: {name}')
        owner, dtype = ARG_FORMATS[trait]
        return kernel, owner, SHARED_ARG[kernel], 'ARG', dtype
    if kernel in SHARED_DTYPE_ARG:
        owner = {'FP32': 'FP32 retained', 'FP16': 'FP16/BF16 retained', 'BF16': 'FP16/BF16 retained'}.get(dtype)
        if owner is None:
            raise ValueError(f'{kernel} instantiated for an unexpected input type {dtype!r}: {name}')
        return kernel, owner, SHARED_DTYPE_ARG[kernel], 'ARG', dtype
    if kernel in SHARED_VALUE:
        owner = {'FP32': 'FP32 retained', 'FP8': 'FP8 retained', 'FP16': 'FP16/BF16 retained',
                 'BF16': 'FP16/BF16 retained'}.get(dtype)
        if owner is None:
            raise ValueError(f'{kernel} instantiated for an unexpected input type {dtype!r}: {name}')
        return kernel, owner, SHARED_VALUE[kernel], 'VALUE', dtype
    if kernel in FP8_ALIGNED_VALUE:
        if dtype != 'FP8':
            raise ValueError(f'{kernel} instantiated for a non-FP8 input type {dtype!r}: {name}')
        return kernel, 'FP8 retained', 'RK AlignedK', 'VALUE', dtype
    # Rotated: FP8, the retained FP32 pass (4 cooperative warps, PrefixReserved) and (1245) the 8-warp p16 FP16 / BF16
    # RuntimeK pass. 1259 deleted the legacy 8-warp RCooperative rotated pass, so any other instantiation is an error.
    if len(args) < 6:
        raise ValueError(f'{kernel} with unexpected template arguments: {name}')
    if dtype == 'FP8':
        return kernel, 'FP8 retained', 'RK RuntimeK Rotated', 'VALUE', dtype
    if dtype == 'FP32' and args[5] == '4':
        return kernel, 'FP32 retained', 'RK RuntimeK Rotated', 'VALUE', dtype
    if dtype in ('FP16', 'BF16') and args[4] == '16' and args[5] == '8':
        return kernel, 'FP16/BF16 retained', 'RK RuntimeK Rotated', 'VALUE', dtype
    raise ValueError(f'{kernel} instantiated outside the retained rotated passes: {name}')


def run(cmd, cwd=None, text_input=None):
    result = subprocess.run(cmd, cwd=cwd, input=text_input, capture_output=True, text=True, errors='replace')
    if result.returncode != 0:
        sys.exit(f"{' '.join(cmd)} failed: {result.stderr.strip()[-2000:]}")
    return result.stdout


def func_symbols(readelf_output: str) -> list[tuple[str, int]]:
    """(mangled name, size) of every defined FUNC symbol with a size (readelf -W -s)."""
    seen, out = set(), []
    for line in readelf_output.splitlines():
        parts = line.strip().split(None, 7)
        if len(parts) != 8 or not parts[0].endswith(':') or parts[3] != 'FUNC' or parts[6] in ('UND', 'ABS'):
            continue
        try:
            size = int(parts[2], 0)  # readelf prints sizes above 99999 in hex
        except ValueError:
            continue
        name = parts[7].split()[-1].split('@@', 1)[0]
        if size and (name, size) not in seen:
            seen.add((name, size))
            out.append((name, size))
    return out


def find_object(build: Path, name: str) -> Path:
    found = sorted(build.rglob(name))
    preferred = [p for p in found if 'ThorCubCudaObjects' in str(p)]
    if not (preferred or found):
        sys.exit(f'no {name} under {build} (build Thor first)')
    return (preferred or found)[0].resolve()


def analyze(build: Path, source: Path, out: Path):
    for program in ('cuobjdump', 'readelf', 'c++filt'):
        if shutil.which(program) is None:
            sys.exit(f'{program} not found on PATH')
    cub = source / 'Utilities/TensorOperations/Cub'
    declared = {}
    for obj_name, src in SOURCES.items():
        declared[obj_name] = set(CUDA_GLOBAL_DECLARATION.findall((cub / src).read_text()))
    unknown = sorted(set().union(*declared.values()) - KNOWN)
    if unknown:
        sys.exit(f'__global__ kernels without a classification (extend classify()): {unknown}')
    out.mkdir(parents=True, exist_ok=True)
    rows, problems = [], []
    for obj_name in SOURCES:
        obj = find_object(build, obj_name)
        with tempfile.TemporaryDirectory() as tmp:
            run(['cuobjdump', '-xelf', 'all', str(obj)], cwd=tmp)
            cubins = sorted(Path(tmp).rglob('*.cubin'))
            if not cubins:
                sys.exit(f'no device ELF in {obj}')
            for cubin in cubins:
                arch = re.search(r'sm_?(\d+)', cubin.name)
                arch = f'sm_{arch.group(1)}' if arch else cubin.name
                symbols = func_symbols(run(['readelf', '-W', '-s', str(cubin)]))
                names = run(['c++filt', '-n'], text_input='\n'.join(s for s, _ in symbols) + '\n').splitlines()
                if len(names) != len(symbols):
                    sys.exit(f'demangle mismatch in {cubin}')
                for (mangled, size), name in zip(symbols, names):
                    try:
                        c = classify(name)
                    except ValueError as error:
                        problems.append(str(error))
                        continue
                    if c is None:
                        kernel_like = bool(re.search(r'[Kk]ernel\s*<', name)) and 'ThorImplementation::' in name
                        if kernel_like:
                            problems.append(f'unclassified Thor kernel in {obj_name}: {name}')
                        c = ('', 'device helper', obj_name.removesuffix('.cu.o'), '', '')
                    kernel, owner, family, kind, dtype = c
                    rows.append({'object': obj_name, 'arch': arch, 'owner': owner, 'family': family, 'kind': kind,
                                 'dtype': dtype, 'bytes': size, 'kernel': kernel, 'name': name})
    with (out / 'device-functions.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ['object'])
        writer.writeheader()
        writer.writerows(rows)
    lines = summarize(rows)
    if problems:
        lines += ['', f'ERROR: {len(problems)} functions could not be classified:'] + [f'  {p}' for p in problems[:40]]
    (out / 'summary.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    if problems:
        sys.exit(1)


OWNERS = ('FP32 retained', 'FP8 retained', 'FP16/BF16 retained', 'FP8+FP32 shared', 'device helper')


def summarize(rows: list[dict]) -> list[str]:
    lines = []
    for arch in sorted({r['arch'] for r in rows}):
        here = [r for r in rows if r['arch'] == arch]
        lines.append(f'# {arch}: SASS text bytes (readelf FUNC sizes) of the dense reducer objects')
        totals = defaultdict(lambda: [0, 0])
        for r in here:
            key = (r['owner'], r['family'], r['kind'])
            totals[key][0] += 1
            totals[key][1] += r['bytes']
        owner_totals = defaultdict(lambda: [0, 0])
        for (owner, _, _), (n, b) in totals.items():
            owner_totals[owner][0] += n
            owner_totals[owner][1] += b
        lines.append(f'  {"owner":17s} {"family":30s} {"kind":12s} {"kernels":>7s} {"bytes":>12s}')
        for owner in OWNERS:
            keys = sorted(k for k in totals if k[0] == owner)
            for key in keys:
                n, b = totals[key]
                count = '' if owner == 'device helper' else str(n)
                lines.append(f'  {key[0]:17s} {key[1]:30s} {key[2]:12s} {count:>7s} {b:>12,}')
            if keys:
                n, b = owner_totals[owner]
                count = '' if owner == 'device helper' else str(n)
                lines.append(f'  {owner + " total":60s} {count:>7s} {b:>12,}')
                lines.append('')
        grand = sum(b for _, b in owner_totals.values())
        lines.append(f'  {"all functions in these objects":60s} {"":>7s} {grand:>12,}')
        for owner in ('FP32 retained', 'FP8 retained', 'FP16/BF16 retained'):
            if owner not in owner_totals:
                continue
            by_family = defaultdict(int)
            for (o, family, _), (_, b) in totals.items():
                if o == owner:
                    by_family[family.split(' ')[0] + ' ' + family.split(' ')[1]] += b
            parts = ', '.join(f'{f} {b / 1e6:.2f} MB' for f, b in sorted(by_family.items()))
            lines.append(f'  {owner}: {owner_totals[owner][1] / 1e6:.2f} MB ({parts})')
        lines.append('')
    lines.append('FP8+FP32 shared: input-independent kernels both dtypes run (counted once). Device helpers are')
    lines.append('noinline functions, not attributed to a kernel. The legacy families were deleted in 1259.')
    lines.append('1118\'s isolated FP8 RK first-stage inventory measured 8.80 MB.')
    return lines


def self_test():
    ns = 'ThorImplementation::CubReductionInternal::(anonymous namespace)::'
    cases = [
        (f'void {ns}fp8OneWindowRuntimeKCompleteKernel<float, 1, 16, 4, false, 0>(float const*, void*)',
         ('FP32 retained', 'RK OneWindow', 'VALUE', 'FP32')),
        (f'void {ns}fp8OneWindowRuntimeKCompleteKernel<__nv_fp8_e4m3, 2, 16, 1, false, 2>(__nv_fp8_e4m3 const*)',
         ('FP8 retained', 'RK OneWindow', 'VALUE', 'FP8')),
        (f'void {ns}narrowFp8PackedFlatRCooperativeFirstStageKernel<float, cuda::std::__4::plus<float>, '
         f'{ns}IdentityFp32, 16, 1, 6>(float const*, float*)', ('FP32 retained', 'RK DeepRSmallK', 'VALUE', 'FP32')),
        (f'void {ns}awkwardAlignmentRotatedShardedFirstStageKernel<float, {ns}PropagatingMinimumFp32, '
         f'{ns}IdentityFp32, 128, 16, 4, false>(float const*)',
         ('FP32 retained', 'RK RuntimeK Rotated', 'VALUE', 'FP32')),
        (f'void {ns}awkwardAlignmentRotatedShardedFirstStageKernel<__half, cuda::std::__4::plus<float>, '
         f'{ns}IdentityFp32, 64, 16, 8, true>(__half const*)',
         ('FP16/BF16 retained', 'RK RuntimeK Rotated', 'VALUE', 'FP16')),  # 1245
        (f'void {ns}fp32AlignedKKernel<float, 1, 128, ({ns[:-len("(anonymous namespace)::")]}TiledRKStageTopology)0>'
         f'(float const*, void*)', ('FP32 retained', 'RK AlignedK', 'VALUE', 'FP32')),
        (f'void {ns}fp8GeneralRParallelKTileP16CompleteKernel<__half, 2, 3>(__half const*, void*)',
         ('FP16/BF16 retained', 'RK RuntimeK PacketComplete', 'VALUE', 'FP16')),  # 1244
        (f'void {ns}fp32AlignedKKernel<__nv_bfloat16, 8, 64, '
         f'({ns[:-len("(anonymous namespace)::")]}TiledRKStageTopology)1>(__nv_bfloat16 const*, void*)',
         ('FP16/BF16 retained', 'RK AlignedK', 'VALUE', 'BF16')),
        (f'void {ns}fp32AlignedKArgKernel<float, 2, 64, true>(float const*)',
         ('FP32 retained', 'RK AlignedK', 'ARG', 'FP32')),
        (f'void {ns}fp32AlignedKArgKernel<__half, 8, 32, false>(__half const*)',
         ('FP16/BF16 retained', 'RK AlignedK', 'ARG', 'FP16')),  # 1248
        (f'void {ns}fp32RotatedArgKernel<__nv_bfloat16, true>(__nv_bfloat16 const*)',
         ('FP16/BF16 retained', 'RK RuntimeK Rotated', 'ARG', 'BF16')),  # 1249
        (f'void {ns}fp32PacketCompleteArgKernel<__half>(__half const*)',
         ('FP16/BF16 retained', 'RK RuntimeK PacketComplete', 'ARG', 'FP16')),  # 1249
        (f'void {ns}fp32OneWindowArgKernel<__half, 2, 8, true>(__half const*)',
         ('FP16/BF16 retained', 'RK OneWindow', 'ARG', 'FP16')),  # 1250
        (f'void {ns}fp32DeepRArgTinyKKernel<float, 4>(float const*)',
         ('FP32 retained', 'RK DeepRSmallK', 'ARG', 'FP32')),
        (f'void {ns}fp32RotatedArgKernel<float, false>(float const*)',
         ('FP32 retained', 'RK RuntimeK Rotated', 'ARG', 'FP32')),
        (f'void {ns}fp32AlignedKCarriedArgKernel<2, 64, true, unsigned long>(float const*)',
         ('FP32 retained', 'RK AlignedK', 'carried ARG', 'FP32')),
        (f'void {ns}fp8KRValueFinishKernel<2>(float const*, void*)',
         ('FP8+FP32 shared', 'KR Split finish', 'VALUE', 'FP8+FP32')),
        (f'void {ns}fp8KRValueGroupKernel<float, 0, 1>(float const*)', ('FP32 retained', 'KR Group', 'VALUE', 'FP32')),
        (f'void {ns}fp32KRArgSplitKernel(float const*, unsigned long*)', ('FP32 retained', 'KR Split', 'ARG', 'FP32')),
        (f'void {ns}fp8KRArgBlockKernel<{ns}Fp8KRArgFormat>(unsigned int const*)',
         ('FP8 retained', 'KR Block', 'ARG', 'FP8')),
        (f'void {ns}fp8KRArgFinishKernel<{ns}F16KRArgFormat>(unsigned long const*)',
         ('FP16/BF16 retained', 'KR Split finish', 'ARG', 'FP16/BF16')),
        (f'void {ns}kParallelFp8CompleteUnifiedKernel<__nv_fp8_e5m2, 8ul, 64, '
         f'({ns[:-len("(anonymous namespace)::")]}Fp8CompleteFinalizePolicy)1>(__nv_fp8_e5m2 const*)',
         ('FP8 retained', 'RK AlignedK', 'VALUE', 'FP8')),
    ]
    for name, expected in cases:
        got = classify(name)
        assert got is not None and got[1:] == expected, (name, got, expected)
    helper = f'{ns}finalizeAdditiveFp32OutOfLine(float, float, bool)'
    assert classify(helper) is None
    nested = f'{ns}fp32AlignedKKernel<float, 1, 32, (X)0>(float const*)::{{lambda()#1}}::operator()() const'
    assert classify(nested) is None
    got = classify(f'void {ns}fp8KRValueGroupKernel<__half, 0, 0>(unsigned int const*)')  # 1233
    assert got is not None and got[1:] == ('FP16/BF16 retained', 'KR Group', 'VALUE', 'FP16'), got
    try:
        classify(f'void {ns}fp8KRValueGroupKernel<double, 0, 0>(unsigned int const*)')
        raise AssertionError('an unexpected instantiation of a retained kernel must fail')
    except ValueError:
        pass
    try:  # 1259: the legacy 8-warp FP32 rotated pass is deleted
        classify(f'void {ns}awkwardAlignmentRotatedShardedFirstStageKernel<float, cuda::std::__4::plus<float>, '
                 f'{ns}IdentityFp32, 64, 16, 8, true>(float const*)')
        raise AssertionError('a legacy rotated instantiation must fail')
    except ValueError:
        pass
    rows = [{'arch': 'sm_120', 'owner': c[1][0], 'family': c[1][1], 'kind': c[1][2], 'bytes': 100} for c in cases]
    rows.append({'arch': 'sm_120', 'owner': 'device helper', 'family': 'CubReductionNarrowFlatRCooperative',
                 'kind': '', 'bytes': 50})
    assert any('FP32 retained total' in line for line in summarize(rows))
    print(f'self-test: {len(cases) + 5} classifications and the summary OK')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--build', type=Path)
    parser.add_argument('--source', type=Path, help='the Thor source root (.. from the build directory)')
    parser.add_argument('--out', type=Path, default=Path('dense-sass'))
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.build is None or args.source is None:
        parser.error('--build and --source are required (or --self-test)')
    analyze(args.build, args.source, args.out)


if __name__ == '__main__':
    main()
